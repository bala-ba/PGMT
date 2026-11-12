"""P2.3/4: real native PPO updates from saved, augmented map/mask pairs.

Synthetic sensor hits exercise the production observation pipeline, without
Isaac. Hooks observe the native computation; no replacement PPO loss, storage,
forward pass, or optimizer is used.
"""
import math

import pytest
import torch
from tensordict import TensorDict

from motion_tracking_rl.algorithms import PPO
from motion_tracking_rl.networks import PGMTActorCritic
from pmt_tasks import pgmt_perception as perception


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))]
GROUPS = {"policy": ["proprio"], "policy_history": ["proprio_history"],
          "command_window": ["command_window"], "critic": ["critic"],
          "terrain_map": ["terrain_map"], "terrain_valid": ["terrain_valid"]}
N, T, EPOCHS, MINIBATCHES = 6, 4, 2, 3


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def exact(actual, expected):
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


class SyntheticTerrainSource:
    def __init__(self, device, kind):
        self.device, self.kind, self.calls = device, kind, 0

    def get_observations(self):
        device = self.device
        ids = torch.arange(N, device=device) + N * self.calls
        self.calls += 1
        y, x = torch.meshgrid(torch.linspace(-1., 1., 21, device=device),
                              torch.linspace(-1., 1., 21, device=device), indexing="ij")
        height = .9 + .6 * x + .3 * y + .15 * x * y + ids[:, None, None] * .003
        hits = torch.zeros(N, 441, 3, device=device)
        hits[..., 2] = .8 - height.flatten(1)
        map_valid = torch.ones(N, 21, 21, dtype=torch.bool, device=device)
        if self.kind == "mixed":
            map_valid[1::3] &= x > -.05  # A boundary crossing the central crops.
            map_valid[2::3] = False
        elif self.kind == "invalid":
            map_valid[:] = False
        pair = perception.terrain_observation_from_hits(
            hits, torch.full((N,), .8, device=device), map_valid.flatten(1),
            perception.terrain_observation_contract(augmentation_enabled=self.kind != "valid"))
        fields = perception.split_terrain_observation({"pgmt_terrain": perception.pack_terrain_observation(pair)})
        fields.update(proprio=torch.randn(N, 96, device=device),
                      proprio_history=torch.randn(N, 10, 96, device=device),
                      command_window=torch.randn(N, 8, 61, device=device),
                      critic=torch.randn(N, 251, device=device), sample_id=ids.float()[:, None])
        return TensorDict(fields, [N], device=device)


def make_algorithm(device, source, *, objective="combined", mode="enabled", per_minibatch=False):
    obs = source.get_observations()
    policy = PGMTActorCritic(
        obs, GROUPS, 29, actor_hidden_dims=[32, 16], critic_hidden_dims=[32, 16],
        actor_obs_normalization=True, critic_obs_normalization=True, init_noise_std=.7,
        terrain_glimpse_cfg={}, terrain_mode=mode).to(device)
    policy.update_normalization(obs)
    algorithm = PPO(
        policy, device=device, num_learning_epochs=EPOCHS, num_mini_batches=MINIBATCHES,
        learning_rate=3e-4, schedule="fixed", normalize_advantage_per_mini_batch=per_minibatch,
        value_loss_coef=0. if objective == "actor" else 1.,
        entropy_coef=.01 if objective == "combined" else 0.,
        value_only_warmup_iters=100 if objective == "critic" else 0,
        debug_raise_on_nonfinite=True)
    algorithm.init_storage("rl", N, T, obs, [29])
    return algorithm, obs


def collect_rollout(algorithm, source, obs):
    """Use the runner's inference context and native transition lifecycle."""
    saved, expected_rewards, rollout_tokens = [], [], []

    def rollout_forward(module, inputs, output):
        assert not output.terrain_tokens.requires_grad
        assert output.terrain_tokens.grad_fn is None
        rollout_tokens.append(output.token_valid.clone())

    hook = algorithm.policy.terrain_glimpse.register_forward_hook(rollout_forward)
    with torch.inference_mode():
        for step in range(T):
            saved.append(obs.clone())
            algorithm.act(obs)
            groups = torch.randn(N, 3, device=source.device) + torch.tensor(
                [.4, .2, .1], device=source.device)
            dones = torch.zeros(N, dtype=torch.bool, device=source.device)
            timeouts = torch.zeros_like(dones)
            if step == 1:
                dones[0] = True  # True terminal; no bootstrap.
            if step == 2:
                dones[1] = timeouts[1] = True
            expected_rewards.append(groups + algorithm.gamma * algorithm.transition.values * timeouts[:, None])
            following = source.get_observations()
            algorithm.process_env_step(following, groups.sum(-1), dones,
                                       {"reward_groups": groups, "time_outs": timeouts})
            # Simulate later reuse of an environment's old buffers AFTER copy.
            # Native act() retains obs by reference until process_env_step().
            obs["terrain_map"].fill_(-123.)
            obs["terrain_valid"].zero_()
            obs = following
        last_values = algorithm.policy.evaluate(obs).clone()
        algorithm.compute_returns(obs)
    hook.remove()
    expected_obs = torch.stack(saved)
    storage = algorithm.storage
    assert storage.step == T
    assert set(storage.observations.keys()) == set(expected_obs.keys())
    assert storage.observations["terrain_map"].shape == (T, N, 21, 21)
    assert storage.observations["terrain_valid"].shape == (T, N, 21, 21)
    for key in expected_obs.keys():
        exact(storage.observations[key], expected_obs[key])
        assert not storage.observations[key].requires_grad
    exact(storage.rewards, torch.stack(expected_rewards))

    # Independent scalar GAE recursion for all three heads, including both ends.
    expected_returns = torch.empty_like(storage.returns)
    for env in range(N):
        for head in range(3):
            carry = 0.
            for step in reversed(range(T)):
                live = 1. - float(storage.dones[step, env, 0])
                next_value = last_values[env, head] if step == T - 1 else storage.values[step + 1, env, head]
                delta = storage.rewards[step, env, head] + algorithm.gamma * next_value * live - storage.values[step, env, head]
                carry = delta + algorithm.gamma * algorithm.lam * live * carry
                expected_returns[step, env, head] = carry + storage.values[step, env, head]
    torch.testing.assert_close(storage.returns, expected_returns)
    expected_adv = (expected_returns - storage.values).sum(-1, keepdim=True)
    if not algorithm.normalize_advantage_per_mini_batch:
        expected_adv = (expected_adv - expected_adv.mean()) / (expected_adv.std() + 1e-8)
    torch.testing.assert_close(storage.advantages, expected_adv)
    assert storage.values.shape == storage.returns.shape == (T, N, 3)
    assert storage.advantages.shape == (T, N, 1)
    assert len(rollout_tokens) == (T + 2 if algorithm.policy.terrain_mode == "enabled" else 0)
    if source.kind == "valid" and rollout_tokens:
        assert all(value.all() for value in rollout_tokens)
    if source.kind == "mixed" and rollout_tokens:
        assert any(value.any() for value in rollout_tokens)
        assert all(not value[2::3].any() for value in rollout_tokens)
    return expected_obs.flatten(0, 1)


def module_grad_l1(module):
    result = 0.
    for parameter in module.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
            result += float(parameter.grad.abs().sum())
    return result


def checked_update(algorithm, source, expected, monkeypatch):
    """Trace actual native minibatches, autograd, and Adam without changing them."""
    policy, storage = algorithm.policy, algorithm.storage
    modules = {"selector": policy.terrain_glimpse.selector,
               "encoder": policy.terrain_glimpse.patch_encoder, "ifm": policy.intent_attention,
               "actor": policy.actor, "critic": policy.critic}
    before = {name: [p.detach().clone() for p in module.parameters()] for name, module in modules.items()}
    buffers = {name: value.clone() for name, value in policy.named_buffers()}
    generator = storage.mini_batch_generator
    expected_ids = expected["sample_id"][:, 0].long()
    seen, current, graphs, updates = [], {}, [], []
    inputs_seen, prepared_seen, intents_seen = [], [], []

    def check_batches(*args, **kwargs):
        for batch in generator(*args, **kwargs):
            observations = batch[0]
            ids = observations["sample_id"][:, 0].long()
            # IDs are a fixture-only unused obs field, independent of shuffle.
            index = (ids[:, None] == expected_ids[None, :]).long().argmax(-1)
            for key in expected.keys():
                exact(observations[key], expected[key][index])
            for actual, original in zip(batch[1:8], (storage.actions, storage.values, storage.advantages,
                    storage.returns, storage.actions_log_prob, storage.mu, storage.sigma)):
                exact(actual, original.flatten(0, 1)[index])
            seen.append(ids.clone())
            current["obs"] = observations
            yield batch

    def glimpse_input(module, inputs):
        assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
        exact(inputs[0], current["obs"]["terrain_map"])
        exact(inputs[1], current["obs"]["terrain_valid"])
        inputs_seen.append(inputs[0].detach().clone())

    def prepared_output(module, inputs, output):
        # The actual input adapter must normalize once from the stored meters.
        exact(output.terrain_map, current["obs"]["terrain_map"].clamp(-3., 3.) / 3.)
        exact(output.terrain_valid, (current["obs"]["terrain_valid"] > .5).float())
        prepared_seen.append(True)

    def crop_output(module, inputs, output):
        centers = inputs[1]
        assert centers.requires_grad and output.height.requires_grad
        centers.retain_grad()
        output.height.retain_grad()
        graphs.append({"centers": centers, "height": output.height})

    def glimpse_output(module, inputs, output):
        assert output.debug is None and output.terrain_tokens.requires_grad
        output.terrain_tokens.retain_grad()
        graphs[-1].update(tokens=output.terrain_tokens, valid=output.token_valid)

    def intent_output(module, inputs, output):
        intents_seen.append(output.detach().clone())

    def before_step(optimizer, args, kwargs):
        row = {name: module_grad_l1(module) for name, module in modules.items()}
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters())
        if policy.terrain_mode == "enabled":
            assert len(graphs) == 1  # One fresh graph shared by both losses.
            for graph in graphs:
                for key in ("centers", "height", "tokens"):
                    grad = graph[key].grad
                    assert grad is None or torch.isfinite(grad).all()
                    row[f"shared_{key}"] = 0. if grad is None else float(grad.abs().sum())
                token_grad = graph["tokens"].grad
                if token_grad is not None:
                    assert token_grad[~graph["valid"]].eq(0).all()
            assert len(inputs_seen) == len(prepared_seen) == 1
        else:
            assert not graphs and not inputs_seen and not prepared_seen
        assert len(intents_seen) == 1
        updates.append(row)
        graphs.clear()
        inputs_seen.clear()
        prepared_seen.clear()
        intents_seen.clear()

    def forbidden(*args, **kwargs):
        raise AssertionError("PPO update attempted to resample an observation/augmentation")

    handles = [policy.terrain_glimpse.register_forward_pre_hook(glimpse_input),
               policy.terrain_glimpse.input_adapter.register_forward_hook(prepared_output),
               policy.terrain_glimpse.crop.register_forward_hook(crop_output),
               policy.terrain_glimpse.register_forward_hook(glimpse_output),
               policy.intent_attention.register_forward_hook(intent_output),
               algorithm.optimizer.register_step_pre_hook(before_step)]
    calls_before = source.calls
    try:
        with monkeypatch.context() as patch:
            patch.setattr(storage, "mini_batch_generator", check_batches)
            patch.setattr(source, "get_observations", forbidden)
            patch.setattr(perception, "terrain_observation_from_hits", forbidden)
            patch.setattr(perception, "pack_terrain_observation", forbidden)
            patch.setattr(perception, "split_terrain_observation", forbidden)
            metrics = algorithm.update()
    finally:
        for handle in handles:
            handle.remove()
    assert source.calls == calls_before and storage.step == 0
    assert len(updates) == len(seen) == EPOCHS * MINIBATCHES
    for epoch in range(EPOCHS):
        order = torch.cat(seen[epoch * MINIBATCHES:(epoch + 1) * MINIBATCHES])
        exact(order.sort().values, expected_ids.sort().values)
        assert not torch.equal(order, expected_ids)
    for name, value in policy.named_buffers():
        exact(value, buffers[name])  # PPO update must not update normalization stats.
    for key in expected.keys():
        exact(storage.observations[key].flatten(0, 1), expected[key])
    assert all(math.isfinite(value) for value in metrics.values())
    deltas = {name: sum(float((p.detach() - old).abs().sum()) for p, old in zip(module.parameters(), before[name]))
              for name, module in modules.items()}
    return updates, deltas, metrics


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind,per_minibatch", [("valid", False), ("mixed", False), ("mixed", True)])
@pytest.mark.parametrize("objective", ["actor", "critic", "combined"])
def test_native_ppo_saved_observations_recomputation_and_real_gradients(
        device, kind, per_minibatch, objective, monkeypatch, record_property):
    torch.manual_seed(23404)
    source = SyntheticTerrainSource(device, kind)
    algorithm, obs = make_algorithm(device, source, objective=objective, per_minibatch=per_minibatch)
    # Remove the direct center-coordinate encoder path for the first backward.
    # Nonzero selector/center gradients must then pass through sampled heights.
    with torch.no_grad():
        algorithm.policy.terrain_glimpse.patch_encoder.mlp[0].weight[:, -2:].zero_()
    total_steps = 0
    for cycle in range(2):
        if cycle:
            obs = source.get_observations()
        expected = collect_rollout(algorithm, source, obs)
        updates, deltas, metrics = checked_update(algorithm, source, expected, monkeypatch)
        total_steps += len(updates)
        for name in ("selector", "encoder", "ifm"):
            assert updates[0][name] > 0 and deltas[name] > 0
            record_property(f"cycle{cycle}_{name}_first_grad_l1", updates[0][name])
            record_property(f"cycle{cycle}_{name}_delta_l1", deltas[name])
        for branch in ("actor", "critic"):
            active = objective in (branch, "combined")
            assert (deltas[branch] > 0) == active
        # Actor-only and critic-only objectives must each reach the shared crop.
        for key in ("centers", "height", "tokens"):
            assert updates[0][f"shared_{key}"] > 0
            record_property(f"cycle{cycle}_shared_{key}_grad_l1", updates[0][f"shared_{key}"])
        assert metrics["value_function"] >= 0
    record_property("optimizer_steps", total_steps)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fallback", ["disabled", "invalid"])
def test_native_ppo_fallback_has_no_terrain_contribution_or_gradient(device, fallback, monkeypatch, record_property):
    torch.manual_seed(23405)
    source = SyntheticTerrainSource(device, "valid" if fallback == "disabled" else "invalid")
    algorithm, obs = make_algorithm(device, source, mode="disabled" if fallback == "disabled" else "enabled")
    expected = collect_rollout(algorithm, source, obs)
    # Identical base weights/stats/input; compare removing all terrain keys.
    policy = algorithm.policy
    with torch.no_grad():
        action, value = policy.act_inference(expected), policy.evaluate(expected)
        mode = policy.terrain_mode
        policy.terrain_mode = "disabled"
        torch.testing.assert_close(policy.act_inference(expected), action, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(policy.evaluate(expected), value, atol=1e-6, rtol=1e-5)
        policy.terrain_mode = mode
        altered = expected.clone()
        altered["terrain_map"] = torch.randn_like(altered["terrain_map"]) * 2.
        if fallback == "disabled":
            altered["terrain_valid"] = 1. - altered["terrain_valid"]
        exact(policy.act_inference(altered), action)
        exact(policy.evaluate(altered), value)
    updates, deltas, _ = checked_update(algorithm, source, expected, monkeypatch)
    for name in ("selector", "encoder"):
        assert all(row[name] == 0. for row in updates)
        assert deltas[name] == 0.  # Fresh Adam has no existing terrain momentum.
    for name in ("ifm", "actor", "critic"):
        assert deltas[name] > 0
    record_property("optimizer_steps", len(updates))
    record_property("terrain_parameter_delta_l1", deltas["selector"] + deltas["encoder"])
