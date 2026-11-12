"""Joint PGMT encoding must preserve sampling, both gradients and PPO updates."""
from copy import deepcopy

import pytest
import torch

from motion_tracking_rl.algorithms import PPO
from motion_tracking_rl.networks import PGMTActorCritic
from motion_tracking_rl.pgmt_value_contract import STAGE1_VALUE_GROUPS, TERRAIN_VALUE_GROUPS
from motion_tracking_rl.utils.pgmt_checkpoint import capture_rng, restore_rng
from test_pgmt_perception_ppo import DEVICES, GROUPS, SyntheticTerrainSource


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def make_policy(obs, heads=4, mode="enabled", **kwargs):
    policy = PGMTActorCritic(
        obs, GROUPS, 29, actor_hidden_dims=[32, 16], critic_hidden_dims=[32, 16],
        actor_obs_normalization=True, critic_obs_normalization=True,
        terrain_glimpse_cfg=None if mode == "blind" else {},
        terrain_mode="disabled" if mode == "blind" else mode,
        value_groups=TERRAIN_VALUE_GROUPS if heads == 4 else STAGE1_VALUE_GROUPS,
        **kwargs).to(obs.device)
    policy.update_normalization(obs)
    return policy


def assert_rng_equal(first, second):
    for key in ("torch", "cuda"):
        if first[key] is None:
            assert second[key] is None
        else:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("heads,mode,kind", [
    (3, "blind", "valid"), (3, "enabled", "mixed"),
    (4, "enabled", "valid"), (4, "enabled", "mixed"),
    (4, "enabled", "invalid"), (4, "disabled", "mixed"),
])
@pytest.mark.parametrize("objective", ["actor", "critic", "combined"])
def test_joint_matches_separate_outputs_rng_and_gradients(device, heads, mode, kind, objective, record_property):
    torch.manual_seed(251015)
    obs = SyntheticTerrainSource(device, kind).get_observations()
    separate = make_policy(obs, heads, mode)
    joint = deepcopy(separate)
    before = {key: value.clone() for key, value in joint.named_buffers()}
    calls = []
    hook = joint.intent_attention.register_forward_hook(lambda *args: calls.append(True))
    rng = capture_rng(device)
    old_action = separate.act(obs)
    old_value = separate.evaluate(obs)
    old_rng = capture_rng(device)
    restore_rng(rng, device)
    action, value = joint.act_and_evaluate(obs)
    assert_rng_equal(capture_rng(device), old_rng)
    assert len(calls) == 1
    for actual, expected in ((action, old_action), (value, old_value),
                             (joint.action_mean, separate.action_mean),
                             (joint.action_std, separate.action_std),
                             (joint.entropy, separate.entropy),
                             (joint.get_actions_log_prob(action), separate.get_actions_log_prob(old_action))):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def loss(policy, actions, values):
        actor = -policy.get_actions_log_prob(actions).mean() - .01 * policy.entropy.mean()
        targets = torch.arange(heads, device=device, dtype=values.dtype) + .5
        critic = (values - targets).square().mean()
        return actor if objective == "actor" else critic if objective == "critic" else actor + critic

    loss(separate, old_action, old_value).backward()
    loss(joint, action, value).backward()
    errors = []
    for (name, actual), (old_name, expected) in zip(joint.named_parameters(), separate.named_parameters()):
        assert name == old_name
        if expected.grad is None:
            assert actual.grad is None
        else:
            assert actual.grad is not None
            torch.testing.assert_close(actual.grad, expected.grad, atol=3e-7, rtol=3e-4, msg=name)
            errors.append(float((actual.grad - expected.grad).abs().max()))
    assert sum(float(p.grad.abs().sum()) for p in joint.intent_attention.parameters() if p.grad is not None) > 0
    for key, value in joint.named_buffers():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    assert joint.get_checkpoint_metadata() == separate.get_checkpoint_metadata()
    record_property("max_gradient_absolute_error", max(errors))
    # A subsequent call must build a fresh graph after weights/observations change.
    torch.optim.Adam(joint.parameters(), lr=1e-4).step()
    joint.zero_grad(set_to_none=True)
    changed = obs.clone()
    changed["proprio"] += .1
    action, value = joint.act_and_evaluate(changed)
    loss(joint, action, value).backward()
    assert len(calls) == 2
    hook.remove()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("state_dependent", [False, True])
@pytest.mark.parametrize("std_type", ["scalar", "log"])
def test_joint_preserves_distribution_variants(device, state_dependent, std_type):
    obs = SyntheticTerrainSource(device, "mixed").get_observations()
    policy = make_policy(obs, state_dependent_std=state_dependent, noise_std_type=std_type)
    with torch.inference_mode():
        rng = capture_rng(device)
        expected_action, expected_value = policy.act(obs), policy.evaluate(obs)
        expected_std = policy.action_std.clone()
        restore_rng(rng, device)
        action, value = policy.act_and_evaluate(obs)
        for actual, expected in ((action, expected_action), (value, expected_value),
                                 (policy.action_std, expected_std)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mean_rollout", [False, True])
def test_four_head_ppo_joint_matches_fallback_over_two_updates(device, mean_rollout):
    torch.manual_seed(251016)
    source = SyntheticTerrainSource(device, "mixed")
    obs = source.get_observations()
    policy = make_policy(obs)
    algorithms = []
    for shared in (False, True):
        model = deepcopy(policy)
        if not shared:
            model.act_and_evaluate = None  # Exercise the production PPO fallback.
        alg = PPO(model, device=device, num_learning_epochs=2, num_mini_batches=2,
                  learning_rate=1e-4, schedule="adaptive", use_mean_action_for_rollout=mean_rollout)
        alg.init_storage("rl", len(obs), 3, obs, [29])
        algorithms.append(alg)
    for _ in range(2):
        observations = [source.get_observations() for _ in range(4)]
        rewards = torch.randn(3, len(obs), 4, device=device)
        rng = capture_rng(device)
        endings, metrics, states = [], [], []
        for alg in algorithms:
            restore_rng(rng, device)
            with torch.inference_mode():
                for step in range(3):
                    alg.act(observations[step])
                    done = torch.zeros(len(obs), device=device, dtype=torch.bool)
                    done[0] = step == 1
                    timeout = torch.zeros_like(done)
                    done[1] = timeout[1] = step == 2
                    alg.process_env_step(observations[step + 1], rewards[step].sum(-1), done,
                                         dict(reward_groups=rewards[step], time_outs=timeout))
                alg.compute_returns(observations[-1])
            states.append({name: getattr(alg.storage, name).clone() for name in (
                "values", "returns", "advantages", "actions", "actions_log_prob")})
            metrics.append(alg.update())
            endings.append(capture_rng(device))
        assert_rng_equal(*endings)
        for name in states[0]:
            torch.testing.assert_close(states[0][name], states[1][name], atol=1e-5, rtol=1e-4)
        assert metrics[0].keys() == metrics[1].keys()
        for name in metrics[0]:
            assert metrics[0][name] == pytest.approx(metrics[1][name], abs=1e-5, rel=1e-4), name
        for name, value in algorithms[0].policy.state_dict().items():
            torch.testing.assert_close(value, algorithms[1].policy.state_dict()[name], atol=1e-5, rtol=1e-4, msg=name)
        for index, slots in algorithms[0].optimizer.state_dict()["state"].items():
            for name, value in slots.items():
                torch.testing.assert_close(value, algorithms[1].optimizer.state_dict()["state"][index][name],
                                           atol=1e-6, rtol=1e-3, msg=name)
