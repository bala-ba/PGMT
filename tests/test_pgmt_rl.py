"""CPU contracts for PGMT architecture and the native PMT grouped-value PPO."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from tensordict import TensorDict

from motion_tracking_rl import compat, registry
from motion_tracking_rl.algorithms import PPO
from motion_tracking_rl.networks import ActorCritic, PGMTActorCritic
from motion_tracking_rl.networks.layers import EmpiricalNormalization
from motion_tracking_rl.runners import OnPolicyRunner
from motion_tracking_rl.storage import RolloutStorage


GROUPS = {"policy": ["proprio"], "policy_history": ["proprio_history"],
          "command_window": ["command_window"], "critic": ["critic"]}


@pytest.fixture(autouse=True)
def _cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def observations(n=4):
    return TensorDict({"proprio": torch.randn(n, 96), "proprio_history": torch.randn(n, 10, 96),
                       "command_window": torch.randn(n, 8, 61), "critic": torch.randn(n, 251)}, [n])


def policy(obs, **kwargs):
    return PGMTActorCritic(obs, GROUPS, 29, actor_hidden_dims=[32, 16], critic_hidden_dims=[32, 16], **kwargs)


def test_registry_and_observation_contract():
    registry.autoload()
    assert registry.get_network("PGMTActorCritic") is PGMTActorCritic
    assert compat.validate("ppo", "pgmt", obs_sets=GROUPS) == "on_policy"
    with pytest.raises(ValueError, match="incompatible"):
        compat.validate("bpo", "pgmt")
    for feature in ("rnd", "symmetry", "recurrent"):
        with pytest.raises(ValueError, match="PGMT stage 1"):
            compat.validate("ppo", "pgmt", **{feature: True})
    with pytest.raises(ValueError, match="observation sets"):
        compat.validate("ppo", "pgmt", obs_sets={"policy", "critic"})
    obs = observations()
    model = policy(obs)
    assert model.act(obs).shape == (4, 29)
    assert model.evaluate(obs).shape == (4, 3)
    assert model.entropy.shape == (4,)
    assert model.get_actions_log_prob(model.action_mean).shape == (4,)
    assert tuple(model.value_groups) == ("upper", "lower", "auxiliary")
    assert model.command_offsets.tolist() == [0, 1, 3, 7, 15, 31, 63, 127]
    obs["proprio_history"] = torch.zeros(4, 9, 96)
    with pytest.raises(ValueError, match="policy_history"):
        model.act(obs)


def test_native_actor_critic_config_disabled_forward_fields():
    # Native RslRlPpoActorCriticCfg.to_dict includes both optional fields even
    # for a policy that does not use the forward/residual architecture.
    model = policy(observations(), forward_hidden_dims=None, forward_obs_normalization=None)
    assert model.num_value_heads == 3
    with pytest.raises(ValueError, match="forward/residual"):
        policy(observations(), forward_hidden_dims=[64])


def test_architecture_matches_local_pgmt_modules():
    """Cross-check against the agreed source, without depending on its wrapper."""
    source = Path(__file__).resolve().parents[2] / "PGMT/pgmt/modules.py"
    if not source.exists():
        pytest.skip("PGMT reference checkout is not present")
    spec = importlib.util.spec_from_file_location("_pgmt_reference_modules", source)
    reference_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference_module
    try:
        spec.loader.exec_module(reference_module)
        cfg = reference_module.ModuleConfig(8, 251, 128, 4, (32, 16), (32, 16), (32,), (32,), 10000.)
        reference = reference_module.PaperModules(cfg, stage=1)
        obs = observations()
        model = policy(obs)
        for name in ("proprio_embed", "command_embed", "history_attention", "intent_attention", "actor", "critic"):
            getattr(model, name).load_state_dict(getattr(reference, name).state_dict())
        with torch.no_grad():
            actions, values, _ = reference(obs["proprio"], obs["proprio_history"], obs["command_window"], obs["critic"])
            torch.testing.assert_close(model.act_inference(obs), actions, rtol=0, atol=0)
            torch.testing.assert_close(model.evaluate(obs), values, rtol=0, atol=0)
    finally:
        sys.modules.pop(spec.name, None)


@pytest.mark.parametrize("noise_std_type", ["scalar", "log"])
def test_native_normalization_gaussian_and_checkpoint(noise_std_type):
    obs = observations()
    model = policy(obs, actor_obs_normalization=True, critic_obs_normalization=True, noise_std_type=noise_std_type)
    assert isinstance(model.actor_obs_normalizer, EmpiricalNormalization)
    with torch.no_grad():
        model.update_normalization(obs)
        native_norm = EmpiricalNormalization(96)
        native_norm.update(obs["proprio"])
        torch.testing.assert_close(model.actor_obs_normalizer(obs["proprio"]), native_norm(obs["proprio"]))
        count = model.actor_obs_normalizer.count.item()
        model.act(obs)
        model.evaluate(obs)
        assert model.actor_obs_normalizer.count.item() == count
        torch.testing.assert_close(model.action_std, torch.ones(4, 29))
        metadata = model.get_checkpoint_metadata()
        restored = PGMTActorCritic(obs, metadata["obs_groups"], metadata["num_actions"], **metadata["policy_kwargs"])
        assert restored.load_state_dict(model.state_dict()) is True
        torch.testing.assert_close(restored.act_inference(obs), model.act_inference(obs), rtol=0, atol=0)
        torch.testing.assert_close(restored.evaluate(obs), model.evaluate(obs), rtol=0, atol=0)
        assert metadata["obs_schema"]["available_obs"]["proprio_history"]["shape"] == [10, 96]


@pytest.mark.parametrize("heads", [1, 3, 4])
@pytest.mark.parametrize("normalize", [False, True])
def test_gae_matches_independent_single_head_returns(heads, normalize):
    torch.manual_seed(8)
    n, t = 4, 6
    storage = RolloutStorage("rl", n, t, observations(n), [29], num_value_heads=heads)
    storage.rewards.copy_(torch.randn(t, n, heads))
    storage.values.copy_(torch.randn(t, n, heads))
    storage.dones[1, 0] = 1
    storage.dones[3, 1] = 1
    last = torch.randn(n, heads)
    # Separate scalar reference recursions also cover true ends and an internal
    # command switch with done=0, which must not introduce a new GAE boundary.
    expected = torch.zeros_like(storage.returns)
    gamma, lam = .99, .95
    for h in range(heads):
        for env in range(n):
            carry = 0.
            for step in reversed(range(t)):
                alive = 1. - float(storage.dones[step, env, 0])
                next_value = last[env, h] if step == t - 1 else storage.values[step + 1, env, h]
                delta = storage.rewards[step, env, h] + gamma * next_value * alive - storage.values[step, env, h]
                carry = delta + gamma * lam * alive * carry
                expected[step, env, h] = carry + storage.values[step, env, h]
    expected_adv = (expected - storage.values).sum(-1, keepdim=True)
    if normalize:
        expected_adv = (expected_adv - expected_adv.mean()) / (expected_adv.std() + 1e-8)
    storage.compute_returns(last, gamma, lam, normalize_advantage=normalize)
    torch.testing.assert_close(storage.returns, expected)
    torch.testing.assert_close(storage.advantages, expected_adv)
    batch = next(storage.mini_batch_generator(2, 1))
    assert batch[2].shape[-1] == heads and batch[3].shape[-1] == 1 and batch[4].shape[-1] == heads


@pytest.mark.parametrize("heads", [3, 4])
def test_grouped_rewards_native_timeout_and_missing_groups(heads):
    obs = observations()
    model = policy(obs, value_groups=("upper", "lower", "auxiliary") if heads == 3 else ("upper", "lower", "terrain", "auxiliary"))
    alg = PPO(model, num_learning_epochs=1, num_mini_batches=1)
    alg.init_storage("rl", 4, 2, obs, [29])
    with torch.no_grad():
        alg.act(obs)
        values = alg.transition.values.clone()
        groups = torch.randn(4, heads)
        original = groups.clone()
        timeouts = torch.tensor([False, True, False, True])
        alg.process_env_step(obs, groups.sum(-1), timeouts, {"reward_groups": groups, "time_outs": timeouts})
        expected = original + alg.gamma * values * timeouts[:, None]
        torch.testing.assert_close(alg.storage.rewards[0], expected)
        torch.testing.assert_close(groups, original)
        alg.act(obs)
        with pytest.raises(ValueError, match="reward_groups"):
            alg.process_env_step(obs, groups.sum(-1), timeouts, {})
        with pytest.raises(ValueError, match="reward_groups"):
            alg.process_env_step(obs, groups.sum(-1), timeouts, {"reward_groups": torch.zeros(4, 1)})


@pytest.mark.parametrize("per_minibatch", [False, True])
@pytest.mark.parametrize("heads", [3, 4])
def test_grouped_ppo_update_gradients_and_value_loss(per_minibatch, heads):
    obs = observations()
    model = policy(obs, value_groups=("upper", "lower", "auxiliary") if heads == 3 else ("upper", "lower", "terrain", "auxiliary"))
    alg = PPO(model, num_learning_epochs=1, num_mini_batches=1, schedule="fixed",
              normalize_advantage_per_mini_batch=per_minibatch, learning_rate=0.)
    alg.init_storage("rl", 4, 3, obs, [29])
    with torch.inference_mode():
        for _ in range(3):
            alg.act(obs)
            groups = torch.randn(4, heads)
            alg.process_env_step(obs, groups.sum(-1), torch.zeros(4, dtype=torch.bool), {"reward_groups": groups})
        alg.compute_returns(obs)
        expected_loss = (alg.storage.values - alg.storage.returns).square().mean().item()
        expected_heads = (alg.storage.values - alg.storage.returns).square().mean((0, 1))
    metrics = alg.update()
    assert metrics["value_function"] == pytest.approx(expected_loss, rel=1e-5)
    for index, name in enumerate(model.value_groups):
        assert metrics[f"value_function_{name}"] == pytest.approx(float(expected_heads[index]), rel=1e-5)
    for module in (model.proprio_embed, model.command_embed, model.history_attention,
                   model.intent_attention, model.actor, model.critic):
        grads = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads) > 0
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert alg.storage.step == 0


def test_native_single_head_scalar_reward_regression():
    obs = TensorDict({"proprio": torch.randn(4, 7), "critic": torch.randn(4, 9)}, [4])
    model = ActorCritic(obs, {"policy": ["proprio"], "critic": ["critic"]}, 2,
                        actor_hidden_dims=[16], critic_hidden_dims=[16])
    alg = PPO(model, num_learning_epochs=1, num_mini_batches=1)
    alg.init_storage("rl", 4, 1, obs, [2])
    with torch.no_grad():
        alg.act(obs)
        old_values = alg.transition.values.clone()
        rewards = torch.tensor([1., 2., -3., 5.])
        timeouts = torch.tensor([False, False, True, False])
        alg.process_env_step(obs, rewards, timeouts, {"time_outs": timeouts})
        expected = rewards + alg.gamma * (old_values * timeouts[:, None]).squeeze(1)
        torch.testing.assert_close(alg.storage.rewards[0, :, 0], expected, rtol=0, atol=0)
        assert alg.storage.rewards.shape == (1, 4, 1)


def test_native_runner_save_and_load(tmp_path):
    class Env:
        num_envs, num_actions, device = 4, 29, "cpu"
        def get_observations(self):
            return obs

    obs = observations()
    cfg = {"num_steps_per_env": 2, "save_interval": 1, "obs_groups": GROUPS,
           "algorithm": {"class_name": "PPO", "num_learning_epochs": 1, "num_mini_batches": 1},
           "policy": {"class_name": "PGMTActorCritic", "actor_hidden_dims": [32, 16],
                      "critic_hidden_dims": [32, 16], "actor_obs_normalization": True,
                      "critic_obs_normalization": True}}
    runner = OnPolicyRunner(Env(), copy.deepcopy(cfg), device="cpu")
    runner.logger_type = "tensorboard"  # Normally initialized by learn().
    with torch.no_grad():
        runner.alg.policy.update_normalization(obs)
    runner.current_learning_iteration = 7
    checkpoint = tmp_path / "model.pt"
    runner.save(str(checkpoint))
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["policy_metadata"]["value_groups"] == ["upper", "lower", "auxiliary"]
    restored = OnPolicyRunner(Env(), copy.deepcopy(cfg), device="cpu")
    restored.load(str(checkpoint))
    assert restored.current_learning_iteration == 7
    with torch.no_grad():
        torch.testing.assert_close(restored.alg.policy.act_inference(obs), runner.alg.policy.act_inference(obs), rtol=0, atol=0)


@pytest.mark.parametrize("load_optimizer,load_environment,expected", [(True, None, True), (False, None, False),
                                                                   (False, True, True), (True, False, False)])
def test_runner_environment_resume_hook_and_weights_only_loading(tmp_path, load_optimizer, load_environment, expected):
    class Env:
        num_envs, num_actions, device = 4, 29, "cpu"
        loaded = None
        def get_observations(self):
            return obs
        def get_checkpoint_state(self):
            return {"levels": torch.tensor([0, 3, 5, 9])}
        def load_checkpoint_state(self, state):
            self.loaded = state

    obs = observations()
    cfg = {"num_steps_per_env": 2, "save_interval": 1, "obs_groups": GROUPS,
           "algorithm": {"class_name": "PPO", "num_learning_epochs": 1, "num_mini_batches": 1},
           "policy": {"class_name": "PGMTActorCritic", "actor_hidden_dims": [32, 16], "critic_hidden_dims": [32, 16]}}
    runner = OnPolicyRunner(Env(), copy.deepcopy(cfg), device="cpu")
    runner.logger_type = "tensorboard"
    file = tmp_path / "curriculum.pt"
    runner.save(str(file))
    target = OnPolicyRunner(Env(), copy.deepcopy(cfg), device="cpu")
    target.load(str(file), load_optimizer=load_optimizer, load_environment=load_environment)
    assert (target.env.loaded is not None) == expected
    if expected:
        assert target.env.loaded["levels"].tolist() == [0, 3, 5, 9]
