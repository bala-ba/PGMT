"""Four-head PGMT migration, PPO and strict persistence (real old checkpoints)."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from motion_tracking_rl.networks.pgmt_actor_critic import PGMTActorCritic
from motion_tracking_rl.pgmt_value_contract import TERRAIN_VALUE_GROUPS
from motion_tracking_rl.runners import PGMTStage2Runner
from motion_tracking_rl.utils.pgmt_checkpoint import capture_rng, restore_rng
from test_pgmt_checkpoint import Env, GROUPS, SOURCE, same

P2_SOURCE = Path(__file__).resolve().parents[1] / "logs/pgmt_p245/resume_160/final.pt"
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))]


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def runner(device="cpu", groups=TERRAIN_VALUE_GROUPS):
    old = torch.load(SOURCE, map_location="cpu", weights_only=False)
    cfg = dict(num_steps_per_env=3, save_interval=1, obs_groups=GROUPS,
        pgmt_training_contract=dict(num_envs=4, task="four-head-test", reward_groups=list(groups)),
        algorithm=dict(class_name="PPO", learning_rate=1e-4, schedule="fixed", num_learning_epochs=2, num_mini_batches=1),
        policy=dict(class_name="PGMTActorCritic", **old["policy_metadata"]["policy_kwargs"],
                    terrain_glimpse_cfg={}, terrain_mode="enabled", value_groups=list(groups)))
    env = Env(device)
    # Probe in the source policy's observed scale. Arbitrary unit-variance raw
    # features can be thousands of normalized stds and saturate the selector.
    for key, normalizer in (("proprio", "actor"), ("proprio_history", "actor"),
                            ("critic", "critic"), ("command_window", "command")):
        prefix = f"{normalizer}_obs_normalizer"
        mean = old["model_state_dict"][prefix + "._mean"].to(device)
        std = old["model_state_dict"][prefix + "._std"].to(device)
        env.obs[key] = mean + .2 * std * env.obs[key]
    return PGMTStage2Runner(env, cfg, device=device)


def update(model):
    alg = model.alg
    head = alg.policy.critic[-1]
    gradients = []
    hook = head.weight.register_hook(lambda grad: gradients.append(grad.detach().abs().sum(-1)))
    module_gradients = {name: [] for name in ("selector", "patch_encoder", "ifm", "actor")}
    hooks = [p.register_hook(lambda grad, name=name: module_gradients[name].append(grad.detach().abs().sum()))
             for name, module in (("selector", alg.policy.terrain_glimpse.selector),
                                  ("patch_encoder", alg.policy.terrain_glimpse.patch_encoder),
                                  ("ifm", alg.policy.intent_attention), ("actor", alg.policy.actor))
             for p in module.parameters()]
    with torch.inference_mode():
        for i in range(3):
            alg.act(model.env.obs)
            groups = torch.randn(4, 4, device=model.device)
            groups[:, 2] -= 2.  # Negative terrain costs give the initially zero row a real target.
            timeout = torch.tensor([False, i == 1, False, False], device=model.device)
            done = timeout.clone()
            done[2] = i == 2
            values = alg.transition.values.clone()
            alg.process_env_step(model.env.obs, groups.sum(-1), done, {"reward_groups": groups, "time_outs": timeout})
            torch.testing.assert_close(alg.storage.rewards[i], groups + alg.gamma * values * timeout[:, None])
        assert alg.storage.values.shape == (3, 4, 4)
        alg.compute_returns(model.env.obs)
        assert alg.storage.returns.shape == (3, 4, 4) and alg.storage.advantages.shape == (3, 4, 1)
    metrics = alg.update()
    hook.remove()
    for h in hooks:
        h.remove()
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert isinstance(head, torch.nn.Linear)
    assert torch.isfinite(head.weight.grad).all() and torch.stack(gradients).sum(0).gt(0).all()
    assert head.weight[2].abs().sum() > 0
    for values in module_gradients.values():
        assert sum(values) > 0
    assert sum(metrics[f"value_function_{name}"] for name in TERRAIN_VALUE_GROUPS) / 4 == pytest.approx(metrics["value_function"], rel=1e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("source", [SOURCE, P2_SOURCE], ids=["stage1_15000", "p2_30"])
def test_real_three_head_migration_and_two_four_head_resume_cycles(device, source, tmp_path):
    model = runner(device)
    saved = torch.load(source, map_location="cpu", weights_only=False)
    env_before = deepcopy(model.env.state)
    new_branch = {k: v.clone() for k, v in model.alg.policy.state_dict().items() if k.startswith("terrain_glimpse.")}
    report = model.initialize_three_head(source)
    assert report["row_mapping"] == {"upper": [0, 0], "lower": [1, 1], "auxiliary": [2, 3]}
    same(env_before, model.env.state)
    assert not model.alg.optimizer.state and model.completed_iterations == 0
    policy = model.alg.policy
    state = policy.state_dict()
    for key, value in saved["model_state_dict"].items():
        if key in report["remapped"]:
            same(state[key][[0, 1, 3]], value)
            assert state[key][2].eq(0).all()
        else:
            same(state[key], value)
    if saved["policy_metadata"]["policy_family"] == "pgmt_stage1":
        same(new_branch, {k: state[k] for k in new_branch})
        assert len(report["initialized"]) == 10 and len(report["copied"]) == 47
    else:
        assert not report["initialized"] and len(report["copied"]) == 57
    metadata = saved["policy_metadata"]
    old = PGMTActorCritic(model.env.obs, metadata["obs_groups"], 29, **metadata["policy_kwargs"]).to(device)
    old.load_state_dict(saved["model_state_dict"], strict=True)
    old.eval(); policy.eval()
    policy.terrain_mode = "disabled" if source == SOURCE else "enabled"
    with torch.no_grad():
        torch.testing.assert_close(policy.act_inference(model.env.obs), old.act_inference(model.env.obs), rtol=1e-6, atol=1e-6)
        values = policy.evaluate(model.env.obs)
        # A 3-row versus 4-row GEMM can select a different float32 reduction.
        # Tensor migration above is bit-exact; also verify the affine map in float64.
        torch.testing.assert_close(values[:, [0, 1, 3]], old.evaluate(model.env.obs), rtol=2e-6, atol=2e-5)
        _, intent = old._encode(model.env.obs)
        hidden = torch.cat((old.critic_obs_normalizer(old.get_critic_obs(model.env.obs)), intent), -1)
        for layer in list(old.critic.children())[:-1]:
            hidden = layer(hidden)
        hidden = hidden.double()
        original_value = torch.nn.functional.linear(hidden, old.critic[-1].weight.double(), old.critic[-1].bias.double())
        mapped_value = torch.nn.functional.linear(hidden, policy.critic[-1].weight.double(), policy.critic[-1].bias.double())
        torch.testing.assert_close(mapped_value[:, [0, 1, 3]], original_value, rtol=1e-12, atol=1e-12)
        assert values[:, 2].eq(0).all()
    policy.terrain_mode = "enabled"
    meta = policy.get_checkpoint_metadata()
    assert meta["policy_family"] == "pgmt_terrain_tracking" and meta["signature"]["num_value_heads"] == 4
    restored_policy = PGMTActorCritic(model.env.obs, meta["obs_groups"], 29, **meta["policy_kwargs"]).to(device)
    restored_policy.load_state_dict(policy.state_dict())
    for cycle in range(2):
        update(model)
        model.alg._set_optimizer_lr(7e-5)
        path = tmp_path / f"four_{cycle}.pt"
        model.save(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["kind"] == "pgmt_terrain_training"
        assert payload["execution_options"]["actor_critic_forward"] == "joint_encode"
        assert payload["execution_options"]["ppo_log_scalars"] == "gpu_accumulate_single_read"
        assert payload["iter"] == payload["next_iteration"] == cycle+1
        rng = capture_rng(device)
        expected_reset = torch.rand(3, device=device)
        restore_rng(rng, device)
        new = runner(device)
        new.load(path)
        same(new.alg.policy.state_dict(), model.alg.policy.state_dict())
        same(new.alg.optimizer.state_dict(), model.alg.optimizer.state_dict())
        same(new.env.state, model.env.state)
        same(new.env.last_reset_draw, expected_reset)
        assert new.alg.learning_rate == 7e-5 and new.completed_iterations == cycle+1
        assert new.current_learning_iteration == cycle+1 and new.env.resets == 1
        model = new


@pytest.mark.parametrize("corruption", ["group_order", "head_signature", "observation", "architecture", "head_shape", "nonfinite", "perception"])
def test_bad_migration_is_rejected_before_any_parameter_change(corruption, tmp_path):
    model = runner()
    saved = torch.load(P2_SOURCE, map_location="cpu", weights_only=False)
    if corruption == "group_order": saved["policy_metadata"]["value_groups"] = ["lower", "upper", "auxiliary"]
    elif corruption == "head_signature": saved["policy_metadata"]["signature"]["num_value_heads"] = 4
    elif corruption == "observation": saved["policy_metadata"]["obs_schema"]["available_obs"]["critic"]["shape"] = [250]
    elif corruption == "architecture": saved["policy_metadata"]["signature"]["embedding_dim"] = 256
    elif corruption == "head_shape": saved["model_state_dict"]["critic.6.weight"] = torch.zeros(4, 128)
    elif corruption == "nonfinite": saved["model_state_dict"]["critic.6.weight"][0, 0] = float("nan")
    elif corruption == "perception": saved["policy_metadata"]["perception_contract"]["fusion"]["terrain_logit_bias"] = 1.
    path = tmp_path / "bad.pt"
    torch.save(saved, path)
    before = deepcopy(model.alg.policy.state_dict())
    with pytest.raises(ValueError):
        model.initialize_three_head(path)
    same(before, model.alg.policy.state_dict())
    assert model.initialization is None and not model.alg.optimizer.state


@pytest.mark.parametrize("field", ["available_obs", "hash", "obs_groups"])
def test_stage1_migration_rejects_incomplete_schema(field, tmp_path):
    model = runner()
    saved = torch.load(SOURCE, map_location="cpu", weights_only=False)
    schema = saved["policy_metadata"]["obs_schema"]
    if field == "available_obs":
        schema[field].pop("critic")
    elif field == "hash":
        schema[field] = "wrong"
    else:
        schema[field]["critic"] = ["proprio"]
    path = tmp_path / "bad_stage1.pt"
    torch.save(saved, path)
    before = deepcopy(model.alg.policy.state_dict())
    with pytest.raises(ValueError, match="schema"):
        model.initialize_three_head(path)
    same(before, model.alg.policy.state_dict())


@pytest.mark.parametrize("corruption", ["kind", "groups", "weights", "relaxation", "perception",
                                       "terminations", "episode_length_s", "optimizer_head"])
def test_four_head_strict_resume_refuses_changed_contracts_atomically(corruption, tmp_path):
    model = runner()
    model.initialize_three_head(SOURCE)
    update(model)
    path = tmp_path / "four.pt"
    model.save(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if corruption == "kind": saved["kind"] = "pgmt_perception_training"
    elif corruption == "groups": saved["policy_metadata"]["value_groups"] = ["upper", "lower", "auxiliary", "terrain"]
    elif corruption in ("weights", "relaxation", "perception", "terminations", "episode_length_s"):
        saved["training_contract"]["runtime"][corruption] = "changed"
    elif corruption == "optimizer_head":
        slots = [v for v in saved["optimizer_state_dict"]["state"].values() if v["exp_avg"].shape == (4, 128)]
        assert len(slots) == 1
        slots[0]["exp_avg"] = torch.zeros(3, 128)
    torch.save(saved, path)
    before = deepcopy(model.alg.policy.state_dict())
    before_optimizer, before_env = deepcopy(model.alg.optimizer.state_dict()), deepcopy(model.env.state)
    with pytest.raises(ValueError):
        model.load(path)
    same(before, model.alg.policy.state_dict())
    same(before_optimizer, model.alg.optimizer.state_dict())
    same(before_env, model.env.state)
    assert model.env.resets == 0


@pytest.mark.parametrize("groups", [["upper", "lower", "auxiliary", "terrain"], ["upper"] * 4, [], 4])
def test_invalid_value_group_order_is_rejected(groups):
    with pytest.raises(ValueError, match="value_groups"):
        runner(groups=groups) if isinstance(groups, list) else PGMTActorCritic(Env("cpu").obs, GROUPS, 29, value_groups=groups)


def test_formal_task_has_27_rewards_four_groups_and_enabled_relaxation():
    from pmt_tasks.builder import build_task_config
    from pmt_tasks.registry_gym import gym_id_for_stem, task_stem_for_id
    cfg = build_task_config("pgmt_stage2_terrain")
    assert list(cfg.reward.groups) == list(cfg.network.value_groups) == list(TERRAIN_VALUE_GROUPS)
    assert cfg.tracking_relaxation.enabled and not cfg.reward.recovery_upward_velocity
    assert len(cfg.reward_weights) == 27
    assert {k: v for k, v in cfg.reward_weights.items() if k.startswith("terrain_")} == dict(
        terrain_touchdown_quality=10., terrain_reference_contact_match=1.5, terrain_slip=-1.,
        terrain_stumble=-20., terrain_contact_switching=-30., terrain_contact_force=-1e-6)
    assert list(build_task_config("pgmt_stage2").reward.groups) == ["upper", "lower", "auxiliary"]
    assert not build_task_config("pgmt_stage2").tracking_relaxation.enabled
    assert task_stem_for_id(gym_id_for_stem("pgmt_stage2_terrain")) == "pgmt_stage2_terrain"
