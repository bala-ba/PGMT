"""P2.4 actual Stage-1 migration, strict resume and RNG/reset ordering."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from motion_tracking_rl.networks import PGMTActorCritic
from motion_tracking_rl.runners import PGMTStage2Runner
from motion_tracking_rl.utils.pgmt_checkpoint import capture_rng, restore_rng
from pmt_tasks.pgmt_perception import terrain_observation_contract

SOURCE = Path(__file__).resolve().parents[1] / "logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt"
GROUPS = {"policy": ["proprio"], "critic": ["critic"], "policy_history": ["proprio_history"],
          "command_window": ["command_window"], "terrain_map": ["terrain_map"], "terrain_valid": ["terrain_valid"]}
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))]


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


class Env:
    num_envs, num_actions = 4, 29
    def __init__(self, device):
        self.device = device
        self.obs = TensorDict(dict(proprio=torch.randn(4, 96), critic=torch.randn(4, 251),
            proprio_history=torch.randn(4, 10, 96), command_window=torch.randn(4, 8, 61),
            terrain_map=torch.randn(4, 21, 21), terrain_valid=torch.ones(4, 21, 21)), [4]).to(device)
        self.state = dict(identity="fixture", levels=torch.tensor([0, 1, 0, 1]), clock=7,
            counts=torch.arange(12.).reshape(4, 3), failed=torch.ones(4), success=torch.ones(4) * 2)
        self.resets = 0
    def get_observations(self):
        return self.obs
    def get_checkpoint_state(self):
        return deepcopy(self.state)
    def validate_checkpoint_state(self, state):
        if state["identity"] != self.state["identity"] or state["clock"] < 0:
            raise ValueError("Environment identity/clock mismatch")
    def load_checkpoint_state(self, state):
        self.validate_checkpoint_state(state)
        self.state = deepcopy(state)
        self.last_reset_draw = torch.rand(3, device=self.device)
        self.resets += 1


def runner(device="cpu"):
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    cfg = dict(num_steps_per_env=2, save_interval=1, obs_groups=GROUPS,
        pgmt_training_contract=dict(num_envs=4, observation=terrain_observation_contract()),
        algorithm=dict(class_name="PPO", learning_rate=1e-4, schedule="fixed", num_learning_epochs=1, num_mini_batches=1),
        policy=dict(class_name="PGMTActorCritic", **source["policy_metadata"]["policy_kwargs"],
                    terrain_glimpse_cfg={}, terrain_mode="enabled"))
    return PGMTStage2Runner(Env(device), cfg, device=device), source


def update(model):
    with torch.inference_mode():
        for _ in range(2):
            model.alg.act(model.env.obs)
            groups = torch.randn(4, 3, device=model.device)
            model.alg.process_env_step(model.env.obs, groups.sum(-1), torch.zeros(4, device=model.device),
                                       {"reward_groups": groups})
        model.alg.compute_returns(model.env.obs)
    model.alg.update()


def same(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a.cpu(), b.cpu(), atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("device", DEVICES)
def test_actual_stage1_migration_and_two_strict_roundtrips(device, tmp_path):
    model, source = runner(device)
    new_before = {k: v.clone() for k, v in model.alg.policy.state_dict().items() if k.startswith("terrain_glimpse.")}
    report = model.initialize_stage1(SOURCE)
    assert len(report["copied"]) == 49 and len(report["initialized"]) == 10
    assert model.completed_iterations == 0 and not model.alg.optimizer.state
    same({k: model.alg.policy.state_dict()[k] for k in new_before}, new_before)
    meta = source["policy_metadata"]
    blind = PGMTActorCritic(model.env.obs, meta["obs_groups"], 29, **meta["policy_kwargs"]).to(device)
    blind.load_state_dict(source["model_state_dict"], strict=True)
    with torch.no_grad():
        model.alg.policy.terrain_mode = "disabled"
        torch.testing.assert_close(model.alg.policy.act_inference(model.env.obs), blind.act_inference(model.env.obs), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(model.alg.policy.evaluate(model.env.obs), blind.evaluate(model.env.obs), atol=1e-5, rtol=1e-5)
        model.alg.policy.terrain_mode = "enabled"
    for cycle in range(2):
        update(model)
        model.alg._set_optimizer_lr(7e-5)
        path = tmp_path / f"checkpoint{cycle}.pt"
        model.save(path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        assert state["iter"] == state["next_iteration"] == cycle + 1
        rng = capture_rng(device)
        expected_reset = torch.rand(3, device=device)
        restore_rng(rng, device)
        fresh, _ = runner(device)
        fresh.load(path)
        same(fresh.alg.policy.state_dict(), model.alg.policy.state_dict())
        same(fresh.alg.optimizer.state_dict(), model.alg.optimizer.state_dict())
        same(fresh.env.state, model.env.state)
        same(fresh.env.last_reset_draw, expected_reset)
        assert fresh.env.resets == 1 and fresh.current_learning_iteration == cycle + 1
        assert fresh.alg.learning_rate == 7e-5
        model = fresh


@pytest.mark.parametrize("corruption", ["kind", "contract", "metadata", "model", "optimizer", "lr", "iteration", "environment", "rng", "optimizer_order", "optimizer_empty", "optimizer_hyper", "progress"])
def test_strict_resume_rejects_before_mutation(corruption, tmp_path):
    model, _ = runner()
    model.initialize_stage1(SOURCE)
    update(model)
    path = tmp_path / "checkpoint.pt"
    model.save(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if corruption == "kind": state["kind"] = "stage1"
    elif corruption == "contract": state["training_contract"]["runtime"]["num_envs"] = 5
    elif corruption == "metadata": state["policy_metadata"]["signature"]["privileged_dim"] = 252
    elif corruption == "model": state["model_state_dict"]["std"][0] = float("nan")
    elif corruption == "optimizer": state["optimizer_state_dict"]["state"][0]["exp_avg"] = torch.zeros(100)
    elif corruption == "optimizer_order": state["optimizer_state_dict"]["param_groups"][0]["params"].reverse()
    elif corruption == "optimizer_empty": state["optimizer_state_dict"]["state"].clear()
    elif corruption == "optimizer_hyper": state["optimizer_state_dict"]["param_groups"][0]["eps"] = .01
    elif corruption == "progress": state["tot_time"] = float("nan")
    elif corruption == "lr": state["learning_rate"] = .1
    elif corruption == "iteration": state["iter"] += 1
    elif corruption == "environment": state["environment_state_dict"]["identity"] = "wrong"
    elif corruption == "rng": state["rng_state"]["torch"] = torch.zeros(2, dtype=torch.uint8)
    torch.save(state, path)
    before = deepcopy(model.alg.policy.state_dict())
    before_optim, before_env = deepcopy(model.alg.optimizer.state_dict()), deepcopy(model.env.state)
    with pytest.raises((ValueError, RuntimeError)):
        model.load(path)
    same(before, model.alg.policy.state_dict())
    same(before_optim, model.alg.optimizer.state_dict())
    same(before_env, model.env.state)
    assert model.env.resets == 0


def test_weights_only_initialization_preserves_new_sampler_contract(tmp_path):
    source, _ = runner()
    source.initialize_stage1(SOURCE)
    update(source)
    checkpoint = tmp_path / "old_sampling.pt"
    source.save(checkpoint)
    target, _ = runner()
    target.training_contract["runtime"]["sampling"] = "pbfm_emd"
    before_env = deepcopy(target.env.state)
    target.initialize_policy(checkpoint)
    same(target.alg.policy.state_dict(), source.alg.policy.state_dict())
    same(target.env.state, before_env)
    assert not target.alg.optimizer.state and target.completed_iterations == 0
    assert target.initialization["kind"] == "policy_weights_only"
    with pytest.raises(ValueError, match="fresh"):
        target.initialize_policy(checkpoint)
