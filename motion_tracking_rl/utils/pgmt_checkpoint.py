"""Explicit PGMT migration and strict resume, independent of Isaac imports."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from motion_tracking_rl.pgmt_value_contract import STAGE1_VALUE_GROUPS, TERRAIN_VALUE_GROUPS


def canonical(value):
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (Path, torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if callable(value):
        return value.__module__ + ":" + value.__qualname__
    raise TypeError(f"Unsupported PGMT contract value: {type(value)}")


def digest(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_tensors(saved, current):
    if saved.keys() != current.keys():
        raise ValueError("PGMT checkpoint tensor keys differ")
    for name, value in saved.items():
        target = current[name]
        if (not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype
                or not torch.isfinite(value).all()):
            raise ValueError(f"Invalid PGMT checkpoint tensor: {name}")


def migrate_stage1(policy, path):
    """Copy every old tensor, require unchanged old semantics, retain new weights."""
    path = Path(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    old_meta, new_meta = saved.get("policy_metadata", {}), policy.get_checkpoint_metadata()
    if old_meta.get("policy_family") != "pgmt_stage1" or new_meta.get("policy_family") != "pgmt_perception_only":
        raise ValueError("Migration requires Stage-1 -> PGMT perception_only")
    for key in ("num_actions", "action_contract", "value_groups"):
        if old_meta.get(key) != new_meta.get(key):
            raise ValueError(f"Stage-1 migration contract differs: {key}")
    for key, value in old_meta["signature"].items():
        if new_meta["signature"].get(key) != value:
            raise ValueError(f"Stage-1 migration architecture differs: {key}")
    for key, group in old_meta["obs_groups"].items():
        if new_meta["obs_groups"].get(key) != group:
            raise ValueError(f"Stage-1 observation group differs: {key}")
    for key, schema in old_meta["obs_schema"]["available_obs"].items():
        if new_meta["obs_schema"]["available_obs"].get(key) != schema:
            raise ValueError(f"Stage-1 observation schema differs: {key}")
    current = policy.state_dict()
    base = {k: v for k, v in current.items() if not k.startswith("terrain_glimpse.")}
    validate_tensors(saved["model_state_dict"], base)
    added = sorted(set(current) - set(base))
    if not added:
        raise ValueError("Missing new perception parameters")
    policy.load_state_dict({k: saved["model_state_dict"][k] if k in base else v for k, v in current.items()}, strict=True)
    for key in base:
        torch.testing.assert_close(policy.state_dict()[key].cpu(), saved["model_state_dict"][key], atol=0, rtol=0)
    return dict(source=str(path.resolve()), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_iteration=saved["iter"], copied=sorted(base), initialized=added,
                source_metadata=old_meta, target_metadata=new_meta,
                optimizer="fresh", completed_iterations=0, sampler_curriculum="fresh")


def migrate_three_head(policy, path):
    """Stage1 or P2 -> four-head terrain policy, validating before mutation.

    All old parameters/statistics are copied exactly except the last critic
    Linear: upper/lower/auxiliary rows move to 0/1/3; terrain row 2 is zero.
    This initializes a new training stage, not an optimizer/curriculum resume.
    """
    path = Path(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    old, new = saved.get("policy_metadata", {}), policy.get_checkpoint_metadata()
    family = old.get("policy_family")
    if (family not in ("pgmt_stage1", "pgmt_perception_only")
            or new.get("policy_family") != "pgmt_terrain_tracking" or policy.terrain_glimpse is None
            or old.get("value_groups") != list(STAGE1_VALUE_GROUPS)
            or new.get("value_groups") != list(TERRAIN_VALUE_GROUPS)
            or old.get("signature", {}).get("num_value_heads") != 3):
        raise ValueError("Four-head migration requires a Stage1/P2 three-head source and terrain target")
    for key in ("num_actions", "action_contract", "policy_class"):
        if old.get(key) != new.get(key):
            raise ValueError(f"Three-head migration contract differs: {key}")
    # Reconstruct the exact expected source architecture, allowing only the
    # declared new perception branch (Stage1) and changed value output schema.
    expected_signature = deepcopy(new["signature"])
    expected_signature["num_value_heads"] = 3
    expected_signature["value_groups"] = list(STAGE1_VALUE_GROUPS)
    expected_kwargs = deepcopy(new["policy_kwargs"])
    expected_kwargs.pop("value_groups")
    if family == "pgmt_stage1":
        for key in ("value_groups", "terrain_mode", "terrain_glimpse_cfg", "terrain_observation_cfg", "perception_contract"):
            expected_signature.pop(key, None)
        for key in ("terrain_mode", "terrain_glimpse_cfg", "terrain_observation_cfg"):
            expected_kwargs.pop(key, None)
    if old.get("signature") != expected_signature or old.get("policy_kwargs") != expected_kwargs:
        raise ValueError("Three-head migration architecture/perception contract differs")
    expected_groups = deepcopy(new["obs_groups"])
    if family == "pgmt_stage1":
        expected_groups.pop("terrain_map")
        expected_groups.pop("terrain_valid")
    if old.get("obs_groups") != expected_groups:
        raise ValueError("Three-head migration observation groups differ")
    old_schema = old.get("obs_schema", {})
    required_obs = {key for names in expected_groups.values() for key in names}
    schema_payload = {key: value for key, value in old_schema.items() if key != "hash"}
    schema_hash = hashlib.sha256(json.dumps(schema_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (not required_obs <= old_schema.get("available_obs", {}).keys()
            or old_schema.get("obs_groups") != expected_groups or old_schema.get("hash") != schema_hash):
        raise ValueError("Three-head migration requires complete, consistent observation schema metadata")
    for key, schema in old_schema["available_obs"].items():
        if new["obs_schema"]["available_obs"].get(key) != schema:
            raise ValueError(f"Three-head migration observation schema differs: {key}")
    if family == "pgmt_perception_only" and (old_schema != new["obs_schema"]
                                               or old.get("perception_contract") != new.get("perception_contract")):
        raise ValueError("P2 migration requires the same observation/perception contract")
    current = policy.state_dict()
    last_linear = [(name, layer) for name, layer in policy.critic.named_modules() if isinstance(layer, torch.nn.Linear)][-1]
    head_keys = [f"critic.{last_linear[0]}.{suffix}" for suffix in ("weight", "bias")]
    base = {key: value for key, value in current.items()
            if family != "pgmt_stage1" or not key.startswith("terrain_glimpse.")}
    expected_tensors = {key: value[:3] if key in head_keys else value for key, value in base.items()}
    validate_tensors(saved["model_state_dict"], expected_tensors)
    new_state = {key: saved["model_state_dict"][key] if key in base else value for key, value in current.items()}
    for key in head_keys:
        new_state[key] = torch.zeros_like(current[key])
        new_state[key][[0, 1, 3]] = saved["model_state_dict"][key].to(new_state[key].device)
    policy.load_state_dict(new_state, strict=True)
    copied = sorted(set(base) - set(head_keys))
    for key in copied:
        torch.testing.assert_close(policy.state_dict()[key].cpu(), saved["model_state_dict"][key], atol=0, rtol=0)
    for key in head_keys:
        actual = policy.state_dict()[key].cpu()
        torch.testing.assert_close(actual[[0, 1, 3]], saved["model_state_dict"][key], atol=0, rtol=0)
        assert torch.count_nonzero(actual[2]) == 0
    return dict(source=str(path.resolve()), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_iteration=saved["iter"], source_family=family, copied=copied,
                initialized=sorted(set(current) - set(base)), remapped=head_keys,
                row_mapping={"upper": [0, 0], "lower": [1, 1], "auxiliary": [2, 3]},
                terrain_row=dict(index=2, initialization="zero weight and bias"),
                source_metadata=old, target_metadata=new, optimizer="fresh",
                completed_iterations=0, sampler_curriculum="fresh")


def capture_rng(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None)


def validate_rng(saved, device):
    random.Random().setstate(saved["python"])
    np.random.RandomState().set_state(saved["numpy"])
    torch.Generator().set_state(saved["torch"].cpu())
    if torch.device(device).type == "cuda":
        torch.Generator(device=device).set_state(saved["cuda"].cpu())
    elif saved["cuda"] is not None:
        raise ValueError("CUDA checkpoint cannot strictly resume on CPU")


def restore_rng(saved, device):
    validate_rng(saved, device)
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["torch"].cpu())
    if saved["cuda"] is not None:
        torch.cuda.set_rng_state(saved["cuda"].cpu(), device)


def validate_optimizer(saved, optimizer):
    groups = saved.get("param_groups", [])
    if len(groups) != len(optimizer.param_groups):
        raise ValueError("Optimizer group count differs")
    ids = []
    for old, live, template in zip(groups, optimizer.param_groups, optimizer.state_dict()["param_groups"]):
        if len(old["params"]) != len(live["params"]):
            raise ValueError("Optimizer parameter count differs")
        if old["params"] != template["params"]:
            raise ValueError("Optimizer parameter order differs")
        if {k: v for k, v in old.items() if k != "lr"} != {k: v for k, v in template.items() if k != "lr"}:
            raise ValueError("Optimizer configuration differs")
        if not math.isfinite(old["lr"]) or old["lr"] <= 0:
            raise ValueError("Invalid optimizer learning rate")
        ids.extend(old["params"])
        for index, parameter in zip(old["params"], live["params"]):
            slots = saved["state"].get(index, {})
            if slots and not {"step", "exp_avg", "exp_avg_sq"} <= slots.keys():
                raise ValueError("Incomplete Adam state")
            for key, value in slots.items():
                if (not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                        or (key != "step" and (value.shape != parameter.shape or value.dtype != parameter.dtype))
                        or (key == "step" and (value.numel() != 1 or value < 0 or value != value.floor()))
                        or (key in ("exp_avg_sq", "max_exp_avg_sq") and (value < 0).any())):
                    raise ValueError(f"Invalid Adam slot {index}/{key}")
    if len(set(ids)) != len(ids) or set(saved["state"]) - set(ids):
        raise ValueError("Optimizer parameter identity differs")


def training_payload(runner, infos=None):
    return dict(kind=_training_kind(runner), version=1, iter=runner.completed_iterations,
                next_iteration=runner.completed_iterations,
                model_state_dict=runner.alg.policy.state_dict(), optimizer_state_dict=runner.alg.optimizer.state_dict(),
                policy_metadata=runner.alg.policy.get_checkpoint_metadata(), training_contract=runner.training_contract,
                environment_state_dict=runner.env.get_checkpoint_state(), rng_state=capture_rng(runner.device),
                learning_rate=runner.alg.learning_rate, tot_timesteps=runner.tot_timesteps,
                tot_time=runner.tot_time, initialization=runner.initialization, infos=infos)


def _training_kind(runner):
    return "pgmt_terrain_training" if tuple(runner.alg.policy.value_groups) == TERRAIN_VALUE_GROUPS else "pgmt_perception_training"


def restore_training(runner, saved):
    if saved.get("kind") != _training_kind(runner) or saved.get("version") != 1:
        raise ValueError("Use explicit migration; strict resume requires the same PGMT checkpoint kind/version")
    if saved.get("training_contract") != runner.training_contract:
        raise ValueError("PGMT training contract differs; strict resume refused")
    if saved.get("policy_metadata") != runner.alg.policy.get_checkpoint_metadata():
        raise ValueError("PGMT policy/schema/perception metadata differs; strict resume refused")
    iteration = saved.get("next_iteration")
    if not isinstance(iteration, int) or iteration < 0 or saved.get("iter") != iteration:
        raise ValueError("Invalid completed/next iteration")
    validate_tensors(saved["model_state_dict"], runner.alg.policy.state_dict())
    validate_optimizer(saved["optimizer_state_dict"], runner.alg.optimizer)
    if iteration > 0 and not saved["optimizer_state_dict"]["state"]:
        raise ValueError("Trained checkpoint has no optimizer state")
    if (not isinstance(saved.get("tot_timesteps"), int) or saved["tot_timesteps"] < 0
            or not isinstance(saved.get("tot_time"), (int, float))
            or not math.isfinite(saved["tot_time"]) or saved["tot_time"] < 0
            or not isinstance(saved.get("initialization"), dict)):
        raise ValueError("Invalid training progress or initialization record")
    lr = saved["learning_rate"]
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Invalid PPO learning rate")
    for group in saved["optimizer_state_dict"]["param_groups"]:
        if not math.isclose(group["lr"], lr * group.get("lr_scale", 1.), rel_tol=1e-12):
            raise ValueError("PPO and optimizer learning rates disagree")
    validate_rng(saved["rng_state"], runner.device)
    runner.env.validate_checkpoint_state(saved["environment_state_dict"])
    sampling = saved["environment_state_dict"].get("sampler", {})
    if sampling.get("kind") == "pbfm_emd":
        global_step = iteration * runner.num_steps_per_env * runner.env.num_envs * runner.gpu_world_size
        if sampling["last_evaluation_step"] > global_step:
            raise ValueError("EMD evaluation progress is ahead of the checkpoint's training progress")
    # All compatibility checks precede mutation. A fresh physical reset may
    # consume RNG; restore BEFORE that reset, never promise a PhysX trajectory replay.
    runner.alg.policy.load_state_dict(saved["model_state_dict"], strict=True)
    runner.alg.optimizer.load_state_dict(saved["optimizer_state_dict"])
    runner.alg.learning_rate = lr
    runner.completed_iterations = runner.current_learning_iteration = iteration
    runner.alg.set_learning_iteration(iteration)
    runner.tot_timesteps, runner.tot_time = saved["tot_timesteps"], saved["tot_time"]
    runner.initialization = deepcopy(saved["initialization"])
    restore_rng(saved["rng_state"], runner.device)
    runner.env.load_checkpoint_state(saved["environment_state_dict"])
    return saved.get("infos")
