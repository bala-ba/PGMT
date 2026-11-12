"""Evaluate every supplied clip from frame zero on flat native Isaac terrain.

    python scripts/eval_pgmt_emd.py --request policy.pt --output result.json --headless

The request is written by PGMTEMDEvaluation. No training, early tracking resets,
normalizer updates, time subsampling, curriculum updates or priority updates run
in this worker. Nonfinite data/unexpected resets invalidate the whole result.
"""
import argparse
from copy import deepcopy
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--request", required=True)
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.headless:
    parser.error("Priority evaluation requires --headless")
# These shims are no-ops on supported Isaac 5 releases and cover this host's 6.x.
from pmt_tasks.isaacsim6_legacy import install_tensor_import_bridge
install_tensor_import_bridge()
app = AppLauncher(args, multi_gpu=False, fast_shutdown=True).app


def main():
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedRLEnv
    from pmt_tasks.isaacsim6_legacy import install_rigid_scene_schema_bridge
    from pmt_tasks.builder import build_env_cfg
    from pmt_tasks.isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from pmt_tasks.mdp.commands.multi_motion_command import SamplingResult
    from pmt_tasks.pgmt_emd_flat_asset import build_flat_eval_asset
    from pmt_tasks.pgmt_emd_sampling import emd_config, joint_trajectory_emd
    from pmt_tasks.pgmt_contact_state import terrain_contact_filters
    from motion_tracking_rl.networks.pgmt_actor_critic import PGMTActorCritic

    install_rigid_scene_schema_bridge()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    request_path, output = Path(args.request), Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    request = torch.load(request_path, map_location="cpu", weights_only=False)
    metadata = request["policy_metadata"]
    config = emd_config(request["sampling_config"])
    files = request["motion_files"]
    if not files or len(files) != len(set(files)):
        raise ValueError("Priority evaluation requires a nonempty unique motion library")
    task = "pgmt_stage2_terrain" if metadata["policy_family"] == "pgmt_terrain_tracking" else "pgmt_stage2"
    cfg = build_env_cfg(task)
    cfg.pmt_motion_paths = files
    cfg.pmt_decimation, cfg.pmt_sim_dt = request["decimation"], request["sim_dt"]
    cfg.__post_init__()
    cfg.scene.num_envs = min(config["evaluation_num_envs"], len(files))
    cfg.seed = config["evaluation_seed"]
    cfg.sim.device = cfg.commands.motion.storage_device = args.device
    cfg.sim.use_fabric = args.device != "cpu"
    cfg.sim.enable_scene_query_support = True
    cfg.pgmt_terrain_asset = build_flat_eval_asset(output.parent / "flat_terrain", cfg.scene.num_envs)
    cfg.scene.terrain.asset_directory = cfg.pgmt_terrain_asset
    cfg.scene.terrain.family_proportions = None
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.curriculum.terrain_progress = None
    cfg.commands.motion.fixed_start_frame = 0
    cfg.commands.motion.placement_mode = "origin"
    cfg.commands.motion.placement_patch_grid = None
    cfg.commands.motion.placement_xy_mode = "center"
    cfg.commands.motion.placement_yaw_range = (0., 0.)
    compatibility = output.parent / "evaluation_compatibility.csv"
    import csv
    with compatibility.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["motion_name", "dismatched terrain"])
        writer.writerows((Path(p).stem, "") for p in files)
    cfg.commands.motion.motion_terrain_compatibility = str(compatibility)
    cfg.pgmt_terrain_observation = deepcopy(metadata["policy_kwargs"]["terrain_observation_cfg"])
    cfg.pgmt_terrain_observation["augmentation_enabled"] = False
    cfg.observations.proprio.enable_corruption = cfg.observations.proprio_history.enable_corruption = False
    for name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
        setattr(cfg.events, name, None)
    # Falling remains part of the evaluated trajectory, as in P-BFM. Never
    # concatenate a reset pose into the joint sequence or score a short prefix.
    for name in list(vars(cfg.terminations)):
        if not name.startswith("_"):
            setattr(cfg.terminations, name, None)
    filters = terrain_contact_filters(cfg.pgmt_terrain_asset, cfg.scene.terrain.prim_path)
    from pmt_tasks.pgmt_contact_state import TERRAIN_FOOT_SENSORS
    for name in TERRAIN_FOOT_SENSORS:
        getattr(cfg.scene, name).filter_prim_paths_expr = filters.copy()
    if request.get("robot_usd"):
        old = cfg.scene.robot.spawn
        shared = {f.name: deepcopy(getattr(old, f.name)) for f in fields(sim_utils.UsdFileCfg)
                  if f.name not in ("func", "usd_path") and hasattr(old, f.name)}
        cfg.scene.robot.spawn = sim_utils.UsdFileCfg(usd_path=request["robot_usd"], **shared)
    raw = ManagerBasedRLEnv(cfg)
    try:
        command, robot = raw.command_manager.get_term("motion"), raw.scene["robot"]
        actual_files = [str(Path(m.source_file).resolve()) for m in command.data_store.motions]
        if set(actual_files) != set(files):
            raise ValueError("Evaluator loaded a different motion library")
        lookup = {path: i for i, path in enumerate(actual_files)}
        assigned = torch.zeros(raw.num_envs, dtype=torch.long, device=raw.device)
        command._draw_samples = lambda ids: SamplingResult(assigned[ids].clone(), torch.zeros_like(assigned[ids]))
        # Separate each simulated robot by 100 m while using the same plane.
        width = int(np.ceil(np.sqrt(raw.num_envs)))
        ids = torch.arange(raw.num_envs, device=raw.device)
        raw.scene.terrain.env_origins[:, 0] = (ids // width - width // 2) * 100.
        raw.scene.terrain.env_origins[:, 1] = (ids % width - width // 2) * 100.
        env = RslRlVecEnvWrapper(raw)
        policy = PGMTActorCritic(env.get_observations(), metadata["obs_groups"], env.num_actions,
                                **deepcopy(metadata["policy_kwargs"])).to(args.device)
        for key in ("signature", "obs_schema", "action_contract", "value_groups", "perception_contract"):
            if policy.get_checkpoint_metadata().get(key) != metadata.get(key):
                raise ValueError(f"Priority evaluation policy mismatch: {key}")
        policy.load_state_dict(request["model_state_dict"], strict=True)
        policy.eval().requires_grad_(False)
        scores, frame_counts = [], []
        with torch.inference_mode():
            for first in range(0, len(files), raw.num_envs):
                batch = files[first:first + raw.num_envs]
                selected = torch.tensor([lookup[p] for p in batch], device=raw.device)
                assigned[:] = selected[-1]
                assigned[:len(batch)] = selected
                obs, _ = env.reset()
                lengths = command.data_store.motion_lengths[selected].tolist()
                if not command.frame_ids.eq(0).all():
                    raise RuntimeError("EMD evaluation must start at frame zero")
                poses = [robot.data.joint_pos[:len(batch)].clone()]
                for step in range(1, max(lengths)):
                    action = policy.act_inference(obs)
                    if not torch.isfinite(action).all():
                        raise RuntimeError("Nonfinite EMD evaluation action")
                    obs, _, done, _ = env.step(action)
                    if done.any():
                        raise RuntimeError("Unexpected reset during full-clip EMD evaluation")
                    poses.append(robot.data.joint_pos[:len(batch)].clone())
                poses = torch.stack(poses).cpu().numpy()
                for local, (mid, length) in enumerate(zip(selected.tolist(), lengths)):
                    reference = command.full_reference_store._stacked_joint_pos[mid, :length].float().cpu().numpy()
                    scores.append(joint_trajectory_emd(poses[:length, local], reference))
                    frame_counts.append(length)
                print(f"[EMD] evaluated {first+len(batch)}/{len(files)} clips", flush=True)
        for key, value in policy.state_dict().items():
            if not torch.equal(value.cpu(), request["model_state_dict"][key]):
                raise RuntimeError(f"Evaluation mutated policy/normalizer: {key}")
        result = dict(metric="joint_uniform_emd_euclidean", terrain="flat", frame_start=0,
                      motion_files=files, emd=scores, frame_counts=frame_counts,
                      global_step=request["global_step"], seed=config["evaluation_seed"],
                      policy_sha256=hashlib.sha256(request_path.read_bytes()).hexdigest(),
                      full_trajectories=True, policy_unchanged=True)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(output)
    finally:
        raw.close()


from pmt_tasks.runtime_options import run_and_close
run_and_close(main, app)
