"""Record a deterministic PGMT terrain/motion matrix in one native Isaac scene.

Uses the same configuration, placement and native termination as the interactive
Stage2 player. Every trial starts at frame zero; failures remain in the output.
"""
import argparse
from copy import deepcopy
from dataclasses import fields
from datetime import datetime
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ORDER = (("flat", "flat", "center"), ("rough", "rough", "center"),
         ("up_stair", "stairs", "up"), ("down_stair", "stairs", "down"),
         ("up_slope", "slope", "up"), ("down_slope", "slope", "down"),
         ("boxes", "boxes", "center"))

if os.environ.get("PMT_PGMT_PHYSICS_WORKER") != "1":
    from pgmt_process_guard import run_guarded
    raise SystemExit(run_guarded([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], ROOT))

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--motions", nargs="+", required=True)
parser.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4])
parser.add_argument("--terrains", nargs="+", choices=[o[0] for o in ORDER],
                    help="Evaluate only selected terrain/direction cases, preserving the standard order")
parser.add_argument("--patch_id", type=int, default=5)
parser.add_argument("--seed", type=int, default=1017)
parser.add_argument("--trial_limit", type=int, default=0)
parser.add_argument("--record_perception", action="store_true", help="Save actual policy maps, glimpse selections and attention for visualization")
parser.add_argument("--log_dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.headless or any(level not in range(10) for level in args.levels):
    parser.error("Use --headless and levels in 0..9")
output = Path(args.log_dir).resolve()
output.mkdir(parents=True, exist_ok=True)
status_path = output / "training_status.json"
if status_path.exists():
    parser.error("Output already exists")
status = dict(status="launching", pid=os.getpid(), training=False,
              run_id=os.environ.get("PMT_PGMT_RUN_ID"), started_at=datetime.now().isoformat())


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def save_status(**values):
    status.update(values, updated_at=datetime.now().isoformat())
    write_json(status_path, status)


save_status()
stop = {"requested": False}
signal.signal(signal.SIGTERM, lambda *_: stop.update(requested=True))
signal.signal(signal.SIGINT, lambda *_: stop.update(requested=True))
from pmt_tasks.isaacsim6_legacy import install_tensor_import_bridge
install_tensor_import_bridge()
app = AppLauncher(args, multi_gpu=False, fast_shutdown=True).app


def main():
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.io import dump_yaml
    from motion_tracking_rl.networks.pgmt_actor_critic import PGMTActorCritic
    from pmt_tasks.builder import build_env_cfg
    from pmt_tasks.isaacsim6_legacy import install_rigid_scene_schema_bridge
    from pmt_tasks.isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from pmt_tasks.mdp.commands.multi_motion_command import SamplingResult
    from pmt_tasks.pgmt_runtime import get_pgmt_state
    from pgmt_force_recording import FootForceRecorder

    install_rigid_scene_schema_bridge()
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = saved["policy_metadata"]
    assert metadata["policy_family"] == "pgmt_terrain_tracking"
    motions = []
    for path in args.motions:
        path = Path(path).resolve()
        with np.load(path) as d:
            xy = d["body_pos_w"][:, 0, :2]
            q = d["body_quat_w"][0, 0]
            source_yaw = math.atan2(float(2*(q[0]*q[3]+q[1]*q[2])), float(1-2*(q[2]**2+q[3]**2)))
            delta = xy[-1] - xy[0]
            distance = float(np.linalg.norm(delta))
            heading = math.atan2(float(delta[1]), float(delta[0])) if distance > 1e-8 else source_yaw
            motions.append(dict(path=str(path), name=path.stem, frames=len(xy),
                yaw=source_yaw-heading, net_distance_m=distance,
                path_length_m=float(np.linalg.norm(np.diff(xy, axis=0), axis=-1).sum()),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    trials = [dict(level=level, motion_index=mi, label=label, terrain=family, direction=direction,
                   directory=f"level{level}/{m['name']}/{col+1:02d}_{label}")
              for level in args.levels for mi, m in enumerate(motions)
              for col, (label, family, direction) in enumerate(ORDER)
              if args.terrains is None or label in args.terrains]
    if args.trial_limit:
        trials = trials[:args.trial_limit]
    write_json(output / "suite.json", dict(checkpoint=str(checkpoint), source_sha256=checkpoint_hash,
        motions=motions, levels=args.levels,
        terrain_order=[o[0] for o in ORDER if args.terrains is None or o[0] in args.terrains], trials=trials,
        seed=args.seed, patch_id=args.patch_id, physics="Isaac / PhysX",
        direction_semantics="up starts at basin center; down starts at x=-2.35m rim; native full clips may turn or cross the basin"))
    cfg = build_env_cfg("pgmt_stage2_terrain")
    # Replaying an archived policy must keep its recorded terrain version.
    cfg.pgmt_terrain_asset = saved["training_contract"]["runtime"]["environment"]["pgmt_terrain_asset"]
    cfg.pmt_motion_paths = [m["path"] for m in motions]
    cfg.__post_init__()
    cfg.scene.num_envs, cfg.seed = 1, args.seed
    cfg.sim.device = cfg.commands.motion.storage_device = args.device
    cfg.sim.use_fabric = cfg.sim.enable_scene_query_support = True
    cfg.curriculum.terrain_progress = None
    compatibility = output / "playback_motion_compatibility.csv"
    with compatibility.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["motion_name", "dismatched terrain"])
        writer.writerows((m["name"], "") for m in motions)
    cfg.commands.motion.motion_terrain_compatibility = str(compatibility)
    cfg.commands.motion.fixed_start_frame = 0
    cfg.commands.motion.require_valid_support_start = False
    cfg.commands.motion.placement_mode = "origin"
    cfg.commands.motion.placement_patch_grid = None
    cfg.commands.motion.placement_xy_range = (0., 0.)
    cfg.pgmt_tracking_relaxation = deepcopy(saved["training_contract"]["runtime"]["environment"]["pgmt_tracking_relaxation"])
    cfg.pgmt_terrain_observation = deepcopy(metadata["policy_kwargs"]["terrain_observation_cfg"])
    cfg.pgmt_terrain_observation["augmentation_enabled"] = False
    cfg.observations.proprio.enable_corruption = cfg.observations.proprio_history.enable_corruption = False
    for name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
        setattr(cfg.events, name, None)
    capacities = dict(gpu_max_rigid_contact_count=2**16, gpu_max_rigid_patch_count=2**14,
        gpu_found_lost_pairs_capacity=2**16, gpu_found_lost_aggregate_pairs_capacity=2**18,
        gpu_total_aggregate_pairs_capacity=2**16, gpu_collision_stack_size=2**26)
    for key, value in capacities.items():
        setattr(cfg.sim.physx, key, value)
    old = cfg.scene.robot.spawn
    shared = {f.name: deepcopy(getattr(old, f.name)) for f in fields(sim_utils.UsdFileCfg)
              if f.name not in ("func", "usd_path") and hasattr(old, f.name)}
    cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
        usd_path=str(ROOT / "logs/pgmt_p1_acceptance/native_robot/main.usd"), **shared)
    dump_yaml(str(output / "env.yaml"), cfg)
    raw = ManagerBasedRLEnv(cfg)
    recorder = None
    try:
        terrain, robot = raw.scene.terrain, raw.scene["robot"]
        command = raw.command_manager.get_term("motion")
        state = get_pgmt_state(raw)
        motion_ids = {Path(m.source_file).stem: i for i, m in enumerate(command.data_store.motions)}
        current = dict(trial=trials[0], recording=False, result=None, steps=0)
        families = terrain.manifest["spec"]["families"]
        manifest_hash = hashlib.sha256((Path(cfg.pgmt_terrain_asset)/"manifest.json").read_bytes()).hexdigest()

        def draw_samples(ids):
            mid = motion_ids[motions[current["trial"]["motion_index"]]["name"]]
            return SamplingResult(torch.full((len(ids),), mid, dtype=torch.long, device=raw.device),
                                  torch.zeros(len(ids), dtype=torch.long, device=raw.device))

        command._draw_samples = draw_samples
        original_reset = raw._reset_idx

        def reset_idx(ids):
            trial = current["trial"]
            if current["recording"]:
                flags = {name: bool(raw.termination_manager.get_term(name)[0])
                         for name in raw.termination_manager.active_terms}
                if any(flags.values()):
                    current["result"] = dict(termination_flags=flags,
                        failed=bool(raw.termination_manager.terminated[0]),
                        motion_finished=flags.get("motion_finished", False),
                        final_reference_frame=int(command.frame_ids[0]),
                        root_position=robot.data.root_pos_w[0].cpu().tolist())
            patch = next(p for p in terrain.surface_records if p["family"] == trial["terrain"]
                         and p["level"] == trial["level"] and p["patch_id"] == args.patch_id)
            terrain.terrain_types.fill_(families.index(trial["terrain"]))
            terrain.terrain_levels.fill_(trial["level"])
            terrain.env_origins[:] = terrain.env_origins.new_tensor(patch["center"])
            offset = -2.35 if trial["direction"] == "down" else 0.
            terrain.env_origins[:, 0] += offset
            terrain.env_origins[:, 2] = terrain.height(terrain.env_origins[:, :2])
            yaw = motions[trial["motion_index"]]["yaw"]
            command.cfg.placement_yaw_range = (yaw, yaw)
            current.update(patch=patch, spawn_xy=terrain.env_origins[0, :2].cpu().tolist(), spawn_offset_x=offset)
            original_reset(ids)

        raw._reset_idx = reset_idx
        env = RslRlVecEnvWrapper(raw)
        obs = env.get_observations()
        policy = PGMTActorCritic(obs, metadata["obs_groups"], env.num_actions,
                                **deepcopy(metadata["policy_kwargs"])).to(args.device)
        actual = policy.get_checkpoint_metadata()
        for key in ("signature", "obs_schema", "action_contract", "value_groups", "perception_contract"):
            assert actual[key] == metadata[key], key
        policy.load_state_dict(saved["model_state_dict"], strict=True)
        policy.eval().requires_grad_(False)
        results = []
        for ti, trial in enumerate(trials):
            if stop["requested"]:
                raise InterruptedError("Suite interrupted")
            current.update(trial=trial, recording=False, result=None, steps=0)
            # Reset the RNG for comparable independent trials; no training occurs.
            raw.seed(args.seed)
            with torch.inference_mode():
                obs, _ = env.reset()
            motion = motions[trial["motion_index"]]
            assert int(command.motion_ids[0]) == motion_ids[motion["name"]]
            assert int(command.frame_ids[0]) == 0
            destination = output / trial["directory"]
            destination.mkdir(parents=True)
            run = dict(status="running", checkpoint=str(checkpoint), motion=motion["path"],
                source_sha256=checkpoint_hash, checkpoint_loaded_strictly=True,
                loaded_iteration=saved["next_iteration"], control_dt=raw.step_dt, physics_dt=cfg.sim.dt,
                arguments=dict(level=trial["level"], patch_id=args.patch_id, direction=trial["direction"]),
                trial=trial, motion_frames=motion["frames"], spawn_xy=current["spawn_xy"],
                observation_noise=False, random_pushes=False, random_dynamics=False,
                curriculum_updates=False, action_mode="actor mean", training=False,
                force_recording=dict(schema_version=1, decimation=cfg.decimation,
                    terrain_manifest_sha256=manifest_hash, motion_sha256=motion["sha256"],
                    terrain_parameters=current["patch"]["parameters"], seed=args.seed,
                    spawn_offset_x=current["spawn_offset_x"], physics_buffer_capacities=capacities,
                    force_semantics="normal_only_world_vectors_without_friction"),
                video_states=dict(joint_names=list(robot.joint_names), body_names=list(robot.body_names),
                    terrain_asset=cfg.pgmt_terrain_asset, physics="Isaac / PhysX", rendering_only_replay=True))
            write_json(destination / "training_status.json", run)
            save_status(status="running", completed_trials=ti, total_trials=len(trials), current_trial=trial)
            recorder = FootForceRecorder(raw, state, destination, full_state=True)
            perception = None
            if args.record_perception:
                from pgmt_perception_recording import PerceptionRecorder
                perception = PerceptionRecorder(raw, state, policy, destination)
                run['perception_recording'] = dict(file='perception.npz', sample_hz=1/raw.step_dt,
                    source='exact pre-action observation passed to policy; no resampling',
                    terminal='terminal pose with last pre-action input held for one control step',
                    scan_shape=[21,21], scan_size_m=[2,2], glimpse_count=4, glimpse_size_m=.4,
                    terrain_mode=policy.terrain_mode)
            current["recording"] = True
            context = dict(episode=1, terrain=trial["terrain"], level=trial["level"], direction=trial["direction"])
            with torch.inference_mode():
                for step in range(raw.max_episode_length + 2):
                    if stop["requested"]:
                        raise InterruptedError("Suite interrupted")
                    if perception is not None:
                        perception.capture(obs)
                    recorder.begin(context)
                    action = policy.act_inference(obs)
                    assert torch.isfinite(action).all()
                    obs, _, done, _ = env.step(action)
                    recorder.finish(bool(done[0]))
                    current["steps"] = step + 1
                    if bool(done[0]):
                        break
            current["recording"] = False
            if perception is not None:
                perception.close(recorder.rows[-1])
                perception = None
            recorder.close()
            recorder = None
            if current["result"] is None:
                raise RuntimeError("Native termination did not occur")
            result = dict(trial, motion=motion["name"], control_steps=current["steps"],
                seconds=current["steps"]*raw.step_dt, **current["result"])
            results.append(result)
            run.update(status="completed", steps=current["steps"], completed_episodes=1, last_episode=result)
            write_json(destination / "training_status.json", run)
            write_json(destination / "result.json", result)
            with (output / "results.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            print(f"[SUITE {ti+1}/{len(trials)}] {trial['directory']} {result['seconds']:.2f}s "
                  f"failed={result['failed']} finished={result['motion_finished']}", flush=True)
        for key, value in policy.state_dict().items():
            assert torch.equal(value.cpu(), saved["model_state_dict"][key]), key
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_hash
        save_status(status="completed", completed_trials=len(results), total_trials=len(trials),
            policy_state_unchanged=True, source_checkpoint_unchanged=True,
            completed_motions=sum(r["motion_finished"] and not r["failed"] for r in results),
            failed_trials=sum(r["failed"] for r in results))
    finally:
        if recorder is not None:
            recorder.close()
        raw.close()


def checked_main():
    try:
        main()
    except BaseException as error:
        save_status(status="failed", error=repr(error))
        raise


from pmt_tasks.runtime_options import run_and_close
run_and_close(checked_main, app)
