"""Deterministic Stage2 policy inference on selectable PGMT terrain.

This viewer uses the trained perception policy, native Isaac dynamics, and the
existing shared reference transform. The supplied motion is used without
training terrain exclusions, distance/straightness screening, or a custom
stair-flight termination.
"""
import argparse
import csv
from copy import deepcopy
from dataclasses import fields
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TERRAIN_FAMILIES = ("flat", "stairs", "slope", "boxes", "rough", "stairs_up", "stairs_down", "platforms")
DIRECTIONAL_TERRAINS = ("stairs", "slope", "stairs_up", "stairs_down")
sys.path.insert(0, str(ROOT))
if os.environ.get("PMT_PGMT_PHYSICS_WORKER") != "1":
    from pgmt_process_guard import run_guarded
    raise SystemExit(run_guarded([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], ROOT))

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", default=str(ROOT / "logs/pgmt_stage2_center2x2/train_01/model_19999.pt"))
parser.add_argument("--motion", default=str(ROOT / "assets/motions/pgmt_clipped_50hz_tcrs_v1/walk1_subject1_clip0.npz"))
parser.add_argument("--terrain", choices=TERRAIN_FAMILIES, default="stairs",
                    help="Initial terrain family; also selectable in the viewer")
parser.add_argument("--terrain_asset", help="Override the checkpoint's recorded terrain asset")
parser.add_argument("--level", type=int, choices=range(10), default=1)
parser.add_argument("--direction", choices=("up", "down"), default="up",
                    help="Stairs/slope: start at the bottom (up) or rim (down); other terrains start at the center")
parser.add_argument("--patch_id", type=int, help="Defaults to patch 0 for P-BFM or patch 5 for legacy assets")
parser.add_argument("--speed", type=float, default=1.)
parser.add_argument("--max_episodes", type=int, default=0, help="0: keep the interactive viewer running")
parser.add_argument("--max_steps", type=int, default=0)
parser.add_argument("--record_forces", action="store_true",
                    help="Record every physics substep's foot forces and matching reference to foot_forces.npz")
parser.add_argument("--record_video_states", action="store_true",
                    help="Include complete named robot poses for offline rendering of the force recording")
parser.add_argument("--record_perception", action="store_true",
                    help="Record exact policy vision and full poses from frame zero; stop after the first native episode")
parser.add_argument("--spawn_offset_x", type=float, default=None,
                    help="Override the local patch X offset for matched force-comparison trials (m)")
parser.add_argument("--seed", type=int, default=1017)
parser.add_argument("--paused", action="store_true")
parser.add_argument("--no_alternate", action="store_true")
parser.add_argument("--hide_reference", action="store_true", help="Start with the visual-only reference robot hidden")
parser.add_argument("--reference_side_by_side", action="store_true", default=True,
                    help="Display reference shifted 1 m in world Y (default)")
parser.add_argument("--reference_overlay", dest="reference_side_by_side", action="store_false",
                    help="Start with reference in the same world coordinates as the policy robot")
parser.add_argument("--log_dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.speed <= 0 or not math.isfinite(args.speed) or args.max_steps < 0 or args.max_episodes < 0:
    parser.error("Invalid speed/step/episode limit")
if args.spawn_offset_x is not None and (not math.isfinite(args.spawn_offset_x) or abs(args.spawn_offset_x) > 2.4):
    parser.error("spawn_offset_x must be finite and inside [-2.4, 2.4] m")
if args.record_forces and not (args.max_steps or args.max_episodes or args.record_perception):
    parser.error("Force recording requires a finite --max_steps or --max_episodes limit")
if args.record_video_states and not args.record_forces:
    parser.error("--record_video_states requires --record_forces")
output = Path(args.log_dir).resolve()
output.mkdir(parents=True, exist_ok=True)
status_path = output / "training_status.json"
if status_path.exists():
    parser.error(f"Refusing to overwrite {output}")
status = dict(status="launching", pid=os.getpid(), started_at=datetime.now().isoformat(),
              run_id=os.environ.get("PMT_PGMT_RUN_ID"),
              checkpoint=args.checkpoint, motion=args.motion, arguments=vars(args), training=False)


def save_status(**values):
    status.update(values, updated_at=datetime.now().isoformat())
    temporary = status_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(status_path)


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
    from pmt_tasks.pgmt_runtime import get_pgmt_state

    install_rigid_scene_schema_bridge()
    if not args.headless:
        # Sim 6's base experience opens a window without the legacy GUI flag
        # read by this Isaac Lab version. Set it before constructing the scene.
        import carb
        carb.settings.get_settings().set_bool("/app/window/enabled", True)
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = saved["policy_metadata"]
    assert metadata["policy_family"] == "pgmt_terrain_tracking"
    with np.load(args.motion) as motion:
        source_xy = motion["body_pos_w"][:, 0, :2].copy()
        q = motion["body_quat_w"][0, 0]
        source_yaw = math.atan2(float(2*(q[0]*q[3]+q[1]*q[2])), float(1-2*(q[2]**2+q[3]**2)))
    delta = source_xy[-1] - source_xy[0]
    motion_distance = float(np.linalg.norm(delta))
    # Rotate the whole reference into the test direction; retain its turns and
    # lateral motion. A stationary/closed-loop clip uses its initial facing.
    heading = math.atan2(float(delta[1]), float(delta[0])) if motion_distance > 1e-8 else source_yaw
    yaw = source_yaw - heading
    relative_xy = source_xy-source_xy[0]
    unit_direction = np.array([math.cos(heading), math.sin(heading)])
    sideways = float(np.abs(relative_xy[:, 0]*unit_direction[1]
                            - relative_xy[:, 1]*unit_direction[0]).max())

    cfg = build_env_cfg("pgmt_stage2_terrain")
    saved_environment = saved["training_contract"]["runtime"]["environment"]
    cfg.pgmt_terrain_asset = args.terrain_asset or saved_environment["pgmt_terrain_asset"]
    asset_spec = json.loads((Path(cfg.pgmt_terrain_asset) / "manifest.json").read_text())["spec"]
    asset_families = asset_spec["families"]
    patch_count = math.prod(asset_spec.get("patch_grid", (1, 1)))
    args.patch_id = (5 if patch_count > 5 else 0) if args.patch_id is None else args.patch_id
    if not 0 <= args.patch_id < patch_count:
        raise ValueError(f"This asset has patch indices 0..{patch_count-1}")
    if args.terrain == "stairs" and "stairs" not in asset_families:
        args.terrain = "stairs_down"
    if args.terrain not in asset_families:
        raise ValueError(f"Terrain {args.terrain!r} is not in this asset: {asset_families}")
    cfg.pmt_motion_paths = [str(Path(args.motion).resolve())]
    cfg.__post_init__()
    cfg.scene.num_envs, cfg.seed = 1, args.seed
    cfg.sim.device = cfg.commands.motion.storage_device = args.device
    cfg.sim.use_fabric = True
    cfg.sim.enable_scene_query_support = True
    cfg.viewer.origin_type = "world"
    cfg.curriculum.terrain_progress = None
    # Explicit playback selection must work on every selectable terrain,
    # including combinations excluded from the adaptive training sampler.
    # Keep this one-motion allow table local to the run; never edit the
    # training CSV or weaken the shared sampler's empty-pool validation.
    playback_compatibility = output / "playback_motion_compatibility.csv"
    with playback_compatibility.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["motion_name", "dismatched terrain"])
        writer.writerow([Path(args.motion).stem, ""])
    cfg.commands.motion.motion_terrain_compatibility = str(playback_compatibility)
    cfg.commands.motion.fixed_start_frame = 0
    cfg.commands.motion.require_valid_support_start = False
    save_status(motion_terrain_filter=False, playback_compatibility=str(playback_compatibility))
    print("[TERRAIN TEST] Using the selected motion on all terrains; training compatibility exclusions disabled.", flush=True)
    cfg.commands.motion.placement_mode = "origin"
    cfg.commands.motion.placement_patch_grid = None
    cfg.commands.motion.placement_xy_range = (0., 0.)
    cfg.commands.motion.placement_yaw_range = (yaw, yaw)
    cfg.pgmt_tracking_relaxation = deepcopy(saved["training_contract"]["runtime"]["environment"]["pgmt_tracking_relaxation"])
    cfg.pgmt_terrain_observation = deepcopy(metadata["policy_kwargs"]["terrain_observation_cfg"])
    cfg.pgmt_terrain_observation["augmentation_enabled"] = False
    cfg.observations.proprio.enable_corruption = False
    cfg.observations.proprio_history.enable_corruption = False
    for name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
        setattr(cfg.events, name, None)
    force_buffer_capacities = {}
    if args.record_forces:
        # The training capacities reserve memory for thousands of robots.
        # These single-robot diagnostic capacities change storage only; solver
        # iterations, materials, collision geometry and timestep stay identical.
        force_buffer_capacities = dict(gpu_max_rigid_contact_count=2**16,
            gpu_max_rigid_patch_count=2**14, gpu_found_lost_pairs_capacity=2**16,
            gpu_found_lost_aggregate_pairs_capacity=2**18,
            gpu_total_aggregate_pairs_capacity=2**16, gpu_collision_stack_size=2**26)
        for key, value in force_buffer_capacities.items():
            setattr(cfg.sim.physx, key, value)
    old = cfg.scene.robot.spawn
    shared = {f.name: deepcopy(getattr(old, f.name)) for f in fields(sim_utils.UsdFileCfg)
              if f.name not in ("func", "usd_path") and hasattr(old, f.name)}
    cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
        usd_path=str(ROOT / "logs/pgmt_p1_acceptance/native_robot/main.usd"), **shared)
    controls = dict(terrain=args.terrain, level=args.level, direction=args.direction, paused=args.paused, single=False,
                    reset=False, alternate=not args.no_alternate, speed=args.speed)
    context = dict(trial_step=0, active=False, episode=0)
    dump_yaml(str(output / "env.yaml"), cfg)
    # A single robot needs no cross-environment broadphase optimization. Isaac's
    # cloner does not create the env-id AABB attribute for this one-env scene.
    raw = ManagerBasedRLEnv(cfg)
    force_recorder = None
    perception_recorder = None
    try:
        terrain, robot = raw.scene.terrain, raw.scene["robot"]
        families = terrain.manifest["spec"]["families"]
        command = raw.command_manager.get_term("motion")
        state = get_pgmt_state(raw)
        original_reset = raw._reset_idx
        results = []
        viewer = None
        ghost = None
        route_curve = None
        latest = [None]

        def prepare_spawn():
            family, level = controls["terrain"], controls["level"]
            direction = controls["direction"] if family in DIRECTIONAL_TERRAINS else "center"
            patch = next(p for p in terrain.surface_records if p["family"] == family
                         and p["level"] == level and p["patch_id"] == args.patch_id)
            center = terrain.env_origins.new_tensor(patch["center"])
            start_at_rim = direction == ("up" if family == "stairs_up" else "down")
            offset = (-asset_spec["patch_size"][0] / 2 + .65) if start_at_rim else 0.0
            if args.spawn_offset_x is not None:
                offset = args.spawn_offset_x
            terrain.terrain_types.fill_(families.index(family))
            terrain.terrain_levels.fill_(level)
            terrain.env_origins[:] = center
            terrain.env_origins[:, 0] += offset
            xy = terrain.env_origins[:, :2]
            terrain.env_origins[:, 2] = terrain.height(xy)
            context.update(trial_step=0, active=False,
                terrain=family, direction=direction, level=level, patch=patch,
                spawn_xy=xy[0].cpu().tolist(), spawn_ground_z=float(terrain.height(xy)[0]))

        def reset_idx(ids):
            if context["active"] and not controls["reset"]:
                flags = {name: bool(raw.termination_manager.get_term(name)[0])
                         for name in raw.termination_manager.active_terms}
                if any(flags.values()):
                    root = robot.data.root_pos_w[0].detach().cpu().tolist()
                    result = dict(episode=context["episode"], terrain=context["terrain"],
                        direction=context["direction"], level=context["level"], patch_id=args.patch_id,
                        control_steps=context["trial_step"]+1, seconds=(context["trial_step"]+1)*raw.step_dt,
                        termination_flags=flags, failed=bool(raw.termination_manager.terminated[0]),
                        motion_finished=flags.get("motion_finished", False),
                        final_reference_frame=int(command.frame_ids[0]), root_position=root,
                        spawn_xy=context["spawn_xy"],
                        displacement_x_m=root[0]-context["spawn_xy"][0],
                        spawn_ground_z=context["spawn_ground_z"],
                        final_ground_z=float(terrain.height(robot.data.root_pos_w[:, :2])[0]))
                    results.append(result)
                    with (output / "episodes.jsonl").open("a") as stream:
                        stream.write(json.dumps(result)+"\n")
                    print("[TERRAIN TEST] " + json.dumps(result), flush=True)
                    if viewer is not None:
                        ghost.update(state.reference())
                        viewer.follow()
                        for _ in range(3): raw.sim.render()
                        viewer.capture(f"episode_{context['episode']:03d}_{context['terrain']}_{context['direction']}_terminal.png")
                        for _ in range(3): raw.sim.render()
                    if controls["alternate"] and context["terrain"] in DIRECTIONAL_TERRAINS:
                        controls["direction"] = "down" if context["direction"] == "up" else "up"
            prepare_spawn()
            original_reset(ids)
            context["episode"] += 1
            context.update(active=True, new_spawn=True)

        raw._reset_idx = reset_idx
        env = RslRlVecEnvWrapper(raw)
        obs = env.get_observations()
        options = deepcopy(metadata["policy_kwargs"])
        policy = PGMTActorCritic(obs, metadata["obs_groups"], env.num_actions, **options).to(args.device)
        actual = policy.get_checkpoint_metadata()
        for key in ("signature", "obs_schema", "action_contract", "value_groups", "perception_contract"):
            assert actual[key] == metadata[key], f"Policy metadata mismatch: {key}"
        policy.load_state_dict(saved["model_state_dict"], strict=True)
        for key, value in policy.state_dict().items():
            assert torch.equal(value.cpu(), saved["model_state_dict"][key]), key
        policy.eval().requires_grad_(False)
        initial_state = {k:v.clone() for k,v in policy.state_dict().items()}
        latest[0] = obs
        panel = label = None
        if not args.headless:
            from isaacsim.core.simulation_manager import SimulationManager
            from pmt_tasks.pgmt_perception_viewer import PGMTPerceptionViewer
            from pmt_tasks.pgmt_reference_ghost import PGMTReferenceGhost
            from pxr import UsdGeom, Vt
            import omni.ui as ui
            SimulationManager.enable_fabric(True)
            viewer = PGMTPerceptionViewer(raw, policy, output)
            if not args.record_perception:
                viewer.toggle()
            viewer.panel.visible = args.record_perception
            ghost = PGMTReferenceGhost(raw.sim.stage, cfg.scene.robot.spawn.usd_path,
                                       robot.body_names, visible=not args.hide_reference)
            ghost.set_side_by_side(args.reference_side_by_side)
            route_curve = UsdGeom.BasisCurves.Define(raw.sim.stage, "/World/PGMTStairsTest/Route")
            route_curve.CreateTypeAttr("linear")
            route_curve.CreateWidthsAttr([.018])
            route_curve.SetWidthsInterpolation("constant")
            route_curve.CreateDisplayColorAttr([(.05,.8,.55)])
            def select(**values):
                controls.update(values, reset=True)
            panel = ui.Window("PGMT Stage2 | Terrain policy test", width=700, height=520)
            panel.position_x, panel.position_y = 20, 60
            with panel.frame, ui.VStack(spacing=5):
                ui.Label(f"{checkpoint.name} | deterministic inference | NO TRAINING", height=25)
                ui.Label(Path(args.motion).name, height=22)
                with ui.HStack(height=30):
                    for family in families:
                        ui.Button(family.upper(), clicked_fn=lambda f=family: select(terrain=f))
                with ui.HStack(height=30):
                    up_button = ui.Button("UP / ascent", clicked_fn=lambda: select(direction="up"))
                    down_button = ui.Button("DOWN / descent", clicked_fn=lambda: select(direction="down"))
                    ui.Button("Reset", clicked_fn=lambda: controls.update(reset=True))
                with ui.HStack(height=30):
                    for level in range(10):
                        ui.Button(f"L{level}", clicked_fn=lambda l=level: select(level=l))
                with ui.HStack(height=30):
                    ui.Button("Pause / Run", clicked_fn=lambda: controls.update(paused=not controls["paused"]))
                    ui.Button("Single step", clicked_fn=lambda: controls.update(paused=True,single=True))
                    alternate_button = ui.Button("Alternate up/down", clicked_fn=lambda: controls.update(alternate=not controls["alternate"]))
                with ui.HStack(height=30):
                    for speed in (.25,.5,1.,2.):
                        ui.Button(f"{speed:g}x", clicked_fn=lambda s=speed: controls.update(speed=s))
                    ui.Button("Perception overlay", clicked_fn=viewer.toggle)
                with ui.HStack(height=30):
                    ui.Button("Reference on / off", clicked_fn=ghost.toggle)
                    ui.Button("Reference overlay", clicked_fn=lambda: ghost.set_side_by_side(False))
                    ui.Button("Reference side by side", clicked_fn=lambda: ghost.set_side_by_side(True))
                label = ui.Label("Preparing inference", word_wrap=True)

        def refresh_spawn():
            if not context.pop("new_spawn", False): return
            if route_curve is not None:
                dxy = source_xy-source_xy[0]
                rot = np.array([[math.cos(-heading),-math.sin(-heading)],
                                [math.sin(-heading),math.cos(-heading)]])
                xy = torch.tensor(dxy@rot.T+context["spawn_xy"],dtype=torch.float32,device=raw.device)
                z = terrain.height(xy)+.025
                route_curve.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(torch.cat([xy,z[:,None]],-1).cpu().numpy()))
                route_curve.CreateCurveVertexCountsAttr([len(xy)])
            if viewer is not None:
                ghost.update(state.reference())
                context["reference_display_audit"] = ghost.audit()
                viewer.follow()
                # Switching while paused must also move any visible scan and
                # glimpse overlay; do not wait for the next policy step.
                viewer.update(latest[0])
                for _ in range(5): raw.sim.render()
                viewer.capture(f"episode_{context['episode']:03d}_{context['terrain']}_{context['direction']}_start.png")
                for _ in range(3): raw.sim.render()
            spawn = dict(episode=context["episode"], terrain=context["terrain"],
                terrain_type=int(terrain.terrain_types[0]), level=context["level"], patch_id=args.patch_id,
                direction=context["direction"], spawn_xy=context["spawn_xy"],
                spawn_ground_z=context["spawn_ground_z"], frame=int(command.frame_ids[0]),
                root=robot.data.root_pos_w[0].cpu().tolist(),
                reference_root=state.reference()["body_pos"][0, 0].cpu().tolist())
            with (output / "spawns.jsonl").open("a") as stream:
                stream.write(json.dumps(spawn) + "\n")
            save_status(terrain=context["terrain"], level=context["level"], direction=context["direction"],
                        patch_id=args.patch_id, last_spawn=spawn,
                        reference_display=context.get("reference_display_audit"))

        refresh_spawn()
        if args.record_forces or args.record_perception:
            from pgmt_force_recording import FootForceRecorder
            force_recorder = FootForceRecorder(raw, state, output,
                                               full_state=args.record_video_states or args.record_perception)
        if args.record_perception:
            from pgmt_perception_recording import PerceptionRecorder
            assert int(command.frame_ids[0]) == 0
            perception_recorder = PerceptionRecorder(raw, state, policy, output)
            save_status(perception_recording=dict(file="perception.npz", initial_frame=0,
                sample_hz=1/raw.step_dt, source="exact pre-action observation passed to policy; no resampling",
                terminal="terminal pose with last pre-action input held for one control step",
                scan_shape=[21,21], scan_size_m=[2,2], glimpse_count=4, glimpse_size_m=.4,
                terrain_mode=policy.terrain_mode, stop_after_first_native_episode=True))
        save_status(status="running", checkpoint_loaded_strictly=True, source_sha256=checkpoint_hash,
            policy_family=metadata["policy_family"], loaded_iteration=saved["next_iteration"],
            simulator="Isaac Sim 6.0.1 (GUI compatibility entry)",
            motion_distance_m=motion_distance, motion_lateral_deviation_m=sideways,
            motion_frames=len(source_xy), motion_geometry_filter=False,
            control_dt=raw.step_dt, physics_dt=cfg.sim.dt, observation_noise=False,
            random_pushes=False, random_dynamics=False, curriculum_updates=False,
            action_mode="actor mean", test_endpoint="native motion exhaustion, timeout, or tracking/map termination",
            termination_terms=list(raw.termination_manager.active_terms),
            reference_display=context.get("reference_display_audit"))
        if force_recorder is not None:
            save_status(force_recording=dict(schema_version=1, file="foot_forces.npz",
                force_semantics="normal_only_world_vectors_without_friction",
                sampling="every_physics_substep_after_scene_update_before_reset",
                reference_sampling="held_control_command_before_step",
                foot_names=list(state.foot_names), decimation=cfg.decimation,
                motion_sha256=hashlib.sha256(Path(args.motion).read_bytes()).hexdigest(),
                terrain_manifest_sha256=hashlib.sha256(
                    (Path(cfg.pgmt_terrain_asset) / "manifest.json").read_bytes()).hexdigest(),
                spawn_offset_x=args.spawn_offset_x, seed=cfg.seed,
                physics_buffer_capacities=force_buffer_capacities,
                terrain_parameters=context["patch"]["parameters"],
                contact_config=cfg.pgmt_contact_events))
            if args.record_video_states or args.record_perception:
                save_status(video_states=dict(joint_names=list(robot.joint_names), body_names=list(robot.body_names),
                    terrain_asset=cfg.pgmt_terrain_asset, physics="Isaac / PhysX", rendering_only_replay=True))
        steps, next_step = 0, time.monotonic()
        trajectory = (output / "trajectory.jsonl").open("w", buffering=1)
        try:
            while app.is_running() and not stop["requested"]:
                if args.max_steps and steps >= args.max_steps: break
                if args.max_episodes and len(results) >= args.max_episodes: break
                if controls["reset"]:
                    # Command metrics can be inference tensors from env.step.
                    # Manual resets must use the same mode as automatic resets.
                    with torch.inference_mode():
                        obs, _ = env.reset()
                    controls["reset"] = False
                    latest[0] = obs
                    refresh_spawn()
                now = time.monotonic()
                advance = controls["single"] or (not controls["paused"] and (args.headless or now >= next_step))
                if advance:
                    controls["single"] = False
                    old_episode = context["episode"]
                    with torch.inference_mode():
                        if perception_recorder is not None:
                            perception_recorder.capture(obs)
                        if force_recorder is not None:
                            force_recorder.begin(context)
                        action = policy.act_inference(obs)
                        assert torch.isfinite(action).all()
                        obs, reward, done, extras = env.step(action)
                        if force_recorder is not None:
                            force_recorder.finish(bool(done[0]))
                        if perception_recorder is not None and bool(done[0]):
                            perception_recorder.close(force_recorder.rows[-1])
                            perception_recorder = None
                    latest[0] = obs
                    steps += 1
                    if context["episode"] == old_episode: context["trial_step"] += 1
                    reference = state.reference()
                    if ghost is not None:
                        ghost.update(reference)
                    refresh_spawn()
                    root = robot.data.root_pos_w[0].cpu().tolist()
                    ref = reference["body_pos"][0,0].cpu().tolist()
                    trajectory.write(json.dumps(dict(step=steps,episode=context["episode"],terrain=context["terrain"],
                        direction=context["direction"],level=context["level"],patch_id=args.patch_id,
                        frame=int(command.frame_ids[0]),root=root,reference_root=ref,
                        done_previous_episode=bool(done[0])))+"\n")
                    if viewer is not None:
                        viewer.update(obs)
                        if context["trial_step"] in (100,200,300):
                            viewer.capture(f"episode_{context['episode']:03d}_{context['terrain']}_{context['direction']}_step{context['trial_step']}.png")
                    if steps % 25 == 0 or bool(done[0]):
                        save_status(status="running", steps=steps, completed_episodes=len(results),
                            terrain=context["terrain"], direction=context["direction"], level=context["level"], root=root,
                            trial_step=context["trial_step"], last_episode=results[-1] if results else None,
                            reference_display=ghost.audit() if ghost is not None else None)
                    next_step = time.monotonic() + raw.step_dt / controls["speed"]
                    if args.record_perception and bool(done[0]):
                        break
                if label is not None:
                    p=context["patch"]["parameters"]
                    geometry=context["patch"]["geometry"]
                    family=context["terrain"]
                    for button in (up_button, down_button, alternate_button):
                        button.enabled = family in DIRECTIONAL_TERRAINS
                    if "source_config" in p:
                        terrain_detail = f"P-BFM {context['patch']['source_family']} / d={context['patch']['difficulty']:.4f}"
                    elif "stair_up_height_m" in p:
                        terrain_detail = f"{family} / d={context['patch']['difficulty']:.4f}"
                    elif family == "stairs":
                        terrain_detail = f"Step height {p['stair_height_m']:.3f} m / depth {p['stair_depth_m']:.2f} m"
                    elif family == "slope":
                        terrain_detail = f"Slope {p['slope_angle_deg']:.2f} deg / bottom width {p['platform_width_m']:.2f} m"
                    elif family == "boxes":
                        terrain_detail = f"Box height amplitude +/-{p['box_height_amplitude_m']:.3f} m"
                    elif family == "rough":
                        terrain_detail = f"Measured rough amplitude {geometry.get('measured_amplitude_m', 0):.3f} m"
                    else:
                        terrain_detail = "Flat surface"
                    alternate_text = str(controls['alternate']) if family in DIRECTIONAL_TERRAINS else "N/A (center start)"
                    recent="none" if not results else f"{results[-1]['terrain']}/{results[-1]['direction']}: " + (
                        "tracking failure" if results[-1]['failed'] else "motion finished" if results[-1]['motion_finished'] else "other end")
                    label.text=(f"{'PAUSED' if controls['paused'] else 'RUNNING'} | {family.upper()} | {context['direction'].upper()} | L{context['level']}\n"
                        f"{terrain_detail} | patch {args.patch_id}\n"
                        f"Frame {int(command.frame_ids[0])}/{len(source_xy)-1} | episode {context['episode']} | {controls['speed']:g}x\n"
                        f"Alternate: {alternate_text} | Completed: {len(results)} | Last: {recent}\n"
                        f"Cyan reference: {'ON' if ghost.visible else 'OFF'} | "
                        f"{'display Y +1 m' if ghost.side_by_side else 'same world coordinates'} | NO PHYSICS\n"
                        "Reference keeps its original height after reset placement; it can pass through terrain.\n"
                        "Green line: reference XY on the terrain.\nClean observations; fixed dynamics; trained termination thresholds.\n"
                        "Full reference motion; native episode termination conditions.")
                if viewer is not None: raw.sim.render()
                if not advance: time.sleep(.005)
        finally:
            trajectory.close()
        assert all(torch.equal(v, initial_state[k]) for k,v in policy.state_dict().items())
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_hash
        save_status(status="completed", steps=steps, completed_episodes=len(results),
                    policy_state_unchanged=True, source_checkpoint_unchanged=True,
                    reference_display=ghost.audit() if ghost is not None else None)
    finally:
        try:
            if perception_recorder is not None and force_recorder is not None and force_recorder.rows:
                perception_recorder.close(force_recorder.rows[-1])
            if force_recorder is not None:
                force_recorder.close()
        finally:
            raw.close()


def checked_main():
    try:
        main()
    except BaseException as error:
        save_status(status="failed", error=repr(error))
        raise


from pmt_tasks.runtime_options import run_and_close
run_and_close(checked_main, app)
