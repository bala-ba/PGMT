"""Visualize the original blind, three-head Stage1 policy on its native plane."""
import argparse
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
sys.path.insert(0, str(ROOT))
if os.environ.get("PMT_PGMT_PHYSICS_WORKER") != "1":
    from pgmt_process_guard import run_guarded
    raise SystemExit(run_guarded([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], ROOT))

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", default=str(ROOT / "logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt"))
parser.add_argument("--motion", default=str(ROOT / "assets/motions/pgmt_clipped_50hz_tcrs_v1/walk1_subject1_clip1.npz"))
parser.add_argument("--speed", type=float, default=1.)
parser.add_argument("--max_steps", type=int, default=0, help="0: keep the viewer open")
parser.add_argument("--paused", action="store_true")
parser.add_argument("--hide_reference", action="store_true")
parser.add_argument("--reference_overlay", action="store_true", help="No display offset for the reference")
parser.add_argument("--record_video_states", action="store_true",
                    help="Record full actual/reference poses and stop after the first native clip ending")
parser.add_argument("--log_dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.speed <= 0 or not math.isfinite(args.speed) or args.max_steps < 0:
    parser.error("Invalid speed/step limit")
output = Path(args.log_dir).resolve()
output.mkdir(parents=True, exist_ok=True)
status_path = output / "training_status.json"
if status_path.exists():
    parser.error(f"Refusing to overwrite {output}")
status = dict(status="launching", pid=os.getpid(), run_id=os.environ.get("PMT_PGMT_RUN_ID"),
              started_at=datetime.now().isoformat(), training=False, arguments=vars(args))


def save_status(**values):
    status.update(values, updated_at=datetime.now().isoformat())
    temporary = status_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(status_path)


save_status()
stop = {"requested": False}
from pmt_tasks.isaacsim6_legacy import install_tensor_import_bridge
install_tensor_import_bridge()
app = AppLauncher(args, multi_gpu=False, fast_shutdown=True).app
# Kit installs signal handlers while starting; install ours afterwards so a
# terminal stop can finish policy/checkpoint checks before closing the app.
signal.signal(signal.SIGINT, lambda *_: stop.update(requested=True))
signal.signal(signal.SIGTERM, lambda *_: stop.update(requested=True))


def main():
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.terrains import TerrainImporter
    from isaaclab.utils.io import dump_yaml
    from motion_tracking_rl.networks.pgmt_actor_critic import PGMTActorCritic
    from pmt_tasks.builder import build_env_cfg
    from pmt_tasks.isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from pmt_tasks.isaacsim6_legacy import install_rigid_scene_schema_bridge
    from pmt_tasks.pgmt_runtime import get_pgmt_state

    install_rigid_scene_schema_bridge()
    if not args.headless:
        import carb
        carb.settings.get_settings().set_bool("/app/window/enabled", True)
    checkpoint = Path(args.checkpoint).resolve()
    source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = saved["policy_metadata"]
    assert metadata["policy_family"] == "pgmt_stage1", "This viewer requires a blind Stage1 checkpoint"
    motion = str(Path(args.motion).resolve())
    cfg = build_env_cfg("pgmt_stage1", overrides=[f"motion.motion_files={json.dumps(motion)}"])
    class LocalPlaneTerrainImporter(TerrainImporter):
        """Native PhysX plane without the unavailable default Nucleus USD."""

        def import_ground_plane(self, name, size=(2.0e6, 2.0e6)):
            from isaacsim.core.utils.stage import get_current_stage
            from pxr import PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics
            stage = get_current_stage()
            path = self.cfg.prim_path + "/" + name
            if path in self.terrain_prim_paths:
                raise ValueError(f"Terrain already exists: {path}")
            self.terrain_prim_paths.append(path)
            # This is the same native helper used by Isaac's GroundPlane class:
            # infinite plane collision, plus a finite mesh used only for display.
            PhysicsSchemaTools.addGroundPlane(stage, path, "Z", min(*size, 100.) / 2.,
                                              (0., 0., 0.), (.55, .58, .62))
            material = self.cfg.physics_material
            material_path = path + "/physicsMaterial"
            material.func(material_path, material)
            colliders = [p for p in Usd.PrimRange(stage.GetPrimAtPath(path)) if p.HasAPI(UsdPhysics.CollisionAPI)]
            assert len(colliders) == 1 and colliders[0].GetTypeName() == "Plane"
            sim_utils.bind_physics_material(str(colliders[0].GetPath()), material_path)
            # Render only the mesh: modern Hydra can otherwise show the
            # coincident collision Plane as well. Its collision stays enabled.
            UsdGeom.Imageable(colliders[0]).MakeInvisible()
            grid = UsdGeom.BasisCurves.Define(stage, path + "/DisplayGrid")
            points = []
            for i in range(-20, 21):
                points.extend([(-20., float(i), .003), (20., float(i), .003),
                               (float(i), -20., .003), (float(i), 20., .003)])
            grid.CreateTypeAttr("linear")
            grid.CreatePointsAttr(points)
            grid.CreateCurveVertexCountsAttr([2] * (len(points) // 2))
            grid.CreateWidthsAttr([.008])
            grid.SetWidthsInterpolation("constant")
            grid.CreateDisplayColorAttr([(.2, .23, .26)])

    cfg.scene.terrain.class_type = LocalPlaneTerrainImporter
    cfg.scene.num_envs, cfg.seed = 1, 1017
    cfg.sim.device = cfg.commands.motion.storage_device = args.device
    cfg.sim.use_fabric = True
    cfg.viewer.origin_type = "world"
    cfg.commands.motion.force_start_frame_zero = True
    cfg.commands.motion.default_reset_joint_prob = 0.
    cfg.commands.motion.pose_range = {}
    cfg.commands.motion.velocity_range = {}
    cfg.commands.motion.joint_position_range = (0., 0.)
    cfg.commands.motion.debug_vis = False
    cfg.scene.contact_forces.debug_vis = False
    cfg.observations.proprio.enable_corruption = False
    cfg.observations.proprio_history.enable_corruption = False
    for name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
        setattr(cfg.events, name, None)
    old = cfg.scene.robot.spawn
    shared = {f.name: deepcopy(getattr(old, f.name)) for f in fields(sim_utils.UsdFileCfg)
              if f.name not in ("func", "usd_path") and hasattr(old, f.name)}
    cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
        usd_path=str(ROOT / "logs/pgmt_p1_acceptance/native_robot/main.usd"), **shared)
    dump_yaml(str(output / "env.yaml"), cfg)
    controls = dict(paused=args.paused, single=False, reset=False, follow=True, speed=args.speed)
    context = dict(active=False, sequence=0, trial_step=0)
    raw = ManagerBasedRLEnv(cfg)
    try:
        robot = raw.scene["robot"]
        command = raw.command_manager.get_term("motion")
        state = get_pgmt_state(raw)
        original_resample = command._resample_command
        endings = []
        video_rows = []
        video_finished = False

        def capture_pose(stamp, terminal=False):
            from pmt_tasks.pgmt_runtime import _tensor
            reference = state.reference()
            frame = min(int(command.frame_ids[0]), int(command.data_store.motion_lengths[0])-1)
            row = dict(time_s=stamp, frame=frame, terminal=terminal)
            for name in ('joint_pos', 'joint_vel', 'body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w'):
                row[name] = _tensor(getattr(robot.data, name))[0].clone()
            for name, key in [('joint_pos','joint_pos'), ('body_pos_w','body_pos'), ('body_quat_w','body_quat')]:
                row['reference_'+name] = reference[key][0].clone()
            video_rows.append(row)

        def resample(ids):
            nonlocal video_finished
            if context["active"] and not controls["reset"]:
                flags = {name: bool(raw.termination_manager.get_term(name)[0])
                         for name in raw.termination_manager.active_terms}
                finished = int(command.frame_ids[0]) >= int(command.data_store.motion_lengths[command.motion_ids[0]])
                result = dict(sequence=context["sequence"], steps=context["trial_step"] + 1,
                    reference_finished=finished, termination_flags=flags,
                    failed=bool(raw.termination_manager.terminated[0]),
                    root=robot.data.root_pos_w[0].cpu().tolist())
                endings.append(result)
                if args.record_video_states and video_rows and not video_finished:
                    # Save the old physical pose before the native command resets it.
                    capture_pose((context['trial_step']+1)*raw.step_dt, terminal=True)
                    video_finished = True
                with (output / "endings.jsonl").open("a") as stream:
                    stream.write(json.dumps(result) + "\n")
                print("[STAGE1 FLAT] " + json.dumps(result), flush=True)
            original_resample(ids)
            context.update(active=True, sequence=context["sequence"] + 1, trial_step=0)

        command._resample_command = resample
        env = RslRlVecEnvWrapper(raw)
        obs = env.get_observations()
        policy = PGMTActorCritic(obs, metadata["obs_groups"], env.num_actions,
                                **deepcopy(metadata["policy_kwargs"])).to(args.device)
        actual = policy.get_checkpoint_metadata()
        for key in ("policy_family", "signature", "obs_schema", "action_contract", "value_groups"):
            assert actual[key] == metadata[key], f"Policy metadata mismatch: {key}"
        policy.load_state_dict(saved["model_state_dict"], strict=True)
        assert all(torch.equal(v.cpu(), saved["model_state_dict"][k]) for k, v in policy.state_dict().items())
        policy.eval().requires_grad_(False)
        initial_state = {k: v.clone() for k, v in policy.state_dict().items()}
        ghost = viewport = panel = label = None
        if not args.headless:
            import omni.ui as ui
            from isaacsim.core.simulation_manager import SimulationManager
            from isaacsim.core.utils.viewports import set_camera_view
            from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
            from pmt_tasks.pgmt_reference_ghost import PGMTReferenceGhost
            SimulationManager.enable_fabric(True)
            viewport = get_active_viewport()
            viewport.set_active_camera(cfg.viewer.cam_prim_path)
            ghost = PGMTReferenceGhost(raw.sim.stage, cfg.scene.robot.spawn.usd_path,
                robot.body_names, visible=not args.hide_reference, prim_path="/World/PGMTStage1Reference")
            ghost.set_side_by_side(not args.reference_overlay)
            panel = ui.Window("PGMT Stage1 | Blind policy on flat ground", width=650, height=360)
            panel.position_x, panel.position_y = 20, 60
            with panel.frame, ui.VStack(spacing=5):
                ui.Label(f"{checkpoint.name} | Stage1 | NO TRAINING", height=25)
                ui.Label(Path(motion).name, height=22)
                with ui.HStack(height=30):
                    ui.Button("Pause / Run", clicked_fn=lambda: controls.update(paused=not controls["paused"]))
                    ui.Button("Single step", clicked_fn=lambda: controls.update(paused=True, single=True))
                    ui.Button("Reset to frame 0", clicked_fn=lambda: controls.update(reset=True))
                with ui.HStack(height=30):
                    for speed in (.25, .5, 1., 2.):
                        ui.Button(f"{speed:g}x", clicked_fn=lambda s=speed: controls.update(speed=s))
                    ui.Button("Follow / free camera", clicked_fn=lambda: controls.update(follow=not controls["follow"]))
                with ui.HStack(height=30):
                    ui.Button("Reference on / off", clicked_fn=ghost.toggle)
                    ui.Button("Reference overlay", clicked_fn=lambda: ghost.set_side_by_side(False))
                    ui.Button("Reference side by side", clicked_fn=lambda: ghost.set_side_by_side(True))
                label = ui.Label("Preparing Stage1 policy", word_wrap=True)

        def refresh_display():
            if ghost is None:
                return
            ghost.update(state.reference())
            if controls["follow"]:
                root = robot.data.root_pos_w[0].cpu().tolist()
                set_camera_view(eye=[v + d for v, d in zip(root, (3., 3., 2.))], target=root,
                                camera_prim_path=str(viewport.camera_path))

        refresh_display()
        if viewport is not None:
            for _ in range(5): raw.sim.render()
            capture_viewport_to_file(viewport, str(output / "start.png"))
            for _ in range(3): raw.sim.render()
        save_status(status="running", checkpoint=str(checkpoint), checkpoint_loaded_strictly=True,
            source_sha256=source_hash, loaded_iteration=saved["iter"], motion=motion,
            policy_family=metadata["policy_family"], value_groups=metadata["value_groups"],
            observation_shapes={k: list(v.shape) for k, v in obs.items()},
            terrain="native plane", terrain_perception=False, action_mode="actor mean",
            simulator="Isaac Sim 6.0.1 (GUI compatibility entry)", control_dt=raw.step_dt,
            physics_dt=cfg.sim.dt, fixed_start_frame=0, observation_noise=False,
            random_pushes=False, random_dynamics=False, reset_noise=False,
            native_stage1_termination=True, termination_terms=list(raw.termination_manager.active_terms),
            episode_length_s=cfg.episode_length_s, clip_end="native command resampling, no forced done",
            reference_display=ghost.audit() if ghost is not None else None)
        if args.record_video_states:
            assert int(command.frame_ids[0]) == 0
            with torch.inference_mode():
                capture_pose(0.)
            save_status(video_states=dict(file='video_states.npz', body_names=list(robot.body_names),
                joint_names=list(robot.joint_names), source_motion_sha256=hashlib.sha256(Path(motion).read_bytes()).hexdigest(),
                env_origin=raw.scene.env_origins[0].cpu().tolist(), initial_frame=0,
                sampling='control states plus native terminal pose before reset',
                stop_after_first_native_clip=True))
        steps, next_step = 0, time.monotonic()
        max_xy_error = 0.
        with (output / "trajectory.jsonl").open("w", buffering=1) as trajectory:
            while app.is_running() and not stop["requested"]:
                if args.max_steps and steps >= args.max_steps:
                    break
                if controls["reset"]:
                    with torch.inference_mode():
                        obs, _ = env.reset()
                    controls["reset"] = False
                    refresh_display()
                advance = controls["single"] or (not controls["paused"] and
                    (args.headless or time.monotonic() >= next_step))
                if advance:
                    controls["single"] = False
                    previous_sequence = context["sequence"]
                    with torch.inference_mode():
                        action = policy.act_inference(obs)
                        assert torch.isfinite(action).all()
                        obs, reward, done, extras = env.step(action)
                        assert torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values())
                    steps += 1
                    if context["sequence"] == previous_sequence:
                        context["trial_step"] += 1
                    if args.record_video_states and not video_finished:
                        with torch.inference_mode():
                            capture_pose(steps*raw.step_dt)
                    reference = state.reference()["body_pos"][0, 0]
                    root = robot.data.root_pos_w[0]
                    xy_error = float(torch.linalg.vector_norm(root[:2] - reference[:2]))
                    max_xy_error = max(max_xy_error, xy_error)
                    trajectory.write(json.dumps(dict(step=steps, sequence=context["sequence"],
                        frame=int(command.frame_ids[0]), root=root.cpu().tolist(),
                        reference_root=reference.cpu().tolist(), xy_error_m=xy_error, done=bool(done[0]))) + "\n")
                    refresh_display()
                    if viewport is not None and steps in (100, 250, 498, 750):
                        capture_viewport_to_file(viewport, str(output / f"step_{steps}.png"))
                    if steps % 25 == 0 or bool(done[0]) or previous_sequence != context["sequence"]:
                        save_status(steps=steps, frame=int(command.frame_ids[0]), sequence=context["sequence"],
                            xy_error_m=xy_error, max_xy_error_m=max_xy_error, endings=len(endings),
                            last_ending=endings[-1] if endings else None,
                            reference_display=ghost.audit() if ghost is not None else None)
                    next_step = time.monotonic() + raw.step_dt / controls["speed"]
                    if args.record_video_states and video_finished:
                        break
                if label is not None:
                    label.text = (f"{'PAUSED' if controls['paused'] else 'RUNNING'} | native flat plane | blind policy\n"
                        f"Frame {int(command.frame_ids[0])}/{int(command.data_store.motion_lengths[0])-1} | "
                        f"sequence {context['sequence']} | {controls['speed']:g}x\n"
                        "White: trained Stage1 policy + physics\nCyan: current reference only, no physics\n"
                        f"Reference: {'ON' if ghost.visible else 'OFF'} | "
                        f"{'display Y +1 m' if ghost.side_by_side else 'same world coordinates'}\n"
                        "Clean observations; fixed dynamics; original Stage1 termination.\n"
                        "Full reference motion; native clip looping.")
                if viewport is not None:
                    raw.sim.render()
                if not advance:
                    time.sleep(.005)
        assert all(torch.equal(v, initial_state[k]) for k, v in policy.state_dict().items())
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == source_hash
        if args.record_video_states:
            if not video_finished:
                raise RuntimeError('Recording interrupted before a native clip ending')
            arrays = {key: torch.stack([r[key] for r in video_rows]).cpu().numpy()
                      if isinstance(video_rows[0][key], torch.Tensor) else np.asarray([r[key] for r in video_rows])
                      for key in video_rows[0]}
            np.savez_compressed(output/'video_states.npz', **arrays)
            save_status(recorded_samples=len(video_rows), recorded_seconds=float(arrays['time_s'][-1]),
                        last_ending=endings[-1], endings=len(endings))
        save_status(status="completed", steps=steps, policy_state_unchanged=True, source_checkpoint_unchanged=True)
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
