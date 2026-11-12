"""Inspect real PhysX foot support/events against same-frame offline labels.

Uses Stage2's unchanged contact sensors/state with the exact Stage1 policy.
Only this viewer fixes a selected terrain placement and disables observation/randomization
noise. The reference skeleton and colored markers have no collision APIs.
"""
import argparse
from collections import deque
from copy import deepcopy
import csv
from dataclasses import fields
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.environ.get("PMT_PGMT_PHYSICS_WORKER") != "1":
    from pgmt_process_guard import run_guarded
    raise SystemExit(run_guarded([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], ROOT))

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--motion", default=str(ROOT / "assets/motions/pgmt_clipped_50hz_tcrs_v1/walk1_subject1_clip0.npz"))
parser.add_argument("--checkpoint", default=str(ROOT / "logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt"))
parser.add_argument("--log_dir", required=True)
parser.add_argument("--speed", type=float, default=.5, help="Wall playback speed; physics dt remains 0.02 s")
parser.add_argument("--max_steps", type=int, default=0, help="0 keeps the viewer running")
parser.add_argument("--paused", action="store_true")
parser.add_argument("--touchdown_points", action="store_true", help="Display the 3x3 clean terrain query per foot")
parser.add_argument("--pause_on_touchdown", action="store_true", help="Pause once at the first actual landing")
parser.add_argument("--terrain", choices=("flat", "stairs", "slope", "boxes", "rough"), default="flat")
parser.add_argument("--level", type=int, choices=range(10), default=0)
parser.add_argument("--patch_id", type=int, choices=range(16), help="Select a native 6x6 m patch; otherwise use region origin")
parser.add_argument("--spawn_xy", type=float, nargs=2, default=(0., 0.), metavar=("X", "Y"),
                    help="Fixed XY offset from selected patch center / region origin (m)")
parser.add_argument("--yaw_deg", type=float, default=0., help="Fixed initial root yaw (degrees)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not math.isfinite(args.speed) or args.speed <= 0 or args.max_steps < 0:
    parser.error("Require positive finite speed and nonnegative max_steps")
if not all(math.isfinite(v) for v in (*args.spawn_xy, args.yaw_deg)):
    parser.error("Spawn offset and yaw must be finite")
output = Path(args.log_dir).resolve()
output.mkdir(parents=True, exist_ok=True)
status_path = output / "training_status.json"
if status_path.exists():
    parser.error(f"Refusing to overwrite {output}")
status = dict(pid=os.getpid(), output=str(output), motion=str(Path(args.motion).resolve()),
              checkpoint=str(Path(args.checkpoint).resolve()), started_at=datetime.now().isoformat(),
              physics="Isaac / PhysX", training=False, headless=args.headless)


def save_status(**values):
    status.update(values, updated_at=datetime.now().isoformat())
    temp = status_path.with_suffix(".tmp")
    temp.write_text(json.dumps(status, indent=2) + "\n")
    temp.replace(status_path)


save_status(status="launching")
from pmt_tasks.isaacsim6_legacy import install_tensor_import_bridge
install_tensor_import_bridge()
app = AppLauncher(args, multi_gpu=False, fast_shutdown=True).app


def main():
    import numpy as np
    import torch
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.io import dump_yaml
    from pmt_tasks.isaacsim6_legacy import install_rigid_scene_schema_bridge
    from pmt_tasks.env_cfgs.pgmt.stage2 import PGMTPerceptionEnvCfg
    from pmt_tasks.env_cfgs.pgmt.stage1 import PGMTObservationsCfg
    from pmt_tasks.agent_cfgs.pgmt import PGMTStage1PPORunnerCfg
    from pmt_tasks.isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from pmt_tasks.pgmt_runtime import get_pgmt_state
    from motion_tracking_rl.networks.pgmt_actor_critic import PGMTActorCritic

    install_rigid_scene_schema_bridge()
    cfg = PGMTPerceptionEnvCfg()
    cfg.pmt_motion_paths = [str(Path(args.motion).resolve())]
    cfg.__post_init__()
    cfg.scene.num_envs = 1
    cfg.seed = 1014
    cfg.sim.device = cfg.commands.motion.storage_device = args.device
    cfg.sim.use_fabric = True
    cfg.scene.height_scanner = None
    cfg.observations = PGMTObservationsCfg()
    cfg.observations.proprio.enable_corruption = False
    cfg.observations.proprio_history.enable_corruption = False
    cfg.curriculum.terrain_progress = None
    cfg.commands.motion.placement_mode = "origin"
    cfg.commands.motion.placement_xy_range = (0., 0.)
    cfg.commands.motion.placement_yaw_range = (math.radians(args.yaw_deg),) * 2
    cfg.episode_length_s = 30.
    cfg.viewer.origin_type = "world"
    for name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
        setattr(cfg.events, name, None)
    old = cfg.scene.robot.spawn
    shared = {f.name: deepcopy(getattr(old, f.name)) for f in fields(sim_utils.UsdFileCfg)
              if f.name not in ("func", "usd_path") and hasattr(old, f.name)}
    cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
        usd_path=str(ROOT / "logs/pgmt_p1_acceptance/native_robot/main.usd"), **shared)
    dump_yaml(str(output / "env.yaml"), cfg)
    raw = ManagerBasedRLEnv(cfg)
    try:
        terrain = raw.scene.terrain
        terrain.terrain_types[:] = terrain.manifest["spec"]["families"].index(args.terrain)
        terrain.terrain_levels[:] = args.level
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        selected_patch = None
        if args.patch_id is not None:
            selected_patch = next(p for p in terrain.surface_records if p["family"] == args.terrain
                                  and p["level"] == args.level and p["patch_id"] == args.patch_id)
            terrain.env_origins[:] = terrain.env_origins.new_tensor(selected_patch["origin"])
        terrain.env_origins[:, :2] += terrain.env_origins.new_tensor(args.spawn_xy)
        if selected_patch is not None:
            bounds = terrain.env_origins.new_tensor(selected_patch["world_bounds"])[:, :2]
            margin = cfg.commands.motion.boundary_margin
            if not ((terrain.env_origins[:, :2] >= bounds[0] + margin)
                    & (terrain.env_origins[:, :2] <= bounds[1] - margin)).all():
                raise ValueError("Selected spawn is outside the patch interior")
        env = RslRlVecEnvWrapper(raw)
        obs = env.get_observations()
        agent = PGMTStage1PPORunnerCfg()
        policy_options = agent.policy.to_dict()
        policy_options.pop("class_name")
        policy = PGMTActorCritic(obs, agent.obs_groups, env.num_actions, **policy_options).to(args.device)
        loaded = torch.load(args.checkpoint, weights_only=False, map_location=args.device)
        actual, expected = policy.get_checkpoint_metadata(), loaded["policy_metadata"]
        assert actual["signature"] == expected["signature"], "Stage1 policy signature mismatch"
        assert actual["obs_schema"]["hash"] == expected["obs_schema"]["hash"], "Stage1 observation schema mismatch"
        policy.load_state_dict(loaded["model_state_dict"], strict=True)
        policy.eval()
        policy.requires_grad_(False)
        state = get_pgmt_state(raw)
        command = raw.command_manager.get_term("motion")
        robot = raw.scene["robot"]
        total_frames = int(command.data_store.motion_lengths[0])
        reference_shift = np.array([0., 1.2, 0.], dtype=np.float32)
        controls = dict(paused=args.paused, single=False, reset=False, speed=args.speed, follow=True,
                        closeup=args.touchdown_points, pause_td=args.pause_on_touchdown)
        touchdown_display = None
        capture_countdown = None
        recent = deque(maxlen=7)
        event_names = ("touchdown", "support_started", "support_ended", "contact_started", "contact_ended", "fast_contact_switch")
        event_counts = {key: [0, 0] for key in event_names}
        last_touchdown = [-100, -100]
        labels, foot_labels, marker_sets = {}, [], []
        viewport = None
        camera_pending = True
        stage = raw.sim.stage
        colors = dict(on=(.1, .95, .3), off=(.45, .45, .45), stance=(0., .9, 1.),
                      swing=(.15, .3, 1.), unknown=(1., .1, .8), event=(1., .8, .05))

        def make_sphere(name, radius):
            path = f"/World/PGMTContactDisplay/{name}"
            shape = UsdGeom.Sphere.Define(stage, path)
            shape.CreateRadiusAttr(radius)
            transform = shape.AddTranslateOp()
            material = UsdShade.Material.Define(stage, path + "/Material")
            shader = UsdShade.Shader.Define(stage, path + "/Material/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            diffuse = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
            emission = shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material)
            return shape, transform, diffuse, emission

        def put_marker(marker, position, color, visible=True):
            shape, transform, diffuse, emission = marker
            transform.Set(Gf.Vec3d(*map(float, position)))
            diffuse.Set(Gf.Vec3f(*color))
            emission.Set(Gf.Vec3f(*color) * .4)
            shape.GetVisibilityAttr().Set("inherited" if visible else "invisible")

        if not args.headless:
            import omni.ui as ui
            from isaacsim.core.simulation_manager import SimulationManager
            from isaacsim.core.utils.viewports import set_camera_view
            from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
            SimulationManager.enable_fabric(True)
            viewport = get_active_viewport()
            marker_sets = [[make_sphere(f"{kind}_{foot}", radius) for foot in ("L", "R")]
                           for kind, radius in (("Measured", .05), ("Reference", .05), ("Touchdown", .075))]
            skeleton = UsdGeom.BasisCurves.Define(stage, "/World/PGMTContactDisplay/ReferenceSkeleton")
            skeleton.CreateTypeAttr("linear")
            skeleton.CreateWidthsAttr([.018])
            skeleton.SetWidthsInterpolation("constant")
            skeleton.CreateDisplayColorAttr([Gf.Vec3f(.2, .8, .85)])
            chains = [["pelvis", "torso_link"]]
            for side in ("left", "right"):
                chains.extend([["pelvis", f"{side}_hip_roll_link", f"{side}_knee_link", f"{side}_ankle_roll_link"],
                               ["torso_link", f"{side}_shoulder_roll_link", f"{side}_elbow_link", f"{side}_wrist_yaw_link"]])
            skeleton_indices = [robot.body_names.index(name) for chain in chains for name in chain]
            initial_pose = state.reference()["body_pos"][0].detach().cpu().numpy()
            skeleton.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(
                (initial_pose[skeleton_indices] + reference_shift).astype(np.float32)))
            skeleton.CreateCurveVertexCountsAttr([len(chain) for chain in chains])
            panel = ui.Window("PGMT P3 | Touchdown 9 points" if args.touchdown_points else "PGMT P3 | Contact / offline labels",
                              width=620, height=920 if args.touchdown_points else 720)
            panel.position_x, panel.position_y = 920 if args.touchdown_points else 590, 35
            with panel.frame, ui.ScrollingFrame():
                with ui.VStack(spacing=6):
                    ui.Label(f"{args.terrain.upper()} L{args.level} | "
                             f"{'region origin' if args.patch_id is None else f'patch {args.patch_id}'} | "
                             "REAL PHYSICS | Stage1 model_15000 | NO TRAINING", height=24)
                    ui.Label(Path(args.motion).name, height=22)
                    with ui.HStack(height=28):
                        ui.Button("Pause / Run", clicked_fn=lambda: controls.update(paused=not controls["paused"]))
                        ui.Button("Step 20 ms", clicked_fn=lambda: controls.update(paused=True, single=True))
                        ui.Button("Reset frame 0", clicked_fn=lambda: controls.update(reset=True))
                    with ui.HStack(height=28):
                        for speed in (.25, .5, 1.):
                            ui.Button(f"{speed:g}x", clicked_fn=lambda speed=speed: controls.update(speed=speed))
                        ui.Button("Follow / Free", clicked_fn=lambda: controls.update(follow=not controls["follow"]))
                    labels["frame"] = ui.Label("Ready: first force sample pending", height=48, word_wrap=True)
                    if args.touchdown_points:
                        from pgmt_touchdown_visualization import TouchdownDisplay
                        touchdown_display = TouchdownDisplay(stage, make_sphere, put_marker, controls)
                    ui.Label("ROBOT: green = support; gray = no support; magenta = unknown", height=22)
                    ui.Label("SKELETON (+1.2 m Y): cyan = offline STANCE; blue = SWING", height=22)
                    ui.Label("Yellow = touchdown highlight (0.30 s display only); event itself = one step", height=28, word_wrap=True)
                    with ui.HStack(height=180):
                        for foot in ("LEFT", "RIGHT"):
                            with ui.VStack(spacing=3):
                                ui.Label(foot, height=22)
                                foot_labels.append(ui.Label("Waiting for first sample", word_wrap=True))
                    ui.Label("RECENT EVENTS (episode / reference frame; newest first)", height=22)
                    labels["events"] = ui.Label("No events yet", height=125, word_wrap=True)
                    ui.Label("c = all-object normal force norm >10 N; s = terrain normal Fz >10 N\n"
                             "Offline labels are reference targets, not force ground truth.\n"
                             "Pause/step buttons preserve control dt; no kinematic robot teleport per frame.",
                             height=60, word_wrap=True)

        csv_path = output / "contacts.csv"
        field_names = ["step", "episode", "episode_step", "frame", "foot", "done", "contact", "support",
                       "contact_valid", "support_valid", "force_norm", "upward_force", "support_time", "non_support_time",
                       "contact_age", "previous_contact_age", "previous_non_support_steps", "reference_contacts",
                       "reference_contact_valid", *event_names]
        steps, resets, renders = 0, 0, 0
        last_sample = None
        ref_pose = state.reference()["body_pos"][0].detach().cpu().numpy()
        next_step = time.monotonic()
        if touchdown_display is not None:
            touchdown_display.sample(state, robot)
        save_status(status="running", checkpoint_loaded_strictly=True, terrain=args.terrain, level=args.level,
                    patch_id=args.patch_id, spawn_xy_offset=list(args.spawn_xy), yaw_deg=args.yaw_deg,
                    placement_xy=command.placement_xy[0].cpu().tolist(),
                    terrain_parameters=None if selected_patch is None else selected_patch["parameters"],
                    total_frames=total_frames, control_dt=raw.step_dt, physics_dt=cfg.sim.dt, csv=str(csv_path),
                    contact_config=cfg.pgmt_contact_events, speed=controls["speed"], paused=controls["paused"],
                    touchdown_points=args.touchdown_points)
        with csv_path.open("w", newline="", buffering=1) as stream:
            writer = csv.DictWriter(stream, fieldnames=field_names)
            writer.writeheader()
            while app.is_running() and (not args.max_steps or steps < args.max_steps):
                now = time.monotonic()
                if controls["reset"]:
                    obs, _ = env.reset()
                    controls.update(reset=False, single=False)
                    last_sample = None
                    last_touchdown[:] = [-100, -100]
                    recent.appendleft("MANUAL RESET -> frame 0 / history invalidated")
                    camera_pending = True
                    if touchdown_display is not None:
                        touchdown_display.reset()
                        touchdown_display.sample(state, robot)
                advance = controls["single"] or (not controls["paused"] and now >= next_step)
                if advance:
                    controls["single"] = False
                    with torch.inference_mode():
                        # Capture BEFORE command advancement/reset, same frame as reward snapshot.
                        sampled_frame = int(command.frame_ids[0])
                        ref_pose = state.reference()["body_pos"][0].detach().cpu().numpy().copy()
                        action = policy.act_inference(obs)
                        if not torch.isfinite(action).all():
                            raise FloatingPointError("Non-finite action")
                        obs, _, dones, _ = env.step(action)
                        sample = {k: v[0].detach().cpu().numpy() for k, v in state.contact_state(for_reward=True).items()}
                        touchdown_sample = state.touchdown_state(for_reward=True) if touchdown_display is not None else None
                    assert int(sample["reference_frame_ids"]) == sampled_frame
                    steps += 1
                    done = bool(dones[0])
                    episode = int(sample["episode_id"])
                    for foot, side in enumerate(("L", "R")):
                        row = dict(step=raw.common_step_counter, episode=episode, episode_step=int(sample["episode_step"]),
                                   frame=sampled_frame, foot=side, done=done)
                        for key in field_names[6:]:
                            row[key] = sample[key][foot].item()
                        writer.writerow(row)
                        active = [key for key in event_names if sample[key][foot]]
                        for key in active:
                            event_counts[key][foot] += 1
                        if active:
                            recent.appendleft(f"E{episode} F{sampled_frame:03d} {side}: " + ", ".join(active))
                        if sample["touchdown"][foot]:
                            last_touchdown[foot] = raw.common_step_counter
                    last_sample = sample
                    if touchdown_display is not None:
                        if not done:
                            touchdown_display.sample(state, robot, touchdown_sample)
                        if not done and bool(touchdown_sample["event"].any()):
                            (output / "touchdown_points.json").write_text(json.dumps(
                                touchdown_display.record(), indent=2) + "\n")
                    if controls["pause_td"] and sample["touchdown"].any() and not done:
                        controls.update(paused=True, pause_td=False)
                        capture_countdown = 12
                        save_status(status="running", steps=steps, paused=True, pause_reason="touchdown",
                                    frame=sampled_frame, event_counts=event_counts)
                    if done:
                        resets += 1
                        recent.appendleft(f"E{episode} F{sampled_frame:03d} DONE -> reset frame 0")
                        # The viewport already contains the reset robot. Show invalid live
                        # state, never terminal forces/old foot markers on the new pose.
                        last_sample = None
                        last_touchdown[:] = [-100, -100]
                        camera_pending = True
                        if touchdown_display is not None:
                            touchdown_display.reset()
                            touchdown_display.sample(state, robot)
                    next_step = time.monotonic() + raw.step_dt / controls["speed"]
                if last_sample is None:
                    display = {k: v[0].detach().cpu().numpy() for k, v in state.contact_state().items()}
                    display["foot_body_pos"] = robot.data.body_pos_w[0, state.foot_body_indices].detach().cpu().numpy()
                    ref_pose = state.reference()["body_pos"][0].detach().cpu().numpy()
                    offline = {f"reference_{k}": v[0].detach().cpu().numpy() for k, v in state.reference_contacts().items()}
                    display.update(offline)
                else:
                    display = last_sample
                if viewport is not None:
                    frame = int(display["reference_frame_ids"])
                    labels["frame"].text = (f"{'PAUSED' if controls['paused'] else 'RUNNING'} | speed {controls['speed']:g}x | "
                                            f"step {raw.common_step_counter} | frame {frame}/{total_frames - 1}\n"
                                            f"Episode {int(display['episode_id'])} | " +
                                            ("reset baseline: force state UNKNOWN" if last_sample is None else "measured + offline: SAME reference frame"))
                    labels["events"].text = "\n".join(recent) or "No events yet"
                    skeleton.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy((ref_pose[skeleton_indices] + reference_shift).astype(np.float32)))
                    for foot in range(2):
                        valid, supported = bool(display["support_valid"][foot]), bool(display["support"][foot])
                        ref_valid = bool(display["reference_contact_valid"][foot])
                        stance = bool(display["reference_contacts"][foot])
                        foot_labels[foot].text = (
                            f"Support s: {'YES' if supported else 'NO'}" + ("" if valid else " (UNKNOWN)") +
                            f"\nTerrain Fz: {display['upward_force'][foot]:7.1f} N"
                            f"\nContact c: {bool(display['contact'][foot])} | norm {display['force_norm'][foot]:.1f} N"
                            f"\nSupport / unloaded: {display['support_time'][foot]:.2f} / {display['non_support_time'][foot]:.2f} s"
                            f"\nTouchdown NOW: {bool(display['touchdown'][foot])}"
                            f"\nFast switch NOW: {bool(display['fast_contact_switch'][foot])}"
                            f"\nOffline: {'UNKNOWN' if not ref_valid else 'STANCE' if stance else 'SWING'}"
                            f"\nTD count: {event_counts['touchdown'][foot]}")
                        pos = display["foot_body_pos"][foot] + np.array([0., 0., .06])
                        ref_pos = ref_pose[int(state.foot_body_indices[foot])] + reference_shift + np.array([0., 0., .06])
                        put_marker(marker_sets[0][foot], pos, colors["unknown" if not valid else "on" if supported else "off"],
                                   not controls["closeup"])
                        put_marker(marker_sets[1][foot], ref_pos, colors["unknown" if not ref_valid else "stance" if stance else "swing"])
                        put_marker(marker_sets[2][foot], pos + np.array([0., 0., .16]), colors["event"],
                                   not controls["closeup"] and (raw.common_step_counter - last_touchdown[foot]) * raw.step_dt < .30)
                    if touchdown_display is not None:
                        touchdown_display.render()
                    if controls["follow"] or camera_pending:
                        if controls["closeup"]:
                            target = robot.data.body_pos_w[0, state.foot_body_indices].mean(0).detach().cpu().numpy()
                            target += np.array([0., 0., .10])
                            eye_offset = np.array([.85, -1.05, .95])
                        else:
                            target = robot.data.body_pos_w[0, state.anchor].detach().cpu().numpy() + np.array([0., .6, -.22])
                            eye_offset = np.array([2.7, -3.4, 2.0])
                        set_camera_view(eye=target + eye_offset, target=target,
                                        camera_prim_path=str(viewport.camera_path))
                        camera_pending = False
                    raw.sim.render()
                    renders += 1
                    if renders == 100:
                        capture_viewport_to_file(viewport, str(output / "viewport.png"))
                    if capture_countdown is not None:
                        capture_countdown -= 1
                        if capture_countdown == 0:
                            capture_viewport_to_file(viewport, str(output / "touchdown_viewport.png"))
                            capture_countdown = None
                if advance and (steps == 1 or steps % 50 == 0):
                    save_status(status="running", steps=steps, resets=resets, event_counts=event_counts,
                                speed=controls["speed"], paused=controls["paused"], frame=int(display["reference_frame_ids"]))
                time.sleep(.005)
        save_status(status="closed", steps=steps, resets=resets, event_counts=event_counts)
    except BaseException as error:
        save_status(status="failed", error=repr(error))
        raise
    finally:
        raw.close()


from pmt_tasks.runtime_options import run_and_close
run_and_close(main, app)
