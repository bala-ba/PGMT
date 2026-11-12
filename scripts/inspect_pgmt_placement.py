"""Initialize the real PMT Isaac Lab environment on PGMT v5 for inspection.

Headless --validate checks all seven styles, ten levels and three yaw angles.
GUI mode holds the reset pose and provides terrain/level/yaw selection buttons.
No policy inference or training is started.
"""
import argparse
import json
import os
from pathlib import Path
import time
import sys

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--motion", default=str(ROOT / "assets/motions/pgmt_clipped_50hz_tcrs_v1/walk1_subject1_clip0.npz"))
parser.add_argument("--terrain", default=str(ROOT / "assets/terrain/pgmt_stage2_v5"))
parser.add_argument("--output", default=str(ROOT / "logs/pgmt_placement"))
parser.add_argument("--family", default="up_stairs", choices=["flat", "slopes", "stairs", "boxes", "random_rough", "up_stairs", "up_slope"])
parser.add_argument("--level", type=int, choices=range(10), default=5)
parser.add_argument("--yaw-deg", type=float, default=90.)
parser.add_argument("--start-frame", type=int, default=0)
parser.add_argument("--validate", action="store_true")
parser.add_argument("--export-scene", action="store_true", help="Export actual reset link poses and a USD scene for the Isaac 6 viewer")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args, multi_gpu=False, fast_shutdown=True)
app = launcher.app


def main():
    import numpy as np
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from pmt_tasks.env_cfgs.pgmt.placement import PGMTPlacementEnvCfg
    from pmt_tasks.pgmt_runtime import get_pgmt_state
    from pmt_tasks.pgmt_reference import collision_centres, in_tile
    from pmt_tasks.pgmt_terrain import FAMILIES
    from pmt_tasks.mdp.commands.pgmt_motion_command import pgmt_motion_finished

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.validate:
        (output / "validation.json").unlink(missing_ok=True)
    status_file = output / ("headless_status.json" if args.headless else "viewer_status.json")

    def status(state, **values):
        temporary = status_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(dict(state=state, pid=os.getpid(), time=time.time(), **values), indent=2) + "\n")
        temporary.replace(status_file)

    status("loading")
    cfg = PGMTPlacementEnvCfg()
    cfg.pmt_motion_paths = [args.motion]
    cfg.pgmt_terrain_asset = args.terrain
    cfg.__post_init__()
    cfg.scene.num_envs = 7
    cfg.seed = 1731
    cfg.sim.device = args.device
    # Headless Isaac Lab disables this by default. Independent PhysX checks
    # must query populated collision shapes, not an absent scene-query manager.
    cfg.sim.enable_scene_query_support = True
    cfg.commands.motion.storage_device = args.device
    cfg.commands.motion.fixed_start_frame = args.start_frame
    cfg.commands.motion.placement_xy_range = (0., 0.)
    cfg.commands.motion.placement_yaw_range = tuple([np.deg2rad(args.yaw_deg)] * 2)
    cfg.observations.proprio.enable_corruption = False
    cfg.observations.proprio_history.enable_corruption = False
    for name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
        setattr(cfg.events, name, None)
    env = ManagerBasedRLEnv(cfg=cfg)
    terrain = env.scene.terrain
    command = env.command_manager.get_term("motion")
    runtime = get_pgmt_state(env)
    ids = torch.arange(7, device=env.device)
    terrain.terrain_types[:] = ids
    import omni.physx
    physics_query = omni.physx.get_physx_scene_query_interface()
    cases = []
    pose_cases = []

    def reset(level, yaw):
        terrain.terrain_levels[:] = level
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        command.cfg.placement_yaw_range = tuple([np.deg2rad(yaw)] * 2)
        env.reset()
        env.sim.forward()

    def validate_case(level, yaw, physical=True):
        reset(level, yaw)
        current, future = runtime.reference(), runtime.reference(future=True)
        tracked = current["body_pos"][:, command.body_indices]
        torch.testing.assert_close(command.body_pos_w, tracked, atol=3e-5, rtol=0)
        torch.testing.assert_close(command.body_quat_w, current["body_quat"][:, command.body_indices], atol=2e-6, rtol=0)
        torch.testing.assert_close(command.body_lin_vel_w, current["body_lin_vel"][:, command.body_indices], atol=2e-6, rtol=0)
        torch.testing.assert_close(command.body_ang_vel_w, current["body_ang_vel"][:, command.body_indices], atol=2e-6, rtol=0)
        torch.testing.assert_close(command.body_pos_relative_w, tracked, atol=1e-4, rtol=0)
        for key in current:
            torch.testing.assert_close(future[key][:, 0], current[key], atol=2e-6, rtol=0)
        original = command._source_reference(command.motion_ids, command.frame_ids)
        torch.testing.assert_close(current["joint_pos"], original["joint_pos"], atol=0, rtol=0)
        torch.testing.assert_close(env.scene["robot"].data.joint_pos, original["joint_pos"], atol=2e-6, rtol=0)
        body_error = float((env.scene["robot"].data.body_pos_w - current["body_pos"]).abs().max())
        if body_error > 3e-4:
            raise AssertionError(f"Reset/full reference body mismatch: {body_error}")
        torch.testing.assert_close(env.scene["robot"].data.root_link_lin_vel_w, current["body_lin_vel"][:, 0], atol=2e-5, rtol=0)
        torch.testing.assert_close(env.scene["robot"].data.root_ang_vel_w, current["body_ang_vel"][:, 0], atol=2e-5, rtol=0)
        assert in_tile(current["body_pos"], terrain.bounds_for(ids), command.cfg.boundary_margin).all()
        assert terrain.contains(future["body_pos"][..., :2]).all()
        centres = collision_centres(current, command.collision_body_ids, command.collision_local_centres)
        distance = terrain.distance(centres)
        gap = distance - command.collision_radii[None]
        assert (gap >= command.cfg.spawn_clearance - 2e-5).all()
        overlaps = []
        if physical:
            # Independent PhysX overlap queries against imported USD collision
            # shapes; ignore the robot itself and count terrain overlaps only.
            def report_hit(hit):
                path = hit.get("collision", "") if isinstance(hit, dict) else getattr(hit, "collision", "")
                if str(path).startswith("/World/ground"):
                    overlaps.append(str(path))
                return True
            for points in centres.cpu().numpy():
                for point, radius in zip(points, command.collision_radii.cpu().numpy()):
                    physics_query.overlap_sphere(float(radius), tuple(float(x) for x in point), report_hit, False)
            if overlaps:
                raise AssertionError(f"Reset collision spheres intersect PhysX terrain: {overlaps[:3]}")
        item = dict(level=level, yaw_deg=yaw, body_max_error_m=body_error,
                    min_collision_gap_m=float(gap.min()), physx_terrain_overlaps=len(overlaps),
                    robot_root_positions=current["body_pos"][:, 0].tolist(),
                    translation=command.reference_translation.tolist(),
                    placement_attempts=command.placement_attempts.tolist())
        cases.append(item)
        if args.export_scene:
            robot = env.scene["robot"]
            pose_cases.append(dict(level=level, yaw_deg=yaw,
                                   body_pos=robot.data.body_pos_w.cpu().tolist(),
                                   body_quat=robot.data.body_quat_w.cpu().tolist()))
        print(json.dumps(dict(level=level, yaw_deg=yaw, body_error=body_error,
                              gap=float(gap.min()), physx_overlaps=len(overlaps))), flush=True)

    def export_scene():
        import omni.usd
        robot = env.scene["robot"]
        scene_path = output / "initialization.usdc"
        omni.usd.get_context().get_stage().Flatten().Export(str(scene_path))
        payload = dict(source="PMT Isaac Lab actual reset body poses", scene=scene_path.name,
                       motion=args.motion, start_frame=args.start_frame, families=list(FAMILIES),
                       terrain_manifest=terrain.manifest, body_names=list(robot.body_names),
                       link_paths=robot.root_physx_view.link_paths, cases=pose_cases)
        (output / "reset_poses.json").write_text(json.dumps(payload) + "\n")
        print(f"[PGMT] Exported reset poses and scene to {output}", flush=True)

    try:
        if args.validate:
            # Prove that the imported USD colliders and cached GPU query mesh
            # occupy the same world coordinates, including negative origins.
            reset(0, 0.)
            ground_probe_errors = []
            for tile in terrain.manifest["tiles"]:
                for x, y, expected_z in tile["world_probes"][1:]:
                    hit = physics_query.raycast_closest((x, y, terrain.ray_top), (0., 0., -1.), 1000.)
                    if not hit.get("hit") or not str(hit.get("collision", "")).startswith("/World/ground"):
                        raise AssertionError(f"Missing PhysX terrain below {tile['name']}: {hit}")
                    height = terrain.height(torch.tensor([[x, y]], dtype=torch.float32, device=env.device)).item()
                    error = max(abs(hit["position"][2] - expected_z), abs(height - expected_z))
                    if error > 5e-5:
                        raise AssertionError(f"PhysX/GPU reference mesh mismatch: {tile['name']} {error}")
                    ground_probe_errors.append(error)
            for level in range(10):
                for yaw in (0., 90., -135.):
                    validate_case(level, yaw)
            # A partial reset must leave every other environment's transform
            # and references unchanged; history is reset only for the selected env.
            previous = runtime.reference()["body_pos"].clone()
            command.cfg.placement_yaw_range = (.3, .3)
            command._resample_command(ids[:1])
            env.sim.forward()
            torch.testing.assert_close(runtime.reference()["body_pos"][1:], previous[1:], atol=0, rtol=0)
            # End of data freezes the reference until a proper episode reset.
            motion_before = command.motion_ids.clone()
            rotation_before = command.reference_rotation.clone()
            command.frame_ids[:] = command.data_store.motion_lengths[command.motion_ids] - 2
            command._update_command()
            assert pgmt_motion_finished(env).all()
            command._update_command()
            assert torch.equal(command.motion_ids, motion_before)
            assert torch.equal(command.reference_rotation, rotation_before)
            report = dict(passed=True, motion=args.motion, start_frame=args.start_frame,
                          terrain=args.terrain, placements=len(cases) * 7,
                          terrain_probes=len(ground_probe_errors), terrain_max_height_error_m=max(ground_probe_errors),
                          current_future_command_reset_consistent=True, source_joints_preserved=True,
                          root_link_velocity_convention=True, partial_reset_isolated=True,
                          segment_end_requires_episode_reset=True, cases=cases)
            (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
            if args.export_scene:
                export_scene()
            status("validated", placements=len(cases) * 7)
            if args.headless:
                return
        validate_case(args.level, args.yaw_deg)
        if args.export_scene:
            export_scene()
        if args.headless:
            status("validated", placements=7)
            return
        import omni.ui as ui
        from isaacsim.core.utils.viewports import set_camera_view
        from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file

        selection = dict(family=FAMILIES.index(args.family), level=args.level, yaw=args.yaw_deg)
        def focus():
            i = selection["family"]
            root = runtime.reference()["body_pos"][i, 0].cpu().numpy()
            target = root + [0., 0., .15]
            eye = target + [3.4, 3.4, 2.2]
            ground_at_camera = terrain.height(torch.as_tensor(eye[None, :2], dtype=torch.float32, device=env.device)).item()
            eye[2] = max(eye[2], ground_at_camera + 1.5)
            set_camera_view(eye=eye, target=target)
            info_label.text = f"{FAMILIES[i]} | L{selection['level']} | yaw {selection['yaw']:g} deg | frame {args.start_frame}"
            status("running", family=FAMILIES[i], level=selection["level"], yaw_deg=selection["yaw"],
                   start_frame=args.start_frame, motion=args.motion, inspection="reset_pose_held")
        def choose_family(i):
            selection["family"] = i
            focus()
        def choose_level(level):
            selection["level"] = level
            reset(level, selection["yaw"])
            focus()
        def choose_yaw(yaw):
            selection["yaw"] = yaw
            reset(selection["level"], yaw)
            focus()
        window = ui.Window("PGMT Initialization", width=460, height=330)
        with window.frame:
            with ui.VStack(spacing=6):
                ui.Label("Reference / command / reset share one rigid transform")
                ui.Label(Path(args.motion).name)
                ui.Label("Inspection holds the reset pose. No policy is running.")
                info_label = ui.Label("")
                for start in (0, 4):
                    with ui.HStack():
                        for i in range(start, min(start + 4, 7)):
                            ui.Button(FAMILIES[i], clicked_fn=lambda i=i: choose_family(i))
                ui.Label("Terrain level")
                with ui.HStack():
                    for level in range(10):
                        ui.Button(str(level), clicked_fn=lambda level=level: choose_level(level))
                ui.Label("Initial world yaw")
                with ui.HStack():
                    for yaw in (0., 90., 180., -90.):
                        ui.Button(str(int(yaw)), clicked_fn=lambda yaw=yaw: choose_yaw(yaw))
        focus()
        for _ in range(30):
            env.sim.render()
        capture_viewport_to_file(get_active_viewport(), str(output / "initialization.png"))
        while app.is_running():
            env.sim.render()
            time.sleep(.01)
        status("closed")
    finally:
        print("[PGMT] Closing inspection environment", flush=True)
        env.close()


try:
    main()
except Exception as error:
    import traceback
    traceback.print_exc()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / ("headless_status.json" if args.headless else "viewer_status.json")).write_text(
        json.dumps(dict(state="failed", pid=os.getpid(), time=time.time(), error=str(error)), indent=2) + "\n"
    )
    raise
finally:
    print("[PGMT] Closing Isaac application", flush=True)
    app.close()
