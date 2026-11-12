"""Audit native PGMT RayCaster geometry, signed observations and map masks."""
import argparse
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default=str(ROOT / "logs/pgmt_height_scan"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
report_file = output / "validation.json"
report_file.unlink(missing_ok=True)
launcher = AppLauncher(args, multi_gpu=False, fast_shutdown=True)
app = launcher.app


def main():
    import math
    import torch
    import omni.physx
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import RayCaster
    from pmt_tasks.env_cfgs.pgmt.placement import PGMTPlacementEnvCfg
    from pmt_tasks.env_cfgs.pgmt.stage1 import PGMTStage1EnvCfg
    from pmt_tasks.mdp.commands.pgmt_motion_command import pgmt_map_out_of_bounds
    from pmt_tasks.pgmt_math import yaw_quat, rotate
    from pmt_tasks.pgmt_runtime import get_pgmt_state
    from pmt_tasks.isaaclab_rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper

    baseline = PGMTStage1EnvCfg()
    assert not hasattr(baseline.scene, "height_scanner")
    assert not hasattr(baseline.observations, "terrain_map")
    cfg = PGMTPlacementEnvCfg()
    cfg.pmt_motion_paths = [str(ROOT / "assets/motions/pgmt_clipped_50hz_tcrs_v1/walk1_subject1_clip0.npz")]
    cfg.__post_init__()
    cfg.scene.num_envs = 7
    cfg.seed = 718
    cfg.sim.device = args.device
    cfg.sim.enable_scene_query_support = True
    cfg.commands.motion.storage_device = args.device
    cfg.commands.motion.placement_xy_range = (0., 0.)
    cfg.observations.proprio.enable_corruption = False
    cfg.observations.proprio_history.enable_corruption = False
    cfg.curriculum.terrain_progress = None
    for name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
        setattr(cfg.events, name, None)
    # Deliberate relocations audit scan/truncation, not policy tracking quality.
    for name in vars(cfg.terminations):
        if name != "map_out_of_bounds":
            setattr(cfg.terminations, name, None)
    env = ManagerBasedRLEnv(cfg=cfg)
    try:
        terrain, robot = env.scene.terrain, env.scene["robot"]
        sensor = env.scene.sensors["height_scanner"]
        command = env.command_manager.get_term("motion")
        state = get_pgmt_state(env)
        ids = torch.arange(7, device=env.device)
        terrain.terrain_types[:] = ids
        assert sensor.num_rays == 441
        assert RayCaster.meshes[sensor.cfg.mesh_prim_paths[0]] is terrain.query_mesh
        assert sensor.cfg.update_period == env.step_dt
        local_y, local_x = torch.meshgrid(torch.linspace(-1., 1., 21, device=env.device),
                                         torch.linspace(-1., 1., 21, device=env.device), indexing="ij")
        local = torch.stack((local_x, local_y, torch.zeros_like(local_x)), -1).reshape(1, 441, 3)
        physics_query = omni.physx.get_physx_scene_query_interface()
        max_error, physics_error, probes, cases = 0., 0., 0, []

        def check(obs, independent=False):
            nonlocal max_error, physics_error, probes
            scan, hits = state.terrain_scan(), sensor.data.ray_hits_w
            assert obs["terrain_map"].shape == obs["terrain_valid"].shape == (7, 21, 21)
            assert obs["terrain_map"].isfinite().all()
            torch.testing.assert_close(obs["terrain_map"], scan["height"])
            torch.testing.assert_close(obs["terrain_valid"], scan["valid"].float())
            query = terrain.query(hits[..., :2])
            assert torch.equal(scan["valid"].flatten(1), query["valid"])
            valid = scan["valid"].flatten(1)
            expected_z = query["height"] - robot.data.body_pos_w[:, state.anchor, 2:3]
            torch.testing.assert_close(scan["height"].flatten(1)[valid], expected_z[valid], atol=1e-4, rtol=0)
            max_error = max(max_error, float((hits[..., 2][valid] - query["height"][valid]).abs().max()))
            assert scan["height"][~scan["valid"]].eq(0).all()
            assert scan["height_w"][~scan["valid"]].isnan().all()
            # Check real torso frame, both row/column axes, and yaw rotation.
            q = yaw_quat(robot.data.body_quat_w[:, state.anchor])
            expected_xy = (rotate(q[:, None].expand(-1, 441, -1), local.expand(7, -1, -1))
                           + robot.data.body_pos_w[:, state.anchor, None])[..., :2]
            torch.testing.assert_close(hits[..., :2][valid], expected_xy[valid], atol=4e-5, rtol=0)
            if independent:
                for e in range(7):
                    for index in (0, 110, 220, 330, 440):
                        if not valid[e, index]:
                            continue
                        point = hits[e, index].tolist()
                        # Center rays can hit the robot; compare terrain only.
                        ground = []

                        def collect(hit):
                            collision = hit.get("collision", "") if isinstance(hit, dict) else hit.collision
                            if str(collision).startswith(terrain.cfg.prim_path + "/"):
                                position = hit["position"] if isinstance(hit, dict) else hit.position
                                ground.append(position[2])
                            return True

                        physics_query.raycast_all((point[0], point[1], terrain.ray_top), (0., 0., -1.), 100., collect)
                        assert ground
                        error = abs(max(ground) - point[2])
                        assert error < 1e-4, (e, index, error)
                        physics_error = max(physics_error, error)
                        probes += 1
            return scan

        def reset(level, yaw):
            terrain.terrain_levels[:] = level
            terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
            command.cfg.placement_yaw_range = (yaw, yaw)
            return env.reset()[0]

        for level in range(10):
            for yaw in (0., math.pi / 2, math.pi, -math.pi / 2):
                obs = reset(level, yaw)
                scan = check(obs, independent=yaw == 0.)
                assert scan["valid"].all()
                assert scan["height_w"][5:, 10, 10].lt(0).all()
                cases.append(dict(level=level, yaw=yaw, inverted_center_world_z=scan["height_w"][5:, 10, 10].tolist()))

        # Move the physical torso (and reference) without changing birth assignment.
        obs = reset(0, 0.)
        birth = (terrain.terrain_levels.clone(), terrain.terrain_types.clone(), terrain.env_origins.clone())

        def relocate_torso(xy):
            old_root = robot.data.body_pos_w[0, 0].clone()
            position = old_root.clone()
            position[:2] += position.new_tensor(xy) - robot.data.body_pos_w[0, state.anchor, :2]
            root = torch.cat((position, robot.data.body_quat_w[0, 0], position.new_zeros(6)))[None]
            robot.write_root_link_state_to_sim(root, env_ids=ids[:1])
            command.reference_translation[0] += position - old_root
            command._world_tracked = None
            env.sim.forward()
            sensor.reset(ids[:1])
            state.invalidate_observations()
            return env.observation_manager.compute()

        # The grid spans two levels and then two families, with independent hits.
        obs = relocate_torso((-96., -72.))
        check(obs)
        labels = terrain.query(sensor.data.ray_hits_w[0, :, :2])
        assert set(labels["tile_level"].tolist()) == {0, 1}
        obs = relocate_torso((-108., -60.))
        check(obs)
        labels = terrain.query(sensor.data.ray_hits_w[0, :, :2])
        assert set(labels["family_index"].tolist()) == {0, 1}

        # Only scan rays cross the valid map edge; robot/reference remain inside.
        obs = relocate_torso((-119.4, -72.))
        scan = check(obs)
        edge_invalid = int((~scan["valid"][0]).sum())
        assert 0 < edge_invalid < 441
        assert not pgmt_map_out_of_bounds(env).any()
        for current, old in zip((terrain.terrain_levels, terrain.terrain_types, terrain.env_origins), birth):
            torch.testing.assert_close(current, old)
        obs, _, terminated, truncated, _ = env.step(torch.zeros(7, env.action_manager.total_action_dim, device=env.device))
        assert not terminated.any() and not truncated.any()
        check(obs)

        # A real robot/reference exit still truncates; returned scan is freshly reset.
        relocate_torso((-121.5, -72.))
        assert pgmt_map_out_of_bounds(env)[0]
        obs, _, terminated, truncated, _ = env.step(torch.zeros(7, env.action_manager.total_action_dim, device=env.device))
        assert not terminated.any() and truncated[0] and not truncated[1:].any()
        assert check(obs)["valid"].all()
        wrapped = RslRlVecEnvWrapper(env)
        tensor_obs = wrapped.get_observations()
        assert tensor_obs["terrain_map"].shape == tensor_obs["terrain_valid"].shape == (7, 21, 21)
        check(tensor_obs)
        assert not hasattr(PGMTStage1EnvCfg().scene, "height_scanner")

        report = dict(passed=True, terrain_asset=cfg.pgmt_terrain_asset, tile_count=70,
                      reset_placements=7 * len(cases), ray_count=441, shape=[7, 21, 21],
                      ordering="row=y,col=x; both axes -1..+1 m", height="ground_world_z - torso_world_z",
                      update_period_s=env.step_dt, max_mesh_height_error_m=max_error,
                      physx_probes=probes, max_physx_height_error_m=physics_error,
                      full_mesh_shared=True, yaw_and_axis_order=True, negative_world_heights=True,
                      cross_tile_scan=True, birth_assignment_unchanged=True, edge_invalid_cells=edge_invalid,
                      scan_edge_does_not_truncate=True, robot_exit_truncates=True,
                      post_reset_scan_fresh=True, wrapper_observations=True, stage1_unchanged=True,
                      native_physics_steps=2, tracking_terminations_disabled_for_audit=True, cases=cases)
        report_file.write_text(json.dumps(report, indent=2) + "\n")
        print("PGMT HEIGHT SCAN VALIDATED: " + json.dumps({k: v for k, v in report.items() if k != "cases"}), flush=True)
    finally:
        env.close()


try:
    main()
except Exception as error:
    report_file.write_text(json.dumps(dict(passed=False, error=repr(error)), indent=2) + "\n")
    traceback.print_exc()
    raise
finally:
    app.close()
