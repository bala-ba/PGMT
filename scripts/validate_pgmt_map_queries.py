"""Validate position-based terrain queries and map-exit truncation in native Isaac."""
import argparse
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default=str(ROOT / "logs/pgmt_map_queries"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
report_file = output / "validation.json"
report_file.unlink(missing_ok=True)
launcher = AppLauncher(args, multi_gpu=False, fast_shutdown=True)
app = launcher.app


def main():
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from pmt_tasks.env_cfgs.pgmt.placement import PGMTPlacementEnvCfg
    from pmt_tasks.mdp.commands.pgmt_motion_command import pgmt_map_out_of_bounds
    from pmt_tasks.pgmt_runtime import get_pgmt_state

    cfg = PGMTPlacementEnvCfg()
    cfg.pmt_motion_paths = [str(ROOT / "assets/motions/pgmt_clipped_50hz_tcrs_v1/walk1_subject1_clip0.npz")]
    cfg.__post_init__()
    cfg.scene.num_envs = 7
    cfg.seed = 716
    cfg.sim.device = args.device
    cfg.commands.motion.storage_device = args.device
    cfg.commands.motion.placement_xy_range = (0., 0.)
    cfg.commands.motion.placement_yaw_range = (0., 0.)
    cfg.observations.proprio.enable_corruption = False
    cfg.observations.proprio_history.enable_corruption = False
    for name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
        setattr(cfg.events, name, None)
    # Isolate this term from unrelated tracking failures during deliberate
    # robot/reference relocation. Production termination settings are unchanged.
    for name in vars(cfg.terminations):
        if name != "map_out_of_bounds":
            setattr(cfg.terminations, name, None)
    assert cfg.terminations.map_out_of_bounds.time_out
    env = ManagerBasedRLEnv(cfg=cfg)
    try:
        command = env.command_manager.get_term("motion")
        terrain, robot = env.scene.terrain, env.scene["robot"]
        state = get_pgmt_state(env)
        ids = torch.arange(7, device=env.device)
        terrain.terrain_types[:] = ids
        terrain.terrain_levels[:] = 0
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        env.reset()
        assigned_levels = terrain.terrain_levels.clone()
        assigned_types = terrain.terrain_types.clone()
        assigned_origins = terrain.env_origins.clone()

        # Independent audit points saved when generating the actual triangles.
        probes, expected_levels, expected_families, expected_groups = [], [], [], []
        for tile in terrain.manifest["tiles"]:
            n = len(tile["world_probes"])
            probes.extend(tile["world_probes"])
            expected_levels.extend([tile["level"]] * n)
            expected_families.extend([tile["family_index"]] * n)
            expected_groups.extend([tile["family_group_index"]] * n)
        probes = torch.tensor(probes, device=env.device)
        result = terrain.query(probes[:, :2])
        assert result["valid"].all()
        torch.testing.assert_close(result["height"], probes[:, 2], atol=1e-4, rtol=0)
        for key, expected in (("tile_level", expected_levels), ("family_index", expected_families),
                              ("family_group_index", expected_groups)):
            assert torch.equal(result[key], torch.tensor(expected, device=env.device))
        height_error = float((result["height"] - probes[:, 2]).abs().max())
        outside = torch.tensor([[-120.1, 0], [120.1, 0], [0, -84.1], [0, 84.1],
                                [float("nan"), 0], [0, float("inf")]], device=env.device)
        invalid = terrain.query(outside)
        assert not invalid["valid"].any() and invalid["height"].isnan().all()
        assert invalid["family_index"].eq(-1).all()

        def check_assignment():
            assert torch.equal(terrain.terrain_levels, assigned_levels)
            assert torch.equal(terrain.terrain_types, assigned_types)
            assert torch.equal(terrain.env_origins, assigned_origins)

        def relocate(xy, *, move_reference=True):
            # Test-only relocation preserves root height above the surface and
            # translates the reference by the same amount when requested.
            old_pos = robot.data.body_pos_w[0, 0].clone()
            position = old_pos.clone()
            position[:2] = position.new_tensor(xy)
            position[2] += terrain.height(position[None, :2])[0] - terrain.height(old_pos[None, :2])[0]
            # Outside map ray misses are intentional; retain original height.
            if not torch.isfinite(position[2]):
                position[2] = old_pos[2]
            root_state = torch.cat((position, robot.data.body_quat_w[0, 0], position.new_zeros(6)))[None]
            robot.write_root_link_state_to_sim(root_state, env_ids=ids[:1])
            if move_reference:
                command.reference_translation[0] += position - old_pos
                command._world_tracked = None
            env.sim.forward()
            state.invalidate_observations()

        def step(expect_truncation=False):
            _, _, terminated, truncated, _ = env.step(torch.zeros(7, env.action_manager.total_action_dim, device=env.device))
            assert not terminated.any()
            assert bool(truncated[0]) == expect_truncation and not truncated[1:].any()
            check_assignment()

        # Cross a level boundary into L1 flat, keeping birth level L0.
        relocate((-84., -72.))
        feet = state.robot_foot_terrain(env_ids=ids[:1])
        assert feet["tile_level"].eq(1).all() and feet["curriculum_level"].eq(0).all()
        assert not pgmt_map_out_of_bounds(env).any()
        step()

        # Stand astride the family seam: each foot must query its own tile.
        foot_y = robot.data.body_pos_w[0, state.foot_body_indices, 1].mean()
        target_y = -60. - float(foot_y - robot.data.body_pos_w[0, 0, 1])
        relocate((-108., target_y))
        feet = state.robot_foot_terrain(env_ids=ids[:1])
        assert set(feet["family_index"][0].tolist()) == {0, 1}
        assert feet["valid"].all() and not pgmt_map_out_of_bounds(env).any()
        step()
        relocate((-108., -48.))
        feet = state.robot_foot_terrain(env_ids=ids[:1])
        assert feet["family_index"].eq(1).all() and feet["assigned_family_index"].eq(0).all()
        assert not pgmt_map_out_of_bounds(env).any()
        step()

        # Actual reference exit and actual robot exit independently trigger a
        # native time_out truncation and reset back to the assigned spawn tile.
        command.reference_translation[0, 0] += 1000.
        command._world_tracked = None
        state.invalidate_observations()
        assert pgmt_map_out_of_bounds(env)[0]
        assert terrain.contains(command.robot_body_pos_w[0, :, :2]).all()
        step(expect_truncation=True)
        assert not pgmt_map_out_of_bounds(env).any()
        relocate((121., -72.), move_reference=False)
        assert pgmt_map_out_of_bounds(env)[0]
        assert terrain.contains(command.body_pos_w[0, :, :2]).all()
        step(expect_truncation=True)
        assert not pgmt_map_out_of_bounds(env).any()

        report = dict(passed=True, tile_count=len(terrain.manifest["tiles"]),
                      map_bounds=[terrain.tile_map.lower.tolist(), terrain.tile_map.upper.tolist()],
                      border_is_valid_map=False, audit_probes=len(probes), max_height_error_m=height_error,
                      position_based_family_and_level=True, both_feet_query_their_own_tile=True,
                      cross_level_and_family_do_not_truncate=True, birth_assignment_unchanged=True,
                      reference_map_exit_truncates=True, robot_map_exit_truncates=True,
                      truncation_not_termination=True, invalid_queries_masked=True, native_physics_steps=5,
                      unrelated_tracking_terminations_disabled_for_validation=True)
        report_file.write_text(json.dumps(report, indent=2) + "\n")
        print("PGMT MAP QUERIES VALIDATED: " + json.dumps(report), flush=True)
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
