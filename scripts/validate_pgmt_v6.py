"""Native v6 acceptance: complete motion pools, terrain scans, and real PPO resume.

Run each check in a fresh Isaac process. Reports describe finite coverage, not
all possible continuous spawn poses or policy convergence.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--check", choices=("poses", "scan", "resume", "perception"), required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--terrain_asset", default=str(ROOT / "assets/terrain/pgmt_stage2_v6"))
parser.add_argument("--checkpoint", default=str(ROOT / "logs/pgmt_v6_blind/20261011_02/model_15999.pt"))
parser.add_argument("--robot_usd", default=str(ROOT / "logs/pgmt_p1_acceptance/native_robot/main.usd"))
parser.add_argument("--isaacsim6_compat", action="store_true")
parser.add_argument("--seed", type=int, default=10411)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
output = Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=True)
report_file = output / "validation.json"
if report_file.exists():
    parser.error(f"Refusing to overwrite an existing audit: {output}")
report = dict(passed=False, check=args.check, pid=os.getpid(), started_at=datetime.now().isoformat(),
              terrain_asset=str(Path(args.terrain_asset).resolve()), seed=args.seed,
              runtime="Isaac Sim 6 + PMT Isaac Lab" if args.isaacsim6_compat else "PMT native Isaac",
              device=args.device, headless=args.headless)


def write_progress(**values):
    report.update(values, updated_at=datetime.now().isoformat())
    path = output / "progress.json"
    path.with_suffix(".tmp").write_text(json.dumps(report, indent=2) + "\n")
    path.with_suffix(".tmp").replace(path)
    print(f"[V6 {args.check}] {json.dumps(values)}", flush=True)


write_progress(status="launching")
if args.isaacsim6_compat:
    from pmt_tasks.isaacsim6_legacy import install_tensor_import_bridge
    install_tensor_import_bridge()
app = AppLauncher(args, multi_gpu=False, fast_shutdown=True).app


def make_env():
    from dataclasses import fields
    from isaaclab.envs import ManagerBasedRLEnv
    import isaaclab.sim as sim_utils
    from isaaclab.utils.io import dump_yaml
    from pmt_tasks.env_cfgs.pgmt.p1_blind import PGMTBlindTerrainEnvCfg
    from pmt_tasks.env_cfgs.pgmt.placement import PGMTPlacementEnvCfg
    from pmt_tasks.env_cfgs.pgmt.stage2 import PGMTPerceptionEnvCfg

    scan_check = args.check in ("scan", "perception")
    cfg = (PGMTPerceptionEnvCfg() if args.check == "perception" else
           PGMTPlacementEnvCfg() if scan_check else PGMTBlindTerrainEnvCfg())
    motions = ROOT / "assets/motions/pgmt_clipped_50hz_tcrs_v1"
    cfg.pmt_motion_paths = [str(motions / "walk1_subject1_clip0.npz" if scan_check else motions)]
    cfg.pgmt_terrain_asset = str(Path(args.terrain_asset).resolve())
    cfg.__post_init__()
    cfg.scene.num_envs = 5 if scan_check else 160
    cfg.seed = args.seed
    cfg.sim.device = cfg.commands.motion.storage_device = args.device
    cfg.sim.enable_scene_query_support = True
    cfg.sim.physx.gpu_collision_stack_size = 2**28
    cfg.scene.contact_forces.debug_vis = False
    cfg.commands.motion.max_placement_attempts = 128
    if args.isaacsim6_compat:
        cfg.sim.use_fabric = True
    old = cfg.scene.robot.spawn
    shared = {f.name: copy.deepcopy(getattr(old, f.name)) for f in fields(sim_utils.UsdFileCfg)
              if f.name not in ("func", "usd_path") and hasattr(old, f.name)}
    cfg.scene.robot.spawn = sim_utils.UsdFileCfg(usd_path=str(Path(args.robot_usd).resolve()), **shared)
    if args.check != "resume":
        cfg.curriculum.terrain_progress = None
        cfg.observations.proprio.enable_corruption = False
        cfg.observations.proprio_history.enable_corruption = False
        for name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
            setattr(cfg.events, name, None)
        for name in vars(cfg.terminations):
            if name != "map_out_of_bounds":
                setattr(cfg.terminations, name, None)
    dump_yaml(str(output / "env.yaml"), cfg)
    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()
    assert env.scene.terrain.has_patches
    assert env.command_manager.get_term("motion").placement_mode == "patch"
    write_progress(status="environment_ready", terrain_sha256=env.scene.terrain.manifest["asset_sha256"])
    return env


def check_pose(env, ids):
    """Actual articulation state and every reference consumer must agree."""
    import torch
    from pmt_tasks.pgmt_runtime import get_pgmt_state
    from pmt_tasks.pgmt_reference import collision_centres, in_tile

    cmd, terrain, robot = env.command_manager.get_term("motion"), env.scene.terrain, env.scene["robot"]
    state = get_pgmt_state(env)
    current, future = state.reference(), state.reference(future=True)
    assert cmd.frame_ids[ids].eq(0).all()
    assert cmd.compatible_motions[terrain.terrain_types[ids], cmd.motion_ids[ids]].all()
    for field in ("body_pos", "body_quat", "body_lin_vel", "body_ang_vel"):
        torch.testing.assert_close(getattr(cmd, field + "_w")[ids],
                                   current[field][ids][:, cmd.body_indices], atol=3e-5, rtol=0)
    for key in current:
        torch.testing.assert_close(future[key][ids, 0], current[key][ids], atol=2e-6, rtol=0)
    source = cmd._source_reference(cmd.motion_ids[ids], cmd.frame_ids[ids])
    for key in ("joint_pos", "joint_vel"):
        torch.testing.assert_close(current[key][ids], source[key], atol=0, rtol=0)
        torch.testing.assert_close(getattr(robot.data, key)[ids], source[key], atol=2e-5, rtol=0)
    error = float((robot.data.body_pos_w[ids] - current["body_pos"][ids]).abs().max())
    assert error <= 5e-5, ("reset position", error)
    orientation = (robot.data.body_quat_w[ids] * current["body_quat"][ids]).sum(-1).abs()
    assert (orientation - 1).abs().max() < 2e-5
    torch.testing.assert_close(robot.data.root_link_lin_vel_w[ids], current["body_lin_vel"][ids, 0], atol=3e-5, rtol=0)
    torch.testing.assert_close(robot.data.root_ang_vel_w[ids], current["body_ang_vel"][ids, 0], atol=3e-5, rtol=0)
    assert terrain.contains(future["body_pos"][ids, ..., :2]).all()
    loc = terrain.tile_map.locate(cmd.placement_xy[ids])
    for name, expected in (("tile_level", terrain.terrain_levels[ids]), ("family_index", terrain.terrain_types[ids]),
                           ("patch_id", cmd.placement_patch_ids[ids])):
        assert torch.equal(loc[name], expected), name
    centres = collision_centres({k: v[ids] for k, v in current.items()}, cmd.collision_body_ids, cmd.collision_local_centres)
    radii = cmd.collision_radii[None]
    bounds = terrain.patch_bounds[terrain.terrain_levels[ids], terrain.terrain_types[ids], cmd.placement_patch_ids[ids]]
    assert in_tile(centres - radii[..., None], bounds, cmd.cfg.boundary_margin).all()
    assert in_tile(centres + radii[..., None], bounds, cmd.cfg.boundary_margin).all()
    gap = terrain.distance(centres) - radii
    assert gap.min() >= cmd.cfg.spawn_clearance - 1e-5
    return error, float(gap.min()), centres


def audit_poses(env):
    import math
    import torch
    import omni.physx
    from pmt_tasks.pgmt_reference import initial_transform, place_on_surface, transform_reference
    from pmt_tasks.pgmt_math import future_offsets
    from pmt_tasks.pgmt_runtime import get_pgmt_state

    cmd, terrain = env.command_manager.get_term("motion"), env.scene.terrain
    state = get_pgmt_state(env)
    mids = torch.arange(cmd.data_store.num_motions, device=env.device)
    first = cmd._source_reference(mids, torch.zeros_like(mids))
    limits = env.scene["robot"].data.joint_pos_limits[0]
    assert ((first["joint_pos"] >= limits[:, 0] - 1e-5) & (first["joint_pos"] <= limits[:, 1] + 1e-5)).all()
    geometry_rows, geometry_cases, rejected, minimum_gap = [], 0, 0, float("inf")
    num_levels, num_families, num_patches = terrain.patch_bounds.shape[:3]
    # Exhaustive discrete coverage: every compatible motion x level x patch,
    # with two reproducible random continuous XY/yaw proposals per combination.
    for level in range(num_levels):
        for family in range(num_families):
            chosen = cmd.compatible_motions[family].nonzero().flatten()
            n = len(chosen)
            ref = {k: v[chosen] for k, v in first.items()}
            fids = chosen[:, None].expand(-1, 8)
            frames = torch.minimum(future_offsets(device=env.device)[None], cmd.data_store.motion_lengths[fids] - 1)
            future = cmd._source_reference(fids.flatten(), frames.flatten())
            future = {k: v.reshape(n, 8, *v.shape[1:]) for k, v in future.items()}
            accepted_motion = torch.zeros(n, dtype=torch.bool, device=env.device)
            rejected_here = 0
            for patch in range(num_patches):
                bounds = terrain.patch_bounds[level, family, patch].expand(n, -1, -1)
                low, high = bounds[:, 0] + cmd.cfg.boundary_margin, bounds[:, 1] - cmd.cfg.boundary_margin
                for _ in range(2):
                    xy = low + (high - low) * torch.rand(n, 2, device=env.device)
                    yaw = (2 * torch.rand(n, device=env.device) - 1) * math.pi
                    q, t = initial_transform(ref["body_pos"][:, 0], ref["body_quat"][:, 0], xy, yaw)
                    t, valid, gap = place_on_surface(ref, q, t, bounds, terrain, cmd.collision_body_ids,
                        cmd.collision_local_centres, cmd.collision_radii, clearance=cmd.cfg.spawn_clearance,
                        boundary_margin=cmd.cfg.boundary_margin)
                    future_world = transform_reference(future, q, t)
                    valid &= terrain.contains(future_world["body_pos"][..., :2]).flatten(1).all(-1)
                    accepted_motion |= valid
                    rejected_here += int((~valid).sum())
                    if valid.any():
                        minimum_gap = min(minimum_gap, float(gap[valid].min()))
                    geometry_cases += n
            assert accepted_motion.all(), (level, family, chosen[~accepted_motion].tolist())
            rejected += rejected_here
            geometry_rows.append(dict(level=level, family=cmd.terrain_families[family], motions=n,
                                      proposals=n * num_patches * 2, rejected=rejected_here,
                                      every_motion_has_valid_candidate=True))
        write_progress(status="candidate_geometry", completed_levels=level + 1,
                       geometry_candidates=geometry_cases, rejected_candidates=rejected)

    # Exercise the unmodified command reset/retry on every compatible motion
    # and region. Patches are chosen by the real sampler, not forced to centers.
    physics = omni.physx.get_physx_scene_query_interface()
    patch_counts = torch.zeros(num_levels, num_families, num_patches, dtype=torch.long, device=env.device)
    physical_cells, physical_queries = set(), 0
    actual_rows, actual_count, max_error, max_attempts = [], 0, 0., 0
    all_ids = torch.arange(env.num_envs, device=env.device)
    (output / "reset_samples.jsonl").write_text("")
    for level in range(num_levels):
        for family in range(num_families):
            chosen = cmd.compatible_motions[family].nonzero().flatten()
            for batch in chosen.split(env.num_envs):
                ids = all_ids[:len(batch)]
                other_q, other_t = cmd.reference_rotation[len(batch):].clone(), cmd.reference_translation[len(batch):].clone()
                terrain.terrain_levels[ids], terrain.terrain_types[ids] = level, family
                terrain.env_origins[ids] = terrain.terrain_origins[level, family]
                cmd.motion_ids[ids], cmd.frame_ids[ids] = batch, 0
                cmd._reset_robot_state(ids)
                cmd._update_motion_buffers_for_envs(ids)
                cmd._compute_relative_poses()
                state.invalidate_observations()
                error, gap, centres = check_pose(env, ids)
                torch.testing.assert_close(cmd.reference_rotation[len(batch):], other_q, atol=0, rtol=0)
                torch.testing.assert_close(cmd.reference_translation[len(batch):], other_t, atol=0, rtol=0)
                max_error, minimum_gap = max(max_error, error), min(minimum_gap, gap)
                max_attempts = max(max_attempts, int(cmd.placement_attempts[ids].max()))
                patch_counts[level, family] += torch.bincount(cmd.placement_patch_ids[ids], minlength=num_patches)
                for j, patch in enumerate(cmd.placement_patch_ids[ids].tolist()):
                    cell = (level, family, patch)
                    if cell in physical_cells:
                        continue
                    overlaps = []
                    def collect(hit):
                        path = hit.get("collision", "") if isinstance(hit, dict) else hit.collision
                        if str(path).startswith(terrain.cfg.prim_path + "/"):
                            overlaps.append(str(path))
                        return True
                    for point, radius in zip(centres[j].tolist(), cmd.collision_radii.tolist()):
                        physics.overlap_sphere(float(radius), tuple(point), collect, False)
                        physical_queries += 1
                    assert not overlaps, ("PhysX overlap", cell, overlaps[:3])
                    physical_cells.add(cell)
                with (output / "reset_samples.jsonl").open("a") as file:
                    file.write(json.dumps(dict(level=level, family=family, motion_ids=batch.tolist(),
                        patch_ids=cmd.placement_patch_ids[ids].tolist(), xy=cmd.placement_xy[ids].tolist(),
                        yaw=cmd.placement_yaw[ids].tolist(), attempts=cmd.placement_attempts[ids].tolist())) + "\n")
                actual_count += len(batch)
            actual_rows.append(dict(level=level, family=cmd.terrain_families[family], motions=len(chosen)))
        write_progress(status="actual_resets", completed_levels=level + 1, actual_resets=actual_count,
                       physical_patches=len(physical_cells), max_pose_error_m=max_error)
    assert patch_counts.gt(0).all(), "Some patch lacks an actual random reset and independent PhysX overlap check"
    report.update(motions=len(mids), compatible_pairs_per_level=int(cmd.compatible_motions.sum()),
                  geometry_candidates=geometry_cases, rejected_candidates=rejected,
                  geometry_rows=geometry_rows, actual_reset_rows=actual_rows, actual_resets=actual_count,
                  physical_patches=len(physical_cells), physx_overlap_queries=physical_queries,
                  physx_terrain_overlaps=0, minimum_collision_gap_m=minimum_gap,
                  max_pose_error_m=max_error, max_placement_attempts=max_attempts,
                  patch_reset_counts=patch_counts.tolist(), current_future_command_reset_consistent=True,
                  source_joints_preserved=True, partial_reset_isolated=True, fixed_start_frame=0,
                  random_start_frames=False, random_xy_yaw=True, samples_per_motion_region_patch=2,
                  scope="Every CSV-compatible motion x 50 regions x 16 patches: two random candidate poses; "
                        "one actual command reset per compatible motion/region; independent PhysX overlaps on every patch")


def audit_scan(env):
    import math
    import torch
    import omni.physx
    from isaaclab.sensors import RayCaster
    from pmt_tasks.pgmt_math import yaw_quat, rotate
    from pmt_tasks.pgmt_runtime import get_pgmt_state
    from pmt_tasks.mdp.commands.pgmt_motion_command import pgmt_map_out_of_bounds
    from pmt_tasks.isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from pmt_tasks.env_cfgs.pgmt.stage1 import PGMTStage1EnvCfg

    terrain, robot, cmd = env.scene.terrain, env.scene["robot"], env.command_manager.get_term("motion")
    state, sensor = get_pgmt_state(env), env.scene.sensors["height_scanner"]
    ids = torch.arange(env.num_envs, device=env.device)
    terrain.terrain_types[:] = ids
    assert sensor.num_rays == 441 and sensor.cfg.update_period == env.step_dt
    assert RayCaster.meshes[sensor.cfg.mesh_prim_paths[0]] is terrain.query_mesh
    local_y, local_x = torch.meshgrid(torch.linspace(-1., 1., 21, device=env.device),
                                     torch.linspace(-1., 1., 21, device=env.device), indexing="ij")
    local = torch.stack((local_x, local_y, torch.zeros_like(local_x)), -1).reshape(1, 441, 3)
    physics = omni.physx.get_physx_scene_query_interface()
    max_error, physics_error, probes, checked_rays = 0., 0., 0, 0

    def check(obs, independent=False):
        nonlocal max_error, physics_error, probes, checked_rays
        from pmt_tasks.pgmt_perception import split_terrain_observation
        obs = split_terrain_observation(obs)
        scan, hits = state.terrain_scan(), sensor.data.ray_hits_w
        assert obs["terrain_map"].shape == obs["terrain_valid"].shape == (env.num_envs, 21, 21)
        assert obs["terrain_map"].isfinite().all()
        if getattr(env.cfg, "pgmt_terrain_observation", None) is not None:
            # Version 3 policy heights use PMT sign/fill/clip, even with noise off.
            expected_policy = torch.where(scan["valid"], -scan["height"],
                                           (sensor.data.pos_w[:, 2] - 30.)[:, None, None]).clamp(-3., 3.)
            torch.testing.assert_close(obs["terrain_map"], expected_policy, atol=1e-5, rtol=0)
        else:
            torch.testing.assert_close(obs["terrain_map"], scan["height"])
        torch.testing.assert_close(obs["terrain_valid"], scan["valid"].float())
        q = yaw_quat(robot.data.body_quat_w[:, state.anchor])
        expected_xy = (rotate(q[:, None].expand(-1, 441, -1), local.expand(env.num_envs, -1, -1))
                       + robot.data.body_pos_w[:, state.anchor, None])[..., :2]
        query = terrain.query(expected_xy)
        valid = scan["valid"].flatten(1)
        assert torch.equal(valid, query["valid"]), "Native ray validity differs from world-position map mask"
        torch.testing.assert_close(hits[..., :2][valid], expected_xy[valid], atol=4e-5, rtol=0)
        expected_z = query["height"] - robot.data.body_pos_w[:, state.anchor, 2:3]
        torch.testing.assert_close(scan["height"].flatten(1)[valid], expected_z[valid], atol=1e-4, rtol=0)
        error = float((hits[..., 2][valid] - query["height"][valid]).abs().max())
        max_error = max(max_error, error)
        assert scan["height"][~scan["valid"]].eq(0).all()
        assert scan["height_w"][~scan["valid"]].isnan().all()
        checked_rays += valid.numel()
        if independent:
            for e in range(env.num_envs):
                for index in (0, 110, 220, 330, 440):
                    if not valid[e, index]:
                        continue
                    point, elevations = hits[e, index].tolist(), []
                    def collect(hit):
                        path = hit.get("collision", "") if isinstance(hit, dict) else hit.collision
                        if str(path).startswith(terrain.cfg.prim_path + "/"):
                            position = hit["position"] if isinstance(hit, dict) else hit.position
                            elevations.append(position[2])
                        return True
                    physics.raycast_all((point[0], point[1], terrain.ray_top), (0., 0., -1.), 100., collect)
                    assert elevations
                    error = abs(max(elevations) - point[2])
                    assert error < 1e-4, (e, index, error)
                    physics_error, probes = max(physics_error, error), probes + 1
        return scan

    def relocate(selected, xy):
        # Controlled sensor poses: keep bodies high enough for a static scan.
        # This is a scan relocation, not a claim about random spawn acceptance.
        old_root = robot.data.body_pos_w[selected, 0].clone()
        torso = robot.data.body_pos_w[selected, state.anchor].clone()
        ground = terrain.height(xy)
        target_z = torch.where(torch.isfinite(ground), ground + 1.5, torso[:, 2])
        delta = torch.cat((xy - torso[:, :2], (target_z - torso[:, 2])[:, None]), -1)
        root = torch.cat((old_root + delta, robot.data.body_quat_w[selected, 0],
                          torch.zeros(len(selected), 6, device=env.device)), -1)
        robot.write_root_link_state_to_sim(root, env_ids=selected)
        cmd.reference_translation[selected] += delta
        cmd._world_tracked = None
        env.sim.forward()
        sensor.reset(selected)
        state.invalidate_observations()
        return env.observation_manager.compute()

    cases = []
    for level in range(10):
        terrain.terrain_levels[:] = level
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        for yaw in (0., math.pi / 2, math.pi, -math.pi / 2):
            cmd.cfg.placement_yaw_range = (yaw, yaw)
            env.reset()
            for patch in range(16):
                bounds = terrain.patch_bounds[level, ids, patch]
                xy = bounds.mean(1) + bounds.new_tensor([.031, .019])
                obs = relocate(ids, xy)
                scan = check(obs, independent=yaw == 0.)
                assert scan["valid"].all()
                assert scan["height_w"][1:3, 10, 10].lt(0).all(), "Inverted stairs/slope pit must retain negative world Z"
                cases.append(dict(level=level, patch=patch, yaw=yaw,
                                  inverted_center_world_z=scan["height_w"][1:3, 10, 10].tolist()))
        write_progress(status="scanning_patches", completed_levels=level + 1, sensor_poses=len(cases) * env.num_envs,
                       checked_rays=checked_rays, independent_physx_probes=probes)

    terrain.terrain_levels.zero_()
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    cmd.cfg.placement_yaw_range = (0., 0.)
    env.reset()
    birth = (terrain.terrain_levels.clone(), terrain.terrain_types.clone(), terrain.env_origins.clone())
    seams = [
        ("patch", [-114.023, -56.91], "patch_id", {0, 4}),
        ("level", [-96.023, -56.91], "tile_level", {0, 1}),
        ("family", [-116.91, -36.023], "family_index", {0, 1}),
    ]
    for name, xy, field, expected in seams:
        check(relocate(ids[:1], torch.tensor([xy], device=env.device)), independent=True)
        actual = terrain.query(sensor.data.ray_hits_w[0, :, :2])[field]
        assert set(actual.tolist()) == expected, (name, actual.unique().tolist())
    edges = [(-119.15, -56.91), (119.15, -56.91), (-116.91, -59.15), (-116.91, 59.15)]
    invalid_counts = []
    for xy in edges:
        scan = check(relocate(ids[:1], torch.tensor([xy], device=env.device)), independent=True)
        invalid = int((~scan["valid"][0]).sum())
        assert 0 < invalid < 441
        assert not pgmt_map_out_of_bounds(env).any(), ("Scan-only edge should not truncate", xy)
        invalid_counts.append(invalid)
        for now, before in zip((terrain.terrain_levels, terrain.terrain_types, terrain.env_origins), birth):
            torch.testing.assert_close(now, before, atol=0, rtol=0)
    obs, _, terminated, truncated, _ = env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))
    assert not terminated.any() and not truncated.any()
    check(obs)
    # A real exit resets only env 0. The returned scan must use the new pose.
    relocate(ids[:1], torch.tensor([[-121.5, -56.91]], device=env.device))
    assert pgmt_map_out_of_bounds(env)[0]
    obs, _, terminated, truncated, _ = env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))
    assert not terminated.any() and truncated[0] and not truncated[1:].any()
    check(obs)
    assert terrain.contains(robot.data.body_pos_w[0, state.anchor, :2])
    wrapped = RslRlVecEnvWrapper(env)
    check(wrapped.get_observations())
    baseline = PGMTStage1EnvCfg()
    assert not hasattr(baseline.scene, "height_scanner") and not hasattr(baseline.observations, "terrain_map")
    report.update(patches=800, regions=50, sensor_poses=len(cases) * env.num_envs, ray_count_per_scan=441,
                  checked_ray_cells=checked_rays, independent_physx_probes=probes, max_mesh_error_m=max_error,
                  max_physx_error_m=physics_error, shape=[env.num_envs, 21, 21], size_m=[2, 2],
                  height="ground_world_z - torso_world_z", ordering="row=y,col=x; both axes -1..+1 m",
                  update_period_s=env.step_dt, yaw_angles_deg=[0, 90, 180, -90], negative_world_heights=True,
                  full_800_patch_mesh_shared=True, cross_patch_level_family_scans=True,
                  birth_assignment_unchanged=True, edge_invalid_cells=invalid_counts,
                  scan_only_map_exit_does_not_truncate=True, robot_exit_truncates=True,
                  post_reset_scan_fresh=True, wrapper_observations=True, stage1_unchanged=True,
                  native_physics_steps=2, cases=cases,
                  controlled_scan_relocation=True, tracking_terminations_disabled_for_audit=True)


def audit_perception(env):
    import torch
    from pgmt_perception_checks import audit_perception_inputs
    from pmt_tasks.pgmt_perception import terrain_observation_contract
    assert env.cfg.pgmt_terrain_observation == terrain_observation_contract()
    # Disable stochastic augmentation for geometry checks; PMT sign/fill remain.
    env.cfg.pgmt_terrain_observation = terrain_observation_contract(augmentation_enabled=False)
    audit_scan(env)
    # Match the native runner: after inference-mode steps, all simulator resets
    # and sensor reads stay in that context (command metrics are inference tensors).
    with torch.inference_mode():
        audit_perception_inputs(env, report, output, write_progress)


def audit_resume(env):
    import math
    import torch
    from motion_tracking_rl import compat  # noqa: F401
    from motion_tracking_rl.runners import OnPolicyRunner
    from pmt_tasks.agent_cfgs.pgmt import PGMTStage1PPORunnerCfg
    from pmt_tasks.isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from pmt_tasks.pgmt_curriculum import SAMPLER_COUNTS
    from pmt_tasks.pgmt_runtime import get_pgmt_state

    def exact(actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), atol=0, rtol=0)
        elif isinstance(expected, dict):
            assert actual.keys() == expected.keys()
            for key in expected:
                exact(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            assert len(actual) == len(expected)
            for a, b in zip(actual, expected):
                exact(a, b)
        else:
            assert actual == expected, (actual, expected)

    source = Path(args.checkpoint).resolve()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    loaded = torch.load(source, map_location="cpu", weights_only=False)
    assert "environment_state_dict" in loaded and loaded["environment_state_dict"]["curriculum"]["levels"].numel() == env.num_envs
    wrapped = RslRlVecEnvWrapper(env)
    assert "height_scanner" not in env.scene.sensors
    assert set(wrapped.get_observations().keys()) == {"proprio", "proprio_history", "command_window", "critic"}
    cfg = PGMTStage1PPORunnerCfg()
    cfg.seed, cfg.device, cfg.logger, cfg.save_interval = args.seed, env.device, "tensorboard", 50
    runner = OnPolicyRunner(wrapped, cfg.to_dict(), log_dir=str(output / "ppo"), device=env.device)
    cmd, terrain, course = env.command_manager.get_term("motion"), env.scene.terrain, env._pgmt_terrain_curriculum
    state = get_pgmt_state(env)
    ids = torch.arange(env.num_envs, device=env.device)
    counters = dict(control_steps=0, done_events=0, sampler_outcomes=0, reset_poses=0,
                    reference_drift_checks=0, contact_freshness_checks=0, ppo_updates=0)
    maximum_pose_error, minimum_gap = 0., float("inf")
    latest_outcome = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    old_reset, old_outcome, old_step, old_log = cmd._reset_robot_state, cmd._record_outcomes, wrapped.step, runner.log

    def audited_reset(selected):
        nonlocal maximum_pose_error, minimum_gap
        old_reset(selected)
        error, gap, _ = check_pose(env, selected)
        maximum_pose_error, minimum_gap = max(maximum_pose_error, error), min(minimum_gap, gap)
        counters["reset_poses"] += len(selected)

    def audited_outcome(selected, terminated):
        eligible = course.sampler_pending[selected] & (course.settled_step[selected] == env.common_step_counter)
        assert latest_outcome[selected[eligible]].ne(env.common_step_counter).all()
        old_outcome(selected, terminated)
        latest_outcome[selected[eligible]] = env.common_step_counter
        counters["sampler_outcomes"] += int(eligible.sum())

    def audited_step(actions):
        before_q, before_t = cmd.reference_rotation.clone(), cmd.reference_translation.clone()
        obs, reward, done, extras = old_step(actions)
        survivors = ~done.bool()
        exact(cmd.reference_rotation[survivors], before_q[survivors])
        exact(cmd.reference_translation[survivors], before_t[survivors])
        contacts = state.reference_contacts()
        exact(contacts["motion_ids"], cmd.motion_ids)
        exact(contacts["frame_ids"], cmd.frame_ids)
        assert all(v.isfinite().all() for v in obs.values()) and reward.isfinite().all()
        torch.testing.assert_close(extras["reward_groups"].sum(-1), reward, atol=2e-6, rtol=2e-5)
        counters["control_steps"] += 1
        counters["done_events"] += int(done.sum())
        counters["reference_drift_checks"] += int(survivors.sum())
        counters["contact_freshness_checks"] += env.num_envs
        return obs, reward, done, extras

    def audited_log(locs, *positional, **keywords):
        old_log(locs, *positional, **keywords)
        assert all(math.isfinite(float(v)) for v in locs["loss_dict"].values())
        assert all(p.isfinite().all() for p in runner.alg.policy.parameters())
        counters["ppo_updates"] += 1
        if counters["ppo_updates"] % 5 == 0:
            write_progress(status="resumed_ppo", checks=dict(counters), iteration=int(locs["it"]))

    cmd._reset_robot_state, cmd._record_outcomes, wrapped.step, runner.log = audited_reset, audited_outcome, audited_step, audited_log

    def restore_and_check(path, saved):
        before_outcomes = counters["sampler_outcomes"]
        runner.load(str(path), load_optimizer=True, map_location=env.device, load_environment=True)
        runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]["lr"]
        assert runner.current_learning_iteration == saved["iter"]
        exact(runner.alg.policy.state_dict(), saved["model_state_dict"])
        exact(runner.alg.optimizer.state_dict(), saved["optimizer_state_dict"])
        exact(wrapped.get_checkpoint_state(), saved["environment_state_dict"])
        assert counters["sampler_outcomes"] == before_outcomes
        assert cmd.frame_ids.eq(0).all() and course.active.all() and not course.sampler_pending.any()
        exact(terrain.env_origins, terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types])
        check_pose(env, ids)
        metadata = runner.alg.policy.get_checkpoint_metadata()
        assert metadata["signature"] == saved["policy_metadata"]["signature"]
        assert metadata["obs_schema"]["hash"] == saved["policy_metadata"]["obs_schema"]["hash"]

    try:
        restore_and_check(source, loaded)
        restored_initial = wrapped.get_checkpoint_state()
        (output / "restored_assignments.json").write_text(json.dumps(dict(
            levels=terrain.terrain_levels.tolist(), families=terrain.terrain_types.tolist(),
            patch_ids=cmd.placement_patch_ids.tolist(), xy=cmd.placement_xy.tolist(),
            yaw=cmd.placement_yaw.tolist(), curriculum_counts=course.counts.sum((0, 1)).tolist()), indent=2) + "\n")
        write_progress(status="original_checkpoint_restored", source_checkpoint=str(source), source_sha256=source_hash,
                       loaded_iteration=runner.current_learning_iteration,
                       curriculum_outcomes=course.counts.sum((0, 1)).tolist())
        # An explicit manual reset must neither score interrupted episodes nor
        # alter the restored adaptive-sampling EMA/pending statistics.
        # The runner owns simulation buffers under inference_mode. Its reset
        # path must keep that context, including this deliberate manual reset.
        with torch.inference_mode():
            env.reset()
        exact(wrapped.get_checkpoint_state(), restored_initial)
        bad = copy.deepcopy(restored_initial)
        bad["curriculum"]["identity"]["reset"]["boundary_margin"] += .01
        try:
            wrapped.load_checkpoint_state(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Mismatched v6 reset settings were not rejected")
        exact(wrapped.get_checkpoint_state(), restored_initial)
        before = {name: value.detach().clone() for name, value in runner.alg.policy.named_parameters()}
        runner.learn(num_learning_iterations=25, init_at_random_ep_len=False)
        change = max(float((value.detach() - before[name]).abs().max()) for name, value in runner.alg.policy.named_parameters())
        assert change > 0 and counters["done_events"] > 0
        assert counters["done_events"] == counters["sampler_outcomes"]
        delta_counts = course.counts.cpu() - restored_initial["curriculum"]["counts"]
        assert delta_counts.sum() == counters["done_events"]
        roundtrip = output / "resumed_roundtrip.pt"
        runner.save(str(roundtrip))
        saved = torch.load(roundtrip, map_location="cpu", weights_only=False)
        # Corrupt live state deliberately, then require bit-exact restoration.
        with torch.inference_mode():
            terrain.terrain_levels.zero_()
            terrain.terrain_types.fill_(4)
            course.counts.zero_()
            course.promotions.zero_()
            course.demotions.zero_()
            for key in SAMPLER_COUNTS:
                getattr(cmd.sampler, key).fill_(19.)
            cmd.sampler._steps_since_update = 0
            runner.current_learning_iteration = 0
            next(runner.alg.policy.parameters()).add_(.01)
        restore_and_check(roundtrip, saved)
        # Native load starts fresh simulator trajectories at frame zero. It
        # does not resume the interrupted rollout's simulator/RNG state.
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
        assert counters["done_events"] == counters["sampler_outcomes"]
        assert (course.counts.cpu() - restored_initial["curriculum"]["counts"]).sum() == counters["done_events"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
        report.update(source_checkpoint=str(source), source_sha256=source_hash, source_unchanged=True,
                      loaded_iteration=int(loaded["iter"]), final_iteration=runner.current_learning_iteration,
                      roundtrip_checkpoint=str(roundtrip), checks=counters, maximum_reset_pose_error_m=maximum_pose_error,
                      minimum_reset_gap_m=minimum_gap, max_parameter_change=change,
                      original_curriculum_outcomes=restored_initial["curriculum"]["counts"].sum((0, 1)).tolist(),
                      subsequent_curriculum_outcomes=(course.counts.cpu() - restored_initial["curriculum"]["counts"]).sum((0, 1)).tolist(),
                      policy_optimizer_iteration_exact=True, curriculum_assignments_origins_statistics_exact=True,
                      sampler_ema_pending_counts_and_clock_exact=True, reset_random_xy_yaw_after_restore=True,
                      manual_and_resume_reset_not_counted=True, reset_identity_mismatch_rejected_before_mutation=True,
                      actual_ppo_after_restore=True, second_checkpoint_roundtrip=True,
                      interrupted_simulator_trajectory_restarted=True, bitwise_trajectory_continuation=False)
    finally:
        if runner.writer is not None:
            runner.writer.close()


def main():
    if args.isaacsim6_compat:
        from pmt_tasks.isaacsim6_legacy import install_rigid_scene_schema_bridge
        install_rigid_scene_schema_bridge()
    env = None
    try:
        env = make_env()
        globals()["audit_" + args.check](env)
        write_progress(passed=True, status="passed", finished_at=datetime.now().isoformat())
    except BaseException as error:
        write_progress(passed=False, status="failed", error=repr(error), finished_at=datetime.now().isoformat())
        raise
    finally:
        report_file.write_text(json.dumps(report, indent=2) + "\n")
        if env is not None:
            env.close()


from pmt_tasks.runtime_options import run_and_close

if __name__ == "__main__":
    run_and_close(main, app)
