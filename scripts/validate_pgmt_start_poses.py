"""Exhaustive frame-zero placement audit for every CSV-allowed motion/tile pair."""
import argparse
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default=str(ROOT / "logs/pgmt_p1_acceptance/start_poses"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=True)
report_file = output / "validation.json"
report_file.unlink(missing_ok=True)
app = AppLauncher(args, multi_gpu=False, fast_shutdown=True).app


def main():
    import math
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from pmt_tasks.env_cfgs.pgmt.p1_blind import PGMTBlindTerrainEnvCfg
    from pmt_tasks.pgmt_math import future_offsets
    from pmt_tasks.pgmt_reference import initial_transform, place_on_surface, transform_reference

    cfg = PGMTBlindTerrainEnvCfg()
    cfg.pmt_motion_paths = [str(ROOT / "assets/motions/pgmt_clipped_50hz_tcrs_v1")]
    cfg.__post_init__()
    cfg.scene.num_envs = 7
    cfg.seed = 721
    cfg.sim.device = cfg.commands.motion.storage_device = args.device
    env = ManagerBasedRLEnv(cfg=cfg)
    try:
        cmd, terrain = env.command_manager.get_term("motion"), env.scene.terrain
        count = cmd.data_store.num_motions
        mids = torch.arange(count, device=env.device)
        first = cmd._source_reference(mids, torch.zeros_like(mids))
        limits = env.scene["robot"].data.joint_pos_limits[0]
        bad_joint = ((first["joint_pos"] < limits[:, 0] - 1e-5)
                     | (first["joint_pos"] > limits[:, 1] + 1e-5)).any(-1)
        joint_errors = [Path(cmd.data_store.motions[i].source_file).name for i in bad_joint.nonzero().flatten().tolist()]
        failures, rows, retried = [], [], []
        min_gap, cases = float("inf"), 0
        # Three deterministic yaw angles cover orientations used by the physical
        # reset audit. Runtime also rejects unsafe placements at any sampled yaw.
        for level in range(10):
            for family in range(7):
                chosen = cmd.compatible_motions[family].nonzero().flatten()
                ref = {k: v[chosen] for k, v in first.items()}
                n = len(chosen)
                for yaw in (0., math.pi / 2, -3 * math.pi / 4):
                    xy = terrain.terrain_origins[level, family, :2].expand(n, -1)
                    rotation, translation = initial_transform(ref["body_pos"][:, 0], ref["body_quat"][:, 0],
                                                               xy, torch.full((n,), yaw, device=env.device))
                    translation, valid, gap = place_on_surface(
                        ref, rotation, translation, terrain.tile_bounds[level, family].expand(n, -1, -1),
                        terrain, cmd.collision_body_ids, cmd.collision_local_centres, cmd.collision_radii,
                        clearance=cmd.cfg.spawn_clearance, boundary_margin=cmd.cfg.boundary_margin)
                    future_ids = chosen[:, None].expand(-1, 8)
                    future_frames = future_offsets(device=env.device)[None].expand(n, -1)
                    future_frames = torch.minimum(future_frames, cmd.data_store.motion_lengths[future_ids] - 1)
                    future = cmd._source_reference(future_ids.reshape(-1), future_frames.reshape(-1))
                    future = {k: v.reshape(n, 8, *v.shape[1:]) for k, v in future.items()}
                    future_world = transform_reference(future, rotation, translation)
                    future_valid = terrain.contains(future_world["body_pos"][..., :2]).flatten(1).all(-1)
                    passed = valid & ~bad_joint[chosen] & future_valid
                    if passed.any():
                        min_gap = min(min_gap, float(gap[passed].min()))
                    cases += n
                    for j in (~passed).nonzero().flatten().tolist():
                        rejected = dict(motion=Path(cmd.data_store.motions[int(chosen[j])].source_file).name,
                                        family=cmd.terrain_families[family], level=level, yaw=yaw,
                                        collision_placement=bool(valid[j]), joint_limits=not bool(bad_joint[chosen[j]]),
                                        future_in_map=bool(future_valid[j]), min_gap_m=float(gap[j]))
                        # Exercise the real command's rejection/retry and physical
                        # reset for this exact motion; never drop it from the CSV.
                        terrain.terrain_types[0], terrain.terrain_levels[0] = family, level
                        terrain.env_origins[0] = terrain.terrain_origins[level, family]
                        cmd.motion_ids[0], cmd.frame_ids[0] = chosen[j], 0
                        try:
                            cmd._reset_robot_state(torch.tensor([0], device=env.device))
                            assert cmd.placement_min_gap[0] >= cmd.cfg.spawn_clearance - 1e-5
                            rejected.update(retry_passed=True, retry_attempts=int(cmd.placement_attempts[0]),
                                            accepted_min_gap_m=float(cmd.placement_min_gap[0]))
                            min_gap = min(min_gap, float(cmd.placement_min_gap[0]))
                            retried.append(rejected)
                        except (RuntimeError, ValueError, AssertionError) as error:
                            rejected["retry_error"] = repr(error)
                            failures.append(rejected)
                    rows.append(dict(level=level, family=cmd.terrain_families[family], yaw=yaw,
                                     motions=n, passed=int(passed.sum()), failed=int((~passed).sum())))
            print(f"[P1 START] L{level} complete; cases={cases}; failures={len(failures)}", flush=True)
        report = dict(passed=not failures and not joint_errors, motions=count, fixed_start_frame=0,
                      compatible_pairs_per_level=int(cmd.compatible_motions.sum()), placement_cases=cases,
                      minimum_collision_gap_m=min_gap, joint_limit_errors=joint_errors,
                      failures=failures, rejected_candidates_retried=retried, rows=rows,
                      method="conservative sphere cover against actual cached triangles; real command retries rejected candidates",
                      xy_jitter=0., random_start_frames=False,
                      scope="all CSV-allowed motions, 70 tiles, three yaw angles; physical overlap checked separately")
        report_file.write_text(json.dumps(report, indent=2) + "\n")
        print("PGMT START POSES: " + json.dumps({k: v for k, v in report.items() if k not in ("rows", "failures")}), flush=True)
        assert report["passed"], f"{len(failures)} unsafe start placements; see {report_file}"
    finally:
        env.close()


try:
    main()
except Exception as error:
    if not report_file.exists():
        report_file.write_text(json.dumps(dict(passed=False, error=repr(error))) + "\n")
    traceback.print_exc()
    raise
finally:
    app.close()
