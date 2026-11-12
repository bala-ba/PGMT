"""Finite native Isaac audit of tracking deadzones; no PPO or checkpoint load."""
import argparse
from copy import deepcopy
from dataclasses import fields
from datetime import datetime
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.environ.get("PMT_PGMT_PHYSICS_WORKER") != "1":
    from pgmt_process_guard import run_guarded
    raise SystemExit(run_guarded([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], ROOT))

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--log_dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
output = Path(args.log_dir).resolve()
output.mkdir(parents=True, exist_ok=True)
status_path = output / "training_status.json"
if status_path.exists():
    parser.error(f"Refusing to overwrite {output}")
report = dict(status="launching", pid=os.getpid(), started_at=datetime.now().isoformat(), arguments=vars(args))
def save(**values):
    report.update(values)
    temp = status_path.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n")
    temp.replace(status_path)
save()
from pmt_tasks.isaacsim6_legacy import install_tensor_import_bridge
install_tensor_import_bridge()
app = AppLauncher(args, multi_gpu=False, fast_shutdown=True).app


def numpy_errors(physical, reference, anchor, groups):
    """Independent float64 rotation-matrix oracle (production uses quaternions)."""
    import numpy as np
    def local(data):
        q = data["body_quat"].cpu().numpy().astype(np.float64)
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        w, x, y, z = np.moveaxis(q, -1, 0)
        rotation = np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                             2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                             2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)
        yaw = np.arctan2(rotation[:, anchor, 1, 0], rotation[:, anchor, 0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        inverse_yaw = np.zeros((len(q), 3, 3))
        inverse_yaw[:, 0, 0] = inverse_yaw[:, 1, 1] = c
        inverse_yaw[:, 0, 1], inverse_yaw[:, 1, 0], inverse_yaw[:, 2, 2] = s, -s, 1.
        position = data["body_pos"].cpu().numpy().astype(np.float64)
        return (np.einsum("nij,nbj->nbi", inverse_yaw, position-position[:, anchor, None]),
                np.einsum("nij,nbjk->nbik", inverse_yaw, rotation))
    p, r = local(physical)
    target_p, target_r = local(reference)
    delta = r @ np.swapaxes(target_r, -1, -2)
    body = groups["lower"].cpu().numpy()
    joints = groups["lower_joints"].cpu().numpy()
    return dict(lower_body_position=np.linalg.norm(p-target_p, axis=-1)[:, body],
        lower_body_orientation=np.arccos(np.clip((np.trace(delta, axis1=-2, axis2=-1)-1)/2, -1, 1))[:, body],
        lower_joint_position=np.abs(physical["joint_pos"].cpu().numpy().astype(np.float64)
                                    - reference["joint_pos"].cpu().numpy().astype(np.float64))[:, joints])


def main():
    from pmt_tasks.isaacsim6_legacy import install_rigid_scene_schema_bridge
    install_rigid_scene_schema_bridge()
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import TerminationTermCfg
    from isaaclab.utils.io import dump_yaml
    from pmt_tasks.builder import build_env_cfg
    from pmt_tasks.env_cfgs.pgmt.terrain_rewards import PGMTContactForceRewardsCfg
    from pmt_tasks.pgmt_math import REWARD_WEIGHTS, reward_terms
    from pmt_tasks.pgmt_runtime import get_pgmt_state, reward_groups_from_manager
    from pmt_tasks.pgmt_tracking_relaxation import RELAXATION_TERMS

    cfg = build_env_cfg("pgmt_stage2", overrides=["tracking_relaxation.enabled=true"])
    cfg.pmt_motion_paths = [str(ROOT / "assets/motions/pgmt_clipped_50hz_tcrs_v1/walk1_subject1_clip0.npz")]
    cfg.__post_init__()
    cfg.rewards, cfg.pgmt_reward_groups = PGMTContactForceRewardsCfg(), ("upper", "lower", "terrain", "auxiliary")
    cfg.scene.num_envs, cfg.seed = 5, 1014
    cfg.sim.device = cfg.commands.motion.storage_device = args.device
    for name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
        setattr(cfg.events, name, None)
    for name in vars(cfg.terminations):
        if name != "map_out_of_bounds":
            setattr(cfg.terminations, name, None)
    done = torch.zeros(5, dtype=torch.bool, device=args.device)
    cfg.terminations.audit_done = TerminationTermCfg(func=lambda env: done, time_out=True)
    old = cfg.scene.robot.spawn
    shared = {f.name: deepcopy(getattr(old, f.name)) for f in fields(sim_utils.UsdFileCfg)
              if f.name not in ("func", "usd_path") and hasattr(old, f.name)}
    cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
        usd_path=str(ROOT / "logs/pgmt_p1_acceptance/native_robot/main.usd"), **shared)
    dump_yaml(str(output / "env.yaml"), cfg)
    save(status="building_environment")
    env = ManagerBasedRLEnv(cfg)
    try:
        env.reset()
        state, terrain, robot = get_pgmt_state(env), env.scene.terrain, env.scene["robot"]
        command = env.command_manager.get_term("motion")
        assert state.tracking_relaxation.config["enabled"]
        assert len(env.reward_manager.active_terms) == 27
        (output / "relaxation_contract.json").write_text(json.dumps(state.tracking_relaxation.contract(), indent=2) + "\n")
        original = state._tracking_relaxation_sample
        captured = {}
        def capture(physical):
            ref = state.reference()
            captured["physical"] = {k: v.clone() for k, v in physical.items()}
            captured["reference"] = {k: v.clone() for k, v in ref.items()}
            captured["strict"] = reward_terms(physical, ref, state.qtarget(), state.previous_target,
                state.previous_velocity, state.previous_ref_velocity,
                env.scene["contact_forces"].data.net_forces_w[:, state.contact_order],
                robot.data.soft_joint_pos_limits, state.groups, anchor=state.anchor, dt=env.step_dt, config=state.config)
            return original(physical)
        state._tracking_relaxation_sample = capture
        steps, max_error, regions = 0, 0., set()
        measurements = []
        actions = torch.zeros(5, env.action_manager.total_action_dim, device=env.device)
        def step():
            nonlocal steps, max_error
            result = env.step(actions)
            sample = state.tracking_relaxation_state(for_reward=True)
            raw = numpy_errors(captured["physical"], captured["reference"], state.anchor, state.groups)
            xy = sample["anchor_xy_w"].cpu().numpy()
            # Rectangle lookup from asset records, independent of bucketize and assignment.
            expected_scale = np.zeros(5)
            for i, point in enumerate(xy):
                matches = [t for t in terrain.manifest["tiles"]
                           if (point >= np.asarray(t["bounds"])[0]).all() and (point < np.asarray(t["bounds"])[1]).all()]
                assert len(matches) <= 1
                if matches:
                    t = matches[0]
                    assert int(sample["tile_level"][i]) == t["level"]
                    assert int(sample["family_index"][i]) == t["family_index"]
                    regions.add((t["level"], t["family"]))
                    expected_scale[i] = t["level"] / 9 if t["family"] in ("stairs", "slope", "boxes") else 0.
                else:
                    assert not sample["tile_valid"][i] and int(sample["tile_level"][i]) == -1
            for term, maximum in RELAXATION_TERMS.items():
                actual_raw = sample[f"{term}_raw_error"].cpu().numpy()
                max_error = max(max_error, float(np.max(np.abs(raw[term]-actual_raw))))
                np.testing.assert_allclose(actual_raw, raw[term], atol=2e-5, rtol=2e-5)
                margin = np.broadcast_to(expected_scale[:, None] * state.tracking_relaxation.config[maximum], (5, 12))
                residual = np.maximum(raw[term] - margin, 0.)
                expected = np.exp(-np.mean(residual**2, axis=-1) / state.config[term.removeprefix("lower_")+"_sigma"]**2)
                np.testing.assert_allclose(sample[f"{term}_margin"].cpu(), margin, atol=1e-7, rtol=1e-6)
                np.testing.assert_allclose(sample[f"{term}_relaxed_error"].cpu(), residual, atol=2e-5, rtol=2e-5)
                np.testing.assert_allclose(state.rewards()[term].cpu(), expected, atol=2e-5, rtol=2e-5)
                index = env.reward_manager.active_terms.index(term)
                np.testing.assert_allclose(env.reward_manager._step_reward[:, index].cpu() * env.step_dt,
                                            expected * REWARD_WEIGHTS[term] * env.step_dt, atol=1e-6, rtol=2e-5)
                assert (sample[f"{term}_relaxed_error"] >= 0).all()
                measurements.append(dict(step=int(env.common_step_counter), term=term,
                    levels=sample["tile_level"].cpu().tolist(), families=sample["family_index"].cpu().tolist(),
                    inside_fraction=(sample[f"{term}_raw_error"] <= sample[f"{term}_margin"]).float().mean(-1).cpu().tolist(),
                    rate_gain=(sample[f"{term}_relaxed_rate"]-sample[f"{term}_strict_rate"]).cpu().tolist()))
            for term in captured["strict"].keys() - RELAXATION_TERMS.keys():
                torch.testing.assert_close(state.rewards()[term], captured["strict"][term], rtol=0, atol=0)
            grouped = reward_groups_from_manager(env.reward_manager, env.step_dt, cfg.pgmt_reward_groups)
            torch.testing.assert_close(grouped.sum(-1), result[1])
            state.rewards(); state.rewards()
            for key, value in sample.items():
                torch.testing.assert_close(state.tracking_relaxation_state(for_reward=True)[key], value, equal_nan=True)
            steps += 1
            return result, sample

        # Actual random placements and physics steps in all five families at every level.
        for level in range(10):
            terrain.terrain_levels[:] = level
            terrain.terrain_types[:] = torch.arange(5, device=env.device)
            terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
            env.reset()
            _, sample = step()
            assert sample["tile_level"].eq(level).all(), sample["tile_level"]
            assert sample["family_index"].tolist() == list(range(5))
        assert len(regions) == 50
        # A physical crossing probe: translate only robot root, keep reference and birth unchanged.
        ids = torch.tensor([1], device=env.device)
        reference_translation = command.reference_translation.clone()
        birth = terrain.terrain_levels.clone(), terrain.terrain_types.clone()
        target_tile = next(t for t in terrain.manifest["tiles"] if t["level"] == 3 and t["family"] == "slope")
        root = robot.data.root_link_state_w[ids].clone()
        delta = root.new_tensor(target_tile["center"][:2]) - robot.data.body_pos_w[ids, state.anchor, :2]
        root[:, :2] += delta
        root[:, 2] = 6.  # Avoid collision impulses during this coordinate/lifecycle probe.
        root[:, 7:] = 0.
        robot.write_root_link_state_to_sim(root, env_ids=ids)
        env.sim.forward()
        _, crossing = step()
        assert crossing["tile_level"][1] == 3 and crossing["birth_level"][1] == 9
        torch.testing.assert_close(command.reference_translation, reference_translation, rtol=0, atol=0)
        torch.testing.assert_close(terrain.terrain_levels, birth[0])
        torch.testing.assert_close(terrain.terrain_types, birth[1])
        # Native partial reset must preserve the terminal reward snapshot.
        frames = command.frame_ids.clone()
        done[1] = True
        _, terminal = step()
        done.zero_()
        assert terminal["reference_frame_ids"][1] == frames[1]
        live = state.tracking_relaxation_state()
        assert not live["sample_valid"][1] and live["lower_body_position_margin"][1].eq(0).all()
        assert live["sample_valid"][[0, 2, 3, 4]].all()
        step()
        # Real map-exit truncation, using the existing full-body bounds condition.
        root = robot.data.root_link_state_w[ids].clone()
        root[:, 0], root[:, 2] = terrain.tile_map.upper[0] + 5., 6.
        root[:, 7:] = 0.
        robot.write_root_link_state_to_sim(root, env_ids=ids)
        env.sim.forward()
        result, outside = step()
        assert result[3][1] and not outside["tile_valid"][1]
        assert all(outside[f"{term}_margin"][1].eq(0).all() for term in RELAXATION_TERMS)
        assert not state.tracking_relaxation_state()["sample_valid"][1]
        (output / "measurements.json").write_text(json.dumps(measurements, indent=2) + "\n")
        save(status="passed", control_steps=steps, regions_checked=len(regions),
             max_raw_error_difference=max_error, active_rewards=env.reward_manager.active_terms,
             crossing_birth_reference_preserved=True, terminal_reset_checked=True,
             map_exit_truncation_checked=True, duplicate_reads_checked=True,
             original_other_18_terms_unchanged=True, weighted_dt_and_four_group_sums_checked=True)
        print("PGMT_TRACKING_RELAXATION_PASSED", json.dumps(report), flush=True)
    except BaseException as error:
        save(status="failed", error=repr(error))
        raise
    finally:
        env.close()


from pmt_tasks.runtime_options import run_and_close
run_and_close(main, app)
