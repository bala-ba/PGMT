"""Finite native Stage2 termination/oracle/reset audit; no PPO update."""
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
status_file = output / "training_status.json"
if status_file.exists():
    parser.error("Refusing to overwrite an existing audit")
report = dict(status="launching", pid=os.getpid(), started_at=datetime.now().isoformat())
def save(**values):
    report.update(values)
    temporary = status_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(status_file)
save()
from pmt_tasks.isaacsim6_legacy import install_tensor_import_bridge
install_tensor_import_bridge()
app = AppLauncher(args, multi_gpu=False, fast_shutdown=True).app


def main():
    from pmt_tasks.isaacsim6_legacy import install_rigid_scene_schema_bridge
    install_rigid_scene_schema_bridge()
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.io import dump_yaml
    from pmt_tasks.builder import build_env_cfg
    from pmt_tasks.isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from pmt_tasks.pgmt_runtime import get_pgmt_state
    from pmt_tasks.mdp.pgmt_terminations import SustainedTrackingFailure
    from pmt_tasks.pgmt_terminations import pgmt_time_out
    from pmt_tasks.pgmt_math import multiply
    from scripts.pgmt_stage2_checks import RolloutAudit

    stage1 = build_env_cfg("pgmt_stage1")
    assert stage1.episode_length_s == 15.
    assert stage1.terminations.time_out.func.__name__ == "time_out"
    assert stage1.terminations.anchor_pos.func.__name__ == "bad_anchor_pos_z_only"
    assert stage1.terminations.ee_body_pos.params["body_names"] == [
        "left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    for task in ("pgmt_stage2", "pgmt_stage2_terrain"):
        cfg = build_env_cfg(task)
        assert cfg.commands.motion.placement_patch_grid == (2, 2)
        assert cfg.episode_length_s == 10. and round(10 / (cfg.sim.dt * cfg.decimation)) == 500
        assert cfg.terminations.time_out.func is pgmt_time_out and cfg.terminations.time_out.time_out
        assert cfg.terminations.time_out.params == {"command_name": "motion", "boundary_tolerance_steps": 2}
        for name in ("root_xy", "anchor_pos", "ee_body_pos", "anchor_ori"):
            term = getattr(cfg.terminations, name)
            assert term.func is SustainedTrackingFailure and term.params["duration_s"] == .2
        assert cfg.terminations.anchor_ori.params["predicate"] is stage1.terminations.anchor_ori.func
        assert cfg.terminations.anchor_ori.params["predicate_params"] == stage1.terminations.anchor_ori.params
        assert cfg.terminations.root_xy.params["predicate_params"]["threshold"] == 1.
        assert cfg.terminations.ee_body_pos.params["predicate_params"]["body_names"] == ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    cfg.pmt_motion_paths = [str(ROOT / "assets/motions/pgmt_clipped_50hz_tcrs_v1/walk1_subject1_clip0.npz")]
    cfg.__post_init__()
    cfg.scene.num_envs, cfg.seed = 5, 1014
    cfg.sim.device = cfg.commands.motion.storage_device = args.device
    for name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
        setattr(cfg.events, name, None)
    old = cfg.scene.robot.spawn
    shared = {f.name: deepcopy(getattr(old, f.name)) for f in fields(sim_utils.UsdFileCfg)
              if f.name not in ("func", "usd_path") and hasattr(old, f.name)}
    cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
        usd_path=str(ROOT / "logs/pgmt_p1_acceptance/native_robot/main.usd"), **shared)
    dump_yaml(str(output / "env.yaml"), cfg)
    save(status="building_environment")
    env = ManagerBasedRLEnv(cfg)
    try:
        wrapped = RslRlVecEnvWrapper(env)
        wrapped.get_observations()
        state, cmd, robot = get_pgmt_state(env), env.command_manager.get_term("motion"), env.scene["robot"]
        terrain, manager, course = env.scene.terrain, env.termination_manager, env._pgmt_terrain_curriculum
        assert cmd.spawn_patch_ids.tolist() == [5, 6, 9, 10]
        assert course.identity["reset"]["placement_patch_grid"] == [2, 2]
        audit = RolloutAudit(wrapped)
        names = list(cmd.cfg.body_names)
        full_names = list(robot.body_names)
        sources = []
        for motion in cmd.full_reference_store.motions:
            with np.load(motion.source_file, allow_pickle=False) as data:
                sources.append(data["body_pos_w"].copy())
        root, torso = names.index("pelvis"), names.index("torso_link")
        wrists = [names.index(n) for n in ("left_wrist_yaw_link", "right_wrist_yaw_link")]
        ids = torch.arange(env.num_envs, device=env.device)
        records, regions, probes = [], set(), {}
        delayed_names = ("root_xy", "anchor_pos", "ee_body_pos", "anchor_ori")
        assert all(manager.get_term_cfg(name).func.timer.required_steps == 10 for name in delayed_names)

        def oracle(tag):
            physical = cmd.robot_body_pos_w.detach().cpu().numpy().astype(np.float64)
            # Read the selected NPZ frame directly, independent of command buffers.
            original = np.stack([sources[int(m)][int(f)]
                                 for m, f in zip(cmd.motion_ids, cmd.frame_ids)]).astype(np.float64)
            source = original[:, [full_names.index(n) for n in names]]
            np.testing.assert_allclose(cmd.source_body_pos.cpu().numpy(), source, atol=0, rtol=0)
            placed = cmd.body_pos_w.detach().cpu().numpy().astype(np.float64)
            ground = terrain.query(cmd.robot_body_pos_w[:, torso, :2])
            height, valid = ground["height"].cpu().numpy().astype(np.float64), ground["valid"].cpu().numpy()
            errors = dict(root_xy=np.linalg.norm(physical[:, root, :2]-placed[:, root, :2], axis=-1),
                anchor_pos=np.abs(physical[:, torso, 2] - (height + source[:, torso, 2])),
                ee_body_pos=np.max(np.abs((physical[:, wrists, 2]-physical[:, root, None, 2]) -
                                         (source[:, wrists, 2]-source[:, root, None, 2])), axis=-1))
            aq = cmd.robot_anchor_quat_w.cpu().numpy().astype(np.float64)
            rq = cmd.anchor_quat_w.cpu().numpy().astype(np.float64)
            errors["anchor_ori"] = np.abs(2 * np.sum(aq[:, 1:3]**2-rq[:, 1:3]**2, axis=-1))
            expected = {k: errors[k] > lim for k, lim in
                        (("root_xy", 1.), ("anchor_pos", .4), ("ee_body_pos", .4), ("anchor_ori", .8))}
            expected["anchor_pos"] &= valid
            steps = env.episode_length_buf.cpu().numpy()
            finished = np.array([int(f) >= len(sources[int(m)]) - 1
                                 for m, f in zip(cmd.motion_ids, cmd.frame_ids)])
            expected["time_out"] = (steps >= 500) | ((steps >= 498) & finished)
            translation = cmd.reference_translation.clone()
            # Compare raw predicates with an independent NPZ/NumPy oracle.
            # Timers use the actual control-step clock and are checked separately.
            for name in delayed_names:
                params = manager.get_term_cfg(name).params
                raw = params["predicate"](env, **params["predicate_params"])
                np.testing.assert_array_equal(raw.cpu().numpy(), expected[name], err_msg=name)
            for _ in range(2):
                counts = {name: manager.get_term_cfg(name).func.timer.counts.clone() for name in delayed_names}
                manager.compute()
                for name in delayed_names:
                    timer = manager.get_term_cfg(name).func.timer
                    torch.testing.assert_close(timer.counts, counts[name], rtol=0, atol=0)
                    torch.testing.assert_close(manager.get_term(name), counts[name] >= 10)
                np.testing.assert_array_equal(manager.get_term("time_out").cpu().numpy(), expected["time_out"])
            torch.testing.assert_close(cmd.reference_translation, translation, rtol=0, atol=0)
            records.append(dict(tag=tag, frame=cmd.frame_ids.tolist(), episode_steps=steps.tolist(), ground=height.tolist(),
                errors={k:v.tolist() for k,v in errors.items()}, flags={k:v.tolist() for k,v in expected.items()}))
            return expected

        @torch.inference_mode()
        def reset(level=0, family=None):
            terrain.terrain_levels[:] = level
            terrain.terrain_types[:] = ids if family is None else family
            terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
            # This is a manual audit reset, not the outcome of the preceding predicate probe.
            manager._terminated_buf.zero_(); manager._truncated_buf.zero_(); manager._term_dones.zero_()
            env.reset()
            assert torch.isin(cmd.placement_patch_ids, cmd.spawn_patch_ids).all()
            center = (terrain.bounds_for(ids)[:, 0] + terrain.bounds_for(ids)[:, 1]) / 2
            assert ((cmd.placement_xy - center).abs() <= 5.75).all()
            records.append(dict(tag="central_spawn", level=level, family=terrain.terrain_types.tolist(),
                patch_ids=cmd.placement_patch_ids.tolist(), xy=cmd.placement_xy.tolist(),
                yaw=cmd.placement_yaw.tolist(), attempts=cmd.placement_attempts.tolist()))

        def step():
            with torch.inference_mode():
                return wrapped.step(torch.zeros(env.num_envs, 29, device=env.device))

        def sustained_probe(name, *, fixed_frame=None):
            # Hold an actual physical pose before each step to isolate persistence
            # from the untrained policy's ability to maintain that pose.
            root_pose = robot.data.root_link_state_w[:1].clone()
            root_pose[:, 7:] = 0
            joint_pose = robot.data.joint_pos[:1].clone()
            timer = manager.get_term_cfg(name).func.timer
            assert timer.counts[0] == 0
            before_counts, before_sampler = course.counts.sum().item(), audit.checks["sampler_outcomes"]
            settled = 0
            for tick in range(1, 11):
                robot.write_root_link_state_to_sim(root_pose, env_ids=ids[:1])
                robot.write_joint_state_to_sim(joint_pose, torch.zeros_like(joint_pose), env_ids=ids[:1])
                if fixed_frame is not None:
                    cmd.frame_ids[0] = fixed_frame
                    cmd._update_motion_buffers()
                    cmd._compute_relative_poses()
                env.sim.forward()
                result = step()
                settled += int(result[2].sum())
                assert bool(result[2][0]) == (tick == 10), (name, tick)
                assert bool(manager.get_term(name)[0]) == (tick == 10), (name, tick)
                if tick < 10:
                    assert timer.counts[0] == tick, (name, tick, timer.counts.tolist())
                    term = manager.get_term_cfg(name)
                    for _ in range(3):
                        assert not term.func(env, **term.params)[0]
                    assert timer.counts[0] == tick
            assert course.last_outcome[0] == 2 and terrain.terrain_levels[0] == 2
            assert course.counts.sum().item() - before_counts == settled
            assert audit.checks["sampler_outcomes"] - before_sampler == settled
            assert all(manager.get_term_cfg(key).func.timer.counts[0] == 0 for key in delayed_names)
            probes[f"{name}_10_physics_steps_then_reset_once"] = True

        for level in range(10):
            reset(level)
            oracle(f"initial_L{level}")
            for family in range(5):
                regions.add((level, family))
            step()
            oracle(f"after_physics_L{level}")
        assert len(regions) == 50

        # Actual XY enters another tile; reference and birth allocation stay fixed.
        reset(9)
        reference_translation = cmd.reference_translation.clone()
        birth_levels, birth_types = terrain.terrain_levels.clone(), terrain.terrain_types.clone()
        target = next(t for t in terrain.manifest["tiles"] if t["level"] == 3 and t["family"] == "slope")
        root_state = robot.data.root_link_state_w.clone()
        target_xy = root_state.new_tensor(target["center"][:2])
        root_state[1, :2] += target_xy - cmd.robot_body_pos_w[1, torso, :2]
        ground = terrain.query(target_xy[None])["height"][0]
        root_state[1, 2] += ground + cmd.source_body_pos[1, torso, 2] - cmd.robot_body_pos_w[1, torso, 2]
        robot.write_root_link_state_to_sim(root_state)
        env.sim.forward()
        flags = oracle("birth_stairs_L9_current_slope_L3")
        assert flags["root_xy"][1] and not flags["anchor_pos"][1]
        torch.testing.assert_close(cmd.reference_translation, reference_translation, rtol=0, atol=0)
        torch.testing.assert_close(terrain.terrain_levels, birth_levels)
        torch.testing.assert_close(terrain.terrain_types, birth_types)
        probes["cross_tile_actual_ground_fixed_reference_birth"] = True

        # Trigger the real root failure, then verify ordinary reset/sampler/course handling.
        reset(3, 0)
        root_state = robot.data.root_link_state_w.clone()
        root_state[0, 0] += 1.3
        root_state[:, 7:] = 0
        robot.write_root_link_state_to_sim(root_state)
        env.sim.forward()
        assert oracle("root_xy_probe")["root_xy"][0]
        sustained_probe("root_xy")
        probes["root_failure_demotes_once"] = True

        reset(3, 0)
        root_state = robot.data.root_link_state_w.clone()
        root_state[0, 2] += .6
        root_state[:, 7:] = 0
        robot.write_root_link_state_to_sim(root_state)
        env.sim.forward()
        flags = oracle("torso_height_probe")
        assert flags["anchor_pos"][0] and not flags["ee_body_pos"][0]
        sustained_probe("anchor_pos")
        probes["global_z_changes_torso_not_relative_wrists"] = True

        reset(3, 0)
        # Keep the real initialized arm pose while advancing only its reference
        # to an existing low-wrist frame. This produces a real >0.4 m mismatch
        # without assuming a particular shoulder angle can reach that height.
        source = sources[int(cmd.motion_ids[0])]
        wrist_full, root_full = full_names.index("left_wrist_yaw_link"), full_names.index("pelvis")
        relative = source[:, wrist_full, 2] - source[:, root_full, 2]
        cmd.frame_ids[0] = int(relative.argmin())
        cmd._update_motion_buffers()
        cmd._compute_relative_poses()
        flags = oracle("physical_wrist_reference_frame_mismatch")
        assert flags["ee_body_pos"][0]
        sustained_probe("ee_body_pos", fixed_frame=int(cmd.frame_ids[0]))
        probes["physical_wrist_failure_on_current_reference_frame"] = True

        reset(3, 0)
        root_state = robot.data.root_link_state_w.clone()
        root_state[0, 2] += 2.
        root_state[0, 3:7] = multiply(root_state.new_tensor([0., 1., 0., 0.]), root_state[0, 3:7])
        root_state[:, 7:] = 0
        robot.write_root_link_state_to_sim(root_state)
        env.sim.forward()
        assert oracle("torso_orientation_probe")["anchor_ori"][0]
        sustained_probe("anchor_ori")

        # Truncations retain immediate behavior even when a tracking error has
        # not accumulated enough consecutive steps to become a failure.
        reset(3, 0)
        root_state = robot.data.root_link_state_w.clone()
        root_state[0, :2] = terrain.tile_map.lower - 2.
        robot.write_root_link_state_to_sim(root_state)
        env.sim.forward()
        result = step()
        assert result[2][0] and result[3]["time_outs"][0] and manager.get_term("map_out_of_bounds")[0]
        assert not manager.terminated[0]
        probes["map_exit_immediate_before_failure_delay"] = True

        reset(3, 0)
        cmd.frame_ids[0] = cmd.data_store.motion_lengths[cmd.motion_ids[0]] - 1
        cmd._update_motion_buffers()
        cmd._compute_relative_poses()
        result = step()
        assert result[2][0] and result[3]["time_outs"][0] and manager.get_term("motion_finished")[0]
        assert not manager.terminated[0]
        probes["motion_finished_immediate"] = True

        # Two-control-step tolerance: do not terminate at 498 if the last frame
        # has not been evaluated yet; completed boundary clips still succeed.
        for initial_frame, expected_steps in ((497, 2), (498, 1)):
            reset(3, 0)
            cmd.frame_ids[:] = initial_frame
            cmd._update_motion_buffers()
            cmd._compute_relative_poses()
            env.episode_length_buf[:] = 497
            course.start_step[:] = env.common_step_counter - 497
            before_counts = int(course.counts.sum())
            before_sampler = audit.checks["sampler_outcomes"]
            assert not oracle(f"boundary_frame_{initial_frame}_step_497")["time_out"].any()
            for tick in range(1, expected_steps + 1):
                result = step()
                if tick < expected_steps:
                    assert not result[2].any() and not manager.get_term("time_out").any()
                    assert env.episode_length_buf.eq(498).all()
            assert manager.get_term("time_out").all() and manager.get_term("motion_finished").all()
            assert result[2].all() and result[3]["time_outs"].all()
            assert not manager.terminated.any()
            assert course.last_outcome.eq(1).all() and terrain.terrain_levels.eq(4).all()
            assert int(course.counts.sum()) - before_counts == env.num_envs
            assert audit.checks["sampler_outcomes"] - before_sampler == env.num_envs
            assert env.episode_length_buf.eq(0).all()
            probes[f"boundary_frame_{initial_frame}_success_reset_once"] = True

        # Boundary predicate and real timeout reset (499 -> 500 on the next step).
        reset(3, 0)
        env.episode_length_buf[:] = 499
        assert not oracle("timeout_499")["time_out"].any()
        result = step()
        assert result[3]["time_outs"].all() and course.last_outcome.eq(4).all()
        assert terrain.terrain_levels.eq(3).all()
        probes["timeout_500_neutral_course"] = True
        assert env.episode_length_buf.eq(0).all()
        (output / "measurements.json").write_text(json.dumps(records, indent=2) + "\n")
        save(status="passed", regions_checked=len(regions), probes=probes,
             checks=audit.checks, active_terms=manager.active_terms, stage1_unchanged=True,
             episode_length_s=env.cfg.episode_length_s, max_episode_length=env.max_episode_length,
             boundary_tolerance_steps=2, boundary_tolerance_s=2 * env.step_dt,
             spawn_patch_grid=list(cmd.cfg.placement_patch_grid), spawn_patch_ids=cmd.spawn_patch_ids.tolist(),
             tracking_duration_s=.2, tracking_required_steps=10,
             repeated_reads_reference_unchanged=True)
        print("PGMT_TERMINATIONS_PASSED", json.dumps(report), flush=True)
    except BaseException as error:
        if "records" in locals():
            (output / "measurements.json").write_text(json.dumps(records, indent=2) + "\n")
        save(status="failed", error=repr(error))
        raise
    finally:
        env.close()


from pmt_tasks.runtime_options import run_and_close
def inference_main():
    import torch
    # Native rollouts allocate inference tensors in scene/command caches; all
    # controlled state writes and manual resets must use the same context.
    with torch.inference_mode():
        main()

run_and_close(inference_main, app)
