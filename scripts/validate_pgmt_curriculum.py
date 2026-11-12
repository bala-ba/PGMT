"""Native curriculum/reset/checkpoint contract audit with controlled termination flags.

This validates state transitions, not policy tracking performance. Original
499-frame motions are loaded; tracking failure/map/timeout flags are injected
at selected boundaries to isolate the curriculum's outcome precedence.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default=str(ROOT / "logs/pgmt_curriculum"))
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
    from isaaclab.managers import RewardTermCfg, TerminationTermCfg
    from pmt_tasks.env_cfgs.pgmt.placement import PGMTPlacementEnvCfg
    from pmt_tasks.isaaclab_rl.rsl_rl.vecenv_wrapper import RslRlVecEnvWrapper
    from pmt_tasks.pgmt_curriculum import SAMPLER_COUNTS
    from motion_tracking_rl import compat  # noqa: F401; network registry
    from motion_tracking_rl.runners import OnPolicyRunner

    flags = {name: torch.zeros(7, dtype=torch.bool, device=args.device) for name in ("failure", "map", "timeout")}
    terminal_rewards = []
    def injected(env, name):
        return flags[name]
    def reward_probe(env):
        command = env.command_manager.get_term("motion")
        if env.termination_manager.dones.any():
            terminal_rewards.append(dict(step=env.common_step_counter, frames=command.frame_ids.clone(),
                                         levels=env.scene.terrain.terrain_levels.clone(), motions=command.motion_ids.clone()))
        return torch.zeros(env.num_envs, device=env.device)

    cfg = PGMTPlacementEnvCfg()
    cfg.pmt_motion_paths = [str(ROOT / "assets/motions/pgmt_clipped_50hz_tcrs_v1" / f"{name}.npz")
                            for name in ("walk1_subject1_clip0", "dance1_subject1_clip0")]
    cfg.__post_init__()
    cfg.scene.num_envs = 7
    cfg.seed = 718
    cfg.sim.device = cfg.commands.motion.storage_device = args.device
    cfg.commands.motion.placement_xy_range = (0., 0.)
    cfg.commands.motion.placement_yaw_range = (0., 0.)
    cfg.commands.motion.adaptive_update_interval = 100000
    cfg.observations.proprio.enable_corruption = False
    cfg.observations.proprio_history.enable_corruption = False
    for name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
        setattr(cfg.events, name, None)
    for name in vars(cfg.terminations):
        if name != "motion_finished":
            setattr(cfg.terminations, name, None)
    cfg.terminations.audit_failure = TerminationTermCfg(func=injected, params={"name": "failure"})
    cfg.terminations.map_out_of_bounds = TerminationTermCfg(func=injected, params={"name": "map"}, time_out=True)
    cfg.terminations.time_out = TerminationTermCfg(func=injected, params={"name": "timeout"}, time_out=True)
    cfg.rewards.auxiliary_curriculum_probe = RewardTermCfg(func=reward_probe, weight=1.)
    env = ManagerBasedRLEnv(cfg=cfg)
    try:
        terrain, robot = env.scene.terrain, env.scene["robot"]
        command = env.command_manager.get_term("motion")
        assigned = torch.tensor([3, 3, 3, 3, 3, 9, 0], device=env.device)
        terrain.terrain_levels[:] = assigned
        terrain.terrain_types[:] = 0
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        wrapped = RslRlVecEnvWrapper(env)
        course = env._pgmt_terrain_curriculum
        assert env.curriculum_manager.active_terms == ["terrain_progress"]
        assert course.counts.sum() == 0 and command.sampler.current_success_motion_count.sum() == 0
        length = int(command.data_store.motion_lengths[0])
        assert (command.data_store.motion_lengths == length).all()
        actions = torch.zeros(7, wrapped.num_actions, device=env.device)
        for step in range(length - 1):
            _, _, done, _ = wrapped.step(actions)
            assert not done.any()
            if (step + 1) % 100 == 0:
                print(f"[AUDIT] Completed {step + 1}/{length} control steps before first boundary", flush=True)
        assert command.frame_ids.eq(length - 1).all()
        flags["failure"][[1, 6]] = True
        flags["map"][2] = True
        flags["timeout"][3] = True
        command.frame_ids[3] -= 1  # Ordinary timeout before actual motion completion.
        command._update_motion_buffers_for_envs(torch.tensor([3], device=env.device))
        course.start_step[4] = env.common_step_counter  # A last-frame jump is not complete tracking.
        before_sampler = {k: getattr(command.sampler, k).clone() for k in SAMPLER_COUNTS}
        old_motions = command.start_motion_ids.clone()
        obs, reward, done, extras = wrapped.step(actions)
        expected_levels = torch.tensor([4, 2, 3, 3, 3, 9, 0], device=env.device)
        assert done.eq(1).all()
        assert course.last_outcome.tolist() == [1, 2, 3, 4, 5, 1, 2]
        assert torch.equal(terrain.terrain_levels, expected_levels)
        assert course.counts.sum((0, 1)).tolist() == [2, 2, 1, 1, 1, 0]
        assert course.promotions.sum() == course.demotions.sum() == 1
        assert torch.equal(terminal_rewards[-1]["levels"], assigned)
        assert terminal_rewards[-1]["frames"][[0, 1, 2, 4, 5, 6]].eq(length - 1).all()
        assert command.frame_ids.eq(1).all()
        torch.testing.assert_close(robot.data.root_pos_w[:, :2], terrain.env_origins[:, :2], atol=2e-5, rtol=0)
        torch.testing.assert_close(extras["reward_groups"].sum(-1), reward, atol=1e-5, rtol=1e-5)
        expected_failed = torch.bincount(old_motions[[1, 6]], minlength=2).float()
        expected_success = torch.bincount(old_motions[[0, 2, 3, 4, 5]], minlength=2).float()
        torch.testing.assert_close(command.sampler.current_failed_motion_count - before_sampler["current_failed_motion_count"], expected_failed)
        torch.testing.assert_close(command.sampler.current_success_motion_count - before_sampler["current_success_motion_count"], expected_success)
        assert any(key.startswith("Curriculum/terrain_progress/") for key in extras["log"])
        # Stale flags and explicit reset in the same step must not settle twice.
        snapshot = wrapped.get_checkpoint_state()
        env.reset()
        torch.testing.assert_close(course.counts.cpu(), snapshot["curriculum"]["counts"])
        for key in SAMPLER_COUNTS:
            torch.testing.assert_close(getattr(command.sampler, key).cpu(), snapshot["sampler"][key])
        for value in flags.values():
            value.zero_()

        runner_cfg = dict(num_steps_per_env=2, save_interval=1,
            obs_groups={"policy": ["proprio"], "policy_history": ["proprio_history"],
                        "command_window": ["command_window"], "critic": ["critic"]},
            algorithm=dict(class_name="PPO", num_learning_epochs=1, num_mini_batches=1),
            policy=dict(class_name="PGMTActorCritic", actor_hidden_dims=[32, 16], critic_hidden_dims=[32, 16]))
        runner = OnPolicyRunner(wrapped, copy.deepcopy(runner_cfg), device=env.device)
        runner.logger_type = "tensorboard"
        runner.current_learning_iteration = 17
        checkpoint = output / "audit_checkpoint.pt"
        runner.save(str(checkpoint))
        saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
        assert "environment_state_dict" in saved
        saved_env = saved["environment_state_dict"]
        terrain.terrain_levels.zero_()
        terrain.terrain_types.fill_(1)
        course.counts.zero_()
        course.promotions.zero_()
        course.demotions.zero_()
        for key in SAMPLER_COUNTS:
            getattr(command.sampler, key).fill_(19.)
        command.sampler._steps_since_update = 0
        runner.current_learning_iteration = 0
        runner.load(str(checkpoint))
        assert runner.current_learning_iteration == 17
        restored = wrapped.get_checkpoint_state()
        for key in ("levels", "families", "counts", "promotions", "demotions"):
            torch.testing.assert_close(restored["curriculum"][key], saved_env["curriculum"][key], atol=0, rtol=0)
        for key in SAMPLER_COUNTS:
            torch.testing.assert_close(restored["sampler"][key], saved_env["sampler"][key], atol=0, rtol=0)
        assert restored["sampler"]["steps_since_update"] == saved_env["sampler"]["steps_since_update"]
        assert command.frame_ids.eq(0).all() and course.active.all() and not course.sampler_pending.any()
        torch.testing.assert_close(terrain.env_origins, terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types])
        torch.testing.assert_close(robot.data.root_pos_w[:, :2], terrain.env_origins[:, :2], atol=2e-5, rtol=0)
        print("[AUDIT] Native runner save/load restored assignments, origins, statistics and sampler", flush=True)
        for step in range(length):
            _, _, done, _ = wrapped.step(actions)
            assert bool(done.all()) == (step == length - 1)
            if (step + 1) % 100 == 0:
                print(f"[AUDIT] Completed {step + 1}/{length} control steps after resume", flush=True)
        assert torch.equal(terrain.terrain_levels, (expected_levels + 1).clamp_max(9))
        assert course.counts.sum((0, 1)).tolist() == [9, 2, 1, 1, 1, 0]
        report = dict(passed=True, source_motion_frames=length, native_physics_steps=2 * length,
                      active_curriculum_terms=env.curriculum_manager.active_terms,
                      outcome_counts=course.counts.sum((0, 1)).tolist(),
                      first_boundary_levels=expected_levels.tolist(), resumed_next_boundary_levels=terrain.terrain_levels.tolist(),
                      success_promotes=True, failure_demotes=True, level_caps=True,
                      map_exit_and_early_timeout_neutral=True, incomplete_last_frame_jump_neutral=True,
                      failure_wins_over_motion_end=True, reward_uses_old_birth_assignment=True,
                      reset_uses_new_origins=True, sampler_outcome_once=True, manual_reset_not_counted=True,
                      runner_checkpoint_roundtrip=True, interrupted_episode_not_counted_on_resume=True,
                      sampler_ema_and_pending_counts_restored=True, subsequent_episode_progresses_after_resume=True,
                      controlled_termination_flags_not_policy_performance=True)
        report_file.write_text(json.dumps(report, indent=2) + "\n")
        print("PGMT CURRICULUM VALIDATED: " + json.dumps(report), flush=True)
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
