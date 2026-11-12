"""Bounded PGMT/PMT integration checks using the native Isaac environment/runner.

Example (from PMT, with the normal PMT environment activated):
  python scripts/validate_pgmt_stage1.py --headless --checkpoint model_19.pt \
    --output logs/pgmt_stage1_validation/interfaces.json --steps 64 --resume-update
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--steps", type=int, default=64)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--motion_file", default=None)
parser.add_argument("--resume-update", action="store_true", help="Run one update through native runner.learn after checks")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1 or args.num_envs < 2:
    parser.error("--steps must be positive and --num_envs must be at least two")
app = AppLauncher(args).app


def main():
    import gymnasium as gym
    import torch
    from motion_tracking_rl import registry
    from pmt_tasks.builder import build_agent_cfg, build_env_cfg
    from pmt_tasks.isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from pmt_tasks.pgmt_runtime import get_pgmt_state
    from pmt_tasks.registry_gym import register_pmt_tasks

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    registry.autoload()
    register_pmt_tasks()
    overrides = [f"motion.motion_files={args.motion_file}"] if args.motion_file else None
    cfg = build_env_cfg("pgmt_stage1", overrides=overrides)
    agent = build_agent_cfg("pgmt_stage1", overrides=overrides)
    native_agent = build_agent_cfg("multimotionv2_flat")
    assert agent.algorithm.to_dict() == native_agent.algorithm.to_dict(), "Native 2B PPO config changed"
    cfg.scene.num_envs = args.num_envs
    cfg.seed = agent.seed
    if args.device:
        cfg.sim.device = agent.device = args.device
    # Numerical interface checks use clean observations; simulator events,
    # randomization, reset and termination remain the native training settings.
    cfg.observations.proprio.enable_corruption = False
    cfg.observations.proprio_history.enable_corruption = False
    env = RslRlVecEnvWrapper(gym.make("PMT-PGMT-Stage1-G1-v0", cfg=cfg))
    runner = registry.get_runner(agent.class_name)(env, agent.to_dict(),
                                                  log_dir=str(output.parent / "resume_update"), device=agent.device)
    report = {"checkpoint": str(Path(args.checkpoint).resolve()), "steps": args.steps,
              "num_envs": args.num_envs, "status": "running", "checks": {}}
    try:
        runner.load(args.checkpoint)
        loaded_iter = runner.current_learning_iteration
        policy = runner.alg.policy
        policy.eval()
        raw = env.unwrapped
        runtime = get_pgmt_state(raw)
        command = raw.command_manager.get_term("motion")
        sampler = command.sampler
        assert command.cfg.sampler_type == "bin_adaptive"
        assert command.cfg.chunk_length == 4000
        assert command.cfg.update_action_offset_with_ref is False
        assert type(command).__name__ == "MultiMotionCommandV2"
        action_term = raw.action_manager.get_term("joint_pos")
        assert action_term.cfg.clip is None
        obs = env.get_observations()
        report["observation_shapes"] = {key: list(value.shape) for key, value in obs.items()}
        max_reward_error = max_target_error = max_history_error = 0.
        done_count = 0
        def count_snapshot():
            return {key: getattr(sampler, key).clone() for key in
                    ("failed_bin_count", "success_bin_count", "current_failed_bin_count", "current_success_bin_count")}
        initial_counts = count_snapshot()
        with torch.no_grad():
            for _ in range(args.steps):
                assert all(torch.isfinite(value).all() for value in obs.values())
                values = policy.evaluate(obs)
                assert values.shape == (args.num_envs, 3) and torch.isfinite(values).all()
                action = policy.act_inference(obs)
                expected_target = action * action_term._scale + action_term._offset
                obs, reward, done, extras = env.step(action)
                groups = extras["reward_groups"]
                assert groups.shape == (args.num_envs, 3)
                torch.testing.assert_close(groups.sum(-1), reward, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(runtime.qtarget(), expected_target, rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(obs["proprio_history"][:, -1], obs["proprio"], rtol=1e-6, atol=1e-6)
                assert torch.isfinite(reward).all() and torch.isfinite(groups).all()
                max_reward_error = max(max_reward_error, (groups.sum(-1) - reward).abs().max().item())
                max_target_error = max(max_target_error, (runtime.qtarget() - expected_target).abs().max().item())
                max_history_error = max(max_history_error, (obs["proprio_history"][:, -1] - obs["proprio"]).abs().max().item())
                done_count += int(done.sum().item())
                history = obs["proprio_history"].clone()
                repeated = env.get_observations()
                torch.testing.assert_close(repeated["proprio_history"], history)
                torch.testing.assert_close(repeated["proprio"], obs["proprio"])

        # Force precisely the native command-internal end-of-chunk path. This
        # does not call env.step or set a done/truncation flag.
        reset_before = raw.reset_buf.clone()
        term_before = raw.termination_manager.terminated.clone()
        timeout_before = raw.termination_manager.time_outs.clone()
        counters_before = raw.episode_length_buf.clone()
        counts_before_internal = count_snapshot()
        command.frame_ids[0] = command.data_store.motion_lengths[command.motion_ids[0]] - 1
        command._update_command()
        assert torch.equal(raw.reset_buf, reset_before)
        assert torch.equal(raw.termination_manager.terminated, term_before)
        assert torch.equal(raw.termination_manager.time_outs, timeout_before)
        assert torch.equal(raw.episode_length_buf, counters_before)
        assert runtime.pending_reset[0]
        internal_obs = raw.observation_manager.compute(update_history=True)
        torch.testing.assert_close(internal_obs["proprio_history"][0],
                                   internal_obs["proprio"][0].expand(10, -1))
        assert not runtime.pending_reset[0]
        torch.testing.assert_close(runtime.previous_target[0], runtime.qtarget()[0])
        final_counts = count_snapshot()
        assert any(not torch.equal(counts_before_internal[k], final_counts[k]) for k in final_counts), \
            "Native sampler did not record the internal clip outcome"

        # A future window at the chunk's last frame must repeat that frame,
        # never read the padding or cross into the source clip's next chunk.
        saved_frames = command.frame_ids.clone()
        command.frame_ids[:] = command.data_store.motion_lengths[command.motion_ids] - 1
        future = runtime.reference(future=True)
        torch.testing.assert_close(future["joint_pos"], future["joint_pos"][:, :1].expand_as(future["joint_pos"]))
        command.frame_ids.copy_(saved_frames)
        runtime.invalidate_observations()

        report["checks"] = {
            "native_2b_ppo_config": True, "value_heads": 3,
            "reward_group_sum_max_abs_error": max_reward_error,
            "actual_target_max_abs_error": max_target_error,
            "proprio_history_latest_max_abs_error": max_history_error,
            "repeated_observation_read_preserves_history": True,
            "done_events_observed": done_count,
            "command_internal_reset_preserves_episode_masks": True,
            "command_internal_reset_repeats_new_history": True,
            "command_internal_reset_records_sampler_outcome": True,
            "sampler_counts_changed": any(not torch.equal(initial_counts[k], final_counts[k]) for k in final_counts),
            "future_window_clamps_at_native_chunk_end": True,
        }
        report["reference_store"] = {
            "native_tracked_bodies": command.data_store.num_bodies,
            "pgmt_bodies": runtime.full_store.num_bodies,
            "chunks": runtime.full_store.num_motions,
            "additional_tensor_bytes": sum(value.numel() * value.element_size()
                                            for value in vars(runtime.full_store).values() if isinstance(value, torch.Tensor)),
        }
        if args.resume_update:
            optimizer_steps_before = [float(state["step"]) for state in runner.alg.optimizer.state.values()
                                      if "step" in state]
            runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
            assert all(torch.isfinite(p).all() for p in policy.parameters())
            assert len(runner.alg.optimizer.state) > 0
            optimizer_steps_after = [float(state["step"]) for state in runner.alg.optimizer.state.values()
                                     if "step" in state]
            assert len(optimizer_steps_before) == len(optimizer_steps_after)
            assert all(after > before for before, after in zip(optimizer_steps_before, optimizer_steps_after))
            resumed_checkpoint = output.parent / "resume_update" / "validated_resume.pt"
            runner.save(str(resumed_checkpoint))
            report["resume"] = {"loaded_iteration": loaded_iter,
                                "iteration_after_one_native_update": runner.current_learning_iteration,
                                "checkpoint": str(resumed_checkpoint), "finite_parameters": True,
                                "optimizer_parameter_states": len(runner.alg.optimizer.state),
                                "optimizer_steps_before": sorted(set(optimizer_steps_before)),
                                "optimizer_steps_after": sorted(set(optimizer_steps_after)),
                                "iteration_note": "Native runner stores the current zero-based iteration; one resumed update may retain the same label."}
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        if runner.writer is not None:
            runner.writer.close()
        env.close()


if __name__ == "__main__":
    from pmt_tasks.runtime_options import run_and_close
    run_and_close(main, app)
