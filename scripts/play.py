"""Play a trained checkpoint on a PMT (Isaac Lab) tracking task — the Isaac-side
counterpart of scripts/mjlab_view_pmt_ckpt.py.

Mirrors scripts/train.py setup (AppLauncher first → register PMT tasks → build env/agent
cfg via the registry → load checkpoint → get_inference_policy → step in a viewer loop).

Example (the G1-MultiMotionV2-Streaming checkpoint, on the 100style task):
  conda activate <env>
  python scripts/play.py \
    --task PMT-G1-MultiMotionV2-Streaming-100style-Flat-v0 \
    --num_envs 1 \
    --motion_file <motion-file-or-dir> \
    --resume_path <checkpoint.pt>

Headless bounded run (no GUI, exits after --max_steps):
  python scripts/play.py ... --headless --max_steps 300

Direct SONIC release-ONNX play (no RSL checkpoint):
  PMT_SONIC_ONNX_DIR=<release-onnx-dir> python scripts/play.py \
    --task PMT-SONIC-G1-MultiMotionV2-Flat-v0 \
    --sonic_mode play \
    --num_envs 1

App-launch-first ordering (required): AppLauncher().app BEFORE importing isaaclab.envs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher
from pmt_tasks.runtime_options import add_visualization_args, configure_viewer, record_video, run_and_close, task_overrides

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True, help="PMT gym task id.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs.")
parser.add_argument("--profile", type=str, default=None, help="PMT paths profile (local|cluster).")
parser.add_argument("--resume_path", type=str, default=None, help="Checkpoint .pt to load.")
parser.add_argument(
    "--sonic_mode",
    type=str,
    default=None,
    choices=["scratch", "finetune_all", "finetune_decoder", "play"],
    help="Override sonic_multimotion_flat mode; play can use release ONNX without --resume_path.",
)
parser.add_argument(
    "--motion_file",
    type=str,
    default=None,
    help="Override motion clip/dir (single .npz or a folder).",
)
parser.add_argument("--max_steps", type=int, default=None, help="Stop after N steps (headless).")
parser.add_argument("--metrics_file", default=None, help="Write finite-value checks and rollout metrics to JSON.")
parser.add_argument("--real_time", action="store_true", help="Limit GUI playback to simulation speed.")
add_visualization_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, extra_args = parser.parse_known_args()
if args_cli.video and not args_cli.state_viz:
    args_cli.enable_cameras = True

if args_cli.profile:
    os.environ["PMT_PROFILE"] = args_cli.profile

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

from motion_tracking_rl import registry  # noqa: E402
from pmt_tasks.builder import build_agent_cfg, build_env_cfg  # noqa: E402
from pmt_tasks.isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from pmt_tasks.registry_gym import gym_id_for_stem, register_pmt_tasks, task_stem_for_id  # noqa: E402


def main():
    registry.autoload()
    register_pmt_tasks()

    overrides = task_overrides(extra_args, args_cli.motion_file)
    if args_cli.sonic_mode is not None:
        sonic_task_id = gym_id_for_stem("sonic_multimotion_flat")
        if args_cli.task != sonic_task_id:
            raise ValueError(f"--sonic_mode is only supported for task {sonic_task_id}.")
        overrides.append(f"sonic_mode={args_cli.sonic_mode}")
    if overrides:
        stem = task_stem_for_id(args_cli.task)
        env_cfg = build_env_cfg(
            stem,
            profile=args_cli.profile,
            overrides=overrides,
        )
        agent_cfg = build_agent_cfg(
            stem,
            profile=args_cli.profile,
            overrides=overrides,
        )
    else:
        env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
        agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    configure_viewer(env_cfg, args_cli)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video and not args_cli.state_viz else None)
    env = record_video(env, args_cli, "logs/play/videos")
    env = RslRlVecEnvWrapper(env)

    runner_class = registry.get_runner(agent_cfg.class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    if args_cli.resume_path is not None:
        print(f"[INFO] loading checkpoint: {args_cli.resume_path}")
        runner.load(args_cli.resume_path)
    elif args_cli.sonic_mode == "play" and agent_cfg.policy.class_name == "OfficialSonicActorCritic":
        print("[INFO] using SONIC release ONNX from agent cfg; no RSL checkpoint loaded.")
    else:
        raise ValueError("--resume_path is required unless --sonic_mode play loads SONIC release ONNX.")

    policy = runner.get_inference_policy(device=agent_cfg.device)

    obs = env.get_observations()  # RslRlVecEnvWrapper returns a TensorDict (not a tuple)
    step = 0
    reward_sum = 0.0
    resets = 0
    metric_sums = {}
    start = time.monotonic()
    try:
        while simulation_app.is_running():
            tick = time.monotonic()
            with torch.inference_mode():
                actions = policy(obs)
                if not torch.isfinite(actions).all():
                    raise FloatingPointError(f"Non-finite policy actions at step {step}.")
                obs, rewards, dones, _ = env.step(actions)
                if not torch.isfinite(rewards).all() or not all(torch.isfinite(v).all() for v in obs.values()):
                    raise FloatingPointError(f"Non-finite rollout tensors at step {step}.")
            reward_sum += float(rewards.mean().item())
            resets += int(dones.sum().item())
            command = env.unwrapped.command_manager.get_term("motion")
            for key, values in command.metrics.items():
                value = float(values.float().mean().item())
                metric_sums[key] = metric_sums.get(key, 0.0) + value
            step += 1
            if args_cli.real_time:
                time.sleep(max(0.0, env.unwrapped.step_dt - (time.monotonic() - tick)))
            if args_cli.max_steps is not None and step >= args_cli.max_steps:
                print(f"[INFO] reached max_steps={args_cli.max_steps}, stopping.")
                break
        if step == 0:
            raise RuntimeError("Playback exited without executing a simulation step.")
        summary = {
            "task": args_cli.task,
            "checkpoint": args_cli.resume_path,
            "num_envs": args_cli.num_envs,
            "steps": step,
            "finite": True,
            "mean_step_reward": reward_sum / step,
            "resets": resets,
            "wall_seconds": time.monotonic() - start,
            "mean_motion_metrics": {key: value / step for key, value in metric_sums.items()},
        }
        print("[PLAY_RESULT] " + json.dumps(summary, allow_nan=False))
        if args_cli.metrics_file:
            output = Path(args_cli.metrics_file)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    run_and_close(main, simulation_app)
