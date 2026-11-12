"""Shared, simulator-independent command-line options for training and playback."""
from __future__ import annotations

import json
from pathlib import Path


def run_and_close(main, simulation_app):
    """Preserve failures across Kit's fast shutdown, which otherwise exits with 0."""
    import traceback
    import omni.kit.app

    exit_code = 0
    try:
        main()
    except BaseException as error:
        traceback.print_exc()
        exit_code = 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        # Kit's fast shutdown can terminate Python before finally unwinds.
        # Set its return code and print errors BEFORE releasing the framework.
        omni.kit.app.get_app().post_quit(exit_code)
        simulation_app.close()
    if exit_code:
        raise SystemExit(exit_code)


def add_visualization_args(parser):
    parser.add_argument("--video", action="store_true", help="Record the simulator viewport as MP4.")
    parser.add_argument("--video_length", type=int, default=250, help="Video length in simulation steps.")
    parser.add_argument("--video_interval", type=int, default=2000, help="Training steps between recordings.")
    parser.add_argument("--video_dir", default=None, help="Output folder for MP4 recordings.")
    parser.add_argument("--follow_robot", action="store_true", help="Keep robot 0 in the viewport.")
    parser.add_argument("--state_viz", action="store_true", help="Render actual Isaac Lab robot states in a separate MuJoCo process.")
    parser.add_argument("--state_viz_display", default=None, help="Show the state renderer on an X display, e.g. :99.")


def task_overrides(raw_args, motion_file=None):
    invalid = [arg for arg in raw_args if "=" not in arg or arg.startswith("-")]
    if invalid:
        raise ValueError(f"Unsupported arguments: {invalid}. Task overrides use key=value.")
    overrides = list(raw_args)
    if motion_file is not None:
        # Apply before env construction: the task's default data path may not exist.
        motion_path = str(Path(motion_file).expanduser().resolve())
        if not Path(motion_path).exists():
            raise FileNotFoundError(motion_path)
        overrides.append(f"motion.motion_files={json.dumps(motion_path)}")
    return overrides


def configure_viewer(env_cfg, args):
    if args.video_length <= 0 or args.video_interval <= 0:
        raise ValueError("Video length and interval must be positive.")
    if args.follow_robot:
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.eye = (3.0, 3.0, 2.0)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.0)


def record_video(env, args, default_dir, *, training=False):
    if args.state_viz:
        from pmt_tasks.state_visualization import StateVisualization

        return StateVisualization(env, args, default_dir, training=training)
    if not args.video:
        return env
    import gymnasium as gym

    return gym.wrappers.RecordVideo(
        env,
        video_folder=str(Path(args.video_dir or default_dir).resolve()),
        step_trigger=(lambda step: step % args.video_interval == 0) if training else (lambda step: step == 0),
        video_length=args.video_length,
        disable_logger=True,
        name_prefix="train" if training else "play",
    )
