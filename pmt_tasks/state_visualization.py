"""Stream actual Isaac Lab states to an isolated renderer, without stepping MuJoCo."""
from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import subprocess
import sys

import gymnasium as gym
import numpy as np


def height_scan_snapshot(sensor, observations):
    """Copy the returned observation; never recompute its randomized noise/mask."""
    hits = sensor.data.ray_hits_w[0].detach().cpu().numpy()
    valid = np.isfinite(hits).all(axis=-1)
    scan = {
        "hits_w": np.where(valid[:, None], hits, 0.0).tolist(),
        "hit_valid": valid.tolist(),
    }
    vision = observations.get("vision")
    if vision is not None:
        values = vision[0].detach().cpu().numpy().reshape(-1)
        count = len(hits)
        if values.size not in (count, 2 * count):
            raise ValueError(f"Height visualization expected {count} or {2 * count} values, got {values.size}.")
        if not np.isfinite(values).all():
            raise ValueError("Non-finite policy height observation.")
        scan["policy_height"] = values[:count].tolist()
        scan["policy_valid"] = (values[count:] > 0.5).tolist() if values.size == 2 * count else valid.tolist()
    return scan


class StateVisualization(gym.Wrapper):
    """Synchronous display of environment 0; failures propagate to the caller.

    Rendering is isolated because Kit and Mesa load incompatible GL libraries.
    The renderer gets positions after each real environment step and never sends
    actions or changes physics. Recording is bounded by --video_length.
    """

    def __init__(self, env, args, default_dir, *, training=False):
        super().__init__(env)
        self.output = Path(args.video_dir or default_dir).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.steps = 0
        self.robot = env.unwrapped.scene["robot"]
        self.height_scanner = env.unwrapped.scene.sensors.get("height_scanner")
        self.log = (self.output / "renderer.log").open("w")
        child_env = os.environ.copy()
        for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH"):
            child_env.pop(key, None)
        child_env.update(MUJOCO_GL="glfw", LIBGL_ALWAYS_SOFTWARE="1", __GLX_VENDOR_LIBRARY_NAME="mesa")
        child_env["DISPLAY"] = args.state_viz_display or os.environ.get("DISPLAY", ":99")
        script = Path(__file__).resolve().parents[1] / "scripts" / "render_isaac_states.py"
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(script)], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1, env=child_env,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        terrain = env.unwrapped.cfg.scene.terrain
        metadata = {
            "output": str(self.output), "task": args.task,
            "mode": "training" if training else "testing",
            "checkpoint": getattr(args, "resume_path", None),
            "joint_names": list(self.robot.joint_names),
            "body_names": list(self.robot.body_names),
            "dt": env.unwrapped.step_dt, "video": args.video,
            "max_frames": args.video_length, "display": bool(args.state_viz_display),
            "terrain_mesh": getattr(terrain, "mesh_path", None),
            "physics": "Isaac Lab / PhysX", "renderer": "MuJoCo (kinematics only)",
        }
        if self.height_scanner is not None:
            pattern = self.height_scanner.cfg.pattern_cfg
            metadata["height_scan"] = {
                "shape_xy": [round(size / pattern.resolution) + 1 for size in pattern.size],
                "size_m": list(pattern.size), "resolution_m": pattern.resolution,
                "ordering": pattern.ordering,
                "source": "Isaac Lab ray hits and returned vision observation (before network normalization)",
            }
        try:
            self._request(metadata)
        except BaseException:
            self.close()
            raise

    def _request(self, message):
        try:
            self.process.stdin.write(json.dumps(message, allow_nan=False) + "\n")
            self.process.stdin.flush()
            if not self.selector.select(timeout=120):
                raise TimeoutError("State renderer did not respond within 120 seconds.")
            response = self.process.stdout.readline()
            if not response or json.loads(response).get("status") != "ok":
                raise RuntimeError(f"State renderer failed: {response!r}; see {self.output / 'renderer.log'}")
        except BrokenPipeError as error:
            raise RuntimeError(f"State renderer stopped; see {self.output / 'renderer.log'}") from error

    def step(self, action):
        result = self.env.step(action)
        self.steps += 1
        data = self.robot.data
        state = {
            "step": self.steps,
            "root_pos": data.root_pos_w[0].detach().cpu().tolist(),
            "root_quat": data.root_quat_w[0].detach().cpu().tolist(),
            "joint_pos": data.joint_pos[0].detach().cpu().tolist(),
            "body_pos": data.body_pos_w[0].detach().cpu().tolist(),
        }
        if self.height_scanner is not None:
            state["height_scan"] = height_scan_snapshot(self.height_scanner, result[0])
        self._request(state)
        return result

    def close(self):
        try:
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.close()
            try:
                code = self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                raise RuntimeError("State renderer did not shut down cleanly.")
            if code:
                raise RuntimeError(f"State renderer exited with {code}; see {self.output / 'renderer.log'}")
        finally:
            self.selector.close()
            self.log.close()
            self.env.close()
