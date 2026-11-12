#!/usr/bin/env python3
"""Replay recorded physics states through VNC without starting a simulator."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path, help="Directory containing metadata.json and states.jsonl.")
    parser.add_argument("--display", default=":99")
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video", action="store_true", help="Also export a new video.")
    args = parser.parse_args()
    if args.loops < 1:
        parser.error("--loops must be positive")
    source = args.recording.resolve()
    output = (args.output or Path("logs/replay") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")).resolve()
    if output == source:
        parser.error("Replay output must differ from the original recording.")
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((source / "metadata.json").read_text())
    lines = (source / "states.jsonl").read_text().splitlines()
    if not lines:
        raise ValueError("No recorded states found.")
    metadata.update(output=str(output), display=True, video=args.video, max_frames=len(lines), replay_source=str(source))
    child_env = os.environ.copy()
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH"):
        child_env.pop(key, None)
    child_env.update(DISPLAY=args.display, MUJOCO_GL="glfw", LIBGL_ALWAYS_SOFTWARE="1", __GLX_VENDOR_LIBRARY_NAME="mesa")
    print(f"Replaying {len(lines)} {metadata['physics']} states on {args.display}; artifacts: {output}", flush=True)
    with (output / "renderer.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).with_name("render_isaac_states.py"))],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, env=child_env)
        def send(line):
            process.stdin.write(line + "\n")
            process.stdin.flush()
            response = process.stdout.readline()
            if not response or json.loads(response).get("status") != "ok":
                raise RuntimeError(f"Replay renderer failed (exit={process.poll()}); see {output / 'renderer.log'}")
        try:
            send(json.dumps(metadata))
            for _ in range(args.loops):
                for line in lines:
                    tick = time.monotonic()
                    send(line)
                    time.sleep(max(0.0, metadata["dt"] - (time.monotonic() - tick)))
        finally:
            process.stdin.close()
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise RuntimeError("Replay renderer did not shut down.")
        if code:
            raise RuntimeError(f"Replay renderer exited with {code}.")


if __name__ == "__main__":
    main()
