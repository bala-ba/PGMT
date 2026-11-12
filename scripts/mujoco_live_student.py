#!/usr/bin/env python3
"""Interactive PMT rollout: policy, camera, physics and viewer share live MjData."""
from contextlib import ExitStack
from datetime import datetime
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time

# Select this checkout when several PMT copies are installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image
import torch

from pmt_tasks.backends.mujoco_standalone import create_adapter, build_model, load_policy, validate_contract
from pmt_tasks.backends.mujoco_depth import MujocoDepthPerception
from scripts.camera_visualization import add_camera_marker
from scripts.height_scan_visualization import LiveHeightPanel, add_scan_geoms, draw_panel, draw_depth_panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path("logs/sim2sim/contract.pt"))
    parser.add_argument("--output", type=Path, default=None)
    motion = parser.add_mutually_exclusive_group()
    motion.add_argument("--motion-index", type=int, default=0)
    motion.add_argument("--motion-file", type=Path, help="PGMT motion override after contract validation.")
    parser.add_argument("--checkpoint", type=Path, help="Override checkpoint location; weights must match contract fixtures.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0, help="Stop after N new control steps; 0 runs until window closes.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth-pitch", type=float, default=55.)
    parser.add_argument("--depth-fovy", type=float, default=70.)
    parser.add_argument("--depth-width", type=int, default=160)
    parser.add_argument("--depth-height", type=int, default=120)
    parser.add_argument("--depth-hz", type=float, default=25.)
    parser.add_argument("--map-max-age", type=float, default=2.)
    parser.add_argument("--speed", type=float, default=1., help="Target simulation/wall-clock speed; processing may be slower.")
    args = parser.parse_args()
    if args.steps < 0 or args.start_frame < 0 or min(args.depth_width, args.depth_height) < 3:
        parser.error("Invalid step/frame count or image dimensions")
    if not np.isfinite([args.speed, args.depth_hz, args.map_max_age, args.depth_pitch, args.depth_fovy]).all() or min(args.speed, args.depth_hz, args.map_max_age) <= 0 or not 0 <= args.depth_pitch <= 90 or not 0 < args.depth_fovy < 180:
        parser.error("Invalid camera, map or speed parameter")
    output = args.output or Path("logs/depth/live") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    contract = torch.load(args.contract, map_location="cpu", weights_only=False)
    pgmt = contract.get("policy_class") == "PGMTActorCritic"
    if args.checkpoint is not None:
        contract["checkpoint"] = str(args.checkpoint.resolve())
    if args.motion_file is not None and not pgmt:
        parser.error("--motion-file requires a PGMT contract")
    if not 0 <= args.motion_index < len(contract["motion_files"] if pgmt else contract["raw_files"]):
        parser.error("motion-index is outside the contract motion list")
    dt = contract["sim_dt"] * contract["decimation"]
    if not pgmt and not np.isclose(round(1 / (dt * args.depth_hz)), 1 / (dt * args.depth_hz)):
        parser.error("Depth rate must divide the control rate")
    camera_config = {"width": args.depth_width, "height": args.depth_height, "fovy_deg": args.depth_fovy,
                     "pitch_deg": args.depth_pitch, "hz": args.depth_hz, "max_age_s": args.map_max_age,
                     "mount_body": "head_link", "mount_position_m": [.08, 0., .48],
                     "mount_reference": "assets/G1_structure.png; approximate head mount"}
    if pgmt:
        camera_config = None
    model, geometry = build_model(contract, output / "model", camera_config)
    adapter = create_adapter(contract, model, args.seed)
    policy = load_policy(contract)
    parity = validate_contract(contract, adapter, policy)
    (output / "parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    if not parity["passed"]:
        raise RuntimeError("Isaac/MuJoCo observation parity failed")
    if args.motion_file is not None:
        contract["motion_files"] = list(contract["motion_files"]) + [str(args.motion_file.resolve())]
        args.motion_index = len(contract["motion_files"]) - 1
    adapter.reset(args.motion_index, args.start_frame)
    if args.start_frame >= (adapter.reference_end if pgmt else len(adapter.raw["joint_pos"])) - 1:
        parser.error("start-frame leaves no motion frames to simulate")
    metadata = {"mode": "live", "physics": "MuJoCo mj_step", "camera": camera_config,
                "geometry": geometry, "checkpoint": contract["checkpoint"], "control_dt": dt,
                "motion_index": args.motion_index, "start_frame": args.start_frame,
                "input": "current physics state + reference motion" if pgmt else "live rendered depth + current physics state", "recorded_state_input": False,
                "task": contract.get("task", "PMT-PPOft-Bigmap"),
                "action_contract": contract.get("action_contract", "q_target = raw_q_ref + action"),
                "initial_previous_target": "exported initial_joint_target" if "initial_joint_target" in contract else "exported q_default" if pgmt else "raw motion target",
                "motion_end": "restart selected native chunk with fresh physics and history" if pgmt else "reset physics, observation history and local map; start a fresh episode"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    paused, restart, free_camera = threading.Event(), threading.Event(), threading.Event()
    stop = threading.Event()
    stop_reason = "requested"
    shutdown_timer = None

    def request_stop(reason="exit_button"):
        nonlocal stop_reason, shutdown_timer
        if stop.is_set():
            return
        stop_reason = reason
        stop.set()
        # Native GUI teardown must not leave an unresponsive process indefinitely.
        def force_shutdown():
            os.write(2, b"[LIVE] GUI shutdown exceeded 6 seconds; terminating process.\n")
            os._exit(1)
        shutdown_timer = threading.Timer(6., force_shutdown)
        shutdown_timer.daemon = True
        shutdown_timer.start()

    def on_signal(signum, frame):
        if stop.is_set():
            os._exit(128 + signum)
        request_stop(signal.Signals(signum).name)

    # Explicit handlers also work when a background launcher inherited SIGINT ignored.
    old_handlers = {s: signal.signal(s, on_signal) for s in (signal.SIGINT, signal.SIGTERM)}

    def on_key(key):
        if key in (256, 81):  # Escape / Q (Q also works if the native viewer consumes Escape)
            request_stop("exit_key")
        elif key == 32:
            paused.clear() if paused.is_set() else paused.set()
        elif key == 82:  # R
            restart.set()
        elif key == 70:  # F
            free_camera.clear() if free_camera.is_set() else free_camera.set()

    steps, resets, camera_frames, viewer_updates = 0, 0, 0, 0
    fall_steps, max_torque = 0, 0.
    started = time.monotonic()
    last_display = -np.inf
    status = "completed"
    with ExitStack() as resources, torch.inference_mode():
        sensor = None if pgmt else resources.enter_context(MujocoDepthPerception(model, camera_config))
        adapter.perception = sensor
        obs = adapter.observations()
        camera_frames += sensor.capture_count if sensor is not None else 0
        prior_threads = set(threading.enumerate())
        viewer = mujoco.viewer.launch_passive(model, adapter.data, key_callback=on_key,
                                              show_left_ui=False, show_right_ui=False)

        def close_viewer():
            viewer.close()
            for thread in set(threading.enumerate()) - prior_threads:
                thread.join(timeout=2)

        resources.callback(close_viewer)
        panel = None
        if sensor is not None:
            panel = LiveHeightPanel(height=656)
            panel.root.title("PMT LIVE | Depth camera + height map")
            import tkinter as tk
            button = tk.Button(panel.root, text="STOP SIMULATION (Q / Esc)", command=request_stop,
                               bg="#b33030", fg="white", font=("sans", 12, "bold"))
            button.pack(fill="x", before=panel.label)
            panel.root.protocol("WM_DELETE_WINDOW", request_stop)
            panel.root.bind("<Escape>", lambda event: request_stop("panel_exit_key"))
            panel.root.bind("<Control-c>", lambda event: request_stop("panel_ctrl_c"))
            resources.callback(panel.close)
        with viewer.lock():
            viewer.opt.geomgroup[:] = 0
            viewer.opt.geomgroup[1:3] = 1
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.2, 130, -28
        source = "PGMT flat policy + mj_step" if pgmt else "policy + mj_step + head depth camera"
        print(f"[LIVE] {source} are running. Output: {output}", flush=True)
        print("[LIVE] Space: pause/resume | R: restart | F: follow | Q / STOP button: quit | bash scripts/pmt.sh mujoco-stop", flush=True)
        try:
            while not stop.is_set() and viewer.is_running() and (args.steps == 0 or steps < args.steps):
                tick = time.monotonic()
                boundary = adapter.reference_end if pgmt else len(adapter.raw["joint_pos"])
                if restart.is_set() or adapter.frame >= boundary - 1:
                    adapter.reset(args.motion_index, args.start_frame)
                    obs = adapter.observations()
                    camera_frames += sensor.capture_count if sensor is not None else 0
                    resets += 1
                    restart.clear()
                    print(f"[LIVE] Fresh physics episode {resets + 1}", flush=True)
                if not paused.is_set():
                    action = policy.act_inference(obs)[0].numpy()
                    adapter.step(action)
                    before = sensor.capture_count if sensor is not None else 0
                    obs = adapter.observations()
                    camera_frames += sensor.capture_count - before if sensor is not None else 0
                    steps += 1
                    max_torque = max(max_torque, float(np.abs(adapter.data.actuator_force).max()))
                    pelvis_up = adapter.data.xmat[model.body("pelvis").id].reshape(3, 3)[2, 2]
                    fall_steps += int(adapter.data.qpos[2] < .3 or pelvis_up < .2)
                if tick - last_display >= 1 / 25:
                    with viewer.lock():
                        if not free_camera.is_set():
                            viewer.cam.lookat[:] = adapter.data.qpos[:3]
                            viewer.cam.lookat[2] = max(.5, viewer.cam.lookat[2])
                        viewer.user_scn.ngeom = 0
                        if sensor is not None:
                            add_scan_geoms(viewer.user_scn, adapter.last_scan, adapter.scan_config)
                            add_camera_marker(viewer.user_scn, model, adapter.data, sensor.camera_id,
                                              args.depth_width / args.depth_height)
                    viewer.set_texts((mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                      f"PMT LIVE | {'PAUSED' if paused.is_set() else 'RUNNING'} | episode {resets + 1}",
                                      f"t={adapter.data.time:.2f}s | steps={steps} | Space: pause | R: restart | F: follow | Q: quit"))
                    viewer.sync()
                    if panel is not None:
                        combined = Image.new("RGB", (304, 616))
                        combined.paste(draw_panel(adapter.last_scan, adapter.scan_config), (0, 0))
                        combined.paste(draw_depth_panel(sensor.last_depth, adapter.data.time - sensor.last_depth_timestamp), (0, 372))
                        panel.update(combined)
                    viewer_updates += 1
                    last_display = tick
                time.sleep(max(0., (dt / args.speed if not paused.is_set() else .02) - (time.monotonic() - tick)))
            if stop.is_set():
                status = stop_reason
            elif not viewer.is_running():
                status = "viewer_closed"
                request_stop(status)
            else:
                request_stop("completed")
        except KeyboardInterrupt:
            status = "interrupted"
            request_stop(status)
    result = {"status": status, "steps": steps, "mj_step_calls": steps * int(contract["decimation"]),
              "simulated_seconds_total": steps * dt, "wall_seconds": time.monotonic() - started,
              "episode_resets": resets, "depth_frames": camera_frames, "viewer_updates": viewer_updates,
              "fall_steps": fall_steps, "max_abs_torque_nm": max_torque,
              "finite": bool(np.isfinite(adapter.data.qpos).all()), "interface_parity_passed": parity["passed"],
              "mujoco_warnings": {str(i): int(w.number) for i, w in enumerate(adapter.data.warning) if w.number},
              "recorded_state_input": False}
    (output / "live_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    # Leave the daemon watchdog armed until process exit: a lingering native
    # viewer thread can otherwise keep Python alive after main() returns.
    for signum, handler in old_handlers.items():
        signal.signal(signum, handler)


if __name__ == "__main__":
    main()
