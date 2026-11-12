"""Replay a PGMT NPZ reference in MuJoCo with saved left/right contact labels.

This is kinematic reference playback, not a policy rollout. --video renders one
complete pass; otherwise the native viewer loops, with Space pause, arrows step,
and R restart. Uses the same G1 model as offline contact preprocessing.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import mujoco
import numpy as np

from render_isaac_states import build_model

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOTION = ROOT / "assets/motions/pgmt_clipped_50hz_tcrs_v1/dance1_subject1_clip0.npz"
COLORS = {1: np.array([.13, .85, .35, 1]), 0: np.array([.15, .48, 1., 1]), -1: np.array([.9, .2, .75, 1])}
WORDS = {1: "STANCE", 0: "SWING", -1: "INVALID"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--output", type=Path, default=ROOT / "logs/pgmt_motion_preview/dance1_subject1_clip0")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    with np.load(args.motion, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    n = len(arrays["joint_pos"])
    fps = float(arrays["fps"].item())
    foot_names = arrays["foot_names"].tolist()
    if foot_names != ["left_ankle_roll_link", "right_ankle_roll_link"]:
        raise ValueError("Expected saved [left, right] contact order")
    if arrays["contacts"].shape != (n, 2) or arrays["contact_valid"].shape != (n, 2):
        raise ValueError("Contact labels do not match the motion timeline")
    manifest = json.loads((args.motion.parent / "manifest.json").read_text())
    if manifest["mesh"]["source"] != "reference_flat_plane":
        raise ValueError("This viewer currently supports clips labeled against a flat reference plane")
    model = build_model()
    model.geom_pos[model.geom("ground").id, 2] = manifest["mesh"]["ground_height_m"]
    data = mujoco.MjData(model)
    joint_names = arrays["joint_names"].tolist()
    body_names = arrays["body_names"].tolist()
    joint_ids = model.jnt_qposadr[[model.joint(name).id for name in joint_names]]
    body_ids = [model.body(name).id for name in body_names]
    root_index = body_names.index("pelvis")
    feet = [model.body(name).id for name in foot_names]
    foot_geoms = [np.where(model.geom_bodyid == body)[0] for body in feet]
    pose_error = 0.

    def set_frame(frame):
        data.qpos[:3] = arrays["body_pos_w"][frame, root_index]
        data.qpos[3:7] = arrays["body_quat_w"][frame, root_index]
        data.qpos[joint_ids] = arrays["joint_pos"][frame]
        data.qvel[:] = 0
        data.time = frame / fps
        mujoco.mj_forward(model, data)
        for foot, geoms in enumerate(foot_geoms):
            state = int(arrays["contacts"][frame, foot]) if arrays["contact_valid"][frame, foot] else -1
            model.geom_rgba[geoms] = COLORS[state]

    for frame in range(n):
        set_frame(frame)
        pose_error = max(pose_error, float(np.max(np.abs(data.xpos[body_ids] - arrays["body_pos_w"][frame]))))
    if pose_error > 2e-5:
        raise ValueError(f"Named joint/body mapping failed FK check: {pose_error} m")
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / ("video_status.json" if args.video else "viewer_status.json")

    def status(state, **extra):
        payload = dict(state=state, pid=os.getpid(), motion=str(args.motion.resolve()),
                       frames=n, fps=fps, playback="kinematic_reference", fk_max_error_m=pose_error, **extra)
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(status_path)

    status("validated")
    print(json.dumps(dict(frames=n, fps=fps, fk_max_error_m=pose_error)), flush=True)
    if args.check_only:
        return
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.7, 135, -22
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[1:3] = 1

    def camera_target(frame):
        camera.lookat[:] = arrays["body_pos_w"][frame, root_index]
        camera.lookat[2] = .70

    def label(frame):
        states = [int(arrays["contacts"][frame, i]) if arrays["contact_valid"][frame, i] else -1 for i in range(2)]
        return states, f"LEFT: {WORDS[states[0]]}     RIGHT: {WORDS[states[1]]}"

    if args.video:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw, ImageFont

        out = args.output / (args.motion.stem + ".mp4")
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        with mujoco.Renderer(model, width=960, height=640) as renderer, imageio.get_writer(
                str(out), fps=fps, codec="libx264", quality=8, macro_block_size=16) as writer:
            for frame in range(n):
                set_frame(frame)
                camera_target(frame)
                renderer.update_scene(data, camera=camera, scene_option=opt)
                img = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(img)
                draw.rectangle([0, 0, 960, 116], fill=(20, 25, 32))
                draw.text((18, 10), args.motion.stem + " | MuJoCo reference playback", font=font, fill="white")
                states, _ = label(frame)
                for i, side in enumerate(("LEFT", "RIGHT")):
                    color = tuple((COLORS[states[i]][:3] * 255).astype(int))
                    draw.text((18 + 320 * i, 44), side + ": " + WORDS[states[i]], font=font, fill=color)
                draw.text((18, 78), f"Frame {frame + 1}/{n} | {frame / fps:.2f}s | green=stance blue=swing purple=invalid", font=font, fill=(205, 210, 220))
                writer.append_data(np.asarray(img))
                if frame in (0, n // 4, n // 2, 3 * n // 4):
                    img.save(args.output / f"frame_{frame:04d}.png")
        status("recorded", video=str(out))
        print(f"Recorded: {out}", flush=True)
        return

    import mujoco.viewer as mj_viewer

    state = dict(frame=0, paused=False)

    def keypress(key):
        if key == 32:
            state["paused"] = not state["paused"]
        elif key in (262, 263):
            state["paused"] = True
            state["frame"] = (state["frame"] + (1 if key == 262 else -1)) % n
        elif key == ord("R"):
            state["frame"] = 0

    set_frame(0)
    camera_target(0)
    with mj_viewer.launch_passive(model, data, key_callback=keypress,
                                  show_left_ui=False, show_right_ui=False) as viewer:
        viewer.opt.geomgroup[:] = opt.geomgroup
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = camera.distance, camera.azimuth, camera.elevation
        status("running", frame=0)
        print("Viewer running | Space pause, Left/Right step, R restart", flush=True)
        last_status = 0.
        while viewer.is_running():
            tick = time.monotonic()
            frame = state["frame"]
            with viewer.lock():
                set_frame(frame)
                camera_target(frame)
                viewer.cam.lookat[:] = camera.lookat
            _, description = label(frame)
            viewer.set_texts((mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                              args.motion.stem + " | Reference playback",
                              f"Frame {frame + 1}/{n}   {frame / fps:.2f}s\n{description}\nSpace pause | Arrows step | R restart"))
            viewer.sync()
            if tick - last_status >= 1:
                status("running", frame=frame, paused=state["paused"])
                last_status = tick
            if not state["paused"]:
                state["frame"] = (frame + 1) % n
            time.sleep(max(0., 1 / (fps * args.speed) - (time.monotonic() - tick)))
    status("closed", frame=state["frame"])


if __name__ == "__main__":
    main()
