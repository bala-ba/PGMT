#!/usr/bin/env python3
"""Render streamed physics states using tutorial-compatible Mesa/Xvfb.

The stdin protocol is metadata JSON followed by state JSON lines; each receives
an acknowledgement. This process never calls mj_step or evaluates a policy.
"""
from __future__ import annotations

import json
import faulthandler
from pathlib import Path
import sys
import threading
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from height_scan_visualization import LiveHeightPanel, add_scan_geoms, draw_panel, draw_depth_panel
from terrain_rendering import add_terrain
from camera_visualization import recorded_camera, add_camera_marker


def build_model(terrain_mesh=None, depth_camera=None):
    asset = Path(__file__).resolve().parents[1] / "TCRS/assets/g1/g1_29dof_rev_1_0.xml"
    root = ET.parse(asset).getroot()
    root.find("compiler").set("meshdir", str(asset.parent / "meshes"))
    if depth_camera is not None:
        body_name, camera_attributes = depth_camera
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            raise ValueError(f"Camera attachment body {body_name} is missing from the viewer model")
        ET.SubElement(body, "camera", **camera_attributes)
    # Replace the robot XML's scene lighting instead of stacking duplicate lights.
    for old in list(root.findall("visual")):
        root.remove(old)
    for worldbody in root.findall("worldbody"):
        for light in list(worldbody.findall("light")):
            worldbody.remove(light)
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="960", offheight="640")
    ET.SubElement(visual, "headlight", ambient="0.30 0.30 0.30", diffuse="0.45 0.45 0.45", specular="0 0 0")
    assets = root.find("asset")
    ET.SubElement(assets, "texture", name="ground", type="2d", builtin="checker", width="512", height="512", rgb1="0.25 0.26 0.27", rgb2="0.32 0.33 0.34")
    ET.SubElement(assets, "material", name="ground", texture="ground", texrepeat="1 1", texuniform="true", reflectance="0", specular="0")
    world = root.find("worldbody")
    ET.SubElement(world, "light", pos="0 0 10", directional="true", dir="-1 -2 -3", diffuse="0.65 0.65 0.65", specular="0 0 0")
    if terrain_mesh:
        add_terrain(world, assets, terrain_mesh)
    else:
        ET.SubElement(world, "geom", name="ground", type="plane", size="0 0 0.05", material="ground", group="2")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def acknowledge():
    print('{"status":"ok"}', flush=True)


def main():
    faulthandler.enable()
    metadata = json.loads(sys.stdin.readline())
    output = Path(metadata["output"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    depth_camera = recorded_camera(metadata)
    model = build_model(metadata["terrain_mesh"], depth_camera)
    data = mujoco.MjData(model)
    camera_id = model.camera("pmt_depth").id if depth_camera is not None else None
    camera_aspect = metadata.get("camera", {}).get("width", 160) / metadata.get("camera", {}).get("height", 120) if depth_camera else 1.
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in metadata["joint_names"]]
    if any(j < 0 for j in joints):
        raise ValueError("The renderer robot is missing an Isaac Lab joint.")
    qpos_ids = model.jnt_qposadr[joints]
    bodies = [(i, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)) for i, name in enumerate(metadata["body_names"])]
    bodies = [(i, j) for i, j in bodies if j >= 0]
    if not bodies:
        raise ValueError("The renderer robot has no matching Isaac Lab bodies.")
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.2, 130, -28
    options = mujoco.MjvOption()
    options.geomgroup[:] = 0
    options.geomgroup[1:3] = 1
    viewer = None
    live_panel = None
    previous_threads = set(threading.enumerate())
    renderer = None
    writer = None
    frames = 0
    max_error = 0.0
    scan_frames = 0
    policy_scan_frames = 0
    depth_frames = depth_stamps = depth_poses = None
    camera_marker_frames = 0
    max_camera_pose_error = 0.
    depth_path = Path(metadata.get("replay_source", metadata["output"])) / "depth_frames.npz"
    if metadata.get("perception") == "depth" and depth_path.exists():
        with np.load(depth_path, allow_pickle=False) as recorded_depth:
            depth_frames = recorded_depth["depth_m"]
            depth_stamps = recorded_depth["timestamps"]
            depth_poses = recorded_depth["world_from_camera"]
    states = (output / "states.jsonl").open("w")
    try:
        if metadata["display"]:
            viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
            viewer.opt.geomgroup[:] = options.geomgroup
            viewer.set_texts((mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                              f"PMT | {metadata['physics']}", ""))
            if "height_scan" in metadata:
                live_panel = LiveHeightPanel(height=616 if depth_frames is not None else 372)
        renderer = mujoco.Renderer(model, width=960, height=640)
        if metadata["video"]:
            writer = imageio.get_writer(str(output / f'{metadata["mode"]}.mp4'), fps=1 / metadata["dt"], codec="libx264", quality=8, macro_block_size=16)
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font = ImageFont.truetype(font_path, 18) if Path(font_path).exists() else ImageFont.load_default()
        acknowledge()
        for line in sys.stdin:
            state = json.loads(line)
            data.qpos[:3] = state["root_pos"]
            data.qpos[3:7] = state["root_quat"]  # Both engines use wxyz.
            data.qpos[qpos_ids] = state["joint_pos"]
            mujoco.mj_forward(model, data)
            error = max(np.linalg.norm(data.xpos[j] - state["body_pos"][i]) for i, j in bodies)
            max_error = max(max_error, float(error))
            # Independent FK check catches wrong joint order and quaternion convention.
            if error > 0.001:
                raise ValueError(f"Isaac/MuJoCo body-position mismatch: {error:.6f} m")
            camera.lookat[:] = data.qpos[:3]
            camera.lookat[2] = max(0.5, camera.lookat[2])
            scan = state.get("height_scan")
            panel = None
            if scan is not None:
                panel = draw_panel(scan, metadata["height_scan"])
            depth_panel = None
            if depth_frames is not None:
                stamp = state["step"] * metadata["dt"]
                idx = max(0, np.searchsorted(depth_stamps, stamp + 1e-8, side="right") - 1)
                depth_panel = draw_depth_panel(depth_frames[idx], stamp - depth_stamps[idx])
                if camera_id is not None and abs(stamp - depth_stamps[idx]) < 1e-6:
                    pose = np.eye(4)
                    pose[:3, 3] = data.cam_xpos[camera_id]
                    pose[:3, :3] = data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1., -1., -1.])
                    camera_error = float(np.max(np.abs(pose - depth_poses[idx])))
                    max_camera_pose_error = max(max_camera_pose_error, camera_error)
                    if camera_error > 1e-4:
                        raise ValueError(f"Camera marker differs from recorded optical pose: {camera_error}")
            if viewer is not None:
                if not viewer.is_running():
                    raise RuntimeError("Visualization window was closed.")
                with viewer.lock():
                    viewer.cam.lookat[:] = camera.lookat
                    viewer.cam.distance = camera.distance
                    viewer.cam.azimuth = camera.azimuth
                    viewer.cam.elevation = camera.elevation
                    viewer.user_scn.ngeom = 0
                    if scan is not None:
                        add_scan_geoms(viewer.user_scn, scan, metadata["height_scan"])
                    if camera_id is not None:
                        add_camera_marker(viewer.user_scn, model, data, camera_id, camera_aspect)
                viewer.sync()
                if panel is not None and live_panel is not None:
                    if depth_panel is None:
                        live_panel.update(panel)
                    else:
                        combined = Image.new("RGB", (304, 616))
                        combined.paste(panel, (0, 0))
                        combined.paste(depth_panel, (0, 372))
                        live_panel.update(combined)
            if frames < metadata["max_frames"]:
                states.write(line)
                renderer.update_scene(data, camera=camera, scene_option=options)
                if scan is not None:
                    add_scan_geoms(renderer.scene, scan, metadata["height_scan"])
                    scan_frames += 1
                    policy_scan_frames += int("policy_height" in scan)
                if camera_id is not None:
                    add_camera_marker(renderer.scene, model, data, camera_id, camera_aspect)
                    camera_marker_frames += 1
                frame = Image.fromarray(renderer.render())
                if panel is not None:
                    frame.paste(panel, (12, frame.height - panel.height - 12))
                if depth_panel is not None:
                    frame.paste(depth_panel, (frame.width - 316, 90))
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 960, 80), fill=(17, 24, 35))
                draw.text((16, 8), f'PMT {metadata["mode"]} | {metadata["task"]}', font=font, fill="white")
                draw.text((16, 34), f'{metadata["physics"]} | actual robot 0 | t={state["step"] * metadata["dt"]:.2f}s', font=font, fill=(160, 205, 240))
                draw.text((16, 57), "Smoke run (not converged)" if metadata["mode"] == "training" else Path(metadata["checkpoint"] or "policy").name, font=font, fill=(210, 218, 228))
                if camera_id is not None:
                    draw.rectangle((328, 581, 952, 633), fill=(17, 24, 35))
                    draw.text((340, 584), "Yellow: camera origin + viewing direction", font=font, fill=(255, 205, 30))
                    mount_label = "head mount (approx.)" if metadata.get("camera", {}).get("mount_body") == "head_link" else "simulation mount"
                    draw.text((340, 607), f"Cyan: field of view | {mount_label}", font=font, fill=(100, 235, 245))
                if writer is not None:
                    writer.append_data(np.asarray(frame))
                frames += 1
                if frames == min(25, metadata["max_frames"]):
                    frame.save(output / "preview.png")
            acknowledge()
    finally:
        states.close()
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        if live_panel is not None:
            live_panel.close()
        if viewer is not None:
            viewer.close()
            for thread in set(threading.enumerate()) - previous_threads:
                thread.join(timeout=10)
    (output / "render_result.json").write_text(json.dumps({
        "recorded_frames": frames, "max_body_position_error_m": max_error,
        "matched_bodies": len(bodies), "physics": metadata["physics"],
        "renderer": metadata["renderer"], "finite": True,
        "height_scan_frames": scan_frames, "policy_height_scan_frames": policy_scan_frames,
        "camera_marker_frames": camera_marker_frames, "max_camera_pose_matrix_error": max_camera_pose_error,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
