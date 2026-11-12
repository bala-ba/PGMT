"""Fresh PGMT MuJoCo rollouts with an uncut frame-0-through-last-frame video.

Run with MUJOCO_GL=egl on a host with GPU access. The policy and mj_step execute
again; each new state is compared against the verified complete batch rollout.
Tracking failures are annotated but never reset or truncate the video.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from pmt_tasks.backends.mujoco_standalone import create_adapter, load_policy, validate_contract
from scripts.mujoco_eval_pgmt_full import reset_full_clip, sha256, write_json


def load_json(path):
    return json.loads(Path(path).read_text())


def visual_model(scene_xml, output):
    """Cosmetic scene additions; physical model/actuators remain identical."""
    root = ET.parse(scene_xml).getroot()
    for old in root.findall("visual"):
        root.remove(old)
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="960", offheight="640")
    ET.SubElement(visual, "quality", offsamples="4", shadowsize="2048")
    ET.SubElement(visual, "headlight", ambient=".3 .3 .3", diffuse=".45 .45 .45", specular="0 0 0")
    assets = root.find("asset")
    ET.SubElement(assets, "texture", name="video_floor_texture", type="2d", builtin="checker",
                  width="512", height="512", rgb1=".25 .27 .29", rgb2=".34 .36 .38")
    ET.SubElement(assets, "material", name="video_floor", texture="video_floor_texture",
                  texrepeat="1 1", texuniform="true", reflectance="0", specular="0")
    world = root.find("worldbody")
    for light in world.findall("light"):
        world.remove(light)
    ET.SubElement(world, "light", pos="0 0 10", directional="true", dir="-1 -2 -3",
                  diffuse=".65 .65 .65", specular="0 0 0")
    ground = world.find("geom[@name='terrain_ground']")
    ground.set("material", "video_floor")
    ground.set("rgba", "1 1 1 1")
    path = output / "scene_video.xml"
    ET.indent(root)
    ET.ElementTree(root).write(path)
    return mujoco.MjModel.from_xml_path(str(path))


def camera_for(data):
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 3.2, 130, -25
    camera.lookat[:] = data.qpos[:3]
    camera.lookat[2] = max(.5, camera.lookat[2])
    return camera


def record(adapter, policy, renderer, options, source_clip, termination_dir, output):
    index, name = source_clip["motion_index"], source_clip["motion"]
    reset_full_clip(adapter, index)
    count = len(adapter.raw["joint_pos"])
    dt = adapter.contract["sim_dt"] * adapter.contract["decimation"]
    if source_clip["source_frames"] != count or sha256(source_clip["motion_file"]) != source_clip["motion_sha256"]:
        raise ValueError("Reference motion changed since the verified full rollout")
    with np.load(source_clip["trajectory"]) as baseline:
        expected_qpos, expected_qvel = baseline["qpos"], baseline["qvel"]
        initial_qpos, initial_qvel = baseline["initial_qpos"], baseline["initial_qvel"]
    np.testing.assert_allclose(adapter.data.qpos, initial_qpos, rtol=0, atol=1e-12)
    np.testing.assert_allclose(adapter.data.qvel, initial_qvel, rtol=0, atol=1e-12)
    stop_frames = {}
    for key in ("pmt_full", "sonic_eval"):
        clip = load_json(termination_dir / key / f"{index:02d}_{name}.json")
        stop_frames[key] = clip["first_failure_frame"]
    qpos = np.empty((count, adapter.model.nq))
    qvel = np.empty((count, adapter.model.nv))
    qpos[0], qvel[0] = adapter.data.qpos, adapter.data.qvel
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    title_font = ImageFont.truetype(font_path, 19)
    font = ImageFont.truetype(font_path, 16)
    video = output / f"{name}.mp4"
    temporary = output / f"{name}.partial.mp4"
    writer = imageio.get_writer(str(temporary), fps=1/dt, codec="libx264", quality=None,
                               macro_block_size=16, ffmpeg_log_level="error",
                               output_params=["-preset", "veryfast", "-crf", "20", "-threads", "4", "-movflags", "+faststart"])
    max_qpos_error = max_qvel_error = 0.
    start = last_log = time.monotonic()
    preview_frames = {0: "start", count//2: "middle", count-1: "end"}
    try:
        with torch.inference_mode():
            obs = adapter.observations()
            for frame in range(count):
                if frame:
                    action = policy.act_inference(obs)[0].numpy()
                    adapter.step(action)
                    data = adapter.data
                    if adapter.frame != frame or abs(data.time - frame*dt) > 1e-6 or any(w.number for w in data.warning):
                        raise RuntimeError(f"Physics/frame discontinuity at {name}:{frame}")
                    qpos[frame], qvel[frame] = data.qpos, data.qvel
                    pe = float(np.max(np.abs(data.qpos - expected_qpos[frame-1])))
                    ve = float(np.max(np.abs(data.qvel - expected_qvel[frame-1])))
                    max_qpos_error, max_qvel_error = max(max_qpos_error, pe), max(max_qvel_error, ve)
                    if max(pe, ve) > 1e-8:
                        raise RuntimeError(f"Fresh rollout changed at {name}:{frame}: qpos={pe}, qvel={ve}")
                    if frame < count-1:
                        obs = adapter.observations()
                data = adapter.data
                body = data.xpos[adapter.body_ids]
                target = adapter.raw["body_pos_w"][frame]
                error = np.linalg.norm(body[adapter.tracked] - target[adapter.tracked], axis=-1).mean()*1000
                renderer.update_scene(data, camera=camera_for(data), scene_option=options)
                rendered = renderer.render()
                # A separate header preserves every pixel of the camera image.
                canvas = Image.new("RGB", (960, 720), (17, 24, 35))
                canvas.paste(Image.fromarray(rendered), (0, 80))
                draw = ImageDraw.Draw(canvas)
                draw.text((14, 5), f"{name}  |  model_15000.pt  |  MuJoCo", font=title_font, fill="white")
                endpoint = "  START" if frame == 0 else ("  END" if frame == count-1 else "")
                draw.text((14, 31), f"t={frame*dt:.2f} / {(count-1)*dt:.2f}s   frame={frame}/{count-1}   MPKPE={error:.1f} mm{endpoint}",
                          font=font, fill=(168, 212, 247))
                labels = []
                for key, label in (("pmt_full", "PMT"), ("sonic_eval", "SONIC")):
                    first = stop_frames[key]
                    labels.append(f"{label}: first failure at {first*dt:.2f}s" if first is not None and frame >= first
                                  else f"{label}: tracking")
                draw.text((14, 55), "  |  ".join(labels) + "  |  full clip, no resets", font=font, fill=(221, 226, 231))
                writer.append_data(np.asarray(canvas))
                if frame in preview_frames:
                    canvas.save(output / "previews" / f"{name}_{preview_frames[frame]}.png")
                now = time.monotonic()
                if now-last_log >= 20:
                    print(f"[VIDEO] {name} {frame+1}/{count} frames, t={frame*dt:.2f}s, export_fps={(frame+1)/(now-start):.1f}", flush=True)
                    last_log = now
    finally:
        writer.close()
    encoded_frames, duration = imageio_ffmpeg.count_frames_and_secs(str(temporary))
    if encoded_frames != count or abs(duration - count*dt) > .03:
        raise RuntimeError(f"Incomplete video: {encoded_frames}/{count} frames, duration={duration}")
    temporary.replace(video)
    np.savez_compressed(output / "trajectories" / f"{name}.npz", motion_frame=np.arange(count),
                        time_s=np.arange(count)*dt, qpos=qpos, qvel=qvel)
    result = {"motion": name, "motion_index": index, "video": str(video.resolve()),
              "frames": count, "encoded_frames": encoded_frames, "fps": 1/dt,
              "video_duration_s": duration, "first_reference_frame": 0, "last_reference_frame": count-1,
              "first_sim_time_s": 0., "last_sim_time_s": (count-1)*dt, "width": 960, "height": 720,
              "mj_step_calls": adapter.physics_steps, "resets_within_motion": 0, "frames_dropped": 0,
              "tracking_failure_truncation": False, "source_motion": source_clip["motion_file"],
              "source_trajectory": source_clip["trajectory"], "first_failure_frames": stop_frames,
              "max_qpos_difference_from_verified_full_rollout": max_qpos_error,
              "max_qvel_difference_from_verified_full_rollout": max_qvel_error,
              "mujoco_warnings": {str(i): int(w.number) for i, w in enumerate(adapter.data.warning) if w.number},
              "wall_seconds": time.monotonic()-start, "video_sha256": sha256(video)}
    write_json(output / f"{name}.json", result)
    print(f"[DONE] {name} frames={count} duration={duration:.2f}s wall={result['wall_seconds']:.1f}s", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--terminations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motions", nargs="+", required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be empty")
    if len(args.motions) != len(set(args.motions)):
        parser.error("Motion names must be distinct")
    source = load_json(args.source / "metadata.json")
    if not load_json(args.source / "verification.json")["passed"]:
        raise ValueError("Source verification failed")
    if sha256(source["checkpoint"]) != source["checkpoint_sha256"] or sha256(source["contract"]) != source["contract_sha256"]:
        raise ValueError("Model or contract changed")
    clips = {c["motion"]: c for c in [load_json(p) for p in sorted((args.source / "clips").glob("*.json"))]}
    if any(name not in clips for name in args.motions):
        parser.error("Unknown motion name; use exact source stem")
    args.output.mkdir(parents=True)
    (args.output / "previews").mkdir()
    (args.output / "trajectories").mkdir()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    contract = torch.load(source["contract"], map_location="cpu", weights_only=False)
    model = visual_model(source["geometry"]["scene_xml"], args.output)
    adapter = create_adapter(contract, model)
    policy = load_policy(contract)
    parity = validate_contract(contract, adapter, policy)
    write_json(args.output / "parity.json", parity)
    if not parity["passed"]:
        raise RuntimeError("Isaac contract parity failed")
    options = mujoco.MjvOption()
    options.geomgroup[:] = 0
    options.geomgroup[1:3] = 1
    metadata = {"checkpoint": source["checkpoint"], "checkpoint_sha256": source["checkpoint_sha256"],
                "source_full_batch": str(args.source.resolve()), "motions": args.motions,
                "physics": f"Fresh MuJoCo {mujoco.__version__} CPU policy + mj_step",
                "renderer": "MuJoCo GPU EGL", "format": "H.264 MP4, 960x720, 50 fps, real-time playback",
                "coverage": "Every source frame exactly once, including frame 0 and last frame; no trimming, no resets, continue after tracking failure",
                "camera": "Fixed world azimuth 130deg, elevation -25deg, distance 3.2m, follows actual pelvis; header outside full camera image",
                "sources_sha256": {str(Path(__file__).resolve()): sha256(__file__),
                                   str(Path(source["geometry"]["scene_xml"])): sha256(source["geometry"]["scene_xml"])}}
    write_json(args.output / "metadata.json", metadata)
    results = []
    with mujoco.Renderer(model, width=960, height=640) as renderer:
        from OpenGL import GL
        metadata["gl_renderer"] = GL.glGetString(GL.GL_RENDERER).decode()
        write_json(args.output / "metadata.json", metadata)
        for name in args.motions:
            results.append(record(adapter, policy, renderer, options, clips[name], args.terminations, args.output))
            write_json(args.output / "manifest.json", {"completed": len(results), "requested": len(args.motions),
                                                       "all_complete": len(results) == len(args.motions), "videos": results})


if __name__ == "__main__":
    main()
