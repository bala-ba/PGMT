"""Render seven terrain columns / three motion rows from recorded Isaac states.

MuJoCo performs rendering and FK validation only, never physics integration.
Terminal poses are explicitly labelled and held after each episode ends.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import render_pgmt_force_comparison as replay
from pgmt_force_analysis import load_run

PANEL, RENDER_H, ROW_H = 480, 336, 432
HEADER, FOOTER = 96, 48
LABELS = dict(flat="FLAT / 平地", rough="ROUGH / 粗糙地面", up_stair="UP STAIR / 上楼梯",
              down_stair="DOWN STAIR / 下楼梯", up_slope="UP SLOPE / 上坡",
              down_slope="DOWN SLOPE / 下坡", boxes="BOXES / 箱体")
MOTION_LABELS = ("步行 WALK", "跑步 RUN", "跳跃 JUMP")
COLORS = ((100, 184, 239), (194, 180, 139), (238, 173, 100), (238, 173, 100),
          (109, 202, 153), (109, 202, 153), (175, 155, 229))


def detail(view, label):
    p = view.patch["parameters"]
    if "stair" in label:
        return f"台阶 {p['stair_height_m']*100:.1f} cm · {'边缘起步' if label.startswith('down') else '中心起步'}"
    if "slope" in label:
        return f"坡度 {p['slope_angle_deg']:.2f}° · {'边缘起步' if label.startswith('down') else '中心起步'}"
    if label == "boxes":
        return f"箱体高度幅度 ±{p['box_height_amplitude_m']*100:.1f} cm"
    if label == "rough":
        amplitude = view.patch["geometry"].get("measured_amplitude_m", 0)
        return f"实测高度幅度 {amplitude*100:.1f} cm"
    return "固定动力学 · 无随机扰动"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--levels", nargs="+", type=int)
    parser.add_argument("--fps", type=int, choices=(25, 50), default=25)
    parser.add_argument("--preview_only", action="store_true")
    parser.add_argument("--wait_for_trials", action="store_true",
                        help="Wait for each level while the acquisition process is running")
    args = parser.parse_args()
    args.source, args.output = args.source.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    suite = json.loads((args.source / "suite.json").read_text())
    levels = args.levels or suite["levels"]
    fonts = {size: ImageFont.truetype(replay.FONT, size) for size in (16, 18, 20, 22, 26, 32)}
    replay.PANEL, replay.RENDER_H = PANEL, RENDER_H
    validations, all_results = [], []
    for level in levels:
        trials = [t for t in suite["trials"] if t["level"] == level]
        if len(trials) != 7 * len(suite["motions"]):
            raise ValueError("Incomplete level matrix")
        if args.wait_for_trials:
            while True:
                ready = all((args.source/t["directory"]/"training_status.json").is_file()
                    and json.loads((args.source/t["directory"]/"training_status.json").read_text()).get("status") == "completed"
                    for t in trials)
                if ready:
                    break
                state = json.loads((args.source/"training_status.json").read_text())
                if state["status"] in ("failed", "failed_physics", "completed"):
                    raise RuntimeError(f"Acquisition stopped with incomplete L{level}: {state['status']}")
                time.sleep(5)
        views, runs, indices = [], [], []
        shared_views, pose_errors = {}, {}
        meshdir = args.output / "render_assets" / f"level{level}"
        meshdir.mkdir(parents=True, exist_ok=True)
        video = args.output / f"level{level}_flat_rough_up-stair_down-stair_up-slope_down-slope_boxes.mp4"
        if video.exists() and not args.preview_only:
            raise FileExistsError(video)
        temporary = args.output / f"level{level}_encoding.mp4"
        writer = None
        width, height = 7*PANEL, HEADER+ROW_H*len(suite["motions"])+FOOTER
        try:
            for t in trials:
                run = load_run(args.source / t["directory"])
                if run["metadata"]["source_sha256"] != suite["source_sha256"]:
                    raise ValueError("Checkpoint mismatch")
                d = run["data"]
                for key in ("body_pos_w", "body_quat_w", "joint_pos"):
                    if not np.isfinite(d[key]).all():
                        raise ValueError(f"Nonfinite {key}")
                if len(np.unique(d["episode"])) != 1:
                    raise ValueError("Only one complete recorded episode per cell is allowed")
                decimation = run["metadata"]["force_recording"]["decimation"]
                ix = np.flatnonzero(d["substep"] == decimation-1)
                if not np.array_equal(d["frame"][ix], np.arange(len(ix))):
                    raise ValueError("Missing or repeated source frames")
                indices.append(ix)
                runs.append(run)
                # One renderer per actual terrain mesh, shared across motions
                # and directions to bound GPU memory to five small contexts.
                if t["terrain"] not in shared_views:
                    shared_views[t["terrain"]] = replay.View(run, meshdir)
                view = shared_views[t["terrain"]]
                view.camera.distance = 3.1
                view.camera.azimuth, view.camera.elevation = 112., -22.
                views.append(view)
                all_results.append(run["metadata"]["last_episode"])
            duration = max(float(r["data"]["time_s"][-1]) for r in runs)
            count = int(np.ceil(duration * args.fps))
            frames = sorted({0, count//2, count-1}) if args.preview_only else range(count)
            cached = {}
            if not args.preview_only:
                writer = imageio.get_writer(str(temporary), fps=args.fps, codec="libx264", quality=None,
                    macro_block_size=16, ffmpeg_params=["-crf", "19", "-preset", "fast", "-threads", "4", "-movflags", "+faststart"])
            for fi in frames:
                stamp = min((fi+1)/args.fps, duration)
                canvas = Image.new("RGB", (width, height), (13, 20, 30))
                draw = ImageDraw.Draw(canvas)
                draw.text((24, 7), f"PGMT · Level {level} · 7 种地形 × 3 类代表动作", font=fonts[32], fill=(241, 246, 251))
                draw.text((24, 54), "train_03 / model_19999.pt  |  Isaac / PhysX 实测轨迹  |  1× 速度  |  每列独立仿真、同一时刻对比",
                          font=fonts[22], fill=(162, 183, 206))
                draw.text((width-230, 21), f"t = {stamp:5.2f} s", font=fonts[26], fill=(235, 242, 249))
                for cell, (trial, run, view, ix) in enumerate(zip(trials, runs, views, indices)):
                    row, col = divmod(cell, 7)
                    x, y = col*PANEL, HEADER+row*ROW_H
                    d, meta = run["data"], run["metadata"]
                    terminal = stamp >= float(d["time_s"][-1])-1e-7
                    k = min(int(np.searchsorted(d["time_s"][ix], stamp, side="left")), len(ix)-1)
                    index = int(ix[k])
                    cache_key = (cell, index)
                    if cache_key not in cached:
                        view.d = d
                        view.max_error = 0.
                        rendered = view.render(index)
                        pose_errors[trial["directory"]] = max(pose_errors.get(trial["directory"], 0.), view.max_error)
                        if k == len(ix)-1:
                            cached[cache_key] = rendered
                    else:
                        rendered = cached[cache_key]
                    draw.rectangle((x, y, x+PANEL-2, y+62), fill=(25, 36, 49))
                    draw.text((x+14, y+3), LABELS[trial["label"]], font=fonts[22], fill=COLORS[col])
                    draw.text((x+14, y+34), detail(view, trial["label"]), font=fonts[18], fill=(194, 209, 222))
                    canvas.paste(rendered, (x, y+64))
                    draw.rectangle((x+9, y+74, x+220, y+105), fill=(22, 32, 44))
                    draw.text((x+17, y+76), MOTION_LABELS[row], font=fonts[20], fill=(240, 245, 250))
                    result = meta["last_episode"]
                    if terminal:
                        failed = result["failed"]
                        outcome = "提前终止 / STOP" if failed else "动作完成 / DONE" if result["motion_finished"] else "截断 / END"
                        color = (255, 137, 123) if failed else (128, 225, 171)
                        draw.rectangle((x+8, y+313, x+PANEL-10, y+393), fill=(30, 30, 36))
                        draw.text((x+18, y+316), f"{outcome} · {result['seconds']:.2f} s", font=fonts[22], fill=color)
                        reasons = ", ".join(k for k,v in result["termination_flags"].items() if v and k not in ("time_out", "motion_finished"))
                        draw.text((x+18, y+351), (reasons[:45] if failed else "结束后保持末帧"), font=fonts[18], fill=(207, 215, 225))
                    caption = f"{Path(meta['motion']).stem} · {min(stamp, result['seconds']):.2f}s"
                    draw.text((x+10, y+402), caption, font=fonts[16], fill=(183, 200, 218))
                    draw.line((x+PANEL-1, y, x+PANEL-1, y+ROW_H-1), fill=(67, 81, 97), width=2)
                draw.text((24, height-39), "蓝脚：左  橙脚：右  |  上行从中心起步，下行从边缘起步；完整动作保留转向及往返  |  红色标记为原生跟踪终止，结束后明确保持末帧",
                          font=fonts[20], fill=(164, 183, 202))
                if fi in (0, count//2, count-1):
                    canvas.save(args.output / f"level{level}_preview_{fi:03d}.jpg", quality=92)
                if writer is not None:
                    writer.append_data(np.asarray(canvas))
                if fi % 50 == 0:
                    print(f"[VIDEO L{level}] {fi+1}/{count}", flush=True)
            if writer is not None:
                writer.close()
                writer = None
                actual_count, actual_duration = imageio_ffmpeg.count_frames_and_secs(str(temporary))
                if actual_count != count or abs(actual_duration-count/args.fps) > .03:
                    raise ValueError("Encoded video duration/frame count mismatch")
                temporary.replace(video)
                validation = dict(level=level, video=str(video), resolution=[width,height], fps=args.fps,
                    frames=count, duration_s=actual_duration, playback_speed=1,
                    physics="Isaac / PhysX", rendering="MuJoCo FK and rendering only; no mj_step",
                    source_checkpoint_sha256=suite["source_sha256"], terminal_hold_explicitly_labelled=True,
                    control_sample_hz=50, physics_sample_hz=200,
                    video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                    fk_max_error_m=pose_errors)
                validations.append(validation)
                (args.output / f"level{level}_validation.json").write_text(json.dumps(validation, indent=2)+"\n")
        finally:
            if writer is not None:
                writer.close()
            for view in shared_views.values():
                view.renderer.close()
    if args.preview_only:
        return
    fields = ["level", "motion", "label", "terrain", "direction", "seconds", "control_steps", "failed", "motion_finished", "termination_flags"]
    with (args.output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**r, "termination_flags": ",".join(k for k,v in r["termination_flags"].items() if v)} for r in all_results)
    if len(validations) == len(suite["levels"]):
        playlist = args.output / "concat.txt"
        playlist.write_text("".join("file '" + str(Path(v['video']).name) + "'\n" for v in validations))
        combined = args.output / "all_levels_1_2_3_4.mp4"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-n",
            "-f", "concat", "-safe", "0", "-i", str(playlist), "-c", "copy", "-movflags", "+faststart", str(combined)], check=True)
        n, seconds = imageio_ffmpeg.count_frames_and_secs(str(combined))
        assert n == sum(v["frames"] for v in validations)
        (args.output / "combined_validation.json").write_text(json.dumps(dict(frames=n,duration_s=seconds,
            source_videos=[v["video"] for v in validations], video_sha256=hashlib.sha256(combined.read_bytes()).hexdigest()),indent=2)+"\n")
    (args.output / "video_validation.json").write_text(json.dumps(validations, indent=2)+"\n")


if __name__ == "__main__":
    main()
