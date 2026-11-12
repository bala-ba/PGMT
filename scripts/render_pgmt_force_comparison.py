"""Render synchronized recorded Isaac trajectories; MuJoCo is rendering only.

No policy evaluation or mj_step is performed here. Named-joint FK is checked
against every recorded Isaac body before any video frame is accepted.
Run with MUJOCO_GL=egl and a host GPU, after --record_video_states acquisition.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

from pgmt_force_analysis import load_run, validate_pairing

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("flat", "stairs", "slope")
LABELS = ("平地", "楼梯", "斜坡")
COLORS = ((92, 171, 224), (236, 168, 91), (104, 192, 144))
FOOT_COLORS = ((72, 190, 248), (255, 155, 75))
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
WIDTH, HEIGHT, PANEL, RENDER_H = 1920, 960, 640, 560


def build_model(run, output):
    meta = run["metadata"]
    asset = ROOT / "TCRS/assets/g1/g1_29dof_rev_1_0.xml"
    root = ET.parse(asset).getroot()
    root.find("compiler").set("meshdir", str(asset.parent / "meshes"))
    for visual in list(root.findall("visual")):
        root.remove(visual)
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(PANEL), offheight=str(RENDER_H))
    ET.SubElement(visual, "quality", shadowsize="2048", offsamples="4")
    ET.SubElement(visual, "headlight", ambient="0.25 0.25 0.25", diffuse="0.35 0.35 0.35", specular="0 0 0")
    ET.SubElement(visual, "rgba", haze="0.78 0.83 0.87 1")
    assets, world = root.find("asset"), root.find("worldbody")
    for body in root.findall("worldbody"):
        for light in list(body.findall("light")):
            body.remove(light)
        for geom in list(body.findall("geom")):
            if geom.get("type") == "plane":
                body.remove(geom)
    for group in root.findall("asset"):
        for texture in list(group.findall("texture")):
            if texture.get("type") == "skybox":
                group.remove(texture)
    ET.SubElement(world, "light", pos="-2 -4 8", directional="true", dir="1 1 -3", diffuse=".55 .55 .55", specular="0 0 0")
    ET.SubElement(assets, "texture", name="sky", type="skybox", builtin="gradient",
                  rgb1=".64 .73 .81", rgb2=".88 .91 .94", width="512", height="3072")
    ET.SubElement(assets, "texture", name="floor_tex", type="2d", builtin="checker",
                  width="512", height="512", rgb1=".34 .40 .46", rgb2=".43 .50 .56")
    ET.SubElement(assets, "material", name="floor_mat", texture="floor_tex", texrepeat="2 2",
                  texuniform="true", reflectance="0", specular="0")
    terrain_root = Path(meta["video_states"]["terrain_asset"])
    manifest = json.loads((terrain_root / "manifest.json").read_text())
    if hashlib.sha256((terrain_root / "manifest.json").read_bytes()).hexdigest() != meta["force_recording"]["terrain_manifest_sha256"]:
        raise ValueError("Terrain changed since recording")
    args = meta["arguments"]
    family = str(run["data"]["terrain"][0])
    patch = next(p for p in manifest["patches"] if p["family"] == family
                 and p["level"] == args["level"] and p["patch_id"] == args["patch_id"])
    origin = np.asarray(patch["center"])
    if family == "flat":
        ET.SubElement(world, "geom", type="plane", size="0 0 .02", group="2", material="floor_mat", contype="0", conaffinity="0")
    else:
        rgb1, rgb2 = ((".50 .36 .23", ".57 .43 .29") if family == "stairs" else
                      (".27 .43 .33", ".34 .51 .40"))
        ET.SubElement(assets, "texture", name="terrain_tex", type="2d", builtin="checker",
                      width="256", height="256", rgb1=rgb1, rgb2=rgb2)
        ET.SubElement(assets, "material", name="terrain_mat", texture="terrain_tex",
                      texrepeat="1 1", reflectance="0", specular="0")
        for index, part in enumerate(manifest["patches"]):
            if (part["family"] != family or part["level"] != args["level"]
                    or abs(part["patch_row"] - patch["patch_row"]) > 1
                    or abs(part["patch_col"] - patch["patch_col"]) > 1):
                continue
            source = terrain_root / part["mesh_file"]
            if hashlib.sha256(source.read_bytes()).hexdigest() != part["mesh_sha256"]:
                raise ValueError(f"Terrain mesh changed: {source}")
            mesh = trimesh.load_mesh(source, process=False)
            mesh.apply_translation(np.asarray(part["center"]) - origin)
            mesh_path = output / f"{family}_{index}.obj"
            # Explicit world-XY UVs make the slope visible without altering any
            # terrain vertices or faces. Each checker square spans 0.5 m.
            obj = ["v " + " ".join(str(v) for v in vertex) for vertex in mesh.vertices]
            obj += [f"vt {v[0]} {v[1]}" for v in mesh.vertices]
            obj += ["vn " + " ".join(str(v) for v in normal) for normal in mesh.face_normals]
            obj += ["f " + " ".join(f"{v+1}/{v+1}/{face_id+1}" for v in face)
                    for face_id, face in enumerate(mesh.faces)]
            mesh_path.write_text("\n".join(obj) + "\n")
            name = f"terrain_{index}"
            ET.SubElement(assets, "mesh", name=name, file=str(mesh_path.resolve()))
            ET.SubElement(world, "geom", type="mesh", mesh=name, group="2", material="terrain_mat",
                          contype="0", conaffinity="0")
    for side, color in zip(("left", "right"), FOOT_COLORS):
        body = root.find(f".//body[@name='{side}_ankle_roll_link']")
        for geom in body.findall("geom"):
            if geom.get("group") == "1":
                geom.set("rgba", " ".join(str(v / 255) for v in color) + " 1")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model, origin, patch


class View:
    def __init__(self, run, output):
        self.run, self.d = run, run["data"]
        self.model, self.origin, self.patch = build_model(run, output)
        self.state = mujoco.MjData(self.model)
        names = run["metadata"]["video_states"]
        self.joints = self.model.jnt_qposadr[[self.model.joint(n).id for n in names["joint_names"]]]
        self.body_ids = [self.model.body(n).id for n in names["body_names"]]
        self.root_id = names["body_names"].index("pelvis")
        self.options = mujoco.MjvOption()
        self.options.geomgroup[:] = 0
        self.options.geomgroup[1:3] = 1
        self.camera = mujoco.MjvCamera()
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 2.8, 120., -20.
        self.renderer = mujoco.Renderer(self.model, width=PANEL, height=RENDER_H)
        self.max_error = 0.

    def render(self, index):
        d, state = self.d, self.state
        state.qpos[:3] = d["body_pos_w"][index, self.root_id] - self.origin
        state.qpos[3:7] = d["body_quat_w"][index, self.root_id]
        state.qpos[self.joints] = d["joint_pos"][index]
        state.qvel[:] = 0.
        mujoco.mj_forward(self.model, state)
        expected = d["body_pos_w"][index] - self.origin
        error = float(np.linalg.norm(state.xpos[self.body_ids] - expected, axis=-1).max())
        self.max_error = max(self.max_error, error)
        if error > .001:
            raise ValueError(f"Rendered pose differs from recorded Isaac pose by {error} m")
        self.camera.lookat[:] = state.qpos[:3] + np.array([.20, 0., -.05])
        self.renderer.update_scene(state, camera=self.camera, scene_option=self.options)
        return Image.fromarray(self.renderer.render())


def force_panel(run, index, ymax, font):
    image = Image.new("RGB", (PANEL, 200), (20, 29, 40))
    draw = ImageDraw.Draw(image)
    d = run["data"]
    # Keep every 200 Hz force sample in the plot, including brief impacts.
    times = d["time_s"]
    tmax = times[-1]
    x0, x1, y0, y1 = 54, PANEL - 18, 56, 173
    for value in (0, ymax / 2, ymax):
        y = y1 - value / ymax * (y1-y0)
        draw.line((x0, y, x1, y), fill=(55, 66, 80))
        draw.text((5, y-10), str(int(value)), font=font, fill=(162, 179, 195))
    draw.text((10, 3), "地形法向力 Fz / N", font=font, fill=(220, 230, 240))
    for foot in range(2):
        f = d["terrain_forces_w"][:index+1, foot, 2]
        xy = np.column_stack((x0 + times[:index+1] / tmax * (x1-x0), y1 - np.clip(f, 0, ymax) / ymax * (y1-y0)))
        if len(xy) > 1:
            draw.line([tuple(v) for v in xy], fill=FOOT_COLORS[foot], width=2)
        peak = float(d["terrain_forces_w"][max(0, index-3):index+1, foot, 2].max())
        draw.text((280 + foot*170, 3), f"{'左' if foot == 0 else '右'} {peak:4.0f} N", font=font, fill=FOOT_COLORS[foot])
    draw.text((10, 28), "数值：当前 20 ms 峰值  |  曲线：200 Hz", font=font, fill=(143, 161, 181))
    x = x0 + times[index] / tmax * (x1-x0)
    draw.line((x, y0, x, y1), fill=(230, 237, 245), width=1)
    for t in (0, 2, 4, 6, 8):
        draw.text((x0 + t/tmax*(x1-x0)-5, 175), str(t), font=font, fill=(163, 179, 196))
    draw.text((x1-32, 175), "10 s", font=font, fill=(163, 179, 196))
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=.5)
    parser.add_argument("--preview_only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.speed <= 2:
        parser.error("speed must be in (0, 2]")
    args.output.mkdir(parents=True, exist_ok=True)
    runs = {name: load_run(args.source / name) for name in NAMES}
    validate_pairing(runs)
    indices = {}
    for name, run in runs.items():
        d = run["data"]
        if "video_states" not in run["metadata"] or "body_pos_w" not in d:
            raise ValueError("Record full poses with --record_video_states first")
        if len(np.unique(d["episode"])) != 1:
            raise ValueError("Video expects exactly one recorded episode per terrain")
        decimation = run["metadata"]["force_recording"]["decimation"]
        indices[name] = np.flatnonzero(d["substep"] == decimation-1)
        for key in ("body_pos_w", "body_quat_w", "joint_pos", "terrain_forces_w"):
            if not np.isfinite(d[key]).all():
                raise ValueError(f"Nonfinite recorded {key}")
    frames = runs["flat"]["data"]["frame"][indices["flat"]]
    for name, run in runs.items():
        np.testing.assert_array_equal(run["data"]["frame"][indices[name]], frames)
    fonts = {size: ImageFont.truetype(FONT, size) for size in (18, 20, 22, 28, 32)}
    ymax = np.ceil(max(float(r["data"]["terrain_forces_w"][..., 2].max()) for r in runs.values()) / 500) * 500
    meshdir = args.output / "meshes"
    meshdir.mkdir(exist_ok=True)
    views = {}
    source_motion = Path(runs["flat"]["metadata"]["motion"]).stem
    source_checkpoint = Path(runs["flat"]["metadata"]["checkpoint"])
    video = args.output / f"{source_motion}_flat_stairs_slope.mp4"
    temporary = args.output / "encoding.mp4"
    if video.exists() and not args.preview_only:
        raise FileExistsError(video)
    writer = None
    try:
        for name, run in runs.items():
            views[name] = View(run, meshdir)
        fps = args.speed / runs["flat"]["metadata"]["control_dt"]
        if not args.preview_only:
            writer = imageio.get_writer(str(temporary), fps=fps, codec="libx264", quality=None,
                macro_block_size=16, ffmpeg_params=["-crf", "19", "-preset", "medium", "-movflags", "+faststart"])
        selected = sorted({0, len(frames)//2, len(frames)-1}) if args.preview_only else range(len(frames))
        for f in selected:
            canvas = Image.new("RGB", (WIDTH, HEIGHT), (13, 20, 30))
            draw = ImageDraw.Draw(canvas)
            draw.text((22, 8), "同一动作 · 平地 / 楼梯 / 斜坡", font=fonts[32], fill=(241, 247, 253))
            stamp = float(runs["flat"]["data"]["time_s"][indices["flat"][f]])
            draw.text((22, 50), f"{source_motion}  |  {source_checkpoint.parent.name} / {source_checkpoint.stem}  |  {args.speed:g}× 播放", font=fonts[20], fill=(154, 175, 197))
            draw.text((1390, 18), f"参考帧 {int(frames[f]):03d} / {int(frames[-1]):03d}   t = {stamp:.2f} s", font=fonts[22], fill=(227, 237, 247))
            for col, name in enumerate(NAMES):
                run, view, index = runs[name], views[name], indices[name][f]
                x = col * PANEL
                draw.rectangle((x, 88, x+PANEL, 158), fill=(24, 35, 48))
                draw.rectangle((x+18, 104, x+23, 140), fill=COLORS[col])
                draw.text((x+36, 99), LABELS[col], font=fonts[28], fill=COLORS[col])
                params = view.patch["parameters"]
                detail = ("相同局部出生点与运动方向" if name == "flat" else
                          f"台阶高 {params['stair_height_m']*100:.1f} cm" if name == "stairs" else
                          f"坡度 {params['slope_angle_deg']:.2f}°")
                draw.text((x+145, 108), detail, font=fonts[20], fill=(206, 220, 234))
                canvas.paste(view.render(index), (x, 160))
                canvas.paste(force_panel(run, index, ymax, fonts[18]), (x, 720))
                if col:
                    draw.line((x, 88, x, 920), fill=(68, 83, 98), width=2)
            draw.text((22, 928), "蓝色：左脚    橙色：右脚    |    真实仿真轨迹，按 reference 帧同步    |    Fz 为法向力世界 Z 分量，不含摩擦力", font=fonts[18], fill=(162, 180, 198))
            if f in (0, len(frames)//2, len(frames)-1):
                canvas.save(args.output / f"preview_{f:03d}.png")
            if writer is not None:
                writer.append_data(np.asarray(canvas))
            if f % 100 == 0:
                print(f"Rendered {f+1}/{len(frames)} synchronized frames", flush=True)
    finally:
        if writer is not None:
            writer.close()
        for view in views.values():
            view.renderer.close()
    if args.preview_only:
        return
    count, duration = imageio_ffmpeg.count_frames_and_secs(str(temporary))
    if count != len(frames) or abs(duration-count/fps) > .03:
        raise ValueError("Encoded frame count or duration mismatch")
    temporary.replace(video)
    result = dict(video=str(video.resolve()), frames=count, fps=fps, duration_s=duration,
        resolution=[WIDTH, HEIGHT], speed=args.speed, first_reference_frame=int(frames[0]), last_reference_frame=int(frames[-1]),
        source=str(args.source.resolve()), physics="recorded Isaac / PhysX", renderer="MuJoCo rendering only; zero mj_step calls",
        full_body_fk_max_error_m={name: v.max_error for name, v in views.items()},
        synchronized=True, frames_dropped=0, video_sha256=hashlib.sha256(video.read_bytes()).hexdigest())
    (args.output / "video_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
