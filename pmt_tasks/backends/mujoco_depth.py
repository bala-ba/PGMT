"""Occlusion-aware MuJoCo camera source for the sensor-independent mapper."""
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from pmt_tasks.depth_perception import CameraIntrinsics, LocalElevationMap


def decode_id_colors(rgb, scene):
    """Decode discrete render IDs; unknown labels stay invalid, never get clipped.

    MuJoCo 3.3.5 Renderer.render(segmentation=True) indexes a (ngeom+1)
    lookup directly. Driver-dependent mixed edge colors can exceed that range.
    Scene segmentation IDs also need not be model geometry IDs or contiguous.
    """
    colors = np.asarray(rgb, dtype=np.uint32)
    labels = colors[..., 0] | (colors[..., 1] << 8) | (colors[..., 2] << 16)
    visible = [g for g in scene.geoms[:scene.ngeom] if g.segid >= 0]
    largest = max((int(g.segid) + 1 for g in visible), default=0)
    lookup = np.full((largest + 1, 2), -1, dtype=np.int32)
    known = np.zeros(largest + 1, dtype=bool)
    known[0] = True  # background is an expected invalid pixel
    for geom in visible:
        label = int(geom.segid) + 1
        lookup[label] = [geom.objid, geom.objtype]
        known[label] = True
    output = np.full((*labels.shape, 2), -1, dtype=np.int32)
    in_bounds = labels < len(lookup)
    output[in_bounds] = lookup[labels[in_bounds]]
    unexpected = int((~in_bounds).sum() + (~known[labels[in_bounds]]).sum())
    return output, unexpected


def render_segmentation(renderer):
    """Read raw ID colors through the public renderer API and validate the IDs."""
    renderer.disable_depth_rendering()
    renderer.disable_segmentation_rendering()
    saved_flags = renderer.scene.flags.copy()
    try:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = True
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = True
        return decode_id_colors(renderer.render(), renderer.scene)
    finally:
        renderer.scene.flags[:] = saved_flags


class MujocoDepthPerception:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __init__(self, model, config, output=None, record=False):
        self.model, self.config = model, dict(config)
        self.camera_id = model.camera("pmt_depth").id
        self.torso_id = model.body("torso_link").id
        robot_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if robot_root < 0:
            robot_root = self.torso_id
        robot_bodies = {robot_root}
        for body_id in range(robot_root + 1, model.nbody):
            if int(model.body_parentid[body_id]) in robot_bodies:
                robot_bodies.add(body_id)
        self.robot_geoms = np.isin(model.geom_bodyid, list(robot_bodies))
        self.intrinsics = CameraIntrinsics.from_fovy(config["width"], config["height"], config["fovy_deg"])
        self.map = LocalElevationMap(max_age_s=config["max_age_s"])
        # A categorical ID buffer must not interpolate colors across object edges.
        # Disable MSAA for this sensor's framebuffer only, preserving viewer settings.
        samples = model.vis.quality.offsamples
        try:
            model.vis.quality.offsamples = 0
            self.renderer = mujoco.Renderer(model, height=config["height"], width=config["width"], max_geom=10000)
        finally:
            model.vis.quality.offsamples = samples
        self.options = mujoco.MjvOption()
        # Keep robot geometry in depth rendering: masking must not reveal behind it.
        self.options.geomgroup[:] = 1
        self.output = Path(output) if output else None
        self.record = record
        self.writer = None
        self.frames = []
        self.reset()

    def reset(self):
        self.map.reset()
        self.next_capture = -np.inf
        self.capture_count = 0
        self.coverage = []
        self.processing_seconds = []
        self.frames.clear()
        self.last_depth = None
        self.last_depth_timestamp = None
        self.invalid_segmentation_pixels = 0

    def scan(self, data):
        import time

        torso = data.xpos[self.torso_id].copy()
        r = data.xmat[self.torso_id].reshape(3, 3)
        yaw = np.arctan2(r[1, 0], r[0, 0])
        captured = data.time + 1e-8 >= self.next_capture
        if captured:
            tick = time.monotonic()
            self.renderer.update_scene(data, camera=self.camera_id, scene_option=self.options)
            self.renderer.enable_depth_rendering()
            raw_depth = self.renderer.render().copy()
            segmentation, invalid_pixels = render_segmentation(self.renderer)
            self.invalid_segmentation_pixels += invalid_pixels
            geom = segmentation[..., 0]
            is_geom = (segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (geom >= 0) & (geom < self.model.ngeom)
            # Identify only robot pixels, not terrain labels. Optical occlusion is
            # already resolved by the depth buffer including ALL scene geometry.
            self_mask = np.zeros(geom.shape, dtype=bool)
            self_mask[is_geom] = self.robot_geoms[geom[is_geom]]
            from scipy.ndimage import binary_dilation
            self_mask = binary_dilation(self_mask, iterations=1)
            depth = np.where(is_geom & ~self_mask & (raw_depth < 5.) & (raw_depth >= .15), raw_depth, 0.)
            self.last_depth = depth.copy()
            self.last_depth_timestamp = float(data.time)
            pose = np.eye(4)
            pose[:3, :3] = data.cam_xmat[self.camera_id].reshape(3, 3) @ np.diag([1., -1., -1.])
            pose[:3, 3] = data.cam_xpos[self.camera_id]
            stats = self.map.integrate(depth, self.intrinsics, pose, data.time, torso[:2])
            self.capture_count += 1
            self.next_capture = data.time + 1 / self.config["hz"]
            self.processing_seconds.append(time.monotonic() - tick)
            if self.record:
                self.frames.append((depth.astype(np.float32), pose.copy(), float(data.time), torso, float(yaw)))
        scan = self.map.sample(torso, yaw, float(data.time))
        self.coverage.append(float(np.mean(scan["policy_valid"])))
        if captured and self.record:
            self.renderer.disable_segmentation_rendering()
            rgb = self.renderer.render().copy()
            self._write_visual(rgb, raw_depth, depth, scan, data.time, stats)
        return scan

    def _write_visual(self, rgb, raw_depth, depth, scan, timestamp, stats):
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw
        from scripts.height_scan_visualization import colors, draw_panel

        if self.writer is None:
            self.writer = imageio.get_writer(str(self.output / "depth_perception.mp4"), fps=self.config["hz"], codec="libx264", quality=7, macro_block_size=1)
        image = Image.new("RGB", (944, 480), (17, 24, 35))
        draw = ImageDraw.Draw(image)
        draw.text((12, 10), f"PMT depth camera -> elevation map -> student | t={timestamp:.2f}s", fill="white")
        draw.text((12, 35), "Camera RGB (robot occludes terrain)", fill="white")
        draw.text((332, 35), "Depth Z: 0.15..5m | pink = masked", fill="white")
        image.paste(Image.fromarray(rgb).resize((320, 240)), (0, 55))
        col = colors(np.clip(raw_depth, .15, 5.), .15, 5.)
        col[depth == 0] = [234, 86, 193]
        image.paste(Image.fromarray(col).resize((320, 240)), (320, 55))
        config = {"shape_xy": [17, 11], "ordering": "xy", "resolution_m": .1}
        image.paste(draw_panel(scan, config), (640, 55))
        draw.multiline_text((12, 315),
                            f"Camera: {self.config['width']}x{self.config['height']} @ {self.config['hz']:g} Hz\n"
                            f"Map coverage: {np.mean(scan['policy_valid']):.1%}; observed surface points: {stats['surface_points']}\n"
                            f"Map age limit: {self.config['max_age_s']:g}s; unknown cells stay invalid\n"
                            "Simulated camera; ground-truth pose used as odometry\n"
                            "No terrain height queries in the depth observation path", fill="white", spacing=12)
        self.writer.append_data(np.asarray(image))
        if self.capture_count in (1, 25):
            image.save(self.output / "depth_preview.png")

    def metrics(self):
        return {"source": "rendered_depth_camera", "camera": self.config, "intrinsics": asdict(self.intrinsics),
                "depth_frames": self.capture_count, "mean_policy_coverage": float(np.mean(self.coverage)),
                "min_policy_coverage": float(np.min(self.coverage)), "max_policy_coverage": float(np.max(self.coverage)),
                "mean_capture_mapping_ms": 1000 * float(np.mean(self.processing_seconds)),
                "invalid_segmentation_pixels": self.invalid_segmentation_pixels,
                "sensor_msaa_samples": 0,
                "pose_source": "MuJoCo ground truth (replace with synchronized odometry on hardware)",
                "self_filter": "rendered robot pixel mask, dilated one pixel; no see-through rendering"}

    def close(self):
        try:
            if self.writer is not None:
                self.writer.close()
            if self.record and self.frames:
                depth, poses, stamps, torso, yaw = zip(*self.frames)
                k = self.intrinsics
                np.savez_compressed(self.output / "depth_frames.npz", depth_m=np.stack(depth),
                                    world_from_camera=np.stack(poses), timestamps=np.array(stamps),
                                    torso_position=np.stack(torso), torso_yaw=np.array(yaw),
                                    intrinsics=np.array([k.width, k.height, k.fx, k.fy, k.cx, k.cy]))
        finally:
            self.renderer.close()
