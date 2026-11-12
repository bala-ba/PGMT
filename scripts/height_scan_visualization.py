"""Display recorded ray hits and the exact policy scan without resampling terrain."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


INVALID_COLOR = np.array([234, 86, 193], dtype=np.uint8)
PALETTE = np.array([[35, 64, 145], [34, 170, 183], [105, 205, 123], [253, 210, 84]], dtype=float)


class LiveHeightPanel:
    """Independent X11 panel, avoiding MuJoCo 3.3.5's unstable set_images path."""

    def __init__(self, height=372):
        import tkinter as tk
        from PIL import ImageTk

        self.image_type = ImageTk.PhotoImage
        self.root = tk.Tk()
        self.root.title("PMT | Local height scan")
        x = max(0, min(1288, self.root.winfo_screenwidth() - 316))
        self.root.geometry(f"304x{height}+{x}+12")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.root.withdraw)
        self.label = tk.Label(self.root, borderwidth=0, highlightthickness=0)
        self.label.pack()
        self.photo = None

    def update(self, panel):
        self.photo = self.image_type(panel, master=self.root)
        self.label.configure(image=self.photo)
        self.root.update_idletasks()
        self.root.update()

    def close(self):
        self.root.destroy()


def colors(values, low, high):
    t = np.clip((np.asarray(values) - low) / (high - low), 0, 1) * (len(PALETTE) - 1)
    lo = np.minimum(t.astype(int), len(PALETTE) - 2)
    return (PALETTE[lo] * (1 - (t - lo)[..., None]) + PALETTE[lo + 1] * (t - lo)[..., None]).astype(np.uint8)


def spatial_grid(values, config):
    """Place +x forward at the top and +y left, respecting RayCaster ordering."""
    nx, ny = config["shape_xy"]
    values = np.asarray(values)
    if values.size != nx * ny:
        raise ValueError("Height scan dimensions do not match metadata.")
    if config["ordering"] == "xy":
        grid = values.reshape(ny, nx).T
    elif config["ordering"] == "yx":
        grid = values.reshape(nx, ny)
    else:
        raise ValueError(f"Unknown height scan ordering: {config['ordering']}")
    return grid[::-1, ::-1]


def draw_panel(scan, config):
    """Two spatially aligned heatmaps; policy noise/dropout is preserved exactly."""
    panel = Image.new("RGB", (304, 372), (17, 24, 35))
    draw = ImageDraw.Draw(panel)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 13)
    title = ImageFont.truetype(font_path, 16)
    draw.text((12, 10), "LOCAL HEIGHT SCAN", font=title, fill="white")
    nx, ny = config["shape_xy"]
    draw.text((12, 35), f"{nx} x {ny} points | {config['resolution_m'] * 100:g} cm spacing", font=font, fill="#b6c5d7")
    draw.text((12, 56), "Front (+x) up / Left (+y) left", font=font, fill="#b6c5d7")
    draw.text((12, 82), "Terrain z (m)", font=font, fill="white")
    draw.text((164, 82), "Policy scan (m)", font=font, fill="white")
    hit_valid = np.asarray(scan["hit_valid"], dtype=bool)
    heights = np.asarray(scan["hits_w"])[:, 2]
    layers = [(12, heights, hit_valid, 0.0, 0.4)]
    if "policy_height" in scan:
        layers.append((164, np.asarray(scan["policy_height"]), np.asarray(scan["policy_valid"], dtype=bool), -0.5, 1.5))
    else:
        draw.multiline_text((164, 148), "No policy scan\n(blind teacher)", font=font, fill="#b6c5d7", spacing=8)
    for x, values, valid, low, high in layers:
        grid = spatial_grid(values, config)
        mask = spatial_grid(valid, config)
        rgb = colors(grid, low, high)
        rgb[~mask] = INVALID_COLOR
        # Nearest-neighbor cells keep missing rays and discrete steps visible.
        heatmap = Image.fromarray(rgb).resize((110, 170), Image.Resampling.NEAREST)
        panel.paste(heatmap, (x + 12, 105))
        draw.rectangle((x + 12, 105, x + 121, 274), outline="#77889b")
        center_x, center_y = x + 67, 190
        draw.line((center_x - 4, center_y, center_x + 4, center_y), fill="white", width=1)
        draw.line((center_x, center_y - 4, center_x, center_y + 4), fill="white", width=1)
        bar = Image.fromarray(colors(np.linspace(low, high, 110)[None, :], low, high)).resize((110, 7))
        panel.paste(bar, (x + 12, 282))
        draw.text((x + 12, 292), f"{low:g} ... {high:g}", font=font, fill="#b6c5d7")
        if valid.any():
            draw.text((x, 314), f"{values[valid].min():.2f} to {values[valid].max():.2f}", font=font, fill="white")
    draw.rectangle((12, 344, 22, 354), fill=tuple(INVALID_COLOR))
    fraction = 1 - np.mean(scan.get("policy_valid", scan["hit_valid"]))
    draw.text((29, 340), f"Invalid / dropout: {fraction:.0%}", font=font, fill="#b6c5d7")
    return panel


def draw_depth_panel(depth_m, age_s):
    """Recorded optical depth, with invalid/self-masked pixels shown explicitly."""
    panel = Image.new("RGB", (304, 244), (17, 24, 35))
    draw = ImageDraw.Draw(panel)
    draw.text((12, 8), f"DEPTH CAMERA | frame age {age_s:.2f}s", fill="white")
    depth = np.asarray(depth_m)
    valid = np.isfinite(depth) & (depth > 0)
    rgb = colors(np.where(valid, depth, .15), .15, 5.)
    rgb[~valid] = INVALID_COLOR
    panel.paste(Image.fromarray(rgb).resize((280, 200)), (12, 30))
    return panel


def add_scan_geoms(scene, scan, config):
    """Overlay measured hit points and grid edges on the actual world geometry."""
    import mujoco

    hits = np.asarray(scan["hits_w"], dtype=float).copy()
    hits[:, 2] += 0.012  # avoid z-fighting with the terrain
    valid = np.asarray(scan["hit_valid"], dtype=bool)
    policy_valid = np.asarray(scan.get("policy_valid", valid), dtype=bool)
    rgb = colors(hits[:, 2] - 0.012, 0, 0.4)
    rgb[~policy_valid] = INVALID_COLOR
    indices = spatial_grid(np.arange(len(hits)), config)
    edges = np.concatenate((np.stack((indices[:-1, :].ravel(), indices[1:, :].ravel()), axis=1),
                            np.stack((indices[:, :-1].ravel(), indices[:, 1:].ravel()), axis=1)))
    needed = int(valid.sum() + np.sum(valid[edges].all(axis=1)))
    if scene.ngeom + needed > scene.maxgeom:
        raise ValueError("MuJoCo scene lacks space for height scan overlay.")
    for a, b in edges:
        if not (valid[a] and valid[b]):
            continue
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.array([0.3, 0.85, 0.9, 0.6]))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 1.0, hits[a], hits[b])
        scene.ngeom += 1
    for i in np.flatnonzero(valid):
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                           np.full(3, 0.013), hits[i], np.eye(3).ravel(), np.r_[rgb[i] / 255, 1.0])
        scene.ngeom += 1
