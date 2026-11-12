"""Seven-family Stage2 geometry following docs/pbfm_terrain.md.

Isaac Lab supplies all mesh generators except Perlin noise. Its native height
field converter also triangulates the Perlin field, with identical collision
and scan geometry. The older v6 generator remains available for reproducibility.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh
import yaml

from pmt_tasks.pgmt_patch_terrain import seed_for
from pmt_tasks.pgmt_terrain import ray_heights
from pmt_tasks.pgmt_terrain_distribution import terrain_proportions


FAMILIES = ("flat", "rough", "stairs_up", "stairs_down", "platforms", "slope", "boxes")
NATIVE_INTERFACES = dict(flat="MeshPlaneTerrainCfg", rough="PerlinRoughTerrainCfg",
    stairs_up="MeshPyramidStairsTerrainCfg", stairs_down="MeshInvertedPyramidStairsTerrainCfg",
    platforms="MeshRandomGridTerrainCfg", slope="HfInvertedPyramidSlopedTerrainCfg",
    boxes="HfDiscreteObstaclesTerrainCfg")
REPEATED_FAMILIES = ("flat", "stairs_up", "stairs_down", "slope")
INVERTED_FAMILIES = ("stairs_down", "slope")


def load_patch_spec(path):
    spec = yaml.safe_load(Path(path).read_text())
    if tuple(spec["families"]) != FAMILIES or spec["num_rows"] != 10 or spec["num_cols"] != len(FAMILIES):
        raise ValueError("PBFM Stage2 requires seven families and ten levels")
    if not spec["curriculum"] or spec["difficulty_range"] != [0., 1.]:
        raise ValueError("PBFM Stage2 requires curriculum over [0, 1]")
    terrain_proportions(spec["families"], spec["family_proportions"])
    if spec["patch_grid"] != [4, 4] or spec["patch_size"] != [6., 6.]:
        raise ValueError("PBFM Stage2 requires 4 x 4 patches of 6 x 6 m")
    np.testing.assert_allclose(np.array(spec["patch_grid"]) * spec["patch_size"], spec["region_size"])
    for family, cls in NATIVE_INTERFACES.items():
        if spec[family]["class_name"] != cls:
            raise ValueError(f"Unsupported terrain interface for {family}")
    for family, key in (("rough", "height_range"), ("stairs_up", "step_height_range"),
                        ("stairs_down", "step_height_range"), ("platforms", "grid_height_range"),
                        ("slope", "slope_range"), ("boxes", "obstacle_height_range")):
        lo, hi = spec[family][key]
        if not 0 <= lo <= hi or not math.isfinite(hi):
            raise ValueError(f"Invalid {family}.{key}")
    rough = spec["rough"]
    if (rough["wavelength"] <= 0 or type(rough["octaves"]) is not int or rough["octaves"] < 1
            or not 0 < rough["persistence"] <= 1 or rough["lacunarity"] < 1):
        raise ValueError("Invalid Perlin noise spectrum")
    width = spec["platforms"]["grid_width"]
    np.testing.assert_allclose(6 - int(6 / width) * width, spec["platforms"]["connection_width"], atol=1e-8)
    return spec


def parameters(spec, difficulty):
    d = float(difficulty)
    if not math.isfinite(d) or not 0 <= d <= 1:
        raise ValueError("difficulty must be finite and in [0, 1]")
    def lerp(family, key):
        lo, hi = spec[family][key]
        return float(lo + d * (hi - lo))
    slope = lerp("slope", "slope_range")
    return dict(difficulty=d, rough_height_amplitude_m=lerp("rough", "height_range"),
        stair_up_height_m=lerp("stairs_up", "step_height_range"),
        stair_down_height_m=lerp("stairs_down", "step_height_range"),
        platform_height_amplitude_m=lerp("platforms", "grid_height_range"),
        slope_coefficient=slope, slope_angle_deg=math.degrees(math.atan(slope)),
        obstacle_height_m=lerp("boxes", "obstacle_height_range"))


def perlin_noise(shape, spacing, wavelength, *, rng, octaves, persistence, lacunarity):
    """Seeded gradient Perlin noise, normalized to maximum absolute value one."""
    xx, yy = np.meshgrid(np.arange(shape[0]) * spacing, np.arange(shape[1]) * spacing, indexing="ij")
    noise = np.zeros(shape, dtype=float)
    weight = 1.
    for octave in range(octaves):
        frequency = lacunarity ** octave / wavelength
        x, y = xx * frequency, yy * frequency
        ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = x - ix, y - iy
        angles = rng.uniform(0, 2 * np.pi, size=(int(ix.max()) + 2, int(iy.max()) + 2))
        gx, gy = np.cos(angles), np.sin(angles)
        def dot(dx, dy):
            return gx[ix + dx, iy + dy] * (fx - dx) + gy[ix + dx, iy + dy] * (fy - dy)
        u, v = fx ** 3 * (fx * (fx * 6 - 15) + 10), fy ** 3 * (fy * (fy * 6 - 15) + 10)
        a, b, c, e = dot(0, 0), dot(1, 0), dot(0, 1), dot(1, 1)
        noise += weight * ((1 - v) * ((1 - u) * a + u * b) + v * ((1 - u) * c + u * e))
        weight *= persistence
    maximum = np.abs(noise).max()
    if maximum == 0:
        raise ValueError("Perlin sampling produced a flat field; change the noise wavelength")
    return noise / maximum


def perlin_height_field(difficulty, cfg):
    """Raw discrete field for Isaac Lab's height_field_to_mesh decorator."""
    shape = tuple(int(s / cfg.horizontal_scale) for s in cfg.size)
    noise = perlin_noise(shape, cfg.horizontal_scale, cfg.wavelength, rng=np.random.default_rng(cfg.seed),
                         octaves=cfg.octaves, persistence=cfg.persistence, lacunarity=cfg.lacunarity)
    height = cfg.height_range[0] + difficulty * (cfg.height_range[1] - cfg.height_range[0])
    return np.rint(noise * height / cfg.vertical_scale).astype(np.int16)


def native_config(spec, family, seed):
    import isaaclab.terrains as tg

    cfg = spec[family]
    size = tuple(spec["patch_size"])
    if family == "rough":
        from isaaclab.terrains.height_field.utils import height_field_to_mesh
        from isaaclab.utils import configclass

        @configclass
        class PerlinRoughTerrainCfg(tg.HfTerrainBaseCfg):
            function = height_field_to_mesh(perlin_height_field)
            height_range: tuple[float, float] = (.02, .06)
            wavelength: float = 1.2
            octaves: int = 3
            persistence: float = .5
            lacunarity: float = 2.
            seed: int = 1731

        return PerlinRoughTerrainCfg(size=size, seed=seed, height_range=tuple(cfg["height_range"]),
            horizontal_scale=cfg["horizontal_scale"], vertical_scale=spec["vertical_scale"],
            border_width=cfg["border_width"], slope_threshold=cfg["slope_threshold"],
            **{key: cfg[key] for key in ("wavelength", "octaves", "persistence", "lacunarity")})
    cls = getattr(tg, cfg["class_name"])
    if family == "flat":
        return cls(size=size)
    if family in ("stairs_up", "stairs_down"):
        return cls(size=size, step_height_range=tuple(cfg["step_height_range"]),
            step_width=cfg["step_width"], platform_width=cfg["platform_width"],
            border_width=cfg["border_width"], holes=False)
    if family == "platforms":
        return cls(size=size, grid_height_range=tuple(cfg["grid_height_range"]), grid_width=cfg["grid_width"],
                   platform_width=cfg["platform_width"], holes=cfg["holes"])
    common = dict(size=size, platform_width=cfg["platform_width"], border_width=cfg["border_width"],
                  horizontal_scale=spec["horizontal_scale"], vertical_scale=spec["vertical_scale"])
    if family == "slope":
        return cls(**common, slope_range=tuple(cfg["slope_range"]), slope_threshold=None)
    return cls(**common, obstacle_height_range=tuple(cfg["obstacle_height_range"]),
        obstacle_width_range=tuple(cfg["obstacle_width_range"]), num_obstacles=cfg["num_obstacles"],
        obstacle_height_mode=cfg["obstacle_height_mode"], slope_threshold=cfg["slope_threshold"])


def audit_patch(mesh, origin, spec, family, difficulty):
    """Check the actual mesh heights, steps, gradients, coverage and zero seams."""
    if not np.isfinite(mesh.vertices).all() or mesh.area_faces.min() < 1e-12:
        raise ValueError(f"Invalid {family} mesh")
    size = np.asarray(spec["patch_size"])
    np.testing.assert_allclose(mesh.bounds[:, :2], [np.zeros(2), size], atol=1e-5)
    p, cfg = parameters(spec, difficulty), spec[family]
    # Avoid exact height-field/grid edges: neighboring vertical faces can
    # produce different valid ray hits after world-coordinate float rounding.
    xx, yy = np.meshgrid(np.linspace(.031, size[0] - .043, 17),
                         np.linspace(.019, size[1] - .057, 17), indexing="ij")
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    audit = dict(vertices=len(mesh.vertices), triangles=len(mesh.faces), bounds=mesh.bounds.tolist())
    if family == "flat":
        np.testing.assert_allclose(mesh.vertices[:, 2], 0., atol=1e-7)
    elif family in ("stairs_up", "stairs_down"):
        sign = 1 if family == "stairs_up" else -1
        height = p["stair_up_height_m" if sign > 0 else "stair_down_height_m"]
        tops = np.unique(np.round(mesh.triangles_center[mesh.face_normals[:, 2] > .99, 2], 7))
        np.testing.assert_allclose(np.diff(tops), height, atol=2e-6)
        n = len(tops) - 2
        w, border = cfg["step_width"], cfg["border_width"]
        tread_xy = np.column_stack((border + (np.arange(n) + .5) * w, np.full(n, size[1] / 2 + .031)))
        np.testing.assert_allclose(ray_heights(mesh, tread_xy), sign * height * np.arange(1, n + 1), atol=2e-6)
        xy = np.vstack((xy, tread_xy))
        for direction in ([1, 0], [-1, 0], [0, 1], [0, -1]):
            line = size / 2 + np.linspace(0, size[0] / 2 - .037, 101)[:, None] * direction
            heights = sign * ray_heights(mesh, line)
            if np.diff(heights).max() > 2e-6 or heights[0] <= heights[-1]:
                raise ValueError("Stairs must have the configured pyramid direction on all four sides")
        audit.update(measured_step_height_m=float(np.diff(tops).mean()), verified_tread_depth_m=w,
                     ring_treads=n, inverted=sign < 0, four_sided_direction_verified=True)
    elif family == "slope":
        points = np.array([[.5, size[1] / 2 + .031], [1.1, size[1] / 2 + .031]])
        heights = ray_heights(mesh, points)
        slope = -(heights[1] - heights[0]) / .6
        tolerance = spec["vertical_scale"] / .6 + .02 * p["slope_coefficient"]
        if abs(slope - p["slope_coefficient"]) > tolerance:
            raise ValueError("Slope coefficient differs from the configured rise/run")
        for direction in ([1, 0], [-1, 0], [0, 1], [0, -1]):
            line = size / 2 + np.linspace(0, size[0] / 2 - .037, 101)[:, None] * direction
            z = ray_heights(mesh, line)
            if np.diff(z).min() < -2e-6 or (p["slope_coefficient"] > .01 and z[-1] <= z[0]):
                raise ValueError("Slope must rise outward on all four sides")
        xy = np.vstack((xy, points))
        audit.update(measured_slope_coefficient=float(slope), inverted=True)
    elif family == "platforms":
        width = cfg["grid_width"]
        count = int(size[0] / width)
        border = (size[0] - count * width) / 2
        axis = border + (np.arange(count) + .5) * width
        xx, yy = np.meshgrid(axis, axis, indexing="ij")
        centers = np.column_stack((xx.ravel(), yy.ravel()))
        heights = ray_heights(mesh, centers)
        if not heights.min() < 0 < heights.max() or abs(heights).max() > p["platform_height_amplitude_m"] + 2e-6:
            raise ValueError("Platform grid heights must be sampled in [-H, H]")
        np.testing.assert_allclose(2 * border, cfg["connection_width"], atol=1e-8)
        xy = centers
        audit.update(grid_count=count ** 2, connection_width_m=2 * border,
                     measured_top_height_min_m=float(heights.min()), measured_top_height_max_m=float(heights.max()))
    elif family == "rough":
        heights = mesh.vertices[:, 2]
        if not heights.min() < 0 < heights.max() or abs(heights).max() > p["rough_height_amplitude_m"] + spec["vertical_scale"] / 2 + 1e-6:
            raise ValueError("Perlin heights differ from the configured amplitude")
        audit.update(measured_height_min_m=float(heights.min()), measured_height_max_m=float(heights.max()))
    else:
        heights = mesh.vertices[:, 2]
        if (mesh.face_normals[:, 2] < -1e-8).any():
            raise ValueError("Discrete obstacle height field has folded downward-facing triangles")
        maximum = math.floor(p["obstacle_height_m"] / spec["vertical_scale"]) * spec["vertical_scale"]
        np.testing.assert_allclose(heights.max(), maximum, atol=2e-6)
        if heights.min() < -1e-6:
            raise ValueError("Fixed discrete obstacles must be positive pillars")
        audit.update(measured_obstacle_height_m=float(heights.max()))
    # The zero-height boundary must not cover negative interior surfaces.
    axis = np.linspace(.031, size[0] - .031, 17)
    edges = np.vstack([np.column_stack((axis, np.full_like(axis, edge))) for edge in (.0001, size[1] - .0001)] +
                      [np.column_stack((np.full_like(axis, edge), axis)) for edge in (.0001, size[0] - .0001)])
    np.testing.assert_allclose(ray_heights(mesh, edges), 0., atol=2e-6)
    xy = np.vstack((xy, edges, origin[None, :2]))
    heights = ray_heights(mesh, xy)
    if not np.isfinite(heights).all():
        raise ValueError("Terrain mesh has a hole in its sampled surface")
    np.testing.assert_allclose(heights[-1], origin[2], atol=2e-6)
    audit.update(probes=np.column_stack((xy, heights)).tolist(), zero_border_verified=True)
    return audit


def make_patch(spec, level, family, patch_id, difficulty):
    seed = seed_for(spec, "texture", level, family, patch_id)
    cfg = native_config(spec, family, seed)
    if family == "platforms":
        import torch
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            meshes, origin = cfg.function(difficulty, cfg)
    else:
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            meshes, origin = cfg.function(difficulty, cfg)
        finally:
            np.random.set_state(state)
    removed = 0
    for part in meshes:
        before = len(part.faces)
        part.update_faces(part.nondegenerate_faces())
        part.remove_unreferenced_vertices()
        removed += before - len(part.faces)
    mesh = trimesh.util.concatenate([part for part in meshes if len(part.faces)])
    native_origin_z = float(origin[2])
    # Query actual support instead of the native 2 m maximum-height origin.
    origin = np.r_[np.array(spec["patch_size"]) / 2 + [.031, .019], 0.]
    origin[2] = ray_heights(mesh, origin[None, :2])[0]
    audit = audit_patch(mesh, origin, spec, family, difficulty)
    audit.update(native_generator=type(cfg).__name__, native_function=f"{cfg.function.__module__}.{cfg.function.__name__}",
                 native_origin_z_m=native_origin_z, removed_degenerate_faces=removed)
    preview = spec["spawn_preview"]
    rng = np.random.default_rng(seed_for(spec, "preview_xy_yaw", level, family, patch_id))
    xy = rng.uniform(preview["boundary_margin"], np.array(spec["patch_size"]) - preview["boundary_margin"],
                     size=(preview["samples_per_patch"], 2))
    samples = np.column_stack((xy, ray_heights(mesh, xy), rng.uniform(*preview["yaw_range"], size=len(xy))))
    return mesh, origin, audit, samples, parameters(spec, difficulty), seed
