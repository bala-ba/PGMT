"""PGMT terrain geometry, with lazy Isaac imports and deterministic local RNGs.

The YAML is the parameter authority. Geometry uses Isaac Lab's pyramid slope /
stairs, a contiguous positive box grid, PGMT's native random height field, and
native inverted-pyramid stairs/slopes. This module
does not change stage 1 or implement the stage 2 motion/reward curriculum.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import trimesh
import yaml

FAMILIES = ("flat", "slopes", "stairs", "boxes", "random_rough", "up_stairs", "up_slope")
FAMILY_GROUPS = {name: name for name in FAMILIES[:5]} | {"up_stairs": "stairs", "up_slope": "slopes"}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "configs/terrain/pgmt_stage2.yaml"


def load_spec(path=DEFAULT_SPEC):
    spec = yaml.safe_load(Path(path).read_text())
    if tuple(spec["families"]) != FAMILIES or spec["num_cols"] != len(FAMILIES):
        raise ValueError("Expected five PGMT terrain families followed by up_stairs / up_slope")
    if spec["num_rows"] != 10 or spec["difficulty_range"] != [0.0, 1.0] or not spec["curriculum"]:
        raise ValueError("PGMT uses ten curriculum rows over [0,1]")
    if any(float(x) <= 0 for x in spec["size"]):
        raise ValueError("Tile sizes must be positive")
    for family, key in (("slopes", "angle_range_deg"), ("stairs", "height_range"),
                        ("stairs", "depth_range"), ("boxes", "height_range"),
                        ("random_rough", "amplitude_range")):
        low, high = spec[family][key]
        if not 0 <= low <= high or not math.isfinite(high):
            raise ValueError(f"Invalid {family}.{key}")
    if spec["slopes"]["angle_range_deg"][1] >= 90:
        raise ValueError("Slopes must be below 90 degrees")
    if not 0 < spec["random_rough"]["horizontal_scale"] <= spec["random_rough"]["downsampled_scale"]:
        raise ValueError("Invalid height-field spacing")
    if spec["vertical_scale"] <= 0 or spec["boxes"]["width"] <= 0 or spec["boxes"]["gap"] != 0:
        raise ValueError("Invalid box / vertical dimensions")
    if spec["random_rough"]["noise_step"] < spec["vertical_scale"]:
        raise ValueError("Noise step must be at least one vertical quantum")
    return spec


def parameters(spec, difficulty):
    d = float(difficulty)
    if not math.isfinite(d) or not 0 <= d <= 1:
        raise ValueError("difficulty must be finite and in [0,1]")
    def lerp(values):
        return float(values[0] + d * (values[1] - values[0]))
    angle = lerp(spec["slopes"]["angle_range_deg"])
    depth = spec["stairs"]["depth_range"]
    return dict(difficulty=d, slope_angle_deg=angle, slope_coefficient=math.tan(math.radians(angle)),
                stair_height_m=lerp(spec["stairs"]["height_range"]),
                stair_depth_m=float(depth[1] - d * (depth[1] - depth[0])),
                box_height_max_m=lerp(spec["boxes"]["height_range"]),
                rough_amplitude_m=lerp(spec["random_rough"]["amplitude_range"]))


def local_rng(spec, family, difficulty):
    identity = f"{spec['seed']}:{family}:{float(difficulty).hex()}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(identity).digest()[:8], "little"))


def plane(size):
    x, y = size
    return trimesh.Trimesh(vertices=[[0, 0, 0], [x, 0, 0], [x, y, 0], [0, y, 0]],
                           faces=[[0, 1, 2], [0, 2, 3]], process=False)


def box_layout(spec, difficulty):
    """PGMT's touching grid footprints, with user-requested positive heights.

    A centered residual border fills the part not divisible by the grid width.
    The center platform is overlaid at the curriculum's maximum box height.
    """
    cfg, size = spec["boxes"], np.asarray(spec["size"], dtype=float)
    width = cfg["width"]
    count = np.floor(size / width).astype(int)
    if np.any(count < 1):
        raise ValueError("No box fits inside the tile")
    extent = count * width
    axes = [(size[i] - extent[i]) / 2 + width / 2 + np.arange(count[i]) * width for i in (0, 1)]
    xx, yy = np.meshgrid(*axes, indexing="ij")
    centers = np.column_stack((xx.ravel(), yy.ravel()))
    low, high = cfg["height_range"][0], parameters(spec, difficulty)["box_height_max_m"]
    heights = local_rng(spec, "boxes", difficulty).uniform(low, high, len(centers))
    if len(heights):
        heights[0] = high  # include the curriculum ceiling, not only smaller random boxes
    return centers, heights


def native_rough(spec, difficulty):
    """Use the same native function/settings as PGMT/pgmt/terrain.py.

    Preserve native interpolation (including overshoot), 1 cm noise steps,
    15 cm mesh spacing, border padding and steep-face correction. Isolate its
    NumPy RNG so sampling rough terrain cannot change later curriculum rows.
    """
    import isaaclab.terrains as tg
    cfg = spec["random_rough"]
    amplitude = parameters(spec, difficulty)["rough_amplitude_m"]
    native = tg.HfRandomUniformTerrainCfg(size=tuple(spec["size"]),
        noise_range=(-amplitude, amplitude), noise_step=cfg["noise_step"],
        downsampled_scale=cfg["downsampled_scale"], border_width=cfg["border_width"],
        horizontal_scale=cfg["horizontal_scale"], vertical_scale=spec["vertical_scale"],
        slope_threshold=cfg["slope_threshold"])
    rng_state = np.random.get_state()
    try:
        np.random.seed(int(local_rng(spec, "random_rough", difficulty).integers(0, 2**32)))
        meshes, origin = native.function(difficulty, native)
    finally:
        np.random.set_state(rng_state)
    # Native steep-face correction can collapse triangles. Remove only these
    # zero-area faces; do not smooth or clip the reference geometry.
    for mesh in meshes:
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
    return meshes, origin


def generate_tile(difficulty, cfg):
    """Isaac SubTerrainBaseCfg function; returns meshes in [0,size] coordinates."""
    import isaaclab.terrains as tg

    spec, family = cfg.parameters, cfg.family
    values = parameters(spec, difficulty)
    size = tuple(cfg.size)
    if family == "flat":
        return [plane(size)], np.array([size[0] / 2, size[1] / 2, 0.])
    if family in ("slopes", "up_slope"):
        slope = values["slope_coefficient"]
        terrain_class = tg.HfInvertedPyramidSlopedTerrainCfg if family == "up_slope" else tg.HfPyramidSlopedTerrainCfg
        native = terrain_class(size=size, slope_range=(slope, slope),
            platform_width=spec["platform_width"], border_width=spec["slopes"]["border_width"],
            horizontal_scale=spec["horizontal_scale"], vertical_scale=spec["vertical_scale"],
            slope_threshold=None)
        return native.function(difficulty, native)
    if family in ("stairs", "up_stairs"):
        height = values["stair_height_m"]
        terrain_class = tg.MeshInvertedPyramidStairsTerrainCfg if family == "up_stairs" else tg.MeshPyramidStairsTerrainCfg
        native = terrain_class(size=size, step_height_range=(height, height),
            step_width=values["stair_depth_m"], platform_width=spec["platform_width"],
            border_width=spec["stairs"]["border_width"], holes=False)
        return native.function(difficulty, native)
    if family == "boxes":
        centers, heights = box_layout(spec, difficulty)
        meshes = [plane(size)]
        width = spec["boxes"]["width"]
        for (x, y), height in zip(centers, heights):
            box = trimesh.creation.box((width, width, height))
            box.apply_translation((x, y, height / 2))
            meshes.append(box)
        high = values["box_height_max_m"]
        platform = trimesh.creation.box((spec["platform_width"], spec["platform_width"], high))
        platform.apply_translation((size[0] / 2, size[1] / 2, high / 2))
        meshes.append(platform)
        return meshes, np.array([size[0] / 2, size[1] / 2, high])
    if family == "random_rough":
        return native_rough(spec, difficulty)
    raise ValueError(f"Unknown terrain family: {family}")


def make_generator_cfg(spec=None, cache_dir=None):
    """Reusable Isaac Lab config. Call after AppLauncher has started Isaac."""
    from isaaclab.terrains import TerrainGeneratorCfg, SubTerrainBaseCfg
    from isaaclab.utils import configclass
    spec = load_spec() if spec is None else spec

    @configclass
    class PGMTTileCfg(SubTerrainBaseCfg):
        function = generate_tile
        family: str = "flat"
        parameters: dict = {}
        geometry_sha256: str = ""

    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return TerrainGeneratorCfg(seed=spec["seed"], curriculum=True, size=tuple(spec["size"]),
        num_rows=spec["num_rows"], num_cols=spec["num_cols"], border_width=spec["border_width"],
        horizontal_scale=spec["horizontal_scale"], vertical_scale=spec["vertical_scale"],
        difficulty_range=tuple(spec["difficulty_range"]), color_scheme="none",
        use_cache=cache_dir is not None, cache_dir=str(cache_dir or "/tmp/pgmt_terrain_cache"),
        sub_terrains={name: PGMTTileCfg(family=name, parameters=spec, geometry_sha256=source_hash,
                                      proportion=1 / len(FAMILIES)) for name in FAMILIES})


def ray_heights(mesh, xy):
    """Independent CPU ray queries against the triangle mesh, not a height proxy."""
    xy = np.asarray(xy, dtype=float)
    origins = np.column_stack((xy, np.full(len(xy), mesh.bounds[1, 2] + 5)))
    directions = np.tile([0., 0., -1.], (len(xy), 1))
    hits, ray_ids, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=True)
    result = np.full(len(xy), -np.inf)
    np.maximum.at(result, ray_ids, hits[:, 2])
    if not np.isfinite(result).all():
        raise ValueError("Terrain mesh has missing downward-ray hits")
    return result


def validate_tile(mesh, spec, family, difficulty):
    """Check real geometry, including treads, touching box cells and roughness."""
    if not np.isfinite(mesh.vertices).all() or len(mesh.faces) == 0:
        raise ValueError("Empty/nonfinite mesh")
    if mesh.area_faces.min() < 1e-12:
        raise ValueError("Degenerate triangle")
    p = parameters(spec, difficulty)
    sx, sy = spec["size"]
    # Probes stay away from exact triangle/vertical-edge boundaries.
    xy = np.array([[sx/2+.031, sy/2+.019], [sx*.2+.031, sy*.5+.019],
                   [sx*.7+.031, sy*.3+.019], [sx*.85+.031, sy*.65+.019]])
    z = ray_heights(mesh, xy)
    metrics = {}
    if family == "flat":
        np.testing.assert_allclose(mesh.vertices[:, 2], 0, atol=1e-8)
    elif family in ("slopes", "up_slope"):
        sign = -1 if family == "up_slope" else 1
        probe = np.array([[sx*.2, sy/2+.031], [sx*.3, sy/2+.031]])
        hz = ray_heights(mesh, probe)
        actual = (hz[1]-hz[0]) / (probe[1,0]-probe[0,0])
        # Height-field discretization: two quantized endpoints, plus grid rounding.
        if abs(actual-sign*p["slope_coefficient"]) > .015:
            raise ValueError(f"Slope gradient differs: {actual} vs {p['slope_coefficient']}")
        metrics["measured_slope_angle_deg"] = math.degrees(math.atan(abs(actual)))
        metrics["native_generator"] = "HfInvertedPyramidSlopedTerrainCfg" if sign < 0 else "HfPyramidSlopedTerrainCfg"
    elif family in ("stairs", "up_stairs"):
        sign = -1 if family == "up_stairs" else 1
        top = mesh.triangles_center[mesh.face_normals[:, 2] > .99]
        zs = np.unique(np.round(top[:, 2], 7))
        increments = np.diff(zs)
        np.testing.assert_allclose(increments, p["stair_height_m"], atol=2e-6)
        w, h, border = p["stair_depth_m"], p["stair_height_m"], spec["stairs"]["border_width"]
        probe = np.array([[border+(k+.5)*w, sy/2+.031] for k in range(5)])
        np.testing.assert_allclose(ray_heights(mesh, probe), sign*h*np.arange(1,6), atol=2e-6)
        # Probe both sides of risers to validate actual horizontal tread depth.
        edges = border + np.arange(1, 5) * w
        xx = np.column_stack((edges-.001, edges+.001)).ravel()
        hz = ray_heights(mesh, np.column_stack((xx, np.full(len(xx), sy/2+.031))))
        np.testing.assert_allclose(hz.reshape(-1,2)[:,1]-hz.reshape(-1,2)[:,0], sign*h, atol=2e-6)
        metrics.update(measured_step_height_m=float(increments.mean()), verified_tread_depth_m=w)
        metrics["native_generator"] = "MeshInvertedPyramidStairsTerrainCfg" if sign < 0 else "MeshPyramidStairsTerrainCfg"
        if sign < 0:
            xy = np.vstack((xy, np.column_stack((xx,np.full(len(xx),sy/2+.031)))))
    elif family == "boxes":
        centers, heights = box_layout(spec, difficulty)
        expected = heights.copy()
        inside = np.all(np.abs(centers - np.array(spec["size"])/2) < spec["platform_width"]/2, axis=1)
        expected[inside] = p["box_height_max_m"]
        np.testing.assert_allclose(ray_heights(mesh, centers), expected, atol=2e-6)
        w = spec["boxes"]["width"]
        # Independent probes across the first row's seams must hit the adjacent
        # box, rather than drop to a ground-height gap as in v1.
        count_y = int(sy/w)
        seams = centers[:count_y-1] + np.array([0, w/2])
        for offset, expected_z in ((-.001, heights[:count_y-1]), (.001, heights[1:count_y])):
            np.testing.assert_allclose(ray_heights(mesh, seams+[0,offset]), expected_z, atol=2e-6)
        np.testing.assert_allclose(ray_heights(mesh, [[sx/2, sy/2]]), p["box_height_max_m"], atol=2e-6)
        if mesh.bounds[0,2] < -1e-7:
            raise ValueError("Boxes must not introduce negative-height pits")
        metrics.update(box_count=len(heights), box_height_min_m=float(heights.min()),
                       box_height_max_m=float(heights.max()), verified_width_m=w, gap_m=0.,
                       platform_height_m=p["box_height_max_m"], verified_seams=len(seams))
        # Pass seam probes to the PhysX audit as well.
        xy = np.vstack((xy, seams[[0, len(seams)//2]]+[0,-.001],
                        seams[[0, len(seams)//2]]+[0,.001]))
        z = ray_heights(mesh, xy)
    elif family == "random_rough":
        hz = mesh.vertices[:,2]
        quantum = spec["vertical_scale"]
        np.testing.assert_allclose(hz/quantum, np.rint(hz/quantum), atol=2e-6)
        if not hz.min() < 0 < hz.max() or np.std(hz) < .1*p["rough_amplitude_m"]:
            raise ValueError("Rough terrain is flat or lacks signed height variation")
        metrics.update(input_noise_amplitude_m=p["rough_amplitude_m"],
                       measured_amplitude_m=float(np.abs(hz).max()), height_std_m=float(np.std(hz)),
                       native_generator="HfRandomUniformTerrainCfg", interpolation_clipped=False)
    if family in ("up_stairs", "up_slope"):
        # A central basin must rise outwards in ALL four directions. This
        # rejects the obsolete one-direction ramp/staircase implementation.
        center = np.array([sx/2,sy/2])
        rays = []
        for direction, extent in (([1,0],sx/2), ([-1,0],sx/2), ([0,1],sy/2), ([0,-1],sy/2)):
            distances = np.linspace(0.,extent-.037,151)
            line = center+distances[:,None]*np.array(direction)
            hz = ray_heights(mesh,line)
            if np.min(np.diff(hz)) < -2e-6:
                raise ValueError("Inverted pyramid must rise from the center in all four directions")
            if (family == "up_stairs" or difficulty > 0) and hz[-1] <= hz[0]:
                raise ValueError("Inverted pyramid center must be below its rim")
            rays.append(line[[0,50,100,150]])
        xy = np.vstack((xy,*rays))
        metrics.update(inverted=True, outward_travel="up", four_sided_outward_ascent=True,
                       center_surface_height_m=float(ray_heights(mesh,[center])[0]))
    z = ray_heights(mesh,xy)
    return dict(vertices=len(mesh.vertices), triangles=len(mesh.faces), bounds=mesh.bounds.tolist(),
                probes=np.column_stack((xy,z)).tolist(), **metrics)
