"""Versioned v6 patch geometry and audits; does not alter v5 or training reset."""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import trimesh
import yaml

from pmt_tasks.pgmt_terrain import local_rng, parameters, ray_heights, validate_tile

FAMILIES = ("flat", "stairs", "slope", "boxes", "rough")
NATIVE_FAMILIES = dict(flat="flat", stairs="up_stairs", slope="up_slope", boxes="boxes", rough="random_rough")
NATIVE_INTERFACES = dict(flat="MeshPlaneTerrainCfg", stairs="MeshInvertedPyramidStairsTerrainCfg",
                         slope="HfInvertedPyramidSlopedTerrainCfg", boxes="MeshRandomGridTerrainCfg",
                         rough="HfRandomUniformTerrainCfg")


def load_patch_spec(path):
    spec = yaml.safe_load(Path(path).read_text())
    if tuple(spec["families"]) != FAMILIES or spec["num_rows"] != 10 or spec["num_cols"] != 5:
        raise ValueError("v6 requires five named families and ten levels")
    if spec["patch_grid"] != [4, 4] or spec["patch_size"] != [6., 6.]:
        raise ValueError("This preview requires 4 x 4 patches of 6 x 6 m")
    np.testing.assert_allclose(np.array(spec["patch_grid"]) * spec["patch_size"], spec["region_size"])
    if any(spec[name]["shape"] != "inverted_pyramid" for name in ("stairs", "slope")):
        raise ValueError("v6 stairs and slope must both be inverted")
    if any(spec[name]["class_name"] != cls for name, cls in NATIVE_INTERFACES.items()):
        raise ValueError("Each v6 family must declare its supported native Isaac Lab interface")
    width = spec["boxes"]["grid_width"]
    size = spec["patch_size"][0]
    np.testing.assert_allclose(size - int(size / width) * width, spec["boxes"]["connection_width"], atol=1e-8)
    return spec


def seed_for(spec, *parts):
    key = ":".join(map(str, (spec["seed"], *parts))).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def patch_spec(spec, level, family, patch_id):
    """Resolve patch settings; retain aliases only for shared parameter/audit helpers."""
    result = copy.deepcopy(spec)
    result.update(size=list(spec["patch_size"]), slopes=spec["slope"], random_rough=spec["rough"],
                  seed=seed_for(spec, "texture", level, family, patch_id))
    result["boxes"]["height_range"] = list(spec["boxes"]["grid_height_range"])
    if family in ("slope", "boxes"):
        result["platform_width"] = spec[family].get("platform_width", spec["platform_width"])
    return result


def native_config(spec, family, difficulty):
    """The sole geometry definition: construct public native terrain configs."""
    import isaaclab.terrains as tg

    cls = getattr(tg, spec[family]["class_name"])
    size = tuple(spec["size"])
    p = parameters(spec, difficulty)
    if family == "flat":
        return cls(size=size)
    if family == "stairs":
        return cls(size=size, step_height_range=tuple(spec["stairs"]["height_range"]),
                   step_width=p["stair_depth_m"], platform_width=spec["platform_width"],
                   border_width=spec["stairs"]["border_width"], holes=False)
    if family == "slope":
        # Angle is linear in difficulty; native slope_range uses tan(angle).
        slope = p["slope_coefficient"]
        return cls(size=size, slope_range=(slope, slope), platform_width=spec["platform_width"],
                   border_width=spec["slope"]["border_width"], horizontal_scale=spec["horizontal_scale"],
                   vertical_scale=spec["vertical_scale"], slope_threshold=None)
    if family == "boxes":
        return cls(size=size, grid_width=spec["boxes"]["grid_width"],
                   grid_height_range=tuple(spec["boxes"]["grid_height_range"]),
                   platform_width=spec["platform_width"], holes=spec["boxes"]["holes"])
    if family == "rough":
        cfg, amplitude = spec["rough"], p["rough_amplitude_m"]
        return cls(size=size, noise_range=(-amplitude, amplitude), noise_step=cfg["noise_step"],
                   downsampled_scale=cfg["downsampled_scale"], border_width=cfg["border_width"],
                   horizontal_scale=cfg["horizontal_scale"], vertical_scale=spec["vertical_scale"],
                   slope_threshold=cfg["slope_threshold"])
    raise ValueError(f"Unknown terrain family: {family}")


def call_native(cfg, spec, family, difficulty):
    """Call the native function with isolated RNG state, never change its shape rules."""
    if family == "boxes":
        import torch

        # Native random_grid_terrain selects CUDA when available. Fork all CUDA
        # generators because torch.manual_seed seeds them as well as the CPU.
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(spec["seed"])
            return cfg.function(difficulty, cfg)
    if family == "rough":
        state = np.random.get_state()
        try:
            # Preserve the already-reviewed native rough realization.
            np.random.seed(int(local_rng(spec, "random_rough", difficulty).integers(0, 2**32)))
            return cfg.function(difficulty, cfg)
        finally:
            np.random.set_state(state)
    return cfg.function(difficulty, cfg)


def audit_patch(mesh, origin, spec, family, difficulty):
    """Probe actual triangles, with stair checks valid for small stair counts."""
    native = NATIVE_FAMILIES[family]
    if family in ("flat", "rough"):
        audit = validate_tile(mesh, spec, native, difficulty)
        xy = np.asarray(audit["probes"])[:, :2]
    elif family == "boxes":
        if not np.isfinite(mesh.vertices).all() or mesh.area_faces.min() < 1e-12:
            raise ValueError("Invalid native grid mesh")
        size = np.array(spec["size"])
        width = spec["boxes"]["grid_width"]
        count = int(size[0] / width)
        border = (size[0] - count * width) / 2
        axis = border + (np.arange(count) + .5) * width
        xx, yy = np.meshgrid(axis, axis, indexing="ij")
        centers = np.column_stack((xx.ravel(), yy.ravel()))
        heights = ray_heights(mesh, centers)
        amplitude = parameters(spec, difficulty)["box_height_max_m"]
        outside = np.any(np.abs(centers - size / 2) >= spec["platform_width"] / 2, axis=1)
        if not heights[outside].min() < 0 < heights[outside].max():
            raise ValueError("Native boxes must have both positive and negative surface heights")
        if np.abs(heights).max() > amplitude + 2e-6:
            raise ValueError("Native grid top heights exceed [-H, H]")
        np.testing.assert_allclose(heights[~outside], amplitude, atol=2e-6)
        probes = [centers]
        if spec["platform_width"] == 0:
            # Every elevated/depressed horizontal face must be a grid-cell
            # triangle, not a larger central platform covering several cells.
            tops = (mesh.face_normals[:, 2] > .99) & (np.abs(mesh.triangles_center[:, 2]) > 1e-7)
            if mesh.area_faces[tops].max() > width * width / 2 + 2e-6:
                raise ValueError("Unexpected wide surface: the boxes center platform is still present")
        np.testing.assert_allclose(2 * border, spec["boxes"]["connection_width"], atol=1e-8)
        # Verify both the zero-height connection band and the transition to
        # the first grid cell on all four sides, not only the patch seam.
        distances = np.array([.001, border / 2, border - .001, border + .001])
        for points, cell_height in (
            (np.column_stack((distances, np.full(4, axis[0]))), heights[0]),
            (np.column_stack((size[0] - distances, np.full(4, axis[0]))), heights[(count - 1) * count]),
            (np.column_stack((np.full(4, axis[0]), distances)), heights[0]),
            (np.column_stack((np.full(4, axis[0]), size[1] - distances)), heights[count - 1])):
            np.testing.assert_allclose(ray_heights(mesh, points), [0., 0., 0., cell_height], atol=2e-6)
            probes.append(points)
        # Check touching cells in the first row and column, away from the
        # central platform. Include negative cells to detect a covering plane.
        for indices, direction in ((np.arange(count), np.array([0., 1.])),
                                   (np.arange(count) * count, np.array([1., 0.]))):
            seams = centers[indices[:-1]] + direction * width / 2
            for offset, expected in ((-.001, heights[indices[:-1]]), (.001, heights[indices[1:]])):
                points = seams + direction * offset
                np.testing.assert_allclose(ray_heights(mesh, points), expected, atol=2e-6)
                probes.append(points)
        np.testing.assert_allclose(ray_heights(mesh, origin[None, :2]), origin[2], atol=2e-6)
        xy = np.vstack(probes)
        audit = dict(vertices=len(mesh.vertices), triangles=len(mesh.faces), bounds=mesh.bounds.tolist(),
                     grid_count=count * count, grid_width_m=width, border_width_per_side_m=border,
                     input_height_amplitude_m=amplitude, measured_top_height_min_m=float(heights[outside].min()),
                     measured_top_height_max_m=float(heights[outside].max()),
                     positive_and_negative_tops_verified=True, platform_enabled=spec["platform_width"] > 0,
                     platform_width_m=spec["platform_width"], origin_height_m=float(origin[2]),
                     connection_width_m=2 * border, connection_band_verified=True,
                     verified_cell_seams=2 * (count - 1))
    elif family == "slope":
        if not np.isfinite(mesh.vertices).all() or mesh.area_faces.min() < 1e-12:
            raise ValueError("Invalid slope mesh")
        sx, sy = spec["size"]
        # Outer ramp probes also remain valid with the former 2 m platform.
        xy = np.array([[.5, sy / 2 + .031], [1.1, sy / 2 + .031]])
        z = ray_heights(mesh, xy)
        actual = -(z[1] - z[0]) / (xy[1, 0] - xy[0, 0])
        expected = parameters(spec, difficulty)["slope_coefficient"]
        tolerance = spec["vertical_scale"] / .6 + .02 * expected
        if abs(actual - expected) > tolerance:
            raise ValueError(f"Outer ramp gradient differs: {actual} vs {expected}")
        probes = [xy]
        for direction in ([1, 0], [-1, 0], [0, 1], [0, -1]):
            line = np.array([sx / 2, sy / 2]) + np.linspace(0, sx / 2 - .037, 101)[:, None] * direction
            heights = ray_heights(mesh, line)
            if np.diff(heights).min() < -2e-6 or (difficulty > 0 and heights[-1] <= heights[0]):
                raise ValueError("Slope basin must rise outwards in all four directions")
            probes.append(line[::10])
        xy = np.vstack(probes)
        audit = dict(vertices=len(mesh.vertices), triangles=len(mesh.faces), bounds=mesh.bounds.tolist(),
                     native_generator="HfInvertedPyramidSlopedTerrainCfg", inverted=True,
                     four_sided_outward_ascent=True, measured_slope_angle_deg=math.degrees(math.atan(actual)),
                     slope_coefficient_tolerance=tolerance, center_surface_height_m=float(origin[2]))
        bottom_faces = np.all(np.abs(mesh.triangles[:, :, 2] - origin[2]) < 2e-6, axis=1)
        audit.update(platform_width_m=spec["platform_width"],
                     measured_bottom_area_m2=float(mesh.area_faces[bottom_faces].sum()))
    else:
        if not np.isfinite(mesh.vertices).all() or mesh.area_faces.min() < 1e-12:
            raise ValueError("Invalid stair mesh")
        p = parameters(spec, difficulty)
        sx, sy = spec["size"]
        surfaces = np.unique(np.round(mesh.triangles_center[mesh.face_normals[:, 2] > .99, 2], 7))
        np.testing.assert_allclose(np.diff(surfaces), p["stair_height_m"], atol=2e-6)
        n = len(surfaces) - 2  # native ring treads, excluding border and center
        w, h, border = p["stair_depth_m"], p["stair_height_m"], spec["stairs"]["border_width"]
        tread_xy = np.column_stack((border + (np.arange(n) + .5) * w, np.full(n, sy / 2 + .031)))
        np.testing.assert_allclose(ray_heights(mesh, tread_xy), -h * np.arange(1, n + 1), atol=2e-6)
        edges = border + np.arange(1, n + 1) * w
        xx = np.column_stack((edges - .001, edges + .001)).ravel()
        edge_xy = np.column_stack((xx, np.full(len(xx), sy / 2 + .031)))
        np.testing.assert_allclose(np.diff(ray_heights(mesh, edge_xy).reshape(-1, 2), axis=1), -h, atol=2e-6)
        probes = [tread_xy, edge_xy]
        for direction in ([1, 0], [-1, 0], [0, 1], [0, -1]):
            line = np.array([sx / 2, sy / 2]) + np.linspace(0, sx / 2 - .037, 101)[:, None] * direction
            z = ray_heights(mesh, line)
            if np.diff(z).min() < -2e-6 or z[-1] <= z[0]:
                raise ValueError("Stair basin must rise outwards in all four directions")
            probes.append(line[::25])
        xy = np.vstack(probes)
        audit = dict(vertices=len(mesh.vertices), triangles=len(mesh.faces), bounds=mesh.bounds.tolist(),
                     native_generator="MeshInvertedPyramidStairsTerrainCfg", inverted=True,
                     four_sided_outward_ascent=True, measured_step_height_m=float(np.diff(surfaces).mean()),
                     verified_tread_depth_m=w, ring_treads=n, center_surface_height_m=float(origin[2]))
    # Native borders must meet at zero with no full plane covering either basin.
    sx, sy = spec["size"]
    a, b = np.linspace(.031, sx - .031, 9), np.linspace(.031, sy - .031, 9)
    edges = np.vstack((np.column_stack((a, np.full(9, .0001))),
                       np.column_stack((a, np.full(9, sy - .0001))),
                       np.column_stack((np.full(9, .0001), b)),
                       np.column_stack((np.full(9, sx - .0001), b))))
    np.testing.assert_allclose(ray_heights(mesh, edges), 0., atol=2e-6)
    center_probe = origin[None, :2] if family == "boxes" else np.array(spec["size"])[None] / 2
    xy = np.vstack((xy, edges, center_probe))
    if family in ("stairs", "slope"):
        center_z = ray_heights(mesh, np.array(spec["size"])[None] / 2)[0]
        np.testing.assert_allclose(center_z, origin[2], atol=2e-6)
        if difficulty > 0 and origin[2] >= 0:
            raise ValueError("Inverted origin must retain its negative height")
    audit.update(probes=np.column_stack((xy, ray_heights(mesh, xy))).tolist(), zero_border_verified=True)
    return audit


def make_patch(spec, level, family, patch_id, difficulty):
    local_spec = patch_spec(spec, level, family, patch_id)
    cfg = native_config(local_spec, family, difficulty)
    meshes, origin = call_native(cfg, local_spec, family, difficulty)
    removed_faces = 0
    if family == "rough" or (family == "boxes" and cfg.platform_width == 0):
        # Native steep-face correction can collapse triangles. Remove only
        # zero-area faces; preserve every nondegenerate native surface.
        # With platform_width=0, the native boxes platform is also zero-area.
        for part in meshes:
            before = len(part.faces)
            part.update_faces(part.nondegenerate_faces())
            part.remove_unreferenced_vertices()
            removed_faces += before - len(part.faces)
        meshes = [part for part in meshes if len(part.faces)]
    mesh = trimesh.util.concatenate(meshes)
    native_origin_z = float(origin[2])
    if family == "slope":
        # Native height-field origins use the maximum of a fixed 2 m square.
        # A smaller platform no longer covers that square; anchor the preview
        # at the actual center surface instead of above the basin floor.
        origin = origin.copy()
        origin[2] = ray_heights(mesh, np.array(local_spec["size"])[None] / 2)[0]
    elif family == "boxes" and cfg.platform_width == 0:
        # Without the platform, native origin Z=+H is no longer a surface.
        # Query just inside the central cell rather than the exact four-cell
        # junction, where float precision makes ray hits ambiguous.
        origin = np.r_[np.array(local_spec["size"]) / 2 + [.031, .019], 0.]
        origin[2] = ray_heights(mesh, origin[None, :2])[0]
    audit = audit_patch(mesh, origin, local_spec, family, difficulty)
    audit.update(native_generator=type(cfg).__name__,
                 native_function=f"{cfg.function.__module__}.{cfg.function.__name__}",
                 removed_degenerate_faces=removed_faces)
    if family in ("slope", "boxes"):
        audit["native_origin_z_m"] = native_origin_z
    rng = np.random.default_rng(seed_for(spec, "preview_xy_yaw", level, family, patch_id))
    preview = spec["spawn_preview"]
    xy = rng.uniform(preview["boundary_margin"], np.array(spec["patch_size"]) - preview["boundary_margin"],
                     size=(preview["samples_per_patch"], 2))
    yaw = rng.uniform(*preview["yaw_range"], size=len(xy))
    samples = np.column_stack((xy, ray_heights(mesh, xy), yaw))
    params = parameters(local_spec, difficulty)
    params["platform_width_m"] = float(local_spec["platform_width"])
    params["box_height_amplitude_m"] = params["box_height_max_m"]
    if family == "boxes":
        params["connection_width_m"] = float(local_spec["boxes"]["connection_width"])
    return mesh, origin, audit, samples, params, local_spec["seed"]
