"""Source collision parity and PMT contracts for the native P-BFM asset."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh

from pmt_tasks.pgmt_contact_state import terrain_contact_filters
from pmt_tasks.pgmt_mujoco_mesh import heightfield_mesh
from pmt_tasks.pgmt_terrain import ray_heights
from pmt_tasks.pgmt_terrain_layout import training_manifest, patch_bounds, centered_patch_ids
from pmt_tasks.pgmt_terrain_query import TerrainTileMap

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "assets/terrain/pgmt_stage2_v8"


@pytest.fixture(scope="module")
def manifest():
    if not (ASSET / "manifest.json").exists():
        pytest.skip("Build the native P-BFM v8 asset first")
    return training_manifest(json.loads((ASSET / "manifest.json").read_text()))


def test_heightfield_diagonal_scale_and_closed_base():
    # A noncoplanar cell distinguishes the MuJoCo diagonal from the opposite
    # diagonal (whose center would be zero). Rectangular extents catch swaps.
    mesh = heightfield_mesh(np.array([[0., 0.], [0., 1.]]), [2., 3., 4., .25])
    np.testing.assert_allclose(mesh.bounds, [[-2., -3., -.25], [2., 3., 4.]])
    np.testing.assert_allclose(ray_heights(mesh, [[0., 0.], [1., 0.], [-1., 0.]]), [2., 2., 1.])
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0


def test_native_layout_difficulty_and_auxiliary_contact_coverage(manifest):
    assert manifest["region_count"] == manifest["patch_count"] == 70
    assert manifest["spec"]["seed"] == 0
    assert manifest["spec"]["patch_size"] == [5., 5.]
    assert manifest["spec"]["spawn_patch_grid"] == [1, 1]
    assert centered_patch_ids((1, 1), (1, 1), device="cpu").tolist() == [0]
    assert manifest["map_bounds"] == [[-25., -17.5], [25., 17.5]]
    assert manifest["walkable_bounds"] == [[-37., -29.5], [37., 29.5]]
    assert manifest["guard_count"] == 84
    assert manifest["auxiliary_surface_count"] == 92
    paths = terrain_contact_filters(ASSET, "/World/ground")
    assert len(paths) == len(set(paths)) == 162
    assert sum("/Guard/" in path for path in paths) == 84
    assert sum("/Border/" in path for path in paths) == 4
    assert sum("/Walls/" in path for path in paths) == 4
    assert not any("*" in path for path in paths)
    patches = patch_bounds(manifest, device="cpu")
    assert patches.shape == (10, 7, 1, 2, 2)
    bounds = patches[:, :, 0]
    query = TerrainTileMap(manifest, bounds)
    located = query.locate(bounds.mean(-2))
    assert located["tile_valid"].all() and located["patch_id"].eq(0).all()
    assert torch.equal(located["tile_level"], torch.arange(10)[:, None].expand(10, 7))
    for patch in manifest["patches"]:
        assert patch["level"] / 10 <= patch["difficulty"] < (patch["level"] + 1) / 10
    # MJLab samples independently per family while consuming its shared RNG.
    assert len({p["difficulty"] for p in manifest["patches"] if p["level"] == 5}) == 7
    source = manifest["source"]["preset"]["sub_terrains"]
    boxes = source["boxes"]
    assert boxes["num_obstacles"] == 50 and boxes["obstacle_height_mode"] == "choice"
    assert boxes["obstacle_height_range"] == [.05, .15]
    assert boxes["platform_width"] == 1.2 and boxes["border_width"] == .5
    assert source["perlin_rough"]["octaves"] == 2
    assert source["perlin_rough"]["scale"] == 20.


def test_saved_ply_rays_match_native_mujoco_references(manifest):
    records = manifest["patches"] + manifest["auxiliary_surfaces"]
    meshes = []
    for record in records:
        path = ASSET / record["mesh_file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["mesh_sha256"]
        mesh = trimesh.load_mesh(path, process=False)
        mesh.apply_translation(record["center"])
        meshes.append(mesh)
    combined = trimesh.util.concatenate(meshes)
    assert hashlib.sha256((ASSET / "terrain.usdc").read_bytes()).hexdigest() == manifest["asset_sha256"]
    probes = np.array([xyz for r in records for xyz in r["world_probes"]] + manifest["seam_probes"])
    for chunk in np.array_split(probes, 48):
        np.testing.assert_allclose(ray_heights(combined, chunk[:, :2]), chunk[:, 2], rtol=0, atol=2e-5)
    boxes = next(p for p in manifest["patches"] if p["family"] == "boxes" and p["level"] == 9)
    heights = np.array(boxes["world_probes"])[:, 2]
    assert heights.min() < -.05 and heights.max() > .05
    np.testing.assert_allclose(ray_heights(combined, [boxes["center"][:2]]), 0., atol=1e-6)
