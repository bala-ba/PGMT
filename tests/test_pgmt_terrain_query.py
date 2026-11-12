"""Spatial tile metadata must not overwrite birth-assigned curriculum levels."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pmt_tasks.pgmt_runtime import PGMTState
from pmt_tasks.pgmt_terrain_query import TerrainTileMap


def fixture():
    manifest = json.loads((Path(__file__).resolve().parents[1]
                          / "assets/terrain/pgmt_stage2_v5/manifest.json").read_text())
    bounds = torch.empty(10, 7, 2, 2)
    for tile in manifest["tiles"]:
        # Manifest audit bounds are in [0,size] local coordinates.
        bounds[tile["level"], tile["family_index"]] = (torch.tensor(tile["geometry"]["bounds"])[:, :2]
            + torch.tensor(tile["center"][:2]) - torch.tensor(manifest["spec"]["size"]) / 2)
    return manifest, bounds


def test_all_70_tiles_use_world_position_and_preserve_semantic_groups():
    manifest, bounds = fixture()
    manifest["tiles"].reverse()  # Input ordering must not become tile identity.
    layout = TerrainTileMap(manifest, bounds)
    points = torch.tensor([tile["center"][:2] for tile in manifest["tiles"]])
    result = layout.locate(points)
    assert result["tile_valid"].all()
    for key, source in (("tile_level", "level"), ("family_index", "family_index"),
                        ("family_group_index", "family_group_index"), ("difficulty", "difficulty")):
        torch.testing.assert_close(result[key], torch.tensor([t[source] for t in manifest["tiles"]]))


def test_shared_edges_and_adjacent_float_values_have_deterministic_ownership():
    layout = TerrainTileMap(*fixture())
    for axis, edges, other in ((0, layout.x_edges, -72.), (1, layout.y_edges, -108.)):
        points = torch.full((len(edges), 3, 2), other)
        points[..., axis] = torch.stack((torch.nextafter(edges, torch.full_like(edges, -torch.inf)), edges,
                                         torch.nextafter(edges, torch.full_like(edges, torch.inf))), -1)
        result = layout.locate(points)
        key = "tile_level" if axis == 0 else "family_index"
        assert result[key][1:-1].tolist() == [[i - 1, i, i] for i in range(1, len(edges) - 1)]
        assert result["tile_valid"][0].tolist() == [False, True, True]
        assert result["tile_valid"][-1].tolist() == [True, False, False]


def test_border_nonfinite_points_and_outside_queries_never_clamp_to_edge_tile():
    layout = TerrainTileMap(*fixture())
    xy = torch.tensor([[-120.1, 0], [120.1, 0], [0, -84.1], [0, 84.1],
                       [float("nan"), 0], [0, float("inf")], [-float("inf"), 0]])
    result = layout.locate(xy)
    assert not result["tile_valid"].any()
    assert result["tile_level"].eq(-1).all()
    assert result["family_index"].eq(-1).all()
    assert result["family_group_index"].eq(-1).all()
    assert result["difficulty"].isnan().all()
    assert layout.locate(torch.empty(0, 2))["tile_level"].shape == (0,)


def test_valid_map_checks_tile_mesh_coverage_not_only_global_rectangle():
    manifest, bounds = fixture()
    bounds[0, 0, 1, 0] -= .25
    layout = TerrainTileMap(manifest, bounds)
    points = torch.tensor([[-96.3, -72.], [-96.1, -72.], [-96., -72.]])
    assert layout.contains(points).tolist() == [True, False, True]


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "center"])
def test_malformed_layout_fails_before_runtime(mutation):
    manifest, bounds = fixture()
    manifest = copy.deepcopy(manifest)
    if mutation == "duplicate":
        manifest["tiles"].append(manifest["tiles"][0])
    elif mutation == "missing":
        manifest["tiles"].pop()
    else:
        manifest["tiles"][-1]["center"][0] += 1
    with pytest.raises(ValueError, match="manifest|grid"):
        TerrainTileMap(manifest, bounds)


def test_foot_lookup_crosses_tiles_without_changing_birth_assignment():
    layout = TerrainTileMap(*fixture())
    terrain = SimpleNamespace(tile_map=layout, query=layout.locate,
                              terrain_levels=torch.tensor([0, 1]), terrain_types=torch.tensor([0, 6]))
    state = PGMTState.__new__(PGMTState)
    state.env = SimpleNamespace(scene=SimpleNamespace(terrain=terrain))
    state.foot_body_indices = torch.tensor([2, 0])
    pos = torch.tensor([[[-108., -72., 1.], [0, 0, 1.], [-108., -72., 1.]],
                        [[-84., 72., 1.], [0, 0, 1.], [-84., 72., 1.]]])
    state.robot = SimpleNamespace(data=SimpleNamespace(body_pos_w=pos))
    before = state.robot_foot_terrain()
    # Left and right feet may simultaneously stand on different tiles.
    pos[0, 2, :2] = torch.tensor([-84., -48.])
    crossed = state.robot_foot_terrain(env_ids=[0])
    assert crossed["tile_level"].tolist() == [[1, 0]]
    assert crossed["family_index"].tolist() == [[1, 0]]
    assert crossed["curriculum_level"].tolist() == [0]
    assert crossed["assigned_family_index"].tolist() == [0]
    assert terrain.terrain_levels.tolist() == [0, 1]
    assert terrain.terrain_types.tolist() == [0, 6]
    assert before["tile_level"].tolist() == [[0, 0], [1, 1]]
    crossed["curriculum_level"].fill_(9)
    assert terrain.terrain_levels.tolist() == [0, 1]
