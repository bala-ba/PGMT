"""Repeated patches share a curriculum region without sharing a spawn point."""
import copy
import json
from pathlib import Path

import pytest
import torch

from pmt_tasks.pgmt_sampling import load_terrain_compatibility
from pmt_tasks.pgmt_terrain_layout import (
    training_manifest, compatibility_families, patch_bounds, sample_patch_positions, centered_patch_ids,
)
from pmt_tasks.pgmt_terrain_query import TerrainTileMap

ROOT = Path(__file__).resolve().parents[1]
DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def layout():
    source = json.loads((ROOT / "assets/terrain/pgmt_stage2_v6/manifest.json").read_text())
    manifest = training_manifest(source)
    bounds = torch.empty(10, 5, 2, 2)
    for region in manifest["tiles"]:
        bounds[region["level"], region["family_index"]] = torch.tensor(region["bounds"])
    return source, manifest, TerrainTileMap(manifest, bounds)


def test_all_800_patches_resolve_to_their_region_and_local_patch():
    source, manifest, query = layout()
    assert "tiles" not in source and "size" not in source["spec"]
    result = query.locate(torch.tensor([p["center"][:2] for p in manifest["patches"]]))
    assert result["tile_valid"].all()
    for field, key in (("tile_level", "level"), ("family_index", "family_index"), ("patch_id", "patch_id")):
        assert result[field].tolist() == [p[key] for p in manifest["patches"]]
    assert result["family_group_index"].tolist() == [
        {"flat": 0, "slope": 1, "stairs": 2, "boxes": 3, "rough": 4}[p["family"]] for p in manifest["patches"]]


def test_crossing_patch_and_region_seams_does_not_change_birth_assignment():
    _, _, query = layout()
    birth_level, birth_family = torch.tensor([0]), torch.tensor([0])
    points = torch.tensor([[-117., -57.], [-111., -57.], [-93., -57.], [120., 0.], [0., 60.]])
    result = query.locate(points)
    assert result["patch_id"].tolist() == [0, 4, 0, -1, -1]
    assert result["tile_level"].tolist() == [0, 0, 1, -1, -1]
    assert birth_level.item() == birth_family.item() == 0
    edge = torch.tensor(-114.)
    x = torch.stack((torch.nextafter(edge, torch.tensor(-torch.inf)), edge,
                     torch.nextafter(edge, torch.tensor(torch.inf))))
    assert query.locate(torch.stack((x, torch.full_like(x, -57.)), -1))["patch_id"].tolist() == [0, 4, 4]


def test_random_spawns_cover_all_patches_and_interiors_with_partial_reset_order():
    _, manifest, query = layout()
    bounds = patch_bounds(manifest, device="cpu")
    torch.manual_seed(672)
    levels, families = torch.tensor([9, 0, 4, 1]).repeat(2500), torch.tensor([4, 0, 1, 2]).repeat(2500)
    before = (levels.clone(), families.clone())
    xy, selected, ids = sample_patch_positions(bounds, levels, families, margin=.25)
    actual = query.locate(xy)
    assert torch.equal(actual["tile_level"], levels) and torch.equal(actual["family_index"], families)
    assert torch.equal(actual["patch_id"], ids)
    assert ((xy >= selected[:, 0] + .25) & (xy <= selected[:, 1] - .25)).all()
    for family in families.unique():
        assert set(ids[families == family].tolist()) == set(range(16))
    offsets = (xy - selected[:, 0]) / (selected[:, 1] - selected[:, 0])
    assert (offsets.std(0) > .24).all()  # Actually spread over each patch, not center jitter.
    assert torch.equal(levels, before[0]) and torch.equal(families, before[1])


@pytest.mark.parametrize("device", DEVICES)
def test_center_two_by_two_spawns_cover_all_regions_without_shrinking_the_map(device):
    _, manifest, query = layout()
    bounds = patch_bounds(manifest, device=device)
    allowed = centered_patch_ids(manifest["spec"]["patch_grid"], (2, 2), device=device)
    assert allowed.tolist() == [5, 6, 9, 10]
    torch.manual_seed(1015)
    # Every level/family, interleaved as in a partial reset with mixed births.
    cells = torch.arange(50, device=device).repeat(400)
    levels, families = cells // 5, cells % 5
    xy, selected, ids = sample_patch_positions(bounds, levels, families, margin=.25, allowed_patch_ids=allowed)
    centers = (bounds[levels, families, 0, 0] + bounds[levels, families, 15, 1]) / 2
    assert ((xy - centers).abs() <= 5.75).all()
    assert ((xy >= selected[:, 0] + .25) & (xy <= selected[:, 1] - .25)).all()
    actual = query.locate(xy.cpu())
    assert torch.equal(actual["tile_level"], levels.cpu())
    assert torch.equal(actual["family_index"], families.cpu())
    assert torch.equal(actual["patch_id"], ids.cpu())
    assert torch.isin(ids, allowed).all()
    for cell in range(50):
        assert set(ids[cells == cell].tolist()) == {5, 6, 9, 10}
    counts = torch.bincount(ids, minlength=16)[allowed].float()
    assert (counts / len(ids) - .25).abs().max() < .015
    # Spawn selection does not redefine the legal walking/querying map.
    outside_spawn = bounds[0, 0, 0].mean(0)
    assert query.locate(outside_spawn.cpu()[None])["tile_valid"].all()
    # Empty partial resets retain their expected shapes.
    xy, selected, ids = sample_patch_positions(bounds, levels[:0], families[:0], margin=.25,
                                               allowed_patch_ids=allowed)
    assert xy.shape == (0, 2) and selected.shape == (0, 2, 2) and ids.shape == (0,)


def test_center_grid_can_restore_full_patch_selection():
    assert centered_patch_ids((4, 4), (4, 4), device="cpu").tolist() == list(range(16))


@pytest.mark.parametrize("version", ["v6", "v8"])
def test_center_xy_uses_the_selected_patch_after_level_and_family_changes(version):
    source = json.loads((ROOT / f"assets/terrain/pgmt_stage2_{version}/manifest.json").read_text())
    manifest = training_manifest(source)
    bounds = patch_bounds(manifest, device="cpu")
    lookup = {(p["level"], p["family_index"], p["patch_id"]): p["center"][:2] for p in source["patches"]}
    # Noncontiguous/reordered births exercise indexing used by partial resets.
    levels = torch.tensor([9, 0, 5, 2, 9])
    families = torch.tensor([4, 1, 0, 3, 2])
    allowed = centered_patch_ids(source["spec"]["patch_grid"], (1, 1) if version == "v8" else (2, 2), device="cpu")
    for _ in range(3):
        xy, selected, ids = sample_patch_positions(bounds, levels, families, margin=.25,
                                                   allowed_patch_ids=allowed, xy_mode="center")
        expected = torch.tensor([lookup[r, c, p] for r, c, p in zip(levels.tolist(), families.tolist(), ids.tolist())])
        torch.testing.assert_close(xy, expected, rtol=0, atol=0)
        assert torch.isin(ids, allowed).all()
        assert ((xy > selected[:, 0]) & (xy < selected[:, 1])).all()
        levels, families = levels.roll(1), families.roll(-1)
    xy, _, ids = sample_patch_positions(bounds, levels[:0], families[:0], margin=.25, xy_mode="center")
    assert xy.shape == (0, 2) and ids.shape == (0,)
    with pytest.raises(ValueError, match="XY mode"):
        sample_patch_positions(bounds, levels, families, margin=.25, xy_mode="typo")


@pytest.mark.parametrize("grid", [(3, 3), (0, 2), (6, 2), (2.0, 2), (True, 2), (2,)])
def test_center_grid_rejects_noncentral_or_invalid_selection(grid):
    with pytest.raises(ValueError, match="grid"):
        centered_patch_ids((4, 4), grid, device="cpu")


@pytest.mark.parametrize("kind", ["missing", "duplicate", "bounds"])
def test_malformed_patch_layout_fails(kind):
    _, manifest, _ = layout()
    manifest = copy.deepcopy(manifest)
    if kind == "missing":
        manifest["patches"].pop()
    elif kind == "duplicate":
        manifest["patches"].append(manifest["patches"][0])
    else:
        manifest["patches"][0]["world_bounds"][0][0] += .1
    with pytest.raises(ValueError, match="patch"):
        patch_bounds(manifest, device="cpu")


def test_v6_exclusions_include_legacy_groups_and_inverted_variants(tmp_path):
    _, manifest, _ = layout()
    path = tmp_path / "compatibility.csv"
    path.write_text("motion_name,dismatched terrain\na,up_stairs\nb,stairs\nc,up_slope\nd,slopes\ne,random_rough\nf,\n")
    allowed = load_terrain_compatibility(path, [f"{name}.npz" for name in "abcdef"],
                                         families=compatibility_families(manifest))
    assert allowed.tolist() == [[True] * 6, [False, False, True, True, True, True],
                                [True, True, False, False, True, True], [True] * 6,
                                [True, True, True, True, False, True]]
    legacy = {"spec": {"families": ["flat", "stairs", "up_stairs"]}}
    assert training_manifest(legacy) is legacy
    assert compatibility_families(legacy) == ("flat", "stairs", "up_stairs")
