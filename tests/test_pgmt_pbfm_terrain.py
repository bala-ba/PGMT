"""PBFM terrain ranges, noise, weighted rollouts and legacy compatibility."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pmt_tasks.builder import build_task_config
from pmt_tasks.pgmt_curriculum import TerrainCurriculum
from pmt_tasks.pgmt_pbfm_terrain import FAMILIES, load_patch_spec, parameters, perlin_noise
from pmt_tasks.pgmt_sampling import load_terrain_compatibility
from pmt_tasks.pgmt_terrain_distribution import allocate_terrain_types, terrain_proportions
from pmt_tasks.pgmt_terrain_layout import PATCH_SCHEMA, compatibility_families, training_manifest, patch_bounds
from pmt_tasks.pgmt_terrain_query import TerrainTileMap

ROOT = Path(__file__).resolve().parents[1]
SPEC = load_patch_spec(ROOT / "configs/terrain/pgmt_stage2_v7.yaml")
PROPORTIONS = dict(zip(FAMILIES, [.2, .1, .2, .2, .1, .1, .1]))


@pytest.mark.parametrize("task", ["pgmt_stage2", "pgmt_stage2_terrain"])
def test_stage2_tasks_select_pbfm_asset_and_weighted_mix(task):
    config = build_task_config(task)
    assert config.terrain.version == "pgmt_stage2_v8"
    assert Path(config.terrain_asset).name == "pgmt_stage2_v8"
    assert tuple(config.terrain.families) == FAMILIES
    assert dict(config.terrain.family_proportions) == PROPORTIONS
    assert config.motion.fixed_start_frame == -1
    assert config.motion.sampler == "emd"
    assert config.motion.placement_xy_mode == "center"


def test_ranges_follow_pbfm_with_slope_in_rise_over_run():
    lo, hi = parameters(SPEC, 0), parameters(SPEC, 1)
    for key, expected in {"rough_height_amplitude_m": (.02, .06),
                          "stair_up_height_m": (.10, .20), "stair_down_height_m": (.10, .20),
                          "platform_height_amplitude_m": (.03, .10), "slope_coefficient": (0, .2),
                          "obstacle_height_m": (.05, .15)}.items():
        assert (lo[key], hi[key]) == pytest.approx(expected)
    assert hi["slope_angle_deg"] == pytest.approx(math.degrees(math.atan(.2)))
    levels = (np.arange(10) + np.random.default_rng(1731).random(10)) / 10
    assert np.all(np.diff([parameters(SPEC, d)["stair_up_height_m"] for d in levels]) > 0)


@pytest.mark.parametrize("num_envs", [0, 1, 5, 7, 10, 100, 4096, 8192])
def test_environment_allocation_preserves_total_and_rounds_each_weight_within_one(num_envs):
    types = allocate_terrain_types(num_envs, FAMILIES, PROPORTIONS, device="cpu")
    counts = torch.bincount(types, minlength=7)
    assert len(types) == num_envs and types.dtype == torch.long
    assert torch.all((counts - torch.tensor(list(PROPORTIONS.values()), dtype=torch.float64) * num_envs).abs() < 1)
    if num_envs == 100:
        assert counts.tolist() == [20, 10, 20, 20, 10, 10, 10]
    # Dictionary insertion order must not silently change terrain assignments.
    assert torch.equal(types, allocate_terrain_types(num_envs, FAMILIES, dict(reversed(list(PROPORTIONS.items()))), device="cpu"))


def test_zero_weight_never_receives_an_environment():
    assert allocate_terrain_types(11, ["a", "b", "c"], dict(a=.5, b=0., c=.5), device="cpu").tolist() == [0]*6 + [2]*5


@pytest.mark.parametrize("change", [{"flat": .3}, {"rough": -.1}, {"flat": float("nan")}, {"unknown": 0.}])
def test_invalid_mixes_fail_before_allocation(change):
    with pytest.raises(ValueError, match="proportions"):
        terrain_proportions(FAMILIES, dict(PROPORTIONS, **change))


def test_curriculum_changes_difficulty_without_changing_training_proportions():
    types = allocate_terrain_types(100, FAMILIES, PROPORTIONS, device="cpu")
    levels = torch.full_like(types, 5)
    origins = torch.zeros(10, 7, 3)
    terrain = SimpleNamespace(terrain_types=types.clone(), terrain_levels=levels, terrain_origins=origins,
                              env_origins=origins[levels, types].clone())
    state = TerrainCurriculum(terrain, {})
    ids = torch.arange(100)
    state.begin(ids, 0, ids, torch.zeros_like(ids), torch.full_like(ids, 10))
    no = torch.zeros_like(ids, dtype=torch.bool)
    state.settle(ids, step=9, motion_ids=ids, frame_ids=torch.full_like(ids, 9),
                 failure=ids % 2 == 0, finished=~no, map_exit=no, timeout=no, other=no)
    assert set(terrain.terrain_levels.tolist()) == {4, 6}
    assert torch.equal(terrain.terrain_types, types)


def test_perlin_is_seeded_bounded_and_spatially_correlated():
    def noise(seed):
        return perlin_noise((101, 101), .05, 1.2, rng=np.random.default_rng(seed),
                            octaves=3, persistence=.5, lacunarity=2.)
    a, b = noise(1731), noise(1732)
    assert np.array_equal(a, noise(1731)) and not np.array_equal(a, b)
    assert np.isfinite(a).all() and np.abs(a).max() == pytest.approx(1.)
    assert a.min() < 0 < a.max()
    # White/uniform noise cannot pass this neighboring-height correlation check.
    assert np.corrcoef(a[:-1].ravel(), a[1:].ravel())[0, 1] > .85


def test_seven_families_keep_stair_direction_exclusions_and_semantic_groups(tmp_path):
    source = dict(schema=PATCH_SCHEMA, spec=SPEC, regions=[dict(family=f, center=[0, 0, 0]) for f in FAMILIES])
    manifest = training_manifest(source)
    assert [t["family_group_index"] for t in manifest["tiles"]] == [0, 4, 2, 2, 3, 1, 3]
    path = tmp_path / "compatibility.csv"
    path.write_text("motion_name,dismatched terrain\na,up_stairs\nb,stairs\nc,boxes\nd,random_rough\ne,up_slope\n")
    allowed = load_terrain_compatibility(path, [f"{name}.npz" for name in "abcde"], families=compatibility_families(manifest))
    assert allowed[:, 0].tolist() == [True, True, True, False, True, True, True]
    assert allowed[:, 1].tolist() == [True, True, False, False, True, True, True]
    assert allowed[:, 2].tolist() == [True, True, True, True, False, True, False]
    assert allowed[:, 3].tolist() == [True, False, True, True, True, True, True]
    assert allowed[:, 4].tolist() == [True, True, True, True, True, False, True]


def test_generated_v7_asset_has_complete_patch_coverage_and_correct_queries():
    path = ROOT / "assets/terrain/pgmt_stage2_v7/manifest.json"
    if not path.is_file():
        pytest.skip("Build the v7 terrain asset to check generated geometry metadata")
    source = json.loads(path.read_text())
    assert source["spec"] == SPEC
    manifest = training_manifest(source)
    assert manifest["region_count"] == 70 and manifest["patch_count"] == 1120
    patches = patch_bounds(manifest, device="cpu")
    bounds = torch.empty(10, 7, 2, 2)
    for region in manifest["tiles"]:
        bounds[region["level"], region["family_index"]] = torch.tensor(region["bounds"])
    query = TerrainTileMap(manifest, bounds)
    actual = query.locate(patches.mean(-2))
    assert actual["tile_valid"].all()
    assert torch.equal(actual["tile_level"], torch.arange(10)[:, None, None].expand(10, 7, 16))
    assert torch.equal(actual["family_index"], torch.arange(7)[None, :, None].expand(10, 7, 16))
    assert torch.equal(actual["patch_id"], torch.arange(16)[None, None, :].expand(10, 7, 16))
