"""Terrain exclusions must survive adaptation, uniform mixing and reset selection."""
import csv
from pathlib import Path

import pytest
import torch

from pmt_tasks.pgmt_sampling import load_terrain_compatibility, masked_motion_probabilities, sample_motion_starts
from pmt_tasks.pgmt_terrain import FAMILIES


def table(tmp_path, rows):
    path = tmp_path / "compatibility.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["motion_name", "dismatched terrain"])
        writer.writerows(rows)
    return path


def test_csv_matches_loader_order_and_expands_semantic_groups(tmp_path):
    path = table(tmp_path, [("dance", "stairs; slopes"), ("walk.npz", ""), ("jump", "up_stairs")])
    allowed = load_terrain_compatibility(path, ["/data/jump.npz", "/data/dance.npz", "/data/walk.npz"])
    assert allowed[FAMILIES.index("flat")].tolist() == [True, True, True]
    assert allowed[FAMILIES.index("stairs")].tolist() == [True, False, True]
    assert allowed[FAMILIES.index("up_stairs")].tolist() == [False, False, True]
    assert not allowed[FAMILIES.index("up_slope"), 1]
    # The full table can be used for a single-motion inspection.
    assert load_terrain_compatibility(path, ["walk.npz"]).all()


@pytest.mark.parametrize("rows,files,match", [
    ([("a", "stairz")], ["a.npz"], "Unknown terrain"),
    ([("a", ""), ("a", "stairs")], ["a.npz"], "duplicate"),
    ([("a", "")], ["b.npz"], "missing"),
    ([("a", "")], ["first/a.npz", "second/a.npz"], "unique"),
])
def test_csv_typos_and_ambiguous_mapping_fail(tmp_path, rows, files, match):
    with pytest.raises(ValueError, match=match):
        load_terrain_compatibility(table(tmp_path, rows), files)


def test_mask_applies_before_both_normalizations_and_uniform_mix():
    failure = torch.tensor([.9, .1, 1.])
    mask = torch.tensor([[True, True, False], [False, True, True]])
    probabilities = masked_motion_probabilities(failure, mask, beta=2., uniform_ratio=.2)
    torch.testing.assert_close(probabilities[0], torch.tensor([.8 * .81 / .82 + .1, .8 * .01 / .82 + .1, 0.]))
    assert (probabilities[~mask] == 0).all()
    assert (probabilities[mask] >= .1).all()
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(2))
    # Changing a forbidden motion's difficulty cannot change this terrain's draw.
    failure[2] = 1e-6
    changed = masked_motion_probabilities(failure, mask, beta=2., uniform_ratio=.2)
    torch.testing.assert_close(changed[0], probabilities[0])


def test_motion_weights_not_biased_by_valid_frame_count_and_frames_are_uniform():
    torch.manual_seed(581)
    valid = torch.tensor([[True, False, False, False], [False, True, True, True], [False] * 4])
    compatible = torch.ones(len(FAMILIES), 3, dtype=torch.bool)
    terrain_ids = torch.zeros(20000, dtype=torch.long)
    motions, frames = sample_motion_starts(torch.tensor([.5, .5, 1.]), compatible, valid, terrain_ids,
                                           beta=2., uniform_ratio=.1)
    assert not (motions == 2).any()
    assert valid[motions, frames].all()
    assert .48 < (motions == 0).float().mean() < .52
    second = frames[motions == 1]
    for frame in (1, 2, 3):
        assert .30 < (second == frame).float().mean() < .36


def test_mixed_terrain_partial_reset_order_fixed_frames_and_empty_pools():
    compatible = torch.zeros(len(FAMILIES), 3, dtype=torch.bool)
    compatible[0, 0] = True
    compatible[2, 1] = True
    valid = torch.tensor([[True, False, True], [True, True, True], [True, True, True]])
    # Noncontiguous/reordered environments must retain their own terrain pool.
    terrain_ids = torch.tensor([2, 0, 2, 0])
    motions, frames = sample_motion_starts(torch.ones(3), compatible, valid, terrain_ids,
                                           beta=2., uniform_ratio=.1, fixed_start_frame=2)
    assert motions.tolist() == [1, 0, 1, 0]
    assert frames.tolist() == [2, 2, 2, 2]
    with pytest.raises(ValueError, match="flat"):
        sample_motion_starts(torch.ones(3), compatible, valid, terrain_ids,
                             beta=2., uniform_ratio=.1, fixed_start_frame=1)
    with pytest.raises(ValueError, match="outside"):
        sample_motion_starts(torch.ones(3), compatible, valid, terrain_ids,
                             beta=2., uniform_ratio=.1, fixed_start_frame=3)
    with pytest.raises(ValueError, match="up_stairs"):
        sample_motion_starts(torch.ones(3), compatible, valid, torch.tensor([5]), beta=2., uniform_ratio=.1)


def test_failure_change_increases_motion_frequency_without_allowing_forbidden_motion():
    torch.manual_seed(9)
    compatible = torch.tensor([[True, True, False]] * len(FAMILIES))
    valid = torch.ones(3, 5, dtype=torch.bool)
    terrains = torch.zeros(10000, dtype=torch.long)
    motions, _ = sample_motion_starts(torch.tensor([.9, .1, 1.]), compatible, valid, terrains,
                                      beta=2., uniform_ratio=.1)
    assert not (motions == 2).any()
    assert (motions == 0).float().mean() > .92


def test_fixed_frame_zero_allows_all_nonempty_clips_and_keeps_terrain_exclusions():
    torch.manual_seed(61)
    # With the support gate disabled the command supplies real frame ranges,
    # including clips whose offline feet never provide support. Padded frames
    # still remain ineligible.
    lengths = torch.tensor([1, 3, 8])
    valid = torch.arange(8)[None] < lengths[:, None]
    compatible = torch.ones(len(FAMILIES), 3, dtype=torch.bool)
    compatible[FAMILIES.index("stairs"), 1] = False
    terrains = torch.tensor([0, FAMILIES.index("stairs")]).repeat(1000)
    motions, frames = sample_motion_starts(torch.ones(3), compatible, valid, terrains,
                                          beta=2., uniform_ratio=.1, fixed_start_frame=0)
    assert frames.eq(0).all()
    assert set(motions[terrains == 0].tolist()) == {0, 1, 2}
    assert not (motions[terrains == FAMILIES.index("stairs")] == 1).any()
