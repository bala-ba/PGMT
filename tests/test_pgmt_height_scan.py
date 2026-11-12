"""Signed scan heights, validity and reset freshness without an Isaac runtime."""
from types import SimpleNamespace

import pytest
import torch

from pmt_tasks.pgmt_height_scan import height_scan_from_hits
from pmt_tasks.pgmt_runtime import PGMTState


def hits():
    y, x = torch.meshgrid(torch.linspace(-1., 1., 21), torch.linspace(-1., 1., 21), indexing="ij")
    return torch.stack((x, y, 2 * x + 3 * y - 12.), -1).reshape(1, 441, 3)


def test_signed_heights_axis_order_and_world_translation_invariance():
    points = hits()
    original = points.clone()
    mask = torch.ones(1, 441, dtype=torch.bool)
    scan = height_scan_from_hits(points, torch.tensor([-11.]), mask)
    assert scan["height"].shape == (1, 21, 21)
    assert scan["valid"].all()
    assert scan["height_w"].lt(0).all()  # No clipping negative absolute elevations.
    assert scan["height"][0, 10, 10] == -1.  # Torso zero, not launch offset or world zero.
    assert scan["height"][0, 10, -1] == 1.  # Ground above torso stays positive.
    torch.testing.assert_close(scan["height"][0, :, 1:] - scan["height"][0, :, :-1],
                               torch.full((21, 20), .2))
    torch.testing.assert_close(scan["height"][0, 1:] - scan["height"][0, :-1],
                               torch.full((20, 21), .3))
    moved = points + torch.tensor([70., -50., 27.])
    shifted = height_scan_from_hits(moved, torch.tensor([16.]), mask)
    torch.testing.assert_close(shifted["height"], scan["height"])
    torch.testing.assert_close(points, original)


def test_map_border_and_missing_rays_are_invalid_even_with_finite_fill():
    points = hits()
    mask = torch.ones(1, 441, dtype=torch.bool)
    mask[0, 0] = False  # Finite mesh hit outside the valid training map.
    points[0, 1, 2] = float("inf")
    points[0, 2, 0] = float("nan")
    points[0, 3, 2] = -float("inf")
    scan = height_scan_from_hits(points, torch.tensor([-11.]), mask)
    assert not scan["valid"].flatten()[:4].any()
    assert scan["valid"].flatten()[4:].all()
    assert scan["height"].isfinite().all()
    assert scan["height"].flatten()[:4].eq(0).all()
    assert scan["height_w"].flatten()[:4].isnan().all()
    assert points[0, 1, 2].isinf()  # Do not mutate clean sensor hits.


def test_invalid_torso_never_produces_finite_valid_observations():
    scan = height_scan_from_hits(hits(), torch.tensor([float("nan")]), torch.ones(1, 441).bool())
    assert not scan["valid"].any()
    assert scan["height"].eq(0).all()


@pytest.mark.parametrize("bad", ["ray_count", "torso", "mask"])
def test_misconfigured_scan_shapes_fail(bad):
    points, torso, mask = hits(), torch.zeros(1), torch.ones(1, 441).bool()
    if bad == "ray_count":
        points = points[:, :187]
    elif bad == "torso":
        torso = torso[:, None]
    else:
        mask = mask[:, :187]
    with pytest.raises(ValueError):
        height_scan_from_hits(points, torso, mask)


def test_runtime_reuses_scan_but_invalidates_after_reset_and_physics_step():
    calls = []

    def contains(xy):
        calls.append(1)
        return xy[..., 0] < .5

    state = PGMTState.__new__(PGMTState)
    state.anchor = 0
    state._obs_version, state._terrain_scan_key = 0, None
    state.robot = SimpleNamespace(data=SimpleNamespace(body_pos_w=torch.tensor([[[0., 0., -11.]]])))
    sensor = SimpleNamespace(data=SimpleNamespace(ray_hits_w=hits()))
    state.env = SimpleNamespace(common_step_counter=0, scene=SimpleNamespace(
        terrain=SimpleNamespace(tile_map=object(), contains=contains), sensors={"height_scanner": sensor}))
    first = state.terrain_scan()
    assert state.terrain_scan() is first and len(calls) == 1
    sensor.data.ray_hits_w[..., 2] -= 3
    state.invalidate_observations()  # The same invalidation used by command reset.
    second = state.terrain_scan()
    assert len(calls) == 2
    torch.testing.assert_close(second["height"][second["valid"]], first["height"][first["valid"]] - 3)
    state.env.common_step_counter += 1
    assert state.terrain_scan() is not second and len(calls) == 3
    state.env.scene.sensors.clear()
    state.invalidate_observations()
    with pytest.raises(RuntimeError, match="height scanner"):
        state.terrain_scan()
