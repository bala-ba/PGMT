"""Analytic sensor geometry and map lifetime tests (no simulator required)."""
import numpy as np
import pytest

from pmt_tasks.depth_perception import CameraIntrinsics, LocalElevationMap, depth_points


def overhead(x=0., y=0., z=1.):
    pose = np.diag([1., -1., -1., 1.])
    pose[:3, 3] = [x, y, z]
    return pose


K = CameraIntrinsics.from_fovy(160, 120, 100.)


def test_metric_depth_projection_with_translated_camera():
    points = depth_points(np.full((120, 160), .8), K, overhead(2, -1))
    np.testing.assert_allclose(points[:, 2], .2, atol=1e-12)
    np.testing.assert_allclose(points[:, :2].mean(axis=0), [2, -1], atol=1e-12)
    # A slant-range interpretation would incorrectly curve this plane.
    assert np.ptp(points[:, 0]) > 2


def test_step_heights_yaw_order_and_policy_relative_height():
    depth = np.ones((120, 160))
    depth[:, 80:] = .8
    mapper = LocalElevationMap()
    mapper.integrate(depth, K, overhead(), 0., [0, 0])
    scan = mapper.sample([0, 0, 1.2], np.pi / 2, .1)
    valid = np.asarray(scan["policy_valid"])
    hits = np.asarray(scan["hits_w"])
    away_from_edge = valid & (np.abs(hits[:, 0]) > .1)
    assert away_from_edge.sum() > 90
    np.testing.assert_allclose(hits[away_from_edge, 2], np.where(hits[away_from_edge, 0] > 0, .2, 0), atol=1e-6)
    np.testing.assert_allclose(np.asarray(scan["policy_height"])[valid], 1.2 - hits[valid, 2])
    indices = np.flatnonzero(valid)
    # PMT xy flattening: x index changes first, then y; rotate yaw only.
    np.testing.assert_allclose(hits[valid, 0], -(-.5 + (indices // 17) * .1), atol=1e-12)
    np.testing.assert_allclose(hits[valid, 1], -.8 + (indices % 17) * .1, atol=1e-12)


def test_unknown_occlusion_and_expiration_never_filled_with_ground():
    mapper = LocalElevationMap(max_age_s=1.)
    scan = mapper.sample([0, 0, 1], 0, 0)
    assert not any(scan["policy_valid"])
    assert np.all(np.asarray(scan["policy_height"]) == -3)
    depth = np.ones((120, 160))
    mask = np.ones(depth.shape, bool)
    mask[40:80, 55:105] = False
    mapper.integrate(depth, K, overhead(), 0., [0, 0], valid_mask=mask)
    assert not mapper.sample([0, 0, 1], 0, .1)["policy_valid"][93]
    assert any(mapper.sample([0, 0, 1], 0, .1)["policy_valid"])
    # Empty new frame preserves valid history until its measured timestamp expires.
    mapper.integrate(np.zeros_like(depth), K, overhead(), .5, [0, 0])
    assert any(mapper.sample([0, 0, 1], 0, .9)["policy_valid"])
    assert not any(mapper.sample([0, 0, 1], 0, 1.01)["policy_valid"])


@pytest.mark.parametrize("shift", [(1., -.7), (-.8, .6), (10., 10.)])
def test_rolling_grid_preserves_world_location_without_wrapping(shift):
    mapper = LocalElevationMap(size_m=4.)
    mapper.integrate(np.full((120, 160), .8), K, overhead(), 0., [0, 0])
    before = mapper.sample([0, 0, 1], 0, 0)
    mapper.integrate(np.zeros((120, 160)), K, overhead(*shift), .1, shift)
    after = mapper.sample([0, 0, 1], 0, .1)
    if max(np.abs(shift)) > 4:
        assert not any(after["policy_valid"])
    else:
        assert before["policy_valid"] == after["policy_valid"]
        np.testing.assert_allclose(before["policy_height"], after["policy_height"])
    mapper.reset()
    assert not any(mapper.sample([0, 0, 1], 0, 0)["policy_valid"])


def test_vertical_wall_and_invalid_depth_do_not_become_floor():
    # Camera faces +X, optical right=-Y, down=-Z. Constant depth is a wall.
    pose = np.eye(4)
    pose[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    assert len(depth_points(np.ones((120, 160)), K, pose)) == 0
    depth = np.zeros((120, 160))
    depth[1:5] = np.nan
    depth[5:10] = np.inf
    depth[10:15] = -1
    depth[15:20] = 10
    assert len(depth_points(depth, K, overhead())) == 0


def test_time_and_calibration_errors_rejected():
    mapper = LocalElevationMap()
    mapper.integrate(np.ones((120, 160)), K, overhead(), 1., [0, 0])
    with pytest.raises(ValueError, match="timestamps"):
        mapper.integrate(np.ones((120, 160)), K, overhead(), .9, [0, 0])
    with pytest.raises(ValueError, match="query time"):
        mapper.sample([0, 0, 1], 0, .9)
    bad_pose = overhead()
    bad_pose[0, 0] = 2
    with pytest.raises(ValueError, match="rigid"):
        depth_points(np.ones((120, 160)), K, bad_pose)
    with pytest.raises(ValueError, match="shape"):
        depth_points(np.ones((10, 10)), K, overhead())
