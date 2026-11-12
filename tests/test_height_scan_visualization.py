"""Protect observation provenance and sensor-to-display spatial correspondence."""
import json
from types import SimpleNamespace

import numpy as np
import torch

from pmt_tasks.state_visualization import height_scan_snapshot
from scripts.height_scan_visualization import add_scan_geoms, spatial_grid


def test_snapshot_preserves_policy_noise_and_dropout_without_mutation():
    hits = torch.tensor([[[1.0, 2.0, 0.2], [2.0, 2.0, 0.4], [float("inf"), 0.0, 0.0]]])
    # A dropped policy ray may still have a perfectly valid physical ray hit.
    vision = torch.tensor([[0.63, -3.0, -3.0, 1.0, 0.0, 0.0]])
    before = vision.clone()
    sensor = SimpleNamespace(data=SimpleNamespace(ray_hits_w=hits))
    snapshot = height_scan_snapshot(sensor, {"vision": vision})
    assert snapshot["hit_valid"] == [True, True, False]
    assert snapshot["policy_valid"] == [True, False, False]
    np.testing.assert_array_equal(snapshot["policy_height"], vision[0, :3].numpy())
    torch.testing.assert_close(vision, before)
    json.dumps(snapshot, allow_nan=False)
    assert "policy_height" not in height_scan_snapshot(sensor, {})


def test_spatial_axes_respect_both_ray_orders():
    # Physical values x + 10*y; x = [-1, 0, 1], y = [-1, 1].
    config = {"shape_xy": [3, 2], "ordering": "xy"}
    xy = [-11, -10, -9, 9, 10, 11]
    expected = [[11, -9], [10, -10], [9, -11]]
    np.testing.assert_array_equal(spatial_grid(xy, config), expected)
    config["ordering"] = "yx"
    np.testing.assert_array_equal(spatial_grid([-11, 9, -10, 10, -9, 11], config), expected)


def test_world_overlay_uses_recorded_hits_and_marks_policy_dropout():
    import mujoco

    config = {"shape_xy": [2, 2], "ordering": "xy"}
    scan = {"hits_w": [[0, 0, .1], [1, 0, .2], [0, 1, .3], [0, 0, 0]],
            "hit_valid": [True, True, True, False], "policy_valid": [True, False, True, False]}
    scene = mujoco.MjvScene(mujoco.MjModel.from_xml_string("<mujoco/>"), maxgeom=20)
    add_scan_geoms(scene, scan, config)
    assert scene.ngeom == 5  # two valid edges, three measured points
    for index, point in enumerate(scan["hits_w"][:3]):
        np.testing.assert_allclose(scene.geoms[index + 2].pos, np.array(point) + [0, 0, .012], atol=1e-6)
    np.testing.assert_allclose(scene.geoms[3].rgba[:3], np.array([234, 86, 193]) / 255, atol=1e-6)
