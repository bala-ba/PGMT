"""Optical geometry/occlusion acceptance; run with MUJOCO_GL=egl."""
import os

import numpy as np
import pytest


def test_unknown_and_sparse_segmentation_ids_are_invalid_not_other_objects():
    from types import SimpleNamespace
    from pmt_tasks.backends.mujoco_depth import decode_id_colors

    # Scene IDs are unrelated to model geom indices and may contain gaps.
    scene = SimpleNamespace(ngeom=2, geoms=[SimpleNamespace(segid=0, objid=42, objtype=5),
                                          SimpleNamespace(segid=2563, objid=100, objtype=5)])
    labels = np.array([[0, 1, 2564], [2624, 256, 0xFFFFFF]], dtype=np.uint32)
    rgb = np.stack([labels & 255, (labels >> 8) & 255, (labels >> 16) & 255], axis=-1).astype(np.uint8)
    decoded, invalid = decode_id_colors(rgb, scene)
    np.testing.assert_array_equal(decoded[0], [[-1, -1], [42, 5], [100, 5]])
    np.testing.assert_array_equal(decoded[1], [[-1, -1]] * 3)
    assert invalid == 3


@pytest.mark.skipif(os.environ.get("MUJOCO_GL") not in ("egl", "osmesa"), reason="Requires explicit headless OpenGL backend")
def test_rendered_depth_step_and_self_occlusion_without_terrain_queries(tmp_path, monkeypatch):
    import mujoco
    from pmt_tasks.backends.mujoco_depth import MujocoDepthPerception

    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <visual><global offwidth="160" offheight="120"/></visual>
      <worldbody>
        <geom type="plane" size="5 5 .01"/>
        <geom type="box" pos=".5 0 .1" size=".2 .4 .1"/>
        <body name="torso_link" pos="0 0 1">
          <camera name="pmt_depth" xyaxes="1 0 0 0 1 0" fovy="100"/>
          <geom type="sphere" pos="0 0 -.35" size=".08"/>
        </body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    def forbidden(*args, **kwargs):
        raise AssertionError("Depth perception must not query terrain heights")
    monkeypatch.setattr(mujoco, "mj_ray", forbidden)
    config = dict(width=160, height=120, fovy_deg=100., pitch_deg=90., hz=25., max_age_s=2.)
    original_samples = model.vis.quality.offsamples
    with MujocoDepthPerception(model, config, tmp_path, record=True) as sensor:
        assert model.vis.quality.offsamples == original_samples  # other renderers retain their settings
        scan = sensor.scan(data)
        valid = np.asarray(scan["policy_valid"])
        hits = np.asarray(scan["hits_w"])
        assert .4 < valid.mean() < .95
        assert not valid[93]  # robot sphere occludes the ground, even after self filtering
        measured = hits[valid]
        inside_step = (measured[:, 0] > .34) & (measured[:, 0] < .66) & (np.abs(measured[:, 1]) < .34)
        flat = (measured[:, 0] < .24) | (measured[:, 0] > .76) | (np.abs(measured[:, 1]) > .46)
        assert inside_step.sum() > 5 and flat.sum() > 20
        np.testing.assert_allclose(measured[inside_step, 2], .2, atol=2e-4)
        np.testing.assert_allclose(measured[flat, 2], 0, atol=2e-4)
        data.time = .02
        sensor.scan(data)
        assert sensor.capture_count == 1  # camera 25Hz, control 50Hz
        data.time = .04
        sensor.scan(data)
        assert sensor.capture_count == 2
        assert sensor.invalid_segmentation_pixels == 0
    with np.load(tmp_path / "depth_frames.npz") as recorded:
        assert recorded["depth_m"].shape == (2, 120, 160)
        assert recorded["depth_m"][0, 60, 80] == 0  # no see-through self mask
        assert np.isfinite(recorded["depth_m"]).all()
