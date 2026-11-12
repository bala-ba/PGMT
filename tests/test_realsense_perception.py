from pathlib import Path

import numpy as np
import pytest

from pmt_tasks.depth_perception import CameraIntrinsics
from pmt_tasks.realsense_perception import RobotSelfFilter, decode_depth_image, subsample_depth


@pytest.mark.parametrize("encoding,dtype,scale", [("16UC1", ">u2", 1000), ("32FC1", ">f4", 1)])
def test_real_depth_units_endianness_and_row_padding(encoding, dtype, scale):
    values = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
    rows = (values * scale).astype(dtype)
    data = b"".join(row.tobytes() + b"\x00\x00\x00\x00" for row in rows)
    decoded = decode_depth_image(data, 3, 2, rows.strides[0] + 4, encoding, True)
    np.testing.assert_allclose(decoded, values)
    with pytest.raises(ValueError, match="stride"):
        decode_depth_image(data[:-1], 3, 2, rows.strides[0] + 4, encoding, True)


def test_stride_preserves_pixel_rays():
    k = CameraIntrinsics(640, 480, 380., 380., 319.5, 239.5)
    depth, small = subsample_depth(np.ones((480, 640)), k, 4)
    assert depth.shape == (120, 160)
    np.testing.assert_allclose((10 - small.cx) / small.fx, (40 - k.cx) / k.fx)


def test_urdf_self_mask_uses_measured_points_and_link_pose(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<robot name="r"><link name="foot"><collision><geometry><sphere radius=".15"/></geometry></collision></link></robot>')
    filt = RobotSelfFilter(urdf, padding=.01)
    k = CameraIntrinsics(11, 11, 10., 10., 5., 5.)
    camera = np.eye(4)
    link = np.eye(4)
    link[2, 3] = 1.
    mask = filt.valid_mask(np.ones((11, 11)), k, camera, {"foot": link})
    assert not mask[5, 5] and mask[0, 0]
    # Points on the floor behind the link are not projected/invented by the mask.
    assert filt.valid_mask(np.full((11, 11), 2.), k, camera, {"foot": link}).all()
    with pytest.raises(KeyError):
        filt.valid_mask(np.ones((11, 11)), k, camera, {})


def test_bundled_g1_collision_geometry_supported():
    path = Path(__file__).resolve().parents[1] / "pmt_tasks/assets/unitree_description/urdf/g1/main.urdf"
    filt = RobotSelfFilter(path)
    assert len(filt.collisions) == 29
    assert "pelvis" in filt.link_names and "left_ankle_roll_link" in filt.link_names
