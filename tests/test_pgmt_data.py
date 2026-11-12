"""The dataset boundary must preserve values, names, frame rate, and G1 FK."""
from pathlib import Path

import mujoco
import numpy as np
import pytest

from scripts.prepare_pgmt_lafan import BFS_BODIES, BFS_JOINTS, BODY_KEYS, convert_arrays, prepare, validate_fk


ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "TCRS/assets/g1/g1_29dof_rev_1_0.xml"


@pytest.fixture
def source():
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    qids = model.jnt_qposadr[[model.joint(name).id for name in BFS_JOINTS]]
    bodies = [model.body(name).id for name in BFS_BODIES]
    frames = 3
    joints = np.random.default_rng(11).uniform(-.1, .1, (frames, 29)).astype(np.float32)
    positions, quaternions = [], []
    for joint in joints:
        data.qpos[:7] = [0, 0, .8, 1, 0, 0, 0]
        data.qpos[qids] = joint
        mujoco.mj_kinematics(model, data)
        positions.append(data.xpos[bodies].copy())
        quaternions.append(data.xquat[bodies].copy())
    return dict(joint_pos=joints, joint_vel=np.zeros_like(joints),
                body_pos=np.asarray(positions, dtype=np.float32), body_quat=np.asarray(quaternions, dtype=np.float32),
                body_lin_vel=np.zeros((frames, 30, 3), dtype=np.float32),
                body_ang_vel=np.zeros((frames, 30, 3), dtype=np.float32),
                joint_names=np.asarray(BFS_JOINTS), body_names=np.asarray(BFS_BODIES),
                fps=np.float32(50), contacts=np.array([[0, 1], [1, 1], [1, 0]], dtype=np.float32))


def test_conversion_uses_names_and_preserves_values(source):
    expected = convert_arrays(source)
    joints = np.arange(29)[::-1]
    bodies = np.roll(np.arange(30), 7)
    source["joint_names"] = source["joint_names"][joints]
    source["body_names"] = source["body_names"][bodies]
    for key in ("joint_pos", "joint_vel"):
        source[key] = source[key][:, joints]
    for key in BODY_KEYS:
        source[key] = source[key][:, bodies]
    actual = convert_arrays(source)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    assert actual["fps"].shape == (1,)


@pytest.mark.parametrize("kind", ["rate", "joint_names", "body_names", "nan", "quaternion", "contact"])
def test_invalid_input_fails_before_writing(source, kind):
    if kind == "rate":
        source["fps"] = np.float32(30)
    elif kind in ("joint_names", "body_names"):
        source[kind][0] = source[kind][1]
    elif kind == "nan":
        source["joint_vel"][0, 0] = np.nan
    elif kind == "quaternion":
        source["body_quat"][0, 0] *= .5
    else:
        source["contacts"][0, 0] = .5
    with pytest.raises(ValueError):
        convert_arrays(source)


def test_fk_detects_wrong_link_positions(source):
    arrays = convert_arrays(source)
    model = mujoco.MjModel.from_xml_path(str(XML))
    assert validate_fk(arrays, model)["position_max_abs_m"] < 2e-5
    arrays["body_pos_w"][1, 5, 0] += .01
    with pytest.raises(ValueError, match="FK mismatch"):
        validate_fk(arrays, model)


def test_prepare_manifest_and_refuses_implicit_overwrite(source, tmp_path):
    inputs = tmp_path / "source"
    inputs.mkdir()
    np.savez(inputs / "motion.npz", **source)
    output = tmp_path / "converted"
    result = prepare(inputs, output, robot_xml=XML, expected_clips=1)
    assert result["total_clips"] == 1 and result["total_frames"] == 3
    assert len(result["clips"][0]["output_sha256"]) == 64
    assert (output / "manifest.json").is_file()
    with np.load(output / "motion.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["contacts"], source["contacts"])
    with pytest.raises(FileExistsError):
        prepare(inputs, output, robot_xml=XML, expected_clips=1)
    with pytest.raises(ValueError, match="Expected 40"):
        prepare(inputs, tmp_path / "another", robot_xml=XML)
