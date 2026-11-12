"""Known geometric offsets, pooled statistics, and real MuJoCo contacts."""
import mujoco
import numpy as np
import pytest

from pmt_tasks.tracking_metrics import (
    aggregate_clips, contact_summary, quaternion_angle, summarize_errors, tracking_errors,
)
from scripts.mujoco_eval_pgmt_full import FootContacts


def test_translation_velocity_units_and_pelvis_alignment():
    ref = np.zeros((2, 30, 3))
    actual = ref + [.1, 0, 0]
    # Non-identity orientation must not rotate the world velocity error.
    quat = np.tile([np.sqrt(.5), 0, 0, np.sqrt(.5)], (2, 1))
    joint = np.zeros((2, 29))
    joint[1] = 2
    errors = tracking_errors(actual, ref, np.tile([0, 1, 0], (2, 1)),
                             np.zeros((2, 3)), quat, -quat, joint,
                             np.zeros_like(joint), np.arange(14))
    result = summarize_errors(errors)
    assert result["mpkpe_mm"] == pytest.approx(100)
    assert result["mpjpe_g_mm"] == pytest.approx(100)
    assert result["root_pos_mm"] == pytest.approx(100)
    assert result["root_vel_mm_s"] == pytest.approx(1000)
    assert result["mpjpe_l_mm"] == pytest.approx(0)
    assert result["mpjpe_all29_l_mm"] == pytest.approx(0)
    assert result["root_orientation_rad"] == pytest.approx(0, abs=1e-7)
    # Mean per-frame RMSE would be 1; pooled RMSE is sqrt(2).
    assert result["joint_rmse_rad"] == pytest.approx(np.sqrt(2))


def test_orientation_and_contacts():
    assert quaternion_angle([np.cos(.5), 0, 0, np.sin(.5)], [2, 0, 0, 0]) == pytest.approx(1)
    result = contact_summary([[1, 0], [1, 0], [0, 0]], [[1, 1], [0, 0], [0, 0]])
    assert {k: result[k] for k in ("tp", "fp", "fn", "tn")} == dict(tp=1, fp=1, fn=1, tn=3)
    assert result["f1"] == pytest.approx(.5)
    assert result["accuracy"] == pytest.approx(4/6)
    assert result["both_feet_accuracy"] == pytest.approx(1/3)
    assert contact_summary([[0, 0]], [[0, 0]])["f1"] == 0


def test_clip_aggregation_pools_squared_errors_and_confusion_counts():
    clips = [dict(evaluated_frames=1, metrics={"joint_rmse_rad": 0., "root_pos_mm": 0.},
                  contact=contact_summary([[1, 1]], [[1, 1]])),
             dict(evaluated_frames=3, metrics={"joint_rmse_rad": 2., "root_pos_mm": 100.},
                  contact=contact_summary([[0, 0]] * 3, [[1, 1]] * 3))]
    result = aggregate_clips(clips)
    assert result["frame_weighted"]["joint_rmse_rad"] == pytest.approx(np.sqrt(3))
    assert result["frame_weighted"]["root_pos_mm"] == pytest.approx(75)
    assert result["equal_clip_mean"]["joint_rmse_rad"] == pytest.approx(1)
    assert result["contact_micro"]["f1"] == pytest.approx(.4)
    assert result["contact_equal_clip_mean"]["f1"] == pytest.approx(.5)


def test_real_ground_contact_force_and_foot_order():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="terrain_ground" type="plane" size="5 5 .1"/>
      <body name="left_ankle_roll_link" pos="0 0 .09"><freejoint/>
        <geom type="sphere" size=".1" mass="1"/></body>
      <body name="right_ankle_roll_link" pos="1 0 1"><freejoint/>
        <geom type="sphere" size=".1" mass="1"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    forces = FootContacts(model).forces(data)
    assert forces[0, 2] > 10
    np.testing.assert_allclose(forces[0], data.qfrc_constraint[:3], atol=1e-9)
    np.testing.assert_array_equal(forces[1], 0)


def test_mujoco_free_joint_translation_velocity_is_world_frame():
    model = mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".001" gravity="0 0 0"/>
      <worldbody><body name="pelvis" pos="0 0 1" quat=".7071067811865476 0 0 .7071067811865476">
      <freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    velocity = np.array([.3, .8, -.2])
    data.qvel[:3] = velocity
    before = data.qpos[:3].copy()
    mujoco.mj_step(model, data)
    np.testing.assert_allclose((data.qpos[:3] - before) / model.opt.timestep, velocity, atol=1e-12)
