"""Offline contact labels must retain TCRS timing and actual mesh support."""
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import trimesh

from TCRS.stair_mppi.contact_detection import detect_foot_contacts, stabilize_contact_mask
from pmt_tasks.pgmt_contacts import ContactConfig, FOOT_NAMES, MeshHeightQuery, foot_collision_probes, label_contacts, reference_plane
from scripts.prepare_pgmt_contact_labels import DEFAULT_XML, prepare, read_csv, resample_and_fk
from scripts.prepare_pgmt_lafan import sha256, validate_fk
from scripts.pmt_npz_to_mjlab import BFS_BODIES, BFS_JOINTS


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(DEFAULT_XML))


def foot_states(left_z, right_z):
    frames = len(left_z)
    position = np.zeros((frames, 2, 3))
    position[:, 0, 1], position[:, 1, 1] = 0.15, -0.15
    position[:, 0, 2], position[:, 1, 2] = left_z, right_z
    quat = np.zeros((frames, 2, 4))
    quat[..., 0] = 1
    return dict(body_names=np.asarray(FOOT_NAMES), body_pos_w=position,
                body_quat_w=quat, body_lin_vel_w=np.zeros_like(position))


def test_tcrs_short_contact_removed_short_gap_closed_no_endpoint_wrap():
    raw = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0], dtype=bool)
    expected = np.zeros(len(raw), dtype=bool)
    expected[7:16] = True
    np.testing.assert_array_equal(stabilize_contact_mask(raw), expected)
    result = detect_foot_contacts(np.full((8, 2), 0.035), np.zeros((8, 2)))
    assert result.contacts.all()  # double stance, no forced alternation
    assert result.height_threshold == 0.06


def test_mesh_gate_rejects_static_airborne_foot_despite_adaptive_height(model):
    arrays = foot_states(np.full(8, 0.8), np.full(8, 0.8))
    labels = label_contacts(arrays, MeshHeightQuery(reference_plane([[-2, -2], [2, 2]])), *foot_collision_probes(model))
    assert labels["contacts_tcrs"].all()  # TCRS percentile alone treats a static elevated pose as support
    assert not labels["contacts"].any()
    assert labels["contact_valid"].all() and labels["swing"].all()


def test_left_right_transitions_and_height_translation_invariance(model):
    left = np.r_[np.full(8, 0.035), np.full(8, 0.2)]
    right = left[::-1]
    geometry = foot_collision_probes(model)
    a = foot_states(left, right)
    first = label_contacts(a, MeshHeightQuery(reference_plane([[-2, -2], [2, 2]])), *geometry)
    np.testing.assert_array_equal(first["support_mode"], [1] * 8 + [2] * 8)
    a["body_pos_w"][..., 2] -= 1.2  # inverted terrain origins may be negative
    second = label_contacts(a, MeshHeightQuery(reference_plane([[-2, -2], [2, 2]], -1.2)), *geometry)
    np.testing.assert_array_equal(second["contacts"], first["contacts"])
    np.testing.assert_allclose(second["foot_probe_clearance"], first["foot_probe_clearance"], atol=1e-7)


def test_mesh_rays_find_top_surface_and_flag_holes():
    plane = reference_plane([[-2, -2], [2, 2]], -0.5)
    box = trimesh.creation.box((1, 1, 0.4))
    mesh = trimesh.util.concatenate([plane, box])
    height, hit = MeshHeightQuery(mesh)(np.array([[0, 0], [1.5, 0], [3, 0]]))
    np.testing.assert_allclose(height[:2], [0.2, -0.5])
    np.testing.assert_array_equal(hit, [True, True, False])
    assert np.isnan(height[-1])


@pytest.mark.parametrize("reason", ["penetration", "missing"])
def test_invalid_geometry_is_unknown_not_negative_training_label(model, reason):
    a = foot_states(np.full(8, -0.1 if reason == "penetration" else 0.035), np.full(8, 0.035))
    if reason == "missing":
        a["body_pos_w"][:, 0, 0] = 10
    labels = label_contacts(a, MeshHeightQuery(reference_plane([[-2, -2], [2, 2]])), *foot_collision_probes(model))
    assert not labels["contact_valid"][:, 0].any()
    assert (labels["foot_state"][:, 0] == -1).all()
    assert not labels["swing"][:, 0].any()
    assert labels["contacts"][:, 1].all()
    if reason == "penetration":
        assert labels["contacts_tcrs"].all()  # preserve ungated native output for audit


def test_rotated_sphere_bottoms_use_world_gravity(model):
    a = foot_states(np.full(8, .15), np.full(8, .15))
    xyzw = Rotation.from_euler("y", 45, degrees=True).as_quat()
    a["body_quat_w"][:] = xyzw[[3, 0, 1, 2]]
    centres, radii = foot_collision_probes(model)
    labels = label_contacts(a, MeshHeightQuery(reference_plane([[-2, -2], [2, 2]])), centres, radii)
    expected = Rotation.from_euler("y", 45, degrees=True).apply(centres.reshape(-1, 3)).reshape(2, 4, 3)[..., 2] + .15 - radii
    np.testing.assert_allclose(labels["foot_probe_clearance"][0], expected, atol=1e-7)


def source_motion():
    times = np.arange(300) / 30
    yaw = times * .2
    quat = Rotation.from_euler("z", yaw).as_quat()
    quat[1::2] *= -1  # identical rotation representation must not create angular spikes
    return dict(time=times, root_pos=np.column_stack([times * .3, times * 0, times * 0 + .8]),
                root_quat_xyzw=quat, joint_pos=np.zeros((300, 29)), source_fps=30.)


def test_resampling_frame_support_quaternion_order_and_world_velocity(model):
    arrays = resample_and_fk(source_motion(), model)
    assert len(arrays["time"]) == 499 and arrays["time"][-1] == 9.96
    np.testing.assert_allclose(arrays["body_lin_vel_w"][:, 0], np.tile([.3, 0, 0], (499, 1)), atol=1e-6)
    np.testing.assert_allclose(arrays["body_ang_vel_w"][:, 0], np.tile([0, 0, .2], (499, 1)), atol=1e-6)
    assert arrays["body_names"].tolist() == BFS_BODIES
    assert validate_fk(arrays, model)["position_max_abs_m"] < 2e-5


def test_csv_to_saved_labels_selection_hashes_and_no_implicit_overwrite(tmp_path):
    source_dir, output = tmp_path / "source", tmp_path / "output"
    source_dir.mkdir()
    source = source_motion()
    columns = ["time", "root_x", "root_y", "root_z", "root_qx", "root_qy", "root_qz", "root_qw"]
    columns += [f"dof_{name}" for name in BFS_JOINTS]
    values = np.column_stack([source["time"], source["root_pos"], source["root_quat_xyzw"], source["joint_pos"]])
    csv = source_dir / "walk_clip0.csv"
    np.savetxt(csv, values, delimiter=",", header=",".join(columns), comments="")
    record = dict(file=csv.name, motion_name="walk", source_start_frame=300, source_end_frame_exclusive=600,
                  frames=300, sha256=sha256(csv))
    (source_dir / "conversion_manifest.json").write_text(json.dumps(dict(schema="hhtools.canonical-robot-csv",
                    quaternion_order="xyzw", sample_rate=30., clips=[record])))
    assert read_csv(csv)["joint_pos"].shape == (300, 29)
    result = prepare(source_dir, output)
    assert result["total_clips"] == 1 and result["total_frames"] == 499
    with np.load(output / "walk_clip0.npz", allow_pickle=False) as data:
        assert data["contacts"].shape == data["contact_valid"].shape == (499, 2)
        assert data["foot_names"].tolist() == list(FOOT_NAMES)
        assert data["segment_id"].item() == "walk:300:600"
        assert data["source_frame"][-1] == pytest.approx(598.8)
        assert data["reference_mesh_sha256"].item() == result["mesh"]["sha256"]
    assert sha256(output / "walk_clip0.npz") == result["clips"][0]["output_sha256"]
    with pytest.raises(FileExistsError):
        prepare(source_dir, output)
    csv.write_text(csv.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        prepare(source_dir, tmp_path / "other")


def test_invalid_contact_configuration_rejected():
    with pytest.raises(ValueError):
        ContactConfig(min_contact_frames=0)
    with pytest.raises(ValueError):
        ContactConfig(mesh_contact_tolerance_m=float("nan"))
