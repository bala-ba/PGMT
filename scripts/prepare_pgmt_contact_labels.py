"""Convert canonical G1 CSV segments and build offline TCRS/mesh contact labels.

The default reference mesh is a flat triangle plane at z=0. A custom --mesh must
already be in the motion's world coordinates. This never warps a reference onto
training terrain or removes clips. CSV -> resample -> named G1 FK -> mesh labels
share one frame timeline; binary labels are not interpolated.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pmt_tasks.pgmt_contacts import ContactConfig, MeshHeightQuery, foot_collision_probes, label_contacts, reference_plane
from scripts.prepare_pgmt_lafan import sha256, validate_fk
from scripts.pmt_npz_to_mjlab import BFS_BODIES, BFS_JOINTS

SCHEMA = "pmt.pgmt-tcrs-mesh-contacts.v1"
DEFAULT_SOURCE = ROOT.parent / "data/lafan_prepared/clipped_csv"
DEFAULT_OUTPUT = ROOT / "assets/motions/pgmt_clipped_50hz_tcrs_v1"
DEFAULT_XML = ROOT / "TCRS/assets/g1/g1_29dof_rev_1_0.xml"


def read_csv(path):
    """Read hhtools named joints and xyzw roots, checking the actual time axis."""
    with Path(path).open() as stream:
        lines = (line for line in stream if not line.startswith("#"))
        rows = np.genfromtxt(lines, delimiter=",", names=True, dtype=np.float64, encoding="utf-8")
    names = rows.dtype.names or ()
    required = ["time", "root_x", "root_y", "root_z", "root_qx", "root_qy", "root_qz", "root_qw"]
    required += [f"dof_{name}" for name in BFS_JOINTS]
    if rows.ndim != 1 or len(rows) < 3 or set(names) != set(required):
        raise ValueError(f"{path}: expected canonical 29-DoF G1 CSV with at least 3 frames")
    values = np.column_stack([rows[name] for name in required])
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite trajectory")
    timestamps = values[:, 0]
    dt = np.diff(timestamps)
    if abs(timestamps[0]) > 1e-9 or (dt <= 0).any() or not np.allclose(dt, dt[0], rtol=0, atol=1e-8):
        raise ValueError(f"{path}: expected uniform increasing timestamps starting at zero")
    quat = values[:, 4:8]
    if not np.allclose(np.linalg.norm(quat, axis=-1), 1, rtol=0, atol=1e-4):
        raise ValueError(f"{path}: expected unit xyzw root quaternions")
    return dict(time=timestamps, root_pos=values[:, 1:4], root_quat_xyzw=quat,
                joint_pos=values[:, 8:], source_fps=1.0 / dt[0])


def resample_and_fk(source, model, fps=50.0):
    """SLERP root rotation, linear translation/joints, link-frame FK/velocities.

    Never extrapolate beyond the last CSV timestamp. A 300-frame 30 Hz input
    therefore has 499 samples at 50 Hz (0..9.96s), not a fabricated 500th frame.
    Joint angles are bounded robot coordinates, not unwrapped cyclic angles.
    """
    import mujoco

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    times = np.arange(int(np.floor(source["time"][-1] * fps + 1e-9)) + 1) / fps
    if len(times) < 3:
        raise ValueError("Resampled trajectory needs at least three frames")
    root_pos = np.column_stack([np.interp(times, source["time"], source["root_pos"][:, i]) for i in range(3)])
    joints = np.column_stack([np.interp(times, source["time"], source["joint_pos"][:, i]) for i in range(29)])
    quat_xyzw = Slerp(source["time"], Rotation.from_quat(source["root_quat_xyzw"]))(times).as_quat()
    joint_ids = [model.joint(name).id for name in BFS_JOINTS]
    qids = model.jnt_qposadr[joint_ids]
    body_ids = [model.body(name).id for name in BFS_BODIES]
    root_joint = int(model.body_jntadr[model.body("pelvis").id])
    if root_joint < 0 or model.jnt_type[root_joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Expected a free pelvis root")
    root_q = int(model.jnt_qposadr[root_joint])
    data = mujoco.MjData(model)
    positions = np.empty((len(times), 30, 3))
    quaternions = np.empty((len(times), 30, 4))
    for frame in range(len(times)):
        data.qpos[root_q:root_q + 3] = root_pos[frame]
        data.qpos[root_q + 3:root_q + 7] = quat_xyzw[frame, [3, 0, 1, 2]]
        data.qpos[qids] = joints[frame]
        mujoco.mj_kinematics(model, data)
        positions[frame] = data.xpos[body_ids]
        quaternions[frame] = data.xquat[body_ids]
    # Relative rotations R_next R_prev^-1 yield world-frame angular velocities.
    rotations = Rotation.from_quat(quaternions[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(-1, 30, 3, 3)
    delta = rotations[1:] @ rotations[:-1].swapaxes(-1, -2)
    intervals = Rotation.from_matrix(delta.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 30, 3) * fps
    angular = np.concatenate([intervals[:1], (intervals[:-1] + intervals[1:]) * 0.5, intervals[-1:]])
    arrays = dict(joint_pos=joints, joint_vel=np.gradient(joints, 1 / fps, axis=0),
                  body_pos_w=positions, body_quat_w=quaternions,
                  body_lin_vel_w=np.gradient(positions, 1 / fps, axis=0), body_ang_vel_w=angular)
    arrays = {key: value.astype(np.float32) for key, value in arrays.items()}
    arrays.update(fps=np.asarray([fps], dtype=np.float32), time=times,
                  joint_names=np.asarray(BFS_JOINTS), body_names=np.asarray(BFS_BODIES))
    return arrays


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def prepare(source_dir, output_dir, *, robot_xml=DEFAULT_XML, fps=50.0, mesh_path=None,
            ground_height=0.0, selection=None, config=ContactConfig(), overwrite=False):
    import mujoco
    import scipy
    import trimesh

    started = time.monotonic()
    source_dir, output_dir, robot_xml = [Path(p).resolve() for p in (source_dir, output_dir, robot_xml)]
    if source_dir == output_dir or source_dir in output_dir.parents:
        raise ValueError("Write derived assets outside the source data directory")
    source_manifest_path = source_dir / "conversion_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest.get("schema") != "hhtools.canonical-robot-csv" or source_manifest.get("quaternion_order") != "xyzw":
        raise ValueError("Expected the canonical CSV conversion manifest with xyzw roots")
    records = {record["file"]: record for record in source_manifest["clips"]}
    files = sorted(source_dir.glob("*.csv"))
    if selection is not None:
        selected = json.loads(Path(selection).read_text())
        if not isinstance(selected, list) or not all(isinstance(name, str) for name in selected) or len(set(selected)) != len(selected):
            raise ValueError("Selection must be a JSON list of unique CSV basenames")
        missing = set(selected) - {p.name for p in files}
        if missing:
            raise ValueError(f"Selected CSV files not found: {sorted(missing)}")
        files = [p for p in files if p.name in selected]
    if not files or any(p.name not in records for p in files):
        raise ValueError("No input clips, or clips absent from conversion_manifest.json")
    files.sort(key=lambda p: (records[p.name]["motion_name"], records[p.name]["source_start_frame"]))
    expected_files = {p.with_suffix(".npz").name for p in files}
    existing = {p.name for p in output_dir.glob("*.npz")}
    if existing - expected_files:
        raise ValueError("Output contains clips outside this selection; use a new output directory")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output is not empty: {output_dir}; choose a new path or --overwrite")

    loaded = []
    for path in files:
        record = records[path.name]
        digest = sha256(path)
        if digest != record["sha256"]:
            raise ValueError(f"Source checksum mismatch: {path}")
        source = read_csv(path)
        if len(source["time"]) != record["frames"] or not np.isclose(source["source_fps"], source_manifest["sample_rate"]):
            raise ValueError(f"Source frame count/rate disagrees with manifest: {path}")
        loaded.append((path, record, source, digest))
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    centres, radii = foot_collision_probes(model)
    if mesh_path is None:
        lo = np.min([s[2]["root_pos"][:, :2].min(axis=0) for s in loaded], axis=0) - 2.0
        hi = np.max([s[2]["root_pos"][:, :2].max(axis=0) for s in loaded], axis=0) + 2.0
        mesh = reference_plane([lo, hi], ground_height)
    else:
        mesh = trimesh.load_mesh(str(mesh_path), process=False)
    MeshHeightQuery(mesh)  # validate before writing any assets
    output_dir.mkdir(parents=True, exist_ok=True)
    # Remove only this run's completion marker before an explicitly requested rebuild.
    (output_dir / "manifest.json").unlink(missing_ok=True)
    mesh_file = output_dir / "reference_mesh.ply"
    mesh.export(mesh_file)
    query = MeshHeightQuery(trimesh.load_mesh(mesh_file, process=False))
    mesh_hash = sha256(mesh_file)
    code_paths = [Path(__file__).resolve(), ROOT / "pmt_tasks/pgmt_contacts.py",
                  ROOT / "TCRS/stair_mppi/contact_detection.py", ROOT / "scripts/pmt_npz_to_mjlab.py"]
    manifest = dict(schema=SCHEMA, source=str(source_dir), source_manifest_sha256=sha256(source_manifest_path),
                    robot_xml=str(robot_xml), robot_xml_sha256=sha256(robot_xml), fps=fps,
                    source_fps=source_manifest["sample_rate"], contact_parameters=asdict(config),
                    foot_order=list(FOOT for FOOT in ("left_ankle_roll_link", "right_ankle_roll_link")),
                    mesh=dict(file=mesh_file.name, sha256=mesh_hash,
                              source=str(Path(mesh_path).resolve()) if mesh_path else "reference_flat_plane",
                              ground_height_m=ground_height if mesh_path is None else None),
                    code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in code_paths},
                    versions=dict(numpy=np.__version__, scipy=scipy.__version__, mujoco=mujoco.__version__, trimesh=trimesh.__version__),
                    omitted_source_clips=len(records) - len(files),
                    selection_sha256=sha256(Path(selection)) if selection else None,
                    algorithm="TCRS adaptive height + speed + two-pass run filter; mesh support gate",
                    resampling="linear root/joints, root quaternion SLERP, named PMT G1 FK; no extrapolation",
                    boundaries="each CSV is independent; no loop, cross-clip smoothing or source-frame padding",
                    support_mode_codes={"-1": "unknown", "0": "neither foot", "1": "left", "2": "right", "3": "both"},
                    clips=[])
    for index, (path, record, source, digest) in enumerate(loaded):
        arrays = resample_and_fk(source, model, fps)
        labels = label_contacts(arrays, query, centres, radii, config)
        arrays.update(labels)
        segment_id = f"{record['motion_name']}:{record['source_start_frame']}:{record['source_end_frame_exclusive']}"
        arrays.update(source_sha256=np.asarray(digest), source_file=np.asarray(path.name),
                      segment_id=np.asarray(segment_id), contact_schema=np.asarray(SCHEMA),
                      source_frame=record["source_start_frame"] + arrays["time"] * source["source_fps"],
                      reference_mesh_sha256=np.asarray(mesh_hash))
        fk = validate_fk(arrays, model)
        destination = output_dir / path.with_suffix(".npz").name
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, destination)
        valid = labels["contact_valid"]
        penetration = labels["foot_probe_clearance"] < -config.penetration_tolerance_m
        item = dict(file=destination.name, segment_id=segment_id, source_file=path.name,
                    source_sha256=digest, output_sha256=sha256(destination),
                    source_start_frame=record["source_start_frame"], source_frames=len(source["time"]),
                    frames=len(arrays["time"]), last_timestamp_s=float(arrays["time"][-1]),
                    contact_frames=labels["contacts"].sum(axis=0).astype(int).tolist(),
                    valid_foot_frames=valid.sum(axis=0).tolist(),
                    invalid_foot_frames=(~valid).sum(axis=0).tolist(),
                    missing_mesh_rays=int((~labels["foot_mesh_hit"]).sum()),
                    penetrating_foot_frames=penetration.any(axis=-1).sum(axis=0).tolist(),
                    neither_foot_frames=int((labels["support_mode"] == 0).sum()),
                    mesh_changed_foot_labels=int(((labels["contacts"] > 0) != labels["contacts_tcrs"]).sum()),
                    adaptive_height_threshold_m=float(labels["contact_height_threshold"]), fk=fk)
        manifest["clips"].append(item)
        if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(loaded):
            print(json.dumps(dict(completed=index + 1, total=len(loaded), clip=path.stem,
                                  elapsed_seconds=round(time.monotonic() - started, 1))), flush=True)
    manifest.update(total_clips=len(loaded), total_frames=sum(c["frames"] for c in manifest["clips"]),
                    elapsed_seconds=time.monotonic() - started)
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--mesh", type=Path, help="Reference triangle mesh, already aligned with CSV world coordinates")
    parser.add_argument("--ground-height", type=float, default=0.0)
    parser.add_argument("--selection", type=Path, help="JSON list of manually selected CSV basenames")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--contact-z-threshold", type=float, default=0.06)
    parser.add_argument("--contact-speed-threshold", type=float, default=0.5)
    parser.add_argument("--min-contact-frames", type=int, default=4)
    parser.add_argument("--min-swing-frames", type=int, default=2)
    parser.add_argument("--mesh-contact-tolerance", type=float, default=0.02)
    parser.add_argument("--penetration-tolerance", type=float, default=0.02)
    args = parser.parse_args()
    config = ContactConfig(args.contact_z_threshold, args.contact_speed_threshold, args.min_contact_frames,
                           args.min_swing_frames, args.mesh_contact_tolerance, args.penetration_tolerance)
    result = prepare(args.source, args.output, robot_xml=args.robot_xml, fps=args.fps, mesh_path=args.mesh,
                     ground_height=args.ground_height, selection=args.selection, config=config, overwrite=args.overwrite)
    print(json.dumps({k: result[k] for k in ("total_clips", "total_frames", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
