"""Convert the existing 50 Hz PGMT LAFAN1 clips to PMT's native NPZ contract.

No retargeting or resampling is performed. Names determine the permutations;
the original arrays, including velocities and contacts, are retained. A sampled
forward-kinematics check uses PMT's G1 XML before each output is committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.pmt_npz_to_mjlab import BFS_BODIES, BFS_JOINTS

BODY_KEYS = {
    "body_pos": "body_pos_w", "body_quat": "body_quat_w",
    "body_lin_vel": "body_lin_vel_w", "body_ang_vel": "body_ang_vel_w",
}
SCHEMA = "pmt.pgmt-lafan.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _permutation(names, expected, kind):
    names = np.asarray(names).tolist()
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise ValueError(f"{kind} names must contain exactly the PMT G1 names, without duplicates")
    return [names.index(name) for name in expected]


def convert_arrays(source):
    """Validate and reorder a PGMT clip; return PMT arrays without changing values."""
    joint_order = _permutation(source["joint_names"], BFS_JOINTS, "Joint")
    body_order = _permutation(source["body_names"], BFS_BODIES, "Body")
    fps = np.asarray(source["fps"])
    if fps.size != 1 or not np.isfinite(fps).all() or float(fps.reshape(-1)[0]) != 50.0:
        raise ValueError("PGMT stage 1 requires 50 Hz source data; this converter does not resample")
    frames = len(source["joint_pos"])
    if frames < 2:
        raise ValueError("A motion needs at least two frames")
    output = {}
    expected_shapes = {"joint_pos": (frames, 29), "joint_vel": (frames, 29)}
    expected_shapes.update({key: (frames, 30, 4 if key == "body_quat" else 3) for key in BODY_KEYS})
    for key, shape in expected_shapes.items():
        value = np.asarray(source[key])
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"{key} must be finite float32 with shape {shape}; got {value.shape}, {value.dtype}")
        if key in BODY_KEYS:
            output[BODY_KEYS[key]] = value[:, body_order].copy()
        else:
            output[key] = value[:, joint_order].copy()
    if not np.allclose(np.linalg.norm(output["body_quat_w"], axis=-1), 1.0, atol=1e-4, rtol=0):
        raise ValueError("body_quat must contain unit wxyz quaternions")
    if "contacts" in source:
        contacts = np.asarray(source["contacts"])
        if contacts.shape != (frames, 2) or not np.isfinite(contacts).all() or not np.isin(contacts, [0, 1]).all():
            raise ValueError("contacts must have shape [frames, 2] and binary values")
        output["contacts"] = contacts.astype(np.float32)
    output.update(fps=np.array([50.0], dtype=np.float32),
                  joint_names=np.asarray(BFS_JOINTS), body_names=np.asarray(BFS_BODIES))
    if "source_sha256" in source:
        output["source_sha256"] = np.asarray(source["source_sha256"])
    return output


def validate_fk(arrays, model, *, sample_count=17, tolerance=2e-5):
    """Check sampled states, including PMT chunk boundaries, against PMT G1 FK."""
    import mujoco

    if sample_count < 2:
        raise ValueError("FK sample_count must be at least two")
    joint_ids = [model.joint(name).id for name in BFS_JOINTS]
    body_ids = [model.body(name).id for name in BFS_BODIES]
    qids = model.jnt_qposadr[joint_ids]
    root_body = model.body("pelvis").id
    root_joint = int(model.body_jntadr[root_body])
    if root_joint < 0 or model.jnt_type[root_joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("PMT FK asset must expose the pelvis free joint")
    root_q = int(model.jnt_qposadr[root_joint])
    frames = len(arrays["joint_pos"])
    boundaries = np.arange(4000, frames, 4000)
    indices = np.unique(np.concatenate((np.linspace(0, frames - 1, min(sample_count, frames), dtype=int),
                                        boundaries, boundaries - 1)))
    data = mujoco.MjData(model)
    max_position = max_rotation = 0.0
    for frame in indices:
        data.qpos[root_q:root_q + 3] = arrays["body_pos_w"][frame, 0]
        data.qpos[root_q + 3:root_q + 7] = arrays["body_quat_w"][frame, 0]
        data.qpos[qids] = arrays["joint_pos"][frame]
        mujoco.mj_kinematics(model, data)
        max_position = max(max_position, float(np.abs(data.xpos[body_ids] - arrays["body_pos_w"][frame]).max()))
        actual = data.xquat[body_ids]
        expected = arrays["body_quat_w"][frame]
        # q and -q represent the same rotation.
        error = np.minimum(np.abs(actual - expected).max(axis=-1), np.abs(actual + expected).max(axis=-1))
        max_rotation = max(max_rotation, float(error.max()))
    if max(max_position, max_rotation) > tolerance:
        raise ValueError(f"PMT G1 FK mismatch: position={max_position:.3g}, quaternion={max_rotation:.3g}")
    return {"sampled_frames": len(indices), "position_max_abs_m": max_position,
            "quaternion_max_abs": max_rotation, "tolerance": tolerance}


def prepare(source_dir, output_dir, *, robot_xml, expected_clips=40, overwrite=False):
    import mujoco

    source_dir, output_dir, robot_xml = map(lambda p: Path(p).resolve(), (source_dir, output_dir, robot_xml))
    files = sorted(source_dir.glob("*.npz"))
    if len(files) != expected_clips:
        raise ValueError(f"Expected {expected_clips} clips, found {len(files)} in {source_dir}")
    if source_dir == output_dir:
        raise ValueError("Converted files must not overwrite the source directory")
    extra = set(p.name for p in output_dir.glob("*.npz")) - {p.name for p in files}
    if extra:
        raise ValueError(f"Output directory contains unexpected clips: {sorted(extra)}")
    if not overwrite and any((output_dir / p.name).exists() for p in files):
        raise FileExistsError("Converted clips already exist; use --overwrite to regenerate them")
    model = mujoco.MjModel.from_xml_path(str(robot_xml))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": SCHEMA, "source_dir": str(source_dir), "output_dir": str(output_dir),
                "robot_xml": str(robot_xml), "robot_xml_sha256": sha256(robot_xml),
                "converter_sha256": sha256(Path(__file__)), "fps": 50.0,
                "quaternion_order": "wxyz", "retargeted_or_resampled": False,
                "joint_names": BFS_JOINTS, "body_names": BFS_BODIES, "clips": []}
    started = time.monotonic()
    for file in files:
        with np.load(file, allow_pickle=False) as source:
            arrays = convert_arrays(source)
        fk = validate_fk(arrays, model)
        destination = output_dir / file.name
        temporary = destination.with_suffix(".npz.tmp")
        source_hash = sha256(file)
        arrays.update(pgmt_source_npz_sha256=np.asarray(source_hash), pmt_conversion_schema=np.asarray(SCHEMA))
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        record = {"name": file.stem, "source_sha256": source_hash, "output_sha256": sha256(destination),
                  "frames": len(arrays["joint_pos"]), "bytes": destination.stat().st_size, "fk": fk}
        manifest["clips"].append(record)
        print(json.dumps({"name": file.stem, "frames": record["frames"], "fk": fk}), flush=True)
    manifest.update(total_clips=len(files), total_frames=sum(c["frames"] for c in manifest["clips"]),
                    elapsed_seconds=time.monotonic() - started)
    temporary = output_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, output_dir / "manifest.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT.parent / "PGMT/assets/motions_50hz")
    parser.add_argument("--output", type=Path, default=ROOT / "assets/motions/pgmt_lafan1_50hz")
    parser.add_argument("--robot-xml", type=Path, default=ROOT / "TCRS/assets/g1/g1_29dof_rev_1_0.xml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = prepare(args.source, args.output, robot_xml=args.robot_xml, overwrite=args.overwrite)
    print(json.dumps({k: manifest[k] for k in ("total_clips", "total_frames", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
