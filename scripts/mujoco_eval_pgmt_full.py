"""Run complete PGMT source clips in flat MuJoCo and export tracking metrics.

Example:
  python scripts/mujoco_eval_pgmt_full.py --contract PATH --output PATH

Each clip starts at reference frame 0, then runs continuously to its last frame.
No tracking-error termination or 4000-frame chunk reset is applied. The policy's
future reference window also crosses training chunk boundaries. Metrics cover
post-step frames 1..T-1, including tracking failures. No videos are rendered.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
import torch

from pmt_tasks.backends.mujoco_standalone import build_model, create_adapter, load_policy, validate_contract
from pmt_tasks.tracking_metrics import aggregate_clips, contact_summary, summarize_errors, tracking_errors


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reset_full_clip(adapter, motion_id):
    adapter.reset(motion_id, 0)
    adapter.reference_start = 0
    adapter.reference_end = len(adapter.raw["joint_pos"])


class FootContacts:
    """Net world contact force on each foot from ground collisions only."""

    names = ("left_ankle_roll_link", "right_ankle_roll_link")

    def __init__(self, model):
        self.model = model
        self.ground = model.geom("terrain_ground").id
        self.foot_for_geom = np.full(model.ngeom, -1, dtype=int)
        feet = [model.body(n).id for n in self.names]
        for geom, body in enumerate(model.geom_bodyid):
            while body:
                if body in feet:
                    self.foot_for_geom[geom] = feet.index(body)
                    break
                body = model.body_parentid[body]
        self.wrench = np.zeros(6)

    def forces(self, data):
        forces = np.zeros((2, 3))
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.geom1 == self.ground:
                geom, sign = contact.geom2, 1
            elif contact.geom2 == self.ground:
                geom, sign = contact.geom1, -1
            else:
                continue
            foot = self.foot_for_geom[geom]
            if foot >= 0:
                mujoco.mj_contactForce(self.model, data, i, self.wrench)
                forces[foot] += sign * contact.frame.reshape(3, 3).T @ self.wrench[:3]
        return forces


def run_clip(adapter, policy, contact_reader, index, output, max_steps, threshold):
    reset_full_clip(adapter, index)
    raw, data = adapter.raw, adapter.data
    total = len(raw["joint_pos"])
    steps = min(max_steps, total - 1) if max_steps else total - 1
    reference_contacts = raw.get("contacts")
    if (reference_contacts is None or reference_contacts.shape != (total, 2)
            or not np.isin(reference_contacts, [0, 1]).all()):
        raise ValueError("Expected binary reference contacts [T, 2], ordered left/right")
    dt = adapter.contract["sim_dt"] * adapter.contract["decimation"]
    qpos = np.empty((steps, adapter.model.nq))
    qvel = np.empty((steps, adapter.model.nv))
    body_pos = np.empty((steps, len(adapter.body_ids), 3))
    force = np.empty((steps, 2, 3))
    actions = np.empty((steps, 29), np.float32)
    counts = np.empty(steps, np.int32)
    initial_qpos, initial_qvel = data.qpos.copy(), data.qvel.copy()
    start = time.monotonic()
    last_log = start
    finished = 0
    failure = None
    with torch.inference_mode():
        obs = adapter.observations()
        try:
            for step in range(steps):
                action = policy.act_inference(obs)[0].numpy()
                adapter.step(action)
                if adapter.frame != step + 1 or abs(data.time - (step + 1) * dt) > 1e-6:
                    raise RuntimeError("Reference frame/time discontinuity (possible numerical autoreset)")
                if any(w.number for w in data.warning):
                    raise RuntimeError("MuJoCo solver warning; remaining frames are not counted")
                qpos[step], qvel[step] = data.qpos, data.qvel
                body_pos[step] = data.xpos[adapter.body_ids]
                force[step] = contact_reader.forces(data)
                counts[step], actions[step] = data.ncon, action
                finished += 1
                if step + 1 < steps:
                    obs = adapter.observations()
                now = time.monotonic()
                if now - last_log >= 20:
                    print(f"[PROGRESS] motion={index} frame={finished}/{steps} sim={data.time:.2f}s wall={now-start:.1f}s", flush=True)
                    last_log = now
        except (FloatingPointError, RuntimeError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
    if not finished:
        raise RuntimeError(f"No valid evaluated frames for motion {index}: {failure}")
    qpos, qvel, body_pos, force, counts, actions = [a[:finished] for a in (qpos, qvel, body_pos, force, counts, actions)]
    frames = np.arange(1, finished + 1)
    root = adapter.contract["body_names"].index("pelvis")
    errors = tracking_errors(body_pos, raw["body_pos_w"][frames], qvel[:, :3],
                             raw["body_lin_vel_w"][frames, root], qpos[:, 3:7],
                             raw["body_quat_w"][frames, root], qpos[:, adapter.qids],
                             raw["joint_pos"][frames], adapter.tracked, root)
    actual_contact = np.linalg.norm(force, axis=-1) > threshold
    ref_contact = reference_contacts[frames] > .5
    motion = Path(adapter.contract["motion_files"][index])
    name = f"{index:02d}_{motion.stem}"
    trajectory = output / "trajectories" / f"{name}.npz"
    np.savez_compressed(trajectory, motion_frame=frames, time_s=frames * dt,
                        qpos=qpos, qvel=qvel, body_pos_w=body_pos,
                        initial_qpos=initial_qpos, initial_qvel=initial_qvel,
                        joint_qpos_indices=adapter.qids, joint_qvel_indices=adapter.vids,
                        action=actions, foot_contact_force_w_N=force, contact_count=counts,
                        actual_contacts=actual_contact, reference_contacts=ref_contact,
                        **errors)
    result = {"motion_index": index, "motion": motion.stem, "motion_file": str(motion.resolve()),
              "motion_sha256": sha256(motion), "source_frames": total, "evaluated_frames": finished,
              "full_clip_completed": finished == total - 1 and failure is None,
              "requested_steps_completed": finished == steps and failure is None,
              "simulated_seconds": finished * dt, "wall_seconds": time.monotonic() - start,
              "mj_step_calls": adapter.physics_steps, "resets_within_clip": 0,
              "runtime_error": failure,
              "mujoco_warnings": {str(i): int(w.number) for i, w in enumerate(data.warning) if w.number},
              "metrics": summarize_errors(errors), "contact": contact_summary(actual_contact, ref_contact),
              "trajectory": str(trajectory.resolve()),
              "diagnostics": {"max_root_pos_mm": float(errors["root_pos_mm"].max()),
                              "p95_root_pos_mm": float(np.percentile(errors["root_pos_mm"], 95)),
                              "fraction_root_pos_over_500mm": float(np.mean(errors["root_pos_mm"] > 500)),
                              "fraction_root_orientation_over_1rad": float(np.mean(errors["root_orientation_rad"] > 1)),
                              "min_pelvis_height_m": float(body_pos[:, root, 2].min())}}
    for side in range(2):
        # Duplicate a column to reuse the two-foot reducer; ratios are unchanged.
        scores = contact_summary(np.repeat(actual_contact[:, side:side+1], 2, axis=1),
                                 np.repeat(ref_contact[:, side:side+1], 2, axis=1))
        result["contact"]["left" if side == 0 else "right"] = {
            k: scores[k] for k in ("f1", "precision", "recall", "accuracy")}
    write_json(output / "clips" / f"{name}.json", result)
    print(f"[CLIP] {index:02d} {motion.stem} frames={finished}/{total-1} "
          f"MPKPE={result['metrics']['mpkpe_mm']:.3f}mm RootPos={result['metrics']['root_pos_mm']:.3f}mm "
          f"contact_F1={result['contact']['f1']:.4f} wall={result['wall_seconds']:.1f}s", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-indices", type=int, nargs="+", help="Default: all source files")
    parser.add_argument("--max-steps", type=int, help="Diagnostic limit only; output marks incomplete clips")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--contact-threshold-n", type=float, default=10.)
    args = parser.parse_args()
    if (args.max_steps is not None and args.max_steps <= 0) or args.threads < 1:
        parser.error("max-steps and threads must be positive")
    if not np.isfinite(args.contact_threshold_n) or args.contact_threshold_n < 0:
        parser.error("contact threshold must be finite and nonnegative")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use an empty output directory to preserve existing results")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "clips").mkdir()
    (args.output / "trajectories").mkdir()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    contract = torch.load(args.contract, map_location="cpu", weights_only=False)
    if contract.get("policy_class") != "PGMTActorCritic" or contract.get("terrain_mesh"):
        raise ValueError("This evaluator requires a flat PGMT contract")
    if contract["body_names"][0] != "pelvis" or len(contract["body_names"]) != 30:
        raise ValueError("Expected G1 body order: pelvis followed by 29 articulated bodies")
    indices = args.motion_indices if args.motion_indices is not None else list(range(len(contract["motion_files"])))
    if not indices or len(indices) != len(set(indices)) or min(indices) < 0 or max(indices) >= len(contract["motion_files"]):
        raise ValueError("Motion indices must be distinct and within the contract file list")
    model, geometry = build_model(contract, args.output / "model")
    adapter = create_adapter(contract, model)
    policy = load_policy(contract)
    parity = validate_contract(contract, adapter, policy)
    write_json(args.output / "parity.json", parity)
    if not parity["passed"]:
        raise RuntimeError(f"Isaac fixture parity failed: {parity}")
    metadata = {
        "checkpoint": contract["checkpoint"], "checkpoint_sha256": sha256(contract["checkpoint"]),
        "contract": str(args.contract.resolve()), "contract_sha256": sha256(args.contract),
        "physics": f"MuJoCo {mujoco.__version__} CPU mj_step", "geometry": geometry,
        "sim_dt": contract["sim_dt"], "decimation": contract["decimation"],
        "control_dt": contract["sim_dt"] * contract["decimation"], "threads": args.threads,
        "motion_indices": indices, "max_steps": args.max_steps, "seed": 42,
        "initialization": "Exact reference frame-0 pose and velocity; no reset noise; exported initial previous target",
        "rollout": "Continuous complete source clips; future references cross native training chunks; clamp only at source end",
        "metric_frames": "Post-step reference frames 1..T-1; initial frame excluded; tracking failures included; no resets within clips",
        "root_body": "pelvis", "policy_anchor_body": contract.get("anchor_body", "torso_link"),
        "tracked_bodies": contract["tracked_bodies"], "body_names": contract["body_names"],
        "joint_names": contract["joint_names"],
        "metric_definitions": {
            "mpkpe_mm": "1000 * mean_{frame,tracked link} ||p_sim_world - p_ref_world||_2; 14 tracked links including pelvis",
            "root_pos_mm": "1000 * mean_frame ||pelvis_sim_world - pelvis_ref_world||_2",
            "root_vel_mm_s": "1000 * mean_frame ||qvel_sim[:3] - reference pelvis body_lin_vel_w||_2; world linear velocity, no finite differences",
            "root_orientation_rad": "mean_frame 2*acos(clip(abs(dot(normalize(q_sim_wxyz),normalize(q_ref_wxyz))),0,1)); pelvis full SO(3) angle",
            "joint_rmse_rad": "sqrt(mean_{frame,29 joint} (q_sim-q_ref)^2); no wrapping; pool MSE before square root",
            "mpjpe_g_mm": "Same 14 tracked body origins and world-space formula as MPKPE; equal by definition",
            "mpjpe_l_mm": "1000*mean ||(p_sim-pelvis_sim)-(p_ref-pelvis_ref)||_2 over the same 14 links; no heading/rotation alignment",
            "mpjpe_all29_g_mm": "World-space position error over all 29 articulated body origins, excluding pelvis",
            "mpjpe_all29_l_mm": "Pelvis-relative position error over all 29 articulated body origins, excluding pelvis",
        },
        "contact": {"feet": list(FootContacts.names), "threshold_N": args.contact_threshold_n,
                    "actual": "Norm of net 3D world force on each foot from ground contacts > threshold; sample after each control step and mj_forward",
                    "reference": "NPZ contacts[:,0:2] > 0.5, left then right",
                    "reference_label_generation": "Heuristic: ankle z <= per-clip feet-z 2nd percentile + 0.05m AND ankle speed < 0.35m/s; not measured force labels",
                    "reference_label_source": "/home/lzp/projects/perceptive_tracking/PGMT/scripts/preprocess_motions.py",
                    "f1": "2TP/(2TP+FP+FN), pooled over frames and both feet; undefined ratios set to zero",
                    "accuracy": "(TP+TN)/(2*evaluated_frames)", "both_feet_accuracy": "Fraction of frames where both foot labels match"},
        "aggregation": "frame_weighted pools all evaluated frames (joint MSE before sqrt); equal_clip_mean averages each clip's reported score; contact_micro pools confusion counts",
        "sources_sha256": {str(p): sha256(p) for p in [Path(__file__), Path("pmt_tasks/tracking_metrics.py"), Path("pmt_tasks/backends/mujoco_standalone.py")]},
        "interface_parity_passed": True,
    }
    write_json(args.output / "metadata.json", metadata)
    start = time.monotonic()
    contacts = FootContacts(model)
    clips = []
    for index in indices:
        clips.append(run_clip(adapter, policy, contacts, index, args.output, args.max_steps, args.contact_threshold_n))
        result = {"completed_clips": len(clips), "requested_clips": len(indices),
                  "full_clips_completed": sum(c["full_clip_completed"] for c in clips),
                  "evaluated_frames": sum(c["evaluated_frames"] for c in clips),
                  "simulated_seconds": sum(c["simulated_seconds"] for c in clips),
                  "mj_step_calls": sum(c["mj_step_calls"] for c in clips),
                  "wall_seconds": time.monotonic() - start,
                  "all_requested_full_clips_completed": len(clips) == len(indices) and all(c["full_clip_completed"] for c in clips),
                  "runtime_passed": all(c["requested_steps_completed"] and not c["mujoco_warnings"] for c in clips),
                  "tracking_failure_policy": "included; full completion is not a tracking success criterion",
                  "checkpoint_sha256": metadata["checkpoint_sha256"], **aggregate_clips(clips)}
        write_json(args.output / "summary.json", result)
        rows = []
        for clip in clips:
            rows.append({k: clip[k] for k in ("motion_index", "motion", "source_frames", "evaluated_frames", "full_clip_completed", "simulated_seconds")} |
                        clip["metrics"] | {f"contact_{k}": clip["contact"][k] for k in ("f1", "accuracy", "precision", "recall", "both_feet_accuracy")} |
                        clip["diagnostics"])
        with (args.output / "per_motion.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(result, indent=2), flush=True)
    if not result["runtime_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
