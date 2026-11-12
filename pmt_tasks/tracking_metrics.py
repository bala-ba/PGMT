"""Tracking errors in SI-derived units; arrays are aligned by reference frame.

Positions are body-frame origins in world coordinates, quaternions are wxyz.
Local MPJPE removes each trajectory's pelvis translation, without rotating it.
"""
from __future__ import annotations

import numpy as np


def quaternion_angle(actual, reference):
    """Shortest SO(3) angle, in [0, pi], invariant to quaternion sign."""
    actual, reference = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    an = np.linalg.norm(actual, axis=-1, keepdims=True)
    rn = np.linalg.norm(reference, axis=-1, keepdims=True)
    if np.any(an == 0) or np.any(rn == 0):
        raise ValueError("Zero-norm quaternion")
    dot = np.sum((actual / an) * (reference / rn), axis=-1)
    return 2 * np.arccos(np.clip(np.abs(dot), 0, 1))


def contact_summary(actual, reference):
    """Micro contact scores over frames and feet; undefined ratios are zero."""
    actual, reference = np.asarray(actual, dtype=bool), np.asarray(reference, dtype=bool)
    if actual.shape != reference.shape or actual.ndim != 2 or actual.shape[1] != 2 or not len(actual):
        raise ValueError("Contact arrays must have the same nonempty [T, 2] shape")
    tp = int(np.sum(actual & reference))
    fp = int(np.sum(actual & ~reference))
    fn = int(np.sum(~actual & reference))
    tn = int(np.sum(~actual & ~reference))
    return contact_scores_from_counts(tp, fp, fn, tn) | {
        "both_feet_accuracy": float(np.mean(np.all(actual == reference, axis=1))),
        "actual_contact_fraction": actual.mean(axis=0).tolist(),
        "reference_contact_fraction": reference.mean(axis=0).tolist(),
    }


def contact_scores_from_counts(tp, fp, fn, tn):
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
            "accuracy": (tp + tn) / max(1, tp + fp + fn + tn)}


def tracking_errors(body_pos, ref_body_pos, root_velocity, ref_root_velocity,
                    root_quat, ref_root_quat, joint_pos, ref_joint_pos, tracked, root=0):
    """Return per-frame errors. Joint MSE stays squared until final aggregation."""
    body_pos, ref_body_pos = np.asarray(body_pos, np.float64), np.asarray(ref_body_pos, np.float64)
    if body_pos.shape != ref_body_pos.shape or body_pos.ndim != 3 or body_pos.shape[-1] != 3:
        raise ValueError("Body positions must have matching [T, B, 3] shapes")
    if not len(body_pos):
        raise ValueError("No evaluated frames")
    arrays = [body_pos, ref_body_pos, root_velocity, ref_root_velocity,
              root_quat, ref_root_quat, joint_pos, ref_joint_pos]
    if any(len(a) != len(body_pos) or not np.isfinite(a).all() for a in arrays):
        raise ValueError("All metric inputs must be finite and aligned by frame")
    delta = body_pos - ref_body_pos
    local_delta = delta - delta[:, root:root + 1]
    distances = np.linalg.norm(delta, axis=-1) * 1000
    local_distances = np.linalg.norm(local_delta, axis=-1) * 1000
    articulated = [i for i in range(body_pos.shape[1]) if i != root]
    return {
        "mpkpe_mm": distances[:, tracked].mean(axis=1),
        "mpjpe_g_mm": distances[:, tracked].mean(axis=1),
        "mpjpe_l_mm": local_distances[:, tracked].mean(axis=1),
        "mpjpe_all29_g_mm": distances[:, articulated].mean(axis=1),
        "mpjpe_all29_l_mm": local_distances[:, articulated].mean(axis=1),
        "root_pos_mm": distances[:, root],
        "root_vel_mm_s": np.linalg.norm(np.asarray(root_velocity, np.float64)
                                        - np.asarray(ref_root_velocity, np.float64), axis=-1) * 1000,
        "root_orientation_rad": quaternion_angle(root_quat, ref_root_quat),
        "joint_mse_rad2": np.mean((np.asarray(joint_pos, np.float64)
                                   - np.asarray(ref_joint_pos, np.float64)) ** 2, axis=-1),
    }


def summarize_errors(errors):
    result = {key: float(np.mean(value)) for key, value in errors.items() if key != "joint_mse_rad2"}
    result["joint_rmse_rad"] = float(np.sqrt(np.mean(errors["joint_mse_rad2"])))
    return result


def aggregate_clips(clips):
    """Frame-weighted scores plus equal-clip scores; pool contact counts for F1."""
    weights = np.asarray([c["evaluated_frames"] for c in clips], np.float64)
    if not len(weights) or np.any(weights <= 0):
        raise ValueError("Aggregation requires nonempty clips")
    keys = clips[0]["metrics"].keys()
    micro, macro = {}, {}
    for key in keys:
        values = np.asarray([c["metrics"][key] for c in clips])
        micro[key] = float(np.sqrt(np.average(values ** 2, weights=weights)) if key == "joint_rmse_rad"
                           else np.average(values, weights=weights))
        macro[key] = float(values.mean())
    counts = {k: sum(c["contact"][k] for c in clips) for k in ("tp", "fp", "fn", "tn")}
    contact = contact_scores_from_counts(**counts)
    contact["both_feet_accuracy"] = float(np.average([c["contact"]["both_feet_accuracy"] for c in clips], weights=weights))
    for key in ("actual_contact_fraction", "reference_contact_fraction"):
        contact[key] = np.average([c["contact"][key] for c in clips], axis=0, weights=weights).tolist()
    return {"frame_weighted": micro, "equal_clip_mean": macro, "contact_micro": contact,
            "contact_equal_clip_mean": {key: float(np.mean([c["contact"][key] for c in clips]))
                                        for key in ("f1", "precision", "recall", "accuracy", "both_feet_accuracy")}}
