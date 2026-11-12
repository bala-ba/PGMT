"""Simulator-independent PMT and SONIC failure predicates for G1 evaluation."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

from pmt_tasks.tracking_metrics import quaternion_angle


def load_protocols(pmt_env, sonic_root):
    # BaseLoader reads the Isaac YAML's Python tags as plain data, without
    # constructing Python objects. Only named scalar parameters are consumed.
    native = yaml.load(Path(pmt_env).read_text(), Loader=yaml.BaseLoader)
    terms = native["terminations"]
    expected = {"anchor_pos": "bad_anchor_pos_z_only", "anchor_ori": "bad_anchor_ori",
                "ee_body_pos": "bad_motion_body_pos_z_only", "time_out": "time_out"}
    if set(terms) != set(expected):
        raise ValueError("Unexpected PMT termination terms; review the saved training configuration")
    for key, suffix in expected.items():
        if terms[key]["func"].split(":")[-1] != suffix:
            raise ValueError(f"Unexpected PMT predicate: {terms[key]['func']}")
    pmt = {"family": "pmt", "anchor": native["commands"]["motion"]["anchor_body"],
           "anchor_height_m": float(terms["anchor_pos"]["params"]["threshold"]),
           "orientation_kind": "projected_gravity_z_difference",
           "orientation_threshold": float(terms["anchor_ori"]["params"]["threshold"]),
           "ee_height_m": float(terms["ee_body_pos"]["params"]["threshold"]),
           "end_effectors": terms["ee_body_pos"]["params"]["body_names"], "timeout_s": None}
    sonic_root = Path(sonic_root)
    base = sonic_root / "gear_sonic/config/manager_env/terminations"
    evaluation = yaml.safe_load((base / "tracking/eval.yaml").read_text())
    ee = yaml.safe_load((base / "terms/ee_body_pos.yaml").read_text())
    motion = yaml.safe_load((sonic_root / "gear_sonic/config/manager_env/commands/terms/motion.yaml").read_text())
    if evaluation["anchor_pos"]["params"]["threshold_adaptive"] or evaluation["ee_body_pos"]["params"]["threshold_adaptive"]:
        raise ValueError("Expected non-adaptive SONIC tracking/eval predicates")
    sonic = {"family": "sonic", "anchor": motion["motion"]["anchor_body"],
             "anchor_height_m": float(evaluation["anchor_pos"]["params"]["threshold"]),
             "orientation_kind": "squared_full_rotation_angle_rad2",
             "orientation_threshold": float(evaluation["anchor_ori_full"]["params"]["threshold"]),
             "ee_height_m": float(evaluation["ee_body_pos"]["params"]["threshold"]),
             "end_effectors": ee["ee_body_pos"]["params"]["body_names"], "timeout_s": None}
    return {"pmt_full": pmt, "sonic_eval": sonic,
            "pmt_native_15s": dict(pmt, timeout_s=float(native["episode_length_s"]))}


def failure_signals(actual_pos, reference_pos, actual_anchor_quat, reference_anchor_quat, body_names, rule):
    """Aligned post-step states; supports a frame or an array of frames.

    Both projects only yaw-align end-effectors for these predicates. Their
    reference Z coordinates are therefore unchanged by the relative transform.
    PMT's gravity-Z difference is dimensionless and ignores yaw.
    """
    actual_pos, reference_pos = np.asarray(actual_pos), np.asarray(reference_pos)
    anchor = body_names.index(rule["anchor"])
    ee = [body_names.index(n) for n in rule["end_effectors"]]
    height = np.abs(actual_pos[..., anchor, 2] - reference_pos[..., anchor, 2])
    ee_height = np.max(np.abs(actual_pos[..., ee, 2] - reference_pos[..., ee, 2]), axis=-1)
    if rule["orientation_kind"] == "projected_gravity_z_difference":
        # quat_apply_inverse(q, [0,0,-1]).z = -1 + 2*(x*x+y*y).
        aq, bq = np.asarray(actual_anchor_quat), np.asarray(reference_anchor_quat)
        orientation = np.abs(2 * np.sum(aq[..., 1:3]**2 - bq[..., 1:3]**2, axis=-1))
    elif rule["orientation_kind"] == "squared_full_rotation_angle_rad2":
        orientation = quaternion_angle(actual_anchor_quat, reference_anchor_quat)**2
    else:
        raise ValueError("Unknown orientation predicate")
    values = {"anchor_height": height, "anchor_orientation": orientation, "end_effector_height": ee_height}
    flags = {"anchor_height": height > rule["anchor_height_m"],
             "anchor_orientation": orientation > rule["orientation_threshold"],
             "end_effector_height": ee_height > rule["ee_height_m"]}
    return values, flags


def first_stop(flags, source_frames, dt, timeout_s=None):
    """First failure or horizon, including the failing frame in metric samples.

    Failure takes precedence if it coincides with the horizon, as in the local
    SONIC MuJoCo batch evaluator. Frame 0 initializes state and is not scored.
    """
    horizon = source_frames - 1
    if timeout_s is not None:
        horizon = min(horizon, math.ceil(timeout_s / dt))
    if horizon <= 0 or any(len(v) < horizon for v in flags.values()):
        raise ValueError("Incomplete input trace or empty evaluation horizon")
    failed = np.logical_or.reduce([np.asarray(v[:horizon], bool) for v in flags.values()])
    hits = np.flatnonzero(failed)
    frame = int(hits[0]) + 1 if len(hits) else horizon
    reasons = [k for k, v in flags.items() if v[frame - 1]] if len(hits) else []
    status = "failure" if reasons else ("motion_end" if frame == source_frames - 1 else "timeout")
    completed_steps = frame - 1 if reasons else frame
    return {"status": status, "first_failure_frame": frame if reasons else None,
            "last_frame": frame, "evaluated_frames": frame, "failure_reasons": reasons,
            "stop_time_s": frame * dt, "horizon_frames": horizon,
            "horizon_passed": not reasons, "full_motion_success": status == "motion_end",
            "motion_completion": completed_steps / (source_frames - 1),
            "horizon_completion": completed_steps / horizon}
