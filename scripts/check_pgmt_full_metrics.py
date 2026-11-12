"""Independently recompute a completed PGMT batch from saved MuJoCo states.

This deliberately does not import tracking_metrics or the evaluator. It checks
frame coverage, reference hashes, per-clip values, and pooled statistics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def check_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-9)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit(output):
    metadata = json.loads((output / "metadata.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert summary["all_requested_full_clips_completed"] and summary["runtime_passed"]
    assert digest(metadata["checkpoint"]) == metadata["checkpoint_sha256"]
    assert digest(metadata["contract"]) == metadata["contract_sha256"]
    for path, expected in metadata["sources_sha256"].items():
        assert digest(path) == expected, f"Source changed since evaluation: {path}"
    tracked = [metadata["body_names"].index(n) for n in metadata["tracked_bodies"]]
    root = metadata["body_names"].index(metadata["root_body"])
    assert root == 0
    accumulated, per_clip, counts = {}, [], np.zeros(4, np.int64)
    total = 0
    both_feet_correct = 0
    max_difference = 0.
    for clip_file in sorted((output / "clips").glob("*.json")):
        clip = json.loads(clip_file.read_text())
        assert clip["full_clip_completed"] and clip["runtime_error"] is None and not clip["mujoco_warnings"]
        assert digest(clip["motion_file"]) == clip["motion_sha256"]
        with np.load(clip["trajectory"]) as saved, np.load(clip["motion_file"]) as reference:
            frames = saved["motion_frame"]
            n = len(frames)
            np.testing.assert_array_equal(frames, np.arange(1, len(reference["joint_pos"])))
            check_close(saved["time_s"], frames * metadata["control_dt"])
            assert clip["evaluated_frames"] == n and clip["source_frames"] == n + 1
            assert clip["mj_step_calls"] == n * metadata["decimation"]
            assert clip["resets_within_clip"] == 0
            actual_pos = saved["body_pos_w"]
            target_pos = reference["body_pos_w"][frames].astype(np.float64)
            check_close(saved["qpos"][:, :3], actual_pos[:, root])
            check_close(saved["initial_qpos"][:3], reference["body_pos_w"][0, root])
            check_close(saved["initial_qvel"][:3], reference["body_lin_vel_w"][0, root])
            global_mm = np.sqrt(np.sum((actual_pos - target_pos)**2, axis=-1)) * 1000
            local_delta = (actual_pos - actual_pos[:, :1]) - (target_pos - target_pos[:, :1])
            local_mm = np.sqrt(np.sum(local_delta**2, axis=-1)) * 1000
            root_vel = saved["qvel"][:, :3] - reference["body_lin_vel_w"][frames, root]
            aq, bq = saved["qpos"][:, 3:7], reference["body_quat_w"][frames, root].astype(np.float64)
            aq /= np.sqrt((aq**2).sum(axis=-1, keepdims=True))
            bq /= np.sqrt((bq**2).sum(axis=-1, keepdims=True))
            orientation = 2 * np.arccos(np.minimum(1, np.abs((aq*bq).sum(axis=-1))))
            joints = saved["qpos"][:, saved["joint_qpos_indices"]] - reference["joint_pos"][frames]
            errors = {"mpkpe_mm": global_mm[:, tracked].mean(axis=1),
                      "mpjpe_g_mm": global_mm[:, tracked].mean(axis=1),
                      "mpjpe_l_mm": local_mm[:, tracked].mean(axis=1),
                      "mpjpe_all29_g_mm": global_mm[:, 1:].mean(axis=1),
                      "mpjpe_all29_l_mm": local_mm[:, 1:].mean(axis=1),
                      "root_pos_mm": global_mm[:, root],
                      "root_vel_mm_s": np.sqrt((root_vel**2).sum(axis=-1)) * 1000,
                      "root_orientation_rad": orientation, "joint_mse_rad2": (joints**2).mean(axis=1)}
            metrics = {k: float(v.mean()) for k, v in errors.items() if k != "joint_mse_rad2"}
            metrics["joint_rmse_rad"] = float(np.sqrt(errors["joint_mse_rad2"].mean()))
            for key, error in errors.items():
                check_close(error, saved[key])
                accumulated[key] = accumulated.get(key, 0.) + error.sum()
            for key, value in metrics.items():
                check_close(value, clip["metrics"][key])
                max_difference = max(max_difference, abs(value-clip["metrics"][key]))
            actual = np.sqrt((saved["foot_contact_force_w_N"]**2).sum(axis=-1)) > metadata["contact"]["threshold_N"]
            ref = reference["contacts"][frames] > .5
            np.testing.assert_array_equal(actual, saved["actual_contacts"])
            np.testing.assert_array_equal(ref, saved["reference_contacts"])
            confusion = [int((actual & ref).sum()), int((actual & ~ref).sum()),
                         int((~actual & ref).sum()), int((~actual & ~ref).sum())]
            for k, v in zip(("tp", "fp", "fn", "tn"), confusion):
                assert v == clip["contact"][k]
            counts += confusion
            both_feet_correct += np.all(actual == ref, axis=1).sum()
            total += n
            per_clip.append(metrics)
    assert len(per_clip) == len(metadata["motion_indices"]) == summary["full_clips_completed"]
    assert total == summary["evaluated_frames"]
    assert summary["mj_step_calls"] == total * metadata["decimation"]
    check_close(summary["simulated_seconds"], total * metadata["control_dt"])
    for key in per_clip[0]:
        if key == "joint_rmse_rad":
            expected = np.sqrt(accumulated["joint_mse_rad2"] / total)
        else:
            expected = accumulated[key] / total
        check_close(summary["frame_weighted"][key], expected)
        check_close(summary["equal_clip_mean"][key], np.mean([c[key] for c in per_clip]))
    tp, fp, fn, tn = counts.tolist()
    contact = {"f1": 2*tp/max(1, 2*tp+fp+fn), "accuracy": (tp+tn)/(2*total),
               "precision": tp/max(1, tp+fp), "recall": tp/max(1, tp+fn),
               "both_feet_accuracy": both_feet_correct/total}
    for key, value in contact.items():
        check_close(summary["contact_micro"][key], value)
    with (output / "per_motion.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(per_clip) and sum(int(r["evaluated_frames"]) for r in rows) == total
    result = {"passed": True, "clips": len(per_clip), "evaluated_frames": total,
              "initialization_frames_excluded": len(per_clip), "source_frames": total + len(per_clip),
              "max_scalar_metric_difference": max_difference,
              "checks": ["complete consecutive frames", "checkpoint/contract/source/reference hashes", "physics step count",
                         "reference initialization", "all per-frame errors independently recomputed", "per-clip and pooled metrics",
                         "contacts recomputed from saved forces", "CSV frame coverage"]}
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    audit(parser.parse_args().output)
