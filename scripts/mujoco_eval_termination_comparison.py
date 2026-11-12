"""Compare PMT/SONIC first-failure prefixes and verify them with fresh physics.

The source contains complete, deterministic MuJoCo trajectories. Each predicate
is passive until termination, so both tests share the same physical prefix.
--verify-online reruns every motion until all protocol results are determined,
checks each first-stop event online, and compares every qpos/qvel/contact sample.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
import torch

from pmt_tasks.backends.mujoco_standalone import create_adapter, load_policy, validate_contract
from pmt_tasks.tracking_metrics import aggregate_clips, contact_summary, summarize_errors
from pmt_tasks.tracking_terminations import failure_signals, first_stop, load_protocols
from scripts.mujoco_eval_pgmt_full import FootContacts, reset_full_clip, sha256, write_json


def load_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(clips):
    successes = [c for c in clips if c["full_motion_success"]]
    counts = Counter(c["status"] for c in clips)
    return {"motions": len(clips), "failure_count": counts["failure"],
            "timeout_count": counts["timeout"], "full_motion_success_count": len(successes),
            "full_motion_success_rate": len(successes)/len(clips) if not counts["timeout"] else None,
            "horizon_pass_rate": sum(c["horizon_passed"] for c in clips)/len(clips),
            "mean_motion_completion": float(np.mean([c["motion_completion"] for c in clips])),
            "mean_horizon_completion": float(np.mean([c["horizon_completion"] for c in clips])),
            "evaluated_frames": sum(c["evaluated_frames"] for c in clips),
            "evaluated_seconds": sum(c["stop_time_s"] for c in clips),
            "failure_reason_counts": dict(Counter(k for c in clips for k in c["failure_reasons"])),
            "all_valid_prefixes": aggregate_clips(clips),
            "full_motion_success_only": aggregate_clips(successes) if successes else None}


def compare_saved(source, output, metadata, rules, model):
    names = metadata["body_names"]
    dt = metadata["control_dt"]
    torso_id = model.body("torso_link").id
    fk_data = mujoco.MjData(model)
    result = {key: [] for key in rules}
    rows = []
    error_keys = ("mpkpe_mm", "mpjpe_g_mm", "mpjpe_l_mm", "mpjpe_all29_g_mm", "mpjpe_all29_l_mm",
                  "root_pos_mm", "root_vel_mm_s", "root_orientation_rad", "joint_mse_rad2")
    for clip_path in sorted((source / "clips").glob("*.json")):
        clip = load_json(clip_path)
        if not clip["full_clip_completed"] or sha256(clip["motion_file"]) != clip["motion_sha256"]:
            raise ValueError(f"Incomplete trajectory or changed reference: {clip_path}")
        with np.load(clip["trajectory"]) as saved, np.load(clip["motion_file"]) as raw:
            frames, qpos = saved["motion_frame"], saved["qpos"]
            np.testing.assert_array_equal(frames, np.arange(1, clip["source_frames"]))
            actual_pos, reference_pos = saved["body_pos_w"], raw["body_pos_w"][frames]
            reference_quat = raw["body_quat_w"][frames]
            torso_quat = np.empty((len(frames), 4))
            for i, state in enumerate(qpos):
                fk_data.qpos[:] = state
                mujoco.mj_kinematics(model, fk_data)
                torso_quat[i] = fk_data.xquat[torso_id]
            errors = {key: saved[key] for key in error_keys}
            actual_contact, ref_contact = saved["actual_contacts"], saved["reference_contacts"]
            traces = {"motion_frame": frames, "torso_quat_wxyz": torso_quat}
            for name, rule in rules.items():
                anchor = names.index(rule["anchor"])
                actual_quat = torso_quat if rule["anchor"] == "torso_link" else qpos[:, 3:7]
                values, flags = failure_signals(actual_pos, reference_pos, actual_quat,
                                               reference_quat[:, anchor], names, rule)
                stop = first_stop(flags, clip["source_frames"], dt, rule["timeout_s"])
                n = stop["evaluated_frames"]
                item = {"protocol": name, "motion_index": clip["motion_index"], "motion": clip["motion"],
                        "source_frames": clip["source_frames"], "source_clip": str(clip_path.resolve()),
                        "source_trajectory": clip["trajectory"], **stop,
                        "metrics": summarize_errors({k: v[:n] for k, v in errors.items()}),
                        "contact": contact_summary(actual_contact[:n], ref_contact[:n]),
                        "termination_values_at_stop": {k: float(v[n-1]) for k, v in values.items()}}
                result[name].append(item)
                write_json(output / name / f"{clip['motion_index']:02d}_{clip['motion']}.json", item)
                rows.append({k: item[k] for k in ("protocol", "motion_index", "motion", "status", "source_frames",
                             "evaluated_frames", "stop_time_s", "first_failure_frame", "motion_completion", "horizon_completion")} |
                            {"failure_reasons": ";".join(item["failure_reasons"])} | item["metrics"] |
                            {f"contact_{k}": item["contact"][k] for k in ("f1", "accuracy", "precision", "recall", "both_feet_accuracy")})
                for key, value in values.items():
                    traces[f"{name}_{key}"] = value
                for key, value in flags.items():
                    traces[f"{name}_{key}_failed"] = value
            np.savez_compressed(output / "termination_traces" / f"{clip['motion_index']:02d}_{clip['motion']}.npz", **traces)
        print(f"[PREFIX] {clip['motion_index']:02d} {clip['motion']} " + " ".join(
            f"{k}={v[-1]['status']}@{v[-1]['last_frame']}" for k, v in result.items()), flush=True)
    write_csv(output / "per_motion.csv", rows)
    summary = {key: summarize(clips) for key, clips in result.items()}
    write_json(output / "summary.json", summary)
    return result


def verify_online(metadata, rules, results, model, output):
    contract = torch.load(metadata["contract"], map_location="cpu", weights_only=False)
    adapter = create_adapter(contract, model)
    policy = load_policy(contract)
    parity = validate_contract(contract, adapter, policy)
    write_json(output / "parity.json", parity)
    if not parity["passed"]:
        raise RuntimeError("Contract fixture parity failed")
    reader = FootContacts(model)
    names, dt = metadata["body_names"], metadata["control_dt"]
    anchors = {k: names.index(v["anchor"]) for k, v in rules.items()}
    anchor_ids = {k: model.body(v["anchor"]).id for k, v in rules.items()}
    logs = []
    started = time.monotonic()
    with torch.inference_mode():
        for i, expected_pmt in enumerate(results["pmt_full"]):
            clip = load_json(expected_pmt["source_clip"])
            reset_full_clip(adapter, clip["motion_index"])
            expected = {key: values[i] for key, values in results.items()}
            limit = max(x["last_frame"] for x in expected.values())
            with np.load(clip["trajectory"]) as saved:
                qpos, qvel = saved["qpos"][:limit], saved["qvel"][:limit]
                forces = saved["foot_contact_force_w_N"][:limit]
            stops = {}
            max_errors = dict(qpos=0., qvel=0., contact_force_N=0.)
            obs = adapter.observations()
            for frame in range(1, limit + 1):
                action = policy.act_inference(obs)[0].numpy()
                adapter.step(action)
                data = adapter.data
                if adapter.frame != frame or abs(data.time - frame*dt) > 1e-6 or any(w.number for w in data.warning):
                    raise RuntimeError(f"Physics/frame error in {clip['motion']} frame {frame}")
                actual_force = reader.forces(data)
                for key, actual, original in (("qpos", data.qpos, qpos[frame-1]),
                                               ("qvel", data.qvel, qvel[frame-1]),
                                               ("contact_force_N", actual_force, forces[frame-1])):
                    error = float(np.max(np.abs(actual - original)))
                    max_errors[key] = max(max_errors[key], error)
                    if error > 1e-8:
                        raise RuntimeError(f"Fresh rollout differs from saved {key}: {error}, motion={clip['motion']}, frame={frame}")
                actual_pos = data.xpos[adapter.body_ids]
                for key, rule in rules.items():
                    if key in stops:
                        continue
                    values, flags = failure_signals(actual_pos, adapter.raw["body_pos_w"][frame],
                        data.xquat[anchor_ids[key]], adapter.raw["body_quat_w"][frame, anchors[key]], names, rule)
                    reasons = [reason for reason, flag in flags.items() if flag]
                    if reasons or frame == expected[key]["horizon_frames"]:
                        stop = {"last_frame": frame, "failure_reasons": reasons,
                                "status": "failure" if reasons else ("motion_end" if frame == len(adapter.raw["joint_pos"])-1 else "timeout")}
                        for field in stop:
                            if stop[field] != expected[key][field]:
                                raise RuntimeError(f"Online/offline stop mismatch: {clip['motion']} {key}: {stop} vs {expected[key]}")
                        stops[key] = stop
                if len(stops) == len(rules):
                    break
                obs = adapter.observations()
            if set(stops) != set(rules):
                raise RuntimeError(f"Missing online stop for {clip['motion']}")
            logs.append({"motion": clip["motion"], "physics_control_steps": frame, "stops": stops,
                         "max_absolute_replay_errors": max_errors})
            verification = {"passed": len(logs) == len(results["pmt_full"]), "verified_motions": len(logs),
                            "protocol_count": len(rules), "verified_protocol_motion_pairs": len(logs)*len(rules),
                            "wall_seconds": time.monotonic()-started,
                            "fresh_physics_control_steps": sum(c["physics_control_steps"] for c in logs),
                            "fresh_mj_step_calls": sum(c["physics_control_steps"] for c in logs)*contract["decimation"],
                            "method": "Fresh mj_step rollouts share the causal prefix across passive termination rules; each rule stops scoring at its own first event",
                            "clips": logs}
            write_json(output / "online_verification.json", verification)
            print(f"[ONLINE] {len(logs)}/40 {clip['motion']} stops match; max_qpos_error={max_errors['qpos']:.3g}", flush=True)
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pmt-env", type=Path, default=Path("logs/pgmt_stage1/pgmt_8192x20000/params/env.yaml"))
    parser.add_argument("--sonic-root", type=Path, default=Path("/home/lzp/projects/GR00T-WholeBodyControl"))
    parser.add_argument("--verify-online", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be empty")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    metadata = load_json(args.source / "metadata.json")
    baseline = load_json(args.source / "summary.json")
    audit = load_json(args.source / "verification.json")
    if not baseline["all_requested_full_clips_completed"] or not audit["passed"] or baseline["completed_clips"] != 40:
        raise ValueError("Expected independently verified complete 40-motion trajectories")
    for path, digest in metadata["sources_sha256"].items():
        if sha256(path) != digest:
            raise ValueError(f"Baseline source changed: {path}")
    if sha256(metadata["checkpoint"]) != metadata["checkpoint_sha256"] or sha256(metadata["contract"]) != metadata["contract_sha256"]:
        raise ValueError("Checkpoint/contract changed")
    rules = load_protocols(args.pmt_env, args.sonic_root)
    for name in [*rules, "termination_traces"]:
        (args.output / name).mkdir(parents=True, exist_ok=True)
    sources = [args.pmt_env, Path("pmt_tasks/mdp/terminations.py"), Path("pmt_tasks/tracking_terminations.py"), Path(__file__),
               args.sonic_root / "gear_sonic/config/manager_env/terminations/tracking/eval.yaml",
               args.sonic_root / "gear_sonic/envs/manager_env/mdp/terminations.py", Path(metadata["geometry"]["scene_xml"])]
    write_json(args.output / "metadata.json", {
        "source": str(args.source.resolve()), "checkpoint": metadata["checkpoint"],
        "checkpoint_sha256": metadata["checkpoint_sha256"], "protocols": rules,
        "source_hashes": {str(p.resolve()): sha256(p) for p in sources},
        "metric_definitions": metadata["metric_definitions"], "contact": metadata["contact"],
        "metric_scope": "Frames 1 through first failing frame inclusive, or horizon; initial frame excluded; no post-failure/reset frames",
        "comparison": "Same PMT model, initial state, nominal flat MuJoCo physics and full-source reference windows; only termination predicates/horizon differ",
        "reference_timing": "Both conditions and metrics compare each post-step state to the same-time reference frame; this is a MuJoCo predicate comparison, not a reproduction of Isaac manager update order",
        "pmt_full_horizon": "PMT native failure predicates; replace its 15-second timeout with full source end; no 4000-frame resets",
        "pmt_native_15s_horizon": "PMT native failure predicates and saved 15-second timeout; starts every motion at frame 0; horizon pass is not whole-motion success",
        "sonic_scope": "Explicit tracking/eval terms only, pelvis anchor; no inherited foot_pos_xyz term; local MuJoCo first-failure metric protocol",
        "simultaneous_failure_and_timeout": "Failure takes precedence for both; matches local SONIC MuJoCo, unlike Isaac callback timeout masking",
        "completion": "(first_failure_frame-1)/(source_frames-1) on failure; 1 for full success; timeout frame/(source_frames-1) for native timeout",
        "online_verification_requested": args.verify_online,
    })
    model = mujoco.MjModel.from_xml_path(metadata["geometry"]["scene_xml"])
    results = compare_saved(args.source, args.output, metadata, rules, model)
    if args.verify_online:
        verify_online(metadata, rules, results, model, args.output)
    print(json.dumps(load_json(args.output / "summary.json"), indent=2), flush=True)


if __name__ == "__main__":
    main()
