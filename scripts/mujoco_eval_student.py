"""Evaluate a PMT vision student or PGMT stage-1 contract with CPU MuJoCo."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import time

# Select the checkout containing this entry point, even if another PMT is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from pmt_tasks.backends.mujoco_standalone import create_adapter, build_model, load_policy, validate_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path("logs/sim2sim/contract.pt"))
    parser.add_argument("--output", type=Path, default=Path("logs/sim2sim/student"))
    motion = parser.add_mutually_exclusive_group()
    motion.add_argument("--motion-index", type=int, default=0)
    motion.add_argument("--motion-file", type=Path, help="PGMT motion override after contract parity validation.")
    parser.add_argument("--checkpoint", type=Path, help="Override checkpoint location; weights must match the contract fixtures.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-height", action="store_true", help="Diagnostic run without height noise/dropout.")
    parser.add_argument("--perception", choices=("raycast", "depth"), default="raycast")
    parser.add_argument("--depth-width", type=int, default=160)
    parser.add_argument("--depth-height", type=int, default=120)
    parser.add_argument("--depth-fovy", type=float, default=70.)
    parser.add_argument("--depth-pitch", type=float, default=55., help="Downward pitch from head forward; simulation parameter, not hardware calibration.")
    parser.add_argument("--depth-hz", type=float, default=25.)
    parser.add_argument("--map-max-age", type=float, default=2.)
    parser.add_argument("--record-depth", action="store_true", help="Save camera/map video and portable depth/pose NPZ.")
    args = parser.parse_args()
    if args.steps < 1 or args.start_frame < 0:
        parser.error("steps must be positive and start-frame nonnegative")
    if args.record_depth and args.perception != "depth":
        parser.error("--record-depth requires --perception depth")
    if args.perception == "depth" and (args.clean_height or min(args.depth_width, args.depth_height) < 3
            or not np.isfinite([args.depth_hz, args.map_max_age, args.depth_fovy, args.depth_pitch]).all()
            or args.depth_hz <= 0 or args.map_max_age <= 0 or not 0 < args.depth_fovy < 180 or not 0 <= args.depth_pitch <= 90):
        parser.error("Invalid camera/map parameters; --clean-height applies only to raycast")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "metrics.json").exists():
        parser.error("Output already has metrics.json; use a new directory.")
    torch.set_num_threads(2)
    contract = torch.load(args.contract, map_location="cpu", weights_only=False)
    if args.checkpoint is not None:
        contract["checkpoint"] = str(args.checkpoint.resolve())
    pgmt = contract.get("policy_class") == "PGMTActorCritic"
    if pgmt and args.perception == "depth":
        parser.error("PGMT stage 1 has no depth observations.")
    if args.motion_file is not None and not pgmt:
        parser.error("--motion-file requires a PGMT contract; vision uses paired clips.")
    dt = contract["sim_dt"] * contract["decimation"]
    if args.perception == "depth" and not np.isclose(round(1 / (dt * args.depth_hz)), 1 / (dt * args.depth_hz)):
        parser.error("Depth rate must divide the control rate (e.g. 25, 10 or 5 Hz at 50 Hz control)")
    camera_config = {"width": args.depth_width, "height": args.depth_height, "fovy_deg": args.depth_fovy,
                     "pitch_deg": args.depth_pitch, "hz": args.depth_hz, "max_age_s": args.map_max_age,
                     "mount_body": "head_link", "mount_position_m": [.08, 0., .48],
                     "mount_reference": "assets/G1_structure.png and head_link.STL; approximate, not calibrated"}
    model, geometry = build_model(contract, args.output / "model", camera_config if args.perception == "depth" else None)
    adapter = create_adapter(contract, model, args.seed, noisy_height=not args.clean_height)
    policy = load_policy(contract)
    parity = validate_contract(contract, adapter, policy)
    (args.output / "parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    if not parity["passed"]:
        raise RuntimeError(f"Isaac/MuJoCo observation parity failed: {parity}")
    adapter.rng = np.random.default_rng(args.seed)
    perception = None
    if args.perception == "depth":
        from pmt_tasks.backends.mujoco_depth import MujocoDepthPerception
        perception = MujocoDepthPerception(model, camera_config, args.output, args.record_depth)
        adapter.perception = perception
        adapter.scan_config["source"] = "Rendered depth camera -> rolling elevation map -> actual policy observation"
    if args.motion_file is not None:
        contract["motion_files"] = list(contract["motion_files"]) + [str(args.motion_file.resolve())]
        args.motion_index = len(contract["motion_files"]) - 1
    adapter.reset(args.motion_index, args.start_frame)
    count = (adapter.reference_end if pgmt else len(adapter.raw["joint_pos"])) - args.start_frame
    steps = min(args.steps, count - 1)
    if steps < 1:
        raise ValueError("No motion frames remain at the requested start frame.")
    dt = contract["sim_dt"] * contract["decimation"]
    motion_files = contract["motion_files"] if pgmt else contract["raw_files"]
    metadata = {"task": contract.get("task", "PMT-PPOft-Bigmap") + " | standalone MuJoCo", "mode": "testing",
                "output": str(args.output.resolve()), "checkpoint": contract["checkpoint"],
                "checkpoint_sha256": hashlib.sha256(Path(contract["checkpoint"]).read_bytes()).hexdigest(),
                "joint_names": contract["joint_names"], "body_names": contract["body_names"],
                "terrain_mesh": contract["terrain_mesh"], "physics": "MuJoCo / mj_step (CPU)",
                "renderer": "MuJoCo", "dt": dt, "video": False, "display": False, "max_frames": steps,
                "height_scan": adapter.scan_config, "geometry": geometry,
                "raw_motion": motion_files[args.motion_index],
                "optimized_motion": motion_files[args.motion_index] if pgmt else contract["optimized_files"][args.motion_index],
                "initialization": "reference pose and velocities, no reset noise",
                "height_noise": not pgmt and not args.clean_height if perception is None else False, "seed": args.seed,
                "perception": "none" if pgmt else args.perception, "camera": camera_config if perception else None,
                "action_contract": contract.get("action_contract", "q_target = raw_q_ref + policy_action, scale=1"),
                "contact_friction": [1., .005, .0001], "sim_dt": contract["sim_dt"],
                "decimation": contract["decimation"]}
    if pgmt:
        metadata.pop("height_scan", None)
        metadata["reference_chunk"] = [adapter.reference_start, adapter.reference_end]
        metadata["initial_previous_target"] = "exported initial_joint_target" if "initial_joint_target" in contract else "exported q_default"
        metadata["rollout_boundary"] = "stops at native motion chunk end; no automatic reset"
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    body_errors, joint_errors, heights, contacts, torques = [], [], [], [], []
    first_fall = None
    height_loss_steps = max_height_loss_steps = 0
    max_reference_height_loss = 0.
    start = time.monotonic()
    with perception if perception is not None else nullcontext(), (args.output / "states.jsonl").open("w") as states, torch.inference_mode():
        obs = adapter.observations()
        for step in range(steps):
            action = policy.act_inference(obs)[0].numpy()
            adapter.step(action)
            # Current state observation is consumed on the next control step.
            obs = adapter.observations()
            d = adapter.data
            body = d.xpos[adapter.body_ids]
            target = adapter.optimized["body_pos_w"][adapter.frame]
            error = np.linalg.norm(body[adapter.tracked] - target[adapter.tracked], axis=-1).mean()
            body_errors.append(float(error))
            joint_errors.append(float(np.sqrt(np.mean((d.qpos[adapter.qids] - adapter.optimized["joint_pos"][adapter.frame]) ** 2))))
            heights.append(float(d.qpos[2]))
            contacts.append(int(d.ncon))
            torques.append(float(np.abs(d.actuator_force).max()))
            reference_height_loss = float(target[0, 2] - d.qpos[2])
            max_reference_height_loss = max(max_reference_height_loss, reference_height_loss)
            height_loss_steps = height_loss_steps + 1 if reference_height_loss > .2 else 0
            max_height_loss_steps = max(max_height_loss_steps, height_loss_steps)
            pelvis_up = d.xmat[model.body("pelvis").id].reshape(3, 3)[2, 2]
            if first_fall is None and (d.qpos[2] < .30 or pelvis_up < .2):
                first_fall = {"step": step + 1, "time_s": (step + 1) * dt,
                              "pelvis_z_m": float(d.qpos[2]), "pelvis_up_z": float(pelvis_up)}
            if abs(d.time - (step + 1) * dt) > 1e-6:
                raise RuntimeError("MuJoCo simulation time jumped (possible numerical autoreset).")
            state = {"step": step + 1, "motion_frame": adapter.frame,
                     "root_pos": d.qpos[:3].tolist(), "root_quat": d.qpos[3:7].tolist(),
                     "joint_pos": d.qpos[adapter.qids].tolist(), "body_pos": body.tolist(),
                     "height_scan": adapter.last_scan, "action": action.tolist(),
                     "joint_target": d.ctrl.tolist(), "actuator_force": d.actuator_force.tolist(),
                     "tracking_body_error_m": float(error), "contacts": int(d.ncon),
                     "reference_pelvis_height_loss_m": reference_height_loss}
            states.write(json.dumps(state, allow_nan=False) + "\n")
            if (step + 1) % 250 == 0:
                print(f"[MUJOCO] {step + 1}/{steps} steps, t={d.time:.2f}s, body error={error:.3f}m, fall={first_fall is not None}", flush=True)
    warnings = {str(i): int(w.number) for i, w in enumerate(adapter.data.warning) if w.number}
    result = {"physics": metadata["physics"], "checkpoint": metadata["checkpoint"],
              "motion_index": args.motion_index, "steps": steps, "mj_step_calls": adapter.physics_steps,
              "simulated_seconds": float(adapter.data.time), "wall_seconds": time.monotonic() - start,
              "finite": True, "resets": 0, "first_fall": first_fall,
              "mean_body_error_m": float(np.mean(body_errors)), "max_body_error_m": float(np.max(body_errors)),
              "mean_joint_rmse_rad": float(np.mean(joint_errors)), "min_pelvis_height_m": min(heights),
              "steps_with_contacts": sum(c > 0 for c in contacts), "max_abs_torque_nm": max(torques),
              "mujoco_warnings": warnings, "isaac_imported": any(k.startswith(("isaaclab", "omni.")) for k in sys.modules),
              "interface_parity_passed": parity["passed"],
              "tracking_passed": first_fall is None and max(body_errors) < .5 and not warnings,
              "tracking_criterion": "no fall (pelvis z < 0.30m or up-z < 0.2), mean tracked-body world error < 0.5m at every step, no solver warnings"}
    result["max_reference_pelvis_height_loss_m"] = max_reference_height_loss
    result["longest_reference_height_loss_seconds"] = max_height_loss_steps * dt
    result["reference_height_loss_threshold_m"] = .2
    result["runtime_passed"] = not warnings and result["finite"] and result["resets"] == 0
    result["tracking_criterion_note"] = ("Legacy absolute-height diagnostic can flag intentional LAFAN1 floor motions; compare the reference-relative height-loss metrics. Training terminations are unchanged." if pgmt else "Legacy path-deviation diagnostic only; not a terrain-adaptation success criterion.")
    if perception is not None:
        result["depth_perception"] = perception.metrics()
    if result["isaac_imported"]:
        raise RuntimeError("Standalone evaluator unexpectedly imported Isaac.")
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
