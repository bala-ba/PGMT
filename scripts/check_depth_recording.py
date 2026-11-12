"""Offline depth-map validation; terrain truth is used ONLY by this checker."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from pmt_tasks.depth_perception import CameraIntrinsics, LocalElevationMap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    args = parser.parse_args()
    folder = args.recording
    metadata = json.loads((folder / "metadata.json").read_text())
    states = {row["step"]: row for row in map(json.loads, (folder / "states.jsonl").read_text().splitlines())}
    model = mujoco.MjModel.from_xml_path(metadata["geometry"]["scene_xml"])
    data = mujoco.MjData(model)
    # Terrain is static; no robot geometry is included in this diagnostic query.
    mujoco.mj_forward(model, data)
    groups = np.array([0, 0, 1, 0, 0, 0], dtype=np.uint8)
    geom_id = np.zeros(1, dtype=np.int32)
    errors = []
    replay_error = 0.
    replay_masks_equal = True
    frames_checked = 0
    mapper = LocalElevationMap(max_age_s=metadata["camera"]["max_age_s"])
    with np.load(folder / "depth_frames.npz", allow_pickle=False) as source:
        k = source["intrinsics"]
        intrinsics = CameraIntrinsics(int(k[0]), int(k[1]), *map(float, k[2:]))
        depth, poses, stamps = source["depth_m"], source["world_from_camera"], source["timestamps"]
        torso, yaw = source["torso_position"], source["torso_yaw"]
        for i, timestamp in enumerate(stamps):
            mapper.integrate(depth[i], intrinsics, poses[i], timestamp, torso[i, :2])
            scan = mapper.sample(torso[i], yaw[i], timestamp)
            step = round(timestamp / metadata["dt"])
            if step not in states:
                continue  # camera initial frame precedes the first saved control state
            recorded = states[step]["height_scan"]
            replay_masks_equal &= scan["policy_valid"] == recorded["policy_valid"]
            replay_error = max(replay_error, float(np.abs(np.asarray(scan["policy_height"]) - recorded["policy_height"]).max()))
            frames_checked += 1
            valid = np.asarray(scan["policy_valid"])
            for hit in np.asarray(scan["hits_w"])[valid]:
                distance = mujoco.mj_ray(model, data, np.r_[hit[:2], 3.], np.array([0., 0., -1.]), groups, True, -1, geom_id)
                if distance >= 0:
                    errors.append(abs(hit[2] - (3. - distance)))
    result = {"camera_frames": len(stamps), "replayed_frames_compared": frames_checked,
              "policy_mask_replay_exact": bool(replay_masks_equal), "max_policy_height_replay_error_m": replay_error,
              "map_height_samples_compared": len(errors), "mean_map_height_error_m": float(np.mean(errors)),
              "p95_map_height_error_m": float(np.percentile(errors, 95)), "max_map_height_error_m": float(np.max(errors)),
              "fraction_height_error_over_5cm": float(np.mean(np.asarray(errors) > .05)),
              "note": "Offline truth comparison includes cells straddling terrain edges; truth never feeds perception or the actor.",
              "replay_passed": bool(replay_masks_equal and replay_error < 1e-5)}
    (folder / "depth_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["replay_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
