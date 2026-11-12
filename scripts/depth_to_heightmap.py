"""Replay rectified metric depth + synchronized poses into PMT vision inputs.

This offline interface accepts real sensor recordings as well as MuJoCo output.
The NPZ schema and live Python API are documented in docs/DEPTH_PERCEPTION.md.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from pmt_tasks.depth_perception import CameraIntrinsics, LocalElevationMap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-age", type=float, default=2.)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as source:
        k = source["intrinsics"]
        intrinsics = CameraIntrinsics(int(k[0]), int(k[1]), *map(float, k[2:]))
        required = ("depth_m", "world_from_camera", "timestamps", "torso_position", "torso_yaw")
        lengths = [len(source[key]) for key in required]
        if len(set(lengths)) != 1 or not lengths[0]:
            raise ValueError("All frame arrays must have the same nonzero length")
        mapper = LocalElevationMap(max_age_s=args.max_age)
        vision = []
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as out:
            for i, timestamp in enumerate(source["timestamps"]):
                torso = source["torso_position"][i]
                mapper.integrate(source["depth_m"][i], intrinsics, source["world_from_camera"][i], timestamp, torso[:2])
                scan = mapper.sample(torso, source["torso_yaw"][i], timestamp)
                row = np.r_[scan["policy_height"], scan["policy_valid"]].astype(np.float32)
                vision.append(row)
                out.write(json.dumps({"timestamp": float(timestamp), "height_scan": scan, "vision": row.tolist()}, allow_nan=False) + "\n")
        print(json.dumps({"frames": len(vision), "vision_shape": [len(vision), 374],
                          "finite": bool(np.isfinite(vision).all()), "output": str(args.output)}))


if __name__ == "__main__":
    main()
