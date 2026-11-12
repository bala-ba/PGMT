#!/usr/bin/env python3
"""Validate that saved videos contain every recorded simulation state."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def check(directory):
    metadata = json.loads((directory / "metadata.json").read_text())
    result = json.loads((directory / "render_result.json").read_text())
    states = [json.loads(line) for line in (directory / "states.jsonl").read_text().splitlines()]
    if not states or len(states) != result["recorded_frames"]:
        raise AssertionError(f"State/frame count mismatch: {directory}")
    joints = np.asarray([state["joint_pos"] for state in states])
    if not np.isfinite(joints).all() or np.max(np.ptp(joints, axis=0)) < 1e-4:
        raise AssertionError(f"Non-finite or static robot trajectory: {directory}")
    if not result["finite"] or result["max_body_position_error_m"] > 0.001:
        raise AssertionError(f"Invalid pose mapping: {directory}")
    if "height_scan" in metadata:
        cells = int(np.prod(metadata["height_scan"]["shape_xy"]))
        if result.get("height_scan_frames") != len(states):
            raise AssertionError(f"Missing height scan frames: {directory}")
        policy_frames = 0
        for state in states:
            scan = state["height_scan"]
            hits = np.asarray(scan["hits_w"])
            if hits.shape != (cells, 3) or not np.isfinite(hits).all() or len(scan["hit_valid"]) != cells:
                raise AssertionError(f"Invalid recorded height scan: {directory}")
            if "policy_height" in scan:
                values = np.asarray(scan["policy_height"])
                if values.shape != (cells,) or not np.isfinite(values).all() or len(scan["policy_valid"]) != cells:
                    raise AssertionError(f"Invalid policy height observation: {directory}")
                policy_frames += 1
        if result.get("policy_height_scan_frames") != policy_frames:
            raise AssertionError(f"Policy height frame count mismatch: {directory}")
    path = directory / f'{metadata["mode"]}.mp4'
    with imageio.get_reader(str(path)) as reader:
        count = reader.count_frames()
        first = reader.get_data(0)
        last = reader.get_data(count - 1)
        fps = reader.get_meta_data()["fps"]
    if count != len(states) or abs(fps - 1 / metadata["dt"]) > 0.1:
        raise AssertionError(f"Video count/frame rate mismatch: {path}")
    if np.std(first[80:]) < 5 or np.mean(np.abs(first.astype(float) - last)) < 0.1:
        raise AssertionError(f"Blank or static video: {path}")
    return {"video": str(path), "frames": count, "fps": fps, "duration_seconds": count / fps,
            "resolution": list(first.shape[:2][::-1]), "max_body_position_error_m": result["max_body_position_error_m"],
            "passed": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = [check(path) for path in args.directories]
    report = json.dumps(results, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
    print(report)
