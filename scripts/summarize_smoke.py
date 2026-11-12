#!/usr/bin/env python3
"""Export real TensorBoard training curves and smoke artifacts to PNG/JSON."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def summarize(root: Path, *, require_smoke_suite=False):
    results = {"root": str(root.resolve()), "training": {}, "playback": {}}
    for directory in sorted({p.parent for p in root.rglob("events.out.tfevents.*")}):
        events = EventAccumulator(str(directory), size_guidance={"scalars": 0}).Reload()
        scalars = {tag: events.Scalars(tag) for tag in events.Tags()["scalars"]}
        if not scalars:
            continue
        if not all(math.isfinite(event.value) for rows in scalars.values() for event in rows):
            raise FloatingPointError(f"Non-finite training metrics: {directory}")
        # Token trackers also export model_<n>_pma.pt; those are adapter exports,
        # not full runner checkpoints.
        checkpoints = sorted(
            (p for p in directory.glob("model_*.pt") if re.fullmatch(r"model_\d+\.pt", p.name)),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        run = {
            "scalar_tags": len(scalars),
            "last_iteration": max(rows[-1].step for tag, rows in scalars.items() if tag.startswith("Loss/")),
            "finite_scalars": True,
            "checkpoints": [str(p.relative_to(root)) for p in checkpoints],
            "last_scalars": {tag: rows[-1].value for tag, rows in scalars.items()},
        }
        if checkpoints:
            last = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)["model_state_dict"]
            if not all(torch.isfinite(v).all().item() for v in last.values() if isinstance(v, torch.Tensor)):
                raise FloatingPointError(f"Non-finite checkpoint: {checkpoints[-1]}")
            run["finite_checkpoint"] = True
            if len(checkpoints) > 1:
                first = torch.load(checkpoints[0], map_location="cpu", weights_only=False)["model_state_dict"]
                changed = [k for k, v in last.items() if k.endswith("weight") and k in first and not torch.equal(first[k], v)]
                if not changed:
                    raise AssertionError(f"No weight updates between checkpoints in {directory}")
                run["changed_weight_tensors"] = len(changed)
        figure, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
        groups = [
            ("Episode reward", ["Train/mean_reward"]),
            ("Training losses", [t for t in scalars if t.startswith("Loss/") and any(k in t for k in ["value_function", "surrogate", "behavior", "distillation"])]),
            ("Throughput (steps/s)", ["Perf/total_fps"]),
        ]
        for axis, (title, tags) in zip(axes, groups):
            for tag in tags:
                if tag in scalars:
                    axis.plot([e.step for e in scalars[tag]], [e.value for e in scalars[tag]], marker=".", label=tag.split("/")[-1])
            axis.set_title(title)
            axis.set_xlabel("Training iteration")
            axis.grid(alpha=0.25)
            if axis.lines:
                axis.legend(fontsize=8)
        figure.suptitle(f"{directory.name} — smoke run (not convergence)")
        figure.savefig(directory / "training_curves.png", dpi=140)
        plt.close(figure)
        results["training"][str(directory.relative_to(root))] = run
    for path in sorted(root.rglob("*.json")):
        if path.name == "summary.json":
            continue
        try:
            data = json.loads(path.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(data, dict) and "finite" in data and "steps" in data:
            if not data["finite"] or data["steps"] <= 0:
                raise AssertionError(f"Failed playback: {path}")
            results["playback"][str(path.relative_to(root))] = data
    if require_smoke_suite:
        expected_training = {"flat_train": 4, "perceptive_train": 1, "distill_train": 1, "finetune_train": 1}
        expected_playback = {"flat_trained_play.json": 200, "flat_pretrained_play.json": 300,
                             "perceptive_trained_play.json": 200, "distill_trained_play.json": 100,
                             "finetune_trained_play.json": 100}
        for name, last_iteration in expected_training.items():
            run = results["training"].get(name, {})
            if not (run.get("finite_checkpoint") and run.get("changed_weight_tensors", 0) > 0
                    and run.get("last_iteration") == last_iteration):
                raise AssertionError(f"Missing or incomplete training stage: {name}")
        for name, steps in expected_playback.items():
            if results["playback"].get(name, {}).get("steps") != steps:
                raise AssertionError(f"Missing or incomplete playback stage: {name}")
        results["smoke_suite_passed"] = True
    (root / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"training_runs": len(results["training"]), "playback_runs": len(results["playback"]), "summary": str(root / "summary.json")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--require-smoke-suite", action="store_true")
    args = parser.parse_args()
    summarize(args.root, require_smoke_suite=args.require_smoke_suite)
