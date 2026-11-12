"""Verify saved native four-head runs, independently of live runner state."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


def equal(a, b):
    if isinstance(b, torch.Tensor):
        torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
    elif isinstance(b, dict):
        assert a.keys() == b.keys()
        for key in b:
            equal(a[key], b[key])
    elif isinstance(b, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def main(root, diagnostics_run=None):
    names = ["stage1_160_01", "resume_160_01", "restore_160_01", "p2_160_01"]
    if diagnostics_run:
        names.append(diagnostics_run)
    status = {name: json.loads((root / name / "training_status.json").read_text()) for name in names}
    checkpoints = {name: {phase: torch.load(root / name / f"{phase}.pt", map_location="cpu", weights_only=False)
                         for phase in ("initial", "final")} for name in names}
    for name in names:
        s, checkpoint = status[name], checkpoints[name]["final"]
        assert s["status"] == "completed" and s["source_checkpoint_unchanged"]
        assert s["policy_value_groups"] == ["upper", "lower", "terrain", "auxiliary"]
        assert len(s["active_rewards"]) == 27 and s["tracking_relaxation"]["enabled"]
        assert checkpoint["kind"] == "pgmt_terrain_training"
        assert checkpoint["policy_metadata"]["signature"]["num_value_heads"] == 4
        assert checkpoint["model_state_dict"]["critic.6.weight"].shape == (4, 128)
        assert checkpoint["model_state_dict"]["critic.6.bias"].shape == (4,)
        assert "tracking_relaxation" in checkpoint["environment_state_dict"]["curriculum"]["identity"]
    migrations = {}
    for name in (names[0], names[3]):
        initial = checkpoints[name]["initial"]
        report = initial["initialization"]
        source_path = Path(report["source"])
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == report["source_sha256"]
        old = torch.load(source_path, map_location="cpu", weights_only=False)
        weights = initial["model_state_dict"]
        for key, value in old["model_state_dict"].items():
            equal(weights[key][[0, 1, 3]] if key in report["remapped"] else weights[key], value)
        assert weights["critic.6.weight"][2].eq(0).all() and weights["critic.6.bias"][2] == 0
        assert initial["iter"] == initial["next_iteration"] == 0
        assert not initial["optimizer_state_dict"]["state"]
        assert initial["environment_state_dict"]["curriculum"]["counts"].sum() == 0
        migrations[name] = dict(source_family=report["source_family"], exact_copied_tensors=len(report["copied"]),
            remapped_tensors=report["remapped"], new_tensors=len(report["initialized"]), terrain_initial_value=0.,
            inference_max_error=status[name]["migration_max_error"])
    restored_fields = ("model_state_dict", "optimizer_state_dict", "policy_metadata", "training_contract",
                       "environment_state_dict", "learning_rate", "next_iteration", "iter", "initialization",
                       "tot_timesteps", "tot_time")
    restore_pairs = [(names[0], names[1]), (names[1], names[2])]
    if diagnostics_run:
        restore_pairs.append((names[2], diagnostics_run))
    for source_name, restored_name in restore_pairs:
        old, new = checkpoints[source_name]["final"], checkpoints[restored_name]["initial"]
        for field in restored_fields:
            equal(old[field], new[field])
        assert all(status[restored_name]["strict_restore"][key] for key in (
            "model", "normalizers", "optimizer", "curriculum", "sampler", "sampling_probabilities", "learning_rate"))
    for field in restored_fields:
        equal(checkpoints[names[2]]["initial"][field], checkpoints[names[2]]["final"][field])
    assert [checkpoints[name]["final"]["next_iteration"] for name in names[:4]] == [3, 5, 5, 0]
    if diagnostics_run:
        assert checkpoints[diagnostics_run]["final"]["next_iteration"] == 6
    training_names = names[:2] + ([diagnostics_run] if diagnostics_run else [])
    value_loss_checks = {}
    for name in training_names:
        final = checkpoints[name]["final"]
        assert final["model_state_dict"]["critic.6.weight"][2].abs().sum() > 0
        latest = status[name]["latest"]
        assert all(value > 0 for value in latest["value_head_change"].values())
        assert all(value > 0 for value in latest["value_head_gradient_l1"].values())
        assert status[name]["reward_group_audit"]["terrain_nonzero_samples"] > 0
        losses = latest["losses"]
        keys = [f"value_function_{group}" for group in status[name]["policy_value_groups"]]
        # Earlier runs predate per-head diagnostics. The optional final update
        # validates that addition without repeating the whole simulator audit.
        if not diagnostics_run or name == diagnostics_run or all(key in losses for key in keys):
            values = [losses[key] for key in keys]
            assert all(math.isfinite(value) and value >= 0 for value in values)
            assert math.isclose(sum(values) / 4, losses["value_function"], rel_tol=2e-6, abs_tol=1e-6)
            value_loss_checks[name] = {key: losses[key] for key in ["value_function", *keys]}
    report = dict(status="passed", task="pgmt_stage2_terrain", num_envs=160, migrations=migrations,
        completed_iterations=[3, 5, 5] + ([6] if diagnostics_run else []),
        repeated_native_restore_exact=list(restored_fields), value_loss_checks=value_loss_checks,
        rng="CPU/CUDA unit test: restored before a fresh physical reset; no trajectory-replay claim",
        active_reward_count=27, value_groups=["upper", "lower", "terrain", "auxiliary"],
        reward_audits={name: status[name]["reward_group_audit"] for name in training_names},
        checkpoints={name: str(root / name / "final.pt") for name in names},
        scope="Finite training/integrity acceptance; terrain completion performance and termination delays remain separate")
    (root / "acceptance_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--diagnostics-run", help="Optional one-update resume from restore_160_01")
    args = parser.parse_args()
    main(args.root.resolve(), args.diagnostics_run)
