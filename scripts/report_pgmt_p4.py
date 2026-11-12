"""Independently verify P4 finite runs and checkpoint restoration on disk."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from report_pgmt_four_head import equal


def main(root):
    names = ("flat_160", "short_160", "resume_160", "restore_160", "scale_1024", "resume_1024")
    states = {name: json.loads((root/name/"training_status.json").read_text()) for name in names}
    checkpoints = {name: {phase: torch.load(root/name/f"{phase}.pt", weights_only=False, map_location="cpu")
                         for phase in ("initial", "final")} for name in names}
    for name, state in states.items():
        assert state["status"] == "completed" and state["source_checkpoint_unchanged"], name
        assert state["policy_value_groups"] == ["upper", "lower", "terrain", "auxiliary"]
        assert len(state["active_rewards"]) == 27
        assert all(p["after"] == 0 and p["filter_collisions"] and p["replicate_physics"] for p in state["physics_setup"])
        checkpoint = checkpoints[name]["final"]
        assert checkpoint["training_contract"]["runtime"]["physx_env_id_bits"] == 0
        assert checkpoint["model_state_dict"]["critic.6.weight"].shape == (4, 128)
        rows = [json.loads(line) for line in (root/name/"progress.jsonl").read_text().splitlines()] if state["completed_updates"] else []
        assert len(rows) == state["completed_updates"]
        for row in rows:
            assert all(math.isfinite(v) for v in row["losses"].values())
            assert all(math.isfinite(v) for v in row["telemetry"].values())
            assert sum(key.startswith("RewardStep/") for key in row["telemetry"]) == 28
            head_loss = [row["losses"][f"value_function_{g}"] for g in state["policy_value_groups"]]
            assert math.isclose(sum(head_loss)/4, row["losses"]["value_function"], rel_tol=2e-6, abs_tol=1e-6)
        if rows:
            assert all(v > 0 for v in rows[-1]["value_head_change"].values())
            assert all(v > 0 for v in rows[-1]["value_head_gradient_l1"].values())
    fields = ("model_state_dict", "optimizer_state_dict", "policy_metadata", "training_contract",
              "environment_state_dict", "learning_rate", "next_iteration", "iter", "initialization",
              "tot_timesteps", "tot_time")
    for source, destination in (("short_160", "resume_160"), ("resume_160", "restore_160"), ("scale_1024", "resume_1024")):
        for field in fields:
            equal(checkpoints[source]["final"][field], checkpoints[destination]["initial"][field])
    for field in fields:
        equal(checkpoints["restore_160"]["initial"][field], checkpoints["restore_160"]["final"][field])
    for name in ("flat_160", "short_160", "scale_1024"):
        initial = checkpoints[name]["initial"]
        migration = initial["initialization"]
        source = Path(migration["source"])
        assert hashlib.sha256(source.read_bytes()).hexdigest() == migration["source_sha256"]
        old = torch.load(source, weights_only=False, map_location="cpu")
        for key, value in old["model_state_dict"].items():
            equal(initial["model_state_dict"][key][[0, 1, 3]] if key in migration["remapped"]
                  else initial["model_state_dict"][key], value)
        assert not initial["optimizer_state_dict"]["state"]
        assert initial["environment_state_dict"]["curriculum"]["counts"].sum() == 0
    assert states["short_160"]["completed_updates"] == 25
    assert states["resume_160"]["completed_updates"] == 5
    assert states["restore_160"]["completed_updates"] == 0
    assert states["scale_1024"]["completed_updates"] == 3
    assert states["resume_1024"]["completed_updates"] == 1
    assert states["resume_1024"]["stop_reason"] == "walltime_limit"
    result = dict(status="passed", restored_fields_exact=list(fields), runs={name:dict(
        num_envs=s["arguments"]["num_envs"], updates=s["completed_updates"],
        next_iteration=s["next_iteration"], mean_collection_s=s["mean_collection_s"], mean_update_s=s["mean_update_s"],
        reward_audit=s["reward_group_audit"], curriculum_outcomes=s.get("curriculum_outcomes"),
        checkpoint=str(root/name/"final.pt")) for name,s in states.items()},
        scope="Numerical/finite training/restore and 1024 scalability; no convergence or 30-second benchmark claim")
    (root/"finite_acceptance.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root.resolve())
