"""Audit whether the loaded PMT actor's actions depend on its height-map input."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from tensordict import TensorDict

from pmt_tasks.backends.mujoco_standalone import load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path("logs/sim2sim/contract.pt"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    contract = torch.load(args.contract, weights_only=False, map_location="cpu")
    policy = load_policy(contract)
    changes = []
    latent_changes = []
    generator = torch.Generator().manual_seed(42)
    with torch.inference_mode():
        for fixture in contract["fixtures"]:
            obs = TensorDict({key: value.clone() for key, value in fixture["obs"].items()}, batch_size=[1])
            original = policy.act_inference(obs).clone()
            latent = policy._compute_actor_context(obs, store_aux_outputs=False)[-1].clone()
            altered = obs.clone()
            altered["vision"][:, :187] = torch.rand((1, 187), generator=generator) * 2 - .5
            altered["vision"][:, 187:] = 1.
            changes.append(float((policy.act_inference(altered) - original).abs().max()))
            latent_changes.append(float((policy._compute_actor_context(altered, store_aux_outputs=False)[-1] - latent).abs().max()))
    state = policy.state_dict()
    gates = {k: float(v.abs().max()) for k, v in state.items() if k.endswith("gate.alpha")}
    report = {"checkpoint": contract["checkpoint"],
              "sha256": hashlib.sha256(Path(contract["checkpoint"]).read_bytes()).hexdigest(),
              "fixtures": len(changes), "max_action_change_rad": max(changes),
              "max_vision_latent_change": max(latent_changes), "gate_max_abs": gates,
              "vision_affects_action_in_test": max(changes) > 1e-7,
              "interpretation": "Zero intent/residual gates structurally block vision from actor actions; a finite rollout does not validate learned terrain perception." if gates and max(gates.values()) == 0 else "Sensitivity test only; not a hardware performance guarantee."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
