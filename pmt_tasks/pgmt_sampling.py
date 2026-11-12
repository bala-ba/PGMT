"""PGMT motion-level sampling conditioned on terrain compatibility and reset validity."""
import csv
from pathlib import Path
import re

import torch

from pmt_tasks.pgmt_terrain import FAMILIES, FAMILY_GROUPS


def load_terrain_compatibility(path, motion_files, *, families=FAMILIES, device="cpu"):
    """Read an exclusion table in loader motion order; blank cells allow all terrains.

    A semantic group such as ``stairs`` also excludes ``up_stairs``. An explicit
    variant such as ``up_stairs`` only excludes that variant. CSV supersets are
    allowed for inspecting a subset of the library; missing selected motions,
    duplicate names and unknown terrain labels fail instead of allowing silently.
    """
    names = [Path(file).stem for file in motion_files]
    if len(names) != len(set(names)):
        raise ValueError("PGMT compatibility requires unique motion file stems")
    if any(family not in FAMILY_GROUPS for family in families):
        raise ValueError(f"Unknown PGMT terrain families: {families}")
    excluded = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not {"motion_name", "dismatched terrain"}.issubset(reader.fieldnames):
            raise ValueError("Compatibility CSV requires motion_name and dismatched terrain columns")
        for line, row in enumerate(reader, 2):
            if None in row or row["dismatched terrain"] is None:
                raise ValueError(f"Malformed compatibility CSV row {line}; quote multi-terrain cells")
            name = row["motion_name"].strip()
            if name.endswith(".npz"):
                name = name[:-4]
            if not name or name in excluded:
                raise ValueError(f"Empty or duplicate motion_name on CSV row {line}: {name!r}")
            terrain_names = set(filter(None, re.split(r"[,;|\s]+", row["dismatched terrain"].strip())))
            unknown = terrain_names - set(FAMILIES)
            if unknown:
                raise ValueError(f"Unknown terrain labels on CSV row {line}: {sorted(unknown)}")
            excluded[name] = terrain_names
    missing = sorted(set(names) - excluded.keys())
    if missing:
        raise ValueError(f"Motions missing from compatibility CSV: {missing}")
    return torch.tensor([
        [family not in excluded[name] and FAMILY_GROUPS[family] not in excluded[name] for name in names]
        for family in families
    ], dtype=torch.bool, device=device)


def masked_motion_probabilities(p_fail, allowed, *, beta, uniform_ratio):
    """Normalize difficulty weights and the uniform component *within* each pool."""
    if beta < 0 or not 0 <= uniform_ratio <= 1:
        raise ValueError("Require beta >= 0 and uniform_ratio in [0, 1]")
    if allowed.ndim != 2 or allowed.shape[1] != len(p_fail):
        raise ValueError("Expected [terrain, motion] eligibility and [motion] failure probabilities")
    if not allowed.any(-1).all():
        raise ValueError("Cannot sample from an empty compatible motion pool")
    weights = p_fail.clamp_min(1e-6).pow(beta)[None] * allowed
    adaptive = weights / weights.sum(-1, keepdim=True)
    uniform = allowed.float() / allowed.sum(-1, keepdim=True)
    probabilities = (1 - uniform_ratio) * adaptive + uniform_ratio * uniform
    return probabilities / probabilities.sum(-1, keepdim=True)


def sample_motion_starts(p_fail, compatible, valid_starts, terrain_ids, *, beta, uniform_ratio,
                         fixed_start_frame=-1, families=FAMILIES):
    """Select motions adaptively, then use a fixed or uniformly drawn eligible frame.

    Keeping these two draws separate prevents clips with more valid support
    frames from gaining unintended motion-level probability. No rejection loop
    can reintroduce an incompatible motion. Failure statistics remain global
    per motion; terrain only conditions eligibility, not the statistic identity.
    """
    if not len(terrain_ids):
        return terrain_ids.clone(), terrain_ids.clone()
    if fixed_start_frame >= valid_starts.shape[1]:
        raise ValueError("Requested start frame is outside all loaded segments")
    available = (valid_starts[:, fixed_start_frame] if fixed_start_frame >= 0 else valid_starts.any(-1))
    groups, inverse = terrain_ids.unique(return_inverse=True)
    eligible = compatible[groups] & available[None]
    empty = ~eligible.any(-1)
    if empty.any():
        names = [families[i] for i in groups[empty].tolist()]
        raise ValueError(f"No compatible motion with a valid reset start for terrain(s): {names}")
    probabilities = masked_motion_probabilities(p_fail, eligible, beta=beta, uniform_ratio=uniform_ratio)
    motions = torch.empty_like(terrain_ids)
    for group in range(len(groups)):
        ids = (inverse == group).nonzero().flatten()
        motions[ids] = torch.multinomial(probabilities[group], len(ids), replacement=True)
    if fixed_start_frame >= 0:
        frames = torch.full_like(motions, fixed_start_frame)
    else:
        frames = torch.multinomial(valid_starts[motions].float(), 1).squeeze(-1)
    return motions, frames
