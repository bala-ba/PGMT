"""Deterministic environment allocation for an explicitly weighted terrain mix."""
import math

import torch


def terrain_proportions(families, proportions):
    """Validate semantic names before converting a mapping to column order."""
    if not families or len(set(families)) != len(families):
        raise ValueError("Terrain families must be nonempty and unique")
    if set(proportions) != set(families):
        raise ValueError("Terrain proportions must contain exactly the asset families")
    values = [float(proportions[name]) for name in families]
    if any(not math.isfinite(p) or p < 0 for p in values) or not math.isclose(sum(values), 1., abs_tol=1e-8):
        raise ValueError("Terrain proportions must be finite, nonnegative and sum to one")
    return values


def allocate_terrain_types(num_envs, families, proportions, *, device):
    """Largest-remainder allocation; each count differs from N*p by less than one.

    Ties follow the declared family order. Types remain fixed through curriculum
    promotions/demotions, so the requested mix also holds for rollout samples.
    """
    if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 0:
        raise ValueError("num_envs must be a nonnegative integer")
    expected = [num_envs * p for p in terrain_proportions(families, proportions)]
    counts = [math.floor(n) for n in expected]
    order = sorted(range(len(families)), key=lambda i: (-(expected[i] - counts[i]), i))
    for i in order[:num_envs - sum(counts)]:
        counts[i] += 1
    return torch.repeat_interleave(torch.arange(len(families), device=device),
                                   torch.tensor(counts, device=device))
