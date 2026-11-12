"""T4: count feet with excessive horizontal all-object contact force.

The nonnegative, unweighted rate is integrated by RewardManager with -20
and the control dt. This is a continuous condition, not a rising-edge event.
"""
import math

import torch


STUMBLE_WEIGHT = -20.0
STUMBLE_CONFIG = dict(vertical_ratio=5.0, bias_n=10.0)


def validate_stumble_config(config):
    if not isinstance(config, dict) or set(config) != set(STUMBLE_CONFIG):
        raise ValueError("Stumble config requires exactly vertical_ratio and bias_n")
    for key, value in config.items():
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"Stumble {key} must be finite and nonnegative")


def empty_stumble(num_envs, *, device):
    return dict(
        forces_w=torch.full((num_envs, 2, 3), float("nan"), device=device),
        force_valid=torch.zeros(num_envs, 2, dtype=torch.bool, device=device),
        horizontal_force=torch.full((num_envs, 2), float("nan"), device=device),
        vertical_force_abs=torch.full((num_envs, 2), float("nan"), device=device),
        threshold=torch.full((num_envs, 2), float("nan"), device=device),
        valid=torch.zeros(num_envs, 2, dtype=torch.bool, device=device),
        stumble=torch.zeros(num_envs, 2, dtype=torch.bool, device=device),
        per_foot=torch.zeros(num_envs, 2, device=device),
        rate=torch.zeros(num_envs, device=device),
    )


@torch.no_grad()
def evaluate_stumble(forces_w, force_valid, config=None):
    """[N,2,3] world forces and [N,2] measurement validity -> [N] cost.

    force_valid is measurement validity, never the contact/support state.
    Unknown or overflowing force measurements are diagnosed as invalid and
    isolated to zero cost, rather than interpreted as a healthy foot.
    """
    config = STUMBLE_CONFIG if config is None else config
    validate_stumble_config(config)
    if (forces_w.ndim != 3 or forces_w.shape[1:] != (2, 3)
            or force_valid.shape != forces_w.shape[:2] or force_valid.dtype != torch.bool
            or not forces_w.is_floating_point() or forces_w.device != force_valid.device):
        raise ValueError("Stumble requires floating [N,2,3] forces and boolean [N,2] validity on one device")
    valid = force_valid & torch.isfinite(forces_w).all(-1)
    safe = torch.where(valid[..., None], forces_w, 0.)
    horizontal = torch.linalg.vector_norm(safe[..., :2], dim=-1)
    vertical = safe[..., 2].abs()
    threshold = config["vertical_ratio"] * vertical + config["bias_n"]
    valid &= torch.isfinite(horizontal) & torch.isfinite(threshold)
    stumble = valid & (horizontal > threshold)
    per_foot = stumble.to(forces_w.dtype)
    return dict(forces_w=forces_w.clone(), force_valid=force_valid.clone(),
                horizontal_force=torch.where(valid, horizontal, float("nan")),
                vertical_force_abs=torch.where(valid, vertical, float("nan")),
                threshold=torch.where(valid, threshold, float("nan")),
                valid=valid, stumble=stumble, per_foot=per_foot, rate=per_foot.sum(-1))
