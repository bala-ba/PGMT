"""T1 support quality from clean terrain, retaining the touchdown reward API.

Returns an unweighted rate. Only RewardManager multiplies weight and control dt.
Settlement requires an exact, valid two-foot reference/support match. The
legacy evaluate_touchdown helper remains available for geometry-only previews.
"""
import math

import torch

from pmt_tasks.pgmt_height_scan import SCAN_SIZE
from pmt_tasks.pgmt_math import inverse_rotate, rotate, yaw_quat


TOUCHDOWN_QUALITY_CONFIG = dict(length_m=.20, width_m=.10, center_x_m=.039, sigma_m=.03,
                                settlement="support_pattern_mean_v2")
TOUCHDOWN_QUALITY_WEIGHT = 10.


def validate_touchdown_config(config):
    if set(config) != set(TOUCHDOWN_QUALITY_CONFIG):
        raise ValueError("Touchdown quality configuration keys do not match the contract")
    if config["settlement"] != TOUCHDOWN_QUALITY_CONFIG["settlement"]:
        raise ValueError("T1 requires support_pattern_mean_v2 settlement")
    for key, value in config.items():
        if key == "settlement":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Touchdown {key} must be a finite number")
        if key != "center_x_m" and value <= 0:
            raise ValueError(f"Touchdown {key} must be positive")


def validate_touchdown_scanner(cfg):
    """The clean geometric mask matches PGMT's torso/yaw 2 m scan.

    Blind evaluation may omit the scanner: its geometric footprint is retained.
    Reject changed scanner geometry instead of silently applying a wrong mask.
    """
    if cfg is None:
        return
    if (not cfg.prim_path.endswith("/torso_link") or cfg.ray_alignment != "yaw"
            or tuple(cfg.pattern_cfg.size) != tuple(SCAN_SIZE)
            or tuple(cfg.offset.pos[:2]) != (0., 0.) or tuple(cfg.offset.rot) != (1., 0., 0., 0.)
            or tuple(cfg.drift_range) != (0., 0.)
            or any(tuple(v) != (0., 0.) for v in cfg.ray_cast_drift_range.values())):
        raise ValueError("Touchdown coverage requires the unshifted PGMT torso/yaw 2 m scan")


def footprint_points(positions, quaternions, config):
    """[K,3]/[K,4] -> [K,9,2], local offset rotates with each foot yaw."""
    x = positions.new_tensor([-.5, 0., .5]) * config["length_m"] + config["center_x_m"]
    y = positions.new_tensor([-.5, 0., .5]) * config["width_m"]
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    local = torch.stack((xx, yy, torch.zeros_like(xx)), -1).reshape(1, 9, 3).expand(len(positions), -1, -1)
    yaw = yaw_quat(quaternions)[:, None].expand(-1, 9, -1)
    return (positions[:, None] + rotate(yaw, local))[..., :2]


def height_quality(heights, valid, sigma_m):
    """Population variance; every one of the nine points must be valid.

    Whole invalid rows are replaced before arithmetic, never NaN multiplied by
    zero. Invalid variance is explicitly unknown (NaN), while reward stays zero.
    """
    if heights.shape != valid.shape or heights.shape[-1] != 9 or valid.dtype != torch.bool:
        raise ValueError("Touchdown requires paired [...,9] heights and boolean validity")
    valid_foot = (valid & torch.isfinite(heights)).all(-1)
    safe = torch.where(valid_foot[..., None], heights, 0.)
    variance = safe.var(dim=-1, unbiased=False)
    valid_foot &= torch.isfinite(variance)
    quality = torch.where(valid_foot, torch.exp(-variance / sigma_m**2), 0.)
    return quality, torch.where(valid_foot, variance, float("nan")), valid_foot


def empty_touchdown(num_envs, *, device, dtype=torch.float32):
    return dict(
        event=torch.zeros(num_envs, 2, device=device, dtype=torch.bool),
        eligible=torch.zeros(num_envs, 2, device=device, dtype=torch.bool),
        pattern_valid=torch.zeros(num_envs, device=device, dtype=torch.bool),
        pattern_matched=torch.zeros(num_envs, device=device, dtype=torch.bool),
        reference_support_count=torch.zeros(num_envs, device=device, dtype=torch.long),
        query_xy_w=torch.full((num_envs, 2, 9, 2), float("nan"), device=device, dtype=dtype),
        height_w=torch.full((num_envs, 2, 9), float("nan"), device=device, dtype=dtype),
        map_valid=torch.zeros(num_envs, 2, 9, device=device, dtype=torch.bool),
        scan_valid=torch.zeros(num_envs, 2, 9, device=device, dtype=torch.bool),
        point_valid=torch.zeros(num_envs, 2, 9, device=device, dtype=torch.bool),
        valid=torch.zeros(num_envs, 2, device=device, dtype=torch.bool),
        variance=torch.full((num_envs, 2), float("nan"), device=device, dtype=dtype),
        quality=torch.zeros(num_envs, 2, device=device, dtype=dtype),
        rate=torch.zeros(num_envs, device=device, dtype=dtype),
    )


@torch.no_grad()
def evaluate_touchdown(events, foot_pos, foot_quat, torso_pos, torso_quat, query, config):
    """Legacy geometry helper: query the selected feet, at most 18 points/env.

    query receives [K*9,2] actual world XY, and returns height/valid of that
    location. Scan coverage is geometric, not the policy's random dropout mask.
    Its raw quality sum is not the continuous T1 reward. Settlement must use
    evaluate_support_quality with actual support and same-frame reference data.
    """
    n = len(events)
    if (events.shape != (n, 2) or events.dtype != torch.bool or foot_pos.shape != (n, 2, 3)
            or foot_quat.shape != (n, 2, 4) or torso_pos.shape != (n, 3) or torso_quat.shape != (n, 4)):
        raise ValueError("Touchdown expects two feet and one torso pose per environment")
    result = empty_touchdown(n, device=foot_pos.device, dtype=foot_pos.dtype)
    result["event"] = events.clone()
    result["eligible"] = events.clone()
    foot_norm, torso_norm = foot_quat.norm(dim=-1), torso_quat.norm(dim=-1)
    foot_ok = torch.isfinite(foot_pos).all(-1) & torch.isfinite(foot_quat).all(-1)
    foot_ok &= torch.isfinite(foot_norm) & (foot_norm > 1e-6)
    torso_ok = torch.isfinite(torso_pos).all(-1) & torch.isfinite(torso_quat).all(-1)
    torso_ok &= torch.isfinite(torso_norm) & (torso_norm > 1e-6)
    env_ids, feet = (events & foot_ok & torso_ok[:, None]).nonzero(as_tuple=True)
    if not len(env_ids):
        return result  # No selected foot: no query and no stale quality.
    pos = foot_pos[env_ids, feet]
    quat = foot_quat[env_ids, feet] / foot_norm[env_ids, feet, None]
    xy = footprint_points(pos, quat, config)
    relative = torch.cat((xy - torso_pos[env_ids, None, :2], torch.zeros_like(xy[..., :1])), -1)
    torso_yaw = yaw_quat(torso_quat[env_ids] / torso_norm[env_ids, None])[:, None].expand(-1, 9, -1)
    local = inverse_rotate(torso_yaw, relative)[..., :2]
    scan_valid = (local.abs() <= local.new_tensor(SCAN_SIZE) / 2).all(-1)
    terrain = query(xy.reshape(-1, 2))
    height, map_valid = terrain["height"].reshape(-1, 9), terrain["valid"].reshape(-1, 9)
    if height.shape != xy.shape[:2] or map_valid.shape != height.shape or map_valid.dtype != torch.bool:
        raise ValueError("Terrain query returned an incompatible height/validity shape")
    point_valid = map_valid & scan_valid & torch.isfinite(height)
    quality, variance, valid = height_quality(height, point_valid, config["sigma_m"])
    for key, value in dict(query_xy_w=xy, height_w=height, map_valid=map_valid, scan_valid=scan_valid,
                           point_valid=point_valid, valid=valid, variance=variance, quality=quality).items():
        result[key][env_ids, feet] = value
    result["rate"] = result["quality"].sum(-1)
    return result


@torch.no_grad()
def evaluate_support_quality(support, support_valid, reference_contact, reference_valid,
                             foot_pos, foot_quat, torso_pos, torso_quat, query, config, *, touchdown):
    """Continuous T1: exact two-foot support pattern, then mean stance quality.

    Both labels AND both physical support measurements must be valid. An extra
    support foot or a missing required foot vetoes the entire environment.
    Matched flight earns zero. Invalid geometry contributes zero without
    shrinking the reference-only denominator. The original touchdown event is
    preserved for diagnostics; it never gates or adds a reward bonus.
    """
    inputs = (support, support_valid, reference_contact, reference_valid, touchdown)
    if (support.ndim != 2 or support.shape[1] != 2
            or any(v.shape != support.shape or v.device != support.device for v in inputs)
            or any(v.dtype != torch.bool for v in (support, support_valid, reference_valid, touchdown))):
        raise ValueError("T1 requires paired [N,2] support/labels and boolean masks on one device")
    ref_valid = reference_valid & ((reference_contact == 0) | (reference_contact == 1))
    ref = reference_contact == 1
    pattern_valid = (support_valid & ref_valid).all(-1)
    matched = pattern_valid & (support == ref).all(-1)
    count = (ref & ref_valid).sum(-1)
    eligible = matched[:, None] & ref
    result = evaluate_touchdown(eligible, foot_pos, foot_quat, torso_pos, torso_quat, query, config)
    result.update(event=touchdown.clone(), pattern_valid=pattern_valid,
                  pattern_matched=matched, reference_support_count=count)
    result["rate"] = result["quality"].sum(-1) / count.clamp_min(1)
    return result
