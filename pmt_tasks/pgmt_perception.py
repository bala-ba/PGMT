"""PMT-compatible policy observations; clean PGMT geometry stays separate."""
import torch

from motion_tracking_rl.pgmt_perception_contract import (
    SCAN_RESOLUTION, SCAN_SHAPE, SCAN_SIZE,
    terrain_observation_contract, validate_terrain_observation_contract,
)


def terrain_observation_from_hits(hits_w, sensor_z, map_valid, config):
    """Same processing order as PMT height_scan_for_vision, plus map bounds.

    Each invocation samples fresh dropout/noise for all environments. There is
    no step/reset cache. Call once for a height/mask pair; store that realized
    pair in rollout, never re-run augmentation during PPO minibatch updates.
    """
    validate_terrain_observation_contract(config)
    count = SCAN_SHAPE[0] * SCAN_SHAPE[1]
    if hits_w.ndim != 3 or hits_w.shape[1:] != (count, 3):
        raise ValueError(f"PGMT perception requires [N,{count},3] hits")
    if sensor_z.shape != hits_w.shape[:1] or map_valid.shape != hits_w.shape[:2]:
        raise ValueError("PGMT perception pose/map mask shape mismatch")
    # Convert out-of-map hits to missing returns before the native PMT pipeline.
    hit_z = hits_w[..., 2].masked_fill(~map_valid.bool(), float("nan"))
    if config["augmentation_enabled"] and config["nan_dropout_prob"] > 0:
        hit_z = hit_z.masked_fill(torch.rand_like(hit_z) < config["nan_dropout_prob"], torch.nan)
    valid = torch.isfinite(hit_z)
    fill = config["invalid_world_z_fill_m"]
    hit_z = torch.nan_to_num(hit_z, nan=fill, posinf=fill, neginf=fill)
    fallback = config["invalid_sensor_z_fill_m"]
    sensor_z = torch.nan_to_num(sensor_z[:, None], nan=fallback, posinf=fallback, neginf=fallback)
    height = sensor_z - hit_z - config["offset_m"]
    lo, hi = config["noise_range_m"]
    if config["augmentation_enabled"] and (lo != 0.0 or hi != 0.0):
        height = height + torch.empty_like(height).uniform_(lo, hi)
    height = height.clamp(*config["clipping_m"])
    # Keep PMT's filled invalid values (usually -3 m), not zero placeholders.
    return dict(height=height.reshape(-1, *SCAN_SHAPE), valid=valid.reshape(-1, *SCAN_SHAPE))


def pack_terrain_observation(pair):
    """One atomic ObservationManager term, so height and mask share dropout."""
    return torch.cat((pair["height"].flatten(1), pair["valid"].float().flatten(1)), dim=-1)


def split_terrain_observation(observations):
    """Unpack without sampling; preserve the training map/mask interface."""
    if "pgmt_terrain" not in observations:
        return observations
    result = dict(observations)
    packed = result.pop("pgmt_terrain")
    count = SCAN_SHAPE[0] * SCAN_SHAPE[1]
    if packed.ndim != 2 or packed.shape[1] != 2 * count:
        raise ValueError(f"PGMT packed perception requires [N,{2 * count}]")
    if "terrain_map" in result or "terrain_valid" in result:
        raise ValueError("PGMT packed and split perception cannot coexist")
    result["terrain_map"] = packed[:, :count].reshape(-1, *SCAN_SHAPE)
    result["terrain_valid"] = packed[:, count:].reshape(-1, *SCAN_SHAPE)
    return result
