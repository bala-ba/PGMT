"""Terrain-aware deadzones for three lower-body tracking errors (reward rates).

The actual torso tile supplies the semantic family and integer level. Geometry
difficulty and birth curriculum assignments are deliberately not used as scale.
Configuration is resolved once; per-step evaluation stays on the tensor device.
"""
from copy import deepcopy
import math
from numbers import Real

import torch

from pmt_tasks.pgmt_terrain_layout import GROUP_IDS


RELAXATION_TERMS = {
    "lower_body_position": "position_max",
    "lower_body_orientation": "orientation_max",
    "lower_joint_position": "joint_max",
}
TRACKING_RELAXATION_CONFIG = {
    "enabled": False,
    "position_max": 0.20,
    "orientation_max": 0.70,
    "joint_max": 0.50,
    "terrain_families": ["stairs", "slope", "boxes"],
    "level_max": 9,
    "alpha": {term: {} for term in RELAXATION_TERMS},
}


def tracking_relaxation_config(config=None):
    """Return an owned, validated config; partial overrides are supported.

Maxima are metres/radians/radians. Alpha maps each reward term to lower-body
link/joint names; omitted names have alpha=1. Empty families or zero maxima
restore strict tracking. Unknown keys/names fail instead of silently disabling.
"""
    result = deepcopy(TRACKING_RELAXATION_CONFIG)
    if config is not None:
        if not isinstance(config, dict) or set(config) - set(result):
            raise ValueError("Unknown tracking relaxation configuration keys")
        result.update(deepcopy(config))
    if type(result["enabled"]) is not bool:
        raise ValueError("Tracking relaxation enabled must be boolean")
    for key in ("position_max", "orientation_max", "joint_max"):
        _nonnegative(result[key], key)
    if type(result["level_max"]) is not int or result["level_max"] <= 0:
        raise ValueError("Tracking relaxation level_max must be a positive integer")
    families = result["terrain_families"]
    if (not isinstance(families, (list, tuple)) or any(not isinstance(f, str) or f not in GROUP_IDS for f in families)
            or len(set(families)) != len(families)):
        raise ValueError("Tracking relaxation terrain_families must be unique semantic terrain names")
    result["terrain_families"] = list(families)
    alpha = result["alpha"]
    if not isinstance(alpha, dict) or set(alpha) - set(RELAXATION_TERMS):
        raise ValueError("Unknown tracking relaxation alpha term")
    for term in RELAXATION_TERMS:
        overrides = alpha.setdefault(term, {})
        if not isinstance(overrides, dict):
            raise ValueError("Tracking relaxation alpha requires name/value mappings")
        for name, value in overrides.items():
            _nonnegative(value, f"alpha.{term}.{name}")
    return result


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Tracking relaxation {name} must be finite and nonnegative")


class TrackingRelaxation:
    """Resolve names once, then compute [N,12] margins from located tiles."""

    def __init__(self, config, body_names, joint_names, groups, *, device, dtype=torch.float32):
        self.config = tracking_relaxation_config(config)
        expected = {f"{side}_{part}" for side in ("left", "right") for part in (
            "hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")}
        self.names, self.caps = {}, {}
        for term, maximum in RELAXATION_TERMS.items():
            joint = term == "lower_joint_position"
            names, ids = (joint_names, groups["lower_joints"]) if joint else (body_names, groups["lower"])
            selected = [names[i] for i in ids.tolist()]
            if len(selected) != 12 or set(selected) != {f"{name}_{'joint' if joint else 'link'}" for name in expected}:
                raise ValueError(f"Tracking relaxation requires the 12 named G1 lower elements: {term}")
            overrides = self.config["alpha"][term]
            if set(overrides) - set(selected):
                raise ValueError(f"Unknown lower element in tracking relaxation alpha: {term}")
            self.names[term] = selected
            values = [self.config[maximum] * overrides.get(name, 1.) for name in selected]
            self.caps[term] = torch.tensor(values, device=device, dtype=dtype)
            if not torch.isfinite(self.caps[term]).all():
                raise ValueError("Tracking relaxation effective caps must fit the reward dtype")
        # Index by semantic ID, independently of dict insertion / asset columns.
        self.allowed = torch.zeros(max(GROUP_IDS.values()) + 1, device=device, dtype=torch.bool)
        for name, index in GROUP_IDS.items():
            self.allowed[index] = name in self.config["terrain_families"]

    def contract(self):
        return dict(version=1, config=deepcopy(self.config), names=deepcopy(self.names),
                    anchor="torso_link", difficulty_source="current_tile_level",
                    family_source="current_family_group_index", mapping="clamp(level/level_max,0,1)",
                    error_rule="clamp(raw_error-margin,min=0)")

    def margins(self, tile):
        level, family = tile["tile_level"], tile["family_group_index"]
        valid = tile["tile_valid"] & (level >= 0) & (family >= 0) & (family < len(self.allowed))
        applicable = valid & self.allowed[family.clamp(0, len(self.allowed) - 1)] & self.config["enabled"]
        scale = (level.to(self.caps["lower_body_position"].dtype) / self.config["level_max"]).clamp(0, 1)
        scale = torch.where(applicable, scale, 0.)
        return {term: scale[:, None] * cap for term, cap in self.caps.items()}, scale, applicable


def relax_tracking_error(error, margin):
    """Subtract before squaring/averaging; preserve NaN/Inf evidence in errors."""
    if error.shape != margin.shape or error.device != margin.device:
        raise ValueError("Tracking relaxation margin must match the selected error shape/device")
    return (error - margin).clamp_min(0)
