"""Control-step foot contact/support state, independent of Isaac and rewards.

Contact uses the all-object normal-force norm. Support uses only the terrain
normal-force world Z component. A snapshot is a copy: readers cannot advance
timers or corrupt another reward's events. Reset never samples stale forces.
"""
from __future__ import annotations

import math
import json
from pathlib import Path
import torch


CONTACT_STATE_CONFIG = {
    "contact_threshold_n": 10.0,
    "support_threshold_n": 10.0,
    "touchdown_min_non_support_steps": 2,
    "fast_switch_seconds": 0.1,
    "switch_grace_steps": 5,
}
TERRAIN_FOOT_SENSORS = ("pgmt_left_foot_terrain", "pgmt_right_foot_terrain")


def terrain_contact_filters(asset_directory, terrain_prim_path):
    """Explicit global colliders: PhysX treats wildcards as per-env pairing.

    A shared terrain object must be a literal filter, not a pattern expanding
    to all regions/patches. P-BFM auxiliary colliders are included when listed.
    Runtime compares this list against the imported USD's actual colliders.
    """
    manifest = json.loads((Path(asset_directory) / "manifest.json").read_text())
    root = f"{terrain_prim_path}/terrain/Terrain"
    if "patches" in manifest:
        paths = [f"{root}/{p['region_name']}/P{p['patch_row']}{p['patch_col']}/Mesh" for p in manifest["patches"]]
    else:
        paths = [f"{root}/{t['name']}/Mesh" for t in manifest["tiles"]]
    for surface in manifest.get("auxiliary_surfaces", []):
        path = surface["prim_path"]
        if not path.startswith("/World/Terrain/"):
            raise ValueError("Auxiliary colliders must belong to /World/Terrain")
        paths.append(f"{terrain_prim_path}/terrain{path.removeprefix('/World')}/Mesh")
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("PGMT terrain manifest must name distinct contact colliders")
    return sorted(paths)


def validate_contact_config(config):
    if set(config) != set(CONTACT_STATE_CONFIG):
        raise ValueError("PGMT contact-state configuration keys do not match the contract")
    for name in ("contact_threshold_n", "support_threshold_n", "fast_switch_seconds"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("touchdown_min_non_support_steps", "switch_grace_steps"):
        if type(config[name]) is not int or config[name] < (1 if name.startswith("touchdown") else 0):
            raise ValueError(f"Invalid integer {name}")


def terrain_force_from_matrix(matrix, num_envs):
    """Reduce one foot's filtered normal forces; never substitute net forces."""
    if matrix is None or matrix.ndim != 4 or matrix.shape[:2] != (num_envs, 1) or matrix.shape[-1] != 3:
        raise ValueError("Expected a terrain-filtered force_matrix_w [N,1,filters,3] per foot")
    if matrix.shape[2] == 0:
        raise ValueError("Terrain contact filter resolved no force columns")
    # NaN/Inf is deliberately retained so the state machine reports unknown.
    return matrix[:, 0].sum(dim=1)


class ContactEventState:
    def __init__(self, num_envs, *, device, dt, config=None):
        if num_envs < 1 or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Positive environment count and finite control dt required")
        self.config = dict(CONTACT_STATE_CONFIG if config is None else config)
        validate_contact_config(self.config)
        self.num_envs, self.device, self.dt = num_envs, torch.device(device), float(dt)
        self._last_step = None
        shape = (num_envs, 2)
        self._sample_step = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.device = self._sample_step.device
        self._episode_step = torch.zeros_like(self._sample_step)
        self._episode_id = torch.zeros_like(self._sample_step)
        self._streams = {}
        for name in ("contact", "support"):
            self._streams[name] = {
                "valid": torch.zeros(shape, dtype=torch.bool, device=self.device),
                "state": torch.zeros(shape, dtype=torch.bool, device=self.device),
                **{key: torch.zeros(shape, dtype=torch.long, device=self.device)
                   for key in ("age", "on_steps", "off_steps", "last_on_steps", "last_off_steps")},
            }
        self._data = self._empty_snapshot()

    def _empty_snapshot(self):
        shape = (self.num_envs, 2)
        result = {key: torch.full((*shape, 3), float("nan"), device=self.device)
                  for key in ("net_forces_w", "terrain_forces_w")}
        result.update({key: torch.full(shape, float("nan"), device=self.device)
                       for key in ("force_norm", "upward_force")})
        for name, off in (("contact", "air"), ("support", "non_support")):
            for key in (name, f"previous_{name}", f"{name}_valid", f"{name}_history_valid",
                        f"{name}_started", f"{name}_ended", f"{name}_switched"):
                result[key] = torch.zeros(shape, dtype=torch.bool, device=self.device)
            for key in (f"{name}_age", f"previous_{name}_age", f"{name}_time", f"{off}_time",
                        f"last_{name}_time", f"last_{off}_time"):
                result[key] = torch.zeros(shape, device=self.device)
            result[f"previous_{off}_steps"] = torch.zeros(shape, dtype=torch.long, device=self.device)
        for key in ("touchdown", "fast_contact_switch"):
            result[key] = torch.zeros(shape, dtype=torch.bool, device=self.device)
        return result

    def reset(self, env_ids):
        """Invalidate selected live rows; same-step update remains a no-op."""
        fast = (getattr(self, '_fast_reset_buffers', False) and isinstance(env_ids, torch.Tensor)
                and env_ids.dtype == torch.long and env_ids.ndim == 1)
        def clear(tensor, value):
            if fast:
                tensor.index_fill_(0, env_ids, value)
            else:
                tensor[env_ids] = value
        for stream in self._streams.values():
            for tensor in stream.values():
                clear(tensor, 0)
        for key, tensor in self._data.items():
            clear(tensor, float("nan") if key in (
                "net_forces_w", "terrain_forces_w", "force_norm", "upward_force") else 0)
        clear(self._sample_step, -1)
        clear(self._episode_step, 0)
        self._episode_id[env_ids] += 1

    def _advance_stream(self, name, off, current, valid, data):
        old = self._streams[name]
        history = old["valid"] & valid
        previous = old["state"] & history
        switched = history & (current != previous)
        started, ended = switched & current, switched & ~current
        previous_age = torch.where(history, old["age"], 0)
        previous_off = torch.where(history, old["off_steps"], 0)
        age = torch.where(history & ~switched, old["age"] + 1, 0)
        on_steps = torch.where(valid & current, torch.where(history, old["on_steps"], 0) + 1, 0)
        off_steps = torch.where(valid & ~current, previous_off + 1, 0)
        last_on = torch.where(history, torch.where(ended, old["on_steps"], old["last_on_steps"]), 0)
        last_off = torch.where(history, torch.where(started, old["off_steps"], old["last_off_steps"]), 0)
        data.update({name: current, f"previous_{name}": previous,
                     f"{name}_valid": valid, f"{name}_history_valid": history,
                     f"{name}_started": started, f"{name}_ended": ended, f"{name}_switched": switched,
                     f"{name}_age": age * self.dt, f"previous_{name}_age": previous_age * self.dt,
                     f"{name}_time": on_steps * self.dt, f"{off}_time": off_steps * self.dt,
                     f"last_{name}_time": last_on * self.dt, f"last_{off}_time": last_off * self.dt,
                     f"previous_{off}_steps": previous_off})
        # Integer comparison keeps exactly 0.1 s outside the strict < boundary.
        if name == "contact":
            limit = math.ceil(self.config["fast_switch_seconds"] / self.dt - 1e-9)
            data["fast_contact_switch"] = switched & (previous_age < limit) & (
                self._episode_step[:, None] > self.config["switch_grace_steps"])
        else:
            data["touchdown"] = started & (previous_off >= self.config["touchdown_min_non_support_steps"])
        self._streams[name] = dict(valid=valid, state=current, age=age, on_steps=on_steps,
                                   off_steps=off_steps, last_on_steps=last_on, last_off_steps=last_off)

    @torch.no_grad()
    def update(self, step, net_forces_w, terrain_forces_w):
        """Advance exactly once per consecutive control step; return whether advanced.

        Invalid forces break only that foot/channel's history. False with valid
        False means unknown, never confirmed air/non-support. The next finite
        reading establishes a baseline without manufacturing an edge.
        """
        if type(step) is not int or step < 0:
            raise ValueError("Control step must be a nonnegative integer")
        if step == self._last_step:
            return False
        if self._last_step is not None and step != self._last_step + 1:
            raise ValueError("Contact state must be updated on consecutive control steps")
        for forces in (net_forces_w, terrain_forces_w):
            if forces.shape != (self.num_envs, 2, 3) or forces.device != self.device or not forces.is_floating_point():
                raise ValueError("Expected floating [N,2,3] forces on the state device")
        net, terrain = net_forces_w.detach().clone(), terrain_forces_w.detach().clone()
        norm = net.norm(dim=-1)
        contact_valid = torch.isfinite(net).all(-1) & torch.isfinite(norm)
        support_valid = torch.isfinite(terrain).all(-1)
        data = dict(net_forces_w=net, terrain_forces_w=terrain, force_norm=norm, upward_force=terrain[..., 2])
        self._episode_step += 1
        self._advance_stream("contact", "air", contact_valid & (norm > self.config["contact_threshold_n"]),
                             contact_valid, data)
        self._advance_stream("support", "non_support", support_valid & (terrain[..., 2] > self.config["support_threshold_n"]),
                             support_valid, data)
        self._sample_step.fill_(step)
        self._data, self._last_step = data, step
        return True

    def snapshot(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        data = dict(self._data, sample_step=self._sample_step, episode_step=self._episode_step,
                    episode_id=self._episode_id)
        return {key: value[ids].clone() for key, value in data.items()}
