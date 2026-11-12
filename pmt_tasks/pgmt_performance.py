"""Optional PGMT execution optimizations, independent of task/checkpoint semantics.

Installed on the one live environment, never on Isaac Lab classes or PMT tasks.
Each switch can be benchmarked independently. Logs retain owned terminal values.
"""
from copy import copy
from types import MethodType

import torch

FEATURES = ("reset_stats", "placement", "contact_sensor", "audit", "reward_compile")


def parse_features(value):
    if value in ("", "none"):
        return frozenset()
    if value == "all":
        return frozenset(FEATURES)
    result = frozenset(value.split(","))
    if not result <= set(FEATURES):
        raise ValueError(f"Unknown PGMT performance features: {sorted(result-set(FEATURES))}")
    return result


def tensor_command_reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    extras = {name: value[ids].mean() for name, value in self.metrics.items()}
    for value in self.metrics.values():
        value[ids] = 0
    self.command_counter[ids] = 0
    self._resample(ids)
    return extras


def tensor_curriculum_reset(self, env_ids=None):
    extras = {}
    for name, state in self._curriculum_state.items():
        if state is not None:
            if isinstance(state, dict):
                extras.update({f"Curriculum/{name}/{key}": value.clone() if isinstance(value, torch.Tensor) else value
                               for key, value in state.items()})
            else:
                extras[f"Curriculum/{name}"] = state.clone() if isinstance(state, torch.Tensor) else state
    for cfg in self._class_term_cfgs:
        cfg.func.reset(env_ids=env_ids)
    return extras


def tensor_termination_reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    stats = self._last_episode_dones.float().mean(0)
    extras = {f"Episode_Termination/{name}": stats[i] for i, name in enumerate(self._term_names)}
    for cfg in self._class_term_cfgs:
        cfg.func.reset(env_ids=ids)
    return extras


def tensor_sampler_update_metrics(self):
    import math
    total = self.failed_motion_count + self.success_motion_count
    failure = self.failed_motion_count / (total + 1e-8)
    weighted = failure.pow(self.beta)
    weighted = weighted / (weighted.sum() + 1e-8)
    probability = weighted * (1-self.uniform_ratio) + self.uniform_ratio/self.num_motions
    self._last_entropy = -(probability * (probability+1e-12).log()).sum() / math.log(self.num_motions)
    self._last_pfail_mean = failure.mean()
    self._last_top1_prob = probability.max()
    # These counts change only on EMA/load, the same lifecycle as this method.
    self._pgmt_total_failed = self.failed_motion_count.sum()
    self._pgmt_total_success = self.success_motion_count.sum()


def tensor_sampler_metrics(self):
    return dict(adaptive_entropy=self._last_entropy, adaptive_pfail_mean=self._last_pfail_mean,
                adaptive_top1_prob=self._last_top1_prob, total_failed=self._pgmt_total_failed,
                total_success=self._pgmt_total_success)


def scatter_sampler_outcomes(self, motion_ids, frame_ids, terminated):
    # Motion-only adaptation; no boolean compaction, scalar test or bincount shape query.
    motion_ids = motion_ids.to(device=self.device, dtype=torch.long)
    failed = terminated.to(device=self.device, dtype=self.current_failed_motion_count.dtype)
    self.current_failed_motion_count.scatter_add_(0, motion_ids, failed)
    self.current_success_motion_count.scatter_add_(0, motion_ids, 1-failed)


def tensor_command_metrics(self):
    from isaaclab.utils.math import quat_error_magnitude
    self.metrics["error_anchor_pos"] = (self.anchor_pos_w-self.robot_anchor_pos_w).norm(dim=-1)
    self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
    self.metrics["error_body_pos"] = (self.body_pos_relative_w-self.robot_body_pos_w).norm(dim=-1).mean(-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(-1)
    self.metrics["error_joint_pos"] = (self.joint_pos-self.robot_joint_pos).norm(dim=-1)
    for name, value in self.sampler.get_metrics().items():
        self.metrics[name] = value.expand(self.num_envs).clone()


def install_performance(env, features):
    """Activate only explicitly requested features on this environment."""
    features = parse_features(features) if isinstance(features, str) else frozenset(features)
    if not features:
        return dict(features=[], version=1)
    if getattr(env, "_pgmt_performance", None) is not None:
        raise RuntimeError("PGMT performance options already installed")
    if tuple(getattr(env.cfg, "pgmt_reward_groups", ())) != ("upper", "lower", "terrain", "auxiliary"):
        raise ValueError("Performance options require the four-head PGMT Stage2 terrain task")
    command = env.command_manager.get_term("motion")
    if command.cfg.enable_frame_sampling:
        raise ValueError("PGMT performance options require motion-only adaptive sampling")
    report = dict(version=1, features=sorted(features))
    if "reset_stats" in features:
        for owner, name, function in (
            (command, "reset", tensor_command_reset),
            (command, "_update_metrics", tensor_command_metrics),
            (env.curriculum_manager, "reset", tensor_curriculum_reset),
            (env.termination_manager, "reset", tensor_termination_reset),
            (command.sampler, "_update_metrics", tensor_sampler_update_metrics),
            (command.sampler, "get_metrics", tensor_sampler_metrics),
            (command.sampler, "update", scatter_sampler_outcomes),
        ):
            if owner is command.sampler and getattr(owner, "kind", None) == "pbfm_emd":
                continue  # EMD has no failure counters or EMA to optimize.
            setattr(owner, name, MethodType(function, owner))
        if getattr(command.sampler, "kind", None) != "pbfm_emd":
            command.sampler._update_metrics()
    if "placement" in features:
        from pmt_tasks.pgmt_fast_placement import install_fast_placement
        report["placement"] = install_fast_placement(command)
    if "contact_sensor" in features:
        sensor = env.scene["contact_forces"]
        report["contact_sensor"] = dict(before_history=sensor.cfg.history_length,
            before_track_air_time=sensor.cfg.track_air_time, history_length=0, track_air_time=False)
        # Do not mutate env.cfg or the checkpoint's semantic configuration.
        sensor.cfg = copy(sensor.cfg)
        sensor.cfg.history_length = 0
        sensor.cfg.track_air_time = False
        sensor._data.net_forces_w_history = sensor._data.net_forces_w.unsqueeze(1)
    if "reward_compile" in features:
        from pmt_tasks.pgmt_math import reward_terms
        # Only the stateless tensor kernel; PhysX, resets and terminal caches remain eager.
        env._pgmt_state._reward_kernel = torch.compile(reward_terms, fullgraph=True, dynamic=False)
        report["reward_compile"] = dict(backend="inductor", fullgraph=True, dynamic=False)
    env._pgmt_performance = report
    return report


class DeferredRewardAudit:
    """Check every reward sample on GPU, read aggregate diagnostics once per PPO."""
    def __init__(self, manager, groups, dt, num_envs, device):
        self.manager, self.groups, self.dt, self.num_envs = manager, tuple(groups), dt, num_envs
        self.indices = [[i for i, name in enumerate(manager.active_terms) if name.startswith(g+"_")]
                        for g in groups]
        self.counts = torch.zeros(4, device=device, dtype=torch.long)
        self.maximum_error = torch.zeros((), device=device)
        self.failed = torch.zeros((), device=device, dtype=torch.bool)
        self.steps = 0

    def add(self, result):
        _, reward, done, extras = result
        grouped = extras["reward_groups"]
        if grouped.shape != (self.num_envs, len(self.groups)):
            raise AssertionError("Incorrect reward group shape")
        total = grouped.sum(-1)
        error = (total-reward).abs()
        failed = ~torch.isfinite(grouped).all() | ~torch.isfinite(reward).all()
        failed |= (error > 2e-6+2e-5*reward.abs()).any()
        for i, indices in enumerate(self.indices):
            expected = self.manager._step_reward[:, indices].sum(-1)*self.dt
            # torch.testing.assert_close float32 defaults.
            failed |= (~torch.isclose(grouped[:, i], expected, rtol=1.3e-6, atol=1e-5)).any()
        self.failed |= failed
        self.maximum_error.copy_(torch.maximum(self.maximum_error, error.max()))
        self.counts += torch.stack((done.sum(), extras["time_outs"].sum(), grouped[:, 2].ne(0).sum(), failed.long()))
        self.steps += 1

    def read(self):
        values = torch.cat((self.counts.double(), self.maximum_error.double()[None], self.failed.double()[None])).cpu().tolist()
        if values[-1]:
            raise AssertionError(f"Reward audit failed in {int(values[3])} control steps")
        return dict(steps=self.steps, done_events=int(values[0]), timeout_events=int(values[1]),
                    terrain_nonzero_samples=int(values[2]), max_group_sum_error=values[4])
