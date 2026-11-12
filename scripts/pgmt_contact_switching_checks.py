"""T5 oracle: replay raw physical force samples with independent NumPy history."""
from decimal import Decimal

import numpy as np
import torch

from pmt_tasks.pgmt_contact_switching import CONTACT_SWITCHING_WEIGHT


class ContactSwitchingAudit:
    def __init__(self, env, state):
        self.env, self.state = env, state
        self.term = env.reward_manager.active_terms.index("terrain_contact_switching")
        self.episode = np.full(env.num_envs, -1, dtype=np.int64)
        self.steps = np.zeros(env.num_envs, dtype=np.int64)
        self.previous = np.zeros((env.num_envs, 2), dtype=bool)
        self.previous_valid = self.previous.copy()
        self.age = np.zeros_like(self.previous, dtype=np.int64)
        self.dt = Decimal(str(env.step_dt))
        self.report = dict(settled_steps=0, triggered_feet=0, started=0, ended=0,
                           grace_excluded=0, slow_excluded=0, exact_boundary_excluded=0,
                           baseline_feet=0, holding_zero_cost_feet=0, nonterrain_triggered_feet=0,
                           unknown_reference_triggered_feet=0, reward_sum=0., max_rate_error=0., examples=[])

    def settled_step(self, total_reward):
        contact = self.state.contact_state(for_reward=True)
        sample = self.state.contact_switching_state(for_reward=True)
        for key in sample.keys() - {"valid", "per_foot", "rate"}:
            torch.testing.assert_close(sample[key], contact[key])
        forces = contact["net_forces_w"].cpu().numpy().astype(np.float64)
        cfg = self.env.cfg.pgmt_contact_events
        norm = np.linalg.norm(forces, axis=-1)
        valid = np.isfinite(forces).all(-1) & np.isfinite(norm)
        current = valid & (norm > cfg["contact_threshold_n"])
        episode = contact["episode_id"].cpu().numpy()
        reset = self.episode != episode
        self.previous_valid[reset] = False
        self.age[reset], self.steps[reset] = 0, 0
        self.steps += 1
        history = self.previous_valid & valid
        switched = history & (current != self.previous)
        # Decimal seconds give an independent exact-boundary oracle, without
        # reusing the production integer threshold or rounded snapshot ages.
        limit = Decimal(str(cfg["fast_switch_seconds"]))
        short = np.array([int(age) * self.dt < limit for age in self.age.flat]).reshape(self.age.shape)
        equal = np.array([int(age) * self.dt == limit for age in self.age.flat]).reshape(self.age.shape)
        grace = self.steps[:, None] <= cfg["switch_grace_steps"]
        expected = switched & short & ~grace
        np.testing.assert_array_equal(contact["contact"].cpu(), current)
        np.testing.assert_array_equal(sample["episode_step"].cpu(), self.steps)
        np.testing.assert_array_equal(sample["valid"].cpu(), history)
        np.testing.assert_array_equal(sample["contact_switched"].cpu(), switched)
        np.testing.assert_array_equal(sample["fast_contact_switch"].cpu(), expected)
        np.testing.assert_allclose(sample["previous_contact_age"].cpu(), np.where(history, self.age, 0) * float(self.dt), atol=1e-6)
        np.testing.assert_array_equal(sample["per_foot"].cpu(), expected.astype(float))
        rate = self.state.rewards()["terrain_contact_switching"]
        np.testing.assert_array_equal(rate.cpu(), expected.sum(-1))
        torch.testing.assert_close(rate, sample["rate"])
        actual = self.env.reward_manager._step_reward[:, self.term] * self.env.step_dt
        torch.testing.assert_close(actual, rate * CONTACT_SWITCHING_WEIGHT * self.env.step_dt)
        assert actual.le(0).all()
        self.age = np.where(history & ~switched, self.age + 1, 0)
        np.testing.assert_allclose(sample["contact_age"].cpu(), self.age * float(self.dt), atol=1e-6)
        self.previous, self.previous_valid, self.episode = current.copy(), valid.copy(), episode.copy()
        before = self.state.contact_state()["episode_step"].clone()
        self.state.rewards(); self.state.rewards()
        for key, value in sample.items():
            torch.testing.assert_close(self.state.contact_switching_state(for_reward=True)[key], value)
        torch.testing.assert_close(before, self.state.contact_state()["episode_step"])
        self.report["settled_steps"] += 1
        self.report["triggered_feet"] += int(expected.sum())
        self.report["started"] += int((expected & current).sum())
        self.report["ended"] += int((expected & ~current).sum())
        self.report["grace_excluded"] += int((switched & short & grace).sum())
        self.report["slow_excluded"] += int((switched & ~short & ~grace).sum())
        self.report["exact_boundary_excluded"] += int((switched & equal & ~grace).sum())
        self.report["baseline_feet"] += int((valid & ~history).sum())
        self.report["holding_zero_cost_feet"] += int((history & ~switched).sum())
        # Ended edges may follow an external collision but have no current force;
        # count external starts only, to avoid mislabelling every no-force edge.
        nonterrain = contact["terrain_forces_w"].abs().sum(-1).cpu().numpy() < 1e-6
        self.report["nonterrain_triggered_feet"] += int((expected & current & nonterrain).sum())
        self.report["unknown_reference_triggered_feet"] += int((
            expected & ~contact["reference_contact_valid"].cpu().numpy()).sum())
        self.report["reward_sum"] += float(actual.sum())
        self.report["max_rate_error"] = max(self.report["max_rate_error"], float(np.max(np.abs(rate.cpu().numpy()-expected.sum(-1)))))
        if expected.any() and len(self.report["examples"]) < 8:
            self.report["examples"].append(dict(step=self.env.common_step_counter,
                started=np.argwhere(expected & current).tolist(), ended=np.argwhere(expected & ~current).tolist(),
                previous_age=sample["previous_contact_age"].cpu().tolist(), rates=rate.cpu().tolist()))
