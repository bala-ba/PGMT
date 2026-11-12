"""T6 oracle: double-precision per-foot squared force and native integration."""
import numpy as np
import torch

from pmt_tasks.pgmt_contact_force import CONTACT_FORCE_WEIGHT
from pmt_tasks.pgmt_runtime import _tensor


class ContactForceAudit:
    def __init__(self, env, state):
        self.env, self.state = env, state
        self.term = env.reward_manager.active_terms.index("terrain_contact_force")
        sensor = env.scene["contact_forces"]
        self.sensor_indices = [sensor.body_names.index(name) for name in state.foot_names]
        self.report = dict(settled_steps=0, positive_force_feet=0, zero_force_feet=0,
                           below_contact_threshold_feet=0, above_350n_feet=0,
                           nonterrain_force_feet=0, unknown_reference_force_feet=0,
                           reward_sum=0., max_rate_error=0., max_relative_rate_error=0., max_force_n=0.)

    def settled_step(self, total_reward):
        contact = self.state.contact_state(for_reward=True)
        sample = self.state.contact_force_state(for_reward=True)
        for key in ("sample_step", "episode_step", "episode_id", "reference_motion_ids", "reference_frame_ids"):
            torch.testing.assert_close(sample[key], contact[key])
        torch.testing.assert_close(sample["forces_w"], contact["net_forces_w"])
        keep = self.state.contact_state()["sample_step"] == contact["sample_step"]
        raw = _tensor(self.env.scene["contact_forces"].data.net_forces_w)
        torch.testing.assert_close(sample["forces_w"][keep], raw[keep][:, self.sensor_indices])
        forces = sample["forces_w"].cpu().numpy().astype(np.float64)
        assert np.isfinite(forces).all() and sample["valid"].all()
        expected_foot = np.sum(forces**2, axis=-1)
        expected = expected_foot.sum(-1)
        np.testing.assert_allclose(sample["force_sq"].cpu(), expected_foot, rtol=2e-6, atol=1e-5)
        np.testing.assert_allclose(sample["per_foot"].cpu(), expected_foot, rtol=2e-6, atol=1e-5)
        rate = self.state.rewards()["terrain_contact_force"]
        np.testing.assert_allclose(rate.cpu(), expected, rtol=2e-6, atol=1e-5)
        torch.testing.assert_close(rate, sample["rate"])
        actual = self.env.reward_manager._step_reward[:, self.term] * self.env.step_dt
        torch.testing.assert_close(actual, rate * CONTACT_FORCE_WEIGHT * self.env.step_dt)
        assert actual.le(0).all()
        before = self.state.contact_state()["episode_step"].clone()
        self.state.rewards(); self.state.rewards()
        for key, value in sample.items():
            torch.testing.assert_close(self.state.contact_force_state(for_reward=True)[key], value)
        torch.testing.assert_close(before, self.state.contact_state()["episode_step"])
        positive = expected_foot > 0.
        nonterrain = contact["terrain_forces_w"].abs().sum(-1).cpu().numpy() < 1e-6
        self.report["settled_steps"] += 1
        self.report["positive_force_feet"] += int(positive.sum())
        self.report["zero_force_feet"] += int((~positive).sum())
        self.report["below_contact_threshold_feet"] += int((positive & (
            expected_foot <= self.env.cfg.pgmt_contact_events["contact_threshold_n"]**2)).sum())
        self.report["above_350n_feet"] += int((expected_foot > 350.**2).sum())
        self.report["nonterrain_force_feet"] += int((positive & nonterrain).sum())
        self.report["unknown_reference_force_feet"] += int((positive & ~contact["reference_contact_valid"].cpu().numpy()).sum())
        self.report["reward_sum"] += float(actual.sum())
        error = np.abs(rate.cpu().numpy() - expected)
        self.report["max_rate_error"] = max(self.report["max_rate_error"], float(error.max()))
        self.report["max_relative_rate_error"] = max(self.report["max_relative_rate_error"],
            float(np.max(error / np.maximum(expected, np.finfo(float).tiny))))
        self.report["max_force_n"] = max(self.report["max_force_n"], float(np.sqrt(expected_foot.max())))
