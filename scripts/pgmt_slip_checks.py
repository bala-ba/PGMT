"""T3 real-physics oracle: NumPy cost and independent link-origin velocity."""
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from pmt_tasks.pgmt_runtime import _tensor
from pmt_tasks.pgmt_slip import SLIP_WEIGHT


class SlipAudit:
    def __init__(self, env, state):
        self.env, self.state = env, state
        self.term = env.reward_manager.active_terms.index("terrain_slip")
        self.report = dict(settled_steps=0, active_feet=0, moving_contact_feet=0, zero_cost_unloaded_feet=0,
                           reward_sum=0., max_rate_error=0., max_link_velocity_error_mps=0.,
                           link_velocity_checks=0, nonterrain_contact_samples=0)

    def settled_step(self, total_reward):
        contact = self.state.contact_state(for_reward=True)
        sample = self.state.slip_state(for_reward=True)
        for key in ("sample_step", "episode_step", "episode_id", "reference_motion_ids", "reference_frame_ids"):
            torch.testing.assert_close(sample[key], contact[key])
        torch.testing.assert_close(sample["velocity_w"], contact["foot_body_lin_vel"])
        forces = contact["net_forces_w"].cpu().numpy().astype(np.float64)
        velocity = contact["foot_body_lin_vel"].cpu().numpy().astype(np.float64)
        gate = np.linalg.norm(forces, axis=-1) > self.env.cfg.pgmt_contact_events["contact_threshold_n"]
        assert np.isfinite(velocity).all() and np.isfinite(forces).all() and sample["valid"].all()
        per_foot = np.sum(velocity[..., :2] ** 2, axis=-1) * gate
        expected = per_foot.sum(-1)
        np.testing.assert_allclose(sample["per_foot"].cpu(), per_foot, atol=1e-6, rtol=2e-6)
        np.testing.assert_allclose(sample["rate"].cpu(), expected, atol=1e-6, rtol=2e-6)
        rate = self.state.rewards()["terrain_slip"]
        torch.testing.assert_close(rate, sample["rate"])
        actual = self.env.reward_manager._step_reward[:, self.term] * self.env.step_dt
        torch.testing.assert_close(actual, rate * SLIP_WEIGHT * self.env.step_dt)
        assert actual.le(0).all()
        # On non-reset rows, independently reconstruct actor/link velocity from
        # COM velocity, angular velocity and the rotated COM offset. No world
        # position subtraction, which loses precision on this large terrain.
        keep = self.state.contact_state()["sample_step"] == contact["sample_step"]
        r, feet = self.state.robot.data, self.state.foot_body_indices
        def array(value):
            return _tensor(value)[keep][:, feet].cpu().numpy().astype(np.float64)
        if keep.any():
            com_v, omega = array(r.body_com_lin_vel_w), array(r.body_com_ang_vel_w)
            quat, offset = array(r.body_link_quat_w), array(r.body_com_pos_b)
            rotated = Rotation.from_quat(quat[..., [1, 2, 3, 0]].reshape(-1, 4)).apply(offset.reshape(-1, 3))
            link_v = com_v - np.cross(omega, rotated.reshape(offset.shape))
            measured = sample["velocity_w"][keep].cpu().numpy()
            np.testing.assert_allclose(measured, link_v, rtol=1e-5, atol=1e-5)
            self.report["max_link_velocity_error_mps"] = max(
                self.report["max_link_velocity_error_mps"], float(np.max(np.abs(measured-link_v))))
            self.report["link_velocity_checks"] += int(keep.sum()) * 2
        before_step = self.state.contact_state()["episode_step"].clone()
        self.state.rewards(); self.state.rewards()
        for key, value in sample.items():
            torch.testing.assert_close(self.state.slip_state(for_reward=True)[key], value)
        torch.testing.assert_close(before_step, self.state.contact_state()["episode_step"])
        self.report["settled_steps"] += 1
        self.report["active_feet"] += int(gate.sum())
        self.report["moving_contact_feet"] += int((per_foot > 1e-8).sum())
        self.report["zero_cost_unloaded_feet"] += int((~gate & (per_foot == 0)).sum())
        self.report["nonterrain_contact_samples"] += int((contact["contact"] & (
            contact["terrain_forces_w"].abs().sum(-1) < 1e-6)).sum())
        self.report["max_rate_error"] = max(self.report["max_rate_error"], float(np.max(np.abs(rate.cpu().numpy()-expected))))
        self.report["reward_sum"] += float(actual.sum())
