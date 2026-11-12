"""T4 real-physics oracle: NumPy threshold and native reward integration."""
import numpy as np
import torch

from pmt_tasks.pgmt_stumble import STUMBLE_WEIGHT


class StumbleAudit:
    def __init__(self, env, state):
        self.env, self.state = env, state
        self.term = env.reward_manager.active_terms.index("terrain_stumble")
        self.previous = None
        self.report = dict(settled_steps=0, triggered_feet=0, zero_cost_feet=0,
                           without_support=0, without_touchdown=0, consecutive_triggered_feet=0,
                           nonterrain_triggered_feet=0, reward_sum=0., max_rate_error=0.)

    def settled_step(self, total_reward):
        contact = self.state.contact_state(for_reward=True)
        sample = self.state.stumble_state(for_reward=True)
        for key in ("sample_step", "episode_step", "episode_id", "reference_motion_ids", "reference_frame_ids"):
            torch.testing.assert_close(sample[key], contact[key])
        torch.testing.assert_close(sample["forces_w"], contact["net_forces_w"])
        # Independent double-precision oracle, using frozen physical forces.
        forces = contact["net_forces_w"].cpu().numpy().astype(np.float64)
        assert np.isfinite(forces).all() and sample["valid"].all()
        cfg = self.env.cfg.pgmt_stumble
        horizontal = np.linalg.norm(forces[..., :2], axis=-1)
        threshold = cfg["vertical_ratio"] * np.abs(forces[..., 2]) + cfg["bias_n"]
        triggered = horizontal > threshold
        expected = triggered.sum(-1)
        np.testing.assert_allclose(sample["horizontal_force"].cpu(), horizontal, rtol=2e-6, atol=1e-5)
        np.testing.assert_allclose(sample["threshold"].cpu(), threshold, rtol=2e-6, atol=1e-5)
        np.testing.assert_array_equal(sample["stumble"].cpu(), triggered)
        np.testing.assert_array_equal(sample["per_foot"].cpu(), triggered.astype(float))
        np.testing.assert_array_equal(sample["rate"].cpu(), expected)
        rate = self.state.rewards()["terrain_stumble"]
        torch.testing.assert_close(rate, sample["rate"])
        actual = self.env.reward_manager._step_reward[:, self.term] * self.env.step_dt
        torch.testing.assert_close(actual, rate * STUMBLE_WEIGHT * self.env.step_dt)
        assert actual.le(0).all()
        before_step = self.state.contact_state()["episode_step"].clone()
        self.state.rewards(); self.state.rewards()
        for key, value in sample.items():
            torch.testing.assert_close(self.state.stumble_state(for_reward=True)[key], value)
        torch.testing.assert_close(before_step, self.state.contact_state()["episode_step"])
        active = sample["stumble"]
        self.report["settled_steps"] += 1
        self.report["triggered_feet"] += int(active.sum())
        self.report["zero_cost_feet"] += int((~active).sum())
        self.report["without_support"] += int((active & ~contact["support"]).sum())
        self.report["without_touchdown"] += int((active & ~contact["touchdown"]).sum())
        self.report["nonterrain_triggered_feet"] += int((active & (
            contact["terrain_forces_w"].abs().sum(-1) < 1e-6)).sum())
        if self.previous is not None:
            same_episode = sample["episode_id"] == self.previous["episode_id"]
            self.report["consecutive_triggered_feet"] += int((
                active & self.previous["stumble"] & same_episode[:, None]).sum())
        self.previous = sample
        self.report["max_rate_error"] = max(self.report["max_rate_error"], float(np.max(np.abs(rate.cpu().numpy()-expected))))
        self.report["reward_sum"] += float(actual.sum())

    def side_collision_scenarios(self, actions, touchdown_audit, match_audit, slip_audit, wall_position,
                                 switching_audit=None, force_audit=None):
        """Translate an intact pose outside a real wall, then approach it.

        Only initial pose/velocity are controlled. Forces come from PhysX,
        are never overwritten, and all four terms use native env.step().
        """
        from pmt_tasks.pgmt_reference import collision_centres

        env, state = self.env, self.state
        robot, command = state.robot, state.command
        ids = torch.tensor([0], device=env.device)
        foot_cover = (command.collision_body_ids[:, None] == state.foot_body_indices[None]).any(-1)
        scenarios = []
        for speed in (.5, 1., 2.):
            command._resample_command(ids)
            env.sim.forward()
            pose = dict(body_pos=robot.data.body_pos_w[ids], body_quat=robot.data.body_quat_w[ids])
            centres = collision_centres(pose, command.collision_body_ids, command.collision_local_centres)[0, foot_cover]
            radii = command.collision_radii[foot_cover]
            root = robot.data.root_link_state_w[ids].clone()
            # Conservative foot cover starts 15 mm left of the wall's X face.
            # Set foot centre height inside the side face, away from its edges.
            shift = root.new_tensor(wall_position) - centres.mean(0)
            shift[0] = wall_position[0] - .1 - .015 - (centres[:, 0] + radii).max()
            root[:, :3] += shift
            root[:, 7:] = 0
            root[:, 7] = speed
            robot.write_root_link_state_to_sim(root, env_ids=ids)
            robot.write_joint_state_to_sim(robot.data.joint_pos[ids], torch.zeros_like(robot.data.joint_vel[ids]), env_ids=ids)
            command.reference_translation[ids] += shift
            command._world_tracked = None
            env.sim.forward()
            state.reset(ids)
            count, examples = 0, []
            for _ in range(10):
                match_audit.before_step()
                _, reward, *_ = env.step(actions)
                match_audit.settled_step(reward)
                slip_audit.settled_step(reward)
                touchdown_audit.settled_step(reward)
                self.settled_step(reward)
                if switching_audit:
                    switching_audit.settled_step(reward)
                if force_audit:
                    force_audit.settled_step(reward)
                contact = state.contact_state(for_reward=True, env_ids=ids)
                sample = state.stumble_state(for_reward=True, env_ids=ids)
                assert contact["terrain_forces_w"].abs().max() < 1e-4
                assert not contact["support"].any() and not contact["touchdown"].any()
                count += int(sample["stumble"].sum())
                if sample["stumble"].any():
                    examples.append(dict(step=env.common_step_counter, forces_w=sample["forces_w"][0].cpu().tolist(),
                                         threshold=sample["threshold"][0].cpu().tolist(), rate=float(sample["rate"][0])))
            scenarios.append(dict(initial_speed_mps=speed, triggered_feet=count, examples=examples))
        self.report["side_collision_scenarios"] = scenarios
        assert sum(s["triggered_feet"] for s in scenarios) > 0, scenarios
