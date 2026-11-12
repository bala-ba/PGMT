"""Independent NumPy/real RewardManager oracle for T2 contact agreement."""
import numpy as np
import torch

from pmt_tasks.pgmt_contact_match import REFERENCE_CONTACT_MATCH_WEIGHT
from pmt_tasks.pgmt_runtime import reward_groups_from_manager


class ContactMatchAudit:
    def __init__(self, env, state):
        self.env, self.state = env, state
        self.term = env.reward_manager.active_terms.index("terrain_reference_contact_match")
        self.report = dict(settled_steps=0, evaluated_feet=0, matching_feet=0, reward_sum=0.,
                           contact_support_disagreements=0, max_rate_error=0., reference_masks=[])
        self.pending_reference = None

    def before_step(self):
        # Read the command timeline BEFORE native advancement/reset. Comparing
        # only two post-step caches would miss a shared one-frame timing error.
        self.pending_reference = self.state.reference_contacts()

    def settled_step(self, total_reward):
        contact = self.state.contact_state(for_reward=True)
        sample = self.state.reference_contact_match_state(for_reward=True)
        ref = self.pending_reference
        assert ref is not None
        for field in ("contacts", "contact_valid", "motion_ids", "frame_ids"):
            torch.testing.assert_close(contact[f"reference_{field}"], ref[field])
        for field in ("sample_step", "episode_step", "episode_id", "reference_motion_ids", "reference_frame_ids"):
            torch.testing.assert_close(sample[field], contact[field])
        forces = contact["net_forces_w"].cpu().numpy()
        norm = np.linalg.norm(forces, axis=-1)
        actual = norm > self.env.cfg.pgmt_contact_events["contact_threshold_n"]
        actual_valid = np.isfinite(forces).all(-1) & np.isfinite(norm)
        labels, valid = ref["contacts"].cpu().numpy(), ref["contact_valid"].cpu().numpy()
        matched = (actual == labels) & actual_valid & valid
        expected = matched.sum(-1) / valid.sum(-1).clip(1)
        np.testing.assert_array_equal(sample["actual_contact"].cpu(), actual & actual_valid)
        np.testing.assert_array_equal(sample["matched"].cpu(), matched)
        np.testing.assert_array_equal(sample["reference_count"].cpu(), valid.sum(-1))
        np.testing.assert_allclose(sample["rate"].cpu(), expected, rtol=0, atol=0)
        rate = self.state.rewards()["terrain_reference_contact_match"]
        torch.testing.assert_close(rate, sample["rate"])
        integrated = self.env.reward_manager._step_reward[:, self.term] * self.env.step_dt
        target = rate * REFERENCE_CONTACT_MATCH_WEIGHT * self.env.step_dt
        torch.testing.assert_close(integrated, target)
        groups = reward_groups_from_manager(self.env.reward_manager, self.env.step_dt,
                                            ("upper", "lower", "terrain", "auxiliary"))
        torch.testing.assert_close(groups.sum(-1), total_reward)
        for _ in range(2):
            self.state.rewards()
            current = self.state.reference_contact_match_state(for_reward=True)
            for key, value in sample.items():
                torch.testing.assert_close(current[key], value)
        self.report["settled_steps"] += 1
        self.report["evaluated_feet"] += int((actual_valid & valid).sum())
        self.report["matching_feet"] += int(matched.sum())
        self.report["contact_support_disagreements"] += int((contact["contact"] != contact["support"]).sum())
        self.report["reward_sum"] += float(target.sum())
        self.report["max_rate_error"] = max(self.report["max_rate_error"], float(np.max(np.abs(rate.cpu().numpy()-expected))))
        self.pending_reference = None

    def reference_mask_scenarios(self, actions, touchdown_audit, slip_audit=None, stumble_audit=None,
                                 switching_audit=None, force_audit=None):
        # Controlled in-memory label fixtures, not edits to source motion files.
        # Real physics, sensor states, event lifecycle and native dt are retained.
        store = self.state.contact_label_store
        labels, valid = store._contacts.clone(), store._valid.clone()
        try:
            for name, contacts, mask in (
                ("both_stance", (True, True), (True, True)),
                ("both_swing", (False, False), (True, True)),
                ("left_only", (True, False), (True, False)),
                ("right_only", (False, True), (False, True)),
                ("both_unknown", (False, False), (False, False)),
            ):
                store._contacts[:] = store._contacts.new_tensor(contacts)
                store._valid[:] = store._valid.new_tensor(mask)
                self.before_step()
                _, reward, *_ = self.env.step(actions)
                self.settled_step(reward)
                touchdown_audit.settled_step(reward)
                if slip_audit:
                    slip_audit.settled_step(reward)
                if stumble_audit:
                    stumble_audit.settled_step(reward)
                if switching_audit:
                    switching_audit.settled_step(reward)
                if force_audit:
                    force_audit.settled_step(reward)
                sample = self.state.reference_contact_match_state(for_reward=True)
                assert sample["reference_count"].eq(sum(mask)).all()
                if not any(mask):
                    assert sample["rate"].eq(0).all()
                self.report["reference_masks"].append(dict(name=name, rates=sample["rate"].cpu().tolist()))
        finally:
            store._contacts.copy_(labels)
            store._valid.copy_(valid)
