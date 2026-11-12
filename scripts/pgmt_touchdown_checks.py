"""Independent PhysX height/NumPy quality checks for the finite P3 audit."""
import math

import numpy as np
import torch

from pmt_tasks.pgmt_touchdown import evaluate_touchdown
from pmt_tasks.pgmt_contact_match import REFERENCE_CONTACT_MATCH_WEIGHT
from pmt_tasks.pgmt_slip import SLIP_WEIGHT
from pmt_tasks.pgmt_stumble import STUMBLE_WEIGHT
from pmt_tasks.pgmt_contact_switching import CONTACT_SWITCHING_WEIGHT
from pmt_tasks.pgmt_contact_force import CONTACT_FORCE_WEIGHT


class TouchdownAudit:
    def __init__(self, env, state):
        import omni.physx
        self.env, self.state, self.terrain = env, state, env.scene.terrain
        self.physx = omni.physx.get_physx_scene_query_interface()
        self.report = dict(geometry_footprints=0, geometry_rays=0, support_rays=0,
                           max_height_error_m=0., max_quality_error=0., settled_steps=0,
                           rewarded_feet=0, reward_sum=0., negative_height_points=0,
                           continuous_without_touchdown=0, mismatched_env_steps=0, unknown_env_steps=0,
                           settlement=state.touchdown_config["settlement"])
        self.query_calls = 0
        original = self.terrain.query

        def counted_query(xy):
            self.query_calls += 1
            return original(xy)

        self.terrain.query = counted_query
        self.term = env.reward_manager.active_terms.index("terrain_touchdown_quality")

    def compare(self, sample, *, geometry=False):
        points = sample["query_xy_w"].cpu().numpy()
        heights = sample["height_w"].cpu().numpy()
        valid = sample["valid"].cpu().numpy()
        for env_id, foot in np.argwhere(sample["eligible"].cpu().numpy()):
            if not np.isfinite(points[env_id, foot]).all():
                continue
            measured = []
            for x, y in points[env_id, foot]:
                elevations = []

                def collect(hit):
                    collision = hit["collision"] if isinstance(hit, dict) else hit.collision
                    if str(collision).startswith(self.terrain.cfg.prim_path + "/terrain/Terrain/"):
                        position = hit["position"] if isinstance(hit, dict) else hit.position
                        elevations.append(float(position[2]))
                    return True

                self.physx.raycast_all((float(x), float(y), self.terrain.ray_top), (0., 0., -1.), 1000., collect)
                measured.append(max(elevations) if elevations else float("nan"))
            measured = np.asarray(measured)
            target = heights[env_id, foot]
            np.testing.assert_allclose(target, measured, atol=1e-4, rtol=0, equal_nan=True)
            error = np.abs(target - measured)
            if np.isfinite(error).any():
                self.report["max_height_error_m"] = max(self.report["max_height_error_m"], float(np.nanmax(error)))
            self.report["geometry_rays" if geometry else "support_rays"] += 9
            self.report["negative_height_points"] += int((measured < -1e-5).sum())
            expected = math.exp(-float(measured.var()) / self.state.touchdown_config["sigma_m"]**2) if valid[env_id, foot] else 0.
            actual = float(sample["quality"][env_id, foot])
            self.report["max_quality_error"] = max(self.report["max_quality_error"], abs(actual - expected))
            assert abs(actual - expected) < 2e-4, (actual, expected)

    def geometry(self):
        # One interior footprint per family/level, both yaw=0 and yaw=90.
        # These are geometry probes, not synthetic claims of physical touchdown.
        patches = [p for p in self.terrain.surface_records if p["patch_row"] == 0 and p["patch_col"] == 0]
        assert len(patches) == 50
        n = len(patches)
        pos = torch.tensor([p["center"] for p in patches], device=self.env.device, dtype=torch.float32)
        pos[:, :2] += pos.new_tensor([.873, .817])
        feet = pos[:, None].expand(-1, 2, -1).clone()
        quat = torch.zeros(n, 2, 4, device=self.env.device)
        quat[:, 0, 0] = 1.
        quat[:, 1, 0] = quat[:, 1, 3] = 2**-.5
        result = evaluate_touchdown(torch.ones(n, 2, device=self.env.device, dtype=torch.bool),
                                    feet, quat, pos, quat[:, 0], self.terrain.query, self.state.touchdown_config)
        assert result["valid"].all()
        self.compare(result, geometry=True)
        self.report["geometry_footprints"] = n * 2
        self.report["families"] = self.terrain.manifest["spec"]["families"]
        self.report["levels"] = list(range(10))

    def settled_step(self, total_reward):
        from pmt_tasks.pgmt_runtime import reward_groups_from_manager
        sample = self.state.touchdown_state(for_reward=True)
        self.compare(sample)
        contact = self.state.contact_state(for_reward=True)
        ref = contact["reference_contacts"]
        ref_valid = contact["reference_contact_valid"] & ((ref == 0) | (ref == 1))
        known = (contact["support_valid"] & ref_valid).all(-1)
        matched = known & (contact["support"] == (ref == 1)).all(-1)
        count = ((ref == 1) & ref_valid).sum(-1)
        eligible = matched[:, None] & (ref == 1)
        torch.testing.assert_close(sample["pattern_valid"], known)
        torch.testing.assert_close(sample["pattern_matched"], matched)
        torch.testing.assert_close(sample["reference_support_count"], count)
        torch.testing.assert_close(sample["eligible"], eligible)
        torch.testing.assert_close(sample["event"], contact["touchdown"])
        assert sample["quality"][~eligible].eq(0).all()
        rate = self.state.rewards()["terrain_touchdown_quality"]
        torch.testing.assert_close(rate, sample["quality"].sum(-1) / count.clamp_min(1))
        torch.testing.assert_close(rate, sample["rate"])
        assert ((rate >= 0) & (rate <= 1)).all()
        actual = self.env.reward_manager._step_reward[:, self.term] * self.env.step_dt
        expected = sample["rate"] * 10. * self.env.step_dt
        torch.testing.assert_close(actual, expected)
        groups = reward_groups_from_manager(self.env.reward_manager, self.env.step_dt,
                                            ("upper", "lower", "terrain", "auxiliary"))
        expected_terrain = expected
        if "terrain_reference_contact_match" in self.env.reward_manager.active_terms:
            expected_terrain = expected + self.state.rewards()["terrain_reference_contact_match"] * REFERENCE_CONTACT_MATCH_WEIGHT * self.env.step_dt
        if "terrain_slip" in self.env.reward_manager.active_terms:
            expected_terrain = expected_terrain + self.state.rewards()["terrain_slip"] * SLIP_WEIGHT * self.env.step_dt
        if "terrain_stumble" in self.env.reward_manager.active_terms:
            expected_terrain = expected_terrain + self.state.rewards()["terrain_stumble"] * STUMBLE_WEIGHT * self.env.step_dt
        if "terrain_contact_switching" in self.env.reward_manager.active_terms:
            expected_terrain = expected_terrain + self.state.rewards()["terrain_contact_switching"] * CONTACT_SWITCHING_WEIGHT * self.env.step_dt
        if "terrain_contact_force" in self.env.reward_manager.active_terms:
            expected_terrain = expected_terrain + self.state.rewards()["terrain_contact_force"] * CONTACT_FORCE_WEIGHT * self.env.step_dt
        torch.testing.assert_close(groups[:, 2], expected_terrain)
        torch.testing.assert_close(groups.sum(-1), total_reward)
        before = self.query_calls
        self.state.rewards(); self.state.rewards()
        after = self.state.touchdown_state(for_reward=True)
        for key in sample:
            torch.testing.assert_close(after[key], sample[key], equal_nan=True)
        assert self.query_calls == before
        self.report["settled_steps"] += 1
        self.report["rewarded_feet"] += int((sample["quality"] > 0).sum())
        self.report["continuous_without_touchdown"] += int(((sample["quality"] > 0) & ~sample["event"]).sum())
        self.report["mismatched_env_steps"] += int((known & ~matched).sum())
        self.report["unknown_env_steps"] += int((~known).sum())
        self.report["reward_sum"] += float(expected.sum())
