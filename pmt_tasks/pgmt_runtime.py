"""PGMT observation/reward caches on PMT's native manager lifecycle.

Only the PGMT task constructs this state. Its complete-body reference view uses
the same native loader and chunks as the command; the command retains PMT's
14 tracked bodies for sampling, termination, reset and metrics.
"""
from __future__ import annotations

import torch

from pmt_tasks.pgmt_contact_reference import CONTACT_FOOT_NAMES
from pmt_tasks.pgmt_contact_match import empty_contact_match, evaluate_reference_contact_match
from pmt_tasks.pgmt_slip import empty_slip, evaluate_slip
from pmt_tasks.pgmt_stumble import empty_stumble, evaluate_stumble, validate_stumble_config
from pmt_tasks.pgmt_contact_switching import empty_contact_switching, evaluate_contact_switching
from pmt_tasks.pgmt_contact_force import empty_contact_force, evaluate_contact_force
from pmt_tasks.pgmt_tracking_relaxation import RELAXATION_TERMS, TrackingRelaxation
from pmt_tasks.pgmt_contact_state import (
    ContactEventState, TERRAIN_FOOT_SENSORS, terrain_force_from_matrix,
)
from pmt_tasks.pgmt_height_scan import height_scan_from_hits
from pmt_tasks.pgmt_math import TRACKING_CONFIG, body_groups, build_observations, future_offsets, reward_terms
from pmt_tasks.pgmt_touchdown import (
    empty_touchdown, evaluate_support_quality, validate_touchdown_config, validate_touchdown_scanner,
)


def _tensor(value):
    return value if isinstance(value, torch.Tensor) else value.torch


class PGMTState:
    def __init__(self, env):
        from pmt_tasks.mdp.commands.multi_motion_command import MotionDataStore

        self.env = env
        self.command = env.command_manager.get_term("motion")
        self.robot = env.scene["robot"]
        self.anchor = self.robot.body_names.index("torso_link")
        self.config = dict(TRACKING_CONFIG)
        self.config.update(getattr(env.cfg, "pgmt_tracking", {}))
        self.groups = body_groups(
            self.robot.body_names, self.robot.joint_names, device=env.device,
            end_effector_names=getattr(env.cfg, "pgmt_ee_acceleration_body_names", None))
        source = self.command.data_store
        self.full_store = getattr(self.command, "full_reference_store", None)
        if self.full_store is None:
            self.full_store = MotionDataStore(device=source.device, storage_device=source.storage_device,
                                             use_fp16=source.dtype == torch.float16, chunk_length=source.chunk_length)
            self.full_store.load_files(self.command.cfg.motion_files, list(range(self.robot.num_bodies)), load_human=False)
        if [m.source_file for m in source.motions] != [m.source_file for m in self.full_store.motions]:
            raise RuntimeError("PGMT complete-body reference chunks do not match the native command")
        if not torch.equal(source.motion_lengths, self.full_store.motion_lengths):
            raise RuntimeError("PGMT reference chunk lengths do not match the native command")
        if self.full_store.num_bodies != self.robot.num_bodies:
            raise ValueError("PGMT requires every robot body in the reference")
        self.contact_order = torch.tensor([env.scene["contact_forces"].body_names.index(name)
                                           for name in self.robot.body_names], device=env.device)
        self.contact_label_store = getattr(self.command, "contact_label_store", None)
        self.foot_names = CONTACT_FOOT_NAMES
        self.foot_body_indices = torch.tensor([self.robot.body_names.index(name) for name in self.foot_names],
                                              device=env.device)
        self.foot_sensor_indices = torch.tensor([env.scene["contact_forces"].body_names.index(name)
                                                for name in self.foot_names], device=env.device)
        if self.contact_label_store is not None:
            if self.contact_label_store.source_files != tuple(m.source_file for m in source.motions):
                raise RuntimeError("PGMT contact labels do not match command motion order")
            if not torch.equal(self.contact_label_store.motion_lengths, source.motion_lengths):
                raise RuntimeError("PGMT contact label lengths do not match command motions")
        self.previous_target = self.qtarget().clone()
        self.previous_velocity = torch.zeros_like(_tensor(self.robot.data.body_link_lin_vel_w))
        self.previous_ref_velocity = torch.zeros_like(self.previous_velocity)
        self.pending_reset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self._obs_version = 0
        self._obs_key = None
        self._obs = None
        self._terrain_scan_key = None
        self._terrain_scan = None
        self._reward_step = None
        self._rewards = None
        self.contact_tracker = None
        self._reward_contacts = None
        self._contact_reward_step = None
        contact_config = getattr(env.cfg, "pgmt_contact_events", None)
        if contact_config is not None:
            self.contact_tracker = ContactEventState(env.num_envs, device=env.device, dt=env.step_dt,
                                                     config=contact_config)
            expected_filter = env.scene.terrain.contact_filter_paths
            for sensor_name, foot_name in zip(TERRAIN_FOOT_SENSORS, self.foot_names):
                sensor = env.scene.sensors.get(sensor_name)
                if sensor is None or sensor.body_names != [foot_name] or sensor.cfg.filter_prim_paths_expr != expected_filter:
                    raise RuntimeError(f"PGMT requires one terrain-filtered sensor per foot: {sensor_name}")
                if sensor.contact_physx_view.filter_count != len(expected_filter):
                    raise RuntimeError(f"PGMT terrain filter columns do not match the imported colliders: {sensor_name}")
            self._contact_context = {f"reference_{key}": value.clone()
                                     for key, value in self.reference_contacts().items()}
            for key, size in (("body_pos", 3), ("body_quat", 4), ("body_lin_vel", 3), ("body_ang_vel", 3)):
                self._contact_context[f"foot_{key}"] = torch.full((env.num_envs, 2, size), float("nan"), device=env.device)
            self._invalidate_contact_context(slice(None))
        touchdown_config = getattr(env.cfg, "pgmt_touchdown_quality", None)
        self.touchdown_config = None
        if touchdown_config is not None:
            if self.contact_tracker is None or self.contact_label_store is None:
                raise ValueError("T1 support quality requires shared support state and reference labels")
            validate_touchdown_config(touchdown_config)
            scanner = env.scene.sensors.get("height_scanner")
            validate_touchdown_scanner(None if scanner is None else scanner.cfg)
            self.touchdown_config = dict(touchdown_config)
            self._touchdown_live = empty_touchdown(env.num_envs, device=env.device)
            self._touchdown_reward = None
        self.contact_match_enabled = getattr(env.cfg, "pgmt_reference_contact_match", False)
        if type(self.contact_match_enabled) is not bool:
            raise ValueError("pgmt_reference_contact_match must be boolean")
        if self.contact_match_enabled:
            if self.contact_tracker is None or self.contact_label_store is None:
                raise ValueError("Reference contact match requires shared contact state and reference labels")
            self._contact_match_live = empty_contact_match(env.num_envs, device=env.device)
            self._contact_match_reward = None
        self.slip_enabled = getattr(env.cfg, "pgmt_slip", False)
        if type(self.slip_enabled) is not bool:
            raise ValueError("pgmt_slip must be boolean")
        if self.slip_enabled:
            if self.contact_tracker is None:
                raise ValueError("Slip requires the shared contact state")
            self._slip_live = empty_slip(env.num_envs, device=env.device)
            self._slip_reward = None
        self.stumble_config = getattr(env.cfg, "pgmt_stumble", None)
        if self.stumble_config is not None:
            if self.contact_tracker is None:
                raise ValueError("Stumble requires the shared contact state")
            validate_stumble_config(self.stumble_config)
            self.stumble_config = dict(self.stumble_config)
            self._stumble_live = empty_stumble(env.num_envs, device=env.device)
            self._stumble_reward = None
        self.contact_switching_enabled = getattr(env.cfg, "pgmt_contact_switching", False)
        if type(self.contact_switching_enabled) is not bool:
            raise ValueError("pgmt_contact_switching must be boolean")
        if self.contact_switching_enabled:
            if self.contact_tracker is None:
                raise ValueError("Contact switching requires the shared contact state")
            self._contact_switching_live = empty_contact_switching(env.num_envs, device=env.device)
            self._contact_switching_reward = None
        self.contact_force_enabled = getattr(env.cfg, "pgmt_contact_force", False)
        if type(self.contact_force_enabled) is not bool:
            raise ValueError("pgmt_contact_force must be boolean")
        if self.contact_force_enabled:
            if self.contact_tracker is None:
                raise ValueError("Contact force requires the shared contact state")
            self._contact_force_live = empty_contact_force(env.num_envs, device=env.device)
            self._contact_force_reward = None
        self._configure_tracking_relaxation()

    def _configure_tracking_relaxation(self):
        config = getattr(self.env.cfg, "pgmt_tracking_relaxation", None)
        self.tracking_relaxation = None
        if config is None:
            return
        resolved = TrackingRelaxation(config, self.robot.body_names, self.robot.joint_names,
                                      self.groups, device=self.env.device,
                                      dtype=_tensor(self.robot.data.joint_pos).dtype)
        if not resolved.config["enabled"]:
            return
        if not hasattr(self.env.scene.terrain, "tile_map"):
            raise ValueError("Tracking relaxation requires the PGMT current-position terrain map")
        self.tracking_relaxation = resolved
        n, device = self.env.num_envs, self.env.device
        self._tracking_relaxation_reward = None
        self._tracking_relaxation_live = {
            key: torch.zeros(n, dtype=torch.bool, device=device)
            for key in ("sample_valid", "tile_valid", "applicable")}
        self._tracking_relaxation_live.update({key: torch.full((n,), -1, dtype=torch.long, device=device)
            for key in ("sample_step", "tile_level", "family_index", "family_group_index", "birth_level",
                        "birth_family_index", "reference_motion_ids", "reference_frame_ids")})
        self._tracking_relaxation_live.update({key: torch.full((n,), float("nan"), device=device)
                                               for key in ("difficulty", "level_scale")})
        self._tracking_relaxation_live["anchor_xy_w"] = torch.full((n, 2), float("nan"), device=device)
        for term in RELAXATION_TERMS:
            for field in ("raw_error", "margin", "relaxed_error", "strict_rate", "relaxed_rate"):
                shape = (n,) if field.endswith("rate") else (n, len(resolved.names[term]))
                self._tracking_relaxation_live[f"{term}_{field}"] = torch.full(shape, float("nan"), device=device)
        self._invalidate_tracking_relaxation(slice(None))

    def _tracking_relaxation_sample(self, state):
        terrain = self.env.scene.terrain
        xy = state["body_pos"][:, self.anchor, :2]
        tile = terrain.tile_map.locate(xy)
        margins, scale, applicable = self.tracking_relaxation.margins(tile)
        snapshot = {key: tile[key].clone() for key in (
            "tile_valid", "tile_level", "family_index", "family_group_index", "difficulty")}
        snapshot.update(sample_valid=torch.ones_like(tile["tile_valid"]),
                        sample_step=torch.full_like(tile["tile_level"], self.env.common_step_counter),
                        anchor_xy_w=xy.clone(), level_scale=scale, applicable=applicable,
                        birth_level=terrain.terrain_levels.clone(), birth_family_index=terrain.terrain_types.clone(),
                        reference_motion_ids=self.command.motion_ids.clone(),
                        reference_frame_ids=self.command.frame_ids.clone())
        return margins, snapshot

    def _invalidate_tracking_relaxation(self, env_ids):
        for key, value in self._tracking_relaxation_live.items():
            if value.dtype == torch.bool or key.endswith("_margin") or key == "level_scale":
                self._clear_reset_rows(value, env_ids, 0)
            elif not value.is_floating_point():
                self._clear_reset_rows(value, env_ids, -1)
            else:
                self._clear_reset_rows(value, env_ids, float("nan"))

    def tracking_relaxation_state(self, *, env_ids=None, for_reward=False):
        """Copied raw/margin/residual/rates and current/birth tile metadata.

        Never queries terrain or recomputes rewards. for_reward=True retains
        the pre-reset terminal snapshot; the live reset rows are invalidated.
        Errors are nonnegative for finite physical inputs, including exact zero.
        """
        if getattr(self, "tracking_relaxation", None) is None:
            raise RuntimeError("Tracking relaxation is not enabled for this PGMT task")
        if for_reward and (self._tracking_relaxation_reward is None or self._reward_step != self.env.common_step_counter):
            raise RuntimeError("No settled tracking relaxation for this control step; call rewards() first")
        data = self._tracking_relaxation_reward if for_reward else self._tracking_relaxation_live
        ids = slice(None) if env_ids is None else env_ids
        return {key: value[ids].clone() for key, value in data.items()}

    def robot_state(self):
        r = self.robot.data
        return {"joint_pos": _tensor(r.joint_pos), "joint_vel": _tensor(r.joint_vel),
                "body_pos": _tensor(r.body_pos_w), "body_quat": _tensor(r.body_quat_w),
                "body_lin_vel": _tensor(r.body_link_lin_vel_w), "body_ang_vel": _tensor(r.body_ang_vel_w),
                "root_quat": _tensor(r.root_quat_w), "root_lin_vel": _tensor(r.root_link_lin_vel_w),
                "root_ang_vel": _tensor(r.root_ang_vel_w)}

    def _reference_indices(self, *, future=False, env_ids=None):
        """Shared pose/contact timeline; retain raw future indices for validity."""
        ids, frames = self.command.motion_ids, self.command.frame_ids
        if env_ids is not None:
            ids, frames = ids[env_ids], frames[env_ids]
        if future:
            offsets = future_offsets(device=frames.device)
            ids = ids[:, None].expand(-1, len(offsets))
            frames = frames[:, None] + offsets
        return ids, frames

    def reference(self, *, future=False):
        ids, frames = self._reference_indices(future=future)
        if future:
            # Clamp each future frame to its current native chunk, never to the
            # longest padded clip or to the source motion beyond this chunk.
            frames = torch.minimum(frames, self.full_store.motion_lengths[ids] - 1)
            shape = ids.shape
            arrays = self.full_store.get_motion_data(ids.reshape(-1), frames.reshape(-1))
            arrays = tuple(a.reshape(*shape, *a.shape[1:]).float() for a in arrays)
        else:
            arrays = tuple(a.float() for a in self.full_store.get_motion_data(ids, frames))
        result = dict(zip(("joint_pos", "joint_vel", "body_pos", "body_quat", "body_lin_vel", "body_ang_vel"), arrays))
        placement = getattr(self.command, "transform_reference", None)
        if placement is not None:
            return placement(result)
        origins = self.env.scene.env_origins
        result["body_pos"] = result["body_pos"] + (origins[:, None, None] if future else origins[:, None])
        return result

    def reference_contacts(self, *, future=False, env_ids=None):
        """Reference foot targets in left/right order, on the simulation device.

        contacts/contact_valid/stance/swing: [N,2], or [N,8,2] for future=True.
        Always mask contact comparisons with contact_valid; neither invalid
        labels nor exhausted future frames are negative contact examples.
        Fresh gathers follow command/reset indices even within the same step.
        Rigid reference placement does not change these categorical targets.
        """
        if self.contact_label_store is None:
            raise RuntimeError("Reference contacts require a PGMT command with labeled motion data")
        ids, frames = self._reference_indices(future=future, env_ids=env_ids)
        return self.contact_label_store.gather(ids, frames)

    def robot_foot_contact_forces(self, *, env_ids=None):
        """All-object normal-force sum [N,2,3], world/left-right; no threshold."""
        forces = _tensor(self.env.scene["contact_forces"].data.net_forces_w)
        if env_ids is not None:
            forces = forces[env_ids]
        return forces[:, self.foot_sensor_indices]

    def robot_foot_terrain_forces(self, *, env_ids=None):
        """Terrain-only normal forces [N,2,3]; excludes robots and outer border."""
        if getattr(self, "contact_tracker", None) is None:
            raise RuntimeError("Terrain-filtered forces require the PGMT P3 contact-state configuration")
        per_foot = []
        for name in TERRAIN_FOOT_SENSORS:
            matrix = self.env.scene.sensors[name].data.force_matrix_w
            per_foot.append(terrain_force_from_matrix(None if matrix is None else _tensor(matrix), self.env.num_envs))
        forces = torch.stack(per_foot, dim=1)
        return forces if env_ids is None else forces[env_ids]

    def contact_state(self, *, env_ids=None, for_reward=False):
        """Read a copy; never sample sensors or advance time from a reward reader.

        Default: last settled physical sample with reset rows invalidated.
        for_reward=True: frozen pre-reset reward sample (including terminal
        rows). The step/frame/episode ids make the timing explicit. Available
        after rewards() has run once; current state is invalid before that.
        """
        tracker = getattr(self, "contact_tracker", None)
        if tracker is None:
            raise RuntimeError("Contact events are not enabled for this PGMT task")
        ids = slice(None) if env_ids is None else env_ids
        if for_reward:
            if self._reward_contacts is None or self._contact_reward_step != self.env.common_step_counter:
                raise RuntimeError("No settled PGMT contact snapshot for this control step; call rewards() first")
            return {key: value[ids].clone() for key, value in self._reward_contacts.items()}
        return dict(tracker.snapshot(env_ids), **{key: value[ids].clone()
                                                  for key, value in self._contact_context.items()})

    def _update_contact_state(self, state, ordered_forces):
        tracker = getattr(self, "contact_tracker", None)
        if tracker is None:
            return
        tracker.update(self.env.common_step_counter, ordered_forces[:, self.foot_body_indices],
                       self.robot_foot_terrain_forces())
        context = {f"reference_{key}": value.clone() for key, value in self.reference_contacts().items()}
        for key in ("body_pos", "body_quat", "body_lin_vel", "body_ang_vel"):
            context[f"foot_{key}"] = state[key][:, self.foot_body_indices].clone()
        self._contact_context = context
        self._reward_contacts = dict(tracker.snapshot(), **{key: value.clone() for key, value in context.items()})
        self._contact_reward_step = self.env.common_step_counter
        if getattr(self, "touchdown_config", None) is not None:
            snapshot = self._reward_contacts
            self._touchdown_reward = evaluate_support_quality(
                snapshot["support"], snapshot["support_valid"],
                snapshot["reference_contacts"], snapshot["reference_contact_valid"],
                context["foot_body_pos"], context["foot_body_quat"],
                state["body_pos"][:, self.anchor], state["body_quat"][:, self.anchor],
                self.env.scene.terrain.query, self.touchdown_config, touchdown=snapshot["touchdown"])
            self._touchdown_live = {key: value.clone() for key, value in self._touchdown_reward.items()}
        if getattr(self, "contact_match_enabled", False):
            snapshot = self._reward_contacts
            self._contact_match_reward = evaluate_reference_contact_match(
                snapshot["contact"], snapshot["contact_valid"],
                snapshot["reference_contacts"], snapshot["reference_contact_valid"])
            self._contact_match_live = {key: value.clone() for key, value in self._contact_match_reward.items()}
        if getattr(self, "slip_enabled", False):
            snapshot = self._reward_contacts
            self._slip_reward = evaluate_slip(snapshot["contact"], snapshot["contact_valid"],
                                               snapshot["foot_body_lin_vel"])
            self._slip_live = {key: value.clone() for key, value in self._slip_reward.items()}
        if getattr(self, "stumble_config", None) is not None:
            snapshot = self._reward_contacts
            self._stumble_reward = evaluate_stumble(snapshot["net_forces_w"], snapshot["contact_valid"],
                                                   self.stumble_config)
            self._stumble_live = {key: value.clone() for key, value in self._stumble_reward.items()}
        if getattr(self, "contact_switching_enabled", False):
            snapshot = self._reward_contacts
            self._contact_switching_reward = evaluate_contact_switching(
                snapshot["fast_contact_switch"], snapshot["contact_history_valid"])
            self._contact_switching_live = {key: value.clone() for key, value in self._contact_switching_reward.items()}
        if getattr(self, "contact_force_enabled", False):
            snapshot = self._reward_contacts
            self._contact_force_reward = evaluate_contact_force(snapshot["net_forces_w"], snapshot["contact_valid"])
            self._contact_force_live = {key: value.clone() for key, value in self._contact_force_reward.items()}

    def contact_force_state(self, *, env_ids=None, for_reward=False):
        """Copied T6 foot forces/squared cost on the pre-reset reward timeline."""
        if not getattr(self, "contact_force_enabled", False):
            raise RuntimeError("Contact force is not configured for this PGMT task")
        contact = self.contact_state(env_ids=env_ids, for_reward=for_reward)
        ids = slice(None) if env_ids is None else env_ids
        data = self._contact_force_reward if for_reward else self._contact_force_live
        if data is None:
            raise RuntimeError("No settled contact force for this control step")
        return dict({key: value[ids].clone() for key, value in data.items()},
                    **{key: contact[key] for key in (
                        "sample_step", "episode_step", "episode_id", "reference_motion_ids", "reference_frame_ids")})

    def contact_switching_state(self, *, env_ids=None, for_reward=False):
        """Copied T5 events/ages/cost on the shared pre-reset reward timeline."""
        if not getattr(self, "contact_switching_enabled", False):
            raise RuntimeError("Contact switching is not configured for this PGMT task")
        contact = self.contact_state(env_ids=env_ids, for_reward=for_reward)
        ids = slice(None) if env_ids is None else env_ids
        data = self._contact_switching_reward if for_reward else self._contact_switching_live
        if data is None:
            raise RuntimeError("No settled contact switching for this control step")
        return dict({key: value[ids].clone() for key, value in data.items()},
                    **{key: contact[key] for key in (
                        "contact", "previous_contact", "contact_valid", "contact_history_valid",
                        "contact_age", "previous_contact_age", "contact_started", "contact_ended", "contact_switched",
                        "sample_step", "episode_step", "episode_id", "reference_motion_ids", "reference_frame_ids")})

    def stumble_state(self, *, env_ids=None, for_reward=False):
        """Copied T4 force/threshold/cost on the pre-reset reward timeline."""
        if getattr(self, "stumble_config", None) is None:
            raise RuntimeError("Stumble is not configured for this PGMT task")
        contact = self.contact_state(env_ids=env_ids, for_reward=for_reward)
        ids = slice(None) if env_ids is None else env_ids
        data = self._stumble_reward if for_reward else self._stumble_live
        if data is None:
            raise RuntimeError("No settled stumble for this control step")
        return dict({key: value[ids].clone() for key, value in data.items()},
                    **{key: contact[key] for key in (
                        "sample_step", "episode_step", "episode_id", "reference_motion_ids", "reference_frame_ids")})

    def slip_state(self, *, env_ids=None, for_reward=False):
        """Copied T3 velocity/gate/cost on the pre-reset reward timeline."""
        if not getattr(self, "slip_enabled", False):
            raise RuntimeError("Slip is not configured for this PGMT task")
        contact = self.contact_state(env_ids=env_ids, for_reward=for_reward)
        ids = slice(None) if env_ids is None else env_ids
        data = self._slip_reward if for_reward else self._slip_live
        if data is None:
            raise RuntimeError("No settled slip for this control step")
        return dict({key: value[ids].clone() for key, value in data.items()},
                    **{key: contact[key] for key in (
                        "sample_step", "episode_step", "episode_id", "reference_motion_ids", "reference_frame_ids")})

    def reference_contact_match_state(self, *, env_ids=None, for_reward=False):
        """Copied T2 match/masks/rate on the shared pre-reset reward timeline."""
        if not getattr(self, "contact_match_enabled", False):
            raise RuntimeError("Reference contact match is not configured for this PGMT task")
        contact = self.contact_state(env_ids=env_ids, for_reward=for_reward)
        ids = slice(None) if env_ids is None else env_ids
        data = self._contact_match_reward if for_reward else self._contact_match_live
        if data is None:
            raise RuntimeError("No settled reference contact match for this control step")
        return dict({key: value[ids].clone() for key, value in data.items()},
                    **{key: contact[key] for key in (
                        "sample_step", "episode_step", "episode_id", "reference_motion_ids", "reference_frame_ids")})

    def touchdown_state(self, *, env_ids=None, for_reward=False):
        """Copied T1 geometry/quality/rate, with the contact snapshot's timeline.

        rate is unweighted/unintegrated. Invalid/unqueried variance and heights
        remain NaN for diagnostics, while quality and rate are always finite.
        event retains the physical touchdown edge; eligible/pattern_matched
        describe the continuous T1 gate and reference_support_count its divisor.
        """
        if getattr(self, "touchdown_config", None) is None:
            raise RuntimeError("Touchdown quality is not configured for this PGMT task")
        contact = self.contact_state(env_ids=env_ids, for_reward=for_reward)
        ids = slice(None) if env_ids is None else env_ids
        data = self._touchdown_reward if for_reward else self._touchdown_live
        if data is None:
            raise RuntimeError("No settled touchdown quality for this control step")
        return dict({key: value[ids].clone() for key, value in data.items()},
                    **{key: contact[key] for key in (
                        "sample_step", "episode_step", "episode_id", "reference_motion_ids", "reference_frame_ids")})

    def _invalidate_contact_context(self, env_ids):
        for key, value in self._contact_context.items():
            if key.startswith("foot_"):
                self._clear_reset_rows(value, env_ids, float("nan"))
            elif value.dtype == torch.bool or key == "reference_contacts":
                self._clear_reset_rows(value, env_ids, 0)
            else:
                self._clear_reset_rows(value, env_ids, -1)

    def robot_foot_terrain(self, *, env_ids=None):
        """Actual terrain under left/right ankles, separate from spawn curriculum.

        query fields are [N,2]; curriculum_level/assigned_family_index are [N].
        Walking across tiles changes tile_level/family/height, never assignment.
        """
        terrain = self.env.scene.terrain
        if not hasattr(terrain, "tile_map"):
            raise RuntimeError("Foot terrain queries require a PGMT terrain map")
        ids = slice(None) if env_ids is None else env_ids
        xy = _tensor(self.robot.data.body_pos_w)[ids][:, self.foot_body_indices, :2]
        result = terrain.query(xy)
        result["curriculum_level"] = terrain.terrain_levels[ids].clone()
        result["assigned_family_index"] = terrain.terrain_types[ids].clone()
        return result

    def qtarget(self):
        # Read the target actually sent to the articulation. In particular, native
        # command resets only change joint state and do not assign a new target.
        return _tensor(self.robot.data.joint_pos_target)

    def terrain_scan(self):
        """Clean [N,21,21] torso-relative height, bool validity and world z.

        Native sensor buffers follow physical torso position/yaw, not the
        reference or birth tile. Cache once per observation version; command
        reset invalidates it even without a physics step. No noise/clipping is
        applied here. Stage 1 remains blind and cannot call this interface.
        """
        key = (self.env.common_step_counter, self._obs_version)
        if key != self._terrain_scan_key:
            terrain = self.env.scene.terrain
            sensor = self.env.scene.sensors.get("height_scanner")
            if sensor is None or not hasattr(terrain, "tile_map"):
                raise RuntimeError("PGMT terrain scans require a height scanner and PGMT terrain map")
            hits = sensor.data.ray_hits_w
            torso_z = _tensor(self.robot.data.body_pos_w)[:, self.anchor, 2]
            self._terrain_scan = height_scan_from_hits(hits, torso_z, terrain.contains(hits[..., :2]))
            self._terrain_scan_key = key
        return self._terrain_scan

    def terrain_observations(self):
        """Fresh PMT-style metric policy map/mask; terrain_scan stays clean."""
        config = getattr(self.env.cfg, "pgmt_terrain_observation", None)
        if config is None:
            return self.terrain_scan()
        from pmt_tasks.pgmt_perception import terrain_observation_from_hits
        sensor = self.env.scene.sensors["height_scanner"]
        hits = sensor.data.ray_hits_w
        return terrain_observation_from_hits(hits, sensor.data.pos_w[:, 2],
                                             self.env.scene.terrain.contains(hits[..., :2]), config)

    def invalidate_observations(self, env_ids=None):
        self._obs_version += 1

    def _clear_reset_rows(self, tensor, env_ids, value):
        if (getattr(self, '_fast_reset_buffers', False) and isinstance(env_ids, torch.Tensor)
                and env_ids.dtype == torch.long and env_ids.ndim == 1):
            tensor.index_fill_(0, env_ids, value)
        else:
            tensor[env_ids] = value

    def reset(self, env_ids):
        self._clear_reset_rows(self.pending_reset, env_ids, True)
        self.previous_target[env_ids] = self.qtarget()[env_ids]
        self.invalidate_observations(env_ids)
        if getattr(self, "tracking_relaxation", None) is not None:
            self._invalidate_tracking_relaxation(env_ids)
        tracker = getattr(self, "contact_tracker", None)
        if tracker is not None:
            tracker.reset(env_ids)
            self._invalidate_contact_context(env_ids)
            if getattr(self, "touchdown_config", None) is not None:
                for key, value in self._touchdown_live.items():
                    self._clear_reset_rows(value, env_ids, float("nan") if key in ("query_xy_w", "height_w", "variance") else 0)
            if getattr(self, "contact_match_enabled", False):
                for value in self._contact_match_live.values():
                    self._clear_reset_rows(value, env_ids, 0)
            if getattr(self, "slip_enabled", False):
                for key, value in self._slip_live.items():
                    self._clear_reset_rows(value, env_ids, float("nan") if key in ("velocity_w", "speed_xy_sq") else 0)
            if getattr(self, "stumble_config", None) is not None:
                for key, value in self._stumble_live.items():
                    self._clear_reset_rows(value, env_ids, float("nan") if key in (
                        "forces_w", "horizontal_force", "vertical_force_abs", "threshold") else 0)
            if getattr(self, "contact_switching_enabled", False):
                for value in self._contact_switching_live.values():
                    self._clear_reset_rows(value, env_ids, 0)
            if getattr(self, "contact_force_enabled", False):
                for key, value in self._contact_force_live.items():
                    self._clear_reset_rows(value, env_ids, float("nan") if key in ("forces_w", "force_sq") else 0)
            # Keep _reward_contacts: terminal events belong to the settled
            # reward, while contact_state() must expose invalid new-episode rows.
        # Command-internal clip resampling does not call ObservationManager.reset.
        # Reset only PGMT history here; native episode masks remain untouched.
        manager = getattr(self.env, "observation_manager", None)
        if manager is not None:
            histories = manager._group_obs_term_history_buffer.get("proprio_history", {})
            for history in histories.values():
                history.reset(batch_ids=env_ids)

    def _initialize_previous(self, state, ref):
        ids = self.pending_reset
        self.previous_velocity[ids] = state["body_lin_vel"][ids]
        self.previous_ref_velocity[ids] = ref["body_lin_vel"][ids]
        self.pending_reset[:] = False

    def observations(self):
        key = (self.env.common_step_counter, self._obs_version)
        if key != self._obs_key:
            state, ref = self.robot_state(), self.reference()
            self._initialize_previous(state, ref)
            self._obs = build_observations(state, ref, self.reference(future=True), self.qtarget(),
                                           self.env.scene.env_origins, anchor=self.anchor, config=self.config)
            # Use PMT's native noise amplitudes for the fields that survive the
            # PGMT layout change. Draw once so current proprio equals history[-1].
            if self.env.cfg.observations.proprio.enable_corruption:
                p = self._obs["proprio"]
                scales = p.new_tensor([0.05] * 6 + [0.2] * 3 + [0.01] * 29 + [0.025] * 29 + [0.] * 29)
                self._obs["proprio"] = p + (2 * torch.rand_like(p) - 1) * scales
            self._obs_key = key
        return self._obs

    def rewards(self):
        # RewardManager evaluates every term once, before command advancement
        # and resets. Repeated term reads in the same step must not advance caches.
        step = self.env.common_step_counter
        if step != self._reward_step:
            state, ref = self.robot_state(), self.reference()
            self._initialize_previous(state, ref)
            target = self.qtarget()
            forces = _tensor(self.env.scene["contact_forces"].data.net_forces_w)[:, self.contact_order]
            self._update_contact_state(state, forces)
            relaxation_kwargs = {}
            if getattr(self, "tracking_relaxation", None) is not None:
                margins, relaxation_snapshot = self._tracking_relaxation_sample(state)
                relaxation_kwargs = dict(tracking_margins=margins, tracking_debug=relaxation_snapshot)
            self._rewards = getattr(self, '_reward_kernel', reward_terms)(state, ref, target, self.previous_target,
                                         self.previous_velocity, self.previous_ref_velocity, forces,
                                         _tensor(self.robot.data.soft_joint_pos_limits), self.groups,
                                         anchor=self.anchor, dt=self.env.step_dt, config=self.config, **relaxation_kwargs)
            if relaxation_kwargs:
                self._tracking_relaxation_reward = relaxation_snapshot
                self._tracking_relaxation_live = {key: value.clone() for key, value in relaxation_snapshot.items()}
            if getattr(self, "touchdown_config", None) is not None:
                self._rewards["terrain_touchdown_quality"] = self._touchdown_reward["rate"]
            if getattr(self, "contact_match_enabled", False):
                self._rewards["terrain_reference_contact_match"] = self._contact_match_reward["rate"]
            if getattr(self, "slip_enabled", False):
                self._rewards["terrain_slip"] = self._slip_reward["rate"]
            if getattr(self, "stumble_config", None) is not None:
                self._rewards["terrain_stumble"] = self._stumble_reward["rate"]
            if getattr(self, "contact_switching_enabled", False):
                self._rewards["terrain_contact_switching"] = self._contact_switching_reward["rate"]
            if getattr(self, "contact_force_enabled", False):
                self._rewards["terrain_contact_force"] = self._contact_force_reward["rate"]
            self.previous_target.copy_(target)
            self.previous_velocity.copy_(state["body_lin_vel"])
            self.previous_ref_velocity.copy_(ref["body_lin_vel"])
            self._reward_step = step
        return self._rewards


def get_pgmt_state(env):
    state = getattr(env, "_pgmt_state", None)
    if state is None:
        state = env._pgmt_state = PGMTState(env)
    return state


def reward_groups_from_manager(manager, dt, group_names):
    """Recover this step's contributions, preserved through manager.reset().

    Isaac stores weighted reward rates in _step_reward (compute divides its
    already integrated contribution by dt). Reset clears episode sums only.
    Reintegrate that frozen step buffer once; never read reset episode sums.
    """
    names = manager.active_terms
    rates = manager._step_reward
    indices = [[i for i, name in enumerate(names) if name.startswith(group + "_")]
               for group in group_names]
    if any(not ids for ids in indices):
        raise ValueError(f"Missing PGMT reward group in {names}")
    if sum(map(len, indices)) != len(names):
        raise ValueError("Every active PGMT reward term must belong to exactly one value group")
    return torch.stack([rates[:, ids].sum(-1) * dt for ids in indices], -1)
