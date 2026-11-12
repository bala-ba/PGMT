"""Terrain-placed PGMT reference command; source joint trajectories stay intact."""
from pathlib import Path
import hashlib

import torch
from isaaclab.utils import configclass

from .multi_motion_command import MotionDataStore, MultiMotionCommandV2, MultiMotionCommandV2Cfg, SamplingResult
from pmt_tasks.pgmt_math import future_offsets
from pmt_tasks.pgmt_contact_reference import ContactLabelStore
from pmt_tasks.pgmt_sampling import load_terrain_compatibility, sample_motion_starts
from pmt_tasks.pgmt_emd_sampling import EMDMotionSampler, sample_emd_starts
from pmt_tasks.pgmt_terrain_layout import centered_patch_ids
from pmt_tasks.pgmt_reference import (collision_sphere_cover, initial_transform,
                                      place_on_surface, transform_reference, transform_positions)

REFERENCE_KEYS = ("joint_pos", "joint_vel", "body_pos", "body_quat", "body_lin_vel", "body_ang_vel")


class PGMTTerrainMotionCommand(MultiMotionCommandV2):
    def __init__(self, cfg, env):
        if cfg.sampler_type not in ("adaptive", "emd") or cfg.enable_frame_sampling:
            raise ValueError("PGMT requires motion-only adaptive or EMD sampling")
        if cfg.sampler_type == "emd" and cfg.require_valid_support_start:
            raise ValueError("EMD sampling requires uniform time without foot-support filtering")
        if cfg.chunk_length or cfg.use_fp16 or cfg.default_reset_joint_prob or cfg.force_start_frame_zero:
            raise ValueError("PGMT placement requires unchunked float32 segments and an unmodified reference pose")
        if cfg.pose_range or cfg.velocity_range or cfg.joint_position_range != (0.0, 0.0):
            raise ValueError("Use shared placement_xy_range/yaw_range, not robot-only reset noise")
        if cfg.placement_xy_mode not in ("center", "random"):
            raise ValueError(f"Unknown placement XY mode: {cfg.placement_xy_mode!r}")
        super().__init__(cfg, env)
        if cfg.body_names[0] != "pelvis":
            raise ValueError("The tracked root must be pelvis")
        self.terrain = env.scene.terrain
        self.placement_mode = ("patch" if self.terrain.has_patches else "origin") if cfg.placement_mode == "auto" else cfg.placement_mode
        if self.placement_mode not in ("patch", "origin") or (self.placement_mode == "patch" and not self.terrain.has_patches):
            raise ValueError("Patch placement requires a repeated-patch terrain asset")
        self.spawn_patch_ids = None
        if cfg.placement_patch_grid is not None:
            if self.placement_mode != "patch":
                raise ValueError("placement_patch_grid requires patch placement; use None for legacy origin placement")
            self.spawn_patch_ids = centered_patch_ids(
                self.terrain.manifest["spec"]["patch_grid"], cfg.placement_patch_grid, device=self.device)
        self.reference_rotation = torch.zeros(self.num_envs, 4, device=self.device)
        self.reference_rotation[:, 0] = 1
        self.reference_translation = torch.zeros(self.num_envs, 3, device=self.device)
        self._world_tracked = None
        self.placement_min_gap = torch.zeros(self.num_envs, device=self.device)
        self.placement_attempts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.placement_patch_ids = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.placement_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.placement_yaw = torch.zeros(self.num_envs, device=self.device)
        ids, centres, radii = collision_sphere_cover(cfg.robot_urdf, list(self.robot.body_names))
        self.collision_body_ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        self.collision_local_centres = torch.as_tensor(centres, dtype=torch.float32, device=self.device)
        self.collision_radii = torch.as_tensor(radii, dtype=torch.float32, device=self.device)
        lengths = self.data_store.motion_lengths
        self.valid_starts = torch.zeros(self.data_store.num_motions, int(lengths.max()), dtype=torch.bool, device=self.device)
        for i, motion in enumerate(self.data_store.motions):
            metadata = motion.metadata
            if metadata["joint_names"].tolist() != list(self.robot.joint_names) or metadata["body_names"].tolist() != list(self.robot.body_names):
                raise ValueError(f"Motion joint/body order differs from the robot: {motion.source_file}")
            if cfg.require_valid_support_start:
                contact = torch.as_tensor(metadata["contacts"], device=self.device).bool()
                valid = torch.as_tensor(metadata["contact_valid"], device=self.device).bool()
                self.valid_starts[i, :motion.num_frames] = valid.all(-1) & contact.any(-1)
            else:
                # Only the actual frame range matters when support filtering is
                # disabled; crawling/airborne/unknown-foot labels do not veto a
                # reset. Keep padded frames excluded for variable-length clips.
                self.valid_starts[i, :motion.num_frames] = True
        self.terrain_families = tuple(self.terrain.manifest["spec"]["families"])
        self.compatible_motions = load_terrain_compatibility(
            cfg.motion_terrain_compatibility, [m.source_file for m in self.data_store.motions],
            families=self.terrain.compatibility_families, device=self.device)
        self.compatibility_sha256 = hashlib.sha256(Path(cfg.motion_terrain_compatibility).read_bytes()).hexdigest()
        self.unavailable_reset_motions = [m.source_file for i, m in enumerate(self.data_store.motions)
                                         if not self.valid_starts[i].any()]
        if self.unavailable_reset_motions:
            print("[PGMT] No valid support reset frame; retaining data but excluding from reset sampling: "
                  + ", ".join(Path(file).name for file in self.unavailable_reset_motions))
        counts = (self.compatible_motions & self.valid_starts.any(-1)[None]).sum(-1).tolist()
        print(f"[PGMT] Motion-level {cfg.sampler_type} sampling; frame adaptation disabled; "
              f"placement_xy_mode={cfg.placement_xy_mode}; "
              f"fixed_start_frame={cfg.fixed_start_frame}; "
              f"require_valid_support_start={cfg.require_valid_support_start}; "
              f"eligible motions per terrain: {dict(zip(self.terrain_families, counts))}; "
              f"compatibility SHA256: {self.compatibility_sha256}")

    def _init_data_store(self):
        super()._init_data_store()
        self.full_reference_store = MotionDataStore(device=self.device, storage_device=self.cfg.storage_device,
                                                    use_fp16=False, chunk_length=0)
        self.full_reference_store.load_files(self.cfg.motion_files, list(range(self.robot.num_bodies)), load_human=False)
        # Training labels are independent of require_valid_support_start. In
        # particular, disabling that reset gate must not erase invalid masks.
        self.contact_label_store = ContactLabelStore(self.data_store.motions, device=self.device)

    def _create_sampler(self):
        if self.cfg.sampler_type == "emd":
            return EMDMotionSampler(self.data_store.num_motions, self.device, self.cfg.emd_sampling)
        return super()._create_sampler()

    def _draw_samples(self, env_ids):
        if self.cfg.sampler_type == "emd":
            return SamplingResult(*sample_emd_starts(
                self.sampler.weights, self.compatible_motions, self.data_store.motion_lengths,
                self.terrain.terrain_types[env_ids], fixed_start_frame=self.cfg.fixed_start_frame))
        motions, frames = sample_motion_starts(
            self.sampler.get_failure_probabilities(), self.compatible_motions, self.valid_starts,
            self.terrain.terrain_types[env_ids], beta=self.sampler.beta,
            uniform_ratio=self.sampler.uniform_ratio, fixed_start_frame=self.cfg.fixed_start_frame,
            families=self.terrain_families)
        return SamplingResult(motions, frames)

    def _record_outcomes(self, env_ids, terminated):
        curriculum = getattr(self._env, "_pgmt_terrain_curriculum", None)
        if curriculum is not None:
            # Only genuine settled episodes update the motion sampler. Initial,
            # manual and checkpoint-restore resets are not successes/failures.
            valid = curriculum.sampler_pending[env_ids] & (curriculum.settled_step[env_ids] == self._env.common_step_counter)
            env_ids, terminated = env_ids[valid], terminated[valid]
            curriculum.sampler_pending[env_ids] = False
        super()._record_outcomes(env_ids, terminated)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        curriculum = getattr(self._env, "_pgmt_terrain_curriculum", None)
        if curriculum is not None:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            curriculum.begin(ids, self._env.common_step_counter, self.motion_ids[ids], self.frame_ids[ids],
                             self.data_store.motion_lengths[self.motion_ids[ids]])

    def transform_reference(self, reference, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        return transform_reference(reference, self.reference_rotation[ids], self.reference_translation[ids])

    def _tracked_reference(self):
        if self._world_tracked is None:
            self._world_tracked = self.transform_reference(dict(zip(REFERENCE_KEYS, (
                self._joint_pos_buf, self._joint_vel_buf, self._body_pos_w_buf, self._body_quat_w_buf,
                self._body_lin_vel_w_buf, self._body_ang_vel_w_buf))))
        return self._world_tracked

    def _update_motion_buffers(self):
        super()._update_motion_buffers()
        self._world_tracked = None

    def _update_motion_buffers_for_envs(self, env_ids):
        super()._update_motion_buffers_for_envs(env_ids)
        self._world_tracked = None

    @property
    def body_pos_w(self):
        return self._tracked_reference()["body_pos"]

    @property
    def source_body_pos(self):
        """Current tracked motion positions before placement; read-only by convention.

        Shares the same motion/frame buffer as body_pos_w. In particular source
        Z excludes terrain origin, placement clearance and episode translation.
        """
        return self._body_pos_w_buf

    @property
    def body_quat_w(self):
        return self._tracked_reference()["body_quat"]

    @property
    def body_lin_vel_w(self):
        return self._tracked_reference()["body_lin_vel"]

    @property
    def body_ang_vel_w(self):
        return self._tracked_reference()["body_ang_vel"]

    @property
    def robot_body_lin_vel_w(self):
        return self.robot.data.body_link_lin_vel_w[:, self.body_indices]

    def _source_reference(self, motion_ids, frames):
        return dict(zip(REFERENCE_KEYS, (a.float() for a in self.full_reference_store.get_motion_data(motion_ids, frames))))

    def _reset_robot_state(self, env_ids):
        fast = getattr(self, '_pgmt_fast_placement', False)
        cached = getattr(self, '_pgmt_start_reference', None)
        use_cached = (cached is not None and self.cfg.fixed_start_frame == 0
                      and getattr(self, '_pgmt_reset_ids', None) is not None)
        reference = ({k: v[self.motion_ids[env_ids]] for k, v in cached.items()} if use_cached
                     else self._source_reference(self.motion_ids[env_ids], self.frame_ids[env_ids]))
        limits = self.robot.data.joint_pos_limits[env_ids]
        qpos = reference["joint_pos"]
        if ((qpos < limits[..., 0] - 1e-5) | (qpos > limits[..., 1] + 1e-5)).any():
            raise ValueError("Selected source pose exceeds physical joint limits; it will not be silently clamped")
        pending = torch.arange(len(env_ids), device=self.device)
        for attempt in range(self.cfg.max_placement_attempts):
            selected_ids = env_ids[pending]
            ref = {key: value[pending] for key, value in reference.items()}
            if self.placement_mode == "patch":
                xy, bounds, patch_ids = self.terrain.sample_patch_positions(
                    selected_ids, self.cfg.boundary_margin, allowed_patch_ids=self.spawn_patch_ids,
                    xy_mode=self.cfg.placement_xy_mode)
            else:
                if self.cfg.placement_xy_mode == "center":
                    xy = self._env.scene.env_origins[selected_ids, :2].clone()
                else:
                    low, high = self.cfg.placement_xy_range
                    xy = self._env.scene.env_origins[selected_ids, :2] + low + (high - low) * torch.rand(len(pending), 2, device=self.device)
                bounds = self.terrain.bounds_for(selected_ids)
                patch_ids = torch.full_like(selected_ids, -1)
            low, high = self.cfg.placement_yaw_range
            yaw = low + (high - low) * torch.rand(len(pending), device=self.device)
            rotation, translation = initial_transform(ref["body_pos"][:, 0], ref["body_quat"][:, 0], xy, yaw)
            translation, valid, gap = place_on_surface(
                ref, rotation, translation, bounds, self.terrain, self.collision_body_ids,
                self.collision_local_centres, self.collision_radii, clearance=self.cfg.spawn_clearance,
                boundary_margin=self.cfg.boundary_margin, geometry_only=fast)
            offsets = future_offsets(device=self.device)
            mids = self.motion_ids[selected_ids, None].expand(-1, len(offsets))
            frames = self.frame_ids[selected_ids, None] + offsets
            frames = torch.minimum(frames, self.data_store.motion_lengths[mids] - 1)
            if fast:
                cached_future = getattr(self, '_pgmt_start_future_positions', None)
                if cached_future is not None and use_cached:
                    points = cached_future[self.motion_ids[selected_ids]]
                else:
                    # Full reference store is on the simulator device in Stage2.
                    points = self.full_reference_store._stacked_body_pos_w[mids, frames].float()
                future_positions = transform_positions(points, rotation, translation)
            else:
                future = self._source_reference(mids.reshape(-1), frames.reshape(-1))
                future = {k: v.reshape(len(pending), len(offsets), *v.shape[1:]) for k, v in future.items()}
                future_positions = transform_reference(future, rotation, translation)['body_pos']
            # Future frames may enter neighboring tiles. Only the initial pose
            # must fit its spawn tile; the future window must fit the full map.
            valid &= self.terrain.contains(future_positions[..., :2]).flatten(1).all(-1)
            chosen = valid.nonzero().flatten() if fast else valid
            accepted = selected_ids[chosen]
            self.reference_rotation[accepted] = rotation[chosen]
            self.reference_translation[accepted] = translation[chosen]
            self.placement_min_gap[accepted] = gap[chosen]
            self.placement_attempts[accepted] = attempt + 1
            self.placement_patch_ids[accepted] = patch_ids[chosen]
            self.placement_xy[accepted] = xy[chosen]
            self.placement_yaw[accepted] = yaw[chosen]
            pending = pending[~valid]
            if not len(pending):
                break
        if len(pending):
            raise RuntimeError(f"No nonpenetrating, in-bounds rigid placement for envs {env_ids[pending].tolist()}")
        self._world_tracked = None
        placed = self.transform_reference(reference, env_ids)
        self.robot.write_joint_state_to_sim(placed["joint_pos"], placed["joint_vel"], env_ids=env_ids)
        self.robot.write_root_link_state_to_sim(torch.cat((placed["body_pos"][:, 0], placed["body_quat"][:, 0],
            placed["body_lin_vel"][:, 0], placed["body_ang_vel"][:, 0]), dim=-1), env_ids=env_ids)
        self.robot.set_joint_position_target(placed["joint_pos"], env_ids=env_ids)
        # The base reset immediately computes robot-relative reference buffers.
        # Flush kinematics first so they use this new root/joint pose, not the
        # previous episode's link transforms. This does not advance physics.
        self._env.sim.forward()

    def _update_command(self):
        # Finish through the environment termination manager, never teleport
        # to another reference inside a running episode.
        self.frame_ids = torch.minimum(self.frame_ids + 1, self.data_store.motion_lengths[self.motion_ids] - 1)
        self.sampler.step()
        self._update_motion_buffers()
        self._compute_relative_poses()
        self._maybe_update_action_offset(slice(None))
        state = getattr(self._env, "_pgmt_state", None)
        if state is not None:
            state.invalidate_observations()


def pgmt_motion_finished(env, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    return command.frame_ids >= command.data_store.motion_lengths[command.motion_ids] - 1


def pgmt_map_out_of_bounds(env, command_name="motion"):
    """Immediately truncate outside the map; crossing spawn-tile edges is legal."""
    command = env.command_manager.get_term(command_name)
    reference_inside = command.terrain.contains(command.body_pos_w[..., :2]).all(-1)
    robot_inside = command.terrain.contains(command.robot_body_pos_w[..., :2]).all(-1)
    return ~(reference_inside & robot_inside)


@configclass
class PGMTTerrainMotionCommandCfg(MultiMotionCommandV2Cfg):
    class_type: type = PGMTTerrainMotionCommand
    robot_urdf: str = str(Path(__file__).resolve().parents[2] / "assets/unitree_description/urdf/g1/main.urdf")
    placement_mode: str = "auto"  # Select an origin (v5) or a patch (v6+).
    placement_xy_mode: str = "random"  # Stage2 uses "center"; legacy P1 retains random XY.
    placement_patch_grid: tuple[int, int] | None = None  # None: all patches; (2, 2): central four of v6.
    placement_xy_range: tuple[float, float] = (-0.25, 0.25)
    placement_yaw_range: tuple[float, float] = (-3.141592653589793, 3.141592653589793)
    spawn_clearance: float = 0.003
    boundary_margin: float = 0.25  # Initial placement only; no margin at internal tile edges during rollout.
    max_placement_attempts: int = 32
    # Temporary stage-2 reset policy: frame 0 without a foot-support gate.
    # Set -1 / True respectively to restore random valid-support starts.
    fixed_start_frame: int = 0
    require_valid_support_start: bool = False
    chunk_length: int = 0
    use_fp16: bool = False
    load_human_motion: bool = False
    sampler_type: str = "adaptive"
    enable_frame_sampling: bool = False
    emd_sampling: dict | None = None
    motion_terrain_compatibility: str = str(Path(__file__).resolve().parents[3]
        / "assets/motions/pgmt_clipped_50hz_tcrs_v1/motion_terrain_compatibility.csv")
    pose_range: dict = {}
    velocity_range: dict = {}
    joint_position_range: tuple[float, float] = (0.0, 0.0)
