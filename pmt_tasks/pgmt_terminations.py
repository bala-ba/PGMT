"""Stage2 time limit and tracking failures using the placed reference.

These functions only read the current pre-reset command/physical state. They
do not change reference placement, command frames, contact caches or outcomes.
"""
import math

import torch


def pgmt_time_out(env, command_name="motion", boundary_tolerance_steps=2):
    """Retain the hard time cap; include completed clips near its boundary.

    Tolerance is measured in control intervals, not physics substeps or a
    floating-point accumulated clock. It must never interrupt an unfinished
    clip early: Isaac checks termination before advancing the command, and
    initial versus automatic resets differ by one command tick.
    """
    if (isinstance(boundary_tolerance_steps, bool)
            or not isinstance(boundary_tolerance_steps, int) or boundary_tolerance_steps < 0):
        raise ValueError("boundary_tolerance_steps must be a nonnegative integer")
    steps = env.episode_length_buf
    hard_limit = env.max_episode_length
    if boundary_tolerance_steps == 0:
        return steps >= hard_limit
    command = env.command_manager.get_term(command_name)
    finished = command.frame_ids >= command.data_store.motion_lengths[command.motion_ids] - 1
    near_limit = steps >= max(1, hard_limit - boundary_tolerance_steps)
    return (steps >= hard_limit) | (near_limit & finished)


class ConsecutiveFailureTimer:
    """Per-environment consecutive control-step failures, counted once per step.

    Reset marks the current step as consumed: a diagnostic read immediately
    after reset cannot count an interval in the newly initialized episode.
    """

    def __init__(self, num_envs, *, dt, duration_s, device, initial_step=0):
        if not math.isfinite(dt) or dt <= 0 or not math.isfinite(duration_s) or duration_s < 0:
            raise ValueError("Failure duration must be finite/nonnegative and dt finite/positive")
        self.required_steps = max(1, math.ceil(duration_s / dt - 1e-12))
        self.counts = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.last_step = torch.full_like(self.counts, initial_step)

    def update(self, failed, step):
        fresh = self.last_step < step
        consecutive = self.last_step == step - 1
        count = torch.where(failed, torch.where(consecutive, self.counts + 1, 1), 0)
        self.counts.copy_(torch.where(fresh, count.clamp_max(self.required_steps), self.counts))
        self.last_step.masked_fill_(fresh, step)
        return self.counts >= self.required_steps

    def reset(self, env_ids=None, *, step):
        ids = slice(None) if env_ids is None else env_ids
        self.counts[ids] = 0
        self.last_step[ids] = step


def pgmt_root_xy_failure(env, command_name="motion", threshold=1.0, root_body="pelvis"):
    """Euclidean XY error against the fixed, episode-placed root trajectory."""
    command = env.command_manager.get_term(command_name)
    root = command.cfg.body_names.index(root_body)
    delta = command.robot_body_pos_w[:, root, :2] - command.body_pos_w[:, root, :2]
    error = torch.linalg.vector_norm(delta, dim=-1)
    return ~torch.isfinite(error) | (error > threshold)


def pgmt_torso_height_failure(env, command_name="motion", threshold=0.4, torso_body="torso_link"):
    """Compare torso Z with terrain(actual torso XY) + ORIGINAL motion torso Z.

    Source Z is measured above the motion's original zero-height floor. Never
    add the episode's placement translation/origin Z a second time. Missing
    outside-map heights are left to map_out_of_bounds truncation; missing rays
    inside the valid map are an error, not an assumed zero-height floor.
    """
    command = env.command_manager.get_term(command_name)
    torso = command.cfg.body_names.index(torso_body)
    actual = command.robot_body_pos_w[:, torso]
    source_height = command.source_body_pos[:, torso, 2]
    ground = command.terrain.query(actual[:, :2])
    if torch.any(ground["tile_valid"] & ~ground["valid"]):
        raise RuntimeError("PGMT torso termination: missing clean height inside the valid terrain map")
    error = (actual[:, 2] - (ground["height"] + source_height)).abs()
    invalid_pose = ~torch.isfinite(actual).all(-1) | ~torch.isfinite(source_height)
    return invalid_pose | (ground["valid"] & (error > threshold))


def pgmt_wrist_height_failure(
    env, command_name="motion", threshold=0.4, root_body="pelvis",
    body_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
):
    """Either wrist's relative-root Z differs from the original motion by > limit."""
    command = env.command_manager.get_term(command_name)
    root = command.cfg.body_names.index(root_body)
    wrists = [command.cfg.body_names.index(name) for name in body_names]
    actual_z, source_z = command.robot_body_pos_w[..., 2], command.source_body_pos[..., 2]
    actual_relative = actual_z[:, wrists] - actual_z[:, root, None]
    source_relative = source_z[:, wrists] - source_z[:, root, None]
    error = (actual_relative - source_relative).abs()
    return (~torch.isfinite(error) | (error > threshold)).any(-1)
