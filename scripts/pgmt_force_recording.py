"""Opt-in, single-env physics-step recording for deterministic PGMT playback.

The scene hook runs AFTER buffer invalidation and BEFORE reward/reset/command
advancement. Sampling sensors does not change the policy or physics. References
are the commands held over that control interval, not interpolated mocap poses.
"""
from pathlib import Path

import numpy as np
import torch


class FootForceRecorder:
    def __init__(self, env, state, output, *, full_state=False):
        if env.num_envs != 1:
            raise ValueError("Force playback recording requires one environment")
        self.env, self.state, self.output = env, state, Path(output)
        self.full_state = full_state
        self.rows, self.pending = [], None
        self.original_update = env.scene.update

        def update(dt):
            self.original_update(dt)
            if self.pending is not None:
                self.sample(dt)

        env.scene.update = update

    def begin(self, context):
        if self.pending is not None:
            raise RuntimeError("Previous force recording step is unfinished")
        command = self.state.command
        ref = self.state.reference()
        labels = self.state.reference_contacts()
        feet = self.state.foot_body_indices
        self.pending = dict(
            episode=context["episode"], terrain=context["terrain"], level=context["level"],
            direction=context["direction"], frame=int(command.frame_ids[0]),
            episode_step=int(self.env.episode_length_buf[0]), rows=[],
            reference_pos_w=ref["body_pos"][0, feet].clone(),
            reference_vel_w=ref["body_lin_vel"][0, feet].clone(),
            reference_quat_w=ref["body_quat"][0, feet].clone(),
            reference_contact=labels["contacts"][0].clone(),
            reference_valid=labels["contact_valid"][0].clone(),
        )

    def sample(self, dt):
        if not np.isclose(dt, self.env.physics_dt):
            raise RuntimeError("Unexpected scene update inside recorded physics step")
        pending = self.pending
        if len(pending["rows"]) >= self.env.cfg.decimation:
            raise RuntimeError("Extra scene update inside recorded control step")
        robot = self.env.scene["robot"]
        feet = self.state.foot_body_indices
        from pmt_tasks.pgmt_runtime import _tensor
        pos = _tensor(robot.data.body_pos_w)[0, feet].clone()
        refpos = pending["reference_pos_w"]
        heights = self.env.scene.terrain.height(torch.cat((pos[:, :2], refpos[:, :2])))
        sample = dict(
            net_forces_w=self.state.robot_foot_contact_forces()[0].clone(),
            terrain_forces_w=self.state.robot_foot_terrain_forces()[0].clone(),
            foot_pos_w=pos, foot_vel_w=_tensor(robot.data.body_lin_vel_w)[0, feet].clone(),
            foot_quat_w=_tensor(robot.data.body_quat_w)[0, feet].clone(),
            ground_z_w=heights[:2].clone(), reference_ground_z_w=heights[2:].clone(),
        )
        if self.full_state:
            sample.update(joint_pos=_tensor(robot.data.joint_pos)[0].clone(),
                          body_pos_w=_tensor(robot.data.body_pos_w)[0].clone(),
                          body_quat_w=_tensor(robot.data.body_quat_w)[0].clone())
        pending["rows"].append(sample)

    def finish(self, done):
        pending = self.pending
        if pending is None or len(pending["rows"]) != self.env.cfg.decimation:
            raise RuntimeError("Missing physics samples")
        frozen = self.state.contact_state(for_reward=True)
        if int(frozen["reference_frame_ids"][0]) != pending["frame"]:
            raise RuntimeError("Recorded reference is not the settled reward reference")
        # Also checks the terminal sample: read the frozen old episode, not the
        # reset robot. Earlier substeps survive automatic resets in our buffer.
        for key in ("net_forces_w", "terrain_forces_w"):
            torch.testing.assert_close(pending["rows"][-1][key], frozen[key][0], equal_nan=True)
        for substep, sample in enumerate(pending.pop("rows")):
            self.rows.append(dict(pending, **sample, substep=substep, done=bool(done),
                                  time_s=(pending["episode_step"] * self.env.step_dt
                                          + (substep + 1) * self.env.physics_dt)))
        self.pending = None

    def close(self):
        self.env.scene.update = self.original_update
        if not self.rows:
            return
        arrays = {}
        for key in self.rows[0]:
            values = [r[key] for r in self.rows]
            arrays[key] = (torch.stack(values).detach().cpu().numpy()
                           if isinstance(values[0], torch.Tensor) else np.asarray(values))
        np.savez_compressed(self.output / "foot_forces.npz", **arrays)
