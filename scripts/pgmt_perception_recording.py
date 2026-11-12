"""Capture exact pre-action policy perception and its physical pose, without resampling."""
from pathlib import Path
import numpy as np
import torch


class PerceptionRecorder:
    def __init__(self, env, state, policy, output):
        self.env, self.state, self.policy = env, state, policy
        self.output = Path(output)
        self.rows = []

    def capture(self, obs):
        from pmt_tasks.pgmt_math import yaw_quat, rotate
        from pmt_tasks.pgmt_runtime import _tensor
        robot = self.env.scene['robot']
        sensor = self.env.scene.sensors['height_scanner']
        debug = self.policy.get_perception_debug(obs)
        if debug.glimpse is None:
            raise ValueError('Perception visualization requires terrain-enabled policy')
        g = debug.glimpse
        torso = _tensor(robot.data.body_pos_w)[0, self.state.anchor]
        quat = yaw_quat(_tensor(robot.data.body_quat_w)[0, self.state.anchor])
        def world(xy, lift):
            local = torch.cat((xy, torch.zeros_like(xy[..., :1])), -1)
            points = rotate(quat.expand(*local.shape[:-1], 4), local) + torso
            z = self.env.scene.terrain.height(points.reshape(-1,3)[:, :2]).reshape(points.shape[:-1])
            points[..., 2] = torch.where(torch.isfinite(z), z + lift, torso[2])
            return points
        centers = g.center_xy_m[0]
        corners = centers[:, None] + centers.new_tensor([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2],[-.2,-.2]])
        row = dict(time_s=float(self.env.episode_length_buf[0])*self.env.step_dt,
            input_time_s=float(self.env.episode_length_buf[0])*self.env.step_dt,
            frame=int(self.state.command.frame_ids[0]), terminal=False,
            joint_pos=_tensor(robot.data.joint_pos)[0].clone(),
            body_pos_w=_tensor(robot.data.body_pos_w)[0].clone(),
            body_quat_w=_tensor(robot.data.body_quat_w)[0].clone(),
            torso_pos_w=torso.clone(), torso_yaw_quat=quat.clone(),
            ray_hits_w=sensor.data.ray_hits_w[0].clone(), sensor_pos_w=sensor.data.pos_w[0].clone(),
            terrain_map=obs['terrain_map'][0].clone(), terrain_valid=obs['terrain_valid'][0].clone(),
            center_xy_m=centers.clone(), centers_w=world(centers, .06), borders_w=world(corners, .035),
            patch_height=g.patch_height[0].clone(), patch_valid=g.point_valid[0].clone(),
            token_valid=g.token_valid[0].clone(), coverage=g.coverage[0].clone(),
            attention_weights=debug.attention.weights[0].clone(),
            terrain_attention_mass=debug.terrain_attention_mass[0].clone())
        valid = row['terrain_valid'].bool().flatten()
        expected = (row['sensor_pos_w'][2]-row['ray_hits_w'][:,2]).clamp(-3.,3.)
        torch.testing.assert_close(row['terrain_map'].flatten()[valid], expected[valid], atol=1e-5, rtol=0)
        self.rows.append(row)

    def close(self, terminal_sample):
        if not self.rows:
            return
        # Native reset has already happened. Preserve the old terminal pose from
        # the physics recorder and explicitly hold the final pre-action input.
        last = dict(self.rows[-1])
        last.update(time_s=float(terminal_sample['time_s']), terminal=True)
        for key in ('joint_pos', 'body_pos_w', 'body_quat_w'):
            last[key] = terminal_sample[key].clone()
        self.rows.append(last)
        arrays = {}
        for key in self.rows[0]:
            values = [row[key] for row in self.rows]
            arrays[key] = torch.stack(values).cpu().numpy() if isinstance(values[0],torch.Tensor) else np.asarray(values)
        np.savez_compressed(self.output/'perception.npz', **arrays)
