"""Equivalent reset geometry, local reference updates and indexed buffer clearing."""
from types import MethodType
import torch
from pmt_tasks.pgmt_math import future_offsets


def fill_rows(tensor, ids, value):
    if isinstance(ids, torch.Tensor) and ids.dtype==torch.long and ids.ndim==1:
        tensor.index_fill_(0, ids, value)
    else:
        tensor[ids]=value


def reset_reward_sums(self, env_ids=None):
    ids=slice(None) if env_ids is None else env_ids
    extras={}
    for name, value in self._episode_sums.items():
        extras['Episode_Reward/'+name]=value[ids].mean()/self._env.max_episode_length_s
        fill_rows(value, ids, 0.)
    for cfg in self._class_term_cfgs:
        cfg.func.reset(env_ids=ids)
    return extras


def reset_history(self, batch_ids=None):
    ids=slice(None) if batch_ids is None else batch_ids
    fill_rows(self._num_pushes, ids, 0)
    if self._buffer is not None:
        if isinstance(ids, torch.Tensor) and ids.dtype==torch.long:
            self._buffer.index_fill_(1, ids, 0.)
        else:
            self._buffer[:,ids,:]=0.


def install_fast_placement(command):
    env=command._env
    if command.full_reference_store.storage_device != torch.device(command.device):
        raise ValueError('Fast placement requires the complete motion store on the simulator device')
    command._pgmt_fast_placement=True
    command._pgmt_reset_ids=None
    # Cache source data only. Random placement and mesh checks are still evaluated
    # on every reset. Other start-frame modes use live indexed source data.
    if command.cfg.fixed_start_frame==0:
        mids=torch.arange(command.full_reference_store.num_motions,device=command.device)
        command._pgmt_start_reference=command._source_reference(mids,torch.zeros_like(mids))
        offsets=future_offsets(device=command.device)
        frames=torch.minimum(offsets[None],command.full_reference_store.motion_lengths[:,None]-1)
        command._pgmt_start_future_positions=command.full_reference_store._stacked_body_pos_w[mids[:,None],frames].clone()

    original_resample=command._resample_command
    original_relative=command._compute_relative_poses
    original_metrics=command._update_metrics

    def resample(ids):
        command._pgmt_reset_ids=ids
        try:
            return original_resample(ids)
        finally:
            command._pgmt_reset_ids=None

    def relative():
        ids=command._pgmt_reset_ids
        if ids is None or isinstance(ids,slice) or len(ids)==command.num_envs:
            return original_relative()
        from isaaclab.utils.math import yaw_quat, quat_mul, quat_inv, quat_apply
        nb=len(command.cfg.body_names)
        anchor=command.anchor_pos_w[ids,None].repeat(1,nb,1)
        anchor_q=command.anchor_quat_w[ids,None].repeat(1,nb,1)
        physical=command.robot_anchor_pos_w[ids,None].repeat(1,nb,1)
        physical_q=command.robot_anchor_quat_w[ids,None].repeat(1,nb,1)
        delta=anchor-physical
        delta[...,:2]=0
        orientation=yaw_quat(quat_mul(physical_q,quat_inv(anchor_q)))
        command.body_quat_relative_w[ids]=quat_mul(orientation,command.body_quat_w[ids])
        command.body_pos_relative_w[ids]=physical+delta+quat_apply(orientation,command.body_pos_w[ids]-anchor)

    def metrics():
        ids=command._pgmt_reset_ids
        if ids is None or isinstance(ids,slice) or len(ids)==command.num_envs:
            return original_metrics()
        from isaaclab.utils.math import quat_error_magnitude
        values=dict(
            error_anchor_pos=(command.anchor_pos_w[ids]-command.robot_anchor_pos_w[ids]).norm(dim=-1),
            error_anchor_rot=quat_error_magnitude(command.anchor_quat_w[ids],command.robot_anchor_quat_w[ids]),
            error_body_pos=(command.body_pos_relative_w[ids]-command.robot_body_pos_w[ids]).norm(dim=-1).mean(-1),
            error_body_rot=quat_error_magnitude(command.body_quat_relative_w[ids],command.robot_body_quat_w[ids]).mean(-1),
            error_joint_pos=(command.joint_pos[ids]-command.robot_joint_pos[ids]).norm(dim=-1))
        for name,value in values.items():
            command.metrics[name][ids]=value
        for name,value in command.sampler.get_metrics().items():
            if not isinstance(value,torch.Tensor):
                value=torch.as_tensor(value,device=command.device)
            command.metrics[name][ids]=value.expand(len(ids))

    command._resample_command=resample
    command._compute_relative_poses=relative
    command._update_metrics=metrics
    env._pgmt_state._fast_reset_buffers=True
    if env._pgmt_state.contact_tracker is not None:
        env._pgmt_state.contact_tracker._fast_reset_buffers=True
    env.reward_manager.reset=MethodType(reset_reward_sums,env.reward_manager)
    for histories in env.observation_manager._group_obs_term_history_buffer.values():
        for history in histories.values():
            history.reset=MethodType(reset_history,history)
    return dict(source_frame_cache=command.cfg.fixed_start_frame==0, geometry_only=True,
                local_reference_updates=True, indexed_buffer_clears=True)
