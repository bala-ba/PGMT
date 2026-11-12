"""Perception videos must use stored policy inputs and old-episode terminal poses."""
from types import SimpleNamespace as NS
import numpy as np
import torch
from scripts.pgmt_perception_recording import PerceptionRecorder


class Scene(dict):
    pass


def fixture(tmp_path):
    scene=Scene()
    pos=torch.tensor([[[0.,0.,.6],[0.,0.,1.]]])
    quat=torch.tensor([[[1.,0.,0.,0.],[2**-.5,0.,0.,2**-.5]]])
    scene['robot']=NS(data=NS(body_pos_w=pos,body_quat_w=quat,joint_pos=torch.zeros(1,2)))
    hits=torch.zeros(1,441,3)
    scene.sensors={'height_scanner':NS(data=NS(ray_hits_w=hits,pos_w=torch.tensor([[0.,0.,1.]])))}
    scene.terrain=NS(height=lambda xy:torch.zeros(len(xy)))
    glimpse=NS(center_xy_m=torch.tensor([[[.5,0.],[0.,.5],[-.5,0.],[0.,-.5]]]),
        patch_height=torch.ones(1,4,5,5)/3,point_valid=torch.ones(1,4,5,5,dtype=torch.bool),
        token_valid=torch.ones(1,4,dtype=torch.bool),coverage=torch.ones(1,4))
    debug=NS(glimpse=glimpse,attention=NS(weights=torch.ones(1,4,1,13)/13),terrain_attention_mass=torch.ones(1,4,1)*4/13)
    policy=NS(get_perception_debug=lambda obs:debug)
    state=NS(anchor=1,command=NS(frame_ids=torch.tensor([0])))
    env=NS(scene=scene,episode_length_buf=torch.tensor([0]),step_dt=.02)
    obs={'terrain_map':torch.ones(1,21,21),'terrain_valid':torch.ones(1,21,21)}
    return PerceptionRecorder(env,state,policy,tmp_path),obs,env


def test_capture_preserves_exact_inputs_and_rotates_glimpse_into_world(tmp_path):
    recorder,obs,env=fixture(tmp_path)
    recorder.capture(obs)
    row=recorder.rows[0]
    torch.testing.assert_close(row['centers_w'][0],torch.tensor([0.,.5,.06]),atol=1e-6,rtol=0)
    obs['terrain_map'].fill_(2)
    env.scene.sensors['height_scanner'].data.ray_hits_w.fill_(7)
    assert (row['terrain_map']==1).all()
    assert (row['ray_hits_w']==0).all()


def test_terminal_pose_is_old_episode_and_input_has_explicit_timestamp(tmp_path):
    recorder,obs,env=fixture(tmp_path)
    recorder.capture(obs)
    terminal=dict(time_s=.02,joint_pos=torch.tensor([.1,.2]),
        body_pos_w=env.scene['robot'].data.body_pos_w[0].clone()+.01,
        body_quat_w=env.scene['robot'].data.body_quat_w[0].clone())
    env.scene['robot'].data.body_pos_w.fill_(999)  # Simulate native automatic reset.
    recorder.close(terminal)
    with np.load(tmp_path/'perception.npz') as d:
        np.testing.assert_allclose(d['time_s'],[0.,.02])
        np.testing.assert_array_equal(d['input_time_s'],[0.,0.])
        np.testing.assert_array_equal(d['terminal'],[False,True])
        np.testing.assert_array_equal(d['body_pos_w'][-1],terminal['body_pos_w'].numpy())
        assert (d['terrain_map']==1).all()
