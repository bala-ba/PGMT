"""Reference reconstruction must preserve source frames and reject changing placement."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scripts.pgmt_reference_replay import recover_reference


def fixture():
    rng=np.random.default_rng(41)
    names=['pelvis','left_ankle_roll_link','right_ankle_roll_link']
    q=np.roll(Rotation.random(18,random_state=rng).as_quat(),1,axis=-1).reshape(6,3,4)
    source=dict(body_names=np.array(names),joint_names=np.array(['joint']),
        joint_pos=np.arange(6)[:,None].astype(float),body_pos_w=rng.normal(size=(6,3,3)),body_quat_w=q)
    rotation=Rotation.from_euler('z',.73);translation=np.array([18.,-12.,-.4])
    frames=np.repeat(np.arange(6),4)
    p=rotation.apply(source['body_pos_w'][frames][:,1:].reshape(-1,3)).reshape(-1,2,3)+translation
    quat=np.roll((rotation*Rotation.from_quat(np.roll(q[frames][:,1:].reshape(-1,4),-1,axis=-1))).as_quat(),1,axis=-1).reshape(-1,2,4)
    recorded=dict(frame=frames,reference_pos_w=p,reference_quat_w=quat)
    return source,recorded,names,rotation,translation


def test_reference_keeps_exact_frames_joint_positions_and_rigid_height():
    source,recorded,names,rotation,translation=fixture()
    frames=np.array([0,2,5,5]) # Terminal frame deliberately holds the old reference.
    data,proof=recover_reference(source,recorded,frames,names,['joint'])
    expected=rotation.apply(source['body_pos_w'][frames].reshape(-1,3)).reshape(4,3,3)+translation
    np.testing.assert_allclose(data['reference_body_pos_w'],expected,atol=1e-12)
    np.testing.assert_array_equal(data['reference_joint_pos'],source['joint_pos'][frames])
    assert not proof['terrain_height_adjustment']
    assert proof['validated_foot_samples']==48


def test_changed_reference_transform_cannot_be_silently_reconstructed():
    source,recorded,names,_,_=fixture()
    recorded['reference_pos_w'][10,1,2]+=.01
    with pytest.raises(ValueError,match='inconsistent'):
        recover_reference(source,recorded,[0,1],names,['joint'])


def test_body_order_mismatch_is_rejected():
    source,recorded,names,_,_=fixture()
    with pytest.raises(ValueError,match='ordering'):
        recover_reference(source,recorded,[0,1],names[::-1],['joint'])
