"""Recover the episode's constant reference transform from recorded targets.

The source file and frame IDs are exact. The shared rigid transform is solved
from one recorded foot pose, then checked against BOTH feet over ALL samples.
No terrain fitting, per-frame height correction, or simulator is used.
"""
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def rotations(wxyz):
    return Rotation.from_quat(np.roll(wxyz, -1, axis=-1))


def recover_reference(source, recorded, frames, body_names, joint_names):
    if source['body_names'].tolist() != body_names or source['joint_names'].tolist() != joint_names:
        raise ValueError('Reference source body/joint ordering differs from recording')
    feet = [body_names.index(n) for n in ('left_ankle_roll_link', 'right_ankle_roll_link')]
    saved_frames = recorded['frame'].astype(np.int64)
    frames = np.asarray(frames, dtype=np.int64)
    if min(saved_frames.min(), frames.min()) < 0 or max(saved_frames.max(),frames.max()) >= len(source['joint_pos']):
        raise ValueError('Reference frame outside original motion')
    first = saved_frames[0]
    rotation = (rotations(recorded['reference_quat_w'][0,0]) *
                rotations(source['body_quat_w'][first,feet[0]]).inv())
    translation = recorded['reference_pos_w'][0,0] - rotation.apply(source['body_pos_w'][first,feet[0]])
    predicted = rotation.apply(source['body_pos_w'][saved_frames][:,feet].reshape(-1,3)).reshape(-1,2,3)+translation
    error = float(np.linalg.norm(predicted-recorded['reference_pos_w'],axis=-1).max())
    predicted_quats = rotation * rotations(source['body_quat_w'][saved_frames][:,feet].reshape(-1,4))
    angular_error = float((rotations(recorded['reference_quat_w'].reshape(-1,4)).inv()*predicted_quats).magnitude().max())
    if error > .0001 or angular_error > 1e-5:
        raise ValueError(f'Recorded reference is inconsistent with source / constant placement: {error} m, {angular_error} rad')
    positions = rotation.apply(source['body_pos_w'][frames].reshape(-1,3)).reshape(len(frames),len(body_names),3)+translation
    quats = np.roll((rotation*rotations(source['body_quat_w'][frames].reshape(-1,4))).as_quat(),1,axis=-1).reshape(len(frames),len(body_names),4)
    data = dict(reference_body_pos_w=positions, reference_body_quat_w=quats,
                reference_joint_pos=source['joint_pos'][frames].copy())
    proof = dict(source='original motion + constant episode placement recovered from recorded reference foot poses',
        validation='both foot positions and orientations at every recorded physics substep',
        validated_foot_samples=len(saved_frames)*2, foot_position_max_error_m=error,
        foot_orientation_max_error_rad=angular_error,
        rotation_wxyz=np.roll(rotation.as_quat(),1).tolist(), translation_m=translation.tolist(),
        terrain_height_adjustment=False, source_joint_positions_unchanged=True)
    return data, proof


def load_recorded_reference(run, perception):
    path = Path(run['metadata']['motion'])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != run['metadata']['force_recording']['motion_sha256']:
        raise ValueError('Reference source motion changed since simulation')
    names = run['metadata']['video_states']
    with np.load(path, allow_pickle=False) as source:
        data, proof = recover_reference(source,run['data'],perception['frame'],names['body_names'],names['joint_names'])
    proof.update(motion_sha256=digest, motion=str(path.resolve()))
    return data, proof
