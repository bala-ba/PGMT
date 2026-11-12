"""Rigid placement must agree across current/future/full/tracked references."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from pmt_tasks.pgmt_reference import (collision_sphere_cover, initial_transform, in_tile, transform_positions,
                                      place_on_surface, transform_reference, yaw_rotation)
from pmt_tasks.pgmt_runtime import PGMTState
from scripts.pmt_npz_to_mjlab import BFS_BODIES


def reference(n=2, bodies=3):
    torch.manual_seed(8)
    pos = torch.randn(n, bodies, 3, dtype=torch.float64)
    q = torch.zeros(n, bodies, 4, dtype=torch.float64)
    q[..., 0] = 1
    return dict(joint_pos=torch.randn(n, 29, dtype=torch.float64), joint_vel=torch.randn(n, 29, dtype=torch.float64),
                body_pos=pos, body_quat=q, body_lin_vel=torch.randn_like(pos), body_ang_vel=torch.randn_like(pos))


def test_start_frame_anchor_and_absolute_yaw():
    position = torch.tensor([[123., -45., .9], [-4., 9., 1.2]], dtype=torch.float64)
    start = yaw_rotation(torch.tensor([.7, -1.3], dtype=torch.float64))
    target = torch.tensor([[12., 48.], [-108., -24.]], dtype=torch.float64)
    yaw = torch.tensor([np.pi / 2, -.2], dtype=torch.float64)
    q, translation = initial_transform(position, start, target, yaw)
    ref = reference(bodies=1)
    ref["body_pos"], ref["body_quat"] = position[:, None], start[:, None]
    actual = transform_reference(ref, q, translation)
    torch.testing.assert_close(actual["body_pos"][:, 0, :2], target)
    torch.testing.assert_close(actual["body_quat"][:, 0], yaw_rotation(yaw))
    torch.testing.assert_close(actual["body_pos"][:, 0, 2], position[:, 2])


def test_positions_orientations_and_both_velocities_share_rotation_and_joints_unchanged():
    ref = reference()
    original = {k: v.clone() for k, v in ref.items()}
    angles = torch.tensor([np.pi / 2, -np.pi / 3], dtype=torch.float64)
    q = yaw_rotation(angles)
    offset = torch.tensor([[10., 20., -5.], [-8., 2., 3.]], dtype=torch.float64)
    actual = transform_reference(ref, q, offset)
    for i, angle in enumerate(angles.tolist()):
        R = Rotation.from_euler("z", angle)
        for k in ("body_pos", "body_lin_vel", "body_ang_vel"):
            expected = R.apply(ref[k][i].numpy())
            if k == "body_pos":
                expected += offset[i].numpy()
            np.testing.assert_allclose(actual[k][i], expected, atol=1e-12)
    assert actual["joint_pos"] is ref["joint_pos"]
    assert actual["joint_vel"] is ref["joint_vel"]
    for k in ref:
        torch.testing.assert_close(ref[k], original[k])
    future = {k: v[:, None].expand(-1, 8, *v.shape[1:]) for k, v in ref.items()}
    transformed_future = transform_reference(future, q, offset)
    for k in ref:
        torch.testing.assert_close(transformed_future[k][:, 0], actual[k])
        torch.testing.assert_close(transformed_future[k][:, -1], actual[k])


class Plane:
    def __init__(self, height):
        self.z = height

    def height(self, xy):
        return torch.full(xy.shape[:-1], self.z, dtype=xy.dtype, device=xy.device)

    def distance(self, points):
        return (points[..., 2] - self.z).abs()


@pytest.mark.parametrize("ground", [0., 3.4, -4.2])
def test_placement_uses_mesh_height_and_conservative_collision_radius(ground):
    ref = reference(n=1, bodies=1)
    ref["body_pos"][:] = torch.tensor([0., 0., .8])
    q = yaw_rotation(torch.zeros(1, dtype=torch.float64))
    translation = torch.zeros(1, 3, dtype=torch.float64)
    bounds = torch.tensor([[[-12., -12.], [12., 12.]]], dtype=torch.float64)
    body_ids = torch.tensor([0])
    local = torch.tensor([[0., 0., -.76]], dtype=torch.float64)
    radii = torch.tensor([.02], dtype=torch.float64)
    placed, valid, gap = place_on_surface(ref, q, translation, bounds, Plane(ground), body_ids, local, radii)
    assert valid.item()
    assert gap.item() == pytest.approx(.003)
    assert placed[0, 2].item() == pytest.approx(ground - .017)


def test_bounds_include_entire_reference_window_and_missing_rays_fail():
    bounds = torch.tensor([[[-1., -1.], [1., 1.]]])
    points = torch.zeros(1, 8, 30, 3)
    assert in_tile(points, bounds).item()
    points[0, -1, -1, 0] = .9
    assert not in_tile(points, bounds).item()
    ref = reference(n=1, bodies=1)
    q = yaw_rotation(torch.zeros(1, dtype=torch.float64))
    _, valid, _ = place_on_surface(ref, q, torch.zeros(1, 3, dtype=torch.float64), bounds,
                                    Plane(float("inf")), torch.tensor([0]), torch.zeros(1, 3, dtype=torch.float64), torch.tensor([.02], dtype=torch.float64))
    assert not valid.item()


def test_training_collision_model_is_used_including_body_and_feet():
    path = Path(__file__).resolve().parents[1] / "pmt_tasks/assets/unitree_description/urdf/g1/main.urdf"
    ids, points, radii = collision_sphere_cover(path, BFS_BODIES)
    assert len(points) > 100
    assert {BFS_BODIES[i] for i in ids} >= {"pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link"}
    assert np.isfinite(points).all() and (radii > 0).all()


def test_position_only_transform_matches_full_reference_exactly():
    ref=reference(n=7,bodies=30)
    q=yaw_rotation(torch.linspace(-3,3,7,dtype=torch.float64))
    translation=torch.randn(7,3,dtype=torch.float64)
    for future in (False,True):
        source={k:v[:,None].expand(-1,8,*v.shape[1:]) for k,v in ref.items()} if future else ref
        expected=transform_reference(source,q,translation)['body_pos']
        assert torch.equal(transform_positions(source['body_pos'],q,translation),expected)


def test_geometry_only_placement_preserves_gap_bounds_and_lift():
    ref=reference(n=7,bodies=3)
    q=yaw_rotation(torch.linspace(-3,3,7,dtype=torch.float64))
    translation=torch.randn(7,3,dtype=torch.float64)
    bounds=torch.tensor([[[-12.,-12.],[12.,12.]]],dtype=torch.float64).expand(7,-1,-1)
    args=(ref,q,translation,bounds,Plane(-1.5),torch.tensor([0,1,2]),torch.zeros(3,3,dtype=torch.float64),torch.full((3,),.02))
    baseline=place_on_surface(*args)
    fast=place_on_surface(*args,geometry_only=True)
    for a,b in zip(baseline,fast):torch.testing.assert_close(a,b,rtol=0,atol=0,equal_nan=True)


def test_runtime_current_and_future_delegate_without_adding_env_origins_again():
    source = reference(n=2)
    arrays = tuple(source[k] for k in ("joint_pos", "joint_vel", "body_pos", "body_quat", "body_lin_vel", "body_ang_vel"))
    q = yaw_rotation(torch.tensor([.8, -.7]))
    t = torch.tensor([[5., 7., -3.], [8., -2., 4.]])
    command = SimpleNamespace(motion_ids=torch.arange(2), frame_ids=torch.tensor([10, 20]),
                              transform_reference=lambda ref: transform_reference(ref, q, t))
    runtime = PGMTState.__new__(PGMTState)
    runtime.command = command
    runtime.env = SimpleNamespace(scene=SimpleNamespace(env_origins=torch.ones(2, 3) * 1000))
    runtime.full_store = SimpleNamespace(motion_lengths=torch.tensor([499, 499]),
                                         get_motion_data=lambda ids, frames: tuple(a[ids] for a in arrays))
    expected = transform_reference({k: v.float() for k, v in source.items()}, q, t)
    current, future = runtime.reference(), runtime.reference(future=True)
    for key in expected:
        torch.testing.assert_close(current[key], expected[key])
        torch.testing.assert_close(future[key][:, 0], current[key])
