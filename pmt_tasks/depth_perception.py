"""Sensor-independent depth -> rolling elevation map -> PMT observation.

Depth is rectified optical-axis Z in metres, not Euclidean range. Optical axes
are right/down/forward; poses map optical coordinates into a gravity-aligned
odometry frame. No simulator, terrain mesh, or ground-truth height is used here.
See docs/DEPTH_PERCEPTION.md for assumptions and reference implementations.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self):
        if self.width < 3 or self.height < 3 or not np.isfinite(
            [self.fx, self.fy, self.cx, self.cy]
        ).all() or min(self.fx, self.fy) <= 0:
            raise ValueError("Invalid pinhole intrinsics")

    @classmethod
    def from_fovy(cls, width, height, fovy_deg):
        if not 0 < fovy_deg < 180:
            raise ValueError("Vertical FOV must be between 0 and 180 degrees")
        f = height / (2 * np.tan(np.deg2rad(fovy_deg) / 2))
        return cls(width, height, f, f, (width - 1) / 2, (height - 1) / 2)


def validate_pose(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Pose must be a finite 4x4 optical-to-odometry transform")
    r = pose[:3, :3]
    if not np.allclose(pose[3], [0, 0, 0, 1]) or not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1, atol=1e-5):
        raise ValueError("Pose must be a proper rigid transform")
    return pose


def depth_points(depth_m, intrinsics, world_from_camera, valid_mask=None,
                 min_depth=.15, max_depth=5., min_normal_z=.5):
    """Back-project depth and reject steep faces/depth discontinuities.

    Surface normals are estimated from neighbouring measured points. Filtering
    does not complete holes or infer hidden surfaces. External self masks must
    describe the robot only, and should include a small calibration margin.
    """
    k = intrinsics
    depth = np.asarray(depth_m, dtype=float)
    if depth.shape != (k.height, k.width):
        raise ValueError("Depth shape differs from intrinsics")
    pose = validate_pose(world_from_camera)
    valid = np.isfinite(depth) & (depth >= min_depth) & (depth < max_depth)
    if valid_mask is not None:
        if np.shape(valid_mask) != depth.shape:
            raise ValueError("Mask shape differs from depth")
        valid &= np.asarray(valid_mask, dtype=bool)
    z = np.where(valid, depth, 0.)
    v, u = np.indices(depth.shape)
    camera_points = np.stack(((u - k.cx) * z / k.fx, (v - k.cy) * z / k.fy, z), axis=-1)
    points = camera_points @ pose[:3, :3].T + pose[:3, 3]
    if min_normal_z > 0:
        dx = points[1:-1, 2:] - points[1:-1, :-2]
        dy = points[2:, 1:-1] - points[:-2, 1:-1]
        normal = np.cross(dx, dy)
        norm = np.linalg.norm(normal, axis=-1)
        support = valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2] & valid[2:, 1:-1] & valid[:-2, 1:-1]
        # Reject neighbours across a depth edge before computing map heights.
        center = z[1:-1, 1:-1]
        jump = np.maximum.reduce([np.abs(z[1:-1, 2:] - center), np.abs(z[1:-1, :-2] - center),
                                  np.abs(z[2:, 1:-1] - center), np.abs(z[:-2, 1:-1] - center)])
        support &= (norm > 1e-10) & (np.abs(normal[..., 2]) >= min_normal_z * norm) & (jump < .10)
        valid[:] = False
        valid[1:-1, 1:-1] = support
    return points[valid]


class LocalElevationMap:
    """Bounded world-aligned rolling grid with observation age and height fusion.

    This lightweight NumPy implementation targets static, single-layer terrain.
    It does not implement SLAM, pose covariance propagation or visibility cleanup.
    """
    def __init__(self, resolution=.05, size_m=6., max_age_s=2., min_points=2):
        if not np.isfinite([resolution, size_m, max_age_s]).all() or min(resolution, size_m, max_age_s) <= 0 or min_points < 1:
            raise ValueError("Map dimensions, age and point count must be positive")
        self.resolution, self.max_age_s, self.min_points = resolution, max_age_s, min_points
        self.n = int(np.ceil(size_m / resolution))
        self.origin = None
        self.height = np.zeros((self.n, self.n))
        self.seen = np.full((self.n, self.n), -np.inf)
        self.last_timestamp = -np.inf

    def reset(self):
        self.origin = None
        self.height.fill(0)
        self.seen.fill(-np.inf)
        self.last_timestamp = -np.inf

    def _recenter(self, center_xy):
        center = np.asarray(center_xy)
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError("Map center must be finite XY")
        origin = np.floor(center / self.resolution).astype(np.int64) - self.n // 2
        if self.origin is not None:
            delta = origin - self.origin
            if np.any(np.abs(delta) >= self.n):
                self.seen.fill(-np.inf)
                self.height.fill(0)
            elif np.any(delta):
                for array, fill in ((self.height, 0), (self.seen, -np.inf)):
                    array[:] = np.roll(array, tuple(-delta), axis=(0, 1))
                    for axis, shift in enumerate(delta):
                        if shift:
                            slices = [slice(None), slice(None)]
                            slices[axis] = slice(-int(shift), None) if shift > 0 else slice(None, -int(shift))
                            array[tuple(slices)] = fill
        self.origin = origin

    def integrate(self, depth_m, intrinsics, world_from_camera, timestamp,
                  center_xy, valid_mask=None):
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or timestamp < self.last_timestamp:
            raise ValueError("Depth timestamps must be finite and nondecreasing; reset after a clock/pose reset")
        points = depth_points(depth_m, intrinsics, world_from_camera, valid_mask)
        self._recenter(center_xy)
        self.last_timestamp = timestamp
        index = np.floor(points[:, :2] / self.resolution).astype(np.int64) - self.origin
        inside = ((index >= 0) & (index < self.n)).all(axis=1)
        points, index = points[inside], index[inside]
        flat = index[:, 0] * self.n + index[:, 1]
        # Use the upper measured surface per cell, not a mean across step levels.
        # Normals above already reject vertical walls and discontinuity pixels.
        top = np.full(self.n * self.n, -np.inf)
        np.maximum.at(top, flat, points[:, 2])
        near_top = points[:, 2] >= top[flat] - .025
        count = np.bincount(flat[near_top], minlength=len(top))
        total = np.bincount(flat[near_top], weights=points[near_top, 2], minlength=len(top))
        cells = np.flatnonzero(count >= self.min_points)
        observed = total[cells] / count[cells]
        old = self.height.ravel()[cells]
        age = timestamp - self.seen.ravel()[cells]
        compatible = (age <= self.max_age_s) & (np.abs(old - observed) < .04)
        self.height.ravel()[cells] = np.where(compatible, .5 * old + .5 * observed, observed)
        self.seen.ravel()[cells] = timestamp
        return {"surface_points": len(points), "updated_cells": len(cells)}

    def sample(self, torso_position, yaw, timestamp):
        """Return PMT's exact XY flattening: x changes fastest, then y."""
        torso = np.asarray(torso_position, dtype=float)
        if torso.shape != (3,) or not np.isfinite(np.r_[torso, yaw, timestamp]).all() or timestamp < self.last_timestamp:
            raise ValueError("Invalid torso pose or query time precedes latest depth")
        x, y = np.meshgrid(np.linspace(-.8, .8, 17), np.linspace(-.5, .5, 11))
        c, s = np.cos(yaw), np.sin(yaw)
        xy = np.stack([x.ravel(), y.ravel()], axis=-1) @ np.array([[c, s], [-s, c]]) + torso[:2]
        valid = np.zeros(187, dtype=bool)
        elevation = np.zeros(187)
        age = np.full(187, -1.)
        if self.origin is not None:
            index = np.floor(xy / self.resolution).astype(np.int64) - self.origin
            inside = ((index >= 0) & (index < self.n)).all(axis=1)
            idx = index[inside]
            observed_at = self.seen[idx[:, 0], idx[:, 1]]
            fresh = (timestamp - observed_at <= self.max_age_s)
            valid[inside] = fresh
            elevation[inside] = self.height[idx[:, 0], idx[:, 1]]
            age[np.flatnonzero(inside)[fresh]] = timestamp - observed_at[fresh]
        hits = np.column_stack([xy, elevation])
        hits[~valid] = 0
        height = np.where(valid, np.clip(torso[2] - elevation, -3, 3), -3.)
        return {"hits_w": hits.tolist(), "hit_valid": valid.tolist(),
                "policy_height": height.tolist(), "policy_valid": valid.tolist(),
                "age_s": age.tolist(), "source": "depth_camera_local_map"}
