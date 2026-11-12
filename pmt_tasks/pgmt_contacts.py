"""Offline TCRS contact labels queried against an explicit reference mesh.

No simulation, terrain adaptation, IK or motion filtering is performed here.
Reference-mesh labels describe the reference surface, not an arbitrary future
training terrain. Missing mesh hits and deep penetration are invalid labels.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from TCRS.stair_mppi.contact_detection import detect_foot_contacts, stabilize_contact_mask

FOOT_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")


@dataclass(frozen=True)
class ContactConfig:
    contact_z_threshold: float = 0.06
    contact_speed_threshold: float = 0.5
    min_contact_frames: int = 4
    min_swing_frames: int = 2
    mesh_contact_tolerance_m: float = 0.02
    penetration_tolerance_m: float = 0.02

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError("Contact parameters must be finite and positive")
        if any(not isinstance(v, int) for v in (self.min_contact_frames, self.min_swing_frames)):
            raise ValueError("Contact run lengths must be integers")


class MeshHeightQuery:
    """Highest triangle hit per downward ray, with an explicit missing-hit mask."""

    def __init__(self, mesh, batch_size=8192):
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0 or not np.isfinite(mesh.vertices).all():
            raise ValueError("Expected a nonempty finite triangle mesh")
        if batch_size < 1:
            raise ValueError("Ray batch_size must be positive")
        self.mesh = mesh
        self.batch_size = batch_size
        self.ray_top = float(mesh.bounds[1, 2]) + 1.0

    def __call__(self, xy):
        xy = np.asarray(xy, dtype=np.float64)
        if xy.shape[-1] != 2 or not np.isfinite(xy).all():
            raise ValueError("Ray coordinates must be finite xy points")
        points = xy.reshape(-1, 2)
        heights = np.full(len(points), -np.inf)
        for start in range(0, len(points), self.batch_size):
            batch = points[start:start + self.batch_size]
            origins = np.column_stack([batch, np.full(len(batch), self.ray_top)])
            directions = np.zeros_like(origins)
            directions[:, 2] = -1.0
            hits, ray_ids, _ = self.mesh.ray.intersects_location(origins, directions, multiple_hits=True)
            np.maximum.at(heights, start + ray_ids, hits[:, 2])
        valid = np.isfinite(heights)
        heights[~valid] = np.nan
        return heights.reshape(xy.shape[:-1]), valid.reshape(xy.shape[:-1])


def reference_plane(bounds_xy, height=0.0):
    """Two real triangles covering the reference trajectory's xy bounds."""
    lo, hi = np.asarray(bounds_xy, dtype=np.float64)
    if lo.shape != (2,) or np.any(hi <= lo) or not np.isfinite([*lo, *hi, height]).all():
        raise ValueError("Invalid reference plane bounds/height")
    return trimesh.Trimesh(vertices=[[lo[0], lo[1], height], [hi[0], lo[1], height],
                                     [hi[0], hi[1], height], [lo[0], hi[1], height]],
                           faces=[[0, 1, 2], [0, 2, 3]], process=False)


def foot_collision_probes(model):
    """Read TCRS G1's four collision spheres per ankle by body name.

    Sphere centres rotate with the foot; each world-space ray probes the sphere
    bottom (world z minus radius), not a rotated local bottom point.
    """
    import mujoco

    centres, radii = [], []
    for name in FOOT_NAMES:
        body_id = model.body(name).id
        ids = np.where((model.geom_bodyid == body_id) &
                       (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE) &
                       ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))[0]
        if len(ids) != 4:
            raise ValueError(f"Expected four collision spheres on {name}, found {len(ids)}")
        centres.append(model.geom_pos[ids].copy())
        radii.append(model.geom_size[ids, 0].copy())
    return np.asarray(centres), np.asarray(radii)


def label_contacts(arrays, query, probe_centres, probe_radii, config=ContactConfig()):
    """Return per-frame [left,right] labels plus geometry diagnostics.

    `contacts` uses TCRS raw height/speed detection, gated by mesh proximity,
    then the original run filter. The final mesh gate prevents that filter from
    filling an actual airborne/missing/penetrating frame. `contacts_tcrs` keeps
    the ungated TCRS result for audit. Unknown labels must be masked by callers.
    """
    names = np.asarray(arrays["body_names"]).tolist()
    ids = [names.index(name) for name in FOOT_NAMES]
    pos = np.asarray(arrays["body_pos_w"], dtype=np.float64)[:, ids]
    quat = np.asarray(arrays["body_quat_w"], dtype=np.float64)[:, ids]
    velocity = np.asarray(arrays["body_lin_vel_w"], dtype=np.float64)[:, ids]
    if len(pos) == 0 or pos.shape[1:] != (2, 3) or velocity.shape != pos.shape or quat.shape != (*pos.shape[:2], 4):
        raise ValueError("Invalid foot state shapes")
    if not all(np.isfinite(a).all() for a in (pos, quat, velocity)):
        raise ValueError("Foot states must be finite")
    if not np.allclose(np.linalg.norm(quat, axis=-1), 1, atol=1e-4, rtol=0):
        raise ValueError("Foot rotations must be unit wxyz quaternions")
    centres, radii = np.asarray(probe_centres), np.asarray(probe_radii)
    if centres.shape != (2, 4, 3) or radii.shape != (2, 4) or not np.isfinite(centres).all() or not np.isfinite(radii).all() or (radii <= 0).any():
        raise ValueError("Expected finite four-sphere foot geometry with positive radii")
    rotations = Rotation.from_quat(quat[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(-1, 2, 3, 3)
    probes = pos[:, :, None] + np.einsum("nfij,fkj->nfki", rotations, centres)
    probes[..., 2] -= radii[None]
    xy = np.concatenate([pos[:, :, None, :2], probes[..., :2]], axis=2)
    heights, hit = query(xy)
    ankle_height = pos[..., 2] - heights[..., 0]
    clearance = probes[..., 2] - heights[..., 1:]
    speed = np.linalg.norm(velocity, axis=-1)
    # Mesh holes/edges and excessive penetration are unknown, never negative
    # contact examples. Both feet remain independent, including double stance.
    valid = hit.all(axis=-1) & (clearance >= -config.penetration_tolerance_m).all(axis=-1)
    mesh_support = valid & (clearance <= config.mesh_contact_tolerance_m).any(axis=-1)
    detected = detect_foot_contacts(
        np.nan_to_num(ankle_height), speed, valid_mask=hit[..., 0],
        contact_z_threshold=config.contact_z_threshold,
        contact_speed_threshold=config.contact_speed_threshold,
        min_contact_frames=config.min_contact_frames, min_swing_frames=config.min_swing_frames,
    )
    raw = detected.raw & mesh_support
    contact = np.column_stack([stabilize_contact_mask(raw[:, i], config.min_contact_frames,
                                                      config.min_swing_frames) for i in range(2)])
    contact &= mesh_support
    state = np.where(valid, contact.astype(np.int8), -1).astype(np.int8)
    support_mode = (contact[:, 0].astype(np.int8) + 2 * contact[:, 1].astype(np.int8))
    support_mode[~valid.all(axis=1)] = -1
    return {
        "contacts": contact.astype(np.float32), "contact_valid": valid,
        "contacts_tcrs": detected.contacts, "contacts_raw": raw,
        "mesh_support": mesh_support, "swing": valid & ~contact,
        "foot_state": state, "support_mode": support_mode,
        "foot_ground_height": heights[..., 0].astype(np.float32),
        "foot_probe_clearance": clearance.astype(np.float32), "foot_mesh_hit": hit,
        "foot_speed": speed.astype(np.float32),
        "contact_height_threshold": np.float32(detected.height_threshold),
        "foot_names": np.asarray(FOOT_NAMES),
        "foot_probe_centres": centres.astype(np.float32), "foot_probe_radii": radii.astype(np.float32),
    }
