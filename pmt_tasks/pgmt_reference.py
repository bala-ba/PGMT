"""Pure tensor rigid reference placement and conservative G1 collision probes."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from pmt_tasks.pgmt_math import conjugate, multiply, rotate, yaw_quat


def yaw_rotation(yaw):
    q = torch.zeros((*yaw.shape, 4), device=yaw.device, dtype=yaw.dtype)
    q[..., 0], q[..., 3] = torch.cos(yaw / 2), torch.sin(yaw / 2)
    return q


def initial_transform(root_pos, root_quat, target_xy, target_yaw):
    """Absolute target yaw, with translation anchoring the selected start frame."""
    q = multiply(yaw_rotation(target_yaw), conjugate(yaw_quat(root_quat)))
    translation = torch.zeros_like(root_pos)
    translation[:, :2] = target_xy - rotate(q, root_pos)[:, :2]
    return q, translation


def transform_reference(reference, rotation, translation):
    """One constant world transform for [env, (future), body, xyz] arrays.

    Positions translate and rotate; quaternions left-multiply; both world
    velocities rotate. Joint angles/velocities and extra labels are preserved.
    Never modifies the source tensors and never adds env_origins again.
    """
    result = dict(reference)
    pos = reference["body_pos"]
    extra = (1,) * (pos.ndim - 2)
    q = rotation.reshape(rotation.shape[0], *extra, 4)
    t = translation.reshape(translation.shape[0], *extra, 3)
    # torch.cross requires equal dimensions, and supports singleton broadcasting.
    result["body_pos"] = rotate(q, pos) + t
    result["body_quat"] = multiply(q, reference["body_quat"])
    result["body_lin_vel"] = rotate(q, reference["body_lin_vel"])
    result["body_ang_vel"] = rotate(q, reference["body_ang_vel"])
    return result


def transform_positions(points, rotation, translation):
    """Position-only equivalent of transform_reference, including future axes."""
    extra = (1,) * (points.ndim - 2)
    return rotate(rotation.reshape(rotation.shape[0], *extra, 4), points) + translation.reshape(
        translation.shape[0], *extra, 3)


def collision_sphere_cover(urdf_path, body_names, spacing=0.02):
    """Cover the training URDF's spheres/capsules with conservative spheres.

    PMT imports cylinders with replace_cylinders_with_capsules=True. Samples
    cover their full centre segments; inflated radii sqrt(r^2+(step/2)^2)
    cover the space between samples as well. Unknown geometry fails explicitly.
    This uses training collision shapes, not the offline TCRS foot spheres.
    """
    if spacing <= 0:
        raise ValueError("Collision cover spacing must be positive")
    body_ids, points, radii = [], [], []
    for link in ET.parse(Path(urdf_path)).getroot().findall("link"):
        for collision in link.findall("collision"):
            name = link.get("name")
            if name not in body_names:
                raise ValueError(f"Unmapped collision link: {name}")
            origin = collision.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            geometry = list(collision.find("geometry"))[0]
            if geometry.tag == "sphere":
                centres, radius = np.zeros((1, 3)), float(geometry.get("radius"))
            elif geometry.tag == "cylinder":
                length, radius = float(geometry.get("length")), float(geometry.get("radius"))
                count = max(2, int(np.ceil(length / spacing)) + 1)
                centres = np.zeros((count, 3))
                centres[:, 2] = np.linspace(-length / 2, length / 2, count)
                radius = float(np.hypot(radius, length / (count - 1) / 2))
            else:
                raise ValueError(f"Unsupported training collision shape: {geometry.tag}")
            points.extend(Rotation.from_euler("xyz", rpy).apply(centres) + xyz)
            radii.extend([radius] * len(centres))
            body_ids.extend([body_names.index(name)] * len(centres))
    if not points:
        raise ValueError("No training collision geometry found")
    return np.asarray(body_ids), np.asarray(points), np.asarray(radii)


def collision_centres(reference, body_ids, local_centres):
    return reference["body_pos"][:, body_ids] + rotate(reference["body_quat"][:, body_ids], local_centres[None])


def in_tile(points, bounds, margin=0.25):
    """Check every point's xy against its environment's rectangular tile."""
    xy = points[..., :2].reshape(len(points), -1, 2)
    return ((xy >= bounds[:, None, 0] + margin) & (xy <= bounds[:, None, 1] - margin)).all(dim=(1, 2))


def place_on_surface(reference, rotation, translation, bounds, query, body_ids, local_centres, radii,
                     *, clearance=0.003, boundary_margin=0.25, max_iterations=8, geometry_only=False):
    """Rigidly lift/lower a start pose above the real mesh; no joint edits.

    `query.height(xy)` returns world heights (inf for missing). Its
    `query.distance(xyz)` returns unsigned closest-triangle distance. Combining
    downward rays with sphere/triangle distances rejects missing surfaces,
    below-ground placement and capsule intersections near terrain edges.
    """
    if geometry_only:
        placed = dict(body_pos=transform_positions(reference['body_pos'], rotation, translation),
                      body_quat=multiply(rotation[:, None], reference['body_quat']))
    else:
        placed = transform_reference(reference, rotation, translation)
    centres = collision_centres(placed, body_ids, local_centres)
    radii = radii[None]
    valid = in_tile(centres - radii[..., None], bounds, boundary_margin)
    valid &= in_tile(centres + radii[..., None], bounds, boundary_margin)
    height = query.height(centres[..., :2])
    valid &= torch.isfinite(height).all(-1)
    shift = (height + radii + clearance - centres[..., 2]).max(-1).values
    shift = torch.where(valid, shift, torch.zeros_like(shift))
    centres[..., 2] += shift[:, None]
    result_translation = translation.clone()
    result_translation[:, 2] += shift
    # Closest-triangle checks also catch sloped surfaces and vertical risers
    # inside the sphere footprint that a centre-only height ray would miss.
    for _ in range(max_iterations):
        distance = query.distance(centres)
        deficit = (radii + clearance - distance).clamp_min(0).max(-1).values
        needs_lift = valid & (deficit > 1e-5)
        if not needs_lift.any():
            break
        lift = torch.where(needs_lift, deficit + clearance, torch.zeros_like(deficit))
        result_translation[:, 2] += lift
        centres[..., 2] += lift[:, None]
    distance = query.distance(centres)
    minimum_gap = (distance - radii).min(-1).values
    valid &= torch.isfinite(distance).all(-1) & (minimum_gap >= clearance - 1e-5)
    # At least one conservative collision sphere must remain close to support;
    # impossible edge placements are rejected instead of floating far above it.
    support_gap = (centres[..., 2] - radii - query.height(centres[..., :2])).min(-1).values
    valid &= torch.isfinite(support_gap) & (support_gap <= 0.05)
    return result_translation, valid, minimum_gap
