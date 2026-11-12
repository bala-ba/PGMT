"""ROS-independent adapters for rectified RealSense depth and URDF self filtering."""
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from pmt_tasks.depth_perception import CameraIntrinsics, validate_pose


def decode_depth_image(data, width, height, step, encoding, bigendian=False):
    """Decode ROS depth encodings, preserving padded rows and byte order.

    ROS 16UC1 depth is millimetres; 32FC1 is metres. This is not the arbitrary
    raw-device Z16 scale: librealsense users must use get_depth_scale() instead.
    """
    kinds = {"16UC1": ("u2", .001), "32FC1": ("f4", 1.)}
    if encoding not in kinds:
        raise ValueError(f"Unsupported metric depth encoding: {encoding}")
    kind, scale = kinds[encoding]
    dtype = np.dtype((">" if bigendian else "<") + kind)
    if min(width, height) <= 0 or step < width * dtype.itemsize or len(data) != height * step:
        raise ValueError("Depth byte buffer/row stride does not match dimensions")
    view = np.ndarray((height, width), dtype=dtype, buffer=bytes(data), strides=(step, dtype.itemsize))
    depth = view.astype(np.float32) * scale
    depth[~np.isfinite(depth) | (depth <= 0)] = 0
    return depth


def subsample_depth(depth, intrinsics, stride):
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("Pixel stride must be a positive integer")
    sampled = depth[::stride, ::stride]
    k = intrinsics
    return sampled, CameraIntrinsics(sampled.shape[1], sampled.shape[0], k.fx / stride, k.fy / stride, k.cx / stride, k.cy / stride)


class RobotSelfFilter:
    """Conservative primitive collision mask from the G1 URDF and measured TF.

    Unknown collision types are rejected rather than silently unmasked. Cylinders
    are expanded to capsules, consistent with the training importer. This removes
    measured robot points only; it never invents points behind the robot.
    """
    def __init__(self, urdf_path, padding=.04):
        if not np.isfinite(padding) or padding < 0:
            raise ValueError("Self-filter padding must be nonnegative")
        self.padding = padding
        self.collisions = []
        for link in ET.parse(urdf_path).getroot().findall("link"):
            for collision in link.findall("collision"):
                origin = collision.find("origin")
                attrs = origin.attrib if origin is not None else {}
                pose = np.eye(4)
                pose[:3, 3] = np.fromstring(attrs.get("xyz", "0 0 0"), sep=" ")
                pose[:3, :3] = Rotation.from_euler("xyz", np.fromstring(attrs.get("rpy", "0 0 0"), sep=" ")).as_matrix()
                primitive = collision.find("geometry")[0]
                if primitive.tag not in ("sphere", "cylinder", "box"):
                    raise ValueError(f"Unsupported collision geometry {primitive.tag} in {link.get('name')}")
                self.collisions.append((link.get("name"), pose, primitive.tag, primitive.attrib))
        self.link_names = sorted({c[0] for c in self.collisions})
        if not self.collisions:
            raise ValueError("URDF has no collision geometry")

    def valid_mask(self, depth, intrinsics, world_from_camera, world_from_links):
        k = intrinsics
        if depth.shape != (k.height, k.width):
            raise ValueError("Depth and calibration shape differ")
        pose = validate_pose(world_from_camera)
        v, u = np.indices(depth.shape)
        camera = np.stack(((u - k.cx) * depth / k.fx, (v - k.cy) * depth / k.fy, depth), axis=-1).reshape(-1, 3)
        world = camera @ pose[:3, :3].T + pose[:3, 3]
        masked = np.zeros(len(world), dtype=bool)
        for link, local_pose, kind, shape in self.collisions:
            transform = validate_pose(world_from_links[link]) @ local_pose
            local = (world - transform[:3, 3]) @ transform[:3, :3]
            if kind == "box":
                half_size = np.fromstring(shape["size"], sep=" ") / 2 + self.padding
                inside = (np.abs(local) <= half_size).all(axis=-1)
            else:
                if kind == "cylinder":
                    half_length = float(shape["length"]) / 2
                    local[:, 2] -= np.clip(local[:, 2], -half_length, half_length)
                inside = (local ** 2).sum(axis=-1) <= (float(shape["radius"]) + self.padding) ** 2
            masked |= inside
        return ~masked.reshape(depth.shape)
