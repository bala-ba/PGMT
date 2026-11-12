"""Render-only camera origin, viewing direction and field-of-view markers."""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def recorded_camera(metadata):
    """Read the actual simulated camera mount, without guessing a G1 extrinsic."""
    if metadata.get("perception") != "depth":
        return None
    scene = metadata.get("geometry", {}).get("scene_xml")
    if not scene or not Path(scene).exists():
        return None
    root = ET.parse(scene).getroot()
    for body in root.iter("body"):
        camera = body.find("camera[@name='pmt_depth']")
        if camera is not None:
            if body.get("name") == "head_link":
                # The viewer robot fuses the fixed head into torso_link. Express
                # the head-mounted camera in that parent frame, preserving its pose.
                source = mujoco.MjModel.from_xml_path(str(scene))
                data = mujoco.MjData(source)
                mujoco.mj_forward(source, data)
                torso = source.body("torso_link").id
                camera_id = source.camera("pmt_depth").id
                inverse = data.xmat[torso].reshape(3, 3).T
                position = inverse @ (data.cam_xpos[camera_id] - data.xpos[torso])
                orientation = inverse @ data.cam_xmat[camera_id].reshape(3, 3)
                quaternion = np.empty(4)
                mujoco.mju_mat2Quat(quaternion, orientation.ravel())
                return "torso_link", {"name": "pmt_depth", "pos": " ".join(map(str, position)),
                                      "quat": " ".join(map(str, quaternion)), "fovy": str(source.cam_fovy[camera_id])}
            return body.get("name"), dict(camera.attrib)
    return None


def add_camera_marker(scene, model, data, camera_id, aspect):
    """Append decorative geoms only; these never enter physics or depth sensing."""
    if scene.ngeom + 12 > scene.maxgeom:
        raise ValueError("Scene lacks space for camera marker")
    origin = data.cam_xpos[camera_id].copy()
    orientation = data.cam_xmat[camera_id].reshape(3, 3)
    forward = -orientation[:, 2]  # MuJoCo looks along local -Z.
    yellow = [1., .78, .05, 1.]
    cyan = [.1, .95, 1., .65]

    def geom(kind, position, size, color, matrix=None):
        result = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(result, kind, np.asarray(size, dtype=float), position,
                           (np.eye(3) if matrix is None else matrix).ravel(), np.asarray(color))
        scene.ngeom += 1
        return result

    def connector(kind, start, end, width, color):
        result = geom(kind, np.zeros(3), [0, 0, 0], color)
        mujoco.mjv_connector(result, kind, width, start, end)

    geom(mujoco.mjtGeom.mjGEOM_SPHERE, origin, [.027] * 3, yellow)
    connector(mujoco.mjtGeom.mjGEOM_ARROW, origin, origin + .35 * forward, .008, yellow)
    # The short frustum illustrates angular coverage; it is not a range limit.
    distance = .45
    half_height = distance * np.tan(np.deg2rad(model.cam_fovy[camera_id]) / 2)
    corners = np.array([[x * half_height * aspect, y * half_height, -distance]
                        for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]) @ orientation.T + origin
    for i in range(4):
        connector(mujoco.mjtGeom.mjGEOM_LINE, origin, corners[i], 1.5, cyan)
        connector(mujoco.mjtGeom.mjGEOM_LINE, corners[i], corners[(i + 1) % 4], 1.5, cyan)
    label_position = origin + [0., 0., .30]
    connector(mujoco.mjtGeom.mjGEOM_LINE, origin, label_position, 2., yellow)
    label = geom(mujoco.mjtGeom.mjGEOM_LABEL, label_position, [0, 0, 0], yellow)
    label.label = "Depth camera (sim)"
