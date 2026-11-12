"""Sharp, individually shaded rendering of the rectangular Bigmap STL pieces."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


def add_terrain(world, assets, path):
    mesh = trimesh.load(path, force="mesh")
    pieces = []
    for part in mesh.split(only_watertight=False):
        low, high = part.bounds[:, 2]
        if high - low < 1e-6:
            if abs(low) > 1e-6 or len(part.faces) != 2:
                return _add_mesh(world, assets, path)
            pieces.append(("ground", part.bounds))
            continue
        top = part.vertices[np.abs(part.vertices[:, 2] - high) < 1e-6, :2]
        if len(top) != 4 or len(part.faces) != 12:
            return _add_mesh(world, assets, path)
        center = top.mean(axis=0)
        top = top[np.argsort(np.arctan2(top[:, 1] - center[1], top[:, 0] - center[0]))]
        edges = np.roll(top, -1, axis=0) - top
        if abs(np.dot(edges[0], edges[1])) > 1e-4:
            return _add_mesh(world, assets, path)
        pieces.append(("box", center, np.linalg.norm(edges[:2], axis=1), low, high,
                       np.arctan2(edges[0, 1], edges[0, 0])))
    # Distinct height bands and flat box normals expose stair risers even when
    # Mesa's passive viewer disables expensive shadow rendering.
    for index, piece in enumerate(pieces):
        if piece[0] == "ground":
            bounds = piece[1]
            center = bounds.mean(axis=0)
            extent = (bounds[1] - bounds[0]) / 2
            ET.SubElement(world, "geom", name=f"terrain_{index}", type="box", group="2",
                          pos=f"{center[0]} {center[1]} -0.01", size=f"{extent[0]} {extent[1]} 0.01",
                          material="ground", contype="0", conaffinity="0")
        else:
            _, center, size, low, high, yaw = piece
            fraction = np.clip(high / .4, 0, 1)
            color = (1 - fraction) * np.array([.76, .73, .64]) + fraction * np.array([.52, .37, .22])
            ET.SubElement(world, "geom", name=f"terrain_{index}", type="box", group="2",
                          pos=f"{center[0]} {center[1]} {(low + high) / 2}",
                          quat=f"{np.cos(yaw / 2)} 0 0 {np.sin(yaw / 2)}",
                          size=f"{size[0] / 2} {size[1] / 2} {(high - low) / 2}",
                          rgba=" ".join(str(v) for v in (*color, 1)),
                          contype="0", conaffinity="0")
    return "individual_boxes"


def _add_mesh(world, assets, path):
    """Keep arbitrary terrain geometry intact if it is not a collection of boxes."""
    ET.SubElement(assets, "mesh", name="terrain", file=str(Path(path).resolve()))
    ET.SubElement(world, "geom", name="terrain", type="mesh", mesh="terrain", group="2",
                  rgba="0.65 0.58 0.46 1", contype="0", conaffinity="0")
    return "mesh"
