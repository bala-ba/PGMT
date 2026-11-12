"""Native RayCaster with the complete mesh supplied by PGMTTerrainImporter."""
from weakref import WeakValueDictionary

from isaaclab.sensors import RayCaster


_terrain_meshes = WeakValueDictionary()


def register_terrain_scan_mesh(prim_path, mesh):
    # Keep ownership in the terrain importer. Registration happens before the
    # sensors exist, but installing in RayCaster.meshes must wait until after
    # config validation (which recursively traverses class attributes).
    _terrain_meshes[prim_path] = mesh


class PGMTHeightScanner(RayCaster):
    """Reuse native grid, pose/yaw tracking, timing, reset and GPU ray casting.

    Only mesh discovery differs: native discovery takes the first USD child,
    while PGMT has multiple meshes (v5: 70, v6: 800). Never silently fall back.
    """

    def _initialize_warp_meshes(self):
        if len(self.cfg.mesh_prim_paths) != 1:
            raise ValueError("PGMT scanner requires one complete terrain mesh")
        path = self.cfg.mesh_prim_paths[0]
        mesh = _terrain_meshes.get(path)
        if mesh is None:
            raise RuntimeError(f"PGMT terrain importer has not registered the scan mesh: {path}")
        RayCaster.meshes[path] = mesh
