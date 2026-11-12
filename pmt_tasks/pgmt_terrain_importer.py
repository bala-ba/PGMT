"""Import the audited PGMT USD and query its same cached triangles on the GPU."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
import warp as wp
from pxr import Usd, UsdPhysics

from isaaclab.sim import get_current_stage
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh
from pmt_tasks.pgmt_terrain_query import TerrainTileMap
from pmt_tasks.pgmt_height_scanner import register_terrain_scan_mesh
from pmt_tasks.pgmt_terrain_layout import (
    PATCH_SCHEMA, training_manifest, compatibility_families, patch_bounds, sample_patch_positions,
)
from pmt_tasks.pgmt_terrain_distribution import allocate_terrain_types


@wp.kernel
def _surface_distance(mesh: wp.uint64, points: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    i = wp.tid()
    p = points[i]
    result = wp.mesh_query_point(mesh, p, 1000.0)
    if result.result:
        nearest = wp.mesh_eval_position(mesh, result.face, result.u, result.v)
        out[i] = wp.length(p - nearest)
    else:
        out[i] = float(1.0e10)


class PGMTTerrainImporter(TerrainImporter):
    def __init__(self, cfg):
        root = Path(cfg.asset_directory).resolve()
        self.manifest = training_manifest(json.loads((root / "manifest.json").read_text()))
        self.has_patches = self.manifest.get("schema") == PATCH_SCHEMA
        self.compatibility_families = compatibility_families(self.manifest)
        self.surface_records = self.manifest["patches"] if self.has_patches else self.manifest["tiles"]
        self.collision_records = list(self.surface_records) + self.manifest.get("auxiliary_surfaces", [])
        usd = root / self.manifest["asset"]
        if hashlib.sha256(usd.read_bytes()).hexdigest() != self.manifest["asset_sha256"]:
            raise ValueError("PGMT terrain USD hash mismatch")
        cfg.usd_path = str(usd)
        super().__init__(cfg)
        # The standalone preview USD includes its own PhysicsScene. Training
        # must use only SimulationContext's scene/dt, otherwise the imported
        # scene also steps at its default rate (not our 200 Hz physics clock).
        self.disabled_embedded_physics_scenes = []
        self.contact_filter_paths = []
        root_prim = get_current_stage().GetPrimAtPath(cfg.prim_path)
        for prim in Usd.PrimRange(root_prim):
            if prim.IsA(UsdPhysics.Scene):
                self.disabled_embedded_physics_scenes.append(str(prim.GetPath()))
                prim.SetActive(False)
            if (prim.HasAPI(UsdPhysics.CollisionAPI)
                    and str(prim.GetPath()).startswith(cfg.prim_path + "/terrain/Terrain/")):
                self.contact_filter_paths.append(str(prim.GetPath()))
        self.contact_filter_paths.sort()
        if len(self.contact_filter_paths) != len(self.collision_records):
            raise ValueError("PGMT terrain collider count does not match the manifest")
        spec = self.manifest["spec"]
        rows, cols = spec["num_rows"], spec["num_cols"]
        origins = np.zeros((rows, cols, 3), dtype=np.float32)
        bounds = np.zeros((rows, cols, 2, 2), dtype=np.float32)
        meshes = []
        for tile in self.collision_records:
            file = root / tile["mesh_file"]
            if hashlib.sha256(file.read_bytes()).hexdigest() != tile["mesh_sha256"]:
                raise ValueError(f"PGMT mesh hash mismatch: {file}")
            mesh = trimesh.load_mesh(file, process=False)
            mesh.apply_translation(tile["center"])
            meshes.append(mesh)
            if not self.has_patches:
                level, family = tile["level"], tile["family_index"]
                origins[level, family] = tile["origin"]
                bounds[level, family] = mesh.bounds[:, :2]
        if self.has_patches:
            for region in self.manifest["tiles"]:
                level, family = region["level"], region["family_index"]
                origins[level, family] = region["origin"]
                bounds[level, family] = region["bounds"]
            self.patch_bounds = patch_bounds(self.manifest, device=self.device)
        self.configure_env_origins(origins)
        self.family_proportions = (cfg.family_proportions if cfg.family_proportions is not None
                                   else spec.get("family_proportions"))
        if self.family_proportions is not None:
            self.terrain_types = allocate_terrain_types(
                cfg.num_envs, spec["families"], self.family_proportions, device=self.device)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        self.tile_bounds = torch.as_tensor(bounds, device=self.device)
        self.tile_map = TerrainTileMap(self.manifest, self.tile_bounds)
        # P-BFM includes traversable guard tiles and a safety strip outside
        # the curriculum grid. They have geometry, but no curriculum label.
        self.walkable_bounds = (torch.as_tensor(self.manifest["walkable_bounds"], device=self.device,
                                               dtype=torch.float32)
                                if "walkable_bounds" in self.manifest
                                else torch.stack((self.tile_map.lower, self.tile_map.upper)))
        combined = trimesh.util.concatenate(meshes)
        self.query_mesh = convert_to_warp_mesh(combined.vertices, combined.faces, device=self.device)
        self.ray_top = float(combined.bounds[1, 2]) + 5.0
        # Share the same colliders with placement, scans and support queries.
        register_terrain_scan_mesh(cfg.prim_path, self.query_mesh)

    def height(self, xy):
        starts = torch.zeros((*xy.shape[:-1], 3), device=xy.device, dtype=torch.float32)
        starts[..., :2], starts[..., 2] = xy, self.ray_top
        directions = torch.zeros_like(starts)
        directions[..., 2] = -1
        return raycast_mesh(starts, directions, self.query_mesh)[0][..., 2]

    def distance(self, points):
        flat = points.reshape(-1, 3).float().contiguous()
        result = torch.empty(len(flat), device=points.device)
        wp.launch(_surface_distance, dim=len(flat), inputs=[self.query_mesh.id, wp.from_torch(flat, dtype=wp.vec3)],
                  outputs=[wp.from_torch(result)], device=self.query_mesh.device)
        return result.reshape(points.shape[:-1])

    def bounds_for(self, env_ids):
        """Assigned spawn tile bounds, used only for initial pose placement."""
        return self.tile_bounds[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def sample_patch_positions(self, env_ids, margin, *, allowed_patch_ids=None, xy_mode="random"):
        return sample_patch_positions(self.patch_bounds, self.terrain_levels[env_ids],
                                      self.terrain_types[env_ids], margin=margin,
                                      allowed_patch_ids=allowed_patch_ids, xy_mode=xy_mode)

    def contains(self, xy):
        if "walkable_bounds" not in self.manifest:
            return self.tile_map.contains(xy)
        return (torch.isfinite(xy).all(-1) & (xy >= self.walkable_bounds[0]).all(-1)
                & (xy < self.walkable_bounds[1]).all(-1))

    def query(self, xy):
        """Clean mesh height and actual tile metadata at world XY positions.

        terrain_levels/types remain the birth assignment. Missing mesh rays or
        out-of-map points have valid=False and NaN height; no zero padding.
        """
        result = self.tile_map.locate(xy)
        # Avoid passing NaN/Inf or distant invalid points to the ray kernel.
        inside = self.contains(xy)
        safe_xy = torch.where(inside[..., None], xy, self.tile_map.lower + self.tile_map.size / 2)
        heights = self.height(safe_xy)
        result["valid"] = inside & torch.isfinite(heights)
        result["height"] = torch.where(result["valid"], heights, float("nan"))
        return result


@configclass
class PGMTTerrainImporterCfg(TerrainImporterCfg):
    class_type: type = PGMTTerrainImporter
    terrain_type: str = "usd"
    asset_directory: str = str(Path(__file__).resolve().parents[1] / "assets/terrain/pgmt_stage2_v5")
    env_spacing: float = 0.0
    family_proportions: dict[str, float] | None = None
