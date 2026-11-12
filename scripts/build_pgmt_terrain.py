"""Generate, audit, and export the PGMT terrain curriculum with Isaac Lab.

Example (pmt environment):
  python scripts/build_pgmt_terrain.py --headless --device cpu
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="configs/terrain/pgmt_stage2.yaml")
parser.add_argument("--output", default="assets/terrain/pgmt_stage2_v5")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
output = Path(args.output).resolve()
if (output / "manifest.json").exists():
    raise FileExistsError(f"Completed terrain asset already exists: {output}; use a new output version")
output.mkdir(parents=True, exist_ok=True)
launcher = AppLauncher(args)
app = launcher.app

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt
from isaaclab.terrains import TerrainGenerator
from pmt_tasks.pgmt_terrain import (
    FAMILIES, FAMILY_GROUPS, generate_tile, load_spec, make_generator_cfg, parameters, ray_heights, validate_tile,
)

COLORS = ((.45,.60,.70), (.38,.64,.42), (.78,.55,.28), (.56,.43,.70), (.31,.62,.63),
          (.84,.42,.22), (.55,.72,.25))


class AuditedGenerator(TerrainGenerator):
    def __init__(self, cfg):
        self.tiles = []
        super().__init__(cfg, device="cpu")

    def _get_terrain_mesh(self, difficulty, cfg):
        self._current_difficulty = float(difficulty)
        return super()._get_terrain_mesh(difficulty, cfg)

    def _add_sub_terrain(self, mesh, origin, row, col, cfg):
        self.tiles.append(dict(mesh=mesh.copy(), local_origin=origin.copy(), level=row, family=cfg.family,
                               family_index=col, difficulty=self._current_difficulty))
        super()._add_sub_terrain(mesh, origin, row, col, cfg)


def author_mesh(stage, path, mesh, material):
    prim = UsdGeom.Mesh.Define(stage, path)
    prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, dtype=np.float32)))
    prim.CreateFaceVertexCountsAttr(np.full(len(mesh.faces), 3, dtype=np.int32))
    prim.CreateFaceVertexIndicesAttr(np.asarray(mesh.faces, dtype=np.int32).ravel())
    prim.CreateSubdivisionSchemeAttr("none")
    prim.CreateDoubleSidedAttr(True)
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(prim.GetPrim()).CreateApproximationAttr("none")
    UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(material)
    return prim


def main():
    spec = load_spec(args.config)
    cfg = make_generator_cfg(spec, output / "cache")
    generator = AuditedGenerator(cfg)
    stage = Usd.Stage.CreateNew(str(output / "terrain.usdc"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1))
    scene.CreateGravityMagnitudeAttr(9.81)
    dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
    dome.CreateIntensityAttr(450.)
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(2200.)
    sun.CreateAngleAttr(1.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(25,-35,-35))
    materials = []
    for family, color in zip(FAMILIES, COLORS):
        material = UsdShade.Material.Define(stage, f"/World/Looks/{family}")
        shader = UsdShade.Shader.Define(stage, f"/World/Looks/{family}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.85)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        physical = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physical.CreateStaticFrictionAttr(1.)
        physical.CreateDynamicFrictionAttr(1.)
        physical.CreateRestitutionAttr(0.)
        materials.append(material)
    half_map = np.array([spec["size"][0]*spec["num_rows"]/2,
                         spec["size"][1]*spec["num_cols"]/2, 0.])
    records = []
    for tile in generator.tiles:
        level, col, family, d = (tile[k] for k in ("level", "family_index", "family", "difficulty"))
        mesh = tile.pop("mesh")
        local = mesh.copy()
        local.apply_translation([spec["size"][0]/2, spec["size"][1]/2, 0])
        audit = validate_tile(local, spec, family, d)
        translation = np.array([(level+.5)*spec["size"][0], (col+.5)*spec["size"][1], 0.]) - half_map
        origin = tile.pop("local_origin") + translation
        np.testing.assert_allclose(origin, generator.terrain_origins[level,col], atol=1e-8)
        name = f"L{level:02d}_{family}"
        prim_path = f"/World/Terrain/{name}"
        tile_prim = UsdGeom.Xform.Define(stage, prim_path)
        tile_prim.AddTranslateOp().Set(Gf.Vec3d(*translation))
        tile_prim.GetPrim().SetCustomData({"family":family, "level":level, "difficulty":d})
        author_mesh(stage, prim_path+"/Mesh", mesh, materials[col])
        file = output / f"{name}.ply"
        mesh.export(file)
        probes = np.asarray(audit["probes"])
        probes[:,:2] -= np.array(spec["size"])/2
        probes += translation
        record = dict(**tile, name=name, prim_path=prim_path, parameters=parameters(spec,d),
                      family_group=FAMILY_GROUPS[family], family_group_index=FAMILIES.index(FAMILY_GROUPS[family]),
                      center=translation.tolist(), origin=origin.tolist(), geometry=audit,
                      world_probes=probes.tolist(), mesh_file=file.name,
                      mesh_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        if family in ("up_stairs", "up_slope"):
            parent_meshes, parent_origin = generate_tile(d,cfg.sub_terrains[FAMILY_GROUPS[family]])
            parent_mesh = trimesh.util.concatenate(parent_meshes)
            local_probes = np.asarray(audit["probes"])
            mirror_error = ray_heights(parent_mesh,local_probes[:,:2])+local_probes[:,2]
            np.testing.assert_allclose(mirror_error,0.,atol=2e-6)
            np.testing.assert_allclose(origin[:2],translation[:2],atol=1e-8)
            np.testing.assert_allclose(origin[2],-parent_origin[2],atol=2e-6)
            record.update(inverted=True, outward_travel="up", shape=audit["native_generator"],
                          mirror_validation=dict(passed=True,reference_family=FAMILY_GROUPS[family],
                                                 same_difficulty=d,max_height_error_m=float(np.abs(mirror_error).max())))
        records.append(record)
        print(f"PASS {name}: d={d:.4f}, {audit['triangles']} triangles", flush=True)
    # Preserve the native generator's surrounding border, separate from all tiles.
    border = generator.terrain_meshes[-1].copy()
    border.apply_translation(-half_map)
    author_mesh(stage, "/World/TerrainBorder", border, materials[0])
    border.export(output / "border.ply")
    # A portable camera is part of the asset; viewer buttons may choose closeups.
    camera = UsdGeom.Camera.Define(stage, "/World/InspectionCamera")
    camera.CreateClippingRangeAttr(Gf.Vec2f(.1, 2000.))
    eye, target = Gf.Vec3d(185,-210,210), Gf.Vec3d(0,0,0)
    view = Gf.Matrix4d().SetLookAt(eye,target,Gf.Vec3d(0,0,1))
    UsdGeom.Xformable(camera).AddTransformOp().Set(view.GetInverse())
    stage.GetRootLayer().Save()
    # Explicit endpoint checks: native row sampling never guarantees d=0 or d=1.
    endpoints = []
    for family, sub_cfg in cfg.sub_terrains.items():
        for d in (0.,1.):
            meshes, _ = generate_tile(d, sub_cfg)
            audit = validate_tile(trimesh.util.concatenate(meshes), spec, family, d)
            endpoints.append(dict(family=family, difficulty=d, parameters=parameters(spec,d), geometry=audit))
    manifest = dict(schema="pmt.pgmt.terrain.v1", version=spec["version"], spec=spec,
        configuration_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        geometry_source_sha256=next(iter(cfg.sub_terrains.values())).geometry_sha256,
        tile_count=len(records), layout="row=level (x), column=family (y)",
        asset="terrain.usdc", asset_sha256=hashlib.sha256((output/"terrain.usdc").read_bytes()).hexdigest(),
        tiles=records, endpoint_checks=endpoints,
        validation={"geometry":"passed", "mesh_ray_probes":sum(len(r["world_probes"]) for r in records),
                    "isaac_physx_collision":"pending viewer validation"})
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"SUCCESS: {len(records)} terrain tiles + {len(endpoints)} endpoint audits exported to {output}",flush=True)


try:
    main()
finally:
    app.close()
