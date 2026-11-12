"""Build versioned Stage2 assets (v8 native P-BFM; v6/v7 still supported)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default=str(ROOT / "configs/terrain/pgmt_stage2_v8.yaml"))
parser.add_argument("--output", help="Defaults to assets/terrain/<config stem>")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
output = Path(args.output or ROOT / "assets/terrain" / Path(args.config).stem).resolve()
if (output / "manifest.json").exists():
    raise FileExistsError(f"Refusing to overwrite completed terrain asset: {output}")
output.mkdir(parents=True, exist_ok=True)
app = AppLauncher(args).app

import numpy as np
import trimesh
import yaml
import isaaclab.terrains as tg
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt
from pmt_tasks.runtime_options import run_and_close

COLORS = ((.40, .57, .65), (.78, .48, .22), (.38, .66, .43), (.52, .39, .67), (.27, .60, .64),
          (.66, .60, .30), (.68, .36, .35))


def author_mesh(stage, path, mesh, material):
    obj = UsdGeom.Mesh.Define(stage, path)
    obj.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(mesh.vertices.astype(np.float32)))
    obj.CreateFaceVertexCountsAttr(np.full(len(mesh.faces), 3, np.int32))
    obj.CreateFaceVertexIndicesAttr(mesh.faces.astype(np.int32).ravel())
    obj.CreateSubdivisionSchemeAttr("none")
    obj.CreateDoubleSidedAttr(True)
    UsdPhysics.CollisionAPI.Apply(obj.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(obj.GetPrim()).CreateApproximationAttr("none")
    UsdShade.MaterialBindingAPI.Apply(obj.GetPrim()).Bind(material)


def main():
    kind = yaml.safe_load(Path(args.config).read_text())["type"]
    if kind == "pgmt_pbfm_source":
        from pmt_tasks.pgmt_pbfm_source_asset import build_source_asset
        return build_source_asset(args.config, output, author_mesh, COLORS)
    if kind == "pgmt_pbfm_patch_regions":
        from pmt_tasks import pgmt_pbfm_terrain as geometry
        repeated_families, inverted_families = geometry.REPEATED_FAMILIES, geometry.INVERTED_FAMILIES
    else:
        from pmt_tasks import pgmt_patch_terrain as geometry
        repeated_families, inverted_families = ("flat", "stairs", "slope"), ("stairs", "slope")
    spec = geometry.load_patch_spec(args.config)
    families, make_patch = spec["families"], geometry.make_patch
    stage = Usd.Stage.CreateNew(str(output / "terrain.usdc"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    UsdGeom.Xform.Define(stage, "/World/Terrain")
    UsdGeom.Xform.Define(stage, "/World/TerrainBorder")
    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1)); scene.CreateGravityMagnitudeAttr(9.81)
    UsdLux.DomeLight.Define(stage, "/World/Sky").CreateIntensityAttr(180.)
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(900.); sun.CreateAngleAttr(1.)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(25, -35, -35))
    materials = []
    for family, color in zip(families, COLORS):
        mat = UsdShade.Material.Define(stage, f"/World/Looks/{family}")
        shader = UsdShade.Shader.Define(stage, f"/World/Looks/{family}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.85)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        physics = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        physics.CreateStaticFrictionAttr(1.); physics.CreateDynamicFrictionAttr(1.); physics.CreateRestitutionAttr(0.)
        materials.append(mat)
    extent = np.array(spec["region_size"]) * [spec["num_rows"], spec["num_cols"]]
    half = extent / 2
    size, grid = np.array(spec["patch_size"]), spec["patch_grid"]
    region_size = np.array(spec["region_size"])
    rng = np.random.default_rng(spec["seed"])
    levels_d = (np.arange(spec["num_rows"]) + rng.random(spec["num_rows"])) / spec["num_rows"]
    regions, patches, endpoints = [], [], []
    # These native shapes are identical within a level. Reuse their audited
    # geometry; boxes/rough are regenerated with independent patch seeds.
    repeated = {}
    for level, difficulty in enumerate(levels_d):
        for family_index, family in enumerate(families):
            region_name = f"L{level:02d}_{family}"
            region_path = f"/World/Terrain/{region_name}"
            UsdGeom.Xform.Define(stage, region_path)
            region_corner = np.array([level, family_index]) * region_size - half
            indices = []
            for row in range(grid[0]):
                for col in range(grid[1]):
                    patch_id = row * grid[1] + col
                    key = (level, family)
                    if family in repeated_families and key in repeated:
                        # Sampling still has its own seed even for repeated geometry.
                        mesh, origin, audit, _, params, _ = repeated[key]
                        from pmt_tasks.pgmt_patch_terrain import seed_for
                        from pmt_tasks.pgmt_terrain import ray_heights
                        sample_rng = np.random.default_rng(seed_for(spec, "preview_xy_yaw", level, family, patch_id))
                        preview = spec["spawn_preview"]
                        xy = sample_rng.uniform(preview["boundary_margin"], size - preview["boundary_margin"],
                                                size=(preview["samples_per_patch"], 2))
                        samples = np.column_stack((xy, ray_heights(mesh, xy), sample_rng.uniform(*preview["yaw_range"], size=len(xy))))
                        texture_seed = seed_for(spec, "texture", level, family, patch_id)
                    else:
                        result = make_patch(spec, level, family, patch_id, float(difficulty))
                        mesh, origin, audit, samples, params, texture_seed = result
                        if family in repeated_families:
                            repeated[key] = result
                    corner = np.r_[region_corner + [row, col] * size, 0.]
                    center = corner + np.r_[size / 2, 0.]
                    centered = mesh.copy(); centered.apply_translation(-np.r_[size / 2, 0.])
                    name = f"{region_name}_P{row}{col}"
                    prim_path = f"{region_path}/P{row}{col}"
                    prim = UsdGeom.Xform.Define(stage, prim_path)
                    prim.AddTranslateOp().Set(Gf.Vec3d(*center))
                    prim.GetPrim().SetCustomData(dict(family=family, level=level, patch_id=patch_id, difficulty=float(difficulty)))
                    author_mesh(stage, prim_path + "/Mesh", centered, materials[family_index])
                    file = output / f"{name}.ply"; centered.export(file)
                    probes = np.asarray(audit["probes"]) + corner
                    world_samples = samples.copy(); world_samples[:, :3] += corner
                    record = dict(name=name, region_name=region_name, family=family, family_index=family_index,
                                  level=level, patch_id=patch_id, patch_row=row, patch_col=col,
                                  difficulty=float(difficulty), texture_seed=texture_seed, parameters=params,
                                  center=center.tolist(), origin=(origin + corner).tolist(),
                                  world_bounds=(mesh.bounds + corner).tolist(), geometry=audit,
                                  world_probes=probes.tolist(), preview_xyzyaw=world_samples.tolist(),
                                  prim_path=prim_path, mesh_file=file.name,
                                  mesh_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                                  inverted=family in inverted_families)
                    patches.append(record); indices.append(len(patches) - 1)
            regions.append(dict(name=region_name, family=family, family_index=family_index, level=level,
                                difficulty=float(difficulty), center=np.r_[region_corner + region_size / 2, 0.].tolist(),
                                bounds=[region_corner.tolist(), (region_corner + region_size).tolist()],
                                prim_path=region_path, patch_indices=indices))
            print(f"PASS {region_name}: 16 patches, d={difficulty:.4f}", flush=True)
    # Four outer strips only: never put a zero-height plane over the pits.
    border = spec["border_width"]
    strips = [((-half[0]-border, -half[1]-border), (extent[0]+2*border, border)),
              ((-half[0]-border, half[1]), (extent[0]+2*border, border)),
              ((-half[0]-border, -half[1]), (border, extent[1])),
              ((half[0], -half[1]), (border, extent[1]))]
    for i, (corner, dims) in enumerate(strips):
        cfg = tg.MeshPlaneTerrainCfg(size=tuple(dims))
        border_meshes, _ = cfg.function(0., cfg)
        mesh = trimesh.util.concatenate(border_meshes)
        mesh.apply_translation([*corner, 0.])
        author_mesh(stage, f"/World/TerrainBorder/Strip{i}", mesh, materials[0])
    seams = []
    nx, ny = spec["num_rows"] * grid[0], spec["num_cols"] * grid[1]
    for i in range(1, nx):
        for j in range(ny):
            for offset in (-.001, 0., .001):
                seams.append([-half[0] + i * size[0] + offset, -half[1] + (j + .5) * size[1], 0.])
    for j in range(1, ny):
        for i in range(nx):
            for offset in (-.001, 0., .001):
                seams.append([-half[0] + (i + .5) * size[0], -half[1] + j * size[1] + offset, 0.])
    camera = UsdGeom.Camera.Define(stage, "/World/InspectionCamera")
    camera.CreateClippingRangeAttr(Gf.Vec2f(.05, 2000.)); camera.CreateFocalLengthAttr(25.)
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(180, -240, 270), Gf.Vec3d(0, 0, 0), Gf.Vec3d(0, 0, 1))
    UsdGeom.Xformable(camera).AddTransformOp().Set(view.GetInverse())
    stage.GetRootLayer().Save()
    for family in families:
        for d in (0., 1.):
            _, origin, audit, _, params, _ = make_patch(spec, -1, family, 0, d)
            endpoints.append(dict(family=family, difficulty=d, origin=origin.tolist(), parameters=params, geometry=audit))
    old_paths = ["configs/terrain/pgmt_stage2.yaml", "pmt_tasks/pgmt_terrain.py", "scripts/build_pgmt_terrain.py",
                 "scripts/view_pgmt_terrain.py", "assets/terrain/pgmt_stage2_v5/manifest.json", "assets/terrain/pgmt_stage2_v5/terrain.usdc"]
    manifest = dict(schema="pmt.pgmt.terrain.patches.v1", version=spec["version"],
                    preview_only=spec["type"] != "pgmt_pbfm_patch_regions", spec=spec,
                    native_interfaces=geometry.NATIVE_INTERFACES, outer_border_interface="MeshPlaneTerrainCfg",
                    region_count=len(regions), patch_count=len(patches), regions=regions, patches=patches,
                    map_bounds=[(-half).tolist(), half.tolist()], seam_probes=seams, endpoint_checks=endpoints,
                    level_difficulty=levels_d.tolist(), asset="terrain.usdc",
                    asset_sha256=hashlib.sha256((output / "terrain.usdc").read_bytes()).hexdigest(),
                    configuration_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                    generator_sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                       {ROOT / "pmt_tasks/pgmt_patch_terrain.py", Path(geometry.__file__).resolve(),
                                        Path(__file__).resolve(), ROOT / "pmt_tasks/pgmt_terrain.py"}},
                    preserved_v5={p:hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in old_paths},
                    spawn_preview_scope="Uniform candidate XY/yaw and surface Z only; full robot placement and training reset are not implemented by this preview.",
                    validation=dict(geometry="passed", native_collision="pending"))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"SUCCESS: {len(regions)} regions / {len(patches)} patches / {len(seams)} seam probes", flush=True)


run_and_close(main, app)
