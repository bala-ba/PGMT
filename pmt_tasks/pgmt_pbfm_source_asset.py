"""Author an Isaac asset from the audited native P-BFM collision export."""
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh


def build_source_asset(config, output, author_mesh, colors):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
    import yaml

    config, output = Path(config), Path(output)
    exported = output / "source_export.json"
    if not exported.exists():
        raise FileNotFoundError(
            "Export P-BFM first with scripts/export_pgmt_pbfm_source.py "
            f"--config {config} --output {output} (MJLab 1.4 / MuJoCo 3.8).")
    source = json.loads(exported.read_text())
    spec = yaml.safe_load(config.read_text())
    if source["spec"] != spec:
        raise ValueError("Source export does not match the build configuration")
    if hashlib.sha256((output / "source.mjb").read_bytes()).hexdigest() != source["source"]["model_sha256"]:
        raise ValueError("Native model checksum mismatch")
    families = spec["families"]
    stage = Usd.Stage.CreateNew(str(output / "terrain.usdc"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    UsdGeom.Xform.Define(stage, "/World/Terrain")
    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    UsdLux.DomeLight.Define(stage, "/World/Sky").CreateIntensityAttr(180.)
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(900.)
    sun.CreateAngleAttr(1.)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(25, -35, -35))
    materials = {}
    for family, color in zip(families + ["guard", "border", "walls"],
                             list(colors) + [(.40, .43, .40), (.45, .48, .51), (.5, .5, .5)]):
        mat = UsdShade.Material.Define(stage, f"/World/Looks/{family}")
        shader = UsdShade.Shader.Define(stage, f"/World/Looks/{family}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.85)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        physics = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        physics.CreateStaticFrictionAttr(1.)
        physics.CreateDynamicFrictionAttr(1.)
        physics.CreateRestitutionAttr(0.)
        materials[family] = mat
    regions = []
    for index, record in enumerate(source["patches"] + source["auxiliary_surfaces"]):
        path = output / record["mesh_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["mesh_sha256"]:
            raise ValueError(f"Source mesh checksum mismatch: {path}")
        mesh = trimesh.load_mesh(path, process=False)
        UsdGeom.Xform.Define(stage, record["prim_path"].rsplit("/", 1)[0])
        prim = UsdGeom.Xform.Define(stage, record["prim_path"])
        prim.AddTranslateOp().Set(Gf.Vec3d(*record["center"]))
        author_mesh(stage, record["prim_path"] + "/Mesh", mesh, materials[record["family"]])
        if "level" in record:
            prim.GetPrim().SetCustomData(dict(family=record["family"], level=record["level"],
                                             patch_id=0, difficulty=record["difficulty"]))
            bounds = np.array(record["world_bounds"])
            record["geometry"] = dict(vertices=len(mesh.vertices), triangles=len(mesh.faces),
                                      min_height_m=float(bounds[0, 2]), max_height_m=float(bounds[1, 2]))
            regions.append(dict(name=record["region_name"], family=record["family"],
                family_index=record["family_index"], level=record["level"], difficulty=record["difficulty"],
                center=record["center"], bounds=bounds[:, :2].tolist(),
                prim_path=record["prim_path"].rsplit("/", 1)[0], patch_indices=[index]))
    # Every native tile has an exact z=0 border, including both sides of seams.
    lo = np.asarray(source["map_bounds"][0])
    size = np.asarray(spec["patch_size"])
    seams = []
    for axis in (0, 1):
        other = 1 - axis
        counts = [spec["num_rows"], spec["num_cols"]]
        for i in range(1, counts[axis]):
            for j in range(counts[other]):
                for offset in (-.001, 0., .001):
                    xy = lo.copy()
                    xy[axis] += i * size[axis] + offset
                    xy[other] += (j + .5) * size[other]
                    seams.append([*xy.tolist(), 0.])
    camera = UsdGeom.Camera.Define(stage, "/World/InspectionCamera")
    camera.CreateClippingRangeAttr(Gf.Vec2f(.05, 2000.))
    camera.CreateFocalLengthAttr(25.)
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(50, -65, 80), Gf.Vec3d(0, 0, 0), Gf.Vec3d(0, 0, 1))
    UsdGeom.Xformable(camera).AddTransformOp().Set(view.GetInverse())
    stage.GetRootLayer().Save()
    root = Path(__file__).resolve().parents[1]
    sources = [root / "scripts/export_pgmt_pbfm_source.py", root / "pmt_tasks/pgmt_mujoco_mesh.py",
               Path(__file__).resolve(), root / "scripts/build_pgmt_patch_terrain.py"]
    manifest = dict(source, schema="pmt.pgmt.terrain.patches.v1", version=spec["version"], preview_only=False,
        native_interfaces="P-BFM training_mix / MJLab 1.4 / MuJoCo 3.8 collision geometry",
        region_count=len(regions), patch_count=len(source["patches"]), regions=regions,
        auxiliary_surface_count=len(source["auxiliary_surfaces"]), seam_probes=seams,
        asset="terrain.usdc", asset_sha256=hashlib.sha256((output / "terrain.usdc").read_bytes()).hexdigest(),
        configuration_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        generator_sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        spawn_preview_scope="Candidate surface XY/yaw only; PMT validates the full robot pose at reset.",
        validation=dict(source["validation"], geometry="passed", native_collision="pending"))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"SUCCESS: {len(regions)} P-BFM training tiles + {manifest['auxiliary_surface_count']} auxiliary meshes", flush=True)
