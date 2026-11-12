"""Small, finite plane asset for independent full-motion priority evaluation."""
import hashlib
import json
from pathlib import Path


def build_flat_eval_asset(directory, num_envs):
    import numpy as np
    import trimesh
    from pxr import Usd, UsdGeom, UsdPhysics, Vt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Tile the coplanar mesh: kilometre-sized individual triangles trigger
    # PhysX cooking stability warnings. Keep ample space around every origin.
    half = int(np.ceil((np.ceil(np.sqrt(num_envs)) * 50 + 128) / 16)) * 16
    axis = np.arange(-half, half + 1, 16, dtype=np.float32)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size, dtype=np.float32)))
    i, j = np.meshgrid(np.arange(len(axis)-1), np.arange(len(axis)-1), indexing="ij")
    a = (i * len(axis) + j).ravel()
    faces = np.concatenate((np.column_stack((a, a+len(axis), a+len(axis)+1)),
                            np.column_stack((a, a+len(axis)+1, a+1)))).astype(np.int32)
    mesh_path, usd_path = directory / "flat.ply", directory / "flat.usda"
    trimesh.Trimesh(vertices=points, faces=faces, process=False).export(mesh_path)
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/World/Terrain/flat/Mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    stage.GetRootLayer().Save()
    manifest = dict(asset=usd_path.name, asset_sha256=hashlib.sha256(usd_path.read_bytes()).hexdigest(),
        spec=dict(num_rows=1, num_cols=1, families=["flat"], size=[float(2*half), float(2*half)]),
        tiles=[dict(name="flat", family="flat", family_index=0, family_group_index=0, level=0,
                    difficulty=0., origin=[0., 0., 0.], center=[0., 0., 0.],
                    mesh_file=mesh_path.name, mesh_sha256=hashlib.sha256(mesh_path.read_bytes()).hexdigest())])
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return str(directory)
