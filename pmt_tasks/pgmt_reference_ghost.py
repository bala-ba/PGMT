"""A visual-only robot driven by the already-placed PGMT reference poses.

Only mesh geometry is copied from the robot asset. No articulation, rigid body,
joint, collider, sensor, or physics property is created for this display.
"""
import numpy as np


class PGMTReferenceGhost:
    def __init__(self, stage, robot_usd, body_names, *, visible=True,
                 prim_path="/World/PGMTStairsReference"):
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

        self.stage, self.path = stage, prim_path
        self.visible, self.side_by_side = visible, False
        self.offset = np.zeros(3)
        self.positions = self.quaternions = None
        self.root = UsdGeom.Xform.Define(stage, prim_path)
        self.ops = []
        self.mesh_count = 0
        source = Usd.Stage.Open(str(robot_usd))
        root = source.GetDefaultPrim()
        cache = UsdGeom.XformCache()
        material = UsdShade.Material.Define(stage, prim_path + "/ReferenceMaterial")
        shader = UsdShade.Shader.Define(stage, prim_path + "/ReferenceMaterial/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0., .65, .85))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0., .12, .16))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.7)
        self.opacity = shader.CreateInput("opacity", Sdf.ValueTypeNames.Float)
        self.opacity.Set(.4)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        for name in body_names:
            body = source.GetPrimAtPath(root.GetPath().AppendChild(name))
            if not body.IsValid():
                raise ValueError(f"Reference display asset is missing body {name}")
            display = UsdGeom.Xform.Define(stage, f"{prim_path}/{name}")
            self.ops.append(display.MakeMatrixXform())
            # Fixed children (head, hands, etc.) remain attached to this body.
            for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
                if not prim.IsA(UsdGeom.Mesh) or "visuals" not in str(prim.GetPath()).split("/"):
                    continue
                src = UsdGeom.Mesh(prim)
                dst = UsdGeom.Mesh.Define(stage, f"{prim_path}/{name}/Visual_{self.mesh_count}")
                # Copy geometry values only, never references or physics APIs.
                for attribute in ("points", "faceVertexCounts", "faceVertexIndices", "normals",
                                  "extent", "orientation", "subdivisionScheme", "doubleSided",
                                  "holeIndices", "cornerIndices", "cornerSharpnesses",
                                  "creaseIndices", "creaseLengths", "creaseSharpnesses"):
                    value = prim.GetAttribute(attribute).Get()
                    if value is not None:
                        dst.GetPrim().GetAttribute(attribute).Set(value)
                dst.SetNormalsInterpolation(src.GetNormalsInterpolation())
                relative, _ = cache.ComputeRelativeTransform(prim, body)
                dst.MakeMatrixXform().Set(relative)
                UsdShade.MaterialBindingAPI.Apply(dst.GetPrim()).Bind(material)
                self.mesh_count += 1
        self._apply_visibility()
        self.assert_visual_only()

    def assert_visual_only(self):
        from pxr import Usd, UsdPhysics

        for prim in Usd.PrimRange(self.root.GetPrim()):
            if (prim.IsA(UsdPhysics.Joint)
                or any("physics" in api.lower() or "physx" in api.lower() for api in prim.GetAppliedSchemas())
                or any(p.GetName().startswith(("physics:", "physx")) for p in prim.GetProperties())):
                raise AssertionError(f"Physics appeared in reference display: {prim.GetPath()}")

    def _apply_visibility(self):
        self.root.GetVisibilityAttr().Set("inherited" if self.visible else "invisible")

    def toggle(self):
        self.visible = not self.visible
        self._apply_visibility()

    def set_side_by_side(self, enabled):
        self.side_by_side = enabled
        self.offset[:] = (0., 1., 0.) if enabled else (0., 0., 0.)
        self.opacity.Set(.85 if enabled else .4)
        if self.positions is not None:
            self._apply_pose()

    def update(self, reference):
        # PGMTState.reference() owns the frame selection and the episode's
        # shared world transform. This viewer does not sample or advance it.
        self.positions = reference["body_pos"][0].detach().cpu().numpy().copy()
        self.quaternions = reference["body_quat"][0].detach().cpu().numpy().copy()
        self._apply_pose()

    def _apply_pose(self):
        from pxr import Gf

        for op, p, q in zip(self.ops, self.positions + self.offset, self.quaternions):
            matrix = Gf.Matrix4d(1.)
            q = q / np.linalg.norm(q)
            matrix.SetRotate(Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:]))))
            matrix.SetTranslateOnly(Gf.Vec3d(*map(float, p)))
            op.Set(matrix)

    def audit(self):
        """Check rendered world poses and absence of physics on the display."""
        from pxr import UsdGeom

        self.assert_visual_only()
        cache = UsdGeom.XformCache()
        position_error = orientation_error = 0.
        for op, p, q in zip(self.ops, self.positions + self.offset, self.quaternions):
            matrix = cache.GetLocalToWorldTransform(op.GetAttr().GetPrim())
            position_error = max(position_error, float(np.max(np.abs(np.asarray(matrix.ExtractTranslation()) - p))))
            rotation = matrix.ExtractRotationQuat()
            actual = np.array([rotation.GetReal(), *rotation.GetImaginary()])
            expected = q.astype(np.float64) / np.linalg.norm(q.astype(np.float64))
            orientation_error = max(orientation_error, float(min(np.linalg.norm(actual-expected),
                                                               np.linalg.norm(actual+expected))))
        if position_error > 2e-5 or orientation_error > 2e-5:
            raise AssertionError(f"Reference display pose mismatch: {position_error}, {orientation_error}")
        return dict(bodies=len(self.ops), visual_meshes=self.mesh_count, physics_prims=0,
                    body_position_error_m=position_error, quaternion_error=orientation_error,
                    visible=self.visible, display_offset_m=self.offset.tolist(),
                    source="PGMTState.reference(); same frame and episode transform; no terrain height adjustment")
