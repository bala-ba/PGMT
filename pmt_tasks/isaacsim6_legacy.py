"""Opt-in import-name bridge for legacy Isaac Lab on the local Isaac Sim 6.

Only the PhysX tensor Python module moved (impl.api -> api). No physics,
integration, observations or policy math is replaced by this bridge.
Install before AppLauncher, since Kit imports Isaac Lab asset extensions.
"""
import importlib
import importlib.abc
import importlib.util
import sys


class _TensorAliases(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "omni.physics.tensors.impl":
            return importlib.util.spec_from_loader(fullname, self, is_package=True)
        if fullname == "omni.physics.tensors.impl.api":
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if module.__name__.endswith(".api"):
            modern = importlib.import_module("omni.physics.tensors.api")
            module.__dict__.update({k: v for k, v in vars(modern).items() if not k.startswith("__")})
            # Legacy Isaac Lab eagerly evaluates these annotations even in a
            # rigid-only scene. Deformable training is outside this bridge's scope.
            module.SoftBodyView = modern.DeformableBodyView
            module.SoftBodyMaterialView = modern.DeformableMaterialView


def install_tensor_import_bridge():
    sys.meta_path.insert(0, _TensorAliases())


def install_rigid_scene_schema_bridge():
    """Preserve the generic material binder's deformable/nondeformable test.

    Called only after Kit starts. PGMT contains rigid colliders and does not
    instantiate any of the legacy deformable-object creation APIs.
    """
    from pxr import PhysxSchema, Usd

    if not hasattr(PhysxSchema, "PhysxDeformableBodyAPI"):
        # Sim 6 registers the replacement in USD but does not expose a Python
        # wrapper. Prim.HasAPI accepts its real Tf.Type for the rigid material
        # binder's predicate; this deliberately provides no deformable creator.
        schema_type = Usd.SchemaRegistry.GetTypeFromSchemaTypeName("PhysxBaseDeformableBodyAPI")
        if schema_type.isUnknown:
            raise RuntimeError("Sim 6 deformable schema predicate is unavailable")
        PhysxSchema.PhysxDeformableBodyAPI = schema_type
