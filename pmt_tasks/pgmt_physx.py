"""Scoped PGMT shared-terrain broadphase setup before the first physics reset."""
from contextlib import contextmanager


@contextmanager
def shared_terrain_scene(bits=0):
    """Keep environment collision IDs, but avoid global-mesh AABB expansion.

    IsaacLab constructs InteractiveScene directly. Temporarily replace only
    that factory with a subclass, restoring it even if construction fails.
    The native Cloner completes before the override; sim.reset happens after.
    """
    if not isinstance(bits, int) or not 0 <= bits <= 4:
        raise ValueError("PGMT environment AABB bits must be an integer in [0,4]")
    from isaaclab.envs import manager_based_env
    from pxr import UsdPhysics

    original = manager_based_env.InteractiveScene
    evidence = []

    class SharedTerrainScene(original):
        def __init__(self, cfg):
            super().__init__(cfg)
            if not cfg.replicate_physics or not cfg.filter_collisions:
                raise ValueError("PGMT requires physics replication and environment collision filtering")
            for prim in self.stage.Traverse():
                if prim.IsA(UsdPhysics.Scene):
                    attribute = prim.GetAttribute("physxScene:envIdInBoundsBitCount")
                    if not attribute.IsValid():
                        raise RuntimeError("PhysX environment AABB-bit attribute is unavailable")
                    before = attribute.Get()
                    if not attribute.Set(bits) or attribute.Get() != bits:
                        raise RuntimeError("Could not configure PGMT shared-terrain broadphase")
                    evidence.append(dict(scene=str(prim.GetPath()), before=before, after=attribute.Get(),
                                         replicate_physics=cfg.replicate_physics,
                                         filter_collisions=cfg.filter_collisions))
            if not evidence:
                raise RuntimeError("No PhysicsScene found before PGMT physics initialization")

    manager_based_env.InteractiveScene = SharedTerrainScene
    try:
        yield evidence
    finally:
        manager_based_env.InteractiveScene = original
