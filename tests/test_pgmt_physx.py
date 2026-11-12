import sys
from types import ModuleType, SimpleNamespace

import pytest

from pmt_tasks.pgmt_physx import shared_terrain_scene
from scripts.run_pgmt_p4_long import is_oom


def fake_native(monkeypatch, *, valid=True):
    attribute = SimpleNamespace(value=4)
    attribute.Get = lambda: attribute.value
    attribute.IsValid = lambda: valid
    def set_value(value):
        attribute.value = value
        return True
    attribute.Set = set_value
    prim = SimpleNamespace(IsA=lambda _: True, GetAttribute=lambda _: attribute,
                           GetPath=lambda: '/physicsScene')
    class NativeScene:
        def __init__(self, cfg):
            self.stage = SimpleNamespace(Traverse=lambda: [prim])
            # Native Cloner writes 4 during construction.
            attribute.value = 4
    module = ModuleType('isaaclab.envs.manager_based_env')
    module.InteractiveScene = NativeScene
    envs = ModuleType('isaaclab.envs')
    envs.manager_based_env = module
    pxr = ModuleType('pxr')
    pxr.UsdPhysics = SimpleNamespace(Scene=object())
    monkeypatch.setitem(sys.modules, 'isaaclab.envs', envs)
    monkeypatch.setitem(sys.modules, 'pxr', pxr)
    return module, NativeScene, attribute


def test_shared_scene_applies_after_cloner_and_restores_factory(monkeypatch):
    module, native, attribute = fake_native(monkeypatch)
    cfg = SimpleNamespace(replicate_physics=True, filter_collisions=True)
    with shared_terrain_scene(0) as evidence:
        module.InteractiveScene(cfg)
        assert attribute.value == 0
        assert evidence == [dict(scene='/physicsScene', before=4, after=0,
                                 replicate_physics=True, filter_collisions=True)]
    assert module.InteractiveScene is native


@pytest.mark.parametrize('valid,filtered', [(False, True), (True, False)])
def test_shared_scene_fails_closed_and_restores_factory(monkeypatch, valid, filtered):
    module, native, _ = fake_native(monkeypatch, valid=valid)
    with pytest.raises((RuntimeError, ValueError)):
        with shared_terrain_scene(0):
            module.InteractiveScene(SimpleNamespace(replicate_physics=True, filter_collisions=filtered))
    assert module.InteractiveScene is native


def test_oom_fallback_does_not_hide_numerical_or_contract_failures():
    assert is_oom({'error': 'OutOfMemoryError(CUDA out of memory)'}, '')
    assert is_oom({'status': 'failed_physics'}, 'PhysX failed to allocate 4096 bytes')
    assert not is_oom({'error': 'AssertionError(nonfinite policy)'}, '')
    assert not is_oom({'error': 'Training contract mismatch'}, '')
