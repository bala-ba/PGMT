"""Finite, externally guarded PhysX initialization probes; no policy or PPO."""
import argparse
from copy import deepcopy
from dataclasses import fields
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.environ.get('PMT_PGMT_PHYSICS_WORKER') != '1':
    from pgmt_process_guard import run_guarded
    raise SystemExit(run_guarded([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], ROOT))

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--log_dir', required=True)
parser.add_argument('--num_envs', type=int, default=160)
parser.add_argument('--spacing', type=float, default=0.0)
parser.add_argument('--bootstrap_z', type=float, default=0.78)
parser.add_argument('--classic_filter', action='store_true')
parser.add_argument('--env_id_bits', type=int, default=None)
parser.add_argument('--disable_terrain_collision', action='store_true')
parser.add_argument('--steps', type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
output = Path(args.log_dir).resolve()
output.mkdir(parents=True, exist_ok=True)
status_path = output / 'training_status.json'
if status_path.exists():
    parser.error(f'Refusing to overwrite {output}')
status = dict(pid=os.getpid(), status='launching', arguments=vars(args), snapshots={})
def save(**values):
    status.update(values)
    temp = status_path.with_suffix('.tmp')
    temp.write_text(json.dumps(status, indent=2) + '\n')
    temp.replace(status_path)
save()
from pmt_tasks.isaacsim6_legacy import install_tensor_import_bridge
install_tensor_import_bridge()
app = AppLauncher(args, multi_gpu=False, fast_shutdown=True).app


def main():
    from pmt_tasks.isaacsim6_legacy import install_rigid_scene_schema_bridge
    install_rigid_scene_schema_bridge()
    import numpy as np
    import torch
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils, PhysicsSchemaTools
    import omni.physx
    from omni.physx.bindings import _physx
    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.scene import InteractiveScene
    from pmt_tasks.builder import build_env_cfg
    from pmt_tasks.env_cfgs.pgmt.stage1 import PGMTObservationsCfg

    def snapshot(env, name, physics=False):
        item = dict(gpu=subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv'],
            capture_output=True, text=True, check=True).stdout,
            torch_allocated=torch.cuda.memory_allocated(args.device),
            torch_reserved=torch.cuda.memory_reserved(args.device))
        if physics:
            interface = _physx.acquire_physx_statistics_interface()
            stage_id = UsdUtils.StageCache.Get().Insert(env.scene.stage).ToLongInt()
            scene_stats = {}
            for prim in env.scene.stage.Traverse():
                if prim.IsA(UsdPhysics.Scene):
                    stats = _physx.PhysicsSceneStats()
                    ok = interface.get_physx_scene_statistics(
                        stage_id, PhysicsSchemaTools.sdfPathToInt(prim.GetPath()), stats)
                    scene_stats[str(prim.GetPath())] = dict(ok=ok, **{
                        key: getattr(stats, key) for key in dir(stats)
                        if not key.startswith('_') and isinstance(getattr(stats, key), (int, float))})
            item['physx'] = scene_stats
        status['snapshots'][name] = item
        save(phase=name)
        print('PGMT_MEMORY_SNAPSHOT', name, json.dumps(item), flush=True)

    def prestartup(env, env_ids):
        if args.env_id_bits is not None:
            for prim in env.scene.stage.Traverse():
                if prim.IsA(UsdPhysics.Scene):
                    prim.GetAttribute('physxScene:envIdInBoundsBitCount').Set(args.env_id_bits)
        if args.classic_filter:
            env.scene.filter_collisions([env.scene.terrain.cfg.prim_path])
        if args.disable_terrain_collision:
            ground = env.scene.stage.GetPrimAtPath(env.scene.terrain.cfg.prim_path)
            for prim in Usd.PrimRange(ground):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
        cache = UsdGeom.XformCache()
        pos = [list(cache.GetLocalToWorldTransform(env.scene.stage.GetPrimAtPath(
            f'{path}/Robot')).ExtractTranslation()) for path in env.scene.env_prim_paths]
        xyz = np.asarray(pos)
        np.save(output / 'bootstrap_positions.npy', xyz)
        attrs = {str(p.GetPath()): {a.GetName(): str(a.Get()) for a in p.GetAttributes()
                 if 'envId' in a.GetName()} for p in env.scene.stage.Traverse() if p.IsA(UsdPhysics.Scene)}
        save(bootstrap=dict(unique_positions=len(np.unique(xyz, axis=0)),
                            min=xyz.min(0).tolist(), max=xyz.max(0).tolist(), scene_attributes=attrs))
        snapshot(env, 'prestartup')

    cfg = build_env_cfg('pgmt_stage2')
    cfg.scene.num_envs, cfg.scene.env_spacing, cfg.seed = args.num_envs, args.spacing, 722
    cfg.sim.device = cfg.commands.motion.storage_device = args.device
    cfg.sim.enable_scene_query_support = True
    cfg.sim.use_fabric = True
    cfg.observations = PGMTObservationsCfg()
    cfg.scene.height_scanner = None
    cfg.scene.robot.init_state.pos = (0., 0., args.bootstrap_z)
    old = cfg.scene.robot.spawn
    shared = {f.name: deepcopy(getattr(old, f.name)) for f in fields(sim_utils.UsdFileCfg)
              if f.name not in ('func', 'usd_path') and hasattr(old, f.name)}
    cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
        usd_path=str(ROOT / 'logs/pgmt_p1_acceptance/native_robot/main.usd'), **shared)
    # Observe the scene before ManagerBasedEnv's first sim.reset(), without
    # prestartup randomization (Isaac Lab forbids those with replication).
    original_scene_init = InteractiveScene.__init__
    def scene_init(scene, *a, **kw):
        original_scene_init(scene, *a, **kw)
        prestartup(SimpleNamespace(scene=scene), None)
    InteractiveScene.__init__ = scene_init
    save(status='building_environment')
    env = ManagerBasedRLEnv(cfg=cfg)
    try:
        snapshot(env, 'constructed', physics=True)
        env.reset()
        snapshot(env, 'reset', physics=True)
        for i in range(args.steps):
            env.step(torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device))
            snapshot(env, f'step_{i+1}', physics=True)
        save(status='passed')
    finally:
        env.close()


from pmt_tasks.runtime_options import run_and_close
run_and_close(main, app)
