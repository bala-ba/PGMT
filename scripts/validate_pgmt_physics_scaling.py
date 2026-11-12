"""Check shared-terrain broadphase keeps environment isolation and tile contacts."""
import argparse
from copy import deepcopy
from dataclasses import fields
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.environ.get("PMT_PGMT_PHYSICS_WORKER") != "1":
    from pgmt_process_guard import run_guarded
    raise SystemExit(run_guarded([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], ROOT))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--log_dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
output = Path(args.log_dir).resolve()
output.mkdir(parents=True, exist_ok=True)
path = output / "training_status.json"
if path.exists():
    parser.error("Output already exists")
report = dict(status="launching", pid=os.getpid())
def save(**values):
    report.update(values)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2)+"\n")
    tmp.replace(path)
save()
from pmt_tasks.isaacsim6_legacy import install_tensor_import_bridge
install_tensor_import_bridge()
app = AppLauncher(args, multi_gpu=False, fast_shutdown=True).app


def main():
    from pmt_tasks.isaacsim6_legacy import install_rigid_scene_schema_bridge
    install_rigid_scene_schema_bridge()
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedRLEnv
    from pmt_tasks.builder import build_env_cfg
    from pmt_tasks.pgmt_physx import shared_terrain_scene
    from pmt_tasks.pgmt_runtime import get_pgmt_state, _tensor
    cfg = build_env_cfg("pgmt_stage2_terrain")
    cfg.pmt_motion_paths = [str(ROOT / "assets/motions/pgmt_clipped_50hz_tcrs_v1/walk1_subject1_clip0.npz")]
    cfg.__post_init__()
    cfg.scene.num_envs = 5
    cfg.seed = 1014
    cfg.sim.device = cfg.commands.motion.storage_device = args.device
    for name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
        setattr(cfg.events, name, None)
    old = cfg.scene.robot.spawn
    shared = {f.name: deepcopy(getattr(old, f.name)) for f in fields(sim_utils.UsdFileCfg)
              if f.name not in ("func", "usd_path") and hasattr(old, f.name)}
    cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
        usd_path=str(ROOT / "logs/pgmt_p1_acceptance/native_robot/main.usd"), **shared)
    with shared_terrain_scene(0) as setup:
        env = ManagerBasedRLEnv(cfg)
    save(status="checking", physics_setup=setup)
    try:
        state = get_pgmt_state(env)
        robot, terrain, command = state.robot, env.scene.terrain, state.command
        ids = torch.arange(5, device=env.device)
        actions = torch.zeros(5, env.action_manager.total_action_dim, device=env.device)
        def reset():
            terrain.terrain_levels.zero_()
            terrain.terrain_types[:] = ids
            terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
            env.termination_manager._terminated_buf.zero_()
            env.termination_manager._truncated_buf.zero_()
            env.termination_manager._term_dones.zero_()
            env.reset()
        with torch.inference_mode():
            reset()
            root = robot.data.root_link_state_w[:1].clone().repeat(2, 1)
            root[:, :3] = root.new_tensor([-117., -57., 5.])
            root[:, 7:] = 0.
            joints = robot.data.joint_pos[:1].clone().repeat(2, 1)
            robot.write_root_link_state_to_sim(root, env_ids=ids[:2])
            robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=ids[:2])
            env.sim.forward()
            max_contact, max_separation = 0., 0.
            for _ in range(8):
                env.step(actions)
                max_contact = max(max_contact, float(_tensor(env.scene["contact_forces"].data.net_forces_w)[:2].abs().max()))
                max_separation = max(max_separation, float((robot.data.body_pos_w[0]-robot.data.body_pos_w[1]).abs().max()))
            assert max_contact < 1e-4 and max_separation < 1e-4, (max_contact, max_separation)
            save(overlapping_airborne_robots=dict(steps=8, max_force_n=max_contact, max_pose_difference_m=max_separation))

            crossing = []
            for family in (1, 2, 3, 4):
                reset()
                # Move the flat-born robot to the independently initialized
                # valid pose on another tile, keeping its reference/birth fixed.
                translation, rotation = command.reference_translation[0].clone(), command.reference_rotation[0].clone()
                birth = terrain.terrain_types[0].clone()
                target = robot.data.root_link_state_w[family:family+1].clone()
                target[:, 7:] = 0.
                joints = robot.data.joint_pos[family:family+1].clone()
                robot.write_root_link_state_to_sim(target, env_ids=ids[:1])
                robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=ids[:1])
                env.sim.forward()
                peak = 0.
                for _ in range(6):
                    _, _, terminated, truncated, _ = env.step(actions)
                    assert not (terminated[0] | truncated[0])
                    sample = state.contact_state(for_reward=True)
                    peak = max(peak, float(sample["terrain_forces_w"][0, :, 2].max()))
                    torch.testing.assert_close(sample["terrain_forces_w"][0], sample["net_forces_w"][0], rtol=1e-4, atol=2e-3)
                    torch.testing.assert_close(command.reference_translation[0], translation, rtol=0, atol=0)
                    torch.testing.assert_close(command.reference_rotation[0], rotation, rtol=0, atol=0)
                    assert terrain.terrain_types[0] == birth == 0
                    located = terrain.tile_map.locate(robot.data.body_pos_w[0, state.anchor, :2])
                    assert located["tile_valid"] and int(located["family_index"]) == family
                assert peak > 10., (family, peak)
                crossing.append(dict(birth_family=0, actual_family=family, max_upward_force_n=peak))
            save(status="passed", control_steps=env.common_step_counter, crossing=crossing,
                 reference_and_birth_preserved=True, collision_filtering_preserved=True)
    except BaseException as error:
        save(status="failed", error=repr(error))
        raise
    finally:
        env.close()

from pmt_tasks.runtime_options import run_and_close
run_and_close(main, app)
