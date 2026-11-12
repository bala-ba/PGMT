"""Finite Isaac audit of terrain-filtered foot forces and contact lifecycle.

No PPO or checkpoint loading. --touchdown adds T1 RewardManager acceptance.
A non-terrain elevated platform provides a physical negative control.
"""
import argparse
from copy import deepcopy
from dataclasses import fields
from datetime import datetime
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
parser.add_argument("--steps", type=int, default=40)
parser.add_argument("--touchdown", action="store_true", help="Enable the implemented T1 term for isolated acceptance")
parser.add_argument("--contact_match", action="store_true", help="Enable T1 + T2 and check reference matching")
parser.add_argument("--slip", action="store_true", help="Enable T1 + T2 + T3 and check contact-gated foot velocity")
parser.add_argument("--stumble", action="store_true", help="Enable T1 through T4 and check excessive horizontal foot force")
parser.add_argument("--contact_switching", action="store_true", help="Enable T1 through T5 and replay fast contact switching")
parser.add_argument("--contact_force", action="store_true", help="Enable all six terrain terms and check squared foot forces")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
output = Path(args.log_dir).resolve()
output.mkdir(parents=True, exist_ok=True)
status_file = output / "training_status.json"
if status_file.exists():
    parser.error(f"Refusing to overwrite {output}")
report = dict(status="launching", pid=os.getpid(), arguments=vars(args), started_at=datetime.now().isoformat())


def save(**values):
    report.update(values)
    temp = status_file.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n")
    temp.replace(status_file)


save()
from pmt_tasks.isaacsim6_legacy import install_tensor_import_bridge
install_tensor_import_bridge()
app = AppLauncher(args, multi_gpu=False, fast_shutdown=True).app


def main():
    from pmt_tasks.isaacsim6_legacy import install_rigid_scene_schema_bridge
    install_rigid_scene_schema_bridge()
    import torch
    from pxr import UsdPhysics
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import TerminationTermCfg
    from isaaclab.utils.io import dump_yaml
    from pmt_tasks.env_cfgs.pgmt.stage2 import PGMTPerceptionEnvCfg
    from pmt_tasks.pgmt_contact_state import TERRAIN_FOOT_SENSORS
    from pmt_tasks.pgmt_runtime import get_pgmt_state

    cfg = PGMTPerceptionEnvCfg()
    cfg.pmt_motion_paths = [str(ROOT / "assets/motions/pgmt_clipped_50hz_tcrs_v1/walk1_subject1_clip0.npz")]
    cfg.__post_init__()
    use_switching = args.contact_switching or args.contact_force
    use_stumble = args.stumble or use_switching
    use_slip = args.slip or use_stumble
    use_match = args.contact_match or use_slip
    use_touchdown = args.touchdown or use_match
    if use_touchdown:
        from pmt_tasks.env_cfgs.pgmt.terrain_rewards import (
            PGMTTouchdownRewardsCfg, PGMTContactMatchRewardsCfg, PGMTSlipRewardsCfg, PGMTStumbleRewardsCfg,
            PGMTContactSwitchingRewardsCfg, PGMTContactForceRewardsCfg,
        )
        if args.contact_force:
            cfg.rewards = PGMTContactForceRewardsCfg()
        elif use_switching:
            cfg.rewards = PGMTContactSwitchingRewardsCfg()
        elif use_stumble:
            cfg.rewards = PGMTStumbleRewardsCfg()
        elif use_slip:
            cfg.rewards = PGMTSlipRewardsCfg()
        else:
            cfg.rewards = PGMTContactMatchRewardsCfg() if use_match else PGMTTouchdownRewardsCfg()
        cfg.pgmt_reward_groups = ("upper", "lower", "terrain", "auxiliary")
    cfg.scene.num_envs = 5
    cfg.seed = 1014
    cfg.sim.device = cfg.commands.motion.storage_device = args.device
    if use_touchdown:
        cfg.sim.enable_scene_query_support = True  # Native PhysX raycast audit in headless mode.
    for name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
        setattr(cfg.events, name, None)
    for name in vars(cfg.terminations):
        setattr(cfg.terminations, name, None)
    done = torch.zeros(5, dtype=torch.bool, device=args.device)
    cfg.terminations.audit_done = TerminationTermCfg(func=lambda env: done, time_out=True)
    old = cfg.scene.robot.spawn
    shared = {f.name: deepcopy(getattr(old, f.name)) for f in fields(sim_utils.UsdFileCfg)
              if f.name not in ("func", "usd_path") and hasattr(old, f.name)}
    cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
        usd_path=str(ROOT / "logs/pgmt_p1_acceptance/native_robot/main.usd"), **shared)
    platform_xy = (-117., -57.)
    cfg.scene.contact_audit_platform = AssetBaseCfg(
        prim_path="/World/PGMTContactAuditPlatform",
        spawn=sim_utils.CuboidCfg(size=(4., 4., .2), collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(*platform_xy, 4.9)),
    )
    wall_position = (-113., -57., 5.)
    if use_stumble:
        # Isolated elevated shin-height obstacle for actual foot-side impacts.
        # It is outside the terrain filter, and only exists in this audit.
        cfg.scene.stumble_audit_wall = AssetBaseCfg(
            prim_path="/World/PGMTStumbleAuditWall",
            spawn=sim_utils.CuboidCfg(size=(.2, 4., .3), collision_props=sim_utils.CollisionPropertiesCfg()),
            init_state=AssetBaseCfg.InitialStateCfg(pos=wall_position),
        )
    dump_yaml(str(output / "env.yaml"), cfg)
    save(status="building_environment")
    env = ManagerBasedRLEnv(cfg)
    try:
        terrain, robot = env.scene.terrain, env.scene["robot"]
        command = env.command_manager.get_term("motion")
        terrain.terrain_levels[:] = 0
        terrain.terrain_types[:] = torch.arange(5, device=env.device)
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        env.reset()
        state = get_pgmt_state(env)
        assert not state.contact_state()["support_valid"].any()
        sensors = {}
        for name in TERRAIN_FOOT_SENSORS:
            sensor = env.scene.sensors[name]
            sensors[name] = dict(bodies=sensor.body_names, filter_count=len(sensor.cfg.filter_prim_paths_expr),
                                  first_filter=sensor.cfg.filter_prim_paths_expr[0],
                                  matrix_shape=list(sensor.data.force_matrix_w.shape))
        collider_paths = [str(p.GetPath()) for p in env.scene.stage.Traverse()
                          if p.HasAPI(UsdPhysics.CollisionAPI) and "/terrain/Terrain/" in str(p.GetPath())]
        assert len(collider_paths) == 800
        save(status="checking", sensors=sensors, terrain_collider_count=len(collider_paths))
        audit = None
        if use_touchdown:
            from pgmt_touchdown_checks import TouchdownAudit
            audit = TouchdownAudit(env, state)
            audit.geometry()
        match_audit = None
        if use_match:
            from pgmt_contact_match_checks import ContactMatchAudit
            match_audit = ContactMatchAudit(env, state)
        slip_audit = None
        if use_slip:
            from pgmt_slip_checks import SlipAudit
            slip_audit = SlipAudit(env, state)
        stumble_audit = None
        if use_stumble:
            from pgmt_stumble_checks import StumbleAudit
            stumble_audit = StumbleAudit(env, state)
        switching_audit = None
        if use_switching:
            from pgmt_contact_switching_checks import ContactSwitchingAudit
            switching_audit = ContactSwitchingAudit(env, state)
        force_audit = None
        if args.contact_force:
            from pgmt_contact_force_checks import ContactForceAudit
            force_audit = ContactForceAudit(env, state)
            assert len(env.reward_manager.active_terms) == 27
        actions = torch.zeros(5, env.action_manager.total_action_dim, device=env.device)
        maxima = torch.zeros(5, 2, device=env.device)
        events = dict(touchdown=0, contact_started=0, contact_ended=0, fast_contact_switch=0)
        for _ in range(args.steps):
            if match_audit:
                match_audit.before_step()
            _, total_reward, *_ = env.step(actions)
            if match_audit:
                match_audit.settled_step(total_reward)
            if slip_audit:
                slip_audit.settled_step(total_reward)
            if stumble_audit:
                stumble_audit.settled_step(total_reward)
            if switching_audit:
                switching_audit.settled_step(total_reward)
            if force_audit:
                force_audit.settled_step(total_reward)
            if audit:
                audit.settled_step(total_reward)
            sample = state.contact_state(for_reward=True)
            assert sample["contact_valid"].all() and sample["support_valid"].all()
            torch.testing.assert_close(sample["support"], sample["terrain_forces_w"][..., 2] > 10.)
            maxima = torch.maximum(maxima, sample["upward_force"])
            for name in events:
                events[name] += int(sample[name].sum())
            before = state.contact_state()
            state.rewards(); state.rewards()
            for key, value in before.items():
                torch.testing.assert_close(state.contact_state()[key], value, equal_nan=True)
        assert maxima[0].max() > 10., maxima

        # Native terminal step: old reference/events frozen, reset rows invalid.
        frame = command.frame_ids.clone()
        done[0] = True
        if match_audit:
            match_audit.before_step()
        _, total_reward, *_ = env.step(actions)
        if force_audit:
            force_audit.settled_step(total_reward)
            assert state.contact_force_state()["rate"][0] == 0. and not state.contact_force_state()["valid"][0].any()
            assert state.contact_force_state()["forces_w"][0].isnan().all()
            torch.testing.assert_close(state.contact_force_state(for_reward=True)["reference_frame_ids"], frame)
        if switching_audit:
            switching_audit.settled_step(total_reward)
            assert state.contact_switching_state()["rate"][0] == 0.
            assert not state.contact_switching_state()["valid"][0].any()
            torch.testing.assert_close(state.contact_switching_state(for_reward=True)["reference_frame_ids"], frame)
        if stumble_audit:
            stumble_audit.settled_step(total_reward)
            assert state.stumble_state()["rate"][0] == 0. and not state.stumble_state()["valid"][0].any()
            assert state.stumble_state()["forces_w"][0].isnan().all()
            torch.testing.assert_close(state.stumble_state(for_reward=True)["reference_frame_ids"], frame)
        if slip_audit:
            slip_audit.settled_step(total_reward)
            assert state.slip_state()["rate"][0] == 0. and not state.slip_state()["valid"][0].any()
            assert state.slip_state()["velocity_w"][0].isnan().all()
            torch.testing.assert_close(state.slip_state(for_reward=True)["reference_frame_ids"], frame)
        if match_audit:
            match_audit.settled_step(total_reward)
            assert not state.reference_contact_match_state()["compared_valid"][0].any()
            assert state.reference_contact_match_state()["rate"][0] == 0.
            torch.testing.assert_close(state.reference_contact_match_state(for_reward=True)["reference_frame_ids"], frame)
        if audit:
            audit.settled_step(total_reward)
            assert state.touchdown_state()["rate"][0] == 0.
            assert not state.touchdown_state()["valid"][0].any()
        done.zero_()
        terminal = state.contact_state(for_reward=True)
        live = state.contact_state()
        torch.testing.assert_close(terminal["reference_frame_ids"], frame)
        assert live["sample_step"][0] == -1 and not live["support_valid"][0].any()
        for key in ("support", "episode_step", "reference_frame_ids", "touchdown"):
            torch.testing.assert_close(live[key][1:], terminal[key][1:])
        state.rewards()
        torch.testing.assert_close(state.contact_state(for_reward=True)["reference_frame_ids"], frame)
        if match_audit:
            match_audit.before_step()
        _, total_reward, *_ = env.step(actions)
        if match_audit:
            match_audit.settled_step(total_reward)
        if audit:
            audit.settled_step(total_reward)
            # First post-reset physical sample may earn continuous support
            # quality. It must still not fabricate a touchdown event.
            assert not state.touchdown_state()["event"][0].any()
        first = state.contact_state()
        if force_audit:
            force_audit.settled_step(total_reward)
        if switching_audit:
            switching_audit.settled_step(total_reward)
            assert state.contact_switching_state()["rate"][0] == 0.
            assert not state.contact_switching_state()["valid"][0].any()
        if stumble_audit:
            stumble_audit.settled_step(total_reward)
        if slip_audit:
            slip_audit.settled_step(total_reward)
        assert first["episode_step"][0] == 1 and not first["support_history_valid"][0].any()
        assert not first["touchdown"][0].any()

        # Put env 0 on an elevated non-terrain collider; net force must respond,
        # but the dedicated terrain filter must report zero, not fake support.
        ids = torch.tensor([0], device=env.device)
        command._resample_command(ids)
        root = robot.data.root_link_state_w[ids].clone()
        feet = robot.data.body_pos_w[0, state.foot_body_indices]
        shift = root.new_tensor([platform_xy[0], platform_xy[1], 0.]) - root[0, :3]
        shift[2] = 5.01 - (feet[:, 2].min() - .035)
        root[:, :3] += shift
        root[:, 7:] = 0
        robot.write_root_link_state_to_sim(root, env_ids=ids)
        robot.write_joint_state_to_sim(robot.data.joint_pos[ids], torch.zeros_like(robot.data.joint_vel[ids]), env_ids=ids)
        command.reference_translation[ids] += shift
        command._world_tracked = None
        env.sim.forward()
        state.reset(ids)
        max_external, max_filtered = 0., 0.
        for _ in range(25):
            if match_audit:
                match_audit.before_step()
            _, total_reward, *_ = env.step(actions)
            if match_audit:
                match_audit.settled_step(total_reward)
            if slip_audit:
                slip_audit.settled_step(total_reward)
            if stumble_audit:
                stumble_audit.settled_step(total_reward)
            if switching_audit:
                switching_audit.settled_step(total_reward)
            if force_audit:
                force_audit.settled_step(total_reward)
            if audit:
                audit.settled_step(total_reward)
                assert state.touchdown_state(for_reward=True)["rate"][0] == 0.
            sample = state.contact_state(for_reward=True, env_ids=ids)
            max_external = max(max_external, float(sample["net_forces_w"].norm(dim=-1).max()))
            max_filtered = max(max_filtered, float(sample["terrain_forces_w"].abs().max()))
            assert not sample["support"].any() and not sample["touchdown"].any()
        assert max_external > 10., max_external
        assert max_filtered < 1e-4, max_filtered
        if match_audit:
            assert match_audit.report["contact_support_disagreements"] > 0
            match_audit.reference_mask_scenarios(actions, audit, slip_audit, stumble_audit, switching_audit, force_audit)
        if stumble_audit:
            stumble_audit.side_collision_scenarios(actions, audit, match_audit, slip_audit, wall_position,
                                                   switching_audit, force_audit)
        if match_audit:
            save(contact_match=match_audit.report)
        if slip_audit:
            assert slip_audit.report["moving_contact_feet"] > 0
            assert slip_audit.report["zero_cost_unloaded_feet"] > 0
            assert slip_audit.report["nonterrain_contact_samples"] > 0
            assert slip_audit.report["settled_steps"] == env.common_step_counter
            save(slip=slip_audit.report)
        if stumble_audit:
            save(stumble=stumble_audit.report)
            assert stumble_audit.report["triggered_feet"] > 0, stumble_audit.report
            assert stumble_audit.report["zero_cost_feet"] > 0
            assert stumble_audit.report["settled_steps"] == env.common_step_counter
        if audit:
            assert audit.report["rewarded_feet"] > 0
            assert audit.report["continuous_without_touchdown"] > 0
            save(touchdown=audit.report)
        if switching_audit:
            save(contact_switching=switching_audit.report)
            assert switching_audit.report["started"] > 0 and switching_audit.report["ended"] > 0
            assert switching_audit.report["grace_excluded"] > 0
            assert switching_audit.report["slow_excluded"] > 0
            assert switching_audit.report["holding_zero_cost_feet"] > 0
            assert switching_audit.report["settled_steps"] == env.common_step_counter
        if force_audit:
            save(contact_force=force_audit.report)
            assert force_audit.report["positive_force_feet"] > 0 and force_audit.report["zero_force_feet"] > 0
            assert force_audit.report["nonterrain_force_feet"] > 0
            assert force_audit.report["above_350n_feet"] > 0
            assert force_audit.report["settled_steps"] == env.common_step_counter
        save(status="passed", control_steps=env.common_step_counter, max_upward_force_per_family=maxima.cpu().tolist(),
             events=events, terminal_reset_checked=True, duplicate_reads_checked=True,
             nonterrain_max_net_force_n=max_external, nonterrain_max_filtered_force_n=max_filtered,
             active_rewards=env.reward_manager.active_terms)
        print("PGMT_CONTACT_STATE_PASSED", json.dumps(report), flush=True)
    except BaseException as error:
        save(status="failed", error=repr(error))
        raise
    finally:
        env.close()


from pmt_tasks.runtime_options import run_and_close
run_and_close(main, app)
