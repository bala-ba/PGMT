"""Finite native collision -> sensor -> P3 auxiliary reward audit (no PPO).

Only initial poses/velocities and a small physical audit obstacle are controlled.
Contact forces and reward inputs are never synthesized or overwritten.
"""
import argparse
from copy import deepcopy
from dataclasses import fields
from datetime import datetime
import hashlib
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
status_file = output / "training_status.json"
if status_file.exists():
    parser.error("Refusing to overwrite an existing audit")
report = dict(status="launching", pid=os.getpid(), started_at=datetime.now().isoformat())


def save(**values):
    report.update(values)
    temporary = status_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(status_file)


save()
from pmt_tasks.isaacsim6_legacy import install_tensor_import_bridge
install_tensor_import_bridge()
app = AppLauncher(args, multi_gpu=False, fast_shutdown=True).app


def main():
    from pmt_tasks.isaacsim6_legacy import install_rigid_scene_schema_bridge
    install_rigid_scene_schema_bridge()
    import numpy as np
    import torch
    from pxr import Usd, UsdGeom, UsdPhysics
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import TerminationTermCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.utils.io import dump_yaml
    from pmt_tasks.builder import build_env_cfg
    from pmt_tasks.isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from pmt_tasks.pgmt_math import conjugate, inverse_rotate, multiply, rotate
    from pmt_tasks.pgmt_runtime import _tensor, get_pgmt_state

    cfg = build_env_cfg("pgmt_stage2_terrain")
    cfg.pmt_motion_paths = [str(ROOT / "assets/motions/pgmt_clipped_50hz_tcrs_v1/walk1_subject1_clip0.npz")]
    cfg.__post_init__()
    cfg.scene.num_envs, cfg.seed = 5, 1014
    cfg.sim.device = cfg.commands.motion.storage_device = args.device
    for name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
        setattr(cfg.events, name, None)
    terminal_probe = dict(enabled=False)

    def terminate_on_real_torso_contact(env):
        force = _tensor(env.scene["contact_forces"].data.net_forces_w)
        index = env.scene["contact_forces"].body_names.index("torso_link")
        result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if terminal_probe["enabled"]:
            result[0] = force[0, index].norm() > 10.
        return result

    # Retain all production terminations. This additional audit-only trigger
    # makes the actual impact step terminal to test the pre-reset reward cache.
    cfg.terminations.audit_body_contact = TerminationTermCfg(func=terminate_on_real_torso_contact)
    old = cfg.scene.robot.spawn
    shared = {f.name: deepcopy(getattr(old, f.name)) for f in fields(sim_utils.UsdFileCfg)
              if f.name not in ("func", "usd_path") and hasattr(old, f.name)}
    asset = ROOT / "logs/pgmt_p1_acceptance/native_robot/main.usd"
    cfg.scene.robot.spawn = sim_utils.UsdFileCfg(usd_path=str(asset), **shared)
    obstacle = (-117., -57., 5.)
    cfg.scene.body_audit_obstacle = AssetBaseCfg(
        prim_path="/World/PGMTBodyAuditObstacle",
        spawn=sim_utils.CuboidCfg(size=(.2, .06, .04), collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=obstacle))
    cfg.scene.body_audit_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        filter_prim_paths_expr=["/World/PGMTBodyAuditObstacle/geometry/mesh"], track_contact_points=True,
        max_contact_data_count_per_prim=32, history_length=0, debug_vis=False)
    dump_yaml(str(output / "env.yaml"), cfg)
    save(status="building_environment", asset=str(asset), asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
    env = ManagerBasedRLEnv(cfg)
    records, contacts, scenarios, physics_samples = [], [], [], []
    arrays = {name: [] for name in ("forces_w", "rates", "contributions", "reward", "groups", "done")}
    try:
        wrapped = RslRlVecEnvWrapper(env)
        wrapped.get_observations()
        state, robot = get_pgmt_state(env), env.scene["robot"]
        command, sensor = state.command, env.scene["contact_forces"]
        names = list(robot.body_names)
        torso = names.index("torso_link")
        foot_names = ("left_ankle_roll_link", "right_ankle_roll_link")
        feet = [names.index(name) for name in foot_names]
        undesired = [i for i, name in enumerate(names) if name not in foot_names]
        sensor_order = [sensor.body_names.index(name) for name in names]
        assert len(names) == 30 and "head_link" not in names
        assert state.anchor == torso and state.contact_order.tolist() == sensor_order
        assert state.groups["undesired"].tolist() == undesired
        reward_names = env.reward_manager.active_terms
        assert len(reward_names) == 27 and tuple(cfg.pgmt_reward_groups) == ("upper", "lower", "terrain", "auxiliary")
        safety_names = ("auxiliary_undesired_contacts", "auxiliary_anchor_contact_force")
        weights = np.array([env.reward_manager.get_term_cfg(name).weight for name in reward_names])
        assert [weights[reward_names.index(name)] for name in safety_names] == [-.1, -1e-5]

        # Independent authored collider geometry, including instance proxies.
        stage = Usd.Stage.Open(str(asset))
        cache = UsdGeom.XformCache()
        colliders = {}
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.CollisionAPI) and "/torso_link/" in str(prim.GetPath()):
                transform = cache.ComputeRelativeTransform(prim, stage.GetPrimAtPath("/g1/torso_link"))[0]
                shape = UsdGeom.Capsule(prim)
                colliders[prim.GetParent().GetName()] = dict(
                    path=str(prim.GetPath()), center=list(transform.ExtractTranslation()),
                    axis=list(transform.TransformDir((0., 0., 1.))),
                    radius=shape.GetRadiusAttr().Get(), height=shape.GetHeightAttr().Get())
        assert len(colliders) == 4

        current = dict(tag="initial")

        audit_sensor = env.scene["body_audit_sensor"]
        original_sim_step = env.sim.step

        def step_with_physics_sample(*args, **kwargs):
            result = original_sim_step(*args, **kwargs)
            # Read-only tensor views at every 200 Hz substep. Production
            # rewards still sample the last substep at the normal 50 Hz rate.
            force = _tensor(sensor.contact_physx_view.get_net_contact_forces(dt=env.physics_dt))
            force = force.reshape(env.num_envs, len(names), 3)[0, sensor_order[torso]].clone()
            filtered = _tensor(audit_sensor.contact_physx_view.get_contact_force_matrix(dt=env.physics_dt))
            filtered = filtered.reshape(env.num_envs, 1, 1, 3)[0, 0, 0].clone()
            if current["tag"].startswith(("head_", "chest_", "terminal_")):
                torch.testing.assert_close(force, filtered, rtol=3e-5, atol=2e-3)
            physics_samples.append(dict(tag=current["tag"], physics_step=env._sim_step_counter,
                torso_force_w=force.cpu().tolist(), torso_force_norm_n=float(force.norm()),
                obstacle_force_w=filtered.cpu().tolist()))
            return result

        env.sim.step = step_with_physics_sample

        def physical_report(raw):
            # GPU PhysX exposes per-point force/normal/position for a known
            # torso-obstacle pair. Map each real contact point to the unique
            # authored capsule using its local-space capsule surface distance.
            # This avoids CPU event reports, unavailable in this GPU runtime.
            filtered = _tensor(audit_sensor.data.force_matrix_w)[0].sum((0, 1))
            force, point, normal, separation, count, start = [
                _tensor(v) for v in audit_sensor.contact_physx_view.get_contact_data(dt=env.physics_dt)]
            begin, size = int(start[0, 0]), int(count[0, 0])
            assert size < audit_sensor.cfg.max_contact_data_count_per_prim
            reconstructed = np.zeros(3)
            points = []
            for index in range(begin, begin + size):
                position = point[index]
                local = inverse_rotate(robot.data.body_quat_w[0, torso], position - robot.data.body_pos_w[0, torso])
                local_np = local.cpu().numpy().astype(np.float64)
                distance = {}
                for name, geometry in colliders.items():
                    delta = local_np - np.array(geometry["center"])
                    axis = np.array(geometry["axis"])
                    nearest = np.clip(np.dot(delta, axis), -geometry["height"]/2, geometry["height"]/2)
                    distance[name] = abs(float(np.linalg.norm(delta - nearest*axis) - geometry["radius"]))
                closest = min(distance, key=distance.get)
                magnitude = float(force[index].reshape(()))
                vector = magnitude * normal[index].cpu().numpy().astype(np.float64)
                reconstructed += vector
                points.append(dict(position=position.cpu().tolist(), local_position=local_np.tolist(),
                    normal=normal[index].cpu().tolist(), normal_force_n=magnitude,
                    separation=float(separation[index].reshape(())), capsule_surface_distance_m=distance,
                    closest_capsule=closest, collider=colliders[closest]["path"]))
                if magnitude > 1.:
                    assert distance[closest] < .03, points[-1]
            np.testing.assert_allclose(reconstructed, filtered.cpu(), rtol=3e-5, atol=2e-3)
            # All elevated probes isolate torso-obstacle contact: the pair
            # force must equal the existing whole-object training sensor.
            if current["tag"].startswith(("head_", "chest_", "terminal_")):
                np.testing.assert_allclose(filtered.cpu(), raw[0, torso], rtol=3e-5, atol=2e-3)
            contacts.append(dict(tag=current["tag"], physics_step=env._sim_step_counter,
                sensor_body="torso_link", other_collider="/World/PGMTBodyAuditObstacle/geometry/mesh",
                points=points, filtered_force_w=filtered.cpu().tolist(),
                reconstructed_force_w=reconstructed.tolist()))
        original_compute = env.reward_manager.compute
        max_errors = dict(rate=0., contribution=0., total=0., group_sum=0.)

        def checked_compute(dt):
            # Read the actual sensor before native reset clears its rows.
            forces = _tensor(sensor.data.net_forces_w).clone()[:, sensor_order]
            raw = forces.cpu().numpy().astype(np.float64)
            assert np.isfinite(raw).all()
            physical_report(raw)
            expected = {
                safety_names[0]: (np.linalg.norm(raw[:, undesired], axis=-1) > 10.).sum(-1),
                safety_names[1]: np.square(raw[:, torso]).sum(-1),
            }
            total = original_compute(dt)
            rates = state.rewards()
            contributions = (env.reward_manager._step_reward * dt).clone()
            for name in safety_names:
                index = reward_names.index(name)
                actual = rates[name].cpu().numpy()
                np.testing.assert_allclose(actual, expected[name], rtol=2e-6, atol=1e-5)
                target = expected[name] * weights[index] * dt
                np.testing.assert_allclose(contributions[:, index].cpu(), target, rtol=3e-6, atol=1e-7)
                assert contributions[:, index].le(0).all()
                max_errors["rate"] = max(max_errors["rate"], float(np.max(np.abs(actual - expected[name]))))
                max_errors["contribution"] = max(max_errors["contribution"],
                    float(np.max(np.abs(contributions[:, index].cpu().numpy() - target))))
            all_rates = torch.stack([rates[name] for name in reward_names], -1)
            expected_all = all_rates.cpu().numpy().astype(np.float64) * weights * dt
            np.testing.assert_allclose(contributions.cpu(), expected_all, rtol=3e-6, atol=2e-7)
            np.testing.assert_allclose(total.cpu(), expected_all.sum(-1), rtol=3e-6, atol=2e-6)
            max_errors["total"] = max(max_errors["total"], float(np.max(np.abs(total.cpu().numpy()-expected_all.sum(-1)))))
            previous = [state.previous_target.clone(), state.previous_velocity.clone(), state.previous_ref_velocity.clone()]
            for _ in range(3):
                for name, value in rates.items():
                    torch.testing.assert_close(state.rewards()[name], value, rtol=0, atol=0)
            for before, after in zip(previous, (state.previous_target, state.previous_velocity, state.previous_ref_velocity)):
                torch.testing.assert_close(before, after, rtol=0, atol=0)
            records.append(dict(tag=current["tag"], control_step=env.common_step_counter,
                physics_step=env._sim_step_counter, torso_force_w=raw[0, torso].tolist(),
                torso_force_norm_n=float(np.linalg.norm(raw[0, torso])),
                contacting_undesired_bodies=[names[i] for i in undesired if np.linalg.norm(raw[0, i]) > 10.],
                rates={name: float(rates[name][0]) for name in safety_names},
                contributions={name: float(contributions[0, reward_names.index(name)]) for name in safety_names}))
            arrays["forces_w"].append(raw)
            arrays["rates"].append(all_rates.cpu().numpy().copy())
            arrays["contributions"].append(contributions.cpu().numpy().copy())
            arrays["reward"].append(total.cpu().numpy().copy())
            return total

        env.reward_manager.compute = checked_compute
        ids = torch.tensor([0], device=env.device)
        actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)

        def reset(tag):
            current["tag"] = tag
            terminal_probe["enabled"] = False
            terrain = env.scene.terrain
            terrain.terrain_levels[:] = 0
            terrain.terrain_types[:] = 0
            terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
            env.termination_manager._terminated_buf.zero_()
            env.termination_manager._truncated_buf.zero_()
            env.termination_manager._term_dones.zero_()
            env.reset()

        def step():
            _, total, done, extras = wrapped.step(actions)
            groups = extras["reward_groups"]
            contributions = arrays["contributions"][-1]
            expected_groups = np.stack([contributions[:, [i for i, name in enumerate(reward_names)
                                       if name.startswith(group + "_")]].sum(-1)
                                       for group in cfg.pgmt_reward_groups], -1)
            np.testing.assert_allclose(groups.cpu(), expected_groups, rtol=3e-6, atol=2e-6)
            torch.testing.assert_close(groups.sum(-1), total, rtol=3e-6, atol=2e-6)
            max_errors["group_sum"] = max(max_errors["group_sum"], float((groups.sum(-1)-total).abs().max()))
            arrays["groups"].append(groups.cpu().numpy().copy())
            arrays["done"].append(done.cpu().numpy().copy())
            # Check the settled rates again AFTER any automatic reset.
            for name, value in records[-1]["rates"].items():
                assert float(state.rewards()[name][0]) == value
            return done

        def place(mesh=None, speed=0.):
            root = robot.data.root_link_state_w[ids].clone()
            root[:, 3:7] = multiply(conjugate(robot.data.body_quat_w[ids, torso]), root[:, 3:7])
            root[:, 7:] = 0
            robot.write_root_link_state_to_sim(root, env_ids=ids)
            env.sim.forward()
            root = robot.data.root_link_state_w[ids].clone()
            if mesh is None:
                shift = root.new_tensor([-121., -57., 5.]) - root[0, :3]
                geometry = None
            else:
                geometry = colliders[mesh]
                centre = robot.data.body_pos_w[0, torso] + rotate(
                    robot.data.body_quat_w[0, torso], root.new_tensor(geometry["center"]))
                target = root.new_tensor(obstacle)
                target[0] = obstacle[0] - .1 - .015 - geometry["radius"]
                # Chest probe stays above the adjacent lower capsule.
                if mesh == "mesh_0":
                    target[2] -= .025
                shift = target - centre
            root[:, :3] += shift
            root[:, 7:] = 0
            root[:, 7] = speed
            robot.write_root_link_state_to_sim(root, env_ids=ids)
            robot.write_joint_state_to_sim(robot.data.joint_pos[ids], torch.zeros_like(robot.data.joint_vel[ids]), env_ids=ids)
            env.sim.forward()
            state.reset(ids)
            # A conservative complete X extent lies 15 mm before the obstacle.
            if geometry:
                assert abs(float((robot.data.body_pos_w[0, torso] + rotate(robot.data.body_quat_w[0, torso],
                    root.new_tensor(geometry["center"])))[0]) + geometry["radius"] - (obstacle[0]-.1-.015)) < 1e-4

        with torch.inference_mode():
            reset("air_control")
            place()
            for _ in range(4):
                step()
                assert np.max(np.abs(arrays["forces_w"][-1][0])) < 1e-5
                assert all(value == 0. for value in records[-1]["rates"].values())

            reset("feet_only_control")
            foot_only_count = 0
            for _ in range(6):
                step()
                force = arrays["forces_w"][-1][0]
                if np.max(np.linalg.norm(force[feet], axis=-1)) > 10. and np.max(np.abs(force[undesired])) < 1e-5:
                    foot_only_count += 1
                    assert all(value == 0. for value in records[-1]["rates"].values())
            assert foot_only_count > 0

            for tag, mesh in (("head", "mesh_3"), ("chest", "mesh_0")):
                for speed in (1., 2.):
                    scenario = f"{tag}_{speed:g}mps"
                    reset(scenario)
                    place(mesh, speed)
                    start = len(records)
                    for _ in range(8):
                        assert not step()[0]
                    selected = records[start:]
                    positive = [r for r in selected if r["torso_force_norm_n"] > 10.]
                    physical = [c for c in contacts if c["tag"] == scenario and any(
                        p["closest_capsule"] == mesh and p["normal_force_n"] > 1. for p in c["points"])]
                    assert positive and physical, (scenario, positive, physical)
                    for row in positive:
                        matching = [c for c in physical if c["physics_step"] == row["physics_step"]]
                        assert len(matching) == 1
                        assert all(p["closest_capsule"] == mesh for p in matching[0]["points"]
                                   if p["normal_force_n"] > 1.)
                        assert row["contacting_undesired_bodies"] == ["torso_link"], row
                        assert row["rates"][safety_names[0]] == 1.
                        np.testing.assert_allclose(row["contributions"][safety_names[0]], -.002, atol=1e-9)
                    scenarios.append(dict(name=scenario, target_collider=colliders[mesh]["path"],
                        positive_steps=len(positive), physical_reports=len(physical),
                        physics_peak_force_n=max(r["torso_force_norm_n"] for r in physics_samples if r["tag"] == scenario),
                        peak=max(positive, key=lambda r:r["torso_force_norm_n"])))
                    save(status="checking", scenarios=scenarios)

            reset("terminal_head_impact")
            place("mesh_3", 2.)
            terminal_probe["enabled"] = True
            terminal = False
            for _ in range(8):
                if step()[0]:
                    terminal = True
                    assert records[-1]["torso_force_norm_n"] > 10.
                    assert env.termination_manager.get_term("audit_body_contact")[0]
                    assert not state.contact_state()["contact_valid"][0].any()
                    # Raw PhysX data describe the last simulated substep even
                    # after teleport/reset. New-episode validity, not an
                    # immediate zero-force guarantee, is the runtime contract.
                    save(post_reset_raw_torso_force_n=float(_tensor(sensor.data.net_forces_w)[0, sensor_order[torso]].norm()))
                    break
            assert terminal
            terminal_probe["enabled"] = False
            current["tag"] = "post_terminal_air"
            place()
            step()
            assert all(value == 0. for value in records[-1]["rates"].values())
            assert np.max(np.abs(arrays["forces_w"][-1][0])) < 1e-5

        save(status="passed", control_steps=env.common_step_counter, environments=env.num_envs,
            physics_dt=env.physics_dt, control_dt=env.step_dt, decimation=cfg.decimation,
            body_names=names, sensor_body_names=list(sensor.body_names), reward_names=reward_names,
            reward_weights=weights.tolist(), torso_colliders=colliders, scenarios=scenarios,
            foot_only_steps=foot_only_count, max_errors=max_errors, contact_reports=len(contacts),
            terminal_old_reward_preserved=True, post_reset_stale_force_absent=True, duplicate_reads_checked=True,
            contact_evidence="GPU contact points/normals/forces; unique capsule surface geometry; obstacle filter",
            scope="Real collision and reward plumbing; no policy learning or necessary crawl contact assessment")
        print("PGMT_BODY_CONTACTS_PASSED", json.dumps(report), flush=True)
    except BaseException as error:
        save(status="failed", error=repr(error), control_steps=env.common_step_counter, scenarios=scenarios)
        raise
    finally:
        (output / "reward_records.json").write_text(json.dumps(records, indent=2)+"\n")
        (output / "physical_contacts.json").write_text(json.dumps(contacts, indent=2)+"\n")
        (output / "physics_samples.json").write_text(json.dumps(physics_samples, indent=2)+"\n")
        if records:
            np.savez_compressed(output / "samples.npz", **{name:np.asarray(value) for name,value in arrays.items()})
        env.close()


from pmt_tasks.runtime_options import run_and_close
run_and_close(main, app)
