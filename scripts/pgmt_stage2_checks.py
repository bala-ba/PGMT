"""P2 real-simulator checks; never resample perception during the PPO diagnostics."""
from copy import deepcopy
import json
import numpy as np
import torch


def equal_state(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), (actual.keys(), expected.keys())
        for k in expected:
            equal_state(actual[k], expected[k])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            equal_state(a, b)
    else:
        assert actual == expected, (actual, expected)


def save_debug(policy, obs, path):
    with torch.no_grad():
        debug = policy.get_perception_debug(obs)
    values = {name: value.detach().cpu().numpy() for name, value in debug.glimpse._asdict().items()}
    values.update(terrain_map=obs['terrain_map'].cpu().numpy(), terrain_valid=obs['terrain_valid'].cpu().numpy(),
                  attention=debug.attention.weights.cpu().numpy(), key_valid=debug.attention.key_valid.cpu().numpy(),
                  terrain_attention_mass=debug.terrain_attention_mass.cpu().numpy())
    np.savez(path, **values)


def center_metrics(centers):
    pairs = torch.triu_indices(4, 4, offset=1, device=centers.device)
    difference = (centers[:, pairs[0]] - centers[:, pairs[1]]).abs()
    return dict(center_std_m=centers.std((0,1)).tolist(),
                center_boundary_fraction=float((centers.abs() > .76).float().mean()),
                center_overlap_fraction=float((difference < .4).all(-1).float().mean()),
                center_overlap_area_fraction=float((.4-difference).clamp_min(0).prod(-1).mean()/.16))


class RolloutAudit:
    def __init__(self, env, viewer=None):
        from pmt_tasks.pgmt_runtime import get_pgmt_state
        raw = env.unwrapped
        state, cmd = get_pgmt_state(raw), raw.command_manager.get_term('motion')
        course = raw._pgmt_terrain_curriculum
        self.checks = dict(reset_poses=0, sampler_outcomes=0, physics_control_steps=0, done_events=0,
                           reference_drift_checks=0, contact_freshness_checks=0)
        last_outcome = torch.full((raw.num_envs,), -1, device=raw.device, dtype=torch.long)
        old_reset, old_outcome, old_step = cmd._reset_robot_state, cmd._record_outcomes, env.step
        def reset(ids):
            old_reset(ids)
            if cmd.cfg.fixed_start_frame >= 0:
                assert cmd.frame_ids[ids].eq(cmd.cfg.fixed_start_frame).all()
            else:
                assert (cmd.frame_ids[ids] >= 0).all()
                assert (cmd.frame_ids[ids] < cmd.data_store.motion_lengths[cmd.motion_ids[ids]] - 1).all()
            if cmd.spawn_patch_ids is not None:
                assert torch.isin(cmd.placement_patch_ids[ids], cmd.spawn_patch_ids).all()
            if cmd.cfg.placement_xy_mode == "center" and cmd.placement_mode == "patch":
                terrain = raw.scene.terrain
                centers = terrain.patch_bounds[terrain.terrain_levels[ids], terrain.terrain_types[ids],
                                               cmd.placement_patch_ids[ids]].mean(dim=1)
                torch.testing.assert_close(cmd.placement_xy[ids], centers, atol=0, rtol=0)
            assert cmd.placement_min_gap[ids].ge(cmd.cfg.spawn_clearance-1e-5).all()
            assert cmd.compatible_motions[raw.scene.terrain.terrain_types[ids], cmd.motion_ids[ids]].all()
            ref = cmd.transform_reference(cmd._source_reference(cmd.motion_ids[ids], cmd.frame_ids[ids]), ids)
            torch.testing.assert_close(raw.scene['robot'].data.body_pos_w[ids], ref['body_pos'], atol=5e-5, rtol=0)
            self.checks['reset_poses'] += len(ids)
        def outcome(ids, terminated):
            eligible = course.sampler_pending[ids] & (course.settled_step[ids] == raw.common_step_counter)
            assert last_outcome[ids[eligible]].ne(raw.common_step_counter).all()
            old_outcome(ids, terminated)
            last_outcome[ids[eligible]] = raw.common_step_counter
            self.checks['sampler_outcomes'] += int(eligible.sum())
        def step(actions):
            rotation, translation = cmd.reference_rotation.clone(), cmd.reference_translation.clone()
            obs, reward, done, extras = old_step(actions)
            survivors = ~done.bool()
            torch.testing.assert_close(cmd.reference_rotation[survivors], rotation[survivors], rtol=0, atol=0)
            torch.testing.assert_close(cmd.reference_translation[survivors], translation[survivors], rtol=0, atol=0)
            contacts = state.reference_contacts()
            assert torch.equal(contacts['motion_ids'], cmd.motion_ids)
            assert torch.equal(contacts['frame_ids'], cmd.frame_ids)
            assert all(torch.isfinite(v).all() for v in obs.values()) and torch.isfinite(reward).all()
            torch.testing.assert_close(extras['reward_groups'].sum(-1), reward, rtol=2e-5, atol=2e-6)
            assert obs['terrain_map'][obs['terrain_valid'] == 0].eq(-3).all()
            self.checks['physics_control_steps'] += 1
            self.checks['done_events'] += int(done.sum())
            self.checks['reference_drift_checks'] += int(survivors.sum())
            self.checks['contact_freshness_checks'] += raw.num_envs
            if viewer and self.checks['physics_control_steps'] % 5 == 0:
                viewer.update(obs)
            return obs, reward, done, extras
        cmd._reset_robot_state, cmd._record_outcomes, env.step = reset, outcome, step


def audit_geometry(env, policy, output, progress):
    """Controlled probes, then restore fresh persistent counts/assignments before PPO."""
    from pmt_tasks.pgmt_runtime import get_pgmt_state
    from pmt_tasks.mdp.commands.pgmt_motion_command import pgmt_map_out_of_bounds
    from pmt_tasks.pgmt_math import yaw_quat, rotate
    from motion_tracking_rl.utils.pgmt_checkpoint import capture_rng, restore_rng
    raw = env.unwrapped
    terrain, robot = raw.scene.terrain, raw.scene['robot']
    sensor, cmd = raw.scene.sensors['height_scanner'], raw.command_manager.get_term('motion')
    state = get_pgmt_state(raw)
    original = deepcopy(env.get_checkpoint_state())
    rng = capture_rng(raw.device)
    ids = torch.arange(raw.num_envs, device=raw.device)
    families = terrain.terrain_types.unique(sorted=True)
    chosen = torch.stack([(terrain.terrain_types == f).nonzero()[0,0] for f in families])
    report = dict(levels=[], low_mid_high=True)
    y, x = torch.meshgrid(torch.linspace(-1,1,21,device=raw.device), torch.linspace(-1,1,21,device=raw.device), indexing='ij')
    local = torch.stack((x,y,torch.zeros_like(x)), -1).reshape(1,441,3)
    def xy_grid():
        q = yaw_quat(robot.data.body_quat_w[:,state.anchor])
        return (rotate(q[:,None].expand(-1,441,-1), local.expand(raw.num_envs,-1,-1))
                + robot.data.body_pos_w[:,state.anchor,None])[...,:2]
    def map_check():
        scan = state.terrain_scan()
        query = terrain.query(xy_grid())
        assert torch.equal(query['valid'].reshape_as(scan['valid']), scan['valid'])
        valid = scan['valid']
        error = float((scan['height_w'][valid] - query['height'].reshape_as(valid)[valid]).abs().max())
        assert error < 1e-4, error
        return error
    import omni.physx
    physics = omni.physx.get_physx_scene_query_interface()
    physx_count, physx_error = 0, 0.
    with torch.inference_mode():
        for level in (0,5,9):
            terrain.terrain_levels.fill_(level)
            terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
            raw.reset()
            assert cmd.frame_ids.eq(0).all() and cmd.placement_min_gap.ge(cmd.cfg.spawn_clearance-1e-5).all()
            obs = env.get_observations()
            error = map_check()
            save_debug(policy, obs[chosen], output / f'geometry_L{level}.npz')
            # Every represented family has a robot scan independently checked against PhysX.
            hits = sensor.data.ray_hits_w[chosen].reshape(-1,3).cpu().tolist()
            for px,py,pz in hits:
                if not np.isfinite([px,py,pz]).all():
                    continue
                elevations = []
                def collect(hit):
                    path = hit.get('collision','') if isinstance(hit,dict) else hit.collision
                    if str(path).startswith(terrain.cfg.prim_path + '/'):
                        position = hit['position'] if isinstance(hit,dict) else hit.position
                        elevations.append(position[2])
                    return True
                physics.raycast_all((px,py,terrain.ray_top),(0.,0.,-1.),1000.,collect)
                assert elevations, (px,py,pz)
                residual = abs(max(elevations)-pz)
                assert residual < 1e-4, residual
                physx_error = max(physx_error,residual)
                physx_count += 1
            report['levels'].append(dict(level=level, environments=raw.num_envs, map_error_m=error,
                min_reset_gap_m=float(cmd.placement_min_gap.min()),
                negative_height_cells=int((state.terrain_scan()['height_w'] < 0).sum())))
            progress(status='geometry_audit', geometry_checks=report)
        rotation, translation = cmd.reference_rotation.clone(), cmd.reference_translation.clone()
        birth_level, birth_family = terrain.terrain_levels.clone(), terrain.terrain_types.clone()
        def move_robot(xy):
            target_z = terrain.height(xy[None])[0]
            if not torch.isfinite(target_z):
                target_z = xy.new_tensor(0.)
            delta = torch.cat((xy-robot.data.body_pos_w[0,state.anchor,:2],
                               (target_z+1.5-robot.data.body_pos_w[0,state.anchor,2]).reshape(1)))
            root = torch.cat((robot.data.body_pos_w[:1,0]+delta,robot.data.body_quat_w[:1,0],
                              torch.zeros(1,6,device=raw.device)), -1)
            robot.write_root_link_state_to_sim(root,env_ids=ids[:1])
            raw.sim.forward()
            sensor.reset(ids[:1])
            state.invalidate_observations()
        target_family = (int(birth_family[0])+1) % len(terrain.manifest['spec']['families'])
        target = terrain.patch_bounds[5,target_family,min(6, terrain.patch_bounds.shape[2]-1)].mean(0)
        move_robot(target)
        crossed = env.get_observations()
        located = terrain.query(robot.data.body_pos_w[0,state.anchor,:2])
        assert int(located['family_index']) == target_family and int(located['tile_level']) == 5
        equal_state(cmd.reference_rotation,rotation)
        equal_state(cmd.reference_translation,translation)
        equal_state(terrain.terrain_levels,birth_level)
        equal_state(terrain.terrain_types,birth_family)
        assert map_check() < 1e-4
        save_debug(policy,crossed[:1],output/'cross_tile.npz')
        report['cross_tile'] = dict(actual_family=target_family, actual_level=5,
            birth_family=int(birth_family[0]), birth_level=int(birth_level[0]), reference_unchanged=True)
        move_robot(terrain.walkable_bounds[0] + .05)
        edge = env.get_observations()
        valid = state.terrain_scan()['valid'][0]
        assert valid.any() and (~valid).any()
        assert edge['terrain_map'][edge['terrain_valid']==0].eq(-3).all()
        save_debug(policy,edge[:1],output/'map_edge.npz')
        report['edge_valid_points'] = int(valid.sum())
        move_robot(terrain.walkable_bounds[0] - 2.)
        outside = env.get_observations()
        assert not outside['terrain_valid'][0].any()
        assert pgmt_map_out_of_bounds(raw)[0]
        debug = policy.get_perception_debug(outside[:1])
        assert not debug.glimpse.token_valid.any()
        assert torch.isfinite(policy.act_inference(outside)).all()
        _,_,done,extras = env.step(torch.zeros(raw.num_envs,29,device=raw.device))
        assert done[0] and extras['time_outs'][0]
        report['outer_map'] = dict(all_patch_invalid=True, finite_policy=True, done=True, time_out=True)
        restore_rng(rng,raw.device)
        env.load_checkpoint_state(original)
        equal_state(env.get_checkpoint_state(),original)
    report.update(physx_scan_cells=physx_count,physx_max_error_m=physx_error,
                  restored_persistent_state_exact=True)
    (output/'geometry_report.json').write_text(json.dumps(report,indent=2)+'\n')
    progress(status='geometry_complete',geometry_checks=report)
