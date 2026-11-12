"""P2.1 checks called inside the established native Isaac v6 audit process."""
import json


def audit_perception_inputs(env, report, output, progress):
    import numpy as np
    import torch
    import omni.physx
    from motion_tracking_rl.storage.rollout_storage import RolloutStorage
    from pmt_tasks.env_cfgs.pgmt.stage2 import PGMTPerceptionEnvCfg
    from pmt_tasks.isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from pmt_tasks.pgmt_curriculum import _identity
    from pmt_tasks.pgmt_math import rotate, yaw_quat
    from pmt_tasks.pgmt_perception import terrain_observation_contract
    from pmt_tasks.mdp.observations import height_scan_for_vision
    from isaaclab.managers import SceneEntityCfg
    from motion_tracking_rl.networks.pgmt_terrain_input import PGMTTerrainInput
    from pmt_tasks.pgmt_runtime import get_pgmt_state

    terrain, robot = env.scene.terrain, env.scene["robot"]
    cmd, state = env.command_manager.get_term("motion"), get_pgmt_state(env)
    sensor = env.scene.sensors["height_scanner"]
    wrapped = RslRlVecEnvWrapper(env)
    ids = torch.arange(env.num_envs, device=env.device)
    physics = omni.physx.get_physx_scene_query_interface()
    y, x = torch.meshgrid(torch.linspace(-1., 1., 21, device=env.device),
                          torch.linspace(-1., 1., 21, device=env.device), indexing="ij")
    local = torch.stack((x, y, torch.zeros_like(x)), -1).reshape(1, 441, 3)
    first_cfg, second_cfg = PGMTPerceptionEnvCfg(), PGMTPerceptionEnvCfg()
    first_cfg.pgmt_terrain_observation["nan_dropout_prob"] = .2
    first_cfg.pgmt_terrain_observation["noise_range_m"][0] = -.1
    assert second_cfg.pgmt_terrain_observation == terrain_observation_contract()
    assert second_cfg.scene.height_scanner is not None
    assert second_cfg.pgmt_terrain_asset.endswith("pgmt_stage2_v6")
    assert second_cfg.commands.motion.motion_files  # Native clipped-data resolution.
    identity_clean = _identity(env, cmd)
    assert identity_clean["terrain_observation"] == terrain_observation_contract(augmentation_enabled=False)
    (output / "clean_terrain_observation_contract.json").write_text(
        json.dumps(identity_clean["terrain_observation"], indent=2) + "\n")

    def expected_xy():
        torso = robot.data.body_pos_w[:, state.anchor]
        q = yaw_quat(robot.data.body_quat_w[:, state.anchor])
        return (rotate(q[:, None].expand(-1, 441, -1), local.expand(env.num_envs, -1, -1))
                + torso[:, None])[..., :2]

    def independent_grid():
        xy = expected_xy().cpu().numpy()
        z = np.full((env.num_envs, 441), np.nan, dtype=np.float32)
        for e in range(env.num_envs):
            for j, point in enumerate(xy[e]):
                elevations = []
                def collect(hit):
                    path = hit.get("collision", "") if isinstance(hit, dict) else hit.collision
                    if str(path).startswith(terrain.cfg.prim_path + "/"):
                        position = hit["position"] if isinstance(hit, dict) else hit.position
                        elevations.append(position[2])
                    return True
                physics.raycast_all((float(point[0]), float(point[1]), terrain.ray_top),
                                    (0., 0., -1.), 1000., collect)
                assert elevations, ("No independent terrain collision", e, j)
                z[e, j] = max(elevations)
        return xy, z.reshape(env.num_envs, 21, 21)

    def relocate(xy):
        torso = robot.data.body_pos_w[:, state.anchor].clone()
        delta = torch.cat((xy - torso[:, :2], (terrain.height(xy) + 1.5 - torso[:, 2])[:, None]), -1)
        root = torch.cat((robot.data.body_pos_w[:, 0] + delta, robot.data.body_quat_w[:, 0],
                          torch.zeros(env.num_envs, 6, device=env.device)), -1)
        robot.write_root_link_state_to_sim(root)
        cmd.reference_translation += delta
        cmd._world_tracked = None
        env.sim.forward()
        sensor.reset(ids)
        state.invalidate_observations()

    # Full 21x21 independent PhysX maps at three levels, five terrain families.
    comparisons, full_probes = [], 0
    for level in (0, 5, 9):
        terrain.terrain_types[:] = ids
        terrain.terrain_levels[:] = level
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        cmd.cfg.placement_yaw_range = (.37, .37)
        env.reset()
        bounds = terrain.patch_bounds[level, ids, 6]
        relocate(bounds.mean(1) + bounds.new_tensor([.31, -.21]))
        obs = wrapped.get_observations()
        assert obs["terrain_valid"].eq(1).all()
        xy, physics_z = independent_grid()
        observed = obs["terrain_map"].cpu().numpy()
        torso_z = robot.data.body_pos_w[:, state.anchor, 2].cpu().numpy()
        expected = (sensor.data.pos_w[:, 2].cpu().numpy()[:, None, None] - physics_z).clip(-3., 3.)
        error = np.abs(observed - expected)
        assert float(error.max()) < 1e-4, (level, float(error.max()))
        np.savez(output / f"terrain_map_L{level}.npz", terrain_map=observed, physx_height=expected,
                 valid=obs["terrain_valid"].cpu().numpy(), absolute_error_m=error,
                 world_xy=xy.reshape(env.num_envs, 21, 21, 2), world_z=physics_z,
                 torso_pos=robot.data.body_pos_w[:, state.anchor].cpu().numpy(),
                 torso_quat=robot.data.body_quat_w[:, state.anchor].cpu().numpy(),
                 families=np.array(cmd.terrain_families))
        render_comparison(output / f"terrain_map_L{level}.png", observed, expected, error,
                          cmd.terrain_families, level)
        comparisons.append(dict(level=level, patch=6, max_error_m=float(error.max()),
                                per_family_max_error_m=error.reshape(env.num_envs, -1).max(1).tolist()))
        full_probes += env.num_envs * 441
        progress(status="p21_independent_maps", full_grid_physx_probes=full_probes, map_comparisons=comparisons)

    # Corrupt only the policy copy. Compare every raw ray and clean query first.
    env.cfg.pgmt_terrain_observation = terrain_observation_contract()
    identity_noisy = _identity(env, cmd)
    assert identity_noisy != identity_clean
    assert identity_clean["terrain_observation"]["augmentation_enabled"] is False  # Copied metadata.
    (output / "terrain_observation_contract.json").write_text(
        json.dumps(identity_noisy["terrain_observation"], indent=2) + "\n")
    raw_before = sensor.data.ray_hits_w.clone()
    clean_before = {k: v.clone() for k, v in state.terrain_scan().items()}
    # Exact parity from identical physical hits/poses and identical CUDA RNG.
    rng = torch.cuda.get_rng_state(env.device)
    pmt = height_scan_for_vision(env, SceneEntityCfg("height_scanner"), nan_dropout_prob=.1,
                                append_validity_mask=True, noise_range=(-.06,.06), clip_range=(-3.,3.))
    torch.cuda.set_rng_state(rng, env.device)
    noisy = wrapped.get_observations()
    actual = torch.cat((noisy["terrain_map"].flatten(1), noisy["terrain_valid"].flatten(1)), -1)
    torch.testing.assert_close(actual, pmt, atol=0, rtol=0)
    repeat = wrapped.get_observations()
    for key in ("terrain_map", "terrain_valid"):
        assert (noisy[key] != repeat[key]).flatten(1).any(1).all(), "Each observation call must resample"
    torch.testing.assert_close(sensor.data.ray_hits_w, raw_before, atol=0, rtol=0)
    for key, value in clean_before.items():
        torch.testing.assert_close(state.terrain_scan()[key], value, atol=0, rtol=0, equal_nan=True)
    before = {key: noisy[key].clone() for key in ("terrain_map", "terrain_valid")}
    before_clean = clean_before["height"].clone()
    assert (noisy["terrain_valid"] == 0).any()
    np.savez(output / "pmt_augmented_maps_L9.npz", terrain_map=noisy["terrain_map"].cpu().numpy(),
             terrain_valid=noisy["terrain_valid"].cpu().numpy(), clean_height_m=before_clean.cpu().numpy(),
             geometry_valid=clean_before["valid"].cpu().numpy(), families=np.array(cmd.terrain_families))
    with torch.inference_mode():
        env._reset_idx(ids[:1])
        env.scene.write_data_to_sim()
        env.sim.forward()
        after = wrapped.get_observations()
    for key in before:
        assert (after[key][1:] != before[key][1:]).flatten(1).any(1).all()
    assert not torch.equal(after["terrain_map"][0], before["terrain_map"][0])
    torch.testing.assert_close(state.terrain_scan()["height"][1:], before_clean[1:], atol=1e-5, rtol=0)

    # Put scan over an outer edge while the robot/reference are still inside.
    xy = robot.data.body_pos_w[:, state.anchor, :2].clone()
    xy[0] = xy.new_tensor([-119.15, -56.91])
    relocate(xy)
    edge = wrapped.get_observations()
    invalid = ~edge["terrain_valid"].bool()
    geometry_invalid = ~state.terrain_scan()["valid"]
    assert geometry_invalid[0].any() and invalid[geometry_invalid].all()
    assert edge["terrain_map"][invalid].eq(-3).all()

    # Actual control steps -> wrapper -> native rollout -> shuffled minibatches.
    # Dummy action/value fields isolate the observation transport; no PPO update.
    terrain.terrain_levels.zero_()
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    env.reset()
    sources, residuals, storage = {}, [], None
    steps, max_dynamic_error, geometry_cells, dropped_cells = 24, 0., 0, 0
    for step in range(steps):
        with torch.inference_mode():
            obs, _, _, _ = wrapped.step(torch.zeros(env.num_envs, 29, device=env.device))
        repeated = wrapped.get_observations()
        for key in ("terrain_map", "terrain_valid"):
            assert (obs[key] != repeated[key]).flatten(1).any(1).all()
        scan = state.terrain_scan()
        valid = scan["valid"]
        query = terrain.query(expected_xy())
        assert torch.equal(query["valid"].reshape_as(valid), valid)
        expected = (query["height"] - robot.data.body_pos_w[:, state.anchor, 2:3]).reshape_as(valid)
        error = float((scan["height"][valid] - expected[valid]).abs().max())
        max_dynamic_error = max(max_dynamic_error, error)
        assert error < 1e-4
        observed_valid = obs["terrain_valid"].bool()
        assert not observed_valid[~valid].any()
        assert obs["terrain_map"][~observed_valid].eq(-3).all()
        assert obs["terrain_map"].isfinite().all() and obs["terrain_map"].abs().max() <= 3.
        network_map, network_mask = PGMTTerrainInput()(obs["terrain_map"], obs["terrain_valid"])
        torch.testing.assert_close(network_map, obs["terrain_map"] / 3, atol=0, rtol=0)
        assert network_map[~observed_valid].eq(-1).all()
        torch.testing.assert_close(network_mask, obs["terrain_valid"], atol=0, rtol=0)
        geometry_cells += int(valid.sum())
        dropped_cells += int((valid & ~observed_valid).sum())
        # Exclude saturating heights when measuring the pre-clipping noise.
        unclipped = observed_valid & (scan["height"].abs() < 2.94)
        residuals.append((obs["terrain_map"] + scan["height"])[unclipped].cpu())
        packed = obs.clone()
        packed["audit_sample_id"] = torch.arange(step*env.num_envs, (step+1)*env.num_envs,
                                                  device=env.device)[:, None].float()
        if storage is None:
            storage = RolloutStorage("rl", env.num_envs, steps, packed, [29], device=env.device, num_value_heads=3)
        t = storage.Transition()
        t.observations, t.actions = packed, torch.zeros(env.num_envs, 29, device=env.device)
        t.rewards = t.values = torch.zeros(env.num_envs, 3, device=env.device)
        t.dones = t.actions_log_prob = torch.zeros(env.num_envs, device=env.device)
        t.action_mean, t.action_sigma = t.actions, torch.ones_like(t.actions)
        storage.add_transitions(t)
        for e in range(env.num_envs):
            sources[step*env.num_envs+e] = {k: v[e].clone() for k, v in packed.items()}
    visits = 0
    for batch, *_ in storage.mini_batch_generator(4, 2):
        rng = torch.cuda.get_rng_state(env.device)
        normalized, mask = PGMTTerrainInput()(batch["terrain_map"], batch["terrain_valid"])
        assert torch.equal(rng, torch.cuda.get_rng_state(env.device))
        torch.testing.assert_close(normalized, batch["terrain_map"] / 3, atol=0, rtol=0)
        torch.testing.assert_close(mask, batch["terrain_valid"], atol=0, rtol=0)
        for i, sample_id in enumerate(batch["audit_sample_id"].flatten().long().tolist()):
            for key, expected in sources[sample_id].items():
                torch.testing.assert_close(batch[key][i], expected, atol=0, rtol=0)
            visits += 1
    assert visits == steps * env.num_envs * 2
    residual = torch.cat(residuals)
    assert abs(float(residual.mean())) < .001
    assert .0335 < float(residual.std()) < .0358
    assert float(residual.min()) >= -.060001 and float(residual.max()) <= .060001
    dropout_rate = dropped_cells / geometry_cells
    assert .09 < dropout_rate < .11
    env.cfg.pgmt_terrain_observation = terrain_observation_contract(augmentation_enabled=False)
    zero = wrapped.get_observations()
    clean = state.terrain_scan()
    expected = torch.where(clean["valid"], -clean["height"],
                           (sensor.data.pos_w[:, 2] - 30.)[:, None, None]).clamp(-3., 3.)
    torch.testing.assert_close(zero["terrain_map"], expected, atol=1e-5, rtol=0)
    torch.testing.assert_close(zero["terrain_valid"], state.terrain_scan()["valid"].float(), atol=0, rtol=0)
    env.cfg.pgmt_terrain_observation = terrain_observation_contract()
    report["p21"] = dict(contract=identity_noisy["terrain_observation"], independent_full_grid_physx_probes=full_probes,
                         full_grid_comparisons=comparisons, native_control_steps=steps,
                         max_dynamic_clean_map_error_m=max_dynamic_error, partial_reset_preserves_other_env_geometry=True,
                         same_step_calls_resample_all_envs=True, partial_reset_next_observation_resamples_all_envs=True,
                         exact_original_pmt_sensor_to_observation_parity=True,
                         sensor_and_clean_query_unchanged_by_augmentation=True,
                         augmented_edge_invalid_minus_three=True, noise_range_m=[-.06, .06],
                         nan_dropout_prob=.1, geometry_valid_samples=geometry_cells, dropped_samples=dropped_cells,
                         measured_dropout_rate=dropout_rate, clip_range_m=[-3., 3.], network_normalization_scale_m=3.,
                         observation_units_m=True, normalization_only_in_network=True,
                         network_preprocessing_deterministic=True, disabled_noise_preserves_pmt_sign_fill_clip=True,
                         noise_samples=residual.numel(), measured_noise_mean_m=float(residual.mean()),
                         measured_noise_std_m=float(residual.std()), measured_noise_min_m=float(residual.min()),
                         measured_noise_max_m=float(residual.max()), rollout_samples=steps*env.num_envs,
                         minibatch_sample_visits=visits, all_observation_groups_preserved=True,
                         config_instances_independent=True, environment_identity_includes_perception=True,
                         stage1_policy_unchanged=True, terrain_glimpse_ifm_connected=False, ppo_updates=0)
    progress(status="p21_inputs_passed", p21=report["p21"])


def render_comparison(path, observed, expected, error, families, level):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(families), 3, figsize=(12, 16), constrained_layout=True)
    for row, family in enumerate(families):
        lo, hi = min(observed[row].min(), expected[row].min()), max(observed[row].max(), expected[row].max())
        if hi - lo < 1e-4:
            hi = lo + .01
        for col, data in enumerate((observed[row], expected[row], error[row] * 1e6)):
            opts = dict(vmin=lo, vmax=hi) if col < 2 else dict(vmin=0, vmax=max(10., float(data.max())))
            im = axes[row, col].imshow(data, origin="lower", extent=(-1, 1, -1, 1),
                                        cmap="viridis" if col < 2 else "magma", **opts)
            axes[row, col].set_title(f"{family} | " + ("terrain_map (m)", "clip(sensor Z - PhysX) (m)", "abs error (um)")[col])
            axes[row, col].set_xlabel("torso yaw X (m)")
            axes[row, col].set_ylabel("torso yaw Y (m)")
            fig.colorbar(im, ax=axes[row, col], shrink=.8)
    fig.suptitle(f"PGMT P2.1 schema 3 | v6 L{level} | PMT-sign 21x21 map vs independent collisions")
    fig.savefig(path, dpi=120)
    plt.close(fig)
