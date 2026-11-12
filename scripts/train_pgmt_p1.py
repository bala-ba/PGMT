"""GUI-capable P1 terrain training from stage 1, without perception injection."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", default=str(ROOT / "logs/pgmt_stage1/pgmt_8192x20000/model_15000.pt"))
parser.add_argument("--motion_dir", default=str(ROOT / "assets/motions/pgmt_clipped_50hz_tcrs_v1"))
parser.add_argument("--terrain_asset", default=str(ROOT / "assets/terrain/pgmt_stage2_v5"),
                    help="Audited v5 tiles or v6 repeated native patches; v6 enables random patch placement")
parser.add_argument("--num_envs", type=int, default=140)
parser.add_argument("--max_iterations", type=int, default=1000, help="Additional PPO updates after loading")
parser.add_argument("--save_interval", type=int, default=50)
parser.add_argument("--log_dir", default=None)
parser.add_argument("--seed", type=int, default=722)
parser.add_argument("--viewer_mode", choices=("follow", "global", "tile"), default="follow",
                    help="Initial GUI view; switch freely in the viewport panel")
parser.add_argument("--viewer_terrain", default="flat",
                    choices=("flat", "slopes", "stairs", "boxes", "random_rough", "up_stairs", "up_slope", "slope", "rough"),
                    help="Initial tile terrain family")
parser.add_argument("--viewer_level", type=int, choices=range(10), default=0,
                    help="Initial tile difficulty level (L0-L9)")
parser.add_argument("--viewer_tile_offset", type=float, nargs=3, metavar=("DX", "DY", "DZ"),
                    help="Camera offset in world axes from the selected tile origin, in meters")
parser.add_argument("--robot_usd", default=None, help="Reuse the audited native URDF conversion without reconverting")
parser.add_argument("--isaacsim6_compat", action="store_true", help="Bridge the relocated PhysX tensor import in Sim 6")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
terrain_source = json.loads((Path(args.terrain_asset) / "manifest.json").read_text())
if args.num_envs < len(terrain_source["spec"]["families"]) or args.max_iterations < 1:
    parser.error("Use at least one environment per terrain family and one PPO update")
if args.viewer_terrain not in terrain_source["spec"]["families"]:
    parser.error("The selected viewer terrain is not in this asset")
output = Path(args.log_dir or ROOT / "logs/pgmt_p1_blind" / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
output.mkdir(parents=True, exist_ok=True)
status_file = output / "training_status.json"
if status_file.exists():
    parser.error(f"Refusing to overwrite an existing acceptance run: {output}")
status = dict(status="launching", pid=os.getpid(), started_at=datetime.now().isoformat(),
              headless=bool(args.headless), checkpoint=str(Path(args.checkpoint).resolve()),
              motion_dir=str(Path(args.motion_dir).resolve()), num_envs=args.num_envs,
              terrain_asset=str(Path(args.terrain_asset).resolve()),
              terrain_sha256=terrain_source["asset_sha256"], terrain_families=terrain_source["spec"]["families"],
              requested_updates=args.max_iterations, completed_updates=0, output=str(output))


def save_status(**values):
    status.update(values, updated_at=datetime.now().isoformat())
    temporary = status_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(status_file)


save_status()
if args.isaacsim6_compat:
    from pmt_tasks.isaacsim6_legacy import install_tensor_import_bridge
    install_tensor_import_bridge()
app = AppLauncher(args, multi_gpu=False, fast_shutdown=True).app


def main():
    if args.isaacsim6_compat:
        from pmt_tasks.isaacsim6_legacy import install_rigid_scene_schema_bridge
        install_rigid_scene_schema_bridge()
    import math
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.utils.io import dump_yaml
    from motion_tracking_rl import compat  # noqa: F401; register PGMT policy
    from motion_tracking_rl.runners import OnPolicyRunner
    from pmt_tasks.agent_cfgs.pgmt import PGMTStage1PPORunnerCfg
    from pmt_tasks.env_cfgs.pgmt.p1_blind import PGMTBlindTerrainEnvCfg
    from pmt_tasks.isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from pmt_tasks.pgmt_runtime import get_pgmt_state

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    checkpoint = Path(args.checkpoint).resolve()
    source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    loaded = torch.load(checkpoint, weights_only=False, map_location="cpu")
    cfg = PGMTBlindTerrainEnvCfg()
    cfg.pmt_motion_paths = [str(Path(args.motion_dir).resolve())]
    cfg.pgmt_terrain_asset = str(Path(args.terrain_asset).resolve())
    cfg.__post_init__()
    if "patches" in terrain_source:
        cfg.commands.motion.max_placement_attempts = 128
        # 800 native patch colliders exceed the default 64 MiB narrow-phase
        # stack even with 160 robots (~90 MiB measured). Never drop contacts.
        cfg.sim.physx.gpu_collision_stack_size = max(cfg.sim.physx.gpu_collision_stack_size, 2**28)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = cfg.commands.motion.storage_device = args.device
    cfg.sim.enable_scene_query_support = True
    if args.isaacsim6_compat:
        # GPU articulation poses render through the native Fabric transform
        # cache. USD transform writes remain suppressed in GPU tensor mode.
        cfg.sim.use_fabric = True
    if args.robot_usd:
        import copy
        from dataclasses import fields
        import isaaclab.sim as sim_utils
        usd = Path(args.robot_usd).resolve()
        if not usd.is_file():
            raise FileNotFoundError(usd)
        old_spawn = cfg.scene.robot.spawn
        shared = {field.name: copy.deepcopy(getattr(old_spawn, field.name))
                  for field in fields(sim_utils.UsdFileCfg)
                  if field.name not in ("func", "usd_path") and hasattr(old_spawn, field.name)}
        cfg.scene.robot.spawn = sim_utils.UsdFileCfg(usd_path=str(usd), **shared)
    cfg.scene.contact_forces.debug_vis = False
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.env_index = 0
    cfg.viewer.eye, cfg.viewer.lookat = (3., 3., 2.), (0., 0., 0.)
    agent = PGMTStage1PPORunnerCfg()
    agent.seed, agent.device = args.seed, args.device
    agent.experiment_name, agent.logger = "pgmt_p1_blind", "tensorboard"
    agent.max_iterations, agent.save_interval = args.max_iterations, args.save_interval
    dump_yaml(str(output / "params/env.yaml"), cfg)
    dump_yaml(str(output / "params/agent.yaml"), agent)
    (output / "launch.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    save_status(status="building_environment", checkpoint_sha256=source_hash)
    raw = ManagerBasedRLEnv(cfg=cfg)
    runner = None
    try:
        env = RslRlVecEnvWrapper(raw)
        if args.isaacsim6_compat and not args.headless:
            # Sim 6 owns its Fabric enable/disable state separately from the
            # legacy /physics/fabricEnabled flag used by Lab 2.3.
            from isaacsim.core.simulation_manager import SimulationManager
            SimulationManager.enable_fabric(True)
        # Recheck terrain collisions in the ACTUAL training runtime, including
        # when GUI compatibility requires a newer simulator than the P1 suite.
        import omni.physx
        terrain = raw.scene.terrain
        query = omni.physx.get_physx_scene_query_interface()
        probe_count, probe_error = 0, 0.
        probe_sets = [(tile["name"], tile["world_probes"]) for tile in terrain.surface_records]
        if terrain.manifest.get("seam_probes"):
            probe_sets.append(("connections", terrain.manifest["seam_probes"]))
        mesh_error = 0.
        for name, probes in probe_sets:
            expected = torch.tensor(probes, device=raw.device, dtype=torch.float32)
            heights = terrain.height(expected[:, :2])
            assert torch.isfinite(heights).all(), f"Missing query mesh: {name}"
            error = float((heights - expected[:, 2]).abs().max())
            assert error < 1e-4, ("query mesh", name, error)
            mesh_error = max(mesh_error, error)
            for x, y, expected_z in probes:
                elevations = []
                def collect(hit):
                    path = hit.get("collision", "") if isinstance(hit, dict) else hit.collision
                    if str(path).startswith(terrain.cfg.prim_path + "/"):
                        position = hit["position"] if isinstance(hit, dict) else hit.position
                        elevations.append(position[2])
                    return True
                query.raycast_all((x, y, terrain.ray_top), (0., 0., -1.), 1000., collect)
                assert elevations, f"Training runtime missing terrain collision: {name}"
                error = abs(max(elevations) - expected_z)
                assert error < 1e-4, (name, error)
                probe_error = max(probe_error, error)
                probe_count += 1
        save_status(training_runtime_terrain_probes=probe_count, training_runtime_height_error_m=probe_error,
                    training_runtime_mesh_query_error_m=mesh_error, terrain_regions=len(terrain.manifest["tiles"]),
                    terrain_patches=len(terrain.manifest.get("patches", [])))
        assert "height_scanner" not in raw.scene.sensors
        assert set(env.get_observations().keys()) == {"proprio", "proprio_history", "command_window", "critic"}
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(output), device=args.device)
        expected = loaded["policy_metadata"]
        actual = runner.alg.policy.get_checkpoint_metadata()
        assert actual["signature"] == expected["signature"], "Checkpoint policy signature differs"
        assert actual["obs_schema"]["hash"] == expected["obs_schema"]["hash"], "Blind observation schema differs"
        runner.load(str(checkpoint), load_optimizer=True, map_location=args.device)
        for name, value in runner.alg.policy.state_dict().items():
            torch.testing.assert_close(value.cpu(), loaded["model_state_dict"][name], rtol=0, atol=0)
        assert runner.current_learning_iteration == loaded["iter"]
        # The native loader restores optimizer groups, but PPO's adaptive LR
        # scalar otherwise remains its config default. Keep both synchronized.
        runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]["lr"]
        initial_lr = runner.alg.learning_rate
        trainable_before = {name: p.detach().clone() for name, p in runner.alg.policy.named_parameters()}
        state = get_pgmt_state(raw)
        cmd = raw.command_manager.get_term("motion")
        course = raw._pgmt_terrain_curriculum
        assert cmd.cfg.fixed_start_frame == 0 and not cmd.cfg.enable_frame_sampling
        assert cmd.cfg.require_valid_support_start is False and cmd.cfg.chunk_length == 0
        assert cmd.cfg.sampler_type == "adaptive"
        assert len(cmd.data_store.motions) > 0
        # Stage-1 initialization starts a new terrain curriculum. A v6 resume
        # intentionally restores nonzero counts; runner.load has already
        # validated its asset/motion/reset identity and sampler payload.
        if "environment_state_dict" not in loaded:
            assert not course.counts.any()
        checks = dict(reset_poses=0, sampler_outcomes=0, physics_control_steps=0, done_events=0,
                      reference_drift_checks=0, contact_freshness_checks=0)
        gui = None
        if not args.headless:
            import omni.ui as ui
            from pxr import Usd, UsdGeom
            from isaacsim.core.utils.viewports import set_camera_view
            from omni.kit.viewport.utility import get_active_viewport
            viewport = get_active_viewport()
            viewport.set_active_camera(cfg.viewer.cam_prim_path)
            # Use the actual viewport camera explicitly: old Lab camera event
            # subscriptions and Sim 6's viewport can otherwise disagree.
            raw.viewport_camera_controller.cfg.origin_type = "world"
            gui = dict(env=0, mode="follow", visible_robots=1)
            tile_records = {(tile["level"], tile["family_index"]): tile for tile in terrain.manifest["tiles"]}
            tile_families = terrain.manifest["spec"]["families"]
            tile_levels = terrain.manifest["spec"]["num_rows"]
            tile_span = max(terrain.manifest["spec"]["size"])
            tile_offset = args.viewer_tile_offset or [.8 * tile_span, -1.2 * tile_span, 2. * tile_span]
            tile_offset_models = [ui.SimpleFloatModel(value) for value in tile_offset]

            def set_robot_visibility(selected_env=None):
                # Visibility only: never deactivate prims or change physics.
                for index in range(raw.num_envs):
                    prim = raw.sim.stage.GetPrimAtPath(f"/World/envs/env_{index}")
                    UsdGeom.Imageable(prim).GetVisibilityAttr().Set(
                        UsdGeom.Tokens.inherited if selected_env is None or index == selected_env
                        else UsdGeom.Tokens.invisible)

            def update_camera():
                root = raw.scene["robot"].data.root_pos_w[gui["env"]].detach().cpu().tolist()
                gui["root"] = root
                if gui["mode"] != "follow":
                    # Global and tile views preserve mouse/keyboard navigation,
                    # even if a robot resets, changes level or crosses tiles.
                    return
                eye = [root[0] + 3., root[1] + 3., root[2] + 2.]
                set_camera_view(eye=eye, target=root, camera_prim_path=str(viewport.camera_path))
                gui.update(eye=eye, camera=str(viewport.camera_path))

            def follow_selected():
                gui.update(mode="follow", visible_robots=1)
                set_robot_visibility(gui["env"])
                update_camera()

            def choose_family(family):
                candidates = (terrain.terrain_types == family).nonzero().flatten()
                if candidates.numel():
                    gui["env"] = int(candidates[0])
                    follow_selected()

            def show_all():
                gui.update(mode="global", visible_robots=raw.num_envs)
                set_robot_visibility()
                bounds = terrain.tile_bounds.reshape(-1, 2, 2)
                lower, upper = bounds[:, 0].amin(0), bounds[:, 1].amax(0)
                center = ((lower + upper) / 2).detach().cpu().tolist()
                span = float((upper - lower).max())
                target = [*center, 0.]
                eye = [center[0] + .8 * span, center[1] - 1.2 * span, 2. * span]
                # Frame the terrain once on entry; subsequent steps leave the
                # camera alone. Clicking this button again restores the overview.
                set_camera_view(eye=eye, target=target, camera_prim_path=str(viewport.camera_path))
                gui.update(eye=eye, camera=str(viewport.camera_path))
                update_camera()

            def view_tile():
                family = tile_family_selector.model.get_item_value_model().as_int
                level = tile_level_selector.model.get_item_value_model().as_int
                tile = tile_records[level, family]
                offset = [model.as_float for model in tile_offset_models]
                if not all(math.isfinite(value) for value in offset) or sum(value * value for value in offset) < 1e-8:
                    tile_label.text = "Camera offset must be finite and non-zero"
                    return
                # Use the fixed terrain asset origin, including negative Z for
                # inverted pyramids. This never consults robot/birth positions.
                target = list(tile["origin"])
                eye = [position + delta for position, delta in zip(target, offset)]
                set_robot_visibility()
                gui.update(mode="tile", visible_robots=raw.num_envs,
                           tile_family=tile["family"], tile_level=level, tile_name=tile["name"],
                           tile_difficulty=tile["difficulty"], tile_origin=target, tile_camera_offset=offset,
                           eye=eye, target=target, camera=str(viewport.camera_path))
                set_camera_view(eye=eye, target=target, camera_prim_path=str(viewport.camera_path))
                tile_label.text = (f"{tile['name']} | d={tile['difficulty']:.3f} | origin "
                                   f"({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}) m")
                update_camera()

            panel = ui.Window("PGMT P1 | Live blind PPO", width=600, height=370)
            with panel.frame:
                with ui.VStack():
                    ui.Label("Actual Isaac physics and PPO; no terrain input to policy")
                    with ui.HStack():
                        ui.Button("Global / free camera", clicked_fn=show_all)
                        ui.Button("Follow selected robot", clicked_fn=follow_selected)
                    ui.Label("Terrain buttons switch to single-robot follow mode")
                    for start in range(0, len(tile_families), 4):
                        with ui.HStack():
                            for family in range(start, min(start + 4, len(tile_families))):
                                ui.Button(cmd.terrain_families[family], clicked_fn=lambda family=family: choose_family(family))
                    ui.Label("View a fixed tile (independent of robot assignment)")
                    with ui.HStack():
                        ui.Label("Terrain", width=60)
                        tile_family_selector = ui.ComboBox(tile_families.index(args.viewer_terrain), *tile_families)
                        ui.Label("Level", width=45)
                        tile_level_selector = ui.ComboBox(args.viewer_level, *[f"L{level}" for level in range(tile_levels)], width=75)
                        ui.Button("View tile", clicked_fn=view_tile, width=95)
                    with ui.HStack():
                        ui.Label("Camera offset (m)", width=135)
                        for axis, model in zip(("dX", "dY", "dZ"), tile_offset_models):
                            ui.Label(axis, width=25)
                            ui.FloatDrag(model=model, step=.5)
                    ui.Label("Click View tile to apply; then move the camera freely")
                    tile_label = ui.Label("Choose a terrain and L0-L9, then click View tile")
                    gui_label = ui.Label("Checkpoint loaded; starting PPO")
            choose_family(0)
            if args.viewer_mode == "global":
                show_all()
            elif args.viewer_mode == "tile":
                view_tile()
        last_outcome_step = torch.full((raw.num_envs,), -1, device=raw.device, dtype=torch.long)
        old_reset, old_outcome, old_step = cmd._reset_robot_state, cmd._record_outcomes, env.step
        patch_counts = torch.zeros(len(cmd.terrain_families), 16, dtype=torch.long, device=raw.device)
        max_placement_attempts = 0

        def check_reset(ids):
            nonlocal max_placement_attempts
            assert cmd.frame_ids[ids].eq(0).all()
            assert cmd.placement_min_gap[ids].ge(cmd.cfg.spawn_clearance - 1e-5).all()
            assert cmd.compatible_motions[raw.scene.terrain.terrain_types[ids], cmd.motion_ids[ids]].all()
            ref = cmd.transform_reference(cmd._source_reference(cmd.motion_ids[ids], cmd.frame_ids[ids]), ids)
            torch.testing.assert_close(raw.scene["robot"].data.body_pos_w[ids], ref["body_pos"], atol=5e-5, rtol=0)
            if terrain.has_patches:
                located = terrain.tile_map.locate(cmd.placement_xy[ids])
                assert located["tile_valid"].all()
                assert torch.equal(located["tile_level"], terrain.terrain_levels[ids])
                assert torch.equal(located["family_index"], terrain.terrain_types[ids])
                assert torch.equal(located["patch_id"], cmd.placement_patch_ids[ids])
                patch_counts.view(-1).add_(torch.bincount(
                    terrain.terrain_types[ids] * 16 + cmd.placement_patch_ids[ids], minlength=patch_counts.numel()))
            max_placement_attempts = max(max_placement_attempts, int(cmd.placement_attempts[ids].max()))
            checks["reset_poses"] += len(ids)

        check_reset(torch.arange(raw.num_envs, device=raw.device))
        (output / "initial_spawns.json").write_text(json.dumps(dict(
            level=terrain.terrain_levels.tolist(), family=terrain.terrain_types.tolist(),
            patch=cmd.placement_patch_ids.tolist(), xy=cmd.placement_xy.tolist(), yaw=cmd.placement_yaw.tolist(),
            min_gap=cmd.placement_min_gap.tolist(), attempts=cmd.placement_attempts.tolist()), indent=2) + "\n")

        def audited_reset(ids):
            old_reset(ids)
            check_reset(ids)

        def audited_outcome(ids, terminated):
            eligible = course.sampler_pending[ids] & (course.settled_step[ids] == raw.common_step_counter)
            assert last_outcome_step[ids[eligible]].ne(raw.common_step_counter).all()
            old_outcome(ids, terminated)
            last_outcome_step[ids[eligible]] = raw.common_step_counter
            checks["sampler_outcomes"] += int(eligible.sum())

        def audited_step(actions):
            before_r, before_t = cmd.reference_rotation.clone(), cmd.reference_translation.clone()
            obs, reward, done, extras = old_step(actions)
            survivors = ~done.bool()
            torch.testing.assert_close(cmd.reference_rotation[survivors], before_r[survivors], rtol=0, atol=0)
            torch.testing.assert_close(cmd.reference_translation[survivors], before_t[survivors], rtol=0, atol=0)
            contacts = state.reference_contacts()
            assert torch.equal(contacts["motion_ids"], cmd.motion_ids)
            assert torch.equal(contacts["frame_ids"], cmd.frame_ids)
            assert all(value.isfinite().all() for value in obs.values()) and reward.isfinite().all()
            torch.testing.assert_close(extras["reward_groups"].sum(-1), reward, rtol=2e-5, atol=2e-6)
            checks["physics_control_steps"] += 1
            checks["done_events"] += int(done.sum())
            checks["reference_drift_checks"] += int(survivors.sum())
            checks["contact_freshness_checks"] += raw.num_envs
            if gui is not None:
                update_camera()
            return obs, reward, done, extras

        cmd._reset_robot_state, cmd._record_outcomes, env.step = audited_reset, audited_outcome, audited_step
        old_log = runner.log

        def audited_log(locs, *log_args, **log_kwargs):
            old_log(locs, *log_args, **log_kwargs)
            losses = {k: float(v) for k, v in locs["loss_dict"].items()}
            assert all(math.isfinite(v) for v in losses.values())
            assert all(p.isfinite().all() for p in runner.alg.policy.parameters())
            delta = max(float((p.detach() - trainable_before[name]).abs().max())
                        for name, p in runner.alg.policy.named_parameters())
            assert delta > 0., "PPO did not change any trainable policy parameter"
            updated = status["completed_updates"] + 1
            save_status(status="training", completed_updates=updated,
                        current_iteration=int(locs["it"]), losses=losses,
                        max_parameter_change=delta, checks=dict(checks),
                        curriculum_outcomes=course.counts.sum((0, 1)).tolist(),
                        terrain_level_histogram=torch.bincount(raw.scene.terrain.terrain_levels, minlength=10).tolist(),
                        learning_rate=runner.alg.learning_rate)
            if terrain.has_patches:
                save_status(spawn_counts_by_family_patch=patch_counts.tolist(),
                            max_placement_attempts_observed=max_placement_attempts)
            with (output / "progress.jsonl").open("a") as file:
                file.write(json.dumps({k: status[k] for k in ("updated_at", "completed_updates", "current_iteration", "losses", "checks")}) + "\n")
            if updated == 1:
                runner.save(str(output / "acceptance_first_update.pt"))
            if gui is not None:
                if gui["mode"] == "global":
                    gui_label.text = f"PPO update {updated} | Global / free camera | {raw.num_envs} robots"
                elif gui["mode"] == "tile":
                    gui_label.text = f"PPO update {updated} | Tile {gui['tile_name']} / free camera"
                else:
                    gui_label.text = f"PPO update {updated} | Follow env {gui['env']} | frame {int(cmd.frame_ids[gui['env']])}"
                pelvis = raw.sim.stage.GetPrimAtPath(f"/World/envs/env_{gui['env']}/Robot/pelvis")
                gui["usd_root"] = list(UsdGeom.Xformable(pelvis).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation())
                gui["usd_visibility"] = str(UsdGeom.Imageable(pelvis).ComputeVisibility())
                gui["usd_pose_error_m"] = sum((a - b) ** 2 for a, b in zip(gui["usd_root"], gui["root"])) ** .5
                gui["physics_updates_usd"] = bool(raw.sim.carb_settings.get("/physics/updateToUsd"))
                gui["fabric_enabled"] = raw.sim.is_fabric_enabled()
                save_status(gui=dict(gui))
            if not args.headless and updated in (1, 5, 50):
                from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
                capture_viewport_to_file(get_active_viewport(), str(output / f"training_viewport_{updated}.png"))

        runner.log = audited_log
        save_status(status="checkpoint_loaded", loaded_iteration=runner.current_learning_iteration,
                    loaded_tensor_count=len(loaded["model_state_dict"]), checkpoint_weights_exact=True,
                    policy_signature_match=True, observation_schema_match=True, perception_injected=False,
                    motion_count=len(cmd.data_store.motions), initial_learning_rate=initial_lr,
                    placement_mode=cmd.placement_mode, compatibility_families=list(terrain.compatibility_families),
                    curriculum_initialized_fresh="environment_state_dict" not in loaded,
                    policy_value_groups=list(runner.alg.policy.value_groups))
        print(f"[P1] Starting real PPO: headless={args.headless}, output={output}", flush=True)
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == source_hash
        save_status(status="completed", source_checkpoint_unchanged=True)
    except BaseException as error:
        save_status(status="failed", error=repr(error))
        raise
    finally:
        if runner is not None and runner.writer is not None:
            runner.writer.close()
        raw.close()


from pmt_tasks.runtime_options import run_and_close


def checked_main():
    try:
        main()
    except BaseException as error:
        save_status(status="failed", error=repr(error))
        raise


run_and_close(checked_main, app)
