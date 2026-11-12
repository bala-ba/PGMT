"""Native Stage2 regression: random resets, PPO -> EMD worker -> strict resume."""
import argparse
from copy import deepcopy
from dataclasses import fields
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher
from pmt_tasks.isaacsim6_legacy import install_tensor_import_bridge

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", default=str(ROOT / "logs/pgmt_stage2_center2x2/train_01/model_19999.pt"))
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
install_tensor_import_bridge()
app = AppLauncher(args, multi_gpu=False, fast_shutdown=True).app


def main():
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.envs import ManagerBasedRLEnv
    from pmt_tasks.isaacsim6_legacy import install_rigid_scene_schema_bridge
    from pmt_tasks.builder import build_env_cfg, build_agent_cfg
    from pmt_tasks.isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from pmt_tasks.pgmt_performance import install_performance
    from motion_tracking_rl.runners import PGMTStage2Runner
    from motion_tracking_rl.utils.pgmt_checkpoint import canonical
    from scripts.pgmt_stage2_checks import equal_state

    install_rigid_scene_schema_bridge()
    torch.set_num_threads(4)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = build_env_cfg("pgmt_stage2_terrain")
    cfg.pmt_motion_paths = [str(ROOT / "assets/motions/pgmt_clipped_50hz_tcrs_v1" / (name + ".npz"))
                            for name in ("walk1_subject1_clip0", "dance1_subject1_clip0")]
    cfg.pgmt_emd_sampling.update(evaluation_interval=20, evaluation_num_envs=2)
    cfg.__post_init__()
    cfg.scene.num_envs, cfg.seed = 10, 1731
    cfg.sim.device = cfg.commands.motion.storage_device = args.device
    cfg.sim.use_fabric = args.device != "cpu"
    for name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
        setattr(cfg.events, name, None)
    old = cfg.scene.robot.spawn
    shared = {f.name: deepcopy(getattr(old, f.name)) for f in fields(sim_utils.UsdFileCfg)
              if f.name not in ("func", "usd_path") and hasattr(old, f.name)}
    cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
        usd_path=str(ROOT / "logs/pgmt_p1_acceptance/native_robot/main.usd"), **shared)
    contract = canonical(cfg.to_dict())
    raw = ManagerBasedRLEnv(cfg)
    try:
        env = RslRlVecEnvWrapper(raw)
        command, terrain = raw.command_manager.get_term("motion"), raw.scene.terrain
        sampler = command.sampler
        original = sampler.state_dict()
        sampled = []
        ids = torch.arange(raw.num_envs, device=raw.device)
        with torch.inference_mode():
            for fast in (False, True):
                if fast:
                    install_performance(raw, {"reset_stats", "placement"})
                for level in (0, 5, 9):
                    terrain.terrain_levels.fill_(level)
                    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
                    for _ in range(3):
                        env.reset()
                        sampled.extend(command.frame_ids.tolist())
                        assert command.frame_ids.ge(0).all() and command.frame_ids.lt(498).all()
                        expected = terrain.patch_bounds[terrain.terrain_levels, terrain.terrain_types,
                                                        command.placement_patch_ids].mean(1)
                        torch.testing.assert_close(command.placement_xy, expected, rtol=0, atol=0)
                        assert command.compatible_motions[terrain.terrain_types, command.motion_ids].all()
                        ref = command._source_reference(command.motion_ids, command.frame_ids)
                        torch.testing.assert_close(raw.scene["robot"].data.joint_pos, ref["joint_pos"], rtol=0, atol=1e-5)
                    partial = torch.tensor([1, 4, 8], device=raw.device)
                    keep = ids[~torch.isin(ids, partial)]
                    before = command.reference_translation[keep].clone()
                    raw._reset_idx(partial)
                    torch.testing.assert_close(command.reference_translation[keep], before, rtol=0, atol=0)
        assert min(sampled) < 50 and max(sampled) > 450
        equal_state(sampler.state_dict(), original)
        agent = build_agent_cfg("pgmt_stage2_terrain")
        agent.device, agent.num_steps_per_env = args.device, 2
        agent.algorithm.num_learning_epochs = agent.algorithm.num_mini_batches = 1
        agent.logger, agent.save_interval = "tensorboard", 1
        agent.pgmt_training_contract = dict(environment=contract)
        runner = PGMTStage2Runner(env, agent.to_dict(), log_dir=str(output), device=args.device)
        runner.initialize_policy(args.checkpoint)
        runner.learn(1, init_at_random_ep_len=False)
        assert sampler.last_evaluation_step == 20 and sampler.evaluation_count == 2
        assert torch.isfinite(sampler.emd).all() and sampler.weights.min() >= 2
        path = output / "final.pt"
        runner.save(path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        expected = deepcopy(sampler.state_dict())
        sampler.weights.fill_(1.)
        sampler.emd.zero_()
        sampler.last_evaluation_step = sampler.evaluation_count = 0
        runner.load(path)
        equal_state(sampler.state_dict(), expected)
        equal_state(env.get_checkpoint_state(), saved["environment_state_dict"])
        assert runner.completed_iterations == 1 and not sampler.due(20)
        report = dict(passed=True, device=args.device, reset_samples=len(sampled), start_frame_range=[min(sampled), max(sampled)],
                      levels=[0, 5, 9], fast_placement=True, partial_reset_isolated=True,
                      ppo_updates=1, evaluation_count=sampler.evaluation_count,
                      emd=sampler.emd.tolist(), weights=sampler.weights.tolist(),
                      strict_resume=True, evaluation_step=sampler.last_evaluation_step)
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print("PGMT_EMD_INTEGRATION_PASSED", json.dumps(report), flush=True)
    finally:
        raw.close()


from pmt_tasks.runtime_options import run_and_close
run_and_close(main, app)
