"""Export a PMT Isaac task contract and numerical fixtures for CPU MuJoCo."""
import argparse
from pathlib import Path
import re
import sys

# Select the checkout containing this entry point, even if another PMT is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="ppofinetune_vision_teacher_stepping_stone_latent_anchor",
                    help="PGMT: pgmt_stage1 or PMT-PGMT-Stage1-G1-v0; default exports the vision student.")
parser.add_argument("--checkpoint", default="checkpoints/pretrained/ppoft_vision_bigmap_walkdance.pt")
parser.add_argument("--motion-file", type=Path, help="PGMT source motion NPZ or directory override.")
parser.add_argument("--output", default="logs/sim2sim/contract.pt")
parser.add_argument("--steps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1:
    parser.error("--steps must be positive")
app = AppLauncher(args).app

import gymnasium as gym
import torch
from tensordict import TensorDict

from pmt_tasks.builder import build_agent_cfg, build_env_cfg
from pmt_tasks.registry_gym import register_pmt_tasks
from pmt_tasks.runtime_options import run_and_close


def copy(value):
    return value.detach().cpu().clone()


def joint_parameter(value):
    """Isaac action terms allow scalar, joint-vector or per-env parameters."""
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.numel() == 1:
        return tensor.reshape(1).expand(29).clone()
    return tensor.reshape(-1, 29)[0].clone()


def main():
    register_pmt_tasks()
    pgmt = args.task in {"pgmt_stage1", "PMT-PGMT-Stage1-G1-v0"}
    if pgmt:
        from motion_tracking_rl.networks.pgmt_actor_critic import PGMTActorCritic as policy_class
        stem, task_id = "pgmt_stage1", "PMT-PGMT-Stage1-G1-v0"
        overrides = []
        if args.motion_file is not None:
            overrides.append(f"motion.motion_files={args.motion_file.resolve()}")
    else:
        from motion_tracking_rl.networks.vision_transformer_actor_critic import VisionTransformerActorCritic as policy_class
        stem = "ppofinetune_vision_teacher_stepping_stone_latent_anchor"
        if args.task not in {stem, "PMT-PPOFinetune-VisionTeacher-SteppingStone-G1-v0"}:
            raise ValueError(f"Unsupported standalone task: {args.task}")
        if args.motion_file is not None:
            raise ValueError("--motion-file is available for PGMT; vision requires paired motion configuration.")
        task_id = "PMT-PPOFinetune-VisionTeacher-SteppingStone-G1-v0"
        overrides = [f"network.base_policy_ckpt={args.checkpoint}"]
    cfg = build_env_cfg(stem, overrides=overrides)
    cfg.scene.num_envs = 1
    cfg.seed = 42
    if pgmt:
        # Export clean observations and the unrandomized training URDF parameters.
        # No training configuration is changed or saved back to the task.
        for group in vars(cfg.observations).values():
            if hasattr(group, "enable_corruption"):
                group.enable_corruption = False
        for name in list(vars(cfg.events)):
            if not name.startswith("_"):
                setattr(cfg.events, name, None)
    env = gym.make(task_id, cfg=cfg).unwrapped
    try:
        obs, _ = env.reset(seed=42)
        agent = build_agent_cfg(stem, overrides=overrides).to_dict()
        kwargs = dict(agent["policy"])
        kwargs.pop("class_name")
        if not pgmt:
            kwargs["base_policy_ckpt"] = None
        cpu_obs = TensorDict({k: copy(v) for k, v in obs.items()}, batch_size=[1])
        policy = policy_class(cpu_obs, agent["obs_groups"], 29, **kwargs)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
        policy.eval()
        robot = env.scene["robot"]
        action_term = env.action_manager.get_term("joint_pos")
        metadata = checkpoint.get("policy_metadata")
        if metadata is None and hasattr(policy, "get_checkpoint_metadata"):
            metadata = policy.get_checkpoint_metadata()
        if metadata is None:
            raise ValueError("Checkpoint/policy must provide observation schema metadata.")
        contract = {
            "schema_version": 2, "task": task_id, "policy_class": policy_class.__name__,
            "checkpoint": str(Path(args.checkpoint).resolve()), "policy_kwargs": kwargs,
            "obs_groups": agent["obs_groups"], "policy_metadata": metadata,
            "joint_names": list(robot.joint_names), "body_names": list(robot.body_names),
            "sim_dt": cfg.sim.dt, "decimation": cfg.decimation,
            "terrain_mesh": None if pgmt else str(Path(cfg.scene.terrain.mesh_path).resolve()),
            "urdf": str(Path(cfg.scene.robot.spawn.asset_path).resolve()),
            "tracked_bodies": list(cfg.commands.motion.body_names),
            "robot": {name: copy(getattr(robot.data, name)[0]) for name in (
                "default_joint_pos", "default_joint_vel", "joint_stiffness", "joint_damping",
                "joint_armature", "joint_effort_limits")},
            "fixtures": [],
        }
        if pgmt:
            from pmt_tasks.pgmt_runtime import get_pgmt_state
            command = env.command_manager.get_term("motion")
            files = [str(Path(path).resolve()) for path in cfg.commands.motion.motion_files]
            chunks = []
            for motion in command.data_store.motions:
                match = re.fullmatch(r"(.*)\[chunk_(\d+)\]", motion.source_file)
                source = match.group(1) if match else motion.source_file
                offset = int(match.group(2)) * int(command.cfg.chunk_length) if match else 0
                chunks.append({"motion_id": files.index(str(Path(source).resolve())),
                               "start": offset, "end": offset + int(motion.num_frames)})
            contract.update(motion_files=files, motion_chunks=chunks,
                            tracking_config=dict(get_pgmt_state(env).config),
                            raw_files=files, optimized_files=files,
                            chunk_length=int(command.cfg.chunk_length),
                            anchor_body=command.cfg.anchor_body,
                            action_scale=joint_parameter(action_term._scale),
                            action_offset=joint_parameter(action_term._offset),
                            initial_joint_target=copy(robot.data.joint_pos_target[0]),
                            action_contract="q_target = q_default + G1_ACTION_SCALE * action",
                            command_frames=8, history_frames=10,
                            export_settings={"observation_corruption": False, "randomization_events": False})
        else:
            contract.update(raw_files=list(cfg.commands.student_motion.motion_files),
                            optimized_files=list(cfg.commands.motion.motion_files))
        previous_command = None
        with torch.inference_mode():
            for _ in range(args.steps):
                cpu_obs = TensorDict({k: copy(v) for k, v in obs.items()}, batch_size=[1])
                action = policy.act_inference(cpu_obs)
                command = env.command_manager.get_term("motion" if pgmt else "student_motion")
                command_id, frame = int(command.motion_ids[0]), int(command.frame_ids[0])
                if pgmt and previous_command is not None and (command_id != previous_command[0] or frame != previous_command[1] + 1):
                    raise RuntimeError("PGMT fixture crossed a command reset; reduce --steps.")
                previous_command = (command_id, frame)
                fixture = {
                    "obs": dict(cpu_obs), "action": copy(action),
                    "motion_id": command_id, "frame": frame,
                    "state": {name: copy(getattr(robot.data, name)[0]) for name in (
                        "root_pos_w", "root_quat_w", "root_link_lin_vel_w", "root_ang_vel_w",
                        "root_lin_vel_b", "root_ang_vel_b", "joint_pos", "joint_vel",
                        "body_pos_w", "body_quat_w")},
                    "previous_action": copy(env.action_manager.action[0]),
                    "action_offset": joint_parameter(action_term._offset),
                }
                if pgmt:
                    chunk = contract["motion_chunks"][command_id]
                    fixture.update(motion_id=chunk["motion_id"], frame=chunk["start"] + frame,
                                   reference_start=chunk["start"], reference_end=chunk["end"],
                                   env_origin=copy(env.scene.env_origins[0]),
                                   previous_target=copy(robot.data.joint_pos_target[0]),
                                   action_target=contract["action_offset"] + contract["action_scale"] * action[0])
                else:
                    fixture["hits_w"] = copy(env.scene.sensors["height_scanner"].data.ray_hits_w[0])
                contract["fixtures"].append(fixture)
                obs, _, terminated, truncated, _ = env.step(action.to(env.device))
                if bool((terminated | truncated).any()):
                    raise RuntimeError("Golden fixture rollout reset; reduce steps or investigate policy.")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(contract, output)
        print(f"[CONTRACT] saved {len(contract['fixtures'])} Isaac fixtures to {output}", flush=True)
    finally:
        env.close()


run_and_close(main, app)
