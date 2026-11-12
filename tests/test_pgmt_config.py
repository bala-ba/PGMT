"""PGMT's declared task keeps the confirmed native 2B training configuration."""
from omegaconf import OmegaConf

from pmt_tasks.builder import build_task_config
from pmt_tasks.pgmt_math import REWARD_WEIGHTS, TRACKING_CONFIG
from pmt_tasks.registry_gym import gym_id_for_stem, task_stem_for_id


def test_pgmt_baseline_axes_and_reward_contract(monkeypatch):
    monkeypatch.setenv("PMT_PGMT_MOTION_ROOT", "/tmp/example-lafan")
    native = build_task_config("multimotionv2_flat")
    task = build_task_config("pgmt_stage1")
    for axis in ("robot", "terrain", "scene", "sensor", "algorithm", "stage", "runner"):
        assert task[axis] == native[axis]
    for key, value in native.motion.items():
        if key != "motion_files":
            assert task.motion[key] == value
    assert task.motion.motion_files == "/tmp/example-lafan"
    assert task.network.name == "PGMTActorCritic"
    assert task.reward.recovery_upward_velocity is False
    assert dict(task.reward_weights) == REWARD_WEIGHTS
    assert dict(task.reward.tracking) == TRACKING_CONFIG
    assert list(task.reward.groups) == ["upper", "lower", "auxiliary"]
    assert "${" not in OmegaConf.to_yaml(task, resolve=True)


def test_pgmt_registry_and_observation_groups():
    assert gym_id_for_stem("pgmt_stage1") == "PMT-PGMT-Stage1-G1-v0"
    assert task_stem_for_id("PMT-PGMT-Stage1-G1-v0") == "pgmt_stage1"
    cfg = build_task_config("pgmt_stage1")
    assert cfg.obs_groups == {"policy": ["proprio"], "policy_history": ["proprio_history"],
                             "command_window": ["command_window"], "critic": ["critic"]}
    assert cfg.obs.history_length == 10
    assert cfg.obs.command_frames == 8


def test_pgmt_perception_task_keeps_three_rewards_and_declares_actual_scan():
    stage1, stage2 = build_task_config('pgmt_stage1'), build_task_config('pgmt_stage2')
    assert stage2.reward == stage1.reward
    assert stage2.obs_groups.critic == stage1.obs_groups.critic
    assert list(stage2.sensor.grid_size) == [2., 2.]
    assert stage2.sensor.grid_resolution == .1
    assert stage2.network.terrain_mode == 'enabled'
    assert stage2.motion.fixed_start_frame == -1 and not stage2.motion.enable_frame_sampling
    assert stage2.motion.sampler == "emd"
    assert stage2.motion.emd.evaluation_interval == 9_600_000
    assert stage2.motion.chunk_length == 0
    assert stage2.algorithm.learning_rate == 1e-4 and stage1.algorithm.learning_rate == 1e-3
    assert task_stem_for_id(gym_id_for_stem('pgmt_stage2')) == 'pgmt_stage2'
