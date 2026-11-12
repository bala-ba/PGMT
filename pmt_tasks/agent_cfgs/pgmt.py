"""The native 2B PPO/runner configuration with the PGMT network substituted."""
from isaaclab.utils import configclass

from pmt_tasks.agent_cfgs.multi_motion_ppo import G1FlatMultiMotionPPORunnerCfg
from pmt_tasks.isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg


@configclass
class PGMTActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "PGMTActorCritic"
    embedding_dim: int = 128
    num_heads: int = 4
    command_frames: int = 8
    history_frames: int = 10
    rope_base: float = 10000.0


@configclass
class PGMTStage1PPORunnerCfg(G1FlatMultiMotionPPORunnerCfg):
    experiment_name = "pgmt_stage1"
    obs_groups = {"policy": ["proprio"], "policy_history": ["proprio_history"],
                  "command_window": ["command_window"], "critic": ["critic"]}
    policy = PGMTActorCriticCfg(
        init_noise_std=1.0, actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128], activation="elu",
        actor_obs_normalization=True, critic_obs_normalization=True,
    )


@configclass
class PGMTPerceptionActorCriticCfg(PGMTActorCriticCfg):
    terrain_glimpse_cfg: dict = {}
    terrain_mode: str = "enabled"
    terrain_observation_cfg: dict | None = None


@configclass
class PGMTStage2PPORunnerCfg(PGMTStage1PPORunnerCfg):
    class_name = "PGMTStage2Runner"
    experiment_name = "pgmt_stage2_perception"
    pgmt_training_contract: dict | None = None  # Bound to the actual env by the entry point.
    obs_groups = {"policy": ["proprio"], "policy_history": ["proprio_history"],
                  "command_window": ["command_window"], "critic": ["critic"],
                  "terrain_map": ["terrain_map"], "terrain_valid": ["terrain_valid"]}
    policy = PGMTPerceptionActorCriticCfg(
        init_noise_std=1.0, actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128], activation="elu",
        actor_obs_normalization=True, critic_obs_normalization=True,
    )

    def __post_init__(self):
        self.algorithm.learning_rate = 1e-4


@configclass
class PGMTTerrainActorCriticCfg(PGMTPerceptionActorCriticCfg):
    value_groups: tuple[str, ...] = ("upper", "lower", "terrain", "auxiliary")


@configclass
class PGMTTerrainPPORunnerCfg(PGMTStage2PPORunnerCfg):
    experiment_name = "pgmt_stage2_terrain"
    policy = PGMTTerrainActorCriticCfg(
        init_noise_std=1.0, actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128], activation="elu",
        actor_obs_normalization=True, critic_obs_normalization=True,
    )
