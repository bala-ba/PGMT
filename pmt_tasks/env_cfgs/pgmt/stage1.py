"""PGMT observations/rewards, with MultiMotionV2-Flat physics and lifecycle."""
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from pmt_tasks.env_cfgs.multi_motion_flat import PMTMultiMotionFlatEnvCfg
from pmt_tasks.mdp.pgmt_observations import pgmt_proprio, pgmt_command_window, pgmt_critic
from pmt_tasks.mdp.pgmt_rewards import pgmt_reward
from pmt_tasks.pgmt_math import REWARD_WEIGHTS, TRACKING_CONFIG, VALUE_GROUPS


@configclass
class PGMTObservationsCfg:
    @configclass
    class ProprioCfg(ObsGroup):
        proprio = ObsTerm(func=pgmt_proprio)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class HistoryCfg(ObsGroup):
        proprio = ObsTerm(func=pgmt_proprio)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = False

    @configclass
    class CommandCfg(ObsGroup):
        command = ObsTerm(func=pgmt_command_window)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        privileged = ObsTerm(func=pgmt_critic)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    proprio = ProprioCfg()
    proprio_history = HistoryCfg()
    command_window = CommandCfg()
    critic = CriticCfg()


# These become regular named RewardManager terms, so each receives native
# per-term episode logging and exactly one time integration. No recovery term:
# PMT's native resets do not carry PGMT's fallen-pose-pool mode.
PGMTRewardsCfg = configclass(type("PGMTRewardsCfg", (), {
    "__module__": __name__,
    **{name: RewTerm(func=pgmt_reward, weight=weight, params={"name": name})
       for name, weight in REWARD_WEIGHTS.items()},
}))


@configclass
class PGMTStage1EnvCfg(PMTMultiMotionFlatEnvCfg):
    observations: PGMTObservationsCfg = PGMTObservationsCfg()
    rewards: PGMTRewardsCfg = PGMTRewardsCfg()
    pgmt_reward_groups: tuple[str, ...] = VALUE_GROUPS
    pgmt_tracking: dict = TRACKING_CONFIG.copy()
    pgmt_recovery_enabled: bool = False
