"""Formal four-value PGMT terrain task; recovery remains disabled."""
from isaaclab.utils import configclass

from motion_tracking_rl.pgmt_value_contract import TERRAIN_VALUE_GROUPS
from pmt_tasks.env_cfgs.pgmt.stage2 import PGMTPerceptionEnvCfg
from pmt_tasks.env_cfgs.pgmt.terrain_rewards import PGMTContactForceRewardsCfg
from pmt_tasks.pgmt_tracking_relaxation import tracking_relaxation_config


@configclass
class PGMTTerrainTrainingEnvCfg(PGMTPerceptionEnvCfg):
    rewards: PGMTContactForceRewardsCfg = PGMTContactForceRewardsCfg()
    pgmt_reward_groups: tuple[str, ...] = TERRAIN_VALUE_GROUPS
    pgmt_tracking_relaxation: dict = tracking_relaxation_config({"enabled": True})
