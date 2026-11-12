"""Implemented P3 terms, staged independently of the P2 three-head task.

The final subclass is used by the formal four-head terrain task as well as
native RewardManager acceptance. The original three-head P2 task is separate.
"""
from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from pmt_tasks.env_cfgs.pgmt.stage1 import PGMTRewardsCfg
from pmt_tasks.mdp.pgmt_rewards import pgmt_reward
from pmt_tasks.pgmt_touchdown import TOUCHDOWN_QUALITY_WEIGHT
from pmt_tasks.pgmt_contact_match import REFERENCE_CONTACT_MATCH_WEIGHT
from pmt_tasks.pgmt_slip import SLIP_WEIGHT
from pmt_tasks.pgmt_stumble import STUMBLE_WEIGHT
from pmt_tasks.pgmt_contact_switching import CONTACT_SWITCHING_WEIGHT
from pmt_tasks.pgmt_contact_force import CONTACT_FORCE_WEIGHT


@configclass
class PGMTTouchdownRewardsCfg(PGMTRewardsCfg):
    terrain_touchdown_quality = RewardTermCfg(
        func=pgmt_reward, weight=TOUCHDOWN_QUALITY_WEIGHT, params={"name": "terrain_touchdown_quality"})


@configclass
class PGMTContactMatchRewardsCfg(PGMTTouchdownRewardsCfg):
    """Independent acceptance of both implemented terms (21 + T1 + T2)."""
    terrain_reference_contact_match = RewardTermCfg(
        func=pgmt_reward, weight=REFERENCE_CONTACT_MATCH_WEIGHT, params={"name": "terrain_reference_contact_match"})


@configclass
class PGMTSlipRewardsCfg(PGMTContactMatchRewardsCfg):
    """Independent acceptance of 21 original terms plus T1/T2/T3."""
    terrain_slip = RewardTermCfg(func=pgmt_reward, weight=SLIP_WEIGHT, params={"name": "terrain_slip"})


@configclass
class PGMTStumbleRewardsCfg(PGMTSlipRewardsCfg):
    """Independent acceptance of 21 original terms plus T1/T2/T3/T4."""
    terrain_stumble = RewardTermCfg(func=pgmt_reward, weight=STUMBLE_WEIGHT, params={"name": "terrain_stumble"})


@configclass
class PGMTContactSwitchingRewardsCfg(PGMTStumbleRewardsCfg):
    """Independent acceptance of 21 original terms plus T1 through T5."""
    terrain_contact_switching = RewardTermCfg(
        func=pgmt_reward, weight=CONTACT_SWITCHING_WEIGHT, params={"name": "terrain_contact_switching"})


@configclass
class PGMTContactForceRewardsCfg(PGMTContactSwitchingRewardsCfg):
    """Independent acceptance of all six terrain terms: 27 terms, four groups."""
    terrain_contact_force = RewardTermCfg(
        func=pgmt_reward, weight=CONTACT_FORCE_WEIGHT, params={"name": "terrain_contact_force"})
