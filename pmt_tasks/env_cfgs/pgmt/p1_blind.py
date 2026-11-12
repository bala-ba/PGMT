"""P1 acceptance: terrain training with the exact stage-1 policy input schema."""
from isaaclab.utils import configclass

from pmt_tasks.env_cfgs.pgmt.placement import PGMTPlacementEnvCfg
from pmt_tasks.env_cfgs.pgmt.stage1 import PGMTObservationsCfg


@configclass
class PGMTBlindTerrainEnvCfg(PGMTPlacementEnvCfg):
    observations: PGMTObservationsCfg = PGMTObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # P1 exercises terrain geometry, motion placement and curriculum, while
        # keeping checkpoint inputs identical. P2's scanner remains available
        # through PGMTPlacementEnvCfg, but is not injected into this policy.
        self.scene.height_scanner = None
