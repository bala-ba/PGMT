"""PGMT placement, curriculum and terrain scan; stage-2 fusion/rewards follow."""
from pathlib import Path

from isaaclab.managers import CurriculumTermCfg, TerminationTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from pmt_tasks.env_cfgs.pgmt.stage1 import PGMTStage1EnvCfg, PGMTObservationsCfg
from pmt_tasks.mdp.pgmt_observations import pgmt_terrain_map, pgmt_terrain_valid
from pmt_tasks.pgmt_height_scan import SCAN_SIZE, SCAN_RESOLUTION
from pmt_tasks.pgmt_height_scanner import PGMTHeightScanner
from pmt_tasks.mdp.commands.pgmt_motion_command import (
    PGMTTerrainMotionCommandCfg, pgmt_motion_finished, pgmt_map_out_of_bounds,
)
from pmt_tasks.pgmt_terrain_importer import PGMTTerrainImporterCfg
from pmt_tasks.pgmt_curriculum import pgmt_terrain_progress


@configclass
class PGMTTerrainCurriculumCfg:
    terrain_progress = CurriculumTermCfg(func=pgmt_terrain_progress)


@configclass
class PGMTTerrainObservationsCfg(PGMTObservationsCfg):
    @configclass
    class TerrainMapCfg(ObsGroup):
        height = ObsTerm(func=pgmt_terrain_map)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class TerrainValidCfg(ObsGroup):
        valid = ObsTerm(func=pgmt_terrain_valid)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    terrain_map = TerrainMapCfg()
    terrain_valid = TerrainValidCfg()


@configclass
class PGMTPlacementEnvCfg(PGMTStage1EnvCfg):
    observations: PGMTTerrainObservationsCfg = PGMTTerrainObservationsCfg()
    curriculum: PGMTTerrainCurriculumCfg = PGMTTerrainCurriculumCfg()
    pmt_sampler_type: str = "adaptive"
    pgmt_terrain_asset: str = str(Path(__file__).resolve().parents[3] / "assets/terrain/pgmt_stage2_v5")

    def __post_init__(self):
        super().__post_init__()
        material = self.scene.terrain.physics_material
        self.scene.terrain = PGMTTerrainImporterCfg(
            prim_path="/World/ground", asset_directory=self.pgmt_terrain_asset,
            collision_group=-1, physics_material=material, max_init_terrain_level=1,
            debug_vis=False,
        )
        self.scene.height_scanner = RayCasterCfg(
            class_type=PGMTHeightScanner,
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(
                resolution=SCAN_RESOLUTION, size=SCAN_SIZE, ordering="xy"),
            mesh_prim_paths=[self.scene.terrain.prim_path],
            max_distance=60.0, update_period=self.decimation * self.sim.dt,
            debug_vis=False,
        )
        base = self.commands.motion
        self.commands.motion = PGMTTerrainMotionCommandCfg(
            asset_name="robot", anchor_body="torso_link", body_names=base.body_names,
            motion_files=base.motion_files, debug_vis=False,
            adaptive_beta=base.adaptive_beta, adaptive_alpha=base.adaptive_alpha,
            adaptive_uniform_ratio=base.adaptive_uniform_ratio,
            adaptive_update_interval=base.adaptive_update_interval,
            adaptive_kernel_size=base.adaptive_kernel_size, adaptive_kernel_lambda=base.adaptive_kernel_lambda,
        )
        self.sim.physics_material = material
        self.terminations.motion_finished = TerminationTermCfg(func=pgmt_motion_finished, time_out=True)
        self.terminations.map_out_of_bounds = TerminationTermCfg(func=pgmt_map_out_of_bounds, time_out=True)
