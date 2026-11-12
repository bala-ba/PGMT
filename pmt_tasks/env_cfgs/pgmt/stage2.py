"""PGMT Stage2 perception and P3 contact state; rewards retain the three P1 groups."""
from pathlib import Path
import json

from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg

from pmt_tasks.env_cfgs.pgmt.placement import PGMTPlacementEnvCfg
from pmt_tasks.env_cfgs.pgmt.stage1 import PGMTObservationsCfg
from pmt_tasks.mdp.pgmt_observations import pgmt_terrain_observation
from pmt_tasks.pgmt_perception import terrain_observation_contract, validate_terrain_observation_contract
from pmt_tasks.pgmt_contact_reference import CONTACT_FOOT_NAMES
from pmt_tasks.pgmt_contact_state import CONTACT_STATE_CONFIG, TERRAIN_FOOT_SENSORS, terrain_contact_filters
from pmt_tasks.pgmt_touchdown import TOUCHDOWN_QUALITY_CONFIG
from pmt_tasks.pgmt_stumble import STUMBLE_CONFIG
from pmt_tasks.pgmt_tracking_relaxation import tracking_relaxation_config
from pmt_tasks.pgmt_terminations import pgmt_time_out, pgmt_root_xy_failure, pgmt_torso_height_failure, pgmt_wrist_height_failure
from pmt_tasks.mdp.pgmt_terminations import SustainedTrackingFailure


@configclass
class PGMTPMTPerceptionObservationsCfg(PGMTObservationsCfg):
    @configclass
    class TerrainCfg(ObsGroup):
        # Single call generates both height and mask, exactly as PMT vision.
        height_and_valid = ObsTerm(func=pgmt_terrain_observation)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    pgmt_terrain = TerrainCfg()


@configclass
class PGMTPerceptionEnvCfg(PGMTPlacementEnvCfg):
    observations: PGMTPMTPerceptionObservationsCfg = PGMTPMTPerceptionObservationsCfg()
    pgmt_terrain_asset: str = str(Path(__file__).resolve().parents[3] / "assets/terrain/pgmt_stage2_v8")
    pgmt_placement_xy_mode: str = "center"
    pgmt_emd_sampling: dict = {}  # Validated P-BFM defaults, configurable through motion.emd.
    pmt_motion_paths: list[str] = [str(Path(__file__).resolve().parents[3] / "assets/motions/pgmt_clipped_50hz_tcrs_v1")]
    # PMT defaults: 10% dropout, uniform +/-6 cm, clip +/-3 m; divide in NETWORK.
    pgmt_terrain_observation: dict = terrain_observation_contract()
    # Shared state for P2 diagnostics and the formal four-head terrain task.
    pgmt_contact_events: dict = CONTACT_STATE_CONFIG.copy()
    # T1 is calculated/cached here; the four-head subclass activates its reward.
    pgmt_touchdown_quality: dict = TOUCHDOWN_QUALITY_CONFIG.copy()
    # T2 uses the all-object contact stream, not the terrain-support stream.
    pgmt_reference_contact_match: bool = True
    # T3 uses the same contact stream and world-XY ankle-link origin velocity.
    pgmt_slip: bool = True
    # T4 counts excessive world-XY all-object force, independent of reference labels.
    pgmt_stumble: dict = STUMBLE_CONFIG.copy()
    # T5 consumes shared fast-contact-switch events; no separate timer.
    pgmt_contact_switching: bool = True
    # T6 sums squared all-object foot forces, including values below 10 N.
    pgmt_contact_force: bool = True
    # P2 is opt-in; the formal four-head subclass enables relaxation by default.
    pgmt_tracking_relaxation: dict = tracking_relaxation_config()
    # Only wrist acceleration mismatch: feet may adapt their motion to terrain.
    # Serialized with the environment contract for strict checkpoint resume.
    pgmt_ee_acceleration_body_names: tuple[str, ...] = ("left_wrist_yaw_link", "right_wrist_yaw_link")

    def __post_init__(self):
        super().__post_init__()
        from pmt_tasks.pgmt_emd_sampling import emd_config
        self.pgmt_emd_sampling = emd_config(self.pgmt_emd_sampling)
        self.commands.motion.sampler_type = "emd"
        self.commands.motion.emd_sampling = self.pgmt_emd_sampling.copy()
        self.commands.motion.fixed_start_frame = -1
        # Stage2 only. Stage1 and historical P1 placement configurations retain
        # their original termination rules. Keep the native orientation predicate.
        self.episode_length_s = 10.0
        # Accept completed clips within two control intervals of the hard cap;
        # never truncate an unfinished reference early just to count a timeout.
        self.terminations.time_out = TerminationTermCfg(
            func=pgmt_time_out, time_out=True,
            params={"command_name": "motion", "boundary_tolerance_steps": 2})
        self.terminations.root_xy = TerminationTermCfg(
            func=pgmt_root_xy_failure, params={"command_name": "motion", "threshold": 1.0, "root_body": "pelvis"})
        self.terminations.anchor_pos = TerminationTermCfg(
            func=pgmt_torso_height_failure,
            params={"command_name": "motion", "threshold": 0.4, "torso_body": "torso_link"})
        self.terminations.ee_body_pos = TerminationTermCfg(
            func=pgmt_wrist_height_failure, params={"command_name": "motion", "threshold": 0.4,
                "root_body": "pelvis", "body_names": ["left_wrist_yaw_link", "right_wrist_yaw_link"]})
        # Each term owns its timer; native TerminationManager resets it per env.
        # Avoid nesting the orientation wrapper on repeated __post_init__ calls.
        for name in ("root_xy", "anchor_pos", "ee_body_pos", "anchor_ori"):
            term = getattr(self.terminations, name)
            if term.func is not SustainedTrackingFailure:
                setattr(self.terminations, name, TerminationTermCfg(func=SustainedTrackingFailure, params={
                    "predicate": term.func, "predicate_params": term.params, "duration_s": 0.2}))
        validate_terrain_observation_contract(self.pgmt_terrain_observation)
        self.pgmt_tracking_relaxation = tracking_relaxation_config(self.pgmt_tracking_relaxation)
        manifest = json.loads((Path(self.pgmt_terrain_asset) / "manifest.json").read_text())
        self.commands.motion.placement_patch_grid = tuple(manifest["spec"].get("spawn_patch_grid", (2, 2)))
        self.commands.motion.placement_xy_mode = self.pgmt_placement_xy_mode
        self.commands.motion.max_placement_attempts = 128
        self.scene.contact_forces.debug_vis = False
        self.sim.physx.gpu_collision_stack_size = 2**28
        filters = terrain_contact_filters(self.pgmt_terrain_asset, self.scene.terrain.prim_path)
        for sensor_name, foot_name in zip(TERRAIN_FOOT_SENSORS, CONTACT_FOOT_NAMES):
            setattr(self.scene, sensor_name, ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{foot_name}",
                filter_prim_paths_expr=filters.copy(),
                history_length=0, track_air_time=False, debug_vis=False,
            ))
