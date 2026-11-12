"""Independent deadzone oracles, real tile layouts, and pre-reset reward caches."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pmt_tasks.pgmt_math import TRACKING_CONFIG, body_groups, reward_terms
from pmt_tasks.pgmt_runtime import PGMTState
from pmt_tasks.pgmt_terrain_layout import GROUP_IDS, training_manifest
from pmt_tasks.pgmt_terrain_query import TerrainTileMap
from pmt_tasks.pgmt_tracking_relaxation import (
    RELAXATION_TERMS, TrackingRelaxation, relax_tracking_error, tracking_relaxation_config,
)
from scripts.pmt_npz_to_mjlab import BFS_BODIES, BFS_JOINTS

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
ROOT = Path(__file__).resolve().parents[1]


def layout(version, device):
    manifest = training_manifest(json.loads((ROOT / f"assets/terrain/pgmt_stage2_{version}/manifest.json").read_text()))
    bounds = torch.empty(manifest["spec"]["num_rows"], manifest["spec"]["num_cols"], 2, 2, device=device)
    for t in manifest["tiles"]:
        if version == "v6":
            bounds[t["level"], t["family_index"]] = bounds.new_tensor(t["bounds"])
        else:
            bounds[t["level"], t["family_index"]] = (bounds.new_tensor(t["geometry"]["bounds"])[:, :2]
                + bounds.new_tensor(t["center"][:2]) - bounds.new_tensor(manifest["spec"]["size"]) / 2)
    return TerrainTileMap(manifest, bounds), manifest


def model(device="cpu", config=None, names=None):
    bodies, joints = (BFS_BODIES, BFS_JOINTS) if names is None else names
    return TrackingRelaxation({"enabled": True, **(config or {})}, bodies, joints,
                              body_groups(bodies, joints, device=device), device=device, dtype=torch.float64)


def sample(n, device):
    groups = body_groups(BFS_BODIES, BFS_JOINTS, device=device)
    pos = torch.zeros(n, 30, 3, dtype=torch.float64, device=device)
    pos[..., 2] = 1.
    quat = torch.zeros(n, 30, 4, dtype=torch.float64, device=device)
    quat[..., 0] = 1.
    ref = dict(body_pos=pos.clone(), body_quat=quat.clone(), body_lin_vel=torch.zeros_like(pos),
               body_ang_vel=torch.zeros_like(pos), joint_pos=pos.new_zeros(n, 29), joint_vel=pos.new_zeros(n, 29))
    state = {key: value.clone() for key, value in ref.items()}
    errors = pos.new_tensor([0., .02, .05, .1, .15, .2, .25, .4, .5, .7, .8, 1.2])
    state["body_pos"][:, groups["lower"], 0] = errors
    state["body_quat"][:, groups["lower"], 0] = torch.cos(errors / 2)
    state["body_quat"][:, groups["lower"], 1] = torch.sin(errors / 2)
    state["joint_pos"][:, groups["lower_joints"]] = errors * pos.new_tensor([1., -1.] * 6)
    target = pos.new_zeros(n, 29)
    forces = torch.ones_like(pos) * 30.
    limits = pos.new_tensor([-3., 3.]).expand(n, 29, 2)
    def rates(**kwargs):
        return reward_terms(state, ref, target, target, ref["body_lin_vel"], ref["body_lin_vel"], forces,
                            limits, groups, anchor=BFS_BODIES.index("torso_link"), dt=.02, **kwargs)
    return state, ref, groups, target, limits, errors, rates


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("version", ["v5", "v6"])
def test_all_families_levels_and_rewards_use_semantic_current_tile(version, device):
    tiles, manifest = layout(version, device)
    points = torch.tensor([t["center"][:2] for t in manifest["tiles"]], device=device)
    located = tiles.locate(points)
    relax = model(device)
    margins, scale, applicable = relax.margins(located)
    state, ref, _, _, _, errors, rates = sample(len(points), device)
    saved_state = {k: v.clone() for k, v in state.items()}
    saved_ref = {k: v.clone() for k, v in ref.items()}
    strict = rates()
    debug = {}
    relaxed = rates(tracking_margins=margins, tracking_debug=debug)
    expected_scale = scale.new_tensor([t["level"] / 9 if t["family_group_index"] in (1, 2, 3) else 0.
                                       for t in manifest["tiles"]])
    torch.testing.assert_close(scale, expected_scale)
    for term, maximum in RELAXATION_TERMS.items():
        expected_margin = (expected_scale * relax.config[maximum])[:, None].expand(-1, 12)
        expected_error = torch.maximum(errors[None] - expected_margin, torch.zeros_like(expected_margin))
        sigma = TRACKING_CONFIG[term.removeprefix("lower_") + "_sigma"]
        torch.testing.assert_close(margins[term], expected_margin)
        torch.testing.assert_close(debug[f"{term}_raw_error"], errors[None].expand_as(expected_margin))
        torch.testing.assert_close(debug[f"{term}_relaxed_error"], expected_error)
        torch.testing.assert_close(relaxed[term], torch.exp(-expected_error.square().mean(-1) / sigma**2))
        assert (relaxed[term] >= strict[term] - 1e-14).all()
        assert (debug[f"{term}_relaxed_error"] >= 0).all()
        torch.testing.assert_close(relaxed[term][scale == 0], strict[term][scale == 0], rtol=0, atol=0)
    for name in strict.keys() - RELAXATION_TERMS.keys():
        torch.testing.assert_close(strict[name], relaxed[name], rtol=0, atol=0)
    assert len(strict) == len(relaxed) == 21
    assert not applicable[located["family_group_index"] == GROUP_IDS["rough"]].any()
    for key in state:
        torch.testing.assert_close(state[key], saved_state[key], rtol=0, atol=0)
        torch.testing.assert_close(ref[key], saved_ref[key], rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_configurable_families_caps_alpha_and_name_order(device):
    bodies, joints = list(reversed(BFS_BODIES)), list(reversed(BFS_JOINTS))
    cfg = dict(terrain_families=["flat", "rough"], position_max=.3, orientation_max=.2, joint_max=.1,
               alpha={"lower_body_position": {"left_knee_link": 0., "right_knee_link": 2.}})
    relax = model(device, cfg, (bodies, joints))
    tiles = dict(tile_valid=torch.ones(5, dtype=torch.bool, device=device),
                 tile_level=torch.tensor([9, 9, 9, 9, 9], device=device),
                 family_group_index=torch.arange(5, device=device))
    margins, scale, _ = relax.margins(tiles)
    assert scale.tolist() == [1., 0., 0., 0., 1.]
    i = relax.names["lower_body_position"].index("left_knee_link")
    j = relax.names["lower_body_position"].index("right_knee_link")
    assert margins["lower_body_position"][:, i].eq(0).all()
    assert margins["lower_body_position"][0, j].item() == pytest.approx(.6)
    assert margins["lower_body_orientation"][4].tolist() == [.2] * 12
    assert margins["lower_joint_position"][0].tolist() == [.1] * 12
    cfg["alpha"]["lower_body_position"]["right_knee_link"] = 90.
    assert relax.contract()["config"]["alpha"]["lower_body_position"]["right_knee_link"] == 2.
    contract = relax.contract()
    contract["names"]["lower_body_position"].clear()
    assert len(relax.names["lower_body_position"]) == 12


@pytest.mark.parametrize("device", DEVICES)
def test_boundaries_exact_zero_saturation_invalid_tiles_and_nonfinite_error(device):
    margin = torch.tensor([[.2] * 5], device=device, dtype=torch.float64)
    error = torch.tensor([[0., .1, .2, .3, float("nan")]], device=device, dtype=torch.float64)
    actual = relax_tracking_error(error, margin)
    torch.testing.assert_close(actual, margin.new_tensor([[0., 0., 0., .1, float("nan")]]), equal_nan=True)
    assert torch.isinf(relax_tracking_error(margin.new_tensor([[float("inf")]]), margin[:, :1])).all()
    tiles = dict(tile_valid=torch.tensor([True, True, False, True], device=device),
                 tile_level=torch.tensor([9, 100, -1, 9], device=device),
                 family_group_index=torch.tensor([2, 2, -1, -1], device=device))
    margins, scale, _ = model(device).margins(tiles)
    assert scale.tolist() == [1., 1., 0., 0.]
    for value in margins.values():
        assert torch.isfinite(value).all() and value[2:].eq(0).all()
    for config in ({"terrain_families": []}, {"position_max": 0., "orientation_max": 0., "joint_max": 0.}):
        assert all(value.eq(0).all() for value in model(device, config).margins(tiles)[0].values())
    disabled = TrackingRelaxation({"enabled": False}, BFS_BODIES, BFS_JOINTS,
                                 body_groups(BFS_BODIES, BFS_JOINTS), device=device)
    assert all(value.eq(0).all() for value in disabled.margins(tiles)[0].values())


@pytest.mark.parametrize("bad", [
    {"enabled": 1}, {"position_max": -1}, {"joint_max": float("nan")}, {"orientation_max": float("inf")},
    {"level_max": 0}, {"level_max": 9.5}, {"terrain_families": "stairs"},
    {"terrain_families": ["stairs", "stairs"]}, {"terrain_families": ["up_stairs"]},
    {"alpha": {"lower_body_orientation": {"left_knee_link": -1.}}},
    {"alpha": {"lower_body_orientation": {"left_knee_link": float("inf")}}},
    {"alpha": {"lower_joint_position": {"typo_joint": 1.}}},
    {"alpha": {"lower_joint_position": {"waist_yaw_joint": 1.}}},
    {"alpha": {"upper_body_position": {}}}, {"positon_max": .2},
    {"position_max": float("inf")}, {"position_max": 1e300, "alpha": {"lower_body_position": {"left_knee_link": 1e300}}},
])
def test_bad_config_fails_before_training(bad):
    with pytest.raises((ValueError, RuntimeError), match="relaxation"):
        model(config=bad)


def test_effective_caps_must_fit_runtime_float32():
    with pytest.raises(ValueError, match="effective caps"):
        TrackingRelaxation({"position_max": 1e300}, BFS_BODIES, BFS_JOINTS,
                           body_groups(BFS_BODIES, BFS_JOINTS), device="cpu")


@pytest.mark.parametrize("device", DEVICES)
def test_patch_and_region_crossing_out_of_map_and_terminal_reset_cache(device):
    tiles, _ = layout("v6", device)
    physical, ref, groups, target, limits, _, _ = sample(2, device)
    class Scene(dict):
        pass
    scene = Scene(contact_forces=SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros_like(physical["body_pos"]))))
    calls = []
    def locate(xy):
        calls.append(xy.clone())
        return tiles.locate(xy)
    scene.terrain = SimpleNamespace(tile_map=SimpleNamespace(locate=locate),
        terrain_levels=torch.zeros(2, dtype=torch.long, device=device),
        terrain_types=torch.zeros(2, dtype=torch.long, device=device))
    runtime = PGMTState.__new__(PGMTState)
    runtime.env = SimpleNamespace(scene=scene, device=device, num_envs=2, step_dt=.02, common_step_counter=1,
                                  cfg=SimpleNamespace(pgmt_tracking_relaxation={"enabled": True}))
    runtime.robot = SimpleNamespace(body_names=BFS_BODIES, joint_names=BFS_JOINTS,
        data=SimpleNamespace(joint_pos=physical["joint_pos"], joint_pos_target=target, soft_joint_pos_limits=limits))
    runtime.anchor, runtime.groups, runtime.config = BFS_BODIES.index("torso_link"), groups, TRACKING_CONFIG.copy()
    runtime.command = SimpleNamespace(motion_ids=torch.tensor([1, 2], device=device),
                                     frame_ids=torch.tensor([5, 8], device=device))
    runtime.robot_state, runtime.reference = lambda: physical, lambda: ref
    runtime.previous_target = target.clone()
    runtime.previous_velocity, runtime.previous_ref_velocity = physical["body_lin_vel"].clone(), ref["body_lin_vel"].clone()
    runtime.pending_reset = torch.zeros(2, dtype=torch.bool, device=device)
    runtime._reward_step, runtime._obs_version = None, 0
    runtime.contact_order = torch.arange(30, device=device)
    runtime._configure_tracking_relaxation()
    assert not runtime.tracking_relaxation_state()["sample_valid"].any()
    with pytest.raises(RuntimeError, match="this control step"):
        runtime.tracking_relaxation_state(for_reward=True)
    def move(row, col, offset=(2., 2.)):
        destination = tiles.lower + tiles.size * physical["body_pos"].new_tensor([row, col])
        destination += destination.new_tensor(offset)
        delta = destination - physical["body_pos"][:, runtime.anchor, :2]
        physical["body_pos"][..., :2] += delta[:, None]
    move(9, 1)  # Actual stairs L9; birth remains flat L0.
    first = runtime.rewards()
    before = runtime.tracking_relaxation_state(for_reward=True)
    assert before["tile_level"].tolist() == [9, 9] and before["birth_level"].tolist() == [0, 0]
    assert before["family_group_index"].tolist() == [2, 2]
    before["lower_body_position_margin"].zero_()
    assert runtime.tracking_relaxation_state()["lower_body_position_margin"].eq(.2).all()
    move(9, 1, (8., 2.))  # Next 6 m patch in the same region, unchanged margins.
    assert runtime.rewards() is first and len(calls) == 1
    runtime.env.common_step_counter += 1
    runtime.rewards()
    assert runtime.tracking_relaxation_state()["lower_body_position_margin"].eq(.2).all()
    move(3, 2)  # Cross to slope L3, immediate new level on the next settled step.
    runtime.env.common_step_counter += 1
    runtime.rewards()
    frozen = runtime.tracking_relaxation_state(for_reward=True)
    torch.testing.assert_close(frozen["lower_body_position_margin"], target.new_full((2, 12), .2 / 3))
    assert frozen["family_group_index"].tolist() == [1, 1]
    runtime.reset(torch.tensor([0], device=device))
    runtime.command.frame_ids[0] = 0
    live = runtime.tracking_relaxation_state()
    assert live["sample_valid"].tolist() == [False, True]
    assert live["lower_body_position_margin"][0].eq(0).all()
    assert live["lower_body_position_raw_error"][0].isnan().all()
    assert runtime.tracking_relaxation_state(for_reward=True)["reference_frame_ids"].tolist() == [5, 8]
    runtime.rewards(); runtime.rewards()
    assert len(calls) == 3
    move(10, 1)  # Outside map, never reuse last valid margin.
    runtime.env.common_step_counter += 1
    with pytest.raises(RuntimeError, match="this control step"):
        runtime.tracking_relaxation_state(for_reward=True)
    runtime.rewards()
    outside = runtime.tracking_relaxation_state(for_reward=True)
    assert outside["tile_level"].tolist() == [-1, -1] and not outside["tile_valid"].any()
    assert outside["difficulty"].isnan().all()
    for term in RELAXATION_TERMS:
        assert outside[f"{term}_margin"].eq(0).all()
        torch.testing.assert_close(outside[f"{term}_raw_error"], outside[f"{term}_relaxed_error"], rtol=0, atol=0)
    assert scene.terrain.terrain_levels.eq(0).all() and scene.terrain.terrain_types.eq(0).all()


def test_task_defaults_and_explicit_overrides_are_owned():
    from pmt_tasks.builder import build_task_config
    cfg = build_task_config("pgmt_stage2")
    assert not cfg.tracking_relaxation.enabled
    changed = build_task_config("pgmt_stage2", overrides=["tracking_relaxation.enabled=true",
        "tracking_relaxation.position_max=0.12", "tracking_relaxation.terrain_families=[stairs,rough]"])
    assert changed.tracking_relaxation.enabled and changed.tracking_relaxation.position_max == .12
    assert list(changed.tracking_relaxation.terrain_families) == ["stairs", "rough"]
    assert "tracking_relaxation" not in build_task_config("pgmt_stage1")
    default = tracking_relaxation_config()
    default["terrain_families"].append("flat")
    assert tracking_relaxation_config()["terrain_families"] == ["stairs", "slope", "boxes"]


def test_enabled_contract_changes_checkpoint_identity_for_caps_families_and_names():
    from pmt_tasks.pgmt_curriculum import _identity, TerrainCurriculum
    class Scene(dict):
        pass
    scene = Scene(robot=SimpleNamespace(body_names=BFS_BODIES, joint_names=BFS_JOINTS))
    scene.terrain = SimpleNamespace(manifest={"asset_sha256": "test"})
    env = SimpleNamespace(scene=scene, cfg=SimpleNamespace(pgmt_tracking_relaxation={"enabled": False}))
    command = SimpleNamespace(terrain_families=["flat", "stairs", "slope", "boxes", "rough"], compatibility_sha256="test",
        data_store=SimpleNamespace(motions=[], motion_lengths=torch.tensor([])),
        sampler=SimpleNamespace(alpha=.1, beta=1., uniform_ratio=.1, update_interval=1, enable_frame_sampling=False),
        cfg=SimpleNamespace(fixed_start_frame=0, require_valid_support_start=False))
    assert "tracking_relaxation" not in _identity(env, command)
    env.cfg.pgmt_tracking_relaxation["enabled"] = True
    original = _identity(env, command)
    assert len(original["tracking_relaxation"]["names"]["lower_body_position"]) == 12
    terrain = SimpleNamespace(terrain_levels=torch.tensor([0]), terrain_types=torch.tensor([0]),
                              terrain_origins=torch.zeros(10, 5, 3), env_origins=torch.zeros(1, 3))
    curriculum = TerrainCurriculum(terrain, original)
    saved = curriculum.state_dict()
    for mutation in ({"position_max": .1}, {"terrain_families": ["stairs"]},
                     {"alpha": {"lower_joint_position": {"left_knee_joint": .5}}}, {"level_max": 8}):
        env.cfg.pgmt_tracking_relaxation = {"enabled": True, **mutation}
        changed = _identity(env, command)
        assert changed != original
        with pytest.raises(ValueError, match="checkpoint"):
            TerrainCurriculum(terrain, changed).validate_state_dict(saved)
