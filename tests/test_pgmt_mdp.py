"""Numerical parity with the local PGMT formulas and native lifecycle boundaries."""
import ast
import importlib.util
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
import yaml

from pmt_tasks.pgmt_math import (REWARD_WEIGHTS, TRACKING_CONFIG, body_groups, build_observations,
                                future_offsets, grouped_rewards, reward_terms)
from pmt_tasks.pgmt_runtime import PGMTState, reward_groups_from_manager
from scripts.pmt_npz_to_mjlab import BFS_BODIES, BFS_JOINTS

PGMT = Path(__file__).resolve().parents[2] / "PGMT"


def legacy_class():
    """Compile only pure observation/reward methods; never import Isaac Sim."""
    source = ast.parse((PGMT / "pgmt/environment.py").read_text())
    methods = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "TrackingEnv")
    selected = [n for n in methods.body if isinstance(n, ast.FunctionDef)
                and n.name in {"observe", "corrected_command", "rewards"}]
    spec = importlib.util.spec_from_file_location("pgmt_reference_math", PGMT / "pgmt/math.py")
    math = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(math)
    namespace = {name: getattr(math, name) for name in
                 ("conjugate", "multiply", "inverse_rotate", "yaw_quat", "angle", "rotation6")}
    namespace.update(torch=torch, yaml=yaml, ROOT=PGMT, future_offsets=future_offsets)
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
                 str(PGMT / "pgmt/environment.py"), "exec"), namespace)
    return type("Legacy", (), {name: namespace[name] for name in ("observe", "corrected_command", "rewards")})


def fixture(n=5):
    torch.manual_seed(123)
    def rand(*shape):
        return torch.randn(*shape, dtype=torch.float64)
    def quat(*shape):
        return torch.nn.functional.normalize(rand(*shape, 4), dim=-1)
    state = {"joint_pos": rand(n, 29), "joint_vel": rand(n, 29),
             "body_pos": rand(n, 30, 3), "body_quat": quat(n, 30),
             "body_lin_vel": rand(n, 30, 3), "body_ang_vel": rand(n, 30, 3),
             "root_quat": quat(n), "root_lin_vel": rand(n, 3), "root_ang_vel": rand(n, 3)}
    ref = {key: value.clone() for key, value in state.items() if not key.startswith("root")}
    ref["joint_pos"] += 0.17 * rand(n, 29)
    ref["body_pos"] += 0.2 * rand(n, 30, 3)
    ref["body_quat"] = quat(n, 30)
    ref["body_lin_vel"] += 0.25 * rand(n, 30, 3)
    ref["contacts"] = torch.zeros(n, 2, dtype=torch.float64)
    future = {k: v[:, None].expand(n, 8, *v.shape[1:]).clone() for k, v in ref.items()}
    qtarget, previous = rand(n, 29), rand(n, 29)
    return state, ref, future, qtarget, previous, rand(n, 3)


def test_observations_match_actual_pgmt_method():
    state, ref, future, target, _, origins = fixture()
    cfg = yaml.safe_load((PGMT / "configs/optimized_v2.yaml").read_text())
    old = legacy_class()()
    old.cfg, old.tr, old.n, old.stage, old.device = cfg, cfg["tracking"], 5, 1, "cpu"
    old.anchor = BFS_BODIES.index("torso_link")
    old.dr = {"observation_noise": False}
    old.reference = lambda: ref
    old.library = SimpleNamespace(get=lambda *_: future)
    old.motion = old.frame = torch.zeros(5, dtype=torch.long)
    old.qtarget = target
    old.history = torch.zeros(5, 10, 96)
    old.robot = SimpleNamespace(data=SimpleNamespace(
        body_quat_w=state["body_quat"], body_pos_w=state["body_pos"], root_quat_w=state["root_quat"],
        root_ang_vel_w=state["root_ang_vel"], root_link_lin_vel_w=state["root_lin_vel"],
        joint_pos=state["joint_pos"], joint_vel=state["joint_vel"]))
    old.scene = SimpleNamespace(env_origins=origins)
    expected = old.observe(update_history=False)
    actual = build_observations(state, ref, future, target, origins, anchor=old.anchor)
    for ours, original in (("proprio", "proprio"), ("command_window", "commands"), ("critic", "privileged")):
        torch.testing.assert_close(actual[ours], expected[original], rtol=1e-12, atol=1e-12)
    assert actual["critic"].shape == (5, 251)
    assert actual["command_window"].shape == (5, 8, 61)
    torch.testing.assert_close(actual["proprio"][:, -29:], target)


def test_reward_groups_match_actual_pgmt_method_with_recovery_disabled():
    state, ref, _, target, previous_target, origins = fixture()
    groups = body_groups(BFS_BODIES, BFS_JOINTS)
    cfg = yaml.safe_load((PGMT / "configs/optimized_v2.yaml").read_text())
    old = legacy_class()()
    old.cfg, old.tr, old.n, old.stage, old.device, old.dt = cfg, cfg["tracking"], 5, 1, "cpu", 0.02
    old.anchor = BFS_BODIES.index("torso_link")
    old.upper, old.lower = groups["upper"], groups["lower"]
    old.uj, old.lj, old.end_effectors = groups["upper_joints"], groups["lower_joints"], groups["end_effectors"]
    old.wj = torch.tensor([i for i, n in enumerate(BFS_JOINTS) if "wrist" in n])
    old.feet = torch.tensor([BFS_BODIES.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")])
    old.undesired, old.force_order = groups["undesired"], torch.arange(30)
    old.qtarget, old.previous_target = target, previous_target
    previous_velocity = state["body_lin_vel"] * 0.8
    previous_ref_velocity = ref["body_lin_vel"] * 0.7
    old.previous_velocity, old.previous_ref_velocity = previous_velocity.clone(), previous_ref_velocity.clone()
    old.recovery = torch.zeros(5, dtype=torch.bool)
    old.previous_contacts = torch.zeros(5, 2, dtype=torch.bool)
    old.contact_age = torch.zeros(5, 2)
    old.weights = {}
    forces = torch.randn(5, 30, 3, dtype=torch.float64) * 25
    limits = torch.tensor([-1., 1.], dtype=torch.float64).expand(5, 29, 2)
    old.robot = SimpleNamespace(data=SimpleNamespace(
        body_pos_w=state["body_pos"], body_quat_w=state["body_quat"],
        body_link_lin_vel_w=state["body_lin_vel"], body_ang_vel_w=state["body_ang_vel"],
        root_pos_w=state["body_pos"][:, 0], root_link_lin_vel_w=state["root_lin_vel"],
        joint_pos=state["joint_pos"], joint_vel=state["joint_vel"], soft_joint_pos_limits=limits))
    old.sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=forces))
    old.scene = SimpleNamespace(env_origins=origins)
    expected, _ = old.rewards(ref)
    rates = reward_terms(state, ref, target, previous_target, previous_velocity, previous_ref_velocity,
                         forces, limits, groups, anchor=old.anchor, dt=.02)
    torch.testing.assert_close(grouped_rewards(rates) * .02, expected, rtol=1e-12, atol=1e-12)


def test_future_window_clamps_each_native_chunk_not_longest_padding():
    runtime = PGMTState.__new__(PGMTState)
    runtime.command = SimpleNamespace(motion_ids=torch.tensor([0, 1]), frame_ids=torch.tensor([3, 1]))
    runtime.env = SimpleNamespace(scene=SimpleNamespace(env_origins=torch.tensor([[1., 2., 3.], [4., 5., 6.]])))
    def get(ids, frames):
        assert torch.all(frames < torch.tensor([5, 3])[ids])
        base = frames.float()
        return (base[:, None].expand(-1, 29), base[:, None].expand(-1, 29),
                base[:, None, None].expand(-1, 30, 3), base[:, None, None].expand(-1, 30, 4),
                base[:, None, None].expand(-1, 30, 3), base[:, None, None].expand(-1, 30, 3))
    runtime.full_store = SimpleNamespace(motion_lengths=torch.tensor([5, 3]), get_motion_data=get)
    result = runtime.reference(future=True)
    torch.testing.assert_close(result["joint_pos"][:, :, 0], torch.tensor([[3., 4., 4., 4., 4., 4., 4., 4.],
                                                                          [1., 2., 2., 2., 2., 2., 2., 2.]]))
    torch.testing.assert_close(result["body_pos"][:, 0, 0, :], torch.tensor([[4., 5., 6.], [5., 6., 7.]]))


@pytest.mark.parametrize("device", ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []))
@pytest.mark.parametrize("reverse_bodies", [False, True])
def test_wrist_only_acceleration_excludes_feet_and_averages_six_components(device, reverse_bodies):
    state, ref, _, target, previous_target, _ = fixture(n=3)
    state = {k: v.to(device) for k, v in state.items()}
    ref = {k: v.to(device) for k, v in ref.items()}
    bodies = list(BFS_BODIES)
    if reverse_bodies:
        bodies.reverse()
        for data in (state, ref):
            for key in ("body_pos", "body_quat", "body_lin_vel", "body_ang_vel"):
                data[key] = data[key].flip(1)
    wrists = ("left_wrist_yaw_link", "right_wrist_yaw_link")
    hands = [bodies.index(name) for name in wrists]
    feet = [bodies.index(name) for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
    previous_velocity, previous_ref = state["body_lin_vel"].clone(), ref["body_lin_vel"].clone()
    # Common nonzero acceleration cancels. Only selected link mismatches remain.
    state["body_lin_vel"] += .02 * 5.
    ref["body_lin_vel"] += .02 * 5.
    state["body_lin_vel"][:, feet] += 100.
    ref["body_lin_vel"][:, feet] -= 100.
    state["body_lin_vel"][1, hands] += target.new_tensor([[3., 4., 0.], [0., 0., 12.]]).to(device) * .02
    state["body_lin_vel"][2, hands] += 100.
    def rates(names):
        return reward_terms(state, ref, target.to(device), previous_target.to(device), previous_velocity,
            previous_ref, torch.zeros_like(previous_velocity), target.new_tensor([-3., 3.]).to(device).expand(3, 29, 2),
            body_groups(bodies, BFS_JOINTS, device=device, end_effector_names=names),
            anchor=bodies.index("torso_link"), dt=.02)
    stage1, stage2 = rates(None), rates(wrists)
    key = "auxiliary_end_effector_acceleration"
    torch.testing.assert_close(stage2[key], target.new_tensor([0., 169. / 6., 10000.]).to(device))
    assert (stage1[key] == 10000.).all()  # Stage1 still sees large foot mismatch.
    for name in stage1.keys() - {key}:
        torch.testing.assert_close(stage1[name], stage2[name], rtol=0, atol=0)


def test_group_extraction_uses_frozen_step_contributions_after_reset():
    names = list(REWARD_WEIGHTS)
    rates = torch.arange(2 * len(names), dtype=torch.float32).reshape(2, -1)
    manager = SimpleNamespace(active_terms=names, _step_reward=rates.clone(),
                              _episode_sums={name: torch.zeros(2) for name in names})
    actual = reward_groups_from_manager(manager, .02, ("upper", "lower", "auxiliary"))
    torch.testing.assert_close(actual.sum(-1), rates.sum(-1) * .02)
    assert actual.shape == (2, 3)
    manager.active_terms = names + ["unassigned"]
    with pytest.raises(ValueError, match="Every active"):
        reward_groups_from_manager(manager, .02, ("upper", "lower", "auxiliary"))


@pytest.mark.parametrize("contact_events", [False, True])
@pytest.mark.parametrize("wrist_only", [False, True])
def test_repeated_reward_reads_and_internal_reset_do_not_advance_cache(contact_events, wrist_only):
    state, ref, _, target, prev, _ = fixture(n=2)
    runtime = PGMTState.__new__(PGMTState)
    runtime.anchor = BFS_BODIES.index("torso_link")
    runtime.config = TRACKING_CONFIG.copy()
    runtime.groups = body_groups(BFS_BODIES, BFS_JOINTS, end_effector_names=(
        "left_wrist_yaw_link", "right_wrist_yaw_link") if wrist_only else None)
    runtime.previous_target = prev.clone()
    runtime.previous_velocity = state["body_lin_vel"].clone()
    runtime.previous_ref_velocity = ref["body_lin_vel"].clone()
    runtime.pending_reset = torch.zeros(2, dtype=torch.bool)
    runtime._reward_step = None
    runtime._obs_version = 0
    runtime.contact_order = torch.arange(30)
    resets = []
    history = SimpleNamespace(reset=lambda *, batch_ids: resets.append(batch_ids.clone()))
    runtime.env = SimpleNamespace(common_step_counter=1, step_dt=.02,
        scene={"contact_forces": SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros(2, 30, 3)))},
        observation_manager=SimpleNamespace(_group_obs_term_history_buffer={"proprio_history": {"proprio": history}}))
    runtime.robot = SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=torch.tensor([-3., 3.]).expand(2, 29, 2),
                                                         joint_pos_target=target))
    runtime.robot_state = MethodType(lambda self: state, runtime)
    runtime.reference = MethodType(lambda self: ref, runtime)
    if contact_events:
        from pmt_tasks.pgmt_contact_state import ContactEventState
        runtime.contact_tracker = ContactEventState(2, device="cpu", dt=.02)
        runtime._contact_context = {}
        runtime._reward_contacts = None
        runtime.foot_body_indices = torch.tensor([0, 1])
        runtime.robot_foot_terrain_forces = MethodType(lambda self: torch.zeros(2, 2, 3), runtime)
        runtime.reference_contacts = MethodType(lambda self: dict(
            frame_ids=torch.tensor([4, 5]), contact_valid=torch.ones(2, 2, dtype=torch.bool)), runtime)
    first = runtime.rewards()
    saved = {key: value.clone() for key, value in first.items()}
    target.add_(.25)
    assert runtime.rewards() is first
    if contact_events:
        assert runtime.contact_state()["episode_step"].tolist() == [1, 1]
    for key in saved:
        torch.testing.assert_close(first[key], saved[key])
    runtime.reset(torch.tensor([0]))
    assert runtime.pending_reset.tolist() == [True, False]
    assert len(resets) == 1 and resets[0].tolist() == [0]
    torch.testing.assert_close(runtime.previous_target[0], target[0])
    # Terminal reward cache survives reset until a new control step is rewarded.
    assert runtime.rewards() is first
    if contact_events:
        assert runtime.contact_state()["episode_step"].tolist() == [0, 1]
        assert runtime.contact_state(for_reward=True)["episode_step"].tolist() == [1, 1]
    runtime.env.common_step_counter += 1
    state["body_lin_vel"][0] += 7.  # New episode velocity must not create an acceleration spike.
    second = runtime.rewards()
    assert second is not first
    assert second["auxiliary_action_rate"][0] == 0
    assert second["auxiliary_end_effector_acceleration"][0] == 0
    if contact_events:
        assert runtime.contact_state()["episode_step"].tolist() == [1, 2]
        assert runtime.contact_state()["support_history_valid"].tolist() == [[False, False], [True, True]]
