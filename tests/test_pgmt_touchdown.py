"""T1 geometry, population variance, validity and once-per-step lifecycle."""
from types import MethodType, SimpleNamespace
import math

import pytest
import torch

from pmt_tasks.pgmt_contact_state import ContactEventState
from pmt_tasks.pgmt_runtime import PGMTState
from pmt_tasks.pgmt_touchdown import (
    TOUCHDOWN_QUALITY_CONFIG, TOUCHDOWN_QUALITY_WEIGHT, empty_touchdown,
    evaluate_touchdown, evaluate_support_quality, footprint_points, height_quality, validate_touchdown_config,
)


def inputs(n=1, device="cpu", dtype=torch.float64):
    positions = torch.zeros(n, 2, 3, device=device, dtype=dtype)
    rotations = torch.zeros(n, 2, 4, device=device, dtype=dtype)
    rotations[..., 0] = 1
    return (torch.ones(n, 2, dtype=torch.bool, device=device), positions, rotations,
            positions[:, 0].clone(), rotations[:, 0].clone())


class Plane:
    def __init__(self, *, slope=0., z=0.):
        self.slope, self.z, self.calls = slope, z, []

    def __call__(self, xy):
        self.calls.append(xy.clone())
        return dict(height=xy[:, 0] * self.slope + self.z, valid=torch.ones_like(xy[:, 0], dtype=torch.bool))


def evaluate(args, query):
    return evaluate_touchdown(*args, query, TOUCHDOWN_QUALITY_CONFIG)


def test_footprint_exact_dimensions_offset_and_each_foot_yaw():
    _, p, q, *_ = inputs()
    q[0, 1] = q.new_tensor([2**-.5, 0, 0, 2**-.5])
    xy = footprint_points(p[0], q[0], TOUCHDOWN_QUALITY_CONFIG)
    x, y = torch.meshgrid(torch.tensor([-.061, .039, .139], dtype=p.dtype),
                         torch.tensor([-.05, 0, .05], dtype=p.dtype), indexing="ij")
    expected = torch.stack((x.flatten(), y.flatten()), -1)
    torch.testing.assert_close(xy[0], expected, atol=1e-15, rtol=1e-14)
    torch.testing.assert_close(xy[1], torch.stack((-expected[:, 1], expected[:, 0]), -1))


@pytest.mark.parametrize("z", [0., -3., 12.])
def test_geometry_helper_flat_offset_invariance_and_raw_two_foot_sum(z):
    query = Plane(z=z)
    result = evaluate(inputs(), query)
    assert len(query.calls) == 1 and query.calls[0].shape == (18, 2)
    assert result["valid"].all() and result["quality"].eq(1).all()
    assert result["variance"].eq(0).all() and result["height_w"].eq(z).all()
    assert result["rate"].item() == 2.


def support_quality(support, ref=None, *, support_valid=None, ref_valid=None, events=None, query=None, poses=None):
    support = torch.as_tensor(support, dtype=torch.bool)
    n = len(support)
    valid = torch.ones_like(support)
    ref = support.float() if ref is None else torch.as_tensor(ref, device=support.device)
    poses = inputs(n, device=support.device) if poses is None else poses
    return evaluate_support_quality(
        support, valid if support_valid is None else support_valid,
        ref, valid if ref_valid is None else ref_valid,
        *poses[1:], Plane() if query is None else query, TOUCHDOWN_QUALITY_CONFIG,
        touchdown=torch.zeros_like(support) if events is None else events)


@pytest.mark.parametrize("actual", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("reference", [(False, False), (True, False), (False, True), (True, True)])
def test_exact_whole_pattern_gate_all_sixteen_combinations(actual, reference):
    query = Plane()
    result = support_quality([actual], [reference], query=query)
    expected = float(actual == reference and any(reference))
    assert result["pattern_valid"].item()
    assert result["pattern_matched"].item() == (actual == reference)
    assert result["rate"].item() == expected
    assert result["reference_support_count"].item() == sum(reference)
    assert len(query.calls) == int(bool(expected))
    assert not result["event"].any()  # Continuous reward needs no touchdown edge.
    if expected:
        assert query.calls[0].shape == (9 * sum(reference), 2)
        assert (result["rate"] * TOUCHDOWN_QUALITY_WEIGHT * .02).item() == pytest.approx(.2)


@pytest.mark.parametrize("channel", ["support", "reference"])
@pytest.mark.parametrize("foot", [0, 1])
def test_unknown_either_foot_vetoes_entire_pattern_including_expected_swing(channel, foot):
    valid = torch.ones(1, 2, dtype=torch.bool)
    valid[0, foot] = False
    query = Plane()
    kwargs = {"support_valid" if channel == "support" else "ref_valid": valid}
    result = support_quality([[True, False]], query=query, **kwargs)
    assert not result["pattern_valid"].any() and not query.calls
    assert result["rate"].item() == 0.


@pytest.mark.parametrize("unknown", [-1., 2., float("nan"), float("inf")])
def test_nonbinary_reference_vetoes_even_if_valid_mask_claims_known(unknown):
    result = support_quality([[True, False]], [[1., unknown]])
    assert not result["pattern_valid"].any() and result["rate"].item() == 0.


def test_invalid_geometry_cannot_shrink_reference_support_denominator():
    def partial(xy):
        valid = torch.ones(len(xy), dtype=torch.bool)
        valid[0] = False
        return dict(height=torch.zeros_like(xy[:, 0]), valid=valid)
    result = support_quality([[True, True]], query=partial)
    assert result["pattern_matched"].item()
    assert result["quality"].tolist() == [[0., 1.]]
    assert result["reference_support_count"].item() == 2
    assert result["rate"].item() == .5


def test_event_cannot_bypass_pattern_gate_or_add_landing_bonus():
    events = torch.ones(1, 2, dtype=torch.bool)
    mismatch = support_quality([[True, True]], [[True, False]], events=events)
    assert mismatch["event"].all() and mismatch["rate"].item() == 0.
    landing = support_quality([[True, True]], events=events)
    holding = support_quality([[True, True]])
    torch.testing.assert_close(landing["rate"], holding["rate"])


def test_slow_extra_steps_lose_support_reward_without_fast_switch_penalty():
    tracker = ContactEventState(1, device="cpu", dt=.02)
    rates, fast = [], []
    for step in range(1, 25):
        force = torch.zeros(1, 2, 3)
        force[..., 2] = 100.
        if 13 <= step <= 18:
            force[0, 0, 2] = 0.
        tracker.update(step, force, force)
        sample = tracker.snapshot()
        result = support_quality(sample["support"], [[1., 1.]], support_valid=sample["support_valid"],
                                 events=sample["touchdown"])
        rates.append(result["rate"].item())
        fast.append(bool(sample["fast_contact_switch"].any()))
    assert not any(fast)
    assert rates == [1.] * 12 + [0.] * 6 + [1.] * 6
    assert (sum(rates) - 24) * 10 * .02 == pytest.approx(-1.2)


def test_slope_population_variance_uses_each_foot_yaw_not_torso_yaw():
    args = inputs()
    args[2][0, 1] = args[2].new_tensor([2**-.5, 0, 0, 2**-.5])
    slope = 3**-.5  # tan(30 degrees)
    result = evaluate(args, Plane(slope=slope, z=-2.))
    expected = result["variance"].new_tensor([[2/3*.1**2*slope**2, 2/3*.05**2*slope**2]])
    torch.testing.assert_close(result["variance"], expected)
    torch.testing.assert_close(result["quality"], torch.exp(-expected/.03**2))


def test_step_edge_variance_uses_population_divisor_nine():
    heights = torch.tensor([[0., 0., 0., 0., 0., .1, .1, .1, .1]], dtype=torch.float64)
    quality, variance, valid = height_quality(heights, torch.ones_like(heights, dtype=torch.bool), .03)
    assert valid.item()
    assert variance.item() == pytest.approx(5*4/81*.1**2)
    assert quality.item() == pytest.approx(math.exp(-(5*4/81*.1**2)/.03**2))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), 0.])
def test_one_invalid_point_suppresses_whole_foot_without_nan_reward(bad):
    def query(xy):
        height, valid = xy[:, 0].clone().zero_(), torch.ones(len(xy), dtype=torch.bool)
        height[0], valid[0] = bad, False
        return dict(height=height, valid=valid)
    result = evaluate(inputs(), query)
    assert result["valid"].tolist() == [[False, True]]
    assert result["quality"].tolist() == [[0., 1.]]
    assert result["variance"][0, 0].isnan()
    assert result["rate"].item() == 1.


def test_nonfinite_height_is_rejected_even_if_query_claims_valid():
    def query(xy):
        return dict(height=torch.full_like(xy[:, 0], float("nan")), valid=torch.ones(len(xy), dtype=torch.bool))
    result = evaluate(inputs(), query)
    assert not result["point_valid"].any() and result["rate"].eq(0).all()


def test_scan_coverage_uses_current_torso_yaw_and_rejects_partially_outside_foot():
    args = inputs()
    args[1][0, :, 0] = .9  # toe x=1.039 outside the 2 m scan, despite valid terrain.
    result = evaluate(args, Plane())
    assert result["map_valid"].all() and not result["scan_valid"].all()
    assert result["rate"].eq(0).all()
    args[1][0, :, :2] = args[1].new_tensor([1., 0.])
    args[4][0] = args[4].new_tensor([math.cos(math.pi/8), 0, 0, math.sin(math.pi/8)])
    assert evaluate(args, Plane())["valid"].all()  # 45-degree torso rotates the square.


def test_query_only_active_foot_and_no_queries_without_events():
    args, query = inputs(3), Plane()
    args[0].zero_()
    result = evaluate(args, query)
    assert not query.calls and result["rate"].eq(0).all()
    args[0][2, 1] = True
    result = evaluate(args, query)
    assert len(query.calls) == 1 and query.calls[0].shape == (9, 2)
    assert result["rate"].tolist() == [0., 0., 1.]


@pytest.mark.parametrize("kind", ["foot_position", "foot_quaternion", "torso_position", "torso_quaternion"])
def test_invalid_pose_never_reaches_ray_query(kind):
    args, query = inputs(), Plane()
    if kind == "foot_position":
        args[1][0, :, 0] = float("nan")
    elif kind == "foot_quaternion":
        args[2].zero_()
    elif kind == "torso_position":
        args[3][0, 0] = float("inf")
    else:
        args[4].fill_(float("nan"))
    result = evaluate(args, query)
    assert not query.calls and result["event"].all()
    assert not result["valid"].any() and result["rate"].eq(0).all()


@pytest.mark.parametrize("key,value", [("sigma_m", 0.), ("length_m", -1.), ("center_x_m", float("nan")),
                                       ("settlement", "touchdown_event_v1")])
def test_bad_configuration_rejected(key, value):
    with pytest.raises(ValueError):
        validate_touchdown_config(dict(TOUCHDOWN_QUALITY_CONFIG, **{key: value}))


def test_settlement_version_distinguishes_changed_training_contract():
    from motion_tracking_rl.utils.pgmt_checkpoint import digest
    old = {k: v for k, v in TOUCHDOWN_QUALITY_CONFIG.items() if k != "settlement"}
    assert digest(old) != digest(TOUCHDOWN_QUALITY_CONFIG)
    with pytest.raises(ValueError, match="contract"):
        validate_touchdown_config(old)


def test_runtime_freezes_terminal_geometry_and_no_requery_or_late_payout_after_reset():
    args, query = inputs(n=2, dtype=torch.float32), Plane()
    runtime = PGMTState.__new__(PGMTState)
    runtime.env = SimpleNamespace(common_step_counter=0, scene=SimpleNamespace(terrain=SimpleNamespace(query=query)))
    runtime.anchor = 2
    runtime.foot_body_indices = torch.tensor([0, 1])
    runtime.contact_tracker = ContactEventState(2, device="cpu", dt=.02)
    runtime.touchdown_config = TOUCHDOWN_QUALITY_CONFIG.copy()
    runtime._contact_context, runtime._touchdown_live = {}, empty_touchdown(2, device="cpu")
    runtime._reward_contacts = runtime._touchdown_reward = None
    runtime._obs_version = 0
    runtime.pending_reset = torch.zeros(2, dtype=torch.bool)
    runtime.previous_target = torch.zeros(2, 29)
    runtime.robot = SimpleNamespace(data=SimpleNamespace(joint_pos_target=torch.zeros(2, 29)))
    labels = dict(motion_ids=torch.tensor([0, 1]), frame_ids=torch.tensor([7, 9]),
                  contacts=torch.ones(2, 2), contact_valid=torch.ones(2, 2, dtype=torch.bool))
    runtime.reference_contacts = MethodType(lambda self: labels, runtime)
    force = torch.zeros(2, 2, 3)
    runtime.robot_foot_terrain_forces = MethodType(lambda self: force, runtime)
    physical = {"body_pos": torch.zeros(2, 3, 3), "body_quat": torch.zeros(2, 3, 4),
                "body_lin_vel": torch.zeros(2, 3, 3), "body_ang_vel": torch.zeros(2, 3, 3)}
    physical["body_quat"][..., 0] = 1.
    for step in (1, 2, 3):
        runtime.env.common_step_counter = step
        if step == 3:
            force[..., 2] = 100.
        runtime._update_contact_state(physical, torch.cat((force, torch.zeros(2, 1, 3)), 1))
    assert len(query.calls) == 1
    reward = runtime.touchdown_state(for_reward=True)
    assert reward["rate"].tolist() == [1., 1.]
    assert reward["pattern_matched"].all() and reward["reference_support_count"].eq(2).all()
    runtime.reset(torch.tensor([0]))
    labels["frame_ids"][0] = 0
    labels["contacts"][0] = 0
    assert runtime.touchdown_state()["rate"].tolist() == [0., 1.]
    assert not runtime.touchdown_state()["pattern_matched"][0]
    frozen = runtime.touchdown_state(for_reward=True)
    assert frozen["reference_frame_ids"].tolist() == [7, 9]
    assert frozen["rate"].tolist() == [1., 1.]
    frozen["quality"].zero_()
    assert runtime.touchdown_state(for_reward=True)["quality"].eq(1).all()
    assert len(query.calls) == 1
    runtime.env.common_step_counter += 1
    with pytest.raises(RuntimeError, match="this control step"):
        runtime.touchdown_state(for_reward=True)
    runtime._update_contact_state(physical, torch.cat((force, torch.zeros(2, 1, 3)), 1))
    assert runtime.touchdown_state()["rate"].tolist() == [0., 1.] and len(query.calls) == 2
    assert not runtime.touchdown_state()["event"].any()


def test_current_geometry_is_requeried_when_support_remains_matched():
    tracker = ContactEventState(1, device="cpu", dt=.02)
    args = inputs(dtype=torch.float32)
    low, high = torch.zeros(1, 2, 3), torch.zeros(1, 2, 3)
    high[..., 2] = 100.
    query = Plane()
    rates = []
    for step, force in enumerate((low, low, high, high, high), 1):
        tracker.update(step, force, force)
        sample = tracker.snapshot()
        args[1][..., 0] = 2. if step in (3, 5) else 0.
        result = support_quality(sample["support"], [[1., 1.]], events=sample["touchdown"], query=query, poses=args)
        rates.append(result["rate"].item())
    assert rates == [0., 0., 0., 1., 0.]  # No stale landing quality or deferred bonus.
    assert len(query.calls) == 3


@pytest.mark.parametrize("matched", [True, False])
def test_nine_point_viewer_keeps_event_and_continuous_reward_separate(matched):
    import numpy as np
    from scripts.pgmt_touchdown_visualization import TouchdownDisplay
    display = TouchdownDisplay.__new__(TouchdownDisplay)
    display.reset()
    positions = torch.zeros(1, 3, 3)
    quaternions = torch.zeros(1, 3, 4)
    quaternions[..., 0] = 1
    state = SimpleNamespace(
        env=SimpleNamespace(device="cpu", scene=SimpleNamespace(terrain=SimpleNamespace(query=Plane()))),
        foot_body_indices=torch.tensor([0, 1]), anchor=2, touchdown_config=TOUCHDOWN_QUALITY_CONFIG)
    robot = SimpleNamespace(data=SimpleNamespace(body_pos_w=positions, body_quat_w=quaternions))
    display.sample(state, robot)
    assert not display.preview["eligible"].any() and display.settlement is None
    sample = support_quality([[True, True]], [[1., 1. if matched else 0.]],
                             events=torch.full((1, 2), not matched, dtype=torch.bool))
    sample.update(reference_frame_ids=torch.tensor([12]), episode_id=torch.tensor([1]), sample_step=torch.tensor([13]))
    before = {k: v.clone() for k, v in sample.items()}
    display.sample(state, robot, sample)
    assert display.settlement["rate"] == float(matched)
    assert display.preview["eligible"].all() == matched
    assert np.isfinite(display.preview["height_w"]).all()  # Unrewarded landing keeps preview geometry.
    assert bool(display.preview["event"].any()) == (not matched)
    assert (display.last_events[0] is None) == matched
    for key in sample:
        torch.testing.assert_close(sample[key], before[key], equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_cuda_quality_and_geometry_parity():
    args = inputs(8, dtype=torch.float32)
    generator = torch.Generator().manual_seed(113)
    args[1].copy_(torch.randn(8, 2, 3, generator=generator) * .6)
    args[2].copy_(torch.nn.functional.normalize(torch.randn(8, 2, 4, generator=generator), dim=-1))
    args[0][0] = False
    cpu = evaluate(args, Plane(slope=.4, z=-2.))
    gpu = evaluate(tuple(x.cuda() for x in args), Plane(slope=.4, z=-2.))
    for key in cpu:
        torch.testing.assert_close(cpu[key], gpu[key].cpu(), atol=1e-6, rtol=1e-5, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_cuda_support_gate_and_continuous_quality_parity():
    generator = torch.Generator().manual_seed(114)
    support = torch.rand(128, 2, generator=generator) > .3
    reference = torch.rand(128, 2, generator=generator) > .3
    valid = torch.rand(128, 2, generator=generator) > .1
    poses = inputs(128, dtype=torch.float32)
    cpu = support_quality(support, reference, ref_valid=valid, query=Plane(slope=.4), poses=poses)
    gpu = support_quality(support.cuda(), reference.cuda(), ref_valid=valid.cuda(), query=Plane(slope=.4),
                          poses=tuple(x.cuda() for x in poses))
    for key in cpu:
        torch.testing.assert_close(cpu[key], gpu[key].cpu(), atol=1e-6, rtol=1e-5, equal_nan=True)
