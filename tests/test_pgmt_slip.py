"""T3 gate, world-XY squared speed, negative weight and frozen timeline."""
from types import MethodType, SimpleNamespace

import pytest
import torch

from pmt_tasks.pgmt_contact_state import ContactEventState
from pmt_tasks.pgmt_runtime import PGMTState
from pmt_tasks.pgmt_slip import SLIP_WEIGHT, empty_slip, evaluate_slip


def test_exact_xy_squared_sum_and_negative_weight_dt_once():
    velocity = torch.tensor([[[3., 4., 123.], [1., -2., -456.]]])
    contact = torch.ones(1, 2, dtype=torch.bool)
    result = evaluate_slip(contact, contact, velocity)
    assert result["speed_xy_sq"].tolist() == [[25., 5.]]
    assert result["rate"].item() == 30.  # Sum over feet, not a mean or a speed norm.
    assert (result["rate"] * SLIP_WEIGHT * .02).item() == pytest.approx(-.6)
    result["velocity_w"].zero_()
    assert velocity[0, 0, 0] == 3.  # Independent diagnostic copy.


@pytest.mark.parametrize("contacts,expected", [([False, False], 0.), ([True, False], 25.),
                                              ([False, True], 5.), ([True, True], 30.)])
def test_contact_gate_for_each_foot(contacts, expected):
    contact = torch.tensor([contacts])
    result = evaluate_slip(contact, torch.ones_like(contact), torch.tensor([[[3., 4., 9.], [1., -2., -9.]]]))
    assert result["rate"].item() == expected
    assert result["per_foot"][~contact].eq(0).all()


def test_vertical_speed_ignored_yaw_invariance_and_no_dead_zone():
    contact = torch.ones(1, 2, dtype=torch.bool)
    velocity = torch.tensor([[[.3, .4, 0.], [0., 0., 5.]]], dtype=torch.float64)
    before = evaluate_slip(contact, contact, velocity)
    rotated = velocity.clone()
    rotated[..., 0], rotated[..., 1] = -velocity[..., 1], velocity[..., 0]
    rotated[..., 2] += 999.
    torch.testing.assert_close(evaluate_slip(contact, contact, rotated)["rate"], before["rate"])
    assert before["per_foot"][0, 1] == 0.
    velocity.zero_(); velocity[0, 0, 0] = .001
    assert evaluate_slip(contact, contact, velocity)["rate"].item() == pytest.approx(1e-6)


def test_all_object_contact_side_forces_and_strict_threshold():
    tracker = ContactEventState(3, device="cpu", dt=.02)
    net = torch.tensor([[[100., 0., 0.]]*2, [[0., 0., -100.]]*2,
                        [[0., 0., 10.], [0., 0., 10.01]]])
    v = torch.ones(3, 2, 3)
    for step in (1, 2, 3):
        tracker.update(step, net, torch.zeros_like(net))
        s = tracker.snapshot()
        assert not s["support"].any() and not s["touchdown"].any()
        assert evaluate_slip(s["contact"], s["contact_valid"], v)["rate"].tolist() == [4., 4., 2.]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1e30])
def test_invalid_velocity_or_overflow_is_unknown_without_nan_cost(value):
    contact = torch.ones(1, 2, dtype=torch.bool)
    v = torch.tensor([[[value, 0., 0.], [3., 4., 0.]]])
    result = evaluate_slip(contact, contact, v)
    assert result["valid"].tolist() == [[False, True]]
    assert result["speed_xy_sq"][0, 0].isnan()
    assert result["rate"].item() == 25.


def test_sum_overflow_and_unknown_contact_are_explicitly_invalid():
    contact = torch.ones(1, 2, dtype=torch.bool)
    overflow = evaluate_slip(contact, contact, torch.tensor([[[1.4e19, 0., 0.]]*2]))
    assert not overflow["valid"].any() and overflow["rate"].eq(0).all()
    result = evaluate_slip(contact, torch.tensor([[False, True]]), torch.tensor([[[3., 4., 0.], [1., 0., 0.]]]))
    assert result["valid"].tolist() == [[False, True]] and result["rate"].item() == 1.


@pytest.mark.parametrize("bad", ["state_dtype", "mask_dtype", "velocity_dtype", "velocity_shape"])
def test_input_contract_rejected(bad):
    c = torch.ones(2, 2, dtype=torch.bool)
    mask, v = c.clone(), torch.zeros(2, 2, 3)
    if bad == "state_dtype": c = c.float()
    if bad == "mask_dtype": mask = mask.float()
    if bad == "velocity_dtype": v = v.long()
    if bad == "velocity_shape": v = v[..., :2]
    with pytest.raises(ValueError, match="Slip requires"):
        evaluate_slip(c, mask, v)


def test_runtime_uses_physical_link_velocity_and_preserves_terminal_cache(monkeypatch):
    state = PGMTState.__new__(PGMTState)
    net = torch.zeros(2, 2, 3); net[..., 0] = 100.
    state.env = SimpleNamespace(common_step_counter=1, step_dt=.02,
                               scene={"contact_forces": SimpleNamespace(data=SimpleNamespace(net_forces_w=net))})
    state.contact_tracker = ContactEventState(2, device="cpu", dt=.02)
    state.slip_enabled = True
    state._slip_live = empty_slip(2, device="cpu")
    state._slip_reward = state._reward_contacts = state._contact_reward_step = None
    state._reward_step, state._obs_version = None, 0
    state.contact_order = state.foot_body_indices = torch.tensor([0, 1])
    state.anchor, state.config, state.groups = 0, {}, {}
    state.pending_reset = torch.ones(2, dtype=torch.bool)
    state.previous_target = torch.zeros(2, 29)
    state.previous_velocity, state.previous_ref_velocity = torch.zeros(2, 2, 3), torch.zeros(2, 2, 3)
    state.robot = SimpleNamespace(data=SimpleNamespace(joint_pos_target=torch.zeros(2, 29),
                                                       soft_joint_pos_limits=torch.zeros(2, 29, 2)))
    labels = dict(contacts=torch.zeros(2, 2), contact_valid=torch.zeros(2, 2, dtype=torch.bool),
                  motion_ids=torch.tensor([0, 1]), frame_ids=torch.tensor([8, 12]))
    state._contact_context = {f"reference_{key}": value.clone() for key, value in labels.items()}
    state._invalidate_contact_context(slice(None))
    physical = dict(body_pos=torch.zeros(2, 2, 3), body_quat=torch.zeros(2, 2, 4),
                    body_lin_vel=torch.tensor([[[3., 4., 0.], [1., 2., 0.]], [[1., 0., 0.], [0., 0., 9.]]]),
                    body_ang_vel=torch.zeros(2, 2, 3))
    reference = {key: value.clone() for key, value in physical.items()}
    reference["body_lin_vel"].fill_(100.)  # T3 must not subtract reference velocity.
    state.reference_contacts = MethodType(lambda self: labels, state)
    state.robot_foot_terrain_forces = MethodType(lambda self: torch.zeros_like(net), state)
    state.robot_state = MethodType(lambda self: physical, state)
    state.reference = MethodType(lambda self: reference, state)
    calls = []
    def old_rewards(*args, **kwargs):
        calls.append(1)
        return {"upper_fixture": torch.zeros(2)}
    monkeypatch.setattr("pmt_tasks.pgmt_runtime.reward_terms", old_rewards)
    assert state.slip_state()["rate"].eq(0).all() and not state.slip_state()["valid"].any()
    with pytest.raises(RuntimeError, match="this control step"):
        state.slip_state(for_reward=True)
    assert state.rewards()["terrain_slip"].tolist() == [30., 1.]
    assert not state.contact_state()["touchdown"].any()
    state.reset(torch.tensor([0]))
    physical["body_lin_vel"].fill_(9.)
    labels["frame_ids"][0], labels["motion_ids"][0] = 0, 1
    for _ in range(3):
        assert state.rewards()["terrain_slip"].tolist() == [30., 1.]
        snapshot = state.slip_state(for_reward=True)
        assert snapshot["reference_frame_ids"].tolist() == [8, 12]
        assert snapshot["reference_motion_ids"].tolist() == [0, 1]
        snapshot["rate"].zero_(); snapshot["velocity_w"].zero_()
    assert len(calls) == 1
    live = state.slip_state()
    assert live["rate"].tolist() == [0., 1.]
    assert live["velocity_w"][0].isnan().all() and not live["valid"][0].any()
    assert state.slip_state(env_ids=torch.tensor([1]))["rate"].item() == 1.
    for step in (2, 3):
        state.env.common_step_counter = step
        with pytest.raises(RuntimeError, match="this control step"):
            state.slip_state(for_reward=True)
        assert state.rewards()["terrain_slip"].tolist() == [324., 324.]
    assert len(calls) == 3
    state.env.common_step_counter = 4
    net.zero_()
    assert state.rewards()["terrain_slip"].eq(0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_cuda_cost_and_masks_parity():
    g = torch.Generator().manual_seed(711)
    c, valid = (torch.rand(8192, 2, generator=g) > .4 for _ in range(2))
    v = torch.randn(8192, 2, 3, generator=g)
    v[0, 0, 0] = float("nan")
    cpu, cuda = evaluate_slip(c, valid, v), evaluate_slip(c.cuda(), valid.cuda(), v.cuda())
    for key in cpu:
        torch.testing.assert_close(cpu[key], cuda[key].cpu(), equal_nan=True, rtol=1e-6, atol=1e-7)
