"""T5 event-only penalty, shared strict age/grace and frozen reward lifecycle."""
from types import MethodType, SimpleNamespace

import pytest
import torch

from pmt_tasks.pgmt_contact_state import CONTACT_STATE_CONFIG, ContactEventState
from pmt_tasks.pgmt_contact_switching import (
    CONTACT_SWITCHING_WEIGHT, empty_contact_switching, evaluate_contact_switching,
)
from pmt_tasks.pgmt_runtime import PGMTState


def forces(contact, *, n=1, device="cpu"):
    result = torch.zeros(n, 2, 3, device=device)
    result[..., 0] = 100. * contact  # Side contact, no terrain support.
    return result


def cost(tracker):
    s = tracker.snapshot()
    return evaluate_contact_switching(s["fast_contact_switch"], s["contact_history_valid"])


@pytest.mark.parametrize("events,valid,expected", [
    ([False, False], [True, True], 0.), ([True, False], [True, True], 1.),
    ([False, True], [True, True], 1.), ([True, True], [True, True], 2.),
    ([True, True], [False, True], 1.), ([True, True], [False, False], 0.),
])
def test_each_foot_sum_measurement_mask_and_weight_dt_once(events, valid, expected):
    event, mask = torch.tensor([events]), torch.tensor([valid])
    result = evaluate_contact_switching(event, mask)
    assert result["rate"].item() == expected
    assert (result["rate"] * CONTACT_SWITCHING_WEIGHT * .02).item() == pytest.approx(-.6 * expected)
    result["fast_contact_switch"].logical_not_(); result["valid"].logical_not_()
    assert event.tolist() == [events] and mask.tolist() == [valid]


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("dt,age_steps,trigger", [(.02, 4, True), (.02, 5, False), (.02, 6, False),
                                                 (.025, 3, True), (.025, 4, False),
                                                 (.03, 3, True), (.03, 4, False)])
def test_both_switch_directions_read_previous_age_strictly(initial, dt, age_steps, trigger):
    cfg = dict(CONTACT_STATE_CONFIG, switch_grace_steps=0)
    tracker = ContactEventState(1, device="cpu", dt=dt, config=cfg)
    for step in range(1, age_steps + 2):
        tracker.update(step, forces(initial), forces(False))
        assert cost(tracker)["rate"].item() == 0.
    tracker.update(age_steps + 2, forces(not initial), forces(False))
    s = tracker.snapshot()
    assert s["contact_switched"].all() and not s["support"].any()
    assert s["previous_contact_age"][0, 0].item() == pytest.approx(age_steps * dt)
    assert s["contact_age"].eq(0).all()
    assert cost(tracker)["rate"].item() == 2. * trigger
    tracker.update(age_steps + 3, forces(not initial), forces(False))
    assert cost(tracker)["rate"].item() == 0.  # Holding contact/air never repeats a switch penalty.


def test_initial_grace_first_five_steps_and_post_reset_baseline():
    tracker = ContactEventState(1, device="cpu", dt=.02)
    for step in range(1, 9):
        tracker.update(step, forces(step % 2 == 0), forces(False))
        assert cost(tracker)["rate"].item() == (2. if step > 5 else 0.)
        assert cost(tracker)["valid"].all().item() == (step > 1)
    tracker.reset([0])
    tracker.update(9, forces(False), forces(False))
    assert not cost(tracker)["valid"].any() and cost(tracker)["rate"].item() == 0.
    assert tracker.snapshot()["episode_step"].item() == 1


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), 1e30])
def test_missing_force_or_norm_overflow_breaks_only_one_foot_history(invalid):
    tracker = ContactEventState(1, device="cpu", dt=.02)
    for step in range(1, 7):
        tracker.update(step, forces(True), forces(False))
    bad = forces(False); bad[0, 0, 0] = invalid
    tracker.update(7, bad, forces(False))
    assert cost(tracker)["valid"].tolist() == [[False, True]]
    assert cost(tracker)["rate"].item() == 0.  # Right old age = .1; left unknown.
    tracker.update(8, forces(True), forces(False))
    assert cost(tracker)["valid"].tolist() == [[False, True]]
    assert cost(tracker)["per_foot"].tolist() == [[0., 1.]]  # Recovery is a baseline, not an edge.
    tracker.update(9, forces(False), forces(False))
    assert cost(tracker)["rate"].item() == 2.


@pytest.mark.parametrize("bad", ["event_dtype", "mask_dtype", "event_shape", "mask_shape"])
def test_input_contract_rejected(bad):
    event = torch.ones(2, 2, dtype=torch.bool)
    mask = event.clone()
    if bad == "event_dtype": event = event.float()
    if bad == "mask_dtype": mask = mask.float()
    if bad == "event_shape": event = event[..., None]
    if bad == "mask_shape": mask = mask[:, :1]
    with pytest.raises(ValueError, match="Contact switching requires"):
        evaluate_contact_switching(event, mask)


def test_runtime_terminal_event_clones_partial_reset_no_repeat_and_invalid_reference(monkeypatch):
    state = PGMTState.__new__(PGMTState)
    net = forces(True, n=2)
    state.env = SimpleNamespace(common_step_counter=1, step_dt=.02,
                               scene={"contact_forces": SimpleNamespace(data=SimpleNamespace(net_forces_w=net))})
    state.contact_tracker = ContactEventState(2, device="cpu", dt=.02)
    state.contact_switching_enabled = True
    state._contact_switching_live = empty_contact_switching(2, device="cpu")
    state._contact_switching_reward = state._reward_contacts = state._contact_reward_step = None
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
                    body_lin_vel=torch.zeros(2, 2, 3), body_ang_vel=torch.zeros(2, 2, 3))
    state.reference_contacts = MethodType(lambda self: labels, state)
    state.robot_foot_terrain_forces = MethodType(lambda self: torch.zeros_like(net), state)
    state.robot_state = MethodType(lambda self: physical, state)
    state.reference = MethodType(lambda self: physical, state)
    calls = []
    def old_rewards(*args, **kwargs):
        calls.append(1)
        return {"upper_fixture": torch.zeros(2)}
    monkeypatch.setattr("pmt_tasks.pgmt_runtime.reward_terms", old_rewards)
    assert not state.contact_switching_state()["valid"].any()
    for step in range(1, 7):
        state.env.common_step_counter = step
        with pytest.raises(RuntimeError, match="this control step"):
            state.contact_switching_state(for_reward=True)
        net.copy_(forces(step % 2 == 1, n=2))
        assert state.rewards()["terrain_contact_switching"].tolist() == ([2., 2.] if step == 6 else [0., 0.])
    state.reset(torch.tensor([0]))
    labels["frame_ids"][0], labels["motion_ids"][0] = 0, 1
    net.copy_(forces(True, n=2))
    for _ in range(3):
        assert state.rewards()["terrain_contact_switching"].tolist() == [2., 2.]
        snapshot = state.contact_switching_state(for_reward=True)
        assert snapshot["reference_frame_ids"].tolist() == [8, 12]
        assert snapshot["reference_motion_ids"].tolist() == [0, 1]
        assert snapshot["contact_ended"].all() and snapshot["previous_contact_age"].eq(0).all()
        snapshot["rate"].zero_(); snapshot["fast_contact_switch"].zero_()
    assert len(calls) == 6
    live = state.contact_switching_state()
    assert live["rate"].tolist() == [0., 2.] and not live["valid"][0].any()
    assert live["episode_step"].tolist() == [0, 6]
    assert state.contact_switching_state(env_ids=torch.tensor([1]))["rate"].item() == 2.
    state.env.common_step_counter = 7
    assert state.rewards()["terrain_contact_switching"].tolist() == [0., 2.]
    state.env.common_step_counter = 8
    assert state.rewards()["terrain_contact_switching"].tolist() == [0., 0.]
    assert len(calls) == 8
    with pytest.raises(RuntimeError, match="not configured"):
        PGMTState.__new__(PGMTState).contact_switching_state()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_cuda_event_replay_parity():
    trackers = [ContactEventState(8192, device=device, dt=.02) for device in ("cpu", "cuda:0")]
    g = torch.Generator().manual_seed(715)
    for step in range(1, 25):
        net = torch.randn(8192, 2, 3, generator=g) * 8.
        if step == 11: net[3, 0, 0] = float("nan")
        if step == 13:
            for tracker in trackers: tracker.reset([0, 5])
        for tracker in trackers:
            tracker.update(step, net.to(tracker.device), torch.zeros_like(net, device=tracker.device))
        cpu, cuda = map(cost, trackers)
        for key in cpu:
            torch.testing.assert_close(cpu[key], cuda[key].cpu())
