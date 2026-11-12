"""T4 strict force boundary, independent foot counting and reward lifecycle."""
from types import MethodType, SimpleNamespace

import pytest
import torch

from pmt_tasks.pgmt_contact_state import ContactEventState
from pmt_tasks.pgmt_runtime import PGMTState
from pmt_tasks.pgmt_stumble import (
    STUMBLE_CONFIG, STUMBLE_WEIGHT, empty_stumble, evaluate_stumble, validate_stumble_config,
)


@pytest.mark.parametrize("force,expected", [
    ([0., 0., 0.], False), ([0., 0., 500.], False),
    ([10., 0., 0.], False), ([10.001, 0., 0.], True),
    ([12., 16., 2.], False), ([12., 16.001, 2.], True),
    ([12., 16., -2.], False), ([12., 16.001, -2.], True),
    ([-21., 0., -2.], True),
])
def test_strict_boundary_vertical_absolute_value_and_no_force_norm_substitution(force, expected):
    forces = torch.tensor([[force, [0., 0., 0.]]])
    result = evaluate_stumble(forces, torch.ones(1, 2, dtype=torch.bool))
    assert result["valid"].all()
    assert result["stumble"].tolist() == [[expected, False]]
    assert result["rate"].item() == float(expected)
    assert (result["rate"] * STUMBLE_WEIGHT * .02).item() == pytest.approx(-.4 * expected)


def test_each_foot_sum_yaw_invariance_and_no_reference_or_support_gate():
    forces = torch.tensor([[[12., 16., 0.], [0., -11., 0.]], [[0., 0., 0.], [0., 11., 0.]]])
    tracker = ContactEventState(2, device="cpu", dt=.02)
    for step in (1, 2, 3):
        tracker.update(step, forces, torch.zeros_like(forces))
        s = tracker.snapshot()
        assert not s["support"].any() and not s["touchdown"].any()
        result = evaluate_stumble(s["net_forces_w"], s["contact_valid"])
        assert result["rate"].tolist() == [2., 1.]
        torch.testing.assert_close(result["rate"] * STUMBLE_WEIGHT * .02, torch.tensor([-.8, -.4]))
    rotated = forces.clone()
    rotated[..., 0], rotated[..., 1] = -forces[..., 1], forces[..., 0]
    torch.testing.assert_close(evaluate_stumble(rotated, s["contact_valid"])["rate"], result["rate"])
    result["forces_w"].zero_()
    assert forces.abs().sum() > 0


@pytest.mark.parametrize("component", [0, 1, 2])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1e38])
def test_unknown_or_overflow_is_invalid_not_healthy(component, value):
    forces = torch.tensor([[[20., 0., 0.], [20., 0., 0.]]])
    forces[0, 0, component] = value
    result = evaluate_stumble(forces, torch.ones(1, 2, dtype=torch.bool))
    assert result["valid"].tolist() == [[False, True]]
    assert result["threshold"][0, 0].isnan() and result["horizontal_force"][0, 0].isnan()
    assert result["rate"].item() == 1.
    assert result["rate"].isfinite().all()


def test_measurement_mask_config_and_disabled_api():
    forces = torch.tensor([[[20., 0., 0.], [20., 0., 2.]]])
    mask = torch.tensor([[False, True]])
    result = evaluate_stumble(forces, mask)
    assert result["valid"].tolist() == [[False, True]] and result["rate"].item() == 0.
    result = evaluate_stumble(forces, mask, dict(vertical_ratio=2., bias_n=1.))
    assert result["rate"].item() == 1.
    with pytest.raises(RuntimeError, match="not configured"):
        PGMTState.__new__(PGMTState).stumble_state()


@pytest.mark.parametrize("config", [False, {}, {"vertical_ratio": 5.},
    dict(vertical_ratio=True, bias_n=10.), dict(vertical_ratio=-1., bias_n=10.),
    dict(vertical_ratio=float("nan"), bias_n=10.), dict(vertical_ratio=5., bias_n=float("inf")),
    dict(vertical_ratio=5., bias_n=-10.), dict(vertical_ratio=5., bias_n="10"),
    dict(vertical_ratio=5., bias_n=10., weight=-20.)])
def test_bad_configuration_rejected(config):
    with pytest.raises(ValueError, match="Stumble"):
        validate_stumble_config(config)


@pytest.mark.parametrize("bad", ["mask_dtype", "force_dtype", "force_shape", "mask_shape"])
def test_input_contract_rejected(bad):
    forces, mask = torch.zeros(2, 2, 3), torch.ones(2, 2, dtype=torch.bool)
    if bad == "mask_dtype": mask = mask.float()
    if bad == "force_dtype": forces = forces.long()
    if bad == "force_shape": forces = forces[..., :2]
    if bad == "mask_shape": mask = mask[:, :1]
    with pytest.raises(ValueError, match="Stumble requires"):
        evaluate_stumble(forces, mask)


def test_runtime_frozen_actual_force_terminal_reset_and_continuous_cost(monkeypatch):
    state = PGMTState.__new__(PGMTState)
    net = torch.zeros(2, 2, 3); net[..., 0] = 100.
    state.env = SimpleNamespace(common_step_counter=1, step_dt=.02,
                               scene={"contact_forces": SimpleNamespace(data=SimpleNamespace(net_forces_w=net))})
    state.contact_tracker = ContactEventState(2, device="cpu", dt=.02)
    state.stumble_config = STUMBLE_CONFIG.copy()
    state._stumble_live = empty_stumble(2, device="cpu")
    state._stumble_reward = state._reward_contacts = state._contact_reward_step = None
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
    assert state.stumble_state()["rate"].eq(0).all() and not state.stumble_state()["valid"].any()
    with pytest.raises(RuntimeError, match="this control step"):
        state.stumble_state(for_reward=True)
    assert state.rewards()["terrain_stumble"].tolist() == [2., 2.]
    assert not state.contact_state()["touchdown"].any()
    state.reset(torch.tensor([0]))
    net.zero_()
    labels["frame_ids"][0], labels["motion_ids"][0] = 0, 1
    for _ in range(3):
        assert state.rewards()["terrain_stumble"].tolist() == [2., 2.]
        snapshot = state.stumble_state(for_reward=True)
        assert snapshot["reference_frame_ids"].tolist() == [8, 12]
        assert snapshot["reference_motion_ids"].tolist() == [0, 1]
        snapshot["rate"].zero_(); snapshot["forces_w"].zero_()
    assert len(calls) == 1
    live = state.stumble_state()
    assert live["rate"].tolist() == [0., 2.]
    assert live["forces_w"][0].isnan().all() and not live["valid"][0].any()
    assert state.stumble_state(env_ids=torch.tensor([1]))["rate"].item() == 2.
    for step in (2, 3):
        state.env.common_step_counter = step
        with pytest.raises(RuntimeError, match="this control step"):
            state.stumble_state(for_reward=True)
        net[..., 0] = 20.
        assert state.rewards()["terrain_stumble"].tolist() == [2., 2.]
    assert len(calls) == 3
    state.env.common_step_counter = 4
    net.zero_()
    assert state.rewards()["terrain_stumble"].eq(0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_cuda_cost_and_masks_parity():
    g = torch.Generator().manual_seed(714)
    forces = torch.randn(8192, 2, 3, generator=g) * 100.
    mask = torch.rand(8192, 2, generator=g) > .2
    forces[0, 0, 0] = float("nan")
    cpu, cuda = evaluate_stumble(forces, mask), evaluate_stumble(forces.cuda(), mask.cuda())
    for key in cpu:
        torch.testing.assert_close(cpu[key], cuda[key].cpu(), equal_nan=True, rtol=1e-6, atol=1e-5)
