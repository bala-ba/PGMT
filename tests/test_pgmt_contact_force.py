"""T6 all-axis per-foot squared force, no thresholds, and frozen lifecycle."""
from types import MethodType, SimpleNamespace

import pytest
import torch

from pmt_tasks.pgmt_contact_state import ContactEventState
from pmt_tasks.pgmt_contact_force import CONTACT_FORCE_WEIGHT, empty_contact_force, evaluate_contact_force
from pmt_tasks.pgmt_runtime import PGMTState


def test_three_axis_squared_force_sum_and_independent_diagnostic_copy():
    forces = torch.tensor([[[3., 4., 12.], [1., -2., -2.]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    result = evaluate_contact_force(forces, mask)
    assert result["force_sq"].tolist() == [[169., 9.]]
    assert result["rate"].item() == 178.
    assert (result["rate"] * CONTACT_FORCE_WEIGHT * .02).item() == pytest.approx(-3.56e-6)
    result["forces_w"].zero_(); result["force_valid"].zero_()
    assert forces[0, 0, 2] == 12. and mask.all()


@pytest.mark.parametrize("magnitude", [0., .001, 5., 10., 10.01, 350., 450.])
def test_no_contact_threshold_force_allowance_or_cap(magnitude):
    forces = torch.tensor([[[0., 0., magnitude], [0., 0., 0.]]])
    tracker = ContactEventState(1, device="cpu", dt=.02)
    for step in (1, 2, 3):
        tracker.update(step, forces, torch.zeros_like(forces))
        sample = tracker.snapshot()
        result = evaluate_contact_force(sample["net_forces_w"], sample["contact_valid"])
        assert result["rate"].item() == pytest.approx(magnitude**2)
        assert (result["rate"] * CONTACT_FORCE_WEIGHT * .02).item() == pytest.approx(-2e-8 * magnitude**2)
        assert not sample["support"].any() and not sample["touchdown"].any()
        if 0. < magnitude <= 10.:
            assert not sample["contact"].any() and result["rate"].item() > 0.


def test_opposite_foot_forces_do_not_cancel_and_rotations_do_not_change_cost():
    forces = torch.tensor([[[100., 0., 0.], [-100., 0., 0.]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    assert evaluate_contact_force(forces, mask)["rate"].item() == 20000.
    forces = torch.tensor([[[3., 4., 12.], [1., -2., -2.]]], dtype=torch.float64)
    rotated = forces[..., [2, 0, 1]].clone()
    rotated[..., :2] *= -1
    torch.testing.assert_close(evaluate_contact_force(rotated, mask)["rate"], evaluate_contact_force(forces, mask)["rate"])


@pytest.mark.parametrize("component", [0, 1, 2])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), 1e30])
def test_nonfinite_or_squared_force_overflow_is_unknown_and_isolated(component, value):
    forces = torch.tensor([[[3., 4., 0.], [3., 4., 0.]]])
    forces[0, 0, component] = value
    result = evaluate_contact_force(forces, torch.ones(1, 2, dtype=torch.bool))
    assert result["valid"].tolist() == [[False, True]]
    assert result["force_sq"][0, 0].isnan()
    assert result["rate"].item() == 25. and result["rate"].isfinite().all()


def test_measurement_mask_and_aggregate_overflow_are_explicitly_invalid():
    forces = torch.tensor([[[3., 4., 0.], [0., 0., 0.]]])
    result = evaluate_contact_force(forces, torch.tensor([[False, True]]))
    assert result["valid"].tolist() == [[False, True]]
    assert result["force_sq"][0, 0].isnan() and result["force_sq"][0, 1] == 0.
    assert result["rate"].item() == 0.
    result = evaluate_contact_force(torch.tensor([[[1.4e19, 0., 0.]]*2]), torch.ones(1, 2, dtype=torch.bool))
    assert result["force_sq"].isfinite().all()  # Each foot fits, but the combined float32 rate does not.
    assert not result["valid"].any() and result["rate"].item() == 0.


@pytest.mark.parametrize("bad", ["mask_dtype", "force_dtype", "force_shape", "mask_shape"])
def test_input_contract_rejected(bad):
    forces, mask = torch.zeros(2, 2, 3), torch.ones(2, 2, dtype=torch.bool)
    if bad == "mask_dtype": mask = mask.float()
    if bad == "force_dtype": forces = forces.long()
    if bad == "force_shape": forces = forces[..., :2]
    if bad == "mask_shape": mask = mask[:, :1]
    with pytest.raises(ValueError, match="Contact force requires"):
        evaluate_contact_force(forces, mask)


def test_runtime_weak_force_first_sample_continuous_cost_and_terminal_cache(monkeypatch):
    state = PGMTState.__new__(PGMTState)
    net = torch.tensor([[[3., 4., 0.], [0., 0., 0.]], [[450., 0., 0.], [0., 0., 0.]]])
    state.env = SimpleNamespace(common_step_counter=1, step_dt=.02,
                               scene={"contact_forces": SimpleNamespace(data=SimpleNamespace(net_forces_w=net))})
    state.contact_tracker = ContactEventState(2, device="cpu", dt=.02)
    state.contact_force_enabled = True
    state._contact_force_live = empty_contact_force(2, device="cpu")
    state._contact_force_reward = state._reward_contacts = state._contact_reward_step = None
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
    assert not state.contact_force_state()["valid"].any() and state.contact_force_state()["rate"].eq(0).all()
    with pytest.raises(RuntimeError, match="this control step"):
        state.contact_force_state(for_reward=True)
    assert state.rewards()["terrain_contact_force"].tolist() == [25., 202500.]
    assert not state.contact_state()["contact"][0].any()
    assert not state.contact_state()["touchdown"].any()
    state.reset(torch.tensor([0]))
    net.zero_()
    labels["frame_ids"][0], labels["motion_ids"][0] = 0, 1
    for _ in range(3):
        assert state.rewards()["terrain_contact_force"].tolist() == [25., 202500.]
        snapshot = state.contact_force_state(for_reward=True)
        assert snapshot["reference_frame_ids"].tolist() == [8, 12]
        assert snapshot["reference_motion_ids"].tolist() == [0, 1]
        snapshot["rate"].zero_(); snapshot["forces_w"].zero_()
    assert len(calls) == 1
    live = state.contact_force_state()
    assert live["rate"].tolist() == [0., 202500.]
    assert live["forces_w"][0].isnan().all() and not live["valid"][0].any()
    assert state.contact_force_state(env_ids=torch.tensor([1]))["rate"].item() == 202500.
    for step in (2, 3):
        state.env.common_step_counter = step
        with pytest.raises(RuntimeError, match="this control step"):
            state.contact_force_state(for_reward=True)
        net[..., 0] = 10.
        assert state.rewards()["terrain_contact_force"].tolist() == [200., 200.]
    assert len(calls) == 3
    state.env.common_step_counter = 4
    net.zero_()
    assert state.rewards()["terrain_contact_force"].eq(0).all()
    with pytest.raises(RuntimeError, match="not configured"):
        PGMTState.__new__(PGMTState).contact_force_state()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_cuda_cost_and_masks_parity():
    g = torch.Generator().manual_seed(716)
    forces = torch.randn(8192, 2, 3, generator=g) * 500.
    mask = torch.rand(8192, 2, generator=g) > .2
    forces[0, 0, 0] = float("nan")
    cpu, cuda = evaluate_contact_force(forces, mask), evaluate_contact_force(forces.cuda(), mask.cuda())
    for key in cpu:
        torch.testing.assert_close(cpu[key], cuda[key].cpu(), equal_nan=True, rtol=1e-6, atol=1e-5)
