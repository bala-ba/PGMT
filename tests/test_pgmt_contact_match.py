"""T2 binary agreement, unknown data, force semantics and reset timeline."""
from itertools import product
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from pmt_tasks.pgmt_contact_match import (
    REFERENCE_CONTACT_MATCH_WEIGHT, empty_contact_match, evaluate_reference_contact_match,
)
from pmt_tasks.pgmt_contact_reference import CONTACT_FOOT_NAMES, ContactLabelStore
from pmt_tasks.pgmt_contact_state import ContactEventState
from pmt_tasks.pgmt_runtime import PGMTState


def test_all_binary_state_and_reference_mask_combinations():
    rows = list(product(product((False, True), repeat=2), repeat=3))
    actual = torch.tensor([r[0] for r in rows])
    reference = torch.tensor([r[1] for r in rows], dtype=torch.float32)
    masks = torch.tensor([r[2] for r in rows])
    result = evaluate_reference_contact_match(actual, torch.ones_like(actual), reference, masks)
    expected = [sum(m and a == r for a, r, m in zip(*row)) / max(sum(row[2]), 1) for row in rows]
    torch.testing.assert_close(result["rate"], torch.tensor(expected))
    assert set(result["rate"].tolist()) == {0., .5, 1.}
    assert result["rate"].max() * REFERENCE_CONTACT_MATCH_WEIGHT * .02 == pytest.approx(.03)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -1., .5])
def test_nonbinary_or_unknown_reference_never_matches(value):
    actual = torch.tensor([[False, True], [False, False]])
    reference = torch.tensor([[value, 1.], [value, value]])
    result = evaluate_reference_contact_match(actual, torch.ones_like(actual), reference, torch.ones_like(actual))
    assert result["reference_valid"].tolist() == [[False, True], [False, False]]
    assert result["reference_count"].tolist() == [1, 0]
    assert result["rate"].tolist() == [1., 0.]


def test_unknown_physical_measurement_cannot_match_swing_or_shrink_denominator():
    actual = torch.tensor([[False, True], [False, False]])
    actual_valid = torch.tensor([[False, True], [False, False]])
    result = evaluate_reference_contact_match(actual, actual_valid, actual.float(), torch.ones_like(actual))
    assert result["compared_valid"].tolist() == actual_valid.tolist()
    assert result["reference_count"].tolist() == [2, 2]
    assert result["rate"].tolist() == [.5, 0.]


def test_all_object_norm_including_side_downward_and_nonterrain_forces():
    state = ContactEventState(4, device="cpu", dt=.02)
    net = torch.tensor([[[100., 0., 0.]]*2, [[0., 0., -100.]]*2,
                        [[0., 0., 100.]]*2, [[0., 0., 10.], [0., 0., 10.01]]])
    for step in (1, 2, 3):
        state.update(step, net, torch.zeros_like(net))
        snapshot = state.snapshot()
        result = evaluate_reference_contact_match(snapshot["contact"], snapshot["contact_valid"],
                                                  torch.ones(4, 2), torch.ones(4, 2, dtype=torch.bool))
        assert result["rate"].tolist() == [1., 1., 1., .5]
        assert not snapshot["support"].any() and not snapshot["touchdown"].any()
    # A nonfinite force is unknown, not inferred to be a valid no-contact state.
    net[0, 0, 1] = float("nan")
    state.update(4, net, torch.zeros_like(net))
    s = state.snapshot()
    assert evaluate_reference_contact_match(s["contact"], s["contact_valid"],
                                            torch.ones(4, 2), torch.ones(4, 2, dtype=torch.bool))["rate"][0] == .5


def test_label_store_motion_order_and_out_of_segment_frames():
    motions = [SimpleNamespace(source_file=f"motion{i}", num_frames=2, metadata=dict(
        foot_names=np.array(CONTACT_FOOT_NAMES[::-1]), contacts=np.array(labels), contact_valid=np.ones((2, 2))))
        for i, labels in enumerate(([[0, 1], [1, 0]], [[1, 0], [0, 1]]))]
    store = ContactLabelStore(motions, device="cpu")
    labels = store.gather(torch.tensor([0, 0, 1, 1, 0]), torch.tensor([0, 1, 0, 1, 2]))
    actual = torch.tensor([[True, False]]*5)
    result = evaluate_reference_contact_match(actual, torch.ones_like(actual), labels["contacts"], labels["contact_valid"])
    assert result["rate"].tolist() == [1., 0., 0., 1., 0.]


@pytest.mark.parametrize("bad", ["shape", "mask_dtype", "state_dtype"])
def test_invalid_input_contract_rejected(bad):
    actual = torch.ones(2, 2, dtype=torch.bool)
    valid, ref, ref_valid = actual.clone(), actual.float(), actual.clone()
    if bad == "shape": ref = ref[:, :1]
    if bad == "mask_dtype": valid = valid.float()
    if bad == "state_dtype": actual = actual.float()
    with pytest.raises(ValueError, match="paired"):
        evaluate_reference_contact_match(actual, valid, ref, ref_valid)


def test_runtime_cache_same_frame_continuous_reward_and_partial_terminal_reset(monkeypatch):
    runtime = PGMTState.__new__(PGMTState)
    forces = torch.zeros(2, 2, 3)
    forces[..., 0] = 100.  # Side forces match contact; terrain support remains false.
    runtime.env = SimpleNamespace(common_step_counter=1, step_dt=.02,
                                  scene={"contact_forces": SimpleNamespace(data=SimpleNamespace(net_forces_w=forces))})
    runtime.contact_tracker = ContactEventState(2, device="cpu", dt=.02)
    runtime.contact_match_enabled = True
    runtime._contact_match_live = empty_contact_match(2, device="cpu")
    runtime._contact_match_reward = runtime._reward_contacts = runtime._contact_reward_step = None
    runtime._contact_context, runtime._reward_step, runtime._obs_version = {}, None, 0
    runtime.contact_order = runtime.foot_body_indices = torch.tensor([0, 1])
    runtime.anchor, runtime.config, runtime.groups = 0, {}, {}
    runtime.pending_reset = torch.ones(2, dtype=torch.bool)
    runtime.previous_target = torch.zeros(2, 29)
    runtime.previous_velocity = torch.zeros(2, 2, 3)
    runtime.previous_ref_velocity = torch.zeros(2, 2, 3)
    runtime.robot = SimpleNamespace(data=SimpleNamespace(joint_pos_target=torch.zeros(2, 29),
                                                         soft_joint_pos_limits=torch.zeros(2, 29, 2)))
    labels = dict(contacts=torch.tensor([[1., 0.], [1., 1.]]), contact_valid=torch.ones(2, 2, dtype=torch.bool),
                  motion_ids=torch.tensor([0, 1]), frame_ids=torch.tensor([8, 12]))
    runtime._contact_context = {f"reference_{key}": value.clone() for key, value in labels.items()}
    runtime._invalidate_contact_context(slice(None))
    physical = dict(body_pos=torch.zeros(2, 2, 3), body_quat=torch.zeros(2, 2, 4),
                    body_lin_vel=torch.zeros(2, 2, 3), body_ang_vel=torch.zeros(2, 2, 3))
    runtime.reference_contacts = MethodType(lambda self: labels, runtime)
    runtime.robot_foot_terrain_forces = MethodType(lambda self: torch.zeros_like(forces), runtime)
    runtime.robot_state = MethodType(lambda self: physical, runtime)
    runtime.reference = MethodType(lambda self: physical, runtime)
    original_calls = []
    def old_rewards(*args, **kwargs):
        original_calls.append(1)
        return {"upper_fixture": torch.zeros(2)}
    monkeypatch.setattr("pmt_tasks.pgmt_runtime.reward_terms", old_rewards)
    assert runtime.reference_contact_match_state()["rate"].eq(0).all()
    with pytest.raises(RuntimeError, match="this control step"):
        runtime.reference_contact_match_state(for_reward=True)
    assert runtime.rewards()["terrain_reference_contact_match"].tolist() == [.5, 1.]
    assert not runtime.contact_state()["touchdown"].any()  # No event/history gate for T2.
    runtime.reset(torch.tensor([0]))
    labels["frame_ids"][0], labels["motion_ids"][0] = 0, 1
    labels["contacts"].zero_()  # Command has advanced/reset; do not reread these labels this step.
    for _ in range(3):
        assert runtime.rewards()["terrain_reference_contact_match"].tolist() == [.5, 1.]
        frozen = runtime.reference_contact_match_state(for_reward=True)
        assert frozen["reference_frame_ids"].tolist() == [8, 12]
        assert frozen["reference_motion_ids"].tolist() == [0, 1]
        frozen["rate"].zero_(); frozen["matched"].zero_()
    assert len(original_calls) == 1
    live = runtime.reference_contact_match_state()
    assert live["rate"].tolist() == [0., 1.] and not live["compared_valid"][0].any()
    assert runtime.reference_contact_match_state(env_ids=torch.tensor([1]))["rate"].tolist() == [1.]
    runtime.env.common_step_counter += 1
    with pytest.raises(RuntimeError, match="this control step"):
        runtime.reference_contact_match_state(for_reward=True)
    assert runtime.rewards()["terrain_reference_contact_match"].eq(0).all()
    for step in (3, 4):
        runtime.env.common_step_counter = step
        labels["contacts"].fill_(1)
        assert runtime.rewards()["terrain_reference_contact_match"].eq(1).all()
    assert len(original_calls) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_cuda_match_and_masks_parity():
    g = torch.Generator().manual_seed(710)
    inputs = [torch.rand(8192, 2, generator=g) > .4 for _ in range(4)]
    cpu = evaluate_reference_contact_match(*inputs)
    gpu = evaluate_reference_contact_match(*(v.cuda() for v in inputs))
    for key in cpu:
        torch.testing.assert_close(cpu[key], gpu[key].cpu(), rtol=0, atol=0)
