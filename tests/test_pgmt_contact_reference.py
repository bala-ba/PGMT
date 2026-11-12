"""Contact targets retain frame validity, left/right order and native reset alignment."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pmt_tasks.pgmt_contact_reference import CONTACT_FOOT_NAMES, ContactLabelStore
from pmt_tasks.pgmt_runtime import PGMTState


def motion(name, contacts, valid, feet=CONTACT_FOOT_NAMES):
    return SimpleNamespace(source_file=name, num_frames=len(contacts), metadata=dict(
        contacts=np.asarray(contacts, dtype=np.float32), contact_valid=np.asarray(valid, dtype=bool),
        foot_names=np.asarray(feet)))


def fixtures():
    return [motion("a.npz", [[1, 0], [0, 1], [1, 1]], [[1, 1], [0, 1], [1, 0]]),
            motion("b.npz", [[1, 0], [0, 1]], [[1, 1], [1, 0]], feet=CONTACT_FOOT_NAMES[::-1])]


def test_source_order_foot_name_reordering_and_invalid_labels_are_not_swing():
    store = ContactLabelStore(fixtures(), device="cpu")
    batch = store.gather(torch.tensor([1, 0, 0]), torch.tensor([0, 1, 2]))
    assert store.source_files == ("a.npz", "b.npz")
    assert batch["contacts"].tolist() == [[0., 1.], [0., 1.], [1., 1.]]
    assert batch["contact_valid"].tolist() == [[True, True], [False, True], [True, False]]
    assert batch["stance"].tolist() == [[False, True], [False, True], [True, False]]
    assert batch["swing"].tolist() == [[True, False], [False, False], [False, False]]
    assert batch["foot_state"].tolist() == [[0, 1], [-1, 1], [1, -1]]
    assert batch["support_mode"].tolist() == [2, -1, -1]
    assert batch["contacts"].dtype == torch.float32
    assert batch["contact_valid"].dtype == torch.bool
    assert batch["foot_state"].dtype == batch["support_mode"].dtype == torch.int8


def test_future_clamps_each_motion_but_masks_exhausted_frames_and_negative_requests():
    store = ContactLabelStore(fixtures(), device="cpu")
    batch = store.gather(torch.tensor([[0, 0, 0], [1, 1, 1]]), torch.tensor([[1, 2, 99], [-1, 1, 2]]))
    assert batch["frame_ids"].tolist() == [[1, 2, 2], [0, 1, 1]]
    assert batch["requested_frame_ids"].tolist() == [[1, 2, 99], [-1, 1, 2]]
    assert batch["frame_valid"].tolist() == [[True, True, False], [False, True, False]]
    assert not batch["contact_valid"][~batch["frame_valid"]].any()
    assert not batch["stance"][~batch["frame_valid"]].any()
    assert not batch["swing"][~batch["frame_valid"]].any()
    assert (batch["foot_state"][~batch["frame_valid"]] == -1).all()


def test_neither_foot_contact_is_valid_and_does_not_mean_unknown():
    store = ContactLabelStore([motion("crawl", [[0, 0]], [[1, 1]])], device="cpu")
    result = store.gather(torch.tensor([0]), torch.tensor([0]))
    assert result["contact_valid"].all()
    assert result["support_mode"].item() == 0
    assert result["swing"].all()  # Foot non-contact; says nothing about knee/hand support.


@pytest.mark.parametrize("corruption,match", [
    (lambda m: m.metadata.pop("contact_valid"), "Missing contact metadata"),
    (lambda m: m.metadata.update(foot_names=np.array(["left", "right"])), "named left/right"),
    (lambda m: m.metadata.update(foot_names=np.array([CONTACT_FOOT_NAMES[0]] * 2)), "named left/right"),
    (lambda m: m.metadata.update(contacts=np.zeros((2, 2))), "unchunked motion shape"),
    (lambda m: m.metadata["contacts"].__setitem__((0, 0), np.nan), "finite binary"),
    (lambda m: m.metadata.update(contact_valid=np.full((3, 2), 2)), "finite binary"),
])
def test_malformed_or_unsliced_metadata_fails_at_load(corruption, match):
    item = fixtures()[0]
    corruption(item)
    with pytest.raises(ValueError, match=match):
        ContactLabelStore([item], device="cpu")


def runtime_fixture():
    state = PGMTState.__new__(PGMTState)
    state.contact_label_store = ContactLabelStore(fixtures(), device="cpu")
    state.command = SimpleNamespace(motion_ids=torch.tensor([0, 1]), frame_ids=torch.tensor([0, 0]))
    state.full_store = SimpleNamespace(motion_lengths=state.contact_label_store.motion_lengths)
    state.pending_reset = torch.zeros(2, dtype=torch.bool)
    state.previous_target = torch.zeros(2, 29)
    state.robot = SimpleNamespace(data=SimpleNamespace(joint_pos_target=torch.zeros(2, 29)))
    state._obs_version = 0
    state.env = SimpleNamespace(common_step_counter=12)
    return state


def test_runtime_partial_reset_and_motion_switch_have_no_same_step_stale_cache():
    state = runtime_fixture()
    first = state.reference_contacts()
    state.command.motion_ids[0] = 1
    state.command.frame_ids[0] = 1
    state.reset(torch.tensor([0]))
    second = state.reference_contacts()
    assert second["contact_valid"][0].tolist() == [False, True]
    assert first["contact_valid"][0].tolist() == [True, True]
    for key in first:
        torch.testing.assert_close(first[key][1], second[key][1])
    # Returned tensors are snapshots; mutating them cannot alter command ids or
    # packed labels, which could otherwise poison later rewards after reset.
    second["contacts"].fill_(99)
    second["motion_ids"].fill_(99)
    assert state.command.motion_ids.tolist() == [1, 1]
    assert state.reference_contacts()["contacts"].max() <= 1


def test_runtime_current_future_subset_alignment_and_advancement():
    state = runtime_fixture()
    current = state.reference_contacts()
    future = state.reference_contacts(future=True)
    assert future["contacts"].shape == (2, 8, 2)
    assert future["requested_frame_ids"][0].tolist() == [0, 1, 3, 7, 15, 31, 63, 127]
    for key in current:
        torch.testing.assert_close(future[key][:, 0], current[key])
    assert not future["contact_valid"][:, 2:].any()
    subset = state.reference_contacts(future=True, env_ids=torch.tensor([1, 0]))
    torch.testing.assert_close(subset["contacts"], future["contacts"].flip(0))
    state.command.frame_ids += 1
    assert state.reference_contacts()["frame_ids"].tolist() == [1, 1]
    assert current["frame_ids"].tolist() == [0, 0]


def test_sensor_order_is_explicit_and_stage1_missing_labels_fail_only_on_request():
    state = runtime_fixture()
    forces = torch.arange(2 * 4 * 3).reshape(2, 4, 3).float()
    state.env.scene = {"contact_forces": SimpleNamespace(data=SimpleNamespace(net_forces_w=forces))}
    state.foot_sensor_indices = torch.tensor([3, 1])
    actual = state.robot_foot_contact_forces(env_ids=torch.tensor([1]))
    torch.testing.assert_close(actual, forces[[1]][:, [3, 1]])
    state.contact_label_store = None
    with pytest.raises(RuntimeError, match="labeled motion"):
        state.reference_contacts()


def test_empty_batches_and_invalid_motion_ids_do_not_wrap_into_last_motion():
    store = ContactLabelStore(fixtures(), device="cpu")
    empty = torch.empty(0, dtype=torch.long)
    assert store.gather(empty, empty)["contacts"].shape == (0, 2)
    with pytest.raises(IndexError):
        store.gather(torch.tensor([-1]), torch.tensor([0]))
