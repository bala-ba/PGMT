"""PGMT contact/support timing, source filtering and reset boundaries."""
from types import SimpleNamespace, MethodType

import pytest
import torch

from pmt_tasks.pgmt_contact_state import (
    CONTACT_STATE_CONFIG, ContactEventState, terrain_force_from_matrix, terrain_contact_filters,
)
from pmt_tasks.pgmt_runtime import PGMTState


def forces(z, x=0., n=1, device="cpu"):
    out = torch.zeros(n, 2, 3, device=device)
    out[..., 0], out[..., 2] = x, z
    return out


def test_force_direction_filtering_and_exact_threshold():
    state = ContactEventState(3, device="cpu", dt=.02)
    net = forces(0., x=100., n=3)
    terrain = forces(0., n=3)
    terrain[0, :, 0] = 100.  # Side impact, no upward load.
    terrain[1, :, 2] = torch.tensor([10., 10.001])
    terrain[2, :, 2] = -100.  # Downward force is not support.
    state.update(1, net, terrain)
    result = state.snapshot()
    assert result["contact"].all()
    assert result["support"].tolist() == [[False, False], [False, True], [False, False]]
    assert not result["contact_started"].any() and not result["support_started"].any()
    assert not result["touchdown"].any() and not result["support_history_valid"].any()
    # All-object upward force can come from self collision; terrain channel wins.
    state.update(2, forces(200., n=3), forces(0., n=3))
    assert state.snapshot()["contact"].all()
    assert not state.snapshot()["support"].any()


def test_touchdown_two_non_support_steps_and_no_repeated_events():
    state = ContactEventState(1, device="cpu", dt=.02)
    low, high = forces(0), forces(100)
    state.update(1, low, low)
    assert state.snapshot()["non_support_time"][0, 0].item() == pytest.approx(.02)
    state.update(2, low, low)
    state.update(3, high, high)
    result = state.snapshot()
    assert result["touchdown"].all()
    assert result["previous_non_support_steps"].eq(2).all()
    assert result["support_age"].eq(0).all()
    assert result["support_time"][0, 0].item() == pytest.approx(.02)
    assert result["last_non_support_time"][0, 0].item() == pytest.approx(.04)
    state.update(4, high, high)
    assert not state.snapshot()["touchdown"].any()
    state.update(5, low, low)
    state.update(6, high, high)
    assert state.snapshot()["support_started"].all()
    assert not state.snapshot()["touchdown"].any()  # One unloaded step is rejected.


def test_switch_age_strict_boundary_and_initial_grace():
    state = ContactEventState(1, device="cpu", dt=.02)
    low, high = forces(0), forces(100)
    for step in range(1, 7):
        state.update(step, high, high)
    state.update(7, low, low)
    result = state.snapshot()
    assert result["contact_ended"].all()
    assert result["previous_contact_age"][0, 0].item() == pytest.approx(.1)
    assert not result["fast_contact_switch"].any()  # Exactly .1 is not < .1.
    state.update(8, high, high)
    assert state.snapshot()["fast_contact_switch"].all()
    state.reset([0])
    for step in range(9, 14):
        f = low if step % 2 else high
        state.update(step, f, f)
        assert not state.snapshot()["fast_contact_switch"].any()
    state.update(14, high, high)
    assert state.snapshot()["fast_contact_switch"].all()


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_invalid_force_breaks_only_affected_channel_without_false_edges(invalid):
    state = ContactEventState(1, device="cpu", dt=.02)
    high = forces(100)
    state.update(1, high, high)
    corrupt = high.clone()
    corrupt[0, 0, 0] = invalid
    state.update(2, high, corrupt)
    result = state.snapshot()
    assert result["support_valid"].tolist() == [[False, True]]
    assert result["contact_valid"].all()
    assert not result["support_ended"].any()
    state.update(3, high, high)
    assert not state.snapshot()["touchdown"].any()
    assert state.snapshot()["support_history_valid"].tolist() == [[False, True]]
    assert state.snapshot()["support_time"][0].tolist() == pytest.approx([.02, .06])
    state.update(4, corrupt, high)
    assert not state.snapshot()["contact_valid"][0, 0]
    assert state.snapshot()["support_valid"].all()


def test_partial_reset_same_step_isolation_and_snapshot_ownership():
    state = ContactEventState(2, device="cpu", dt=.02)
    low, high = forces(0, n=2), forces(100, n=2)
    state.update(20, low, low)
    state.update(21, low, low)
    state.update(22, high, high)
    frozen = state.snapshot()
    assert frozen["touchdown"].all()
    state.reset(torch.tensor([0]))
    assert not state.update(22, low, low)
    current = state.snapshot()
    assert current["sample_step"].tolist() == [-1, 22]
    assert current["touchdown"].tolist() == [[False, False], [True, True]]
    assert current["episode_step"].tolist() == [0, 3]
    assert current["episode_id"].tolist() == [1, 0]
    assert frozen["touchdown"].all()  # reset cannot rewrite old rewards.
    current["contact"].fill_(False)
    assert state.snapshot()["contact"][1].all()
    state.update(23, high, high)
    assert not state.snapshot()["touchdown"].any()
    assert state.snapshot()["support_history_valid"].tolist() == [[False, False], [True, True]]
    with pytest.raises(ValueError, match="consecutive"):
        state.update(25, high, high)
    with pytest.raises(ValueError, match="consecutive"):
        state.update(22, high, high)


def test_filter_matrix_sums_all_terrain_objects_and_rejects_missing_channel():
    matrix = torch.tensor([[[[1., 2., 8.], [3., 4., 7.]]]])
    torch.testing.assert_close(terrain_force_from_matrix(matrix, 1), torch.tensor([[4., 6., 15.]]))
    for bad in (None, torch.zeros(1, 2, 3), torch.zeros(1, 2, 1, 3), torch.zeros(1, 1, 0, 3)):
        with pytest.raises(ValueError):
            terrain_force_from_matrix(bad, 1)


@pytest.mark.parametrize("version,count", [("v5", 70), ("v6", 800)])
def test_filter_paths_cover_each_manifest_collider_without_wildcard_or_border(version, count):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    paths = terrain_contact_filters(root / f"assets/terrain/pgmt_stage2_{version}", "/World/ground")
    assert len(set(paths)) == len(paths) == count
    assert all(p.startswith("/World/ground/terrain/Terrain/") and p.endswith("/Mesh") for p in paths)
    assert not any("*" in p or "Border" in p for p in paths)


@pytest.mark.parametrize("key,value", [("contact_threshold_n", float("nan")), ("support_threshold_n", -1),
                                       ("fast_switch_seconds", 0), ("switch_grace_steps", 1.5),
                                       ("touchdown_min_non_support_steps", 0)])
def test_invalid_configuration_rejected(key, value):
    config = dict(CONTACT_STATE_CONFIG, **{key: value})
    with pytest.raises(ValueError):
        ContactEventState(1, device="cpu", dt=.02, config=config)


@pytest.mark.parametrize('fast', [False, True])
def test_runtime_reward_snapshot_keeps_terminal_reference_and_live_reset_is_invalid(fast):
    state = PGMTState.__new__(PGMTState)
    state.env = SimpleNamespace(common_step_counter=1)
    state.contact_tracker = ContactEventState(2, device="cpu", dt=.02)
    state._fast_reset_buffers = state.contact_tracker._fast_reset_buffers = fast
    state.foot_body_indices = torch.tensor([1, 0])
    state._contact_context = {}
    state._reward_contacts = None
    state.pending_reset = torch.zeros(2, dtype=torch.bool)
    state.previous_target = torch.zeros(2, 29)
    state.robot = SimpleNamespace(data=SimpleNamespace(joint_pos_target=torch.zeros(2, 29)))
    state._obs_version = 0
    terrain = forces(0, n=2)
    state.robot_foot_terrain_forces = MethodType(lambda self: terrain, state)
    labels = dict(contacts=torch.ones(2, 2), contact_valid=torch.ones(2, 2, dtype=torch.bool),
                  frame_ids=torch.tensor([3, 7]), motion_ids=torch.tensor([1, 2]))
    state.reference_contacts = MethodType(lambda self: labels, state)
    physical = {k: torch.ones(2, 2, size) for k, size in (
        ("body_pos", 3), ("body_quat", 4), ("body_lin_vel", 3), ("body_ang_vel", 3))}
    for step in range(1, 4):
        state.env.common_step_counter = step
        if step == 3:
            terrain.fill_(100.)
        state._update_contact_state(physical, terrain)
    frozen = state.contact_state(for_reward=True)
    assert frozen["touchdown"].all()
    state.reset(torch.tensor([0]))
    labels["frame_ids"][0] = 0
    current = state.contact_state()
    assert current["reference_frame_ids"].tolist() == [-1, 7]
    assert not current["support_valid"][0].any()
    assert current["foot_body_pos"][0].isnan().all()
    old = state.contact_state(for_reward=True)
    assert old["reference_frame_ids"].tolist() == [3, 7]
    assert old["touchdown"].all()
    old["touchdown"].fill_(False)
    assert state.contact_state(for_reward=True)["touchdown"].all()
    state.env.common_step_counter += 1
    with pytest.raises(RuntimeError, match="this control step"):
        state.contact_state(for_reward=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_cuda_replay_parity():
    cpu = ContactEventState(8, device="cpu", dt=.02)
    gpu = ContactEventState(8, device="cuda:0", dt=.02)
    generator = torch.Generator().manual_seed(73)
    for step in range(1, 41):
        net = torch.randn(8, 2, 3, generator=generator) * 12
        terrain = torch.randn(8, 2, 3, generator=generator) * 20
        if step == 13:
            terrain[3, 0, 2] = float("nan")
        if step == 20:
            cpu.reset([0, 5]); gpu.reset([0, 5])
        cpu.update(step, net, terrain)
        gpu.update(step, net.cuda(), terrain.cuda())
        for key, value in cpu.snapshot().items():
            torch.testing.assert_close(gpu.snapshot()[key].cpu(), value, equal_nan=True)
