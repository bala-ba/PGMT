"""Birth-based terrain progress, exclusive outcomes and restart-safe persistence."""
import copy
from types import SimpleNamespace

import pytest
import torch

from pmt_tasks.pgmt_curriculum import (SAMPLER_COUNTS, TerrainCurriculum,
                                      load_training_state_dict, training_state_dict, _identity)


def fixture(n=6):
    origins = torch.zeros(10, 7, 3)
    origins[..., 0] = torch.arange(10)[:, None] * 24
    origins[..., 1] = torch.arange(7)[None] * 24
    origins[..., 2] = -torch.arange(10)[:, None]
    levels, families = torch.full((n,), 5), torch.arange(n) % 7
    terrain = SimpleNamespace(terrain_levels=levels, terrain_types=families, terrain_origins=origins,
                              env_origins=origins[levels, families].clone())
    return TerrainCurriculum(terrain, dict(terrain="test", motion="test"))


def begin(state, *, step=0, start=0, length=499):
    ids = torch.arange(state.num_envs)
    state.begin(ids, step, ids, torch.full_like(ids, start), torch.full_like(ids, length))
    return ids


def settle(state, *, step=498, **overrides):
    ids = torch.arange(state.num_envs)
    values = dict(step=step, motion_ids=ids, frame_ids=torch.full_like(ids, 498),
                  **{key: torch.zeros_like(ids, dtype=torch.bool) for key in
                     ("failure", "finished", "map_exit", "timeout", "other")})
    for key, value in overrides.items():
        values[key] = torch.as_tensor(value, dtype=values[key].dtype)
    state.settle(ids, **values)


def test_exclusive_success_failure_and_neutral_reasons_update_birth_level_and_origins():
    state = fixture()
    begin(state)
    state.start_frame[4] = 249
    settle(state, finished=[1, 1, 1, 0, 1, 1], failure=[0, 1, 0, 0, 0, 0],
           map_exit=[0, 0, 1, 0, 0, 0], timeout=[1, 0, 0, 1, 0, 0], other=[0, 0, 0, 0, 0, 1])
    assert state.last_outcome.tolist() == [1, 2, 3, 4, 5, 6]
    assert state.terrain.terrain_levels.tolist() == [6, 4, 5, 5, 5, 5]
    assert state.counts.sum((0, 1)).tolist() == [1] * 6
    assert state.promotions[0, 5] == 1 and state.demotions[1, 5] == 1
    torch.testing.assert_close(state.terrain.env_origins,
        state.terrain.terrain_origins[state.terrain.terrain_levels, state.terrain.terrain_types])


def test_caps_and_duplicate_settlement_and_same_step_reset_do_not_double_count():
    state = fixture(2)
    state.terrain.terrain_levels[:] = torch.tensor([9, 0])
    begin(state)
    settle(state, finished=[1, 1], failure=[0, 1])
    settle(state, finished=[1, 1], failure=[0, 1])
    begin(state, step=498)
    settle(state, finished=[1, 1], failure=[0, 1])
    assert state.terrain.terrain_levels.tolist() == [9, 0]
    assert state.counts.sum() == 2
    assert state.promotions.sum() == state.demotions.sum() == 0


def test_initial_manual_and_unfinished_resets_have_no_outcomes():
    state = fixture(2)
    settle(state, finished=[1, 1])  # No active episode yet, even with stale flags.
    begin(state)
    settle(state, step=12)  # Explicit user reset before any actual done.
    assert state.counts.sum() == 0 and not state.sampler_pending.any()


@pytest.mark.parametrize("corruption", ["clock", "motion", "start", "frame"])
def test_reaching_a_last_frame_flag_alone_cannot_award_success(corruption):
    state = fixture(1)
    begin(state)
    args = dict(finished=[1])
    if corruption == "clock":
        args["step"] = 1
    elif corruption == "motion":
        args["motion_ids"] = [1]
    elif corruption == "start":
        state.start_frame[0] = 498
    else:
        args["frame_ids"] = [497]
    settle(state, **args)
    assert state.last_outcome.item() == 5
    assert state.terrain.terrain_levels.item() == 5


def test_partial_reset_updates_only_chosen_environment_and_uses_birth_assignment():
    state = fixture(3)
    begin(state)
    before = state.terrain.env_origins.clone()
    ids = torch.tensor([1])
    state.settle(ids, step=498, motion_ids=ids, frame_ids=torch.tensor([498]), failure=torch.tensor([False]),
                 finished=torch.tensor([True]), map_exit=torch.tensor([False]), timeout=torch.tensor([False]), other=torch.tensor([False]))
    assert state.terrain.terrain_levels.tolist() == [5, 6, 5]
    torch.testing.assert_close(before[[0, 2]], state.terrain.env_origins[[0, 2]], atol=0, rtol=0)
    assert state.active.tolist() == [True, False, True]


def test_curriculum_checkpoint_restores_stats_and_drops_interrupted_episode():
    source = fixture(2)
    begin(source)
    settle(source, finished=[1, 0], failure=[0, 1])
    begin(source, step=498)
    saved = source.state_dict()
    target = fixture(2)
    begin(target)
    target.load_state_dict(saved)
    for key, value in saved.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(target.state_dict()[key], value)
    assert not target.active.any() and not target.sampler_pending.any()
    settle(target, step=1000, finished=[1, 1])
    assert target.counts.sum() == 2
    saved["levels"].zero_()
    assert target.terrain.terrain_levels.tolist() == [6, 4]


def test_center_reset_cannot_silently_resume_random_xy_curriculum():
    state = fixture(2)
    state.terrain.manifest = {"asset_sha256": "same geometry"}
    command = SimpleNamespace(terrain_families=["flat"], compatibility_sha256="same motions",
        data_store=SimpleNamespace(motions=[], motion_lengths=torch.tensor([])),
        sampler=SimpleNamespace(alpha=.1, beta=1., uniform_ratio=.1, update_interval=1, enable_frame_sampling=False),
        cfg=SimpleNamespace(fixed_start_frame=0, require_valid_support_start=False))
    env = SimpleNamespace(scene=SimpleNamespace(terrain=state.terrain), cfg=SimpleNamespace())
    historical = _identity(env, command)
    command.cfg.placement_xy_mode = "random"
    assert _identity(env, command) == historical
    state.identity = historical
    saved = state.state_dict()
    command.cfg.placement_xy_mode = "center"
    centered = _identity(env, command)
    assert centered["reset"]["placement_xy_mode"] == "center"
    with pytest.raises(ValueError, match="checkpoint"):
        TerrainCurriculum(state.terrain, centered).validate_state_dict(saved)


@pytest.mark.parametrize("field,value", [("version", 9), ("identity", {}), ("levels", torch.tensor([10, 0])),
                                         ("families", torch.tensor([7, 0])), ("counts", torch.zeros(2))])
def test_bad_checkpoints_fail_before_mutation(field, value):
    state = fixture(2)
    saved = state.state_dict()
    saved[field] = value
    with pytest.raises(ValueError):
        state.load_state_dict(saved)
    assert state.terrain.terrain_levels.tolist() == [5, 5]
    assert state.counts.sum() == 0


def test_training_resume_restores_sampler_and_reset_does_not_count_interruption():
    state = fixture(2)
    sampler = SimpleNamespace(**{key: torch.tensor([2., 3.]) for key in SAMPLER_COUNTS},
                              _steps_since_update=7, update_interval=240, _update_metrics=lambda: None)
    resets = []
    def reset():
        assert not state.active.any() and not state.sampler_pending.any()
        resets.append(True)
        begin(state, step=800)
    env = SimpleNamespace(_pgmt_terrain_curriculum=state, reset=reset,
                          command_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(sampler=sampler)))
    begin(state)
    settle(state, finished=[1, 0], failure=[0, 1])
    saved = training_state_dict(env)
    state.counts.zero_()
    state.terrain.terrain_levels.zero_()
    for key in SAMPLER_COUNTS:
        getattr(sampler, key).zero_()
    load_training_state_dict(env, saved)
    assert state.counts.sum() == 2 and state.terrain.terrain_levels.tolist() == [6, 4]
    assert len(resets) == 1 and sampler._steps_since_update == 7
    for key in SAMPLER_COUNTS:
        torch.testing.assert_close(getattr(sampler, key), torch.tensor([2., 3.]))
    bad = copy.deepcopy(saved)
    bad["curriculum"]["levels"].fill_(1)
    bad["sampler"]["failed_motion_count"][0] = float("nan")
    with pytest.raises(ValueError, match="sampler"):
        load_training_state_dict(env, bad)
    assert state.terrain.terrain_levels.tolist() == [6, 4] and len(resets) == 1
