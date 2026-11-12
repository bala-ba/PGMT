"""Clip/time independence, exact OT priorities, atomic updates and resume."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pmt_tasks.pgmt_emd_sampling import EMDMotionSampler, joint_trajectory_emd, sample_emd_starts
from pmt_tasks.pgmt_curriculum import TerrainCurriculum, training_state_dict, load_training_state_dict
from motion_tracking_rl.utils.pgmt_emd_evaluation import PGMTEMDEvaluation


def test_emd_matches_uniform_transport_and_is_not_aligned_tracking_rmse():
    from scipy.optimize import linprog
    from scipy.spatial.distance import cdist
    reference = np.array([[0., 0.], [1., 2.], [3., 4.]])
    actual = reference[[2, 0, 1]]
    assert joint_trajectory_emd(actual, reference) == 0
    actual = actual + [[.2, .4], [.5, -.2], [-.2, .1]]
    cost = cdist(actual, reference)
    constraints = np.vstack([np.kron(np.eye(3), np.ones((1, 3))), np.kron(np.ones((1, 3)), np.eye(3))])
    expected = linprog(cost.flatten(), A_eq=constraints, b_eq=np.full(6, 1/3), bounds=(0, None), method="highs")
    assert expected.success
    assert joint_trajectory_emd(actual, reference) == pytest.approx(expected.fun)
    with pytest.raises(ValueError, match="complete"):
        joint_trajectory_emd(actual[:2], reference)
    actual[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        joint_trajectory_emd(actual, reference)


def test_motion_priority_and_uniform_time_are_separate_and_keep_terrain_exclusions():
    torch.manual_seed(83)
    lengths = torch.tensor([3, 11, 499])
    terrains = torch.zeros(18000, dtype=torch.long)
    compatible = torch.tensor([[True, True, False]])
    weights = torch.tensor([2., 16., 100000.])
    motions, frames = sample_emd_starts(weights, compatible, lengths, terrains)
    assert not motions.eq(2).any()
    assert motions.eq(1).float().mean().item() == pytest.approx(8/9, abs=.015)
    for mid in (0, 1):
        histogram = torch.bincount(frames[motions == mid], minlength=int(lengths[mid]-1)).float()
        torch.testing.assert_close(histogram / histogram.sum(), torch.full_like(histogram, 1/len(histogram)), atol=.035, rtol=0)
    # Changing priorities changes clip frequency, never the conditional time law.
    motions, frames = sample_emd_starts(torch.ones(3), compatible, lengths, terrains)
    assert motions.eq(1).float().mean().item() == pytest.approx(.5, abs=.015)
    assert frames.max() < 10 and frames.min() == 0
    _, frames = sample_emd_starts(weights, compatible, lengths, terrains[:8], fixed_start_frame=0)
    assert frames.eq(0).all()


def test_emd_weights_replace_without_ema_uniform_mix_or_failure_updates():
    sampler = EMDMotionSampler(4, "cpu")
    torch.testing.assert_close(sampler.get_sampling_probabilities(), torch.full((4,), .25))
    sampler.update_emd([0., .5, 1., 9.], 9_600_000)
    torch.testing.assert_close(sampler.weights, torch.tensor([2., 2., 4., 16.]))
    before = sampler.state_dict()
    sampler.update(torch.arange(4), torch.zeros(4), torch.ones(4, dtype=torch.bool))
    for _ in range(241):
        sampler.step()
    torch.testing.assert_close(sampler.weights, before["weights"], rtol=0, atol=0)
    sampler.update_emd([2., 2., .5, .5], 19_200_000)
    torch.testing.assert_close(sampler.weights, torch.tensor([16., 16., 2., 2.]))
    for bad in ([1., float("nan"), 2., 3.], [1.], [-1., 0., 0., 0.]):
        before = sampler.state_dict()
        with pytest.raises(ValueError):
            sampler.update_emd(bad, 28_800_000)
        torch.testing.assert_close(sampler.weights, before["weights"], rtol=0, atol=0)
        assert sampler.last_evaluation_step == 19_200_000


def test_evaluation_schedule_uses_global_transitions_and_restores_progress():
    sampler = EMDMotionSampler(2, "cpu", dict(evaluation_interval=24))
    runner = SimpleNamespace(num_steps_per_env=3, env=SimpleNamespace(num_envs=4), gpu_world_size=1,
                             gpu_global_rank=0, is_distributed=False, writer=None, disable_logs=False)
    evaluator = PGMTEMDEvaluation(runner, SimpleNamespace(sampler=sampler))
    calls = []
    evaluator._evaluate = lambda step: calls.append(step) or [1., 2.]
    evaluator.after_update(1)
    assert not calls
    evaluator.after_update(2)
    assert calls == [24]
    restored = EMDMotionSampler(2, "cpu", sampler.config)
    restored.load_state_dict(sampler.state_dict())
    evaluator.command.sampler = restored
    evaluator.after_update(2)
    evaluator.after_update(3)
    assert calls == [24]
    evaluator.after_update(4)
    assert calls == [24, 48]
    evaluator._evaluate = lambda _: (_ for _ in ()).throw(ValueError("bad evaluation"))
    with pytest.raises(RuntimeError, match="priorities were not changed"):
        evaluator.after_update(6)
    assert restored.last_evaluation_step == 48


def test_start_and_resume_evaluation_can_replace_priorities_at_the_same_step():
    sampler = EMDMotionSampler(2, "cpu")
    runner = SimpleNamespace(num_steps_per_env=3, env=SimpleNamespace(num_envs=4), gpu_world_size=1,
                             gpu_global_rank=0, is_distributed=False, writer=None, disable_logs=False)
    evaluator = PGMTEMDEvaluation(runner, SimpleNamespace(sampler=sampler))
    evaluator._evaluate = lambda step: [.5, 1.5]
    evaluator.after_update(0, force=True)
    restored = EMDMotionSampler(2, "cpu")
    restored.load_state_dict(sampler.state_dict())
    assert restored.last_evaluation_step == 0 and restored.evaluation_count == 1
    evaluator.command.sampler = restored
    evaluator._evaluate = lambda step: [1., 2.]
    evaluator.after_update(0, force=True)
    assert restored.evaluation_count == 2
    torch.testing.assert_close(restored.weights, torch.tensor([4., 16.]))


def test_random_start_completion_and_emd_state_are_checkpointed_atomically():
    sampler = EMDMotionSampler(2, "cpu")
    sampler.update_emd([.7, 1.6], 9_600_000)
    origins = torch.zeros(10, 1, 3)
    terrain = SimpleNamespace(terrain_levels=torch.tensor([3, 3]), terrain_types=torch.zeros(2, dtype=torch.long),
                              terrain_origins=origins, env_origins=origins[[3, 3], [0, 0]].clone())
    course = TerrainCurriculum(terrain, dict(sampler=dict(kind="pbfm_emd")))
    ids = torch.arange(2)
    course.begin(ids, 100, ids, torch.tensor([100, 450]), torch.full((2,), 499))
    course.settle(ids, step=148, motion_ids=ids, frame_ids=torch.full((2,), 498),
                  failure=torch.tensor([False, False]), finished=torch.ones(2, dtype=torch.bool),
                  map_exit=torch.zeros(2, dtype=torch.bool), timeout=torch.zeros(2, dtype=torch.bool),
                  other=torch.zeros(2, dtype=torch.bool))
    assert course.last_outcome.tolist() == [5, 1]  # full suffix, not just a stale final-frame flag
    assert terrain.terrain_levels.tolist() == [3, 4]
    env = SimpleNamespace(_pgmt_terrain_curriculum=course,
        command_manager=SimpleNamespace(get_term=lambda _: SimpleNamespace(sampler=sampler)), reset=lambda: None)
    saved = training_state_dict(env)
    sampler.weights.fill_(1)
    sampler.evaluation_count = sampler.last_evaluation_step = 0
    load_training_state_dict(env, saved)
    assert sampler.last_evaluation_step == 9_600_000 and not course.active.any()
    for key in ("weights", "emd"):
        torch.testing.assert_close(getattr(sampler, key).cpu(), saved["sampler"][key], rtol=0, atol=0)
    bad = deepcopy(saved)
    bad["curriculum"]["levels"].zero_()
    bad["sampler"]["weights"][0] += 1
    with pytest.raises(ValueError, match="weights"):
        load_training_state_dict(env, bad)
    assert terrain.terrain_levels.tolist() == [3, 4]


def _distributed_evaluation_worker(rank, rendezvous, output):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        sampler = EMDMotionSampler(2, "cpu", dict(evaluation_interval=24))
        runner = SimpleNamespace(num_steps_per_env=3, env=SimpleNamespace(num_envs=4), gpu_world_size=2,
                                 gpu_global_rank=rank, is_distributed=True, writer=None, disable_logs=rank != 0)
        evaluator = PGMTEMDEvaluation(runner, SimpleNamespace(sampler=sampler))
        def evaluate(step):
            assert rank == 0 and step == 24
            return [.7, 1.6]
        evaluator._evaluate = evaluate
        evaluator.after_update(1)
        assert sampler.last_evaluation_step == 24
        def fail(_):
            raise ValueError("invalid full-library evaluation")
        evaluator._evaluate = fail
        try:
            evaluator.after_update(2)
        except RuntimeError as error:
            assert "priorities were not changed" in str(error)
        else:
            raise AssertionError("Evaluation errors must reach every rank")
        torch.save(sampler.state_dict(), f"{output}/rank_{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_two_rank_emd_broadcast_and_collective_failure(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_distributed_evaluation_worker,
             args=((tmp_path / "rendezvous").as_uri(), str(tmp_path)), nprocs=2, join=True)
    left = torch.load(tmp_path / "rank_0.pt", weights_only=False)
    right = torch.load(tmp_path / "rank_1.pt", weights_only=False)
    assert left["last_evaluation_step"] == right["last_evaluation_step"] == 24
    assert left["evaluation_count"] == right["evaluation_count"] == 1
    torch.testing.assert_close(left["weights"], right["weights"], rtol=0, atol=0)
