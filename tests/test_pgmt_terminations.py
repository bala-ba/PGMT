"""Stage2 failures: independent coordinates, threshold edges and live references."""
from types import SimpleNamespace

import pytest
import torch

from pmt_tasks.pgmt_terminations import (
    ConsecutiveFailureTimer, pgmt_time_out, pgmt_root_xy_failure, pgmt_torso_height_failure, pgmt_wrist_height_failure,
)

DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
NAMES = ["right_wrist_yaw_link", "torso_link", "left_ankle_roll_link", "pelvis",
         "left_wrist_yaw_link", "right_ankle_roll_link"]


def timeout_fixture(device, steps, frames, *, lengths=None, max_steps=500):
    command = SimpleNamespace(
        frame_ids=torch.tensor(frames, device=device),
        motion_ids=torch.arange(len(frames), device=device),
        data_store=SimpleNamespace(motion_lengths=torch.tensor(
            lengths if lengths is not None else [499] * len(frames), device=device)))
    env = SimpleNamespace(episode_length_buf=torch.tensor(steps, device=device),
                          max_episode_length=max_steps,
                          command_manager=SimpleNamespace(get_term=lambda name: command))
    return env, command


@pytest.mark.parametrize("device", DEVICES)
def test_timeout_tolerance_includes_finished_boundary_but_never_skips_last_frame(device):
    env, _ = timeout_fixture(device,
        [0, 496, 497, 498, 499, 498, 499, 500, 501, 499],
        [0, 498, 498, 498, 498, 497, 497, 0, 0, 498])
    assert pgmt_time_out(env).tolist() == [False, False, False, True, True, False, False, True, True, True]
    # Disabling the tolerance restores the original 500-step predicate exactly.
    assert pgmt_time_out(env, boundary_tolerance_steps=0).tolist() == [False] * 7 + [True, True, False]
    assert pgmt_time_out(env, boundary_tolerance_steps=1).tolist() == [False, False, False, False, True, False, False, True, True, True]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("max_steps", [400, 500, 1000])
def test_timeout_uses_control_steps_and_current_motion_without_cached_reset_state(device, max_steps):
    env, command = timeout_fixture(device, [max_steps - 2] * 3, [98, 498, 998],
                                   lengths=[99, 499, 999], max_steps=max_steps)
    assert pgmt_time_out(env).all()
    # Different per-env lengths, partial reset and subsequent reads use live data.
    env.episode_length_buf[0] = 0
    command.frame_ids[0] = 0
    command.motion_ids[1] = 2
    before = command.frame_ids.clone()
    for _ in range(3):
        assert pgmt_time_out(env).tolist() == [False, False, True]
    torch.testing.assert_close(command.frame_ids, before)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("initial_frame", [0, 1])
def test_timeout_initial_and_automatic_reset_preserve_original_motion_end_step(device, initial_frame):
    # Reproduce native ordering: physics/episode clock -> termination -> command
    # advance. Auto-reset advances the newly selected command on the reset step.
    env, command = timeout_fixture(device, [0], [initial_frame])
    for step in range(1, 500):
        env.episode_length_buf.fill_(step)
        finished = bool(command.frame_ids[0] == 498)
        timeout = bool(pgmt_time_out(env)[0])
        if finished or timeout:
            assert step == 499 - initial_frame
            assert finished and timeout
            break
        command.frame_ids += 1
    else:
        pytest.fail("The motion must finish before the hard time cap")


@pytest.mark.parametrize("tolerance", [-1, .5, True])
def test_timeout_rejects_invalid_tolerance(tolerance):
    env, _ = timeout_fixture("cpu", [0], [0])
    with pytest.raises(ValueError, match="nonnegative integer"):
        pgmt_time_out(env, boundary_tolerance_steps=tolerance)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dt,required", [(.02, 10), (.025, 8), (.03, 7)])
def test_failure_duration_boundary_and_recovery_restart(device, dt, required):
    timer = ConsecutiveFailureTimer(2, dt=dt, duration_s=.2, device=device)
    bad = torch.tensor([True, False], device=device)
    assert not timer.update(bad, 0).any()  # No elapsed control interval at initialization.
    for step in range(1, required):
        assert not timer.update(bad, step).any()
        for _ in range(3):
            assert not timer.update(bad, step).any()  # Reads cannot advance time.
    assert timer.update(bad, required).tolist() == [True, False]
    assert not timer.update(~bad, required + 1).any()
    assert timer.counts.tolist() == [0, 1]
    for step in range(required + 2, 2 * required + 1):
        assert not timer.update(bad, step).any()
    assert timer.update(bad, 2 * required + 1).tolist() == [True, False]


@pytest.mark.parametrize("device", DEVICES)
def test_each_failure_term_and_environment_has_independent_timer(device):
    timers = [ConsecutiveFailureTimer(2, dt=.02, duration_s=.2, device=device) for _ in range(4)]
    for step in range(1, 11):
        for index, timer in enumerate(timers):
            # Alternating failure categories must never combine into ten bad steps.
            bad = torch.tensor([index == step % 4, index == 0], device=device)
            assert timer.update(bad, step).tolist() == [False, step == 10 and index == 0]


@pytest.mark.parametrize("device", DEVICES)
def test_partial_reset_and_same_step_reads_cannot_reuse_old_failures(device):
    timer = ConsecutiveFailureTimer(2, dt=.02, duration_s=.2, device=device)
    bad = torch.ones(2, dtype=torch.bool, device=device)
    for step in range(1, 10):
        timer.update(bad, step)
    timer.reset(torch.tensor([0], device=device), step=9)
    assert not timer.update(bad, 9).any()
    assert timer.counts.tolist() == [0, 9]
    assert timer.update(bad, 10).tolist() == [False, True]
    assert timer.counts.tolist() == [1, 10]
    # A changed diagnostic predicate within the same control step is not a new sample.
    assert timer.update(~bad, 10).tolist() == [False, True]
    timer.reset(step=10)
    assert timer.counts.tolist() == [0, 0]
    assert not timer.update(bad, 10).any()
    assert not timer.update(bad, 11).any()
    # Missing observations cannot establish consecutive failure across a gap.
    timer.update(bad, 20)
    assert timer.counts.tolist() == [1, 1]


def fixture(device, n=6):
    source = torch.zeros(n, len(NAMES), 3, device=device, dtype=torch.float64)
    source[:, NAMES.index("pelvis"), 2] = .8
    source[:, NAMES.index("torso_link"), 2] = 1.
    source[:, [NAMES.index(name) for name in ("left_wrist_yaw_link", "right_wrist_yaw_link")], 2] = 1.2
    placed = source + source.new_tensor([7., 9., 3.])
    actual = placed.clone()
    calls = []
    def query(xy):
        calls.append(xy.clone())
        valid = torch.isfinite(xy).all(-1) & (xy[:, 0] < 100.)
        # A sloped negative floor, deliberately unrelated to the spawn origin.
        height = .125 * xy[:, 0] - 1.875
        return dict(height=torch.where(valid, height, float("nan")), valid=valid, tile_valid=valid)
    command = SimpleNamespace(cfg=SimpleNamespace(body_names=NAMES), source_body_pos=source,
        body_pos_w=placed, robot_body_pos_w=actual, terrain=SimpleNamespace(query=query))
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))
    return env, command, calls


@pytest.mark.parametrize("device", DEVICES)
def test_root_is_pelvis_euclidean_xy_and_placed_reference(device):
    env, c, _ = fixture(device)
    root, torso = NAMES.index("pelvis"), NAMES.index("torso_link")
    c.robot_body_pos_w[:, root, :2] = 0.
    c.body_pos_w[:, root, :2] = c.body_pos_w.new_tensor([[0., 0.], [.6, .8], [1.0001, 0.],
                                                                       [.8, .8], [0., -1.0001], [0., 0.]])
    c.robot_body_pos_w[:, root, 2] += 80.  # Z does not affect planar error.
    c.robot_body_pos_w[:, torso, 0] += 80.  # torso is not the root.
    assert pgmt_root_xy_failure(env).tolist() == [False, False, True, True, True, False]
    before = c.body_pos_w.clone()
    pgmt_root_xy_failure(env)
    torch.testing.assert_close(c.body_pos_w, before, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_torso_uses_actual_xy_original_z_signed_ground_and_strict_boundary(device):
    env, c, calls = fixture(device)
    torso = NAMES.index("torso_link")
    # Ground at actual x=7 is -1, source torso Z=1, hence target exactly zero.
    c.robot_body_pos_w[:, torso, 2] = c.body_pos_w.new_tensor([0., .4, -.4, .4001, -.4001, 0.])
    c.body_pos_w[:, torso, :2] += 30.  # Ignore reference XY for ground queries.
    c.body_pos_w[:, torso, 2] += 50.  # Never add placed Z to ground.
    assert pgmt_torso_height_failure(env).tolist() == [False, False, False, True, True, False]
    torch.testing.assert_close(calls[-1], c.robot_body_pos_w[:, torso, :2])
    # Cross to a new height without changing the reference placement.
    c.robot_body_pos_w[:, torso, 0] += 8.
    c.robot_body_pos_w[:, torso, 2] += 1.
    assert pgmt_torso_height_failure(env).tolist() == [False, False, False, True, True, False]


@pytest.mark.parametrize("device", DEVICES)
def test_wrist_error_is_relative_to_each_pelvis_and_ignores_feet(device):
    env, c, _ = fixture(device)
    root = NAMES.index("pelvis")
    wrists = [NAMES.index(n) for n in ("left_wrist_yaw_link", "right_wrist_yaw_link")]
    c.source_body_pos[..., 2] = 0.
    c.robot_body_pos_w[..., 2] = 0.
    c.robot_body_pos_w[:, wrists[0], 2] = c.body_pos_w.new_tensor([0., .4, -.4, .4001, 0., .1])
    c.robot_body_pos_w[4, wrists[1], 2] = -.4001
    c.robot_body_pos_w[:, [NAMES.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")], 2] += 90.
    assert pgmt_wrist_height_failure(env).tolist() == [False, False, False, True, True, False]
    # Large common translations cancel; a pelvis-only shift does not cancel.
    c.robot_body_pos_w[..., 2] += 4.
    c.source_body_pos[..., 2] -= 3.
    result = pgmt_wrist_height_failure(env)
    assert result[[0, 3, 4, 5]].tolist() == [False, True, True, False]
    c.robot_body_pos_w[0, root, 2] += .6
    assert pgmt_wrist_height_failure(env)[0]


@pytest.mark.parametrize("device", DEVICES)
def test_current_motion_frame_reset_and_repeat_reads_have_no_stale_state(device):
    env, c, _ = fixture(device, n=2)
    torso = NAMES.index("torso_link")
    c.robot_body_pos_w[:, torso, 2] = 0.
    expected = [fn(env).clone() for fn in (pgmt_root_xy_failure, pgmt_torso_height_failure, pgmt_wrist_height_failure)]
    for _ in range(3):
        for fn, value in zip((pgmt_root_xy_failure, pgmt_torso_height_failure, pgmt_wrist_height_failure), expected):
            torch.testing.assert_close(fn(env), value)
    c.source_body_pos[0, torso, 2] += .5
    assert pgmt_torso_height_failure(env).tolist() == [True, False]
    c.source_body_pos[0, torso, 2] -= .5
    assert pgmt_torso_height_failure(env).tolist() == [False, False]


@pytest.mark.parametrize("device", DEVICES)
def test_no_zero_height_fallback_outside_map_and_missing_inside_ray_is_error(device):
    env, c, _ = fixture(device, n=1)
    torso = NAMES.index("torso_link")
    c.robot_body_pos_w[:, torso, 0] = 101.
    assert not pgmt_torso_height_failure(env).any()  # Existing map truncation owns this case.
    c.terrain.query = lambda xy: dict(height=xy.new_full((1,), float("nan")),
        tile_valid=torch.ones(1, dtype=torch.bool, device=device), valid=torch.zeros(1, dtype=torch.bool, device=device))
    with pytest.raises(RuntimeError, match="missing clean height"):
        pgmt_torso_height_failure(env)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fn,body", [(pgmt_root_xy_failure, "pelvis"),
    (pgmt_torso_height_failure, "torso_link"), (pgmt_wrist_height_failure, "left_wrist_yaw_link")])
def test_nonfinite_pose_does_not_silently_pass(device, fn, body):
    env, c, _ = fixture(device, n=1)
    c.robot_body_pos_w[:, NAMES.index(body)] = float("nan")
    assert fn(env).all()
