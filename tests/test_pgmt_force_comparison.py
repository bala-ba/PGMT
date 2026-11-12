"""Physics sampling/reset integrity and paired force/event analysis."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


analysis = module("pgmt_force_analysis")
recording = module("pgmt_force_recording")


def write_run(path, terrain, onset=.30, *, force=100., stop=.8):
    path.mkdir()
    dt, decimation = .005, 4
    count = round(stop / dt)
    times = (np.arange(count) + 1) * dt
    f = np.zeros((count, 2, 3))
    f[times >= onset - 1e-8, :, 2] = force
    frames = np.arange(count) // decimation
    data = dict(time_s=times, episode=np.ones(count, dtype=int), frame=frames,
        substep=np.arange(count) % decimation, terrain=np.full(count, terrain),
        reference_contact=np.repeat((frames >= 15)[:, None], 2, axis=1),
        reference_valid=np.ones((count, 2), dtype=bool),
        net_forces_w=f.copy(), terrain_forces_w=f,
        foot_pos_w=np.zeros_like(f), reference_pos_w=np.zeros_like(f), foot_vel_w=np.zeros_like(f))
    metadata = dict(status="completed", source_sha256="same-policy", physics_dt=dt, control_dt=.02,
        observation_noise=False, random_pushes=False, random_dynamics=False, curriculum_updates=False,
        action_mode="actor mean",
        arguments=dict(patch_id=5, direction="up"), force_recording=dict(schema_version=1,
        motion_sha256="same-motion", terrain_manifest_sha256="same-mesh", seed=1, spawn_offset_x=0.,
        force_semantics="normal", reference_sampling="held", decimation=4,
        sampling="each-physics-step", foot_names=["left", "right"], physics_buffer_capacities={},
        contact_config=dict(support_threshold_n=10., touchdown_min_non_support_steps=2)))
    np.savez(path / "foot_forces.npz", **data)
    (path / "training_status.json").write_text(json.dumps(metadata))
    return analysis.load_run(path)


def test_force_impulse_and_event_delay_have_physics_units(tmp_path):
    flat = write_run(tmp_path / "flat", "flat")
    stairs = write_run(tmp_path / "stairs", "stairs", onset=.35, force=200.)
    report = analysis.compare(dict(flat=flat, stairs=stairs))
    event = report["events"]["stairs"]["actual"][0]
    assert event["reference_frame"] == 15
    assert event["delay_s"] == pytest.approx(.05)
    assert event["window_complete"]
    assert event["peak_terrain_fz_n"] == 200.
    # Includes the interval ending at onset: 31 samples, not control_dt * 31.
    assert event["upward_normal_impulse_ns"] == pytest.approx(31.)
    assert report["metrics"][0]["common_samples"] == 140


def test_missing_events_do_not_shift_the_next_landing():
    assert analysis.match_events([1.04, 3.01], [1., 2., 3.]) == [(0, 0), (1, 2)]
    assert analysis.match_events([.5, 1.02, 2.01], [1., 2.]) == [(1, 0), (2, 1)]
    assert analysis.match_events([], [1.]) == []


def test_multiple_nearby_contacts_mark_matching_ambiguous(tmp_path):
    run = write_run(tmp_path / "flat", "flat", onset=.20)
    # Another valid unloading/contact cycle near the single reference landing.
    data = run["data"]
    data["terrain_forces_w"][45:60, :, 2] = 0.
    events = analysis.analyze_events(run)
    assert events["reference"][0]["candidate_actual_count"] == 2
    matched = [e for e in events["actual"] if e["reference_frame"] is not None]
    assert all(e["ambiguous_match"] for e in matched)


def test_initial_contact_invalid_and_short_bounce_not_touchdown():
    dt = .005
    values = np.array([100.] * 10 + [0.] * 8 + [100.] + [0.] * 8 + [100.] * 3)
    assert analysis.support_onsets(values, np.ones(len(values), bool), dt) == [27]
    valid = np.ones(len(values), bool)
    valid[26] = False
    assert analysis.support_onsets(values, valid, dt) == []


def test_terminal_window_and_missing_data_are_not_zero_or_complete(tmp_path):
    flat = write_run(tmp_path / "flat", "flat")
    stair = write_run(tmp_path / "stairs", "stairs", onset=.35, stop=.4)
    event = analysis.analyze_events(stair)["actual"][0]
    assert not event["window_complete"] and event["peak_terrain_fz_n"] is None
    stair["data"]["terrain_forces_w"][30, 0] = np.nan
    report = analysis.compare(dict(flat=flat, stairs=stair))
    row = next(r for r in report["metrics"] if r["terrain"] == "stairs" and r["foot"] == "L")
    assert row["valid_force_samples"] == row["common_samples"] - 1
    assert row["upward_normal_impulse_ns"] is None
    assert report["coverage"]["stairs"]["max_frame"] == 19


@pytest.mark.parametrize("field", ["motion_sha256", "seed", "spawn_offset_x", "force_semantics"])
def test_unpaired_conditions_rejected(tmp_path, field):
    flat = write_run(tmp_path / "flat", "flat")
    stair = copy.deepcopy(flat)
    stair["data"]["terrain"] = np.full(len(stair["data"]["time_s"]), "slope")
    stair["metadata"]["force_recording"][field] = "different"
    with pytest.raises(ValueError, match="Unpaired"):
        analysis.compare(dict(flat=flat, slope=stair))


def test_no_events_matched_across_episode_reset(tmp_path):
    run = write_run(tmp_path / "flat", "flat", onset=.35)
    data = run["data"]
    cut = 64  # .32 seconds: reference landed in ep1; actual contact in ep2.
    data["episode"][cut:] = 2
    data["time_s"][cut:] -= cut * .005
    events = analysis.analyze_events(run)
    assert not events["reference"][0]["matched"]
    assert all(e["reference_frame"] is None for e in events["actual"])


def test_loader_rejects_missing_physics_interval(tmp_path):
    run = write_run(tmp_path / "flat", "flat")
    data = {k: np.delete(v, 25, axis=0) for k, v in run["data"].items()}
    np.savez(tmp_path / "flat" / "foot_forces.npz", **data)
    with pytest.raises(ValueError, match="Nonconsecutive"):
        analysis.load_run(tmp_path / "flat")


@pytest.mark.parametrize("full_state", [False, True])
def test_recorder_keeps_all_substeps_and_terminal_pose_before_reset(tmp_path, full_state):
    force = torch.zeros(1, 2, 3)
    pose = torch.zeros_like(force)
    frozen = {}

    class Scene(dict):
        def update(self, dt):
            force.add_(1.)
            pose.add_(.1)

    scene = Scene(robot=SimpleNamespace(data=SimpleNamespace(body_pos_w=pose,
        body_lin_vel_w=pose, body_quat_w=torch.zeros(1, 2, 4), joint_pos=pose[..., 0])))
    scene.terrain = SimpleNamespace(height=lambda xy: torch.zeros(len(xy)))
    env = SimpleNamespace(num_envs=1, scene=scene, physics_dt=.005, step_dt=.02,
                          cfg=SimpleNamespace(decimation=4), episode_length_buf=torch.tensor([0]))
    state = SimpleNamespace(command=SimpleNamespace(frame_ids=torch.tensor([7])),
        foot_body_indices=torch.tensor([0, 1]),
        reference=lambda: dict(body_pos=pose, body_lin_vel=pose, body_quat=torch.zeros(1, 2, 4)),
        reference_contacts=lambda: dict(contacts=torch.zeros(1, 2), contact_valid=torch.ones(1, 2, dtype=torch.bool)),
        robot_foot_contact_forces=lambda: force, robot_foot_terrain_forces=lambda: force,
        contact_state=lambda **kwargs: frozen)
    recorder = recording.FootForceRecorder(env, state, tmp_path, full_state=full_state)
    recorder.begin(dict(episode=1, terrain="flat", level=0, direction="center"))
    for _ in range(4):
        scene.update(.005)
    frozen.update(net_forces_w=force.clone(), terrain_forces_w=force.clone(), reference_frame_ids=torch.tensor([7]))
    # Simulate auto-reset overwriting live sensors/pose/frame before env.step returns.
    force.zero_()
    pose.zero_()
    state.command.frame_ids.fill_(0)
    recorder.finish(True)
    recorder.close()
    with np.load(tmp_path / "foot_forces.npz") as data:
        np.testing.assert_allclose(data["net_forces_w"][:, 0, 2], [1, 2, 3, 4])
        np.testing.assert_allclose(data["foot_pos_w"][:, 0, 2], [.1, .2, .3, .4])
        np.testing.assert_allclose(data["reference_pos_w"], 0.)
        assert np.all(data["frame"] == 7) and data["done"].all()
        np.testing.assert_allclose(data["time_s"], [.005, .01, .015, .02])
        if full_state:
            np.testing.assert_allclose(data["body_pos_w"][:, 0, 2], [.1, .2, .3, .4])
            np.testing.assert_allclose(data["joint_pos"][:, 0], [.1, .2, .3, .4])


def test_plots_and_report_export(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    runs = {name: write_run(tmp_path / name, name) for name in ("flat", "stairs", "slope")}
    report = analysis.compare(runs)
    analysis.write_report(runs, tmp_path / "report", report)
    assert report["common_complete_touchdowns"] == 2
    assert (tmp_path / "report" / "reference_aligned.png").stat().st_size > 1000
    assert (tmp_path / "report" / "touchdown_aligned.png").stat().st_size > 1000
    json.loads((tmp_path / "report" / "comparison.json").read_text())
