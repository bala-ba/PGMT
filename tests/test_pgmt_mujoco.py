"""CPU physics, input layout and contract checks for the PGMT standalone path."""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import torch

from pmt_tasks.backends.mujoco_standalone import (
    PGMTAdapter, StudentAdapter, build_model, create_adapter, load_policy, validate_contract,
)
from scripts.pmt_npz_to_mjlab import BFS_BODIES, BFS_JOINTS


@pytest.fixture
def flat_rig(tmp_path):
    # A named 29-joint rig keeps the test independent of downloaded G1 assets.
    # Production build_model still imports its supplied training URDF unchanged.
    robot = ET.Element("robot", name="g1_test")
    for index, name in enumerate(BFS_BODIES):
        link = ET.SubElement(robot, "link", name=name)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "mass", value="1")
        ET.SubElement(inertial, "inertia", ixx=".01", ixy="0", ixz="0", iyy=".01", iyz="0", izz=".01")
        visual = ET.SubElement(link, "visual")
        ET.SubElement(ET.SubElement(visual, "geometry"), "sphere", radius=".01")
        if index:
            joint = ET.SubElement(robot, "joint", name=BFS_JOINTS[index - 1], type="revolute")
            ET.SubElement(joint, "parent", link="pelvis")
            ET.SubElement(joint, "child", link=name)
            ET.SubElement(joint, "origin", xyz=f"{index * .01} 0 .1")
            ET.SubElement(joint, "axis", xyz="0 1 0")
            ET.SubElement(joint, "limit", lower="-3", upper="3", effort="50", velocity="30")
    urdf = tmp_path / "assets/urdf/g1/main.urdf"
    urdf.parent.mkdir(parents=True)
    ET.ElementTree(robot).write(urdf)
    offset = np.linspace(-.1, .1, 29)
    contract = {
        "policy_class": "PGMTActorCritic", "joint_names": BFS_JOINTS, "body_names": BFS_BODIES,
        "tracked_bodies": BFS_BODIES, "anchor_body": "torso_link", "urdf": str(urdf),
        "sim_dt": .005, "decimation": 4, "terrain_mesh": None, "chunk_length": 10,
        "action_scale": np.linspace(.2, .8, 29), "action_offset": offset,
        "robot": {"default_joint_pos": offset, "default_joint_vel": np.zeros(29),
                  "joint_stiffness": np.full(29, 10.), "joint_damping": np.ones(29),
                  "joint_armature": np.full(29, .01), "joint_effort_limits": np.full(29, 50.)},
    }
    model, geometry = build_model(contract, tmp_path / "model")
    assert geometry["terrain_boxes"] == 0
    assert geometry["source_terrain"] is None
    data = mujoco.MjData(model)
    qids = [model.jnt_qposadr[model.joint(n).id] for n in BFS_JOINTS]
    bodies = [model.body(n).id for n in BFS_BODIES]
    n = 30
    joint_pos = np.arange(n)[:, None] * .001 + np.zeros((n, 29))
    pos, quat = [], []
    for q in joint_pos:
        data.qpos[:3] = [0., 0., .9]
        data.qpos[3:7] = [1., 0., 0., 0.]
        data.qpos[qids] = q
        mujoco.mj_forward(model, data)
        pos.append(data.xpos[bodies].copy())
        quat.append(data.xquat[bodies].copy())
    motion_path = tmp_path / "motion.npz"
    np.savez(motion_path, fps=np.array([50.], np.float32), joint_pos=joint_pos,
             joint_vel=np.full((n, 29), .05), body_pos_w=pos, body_quat_w=quat,
             body_lin_vel_w=np.zeros((n, 30, 3)), body_ang_vel_w=np.zeros((n, 30, 3)),
             joint_names=BFS_JOINTS, body_names=BFS_BODIES)
    contract["motion_files"] = [str(motion_path)]
    contract["raw_files"] = contract["optimized_files"] = contract["motion_files"]
    return contract, model


def test_flat_model_native_actuators_targets_and_history(flat_rig):
    contract, model = flat_rig
    adapter = create_adapter(contract, model)
    assert isinstance(adapter, PGMTAdapter)
    adapter.reset(0)
    obs = adapter.observations()
    assert obs["proprio"].shape == (1, 96)
    assert obs["proprio_history"].shape == (1, 10, 96)
    assert obs["command_window"].shape == (1, 8, 61)
    assert obs["critic"].shape == (1, 251)
    torch.testing.assert_close(obs["proprio_history"], obs["proprio"][:, None].expand(-1, 10, -1))
    torch.testing.assert_close(adapter.observations()["proprio_history"], obs["proprio_history"])
    action = np.linspace(-.2, .2, 29)
    expected_target = contract["action_offset"] + contract["action_scale"] * action
    adapter.step(action)
    np.testing.assert_allclose(adapter.data.ctrl, expected_target, atol=1e-12)
    assert adapter.physics_steps == 4
    assert adapter.data.time == pytest.approx(.02)
    next_obs = adapter.observations()
    np.testing.assert_allclose(next_obs["proprio"][0, -29:], expected_target, atol=1e-7)
    torch.testing.assert_close(next_obs["proprio_history"][:, :-1], obs["proprio_history"][:, 1:])
    torch.testing.assert_close(adapter.observations()["proprio_history"], next_obs["proprio_history"])
    assert np.isfinite(adapter.data.qpos).all()


def test_future_window_respects_source_chunk_boundary(flat_rig):
    contract, model = flat_rig
    adapter = PGMTAdapter(contract, model)
    adapter.reset(0, 8)
    obs = adapter.observations()
    # Offsets 0,1,3,... all clamp to the native first chunk's final frame 9.
    np.testing.assert_allclose(obs["command_window"][0, :, 0], [.008] + [.009] * 7, atol=1e-8)
    adapter.reset(0, 18)
    np.testing.assert_allclose(adapter.observations()["command_window"][0, :, 0], [.018] + [.019] * 7, atol=1e-8)
    assert adapter.reference_start == 10 and adapter.reference_end == 20


def test_full_clip_evaluation_crosses_training_chunks_without_reset(flat_rig):
    from scripts.mujoco_eval_pgmt_full import reset_full_clip

    contract, model = flat_rig
    adapter = PGMTAdapter(contract, model)
    reset_full_clip(adapter, 0)
    assert adapter.reference_start == 0 and adapter.reference_end == 30
    for _ in range(21):
        adapter.step(np.zeros(29))
    assert adapter.frame == 21
    assert adapter.physics_steps == 84
    assert adapter.data.time == pytest.approx(.42)
    np.testing.assert_allclose(adapter.observations()["command_window"][0, :, 0],
                               [.021, .022, .024, .028, .029, .029, .029, .029], atol=1e-8)


def test_native_checkpoint_and_fixture_parity_detect_wrong_target(flat_rig, tmp_path):
    from motion_tracking_rl.networks.pgmt_actor_critic import PGMTActorCritic

    contract, model = flat_rig
    adapter = PGMTAdapter(contract, model)
    adapter.reset(0)
    obs = adapter.observations()
    groups = {"policy": ["proprio"], "policy_history": ["proprio_history"],
              "command_window": ["command_window"], "critic": ["critic"]}
    kwargs = dict(embedding_dim=16, num_heads=4, actor_hidden_dims=[32], critic_hidden_dims=[32],
                  actor_obs_normalization=True, critic_obs_normalization=True)
    policy = PGMTActorCritic(obs, groups, 29, **kwargs)
    policy.update_normalization(obs)
    policy.eval()
    path = tmp_path / "model.pt"
    torch.save({"model_state_dict": policy.state_dict(), "policy_metadata": policy.get_checkpoint_metadata()}, path)
    contract.update(checkpoint=str(path), obs_groups=groups, policy_kwargs=kwargs,
                    policy_metadata=policy.get_checkpoint_metadata(), fixtures=[])
    loaded = load_policy(contract)
    with torch.inference_mode():
        torch.testing.assert_close(policy.act_inference(obs), loaded.act_inference(obs))
        for _ in range(3):
            obs = adapter.observations()
            action = loaded.act_inference(obs)
            d = adapter.data
            fixture = {
                "obs": {k: v.clone() for k, v in obs.items()}, "action": action.clone(),
                "motion_id": 0, "frame": adapter.frame, "reference_start": 0, "reference_end": 10,
                "previous_target": adapter.previous_target.copy(), "env_origin": np.zeros(3),
                "action_target": adapter.action_targets(action[0].numpy()),
                "state": {"root_pos_w": d.qpos[:3].copy(), "root_quat_w": d.qpos[3:7].copy(),
                          "root_link_lin_vel_w": d.qvel[:3].copy(),
                          "root_ang_vel_w": np.zeros(3), "joint_pos": d.qpos[adapter.qids].copy(),
                          "joint_vel": d.qvel[adapter.vids].copy(), "body_pos_w": d.xpos[adapter.body_ids].copy()},
            }
            from pmt_tasks.backends.mujoco_standalone import rotation
            fixture["state"]["root_ang_vel_w"] = rotation(d.qpos[3:7]).apply(d.qvel[3:6])
            contract["fixtures"].append(fixture)
            adapter.step(action[0].numpy())
    assert validate_contract(contract, PGMTAdapter(contract, model), loaded)["passed"]
    bad = deepcopy(contract)
    bad["fixtures"][1]["action_target"][0] += .1
    assert not validate_contract(bad, PGMTAdapter(bad, model), loaded)["passed"]
    with pytest.raises(ValueError, match="Isaac parity fixtures"):
        validate_contract({**contract, "fixtures": []}, adapter, loaded)


def test_vision_dispatch_and_residual_targets_are_preserved(flat_rig):
    contract, model = flat_rig
    contract = {**contract, "policy_class": "VisionTransformerActorCritic"}
    adapter = create_adapter(contract, model)
    assert type(adapter) is StudentAdapter
    adapter.reset(0, 2)
    action = np.linspace(-.1, .1, 29)
    np.testing.assert_allclose(adapter.action_targets(action), adapter.raw["joint_pos"][2] + action)


def test_live_pgmt_runs_physics_without_depth_or_panel(flat_rig, tmp_path, monkeypatch):
    """Exercise live PGMT dispatch with the window mocked, real mj_step retained."""
    from contextlib import nullcontext
    import json
    from types import SimpleNamespace
    import scripts.mujoco_live_student as live

    contract, model = flat_rig
    contract_path = tmp_path / "contract.pt"
    contract["checkpoint"] = "test_policy"
    torch.save(contract, contract_path)
    output = tmp_path / "live"
    camera = SimpleNamespace(lookat=np.zeros(3), distance=0, azimuth=0, elevation=0)
    viewer = SimpleNamespace(lock=lambda: nullcontext(),
                             opt=SimpleNamespace(geomgroup=np.zeros(6, dtype=int)), cam=camera,
                             user_scn=SimpleNamespace(ngeom=0), is_running=lambda: True,
                             sync=lambda: None, set_texts=lambda *args: None, close=lambda: None)
    def forbidden(*args, **kwargs):
        raise AssertionError("PGMT live must not initialize depth or its height-map panel")
    monkeypatch.setattr(live, "MujocoDepthPerception", forbidden)
    monkeypatch.setattr(live, "LiveHeightPanel", forbidden)
    monkeypatch.setattr(live.mujoco.viewer, "launch_passive", lambda *args, **kwargs: viewer)
    monkeypatch.setattr(live, "build_model", lambda *args, **kwargs: (model, {}))
    monkeypatch.setattr(live, "load_policy", lambda contract: SimpleNamespace(act_inference=lambda obs: torch.zeros(1, 29)))
    monkeypatch.setattr(live, "validate_contract", lambda *args: {"passed": True})
    monkeypatch.setattr(live.threading, "Timer", lambda *args: SimpleNamespace(start=lambda: None, daemon=False))
    monkeypatch.setattr(live.time, "sleep", lambda delay: None)
    monkeypatch.setattr(live.sys, "argv", ["mujoco_live_student.py", "--contract", str(contract_path),
                                           "--output", str(output), "--steps", "2"])
    live.main()
    result = json.loads((output / "live_result.json").read_text())
    assert result["steps"] == 2 and result["mj_step_calls"] == 8
    assert result["depth_frames"] == 0 and result["finite"]
    assert result["viewer_updates"] >= 1
