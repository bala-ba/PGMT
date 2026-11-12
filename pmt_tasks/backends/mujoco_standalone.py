"""CPU MuJoCo rollout adapters for PMT vision students and PGMT stage 1.

No Isaac imports or simulator state streaming: observations come from MjData,
terrain rays use mj_ray, and joint targets are integrated by mj_step.
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from tensordict import TensorDict


def rotation(quat):
    return Rotation.from_quat(np.asarray(quat)[..., [1, 2, 3, 0]])


def build_model(contract, output, depth_camera=None):
    """Use the training URDF and preserve Bigmap's boxes (no convex-hull shortcut)."""
    import trimesh

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    urdf = Path(contract["urdf"])
    root = ET.parse(urdf).getroot()
    ET.SubElement(ET.SubElement(root, "mujoco"), "compiler", discardvisual="false",
                  fusestatic="false", meshdir=str(urdf.parents[2] / "meshes/g1"))
    imported = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    intermediate = output / "robot_from_urdf.xml"
    mujoco.mj_saveLastXML(str(intermediate), imported)
    root = ET.parse(intermediate).getroot()
    # MuJoCo 3.3.5 can crash while fusing a fixed child carrying a camera.
    # Preserve fixed links for camera scenes; their rigid transforms/inertias stay intact.
    root.find("compiler").set("fusestatic", "false" if depth_camera is not None else "true")
    ET.SubElement(root, "option", timestep=str(contract["sim_dt"]), integrator="implicitfast",
                  solver="Newton", iterations="50", tolerance="1e-8", gravity="0 0 -9.81")
    world = root.find("worldbody")
    pelvis = world.find("body[@name='pelvis']")
    ET.SubElement(pelvis, "freejoint", name="floating_base")
    if depth_camera is not None:
        pitch = np.deg2rad(depth_camera["pitch_deg"])
        # G1_structure.png places the depth camera in the upper front of the head.
        # head_link's mesh is offset from its frame: z spans 0.325..0.531 m.
        # Keep the optical center just outside the forehead mesh, not at the frame origin.
        mount_body = depth_camera.get("mount_body", "head_link")
        mount_position = depth_camera.get("mount_position_m", [.08, 0., .48])
        mount = root.find(f".//body[@name='{mount_body}']")
        if mount is None:
            raise ValueError(f"Depth camera attachment body is missing: {mount_body}")
        ET.SubElement(mount, "camera", name="pmt_depth", pos=" ".join(map(str, mount_position)),
                      xyaxes=f"0 -1 0 {np.sin(pitch)} 0 {np.cos(pitch)}",
                      fovy=str(depth_camera["fovy_deg"]))
        visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "global", offwidth=str(depth_camera["width"]), offheight=str(depth_camera["height"]))
        ET.SubElement(visual, "headlight", ambient=".4 .4 .4", diffuse=".5 .5 .5", specular="0 0 0")
        ET.SubElement(world, "light", pos="0 0 10", dir="-1 -2 -3", directional="true", diffuse=".6 .6 .6")
        # Absolute clipping planes are set after compilation (extent depends on Bigmap).
    for geom in pelvis.iter("geom"):
        if geom.get("type") == "cylinder":
            geom.set("type", "capsule")  # same replacement used by Isaac's URDF importer
        geom.set("friction", "1 0.005 0.0001")
    actuator = ET.SubElement(root, "actuator")
    parameters = contract["robot"]
    for i, name in enumerate(contract["joint_names"]):
        joint = root.find(f".//joint[@name='{name}']")
        joint.set("armature", str(float(parameters["joint_armature"][i])))
        joint.set("damping", "0")
        joint.set("frictionloss", "0")
        effort = float(parameters["joint_effort_limits"][i])
        ET.SubElement(actuator, "position", name=name, joint=name,
                      kp=str(float(parameters["joint_stiffness"][i])),
                      kv=str(float(parameters["joint_damping"][i])),
                      forcerange=f"{-effort} {effort}", forcelimited="true")
    terrain_mesh = contract.get("terrain_mesh")
    mesh = trimesh.load(terrain_mesh, force="mesh") if terrain_mesh else None
    boxes = 0
    for part in mesh.split(only_watertight=False) if mesh is not None else ():
        z0, z1 = part.bounds[:, 2]
        if z1 - z0 < 1e-6:
            continue
        top = part.vertices[np.abs(part.vertices[:, 2] - z1) < 1e-6, :2]
        if len(top) != 4 or len(part.faces) != 12:
            raise ValueError("Bigmap geometry is not an exact rectangular box.")
        center = top.mean(axis=0)
        top = top[np.argsort(np.arctan2(top[:, 1] - center[1], top[:, 0] - center[0]))]
        edges = np.roll(top, -1, axis=0) - top
        lengths = np.linalg.norm(edges, axis=1)
        if abs(np.dot(edges[0], edges[1])) > 1e-4:
            raise ValueError("Terrain component has non-orthogonal edges.")
        yaw = np.arctan2(edges[0, 1], edges[0, 0])
        ET.SubElement(world, "geom", name=f"terrain_box_{boxes}", type="box", group="2",
                      pos=f"{center[0]} {center[1]} {(z0 + z1) / 2}",
                      quat=f"{np.cos(yaw / 2)} 0 0 {np.sin(yaw / 2)}",
                      size=f"{lengths[0] / 2} {lengths[1] / 2} {(z1 - z0) / 2}",
                      friction="1 0.005 0.0001", rgba="0.68 0.60 0.46 1" if depth_camera else "0.42 0.5 0.57 1")
        boxes += 1
    ET.SubElement(world, "geom", name="terrain_ground", type="plane", group="2",
                  size="60 60 .01", friction="1 0.005 0.0001", rgba="0.3 0.3 0.3 1" if depth_camera else "0.3 0.4 0.5 1")
    path = output / "scene.xml"
    ET.indent(root)
    ET.ElementTree(root).write(path)
    model = mujoco.MjModel.from_xml_path(str(path))
    if depth_camera is not None:
        model.vis.map.znear = .02 / model.stat.extent
        model.vis.map.zfar = 10. / model.stat.extent
    return model, {"terrain_boxes": boxes, "scene_xml": str(path.resolve()),
                   "source_urdf": str(urdf), "source_terrain": terrain_mesh}


class StudentAdapter:
    def __init__(self, contract, model, seed=42, noisy_height=True):
        self.contract, self.model = contract, model
        self.data = mujoco.MjData(model)
        self.rng = np.random.default_rng(seed)
        self.noisy_height = noisy_height
        self.joints = np.array([model.joint(n).id for n in contract["joint_names"]])
        self.qids = model.jnt_qposadr[self.joints]
        self.vids = model.jnt_dofadr[self.joints]
        self.body_ids = np.array([model.body(n).id for n in contract["body_names"]])
        self.tracked = np.array([contract["body_names"].index(n) for n in contract["tracked_bodies"]])
        self.torso = model.body("torso_link").id
        self.anchor_index = contract["body_names"].index("torso_link")
        self.default_q = np.asarray(contract["robot"]["default_joint_pos"])
        self.default_dq = np.asarray(contract["robot"]["default_joint_vel"])
        self.scan_config = {"shape_xy": [17, 11], "ordering": "xy", "resolution_m": .1,
                            "size_m": [1.6, 1.0], "source": "MuJoCo mj_ray and actual policy observation"}
        x, y = np.meshgrid(np.linspace(-.8, .8, 17), np.linspace(-.5, .5, 11))
        self.ray_xy = np.stack([x.ravel(), y.ravel()], axis=1)
        self.histories = {}
        self.previous_action = np.zeros(29)
        self.physics_steps = 0
        self.perception = None

    def load_motion(self, motion_id):
        self.motion_id = motion_id
        self.raw = dict(np.load(self.contract["raw_files"][motion_id]))
        self.optimized = dict(np.load(self.contract["optimized_files"][motion_id]))
        fps = float(np.asarray(self.raw["fps"]).reshape(-1)[0])
        if abs(fps * self.contract["sim_dt"] * self.contract["decimation"] - 1) > 1e-6:
            raise ValueError("Motion fps must match control rate.")
        if len(self.raw["joint_pos"]) != len(self.optimized["joint_pos"]):
            raise ValueError("Raw/optimized motion pair lengths differ.")
        inv = rotation(self.raw["body_quat_w"][:, self.anchor_index]).inv()
        count = len(self.raw["joint_pos"])
        self.command_features = np.concatenate([
            inv.apply(self.raw["body_lin_vel_w"][:, self.anchor_index]),
            inv.apply(self.raw["body_ang_vel_w"][:, self.anchor_index]),
            inv.apply(np.tile([0, 0, -1], (count, 1))), self.raw["joint_pos"],
        ], axis=-1)

    def set_state(self, state):
        self.data.qpos[:3] = state["root_pos_w"]
        self.data.qpos[3:7] = state["root_quat_w"]
        self.data.qpos[self.qids] = state["joint_pos"]
        self.data.qvel[:3] = state["root_link_lin_vel_w"]
        self.data.qvel[3:6] = rotation(state["root_quat_w"]).inv().apply(state["root_ang_vel_w"])
        self.data.qvel[self.vids] = state["joint_vel"]
        mujoco.mj_forward(self.model, self.data)

    def reset(self, motion_id, frame=0):
        self.load_motion(motion_id)
        if not 0 <= frame < len(self.raw["joint_pos"]):
            raise ValueError("Start frame is outside the motion clip.")
        self.frame = frame
        mujoco.mj_resetData(self.model, self.data)
        m = self.optimized
        self.set_state({"root_pos_w": m["body_pos_w"][frame, 0], "root_quat_w": m["body_quat_w"][frame, 0],
                        "root_link_lin_vel_w": m["body_lin_vel_w"][frame, 0],
                        "root_ang_vel_w": m["body_ang_vel_w"][frame, 0],
                        "joint_pos": m["joint_pos"][frame], "joint_vel": m["joint_vel"][frame]})
        self.histories.clear()
        self.previous_action[:] = 0
        self.data.ctrl[:] = self.raw["joint_pos"][frame]
        if self.perception is not None:
            self.perception.reset()

    def scan(self):
        pose = self.data.xpos[self.torso]
        r = self.data.xmat[self.torso].reshape(3, 3)
        yaw = np.arctan2(r[1, 0], r[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        xy = self.ray_xy @ np.array([[c, s], [-s, c]]) + pose[:2]
        hits = np.zeros((len(xy), 3))
        valid = np.zeros(len(xy), dtype=bool)
        groups = np.array([0, 0, 1, 0, 0, 0], dtype=np.uint8)
        geom_id = np.zeros(1, dtype=np.int32)
        for i, p in enumerate(xy):
            origin = np.r_[p, pose[2] + 20]
            distance = mujoco.mj_ray(self.model, self.data, origin, np.array([0., 0., -1.]), groups, True, -1, geom_id)
            valid[i] = 0 <= distance <= 30
            if valid[i]:
                hits[i] = origin + [0, 0, -distance]
        mask = valid.copy()
        if self.noisy_height:
            mask &= self.rng.random(len(mask)) >= .1
        height = pose[2] - np.where(mask, hits[:, 2], 30.)
        if self.noisy_height:
            height += self.rng.uniform(-.06, .06, len(height))
        height = np.clip(height, -3, 3)
        return {"hits_w": hits.tolist(), "hit_valid": valid.tolist(),
                "policy_height": height.tolist(), "policy_valid": mask.tolist()}

    def history(self, name, value):
        value = np.asarray(value, dtype=np.float32)
        if name not in self.histories:
            self.histories[name] = np.repeat(value[None], 10, axis=0)
        else:
            self.histories[name] = np.concatenate([self.histories[name][1:], value[None]], axis=0)
        return self.histories[name]

    def observations(self, vision_override=None):
        q = self.data.qpos[self.qids] - self.default_q
        dq = self.data.qvel[self.vids] - self.default_dq
        omega = self.data.qvel[3:6]
        gravity = rotation(self.data.qpos[3:7]).inv().apply([0, 0, -1])
        proprio = np.r_[gravity, omega, q, dq, self.previous_action]
        ref = self.raw
        count = len(ref["joint_pos"])
        frame = self.frame
        command = np.r_[ref["joint_pos"][frame], ref["joint_vel"][frame]]
        future = np.minimum(frame + np.arange(10) * 5, count - 1)
        sonic = np.concatenate([ref["joint_pos"][future], ref["joint_vel"][future]], axis=-1).ravel()
        policy = np.concatenate([self.history("command", command).ravel(), sonic,
                                 self.history("omega", omega).ravel(), self.history("q", q).ravel(),
                                 self.history("dq", dq).ravel(), self.history("action", self.previous_action).ravel()])
        window = np.clip(frame + np.arange(-10, 11), 0, count - 1)
        delta = ref["body_pos_w"][window, self.anchor_index] - ref["body_pos_w"][frame, self.anchor_index]
        delta = rotation(ref["body_quat_w"][frame, self.anchor_index]).inv().apply(delta)
        self.last_scan = self.scan() if self.perception is None else self.perception.scan(self.data)
        vision = np.r_[self.last_scan["policy_height"], self.last_scan["policy_valid"]] if vision_override is None else vision_override
        values = {"policy": policy, "proprio": proprio,
                  "proprio_history": self.history("proprio", proprio),
                  "command_window": self.command_features[window], "motion_anchor_delta_window": delta,
                  "vision": vision}
        return TensorDict({k: torch.as_tensor(np.asarray(v).copy(), dtype=torch.float32).unsqueeze(0) for k, v in values.items()}, batch_size=[1])

    def action_targets(self, action):
        """Vision student residual targets; other policy families override this."""
        return self.raw["joint_pos"][self.frame] + np.asarray(action)

    def step(self, action):
        action = np.asarray(action).reshape(29)
        if not np.isfinite(action).all():
            raise FloatingPointError("Non-finite policy action.")
        self.data.ctrl[:] = self.action_targets(action)
        for _ in range(self.contract["decimation"]):
            mujoco.mj_step(self.model, self.data)
            self.physics_steps += 1
        mujoco.mj_forward(self.model, self.data)
        self.previous_action = action.copy()
        self.frame = min(self.frame + 1, len(self.raw["joint_pos"]) - 1)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Non-finite MuJoCo physics state.")


def load_policy(contract):
    if contract.get("policy_class") == "PGMTActorCritic":
        from motion_tracking_rl.networks.pgmt_actor_critic import PGMTActorCritic as policy_class
    else:
        from motion_tracking_rl.networks.vision_transformer_actor_critic import VisionTransformerActorCritic as policy_class

    schema = contract["policy_metadata"]["obs_schema"]["available_obs"]
    dummy = TensorDict({name: torch.zeros(1, *spec["shape"]) for name, spec in schema.items()}, batch_size=[1])
    policy = policy_class(dummy, contract["obs_groups"], 29, **contract["policy_kwargs"])
    state = torch.load(contract["checkpoint"], map_location="cpu", weights_only=False)["model_state_dict"]
    policy.load_state_dict(state, strict=True)
    policy.eval()
    return policy


def validate_contract(contract, adapter, policy):
    """Compare against real Isaac fixtures before trusting a MuJoCo rollout."""
    if isinstance(adapter, PGMTAdapter):
        return validate_pgmt_contract(contract, adapter, policy)
    if not contract.get("fixtures"):
        raise ValueError("A MuJoCo contract must contain Isaac parity fixtures.")
    errors = {}
    body_error = ray_error = action_error = offset_error = 0.
    adapter.histories.clear()
    with torch.inference_mode():
        for fixture in contract["fixtures"]:
            if not hasattr(adapter, "motion_id") or adapter.motion_id != fixture["motion_id"]:
                adapter.load_motion(fixture["motion_id"])
            adapter.frame = fixture["frame"]
            adapter.set_state(fixture["state"])
            adapter.previous_action = np.asarray(fixture["previous_action"])
            actual = adapter.observations(vision_override=fixture["obs"]["vision"][0].numpy())
            for key, value in actual.items():
                error = float(torch.max(torch.abs(value - fixture["obs"][key])))
                errors[key] = max(errors.get(key, 0), error)
            predicted = policy.act_inference(actual)
            action_error = max(action_error, float(torch.max(torch.abs(predicted - fixture["action"]))))
            body_error = max(body_error, float(np.max(np.linalg.norm(adapter.data.xpos[adapter.body_ids] - np.asarray(fixture["state"]["body_pos_w"]), axis=-1))))
            ray_error = max(ray_error, float(np.max(np.abs(np.asarray(adapter.last_scan["hits_w"]) - np.asarray(fixture["hits_w"])))))
            offset_error = max(offset_error, float(np.max(np.abs(adapter.raw["joint_pos"][adapter.frame] - np.asarray(fixture["action_offset"])))))
    result = {"fixtures": len(contract["fixtures"]), "max_observation_errors": errors,
              "max_action_error_rad": action_error, "max_body_fk_error_m": body_error,
              "max_ray_hit_error_m": ray_error, "max_action_offset_error_rad": offset_error}
    result["passed"] = max(errors.values()) < 2e-4 and action_error < 2e-4 and body_error < 1e-4 and ray_error < 2e-4 and offset_error < 1e-6
    return result


class PGMTAdapter(StudentAdapter):
    """PGMT observations on PMT's native URDF, actuators and CPU physics loop."""

    def __init__(self, contract, model, seed=42, noisy_height=False):
        super().__init__(contract, model, seed, noisy_height=False)
        self.anchor_index = contract["body_names"].index(contract.get("anchor_body", "torso_link"))
        self.action_scale = np.asarray(contract["action_scale"], dtype=np.float64).reshape(29)
        self.action_offset = np.asarray(contract.get("action_offset", self.default_q), dtype=np.float64).reshape(29)
        self.previous_target = self.action_offset.copy()
        self.env_origin = np.zeros(3, dtype=np.float32)
        self._history_stamp = None
        self.scan_config = None
        self.last_scan = None

    def load_motion(self, motion_id):
        files = self.contract["motion_files"]
        if not 0 <= motion_id < len(files):
            raise IndexError(f"Motion index {motion_id} outside 0..{len(files) - 1}")
        self.motion_id = motion_id
        with np.load(files[motion_id], allow_pickle=False) as data:
            self.raw = {k: data[k] for k in data.files}
        self.optimized = self.raw
        fps = float(np.asarray(self.raw["fps"]).reshape(-1)[0])
        if not np.isclose(fps * self.contract["sim_dt"] * self.contract["decimation"], 1.):
            raise ValueError("Motion fps must match control rate.")
        count = len(self.raw["joint_pos"])
        expected = {"joint_pos": (count, 29), "joint_vel": (count, 29),
                    "body_pos_w": (count, len(self.body_ids), 3),
                    "body_quat_w": (count, len(self.body_ids), 4),
                    "body_lin_vel_w": (count, len(self.body_ids), 3),
                    "body_ang_vel_w": (count, len(self.body_ids), 3)}
        for key, shape in expected.items():
            if self.raw[key].shape != shape or not np.isfinite(self.raw[key]).all():
                raise ValueError(f"Invalid PGMT motion field {key}: expected finite {shape}")
        for key in ("joint_names", "body_names"):
            if key in self.raw and self.raw[key].tolist() != self.contract[key]:
                raise ValueError(f"Motion {key} differ from the exported robot order.")
        self.reference_start = 0
        self.reference_end = len(self.raw["joint_pos"])

    def select_chunk(self, frame):
        chunk = int(self.contract.get("chunk_length", 0))
        self.reference_start = (frame // chunk) * chunk if chunk > 0 else 0
        self.reference_end = min(self.reference_start + chunk, len(self.raw["joint_pos"])) if chunk > 0 else len(self.raw["joint_pos"])

    def reset(self, motion_id, frame=0):
        self.env_origin[:] = 0
        super().reset(motion_id, frame)
        self.select_chunk(frame)
        # Use the actual target after the exported Isaac initial reset. Native
        # reset may retain a target instead of assigning the reference pose.
        self.previous_target = np.asarray(self.contract.get("initial_joint_target", self.action_offset), dtype=np.float64).copy()
        self.data.ctrl[:] = self.previous_target
        self.physics_steps = 0
        self._history_stamp = None

    def action_targets(self, action):
        return self.action_offset + self.action_scale * np.asarray(action)

    def step(self, action):
        super().step(action)
        self.previous_target = self.data.ctrl.copy()
        self.frame = min(self.frame, self.reference_end - 1)

    def _reference(self, frames):
        result = {}
        for name in ("joint_pos", "joint_vel", "body_pos", "body_quat", "body_lin_vel", "body_ang_vel"):
            key = name if name.startswith("joint") else name + "_w"
            array = np.asarray(self.raw[key][frames]).copy()
            if name == "body_pos":
                array += self.env_origin
            result[name] = torch.as_tensor(array, dtype=torch.float32).unsqueeze(0)
        return result

    def observations(self, vision_override=None):
        from pmt_tasks.pgmt_math import TRACKING_CONFIG, build_observations, future_offsets

        def tensor(value):
            return torch.as_tensor(np.asarray(value).copy(), dtype=torch.float32).unsqueeze(0)

        # Joint qvel and floating-base angular qvel use MuJoCo's local root frame;
        # its translational qvel is the root-link world velocity.
        state = {
            "joint_pos": tensor(self.data.qpos[self.qids]),
            "joint_vel": tensor(self.data.qvel[self.vids]),
            "body_pos": tensor(self.data.xpos[self.body_ids]),
            "body_quat": tensor(self.data.xquat[self.body_ids]),
            "root_quat": tensor(self.data.qpos[3:7]),
            "root_ang_vel": tensor(rotation(self.data.qpos[3:7]).apply(self.data.qvel[3:6])),
            "root_lin_vel": tensor(self.data.qvel[:3]),
        }
        offsets = future_offsets(int(self.contract.get("command_frames", 8)), device="cpu").numpy()
        frames = np.minimum(self.frame + offsets, self.reference_end - 1)
        values = build_observations(state, self._reference(self.frame), self._reference(frames),
                                    tensor(self.previous_target), tensor(self.env_origin),
                                    anchor=self.anchor_index, config=self.contract.get("tracking_config", TRACKING_CONFIG))
        proprio = values["proprio"][0].numpy()
        stamp = (self.motion_id, self.frame, self.physics_steps)
        if stamp != self._history_stamp:
            if "proprio" not in self.histories:
                self.histories["proprio"] = np.repeat(proprio[None], 10, axis=0)
            self.histories["proprio"] = np.concatenate((self.histories["proprio"][1:], proprio[None]), axis=0)
            self._history_stamp = stamp
        values["proprio_history"] = tensor(self.histories["proprio"])
        return TensorDict(values, batch_size=[1])


def create_adapter(contract, model, seed=42, noisy_height=True):
    if contract.get("policy_class") == "PGMTActorCritic":
        return PGMTAdapter(contract, model, seed)
    return StudentAdapter(contract, model, seed, noisy_height)


def validate_pgmt_contract(contract, adapter, policy):
    """Validate actor, critic, history, FK and decoded targets on Isaac fixtures."""
    fixtures = contract.get("fixtures", [])
    if not fixtures:
        raise ValueError("A PGMT contract must contain Isaac parity fixtures.")
    errors = {}
    action_error = body_error = target_error = 0.
    adapter.histories.clear()
    with torch.inference_mode():
        for index, fixture in enumerate(fixtures):
            if not hasattr(adapter, "motion_id") or adapter.motion_id != fixture["motion_id"]:
                adapter.load_motion(fixture["motion_id"])
            adapter.frame = int(fixture["frame"])
            adapter.reference_start = int(fixture["reference_start"])
            adapter.reference_end = int(fixture["reference_end"])
            adapter.env_origin = np.asarray(fixture.get("env_origin", np.zeros(3)), dtype=np.float32)
            adapter.set_state(fixture["state"])
            adapter.previous_target = np.asarray(fixture["previous_target"])
            adapter.physics_steps = index
            if index == 0:
                # The rollout may start mid-history; seed only the existing prefix.
                history = np.asarray(fixture["obs"]["proprio_history"])[0]
                adapter.histories["proprio"] = np.concatenate((np.zeros_like(history[:1]), history[:-1]), axis=0)
            adapter._history_stamp = None
            actual = adapter.observations()
            for key, value in actual.items():
                error = float((value - fixture["obs"][key]).abs().max())
                errors[key] = max(errors.get(key, 0.), error)
            prediction = policy.act_inference(actual)
            action_error = max(action_error, float((prediction - fixture["action"]).abs().max()))
            body_error = max(body_error, float(np.linalg.norm(adapter.data.xpos[adapter.body_ids] - np.asarray(fixture["state"]["body_pos_w"]), axis=-1).max()))
            target_error = max(target_error, float(np.max(np.abs(adapter.action_targets(prediction[0].numpy()) - np.asarray(fixture["action_target"])))))
    return {
        "fixtures": len(fixtures), "max_observation_errors": errors,
        "max_action_error": action_error, "max_body_fk_error_m": body_error,
        "max_action_target_error_rad": target_error,
        "history_prefix_seeded_from_first_fixture": True,
        "passed": max(errors.values()) < 2e-4 and action_error < 2e-4 and body_error < 1e-4 and target_error < 2e-4,
    }
