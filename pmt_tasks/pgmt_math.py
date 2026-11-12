"""PGMT stage-1 tensor math shared by Isaac and standalone MuJoCo.

Formulas and field meanings follow PGMT/pgmt/environment.py. Quaternions are
wxyz, body arrays use the full robot body order, and rewards are rates: the
Isaac RewardManager applies dt exactly once.
"""
from __future__ import annotations

import torch

from pmt_tasks.pgmt_tracking_relaxation import RELAXATION_TERMS, relax_tracking_error

TRACKING_CONFIG = {
    "body_position_sigma": 0.15, "body_orientation_sigma": 0.4,
    "body_linear_velocity_sigma": 1.0, "body_angular_velocity_sigma": 3.14,
    "joint_position_sigma": 0.4, "joint_velocity_sigma": 5.0,
    "root_orientation_sigma": 0.4, "root_velocity_sigma": 1.0,
    "anchor_position_sigma": 0.5, "anchor_deadzone": 0.3,
    "position_gain": 0.5, "correction_cap": 0.5, "speed_gate_saturation": 0.5,
}
VALUE_GROUPS = ("upper", "lower", "auxiliary")
REWARD_WEIGHTS = {
    "upper_body_position": 1.0, "upper_body_orientation": 1.0,
    "upper_body_linear_velocity": 0.5, "upper_body_angular_velocity": 0.5,
    "upper_joint_position": 1.0, "upper_joint_velocity": 0.5,
    "lower_body_position": 0.5, "lower_body_orientation": 2.0,
    "lower_body_linear_velocity": 0.5, "lower_body_angular_velocity": 0.5,
    "lower_joint_position": 0.5, "lower_joint_velocity": 0.5,
    "auxiliary_root_orientation": 0.5, "auxiliary_root_velocity": 2.0,
    "auxiliary_anchor_position": 1.0,
    "auxiliary_base_acceleration": -0.001,
    "auxiliary_end_effector_acceleration": -0.001,
    "auxiliary_action_rate": -0.05, "auxiliary_joint_limit": -15.0,
    "auxiliary_undesired_contacts": -0.1, "auxiliary_anchor_contact_force": -1e-5,
}


def conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def multiply(q, r):
    w1, v1 = q[..., :1], q[..., 1:]
    w2, v2 = r[..., :1], r[..., 1:]
    return torch.cat((w1 * w2 - (v1 * v2).sum(-1, keepdim=True),
                      w1 * v2 + w2 * v1 + torch.cross(v1, v2, dim=-1)), -1)


def rotate(q, v):
    uv = torch.cross(q[..., 1:], v, dim=-1)
    return v + 2 * (q[..., :1] * uv + torch.cross(q[..., 1:], uv, dim=-1))


def inverse_rotate(q, v):
    return rotate(conjugate(q), v)


def yaw_quat(q):
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    out = torch.zeros_like(q)
    out[..., 0], out[..., 3] = torch.cos(yaw / 2), torch.sin(yaw / 2)
    return out


def angle(q, r):
    e = multiply(q, conjugate(r))
    return 2 * torch.atan2(e[..., 1:].norm(dim=-1), e[..., 0].abs().clamp_min(1e-9))


def rotation6(q):
    x = torch.zeros(q.shape[:-1] + (3,), device=q.device, dtype=q.dtype)
    x[..., 0] = 1
    y = torch.zeros_like(x)
    y[..., 1] = 1
    return torch.cat((rotate(q, x), rotate(q, y)), -1)


def future_offsets(command_frames=8, *, device=None):
    return 2 ** torch.arange(command_frames, device=device, dtype=torch.long) - 1


def corrected_command(state, ref, *, anchor, config=TRACKING_CONFIG):
    anchorq = ref["body_quat"][:, anchor]
    velocity = inverse_rotate(anchorq, ref["body_lin_vel"][:, anchor])
    error = inverse_rotate(anchorq, ref["body_pos"][:, anchor] - state["body_pos"][:, anchor])
    gate = (velocity[:, :2].norm(dim=-1, keepdim=True) / config["speed_gate_saturation"]).clamp(0, 1)
    velocity[:, :2] += (gate * config["position_gain"] * error[:, :2]).clamp(
        -config["correction_cap"], config["correction_cap"])
    velocity[:, 2] = 0
    return velocity


def build_observations(state, ref, future, qtarget, env_origins, *, anchor, config=TRACKING_CONFIG):
    """Build clean proprio [N,96], future commands [N,8,61], critic [N,251].

    ``state`` and ``ref`` have joint_pos/joint_vel/body_pos/body_quat/
    body_lin_vel/body_ang_vel. State additionally has root_quat/root_lin_vel/
    root_ang_vel (world frame). Future has the same reference keys with a
    window dimension after N. All body fields include all 30 G1 bodies.
    """
    error = multiply(conjugate(state["body_quat"][:, anchor]), ref["body_quat"][:, anchor])
    orient = rotation6(error)
    omega = inverse_rotate(state["root_quat"], state["root_ang_vel"])
    proprio = torch.cat((orient, omega, state["joint_pos"], state["joint_vel"] * 0.05, qtarget), -1)
    current = corrected_command(state, ref, anchor=anchor, config=config)
    velocity = inverse_rotate(future["body_quat"][:, :, anchor], future["body_lin_vel"][:, :, anchor])
    nominal = inverse_rotate(ref["body_quat"][:, anchor], ref["body_lin_vel"][:, anchor])
    velocity[:, :, :2] += (current[:, :2] - nominal[:, :2])[:, None]
    velocity[:, :, 2] = 0
    commands = torch.cat((future["joint_pos"], future["joint_vel"] * 0.05, velocity), -1)
    nb = state["body_pos"].shape[1]
    rq = yaw_quat(state["body_quat"][:, anchor])[:, None].expand(-1, nb, -1)
    tq = yaw_quat(ref["body_quat"][:, anchor])[:, None].expand_as(rq)
    local = inverse_rotate(rq, state["body_pos"] - state["body_pos"][:, anchor, None])
    target = inverse_rotate(tq, ref["body_pos"] - ref["body_pos"][:, anchor, None])
    critic = torch.cat((orient, inverse_rotate(state["root_quat"], state["root_lin_vel"]), omega,
                        state["joint_pos"], state["joint_vel"] * 0.05, qtarget,
                        ref["joint_pos"], ref["joint_vel"] * 0.05, (local - target).flatten(1),
                        ref["body_pos"][:, 0, 2:3] - env_origins[:, 2:3], current), -1)
    return {"proprio": proprio, "command_window": commands, "critic": critic}


def body_groups(body_names, joint_names, *, device=None, end_effector_names=None):
    """Resolve body sets; end_effectors is used only by acceleration mismatch.

    The default preserves Stage1's two feet and two wrists. Stage2 supplies
    wrist names so terrain-driven foot accelerations are excluded from this term.
    """
    def containing(names, parts):
        return torch.tensor([i for i, name in enumerate(names) if any(p in name for p in parts)],
                            dtype=torch.long, device=device)
    feet = [body_names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    if end_effector_names is None:
        end_effector_names = ("left_ankle_roll_link", "right_ankle_roll_link",
                              "left_wrist_yaw_link", "right_wrist_yaw_link")
    if (isinstance(end_effector_names, str) or not end_effector_names
            or len(set(end_effector_names)) != len(end_effector_names)):
        raise ValueError("EE acceleration bodies must be a nonempty sequence of unique body names")
    ee = [body_names.index(name) for name in end_effector_names]
    return {
        "upper": containing(body_names, ("shoulder", "elbow", "wrist", "torso")),
        "lower": containing(body_names, ("hip", "knee", "ankle")),
        "upper_joints": containing(joint_names, ("shoulder", "elbow", "wrist", "waist")),
        "lower_joints": containing(joint_names, ("hip", "knee", "ankle")),
        "end_effectors": torch.tensor(ee, dtype=torch.long, device=device),
        "undesired": torch.tensor([i for i in range(len(body_names)) if i not in feet],
                                   dtype=torch.long, device=device),
    }


def reward_terms(state, ref, qtarget, previous_target, previous_velocity, previous_ref_velocity,
                 forces, joint_limits, groups, *, anchor, dt, config=TRACKING_CONFIG,
                 tracking_margins=None, tracking_debug=None):
    """Unweighted rates; optional lower-only deadzones precede square/mean/exp.

    tracking_margins comes from TrackingRelaxation. With None the original
    strict path is unchanged. An optional dict collects owned error diagnostics.
    No reward weight or dt is applied here.
    """
    nb = state["body_pos"].shape[1]
    rq = yaw_quat(state["body_quat"][:, anchor])[:, None].expand(-1, nb, -1)
    tq = yaw_quat(ref["body_quat"][:, anchor])[:, None].expand_as(rq)
    local = inverse_rotate(rq, state["body_pos"] - state["body_pos"][:, anchor, None])
    target = inverse_rotate(tq, ref["body_pos"] - ref["body_pos"][:, anchor, None])
    errors = {
        "body_position": (local - target).norm(dim=-1),
        "body_orientation": angle(multiply(conjugate(rq), state["body_quat"]),
                                   multiply(conjugate(tq), ref["body_quat"])),
        "body_linear_velocity": (inverse_rotate(rq, state["body_lin_vel"]) - inverse_rotate(tq, ref["body_lin_vel"])).norm(dim=-1),
        "body_angular_velocity": (inverse_rotate(rq, state["body_ang_vel"]) - inverse_rotate(tq, ref["body_ang_vel"])).norm(dim=-1),
        "joint_position": (state["joint_pos"] - ref["joint_pos"]).abs(),
        "joint_velocity": (state["joint_vel"] - ref["joint_vel"]).abs(),
    }
    def exp(x, sigma):
        return torch.exp(-x.square().mean(-1) / sigma ** 2)
    result = {}
    for group in ("upper", "lower"):
        for term, error in errors.items():
            ids = groups[group + "_joints"] if term.startswith("joint") else groups[group]
            name = f"{group}_{term}"
            selected = error[:, ids]
            raw = selected
            if tracking_margins is not None and name in RELAXATION_TERMS:
                selected = relax_tracking_error(raw, tracking_margins[name])
            result[name] = exp(selected, config[term + "_sigma"])
            if tracking_debug is not None and name in RELAXATION_TERMS:
                margin = torch.zeros_like(raw) if tracking_margins is None else tracking_margins[name]
                tracking_debug.update({f"{name}_raw_error": raw.clone(), f"{name}_margin": margin.clone(),
                                       f"{name}_relaxed_error": selected.clone(),
                                       f"{name}_strict_rate": exp(raw, config[term + "_sigma"]),
                                       f"{name}_relaxed_rate": result[name].clone()})
    rootangle = angle(state["body_quat"][:, anchor], ref["body_quat"][:, anchor])
    vactual = inverse_rotate(ref["body_quat"][:, anchor], state["body_lin_vel"][:, anchor])
    vtarget = corrected_command(state, ref, anchor=anchor, config=config)
    drift = (state["body_pos"][:, anchor, :2] - ref["body_pos"][:, anchor, :2]).norm(dim=-1)
    acceleration = (state["body_lin_vel"] - previous_velocity) / dt
    refacc = (ref["body_lin_vel"] - previous_ref_velocity) / dt
    lower, upper = joint_limits.unbind(-1)
    result.update({
        "auxiliary_root_orientation": torch.exp(-rootangle.square() / config["root_orientation_sigma"] ** 2),
        "auxiliary_root_velocity": exp(vactual[:, :2] - vtarget[:, :2], config["root_velocity_sigma"]),
        "auxiliary_anchor_position": torch.exp(-(drift - config["anchor_deadzone"]).clamp_min(0).square() / config["anchor_position_sigma"] ** 2),
        "auxiliary_base_acceleration": acceleration[:, 0, 2].square().clamp_max(10000),
        "auxiliary_end_effector_acceleration": (acceleration[:, groups["end_effectors"]] - refacc[:, groups["end_effectors"]]).square().mean((1, 2)).clamp_max(10000),
        "auxiliary_action_rate": (qtarget - previous_target).square().sum(-1),
        "auxiliary_joint_limit": ((lower - state["joint_pos"]).clamp_min(0) + (state["joint_pos"] - upper).clamp_min(0)).sum(-1),
        "auxiliary_undesired_contacts": (forces[:, groups["undesired"]].norm(dim=-1) > 10).float().sum(-1),
        "auxiliary_anchor_contact_force": forces[:, anchor].square().sum(-1),
    })
    return result


def grouped_rewards(terms, weights=REWARD_WEIGHTS):
    return torch.stack([sum(terms[name] * weight for name, weight in weights.items()
                            if name.startswith(group + "_")) for group in VALUE_GROUPS], -1)
