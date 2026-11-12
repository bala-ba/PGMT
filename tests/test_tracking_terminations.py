"""Distinguish PMT's torso gravity test from SONIC's pelvis SO(3) test."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pmt_tasks.pgmt_math import inverse_rotate
from pmt_tasks.tracking_terminations import failure_signals, first_stop


NAMES = ["pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link",
         "left_wrist_yaw_link", "right_wrist_yaw_link"]
PMT = dict(anchor="torso_link", anchor_height_m=.4, orientation_kind="projected_gravity_z_difference",
           orientation_threshold=.8, ee_height_m=.4, end_effectors=NAMES[2:])
SONIC = dict(anchor="pelvis", anchor_height_m=.25, orientation_kind="squared_full_rotation_angle_rad2",
             orientation_threshold=1., ee_height_m=.25, end_effectors=NAMES[2:])


def test_yaw_is_ignored_by_pmt_but_detected_by_sonic():
    pos = np.zeros((1, 6, 3))
    actual, ref = [[np.cos(.6), 0, 0, np.sin(.6)]], [[1., 0, 0, 0]]
    _, pmt = failure_signals(pos, pos, actual, ref, NAMES, PMT)
    _, sonic = failure_signals(pos, pos, actual, ref, NAMES, SONIC)
    assert not any(v[0] for v in pmt.values())
    assert sonic["anchor_orientation"][0]


def test_anchor_choice_and_end_effectors():
    pos, ref = np.zeros((2, 6, 3)), np.zeros((2, 6, 3))
    pos[0, 1, 2] = .41
    pos[1, 0, 2] = .26
    q = np.tile([1., 0, 0, 0], (2, 1))
    _, pmt = failure_signals(pos, ref, q, q, NAMES, PMT)
    _, sonic = failure_signals(pos, ref, q, q, NAMES, SONIC)
    np.testing.assert_array_equal(pmt["anchor_height"], [True, False])
    np.testing.assert_array_equal(sonic["anchor_height"], [False, True])
    pos = ref.copy()
    pos[0, 4, 2] = .3
    _, pmt = failure_signals(pos, ref, q, q, NAMES, PMT)
    _, sonic = failure_signals(pos, ref, q, q, NAMES, SONIC)
    assert not pmt["end_effector_height"][0] and sonic["end_effector_height"][0]


def test_pmt_predicates_match_original_source_functions():
    # Execute the repository's actual predicate function bodies without starting
    # Isaac. Stub only the manager/asset access needed to feed known states.
    source = Path(__file__).resolve().parents[1] / "pmt_tasks/mdp/terminations.py"
    wanted = {"exceeded_tracking_error", "bad_anchor_pos_z_only", "bad_anchor_ori",
              "exceeded_body_height", "bad_motion_body_pos_z_only"}
    nodes = [node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    namespace = {"torch": torch, "math_utils": SimpleNamespace(quat_apply_inverse=inverse_rotate),
                 "_get_body_indexes": lambda command, names: [NAMES.index(n) for n in names]}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    rng = np.random.default_rng(5)
    actual, reference = rng.normal(0, .3, (128, 6, 3)), rng.normal(0, .3, (128, 6, 3))
    aq, bq = rng.normal(size=(128, 4)), rng.normal(size=(128, 4))
    aq /= np.linalg.norm(aq, axis=-1, keepdims=True)
    bq /= np.linalg.norm(bq, axis=-1, keepdims=True)
    tensor = torch.as_tensor
    command = SimpleNamespace(anchor_pos_w=tensor(reference[:, 1]), robot_anchor_pos_w=tensor(actual[:, 1]),
        anchor_quat_w=tensor(bq), robot_anchor_quat_w=tensor(aq),
        body_pos_relative_w=tensor(reference), robot_body_pos_w=tensor(actual))
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: command),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(GRAVITY_VEC_W=tensor(np.tile([0., 0., -1.], (128, 1)))) )})
    _, flags = failure_signals(actual, reference, aq, bq, NAMES, PMT)
    expected = {
        "anchor_height": namespace["bad_anchor_pos_z_only"](env, "motion", .4),
        "anchor_orientation": namespace["bad_anchor_ori"](env, SimpleNamespace(name="robot"), "motion", .8),
        "end_effector_height": namespace["bad_motion_body_pos_z_only"](env, "motion", .4, NAMES[2:]),
    }
    for key, value in flags.items():
        np.testing.assert_array_equal(value, expected[key].numpy())


def test_first_failure_is_included_and_timeout_is_not_full_success():
    flags = {"height": np.zeros(1000, bool), "orientation": np.zeros(1000, bool)}
    flags["height"][799] = True
    full = first_stop(flags, 1001, .02)
    assert full["status"] == "failure" and full["evaluated_frames"] == 800
    assert full["motion_completion"] == pytest.approx(.799)
    timeout = first_stop(flags, 1001, .02, 15)
    assert timeout["status"] == "timeout" and timeout["evaluated_frames"] == 750
    assert timeout["horizon_passed"] and not timeout["full_motion_success"]
    assert timeout["motion_completion"] == pytest.approx(.75)
    flags["height"][749] = True
    assert first_stop(flags, 1001, .02, 15)["status"] == "failure"
    flags["height"][:] = False
    assert first_stop(flags, 1001, .02)["full_motion_success"]
