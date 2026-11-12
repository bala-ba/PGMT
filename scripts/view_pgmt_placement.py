"""Show real Isaac Lab reset states in Isaac Sim 6 when the 5.1 GUI is incompatible.

First run inspect_pgmt_placement.py --headless --validate --export-scene with
the PMT Python environment. This viewer holds exported native link poses;
it does not replay motion, advance physics or infer policy actions.
"""
import argparse
import json
import os
from pathlib import Path
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--snapshot", default="logs/pgmt_placement")
parser.add_argument("--gpu", type=int, default=1)
parser.add_argument("--family", default="up_stairs")
parser.add_argument("--level", type=int, default=5)
parser.add_argument("--yaw-deg", type=float, default=90.)
args = parser.parse_args()
directory = Path(args.snapshot).resolve()
data = json.loads((directory / "reset_poses.json").read_text())
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp

app = SimulationApp(dict(headless=False, width=1600, height=1000,
    window_width=1600, window_height=1000, multi_gpu=False, active_gpu=args.gpu,
    physics_gpu=args.gpu, limit_cpu_threads=8,
    extra_args=["--/app/useFabricSceneDelegate=false",
                "--/rtx/hydra/readTransformsFromFabricInRenderDelegate=false"]))

import numpy as np
import carb
import omni.usd
import omni.ui as ui
import omni.timeline
from pxr import Gf, UsdGeom, UsdLux
from isaacsim.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file


def status(state, **values):
    file = directory / "viewer_status.json"
    temporary = file.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(state=state, pid=os.getpid(), time=time.time(),
        renderer="Isaac Sim 6", source=data["source"], **values), indent=2) + "\n")
    temporary.replace(file)


def main():
    status("loading")
    if not omni.usd.get_context().open_stage(str(directory / data["scene"])):
        raise RuntimeError("Could not open the exported PMT scene")
    for _ in range(40):
        app.update()
    omni.timeline.get_timeline_interface().stop()
    stage = omni.usd.get_context().get_stage()
    camera_path = "/World/PGMTInspectionCamera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr(30.)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 2000.))
    viewport = get_active_viewport()
    viewport.camera_path = camera_path
    carb.settings.get_settings().set_int("/persistent/app/viewport/displayOptions", 0)
    # The source terrain and environment each have lights. Keep lighting
    # consistent with the previously inspected v5 terrain viewer.
    for prim in stage.Traverse():
        if prim.IsA(UsdLux.DomeLight):
            UsdLux.DomeLight(prim).GetIntensityAttr().Set(180.)
        elif prim.IsA(UsdLux.DistantLight):
            UsdLux.DistantLight(prim).GetIntensityAttr().Set(900.)
    families = data["families"]
    cases = {(case["level"], case["yaw_deg"]): case for case in data["cases"]}
    levels = sorted({key[0] for key in cases})
    yaws = sorted({key[1] for key in cases})
    selection = dict(family=families.index(args.family), level=args.level, yaw=args.yaw_deg)
    if (selection["level"], selection["yaw"]) not in cases:
        raise ValueError("Requested level/yaw was not exported")
    link_paths = data["link_paths"]
    body_xforms = []
    for paths in link_paths:
        group = []
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid() or prim.IsInstanceProxy():
                raise ValueError(f"Cannot author exported rigid link: {path}")
            xform = UsdGeom.Xformable(prim)
            op = xform.MakeMatrixXform()
            xform.SetResetXformStack(True)
            group.append(op)
        body_xforms.append(group)

    def apply_case():
        case = cases[selection["level"], selection["yaw"]]
        for i, ops in enumerate(body_xforms):
            for j, op in enumerate(ops):
                p, q = case["body_pos"][i][j], case["body_quat"][i][j]
                transform = Gf.Matrix4d(1.)
                transform.SetRotate(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))
                transform.SetTranslateOnly(Gf.Vec3d(*p))
                op.Set(transform)
        # Independently verify the rendered USD world transforms against the
        # PhysX link states exported by the real PMT reset, including parents.
        cache = UsdGeom.XformCache()
        max_error = 0.
        for i, paths in enumerate(link_paths):
            for j, path in enumerate(paths):
                matrix = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
                error = np.max(np.abs(np.asarray(matrix.ExtractTranslation()) - case["body_pos"][i][j]))
                max_error = max(max_error, float(error))
                q = matrix.ExtractRotationQuat()
                actual = np.array([q.GetReal(), *q.GetImaginary()])
                expected = np.asarray(case["body_quat"][i][j])
                expected /= np.linalg.norm(expected)
                if min(np.linalg.norm(actual-expected), np.linalg.norm(actual+expected)) > 2e-5:
                    raise AssertionError(f"USD orientation differs from PMT reset: {path}")
        if max_error > 2e-5:
            raise AssertionError(f"USD position differs from PMT reset: {max_error}")
        return max_error

    def focus(wide=False):
        case = cases[selection["level"], selection["yaw"]]
        root = np.asarray(case["body_pos"][selection["family"]][0])
        target = root + [0., 0., -.1]
        eye = target + ([22., -28., 25.] if wide else [2.4, -3., 2.0])
        set_camera_view(eye, target, camera_prim_path=camera_path)
        label.text = (f"{families[selection['family']]} | L{selection['level']} | yaw {selection['yaw']:g} deg\n"
                      f"Source frame: {data['start_frame']}\n"
                      "Actual PMT reset poses, held for inspection.\n"
                      "Physics and policy are not advancing.")
        status("running", family=families[selection["family"]], level=selection["level"],
               yaw_deg=selection["yaw"], start_frame=data["start_frame"], usd_body_position_error_m=last_error,
               inspection="actual_reset_pose_held")

    def choose_family(i):
        selection["family"] = i
        focus()

    def choose_case(key, value):
        nonlocal last_error
        selection[key] = value
        last_error = apply_case()
        focus()

    window = ui.Window("PGMT Initialization", width=480, height=350)
    with window.frame:
        with ui.VStack(spacing=6):
            ui.Label(Path(data["motion"]).name)
            ui.Label("Shared reference transform / real PMT reset")
            for start in (0, 4):
                with ui.HStack(height=30):
                    for i in range(start, min(start + 4, len(families))):
                        ui.Button(families[i], clicked_fn=lambda i=i: choose_family(i))
            ui.Label("Terrain level", height=22)
            with ui.HStack(height=30):
                for level in levels:
                    ui.Button(str(level), clicked_fn=lambda level=level: choose_case("level", level))
            with ui.HStack(height=30):
                ui.Label("Yaw", width=50)
                for yaw in yaws:
                    ui.Button(str(int(yaw)), clicked_fn=lambda yaw=yaw: choose_case("yaw", yaw))
            with ui.HStack(height=30):
                ui.Button("Robot close up", clicked_fn=lambda: focus(False))
                ui.Button("Whole tile", clicked_fn=lambda: focus(True))
            label = ui.Label("", word_wrap=True)
    last_error = apply_case()
    focus()
    for _ in range(60):
        app.update()
    screenshot = directory / "initialization.png"
    screenshot.unlink(missing_ok=True)
    capture_viewport_to_file(viewport, str(screenshot))
    print("PGMT INITIALIZATION VIEWER READY", flush=True)
    while app.is_running():
        app.update()
        time.sleep(.01)
    status("closed")


try:
    main()
except BaseException as error:
    status("failed", error=repr(error))
    raise
finally:
    app.close()
