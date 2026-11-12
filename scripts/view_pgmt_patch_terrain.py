"""Inspect versioned Stage2 patch terrain in Isaac Sim 6 (no training)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--asset", default=str(ROOT / "assets/terrain/pgmt_stage2_v8"))
parser.add_argument("--output", default=str(ROOT / "logs/pgmt_terrain_preview_v8"))
parser.add_argument("--family", help="Asset family name; defaults to its descending stairs")
parser.add_argument("--level", type=int, choices=range(10), default=5)
parser.add_argument("--patch", type=int, help="Patch index; defaults to 0 for P-BFM, 5 for legacy assets")
parser.add_argument("--view", choices=["overview", "region", "patch", "close"], default="region")
parser.add_argument("--show-samples", action="store_true", help="Show candidate surface XY/yaw; not validated robot poses")
parser.add_argument("--gpu", type=int, default=None)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--validate-only", action="store_true")
parser.add_argument("--skip-captures", action="store_true", help="Open the selected view without the automatic screenshot tour")
args = parser.parse_args()
if args.gpu is None:
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,display_active,pci.bus_id", "--format=csv,noheader"],
        text=True, timeout=10)
    rows = [[part.strip() for part in line.split(",")] for line in devices.splitlines()]
    connected = [int(row[0]) for row in rows if row[1] == "Enabled"]
    if not connected:
        buses = set()
        for status in Path("/sys/class/drm").glob("card*-*/status"):
            if status.read_text().strip() == "connected":
                card = status.parent.name.split("-", 1)[0]
                bus = (status.parent.parent / card / "device").resolve().name
                buses.add(":".join(bus.lower().split(":")[-2:]))
        connected = [int(row[0]) for row in rows if ":".join(row[2].lower().split(":")[-2:]) in buses]
    args.gpu = connected[0] if connected and not args.headless else 0
asset, output = Path(args.asset).resolve(), Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=True)
manifest = json.loads((asset / "manifest.json").read_text())
families = manifest["spec"]["families"]
grid = manifest["spec"]["patch_grid"]
patches_per_region = grid[0] * grid[1]
args.patch = (5 if patches_per_region > 5 else 0) if args.patch is None else args.patch
if not 0 <= args.patch < patches_per_region:
    parser.error(f"This asset has patch indices 0..{patches_per_region-1}")
args.family = args.family or ("stairs_down" if "stairs_down" in families else "stairs")
if args.family not in families:
    parser.error(f"Unknown family {args.family!r}; this asset has {families}")
if manifest["schema"] != "pmt.pgmt.terrain.patches.v1":
    raise ValueError("This viewer requires a patch-region manifest")
usd = asset / manifest["asset"]
if hashlib.sha256(usd.read_bytes()).hexdigest() != manifest["asset_sha256"]:
    raise ValueError("Terrain USD checksum mismatch")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp

app = SimulationApp(dict(headless=args.headless, width=1600, height=1000,
    window_width=1600, window_height=1000, multi_gpu=False, active_gpu=args.gpu,
    physics_gpu=args.gpu, limit_cpu_threads=8,
    extra_args=["--/app/useFabricSceneDelegate=false",
                "--/rtx/hydra/readTransformsFromFabricInRenderDelegate=false",
                "--/exts/omni.syntheticdata/menubar/visible=false"]))

import numpy as np
import carb
import omni.physx
import omni.timeline
import omni.ui as ui
import omni.usd
from pxr import Gf, UsdGeom, Vt
from isaacsim.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
from pmt_tasks.runtime_options import run_and_close


def save_status(state, **values):
    path = output / "status.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(state=state, pid=os.getpid(), display=os.getenv("DISPLAY"),
        asset=str(usd), time=time.time(), **values), indent=2) + "\n")
    temp.replace(path)


def validate_collision():
    """Probe the visible USD mesh's actual PhysX collision, including every seam."""
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(40):
        app.update()
    query = omni.physx.get_physx_scene_query_interface()
    surfaces = manifest["patches"] + manifest.get("auxiliary_surfaces", [])
    top = max(p["world_bounds"][1][2] for p in surfaces) + 5.
    bottom = min(p["world_bounds"][0][2] for p in surfaces) - 5.
    errors, family_errors, sample_errors = [], {}, []

    def probe(x, y, z, name):
        hit = query.raycast_closest((float(x), float(y), top), (0., 0., -1.), top - bottom)
        if not hit.get("hit"):
            raise RuntimeError(f"PhysX ray missed {name}: {(x, y)}")
        error = abs(float(hit["position"][2]) - z)
        if error > .002:
            raise RuntimeError(f"PhysX / mesh mismatch on {name}: {error:.6f} m at {(x, y)}")
        return error

    for p in surfaces:
        local = [probe(*xyz, p["name"]) for xyz in p["world_probes"]]
        # Candidate point Z is also queried from the actual collision, not a
        # height proxy. This still does not certify a robot footprint or pose.
        samples = [probe(*sample[:3], p["name"] + " candidate") for sample in p.get("preview_xyzyaw", [])]
        errors.extend(local)
        sample_errors.extend(samples)
        family_errors[p["family"]] = max(family_errors.get(p["family"], 0.), *local, *samples)
    seam_errors = [probe(*xyz, "patch seam") for xyz in manifest["seam_probes"]]
    timeline.pause()
    result = dict(passed=True, regions=manifest["region_count"], patches=manifest["patch_count"],
        geometry_rays=len(errors), candidate_surface_rays=len(sample_errors), seam_rays=len(seam_errors),
        total_rays=len(errors) + len(sample_errors) + len(seam_errors),
        max_height_error_m=max(errors + sample_errors + seam_errors),
        max_seam_error_m=max(seam_errors, default=0.), per_family_max_error_m=family_errors,
        tolerance_m=.002, renderer="Isaac Sim 6", physics="PhysX triangle-mesh collision",
        asset_sha256=manifest["asset_sha256"], robot_placement_validated=False)
    (output / "collision_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print("PHYSX VALIDATED:", json.dumps(result), flush=True)
    return result


def main():
    save_status("loading")
    if not omni.usd.get_context().open_stage(str(usd)):
        raise RuntimeError("Could not open terrain USD")
    for _ in range(60):
        app.update()
    stage = omni.usd.get_context().get_stage()
    carb.settings.get_settings().set_int("/persistent/app/viewport/displayOptions", 0)
    validation = validate_collision()
    if args.validate_only:
        save_status("validated", validation=validation)
        return

    viewport = get_active_viewport()
    viewport.camera_path = "/World/InspectionCamera"
    families = manifest["spec"]["families"]
    regions = {(r["family_index"], r["level"]): r for r in manifest["regions"]}
    state = dict(family=families.index(args.family), level=args.level, patch=args.patch,
                 mode=args.view, samples=args.show_samples)
    window = ui.Window(f"{manifest['version']} Patch Inspector", width=500, height=600)
    label = None

    def draw_samples(region):
        stage.RemovePrim("/World/PreviewSamples")
        if not state["samples"] or state["mode"] == "overview":
            return
        ids = region["patch_indices"]
        if state["mode"] in ("patch", "close"):
            ids = [ids[state["patch"]]]
        data = np.asarray([s for i in ids for s in manifest["patches"][i]["preview_xyzyaw"]])
        points = data[:, :3].copy()
        points[:, 2] += .06
        tips = points + np.column_stack((.3 * np.cos(data[:, 3]), .3 * np.sin(data[:, 3]), np.zeros(len(data))))
        dots = UsdGeom.Points.Define(stage, "/World/PreviewSamples/Points")
        dots.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        dots.CreateWidthsAttr([.07])
        dots.SetWidthsInterpolation("constant")
        dots.CreateDisplayColorAttr([Gf.Vec3f(1., .9, .05)])
        lines = UsdGeom.BasisCurves.Define(stage, "/World/PreviewSamples/Yaw")
        lines.CreateTypeAttr("linear")
        lines.CreateCurveVertexCountsAttr([2] * len(data))
        lines.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.stack((points, tips), axis=1).reshape(-1, 3).astype(np.float32)))
        lines.CreateWidthsAttr([.018])
        lines.SetWidthsInterpolation("constant")
        lines.CreateDisplayColorAttr([Gf.Vec3f(1., .9, .05)])

    def focus(mode=None):
        if mode is not None:
            state["mode"] = mode
        mode = state["mode"]
        region = regions[state["family"], state["level"]]
        patch = manifest["patches"][region["patch_indices"][state["patch"]]]
        for r in manifest["regions"]:
            visible = mode == "overview" or r is region
            UsdGeom.Imageable(stage.GetPrimAtPath(r["prim_path"])).GetVisibilityAttr().Set("inherited" if visible else "invisible")
            for i in r["patch_indices"]:
                p = manifest["patches"][i]
                selected = mode in ("overview", "region") or p is patch
                UsdGeom.Imageable(stage.GetPrimAtPath(p["prim_path"])).GetVisibilityAttr().Set("inherited" if selected else "invisible")
        for path in ["/World/TerrainBorder"] + [p["prim_path"] for p in manifest.get("auxiliary_surfaces", [])]:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                UsdGeom.Imageable(prim).GetVisibilityAttr().Set("inherited" if mode == "overview" else "invisible")
        patch_scale = max(manifest["spec"]["patch_size"]) / 6.
        region_scale = max(manifest["spec"]["region_size"]) / 24.
        if mode == "overview":
            bounds = np.array(manifest.get("walkable_bounds", manifest["map_bounds"]))
            target, eye = np.zeros(3), np.array([.75, -.95, 1.1]) * (bounds[1]-bounds[0]).max()
        elif mode == "region":
            target = np.array(region["center"], dtype=float)
            target[2] = .4 * patch["origin"][2]
            eye = target + np.array([27.6, -34.8, 40.8]) * region_scale
        elif mode == "patch":
            target = np.array(patch["center"], dtype=float)
            target[2] = .4 * patch["origin"][2]
            eye = target + np.array([7.25, -9.25, 10.625]) * patch_scale
        else:
            target = np.array(patch["center"], dtype=float)
            target[2] = patch["origin"][2]
            eye = target + np.array([3., -4., 3.5 if patch["family"] != "rough" else 2.]) * patch_scale
        set_camera_view(eye, target, camera_prim_path="/World/InspectionCamera")
        draw_samples(region)
        p = patch["parameters"]
        if "source_config" in p:
            source = p["source_config"]
            details = f"P-BFM source: {patch['source_family']}\n"
            for key in ("height_range", "step_height_range", "step_width", "grid_width", "grid_height_range",
                        "slope_range", "num_obstacles", "obstacle_width_range", "obstacle_height_range"):
                if key in source:
                    details += f"{key}: {source[key]}\n"
            details += f"Training proportion: {manifest['spec']['family_proportions'][patch['family']]:.0%}\n"
        elif "stair_up_height_m" in p:
            details = (
                f"Stairs up / down height: {p['stair_up_height_m']:.3f} / {p['stair_down_height_m']:.3f} m\n"
                f"Slope rise/run: {p['slope_coefficient']:.3f} ({p['slope_angle_deg']:.2f} deg)\n"
                f"Grid amplitude: +/-{p['platform_height_amplitude_m']:.3f} m\n"
                f"Discrete obstacle height: {p['obstacle_height_m']:.3f} m\n"
                f"Perlin amplitude: +/-{p['rough_height_amplitude_m']:.3f} m\n"
                f"Training proportion: {manifest['spec']['family_proportions'][patch['family']]:.0%}\n")
        else:
            details = (f"Stair height / depth: {p['stair_height_m']:.3f} / {p['stair_depth_m']:.3f} m\n"
                f"Slope input angle: {p['slope_angle_deg']:.2f} deg\n"
                f"Grid amplitude: +/-{p['box_height_max_m']:.3f} m\n"
                f"Rough amplitude: +/-{p['rough_amplitude_m']:.3f} m\n")
        label.text = (
            f"{patch['family']}  L{patch['level']}  d={patch['difficulty']:.4f}  P{patch['patch_row']}{patch['patch_col']}\n"
            f"Region: {manifest['spec']['region_size']} m / {grid[0]} x {grid[1]} patches\n"
            + ("Inverted basin: center low, four sides rise\n" if patch["inverted"] else "")
            + details
            + f"PhysX: {validation['total_rays']} rays passed\n"
            "Yellow dots / lines: candidate surface XY / yaw only.\n"
            "Preview markers do not validate robot placement.")
        save_status("running", selection=state, validation=validation)

    def on_family(model, item):
        state["family"] = model.get_item_value_model().as_int
        focus("region")

    def on_level(model, item):
        state["level"] = model.get_item_value_model().as_int
        focus()

    def on_patch(model, item):
        state["patch"] = model.get_item_value_model().as_int
        focus("patch")

    def on_samples(model):
        state["samples"] = model.as_bool
        focus()

    with window.frame:
        with ui.VStack(spacing=7):
            ui.Label(f"{len(families)} styles x {manifest['spec']['num_rows']} levels x {patches_per_region} patches = {manifest['patch_count']} patches", height=24)
            for title, initial, items, callback in (
                ("Terrain", state["family"], families, on_family),
                ("Level", state["level"], [f"L{i}" for i in range(10)], on_level),
                ("Patch", state["patch"], [f"P{i // grid[1]}{i % grid[1]} ({i})" for i in range(patches_per_region)], on_patch)):
                with ui.HStack(height=26):
                    ui.Label(title, width=60)
                    combo = ui.ComboBox(initial, *items)
                    combo.model.add_item_changed_fn(callback)
                    if title == "Terrain":
                        family_box = combo
                    elif title == "Level":
                        level_box = combo
            with ui.HStack(height=28, spacing=5):
                for title, mode in (("Overview", "overview"), (f"Region ({patches_per_region})", "region"), ("Patch", "patch"), ("Close", "close")):
                    ui.Button(title, clicked_fn=lambda m=mode: focus(m))
            with ui.HStack(height=24):
                checkbox = ui.CheckBox(width=24)
                checkbox.model.set_value(args.show_samples)
                checkbox.model.add_value_changed_fn(on_samples)
                ui.Label("Show candidate XY / yaw")
            label = ui.Label("", height=250, word_wrap=True)
            ui.Label("Alt + mouse: orbit / pan / zoom. Static preview; no training.", word_wrap=True, height=30)

    def capture(name):
        for _ in range(50):
            app.update()
        preview = output / name
        preview.unlink(missing_ok=True)
        capture_viewport_to_file(viewport, str(preview))
        deadline = time.monotonic() + 45.
        while not preview.exists():
            app.update()
            if time.monotonic() > deadline:
                raise RuntimeError("Viewport capture timed out; check the display GPU")

    if not args.skip_captures:
        focus("overview")
        capture("overview.png")
        for family in families:
            family_box.model.get_item_value_model().set_value(families.index(family))
            level_box.model.get_item_value_model().set_value(5)
            focus("region")
            capture(f"{family}_L5_region.png")
            if family != "flat":
                level_box.model.get_item_value_model().set_value(9)
                focus("patch")
                capture(f"{family}_L9_patch.png")
    family_box.model.get_item_value_model().set_value(families.index(args.family))
    level_box.model.get_item_value_model().set_value(args.level)
    if not args.skip_captures:
        checkbox.model.set_value(True)
        focus("region")
        capture("candidate_xy_yaw.png")
    checkbox.model.set_value(args.show_samples)
    focus(args.view)
    print(f"{manifest['version']} PATCH INSPECTOR READY", flush=True)
    last_status = 0.
    while app.is_running():
        app.update()
        if time.monotonic() - last_status > 3:
            save_status("running", selection=state, validation=validation)
            last_status = time.monotonic()
        time.sleep(.015)
    save_status("closed", validation=validation)


def guarded_main():
    try:
        main()
    except BaseException as exc:
        save_status("failed", error=repr(exc))
        raise


run_and_close(guarded_main, app)
