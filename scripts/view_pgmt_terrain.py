"""Open the generated terrain in a native Isaac Sim window with inspection UI.

Use the installed Isaac Sim 6 environment for the current RTX driver:
  /home/lzp/miniconda3/envs/isaacsim/bin/python scripts/view_pgmt_terrain.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--asset", default="assets/terrain/pgmt_stage2_v5")
parser.add_argument("--output", default="logs/pgmt_terrain_preview_v5")
parser.add_argument("--family", choices=["flat", "slopes", "stairs", "boxes", "random_rough", "up_stairs", "up_slope"], default="up_stairs")
parser.add_argument("--level", type=int, choices=range(10), default=5)
parser.add_argument("--view", choices=["overview", "tile", "close"], default="tile")
parser.add_argument("--gpu", type=int, default=None, help="Default: GPU connected to the desktop")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--validate-only", action="store_true")
args = parser.parse_args()
if args.gpu is None:
    devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,display_active,pci.bus_id", "--format=csv,noheader"],
                                      text=True, timeout=10)
    rows = [[item.strip() for item in line.split(",")] for line in devices.splitlines()]
    connected = [int(row[0]) for row in rows if row[1] == "Enabled"]
    # display_active becomes Disabled while the monitor sleeps. The physical
    # DRM connector remains connected and identifies the correct PCI device.
    if not connected:
        buses = set()
        for status in Path("/sys/class/drm").glob("card*-*/status"):
            if status.read_text().strip() == "connected":
                card = status.parent.name.split("-",1)[0]
                bus = (status.parent.parent/card/"device").resolve().name
                buses.add(":".join(bus.lower().split(":")[-2:]))
        connected = [int(row[0]) for row in rows if ":".join(row[2].lower().split(":")[-2:]) in buses]
    args.gpu = connected[0] if connected and not args.headless else 0
asset = Path(args.asset).resolve()
output = Path(args.output).resolve()
output.mkdir(parents=True,exist_ok=True)
manifest = json.loads((asset/"manifest.json").read_text())
usd = asset/manifest["asset"]
if hashlib.sha256(usd.read_bytes()).hexdigest() != manifest["asset_sha256"]:
    raise ValueError("USD asset checksum differs from manifest")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp

app = SimulationApp(dict(headless=args.headless, width=1600, height=1000,
    window_width=1600, window_height=1000, multi_gpu=False, active_gpu=args.gpu,
    physics_gpu=args.gpu, limit_cpu_threads=8,
    extra_args=["--/app/useFabricSceneDelegate=false",
                "--/rtx/hydra/readTransformsFromFabricInRenderDelegate=false"]))

import numpy as np
import carb
import omni.usd
import omni.ui as ui
import omni.timeline
import omni.physx
from pxr import UsdGeom, UsdLux
from isaacsim.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file


def save_status(state, **values):
    path = output/"status.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(state=state,pid=os.getpid(),display=os.getenv("DISPLAY"),
        asset=str(usd),time=time.time(),**values),indent=2)+"\n")
    temp.replace(path)


def main():
    save_status("loading")
    if not omni.usd.get_context().open_stage(str(usd)):
        raise RuntimeError("Could not open generated terrain USD")
    for _ in range(60):
        app.update()
    stage = omni.usd.get_context().get_stage()
    carb.settings.get_settings().set_int("/persistent/app/viewport/displayOptions", 0)
    UsdGeom.Camera(stage.GetPrimAtPath("/World/InspectionCamera")).GetFocalLengthAttr().Set(25.)
    UsdLux.DomeLight(stage.GetPrimAtPath("/World/Sky")).GetIntensityAttr().Set(180.)
    UsdLux.DistantLight(stage.GetPrimAtPath("/World/Sun")).GetIntensityAttr().Set(900.)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(40):
        app.update()
    query = omni.physx.get_physx_scene_query_interface()
    errors = []
    for tile in manifest["tiles"]:
        bounds = np.asarray(tile["geometry"]["bounds"])
        ray_top = float(bounds[1,2]+5.)
        ray_length = float(bounds[1,2]-bounds[0,2]+10.)
        for x,y,z in tile["world_probes"]:
            hit = query.raycast_closest((x,y,ray_top),(0.,0.,-1.),ray_length)
            if not hit.get("hit"):
                raise RuntimeError(f"PhysX ray missed {tile['name']} at {(x,y)}")
            err = abs(float(hit["position"][2])-z)
            if err > .002:
                raise RuntimeError(f"PhysX / mesh mismatch on {tile['name']}: {err} m")
            errors.append(err)
    timeline.pause()
    validation = dict(passed=True,tiles=manifest["tile_count"],rays=len(errors),
        max_height_error_m=max(errors),renderer="Isaac Sim 6",physics="PhysX triangle-mesh collision",
        asset_sha256=manifest["asset_sha256"])
    (output/"collision_validation.json").write_text(json.dumps(validation,indent=2)+"\n")
    print("PHYSX VALIDATED:",json.dumps(validation),flush=True)
    if args.validate_only:
        save_status("validated",validation=validation)
        return

    viewport = get_active_viewport()
    viewport.camera_path = "/World/InspectionCamera"
    families = manifest["spec"]["families"]
    tiles = {(t["family_index"],t["level"]):t for t in manifest["tiles"]}
    initial_family = args.family if args.family in families else families[0]
    state = dict(family=families.index(initial_family),level=args.level,mode=args.view)
    window = ui.Window("PGMT Terrain Inspector",width=440,height=390)
    label = None

    def focus(mode="tile"):
        state["mode"] = mode
        tile = tiles[state["family"],state["level"]]
        for t in manifest["tiles"]:
            imageable = UsdGeom.Imageable(stage.GetPrimAtPath(t["prim_path"]))
            if mode == "overview" or t is tile:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
        border = UsdGeom.Imageable(stage.GetPrimAtPath("/World/TerrainBorder"))
        if mode == "overview":
            border.MakeVisible()
            extent = np.asarray(manifest["spec"]["size"])*[manifest["spec"]["num_rows"],len(families)]
            scale = np.linalg.norm(extent)/np.linalg.norm([240.,120.])
            eye,target = np.array([220.,-275.,275.])*scale,np.zeros(3)
        elif tile.get("inverted"):
            border.MakeInvisible()
            bounds = np.asarray(tile["geometry"]["bounds"])
            if mode == "close":
                target = np.asarray(tile["origin"])+[0.,-.6,.3]
                pit_depth = bounds[1,2]-tile["origin"][2]
                eye = target+[3.,-4.,max(5.,.6*pit_depth+2.)]
            else:
                target = np.asarray(tile["center"])+[0.,0.,tile["origin"][2]*.5]
                radius = np.linalg.norm((bounds[1]-bounds[0])/2)
                direction = np.asarray([1.,-1.,1.6])
                eye = target+direction/np.linalg.norm(direction)*max(60.,4.5*radius)
        else:
            border.MakeInvisible()
            target = np.array(tile["origin"],dtype=float)
            if mode == "close":
                target += np.array([0.,-2.,-.5])
                if tile["family"] == "random_rough":
                    target[2] = 0.
                    eye = target + np.array([2.5,-3.5,2.])
                else:
                    eye = target + np.array([6.,-8.,6.])
            else:
                target[2] *= .5
                eye = target + np.array([22.,-28.,25.])
        set_camera_view(eye,target,camera_prim_path="/World/InspectionCamera")
        p = tile["parameters"]
        extra = (f"{tile['shape']}\nCenter low; ascend outwards on four sides\n"
                 if tile.get("inverted") else "")
        label.text = (f"{tile['family']}   L{tile['level']}   d={tile['difficulty']:.4f}\n"
            + extra +
            f"Slope angle: {p['slope_angle_deg']:.2f} deg\n"
            f"Stair height / depth: {p['stair_height_m']:.3f} / {p['stair_depth_m']:.3f} m\n"
            f"Box ceiling: {p['box_height_max_m']:.3f} m\n"
            f"Rough input noise half-range: {p['rough_amplitude_m']:.3f} m\n"
            f"PhysX: {len(errors)} rays passed, max error {max(errors):.6f} m")
        save_status("running",selection=state,validation=validation)

    def on_family(model,item):
        state["family"] = model.get_item_value_model().as_int
        focus("tile")

    def on_level(model,item):
        state["level"] = model.get_item_value_model().as_int
        focus("tile")

    with window.frame:
        with ui.VStack(spacing=8):
            ui.Label(f"{len(families)} terrain styles x {manifest['spec']['num_rows']} difficulty levels",height=24)
            with ui.HStack(height=28):
                ui.Label("Terrain",width=75)
                family_box = ui.ComboBox(state["family"],*families)
                family_box.model.add_item_changed_fn(on_family)
            with ui.HStack(height=28):
                ui.Label("Level",width=75)
                level_box = ui.ComboBox(state["level"],*[f"L{i}" for i in range(10)])
                level_box.model.add_item_changed_fn(on_level)
            with ui.HStack(height=30,spacing=6):
                ui.Button("Overview",clicked_fn=lambda:focus("overview"))
                ui.Button("Whole tile",clicked_fn=lambda:focus("tile"))
                ui.Button("Close up",clicked_fn=lambda:focus("close"))
            label = ui.Label("",word_wrap=True,height=180)
            ui.Label("Viewport: Alt + mouse to orbit / pan / zoom. No training is running.",word_wrap=True)
    def capture(name):
        for _ in range(50):
            app.update()
        preview = output/name
        preview.unlink(missing_ok=True)
        capture_viewport_to_file(viewport,str(preview))
        deadline = time.monotonic()+60
        while not preview.exists():
            app.update()
            if time.monotonic() > deadline:
                raise RuntimeError("Viewport did not render; verify --gpu matches the display-connected GPU")

    focus("overview")
    capture("overview.png")
    for family in ("boxes", "random_rough", "up_stairs", "up_slope"):
        if family not in families:
            continue
        family_box.model.get_item_value_model().set_value(families.index(family))
        level_box.model.get_item_value_model().set_value(9 if family in ("boxes","random_rough") else 5)
        if family in ("up_stairs", "up_slope"):
            focus("tile")
            capture(f"{family}_tile.png")
        focus("close")
        capture(f"{family}_close.png")
    family_box.model.get_item_value_model().set_value(families.index(initial_family))
    level_box.model.get_item_value_model().set_value(args.level)
    focus(args.view)
    print("PGMT TERRAIN INSPECTOR READY",flush=True)
    last_status = 0.
    while app.is_running():
        app.update()
        if time.monotonic()-last_status > 3:
            save_status("running",selection=state,validation=validation)
            last_status = time.monotonic()
        time.sleep(.015)
    save_status("closed",validation=validation)


try:
    main()
except BaseException as exc:
    save_status("failed",error=repr(exc))
    raise
finally:
    app.close()
