"""Interactive inspection of the actual P2.1 environment and height scanner."""
import argparse
import copy
from dataclasses import fields
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--family", default="stairs", choices=("flat", "stairs", "slope", "boxes", "rough"))
parser.add_argument("--level", type=int, default=5, choices=range(10))
parser.add_argument("--output", default=None)
parser.add_argument("--frames", type=int, default=0, help="Close after N renders; 0 keeps the inspector open")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
output = Path(args.output or ROOT / "logs/pgmt_perception_viewer" / datetime.now().strftime("%Y%m%d_%H%M%S"))
output.mkdir(parents=True, exist_ok=True)


def status(value, **details):
    path = output / "status.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(status=value, pid=os.getpid(), time=datetime.now().isoformat(),
                                   output=str(output.resolve()), **details), indent=2) + "\n")
    temp.replace(path)


status("launching")
from pmt_tasks.isaacsim6_legacy import install_tensor_import_bridge
install_tensor_import_bridge()
app = AppLauncher(args, multi_gpu=False, fast_shutdown=True).app


def main():
    import numpy as np
    import torch
    import omni.ui as ui
    from pxr import Gf, UsdGeom, UsdShade, Vt
    from isaaclab.envs import ManagerBasedRLEnv
    import isaaclab.sim as sim_utils
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.core.utils.viewports import set_camera_view
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    from pmt_tasks.isaacsim6_legacy import install_rigid_scene_schema_bridge
    from pmt_tasks.env_cfgs.pgmt.stage2 import PGMTPerceptionEnvCfg
    from pmt_tasks.pgmt_runtime import get_pgmt_state
    from pmt_tasks.pgmt_math import yaw_quat, rotate
    from pmt_tasks.pgmt_perception import terrain_observation_contract, split_terrain_observation

    install_rigid_scene_schema_bridge()
    cfg = PGMTPerceptionEnvCfg()
    # Show clean geometry, and verify the separate PMT-sign observation with noise off.
    cfg.pgmt_terrain_observation = terrain_observation_contract(augmentation_enabled=False)
    cfg.pmt_motion_paths = [str(ROOT / "assets/motions/pgmt_clipped_50hz_tcrs_v1/walk1_subject1_clip0.npz")]
    cfg.__post_init__()
    cfg.scene.num_envs = 5
    cfg.seed = 10412
    cfg.sim.device = cfg.commands.motion.storage_device = args.device
    cfg.sim.use_fabric = True
    cfg.sim.enable_scene_query_support = True
    cfg.curriculum.terrain_progress = None
    cfg.observations.proprio.enable_corruption = False
    cfg.observations.proprio_history.enable_corruption = False
    for name in ("physics_material", "add_joint_default_pos", "base_com", "push_robot"):
        setattr(cfg.events, name, None)
    old = cfg.scene.robot.spawn
    shared = {f.name: copy.deepcopy(getattr(old, f.name)) for f in fields(sim_utils.UsdFileCfg)
              if f.name not in ("func", "usd_path") and hasattr(old, f.name)}
    cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
        usd_path=str(ROOT / "logs/pgmt_p1_acceptance/native_robot/main.usd"), **shared)
    cfg.viewer.origin_type = "world"
    env = ManagerBasedRLEnv(cfg=cfg)
    SimulationManager.enable_fabric(True)
    try:
        env.reset()
        terrain, robot = env.scene.terrain, env.scene["robot"]
        cmd = env.command_manager.get_term("motion")
        state, sensor = get_pgmt_state(env), env.scene.sensors["height_scanner"]
        ids = torch.arange(5, device=env.device)
        families = cmd.terrain_families
        selection = dict(family=families.index(args.family), level=args.level, view="close", yaw=0., markers=True)
        stage = env.sim.stage
        viewport = get_active_viewport()
        pending = ["reset"]
        frames = 0

        def make_markers(name, color, radius):
            base = f"/World/PGMTScan/{name}"
            material = UsdShade.Material.Define(stage, base + "/Material")
            shader = UsdShade.Shader.Define(stage, base + "/Material/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            from pxr import Sdf
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color) * .35)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            sphere = UsdGeom.Sphere.Define(stage, base + "/Prototype")
            sphere.CreateRadiusAttr(radius)
            UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material)
            points = UsdGeom.PointInstancer.Define(stage, base + "/Points")
            points.CreatePrototypesRel().SetTargets([sphere.GetPath()])
            return points

        good = make_markers("Valid", (0., 1., .9), .016)
        bad = make_markers("Invalid", (1., .1, .1), .022)

        def update_markers():
            e = selection["family"]
            scan = state.terrain_scan()
            obs = split_terrain_observation(env.observation_manager.compute())
            expected = torch.where(scan["valid"], -scan["height"],
                                   (sensor.data.pos_w[:, 2] - 30.)[:, None, None]).clamp(-3., 3.)
            torch.testing.assert_close(obs["terrain_map"], expected, atol=1e-5, rtol=0)
            valid = scan["valid"][e].flatten()
            points = sensor.data.ray_hits_w[e].clone()
            points[:, 2] += .02  # Display lift only; measurement tensors are untouched.
            y, x = torch.meshgrid(torch.linspace(-1., 1., 21, device=env.device),
                                  torch.linspace(-1., 1., 21, device=env.device), indexing="ij")
            local = torch.stack((x, y, torch.zeros_like(x)), -1).reshape(441, 3)
            torso = robot.data.body_pos_w[e, state.anchor]
            invalid_points = rotate(yaw_quat(robot.data.body_quat_w[e, state.anchor]).expand(441, -1), local) + torso
            for marker, data in ((good, points[valid]), (bad, invalid_points[~valid])):
                array = data.cpu().numpy().astype(np.float32)
                marker.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(array))
                marker.GetProtoIndicesAttr().Set([0] * len(array))
                UsdGeom.Imageable(marker).GetVisibilityAttr().Set("inherited" if selection["markers"] else "invisible")
            map_value = scan["height"][e]
            lo, hi = float(map_value.flatten()[valid].min()), float(map_value.flatten()[valid].max())
            patch = int(cmd.placement_patch_ids[e])
            label.text = (f"{families[e]} | L{selection['level']} | patch {patch}\n"
                          f"21 x 21 | 2 x 2 m | valid {int(valid.sum())}/441\n"
                          f"Ground - torso: [{lo:.3f}, {hi:.3f}] m\n"
                          "Cyan: measured surface (+2 cm display lift)\n"
                          "Red: invalid cells shown at torso height\n"
                          "Frame 0 held; clean metric scan (augmentation OFF)")
            np.savez(output / "current_scan.npz", terrain_map=obs["terrain_map"][e].cpu().numpy(),
                     terrain_valid=obs["terrain_valid"][e].cpu().numpy(), clean_ground_minus_torso_m=map_value.cpu().numpy(),
                     hits_w=sensor.data.ray_hits_w[e].cpu().numpy(),
                     torso_pos=torso.cpu().numpy(), torso_quat=robot.data.body_quat_w[e, state.anchor].cpu().numpy())
            status("ready", family=families[e], level=selection["level"], patch=patch,
                   valid_points=int(valid.sum()), augmentation_enabled=False, map_units="m", headless=args.headless,
                   physics_steps=0, view=selection["view"], display_offset_m=.02)

        def focus():
            e = selection["family"]
            root = robot.data.body_pos_w[e, state.anchor].cpu().numpy()
            target = root + np.array([0., 0., -.45])
            mode = selection["view"]
            if mode == "top":
                delta = np.array([.01, -.01, 5.])
            elif mode == "tile":
                target = terrain.terrain_origins[selection["level"], e].cpu().numpy()
                delta = np.array([22., -28., 30.])
            else:
                delta = np.array([2.7, -3.3, 2.4])
            set_camera_view(eye=target + delta, target=target, camera_prim_path=str(viewport.camera_path))
            for index in range(5):
                UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/envs/env_{index}")).GetVisibilityAttr().Set(
                    "inherited" if index == e else "invisible")

        def request(key, value, reset=False):
            selection[key] = value
            pending.append("reset" if reset else "view")

        panel = ui.Window("PGMT P2.1 | Terrain map inspector", width=540, height=330)
        with panel.frame:
            with ui.VStack(spacing=6):
                ui.Label("Actual P2.1 scanner / held native robot reset")
                with ui.HStack(height=28):
                    for i, family in enumerate(families):
                        ui.Button(family, clicked_fn=lambda i=i: request("family", i))
                with ui.HStack(height=28):
                    for level in range(10):
                        ui.Button(f"L{level}", clicked_fn=lambda level=level: request("level", level, True))
                with ui.HStack(height=28):
                    ui.Button("Resample XY", clicked_fn=lambda: pending.append("reset"))
                    for angle in (0, 90, 180, -90):
                        ui.Button(f"yaw {angle}", clicked_fn=lambda angle=angle: request("yaw", angle, True))
                with ui.HStack(height=28):
                    for view in ("close", "top", "tile"):
                        ui.Button(view, clicked_fn=lambda view=view: request("view", view))
                    ui.Button("Toggle scan", clicked_fn=lambda: request("markers", not selection["markers"]))
                label = ui.Label("Loading scan...", word_wrap=True)
        while app.is_running() and (not args.frames or frames < args.frames):
            if pending:
                reset = "reset" in pending
                pending.clear()
                with torch.inference_mode():
                    if reset:
                        terrain.terrain_types[:] = ids
                        terrain.terrain_levels[:] = selection["level"]
                        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
                        angle = selection["yaw"] * np.pi / 180.
                        cmd.cfg.placement_yaw_range = (angle, angle)
                        env.reset()
                    update_markers()
                    focus()
            env.sim.render()  # Native render() does not advance physics.
            if frames == 90:
                capture_viewport_to_file(viewport, str((output / "viewport.png").resolve()))
            frames += 1
            time.sleep(.015)
        status("closed")
    except BaseException as error:
        status("failed", error=repr(error))
        raise
    finally:
        env.close()


from pmt_tasks.runtime_options import run_and_close
if __name__ == "__main__":
    run_and_close(main, app)
