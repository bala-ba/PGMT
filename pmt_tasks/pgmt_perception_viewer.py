"""Display-only PGMT cameras and saved-observation glimpse overlays."""
import math


class PGMTPerceptionViewer:
    def __init__(self, env, policy, output):
        import omni.ui as ui
        from pxr import UsdGeom
        from omni.kit.viewport.utility import get_active_viewport
        self.env, self.policy, self.output = env, policy, output
        self.mode, self.index, self.enabled = "follow", 0, True
        self.viewport = get_active_viewport()
        self.viewport.set_active_camera(env.cfg.viewer.cam_prim_path)
        env.viewport_camera_controller.cfg.origin_type = "world"
        self.families = env.scene.terrain.manifest["spec"]["families"]
        self.stage = env.sim.stage
        self.points = UsdGeom.Points.Define(self.stage, "/World/PGMTPerception/Scan")
        self.centers = UsdGeom.Points.Define(self.stage, "/World/PGMTPerception/Centers")
        self.curves = UsdGeom.BasisCurves.Define(self.stage, "/World/PGMTPerception/Patches")
        self.curves.CreateTypeAttr("linear")
        self.curves.CreateWrapAttr("nonperiodic")
        self.curves.CreateCurveVertexCountsAttr([5] * 4)
        self.curves.CreateWidthsAttr([.018] * 20)
        self.curves.SetWidthsInterpolation("vertex")
        self.curves.CreateDisplayColorAttr([(1., .0, .6)])
        self.panel = ui.Window("PGMT P2 | Perception PPO", width=650, height=640)
        with self.panel.frame:
            with ui.VStack():
                with ui.HStack():
                    ui.Button("All robots / free camera", clicked_fn=self.global_view)
                    ui.Button("Follow selected", clicked_fn=self.follow)
                    ui.Button("Toggle perception", clicked_fn=self.toggle)
                with ui.HStack():
                    for family, name in enumerate(self.families):
                        ui.Button(name, clicked_fn=lambda f=family: self.choose(f))
                with ui.HStack():
                    self.family = ui.ComboBox(0, *self.families)
                    self.level = ui.ComboBox(0, *[f"L{i}" for i in range(10)])
                    ui.Button("View fixed tile", clicked_fn=self.tile_view)
                with ui.HStack():
                    ui.Label("Tile camera offset (m)")
                    self.offset = [ui.SimpleFloatModel(v) for v in (19., -29., 48.)]
                    for model in self.offset:
                        ui.FloatDrag(model=model, step=.5)
                self.label = ui.Label("Preparing saved-observation diagnostics", word_wrap=True)
        self.follow()

    def visibility(self, selected=None):
        from pxr import UsdGeom
        for i in range(self.env.num_envs):
            UsdGeom.Imageable(self.stage.GetPrimAtPath(f"/World/envs/env_{i}")).GetVisibilityAttr().Set(
                "inherited" if selected is None or selected == i else "invisible")

    def camera(self, target, offset):
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=[a + b for a, b in zip(target, offset)], target=target,
                        camera_prim_path=str(self.viewport.camera_path))

    def follow(self):
        self.mode = "follow"
        self.visibility(self.index)
        self.camera(self.env.scene["robot"].data.root_pos_w[self.index].cpu().tolist(), [3., 3., 2.])

    def choose(self, family):
        ids = (self.env.scene.terrain.terrain_types == family).nonzero().flatten()
        if len(ids):
            self.index = int(ids[0])
            self.follow()

    def global_view(self):
        self.mode = "global"
        self.visibility()
        terrain = self.env.scene.terrain
        lower, upper = terrain.tile_map.lower, terrain.tile_map.upper
        center = ((lower + upper) / 2).cpu().tolist()
        size = float((upper - lower).max())
        self.camera([*center, 0.], [.8 * size, -1.2 * size, 2. * size])

    def tile_view(self):
        family = self.family.model.get_item_value_model().as_int
        level = self.level.model.get_item_value_model().as_int
        offset = [v.as_float for v in self.offset]
        if not all(math.isfinite(v) for v in offset) or sum(v*v for v in offset) < 1e-8:
            return
        self.mode = "tile"
        self.visibility()
        self.camera(self.env.scene.terrain.terrain_origins[level, family].cpu().tolist(), offset)

    def toggle(self):
        from pxr import UsdGeom
        self.enabled = not self.enabled
        for item in (self.points, self.centers, self.curves):
            UsdGeom.Imageable(item).GetVisibilityAttr().Set("inherited" if self.enabled else "invisible")

    def update(self, obs):
        import numpy as np
        import torch
        from pxr import Vt
        from pmt_tasks.pgmt_runtime import get_pgmt_state
        from pmt_tasks.pgmt_math import yaw_quat, rotate
        if self.mode == "follow":
            self.camera(self.env.scene["robot"].data.root_pos_w[self.index].cpu().tolist(), [3., 3., 2.])
        if not self.enabled:
            return
        e = self.index
        with torch.no_grad():
            debug = self.policy.get_perception_debug(obs[e:e + 1])
        if debug.glimpse is None:
            return
        state = get_pgmt_state(self.env)
        robot = self.env.scene["robot"]
        torso = robot.data.body_pos_w[e, state.anchor]
        quat = yaw_quat(robot.data.body_quat_w[e, state.anchor])
        def world(xy):
            local = torch.cat((xy, torch.zeros_like(xy[..., :1])), -1)
            return rotate(quat.expand(*local.shape[:-1], 4), local) + torso
        y, x = torch.meshgrid(torch.linspace(-1., 1., 21, device=torso.device),
                              torch.linspace(-1., 1., 21, device=torso.device), indexing="ij")
        grid = world(torch.stack((x, y), -1).reshape(-1, 2))
        valid = obs["terrain_valid"][e].flatten() > .5
        sensor_z = self.env.scene.sensors["height_scanner"].data.pos_w[e, 2]
        grid[:, 2] = torch.where(valid, sensor_z - obs["terrain_map"][e].flatten() + .025, torso[2])
        self.points.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(grid.cpu().numpy().astype(np.float32)))
        self.points.CreateWidthsAttr([.025] * 441)
        colors = torch.where(valid[:, None], torso.new_tensor([0., 1., 1.]), torso.new_tensor([1., .1, .1]))
        self.points.CreateDisplayColorAttr(Vt.Vec3fArray.FromNumpy(colors.cpu().numpy()))
        centers_xy = debug.glimpse.center_xy_m[0]
        centers = world(centers_xy)
        height = self.env.scene.terrain.height(centers[:, :2])
        centers[:, 2] = torch.where(torch.isfinite(height), height + .08, torso[2])
        self.centers.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(centers.cpu().numpy().astype(np.float32)))
        self.centers.CreateWidthsAttr([.09] * 4)
        self.centers.CreateDisplayColorAttr([(1., .6, .0)])
        corners = centers_xy[:, None] + torso.new_tensor([[-.2,-.2],[.2,-.2],[.2,.2],[-.2,.2],[-.2,-.2]])
        border = world(corners.reshape(-1, 2))
        z = self.env.scene.terrain.height(border[:, :2])
        border[:, 2] = torch.where(torch.isfinite(z), z + .05, torso[2])
        self.curves.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(border.cpu().numpy().astype(np.float32)))
        cmd = self.env.command_manager.get_term("motion")
        self.label.text = (f"env {e} | {self.families[int(self.env.scene.terrain.terrain_types[e])]} | "
            f"L{int(self.env.scene.terrain.terrain_levels[e])} | frame {int(cmd.frame_ids[e])}\n"
            f"Saved augmented map: {int(valid.sum())}/441 valid; cyan valid / red invalid\n"
            f"Patch valid: {debug.glimpse.token_valid[0].tolist()}\n"
            f"Coverage: {[round(v,2) for v in debug.glimpse.coverage[0].tolist()]}\n"
            f"Patch 0 normalized heights (-1 invalid):\n{debug.glimpse.patch_height[0,0].cpu().numpy().round(2)}\n"
            f"Patch 0 mask:\n{debug.glimpse.point_valid[0,0].int().cpu().numpy()}\n"
            f"Terrain attention per head: {[round(v,3) for v in debug.terrain_attention_mass[0,:,0].tolist()]}\n"
            "Orange: centers; magenta: 0.4 m patch spans; display height lifted")

    def capture(self, name):
        from omni.kit.viewport.utility import capture_viewport_to_file
        capture_viewport_to_file(self.viewport, str(self.output / name))
