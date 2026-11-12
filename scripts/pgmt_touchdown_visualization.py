"""Render-only touchdown inspection; preview never advances reward/event state."""
import numpy as np


class TouchdownDisplay:
    def __init__(self, stage, make_sphere, put_marker, controls):
        import omni.ui as ui
        from pxr import Gf, UsdGeom

        self.put_marker, self.controls = put_marker, controls
        self.last_events = [None, None]
        self.preview = None
        self.settlement = None
        self.hits = [[make_sphere(f"NinePointHit_{foot}_{i}", .009) for i in range(9)] for foot in range(2)]
        self.lifted = [[make_sphere(f"NinePointLift_{foot}_{i}", .009) for i in range(9)] for foot in range(2)]
        self.lines = UsdGeom.BasisCurves.Define(stage, "/World/PGMTContactDisplay/NinePointLines")
        self.lines.CreateTypeAttr("linear")
        self.lines.CreatePointsAttr([])
        self.lines.CreateCurveVertexCountsAttr([])
        self.lines.CreateWidthsAttr([.0015])
        self.lines.SetWidthsInterpolation("constant")
        self.lines.CreateDisplayColorAttr([Gf.Vec3f(.2, .8, .9)])
        self.labels, self.cells = [], []
        ui.Label("T1 SUPPORT QUALITY: 20 x 10 cm | toe offset +3.9 cm | 9 points / foot", height=24)
        ui.Label("Ground dots: green=valid, red=invalid. Cyan copies are raised 25 cm\n"
                 "to reveal points hidden by feet; lines end at queried ground heights.\n"
                 "Yellow=touchdown event only. T1 rewards continuous matched support.", height=55, word_wrap=True)
        with ui.HStack(height=28):
            ui.Button("Feet / overview", clicked_fn=lambda: controls.update(closeup=not controls["closeup"]))
            ui.Button("Pause at next touchdown", clicked_fn=lambda: controls.update(pause_td=True, paused=False))
        self.mode = ui.Label("", height=45, word_wrap=True)
        with ui.HStack(height=180):
            for side in ("LEFT", "RIGHT"):
                with ui.VStack(spacing=3):
                    ui.Label(f"{side} | toe (+X) at top; +Y at left", height=22)
                    rows = []
                    # Kernel order: X outer, Y inner. Show toe at top, left at left.
                    for row in ((8, 7, 6), (5, 4, 3), (2, 1, 0)):
                        with ui.HStack(height=23):
                            for index in row:
                                rows.append((index, ui.Label("--", style={"color": 0xFF55DD77})))
                    self.cells.append(rows)
                    self.labels.append(ui.Label("Waiting for first query", height=78, word_wrap=True))
        self.last_label = ui.Label("No touchdown yet. Last-event data stays frozen until reset.", height=65, word_wrap=True)

    def reset(self):
        self.last_events = [None, None]
        self.preview = None
        self.settlement = None

    def sample(self, state, robot, event=None):
        import torch
        from pmt_tasks.pgmt_touchdown import evaluate_touchdown

        # Separate geometry-only query, including swing feet. No state mutation,
        # reward call, synthetic force, or modification of the training gate.
        with torch.inference_mode():
            live = evaluate_touchdown(
                torch.ones(1, 2, device=state.env.device, dtype=torch.bool),
                robot.data.body_pos_w[:, state.foot_body_indices],
                robot.data.body_quat_w[:, state.foot_body_indices],
                robot.data.body_pos_w[:, state.anchor], robot.data.body_quat_w[:, state.anchor],
                state.env.scene.terrain.query, state.touchdown_config)
        self.preview = {k: v[0].detach().cpu().numpy() for k, v in live.items() if v.ndim >= 2 or k == "rate"}
        self.preview["event"][:] = False  # all-true input above means preview, never an event.
        self.preview["eligible"][:] = False
        self.preview["rate"] = np.asarray(0.)
        if event is not None:
            self.settlement = {key: event[key][0].item() for key in
                               ("rate", "pattern_valid", "pattern_matched", "reference_support_count")}
            self.preview["rate"] = np.asarray(self.settlement["rate"])
            self.preview["event"][:] = event["event"][0].detach().cpu().numpy()
            self.preview["eligible"][:] = event["eligible"][0].detach().cpu().numpy()
            for foot in range(2):
                if bool(event["eligible"][0, foot]):
                    # Show the exact settled geometry on every eligible step,
                    # including steady support with no touchdown event.
                    for key in self.preview:
                        if key != "rate":
                            self.preview[key][foot] = event[key][0, foot].detach().cpu().numpy()
                if bool(event["event"][0, foot]):
                    # An unmatched landing has no reward query. Keep its live
                    # geometry preview, explicitly marked as ineligible.
                    snapshot = {k: v[foot].copy() for k, v in self.preview.items() if k != "rate"}
                    snapshot.update(frame=int(event["reference_frame_ids"][0]),
                                    episode=int(event["episode_id"][0]), step=int(event["sample_step"][0]))
                    self.last_events[foot] = snapshot

    def render(self):
        from pxr import Vt
        if self.preview is None:
            return
        p = self.preview
        lines = []
        self.mode.text = "LIVE geometry preview | world height (m) | numbers 1..9 identify kernel points"
        if self.settlement is not None:
            s = self.settlement
            self.mode.text += f"\nT1 rate {s['rate']:.4f} | pattern match {s['pattern_matched']} | labels/forces valid {s['pattern_valid']}"
        for foot in range(2):
            xy, z = p["query_xy_w"][foot], p["height_w"][foot]
            valid, event = p["point_valid"][foot], bool(p["event"][foot])
            for i in range(9):
                # An invalid height has no physical surface position. Show only
                # a red raised marker at a diagnostic plane; do not invent a hit.
                known = bool(np.isfinite(xy[i]).all() and np.isfinite(z[i]))
                plane = float(z[i]) if np.isfinite(z[i]) else 0.
                bottom = np.array([*xy[i], plane + .012])
                top = np.array([*xy[i], plane + .25])
                color = (1., .8, .05) if event and valid[i] else (.1, .95, .3) if valid[i] else (1., .1, .2)
                self.put_marker(self.hits[foot][i], bottom if known else np.zeros(3), color, known)
                self.put_marker(self.lifted[foot][i], top if np.isfinite(top).all() else np.zeros(3),
                                (1., .8, .05) if event and valid[i] else (0., .85, 1.) if valid[i] else (1., .1, .2),
                                bool(np.isfinite(top).all()))
                if known:
                    lines.extend((bottom, top))
            for a, b in ((0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8),
                         (0, 3), (3, 6), (1, 4), (4, 7), (2, 5), (5, 8)):
                if valid[a] and valid[b]:
                    lines.extend((np.array([*xy[a], z[a] + .25]), np.array([*xy[b], z[b] + .25])))
            for index, cell in self.cells[foot]:
                cell.text = f"{index+1}: {z[index]:+.3f}" if np.isfinite(z[index]) else f"{index+1}: MISS"
                cell.style = {"color": 0xFF55DD77 if valid[index] else 0xFF5555FF}
            self.labels[foot].text = (
                f"Valid {int(valid.sum())}/9 | {'T1 ELIGIBLE' if p['eligible'][foot] else 'preview only'}"
                f"\nVariance {p['variance'][foot]:.6g} m^2 | Q {p['quality'][foot]:.4f}"
                f"\nMap {int(p['map_valid'][foot].sum())}/9 | scan {int(p['scan_valid'][foot].sum())}/9")
        self.lines.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(lines, dtype=np.float32).reshape(-1, 3)))
        self.lines.GetCurveVertexCountsAttr().Set([2] * (len(lines) // 2))
        messages = []
        for foot, s in enumerate(self.last_events):
            side = "LEFT" if foot == 0 else "RIGHT"
            messages.append(f"{side} last: none" if s is None else
                            f"{side} last: E{s['episode']} F{s['frame']} | valid {int(s['point_valid'].sum())}/9 | "
                            f"var {float(s['variance']):.6g} | Q {float(s['quality']):.4f} | eligible {bool(s['eligible'])}")
        self.last_label.text = "LAST LANDING EVENT (T1 settles continuously):\n" + "\n".join(messages)

    def record(self):
        def serial(value):
            if isinstance(value, dict):
                return {k: serial(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [serial(v) for v in value]
            return value.tolist() if isinstance(value, np.ndarray) else value
        return serial(dict(preview=self.preview, settlement=self.settlement, last_events=self.last_events))
