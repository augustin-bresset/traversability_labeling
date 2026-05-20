"""Interactive 3-D traversability viewer built on Open3D GUI.

Displays per scan:
  - Point cloud (coloured by traversability, intensity, or height)
  - Robot footprint wireframe at the sensor origin
  - Past/future trajectory in the scan's local frame

Keyboard shortcuts:
    -> / L   next scan
    <- / H   previous scan
    R       reset camera
    F       top-down (bird's-eye) view
    T       cycle colour mode  (traversability / intensity / height)
    J       toggle trajectory
    K       toggle robot footprint
    V       toggle forward-only accumulation
    N       toggle forward-mask preview
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.cm as _cm
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from src.datasets.rellis import Rellis3DSequence
from src.traversability.labeler import TraversabilityLabeler


PANEL_W = 270
POINT_SIZE = 2.0
DISPLAY_MODES = ["Traversability", "Intensity", "Height"]
TRAIL_MAX_PTS = 500_000   # cap trail point count to keep GPU load bounded


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lineset(pts: np.ndarray, edges: list, color: list) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    ls.lines  = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(edges), 1)))
    return ls


def _color_tile(rgb_float: list, size: int = 12) -> o3d.geometry.Image:
    tile = np.full((size, size, 3), [int(c * 255) for c in rgb_float], dtype=np.uint8)
    return o3d.geometry.Image(tile)


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

class TraversabilityViewer:
    """
    Interactive viewer for a single RELLIS-3D sequence.

    Labels are loaded from pre-computed .trav files if label_dir is provided,
    otherwise computed on-the-fly and cached for instant re-navigation.
    """

    # Colours (float RGB 0–1)
    C_TRAV        = [52/255, 211/255, 153/255]  # green  – confirmed traversable
    C_GROUND      = [0.50, 0.50, 0.50]          # mid-gray – ground, not confirmed
    C_OTHER       = [0.25, 0.25, 0.25]          # dark gray – above/below ground
    C_TRAJ_PAST   = [0.20, 0.60, 1.00]          # blue
    C_TRAJ_FUTURE = [1.00, 0.60, 0.10]          # orange
    C_ROBOT       = [1.00, 0.90, 0.00]          # yellow
    C_TRAIL       = [0.95, 0.25, 0.70]          # magenta – persistent traversable trail

    def __init__(
        self,
        seq: Rellis3DSequence,
        poses: Optional[List[np.ndarray]],
        labeler: TraversabilityLabeler,
        label_dir: Optional[Path] = None,
        robot_shape: str = "square",
        robot_size: float = 1.0,
        start_idx: int = 0,
    ):
        self.seq = seq
        self.poses = poses
        self.labeler = labeler
        self.label_dir = Path(label_dir) if label_dir is not None else None
        self.robot_shape = robot_shape
        self.robot_size = robot_size
        self.current_idx = start_idx

        self._label_cache: dict[int, np.ndarray] = {}
        self._scan_cache:  dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        self._show_trav = True
        self._show_ground = True
        self._show_other = True
        self._show_traj_past = True
        self._show_traj_future = True
        self._show_robot = True
        self._show_trail_pts = True
        self._display_mode = 0  # index into DISPLAY_MODES
        self._n_past = 0        # number of past scans to accumulate
        self._n_future = 0      # number of future scans to accumulate
        self._accum_step = 1    # stride between accumulated scans
        self._forward_accum = labeler.forward_accum
        self._fwd_only = False

        # Persistent traversable-point trail (world frame, Nx3)
        self._trail_active = False
        self._trav_trail_world: Optional[np.ndarray] = None
        self._trail_visited: set = set()  # scan indices already added to trail

        # Guard against recursive slider callbacks when setting int_value
        self._updating = False

        # GUI widget refs
        self._window = None
        self._scene: Optional[gui.SceneWidget] = None
        self._lbl_frame: Optional[gui.Label] = None
        self._lbl_npts: Optional[gui.Label] = None
        self._lbl_stats: Optional[gui.Label] = None
        self._lbl_mode: Optional[gui.Label] = None
        self._lbl_accum: Optional[gui.Label] = None
        self._lbl_trail: Optional[gui.Label] = None
        self._cb_trav:        Optional[gui.Checkbox] = None
        self._cb_ground:      Optional[gui.Checkbox] = None
        self._cb_other:       Optional[gui.Checkbox] = None
        self._cb_traj_past:   Optional[gui.Checkbox] = None
        self._cb_traj_future: Optional[gui.Checkbox] = None
        self._cb_robot:       Optional[gui.Checkbox] = None
        self._cb_trail_pts:   Optional[gui.Checkbox] = None
        self._cb_trail: Optional[gui.Checkbox] = None
        self._cb_fwd_accum: Optional[gui.Checkbox] = None
        self._cb_fwd_only:  Optional[gui.Checkbox] = None
        self._idx_slider: Optional[gui.Slider] = None
        self._num_edit: Optional[gui.NumberEdit] = None
        self._past_slider: Optional[gui.Slider] = None
        self._future_slider: Optional[gui.Slider] = None
        self._step_slider: Optional[gui.Slider] = None
        self._mat = None
        self._line_mat = None
        self._camera_initialized = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def launch(
        seq: Rellis3DSequence,
        poses: Optional[List[np.ndarray]],
        labeler: TraversabilityLabeler,
        label_dir: Optional[Path] = None,
        robot_shape: str = "square",
        robot_size: float = 1.0,
        start_idx: int = 0,
    ) -> None:
        app = gui.Application.instance
        app.initialize()
        v = TraversabilityViewer(
            seq, poses, labeler, label_dir, robot_shape, robot_size, start_idx
        )
        v._build_window()
        app.run()

    # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------

    def _build_window(self) -> None:
        app = gui.Application.instance
        w = app.create_window(f"Traversability - {self.seq.name}", 1600, 950)
        self._window = w
        em = w.theme.font_size

        # Materials
        self._mat = rendering.MaterialRecord()
        self._mat.shader = "defaultUnlit"
        self._mat.point_size = POINT_SIZE

        # "unlitLine" is the correct Filament shader for LineSet geometries.
        # "defaultUnlit" requires UV0 which LineSets don't have -> crash.
        self._line_mat = rendering.MaterialRecord()
        self._line_mat.shader = "unlitLine"
        self._line_mat.line_width = 2.5

        # 3-D scene
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.scene.set_background([0.07, 0.07, 0.07, 1.0])
        self._scene.set_on_key(self._on_key)

        # ---- Left panel ----
        panel = gui.Vert(int(0.4 * em), gui.Margins(int(0.6 * em)))

        # Navigation
        panel.add_child(self._sec("Navigation", em))
        self._lbl_frame = gui.Label("-")
        self._lbl_frame.text_color = gui.Color(0.85, 0.85, 0.85)
        panel.add_child(self._lbl_frame)
        self._lbl_npts = gui.Label("Points: -")
        self._lbl_npts.text_color = gui.Color(0.6, 0.6, 0.6)
        panel.add_child(self._lbl_npts)

        nav = gui.Horiz(int(0.3 * em))
        b_prev = gui.Button("< Prev  [H]"); b_prev.set_on_clicked(self._on_prev)
        b_next = gui.Button("Next >  [L]"); b_next.set_on_clicked(self._on_next)
        nav.add_stretch(); nav.add_child(b_prev); nav.add_child(b_next); nav.add_stretch()
        panel.add_child(nav)

        # Full-range slider for fast scrubbing
        self._idx_slider = gui.Slider(gui.Slider.INT)
        self._idx_slider.set_limits(0, len(self.seq) - 1)
        self._idx_slider.int_value = self.current_idx
        self._idx_slider.set_on_value_changed(self._on_slider_changed)
        panel.add_child(self._idx_slider)

        # Jump-to-index row
        jump_row = gui.Horiz(int(0.3 * em))
        lbl_go = gui.Label("Go to #")
        lbl_go.text_color = gui.Color(0.6, 0.6, 0.6)
        self._num_edit = gui.NumberEdit(gui.NumberEdit.INT)
        self._num_edit.int_value = self.current_idx
        b_go = gui.Button("Go")
        b_go.set_on_clicked(self._on_jump)
        jump_row.add_child(lbl_go)
        jump_row.add_child(self._num_edit)
        jump_row.add_child(b_go)
        panel.add_child(jump_row)

        cam = gui.Horiz(int(0.3 * em))
        b_top   = gui.Button("Top  [F]");    b_top.set_on_clicked(self._look_top)
        b_lidar = gui.Button("LiDAR  [E]");  b_lidar.set_on_clicked(self._look_lidar)
        b_reset = gui.Button("Reset  [R]");  b_reset.set_on_clicked(self._reset_camera)
        cam.add_stretch()
        cam.add_child(b_top)
        cam.add_child(b_lidar)
        cam.add_child(b_reset)
        cam.add_stretch()
        panel.add_child(cam)
        panel.add_child(gui.Label(""))

        # Colour mode
        panel.add_child(self._sec("Colour mode  [T]", em))
        self._lbl_mode = gui.Label(DISPLAY_MODES[self._display_mode])
        self._lbl_mode.text_color = gui.Color(0.7, 0.9, 1.0)
        panel.add_child(self._lbl_mode)
        b_mode = gui.Button("Cycle  [T]"); b_mode.set_on_clicked(self._on_cycle_mode)
        panel.add_child(b_mode)
        panel.add_child(gui.Label(""))

        # Legend — each entry is a visibility checkbox
        panel.add_child(self._sec("Legend", em))
        _sel_row = gui.Horiz(int(0.3 * em))
        b_all  = gui.Button("Show all");  b_all.set_on_clicked(lambda: self._set_all_legend(True))
        b_none = gui.Button("Hide all"); b_none.set_on_clicked(lambda: self._set_all_legend(False))
        _sel_row.add_child(b_all); _sel_row.add_child(b_none)
        panel.add_child(_sel_row)
        _legend_items = [
            ("Traversable",         self.C_TRAV,        "_show_trav",        "_cb_trav"),
            ("Ground (unlabelled)", self.C_GROUND,       "_show_ground",      "_cb_ground"),
            ("Other points",        self.C_OTHER,        "_show_other",       "_cb_other"),
            ("Trajectory - past",   self.C_TRAJ_PAST,    "_show_traj_past",   "_cb_traj_past"),
            ("Trajectory - future", self.C_TRAJ_FUTURE,  "_show_traj_future", "_cb_traj_future"),
            ("Robot footprint",     self.C_ROBOT,        "_show_robot",       "_cb_robot"),
            ("Traversable trail",   self.C_TRAIL,        "_show_trail_pts",   "_cb_trail_pts"),
        ]
        for label_text, rgb, state_attr, cb_attr in _legend_items:
            row = gui.Horiz(int(0.3 * em))
            row.add_child(gui.ImageWidget(_color_tile(rgb)))
            cb = gui.Checkbox(label_text)
            cb.checked = getattr(self, state_attr)
            cb.set_on_checked(lambda v, a=state_attr: self._on_legend_toggle(a, v))
            setattr(self, cb_attr, cb)
            row.add_child(cb)
            panel.add_child(row)
        panel.add_child(gui.Label(""))

        # Overlays
        panel.add_child(self._sec("Overlays", em))
        self._cb_fwd_accum = gui.Checkbox("Forward accum  [V]")
        self._cb_fwd_accum.checked = self._forward_accum
        self._cb_fwd_accum.set_on_checked(self._on_fwd_accum_toggled)
        panel.add_child(self._cb_fwd_accum)

        self._cb_fwd_only = gui.Checkbox("Forward mask preview  [N]")
        self._cb_fwd_only.checked = self._fwd_only
        self._cb_fwd_only.set_on_checked(lambda v: self._set_fwd_only(v))
        panel.add_child(self._cb_fwd_only)
        panel.add_child(gui.Label(""))

        # Accumulated scans
        panel.add_child(self._sec("Accumulate scans", em))
        self._lbl_accum = gui.Label("Past: 0  Future: 0  (current only)")
        self._lbl_accum.text_color = gui.Color(0.7, 0.7, 0.7)
        panel.add_child(self._lbl_accum)

        lbl_past = gui.Label("Past scans N")
        lbl_past.text_color = gui.Color(0.55, 0.55, 0.55)
        panel.add_child(lbl_past)
        self._past_slider = gui.Slider(gui.Slider.INT)
        self._past_slider.set_limits(0, 50)
        self._past_slider.int_value = 0
        self._past_slider.set_on_value_changed(self._on_past_changed)
        panel.add_child(self._past_slider)

        lbl_future = gui.Label("Future scans M")
        lbl_future.text_color = gui.Color(0.55, 0.55, 0.55)
        panel.add_child(lbl_future)
        self._future_slider = gui.Slider(gui.Slider.INT)
        self._future_slider.set_limits(0, 50)
        self._future_slider.int_value = 0
        self._future_slider.set_on_value_changed(self._on_future_changed)
        panel.add_child(self._future_slider)

        lbl_step = gui.Label("Step (every K-th scan)")
        lbl_step.text_color = gui.Color(0.55, 0.55, 0.55)
        panel.add_child(lbl_step)
        self._step_slider = gui.Slider(gui.Slider.INT)
        self._step_slider.set_limits(1, 20)
        self._step_slider.int_value = 1
        self._step_slider.set_on_value_changed(self._on_step_changed)
        panel.add_child(self._step_slider)
        panel.add_child(gui.Label(""))

        # Traversable trail
        panel.add_child(self._sec("Traversable trail  [M]", em))
        self._lbl_trail = gui.Label("Off - 0 pts")
        self._lbl_trail.text_color = gui.Color(0.7, 0.7, 0.7)
        panel.add_child(self._lbl_trail)
        self._cb_trail = gui.Checkbox("Record while navigating")
        self._cb_trail.checked = self._trail_active
        self._cb_trail.set_on_checked(self._on_trail_toggled)
        panel.add_child(self._cb_trail)
        b_clear = gui.Button("Clear trail")
        b_clear.set_on_clicked(self._on_trail_clear)
        panel.add_child(b_clear)
        panel.add_child(gui.Label(""))

        # Traversability stats
        panel.add_child(self._sec("Stats", em))
        self._lbl_stats = gui.Label("-")
        self._lbl_stats.text_color = gui.Color(0.75, 0.75, 0.75)
        panel.add_child(self._lbl_stats)
        panel.add_child(gui.Label(""))

        # Robot info
        panel.add_child(self._sec("Robot", em))
        has_poses = self.poses is not None
        info = gui.Label(
            f"Shape : {self.robot_shape}\n"
            f"Size  : {self.robot_size} m\n"
            f"Poses : {'yes' if has_poses else 'none (no labels)'}\n"
            f"Window: ±{self.labeler.trajectory_window}"
        )
        info.text_color = gui.Color(0.65, 0.65, 0.65)
        panel.add_child(info)

        # Layout
        w.add_child(self._scene)
        w.add_child(panel)
        self._panel = panel
        w.set_on_layout(self._on_layout)

        self._refresh()

    @staticmethod
    def _sec(text: str, em: float) -> gui.Label:
        lbl = gui.Label(text.upper())
        lbl.text_color = gui.Color(0.4, 0.75, 1.0)
        return lbl

    def _on_layout(self, _ctx) -> None:
        r = self._window.content_rect
        self._panel.frame = gui.Rect(r.x, r.y, PANEL_W, r.height)
        self._scene.frame = gui.Rect(r.x + PANEL_W, r.y, r.width - PANEL_W, r.height)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    _KEY_DOWN = getattr(gui.KeyEvent, "DOWN", None) or gui.KeyEvent.Type.DOWN

    def _on_key(self, event) -> int:
        if event.type != self._KEY_DOWN:
            return gui.Widget.EventCallbackResult.IGNORED
        k = event.key
        H = gui.Widget.EventCallbackResult.HANDLED
        if k in (gui.KeyName.RIGHT, ord("l"), ord("L")): self._on_next();          return H
        if k in (gui.KeyName.LEFT,  ord("h"), ord("H")): self._on_prev();          return H
        if k in (ord("r"), ord("R")): self._reset_camera();                         return H
        if k in (ord("f"), ord("F")): self._look_top();                             return H
        if k in (ord("e"), ord("E")): self._look_lidar();                           return H
        if k in (ord("t"), ord("T")): self._on_cycle_mode();                        return H
        if k in (ord("j"), ord("J")):
            new_val = not (self._show_traj_past and self._show_traj_future)
            self._show_traj_past = new_val
            self._show_traj_future = new_val
            if self._cb_traj_past:   self._cb_traj_past.checked   = new_val
            if self._cb_traj_future: self._cb_traj_future.checked = new_val
            self._refresh_overlays()
            return H
        if k in (ord("k"), ord("K")):
            self._show_robot = not self._show_robot
            if self._cb_robot: self._cb_robot.checked = self._show_robot
            self._refresh_overlays()
            return H
        if k in (ord("m"), ord("M")):
            self._trail_active = not self._trail_active
            self._cb_trail.checked = self._trail_active
            self._update_trail_label()
            return H
        if k in (ord("v"), ord("V")):
            self._on_fwd_accum_toggled(not self._forward_accum)
            self._cb_fwd_accum.checked = self._forward_accum
            return H
        if k in (ord("n"), ord("N")):
            self._set_fwd_only(not self._fwd_only)
            self._cb_fwd_only.checked = self._fwd_only
            return H
        return gui.Widget.EventCallbackResult.IGNORED

    def _on_next(self) -> None:
        self.current_idx = (self.current_idx + 1) % len(self.seq)
        self._refresh()

    def _on_prev(self) -> None:
        self.current_idx = (self.current_idx - 1) % len(self.seq)
        self._refresh()

    def _on_slider_changed(self, val: float) -> None:
        if self._updating:
            return
        new_idx = max(0, min(int(val), len(self.seq) - 1))
        if new_idx != self.current_idx:
            self.current_idx = new_idx
            self._refresh()

    def _on_jump(self) -> None:
        idx = max(0, min(int(self._num_edit.int_value), len(self.seq) - 1))
        self.current_idx = idx
        self._refresh()

    def _on_past_changed(self, val: float) -> None:
        self._n_past = int(val)
        self._update_accum_label()
        self._refresh()

    def _on_future_changed(self, val: float) -> None:
        self._n_future = int(val)
        self._update_accum_label()
        self._refresh()

    def _on_step_changed(self, val: float) -> None:
        self._accum_step = int(val)
        self._update_accum_label()
        self._refresh()

    def _update_accum_label(self) -> None:
        n, m, s = self._n_past, self._n_future, self._accum_step
        if n == 0 and m == 0:
            self._lbl_accum.text = "Past: 0  Future: 0  (current only)"
        else:
            parts = []
            if n > 0:
                parts.append(f"Past: {n}  (~{n*s} scans)")
            if m > 0:
                parts.append(f"Future: {m}  (~{m*s} scans)")
            self._lbl_accum.text = "  ".join(parts) + f"  step={s}"

    def _set_fwd_only(self, active: bool) -> None:
        self._fwd_only = active
        self._refresh()

    def _on_fwd_accum_toggled(self, active: bool) -> None:
        self._forward_accum = active
        self.labeler.forward_accum = active
        # Invalidate label cache - accumulated clouds change with this setting.
        self._label_cache.clear()
        self._refresh()

    def _on_trail_toggled(self, active: bool) -> None:
        self._trail_active = active
        self._update_trail_label()

    def _on_trail_clear(self) -> None:
        self._trav_trail_world = None
        self._trail_visited.clear()
        self._update_trail_label()
        self._scene.scene.remove_geometry("trav_trail")

    def _update_trail_label(self) -> None:
        n = len(self._trav_trail_world) if self._trav_trail_world is not None else 0
        state = "On" if self._trail_active else "Off"
        self._lbl_trail.text = f"{state} - {n:,} pts"

    def _on_cycle_mode(self) -> None:
        self._display_mode = (self._display_mode + 1) % len(DISPLAY_MODES)
        self._lbl_mode.text = DISPLAY_MODES[self._display_mode]
        self._refresh()

    def _on_legend_toggle(self, attr: str, value: bool) -> None:
        setattr(self, attr, value)
        if attr in ("_show_trav", "_show_ground", "_show_other"):
            self._refresh()
        elif attr in ("_show_traj_past", "_show_traj_future", "_show_robot"):
            self._refresh_overlays()
        elif attr == "_show_trail_pts":
            self._update_trav_trail(self.current_idx)

    _LEGEND_ATTRS = [
        ("_show_trav",        "_cb_trav"),
        ("_show_ground",      "_cb_ground"),
        ("_show_other",       "_cb_other"),
        ("_show_traj_past",   "_cb_traj_past"),
        ("_show_traj_future", "_cb_traj_future"),
        ("_show_robot",       "_cb_robot"),
        ("_show_trail_pts",   "_cb_trail_pts"),
    ]

    def _set_all_legend(self, value: bool) -> None:
        for state_attr, cb_attr in self._LEGEND_ATTRS:
            setattr(self, state_attr, value)
            cb = getattr(self, cb_attr)
            if cb is not None:
                cb.checked = value
        self._refresh()
        self._update_trav_trail(self.current_idx)

    # ------------------------------------------------------------------
    # Data access (lazy cache)
    # ------------------------------------------------------------------

    def _get_scan(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if idx not in self._scan_cache:
            self._scan_cache[idx] = self.seq.get_scan(idx)
        return self._scan_cache[idx]

    def _get_labels(self, idx: int, xyz: np.ndarray) -> np.ndarray:
        if idx in self._label_cache:
            return self._label_cache[idx]

        # Try loading pre-computed labels
        if self.label_dir is not None:
            label_file = self.label_dir / (Path(self.seq.cloud_files[idx]).stem + ".trav")
            if label_file.exists():
                labels = np.fromfile(str(label_file), dtype=np.uint8)
                if len(labels) == len(xyz):
                    self._label_cache[idx] = labels
                    return labels

        # Compute on-the-fly
        if self.poses is not None:
            labels = self.labeler.label_scan(xyz, self.poses, idx)
        else:
            labels = np.zeros(len(xyz), dtype=np.uint8)

        self._label_cache[idx] = labels
        return labels

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        idx = self.current_idx
        xyz, intensity = self._get_scan(idx)

        # Sync navigation widgets (guard against triggering callbacks)
        self._updating = True
        if self._idx_slider is not None:
            self._idx_slider.int_value = idx
        if self._num_edit is not None:
            self._num_edit.int_value = idx
        self._updating = False

        if (self._n_past > 0 or self._n_future > 0) and self.poses is not None:
            # Accumulate past and/or future scans with origin tracking.
            # label_accumulated ensures each point is only matched against poses
            # AFTER it was observed - correct for both past and future scans.
            xyz_disp, intensity_disp, scan_origins = self._accumulate_scans(
                idx, xyz, intensity
            )
            labels_disp = self.labeler.label_accumulated(
                xyz_disp, scan_origins, self.poses, idx
            )
            labels_curr = labels_disp[:len(xyz)]   # stats from current scan
        else:
            xyz_disp, intensity_disp = xyz, intensity
            labels_disp = self._get_labels(idx, xyz)   # uses per-scan cache
            labels_curr = labels_disp

        # Feed the persistent trail with current-scan traversable points.
        if self._trail_active and self.poses is not None and idx not in self._trail_visited:
            self._append_to_trail(idx, xyz, labels_curr)

        fname = Path(self.seq.cloud_files[idx]).name
        self._lbl_frame.text = f"Scan {idx + 1} / {len(self.seq)}\n{fname}"
        self._lbl_npts.text  = f"Points: {len(xyz_disp):,}"

        n_trav = int(labels_curr.sum())
        pct    = 100.0 * n_trav / max(len(xyz), 1)
        self._lbl_stats.text = (
            f"Traversable : {n_trav:,} pts\n"
            f"            = {pct:.1f}%\n"
            f"(current scan, {len(xyz):,} pts)"
        )

        # Forward-only preview: apply only to current scan (first len(xyz) points).
        # Accumulated points from past scans are kept as-is.
        if self._fwd_only and self.poses is not None:
            n_curr = len(xyz)
            fmask_curr = self.labeler.forward_mask(xyz_disp[:n_curr], self.poses, idx)
            full_fmask = np.ones(len(xyz_disp), dtype=bool)
            full_fmask[:n_curr] = fmask_curr
            xyz_disp       = xyz_disp[full_fmask]
            intensity_disp = intensity_disp[full_fmask]
            labels_disp    = labels_disp[full_fmask]

        # Category visibility filter (always applied, all colour modes)
        if not (self._show_trav and self._show_ground and self._show_other):
            h_mask = (
                (xyz_disp[:, 2] >= self.labeler.height_min)
                & (xyz_disp[:, 2] <= self.labeler.height_max)
            )
            vis = np.zeros(len(xyz_disp), dtype=bool)
            if self._show_trav:   vis |= (labels_disp == 1)
            if self._show_ground: vis |= (h_mask & (labels_disp != 1))
            if self._show_other:  vis |= ~h_mask
            xyz_disp       = xyz_disp[vis]
            intensity_disp = intensity_disp[vis]
            labels_disp    = labels_disp[vis]

        self._update_cloud(xyz_disp, intensity_disp, labels_disp)
        self._update_trav_trail(idx)
        self._refresh_overlays(xyz)

    def _refresh_overlays(self, xyz: Optional[np.ndarray] = None) -> None:
        if xyz is None:
            xyz, _ = self._get_scan(self.current_idx)
        self._update_trajectory()
        self._update_robot(xyz)

    # ------------------------------------------------------------------
    # Scan accumulation
    # ------------------------------------------------------------------

    def _accumulate_scans(
        self,
        current_idx: int,
        xyz_curr: np.ndarray,
        intensity_curr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Stack past and future scans into the current scan's coordinate frame.

        Returns xyz, intensity, and scan_origins (origin scan index per point).
        The scan_origins array is passed to label_accumulated so that each point
        is only matched against poses AFTER it was observed.

        - Past scan points: forward-filter optionally applied; each point labeled
          by poses after its origin scan.
        - Future scan points: labeled by poses after the future scan's index,
          showing terrain that the robot will traverse later.
        """
        T_scan_world = np.linalg.inv(self.poses[current_idx])

        all_xyz       = [xyz_curr]
        all_intensity = [intensity_curr]
        all_origins   = [np.full(len(xyz_curr), current_idx, dtype=np.int32)]

        step    = self._accum_step
        MAX_PTS = 20_000

        def _transform_scan(k: int, apply_fwd_filter: bool) -> None:
            xyz_k, intensity_k = self._get_scan(k)
            if apply_fwd_filter and self._forward_accum:
                fmask = self.labeler.forward_mask(xyz_k, self.poses, k)
                xyz_k       = xyz_k[fmask]
                intensity_k = intensity_k[fmask]
            if len(xyz_k) == 0:
                return
            ds = max(1, len(xyz_k) // MAX_PTS)
            xyz_k       = xyz_k[::ds]
            intensity_k = intensity_k[::ds]
            T_scan_k = T_scan_world @ self.poses[k]
            R, t = T_scan_k[:3, :3], T_scan_k[:3, 3]
            all_xyz.append(((R @ xyz_k.T).T + t).astype(np.float32))
            all_intensity.append(intensity_k)
            all_origins.append(np.full(len(xyz_k), k, dtype=np.int32))

        # Past scans: go back n_past*step frames, sample every step-th
        if self._n_past > 0:
            past_start = max(0, current_idx - self._n_past * step)
            for k in range(past_start, current_idx, step):
                _transform_scan(k, apply_fwd_filter=True)

        # Future scans: go forward n_future*step frames, sample every step-th
        if self._n_future > 0:
            future_end = min(len(self.seq), current_idx + self._n_future * step + 1)
            for k in range(current_idx + step, future_end, step):
                _transform_scan(k, apply_fwd_filter=False)

        return (
            np.vstack(all_xyz),
            np.concatenate(all_intensity),
            np.concatenate(all_origins),
        )

    def _append_to_trail(
        self,
        idx: int,
        xyz_local: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Transform traversable points of scan idx to world frame and append to trail."""
        trav_mask = labels == 1
        if not trav_mask.any():
            self._trail_visited.add(idx)
            self._update_trail_label()
            return

        xyz_trav = xyz_local[trav_mask]
        T = self.poses[idx]
        R, t = T[:3, :3], T[:3, 3]
        xyz_world = (R @ xyz_trav.T).T + t

        # Subsample if adding would exceed cap
        if self._trav_trail_world is not None:
            available = TRAIL_MAX_PTS - len(self._trav_trail_world)
            if available <= 0:
                self._trail_visited.add(idx)
                return
            if len(xyz_world) > available:
                step = max(1, len(xyz_world) // available)
                xyz_world = xyz_world[::step]

        self._trav_trail_world = (
            xyz_world if self._trav_trail_world is None
            else np.vstack([self._trav_trail_world, xyz_world])
        )
        self._trail_visited.add(idx)
        self._update_trail_label()

    def _update_trav_trail(self, current_idx: int) -> None:
        """Render the persistent trail in the current scan's local frame."""
        scene = self._scene.scene
        scene.remove_geometry("trav_trail")

        if not self._show_trail_pts:
            return
        if (
            self._trav_trail_world is None
            or len(self._trav_trail_world) == 0
            or self.poses is None
        ):
            return

        # Transform world-frame trail into current scan's local frame
        T_scan_world = np.linalg.inv(self.poses[current_idx])
        R, t = T_scan_world[:3, :3], T_scan_world[:3, 3]
        xyz_local = (R @ self._trav_trail_world.T).T + t

        trail_mat = rendering.MaterialRecord()
        trail_mat.shader = "defaultUnlit"
        trail_mat.point_size = POINT_SIZE

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_local.astype(np.float64))
        color = np.tile(self.C_TRAIL, (len(xyz_local), 1))
        pcd.colors = o3d.utility.Vector3dVector(color)
        scene.add_geometry("trav_trail", pcd, trail_mat)

    def _update_cloud(
        self,
        xyz:       np.ndarray,
        intensity: np.ndarray,
        labels:    np.ndarray,
    ) -> None:
        scene = self._scene.scene
        scene.remove_geometry("cloud")

        colors = self._compute_colors(xyz, intensity, labels)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors)
        scene.add_geometry("cloud", pcd, self._mat)

        if not self._camera_initialized:
            self._look_top()
            self._camera_initialized = True

    def _compute_colors(
        self,
        xyz:       np.ndarray,
        intensity: np.ndarray,
        labels:    np.ndarray,
    ) -> np.ndarray:
        n = len(xyz)

        if self._display_mode == 1:  # Intensity
            i_max = intensity.max()
            gray = intensity / i_max if i_max > 0 else np.zeros(n)
            gray = np.clip(gray, 0, 1)
            return np.stack([gray, gray, gray], axis=1).astype(np.float64)

        if self._display_mode == 2:  # Height
            z = xyz[:, 2]
            t = (z - z.min()) / max(z.max() - z.min(), 1e-6)
            return _cm.viridis(t)[:, :3].astype(np.float64)

        # Mode 0: Traversability
        height_mask = (
            (xyz[:, 2] >= self.labeler.height_min)
            & (xyz[:, 2] <= self.labeler.height_max)
        )
        colors = np.empty((n, 3), dtype=np.float64)
        colors[:] = self.C_OTHER
        colors[height_mask] = self.C_GROUND
        colors[labels == 1] = self.C_TRAV
        return colors

    def _line_material(self, rgb: list) -> rendering.MaterialRecord:
        """Create an unlitLine material with the given RGB colour."""
        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = self._line_mat.line_width
        mat.base_color = [rgb[0], rgb[1], rgb[2], 1.0]
        return mat

    # ------------------------------------------------------------------
    # Trajectory overlay
    # ------------------------------------------------------------------

    def _update_trajectory(self) -> None:
        scene = self._scene.scene
        scene.remove_geometry("traj_past")
        scene.remove_geometry("traj_future")

        if self.poses is None:
            return
        if not self._show_traj_past and not self._show_traj_future:
            return

        idx          = self.current_idx
        T_scan_world = np.linalg.inv(self.poses[idx])
        win          = self.labeler.trajectory_window
        lo           = max(0, idx - win)
        hi           = min(len(self.poses), idx + win + 1)

        def _build_ls(indices: list):
            if not indices:
                return None
            raw = [(T_scan_world @ self.poses[k])[:3, 3] for k in indices]
            # Filter consecutive near-duplicate points (avoids zero-AABB crash).
            pts = [raw[0]]
            for p in raw[1:]:
                if np.linalg.norm(p - pts[-1]) > 1e-3:
                    pts.append(p)
            if len(pts) < 2:
                return None
            pts_arr = np.array(pts, dtype=np.float64)
            edges = [[i, i + 1] for i in range(len(pts) - 1)]
            # colours stored in LineSet (unused by unlitLine, but harmless)
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(pts_arr)
            ls.lines  = o3d.utility.Vector2iVector(edges)
            return ls

        if self._show_traj_past:
            ls_past = _build_ls(list(range(lo, idx + 1)))
            if ls_past is not None:
                scene.add_geometry("traj_past", ls_past, self._line_material(self.C_TRAJ_PAST))

        if self._show_traj_future:
            ls_future = _build_ls(list(range(idx, hi)))
            if ls_future is not None:
                scene.add_geometry("traj_future", ls_future, self._line_material(self.C_TRAJ_FUTURE))

    # ------------------------------------------------------------------
    # Robot footprint overlay
    # ------------------------------------------------------------------

    def _update_robot(self, xyz: np.ndarray) -> None:
        scene = self._scene.scene
        scene.remove_geometry("robot")

        if not self._show_robot:
            return

        ground_z = float(np.percentile(xyz[:, 2], 5))
        top_z    = ground_z + self.robot_size * 0.8

        ls = self._make_footprint_ls(ground_z, top_z)
        scene.add_geometry("robot", ls, self._line_material(self.C_ROBOT))

    def _make_footprint_ls(self, z_bot: float, z_top: float) -> o3d.geometry.LineSet:
        s = self.robot_size / 2.0

        if self.robot_shape == "round":
            n_seg = 24
            angles = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
            bot = [[s * np.cos(a), s * np.sin(a), z_bot] for a in angles]
            top = [[s * np.cos(a), s * np.sin(a), z_top] for a in angles]
        else:  # square
            bot = [[-s, -s, z_bot], [s, -s, z_bot], [s, s, z_bot], [-s, s, z_bot]]
            top = [[-s, -s, z_top], [s, -s, z_top], [s, s, z_top], [-s, s, z_top]]

        n   = len(bot)
        pts = np.array(bot + top, dtype=np.float64)
        edges = (
            [[i, (i + 1) % n]     for i in range(n)] +   # bottom ring
            [[n + i, n + (i+1)%n] for i in range(n)] +   # top ring
            [[i, n + i]           for i in range(n)]      # verticals
        )
        return _make_lineset(pts, edges, self.C_ROBOT)

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------

    def _reset_camera(self) -> None:
        if not self._camera_initialized:
            return
        xyz, _ = self._get_scan(self.current_idx)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        bounds = pcd.get_axis_aligned_bounding_box()
        self._scene.setup_camera(60, bounds, bounds.get_center())

    def _look_top(self) -> None:
        xyz, _ = self._get_scan(self.current_idx)
        r   = float(np.percentile(np.linalg.norm(xyz[:, :2], axis=1), 90))
        alt = max(r * 1.2, 20.0)
        self._scene.scene.camera.look_at(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, alt],
            [1.0, 0.0, 0.0],
        )

    def _look_lidar(self) -> None:
        """First-person view: camera at sensor origin looking in the direction of travel."""
        # Determine forward direction in scan-local frame.
        # Default to +x if poses are unavailable or robot is stationary.
        fwd = np.array([1.0, 0.0])
        if self.poses is not None:
            idx = self.current_idx
            T_scan_world = np.linalg.inv(self.poses[idx])
            if idx + 1 < len(self.poses):
                other = (T_scan_world @ self.poses[idx + 1])[:3, 3]
            elif idx > 0:
                other = -(T_scan_world @ self.poses[idx - 1])[:3, 3]
            else:
                other = np.array([1.0, 0.0, 0.0])
            norm = np.linalg.norm(other[:2])
            if norm > 1e-6:
                fwd = other[:2] / norm

        eye    = [0.0, 0.0, 0.3]
        center = [float(fwd[0]) * 20.0, float(fwd[1]) * 20.0, 0.0]
        up     = [0.0, 0.0, 1.0]

        # setup_camera recalibrates the near/far clipping planes based on the
        # current scene bounds - necessary because a camera at the origin would
        # otherwise be clipped by the near plane set during the initial view.
        # look_at then overrides the camera position to the sensor origin.
        xyz, _ = self._get_scan(self.current_idx)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        bounds = pcd.get_axis_aligned_bounding_box()
        self._scene.setup_camera(70.0, bounds, center)
        self._scene.scene.camera.look_at(center, eye, up)
