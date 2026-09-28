"""The geometry section: draw and edit a case's layers on a Bokeh canvas.

The in-page half of the hybrid the owner chose on 2026-09-28 (the Gmsh half is
`gmsh_import.py`).  One canvas shows all four layers; one layer at a time is
editable, with one tool at a time:

* **Move & draw**: drag a shape to move it, Shift+drag to draw a new one,
  click one and press Backspace to delete it (Bokeh's `BoxEditTool`, or
  `PointDrawTool` for devices, where a click adds one).
* **Resize**: drag a corner handle (`PointDrawTool` on the corners); the
  opposite corner stays put.
* **Pan & zoom**: move the view without editing anything.

Every change is snapped to the grid, applied to the case through
`Workbench.edit` (so it is validated, undoable and saved like any other edit),
and drawn back from the case, so the canvas can never show a shape the case
does not hold.  The warnings drawn on the canvas are `geometry.analyse_windows`,
the computation `spec.check` refuses on.

Bokeh gives each gesture to one tool at a time, so the drag-to-move tool and the
drag-a-corner tool cannot both be live; that is why Resize is a separate tool.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pandas as pd
import panel as pn
from bokeh.models import (BoxEditTool, BoxZoomTool, ColumnDataSource, HoverTool, LabelSet,
                          PanTool, PointDrawTool, Range1d, ResetTool, SaveTool,
                          SingleIntervalTicker, WheelZoomTool)
from bokeh.plotting import figure

from . import geometry as geo
from . import registry
from .spec import Boundary, CaseSpec, Device, Region, Window, check, family_boundaries

if TYPE_CHECKING:                                             # pragma: no cover
    from .app import Workbench

LAYERS = {"windows": "Windows", "regions": "Regions", "devices": "Devices",
          "boundaries": "Boundaries"}
TOOLS = {"move": "Move & draw", "resize": "Resize", "pan": "Pan & zoom"}

C_DOMAIN = "#64748b"
C_WINDOW = "#2a9d8f"
C_OVERLAP = "#e9a23b"
C_ROTOR = "#d1495b"
C_BAD = "#c1121f"
C_GAP = "#6b7280"
C_HANDLE = "#0e6874"

#: Common materials get a colour that reads as the material; others take the
#: palette in order of first appearance.
MATERIAL_COLOURS = {"steel": "#8d99ae", "copper": "#c8733a", "aluminium": "#b8c4d6",
                    "aluminum": "#b8c4d6", "air": "#bde0fe", "water": "#4895ef",
                    "concrete": "#a8a29e", "wood": "#b08968"}
PALETTE = ("#6d597a", "#90be6d", "#577590", "#f4a261", "#e76f51", "#43aa8b", "#f9c74f",
           "#277da1")
BOUNDARY_COLOURS = {"inlet": "#1d4ed8", "freestream": "#0891b2", "outlet": "#7c3aed",
                    "wall": "#334155", "fixed-temperature": "#dc2626", "insulated": "#a16207",
                    "heat-flux": "#ea580c", "fixed-potential": "#16a34a",
                    "no-current": "#6b7280"}

HELP = {
    ("windows", "move"): "Drag a window to move it. **Shift+drag** on empty space to draw a "
                         "new one. Click a window, then **Backspace** (or *Delete selected*) "
                         "to remove it; Esc clears the selection.",
    ("windows", "resize"): "Drag a **corner handle** to resize that window; the opposite "
                           "corner stays put.",
    ("regions", "move"): "Drag a region to move it. **Shift+drag** to draw a new one in the "
                         "material chosen below. Click one, then **Backspace** to remove it. "
                         "Regions stack in list order: a later one takes the cells it covers. "
                         "Imported polygons are shown but not moved here.",
    ("regions", "resize"): "Drag a **corner handle** to resize a rectangular region.",
    ("devices", "move"): "**Click** empty space to add a rotor, drag one to move it, click one "
                         "and press **Backspace** to remove it.",
    ("devices", "resize"): "Devices have no corners; set a rotor's diameter in the table.",
    ("boundaries", "move"): "Boundaries are set in the table below and drawn along the "
                            "domain's edges.",
    ("boundaries", "resize"): "Boundaries are set in the table below.",
}


def _pan_help(layer: str) -> str:
    return "Drag to pan, scroll to zoom. Nothing is edited in this tool."


def material_colours(regions) -> dict[str, str]:
    out: dict[str, str] = {}
    k = 0
    for r in regions:
        if r.material in out:
            continue
        if r.material.lower() in MATERIAL_COLOURS:
            out[r.material] = MATERIAL_COLOURS[r.material.lower()]
        else:
            out[r.material] = PALETTE[k % len(PALETTE)]
            k += 1
    return out


def _next_id(prefix: str, taken) -> str:
    taken = set(taken)
    k = len(taken) + 1
    while f"{prefix}{k}" in taken:
        k += 1
    return f"{prefix}{k}"


class GeometryEditor:
    """The Geometry step's workspace for one version of the layout.

    Rebuilt when the layer, the tool, the domain or the view options change;
    otherwise `sync` redraws it from the case in place, keeping the view.
    """

    def __init__(self, wb: "Workbench", ranges: tuple[float, float, float, float] | None = None):
        self.wb = wb
        self.state = wb.geo_state
        self._syncing = False
        d = wb.spec.domain
        self._domain = (d.nx, d.ny)
        self._build_sources()
        self.fig = self._figure(ranges)
        self.fig_pane = pn.pane.Bokeh(self.fig, sizing_mode="stretch_width")
        self.summary = pn.pane.HTML(sizing_mode="stretch_width", margin=(0, 10))
        self.issues = pn.pane.Markdown(sizing_mode="stretch_width", margin=(0, 10))
        self._build_controls()
        self.sync()

    # ------------------------------------------------------------------
    # data sources
    # ------------------------------------------------------------------
    def _build_sources(self) -> None:
        cols = dict(x=[], y=[], w=[], h=[], id=[])
        self.src_win = ColumnDataSource(dict(cols))
        self.src_reg_edit = ColumnDataSource(dict(cols))          # rectangles, for editing
        self.src_reg_draw = ColumnDataSource(dict(xs=[], ys=[], id=[], material=[], colour=[],
                                                  shape=[]))
        self.src_handles = ColumnDataSource(dict(x=[], y=[], owner=[], corner=[]))
        self.src_dev = ColumnDataSource(dict(x=[], y=[], id=[]))
        self.src_dev_seg = ColumnDataSource(dict(x=[], y0=[], y1=[], id=[]))
        self.src_overlap = ColumnDataSource(dict(x=[], y=[], w=[], h=[]))
        self.src_bad = ColumnDataSource(dict(left=[], right=[], bottom=[], top=[]))
        self.src_gap = ColumnDataSource(dict(left=[], right=[], bottom=[], top=[]))
        self.src_cross = ColumnDataSource(dict(x=[], y=[], names=[]))
        self.src_bc = ColumnDataSource(dict(x0=[], y0=[], x1=[], y1=[], kind=[], id=[],
                                            colour=[], lx=[], ly=[], angle=[], text=[]))
        self.src_win.on_change("data", self._on_windows)
        self.src_reg_edit.on_change("data", self._on_regions)
        self.src_handles.on_change("data", self._on_handles)
        self.src_dev.on_change("data", self._on_devices)

    # ------------------------------------------------------------------
    # the canvas
    # ------------------------------------------------------------------
    def _figure(self, ranges):
        d = self.wb.spec.domain
        pad = max(8, int(0.05 * max(d.nx, d.ny)))
        if ranges is None:
            ranges = (-pad, d.nx + pad, -pad, d.ny + pad)
        x0, x1, y0, y1 = ranges
        aspect = (x1 - x0) / max(y1 - y0, 1e-9)
        p = figure(x_range=Range1d(x0, x1), y_range=Range1d(y0, y1),
                   width=1000, height=max(240, int(1000 / aspect)),
                   # square cells, and no taller than about 470 px so the tables
                   # below stay in reach on a laptop screen
                   sizing_mode="scale_width", max_width=int(470 * aspect) + 80,
                   tools=[], toolbar_location="right", output_backend="canvas")
        p.xaxis.axis_label, p.yaxis.axis_label = "x (cells)", "y (cells)"
        for g in (p.xgrid, p.ygrid):
            g.visible = self.wb.view_opts["grid"]
            g.ticker = SingleIntervalTicker(interval=max(self.state["snap"], 8))
            g.grid_line_alpha = 0.35
        p.rect(x=d.nx / 2, y=d.ny / 2, width=d.nx, height=d.ny, fill_color="#ffffff",
               fill_alpha=1.0, line_color=C_DOMAIN, line_width=2, level="underlay")

        # regions: filled shapes in stacking order, then the editable outlines
        rr = p.multi_polygons("xs", "ys", source=self.src_reg_draw, fill_color="colour",
                              fill_alpha=0.45, line_color="colour", line_alpha=0.9,
                              line_width=1)
        p.add_tools(HoverTool(renderers=[rr], visible=False, tooltips=[("region", "@id"),
                                                        ("material", "@material"),
                                                        ("shape", "@shape")]))
        editing_regions = self.state["layer"] == "regions"
        reg_edit = p.rect("x", "y", "w", "h", source=self.src_reg_edit, fill_alpha=0.0,
                          line_color="#1f2937", line_dash="dashed",
                          line_alpha=0.9 if editing_regions else 0.0, line_width=1.5,
                          selection_line_color=C_HANDLE, selection_line_width=3,
                          nonselection_line_alpha=0.9 if editing_regions else 0.0)

        # warnings under the windows
        p.quad("left", "right", "bottom", "top", source=self.src_gap, fill_color=C_GAP,
               fill_alpha=0.15, hatch_pattern="/", hatch_color=C_GAP, hatch_alpha=0.6,
               line_alpha=0)
        bad = p.quad("left", "right", "bottom", "top", source=self.src_bad, fill_color=C_BAD,
                     fill_alpha=0.18, hatch_pattern="x", hatch_color=C_BAD, hatch_alpha=0.7,
                     line_alpha=0)
        p.add_tools(HoverTool(renderers=[bad], visible=False, tooltips=[("", "no window at full weight")]))
        if self.wb.view_opts["overlaps"]:
            p.rect("x", "y", "w", "h", source=self.src_overlap, fill_color=C_OVERLAP,
                   fill_alpha=0.22, line_alpha=0)

        editing_windows = self.state["layer"] == "windows"
        win = p.rect("x", "y", "w", "h", source=self.src_win, fill_color=C_WINDOW,
                     fill_alpha=0.06, line_color=C_WINDOW,
                     line_width=2 if editing_windows else 1.2,
                     line_alpha=1.0 if editing_windows else 0.55,
                     selection_fill_alpha=0.25, selection_line_width=3,
                     nonselection_fill_alpha=0.06,
                     nonselection_line_alpha=1.0 if editing_windows else 0.55)
        p.add_tools(HoverTool(renderers=[win], visible=False, tooltips=[("window", "@id"),
                                                         ("size", "@w x @h cells")]))
        if self.wb.view_opts["labels"]:
            p.add_layout(LabelSet(x="x", y="y", text="id", source=self.src_win,
                                  text_align="center", text_baseline="middle",
                                  text_font_size="10px", text_color=C_WINDOW,
                                  text_alpha=0.8))
        cross = p.scatter("x", "y", source=self.src_cross, marker="diamond", size=9,
                          fill_color="#ffffff", line_color="#374151", line_width=1.5)
        p.add_tools(HoverTool(renderers=[cross], visible=False, tooltips=[("cross-point", "@names")]))

        # boundaries, just outside the domain
        bcr = p.segment("x0", "y0", "x1", "y1", source=self.src_bc, line_color="colour",
                        line_width=6, line_cap="butt")
        p.add_tools(HoverTool(renderers=[bcr], visible=False, tooltips=[("boundary", "@id"),
                                                         ("kind", "@text")]))
        p.add_layout(LabelSet(x="lx", y="ly", text="text", angle="angle", source=self.src_bc,
                              text_align="center", text_baseline="middle",
                              text_font_size="10px", text_color="colour",
                              background_fill_color="#ffffff", background_fill_alpha=0.8))

        # devices
        p.segment("x", "y0", "x", "y1", source=self.src_dev_seg, line_color=C_ROTOR,
                  line_width=4)
        dev = p.scatter("x", "y", source=self.src_dev, size=9, fill_color="#ffffff",
                        line_color=C_ROTOR, line_width=2, selection_fill_color=C_ROTOR,
                        nonselection_fill_alpha=1.0)
        p.add_tools(HoverTool(renderers=[dev], visible=False, tooltips=[("device", "@id")]))
        if self.wb.view_opts["labels"]:
            p.add_layout(LabelSet(x="x", y="y1", text="id", source=self.src_dev_seg,
                                  x_offset=4, y_offset=2, text_font_size="11px",
                                  text_color=C_ROTOR))

        # resize handles, drawn only for the Resize tool
        resize = self.state["tool"] == "resize" and self.state["layer"] in ("windows",
                                                                           "regions")
        hnd = p.scatter("x", "y", source=self.src_handles, marker="square", size=9,
                        fill_color="#ffffff", line_color=C_HANDLE, line_width=2,
                        visible=resize, nonselection_fill_alpha=1.0)

        pan, wheel = PanTool(), WheelZoomTool()
        zoom = BoxZoomTool(match_aspect=True)
        p.add_tools(pan, wheel, zoom, ResetTool(), SaveTool())
        p.toolbar.active_scroll = wheel
        edit_tool = None
        layer, tool = self.state["layer"], self.state["tool"]
        if tool == "move" and layer == "windows":
            edit_tool = BoxEditTool(renderers=[win], empty_value="",
                                    description="Move, draw (Shift+drag) and delete windows")
        elif tool == "move" and layer == "regions":
            edit_tool = BoxEditTool(renderers=[reg_edit], empty_value="",
                                    description="Move, draw (Shift+drag) and delete regions")
        elif tool == "move" and layer == "devices":
            edit_tool = PointDrawTool(renderers=[dev], empty_value="", add=True, drag=True,
                                      description="Add (click), move and delete devices")
        elif resize:
            edit_tool = PointDrawTool(renderers=[hnd], empty_value="", add=False, drag=True,
                                      description="Resize by dragging a corner")
        if edit_tool is not None:
            p.add_tools(edit_tool)
            p.toolbar.active_multi = edit_tool
            p.toolbar.active_drag = None
        else:
            p.toolbar.active_drag = pan
        p.toolbar.logo = None
        return p

    def ranges(self) -> tuple[float, float, float, float]:
        xr, yr = self.fig.x_range, self.fig.y_range
        return float(xr.start), float(xr.end), float(yr.start), float(yr.end)

    # ------------------------------------------------------------------
    # case -> canvas and tables
    # ------------------------------------------------------------------
    def _set(self, src: ColumnDataSource, data: dict) -> None:
        self._syncing = True
        try:
            src.data = data
            src.selected.indices = []
        finally:
            self._syncing = False

    def sync(self) -> None:
        """Redraw every layer, warning and table from the case."""
        s, d = self.wb.spec, self.wb.spec.domain
        ws = s.windows
        self._set(self.src_win, dict(x=[w.x0 + w.nx / 2 for w in ws],
                                     y=[w.y0 + w.ny / 2 for w in ws],
                                     w=[w.nx for w in ws], h=[w.ny for w in ws],
                                     id=[w.id for w in ws]))
        rects = [(k, r) for k, r in enumerate(s.regions) if r.shape == "rect"]
        self._rect_index = [k for k, _ in rects]
        self._set(self.src_reg_edit, dict(x=[r.x0 + r.nx / 2 for _, r in rects],
                                          y=[r.y0 + r.ny / 2 for _, r in rects],
                                          w=[r.nx for _, r in rects],
                                          h=[r.ny for _, r in rects],
                                          id=[r.id for _, r in rects]))
        cmap = material_colours(s.regions)
        xs, ys = [], []
        for r in s.regions:
            if r.shape == "rect":
                ring = geo.corners((r.x0, r.y0, r.nx, r.ny))
                xs.append([[[p[0] for p in ring]]])
                ys.append([[[p[1] for p in ring]]])
            else:
                rings = [r.points] + list(r.holes or [])
                xs.append([[[p[0] for p in ring] for ring in rings]])
                ys.append([[[p[1] for p in ring] for ring in rings]])
        self._set(self.src_reg_draw, dict(xs=xs, ys=ys, id=[r.id for r in s.regions],
                                          material=[r.material for r in s.regions],
                                          colour=[cmap[r.material] for r in s.regions],
                                          shape=[r.shape for r in s.regions]))
        self._sync_handles()
        self._set(self.src_dev, dict(x=[v.x / d.dx for v in s.devices],
                                     y=[v.y / d.dx for v in s.devices],
                                     id=[v.id for v in s.devices]))
        self._set(self.src_dev_seg, dict(x=[v.x / d.dx for v in s.devices],
                                         y0=[(v.y - v.diameter / 2) / d.dx for v in s.devices],
                                         y1=[(v.y + v.diameter / 2) / d.dx for v in s.devices],
                                         id=[v.id for v in s.devices]))
        self._sync_warnings()
        self._sync_boundaries()
        self._sync_tables()

    def _boxes(self) -> list[tuple[int, geo.Box]]:
        """The boxes the Resize tool works on: (index into the layer, box)."""
        s = self.wb.spec
        if self.state["layer"] == "windows":
            return [(k, (w.x0, w.y0, w.nx, w.ny)) for k, w in enumerate(s.windows)]
        if self.state["layer"] == "regions":
            return [(k, (r.x0, r.y0, r.nx, r.ny)) for k, r in enumerate(s.regions)
                    if r.shape == "rect"]
        return []

    def _sync_handles(self) -> None:
        xs, ys, owner, corner = [], [], [], []
        for k, box in self._boxes():
            for c, (x, y) in enumerate(geo.corners(box)):
                xs.append(x)
                ys.append(y)
                owner.append(k)
                corner.append(c)
        self._set(self.src_handles, dict(x=xs, y=ys, owner=owner, corner=corner))

    def _sync_warnings(self) -> None:
        s, d = self.wb.spec, self.wb.spec.domain
        if s.windows:
            an = geo.analyse_windows(d.nx, d.ny, [(w.id, (w.x0, w.y0, w.nx, w.ny))
                                                  for w in s.windows], s.coupling.ramp_cells)
            bad, gap = geo.mask_to_boxes(an.ramp_only), geo.mask_to_boxes(an.uncovered)
            ov, cps = an.overlaps, an.cross_points
            n_bad, n_gap = int(an.ramp_only.sum()), int(an.uncovered.sum())
            n_thin = len(an.thin_pairs)
        else:
            bad, gap, ov, cps = [], [(0, 0, d.nx, d.ny)], [], []
            n_bad, n_gap, n_thin = 0, d.nx * d.ny, 0
        quad = lambda bs: dict(left=[b[0] for b in bs], right=[b[0] + b[2] for b in bs],  # noqa: E731
                               bottom=[b[1] for b in bs], top=[b[1] + b[3] for b in bs])
        self._set(self.src_bad, quad(bad))
        self._set(self.src_gap, quad(gap))
        self._set(self.src_overlap, dict(x=[b[0] + b[2] / 2 for _a, _b, b in ov],
                                         y=[b[1] + b[3] / 2 for _a, _b, b in ov],
                                         w=[b[2] for _a, _b, b in ov],
                                         h=[b[3] for _a, _b, b in ov]))
        self._set(self.src_cross, dict(x=[b[0] + b[2] / 2 for _n, b in cps],
                                       y=[b[1] + b[3] / 2 for _n, b in cps],
                                       names=[", ".join(n) for n, _b in cps]))
        ramp = s.coupling.ramp_cells
        parts = []
        if n_bad:
            parts.append(f"<span style='color:{C_BAD}'><b>{n_bad:,}</b> cells with no window "
                         f"at full weight</span> (red hatching; overlap by at least "
                         f"{2 * ramp} cells where windows meet)")
        if n_gap:
            parts.append(f"<span style='color:{C_GAP}'><b>{n_gap:,}</b> cells in no "
                         f"window</span> (grey hatching)")
        if n_thin:
            parts.append(f"<b>{n_thin}</b> thin seam{'s' * (n_thin > 1)}")
        parts.append(f"{len(cps)} cross-point{'s' * (len(cps) != 1)} (diamonds)")
        ok = not (n_bad or n_gap)
        lead = ("<b style='color:#15803d'>Windows cover the domain, and every cell has a "
                "window at full weight.</b> " if ok else "")
        self.summary.object = (f"<div style='font-size:13px'>{lead}"
                               + " &middot; ".join(parts) + "</div>")
        issues = [i for i in check(s) if i.step == "geometry"]
        if issues:
            self.issues.object = "\n".join(f"- **{i.severity}**: {i.message}"
                                           for i in issues[:8]) + (
                f"\n- ... and {len(issues) - 8} more (step 4)" if len(issues) > 8 else "")
        else:
            self.issues.object = "*No geometry issues.*"

    def _sync_boundaries(self) -> None:
        s, d = self.wb.spec, self.wb.spec.domain
        off = max(3.0, 0.02 * max(d.nx, d.ny))
        rows = {k: [] for k in ("x0", "y0", "x1", "y1", "kind", "id", "colour", "lx", "ly",
                                "angle", "text")}
        for b in s.boundaries:
            n = geo.edge_length(b.edge, d.nx, d.ny)
            stop = n if b.stop is None else b.stop
            x0, y0, x1, y1 = geo.edge_segment_xy(b.edge, b.start, stop, d.nx, d.ny)
            dx, dy = {"left": (-off, 0), "right": (off, 0), "bottom": (0, -off),
                      "top": (0, off)}[b.edge]
            rows["x0"].append(x0 + dx)
            rows["y0"].append(y0 + dy)
            rows["x1"].append(x1 + dx)
            rows["y1"].append(y1 + dy)
            rows["kind"].append(b.kind)
            rows["id"].append(b.id)
            rows["colour"].append(BOUNDARY_COLOURS.get(b.kind, "#111827"))
            # the label sits just inside the domain, beside its segment
            rows["lx"].append((x0 + x1) / 2 - 2.2 * dx)
            rows["ly"].append((y0 + y1) / 2 - 2.2 * dy)
            rows["angle"].append(math.pi / 2 if b.edge in ("left", "right") else 0.0)
            val = "" if b.value is None else f" = {b.value:g}"
            rows["text"].append(f"{b.kind}{val}")
        self._set(self.src_bc, rows)

    # ------------------------------------------------------------------
    # canvas -> case
    # ------------------------------------------------------------------
    def _snap(self) -> int:
        return int(self.state["snap"])

    def _boxes_from(self, data: dict) -> list[tuple[str, geo.Box | None, bool]]:
        """Rows of an edited box source as (id, snapped box, is new)."""
        d = self.wb.spec.domain
        out = []
        for x, y, w, h, i in zip(data["x"], data["y"], data["w"], data["h"], data["id"]):
            new = i in ("", None, 0)
            if new and (abs(w) < 2 or abs(h) < 2):
                out.append(("", None, True))              # a click, not a drawn box
                continue
            box = geo.snap_box(x - w / 2, y - h / 2, x + w / 2, y + h / 2, self._snap(),
                               d.nx, d.ny)
            out.append((str(i) if not new else "", box, new))
        return out

    def _on_windows(self, _attr, _old, new) -> None:
        if self._syncing:
            return
        rows = self._boxes_from(new)
        before = {w.id: w for w in self.wb.spec.windows}
        taken = set(before)
        result, drawn = [], []
        for wid, box, is_new in rows:
            if box is None:
                continue
            if is_new:
                wid = _next_id("F", taken)
                taken.add(wid)
                drawn.append(wid)
            result.append(Window(id=wid, x0=box[0], y0=box[1], nx=box[2], ny=box[3]))
        kept = {w.id for w in result}
        gone = [w for w in before if w not in kept]
        moved = [w.id for w in result if w.id in before
                 and (w.x0, w.y0, w.nx, w.ny) != (before[w.id].x0, before[w.id].y0,
                                                  before[w.id].nx, before[w.id].ny)]
        label = _describe("window", drawn, moved, gone)
        if label is None:
            self.sync()                                   # restore the snapped drawing
            return

        def apply(c: CaseSpec):
            c.windows = result
        if not self.wb.edit(apply, label):
            self.sync()

    def _on_regions(self, _attr, _old, new) -> None:
        if self._syncing:
            return
        rows = self._boxes_from(new)
        spec = self.wb.spec
        rect_ids = {spec.regions[k].id: k for k in self._rect_index}
        taken = {r.id for r in spec.regions}
        updated = {}
        drawn: list[Region] = []
        for rid, box, is_new in rows:
            if box is None:
                continue
            if is_new:
                rid = _next_id("R", taken)
                taken.add(rid)
                drawn.append(Region(id=rid, material=self.state["material"] or "material-1",
                                    x0=box[0], y0=box[1], nx=box[2], ny=box[3]))
            elif rid in rect_ids:
                updated[rid] = box
        gone = [rid for rid in rect_ids if rid not in updated]
        moved = [rid for rid, b in updated.items()
                 if b != (spec.regions[rect_ids[rid]].x0, spec.regions[rect_ids[rid]].y0,
                          spec.regions[rect_ids[rid]].nx, spec.regions[rect_ids[rid]].ny)]
        label = _describe("region", [r.id for r in drawn], moved, gone)
        if label is None:
            self.sync()
            return

        def apply(c: CaseSpec):
            out = []
            for r in c.regions:
                if r.shape == "rect" and r.id in gone:
                    continue
                if r.shape == "rect" and r.id in updated:
                    b = updated[r.id]
                    r.x0, r.y0, r.nx, r.ny = b
                out.append(r)
            c.regions = out + drawn
        if not self.wb.edit(apply, label):
            self.sync()

    def _on_handles(self, _attr, _old, new) -> None:
        if self._syncing:
            return
        d = self.wb.spec.domain
        boxes = dict(self._boxes())
        changed: dict[int, geo.Box] = {}
        for x, y, k, c in zip(new["x"], new["y"], new["owner"], new["corner"]):
            if k in changed or k not in boxes:
                continue
            ex, ey = geo.corners(boxes[k])[int(c)]
            if abs(x - ex) > 1e-9 or abs(y - ey) > 1e-9:
                b = geo.resize_from_corner(boxes[k], int(c), x, y, self._snap(), d.nx, d.ny)
                if b is not None and b != boxes[k]:
                    changed[k] = b
        if not changed:
            self.sync()
            return
        layer = self.state["layer"]
        spec = self.wb.spec
        names = [(spec.windows if layer == "windows" else spec.regions)[k].id for k in changed]

        def apply(c: CaseSpec):
            items = c.windows if layer == "windows" else c.regions
            for k, b in changed.items():
                items[k].x0, items[k].y0, items[k].nx, items[k].ny = b
        if not self.wb.edit(apply, f"resized {', '.join(names)}"):
            self.sync()

    def _on_devices(self, _attr, _old, new) -> None:
        if self._syncing:
            return
        spec, d = self.wb.spec, self.wb.spec.domain
        before = {v.id: v for v in spec.devices}
        taken = set(before)
        step = self._snap()
        result, drawn = [], []
        for x, y, i in zip(new["x"], new["y"], new["id"]):
            sx = geo.snap_value(x, step, d.nx) * d.dx
            sy = geo.snap_value(y, step, d.ny) * d.dx
            if i in ("", None, 0):
                vid = _next_id("R", taken)
                taken.add(vid)
                drawn.append(vid)
                result.append(Device(id=vid, x=sx, y=sy))
            else:
                old = before.get(str(i))
                if old is None:
                    continue
                result.append(old.model_copy(update=dict(x=sx, y=sy)))
        kept = {v.id for v in result}
        gone = [v for v in before if v not in kept]
        moved = [v.id for v in result if v.id in before
                 and (v.x, v.y) != (before[v.id].x, before[v.id].y)]
        label = _describe("device", drawn, moved, gone)
        if label is None:
            self.sync()
            return

        def apply(c: CaseSpec):
            c.devices = result
        if not self.wb.edit(apply, label):
            self.sync()

    # ------------------------------------------------------------------
    # controls and tables
    # ------------------------------------------------------------------
    def _build_controls(self) -> None:
        st = self.state
        self.layer_sel = pn.widgets.RadioButtonGroup(
            name="Layer", options={v: k for k, v in LAYERS.items()}, value=st["layer"],
            button_type="primary", button_style="outline")
        self.tool_sel = pn.widgets.RadioButtonGroup(
            name="Tool", options={v: k for k, v in TOOLS.items()}, value=st["tool"],
            button_type="primary", button_style="outline")
        self.snap_sel = pn.widgets.Select(name="", options={f"snap {s} cell{'s' * (s > 1)}": s
                                                           for s in geo.SNAP_STEPS},
                                          value=st["snap"], width=130)
        self.layer_sel.param.watch(lambda e: self._switch(layer=e.new), "value")
        self.tool_sel.param.watch(lambda e: self._switch(tool=e.new), "value")
        self.snap_sel.param.watch(lambda e: self._switch(snap=e.new), "value")
        self.help = pn.pane.Markdown(self._help_text(), sizing_mode="stretch_width",
                                     margin=(0, 10), styles={"font-size": "13px"})

        self.tables = {
            "windows": pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=240,
                                            layout="fit_data_table", selectable=True,
                                            sizing_mode="stretch_width"),
            "regions": pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=240,
                                            layout="fit_data_table", selectable=True,
                                            sizing_mode="stretch_width"),
            "devices": pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=240,
                                            layout="fit_data_table", selectable=True,
                                            sizing_mode="stretch_width"),
            "boundaries": pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=200,
                                               layout="fit_data_table", selectable=True,
                                               sizing_mode="stretch_width"),
        }
        for key, t in self.tables.items():
            t.on_edit(lambda e, key=key: self._on_table_edit(key, e))

    def _help_text(self) -> str:
        layer, tool = self.state["layer"], self.state["tool"]
        return (_pan_help(layer) if tool == "pan" else HELP[(layer, tool)]) + (
            f" Snapping to **{self.state['snap']}** cell{'s' * (self.state['snap'] > 1)}.")

    def _switch(self, **kw) -> None:
        self.state.update(kw)
        self.wb.log("geometry: " + ", ".join(f"{k} = {v}" for k, v in kw.items()))
        self.wb.show("geometry", keep_view=True)

    def _table_frames(self) -> dict[str, pd.DataFrame]:
        s, d = self.wb.spec, self.wb.spec.domain
        win = pd.DataFrame([dict(id=w.id, x0=w.x0, y0=w.y0, width=w.nx, height=w.ny,
                                 cells=w.nx * w.ny) for w in s.windows],
                           columns=["id", "x0", "y0", "width", "height", "cells"])
        reg = pd.DataFrame([dict(order=k + 1, id=r.id, material=r.material, shape=r.shape,
                                 x0=r.x0, y0=r.y0, width=r.nx, height=r.ny)
                            for k, r in enumerate(s.regions)],
                           columns=["order", "id", "material", "shape", "x0", "y0", "width",
                                    "height"])
        dev = pd.DataFrame([dict(id=v.id, kind=v.kind, x_D=v.x, y_D=v.y,
                                 diameter_D=v.diameter, yaw_deg=v.yaw_deg) for v in s.devices],
                           columns=["id", "kind", "x_D", "y_D", "diameter_D", "yaw_deg"])
        bnd = pd.DataFrame([dict(id=b.id, edge=b.edge, start=b.start,
                                 stop=geo.edge_length(b.edge, d.nx, d.ny) if b.stop is None
                                 else b.stop, kind=b.kind,
                                 value="" if b.value is None else f"{b.value:g}")
                            for b in s.boundaries],
                           columns=["id", "edge", "start", "stop", "kind", "value"])
        return {"windows": win, "regions": reg, "devices": dev, "boundaries": bnd}

    def _sync_tables(self) -> None:
        fam = _family(self.wb.spec)
        kinds = list(fam.boundary_kinds) if fam and fam.boundary_kinds else \
            [k.id for k in registry.BOUNDARY_KINDS]
        fixed = bool(fam and fam.fixed_boundaries)
        editors = {
            "windows": {"cells": None},
            "regions": {"order": None, "shape": None},
            "devices": {"kind": None},
            "boundaries": ({c: None for c in ("id", "edge", "start", "stop", "kind", "value")}
                           if fixed else
                           {"edge": {"type": "list", "values": list(geo.EDGES)},
                            "kind": {"type": "list", "values": kinds}}),
        }
        for key, df in self._table_frames().items():
            t = self.tables[key]
            t.editors = editors[key]
            t.value = df

    def _on_table_edit(self, key: str, e) -> None:
        row, col, val = e.row, e.column, e.value
        spec = self.wb.spec
        try:
            if key == "windows":
                field = {"id": "id", "x0": "x0", "y0": "y0", "width": "nx", "height": "ny"}[col]
                wid = spec.windows[row].id

                def apply(c):
                    setattr(c.windows[row], field, str(val) if field == "id" else int(val))
                label = f"window {wid}: {col} = {val}"
            elif key == "regions":
                reg = spec.regions[row]
                if reg.shape == "polygon" and col in ("x0", "y0", "width", "height"):
                    self.wb.notify("warning", f"region {reg.id} is an imported polygon; "
                                              f"change its shape in Gmsh and import it again")
                    self._sync_tables()
                    return
                field = {"id": "id", "material": "material", "x0": "x0", "y0": "y0",
                         "width": "nx", "height": "ny"}[col]

                def apply(c):
                    setattr(c.regions[row], field,
                            str(val) if field in ("id", "material") else int(val))
                label = f"region {reg.id}: {col} = {val}"
            elif key == "devices":
                field = {"id": "id", "x_D": "x", "y_D": "y", "diameter_D": "diameter",
                         "yaw_deg": "yaw_deg"}[col]
                vid = spec.devices[row].id

                def apply(c):
                    setattr(c.devices[row], field, str(val) if field == "id" else float(val))
                label = f"device {vid}: {col} = {val}"
            else:
                bid = spec.boundaries[row].id
                d = spec.domain

                def apply(c):
                    b = c.boundaries[row]
                    if col in ("id", "edge", "kind"):
                        setattr(b, col, str(val))
                    elif col == "start":
                        b.start = int(val)
                    elif col == "stop":
                        n = geo.edge_length(b.edge, d.nx, d.ny)
                        b.stop = None if int(val) == n else int(val)
                    elif col == "value":
                        b.value = None if str(val).strip() == "" else float(val)
                label = f"boundary {bid}: {col} = {val}"
        except (KeyError, IndexError):
            self._sync_tables()
            return
        try:
            ok = self.wb.edit(apply, label)
        except (TypeError, ValueError) as exc:
            self.wb.notify("error", f"not applied: {col} = {val!r}: {exc}")
            ok = False
        if not ok:
            self._sync_tables()

    # ------------------------------------------------------------------
    # buttons
    # ------------------------------------------------------------------
    def _selected(self, key: str) -> list[int]:
        """Rows selected in the table or on the canvas, as indices into the layer."""
        rows = set(self.tables[key].selection or [])
        if key == "windows":
            rows |= set(self.src_win.selected.indices)
        elif key == "regions":
            rows |= {self._rect_index[i] for i in self.src_reg_edit.selected.indices
                     if i < len(self._rect_index)}
        elif key == "devices":
            rows |= set(self.src_dev.selected.indices)
        return sorted(rows)

    def delete_selected(self) -> None:
        key = self.state["layer"]
        items = getattr(self.wb.spec, key)
        rows = [r for r in self._selected(key) if r < len(items)]
        if not rows:
            self.wb.notify("info", f"select {key} in the table or on the canvas first")
            return
        names = [items[r].id for r in rows]

        def apply(c):
            setattr(c, key, [x for k, x in enumerate(getattr(c, key)) if k not in rows])
        self.wb.edit(apply, f"deleted {key[:-1]} {', '.join(names)}")

    def add_shape(self) -> None:
        key = self.state["layer"]
        s, d = self.wb.spec, self.wb.spec.domain
        x0, x1, y0, y1 = self.ranges()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if key in ("windows", "regions"):
            size = min(128, d.nx, d.ny)
            box = geo.snap_box(cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2,
                               self._snap(), d.nx, d.ny)
            if key == "windows":
                wid = _next_id("F", [w.id for w in s.windows])
                new = Window(id=wid, x0=box[0], y0=box[1], nx=box[2], ny=box[3])
            else:
                wid = _next_id("R", [r.id for r in s.regions])
                new = Region(id=wid, material=self.state["material"] or "material-1",
                             x0=box[0], y0=box[1], nx=box[2], ny=box[3])
        elif key == "devices":
            wid = _next_id("R", [v.id for v in s.devices])
            new = Device(id=wid, x=geo.snap_value(cx, self._snap(), d.nx) * d.dx,
                         y=geo.snap_value(cy, self._snap(), d.ny) * d.dx)
        else:
            taken = [b.id for b in s.boundaries]
            wid = _next_id("B", taken)
            fam = _family(s)
            kind = (fam.boundary_kinds[0] if fam and fam.boundary_kinds
                    else registry.BOUNDARY_KINDS[0].id)
            new = Boundary(id=wid, edge="left", kind=kind)

        def apply(c):
            getattr(c, key).append(new)
        self.wb.edit(apply, f"added {key[:-1]} {wid}")

    def restack(self, step: int) -> None:
        rows = self._selected("regions")
        if len(rows) != 1:
            self.wb.notify("info", "select one region to move it up or down the stack")
            return
        k = rows[0]
        j = k + step
        if not 0 <= j < len(self.wb.spec.regions):
            return
        rid = self.wb.spec.regions[k].id

        def apply(c):
            c.regions[k], c.regions[j] = c.regions[j], c.regions[k]
        if self.wb.edit(apply, f"region {rid} moved {'up' if step > 0 else 'down'} the stack"):
            self.tables["regions"].selection = [j]

    def use_family_boundaries(self) -> None:
        fid = self.wb.spec.physics.family

        def apply(c):
            c.boundaries = family_boundaries(fid)
        self.wb.edit(apply, f"boundaries set to the {fid} family's")

    # ------------------------------------------------------------------
    # the view
    # ------------------------------------------------------------------
    def view(self):
        wb = self.wb
        key = self.state["layer"]
        s = wb.spec
        fam = _family(s)
        buttons = []

        def btn(name, fn, kind="default", width=None):
            b = pn.widgets.Button(name=name, button_type=kind, width=width or 150)
            b.on_click(lambda _e: fn())
            buttons.append(b)
            return b

        extra = []
        if key == "windows":
            btn("Generate a tiling...", wb.dialog_tiling, "primary", 170)
            btn("Add window", self.add_shape)
            btn("Delete selected", self.delete_selected)
        elif key == "regions":
            mat = pn.widgets.TextInput(placeholder="material for new regions",
                                       value=self.state["material"], width=190)
            mat.param.watch(lambda e: self.state.update(material=e.new.strip()), "value")
            buttons.append(mat)
            btn("Add region", self.add_shape)
            btn("Delete selected", self.delete_selected)
            btn("Move up the stack", lambda: self.restack(+1), width=160)
            btn("Move down the stack", lambda: self.restack(-1), width=170)
            if fam is not None and "regions" not in fam.layers:
                extra.append(pn.pane.Alert(
                    f"The **{fam.label}** family does not read material regions: its solver "
                    f"would ignore them. They are for the conduction family (showcase case "
                    f"2), which is planned.", alert_type="warning", sizing_mode="stretch_width"))
        elif key == "devices":
            btn("Add rotor", self.add_shape)
            btn("Delete selected", self.delete_selected)
        else:
            if fam is not None and fam.fixed_boundaries:
                btn("Use the family's boundaries", self.use_family_boundaries, "primary", 220)
                extra.append(pn.pane.Alert(
                    f"**Fixed by the solver.** {fam.fixed_boundaries_why}",
                    alert_type="info", sizing_mode="stretch_width"))
            else:
                btn("Add boundary segment", self.add_shape, "primary", 200)
                btn("Delete selected", self.delete_selected)
        btn("Import from Gmsh (.msh)...", wb.dialog_gmsh, width=210)

        return pn.Column(
            _row(self.layer_sel, self.tool_sel, self.snap_sel),
            self.help,
            self.fig_pane,
            self.summary,
            _row(*buttons),
            *extra,
            pn.pane.Markdown(f"**{LAYERS[key]}** "
                             + ("(read-only: the solver fixes them)"
                                if key == "boundaries" and fam and fam.fixed_boundaries
                                else "(click a cell to edit it; selected rows are what "
                                     "*Delete selected* removes)"),
                             margin=(0, 10)),
            self.tables[key],
            pn.pane.Markdown("**Geometry issues**", margin=(6, 10, 0, 10)),
            self.issues,
            sizing_mode="stretch_width")


def _family(spec):
    try:
        return registry.family(spec.physics.family)
    except KeyError:
        return None


def _describe(what: str, drawn, moved, gone) -> str | None:
    parts = []
    if drawn:
        parts.append(f"drew {what} {', '.join(drawn)}")
    if moved:
        parts.append(f"moved {what} {', '.join(moved)}")
    if gone:
        parts.append(f"deleted {what} {', '.join(gone)}")
    return "; ".join(parts) if parts else None


def _row(*objs):
    return pn.FlexBox(*objs, flex_wrap="wrap", gap="8px 12px", align_items="center",
                      sizing_mode="stretch_width")


__all__ = ["GeometryEditor", "LAYERS", "TOOLS", "material_colours"]
