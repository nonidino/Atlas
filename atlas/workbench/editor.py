"""The Model tab: a case's shape, materials, boundaries, windows and physics on one
canvas, with a layer rail on the left and an inspector on the right.

The owner's request of 2026-09-29, once drawn shapes ran in every family: "go
through the entire UI to make it much more simple and intuitive".  Planned first
as seven screens on a Claude Design canvas, then built here:

* **the rail**: the layers -- Shape, Materials, Boundaries, Windows, Physics, and
  Rotors or Circuit when the family reads them -- and the chosen layer's items,
  each a click from being selected;
* **the canvas**: every layer drawn, the chosen one emphasized, and three tools --
  **Select**, **Draw**, **Pan** -- with two one-click shapes, **Rectangle** (click
  two opposite corners) and **Circle** (click its centre, then a point on it);
* **the inspector** (`inspector.py`): the selection's settings and nothing else, or
  with nothing selected the layer's own (for the windows: automatic or your own).

**Selecting a shape shows its handles**, so there is no separate reshape tool:
drag a square to move a vertex, a circle to bend its edge into an arc (back onto
the line to straighten it), the diamond to move the whole shape; double-click a
circle to add a vertex there, a square to make the curve smooth through it (again
for a corner); a click then Backspace on a handle removes it.  A rectangle has a
square at each corner and a diamond.

Every change goes through `Workbench.edit` -- validated, undoable, saved like any
other -- and the canvas is drawn back from the case, so it cannot show a shape the
case does not hold.  The warnings drawn on it are `geometry.analyse_case`, the
computation `spec.check` refuses on.

Bokeh gives each gesture to one tool; in Select the handles' drag tool holds the
drag, and a click still reaches the page as a tap, which is how a click selects.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import panel as pn
from bokeh.events import DoubleTap, Tap
from bokeh.models import (BoxZoomTool, ColumnDataSource, CustomJS, HoverTool, LabelSet,
                          PanTool, PointDrawTool, Range1d, ResetTool, SaveTool,
                          SingleIntervalTicker, WheelZoomTool)
from bokeh.plotting import figure

#: square cells whatever size the canvas is given: the view widens in one direction
#: until a cell is as tall as it is wide.  Scaling the whole plot to an aspect ratio
#: made cells 20% taller than wide, because the axes take fixed pixels (seen at
#: 1,090 x 620).  A change of 0.5% or less is left alone, so the callback that the
#: change itself triggers ends there.
SQUARE_JS = """
const w = p.inner_width, h = p.inner_height;
if (!(w > 0 && h > 0)) { return; }
const sx = (xr.end - xr.start) / w, sy = (yr.end - yr.start) / h;
if (Math.abs(sx - sy) <= 0.005 * Math.max(sx, sy)) { return; }
if (sx > sy) {
    const c = 0.5 * (yr.start + yr.end), half = 0.5 * sx * h;
    yr.setv({start: c - half, end: c + half});
} else {
    const c = 0.5 * (xr.start + xr.end), half = 0.5 * sy * w;
    xr.setv({start: c - half, end: c + half});
}
"""

from . import geometry as geo
from . import registry
from . import shapes
from .spec import (Attachment, Boundary, CaseSpec, Device, Layout, Outline, Region, Window,
                   check, family_boundaries)

if TYPE_CHECKING:                                             # pragma: no cover
    from .app import Workbench

#: layer key -> the rail's name for it (the keys are the case file's own words)
LAYERS = {"domain": "Shape", "regions": "Materials", "boundaries": "Boundaries",
          "windows": "Windows", "physics": "Physics", "devices": "Rotors",
          "attachments": "Circuit"}
#: the tools; Rectangle and Circle are two clicks each
TOOLS = {"select": "Select", "draw": "Draw", "rect": "Rectangle", "circle": "Circle",
         "pan": "Pan"}
#: the names of the tools before 2026-09-29's rebuild, still accepted
TOOL_ALIASES = {"move": "select", "resize": "select", "shape": "draw"}
#: the layers a shape can be drawn on
DRAWABLE = ("domain", "windows", "regions")

ACCENT = "#0e6874"
C_DOMAIN = "#64748b"
C_WINDOW = "#2a9d8f"
C_OVERLAP = "#e9a23b"
C_ROTOR = "#d1495b"
C_BAD = "#c1121f"
C_GAP = "#6b7280"
C_HANDLE = "#0e6874"
C_EDGE = "#c2410c"
C_VOID = "#f1f5f9"
C_PENDING = "#7c3aed"
C_SELECT = "#0e6874"

#: Common materials get a colour that reads as the material; others take the
#: palette in order of first appearance.
MATERIAL_COLOURS = {"steel": "#8d99ae", "copper": "#c8733a", "aluminium": "#b8c4d6",
                    "aluminum": "#b8c4d6", "air": "#bde0fe", "water": "#4895ef",
                    "concrete": "#a8a29e", "wood": "#b08968", "graphite": "#4b5563",
                    "shallow-fast": "#90e0ef", "deep-slow": "#0077b6", "pool": "#023e8a",
                    "chip": "#374151", "nichrome": "#9ca3af", "constantan": "#b45309",
                    "helium": "#fde68a"}
PALETTE = ("#6d597a", "#90be6d", "#577590", "#f4a261", "#e76f51", "#43aa8b", "#f9c74f",
           "#277da1")
BOUNDARY_COLOURS = {"inlet": "#1d4ed8", "freestream": "#0891b2", "outlet": "#7c3aed",
                    "wall": "#334155", "fixed-temperature": "#dc2626", "insulated": "#a16207",
                    "heat-flux": "#ea580c", "fixed-potential": "#16a34a",
                    "no-current": "#6b7280", "electrode": "#b45309",
                    "river-inlet": "#1d4ed8", "river-outlet": "#7c3aed", "bank": "#4d7c0f",
                    "rigid-wall": "#1f2937", "clamped": "#111827", "free": "#9ca3af",
                    "load-x": "#be123c", "load-y": "#be123c", "coolant-inlet": "#1d4ed8",
                    "coolant-outlet": "#7c3aed"}

#: the circuit schematic's colours: batteries, resistors, nodes
C_BATTERY = "#b45309"
C_RESISTOR = "#1d4ed8"
C_NODE = "#111827"


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


def layers_for(spec) -> list[str]:
    """The layers the rail shows: the ones every case has, and a family's own."""
    fam = _family(spec)
    reads = fam.layers if fam is not None else ("regions", "windows", "boundaries")
    out = ["domain"]
    if "regions" in reads or spec.regions:
        out.append("regions")
    out += ["boundaries", "windows", "physics"]
    if "devices" in reads or spec.devices:
        out.append("devices")
    if "attachments" in reads or spec.attachments:
        out.append("attachments")
    return out


class GeometryEditor:
    """The Model tab for one version of the case.

    Rebuilt when the layer, the tool, the domain's size or the view options change;
    otherwise `sync` redraws it from the case in place, keeping the view, and a new
    selection redraws only its handles, the rail's list and the inspector.
    """

    def __init__(self, wb: "Workbench", ranges: tuple[float, float, float, float] | None = None):
        self.wb = wb
        self.state = wb.geo_state
        st = self.state
        st.setdefault("pending", [])
        st.setdefault("draw_edges", "line")
        st.setdefault("draw_as", "outline")
        st.setdefault("selected", None)
        st.setdefault("sub", None)
        st["tool"] = TOOL_ALIASES.get(st.get("tool", "select"), st.get("tool", "select"))
        if st["tool"] not in TOOLS:
            st["tool"] = "select"
        if st.get("layer") not in layers_for(wb.spec):
            st["layer"] = "domain"
        self._syncing = False
        d = wb.spec.domain
        self._domain = (d.nx, d.ny)
        #: the rail's layers when built: a family that reads other layers rebuilds it
        self._layers = layers_for(wb.spec)
        self.layer_col = pn.Column(sizing_mode="stretch_width", margin=0)
        self._auto_select()
        self._build_sources()
        self.fig = self._figure(ranges)
        self.fig_pane = pn.pane.Bokeh(self.fig, sizing_mode="stretch_both")
        #: fixed heights: the hint changes with every click while drawing ("2 points
        #: placed..."), and when it wrapped to another line the canvas resized under
        #: the pointer, so the next click landed somewhere else (seen: 8 clicks made
        #: a 3-vertex outline, 2026-09-30)
        self.hint = pn.pane.HTML(sizing_mode="stretch_width", margin=(0, 8), height=38,
                                 styles={"font-size": "13px", "color": "#334155",
                                         "line-height": "1.4", "overflow": "hidden"})
        self.summary = pn.pane.HTML(sizing_mode="stretch_width", margin=(0, 8), height=34,
                                    styles={"font-size": "11px", "color": "#475569",
                                            "line-height": "1.4", "overflow": "hidden"})
        self.rail_items = pn.Column(sizing_mode="stretch_width", margin=0)
        self.inspector = pn.Column(sizing_mode="stretch_width", margin=0)
        self.sync()

    # ------------------------------------------------------------------
    # selection
    # ------------------------------------------------------------------
    def _auto_select(self) -> None:
        """The shape layer's one outline is selected on arrival, so its handles are
        there to drag; a selection that no longer exists is dropped."""
        st, s = self.state, self.wb.spec
        if not self._exists(st.get("selected")):
            st["selected"], st["sub"] = None, None
        if st["layer"] == "domain" and st.get("selected") is None \
                and s.domain.outline is not None and st["tool"] == "select":
            st["selected"] = "domain:outline"

    def _exists(self, ref: str | None) -> bool:
        if ref is None:
            return False
        s = self.wb.spec
        p = ref.split(":")
        try:
            if p[0] == "domain":
                return (s.domain.outline is not None if p[1] == "outline"
                        else int(p[2]) < len(s.domain.holes))
            if p[0] == "material":
                return ref[len("material:"):] in s.materials or any(
                    r.material == ref[len("material:"):] for r in s.regions)
            coll = {"regions": s.regions, "windows": s.windows, "boundary": s.boundaries,
                    "devices": s.devices, "parts": s.attachments}[p[0]]
            return int(p[1]) < len(coll)
        except (KeyError, IndexError, ValueError):
            return False

    @property
    def selected(self) -> str | None:
        return self.state.get("selected")

    def select(self, ref: str | None, sub: int | None = None) -> None:
        """Select ``ref`` (or nothing): its handles, the rail's highlight and the
        inspector follow."""
        if ref is not None and not self._exists(ref):
            ref = None
        self.state["selected"], self.state["sub"] = ref, sub
        self._sync_handles()
        self._sync_selection()
        self.refresh_panels()

    def selected_item(self):
        """The case object the selection names, or None."""
        ref = self.selected
        if ref is None:
            return None
        s = self.wb.spec
        p = ref.split(":")
        if p[0] == "domain":
            return s.domain.outline if p[1] == "outline" else s.domain.holes[int(p[2])]
        if p[0] == "material":
            return ref[len("material:"):]
        coll = {"regions": s.regions, "windows": s.windows, "boundary": s.boundaries,
                "devices": s.devices, "parts": s.attachments}[p[0]]
        return coll[int(p[1])]

    # ------------------------------------------------------------------
    # data sources
    # ------------------------------------------------------------------
    def _build_sources(self) -> None:
        self.src_reg_draw = ColumnDataSource(dict(xs=[], ys=[], id=[], material=[], colour=[],
                                                  shape=[]))
        self.src_win = ColumnDataSource(dict(x=[], y=[], w=[], h=[], id=[]))
        self.src_win_curve = ColumnDataSource(dict(xs=[], ys=[], id=[], cells=[]))
        self.src_win_gen = ColumnDataSource(dict(xs=[], ys=[]))
        self.src_win_lbl = ColumnDataSource(dict(x=[], y=[], id=[]))
        self.src_handles = ColumnDataSource(dict(x=[], y=[], owner=[], corner=[]))
        self.src_dev = ColumnDataSource(dict(x=[], y=[], id=[]))
        self.src_dev_seg = ColumnDataSource(dict(x=[], y0=[], y1=[], id=[]))
        self.src_overlap = ColumnDataSource(dict(x=[], y=[], w=[], h=[]))
        self.src_overlap_q = ColumnDataSource(dict(left=[], right=[], bottom=[], top=[]))
        self.src_bad = ColumnDataSource(dict(left=[], right=[], bottom=[], top=[]))
        self.src_gap = ColumnDataSource(dict(left=[], right=[], bottom=[], top=[]))
        self.src_cross = ColumnDataSource(dict(x=[], y=[], names=[]))
        self.src_bc = ColumnDataSource(dict(x0=[], y0=[], x1=[], y1=[], kind=[], id=[],
                                            colour=[], lx=[], ly=[], angle=[], text=[]))
        self.src_bc_drawn = ColumnDataSource(dict(xs=[], ys=[], colour=[], id=[], text=[],
                                                  lx=[], ly=[]))
        self.src_att = ColumnDataSource(dict(x0=[], y0=[], x1=[], y1=[], lx=[], ly=[],
                                             text=[], colour=[], id=[]))
        self.src_att_nodes = ColumnDataSource(dict(x=[], y=[], name=[]))
        self.src_dom_cells = ColumnDataSource(dict(left=[], right=[], bottom=[], top=[]))
        self.src_dom_line = ColumnDataSource(dict(xs=[], ys=[]))
        self.src_vtx = ColumnDataSource(dict(x=[], y=[], ref=[], k=[], label=[], what=[]))
        self.src_edge = ColumnDataSource(dict(x=[], y=[], ref=[], k=[], label=[], what=[]))
        self.src_move = ColumnDataSource(dict(x=[], y=[], ref=[], label=[], what=[]))
        self.src_pending = ColumnDataSource(dict(x=[], y=[]))
        #: the selection, drawn over everything: an outline, or a boundary's edge
        self.src_sel = ColumnDataSource(dict(xs=[], ys=[]))
        self.src_sub = ColumnDataSource(dict(xs=[], ys=[]))
        self.src_handles.on_change("data", self._on_handles)
        self.src_dev.on_change("data", self._on_devices)
        self.src_vtx.on_change("data", self._on_vertices)
        self.src_edge.on_change("data", self._on_edges)
        self.src_move.on_change("data", self._on_move)
        self._handle_rows = {"vtx": [], "edge": [], "move": []}

    # ------------------------------------------------------------------
    # the canvas
    # ------------------------------------------------------------------
    def _circuit_shown(self) -> bool:
        return self.state["layer"] == "attachments" or bool(self.wb.spec.attachments)

    def _circuit_y(self) -> float:
        """Where the circuit's free nodes sit: a row below the domain."""
        d = self.wb.spec.domain
        pad = max(8, int(0.05 * max(d.nx, d.ny)))
        return -(pad + 0.18 * d.ny)

    def default_ranges(self) -> tuple[float, float, float, float]:
        d = self.wb.spec.domain
        pad = max(8, int(0.05 * max(d.nx, d.ny)))
        circuit = self._circuit_shown()
        bottom = self._circuit_y() - pad if circuit else -pad
        side = pad + (3.0 * max(3.0, 0.02 * max(d.nx, d.ny)) if circuit else 0.0)
        return (-side, d.nx + side, bottom, d.ny + pad)

    def _figure(self, ranges):
        d = self.wb.spec.domain
        if ranges is None:
            ranges = self.default_ranges()
        x0, x1, y0, y1 = ranges
        aspect = (x1 - x0) / max(y1 - y0, 1e-9)
        p = figure(x_range=Range1d(x0, x1), y_range=Range1d(y0, y1),
                   width=1000, height=max(240, int(1000 / aspect)),
                   # all the room the window leaves, cells kept square by SQUARE_JS:
                   # scaled to the width alone, the page ran past a laptop's screen
                   # (the owner's, 2026-09-30)
                   sizing_mode="stretch_both",
                   tools=[], toolbar_location=None, output_backend="canvas",
                   background_fill_color="#e9edf1", border_fill_color="#e9edf1")
        square = CustomJS(args=dict(p=p, xr=p.x_range, yr=p.y_range), code=SQUARE_JS)
        for attr in ("inner_width", "inner_height"):
            p.js_on_change(attr, square)
        for r in (p.x_range, p.y_range):                  # Fit, a box zoom
            r.js_on_change("start", square)
            r.js_on_change("end", square)
        p.xaxis.axis_label, p.yaxis.axis_label = "x (cells)", "y (cells)"
        for ax in (p.xaxis, p.yaxis):
            ax.axis_label_text_font_size = "11px"
            ax.major_label_text_font_size = "10px"
            ax.axis_line_color = "#cbd5e1"
        for g in (p.xgrid, p.ygrid):
            g.visible = self.wb.view_opts["grid"]
            g.ticker = SingleIntervalTicker(interval=max(self.state["snap"], 8))
            g.grid_line_alpha = 0.35
        layer, tool = self.state["layer"], self.state["tool"]
        drawn_domain = d.outline is not None or bool(d.holes)
        p.rect(x=d.nx / 2, y=d.ny / 2, width=d.nx, height=d.ny, fill_color="#ffffff",
               fill_alpha=1.0, line_color=C_DOMAIN, line_width=1 if drawn_domain else 2,
               line_dash="dotted" if drawn_domain else "solid", level="underlay")

        # regions: filled shapes in stacking order
        strong = layer == "regions"
        p.multi_polygons("xs", "ys", source=self.src_reg_draw, fill_color="colour",
                         fill_alpha=0.5 if strong else 0.32, line_color="colour",
                         line_alpha=0.9 if strong else 0.5, line_width=1)
        # outside a drawn domain: pale and dotted, over the regions
        p.quad("left", "right", "bottom", "top", source=self.src_dom_cells, fill_color=C_VOID,
               fill_alpha=1.0, line_alpha=0, hatch_pattern=".", hatch_color="#94a3b8",
               hatch_alpha=0.7, hatch_scale=8)

        # windows and their warnings
        wins = layer == "windows"
        if wins or layer in ("domain", "physics"):
            p.quad("left", "right", "bottom", "top", source=self.src_gap, fill_color=C_GAP,
                   fill_alpha=0.15, hatch_pattern="/", hatch_color=C_GAP, hatch_alpha=0.6,
                   line_alpha=0)
            p.quad("left", "right", "bottom", "top", source=self.src_bad, fill_color=C_BAD,
                   fill_alpha=0.18, hatch_pattern="x", hatch_color=C_BAD, hatch_alpha=0.7,
                   line_alpha=0)
        if self.wb.view_opts["overlaps"] and wins:
            p.rect("x", "y", "w", "h", source=self.src_overlap, fill_color=C_OVERLAP,
                   fill_alpha=0.3, line_alpha=0)
            p.quad("left", "right", "bottom", "top", source=self.src_overlap_q,
                   fill_color=C_OVERLAP, fill_alpha=0.3, line_alpha=0)
        w_alpha, w_width = (1.0, 2.0) if wins else (0.35, 1.0)
        p.multi_polygons("xs", "ys", source=self.src_win_curve, fill_color=C_WINDOW,
                         fill_alpha=0.06 if wins else 0.0, line_color=C_WINDOW,
                         line_width=w_width, line_alpha=w_alpha)
        p.multi_line("xs", "ys", source=self.src_win_gen, line_color=C_WINDOW,
                     line_width=w_width + 0.4, line_alpha=w_alpha)
        p.rect("x", "y", "w", "h", source=self.src_win, fill_color=C_WINDOW,
               fill_alpha=0.05 if wins else 0.0, line_color=C_WINDOW, line_width=w_width,
               line_alpha=w_alpha)
        if self.wb.view_opts["labels"] and wins:
            for src in (self.src_win_lbl, self.src_win):
                p.add_layout(LabelSet(x="x", y="y", text="id", source=src,
                                      text_align="center", text_baseline="middle",
                                      text_font_size="11px", text_font_style="bold",
                                      text_color="#1f7f74"))
        if wins:
            cross = p.scatter("x", "y", source=self.src_cross, marker="diamond", size=9,
                              fill_color="#ffffff", line_color="#374151", line_width=1.5)
            p.add_tools(HoverTool(renderers=[cross], tooltips=[("cross-point", "@names")]))

        # the domain's drawn outline and holes, exactly as drawn
        if drawn_domain:
            p.multi_line("xs", "ys", source=self.src_dom_line, line_color="#111827",
                         line_width=2.5 if layer == "domain" else 1.6)

        # boundaries: each edge in its condition's colour
        bnd = layer == "boundaries"
        bw = 7 if bnd else 3
        p.multi_line("xs", "ys", source=self.src_bc_drawn, line_color="colour",
                     line_width=bw, line_alpha=0.9 if bnd else 0.55)
        p.segment("x0", "y0", "x1", "y1", source=self.src_bc, line_color="colour",
                  line_width=bw, line_cap="butt", line_alpha=0.9 if bnd else 0.55)
        if bnd:
            for src, angle in ((self.src_bc_drawn, None), (self.src_bc, "angle")):
                kw = dict(angle=angle) if angle else {}
                p.add_layout(LabelSet(x="lx", y="ly", text="text", source=src,
                                      text_align="center", text_baseline="middle",
                                      text_font_size="11px", text_color="colour",
                                      background_fill_color="#ffffff",
                                      background_fill_alpha=0.85, **kw))

        # the circuit: a schematic of nodes and parts
        if self._circuit_shown():
            circ = layer == "attachments"
            p.segment("x0", "y0", "x1", "y1", source=self.src_att, line_color="colour",
                      line_width=3 if circ else 2, line_alpha=0.9 if circ else 0.5)
            p.add_layout(LabelSet(x="lx", y="ly", text="text", source=self.src_att,
                                  text_align="center", text_baseline="middle",
                                  text_font_size="11px", text_color="colour",
                                  background_fill_color="#ffffff",
                                  background_fill_alpha=0.85))
            p.scatter("x", "y", source=self.src_att_nodes, size=8, fill_color="#ffffff",
                      line_color=C_NODE, line_width=2)
            p.add_layout(LabelSet(x="x", y="y", text="name", source=self.src_att_nodes,
                                  x_offset=6, y_offset=6, text_font_size="11px",
                                  text_color=C_NODE))

        # rotors
        p.segment("x", "y0", "x", "y1", source=self.src_dev_seg, line_color=C_ROTOR,
                  line_width=4)
        dev = p.scatter("x", "y", source=self.src_dev, size=10, fill_color="#ffffff",
                        line_color=C_ROTOR, line_width=2, selection_fill_color=C_ROTOR,
                        nonselection_fill_alpha=1.0)
        if self.wb.view_opts["labels"]:
            p.add_layout(LabelSet(x="x", y="y1", text="id", source=self.src_dev_seg,
                                  x_offset=4, y_offset=2, text_font_size="11px",
                                  text_color=C_ROTOR))

        # the selection, over everything
        p.multi_line("xs", "ys", source=self.src_sel, line_color=C_SELECT, line_width=3,
                     line_dash="dashed")
        p.multi_line("xs", "ys", source=self.src_sub, line_color=C_EDGE, line_width=5,
                     line_alpha=0.9)

        # the selected shape's handles: squares on vertices (or a rectangle's
        # corners), circles on edges, a diamond to move it
        editing = tool == "select"
        hnd = p.scatter("x", "y", source=self.src_handles, marker="square", size=10,
                        fill_color="#ffffff", line_color=C_HANDLE, line_width=2,
                        visible=editing, nonselection_fill_alpha=1.0)
        vtx = p.scatter("x", "y", source=self.src_vtx, marker="square", size=10,
                        fill_color="#ffffff", line_color=C_HANDLE, line_width=2,
                        visible=editing, nonselection_fill_alpha=1.0,
                        selection_fill_color=C_HANDLE)
        edg = p.scatter("x", "y", source=self.src_edge, marker="circle", size=10,
                        fill_color="#ffffff", line_color=C_EDGE, line_width=2,
                        visible=editing, nonselection_fill_alpha=1.0,
                        selection_fill_color=C_EDGE)
        mv = p.scatter("x", "y", source=self.src_move, marker="diamond", size=16,
                       fill_color=C_HANDLE, line_color="#ffffff", line_width=1.5,
                       visible=editing, nonselection_fill_alpha=1.0)
        p.add_tools(HoverTool(renderers=[vtx, edg, mv], tooltips=[("", "@label"),
                                                                 ("", "@what")]))

        # the shape being drawn: its points so far, and the edge that will close it
        drawing = tool in ("draw", "rect", "circle")
        p.line("x", "y", source=self.src_pending, line_color=C_PENDING, line_width=2,
               line_dash="dashed", visible=drawing)
        p.scatter("x", "y", source=self.src_pending, size=8, fill_color=C_PENDING,
                  line_color="#ffffff", visible=drawing)

        pan, wheel = PanTool(), WheelZoomTool()
        p.add_tools(pan, wheel, BoxZoomTool(match_aspect=True), ResetTool(), SaveTool())
        p.toolbar.active_scroll = wheel
        p.on_event(Tap, self._on_tap)
        p.on_event(DoubleTap, self._on_double_tap)
        if editing:
            renderers = [hnd, vtx, edg, mv] + ([dev] if layer == "devices" else [])
            edit = PointDrawTool(renderers=renderers, empty_value="", add=False, drag=True,
                                 description="Drag a handle")
            p.add_tools(edit)
            p.toolbar.active_multi = edit
            p.toolbar.active_drag = None
        elif tool == "pan":
            p.toolbar.active_drag = pan
        else:
            p.toolbar.active_drag = None
        p.toolbar.logo = None
        return p

    def ranges(self) -> tuple[float, float, float, float]:
        xr, yr = self.fig.x_range, self.fig.y_range
        return float(xr.start), float(xr.end), float(yr.start), float(yr.end)

    def fit(self) -> None:
        x0, x1, y0, y1 = self.default_ranges()
        self.fig.x_range.start, self.fig.x_range.end = x0, x1
        self.fig.y_range.start, self.fig.y_range.end = y0, y1

    # ------------------------------------------------------------------
    # case -> canvas
    # ------------------------------------------------------------------
    def _set(self, src: ColumnDataSource, data: dict) -> None:
        self._syncing = True
        try:
            src.data = data
            src.selected.indices = []
        finally:
            self._syncing = False

    def sync(self) -> None:
        """Redraw every layer, warning, handle, list and the inspector from the case."""
        s, d = self.wb.spec, self.wb.spec.domain
        if not self._exists(self.selected):
            self.state["selected"], self.state["sub"] = None, None
        ws = [w for w in s.windows if w.shape == "rect"]
        self._set(self.src_win, dict(x=[w.x0 + w.nx / 2 for w in ws],
                                     y=[w.y0 + w.ny / 2 for w in ws],
                                     w=[w.nx for w in ws], h=[w.ny for w in ws],
                                     id=[w.id for w in ws]))
        act = geo.domain_mask(d)
        cw = [w for w in s.windows if w.shape == "curve"]
        xs, ys, cells, lx, ly = [], [], [], [], []
        for w in cw:
            rings = [w.outline.ring()] + [h.ring() for h in w.holes]
            xs.append([[r[:, 0].tolist() for r in rings]])
            ys.append([[r[:, 1].tolist() for r in rings]])
            m = geo.window_mask(w, d.nx, d.ny) & act
            cells.append(int(m.sum()))
            x, y = _inside_point(m, w.outline)
            lx.append(x)
            ly.append(y)
        self._set(self.src_win_curve, dict(xs=xs, ys=ys, id=[w.id for w in cw], cells=cells))
        gw = [w for w in s.windows if w.shape == "cells"]
        gx, gy = [], []
        for w in gw:
            m = geo.window_mask(w, d.nx, d.ny) & act
            for line in geo.inner_boundary(m, act):
                gx.append(line[:, 0].tolist())
                gy.append(line[:, 1].tolist())
            x, y = _inside_point(m, None)
            lx.append(x)
            ly.append(y)
        self._set(self.src_win_gen, dict(xs=gx, ys=gy))
        self._set(self.src_win_lbl, dict(x=lx, y=ly, id=[w.id for w in cw] + [w.id for w in gw]))
        if d.outline is not None or d.holes:
            boxes = geo.mask_to_boxes(~act)
            self._set(self.src_dom_cells, dict(left=[b[0] for b in boxes],
                                               right=[b[0] + b[2] for b in boxes],
                                               bottom=[b[1] for b in boxes],
                                               top=[b[1] + b[3] for b in boxes]))
            lines = ([d.outline.ring()] if d.outline is not None else []) + \
                [h.ring() for h in d.holes]
            self._set(self.src_dom_line, dict(
                xs=[np.append(r[:, 0], r[0, 0]).tolist() for r in lines],
                ys=[np.append(r[:, 1], r[0, 1]).tolist() for r in lines]))
        else:
            self._set(self.src_dom_cells, dict(left=[], right=[], bottom=[], top=[]))
            self._set(self.src_dom_line, dict(xs=[], ys=[]))
        self._sync_pending()
        cmap = material_colours(s.regions)
        xs, ys = [], []
        for r in s.regions:
            rings = _region_rings(r)
            xs.append([[[p[0] for p in ring] for ring in rings]])
            ys.append([[[p[1] for p in ring] for ring in rings]])
        self._set(self.src_reg_draw, dict(xs=xs, ys=ys, id=[r.id for r in s.regions],
                                          material=[r.material for r in s.regions],
                                          colour=[cmap[r.material] for r in s.regions],
                                          shape=[r.shape for r in s.regions]))
        self._set(self.src_dev, dict(x=[v.x / d.dx for v in s.devices],
                                     y=[v.y / d.dx for v in s.devices],
                                     id=[v.id for v in s.devices]))
        self._set(self.src_dev_seg, dict(x=[v.x / d.dx for v in s.devices],
                                         y0=[(v.y - v.diameter / 2) / d.dx for v in s.devices],
                                         y1=[(v.y + v.diameter / 2) / d.dx for v in s.devices],
                                         id=[v.id for v in s.devices]))
        self._sync_warnings()
        self._sync_boundaries()
        self._sync_attachments()
        self._sync_handles()
        self._sync_selection()
        self.refresh_panels()

    def refresh_panels(self) -> None:
        from . import inspector
        self.layer_col.objects = inspector.layer_buttons(self)
        self.rail_items.objects = inspector.rail_items(self)
        self.inspector.objects = inspector.panel(self)
        self.hint.object = self._hint_html()

    # -- the handles -----------------------------------------------------------
    def curves(self) -> list[tuple[str, Outline]]:
        """The selected drawn shape (and a window's holes), with their references."""
        s, ref = self.wb.spec, self.selected
        if ref is None:
            return []
        p = ref.split(":")
        if p[0] == "domain":
            return [(ref, self.selected_item())]
        if p[0] == "windows":
            w = s.windows[int(p[1])]
            if w.shape == "curve":
                return [(ref, w.outline)] + [(f"{ref}:hole:{h}", o)
                                             for h, o in enumerate(w.holes)]
        if p[0] == "regions":
            r = s.regions[int(p[1])]
            if r.shape == "curve":
                return [(ref, r.outline)]
        return []

    def _boxes(self) -> list[tuple[int, geo.Box]]:
        """The selected rectangle, for its corner squares: (index, box)."""
        s, ref = self.wb.spec, self.selected
        if ref is None:
            return []
        p = ref.split(":")
        if p[0] == "windows" and s.windows[int(p[1])].shape == "rect":
            w = s.windows[int(p[1])]
            return [(int(p[1]), (w.x0, w.y0, w.nx, w.ny))]
        if p[0] == "regions" and s.regions[int(p[1])].shape == "rect":
            r = s.regions[int(p[1])]
            return [(int(p[1]), (r.x0, r.y0, r.nx, r.ny))]
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
        v = {"x": [], "y": [], "ref": [], "k": [], "label": [], "what": []}
        e = {"x": [], "y": [], "ref": [], "k": [], "label": [], "what": []}
        m = {"x": [], "y": [], "ref": [], "label": [], "what": []}
        for ref, o in self.curves():
            ks, bs = o.kinds(), o.bulges()
            name = shape_label(self.wb.spec, ref)
            for k, (x, y) in enumerate(o.points):
                v["x"].append(float(x))
                v["y"].append(float(y))
                v["ref"].append(ref)
                v["k"].append(k)
                v["label"].append(f"{name}, vertex {k}")
                v["what"].append("drag to move; double-click: smooth or corner")
                mx, my = shapes.edge_midpoint(o.points, ks, bs, k)
                e["x"].append(float(mx))
                e["y"].append(float(my))
                e["ref"].append(ref)
                e["k"].append(k)
                e["label"].append(f"{name}, edge {k} ({ks[k]})")
                e["what"].append("drag to bend; double-click: add a vertex")
            if ":hole:" in ref and not ref.startswith("domain"):
                continue                               # a window's hole moves with it
            c = np.mean(np.asarray(o.points, dtype=float), axis=0)
            m["x"].append(float(c[0]))
            m["y"].append(float(c[1]))
            m["ref"].append(ref)
            m["label"].append(name)
            m["what"].append("drag to move the whole shape")
        for k, (x0, y0, w, h) in self._boxes():
            m["x"].append(x0 + w / 2.0)
            m["y"].append(y0 + h / 2.0)
            m["ref"].append(self.selected)
            m["label"].append(shape_label(self.wb.spec, self.selected))
            m["what"].append("drag to move it")
        #: what the handles showed, so a drag can be told from a Backspace
        self._handle_rows = {"vtx": [(r, k, x, y) for x, y, r, k in
                                     zip(v["x"], v["y"], v["ref"], v["k"])],
                             "edge": [(r, k, x, y) for x, y, r, k in
                                      zip(e["x"], e["y"], e["ref"], e["k"])],
                             "move": [(r, x, y) for x, y, r in zip(m["x"], m["y"], m["ref"])]}
        self._set(self.src_vtx, v)
        self._set(self.src_edge, e)
        self._set(self.src_move, m)

    def _sync_selection(self) -> None:
        """The selection's outline (dashed), and a selected edge (thick)."""
        s, d, ref = self.wb.spec, self.wb.spec.domain, self.selected
        xs, ys, sx, sy = [], [], [], []
        if ref is not None:
            p = ref.split(":")
            if p[0] == "regions":
                for ring in _region_rings(s.regions[int(p[1])]):
                    xs.append([q[0] for q in ring] + [ring[0][0]])
                    ys.append([q[1] for q in ring] + [ring[0][1]])
            elif p[0] == "windows":
                w = s.windows[int(p[1])]
                if w.shape == "rect":
                    ring = geo.corners((w.x0, w.y0, w.nx, w.ny))
                    xs.append([q[0] for q in ring] + [ring[0][0]])
                    ys.append([q[1] for q in ring] + [ring[0][1]])
                elif w.shape == "curve":
                    r = w.outline.ring()
                    xs.append(np.append(r[:, 0], r[0, 0]).tolist())
                    ys.append(np.append(r[:, 1], r[0, 1]).tolist())
                else:
                    m = geo.window_mask(w, d.nx, d.ny) & geo.domain_mask(d)
                    for line in geo.contour_lines(m.astype(float), 0.5):
                        xs.append(line[:, 0].tolist())
                        ys.append(line[:, 1].tolist())
            elif p[0] == "boundary":
                # thick and orange: a dashed line vanished under the edge's own stroke
                # (seen in the page)
                for poly in boundary_polylines(s, s.boundaries[int(p[1])]):
                    if not s.boundaries[int(p[1])].drawn and d.outline is None:
                        off = max(3.0, 0.02 * max(d.nx, d.ny))
                        shift = {"left": (-off, 0), "right": (off, 0), "bottom": (0, -off),
                                 "top": (0, off)}[s.boundaries[int(p[1])].edge]
                        poly = poly + np.asarray(shift)
                    sx.append(poly[:, 0].tolist())
                    sy.append(poly[:, 1].tolist())
            sub = self.state.get("sub")
            if sub is not None and p[0] in ("domain", "windows", "regions"):
                o = get_outline(s, ref) if p[0] == "domain" or len(p) == 2 else None
                if o is not None and 0 <= sub < len(o.points):
                    poly = shapes.edges_sampled(o.points, o.edges, o.bulge)[sub]
                    sx.append(poly[:, 0].tolist())
                    sy.append(poly[:, 1].tolist())
        self._set(self.src_sel, dict(xs=xs, ys=ys))
        self._set(self.src_sub, dict(xs=sx, ys=sy))

    def _sync_pending(self) -> None:
        pend = self.state.get("pending") or []
        tool = self.state["tool"]
        if tool == "rect" and len(pend) == 1:
            closing = pend
        elif tool == "circle" and len(pend) == 1:
            closing = pend
        else:
            closing = pend + [pend[0]] if len(pend) > 2 else pend
        self._set(self.src_pending, dict(x=[p[0] for p in closing],
                                         y=[p[1] for p in closing]))

    # -- warnings, boundaries, the circuit --------------------------------------
    def _sync_warnings(self) -> None:
        s, d = self.wb.spec, self.wb.spec.domain
        pieces = s.coupling.style in ("C", "D", "split", "M")
        plain = geo.is_plain(s)
        act = geo.domain_mask(d)
        ov_cells: list[geo.Box] = []
        n_thin = 0
        if s.windows and pieces:
            count = np.zeros((d.ny, d.nx), dtype=np.int32)
            if plain:
                for w in s.windows:
                    count[w.y0:w.y0 + w.ny, w.x0:w.x0 + w.nx] += 1
            else:
                for _wid, m in geo.window_masks(s):
                    count += m
            empty = (count == 0) & act
            bad, gap = geo.mask_to_boxes(count > 1), geo.mask_to_boxes(empty)
            ov, cps = [], []
            n_bad, n_gap = int((count > 1).sum()), int(empty.sum())
        elif s.windows:
            an = geo.analyse_case(s)
            bad, gap = geo.mask_to_boxes(an.ramp_only), geo.mask_to_boxes(an.uncovered)
            ov, cps = an.overlaps, an.cross_points
            if not plain:
                ov_cells = geo.mask_to_boxes(an.overlap_mask)
                ov = []
            n_bad, n_gap = int(an.ramp_only.sum()), int(an.uncovered.sum())
            n_thin = len(an.thin_pairs)
        else:
            bad, gap, ov, cps = [], geo.mask_to_boxes(act), [], []
            n_bad, n_gap = 0, int(act.sum())
        quad = lambda bs: dict(left=[b[0] for b in bs], right=[b[0] + b[2] for b in bs],  # noqa: E731
                               bottom=[b[1] for b in bs], top=[b[1] + b[3] for b in bs])
        self._set(self.src_bad, quad(bad))
        self._set(self.src_gap, quad(gap))
        self._set(self.src_overlap_q, quad(ov_cells))
        self._set(self.src_overlap, dict(x=[b[0] + b[2] / 2 for _a, _b, b in ov],
                                         y=[b[1] + b[3] / 2 for _a, _b, b in ov],
                                         w=[b[2] for _a, _b, b in ov],
                                         h=[b[3] for _a, _b, b in ov]))
        self._set(self.src_cross, dict(x=[b[0] + b[2] / 2 for _n, b in cps],
                                       y=[b[1] + b[3] / 2 for _n, b in cps],
                                       names=[", ".join(n) for n, _b in cps]))
        self.windows_ok = not (n_bad or n_gap)
        ramp = s.coupling.ramp_cells
        parts = []
        if n_bad and pieces:
            parts.append(f"<span style='color:{C_BAD}'><b>{n_bad:,}</b> cells in more than "
                         f"one window</span>")
        elif n_bad:
            parts.append(f"<span style='color:{C_BAD}'><b>{n_bad:,}</b> cells with no window "
                         f"at full weight</span> (overlap by at least {2 * ramp})")
        if n_gap:
            parts.append(f"<span style='color:{C_GAP}'><b>{n_gap:,}</b> cells in no "
                         f"window</span>")
        if n_thin:
            parts.append(f"<b>{n_thin}</b> thin seam{'s' * (n_thin > 1)}")
        cells = int(act.sum())
        what = (f"{cells:,} cells in the domain" if (d.outline is not None or d.holes)
                else f"{d.nx} &times; {d.ny} cells")
        lead = [what, f"{len(s.windows)} window{'s' * (len(s.windows) != 1)}"]
        if cps and not pieces:
            lead.append(f"{len(cps)} cross-point{'s' * (len(cps) != 1)}")
        self.summary.object = " &middot; ".join(lead + parts)

    def _sync_boundaries(self) -> None:
        s, d = self.wb.spec, self.wb.spec.domain
        off = max(3.0, 0.02 * max(d.nx, d.ny))
        rows = {k: [] for k in ("x0", "y0", "x1", "y1", "kind", "id", "colour", "lx", "ly",
                                "angle", "text")}
        drawn = (dict(geo.drawn_edges(d, step=0.5)) if (d.outline is not None or d.holes)
                 else {})
        dr = {k: [] for k in ("xs", "ys", "colour", "id", "text", "lx", "ly")}
        for b in s.boundaries:
            if not b.drawn:
                continue
            poly = drawn.get(b.edge)
            if poly is None:
                continue
            val = "" if b.value is None else f" = {b.value:g}"
            dr["xs"].append(poly[:, 0].tolist())
            dr["ys"].append(poly[:, 1].tolist())
            dr["colour"].append(BOUNDARY_COLOURS.get(b.kind, "#111827"))
            dr["id"].append(b.id)
            dr["text"].append(f"{b.kind}{val}")
            mid = poly[len(poly) // 2]
            dr["lx"].append(float(mid[0]))
            dr["ly"].append(float(mid[1]))
        # a drawn edge with no condition: dashed red, so it is found
        have = {b.edge for b in s.boundaries}
        for name, poly in drawn.items():
            if name in have:
                continue
            dr["xs"].append(poly[:, 0].tolist())
            dr["ys"].append(poly[:, 1].tolist())
            dr["colour"].append(C_BAD)
            dr["id"].append("")
            dr["text"].append("no condition")
            mid = poly[len(poly) // 2]
            dr["lx"].append(float(mid[0]))
            dr["ly"].append(float(mid[1]))
        self._set(self.src_bc_drawn, dr)
        for b in s.boundaries:
            if b.drawn or d.outline is not None:
                continue
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
            rows["lx"].append((x0 + x1) / 2 - 2.2 * dx)
            rows["ly"].append((y0 + y1) / 2 - 2.2 * dy)
            rows["angle"].append(math.pi / 2 if b.edge in ("left", "right") else 0.0)
            val = "" if b.value is None else f" = {b.value:g}"
            rows["text"].append(f"{b.kind}{val}")
        self._set(self.src_bc, rows)

    def circuit_nodes(self) -> dict[str, tuple[float, float]]:
        """Each circuit node's place on the canvas: an electrode just outside its
        edge (a grid edge's segment, or a drawn edge's middle, pushed out along its
        normal), any other node in a row below the domain."""
        s, d = self.wb.spec, self.wb.spec.domain
        off = max(3.0, 0.02 * max(d.nx, d.ny))
        pos: dict[str, tuple[float, float]] = {}
        drawn = dict(geo.drawn_edges(d, step=0.5)) if (d.outline is not None or d.holes) \
            else {}
        cx, cy = d.nx / 2.0, d.ny / 2.0
        for b in s.boundaries:
            if b.kind != "electrode":
                continue
            if b.drawn:
                poly = drawn.get(b.edge)
                if poly is None:
                    continue
                mid = poly[len(poly) // 2]
                a, z = poly[max(len(poly) // 2 - 1, 0)], poly[min(len(poly) // 2 + 1,
                                                                  len(poly) - 1)]
                t = np.asarray(z) - np.asarray(a)
                nrm = np.array([t[1], -t[0]]) / max(float(np.hypot(*t)), 1e-12)
                if np.dot(nrm, np.asarray(mid) - np.array([cx, cy])) < 0:
                    nrm = -nrm
                pos[b.id] = (float(mid[0] + 2.5 * off * nrm[0]),
                             float(mid[1] + 2.5 * off * nrm[1]))
                continue
            n = geo.edge_length(b.edge, d.nx, d.ny)
            x0, y0, x1, y1 = geo.edge_segment_xy(b.edge, b.start, n if b.stop is None
                                                 else b.stop, d.nx, d.ny)
            dx, dy = {"left": (-1, 0), "right": (1, 0), "bottom": (0, -1),
                      "top": (0, 1)}[b.edge]
            pos[b.id] = ((x0 + x1) / 2 + 2.5 * off * dx, (y0 + y1) / 2 + 2.5 * off * dy)
        free = []
        for a in s.attachments:
            for n in (a.a, a.b):
                if n not in pos and n not in free:
                    free.append(n)
        for k, n in enumerate(free):
            pos[n] = (d.nx * (k + 1) / (len(free) + 1), self._circuit_y())
        return pos

    def _sync_attachments(self) -> None:
        s = self.wb.spec
        pos = self.circuit_nodes()
        rows = {k: [] for k in ("x0", "y0", "x1", "y1", "lx", "ly", "text", "colour", "id")}
        seen: dict[tuple[str, str], int] = {}
        for a in s.attachments:
            if a.a not in pos or a.b not in pos:
                continue
            (xa, ya), (xb, yb) = pos[a.a], pos[a.b]
            key = tuple(sorted((a.a, a.b)))
            k = seen.get(key, 0)
            seen[key] = k + 1
            t = 0.5 + 0.18 * k * (-1) ** k
            rows["x0"].append(xa)
            rows["y0"].append(ya)
            rows["x1"].append(xb)
            rows["y1"].append(yb)
            rows["lx"].append(xa + t * (xb - xa))
            rows["ly"].append(ya + t * (yb - ya))
            if a.kind == "battery":
                inner = f", {a.internal:g} ohm inside" if a.internal else ""
                rows["text"].append(f"{a.id}: {a.value:g} V{inner} (+ at {a.a})")
                rows["colour"].append(C_BATTERY)
            else:
                rows["text"].append(f"{a.id}: {a.value:g} ohm")
                rows["colour"].append(C_RESISTOR)
            rows["id"].append(a.id)
        self._set(self.src_att, rows)
        used = {n for a in s.attachments for n in (a.a, a.b)}
        names = [n for n in pos if n in used or n in {b.id for b in s.boundaries
                                                       if b.kind == "electrode"}]
        self._set(self.src_att_nodes, dict(x=[pos[n][0] for n in names],
                                           y=[pos[n][1] for n in names], name=names))

    # ------------------------------------------------------------------
    # a click on the canvas
    # ------------------------------------------------------------------
    def _snap(self) -> int:
        return int(self.state["snap"])

    def _snap_pt(self, x: float, y: float) -> tuple[float, float]:
        d = self.wb.spec.domain
        step = self._snap()
        return (float(geo.snap_value(x, step, d.nx)), float(geo.snap_value(y, step, d.ny)))

    def _tolerance(self) -> float:
        x0, x1, _y0, _y1 = self.ranges()
        return max(1.5, 0.02 * (x1 - x0))

    def _near_handle(self, x: float, y: float) -> tuple[str, tuple] | None:
        """The selected shape's handle under a click, if any: ("vtx" | "edge" |
        "move" | "corner", its row)."""
        tol = self._tolerance()
        best, dist = None, tol
        for kind in ("vtx", "edge", "move"):
            for row in self._handle_rows.get(kind, []):
                px, py = row[-2], row[-1]
                dd = math.hypot(px - x, py - y)
                if dd <= dist:
                    best, dist = (kind, row), dd
        for k, box in self._boxes():
            for c, (px, py) in enumerate(geo.corners(box)):
                dd = math.hypot(px - x, py - y)
                if dd <= dist:
                    best, dist = ("corner", (k, c)), dd
        return best

    def _on_tap(self, ev) -> None:
        st = self.state
        tool, layer = st["tool"], st["layer"]
        x, y = float(ev.x), float(ev.y)
        if tool == "draw" and layer in DRAWABLE:
            p = self._snap_pt(x, y)
            pend = [tuple(q) for q in st.get("pending") or []]
            if pend and pend[-1] == p:
                return
            pend.append(p)
            st["pending"] = pend
            self._sync_pending()
            self.hint.object = self._hint_html()
        elif tool == "draw" and layer == "devices":
            self.add_rotor(x, y)
        elif tool in ("rect", "circle") and layer in DRAWABLE:
            p = self._snap_pt(x, y) if tool == "rect" else (x, y)
            pend = [tuple(q) for q in st.get("pending") or []]
            if not pend:
                st["pending"] = [p]
                self._sync_pending()
                self.hint.object = self._hint_html()
                return
            st["pending"] = []
            self._sync_pending()
            if tool == "rect":
                self.add_rectangle(pend[0], p)
            else:
                c = self._snap_pt(*pend[0])
                self.add_circle(c, math.hypot(p[0] - c[0], p[1] - c[1]))
        elif tool == "select":
            hit = self._near_handle(x, y)
            if hit is not None:
                kind, row = hit
                if kind == "edge":                     # a click on an edge's circle
                    self.select(self.selected, sub=int(row[1]))
                return
            self.select(self.hit(x, y))

    def _on_double_tap(self, ev) -> None:
        st = self.state
        if st["tool"] == "draw" and st["layer"] in DRAWABLE:
            self.close_shape()
        elif st["tool"] == "select":
            self.double_click_handle(float(ev.x), float(ev.y))

    def hit(self, x: float, y: float) -> str | None:
        """What a click at ``(x, y)`` selects on the active layer, or None."""
        s, d = self.wb.spec, self.wb.spec.domain
        layer = self.state["layer"]
        i, j = int(math.floor(x)), int(math.floor(y))
        on_grid = 0 <= i < d.nx and 0 <= j < d.ny
        tol = self._tolerance()
        pt = np.array([[x, y]])
        if layer == "domain":
            for h, o in enumerate(d.holes):
                if shapes.contains(o.ring(), pt)[0]:
                    return f"domain:hole:{h}"
            if d.outline is not None:
                ring = d.outline.ring()
                near = float(np.min(np.hypot(ring[:, 0] - x, ring[:, 1] - y))) <= tol
                if near or shapes.contains(ring, pt)[0]:
                    return "domain:outline"
            return None
        if layer == "regions":
            for k in range(len(s.regions) - 1, -1, -1):
                r = s.regions[k]
                if r.shape == "curve":
                    if shapes.contains(r.outline.ring(), pt)[0]:
                        return f"regions:{k}"
                elif on_grid and geo.region_mask(r, d.nx, d.ny)[j, i]:
                    return f"regions:{k}"
            return None
        if layer == "windows":
            if not on_grid:
                return None
            best, depth = None, -1.0
            from scipy.ndimage import distance_transform_edt
            for k, (_wid, m) in enumerate(geo.window_masks(s)):
                if m[j, i]:
                    dd = float(distance_transform_edt(np.pad(m, 1))[j + 1, i + 1])
                    if dd > depth:
                        best, depth = f"windows:{k}", dd
            return best
        if layer == "boundaries":
            best, dist = None, 3.0 * tol
            for k, b in enumerate(s.boundaries):
                for poly in boundary_polylines(s, b):
                    dd = _distance_to_polyline(poly, x, y)
                    if dd < dist:
                        best, dist = f"boundary:{k}", dd
            return best
        if layer == "devices":
            best, dist = None, 2.0 * tol
            for k, v in enumerate(s.devices):
                dd = math.hypot(v.x / d.dx - x, v.y / d.dx - y)
                if dd < dist:
                    best, dist = f"devices:{k}", dd
            return best
        if layer == "attachments":
            pos = self.circuit_nodes()
            best, dist = None, 3.0 * tol
            for k, a in enumerate(s.attachments):
                if a.a in pos and a.b in pos:
                    poly = np.array([pos[a.a], pos[a.b]])
                    dd = _distance_to_polyline(poly, x, y)
                    if dd < dist:
                        best, dist = f"parts:{k}", dd
            return best
        return None

    # ------------------------------------------------------------------
    # canvas -> case: the handles
    # ------------------------------------------------------------------
    def _on_handles(self, _attr, _old, new) -> None:
        """A rectangle's corner dragged; the opposite corner stays put."""
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
        key = self.selected.split(":")[0]
        spec = self.wb.spec
        names = [getattr(spec, key)[k].id for k in changed]

        def apply(c: CaseSpec):
            items = getattr(c, key)
            for k, b in changed.items():
                items[k].x0, items[k].y0, items[k].nx, items[k].ny = b
        if not self.wb.edit(apply, f"resized {', '.join(names)}"):
            self.sync()

    def _on_devices(self, _attr, _old, new) -> None:
        """A rotor dragged (snapped) or removed (Backspace)."""
        if self._syncing:
            return
        spec, d = self.wb.spec, self.wb.spec.domain
        before = {v.id: v for v in spec.devices}
        step = self._snap()
        result = []
        for x, y, i in zip(new["x"], new["y"], new["id"]):
            old = before.get(str(i))
            if old is None:
                continue
            sx = geo.snap_value(x, step, d.nx) * d.dx
            sy = geo.snap_value(y, step, d.ny) * d.dx
            result.append(old.model_copy(update=dict(x=sx, y=sy)))
        kept = {v.id for v in result}
        gone = [v for v in before if v not in kept]
        moved = [v.id for v in result if (v.x, v.y) != (before[v.id].x, before[v.id].y)]
        label = _describe("rotor", [], moved, gone)
        if label is None:
            self.sync()
            return

        def apply(c: CaseSpec):
            c.devices = result
        if not self.wb.edit(apply, label):
            self.sync()

    def _on_vertices(self, _attr, _old, new) -> None:
        """A vertex dragged (moved, snapped) or removed (Backspace)."""
        if self._syncing:
            return
        rows = self._handle_rows["vtx"]
        got = list(zip(new["ref"], new["k"], new["x"], new["y"]))
        spec = self.wb.spec
        if len(got) < len(rows):
            have = {(r, int(k)) for r, k, _x, _y in got}
            gone = [(r, k) for r, k, _x, _y in rows if (r, k) not in have]
            ref = gone[0][0]
            o = get_outline(spec, ref)
            pts, ks, bs = list(o.points), o.kinds(), o.bulges()
            origin = list(range(len(pts)))
            for k in sorted((k for r, k in gone if r == ref), reverse=True):
                if len(pts) <= 3:
                    self.wb.notify("info", "a shape keeps at least three vertices; delete "
                                           "the shape instead")
                    self.sync()
                    return
                pts, ks, bs, keep = shapes.remove_vertex(pts, ks, bs, k)
                origin = [origin[j] for j in keep]
            self._edit_shape(ref, Outline(points=pts, edges=ks, bulge=bs), origin,
                             f"{shape_label(spec, ref)}: removed a vertex")
            return
        for (r, k, x0, y0), (_r, _k, x, y) in zip(rows, got):
            if abs(x - x0) > 1e-9 or abs(y - y0) > 1e-9:
                o = get_outline(spec, r)
                p = self._snap_pt(x, y)
                pts, ks, bs = shapes.move_vertex(o.points, o.kinds(), o.bulges(), k, *p)
                self._edit_shape(r, Outline(points=pts, edges=ks, bulge=bs), None,
                                 f"{shape_label(spec, r)}: moved vertex {k} to {p}")
                return
        self.sync()

    def _on_edges(self, _attr, _old, new) -> None:
        """An edge's circle dragged (the edge bends into an arc, or straightens) or
        removed (Backspace: straight)."""
        if self._syncing:
            return
        rows = self._handle_rows["edge"]
        got = list(zip(new["ref"], new["k"], new["x"], new["y"]))
        spec = self.wb.spec
        if len(got) < len(rows):
            have = {(r, int(k)) for r, k, _x, _y in got}
            r, k = next((r, k) for r, k, _x, _y in rows if (r, k) not in have)
            o = get_outline(spec, r)
            pts, ks, bs = shapes.set_kind(o.points, o.kinds(), o.bulges(), k, "line")
            self._edit_shape(r, Outline(points=pts, edges=ks, bulge=bs), None,
                             f"{shape_label(spec, r)}: edge {k} straight")
            return
        for (r, k, x0, y0), (_r, _k, x, y) in zip(rows, got):
            if abs(x - x0) > 1e-9 or abs(y - y0) > 1e-9:
                o = get_outline(spec, r)
                pts, ks, bs = shapes.bend(o.points, o.kinds(), o.bulges(), k, (x, y))
                what = f"an arc (bulge {bs[k]:.3g})" if ks[k] == "arc" else "straight"
                self.state["sub"] = k
                self._edit_shape(r, Outline(points=pts, edges=ks, bulge=bs), None,
                                 f"{shape_label(spec, r)}: edge {k} {what}")
                return
        self.sync()

    def _on_move(self, _attr, _old, new) -> None:
        """A diamond dragged (the shape moves, by whole snap steps) or removed
        (Backspace: the shape is deleted)."""
        if self._syncing:
            return
        rows = self._handle_rows["move"]
        got = list(zip(new["ref"], new["x"], new["y"]))
        if len(got) < len(rows):
            have = {r for r, _x, _y in got}
            gone = [r for r, _x, _y in rows if r not in have]
            self.delete_shapes(gone)
            return
        step = self._snap()
        for (r, x0, y0), (_r, x, y) in zip(rows, got):
            if abs(x - x0) > 1e-9 or abs(y - y0) > 1e-9:
                dx = round((x - x0) / step) * step
                dy = round((y - y0) / step) * step
                if dx == 0 and dy == 0:
                    break
                self.translate_shape(r, float(dx), float(dy))
                return
        self.sync()

    def double_click_handle(self, x: float, y: float) -> None:
        """A double click on an edge's circle adds a vertex there; on a vertex's
        square it makes the curve smooth through that vertex (both its edges
        splines), or a corner again (both straight)."""
        tol = self._tolerance()

        def nearest(rows):
            best, dist = None, tol
            for row in rows:
                dd = math.hypot(row[2] - x, row[3] - y)
                if dd <= dist:
                    best, dist = row, dd
            return best
        spec = self.wb.spec
        hit = nearest(self._handle_rows["edge"])
        if hit is not None:
            ref, k = hit[0], hit[1]
            o = get_outline(spec, ref)
            pts, ks, bs, origin = shapes.split(o.points, o.kinds(), o.bulges(), k)
            self._edit_shape(ref, Outline(points=pts, edges=ks, bulge=bs), origin,
                             f"{shape_label(spec, ref)}: a vertex added on edge {k}")
            return
        hit = nearest(self._handle_rows["vtx"])
        if hit is not None:
            ref, k = hit[0], hit[1]
            o = get_outline(spec, ref)
            n = len(o.points)
            ks = o.kinds()
            smooth = not (ks[(k - 1) % n] == "spline" and ks[k] == "spline")
            kind = "spline" if smooth else "line"
            pts, ks2, bs2 = shapes.set_kind(o.points, ks, o.bulges(), (k - 1) % n, kind)
            pts, ks2, bs2 = shapes.set_kind(pts, ks2, bs2, k, kind)
            self._edit_shape(ref, Outline(points=pts, edges=ks2, bulge=bs2), None,
                             f"{shape_label(spec, ref)}: "
                             f"{'smooth' if smooth else 'a corner'} at vertex {k}")

    # ------------------------------------------------------------------
    # drawing, and the one-click shapes
    # ------------------------------------------------------------------
    def undo_vertex(self) -> None:
        self.state["pending"] = list(self.state.get("pending") or [])[:-1]
        self._sync_pending()
        self.hint.object = self._hint_html()

    def cancel_shape(self) -> None:
        self.state["pending"] = []
        self._sync_pending()
        self.hint.object = self._hint_html()

    def close_shape(self) -> None:
        """The points placed so far become a closed shape on the active layer."""
        st = self.state
        pts: list[tuple[float, float]] = []
        for p in (tuple(q) for q in st.get("pending") or []):
            if not pts or pts[-1] != p:
                pts.append(p)
        if len(pts) > 1 and pts[-1] == pts[0]:
            pts.pop()
        if len(pts) < 3:
            self.wb.notify("info", "place at least three points, then double-click (or "
                                   "Finish)")
            return
        kind = st.get("draw_edges", "line")
        o = Outline(points=pts, edges=[kind] * len(pts), bulge=[0.0] * len(pts))
        why = o.problems()
        st["pending"] = []
        self._sync_pending()
        if why:
            self.wb.notify("error", f"not drawn: {why[0]}")
            return
        self.add_drawn(o)

    def add_rectangle(self, a, b) -> None:
        """Two opposite corners: a rectangular window or region, or the domain's
        outline (or a hole) as a drawn rectangle."""
        d = self.wb.spec.domain
        layer = self.state["layer"]
        x0, x1 = sorted((a[0], b[0]))
        y0, y1 = sorted((a[1], b[1]))
        if x1 - x0 < 1 or y1 - y0 < 1:
            self.wb.notify("info", "a rectangle needs two different corners")
            return
        if layer == "domain":
            self.add_drawn(Outline.of(shapes.rectangle(x0, y0, x1 - x0, y1 - y0)))
            return
        box = geo.snap_box(x0, y0, x1, y1, self._snap(), d.nx, d.ny)
        if box is None:
            return
        s = self.wb.spec
        if layer == "windows":
            wid = _next_id("F", [w.id for w in s.windows])
            new = Window(id=wid, x0=box[0], y0=box[1], nx=box[2], ny=box[3])

            def apply(c):
                c.layout = None
                c.windows.append(new)
            if self.wb.edit(apply, f"drew window {wid}"):
                self._done_drawing(f"windows:{len(self.wb.spec.windows) - 1}")
        else:
            rid = _next_id("R", [r.id for r in s.regions])
            mat = self.state.get("material") or "material-1"
            new = Region(id=rid, material=mat, x0=box[0], y0=box[1], nx=box[2], ny=box[3])

            def apply(c):
                c.regions.append(new)
                with_material(c, mat)
            if self.wb.edit(apply, f"drew region {rid} ({mat})"):
                self._done_drawing(f"regions:{len(self.wb.spec.regions) - 1}")

    def add_circle(self, c, r: float) -> None:
        if r < 1.0:
            self.wb.notify("info", "a circle needs a radius of at least a cell: click its "
                                   "centre, then a point on it")
            return
        self.add_drawn(Outline.of(shapes.circle(float(c[0]), float(c[1]), float(r))))

    def add_drawn(self, o: Outline) -> None:
        """A drawn shape added to the active layer: the domain's outline (replacing
        the whole grid, or the outline before), a hole in the domain, a window or a
        region.  A drawn domain edge gets its own boundary (`spec.derived_kind`)."""
        s, layer = self.wb.spec, self.state["layer"]
        if layer == "domain" and self.state.get("draw_as") == "hole":
            h = len(s.domain.holes)

            def apply(c):
                c.domain.holes.append(o)
                rebase_boundaries(c, f"hole{h}", len(o.points), None)
                follow_the_domain(c)
            label, then = f"cut hole {h} in the domain", f"domain:hole:{h}"
        elif layer == "domain":
            def apply(c):
                if c.domain.outline is None:
                    c.boundaries = [b for b in c.boundaries if b.drawn]
                c.domain.outline = o
                rebase_boundaries(c, "outline", len(o.points), None)
                follow_the_domain(c)
            label, then = "drew the domain's outline", "domain:outline"
        elif layer == "windows":
            wid = _next_id("F", [w.id for w in s.windows])

            def apply(c):
                c.layout = None
                c.windows.append(Window(id=wid, shape="curve", outline=o))
            label, then = f"drew window {wid}", f"windows:{len(s.windows)}"
        else:
            rid = _next_id("R", [r.id for r in s.regions])
            mat = self.state.get("material") or "material-1"

            def apply(c):
                c.regions.append(Region(id=rid, material=mat, shape="curve", outline=o))
                with_material(c, mat)
            label, then = f"drew region {rid} ({mat})", f"regions:{len(s.regions)}"
        #: a drawn domain (or hole) is a new shape: what it now lacks to run -- a
        #: material, an inlet, a clamp -- is filled in and said (`starter`)
        if self.wb.edit(apply, label, defaults=(layer == "domain")):
            self._done_drawing(then)
        else:
            self.sync()

    def _done_drawing(self, then: str) -> None:
        """Every tool adds one thing, then hands back to Select with it selected, so
        the next click edits what was just made (seen: Rectangle stayed armed after
        its region while Draw had handed back)."""
        if self.state["tool"] != "select":
            self.state["tool"] = "select"
            self.state["selected"], self.state["sub"] = then, None
            self.wb.show("model", keep_view=True)
        else:
            self.select(then)

    def add_ready(self, what: str) -> None:
        """A ready-made circle, ellipse or hexagon in the middle of the view (for the
        domain's outline: of the grid, filling most of it)."""
        d = self.wb.spec.domain
        layer = self.state["layer"]
        if layer == "domain" and self.state.get("draw_as") != "hole":
            cx, cy, r = d.nx / 2.0, d.ny / 2.0, 0.45 * min(d.nx, d.ny)
            rx = 0.45 * d.nx
        else:
            x0, x1, y0, y1 = self.ranges()
            cx, cy = self._snap_pt((x0 + x1) / 2.0, (y0 + y1) / 2.0)
            r = max(2.0, 0.2 * min(d.nx, d.ny, x1 - x0, y1 - y0))
            rx = 1.5 * r
        if what == "circle":
            shape = shapes.circle(cx, cy, r)
        elif what == "ellipse":
            shape = shapes.ellipse(cx, cy, rx, r)
        else:
            shape = shapes.regular(cx, cy, r, 6)
        self.add_drawn(Outline.of(shape))

    def add_rotor(self, x: float, y: float) -> None:
        s, d = self.wb.spec, self.wb.spec.domain
        vid = _next_id("R", [v.id for v in s.devices])
        new = Device(id=vid, x=geo.snap_value(x, self._snap(), d.nx) * d.dx,
                     y=geo.snap_value(y, self._snap(), d.ny) * d.dx)

        def apply(c):
            c.devices.append(new)
        if self.wb.edit(apply, f"added rotor {vid}"):
            self._done_drawing(f"devices:{len(self.wb.spec.devices) - 1}")

    # ------------------------------------------------------------------
    # edits the inspector and the handles make
    # ------------------------------------------------------------------
    def _edit_shape(self, ref: str, new: Outline, origin: list[int] | None,
                    label: str) -> None:
        """Replace the drawn shape ``ref`` with ``new``.  ``origin`` (new edge -> old
        edge) is given when the edges were renumbered, so a domain edge's boundary
        follows its edge."""
        why = new.problems()
        if why:
            self.wb.notify("error", f"not applied: {shape_label(self.wb.spec, ref)}: {why[0]}")
            self.sync()
            return

        def apply(c):
            set_outline(c, ref, new)
            ring = ring_of(ref)
            if ring is not None and origin is not None:
                rebase_boundaries(c, ring, len(new.points), origin)
        if not self.wb.edit(apply, label):
            self.sync()

    def set_edge_kind(self, ref: str, k: int, kind: str, bulge: float | None = None) -> None:
        o = get_outline(self.wb.spec, ref)
        try:
            pts, ks, bs = shapes.set_kind(o.points, o.kinds(), o.bulges(), k, kind, bulge)
        except ValueError as exc:
            self.wb.notify("error", f"not applied: {exc}")
            return
        self.state["sub"] = k
        self._edit_shape(ref, Outline(points=pts, edges=ks, bulge=bs), None,
                         f"{shape_label(self.wb.spec, ref)}: edge {k} {kind}")

    def translate_shape(self, ref: str, dx: float, dy: float) -> None:
        """Move a drawn shape or a rectangle; a window's or the domain's holes move with
        its outline, and a rectangle stays inside the grid."""
        spec = self.wb.spec
        d = spec.domain
        p = ref.split(":")
        if p[0] in ("windows", "regions") and len(p) == 2:
            item = getattr(spec, p[0])[int(p[1])]
            if item.shape == "rect":
                x0 = int(min(max(item.x0 + dx, 0), d.nx - item.nx))
                y0 = int(min(max(item.y0 + dy, 0), d.ny - item.ny))

                def apply_rect(c):
                    it = getattr(c, p[0])[int(p[1])]
                    it.x0, it.y0 = x0, y0
                if not self.wb.edit(apply_rect, f"{item.id}: moved to ({x0}, {y0})"):
                    self.sync()
                return
            if item.shape != "curve":
                self.wb.notify("info", f"{item.id} is not drawn here, so it cannot be moved "
                                       f"here")
                self.sync()
                return

        def moved(o: Outline) -> Outline:
            return Outline(points=shapes.translate(o.points, dx, dy), edges=o.kinds(),
                           bulge=o.bulges())

        def apply(c):
            set_outline(c, ref, moved(get_outline(c, ref)))
            if ref == "domain:outline":
                c.domain.holes = [moved(h) for h in c.domain.holes]
            elif p[0] == "windows" and len(p) == 2:
                w = c.windows[int(p[1])]
                w.holes = [moved(h) for h in w.holes]
        if not self.wb.edit(apply, f"{shape_label(spec, ref)}: moved by ({dx:g}, {dy:g})"):
            self.sync()

    def delete_shapes(self, refs: list[str]) -> None:
        """Delete drawn shapes by reference: a window or region, a window's hole, a
        hole in the domain (with its boundaries), or the domain's outline (the
        domain is the whole grid again)."""
        spec = self.wb.spec
        fid = spec.physics.family
        names = [shape_label(spec, r) for r in refs]

        def apply(c):
            for ref in sorted(refs, key=lambda r: [int(t) if t.isdigit() else 0
                                                   for t in r.split(":")], reverse=True):
                p = ref.split(":")
                if ref == "domain:outline":
                    c.domain.outline = None
                    c.boundaries = [b for b in c.boundaries
                                    if not b.edge.startswith("outline:")]
                    if not any(not b.drawn for b in c.boundaries):
                        c.boundaries = [b for b in family_boundaries(fid)] + c.boundaries
                elif p[0] == "domain":
                    h = int(p[2])
                    del c.domain.holes[h]
                    drop_hole_boundaries(c, h)
                elif len(p) == 4:
                    del c.windows[int(p[1])].holes[int(p[3])]
                elif p[0] == "windows":
                    # the layout is not cleared here: `Workbench.edit` sees windows
                    # edited by hand, hands them to the case and says so
                    del c.windows[int(p[1])]
                else:
                    del c.regions[int(p[1])]
        self.state["selected"], self.state["sub"] = None, None
        if not self.wb.edit(apply, f"deleted {', '.join(names)}"):
            self.sync()

    def delete_selected(self) -> None:
        """Delete what is selected, whatever it is."""
        ref = self.selected
        if ref is None:
            self.wb.notify("info", "select something first: click it on the canvas or in "
                                   "the list")
            return
        p = ref.split(":")
        if p[0] in ("domain", "windows", "regions"):
            self.delete_shapes([ref])
            return
        spec = self.wb.spec
        key = {"boundary": "boundaries", "devices": "devices", "parts": "attachments"}.get(p[0])
        if key is None:
            return
        k = int(p[1])
        name = getattr(spec, key)[k].id

        def apply(c):
            del getattr(c, key)[k]
        self.state["selected"], self.state["sub"] = None, None
        self.wb.edit(apply, f"deleted {name}")

    def set_layout(self, follow: bool, cut: str, along: int, across: int) -> None:
        """Windows that follow the domain (generated from its shape, `layout.py`), or
        not (the windows there now stay, as the case's own)."""
        if not follow:
            if self.wb.spec.layout is None:
                return

            def apply(c):
                c.layout = None
            self.wb.edit(apply, "the windows are your own now; they stay as they are")
            return
        new = Layout(cut=cut, along=max(1, int(along or 1)), across=max(1, int(across or 1)))
        what = ("one per material" if cut == "materials" else
                f"{new.along} along" + (f" x {new.across} across" if new.across > 1 else ""))

        def apply(c):
            c.layout = new
        if not self.wb.edit(apply, f"the windows follow the domain: {what}"):
            self.sync()

    def reset_domain(self) -> None:
        s = self.wb.spec
        if s.domain.outline is None and not s.domain.holes:
            self.wb.notify("info", "the domain is already the whole grid")
            return
        fid = s.physics.family

        def apply(c):
            c.domain.outline = None
            c.domain.holes = []
            c.boundaries = [b for b in c.boundaries if not b.drawn]
            if not c.boundaries:
                c.boundaries = family_boundaries(fid)
        self.state["selected"], self.state["sub"] = None, None
        self.wb.edit(apply, "the domain is the whole grid again", defaults=True)

    def add_part(self, kind: str) -> None:
        """A battery or a resistor, wired between the first electrode and a new free
        node, to be edited in the inspector."""
        s = self.wb.spec
        electrodes = [b.id for b in s.boundaries if b.kind == "electrode"]
        taken = [a.id for a in s.attachments]
        pid = _next_id("B" if kind == "battery" else "R", taken)
        nodes = {n for a in s.attachments for n in (a.a, a.b)}
        free = _next_id("n", nodes | set(electrodes))
        new = Attachment(id=pid, kind=kind, value=12.0 if kind == "battery" else 1.0,
                         a=free if kind == "battery" else (electrodes[0] if electrodes
                                                           else "n1"),
                         b=electrodes[0] if (kind == "battery" and electrodes) else free)

        def apply(c):
            c.attachments.append(new)
        if self.wb.edit(apply, f"added {kind} {pid}"):
            self.select(f"parts:{len(self.wb.spec.attachments) - 1}")

    def restack(self, step: int) -> None:
        ref = self.selected
        if ref is None or not ref.startswith("regions:"):
            self.wb.notify("info", "select a region to move it up or down the stack")
            return
        k = int(ref.split(":")[1])
        j = k + step
        if not 0 <= j < len(self.wb.spec.regions):
            return
        rid = self.wb.spec.regions[k].id

        def apply(c):
            c.regions[k], c.regions[j] = c.regions[j], c.regions[k]
        self.state["selected"] = f"regions:{j}"
        if not self.wb.edit(apply, f"region {rid} moved "
                                   f"{'forward' if step > 0 else 'backward'}"):
            self.state["selected"] = ref

    def use_family_boundaries(self) -> None:
        from .spec import default_boundaries

        def apply(c):
            c.boundaries = default_boundaries(c)
        self.wb.edit(apply, "boundaries set to the family's")

    # ------------------------------------------------------------------
    # switching the layer and the tool
    # ------------------------------------------------------------------
    def _switch(self, **kw) -> None:
        if "tool" in kw:
            kw["tool"] = TOOL_ALIASES.get(kw["tool"], kw["tool"])
        if "layer" in kw or "tool" in kw:
            self.state["pending"] = []              # a half-drawn shape does not carry over
        if "layer" in kw and kw["layer"] != self.state.get("layer"):
            self.state["selected"], self.state["sub"] = None, None
        self.state.update(kw)
        self.wb.log("model: " + ", ".join(f"{k} = {v}" for k, v in kw.items()))
        self.wb.show("model", keep_view=True)

    def _hint_html(self) -> str:
        """One line under the canvas: what a click does now."""
        st, s = self.state, self.wb.spec
        layer, tool = st["layer"], st["tool"]
        pend = st.get("pending") or []
        what = {"domain": ("the domain's outline" if st.get("draw_as") != "hole"
                           else "a hole in the domain"),
                "windows": "a window", "regions": f"a region of {st.get('material')}"}
        if tool == "pan":
            return "Drag to pan, scroll to zoom. Nothing is edited with this tool."
        if tool == "draw" and layer == "devices":
            return "<b>Click</b> to place a rotor."
        if tool == "draw" and layer in DRAWABLE:
            if not pend:
                return (f"Drawing {what[layer]}: <b>click</b> to place each point, "
                        f"<b>double-click</b> to finish.")
            return (f"{len(pend)} point{'s' * (len(pend) != 1)} placed: keep clicking, "
                    f"<b>double-click</b> (or Finish) to close the shape.")
        if tool == "rect" and layer in DRAWABLE:
            return (f"Rectangle as {what[layer]}: <b>click</b> one corner, then the "
                    f"opposite one." if not pend else "Now <b>click</b> the opposite corner.")
        if tool == "circle" and layer in DRAWABLE:
            return (f"Circle as {what[layer]}: <b>click</b> its centre, then a point on "
                    f"it." if not pend else "Now <b>click</b> a point on the circle.")
        if tool in ("draw", "rect", "circle"):
            return "Nothing is drawn on this layer: choose Shape, Materials or Windows."
        ref = self.selected
        if ref is not None and self.curves():
            return ("Drag a <b>square</b> to move a vertex, a <b>circle</b> to bend its edge, "
                    "the <b>diamond</b> to move it. Double-click a circle to add a vertex.")
        if ref is not None and self._boxes():
            #: a rectangle has corners and a centre, no edges to bend (seen: the
            #: curve hint shown for a rectangle region named circles it did not have)
            return ("Drag a <b>corner</b> to resize it, the <b>diamond</b> to move it; "
                    "the inspector has its exact place.")
        return {"domain": ("Click the shape to reshape it, or <b>Draw</b> its outline. "
                           "The grid is the domain until you do."),
                "regions": "Click a region to select it, or draw one with the tools.",
                "boundaries": "Click an edge to set its condition.",
                "windows": ("Click a window to select it." if s.layout is None else
                            "The windows follow the shape; choose how in the inspector."),
                "physics": "The physics, the time and the grid are in the inspector.",
                "devices": "Click a rotor to select it and drag it; <b>Draw</b> places new "
                           "ones.",
                "attachments": "Click a part of the circuit to edit it."}.get(layer, "")

    # ------------------------------------------------------------------
    # the view
    # ------------------------------------------------------------------
    def view(self):
        from . import inspector
        return inspector.model_layout(self)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _family(spec):
    try:
        return registry.family(spec.physics.family)
    except KeyError:
        return None


def material_names(spec) -> list[str]:
    """What a new region may be made of: the case's materials, then the family
    library's others."""
    fam = _family(spec)
    lib = fam.material_library() if fam is not None else {}
    return list(spec.materials) + [m for m in lib if m not in spec.materials]


def with_material(c: CaseSpec, name: str) -> None:
    """A region drawn in a material the case does not define yet takes the family
    library's properties for it, in the same edit, when the library has that
    name; any other name is left for the inspector (and the check says so)."""
    if name in c.materials:
        return
    fam = _family(c)
    lib = fam.material_library() if fam is not None else {}
    if name in lib:
        c.materials[name] = dict(lib[name])


def _region_rings(r) -> list[list[tuple[float, float]]]:
    if r.shape == "rect":
        return [list(geo.corners((r.x0, r.y0, r.nx, r.ny)))]
    if r.shape == "curve":
        ring = r.outline.ring()
        return [[(float(a), float(b)) for a, b in ring]]
    return [list(r.points)] + [list(h) for h in (r.holes or [])]


def boundary_polylines(spec, b) -> list[np.ndarray]:
    """Where a boundary lies, as polylines in cells: its drawn edge, or its segment of
    a grid edge."""
    d = spec.domain
    if b.drawn:
        for name, poly in geo.drawn_edges(d, step=0.5):
            if name == b.edge:
                return [poly]
        return []
    n = geo.edge_length(b.edge, d.nx, d.ny)
    stop = n if b.stop is None else b.stop
    x0, y0, x1, y1 = geo.edge_segment_xy(b.edge, b.start, stop, d.nx, d.ny)
    return [np.array([[x0, y0], [x1, y1]], dtype=float)]


def _distance_to_polyline(poly: np.ndarray, x: float, y: float) -> float:
    poly = np.asarray(poly, dtype=float)
    if len(poly) == 1:
        return float(np.hypot(poly[0, 0] - x, poly[0, 1] - y))
    a, b = poly[:-1], poly[1:]
    ab = b - a
    L2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300)
    t = np.clip(((x - a[:, 0]) * ab[:, 0] + (y - a[:, 1]) * ab[:, 1]) / L2, 0.0, 1.0)
    nx_, ny_ = a[:, 0] + t * ab[:, 0], a[:, 1] + t * ab[:, 1]
    return float(np.min(np.hypot(nx_ - x, ny_ - y)))


def _edge_summary(o: Outline) -> str:
    """``2 lines, 2 arcs``: a drawn shape's edges by kind."""
    ks = o.kinds()
    parts = [f"{ks.count(k)} {k}{'s' * (ks.count(k) != 1)}" for k in shapes.EDGE_KINDS
             if ks.count(k)]
    return ", ".join(parts)


def _describe(what: str, drawn, moved, gone) -> str | None:
    parts = []
    if drawn:
        parts.append(f"drew {what} {', '.join(drawn)}")
    if moved:
        parts.append(f"moved {what} {', '.join(moved)}")
    if gone:
        parts.append(f"deleted {what} {', '.join(gone)}")
    return "; ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# drawn shapes by reference, and the boundaries that follow a domain's edges
# ---------------------------------------------------------------------------
#: a drawn shape is named by where it lives in the case: ``domain:outline``,
#: ``domain:hole:<h>``, ``windows:<i>``, ``windows:<i>:hole:<h>``, ``regions:<i>``


def get_outline(c: CaseSpec, ref: str) -> Outline:
    p = ref.split(":")
    if p[0] == "domain":
        return c.domain.outline if p[1] == "outline" else c.domain.holes[int(p[2])]
    obj = (c.windows if p[0] == "windows" else c.regions)[int(p[1])]
    return obj.holes[int(p[3])] if len(p) > 2 else obj.outline


def set_outline(c: CaseSpec, ref: str, o: Outline) -> None:
    """Put ``o`` in place of the shape ``ref``.  A window's or region's bounding box
    is re-derived when the edit is validated (`Workbench.edit`)."""
    p = ref.split(":")
    if p[0] == "domain":
        if p[1] == "outline":
            c.domain.outline = o
        else:
            c.domain.holes[int(p[2])] = o
        return
    obj = (c.windows if p[0] == "windows" else c.regions)[int(p[1])]
    if len(p) > 2:
        obj.holes[int(p[3])] = o
    else:
        obj.outline = o


def ring_of(ref: str) -> str | None:
    """The name a domain shape's edges carry in a boundary (``outline``,
    ``hole<h>``); None for a window or region, whose edges bound nothing."""
    p = ref.split(":")
    if p[0] != "domain":
        return None
    return "outline" if p[1] == "outline" else f"hole{p[2]}"


def shape_label(c: CaseSpec, ref: str | None) -> str:
    if ref is None:
        return ""
    p = ref.split(":")
    if p[0] == "domain":
        return "the domain's outline" if p[1] == "outline" else f"hole {p[2]}"
    if p[0] == "boundary":
        b = c.boundaries[int(p[1])]
        return f"boundary {b.id}"
    if p[0] == "devices":
        return f"rotor {c.devices[int(p[1])].id}"
    if p[0] == "parts":
        return f"part {c.attachments[int(p[1])].id}"
    if p[0] == "material":
        return f"material {ref[len('material:'):]}"
    obj = (c.windows if p[0] == "windows" else c.regions)[int(p[1])]
    what = "window" if p[0] == "windows" else "region"
    return f"{what} {obj.id}" + (f", hole {p[3]}" if len(p) > 2 else "")


def rebase_boundaries(c: CaseSpec, ring: str, n_edges: int,
                      origin: list[int] | None) -> None:
    """One boundary per edge of a drawn ring (``outline`` or ``hole<h>``) after an
    edit.  New edge ``j`` takes the condition old edge ``origin[j]`` had (an edge
    split in two gives both halves its condition); with no ``origin`` -- a new
    ring -- every edge takes the family's own choice (`spec.derived_kind`)."""
    from .spec import derived_kind
    prefix = ring + ":"
    old = {b.edge: b for b in c.boundaries if b.edge.startswith(prefix)}
    keep = [b for b in c.boundaries if not b.edge.startswith(prefix)]
    taken = {b.id for b in keep}
    new = []
    for j in range(n_edges):
        src = old.get(f"{ring}:{origin[j]}") if origin is not None else None
        if src is not None and src.id not in taken:
            b = src.model_copy(update={"edge": f"{ring}:{j}"})
        elif src is not None:
            b = src.model_copy(update={"edge": f"{ring}:{j}", "id": _next_id("B", taken)})
        else:
            b = Boundary(id=_next_id("B", taken), edge=f"{ring}:{j}",
                         kind=derived_kind(c, f"{ring}:{j}"))
        taken.add(b.id)
        new.append(b)
    c.boundaries = keep + new


def follow_the_domain(c: CaseSpec) -> None:
    """A domain that has just been drawn: rectangles do not fit it, so windows that
    are all still rectangles give way to windows that follow it (`layout.py`) --
    as many along it as there were (at least two), one per material for style C
    when there are two materials, one window for D and the split.  Windows the
    person has drawn are theirs and stay."""
    if c.layout is not None or any(w.shape != "rect" for w in c.windows):
        return
    style = c.coupling.style
    materials = {r.material for r in c.regions}
    if style in ("C", "M"):
        c.layout = Layout(cut="materials") if len(materials) >= 2 else Layout(along=2)
    elif style in ("D", "split"):
        c.layout = Layout(along=1)
    else:
        c.layout = Layout(along=max(2, len(c.windows)))


def drop_hole_boundaries(c: CaseSpec, h: int) -> None:
    """Hole ``h`` is gone: its boundaries go, and later holes' shift down by one."""
    out = []
    for b in c.boundaries:
        m = geo.DRAWN_EDGE.match(b.edge)
        if m and m.group(2) is not None:
            k = int(m.group(2))
            if k == h:
                continue
            if k > h:
                b = b.model_copy(update={"edge": f"hole{k - 1}:{m.group(3)}"})
        out.append(b)
    c.boundaries = out


def _inside_point(mask: np.ndarray, outline: Outline | None) -> tuple[float, float]:
    """Where to write a shape's name: its cell furthest from its edge (inside it even
    when it is not convex), or its vertices' mean when it holds no cell."""
    if mask.any():
        from scipy.ndimage import distance_transform_edt
        d = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
        j, i = np.unravel_index(int(np.argmax(d)), d.shape)
        return float(i + 0.5), float(j + 0.5)
    if outline is None:
        return float("nan"), float("nan")
    c = np.mean(np.asarray(outline.points, dtype=float), axis=0)
    return float(c[0]), float(c[1])


__all__ = ["GeometryEditor", "LAYERS", "TOOLS", "material_colours", "layers_for",
           "material_names", "with_material", "boundary_polylines"]
