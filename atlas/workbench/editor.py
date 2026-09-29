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

**Drawn shapes** (case file 0.4, the owner's request of 2026-09-29): on the
Domain, Windows and Regions layers, **Draw shape** places vertices by clicking
and closes the shape with a double click, and **Reshape** gives every drawn
shape three kinds of handle -- a square on each vertex (drag it; Backspace
removes it; double-click it to make the curve smooth through it), a circle on
each edge (drag it off the line to bend the edge into a circular arc, back onto
it to straighten it; double-click it to add a vertex there), and a diamond that
moves the whole shape.  Each edge is a line, an arc or a spline (`shapes.py`),
set by those handles or in the Edges table.  The solvers get the cells whose
centres lie inside, and the canvas draws both: the domain's cells stepped as
the solver sees them, and the exact curve over them.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import panel as pn
from bokeh.events import DoubleTap, Tap
from bokeh.models import (BoxEditTool, BoxZoomTool, ColumnDataSource, HoverTool, LabelSet,
                          PanTool, PointDrawTool, Range1d, ResetTool, SaveTool,
                          SingleIntervalTicker, WheelZoomTool)
from bokeh.plotting import figure

from . import geometry as geo
from . import registry
from . import shapes
from .spec import (Attachment, Boundary, CaseSpec, Device, Layout, Outline, Region, Window,
                   check, family_boundaries)

if TYPE_CHECKING:                                             # pragma: no cover
    from .app import Workbench

LAYERS = {"domain": "Domain", "windows": "Windows", "regions": "Regions",
          "devices": "Devices", "boundaries": "Boundaries", "attachments": "Circuit"}
TOOLS = {"move": "Move & draw", "shape": "Draw shape", "resize": "Reshape",
         "pan": "Pan & zoom"}
#: the layers a shape can be drawn on
DRAWABLE = ("domain", "windows", "regions")

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

#: Common materials get a colour that reads as the material; others take the
#: palette in order of first appearance.
MATERIAL_COLOURS = {"steel": "#8d99ae", "copper": "#c8733a", "aluminium": "#b8c4d6",
                    "aluminum": "#b8c4d6", "air": "#bde0fe", "water": "#4895ef",
                    "concrete": "#a8a29e", "wood": "#b08968", "graphite": "#4b5563",
                    "shallow-fast": "#90e0ef", "deep-slow": "#0077b6", "pool": "#023e8a",
                    "chip": "#374151"}
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

_RESHAPE = ("Drag a **square** to move a vertex, a **circle** to bend its edge into an arc "
            "(drag it back onto the line to straighten it), a **diamond** to move the whole "
            "shape. **Double-click** a circle to add a vertex there, a square to make the "
            "curve smooth through it (again for a corner). Click a square, then "
            "**Backspace**, to remove that vertex. Each edge's kind is also in the Edges table.")
_DRAW = ("**Click** to place each vertex; **double-click** (or *Close shape*) to close it. "
         "New edges are straight or smooth, as chosen below; bend any of them afterwards "
         "with **Reshape**. Or add a ready-made circle or ellipse.")
_NOT_DRAWN = ("Nothing is drawn on this layer. Shapes are drawn on the Domain, Windows and "
              "Regions layers.")

HELP = {
    ("domain", "move"): "The domain is the whole grid until you draw its outline. **Draw "
                        "shape** draws the outline or cuts a hole in it (choose below); "
                        "**Reshape** moves its vertices and bends its edges. The grid's size "
                        "is set in step 1 (Case). Grey is outside the domain: the white cells "
                        "are what the solver sees, and the dark line is the shape as drawn.",
    ("domain", "shape"): _DRAW + " It becomes the domain's outline, or a hole in it.",
    ("domain", "resize"): _RESHAPE,
    ("windows", "move"): "Tick **Windows follow the domain** (below) to have them generated "
                         "from the domain's own shape and kept fitting it as it changes. Or "
                         "make your own: drag a window to move it, **Shift+drag** on empty "
                         "space to draw a rectangle, *Draw shape* for any other shape; click "
                         "one, then **Backspace** (or *Delete selected*) to remove it. Drawn "
                         "(curved) windows move with Reshape's diamond.",
    ("windows", "shape"): _DRAW + " It becomes a new window: it holds the domain's cells "
                          "whose centres it contains.",
    ("windows", "resize"): "Drag a **corner handle** to resize a rectangular window; the "
                           "opposite corner stays put. " + _RESHAPE,
    ("regions", "move"): "Drag a region to move it. **Shift+drag** to draw a new one in the "
                         "material chosen below. Click one, then **Backspace** to remove it. "
                         "Regions stack in list order: a later one takes the cells it covers. "
                         "Imported polygons are shown but not moved here.",
    ("regions", "shape"): _DRAW + " It becomes a new region in the material chosen below.",
    ("regions", "resize"): "Drag a **corner handle** to resize a rectangular region. "
                           + _RESHAPE,
    ("devices", "shape"): _NOT_DRAWN,
    ("boundaries", "shape"): _NOT_DRAWN,
    ("attachments", "shape"): _NOT_DRAWN,
    ("devices", "move"): "**Click** empty space to add a rotor, drag one to move it, click one "
                         "and press **Backspace** to remove it.",
    ("devices", "resize"): "Devices have no corners; set a rotor's diameter in the table.",
    ("boundaries", "move"): "Boundaries are set in the table below and drawn along the "
                            "domain's edges.",
    ("boundaries", "resize"): "Boundaries are set in the table below.",
    ("attachments", "move"): "The circuit is set in the table below and drawn here as a "
                             "schematic: each electrode is a node beside its edge segment, "
                             "the circuit's other nodes sit below the domain, and each part "
                             "runs between its two nodes. A battery's `a` is its + terminal.",
    ("attachments", "resize"): "The circuit is set in the table below.",
}

#: the circuit schematic's colours: batteries, resistors, nodes
C_BATTERY = "#b45309"
C_RESISTOR = "#1d4ed8"
C_NODE = "#111827"


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
        #: drawing: the vertices placed so far, whether new edges are straight or
        #: smooth, and whether a shape drawn on the Domain layer is its outline or a hole
        self.state.setdefault("pending", [])
        self.state.setdefault("draw_edges", "line")
        self.state.setdefault("draw_as", "outline")
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
        self.src_att = ColumnDataSource(dict(x0=[], y0=[], x1=[], y1=[], lx=[], ly=[],
                                             text=[], colour=[], id=[]))
        self.src_att_nodes = ColumnDataSource(dict(x=[], y=[], name=[]))
        # drawn shapes: the grid's cells outside the domain (``src_dom_cells``, drawn
        # grey) and the domain's outline, curved windows, overlaps of
        # curved windows, drawn boundaries, the handles, and the shape being drawn
        self.src_dom_cells = ColumnDataSource(dict(left=[], right=[], bottom=[], top=[]))
        self.src_dom_line = ColumnDataSource(dict(xs=[], ys=[]))
        self.src_win_curve = ColumnDataSource(dict(xs=[], ys=[], id=[], cells=[]))
        #: generated windows (schema 0.5): their cuts across the domain, smooth
        self.src_win_gen = ColumnDataSource(dict(xs=[], ys=[]))
        self.src_win_lbl = ColumnDataSource(dict(x=[], y=[], id=[]))
        self.src_overlap_q = ColumnDataSource(dict(left=[], right=[], bottom=[], top=[]))
        self.src_bc_drawn = ColumnDataSource(dict(xs=[], ys=[], colour=[], id=[], text=[],
                                                  lx=[], ly=[]))
        self.src_vtx = ColumnDataSource(dict(x=[], y=[], ref=[], k=[], label=[], what=[]))
        self.src_edge = ColumnDataSource(dict(x=[], y=[], ref=[], k=[], label=[], what=[]))
        self.src_move = ColumnDataSource(dict(x=[], y=[], ref=[], label=[], what=[]))
        self.src_pending = ColumnDataSource(dict(x=[], y=[]))
        self.src_win.on_change("data", self._on_windows)
        self.src_reg_edit.on_change("data", self._on_regions)
        self.src_handles.on_change("data", self._on_handles)
        self.src_dev.on_change("data", self._on_devices)
        self.src_vtx.on_change("data", self._on_vertices)
        self.src_edge.on_change("data", self._on_edges)
        self.src_move.on_change("data", self._on_move)

    # ------------------------------------------------------------------
    # the canvas
    # ------------------------------------------------------------------
    def _circuit_shown(self) -> bool:
        fam = _family(self.wb.spec)
        return bool(self.wb.spec.attachments) or bool(fam and "attachments" in fam.layers)

    def _circuit_y(self) -> float:
        """Where the circuit's free nodes sit: a row below the domain."""
        d = self.wb.spec.domain
        pad = max(8, int(0.05 * max(d.nx, d.ny)))
        return -(pad + 0.18 * d.ny)

    def _figure(self, ranges):
        d = self.wb.spec.domain
        pad = max(8, int(0.05 * max(d.nx, d.ny)))
        if ranges is None:
            circuit = self._circuit_shown()
            bottom = self._circuit_y() - pad if circuit else -pad
            #: an electrode's node sits 2.5 boundary offsets outside its edge; the
            #: sides widen so it is not cut in half by the canvas's edge (seen)
            side = pad + (3.0 * max(3.0, 0.02 * max(d.nx, d.ny)) if circuit else 0.0)
            ranges = (-side, d.nx + side, bottom, d.ny + pad)
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
        layer, tool = self.state["layer"], self.state["tool"]
        drawn_domain = d.outline is not None or bool(d.holes)
        if drawn_domain:
            # the grid in white; what lies outside the domain is greyed over the
            # regions below (a region may reach past the domain), stepped exactly as
            # the solver sees the domain's cells, and the drawn outline goes on top
            p.rect(x=d.nx / 2, y=d.ny / 2, width=d.nx, height=d.ny, fill_color="#ffffff",
                   fill_alpha=1.0, line_color=C_DOMAIN, line_width=1, line_dash="dotted",
                   level="underlay")
        else:
            p.rect(x=d.nx / 2, y=d.ny / 2, width=d.nx, height=d.ny, fill_color="#ffffff",
                   fill_alpha=1.0, line_color=C_DOMAIN, line_width=2, level="underlay")

        # regions: filled shapes in stacking order, then the editable outlines
        rr = p.multi_polygons("xs", "ys", source=self.src_reg_draw, fill_color="colour",
                              fill_alpha=0.45, line_color="colour", line_alpha=0.9,
                              line_width=1)
        #: while drawing or reshaping, only the handles say what they are: the
        #: regions' and windows' own tips would stack over every handle
        quiet = tool in ("shape", "resize")
        if not quiet:
            p.add_tools(HoverTool(renderers=[rr], visible=False,
                                  tooltips=[("region", "@id"), ("material", "@material"),
                                            ("shape", "@shape")]))
        editing_regions = self.state["layer"] == "regions"
        reg_edit = p.rect("x", "y", "w", "h", source=self.src_reg_edit, fill_alpha=0.0,
                          line_color="#1f2937", line_dash="dashed",
                          line_alpha=0.9 if editing_regions else 0.0, line_width=1.5,
                          selection_line_color=C_HANDLE, selection_line_width=3,
                          nonselection_line_alpha=0.9 if editing_regions else 0.0)

        # outside a drawn domain: pale and dotted, over the regions (the warnings'
        # hatching is diagonal, so the two cannot be mistaken for each other)
        p.quad("left", "right", "bottom", "top", source=self.src_dom_cells, fill_color=C_VOID,
               fill_alpha=1.0, line_alpha=0, hatch_pattern=".", hatch_color="#94a3b8",
               hatch_alpha=0.7, hatch_scale=8)

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
            p.quad("left", "right", "bottom", "top", source=self.src_overlap_q,
                   fill_color=C_OVERLAP, fill_alpha=0.22, line_alpha=0)

        editing_windows = self.state["layer"] == "windows"
        wc = p.multi_polygons("xs", "ys", source=self.src_win_curve, fill_color=C_WINDOW,
                              fill_alpha=0.06, line_color=C_WINDOW,
                              line_width=2 if editing_windows else 1.2,
                              line_alpha=1.0 if editing_windows else 0.55)
        if not quiet:
            p.add_tools(HoverTool(renderers=[wc], visible=False,
                                  tooltips=[("window", "@id"), ("cells", "@cells")]))
        # windows generated from the geometry: where each one ends inside the domain
        p.multi_line("xs", "ys", source=self.src_win_gen, line_color=C_WINDOW,
                     line_width=2.2 if editing_windows else 1.5,
                     line_alpha=1.0 if editing_windows else 0.75)
        if self.wb.view_opts["labels"]:
            p.add_layout(LabelSet(x="x", y="y", text="id", source=self.src_win_lbl,
                                  text_align="center", text_baseline="middle",
                                  text_font_size="10px", text_color=C_WINDOW,
                                  text_alpha=0.8))
        win = p.rect("x", "y", "w", "h", source=self.src_win, fill_color=C_WINDOW,
                     fill_alpha=0.06, line_color=C_WINDOW,
                     line_width=2 if editing_windows else 1.2,
                     line_alpha=1.0 if editing_windows else 0.55,
                     selection_fill_alpha=0.25, selection_line_width=3,
                     nonselection_fill_alpha=0.06,
                     nonselection_line_alpha=1.0 if editing_windows else 0.55)
        if not quiet:
            p.add_tools(HoverTool(renderers=[win], visible=False,
                                  tooltips=[("window", "@id"), ("size", "@w x @h cells")]))
        if self.wb.view_opts["labels"]:
            p.add_layout(LabelSet(x="x", y="y", text="id", source=self.src_win,
                                  text_align="center", text_baseline="middle",
                                  text_font_size="10px", text_color=C_WINDOW,
                                  text_alpha=0.8))
        cross = p.scatter("x", "y", source=self.src_cross, marker="diamond", size=9,
                          fill_color="#ffffff", line_color="#374151", line_width=1.5)
        p.add_tools(HoverTool(renderers=[cross], visible=False, tooltips=[("cross-point", "@names")]))

        # the domain's drawn outline and holes, exactly as drawn
        if drawn_domain:
            p.multi_line("xs", "ys", source=self.src_dom_line, line_color="#111827",
                         line_width=2.5 if layer == "domain" else 1.5)

        # boundaries on drawn edges: the edge itself, in the condition's colour
        bcd = p.multi_line("xs", "ys", source=self.src_bc_drawn, line_color="colour",
                           line_width=6, line_alpha=0.75)
        p.add_tools(HoverTool(renderers=[bcd], visible=False, tooltips=[("boundary", "@id"),
                                                         ("", "@text")]))
        p.add_layout(LabelSet(x="lx", y="ly", text="text", source=self.src_bc_drawn,
                              text_align="center", text_baseline="middle",
                              text_font_size="10px", text_color="colour",
                              background_fill_color="#ffffff", background_fill_alpha=0.8))

        # boundaries, just outside the domain
        bcr = p.segment("x0", "y0", "x1", "y1", source=self.src_bc, line_color="colour",
                        line_width=6, line_cap="butt")
        p.add_tools(HoverTool(renderers=[bcr], visible=False, tooltips=[("boundary", "@id"),
                                                         ("kind", "@text")]))
        p.add_layout(LabelSet(x="lx", y="ly", text="text", angle="angle", source=self.src_bc,
                              text_align="center", text_baseline="middle",
                              text_font_size="10px", text_color="colour",
                              background_fill_color="#ffffff", background_fill_alpha=0.8))

        # the circuit: a schematic of nodes and parts, over the canvas
        if self._circuit_shown():
            editing_circuit = self.state["layer"] == "attachments"
            att = p.segment("x0", "y0", "x1", "y1", source=self.src_att, line_color="colour",
                            line_width=3 if editing_circuit else 2,
                            line_alpha=0.9 if editing_circuit else 0.55, line_dash="solid")
            p.add_tools(HoverTool(renderers=[att], visible=False,
                                  tooltips=[("part", "@id"), ("", "@text")]))
            p.add_layout(LabelSet(x="lx", y="ly", text="text", source=self.src_att,
                                  text_align="center", text_baseline="middle",
                                  text_font_size="10px", text_color="colour",
                                  background_fill_color="#ffffff",
                                  background_fill_alpha=0.85))
            p.scatter("x", "y", source=self.src_att_nodes, size=8, fill_color="#ffffff",
                      line_color=C_NODE, line_width=2)
            p.add_layout(LabelSet(x="x", y="y", text="name", source=self.src_att_nodes,
                                  x_offset=6, y_offset=6, text_font_size="10px",
                                  text_color=C_NODE))

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

        # reshape handles, drawn only for the Reshape tool: rectangles' corners, and a
        # drawn shape's vertices (squares), edges (circles) and body (a diamond)
        resize = tool == "resize" and layer in DRAWABLE
        hnd = p.scatter("x", "y", source=self.src_handles, marker="square", size=9,
                        fill_color="#ffffff", line_color=C_HANDLE, line_width=2,
                        visible=resize, nonselection_fill_alpha=1.0)
        vtx = p.scatter("x", "y", source=self.src_vtx, marker="square", size=10,
                        fill_color="#ffffff", line_color=C_HANDLE, line_width=2,
                        visible=resize, nonselection_fill_alpha=1.0,
                        selection_fill_color=C_HANDLE)
        edg = p.scatter("x", "y", source=self.src_edge, marker="circle", size=10,
                        fill_color="#ffffff", line_color=C_EDGE, line_width=2,
                        visible=resize, nonselection_fill_alpha=1.0,
                        selection_fill_color=C_EDGE)
        mv = p.scatter("x", "y", source=self.src_move, marker="diamond", size=15,
                       fill_color=C_HANDLE, line_color="#ffffff", line_width=1.5,
                       visible=resize, nonselection_fill_alpha=1.0)
        p.add_tools(HoverTool(renderers=[vtx, edg, mv], visible=False,
                              tooltips=[("", "@label"), ("", "@what")]))

        # the shape being drawn: its vertices so far, and the edge that will close it
        drawing = tool == "shape" and layer in DRAWABLE
        p.line("x", "y", source=self.src_pending, line_color=C_PENDING, line_width=2,
               line_dash="dashed", visible=drawing)
        p.scatter("x", "y", source=self.src_pending, size=8, fill_color=C_PENDING,
                  line_color="#ffffff", visible=drawing)

        pan, wheel = PanTool(), WheelZoomTool()
        zoom = BoxZoomTool(match_aspect=True)
        p.add_tools(pan, wheel, zoom, ResetTool(), SaveTool())
        p.toolbar.active_scroll = wheel
        edit_tool = None
        # clicks: a vertex in Draw shape, and a double click closes the shape (or, in
        # Reshape, adds a vertex on an edge or smooths the curve through a vertex)
        p.on_event(Tap, self._on_tap)
        p.on_event(DoubleTap, self._on_double_tap)
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
            edit_tool = PointDrawTool(renderers=[hnd, vtx, edg, mv], empty_value="",
                                      add=False, drag=True,
                                      description="Reshape: drag a corner, vertex, edge or "
                                                  "the whole shape")
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
        #: the box tool edits rectangles only; a drawn window is its own glyph
        ws = [w for w in s.windows if w.shape == "rect"]
        self._win_rect_index = [k for k, w in enumerate(s.windows) if w.shape == "rect"]
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
        # generated windows: their cuts across the domain as smooth curves (the
        # domain's own outline draws the rest of each one's edge), and a label
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
            boxes = geo.mask_to_boxes(~act)                # the cells outside the domain
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
        pend = self.state.get("pending") or []
        closing = pend + [pend[0]] if len(pend) > 2 else pend
        self._set(self.src_pending, dict(x=[p[0] for p in closing],
                                         y=[p[1] for p in closing]))
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
            elif r.shape == "curve":
                ring = r.outline.ring()
                xs.append([[ring[:, 0].tolist()]])
                ys.append([[ring[:, 1].tolist()]])
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
        self._sync_attachments()
        self._sync_tables()

    def _boxes(self) -> list[tuple[int, geo.Box]]:
        """The boxes the Resize tool works on: (index into the layer, box)."""
        s = self.wb.spec
        if self.state["layer"] == "windows":
            return [(k, (w.x0, w.y0, w.nx, w.ny)) for k, w in enumerate(s.windows)
                    if w.shape == "rect"]
        if self.state["layer"] == "regions":
            return [(k, (r.x0, r.y0, r.nx, r.ny)) for k, r in enumerate(s.regions)
                    if r.shape == "rect"]
        return []

    def curves(self) -> list[tuple[str, Outline]]:
        """The drawn shapes of the active layer, each with its reference."""
        s, layer = self.wb.spec, self.state["layer"]
        out: list[tuple[str, Outline]] = []
        if layer == "domain":
            if s.domain.outline is not None:
                out.append(("domain:outline", s.domain.outline))
            out += [(f"domain:hole:{h}", o) for h, o in enumerate(s.domain.holes)]
        elif layer == "windows":
            for i, w in enumerate(s.windows):
                if w.shape == "curve":
                    out.append((f"windows:{i}", w.outline))
                    out += [(f"windows:{i}:hole:{h}", o) for h, o in enumerate(w.holes)]
        elif layer == "regions":
            out += [(f"regions:{i}", r.outline) for i, r in enumerate(s.regions)
                    if r.shape == "curve"]
        return out

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
            c = np.mean(np.asarray(o.points, dtype=float), axis=0)
            m["x"].append(float(c[0]))
            m["y"].append(float(c[1]))
            m["ref"].append(ref)
            m["label"].append(name)
            m["what"].append("drag to move the whole shape")
        #: what the handles showed, so a drag can be told from a Backspace
        self._handle_rows = {"vtx": [(r, k, x, y) for x, y, r, k in
                                     zip(v["x"], v["y"], v["ref"], v["k"])],
                             "edge": [(r, k, x, y) for x, y, r, k in
                                      zip(e["x"], e["y"], e["ref"], e["k"])],
                             "move": [(r, x, y) for x, y, r in zip(m["x"], m["y"], m["ref"])]}
        self._set(self.src_vtx, v)
        self._set(self.src_edge, e)
        self._set(self.src_move, m)

    def _sync_warnings(self) -> None:
        s, d = self.wb.spec, self.wb.spec.domain
        pieces = s.coupling.style in ("C", "D")
        plain = geo.is_plain(s)
        act = geo.domain_mask(d)
        ov_cells: list[geo.Box] = []
        if s.windows and pieces:
            # styles C and D: pieces that meet along faces; an overlap is the error
            # and there is no ramp to be at full weight in
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
            n_bad, n_gap, n_thin = int((count > 1).sum()), int(empty.sum()), 0
        elif s.windows:
            an = geo.analyse_case(s)
            bad, gap = geo.mask_to_boxes(an.ramp_only), geo.mask_to_boxes(an.uncovered)
            ov, cps = an.overlaps, an.cross_points
            if not plain:
                # a drawn overlap is drawn cell by cell, not as its bounding box
                ov_cells = geo.mask_to_boxes(an.overlap_mask)
                ov = []
            n_bad, n_gap = int(an.ramp_only.sum()), int(an.uncovered.sum())
            n_thin = len(an.thin_pairs)
        else:
            bad, gap, ov, cps = [], geo.mask_to_boxes(act), [], []
            n_bad, n_gap, n_thin = 0, int(act.sum()), 0
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
        ramp = s.coupling.ramp_cells
        parts = []
        if n_bad and pieces:
            parts.append(f"<span style='color:{C_BAD}'><b>{n_bad:,}</b> cells in more than "
                         f"one window</span> (red hatching; style {s.coupling.style}'s "
                         f"pieces meet along faces and do not overlap)")
        elif n_bad:
            parts.append(f"<span style='color:{C_BAD}'><b>{n_bad:,}</b> cells with no window "
                         f"at full weight</span> (red hatching; overlap by at least "
                         f"{2 * ramp} cells where windows meet)")
        if n_gap:
            parts.append(f"<span style='color:{C_GAP}'><b>{n_gap:,}</b> cells in no "
                         f"window</span> (grey hatching)")
        if n_thin:
            parts.append(f"<b>{n_thin}</b> thin seam{'s' * (n_thin > 1)}")
        if not pieces:
            parts.append(f"{len(cps)} cross-point{'s' * (len(cps) != 1)} (diamonds)")
        ok = not (n_bad or n_gap)
        lead = ""
        if ok and pieces:
            lead = ("<b style='color:#15803d'>The windows are pieces that tile the domain "
                    f"without overlapping (style {s.coupling.style}).</b> ")
        elif ok:
            lead = ("<b style='color:#15803d'>Windows cover the domain, and every cell has a "
                    "window at full weight.</b> ")
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
        self._set(self.src_bc_drawn, dr)
        for b in s.boundaries:
            if b.drawn:
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
            # the label sits just inside the domain, beside its segment
            rows["lx"].append((x0 + x1) / 2 - 2.2 * dx)
            rows["ly"].append((y0 + y1) / 2 - 2.2 * dy)
            rows["angle"].append(math.pi / 2 if b.edge in ("left", "right") else 0.0)
            val = "" if b.value is None else f" = {b.value:g}"
            rows["text"].append(f"{b.kind}{val}")
        self._set(self.src_bc, rows)

    def circuit_nodes(self) -> dict[str, tuple[float, float]]:
        """Each circuit node's place on the canvas: an electrode just outside the
        middle of its edge segment, any other node in a row below the domain."""
        s, d = self.wb.spec, self.wb.spec.domain
        off = max(3.0, 0.02 * max(d.nx, d.ny))
        pos: dict[str, tuple[float, float]] = {}
        for b in s.boundaries:
            if b.kind != "electrode" or b.drawn:      # the circuit family runs on the grid's edges
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
            (xa, ya), (xb, yb) = pos[a.a], pos[a.b]
            key = tuple(sorted((a.a, a.b)))
            k = seen.get(key, 0)
            seen[key] = k + 1
            # parts between the same two nodes: labels stepped apart along the wire
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

    def add_part(self, kind: str) -> None:
        """A battery or a resistor, wired between the first electrode and a new free
        node, to be edited in the table."""
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
        self.wb.edit(apply, f"added {kind} {pid}")

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
        #: the box tool sees the rectangles only; a drawn window keeps its place
        before = {w.id: w for w in self.wb.spec.windows if w.shape == "rect"}
        taken = {w.id for w in self.wb.spec.windows}
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
        upd = {w.id: w for w in result}

        def apply(c: CaseSpec):
            out = [w if w.shape == "curve" else upd[w.id] for w in c.windows
                   if w.shape == "curve" or w.id in upd]
            c.windows = out + [upd[i] for i in drawn]
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
            for r in drawn:
                with_material(c, r.material)
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
    # drawn shapes: drawing one, the ready-made ones, and the handles
    # ------------------------------------------------------------------
    def _snap_pt(self, x: float, y: float) -> tuple[float, float]:
        d = self.wb.spec.domain
        step = self._snap()
        return (float(geo.snap_value(x, step, d.nx)), float(geo.snap_value(y, step, d.ny)))

    def _sync_pending(self) -> None:
        pend = self.state.get("pending") or []
        closing = pend + [pend[0]] if len(pend) > 2 else pend
        self._set(self.src_pending, dict(x=[p[0] for p in closing],
                                         y=[p[1] for p in closing]))

    def _on_tap(self, ev) -> None:
        """Draw shape: a click places a vertex (snapped)."""
        st = self.state
        if st["tool"] != "shape" or st["layer"] not in DRAWABLE:
            return
        p = self._snap_pt(ev.x, ev.y)
        pend = [tuple(q) for q in st.get("pending") or []]
        if pend and pend[-1] == p:
            return
        pend.append(p)
        st["pending"] = pend
        self._sync_pending()

    def _on_double_tap(self, ev) -> None:
        st = self.state
        if st["layer"] not in DRAWABLE:
            return
        if st["tool"] == "shape":
            self.close_shape()
        elif st["tool"] == "resize":
            self.double_click_handle(ev.x, ev.y)

    def undo_vertex(self) -> None:
        self.state["pending"] = list(self.state.get("pending") or [])[:-1]
        self._sync_pending()

    def cancel_shape(self) -> None:
        self.state["pending"] = []
        self._sync_pending()

    def close_shape(self) -> None:
        """The vertices placed so far become a closed shape on the active layer."""
        st = self.state
        pts: list[tuple[float, float]] = []
        # a double click lands as a tap on the last point too
        for p in (tuple(q) for q in st.get("pending") or []):
            if not pts or pts[-1] != p:
                pts.append(p)
        if len(pts) > 1 and pts[-1] == pts[0]:
            pts.pop()
        if len(pts) < 3:
            self.wb.notify("info", "place at least three vertices, then double-click (or "
                                   "Close shape)")
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

    def add_drawn(self, o: Outline) -> None:
        """A drawn shape added to the active layer: the domain's outline (replacing
        the whole grid, or the outline before), a hole in the domain, a window or a
        region.  A drawn domain edge gets its own boundary, the family's default."""
        s, layer = self.wb.spec, self.state["layer"]
        if layer == "domain" and self.state.get("draw_as") == "hole":
            h = len(s.domain.holes)

            def apply(c):
                c.domain.holes.append(o)
                rebase_boundaries(c, f"hole{h}", len(o.points), None)
                follow_the_domain(c)
            label = f"cut hole {h} in the domain"
        elif layer == "domain":
            def apply(c):
                if c.domain.outline is None:
                    # the grid's edges no longer bound the domain
                    c.boundaries = [b for b in c.boundaries if b.drawn]
                c.domain.outline = o
                rebase_boundaries(c, "outline", len(o.points), None)
                follow_the_domain(c)
            label = "drew the domain's outline"
        elif layer == "windows":
            wid = _next_id("F", [w.id for w in s.windows])

            def apply(c):
                c.windows.append(Window(id=wid, shape="curve", outline=o))
            label = f"drew window {wid}"
        else:
            rid = _next_id("R", [r.id for r in s.regions])
            mat = self.state["material"] or "material-1"

            def apply(c):
                c.regions.append(Region(id=rid, material=mat, shape="curve", outline=o))
                with_material(c, mat)
            label = f"drew region {rid} ({mat})"
        if not self.wb.edit(apply, label):
            self.sync()

    def add_ready(self, what: str) -> None:
        """A ready-made circle, ellipse or hexagon, in the middle of the view (for the
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
                                           "the shape with its diamond instead")
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
        """An edge's handle dragged (the edge bends into an arc through the handle's
        distance from its chord, or straightens) or removed (Backspace: straight)."""
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
                self._edit_shape(r, Outline(points=pts, edges=ks, bulge=bs), None,
                                 f"{shape_label(spec, r)}: edge {k} {what}")
                return
        self.sync()

    def _on_move(self, _attr, _old, new) -> None:
        """A shape's diamond dragged (the shape moves, by whole snap steps) or removed
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

    def translate_shape(self, ref: str, dx: float, dy: float) -> None:
        """Move a drawn shape; a window's or the domain's holes move with its outline."""
        spec = self.wb.spec

        def moved(o: Outline) -> Outline:
            return Outline(points=shapes.translate(o.points, dx, dy), edges=o.kinds(),
                           bulge=o.bulges())

        def apply(c):
            set_outline(c, ref, moved(get_outline(c, ref)))
            p = ref.split(":")
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
            # the highest indices first, so earlier ones keep their places
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
                    del c.windows[int(p[1])]
                else:
                    del c.regions[int(p[1])]
        if not self.wb.edit(apply, f"deleted {', '.join(names)}"):
            self.sync()

    def double_click_handle(self, x: float, y: float) -> None:
        """Reshape: a double click on an edge's circle adds a vertex there; on a
        vertex's square it makes the curve smooth through that vertex (both its
        edges splines), or a corner again (both straight)."""
        x0, x1, _y0, _y1 = self.ranges()
        tol = max(1.5, 0.02 * (x1 - x0))

        def nearest(rows):
            best, dist = None, tol
            for row in rows:
                d = math.hypot(row[2] - x, row[3] - y)
                if d <= dist:
                    best, dist = row, d
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

    def set_layout(self, follow: bool, cut: str, along: int, across: int) -> None:
        """Windows that follow the domain (generated from its shape, `layout.py`), or
        not (the windows there now stay, as the case's own)."""
        if not follow:
            if self.wb.spec.layout is None:
                return

            def apply(c):
                c.layout = None
            self.wb.edit(apply, "the windows no longer follow the domain; they stay as "
                                "they are")
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
        self.wb.edit(apply, "the domain is the whole grid again")

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
            "domain": pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=150,
                                           layout="fit_data_table", selectable=True,
                                           sizing_mode="stretch_width"),
            "edges": pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=220,
                                          layout="fit_data_table", selectable=True,
                                          sizing_mode="stretch_width",
                                          hidden_columns=["ref"]),
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
            "attachments": pn.widgets.Tabulator(pd.DataFrame(), show_index=False, height=200,
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
        if "layer" in kw or "tool" in kw:
            self.state["pending"] = []             # a half-drawn shape does not carry over
        self.state.update(kw)
        self.wb.log("geometry: " + ", ".join(f"{k} = {v}" for k, v in kw.items()))
        self.wb.show("geometry", keep_view=True)

    def _table_frames(self) -> dict[str, pd.DataFrame]:
        s, d = self.wb.spec, self.wb.spec.domain
        plain = geo.is_plain(s)
        act = geo.domain_mask(d)

        def n_cells(w) -> int:
            if plain:
                return w.nx * w.ny
            return int((geo.window_mask(w, d.nx, d.ny) & act).sum())
        kind = {"rect": "rectangle", "curve": "drawn", "cells": "generated"}
        win = pd.DataFrame([dict(id=w.id, shape=kind[w.shape],
                                 x0=w.x0, y0=w.y0, width=w.nx, height=w.ny, cells=n_cells(w))
                            for w in s.windows],
                           columns=["id", "shape", "x0", "y0", "width", "height", "cells"])
        reg = pd.DataFrame([dict(order=k + 1, id=r.id, material=r.material,
                                 shape="drawn" if r.shape == "curve" else r.shape,
                                 x0=r.x0, y0=r.y0, width=r.nx, height=r.ny)
                            for k, r in enumerate(s.regions)],
                           columns=["order", "id", "material", "shape", "x0", "y0", "width",
                                    "height"])
        dom_rows = []
        if d.outline is None and not d.holes:
            dom_rows.append(dict(shape="the whole grid", vertices=4, edges="4 lines",
                                 cells=d.nx * d.ny))
        if d.outline is not None:
            dom_rows.append(dict(shape="outline", vertices=len(d.outline.points),
                                 edges=_edge_summary(d.outline), cells=int(act.sum())))
        for h, o in enumerate(d.holes):
            removed = int(geo.shape_mask(o, (), d.nx, d.ny).sum())
            dom_rows.append(dict(shape=f"hole {h}", vertices=len(o.points),
                                 edges=_edge_summary(o), cells=-removed))
        dom = pd.DataFrame(dom_rows, columns=["shape", "vertices", "edges", "cells"])
        edge_rows = []
        for ref, o in self.curves():
            pts, ks, bs = o.points, o.kinds(), o.bulges()
            for k in range(len(pts)):
                a, b = pts[k], pts[(k + 1) % len(pts)]
                edge_rows.append(dict(ref=ref, shape=shape_label(s, ref), edge=k, kind=ks[k],
                                      bulge=round(bs[k], 6) if ks[k] == "arc" else 0.0,
                                      start=f"({a[0]:g}, {a[1]:g})",
                                      end=f"({b[0]:g}, {b[1]:g})"))
        edges = pd.DataFrame(edge_rows, columns=["ref", "shape", "edge", "kind", "bulge",
                                                 "start", "end"])
        dev = pd.DataFrame([dict(id=v.id, kind=v.kind, x_D=v.x, y_D=v.y,
                                 diameter_D=v.diameter, yaw_deg=v.yaw_deg) for v in s.devices],
                           columns=["id", "kind", "x_D", "y_D", "diameter_D", "yaw_deg"])
        bnd = pd.DataFrame([dict(id=b.id, edge=b.edge,
                                 start="" if b.drawn else b.start,
                                 stop="" if b.drawn else (
                                     geo.edge_length(b.edge, d.nx, d.ny) if b.stop is None
                                     else b.stop), kind=b.kind,
                                 value="" if b.value is None else f"{b.value:g}")
                            for b in s.boundaries],
                           columns=["id", "edge", "start", "stop", "kind", "value"])
        att = pd.DataFrame([dict(id=a.id, kind=a.kind, value=a.value,
                                 unit="V" if a.kind == "battery" else "ohm",
                                 internal_ohm=a.internal, a=a.a, b=a.b)
                            for a in s.attachments],
                           columns=["id", "kind", "value", "unit", "internal_ohm", "a", "b"])
        return {"domain": dom, "edges": edges, "windows": win, "regions": reg, "devices": dev,
                "boundaries": bnd, "attachments": att}

    def _sync_tables(self) -> None:
        fam = _family(self.wb.spec)
        kinds = list(fam.boundary_kinds) if fam and fam.boundary_kinds else \
            [k.id for k in registry.BOUNDARY_KINDS]
        fixed = bool(fam and fam.fixed_boundaries)
        d = self.wb.spec.domain
        edge_names = ((geo.drawn_edge_names(d) if (d.outline is not None or d.holes) else [])
                      + ([] if d.outline is not None else list(geo.EDGES)))
        editors = {
            "domain": {c: None for c in ("shape", "vertices", "edges", "cells")},
            "edges": {"shape": None, "edge": None, "start": None, "end": None,
                      "kind": {"type": "list", "values": list(shapes.EDGE_KINDS)}},
            "windows": {"cells": None, "shape": None},
            "regions": {"order": None, "shape": None},
            "devices": {"kind": None},
            "boundaries": ({c: None for c in ("id", "edge", "start", "stop", "kind", "value")}
                           if fixed else
                           {"edge": {"type": "list", "values": edge_names},
                            "kind": {"type": "list", "values": kinds}}),
            "attachments": {"kind": {"type": "list", "values": ["battery", "resistor"]},
                            "unit": None},
        }
        for key, df in self._table_frames().items():
            t = self.tables[key]
            t.editors = editors[key]
            t.value = df

    def _on_table_edit(self, key: str, e) -> None:
        row, col, val = e.row, e.column, e.value
        spec = self.wb.spec
        if key == "edges":
            self._on_edge_table(row, col, val)
            return
        drawn = None
        if key in ("windows", "regions") and 0 <= row < len(getattr(spec, key)):
            item = getattr(spec, key)[row]
            if item.shape == "cells" and col in ("x0", "y0", "width", "height"):
                self.wb.notify("info", f"window {item.id} is generated from the domain's shape; "
                                       f"change how the windows follow it above the table, or "
                                       f"untick 'Windows follow the domain' and draw your own")
                self._sync_tables()
                return
            if item.shape == "curve" and col in ("x0", "y0", "width", "height"):
                drawn = (item, f"{key}:{row}")
        if drawn is not None:
            item, ref = drawn
            if col in ("width", "height"):
                self.wb.notify("info", f"{item.id} is drawn: its size is its outline's; "
                                       f"reshape it on the canvas (Reshape)")
                self._sync_tables()
                return
            try:
                delta = float(val) - float(getattr(item, col))
            except (TypeError, ValueError):
                self._sync_tables()
                return
            self.translate_shape(ref, delta if col == "x0" else 0.0,
                                 delta if col == "y0" else 0.0)
            return
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
                    if field == "material":
                        with_material(c, str(val))
                label = f"region {reg.id}: {col} = {val}"
            elif key == "devices":
                field = {"id": "id", "x_D": "x", "y_D": "y", "diameter_D": "diameter",
                         "yaw_deg": "yaw_deg"}[col]
                vid = spec.devices[row].id

                def apply(c):
                    setattr(c.devices[row], field, str(val) if field == "id" else float(val))
                label = f"device {vid}: {col} = {val}"
            elif key == "attachments":
                field = {"id": "id", "kind": "kind", "value": "value",
                         "internal_ohm": "internal", "a": "a", "b": "b"}[col]
                pid = spec.attachments[row].id

                def apply(c):
                    setattr(c.attachments[row], field,
                            float(val) if field in ("value", "internal") else str(val).strip())
                    # pydantic does not validate on assignment: check the part as a whole
                    Attachment.model_validate(c.attachments[row].model_dump())
                label = f"part {pid}: {col} = {val}"
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

    def _on_edge_table(self, row: int, col: str, val) -> None:
        """The Edges table: an edge's kind (line, arc, spline) or an arc's bulge."""
        df = self.tables["edges"].value
        if not (0 <= row < len(df)) or col not in ("kind", "bulge"):
            self._sync_tables()
            return
        ref, k = str(df.iloc[row]["ref"]), int(df.iloc[row]["edge"])
        o = get_outline(self.wb.spec, ref)
        try:
            if col == "kind":
                pts, ks, bs = shapes.set_kind(o.points, o.kinds(), o.bulges(), k, str(val))
            else:
                b = float(val)
                if abs(b) > shapes.MAX_BULGE:
                    raise ValueError(f"at most {shapes.MAX_BULGE:g}")
                pts, ks, bs = shapes.set_kind(o.points, o.kinds(), o.bulges(), k,
                                              "arc" if b else "line", b if b else None)
        except (TypeError, ValueError) as exc:
            self.wb.notify("error", f"not applied: {col} = {val!r}: {exc}")
            self._sync_tables()
            return
        self._edit_shape(ref, Outline(points=pts, edges=ks, bulge=bs), None,
                         f"{shape_label(self.wb.spec, ref)}: edge {k} {col} = {val}")

    # ------------------------------------------------------------------
    # buttons
    # ------------------------------------------------------------------
    def _selected(self, key: str) -> list[int]:
        """Rows selected in the table or on the canvas, as indices into the layer."""
        rows = set(self.tables[key].selection or [])
        if key == "windows":
            rows |= {self._win_rect_index[i] for i in self.src_win.selected.indices
                     if i < len(self._win_rect_index)}
        elif key == "regions":
            rows |= {self._rect_index[i] for i in self.src_reg_edit.selected.indices
                     if i < len(self._rect_index)}
        elif key == "devices":
            rows |= set(self.src_dev.selected.indices)
        return sorted(rows)

    def delete_selected(self) -> None:
        key = self.state["layer"]
        if key == "domain":
            # the table's rows: the outline (when drawn), then the holes
            d = self.wb.spec.domain
            refs = []
            for r in self.tables["domain"].selection or []:
                if d.outline is not None:
                    refs.append("domain:outline" if r == 0 else f"domain:hole:{r - 1}")
                elif r < len(d.holes):
                    refs.append(f"domain:hole:{r}")
            if not refs:
                self.wb.notify("info", "select the outline or a hole in the table first")
                return
            self.delete_shapes(refs)
            return
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
            edge = "left"
            if d.outline is not None or d.holes:
                have = {b.edge for b in s.boundaries}
                free = [n for n in geo.drawn_edge_names(d) if n not in have]
                if not free and d.outline is not None:
                    self.wb.notify("info", "every drawn edge has its condition; change it in "
                                           "the table (add a vertex to split an edge)")
                    return
                edge = free[0] if free else "left"
            new = Boundary(id=wid, edge=edge, kind=kind)

        def apply(c):
            getattr(c, key).append(new)
            if key == "regions":
                with_material(c, new.material)
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
        draw_bar = []
        if self.state["tool"] == "shape" and key in DRAWABLE:
            edges_sel = pn.widgets.RadioButtonGroup(
                name="New edges", options={"straight edges": "line", "smooth (spline)": "spline"},
                value=self.state.get("draw_edges", "line"), button_type="primary",
                button_style="outline")
            edges_sel.param.watch(lambda e: self.state.update(draw_edges=e.new), "value")
            draw_bar.append(edges_sel)
            if key == "domain":
                as_sel = pn.widgets.RadioButtonGroup(
                    name="Draw", options={"the outline": "outline", "a hole": "hole"},
                    value=self.state.get("draw_as", "outline"), button_type="primary",
                    button_style="outline")
                as_sel.param.watch(lambda e: self.state.update(draw_as=e.new), "value")
                draw_bar.append(as_sel)

            def dbtn(name, fn, kind="default", width=None):
                b = pn.widgets.Button(name=name, button_type=kind, width=width or 120)
                b.on_click(lambda _e: fn())
                draw_bar.append(b)
            dbtn("Close shape", self.close_shape, "primary")
            dbtn("Undo vertex", self.undo_vertex)
            dbtn("Cancel", self.cancel_shape, width=90)
            dbtn("Circle", lambda: self.add_ready("circle"), width=90)
            dbtn("Ellipse", lambda: self.add_ready("ellipse"), width=90)
            dbtn("Hexagon", lambda: self.add_ready("hexagon"), width=100)
        if key == "domain":
            btn("Reset to the whole grid", self.reset_domain, width=190)
            btn("Delete selected", self.delete_selected)
            if fam is not None and not fam.drawn_shapes:
                extra.append(pn.pane.Alert(
                    f"The **{fam.label}** family runs on rectangles only, so the check "
                    f"refuses a drawn domain or window for it: {fam.drawn_why}. The families "
                    f"that run on drawn shapes: "
                    + "; ".join(f.label for f in registry.FAMILIES if f.drawn_shapes) + ".",
                    alert_type="warning", sizing_mode="stretch_width"))
        elif key == "windows":
            lay = s.layout
            follow = pn.widgets.Checkbox(name="Windows follow the domain", value=lay is not None,
                                         width=210)
            cut = pn.widgets.Select(options={"cut along its length": "along",
                                             "one per material": "materials"},
                                    value=lay.cut if lay else "along", width=170)
            n_along = pn.widgets.IntInput(name="along", value=lay.along if lay else
                                          max(2, len(s.windows)), start=1, end=32, width=80)
            n_across = pn.widgets.IntInput(name="across", value=lay.across if lay else 1,
                                           start=1, end=8, width=80)

            def relayout(_e=None):
                self.set_layout(follow.value, cut.value, n_along.value, n_across.value)
            for w in (follow, cut, n_along, n_across):
                w.param.watch(relayout, "value")
            extra.append(_row(follow, cut, n_along, n_across, pn.pane.Markdown(
                "<small>Generated from the domain's shape: cut at level curves of its own "
                "coordinates (along its length, and across it), or one piece per material, "
                "and generated again whenever the domain changes. Editing a window by hand "
                "makes the windows yours.</small>", width=420)))
            btn("Generate a tiling...", wb.dialog_tiling, "primary", 170)
            btn("Add window", self.add_shape)
            btn("Delete selected", self.delete_selected)
        elif key == "regions":
            names = material_names(s)
            #: the shell's generic default names no material any family defines
            if self.state["material"] in ("", "material-1") and names:
                self.state["material"] = names[0]
            mat = pn.widgets.AutocompleteInput(
                placeholder="material for new regions", value=self.state["material"],
                options=names, restrict=False, min_characters=0, case_sensitive=False,
                width=190)
            mat.param.watch(lambda e: self.state.update(material=(e.new or "").strip()),
                            "value")
            buttons.append(mat)
            btn("Add region", self.add_shape)
            btn("Delete selected", self.delete_selected)
            btn("Move up the stack", lambda: self.restack(+1), width=160)
            btn("Move down the stack", lambda: self.restack(-1), width=170)
            if fam is not None and "regions" not in fam.layers:
                readers = [f.label for f in registry.FAMILIES
                           if "regions" in f.layers and f.status == "ready-to-wire"]
                extra.append(pn.pane.Alert(
                    f"The **{fam.label}** family does not read material regions: its solver "
                    f"would ignore them. The families that do: {'; '.join(readers)}.",
                    alert_type="warning", sizing_mode="stretch_width"))
        elif key == "devices":
            btn("Add rotor", self.add_shape)
            btn("Delete selected", self.delete_selected)
        elif key == "attachments":
            reads = fam is not None and "attachments" in fam.layers
            btn("Add battery", lambda: self.add_part("battery"), "primary", 140).disabled = \
                not reads
            btn("Add resistor", lambda: self.add_part("resistor"), width=140).disabled = \
                not reads
            btn("Delete selected", self.delete_selected)
            if not reads:
                readers = [f.label for f in registry.FAMILIES
                           if "attachments" in f.layers and f.status == "ready-to-wire"]
                extra.append(pn.pane.Alert(
                    f"The **{fam.label if fam else '?'}** family does not read lumped parts: "
                    f"its solver would ignore a circuit. The families that do: "
                    f"{'; '.join(readers)}.", alert_type="warning",
                    sizing_mode="stretch_width"))
            else:
                extra.append(pn.pane.Markdown(
                    "<small>A node named like an **electrode** boundary is that electrode; "
                    "any other name is a free node of the circuit, and `ground` is 0 V "
                    "(with no ground, the first electrode is). Mark electrodes on the "
                    "Boundaries layer.</small>", sizing_mode="stretch_width"))
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

        edges_part = []
        if key in DRAWABLE and self.curves():
            edges_part = [
                pn.pane.Markdown("**Edges** of the drawn shapes on this layer: each is a "
                                 "straight `line`, a circular `arc` (its bulge is tan of a "
                                 "quarter of its angle; 1 is a semicircle, negative bows the "
                                 "other way) or a `spline` through its vertices. Set the kind "
                                 "or the bulge here, or with Reshape's handles.",
                                 margin=(6, 10, 0, 10), styles={"font-size": "13px"}),
                self.tables["edges"]]
        return pn.Column(
            _row(self.layer_sel, self.tool_sel, self.snap_sel),
            self.help,
            *([_row(*draw_bar)] if draw_bar else []),
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
            *edges_part,
            pn.pane.Markdown("**Geometry issues**", margin=(6, 10, 0, 10)),
            self.issues,
            sizing_mode="stretch_width")


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
    name; any other name is left for the Physics step (and the check says so)."""
    if name in c.materials:
        return
    fam = _family(c)
    lib = fam.material_library() if fam is not None else {}
    if name in lib:
        c.materials[name] = dict(lib[name])


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


def _row(*objs):
    return pn.FlexBox(*objs, flex_wrap="wrap", gap="8px 12px", align_items="center",
                      sizing_mode="stretch_width")


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


def shape_label(c: CaseSpec, ref: str) -> str:
    p = ref.split(":")
    if p[0] == "domain":
        return "the domain's outline" if p[1] == "outline" else f"hole {p[2]}"
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
            # the family's own choice: a river drawn to the grid's left edge enters
            # there, a wind farm's inner edges are walls (`spec.derived_kind`)
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
    if style == "C":
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


__all__ = ["GeometryEditor", "LAYERS", "TOOLS", "material_colours"]
