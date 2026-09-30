"""The Model tab's panels: the layer rail, the tool bar, the footer and the inspector.

`editor.GeometryEditor` owns the canvas and every edit; this module builds the
widgets around it from the case, each one applying its change through
`Workbench.edit` (validated, undoable) and redrawn from the case afterwards.

The inspector shows what is selected and only its settings; with nothing
selected it shows the layer's own: how the windows are cut and joined, the
physics and the grid, a material's properties.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Callable

import panel as pn

from . import geometry as geo
from . import registry
from . import shapes
from .spec import Boundary

if TYPE_CHECKING:                                             # pragma: no cover
    from .editor import GeometryEditor

ACCENT = "#0e6874"

#: left-aligned, full-width buttons for the rail
#: the Bootstrap design's own rules are more specific than a widget's stylesheet,
#: so the colours here are !important (seen in the page: the design's blue won)
RAIL_CSS = """
.bk-btn { justify-content: flex-start !important; text-align: left; font-size: 14px;
          padding: 6px 10px; border-radius: 8px !important; }
.bk-btn-light { background: transparent !important; border-color: transparent !important;
                color: #1e293b !important; }
.bk-btn-light:hover { background: #f1f5f9 !important; }
.bk-btn-primary { background: #e3f1f2 !important; border-color: #e3f1f2 !important;
                  color: #0b5560 !important; font-weight: 600; }
.bk-btn-primary:hover { background: #d4ebed !important; }
"""
ITEM_CSS = """
.bk-btn { justify-content: flex-start !important; text-align: left; font-size: 13px;
          padding: 5px 10px; border-radius: 8px !important; white-space: normal; }
.bk-btn-light { background: transparent !important; border-color: transparent !important;
                color: #0f172a !important; }
.bk-btn-light:hover { background: #f1f5f9 !important; }
.bk-btn-primary { background: #f1f9fa !important; border-color: #9ccfd4 !important;
                  color: #0f172a !important; font-weight: 600; }
"""
TOOL_CSS = """
.bk-btn-group .bk-btn { font-size: 13px; padding: 6px 12px; }
.bk-btn-primary.bk-active { background: #0e6874 !important; border-color: #0e6874 !important;
                            color: #ffffff !important; }
.bk-btn-primary { color: #0e6874 !important; border-color: #9ccfd4 !important; }
"""

SECTION = ("font-size: 11px; font-weight: 600; letter-spacing: 0.6px; text-transform: uppercase; "
           "color: #5b6b7c; margin: 0")


# ---------------------------------------------------------------------------
# the whole tab
# ---------------------------------------------------------------------------


def model_layout(ed: "GeometryEditor"):
    rail = pn.Column(_label("Layers"), ed.layer_col, _divider(), ed.rail_items,
                     width=236, margin=(8, 4, 8, 8),
                     styles={"background": "#ffffff", "border-radius": "10px",
                             "border": "1px solid #e2e8f0"})
    center = pn.Column(toolbar(ed), ed.fig_pane, footer(ed), sizing_mode="stretch_width",
                       margin=(8, 4, 8, 4))
    side = pn.Column(ed.inspector, width=330, margin=(8, 8, 8, 4),
                     styles={"background": "#ffffff", "border-radius": "10px",
                             "border": "1px solid #e2e8f0"})
    return pn.Row(rail, center, side, sizing_mode="stretch_width")


def layer_buttons(ed: "GeometryEditor") -> list:
    from .editor import LAYERS, layers_for
    s = ed.wb.spec
    out = []
    for key in layers_for(s):
        label = f"{LAYERS[key]}{_layer_count(ed, key)}"
        b = pn.widgets.Button(name=label, sizing_mode="stretch_width", margin=(1, 4),
                              button_type="primary" if key == ed.state["layer"] else "light",
                              stylesheets=[RAIL_CSS])
        b.on_click(lambda _e, key=key: ed._switch(layer=key))
        out.append(b)
    return out


def _layer_count(ed: "GeometryEditor", key: str) -> str:
    s = ed.wb.spec
    d = s.domain
    if key == "domain":
        return "  · drawn" if (d.outline is not None or d.holes) else "  · the grid"
    if key == "regions":
        return f"  · {len({r.material for r in s.regions})}"
    if key == "boundaries":
        missing = sum(1 for i in ed.wb.issues() if i.severity == "error"
                      and ("boundary" in i.message or "condition" in i.message))
        return "  · needs work" if missing else f"  · {len(s.boundaries)}"
    if key == "windows":
        return (f"  · {len(s.windows)} auto" if s.layout is not None
                else f"  · {len(s.windows)}")
    if key == "physics":
        return f"  · {s.run.mode}"
    if key == "devices":
        return f"  · {len(s.devices)}"
    if key == "attachments":
        return f"  · {len(s.attachments)}"
    return ""


def toolbar(ed: "GeometryEditor"):
    from .editor import DRAWABLE, TOOLS
    st = ed.state
    layer = st["layer"]
    offered = ["select", "draw", "rect", "circle", "pan"]
    if layer not in DRAWABLE:
        offered = ["select"] + (["draw"] if layer == "devices" else []) + ["pan"]
    if layer == "windows" and ed.wb.spec.layout is not None:
        offered = ["select", "pan"]                  # automatic windows are not drawn
    tool = st["tool"] if st["tool"] in offered else "select"
    names = {"select": "↖ Select", "draw": "✎ Draw", "rect": "▭ Rectangle",
             "circle": "◯ Circle", "pan": "✥ Pan"}
    if layer == "devices":
        names["draw"] = "✚ Place rotors"
    tools = pn.widgets.RadioButtonGroup(options={names[k]: k for k in offered}, value=tool,
                                        button_type="primary", button_style="outline",
                                        stylesheets=[TOOL_CSS], margin=(0, 8, 0, 0))
    tools.param.watch(lambda e: ed._switch(tool=e.new), "value")
    items: list = [tools]
    if tool in ("draw", "rect", "circle") and layer in DRAWABLE:
        if layer == "domain":
            as_ = pn.widgets.RadioButtonGroup(options={"Outline": "outline", "Hole": "hole"},
                                              value=st.get("draw_as", "outline"),
                                              button_type="light", margin=(0, 8, 0, 0))
            as_.param.watch(lambda e: _set_state(ed, draw_as=e.new), "value")
            items.append(as_)
        if tool == "draw":
            edges = pn.widgets.RadioButtonGroup(options={"Straight": "line",
                                                         "Smooth": "spline"},
                                                value=st.get("draw_edges", "line"),
                                                button_type="light", margin=(0, 8, 0, 0))
            edges.param.watch(lambda e: _set_state(ed, draw_edges=e.new), "value")
            items.append(edges)
            items += [_button("Finish", ed.close_shape, "primary", 80),
                      _button("Undo point", ed.undo_vertex, width=100),
                      _button("Cancel", ed.cancel_shape, width=80)]
    fit = _button("Fit", ed.fit, width=60)
    return pn.Row(*items, pn.layout.HSpacer(), fit, sizing_mode="stretch_width",
                  margin=(0, 0, 6, 0))


def footer(ed: "GeometryEditor"):
    from .editor import TOOLS  # noqa: F401
    wb = ed.wb
    snap = pn.widgets.Select(options={f"Snap: {s} cell{'s' * (s > 1)}": s
                                      for s in geo.SNAP_STEPS},
                             value=ed.state["snap"], width=130, margin=(0, 8, 0, 0))
    snap.param.watch(lambda e: ed._switch(snap=e.new), "value")
    shown = [k for k in ("grid", "overlaps", "labels") if wb.view_opts[k]]
    view = pn.widgets.CheckButtonGroup(options={"Grid": "grid", "Overlaps": "overlaps",
                                                "Labels": "labels"}, value=shown,
                                       button_type="light", margin=(0, 8, 0, 0))

    def set_view(e):
        for k in ("grid", "overlaps", "labels"):
            wb.view_opts[k] = k in e.new
        wb.show("model", keep_view=True)
    view.param.watch(set_view, "value")
    ed.summary.min_width = 300
    return pn.Column(pn.Row(ed.hint, sizing_mode="stretch_width"),
                     pn.Row(snap, view, pn.layout.HSpacer(), ed.summary,
                            sizing_mode="stretch_width", align="center"),
                     sizing_mode="stretch_width", margin=(4, 0, 0, 0))


def _set_state(ed: "GeometryEditor", **kw) -> None:
    ed.state.update(kw)
    ed.hint.object = ed._hint_html()


# ---------------------------------------------------------------------------
# the rail's list: the active layer's items
# ---------------------------------------------------------------------------


def rail_items(ed: "GeometryEditor") -> list:
    s = ed.wb.spec
    d = s.domain
    layer = ed.state["layer"]
    sel = ed.selected
    out: list = []

    def item(label: str, ref: str | None):
        b = pn.widgets.Button(name=label, sizing_mode="stretch_width", margin=(1, 4),
                              button_type="primary" if ref is not None and ref == sel
                              else "light", stylesheets=[ITEM_CSS])
        b.on_click(lambda _e, ref=ref: ed.select(ref))
        return b

    if layer == "domain":
        out.append(_label("In this layer"))
        if d.outline is not None:
            out.append(item(f"Outline · {len(d.outline.points)} vertices", "domain:outline"))
        else:
            out.append(_note(f"The domain is the whole {d.nx} × {d.ny} grid."))
        for h, o in enumerate(d.holes):
            out.append(item(f"Hole {h} · {len(o.points)} vertices", f"domain:hole:{h}"))
        #: each button says what the next shape is: after Cut a hole, draw_as stays
        #: "hole", and Draw the outline would have cut a second hole
        out += [_button("Draw the outline", lambda: (ed.state.update(draw_as="outline"),
                                                     ed._switch(tool="draw")), width=None,
                        stretch=True),
                _button("Cut a hole", lambda: (ed.state.update(draw_as="hole"),
                                               ed._switch(tool="draw")), width=None,
                        stretch=True)]
        if d.outline is not None or d.holes:
            out.append(_button("Use the whole grid", ed.reset_domain, width=None,
                               stretch=True))
        out.append(_note(f"Grid {d.nx} × {d.ny} cells of {d.dx:g} "
                         f"{_unit(s)}. A drawn shape holds the cells whose centres it "
                         f"contains."))
    elif layer == "regions":
        from .editor import material_colours, material_names
        #: every material of the case, used or not: a new one used by no region yet
        #: was listed nowhere, so it could not be found again (seen in the page)
        out.append(_label("Materials"))
        cmap = material_colours(s.regions)
        used: dict[str, int] = {m: 0 for m in s.materials}
        for r in s.regions:
            used[r.material] = used.get(r.material, 0) + 1
        for m, n in used.items():
            out.append(pn.Row(pn.pane.HTML(f"<span style='display:inline-block;width:14px;"
                                           f"height:14px;border-radius:4px;background:"
                                           f"{cmap.get(m, '#cbd5e1')}'></span>", width=18,
                                           margin=(9, 0, 0, 8)),
                              item(f"{m} · {n} region{'s' * (n != 1)}", f"material:{m}"),
                              sizing_mode="stretch_width", margin=0))
        out.append(_label("Regions"))
        for k, r in enumerate(s.regions):
            out.append(item(f"{r.id} · {r.material}", f"regions:{k}"))
        names = material_names(s)
        if ed.state.get("material") in ("", "material-1", None) and names:
            ed.state["material"] = names[0]
        pick = pn.widgets.Select(name="New regions are made of", options=names or ["material-1"],
                                 value=ed.state.get("material") if ed.state.get("material")
                                 in names else (names[0] if names else "material-1"),
                                 sizing_mode="stretch_width", margin=(8, 8))
        pick.param.watch(lambda e: ed.state.update(material=e.new), "value")
        out.append(pick)
        out.append(_note("Draw with Draw, Rectangle or Circle to paint a region. A later "
                         "region covers an earlier one."))
    elif layer == "boundaries":
        out.append(_label("Edges"))
        for k, b in enumerate(s.boundaries):
            val = "" if b.value is None else f" {b.value:g}"
            where = b.edge if b.drawn else (
                f"{b.edge} {b.start}–"
                f"{geo.edge_length(b.edge, d.nx, d.ny) if b.stop is None else b.stop}")
            out.append(item(f"{where} · {b.kind}{val}", f"boundary:{k}"))
        have = {b.edge for b in s.boundaries}
        if d.outline is not None or d.holes:
            for name in geo.drawn_edge_names(d):
                if name not in have:
                    out.append(_note(f"<b style='color:#b91c1c'>{name}: no condition</b>"))
        fam = _family(s)
        if fam is not None and (fam.fixed_boundaries or fam.drawn_derived):
            out.append(_button("Use the family's boundaries", ed.use_family_boundaries,
                               width=None, stretch=True))
        else:
            out.append(_note("Every edge takes one condition; add a vertex in Shape to "
                             "split a drawn edge in two."))
    elif layer == "windows":
        out.append(_label("Windows"))
        act = geo.domain_mask(d)
        for k, (w, (_wid, m)) in enumerate(zip(s.windows, geo.window_masks(s))):
            out.append(item(f"{w.id} · {int((m & act).sum()):,} cells", f"windows:{k}"))
        if not s.windows:
            out.append(_note("No windows yet: choose Automatic in the inspector, or draw "
                             "your own."))
    elif layer == "physics":
        fam = _family(s)
        out.append(_label("What it solves"))
        out.append(_note(html.escape(fam.label if fam else s.physics.family)))
    elif layer == "devices":
        out.append(_label("Rotors"))
        for k, v in enumerate(s.devices):
            out.append(item(f"{v.id} · x {v.x:g}, y {v.y:g} {_unit(s)}", f"devices:{k}"))
        out.append(_note("Place rotors with the Place rotors tool; drag them with Select."))
    elif layer == "attachments":
        out.append(_label("Circuit"))
        for k, a in enumerate(s.attachments):
            unit = "V" if a.kind == "battery" else "ohm"
            out.append(item(f"{a.id} · {a.kind} {a.value:g} {unit}", f"parts:{k}"))
        if not _reads(s, "attachments"):
            #: a circuit left from another family: shown so it can be deleted, but
            #: nothing to add (seen before the rebuild: the Add buttons were live)
            out.append(_ignored_note(s, "lumped parts"))
            return out
        out += [_button("Add a battery", lambda: ed.add_part("battery"), width=None,
                        stretch=True),
                _button("Add a resistor", lambda: ed.add_part("resistor"), width=None,
                        stretch=True)]
        out.append(_note("A node named like an electrode is that electrode; ground is 0 V. "
                         "Mark electrodes in Boundaries."))
    return out


# ---------------------------------------------------------------------------
# the inspector
# ---------------------------------------------------------------------------


def panel(ed: "GeometryEditor") -> list:
    ref = ed.selected
    layer = ed.state["layer"]
    try:
        if ref is None:
            return _layer_panel(ed, layer)
        kind = ref.split(":")[0]
        return {"domain": _shape_panel, "regions": _region_panel, "material": _material_panel,
                "windows": _window_panel, "boundary": _boundary_panel,
                "devices": _device_panel, "parts": _part_panel}[kind](ed, ref)
    except (IndexError, KeyError):
        return _layer_panel(ed, layer)


def _layer_panel(ed: "GeometryEditor", layer: str) -> list:
    s = ed.wb.spec
    if layer == "windows":
        return _windows_settings(ed)
    if layer == "physics":
        return _physics_settings(ed)
    if layer == "domain":
        d = s.domain
        drawn = d.outline is not None or bool(d.holes)
        cells = int(geo.domain_mask(d).sum())
        return [_head("Shape", "The domain",
                      f"{cells:,} cells" + (" inside the drawn shape" if drawn
                                            else f", the whole {d.nx} × {d.ny} grid")),
                _note("Draw the outline with <b>Draw</b>, <b>Rectangle</b> or <b>Circle</b>; "
                      "its edges can be straight, arcs or smooth. Holes are cut the same way "
                      "(press <i>Cut a hole</i> on the left first). The grid's size is in "
                      "Physics.")]
    if layer == "regions":
        fam = _family(s)
        #: the family's own word for them: a river's reaches, sound's media
        title = (fam.materials_title if fam is not None and fam.materials_title != "Materials"
                 else "Regions and materials")
        return [_head("Materials", title, ""),
                _note("Click a region to change what it is made of, or a material in the "
                      "list to set its properties."), _library(ed)] + _new_material(ed)
    if layer == "attachments" and not _reads(s, "attachments"):
        return [_head("Circuit", "Batteries and resistors", f"{len(s.attachments)} parts"),
                _ignored_note(s, "lumped parts")]
    if layer == "boundaries":
        fam = _family(s)
        extra = []
        if fam is not None and fam.fixed_boundaries:
            extra.append(_note(f"<b>Fixed by the solver.</b> {html.escape(fam.fixed_boundaries_why)}"))
        if fam is not None and fam.drawn_derived:
            extra.append(_note("On a drawn domain, each drawn edge takes the condition of the "
                               "grid edge it lies along, and "
                               f"<b>{fam.drawn_default}</b> elsewhere; they follow the shape "
                               "as it changes."))
        return [_head("Boundaries", "Conditions on the edges", ""),
                _note("Click an edge on the canvas, or in the list, to set its condition.")] + extra
    if layer == "devices":
        return [_head("Rotors", "Actuator disks", f"{len(s.devices)} rotors"),
                _note("Place rotors with the <b>Place rotors</b> tool; click one to set its "
                      "position and diameter.")]
    if layer == "attachments":
        return [_head("Circuit", "Batteries and resistors", f"{len(s.attachments)} parts"),
                _note("Click a part to edit it; add parts in the list on the left.")]
    return []


# -- the shape -------------------------------------------------------------


def _shape_panel(ed: "GeometryEditor", ref: str) -> list:
    from .editor import get_outline, shape_label
    s = ed.wb.spec
    o = get_outline(s, ref)
    ks, bs = o.kinds(), o.bulges()
    sub = ed.state.get("sub")
    title = "Outline" if ref == "domain:outline" else f"Hole {ref.split(':')[2]}"
    out = [_head("Shape", title, f"{len(o.points)} vertices, closed",
                 lambda: ed.delete_shapes([ref]))]
    if sub is not None and 0 <= sub < len(o.points):
        out += _edge_editor(ed, ref, o, sub)
    else:
        out.append(_note("Click an edge's circle to change how it is drawn."))
    out.append(_label("All edges"))
    for k in range(len(o.points)):
        cond = ""
        edge_name = (f"outline:{k}" if ref == "domain:outline"
                     else f"hole{ref.split(':')[2]}:{k}")
        for b in s.boundaries:
            if b.edge == edge_name:
                cond = f" · {b.kind}"
        lab = f"{k} · {ks[k]}" + (f" ({bs[k]:.2g})" if ks[k] == "arc" else "") + cond
        btn = pn.widgets.Button(name=lab, sizing_mode="stretch_width", margin=(1, 8),
                                button_type="primary" if k == sub else "light",
                                stylesheets=[ITEM_CSS])
        btn.on_click(lambda _e, k=k: ed.select(ref, sub=k))
        out.append(btn)
    out.append(_note(f"{_edge_summary(o)}. {html.escape(shape_label(s, ref)).capitalize()} "
                     f"is resolved to the cells whose centres it contains."))
    return out


def _edge_editor(ed: "GeometryEditor", ref: str, o, k: int) -> list:
    ks, bs = o.kinds(), o.bulges()
    kind = pn.widgets.RadioButtonGroup(options={"Straight": "line", "Arc": "arc",
                                                "Smooth": "spline"}, value=ks[k],
                                       button_type="primary", button_style="outline",
                                       sizing_mode="stretch_width", margin=(4, 8))
    kind.param.watch(lambda e: ed.set_edge_kind(ref, k, e.new), "value")
    out = [_label(f"Edge {k}"), kind]
    if ks[k] == "arc":
        bulge = pn.widgets.FloatInput(name="Bulge (1 is a semicircle; negative bows the "
                                           "other way)", value=float(bs[k]), step=0.05,
                                      format="0.0000",          # not seventeen digits
                                      start=-shapes.MAX_BULGE, end=shapes.MAX_BULGE,
                                      sizing_mode="stretch_width", margin=(4, 8))
        bulge.param.watch(lambda e: ed.set_edge_kind(ref, k, "arc" if e.new else "line",
                                                     float(e.new) if e.new else None), "value")
        out.append(bulge)
    out.append(_note("A smooth edge is a spline through its vertices and their "
                     "neighbours; an arc bends as far as its circle is dragged."))
    return out


def _edge_summary(o) -> str:
    from .editor import _edge_summary as es
    return es(o).capitalize()


# -- materials and regions ---------------------------------------------------


def _region_panel(ed: "GeometryEditor", ref: str) -> list:
    from .editor import material_names, with_material
    s = ed.wb.spec
    k = int(ref.split(":")[1])
    r = s.regions[k]
    if r.shape == "curve":
        shape = f"drawn ({_edge_summary(r.outline).lower()})"
    else:
        shape = {"rect": "a rectangle", "polygon": "a polygon imported from Gmsh"}[r.shape]
    out = [_head("Region", r.id, shape, ed.delete_selected)]
    names = material_names(s)
    if r.material not in names:
        names = [r.material] + names
    mat = pn.widgets.Select(name="Made of", options=names, value=r.material,
                            sizing_mode="stretch_width", margin=(4, 8))

    def set_mat(e):
        def apply(c):
            c.regions[k].material = e.new
            with_material(c, e.new)
        ed.wb.edit(apply, f"region {r.id}: made of {e.new}")
    mat.param.watch(set_mat, "value")
    out.append(mat)
    out += _material_props(ed, r.material)
    later = [x.id for x in s.regions[k + 1:]]
    out.append(_label("Stacking"))
    out.append(_note(f"Covers the regions listed before it where they overlap"
                     + (f"; {', '.join(later)} cover{'s' * (len(later) == 1)} it." if later
                        else ".")))
    out.append(pn.Row(_button("Bring forward", lambda: ed.restack(+1), width=130),
                      _button("Send backward", lambda: ed.restack(-1), width=130),
                      margin=(0, 4)))
    if r.shape == "rect":
        out += _box_fields(ed, "regions", k, r)
    elif r.shape == "curve":
        out += _drawn_edges(ed, ref, r.outline)
    return out


def _material_panel(ed: "GeometryEditor", ref: str) -> list:
    s = ed.wb.spec
    m = ref[len("material:"):]
    n = sum(1 for r in s.regions if r.material == m)
    return [_head("Material", m, f"used by {n} region{'s' * (n != 1)}")] + \
        _material_props(ed, m) + [_library(ed)] + _new_material(ed)


def _material_props(ed: "GeometryEditor", m: str) -> list:
    s = ed.wb.spec
    fam = _family(s)
    if fam is None or not fam.material_props:
        return []
    props = s.materials.get(m, {})
    out = [_label(f"{m}: properties")]
    for p in fam.material_props:
        w = pn.widgets.FloatInput(name=f"{p.label}" + (f", {p.unit}" if p.unit else ""),
                                  value=float(props.get(p.name, p.default)),
                                  sizing_mode="stretch_width", margin=(2, 8))

        def set_prop(e, name=p.name):
            val = float(e.new)
            ed.wb.edit(lambda c: c.materials.setdefault(m, {}).__setitem__(name, val),
                       f"material {m}: {name} = {val:g}")
        w.param.watch(set_prop, "value")
        out.append(w)
    if m not in s.materials:
        out.append(_note(f"<b style='color:#b91c1c'>{html.escape(m)} has no properties "
                         f"yet</b>: the values shown are the family's defaults until you "
                         f"change one."))
    out.append(_note(html.escape(fam.materials_note.split(" A region's")[0])))
    return out


def _library(ed: "GeometryEditor"):
    s = ed.wb.spec
    fam = _family(s)
    lib = fam.material_library() if fam is not None else {}
    missing = [m for m in sorted(lib) if m not in s.materials]
    if not missing:
        return pn.Spacer(height=0)
    pick = pn.widgets.Select(name="Add a material from the library",
                             options=["choose..."] + missing, value="choose...",
                             sizing_mode="stretch_width", margin=(8, 8))

    def add(e):
        if e.new in lib:
            name = e.new
            ed.wb.edit(lambda c: c.materials.__setitem__(name, dict(lib[name])),
                       f"material {name} added from the library")
            ed.select(f"material:{name}")
    pick.param.watch(add, "value")
    return pick


def _new_material(ed: "GeometryEditor") -> list:
    """A material of the person's own, by name, starting from the family's defaults
    (the rebuilt page had lost this: the old table took any name typed in)."""
    s = ed.wb.spec
    fam = _family(s)
    if fam is None or not fam.material_props:
        return []
    #: added when the name arrives (Enter, or leaving the field), not by a button: a
    #: click reached the server before the typed name did, and said to type one
    #: (seen in the page)
    name = pn.widgets.TextInput(name="Or a new material: its name, then Enter",
                                placeholder="its name", sizing_mode="stretch_width",
                                margin=(8, 8, 2, 8))

    def add(e):
        m = (e.new or "").strip()
        if not m:
            return
        if m in ed.wb.spec.materials:
            ed.wb.notify("info", f"{m} is already one of this case's materials")
            return
        props = {p.name: float(p.default) for p in fam.material_props}
        if ed.wb.edit(lambda c: c.materials.__setitem__(m, dict(props)),
                      f"material {m} added, with the family's defaults"):
            ed.select(f"material:{m}")
    name.param.watch(add, "value")
    return [name]


def _reads(spec, layer: str) -> bool:
    fam = _family(spec)
    return fam is not None and layer in fam.layers


def _ignored_note(spec, what: str):
    fam = _family(spec)
    return _note(f"<b style='color:#b45309'>The {html.escape(fam.label if fam else '?')} "
                 f"family does not read {what}</b>: its solver ignores these. Delete them, "
                 f"or choose a family that reads them.")


# -- windows ----------------------------------------------------------------


def _windows_settings(ed: "GeometryEditor") -> list:
    s = ed.wb.spec
    wb = ed.wb
    fam = _family(s)
    lay = s.layout
    auto = pn.widgets.RadioButtonGroup(options={"Automatic": True, "My own": False},
                                       value=lay is not None, button_type="primary",
                                       button_style="outline", sizing_mode="stretch_width",
                                       margin=(4, 8))
    cut = pn.widgets.RadioBoxGroup(options={"Along the shape's length": "along",
                                            "One piece per material": "materials"},
                                   value=lay.cut if lay else "along", margin=(4, 8))
    n_along = pn.widgets.IntInput(name="Pieces along", value=lay.along if lay else
                                  max(2, len(s.windows)), start=1, end=32, width=130)
    n_across = pn.widgets.IntInput(name="Pieces across", value=lay.across if lay else 1,
                                   start=1, end=8, width=130)

    def relayout(_e=None):
        ed.set_layout(bool(auto.value), cut.value, n_along.value, n_across.value)
    for w in (auto, cut, n_along, n_across):
        w.param.watch(relayout, "value")
    out = [_head("Windows", "How the domain is cut", ""), auto]
    if lay is not None:
        out += [cut, pn.Row(n_along, n_across, margin=(0, 4))]
        out.append(_note("Generated from the shape: cut at level curves of its own "
                         "coordinates, or one piece per material, and cut again whenever the "
                         "shape or the materials change. Editing a window by hand switches to "
                         "My own."))
    else:
        out.append(_button("Lay a grid of rectangles...", wb.dialog_tiling, width=None,
                           stretch=True))
        out.append(_note("Draw windows of any shape with <b>Draw</b>, <b>Rectangle</b> or "
                         "<b>Circle</b>; click one to reshape or delete it."))
    # how the windows are joined
    styles = list(fam.styles) if fam is not None else ["A"]
    style = pn.widgets.Select(name="Windows are joined by",
                              options={f"{k}: {registry.STYLES[k]}": k for k in styles},
                              value=s.coupling.style if s.coupling.style in styles else styles[0],
                              sizing_mode="stretch_width", margin=(12, 8, 4, 8),
                              disabled=len(styles) < 2)
    style.param.watch(lambda e: wb.edit(lambda c: setattr(c.coupling, "style", e.new),
                                        f"coupling style = {e.new}"), "value")
    out.append(style)
    if s.coupling.style in ("A", "B"):
        ramp = pn.widgets.IntInput(name="Blend width (cells)", value=s.coupling.ramp_cells,
                                   start=1, width=150, margin=(4, 8))
        ramp.param.watch(lambda e: wb.edit(lambda c: setattr(c.coupling, "ramp_cells",
                                                             int(e.new)),
                                           f"blend width = {e.new}"), "value")
        out.append(ramp)
        out.append(_note(f"Where windows meet they overlap by at least twice the blend "
                         f"width, {2 * s.coupling.ramp_cells} cells."))
    ok = getattr(ed, "windows_ok", True)
    geo_errors = [i for i in wb.issues() if i.severity == "error" and "window" in i.message]
    if s.windows and ok and not geo_errors:
        #: the rule the check applies: pieces (C, D, the split) tile the domain,
        #: overlapping windows (A, B) give every cell one at full weight
        good = ("The pieces tile the domain without overlapping."
                if s.coupling.style in ("C", "D", "split")
                else "Every cell has a window at full weight.")
        out.append(_note(f"<b style='color:#15803d'>✓ {good}</b>"))
    elif geo_errors:
        out.append(_note(f"<b style='color:#b91c1c'>{html.escape(geo_errors[0].message)}"
                         f"</b>"))
    return out


def _window_panel(ed: "GeometryEditor", ref: str) -> list:
    s = ed.wb.spec
    d = s.domain
    k = int(ref.split(":")[1])
    w = s.windows[k]
    cells = int((geo.window_mask(w, d.nx, d.ny) & geo.domain_mask(d)).sum())
    what = {"rect": "a rectangle", "curve": "drawn", "cells": "generated from the shape"}[w.shape]
    out = [_head("Window", w.id, f"{what}, {cells:,} cells", ed.delete_selected)]
    name = pn.widgets.TextInput(name="Name", value=w.id, sizing_mode="stretch_width",
                                margin=(4, 8))
    name.param.watch(lambda e: ed.wb.edit(lambda c: setattr(c.windows[k], "id",
                                                            e.new.strip() or w.id),
                                          f"window {w.id} renamed {e.new}"), "value")
    out.append(name)
    if w.shape == "rect":
        out += _box_fields(ed, "windows", k, w)
    elif w.shape == "cells":
        out.append(_note("This window follows the shape. Change how the windows are cut with "
                         "nothing selected, or switch to My own to edit it."))
    else:
        out += _drawn_edges(ed, ref, w.outline)
    return out


def _drawn_edges(ed: "GeometryEditor", ref: str, o) -> list:
    """A drawn window's or region's selected edge, as the outline's is edited: only
    the domain's inspector had this, so a window's arc could be dragged but not
    typed."""
    sub = ed.state.get("sub")
    if sub is not None and 0 <= sub < len(o.points):
        return _edge_editor(ed, ref, o, sub)
    return [_note("Click an edge's circle to change how it is drawn; drag the squares, "
                  "circles and diamond to reshape or move it.")]


def _box_fields(ed: "GeometryEditor", key: str, k: int, item) -> list:
    fields = []
    for label, attr in (("x", "x0"), ("y", "y0"), ("width", "nx"), ("height", "ny")):
        w = pn.widgets.IntInput(name=label, value=int(getattr(item, attr)), width=70,
                                margin=(2, 4))

        def set_(e, attr=attr, label=label):
            def apply(c):
                it = getattr(c, key)[k]
                setattr(it, attr, int(e.new))
                if key == "windows":
                    c.layout = None
            ed.wb.edit(apply, f"{item.id}: {label} = {e.new}")
        w.param.watch(set_, "value")
        fields.append(w)
    return [_label("Place, in cells"), pn.Row(*fields, margin=(0, 4))]


# -- boundaries -------------------------------------------------------------


def _boundary_panel(ed: "GeometryEditor", ref: str) -> list:
    s = ed.wb.spec
    d = s.domain
    k = int(ref.split(":")[1])
    b = s.boundaries[k]
    fam = _family(s)
    kinds = list(fam.boundary_kinds) if fam is not None and fam.boundary_kinds else \
        [x.id for x in registry.BOUNDARY_KINDS]
    fixed = fam is not None and ((fam.fixed_boundaries and not b.drawn)
                                 or (fam.drawn_derived and b.drawn))
    where = (f"the drawn edge {b.edge}" if b.drawn else
             f"the grid's {b.edge} edge, cells {b.start} to "
             f"{geo.edge_length(b.edge, d.nx, d.ny) if b.stop is None else b.stop}")
    out = [_head("Boundary", b.id, where, None if fixed else ed.delete_selected)]
    labels = {x.id: f"{x.id}: {x.meaning}" for x in registry.BOUNDARY_KINDS}
    cond = pn.widgets.Select(name="Condition", options={labels.get(x, x): x for x in kinds},
                             value=b.kind if b.kind in kinds else kinds[0],
                             sizing_mode="stretch_width", margin=(4, 8), disabled=bool(fixed))

    def set_kind(e):
        def apply(c):
            c.boundaries[k].kind = e.new
            if not registry.boundary_kind(e.new).needs_value:
                c.boundaries[k].value = None
        ed.wb.edit(apply, f"boundary {b.id}: {e.new}")
    cond.param.watch(set_kind, "value")
    out.append(cond)
    bk = registry.boundary_kind(b.kind) if b.kind in {x.id for x in registry.BOUNDARY_KINDS} \
        else None
    if bk is not None and bk.needs_value:
        val = pn.widgets.FloatInput(name=f"Value ({bk.unit})" if bk.unit else "Value",
                                    value=float(b.value) if b.value is not None else None,
                                    sizing_mode="stretch_width", margin=(4, 8),
                                    disabled=bool(fixed))
        val.param.watch(lambda e: ed.wb.edit(lambda c: setattr(c.boundaries[k], "value",
                                                               None if e.new is None
                                                               else float(e.new)),
                                             f"boundary {b.id}: value = {e.new}"), "value")
        out.append(val)
    else:
        out.append(_note("This condition takes no value."))
    if not b.drawn and d.outline is None and not fixed:
        n = geo.edge_length(b.edge, d.nx, d.ny)
        start = pn.widgets.IntInput(name="From cell", value=b.start, start=0, end=n - 1,
                                    width=120)
        stop = pn.widgets.IntInput(name="To cell", value=n if b.stop is None else b.stop,
                                   start=1, end=n, width=120)

        def set_span(_e):
            def apply(c):
                c.boundaries[k].start = int(start.value)
                c.boundaries[k].stop = None if int(stop.value) == n else int(stop.value)
            ed.wb.edit(apply, f"boundary {b.id}: cells {start.value} to {stop.value}")
        start.param.watch(set_span, "value")
        stop.param.watch(set_span, "value")
        out += [_label("Along the edge"), pn.Row(start, stop, margin=(0, 4)),
                _button("Split it in two", lambda: _split_segment(ed, k), width=None,
                        stretch=True)]
    if fixed:
        out.append(_note("<b>Fixed by the solver</b>: " + html.escape(
            fam.fixed_boundaries_why if not b.drawn else
            "each drawn edge takes the condition of the grid edge it lies along, and "
            f"{fam.drawn_default} elsewhere.")))
    return out


def _split_segment(ed: "GeometryEditor", k: int) -> None:
    """A grid edge's segment cut in two at its middle; the new half is a copy."""
    s = ed.wb.spec
    d = s.domain
    b = s.boundaries[k]
    n = geo.edge_length(b.edge, d.nx, d.ny)
    stop = n if b.stop is None else b.stop
    mid = (b.start + stop) // 2
    if mid <= b.start:
        ed.wb.notify("info", "the segment is one cell long")
        return
    from .editor import _next_id
    nid = _next_id("B", [x.id for x in s.boundaries])

    def apply(c):
        c.boundaries[k].stop = mid
        c.boundaries.insert(k + 1, Boundary(id=nid, edge=b.edge, kind=b.kind, value=b.value,
                                            start=mid, stop=b.stop))
    ed.wb.edit(apply, f"boundary {b.id} split at cell {mid}")


# -- rotors and the circuit --------------------------------------------------


def _device_panel(ed: "GeometryEditor", ref: str) -> list:
    s = ed.wb.spec
    k = int(ref.split(":")[1])
    v = s.devices[k]
    unit = _unit(s)
    out = [_head("Rotor", v.id, "an actuator disk", ed.delete_selected)]
    for label, attr in (("Disk plane, x", "x"), ("Disk centre, y", "y"),
                        ("Diameter", "diameter")):
        w = pn.widgets.FloatInput(name=f"{label} ({unit})", value=float(getattr(v, attr)),
                                  step=0.125, sizing_mode="stretch_width", margin=(2, 8))
        w.param.watch(lambda e, attr=attr: ed.wb.edit(
            lambda c: setattr(c.devices[k], attr, float(e.new)),
            f"rotor {v.id}: {attr} = {e.new}"), "value")
        out.append(w)
    return out


def _part_panel(ed: "GeometryEditor", ref: str) -> list:
    from .spec import Attachment
    s = ed.wb.spec
    k = int(ref.split(":")[1])
    a = s.attachments[k]
    out = [_head("Circuit", a.id, "a battery" if a.kind == "battery" else "a resistor",
                 ed.delete_selected)]
    value = pn.widgets.FloatInput(name="EMF (V)" if a.kind == "battery" else "Resistance (ohm)",
                                  value=float(a.value), sizing_mode="stretch_width",
                                  margin=(2, 8))

    def edit(field, val, label):
        def apply(c):
            setattr(c.attachments[k], field, val)
            Attachment.model_validate(c.attachments[k].model_dump())
        try:
            ed.wb.edit(apply, f"part {a.id}: {label}")
        except (TypeError, ValueError) as exc:
            ed.wb.notify("error", f"not applied: {exc}")
            ed.sync()                          # the fields show the case again
    value.param.watch(lambda e: edit("value", float(e.new), f"value = {e.new}"), "value")
    out.append(value)
    if a.kind == "battery":
        inner = pn.widgets.FloatInput(name="Internal resistance (ohm)", value=float(a.internal),
                                      start=0.0, sizing_mode="stretch_width", margin=(2, 8))
        inner.param.watch(lambda e: edit("internal", float(e.new), f"internal = {e.new}"),
                          "value")
        out.append(inner)
    na = pn.widgets.TextInput(name="From node (a; a battery's +)", value=a.a,
                              sizing_mode="stretch_width", margin=(2, 8))
    nb = pn.widgets.TextInput(name="To node (b)", value=a.b, sizing_mode="stretch_width",
                              margin=(2, 8))
    na.param.watch(lambda e: edit("a", e.new.strip(), f"a = {e.new}"), "value")
    nb.param.watch(lambda e: edit("b", e.new.strip(), f"b = {e.new}"), "value")
    out += [na, nb]
    electrodes = [b.id for b in s.boundaries if b.kind == "electrode"]
    out.append(_note("Electrodes: " + (", ".join(electrodes) if electrodes else "none yet "
                                       "(mark an edge as an electrode in Boundaries)")
                     + ". Any other name is a node of the circuit; ground is 0 V."))
    return out


# -- physics ----------------------------------------------------------------


def _physics_settings(ed: "GeometryEditor") -> list:
    wb = ed.wb
    s = wb.spec
    fam = _family(s)
    out = [_head("Physics", fam.label if fam else s.physics.family, "")]
    if fam is None:
        return out
    for p in fam.params:
        w = pn.widgets.FloatInput(name=f"{p.label}" + (f" ({p.unit})" if p.unit else ""),
                                  value=float(s.physics.get(p.name)),
                                  sizing_mode="stretch_width", margin=(2, 8))
        w.param.watch(lambda e, name=p.name: wb.edit(
            lambda c: c.physics.params.__setitem__(name, float(e.new)),
            f"physics.{name} = {e.new}"), "value")
        out.append(w)
        if p.help:
            out.append(_note(html.escape(p.help)))
    out.append(_label("Time"))
    if len(fam.modes) > 1:
        mode = pn.widgets.RadioButtonGroup(options={"Steady": "steady",
                                                    "Changing in time": "transient"},
                                           value=s.run.mode, button_type="primary",
                                           button_style="outline", sizing_mode="stretch_width",
                                           margin=(4, 8))
        mode.param.watch(lambda e: wb.edit(lambda c: setattr(c.run, "mode", e.new),
                                           f"run.mode = {e.new}"), "value")
        out.append(mode)
    else:
        out.append(_note(f"{fam.modes[0].capitalize()} only, in this family."))
    if s.run.mode == "transient":
        tunit = "D/U" if fam.id == "incompressible-2d" else "s"
        #: the Run tab counts macro-steps, so the field that sets one's length says so
        step = pn.widgets.FloatInput(name=f"Macro-step ({tunit})", value=s.run.macro_dt,
                                     start=1e-12, sizing_mode="stretch_width", margin=(2, 8))
        step.param.watch(lambda e: wb.edit(lambda c: setattr(c.run, "macro_dt", float(e.new)),
                                           f"run.macro_dt = {e.new}"), "value")
        out.append(step)
    # the grid
    d = s.domain
    out.append(_label("Grid"))
    gx = pn.widgets.IntInput(name="Width (cells)", value=d.nx, start=1, width=86)
    gy = pn.widgets.IntInput(name="Height (cells)", value=d.ny, start=1, width=86)
    gd = pn.widgets.FloatInput(name=f"Cell ({_unit(s)})", value=d.dx, start=1e-12, width=86)
    gx.param.watch(lambda e: wb.edit(lambda c: setattr(c.domain, "nx", int(e.new)),
                                     f"domain.nx = {e.new}"), "value")
    gy.param.watch(lambda e: wb.edit(lambda c: setattr(c.domain, "ny", int(e.new)),
                                     f"domain.ny = {e.new}"), "value")
    gd.param.watch(lambda e: wb.edit(lambda c: setattr(c.domain, "dx", float(e.new)),
                                     f"domain.dx = {e.new}"), "value")
    out.append(pn.Row(gx, gy, gd, margin=(0, 4)))
    out.append(_note(f"{d.nx * d.dx:.4g} × {d.ny * d.dx:.4g} {_unit(s)}. Drawn shapes "
                     f"keep their place in cells."))
    # the solver's settings, folded away
    solver = _solver_settings(ed)
    if solver:
        out.append(pn.Card(*solver, title="Solver settings", collapsed=True,
                           sizing_mode="stretch_width", margin=(8, 8)))
    out.append(pn.Card(pn.pane.Markdown("\n".join([f"- {x}" for x in fam.solvers]
                                                  + [f"\n{fam.note}"]),
                                        sizing_mode="stretch_width",
                                        styles={"font-size": "12px"}),
                       title="What it runs on", collapsed=True, sizing_mode="stretch_width",
                       margin=(4, 8)))
    return out


def _solver_settings(ed: "GeometryEditor") -> list:
    wb = ed.wb
    s = wb.spec
    fam = _family(s)
    style = s.coupling.style
    out: list = []
    if fam.id == "incompressible-2d":
        asm = pn.widgets.Select(name="Assembly", value=s.coupling.assembly,
                                options={"blend, then one global projection": "projected",
                                         "blend only": "blend"}, sizing_mode="stretch_width")
        asm.param.watch(lambda e: wb.edit(lambda c: setattr(c.coupling, "assembly", e.new),
                                          f"coupling.assembly = {e.new}"), "value")
        ell = pn.widgets.Select(name="Pressure solve", value=s.coupling.elliptic,
                                options={"exposed: once, on the assembled field": "exposed",
                                         "embedded: inside every window": "embedded"},
                                sizing_mode="stretch_width")
        ell.param.watch(lambda e: wb.edit(lambda c: setattr(c.coupling, "elliptic", e.new),
                                          f"coupling.elliptic = {e.new}"), "value")
        out += [asm, ell]
    iterates = style in ("B", "C", "D") and not fam.explicit_coupling
    if iterates:
        tols = [1e-6, 1e-8, 1e-10, 1e-12, 1e-13]
        if s.coupling.tolerance not in tols:
            tols.append(s.coupling.tolerance)
        tol = pn.widgets.Select(name="Tolerance (update over the field's scale)",
                                value=s.coupling.tolerance,
                                options={f"{t:.0e}": t for t in sorted(tols, reverse=True)},
                                sizing_mode="stretch_width")
        tol.param.watch(lambda e: wb.edit(lambda c: setattr(c.coupling, "tolerance",
                                                            float(e.new)),
                                          f"coupling.tolerance = {e.new}"), "value")
        its = pn.widgets.IntInput(name="Most iterations", value=s.coupling.max_iterations,
                                  start=1, sizing_mode="stretch_width")
        its.param.watch(lambda e: wb.edit(lambda c: setattr(c.coupling, "max_iterations",
                                                            int(e.new)),
                                          f"coupling.max_iterations = {e.new}"), "value")
        out += [tol, its]
    if style in ("C", "D") and iterates:
        rel = pn.widgets.FloatInput(name="First relaxation factor", value=s.coupling.relaxation,
                                    start=1e-6, end=2.0, step=0.05, sizing_mode="stretch_width")
        rel.param.watch(lambda e: wb.edit(lambda c: setattr(c.coupling, "relaxation",
                                                            float(e.new)),
                                          f"coupling.relaxation = {e.new}"), "value")
        ait = pn.widgets.Checkbox(name="then Aitken's rule adapts it", value=s.coupling.aitken)
        ait.param.watch(lambda e: wb.edit(lambda c: setattr(c.coupling, "aitken", bool(e.new)),
                                          f"coupling.aitken = {e.new}"), "value")
        out += [rel, ait]
    if style == "C" and iterates:
        sides = {"auto: a floating piece, else the lower conductivity": "auto"}
        sides.update({w.id: w.id for w in s.windows})
        side = pn.widgets.Select(name="Dirichlet side",
                                 value=(s.coupling.dirichlet_side
                                        if s.coupling.dirichlet_side in sides.values()
                                        else "auto"), options=sides,
                                 sizing_mode="stretch_width")
        side.param.watch(lambda e: wb.edit(lambda c: setattr(c.coupling, "dirichlet_side",
                                                             e.new),
                                           f"coupling.dirichlet_side = {e.new}"), "value")
        out.append(side)
    if fam.explicit_coupling:
        out.append(_note("An explicit exchange every step: nothing iterates."))
    return out


# ---------------------------------------------------------------------------
# small parts
# ---------------------------------------------------------------------------


def _family(spec):
    try:
        return registry.family(spec.physics.family)
    except KeyError:
        return None


def _unit(spec) -> str:
    fam = _family(spec)
    return fam.length_unit if fam is not None else ""


def _label(text: str):
    return pn.pane.HTML(f"<div style='{SECTION}'>{html.escape(text)}</div>",
                        margin=(10, 12, 2, 12), sizing_mode="stretch_width")


def _note(text: str):
    return pn.pane.HTML(f"<div style='font-size:12px;line-height:1.45;color:#475569'>{text}"
                        f"</div>", margin=(4, 12), sizing_mode="stretch_width")


def _divider():
    return pn.pane.HTML("<div style='height:1px;background:#e2e8f0'></div>",
                        margin=(8, 8), sizing_mode="stretch_width")


def _head(kind: str, title: str, subtitle: str, delete: Callable | None = None):
    def esc(t):                  # element text: quotes need no escaping
        return html.escape(t, quote=False)
    text = pn.pane.HTML(f"<div style='{SECTION}'>{esc(kind)}</div>"
                        f"<div style='font-size:18px;font-weight:650;color:#0f172a;"
                        f"margin-top:2px'>{esc(title)}</div>"
                        + (f"<div style='font-size:13px;color:#5b6b7c;margin-top:2px'>"
                           f"{esc(subtitle)}</div>" if subtitle else ""),
                        sizing_mode="stretch_width", margin=(12, 8, 6, 12))
    if delete is None:
        return text
    b = pn.widgets.Button(name="Delete", button_type="danger", button_style="outline",
                          width=76, margin=(14, 10, 0, 0))
    b.on_click(lambda _e: delete())
    return pn.Row(text, b, sizing_mode="stretch_width", margin=0)


def _button(name: str, fn: Callable, kind: str = "default", width: int | None = 120,
            stretch: bool = False):
    b = pn.widgets.Button(name=name, button_type=kind if kind != "default" else "default",
                          width=None if stretch else width,
                          sizing_mode="stretch_width" if stretch else None,
                          margin=(4, 8))
    b.on_click(lambda _e: fn())
    return b


__all__ = ["model_layout", "layer_buttons", "toolbar", "footer", "rail_items", "panel"]
