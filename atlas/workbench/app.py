"""Atlas Workbench: the shell.

The menu and overall structure of the decomposition GUI, built on Panel and Bokeh
(both BSD-3, both already installed, nothing downloaded).  What works now:
cases (new, examples, open, save, save as, import, export), undo and redo, the
case and physics forms, a read-only preview of the domain, its windows and its
rotors, and the structural check.  What does not work yet says so where it
would be, with the reason: the geometry editing tools (being designed with the
owner), the Atlas compile of a case file, and the runner.

Every action goes through `Workbench.dispatch`, so a test can drive the whole
menu without a browser.
"""

from __future__ import annotations

import datetime
import glob
import html
import io
import os
from typing import Callable

import pandas as pd
import panel as pn
from bokeh.models import ColumnDataSource, HoverTool, LabelSet, Range1d, SingleIntervalTicker
from bokeh.plotting import figure
from pydantic import ValidationError

from . import registry
from .spec import EXAMPLES, CaseSpec, blank_case, check, example_case, slug, summary

#: Native form controls and scrollbars follow the page, not the OS dark-mode setting.
LIGHT_CSS = ":root { color-scheme: light; }"

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CASES_DIR = os.path.join(ROOT, "out", "workbench", "cases")
VERSION = "0.1 (shell)"

#: The workflow, in order.  key -> title.
STEPS: tuple[tuple[str, str], ...] = (
    ("case", "Case"),
    ("geometry", "Geometry"),
    ("physics", "Physics & coupling"),
    ("check", "Check"),
    ("run", "Run & compare"),
    ("results", "Results"),
)

#: Why each unbuilt action is unavailable, shown wherever the action would be.
NOT_BUILT = {
    "geometry": ("The tools for drawing and editing windows and devices are being designed "
                 "with you next. This preview draws the case as it stands."),
    "compile": ("Compiling a case with the Atlas compiler needs the case-to-graph loader "
                "(plan step 2). Not built yet."),
    "run": ("The runner that marches a case file (plan step 3) is not built yet. The "
            "measurements it will reproduce are in W346."),
    "results": "Results appear after a run, and the runner is not built yet.",
}

# colours chosen to read on both the light and the dark theme
C_DOMAIN = "#64748b"
C_WINDOW = "#2a9d8f"
C_OVERLAP = "#e9a23b"
C_ROTOR = "#d1495b"
ACCENT = "#0e6874"
HEADER = "#16303f"


class Workbench:
    """One browser session's state, views and actions."""

    def __init__(self, cases_dir: str | None = None, spec: CaseSpec | None = None):
        self.cases_dir = cases_dir or DEFAULT_CASES_DIR
        self.spec: CaseSpec = spec if spec is not None else example_case("wake-array-3")
        self.path: str | None = None
        self.dirty = False
        self.history: list[str] = []
        self.future: list[str] = []
        self.view_opts = {"grid": False, "overlaps": True, "labels": True, "log": True}
        self.active = "case"
        self.log_lines: list[str] = []
        self._build()
        self.log("opened the example case 'wake-array-3'")

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    def log(self, text: str) -> None:
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_lines.insert(0, f"{stamp}  {text}")
        del self.log_lines[200:]
        if hasattr(self, "log_pane"):
            self.log_pane.object = self._log_html()

    def notify(self, kind: str, text: str) -> None:
        """A toast in the browser when there is one; always a log line."""
        self.log(text)
        note = pn.state.notifications
        if note is not None:
            getattr(note, kind)(text, duration=6000 if kind != "success" else 3000)

    def set_spec(self, spec: CaseSpec, label: str, *, record: bool = True,
                 path: str | None | object = ..., dirty: bool = True) -> None:
        """Replace the whole case (new, open, undo): every view is rebuilt."""
        if record:
            self.history.append(self.spec.to_json())
            del self.history[:-100]
            self.future.clear()
        self.spec = spec
        if path is not ...:
            self.path = path
        self.dirty = dirty
        self.log(label)
        self.refresh(rebuild=True)

    def edit(self, apply: Callable[[CaseSpec], None], label: str) -> bool:
        """Change one field.  Validated as a whole case; refused if invalid."""
        new = self.spec.copy_deep()
        apply(new)
        try:
            new = CaseSpec.model_validate(new.model_dump())
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(x) for x in first.get("loc", ()))
            self.notify("error", f"not applied: {where}: {first.get('msg')}")
            return False
        self.history.append(self.spec.to_json())
        del self.history[:-100]
        self.future.clear()
        self.spec = new
        self.dirty = True
        self.log(label)
        self.refresh(rebuild=False)
        return True

    def undo(self) -> None:
        if not self.history:
            self.notify("info", "nothing to undo")
            return
        self.future.append(self.spec.to_json())
        self.set_spec(CaseSpec.from_json(self.history.pop()), "undo", record=False)

    def redo(self) -> None:
        if not self.future:
            self.notify("info", "nothing to redo")
            return
        self.history.append(self.spec.to_json())
        self.set_spec(CaseSpec.from_json(self.future.pop()), "redo", record=False)

    # ------------------------------------------------------------------
    # files
    # ------------------------------------------------------------------
    def case_files(self) -> list[str]:
        files = glob.glob(os.path.join(self.cases_dir, "*.json"))
        return sorted(files, key=os.path.getmtime, reverse=True)

    def save_to(self, path: str) -> None:
        self.spec.save(path)
        self.path = path
        self.dirty = False
        self.notify("success", f"saved {os.path.relpath(path, ROOT)}")
        self.refresh(rebuild=False)

    def open_path(self, path: str) -> None:
        try:
            spec = CaseSpec.load(path)
        except (OSError, ValidationError, ValueError) as exc:
            self.notify("error", f"could not open {os.path.basename(path)}: {exc}")
            return
        self.set_spec(spec, f"opened {os.path.relpath(path, ROOT)}", path=path, dirty=False)

    def import_text(self, text: str, name: str = "imported file") -> None:
        try:
            spec = CaseSpec.from_json(text)
        except (ValidationError, ValueError) as exc:
            self.notify("error", f"{name} is not a workbench case: {exc}")
            return
        self.set_spec(spec, f"imported {name}", path=None)

    # ------------------------------------------------------------------
    # the one entry point every menu item and button goes through
    # ------------------------------------------------------------------
    def dispatch(self, action: str) -> None:
        group, _, rest = action.partition(":")
        if action == "file:new":
            self.set_spec(blank_case(), "new blank case", path=None)
        elif group == "file" and rest.startswith("example:"):
            key = rest.split(":", 1)[1]
            self.set_spec(example_case(key), f"new case from the example {key!r}", path=None)
        elif action == "file:open":
            self._dialog_open()
        elif action == "file:save":
            if self.path:
                self.save_to(self.path)
            else:
                self._dialog_save_as()
        elif action == "file:save-as":
            self._dialog_save_as()
        elif action == "file:import":
            self._dialog_import()
        elif action == "file:export":
            self._dialog_export()
        elif action == "edit:undo":
            self.undo()
        elif action == "edit:redo":
            self.redo()
        elif action == "edit:revert":
            if self.path and os.path.isfile(self.path):
                self.open_path(self.path)
            else:
                self.notify("info", "this case has never been saved, so there is nothing to revert to")
        elif group == "view":
            self.view_opts[rest] = not self.view_opts[rest]
            self.view_menu.items = self._view_items()
            self.log(f"view: {rest} {'on' if self.view_opts[rest] else 'off'}")
            self.log_card.visible = self.view_opts["log"]
            if self.active == "geometry":
                self.show("geometry")
        elif action == "run:check":
            self.show("check")
        elif action == "run:compile":
            self.notify("warning", NOT_BUILT["compile"])
        elif action in ("run:decomposed", "run:full", "run:both"):
            self.notify("warning", NOT_BUILT["run"])
        elif action == "run:stop":
            self.notify("info", "nothing is running")
        elif action == "help:guide":
            self._dialog(self._guide())
        elif action == "help:about":
            self._dialog(self._about())
        elif action == "help:docs":
            self._dialog(self._docs())
        else:
            self.notify("error", f"unknown action {action!r}")

    # ------------------------------------------------------------------
    # layout
    # ------------------------------------------------------------------
    def _build(self) -> None:
        def menu(name, items):
            m = pn.widgets.MenuButton(name=name, items=items, button_type="light",
                                      button_style="solid", width=88, margin=(0, 3),
                                      align="center")
            m.on_click(lambda e: self.dispatch(e.new))
            return m

        examples = [(f"New from example: {label}", f"file:example:{key}")
                    for key, (label, _c, _r) in EXAMPLES.items()]
        self.file_menu = menu("File", [("New blank case", "file:new"), *examples, None,
                                       ("Open...", "file:open"), ("Save", "file:save"),
                                       ("Save as...", "file:save-as"), None,
                                       ("Import JSON...", "file:import"),
                                       ("Export JSON...", "file:export")])
        self.edit_menu = menu("Edit", [("Undo", "edit:undo"), ("Redo", "edit:redo"), None,
                                       ("Revert to saved", "edit:revert")])
        self.view_menu = menu("View", self._view_items())
        self.run_menu = menu("Run", [("Check case", "run:check"),
                                     ("Compile with Atlas (not built)", "run:compile"), None,
                                     ("Run decomposed (not built)", "run:decomposed"),
                                     ("Run full domain (not built)", "run:full"),
                                     ("Run both and compare (not built)", "run:both"),
                                     ("Stop", "run:stop")])
        self.help_menu = menu("Help", [("Workflow guide", "help:guide"),
                                       ("Where things are documented", "help:docs"), None,
                                       ("About the workbench", "help:about")])
        #: the case bar: which case is open and whether it is saved.  In the
        #: workspace rather than the header, where the template gives it no width.
        self.status = pn.pane.HTML(sizing_mode="stretch_width", margin=(0, 10, 6, 10),
                                   styles={"font-size": "13px", "opacity": "0.85"})

        self.nav = pn.widgets.RadioButtonGroup(options=self._nav_options(), value="case",
                                               orientation="vertical", button_type="light",
                                               button_style="outline",
                                               sizing_mode="stretch_width")
        self._syncing_nav = False
        self.nav.param.watch(lambda e: None if self._syncing_nav else self.show(e.new), "value")
        self.issues_pane = pn.pane.Markdown(sizing_mode="stretch_width", margin=(0, 10))

        self.workspace = pn.Column(sizing_mode="stretch_width")
        self.log_pane = pn.pane.HTML(self._log_html(), sizing_mode="stretch_width",
                                     styles={"max-height": "150px", "overflow-y": "auto"})
        self.log_card = pn.Card(self.log_pane, title="Activity", collapsed=False,
                                sizing_mode="stretch_width", margin=(10, 0))
        self.modal_body = pn.Column(sizing_mode="stretch_width")

        #: The Bootstrap template, not the Fast one.  Fast's web components follow
        #: the operating system's dark-mode setting on their own while the page
        #: around them stays light, which put dark magenta-bordered inputs on a
        #: light page and white text on the light sidebar (the owner's screenshot,
        #: 2026-09-28).  Bootstrap draws one consistent light theme whatever the OS
        #: says, has no theme switch that reloads the page, and loads nothing from
        #: outside the local server.
        self.tpl = pn.template.BootstrapTemplate(
            title="Atlas Workbench",
            header=[pn.Row(self.file_menu, self.edit_menu, self.view_menu, self.run_menu,
                           self.help_menu, align="center", margin=(0, 8))],
            sidebar=[pn.pane.Markdown("#### Workflow", margin=(0, 10)), self.nav,
                     pn.pane.Markdown("#### Case check", margin=(10, 10, 0, 10)),
                     self.issues_pane],
            main=[pn.Column(self.status, self.workspace, self.log_card,
                            sizing_mode="stretch_width")],
            modal=[self.modal_body],
            header_background=HEADER, header_color="#ffffff",
            sidebar_width=260,
            raw_css=[LIGHT_CSS],
        )
        self.refresh(rebuild=True)

    def _view_items(self):
        mark = lambda k: "[x] " if self.view_opts[k] else "[ ] "              # noqa: E731
        return [(mark("grid") + "Cell grid", "view:grid"),
                (mark("overlaps") + "Window overlaps", "view:overlaps"),
                (mark("labels") + "Device labels", "view:labels"), None,
                (mark("log") + "Activity log", "view:log")]

    def _nav_options(self) -> dict[str, str]:
        issues = check(self.spec)
        out = {}
        for n, (key, title) in enumerate(STEPS, 1):
            errs = sum(1 for i in issues if i.step == key and i.severity == "error")
            warns = sum(1 for i in issues if i.step == key and i.severity == "warning")
            if key in ("run", "results"):
                mark = "not built"
            elif errs:
                mark = f"{errs} error{'s' * (errs > 1)}"
            elif warns:
                mark = f"{warns} warning{'s' * (warns > 1)}"
            else:
                mark = "ok" if key != "check" else ""
            out[f"{n}. {title}" + (f"  ({mark})" if mark else "")] = key
        return out

    def refresh(self, rebuild: bool) -> None:
        """Header, sidebar and (when asked) the active step's workspace."""
        name = html.escape(self.spec.name)
        if self.path:
            state = "unsaved changes" if self.dirty else "saved"
            where = html.escape(os.path.relpath(self.path, ROOT))
            self.status.object = f"Case <b>{name}</b> &middot; {state} &middot; <code>{where}</code>"
        else:
            state = "unsaved changes" if self.dirty else "not saved yet"
            self.status.object = f"Case <b>{name}</b> &middot; {state}"
        self.nav.options = self._nav_options()
        issues = check(self.spec)
        n = summary(issues)
        lines = [f"**{n['error']}** errors, **{n['warning']}** warnings"]
        lines += [f"- {i.severity}: {html.escape(i.message)}" for i in issues[:6]]
        if len(issues) > 6:
            lines.append(f"- ... and {len(issues) - 6} more (step 4)")
        self.issues_pane.object = "\n".join(lines)
        if rebuild:
            self.show(self.active)

    def show(self, key: str) -> None:
        self.active = key
        if self.nav.value != key:
            #: moving the selector from code must not re-enter show() through
            #: the selector's own watcher and build the view twice
            self._syncing_nav = True
            try:
                self.nav.value = key
            finally:
                self._syncing_nav = False
        view = {"case": self._view_case, "geometry": self._view_geometry,
                "physics": self._view_physics, "check": self._view_check,
                "run": self._view_run, "results": self._view_results}[key]
        self.workspace.objects = [view()]

    def _log_html(self) -> str:
        rows = "".join(f"<div>{html.escape(line)}</div>" for line in self.log_lines)
        return f"<div style='font-family:ui-monospace,Consolas,monospace;font-size:12px'>{rows}</div>"

    @staticmethod
    def _title(text: str, sub: str = "") -> pn.pane.Markdown:
        return pn.pane.Markdown(f"## {text}" + (f"\n{sub}" if sub else ""),
                                sizing_mode="stretch_width", margin=(0, 10))

    # ------------------------------------------------------------------
    # the six step views
    # ------------------------------------------------------------------
    def _bind(self, widget, apply: Callable[[CaseSpec, object], None], label: str):
        widget.param.watch(lambda e: self.edit(lambda s: apply(s, e.new),
                                               f"{label} = {e.new}"), "value")
        return widget

    def _view_case(self):
        s = self.spec
        fams = {f.label: f.id for f in registry.FAMILIES}
        name = self._bind(pn.widgets.TextInput(name="Case name", value=s.name),
                          lambda c, v: setattr(c, "name", v), "name")
        desc = self._bind(pn.widgets.TextAreaInput(name="Description", value=s.description,
                                                   height=80),
                          lambda c, v: setattr(c, "description", v), "description")
        fam = self._bind(pn.widgets.Select(name="Physics family", options=fams,
                                           value=s.physics.family,
                                           disabled_options=[f.id for f in registry.FAMILIES
                                                             if f.status != "ready-to-wire"]),
                         lambda c, v: setattr(c.physics, "family", v), "family")
        nx = self._bind(pn.widgets.IntInput(name="Domain width (cells)", value=s.domain.nx,
                                            start=1),
                        lambda c, v: setattr(c.domain, "nx", v), "domain.nx")
        ny = self._bind(pn.widgets.IntInput(name="Domain height (cells)", value=s.domain.ny,
                                            start=1),
                        lambda c, v: setattr(c.domain, "ny", v), "domain.ny")
        dx = self._bind(pn.widgets.FloatInput(name="Cell size (D)", value=s.domain.dx,
                                              start=1e-6, step=0.001, format="0.00000"),
                        lambda c, v: setattr(c.domain, "dx", v), "domain.dx")
        d = s.domain
        size = pn.pane.Markdown(f"The domain is **{d.nx * d.dx:.3g} x {d.ny * d.dx:.3g} D** "
                                f"({d.nx} x {d.ny} = {d.nx * d.ny:,} cells). It has "
                                f"**{len(s.windows)} windows** and **{len(s.devices)} devices**.")
        notes = pn.pane.Markdown(
            "\n".join(f"- **{f.label}**: {'available' if f.status == 'ready-to-wire' else 'not yet'}. {f.note}"
                      for f in registry.FAMILIES), sizing_mode="stretch_width")
        return pn.Column(
            self._title("1. Case", "Name the case, choose what physics it runs, and size the domain."),
            _wrap(pn.Column(name, desc, fam, width=420),
                   pn.Column(nx, ny, dx, size, width=320)),
            pn.Card(notes, title="Physics families", collapsed=True, sizing_mode="stretch_width"),
            sizing_mode="stretch_width")

    def _geometry_figure(self):
        s, d = self.spec, self.spec.domain
        pad = max(4, int(0.02 * max(d.nx, d.ny)))
        p = figure(x_range=Range1d(-pad, d.nx + pad), y_range=Range1d(-pad, d.ny + pad),
                   match_aspect=True, sizing_mode="stretch_width", height=460,
                   tools="pan,wheel_zoom,box_zoom,reset,save", toolbar_location="right")
        p.xaxis.axis_label, p.yaxis.axis_label = "x (cells)", "y (cells)"
        for g in (p.xgrid, p.ygrid):
            g.visible = self.view_opts["grid"]
            g.ticker = SingleIntervalTicker(interval=16)
            g.grid_line_alpha = 0.35
        p.rect(x=d.nx / 2, y=d.ny / 2, width=d.nx, height=d.ny, fill_alpha=0,
               line_color=C_DOMAIN, line_width=2)
        ws = s.windows
        src = ColumnDataSource(dict(
            x=[w.x0 + w.nx / 2 for w in ws], y=[w.y0 + w.ny / 2 for w in ws],
            w=[w.nx for w in ws], h=[w.ny for w in ws], id=[w.id for w in ws],
            box=[f"x {w.x0}..{w.x0 + w.nx}, y {w.y0}..{w.y0 + w.ny}" for w in ws]))
        wr = p.rect("x", "y", "w", "h", source=src, fill_color=C_WINDOW, fill_alpha=0.07,
                    line_color=C_WINDOW, line_width=1.5)
        p.add_tools(HoverTool(renderers=[wr], tooltips=[("window", "@id"), ("cells", "@box")]))
        if self.view_opts["overlaps"]:
            ov = dict(x=[], y=[], w=[], h=[])
            for i, a in enumerate(ws):
                for b in ws[i + 1:]:
                    x0, x1 = max(a.x0, b.x0), min(a.x0 + a.nx, b.x0 + b.nx)
                    y0, y1 = max(a.y0, b.y0), min(a.y0 + a.ny, b.y0 + b.ny)
                    if x1 > x0 and y1 > y0:
                        ov["x"].append((x0 + x1) / 2)
                        ov["y"].append((y0 + y1) / 2)
                        ov["w"].append(x1 - x0)
                        ov["h"].append(y1 - y0)
            p.rect("x", "y", "w", "h", source=ColumnDataSource(ov), fill_color=C_OVERLAP,
                   fill_alpha=0.22, line_alpha=0)
        if s.devices:
            dsrc = ColumnDataSource(dict(
                x=[v.x / d.dx for v in s.devices],
                y0=[(v.y - v.diameter / 2) / d.dx for v in s.devices],
                y1=[(v.y + v.diameter / 2) / d.dx for v in s.devices],
                id=[v.id for v in s.devices]))
            dr = p.segment("x", "y0", "x", "y1", source=dsrc, line_color=C_ROTOR, line_width=4)
            p.add_tools(HoverTool(renderers=[dr], tooltips=[("device", "@id")]))
            if self.view_opts["labels"]:
                p.add_layout(LabelSet(x="x", y="y1", text="id", source=dsrc, x_offset=4,
                                      y_offset=2, text_font_size="11px", text_color=C_ROTOR))
        return p

    def _view_geometry(self):
        s, d = self.spec, self.spec.domain
        wdf = pd.DataFrame([dict(window=w.id, x0=w.x0, y0=w.y0, width=w.nx, height=w.ny,
                                 width_D=round(w.nx * d.dx, 4), height_D=round(w.ny * d.dx, 4))
                            for w in s.windows])
        ddf = pd.DataFrame([dict(device=v.id, kind=v.kind, x_D=v.x, y_D=v.y,
                                 diameter_D=v.diameter, yaw_deg=v.yaw_deg) for v in s.devices])
        tables = _wrap(
            pn.Column("**Windows**", pn.widgets.Tabulator(wdf, disabled=True, show_index=False,
                                                          height=220, layout="fit_data_table")),
            pn.Column("**Devices**", pn.widgets.Tabulator(ddf, disabled=True, show_index=False,
                                                          height=220, layout="fit_data_table")))
        return pn.Column(
            self._title("2. Geometry", "The domain, the windows it is cut into, and the devices in it."),
            pn.pane.Alert(NOT_BUILT["geometry"], alert_type="info", sizing_mode="stretch_width"),
            pn.pane.Bokeh(self._geometry_figure(), sizing_mode="stretch_width"),
            pn.pane.Markdown(f"Windows teal, overlaps amber, rotor disks red. "
                             f"{len(s.windows)} windows, {len(s.devices)} devices. "
                             "Toggle the grid, overlaps and labels in **View**."),
            tables, sizing_mode="stretch_width")

    def _view_physics(self):
        s = self.spec
        nu = self._bind(pn.widgets.FloatInput(name="Viscosity nu", value=s.physics.nu,
                                              start=1e-9, step=1e-4, format="0.00000"),
                        lambda c, v: setattr(c.physics, "nu", v), "nu")
        u = self._bind(pn.widgets.FloatInput(name="Freestream speed", value=s.physics.u_inf,
                                             start=1e-6, step=0.1),
                       lambda c, v: setattr(c.physics, "u_inf", v), "u_inf")
        dt = self._bind(pn.widgets.FloatInput(name="Macro-step", value=s.run.macro_dt,
                                              start=1e-6, step=0.05),
                        lambda c, v: setattr(c.run, "macro_dt", v), "macro_dt")
        ramp = self._bind(pn.widgets.IntInput(name="Ramp width (cells)",
                                              value=s.coupling.ramp_cells, start=1),
                          lambda c, v: setattr(c.coupling, "ramp_cells", v), "ramp_cells")
        asm = self._bind(pn.widgets.Select(name="Assembly", value=s.coupling.assembly,
                                           options={"blend, then one global projection": "projected",
                                                    "blend only": "blend"}),
                         lambda c, v: setattr(c.coupling, "assembly", v), "assembly")
        ell = self._bind(pn.widgets.Select(name="Pressure solve", value=s.coupling.elliptic,
                                           options={"exposed: once, on the assembled field": "exposed",
                                                    "embedded: inside every window": "embedded"}),
                         lambda c, v: setattr(c.coupling, "elliptic", v), "elliptic")
        fam = registry.family(s.physics.family)
        why = pn.pane.Markdown(
            "**Why the defaults.** The projected assembly with the pressure solve exposed is "
            "the arrangement the measurements selected: every other combination leaves its "
            "velocity band within 80 macro-steps at six windows (W100). The ramp is the "
            "partition of unity's width; windows that meet must overlap by at least twice it.",
            sizing_mode="stretch_width")
        return pn.Column(
            self._title("3. Physics & coupling", f"{fam.label}."),
            _wrap(pn.Column("**Flow**", nu, u, dt, width=320),
                   pn.Column("**Coupling between windows**", ramp, asm, ell, width=360)),
            why,
            pn.Card(pn.pane.Markdown("\n".join([f"- {x}" for x in fam.solvers]
                                               + [f"\n{fam.note}"]
                                               + [f"\nSources: {', '.join(fam.sources)}"])),
                    title="What this family runs on", collapsed=True, sizing_mode="stretch_width"),
            sizing_mode="stretch_width")

    def _view_check(self):
        issues = check(self.spec)
        n = summary(issues)
        df = pd.DataFrame([dict(severity=i.severity, step=i.step, issue=i.message)
                           for i in issues] or [dict(severity="", step="", issue="no issues")])
        again = pn.widgets.Button(name="Check again", button_type="primary", width=140)
        again.on_click(lambda e: self.show("check"))
        compile_btn = pn.widgets.Button(name="Compile with the Atlas compiler",
                                        button_type="default", disabled=True, width=260)
        self.log(f"checked the case: {n['error']} errors, {n['warning']} warnings")
        return pn.Column(
            self._title("4. Check", "What is wrong with the case before anything runs."),
            pn.Row(again, compile_btn),
            pn.pane.Markdown(f"**{n['error']} errors, {n['warning']} warnings.** "
                             f"*Compile:* {NOT_BUILT['compile']}"),
            pn.widgets.Tabulator(df, disabled=True, show_index=False, layout="fit_data_stretch",
                                 sizing_mode="stretch_width", height=260),
            sizing_mode="stretch_width")

    def _view_run(self):
        s = self.spec
        steps = self._bind(pn.widgets.IntInput(name="Macro-steps", value=s.run.steps, start=1),
                           lambda c, v: setattr(c.run, "steps", v), "run.steps")
        threads = self._bind(pn.widgets.IntSlider(name="Threads for the windows",
                                                  value=min(s.run.threads, os.cpu_count() or 1),
                                                  start=1, end=os.cpu_count() or 1),
                             lambda c, v: setattr(c.run, "threads", v), "run.threads")
        arms = pn.widgets.CheckBoxGroup(options=["Decomposed, serial", "Decomposed, parallel",
                                                 "Full domain"],
                                        value=["Decomposed, parallel", "Full domain"],
                                        inline=False, disabled=True)
        run_btn = pn.widgets.Button(name="Run and compare", button_type="primary",
                                    disabled=True, width=180)
        stop_btn = pn.widgets.Button(name="Stop", disabled=True, width=90)
        panes = [pn.Card(pn.pane.Markdown("*appears here during a run*"), title=t,
                         width=240, height=220, collapsible=False)
                 for t in ("Decomposed field", "Full-domain field", "Difference")]
        return pn.Column(
            self._title("5. Run & compare", "March the decomposition and the full-domain "
                                            "solve in turns, on this machine."),
            pn.pane.Alert(NOT_BUILT["run"], alert_type="warning", sizing_mode="stretch_width"),
            _wrap(pn.Column(steps, threads, width=320), pn.Column("**Arms**", arms, width=230),
                   pn.Column(run_btn, stop_btn,
                             pn.indicators.Progress(value=0, max=100, width=180, active=False))),
            _wrap(*panes),
            sizing_mode="stretch_width")

    def _view_results(self):
        cols = ["arm", "seconds per step", "vs full domain", "farm power",
                "farm power vs full domain", "rms velocity difference"]
        return pn.Column(
            self._title("6. Results", "Speed and accuracy of each arm against the full domain."),
            pn.pane.Alert(NOT_BUILT["results"], alert_type="info", sizing_mode="stretch_width"),
            pn.widgets.Tabulator(pd.DataFrame(columns=cols), disabled=True, show_index=False,
                                 sizing_mode="stretch_width", height=160),
            pn.pane.Markdown("For this family, the measured answer is on the wiki page "
                             "`decomposition-speed-by-rotor-count`: 3.4-5.7x faster from 5 to "
                             "21 rotors in parallel, farm power within 2.3-8.3%."),
            sizing_mode="stretch_width")

    # ------------------------------------------------------------------
    # dialogs
    # ------------------------------------------------------------------
    def _dialog(self, *objs) -> None:
        close = pn.widgets.Button(name="Close", width=90)
        close.on_click(lambda e: self.tpl.close_modal())
        self.modal_body.objects = [*objs, close]
        self.tpl.open_modal()

    def _dialog_open(self) -> None:
        files = self.case_files()
        rel = {os.path.basename(f): f for f in files}
        if not rel:
            self._dialog(pn.pane.Markdown(
                f"### Open a case\nNo saved cases in `{os.path.relpath(self.cases_dir, ROOT)}` "
                "yet. Use **File > Save** first, or **File > Import JSON**."))
            return
        pick = pn.widgets.Select(name="Saved cases (newest first)", options=rel, width=360)
        go = pn.widgets.Button(name="Open", button_type="primary", width=90)

        def do(_e):
            self.tpl.close_modal()
            self.open_path(pick.value)
        go.on_click(do)
        self._dialog(pn.pane.Markdown("### Open a case"), pick, go)

    def _dialog_save_as(self) -> None:
        name = pn.widgets.TextInput(name="File name", value=slug(self.spec.name), width=300)
        warn = pn.pane.Markdown("")
        go = pn.widgets.Button(name="Save", button_type="primary", width=90)
        target = lambda: os.path.join(self.cases_dir, slug(name.value) + ".json")    # noqa: E731

        def check_name(_e=None):
            exists = os.path.isfile(target())
            warn.object = (f"`{os.path.relpath(target(), ROOT)}` exists and will be replaced."
                           if exists else f"Saves to `{os.path.relpath(target(), ROOT)}`.")
            go.name = "Replace" if exists else "Save"
        name.param.watch(check_name, "value")
        check_name()

        def do(_e):
            self.tpl.close_modal()
            self.save_to(target())
        go.on_click(do)
        self._dialog(pn.pane.Markdown("### Save the case as"), name, warn, go)

    def _dialog_import(self) -> None:
        up = pn.widgets.FileInput(accept=".json", multiple=False)

        def got(e):
            if e.new:
                self.tpl.close_modal()
                self.import_text(e.new.decode("utf-8"), up.filename or "file")
        up.param.watch(got, "value")
        self._dialog(pn.pane.Markdown("### Import a case (JSON)"), up)

    def _dialog_export(self) -> None:
        dl = pn.widgets.FileDownload(callback=lambda: io.StringIO(self.spec.to_json()),
                                     filename=slug(self.spec.name) + ".json",
                                     button_type="primary", width=220,
                                     label="Download the case as JSON")
        self._dialog(pn.pane.Markdown("### Export the case"), dl)

    def _guide(self):
        return pn.pane.Markdown(
            "### The workflow\n"
            "1. **Case**: name it, choose the physics family, size the domain.\n"
            "2. **Geometry**: cut the domain into windows and place the devices "
            "*(editing tools being designed together)*.\n"
            "3. **Physics & coupling**: the flow constants, and how windows are joined.\n"
            "4. **Check**: what is wrong before anything runs; later, the Atlas compiler's "
            "verdict per seam.\n"
            "5. **Run & compare**: march the decomposition and the full-domain solve in turns "
            "*(not built)*.\n"
            "6. **Results**: speed, farm power and the field difference *(not built)*.\n\n"
            "Every change can be undone (**Edit > Undo**). A case is one JSON file; "
            "**File > Save** writes it to the cases folder.", width=560)

    def _about(self):
        return pn.pane.Markdown(
            f"### Atlas Workbench {VERSION}\n"
            "A tool to build a domain decomposition by hand, run it, and compare it with the "
            "full-domain solve. This is the shell: the menus, the workflow and the case file.\n\n"
            f"- Cases folder: `{os.path.relpath(self.cases_dir, ROOT)}`\n"
            "- Built on **Panel** and **Bokeh** (both BSD-3-Clause, already installed; "
            "nothing is downloaded and the page makes no outside requests).\n"
            "- Unbuilt actions stay visible and say why.", width=560)

    def _docs(self):
        return pn.pane.Markdown(
            "### Where things are documented\n"
            "- The plan this builds: `wiki/concepts/Atlas 0.1/atlas-0.1-outcome/"
            "outcome-c4-path-to-declarative-cases.md`\n"
            "- What the wind-farm family measured: `wiki/concepts/Atlas 0.1/common/"
            "decomposition-speed-by-rotor-count.md`\n"
            "- The case file's schema: `atlas/workbench/spec.py`\n"
            "- The physics families and what each is missing: `atlas/workbench/registry.py`",
            width=620)


def _wrap(*objs):
    """A row that wraps onto the next line instead of clipping at the edge."""
    return pn.FlexBox(*objs, flex_wrap="wrap", gap="8px 16px", sizing_mode="stretch_width")


def create_app(cases_dir: str | None = None):
    """One session.  ``?case=<file>.json`` opens that saved case on load."""
    pn.extension("tabulator", notifications=True)
    wb = Workbench(cases_dir)
    wanted = (pn.state.session_args or {}).get("case", [b""])[0]
    wanted = wanted.decode("utf-8") if isinstance(wanted, bytes) else str(wanted)
    if wanted:
        path = case_from_url(wanted, wb.cases_dir)
        if path:
            wb.open_path(path)
            wb.history.clear()          # undo must not step back into the example
        else:
            wb.notify("warning", f"no saved case called {os.path.basename(wanted)!r}")
    return wb.tpl


def case_from_url(value: str, cases_dir: str) -> str | None:
    """The saved case a ``?case=`` argument names, or None.

    Only a file name, looked up inside the cases folder: a URL must not be able to
    point the server at any other path (``../`` and absolute paths are cut to
    their last component).
    """
    name = os.path.basename(value.replace("\\", "/"))
    if not name.endswith(".json"):
        return None
    path = os.path.join(cases_dir, name)
    return path if os.path.isfile(path) else None


def serve(port: int = 8020, open_browser: bool = False, cases_dir: str | None = None):
    pn.extension("tabulator", notifications=True)
    print(f"Atlas Workbench on http://127.0.0.1:{port}/  (Ctrl-C to stop)", flush=True)
    pn.serve({"/": lambda: create_app(cases_dir)}, port=port, address="127.0.0.1",
             show=open_browser, title="Atlas Workbench",
             websocket_origin=[f"127.0.0.1:{port}", f"localhost:{port}"])


__all__ = ["Workbench", "create_app", "case_from_url", "serve", "STEPS",
           "NOT_BUILT"]
