"""Atlas Workbench.

The decomposition GUI, built on Panel and Bokeh (both BSD-3, both already
installed, nothing downloaded).  What works: cases (new, examples, open, save,
save as, import, export), undo and redo, the case and physics forms, the
geometry section (`editor.py`: windows, regions, devices and boundaries drawn
and edited on a canvas, a tiling generator, and import from Gmsh), the check,
and the runner (`runner.py`, `runview.py`): a committed case marched decomposed
(serially and on threads) and on the full domain in turns, in a background
thread, streamed to the page, stoppable, and recorded beside the case file.
What does not work yet says so where it would be, with the reason.

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
from pydantic import ValidationError

from . import geometry as geo
from . import registry
from .editor import GeometryEditor
from .gmsh_import import GmshImportError, read_msh_bytes
from .runner import ARM_LABELS, CaseRun, RunRefused, results_dir_for
from .runview import RunPanel, results_view
from .spec import (EXAMPLES, Boundary, CaseSpec, Region, Window, blank_case, check,
                   example_case, slug, summary)

#: Native form controls and scrollbars follow the page, not the OS dark-mode setting.
LIGHT_CSS = ":root { color-scheme: light; }"
MENU_CSS = ".bk-menu { width: max-content; min-width: 100%; }"

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CASES_DIR = os.path.join(ROOT, "out", "workbench", "cases")
VERSION = "0.3 (runner)"
#: How often the page reads a running case's progress, in milliseconds.
POLL_MS = 400

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
    "compile": ("Compiling a case with the Atlas compiler needs the case-to-graph loader "
                "(plan step 2). Not built yet."),
}

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
        #: the geometry section's layer, tool, snap and new-region material; kept
        #: here so they survive the editor being rebuilt
        self.geo_state = {"layer": "windows", "tool": "move", "snap": geo.DEFAULT_SNAP,
                          "material": "material-1"}
        self.geo_editor: GeometryEditor | None = None
        self._issues_memo: tuple[str, list] | None = None
        self._keep_view = False
        self.active = "case"
        self.log_lines: list[str] = []
        #: the run marching now, or the last one; its record is the Results step
        self.run: CaseRun | None = None
        #: the arms the Run step has ticked (a run choice, not part of the case)
        self.run_arms: list[str] = ["serial", "parallel", "full"]
        self.run_panel: RunPanel | None = None
        #: the Run step's (run, stop, blocked) buttons, so the poll can switch them
        self._run_buttons: tuple | None = None
        self._poll_cb = None
        self._reported: CaseRun | None = None
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

    def issues(self):
        """`check(self.spec)`, computed once per version of the case."""
        key = self.spec.to_json()
        if self._issues_memo is None or self._issues_memo[0] != key:
            self._issues_memo = (key, check(self.spec))
        return self._issues_memo[1]

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
        elif action == "file:import-gmsh":
            self.dialog_gmsh()
        elif action == "geometry:tiling":
            self.dialog_tiling()
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
                self.show("geometry", keep_view=True)
        elif action == "run:check":
            self.show("check")
        elif action == "run:compile":
            self.notify("warning", NOT_BUILT["compile"])
        elif action == "run:both":
            self.start_run(self.run_arms)
        elif action == "run:decomposed":
            self.start_run(["serial", "parallel"])
        elif action == "run:full":
            self.start_run(["full"])
        elif action == "run:stop":
            self.stop_run()
        elif action == "run:results":
            self.show("results")
        elif action == "help:guide":
            self._dialog(self._guide())
        elif action == "help:about":
            self._dialog(self._about())
        elif action == "help:docs":
            self._dialog(self._docs())
        else:
            self.notify("error", f"unknown action {action!r}")

    # ------------------------------------------------------------------
    # running
    # ------------------------------------------------------------------
    def start_run(self, arms, blocking: bool = False) -> CaseRun | None:
        """March the case as it stands now, in turns, in a background thread.

        The case is copied here -- the commit -- so an edit made while it marches
        does not reach the run, and the page says which version is marching.
        ``blocking`` runs it in the calling thread (for tests and scripts).
        """
        if self.run is not None and self.run.active:
            self.notify("warning", "a run is already marching; Stop it first (Run > Stop)")
            return None
        errors = [i for i in self.issues() if i.severity == "error"]
        if errors:
            self.notify("error", f"the case has {len(errors)} error"
                                 f"{'s' * (len(errors) > 1)} and cannot run; step 4 lists "
                                 f"them: {errors[0].message}")
            self.show("check")
            return None
        try:
            run = CaseRun(self.spec, arms=tuple(arms), steps=self.spec.run.steps,
                          threads=self.spec.run.threads,
                          results_dir=results_dir_for(self.path, self.cases_dir,
                                                      self.spec.name),
                          case_path=self.path)
        except (RunRefused, KeyError) as exc:
            self.notify("error", f"cannot run: {exc}")
            return None
        self.run = run
        self._reported = None
        self.log(f"run started: {run.label()}")
        if blocking:
            self.show("run")
            run.run_blocking()
            self.poll()
        else:
            run.start()                    # active from here, so the view shows Stop live
            self.show("run")
            self._start_polling()
        return run

    def stop_run(self) -> None:
        if self.run is None or not self.run.active:
            self.notify("info", "nothing is running")
            return
        self.run.stop()
        self.log("stop requested; the run stops after the arm-step in progress")
        self.poll()

    def _start_polling(self) -> None:
        if self._poll_cb is not None or pn.state.curdoc is None:
            return
        self._poll_cb = pn.state.add_periodic_callback(self.poll, period=POLL_MS)

    def _stop_polling(self) -> None:
        if self._poll_cb is not None:
            try:
                self._poll_cb.stop()
            except Exception:                                  # pragma: no cover
                pass
            self._poll_cb = None

    def poll(self) -> None:
        """Read the run's progress into the page; report once when it ends."""
        run = self.run
        if run is None:
            return
        if self.run_panel is not None and self.active == "run" and self.run_panel.run is run:
            self.run_panel.update()
            if self._run_buttons is not None:
                run_btn, stop_btn, blocked = self._run_buttons
                run_btn.disabled = run.active or blocked
                stop_btn.disabled = not run.active
        label = self._nav_options()
        if list(label) != list(self.nav.options):
            self._syncing_nav = True
            try:
                self.nav.options = label
            finally:
                self._syncing_nav = False
        if run.active or self._reported is run:
            return
        self._reported = run
        self._stop_polling()
        p = run.progress()
        if run.status == "failed":
            self.notify("error", f"the run failed: {p.error}")
        else:
            res = run.results or {}
            fails = [c["title"] for c in res.get("checks", []) if c["passed"] is False]
            what = "stopped" if run.status == "stopped" else "finished"
            self.notify("warning" if (fails or run.status == "stopped") else "success",
                        f"run {what} after {p.step} macro-steps"
                        + (f"; FAILED: {', '.join(fails)}" if fails else
                           "; every measured check passed")
                        + (f"; saved {os.path.relpath(run.record_path, ROOT)}"
                           if run.record_path else ""))
        #: the step was built while the run marched (its notice, its buttons); the
        #: finished run's step is a different view, so it is rebuilt
        if self.active in ("run", "results"):
            self.show(self.active, keep_view=True)

    # ------------------------------------------------------------------
    # layout
    # ------------------------------------------------------------------
    def _build(self) -> None:
        def menu(name, items):
            #: the dropdown is as wide as the button unless told otherwise, and the
            #: longer items then spill over the page
            m = pn.widgets.MenuButton(name=name, items=items, button_type="light",
                                      button_style="solid", width=88, margin=(0, 3),
                                      align="center", stylesheets=[MENU_CSS])
            m.on_click(lambda e: self.dispatch(e.new))
            return m

        examples = [(f"New from example: {ex.label}", f"file:example:{key}")
                    for key, ex in EXAMPLES.items()]
        self.file_menu = menu("File", [("New blank case", "file:new"), *examples, None,
                                       ("Open...", "file:open"), ("Save", "file:save"),
                                       ("Save as...", "file:save-as"), None,
                                       ("Import JSON...", "file:import"),
                                       ("Export JSON...", "file:export"), None,
                                       ("Import geometry from Gmsh (.msh)...",
                                        "file:import-gmsh")])
        self.edit_menu = menu("Edit", [("Undo", "edit:undo"), ("Redo", "edit:redo"), None,
                                       ("Generate a window tiling...", "geometry:tiling"),
                                       None, ("Revert to saved", "edit:revert")])
        self.view_menu = menu("View", self._view_items())
        self.run_menu = menu("Run", [("Check case", "run:check"),
                                     ("Compile with Atlas (not built)", "run:compile"), None,
                                     ("Run decomposed (serial and parallel)", "run:decomposed"),
                                     ("Run full domain", "run:full"),
                                     ("Run the ticked arms and compare", "run:both"),
                                     ("Stop", "run:stop"), None,
                                     ("Show the results", "run:results")])
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
        issues = self.issues()
        out = {}
        total_errs = sum(1 for i in issues if i.severity == "error")
        for n, (key, title) in enumerate(STEPS, 1):
            errs = sum(1 for i in issues if i.step == key and i.severity == "error")
            warns = sum(1 for i in issues if i.step == key and i.severity == "warning")
            if key == "run":
                r = self.run
                if r is not None and r.active:
                    mark = f"running {r.progress().step}/{r.steps}"
                else:
                    mark = "blocked by errors" if total_errs else "ready"
            elif key == "results":
                r = self.run
                if r is None or r.active:
                    mark = "no run yet" if r is None else "after the run"
                elif r.status == "failed":
                    mark = "run failed"
                else:
                    fails = sum(1 for c in (r.results or {}).get("checks", [])
                                if c["passed"] is False)
                    mark = (f"{fails} check{'s' * (fails > 1)} failed" if fails
                            else ("stopped" if r.status == "stopped" else "done"))
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
        issues = self.issues()
        n = summary(issues)
        lines = [f"**{n['error']}** errors, **{n['warning']}** warnings"]
        lines += [f"- {i.severity}: {html.escape(i.message)}" for i in issues[:6]]
        if len(issues) > 6:
            lines.append(f"- ... and {len(issues) - 6} more (step 4)")
        self.issues_pane.object = "\n".join(lines)
        if rebuild:
            self.show(self.active, keep_view=True)
        elif self.active == "geometry" and self.geo_editor is not None:
            d = self.spec.domain
            if self.geo_editor._domain == (d.nx, d.ny):
                self.geo_editor.sync()
            else:
                self.show("geometry")                     # the canvas is the domain's size

    def show(self, key: str, keep_view: bool = False) -> None:
        self._keep_view = keep_view
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

    def _view_geometry(self):
        d = self.spec.domain
        old = self.geo_editor
        ranges = (old.ranges() if (self._keep_view and old is not None
                                   and old._domain == (d.nx, d.ny)) else None)
        self.geo_editor = GeometryEditor(self, ranges)
        return pn.Column(
            self._title("2. Geometry", "Cut the domain into windows, lay out materials, place "
                                       "the devices and set the boundaries."),
            self.geo_editor.view(), sizing_mode="stretch_width")

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
        issues = self.issues()
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
        cpus = os.cpu_count() or 1
        steps = self._bind(pn.widgets.IntInput(name="Macro-steps", value=s.run.steps, start=1,
                                               width=150),
                           lambda c, v: setattr(c.run, "steps", v), "run.steps")
        threads = self._bind(pn.widgets.IntSlider(name="Threads (parallel arm)",
                                                  value=min(s.run.threads, cpus),
                                                  start=1, end=cpus, width=260),
                             lambda c, v: setattr(c.run, "threads", v), "run.threads")
        opts = {ARM_LABELS[a]: a for a in ("serial", "parallel", "full")}
        arms = pn.widgets.CheckBoxGroup(options=opts, value=list(self.run_arms), inline=False)

        def set_arms(e):
            self.run_arms = [a for a in ("serial", "parallel", "full") if a in e.new]
        arms.param.watch(set_arms, "value")
        running = self.run is not None and self.run.active
        n_err = sum(1 for i in self.issues() if i.severity == "error")
        try:
            fam = registry.family(s.physics.family)
        except KeyError:
            fam = None
        runnable = fam is not None and bool(fam.adapter)
        run_btn = pn.widgets.Button(name="Run and compare", button_type="primary", width=170,
                                    disabled=running or bool(n_err) or not runnable)
        run_btn.on_click(lambda e: self.dispatch("run:both"))
        stop_btn = pn.widgets.Button(name="Stop", button_type="danger", width=90,
                                     disabled=not running)
        stop_btn.on_click(lambda e: self.dispatch("run:stop"))
        self._run_buttons = (run_btn, stop_btn, bool(n_err) or not runnable)
        why = []
        if not runnable:
            why.append(f"The {fam.label if fam else s.physics.family} family has no runner "
                       f"yet. {fam.note if fam else ''}")
        if n_err:
            why.append(f"The case has {n_err} error{'s' * (n_err > 1)}; step 4 lists them.")
        if running:
            why.append("A run is marching. Edits you make now do not reach it: it marches "
                       "the copy committed when Run was pressed.")
        self.run_panel = RunPanel(self, self.run)
        body = [self._title("5. Run & compare", "March the decomposition and the full-domain "
                                                "solve in turns, on this machine: every "
                                                "macro-step each ticked arm takes one step, "
                                                "in a rotating order.")]
        if why:
            body.append(pn.pane.Alert(" ".join(why), alert_type="warning",
                                      sizing_mode="stretch_width"))
        body.append(_wrap(pn.Column(steps, threads, width=280),
                          pn.Column("**Arms**", arms, width=220),
                          pn.Column(run_btn, stop_btn, width=190)))
        if self.run is None:
            body.append(pn.pane.Markdown("*Nothing has run in this session yet. The fields, "
                                         "the time per step of each arm, and the family's "
                                         "own metric appear here while it marches.*",
                                         margin=(0, 10)))
        else:
            body.append(self.run_panel.live_view())
            if not self.run.active:
                goto = pn.widgets.Button(name="Show the results", button_type="primary",
                                         width=170)
                goto.on_click(lambda e: self.dispatch("run:results"))
                body.append(goto)
        return pn.Column(*body, sizing_mode="stretch_width")

    def _view_results(self):
        res = None if self.run is None or self.run.active else self.run.results
        body = [self._title("6. Results", "Speed and accuracy of each arm against the full "
                                          "domain, and the case's sanity checks, from the "
                                          "last run's record.")]
        if self.run is not None and self.run.active:
            body.append(pn.pane.Alert("A run is marching; its results appear here when it "
                                      "ends.", alert_type="info", sizing_mode="stretch_width"))
        elif self.run is not None and self.run.status == "failed":
            body.append(pn.pane.Alert(f"The last run failed: {html.escape(self.run.progress().error or '')}",
                                      alert_type="danger", sizing_mode="stretch_width"))
        else:
            body.append(results_view(self, res))
        return pn.Column(*body, sizing_mode="stretch_width")

    # ------------------------------------------------------------------
    # dialogs
    # ------------------------------------------------------------------
    def _dialog(self, *objs) -> None:
        close = pn.widgets.Button(name="Close", width=90)
        close.on_click(lambda e: self.tpl.close_modal())
        self.modal_body.objects = [*objs, close]
        self.tpl.open_modal()

    def dialog_tiling(self) -> None:
        d, ramp = self.spec.domain, self.spec.coupling.ramp_cells
        cols = pn.widgets.IntInput(name="Columns", value=3, start=1, end=40, width=110)
        rows = pn.widgets.IntInput(name="Rows", value=2, start=1, end=40, width=110)
        ov = pn.widgets.IntInput(name="Overlap (cells)", value=2 * ramp, start=0, width=130)
        note = pn.pane.Markdown(sizing_mode="stretch_width")
        go = pn.widgets.Button(name="Replace the windows", button_type="primary", width=200)

        def preview(_e=None):
            try:
                t = geo.tile(d.nx, d.ny, cols.value or 1, rows.value or 1, ov.value or 0)
            except ValueError as exc:
                note.object = f"**Does not fit:** {exc}."
                go.disabled = True
                return
            _n, (_x, _y, w, h) = t[0]
            short = "" if (ov.value or 0) >= 2 * ramp else (
                f" **An overlap under {2 * ramp} cells (twice the ramp) leaves cells with no "
                f"window at full weight, and the check will refuse it.**")
            note.object = (f"{len(t)} windows of **{w} x {h}** cells on the "
                           f"{d.nx} x {d.ny} domain, overlapping by at least {ov.value} cells, "
                           f"spread so the first starts at 0 and the last ends at the edge."
                           f"{short} This replaces the {len(self.spec.windows)} windows there "
                           f"now (Undo brings them back).")
            go.disabled = False
        for wdg in (cols, rows, ov):
            wdg.param.watch(preview, "value")
        preview()

        def do(_e):
            t = geo.tile(d.nx, d.ny, cols.value, rows.value, ov.value)
            self.tpl.close_modal()

            def apply(c):
                c.windows = [Window(id=n, x0=b[0], y0=b[1], nx=b[2], ny=b[3]) for n, b in t]
            self.edit(apply, f"generated a {cols.value} x {rows.value} tiling, overlap "
                             f"{ov.value}")
        go.on_click(do)
        self._dialog(pn.pane.Markdown("### Generate a window tiling"),
                     pn.Row(cols, rows, ov), note, go)

    def dialog_gmsh(self) -> None:
        up = pn.widgets.FileInput(accept=".msh", multiple=False)
        dx = pn.widgets.FloatInput(name="Cell size, in the file's length unit",
                                   value=self.spec.domain.dx, start=1e-9, step=0.001,
                                   format="0.000000", width=260)
        go = pn.widgets.Button(name="Import", button_type="primary", width=110)
        guide = pn.pane.Markdown(
            "### Import geometry from Gmsh\n"
            "Draw the geometry in Gmsh, name its **physical groups**, then *Mesh > 2D* and "
            "*File > Save Mesh*. The names this reads:\n\n"
            "| physical group | dimension | becomes |\n|---|---|---|\n"
            "| `domain` | surface | the domain's extent (optional) |\n"
            "| `region:<material>` | surface | a material region, holes kept |\n"
            "| `window:<id>` | surface | a window (a rectangle on cell boundaries) |\n"
            "| `bc:<kind>` or `bc:<kind>=<value>` | curve | a boundary on the domain's edge |\n\n"
            "Layers the file defines replace the case's; the others are kept. Only `.msh` "
            "is read: a `.geo` file is a script and could run commands.",
            width=620)

        def do(_e):
            if not up.value:
                self.notify("warning", "choose a .msh file first")
                return
            try:
                g = read_msh_bytes(up.value, float(dx.value or 0), up.filename or "upload.msh")
                regions = [Region(**r) for r in g.regions]
                windows = [Window(**w) for w in g.windows]
                bounds = [Boundary(**b) for b in g.boundaries]
            except (GmshImportError, ValidationError, ValueError) as exc:
                self.notify("error", f"could not import {up.filename}: {exc}")
                return
            self.tpl.close_modal()
            self.apply_gmsh(g, float(dx.value), up.filename or "upload.msh",
                            regions, windows, bounds)
        go.on_click(do)
        self._dialog(guide, up, dx, go)

    def apply_gmsh(self, g, cell: float, filename: str, regions=None, windows=None,
                   bounds=None) -> bool:
        """Put an imported Gmsh geometry into the case; layers it lacks are kept."""
        regions = [Region(**r) for r in g.regions] if regions is None else regions
        windows = [Window(**w) for w in g.windows] if windows is None else windows
        bounds = [Boundary(**b) for b in g.boundaries] if bounds is None else bounds

        def apply(c):
            c.domain.nx, c.domain.ny, c.domain.dx = g.nx, g.ny, cell
            if regions:
                c.regions = regions
            if windows:
                c.windows = windows
            if bounds:
                c.boundaries = bounds
        what = (f"{g.nx} x {g.ny} cells, {len(regions)} regions, {len(windows)} windows, "
                f"{len(bounds)} boundary segments")
        ok = self.edit(apply, f"imported geometry from {filename}: {what}")
        if ok and (g.notes or g.ignored):
            extra = [f"- {n}" for n in g.notes] + [f"- ignored: {n}" for n in g.ignored]
            self._dialog(pn.pane.Markdown(f"### Imported {html.escape(filename)}\n{what}.\n\n"
                                          + "\n".join(extra), width=560))
        return ok

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
            "2. **Geometry**: cut the domain into windows, lay out material regions, place "
            "the devices and set the boundaries, on a canvas or from Gmsh.\n"
            "3. **Physics & coupling**: the flow constants, and how windows are joined.\n"
            "4. **Check**: what is wrong before anything runs; later, the Atlas compiler's "
            "verdict per seam *(the compile is not built yet)*.\n"
            "5. **Run & compare**: march the decomposition (serially and on threads) and the "
            "full-domain solve in turns, with the fields, the difference and the time per "
            "step live. **Run > Stop** stops after the arm-step in progress.\n"
            "6. **Results**: each arm's time per step and its ratio to the full domain, the "
            "family's own metric, the field difference, and the sanity checks against their "
            "registered tolerances. Saved beside the case file as JSON.\n\n"
            "Every change can be undone (**Edit > Undo**). A case is one JSON file; "
            "**File > Save** writes it to the cases folder.", width=560)

    def _about(self):
        return pn.pane.Markdown(
            f"### Atlas Workbench {VERSION}\n"
            "A tool to build a domain decomposition by hand, run it, and compare it with the "
            "full-domain solve. Built so far: the menus, the workflow, the case file, the "
            "geometry section, and the runner for the wind-farm family.\n\n"
            f"- Cases folder: `{os.path.relpath(self.cases_dir, ROOT)}`\n"
            "- Built on **Panel** and **Bokeh** (both BSD-3-Clause, already installed; "
            "nothing is downloaded and the page makes no outside requests).\n"
            "- Unbuilt actions stay visible and say why.", width=560)

    def _docs(self):
        return pn.pane.Markdown(
            "### Where things are documented\n"
            "- The plan this builds: `wiki/concepts/Atlas 0.1/atlas-0.1-outcome/"
            "outcome-c4-path-to-declarative-cases.md` and `showcase-library-plan.md`\n"
            "- The geometry rules and the Gmsh naming convention: "
            "`atlas/workbench/geometry.py`, `atlas/workbench/gmsh_import.py`\n"
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
    #: every request in the server's log, so a check of the served page can see the
    #: GET it made reach THIS server (a reused server once served old code)
    import logging
    acc = logging.getLogger("tornado.access")
    acc.setLevel(logging.INFO)
    if not acc.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        acc.addHandler(h)
        acc.propagate = False
    print(f"Atlas Workbench on http://127.0.0.1:{port}/  (Ctrl-C to stop)", flush=True)
    pn.serve({"/": lambda: create_app(cases_dir)}, port=port, address="127.0.0.1",
             show=open_browser, title="Atlas Workbench",
             websocket_origin=[f"127.0.0.1:{port}", f"localhost:{port}"])


__all__ = ["Workbench", "create_app", "case_from_url", "serve", "STEPS",
           "NOT_BUILT"]
