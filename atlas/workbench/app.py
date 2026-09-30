"""Atlas Workbench.

The decomposition GUI, built on Panel and Bokeh (both BSD-3, both already
installed, nothing downloaded).

**One screen to model, one to run** (2026-09-29, the owner: "go through the entire
UI to make it much more simple and intuitive... plan it out in claude design
first"; the plan is seven screens on a Claude Design canvas).  The page has:

* a header: the case's name, the two tabs, the physics, Undo and Redo, one **Case**
  menu (new, examples, open, save, import, export, Gmsh, history), a **status chip**
  that says whether the case can run and lists its problems, each linked to where
  it is fixed, and **Run**;
* **Model** (`editor.py`, `inspector.py`): a layer rail, one canvas with five tools,
  and an inspector for what is selected;
* **Run & results** (`runview.py`): Run and the Atlas compiler's verdict at the top;
  the run marching live -- the fields, the time per step, the convergence -- and
  when it ends four cards (speed, agreement, balance, checks) over the details and
  the fields.

Every action still goes through `Workbench.dispatch`, and every edit through
`Workbench.edit`, so a test can drive the whole page without a browser.
"""

from __future__ import annotations

import datetime
import glob
import html
import io
import os
from typing import Callable

import panel as pn
from pydantic import ValidationError

from . import geometry as geo
from . import registry
from .compile import CompileJob
from .editor import GeometryEditor
from .gmsh_import import GmshImportError, read_msh_bytes
from .runner import (ARM_LABELS, CaseRun, RunRefused, adapter_for, arm_labels_for,
                     arms_for, results_dir_for, step_label_for)
from .runview import RunPanel, results_view
from .spec import (EXAMPLES, Boundary, CaseSpec, Region, Window, adapt_to_family,
                   check, example_case, slug, summary)

#: Native form controls and scrollbars follow the page, not the OS dark-mode setting.
LIGHT_CSS = ":root { color-scheme: light; }"
MENU_CSS = ".bk-menu { width: max-content; min-width: 100%; }"
#: the header's widgets, on the dark header
#: the Bootstrap design's own rules are more specific than a widget's stylesheet,
#: so the colours here are !important (seen in the page: grey boxes on the header)
HEADER_CSS = """
.bk-btn { font-size: 13px; }
.bk-btn-light { background: transparent !important; color: #e6eef2 !important;
                border-color: #3b5a6b !important; }
.bk-btn-light:hover { background: #22404f !important; color: #ffffff !important; }
.bk-btn-light:disabled { color: #6f8796 !important; border-color: #2c4654 !important; }
"""
TABS_CSS = """
.bk-btn-group .bk-btn { font-size: 14px; padding: 6px 16px; background: transparent !important;
                        color: #c9d6de !important; border-color: transparent !important; }
.bk-btn-group .bk-btn.bk-active { background: #2a4a5c !important; color: #ffffff !important;
                                  font-weight: 600; }
"""
PAGE_CSS = """
body { background: #f4f6f8; }
#main { background: #f4f6f8; }
.title { font-size: 1.25rem !important; }
"""

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CASES_DIR = os.path.join(ROOT, "out", "workbench", "cases")
VERSION = "0.5 (one screen to model, one to run)"
#: How often the page reads a running case's progress, in milliseconds.
POLL_MS = 400

#: The two tabs.  key -> title.  (Before 2026-09-29's rebuild the page had six
#: workflow steps; their keys still open the tab that holds what they held.)
TABS: tuple[tuple[str, str], ...] = (("model", "Model"), ("run", "Run & results"))
LEGACY = {"case": ("model", "physics"), "geometry": ("model", None),
          "physics": ("model", "physics"), "check": ("model", None),
          "results": ("run", None), "run": ("run", None), "model": ("model", None)}

ACCENT = "#0e6874"
HEADER = "#16303f"

#: Each family's "how it is coupled", under the Windows inspector.
WHY = {
    "incompressible-2d": (
        "The projected assembly with the pressure solve exposed is the arrangement the "
        "measurements selected: every other combination leaves its velocity band within "
        "80 macro-steps at six windows (W100). The blend width is the partition of "
        "unity's ramp; windows that meet must overlap by at least twice it."),
    "conduction-2d": (
        "Style B iterates overlapping windows until the update is below the tolerance. "
        "Style C couples exactly two windows that meet along a face, one given the "
        "interface's temperatures (Dirichlet), the other the heat the first sends across "
        "(Neumann); its unrelaxed iteration contracts when the Dirichlet side conducts "
        "less, and the relaxation and Aitken's rule carry it when it does not."),
    "electric-2d": (
        "Style D: the plate is given its electrodes' potentials and returns their "
        "currents; the circuit is given those currents and returns the potentials, "
        "relaxed with the first factor and Aitken's rule."),
    "acoustics-2d": (
        "Style C with an explicit step: the second piece sends the first its pressures "
        "beside the interface, the first updates the interface's velocities and sends "
        "them back, and both step. One exchange closes an explicit step, so the pieces "
        "are the full domain to the bit."),
    "elasticity-2d": (
        "Style B: every window is the nodes of its cells, solved with its outside "
        "neighbours' latest displacements held, blended by the partition of unity until "
        "the update is below the tolerance. Plain Schwarz has no coarse level and "
        "converges slowly on a bending structure; the run reports how slowly."),
    "conjugate-heat-2d": (
        "Style C at a seam between two physics: the channel and the block meet at the "
        "wall; the side that conducts less is given the wall's temperatures, the other "
        "the heat taken through the wall, relaxed until they agree. No coolant crosses "
        "the seam."),
    "thermoelastic-2d": (
        "A split by physics: a conduction agent and an elasticity agent on the same "
        "mesh, the whole temperature field crossing between them, synchronously or a "
        "step behind. The case is one window covering the domain."),
    "transport-2d": (
        "Style A with an explicit step: every step each window is cut from the river, "
        "takes one step with its neighbours' values on its cut faces, and the windows "
        "are blended by the partition of unity, so the arms agree to round-off."),
}


def layer_for(issue) -> str:
    """The Model layer where an issue is fixed, from what it names."""
    m = issue.message.lower()
    for words, layer in ((("device", "rotor", "disk"), "devices"),
                         (("attachment", "circuit", "battery", "electrode has nothing"),
                          "attachments"),
                         (("window", "overlap", "full weight", "cross-point", "seam",
                           "pieces", "piece", "tiling"), "windows"),
                         (("boundary", "edge", "inlet", "outlet", "electrode", "clamp",
                           "coolant", "river enters", "river leaves", "nothing holds",
                           "no edge is loaded"), "boundaries"),
                         (("material", "region"), "regions"),
                         (("outline", "hole", "domain", "grid"), "domain")):
        if any(w in m for w in words):
            return layer
    return "physics"


class Workbench:
    """One browser session's state, views and actions."""

    def __init__(self, cases_dir: str | None = None, spec: CaseSpec | None = None):
        self.cases_dir = cases_dir or DEFAULT_CASES_DIR
        self.spec: CaseSpec = spec if spec is not None else example_case("wake-array-3")
        self.path: str | None = None
        self.dirty = False
        self.history: list[str] = []
        self.future: list[str] = []
        self.view_opts = {"grid": False, "overlaps": True, "labels": True, "log": False}
        #: the Model tab's layer, tool, snap, new-region material and selection;
        #: kept here so they survive the editor being rebuilt
        self.geo_state = {"layer": "domain", "tool": "select", "snap": geo.DEFAULT_SNAP,
                          "material": "material-1", "selected": None, "sub": None}
        self.geo_editor: GeometryEditor | None = None
        self._issues_memo: tuple[str, list] | None = None
        self._keep_view = False
        self.active = "model"
        self.log_lines: list[str] = []
        #: the run marching now, or the last one; its record is the Run tab's results
        self.run: CaseRun | None = None
        #: the arms the Run tab has ticked (a run choice, not part of the case)
        self.run_arms: list[str] = ["serial", "parallel", "full"]
        self.run_panel: RunPanel | None = None
        #: the Run tab's (run, stop, blocked) buttons, so the poll can switch them
        self._run_buttons: tuple | None = None
        self._poll_cb = None
        self._reported: CaseRun | None = None
        #: the compile running now, or the last one
        self.compile_job: CompileJob | None = None
        self._compile_reported: CompileJob | None = None
        #: which case is open (new, an example, a file, an import each count one),
        #: and which one the last compile was of: another case's verdict is not shown
        #: as this one's
        self.case_serial = 0
        self.compiled_serial: int | None = None
        self._build()
        self.log(f"opened the example case {self.spec.name!r}")

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    def log(self, text: str) -> None:
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_lines.insert(0, f"{stamp}  {text}")
        del self.log_lines[200:]

    def issues(self):
        """`check(self.spec)`, computed once per version of the case."""
        key = self.spec.to_json()
        if self._issues_memo is None or self._issues_memo[0] != key:
            self._issues_memo = (key, check(self.spec))
        return self._issues_memo[1]

    def notify(self, kind: str, text: str, duration: int | None = None) -> None:
        """A toast in the browser when there is one; always a log line."""
        self.log(text)
        note = pn.state.notifications
        if note is not None:
            getattr(note, kind)(text, duration=duration or (6000 if kind != "success"
                                                            else 3000))

    def set_spec(self, spec: CaseSpec, label: str, *, record: bool = True,
                 path: str | None | object = ..., dirty: bool = True) -> None:
        """Replace the whole case (new, open, undo): every view is rebuilt."""
        if record:
            self.history.append(self.spec.to_json())
            del self.history[:-100]
            self.future.clear()
            self.case_serial += 1
            # another case: nothing of the last one stays selected.  Undo and redo
            # keep the selection (the editor drops one that no longer exists), so
            # what was being edited is still in the inspector
            self.geo_state["selected"], self.geo_state["sub"] = None, None
        self.spec = spec
        if path is not ...:
            self.path = path
        self.dirty = dirty
        self.geo_state["pending"] = []
        self.log(label)
        self.refresh(rebuild=True)

    def edit(self, apply: Callable[[CaseSpec], None], label: str,
             defaults: bool = False, say: list[str] | None = None) -> bool:
        """Change the case.  Validated as a whole case; refused if invalid.

        When the windows follow the domain (``layout``), an edit that changes what
        they follow -- the domain, the materials, the style, the ramp, the layout's
        own settings -- generates them again here, before the case is validated,
        so no view ever shows windows that no longer fit the geometry.  A family
        whose solver fixes its drawn edges' conditions has them derived here too.

        ``defaults``: the edit draws the domain or chooses the physics, so what the
        case now lacks to run is filled in (`starter.fill_defaults`), each thing
        said in the log and a note, and all of it undone with the edit.  ``say``:
        what else the note says first (filled in by ``apply``, read after it)."""
        new = self.spec.copy_deep()
        apply(new)
        taken = False
        if (self.spec.layout is not None and new.layout is not None
                and [w.model_dump() for w in new.windows]
                != [w.model_dump() for w in self.spec.windows]):
            # the windows were edited by hand: they are the case's own from now on,
            # and stop following the domain (the layout would undo the edit)
            new.layout = None
            taken = True
        if new.layout is not None:
            from . import layout as lay
            try:
                if lay.refresh(new):
                    label += (f"; the windows follow the domain: {len(new.windows)} "
                              f"generated")
            except lay.LayoutError as exc:
                self.notify("error", f"not applied: the windows cannot follow this geometry: "
                                     f"{exc}")
                self.refresh(rebuild=False)
                return False
        from .spec import derive_boundaries
        if derive_boundaries(new):
            label += "; the drawn edges' conditions follow the geometry"
        filled: list[str] = []
        if defaults:
            from .starter import fill_defaults
            filled = fill_defaults(new)
            if filled:
                label += "; to start: " + "; ".join(filled)
        try:
            new = CaseSpec.model_validate(new.model_dump())
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(x) for x in first.get("loc", ()))
            self.notify("error", f"not applied: {where}: {first.get('msg')}")
            #: the views go back to the case as it is: a refused width stayed in
            #: its field, reading as applied
            self.refresh(rebuild=False)
            return False
        self.history.append(self.spec.to_json())
        del self.history[:-100]
        self.future.clear()
        self.spec = new
        self.dirty = True
        self.log(label)
        if taken:
            self.notify("info", "the windows are your own now and no longer follow the "
                                "shape (Model, Windows: choose Automatic to hand them back)")
        if filled or say:
            # what was assumed is said where the person is looking, not only logged,
            # in one note (two at once were too much to read, seen)
            parts = ["; ".join(say)] if say else []
            if filled:
                parts.append(("to start, it was given " if say else
                              "to start, the case was given ") + "; ".join(filled))
            text = ". ".join(t[:1].upper() + t[1:] for t in parts)
            self.notify("info", text + ". Change any of it in its layer; Undo takes it "
                        "all back.", duration=min(20000, 6000 + 40 * len(text)))
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
        #: a fresh case has nothing of the person's to lose: "not saved yet", not
        #: "unsaved changes"
        if action == "file:new":
            self._dialog_new()
        elif group == "file" and rest.startswith("new:"):
            # a new case of one physics, ready to run on the whole grid (a blank case
            # was a wind farm with no windows, an error from its first moment)
            from .starter import new_case
            fid = rest.split(":", 1)[1]
            self.set_spec(new_case(fid), f"new {registry.short_label(fid)} case", path=None,
                          dirty=False)
            self.geo_state["layer"], self.geo_state["tool"] = "domain", "select"
            self.show("model")
            self.notify("info", f"A new {registry.short_label(fid)} case on the whole grid, "
                                f"ready to run. Draw its shape (Shape: Draw the outline), "
                                f"or press Run.")
        elif group == "file" and rest.startswith("example:"):
            key = rest.split(":", 1)[1]
            self.set_spec(example_case(key), f"new case from the example {key!r}", path=None,
                          dirty=False)
        elif action == "file:gallery":
            self._dialog_gallery()
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
        elif action == "file:rename":
            self._dialog_rename()
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
                self.notify("info", "this case has never been saved, so there is nothing to "
                                    "revert to")
        elif action == "view:history":
            self._dialog(self._history())
        elif group == "view" and rest in self.view_opts:
            self.view_opts[rest] = not self.view_opts[rest]
            self.log(f"view: {rest} {'on' if self.view_opts[rest] else 'off'}")
            if self.active == "model":
                self.show("model", keep_view=True)
        elif action == "run:check":
            self._dialog_problems()
        elif action == "run:compile":
            self.start_compile()
        elif action == "run:both":
            self.start_run(self.run_arms)
        elif action == "run:decomposed":
            self.start_run(["serial", "parallel"])
        elif action == "run:full":
            self.start_run(["full"])
        elif action == "run:stop":
            self.stop_run()
        elif action == "run:results":
            self.show("run")
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
            self.notify("warning", "a run is already marching; Stop it first")
            return None
        if self.compile_job is not None and self.compile_job.active:
            self.notify("warning", "a compile is running; a run beside it would time the "
                                   "compile too, so wait for it")
            return None
        errors = [i for i in self.issues() if i.severity == "error"]
        if errors:
            self.notify("error", f"the case has {len(errors)} "
                                 f"{_plural(len(errors), 'problem')} and cannot run: "
                                 f"{errors[0].message}")
            self._dialog_problems()
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

    def start_compile(self, blocking: bool = False) -> CompileJob | None:
        """Compile the case as it stands now with the Atlas compiler, in a background
        thread: the case's graph from its family, the compiler's verdict per seam."""
        if self.compile_job is not None and self.compile_job.active:
            self.notify("info", "a compile is already running")
            return None
        if self.run is not None and self.run.active:
            self.notify("warning", "a run is marching; a compile beside it would be timed "
                                   "with it, so compile after it")
            return None
        errors = [i for i in self.issues() if i.severity == "error"]
        if errors:
            self.notify("error", f"the case has {len(errors)} "
                                 f"{_plural(len(errors), 'problem')} and cannot be compiled: "
                                 f"{errors[0].message}")
            self._dialog_problems()
            return None
        job = CompileJob(self.spec, results_dir=results_dir_for(self.path, self.cases_dir,
                                                                 self.spec.name))
        self.compile_job = job
        self.compiled_serial = self.case_serial
        self._compile_reported = None
        self.log(f"compile started: {self.spec.name} ({len(self.spec.windows)} windows)")
        if blocking:
            job.run_blocking()
            self.poll()
            self.show("run")
        else:
            job.start()
            self.show("run")
            self._start_polling()
        return job

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
        """Read the run's progress into the page; report once when it ends.  The same
        for a compile."""
        self._poll_compile()
        run = self.run
        if run is None:
            if not (self.compile_job is not None and self.compile_job.active):
                self._stop_polling()
            return
        if self.run_panel is not None and self.active == "run" and self.run_panel.run is run:
            self.run_panel.update()
            if self._run_buttons is not None:
                run_btn, stop_btn, blocked = self._run_buttons
                run_btn.disabled = run.active or blocked
                stop_btn.disabled = not run.active
        self._refresh_header()
        if run.active or self._reported is run:
            return
        self._reported = run
        if not (self.compile_job is not None and self.compile_job.active):
            self._stop_polling()
        p = run.progress()
        if run.status == "failed":
            self.notify("error", f"the run failed: {p.error}")
        else:
            res = run.results or {}
            fails = [c["title"] for c in res.get("checks", []) if c["passed"] is False]
            what = "stopped" if run.status == "stopped" else "finished"
            self.notify("warning" if (fails or run.status == "stopped") else "success",
                        f"run {what} after {p.step} {run.step_label}s"
                        + (f"; FAILED: {', '.join(fails)}" if fails else
                           "; every measured check passed")
                        + (f"; saved {os.path.relpath(run.record_path, ROOT)}"
                           if run.record_path else ""))
        if self.active == "run":
            self.show("run", keep_view=True)

    def _poll_compile(self) -> None:
        job = self.compile_job
        if (job is None or job.active or job.status == "created"
                or self._compile_reported is job):
            return
        self._compile_reported = job
        if job.status == "failed":
            self.notify("error", f"the compile failed: {job.error}")
        else:
            s = job.summary
            self.notify("success" if s.verdict == "admit" else "warning",
                        f"compiled in {s.seconds:.1f} s: {s.verdict}"
                        + (f" ({len(s.seams)} {_plural(len(s.seams), 'seam')})"
                           if s.seams else "")
                        + (f"; saved {os.path.relpath(job.record_path, ROOT)}"
                           if job.record_path else ""))
        if not (self.run is not None and self.run.active):
            self._stop_polling()
        if self.active == "run":
            self.show("run", keep_view=True)

    # ------------------------------------------------------------------
    # the page
    # ------------------------------------------------------------------
    def _build(self) -> None:
        self.name_btn = pn.widgets.Button(name=self.spec.name, button_type="light",
                                          stylesheets=[HEADER_CSS], margin=(0, 4),
                                          align="center")
        self.name_btn.on_click(lambda _e: self.dispatch("file:rename"))
        #: what the case simulates, over whether it is saved: two short lines, so the
        #: header fits a laptop screen at 125% (the owner's: about 1,090 px, seen)
        self.family_note = pn.pane.HTML("", margin=(0, 8, 0, 8),
                                        styles={"font-size": "12px", "color": "#e6eef2",
                                                "line-height": "1.2"})
        self.save_state = pn.pane.HTML("", margin=(0, 8, 0, 8),
                                       styles={"font-size": "11px", "color": "#9fb3bf",
                                               "line-height": "1.2"})
        self.tabs = pn.widgets.RadioButtonGroup(options={t: k for k, t in TABS},
                                                value="model", button_type="light",
                                                stylesheets=[TABS_CSS], margin=(0, 12),
                                                align="center")
        self._syncing_tabs = False
        self.tabs.param.watch(lambda e: None if self._syncing_tabs else self.show(e.new),
                              "value")
        fams = [f for f in registry.FAMILIES if f.status == "ready-to-wire"]
        self._fam_by_label = {f.label: f.id for f in fams}
        #: the physics is chosen in the Physics layer's inspector, where its settings
        #: are: in the header it took 250 px the header did not have
        self.family_sel = pn.widgets.Select(name="What it simulates",
                                            options=[f.label for f in fams],
                                            value=self._family_label(),
                                            sizing_mode="stretch_width", margin=(4, 8))
        self._syncing_family = False
        self.family_sel.param.watch(self._on_family, "value")
        #: Undo and Redo sit in the canvas's toolbar, beside the edits they undo
        self.undo_btn = pn.widgets.Button(name="↶", button_type="light",
                                          description="Undo the last change", width=40,
                                          margin=(0, 2))
        self.undo_btn.on_click(lambda _e: self.dispatch("edit:undo"))
        self.redo_btn = pn.widgets.Button(name="↷", button_type="light",
                                          description="Redo", width=40, margin=(0, 2))
        self.redo_btn.on_click(lambda _e: self.dispatch("edit:redo"))
        #: the examples are in the gallery (Examples...), one line each with what it
        #: shows; listed here too they ran the menu off the page (seen)
        self.case_menu = pn.widgets.MenuButton(
            name="Case", button_type="light", width=78, stylesheets=[HEADER_CSS, MENU_CSS],
            margin=(0, 4), align="center",
            items=[("New case...", "file:new"), ("Examples...", "file:gallery"), None,
                   ("Open...", "file:open"), ("Save", "file:save"),
                   ("Save as...", "file:save-as"), ("Rename...", "file:rename"),
                   ("Revert to saved", "edit:revert"), None,
                   ("Import JSON...", "file:import"), ("Export JSON...", "file:export"),
                   ("Import geometry from Gmsh (.msh)...", "file:import-gmsh"), None,
                   ("History of this session", "view:history")])
        self.case_menu.on_click(lambda e: self.dispatch(e.new))
        self.status_btn = pn.widgets.Button(name="", width=140, margin=(0, 6), align="center")
        self.status_btn.on_click(lambda _e: self.dispatch("run:check"))
        self.run_btn = pn.widgets.Button(name="▶ Run", button_type="light", width=84,
                                         margin=(0, 4), align="center",
                                         stylesheets=[":host .bk-btn { background: #ffffff; "
                                                      "color: #0e6874; font-weight: 700; "
                                                      "border-color: #ffffff; }"])
        self.run_btn.on_click(lambda _e: self._run_or_stop())
        self.help_menu = pn.widgets.MenuButton(
            name="?", button_type="light", width=44, stylesheets=[HEADER_CSS, MENU_CSS],
            margin=(0, 4), align="center",
            items=[("How it works", "help:guide"), ("Where things are documented",
                                                    "help:docs"),
                   ("About the workbench", "help:about")])
        self.help_menu.on_click(lambda e: self.dispatch(e.new))
        self.workspace = pn.Column(sizing_mode="stretch_width", margin=0)
        self.modal_body = pn.Column(sizing_mode="stretch_width")

        #: The Bootstrap template, not the Fast one.  Fast's web components follow
        #: the operating system's dark-mode setting on their own while the page
        #: around them stays light, which put dark magenta-bordered inputs on a
        #: light page and white text on the light sidebar (the owner's screenshot,
        #: 2026-09-28).  Bootstrap draws one consistent light theme whatever the OS
        #: says, has no theme switch that reloads the page, and loads nothing from
        #: outside the local server.  No sidebar: the Model tab has its own rail.
        self.tpl = pn.template.BootstrapTemplate(
            title="Atlas Workbench",
            header=[pn.Row(self.name_btn,
                           pn.Column(self.family_note, self.save_state, margin=0,
                                     align="center"),
                           self.tabs, pn.layout.HSpacer(), self.status_btn, self.run_btn,
                           self.case_menu, self.help_menu,
                           align="center", sizing_mode="stretch_width", margin=(0, 8))],
            main=[self.workspace],
            modal=[self.modal_body],
            header_background=HEADER, header_color="#ffffff",
            raw_css=[LIGHT_CSS, PAGE_CSS],
        )
        self.refresh(rebuild=True)

    def _family_label(self) -> str:
        try:
            return registry.family(self.spec.physics.family).label
        except KeyError:                                   # pragma: no cover
            return self.spec.physics.family

    def _on_family(self, e) -> None:
        """A family is more than a label: its parameters, style, run mode and
        boundaries follow it (`spec.adapt_to_family`)."""
        if self._syncing_family:
            return
        fid = self._fam_by_label.get(e.new)
        if fid is None or fid == self.spec.physics.family:
            return
        notes: list[str] = []

        def apply(c):
            notes.append(f"now {registry.short_label(fid)}, from its example")
            notes.extend(adapt_to_family(c, fid))
        if self.edit(apply, f"physics = {fid}", defaults=True, say=notes):
            self.show(self.active, keep_view=True)
        else:
            self._refresh_header()

    def _run_or_stop(self) -> None:
        if self.run is not None and self.run.active:
            self.dispatch("run:stop")
        else:
            self.dispatch("run:both")

    def _refresh_header(self) -> None:
        self.name_btn.name = f"{self.spec.name} ✎"
        if self.path:
            state = "unsaved changes" if self.dirty else "saved"
        else:
            state = "unsaved changes" if self.dirty else "not saved yet"
        self.save_state.object = html.escape(state)
        self.family_note.object = html.escape(registry.short_label(self.spec.physics.family))
        label = self._family_label()
        if self.family_sel.value != label:
            self._syncing_family = True
            try:
                self.family_sel.value = label
            finally:
                self._syncing_family = False
        n = summary(self.issues())
        running = self.run is not None and self.run.active
        if running:
            p = self.run.progress()
            self.status_btn.name = f"Marching {p.step} of {p.steps}"
            self.status_btn.button_type = "primary"
        elif n["error"]:
            self.status_btn.name = f"⚠ {n['error']} {_plural(n['error'], 'problem')}"
            self.status_btn.button_type = "danger"
        elif n["warning"]:
            self.status_btn.name = (f"✓ Ready · {n['warning']} "
                                    f"{_plural(n['warning'], 'warning')}")
            self.status_btn.button_type = "warning"
        else:
            self.status_btn.name = "✓ Ready to run"
            self.status_btn.button_type = "success"
        self.run_btn.name = "■ Stop" if running else "▶ Run"
        self.undo_btn.disabled = not self.history
        self.redo_btn.disabled = not self.future

    def refresh(self, rebuild: bool) -> None:
        """The header and (when asked) the active tab."""
        self._refresh_header()
        if rebuild:
            self.show(self.active, keep_view=True)
        elif self.active == "model" and self.geo_editor is not None:
            d = self.spec.domain
            from .editor import layers_for
            if (self.geo_editor._domain == (d.nx, d.ny)
                    and self.geo_state.get("layer") in layers_for(self.spec)
                    and self.geo_editor._layers == layers_for(self.spec)):
                self.geo_editor.sync()
            else:
                self.show("model", keep_view=self.geo_editor._domain == (d.nx, d.ny))

    def show(self, key: str, keep_view: bool = False) -> None:
        tab, layer = LEGACY.get(key, (key, None))
        self._keep_view = keep_view
        self.active = tab
        if layer is not None:
            self.geo_state["layer"] = layer
            self.geo_state["selected"], self.geo_state["sub"] = None, None
        if self.tabs.value != tab:
            #: moving the tabs from code must not re-enter show() through their watcher
            self._syncing_tabs = True
            try:
                self.tabs.value = tab
            finally:
                self._syncing_tabs = False
        view = {"model": self._view_model, "run": self._view_run}[tab]
        self.workspace.objects = [view()]
        self._refresh_header()

    # ------------------------------------------------------------------
    # the two tabs
    # ------------------------------------------------------------------
    def _view_model(self):
        d = self.spec.domain
        old = self.geo_editor
        ranges = (old.ranges() if (self._keep_view and old is not None
                                   and old._domain == (d.nx, d.ny)) else None)
        self.geo_editor = GeometryEditor(self, ranges)
        return self.geo_editor.view()

    def _view_run(self):
        from .runview import compile_view, run_controls
        s = self.spec
        cpus = os.cpu_count() or 1
        try:
            fam = registry.family(s.physics.family)
        except KeyError:
            fam = None
        runnable = fam is not None and bool(fam.adapter)
        available, unavailable = ("serial", "parallel", "full"), {}
        label = "macro-step"
        names = dict(ARM_LABELS)
        if runnable:
            mod = adapter_for(s.physics.family)
            available, unavailable = arms_for(mod, s)
            label = step_label_for(mod, s)
            names = arm_labels_for(mod)
        steps = self._bind(pn.widgets.IntInput(name=f"{label[0].upper()}{label[1:]}s",
                                               value=s.run.steps, start=1, width=140),
                           lambda c, v: setattr(c.run, "steps", v), "run.steps")
        threads = self._bind(pn.widgets.IntSlider(name="Threads (for the threaded arm)",
                                                  value=min(s.run.threads, cpus),
                                                  start=1, end=cpus, width=240,
                                                  disabled="parallel" not in available),
                             lambda c, v: setattr(c.run, "threads", v), "run.threads")
        opts = {names[a]: a for a in available}
        arms = pn.widgets.CheckBoxGroup(options=opts,
                                        value=[a for a in self.run_arms if a in available],
                                        inline=False)

        def set_arms(e):
            kept = [a for a in self.run_arms if a not in available]
            self.run_arms = [a for a in ("serial", "parallel", "full") if a in e.new
                             or a in kept]
        arms.param.watch(set_arms, "value")
        running = self.run is not None and self.run.active
        n_err = sum(1 for i in self.issues() if i.severity == "error")
        run_btn = pn.widgets.Button(name="▶ Run and compare", button_type="primary",
                                    width=180, disabled=running or bool(n_err) or not runnable)
        run_btn.on_click(lambda e: self.dispatch("run:both"))
        stop_btn = pn.widgets.Button(name="■ Stop", button_type="danger", width=90,
                                     disabled=not running)
        stop_btn.on_click(lambda e: self.dispatch("run:stop"))
        self._run_buttons = (run_btn, stop_btn, bool(n_err) or not runnable)
        why = []
        if not runnable:
            why.append(f"The {fam.label if fam else s.physics.family} family has no runner "
                       f"yet. {fam.note if fam else ''}")
        if n_err:
            why.append(f"The case has {n_err} {_plural(n_err, 'problem')}: the status chip "
                       f"in the header lists them, each with where to fix it.")
        self.run_panel = RunPanel(self, self.run)
        #: the compile card sits under the controls: a finished run or compile
        #: rebuilds this tab at its top, so a verdict at the bottom was out of sight
        body = [run_controls(self, run_btn, stop_btn, steps, threads, arms, label,
                             [f"No {names[a].lower()} arm: {w}." for a, w in unavailable.items()],
                             why),
                compile_view(self)]
        if self.run is not None:
            if self.run.active:
                body.append(self.run_panel.live_view())
            elif self.run.status == "failed":
                body.append(pn.pane.Alert(
                    f"The run failed: {html.escape(self.run.progress().error or '')}",
                    alert_type="danger", sizing_mode="stretch_width"))
                body.append(self.run_panel.live_view())
            else:
                #: the cards are what a person reads first, so they come before the
                #: fields; the progress line would only repeat the controls' "Last run"
                body.append(results_view(self, self.run.results))
                body.append(self.run_panel.live_view(progress=False))
        else:
            body.append(results_view(self, None))
        return pn.Column(*body, sizing_mode="stretch_width", margin=(8, 16))

    def _bind(self, widget, apply: Callable[[CaseSpec, object], None], label: str):
        widget.param.watch(lambda e: self.edit(lambda s: apply(s, e.new),
                                               f"{label} = {e.new}"), "value")
        return widget

    # ------------------------------------------------------------------
    # dialogs
    # ------------------------------------------------------------------
    def _dialog(self, *objs) -> None:
        close = pn.widgets.Button(name="Close", width=90)
        close.on_click(lambda e: self.tpl.close_modal())
        #: its own white panel: the template's dialog box does not grow with a Panel
        #: layout, so the content ran over the dimmed page (seen in the page)
        self.modal_body.objects = [pn.Column(*objs, close, sizing_mode="stretch_width",
                                             styles={"background": "#ffffff",
                                                     "border-radius": "12px",
                                                     "padding": "8px 12px"})]
        self.tpl.open_modal()

    def _dialog_problems(self) -> None:
        """The status chip's list: each problem, a button to where it is fixed, and,
        where the repair is plain (`starter.FIXES`), a button that makes it."""
        from .starter import fix_label
        issues = self.issues()
        n = summary(issues)
        rows = []
        colour = {"error": "#b91c1c", "warning": "#b45309", "info": "#475569"}
        for i in issues:
            go = pn.widgets.Button(name="Show me", button_type="primary", button_style="outline",
                                   width=96, align="center")

            def goto(_e, i=i):
                self.tpl.close_modal()
                self.geo_state["layer"] = layer_for(i)
                self.geo_state["selected"], self.geo_state["sub"] = None, None
                self.geo_state["tool"] = "select"
                self.show("model")
            go.on_click(goto)
            buttons = [go]
            if i.fix:
                fx = pn.widgets.Button(name=fix_label(i.fix), button_type="primary",
                                       align="center", margin=(4, 4))
                fx.on_click(lambda _e, key=i.fix: self.fix([key]))
                buttons.insert(0, fx)
            rows.append(pn.Row(pn.pane.HTML(
                f"<div style='font-size:13px;line-height:1.4'><b style='color:"
                f"{colour[i.severity]}'>{i.severity}</b> &middot; {html.escape(i.message)}"
                f"</div>", sizing_mode="stretch_width"), *buttons,
                sizing_mode="stretch_width"))
        head = (f"### {n['error']} {_plural(n['error'], 'problem')} "
                f"{'stops' if n['error'] == 1 else 'stop'} the run"
                if n["error"] else "### Ready to run")
        sub = (f"{n['warning']} {_plural(n['warning'], 'warning')}, {n['info']} "
               f"{_plural(n['info'], 'note')}.")
        keys = list(dict.fromkeys(i.fix for i in issues if i.fix))
        top: list = [pn.pane.Markdown(head + "\n" + sub)]
        if len(keys) > 1:
            fix_all = pn.widgets.Button(name=f"Fix the {len(keys)} that can be fixed",
                                        button_type="primary", margin=(0, 10, 8, 10))
            fix_all.on_click(lambda _e: self.fix(keys))
            top.append(fix_all)
        self._dialog(*top, pn.Column(*rows, sizing_mode="stretch_width",
                                     styles={"max-height": "60vh", "overflow-y": "auto"}))

    def fix(self, keys: list[str]) -> bool:
        """Make the plain repairs ``keys`` (`starter.apply_fix`), as one edit, and say
        what was done; the problems list comes back while any problem stays."""
        from .starter import apply_fix
        done: list[str] = []

        def apply(c):
            for key in keys:
                r = apply_fix(c, key)
                if r:
                    done.append(r)
        self.tpl.close_modal()
        ok = self.edit(apply, "fixed: " + ", ".join(keys))
        if ok and done:
            self.notify("success", "Done: " + "; ".join(done) + ".")
        elif ok:
            self.notify("info", "nothing needed doing: the case already had it")
        if any(i.severity == "error" for i in self.issues()):
            self._dialog_problems()
        return ok

    def _dialog_rename(self) -> None:
        name = pn.widgets.TextInput(name="Name", value=self.spec.name, width=320)
        desc = pn.widgets.TextAreaInput(name="Description", value=self.spec.description,
                                        width=420, height=90)
        go = pn.widgets.Button(name="Apply", button_type="primary", width=90)

        def do(_e):
            self.tpl.close_modal()

            def apply(c):
                c.name = name.value.strip() or c.name
                c.description = desc.value
            self.edit(apply, f"renamed {name.value!r}")
        go.on_click(do)
        self._dialog(pn.pane.Markdown("### The case"), name, desc, go)

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
                f" **An overlap under {2 * ramp} cells (twice the blend width) leaves cells "
                f"with no window at full weight, and the check will refuse it.**")
            note.object = (f"{len(t)} windows of **{w} x {h}** cells on the "
                           f"{d.nx} x {d.ny} grid, overlapping by at least {ov.value} cells."
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
                c.layout = None               # a tiling of rectangles is the case's own
            self.edit(apply, f"laid a {cols.value} x {rows.value} grid of windows, overlap "
                             f"{ov.value}")
        go.on_click(do)
        self._dialog(pn.pane.Markdown("### Lay a grid of rectangular windows"),
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
                c.layout = None
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

    def _dialog_gallery(self) -> None:
        """Every showcase example, one line each, and a button that opens it."""
        rows = []
        for key, ex in EXAMPLES.items():
            try:
                fam = registry.family(ex.family).label
            except KeyError:                               # pragma: no cover
                fam = ex.family
            go = pn.widgets.Button(name="Open", button_type="primary", width=80,
                                   align="center")

            def open_(_e, key=key):
                self.tpl.close_modal()
                self.dispatch(f"file:example:{key}")
            go.on_click(open_)
            esc = lambda t: html.escape(t, quote=False)                     # noqa: E731
            rows.append(pn.Row(go, pn.pane.Markdown(
                f"**{esc(ex.label)}**<br>{esc(ex.description)}<br>"
                f"<small>{esc(fam)}; style {esc(ex.style)}</small>",
                sizing_mode="stretch_width", margin=(0, 8)), sizing_mode="stretch_width"))
        self._dialog(pn.pane.Markdown(
            "### Examples\nOne case per coupling style and physics, and one drawn case per "
            "family, each run and compared on this laptop in under two minutes. These are "
            "showcases, not research records: `wiki/concepts/Atlas 0.1/atlas-0.1-outcome/"
            "showcase-gallery.md` lists what each one measured."),
            pn.Column(*rows, sizing_mode="stretch_width",
                      styles={"max-height": "60vh", "overflow-y": "auto"}))

    def _dialog_new(self) -> None:
        """What the new case simulates: one button per physics that runs."""
        rows = []
        for f in registry.FAMILIES:
            if f.status != "ready-to-wire":
                continue
            go = pn.widgets.Button(name="Start", button_type="primary", width=80,
                                   align="center")

            def start(_e, fid=f.id):
                self.tpl.close_modal()
                self.dispatch(f"file:new:{fid}")
            go.on_click(start)
            rows.append(pn.Row(go, pn.pane.HTML(
                f"<div style='font-size:14px;font-weight:600'>"
                f"{html.escape(registry.short_label(f.id).capitalize())}</div>"
                f"<div style='font-size:12px;color:#475569'>{html.escape(f.label)}</div>",
                sizing_mode="stretch_width", margin=(0, 8)), sizing_mode="stretch_width"))
        self._dialog(pn.pane.Markdown(
            "### New case: what does it simulate?\nIt starts on the whole grid, with that "
            "physics' usual scale and settings and everything it needs to run; then draw "
            "its shape. *Examples...* has finished cases to start from instead."),
            pn.Column(*rows, sizing_mode="stretch_width",
                      styles={"max-height": "60vh", "overflow-y": "auto"}))

    def _dialog_open(self) -> None:
        files = self.case_files()
        rel = {os.path.basename(f): f for f in files}
        if not rel:
            self._dialog(pn.pane.Markdown(
                f"### Open a case\nNo saved cases in `{os.path.relpath(self.cases_dir, ROOT)}` "
                "yet. Use **Case > Save** first, or **Case > Import JSON**."))
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

    def _history(self):
        rows = "".join(f"<div>{html.escape(line)}</div>" for line in self.log_lines)
        return pn.Column(pn.pane.Markdown("### This session's history"),
                         pn.pane.HTML(f"<div style='font-family:ui-monospace,Consolas,"
                                      f"monospace;font-size:12px'>{rows}</div>",
                                      styles={"max-height": "60vh", "overflow-y": "auto"},
                                      width=640))

    def _guide(self):
        return pn.pane.Markdown(
            "### How it works\n"
            "**Model.** Pick a layer on the left, then click on the canvas:\n"
            "- **Shape**: draw the domain's outline and holes (lines, arcs, smooth "
            "curves); the grid is the domain until you do.\n"
            "- **Materials**: paint regions; click one to choose what it is made of.\n"
            "- **Boundaries**: click an edge to set its condition.\n"
            "- **Windows**: *Automatic* cuts the shape for you and follows it; *My own* "
            "lets you draw them. Choose how they are joined here too.\n"
            "- **Physics**: the family's numbers, the time, the grid.\n\n"
            "Five tools: **Select** (click a shape to reshape it with its handles), "
            "**Draw** (click points, double-click to finish), **Rectangle** and "
            "**Circle** (two clicks each) and **Pan**. A tool adds one shape, then hands "
            "back to Select with the new shape selected.\n\n"
            "The **status chip** says whether the case can run; click it for the list, each "
            "problem with a button to where it is fixed. **Run** marches the decomposition "
            "and the whole domain in turns and compares them; the results and the Atlas "
            "compiler's verdict are on **Run & results**. Every change can be undone.",
            width=600)

    def _about(self):
        return pn.pane.Markdown(
            f"### Atlas Workbench {VERSION}\n"
            "A tool to build a domain decomposition by hand, compile it with the Atlas "
            "compiler, run it, and compare it with the full-domain solve, in eight physics "
            "families on drawn shapes.\n\n"
            f"- Cases folder: `{os.path.relpath(self.cases_dir, ROOT)}`\n"
            "- Built on **Panel** and **Bokeh** (both BSD-3-Clause, already installed; "
            "nothing is downloaded and the page makes no outside requests).\n"
            "- Every number the page shows is measured on this machine and kept in a "
            "record beside the case file.", width=560)

    def _docs(self):
        return pn.pane.Markdown(
            "### Where things are documented\n"
            "- The workbench: `atlas/workbench/README.md`\n"
            "- The plan it builds: `wiki/concepts/Atlas 0.1/atlas-0.1-outcome/"
            "outcome-c4-path-to-declarative-cases.md` and `showcase-library-plan.md`\n"
            "- What each example measured: `showcase-gallery.md`\n"
            "- The case file's schema: `atlas/workbench/spec.py`\n"
            "- The physics families: `atlas/workbench/registry.py`", width=620)


def _plural(n: int, word: str) -> str:
    return word if n == 1 else word + "s"


def create_app(cases_dir: str | None = None):
    """One session.  ``?case=<file>.json`` opens that saved case on load."""
    pn.extension("tabulator", notifications=True)
    if pn.state.notifications is not None:
        # bottom right: at the top right the notes covered Run and the Case menu
        # (seen on a laptop, 2026-09-30)
        pn.state.notifications.position = "bottom-right"
    wb = Workbench(cases_dir)
    wanted = (pn.state.session_args or {}).get("case", [b""])[0]
    wanted = wanted.decode("utf-8") if isinstance(wanted, bytes) else str(wanted)
    if wanted:
        path = case_from_url(wanted, wb.cases_dir)
        if path:
            wb.open_path(path)
            wb.history.clear()          # undo must not step back into the example
            wb._refresh_header()
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


__all__ = ["Workbench", "create_app", "case_from_url", "serve", "TABS", "layer_for"]
