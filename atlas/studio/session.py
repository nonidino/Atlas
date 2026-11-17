"""One page's state and every action it can take, with no GUI library in it.

The studio's server (`server.py`) holds one `Session` per open page and turns each
WebSocket message into a call here (`protocol.py`); a test drives the same calls
with no browser and no socket.  The rules are the retired Panel workbench's
(`atlas/workbench/app.py`, `Workbench`), kept where they were decided:

* **a case is what it simulates** (case@0.6): choosing another kind starts a new
  case of it, ready to run (`starter.new_case_from`), and one Undo brings the old
  one back with its file and key;
* **every edit is validated as a whole case** before it is kept: a refused edit
  changes nothing, and the page goes back to the case as it is;
* windows that follow the domain are cut again whenever what they follow changes
  (`layout.refresh`); a family whose solver fixes its drawn edges' conditions has
  them derived (`spec.derive_boundaries`); a newly drawn shape or newly chosen
  physics gets what it lacks to run (`starter.fill_defaults`), each thing said;
* **a run is a commit**: the case is copied when Run is pressed, and the page is
  told which version is marching; one run or compile at a time.

**The geometry is a feature tree** (case file 0.7, `document.py`) beside the engine's
case.  An edit that changes the tree evaluates it again (`geometry_eval.apply`): the
engine's domain, holes and regions are regenerated, a boundary set on a drawn edge
follows the side it was set on (`remap_boundaries`), and the new shape gets what it
lacks to run, as the old workbench's drawing did.  The grid keeps its size unless the
shape outgrows it, except for the first shape of an empty drawing, which the grid
fits.  What the tree cannot be turned into (no shape, a domain in pieces, an outline
that crosses itself) is a problem in the status chip, beside the engine's own.

What the page is told after every message is `state()`: the case, its tree, its
problems in plain words with their one-click fixes, the status chip, and the engine's
geometry to draw.  Notices are queued in ``pushes`` and sent with the reply.
"""

from __future__ import annotations

import base64
import datetime
import glob
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import ValidationError

from atlas.workbench import learned_case, registry
from atlas.workbench.spec import (EXAMPLES, FAST_EXAMPLES, CaseSpec, check, example_case,
                                  slug)

from . import document as doc
from . import geometry_eval as GE
from .drawing import drawing

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#: the same folder the retired workbench saved to, so a case saved in either opens
#: in the other
DEFAULT_CASES_DIR = os.path.join(ROOT, "out", "workbench", "cases")
#: the kind a new page starts with: heat conduction is the chart operator's own
#: physics (CO-D), and runs on the bare grid
START_FAMILY = "conduction-2d"
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,80}$")


def new_key(fid: str) -> str:
    """The studio's key for a new case of the kind ``fid``: its kind and the moment it
    was started (a case has no name since case@0.6; a file and a record need one)."""
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return slug(f"{registry.short_label(fid)}-{stamp}")


def layer_for(message: str) -> str:
    """Where in the page a problem is fixed, from what it names (the retired
    workbench's `app.layer_for`, with the studio's tabs)."""
    m = message.lower()
    for words, where in ((("device", "rotor", "disk"), "physics"),
                         (("attachment", "circuit", "battery", "electrode has nothing"),
                          "physics"),
                         (("window", "overlap", "full weight", "cross-point", "seam",
                           "pieces", "piece", "tiling"), "pieces"),
                         (("boundary", "edge", "inlet", "outlet", "electrode", "clamp",
                           "coolant", "river enters", "river leaves", "nothing holds",
                           "no edge is loaded"), "physics"),
                         (("material", "region"), "geometry"),
                         (("outline", "hole", "domain", "grid", "shape"), "geometry")):
        if any(w in m for w in words):
            return where
    return "physics"


@dataclass(frozen=True)
class Version:
    """One step of Undo: the case, its tree, and which file it belongs to.  A switch
    of kind replaces the whole case, so Undo brings back its file and key too."""
    json: str
    geom: str
    path: str | None
    key: str
    dirty: bool
    notes: tuple = ()


class Refused(ValueError):
    """An action the session will not take as asked; the message says why."""


def _plural(n: int, word: str) -> str:
    return word if n == 1 else word + "s"


class Session:
    """One page's case, its history and its jobs."""

    def __init__(self, cases_dir: str | None = None, spec: CaseSpec | None = None,
                 key: str | None = None):
        from atlas.workbench.starter import new_case_from
        self.cases_dir = os.path.abspath(cases_dir or DEFAULT_CASES_DIR)
        if spec is None:
            spec, _example, _filled = new_case_from(START_FAMILY)
            key = key or new_key(START_FAMILY)
        self.spec: CaseSpec = spec
        self.geom: doc.Geometry = doc.migrate(spec)
        #: what the last evaluation of the tree could not build, or noted
        self.geo_notes: list[dict] = []
        self._prov: dict | None = None          # the engine edges' sides, when known
        self.path: str | None = None
        self.key = key or new_key(spec.physics.family)
        self.dirty = False
        self.history: list[Version] = []
        self.future: list[Version] = []
        #: which case is open (new, an example, a file, an import each count one); a
        #: compile or run of another case is not shown as this one's
        self.serial = 0
        #: bumped by every change the page must redraw
        self.version = 0
        self.log_lines: list[str] = []
        self.pushes: list[dict] = []
        self._issues_memo: tuple | None = None
        self.run = None
        self.run_serial: int | None = None
        self.run_arms: list[str] = ["serial", "parallel", "full"]
        self.compile_job = None
        self.compiled_serial: int | None = None

    # ------------------------------------------------------------------
    # what the page is told
    # ------------------------------------------------------------------
    def log(self, text: str) -> None:
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_lines.insert(0, f"{stamp}  {text}")
        del self.log_lines[200:]

    def notify(self, kind: str, text: str, ms: int | None = None) -> None:
        """A notice in the page (kind: info, success, warning, error); always logged."""
        self.log(text)
        if ms is None:
            ms = min(20000, 5000 + 40 * len(text)) if kind != "success" else 3500
        self.pushes.append({"type": "notice", "kind": kind, "text": text, "ms": ms})

    def drain(self) -> list[dict]:
        out, self.pushes = self.pushes, []
        return out

    def issues(self):
        """`spec.check(self.spec)`, computed once per version of the case (every change
        bumps ``version``; the spec is read once, so a change landing meanwhile is
        checked again on the next call)."""
        key, spec = self.version, self.spec
        if self._issues_memo is None or self._issues_memo[0] != (key, id(spec)):
            self._issues_memo = ((key, id(spec)), check(spec))
        return self._issues_memo[1]

    def has_shape(self) -> bool:
        consumed = self.geom.consumed()
        return any(f.role == "add" and not f.hidden and f.id not in consumed
                   for f in self.geom.features)

    def geometry_errors(self) -> list[dict]:
        out = [n for n in self.geo_notes if n.get("severity") == "error"]
        if not self.has_shape() and not any(n.get("kind") == "empty" for n in out):
            out.insert(0, {"severity": "error", "kind": "empty",
                           "message": "No shape yet: pick a start, draw, or import"})
        return out

    def issue_rows(self) -> list[dict]:
        from atlas.workbench.starter import FIXES, fix_label
        rows = []
        for n in self.geometry_errors() + [n for n in self.geo_notes
                                           if n.get("severity") != "error"]:
            fix = n.get("fix")
            rows.append({"severity": n.get("severity", "warning"), "step": "geometry",
                         "message": n["message"][:1].upper() + n["message"][1:],
                         "fix": f"geo:{fix}" if fix else None,
                         "fix_label": GEO_FIXES.get(fix), "where": "geometry",
                         "feature": n.get("feature")})
        rows += [{"severity": i.severity, "step": i.step, "message": i.message,
                  "fix": i.fix if i.fix in FIXES else None,
                  "fix_label": fix_label(i.fix) if i.fix in FIXES else None,
                  "where": layer_for(i.message)} for i in self.issues()]
        return rows

    def status(self) -> dict:
        """The header's chip: whether the case can run, in one or two words."""
        run = self.run
        if run is not None and run.active:
            p = run.progress()
            return {"kind": "busy", "text": f"Running {p.step}/{p.steps}",
                    "tip": f"Marching {run.step_label}s; Stop is in Run"}
        if self.compile_job is not None and self.compile_job.active:
            return {"kind": "busy", "text": "Compiling",
                    "tip": "The Atlas compiler is checking every seam"}
        geo = self.geometry_errors()
        if geo and geo[0].get("kind") == "empty":
            return {"kind": "warn", "text": "No shape yet", "tip": "Pick a start, draw, or import"}
        errors = geo + [i for i in self.issues() if i.severity == "error"]
        warns = [n for n in self.geo_notes if n.get("severity") == "warning"] + \
            [i for i in self.issues() if i.severity == "warning"]
        if errors:
            n = len(errors)
            first = errors[0]["message"] if isinstance(errors[0], dict) else errors[0].message
            return {"kind": "err", "text": f"{n} {_plural(n, 'problem')}",
                    "tip": f"Cannot run: {first}"}
        if warns:
            n = len(warns)
            first = warns[0]["message"] if isinstance(warns[0], dict) else warns[0].message
            return {"kind": "warn", "text": f"{n} {_plural(n, 'note')}",
                    "tip": f"Can run. {first[:1].upper() + first[1:]}"}
        return {"kind": "ok", "text": "Ready", "tip": "Everything needed to run is set"}

    def meta(self) -> dict:
        fam = registry.family(self.spec.physics.family)
        return {"key": self.key,
                "file": os.path.basename(self.path) if self.path else None,
                "dirty": self.dirty, "read_only": self.read_only,
                "can_undo": bool(self.history), "can_redo": bool(self.future),
                "serial": self.serial, "version": self.version,
                "family": fam.id, "family_label": registry.short_label(fam.id),
                "unit": fam.length_unit or "cells",
                "fast_example": FAST_EXAMPLES.get(fam.id),
                "schema": doc.SCHEMA_06 if doc.is_migration(self.spec, self.geom)
                else doc.SCHEMA_07,
                "default_material": GE.default_material(self.spec)}

    def state(self) -> dict:
        return {"type": "state", "case": json.loads(self.spec.to_json()),
                "geometry": json.loads(self.geom.model_dump_json()),
                "meta": self.meta(), "issues": self.issue_rows(),
                "status": self.status(), "drawing": drawing(self.spec)}

    @staticmethod
    def catalog() -> dict:
        """What never changes while the server runs: the kinds, the examples."""
        from . import starts
        fams = []
        for f in registry.FAMILIES:
            fams.append({
                "id": f.id, "label": f.label, "short": registry.short_label(f.id),
                "runs": f.id in registry.available_ids(), "note": f.note,
                "styles": list(f.styles), "modes": list(f.modes),
                "unit": f.length_unit or "cells", "layers": list(f.layers),
                "boundary_kinds": [{"id": k, "meaning": registry.boundary_kind(k).meaning,
                                    "needs_value": registry.boundary_kind(k).needs_value,
                                    "unit": registry.boundary_kind(k).unit}
                                   for k in f.boundary_kinds],
                "params": [{"name": p.name, "label": p.label, "unit": p.unit,
                            "default": p.default, "min": p.minimum, "max": p.maximum,
                            "help": p.help} for p in f.params],
                "material_props": [{"name": p.name, "label": p.label, "unit": p.unit,
                                    "default": p.default, "help": p.help}
                                   for p in f.material_props],
                "materials": f.material_library(),
                "fast_example": FAST_EXAMPLES.get(f.id)})
        exs = [{"key": e.key, "family": e.family, "shows": e.shows, "style": e.style,
                "fast": FAST_EXAMPLES.get(e.family) == e.key,
                "fixed": e.key == learned_case.KEY} for e in EXAMPLES.values()]
        return {"type": "catalog", "families": fams, "examples": exs,
                "styles": dict(registry.STYLES), "starts": starts.CATALOG,
                "benchmarks": starts.BENCHMARKS}

    # ------------------------------------------------------------------
    # versions
    # ------------------------------------------------------------------
    @property
    def read_only(self) -> bool:
        """The learned case is fixed: its experts were trained for this farm."""
        return self.key == learned_case.KEY

    def _version(self) -> Version:
        return Version(self.spec.to_json(), self.geom.model_dump_json(), self.path, self.key,
                       self.dirty, tuple(json.dumps(n) for n in self.geo_notes))

    def _changed(self) -> None:
        self.version += 1

    def set_spec(self, spec: CaseSpec, label: str, *, record: bool = True,
                 path: str | None | object = ..., dirty: bool = True,
                 key: str | None = None, geom: doc.Geometry | None = None,
                 notes: list[dict] | None = None) -> None:
        """Replace the whole case (a new kind, start over, an example, open, undo)."""
        if record:
            self.history.append(self._version())
            del self.history[:-100]
            self.future.clear()
            self.serial += 1
        self.spec = spec
        self.geom = geom if geom is not None else doc.migrate(spec)
        self.geo_notes = list(notes or [])
        self._prov = None
        if path is not ...:
            self.path = path
        if key is not None:
            self.key = key
        self.dirty = dirty
        self.log(label)
        self._changed()

    def _provenance(self) -> dict:
        """The current engine edges' sides (computed once per tree)."""
        if self._prov is None:
            r = GE.evaluate(self.geom, self.spec, keep=(self.spec.domain.nx, self.spec.domain.ny))
            self._prov = r.provenance
        return self._prov

    def _commit(self, new: CaseSpec, geom: doc.Geometry, label: str, *, defaults: bool,
                say: list[str], dx: float | None = None) -> bool:
        """The edit pipeline: the tree evaluated when it changed (or the grid did), the
        windows cut again, the drawn edges' conditions derived, what a new shape lacks
        filled in, and the whole case validated."""
        notes = self.geo_notes
        geo_changed = (geom.model_dump() != self.geom.model_dump()) or dx is not None
        prov = self._prov
        if geo_changed:
            if any(f.role == "add" for f in geom.features):
                had = self.has_shape()
                if dx is not None:
                    s = self.spec.domain.dx / dx
                    keep = (int(math.ceil(self.spec.domain.nx * s - 1e-7)),
                            int(math.ceil(self.spec.domain.ny * s - 1e-7)))
                else:
                    keep = (new.domain.nx, new.domain.ny) if had else None
                old_prov = self._provenance() if had else {}
                applied, res = GE.apply(new, geom, dx=dx, keep=keep)
                notes = list(res.notes)
                if res.domain is not None:
                    was_drawn = self.spec.domain.outline is not None
                    new = applied
                    new.boundaries, lost = GE.remap_boundaries(new.boundaries, old_prov,
                                                               res.provenance)
                    say += lost
                    settle_boundaries(new, was_drawn)
                    if not was_drawn and new.domain.outline is not None:
                        follow_the_domain(new)
                    if dx is not None:
                        _scale_cells(new, self.spec.domain.dx / dx)
                    prov = res.provenance
                defaults = True                    # a new shape gets what it lacks to run
            else:
                notes, prov = [], None
        taken = False
        if (self.spec.layout is not None and new.layout is not None
                and [w.model_dump() for w in new.windows]
                != [w.model_dump() for w in self.spec.windows]):
            new.layout = None                  # windows edited by hand are the case's own
            taken = True
        if new.layout is not None:
            from atlas.workbench import layout as lay
            try:
                if lay.refresh(new):
                    label += (f"; the windows follow the domain: {len(new.windows)} "
                              f"generated")
            except lay.LayoutError as exc:
                if not defaults:
                    self.notify("error", f"not applied: the windows cannot follow this "
                                         f"geometry: {exc}")
                    self._changed()
                    return False
                new.windows, new.layout = [], None
                say.append(f"the windows could not be cut as before ({exc}), so they were "
                           f"cut again from the new shape")
        from atlas.workbench.spec import derive_boundaries
        if derive_boundaries(new):
            label += "; the drawn edges' conditions follow the geometry"
        filled: list[str] = []
        if defaults and not any(n.get("severity") == "error" for n in notes):
            from atlas.workbench.starter import fill_defaults
            filled = fill_defaults(new)
            if filled:
                label += "; to start: " + "; ".join(filled)
                if new.layout is not None:
                    from atlas.workbench import layout as lay
                    try:
                        lay.refresh(new)
                    except lay.LayoutError:
                        pass                                  # the check says why
        try:
            new = CaseSpec.model_validate(new.model_dump())
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(x) for x in first.get("loc", ()))
            self.notify("error", f"not applied: {where}: {first.get('msg')}")
            self._changed()
            return False
        self.history.append(self._version())
        del self.history[:-100]
        self.future.clear()
        self.spec = new
        self.geom = geom
        self.geo_notes = notes
        self._prov = prov if geo_changed else self._prov
        self.dirty = True
        self.log(label)
        if taken:
            self.notify("info", "The windows are your own now and no longer follow the "
                                "shape (Pieces: choose Automatic to hand them back).")
        if filled or say:
            parts = ["; ".join(say)] if say else []
            if filled:
                parts.append(("to start, it was given " if say else
                              "to start, the case was given ") + "; ".join(filled))
            text = ". ".join(t[:1].upper() + t[1:] for t in parts)
            self.notify("info", text + ". Undo takes it all back.")
        self._changed()
        return True

    def edit(self, apply: Callable[[CaseSpec], Any], label: str, *, defaults: bool = False,
             say: list[str] | None = None) -> bool:
        """Change the case (not its tree).  Validated as a whole case; refused if
        invalid (the retired workbench's `Workbench.edit`, rule for rule)."""
        if self.read_only:
            self.notify("info", learned_case.READ_ONLY + " Choose another example, or a "
                                "kind in Case, New, to edit a case.")
            return False
        new = self.spec.copy_deep()
        try:
            apply(new)
        except Refused as exc:
            self.notify("error", f"not applied: {exc}")
            self._changed()
            return False
        return self._commit(new, self.geom, label, defaults=defaults, say=list(say or []))

    def edit_ops(self, ops: list, label: str, defaults: bool = False) -> bool:
        """An edit as data: ``ops`` applied in order to the document as JSON (the case's
        fields, and ``geometry`` for the tree; `apply_ops`)."""
        if self.read_only:
            self.notify("info", learned_case.READ_ONLY + " Choose another example, or a "
                                "kind in Case, New, to edit a case.")
            return False
        data = json.loads(self.spec.to_json())
        data["geometry"] = json.loads(self.geom.model_dump_json())
        try:
            apply_ops(data, ops)
            geom = doc.Geometry.model_validate(data.pop("geometry"))
            new = CaseSpec.model_validate(data)
        except Refused as exc:
            self.notify("error", f"not applied: {exc}")
            self._changed()
            return False
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(x) for x in first.get("loc", ()))
            msg = first.get("msg", "")
            self.notify("error", f"not applied: {where}: {msg}" if where else f"not applied: {msg}")
            self._changed()
            return False
        return self._commit(new, geom, label, defaults=defaults, say=[])

    def _restore(self, v: Version, label: str) -> None:
        self.set_spec(CaseSpec.from_json(v.json), label, record=False, path=v.path,
                      dirty=v.dirty, key=v.key, geom=doc.Geometry.model_validate_json(v.geom),
                      notes=[json.loads(n) for n in v.notes])

    def undo(self) -> None:
        if not self.history:
            self.notify("info", "Nothing to undo.")
            return
        self.future.append(self._version())
        self._restore(self.history.pop(), "undo")

    def redo(self) -> None:
        if not self.future:
            self.notify("info", "Nothing to redo.")
            return
        self.history.append(self._version())
        self._restore(self.future.pop(), "redo")

    # ------------------------------------------------------------------
    # the geometry's own actions
    # ------------------------------------------------------------------
    def geometry_check(self, n: int | None = None, page_areas: dict | None = None) -> dict:
        from . import geocheck
        if not self.has_shape():
            return {"type": "check", "n": n, "notes": self.geometry_errors(), "checks": {},
                    "areas": []}
        return geocheck.check_view(self.spec, self.geom, n, page_areas)

    def geometry_fix(self, key: str) -> bool:
        """A one-click fix the check offers: ``grid:<n>``, ``widen:<feature>``, or
        ``origin`` (move the drawing so the domain starts at the origin)."""
        from . import geocheck
        if key.startswith("grid:"):
            try:
                n = int(key.split(":", 1)[1])
            except ValueError:
                raise Refused(f"no grid {key!r}") from None
            if not 8 <= n <= 4096:
                raise Refused("a grid of 8 to 4096 cells along the longer side")
            dx = geocheck.dx_for(self.spec, n)
            return self._commit(self.spec.copy_deep(), self.geom, f"grid n = {n}",
                                defaults=True, say=[f"the grid is now {n} cells along its "
                                                    f"longer side"], dx=dx)
        if key.startswith("widen:"):
            fid = key.split(":", 1)[1]
            f = self.geom.by_id().get(fid)
            if f is None:
                raise Refused(f"no feature {fid!r}")
            h = self.spec.domain.dx
            g = self.geom.model_copy(deep=True)
            t = g.by_id()[fid]
            if t.kind == "rect":
                if t.w < t.h:
                    t.x -= (h - t.w) / 2
                    t.w = h
                else:
                    t.y -= (h - t.h) / 2
                    t.h = h
            elif t.kind == "ellipse":
                if t.rx < t.ry:
                    t.rx = h / 2
                else:
                    t.ry = h / 2
            else:
                raise Refused(f"{f.label} cannot be widened in one click: drag its edge")
            return self._commit(self.spec.copy_deep(), g, f"widened {f.label} to one cell",
                                defaults=False, say=[])
        if key == "origin":
            res = GE.evaluate(self.geom, self.spec)
            x0, y0, _x1, _y1 = res.shape.bounds if not res.shape.is_empty else (0, 0, 0, 0)
            g = self.geom.model_copy(deep=True)
            for f in g.features:
                translate(f, -x0, -y0)
            return self._commit(self.spec.copy_deep(), g, "moved the drawing to the origin",
                                defaults=False, say=[])
        raise Refused(f"no fix {key!r}")

    def geometry_start(self, shape: str, params: dict) -> bool:
        """The start screen: a parametrised starting shape replaces the drawing, and the
        grid fits it."""
        from . import starts
        try:
            feats = starts.build(shape, params, material=GE.default_material(self.spec),
                                 family=self.spec.physics.family)
        except ValueError as exc:
            raise Refused(str(exc)) from None
        dx = self.spec.domain.dx
        unit = registry.family(self.spec.physics.family).length_unit or "cells"
        for f in feats:
            if f.role == "add":
                x0, y0, x1, y1 = GE.Evaluator(doc.Geometry(features=[f]), dx / 64).shapes[f.id].bounds
                if min(x1 - x0, y1 - y0) < 4 * dx:
                    raise Refused(f"at this kind's cell size ({dx:g} {unit}) a shape needs at "
                                  f"least 4 cells across; make it at least {4 * dx:g} {unit}")
        g = doc.Geometry(parameters=list(self.geom.parameters), features=feats)
        # an empty drawing first, so the grid fits the new shape
        self.geom = doc.Geometry(parameters=list(self.geom.parameters))
        self._prov = None
        return self._commit(self.spec.copy_deep(), g, f"started from {shape}", defaults=True,
                            say=[])

    def geometry_import(self, fmt: str, data: bytes, name: str) -> bool:
        """A CAD file's shapes as features: a Gmsh mesh (with its regions and
        boundaries) or a STEP file (each face, a 2-D section of a solid)."""
        from . import imports
        try:
            feats, extra = imports.load(fmt, data, name, self.spec, self.geom)
        except (ValueError, RuntimeError, OSError) as exc:
            raise Refused(f"could not import {name}: {exc}") from None
        replace = extra.pop("replace", False)
        said = [f"{len(feats)} {_plural(len(feats), 'shape')} from {name}"]
        said += [str(n) for n in extra.pop("notes", [])]
        if replace:                       # a mesh is the whole geometry: the grid fits it
            g = doc.Geometry(parameters=list(self.geom.parameters), features=feats)
            self.geom = doc.Geometry(parameters=list(self.geom.parameters))
            self._prov = None
        else:
            g = self.geom.model_copy(deep=True)
            g.features += feats
        new = self.spec.copy_deep()
        for k, v in extra.items():
            setattr(new, k, v)
        return self._commit(new, g, f"imported {name}", defaults=True, say=said)

    # ------------------------------------------------------------------
    # files
    # ------------------------------------------------------------------
    def files(self) -> list[dict]:
        out = []
        for p in sorted(glob.glob(os.path.join(self.cases_dir, "*.json")),
                        key=os.path.getmtime, reverse=True):
            out.append({"name": os.path.basename(p),
                        "modified": datetime.datetime.fromtimestamp(
                            os.path.getmtime(p)).isoformat(timespec="minutes")})
        return out

    def _file(self, name: str) -> str:
        """A file name inside the cases folder; never a path."""
        name = (name or "").strip()
        if not name.lower().endswith(".json"):
            name += ".json"
        if not NAME.match(name) or os.path.basename(name) != name or ".." in name:
            raise Refused(f"{name!r} is not a plain file name (letters, digits, spaces, "
                          f"dots, dashes and underscores)")
        return os.path.join(self.cases_dir, name)

    def save(self, name: str | None = None) -> str:
        """Save to ``name`` (in the cases folder), or to the case's own file.  A tree
        that adds nothing to the engine's own geometry is saved as 0.6."""
        path = self._file(name) if name else self.path
        if path is None:
            raise Refused("this case has no file yet: give it a name")
        doc.save(path, self.spec, self.geom)
        self.path = path
        self.key = os.path.splitext(os.path.basename(path))[0]
        self.dirty = False
        self.notify("success", f"Saved {os.path.basename(path)}.")
        self._changed()
        return path

    def open(self, name: str) -> None:
        path = self._file(name)
        if not os.path.isfile(path):
            raise Refused(f"there is no {os.path.basename(path)} in the cases folder")
        try:
            with open(path, encoding="utf-8") as fh:
                spec, geom, _schema = doc.loads(fh.read())
        except (OSError, ValidationError, ValueError) as exc:
            raise Refused(f"could not open {os.path.basename(path)}: {exc}") from None
        self.set_spec(spec, f"opened {os.path.basename(path)}", path=path, dirty=False,
                      key=os.path.splitext(os.path.basename(path))[0], geom=geom)

    def revert(self) -> None:
        if not (self.path and os.path.isfile(self.path)):
            self.notify("info", "This case has never been saved, so there is nothing to "
                                "revert to.")
            return
        self.open(os.path.basename(self.path))

    def import_text(self, text: str, name: str = "imported file") -> None:
        try:
            spec, geom, _schema = doc.loads(text)
        except (ValidationError, ValueError) as exc:
            raise Refused(f"{name} is not an Atlas case: {exc}") from None
        stem = slug(os.path.splitext(os.path.basename(name))[0])
        self.set_spec(spec, f"imported {name}", path=None, geom=geom,
                      key=stem if stem != "untitled" else new_key(spec.physics.family))

    def export(self) -> dict:
        return {"name": f"{self.key}.json", "text": doc.dumps(self.spec, self.geom)}

    def start_new(self, fid: str, over: bool = False) -> None:
        """A new case of the kind ``fid``, ready to run, in place of the open one; one
        Undo brings that back (the owner's D2 and D3, 2026-09-30)."""
        from atlas.workbench.starter import new_case_from
        if fid not in registry.available_ids():
            raise Refused(f"no kind of simulation called {fid!r} runs here")
        old = registry.short_label(self.spec.physics.family)
        new = registry.short_label(fid)
        case, example, filled = new_case_from(fid)
        label = f"{'started over: ' if over else ''}a new {new} case"
        if filled:
            label += "; to start: " + "; ".join(filled)
        self.set_spec(case, label, path=None, dirty=False, key=new_key(fid))
        first = f"Started over: a new {new} case" if over else f"Started a new {new} case"
        where = ("It is ready to run on the whole grid: draw its shape, or press Run."
                 if example is None else
                 f"It starts from an example ({EXAMPLES[example].shows.lower()}), since a "
                 f"{new} case needs more than the bare grid.")
        given = f" To start, it was given {'; '.join(filled)}." if filled else ""
        self.notify("info", f"{first}. Undo brings back the {old} case"
                            f"{' as it was' if over else ''}. {where}{given}")

    def example(self, key: str) -> None:
        if key not in EXAMPLES:
            raise Refused(f"no example {key!r}")
        self.set_spec(example_case(key), f"opened the example: {EXAMPLES[key].shows}",
                      path=None, dirty=False, key=key)
        arms = dict(EXAMPLES[key].params).get("arms")
        if arms:
            self.run_arms = [a for a in ("serial", "parallel", "full", "learned", "coarse")
                             if a in str(arms).split()]
        else:
            self.run_arms = ["serial", "parallel", "full"]
        self.notify("info", f"{EXAMPLES[key].shows}. Undo brings back the case before it.")

    def fast(self) -> None:
        fid = self.spec.physics.family
        key = FAST_EXAMPLES.get(fid)
        if key is None:
            raise Refused(f"there is no fast example of a {registry.short_label(fid)} "
                          f"case yet")
        self.example(key)

    def fix(self, keys: list[str]) -> bool:
        """The plain repairs ``keys`` (`starter.apply_fix`), as one edit; a geometry
        fix (``geo:<key>``) through `geometry_fix`."""
        from atlas.workbench.starter import FIXES, apply_fix
        geo = [k[4:] for k in keys if k.startswith("geo:")]
        keys = [k for k in keys if not k.startswith("geo:")]
        ok = True
        for g in geo:
            ok = self.geometry_fix(g) and ok
        if not keys:
            return ok
        unknown = [k for k in keys if k not in FIXES]
        if unknown:
            raise Refused(f"no fix called {unknown[0]!r}")
        done: list[str] = []

        def apply(c):
            for key in keys:
                r = apply_fix(c, key)
                if r:
                    done.append(r)
        ok = self.edit(apply, "fixed: " + ", ".join(keys)) and ok
        if ok and done:
            self.notify("success", "Done: " + "; ".join(done) + ".")
        elif ok:
            self.notify("info", "Nothing needed doing: the case already had it.")
        return ok

    # ------------------------------------------------------------------
    # runs and compiles
    # ------------------------------------------------------------------
    def _errors(self) -> list:
        return [i for i in self.issues() if i.severity == "error"]

    def start_run(self, arms: list[str] | None = None, blocking: bool = False):
        """March the case as it stands now (the commit), in turns, in a background
        thread; ``blocking`` runs it here (tests)."""
        from atlas.workbench.runner import CaseRun, RunRefused, results_dir_for
        if self.run is not None and self.run.active:
            raise Refused("a run is already marching; Stop it first")
        if self.compile_job is not None and self.compile_job.active:
            raise Refused("a compile is running; a run beside it would time the compile "
                          "too, so wait for it")
        geo = self.geometry_errors()
        if geo:
            raise Refused(f"the geometry cannot run: {geo[0]['message']}")
        errors = self._errors()
        if errors:
            raise Refused(f"the case has {len(errors)} {_plural(len(errors), 'problem')} "
                          f"and cannot run: {errors[0].message}")
        try:
            run = CaseRun(self.spec, arms=tuple(arms or self.run_arms),
                          steps=self.spec.run.steps, threads=self.spec.run.threads,
                          results_dir=results_dir_for(self.path, self.cases_dir, self.key),
                          case_path=self.path, case_label=self.key)
        except (RunRefused, KeyError) as exc:
            raise Refused(f"cannot run: {exc}") from None
        self.run = run
        self.run_serial = self.serial
        self.log(f"run started: {run.label()}")
        if blocking:
            run.run_blocking()
        else:
            run.start()
        self._changed()
        return run

    def stop_run(self) -> None:
        if self.run is None or not self.run.active:
            self.notify("info", "Nothing is running.")
            return
        self.run.stop()
        self.log("stop requested; the run stops after the arm-step in progress")

    def start_compile(self, blocking: bool = False):
        from atlas.workbench.compile import CompileJob
        from atlas.workbench.runner import results_dir_for
        if self.compile_job is not None and self.compile_job.active:
            raise Refused("a compile is already running")
        if self.run is not None and self.run.active:
            raise Refused("a run is marching; a compile beside it would be timed with it, "
                          "so compile after it")
        errors = self._errors()
        if errors:
            raise Refused(f"the case has {len(errors)} {_plural(len(errors), 'problem')} "
                          f"and cannot be compiled: {errors[0].message}")
        job = CompileJob(self.spec, results_dir=results_dir_for(self.path, self.cases_dir,
                                                                 self.key),
                         case_label=self.key)
        self.compile_job = job
        self.compiled_serial = self.serial
        self.log(f"compile started ({len(self.spec.windows)} windows)")
        if blocking:
            job.run_blocking()
        else:
            job.start()
        self._changed()
        return job

    def close(self) -> None:
        """The page went away: a run it started stops (it would time nothing)."""
        if self.run is not None and self.run.active:
            self.run.stop()


#: the geometry's own one-click fixes, by key
GEO_FIXES = {"origin": "Move the drawing to the origin"}


def translate(f: doc.Feature, dx: float, dy: float) -> None:
    """Move a feature by (dx, dy), in place."""
    f.x += dx
    f.y += dy
    f.cx += dx
    f.cy += dy
    f.points = [(x + dx, y + dy) for x, y in f.points]
    f.handles = [None if h is None else (h[0] + dx, h[1] + dy, h[2] + dx, h[3] + dy)
                 for h in f.handles]


def settle_boundaries(c: CaseSpec, was_drawn: bool) -> None:
    """Every edge of the new domain holds a condition, and only edges that exist do
    (the retired editor's `rebase_boundaries`): with a drawn outline the grid's edges
    hold none; back on the whole grid they get the family's; a drawn edge with none
    gets the one the family gives it of its own accord (`spec.derived_kind`)."""
    from atlas.workbench import geometry as wgeo
    from atlas.workbench.spec import Boundary, derived_kind, family_boundaries
    d = c.domain
    names = set(wgeo.drawn_edge_names(d))
    keep = [b for b in c.boundaries if (b.edge in names if b.drawn else d.outline is None)]
    if d.outline is None and was_drawn and not any(not b.drawn for b in keep):
        keep = list(family_boundaries(c.physics.family)) + keep
    taken = {b.id for b in keep}
    have = {b.edge for b in keep}
    for name in wgeo.drawn_edge_names(d):
        if name in have:
            continue
        bid = "B-" + name.replace(":", "-")
        k = 2
        while bid in taken:
            bid, k = f"B-{name.replace(':', '-')}-{k}", k + 1
        taken.add(bid)
        keep.append(Boundary(id=bid, edge=name, kind=derived_kind(c, name)))
    c.boundaries = keep


def follow_the_domain(c: CaseSpec) -> None:
    """A domain that has just been drawn: rectangles do not fit it, so windows that
    are all still rectangles give way to windows that follow it (the retired editor's
    rule: as many along it as there were, one per material for style C or M when there
    are two, one for D and the split).  Windows drawn by hand stay."""
    from atlas.workbench.spec import Layout
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


def _scale_cells(spec: CaseSpec, s: float) -> None:
    """A new cell size: what is counted in cells scales with it (rectangular windows,
    boundary spans on the grid's edges); windows that follow the domain are cut again."""
    for w in spec.windows:
        if w.shape == "rect":
            w.x0, w.y0 = int(round(w.x0 * s)), int(round(w.y0 * s))
            w.nx, w.ny = max(1, int(round(w.nx * s))), max(1, int(round(w.ny * s)))
    if any(w.shape == "cells" for w in spec.windows) and spec.layout is None:
        spec.windows = []                       # cut again by the windows fix
    for b in spec.boundaries:
        if not b.drawn:
            b.start = int(round(b.start * s))
            if b.stop is not None:
                b.stop = max(b.start + 1, int(round(b.stop * s)))
    spec.coupling.ramp_cells = max(1, int(round(spec.coupling.ramp_cells * s)))


# ---------------------------------------------------------------------------
# edits as data
# ---------------------------------------------------------------------------


def _walk(doc_, path: list):
    cur = doc_
    for k in path:
        if isinstance(cur, list):
            if not isinstance(k, int) or not -len(cur) <= k < len(cur):
                raise Refused(f"no item {k!r} in a list of {len(cur)}")
            cur = cur[k]
        elif isinstance(cur, dict):
            if not isinstance(k, str) or k not in cur:
                raise Refused(f"no field {k!r}")
            cur = cur[k]
        else:
            raise Refused(f"cannot go into {type(cur).__name__} with {k!r}")
    return cur


def apply_ops(doc_: dict, ops: list) -> dict:
    """Apply edit operations to a case as JSON, in place, and return it.

    Each op is ``{"op": ..., "path": [...]}`` with the path a list of keys and list
    indices from the case's root:

    * ``set``: ``value`` replaces the item at ``path`` (or adds a new key to a dict);
    * ``insert``: ``value`` goes into the list at ``path`` at ``index`` (the end
      when omitted);
    * ``remove``: the item at ``path`` goes;
    * ``move``: in the list at ``path``, the item at ``from`` moves to ``to``.

    The caller validates the result as a whole case; nothing here knows the schema.
    """
    if not isinstance(ops, list) or not ops:
        raise Refused("an edit needs at least one operation")
    for op in ops:
        if not isinstance(op, dict) or "op" not in op or not isinstance(op.get("path"), list):
            raise Refused("each operation needs 'op' and a 'path' list")
        kind, path = op["op"], op["path"]
        if kind == "set":
            if not path:
                raise Refused("set needs a path")
            parent = _walk(doc_, path[:-1])
            k = path[-1]
            if isinstance(parent, list):
                if not isinstance(k, int) or not -len(parent) <= k < len(parent):
                    raise Refused(f"no item {k!r} to set")
                parent[k] = op.get("value")
            elif isinstance(parent, dict):
                if not isinstance(k, str):
                    raise Refused(f"a field name must be text, not {k!r}")
                parent[k] = op.get("value")
            else:
                raise Refused(f"cannot set {k!r} in {type(parent).__name__}")
        elif kind == "insert":
            target = _walk(doc_, path)
            if not isinstance(target, list):
                raise Refused("insert needs a list")
            i = op.get("index", len(target))
            if not isinstance(i, int) or not 0 <= i <= len(target):
                raise Refused(f"no place {i!r} in a list of {len(target)}")
            target.insert(i, op.get("value"))
        elif kind == "remove":
            if not path:
                raise Refused("remove needs a path")
            parent = _walk(doc_, path[:-1])
            k = path[-1]
            if isinstance(parent, list):
                if not isinstance(k, int) or not -len(parent) <= k < len(parent):
                    raise Refused(f"no item {k!r} to remove")
                parent.pop(k)
            elif isinstance(parent, dict):
                if k not in parent:
                    raise Refused(f"no field {k!r} to remove")
                parent.pop(k)
            else:
                raise Refused("remove needs a list or an object")
        elif kind == "move":
            target = _walk(doc_, path)
            a, b = op.get("from"), op.get("to")
            if (not isinstance(target, list) or not isinstance(a, int)
                    or not isinstance(b, int) or not 0 <= a < len(target)
                    or not 0 <= b < len(target)):
                raise Refused("move needs a list and two places in it")
            target.insert(b, target.pop(a))
        else:
            raise Refused(f"unknown operation {kind!r}")
    return doc_


__all__ = ["DEFAULT_CASES_DIR", "GEO_FIXES", "Refused", "Session", "Version", "apply_ops",
           "layer_for", "new_key", "translate"]
