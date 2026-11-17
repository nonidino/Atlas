"""The WebSocket protocol between the page and the server, as plain functions.

**The page sends** one JSON object per message: ``{"id": <number>, "type": <name>,
...arguments}``.  **The server answers** each with ``{"type": "reply", "re": <id>,
"ok": true|false, ...}`` (``error`` says why when not ok), after first sending what
the message changed:

* ``state``: the case, its problems with their fixes, the status chip, and the
  geometry to draw (`Session.state`), whenever the case or its file changed;
* ``notice``: a message for the person (info, success, warning, error).

**The server also pushes**, without being asked:

* ``catalog`` once, on ``hello``: the kinds of simulation and the examples;
* ``run``: a run's progress while it marches and its result cards when it ends
  (`run_frame`); the price is a ratio only (`runcards`);
* ``fields``: the arms' fields, quantised to 16 bits, when a new snapshot exists;
* ``compile``: a compile's progress and the compiler's verdict per seam.

Every message the page can send is in `HANDLERS`, with the arguments it takes;
`handle` refuses anything else before it reaches the session.  The tests drive
`handle` with no socket (``tests/test_studio_shell.py``).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .session import Refused, Session

#: (type name, required) per argument
Args = dict[str, tuple[type | tuple[type, ...], bool]]


@dataclass(frozen=True)
class Handler:
    fn: Callable[..., Any]
    args: Args
    #: whether the reply is followed by a ``state`` push (the case may have changed)
    changes: bool = True


HANDLERS: dict[str, Handler] = {}


def handler(name: str, args: Args | None = None, changes: bool = True):
    def wrap(fn):
        HANDLERS[name] = Handler(fn, args or {}, changes)
        return fn
    return wrap


# -- the case ---------------------------------------------------------------------

@handler("case.new", {"family": (str, True)})
def _new(s: Session, family: str):
    s.start_new(family)


@handler("case.start_over")
def _over(s: Session):
    s.start_new(s.spec.physics.family, over=True)


@handler("case.example", {"key": (str, True)})
def _example(s: Session, key: str):
    s.example(key)


@handler("case.fast")
def _fast(s: Session):
    s.fast()


@handler("case.files", changes=False)
def _files(s: Session):
    return {"files": s.files()}


@handler("case.open", {"name": (str, True)})
def _open(s: Session, name: str):
    s.open(name)


@handler("case.save", {"name": (str, False)})
def _save(s: Session, name: str | None = None):
    path = s.save(name)
    return {"file": path.replace("\\", "/").rsplit("/", 1)[-1]}


@handler("case.revert")
def _revert(s: Session):
    s.revert()


@handler("case.import", {"text": (str, True), "name": (str, False)})
def _import(s: Session, text: str, name: str = "imported file"):
    s.import_text(text, name)


@handler("case.export", changes=False)
def _export(s: Session):
    return s.export()


@handler("case.history", changes=False)
def _history(s: Session):
    return {"lines": list(s.log_lines)}


@handler("case.fix", {"keys": (list, True)})
def _fix(s: Session, keys: list):
    if not all(isinstance(k, str) for k in keys):
        raise Refused("fix keys are text")
    s.fix(list(keys))


@handler("edit.apply", {"ops": (list, True), "label": (str, False),
                        "defaults": (bool, False)})
def _edit(s: Session, ops: list, label: str = "edit", defaults: bool = False):
    return {"applied": s.edit_ops(ops, label, defaults=defaults)}


@handler("edit.undo")
def _undo(s: Session):
    s.undo()


@handler("edit.redo")
def _redo(s: Session):
    s.redo()


# -- the geometry's own actions ------------------------------------------------------

@handler("geometry.check", {"n": (int, False), "areas": (dict, False)}, changes=False)
def _check(s: Session, n: int | None = None, areas: dict | None = None):
    """The Check view at ``n`` cells along the longer side; ``areas``: the page's own
    (Paper.js) areas by feature id, to set beside the server's."""
    if n is not None and not 4 <= n <= 4096:
        raise Refused("a grid of 4 to 4096 cells along the longer side")
    return {"check": s.geometry_check(n, areas)}


@handler("geometry.fix", {"key": (str, True)})
def _geo_fix(s: Session, key: str):
    s.geometry_fix(key)


@handler("geometry.start", {"shape": (str, True), "params": (dict, False)})
def _geo_start(s: Session, shape: str, params: dict | None = None):
    s.geometry_start(shape, params or {})


@handler("geometry.import", {"format": (str, True), "data": (str, True), "name": (str, True)})
def _geo_import(s: Session, format: str, data: str, name: str):
    try:
        raw = base64.b64decode(data, validate=True)
    except ValueError:
        raise Refused("the file's data is not base64") from None
    if len(raw) > 64 * 1024 * 1024:
        raise Refused("a file of at most 64 MB")
    s.geometry_import(format, raw, name)


# -- runs and compiles -------------------------------------------------------------

@handler("run.start", {"arms": (list, False)})
def _run(s: Session, arms: list | None = None):
    if arms is not None and not all(isinstance(a, str) for a in arms):
        raise Refused("arms are text")
    run = s.start_run(list(arms) if arms else None)
    return {"arms": list(run.arms), "dropped": dict(run.dropped)}


@handler("run.stop")
def _stop(s: Session):
    s.stop_run()


@handler("compile.start")
def _compile(s: Session):
    s.start_compile()


@handler("state", changes=True)
def _state(s: Session):
    """Ask for the state again (the page reconnected)."""


def handle(session: Session, msg: Any) -> tuple[dict, list[dict]]:
    """One message from the page: (the reply, what to send before it)."""
    if not isinstance(msg, dict):
        return {"type": "reply", "re": None, "ok": False,
                "error": "a message is a JSON object"}, []
    mid = msg.get("id")
    kind = msg.get("type")
    h = HANDLERS.get(kind) if isinstance(kind, str) else None
    if h is None:
        return {"type": "reply", "re": mid, "ok": False,
                "error": f"unknown message type {kind!r}"}, []
    extra = sorted(set(msg) - set(h.args) - {"id", "type"})
    if extra:
        return {"type": "reply", "re": mid, "ok": False,
                "error": f"{kind} does not take {', '.join(map(repr, extra))}"}, []
    kwargs = {}
    for name, (typ, required) in h.args.items():
        if name not in msg or msg[name] is None:
            if required:
                return {"type": "reply", "re": mid, "ok": False,
                        "error": f"{kind} needs {name!r}"}, []
            continue
        v = msg[name]
        bad = (not isinstance(v, bool)) if typ is bool else (
            not isinstance(v, typ) or isinstance(v, bool))
        if bad:
            return {"type": "reply", "re": mid, "ok": False,
                    "error": f"{kind}: {name!r} has the wrong type"}, []
        kwargs[name] = v
    before = session.version
    try:
        out = h.fn(session, **kwargs) or {}
        # an edit the case refused is not an error of the message, but it did nothing:
        # the page goes back to the case as it is
        reply = {"type": "reply", "re": mid, "ok": out.get("applied", True) is not False,
                 **out}
    except Refused as exc:
        session.notify("error", str(exc)[:1].upper() + str(exc)[1:] + ".")
        reply = {"type": "reply", "re": mid, "ok": False, "error": str(exc)}
    pushes = session.drain()
    if h.changes and (session.version != before or kind == "state"):
        pushes.insert(0, session.state())
    return reply, pushes


# -- what is pushed while a job runs -------------------------------------------------

def quantise(f: np.ndarray) -> dict:
    """A field as 16-bit levels between its finite minimum and maximum; 65535 marks a
    cell outside the domain (NaN).  Little-endian, row 0 at the bottom."""
    a = np.asarray(f, dtype=np.float64)
    ok = np.isfinite(a)
    if ok.any():
        lo, hi = float(a[ok].min()), float(a[ok].max())
    else:
        lo, hi = 0.0, 0.0
    span = hi - lo
    q = np.full(a.shape, 65535, dtype="<u2")
    if span > 0:
        q[ok] = np.round((a[ok] - lo) / span * 65534.0).astype("<u2")
    else:
        q[ok] = 0
    return {"ny": int(a.shape[0]), "nx": int(a.shape[1]), "min": lo, "max": hi,
            "data": base64.b64encode(q.tobytes()).decode("ascii")}


@dataclass
class Watch:
    """What the page has already been sent of the current jobs."""
    run: Any = None
    run_key: tuple = ()
    fields: tuple = ()
    compile: Any = None
    compile_status: str = ""


def run_frame(session: Session, w: Watch) -> list[dict]:
    """The frames to push now: the run's progress when it moved, its fields when a
    new snapshot exists, its cards once it ends, and the compile's state."""
    out: list[dict] = []
    run = session.run
    if run is not None:
        p = run.progress()
        # the fields first: a run's last frame then arrives after its last fields
        if (id(run), p.field_version) != w.fields and p.fields:
            w.fields = (id(run), p.field_version)
            out.append({"type": "fields", "version": p.field_version, "stride": p.stride,
                        "dx": float(session.spec.domain.dx),
                        "arms": {a: quantise(f) for a, f in p.fields.items()}})
        key = (id(run), p.status, p.step, p.arm, p.message)
        if run is not w.run or key != w.run_key:
            frame = {"type": "run", "status": p.status, "step": p.step, "steps": p.steps,
                     "arm": p.arm, "arm_label": run.arm_labels.get(p.arm, p.arm),
                     "message": p.message, "error": p.error, "arms": list(run.arms),
                     "labels": {a: run.arm_labels.get(a, a) for a in run.arms},
                     "step_label": run.step_label, "committed_at": run.committed_at,
                     "this_case": session.run_serial == session.serial,
                     "convergence": p.convergence,
                     "series": {a: {k: v[-200:] for k, v in d.items()}
                                for a, d in p.series.items()}}
            if p.status in ("done", "stopped") and run.results is not None:
                from .runcards import cards
                frame["cards"] = cards(run.results)
            out.append(frame)
            w.run, w.run_key = run, key
    job = session.compile_job
    if job is not None and (job is not w.compile or job.status != w.compile_status):
        frame = {"type": "compile", "status": job.status, "error": job.error,
                 "this_case": session.compiled_serial == session.serial}
        if job.summary is not None:
            frame["summary"] = job.summary.as_dict()
        out.append(frame)
        w.compile, w.compile_status = job, job.status
    return out


__all__ = ["HANDLERS", "Handler", "Watch", "handle", "quantise", "run_frame"]
