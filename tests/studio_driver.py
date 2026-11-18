"""Driving the studio from a test, without a browser or a socket.

Every action of the studio's page is one message (`atlas/studio/protocol.py`), which
the server applies to the page's `Session` through `protocol.handle`; these helpers
send the same messages the same way and read what the page would be told: the
reply, the notices pushed with it, and the state.  They replace ``workbench_ui.py``,
which clicked the retired Panel page's widgets.

What they cannot see is whether the page draws it: that is checked by driving the
real page (`scripts/studio_drive.py` and its walks), the project's rule since the
RaceLab control that did nothing.

Shapes are given in cells, as the retired editor's outlines were, and sent in the
physics' length unit (cells times the case's cell size), as the page sends them.
"""

from __future__ import annotations

import json

from atlas.studio import protocol
from atlas.studio.session import Session
from atlas.workbench.spec import example_case


def session(cases_dir, key: str = "wake-array-3") -> Session:
    """A page's session on the example ``key``, saving into ``cases_dir``.  The
    retired workbench opened on the wind farm, which its tests assumed."""
    return Session(cases_dir=str(cases_dir), spec=example_case(key), key=key)


def synchronous(monkeypatch) -> None:
    """Runs and compiles march inside the message that starts them (the server starts
    each in a background thread), so a test reads their results from the reply on."""
    from atlas.workbench import compile as compile_
    from atlas.workbench import runner
    monkeypatch.setattr(runner.CaseRun, "start", lambda self: self.run_blocking())
    monkeypatch.setattr(compile_.CompileJob, "start", lambda self: self.run_blocking())


def idle(monkeypatch) -> None:
    """Runs and compiles are created and never marched: a test about the actions, not
    the runs, would otherwise time a background thread against the rest of the suite."""
    from atlas.workbench import compile as compile_
    from atlas.workbench import runner
    monkeypatch.setattr(runner.CaseRun, "start", lambda self: None)
    monkeypatch.setattr(compile_.CompileJob, "start", lambda self: None)


def frames(s: Session) -> dict[str, dict]:
    """What the server pushes the page about the jobs now (`protocol.run_frame`): the
    run's frame (its cards once it ends), its fields, and the compile's, by type."""
    return {f["type"]: f for f in protocol.run_frame(s, protocol.Watch())}


def send(s: Session, kind: str, **args) -> tuple[dict, list[dict]]:
    """One message as the page sends it: (the reply, what was pushed before it)."""
    return protocol.handle(s, {"id": 1, "type": kind, **args})


def ok(s: Session, kind: str, **args) -> list[dict]:
    """A message that must be applied; what was pushed with it."""
    reply, pushes = send(s, kind, **args)
    assert reply["ok"], (kind, reply, notices(pushes))
    return pushes


def refused(s: Session, kind: str, **args) -> tuple[str, list[dict]]:
    """A message that must be refused: (its reason, what was pushed with it)."""
    reply, pushes = send(s, kind, **args)
    assert reply["ok"] is False, (kind, reply)
    return reply.get("error") or " ".join(notices(pushes, "error")), pushes


def notices(pushes: list[dict], kind: str | None = None) -> list[str]:
    """The notices among ``pushes`` (of one kind: info, success, warning, error)."""
    return [p["text"] for p in pushes
            if p["type"] == "notice" and (kind is None or p["kind"] == kind)]


def set_op(path, value) -> dict:
    return {"op": "set", "path": list(path), "value": value}


def edit(s: Session, *ops: dict, label: str = "edit") -> tuple[dict, list[dict]]:
    """An edit as data, as the page sends one (``edit.apply``)."""
    return send(s, "edit.apply", ops=list(ops), label=label)


def errors(s: Session) -> list[str]:
    """The problems that stop the case from running, the geometry's first."""
    return [r["message"] for r in s.issue_rows() if r["severity"] == "error"]


def features(s: Session) -> list[dict]:
    """The drawing as the page holds it: the feature tree's features, as data."""
    return json.loads(s.geom.model_dump_json())["features"]


def path(fid: str, points, dx: float, edges=None, bulge=None, role: str = "add",
         name: str | None = None, material: str | None = None) -> dict:
    """A drawn path (`document.Feature`, kind ``path``), its ``points`` in cells."""
    n = len(points)
    return {"id": fid, "kind": "path", "role": role, "name": name or fid,
            "material": material, "points": [[float(x) * dx, float(y) * dx] for x, y in points],
            "edges": list(edges or ["line"] * n), "bulge": [float(b) for b in (bulge or [0.0] * n)],
            "handles": [None] * n}


def rect(fid: str, x0, y0, nx, ny, dx: float, role: str = "region", name: str | None = None,
         material: str | None = None) -> dict:
    """A rectangle from its corner and size, in cells."""
    return {"id": fid, "kind": "rect", "role": role, "name": name or fid, "material": material,
            "x": x0 * dx, "y": y0 * dx, "w": nx * dx, "h": ny * dx}


def circle(fid: str, cx, cy, r, dx: float, role: str = "region", name: str | None = None,
           material: str | None = None) -> dict:
    """A circle from its centre and radius, in cells."""
    return {"id": fid, "kind": "ellipse", "role": role, "name": name or fid,
            "material": material, "cx": cx * dx, "cy": cy * dx, "rx": r * dx, "ry": r * dx}


def add_feature(s: Session, feature: dict, label: str = "drew a shape"):
    """One shape drawn on the canvas: the page inserts it at the end of the tree."""
    return edit(s, {"op": "insert", "path": ["geometry", "features"], "value": feature},
                label=label)


def draw_domain(s: Session, points, edges=None, bulge=None, label: str = "drew the outline"):
    """The domain drawn again as one outline (``points`` in cells), on the case's own
    grid and in one edit, as the retired editor's Draw replaced the outline: the
    drawing's adding shapes give way to it, and its holes and regions stay."""
    keep = [f for f in features(s) if f["role"] != "add"]
    f = path("outline", points, s.spec.domain.dx, edges, bulge, name="Outline")
    return edit(s, set_op(["geometry", "features"], [f] + keep), label=label)


def draw_from_blank(s: Session, feature: dict, label: str = "drew the domain"):
    """The studio's own first scenario: Start, *Blank*, then one shape drawn; the grid
    fits the first shape of an empty drawing."""
    ok(s, "geometry.start", shape="blank")
    return add_feature(s, feature, label=label)


def family(s: Session, fid: str) -> dict:
    """The kind ``fid`` as the page's catalog describes it."""
    return next(f for f in s.catalog()["families"] if f["id"] == fid)


__all__ = ["add_feature", "circle", "draw_domain", "draw_from_blank", "edit", "errors",
           "family", "features", "frames", "idle", "notices", "ok", "path", "rect",
           "refused", "send", "session", "set_op", "synchronous"]
