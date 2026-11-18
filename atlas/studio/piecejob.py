"""The Pieces screen's server side: every action on the layout as a background job, and
the summary the page draws (plan §5).

A cut can take a minute (the S-channel's strip cut solves two harmonic problems on a
1153 x 129 mesh), so every action (`cut`, `split`, `merge`, a chart method, experts, the
classical check) runs in a thread; the page is pushed the job's state and, when it
ends, the layout's summary:

* each piece: its outline and every eighth chart line to draw, its material, its chart
  method, Proposition 3's certificate (the smallest ``det J`` relative to its mean, and
  why it fails when it does), the largest skew of a cell corner, the CO-D card's
  envelope quantities and refusals, its expert and, for every expert offered, whether
  it is admitted on this piece and why not;
* each seam and outer side: its verdict (conforming seams *admit*, nonconforming ones
  *admit-uncertified*, outer sides their boundary condition) and its port (THERM);
* the check: the coupled classical solve against `host.monolithic`, with the experts'
  16 port modes (the truncation floor) and with every side's whole trace (round-off on
  conforming seams).

The layout belongs to the session, not the case file; a change to the domain, its
regions, materials or boundaries marks it stale.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
import traceback

import numpy as np

from . import pieces as P

COLOURS = {"classical": "C", "H2r": "R", "TL0": "T0", "TL-s0": "TL", "TL-s1": "TL",
           "Sp-s0": "S′", "Sp-s1": "S′"}


def fingerprint(spec) -> str:
    """What a layout depends on: the domain, regions, materials, boundaries, the step."""
    d = json.loads(spec.to_json())
    keep = {k: d.get(k) for k in ("domain", "regions", "materials", "boundaries", "run", "physics")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:16]


def max_skew(X: np.ndarray) -> float:
    """The largest departure from 90 degrees of a cell corner's angle (two corners per
    cell, its first and its opposite)."""
    dxi = X[:, 1:] - X[:, :-1]
    det = X[1:, :] - X[:-1, :]
    a, b = dxi[:-1, :], det[:, :-1]
    c, d = -dxi[1:, :], -det[:, 1:]

    def ang(u, v):
        cr = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
        dt = u[..., 0] * v[..., 0] + u[..., 1] * v[..., 1]
        return np.degrees(np.abs(np.arctan2(cr, dt)))
    sk = np.maximum(np.abs(90.0 - ang(a, b)), np.abs(90.0 - ang(c, d)))
    return float(sk.max())


def _round(a, k=6):
    return np.round(np.asarray(a, float), k).tolist()


def summarise(cut: P.Cut, spec, rows: list, envelope) -> dict:
    from atlas.chartop import host as hs
    from . import experts as X
    probs = P.problems(cut, spec)
    out = {"cutter": cut.cutter, "params": cut.params, "n": cut.n, "notes": list(cut.notes),
           "chords": P.chords(cut), "pieces": [], "seams": None, "refused": None}
    for i, (pc, pr) in enumerate(zip(cut.pieces, probs)):
        Xg = pr.chart.X
        n = Xg.shape[0] - 1
        step = max(1, n // 8)
        sub = max(1, n // 32)
        lines = [Xg[j, ::sub] for j in range(0, n + 1, step)] + [Xg[::sub, k] for k in range(0, n + 1, step)]
        ring = np.vstack([Xg[0, :], Xg[:, -1], Xg[-1, ::-1], Xg[::-1, 0]])[:: max(1, n // 24)]
        ce = P.certificate(pr.chart, pr.log_kappa_hat, None if pr.steady else pr.fo, envelope)
        adm = {}
        for r in rows:
            if r["offered"]:
                try:
                    adm[r["key"]] = X.admitted(r["key"], pr, envelope)
                except Exception as exc:
                    adm[r["key"]] = f"cannot be measured: {type(exc).__name__}"
        out["pieces"].append({
            "index": i, "label": pc.label, "material": pc.material, "own": pc.own,
            "method": pc.method, "chart": pr.chart.method,
            "outline": _round(ring), "lines": [_round(l) for l in lines],
            "sides_xy": [_round(Xg[0, ::sub]), _round(Xg[::sub, -1]), _round(Xg[-1, ::-1][::sub]),
                         _round(Xg[::-1, 0][::sub])],
            "centre": _round(Xg[n // 2, n // 2]),
            "certificate": {"ok": ce["ok"], "min_detJ_rel": ce["min_detJ_rel"], "reason": ce["reason"],
                            "nonpositive": ce["nonpositive_corners"], "simple": ce["simple"]},
            "skew": max_skew(Xg) if ce["ok"] else None,
            "envelope": {"measure": ce.get("measure"), "refusals": ce.get("refusals") or []},
            "fo": None if pr.steady else float(pr.fo), "kappa": float(pr.kappa_ref),
            "expert": pc.expert, "admitted": adm,
            "square": X.is_square(pr), "chords": pc.chords,
        })
    try:
        L = hs.Layout(probs, P.make_bc(spec), kind=P.KIND, m=P.M)
        out["seams"] = P.seams(L)
        sides = {}
        for e in L.edges:
            i, s = e.owner
            if e.other is None:
                sides[(i, s)] = {"to": None, "kind": e.bc[0] if e.bc else "insulated"}
            else:
                j, t = e.other[0], e.other[1]
                sides[(i, s)] = {"to": j, "kind": "THERM", "conforming": bool(e.conforming)}
                sides[(j, t)] = {"to": i, "kind": "THERM", "conforming": bool(e.conforming)}
        for p in out["pieces"]:
            p["sides"] = [sides.get((p["index"], s), {"to": None, "kind": "?"}) for s in range(4)]
    except Exception as exc:
        out["refused"] = str(exc)
        for p in out["pieces"]:
            p["sides"] = [{"to": None, "kind": "?"} for _ in range(4)]
    ok = [p["certificate"]["ok"] for p in out["pieces"]]
    out["certified"] = [sum(ok), len(ok)]
    dets = [p["certificate"]["min_detJ_rel"] for p in out["pieces"] if p["certificate"]["min_detJ_rel"] is not None]
    out["min_detJ_rel"] = min(dets) if dets else None
    sk = [p["skew"] for p in out["pieces"] if p["skew"] is not None]
    out["max_skew"] = max(sk) if sk else None
    return out


def best_admitted(summary: dict, rows: list) -> list[str]:
    """Each piece's cheapest admitted expert: H2r on a square of one material, else the
    trained two-level network where its card admits it, else the classical solve."""
    offered = [r["key"] for r in rows if r["offered"]]
    out = []
    for p in summary["pieces"]:
        adm = p["admitted"]
        if "H2r" in offered and adm.get("H2r") is None and p["square"]:
            out.append("H2r")
        elif "TL-s0" in offered and adm.get("TL-s0") is None:
            out.append("TL-s0")
        else:
            out.append("classical")
    return out


class PiecesJob:
    """One action on the layout, in the background.  ``fn(cut) -> (cut, extra)``."""
    _ids = 0

    def __init__(self, session, label: str, fn):
        PiecesJob._ids += 1
        self.id = PiecesJob._ids
        self.session = session
        self.label = label
        self.fn = fn
        self.status = "created"
        self.error: str | None = None
        self.started = time.time()
        self.finished: float | None = None
        self._thread = None

    @property
    def active(self) -> bool:
        return self.status in ("starting", "working")

    def start(self):
        self.status = "starting"
        self._thread = threading.Thread(target=self._run, name="studio-pieces", daemon=True)
        self._thread.start()
        return self

    def run_blocking(self):
        self.status = "starting"
        self._run()
        return self

    def _run(self):
        s = self.session
        self.status = "working"
        try:
            s._pieces_apply(self.fn)
            self.status = "done"
        except P.CutError as exc:
            self.error = str(exc)
            self.status = "failed"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            self.status = "failed"
        self.finished = time.time()

    def progress(self) -> dict:
        return {"id": self.id, "status": self.status, "label": self.label, "error": self.error,
                "seconds": round((self.finished or time.time()) - self.started, 1)}


__all__ = ["COLOURS", "PiecesJob", "best_admitted", "fingerprint", "max_skew", "summarise"]
