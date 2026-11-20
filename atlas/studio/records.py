"""The chart-operator records, read only (plan §7): never a way to re-run anything.

Every registered evaluation under ``out/chartop/`` is found by its file name, newest
first: ``evaluation-<kind>-<stamp>.json`` (Stage 0's is ``evaluation-<stamp>.json``) and,
for a kind registered but not yet run, ``registration-<kind>.json``.  Nothing about a
kind is written in this module: a kind's gates are read from ``atlas/chartop/gates_<kind>.py``
(each gate's key, its registered title and its bar), its verdicts from its record, where
they are kept in one of three ways (``gates[key]``, ``gates[arm][key]`` for a record that
judges several arms, or the record's own ``key``), and its arms from the record or the
registration.  **So the coolant's registration (``registration-cot.json``), once its file
exists, appears with no change to the page's code** (tested with a file made for the test).
Until then its row is listed as not registered, with its planned arms; it replaced the
Stage 0b placeholder on 2026-11-19, when the owner put the coolant first and Stage 0b
after it.  The same day the owner stopped the coolant's learned track in Phase 3, so the
row reads "stopped, not registered" and says why.

For the selected evaluation the page shows:
* its stamp, its code commit and whether it was a registered run (the gates and code
  fixed by hash before anything was measured);
* its gates, each a chip whose hover text is the gate's registered title and bar;
* its prices, each labelled with its problem set: a geometric mean over every priced
  problem is a different number from one over the held-out problems, and the page never
  sets the two beside each other as if comparable;
* rho = 1 - eps_TL / eps_TL0 per test piece (1 perfect, 0 no help, below 0 worse than
  TL0), as a histogram with its median and 10th percentile, from the record's test rows.
"""

from __future__ import annotations

import glob
import importlib
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHARTOP = os.path.join(ROOT, "out", "chartop")

NAMES = {"twolevel": "Two-level", "squares": "Squares", "hybrid": "Hybrid", "prior": "Prior",
         "forms": "Forms", "stage0": "Stage 0", "cot": "Coolant (CO-T)"}
#: the coolant plan's kind and arms ([[chart-operator-stage-0b-and-1-plan]] section 15): its
#: row is listed, not registered, until ``registration-cot.json`` exists
COOLANT = "cot"
COOLANT_ARMS = ("TL0", "arm P", "arm R", "classical")
#: what a record's prices are over, by the names records give their problem sets
SETS = {"all": "all priced", "held-out six": "held-out 6", "R2's sixteen": "squares (R2's 16)"}
_EVAL = re.compile(r"^evaluation-(?:(?P<kind>[a-z][a-z0-9]*)-)?(?P<stamp>\d{8}-\d{6})\.json$")
_REG = re.compile(r"^registration-(?P<kind>[a-z][a-z0-9]*)(?:-[a-z0-9-]+)?\.json$")


def _load(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _gates_module(kind: str):
    name = "gates" if kind == "stage0" else f"gates_{kind}"
    try:
        return importlib.import_module(f"atlas.chartop.{name}")
    except ImportError:
        return None


def _name(kind: str) -> str:
    """A kind's display name: its known name, else by rule (stage0b reads Stage 0b)."""
    if kind in NAMES:
        return NAMES[kind]
    m = re.match(r"^stage(\d+)([a-z]*)$", kind)
    return f"Stage {m.group(1)}{m.group(2)}" if m else kind.capitalize()


def _titles(kind: str, reg: dict | None = None, rec: dict | None = None) -> list:
    """Each gate's key, registered title and bar: from atlas/chartop/gates_<kind>.py, else
    from the registration's own list, else the record's gate keys as they are."""
    mod = _gates_module(kind)
    out = [{"key": g.key, "title": getattr(g, "title", ""), "bar": getattr(g, "bar", "")}
           for g in (getattr(mod, "GATES", ()) or ())]
    if out:
        return out
    gl = (reg or {}).get("gates")
    if isinstance(gl, list):
        return [{"key": g.get("key"), "title": g.get("title", ""), "bar": g.get("bar", "")}
                for g in gl if isinstance(g, dict) and g.get("key")]
    if isinstance(gl, dict):
        return [{"key": k, "title": (v or {}).get("title", "") if isinstance(v, dict) else str(v), "bar": (v or {}).get("bar", "") if isinstance(v, dict) else ""}
                for k, v in gl.items()]
    g = (rec or {}).get("gates")
    if isinstance(g, dict):
        return [{"key": k, "title": "", "bar": "(no registered title or bar found)"} for k in g]
    return []


def _arms(rec: dict, reg: dict | None) -> list:
    for src in (rec, reg or {}):
        for k in ("arms", "weights", "runs", "mains"):
            v = src.get(k)
            if isinstance(v, dict) and v:
                return list(v)
            if isinstance(v, list) and v and isinstance(v[0], str):
                return list(v)
    return []


def verdict(rec: dict, key: str, main: str | None = None):
    """A gate's verdict in a record: True, False, or None when it is reported, not
    judged; and where it was read (for the test that reads both)."""
    g = rec.get("gates")
    if isinstance(g, dict):
        if isinstance(g.get(key), dict):
            v = g[key]
            return (v.get("met") if "met" in v else None), f"gates.{key}"
        arms = [a for a, v in g.items() if isinstance(v, dict) and isinstance(v.get(key), dict)]
        if arms:
            arm = main if main in arms else arms[0]
            v = g[arm][key]
            return (v.get("met") if "met" in v else None), f"gates.{arm}.{key}"
    v = rec.get(key)
    if isinstance(v, dict) and "met" in v:
        return v["met"], key
    return None, None


def prices(rec: dict) -> list:
    """Every price the record judged, as (arm, problem set, value), labelled by its set."""
    out = []
    g = rec.get("gates") if isinstance(rec.get("gates"), dict) else {}
    for key, v in list(g.items()) + [(k, rec[k]) for k in rec if k.startswith("PR")]:
        if not isinstance(v, dict) or "geometric_mean" not in v:
            continue
        gm = v["geometric_mean"]
        if isinstance(gm, dict):
            for arm, sets in gm.items():
                if isinstance(sets, dict):
                    for s, val in sets.items():
                        out.append({"gate": key, "arm": arm, "set": s, "label": SETS.get(s, s), "value": float(val)})
        elif isinstance(gm, (int, float)):
            out.append({"gate": key, "arm": key, "set": "R2's sixteen" if key == "PR2" else "priced",
                        "label": SETS.get("R2's sixteen", "priced") if key == "PR2" else "priced",
                        "value": float(gm), "bar": v.get("bar")})
    return out


def rho(stem: str, main: str | None) -> dict | None:
    """rho per test piece from the record's test rows, as a histogram."""
    if not main:
        return None
    path = f"{stem}-test-rows-{main}.npz"
    if not os.path.isfile(path):
        path = f"{stem}-test-rows-{main}-s0.npz"          # a record that files its rows per seed
    if not os.path.isfile(path):
        return None
    z = np.load(path)
    if "rho" not in z:
        return None
    r = np.asarray(z["rho"], float)
    r = r[np.isfinite(r)]
    edges = np.concatenate([[-np.inf], np.linspace(0.0, 1.0, 21)])
    clipped = np.clip(r, -1.0, 1.0)
    counts = [int((clipped < 0).sum())] + [int(((clipped >= a) & (clipped < b if b < 1 else clipped <= b)).sum())
                                         for a, b in zip(edges[1:-1], edges[2:])]
    return {"pieces": int(len(r)), "median": float(np.median(r)), "p10": float(np.percentile(r, 10)),
            "below_zero": int((r < 0).sum()), "bins": counts, "edges": [0.0] + np.linspace(0, 1, 21).tolist()[1:],
            "arm": main}


def evaluations(chartop: str = CHARTOP) -> list:
    """Every registered evaluation, newest first; a registration with no evaluation yet
    is listed as registered, not run."""
    out, kinds = [], set()
    for path in glob.glob(os.path.join(chartop, "evaluation-*.json")):
        m = _EVAL.match(os.path.basename(path))
        if not m:
            continue
        kind = m.group("kind") or "stage0"
        stamp = m.group("stamp")
        try:
            rec = _load(path)
        except (OSError, ValueError):
            continue
        reg_path = os.path.join(chartop, f"registration-{kind}.json")
        reg = _load(reg_path) if os.path.isfile(reg_path) else None
        main = rec.get("main") if isinstance(rec.get("main"), str) else None
        if main is None and isinstance(rec.get("gates"), dict):
            arms = [a for a, v in rec["gates"].items() if isinstance(v, dict) and v and all(isinstance(x, dict) for x in v.values())]
            main = arms[0] if arms else None
        gates = []
        for t in _titles(kind, reg, rec):
            met, where = verdict(rec, t["key"], main)
            gates.append(dict(t, met=met, read_from=where))
        code = rec.get("code")
        if isinstance(code, dict):
            code = code.get("commit") or code.get("head")
        out.append({
            "kind": kind, "name": _name(kind), "stamp": stamp,
            "file": os.path.basename(path), "written": rec.get("written"), "code": code,
            "registered_run": rec.get("registered_run"), "registration": str(rec.get("registration") or "")[:400],
            "arms": _arms(rec, reg), "main": main, "gates": gates, "prices": prices(rec),
            "rho": rho(path[:-len(".json")], main), "outcome": rec.get("outcome"),
            "machine": rec.get("machine") or rec.get("platform"), "status": "evaluated"})
        kinds.add(kind)
    for path in glob.glob(os.path.join(chartop, "registration-*.json")):
        m = _REG.match(os.path.basename(path))
        if not m or m.group("kind") in kinds:
            continue
        kind = m.group("kind")
        try:
            reg = _load(path)
        except (OSError, ValueError):
            continue
        out.append({"kind": kind, "name": _name(kind),
                    "stamp": None, "file": os.path.basename(path), "written": reg.get("written") or reg.get("registered"),
                    "code": reg.get("code"), "registered_run": None, "registration": str(reg.get("registration") or "")[:400],
                    "arms": _arms({}, reg), "main": None, "gates": [dict(t, met=None, read_from=None) for t in _titles(kind, reg)],
                    "prices": [], "rho": None, "outcome": None, "machine": None, "status": "registered, not run"})
        kinds.add(kind)
    out.sort(key=lambda e: (e["stamp"] or "99999999-999999"), reverse=True)
    if COOLANT not in kinds:
        out.insert(0, {"kind": COOLANT, "name": _name(COOLANT), "status": "stopped, not registered", "stamp": None,
                       "arms": list(COOLANT_ARMS), "gates": [], "prices": [], "rho": None, "file": None,
                       "note": "The coolant plan (CO-T on laminar cold plates) drafted arm P, a learned coarse basis, "
                               "and arm R, the fine residual on TL0, gated on accuracy (E4, CL-A) and capacity "
                               "(CAP). Its learned track stopped in Phase 3 on 2026-11-19, before any dataset: CAP "
                               "failed at 1.1M and 3M parameters, and the owner stopped it. Never registered; the "
                               "classical CO-T expert stays"})
    return out


__all__ = ["evaluations", "prices", "rho", "verdict"]
