"""Run, phase P5: Single and Compare on the chart operator's pieces (plan §6).

Each window is an arm:

* **full domain**: the cut's pieces merged into one mesh and solved once
  (`host.monolithic`): the same discretisation as the pieces, with no seams;
* **chart operator**: the pieces coupled by the host (`host.Layout`), each answered by
  its expert: *mixed* (as assigned on Pieces) or one expert on every piece.  A piece the
  expert's card refuses is refused, named, before anything is solved.

**Compare** runs A and B **in turns** on this machine, steady or marching the case's
steps (each arm from its own previous field, so B's error is its own, accumulated), and
reports, against A: the energy-norm error of B, the worst piece's, the largest nodal
error, the jump in T across seams, and the heat balance of each.  **Single** runs one
window and measures it against the full domain the same way.  **The price is a ratio**:
B's time over A's (each a median over turns), with A's own spread beside it (its 90th
over 10th percentile, the A/A noise) and the machine's state (power, other Python
processes) recorded; no raw seconds are sent to the page.

Fields reach the page as images: every piece's nodal field sampled onto a uniform grid
over the domain (each pixel its nearest node) and quantised to 16 bits, as the windows
runner's fields are.
"""

from __future__ import annotations

import math
import threading
import time
import traceback
from dataclasses import replace

import numpy as np

from . import experts as X
from . import pieces as P

MAX_STEPS = 40
TURNS_STEADY = 5
RASTER = 240


def _energy(problems, ref, fields):
    """(||ref - f||_M, ||ref||_M) summed over pieces with each piece's kappa_ref, and the
    per-piece relative errors."""
    num = den = 0.0
    per = []
    for p, u, v in zip(problems, ref, fields):
        M = p.matrix()
        e = u - v
        a = p.kappa_ref * float(e @ (M @ e))
        b = p.kappa_ref * float(u @ (M @ u))
        num += a
        den += b
        per.append(math.sqrt(max(a, 0.0) / b) if b > 0 else 0.0)
    return math.sqrt(max(num, 0.0)), math.sqrt(max(den, 0.0)), per


def seam_jumps(layout, fields) -> list:
    """The largest jump in T across each seam (zero for a shared trace)."""
    from atlas.chartop import ports as pt
    out = []
    for e in layout.edges:
        if e.other is None:
            continue
        (i, s), (j, t) = e.owner, e.other
        ni, nj = layout.problems[i].n, layout.problems[j].n
        if ni != nj:
            continue
        a = fields[i][_side_nodes(ni, s)]
        b = fields[j][_side_nodes(nj, t)][::-1]
        out.append(float(np.abs(a - b).max()))
    return out


def _side_nodes(n: int, s: int) -> np.ndarray:
    idx = np.arange((n + 1) * (n + 1)).reshape(n + 1, n + 1)
    return {0: idx[0, :], 1: idx[:, -1], 2: idx[-1, ::-1], 3: idx[::-1, 0]}[s]


def balance(layout, fields, mono_index=None) -> float | None:
    """|net heat through the boundary| over the heat in, from the merged mesh's matrix
    applied to the field: zero for a conserving solution of a steady problem with no
    source."""
    import scipy.sparse as sp
    from scipy.spatial import cKDTree
    probs = layout.problems
    if not all(p.steady for p in probs):
        return None
    pts = np.concatenate([p.chart.X.reshape(-1, 2) for p in probs])
    tree = cKDTree(pts)
    gid = -np.ones(len(pts), int)
    ng = 0
    for k in range(len(pts)):
        if gid[k] >= 0:
            continue
        for q in tree.query_ball_point(pts[k], layout.tol):
            gid[q] = ng
        ng += 1
    offs = np.cumsum([0] + [(p.n + 1) ** 2 for p in probs])
    rows, cols, vals = [], [], []
    u = np.zeros(ng)
    for i, p in enumerate(probs):
        G = gid[offs[i]:offs[i + 1]]
        M = p.matrix().tocoo()
        rows.append(G[M.row]); cols.append(G[M.col]); vals.append(p.kappa_ref * M.data)
        u[G] = fields[i]
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ng, ng))
    r = K @ u                                             # the nodal flux each node gives off
    dnodes = set()
    for e in layout.edges:
        if e.other is None and e.bc is not None and e.bc[0] == "dirichlet":
            i, s = e.owner
            for q in _side_nodes(probs[i].n, s):
                dnodes.add(int(gid[offs[i] + q]))
    d = np.array(sorted(dnodes), int)
    if not len(d):
        return None
    flux = r[d]
    inflow = float(flux[flux > 0].sum())
    return abs(float(flux.sum())) / inflow if inflow > 0 else None


def raster(problems, fields, width: int = RASTER) -> dict:
    """The pieces' nodal fields on a uniform grid over their bounding box (nearest node,
    outside the pieces NaN), quantised like the windows runner's fields."""
    from scipy.spatial import cKDTree
    from .protocol import quantise
    pts = np.concatenate([p.chart.X.reshape(-1, 2) for p in problems])
    vals = np.concatenate([np.asarray(f, float).ravel() for f in fields])
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    W = max(x1 - x0, 1e-300)
    H = max(y1 - y0, 1e-300)
    nx = width
    ny = max(2, int(round(width * H / W)))
    xs = x0 + (np.arange(nx) + 0.5) * W / nx
    ys = y0 + (np.arange(ny) + 0.5) * H / ny
    XX, YY = np.meshgrid(xs, ys)
    q = np.column_stack([XX.ravel(), YY.ravel()])
    tree = cKDTree(pts)
    d, k = tree.query(q)
    img = vals[k]
    # a pixel farther from every node than a coarse cell's half-diagonal is outside
    spacing = max(float(np.max(np.hypot(*np.diff(p.chart.X[:, :2, :], axis=1).reshape(-1, 2).T))) for p in problems)
    img[d > 1.5 * spacing] = np.nan
    out = quantise(img.reshape(ny, nx))
    out["box"] = [float(x0), float(y0), float(x1), float(y1)]
    return out


class Arm:
    """One window: how it solves, and what it built."""

    def __init__(self, spec: dict, cut: P.Cut, rows: list):
        self.method = spec.get("method", "full")
        self.solver = spec.get("solver", "classical")
        if self.method not in ("full", "chart"):
            raise P.CutError(f"a window's method is the full domain or the chart operator, not {self.method!r}")
        if self.method == "chart":
            keys = [p.expert for p in cut.pieces] if self.solver == "mixed" else [self.solver] * len(cut.pieces)
            self.keys = keys
            self.experts = {k: X.build(k, rows=rows) for k in dict.fromkeys(keys)}
            self.expert = (self.experts[keys[0]] if len(self.experts) == 1 else X.Mixed(keys, self.experts))
        self.label = ("full domain" if self.method == "full"
                      else f"chart operator, {self.solver if self.solver != 'mixed' else 'mixed experts'}")

    def solve(self, layout):
        from atlas.chartop import host as hs
        if self.method == "full":
            return hs.monolithic(layout).fields, None
        cp = layout.solve(self.expert)
        return cp.fields, cp


class CORun:
    """A Single or Compare run in a thread; the page reads `progress`."""

    def __init__(self, session, mode: str, a: dict, b: dict | None):
        self.session = session
        self.mode = mode
        self.specs = {"A": a, "B": b} if mode == "compare" else {"A": a}
        self.status = "created"
        self.error: str | None = None
        self.step = 0
        self.steps = 0
        self.result: dict | None = None
        self.frames: list = []
        self.version = 0
        self._stop = threading.Event()
        self._thread = None
        self.lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self.status in ("starting", "running")

    def start(self):
        self.status = "starting"
        self._thread = threading.Thread(target=self._run, name="studio-corun", daemon=True)
        self._thread.start()
        return self

    def run_blocking(self):
        self.status = "starting"
        self._run()
        return self

    def stop(self):
        self._stop.set()

    def progress(self) -> dict:
        with self.lock:
            return {"type": "corun", "status": self.status, "error": self.error, "mode": self.mode,
                    "step": self.step, "steps": self.steps, "result": self.result,
                    "frames": len(self.frames), "version": self.version,
                    "labels": {k: (v or {}).get("label") for k, v in self.specs.items()}}

    def frame(self, k: int) -> dict | None:
        with self.lock:
            return self.frames[k] if 0 <= k < len(self.frames) else None

    def _run(self):
        from atlas.chartop import host as hs
        from atlas.workbench import machine
        from . import cot
        s = self.session
        if cot.is_cot(s.spec):
            return cot.run(self)                  # the coolant's CO-T pieces (cot.py)
        self.status = "running"
        try:
            spec, cut = s.spec, s.cut
            if cut is None:
                raise P.CutError("cut the domain into pieces first (Pieces)")
            rows = s.experts()
            arms = {k: Arm(v, cut, rows) for k, v in self.specs.items()}
            for k, v in self.specs.items():
                v["label"] = arms[k].label
            probs = P.problems(cut, spec)
            layout = hs.Layout(probs, P.make_bc(spec), kind=P.KIND, m=P.M)
            from atlas.chartop.data import ENVELOPE
            for k, arm in arms.items():
                if arm.method == "chart":
                    refused = []
                    for i, (key, pr) in enumerate(zip(arm.keys, probs)):
                        why = X.admitted(key, pr, ENVELOPE)
                        if why:
                            refused.append(f"{cut.pieces[i].label}: {key} refused, {why}")
                    if refused:
                        raise P.CutError(f"window {k}: " + "; ".join(refused[:3]) + (" …" if len(refused) > 3 else ""))
            steady = all(p.steady for p in probs)
            self.steps = 1 if steady else max(1, min(int(spec.run.steps), MAX_STEPS))
            times = {k: [] for k in arms}
            prev = {k: None for k in arms}
            ref_arm = "A" if self.mode == "compare" else None
            full = None if self.mode == "compare" else Arm({"method": "full"}, cut, rows)
            turns = TURNS_STEADY if steady else 1
            last = {}
            per_step = []
            for step in range(self.steps):
                if self._stop.is_set():
                    break
                order = list(arms) if step % 2 == 0 else list(arms)[::-1]
                out = {}
                for turn in range(turns):
                    for k in (order if turn % 2 == 0 else order[::-1]):
                        lay = layout
                        if not steady and prev[k] is not None:
                            lay.problems = [replace(p, g=(f.reshape(p.n + 1, p.n + 1) / p.fo)) for p, f in zip(probs, prev[k])]
                        elif not steady:
                            lay.problems = probs
                        t0 = time.perf_counter()
                        fields, cp = arms[k].solve(lay)
                        times[k].append(time.perf_counter() - t0)
                        out[k] = (fields, cp, list(lay.problems))
                if self.mode == "single":
                    lay = layout
                    lay.problems = out["A"][2]
                    ref_fields, _ = full.solve(lay)
                else:
                    ref_fields = out[ref_arm][0]
                test = "B" if self.mode == "compare" else "A"
                tf, tcp, tprobs = out[test]
                num, den, per = _energy(tprobs, ref_fields, tf)
                mx = max(float(np.abs(a - b).max()) for a, b in zip(ref_fields, tf))
                span = float(np.ptp(np.concatenate([np.ravel(f) for f in ref_fields]))) or 1.0
                rec = {"step": step + 1, "energy_rel": num / den if den > 0 else 0.0, "max_err": mx,
                       "max_over_range": mx / span, "per_piece": per,
                       "worst_piece": int(np.argmax(per)) if per else None}
                if tcp is not None:
                    rec["min_eig"] = tcp.min_eig()
                    rec["calls_per_piece"] = (tcp.calls or 0) / len(tprobs)
                per_step.append(rec)
                for k in arms:
                    prev[k] = [np.asarray(f, float).ravel() for f in out[k][0]]
                fr = {"A": raster(out["A"][2], out["A"][0])}
                if self.mode == "compare":
                    fr["B"] = raster(out["B"][2], out["B"][0])
                    fr["difference"] = raster(out["B"][2], [b - a for a, b in zip(out["A"][0], out["B"][0])])
                else:
                    fr["full domain"] = raster(out["A"][2], ref_fields)
                    fr["difference"] = raster(out["A"][2], [a - r for r, a in zip(ref_fields, out["A"][0])])
                last = {"layout": layout, "out": out, "ref": ref_fields, "test": test}
                with self.lock:
                    self.frames.append(fr)
                    self.step = step + 1
                    self.version += 1
            # the cards
            res = {"steps": self.steps, "steady": steady, "accuracy": per_step[-1] if per_step else None,
                   "series": [{k: r[k] for k in ("step", "energy_rel", "max_err")} for r in per_step],
                   "pieces": [{"label": p.label, "expert": (arms[last["test"]].keys[i] if arms[last["test"]].method == "chart" else "full domain")}
                              for i, p in enumerate(cut.pieces)]}
            if last:
                L = last["layout"]
                tf = last["out"][last["test"]][0]
                jumps = seam_jumps(L, tf)
                res["seams"] = {"max_jump": max(jumps) if jumps else 0.0, "count": len(jumps)}
                res["balance"] = {k: balance(L, last["out"][k][0]) for k in arms}
            if self.mode == "compare":
                ta, tb = np.median(times["A"]), np.median(times["B"])
                res["price"] = {"ratio": float(tb / ta) if ta > 0 else None,
                                "meaning": "B's time over A's, each the median of its turns on this machine: below 1, B is faster",
                                "aa_spread": float(np.percentile(times["A"], 90) / np.percentile(times["A"], 10)) if len(times["A"]) > 1 else None,
                                "turns": len(times["A"])}
            # E4's bar in one frame: the classical superelement's own max error is the floor
            test_arm = arms[last["test"]] if last else None
            if last and test_arm.method == "chart" and steady:
                from atlas.chartop import gates_twolevel as GT
                L = last["layout"]
                cl = L.solve(hs.ClassicalExpert(), refuse_outside=False)
                ref = last["ref"]
                floor = max(float(np.abs(a - b).max()) for a, b in zip(ref, cl.fields))
                span = float(np.ptp(np.concatenate([np.ravel(f) for f in ref]))) or 1.0
                acc = per_step[-1]
                ok = acc["energy_rel"] <= GT.E4_ENERGY and acc["max_err"] <= floor + GT.E4_MAX_OF_RANGE * span
                res["e4"] = {"passed": bool(ok), "energy_bar": GT.E4_ENERGY, "max_bar": floor + GT.E4_MAX_OF_RANGE * span,
                             "floor_max": floor, "title": "E4's bar, one frame",
                             "tip": ("E4 (gates_twolevel): energy error <= 0.02 and max error <= the 16-mode classical "
                                     "superelement's own max error + 0.01 of the range. The registration judges it on the "
                                     "held-out problems in two frames (as registered, and 300 K lower); this is the bar "
                                     "applied to this run only.")}
            res["certificate"] = {"class": "G0", "charts": [sum(1 for p in probs if P.certificate(p.chart)["ok"]), len(probs)],
                                  "min_eig": per_step[-1].get("min_eig") if per_step else None,
                                  "note": "G0: the assembled interface matrix stays positive semidefinite whatever the networks answer (Theorem 1, Lean-checked); Céa constant 1"}
            res["machine"] = machine.machine_record()
            res["threads"] = _threads()
            with self.lock:
                self.result = res
                self.version += 1
            self.status = "stopped" if self._stop.is_set() else "done"
        except P.CutError as exc:
            self.error = str(exc)
            self.status = "failed"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            self.status = "failed"


def _threads() -> dict:
    out = {}
    try:
        import torch
        out["torch"] = torch.get_num_threads()
    except Exception:
        pass
    return out


__all__ = ["Arm", "CORun", "balance", "raster", "seam_jumps"]
