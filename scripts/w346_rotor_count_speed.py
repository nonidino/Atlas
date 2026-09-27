"""W346 -- does classical domain decomposition beat the full-domain solve as the farm grows?

The owner's target for outcome C1 (2026-09-26): *when there are a large number of
rotors, the decomposition works faster.*  The record before this script said the
opposite as often as it said yes: the composed classical column cost 0.93, 1.31
and 1.02 of a monolith step at 2, 6 and 12 windows (Tier 48's ledger), 1.54x
slower at six windows on one box and 3.98x faster at twelve on another.  None of
those runs controlled for WHERE a decomposed column can save time, so this one
separates the three ways it can:

  ``E``      the as-built composed classical column: the elliptic part EXPOSED
             (each window advects and diffuses; the composition layer projects the
             assembled field once per macro-step, `L2/R10`'s arrangement), windows
             stepped as one serial batch.  Its saving over the monolith, if any, is
             algorithmic (one global projection per macro-step instead of one per
             sub-step) and cache locality (128 x 128 windows fit in cache).
  ``Ep<T>``  the same column with its windows split across ``T`` threads, every
             chunk forced to the batch's own sub-step count, so the result is
             **bit for bit** ``E``'s (asserted).  Its extra saving is parallelism,
             which a monolith cannot have without being cut.
  ``Elts<T>`` each window takes the sub-steps ITS OWN velocity needs (local time
             stepping), on ``T`` threads.  Not bitwise; its accuracy against the
             monolith is measured beside ``E``'s.
  ``F``      the monolith: the same discretization on the undivided domain, one
             core, as built.  ``Fw<T>`` is the monolith with ``scipy.fft`` given
             ``T`` workers, the one global operation it could thread.

Accuracy: every marched arm against the monolith, from the freestream, ``K``
macro-steps: farm power and the rms velocity difference.  Speed: interleaved, the
minimum over repeats of the mean per call, on the state the monolith reached.

Writes ``out/w346/w346.json`` after every rung.  Nothing is downloaded.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse                                                         # noqa: E402
import io                                                               # noqa: E402
import json                                                             # noqa: E402
import sys                                                              # noqa: E402
import time                                                             # noqa: E402
from concurrent.futures import ThreadPoolExecutor                       # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

import numpy as np                                                      # noqa: E402
import scipy.fft                                                        # noqa: E402

import w100_scaling_ladder as L                                         # noqa: E402
from atlas.cases import scaling_ladder as sl                            # noqa: E402
from atlas.cases import wake_array as wa                                # noqa: E402

OUT = os.path.join(HERE, "out", "w346")

#: (columns, rows) of windows.  N = 2 ... 24 are CS-7's rungs; 48 and 80 extend
#: the ladder by the same motif rule, which gives 21 and 36 rotors.
RUNGS = {2: (2, 1), 6: (3, 2), 12: (4, 3), 24: (6, 4), 48: (8, 6), 80: (10, 8)}

#: **Registered before any arm ran** (2026-09-26).  Evaluated by `evaluate`.
PREDICTIONS = {
    "B1": "control: the threaded column Ep<T> is bit for bit the serial column E "
          "at every rung and thread count, and E is bit for bit Tier 48's `Maps.E` "
          "at N = 6",
    "B2": "the serial composed column E is FASTER than the monolith (cost ratio "
          "below 1) at N >= 24, and its ratio falls as N grows",
    "B3": "with 8 threads, Ep8 is at least 3x faster than the monolith at N >= 24",
    "B4": "local time stepping takes at least 10% fewer window sub-steps than the "
          "common count at N >= 24",
    "B5": "accuracy: the composed column's farm power is within 25% of the "
          "monolith's at every rung, and local time stepping moves it by under 1% "
          "of the monolith's",
    "B6": "the monolith-over-Ep8 speedup rises monotonically with the rotor count",
}


def evaluate(art):
    out = {}
    rungs = art.get("rungs", {})

    def v(ok):
        return "not measured" if ok is None else ("held" if ok else "FAILED")

    bit = []
    for key, r in rungs.items():
        bit += [x for x in r.get("bitwise", {}).values()]
    ctrl = art.get("control_maps_E")
    ok = None if not bit else (all(bit) and (ctrl is None or ctrl))
    out["B1"] = {"got": {"threaded==serial": bit, "E==Maps.E(N6)": ctrl}, "verdict": v(ok)}

    ratios = {k: r["cost"]["ratio"]["E"] for k, r in rungs.items() if "cost" in r}
    big = {k: x for k, x in ratios.items() if int(k[1:]) >= 24}
    order = [ratios[k] for k in sorted(ratios, key=lambda s: int(s[1:]))]
    ok = None if not big else (all(x < 1 for x in big.values())
                               and all(b <= a for a, b in zip(order, order[1:])))
    out["B2"] = {"got": ratios, "verdict": v(ok)}

    sp = {k: 1.0 / r["cost"]["ratio"]["Ep8"] for k, r in rungs.items()
          if "cost" in r and "Ep8" in r["cost"]["ratio"]}
    big = {k: x for k, x in sp.items() if int(k[1:]) >= 24}
    out["B3"] = {"got": sp, "verdict": v(None if not big else all(x >= 3 for x in big.values()))}

    sav = {k: r["lts"]["substep_saving"] for k, r in rungs.items() if "lts" in r}
    big = {k: x for k, x in sav.items() if int(k[1:]) >= 24}
    out["B4"] = {"got": sav, "verdict": v(None if not big else all(x >= 0.10 for x in big.values()))}

    acc = {k: r["accuracy"] for k, r in rungs.items() if "accuracy" in r}
    ok = None
    if acc:
        ok = all(abs(a["E"]["power_rel_diff"]) <= 0.25 for a in acc.values()) and all(
            abs(a["Elts8"]["power_rel_diff"] - a["E"]["power_rel_diff"]) <= 0.01
            for a in acc.values() if "Elts8" in a)
    out["B5"] = {"got": {k: {arm: a[arm]["power_rel_diff"] for arm in a
                             if isinstance(a[arm], dict) and "power_rel_diff" in a[arm]}
                         for k, a in acc.items()}, "verdict": v(ok)}

    pts = sorted(((r["n_rotors"], 1.0 / r["cost"]["ratio"]["Ep8"]) for r in rungs.values()
                  if "cost" in r and "Ep8" in r["cost"]["ratio"]), key=lambda p: p[0])
    ok = None if len(pts) < 3 else all(b[1] >= a[1] for a, b in zip(pts, pts[1:]))
    out["B6"] = {"got": pts, "verdict": v(ok)}
    return out


# ---------------------------------------------------------------------------


def _retry(fn, attempts=40, pause=0.25):
    for k in range(attempts):
        try:
            return fn()
        except PermissionError:
            if k == attempts - 1:
                raise
            time.sleep(pause)


def _f(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return [_f(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): _f(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_f(v) for v in x]
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    return x


def persist(art):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "w346.json")
    tmp = path + ".tmp"

    def write():
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_f(art), fh, indent=1)

    _retry(write)
    _retry(lambda: os.replace(tmp, path))
    return path


def load_art():
    path = os.path.join(OUT, "w346.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


# ---------------------------------------------------------------------------


class _Umax:
    """A backend proxy whose ``amax`` returns a fixed value.

    ``WindowNS.step_batch`` reads its sub-step count off
    ``b.amax(b.hypot(u, v))`` of the batch it is handed -- the only call to
    ``amax`` in the class -- so a chunk of the batch would otherwise pick its
    own count.  Handing every chunk the whole batch's maximum makes the chunks
    take the serial column's sub-steps, which is what makes them bitwise.
    """

    def __init__(self, inner, value):
        self._inner, self._value = inner, value

    def amax(self, _x):
        return self._value

    def __getattr__(self, name):
        return getattr(self._inner, name)


class Rung:
    def __init__(self, n: int, threads=(2, 4, 8, 14)):
        self.n = n
        self.r = sl.rung(*RUNGS[n])
        self.t = self.r.tiling
        self.active = {x.rotor_id for x in self.t.rotors}
        self.ny, self.nx = self.r.shape
        self.mono = sl.reference_monolith(self.nx, self.ny, wa.NU_REF)
        self.assembly = L.assembly_for(self.r)
        cls = wa._no_projection_class()
        mk = lambda: cls(nu=wa.NU_REF, length=wa.S_LEN, n=wa.N, cfl=0.4,     # noqa: E731
                         transmission="dirichlet")
        self.serial = mk()
        self.tmax = max(threads)
        self.pool_solvers = [mk() for _ in range(max(self.tmax, len(self.t.names)))]
        self.pools = {T: ThreadPoolExecutor(max_workers=T) for T in threads}
        self.substeps = {}

    # -- shared pieces, exactly as `w202_kill_tests.Maps` ------------------
    def forcing(self, u):
        fx, rec = L._forcing(self.r, u, self.active)
        return fx, rec

    def band(self, u, v):
        u, v = L._band(self.r, u, v)
        u[:, -1] = wa.U_INF
        v[:, -1] = 0.0
        return u, v

    def power(self, rec):
        #: `_forcing`'s record maps each rotor to its disk-averaged inflow
        return float(sum(L.power_of(ud) for ud in rec.values())) if rec else 0.0

    # -- the arms -----------------------------------------------------------
    def F(self, u, v, workers=None):
        fx, rec = self.forcing(u)
        if workers:
            with scipy.fft.set_workers(workers):
                uu, vv = self.mono.step_batch(u[None], v[None], wa.MACRO_DT, bc0=None,
                                              force=(fx[None], np.zeros_like(fx)[None]))
        else:
            uu, vv = self.mono.step_batch(u[None], v[None], wa.MACRO_DT, bc0=None,
                                          force=(fx[None], np.zeros_like(fx)[None]))
        self.substeps["F"] = int(self.mono.last_substeps)
        u1, v1 = self.band(uu[0], vv[0])
        return u1, v1, rec

    def _assemble(self, u1, v1):
        au, av = wa.assemble_conservative(self.t, u1, v1, self.assembly)
        return self.band(au, av)

    def E(self, u, v):
        fx, rec = self.forcing(u)
        us, vs, fs = self.t.cut(u), self.t.cut(v), self.t.cut(fx)
        u1, v1 = self.serial.step_batch(us, vs, wa.MACRO_DT, bc0=None,
                                        force=(fs, np.zeros_like(fs)))
        self.substeps["E"] = int(self.serial.last_substeps) * len(self.t.names)
        a, b = self._assemble(u1, v1)
        return a, b, rec

    def Ep(self, u, v, T: int, lts: bool = False):
        fx, rec = self.forcing(u)
        us, vs, fs = self.t.cut(u), self.t.cut(v), self.t.cut(fx)
        B = us.shape[0]
        if lts:
            chunks = [np.array([k]) for k in range(B)]
        else:
            chunks = [c for c in np.array_split(np.arange(B), T) if c.size]
        umax = float(np.max(np.hypot(us, vs)))
        u1, v1 = np.empty_like(us), np.empty_like(vs)
        subs = [0] * len(chunks)

        def work(i):
            idx = chunks[i]
            s = self.pool_solvers[i]
            if not lts:
                s.b = _Umax(getattr(s.b, "_inner", s.b), umax)
            elif isinstance(s.b, _Umax):
                s.b = s.b._inner
            a, b = s.step_batch(us[idx], vs[idx], wa.MACRO_DT, bc0=None,
                                force=(fs[idx], np.zeros_like(fs[idx])))
            u1[idx], v1[idx] = a, b
            subs[i] = int(s.last_substeps) * idx.size

        list(self.pools[T].map(work, range(len(chunks))))
        self.substeps["Elts" if lts else "Ep"] = int(sum(subs))
        a, b = self._assemble(u1, v1)
        return a, b, rec

    def freestream(self):
        return np.ones(self.r.shape), np.zeros(self.r.shape)


def control_maps_E(art):
    """E against Tier 48's own `Maps.E` at N = 6, from a developed state."""
    import w202_kill_tests as K                                          # noqa: PLC0415
    M = K.Maps(6, load_poseidon=False)
    R = Rung(6, threads=(2,))
    u, v = R.freestream()
    for _ in range(8):
        u, v, _ = R.F(u, v)
    a1, b1 = M.E(u.copy(), v.copy())
    a2, b2, _ = R.E(u.copy(), v.copy())
    ok = bool(np.array_equal(a1, a2) and np.array_equal(b1, b2))
    art["control_maps_E"] = ok
    persist(art)
    print(f"  control: E == Maps.E at N6: {ok}", flush=True)


def stage_rung(n, art, K_steps=40, repeats=3, calls=2, threads=(2, 4, 8, 14),
               accuracy=True):
    R = Rung(n, threads=threads)
    out = art.setdefault("rungs", {}).setdefault(f"N{n}", {})
    out.update({"windows": len(R.t.names), "n_rotors": len(R.active),
                "shape": [R.ny, R.nx], "cells": R.ny * R.nx})
    t0 = time.time()
    # --- accuracy: march each arm from the freestream ---
    if accuracy and "accuracy" not in out:
        arms = {"F": lambda u, v: R.F(u, v), "E": lambda u, v: R.E(u, v),
                "Elts8": lambda u, v: R.Ep(u, v, 8, lts=True)}
        fin = {}
        for name, fn in arms.items():
            u, v = R.freestream()
            pw, subs, t1 = [], [], time.time()
            for _s in range(K_steps):
                u, v, rec = fn(u, v)
                pw.append(R.power(rec))
                subs.append(R.substeps.get("Elts" if name == "Elts8" else name))
                if not (np.isfinite(u).all() and np.isfinite(v).all()):
                    raise RuntimeError(f"{name} not finite at N{n} step {_s}")
            fin[name] = {"u": u, "v": v, "power_last5": float(np.mean(pw[-5:])),
                         "power_trace": pw, "substeps": subs,
                         "wall_s": time.time() - t1}
            print(f"    N{n} {name}: farm power {fin[name]['power_last5']:.4f}  "
                  f"{time.time() - t1:.0f}s", flush=True)
        acc = {}
        pF = fin["F"]["power_last5"]
        for name in ("E", "Elts8"):
            du = fin[name]["u"] - fin["F"]["u"]
            dv = fin[name]["v"] - fin["F"]["v"]
            acc[name] = {"power": fin[name]["power_last5"],
                         "power_rel_diff": (fin[name]["power_last5"] - pF) / pF if pF else None,
                         "rms_vs_monolith": float(np.sqrt(np.mean(du * du + dv * dv))),
                         "wall_s": fin[name]["wall_s"]}
        acc["F"] = {"power": pF, "wall_s": fin["F"]["wall_s"]}
        acc["K_steps"] = K_steps
        out["accuracy"] = acc
        e_subs = [s for s in fin["E"]["substeps"] if s]
        l_subs = [s for s in fin["Elts8"]["substeps"] if s]
        out["lts"] = {"window_substeps_common": int(sum(e_subs)),
                      "window_substeps_lts": int(sum(l_subs)),
                      "substep_saving": 1.0 - sum(l_subs) / sum(e_subs)}
        np.savez(os.path.join(OUT, f"state_N{n}.npz"), u=fin["F"]["u"], v=fin["F"]["v"])
        persist(art)
    # --- the developed state every timing is taken on ---
    st = os.path.join(OUT, f"state_N{n}.npz")
    if os.path.isfile(st):
        z = np.load(st)
        u, v = z["u"], z["v"]
    else:
        u, v = R.freestream()
        for _ in range(12):
            u, v, _ = R.F(u, v)
    # --- bitwise control: the threaded column against the serial one ---
    ref = R.E(u.copy(), v.copy())
    bit = {}
    for T in threads:
        got = R.Ep(u.copy(), v.copy(), T)
        bit[f"Ep{T}"] = bool(np.array_equal(got[0], ref[0]) and np.array_equal(got[1], ref[1]))
    out["bitwise"] = bit
    # --- timing: interleaved, min over repeats of the mean per call ---
    arms = {"F": lambda: R.F(u.copy(), v.copy()),
            "Fw8": lambda: R.F(u.copy(), v.copy(), workers=8),
            "E": lambda: R.E(u.copy(), v.copy())}
    for T in threads:
        arms[f"Ep{T}"] = (lambda T=T: R.Ep(u.copy(), v.copy(), T))
    arms["Elts8"] = lambda: R.Ep(u.copy(), v.copy(), 8, lts=True)
    best = {k: float("inf") for k in arms}
    for _ in range(repeats):
        for k, fn in arms.items():
            t1 = time.perf_counter()
            for _c in range(calls):
                fn()
            best[k] = min(best[k], (time.perf_counter() - t1) / calls)
    out["cost"] = {"seconds_per_step": best,
                   "ratio": {k: best[k] / best["F"] for k in best},
                   "repeats": repeats, "calls": calls,
                   "substeps_F": R.substeps.get("F")}
    out["wall_s"] = time.time() - t0
    for p in R.pools.values():
        p.shutdown()
    persist(art)
    print(f"  N{n} ({len(R.active)} rotors): F {best['F']:.3f}s  E/F "
          f"{best['E'] / best['F']:.3f}  Ep8/F {best['Ep8'] / best['F']:.3f}  "
          f"Elts8/F {best['Elts8'] / best['F']:.3f}  bitwise {all(bit.values())}",
          flush=True)


def stage_retime(n, art, repeats=5, calls=3, threads=(4, 8)):
    """An independent second draw of the timing, in a fresh process.

    The first pass's thread-count column was non-monotone (at N24 two threads
    read 0.681 of the monolith and four 0.176), which is the signature of a
    timing taken once.  This re-times the arms that carry the claim -- the
    monolith, the serial column and the threaded one -- on the same saved
    state, and the two draws are reported side by side rather than merged.
    """
    R = Rung(n, threads=threads)
    z = np.load(os.path.join(OUT, f"state_N{n}.npz")) if os.path.isfile(
        os.path.join(OUT, f"state_N{n}.npz")) else None
    if z is not None:
        u, v = z["u"], z["v"]
    else:
        u, v = R.freestream()
        for _ in range(12):
            u, v, _ = R.F(u, v)
    arms = {"F": lambda: R.F(u.copy(), v.copy()), "E": lambda: R.E(u.copy(), v.copy())}
    for T in threads:
        arms[f"Ep{T}"] = (lambda T=T: R.Ep(u.copy(), v.copy(), T))
    samples = {k: [] for k in arms}
    for _ in range(repeats):
        for k, fn in arms.items():
            t1 = time.perf_counter()
            for _c in range(calls):
                fn()
            samples[k].append((time.perf_counter() - t1) / calls)
    best = {k: min(s) for k, s in samples.items()}
    med = {k: float(np.median(s)) for k, s in samples.items()}
    out = art.setdefault("rungs", {}).setdefault(f"N{n}", {})
    out["cost_replicate"] = {
        "seconds_min": best, "seconds_median": med,
        "ratio_min": {k: best[k] / best["F"] for k in best},
        "ratio_median": {k: med[k] / med["F"] for k in med},
        "samples": samples, "repeats": repeats, "calls": calls}
    for p in R.pools.values():
        p.shutdown()
    persist(art)
    print(f"  retime N{n}: E/F {best['E'] / best['F']:.3f}  Ep4/F "
          f"{best['Ep4'] / best['F']:.3f}  Ep8/F {best['Ep8'] / best['F']:.3f}  "
          f"(medians {med['Ep4'] / med['F']:.3f}, {med['Ep8'] / med['F']:.3f})", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--retime", default="")
    ap.add_argument("--rungs", default="2,6,12,24,48")
    ap.add_argument("--timing-only", default="80")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--control", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    #: Keep the machine awake for THIS process only: a Standby transition in the
    #: middle of a timing run has already cost this project a suite.  The request
    #: lapses when the process exits; no power setting is changed.
    if sys.platform == "win32":
        import ctypes                                                     # noqa: PLC0415
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    art = load_art()
    art["what"] = "W346: classical domain decomposition against the monolith, by rotor count"
    art["predictions"] = PREDICTIONS
    art.setdefault("registered", "2026-09-26, before any arm ran")
    art["host"] = {"cpu_count": os.cpu_count()}
    if a.control:
        control_maps_E(art)
    for n in [int(x) for x in a.rungs.split(",") if x]:
        stage_rung(n, art, K_steps=a.steps)
    for n in [int(x) for x in a.timing_only.split(",") if x]:
        stage_rung(n, art, accuracy=False)
    for n in [int(x) for x in a.retime.split(",") if x]:
        stage_retime(n, art)
    art["evaluation"] = evaluate(art)
    print("wrote", persist(art), flush=True)
    for k, v in art["evaluation"].items():
        print(f"  {k}: {v['verdict']}  {v['got']}", flush=True)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    main()
