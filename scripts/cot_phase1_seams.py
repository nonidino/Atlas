"""cot_phase1_seams -- the coolant plan's Phase 1 measurement: are CO-T's seams stable up
to a piece Peclet number of 200?

[[chart-operator-stage-0b-and-1-plan]] section 5, Phase 1: "Seam stability measured up to
Pe_piece = 200.  Stop if the seams are not stable at 200: report it, and lower the cap
with the owner."  The classical expert only (exact piece solves): nothing learned.

**What "stable" means here** is `CRITERIA`, fixed in this file before its first run
(2026-11-18) and written into the record word for word.  The seams are stable at a
Peclet number when S1 to S6 all hold there:

- S1, the positive control: with every side's whole trace (m = n) the coupled answer
  equals `host.monolithic` on the same pieces to rounding;
- S2, the coupled interface matrix is positive real at every port count;
- S3, every piece is passive under the ADVEC splits (its exact port matrix has a
  positive semidefinite symmetric part);
- S4, quasi-optimality: at 16 modes per side the coupled field's error is within 10
  times the oracle's (the reference's own traces projected onto the same port space,
  each piece solved exactly inside them);
- S5, no amplification along a channel cut into 1, 2, 4 and 8 pieces at Pe 200: S4 at
  every length;
- S6, Petrov-Galerkin: perturbing every piece's trial and adjoint modes by eps moves
  the interface matrix by O(eps^2) (slope >= 1.8 from eps = 0.1 to 0.003).

Reported beside them, not judged: the THERM-only pairing's passivity (expected
indefinite), the error against m, the outlet's mixing-cup temperature, the balance, and
the Galerkin test's slope (expected 1 where the flow matters).

**The grid.**  Pe in {0, 20, 50, 100, 200}; the three layouts of `transport_cases`
(channel K = 4, plate K = 3, bend 2 x 3), steady and one implicit step (dt = 1 s:
Fourier numbers about 0.07 in the water and 28 in the aluminium).  S1 runs at the
coarsest grid the cell Peclet number admits (n = 32 up to Pe 64, 64 up to 128, 128
above), S2-S4 and S6 at n = 128 for every Pe (one grid, so only Pe varies), S5 at
n = 128.

Writes ``out/chartop/cot/phase1-seams.json`` and ``.txt``.  ``--quick`` runs a small grid
for the tests (n = 16 and 32, Pe up to 50; nothing is judged).

    python scripts/cot_phase1_seams.py [--quick] [--out DIR]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np

from atlas.chartop import fe
from atlas.chartop import transport as tr
from atlas.chartop import transport_cases as tc
from atlas.chartop import transport_host as th
from atlas.chartop.host import monolithic

#: registered 2026-11-18, before the first run (see the module docstring)
CRITERIA = {
    "S1": "positive control: with whole traces (m = n), max |u - u_monolithic| <= 1e-10 "
          "max |u_monolithic|, every layout, Pe and problem",
    "S2": "positive real: the smallest eigenvalue of the symmetric part of the free "
          "interface matrix is > 0 at m = 4, 8, 16 and 32, every layout, Pe and problem",
    "S3": "passive pieces: under the skew and upwind splits every piece's exact port "
          "matrix (m = 16) has lambda_min(sym) >= -1e-10 max |lambda|",
    "S4": "quasi-optimal: at m = 16 the coupled relative energy error <= 10 times the "
          "oracle's, wherever the oracle's exceeds 1e-10, every layout, Pe and problem",
    "S5": "no amplification: on the channel at Pe 200 (n = 128, m = 16) cut into 1, 2, 4 "
          "and 8 pieces, S4 holds at every length",
    "S6": "Petrov-Galerkin: the slope of log ||S~ - S|| against log eps over eps = 0.1, "
          "0.03, 0.01, 0.003 is >= 1.8 with adjoint modes, every layout and Pe in the set",
}
S1_TOL, S3_TOL, S4_FACTOR, S4_FLOOR, S6_SLOPE = 1e-10, -1e-10, 10.0, 1e-10, 1.8
PES = (0.0, 20.0, 50.0, 100.0, 200.0)
MS = (4, 8, 16, 32)
EPS = (0.1, 0.03, 0.01, 0.003)
G6_PES = (0.0, 20.0, 200.0)
CHAIN_K = (1, 2, 4, 8)
DT = 1.0
LAYOUTS = (("channel", {"K": 4}), ("plate", {"K": 3}), ("bend", {"n_r": 2, "n_th": 3}))


def _n_for(pe: float, choices=(32, 64, 128)) -> int:
    for n in choices:
        if pe / n <= tr.CELL_PE_MAX:
            return n
    raise ValueError(f"no grid admits Pe {pe}")


def _select(r: tr.TransportResponse, n: int, m_from: int, m: int) -> tr.TransportResponse:
    """The responses for m modes per side, taken from those for m_from >= m: the
    sine-hat basis for m is a subset of the columns for m_from (`ports.basis`)."""
    cols = list(range(4))
    for s in range(4):
        base = 4 + s * (m_from - 1)
        cols += [base + j - 1 for j in range(1, m)]
    return tr.TransportResponse(r.H[:, cols], None if r.G is None else r.G[:, cols], r.up)


def _smooth_fields(n: int, count: int, rng) -> np.ndarray:
    """``count`` smooth random interior fields ((n-1)^2 each), unit norm."""
    t = np.arange(1, n) / n
    XI, ETA = np.meshgrid(t, t)
    out = np.zeros(((n - 1) ** 2, count))
    for k in range(count):
        f = np.zeros_like(XI)
        for a in range(1, 6):
            for b in range(1, 6):
                f += (rng.standard_normal() * math.exp(-(a * a + b * b) / 10.0)
                      * np.sin(a * math.pi * XI) * np.sin(b * math.pi * ETA))
        v = f.ravel()
        out[:, k] = v / np.linalg.norm(v)
    return out


def _slope(xs, ys) -> float:
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:
        return "unknown"


def _write(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    for k in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:          # OneDrive holds a fresh file for a moment
            time.sleep(0.5 * (k + 1))
    os.replace(tmp, path)


def _bench(name: str, extra: dict, n: int, pe: float, problem: str):
    kw = dict(extra, n=n, pe=pe, problem=problem)
    if problem == "step":
        kw["dt"] = DT
    return tc.build(name, **kw)


def positive_control(name, extra, pe, problem, n) -> dict:
    b = _bench(name, extra, n, pe, problem)
    L = b.layout(m=n)
    mono = monolithic(L)
    cp = L.solve(tr.ClassicalTransportExpert())
    top = max(float(np.max(np.abs(u))) for u in mono.fields)
    err = max(float(np.max(np.abs(u - v))) for u, v in zip(mono.fields, cp.fields))
    bal = th.balance(L, cp.fields, mono)
    return {"layout": name, "pe": pe, "problem": problem, "n": n, "m": n,
            "max_err_over_max": err / top, "balance_rel": bal["relative"],
            "S1": err <= S1_TOL * top}


def truncation(name, extra, pe, problem, n) -> dict:
    """S2-S4 on one layout at one Pe: exact responses at m = max(MS), then each m."""
    b = _bench(name, extra, n, pe, problem)
    m_top = max(MS)
    Ltop = b.layout(m=m_top)
    mono = monolithic(Ltop)
    exact_top = tr.ClassicalTransportExpert().respond(b.problems, Ltop.pb)
    rows = []
    for m in MS:
        L = b.layout(m=m)
        resp = [_select(r, p.n, m_top, m) for r, p in zip(exact_top, b.problems)]
        cp = L.couple(resp)
        er = cp.errors(mono.fields)
        c_or = L.trace_coefficients(mono.fields)
        orc = L.reconstruct(resp, c_or)
        cp_or = th.TransportCoupled(L, c_or, cp.S, cp.chi, cp.free, orc, [], [], resp, None)
        eo = cp_or.errors(mono.fields)
        ratio = (er["energy_rel"] / eo["energy_rel"]) if eo["energy_rel"] > S4_FLOOR else None
        rows.append({"m": m, "energy_rel": er["energy_rel"], "max_rel": er["max_rel"],
                     "outlet_bulk_err_rel": er["outlet_bulk_err_rel"],
                     "oracle_energy_rel": eo["energy_rel"], "oracle_max_rel": eo["max_rel"],
                     "ratio": ratio, "sym_min_eig": cp.sym_min_eig(),
                     "balance_rel": th.balance(L, cp.fields, mono)["relative"]})
    # S3 and the splits at m = 16
    pas = {}
    same = None
    resp16 = [_select(r, p.n, m_top, 16) for r, p in zip(exact_top, b.problems)]
    ref16 = None
    for split in th.SPLITS:
        L = b.layout(m=16, split=split)
        pas[split] = min(L.piece_passivity(i, resp16[i]) for i in range(L.P))
        cp = L.couple(resp16)
        if ref16 is None:
            ref16 = cp
        else:
            d = max(float(np.max(np.abs(u - v))) for u, v in zip(cp.fields, ref16.fields))
            same = max(same or 0.0, d / max(float(np.max(np.abs(v))) for v in ref16.fields))
    r16 = next(r for r in rows if r["m"] == 16)
    pe_pieces = [p.peclet() for p in b.problems]
    out = {"layout": name, "pe": pe, "problem": problem, "n": n,
           "pe_pieces_max": max(pe_pieces), "cell_pe_max": max(p.cell_peclet_max() for p in b.problems),
           "rows": rows, "passivity": pas, "splits_max_diff_rel": same,
           "S2": all(r["sym_min_eig"] > 0.0 for r in rows),
           "S3": pas["skew"] >= S3_TOL and pas["upwind"] >= S3_TOL,
           "S4": (r16["ratio"] is None) or (r16["ratio"] <= S4_FACTOR),
           "S4_ratio_m16": r16["ratio"]}
    return out


def chain(K: int, n: int, pe: float) -> dict:
    out = truncation("channel", {"K": K}, pe, "steady", n)
    r16 = next(r for r in out["rows"] if r["m"] == 16)
    return {"K": K, "pe": pe, "n": n, "energy_rel": r16["energy_rel"],
            "oracle_energy_rel": r16["oracle_energy_rel"], "ratio": r16["ratio"],
            "max_rel": r16["max_rel"], "outlet_bulk_err_rel": r16["outlet_bulk_err_rel"],
            "S5": out["S4"]}


def g0_prime(name, extra, pe, n, m=16, seed=20261118) -> dict:
    b = _bench(name, extra, n, pe, "steady")
    L = b.layout(m=m)
    exact = tr.ClassicalTransportExpert().respond(b.problems, L.pb)
    base = L.couple(exact)
    S0, c0 = base.S, base.c
    rng = np.random.default_rng(seed)
    fields = {}
    pg, gal, cpg, cgal = [], [], [], []
    for eps in EPS:
        pert = []
        for i, r in enumerate(exact):
            if i not in fields:
                k = r.H.shape[1]
                fields[i] = (_smooth_fields(b.problems[i].n, k, rng),
                             _smooth_fields(b.problems[i].n, k, rng))
            EV, EW = fields[i]
            sH = np.linalg.norm(r.H, axis=0)
            sG = np.linalg.norm(r.G, axis=0)
            pert.append(tr.TransportResponse(r.H + eps * EV * sH, r.G + eps * EW * sG, r.up))
        cpp = L.couple(pert)
        pg.append(float(np.linalg.norm(cpp.S - S0)))
        cpg.append(float(np.linalg.norm(cpp.c - c0) / np.linalg.norm(c0)))
        gl = L.couple([tr.TransportResponse(r.H, None, r.up) for r in pert])
        gal.append(float(np.linalg.norm(gl.S - S0)))
        cgal.append(float(np.linalg.norm(gl.c - c0) / np.linalg.norm(c0)))
    s_pg, s_g = _slope(EPS, pg), _slope(EPS, gal)
    return {"layout": name, "pe": pe, "n": n, "m": m, "eps": list(EPS),
            "S_err_pg": pg, "S_err_galerkin": gal, "c_err_pg": cpg, "c_err_galerkin": cgal,
            "slope_pg": s_pg, "slope_galerkin": s_g,
            "slope_c_pg": _slope(EPS, cpg), "slope_c_galerkin": _slope(EPS, cgal),
            "S6": s_pg >= S6_SLOPE}


def run(quick: bool, out_dir: str) -> dict:
    t0 = time.time()
    pes = (0.0, 20.0, 50.0) if quick else PES
    n_trunc = 32 if quick else 128
    rec = {"what": "CO-T Phase 1: seam stability up to Pe 200 (the coolant plan)",
           "written": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "commit": _commit(), "quick": quick, "criteria": CRITERIA,
           "model": "(a) side walls only (the owner, 2026-11-18)",
           "positive_control": [], "truncation": [], "chain": [], "g0_prime": []}
    for name, extra in LAYOUTS:
        for problem in ("steady", "step"):
            for pe in pes:
                n = _n_for(pe, (16, 32) if quick else (32, 64, 128))
                rec["positive_control"].append(positive_control(name, extra, pe, problem, n))
                print(f"S1 {name} {problem} Pe {pe:g} n {n}: "
                      f"{rec['positive_control'][-1]['max_err_over_max']:.2e}", flush=True)
                rec["truncation"].append(truncation(name, extra, pe, problem, n_trunc))
                t = rec["truncation"][-1]
                print(f"   m=16 energy {t['rows'][2]['energy_rel']:.3e} oracle "
                      f"{t['rows'][2]['oracle_energy_rel']:.3e} ratio {t['S4_ratio_m16']} "
                      f"passivity {t['passivity']}", flush=True)
    for K in ((1, 2) if quick else CHAIN_K):
        pe = 50.0 if quick else 200.0
        rec["chain"].append(chain(K, 32 if quick else 128, pe))
        print(f"S5 chain K {K}: {rec['chain'][-1]}", flush=True)
    for name, extra in LAYOUTS[:2]:
        for pe in ((0.0, 20.0) if quick else G6_PES):
            n = 32 if quick else 128
            rec["g0_prime"].append(g0_prime(name, extra, pe, n))
            g = rec["g0_prime"][-1]
            print(f"S6 {name} Pe {pe:g}: slope pg {g['slope_pg']:.3f} galerkin "
                  f"{g['slope_galerkin']:.3f}", flush=True)
    verdict = {}
    for key in ("S1",):
        verdict[key] = all(r[key] for r in rec["positive_control"])
    for key in ("S2", "S3", "S4"):
        verdict[key] = all(r[key] for r in rec["truncation"])
    verdict["S5"] = all(r["S5"] for r in rec["chain"])
    verdict["S6"] = all(r["S6"] for r in rec["g0_prime"])
    at200 = {}
    for key in ("S1",):
        at200[key] = all(r[key] for r in rec["positive_control"] if r["pe"] == 200.0)
    for key in ("S2", "S3", "S4"):
        at200[key] = all(r[key] for r in rec["truncation"] if r["pe"] == 200.0)
    at200["S5"] = verdict["S5"]
    at200["S6"] = all(r["S6"] for r in rec["g0_prime"] if r["pe"] == 200.0)
    rec["verdict"] = verdict
    rec["verdict_at_pe_200"] = at200
    rec["stable_at_200"] = (None if quick else all(at200.values()))
    rec["seconds"] = round(time.time() - t0, 1)
    os.makedirs(out_dir, exist_ok=True)
    stem = "phase1-seams-quick" if quick else "phase1-seams"
    _write(os.path.join(out_dir, stem + ".json"), json.dumps(rec, indent=1, default=float) + "\n")
    _write(os.path.join(out_dir, stem + ".txt"), report(rec))
    return rec


def _f(x, fmt=".3g"):
    return "-" if x is None else format(x, fmt)


def report(rec: dict) -> str:
    L = [rec["what"], "", f"written  {rec['written']}", f"commit   {rec['commit']}",
         f"model    {rec['model']}", "", "CRITERIA (registered before the first run)"]
    for k, v in rec["criteria"].items():
        L.append(f"  {k}  {v}")
    L += ["", "S1  POSITIVE CONTROL (whole traces against the monolithic solve)",
          "  layout   problem  Pe     n    max|err|/max|u|  balance"]
    for r in rec["positive_control"]:
        L.append(f"  {r['layout']:8s} {r['problem']:7s} {r['pe']:5g}  {r['n']:4d} "
                 f"{r['max_err_over_max']:.2e}         {r['balance_rel']:.1e}  "
                 f"{'ok' if r['S1'] else 'FAIL'}")
    L += ["", "S2-S4  TRUNCATION (exact pieces, m modes per side; ratio = coupled / oracle energy error)"]
    for t in rec["truncation"]:
        L.append(f"  {t['layout']} {t['problem']} Pe {t['pe']:g} (pieces' max Pe "
                 f"{t['pe_pieces_max']:.1f}, cell Pe {t['cell_pe_max']:.2f}), n = {t['n']}: "
                 f"passivity skew {t['passivity']['skew']:.1e} upwind {t['passivity']['upwind']:.1e} "
                 f"THERM-only {t['passivity']['galerkin']:.1e}; splits differ by "
                 f"{_f(t['splits_max_diff_rel'], '.1e')}")
        for r in t["rows"]:
            L.append(f"     m {r['m']:2d}  energy {r['energy_rel']:.3e}  oracle {r['oracle_energy_rel']:.3e}"
                     f"  ratio {_f(r['ratio'], '.3f')}  max {r['max_rel']:.2e}  outlet "
                     f"{_f(r['outlet_bulk_err_rel'], '.1e')}  sym S {r['sym_min_eig']:.1e}  "
                     f"balance {r['balance_rel']:.1e}")
        L.append(f"     S2 {'ok' if t['S2'] else 'FAIL'}  S3 {'ok' if t['S3'] else 'FAIL'}  "
                 f"S4 {'ok' if t['S4'] else 'FAIL'}")
    L += ["", "S5  CHAIN (channel cut into K pieces, m = 16)"]
    for c in rec["chain"]:
        L.append(f"  K {c['K']}  Pe {c['pe']:g}  energy {c['energy_rel']:.3e}  oracle "
                 f"{c['oracle_energy_rel']:.3e}  ratio {_f(c['ratio'], '.3f')}  max {c['max_rel']:.2e}"
                 f"  outlet {_f(c['outlet_bulk_err_rel'], '.1e')}  {'ok' if c['S5'] else 'FAIL'}")
    L += ["", "S6  PETROV-GALERKIN (slopes of the interface matrix's error against eps)"]
    for g in rec["g0_prime"]:
        L.append(f"  {g['layout']} Pe {g['pe']:g} n {g['n']}: adjoint test {g['slope_pg']:.3f} "
                 f"(coefficients {g['slope_c_pg']:.3f}); trial test {g['slope_galerkin']:.3f} "
                 f"(coefficients {g['slope_c_galerkin']:.3f})  {'ok' if g['S6'] else 'FAIL'}")
    L += ["", "VERDICT", "  every Pe:  " + "  ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in rec["verdict"].items()),
          "  at Pe 200: " + "  ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in rec["verdict_at_pe_200"].items()),
          f"  stable at 200: {rec['stable_at_200']}", f"  {rec['seconds']} s", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "chartop", "cot"))
    a = ap.parse_args(argv)
    rec = run(a.quick, a.out)
    print("stable at 200:", rec["stable_at_200"], "| verdict", rec["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
