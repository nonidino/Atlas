"""Stage 0 (W355): validate the tier-0 host with EXACT classical modes, before any network.

    python scripts/chartop_validate_host.py [--n 64]

On the three held-out benchmarks (`atlas/chartop/benchmarks.py`), steady and one step:

1. every chart's certificate (Proposition 3) and envelope quantities, and the chart rule's
   choice for the s-channel's end pieces (the conformal chart tried, refused, Winslow used);
2. the host with **all** port modes (m = n: hats and every side sine) against the
   full-domain solve in the same discretisation -- they must agree to rounding;
3. the host with 16 modes per side: the port-truncation floor, in the energy norm and in
   the maximum nodal error over the range (gate SE3's ``||u - u_m||_M``);
4. the analytic solution where one exists (bend-3: ``T = 400 - 100 theta / (pi/2)``) and the
   fine harmonic coordinate for the s-channel (``T = 400 - 100 u``).

Writes ``out/chartop/host_validation.json``.  Classical solves only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from atlas.chartop import benchmarks as bm, certificate as cert, chart as ch  # noqa: E402
from atlas.chartop import data, host as hs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "chartop", "host_validation.json"))
    args = ap.parse_args()
    n = args.n
    rec = {"written": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "n": n,
           "envelope": data.ENVELOPE.as_dict(), "benchmarks": []}
    for name in bm.NAMES:
        for problem in ("steady", "step"):
            t0 = time.perf_counter()
            kw = {"envelope": data.ENVELOPE} if name == "s-channel" else {}
            b = bm.build(name, n, problem, **kw)
            r = {"name": name, "problem": problem, "pieces": len(b.problems)}
            if problem == "steady":
                rows = []
                for p in b.problems:
                    c = cert.certify(p.chart.X)
                    m = cert.measure(p.chart, p.log_kappa_hat, p.fo)
                    rows.append({"label": p.label, "method": p.chart.method, "certified": c.ok,
                                 "sup_abs_mu": m["sup_abs_mu"], "logJ_range": m["logJ_range"],
                                 "s_grad": m["s_grad"],
                                 "corner_angles_deg": [round(float(a), 2) for a in ch.corner_angles(p.chart.X)],
                                 "refusals": cert.refusals(m, data.ENVELOPE)})
                r["charts"] = rows
                if name == "s-channel":
                    r["modulus"] = b.meta["modulus"]
                    r["duality_Eu_Ev"] = b.meta["duality_Eu_Ev"]
                    r["end_piece_conformal_logJ_range"] = [x["conformal"]["logJ_range"]
                                                           for x in b.meta["end_piece_charts"]]
            else:
                r["fo"] = [p.fo for p in b.problems]
            L16 = b.layout("sine-hat", 16)
            r["layout"] = {"vertices": L16.nv, "edges": len(L16.edges), "unknowns_m16": L16.N,
                           "conforming": all(e.conforming for e in L16.edges)}
            mono = hs.monolithic(L16)
            span = (max(float(np.max(u)) for u in mono.fields) - min(float(np.min(u)) for u in mono.fields))
            for m_, key in ((n, "all_modes"), (16, "m16")):
                L = b.layout("sine-hat", m_)
                cp = L.solve(hs.ClassicalExpert())
                num, den = cp.energy_error(mono.fields)
                mx = max(float(np.max(np.abs(u - v))) for u, v in zip(mono.fields, cp.fields))
                r[key] = {"energy_rel": num / den, "energy": num, "max_over_range": mx / span,
                          "lambda_min": cp.min_eig(), "calls": cp.calls}
            if name == "bend-3" and problem == "steady":
                ex = b.meta["exact"]
                r["monolithic_vs_analytic_max_K"] = max(
                    float(np.max(np.abs(u - ex(p.chart.X[..., 0].ravel(), p.chart.X[..., 1].ravel()))))
                    for p, u in zip(b.problems, mono.fields))
            if name == "s-channel" and problem == "steady":
                from scipy.interpolate import griddata
                X, U = b.meta["harmonic_u"]
                err = 0.0
                for p, u in zip(b.problems, mono.fields):
                    Tref = 400.0 - 100.0 * griddata(X.reshape(-1, 2), U.ravel(),
                                                    p.chart.X.reshape(-1, 2), method="linear")
                    err = max(err, float(np.nanmax(np.abs(u - Tref))))
                r["monolithic_vs_fine_harmonic_max_K"] = err
            r["span_K"] = span
            r["elapsed_s"] = time.perf_counter() - t0
            print(f"{name} {problem}: all modes {r['all_modes']['energy_rel']:.2e}, "
                  f"16 modes {r['m16']['energy_rel']:.3e} (max {r['m16']['max_over_range']:.2e} of range)",
                  flush=True)
            rec["benchmarks"].append(r)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
