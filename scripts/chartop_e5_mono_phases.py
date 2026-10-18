"""Stage 0 (W355): E5's monolithic arm split into its phases, so no ratio flatters a network.

    python scripts/chartop_e5_mono_phases.py

`host.monolithic` merges the pieces' coincident nodes with a Python loop over a k-d
tree, assembles every piece's matrix into one, and factorises.  The merge is this
reference implementation's overhead: a production code numbers its mesh once, or
vectorises the numbering.  On each benchmark problem (n = 64), the median of 5 of:
the node merge, the global assembly (from piece matrices already assembled), the
SuperLU factorisation, and one solve.  Writes ``out/chartop/e5-mono-phases-<host>.json``.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import socket
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from atlas.chartop import benchmarks as bm, data, fe  # noqa: E402
from atlas.chartop import ports as pt  # noqa: E402
from chartop_price import power_state  # noqa: E402


def phases(L):
    t = {}
    t0 = time.perf_counter()
    pts = [p.chart.X.reshape(-1, 2) for p in L.problems]
    allp = np.concatenate(pts)
    tree = cKDTree(allp)
    gid = -np.ones(len(allp), int)
    ng = 0
    for k in range(len(allp)):
        if gid[k] >= 0:
            continue
        for q in tree.query_ball_point(allp[k], L.tol):
            gid[q] = ng
        ng += 1
    offs = np.cumsum([0] + [len(x) for x in pts])
    G = [gid[offs[i]:offs[i + 1]] for i in range(L.P)]
    t["merge_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    rows, cols, vals = [], [], []
    rhs = np.zeros(ng)
    for i, p in enumerate(L.problems):
        M = p._M.tocoo()
        rows.append(G[i][M.row]); cols.append(G[i][M.col]); vals.append(p.kappa_ref * M.data)
        np.add.at(rhs, G[i], p.kappa_ref * p.load())
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(ng, ng))
    dval = {}
    for e in L.edges:
        if e.other is None and e.bc is not None and e.bc[0] == "dirichlet":
            i, s = e.owner
            pos = pt.side_positions(L.problems[i].n, s)
            loop = fe.grid(L.problems[i].n)[0]
            for q, v in zip(loop[pos], e.bc[1]):
                dval[int(G[i][q])] = float(v)
    dn = np.array(sorted(dval), int)
    fn = np.setdiff1d(np.arange(ng), dn)
    ud = np.array([dval[d] for d in dn])
    Kff = K[fn][:, fn].tocsc()
    rr = rhs[fn] - K[fn][:, dn] @ ud
    t["assemble_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    lu = spla.splu(Kff)
    t["factor_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    lu.solve(rr)
    t["solve_s"] = time.perf_counter() - t0
    t["unknowns"] = int(len(fn))
    return t


def main() -> int:
    rec = {"written": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "host": socket.gethostname(), "power": power_state(), "rows": []}
    for name in bm.NAMES:
        for problem in ("steady", "step"):
            kw = {"envelope": data.ENVELOPE} if name == "s-channel" else {}
            b = bm.build(name, 64, problem, **kw)
            L = b.layout("sine-hat", 16)
            for p in L.problems:
                p._M = p.matrix()
            runs = [phases(L) for _ in range(5)]
            med = {k: statistics.median(r[k] for r in runs) for k in runs[0] if k.endswith("_s")}
            med["unknowns"] = runs[0]["unknowns"]
            med.update(name=name, problem=problem)
            print(json.dumps(med), flush=True)
            rec["rows"].append(med)
    path = os.path.join(ROOT, "out", "chartop", "e5-mono-phases-%s.json" % socket.gethostname()[:24])
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
