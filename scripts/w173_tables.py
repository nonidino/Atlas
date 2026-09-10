"""W173 -- the tables the write-up quotes, derived from the artifacts.

Nothing is measured here.  This reads `out/w173/w173.json`,
`out/w173/seam_operators.npz` and `out/w173/transverse.json` and prints the
tables that go into the wiki page, so that no number in the page is typed by
hand.  Run it and paste; if a number in the page and a number here disagree, the
page is wrong.
"""

from __future__ import annotations

import io
import json
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "out", "w173")


def load(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def at(rows, r, key):
    return next((x[key] for x in rows if x["r"] == r), None)


def main():
    d = load("w173.json")
    npz_path = os.path.join(OUT, "seam_operators.npz")
    ops = np.load(npz_path) if os.path.exists(npz_path) else {}

    print("#" * 78)
    print("# TABLE 1 -- the band truncation of the seam operator")
    print("#" * 78)
    print("%-38s %-9s %9s %9s %9s %9s %9s"
          % ("subject", "||S||2", "rel(2)", "rel(8)", "rel(20)", "rel(64)",
             "slope/cell"))
    for s in d["subjects"]:
        rows = s["linear"]["rows"]
        ts = s.get("terminal_slope_2norm") or {}
        print("%-38s %-9.4g %9.4f %9.4f %9.4f %9.4f %9.2e"
              % (s["label"][:38], s["linear"]["norm2"],
                 at(rows, 2, "E2_rel"), at(rows, 8, "E2_rel"),
                 at(rows, 20, "E2_rel"), at(rows, 64, "E2_rel"),
                 ts.get("slope_per_cell", float("nan"))))

    print()
    print("#" * 78)
    print("# TABLE 2 -- r*, the radius at which the truncation reaches a target")
    print("#          n = 128, so r = 127 is 'keep the whole operator' and is")
    print("#          marked trivial: it is arithmetic, not decay")
    print("#" * 78)
    print("%-38s %s" % ("subject", "  ".join(
        "rel<%.0e" % t for t in d["targets"]["relative"])))
    for s in d["subjects"]:
        cells = []
        for x in s["r_star_relative_2norm"]:
            if x["r_star"] is None:
                cells.append("  never")
            elif x["trivial"]:
                cells.append("  (127)")
            else:
                cells.append("%7d" % x["r_star"])
        print("%-38s %s" % (s["label"][:38], " ".join(cells)))
    print()
    print("%-38s %s" % ("subject  [ABSOLUTE, 2-norm]", "  ".join(
        "<%.3g" % t for t in d["targets"]["absolute"])))
    for s in d["subjects"]:
        cells = []
        for x in s["r_star_absolute_2norm"]:
            if x["r_star"] is None:
                cells.append("   never")
            elif x["trivial"]:
                cells.append("   (127)")
            else:
                cells.append("%8d" % x["r_star"])
        print("%-38s %s" % (s["label"][:38], " ".join(cells)))

    print()
    print("# TABLE 2b -- the same, under CIRCULAR distance.  The checkpoint was")
    print("#          trained on the periodic unit square and treats the window's")
    print("#          two ends as neighbours, so this is the MOST GENEROUS reading:")
    print("#          it counts a wrap-around coupling as near rather than far.")
    print("#          The trivial radius here is 64, not 127 -- circular distance")
    print("#          on 128 cells saturates at n/2 -- so 64 is 'keep everything'.")
    print("%-38s %s" % ("subject", "  ".join(
        "rel<%.0e" % t for t in d["targets"]["relative"])))
    for s in d["subjects"]:
        n = s["n"]
        cells = []
        for x in s.get("r_star_relative_2norm_circular", []):
            r = x["r_star"]
            cells.append("  never" if r is None
                         else ("%6s" % ("(%d)" % r) if r >= n // 2 else "%7d" % r))
        print("%-38s %s" % (s["label"][:38], " ".join(cells)))

    print()
    print("#" * 78)
    print("# TABLE 3 -- the per-CELL reading, which is what the proposal's words say")
    print("#          mean |S[i,j]| at |i-j| = d over interior columns, and the")
    print("#          row-sum the same kernel implies past r=20")
    print("#" * 78)
    print("%-38s %9s %9s %9s %9s %9s"
          % ("subject", "d=0", "d=4", "d=20", "d=31", "sum d>20/peak"))
    for s in d["subjects"]:
        p = [x for x in s["column_decay"]["profile"]]
        g = np.array([np.nan if x is None else x for x in p], dtype=float)
        peak = np.nanmax(g)
        tail = float(np.nansum(g[21:]))
        print("%-38s %9.3g %9.3g %9.3g %9.3g %9.3g"
              % (s["label"][:38], g[0], g[4], g[20], g[31], tail / peak))

    print()
    print("#" * 78)
    print("# TABLE 4 -- the far-field block's spectrum at r = 20")
    print("#          a plateau can be many small modes or a few coherent ones")
    print("#" * 78)
    print("%-38s %10s %10s %8s" % ("subject", "share_1", "share_top4", "rank99"))
    for s in d["subjects"]:
        fs = next((x for x in s.get("far_field_spectrum", []) if x["r"] == 20), None)
        if fs:
            print("%-38s %10.4f %10.4f %8d"
                  % (s["label"][:38], fs["share_1"], fs["share_top4"],
                     fs["effective_rank_99"]))

    print()
    print("#" * 78)
    print("# TABLE 5 -- sigma_min, and the truncation measured against IT")
    print("#          the bound's own ratio is ||Lambda - Lambda~|| / beta, not")
    print("#          / ||Lambda||.  sigma_min of the full face operator is not")
    print("#          the compiler's beta (16 declared modes, not 128) and is")
    print("#          quoted as a diagnostic of the same shape")
    print("#" * 78)
    if len(ops):
        print("%-38s %10s %10s %10s %10s"
              % ("key", "sigma_min", "sigma_max", "E2(20)/smin", "E2(64)/smin"))
        for k in sorted(ops.files):
            S = ops[k]
            sv = np.linalg.svd(S, compute_uv=False)
            smin, smax = float(sv[-1]), float(sv[0])
            n = S.shape[0]
            i = np.arange(n)[:, None]
            j = np.arange(n)[None, :]
            out = []
            for r in (20, 64):
                T = S * (np.abs(i - j) > r)
                out.append(float(np.linalg.norm(T, 2)) / smin if smin > 0
                           else float("inf"))
            print("%-38s %10.3e %10.3e %10.3e %10.3e"
                  % (k[:38], smin, smax, out[0], out[1]))
    else:
        print("(seam_operators.npz not present)")

    print()
    print("#" * 78)
    print("# TABLE 6 -- is S an operator or the instrument's floor?")
    print("#          a linear response has S(a) = S(a/2) exactly")
    print("#" * 78)
    for lab, lad in (d.get("amplitude_ladder") or {}).items():
        print(lab, " f0_max = %.4g" % lad["f0_max"])
        print("   %-10s %12s %12s %14s" % ("amplitude", "||S_a||F",
                                           "rel vs 1e-2", "cancel floor F"))
        for x in lad["rows"]:
            print("   %-10.0e %12.4g %12.4g %14.4g"
                  % (x["amplitude"], x["normF"], x["rel_disagreement_vs_ref"],
                     x["cancellation_floor_normF"]))

    print()
    print("#" * 78)
    print("# TABLE 7 -- Pi(d): what a halo of radius d would buy in the sigma bound")
    print("#          master-error-bound 4.1: sigma <= C_mu * Pi * ||d_lambda||")
    print("#" * 78)
    for p in d["Pi_of_halo"]:
        print("   d = %3d   Pi = %.4f" % (p["d_cells"], p["contaminated_weight"]))

    tr = load("transverse.json")
    if tr:
        print()
        print("#" * 78)
        print("# TABLE 8 -- the TRANSVERSE reach: R10/halo's own quantity")
        print("#          how far into the window a ring datum travels in one step")
        print("#" * 78)
        print("%-30s %-6s %-4s %6s %8s %8s %8s %8s"
              % ("subject", "a", "j0", "reach", "d*(1e-1)", "d*(1e-2)",
                 "d*(1e-4)", "ell"))
        for s in tr["subjects"]:
            for r in s["rows"]:
                ds = {x["target_relative"]: x for x in r["depth_star_max"]["rows"]}
                f = r.get("decay_fit", {})
                ell = (f.get("exponential") or {}).get("ell_cells")
                print("%-30s %-6.0e %-4d %6d %8s %8s %8s %8s"
                      % (s["label"][:30], r["amplitude"], r["j0"],
                         r["nonzero_reach"],
                         ds[1e-1]["depth_star"], ds[1e-2]["depth_star"],
                         ds[1e-4]["depth_star"],
                         ("%.3g" % ell) if ell else "n/a"))

    w174 = os.path.join(HERE, "out", "w174", "w174.json")
    if os.path.exists(w174):
        with open(w174, encoding="utf-8") as f:
            v = json.load(f)
        print()
        print("#" * 78)
        print("# TABLE 9 -- W174: is the refusal EARNED?")
        print("#" * 78)
        for name, arr in v["arrangements"].items():
            print(name, " declared reach", v["declared_reach"])
            for r in arr["rows"]:
                print("   halo %3d  rule %-7s  contraction %-12s  %s"
                      % (r["halo"],
                         "refuse" if r["halo_rule_refuses"] else "admit",
                         r["contraction"], r["cell"]))
            print("   tally:", arr["tally"])
            print("   threshold:", arr["threshold"])


if __name__ == "__main__":
    main()
