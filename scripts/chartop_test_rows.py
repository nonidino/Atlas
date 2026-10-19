"""Stage 0 (W355), POST HOC: the one evaluation's per-example test rows, broken down.

    python scripts/chartop_test_rows.py --rows out/chartop/evaluation-<time>-test-rows.npz

Not a gate, and nothing is recomputed: it reads the rows `scripts/chartop_evaluate.py`
wrote (per test piece and arm: e_H, e_p, sup|mu|, input frequency, n, seed, Fourier
number; ``fo < 0`` marks a steady piece) and prints the medians and 90th percentiles
of e_H and e_p by Fourier-number decade (S2: does the error grow as the step shrinks?),
by resolution and by sup|mu| bin, for every arm.  Writes
``out/chartop/test-rows-breakdown.json``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from atlas.chartop import gates as G  # noqa: E402

FO_BANDS = (("steady", None, None), ("Fo 0.01-0.1", 1e-2, 1e-1), ("Fo 0.1-1", 1e-1, 1.0),
            ("Fo 1-10", 1.0, 10.0), ("Fo 10-100", 10.0, 100.0 * (1 + 1e-12)))


def stats(e_H: np.ndarray, e_p: np.ndarray) -> dict:
    if len(e_H) == 0:
        return {"count": 0}
    return {"count": int(len(e_H)), "e_H_median": float(np.median(e_H)),
            "e_H_p90": float(np.quantile(e_H, 0.9)), "e_p_median": float(np.median(e_p)),
            "e_p_p90": float(np.quantile(e_p, 0.9))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "chartop", "test-rows-breakdown.json"))
    args = ap.parse_args()
    z = np.load(args.rows, allow_pickle=False)
    arms = sorted({k.split("__")[0] for k in z.files})
    out = {"written": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "post_hoc": "a breakdown of the one evaluation's rows; not a gate",
           "rows": os.path.relpath(args.rows, ROOT), "arms": {}}
    for a in arms:
        e_H, e_p = z[f"{a}__e_H"], z[f"{a}__e_p"]
        fo, n, mu = z[f"{a}__fo"], z[f"{a}__n"], z[f"{a}__sup_mu"]
        rec = {"by_fourier": {}, "by_resolution": {}, "by_mu_bin": {}}
        for name, lo, hi in FO_BANDS:
            sel = fo < 0 if lo is None else (fo >= lo) & (fo < hi)
            rec["by_fourier"][name] = stats(e_H[sel], e_p[sel])
        for r in sorted(set(n.tolist())):
            rec["by_resolution"][str(r)] = stats(e_H[n == r], e_p[n == r])
        for k, (lo, hi) in enumerate(G.MU_BINS):
            sel = np.array([G.mu_bin_of(m) == k for m in mu])
            rec["by_mu_bin"][f"[{lo}, {hi})"] = stats(e_H[sel], e_p[sel])
        out["arms"][a] = rec
        print(a)
        for part, d in rec.items():
            for key, s in d.items():
                if s["count"]:
                    print(f"  {part:14s} {key:12s} n={s['count']:5d}  e_H {s['e_H_median']:.4f} / "
                          f"{s['e_H_p90']:.4f}   e_p {s['e_p_median']:.4f} / {s['e_p_p90']:.4f}")
    path = args.out
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
