"""Stage 0 (W355): what the output form asks of the network, on validation seeds.

    python scripts/chartop_diagnose.py [--count 24]

Two diagnostics, classical solves only, on the first examples of the validation split
(``SEED_BASE['val'] + 100 + i``), never the test split:

1. **Where e_H lives.**  The plain square's principal part ``H0`` against the exact modes:
   the share of ``sum_k ||H0_k - H_k||^2_{A_I}`` (the numerator of e_H^2) carried by the
   corner hats and by the side sines grouped by frequency.  e_H weights each mode by its
   energy, which grows with the sine's frequency.
2. **What the bubble form asks.**  The correction the bubble-weighted head must output,
   ``N = (H - H0) / beta`` with ``beta = 16 xi (1 - xi) eta (1 - eta)``: its root mean square
   by mode group, its maximum, the distance of the maximum from the nearest corner, and
   how the maximum grows with the resolution (n = 32, 64, 128 on the same pieces).  Under a
   shearing chart the exact mode near a corner does not vanish like ``xi eta`` while the
   bubble does, so the quotient is large there.

Writes ``out/chartop/diagnose-<time>.json``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from atlas.chartop import data, fe, model as md, ports  # noqa: E402

GROUPS = {"hats": slice(0, 4), "sines j=1-3": (0, 3), "sines j=4-8": (3, 8), "sines j=9-15": (8, 15)}


def group_values(v: np.ndarray) -> dict:
    """v: per-column values (64,) -> summed by group."""
    s = v[4:].reshape(4, 15)
    return {"hats": float(v[:4].sum()), "sines j=1-3": float(s[:, :3].sum()),
            "sines j=4-8": float(s[:, 3:8].sum()), "sines j=9-15": float(s[:, 8:].sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "chartop"))
    args = ap.parse_args()
    rec = {"written": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "seeds": [data.SEEDS["val"] + 100, data.SEEDS["val"] + 100 + args.count - 1],
           "share_of_eH2": [], "bubble": {}}
    shares = []
    for i in range(args.count):
        e = data.draw(data.SEEDS["val"] + 100 + i, n=64)
        a = data.arrays(e)
        Q = ports.basis("sine-hat", 64, 16).Q
        lab = data.labels_from_arrays(a, Q)
        ps = md.plain_square(64, "sine-hat", 16, "cpu")
        H0 = ps.principal(torch.tensor([float(a["s0"])], dtype=torch.float64))[0].numpy()
        E = lab["H"] - H0.reshape(64, -1).T
        M = fe.assemble(64, a["a11"].astype(float), a["a12"].astype(float), a["a22"].astype(float),
                        a["sigma"].astype(float)).tocsr()
        inner = fe.grid(64)[1]
        A_I = M[inner][:, inner]
        ek = np.einsum("ik,ik->k", E, A_I @ E)
        g = group_values(ek / ek.sum())
        shares.append(g)
        rec["share_of_eH2"].append({"seed": e.seed, "e_H_principal": float(np.sqrt(ek.sum() / lab["trace_Lambda"])),
                                    "sup_mu": e.envelope["sup_abs_mu"], **g})
    rec["share_of_eH2_median"] = {k: float(np.median([s[k] for s in shares])) for k in shares[0]}
    # the bubble's quotient on the same pieces at three resolutions
    for n in (32, 64, 128):
        ps = md.plain_square(n, "sine-hat", 16, "cpu")
        beta = ps.beta[1:-1, 1:-1].numpy()
        t = np.arange(1, n) / n
        XI, ET = np.meshgrid(t, t)
        dcorner = np.minimum.reduce([np.hypot(XI - x, ET - y) for x in (0, 1) for y in (0, 1)])
        Q = ports.basis("sine-hat", n, 16).Q
        rows = []
        for i in range(min(8, args.count)):
            e = data.draw(data.SEEDS["val"] + 100 + i, n=n)
            a = data.arrays(e)
            lab = data.labels_from_arrays(a, Q)
            H = lab["H"].T.reshape(64, n - 1, n - 1)
            H0 = ps.principal(torch.tensor([float(a["s0"])], dtype=torch.float64))[0].numpy()
            N = (H - H0) / beta[None]
            k = np.unravel_index(int(np.argmax(np.abs(N))), N.shape)
            rows.append({"seed": e.seed, "rms_by_group": group_values(np.sqrt(np.mean(N ** 2, axis=(1, 2)))),
                         "max_abs": float(np.abs(N).max()),
                         "max_at_corner_distance": float(dcorner[k[1], k[2]]),
                         "max_mode": int(k[0])})
        rec["bubble"][str(n)] = {"examples": rows,
                                 "max_abs_median": float(np.median([r["max_abs"] for r in rows])),
                                 "max_at_corner_distance_median": float(np.median([r["max_at_corner_distance"]
                                                                                   for r in rows]))}
        print(f"n={n}: median max |N| {rec['bubble'][str(n)]['max_abs_median']:.3g} at a median "
              f"{rec['bubble'][str(n)]['max_at_corner_distance_median']:.3g} from a corner", flush=True)
    print("share of e_H^2 (median):", {k: round(v, 3) for k, v in rec["share_of_eH2_median"].items()})
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "diagnose-%s.json" % dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
