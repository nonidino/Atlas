"""W112 stage G -- march the search's answer as far as the thing it claims to fix.

    python scripts/w112_horizon.py [--out out/w112] [--steps 32]

`scripts/w112_seam_placement.py` reports a gate that is passed at **one
exchange interval**: on CS-7's N=6 geometry the search finds a decomposition at
$0.2685\\times$ the hand-chosen tiling's composed defect, and the hand-chosen
tiling ranks 138th of 178.  The same run's stage E then reports that at **eight**
exchange intervals the same cut is only $0.9455\\times$ -- and that the
eight-interval best available, which no criterion picked, is $0.51\\times$.

So the one-interval advantage is largely gone by eight intervals, and the
obvious question is whether eight is itself long enough to say so.  **This
vault has been wrong about exactly that twice** -- [[tier0-measurements]] 18.5's
published state was on a diverging trajectory ten to twenty macro-steps short
of where it would have shown, and 19.6's repair was stable exactly as far as
its positive control was marched and died at 38 from the developed state.  A
third instance would be a pattern nobody could call an accident.

So: take three decompositions -- the hand-chosen tiling, the one-interval
winner, and the eight-interval winner -- and march all three against the
monolith for as long as the rollout stays finite, reporting the ratio at every
step rather than at one.

**Scope, stated because it bounds every number here.** Both columns are
`exposed` agents with **no projection anywhere** -- neither inside a window nor
after the assembly.  That makes the comparison clean, because ``E`` and
``E_i`` are then the same operator and the difference is the restriction and
nothing else, which is what a placement criterion is being graded on.  It also
means this is **not an incompressible rollout**: it is `case-study-scaling-
ladder-atlas-0.1` 5's fourth row, the all-exposed graph that runs without ever
applying the elliptic part, which is **W105**.  A production rollout applies one
global projection after the blend, and whether the placement advantage survives
THAT is a different measurement and is not this one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from atlas.cases import seam_placement as sp                       # noqa: E402
from atlas.cases import wake_array as wa                           # noqa: E402


def say(msg: str) -> None:
    print(msg, flush=True)


def parse(label: str) -> sp.RectDecomposition:
    """``3x2|x53-171|y168|h8`` -> the decomposition it names."""
    _shape, xs, ys, h = label.split("|")
    def cuts(part):
        body = part[1:]
        return () if body == "none" else tuple(int(v) for v in body.split("-"))
    t = wa.ArrayTiling(n_col=3, n_row=2)
    return sp.RectDecomposition(nx=t.nx, ny=t.ny, x_cuts=cuts(xs),
                                y_cuts=cuts(ys), halo=int(h[1:]))


def march(dec, u, v, dt, n_steps, nu=wa.NU_REF):
    """Composed and monolith side by side, reporting the ratio at EVERY step.

    `run-artifacts-and-warnings`' own rule: the trajectory is serialized, not
    just its endpoints, because writing only the last value after every step
    keeps no history at all -- and the whole question here is the shape of the
    curve rather than its end.
    """
    cu, cv = u.copy(), v.copy()
    mu, mv = u.copy(), v.copy()
    rows = []
    for k in range(1, n_steps + 1):
        lu, lv, _ = sp.local_steps(dec, cu, cv, dt, nu)
        cu, cv = dec.assemble(lu), dec.assemble(lv)
        mu, mv, _ = sp.monolith_step(dec.nx, dec.ny, mu, mv, dt, nu)
        num = float(np.sqrt(np.sum((cu - mu) ** 2) + np.sum((cv - mv) ** 2)))
        den = float(np.sqrt(np.sum(mu ** 2) + np.sum(mv ** 2)))
        rows.append({
            "step": k, "defect": num / max(den, 1e-300),
            "u_max_composed": float(np.max(np.hypot(cu, cv))),
            "u_max_monolith": float(np.max(np.hypot(mu, mv))),
            "div_rms_composed": wa.divergence_rms(cu, cv),
            "finite": bool(np.all(np.isfinite(cu)) and np.all(np.isfinite(cv))),
        })
        if not rows[-1]["finite"]:
            say(f"      NOT FINITE at step {k} -- stopping")
            break
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_ROOT, "out", "w112"))
    ap.add_argument("--steps", type=int, default=32)
    args = ap.parse_args()

    with open(os.path.join(args.out, "w112.json"), encoding="utf-8") as fh:
        rep = json.load(fh)
    g = rep["geometries"]["N6"]
    by = {s["decomposition"]["label"]: s for s in g["scores"]}
    pool = [l for l in g["pools"]["PLACE"]["labels"] if l in by]
    hand = g["hand_chosen"]["label"]
    win1 = min(pool, key=lambda l: by[l]["defect"]["defect"])
    win8 = min(pool, key=lambda l: by[l]["marched"]["defect_8_intervals"])

    d6 = np.load(os.path.join(_ROOT, sp.HAND_CHOSEN["N6"]["state"]))
    u, v = d6["reference_all_u"], d6["reference_all_v"]
    dt = sp.exchange_interval(u, v)
    say(f"marching {args.steps} exchange intervals of dt = {dt:.6e} "
        f"({args.steps * dt:.4f} time units = "
        f"{args.steps * dt / wa.MACRO_DT:.3f} macro-steps)")

    tracks = {}
    for tag, label in (("hand_chosen", hand),
                       ("one_interval_winner", win1),
                       ("eight_interval_winner", win8)):
        say(f"  {tag:22s} {label}")
        t0 = time.perf_counter()
        tracks[tag] = {"label": label, "rows": march(parse(label), u, v, dt,
                                                     args.steps)}
        say(f"      {len(tracks[tag]['rows'])} steps in "
            f"{time.perf_counter() - t0:.0f} s")

    # the ratio, step by step: the number the case study's gate is about
    h = {r["step"]: r["defect"] for r in tracks["hand_chosen"]["rows"]}
    for tag in ("one_interval_winner", "eight_interval_winner"):
        for r in tracks[tag]["rows"]:
            r["over_hand_chosen"] = r["defect"] / max(h.get(r["step"], np.nan), 1e-300)

    out = {
        "row": "W112", "stage": "G -- the horizon",
        "steps": args.steps, "dt": dt,
        "macro_steps_marched": args.steps * dt / wa.MACRO_DT,
        "scope": ("both columns are exposed agents with NO projection anywhere; "
                  "this is CS-7 5's fourth row (W105), stable and not "
                  "incompressible. A production rollout applies one global "
                  "projection after the blend and this does not measure that"),
        "tracks": tracks,
    }
    for tag in ("one_interval_winner", "eight_interval_winner"):
        rs = tracks[tag]["rows"]
        out[tag + "_ratio"] = {
            "at_1": rs[0]["over_hand_chosen"],
            "at_8": next((r["over_hand_chosen"] for r in rs if r["step"] == 8), None),
            "at_16": next((r["over_hand_chosen"] for r in rs if r["step"] == 16), None),
            "at_last": rs[-1]["over_hand_chosen"],
            "last_step": rs[-1]["step"],
            "min_ratio": min(r["over_hand_chosen"] for r in rs),
            "max_ratio": max(r["over_hand_chosen"] for r in rs),
        }
    path = os.path.join(args.out, "horizon.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    say(f"\nwrote {path}")

    say("\n" + "=" * 70)
    say(f"{'step':>5} {'hand':>12} {'1-int winner':>12} {'ratio':>8} "
        f"{'8-int winner':>12} {'ratio':>8}")
    a = {r["step"]: r for r in tracks["one_interval_winner"]["rows"]}
    b = {r["step"]: r for r in tracks["eight_interval_winner"]["rows"]}
    for r in tracks["hand_chosen"]["rows"]:
        k = r["step"]
        if k in (1, 2, 4, 8, 12, 16, 20, 24, 28, 32) or k == args.steps:
            say(f"{k:5d} {r['defect']:12.4e} {a[k]['defect']:12.4e} "
                f"{a[k]['over_hand_chosen']:8.4f} {b[k]['defect']:12.4e} "
                f"{b[k]['over_hand_chosen']:8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
