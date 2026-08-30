"""W7's remaining half: the two-rate test R9 has been owed since 2026-08-27.

    python scripts/w7_multirate_matching.py [--out out/w7m] [--stages ...]

`w7_time_integrated_interface.py` built the time-integrated condition for the
SINGLE-rate case and it failed both of §5's tests, because §5's defect was
**spatial** -- at a shared-layer seam ``F_A + F_B = -nu h d2w/dn2`` exactly, and
integrating in time cannot touch a second derivative in space.  That retraction
ended with a sentence this script exists to discharge:

    "R9's time-integration remains right for what R9 is about -- MULTIRATE
     conservation, where two clocks genuinely need a common integral -- and the
     two-rate test is still owed."

The worklist states the definition of done as *"a two-rate test where pointwise
matching demonstrably leaks and integrated matching does not"*.  Leaks is the
operative word: the claim is a **conservation** claim, not an accuracy one, so the
measurement is a closed energy balance over one exchange interval and not a
residual norm.

The target is `thermal_seam` at ``clocks="native"``, whose two agents are
genuinely 500:1 apart -- 0.1 ms of compressible gas against a 50 ms backward-Euler
shell -- and which still compiles to `refuse` on E4.

Stages:
    leak      what a pointwise match across two clocks actually loses
    match     the time-integrated condition, against the pointwise one
    verdict   what declaring it moves on the compile
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from atlas.cases import thermal_seam as T                            # noqa: E402

STAGES = ("leak", "lagvsleak", "match", "verdict")

#: The interface state to hold the gas at while its transient is recorded. The
#: tightly coupled trace at the matched clock, from Tier 16 -- so the transient
#: measured here is the one a converged coupling would actually see, not one
#: manufactured by starting somewhere arbitrary.
LAMBDA_STAR = 372.1377

#: The ratios sliced out of the one recorded march. 1 is the single-rate control
#: and must show no leak at all; 500 is `thermal_seam`'s own declared mismatch.
RATIOS = (1, 2, 5, 10, 25, 50, 100, 250, 500)


def _march(gas, n_steps, lam):
    """March the gas ``n_steps`` of its OWN clock at a held wall temperature.

    Returns the per-sub-step wall heat flux, one row per gas step, which is the
    agent's own sub-step quadrature of the interface flux.  Recorded once and
    sliced, because every ratio below 500 is a prefix of the same march -- the
    whole sweep costs one march rather than nine.
    """
    U = gas._U0.copy()
    q_rows, f_rows = [], []
    for _ in range(n_steps):
        U, _ = gas._solver(lam).advance(U, gas.dt)
        q, T_face = gas._wall_heat_flux(U, lam)
        q_rows.append(np.asarray(q, float).copy())
        # the DECLARED flow, q_n/T -- computed with the agent's own conversion so
        # the recorded history and `respond` cannot disagree about the convention
        f_rows.append(np.asarray(
            T._as_flow(q, T_face, gas.flux_convention), float).copy())
    return np.array(q_rows), np.array(f_rows)


def stage_leak(out: dict) -> None:
    print("=== W7: what a POINTWISE match across two clocks loses ===")
    print("  The conservation statement over one exchange interval is")
    print("      int_Omega (u^{n+1} - u^n)  =  - int_t^{t+DT} oint F.n")
    print("  so the quantity the two sides must agree on is the TIME-INTEGRAL of the")
    print("  flux. A pointwise match hands the slow side one instantaneous value and")
    print("  multiplies it by the whole interval. Whether that matters is a fact about")
    print("  the fast side's transient, and it has never been measured here.")
    print()
    gas = T.GasAgent(dt=T.DT_GAS)
    lam = np.full(T.N_SEAM, LAMBDA_STAR)
    n = max(RATIOS)
    t0 = time.time()
    q, _flow = _march(gas, n, lam)
    print(f"  marched {n} gas steps of {T.DT_GAS:g} s in {time.time() - t0:.1f} s "
          f"({n * T.DT_GAS:g} s of physical time, the shell's own step)", flush=True)
    print()

    # Per-cell energy per unit area over the interval, both ways.
    print(f"  {'ratio':>7}{'DT [s]':>11}{'integrated':>14}{'pointwise':>14}"
          f"{'leak':>12}{'|leak| rel':>12}")
    rows = []
    for r in RATIOS:
        window = q[:r]
        dt_sub = T.DT_GAS
        # Each side integrates with its OWN quadrature: the gas has r sub-steps,
        # the shell has one, and a pointwise match is the shell's single sample
        # scaled over the whole interval.
        e_int = float(np.sum(window.mean(axis=1)) * dt_sub)
        e_pt = float(window[-1].mean() * (r * dt_sub))
        leak = e_pt - e_int
        rel = abs(leak) / abs(e_int) if e_int != 0.0 else float("nan")
        rows.append(dict(ratio=r, DT=r * dt_sub, integrated=e_int, pointwise=e_pt,
                         leak=leak, rel=rel))
        print(f"  {r:7d}{r * dt_sub:11.4g}{e_int:14.6e}{e_pt:14.6e}"
              f"{leak:12.3e}{rel:12.4f}")
    out["leak"] = rows
    out["lambda"] = LAMBDA_STAR

    one = [r for r in rows if r["ratio"] == 1][0]
    full = [r for r in rows if r["ratio"] == max(RATIOS)][0]
    print()
    print(f"  ratio 1 is the single-rate control: leak {one['rel']:.3e}, and it must be")
    print("  zero to machine precision because one sub-step IS the interval. Anything")
    print("  else would mean the instrument is measuring itself.")
    print(f"  At {full['ratio']}:1 the pointwise match delivers "
          f"{full['pointwise'] / full['integrated']:.4f}x the heat the gas actually")
    print("  transported over the interval.")


def stage_lagvsleak(out: dict) -> None:
    """The leak against the OTHER multirate term, in one norm.

    Stage `leak` measures what a pointwise match loses to the flux transient.
    That is what R9 refuses over -- and it is not the only thing a 500:1 exchange
    interval costs.  The trace is also held stale for the whole interval, and
    that defect is `sigma`, which Tier 16 measures in interface power.  Both are
    put in that one norm here, on the same interval, because a rule that refuses
    a graph over the smaller of two terms is refusing the wrong thing and no
    argument decides which is smaller.
    """
    from atlas.multiphysics import interface_power
    from atlas.ports import ResponseHalf

    print("=== the leak against the lag, in interface power, over one interval ===")
    gas = T.GasAgent(dt=T.DT_GAS)
    n = max(RATIOS)
    DT = n * T.DT_GAS
    lam_star = np.full(T.N_SEAM, LAMBDA_STAR)
    lam_lag = np.full(T.N_SEAM, T.T_WALL_0)

    t0 = time.time()
    _q1, flow_star = _march(gas, n, lam_star)
    print(f"  marched at the converged trace ({LAMBDA_STAR:.4f} K) in "
          f"{time.time() - t0:.0f} s", flush=True)
    gas2 = T.GasAgent(dt=T.DT_GAS)
    t0 = time.time()
    _q2, flow_lag = _march(gas2, n, lam_lag)
    print(f"  marched at the lagged trace ({T.T_WALL_0:.4f} K) in "
          f"{time.time() - t0:.0f} s", flush=True)

    def P(lam, f):
        return interface_power(lam, f, ResponseHalf.FLOW, T.H_SEAM)

    p_int = P(lam_star, flow_star.mean(axis=0))
    p_pt = P(lam_star, flow_star[-1])
    p_lag = P(lam_lag, flow_lag.mean(axis=0))
    scale = abs(p_int)
    leak = abs(p_pt - p_int) / scale
    lag = abs(p_lag - p_int) / scale
    print()
    print(f"  exchange interval        DT = {DT:g} s ({n}:1 against the gas)")
    print(f"  reference interface power   = {p_int:.6e} W/m")
    print(f"  R9's term  (pointwise vs integrated flux) = {leak:.4e}")
    print(f"  the lag    (stale trace over the interval) = {lag:.4e}")
    print(f"  ratio lag / leak            = {lag / leak:.4g}")
    out["lagvsleak"] = dict(DT=DT, ratio=n, power_reference=p_int, leak=leak,
                            lag=lag, lag_over_leak=lag / leak)
    print()
    print("  Both are dimensionless defects in the SAME norm -- interface power, the")
    print("  one unit both sides of the bond share (Tier 16) -- so they are directly")
    print("  comparable and the master bound would add them as scalars.")


def stage_match(out: dict) -> None:
    print("=== the two conditions, on the same recorded march ===")
    print("  Both are read off ONE run, so no convention drifts between them --")
    print("  `w7_time_integrated_interface`'s rule, and the reason its comparison")
    print("  was trustworthy even though its verdict was negative.")
    print()
    gas = T.GasAgent(dt=T.DT_GAS)
    shell = T.ShellAgent(dt=T.DT_GAS)          # a single shell step per interval

    rows = []
    print(f"  {'ratio':>7}{'pointwise resid':>18}{'integrated resid':>19}"
          f"{'ratio p/i':>12}")
    for r in (1, 5, 25, 100, 500):
        DT = r * T.DT_GAS
        lam = np.full(T.N_SEAM, LAMBDA_STAR)
        _q, flow = _march(gas, r, lam)
        # The shell over the SAME interval, at its own (single) step.
        sh = T.ShellAgent(dt=DT)
        f_shell = np.asarray(sh.respond("inner:THERM", lam), float)
        # The gas in both conventions, off the SAME recorded march: the pointwise
        # condition reads the last sub-step, the time-integrated one the mean.
        f_gas_pt = flow[-1]
        f_gas_int = flow.mean(axis=0)
        r_pt = float(np.linalg.norm(f_gas_pt + f_shell))
        r_int = float(np.linalg.norm(f_gas_int + f_shell))
        rows.append(dict(ratio=r, pointwise=r_pt, integrated=r_int,
                         ratio_p_over_i=r_pt / r_int if r_int else float("nan")))
        print(f"  {r:7d}{r_pt:18.6e}{r_int:19.6e}"
              f"{r_pt / r_int if r_int else float('nan'):12.4f}", flush=True)
    out["match"] = rows
    print()
    print("  The two conditions coincide at ratio 1 by construction and separate as")
    print("  the clocks do. Which residual is SMALLER is not the point and is not")
    print("  read as one: the claim R9 makes is conservation, and stage `leak` is")
    print("  where that is decided.")


def stage_verdict(out: dict) -> None:
    print("=== what the compile does with a multirate graph ===")
    from atlas import compile_scheme

    rows = []
    for clocks in ("matched", "native"):
        g, _ = T.build(mode="split-step", clocks=clocks)
        res = compile_scheme(g, probe_state="duct, T_hot=900 K")
        fired = [(d.rule, d.verdict.value) for d in res.decisions._decisions
                 if d.rule in ("R9", "R9/order", "E3")]
        rows.append(dict(clocks=clocks, verdict=res.verdict.value,
                         flux_matching=getattr(res.scheme, "flux_matching", None),
                         multirate=getattr(res.scheme, "multirate", None),
                         fired=[f"{a}:{b}" for a, b in fired]))
        print(f"  clocks={clocks:<8} -> {res.verdict.value:<20} "
              f"flux_matching={getattr(res.scheme, 'flux_matching', None)!r}")
        for a, b in fired:
            print(f"      {a:<10} {b}")
    out["verdict"] = rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_ROOT, "out", "w7m"))
    ap.add_argument("--stages", nargs="*", default=list(STAGES), choices=STAGES)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    out: dict = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": a.stages}
    for s in a.stages:
        print()
        globals()[f"stage_{s}"](out)
    path = os.path.join(a.out, "w7m.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print()
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
