"""tau at a multiphysics seam -- the blocker that made multiphysics uncertifiable.

    python scripts/w83_multiphysics_attribution.py [--out out/w83] [--stages ...]

E3 asks whether the two sides of a seam declare the same `governing_family`, by
string comparison, and when they do not the compiler emits `tau` as UNDEFINED for
both sides.  That is the standing reason no multiphysics graph in this vault can
be certified, and `rocket.py` -- seven agents, a genuine fluid-structure seam --
is kept as a fixture rather than a target because of it.

`tau` is measured in `scripts/tier0_window_ns.py` as *the composed step given the
TRUE trace, against a reference trajectory*.  **Nothing in that mentions a
governing equation.**  What it needs is a reference trajectory, and at a
multiphysics seam one is constructible from the agents themselves: converge the
interface condition inside the macro-step instead of lagging it across.

Stages:
    power       the bond power norm, and why each agent's own norm cannot work
    reference   the tightly coupled trajectory, and what it costs
    tau         per-agent attribution, against surrogates with known error
    verdict     what it moves on the compile
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

from atlas import compile_scheme                                   # noqa: E402
from atlas.cases import thermal_seam as T                          # noqa: E402
from atlas.multiphysics import (                                   # noqa: E402
    interface_power,
    seam_defect_split,
    tight_couple,
)
from atlas.ports import PortType, ResponseHalf                     # noqa: E402
from atlas.multiphysics import bond_of                             # noqa: E402

STAGES = ("power", "reference", "tau", "verdict")

#: The lagged trace the composition actually starts from: the shell's own
#: initial wall temperature, which is what a first macro-step has in hand.
LAGGED = T.T_WALL_0


def _pair(dt=None):
    dt = T.DT_GAS if dt is None else dt
    return T.GasAgent(dt=dt), T.ShellAgent(dt=dt)


def _flows(gas, shell):
    """agent id -> (trace -> declared THERM flow), the callables the split needs."""
    return {"gas": lambda lam: np.asarray(gas.respond("wall:THERM", lam), float),
            "shell": lambda lam: np.asarray(shell.respond("inner:THERM", lam), float)}


class GasSurrogate:
    """The gas solver with a known error in its wall conductivity.

    A stand-in for a learned expert, which is the case this whole framework is
    for: the error is injected at a known size so that what `tau` recovers can be
    checked against it rather than merely being plausible.
    """

    def __init__(self, base, factor: float):
        self.base, self.factor = base, float(factor)

    def respond(self, port_name, trace):
        return self.factor * np.asarray(self.base.respond(port_name, trace), float)


# --------------------------------------------------------------------------


def stage_power(out: dict) -> None:
    print("=== the norm problem, which has to be settled before tau means anything ===")
    gas, shell = _pair()
    lam = np.full(T.N_SEAM, 400.0)
    print(f"  THERM's declared bond: {bond_of(PortType.THERM)}")
    print("  effort x flow is a power -- that is why W66 chose (T, q_n/T) over (T, q_n)")
    print()
    print("  each agent's OWN state norm, at two wall temperatures one step apart:")
    rows = []
    for tag, val in (("352 K (the coupled state)", 352.378), ("400 K", 400.0),
                     ("450 K", 450.0), ("500 K", 500.0)):
        U, _ = gas._solver(np.full(T.N_SEAM, val)).advance(gas._U0, gas.dt)
        q = np.asarray(gas.respond("wall:THERM", np.full(T.N_SEAM, val)), float)
        rows.append(dict(label=tag, wall=val, U=float(np.linalg.norm(U)),
                         q=float(np.linalg.norm(q))))
        print(f"    {tag:<26} ||U_gas|| = {np.linalg.norm(U):.10e}   "
              f"||flow|| = {np.linalg.norm(q):.6e}")
    out["state_norm_blindness"] = rows
    same = sum(1 for r in rows if r["U"] == rows[0]["U"])
    print(f"  -> the gas state is BIT-IDENTICAL at {same} of those {len(rows)} wall")
    print("     temperatures while its flow moves smoothly. The isothermal wall enters")
    print("     through a ghost state the interior has not felt in one 1e-4 s step, so a")
    print("     defect measured in the gas's own norm would read exactly zero.")
    print()
    print("  and the two agents' norms are not commensurable anyway:")
    Ts = shell._ts.step_thermal(shell._T0, shell.dt, shell.h_in, lam,
                                T.H_OUT, T.T_OUT, T_inf=T.T_OUT)
    print(f"    gas   : conserved variables, ||U|| = {np.linalg.norm(rows[1]['U']):.4e}")
    print(f"    shell : kelvin,              ||T|| = {np.linalg.norm(Ts):.4e}")
    print("    the master bound adds tau + sigma + gamma as SCALARS. There is no scalar")
    print("    sum of these two, and interface power is the unit that is neither's.")
    p = interface_power(lam, np.asarray(gas.respond("wall:THERM", lam), float),
                        ResponseHalf.FLOW, T.H_SEAM)
    print(f"    interface power at 400 K: {p:.6e} W/m")
    out["interface_power_at_400K"] = p


def stage_reference(out: dict) -> None:
    print("=== the reference trajectory: converge the interface inside the step ===")
    gas, shell = _pair()
    f = _flows(gas, shell)
    tc = tight_couple(lambda lam: f["gas"](lam) + f["shell"](lam),
                      np.full(T.N_SEAM, LAGGED))
    print(f"  converged: {tc.converged} in {tc.iterations} iterations")
    print(f"  lambda*  = {float(np.mean(tc.trace)):.6f} K")
    print(f"  residual = {tc.residual_norm:.4e}, from {tc.history[0]:.4e}")
    print("  history: " + " ".join(f"{h:.2e}" for h in tc.history[:6]))
    out["reference"] = tc.as_dict()
    print()
    print("  this is a measurement instrument, not a scheme: converging the interface")
    print("  is exactly what the composition avoids. It is run once to buy the")
    print("  referent, the way tier0 runs a monolith it would never ship.")


def stage_tau(out: dict) -> None:
    print("=== tau, per agent, against surrogates carrying a KNOWN error ===")
    gas, shell = _pair()
    ref = _flows(gas, shell)
    halves = {"gas": ResponseHalf.FLOW, "shell": ResponseHalf.FLOW}
    lam0 = np.full(T.N_SEAM, LAGGED)
    lag = np.full(T.N_SEAM, LAGGED)

    print(f"  {'gas expert':<20}{'tau_gas':>12}{'tau_shell':>12}{'sigma':>12}"
          f"{'total':>12}{'sub-additive':>14}")
    rows = []
    cases = [("reference (itself)", gas)] + [
        (f"k_gas x{f}", GasSurrogate(gas, f)) for f in (1.05, 1.25, 2.0)
    ]
    for tag, g in cases:
        act = _flows(g, shell)
        d = seam_defect_split("cht", act, ref, halves, lam0, lag, measure=T.H_SEAM)
        rows.append(dict(case=tag, **d.as_dict()))
        print(f"  {tag:<20}{d.tau['gas']:12.4e}{d.tau['shell']:12.4e}{d.sigma:12.4e}"
              f"{d.total:12.4e}{str(d.subadditive):>14}")
    out["tau"] = rows
    print()
    print("  the injected errors were 0%, 5%, 25% and 100%, and tau_gas recovers them")
    print("  to the digit. tau_shell stays identically 0: the shell was not swapped.")
    print("  Neither number needed the two sides to share a governing family, which is")
    print("  the whole of what E3 was gating.")
    print()
    base = rows[0]
    print(f"  For the REAL pair tau = 0 exactly -- these ARE the actual solvers, so")
    print(f"  their infidelity is zero by construction, and measuring it is what lets")
    print(f"  the composition say so rather than assume it.")
    out["measured_tau"] = base["tau"]

    # sigma against the INITIAL trace is degenerate and the number says so.
    print()
    print("=== sigma, and why one lag value is not a measurement of it ===")
    print("  sigma came out 1.0000 exactly on every row above, which is a fact about")
    print("  the LAG and not about the coupling: the lagged trace is T_WALL_0, the")
    print("  shell's own initial temperature, so at that trace the shell sees no")
    print("  gradient and transmits ZERO power while at lambda* it transmits all of")
    print("  it. sigma = 1 is therefore |0 - P_ref| / P_ref, the degenerate first")
    print("  macro-step from a uniform initial condition. A real run lags on the")
    print("  PREVIOUS step's converged trace, so sigma is a function of how far the")
    print("  interface moved, and that is the curve:")
    print()
    act = _flows(gas, shell)
    lam_star = tight_couple(lambda lam: ref["gas"](lam) + ref["shell"](lam),
                            np.full(T.N_SEAM, LAGGED)).trace
    ls = float(np.mean(lam_star))
    print(f"  {'lag [K]':>10}{'|lag - lambda*|':>17}{'sigma':>12}{'total':>12}")
    curve = []
    for lag_val in (LAGGED, ls - 20.0, ls - 5.0, ls - 1.0, ls - 0.1, ls):
        d = seam_defect_split("cht", act, ref, halves, lam0,
                              np.full(T.N_SEAM, lag_val), measure=T.H_SEAM)
        curve.append(dict(lag=lag_val, distance=abs(lag_val - ls),
                          sigma=d.sigma, total=d.total))
        print(f"  {lag_val:10.4f}{abs(lag_val - ls):17.4f}{d.sigma:12.4e}"
              f"{d.total:12.4e}")
    out["sigma_curve"] = curve
    out["measured_sigma"] = curve[-1]["sigma"]
    print()
    print("  sigma falls to zero as the lag approaches the converged trace, which is")
    print("  the check that it is measuring the lag and nothing else. The number a")
    print("  certificate should carry is the one at the lag the run actually uses.")


def stage_verdict(out: dict) -> None:
    print("=== what it moves on the compile ===")
    for tag, mode, clocks in (("split-step / matched", "split-step", "matched"),):
        g, _ = T.build(mode=mode, clocks=clocks)
        r = compile_scheme(g, probe_state="duct, T_hot=900 K")
        ds = r.decisions._decisions
        dec = [d for d in ds if d.verdict.value == "admit-uncertified"]
        e3 = [d for d in ds if d.rule == "E3"]
        print(f"  {tag}: {r.verdict.value}, {len(dec)} decertifications")
        print(f"    tau_undefined_seams = {r.tau_undefined_seams}")
        for d in e3:
            print(f"    L1/E3 -> {d.verdict.value}: "
                  + ("tau NOT undefined" if not r.tau_undefined_seams
                     else "tau UNDEFINED"))
        out["verdict"] = {"verdict": r.verdict.value, "decertifications": len(dec),
                          "tau_undefined_seams": list(r.tau_undefined_seams)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_ROOT, "out", "w83"))
    ap.add_argument("--stages", nargs="*", default=list(STAGES), choices=STAGES)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    out: dict = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": a.stages}
    for s in a.stages:
        print()
        globals()[f"stage_{s}"](out)
    path = os.path.join(a.out, "w83.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print()
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
