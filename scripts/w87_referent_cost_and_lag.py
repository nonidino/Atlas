"""Tier 16's three loose ends: the referent's test, its cost, and sigma's lag.

    python scripts/w87_referent_cost_and_lag.py [--out out/w87] [--stages ...]

Tier 16 made the reference pair load-bearing and left three things about it
unfinished.  None of them is new theory; each retires debt the tier itself
opened, and each is decided by a measurement rather than by a preference.

  W87  `lambda_ref` decides whether tau is UNDEFINED at a multiphysics seam and
       has no test.  The claim it makes is that a reference PAIR converges, and
       that is an experiment -- so the question is not whether a string can be
       checked but what the check can actually SEE.
  W85  the referent costs 2(n+1) solves per Newton iteration on a dense
       finite-difference Jacobian.  The seam's probed operator S is already paid
       for.  Is it a good enough Jacobian, and does `operator_content` predict
       when it is?
  W86  sigma is a function of the lag and the certificate carries one number
       with the lag in a provenance string nothing reads.  W77 derived
       `probe_state` from the probe rather than declaring it; can the lag be
       derived from the graph the same way?

Stages:
    refcheck   W87 -- what a converged referent does and does not certify
    jacobian   W85 -- S against dense FD, and whether omega predicts the outcome
    lag        W86 -- is the lag derivable from the graph, or must it be declared
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
from atlas.conformance import _test_lambda_ref                       # noqa: E402
from atlas.multiphysics import numerical_jacobian, tight_couple      # noqa: E402

STAGES = ("refcheck", "jacobian", "lag")

#: The lagged trace a first macro-step has in hand: the shell's own initial
#: wall temperature.  Same starting point as `w83_multiphysics_attribution`.
LAGGED = T.T_WALL_0


def _pair(dt=None):
    dt = T.DT_GAS if dt is None else dt
    return T.GasAgent(dt=dt), T.ShellAgent(dt=dt)


def _flows(gas, shell):
    return {"gas": lambda lam: np.asarray(gas.respond("wall:THERM", lam), float),
            "shell": lambda lam: np.asarray(shell.respond("inner:THERM", lam), float)}


class Blind:
    """A reference that ignores its boundary datum entirely.

    W76's instrument, reused: the expert whose block is zero.  Here it stands for
    a `lambda_ref` naming an expert that cannot participate in an interface
    condition at all -- `CASE-STUDY-GUIDE` mistake 6, where the interface problem
    is *empty* rather than hard.
    """

    def __init__(self, base, port, trace):
        self.value = np.asarray(base.respond(port, trace), float).copy()

    def __call__(self, _lam):
        return self.value


class SignFlipped:
    """A reference on the wrong side of its own bond.

    Both sides then source power into the seam and the bond balances nowhere.
    This is what a `response_half` that is wrong ON THE REFERENCE looks like from
    the referent's side, and it is the case a converged solve should catch.
    """

    def __init__(self, f):
        self.f = f

    def __call__(self, lam):
        return -np.asarray(self.f(lam), float)


# --------------------------------------------------------------------------
# W87 -- the test, and what it can see
# --------------------------------------------------------------------------


def stage_refcheck(out: dict) -> None:
    print("=== W87: lambda_ref names an experiment, so run it ===")
    gas, shell = _pair()
    ref = _flows(gas, shell)
    lam0 = np.full(T.N_SEAM, LAGGED)
    caps = T.gas_capabilities(gas)
    print(f"  the gas record declares lambda_ref = {caps.lambda_ref!r}")
    print()

    cases = [
        ("the real pair", ref),
        ("shell sign-flipped", {"gas": ref["gas"], "shell": SignFlipped(ref["shell"])}),
        ("shell blind", {"gas": ref["gas"], "shell": Blind(shell, "inner:THERM", lam0)}),
        ("gas blind", {"gas": Blind(gas, "wall:THERM", lam0), "shell": ref["shell"]}),
        ("both blind", {"gas": Blind(gas, "wall:THERM", lam0),
                        "shell": Blind(shell, "inner:THERM", lam0)}),
    ]
    # The two reservoirs bound every interface state the seam can physically
    # reach: the shell's outer sink and the gas's inlet. A trace outside them is
    # not a coupled state, it is an artefact of the algebra.
    lo, hi = T.T_OUT, T.T_HOT
    print(f"  {'reference pair':<22}{'verdict':>19}{'iters':>7}{'residual':>12}"
          f"{'lambda* [K]':>14}{'in [%g, %g]' % (lo, hi):>16}")
    rows = []
    for tag, pair in cases:
        t = _test_lambda_ref(caps, pair, lam0)

        def residual(lam, _p=pair):
            return sum(np.asarray(_p[a](lam), float).ravel() for a in sorted(_p))

        tc = tight_couple(residual, lam0)
        star = float(np.mean(tc.trace))
        inside = bool(lo <= star <= hi)
        rows.append(dict(case=tag, verdict=t.verdict.value, measured=str(t.measured),
                         residual=t.residual, lam_star=star, admissible=inside,
                         iterations=tc.iterations))
        it = str(t.measured).split()[-2] if "iterations" in str(t.measured) else "-"
        r = "-" if t.residual is None else f"{t.residual:.3e}"
        print(f"  {tag:<22}{t.verdict.value:>19}{it:>7}{r:>12}{star:14.4f}"
              f"{str(inside):>16}")
    out["refcheck"] = rows
    out["reservoirs"] = [lo, hi]

    print()
    print("  and the two branches that are not a solve at all:")
    caps_none = T.gas_capabilities(gas)
    caps_none.lambda_ref = None
    t_none = _test_lambda_ref(caps_none, ref, lam0)
    t_unrun = _test_lambda_ref(caps, None, None)
    print(f"    lambda_ref undeclared -> {t_none.verdict.value}")
    print(f"    declared, pair not supplied -> {t_unrun.verdict.value}")
    out["refcheck_unrun"] = {"undeclared": t_none.verdict.value,
                             "not_supplied": t_unrun.verdict.value}

    print()
    print("  READ THE TABLE, not the headline. FOUR of the five wrong pairs converge.")
    print("  A monotone residual has a root whichever way its two halves lean, so a")
    print("  blind side and even a SIGN-FLIPPED side still balance -- at a different")
    print("  trace. Only `both blind` fails, and it fails because the interface")
    print("  problem is EMPTY (CASE-STUDY-GUIDE mistake 6), not because the referent")
    print("  is bad. That is precisely one falsifiable claim, and it is the one the")
    print("  compiler needs: L1/E3 promotes on a declaration that a referent EXISTS.")
    print()
    print("  Is the admissible interval a second discriminator? Measured, no:")
    star_blind = [r for r in rows if r["case"] == "shell blind"][0]["lam_star"]
    print(f"    `shell blind` lands at {star_blind:.4f} K against a ceiling of {hi} K --")
    print(f"    outside by {star_blind - hi:.4f} K, a relative margin of "
          f"{(star_blind - hi) / (hi - lo):.2e} over the reservoir span. It is pinned")
    print("    AT the gas's own inlet temperature, because a constant shell flux is")
    print("    balanced only where the gas transmits that constant, and a rule keyed")
    print("    on a 1e-5 relative excursion is not a rule. `shell sign-flipped` lands")
    print("    at 425.69 K, comfortably inside, and is not separated at all.")

    print()
    print("=== and what an undetected wrong referent costs the attribution ===")
    from atlas.multiphysics import seam_defect_split
    from atlas.ports import ResponseHalf

    halves = {"gas": ResponseHalf.FLOW, "shell": ResponseHalf.FLOW}
    consequence = []
    for tag, pair in cases:
        if tag == "both blind":
            continue                      # no referent, so no attribution to score
        d = seam_defect_split("cht", ref, pair, halves, lam0, lam0, measure=T.H_SEAM)
        consequence.append(dict(reference=tag, tau_gas=d.tau["gas"],
                                tau_shell=d.tau["shell"], lam_star=d.trace_reference))
        print(f"  reference = {tag:<22} tau_gas = {d.tau['gas']:.4e}   "
              f"tau_shell = {d.tau['shell']:.4e}")
    out["wrong_referent_cost"] = consequence
    print()
    print("  the ACTUAL pair is the real solvers in every row, so the true tau is 0.")
    print("  Every nonzero entry is the wrong referent being charged to an agent that")
    print("  is exactly its own reference. That is what this test does not catch, and")
    print("  it is `validity`'s shape stated as a number rather than as a caveat.")


# --------------------------------------------------------------------------
# W85 -- the referent's cost, against the operator already paid for
# --------------------------------------------------------------------------


def _seam_operator(graph, budget=None, probe_state="duct", seam_base=None):
    """The seam's probed S and its transfer, by the compiler's own route.

    `tier0_window_ns.probe_all_seams`'s pattern: go through `_derive_transfer` so
    the space and prolongations are the ones the compile would use, rather than a
    second construction that could disagree with it.
    """
    from atlas.compiler import _Context, _derive_transfer          # noqa: PLC2701
    from atlas.envelope import EnvelopeStamp
    from atlas.holes import HoleLedger
    from atlas.probe import ProbeBudget, assemble_seam
    from atlas.verdict import DecisionRecord

    budget = budget or ProbeBudget()
    ctx = _Context(graph=graph, budget=None, probe_budget=budget, references={},
                   record=DecisionRecord(), stamp=EnvelopeStamp(), holes=HoleLedger(),
                   probe_state=probe_state, depth=0)
    conn = graph.connections[0]
    tr = _derive_transfer(ctx, conn)
    op = assemble_seam(graph, conn, tr, budget, {},
                       expected_null_dim=conn.expected_null_dim,
                       probe_state=probe_state, seam_base=seam_base)
    return op, tr


class Counted:
    """A residual that counts what it costs.

    One evaluation is one call to each side, so the solve count is ``2 x calls``
    for a two-sided seam.  A solver that refuses the state it is handed counts as
    an evaluation and reports a huge residual rather than killing the sweep --
    an iteration that leaves the physical range is a result about the Jacobian.
    """

    def __init__(self, f):
        self.f, self.calls, self.failures = f, 0, 0

    def __call__(self, lam):
        self.calls += 1
        try:
            return np.asarray(self.f(lam), float).ravel()
        except Exception:                                             # noqa: BLE001
            self.failures += 1
            return np.full(np.asarray(lam).size, 1e12)

    @property
    def solves(self) -> int:
        return 2 * self.calls


def _coarse_newton(residual, trace0, S, P, R, tol=1e-10, max_iter=20, backtracks=5):
    """Newton in M, prolonged to V: the seam's own operator as the Jacobian.

    ``S`` is dim_M x dim_M and the trace lives in V, so the step is
    ``P S^{-1} R r`` -- solve on the interface space the transmission already
    uses and prolong the correction.  The structural question this measures is
    whether the residual lives in ``range(P)``: anything outside it is
    untouchable by this step, exactly as a uniform shift is the only thing a
    scalar secant can move.

    **Backtracked**, and that is not a convenience.  An unsafeguarded step from
    the split-base operator walks the gas solver out of its physical range -- it
    raises ``non-positive dt`` -- so the undamped comparison measures the solver
    failing rather than the Jacobian.  Any implementation that shipped this would
    backtrack; measuring the version that would ship is the honest comparison.
    """
    lam = np.asarray(trace0, dtype=float).ravel().copy()
    r = np.asarray(residual(lam), float).ravel()
    hist = [float(np.linalg.norm(r))]
    cut = 0
    n_it = 0
    for n_it in range(1, max_iter + 1):
        if hist[-1] <= tol * max(1.0, float(np.linalg.norm(lam))):
            break
        try:
            x = np.linalg.solve(S, -(R @ r))
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(S, -(R @ r), rcond=None)[0]
        step = P @ x
        alpha, best = 1.0, None
        for _ in range(backtracks + 1):
            trial = lam + alpha * step
            rt = np.asarray(residual(trial), float).ravel()
            nt = float(np.linalg.norm(rt))
            if nt < hist[-1]:
                best = (trial, rt, nt)
                break
            alpha *= 0.5
            cut += 1
        if best is None:
            break                       # no descent along this direction at all
        lam, r, nrm = best
        hist.append(nrm)
    ok = hist[-1] <= tol * max(1.0, float(np.linalg.norm(lam)))
    return dict(trace=lam, converged=bool(ok), iterations=n_it,
                residual_norm=hist[-1], history=hist, backtracks=cut)


def stage_jacobian(out: dict) -> None:
    print("=== W85: is the seam's own S a good enough Jacobian for the referent? ===")
    print("  the dense FD Jacobian costs n+1 residual evaluations = 2(n+1) solves.")
    print(f"  Here n = {T.N_SEAM}, so {2 * (T.N_SEAM + 1)} solves per Newton solve, and the")
    print("  seam's S is already assembled by the compiler for every seam it sees.")
    print()

    rows = []
    # omega ~ C Bi (k_max ell)^2 with ell = sqrt(alpha dt) (W71), so the exchange
    # interval moves operator content through the SECOND group while leaving the
    # Biot number alone. The film coefficient moves the first group and was tried
    # first: at h_in = 50 the referent's own Newton wanders into states the gas
    # solver refuses, which is a fact about the referent and not about S, so it
    # would have confounded the comparison this stage exists to make.
    # dt moves the second group, h_in the first, and BOTH are needed because
    # three points spanning 1.34x in omega cannot test a predictor. Only the
    # STIFF end of h_in is swept: at h_in = 50 the referent's own Newton wanders
    # into states the gas solver refuses, which is a fact about the referent.
    for dt, h_in in ((T.DT_GAS, T.H_IN_NOMINAL), (2.0 * T.DT_GAS, T.H_IN_NOMINAL),
                     (4.0 * T.DT_GAS, T.H_IN_NOMINAL), (T.DT_GAS, 5.0e3),
                     (T.DT_GAS, 5.0e4)):
        experts = {"gas": T.GasAgent(dt=dt),
                   "shell": T.ShellAgent(dt=dt, expose_elliptic=True, h_in=h_in)}
        graph, _ = T.build(mode="split-step", clocks="matched", experts=experts,
                           measured=None)
        f = _flows(experts["gas"], experts["shell"])
        lam0 = np.full(T.N_SEAM, LAGGED)

        # (a) the baseline: dense FD, and what it costs.
        c_fd = Counted(lambda lam, _f=f: _f["gas"](lam) + _f["shell"](lam))
        fd = tight_couple(c_fd, lam0)
        star = fd.trace

        # (b) S as the compiler assembles it TODAY -- each side on its own
        #     declared probe_base, which on this seam are 500 K apart (W74's
        #     class). Lambda_M is then a sum of Jacobians taken at different
        #     points, which is what W74 says is not a Jacobian; using it to take
        #     a Newton step is the first place that has a numerical consequence
        #     rather than a certificate one.
        op_split, tr = _seam_operator(graph)
        P = tr.prolongations["gas"].effective_matrix()
        R = tr.prolongations["gas"].adjoint(tr.space)

        # (c) S at a CONSISTENT seam base -- and the base a real run has in hand
        #     is the previous macro-step's converged trace, which is what this is.
        op_joint, _ = _seam_operator(graph, seam_base=R @ star)

        variants = {}
        for tag, op in (("S split-base", op_split), ("S seam-base", op_joint)):
            c = Counted(lambda lam, _f=f: _f["gas"](lam) + _f["shell"](lam))
            cs = _coarse_newton(c, lam0, op.S, P, R)
            # The MECHANISM, not the correlation. A step of the form P S^-1 R r
            # lives in range(P), so the part of the residual orthogonal to it is
            # untouchable at any number of iterations. If the plateau IS that
            # part, `in_subspace` is driven to zero while the total is not.
            r_stop = np.asarray(f["gas"](cs["trace"]) + f["shell"](cs["trace"]), float)
            r_in = float(np.linalg.norm(P @ (R @ r_stop)))
            variants[tag] = dict(
                omega=op.operator_content, iterations=cs["iterations"],
                residual=cs["residual_norm"], converged=cs["converged"],
                solves=c.solves, failures=c.failures, backtracks=cs["backtracks"],
                trace_gap=float(np.max(np.abs(cs["trace"] - star))),
                residual_in_subspace=r_in,
                residual_out_of_subspace=float(np.linalg.norm(r_stop - P @ (R @ r_stop))),
            )
        rows.append(dict(dt=dt, h_in=h_in, omega_split=op_split.operator_content,
                         omega_joint=op_joint.operator_content, beta=op_joint.beta,
                         fd_iters=fd.iterations, fd_residual=fd.residual_norm,
                         fd_converged=fd.converged, fd_solves=c_fd.solves,
                         lam_star=float(np.mean(star)), variants=variants))
        print(f"  dt = {dt:.2e}  h_in = {h_in:<8.0f} omega(split) = "
              f"{op_split.operator_content:.3e}   omega(seam) = "
              f"{op_joint.operator_content:.3e}", flush=True)

    print()
    print(f"  {'dt':>10}{'h_in':>8}{'jacobian':>15}{'omega':>11}{'iters':>7}"
          f"{'residual':>11}{'solves':>8}{'|lam - lam_FD|':>16}")
    for r in rows:
        print(f"  {r['dt']:10.2e}{r['h_in']:8.0f}{'dense FD':>15}{'-':>11}{r['fd_iters']:7d}"
              f"{r['fd_residual']:11.2e}{r['fd_solves']:8d}{0.0:16.3e}")
        for tag, v in r["variants"].items():
            print(f"  {'':18}{tag:>15}{v['omega']:11.3e}{v['iterations']:7d}"
                  f"{v['residual']:11.2e}{v['solves']:8d}{v['trace_gap']:16.3e}"
                  + ("" if v["converged"] else "   NOT CONVERGED")
                  + (f"  [{v['failures']} solver refusals]" if v["failures"] else ""))
            print(f"  {'':18}{'':>15}   residual in range(P) = "
                  f"{v['residual_in_subspace']:.3e}, orthogonal to it = "
                  f"{v['residual_out_of_subspace']:.3e}")
    out["jacobian"] = rows


# --------------------------------------------------------------------------
# W86 -- is the lag derivable, or does it have to stay declared?
# --------------------------------------------------------------------------


def _advance(gas, shell, lam):
    """One macro-step of the TIGHTLY COUPLED pair, in place.

    Both agents see the same interface temperature, which is what "converged
    inside the macro-step" means, so this walks the reference trajectory `tau`
    and `sigma` are both defined against.
    """
    U, _ = gas._solver(lam).advance(gas._U0, gas.dt)
    Tn = shell._ts.step_thermal(shell._T0, shell.dt, shell.h_in, lam,
                                T.H_OUT, T.T_OUT, T_inf=T.T_OUT)
    gas._U0, shell._T0 = U, Tn


def stage_lag(out: dict) -> None:
    print("=== W86: sigma is a function of the lag, and the lag is whose property? ===")
    gas, shell = _pair()
    f = _flows(gas, shell)
    from atlas.multiphysics import seam_defect_split
    from atlas.ports import ResponseHalf

    halves = {"gas": ResponseHalf.FLOW, "shell": ResponseHalf.FLOW}
    lam0 = np.full(T.N_SEAM, LAGGED)

    def residual(lam):
        return f["gas"](lam) + f["shell"](lam)

    # One dense Jacobian, reused as a chord across the walk. Each re-convergence
    # is then a few residual evaluations rather than another 2(n+1) solves, and
    # after the first step the trace moves by ~1e-3 K so a frozen Jacobian is
    # more than good enough -- stated because it is an approximation.
    J = numerical_jacobian(residual, lam0)
    star = tight_couple(residual, lam0, jacobian=J).trace
    print(f"  step 0: lambda* = {float(np.mean(star)):.6f} K", flush=True)

    print()
    print("  first, the number the case file declares, against its own cited source:")
    declared = T.MEASURED_W83.sigma
    art = os.path.join(_ROOT, "out", "w83", "w83.json")
    in_artifact = None
    if os.path.isfile(art):
        with open(art, encoding="utf-8") as fh:
            in_artifact = json.load(fh).get("measured_sigma")
    print(f"    MEASURED_W83.sigma        = {declared:.6e}")
    print(f"    w83.json measured_sigma   = "
          + ("absent" if in_artifact is None else f"{in_artifact:.6e}"))
    out["declared_sigma"] = declared
    out["artifact_sigma"] = in_artifact

    rows = []
    print()
    print("  now walk the reference trajectory and measure the lag it actually carries:")
    print(f"  {'step':>5}{'lambda* [K]':>15}{'drift [K]':>13}{'sigma(drift)':>15}"
          f"{'slope [1/K]':>14}{'slope x drift':>15}")
    prev = star
    for step in range(1, 6):
        _advance(gas, shell, prev)
        nxt = tight_couple(residual, prev, jacobian=J)
        cur = nxt.trace
        drift = float(np.max(np.abs(cur - prev)))
        # sigma at the lag a real run carries: the PREVIOUS step's converged
        # trace, which is exactly what a lagged composition has in hand.
        d_lag = seam_defect_split("cht", f, f, halves, cur, prev, measure=T.H_SEAM,
                                  jacobian=J)
        # and the local slope, from a small probe displacement about the root
        eps = 1.0e-3
        d_eps = seam_defect_split("cht", f, f, halves, cur, cur - eps,
                                  measure=T.H_SEAM, jacobian=J)
        slope = d_eps.sigma / eps
        rows.append(dict(step=step, lam=float(np.mean(cur)), drift=drift,
                         sigma=d_lag.sigma, slope=slope, predicted=slope * drift))
        print(f"  {step:5d}{float(np.mean(cur)):15.6f}{drift:13.4e}{d_lag.sigma:15.4e}"
              f"{slope:14.4e}{slope * drift:15.4e}", flush=True)
        prev = cur
    out["lag_walk"] = rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_ROOT, "out", "w87"))
    ap.add_argument("--stages", nargs="*", default=list(STAGES), choices=STAGES)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    out: dict = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": a.stages}
    for s in a.stages:
        print()
        globals()[f"stage_{s}"](out)
    path = os.path.join(a.out, "w87.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print()
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
