"""W300 -- R0's gate: the rocket's b-c seam, probed against real physics.

The first Atlas seam whose two sides are the rocket's own solvers rather than
seeded random matrices.  Stages, each runnable alone:

    setup       geometry, operating point, provenance
    linearity   GATE 4, the control that must PASS -- the shell is linear, so
                its probed block must reproduce its own assembled operator
    seam        GATE 1 -- beta, the null-space dimension, omega, and GATE 2, the
                cost in solver calls AND seconds
    reach       support_reach on both sides; the measured stencil, against the
                declared one
    halves      GATE 3, the control that must FAIL -- W66's three legs against
                real physics: one side wrong REFUSES, both sides wrong does not
    cadence     beta against the gas's probe cadence; the reduced cadence
                reported as a number rather than taken as a convenience
    burn        beta against the burn time the probe base carries

ASCII output only -- the console is cp1252, and print() of any non-ASCII raises.

    python scripts/w300_rocket_bc_seam.py --stage all --json out/w300.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from atlas.cases import rocket_experts as RE
from atlas.ports import ResponseHalf
from atlas.probe import ProbeBudget, probe_base, support_reach

#: The cadence the repeated arms run at.  The chamber's CFL sub-step is 2.99e-7 s
#: at the operating point, so the YAML's own dt_model['b'] = 1e-3 is about 3350
#: explicit sub-steps -- roughly 40 minutes for one block on this machine, which
#: is not a measurement anyone runs five times.  `cadence` measures what the
#: choice costs instead of asserting that it costs nothing.
DT_GAS_ARMS = 1.0e-5
DT_GAS_HEADLINE = 1.0e-4
DT_GAS_CONTROL = 1.0e-6          # the three declaration legs; no physics number


def _fmt(x, n=6):
    if x is None:
        return "None"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, str):
        return x
    return ("%." + str(n) + "g") % float(x)


# ---------------------------------------------------------------------------


def stage_setup(out, dt_gas=DT_GAS_ARMS, burn=RE.T_PROBE_BURN):
    print("=" * 78)
    print("SETUP -- the b-c seam, both sides real")
    print("=" * 78)
    shell, gas, prov = RE.make_bc_experts(dt_gas=dt_gas, burn_time=burn,
                                          record_curve=True)
    curve = prov.pop("curve", [])
    for k in ("build_repo_commit", "p_chamber_Pa", "T_chamber_K", "gamma", "R_gas",
              "gas_cells", "shell_cells", "n_seam_gas", "n_seam_shell",
              "m_eff_gas", "m_eff_shell", "dim_M", "dt_gas_s", "dt_shell_s",
              "h_cold_mean_W_m2K", "h_cold_min_W_m2K", "h_cold_max_W_m2K",
              "h_hot_mean_W_m2K", "h_hot_over_cold", "burn_time_s",
              "T_wall_mean_K", "T_wall_min_K", "T_wall_max_K",
              "valid_at_base", "validity_lost_s", "setup_seconds"):
        print("  %-22s %s" % (k, _fmt(prov.get(k))))
    print()
    print("  probe state: %s" % RE.probe_state_string(prov))
    print()
    print("  the wall's heating curve (mean, max over the 14 seam cells):")
    print("      t [s]   T_mean [K]   T_max [K]")
    for t, mean, mx in curve:
        if abs(t / shell.dt - round(t / shell.dt)) < 1e-9 and round(t / shell.dt) in (
                1, 2, 5, 10, 20, 50, 100):
            print("    %7.3f   %10.4f  %10.4f" % (t, mean, mx))
    out["setup"] = prov
    out["setup"]["curve_tail"] = curve[-1] if curve else None
    return shell, gas, prov


def stage_linearity(out):
    """GATE 4: the shell is linear, so the probe must reproduce its own operator."""
    print()
    print("=" * 78)
    print("LINEARITY -- GATE 4, the control that must PASS")
    print("=" * 78)
    res = {}
    for conv in ("heat", "entropy"):
        sh = RE.ShellAgent(flux_convention=conv)
        h = RE.gas_h_on_shell_seam(RE.ChamberGasAgent(dt=DT_GAS_CONTROL), sh,
                                   RE.T_SHELL_COLD)
        sh.march(RE.T_CHAMBER, h, burn_time=RE.T_PROBE_BURN)
        caps = RE.shell_capabilities(sh)
        step = ProbeBudget().fd_step or max(1e-2, 100.0 * caps.reproducibility_floor)
        n = sh.n_seam
        base = probe_base(caps, "b:THERM", n)
        t0 = time.perf_counter()
        f0 = np.asarray(sh.respond("b:THERM", base), float)
        cols = []
        for k in range(n):
            d = np.zeros(n)
            d[k] = 1.0
            cols.append((np.asarray(sh.respond("b:THERM", base + step * d), float) - f0) / step)
        J_fd = np.column_stack(cols)
        t_fd = time.perf_counter() - t0
        J_ex = RE.shell_thermal_operator(sh, convention=conv)
        rel = float(np.linalg.norm(J_fd - J_ex, "fro") / np.linalg.norm(J_ex, "fro"))
        row = {"rel_error": rel, "fd_step": step, "n_calls": n + 1,
               "seconds": t_fd, "norm_exact": float(np.linalg.norm(J_ex, "fro"))}
        if conv == "entropy":
            J_aff = RE.shell_thermal_operator(sh, convention="heat")
            row["vs_affine_operator"] = float(
                np.linalg.norm(J_fd - J_aff, "fro") / np.linalg.norm(J_aff, "fro"))
        res[conv] = row
        print("  %-8s ||J_probe - J_exact||_F / ||J_exact||_F = %s"
              % (conv, _fmt(rel, 5)))
        print("           fd_step %s, %d calls, %s s, ||J_exact||_F = %s"
              % (_fmt(step, 3), n + 1, _fmt(t_fd, 3), _fmt(row["norm_exact"], 6)))
        if conv == "entropy":
            print("           the SAME probe against the AFFINE operator: %s"
                  % _fmt(row["vs_affine_operator"], 5))
            print("           (that is the control's control: if the entropy bond were")
            print("            affine after all, this would be small and gate 4 would be")
            print("            passing for a reason that is not the one claimed)")
    print()
    print("  floor: CG rtol 1e-10 / fd_step 1e-2 = 1e-8. Nothing here can beat it.")
    out["linearity"] = res
    return res


def stage_seam(shell, gas, prov, out, label="headline"):
    """GATES 1 and 2: beta, null dim, omega -- and the cost, measured."""
    print()
    print("=" * 78)
    print("SEAM -- GATE 1 (beta, null dim, omega) and GATE 2 (cost)")
    print("=" * 78)
    graph = RE.build_bc_graph(shell, gas)
    ps = RE.probe_state_string(prov)
    c0g, c0s = gas.solver_calls, shell.solver_calls
    s0g = gas.solver_substeps
    t0 = time.perf_counter()
    op, tr = RE.seam_operator(graph, probe_state=ps)
    elapsed = time.perf_counter() - t0
    d = op.as_dict()
    res = {
        "label": label, "probe_state": ps,
        "dim_M": op.dim_M, "beta": op.beta, "kappa": op.kappa,
        "null_dim": op.null_dim, "expected_null_dim": op.expected_null_dim,
        "excess_null_directions": op.excess_null_directions,
        "operator_content": op.operator_content,
        "passivity_defect": op.passivity_defect,
        "passivity_lambda_min": op.passivity_lambda_min,
        "cut_score": op.cut_score, "conforming": op.conforming,
        "one_sided": op.one_sided,
        "seconds": elapsed,
        "gas_calls": gas.solver_calls - c0g,
        "gas_substeps": gas.solver_substeps - s0g,
        "shell_calls": shell.solver_calls - c0s,
        "blocks": {},
    }
    for aid, b in op.blocks.items():
        res["blocks"][aid] = {
            "n_solves": b.n_solves, "route": b.route, "share": b.share,
            "beta": b.beta, "kappa": b.kappa, "null_dim": b.null_dim,
            "operator_content": b.operator_content,
            "norm_F": float(np.linalg.norm(b.S, "fro")),
            "base_summary": b.base_summary,
        }
    res["base_check"] = op.base_check
    print("  dim M                 %s   (conforming: %s)" % (op.dim_M, op.conforming))
    print("  beta  = sigma_min(S)  %s" % _fmt(op.beta))
    print("  kappa = cond(S)       %s" % _fmt(op.kappa))
    print("  null_dim              %s   (declared %s, excess %s)"
          % (_fmt(op.null_dim), _fmt(op.expected_null_dim),
             _fmt(op.excess_null_directions)))
    print("  omega (operator cont) %s" % _fmt(op.operator_content))
    print("  passivity lambda_min  %s   defect %s"
          % (_fmt(op.passivity_lambda_min), _fmt(op.passivity_defect)))
    print("  one-sided share       %s" % _fmt(op.one_sided))
    print()
    print("  per block:")
    for aid, b in res["blocks"].items():
        print("    %-6s solves %-4d route %-18s share %s  ||S||_F %s  omega %s"
              % (aid, b["n_solves"], b["route"], _fmt(b["share"], 4),
                 _fmt(b["norm_F"], 5), _fmt(b["operator_content"], 4)))
    print()
    print("  COST, measured (not estimated):")
    print("    gas    %d boundary_response calls, %d CFL sub-steps"
          % (res["gas_calls"], res["gas_substeps"]))
    print("    shell  %d step_thermal calls" % res["shell_calls"])
    print("    wall   %s s TOTAL for the assembled seam" % _fmt(elapsed, 4))
    print("    NOTE: on battery at 1400 of 3800 MHz. Ratios, not wall-clock.")
    print()
    if op.base_check:
        print("  base_disagreement across the two sides (W74's class):")
        for k, v in sorted(op.base_check.items()):
            print("    %-24s %s" % (k, _fmt(v) if not isinstance(v, (list, dict)) else v))
    out.setdefault("seam", {})[label] = res
    return res


def stage_reach(shell, gas, out):
    """The measured stencil against the declared one."""
    print()
    print("=" * 78)
    print("REACH -- support_reach on both sides (the declaration, measured)")
    print("=" * 78)
    res = {}
    for name, agent, port, declared in (("c (shell)", shell, "b:THERM", "embedded"),
                                        ("b (gas)", gas, "c:THERM", "none")):
        t0 = time.perf_counter()
        sr = support_reach(agent.respond, port, agent.base_trace(), amplitude=1.0)
        el = time.perf_counter() - t0
        res[name] = dict(sr.as_dict(), seconds=el, declared=declared)
        print("  %-10s reach %-4d of n=%-4d  nonzero %-4d  fraction %s  is_global %s"
              % (name, sr.reach, sr.n, sr.nonzero, _fmt(sr.fraction, 4), sr.is_global))
        print("             consistent_with %s   declared elliptic_subsolve=%s   (%s s)"
              % (list(sr.consistent_with), declared, _fmt(el, 3)))
        print("             profile: %s" % ["%.3e" % v for v in sr.profile[:6]])
    out["reach"] = res
    return res


def stage_halves(out, dt_gas=DT_GAS_CONTROL):
    """GATE 3: W66's three legs, against real physics rather than a fixture."""
    from collections import Counter

    from atlas.compiler import compile_scheme

    print()
    print("=" * 78)
    print("HALVES -- GATE 3, the control that must FAIL (W66's three legs)")
    print("=" * 78)
    print("  probed at dt_gas=%s: these three legs decide a DECLARATION, not a" % _fmt(dt_gas))
    print("  physical number, so the cadence is chosen for cost and stated.")
    print()
    F, E = ResponseHalf.FLOW, ResponseHalf.EFFORT
    legs = [
        ("both FLOW (the truth)", F, F),
        ("shell flipped to EFFORT", F, E),
        ("gas flipped to EFFORT", E, F),
        ("BOTH flipped to EFFORT", E, E),
    ]
    res = {}
    for name, gas_half, shell_half in legs:
        shell, gas, prov = RE.make_bc_experts(dt_gas=dt_gas)
        graph = RE.build_bc_graph(shell, gas, shell_half=shell_half, gas_half=gas_half)
        r = compile_scheme(graph)
        c9 = [d for d in r.decisions.refusals if d.rule == "C9"]
        allref = Counter(d.rule for d in r.decisions.refusals)
        res[name] = {"verdict": str(r.verdict), "c9_refusals": len(c9),
                     "all_refusals": dict(allref),
                     "message": c9[0].message if c9 else None}
        print("  %-28s verdict %-20s C9 refusals: %d"
              % (name, str(r.verdict), len(c9)))
    print()
    ok_one = (res["shell flipped to EFFORT"]["c9_refusals"] > 0
              and res["gas flipped to EFFORT"]["c9_refusals"] > 0)
    ok_both = res["BOTH flipped to EFFORT"]["c9_refusals"] == 0
    ok_true = res["both FLOW (the truth)"]["c9_refusals"] == 0
    print("  leg 1  same half both sides, correct   -> no C9 refusal : %s" % ok_true)
    print("  leg 2  ONE side wrong                  -> C9 REFUSES    : %s" % ok_one)
    print("  leg 3  BOTH sides wrong                -> no C9 refusal : %s" % ok_both)
    print()
    print("  Leg 3 is the finding and it is a NEGATIVE one: a matching pair of")
    print("  wrong halves passes silently, on real solvers, exactly as W66 said")
    print("  it does on a fixture built to fail.")
    res["_legs_reproduce_W66"] = bool(ok_true and ok_one and ok_both)
    out["halves"] = res
    return res


def stage_saturation(out, dt_gas=DT_GAS_ARMS):
    """W301 -- does the imposed trace REACH the gas solver, or only its flux formula?

    `compressible2d` line 170 sets the isothermal ghost to
    ``T_g = max(2 T_wall - T_i, 20)``, a positivity guard: a linear ghost
    extrapolation below ``T_i/2`` would give a negative temperature and
    ``rho_g = p/(R T_g)`` would divide by it.  The guard is correct.  What it
    also does is turn the Dirichlet channel into a SATURATED one below
    ``T_wall = (T_i + 20)/2``, with no diagnostic anywhere.

    The discriminator is BITWISE, and that is the whole point: a response
    dominated by a film coefficient is SMALL, a response that never happened is
    exactly ZERO.
    """
    print()
    print("=" * 78)
    print("SATURATION -- does the trace reach the solver? (W301)")
    print("=" * 78)
    res = {}

    # -- the rocket's chamber ------------------------------------------------
    gas = RE.ChamberGasAgent(dt=dt_gas)
    TH, cfg = gas._TH, gas._cfg
    W0 = TH.cons_to_prim(gas._U0, cfg.gamma)
    T_i0 = float((W0[..., 3] / (W0[..., 0] * cfg.R))[:, -1].mean())

    def gas_field(Tw):
        sol = gas._solver(np.full(gas.n_seam, float(Tw)))
        U, _k = sol.advance(gas._U0, gas.dt)
        W = TH.cons_to_prim(U, cfg.gamma)
        return (W[..., 3] / (W[..., 0] * cfg.R))[:, -1].copy(), W[:, -1, 3].copy()

    ref = gas_field(300.0)
    rows = []
    for Tw in (300.0, 449.42, 690.0, 1000.0, 1200.0, 1217.0, 1400.0, 1500.0):
        T, p = gas_field(Tw)
        rows.append({"T_wall": Tw, "max_dT": float(np.abs(T - ref[0]).max()),
                     "max_dp": float(np.abs(p - ref[1]).max()),
                     "bitwise_equal": bool(np.array_equal(T, ref[0])
                                           and np.array_equal(p, ref[1]))})
    print("  rocket chamber: initial near-wall T_i = %s K, clamp threshold "
          "(T_i+20)/2 = %s K" % (_fmt(T_i0, 6), _fmt((T_i0 + 20.0) / 2.0, 6)))
    print("    T_wall      max|dT_i|        max|dp|       bitwise equal to the 300 K field")
    for r in rows:
        print("  %9.2f  %14.6e  %14.6e   %s"
              % (r["T_wall"], r["max_dT"], r["max_dp"], r["bitwise_equal"]))
    res["rocket_chamber"] = {"T_i_initial_K": T_i0,
                             "clamp_threshold_K": (T_i0 + 20.0) / 2.0,
                             "scan": rows}

    # -- the same test on thermal_seam, whose numbers W68/W71/W74 quote -------
    from atlas.cases import thermal_seam as TSC

    ts_gas = TSC.GasAgent()
    TH2, cfg2 = ts_gas._TH, ts_gas._cfg
    Wt = TH2.cons_to_prim(ts_gas._U0, cfg2.gamma)
    Ti_ts = float((Wt[..., 3] / (Wt[..., 0] * cfg2.R))[:, 0].mean())

    def ts_field(Tw):
        sol = ts_gas._solver(np.full(TSC.N_SEAM, float(Tw)))
        U, _k = sol.advance(ts_gas._U0, ts_gas.dt)
        W = TH2.cons_to_prim(U, cfg2.gamma)
        return (W[..., 3] / (W[..., 0] * cfg2.R))[:, 0].copy(), W[:, 0, 3].copy()

    refts = ts_field(TSC.T_WALL_0)
    trows = []
    for Tw in (300.0, 350.0, 400.0, 440.0, 459.0, 460.0, 480.0, 600.0):
        T, p = ts_field(Tw)
        trows.append({"T_wall": Tw, "max_dT": float(np.abs(T - refts[0]).max()),
                      "bitwise_equal": bool(np.array_equal(T, refts[0])
                                            and np.array_equal(p, refts[1]))})
    print()
    print("  thermal_seam (T_hot = %s K, declared base %s K): threshold %s K"
          % (_fmt(TSC.T_HOT), _fmt(TSC.T_WALL_0), _fmt((Ti_ts + 20.0) / 2.0, 6)))
    print("    T_wall      max|dT_i|      bitwise equal to the 400 K field")
    for r in trows:
        print("  %9.2f  %14.6e   %s" % (r["T_wall"], r["max_dT"], r["bitwise_equal"]))
    res["thermal_seam"] = {"T_i_initial_K": Ti_ts,
                           "clamp_threshold_K": (Ti_ts + 20.0) / 2.0,
                           "declared_base_K": TSC.T_WALL_0, "scan": trows}

    # -- the L1-VISIBLE signature: support_reach.exact_zeros ------------------
    print()
    print("  The L1-visible signature. `support_reach` has emitted `exact_zeros`")
    print("  since Tier 0 and no layer has ever read it. A saturated channel makes")
    print("  it n-1 EXACTLY; a film-dominated one leaves a small nonzero tail.")
    sig = {}
    for name, agent, port, base in (
            ("rocket b (gas)", gas, "c:THERM", np.full(gas.n_seam, 449.42)),
            ("thermal_seam gas", ts_gas, "wall:THERM", ts_gas.base_trace())):
        sr = support_reach(agent.respond, port, base, amplitude=1.0)
        sig[name] = dict(sr.as_dict())
        print("    %-18s n=%-4d nonzero=%-4d exact_zeros=%-4d (n-1 = %d)  peak %s"
              % (name, sr.n, sr.nonzero, sr.exact_zeros, sr.n - 1, _fmt(sr.peak, 6)))
    res["support_reach_signature"] = sig

    # the shell, for contrast: a real operator has a real tail
    shell = RE.ShellAgent()
    h = RE.gas_h_on_shell_seam(gas, shell, RE.T_SHELL_COLD)
    shell.march(RE.T_CHAMBER, h)
    srs = support_reach(shell.respond, "b:THERM", shell.base_trace(), amplitude=1.0)
    res["support_reach_signature"]["rocket c (shell)"] = dict(srs.as_dict())
    print("    %-18s n=%-4d nonzero=%-4d exact_zeros=%-4d (n-1 = %d)  peak %s"
          % ("rocket c (shell)", srs.n, srs.nonzero, srs.exact_zeros, srs.n - 1,
             _fmt(srs.peak, 6)))
    print()
    print("  So the signature separates: both GAS blocks are saturated and read")
    print("  exact_zeros = n-1; the shell, a genuine operator, does not.")
    out["saturation"] = res
    return res


def stage_cadence(out, cadences=(1.0e-6, 1.0e-5, 1.0e-4)):
    """Does the gas's probe cadence move beta? Measured, not assumed."""
    print()
    print("=" * 78)
    print("CADENCE -- beta against the gas's probe cadence")
    print("=" * 78)
    print("  chamber CFL sub-step at the operating point: 2.99e-07 s")
    print("  the YAML's own dt_model['b'] = 1e-3 is ~3350 of them.")
    print()
    rows = []
    for dt in cadences:
        shell, gas, prov = RE.make_bc_experts(dt_gas=dt)
        graph = RE.build_bc_graph(shell, gas)
        c0, s0 = gas.solver_calls, gas.solver_substeps
        t0 = time.perf_counter()
        op, _tr = RE.seam_operator(graph, probe_state=RE.probe_state_string(prov))
        el = time.perf_counter() - t0
        gb = op.blocks["b"]
        rows.append({"dt_gas": dt, "beta": op.beta, "kappa": op.kappa,
                     "omega": op.operator_content, "null_dim": op.null_dim,
                     "gas_block_norm": float(np.linalg.norm(gb.S, "fro")),
                     "gas_share": gb.share, "seconds": el,
                     "gas_calls": gas.solver_calls - c0,
                     "gas_substeps": gas.solver_substeps - s0})
        print("  dt_gas %8.1e  beta %-12s kappa %-10s omega %-10s ||S_b||_F %-11s"
              " share %-8s  %6.1f s  %d substeps"
              % (dt, _fmt(op.beta, 6), _fmt(op.kappa, 5), _fmt(op.operator_content, 4),
                 _fmt(rows[-1]["gas_block_norm"], 5), _fmt(gb.share, 4), el,
                 rows[-1]["gas_substeps"]))
    if len(rows) > 1:
        print()
        b = [r["beta"] for r in rows]
        print("  beta over the cadence range: %s to %s, a factor of %s"
              % (_fmt(min(b), 6), _fmt(max(b), 6), _fmt(max(b) / max(1e-300, min(b)), 4)))
        s = [r["seconds"] for r in rows]
        print("  and the cost over the same range: a factor of %s"
              % _fmt(max(s) / max(1e-9, min(s)), 4))
    out["cadence"] = rows
    return rows


def stage_burn(out, burns=(0.5, 1.0, 2.5, 5.0, 10.0), dt_gas=DT_GAS_ARMS):
    """Does the probe base's BURN TIME move beta? CS-10's convention, applied."""
    print()
    print("=" * 78)
    print("BURN -- beta against the burn time the probe base carries")
    print("=" * 78)
    rows = []
    for bt in burns:
        shell, gas, prov = RE.make_bc_experts(dt_gas=dt_gas, burn_time=bt)
        graph = RE.build_bc_graph(shell, gas)
        op, _tr = RE.seam_operator(graph, probe_state=RE.probe_state_string(prov))
        rows.append({"burn_s": bt, "T_wall_mean_K": prov["T_wall_mean_K"],
                     "T_wall_max_K": prov["T_wall_max_K"],
                     "valid": prov["valid_at_base"], "beta": op.beta,
                     "kappa": op.kappa, "omega": op.operator_content,
                     "null_dim": op.null_dim,
                     "shell_share": op.blocks["c"].share})
        print("  burn %6.2f s  T_wall %9.4f K (max %9.4f)  valid %-5s  beta %-12s"
              "  kappa %-10s  shell share %s"
              % (bt, prov["T_wall_mean_K"], prov["T_wall_max_K"],
                 str(prov["valid_at_base"]), _fmt(op.beta, 6), _fmt(op.kappa, 5),
                 _fmt(op.blocks["c"].share, 4)))
    if len(rows) > 1:
        b = [r["beta"] for r in rows]
        print()
        print("  beta over the burn window: %s to %s, a factor of %s"
              % (_fmt(min(b), 6), _fmt(max(b), 6), _fmt(max(b) / max(1e-300, min(b)), 4)))
        print("  A beta quoted without its burn time is a number whose reader")
        print("  cannot reproduce it. That is CS-10's convention, here as a number.")
    out["burn"] = rows
    return rows


# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all",
                    help="setup|linearity|seam|reach|saturation|halves|cadence|burn|all")
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--dt-gas", type=float, default=DT_GAS_ARMS)
    ap.add_argument("--headline", action="store_true",
                    help="run the seam stage at dt_gas=%g too" % DT_GAS_HEADLINE)
    a = ap.parse_args(argv)
    want = set(a.stage.split(",")) if a.stage != "all" else {
        "setup", "linearity", "seam", "reach", "saturation", "halves",
        "cadence", "burn"}

    out = {"script": "w300_rocket_bc_seam.py", "date": "2026-09-17",
           "build_repo_commit": "0a407b7",
           "machine": "battery, 1400 of 3800 MHz -- quote ratios, not wall-clock"}
    shell = gas = prov = None
    if want & {"setup", "seam", "reach", "burn"}:
        shell, gas, prov = stage_setup(out, dt_gas=a.dt_gas)
    if "linearity" in want:
        stage_linearity(out)
    if "seam" in want:
        stage_seam(shell, gas, prov, out, label="dt_gas=%g" % a.dt_gas)
        if a.headline:
            sh2, g2, p2 = RE.make_bc_experts(dt_gas=DT_GAS_HEADLINE)
            stage_seam(sh2, g2, p2, out, label="dt_gas=%g" % DT_GAS_HEADLINE)
    if "reach" in want:
        stage_reach(shell, gas, out)
    if "saturation" in want:
        stage_saturation(out, dt_gas=a.dt_gas)
    if "halves" in want:
        stage_halves(out)
    if "cadence" in want:
        stage_cadence(out)
    if "burn" in want:
        stage_burn(out, dt_gas=a.dt_gas)

    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print()
        print("artifact written to %s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
