"""W304 / W303 -- the common seam base, and what E7's failure at b-c actually is.

Tier 76 published ``beta = 0.149619`` for the rocket's b-c seam **with a
decertification attached** saying the two sides were linearised about states
1374 apart on M, 876% of the base norm, so that number is "not merely the wrong
point but no point at all".  It also flipped `E7` to `fails` with no verdict on
whether that is physics or a sign convention.  Both are numbers with no meaning
yet, and this script is what gives them one.

Stages:

    lamstar     the consistent interface state, by Newton on the interface
                residual -- built in M, not in V, because the two sides carry
                112 and 14 cells
    rebase      beta, kappa, omega and the block shares at the COMMON base,
                against Tier 76's mismatched values
    sweep       beta over the admissible interval, so the mismatched value can
                be placed inside it or outside it rather than merely compared
    passivity   W303: where the negative mode lives, per block, and what
                declaring `effort_normal` does to it

ASCII output only -- the console is cp1252.

    python scripts/w304_seam_base.py --json out/w304.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from atlas.cases import rocket_experts as RE
from atlas.multiphysics import tight_couple

#: The cadence the repeated arms run at. Measured in Tier 76: beta moves 0.3%
#: over a 100x cadence range while the cost moves 84.75x in sub-steps, so this
#: is a priced trade and not a convenience.
DT_GAS = 1.0e-5

#: **The admissible interval, by `w87_referent_cost_and_lag`'s convention**: the
#: two reservoirs the seam can physically reach, which here are the shell's
#: outer sink and the gas's own chamber. A trace outside them is not a coupled
#: state, it is an artefact of the algebra.
LAM_LO, LAM_HI = RE.T_AMBIENT, RE.T_CHAMBER


def _fmt(x, n=6):
    if x is None:
        return "None"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, str):
        return x
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return ("%." + str(n) + "g") % float(x)


def _setup():
    shell, gas, prov = RE.make_bc_experts(dt_gas=DT_GAS)
    graph = RE.build_bc_graph(shell, gas)
    from atlas.compiler import _Context, _derive_transfer          # noqa: PLC2701
    from atlas.envelope import EnvelopeStamp
    from atlas.holes import HoleLedger
    from atlas.probe import ProbeBudget
    from atlas.verdict import DecisionRecord
    ctx = _Context(graph=graph, budget=None, probe_budget=ProbeBudget(),
                   references={}, record=DecisionRecord(), stamp=EnvelopeStamp(),
                   holes=HoleLedger(), probe_state="w304", depth=0)
    tr = _derive_transfer(ctx, graph.connections[0])
    return shell, gas, prov, graph, tr


def _residual_in_M(shell, gas, tr):
    """The interface residual as a map on M, not on V.

    **This is the one place the rocket differs structurally from
    `thermal_seam`, and it is not a detail.**  There both sides carry 48 cells,
    so `w87` can add their responses in V directly.  Here the gas has 112 wall
    cells and the shell 14, over the same 0.28 m of z, so `gas(lam) + shell(lam)`
    is not even a shape that exists.  The residual lives on the common interface
    space: prolong the multiplier to each side, take each side's flow, reduce
    both back with the forced adjoint, and add THERE.
    """
    Pb, Pc = tr.prolongations["b"], tr.prolongations["c"]
    Rb, Rc = Pb.adjoint(tr.space), Pc.adjoint(tr.space)

    def residual(mu):
        mu = np.asarray(mu, dtype=float).ravel()
        vb = Pb.prolong(mu, tr.space)
        vc = Pc.prolong(mu, tr.space)
        fb = np.asarray(gas.respond("c:THERM", vb), float)
        fc = np.asarray(shell.respond("b:THERM", vc), float)
        return Rb @ fb + Rc @ fc

    return residual, (Pb, Pc, Rb, Rc)


def _mu_for_uniform(tr, Pc, value):
    """The multiplier whose prolongation onto the shell is a uniform `value`.

    The basis is orthonormal in the arc-length Gram, so the constant mode is the
    first column and every other mode integrates to zero: a uniform field is the
    first coefficient alone. Taken from the basis rather than assumed, because a
    basis whose first column was not constant would make this silently wrong.
    """
    P = Pc.effective_matrix()
    n = P.shape[0]
    mu, res, _rank, _sv = np.linalg.lstsq(P, np.full(n, float(value)), rcond=None)
    del res
    return mu


def stage_lamstar(shell, gas, tr, out, n_scan=11):
    """The consistent interface state -- by a bracketed SCAN, then Newton.

    **The first version of this stage ran Newton blind and was killed after
    204 s of CPU without converging.  That was not a solver problem.**  With
    both sides carrying the same film coefficient -- `generate.py` hands the
    GAS's own conduction-limited h to the SHELL's Robin channel, so they are
    equal by construction -- the two reported fluxes are

        q_gas   = h (T_i - lam)        T_i frozen: the gas channel is saturated
        q_shell = h (lam - T_face(lam))

    and their SUM is ``h (T_i - T_face(lam))``, in which **lam cancels to first
    order**.  Its only lam-dependence is the shell's face response over one
    50 ms step, which the metal's thermal mass makes small, so the sum's root is
    where the shell's face reaches the gas's near-wall temperature -- thermal
    equilibrium, not a state one macro step can reach.  A Newton on a residual
    that is nearly constant takes enormous steps, and that is what happened.

    So both candidate conditions are SCANNED over the admissible interval first.
    A scan costs one gas call per point and shows a sign change or its absence
    directly, which is the evidence; Newton is then run only where a root is
    actually bracketed.
    """
    print("=" * 78)
    print("LAMSTAR -- which interface condition has a root, and where")
    print("=" * 78)
    Pb, Pc = tr.prolongations["b"], tr.prolongations["c"]
    Rb, Rc = Pb.adjoint(tr.space), Pc.adjoint(tr.space)

    def halves(mu):
        mu = np.asarray(mu, dtype=float).ravel()
        fb = np.asarray(gas.respond("c:THERM", Pb.prolong(mu, tr.space)), float)
        fc = np.asarray(shell.respond("b:THERM", Pc.prolong(mu, tr.space)), float)
        return Rb @ fb, Rc @ fc

    wall_now = float(shell._face_T(shell._T0)[shell._cells].mean())
    calls0 = gas.solver_calls
    vals = np.linspace(LAM_LO, LAM_HI, n_scan)
    rows = []
    print("  one gas call per point; both conditions read off the SAME two halves.")
    print("     base [K]      ||gas||      ||shell||    sum (scalar)  difference")
    for v in vals:
        gb, gc = halves(_mu_for_uniform(tr, Pc, v))
        s_sum = float(np.dot(gb + gc, np.ones_like(gb)) / np.sqrt(gb.size))
        s_dif = float(np.dot(gb - gc, np.ones_like(gb)) / np.sqrt(gb.size))
        rows.append({"base_K": float(v), "gas_norm": float(np.linalg.norm(gb)),
                     "shell_norm": float(np.linalg.norm(gc)),
                     "sum_scalar": s_sum, "diff_scalar": s_dif})
        print("  %11.2f  %12.5e  %12.5e  %13.5e  %13.5e"
              % (v, rows[-1]["gas_norm"], rows[-1]["shell_norm"], s_sum, s_dif))

    def brackets(key):
        out_b = []
        for a, b in zip(rows[:-1], rows[1:]):
            if a[key] == 0.0 or a[key] * b[key] < 0.0:
                out_b.append((a["base_K"], b["base_K"]))
        return out_b

    br_sum, br_dif = brackets("sum_scalar"), brackets("diff_scalar")
    print()
    print("  sign changes over [%g, %g] K:" % (LAM_LO, LAM_HI))
    print("    SUM        (Steklov-Poincare, what Atlas assembles today): %s"
          % (br_sum or "NONE"))
    print("    DIFFERENCE (what `effort_normal` would declare):           %s"
          % (br_dif or "NONE"))

    res = {"scan": rows, "brackets_sum": br_sum, "brackets_difference": br_dif,
           "admissible_interval_K": [LAM_LO, LAM_HI], "newton_start_K": wall_now,
           "gas_base_K": float(gas.base_trace().mean()),
           "shell_base_K": float(shell.base_trace().mean())}

    star = None
    for tag, br, sign in (("sum", br_sum, +1.0), ("difference", br_dif, -1.0)):
        if not br:
            res[tag + "_root"] = None
            continue
        lo, hi = br[0]

        def scalar(x, _s=sign):
            gb, gc = halves(_mu_for_uniform(tr, Pc, float(x)))
            v = gb + _s * gc
            return float(np.dot(v, np.ones_like(v)) / np.sqrt(v.size))

        a_, b_ = lo, hi
        fa = scalar(a_)
        for _ in range(40):
            m = 0.5 * (a_ + b_)
            fm = scalar(m)
            if fa * fm <= 0.0:
                b_ = m
            else:
                a_, fa = m, fm
            if abs(b_ - a_) < 1e-6 * max(1.0, abs(m)):
                break
        root = 0.5 * (a_ + b_)
        res[tag + "_root"] = float(root)
        print("    %s root by bisection: %s K" % (tag, _fmt(root, 7)))
        if tag == "difference":
            star = _mu_for_uniform(tr, Pc, root)

    res["gas_calls"] = gas.solver_calls - calls0
    print()
    print("  cost: %d gas boundary_response calls" % res["gas_calls"])
    print()
    print("  the two sides' own declared bases: gas %s K, shell %s K"
          % (_fmt(res["gas_base_K"], 7), _fmt(res["shell_base_K"], 7)))
    out["lamstar"] = res
    if star is None:
        print("  NO admissible root on either condition; the rebase stage has no")
        print("  common base to use and is skipped.")
    return star


def _op_at(shell, gas, graph, base_M, label):
    op, _tr = RE.seam_operator(graph, probe_state=label, seam_base=base_M)
    sym = 0.5 * (op.S + op.S.T)
    row = {
        "label": label, "beta": op.beta, "kappa": op.kappa,
        "omega": op.operator_content, "null_dim": op.null_dim,
        "norm_F": float(np.linalg.norm(op.S, "fro")),
        "passivity_defect": op.passivity_defect,
        "passivity_lambda_min": op.passivity_lambda_min,
        "sym_eigs_min": float(np.linalg.eigvalsh(sym).min()),
        "blocks": {},
    }
    for aid, b in op.blocks.items():
        bs = 0.5 * (b.S + b.S.T)
        row["blocks"][aid] = {
            "share": b.share, "norm_F": float(np.linalg.norm(b.S, "fro")),
            "omega": b.operator_content,
            "sym_lambda_min": float(np.linalg.eigvalsh(bs).min()),
            "sym_lambda_max": float(np.linalg.eigvalsh(bs).max()),
            "trace": float(np.trace(b.S)),
        }
    return op, row


def stage_rebase(shell, gas, graph, star, out):
    print()
    print("=" * 78)
    print("REBASE -- the seam at the COMMON base, against Tier 76's mismatched one")
    print("=" * 78)
    _op0, split = _op_at(shell, gas, graph, None, "split-base (Tier 76)")
    _op1, joint = _op_at(shell, gas, graph, star, "seam-base (lambda*)")
    for row in (split, joint):
        print("  %-24s beta %-12s kappa %-10s omega %-9s ||S||_F %-10s"
              % (row["label"], _fmt(row["beta"], 6), _fmt(row["kappa"], 5),
                 _fmt(row["omega"], 4), _fmt(row["norm_F"], 5)))
        for aid, b in sorted(row["blocks"].items()):
            print("      block %-3s share %-8s ||S||_F %-10s sym lambda in [%s, %s]"
                  % (aid, _fmt(b["share"], 4), _fmt(b["norm_F"], 5),
                     _fmt(b["sym_lambda_min"], 4), _fmt(b["sym_lambda_max"], 4)))
    r = joint["beta"] / max(1e-300, split["beta"])
    print()
    print("  beta moves %s -> %s, a factor of %s"
          % (_fmt(split["beta"], 6), _fmt(joint["beta"], 6), _fmt(r, 5)))
    print("  (thermal_seam's same move was 0.3757 -> 1.23807, a factor of 3.30)")
    out["rebase"] = {"split": split, "joint": joint, "beta_ratio": float(r)}
    return split, joint


def stage_sweep(shell, gas, graph, tr, out, n=9):
    """beta over a COMMON base swept across the admissible interval.

    The point is not the curve, it is whether Tier 76's mismatched value is
    inside it. On thermal_seam the mismatched 0.3757 lay BELOW the whole range
    [0.4200, 1.7689] -- not the wrong point but no point at all.
    """
    print()
    print("=" * 78)
    print("SWEEP -- beta over a common base across the admissible interval")
    print("=" * 78)
    _res, (Pb, Pc, _Rb, _Rc) = _residual_in_M(shell, gas, tr)
    del _res, Pb
    rows = []
    # log-spaced over the reachable band, so the metal end is resolved
    # the measured root is included explicitly: a sweep that steps over the one
    # state the seam is consistent at cannot place the mismatched value against it.
    root = out.get("lamstar", {}).get("difference_root")
    extra = [1400.0, 2100.0, LAM_HI] + ([float(root)] if root else [])
    vals = np.unique(np.concatenate([
        np.linspace(LAM_LO, 900.0, n - 3), np.array(extra)]))
    for v in vals:
        mu = _mu_for_uniform(tr, Pc, v)
        op, _tr2 = RE.seam_operator(graph, probe_state="sweep %.1f K" % v,
                                    seam_base=mu)
        rows.append({"base_K": float(v), "beta": op.beta, "kappa": op.kappa,
                     "omega": op.operator_content,
                     "passivity_lambda_min": op.passivity_lambda_min})
        print("  base %8.2f K   beta %-12s kappa %-10s omega %-9s sym_min %s"
              % (v, _fmt(op.beta, 6), _fmt(op.kappa, 5), _fmt(op.operator_content, 4),
                 _fmt(op.passivity_lambda_min, 4)))
    betas = [r["beta"] for r in rows if r["beta"] is not None]
    out["sweep"] = {"rows": rows, "beta_min": min(betas), "beta_max": max(betas)}
    print()
    print("  beta over the admissible interval: [%s, %s]"
          % (_fmt(min(betas), 6), _fmt(max(betas), 6)))
    return rows


def stage_passivity(shell, gas, graph, star, out):
    """W303 -- where the negative mode lives, and what effort_normal does to it."""
    print()
    print("=" * 78)
    print("PASSIVITY -- W303: is E7's failure physics or a convention?")
    print("=" * 78)
    _op, row = _op_at(shell, gas, graph, star, "seam-base")
    print("  at the common base, the ASSEMBLED symmetric part has lambda_min %s"
          % _fmt(row["sym_eigs_min"], 5))
    print("  and per block:")
    for aid, b in sorted(row["blocks"].items()):
        print("    %-3s sym lambda in [%s, %s]   trace %s   ||S||_F %s"
              % (aid, _fmt(b["sym_lambda_min"], 5), _fmt(b["sym_lambda_max"], 5),
                 _fmt(b["trace"], 5), _fmt(b["norm_F"], 5)))
    ratio = (row["blocks"]["b"]["norm_F"]
             / max(1e-300, row["blocks"]["c"]["norm_F"]))
    print("  the gas block is %sx the shell's in Frobenius norm." % _fmt(ratio, 4))

    # the DIFFERENCE assembly, which is what declaring effort_normal buys
    print()
    print("  and the difference assembly (what `effort_normal` would declare):")
    ops = {aid: b for aid, b in _op.blocks.items()}
    S_sum = ops["b"].S + ops["c"].S
    S_diff = ops["b"].S - ops["c"].S
    res = {"per_block": row["blocks"], "assembled_sym_min": row["sym_eigs_min"],
           "gas_over_shell_normF": float(ratio)}
    for tag, S in (("sum (Steklov-Poincare, as declared)", S_sum),
                   ("difference (effort_normal declared)", S_diff)):
        sym = 0.5 * (S + S.T)
        ev = np.linalg.eigvalsh(sym)
        sv = np.linalg.svd(S, compute_uv=False)
        print("    %-38s ||S||_F %-10s sigma_min %-11s sym lambda_min %s"
              % (tag, _fmt(float(np.linalg.norm(S, "fro")), 5),
                 _fmt(float(sv.min()), 5), _fmt(float(ev.min()), 5)))
        res[tag.split(" ")[0]] = {
            "norm_F": float(np.linalg.norm(S, "fro")),
            "sigma_min": float(sv.min()), "sym_lambda_min": float(ev.min()),
            "definite": bool(ev.min() > 0.0)}
    print()
    print("  Q4 asked whether the difference is BETTER. Read the two rows above.")
    out["passivity"] = res
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--stage", default="all")
    a = ap.parse_args(argv)
    want = (set(a.stage.split(",")) if a.stage != "all"
            else {"lamstar", "rebase", "sweep", "passivity"})

    out = {"script": "w304_seam_base.py", "date": "2026-09-17",
           "build_repo_commit": "0a407b7", "dt_gas": DT_GAS,
           "admissible_K": [LAM_LO, LAM_HI]}
    shell, gas, prov, graph, tr = _setup()
    out["probe_state"] = RE.probe_state_string(prov)
    print("probe state: %s" % out["probe_state"])
    print()
    star = stage_lamstar(shell, gas, tr, out)
    if "rebase" in want:
        stage_rebase(shell, gas, graph, star, out)
    if "sweep" in want:
        stage_sweep(shell, gas, graph, tr, out)
    if "passivity" in want:
        stage_passivity(shell, gas, graph, star, out)

    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print()
        print("artifact written to %s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
