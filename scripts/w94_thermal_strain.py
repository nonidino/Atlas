"""CS-9: what is the BOND for a two-way volumetric coupling?  (W94, W70, W32)

    python scripts/w94_thermal_strain.py [--out out/w94] [--stages ...]

Splits `thermostruct2d.ThermoStruct2D` -- the build-repo solver `thermal_seam`
already uses -- into a conduction agent and an elasticity agent coupled through
thermal strain, which is a VOLUME term, and requires it to cross as a declared
bond.  The unsplit solve is the referent, so nothing here derives a ground truth.

Stages, and what each decides:

    operators   G, extracted from the expert's own quadrature and ASSERTED
                against `solve_mechanical`; the exact identity G = G_surf +
                G_body; and the uniform-dT positive control the surface
                reduction has to pass
    referent    the unsplit march, one-way and two-way, and what the Biot term
                is worth
    split       the synchronous split (bit for bit) and the lagged one (the
                splitting error), with its order in dt
    routes      what the closed vocabulary can express: a surface MECH bond and
                a GlobalField, and how each is wrong
    power       port-algebra §6's R(t) with the volume term named, both
                conjugate readings, and the energy that belongs to neither agent
    compile     the three routes through `compile_scheme`
    amendment   `PortAmendment`'s six fields, decided from the measurements

Environment: set ATLAS_BUILD_REPO if the build repo is not at
~/physics-foundation-model.
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
from atlas.cases import thermal_strain as TS                       # noqa: E402
from atlas.holes import NamedHoleError                             # noqa: E402

STAGES = ("operators", "referent", "split", "routes", "power", "compile", "amendment")

nrm = np.linalg.norm


def _rel(a, b) -> float:
    return float(nrm(np.asarray(a) - np.asarray(b)) / nrm(np.asarray(b)))


# ---------------------------------------------------------------------------


def stage_operators(out: dict) -> None:
    print("== operators: is G the expert's own thermal load, and does it split? ==")
    mesh, ts, ops = TS.coupling()
    n = mesh.n_nodes
    rng = np.random.default_rng(0)

    checks = {}
    for label, dT in (
        ("uniform-100K", np.full(n, 100.0)),
        ("streak-transient", TS.monolith(n_steps=TS.N_STEPS)["T"] - TS.T_REF),
        ("random-50K", 50.0 * rng.standard_normal(n)),
    ):
        v = ops.verify_against_expert(ts, dT)
        v["body_load_fraction"] = float(nrm(ops.G_body.dot(dT)) / nrm(ops.G.dot(dT)))
        v["surf_load_fraction"] = float(nrm(ops.G_surf.dot(dT)) / nrm(ops.G.dot(dT)))
        checks[label] = v
        print(f"  {label:>18}:  G vs solve_mechanical {v['G_vs_solve_mechanical']:.2e}"
              f"   G = Gs+Gb {v['G_surf_plus_G_body']:.2e}"
              f"   ||Gb dT||/||G dT|| {v['body_load_fraction']:.4f}")

    print(f"  beta = E alpha / (1-nu)      = {ops.beta:.6e} Pa/K")
    print(f"  Biot coupling number delta   = {ops.delta:.6e}")
    print(f"  nodes {n}, elements {ts.conn.shape[0]}, dofs {2 * n}")
    print("  the uniform control is the one that decides field 5: G_body is "
          f"{checks['uniform-100K']['body_load_fraction']:.2e} of the load there -- "
          "machine zero -- so the surface reduction is exact and W70's nearly "
          "isothermal one-way shell could not have seen this.")
    out["operators"] = {
        "checks": checks,
        "beta_Pa_per_K": ops.beta,
        "delta_biot": ops.delta,
        "n_nodes": int(n), "n_elements": int(ts.conn.shape[0]), "n_dofs": int(2 * n),
        "uniform_control_body_load": checks["uniform-100K"]["body_load_fraction"],
    }


def stage_referent(out: dict) -> None:
    print("== referent: the unsplit ThermoStruct2D solve, one-way and two-way ==")
    m1 = TS.monolith()
    m2 = TS.monolith(two_way=True)
    elas = m1["agents"][1]

    d_eps = elas.stress(m1["u"], None)                 # D eps(u), no eigenstrain
    d_eps0 = d_eps - m1["sigma"]                       # D eps_0, by construction
    cancel = float(nrm(d_eps) / nrm(m1["sigma"]))

    res = {
        "T_range": [float(m1["T"].min()), float(m1["T"].max())],
        "sigma_norm_Pa": float(nrm(m1["sigma"])),
        "sigma_max_Pa": float(np.abs(m1["sigma"]).max()),
        "D_eps_norm_Pa": float(nrm(d_eps)),
        "D_eps0_norm_Pa": float(nrm(d_eps0)),
        "cancellation_ratio": cancel,
        "two_way_iterations_median": int(np.median(m2["iterations"])),
        "two_way_dT_max_K": float(np.abs(m2["T"] - m1["T"]).max()),
        "two_way_dsigma_rel": _rel(m2["sigma"], m1["sigma"]),
    }
    print(f"  T in [{res['T_range'][0]:.3f}, {res['T_range'][1]:.3f}] K after "
          f"{TS.N_STEPS} steps of {TS.DT_MACRO} s")
    print(f"  |sigma|_F = {res['sigma_norm_Pa']:.6e} Pa, max {res['sigma_max_Pa']:.4e} Pa")
    print(f"  |D eps(u)| = {res['D_eps_norm_Pa']:.6e} and |D eps_0| = "
          f"{res['D_eps0_norm_Pa']:.6e}: the stress is a "
          f"{cancel:.1f}x CANCELLATION, so every relative error below is on the residual")
    print(f"  two-way: fixed point {res['two_way_iterations_median']} iterations, "
          f"reverse half moves T by {res['two_way_dT_max_K']:.4f} K and sigma by "
          f"{res['two_way_dsigma_rel']:.4e}")
    out["referent"] = res


def stage_split(out: dict) -> None:
    print("== split: does the split graph reproduce the monolith's stress field? ==")
    m1 = TS.monolith()
    rows = []
    for lag in (0, 1, 2):
        s = TS.split("global-field", lag=lag)
        rows.append({"lag": lag,
                     "rel_stress_error": _rel(s["sigma"], m1["sigma"]),
                     "bitwise_identical": bool(np.array_equal(s["sigma"], m1["sigma"])),
                     "rel_u_error": _rel(s["u"], m1["u"])})
        print(f"  lag {lag} macro-step(s): rel stress {rows[-1]['rel_stress_error']:.6e}"
              f"   bitwise {rows[-1]['bitwise_identical']}")
    print("  lag 0 is a CONTROL on the plumbing, not a measurement: the same "
          "arithmetic in the same order. W106 -- a floor of exactly zero bounds nothing.")

    print("  order of the splitting error in dt (lag = 1 macro-step):")
    order = []
    for dt, ns in ((2.0e-1, 10), (1.0e-1, 20), (5.0e-2, 40), (2.5e-2, 80)):
        mm = TS.monolith(n_steps=ns, dt=dt)
        ss = TS.split("global-field", n_steps=ns, dt=dt, lag=1)
        e = _rel(ss["sigma"], mm["sigma"])
        order.append({"dt": dt, "n_steps": ns, "rel_stress_error": e})
        print(f"    dt = {dt:<8g} rel {e:.6e}")
    slopes = [np.log2(order[i]["rel_stress_error"] / order[i + 1]["rel_stress_error"])
              for i in range(len(order) - 1)]
    print("    log2 ratios between successive halvings: "
          + ", ".join(f"{s:.3f}" for s in slopes) + "  (1.0 = first order)")

    # the two-way staggered pass, against the converged two-way referent
    m2 = TS.monolith(two_way=True)
    s2 = TS.split("global-field", lag=0, two_way=True)
    stag = {"vs_converged_two_way": _rel(s2["sigma"], m2["sigma"]),
            "vs_one_way": _rel(s2["sigma"], m1["sigma"]),
            "dT_max_vs_converged_K": float(np.abs(s2["T"] - m2["T"]).max())}
    print("  two-way STAGGERED (one pass, the reverse term lagged one step):")
    print(f"    vs converged two-way {stag['vs_converged_two_way']:.4e};  "
          f"vs one-way {stag['vs_one_way']:.4e}  -- it recovers "
          f"{100 * (1 - stag['vs_converged_two_way'] / stag['vs_one_way']):.1f}% "
          "of the coupling it is there to capture")
    out["split"] = {"lag_sweep": rows, "dt_order": order,
                    "log2_ratios": [float(s) for s in slopes],
                    "two_way_staggered": stag}


def stage_routes(out: dict) -> None:
    print("== routes: what the CLOSED vocabulary can express, and how it is wrong ==")
    m1 = TS.monolith()
    mesh, ts, ops = TS.coupling()
    elas = m1["agents"][1]
    dT = m1["T"] - TS.T_REF

    u_V = elas.solve(ops.G.dot(dT))
    u_S = elas.solve(ops.G_surf.dot(dT))
    u_B = elas.solve(ops.G_body.dot(dT))
    surf = TS.split("surface-mech")

    # the uniform-dT positive control the surface reduction must pass
    dT_u = np.full(mesh.n_nodes, 100.0)
    ctrl_load = float(nrm(ops.G_body.dot(dT_u)) / nrm(ops.G.dot(dT_u)))
    u_cV = elas.solve(ops.G.dot(dT_u))
    u_cS = elas.solve(ops.G_surf.dot(dT_u))
    ctrl_u = float(nrm(u_cS - u_cV) / max(nrm(u_cV), 1e-300))

    res = {
        "surface_mech": {
            "rel_stress_error": _rel(surf["sigma"], m1["sigma"]),
            "body_load_fraction": float(nrm(ops.G_body.dot(dT)) / nrm(ops.G.dot(dT))),
            "displacement_amplification": float(nrm(u_S) / nrm(u_V)),
            "body_only_displacement_amplification": float(nrm(u_B) / nrm(u_V)),
        },
        "uniform_control": {
            "G_body_load_fraction": ctrl_load,
            "rel_displacement_error": ctrl_u,
        },
        "global_field": {
            "rel_stress_error": _rel(TS.split("global-field")["sigma"], m1["sigma"]),
        },
    }
    sm = res["surface_mech"]
    print("  surface MECH  (t = beta dT n on the body's own boundary):")
    print(f"    G_body is {sm['body_load_fraction']:.4f} of the load, and dropping it "
          f"amplifies the displacement {sm['displacement_amplification']:.1f}x")
    print(f"    rel stress error {sm['rel_stress_error']:.4e}  -- the two halves of the "
          "load nearly CANCEL, so a 10% load defect is not a 10% answer")
    print(f"  positive control (uniform dT): ||G_body dT||/||G dT|| = {ctrl_load:.3e} "
          f"-- machine zero -- and rel displacement error {ctrl_u:.3e}: the surface "
          "reduction is EXACT there, which is why W70's one-way case never saw this")
    print("  GlobalField   (eps_0 as a declared global field):")
    print(f"    rel stress error {res['global_field']['rel_stress_error']:.3e} -- "
          "numerically EXACT, and certified by nothing. See the compile stage.")
    out["routes"] = res


def stage_power(out: dict) -> None:
    print("== power: R(t) with the volume term named, and both readings of it ==")
    rows = []
    for dt, ns in ((2.0e-1, 10), (1.0e-1, 20), (5.0e-2, 40), (2.5e-2, 80), (1.25e-2, 160)):
        m = TS.monolith(n_steps=ns, dt=dt)
        cond, elas = m["agents"]
        (Ta, ua), (Tb, ub) = m["history"][-2], m["history"][-1]
        r = TS.power_residual(cond, elas, m["ops"], Ta, ua, Tb, ub, dt)
        r["dt"] = dt
        rows.append(r)
    r = rows[2]
    print(f"  at dt = {r['dt']}:")
    print(f"    E_th = {r['E_th']:.6e} J/m   E_el = {r['E_el']:.6e} J/m   "
          f"ratio {r['E_el_over_E_th']:.3e}")
    print(f"    dE_th/dt = {r['dE_th_dt']:.6e}   P_Gamma = {r['P_gamma']:.6e} W/m "
          "-- the THERMAL balance closes on its surface port alone")
    print(f"    P_Omega, elasticity-side reading  = {r['P_omega_A']:.6e} W/m")
    print(f"    P_Omega, conduction-side reading  = {r['P_omega_B']:.6e} W/m")
    print(f"    the two differ by {abs(r['P_omega_B'] / r['P_omega_A']):.1f}x, and their "
          "SUM is a total derivative:")
    print(f"      P_A + P_B = {r['conjugate_gap']:.6e}   d(E_shared)/dt = "
          f"{r['d_E_shared_dt']:.6e}")
    print(f"    cross term E_cross = {r['E_cross']:.6e} J/m, "
          f"{abs(r['E_cross'] / r['E_el']):.1f}x the elastic energy")
    print(f"    R global: {r['R_global_rel']:.4e} without the volume term, "
          f"{r['R_global_with_volume_rel']:.4e} with it")
    print(f"    P_Omega / P_Gamma = {r['P_omega_over_P_gamma']:.3e} -- R(t) is blind to "
          "the coupling under test by six orders of magnitude")
    print("  the ELASTICITY agent's own balance, dE_el/dt = -P_Omega, in dt:")
    for row in rows:
        print(f"    dt = {row['dt']:<8g} relative residual {row['R_elas_rel']:.4e}")
    ratios = [np.log2(rows[i]["R_elas_rel"] / rows[i + 1]["R_elas_rel"])
              for i in range(len(rows) - 1)]
    print("    log2 ratios: " + ", ".join(f"{x:.3f}" for x in ratios)
          + "  (1.0 = first order; the bond's power closes the subsystem)")
    conj = [abs(row["P_omega_B"] / row["P_omega_A"]) for row in rows]
    out["power"] = {"rows": rows, "R_elas_log2_ratios": [float(x) for x in ratios],
                    "conjugate_ratio": float(np.median(conj)),
                    "cross_over_elastic": float(abs(rows[2]["E_cross"] / rows[2]["E_el"]))}


def stage_compile(out: dict) -> None:
    print("== compile: the three routes through compile_scheme ==")
    res = {}
    try:
        TS.build(route="volumetric")
        raise AssertionError("the volumetric route did not refuse; that is the bug")
    except NamedHoleError as exc:
        res["volumetric"] = {"verdict": "refuse", "raised": "NamedHoleError",
                             "hole": exc.hole.name, "gap": exc.hole.gap,
                             "detail": str(exc).split(". Gap")[0]}
        print(f"  volumetric   -> {res['volumetric']['raised']} from "
              f"holes.{res['volumetric']['hole']} ({res['volumetric']['gap']}). "
              "The package refuses the sixth port type at the line it would enter.")

    for route in ("surface-mech", "global-field"):
        g, _e = TS.build(route=route)
        r = compile_scheme(g)
        stamp = dict(zip(("E1", "E2", "E3", "E4", "E5", "E6", "E7"),
                         r.envelope.tuple()))
        res[route] = {
            "verdict": r.verdict.value,
            "envelope": stamp,
            "refusals": [f"{d.layer}/{d.rule}" for d in r.decisions.refusals],
            "n_decertifications": len(r.decisions.decertifications),
            "unmeasured": sorted(r.unmeasured),
            "decertified_rules": sorted({f"{d.layer}/{d.rule}"
                                         for d in r.decisions.decertifications}),
            "seams": len(g.connections),
        }
        for sid, so in r.seam_operators.items():
            blocks = {k: float(np.linalg.norm(np.asarray(
                getattr(b_, "matrix", getattr(b_, "S", b_)))))
                for k, b_ in (so.blocks or {}).items()}
            res[route]["seam"] = {
                "seam_id": sid, "dim_M": so.dim_M, "null_dim": so.null_dim,
                "expected_null_dim": so.expected_null_dim,
                "one_sided": so.one_sided, "beta": so.beta,
                "operator_content": so.operator_content,
                "block_norms": blocks,
            }
            print(f"       seam {sid}: dim M {so.dim_M}, null {so.null_dim} "
                  f"(declared {so.expected_null_dim}), one_sided {so.one_sided}, "
                  f"block norms {blocks}")
            print("       the conduction block is EXACTLY zero: a temperature field "
                  "does not know its boundary is moving, so the interface problem "
                  "this route poses is one-sided to machine zero "
                  "(CASE-STUDY-GUIDE mistake 6, measured)")
        print(f"  {route:<13}-> {r.verdict.value}   stamp "
              + " ".join(f"{k}:{v}" for k, v in stamp.items()))
        print(f"       refusals {res[route]['refusals']}, "
              f"{res[route]['n_decertifications']} decertifications, "
              f"unmeasured {res[route]['unmeasured']}")
    a, b = res["surface-mech"], res["global-field"]
    only_mech = sorted(set(a["decertified_rules"]) - set(b["decertified_rules"]))
    only_gf = sorted(set(b["decertified_rules"]) - set(a["decertified_rules"]))
    res["stamp_trade"] = {"only_surface_mech": only_mech, "only_global_field": only_gf,
                          "unmeasured_only_global_field":
                              sorted(set(b["unmeasured"]) - set(a["unmeasured"]))}
    print("  the comparison that is the finding: the GlobalField route is numerically "
          "EXACT and the MECH route is 1.3e3 wrong, and the EXACT one carries the "
          "better stamp -- E3 " + a["envelope"]["E3"] + " -> " + b["envelope"]["E3"]
          + ", E7 " + a["envelope"]["E7"] + " -> " + b["envelope"]["E7"]
          + f", and {a['n_decertifications'] - b['n_decertifications']} fewer "
          "decertifications.")
    print(f"    only the MECH route decertifies at {only_mech} -- every one a SEAM "
          "property.")
    print(f"    only the GlobalField route decertifies at {only_gf}, which says it is "
          "not a composition and nothing about whether its coupling is sound.")
    print("    it is not uniformly better and that is worth saying: beta is "
          f"unmeasurable without a seam, so {res['stamp_trade']['unmeasured_only_global_field']} "
          "is unmeasured there and not here. Four seam-level decertifications traded "
          "for one 'this is not a composition'.")
    res["stamp_paradox"] = {
        "E3_surface_mech": a["envelope"]["E3"],
        "E3_global_field": b["envelope"]["E3"],
        "E7_surface_mech": a["envelope"]["E7"],
        "E7_global_field": b["envelope"]["E7"],
        "decertifications_surface_mech": a["n_decertifications"],
        "decertifications_global_field": b["n_decertifications"],
    }
    out["compile"] = res


def stage_amendment(out: dict) -> None:
    print("== amendment: PortAmendment's six fields, decided from the measurements ==")
    # The verdict is a function of the measurements, so the stages that produce
    # them run first whether or not they were asked for. A partial --stages run
    # that silently graded an amendment on defaults would be the exact failure
    # `evaluate_amendment`'s `unmeasured` branch exists to prevent.
    for dep in ("operators", "routes", "power"):
        if dep not in out:
            print(f"  (running dependency stage {dep!r} first)")
            globals()[f"stage_{dep}"](out)
    ev = {
        "conjugate_ratio": out["power"]["conjugate_ratio"],
        "cross_over_elastic": out["power"]["cross_over_elastic"],
        "dim_M_over_dim_V": 1.0,
        "probe_solves": float(out["operators"]["n_nodes"] + 1),
        "surface_route_stress_error": out["routes"]["surface_mech"]["rel_stress_error"],
        "uniform_control": out["routes"]["uniform_control"]["G_body_load_fraction"],
    }
    verdict = TS.evaluate_amendment(ev)
    for name in ("bond", "mapping", "transfer", "dtn_reading", "distinctness", "exercise"):
        f = verdict["fields"][name]
        print(f"  {name:<13} {f['verdict']:<16} {f['why'][:110]}")
    print()
    print(f"  VERDICT: {verdict['verdict'].upper()}  (failed: "
          f"{', '.join(verdict['failed']) or 'none'})")
    print(f"  seventh field proposed: {verdict['seventh_field']['name']} -- "
          f"{verdict['seventh_field']['declared_interface']}")
    print(f"  scope: {verdict['scope']}")
    out["amendment"] = {"evidence": ev, "verdict": verdict}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_ROOT, "out", "w94"))
    ap.add_argument("--stages", nargs="*", default=list(STAGES), choices=STAGES)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "w94.json")
    # MERGE into an existing artifact rather than overwrite it. A `--stages compile`
    # run used to replace the whole file with one stage, silently discarding six --
    # which is `serialise-the-trajectory`'s failure in miniature and cost one
    # re-run to notice. The stage list actually present is recorded beside the one
    # asked for, so a partial artifact can never claim to be a whole one.
    out: dict = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            out = json.load(fh)
    out.update({"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "stages_requested": list(a.stages),
                "case": "thermal-strain (CS-9)", "build_repo": TS.build_repo()})
    out.pop("stages", None)
    for s in a.stages:
        print()
        globals()[f"stage_{s}"](out)
        # persist after every stage: a run that dies in stage 6 must not lose 5
        out["stages_present"] = sorted(k for k in out if k in STAGES)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=float)
    print()
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
