"""W68, W69, W70, W71 -- one measurement closes three of them.

    python scripts/w73_locality_and_scope.py [--out out/w73] [--stages ...]

The 2026-08-29 session left five holes and four of them turned out to be the
same question asked from four directions: **how much of a probed block is the
expert's operator, and how much is the boundary condition sitting on top of it?**

  W68  `elliptic_signature` read "consistent with EXPOSED or NONE" on a KNOWN
       embedded conduction solve.  Not because it measures non-normality where
       globality was wanted -- that diagnosis is true and secondary -- but
       because the block it was given was 5.0001*I, the film coefficient, with
       the conduction operator 300x underneath.
  W71  the multiphysics seam is better conditioned than the single-physics one
       (kappa 1.0002 against 1.20 and 21.7) for the same reason, and the
       one-paragraph sketch is replaced here by a closed form with a fitted
       constant.
  W69  `stencil_radius` had no test.  The test is a delta on the seam -- the
       spatial-domain form of the same measurement -- and running it exposes
       `required_halo()` consulting two fields of the three it needs.
  W70  the shell's uncertified half, measured rather than described.

Stages:
    blocks   per-agent omega on window_ns (both modes) and thermal_seam
    law      omega = C Bi (k_max ell)^2, swept over dt and over h
    support  delta on the seam: spread, and the declared halo beside it
    scope    W70's three numbers
    audit    every declared field against the conformance suite that tests it
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from atlas import ProbeBudget, compile_scheme                        # noqa: E402
from atlas.capability import ExpertCapabilities, TimeDiscretization  # noqa: E402
from atlas.cases import thermal_seam as T                            # noqa: E402
from atlas.cases import window_ns as W                               # noqa: E402
from atlas.cases.poseidon import elliptic_signature                  # noqa: E402
from atlas.conformance import run_conformance                        # noqa: E402
from atlas.probe import OPERATOR_CONTENT_FLOOR, operator_content     # noqa: E402

K_SOLID = 120.0
ALPHA = K_SOLID / (2700.0 * 900.0)
STAGES = ("blocks", "law", "support", "scope", "audit")


def stats(S: np.ndarray) -> dict:
    n = S.shape[0]
    nrm = float(np.linalg.norm(S))
    sv = np.linalg.svd(S, compute_uv=False)
    return dict(
        norm_fro=nrm, norm_2=float(np.linalg.norm(S, 2)),
        c=float(np.trace(S) / n),
        omega=operator_content(S),
        kappa=float(sv[0] / max(sv[-1], 1e-300)),
        asym=float(np.linalg.norm(S - S.T) / max(nrm, 1e-300)),
    )


# --------------------------------------------------------------------------
# blocks -- the measurement W60's calibration should have been taken on
# --------------------------------------------------------------------------


def stage_blocks(out: dict) -> None:
    print("=== per-agent blocks: omega, and what elliptic_signature says now ===")
    print(f"  {'graph':<24}{'block':<10}{'|S|_F':>10}{'omega':>11}{'kappa':>11}"
          f"{'asym':>11}  verdict")
    rows = []
    st = np.load(os.path.join(_ROOT, "out", "tier0b", "s0_state.npz"))
    cases = []
    for mode in ("split-step", "as-built"):
        g, _ = W.build(st["u"], st["v"], mode=mode)
        cases.append((f"window_ns {mode}", g, "sx0", "s0"))
    g, _ = T.build(mode="split-step", clocks="matched")
    cases.append(("thermal_seam split-step", g, "cht", "duct"))

    for tag, graph, seam, state in cases:
        r = compile_scheme(graph, probe_state=state)
        op = r.seam_operators[seam]
        items = [("ASSEMBLED", op.S)] + [(k, b.S) for k, b in sorted(op.blocks.items())]
        for name, S in items:
            s = stats(S)
            v = elliptic_signature(S)["verdict"]
            rows.append(dict(graph=tag, block=name, **s, verdict=v))
            print(f"  {tag:<24}{name:<10}{s['norm_fro']:10.5f}{s['omega']:11.3e}"
                  f"{s['kappa']:11.4f}{s['asym']:11.3e}  {v[:44]}")
        print(f"  {'':<24}-> {r.verdict.value}, "
              f"L4/operator-content: "
              f"{[d.rule for d in r.decisions.decertifications if d.rule == 'operator-content'] or 'admit'}")
    out["blocks"] = rows

    print()
    print("  eps stability -- is a thin block noise, or a resolved small number?")
    g, _ = W.build(st["u"], st["v"], mode="split-step")
    norms: dict[str, list[float]] = {}
    for eps in (1e-2, 1e-3, 1e-4, 1e-5):
        op = compile_scheme(g, probe_budget=ProbeBudget(fd_step=eps),
                            probe_state="s0").seam_operators["sx0"]
        for k, b in op.blocks.items():
            norms.setdefault(k, []).append(float(np.linalg.norm(b.S)))
    for k, v in sorted(norms.items()):
        print(f"    block {k}: ||S|| over eps 1e-2..1e-5 spread "
              f"{max(v) / min(v):.5f}x  -> {'RESOLVED' if max(v) / min(v) < 1.01 else 'NOISE'}")
    out["eps_stability"] = {k: v for k, v in norms.items()}


# --------------------------------------------------------------------------
# law -- omega = C Bi (k_max ell)^2
# --------------------------------------------------------------------------


def _shell_block(agent, m_eff=None) -> np.ndarray:
    m_eff = T.M_EFF if m_eff is None else m_eff
    P = T.fourier_basis(T.N_SEAM, m_eff)
    base = agent.base_trace()
    f0 = agent.respond("inner:THERM", base)
    eps = 1e-3
    cols = [(agent.respond("inner:THERM", base + eps * P[:, j]) - f0) / eps
            for j in range(P.shape[1])]
    return P.T @ (T.H_SEAM * np.stack(cols, axis=1))


def _k_max(m_eff=None) -> float:
    P = T.fourier_basis(T.N_SEAM, T.M_EFF if m_eff is None else m_eff)
    ks = []
    for j in range(P.shape[1]):
        ks.append(2.0 * np.pi * int(np.argmax(np.abs(np.fft.rfft(P[:, j])))) / T.L_Z)
    return float(max(ks))


def stage_law(out: dict) -> None:
    k_max = _k_max()
    print("=== omega = C * Bi * (k_max ell)^2 ===")
    print(f"  k_max = {k_max:.2f} rad/m  (shortest retained wavelength "
          f"{2000 * np.pi / k_max:.2f} mm), shell thickness {1000 * T.T_SHELL:.1f} mm")
    print(f"  {'sweep':<7}{'dt':>10}{'h_in':>10}{'Bi':>11}{'Pi':>12}{'omega':>12}{'C':>9}")
    rows = []
    grid = ([("dt", dt, 500.0) for dt in
             (1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 1.0, 10.0, 100.0)]
            + [("h", 1e-4, h) for h in (5.0, 50.0, 5e2, 5e3, 5e4, 5e5, 5e6, 5e7)])
    for tag, dt, h in grid:
        S = _shell_block(T.ShellAgent(dt=dt, h_in=h))
        om = operator_content(S)
        Bi = h * T.T_SHELL / K_SOLID
        Pi = Bi * (k_max * np.sqrt(ALPHA * dt)) ** 2
        C = om / Pi if Pi > 0 else float("nan")
        rows.append(dict(sweep=tag, dt=dt, h=h, Bi=Bi, Pi=Pi, omega=om, C=C))
        print(f"  {tag:<7}{dt:10.1e}{h:10.1e}{Bi:11.3e}{Pi:12.4e}{om:12.4e}{C:9.4f}")
    out["law"] = rows
    fitted = [r["C"] for r in rows if 1e-5 <= r["Pi"] <= 1.0]
    floor = [r["omega"] for r in rows if r["Pi"] < 1e-6]
    print(f"  C over the {len(fitted)} points with Pi in [1e-5, 1]: "
          f"[{min(fitted):.4f}, {max(fitted):.4f}], spread {max(fitted)/min(fitted):.2f}x")
    print(f"  below Pi = 1e-6 omega stops falling and sits at "
          f"{min(floor):.2e}-{max(floor):.2e}: the probe's own noise floor for omega")
    out["C_range"] = [min(fitted), max(fitted)]
    out["omega_noise_floor"] = max(floor)


# --------------------------------------------------------------------------
# support -- the same locality, in the spatial domain
# --------------------------------------------------------------------------


def stage_support(out: dict) -> None:
    print("=== a delta on the seam: how far does the response reach? ===")
    rows = []
    for tag, agent, port in (("gas", T.GasAgent(dt=T.DT_GAS), "wall:THERM"),
                             ("shell", T.ShellAgent(dt=T.DT_SHELL), "inner:THERM")):
        base = agent.base_trace()
        f0 = np.asarray(agent.respond(port, base), float).ravel()
        d = np.zeros(T.N_SEAM)
        j0 = T.N_SEAM // 2
        d[j0] = 1.0
        g = np.abs(np.asarray(agent.respond(port, base + d), float).ravel() - f0)
        peak = float(g.max())
        above = np.nonzero(g > 1e-6 * peak)[0]
        spread = int(np.max(np.abs(above - j0)))
        tail = [float(g[j0 + k]) for k in range(0, 7)]
        rows.append(dict(agent=tag, spread=spread, peak=peak, tail=tail))
        print(f"  {tag:<6} spread {spread} cells, peak {peak:.4e}")
        print(f"         profile out from the pole: "
              + " ".join(f"{v:.3e}" for v in tail))
    out["support"] = rows
    print()
    print("  and what the records declare, now that required_halo consults")
    print("  time_discretization (W69):")
    for tag, caps in (("gas", T.gas_capabilities(T.GasAgent(dt=T.DT_GAS))),
                      ("shell", T.shell_capabilities(T.ShellAgent(dt=T.DT_SHELL)))):
        print(f"    {tag:<6} time_discretization={caps.time_discretization.value:<9}"
              f"stencil_radius={caps.stencil_radius}  "
              f"substeps={caps.substeps_per_macro_step}  "
              f"required_halo={caps.required_halo()}")
        out.setdefault("declared_halo", {})[tag] = dict(
            time_discretization=caps.time_discretization.value,
            stencil_radius=caps.stencil_radius,
            substeps=caps.substeps_per_macro_step,
            required_halo=caps.required_halo(),
        )


# --------------------------------------------------------------------------
# scope -- W70's three numbers
# --------------------------------------------------------------------------


def stage_scope(out: dict) -> None:
    import inspect

    _C2, TS, _TH, _GR = T.load_solvers()
    params = list(inspect.signature(TS.ThermoStruct2D.step_thermal).parameters)
    mech = [p for p in params if p in ("u", "disp", "displacement", "strain",
                                       "sigma", "stress", "eps")]
    print("=== W70: the half of the shell that is on no port ===")
    print(f"  step_thermal parameters: {params}")
    print(f"  mechanical ones among them: {mech or 'NONE'} -> "
          f"{'two-way' if mech else 'ONE-WAY, so the thermal subsystem is closed'}")
    out["one_way"] = not mech
    out["step_thermal_params"] = params

    a = T.ShellAgent(dt=T.DT_SHELL)
    ts = a._ts
    print()
    print(f"  {'T_gas':>8}{'H_th (declared)':>20}{'H_el (on no port)':>20}{'ratio':>12}")
    rows = []
    for T_gas in (400.0, 600.0, 900.0, 1200.0):
        Tf = ts.step_thermal(np.full(ts.mesh.n_nodes, T.T_WALL_0), 1.0e3, a.h_in,
                             np.full(T.N_SEAM, T_gas), T.H_OUT, T.T_OUT, T_inf=T.T_OUT)
        u, _sig = ts.solve_mechanical(Tf, T.P_0, 0.5 * T.P_0)
        H_th = float(0.5 * Tf @ ts.M_th.dot(Tf))
        H_el = float(0.5 * u.ravel() @ ts.K_me.dot(u.ravel()))
        rows.append(dict(T_gas=T_gas, H_th=H_th, H_el=H_el, ratio=H_el / H_th))
        print(f"  {T_gas:8.1f}{H_th:20.6e}{H_el:20.6e}{H_el / H_th:12.3e}")
    out["energies"] = rows

    print()
    print("  sensitivity to the PORT variable, by interface mode:")
    print(f"  {'mode':>6}{'d|u|/|u|':>14}{'d|sigma|/|sigma|':>20}{'d(flow)/flow':>16}")
    base = np.full(T.N_SEAM, T.T_HOT)
    T0 = ts.step_thermal(np.full(ts.mesh.n_nodes, T.T_WALL_0), 1.0e3, a.h_in, base,
                         T.H_OUT, T.T_OUT, T_inf=T.T_OUT)
    u0, s0 = ts.solve_mechanical(T0, T.P_0, 0.5 * T.P_0)
    f0 = a.respond("inner:THERM", base)
    P = T.fourier_basis(T.N_SEAM, T.M_EFF)
    sens = []
    for j in (0, 1, 7, 15):
        Tj = ts.step_thermal(np.full(ts.mesh.n_nodes, T.T_WALL_0), 1.0e3, a.h_in,
                             base + P[:, j], T.H_OUT, T.T_OUT, T_inf=T.T_OUT)
        uj, sj = ts.solve_mechanical(Tj, T.P_0, 0.5 * T.P_0)
        du = float(np.linalg.norm(uj - u0) / np.linalg.norm(u0))
        ds = float(np.linalg.norm(sj - s0) / np.linalg.norm(s0))
        fj = a.respond("inner:THERM", base + P[:, j])
        df = float(np.linalg.norm(fj - f0) / np.linalg.norm(f0))
        sens.append(dict(mode=j, du=du, ds=ds, dflow=df))
        print(f"  {j:6d}{du:14.5e}{ds:20.5e}{df:16.5e}")
    out["sensitivity"] = sens
    lo, hi = sens[0], sens[-1]
    print(f"  displacement falls {lo['du'] / hi['du']:.0f}x from mode 0 to mode 15; "
          f"stress RISES {hi['ds'] / lo['ds']:.1f}x and ends "
          f"{hi['ds'] / hi['dflow']:.2f}x the certified flow's sensitivity")


# --------------------------------------------------------------------------
# audit -- W69's actual deliverable
# --------------------------------------------------------------------------


def stage_audit(out: dict) -> None:
    print("=== W69: every declared field, against the test that certifies it ===")
    caps = T.shell_capabilities(T.ShellAgent(dt=T.DT_SHELL))
    cert = run_conformance(caps, probe_state="duct")
    print("  (shell record; the port carries its own space and prolongation, so the")
    print("   two new tests run against the real solver rather than declining)")
    print(f"  {'field':<20}{'verdict':<20}{'silent if false':<17}cost")
    tested = set()
    for t in cert.tests:
        tested.add(t.field_name)
        print(f"  {t.field_name:<20}{t.verdict.value:<20}"
              f"{str(t.silent_if_false):<17}{t.cost}")
    out["conformance"] = [t.as_dict() for t in cert.tests]

    declared = [f for f in ExpertCapabilities.__dataclass_fields__
                if not f.startswith("_")]
    untested = [f for f in declared if f not in tested]
    print()
    print(f"  {len(tested)} fields carry a FieldTest; {len(untested)} do not:")
    print("   ", ", ".join(sorted(untested)))
    out["untested_fields"] = sorted(untested)
    print()
    print("  of those with a test, the ones no test can VERIFY:")
    for f, why in (("validity", "falsifiable only; verifying needs the reference "
                                "the predicate exists to replace -> DECERTIFIES"),
                   ("response_half", "no dimensionless diagnostic separates O(1) "
                                     "from O(1) (W66) -> L3/C9 refuses a "
                                     "DISAGREEMENT, passes a matching pair"),
                   ("governing_family", "a free-form string compared across a seam "
                                        "by E3 and read by nothing else -> a "
                                        "matching false pair PROMOTES")):
        print(f"    {f:<18}{why}")
    out["unverifiable"] = ["validity", "response_half", "governing_family"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join("out", "w73"))
    ap.add_argument("--stages", default=",".join(STAGES))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    want = [s.strip() for s in a.stages.split(",") if s.strip()]
    out: dict = {"date": "2026-08-30", "floor": OPERATOR_CONTENT_FLOOR}
    for name in STAGES:
        if name not in want:
            continue
        t0 = time.perf_counter()
        globals()[f"stage_{name}"](out)
        print(f"  [{name}: {time.perf_counter() - t0:.1f}s]")
        print()
    p = os.path.join(a.out, "w73.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("written", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
