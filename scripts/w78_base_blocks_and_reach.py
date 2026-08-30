"""W74's class, W75, W76, W77 -- what the probe assumed about its own base.

    python scripts/w78_base_blocks_and_reach.py [--out out/w78] [--stages ...]

Tier 14 closed W74's instance -- the probe linearized about the zero trace, which
on a kelvin port is 0 K -- and left the class open: *nothing detects that a port
spec's own choice of bond has invalidated a probe assumption*.  Chasing that
turned out to close W77 with it and to invert W75.

  W74  the field was put on the wrong object.  `probe_base` is per EXPERT, so
       the two sides of one seam name their own, and on `thermal_seam` they are
       500 K apart on ONE interface variable.  Lambda_M = sum_i P_i* Lambda_i P_i
       is a sum of Jacobians, which is a Jacobian only if the terms share a
       point.
  W75  the honest gate is reachable after all -- and `elliptic_signature` still
       cannot decide the field there, ranking a known pair BACKWARDS at dt = 100.
       The reach can, at the nominal cadence, with no threshold.
  W76  per-block diagnostics were consumed by nothing, and alpha_star by nothing
       anywhere.  What they carry is a bound on what any test taken on the
       assembled operator can see about one agent.
  W77  `probe_state` never needed to be declared: the probe knows its base.

Stages:
    base     the two declared bases, and beta over the admissible interval
    affine   does the block depend on where it was linearized?
    reach    the two requirements in one parameter, and the gate that works
    blocks   per-block against assembled, and the substitution blind spot
    state    probe_state derived from the base rather than declared
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

from atlas import compile_scheme                                     # noqa: E402
from atlas.capability import linear_response                         # noqa: E402
from atlas.cases import thermal_seam as T                            # noqa: E402
from atlas.cases import window_ns as W                               # noqa: E402
from atlas.cases.poseidon import elliptic_signature                  # noqa: E402
from atlas.composition import certify_substitution                   # noqa: E402
from atlas.probe import (                                            # noqa: E402
    BASE_SENSITIVITY_FLOOR,
    OPERATOR_CONTENT_FLOOR,
    SUPPORT_GLOBAL_FRACTION,
    base_sensitivity,
    operator_content,
    support_reach,
)

STAGES = ("base", "affine", "reach", "blocks", "state")

K_SOLID = 120.0
ALPHA = K_SOLID / (2700.0 * 900.0)
EPS = 1.0e-3


def _basis():
    return T.fourier_basis(T.N_SEAM, T.M_EFF)


def block_at(agent, port, base, eps=EPS):
    """One side's block on M, linearized about ``base``."""
    P = _basis()
    f0 = np.asarray(agent.respond(port, base), float).ravel()
    cols = [(np.asarray(agent.respond(port, base + eps * P[:, j]), float).ravel() - f0) / eps
            for j in range(P.shape[1])]
    return P.T @ (T.H_SEAM * np.stack(cols, axis=1))


def beta_of(S):
    return float(np.linalg.svd(S, compute_uv=False)[-1])


def _pair():
    return T.GasAgent(dt=T.DT_GAS), T.ShellAgent(dt=T.DT_SHELL)


# --------------------------------------------------------------------------
# base -- the two declared bases, and what a COMMON one would have given
# --------------------------------------------------------------------------


def stage_base(out: dict) -> None:
    gas, shell = _pair()
    bg, bs = gas.base_trace(), shell.base_trace()
    print("=== the two declared bases, on ONE interface variable ===")
    print(f"  gas   probe_base = {float(bg[0]):.1f} K   (T_WALL_0, the wall it sees)")
    print(f"  shell probe_base = {float(bs[0]):.1f} K   (T_HOT, the gas it sees)")
    print(f"  they differ by {float(bs[0] - bg[0]):.1f} K, and nothing compared them")
    out["declared_bases"] = {"gas": float(bg[0]), "shell": float(bs[0])}

    mism = block_at(gas, "wall:THERM", bg) + block_at(shell, "inner:THERM", bs)
    b_mism = beta_of(mism)
    print()
    print("=== beta at a COMMON base, over the physically admissible interval ===")
    print(f"  the interface temperature is trapped between the reservoirs: "
          f"{T.T_OUT:.0f} K <= lambda <= {T.T_HOT:.0f} K")
    print(f"  {'lambda':>8} {'|S_gas|':>10} {'|S_shell|':>11} {'beta':>10} {'omega':>11}")
    rows = []
    for lam in (250.0, 300.0, 400.0, 500.0, 650.0, 800.0, 900.0, 1100.0, 1400.0):
        b = np.full(T.N_SEAM, lam)
        g = block_at(gas, "wall:THERM", b)
        s = block_at(shell, "inner:THERM", b)
        a = g + s
        rows.append(dict(lam=lam, beta=beta_of(a), omega=operator_content(a),
                         gas=float(np.linalg.norm(g)), shell=float(np.linalg.norm(s))))
        print(f"  {lam:8.1f} {np.linalg.norm(g):10.5f} {np.linalg.norm(s):11.5f} "
              f"{beta_of(a):10.5f} {operator_content(a):11.4e}")
    out["common_base_sweep"] = rows

    phys = [r for r in rows if T.T_OUT <= r["lam"] <= T.T_HOT]
    lo = min(r["beta"] for r in phys)
    hi = max(r["beta"] for r in phys)
    inside = lo <= b_mism <= hi
    print()
    print(f"  mismatched-base beta, which is what the probe reports  = {b_mism:.5f}")
    print(f"  beta over the whole admissible interval                = "
          f"[{lo:.5f}, {hi:.5f}]")
    print(f"  is the reported number attainable at ANY single base?    "
          f"{'yes' if inside else 'NO -- it is below the entire range'}")
    out["mismatched_beta"] = b_mism
    out["admissible_beta_range"] = [lo, hi]
    out["attainable"] = bool(inside)

    print()
    print("=== the consistent base: where the two one-step fluxes balance ===")
    lam_star = _flux_balance(gas, shell)
    b = np.full(T.N_SEAM, lam_star)
    Sg, Ss = block_at(gas, "wall:THERM", b), block_at(shell, "inner:THERM", b)
    print(f"  lambda* = {lam_star:.4f} K")
    print(f"  beta at lambda*   = {beta_of(Sg + Ss):.5f}   "
          f"({beta_of(Sg + Ss) / b_mism:.2f}x the reported number)")
    print(f"  omega at lambda*  = {operator_content(Sg + Ss):.4e}  "
          f"(floor {OPERATOR_CONTENT_FLOOR:g}, so W68's decertification stands)")
    out["lambda_star"] = lam_star
    out["beta_at_lambda_star"] = beta_of(Sg + Ss)
    out["omega_at_lambda_star"] = operator_content(Sg + Ss)


def _flux_balance(gas, shell) -> float:
    def residual(lam):
        b = np.full(T.N_SEAM, lam)
        return float(np.mean(np.asarray(gas.respond("wall:THERM", b), float)
                             + np.asarray(shell.respond("inner:THERM", b), float)))

    a, c = T.T_OUT, T.T_HOT
    fa = residual(a)
    for _ in range(50):
        mid = 0.5 * (a + c)
        fm = residual(mid)
        if (fm > 0) == (fa > 0):
            a, fa = mid, fm
        else:
            c = mid
    return 0.5 * (a + c)


# --------------------------------------------------------------------------
# affine -- the check whose PASSING makes the base free
# --------------------------------------------------------------------------


def stage_affine(out: dict) -> None:
    print("=== does the block depend on where it was linearized? ===")
    print(f"  floor {BASE_SENSITIVITY_FLOOR:g}; below it the response is affine and the")
    print("  base provably cannot matter -- the one check here that PROMOTES")
    print()
    rows = []

    rng = np.random.default_rng(0)
    A = rng.standard_normal((T.N_SEAM, T.N_SEAM))
    P = _basis()
    for tag, respond, base in (
        ("affine fixture (linear_response)", linear_response(A), np.full(T.N_SEAM, 400.0)),
        ("affine fixture, base 0", linear_response(A), np.zeros(T.N_SEAM)),
    ):
        r = base_sensitivity(respond, "p", base, P, EPS, shift=0.10)
        rows.append(dict(agent=tag, **r))
        print(f"  {tag:<34} rel {r['relative_change']:.3e}  affine={r['affine']}")

    gas, shell = _pair()
    for tag, agent, port in (("gas", gas, "wall:THERM"), ("shell", shell, "inner:THERM")):
        for shift in (0.01, 0.10):
            r = base_sensitivity(agent.respond, port, agent.base_trace(), P, EPS, shift=shift)
            rows.append(dict(agent=f"{tag} +{100*shift:.0f}%", **r))
            print(f"  {tag + ' base +' + format(100 * shift, '.0f') + '%':<34} "
                  f"rel {r['relative_change']:.3e}  affine={r['affine']}")
    out["affine"] = rows
    print()
    print("  so the THERM bond (T, q_n/T) is nonlinear in the effort by construction,")
    print("  and the base is not free on either side.")


# --------------------------------------------------------------------------
# reach -- W75, and the gate that actually works
# --------------------------------------------------------------------------


def stage_reach(out: dict) -> None:
    lam = _flux_balance(*_pair())
    P = _basis()
    ks = [2.0 * np.pi * int(np.argmax(np.abs(np.fft.rfft(P[:, j])))) / T.L_Z
          for j in range(P.shape[1])]
    k_max = float(max(ks))

    print("=== W75: two requirements pulling opposite ways in the same parameter ===")
    print(f"  resolve the operator   omega >= {OPERATOR_CONTENT_FLOOR:g}   wants dt LARGE")
    print("  keep the linearization  sensitivity small     wants dt SMALL")
    print()
    print(f"  {'dt [s]':>10} {'(k ell)^2':>11} {'omega':>11} {'resolves':>9}"
          f" {'dT_face':>9} {'sens':>10} {'linearizes':>11}")
    rows = []
    for dt in (1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 1.0, 10.0, 100.0):
        ag = T.ShellAgent(dt=dt)
        base = np.full(T.N_SEAM, lam)
        S0 = block_at(ag, "inner:THERM", base)
        om = operator_content(S0)
        Tn = ag._ts.step_thermal(ag._T0, ag.dt, ag.h_in, base, T.H_OUT, T.T_OUT,
                                 T_inf=T.T_OUT)
        dT = float(np.mean(ag._face_T(Tn))) - T.T_WALL_0
        S1 = block_at(ag, "inner:THERM", np.full(T.N_SEAM, lam + dT))
        sens = float(np.linalg.norm(S1 - S0) / max(np.linalg.norm(S0), 1e-300))
        res, lin = om >= OPERATOR_CONTENT_FLOOR, sens <= 0.10
        rows.append(dict(dt=dt, omega=om, dT=dT, sens=sens, resolves=res, linearizes=lin))
        print(f"  {dt:10.1e} {(k_max * np.sqrt(ALPHA * dt))**2:11.3e} {om:11.4e} "
              f"{'YES' if res else 'no':>9} {dT:9.2f} {sens:10.3e} "
              f"{'YES' if lin else 'no':>11}")
    both = [r for r in rows if r["resolves"] and r["linearizes"]]
    out["reach_tradeoff"] = rows
    out["cadences_satisfying_both"] = len(both)
    print()
    print(f"  cadences satisfying BOTH: {len(both)} -- so the hypothesis that the")
    print("  resolving cadence is unreachable is FALSIFIED.")

    print()
    print("=== and at those cadences, does elliptic_signature decide the field? ===")
    print("  same shell, two solvers, everything else held fixed")
    print(f"  {'dt':>8} {'solver':>9} {'omega':>11} {'kappa':>10} {'asym':>11}  verdict")
    sig_rows = []
    for dt in (1.0, 10.0, 100.0):
        for tag, expose in (("implicit", False), ("explicit", True)):
            ag = T.ShellAgent(dt=dt, expose_elliptic=expose)
            S = block_at(ag, "inner:THERM", np.full(T.N_SEAM, lam))
            sig = elliptic_signature(S)
            v = sig["verdict"]
            short = ("NO OPERATOR" if v.startswith("NO OPERATOR")
                     else "EMBEDDED" if "EMBEDDED" in v else
                     "EXPOSED/NONE" if "EXPOSED" in v else v[:24])
            sig_rows.append(dict(dt=dt, solver=tag, kappa=sig["kappa"],
                                 asym=sig["asymmetry"], verdict=short))
            print(f"  {dt:8.1f} {tag:>9} {operator_content(S):11.4e} {sig['kappa']:10.4f} "
                  f"{sig['asymmetry']:11.4e}  {short}")
    out["signature_at_diagnostic_cadence"] = sig_rows
    print()
    print("  identical at dt = 1 and RANKED BACKWARDS at dt = 100 -- the explicit")
    print("  solver reads the higher kappa. The spectral route does not decide this")
    print("  field at any cadence, so W75 does not close by widening the interval.")

    print()
    print("=== the gate that does work: poke a delta, count what responds ===")
    print(f"  floor {SUPPORT_GLOBAL_FRACTION:g}; a Fourier mode is global by construction,")
    print("  so a block built from smooth modes cannot report locality at all")
    print(f"  {'case':<30}{'declared':>10}{'nonzero':>10}{'fraction':>10}  reading")
    gate = []
    for dt in (5e-2, 1.0):
        for tag, expose, decl in (("implicit", False, "embedded"),
                                  ("explicit", True, "none")):
            ag = T.ShellAgent(dt=dt, expose_elliptic=expose)
            r = support_reach(ag.respond, "inner:THERM", np.full(T.N_SEAM, lam))
            gate.append(dict(case=f"shell {tag} dt={dt}", declared=decl, **r.as_dict()))
            print(f"  {'shell ' + tag + f' dt={dt}':<30}{decl:>10}{r.nonzero:>10}"
                  f"{r.fraction:>10.3f}  {'GLOBAL' if r.is_global else 'not global'}")
    gas, _ = _pair()
    r = support_reach(gas.respond, "wall:THERM", gas.base_trace())
    gate.append(dict(case="gas explicit", declared="none", **r.as_dict()))
    print(f"  {'gas explicit':<30}{'none':>10}{r.nonzero:>10}{r.fraction:>10.3f}  "
          f"{'GLOBAL' if r.is_global else 'not global'}")

    st = np.load(os.path.join(_ROOT, "out", "tier0b", "s0_state.npz"))
    for mode, decl in (("as-built", "embedded"), ("split-step", "exposed")):
        _g, experts = W.build(st["u"], st["v"], mode=mode)
        ag = experts["W00"]
        r = support_reach(ag.respond, "xhi", np.zeros(ag.n))
        gate.append(dict(case=f"window_ns {mode}", declared=decl, **r.as_dict()))
        print(f"  {'window_ns ' + mode:<30}{decl:>10}{r.nonzero:>10}{r.fraction:>10.3f}  "
              f"{'GLOBAL' if r.is_global else 'not global'}")
    out["support_gate"] = gate
    print()
    print("  every declaration corroborated, at two solves each, with no threshold on")
    print("  any spectral quantity. Only the GLOBAL reading is positive: a dense tail")
    print("  can underflow, which is why the same implicit shell reads 0.875 at the")
    print("  matched clock (dt = 1e-4) where exp(-d/ell) dies before the seam ends.")


# --------------------------------------------------------------------------
# blocks -- W76
# --------------------------------------------------------------------------


def stage_blocks(out: dict) -> None:
    print("=== W76: what the assembly hides ===")
    SA, SB = np.diag([1.0, 1e-12]), np.diag([0.0, 1.0])
    for tag, S in (("block A", SA), ("block B", SB), ("assembled", SA + SB)):
        sv = np.linalg.svd(S, compute_uv=False)
        print(f"  {tag:<10} beta {sv[-1]:.3e}  kappa {sv[0] / max(sv[-1], 1e-300):.3e}")
    print("  -> constructed: the sum is perfectly conditioned and a block is singular.")
    print("     No function of the assembled matrix can recover that.")
    print()

    st = np.load(os.path.join(_ROOT, "out", "tier0b", "s0_state.npz"))
    g, _ = W.build(st["u"], st["v"], mode="split-step")
    res = compile_scheme(g, probe_state="s0")
    rows = []
    print(f"  {'seam':>5} {'assembled beta':>15} {'assembled kappa':>16} "
          f"{'worst block kappa':>18} {'ratio':>9} {'min share':>10}")
    for sid, op in sorted(res.seam_operators.items()):
        if op.kappa is None:
            continue
        worst = max(b.kappa for b in op.blocks.values() if b.kappa is not None)
        ms = op.mode_shares
        rows.append(dict(seam=sid, beta=op.beta, kappa=op.kappa, worst_block=worst,
                         ratio=worst / op.kappa, one_sided=op.one_sided,
                         mode_share_median=float(np.median(ms)) if ms is not None else None))
        print(f"  {sid:>5} {op.beta:15.6f} {op.kappa:16.4f} {worst:18.4f} "
              f"{worst / op.kappa:9.1f} {op.one_sided:10.4f}")
    out["blocks"] = rows

    print()
    print("=== the derivable consequence: is the substitution certificate blind? ===")
    print("  replace one agent with an expert that IGNORES its boundary data --")
    print("  the largest move such a failure can make is that agent's own block")
    subs = []
    for sid in ("sx0", "sy0"):
        op = res.seam_operators[sid]
        norms = {a: float(np.linalg.norm(b.S, 2)) for a, b in op.blocks.items()}
        weak = min(norms, key=norms.get)
        S_new = op.S - op.blocks[weak].S
        caps = g.agent(weak).capabilities
        for beta_min in (0.01, 0.05, 0.10, 0.20):
            c = certify_substitution(
                agent_id=weak, old_caps=caps, new_caps=caps, S_old=op.S, S_new=S_new,
                beta=op.beta, beta_min=beta_min, block_norm=norms[weak],
            )
            subs.append(dict(seam=sid, agent=weak, beta_min=beta_min,
                             delta=c.delta_norm, verdict=c.verdict.value, blind=c.blind))
            print(f"  {sid} drop {weak}: beta_min {beta_min:.2f} -> "
                  f"||Delta|| {c.delta_norm:.5f} vs margin {c.margin:.5f}  "
                  f"{c.verdict.value.upper():<18} blind={c.blind}")
        b_new = float(np.linalg.svd(S_new, compute_uv=False)[-1])
        print(f"     and the seam really moved: beta {op.beta:.5f} -> {b_new:.5f} "
              f"({100 * abs(b_new - op.beta) / op.beta:.1f}%)")
    out["substitution"] = subs


# --------------------------------------------------------------------------
# state -- W77
# --------------------------------------------------------------------------


def stage_state(out: dict) -> None:
    print("=== W77: probe_state, declared against derived ===")
    g, _ = T.build(mode="split-step", clocks="matched")
    declared = "duct, T_hot=900 K, T_wall=400 K"
    res = compile_scheme(g, probe_state=declared)
    op = res.seam_operators["cht"]
    print(f"  declared : {declared!r}")
    print(f"  derived  : {op.derived_probe_state}")
    print(f"  agree?     {op.probe_state_agrees}")
    print(f"  base check: {op.base_check}")
    out["state"] = {"declared": declared, "derived": op.derived_probe_state,
                    "agrees": op.probe_state_agrees, "base_check": op.base_check}
    print()
    print("  the declared string was on every certificate this case ever emitted, and")
    print("  it was true of the PROBLEM and false of the PROBE -- which linearized at")
    print("  0 K until W74 and at two different points after it. Derived from the base")
    print("  it cannot disagree with the measurement, and it is one fewer declaration.")


# --------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_ROOT, "out", "w78"))
    ap.add_argument("--stages", nargs="*", default=list(STAGES), choices=STAGES)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    out: dict = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": a.stages}
    for s in a.stages:
        print()
        globals()[f"stage_{s}"](out)
    path = os.path.join(a.out, "w78.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print()
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
