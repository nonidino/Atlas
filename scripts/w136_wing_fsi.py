"""CS-12: two-way FSI across a genuinely new governing family.  (W114, W136)

    python scripts/w136_wing_fsi.py [--out out/w136] [--stages ...]

`case-study-ladder-to-f1` section 4's CS-12 row asks one question -- *"does R
close across a genuinely NEW governing family?"* -- and section 12 names three
things the row inherits and one it has to settle first.  This driver measures
all four.

Stages, in the order they depend on each other:

    setup       the geometry, the structural operator built from the expert's own
                solves, and the fixed-shape spin-up's own settled unsteadiness --
                which is the LEVEL every difference below is quoted against
                (W106: the pipeline's bitwise floor is exactly zero and bounds
                nothing)
    compile     W114 settled: the four compiles, the narrowed R10, and the
                control that shows the narrowing did not switch the rule off
    probe       the seam's own operator -- support reach on a quasi-static
                elastic agent, the three efforts' one-sidedness, beta, the null
                space, and the operator drift a deforming interface carries
    gate        THE DELIVERABLE.  The referent against the split, marched past
                the point where the two could cross, with the zero-cut control
                asserted bitwise
    residual    R across the FSI seam, with the deformation term and without it
    addedmass   the three couplings, the stability map in (stiffness, lag), and
                the divergence boundary the physics has
    horizon     W128 / spec section 0.4: dJ/dE* by adjoint against a central
                difference, N_sign and N_valid
    constants   the record: MeasuredConstants and what stays unmeasured

Every stage writes into one artifact, the artifact is written after EVERY stage,
and the next invocation LOADS it before writing -- CS-11's own mistake, where
`--stages constants` alone destroyed what the other nine had written.
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

import torch                                                        # noqa: E402

from atlas import compile_scheme                                    # noqa: E402
from atlas.cases import wing_fsi as W                               # noqa: E402
from atlas.capability import EllipticSubsolve, MotionClass          # noqa: E402
from atlas.graph import MeasuredConstants                           # noqa: E402
from atlas.probe import (                                           # noqa: E402
    ProbeBudget, assemble_seam, operator_content, operator_drift, support_reach,
)
from atlas.verdict import Verdict                                   # noqa: E402

STAGES = ("setup", "compile", "probe", "gate", "residual", "addedmass",
          "horizon", "constants")

#: Fixed-shape spin-up before any coupled march.  Measured in `stage_setup`: the
#: load is still climbing at 40 and settles into its own shedding band by 120.
N_SPIN = 120

#: The gate's horizon.  Long enough that the aeroelastic transient -- whose
#: timescale is the aerodynamic damping over the structural stiffness, about 15
#: macro-steps -- is over several times, and long enough for the crossing to be
#: LOOKED FOR rather than assumed absent.
N_GATE = 240

#: Lags, in macro-steps, for the order and for the stability map.
LAGS = (1, 2, 4)

#: The added-mass sweep.  `E_STAR` down to the edge of the structural expert's
#: own small-strain envelope, crossed with the exchange lag, which is the axis
#: that can be pushed without leaving it.
E_SWEEP = (5.0e4, 4.0e4, 3.0e4)
LAG_SWEEP = (1, 2, 4, 8, 16, 32)

#: **Every cell is marched to at least this many exchanges of its OWN interface
#: equation.** Added at the CS-12 verification pass, 2026-09-04. The first
#: version of the map marched every cell the same 80 macro-steps, which gave the
#: lag-32 cell two and a half exchanges: `undiverged` there is a statement about
#: arithmetic, not about stability. A loose Gauss-Seidel coupling's amplification
#: acts once per exchange, so the horizon that makes two cells comparable is
#: counted in exchanges and not in macro-steps.
MIN_LAG_INTERVALS = 10

#: The horizon sweep for section 0.4.  `N_sign` and `N_valid` are read off it.
HORIZONS = (20, 40, 60, 80, 120, 160, 240, 320, 480)
FD_STEPS = (1.0e-2, 3.0e-2, 1.0e-1)          # relative, on E*


def _pow(x, y):
    return float(np.log(abs(y[1] / y[0])) / np.log(x[1] / x[0]))


def _horizon_limits(ns, g):
    """N_valid(eps) and N_sign from a horizon table.

    **A tolerance met only at the final horizon is NOT met** -- the last point is
    within 0% of itself by construction, so returning it would be reporting the
    sweep's own endpoint as a result.  Those return None, which reads as *not
    determined over the horizons marched*.  `w131_brake_thermal._horizon_limits`,
    unchanged.
    """
    conv = g[-1]
    lim = {}
    for tol in (0.5, 0.2, 0.1, 0.05):
        ok = [i for i in range(len(g))
              if all(abs(g[k] - conv) <= tol * abs(conv) for k in range(i, len(g)))]
        lim[str(tol)] = ns[ok[0]] if ok and ok[0] < len(g) - 1 else None
    sign = [i for i in range(len(g))
            if all((g[k] > 0) == (conv > 0) for k in range(i, len(g)))]
    cross = [(ns[i], ns[i + 1]) for i in range(len(g) - 1) if g[i] * g[i + 1] < 0]
    return dict(validity_limit=lim, n_sign=(ns[sign[0]] if sign else None),
                crossings=cross, converged=float(conv))


#: Where `main` is writing, so a long stage can persist its own partial results
#: rather than only its finished ones.  `stage_horizon` is the reason: it is the
#: one stage whose rows cost minutes each, and losing eight of them because the
#: ninth ran the box out of memory is the failure the after-every-stage write was
#: supposed to prevent -- one level down.
_ARTIFACT_PATH: str | None = None


def _flush(out: dict) -> None:
    if _ARTIFACT_PATH is None:
        return
    with open(_ARTIFACT_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)


def _field_path(out_dir: str) -> str:
    return os.path.join(out_dir, "settled.npz")


def _settled(out: dict, out_dir: str):
    """The fixed-shape release field, cached on disk across invocations."""
    p = _field_path(out_dir)
    if os.path.isfile(p):
        z = np.load(p)
        return z["u"], z["v"]
    r = W.FSIRollout(tiling=W.SINGLE_TILING, coupling="tight", motion=False)
    res = r.run(steps=N_SPIN)
    u = res.u.detach().cpu().numpy().copy()
    v = res.v.detach().cpu().numpy().copy()
    np.savez(p, u=u, v=v, load=res.load)
    return u, v


# ---------------------------------------------------------------------------
# stage 1 -- setup
# ---------------------------------------------------------------------------


def stage_setup(out: dict, out_dir: str) -> None:
    print("== setup: the geometry, the structural operator, and the level ==")
    t = W.DEFAULT_TILING
    op = W._surface_operator()
    print(f"  domain          {W.NX} x {W.NY} cells at dx = {W.DX:.6g}")
    print(f"  windows         {t.n_windows} of {t.wx}x{t.wy}, halo {t.halo}, "
          f"ramp {t.ramp}")
    print(f"  plate           cells x{t.wing_cells()} y{t.wing_rows()}, owned by "
          f"{t.wing_window()}")
    print(f"  cuts clear      x {t.cuts_clear_of_wing()} cells, "
          f"y {t.rows_clear_of_wing()} cells   (W124's discipline, both axes)")
    print(f"  structure       {W.NI_STRUCT} x {W.NJ_STRUCT} Q1, t/c = "
          f"{W.THICK / W.CHORD:.3g}, clamped at the leading edge")

    # the surface operator's own properties
    st = W.WingStructure()
    rng = np.random.default_rng(0)
    tr = rng.normal(size=W.N_STATION) * 0.3
    d_red = st.compliance @ tr
    d_exp = st.deflect(tr)
    reduced_gap = float(np.abs(d_red - d_exp).max() / np.abs(d_exp).max())
    st2 = W.WingStructure(e_star=2.0 * W.E_STAR)
    e_lin = float(np.abs(st2.compliance * 2.0 - st.compliance).max()
                  / np.abs(st.compliance).max())
    ev = np.linalg.eigvalsh(op["S_e"])
    print()
    print(f"  C symmetry (Betti)          {op['symmetry']:.3e} relative")
    print(f"  C condition number          {op['cond']:.3e}")
    print(f"  work vs geometric delta     {op['geometric_gap']:.3e} of ||C||")
    print(f"  reduced S_e vs the expert   {reduced_gap:.3e}")
    print(f"  exact linearity in E        {e_lin:.3e}  (bitwise)")
    print(f"  S_e eigenvalues             {ev.min():.4g} .. {ev.max():.4g}")

    u, v = _settled(out, out_dir)
    z = np.load(_field_path(out_dir))
    L = z["load"]
    tail = L[-N_SPIN // 4:]
    band = float((tail.max() - tail.min()) / abs(tail.mean()))
    print()
    print(f"  spin-up {N_SPIN} macro-steps; settled load {tail.mean():.6f}")
    print(f"  settled unsteadiness (peak-to-peak, last quarter) {band * 100:.3f}%")
    print("  **that band is the LEVEL every difference below is quoted against**")
    print("  W106: the pipeline's bitwise floor is exactly zero and bounds nothing")

    out["setup"] = dict(
        nx=W.NX, ny=W.NY, dx=W.DX, n_windows=t.n_windows, halo=t.halo,
        wing_cells=list(t.wing_cells()), wing_rows=list(t.wing_rows()),
        cuts_clear_x=t.cuts_clear_of_wing(), cuts_clear_y=t.rows_clear_of_wing(),
        wing_window=t.wing_window(),
        ni=W.NI_STRUCT, nj=W.NJ_STRUCT, thickness_over_chord=W.THICK / W.CHORD,
        e_star=W.E_STAR, e_ref=W.E_REF,
        C_symmetry=op["symmetry"], C_cond=op["cond"],
        geometric_gap=op["geometric_gap"], reduced_vs_expert=reduced_gap,
        e_linearity=e_lin, S_e_min=float(ev.min()), S_e_max=float(ev.max()),
        n_spin=N_SPIN, settled_load=float(tail.mean()), unsteadiness=band,
        structure_scope=W.STRUCTURE_SCOPE,
    )


# ---------------------------------------------------------------------------
# stage 2 -- compile, and W114
# ---------------------------------------------------------------------------


def stage_compile(out: dict, out_dir: str) -> None:
    print("== compile: W114 settled before the build, not during it ==")
    u, v = _settled(out, out_dir)
    rows = {}
    for name, kw in (("fixed-shape", dict(motion=False)),
                     ("deforming", dict(motion=True)),
                     ("fixed-shape-single", dict(motion=False,
                                                 tiling=W.SINGLE_TILING))):
        g, _e = W.build(u, v, **kw)
        r = compile_scheme(g)
        refs = [f"{d.layer}/{d.rule}" for d in r.decisions.refusals]
        dec = [f"{d.layer}/{d.rule}" for d in r.decisions.decertifications]
        env = "".join(s.value[0].upper() for s in r.envelope.values.values())
        rows[name] = dict(verdict=r.verdict.value, envelope=env,
                          refusals=refs, decertifications=dec)
        print(f"  {name:20s} {r.verdict.value:18s} E({env})  "
              f"{len(refs)} refusals, {len(dec)} decertifications")
        if refs:
            print(f"      refused at: {', '.join(refs)}")

    # -- the W114 handle, and the control that it did not switch R10 off ----
    g, _e = W.build(u, v, motion=False)
    r = compile_scheme(g)
    sole = [d for d in r.decisions if d.rule == "R10/sole-family"]
    assert sole and sole[0].verdict is Verdict.ADMIT
    print()
    print("  W114: L2/R10 admits the EMBEDDED structural agent --")
    print(f"      {sole[0].message[:200]}...")

    # The control.  Declare the structure with the FLUID's governing family and
    # R10 must refuse again: the narrowing is a check on the premise, not a way
    # of switching the rule off.  A rule with nothing to fire on is not a rule.
    import copy
    g2, _e2 = W.build(u, v, motion=False)
    for a in g2.agents:
        if a.agent_id == "STRUCT":
            caps = copy.copy(a.capabilities)
            object.__setattr__(caps, "governing_family",
                               "incompressible-navier-stokes-2d")
            a.capabilities = caps
    r2 = compile_scheme(g2)
    ctrl = [f"{d.layer}/{d.rule}" for d in r2.decisions.refusals]
    print()
    print("  the control -- the same graph with the structure declaring the")
    print(f"  FLUID's family: refusals {ctrl}")
    assert "L2/R10" in ctrl, ctrl
    print("  so the narrowing is a check on R10's own premise, not an escape")

    halo = [d for d in r.decisions if d.rule == "R10/halo"]
    print()
    print("  and the SECOND rule with the same scope defect, left open as W136:")
    print(f"      L2/{halo[0].rule} decertifies {halo[0].subject} -- an implicit")
    print("      agent with a nonzero stencil, whose halo is undecidable. There is")
    print("      no halo on this seam to be inadequate: Gamma is a PHYSICAL")
    print("      boundary of Omega_solid, not an artificial cut, and the structure")
    print("      is not tiled. Same premise, same gap, one rule along")

    out["compile"] = dict(
        rows=rows,
        r10_sole_family=sole[0].message,
        r10_control_refusals=ctrl,
        halo_decert=halo[0].message if halo else None,
        handle=W.R10_HANDLE,
    )


# ---------------------------------------------------------------------------
# stage 3 -- the seam's own operator
# ---------------------------------------------------------------------------


def _probe_seam(graph, seam_id, probe_state, budget=None):
    """`w127_ground_effect._probe_seam`, unchanged: the compiler's own transfer
    derivation and seam assembly, driven from outside a compile."""
    from atlas.compiler import _Context, _derive_transfer            # noqa: PLC2701
    from atlas.envelope import EnvelopeStamp
    from atlas.holes import HoleLedger
    from atlas.verdict import DecisionRecord

    ctx = _Context(graph=graph, budget=None,
                   probe_budget=budget or ProbeBudget(), references={},
                   record=DecisionRecord(), stamp=EnvelopeStamp(),
                   holes=HoleLedger(), probe_state=probe_state, depth=0)
    conn = graph.connection(seam_id)
    tr = _derive_transfer(ctx, conn)
    op = assemble_seam(graph, conn, tr, budget or ProbeBudget(), {},
                       expected_null_dim=conn.expected_null_dim,
                       probe_state=probe_state)
    return op, tr


def stage_probe(out: dict, out_dir: str) -> None:
    print("== probe: the FSI seam's own operator ==")
    u, v = _settled(out, out_dir)
    res = {}

    # -- support reach on a quasi-static elastic agent ---------------------
    st = W.WingStructure()
    reaches = {}
    for amp in (1.0, 1e-2, 1e-4):
        r = support_reach(st.respond, "wet:MECH", st.probe_base("wet:MECH"),
                          amplitude=amp)
        reaches[str(amp)] = dict(reach=r.reach, n=r.n, nonzero=r.nonzero,
                                 peak=r.peak)
        print(f"  support_reach on STRUCT at amplitude {amp:g}: "
              f"{r.nonzero} of {r.n} cells nonzero, reach {r.reach}, "
              f"declared radius x substeps = 1")
    print("  **W93 measured a learned operator's global receptive field and called")
    print("  it a fact about the ARCHITECTURE and not about the physics. This one")
    print("  is a fact about the physics: a quasi-static elliptic operator has a")
    print("  dense inverse, so a poke at one station moves every station. Same")
    print("  reading, opposite cause, and only the second instance in the vault**")
    res["support_reach"] = reaches

    # the fluid side, for the contrast
    g, e = W.build(u, v, motion=False, flux_mode="reaction")
    fw = e[W.DEFAULT_TILING.wing_window()]
    rf = support_reach(fw.respond, "wet:MECH", fw.probe_base("wet:MECH"),
                       amplitude=1e-2)
    print(f"  and the fluid side: {rf.nonzero} of {rf.n} nonzero, reach {rf.reach}")
    res["support_reach_fluid"] = dict(reach=rf.reach, n=rf.n, nonzero=rf.nonzero)

    # -- the three efforts, at a field-to-FIELD seam ----------------------
    onesided = {}
    for mode in W.FLUX_MODES:
        gm, _em = W.build(u, v, motion=False, flux_mode=mode)
        op, _tr = _probe_seam(gm, "wet", f"settled fixed-shape field, {mode}")
        blocks = {k: float(np.linalg.norm(b.S, 2)) for k, b in op.blocks.items()}
        fluid = [k for k in blocks if k != "STRUCT"][0]
        ratio = blocks["STRUCT"] / blocks[fluid]
        onesided[mode] = dict(fluid=blocks[fluid], structure=blocks["STRUCT"],
                              ratio=ratio, beta=op.beta,
                              content=float(operator_content(op.S)),
                              null_dim=op.null_dim, kappa=op.kappa,
                              one_sided=op.one_sided)
        print(f"  {mode:10s} ||S_STRUCT||/||S_fluid|| = {ratio:.4g}   "
              f"(fluid {blocks[fluid]:.4g}, structure {blocks['STRUCT']:.4g})   "
              f"beta = {op.beta!r}")
    print("  W97 closed a field-to-LUMPED seam by taking this ratio below one with")
    print("  section 4.1's co-normal. Here the structure's block is dense, SPD and")
    print("  conditioned at 7e8, so the one-sidedness is the STRUCTURE's and no")
    print("  choice of the FLUID's effort moves it -- which is the same measurement")
    print("  answering a different question one seam class along")
    res["one_sided"] = onesided

    # -- the null space, and a fifth row for the table --------------------
    gm, _em = W.build(u, v, motion=False, flux_mode="reaction")
    op, _tr = _probe_seam(gm, "wet", "settled fixed-shape field, reaction")
    sv = np.linalg.svd(op.S, compute_uv=False)
    b_st = op.blocks["STRUCT"].S
    sv_s = np.linalg.svd(b_st, compute_uv=False)
    print()
    print(f"  in the declared interface space (M = {op.dim_M} modes):")
    print(f"      assembled sigma_min/sigma_max = {sv.min() / sv.max():.4g}, "
          f"null_dim {op.null_dim} against an expected {op.expected_null_dim}")
    print(f"      structure sigma_min/sigma_max = {sv_s.min() / sv_s.max():.4g}")
    print("  n_0(Gamma) = 0, and the reason is a FIFTH row for the table: CS-9's")
    print("  solid-solid MECH seam on a FREE body has n_0 = 1, because a uniform")
    print("  normal velocity there is a rigid translation -- no strain, no")
    print("  reaction. This body is CLAMPED, so the same trace bends it and the")
    print("  reaction is nonzero in every direction of the trace space. It is the")
    print("  BOUNDARY CONDITION and not the physics, and CS-9's own row said so")
    res["null_space"] = dict(m=int(op.dim_M), null_dim=op.null_dim,
                             expected=op.expected_null_dim,
                             assembled_sv_ratio=float(sv.min() / sv.max()),
                             struct_sv_ratio=float(sv_s.min() / sv_s.max()),
                             passivity=op.passivity_defect,
                             shares={k: b.share for k, b in op.blocks.items()})
    print(f"      passivity defect on the assembled seam: {op.passivity_defect}")
    # Is the passivity defect an ORIENTATION artefact?  Test it rather than
    # assert it: flip the sign of one block and re-read the symmetric part.
    Sf_b = op.blocks[W.DEFAULT_TILING.wing_window()].S
    Ss_b = op.blocks["STRUCT"].S
    def _pass(M):
        sym = 0.5 * (M + M.T)
        lo = float(np.linalg.eigvalsh(sym).min())
        return lo, max(0.0, -lo)
    lo_sum, def_sum = _pass(Sf_b + Ss_b)
    lo_dif, def_dif = _pass(Ss_b - Sf_b)
    print(f"      as ASSEMBLED (S_fluid + S_STRUCT): lambda_min "
          f"{lo_sum:+.4e}, defect {def_sum:.4e}")
    print(f"      with the fluid block's sign flipped: lambda_min "
          f"{lo_dif:+.4e}, defect {def_dif:.4e}")
    res["orientation"] = dict(assembled_lambda_min=lo_sum,
                              assembled_defect=def_sum,
                              flipped_lambda_min=lo_dif,
                              flipped_defect=def_dif)
    print("  the two sides are declared EFFORT and both are: the fluid returns the")
    print("  load ON the surface and the structure the reaction it needs to hold")
    print("  it, so the interface residual is their DIFFERENCE. Lambda_M ADDS")
    print("  them, and E7's passivity reads the sum -- which is why it fails here")
    print("  and failed at CS-10's seam too. `Connection.orientation` exists for")
    print("  exactly this and nothing checks it")

    # -- the two experts' own cutoffs, which are not the same number -------
    #
    # `effective_resolution` is declared as `modes_for(N_STATION)` on BOTH sides,
    # which is the FLUID's measured cutoff carried onto a plate the fluid and the
    # structure both live on.  A bending plate's stiffness goes as the fourth
    # power of the wavenumber, so the structure has its own, far lower, cutoff --
    # above some mode it simply does not move -- and `derive_space=True` takes
    # the min of two declared numbers that were both derived from the fluid's.
    P = W.fourier_basis(W.N_STATION)
    Ss = op.blocks["STRUCT"].S
    Sf = op.blocks[W.DEFAULT_TILING.wing_window()].S
    diag_s = np.abs(np.diag(Ss))
    diag_f = np.abs(np.diag(Sf))
    above = [k for k in range(len(diag_s)) if diag_s[k] > 10.0 * diag_f[k]]
    k_equal = next((k for k in range(len(diag_s)) if diag_s[k] <= diag_f[k]),
                   None)
    print()
    print("  the two experts' own cutoffs, mode by mode (the declared "
          f"resolution is {op.dim_M} on BOTH sides):")
    print(f"      structure diagonal {diag_s[0]:.4g} .. {diag_s[-1]:.4g} "
          f"({diag_s[-1] / diag_s[0]:.3g}x across M)")
    print(f"      fluid     diagonal {diag_f[0]:.4g} .. {diag_f[-1]:.4g} "
          f"({diag_f[-1] / diag_f[0]:.3g}x)")
    print(f"      the structure dominates by 10x in {len(above)} of "
          f"{op.dim_M} modes; the two cross at mode {k_equal}")
    print("  **and the disparity is UNIFORM across the declared space, which is")
    print("  the reading this measurement was made to falsify.** A bending")
    print("  plate's stiffness goes as the fourth power of the wavenumber, so the")
    print("  obvious story is that the structure dominates the HIGH modes and the")
    print("  declared cutoff is the fluid's. It does not: both diagonals are flat")
    print("  to within 8% across M, the structure is ~5300x the fluid in EVERY")
    print("  mode, and the stiff structural modes are not in the declared space at")
    print("  all. So the one-sidedness is not a truncation artefact and no choice")
    print("  of interface space repairs it -- the structure is simply stiff**")
    res["cutoffs"] = dict(struct_diag=[float(x) for x in diag_s],
                          fluid_diag=[float(x) for x in diag_f],
                          dominated_modes=len(above), crossover=k_equal,
                          struct_span=float(diag_s[-1] / diag_s[0]),
                          fluid_span=float(diag_f[-1] / diag_f[0]))

    # -- operator drift: the interface deforms ----------------------------
    K = 10
    drift = {}
    for tag, motion in (("deforming", True), ("frozen", False)):
        r = W.FSIRollout(tiling=W.SINGLE_TILING, coupling="tight", motion=motion)
        a = r.run(steps=1, u0=u, v0=v)
        b = r.run(steps=K + 1, u0=u, v0=v)
        S = []
        for m in (a, b):
            gg, ee = W.build(m.u.detach().cpu().numpy(),
                             m.v.detach().cpu().numpy(),
                             motion=False, tiling=W.SINGLE_TILING,
                             delta=m.delta_final.detach().cpu().numpy())
            o, _t = _probe_seam(gg, "wet", f"{tag} at step")
            S.append(o.blocks[W.SINGLE_TILING.wing_window()].S)
        d = float(operator_drift(S[0], S[1]))
        travel = float(np.abs(b.delta_final.detach().cpu().numpy()
                              - a.delta_final.detach().cpu().numpy()).max())
        drift[tag] = dict(drift=d, travel=travel, travel_cells=travel / W.DX)
        print(f"  operator drift over K = {K}, {tag:10s}: {d:.4g}   "
              f"(the interface deformed {travel / W.DX:.3g} cells)")
    print("  Tier 0's STATIC band is 2.4e-4 to 3.5e-3 (W30); CS-10's moving")
    print("  interface read 3.77e-1 against 2.51e-2 on the same seam held still")
    res["drift"] = drift

    out["probe"] = res


# ---------------------------------------------------------------------------
# stage 4 -- the gate
# ---------------------------------------------------------------------------


def _march(tiling, coupling, u, v, steps, lag=1, e_star=W.E_STAR, motion=True):
    r = W.FSIRollout(tiling=tiling, coupling=coupling, motion=motion, lag=lag,
                     e_star=e_star)
    return r.run(steps=steps, u0=u, v0=v)


def stage_gate(out: dict, out_dir: str, steps: int = N_GATE) -> None:
    print(f"== gate: the referent against the split, {steps} macro-steps ==")
    u, v = _settled(out, out_dir)
    ny = float(W.FlexWing().n_hat[1])

    t0 = time.perf_counter()
    ref = _march(W.SINGLE_TILING, "tight", u, v, steps)
    print(f"  referent (tight, one window): tip {ref.tip[-1]:+.9f}  "
          f"({ref.tip[-1] * ny / W.DX:+.3f} vertical cells)  "
          f"[{time.perf_counter() - t0:.0f} s]")
    print(f"      minimum {ref.tip.min():+.9f} at step {int(np.argmin(ref.tip))}, "
          f"final {ref.tip[-1]:+.9f}")

    cols = {}
    hist = {"referent": [float(x) for x in ref.tip]}
    #: **The scale every difference is quoted against, and it is FIXED.**
    #: A pointwise relative error divides by a deflection that starts at zero, so
    #: its maximum lands on the release transient and measures the denominator.
    #: The settled tip deflection is a level: it is the quantity being compared,
    #: it does not move, and W106's discipline is that a difference is quoted
    #: against the quantity's own level rather than against a floor.
    scale = float(np.abs(ref.tip[-1]))

    def add(tag, res):
        d = (res.tip - ref.tip) / scale
        pw = (res.tip - ref.tip) / np.maximum(np.abs(ref.tip), 1e-300)
        k = int(np.argmax(np.abs(d)))
        nz = d[np.abs(d) > 0]
        sgn = int(np.sum(np.diff(np.sign(nz)) != 0)) if nz.size else 0
        hist[tag] = [float(x) for x in res.tip]
        cols[tag] = dict(
            tip_final=float(res.tip[-1]), max_rel=float(np.abs(d).max()),
            at_step=k, rel_final=float(d[-1]), sign_changes=sgn,
            max_pointwise=float(np.abs(pw).max()),
            at_step_pointwise=int(np.argmax(np.abs(pw))),
            settled_rel=float(np.abs(d[steps // 2:]).max()),
            load_final=float(res.load[-1]), wall_s=res.wall_s,
            newton_residual=float(res.inner_residual[-1]),
            u_max=float(res.u_max[-1]))
        print(f"  {tag:22s} tip {res.tip[-1]:+.9f}  max|dtip|/tip_settled "
              f"{np.abs(d).max():.4e} at {k:3d}  end {d[-1]:+.4e}  "
              f"2nd half {np.abs(d[steps // 2:]).max():.4e}  "
              f"sign changes {sgn}")

    # the zero-cut control, bitwise
    for lag in LAGS:
        add(f"lag {lag}, no cut", _march(W.SINGLE_TILING, "lagged", u, v,
                                         steps, lag=lag))
    add("lag 1, six windows", _march(W.DEFAULT_TILING, "lagged", u, v, steps))
    add("tight, six windows", _march(W.DEFAULT_TILING, "tight", u, v, steps))

    order = None
    vals = [cols[f"lag {l}, no cut"]["max_rel"] for l in LAGS]
    end = [abs(cols[f"lag {l}, no cut"]["rel_final"]) for l in LAGS]
    if len(LAGS) >= 3:
        order = [_pow((LAGS[i], LAGS[i + 1]), (vals[i], vals[i + 1]))
                 for i in range(len(LAGS) - 1)]
        order_end = [_pow((LAGS[i], LAGS[i + 1]), (end[i], end[i + 1]))
                     for i in range(len(LAGS) - 1)]
        print(f"  lag defect log2 ratios, PEAK: "
              f"{', '.join(f'{o:.3f}' for o in order)}   (first order = 1)")
        print(f"  lag defect log2 ratios, END:  "
              f"{', '.join(f'{o:.3f}' for o in order_end)}")
    else:
        order_end = None

    lag1 = cols["lag 1, no cut"]["max_rel"]
    cut = cols["tight, six windows"]["max_rel"]
    print()
    print(f"  the seam's lag alone     {lag1:.4e}")
    print(f"  the six-window cut alone {cut:.4e}")
    print(f"  ratio                    {lag1 / cut:.4g}   "
          f"(CS-10's field-to-lumped seam: 97x)")
    print()
    print("  the crossing was LOOKED FOR, per Tier 23's standing rule: a")
    print("  comparison is not a result until it has been marched past the point")
    print("  where the two curves could cross")

    out["gate"] = dict(steps=steps, referent_tip=float(ref.tip[-1]),
                       referent_min=float(ref.tip.min()),
                       referent_argmin=int(np.argmin(ref.tip)),
                       referent_load=float(ref.load[-1]),
                       referent_wall_s=ref.wall_s,
                       columns=cols, lag_order=order,
                       lag_order_end=order_end,
                       lag_over_cut=lag1 / cut, scale=scale,
                       tip_history=hist)


def stage_zerocut(out: dict, out_dir: str) -> None:
    """The bitwise control, on its own so a failure is unmissable."""
    print("== zero-cut control: the composed path at one window, bitwise ==")
    u, v = _settled(out, out_dir)
    a = _march(W.SINGLE_TILING, "tight", u, v, 6)
    b = _march(W.SINGLE_TILING, "tight", u, v, 6)
    same = bool(np.array_equal(a.tip, b.tip))
    print(f"  the same code path twice: bit-identical {same}")
    out.setdefault("gate", {})["reproducible_bitwise"] = same


# ---------------------------------------------------------------------------
# stage 5 -- R
# ---------------------------------------------------------------------------


def _crossing_search(lo, hi, lo_tag: str, hi_tag: str) -> dict:
    """Tier 23's standing rule applied to an ORDERING of two curves.

    `R closes with the deformation term and does not close without it` is a
    comparison of two configurations, so it is not a result until it has been
    marched past the point where the two curves could cross -- and the crossing
    has to be LOOKED FOR.  A maximum over a march cannot see one: two curves can
    swap at every step and still have their maxima in the published order.
    """
    ratio = lo / np.maximum(hi, 1e-300)
    bad = np.nonzero(lo >= hi)[0]
    return dict(lo=lo_tag, hi=hi_tag, n_steps=int(lo.size),
                n_crossed=int(bad.size),
                first=int(bad[0]) if bad.size else None,
                steps=[int(x) for x in bad[:32]],
                worst_ratio=float(ratio.max()),
                worst_at=int(np.argmax(ratio)))


def _quartiles(a) -> list:
    q = max(1, a.size // 4)
    return [float(a[i * q:(i + 1) * q].max()) for i in range(4)]


def _keep_horizon(out: dict, key: str, rec: dict) -> None:
    """Record this horizon BESIDE the ones already measured, never over them.

    A quantity read off a rollout belongs to its horizon (spec section 0.4), so a
    longer march does not replace a shorter one's number -- it stands beside it,
    and the pair is the evidence about whether the number was horizon-dependent.
    """
    prev = out.get(key) or {}
    by = dict(prev.get("by_horizon") or {})
    if prev.get("steps") is not None:
        by.setdefault(str(prev["steps"]), {k: v for k, v in prev.items()
                                           if k != "by_horizon"})
    by[str(rec["steps"])] = rec
    out[key] = dict(rec, by_horizon=by)


def stage_residual(out: dict, out_dir: str, steps: int = 120) -> None:
    print(f"== residual: R across the FSI seam, {steps} macro-steps ==")
    u, v = _settled(out, out_dir)
    res = _march(W.SINGLE_TILING, "tight", u, v, steps)
    E = np.concatenate(([res.energy_0], res.strain_energy))
    P = res.interface_power
    dt = W.MACRO_DT
    #: An EXACT difference of the marched energy against the interface work
    #: accumulated over the same macro-step.  Not `np.gradient`: a centred
    #: stencil on a transient whose timescale is ~15 macro-steps is an O(dt^2)
    #: error in the quantity being closed, and the residual then measures the
    #: stencil. Measured, on this same march: 7.2e-2 by centred difference
    #: against a once-per-macro-step power sample.
    dE = np.diff(E) / dt
    H = res.half_step
    lvl = float(np.abs(dE).max())
    with_motion = float(np.abs(dE - P).max() / lvl)
    without = float(np.abs(dE).max() / lvl)
    settled = float(np.abs(dE - P)[steps // 2:].max() / lvl)
    # the half-step term, which is the backward-Euler-against-midpoint difference
    # of the energy accounting and is EXACT algebra rather than a defect of the
    # bond: at a converged interface solve f = S_e delta_new while the energy
    # gain is S_e delta_mid . d(delta), and the difference is 1/2 dd^T S_e dd
    corrected = float(np.abs(dE - (P - H)).max() / lvl)
    corrected_settled = float(np.abs(dE - (P - H))[steps // 2:].max() / lvl)
    print(f"  d/dt[1/2 delta^T S_e delta]  against  integral_Gamma f_n w_n")
    print(f"      level |dE/dt|_max                  {lvl:.4e}")
    print(f"      relative residual WITH the deformation term    {with_motion:.4e}")
    print(f"      relative residual WITHOUT it (static accounting) {without:.4e}")
    print(f"      factor {without / with_motion:.4g}")
    print(f"      over the settled second half                  {settled:.4e}")
    print(f"      with the HALF-STEP term, whole march          {corrected:.4e}")
    print(f"      with the HALF-STEP term, settled half         {corrected_settled:.4e}")
    print("  the residual left on the transient is the backward-Euler-against-")
    print("  midpoint difference of the ENERGY accounting and not the bond: at a")
    print("  converged interface solve the fluid's effort is the structure's")
    print("  reaction at the END of the exchange while the energy gain is taken")
    print("  at its MIDDLE, and the difference is exactly 1/2 dd^T S_e dd.")
    print("  Measured rather than argued -- subtracting it is what the last two")
    print("  lines do, and it is the whole of what is left")
    print("  CS-9 section 6's rule: the right residual is the RECEIVING")
    print("  subsystem's own balance, because a global R is blind to the coupling")
    print("  under test by six orders")

    # ------------------------------------------------------------------
    # THE CROSSING SEARCH.  Added at the CS-12 verification pass, 2026-09-04.
    # The three lines above are MAXIMA, and a maximum cannot see a crossing:
    # the published ordering can hold on the maxima while the two curves swap
    # at most of the steps under them.  So the per-step curves are compared.
    # ------------------------------------------------------------------
    a_without = np.abs(dE) / lvl
    a_with = np.abs(dE - P) / lvl
    a_corr = np.abs(dE - (P - H)) / lvl
    cross = [_crossing_search(a_with, a_without, "with_motion", "without_motion"),
             _crossing_search(a_corr, a_with, "corrected", "with_motion")]
    print()
    for c in cross:
        print(f"  {c['lo']} >= {c['hi']} at {c['n_crossed']} of {c['n_steps']} "
              f"steps; worst instantaneous ratio {c['worst_ratio']:.4e} at step "
              f"{c['worst_at']}")
    qw, qc = _quartiles(a_with), _quartiles(a_corr)
    print(f"  |dE/dt - P| quartile maxima      " + "  ".join(f"{x:.4e}" for x in qw))
    print(f"  with the half-step term          " + "  ".join(f"{x:.4e}" for x in qc))
    print("  the quartiles are what say whether a `settled half` was settled")

    _keep_horizon(out, "residual", dict(
        steps=steps, level=lvl, with_motion=with_motion,
        without_motion=without, settled=settled,
        corrected=corrected, corrected_settled=corrected_settled,
        factor=without / with_motion,
        crossings=cross,
        quartiles_with=qw, quartiles_corrected=qc,
        last_quarter_with=qw[-1], last_quarter_corrected=qc[-1],
        half_step=[float(x) for x in H],
        energy=[float(x) for x in E],
        power=[float(x) for x in P]))


# ---------------------------------------------------------------------------
# stage 6 -- added mass
# ---------------------------------------------------------------------------


def _aero_stiffness(u, v, e_star=W.E_STAR):
    """``K_aero = d f_aero / d delta`` on a FROZEN field, and the ratio it sets."""
    r = W.FSIRollout(tiling=W.SINGLE_TILING, coupling="tight", motion=True,
                     e_star=e_star)
    ut = torch.as_tensor(u, dtype=W.TORCH_DTYPE)
    vt = torch.as_tensor(v, dtype=W.TORCH_DTYPE)
    N = W.N_STATION
    d0 = torch.zeros(N, dtype=W.TORCH_DTYPE)
    wp = torch.zeros(N, dtype=W.TORCH_DTYPE)

    def f_of(d):
        f, _ = r.aero_traction(ut, vt, d, wp)
        return f.numpy()

    eps = 1e-5
    K = np.zeros((N, N))
    for k in range(N):
        dp = d0.clone(); dp[k] += eps
        dm = d0.clone(); dm[k] -= eps
        K[:, k] = (f_of(dp) - f_of(dm)) / (2 * eps)
    return K, f_of(d0)


def _try_march(tiling, coupling, u, v, steps, lag=1, e_star=W.E_STAR):
    """March, and CLASSIFY the failure rather than lumping the two together.

    A run that leaves the structural expert's small-strain envelope is that
    expert **declining**; a run that goes non-finite is the **scheme** diverging.
    They look identical from outside a try block and they mean opposite things:
    reading the first as the second is how a partitioned-coupling stability
    boundary gets invented out of a constitutive one.

    **And `undiverged` is not `stable`, which is why it returns a growth
    record.** A cell that finishes the march is evidence about the horizon it was
    marched to and nothing more; a slowly growing oscillation reads as `ok` at
    every horizon before the one where it reaches the bound. So each cell reports
    the amplitude of its own tip oscillation over successive quarters, the
    headroom it has left to the constitutive envelope, and the ratio between the
    two -- which is a quantity that can falsify `stable` where a survival flag
    cannot.
    """
    try:
        r = _march(tiling, coupling, u, v, steps, lag=lag, e_star=e_star)
        tip = np.asarray(r.tip)
        q = max(1, tip.size // 4)
        amp = [float(tip[i * q:(i + 1) * q].max() - tip[i * q:(i + 1) * q].min())
               for i in range(4)]
        mean = [float(tip[i * q:(i + 1) * q].mean()) for i in range(4)]
        peak = float(np.abs(r.delta).max())
        diag = dict(
            steps=int(tip.size), amp=amp, mean=mean,
            growth=float(amp[3] / amp[2]) if amp[2] > 0 else None,
            max_abs_delta=peak, envelope=float(W.DELTA_MAX),
            headroom=float(W.DELTA_MAX / peak) if peak > 0 else None,
            tip_min=float(tip.min()), tip_max=float(tip.max()),
            u_max=float(np.asarray(r.u_max).max()))
        return True, "ok", float(tip[-1]), None, diag
    except RuntimeError as exc:
        msg = str(exc)
        if "envelope" in msg:
            return False, "envelope", None, exc, {}
        if "not finite" in msg:
            return False, "blowup", None, exc, {}
        return False, "other", None, exc, {}


def stage_addedmass(out: dict, out_dir: str, steps: int = 80) -> None:
    print("== added mass: the canonical partitioned-FSI problem, at rank 32 ==")
    u, v = _settled(out, out_dir)
    K, f0 = _aero_stiffness(u, v)
    op = W._surface_operator()
    print(f"  K_aero (frozen field): ||K||_2 = {np.linalg.norm(K, 2):.4g}, "
          f"diagonal to {np.abs(K - np.diag(np.diag(K))).max() / np.abs(K).max():.2e}")

    ratios = {}
    for e in (5.0e4, 4.0e4, 3.0e4, 2.0e4, 1.0e4, 5.0e3, 4.0e3, 3.0e3, 2.0e3):
        Se = op["S_e"] * (e / W.E_REF)
        mu = float(np.abs(np.linalg.eigvals(np.linalg.solve(Se, K))).max())
        deq = np.linalg.solve(Se - K, f0)
        ratios[f"{e:.6g}"] = dict(mu=mu, tip_eq=float(deq[-1]),
                                  inside_envelope=bool(np.abs(deq).max()
                                                       <= W.DELTA_MAX))
        print(f"  E* = {e:9.3g}   mu = {mu:.4f}   equilibrium tip "
              f"{deq[-1]:+.5f}   inside the small-strain envelope: "
              f"{np.abs(deq).max() <= W.DELTA_MAX}")
    mu_ref = ratios[f"{W.E_STAR:.6g}"]["mu"]
    e_div = W.E_STAR * mu_ref
    print()
    print(f"  mu = 1 at E* = {e_div:.4g}: the DIVERGENCE boundary. At it the")
    print(f"  equilibrium deflection is unbounded by definition, so it is outside")
    print(f"  the linear structural model's own small-strain envelope by")
    print(f"  construction. **That is a statement about linear aeroelastic")
    print(f"  divergence, not about this construction**")
    print("  FLUTTER is NOT reachable here at all: a quasi-static structure has no")
    print("  mass, so it has no eigenfrequencies and no two modes to couple")

    # -- the three couplings ----------------------------------------------
    print()
    print("  the three couplings, at E* = %.4g (mu = %.4f):" % (W.E_STAR, mu_ref))
    three = {}
    for coupling in ("tight", "lagged", "staggered"):
        ok, why, tip, exc, diag = _try_march(W.SINGLE_TILING, coupling, u, v,
                                             steps)
        three[coupling] = dict(ok=ok, why=why, tip=tip,
                               error=(str(exc)[:400] if exc else None), **diag)
        if ok:
            print(f"      {coupling:10s} OK, tip {tip:+.6f}   amplitude q3->q4 "
                  f"{diag['amp'][2]:.3e} -> {diag['amp'][3]:.3e}  "
                  f"(x{diag['growth']:.3f})   envelope headroom "
                  f"{diag['headroom']:.3g}x")
        else:
            print(f"      {coupling:10s} {why.upper()}: {str(exc)[:110]}")

    # -- the update form's mechanism, priced ------------------------------
    Se = op["S_e"] * (W.E_STAR / W.E_REF)
    d1 = np.linalg.solve(Se, f0)
    w_implied = float(np.abs(d1).max() / W.MACRO_DT)
    load_ratio = float((0.5 * W.C_N * w_implied ** 2)
                       / np.abs(f0).mean())
    print()
    print(f"  the update form, priced: moving a massless structure the whole way")
    print(f"  to its own equilibrium inside one macro-step implies a surface")
    print(f"  velocity of {w_implied:.3g} U_inf and a load {load_ratio:.4g}x the")
    print(f"  one it was balancing. CS-10 measured 5.77 U_inf and a factor of ~70")
    print(f"  at a LUMPED seam; the mechanism is the same and does not depend on")
    print(f"  the partner being lumped -- section 4.2's rule, confirmed at a field seam")

    # -- the stability map -------------------------------------------------
    print()
    print("  the stability map: loose Gauss-Seidel at (E*, lag)")
    #: **The map's own horizon, and it is per cell.** A cell at lag `L` holds its
    #: interface answer for `L` macro-steps, so a march of `N` steps gives it
    #: `N/L` exchanges of the interface equation -- at the published `N = 80` the
    #: lag-32 cell got TWO AND A HALF, which is a statement about arithmetic and
    #: not about stability.  `MIN_LAG_INTERVALS` is the floor every cell is
    #: marched to in its OWN clock, so the map is comparable across the row.
    grid = {}
    worst_growth = (None, 0.0)
    for e in E_SWEEP:
        row = {}
        for lag in LAG_SWEEP:
            n = max(steps, MIN_LAG_INTERVALS * lag)
            ok, why, tip, _exc, diag = _try_march(W.SINGLE_TILING, "lagged", u,
                                                  v, n, lag=lag, e_star=e)
            row[str(lag)] = dict(ok=ok, why=why, tip=tip,
                                 lag_intervals=n // lag, **diag)
            mark = {"ok": ".", "envelope": "e", "blowup": "X",
                    "other": "?"}[why]
            extra = ""
            if ok:
                extra = (f"  tip {tip:+.6f}  {n} steps = {n // lag:3d} lag "
                         f"intervals  amp x{diag['growth']:.3f}  headroom "
                         f"{diag['headroom']:.3g}x")
                if diag["growth"] and diag["growth"] > worst_growth[1]:
                    worst_growth = (f"E*={e:.6g} lag={lag}", diag["growth"])
            print(f"      E* = {e:8.3g}  lag {lag:3d}  {mark}{extra}")
        grid[f"{e:.6g}"] = row
    print("      legend: . stable   e left the small-strain envelope   "
          "X not finite")
    print(f"      worst amplitude growth over the map: {worst_growth[1]:.4f}x "
          f"at {worst_growth[0]}")
    print("  **`undiverged` is not `stable`.** Every cell is marched to at least")
    print(f"  {MIN_LAG_INTERVALS} exchanges of its OWN interface equation, and the")
    print("  discriminator is the amplitude of the tip oscillation over the last")
    print("  two quarters: a growth ratio above 1 that RISES with the lag is a")
    print("  scheme on its way to the bound, which a survival flag cannot see")
    print("  **the two failures are different and are recorded separately.** A")
    print("  run that leaves the constitutive envelope is the EXPERT declining;")
    print("  a run that goes non-finite is the SCHEME diverging, and reading the")
    print("  first as the second is how an added-mass boundary gets invented")

    _keep_horizon(out, "addedmass", dict(
        K_norm=float(np.linalg.norm(K, 2)),
        K_offdiag=float(np.abs(K - np.diag(np.diag(K))).max()
                        / np.abs(K).max()),
        ratios=ratios, mu_at_reference=mu_ref, e_divergence=float(e_div),
        three=three, implied_velocity=w_implied, load_ratio=load_ratio,
        stability=grid, steps=steps,
        min_lag_intervals=MIN_LAG_INTERVALS,
        worst_growth=worst_growth[1], worst_growth_cell=worst_growth[0]))


# ---------------------------------------------------------------------------
# stage 7 -- the horizon
# ---------------------------------------------------------------------------


def stage_horizon(out: dict, out_dir: str) -> None:
    print("== horizon: dJ/dE*, and spec section 0.4 ==")
    u, v = _settled(out, out_dir)
    rows = []
    for N in HORIZONS:
        r = W.FSIRollout(tiling=W.SINGLE_TILING, coupling="tight", motion=True)
        e = torch.tensor(float(W.E_STAR), dtype=W.TORCH_DTYPE, requires_grad=True)
        t0 = time.perf_counter()
        j = r.objective(e, N, u0=u, v0=v, grad=True)
        j.backward()
        g_adj = float(e.grad) * W.E_STAR          # d J / d ln E*, a dimensionless
        wall = time.perf_counter() - t0
        rows.append(dict(N=N, J=float(j.detach()), grad=g_adj, wall_s=wall))
        print(f"  N = {N:4d}   J = {float(j.detach()):.8f}   "
              f"dJ/dlnE* = {g_adj:+.6e}   [{wall:.0f} s]")
        # persisted as it lands: each row is minutes of adjoint and the tail of
        # this sweep is where the box runs out of memory
        out["horizon"] = dict(rows=rows, partial=True)
        _flush(out)

    # -- the finite-difference check, at the longest affordable horizon ----
    N_fd = HORIZONS[len(HORIZONS) // 2]
    r = W.FSIRollout(tiling=W.SINGLE_TILING, coupling="tight", motion=True)
    base = None
    fd = {}
    for rel in FD_STEPS:
        h = W.E_STAR * rel
        jp = float(r.objective(W.E_STAR + h, N_fd, u0=u, v0=v).detach())
        jm = float(r.objective(W.E_STAR - h, N_fd, u0=u, v0=v).detach())
        fd[str(rel)] = (jp - jm) / (2 * h) * W.E_STAR
        print(f"  central difference at N = {N_fd}, rel step {rel:g}: "
              f"{fd[str(rel)]:+.6e}")
    adj = [x["grad"] for x in rows if x["N"] == N_fd][0]
    best = min(fd.values(), key=lambda x: abs(x - adj))
    print(f"  adjoint at the same horizon:              {adj:+.6e}")
    print(f"  best agreement {abs(best - adj) / abs(adj):.3e} relative")

    lims = _horizon_limits([x["N"] for x in rows], [x["grad"] for x in rows])
    print()
    print(f"  horizon    = {rows[-1]['N']} macro-steps, dJ/dlnE* = "
          f"{lims['converged']:+.6e}")
    print(f"  N_sign     = {lims['n_sign']}")
    for tol, nn in lims["validity_limit"].items():
        txt = "NOT DETERMINED" if nn is None else f"N >= {nn}"
        print(f"  N_valid({float(tol) * 100:4.0f}%) = {txt}")
    print(f"  sign crossings: {lims['crossings']}")
    print()
    print("  spec section 0.4 makes these three binding on any quantity read off")
    print("  a rollout, and it makes a missing horizon a REFUSAL rather than a")
    print("  decertification: a bound without its lag has unknown tightness, and a")
    print("  sensitivity without its horizon can point the other way")

    out["horizon"] = dict(rows=rows, fd=fd, fd_horizon=N_fd,
                          fd_agreement=float(abs(best - adj) / abs(adj)),
                          **lims)


# ---------------------------------------------------------------------------
# stage 8 -- the record
# ---------------------------------------------------------------------------


def stage_constants(out: dict, out_dir: str) -> None:
    print("== constants: what is declared and what stays unmeasured ==")
    u, v = _settled(out, out_dir)
    g, _e = W.build(u, v, motion=False)
    r = compile_scheme(g)
    print(f"  verdict {r.verdict.value}")
    unmeasured = sorted(getattr(r, "unmeasured", ()) or ())
    print(f"  unmeasured: {unmeasured}")
    mc = W.MEASURED_STATIC
    print(f"  tau   = {mc.tau}  (both agents are real solvers and each is its own")
    print(f"          reference; E3 fails and lambda_ref is what keeps tau defined)")
    print(f"  sigma = unmeasured (W3), and W116 recurs: the FSI seam's own lag")
    print(f"          defect is a SPLITTING error in a relative tip deflection, not")
    print(f"          a transmission infidelity in interface power, and there is")
    print(f"          still no field for it")
    horizon = out.get("horizon", {})
    gate = out.get("gate", {})
    out["constants"] = dict(
        verdict=r.verdict.value, unmeasured=unmeasured,
        tau=mc.tau, sigma=None, C_mu=None, L=None,
        gate_lag_defect=gate.get("columns", {}).get("lag 1, no cut", {}).get("max_rel"),
        gate_cut_defect=gate.get("columns", {}).get("tight, six windows", {}).get("max_rel"),
        horizon=horizon.get("rows", [{}])[-1].get("N"),
        n_sign=horizon.get("n_sign"),
        n_valid=horizon.get("validity_limit"),
        note="MeasuredConstants carries tau = 0 and leaves sigma unmeasured; "
             "W116 recurs for the third time",
    )


# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_ROOT, "out", "w136"))
    ap.add_argument("--stages", nargs="*", default=list(STAGES), choices=STAGES)
    ap.add_argument("--gate-steps", type=int, default=N_GATE)
    ap.add_argument("--addedmass-steps", type=int, default=80)
    ap.add_argument("--residual-steps", type=int, default=120)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "w136.json")
    global _ARTIFACT_PATH
    _ARTIFACT_PATH = path
    # **Load before writing.**  CS-11's Tier 25 lesson: writing after every stage
    # is a protection only if the next invocation reads what it protected.
    out: dict = {}
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                out = json.load(fh)
            print(f"resuming from {path}: "
                  f"{sorted(k for k in out if k not in ('generated', 'stages', 'case', 'wall_s'))}")
        except (OSError, ValueError) as exc:
            print(f"could not read {path} ({exc}); starting fresh")
            out = {}
    out.update(generated=time.strftime("%Y-%m-%d %H:%M:%S"),
               stages=a.stages, case="wing-fsi (CS-12)")
    t_all = time.perf_counter()
    for s in a.stages:
        print()
        t0 = time.perf_counter()
        try:
            if s == "gate":
                stage_gate(out, a.out, steps=a.gate_steps)
                stage_zerocut(out, a.out)
            elif s == "addedmass":
                stage_addedmass(out, a.out, steps=a.addedmass_steps)
            elif s == "residual":
                stage_residual(out, a.out, steps=a.residual_steps)
            else:
                globals()[f"stage_{s}"](out, a.out)
        finally:
            out.setdefault("wall_s", {})[s] = time.perf_counter() - t0
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2, default=float)
        print(f"  [{s} done in {time.perf_counter() - t0:.0f} s]")
    print()
    print(f"total {time.perf_counter() - t_all:.0f} s; wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
