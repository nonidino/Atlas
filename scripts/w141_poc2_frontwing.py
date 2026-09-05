"""PoC 2: CS-10 and CS-12 assembled, and a constrained design search on the pair.

    python scripts/w141_poc2_frontwing.py [--out out/w141] [--stages ...]

CS-10 built a wing that RIDES and does not bend; CS-12 built the same wing
BENDING on a mount that does not move.  This assembles the two into one graph
with both surface seams live at once, differentiates the whole rollout in four
design knobs, and runs a **constrained** search -- maximise downforce subject to
a stress ceiling and a deflection ceiling -- against a population baseline, as
`poc1a-frozen-expert-results` did for the unconstrained layout problem.

Stages, in the order they depend on each other:

    setup       the thickness scaling the design box is drawn from -- including
                where the Q1 mesh's shear locking INVERTS the stress trend -- the
                surface operator and stress map, and a fixed-shape spin-up at the
                reference design whose settled unsteadiness is the LEVEL
                everything below is quoted against (W106)
    controls    THE ASSEMBLY'S OWN GATE.  Each parent is a limit of the joint
                interface system and both limits are measured: k -> inf must
                reproduce CS-12's 32-equation solve, S_e -> inf must reproduce
                CS-10's scalar one.  Plus the zero-cut bitwise control and the
                stress map against the expert's own solve
    compile     the assembled graph's verdicts, PER SEAM, which is what the demo's
                indicator reads; the R10 control; and whether any verdict moves
                over the design box
    gate        the referent against the split, marched past the point where the
                columns could cross, with BOTH seams' lag defects separated
    residual    R across both seams at once, with the crossing looked for
    gradient    the four-knob adjoint against central differences, the thickness
                difference step swept, and section 0.4's three fields per knob
    design      THE DELIVERABLE.  Constrained gradient search against a
                population baseline on the same objective, budget and penalty
    ablation    is the COUPLING doing the work?  The two seams solved in TURN
                against the joint system, and each seam frozen in turn

Every stage writes into one artifact, the artifact is written after EVERY stage,
and the next invocation LOADS it before writing -- CS-11's own mistake, which
`scripts/w136_wing_fsi.py` fixed and this inherits.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import torch                                                        # noqa: E402

from atlas import compile_scheme                                    # noqa: E402
from atlas.cases import front_wing as F                             # noqa: E402
from atlas.cases import ground_effect as G                          # noqa: E402
from atlas.cases import wing_fsi as W                               # noqa: E402
from atlas.verdict import Verdict                                   # noqa: E402

STAGES = ("setup", "controls", "compile", "gate", "residual", "cost",
          "gradient", "fdsweep", "archive", "design", "design_tight",
          "recompare", "penalty", "ablation")

#: **One CPU thread, and it is measured rather than assumed.**
#:
#: This workload is six 80x80 windows in float64 with a 33-square dense solve and
#: one FFT per exchange -- tensors small enough that OpenMP's fork/join per
#: operation costs more than the operation.  Measured by `stage_cost` on the box
#: this ran on, three times: a bare composed macro-step is 33.3-37.1 ms on ONE
#: thread and 63.4-73.3 ms on eight, a **1.9-2.0x slowdown** from adding cores,
#: and a gradient iterate costs 5.4-5.8 objective forwards.  Bitwise-checked
#: first: 12 macro-steps at one thread and at eight agree to the BIT on the load,
#: the ride height, the tip deflection and the peak von Mises, so the thread
#: count is a clock and not a variable.
#:
#: **The first version of these numbers said 5.8x and 9.47 and both were
#: artefacts** -- `fwd` timed a bare march and `adj` timed a taped,
#: back-propagated objective, unwarmed, over 8 macro-steps.  A ratio between two
#: differently-defined quantities is not a ratio.  Three warmed, matched runs
#: agree to 5%.
#:
#: **And it is why this does not want a GPU.** A workload that will not use eight
#: CPU cores is bound on per-operation overhead, not on arithmetic; an
#: accelerator adds launch latency and host synchronisation to the same operation
#: count. The ratio is what is quoted and not the milliseconds (W-`wall-clock`
#: discipline): the absolute numbers are a claim about a box in a power state.
DEFAULT_THREADS = 1

#: Fixed-shape, fixed-height spin-up before any coupled march.  CS-10 and CS-12
#: both use 120 and both measure the load still climbing at 40.
N_SPIN = 120

#: The gate's horizon.  CS-12's, so the two are read side by side.
N_GATE = 240

#: The objective's horizon.  CS-12 measured `N_valid(10%) = 80` for its own knob,
#: so 120 is inside the range where a sensitivity is within 10% of its converged
#: value -- and `stage_gradient` measures it again for all four knobs here rather
#: than inheriting it.
N_OBJ = 120

LAGS = (1, 2, 4)


def _pow(xs, ys):
    return math.log2(abs(ys[1]) / abs(ys[0])) / math.log2(xs[1] / xs[0])


def _crossing_search(lo, hi, lo_tag: str, hi_tag: str) -> dict:
    """An ORDERING of two curves is not a result until the crossing is sought."""
    ratio = lo / np.maximum(hi, 1e-300)
    bad = np.nonzero(lo >= hi)[0]
    return dict(lo=lo_tag, hi=hi_tag, n_steps=int(lo.size),
                n_crossed=int(bad.size),
                first=int(bad[0]) if bad.size else None,
                steps=[int(x) for x in bad[:32]],
                worst_ratio=float(ratio.max()), worst_at=int(np.argmax(ratio)))


def _quartiles(a) -> list:
    q = max(1, a.size // 4)
    return [float(a[i * q:(i + 1) * q].max()) for i in range(4)]


def _horizon_limits(ns, g) -> dict:
    """Spec section 0.4's three fields: the horizon, N_sign, N_valid."""
    ns = list(ns); g = list(g)
    cross = [ns[i + 1] for i in range(len(g) - 1)
             if g[i] * g[i + 1] < 0.0]
    conv = g[-1]
    lim = {}
    for tol in (0.10, 0.05):
        nn = None
        for i in range(len(ns)):
            if all(abs(x - conv) <= tol * abs(conv) for x in g[i:]):
                nn = ns[i]
                break
        lim[str(tol)] = nn
    return dict(converged=float(conv), crossings=cross,
                n_sign=(cross[0] if cross else ns[0]),
                n_sign_is_bound=not cross,
                validity_limit=lim)


# ---------------------------------------------------------------------------
# artifact plumbing
# ---------------------------------------------------------------------------

_ARTIFACT_PATH = None


def _flush(out: dict) -> None:
    """Write the artifact NOW, mid-stage.

    The driver already writes after every stage, in a `finally`, which
    protects a stage that finishes.  The design search is 47 minutes of
    the run and a crash inside it would lose all of it, so it flushes
    after each of its two columns as well.  A long run persists its state
    on the way through, not only at the end.
    """
    if _ARTIFACT_PATH:
        with open(_ARTIFACT_PATH, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=float)


def _field_path(out_dir: str) -> str:
    return os.path.join(out_dir, "settled.npz")


def _settled(out: dict, out_dir: str):
    p = _field_path(out_dir)
    if not os.path.isfile(p):
        raise SystemExit(f"{p} is missing: run `--stages setup` first")
    d = np.load(p)
    return d["u"], d["v"]


def _design(**kw) -> dict:
    d = dict(F.DESIGN_REF)
    d.update(kw)
    return d


def _clip(d: dict) -> dict:
    return {k: float(np.clip(v, *F.DESIGN_BOX[k])) for k, v in d.items()}


# ---------------------------------------------------------------------------
# stage 1 -- the design box, measured
# ---------------------------------------------------------------------------


def von_mises_np(sig: np.ndarray) -> np.ndarray:
    sx, sy, sxy = sig[..., 0], sig[..., 1], sig[..., 2]
    return np.sqrt(np.maximum(sx * sx - sx * sy + sy * sy + 3.0 * sxy * sxy, 0.0))


def stage_setup(out: dict, out_dir: str) -> None:
    print("== setup: the design box, measured, and the spin-up ==")
    tiling = F.DEFAULT_TILING
    print(f"  domain {F.NX}x{F.NY} at dx = 1/{int(round(1/F.DX))}, "
          f"{tiling.n_windows} windows of {tiling.wx}x{tiling.wy}, halo "
          f"{tiling.halo}")

    # -- the thickness scaling, which is where the box comes from ----------
    print()
    print("  the thickness scaling: ||S_e|| is the WRONG quantity and the")
    print("  response quantities are the right ones")
    ratios = (0.5, 0.65, 0.8, 1.0, 1.25, 1.6, 2.0)
    #: a uniform normal traction of CS-12's own settled magnitude: the shape a
    #: cantilever's design case has
    ny_hat = float(W.FlexWing().n_hat[1])
    trac = np.full(F.N_STATION, F.LOAD_REF / (F.CHORD * ny_hat))
    rows = []
    for r in ratios:
        tc = F.TC_REF * r
        op = F.surface_operator(tc)
        S = op["S_e"] * (F.E_STAR_REF / F.E_REF)
        d = np.linalg.solve(S, trac)
        vm = float(von_mises_np(np.einsum("k,kea->ea", trac,
                                          op["sigma_map"])).max())
        ev = np.linalg.eigvalsh(0.5 * (S + S.T))
        rows.append(dict(ratio=r, tc=tc, tip=float(d[-1]),
                         delta_max=float(np.abs(d).max()), vm=vm,
                         norm=float(np.linalg.norm(S, 2)),
                         eig_min=float(ev.min()), eig_max=float(ev.max())))
        print(f"    t/c = {tc:.4f}   tip {d[-1]:+.5e}   peak vM {vm:9.4g}   "
              f"||S_e|| {np.linalg.norm(S, 2):.4e}   eig_min {ev.min():.4g}")
    tip_p = [_pow((a["ratio"], b["ratio"]), (a["tip"], b["tip"]))
             for a, b in zip(rows[:-1], rows[1:])]
    vm_p = [_pow((a["ratio"], b["ratio"]), (a["vm"], b["vm"]))
            for a, b in zip(rows[:-1], rows[1:])]
    norm_p = [_pow((a["ratio"], b["ratio"]), (a["norm"], b["norm"]))
              for a, b in zip(rows[:-1], rows[1:])]
    print()
    print("    tip deflection exponent (beam theory -3): "
          + ", ".join(f"{p:+.3f}" for p in tip_p))
    print("    peak von Mises  exponent (beam theory -2): "
          + ", ".join(f"{p:+.3f}" for p in vm_p))
    print("    ||S_e||_2       exponent                  : "
          + ", ".join(f"{p:+.3f}" for p in norm_p))
    #: **The inversion, and it is what the box's lower bound is placed above.**
    #: Q1 elements lock in bending and the locking gets worse as the elements get
    #: more slender, so below some thickness the peak stress RISES with the
    #: thickness -- the wrong way -- and a search run through the inversion is
    #: optimising an element formulation.
    inv = [(rows[i]["tc"], rows[i + 1]["tc"]) for i, p in enumerate(vm_p) if p > 0]
    print(f"    von Mises trend INVERTS on t/c intervals: {inv}")
    print(f"    the design box puts its lower bound at t/c = "
          f"{F.DESIGN_BOX['tc'][0]}, above the inversion")
    print("    **||S_e||_2 is non-monotone in the thickness while every response")
    print("    quantity is monotone**: the norm is set by the STIFFEST mode,")
    print("    which on a two-element-deep Q1 mesh is a locked shear mode")

    # -- the spin-up -------------------------------------------------------
    print()
    print(f"  fixed-shape, fixed-height spin-up at the reference design, "
          f"{N_SPIN} macro-steps")
    r = F.FrontWingRollout(tiling=tiling, coupling="tight", motion=False)
    t0 = time.perf_counter()
    res = r.run(steps=N_SPIN, u0=None, v0=None)
    load, tip = res.settled()["load"], res.settled()["tip"]
    q = res.load[-N_SPIN // 4:]
    unsteady = float((q.max() - q.min()) / abs(q.mean()))
    print(f"    settled load {load:.9f}   u_max {res.u_max[-1]:.4f}   "
          f"[{time.perf_counter() - t0:.0f} s]")
    print(f"    settled unsteadiness over the last quarter: "
          f"{unsteady * 100:.3f}% peak-to-peak -- THE LEVEL (W106: the pipeline's")
    print(f"    bitwise floor is exactly zero and bounds nothing)")
    np.savez(_field_path(out_dir),
             u=res.u.detach().cpu().numpy(), v=res.v.detach().cpu().numpy())

    # -- does the plate stay clear of the cuts over the whole box? ---------
    lo, hi = F.DESIGN_BOX["h0"]
    rows_lo = int(math.floor(lo / F.DX))
    rows_hi = int(math.ceil((hi + F.CHORD * math.sin(F.ALPHA)) / F.DX))
    y_cut = tiling.stride                       # the y-overlap band's lower edge
    print()
    print(f"  W124, on both axes and over the WHOLE design box: the plate spans "
          f"rows {rows_lo}-{rows_hi} as h0 runs {lo}-{hi}; the y-overlap band "
          f"starts at row {y_cut}, clear by {y_cut - rows_hi} cells")
    print(f"    x: the plate spans cells {tiling.wing_cells()}, the x-overlap "
          f"bands are clear by {tiling.cuts_clear_of_wing()} cells")

    out["setup"] = dict(
        nx=F.NX, ny=F.NY, dx=F.DX, n_windows=tiling.n_windows,
        halo=tiling.halo, n_spin=N_SPIN,
        design_ref=dict(F.DESIGN_REF), design_box={k: list(v) for k, v in
                                                   F.DESIGN_BOX.items()},
        sigma_ceiling=F.SIGMA_CEIL, delta_ceiling=F.DELTA_CEIL,
        delta_envelope=F.DELTA_MAX,
        thickness=rows, tip_exponent=tip_p, vm_exponent=vm_p,
        norm_exponent=norm_p, vm_inversion=[list(x) for x in inv],
        settled_load=load, settled_tip=tip, unsteadiness=unsteady,
        rows_clear_y=int(y_cut - rows_hi),
        cells_clear_x=int(tiling.cuts_clear_of_wing()),
        thickness_scope=F.THICKNESS_SCOPE)


# ---------------------------------------------------------------------------
# stage 2 -- the assembly's own gate: each parent is a limit
# ---------------------------------------------------------------------------


def stage_controls(out: dict, out_dir: str) -> None:
    print("== controls: each parent is a LIMIT of the joint interface system ==")
    u, v = _settled(out, out_dir)
    opt = dict(dtype=F.TORCH_DTYPE)
    ut, vt = torch.as_tensor(u, **opt), torch.as_tensor(v, **opt)
    S = F.N_STATION
    zeros = torch.zeros(S, **opt)
    zero = torch.zeros((), **opt)
    r = F.FrontWingRollout(tiling=F.SINGLE_TILING, coupling="tight")
    e = torch.tensor(F.E_STAR_REF, **opt)
    tc = torch.tensor(F.TC_REF, **opt)
    S_e, sig_map = r.structure(e, tc)
    rec: dict = {}

    # -- limit 1: k -> infinity is CS-12 -----------------------------------
    print()
    print("  LIMIT 1  k -> infinity: the mount cannot move, so the remaining 32")
    print("  equations must be `wing_fsi.FSIRollout.solve_interface` exactly")
    rw = W.FSIRollout(tiling=W.SINGLE_TILING, coupling="tight")
    h = torch.tensor(W.Y_MOUNT, **opt)
    kbig = torch.tensor(1.0e12, **opt)
    wd, vm, res, _ = r.solve_seams(ut, vt, zeros, h, zeros, zero, S_e, kbig,
                                   h + F.LOAD_REF / kbig)
    wp_ref, res_ref, _ = rw.solve_interface(ut, vt, zeros, zeros, e)
    d1 = float((wd - wp_ref).abs().max())
    scale1 = float(wp_ref.abs().max())
    print(f"    max|w_delta - CS-12's w_plate| = {d1:.4e}  against |w| = "
          f"{scale1:.4e}   relative {d1 / scale1:.3e}")
    print(f"    v_mount = {float(vm):.4e}   (the mount is held)")
    print(f"    the two residuals: joint {float(res):.6e}  CS-12 "
          f"{float(res_ref):.6e}")
    rec["k_to_infinity"] = dict(abs_diff=d1, scale=scale1, rel=d1 / scale1,
                                v_mount=float(vm), res=float(res),
                                res_cs12=float(res_ref))

    # -- limit 2: S_e -> infinity is CS-10 ---------------------------------
    print()
    print("  LIMIT 2  S_e -> infinity: the plate cannot bend, so the remaining")
    print("  scalar equation must be CS-10's, INCLUDING its analytic derivative")
    print("  -k dt - C_N sum |w| cos^2(alpha) ds")
    h10 = torch.tensor(F.H0_REF - F.LOAD_REF / F.K_REF, **opt)
    k10 = torch.tensor(F.K_REF, **opt)
    h010 = torch.tensor(F.H0_REF, **opt)
    S_big = S_e * 1.0e10
    wd2, vm2, res2, _ = r.solve_seams(ut, vt, zeros, h10, zeros, zero, S_big,
                                      k10, h010)
    # CS-10's own scalar Newton, on THESE stations, written out here so the
    # comparison is against the equation and not against a shared call
    wing = r.wing
    ny_hat = wing.n_hat[1]
    w_ext = wing.external_normal(ut, vt, zeros, r.ny, r.nx, h=h10)
    vp = torch.zeros((), **opt)
    for _ in range(r.n_inner):
        w = w_ext - vp * ny_hat
        load = -(wing.normal_traction(w) * ny_hat * r.ds).sum()
        rr = k10 * (h010 - h10 - r.dt_ex * vp) - load
        dr = -k10 * r.dt_ex - (wing.c_n * torch.abs(w) * ny_hat ** 2 * r.ds).sum()
        vp = vp - rr / dr
    d2 = float(abs(vm2 - vp))
    print(f"    |v_mount - CS-10's v_plate| = {d2:.4e}  against |v| = "
          f"{float(abs(vp)):.4e}   relative {d2 / max(float(abs(vp)), 1e-300):.3e}")
    print(f"    max|w_delta| = {float(wd2.abs().max()):.4e}   (the plate is rigid)")
    rec["S_to_infinity"] = dict(abs_diff=d2, scale=float(abs(vp)),
                                rel=d2 / max(float(abs(vp)), 1e-300),
                                w_delta_max=float(wd2.abs().max()),
                                res=float(res2))

    # -- the stress map against the expert's own solve ---------------------
    print()
    print("  the stress map is the EXPERT's own answer, not a second model")
    op = F.surface_operator(F.TC_REF)
    q = np.linspace(0.3, 1.0, F.N_STATION)
    mapped = np.einsum("k,kea->ea", q, op["sigma_map"])
    T = np.full(op["mesh"].n_nodes, W.T_REF)
    _uu, direct = op["ts"].solve_mechanical(T, 0.5 * q, -0.5 * q, T_ref=W.T_REF,
                                            clamp_nodes=op["clamp"])
    d3 = float(np.abs(mapped - direct).max() / np.abs(direct).max())
    print(f"    sum_k q_k sigma_map[k]  vs  solve_mechanical(q): {d3:.4e}")
    print(f"    (W139: `WingStructure.stress` is NOT this -- it re-solves at")
    print(f"    E_ref under the traction that would produce delta at E_ref, so")
    print(f"    it returns (E_ref/E*) times the stress at the state asked about)")
    rec["stress_map"] = dict(rel=d3)

    # -- the zero-cut control, bitwise -------------------------------------
    print()
    print("  the zero-cut control: the composed path at ONE window, twice")
    a = F.FrontWingRollout(tiling=F.SINGLE_TILING, coupling="tight").run(
        steps=6, u0=u, v0=v)
    b = F.FrontWingRollout(tiling=F.SINGLE_TILING, coupling="tight").run(
        steps=6, u0=u, v0=v)
    same = bool(np.array_equal(a.tip, b.tip) and np.array_equal(a.h, b.h)
                and np.array_equal(a.load, b.load))
    print(f"    bit-identical: {same}  -- a CONTROL and not a floor (W106)")
    rec["reproducible_bitwise"] = same

    ok = (rec["k_to_infinity"]["rel"] < 1e-7
          and rec["S_to_infinity"]["rel"] < 1e-7
          and rec["stress_map"]["rel"] < 1e-9 and same)
    print()
    print(f"  ASSEMBLY GATE: {'PASS' if ok else 'FAIL'} -- a composition that "
          f"does not reduce to its parts is not an assembly")
    rec["pass"] = ok
    out["controls"] = rec


# ---------------------------------------------------------------------------
# stage 3 -- the compile, per seam
# ---------------------------------------------------------------------------


GRAPH_SUBJECTS = ("<graph>", "<assembly>", "<run>", "<scheme>")


def seam_verdicts(graph, result) -> dict:
    """**Every seam's own verdict, from the compiler's own decisions.**

    This is what the demo's indicator reads and it is not a display choice: a
    `Decision` carries a `subject`, which is an agent, a seam id or a
    ``AGENT.PORT`` pair, and nothing in this package had ever grouped them by
    seam.  A decision reaches a seam when its subject IS the seam, or is one of
    the two ports the seam pairs, or is one of the two agents the seam joins.
    Graph-level subjects reach EVERY seam, which is why nothing is green.
    """
    by_seam: dict[str, dict] = {}
    for c in graph.connections:
        ports = {f"{c.a[0]}.{c.a[1]}", f"{c.b[0]}.{c.b[1]}"}
        agents = {c.a[0], c.b[0]}
        rows = {"refuse": [], "admit-uncertified": [], "global": []}
        for d in result.decisions.decisions:
            if d.verdict is Verdict.ADMIT:
                continue
            subs = {s.strip() for s in str(d.subject or "").split(",")}
            tag = f"{d.layer}/{d.rule}"
            if subs & ({c.seam_id} | ports | agents):
                rows[d.verdict.value].append(tag)
            elif subs & set(GRAPH_SUBJECTS):
                rows["global"].append(tag)
        verdict = ("refuse" if rows["refuse"]
                   else "admit-uncertified"
                   if (rows["admit-uncertified"] or rows["global"])
                   else "admit")
        by_seam[c.seam_id] = dict(
            verdict=verdict,
            colour={"refuse": "red", "admit-uncertified": "amber",
                    "admit": "green"}[verdict],
            kind=("fluid-structure" if c.seam_id == "wet"
                  else "field-lumped" if c.seam_id == "mount"
                  else "fluid-fluid"),
            a=f"{c.a[0]}.{c.a[1]}", b=f"{c.b[0]}.{c.b[1]}",
            refusals=sorted(set(rows["refuse"])),
            decertifications=sorted(set(rows["admit-uncertified"])),
            global_decertifications=sorted(set(rows["global"])))
    return by_seam


def stage_compile(out: dict, out_dir: str) -> None:
    print("== compile: the assembled graph, and every seam's own verdict ==")
    u, v = _settled(out, out_dir)
    rows: dict = {}
    seams: dict = {}
    for tag, motion in (("fixed-shape", False), ("riding", True)):
        g, _e = F.build(u, v, motion=motion)
        r = compile_scheme(g)
        refus = sorted({f"{d.layer}/{d.rule}" for d in r.decisions.refusals})
        dec = sorted({f"{d.layer}/{d.rule}" for d in r.decisions.decertifications})
        rows[tag] = dict(verdict=r.verdict.value,
                         envelope="".join(s.name[0] for s in r.envelope.stamp)
                         if hasattr(r.envelope, "stamp") else str(r.envelope),
                         refusals=refus, decertifications=dec,
                         unmeasured=sorted(getattr(r, "unmeasured", ()) or ()))
        seams[tag] = seam_verdicts(g, r)
        print(f"  {tag:14s} {r.verdict.value:18s} refusals {refus or '[]'}")
        for sid, s in seams[tag].items():
            print(f"      {sid:6s} {s['colour']:6s} {s['kind']:16s} "
                  f"refuse={s['refusals'] or '-'} "
                  f"decert={s['decertifications'] or '-'}")

    # -- W114's narrowing, in the ASSEMBLY, with its control ---------------
    print()
    print("  W114 in the assembly: SUSP declares the FLOW's governing family, so")
    print("  STRUCT is still the only plane-stress-elasticity-2d agent and R10's")
    print("  premise check still clears it.  The CONTROL declares STRUCT with the")
    print("  fluid's family and the refusal must come back")
    g_ctl, _ = F.build(u, v, motion=False,
                       struct_family="incompressible-navier-stokes-2d")
    r_ctl = compile_scheme(g_ctl)
    ctl = sorted({f"{d.layer}/{d.rule}" for d in r_ctl.decisions.refusals})
    print(f"    control verdict {r_ctl.verdict.value}, refusals {ctl}")
    assert "L2/R10" in ctl, ctl

    # -- does any verdict MOVE over the design box? ------------------------
    print()
    print("  does a seam's verdict move with the DESIGN?  The demo's indicator")
    print("  is live, so whether it can change has to be measured rather than")
    print("  implied by animating it")
    corners = []
    for e_star in F.DESIGN_BOX["e_star"]:
        for tc in F.DESIGN_BOX["tc"]:
            for k in F.DESIGN_BOX["k"]:
                for h0 in F.DESIGN_BOX["h0"]:
                    corners.append(_design(e_star=e_star, tc=tc, k=k, h0=h0))
    moved = {}
    base = None
    for d in corners:
        g, _e = F.build(u, v, motion=False, design=d, h=d["h0"] - F.LOAD_REF / d["k"])
        rr = compile_scheme(g)
        sv = {s: x["verdict"] for s, x in seam_verdicts(g, rr).items()}
        if base is None:
            base = sv
        elif sv != base:
            moved[json.dumps(d, sort_keys=True)] = sv
    print(f"    {len(corners)} corners of the design box: "
          f"{len(moved)} produce a different per-seam verdict map")
    if not moved:
        print("    **The verdict is a property of the DECLARATION, not of the")
        print("    design point** -- so the indicator's colours move when the")
        print("    interface starts moving and not when a knob turns, and the")
        print("    quantities behind them (the passivity defect, beta, the null")
        print("    ratio) are what move with the design.  Said plainly on the")
        print("    page rather than animated as if it were otherwise")

    out["compile"] = dict(rows=rows, seams=seams,
                          r10_control_refusals=ctl,
                          design_corners=len(corners),
                          verdict_moves_with_design=len(moved),
                          moved=list(moved)[:4])


# ---------------------------------------------------------------------------
# stage 4 -- the gate
# ---------------------------------------------------------------------------


def _march(tiling, coupling, u, v, steps, lag=1, design=None, motion=True):
    r = F.FrontWingRollout(tiling=tiling, coupling=coupling, motion=motion,
                           lag=lag, design=design)
    return r.run(steps=steps, u0=u, v0=v)


def stage_gate(out: dict, out_dir: str, steps: int = N_GATE) -> None:
    print(f"== gate: the referent against the split, {steps} macro-steps ==")
    u, v = _settled(out, out_dir)
    t0 = time.perf_counter()
    ref = _march(F.SINGLE_TILING, "tight", u, v, steps)
    print(f"  referent (tight, one window): load {ref.load[-1]:+.9f}  h "
          f"{ref.h[-1]:.9f}  tip {ref.tip[-1]:+.9f}  "
          f"[{time.perf_counter() - t0:.0f} s]")
    print(f"      h    min {ref.h.min():.9f} at {int(np.argmin(ref.h))}, "
          f"max {ref.h.max():.9f} at {int(np.argmax(ref.h))}")
    print(f"      tip  min {ref.tip.min():+.9f} at {int(np.argmin(ref.tip))}")

    #: **Two fixed scales, one per seam.**  A pointwise relative error divides by
    #: a deflection that starts at zero and its maximum lands on the release
    #: transient, measuring the denominator -- CS-12's rule, and here there are
    #: two quantities to apply it to.
    scale_tip = float(abs(ref.tip[-1]))
    scale_h = float(abs(ref.h[-1]))
    cols, hist = {}, {"referent": dict(tip=[float(x) for x in ref.tip],
                                       h=[float(x) for x in ref.h],
                                       load=[float(x) for x in ref.load])}

    def add(tag, res):
        row = {}
        for key, cur, base, sc in (("tip", res.tip, ref.tip, scale_tip),
                                   ("h", res.h, ref.h, scale_h)):
            d = (cur - base) / sc
            nz = d[np.abs(d) > 0]
            row[key] = dict(
                max_rel=float(np.abs(d).max()), at_step=int(np.argmax(np.abs(d))),
                rel_final=float(d[-1]),
                settled_rel=float(np.abs(d[steps // 2:]).max()),
                sign_changes=int(np.sum(np.diff(np.sign(nz)) != 0)) if nz.size else 0)
        row.update(tip_final=float(res.tip[-1]), h_final=float(res.h[-1]),
                   load_final=float(res.load[-1]),
                   vm_max=float(res.vm_max.max()), wall_s=res.wall_s,
                   newton_residual=float(res.inner_residual[-1]))
        cols[tag] = row
        hist[tag] = dict(tip=[float(x) for x in res.tip],
                         h=[float(x) for x in res.h],
                         load=[float(x) for x in res.load])
        print(f"  {tag:22s} tip {res.tip[-1]:+.9f} h {res.h[-1]:.7f}   "
              f"d(tip) {row['tip']['max_rel']:.4e}@{row['tip']['at_step']:3d} "
              f"({row['tip']['sign_changes']} crossings)   "
              f"d(h) {row['h']['max_rel']:.4e}@{row['h']['at_step']:3d} "
              f"({row['h']['sign_changes']} crossings)")

    for lag in LAGS:
        add(f"lag {lag}, no cut", _march(F.SINGLE_TILING, "lagged", u, v, steps,
                                         lag=lag))
    add("lag 1, six windows", _march(F.DEFAULT_TILING, "lagged", u, v, steps))
    add("tight, six windows", _march(F.DEFAULT_TILING, "tight", u, v, steps))

    order = {}
    for key in ("tip", "h"):
        vals = [cols[f"lag {l}, no cut"][key]["max_rel"] for l in LAGS]
        order[key] = [_pow((LAGS[i], LAGS[i + 1]), (vals[i], vals[i + 1]))
                      for i in range(len(LAGS) - 1)]
        print(f"  lag defect log2 ratios in {key:3s}: "
              + ", ".join(f"{o:.3f}" for o in order[key]) + "   (first order = 1)")

    ratio = {k: cols["lag 1, no cut"][k]["max_rel"]
                / cols["tight, six windows"][k]["max_rel"] for k in ("tip", "h")}
    print()
    for k in ("tip", "h"):
        print(f"  {k:3s}: the seams' lag alone {cols['lag 1, no cut'][k]['max_rel']:.4e}"
              f"   the six-window cut alone "
              f"{cols['tight, six windows'][k]['max_rel']:.4e}"
              f"   ratio {ratio[k]:.4g}")
    print("  CS-10 measured 97x at its field-to-lumped seam and CS-12 237x at")
    print("  its field-to-field one; this graph carries BOTH at once")
    print("  the crossing was LOOKED FOR, per Tier 23's standing rule")

    out["gate"] = dict(steps=steps, columns=cols, lag_order=order,
                       lag_over_cut=ratio, scale_tip=scale_tip, scale_h=scale_h,
                       referent=dict(tip=float(ref.tip[-1]), h=float(ref.h[-1]),
                                     load=float(ref.load[-1]),
                                     tip_min=float(ref.tip.min()),
                                     h_min=float(ref.h.min()),
                                     h_argmin=int(np.argmin(ref.h)),
                                     tip_argmin=int(np.argmin(ref.tip)),
                                     wall_s=ref.wall_s),
                       history=hist)


# ---------------------------------------------------------------------------
# stage 5 -- R across BOTH seams
# ---------------------------------------------------------------------------


def stage_residual(out: dict, out_dir: str, steps: int = 480) -> None:
    print(f"== residual: R across BOTH seams, {steps} macro-steps ==")
    u, v = _settled(out, out_dir)
    res = _march(F.SINGLE_TILING, "tight", u, v, steps)
    #: The receiving subsystems' own balance -- CS-9 section 6's rule -- and here
    #: there are TWO receivers, so the energy is the sum of the structure's
    #: strain energy and the spring's, and the power is the interface power over
    #: the whole plate, which both seams share.
    #: **Both receivers' energies at RELEASE, and the spring's is not zero.**
    #: The plate starts flat so its strain energy at release is identically zero,
    #: which makes it easy to carry one initial value and not the other -- and
    #: the suspension is already carrying the load at release by construction, so
    #: its energy is the largest single number in the balance.  Taking
    #: `spring_energy[0]` (the value AFTER the first macro-step) as the value
    #: BEFORE it put the first step's residual at 1.81 against a static
    #: accounting's 1.00, on a march whose every other step closed to 5e-8.
    E = np.concatenate(([res.energy_0 + res.spring_energy_0],
                        res.strain_energy + res.spring_energy))
    P = res.interface_power
    H = res.half_step
    dE = np.diff(E) / F.MACRO_DT
    lvl = float(np.abs(dE).max())
    a_without = np.abs(dE) / lvl
    a_with = np.abs(dE - P) / lvl
    a_corr = np.abs(dE - (P - H)) / lvl
    print(f"  level |dE/dt|_max = {lvl:.6e}")
    print(f"    without the motion terms (a static accounting) "
          f"{a_without.max():.4e}")
    print(f"    with them                                      "
          f"{a_with.max():.4e}")
    print(f"    with the half-step term as well                "
          f"{a_corr.max():.4e}")
    cross = [_crossing_search(a_with, a_without, "with_motion", "without_motion"),
             _crossing_search(a_corr, a_with, "corrected", "with_motion")]
    for c in cross:
        print(f"    {c['lo']} >= {c['hi']} at {c['n_crossed']} of {c['n_steps']}"
              f" steps; worst ratio {c['worst_ratio']:.4e} at {c['worst_at']}")
    qw, qc = _quartiles(a_with), _quartiles(a_corr)
    print("    quartile maxima, with  : " + "  ".join(f"{x:.4e}" for x in qw))
    print("    quartile maxima, corr  : " + "  ".join(f"{x:.4e}" for x in qc))
    print("  W140's discipline: a quantity read over a WINDOW is quoted with the")
    print("  window and with the evidence the window is what it is called")
    out["residual"] = dict(
        steps=steps, level=lvl, without_motion=float(a_without.max()),
        with_motion=float(a_with.max()), corrected=float(a_corr.max()),
        settled=float(a_with[steps // 2:].max()),
        last_quarter=qw[-1], factor=float(a_without.max() / a_with.max()),
        crossings=cross, quartiles_with=qw, quartiles_corrected=qc,
        energy=[float(x) for x in E], power=[float(x) for x in P],
        half_step=[float(x) for x in H])


# ---------------------------------------------------------------------------
# stage 5b -- what a step costs, and why there is no GPU column
# ---------------------------------------------------------------------------


def stage_cost(out: dict, out_dir: str) -> None:
    print("== cost: what a step costs, and whether more cores or a GPU help ==")
    u, v = _settled(out, out_dir)
    import torch as _t

    def fwd(n, steps=10):
        _t.set_num_threads(n)
        r = F.FrontWingRollout(tiling=F.DEFAULT_TILING, coupling="tight")
        r.run(steps=2, u0=u, v0=v)                       # warm
        t0 = time.perf_counter()
        r.run(steps=steps, u0=u, v0=v)
        return (time.perf_counter() - t0) / steps

    #: **The pair that governs the gradient-vs-population comparison, matched.**
    #:
    #: The first version of this stage timed a TAPED, back-propagated objective
    #: against a BARE march -- `run()`, which computes none of the objective's
    #: quantities -- and did it over 8 macro-steps with no warm-up, so the
    #: operator cache miss and torch's first-call cost landed on the adjoint
    #: alone.  It read 9.47, and the design stage's own end-to-end premium over
    #: 30 iterates of 120 macro-steps is nothing like that.  A ratio between two
    #: differently-defined quantities is not a ratio.
    #:
    #: What a population evaluation actually pays is `objective(grad=False)` --
    #: margins, stress map and all -- and what an Adam iterate pays is
    #: `objective(grad=True)` plus the reverse sweep.  Both warmed, both at the
    #: same step count, and the design stage's wall-clock is the check on it.
    def obj(n, want_grad, steps=20):
        _t.set_num_threads(n)

        def once(k):
            r = F.FrontWingRollout(tiling=F.DEFAULT_TILING, coupling="tight")
            if not want_grad:
                r.objective(dict(F.DESIGN_REF), steps=k, u0=u, v0=v, grad=False)
                return
            lv = {q: _t.tensor(float(F.DESIGN_REF[q]), dtype=F.TORCH_DTYPE,
                               requires_grad=True) for q in F.DESIGN_KEYS}
            J, gs, gd, _a, _b = r.objective(lv, steps=k, u0=u, v0=v, grad=True)
            _t.autograd.grad(F.penalised(J, gs, gd), list(lv.values()),
                             allow_unused=True)

        once(2)                                          # warm
        t0 = time.perf_counter()
        once(steps)
        return (time.perf_counter() - t0) / steps

    cores = os.cpu_count() or 1
    counts = [n for n in (1, 2, 4, 8) if n <= cores]
    fwd_ms = {n: fwd(n) * 1e3 for n in counts}
    obj_fwd_ms = {n: obj(n, False) * 1e3 for n in (counts[0], counts[-1])}
    adj_ms = {n: obj(n, True) * 1e3 for n in (counts[0], counts[-1])}
    for n in counts:
        print(f"  bare march,      {n:2d} thread(s)   {fwd_ms[n]:8.1f} "
              f"ms/macro-step   vs 1 thread "
              f"{fwd_ms[counts[0]] / fwd_ms[n]:5.2f}x")
    for n in obj_fwd_ms:
        print(f"  objective fwd,   {n:2d} thread(s)   {obj_fwd_ms[n]:8.1f} "
              f"ms/macro-step   <- what a POPULATION evaluation pays")
    for n, ms in adj_ms.items():
        print(f"  + reverse sweep, {n:2d} thread(s)   {ms:8.1f} "
              f"ms/macro-step   <- what an ADAM iterate pays")
    ratio = adj_ms[counts[0]] / obj_fwd_ms[counts[0]]
    bare_ratio = adj_ms[counts[0]] / fwd_ms[counts[0]]
    print(f"  one gradient iterate costs {ratio:.2f} objective forwards "
          f"(CS-10 measured 4.3-6.7 on its own column)")
    print(f"    and {bare_ratio:.2f} BARE marches, which is the number the first")
    print("    version of this stage reported -- a different quantity, and not")
    print("    the one the design search amortises")
    print()
    print("  **More cores make this SLOWER, and that is the answer to the GPU")
    print("  question.** Six 80x80 windows in float64 with a 33-square dense")
    print("  solve and one FFT per exchange are tensors small enough that")
    print("  OpenMP's fork/join per operation costs more than the operation. A")
    print("  workload that will not use eight CPU cores is bound on per-operation")
    print("  overhead rather than on arithmetic, and an accelerator adds launch")
    print("  latency and host synchronisation to the same operation count.")
    print("  The RATIO is the claim; the milliseconds are a claim about a box in")
    print("  a power state, and the PoC 1a demo's own table was wrong by 4x the")
    print("  next time anyone measured it.")

    # -- and the thread count must not be a variable ----------------------
    _t.set_num_threads(counts[0])
    a = F.FrontWingRollout(tiling=F.DEFAULT_TILING, coupling="tight").run(
        steps=12, u0=u, v0=v)
    _t.set_num_threads(counts[-1])
    b = F.FrontWingRollout(tiling=F.DEFAULT_TILING, coupling="tight").run(
        steps=12, u0=u, v0=v)
    _t.set_num_threads(DEFAULT_THREADS)
    same = bool(np.array_equal(a.load, b.load) and np.array_equal(a.h, b.h)
                and np.array_equal(a.tip, b.tip)
                and np.array_equal(a.vm_max, b.vm_max))
    print()
    print(f"  the same 12 macro-steps at {counts[0]} and {counts[-1]} threads: "
          f"bit-identical {same}")
    print("  -- checked BEFORE the thread count was changed, because a threaded")
    print("  reduction can reorder a floating-point sum and this run's every")
    print("  number would then carry that as its level")

    out["cost"] = dict(
        cores=cores, forward_ms={str(k): v for k, v in fwd_ms.items()},
        objective_forward_ms={str(k): v for k, v in obj_fwd_ms.items()},
        adjoint_ms={str(k): v for k, v in adj_ms.items()},
        #: the number the gradient-vs-population comparison amortises
        adjoint_over_forward=ratio,
        #: the same numerator over a BARE march, kept because the first version
        #: of this stage published it and the two must not be confused
        adjoint_over_bare_march=bare_ratio,
        thread_slowdown=fwd_ms[counts[-1]] / fwd_ms[counts[0]],
        threads_bitwise=same, threads_used=DEFAULT_THREADS,
        note="the ratio is the claim; the milliseconds are a claim about a box "
             "in a power state. `adjoint_over_forward` is measured against the "
             "objective's own forward -- margins, stress map and all, which is "
             "what a population evaluation pays -- warmed and at a matched step "
             "count. Against a bare march() it reads "
             f"{bare_ratio:.2f}, which is a different quantity.")


# ---------------------------------------------------------------------------
# stage 6 -- the gradient
# ---------------------------------------------------------------------------


HORIZONS = (20, 40, 80, 120, 160, 240)


def _objective(design, steps, u, v, grad_keys=(), tiling=None):
    """``(J, g_sigma, g_delta, true_sigma, true_delta, leaves)``.

    The first pair of margins is the SMOOTH maximum the penalty is built on; the
    second is the TRUE maximum feasibility is judged by.  Both are carried
    because `logsumexp` overshoots by ``temp * log(n_tail)`` and a constraint is
    not the place to leave that unmeasured.
    """
    tiling = tiling or F.DEFAULT_TILING
    leaves = {}
    d = {}
    for k, x in design.items():
        if k in grad_keys:
            t = torch.tensor(float(x), dtype=F.TORCH_DTYPE, requires_grad=True)
            leaves[k] = t
            d[k] = t
        else:
            d[k] = float(x)
    r = F.FrontWingRollout(tiling=tiling, coupling="tight", motion=True)
    J, gs, gd, ts, td = r.objective(d, steps=steps, u0=u, v0=v,
                                    grad=bool(grad_keys))
    return J, gs, gd, ts, td, leaves


def _grads(y, leaves, retain=True):
    g = torch.autograd.grad(y, list(leaves.values()), retain_graph=retain,
                            allow_unused=True)
    return {k: (0.0 if x is None else float(x)) for k, x in zip(leaves, g)}


def stage_gradient(out: dict, out_dir: str) -> None:
    print("== gradient: four knobs, the adjoint against central differences ==")
    u, v = _settled(out, out_dir)
    keys = F.DESIGN_KEYS
    base = dict(F.DESIGN_REF)

    # -- the thickness difference step, swept -----------------------------
    print()
    print("  the OPERATOR difference step for the thickness, swept.  The rollout")
    print("  is exactly differentiable in E*, k and h0 and finite-differenced in")
    print("  the thickness -- on the OPERATOR only, which is one scalar's worth of")
    print("  32 solves of a 198-dof system, evaluated away from any rollout")
    step_rows = []
    for st in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        F._OperatorOfThickness.STEP = st
        F.surface_operator.cache_clear() if hasattr(F.surface_operator,
                                                    "cache_clear") else None
        t = torch.tensor(F.TC_REF, dtype=F.TORCH_DTYPE, requires_grad=True)
        S, M = F.operator_of(t)
        (S.sum() + M.sum()).backward()
        step_rows.append(dict(step=st, grad=float(t.grad)))
        print(f"    step {st:g}   d(sum S + sum M)/d(t/c) = {float(t.grad):+.8e}")
    F._OperatorOfThickness.STEP = 1.0e-4

    # -- adjoint vs central difference on the composed objective -----------
    print()
    print(f"  the adjoint against a central difference of J, at N = {N_OBJ}")
    J, gs, gd, ts, td, leaves = _objective(base, N_OBJ, u, v, grad_keys=keys)
    adj_J = _grads(J, leaves)
    adj_s = _grads(gs, leaves)
    adj_d = _grads(gd, leaves, retain=True)
    j0 = float(J)
    print(f"    J = {j0:.9f}")
    print(f"    stress margin      smooth {float(gs):+.6f}   true "
          f"{float(ts):+.6f}   gap {float(gs) - float(ts):+.6f}")
    print(f"    deflection margin  smooth {float(gd):+.6f}   true "
          f"{float(td):+.6f}   gap {float(gd) - float(td):+.6f}")
    print("    the penalty is built on the SMOOTH pair and feasibility is")
    print("    judged on the TRUE pair; the gap is logsumexp's own overshoot")
    fd = {}
    for k in keys:
        rel = 1e-3 if k != "tc" else 1e-2
        h = rel * abs(base[k])
        jp, *_ = _objective(dict(base, **{k: base[k] + h}), N_OBJ, u, v)
        jm, *_ = _objective(dict(base, **{k: base[k] - h}), N_OBJ, u, v)
        c = (float(jp) - float(jm)) / (2 * h)
        fd[k] = dict(step=h, fd=c, adjoint=adj_J[k],
                     rel=abs(c - adj_J[k]) / max(abs(adj_J[k]), 1e-300))
        print(f"    dJ/d{k:7s} adjoint {adj_J[k]:+.6e}   central "
              f"{c:+.6e}   relative {fd[k]['rel']:.3e}")

    # -- section 0.4's three fields, per knob ------------------------------
    print()
    print("  spec section 0.4, PER KNOB: the horizon, N_sign and N_valid.  CS-10's")
    print("  dJ/dh0 and CS-11's dJ/dU both change sign with the horizon and")
    print("  CS-12's dJ/dE* does not, so this is measured and not inherited")
    rows = []
    for N in HORIZONS:
        Jn, _gs, _gd, _ts, _td, ln = _objective(base, N, u, v, grad_keys=keys)
        g = _grads(Jn, ln, retain=False)
        rows.append(dict(N=N, J=float(Jn), **{f"d_{k}": g[k] for k in keys}))
        print(f"    N = {N:4d}  J = {float(Jn):.8f}   "
              + "  ".join(f"d{k} {g[k]:+.4e}" for k in keys))
    lims = {}
    for k in keys:
        lims[k] = _horizon_limits([r["N"] for r in rows],
                                  [r[f"d_{k}"] for r in rows])
        nn = lims[k]
        txt = ("no crossing over the horizons marched"
               if nn["n_sign_is_bound"] else f"crossings at {nn['crossings']}")
        print(f"    {k:7s} N_sign {'<= ' if nn['n_sign_is_bound'] else '= '}"
              f"{nn['n_sign']}  ({txt});  N_valid(10%) = "
              f"{nn['validity_limit']['0.1']}, N_valid(5%) = "
              f"{nn['validity_limit']['0.05']}")

    out["gradient"] = dict(
        horizon=N_OBJ, J=j0, stress_margin=float(gs),
        deflection_margin=float(gd), true_stress_margin=float(ts),
        true_deflection_margin=float(td),
        smooth_gap=dict(sigma=float(gs) - float(ts),
                        delta=float(gd) - float(td)),
        adjoint_J=adj_J, adjoint_stress=adj_s, adjoint_deflection=adj_d,
        fd=fd, step_sweep=step_rows, rows=rows, limits=lims,
        worst_fd_agreement=max(fd[k]["rel"] for k in keys))


# ---------------------------------------------------------------------------
# stage 6b -- the finite-difference STEP, swept, per knob
# ---------------------------------------------------------------------------


#: Relative central-difference steps.  CS-10 swept five decades and reported a
#: truncation branch monotone over all of them with the cancellation branch not
#: reached; CS-12 swept one order.  A single step is not a check.
FD_STEPS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)


def stage_fdsweep(out: dict, out_dir: str, steps: int = N_OBJ) -> None:
    """**One finite-difference step is not a check, and this is why it matters
    here.**

    `stage_gradient` compares the adjoint against ONE central difference per
    knob, at a relative step of 1e-3, and two of the four came back at 4.4e-2 and
    1.1e-1 -- against CS-10's 1.3e-6 and CS-12's 1.3e-5.  Either the adjoint is
    wrong or the difference is, and a single step cannot tell them apart.

    The suspect is not the adjoint.  `FlexWing.forcing` places the plate's
    Gaussian kernels on an INTEGER stamping box whose corner is
    ``round(h/dx - 0.5)`` on a **detached** value -- a window on the lattice, not
    a parameter, and `DiskBank.forcing`'s own device.  That corner is a **step
    function of the ride height**: it is constant almost everywhere and jumps by
    one cell at each rounding boundary.  So the objective is **piecewise smooth**
    in ``h0``, the adjoint sees the smooth part exactly and is blind to the jump
    by construction, and a central difference whose two points STRADDLE a
    boundary measures the jump.

    The discriminator is the shape of the curve, not any one value: a genuine
    truncation branch falls monotonically as the step shrinks and then turns up
    into cancellation, while a quantisation artefact is erratic and does not
    improve with a smaller step until the step is small enough that no boundary
    is straddled.  This sweeps seven steps over three decades and reports both
    the best agreement and whether the branch is monotone.
    """
    print(f"== fdsweep: the finite-difference STEP, per knob, at N = {steps} ==")
    u, v = _settled(out, out_dir)
    base = dict(F.DESIGN_REF)
    J, _gs, _gd, _ts, _td, leaves = _objective(base, steps, u, v,
                                               grad_keys=F.DESIGN_KEYS)
    adj = _grads(J, leaves, retain=False)
    print(f"  J = {float(J):.9f}")
    rows = {}
    for k in F.DESIGN_KEYS:
        print(f"  d J / d {k}   adjoint {adj[k]:+.8e}")
        r = []
        for rel in FD_STEPS:
            h = rel * abs(base[k])
            jp, *_ = _objective(dict(base, **{k: base[k] + h}), steps, u, v)
            jm, *_ = _objective(dict(base, **{k: base[k] - h}), steps, u, v)
            c = (float(jp) - float(jm)) / (2 * h)
            e = abs(c - adj[k]) / max(abs(adj[k]), 1e-300)
            r.append(dict(rel=rel, step=h, fd=c, rel_err=e))
            print(f"      rel step {rel:8.1e}  central {c:+.8e}  "
                  f"relative {e:.3e}", flush=True)
        best = min(r, key=lambda x: x["rel_err"])
        errs = [x["rel_err"] for x in r]
        #: monotone down over the first half is the truncation branch's signature
        mono = all(errs[i] >= errs[i + 1] for i in range(len(errs) // 2))
        rows[k] = dict(adjoint=adj[k], sweep=r, best=best, monotone=mono)
        print(f"      best {best['rel_err']:.3e} at relative step "
              f"{best['rel']:g}; truncation branch monotone: {mono}")
    worst = max(rows[k]["best"]["rel_err"] for k in F.DESIGN_KEYS)
    print()
    print(f"  worst best-agreement over the four knobs: {worst:.3e}")
    print("  a knob whose error does NOT fall with the step is one whose")
    print("  objective is piecewise smooth -- the stamping box's integer corner")
    print("  moves by a cell -- and there the ADJOINT is the smooth branch's")
    print("  derivative and the difference is measuring the jump")
    out["fdsweep"] = dict(steps=steps, J=float(J), knobs=rows,
                          worst_best=worst,
                          fd_steps=[float(x) for x in FD_STEPS])


# ---------------------------------------------------------------------------
# the comparison arithmetic, re-derivable from the artifact alone
# ---------------------------------------------------------------------------


def _compare(design: dict) -> dict:
    """**Three ratios, and the flattering one is not the headline.**

    Re-derives the gradient-vs-population comparison from the two stored
    histories alone, so every ratio the results page quotes can be recomputed
    from `w141.json` without re-marching. `stage_design` calls this at the end
    and `stage_recompare` calls it on an artifact already on disk.

    The obvious ratio -- the population's whole budget divided by the gradient
    rollouts needed to match it -- **overstates the gradient**, because CMA-ES
    reached its own best partway through that budget and every evaluation after
    that bought nothing. So the honest rollout ratio divides by the evaluation
    at which the baseline actually peaked.

    And rollouts are not the currency anyone pays. **One Adam iterate costs a
    forward march AND a reverse sweep**; one population evaluation costs a
    forward march alone, and `stage_cost` measures the adjoint at an order of
    magnitude over the forward per macro-step. The wall-clock ratio is
    therefore several times smaller than the rollout ratio, and it is the one a
    practitioner is choosing between. All three are recorded; the page quotes
    all three.
    """
    g_hist = design.get("gradient", {}).get("history") or []
    pop_hist = design.get("population", {}).get("history") or []
    best_g = design.get("gradient", {}).get("best_feasible")
    best_p = design.get("population", {}).get("best_feasible")
    n_pop_used = len(pop_hist)

    #: the first gradient iterate that is BOTH feasible and at least as good as
    #: the baseline's best feasible point.  `it` is 0-based, so +1 is a count.
    reach = None
    if best_p:
        for h in g_hist:
            if h["feasible"] and h["J"] >= best_p["J"]:
                reach = h["it"] + 1
                break
    evals_to_best = (best_p or {}).get("n_evals")

    wall_g = design.get("gradient", {}).get("wall_s")
    wall_p = design.get("population", {}).get("wall_s")
    #: seconds per iterate on each side.  The gradient history carries one entry
    #: per rollout INCLUDING a final bare evaluation that pays no reverse sweep,
    #: so the divisor is the number of ADAM STEPS -- what `reach` counts -- and
    #: not the history's length.  Dividing by the length would make a gradient
    #: iterate look 3% cheaper than it is, which flatters the ratio below.
    n_iters = design.get("n_grad") or max(1, len(g_hist) - 1)
    s_per_grad = (wall_g / n_iters) if wall_g else None
    s_per_pop = (wall_p / n_pop_used) if (wall_p and n_pop_used) else None

    out = dict(
        rollouts_to_match=reach,
        pop_evals_used=n_pop_used,
        pop_evals_to_best=evals_to_best,
        #: what the driver printed: flattering, and kept so the page can say so
        ratio_full_budget=(n_pop_used / reach) if reach else None,
        #: the honest rollout ratio
        ratio_to_pop_best=(evals_to_best / reach) if (reach and evals_to_best)
        else None,
        s_per_grad_iterate=s_per_grad,
        s_per_pop_eval=s_per_pop,
        #: how many population evaluations one gradient iterate costs, in
        #: wall-clock.  This is the forward+adjoint premium, measured on this
        #: run rather than taken from `stage_cost`'s per-macro-step numbers.
        grad_iterate_in_pop_evals=(s_per_grad / s_per_pop)
        if (s_per_grad and s_per_pop) else None,
        ratio_wall=None,
        best_gradient_J=(best_g or {}).get("J"),
        best_population_J=(best_p or {}).get("J"),
        quality_ratio=((best_g["J"] / best_p["J"])
                       if (best_g and best_p and best_p["J"]) else None),
    )
    if reach and evals_to_best and s_per_grad and s_per_pop:
        out["ratio_wall"] = (evals_to_best * s_per_pop) / (reach * s_per_grad)
    return out


#: **W145's evidence, kept rather than overwritten.**
#:
#: `FrontWingRollout.run` enforced both experts' declared envelopes at every
#: macro-step and `objective` enforced neither -- and the design search runs on
#: `objective`. So the first constrained run walked the ride height down through
#: the suspension's declared floor and nothing declined: its reported optimum
#: leaves the floor at macro-step 4 of 120, and what found it was the ABLATION
#: crashing when it marched the same design through `run`.
#:
#: The run is not deleted. It is what happened, it is the evidence for the row,
#: and the difference between it and the enforced run is the measurement of what
#: the missing check was worth.
W145_ARCHIVE = {"design": "design_unenforced", "penalty": "penalty_unenforced"}


def stage_archive(out: dict, out_dir: str) -> None:
    """Move the pre-W145 search records aside so the re-run does not erase them.

    Idempotent: a second call is a no-op, because the destination already
    exists and the source is then whatever the enforced re-run wrote.
    """
    for src, dst in W145_ARCHIVE.items():
        if dst in out:
            print(f"  {dst} already present; leaving {src!r} alone")
            continue
        if src not in out:
            print(f"  {src} absent; nothing to archive")
            continue
        rec = out.pop(src)
        rec["note"] = (
            "THE PRE-W145 RUN, kept because it is the evidence. "
            "`FrontWingRollout.objective` did not enforce either expert's "
            "declared envelope -- only `run` did -- so this search optimised "
            "the ride height down through the suspension's declared floor of "
            f"{G.H_FLOOR:.7g} and nothing declined. Its reported optimum "
            "settles at h = 0.0350, which is 2.24 cells of gap against the "
            "4.8 CS-10 measured down to, and marching that same design through "
            "`run` raises at macro-step 4. Re-run with the check on the "
            "objective's own path."
        )
        out[dst] = rec
        print(f"  archived {src!r} -> {dst!r}")


def stage_recompare(out: dict, out_dir: str) -> None:
    """Recompute `design.comparison` from the histories already in the artifact.

    Costs no marching: it exists so a correction to the comparison arithmetic
    does not require re-running a 49-minute search, and so the numbers on the
    page are a function of the recorded run rather than of a print statement
    that has since been edited.
    """
    for key in ("design", "design_tight", "design_unenforced", "design_loose"):
        d = out.get(key)
        if not isinstance(d, dict) or "gradient" not in d:
            print(f"  {key}: absent, skipped")
            continue
        c = _compare(d)
        d["comparison"] = c
        print(f"  {key}:")
        print(f"    quality        gradient {c['best_gradient_J']}  vs  "
              f"population {c['best_population_J']}  "
              f"({c['quality_ratio']:.4f}x)" if c["quality_ratio"] else
              f"    quality        not both present")
        print(f"    rollouts       {c['rollouts_to_match']} to match a "
              f"baseline that peaked at evaluation {c['pop_evals_to_best']} "
              f"of {c['pop_evals_used']}")
        if c["ratio_to_pop_best"]:
            print(f"    ratio, rollouts to the baseline's PEAK   "
                  f"{c['ratio_to_pop_best']:.4g}x   "
                  f"(over its whole budget it would read "
                  f"{c['ratio_full_budget']:.4g}x, which is the flattering form)")
        if c["ratio_wall"]:
            print(f"    one gradient iterate costs "
                  f"{c['grad_iterate_in_pop_evals']:.3g} population "
                  f"evaluations in wall-clock")
            print(f"    ratio, WALL-CLOCK to the baseline's peak "
                  f"{c['ratio_wall']:.4g}x   <-- the number a practitioner pays")


# ---------------------------------------------------------------------------
# stage 7 -- the constrained design search
# ---------------------------------------------------------------------------


#: The penalty and its weight live in `atlas/cases/front_wing.py`, so the demo
#: and this driver optimise the SAME objective rather than two that agree by
#: inspection.  See `front_wing.penalised` for the two properties it declares.
PENALTY = F.PENALTY
_penalised = F.penalised


def _evaluate(design, steps, u, v, want_grad: bool, weight: float = PENALTY):
    """**Driven by the smooth margins, JUDGED by the true ones.**

    `logsumexp` overshoots the maximum it smooths by ``temp * log(n_tail)``, so a
    design called infeasible on the smooth margin may be feasible on the real
    one.  The penalty -- and therefore the gradient -- uses the smooth pair
    because a hard `max` has a subgradient that chatters; `feasible` uses the
    true pair, because that is the question.  The gap is recorded at every
    evaluation so the optimum can be reported with it.
    """
    keys = F.DESIGN_KEYS if want_grad else ()
    try:
        J, gs, gd, ts, td, leaves = _objective(design, steps, u, v,
                                               grad_keys=keys)
    except RuntimeError as exc:
        #: **An expert DECLINING is a legitimate outcome of a design, not a
        #: crash of the search.**  CS-12 section 7.1's rule: a run that
        #: leaves a constitutive envelope is the expert saying no, and a
        #: run that goes non-finite is the scheme diverging.  Either way
        #: the design is infeasible and the search must carry on -- a
        #: population sampler WILL propose designs outside an envelope,
        #: and losing the whole run to one of them is losing the run to a
        #: correct answer.
        msg = str(exc)
        why = ("envelope" if "envelope" in msg
               else "blowup" if "finite" in msg else "other")
        return dict(J=0.0, g_sigma=1.0, g_delta=1.0, true_sigma=1.0,
                    true_delta=1.0, smooth_gap_sigma=0.0,
                    smooth_gap_delta=0.0, L=-1.0e3, grad=None,
                    feasible=False, declined=why, error=msg[:200])
    L = _penalised(J, gs, gd, weight)
    g = _grads(L, leaves, retain=False) if want_grad else None
    return dict(J=float(J), g_sigma=float(gs), g_delta=float(gd),
                true_sigma=float(ts), true_delta=float(td),
                smooth_gap_sigma=float(gs) - float(ts),
                smooth_gap_delta=float(gd) - float(td),
                L=float(L), grad=g,
                feasible=bool(float(ts) <= 0.0 and float(td) <= 0.0))


def _unit(d: dict) -> np.ndarray:
    return np.array([(d[k] - F.DESIGN_BOX[k][0])
                     / (F.DESIGN_BOX[k][1] - F.DESIGN_BOX[k][0])
                     for k in F.DESIGN_KEYS])


def _from_unit(x) -> dict:
    return {k: float(F.DESIGN_BOX[k][0] + np.clip(x[i], 0.0, 1.0)
                     * (F.DESIGN_BOX[k][1] - F.DESIGN_BOX[k][0]))
            for i, k in enumerate(F.DESIGN_KEYS)}


def stage_design(out: dict, out_dir: str, steps: int = N_OBJ,
                 n_grad: int = 30, n_pop: int = 200, weight: float = PENALTY,
                 seed: int = 0) -> None:
    print(f"== design: constrained search, {steps}-macro-step objective ==")
    u, v = _settled(out, out_dir)
    print(f"  maximise the settled downforce subject to")
    print(f"      peak von Mises <= {F.SIGMA_CEIL:g}   and   "
          f"peak |delta| <= {F.DELTA_CEIL:g}")
    print(f"  both as MARGINS (value/ceiling - 1), so a feasible design is")
    print(f"  non-positive on both and the two are commensurable without a")
    print(f"  weight nobody measured.  Penalty weight {weight:g}, declared.")
    start = dict(F.DESIGN_REF)
    ev0 = _evaluate(start, steps, u, v, want_grad=False, weight=weight)
    print()
    print(f"  start {start}")
    print(f"    J = {ev0['J']:.8f}  stress margin {ev0['true_sigma']:+.5f}  "
          f"deflection margin {ev0['true_delta']:+.5f}  feasible "
          f"{ev0['feasible']}  (smooth-max gap "
          f"{ev0['smooth_gap_sigma']:+.4f}, {ev0['smooth_gap_delta']:+.4f})")

    # -- the gradient column ----------------------------------------------
    print()
    print(f"  ADAM on the adjoint, {n_grad} steps, in the box's own unit cube")
    x = _unit(start)
    m = np.zeros_like(x); vv = np.zeros_like(x)
    lr, b1, b2, eps = 0.06, 0.9, 0.999, 1e-8
    span = np.array([F.DESIGN_BOX[k][1] - F.DESIGN_BOX[k][0]
                     for k in F.DESIGN_KEYS])
    g_hist = []
    t_grad = time.perf_counter()
    for it in range(n_grad):
        d = _from_unit(x)
        ev = _evaluate(d, steps, u, v, want_grad=True, weight=weight)
        if ev.get("grad") is None:
            # the expert declined at this design: back off along the last
            # good direction rather than stepping on a gradient nobody has
            print(f"    {it:3d}  DECLINED ({ev.get('declined')}): "
                  f"{ev.get('error', '')[:90]}", flush=True)
            g_hist.append(dict(it=it, design=d, declined=ev.get("declined"),
                               J=0.0, feasible=False))
            x = np.clip(0.5 * (x + _unit(start)), 0.0, 1.0)
            continue
        g_unit = np.array([ev["grad"][k] for k in F.DESIGN_KEYS]) * span
        g_hist.append(dict(it=it, design=d, **{k: ev[k] for k in
                                               ("J", "g_sigma", "g_delta",
                                                "true_sigma", "true_delta",
                                                "L", "feasible")}))
        print(f"    {it:3d}  J {ev['J']:.7f}  L {ev['L']:.7f}  "
              f"sig {ev['true_sigma']:+.4f}  del {ev['true_delta']:+.4f}  "
              f"{'ok ' if ev['feasible'] else 'VIOL'}  "
              + "  ".join(f"{k}={d[k]:.4g}" for k in F.DESIGN_KEYS), flush=True)
        m = b1 * m + (1 - b1) * g_unit
        vv = b2 * vv + (1 - b2) * g_unit ** 2
        mh = m / (1 - b1 ** (it + 1)); vh = vv / (1 - b2 ** (it + 1))
        x = np.clip(x + lr * mh / (np.sqrt(vh) + eps), 0.0, 1.0)
    d_final = _from_unit(x)
    ev_g = _evaluate(d_final, steps, u, v, want_grad=False, weight=weight)
    g_hist.append(dict(it=n_grad, design=d_final,
                       **{k: ev_g[k] for k in ("J", "g_sigma", "g_delta",
                                               "true_sigma", "true_delta",
                                               "L", "feasible")}))
    wall_grad = time.perf_counter() - t_grad
    out["design"] = dict(state="gradient-done", steps=steps,
                        weight=weight, n_grad=n_grad,
                        gradient=dict(history=g_hist, final=d_final,
                                      wall_s=wall_grad))
    _flush(out)
    #: **Rollouts, not iterations.**  One Adam step costs one forward march plus
    #: one adjoint; the baseline pays one march per evaluation.  The comparable
    #: currency is what `poc1a-frozen-expert-results` used: how many composed
    #: rollouts each method spent.
    print(f"    gradient optimum: J = {ev_g['J']:.8f}  feasible "
          f"{ev_g['feasible']}  [{wall_grad:.0f} s, {n_grad} rollouts + "
          f"{n_grad} adjoints]")

    # -- the population baseline ------------------------------------------
    #
    #: **CMA-ES, the same optimiser `poc1a-frozen-expert-results` used**, on the
    #: SAME objective, the SAME penalty, the SAME box and a stated budget.  The
    #: first version of this stage rolled its own Gaussian sampler with a
    #: shrinking sigma, which is a weaker baseline than CMA-ES and would have
    #: flattered the gradient column; a comparison is only worth reporting
    #: against the strongest baseline available, and PoC 1a's is available.
    print()
    print(f"  the population baseline: CMA-ES, the SAME objective, the SAME")
    print(f"  penalty, the same box, {n_pop} evaluations")
    import cma
    pop_hist, best = [], None
    t_pop = time.perf_counter()
    x0 = _unit(start)
    es = cma.CMAEvolutionStrategy(
        list(x0), 0.30,
        {"bounds": [0.0, 1.0], "seed": seed + 1, "verbose": -9,
         "maxfevals": n_pop})
    gen = 0
    while not es.stop() and len(pop_hist) < n_pop:
        xs = es.ask()
        vals = []
        for xi in xs:
            di = _from_unit(np.asarray(xi))
            e = _evaluate(di, steps, u, v, want_grad=False, weight=weight)
            # CMA-ES MINIMISES, and the objective is a maximisation
            vals.append(-e["L"])
            pop_hist.append(dict(gen=gen, design=di,
                                 **{k: e[k] for k in ("J", "g_sigma", "g_delta",
                                                      "true_sigma", "true_delta",
                                                      "L", "feasible")}))
            if e["feasible"] and (best is None or e["J"] > best["J"]):
                best = dict(design=di, n_evals=len(pop_hist),
                            **{k: e[k] for k in ("J", "g_sigma", "g_delta",
                                                 "true_sigma", "true_delta",
                                                 "L", "feasible")})
        es.tell(xs, vals)
        m = _from_unit(np.asarray(es.result.xfavorite))
        print(f"    gen {gen:3d}  evals {len(pop_hist):4d}  best L "
              f"{max(-x for x in vals):.7f}  sigma {es.sigma:.4f}  mean "
              + "  ".join(f"{k}={m[k]:.4g}" for k in F.DESIGN_KEYS), flush=True)
        gen += 1
    wall_pop = time.perf_counter() - t_pop
    n_pop_used = len(pop_hist)
    out["design"]["population"] = dict(history=pop_hist, best_feasible=best,
                                       wall_s=wall_pop, optimiser="cma-es",
                                       generations=gen)
    _flush(out)

    # -- the comparison ----------------------------------------------------
    feas_g = [h for h in g_hist if h["feasible"]]
    best_g = max(feas_g, key=lambda h: h["J"]) if feas_g else None
    print()
    print("  the comparison, in ROLLOUTS, on the same objective and budget:")
    print(f"    start                          J = {ev0['J']:.8f}")
    if best_g:
        print(f"    gradient, best FEASIBLE        J = {best_g['J']:.8f}  "
              f"at iteration {best_g['it']}")
    if best:
        print(f"    population, best FEASIBLE      J = {best['J']:.8f}  "
              f"over {n_pop_used} evaluations")
    if best_g and best:
        print(f"    gradient/population            "
              f"{best_g['J'] / best['J']:.4f}x on J")
    #: rollouts to reach the population's own best, which is the number PoC 1a
    #: reports and is the only one that is a RATIO rather than a difference
    reach = None
    if best:
        for h in g_hist:
            if h["feasible"] and h["J"] >= best["J"]:
                reach = h["it"] + 1
                break
    print(f"    gradient rollouts to reach the population's best: "
          f"{reach if reach is not None else 'not reached'}")
    if reach:
        print(f"    ratio {n_pop_used / reach:.4g}x  "
              f"(PoC 1a measured 8.0x at 36 variables and >30.9x at 75, "
              f"against CMA-ES; this is 4 variables against CMA-ES)")
    if best_g:
        print()
        print(f"    IS EITHER CEILING ACTIVE AT THE OPTIMUM?  stress margin "
              f"{best_g['true_sigma']:+.4f}, deflection margin "
              f"{best_g['true_delta']:+.4f}")
        print(f"    (0 means the ceiling is exactly met; a large negative "
              f"number means it was never reached and the search was a BOX "
              f"search)")
        onbox = {k: best_g["design"][k] for k in F.DESIGN_KEYS}
        print("    optimum: " + "  ".join(f"{k}={onbox[k]:.5g}"
                                          for k in F.DESIGN_KEYS))

    out["design"] = dict(
        steps=steps, weight=weight, n_grad=n_grad, n_pop=n_pop_used, seed=seed,
        start=dict(start), start_eval={k: ev0[k] for k in
                                       ("J", "g_sigma", "g_delta", "true_sigma",
                                        "true_delta", "L", "feasible")},
        gradient=dict(history=g_hist, final=d_final,
                      final_eval={k: ev_g[k] for k in ("J", "g_sigma", "g_delta",
                                                       "true_sigma", "true_delta",
                                                       "L", "feasible")},
                      best_feasible=best_g, wall_s=wall_grad),
        population=dict(history=pop_hist, best_feasible=best, wall_s=wall_pop,
                        optimiser="cma-es"),
        rollouts_to_match=reach,
        ratio=(n_pop_used / reach) if reach else None,
        ceilings=dict(sigma=F.SIGMA_CEIL, delta=F.DELTA_CEIL),
        box={k: list(v) for k, v in F.DESIGN_BOX.items()},
        #: **Is either ceiling ACTIVE at the optimum?**  A constraint the search
        #: never reaches is not a constraint, and the first run's pair were not
        #: reached at all -- recorded as `design_loose`.  This is the check that
        #: says whether the word "constrained" is earned, and it is recorded
        #: rather than asserted.
        active=dict(
            sigma=(abs(best_g["true_sigma"]) < 0.02) if best_g else None,
            delta=(abs(best_g["true_delta"]) < 0.02) if best_g else None,
            sigma_margin=best_g["true_sigma"] if best_g else None,
            delta_margin=best_g["true_delta"] if best_g else None),
        on_box=({k: ("lo" if abs(best_g["design"][k] - F.DESIGN_BOX[k][0])
                     < 1e-6 * max(1.0, abs(F.DESIGN_BOX[k][0]))
                     else "hi" if abs(best_g["design"][k] - F.DESIGN_BOX[k][1])
                     < 1e-6 * max(1.0, abs(F.DESIGN_BOX[k][1])) else "interior")
                 for k in F.DESIGN_KEYS} if best_g else None))
    #: the honest comparison, recorded beside the flattering one -- see
    #: `_compare`.  Same call `stage_recompare` makes, so a re-derivation and a
    #: fresh run agree by construction.
    out["design"]["comparison"] = _compare(out["design"])


#: The stress ceiling for the TIGHT column.  See `stage_design_tight`.
SIGMA_CEIL_TIGHT = 180.0


def stage_design_tight(out: dict, out_dir: str, steps: int = N_OBJ,
                       n_grad: int = 30, n_pop: int = 200,
                       weight: float = PENALTY, seed: int = 0) -> None:
    """**The same search with the stress ceiling placed where it binds INSIDE
    the envelope.**

    With W145's envelope check live, the primary column is stopped by
    `Suspension.validity`'s floor and *neither* named ceiling is active -- the
    search reaches only 2.3% under the stress ceiling before the ride height
    runs out. That is a result about the model and it is reported as one. But it
    leaves the constrained-search capability undemonstrated on the constraints
    the run was specified with, so this column places the stress ceiling at 180
    -- below the ~195 the envelope lets the search reach -- and asks the same
    question again.

    **The ceiling is a declared design constraint and not a material property,
    and this is the second time it has been placed** (600 -> 200 after the first
    run turned out to be a box search, 200 -> 180 here). Section 2.2's rule is
    the whole justification and it is not weakened by being applied twice: a
    ceiling the search cannot reach is not a constraint, and where "cannot
    reach" now includes "the validity envelope stops it first". What must not
    happen is a ceiling moved until the *answer* is agreeable, so both prior
    placements are kept in the artifact with their outcomes.
    """
    primary = out.pop("design", None)
    ceil0 = F.SIGMA_CEIL
    print("== design_tight: the stress ceiling placed BELOW what the envelope "
          "lets the search reach ==")
    print(f"  sigma ceiling {ceil0:g} -> {SIGMA_CEIL_TIGHT:g}; everything else "
          f"identical")
    try:
        F.SIGMA_CEIL = SIGMA_CEIL_TIGHT
        stage_design(out, out_dir, steps=steps, n_grad=n_grad, n_pop=n_pop,
                     weight=weight, seed=seed)
        rec = out.pop("design")
        rec["note"] = (
            f"The same search with SIGMA_CEIL={SIGMA_CEIL_TIGHT:g} instead of "
            f"{ceil0:g}, so the stress ceiling binds INSIDE the suspension's "
            f"declared envelope. The primary column is stopped by the envelope "
            f"and neither named ceiling is active there; this column exists to "
            f"ask whether the search rides a ceiling when a ceiling is what it "
            f"meets first.")
        rec["sigma_ceiling"] = SIGMA_CEIL_TIGHT
        out["design_tight"] = rec
    finally:
        F.SIGMA_CEIL = ceil0
        if primary is not None:
            out["design"] = primary


# ---------------------------------------------------------------------------
# stage 7b -- what the penalty weight is worth
# ---------------------------------------------------------------------------


def stage_penalty(out: dict, out_dir: str, steps: int = N_OBJ,
                  n_grad: int = 8,
                  weights: tuple = (10.0, 40.0, 160.0)) -> None:
    """**An exterior penalty's optimum sits OUTSIDE the feasible set, and by how
    much is a property of the weight.**

    That is textbook and it is also visible in the main run: at weight 40 the
    gradient column converges to an iterate about 0.36% over the stress ceiling,
    and what is reported as the answer is the best strictly FEASIBLE iterate
    rather than the last one.  A weight is a declared number and its consequence
    should be measured rather than argued, so the gradient column is re-run at a
    16x range of weights and the residual violation is reported at each.

    Short columns: what is being measured is where each weight's equilibrium
    sits, not how good an optimum it reaches.
    """
    print(f"== penalty: where the exterior penalty's equilibrium sits ==")
    u, v = _settled(out, out_dir)
    start = dict(F.DESIGN_REF)
    span = np.array([F.DESIGN_BOX[k][1] - F.DESIGN_BOX[k][0]
                     for k in F.DESIGN_KEYS])
    rows = {}
    for w in weights:
        x = _unit(start)
        m = np.zeros_like(x); vv = np.zeros_like(x)
        lr, b1, b2 = 0.06, 0.9, 0.999
        hist = []
        for it in range(n_grad):
            ev = _evaluate(_from_unit(x), steps, u, v, want_grad=True, weight=w)
            if ev.get("grad") is None:
                #: back off exactly as the main loop does rather than `break`.
                #: Breaking truncated each weight's column at a different
                #: iteration and the sweep is a comparison BETWEEN weights, so
                #: unequal lengths would make it one.
                hist.append(dict(it=it, J=0.0, true_sigma=1.0, true_delta=1.0,
                                 feasible=False, declined=ev.get("declined")))
                x = np.clip(0.5 * (x + _unit(F.DESIGN_REF)), 0.0, 1.0)
                continue
            hist.append(dict(it=it, J=ev["J"], true_sigma=ev["true_sigma"],
                             true_delta=ev["true_delta"],
                             feasible=ev["feasible"]))
            g = np.array([ev["grad"][k] for k in F.DESIGN_KEYS]) * span
            m = b1 * m + (1 - b1) * g
            vv = b2 * vv + (1 - b2) * g * g
            x = np.clip(x + lr * (m / (1 - b1 ** (it + 1)))
                        / (np.sqrt(vv / (1 - b2 ** (it + 1))) + 1e-8), 0.0, 1.0)
        #: the violation is a property of the last iterate the experts actually
        #: ANSWERED at; a declined one carries a sentinel margin of 1.0 and
        #: reading it would report a 100% violation that nothing measured
        answered = [h for h in hist if not h.get("declined")]
        last = answered[-1] if answered else (hist[-1] if hist else {})
        feas = [h for h in hist if h["feasible"]]
        best = max(feas, key=lambda h: h["J"]) if feas else None
        rows[str(w)] = dict(history=hist, last=last, best_feasible=best,
                            n_declined=sum(1 for h in hist if h.get("declined")),
                            violation=max(0.0, last.get("true_sigma", 0.0)))
        tail = (f"   best feasible J {best['J']:.7f} at iteration {best['it']}"
                if best else "   no feasible iterate")
        print(f"  weight {w:6.1f}: last iterate J {last.get('J', float('nan')):.7f}"
              f"  stress margin {last.get('true_sigma', float('nan')):+.5f}"
              f"  -> violation {rows[str(w)]['violation'] * 100:.3f}% of the "
              f"ceiling{tail}", flush=True)
    print()
    print("  a heavier weight moves the equilibrium closer to the boundary from")
    print("  outside, which is what an exterior penalty does; what is REPORTED as")
    print("  the answer is the best strictly feasible iterate, at every weight")
    out["penalty"] = dict(steps=steps, n_grad=n_grad,
                          weights=[float(w) for w in weights], rows=rows)


# ---------------------------------------------------------------------------
# stage 8 -- is the COUPLING doing the work?
# ---------------------------------------------------------------------------


def stage_ablation(out: dict, out_dir: str, steps: int = N_OBJ) -> None:
    print("== ablation: is the coupling BETWEEN the two seams doing any work? ==")
    u, v = _settled(out, out_dir)
    base = dict(F.DESIGN_REF)
    rows = {}
    for tag, kw in (("joint (both seams, one system)", dict(coupling="tight")),
                    ("split (the two seams in turn)", dict(coupling="split"))):
        r = F.FrontWingRollout(tiling=F.SINGLE_TILING, motion=True,
                               design=base, **kw)
        res = r.run(steps=steps, u0=u, v0=v)
        s = res.settled()
        rows[tag] = dict(load=s["load"], h=s["h"], tip=s["tip"],
                         vm_max=s["vm_max"],
                         newton_residual=float(res.inner_residual[-1]),
                         newton_r0=float(res.inner_residual_0[-1]),
                         wall_s=res.wall_s)
        print(f"  {tag:34s} load {s['load']:.8f}  h {s['h']:.7f}  "
              f"tip {s['tip']:+.7f}  residual {res.inner_residual[-1]:.3e}")
    a, b = rows["joint (both seams, one system)"], rows["split (the two seams in turn)"]
    print(f"    the two schemes differ in the settled load by "
          f"{abs(a['load'] - b['load']) / abs(a['load']):.4e} and in the")
    print(f"    interface residual by {b['newton_residual'] / a['newton_residual']:.4g}x")

    # -- each seam frozen in turn ------------------------------------------
    both_h = rows["joint (both seams, one system)"]["h"]
    print()
    print("  and each seam frozen in turn, which is what the ASSEMBLY buys over")
    print("  either parent alone")
    #: the reference column is pinned at the height the LIVE reference
    #: design settles to, which is what the frozen column has to share
    frozen = _freeze_each_seam(base, steps, u, v, h_live=both_h)
    both = rows["joint (both seams, one system)"]
    print(f"    both seams live                        load {both['load']:.8f}  "
          f"h {both['h']:.7f}  tip {both['tip']:+.7f}")

    out["ablation"] = dict(steps=steps, schemes=rows, frozen=frozen,
                           at=dict(F.DESIGN_REF))

    # -- and the same ablation AT THE OPTIMUM ------------------------------
    #
    #: **An ablation at the reference design is a statement about the reference
    #: design.**  §7.2 found the elastic seam worth 4.1% of the downforce and the
    #: suspension seam 0.5%, and then wanted to say that a search which STIFFENS
    #: the wing shrinks the first -- which is an inference about a design point
    #: the ablation had not been run at.  It costs three rollouts to stop
    #: inferring it, so it is measured instead.
    opt = ((out.get("design") or {}).get("gradient") or {}).get("best_feasible")
    if not opt or "design" not in opt:
        print()
        print("  (no design optimum in the artifact yet -- run `--stages design` "
              "first to get the ablation AT the optimum)")
        return
    d_opt = {k: float(opt["design"][k]) for k in F.DESIGN_KEYS}
    print()
    print("  the SAME ablation at the constrained optimum, because 'the elastic")
    print("  seam is worth less at a stiffer design' is otherwise an inference")
    print("    optimum: " + "  ".join(f"{k}={d_opt[k]:.5g}" for k in F.DESIGN_KEYS))
    r = F.FrontWingRollout(tiling=F.SINGLE_TILING, coupling="tight",
                           motion=True, design=d_opt)
    res = r.run(steps=steps, u0=u, v0=v)
    s = res.settled()
    live = dict(load=s["load"], h=s["h"], tip=s["tip"], vm_max=s["vm_max"])
    print(f"    both seams live                        load {live['load']:.8f}  "
          f"h {live['h']:.7f}  tip {live['tip']:+.7f}")
    frozen_opt = _freeze_each_seam(d_opt, steps, u, v,
                                   h_live=live["h"])
    worth = {}
    for tag, rec in frozen_opt.items():
        if rec.get("ok"):
            worth[tag] = abs(rec["load"] - live["load"]) / abs(live["load"])
    ref_worth = {}
    for tag, rec in frozen.items():
        if rec.get("ok"):
            ref_worth[tag] = abs(rec["load"] - both["load"]) / abs(both["load"])
    print()
    print("    what each seam is WORTH, at the reference and at the optimum:")
    for tag in frozen:
        a_, b_ = ref_worth.get(tag), worth.get(tag)
        if a_ is None or b_ is None:
            continue
        print(f"      {tag:38s} {a_ * 100:6.2f}%  ->  {b_ * 100:6.2f}%")
    out["ablation"]["at_optimum"] = dict(design=d_opt, live=live,
                                         frozen=frozen_opt, worth=worth,
                                         worth_at_reference=ref_worth)


def _freeze_each_seam(base: dict, steps: int, u, v, h_live=None) -> dict:
    """Freeze each seam in turn around `base` and march both.

    **Freezing the suspension has to pin the ride height at the height the LIVE
    design actually settles to**, or the frozen column is a different operating
    point and the difference is a transit rather than the seam's worth. Using
    the reference-load release height ``h0 - LOAD_REF/k`` is only the same thing
    when the design's own settled load is near ``LOAD_REF``: at the reference
    design the two agree to $4\\times10^{-4}$, and at the constrained optimum --
    which carries $19\\%$ more load on a softer spring -- they differ by $37\\%$,
    which would have been read as the seam's worth. So `h_live` is passed in
    when the caller knows it.
    """
    frozen = {}
    for tag, k in (("no suspension (CS-12 alone, k -> inf)", 1.0e9),
                   ("no structure (CS-10 alone, E* -> inf)", None)):
        d = dict(base)
        if k is not None:
            d["k"] = k
            d["h0"] = (base["h0"] - F.LOAD_REF / base["k"] + F.LOAD_REF / k
                       if h_live is None else float(h_live) + F.LOAD_REF / k)
        else:
            d["e_star"] = 1.0e9
        try:
            r = F.FrontWingRollout(tiling=F.SINGLE_TILING, coupling="tight",
                                   motion=True, design=d)
            res = r.run(steps=steps, u0=u, v0=v)
            s = res.settled()
            frozen[tag] = dict(load=s["load"], h=s["h"], tip=s["tip"],
                               vm_max=s["vm_max"], ok=True)
            print(f"    {tag:38s} load {s['load']:.8f}  h {s['h']:.7f}  "
                  f"tip {s['tip']:+.7f}")
        except RuntimeError as exc:
            frozen[tag] = dict(ok=False, why=str(exc)[:200])
            print(f"    {tag:38s} DECLINED: {str(exc)[:90]}")
    return frozen


# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_ROOT, "out", "w141"))
    ap.add_argument("--stages", nargs="*", default=list(STAGES), choices=STAGES)
    ap.add_argument("--gate-steps", type=int, default=N_GATE)
    ap.add_argument("--residual-steps", type=int, default=480)
    ap.add_argument("--obj-steps", type=int, default=N_OBJ)
    ap.add_argument("--grad-steps", type=int, default=30)
    ap.add_argument("--pop-evals", type=int, default=200)
    ap.add_argument("--penalty", type=float, default=PENALTY)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                    help="torch CPU threads. ONE by default and it is measured: this workload is 5.8x SLOWER on eight, and the two agree bitwise, so the thread count is a clock rather than a variable. See `stage_cost`.")
    a = ap.parse_args(argv)
    torch.set_num_threads(max(1, int(a.threads)))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "w141.json")
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
               stages=a.stages, case="front-wing (PoC 2)")
    t_all = time.perf_counter()
    for s in a.stages:
        print()
        t0 = time.perf_counter()
        try:
            if s == "gate":
                stage_gate(out, a.out, steps=a.gate_steps)
            elif s == "residual":
                stage_residual(out, a.out, steps=a.residual_steps)
            elif s == "design":
                stage_design(out, a.out, steps=a.obj_steps,
                             n_grad=a.grad_steps, n_pop=a.pop_evals,
                             weight=a.penalty, seed=a.seed)
            elif s == "design_tight":
                stage_design_tight(out, a.out, steps=a.obj_steps,
                                   n_grad=a.grad_steps, n_pop=a.pop_evals,
                                   weight=a.penalty, seed=a.seed)
            elif s == "ablation":
                stage_ablation(out, a.out, steps=a.obj_steps)
            elif s == "fdsweep":
                stage_fdsweep(out, a.out, steps=a.obj_steps)
            elif s == "penalty":
                stage_penalty(out, a.out, steps=a.obj_steps)
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
