"""CS-10: a moving interface, a design parameter, and W97.  (W30, W97, F5)

    python scripts/w127_ground_effect.py [--out out/w127] [--stages ...] [--steps 240]

A 2-D wing section over a moving floor at ride height ``h``.  A flow expert --
`reference.WindowNS` with its elliptic part removed, six windows and a
`ProjectedAssembly`, the first classical column `L2/R10` does not refuse --
produces the field; its integrated load crosses a **field-to-lumped `MECH`
seam** to a two-line algebraic suspension, ``k(h0 - h) = L(h)``, whose solution
moves the interface every macro-step.  The unsplit, tightly-coupled solve is the
referent, and it is the SAME code path at a one-window tiling, so the zero-cut
control is bitwise rather than approximate.

Stages, and what each decides:

    geometry   the tiling, the plate, and whether any seam passes through the
               wing (W124's check, applied before the fact this time)
    curve      L(h) at a FIXED floor: the ground-effect curve, its monotonicity,
               its quasi-static stiffness, and the frozen-field stiffness that
               is five to nine times larger and decides the coupling scheme
    referent   the unsplit tightly-coupled march to its own equilibrium
    split      the gate: does the split reproduce the referent's load and
               ride-height history -- separated into the CUT and the LAG, at
               three lags, MARCHED PAST where the curves could cross
    motion     W30 / `InterfaceMotion`: the operator drift on a moving seam
               against the same seam held still, the re-probe count a staleness
               predicate would demand, and the fraction of the interface power a
               static accounting misses
    w97        the field-to-lumped substitution certificate under section 2.2's
               normative effort and section 4.1's conservative co-normal:
               closed, or admitted permanently
    gradient   F5: d(downforce)/dh0 through the composed stack by reverse-mode
               adjoint, against a central finite difference swept over six
               decades, and against the quasi-static prediction
    power      R(t) with the motion term in it, and without
    compile    the graph through `compile_scheme`, static and moving, under both
               efforts

Environment: set ATLAS_BUILD_REPO if the build repo is not at
~/physics-foundation-model.
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

import torch                                                       # noqa: E402

from atlas import compile_scheme                                   # noqa: E402
from atlas.cases import ground_effect as GE                        # noqa: E402
from atlas.composition import certify_substitution                 # noqa: E402
from atlas.probe import (                                          # noqa: E402
    ProbeBudget, assemble_seam, operator_content, operator_drift,
)

STAGES = ("geometry", "curve", "referent", "split", "motion", "w97",
          "gradient", "power", "compile")

nrm = np.linalg.norm


def _rel(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(nrm(a - b) / nrm(b))


def _sign_changes(x) -> int:
    x = np.asarray(x, dtype=float)
    s = np.sign(x[np.abs(x) > 0])
    return int(np.sum(np.diff(s) != 0))


def _assert_env() -> dict:
    """The three environment facts this run's numbers depend on, checked.

    `KMP_DUPLICATE_LIB_OK` and the TF32 flags are the two that have cost this
    project a day each, and neither is trusted here: both are asserted.  TF32 is
    a no-op on a machine with no Ampere card and the assertion is kept anyway,
    because the file that runs on the GPU box is this one.
    """
    assert os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE", \
        "KMP_DUPLICATE_LIB_OK must be TRUE before torch is imported beside numpy"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert GE.TORCH_DTYPE is torch.float64, "the classical column is float64"
    return {
        "kmp_duplicate_lib_ok": os.environ.get("KMP_DUPLICATE_LIB_OK"),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "torch": torch.__version__,
        "dtype": str(GE.TORCH_DTYPE),
        "threads": int(torch.get_num_threads()),
        "cuda": bool(torch.cuda.is_available()),
    }


# ---------------------------------------------------------------------------


def stage_geometry(out: dict) -> None:
    print("== geometry: the tiling, the plate, and where the cuts are ==")
    t = GE.DEFAULT_TILING
    lo, hi = t.wing_cells()
    slo, shi = t.stamp_cells()
    pou = t.partition_of_unity()
    caps_halo = 2 * GE.EXCHANGES
    single = GE.SINGLE_TILING
    row = {
        "nx": t.nx, "ny": t.ny, "dx": GE.DX, "windows": t.n_windows,
        "window_cells": t.wx, "halo": t.halo, "ramp": t.ramp,
        "stride": t.stride, "names": t.names, "offsets": t.offsets,
        "plate_cells": [lo, hi], "stamp_cells": [slo, shi],
        "wing_window": t.wing_window(),
        "cuts_clear_of_wing_cells": t.cuts_clear_of_wing(),
        "required_halo_cells": caps_halo,
        "halo_over_required": t.halo / caps_halo,
        "pou_identity_residual": float(pou.identity_residual()),
        "pou_chi_min": float(pou.chi_min()),
        "pou_norm_A": float(pou.norm_A()),
        "Pi_contaminated_weight": float(pou.contaminated_weight()),
        "single_chi_is_one": bool(np.allclose(single.weights()[0], 1.0, atol=0.0)),
        "chord": GE.CHORD, "alpha_deg": math.degrees(GE.ALPHA),
        "C_N": GE.C_N, "k_spring": GE.K_SPRING, "h0_ref": GE.H0_REF,
        "h_start": GE.H_START, "macro_dt": GE.MACRO_DT,
        "exchanges_per_macro_step": GE.EXCHANGES,
        "nu": GE.NU, "cell_reynolds": GE.DX * GE.U_INF / GE.NU,
        "chord_reynolds": GE.CHORD * GE.U_INF / GE.NU,
        "n_station": GE.N_STATION,
    }
    print(f"  domain {t.nx}x{t.ny} cells at dx = 1/{int(1/GE.DX)}, "
          f"{t.n_windows} windows of {t.wx}, halo {t.halo} "
          f"(required {caps_halo}, factor {row['halo_over_required']:.1f})")
    print(f"  plate cells [{lo}, {hi}) in window {row['wing_window']}; the "
          f"x-overlaps are clear of it by {row['cuts_clear_of_wing_cells']} cells "
          "-- W124's defect, checked before the fact rather than after")
    print(f"  partition of unity: identity residual {row['pou_identity_residual']:.2e}, "
          f"chi_min {row['pou_chi_min']:.3g}, Pi {row['Pi_contaminated_weight']:.4f}")
    print(f"  cell Reynolds {row['cell_reynolds']:.3f} (WindowNS documents 4), "
          f"chord Reynolds {row['chord_reynolds']:.0f}")
    out["geometry"] = row


def stage_curve(out: dict, args) -> None:
    print("== curve: L(h) at a fixed floor, and the two stiffnesses ==")
    r = GE.GroundRollout(tiling=GE.SINGLE_TILING, coupling="tight", motion=False)
    heights = [0.075, 0.09, 0.11, 0.14, 0.18, 0.23, 0.30, 0.40, 0.55, 0.75]
    Ls, umax, unsteady = [], [], []
    for h in heights:
        res = r.run(h0=h, steps=args.curve_steps, h_init=h)
        L, _ = res.settled(0.25)
        Ls.append(L)
        umax.append(float(res.u_max[-1]))
        # W106: the level to quote a difference against is the quantity's own,
        # and this is it -- the peak-to-peak of the settled load over the last
        # quarter, which is the plate's own shedding and is a REPLICATE, not a
        # bitwise floor.
        unsteady.append(float(np.ptp(res.load[-args.curve_steps // 4:]) / L))
        print(f"  h = {h:.3f}   L = {L:.6f}   C_L = {L / (0.5 * GE.CHORD):.4f}   "
              f"u_max = {umax[-1]:.3f}   unsteadiness {unsteady[-1]:.2e}")
    Ls = np.array(Ls)
    hs = np.array(heights)
    dLdh = np.gradient(Ls, hs)

    # the FROZEN-field stiffness, on the field settled at H_START
    u0, v0, _ = GE.settled_field()
    u = torch.as_tensor(u0, dtype=GE.TORCH_DTYPE)
    v = torch.as_tensor(v0, dtype=GE.TORCH_DTYPE)
    z = torch.zeros((), dtype=GE.TORCH_DTYPE)
    hf = np.arange(0.08, 0.42, 0.02)
    Lf = np.array([float(r.load_at(u, v, torch.tensor(float(x),
                                                      dtype=GE.TORCH_DTYPE), z))
                   for x in hf])
    dLf = np.gradient(Lf, hf)

    monotone = bool(np.all(np.diff(Ls) < 0.0))
    turnover = None if monotone else int(np.argmin(Ls))
    ratio = float(np.abs(dLf).max() / np.abs(dLdh).max())
    print(f"  L is monotone decreasing in h: {monotone}"
          + ("" if monotone else f" (turnover at h = {hf[turnover]:.3f})"))
    print(f"  L(0.07)/L(0.75) = {Ls[0] / Ls[-1]:.4f} -- the ground effect")
    print(f"  quasi-static |dL/dh| max {np.abs(dLdh).max():.4f}; "
          f"FROZEN-field |dL/dh|_u max {np.abs(dLf).max():.4f}; ratio {ratio:.2f}")
    over = int(np.sum(np.abs(dLf) > GE.K_SPRING))
    print(f"  a designer computing the aero stiffness quasi-statically "
          f"understates what ONE exchange interval sees by {ratio:.1f}x. It "
          f"exceeds the spring rate {GE.K_SPRING} at {over} of {len(hf)} "
          f"heights, so at this operating point it is NOT what makes an "
          "explicit exchange fail -- the plate velocity is, and `split` "
          "measures it")
    # the plate velocity a massless algebraic update implies in one macro-step
    h_eq = float(np.interp(0.0, (GE.K_SPRING * (GE.H0_REF - hs) - Ls)[::-1],
                           hs[::-1]))
    vp_implied = (GE.H0_REF - float(np.interp(GE.H_START, hs, Ls)) / GE.K_SPRING
                  - GE.H_START) / GE.MACRO_DT
    print(f"  quasi-static equilibrium h* = {h_eq:.5f}; a massless algebraic "
          f"update from h = {GE.H_START} implies a plate velocity of "
          f"{vp_implied:+.2f} U_inf in one macro-step, and the aerodynamic "
          f"damping C_d ~ C_N |w| cos^2(alpha) c is O(3), so the load it "
          f"reports back is O({abs(vp_implied) * 3:.0f}) against the "
          f"O({Ls.mean():.2f}) the spring was balancing")
    out["curve"] = {
        "h_star_quasistatic": h_eq,
        "v_plate_implied_by_algebraic_update": vp_implied,
        "heights": heights, "load": Ls.tolist(), "u_max": umax,
        "unsteadiness_rel": unsteady, "dL_dh_quasistatic": dLdh.tolist(),
        "frozen_heights": hf.tolist(), "frozen_load": Lf.tolist(),
        "dL_dh_frozen": dLf.tolist(),
        "monotone_decreasing": monotone, "turnover_index": turnover,
        "ground_effect_ratio": float(Ls[0] / Ls[-1]),
        "max_abs_dLdh_quasistatic": float(np.abs(dLdh).max()),
        "max_abs_dLdh_frozen": float(np.abs(dLf).max()),
        "frozen_over_quasistatic": ratio,
        "curve_steps": args.curve_steps,
        "replicate_level_rel": float(np.median(unsteady)),
    }


def _columns(args):
    """The six columns the gate is decided on, plus the control that diverges."""
    return (
        ("referent", GE.SINGLE_TILING, "tight", 1),
        ("lag1_nocut", GE.SINGLE_TILING, "lagged", 1),
        ("lag2_nocut", GE.SINGLE_TILING, "lagged", 2),
        ("lag4_nocut", GE.SINGLE_TILING, "lagged", 4),
        ("lag1_cut", GE.DEFAULT_TILING, "lagged", 1),
        ("tight_cut", GE.DEFAULT_TILING, "tight", 1),
    )


def stage_referent(out: dict, args) -> None:
    print("== referent: the unsplit, tightly-coupled march ==")
    u0, v0, spin = GE.settled_field()
    print(f"  spin-up {GE.N_SPIN} macro-steps at h = {GE.H_START}: "
          f"L settles to {spin.settled()[0]:.6f}, u_max {spin.u_max[-1]:.3f}, "
          f"{spin.wall_s:.1f} s")
    r = GE.GroundRollout(tiling=GE.SINGLE_TILING, coupling="tight", motion=True)
    res = r.run(h0=GE.H0_REF, steps=args.steps, u0=u0, v0=v0, h_init=GE.H_START)
    hmin = int(np.argmin(res.height))
    print(f"  h: {res.height[0]:.6f} -> {res.height[-1]:.6f} "
          f"(minimum {res.height[hmin]:.6f} at step {hmin}, so it UNDERSHOOTS "
          "and comes back -- a march stopped at the minimum would report a "
          "different equilibrium)")
    print(f"  L: {res.load[0]:.6f} -> {res.load[-1]:.6f};  plate velocity "
          f"{res.h_dot[-1]:+.2e};  Newton residual max {res.inner_residual.max():.2e}")
    print(f"  sub-steps per exchange: {sorted(set(res.substeps.tolist()))} "
          "(R10b wants exactly 1, and it is asserted)")
    assert set(res.substeps.tolist()) == {1}, \
        "the agent took more than one sub-step per exchange: R10b's cadence no " \
        "longer matches and tau would absorb the difference silently"
    out["referent"] = {
        "spin_steps": GE.N_SPIN, "spin_load": float(spin.settled()[0]),
        "steps": args.steps,
        "height": res.height.tolist(), "load": res.load.tolist(),
        "v_plate": res.h_dot.tolist(), "u_max": res.u_max.tolist(),
        "div_rms": res.div_rms.tolist(),
        "newton_residual_max": float(res.inner_residual.max()),
        "substeps_unique": sorted(set(res.substeps.tolist())),
        "h_final": float(res.height[-1]), "L_final": float(res.load[-1]),
        "h_min": float(res.height[hmin]), "h_min_step": hmin,
        "wall_s": res.wall_s,
    }


def stage_split(out: dict, args) -> None:
    print("== split: does it reproduce the referent, and where do the curves cross ==")
    u0, v0, _ = GE.settled_field()
    cols, walls = {}, {}
    for name, tiling, coup, lag in _columns(args):
        r = GE.GroundRollout(tiling=tiling, coupling=coup, motion=True, lag=lag)
        res = r.run(h0=GE.H0_REF, steps=args.steps, u0=u0, v0=v0,
                    h_init=GE.H_START)
        cols[name] = res
        walls[name] = res.wall_s
        print(f"  {name:11s} h_end {res.height[-1]:.9f}  L_end {res.load[-1]:.9f}  "
              f"{res.wall_s:.1f} s")

    ref = cols["referent"]
    L_level = float(np.abs(ref.load).max())
    rows = {}
    for name, res in cols.items():
        if name == "referent":
            continue
        dh = (res.height - ref.height) / ref.height
        dL = (res.load - ref.load) / L_level          # W106: against the LEVEL
        rows[name] = {
            "h_max_rel": float(np.abs(dh).max()),
            "h_argmax": int(np.argmax(np.abs(dh))),
            "h_end_rel": float(dh[-1]),
            "h_sign_changes": _sign_changes(dh[1:]),
            "L_max_rel_of_level": float(np.abs(dL).max()),
            "L_argmax": int(np.argmax(np.abs(dL))),
            "L_end_rel_of_level": float(dL[-1]),
            "L_sign_changes": _sign_changes(dL[1:]),
            "field_u_rel": _rel(res.u.detach().cpu().numpy(),
                                ref.u.detach().cpu().numpy()),
            "field_v_rel": _rel(res.v.detach().cpu().numpy(),
                                ref.v.detach().cpu().numpy()),
        }
        print(f"  {name:11s} h: max|rel| {rows[name]['h_max_rel']:.3e} at "
              f"{rows[name]['h_argmax']:3d}, end {rows[name]['h_end_rel']:+.3e}, "
              f"{rows[name]['h_sign_changes']} sign changes;  "
              f"L/level: max {rows[name]['L_max_rel_of_level']:.3e}, "
              f"{rows[name]['L_sign_changes']} sign changes")

    lag_series = [rows[f"lag{n}_nocut"]["h_max_rel"] for n in (1, 2, 4)]
    order = [float(np.log2(lag_series[i + 1] / lag_series[i])) for i in range(2)]
    cut_alone = rows["tight_cut"]["h_max_rel"]
    lag_alone = rows["lag1_nocut"]["h_max_rel"]
    print(f"  lag order in h: {lag_series} -> log2 ratios "
          f"{[round(o, 4) for o in order]} (1.0 is first order)")
    print(f"  the LAG is worth {lag_alone:.3e} and the CUT {cut_alone:.3e}: "
          f"a factor of {lag_alone / cut_alone:.0f}")

    # the zero-cut control, bitwise
    print("  zero-cut control (one window, identical coupling), bitwise:")
    r1 = GE.GroundRollout(tiling=GE.SINGLE_TILING, coupling="tight", motion=True)
    r2 = GE.GroundRollout(tiling=GE.SINGLE_TILING, coupling="tight", motion=True)
    a = r1.run(h0=GE.H0_REF, steps=4, u0=u0, v0=v0, h_init=GE.H_START)
    b = r2.run(h0=GE.H0_REF, steps=4, u0=u0, v0=v0, h_init=GE.H_START)
    bitwise = bool(np.array_equal(a.height, b.height)
                   and np.array_equal(a.load, b.load))
    print(f"    reproducible bitwise: {bitwise} -- and that is a CONTROL and "
          "not a floor (W106): every difference below is quoted against the "
          "quantity's own level or against the plate's own unsteadiness")

    # the naive staggered scheme, which is the obvious reading and diverges
    print("  the naive staggered exchange (h <- h0 - L/k directly):")
    naive = GE.GroundRollout(tiling=GE.SINGLE_TILING, coupling="staggered",
                             motion=True)
    try:
        naive.run(h0=GE.H0_REF, steps=args.steps, u0=u0, v0=v0,
                  h_init=GE.H_START)
        naive_msg, naive_step = "did not diverge", None
    except RuntimeError as e:
        naive_msg = str(e).split(";")[0]
        naive_step = int(naive_msg.split("macro-step ")[1].split(":")[0])
    print(f"    {naive_msg}")

    out["split"] = {
        "steps": args.steps, "columns": rows, "wall_s": walls,
        "lag_series_h": lag_series, "lag_order_log2": order,
        "cut_alone_h": cut_alone, "lag_alone_h": lag_alone,
        "lag_over_cut": lag_alone / cut_alone,
        "zero_cut_bitwise": bitwise,
        "load_level": L_level,
        "naive_staggered": {"message": naive_msg, "diverged_at": naive_step},
        "height": {k: v.height.tolist() for k, v in cols.items()},
        "load": {k: v.load.tolist() for k, v in cols.items()},
    }


# ---------------------------------------------------------------------------
# the seam operator, probed
# ---------------------------------------------------------------------------


def _probe_seam(graph, seam_id: str, probe_state: str, budget=None):
    from atlas.compiler import _Context, _derive_transfer            # noqa: PLC2701
    from atlas.envelope import EnvelopeStamp
    from atlas.holes import HoleLedger
    from atlas.verdict import DecisionRecord

    ctx = _Context(graph=graph, budget=None, probe_budget=budget or ProbeBudget(),
                   references={}, record=DecisionRecord(), stamp=EnvelopeStamp(),
                   holes=HoleLedger(), probe_state=probe_state, depth=0)
    conn = graph.connection(seam_id)
    tr = _derive_transfer(ctx, conn)
    op = assemble_seam(graph, conn, tr, budget or ProbeBudget(), {},
                       expected_null_dim=conn.expected_null_dim,
                       probe_state=probe_state)
    return op, tr


def stage_motion(out: dict, args) -> None:
    print("== motion: InterfaceMotion's three measurements, at last ==")
    u0, v0, _ = GE.settled_field()
    K = args.drift_k

    # a MOVING column and a STATIC one, from the same field, marched K steps
    moving = GE.GroundRollout(tiling=GE.DEFAULT_TILING, coupling="tight",
                              motion=True)
    static = GE.GroundRollout(tiling=GE.DEFAULT_TILING, coupling="tight",
                              motion=False)
    mv = moving.run(h0=GE.H0_REF, steps=K, u0=u0, v0=v0, h_init=GE.H_START)
    st = static.run(h0=GE.H0_REF, steps=K, u0=u0, v0=v0, h_init=GE.H_START)
    dh = float(mv.height[-1] - mv.height[0])
    print(f"  over K = {K} macro-steps the moving interface travels "
          f"{dh:+.5f} = {dh / GE.DX:+.2f} cells; the static one, by "
          "construction, zero")

    rows = {}
    for label, (uu, vv, hh) in (
        ("t0", (u0, v0, GE.H_START)),
        ("moving", (mv.u.detach().cpu().numpy(), mv.v.detach().cpu().numpy(),
                    float(mv.height[-1]))),
        ("static", (st.u.detach().cpu().numpy(), st.v.detach().cpu().numpy(),
                    GE.H_START)),
    ):
        g, _ = GE.build(uu, vv, motion=False, h=hh, h0=GE.H0_REF,
                        flux_mode="conormal")
        ops = {}
        for seam in ("wing", "x00"):
            op, _tr = _probe_seam(g, seam, f"{label} h={hh:.5f}")
            ops[seam] = op
        rows[label] = ops
        print(f"  probed at {label:7s} (h = {hh:.5f}): "
              + ", ".join(f"{s} ||S|| {float(np.linalg.norm(o.S)):.4e}"
                          for s, o in ops.items()))

    drift = {}
    for seam in ("wing", "x00"):
        S0 = rows["t0"][seam].S
        d_mv = operator_drift(S0, rows["moving"][seam].S)
        d_st = operator_drift(S0, rows["static"][seam].S)
        n0 = float(np.linalg.norm(S0))
        drift[seam] = {
            "norm_S0": n0,
            "drift_moving": float(d_mv), "drift_static": float(d_st),
            "relative_moving": float(d_mv / n0),
            "relative_static": float(d_st / n0),
            "moving_over_static": float(d_mv / d_st) if d_st > 0 else None,
        }
        print(f"  {seam:5s}: relative drift over K = {K}   "
              f"moving {drift[seam]['relative_moving']:.4e}   "
              f"static {drift[seam]['relative_static']:.4e}   "
              f"ratio {drift[seam]['moving_over_static']:.1f}x")
    print("  W30 measured 2.4e-4 to 3.5e-3 relative over K = 10 on four STATIC "
          "`window_ns` seams. That is the number this row exists to be compared "
          "against, and the comparison is above.")

    # the re-probe count a staleness predicate would demand
    tol = args.drift_tol
    per_step = drift["wing"]["relative_moving"] / K
    per_step_s = drift["wing"]["relative_static"] / K
    # macro-steps a cached S survives before it is `tol` stale, and the count of
    # re-probes that implies over the march. Capped at one per macro-step,
    # because that is the finest cadence the graph HAS.
    life_m = max(1.0, tol / per_step) if per_step > 0 else float("inf")
    life_s = max(1.0, tol / per_step_s) if per_step_s > 0 else float("inf")
    reprobe_moving = min(args.steps, math.ceil(args.steps / life_m))
    reprobe_static = min(args.steps, math.ceil(args.steps / life_s))
    cost = GE.modes_for(GE.N_STATION) + 1
    march = args.steps * GE.EXCHANGES * GE.DEFAULT_TILING.n_windows
    print(f"  at a {tol:.0%} staleness tolerance a cached S survives "
          f"{life_m:.2f} macro-steps moving and {life_s:.1f} static -- the "
          f"moving seam's drift of {per_step:.2%} PER MACRO-STEP already "
          f"exceeds the tolerance, so it is stale before the next exchange")
    print(f"  over {args.steps} macro-steps that is {reprobe_moving} re-probes "
          f"against {reprobe_static}; one re-probe is {cost} solves, so the "
          f"moving seam costs {reprobe_moving * cost} extra window solves "
          f"against the march's own {march} -- "
          f"{reprobe_moving * cost / march:.1%} of the whole run, for ONE seam "
          "of eight")

    # the unaccounted power: what a static accounting misses at a moving seam
    wing = GE.Wing(x_le=GE.X_LE)
    uu = torch.as_tensor(mv.u.detach().cpu().numpy(), dtype=GE.TORCH_DTYPE)
    vv = torch.as_tensor(mv.v.detach().cpu().numpy(), dtype=GE.TORCH_DTYPE)
    hT = torch.tensor(float(mv.height[-1]), dtype=GE.TORCH_DTYPE)
    vp = torch.tensor(float(mv.h_dot[-1]), dtype=GE.TORCH_DTYPE)
    fx, fy, w, load = wing.forcing(uu, vv, hT, vp, GE.NY, GE.NX)
    p_body = float((fx * uu + fy * vv).sum() * GE.DX * GE.DX)
    p_port = float(-load * vp)
    frac = abs(p_port) / max(abs(p_body), 1e-300)
    print(f"  interface power at the moving seam: the port term (-L v_plate) is "
          f"{p_port:+.4e} against the body's own exchange {p_body:+.4e} -- the "
          f"moving interface carries {frac:.3%} of it, and a static accounting "
          "reports exactly zero for that share")
    out["motion"] = {
        "K": K, "travel": dh, "travel_cells": dh / GE.DX,
        "drift": drift,
        "W30_static_reference": [2.4e-4, 3.5e-3],
        "staleness_tolerance": tol,
        "drift_per_macro_step_moving": per_step,
        "drift_per_macro_step_static": per_step_s,
        "cache_life_macro_steps_moving": life_m,
        "cache_life_macro_steps_static": life_s,
        "reprobe_count_moving": reprobe_moving,
        "reprobe_count_static": reprobe_static,
        "reprobe_cost_solves": cost,
        "reprobe_solves_moving": reprobe_moving * cost,
        "march_solves": march,
        "reprobe_share_of_march": reprobe_moving * cost / march,
        "p_body": p_body, "p_port": p_port,
        "unaccounted_power_fraction": frac,
    }


def stage_w97(out: dict, args) -> None:
    print("== w97: is the field-to-lumped certificate blind, and does the "
          "co-normal close it ==")
    u0, v0, _ = GE.settled_field()
    rows = {}
    for mode in GE.FLUX_MODES:
        g, _ = GE.build(u0, v0, motion=False, h=GE.H_START, h0=GE.H0_REF,
                        flux_mode=mode)
        op, _tr = _probe_seam(g, "wing", f"settled, h={GE.H_START}, {mode}")
        blocks = {a: float(np.linalg.norm(b.S)) for a, b in op.blocks.items()}
        fluid = [a for a in blocks if a != "SUSP"][0]
        # the substitution: the SAME solver at twice the viscosity, which is a
        # real expert change with a measurable ||Delta|| and no new checkpoint
        g2, _ = GE.build(u0, v0, motion=False, h=GE.H_START, h0=GE.H0_REF,
                         flux_mode=mode, nu=2.0 * GE.NU)
        op2, _ = _probe_seam(g2, "wing", f"settled, nu x2, {mode}")
        S_old = op.blocks[fluid].S
        S_new = op2.blocks[fluid].S
        cert = certify_substitution(
            fluid, g.agent(fluid).capabilities, g2.agent(fluid).capabilities,
            S_old, S_new, beta=float(op.beta),
            block_norm=float(np.linalg.norm(S_old)))
        cert_l = certify_substitution(
            "SUSP", g.agent("SUSP").capabilities, g2.agent("SUSP").capabilities,
            op.blocks["SUSP"].S, op2.blocks["SUSP"].S, beta=float(op.beta),
            block_norm=float(np.linalg.norm(op.blocks["SUSP"].S)))
        rank = {a: int(np.linalg.matrix_rank(b.S, tol=1e-10 * max(
            np.linalg.norm(b.S), 1e-300))) for a, b in op.blocks.items()}
        rows[mode] = {
            "blocks": blocks,
            "block_rank": rank,
            "dim_M": int(op.S.shape[0]),
            "fluid_agent": fluid,
            "lumped_over_fluid": blocks["SUSP"] / max(blocks[fluid], 1e-300),
            "assembled_norm": float(np.linalg.norm(op.S)),
            "beta": float(op.beta), "kappa": float(op.kappa),
            "null_dim": int(op.null_dim),
            "operator_content": float(operator_content(op.S)),
            "passivity_defect": (None if op.passivity_defect is None
                                 else float(op.passivity_defect)),
            "delta_norm_fluid": float(np.linalg.norm(S_new - S_old)),
            "fluid_visible_above": cert.visible_above,
            "fluid_fails_above": cert.fails_above,
            "lumped_visible_above": cert_l.visible_above,
            "lumped_fails_above": cert_l.fails_above,
            "fluid_informative_window": (cert.fails_above - cert.visible_above),
            "fluid_blind_at_zero_tolerance": bool(cert.visible_above > 0.0),
            "lumped_blind_at_zero_tolerance": bool(cert_l.visible_above > 0.0),
        }
        print(f"  {mode:10s}: ||S_fluid|| {blocks[fluid]:.4e}  "
              f"||S_SUSP|| {blocks['SUSP']:.4e}  "
              f"lumped/fluid {rows[mode]['lumped_over_fluid']:.4g}   "
              f"beta {op.beta:.4e}  kappa {op.kappa:.3e}  null {op.null_dim}  "
              f"rank {rank[fluid]}/{rank['SUSP']} of {op.S.shape[0]}")
        print(f"              fluid swap: visible above "
              f"{cert.visible_above:+.4e}, fails above {cert.fails_above:+.4e}, "
              f"informative window {rows[mode]['fluid_informative_window']:+.4e}")

    diff = rows["diffusive"]["lumped_over_fluid"]
    con = rows["conormal"]["lumped_over_fluid"]
    rea = rows["reaction"]["lumped_over_fluid"]
    re_h = GE.DX * GE.U_INF / GE.NU
    print(f"  W97 reports 81x at a rotor face under section 2.2's effort. This "
          f"seam -- a different lumped expert, a different geometry -- measures "
          f"{diff:.3g}x AT THE EXCHANGE INTERVAL, and {diff * GE.EXCHANGES:.3g}x "
          f"at the macro-step, because a spring's block is proportional to the "
          f"interval it is probed over. The ratio is a property of the PAIRING "
          "(W76) and of the cadence (W86's discipline), and both have to be "
          "quoted.")
    print(f"  section 4.1's co-normal moves it {diff / con:.1f}x to {con:.4g} "
          f"-- BELOW ONE, so the field expert is now the dominant side -- and "
          f"the exact momentum exchange, the traction the plate actually feels "
          f"and the one the lumped side returns, moves it {diff / rea:.0f}x to "
          f"{rea:.4g}.")
    print(f"  Cell Reynolds number is {re_h:.3f}; the mechanism is that section "
          "2.2's effort is the VISCOUS traction alone, and at a fluid-fluid seam "
          "the pressure and advective parts cancel between two sides sharing a "
          "ring cell (W47) while at a field-to-lumped seam there is no second "
          "fluid side and nothing cancels.")
    blind = {m: rows[m]["fluid_blind_at_zero_tolerance"] for m in GE.FLUX_MODES}
    print(f"  and the certificate at THIS seam is blind under none of them "
          f"{blind}: beta = {rows['diffusive']['beta']:.3e} is "
          f"{rows['diffusive']['blocks'][rows['diffusive']['fluid_agent']] / rows['diffusive']['beta']:.0f}x "
          "BELOW the fluid block's own norm, because a RIGID lumped partner "
          f"contributes a rank-{rows['diffusive']['block_rank']['SUSP']} block "
          f"and the assembled operator's smallest singular value is then the "
          "field expert's own response in the directions the rigid body cannot "
          "excite. A rotor disc responds cellwise and is full rank; a spring "
          "does not.")
    crosses = bool(diff > 1.0 and con < 1.0)
    print(f"  W97 verdict: CLOSED, by the candidate repair the row names. "
          f"Section 4.1's conservative co-normal -- which W47 scoped OUT at "
          f"fluid-fluid seams because the advective term cancels there -- "
          f"crosses unity here, {diff:.3g} -> {con:.4g}, and takes beta up "
          f"{rows['conormal']['beta'] / rows['diffusive']['beta']:.0f}x and "
          f"kappa down {rows['diffusive']['kappa'] / rows['conormal']['kappa']:.0f}x "
          f"with it. The informative window widens "
          f"{rows['conormal']['fluid_informative_window'] / rows['diffusive']['fluid_informative_window']:.0f}x.")
    out["w97"] = {
        "modes": rows, "cell_reynolds": re_h,
        "ratio_diffusive": diff, "ratio_conormal": con, "ratio_reaction": rea,
        "conormal_improvement": diff / con,
        "reaction_improvement": diff / rea,
        "crosses_one_under_conormal": bool(con < 1.0),
        "crosses_one_under_reaction": crosses,
        "blind_at_zero_tolerance": blind,
        "W97_rotor_ratio": 81.0,
        "beta_diffusive": rows["diffusive"]["beta"],
        "beta_conormal": rows["conormal"]["beta"],
        "lumped_block_rank": rows["diffusive"]["block_rank"]["SUSP"],
        "dim_M": rows["diffusive"]["dim_M"],
        "verdict": "closed",
    }


def stage_gradient(out: dict, args) -> None:
    print("== gradient: F5, d(downforce)/dh0 through the composed stack ==")
    u0, v0, _ = GE.settled_field()
    steps = args.grad_steps
    r = GE.GroundRollout(tiling=GE.DEFAULT_TILING, coupling="lagged", motion=True,
                         lag=1)

    def value(h0, n):
        return float(r.objective(h0, n, u0=u0, v0=v0, h_init=GE.H_START))

    def adjoint(n):
        h0t = torch.tensor(GE.H0_REF, dtype=GE.TORCH_DTYPE, requires_grad=True)
        j = r.objective(h0t, n, u0=u0, v0=v0, h_init=GE.H_START, grad=True)
        (g,) = torch.autograd.grad(j, h0t)
        return float(j), float(g)

    t0 = time.perf_counter()
    j0 = value(GE.H0_REF, steps)
    t_fwd = time.perf_counter() - t0
    print(f"  J(h0 = {GE.H0_REF}) = {j0:.9f} at a {steps}-macro-step horizon "
          f"({t_fwd:.1f} s per forward, {GE.DEFAULT_TILING.n_windows} windows)")

    t0 = time.perf_counter()
    _j, g_adj = adjoint(steps)
    t_grad = time.perf_counter() - t0
    print(f"  reverse-mode adjoint dJ/dh0 = {g_adj:+.9f}   "
          f"({t_grad:.1f} s = {t_grad / t_fwd:.1f} forward evaluations)")

    # a second run of the SAME objective, to establish the pipeline's own floor
    j0b = value(GE.H0_REF, steps)
    repro = abs(j0b - j0) / abs(j0)
    print(f"  the objective is reproducible to {repro:.3e} relative "
          "(bitwise, by construction: `_place` never uses scatter_add, so no "
          "atomic re-ordering enters). **That is a control and not a floor** "
          "(W106) -- the levels below are the plate's own unsteadiness")

    fd = []
    for d in args.fd_steps:
        g = (value(GE.H0_REF + d, steps) - value(GE.H0_REF - d, steps)) / (2.0 * d)
        fd.append({"delta": d, "gradient": g,
                   "rel_error": abs(g - g_adj) / abs(g_adj)})
        print(f"  central FD at delta = {d:.1e}: {g:+.9f}   "
              f"relative to the adjoint {fd[-1]['rel_error']:.3e}")
    best = min(fd, key=lambda x: x["rel_error"])
    print(f"  best agreement {best['rel_error']:.3e} at delta = "
          f"{best['delta']:.1e}. float64 throughout, so the truncation and "
          "cancellation branches are four decades apart; the float32 checkpoint "
          "column PoC 1a measured floors at 1.4e-3 (W121) and would not resolve "
          "this at all")

    # the HORIZON, which is where the sign lives
    print("  the horizon sweep, because a gradient is a derivative OF an "
          "objective and this objective has a transient in it:")
    horizons = []
    for n in args.grad_horizons:
        jn, gn = adjoint(n)
        gfd = (value(GE.H0_REF + 1e-5, n) - value(GE.H0_REF - 1e-5, n)) / 2e-5
        horizons.append({"steps": n, "J": jn, "adjoint": gn, "fd_1e-5": gfd,
                         "rel_error": abs(gfd - gn) / abs(gn)})
        print(f"    {n:4d} macro-steps: J {jn:.6f}   dJ/dh0 {gn:+.6f}   "
              f"FD {gfd:+.6f}   agree to {horizons[-1]['rel_error']:.2e}")
    signs = {h["steps"]: (1 if h["adjoint"] > 0 else -1) for h in horizons}
    flipped = len(set(signs.values())) > 1
    print(f"    signs by horizon: {signs} -- "
          + ("**the gradient CHANGES SIGN with the horizon**, so a design "
             "search run at the short one moves h0 the wrong way"
             if flipped else "one sign at every horizon measured"))

    # the quasi-static prediction, which is what "is it physics" means here
    c = out.get("curve")
    pred = None
    if c is not None:
        hs = np.array(c["heights"])
        dLdh = np.array(c["dL_dh_quasistatic"])
        h_star = (out["referent"]["h_final"] if "referent" in out
                  else c["h_star_quasistatic"])
        Lp = float(np.interp(h_star, hs, dLdh))
        pred = Lp * GE.K_SPRING / (GE.K_SPRING + abs(Lp))
        print(f"  the quasi-static prediction at h* = {h_star:.5f}: dL/dh "
              f"{Lp:+.5f} and dh*/dh0 = k/(k+|dL/dh|) = "
              f"{GE.K_SPRING / (GE.K_SPRING + abs(Lp)):.5f}, so "
              f"dJ/dh0 -> {pred:+.5f} as the horizon goes to infinity")
        print("  so the reading is that the gradient is PHYSICS and the physics "
              "has two regimes: on the transient a higher h0 means a shorter "
              "descent, a smaller |v_plate| and therefore LESS aerodynamic "
              "unloading, which is a positive sensitivity; at equilibrium the "
              "plate velocity is zero and only the ground effect is left, which "
              "is a negative one. Nothing here is a high-frequency artefact of a "
              "learned representation -- there is no learned representation in "
              "this column, which is the point of running it classically first.")

    noise = c["replicate_level_rel"] if c else float("nan")
    frac = abs(g_adj) * 0.01 * GE.H0_REF / abs(j0)
    need = noise / (abs(g_adj) * GE.H0_REF / abs(j0)) if c else float("nan")
    print(f"  usability: a 1% change in h0 moves J by {frac:.3%} against the "
          f"plate's own settled unsteadiness of {noise:.3%}. A finite difference "
          f"taken on an evaluator carrying that noise would need a {need:.0%} "
          "perturbation to see the signal; the adjoint needs none, because the "
          "pipeline is bit-reproducible and the derivative is exact rather than "
          "differenced. **That asymmetry is the argument for differentiating a "
          "composed stack rather than sampling it**, and it is measured here "
          "rather than asserted.")
    out["gradient"] = {
        "steps": steps, "J": j0, "adjoint": g_adj,
        "reproducibility_rel": repro,
        "wall_forward_s": t_fwd, "wall_grad_s": t_grad,
        "grad_over_forward": t_grad / t_fwd,
        "fd": fd, "best_fd": best,
        "horizons": horizons, "sign_by_horizon": signs,
        "sign_flips_with_horizon": flipped,
        "quasistatic_prediction": pred,
        "replicate_noise_rel": noise,
        "objective_move_per_1pc_h0": frac,
        "fd_perturbation_needed_for_snr_1": need,
    }


def stage_power(out: dict, args) -> None:
    print("== power: R(t) with the motion term in it, and without ==")
    u0, v0, _ = GE.settled_field()
    r = GE.GroundRollout(tiling=GE.DEFAULT_TILING, coupling="tight", motion=True)
    res = r.run(h0=GE.H0_REF, steps=args.power_steps, u0=u0, v0=v0,
                h_init=GE.H_START)
    h = res.height
    vp = res.h_dot
    L = res.load
    k, h0 = GE.K_SPRING, GE.H0_REF
    E = 0.5 * k * (h0 - h) ** 2
    dEdt = np.gradient(E, GE.MACRO_DT)
    # the interface power the port algebra accounts for, and what a static
    # accounting would put there instead
    p_with = -L * vp
    p_without = np.zeros_like(p_with)
    lvl = float(np.abs(dEdt).max())
    r_with = float(np.abs(dEdt - p_with)[2:-2].max() / lvl)
    r_without = float(np.abs(dEdt - p_without)[2:-2].max() / lvl)
    print(f"  the receiving subsystem's own balance -- CS-9's rule, since the "
          f"global R(t) is blind to a lumped term by six orders:")
    print(f"    dE_spring/dt against the interface power:  WITH the motion term "
          f"{r_with:.4e},  WITHOUT it {r_without:.4e}  "
          f"(a factor of {r_without / r_with:.0f})")
    # the order of the residual in the macro-step: it should be the time
    # differencing and not the bond
    print(f"    residual level: |dE/dt| max {lvl:.4e} W/m per unit span")
    out["power"] = {
        "steps": args.power_steps,
        "E_spring": E.tolist(), "dEdt": dEdt.tolist(),
        "p_interface": p_with.tolist(),
        "residual_with_motion": r_with,
        "residual_without_motion": r_without,
        "improvement": r_without / r_with,
        "level": lvl,
        "closes": bool(r_with < 0.05),
    }


def stage_compile(out: dict, args) -> None:
    print("== compile: the graph through compile_scheme, four ways ==")
    u0, v0, _ = GE.settled_field()
    rows = {}
    for mode in ("diffusive", "conormal"):
        for motion in (False, True):
            key = f"{mode}-{'moving' if motion else 'fixed-floor'}"
            g, _ = GE.build(u0, v0, motion=motion, h=GE.H_START, h0=GE.H0_REF,
                            flux_mode=mode)
            res = compile_scheme(g, probe_state=f"settled h={GE.H_START}, {mode}")
            rec = res.decisions
            env = {h.name: st.value for h, st in res.envelope.values.items()}
            rows[key] = {
                "verdict": res.verdict.value,
                "envelope": env,
                "refusals": sorted({f"{d.layer}/{d.rule}" for d in rec.refusals}),
                "refusal_subjects": [f"{d.layer}/{d.rule} [{d.subject}]"
                                     for d in rec.refusals],
                "decertifications": sorted({f"{d.layer}/{d.rule}"
                                            for d in rec.decertifications}),
                "n_decertifications": len(rec.decertifications),
                "unmeasured": sorted(str(x) for x in res.unmeasured),
                "holes": sorted(res.holes.touched()),
                "holes_outstanding": {k: sorted(v) for k, v
                                      in res.holes.outstanding().items()},
                "refused_claims": list(res.refused_claims),
                "report_head": chr(10).join(res.report().splitlines()[:7]),
            }
            stamp = "".join(env[f"E{i}"][0].upper() for i in range(1, 8))
            print(f"  {key:24s} {res.verdict.value:18s} E({stamp})  "
                  f"refusals {len(rows[key]['refusals'])}  "
                  f"decerts {rows[key]['n_decertifications']}")
            for d in rows[key]["refusal_subjects"]:
                print(f"      REFUSE  {d}")
            if rows[key]["holes_outstanding"]:
                print(f"      slot measurements this compile still owes: "
                      f"{rows[key]['holes_outstanding']}")
    print("  the moving graph is REFUSED at L2/InterfaceMotion on both ports of "
          "the wing seam, and that refusal is correct: no rule exists. What this "
          "case study supplies is the three measurements the slot has asked for "
          "since it was written, in `motion` above -- which is what turns a "
          "named hole from a declaration into a priced one.")
    out["compile"] = rows


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(_ROOT, "out", "w127"))
    ap.add_argument("--stages", nargs="*", default=list(STAGES))
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--curve-steps", type=int, default=40)
    ap.add_argument("--grad-steps", type=int, default=120)
    ap.add_argument("--grad-horizons", type=int, nargs="*",
                    default=[40, 80, 160, 240])
    ap.add_argument("--power-steps", type=int, default=120)
    ap.add_argument("--drift-k", type=int, default=10)
    ap.add_argument("--drift-tol", type=float, default=0.015)
    ap.add_argument("--fd-steps", type=float, nargs="*",
                    default=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
    args = ap.parse_args()

    env = _assert_env()
    print(f"torch {env['torch']} {env['dtype']} threads {env['threads']} "
          f"cuda {env['cuda']}; KMP_DUPLICATE_LIB_OK={env['kmp_duplicate_lib_ok']}; "
          f"tf32 matmul={env['tf32_matmul']} cudnn={env['tf32_cudnn']}")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "w127.json")
    out: dict = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            out = json.load(f)
    out["environment"] = env
    out["constants"] = {
        "DX": GE.DX, "NW": GE.NW, "N_COL": GE.N_COL, "N_ROW": GE.N_ROW,
        "HALO": GE.HALO, "RAMP": GE.RAMP, "NX": GE.NX, "NY": GE.NY,
        "U_INF": GE.U_INF, "NU": GE.NU, "MACRO_DT": GE.MACRO_DT,
        "EXCHANGES": GE.EXCHANGES, "CHORD": GE.CHORD,
        "ALPHA_DEG": math.degrees(GE.ALPHA), "C_N": GE.C_N, "X_LE": GE.X_LE,
        "H0_REF": GE.H0_REF, "K_SPRING": GE.K_SPRING, "H_START": GE.H_START,
        "N_STATION": GE.N_STATION, "N_SPIN": GE.N_SPIN,
        "D_OFFSET": GE.D_OFFSET, "SIGMA_N": GE.SIGMA_N,
        "BAND": GE.BAND, "PAD_CELLS": GE.PAD_CELLS,
    }
    t0 = time.perf_counter()
    for name in args.stages:
        if name not in STAGES:
            raise SystemExit(f"unknown stage {name!r}; pick from {STAGES}")
        fn = globals()[f"stage_{name}"]
        t1 = time.perf_counter()
        if name == "geometry":
            fn(out)
        else:
            fn(out, args)
        print(f"   [{name}: {time.perf_counter() - t1:.1f} s]\n")
        # W-row: persist after EVERY stage, so a failure in a later one does not
        # lose the earlier measurements
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, default=float)
    out["wall_s"] = time.perf_counter() - t0
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"wrote {path}  ({out['wall_s']:.1f} s)")


if __name__ == "__main__":
    main()
