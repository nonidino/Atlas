"""Tier 19 -- how composition error grows with interface count.

Driver for `atlas/cases/scaling_ladder.py`.  Everything [[tier0-measurements]] 19
quotes comes from here; artifact `out/w100/w100.json`, cached states
`out/w100/state_<rung>_<expert>.npz`.

    python scripts/w100_scaling_ladder.py                 # everything, ~2 hours
    python scripts/w100_scaling_ladder.py --quick          # N <= 6, short march
    python scripts/w100_scaling_ladder.py --reuse-state    # skip the marches
    python scripts/w100_scaling_ladder.py --only r12,long_march --long-steps 120
                                                          # W100 alone, ~1 hour

The question, and why it is the one that decides
-------------------------------------------------

`f1-pathmap-and-end-goal` §5's **F1** fails if composition error grows
super-linearly in interface count.  §3.2 calls rung 9 -- fifteen to twenty
coupled agents -- *the rung that decides everything*, and schedules this sweep as
its early warning: *"a cheap look at the expensive question, and a bad result
there is a reason to stop and rethink, not to press on."*  It had never been run
at any N.  `case-study-ladder-to-f1` §4 then schedules nine further case studies
on the answer, and §4's last row says outright that CS-16 **is not built** if
this comes back super-linear.

So the deliverable is an exponent with an error bar, in two columns, over 24x.

What it measures, in order
---------------------------

  1. the ladder marches -- five rungs, two experts, four configurations each,
     with the turbine-free noise floor at every rung and wall time per macro-step
     per agent, which is Claim B's other half
  2. the composed defect at a COMMON state, per rung, in three forms: the
     reference-free blend spread (both columns), the chi-weighted L2/C2 bound and
     the composed defect against the monolith (classical column, because there is
     no monolith for a checkpoint at any resolution -- W95)
  3. **W58**: which of the two `cut_defect_bound` definitions each number is, and
     whether the disjointness condition that makes them equal holds, decided from
     the contaminated geometry alone
  4. Pi and the implied C_mu at every rung
  5. the accumulated defect -- composed rollout against monolith rollout, with
     and without the disks, which is the error a user of the composed system
     actually carries
  6. tau, sigma and gamma on one wake seam per rung, depth- AND harness-tagged
     (**W54**)
  7. **W81**: `certify_substitution` with no beta_min, reporting the thresholds;
     with beta_min derived from this rung's own eps_tol = min(tau, sigma); and
     swept, for continuity with Tier 18
  8. the compile at every rung: verdict, refusals, decertifications, and how the
     decertification count grows with N
  9. the fit: log(defect) against log(interface count), slope with a t-based
     confidence interval, plus the pairwise local slopes so curvature is visible
 10. **W100**: `R12` at both assemblies, at every rung -- the compile deciding
     L6/C2 on two graphs that differ in one field
 11. **W100**: the march continued to 120 macro-steps from the freestream, in
     four columns, which is where the instability the run was not looking for is
     reproduced and where the repair is shown to hold

The one thing here that IS a scheme
------------------------------------

Everything above item 10 is instrumentation: the march is a measurement
instrument that produces the state every probe is taken at, exactly as
`w93_wake_array.march` is.  **Item 11 is not.**  The projected assembly it runs
the classical column through is a declared composition-layer step --
`assembly.ProjectedAssembly`, checked by `R12`, carried on the emitted defect's
harness -- and the reason it had to become one is that as a line in a driver it
was a positive control that no case study shipped, beside a column whose
published trajectory does not continue past macro-step 82.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np                                                   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.assembly import C_MU_HALO                                 # noqa: E402
from atlas.capability import EllipticSubsolve                        # noqa: E402
from atlas.cases import scaling_ladder as sl                         # noqa: E402
from atlas.cases import wake_array as wa                             # noqa: E402
from atlas.composition import (                                      # noqa: E402
    beta_min_from_tolerance,
    certify_substitution,
)
from atlas.compiler import compile_scheme                            # noqa: E402
from atlas.graph import MeasuredConstants                            # noqa: E402
from atlas.emit import HarnessParameters                             # noqa: E402
from atlas.multiphysics import (                                     # noqa: E402
    lag_distance,
    seam_defect_split,
    tight_couple,
)
from atlas.ports import ResponseHalf                                 # noqa: E402
from atlas.probe import (                                            # noqa: E402
    ProbeBudget,
    aggregated_neighbour_disagreement,
    assemble_seam,
    restriction_defect_bound,
)
from atlas.scheme import Budget                                      # noqa: E402
from atlas.transfer import InterfaceSpace, SeamTransfer              # noqa: E402
from atlas.verdict import ADMIT                                   # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "out", "w100")

#: Freestream band, in cells -- the domain's own outer boundary.  `w93`'s number
#: and its reason: without holding the inlet and both laterals at ``(U_INF, 0)``
#: the disks drain the box, every turbine's inflow falling monotonically to 0.06
#: by step 45.  At larger N there are more disks and it gets worse, not better,
#: which is why this is held fixed rather than scaled.
BAND = 8

#: Macro-steps per march.  60 is what Tier 18 ran, and the N=6 reproduction
#: control is only a control at the same number.
STEPS = 60


def _f(x):
    """JSON-safe."""
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return [_f(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): _f(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_f(v) for v in x]
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    return x


def banner(s):
    print()
    print("=" * 78)
    print(s)
    print("=" * 78, flush=True)


# ---------------------------------------------------------------------------
# 1. the march, at any rung
# ---------------------------------------------------------------------------


def disk_rows(r, y_centre):
    y = (np.arange(r.tiling.ny) + 0.5) * wa.DX
    return np.abs(y - y_centre) <= 0.5 * wa.ROTOR_D + 1e-9


def _forcing(r, u, active, thickness=None):
    """The disks' body force on the whole domain, and each disk's own inflow.

    Lifted from `w93_wake_array.march` unchanged, including the derived smearing
    thickness ``Delta_d = <U_d> dt``: a body-force impulse over one macro-step is
    ``T dt / (A Delta_d)`` and momentum theory allows a crossing parcel to lose
    ``T / (A <U_d>)``, so the two agree only there.  At `disk.py`'s default
    ``0.1 D`` the impulse is roughly 2x what momentum theory allows.
    """
    import importlib
    dk = importlib.import_module("atlas_windfarm_reference.disk")
    x_c = (np.arange(r.tiling.nx) + 0.5) * wa.DX
    y_c = (np.arange(r.tiling.ny) + 0.5) * wa.DX
    fx = np.zeros(r.shape)
    rec = {}
    for rot in r.tiling.rotors:
        rows = disk_rows(r, rot.y_centre)
        i_up = int(np.argmin(np.abs(x_c - (rot.x_plane - 0.25))))
        ud = float(np.mean(u[rows, i_up]))
        rec[rot.rotor_id] = ud
        if rot.rotor_id in active:
            thick = (max(ud, 0.05) * wa.MACRO_DT if thickness is None
                     else float(thickness))
            d = dk.ActuatorDisk(thickness=thick)
            st = d(max(ud, 1e-3))
            fx = fx + d.body_force_field(x_c, y_c, wa.DX, wa.DX, st.thrust,
                                         x0=rot.x_plane - 0.5 * thick,
                                         y0=rot.y_centre - 0.5 * wa.ROTOR_D)
    return fx, rec


def _band(r, u, v):
    """Inlet and both laterals held at freestream; the outlet left free."""
    u[:, :BAND] = wa.U_INF
    v[:, :BAND] = 0.0
    u[:BAND, :] = wa.U_INF
    v[:BAND, :] = 0.0
    u[-BAND:, :] = wa.U_INF
    v[-BAND:, :] = 0.0
    return u, v


def composed_step(r, u, v, fx, kind, ex, assembly=None):
    """ONE composed macro-step: local solves, assembly, then the global operators.

    This is `w93_wake_array.march`'s body with the rung's tiling substituted, and
    it is deliberately a transcription rather than a re-derivation: the N=6 rung
    is a reproduction control, and a control against a re-implementation controls
    nothing.

    Returns ``(u1, v1, locals_)`` where ``locals_`` maps window name to that
    window's OWN one-macro-step solution before assembly -- which is the object
    every composition-defect measurement in stage 2 is taken on.

    **W98's class lives in the last two lines.**  Transport and pressure are
    GLOBAL operations belonging to the composition layer, and `step_many` will do
    both per window and periodically if asked.  At N=1 that is invisible; at
    N=24 the domain is larger and the periodic image is further away, so a
    per-window operator would make the defect look like it IMPROVES with N --
    a false confirmation of exactly the claim this run is testing.  So
    ``galilean=False, project=False`` and `wake_array.transport_and_project` on
    the assembled domain, once.

    **``assembly`` is W100's step and it is the classical column's repair.**
    Pass a `wake_array.projected_assembly` and the classical branch below runs
    the composition layer's declared operator -- blend, then one global Leray
    projection on the ASSEMBLED field -- rather than the blend alone.  Pass None
    and it runs the assembly the case study shipped before 2026-08-31, which is
    the blend alone, and whose rollout is not finite past macro-step 82 at six
    windows.  The Poseidon branch takes no ``assembly`` argument because it has
    had the step since W98: its ``transport_and_project`` call applies the same
    projection to the same assembled field on the same padded domain, FUSED with
    the transport in one spectral pass -- one ``_extend``, one forward FFT, the
    projection and the phase factor applied to the same coefficients, one inverse
    FFT.  **That is not the same thing as calling the API twice**, and the
    difference is measured rather than assumed: two calls re-pad in between, and
    `_extend`'s taper is read off the field's last column, which the first call
    changed.  Measured on a smooth test field the two orderings differ by 55% of
    the field's own scale, so the claim here is about the operator inside one
    pass and `tests/test_tier19_scaling_ladder.py` asserts the substantive half
    -- that this call removes the divergence the blend created -- rather than an
    equivalence that does not hold.
    """
    t = r.tiling
    if kind == "poseidon":
        ufs, vfs = t.cut(u - wa.U_INF), t.cut(v)
        u1, v1 = ex.step_many(ufs, vfs, wa.MACRO_DT, galilean=False, project=False)
        # the checkpoint's own mean drift removed, the INCOMING window mean kept:
        # a periodic box with no force conserves it, and the deficit a disk puts
        # there has to survive long enough to be carried out of the domain.
        u1 = u1 + (ufs.mean(axis=(1, 2)) - u1.mean(axis=(1, 2)))[:, None, None]
        v1 = v1 + (vfs.mean(axis=(1, 2)) - v1.mean(axis=(1, 2)))[:, None, None]
        locals_ = {name: (wa.U_INF + u1[k], v1[k]) for k, name in enumerate(t.names)}
        uf, vf = t.assemble(u1, v1)
        # `step_many` DROPS `force` when galilean is False, and silently (W99),
        # so the disks are applied here. The partition of unity sums to one, so
        # applying fx after assembly is applying it before.
        uf = uf + fx * wa.MACRO_DT
        uf, vf = wa.transport_and_project(uf, vf, wa.MACRO_DT, wa.U_INF)
        return wa.U_INF + uf, vf, locals_
    us, vs = t.cut(u), t.cut(v)
    fs = t.cut(fx)
    u1, v1 = ex.step_batch(us, vs, wa.MACRO_DT, bc0=None,
                           force=(fs, np.zeros_like(fs)))
    locals_ = {name: (u1[k], v1[k]) for k, name in enumerate(t.names)}
    if assembly is None:
        au, av = t.assemble(u1, v1)
    else:
        au, av = wa.assemble_conservative(t, u1, v1, assembly)
    return au, av, locals_


def march(r, active, n_steps, kind="poseidon", thickness=None, snapshots=None,
          snap_every=2, record_prev=False, state0=None, frozen_force=None):
    """Advance the rung; return the final state, each turbine's inflow history.

    ``state0`` starts from a given field instead of the freestream, and
    ``frozen_force`` holds the disks' body force fixed instead of letting each
    disk re-read its own inflow.  Both exist for `stage_accumulated`: a uniform
    field is a fixed point of the unforced composed step AND of the monolith, so
    an accumulated defect measured from the freestream with the disks off is
    exactly zero and measures nothing, while one measured with the disks live
    includes the composition error going round the closure loop.  Freezing the
    force on a developed field separates the two.
    """
    ex = (wa.scaled_expert() if kind == "poseidon"
          else wa.reference_solver(wa.NU_REF))
    u = np.ones(r.shape) if state0 is None else np.array(state0[0], dtype=float)
    v = np.zeros(r.shape) if state0 is None else np.array(state0[1], dtype=float)
    hist, prev = [], None
    for step in range(n_steps):
        if snapshots is not None and step % snap_every == 0:
            snapshots.append((u[::2, ::2].copy(), v[::2, ::2].copy()))
        if frozen_force is None:
            fx, rec = _forcing(r, u, active, thickness)
        else:
            fx, rec = frozen_force, {}
        hist.append(rec)
        if record_prev:
            prev = (u.copy(), v.copy())
        u, v, _ = composed_step(r, u, v, fx, kind, ex)
        u, v = _band(r, u, v)
        if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
            raise RuntimeError(
                f"{r.label} {kind} march diverged at macro-step {step}: the state "
                "is not finite. A NaN here becomes a NaN in every stage after it, "
                "so it is raised where it happened")
    return u, v, hist, prev


def march_monolith(r, active, n_steps, nu=wa.NU_REF, thickness=None,
                   state0=None, frozen_force=None):
    """The same march with NO cut: the referent E, undivided.

    Only the classical column has one.  A checkpoint fixed at 128x128 has no
    monolith at any resolution ever (**W95**), which is why the reference solver
    is load-bearing here rather than a courtesy -- it is the only thing in the
    experiment that can say what the right answer was.
    """
    mono = sl.reference_monolith(r.tiling.nx, r.tiling.ny, nu)
    u = np.ones(r.shape) if state0 is None else np.array(state0[0], dtype=float)
    v = np.zeros(r.shape) if state0 is None else np.array(state0[1], dtype=float)
    hist = []
    for _ in range(n_steps):
        if frozen_force is None:
            fx, rec = _forcing(r, u, active, thickness)
        else:
            fx, rec = frozen_force, {}
        hist.append(rec)
        uu, vv = mono.step_batch(u[None], v[None], wa.MACRO_DT, bc0=None,
                                 force=(fx[None], np.zeros_like(fx)[None]))
        u, v = uu[0], vv[0]
        u, v = _band(r, u, v)
        if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
            raise RuntimeError(
                f"{r.label} monolith march diverged: the state is not finite")
    return u, v, hist


#: ``||div u||_rms`` and the global Leray projection, **from the case module**
#: since 2026-08-31 (W100).  Both used to be defined here, which is how the
#: measurement and the operator that repairs what it measures could disagree
#: without anything noticing -- and it is why the projection was a control in a
#: driver rather than a step in the scheme.  `wake_array.leray_projection` is
#: now a declared `assembly.ConstraintProjection`, these two ARE its callables,
#: and `L6/C2` is what the compile checks.
divergence_rms = wa.divergence_rms
project_global = wa.project_assembled


def assembly_for(r):
    """The rung's declared assembly: the blend AND the projection (L6/C2)."""
    return wa.projected_assembly(r.tiling)


def power_of(ud):
    import importlib
    dk = importlib.import_module("atlas_windfarm_reference.disk")
    return float(dk.ActuatorDisk()(max(float(ud), 1e-6)).power)


def settled(hist, key, n=8):
    tail = [h[key] for h in hist[-n:]]
    return float(np.mean(tail)), float(np.max(tail) - np.min(tail))


def stage_march(r, kind, steps, states):
    """Four configurations at one rung, and the wall time Claim B's other half is."""
    out = {"rung": r.label, "expert": kind, "n_steps": steps,
           "n_windows": r.n_windows, "n_cells": r.n_cells}
    rotors = [x.rotor_id for x in r.tiling.rotors]
    first = rotors[0] if rotors else None
    # One untimed step first. The checkpoint's loader, the FFT plans and the
    # DCT tables are all first-call costs, and a wall time that includes them is
    # a measurement of the import rather than of the composition -- which at the
    # smallest rung is most of it.
    march(r, set(), 1, kind)
    configs = [("free", set())]
    if first:
        configs += [(f"solo_{first}", {first}),
                    ("first_off", set(rotors[1:])),
                    ("all", set(rotors))]
    runs = {}
    for label, active in configs:
        t0 = time.time()
        u, v, hist, prev = march(r, active, steps, kind,
                                 record_prev=(label == "all"))
        wall = time.time() - t0
        runs[label] = {"state": (u, v), "hist": hist, "prev": prev}
        out.setdefault("wall", {})[label] = {
            "seconds": wall,
            "per_macro_step": wall / steps,
            "per_macro_step_per_agent": wall / steps / r.n_windows,
        }
        line = "  ".join(f"{k} Ud={settled(hist, k)[0]:.4f}" for k in rotors[:4])
        print(f"    {label:10s} {line}   umin={u.min():+.4f} umax={u.max():+.4f}"
              f"  [{wall:.1f}s, {wall / steps / r.n_windows * 1e3:.1f} ms/step/agent]",
              flush=True)

    # the turbine-free control: the rung's own numerical noise floor
    if rotors:
        free = {k: settled(runs["free"]["hist"], k) for k in rotors}
        out["turbine_free_control"] = {
            k: {"Ud": a, "spread": b, "drift_from_unity": a - 1.0}
            for k, (a, b) in free.items()}
        out["noise_floor_max_drift"] = max(abs(a - 1.0) for a, _ in free.values())
        print(f"    turbine-free control: worst drift from freestream "
              f"{100 * out['noise_floor_max_drift']:+.3f}%")
    else:
        uu, _vv = runs["free"]["state"]
        out["turbine_free_control"] = {}
        out["noise_floor_max_drift"] = float(np.max(np.abs(uu - wa.U_INF)))
        print(f"    no rotors at this rung; turbine-free field departs from "
              f"freestream by {out['noise_floor_max_drift']:.3e}")

    # the power table, on the parent's own control assignment: a turbine's
    # control is the run in which everything upstream of it is off
    if rotors:
        ud_solo, _ = settled(runs[f"solo_{first}"]["hist"], first)
        p_iso = power_of(ud_solo)
        total = 0.0
        out["turbines"] = {}
        for rot in r.tiling.rotors:
            rid = rot.rotor_id
            ctl = f"solo_{first}" if rid == first else "first_off"
            ud_a, sp = settled(runs["all"]["hist"], rid)
            ud_c, _ = settled(runs[ctl]["hist"], rid)
            pa, pc = power_of(ud_a), power_of(ud_c)
            total += pa
            out["turbines"][rid] = {
                "Ud_array": ud_a, "Ud_spread": sp, "Ud_control": ud_c,
                "P_array": pa, "P_control": pc,
                "wake_loss_pct": 100.0 * (1.0 - pa / pc), "control_run": ctl,
                "x_plane": rot.x_plane, "y_centre": rot.y_centre}
        eff = total / (len(rotors) * p_iso)
        out["isolated"] = {"Ud": ud_solo, "P": p_iso,
                           "momentum_theory_Ud": 2.0 / 3.0,
                           "rel_error": ud_solo / (2.0 / 3.0) - 1.0}
        out["array_efficiency"] = eff
        out["array_loss_pct"] = 100.0 * (1.0 - eff)
        print(f"    array loss  1 - sum P / ({len(rotors)} P_isolated) = "
              f"{100 * (1 - eff):.2f}%")

    states[(r.label, kind)] = {
        "all": runs["all"]["state"] if rotors else runs["free"]["state"],
        "prev": runs["all"]["prev"] if rotors else None,
        "free": runs["free"]["state"],
    }
    return out


# ---------------------------------------------------------------------------
# 2. the composed defect at a common state -- the F1 quantity
# ---------------------------------------------------------------------------


def blend_spread(pou, locals_u, locals_v):
    """sqrt(V_chi) on the velocity VECTOR, cellwise: the reference-free defect.

    ``V_chi = sum_i chi_i |w_i|^2 - |sum_i chi_i w_i|^2`` is the chi-weighted
    variance of the local one-macro-step solutions.  Three things make it the
    right common currency for the two columns:

      * **It needs no monolith**, so the Poseidon column can carry it, and W95
        says the Poseidon column can carry nothing else.
      * **It is exactly the term L6/C1's identity removes.**  Cellwise,
        ``|A(u) - u*|^2 = sum_i chi_i |u_i - u*|^2 - V_chi`` for any weights
        summing to one, so V_chi is the margin by which blending beats the
        weighted average of what it blends -- and it is non-negative for all data
        exactly when chi >= 0, which R11 enforces.
      * **On the overlap the common mode cancels**: ``u_i - u_j = D_i - D_j``
        there, so what it measures is the part of the restriction defect the cut
        itself creates and not the physics both windows agree about.

    It is identically zero at N=1, where chi == 1 and there is one local solve --
    which is the zero control, and it holds in BOTH columns.
    """
    v_u = pou.blend_variance(locals_u)
    v_v = pou.blend_variance(locals_v)
    return np.sqrt(np.maximum(v_u + v_v, 0.0))


def _norms(field, n_cells):
    return {"l2": float(np.linalg.norm(field)),
            "rms": float(np.linalg.norm(field) / np.sqrt(n_cells)),
            "max": float(np.max(np.abs(field))) if field.size else 0.0}


def solo_state(r, states, steps):
    """The ONE-turbine state, marched by the reference solver and cached.

    **The confound this exists to remove.** The headline defect is measured at
    each rung's own developed array, and a bigger array is not only more
    interfaces -- it is deeper wakes and stronger gradients. At N=24 the field
    reaches ``u_min = 0.183`` against N=2's ``0.509``, so a defect that grows
    with N could be growing because the physics got harder and not because the
    cut got longer, and the two are not separable in that measurement.

    With only the FIRST turbine running, the flow near it is the same at every
    rung -- same inflow, same disk, same wake, same position relative to the
    inlet -- and every window added beyond it sits in near-freestream. So a
    defect that still grows here is growing on interface count alone. It is the
    control that says which half of the headline is which.
    """
    key = (r.label, "solo")
    if key in states:
        return states[key]
    cache = os.path.join(OUT, f"solo_{r.label}.npz")
    if os.path.isfile(cache):
        z = np.load(cache)
        states[key] = (z["u"], z["v"])
        return states[key]
    rotors = [x.rotor_id for x in r.tiling.rotors]
    active = {rotors[0]} if rotors else set()
    u, v, _h, _p = march(r, active, steps, "reference")
    np.savez_compressed(cache, u=u, v=v)
    states[key] = (u, v)
    return states[key]


def stage_defect(r, state, experts_kind, art_rung):
    """The one-exchange-interval composition defect at one rung, both columns.

    The state is the SAME for both columns -- the classical composed field -- so
    the two numbers differ by the expert and by nothing else.  Measuring each
    column at its own developed state would confound the composition defect with
    the fact that the two experts are in different flows.
    """
    t = r.tiling
    pou = t.partition_of_unity()
    u, v = state
    out = {"rung": r.label, "n_overlaps": r.n_overlaps, "n_seams": r.n_seams,
           "n_windows": r.n_windows, "n_cells": r.n_cells}

    # Pi, and what it implies for C_mu -- geometry, so it is the same in both
    # columns and is reported once.
    out["Pi"] = pou.contaminated_weight()
    out["norm_A"] = pou.norm_A()
    out["chi_min"] = pou.chi_min()
    out["identity_residual"] = pou.identity_residual()
    out["W58_disjointness"] = pou.contaminated_multiplicity()

    fx = np.zeros(r.shape)                     # the disks are OFF for the defect
    for kind in ("reference", "poseidon"):
        ex = (wa.scaled_expert() if kind == "poseidon"
              else wa.reference_solver(wa.NU_REF))
        t0 = time.time()
        au, av, locals_ = composed_step(r, u, v, fx, kind, ex)
        lu = {k: w[0].reshape(-1) for k, w in locals_.items()}
        lv = {k: w[1].reshape(-1) for k, w in locals_.items()}
        row = {"wall_s": time.time() - t0}

        spread = blend_spread(pou, lu, lv)
        row["blend_spread"] = _norms(spread, r.n_cells)
        row["blend_spread"]["form"] = "reference-free"

        # The reference-free surrogate, in BOTH aggregations -- which is where
        # W58 turned out to have a second hypothesis nobody had stated. Both are
        # computed on the GLOBAL index space and on the joint (u, v) field, so a
        # cell's two values are directly comparable and the norm is the vector
        # one the bound is taken in.
        ng = r.n_cells
        glob = {}
        for name in t.names:
            g = np.zeros(2 * ng)
            g[pou.indices[name]] = lu[name]
            g[ng + pou.indices[name]] = lv[name]
            glob[name] = g
        support = {name: np.zeros(ng, dtype=bool) for name in t.names}
        for name in t.names:
            w = np.asarray(pou.weights[name], dtype=float)
            support[name][pou.indices[name][w > 0.0]] = True
        overlaps = {}
        for a, b in r.overlap_pairs():
            m = support[a] & support[b]
            if m.any():
                overlaps[(a, b)] = np.concatenate([m, m])
        nd = aggregated_neighbour_disagreement(glob, overlaps,
                                               n_global=2 * ng)
        row["neighbour_disagreement"] = nd["pairwise_max"]
        row["neighbour_disagreement_aggregated"] = nd["aggregated"]
        row["neighbour_disagreement_forms"] = nd

        if kind == "reference":
            # the monolith: the only place E exists
            mono = sl.reference_monolith(t.nx, t.ny, wa.NU_REF)
            mu, mv = mono.step_batch(u[None], v[None], wa.MACRO_DT, bc0=None,
                                     force=(fx[None], np.zeros_like(fx)[None]))
            Eu, Ev = mu[0], mv[0]
            comp = np.hypot(au - Eu, av - Ev)
            row["composed_defect"] = _norms(comp, r.n_cells)
            row["monolith_norm"] = float(np.linalg.norm(np.hypot(Eu, Ev)))
            row["composed_defect"]["relative"] = (
                row["composed_defect"]["l2"] / row["monolith_norm"]
                if row["monolith_norm"] else None)
            # L2/C2: D_i = E_i R_i u - R_i E u, on the joint (u, v) field.
            # **Lifted to the GLOBAL grid first.** The identity the bound
            # evaluates is A - Eu = sum_i R_i^T chi_i D_i, and `R_i^T` is exactly
            # what puts each window's contribution at its own cells; on a uniform
            # tiling the un-lifted arrays are all the same length, so passing them
            # raises nothing and stacks every window at cell 0. `n_global` makes
            # that an error rather than a silent inflation.
            ref_cut_u, ref_cut_v = t.cut(Eu), t.cut(Ev)
            steps_, refs_, wts = {}, {}, {}
            for k, name in enumerate(t.names):
                idx = pou.indices[name]
                gidx = np.concatenate([idx, ng + idx])
                s_ = np.zeros(2 * ng)
                r_ = np.zeros(2 * ng)
                w_ = np.zeros(2 * ng)
                s_[gidx] = np.concatenate([lu[name], lv[name]])
                r_[gidx] = np.concatenate([ref_cut_u[k].reshape(-1),
                                           ref_cut_v[k].reshape(-1)])
                w_[gidx] = np.concatenate([pou.weights[name], pou.weights[name]])
                steps_[name], refs_[name], wts[name] = s_, r_, w_
            cut = restriction_defect_bound(steps_, refs_, wts, n_global=2 * ng)
            row["cut_defect_bound"] = {
                "chi_weighted": cut["cut_defect_bound_chi_weighted"],
                "max": cut["cut_defect_bound_max"],
                "worst_agent": cut["worst_agent"],
            }
            actual = row["composed_defect"]["l2"]
            mx = cut["cut_defect_bound_max"]
            row["tightness"] = {
                "chi_weighted_over_actual": (
                    cut["cut_defect_bound_chi_weighted"] / actual
                    if actual > 0 else None),
                "max_over_actual": (mx / actual if actual > 0 else None),
                # W58's own equality, both aggregations. The theorem says the
                # AGGREGATED surrogate equals the MAX form under disjointness;
                # the pairwise one is compared beside it because that is the form
                # the framework has had since Tier 0.
                "aggregated_surrogate_over_max": (
                    nd["aggregated"] / mx if mx > 0 else None),
                "pairwise_surrogate_over_max": (
                    nd["pairwise_max"] / mx if mx > 0 else None),
            }
        out[kind] = row

    ref, pos = out["reference"], out["poseidon"]
    print(f"  {r.label:4s} int={r.n_overlaps:3d}  Pi={out['Pi']:.4f}  "
          f"spread(ref)={ref['blend_spread']['l2']:.4e}  "
          f"spread(pos)={pos['blend_spread']['l2']:.4e}")
    if "composed_defect" in ref:
        tg = ref["tightness"]
        print(f"        composed vs monolith = {ref['composed_defect']['l2']:.4e}  "
              f"chi-weighted bound = {ref['cut_defect_bound']['chi_weighted']:.4e}  "
              f"tightness = {_fmt(tg['chi_weighted_over_actual'])}")
        print(f"        W58 surrogate/max form: aggregated "
              f"{_fmt(tg['aggregated_surrogate_over_max'])}, pairwise "
              f"{_fmt(tg['pairwise_surrogate_over_max'])}  "
              f"(disjoint: {out['W58_disjointness']['disjoint']}, "
              f"{out['W58_disjointness']['shared_cells']} shared cells)")
    return out


def _fmt(x, spec=".4g"):
    return "n/a" if x is None else format(x, spec)


# ---------------------------------------------------------------------------
# 3. the accumulated defect -- what a user of the composed system carries
# ---------------------------------------------------------------------------


def fixed_physics_defect(r, state):
    """The composed defect at the one-turbine state: interfaces only.

    The same three numbers as the headline -- blend spread in both columns and
    the composed defect against the monolith -- taken at `solo_state`, where the
    physics is identical at every rung by construction.
    """
    t = r.tiling
    pou = t.partition_of_unity()
    u, v = state
    fx = np.zeros(r.shape)
    out = {"rung": r.label, "n_overlaps": r.n_overlaps, "n_cells": r.n_cells}
    for kind in ("reference", "poseidon"):
        ex = (wa.scaled_expert() if kind == "poseidon"
              else wa.reference_solver(wa.NU_REF))
        au, av, locals_ = composed_step(r, u, v, fx, kind, ex)
        lu = {k: w[0].reshape(-1) for k, w in locals_.items()}
        lv = {k: w[1].reshape(-1) for k, w in locals_.items()}
        row = {"blend_spread": _norms(blend_spread(pou, lu, lv), r.n_cells)}
        if kind == "reference":
            mono = sl.reference_monolith(t.nx, t.ny, wa.NU_REF)
            mu, mv = mono.step_batch(u[None], v[None], wa.MACRO_DT, bc0=None,
                                     force=(fx[None], np.zeros_like(fx)[None]))
            row["composed_defect"] = _norms(np.hypot(au - mu[0], av - mv[0]),
                                            r.n_cells)
            row["monolith_norm"] = float(np.linalg.norm(np.hypot(mu[0], mv[0])))
        out[kind] = row
    print(f"  {r.label:4s} int={r.n_overlaps:3d}  one-turbine control: "
          f"spread(ref)={out['reference']['blend_spread']['l2']:.4e}  "
          f"spread(pos)={out['poseidon']['blend_spread']['l2']:.4e}  "
          f"composed={out['reference'].get('composed_defect', {}).get('l2', 0):.4e}",
          flush=True)
    return out


def stage_stability(r, state0, steps, band=3.0):
    """Composed rollout against the monolith, from the developed state -- and the
    step at which the classical column stops being a rollout at all.

    **This stage was an accumulated-defect measurement until the measurement
    refused to be one.**  Three attempts, and each failure is on the record
    because each is the obvious thing to try:

      * *freestream, disks off* -- exactly **0.0000e+00** at every rung. A uniform
        field is a fixed point of the composed step AND of the monolith, so it is
        a correct answer to a useless question.
      * *developed state, disks FROZEN* -- diverges. An actuator disk's thrust
        must track its own inflow; freezing it removes the closure's negative
        feedback and the deceleration runs away. That instability is the disk
        model's and would have been charged to the composition.
      * *developed state, disks live or off* -- **also diverges, and this one is
        the composition's.**  At N = 6 the classical composed rollout leaves the
        band between macro-step 70 and 80 and is not finite by 82; the monolith,
        from the same state under the same forcing, holds ``u_max`` at 1.33
        throughout. The Poseidon column is stable to 110.

    So what is reported here is the **divergence onset** and the mechanism, plus
    the accumulated defect at a short horizon where every column is still inside
    the band.  A defect quoted past the onset is a number about a blow-up.

    ``+projection`` was the positive control and is now the **declared step**
    (W100, 2026-08-31): the classical column assembled through
    `wake_array.projected_assembly`, which is a blend followed by one global
    Leray projection on the assembled field.  It is what `L6/C2` states, what
    `R12` checks at compile time, and what the Poseidon column has done since
    W98.  The tag is kept for continuity with the numbers already published.
    """
    out = {"rung": r.label, "n_steps": steps, "band": band,
           "short_horizon": min(10, steps)}
    rotors = {x.rotor_id for x in r.tiling.rotors}
    mono = sl.reference_monolith(r.tiling.nx, r.tiling.ny, wa.NU_REF)

    # the referent trajectory, once
    mu, mv = np.array(state0[0]), np.array(state0[1])
    mono_traj = []
    for _ in range(steps):
        fx, _rec = _forcing(r, mu, rotors)
        uu, vv = mono.step_batch(mu[None], mv[None], wa.MACRO_DT, bc0=None,
                                 force=(fx[None], np.zeros_like(fx)[None]))
        mu, mv = _band(r, uu[0], vv[0])
        mono_traj.append((mu.copy(), mv.copy()))
    out["monolith"] = {
        "u_max": float(max(np.max(a) for a, _ in mono_traj)),
        "u_min": float(min(np.min(a) for a, _ in mono_traj)),
        "div_rms_final": divergence_rms(*mono_traj[-1]),
        "finite": bool(np.all(np.isfinite(mono_traj[-1][0]))),
    }

    for tag, kind, proj in (("reference", "reference", False),
                            ("reference+projection", "reference", True),
                            ("poseidon", "poseidon", False)):
        ex = (wa.scaled_expert() if kind == "poseidon"
              else wa.reference_solver(wa.NU_REF))
        assembly = assembly_for(r) if proj else None
        u, v = np.array(state0[0]), np.array(state0[1])
        row = {"div_rms": [], "u_max": [], "defect_l2": [], "diverged_at": None}
        t0 = time.time()
        for k in range(steps):
            fx, _rec = _forcing(r, u, rotors)
            u, v, _loc = composed_step(r, u, v, fx, kind, ex, assembly=assembly)
            u, v = _band(r, u, v)
            finite = bool(np.all(np.isfinite(u)) and np.all(np.isfinite(v)))
            umax = float(np.max(np.abs(u))) if finite else float("inf")
            row["u_max"].append(umax if finite else None)
            row["div_rms"].append(divergence_rms(u, v) if finite else None)
            mo_u, mo_v = mono_traj[k]
            row["defect_l2"].append(
                float(np.linalg.norm(np.hypot(u - mo_u, v - mo_v)))
                if finite else None)
            if row["diverged_at"] is None and (not finite or umax > band):
                row["diverged_at"] = k + 1
            if not finite:
                break
        row["wall_s"] = time.time() - t0
        h = row["short_horizon"] = out["short_horizon"]
        row["defect_at_short_horizon"] = (row["defect_l2"][h - 1]
                                          if len(row["defect_l2"]) >= h else None)
        row["stable"] = row["diverged_at"] is None
        out[tag] = row
        print(f"  {r.label:4s} {tag:22s} diverged_at={row['diverged_at']}  "
              f"div_rms {_fmt(row['div_rms'][0])} -> "
              f"{_fmt(next((x for x in reversed(row['div_rms']) if x is not None), None))}"
              f"  defect@{h}={_fmt(row['defect_at_short_horizon'])}", flush=True)
    return out



def stage_r12(r):
    """The compile-side half of W100: R12 at both declarations, at this rung.

    The repair is a *declaration* before it is an operator, so the record has to
    show the compile deciding it.  Two graphs per column, identical but for the
    assembly: `scaling_ladder.build(..., assembly_projection=False)` is the bare
    partition of unity the case study shipped, and True is the projected assembly
    of L6/C2.  Nothing else differs, which is the point -- one field on one
    object moves one rule.
    """
    u = np.ones(r.shape)
    v = np.zeros(r.shape)
    out = {"rung": r.label, "n_windows": r.n_windows, "columns": []}
    for kind, ell in (("reference", EllipticSubsolve.EMBEDDED),
                      ("poseidon", EllipticSubsolve.UNKNOWN)):
        for proj in (False, True):
            try:
                graph, _ = sl.build(u, v, r, kind=kind, elliptic=ell,
                                    assembly_projection=proj)
            except Exception as exc:                               # noqa: BLE001
                # N=1 has no artificial face, so it has no port and the record
                # refuses before the compiler is reached. That is the control,
                # not a failure -- see `scaling_ladder.build`.
                out["columns"].append({"expert": kind, "assembly_projection": proj,
                                       "refused_at_declaration": repr(exc)})
                print(f"  {r.label:4s} {kind:9s} "
                      f"{'projected' if proj else 'bare     '}  "
                      f"refused at the declaration: {exc}", flush=True)
                continue
            # **No probe.** R12 is decided from the declaration -- scope, stage,
            # cadence -- exactly as R11 decides convexity from chi, and that is
            # the property being demonstrated. Letting `Budget()` default to
            # `allow_probe=True` here would run the transmission probe at every
            # rung for a rule that never reads it; measured, that is 134 s at
            # N=6 on the checkpoint and is why `stage_compile` gates it at 30
            # seams.
            res = compile_scheme(graph, budget=Budget(allow_probe=False))
            r12 = [d for d in (list(res.decisions.refusals)
                               + list(res.decisions.decertifications)
                               + list(res.decisions.of_verdict(ADMIT)))
                   if d.rule == "R12"]
            harness = HarnessParameters.from_graph(graph)
            out["columns"].append({
                "expert": kind,
                "assembly_projection": proj,
                "verdict": res.verdict.name,
                "R12": [{"verdict": d.verdict.name, "message": d.message}
                        for d in r12],
                "harness_assembly_projection": harness.assembly_projection,
            })
            verdicts = ",".join(d.verdict.name for d in r12) or "-"
            print(f"  {r.label:4s} {kind:9s} "
                  f"{'projected' if proj else 'bare     '}  "
                  f"verdict={res.verdict.name:18s} R12={verdicts}", flush=True)
    return out


def stage_long_march(r, steps, band=3.0, snap_every=20, control_rungs_up_to=6):
    """**W100.** The CS-7 march, continued past the 60 macro-steps Tier 18 published.

    Tier 18 marched the wake array 60 macro-steps and published every number
    measured at that state.  Nothing marched further, and the reason it matters
    is that the classical composed column does not survive much further: from the
    freestream, at six windows, it leaves the band between step 70 and 80 and is
    **not finite by macro-step 82**, while the monolith on the same domain under
    the same forcing holds ``u_max`` at 1.33 throughout.

    Four columns, from the freestream, with the disks live, at one rung:

      ``monolith``            the referent -- no cut, so no assembly, so no
                              commutator. It is what says the blow-up belongs to
                              the composition and not to the physics or the disk
                              model.
      ``reference``                    the classical composed column with the
                                       assembly the case study shipped before
                                       2026-08-31: a bare partition-of-unity
                                       blend, and its elliptic part inside each
                                       agent. This is the reproduction.
      ``reference+projected``          the same agents with L6/C2's step added on
                                       top. **This is the arrangement section
                                       19.6 called the positive control, and at
                                       this length it fails**: the pressure is
                                       applied twice, once per window and once
                                       globally, and the column leaves the band
                                       EARLIER than the bare one.
      ``reference-exposed+projected``  the elliptic part taken OUT of the agents
                                       (`wake_array.exposed_reference_solver`)
                                       and applied once, globally, after the
                                       blend. R10 + R10b + R12 together, which is
                                       what the framework has prescribed since
                                       Tier 0. This is the stable one.
      ``reference-exposed``            the same agents with no global elliptic
                                       part at all -- the control that says the
                                       projection is doing the work rather than
                                       the removal.
      ``poseidon``                     the checkpoint's column, which has had
                                       exactly this arrangement since W98 and is
                                       the reason it was stable while the
                                       classical one was not.

    What is reported per column is the whole ``||div u||_rms`` trace rather than
    an endpoint, because the claim is about a TREND: L6/C2 says the blend
    manufactures constraint residual every macro-step, so an assembly without the
    projection accumulates it and one with it does not.  ``div_trend`` is the
    last finite value over the first.

    The monolith's state is kept every ``snap_every`` steps and every composed
    column is measured against it there, so "stable" is not the whole claim --
    a column that stayed finite while drifting off the referent would be visible
    as a rising ``defect_l2``.  Keeping every step would be 5 MB a step at the
    top rung, which is why it is a subsample and why the subsample is declared.
    """
    out = {"rung": r.label, "n_windows": r.n_windows, "n_overlaps": r.n_overlaps,
           "n_steps": steps, "band": band, "snap_every": snap_every,
           "columns": {}}
    rotors = {x.rotor_id for x in r.tiling.rotors}
    snaps: dict[int, tuple] = {}

    def record(tag, advance, keep_snaps=False):
        row = {"u_max": [], "div_rms": [], "diverged_at": None, "finite_to": 0,
               "defect_at": {}}
        u, v = np.ones(r.shape), np.zeros(r.shape)
        t0 = time.time()
        for k in range(steps):
            fx, _rec = _forcing(r, u, rotors)
            u, v = advance(u, v, fx)
            u, v = _band(r, u, v)
            finite = bool(np.all(np.isfinite(u)) and np.all(np.isfinite(v)))
            if not finite:
                row["u_max"].append(None)
                row["div_rms"].append(None)
                if row["diverged_at"] is None:
                    row["diverged_at"] = k + 1
                row["not_finite_at"] = k + 1
                break
            row["finite_to"] = k + 1
            umax = float(np.max(np.abs(u)))
            row["u_max"].append(umax)
            row["div_rms"].append(divergence_rms(u, v))
            if (k + 1) % snap_every == 0:
                if keep_snaps:
                    snaps[k + 1] = (u.copy(), v.copy())
                elif (k + 1) in snaps:
                    mu, mv = snaps[k + 1]
                    row["defect_at"][k + 1] = float(
                        np.linalg.norm(np.hypot(u - mu, v - mv)))
                # **An instrument that reports nothing until it finishes is how a
                # stall reads as a result.** This one printed once per COLUMN,
                # and a column that should take 40 s took 14 minutes with the
                # log saying nothing at all -- so it says where it is.
                print(f"      {r.label} {tag:22s} step {k + 1:4d}/{steps}  "
                      f"u_max={_fmt(umax)}  {time.time() - t0:6.0f}s", flush=True)
            if row["diverged_at"] is None and umax > band:
                row["diverged_at"] = k + 1
            if umax > 10.0 * band:
                # **Stop, and say that the loop stopped rather than the field
                # became non-finite.** Past the band every further step is a
                # number about a blow-up -- section 19.6's own argument for
                # re-scoping the rollout defect -- and `WindowNS` picks its
                # sub-step count from the advective CFL, so a column at
                # u_max = 121 costs a hundred times a healthy one per macro-step.
                # Marching a diverged column to 120 buys nothing and costs hours.
                row["stopped_at"] = k + 1
                row["stopped_because"] = (
                    f"|u| = {umax:.4g} exceeded 10x the band {band}; the column "
                    "had already left the band at macro-step "
                    f"{row['diverged_at']}")
                break
        row["wall_s"] = time.time() - t0
        fin = [x for x in row["div_rms"] if x is not None]
        row["div_rms_first"] = fin[0] if fin else None
        row["div_rms_last"] = fin[-1] if fin else None
        row["div_rms_max"] = max(fin) if fin else None
        row["div_rms_argmax"] = (int(np.argmax(fin)) + 1) if fin else None
        row["div_trend"] = (fin[-1] / fin[0]) if fin and fin[0] > 0 else None
        # **The trend that answers the claim is the DEVELOPED one.** A march
        # from the freestream is a transient: the wake is still forming for the
        # first half of it and every column's divergence rises while it does,
        # which is the physics rather than the assembly. So the quantity is the
        # last quarter's mean over the third quarter's -- both taken after the
        # flow has developed, and both on the same column -- and `div_trend`
        # (last over first) is kept beside it because it is the number a reader
        # would compute and it should not be hidden for being less kind.
        q = len(fin) // 4
        if q >= 2:
            third = float(np.mean(fin[2 * q:3 * q]))
            fourth = float(np.mean(fin[3 * q:]))
            row["div_tail_trend"] = (fourth / third) if third > 0 else None
            row["div_tail_window"] = [2 * q + 1, 3 * q, 3 * q + 1, len(fin)]
        else:
            row["div_tail_trend"] = None
            row["div_tail_window"] = None
        row["flat_or_falling"] = (None if row["div_tail_trend"] is None
                                  else bool(row["div_tail_trend"] <= 1.0))
        seen = [x for x in row["u_max"] if x is not None]
        row["u_max_overall"] = max(seen) if seen else None
        row["stable"] = bool(row["diverged_at"] is None
                             and row["finite_to"] == steps
                             and row.get("stopped_at") is None)
        out["columns"][tag] = row
        print(f"  {r.label:4s} {tag:22s} finite_to={row['finite_to']:4d}/{steps} "
              f"diverged_at={str(row['diverged_at']):5s} "
              f"u_max={_fmt(row['u_max_overall'])}  div {_fmt(row['div_rms_first'])} "
              f"-> {_fmt(row['div_rms_last'])}  ({row['wall_s']:.0f}s)", flush=True)

    mono = sl.reference_monolith(r.tiling.nx, r.tiling.ny, wa.NU_REF)

    def advance_mono(u, v, fx):
        uu, vv = mono.step_batch(u[None], v[None], wa.MACRO_DT, bc0=None,
                                 force=(fx[None], np.zeros_like(fx)[None]))
        return uu[0], vv[0]

    record("monolith", advance_mono, keep_snaps=True)

    columns = [("reference", "reference", False),
               ("reference+projected", "reference", True),
               ("reference-exposed+projected", "reference_exposed", True),
               # The control that says the projection is load-bearing rather
               # than incidental: the same exposed agents with NO global
               # elliptic part at all. It is a mechanism control and not a
               # scaling claim, so it runs on the small rungs only.
               ("reference-exposed", "reference_exposed", False),
               ("poseidon", "poseidon", False)]
    if r.n_windows > control_rungs_up_to:
        columns = [c for c in columns if c[0] != "reference-exposed"]

    for tag, kind, proj in columns:
        ex = (wa.scaled_expert() if kind == "poseidon"
              else wa.exposed_reference_solver(wa.NU_REF)
              if kind == "reference_exposed" else wa.reference_solver(wa.NU_REF))
        assembly = assembly_for(r) if proj else None

        def advance(u, v, fx, kind=kind, ex=ex, assembly=assembly):
            u1, v1, _loc = composed_step(r, u, v, fx, kind, ex, assembly=assembly)
            return u1, v1

        record(tag, advance)

    # the verdict this stage exists to return, stated rather than left to a reader
    cols = out["columns"]
    out["verdict"] = {
        "bare_assembly_survives": cols.get("reference", {}).get("stable"),
        "projected_assembly_survives": cols.get("reference+projected", {}).get("stable"),
        "checkpoint_survives": cols.get("poseidon", {}).get("stable"),
        "monolith_survives": cols.get("monolith", {}).get("stable"),
        "bare_not_finite_at": cols.get("reference", {}).get("not_finite_at"),
        "projected_div_trend": cols.get("reference+projected", {}).get("div_trend"),
        "projected_div_tail_trend":
            cols.get("reference+projected", {}).get("div_tail_trend"),
        "projected_div_flat_or_falling":
            cols.get("reference+projected", {}).get("flat_or_falling"),
        # the two columns side by side at the last step BOTH reached, which is
        # the comparison the mechanism predicts and the only one that is not a
        # comparison between a number and a NaN
        "div_ratio_bare_over_projected": _bare_over_projected(cols),
    }
    return out


def _bare_over_projected(cols):
    """``||div u||`` of the bare assembly over the projected one, at the last
    macro-step both were finite. None when the bare column never ran."""
    bare = cols.get("reference", {}).get("div_rms") or []
    proj = cols.get("reference+projected", {}).get("div_rms") or []
    k = min(len([x for x in bare if x is not None]),
            len([x for x in proj if x is not None]))
    if k == 0 or not proj[k - 1]:
        return None
    return {"at_step": k, "bare": bare[k - 1], "projected": proj[k - 1],
            "ratio": bare[k - 1] / proj[k - 1]}


# ---------------------------------------------------------------------------
# 4-7. the seam: tau/sigma/gamma, and the substitution certificate
# ---------------------------------------------------------------------------


def representative_seam(r):
    """One seam per rung, chosen by the same rule at every rung.

    The parent's headline seam is ``x1r0_bypass`` -- the open flow beside R2, on
    the plane 3.5 D downstream of R1, so it is inside a developed wake and it is
    fluid-to-fluid, where the port's own nu cancels exactly.  The rule that
    picks it is "the highest-column bypass seam of row 0", and it picks the same
    kind of seam at every rung.  Falls back to the first fluid-fluid seam where
    no bypass exists.
    """
    conns = wa.connections(r.tiling)
    byp = [c for c in conns if c.seam_id.endswith("r0_bypass")]
    if byp:
        return max(byp, key=lambda c: c.seam_id).seam_id
    full = [c for c in conns if c.seam_id.endswith("_full")]
    return full[0].seam_id if full else (conns[0].seam_id if conns else None)


def transfer_for(graph, seam_id):
    conn = graph.connection(seam_id)
    prolongations, dims = {}, []
    for agent_id, name in (conn.a, conn.b):
        p = graph.agent(agent_id).port(name).prolongation
        prolongations[agent_id] = p
        dims.append(p.dim_M)
    if dims[0] != dims[1]:
        raise ValueError(f"{seam_id}: dim M disagrees, {dims}")
    return SeamTransfer(
        seam_id=seam_id, port_type=conn.port_type,
        space=InterfaceSpace(seam_id=seam_id, dim=dims[0],
                             note="derived: dim M = min_i m_i_eff"),
        prolongations=prolongations)


def seam_base_M(graph, experts, seam_id):
    conn = graph.connection(seam_id)
    tr = transfer_for(graph, seam_id)
    coeffs = []
    for agent_id, name in (conn.a, conn.b):
        base_V = np.asarray(experts[agent_id].probe_base(name), dtype=float)
        R = tr.prolongations[agent_id].adjoint(tr.space)
        coeffs.append(R @ base_V)
    return 0.5 * (coeffs[0] + coeffs[1]), coeffs


def seam_callables(graph, experts, seam_id):
    conn = graph.connection(seam_id)
    return {agent_id: (lambda t, a=agent_id, n=name: experts[a].respond(n, t))
            for agent_id, name in (conn.a, conn.b)}


def stage_seam(r, state, prev_state, harness):
    """tau, sigma and gamma on one wake seam, plus the W81 certificate.

    Depth AND harness tagged, which is **W54**: three composition-layer defects
    had worn an agent's label before W98 made it four, and every one of them was
    a function of the overlap, chi's shape, the exchange cadence or the elliptic
    placement -- none of which the depth tag records.  This ladder varies exactly
    one harness parameter by design (the tiling), so anything else moving in the
    tag is a bug the schema now shows.
    """
    seam_id = representative_seam(r)
    if seam_id is None:
        return {"rung": r.label, "seam": None,
                "note": "no seams at this rung: a single window is not a composition"}
    out = {"rung": r.label, "seam": seam_id, "harness": harness.as_dict(),
           "depth": 0}

    ex_p = sl.make_experts(*state, r, "poseidon")
    ex_r = sl.make_experts(*state, r, "reference")
    g_p, _ = sl.build(*state, r, kind="poseidon", experts=ex_p)
    g_r, _ = sl.build(*state, r, kind="reference", experts=ex_r)
    conn = g_p.connection(seam_id)
    ids = [conn.a[0], conn.b[0]]
    names = {conn.a[0]: conn.a[1], conn.b[0]: conn.b[1]}
    halves = {a: ResponseHalf.EFFORT for a in ids}

    tr = transfer_for(g_p, seam_id)
    base_M, sides = seam_base_M(g_p, ex_p, seam_id)
    op_r = assemble_seam(g_r, g_r.connection(seam_id), tr, budget=ProbeBudget(),
                         expected_null_dim=conn.expected_null_dim,
                         probe_state=f"CS-7 rung {r.label}, developed array",
                         seam_base=base_M)
    op_p = assemble_seam(g_p, conn, tr, budget=ProbeBudget(),
                         expected_null_dim=conn.expected_null_dim,
                         probe_state=f"CS-7 rung {r.label}, developed array",
                         seam_base=base_M)
    out["operator"] = {
        "reference": {"beta": op_r.beta, "kappa": op_r.kappa,
                      "norm": float(np.linalg.norm(op_r.S, 2)),
                      "operator_content": op_r.operator_content,
                      "one_sided": op_r.one_sided, "dim_M": op_r.dim_M},
        "poseidon": {"beta": op_p.beta, "kappa": op_p.kappa,
                     "norm": float(np.linalg.norm(op_p.S, 2)),
                     "operator_content": op_p.operator_content,
                     "one_sided": op_p.one_sided, "dim_M": op_p.dim_M},
        "Xi": (float(np.linalg.norm(op_p.S, 2) / np.linalg.norm(op_r.S, 2))
               if np.linalg.norm(op_r.S, 2) > 0 else None),
        "seam_base_spread": float(np.max(np.abs(sides[0] - sides[1]))),
    }

    # tau: Poseidon substituted for the reference solver, against the reference
    # PAIR -- which is what `lambda_ref` names, because there is no monolith for
    # the checkpoint at any resolution (W95).
    trace0 = np.mean([np.asarray(ex_p[a].probe_base(names[a])) for a in ids], axis=0)
    lagged = trace0
    if prev_state is not None:
        prev_ex = sl.make_experts(*prev_state, r, "poseidon")
        lagged = np.mean([np.asarray(prev_ex[a].probe_base(names[a])) for a in ids],
                         axis=0)
    out["lag_distance"] = float(lag_distance(lagged, trace0))
    rc = seam_callables(g_r, ex_r, seam_id)
    pc = seam_callables(g_p, ex_p, seam_id)
    tc = tight_couple(lambda lam: sum(np.asarray(rc[a](lam)).ravel() for a in ids),
                      trace0)
    out["referent"] = {"converged": bool(tc.converged),
                       "iterations": int(tc.iterations),
                       "residual_norm": float(tc.residual_norm)}
    sd = seam_defect_split(seam_id, dict(pc), rc, halves, trace0, lagged,
                           measure=wa.DX, jacobian="fd")
    out["split"] = sd.as_dict()
    tau_tot = (sum(float(x) for x in sd.tau.values()) if sd.converged else None)
    out["tau_total"] = tau_tot
    out["sigma"] = float(sd.sigma) if sd.converged else None
    # gamma -- the SOLVE infidelity -- is not a seam quantity and
    # `seam_defect_split` does not produce one. On a halo scheme there is no
    # interface solve to be infidel to, which is W49's own observation, and what
    # plays gamma's part is the assembly: it is measured globally in stage 2 as
    # the composed defect against the monolith, not here. Recorded as absent
    # rather than as zero, because zero is a measurement and this is not one.
    out["gamma"] = None
    out["gamma_note"] = (
        "not measured at a seam: a halo scheme poses no interface problem, so "
        "there is no solve infidelity localized here. The assembly's own "
        "contribution is the composed-defect-minus-bound residual in stage 2")

    # **W81.** The same swap, three ways: no tolerance (thresholds reported),
    # a tolerance DERIVED from this rung's own defect terms, and the Tier 18
    # sweep for continuity.
    target = next(a for a, _n in (conn.a, conn.b) if a.startswith("F"))
    S_old = op_r.blocks[target].S
    S_new = op_p.blocks[target].S
    bn = float(np.linalg.norm(S_old, 2))
    kw = dict(passivity_old=op_r.blocks[target].passivity_lambda_min,
              passivity_new=op_p.blocks[target].passivity_lambda_min,
              block_norm=bn)
    caps_old = g_r.agent(target).capabilities
    caps_new = g_p.agent(target).capabilities
    none_cert = certify_substitution(target, caps_old, caps_new, S_old, S_new,
                                     beta=op_r.beta, **kw)
    eps, eps_src = beta_min_from_tolerance(tau_tot, out["sigma"])
    derived = certify_substitution(target, caps_old, caps_new, S_old, S_new,
                                   beta=op_r.beta, tau=tau_tot,
                                   sigma=out["sigma"], **kw)
    sweep = []
    for bm in (0.0, 1e-12, 0.01, 0.1, 0.25, 0.5):
        sweep.append({"beta_min": bm,
                      **certify_substitution(target, caps_old, caps_new, S_old,
                                             S_new, beta=op_r.beta, beta_min=bm,
                                             **kw).as_dict()})
    out["W81"] = {
        "target": target, "block_norm": bn, "beta": op_r.beta,
        "no_tolerance": none_cert.as_dict(),
        "eps_tol": eps, "eps_tol_source": eps_src,
        "derived": derived.as_dict(),
        "sweep": sweep,
        "derived_inside_window": (
            None if (eps is None or none_cert.visible_above is None)
            else bool(none_cert.visible_above <= eps <= none_cert.fails_above)),
    }
    print(f"  {r.label:4s} seam {seam_id:14s} beta={op_r.beta:.4g} "
          f"||S_i||={bn:.4g} tau={_fmt(tau_tot)} sigma={_fmt(out['sigma'])}")
    print(f"        W81: visible above beta_min={_fmt(none_cert.visible_above)}, "
          f"refused above {_fmt(none_cert.fails_above)}; derived eps_tol="
          f"{_fmt(eps)} -> inside window: {out['W81']['derived_inside_window']}")
    return out


# ---------------------------------------------------------------------------
# 8. the compiles
# ---------------------------------------------------------------------------


def measured_for(r, art, kind):
    """This rung's own constants, in the form each column can actually supply.

    **This is where W58 is demonstrated rather than described.**  The classical
    column has a monolith, so it can declare the CHI-WEIGHTED form -- L2/C2's own
    quantity.  The checkpoint has no monolith at any resolution (W95), so the
    only form it can supply is the REFERENCE-FREE neighbour disagreement, and
    whether that may be quoted as the bound is decided by the disjointness of the
    contaminated sets, which this ladder makes fail somewhere between N=2 and
    N=24.  Two columns, two forms, one field on the record saying which -- which
    is the whole of the row.
    """
    d = next((x for x in art.get("defect", []) if x["rung"] == r.label), None)
    if d is None:
        return None
    seam = next((x for x in art.get("seam", []) if x["rung"] == r.label), None)
    tau = seam.get("tau_total") if seam else None
    sigma = seam.get("sigma") if seam else None
    if kind == "reference":
        value = d["reference"].get("cut_defect_bound", {}).get("chi_weighted")
        form = "chi-weighted"
        tau, sigma = 0.0, sigma          # the referent's infidelity against
        #                                  itself is zero by construction
    else:
        value = d["poseidon"]["neighbour_disagreement_aggregated"]
        form = "neighbour-disagreement-aggregated"
    if value is None:
        return None
    pi = d.get("Pi")
    return MeasuredConstants(
        L=None, tau=tau, sigma=sigma, gamma=None,
        C_mu=C_MU_HALO, norm_A=d.get("norm_A"),
        cut_defect_bound=float(value), cut_defect_bound_form=form,
        probe_state=f"CS-7 rung {r.label}, developed array, disks off",
        scheme="overlapping halo, one exchange per macro-step",
        depth=0,
        source=(f"scripts/w100_scaling_ladder.py, Pi = {pi:.4g}"
                if pi is not None else "scripts/w100_scaling_ladder.py"))


def stage_compile(r, state, probe_max_seams, art):
    out = {"rung": r.label, "n_seams": r.n_seams, "n_windows": r.n_windows}
    if r.n_seams == 0:
        try:
            sl.build(*state, r, kind="reference")
            out["error"] = "expected a refusal at N=1 and did not get one"
        except Exception as exc:                                   # noqa: BLE001
            out["record_refusal"] = f"{type(exc).__name__}: {exc}"
            print(f"  {r.label:4s} the RECORD refuses before the compiler is "
                  f"reached: {exc}")
        return out
    for kind in ("reference", "poseidon"):
        measured = measured_for(r, art, kind)
        out.setdefault("measured", {})[kind] = (
            measured.as_dict() if measured is not None else None)
        for probe in (False, True):
            if probe and r.n_seams > probe_max_seams:
                continue
            t0 = time.time()
            g, _ = sl.build(*state, r, kind=kind, measured=measured,
                            elliptic=EllipticSubsolve.UNKNOWN)
            res = compile_scheme(
                g, budget=Budget(allow_probe=probe),
                probe_state=f"CS-7 rung {r.label}, developed array")
            key = f"{kind}{'_probed' if probe else ''}"
            groups = {}
            for d in res.decisions.decertifications:
                groups[f"{d.layer}/{d.rule}"] = groups.get(f"{d.layer}/{d.rule}", 0) + 1
            c2 = [d.as_dict() for d in res.decisions
                  if d.rule.startswith("C2")]
            out[key] = {
                "verdict": res.verdict.value,
                "L2_C2": c2,
                "n_refusals": len(res.decisions.refusals),
                "n_decertifications": len(res.decisions.decertifications),
                "decertifications_by_rule": groups,
                "refusals_by_rule": sorted({f"{d.layer}/{d.rule}"
                                            for d in res.decisions.refusals}),
                "unmeasured": list(res.unmeasured),
                "stamp": {h.name: st.value for h, st in res.envelope.values.items()},
                "harness": (res.artifact.bound_terms.harness.as_dict()
                            if getattr(res, "artifact", None) is not None
                            and res.artifact.bound_terms.harness is not None
                            else None),
                "wall_s": time.time() - t0,
            }
            print(f"  {r.label:4s} {key:18s} {res.verdict.value:18s} "
                  f"{len(res.decisions.refusals):3d} refusals, "
                  f"{len(res.decisions.decertifications):3d} decertifications "
                  f"[{time.time() - t0:.1f}s]", flush=True)
    return out


# ---------------------------------------------------------------------------
# 9. the fit -- an exponent with an error bar, not an adjective
# ---------------------------------------------------------------------------


def power_fit(x, y, label=""):
    """Ordinary least squares of log y on log x, with a t-based interval.

    Four rungs carry a nonzero interface count, so the fit has ``n - 2 = 2``
    degrees of freedom and the 95% interval uses ``t(0.975, 2) = 4.303``.  That
    interval is wide and it is honestly wide: it is what four points buy.  The
    pairwise local slopes are reported beside it, because a fitted exponent
    averages over curvature and curvature is exactly what a super-linear onset
    would look like.
    """
    x = np.asarray([float(v) for v in x], dtype=float)
    y = np.asarray([float(v) for v in y], dtype=float)
    keep = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 2:
        return {"label": label, "n": int(x.size), "exponent": None,
                "note": "fewer than two usable points"}
    lx, ly = np.log(x), np.log(y)
    n = lx.size
    A = np.column_stack([lx, np.ones(n)])
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    resid = ly - A @ coef
    dof = n - 2
    out = {"label": label, "n": int(n), "exponent": slope,
           "prefactor": float(np.exp(intercept)),
           "x": [float(v) for v in x], "y": [float(v) for v in y],
           "local_slopes": [
               {"from": float(x[i]), "to": float(x[i + 1]),
                "slope": float((ly[i + 1] - ly[i]) / (lx[i + 1] - lx[i]))}
               for i in range(n - 1)],
           }
    if dof > 0:
        s2 = float(resid @ resid) / dof
        sxx = float(np.sum((lx - lx.mean()) ** 2))
        se = float(np.sqrt(s2 / sxx)) if sxx > 0 else float("inf")
        tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}.get(dof, 2.0)
        out.update({"stderr": se, "dof": dof, "t_crit_95": tcrit,
                    "ci95": [slope - tcrit * se, slope + tcrit * se],
                    "r2": float(1.0 - (resid @ resid) /
                                max(float(np.sum((ly - ly.mean()) ** 2)), 1e-300)),
                    "super_linear": bool(slope - tcrit * se > 1.0),
                    "sub_linear": bool(slope + tcrit * se < 1.0)})
    else:
        out.update({"stderr": None, "dof": dof,
                    "note": "two points: a slope with no residual and no interval"})
    return out


def stage_fit(art):
    banner("9. the fit -- how composition error grows with interface count")
    rows = art["defect"]
    series = {}
    for key, getter in (
        ("blend_spread_l2_reference",
         lambda d: d["reference"]["blend_spread"]["l2"]),
        ("blend_spread_l2_poseidon",
         lambda d: d["poseidon"]["blend_spread"]["l2"]),
        ("blend_spread_rms_reference",
         lambda d: d["reference"]["blend_spread"]["rms"]),
        ("blend_spread_rms_poseidon",
         lambda d: d["poseidon"]["blend_spread"]["rms"]),
        ("blend_spread_max_reference",
         lambda d: d["reference"]["blend_spread"]["max"]),
        ("blend_spread_max_poseidon",
         lambda d: d["poseidon"]["blend_spread"]["max"]),
        ("composed_defect_l2_reference",
         lambda d: d["reference"].get("composed_defect", {}).get("l2")),
        ("cut_defect_bound_chi_weighted",
         lambda d: d["reference"].get("cut_defect_bound", {}).get("chi_weighted")),
        ("neighbour_disagreement_reference",
         lambda d: d["reference"]["neighbour_disagreement"]),
        ("neighbour_disagreement_poseidon",
         lambda d: d["poseidon"]["neighbour_disagreement"]),
        ("aggregated_surrogate_reference",
         lambda d: d["reference"]["neighbour_disagreement_aggregated"]),
        ("aggregated_surrogate_poseidon",
         lambda d: d["poseidon"]["neighbour_disagreement_aggregated"]),
        # PER INTERFACE. If the composition defect is local to the overlaps and
        # its per-overlap size is constant, the total adds in quadrature and
        # grows like sqrt(interfaces) -- a benign result that must not be
        # reported as a decay. This series is the total divided by
        # sqrt(interfaces), so a FLAT one (exponent 0) says exactly that, and a
        # rising one is the thing F1 is actually afraid of.
        ("per_interface_reference",
         lambda d: (d["reference"]["blend_spread"]["l2"]
                    / np.sqrt(d["n_overlaps"])) if d["n_overlaps"] else None),
        ("per_interface_poseidon",
         lambda d: (d["poseidon"]["blend_spread"]["l2"]
                    / np.sqrt(d["n_overlaps"])) if d["n_overlaps"] else None),
        ("composed_defect_relative_reference",
         lambda d: d["reference"].get("composed_defect", {}).get("relative")),
    ):
        xs, ys = [], []
        for d in rows:
            try:
                y = getter(d)
            except (KeyError, TypeError):
                y = None
            if y is None:
                continue
            xs.append(d["n_overlaps"])
            ys.append(y)
        series[key] = power_fit(xs, ys, key)

    for key, getter in (
        ("fixed_physics_spread_reference",
         lambda d: d["reference"]["blend_spread"]["l2"]),
        ("fixed_physics_spread_poseidon",
         lambda d: d["poseidon"]["blend_spread"]["l2"]),
        ("fixed_physics_composed_reference",
         lambda d: d["reference"].get("composed_defect", {}).get("l2")),
    ):
        xs, ys = [], []
        for d in art.get("fixed_physics", []):
            try:
                y = getter(d)
            except (KeyError, TypeError):
                y = None
            if y is None or not d["n_overlaps"]:
                continue
            xs.append(d["n_overlaps"])
            ys.append(y)
        series[key] = power_fit(xs, ys, key)

    # The rollout defect, at the SHORT horizon only. Past the divergence onset
    # the classical column is not a rollout, and a defect quoted there is a
    # number about a blow-up rather than about composition -- so a rung whose
    # column left the band before the horizon is dropped rather than fitted.
    for key, col in (("rollout_defect_reference", "reference"),
                     ("rollout_defect_reference_projected", "reference+projection"),
                     ("rollout_defect_poseidon", "poseidon")):
        xs, ys = [], []
        for d in art.get("stability", []):
            r = next((x for x in art["defect"] if x["rung"] == d["rung"]), None)
            row = d.get(col) or {}
            y = row.get("defect_at_short_horizon")
            onset = row.get("diverged_at")
            if r is None or y is None or not r["n_overlaps"]:
                continue
            # A column that diverged ANYWHERE inside the measured window is
            # dropped, not merely one that crossed the band before the horizon.
            # The growth is exponential well before the crossing: measured, the
            # classical column at N=6 leaves the band at macro-step 14 and its
            # defect at step 10 already reads 57.3 against the 9.0 the same
            # column reads with the composition layer's projection restored.
            if onset is not None:
                continue
            xs.append(r["n_overlaps"])
            ys.append(y)
        series[key] = power_fit(xs, ys, key)

    print(f"  {'series':38s} {'p':>7s} {'95% CI':>20s} {'r2':>6s}  verdict")
    for key, f in series.items():
        if f.get("exponent") is None:
            print(f"  {key:38s}      -- {'':>20s}       (n={f['n']})")
            continue
        ci = f.get("ci95")
        cis = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "--"
        verdict = ("SUPER-LINEAR" if f.get("super_linear") else
                   "sub-linear" if f.get("sub_linear") else
                   "linear or indistinguishable")
        print(f"  {key:38s} {f['exponent']:+7.3f} {cis:>20s} "
              f"{f.get('r2', float('nan')):6.3f}  {verdict}")
    return series


# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--quick", action="store_true",
                    help="rungs up to N=6 and a 20-step march")
    ap.add_argument("--rungs", default="",
                    help="comma-separated window counts, e.g. 1,2,6")
    ap.add_argument("--reuse-state", action="store_true")
    ap.add_argument("--skip-accumulated", action="store_true")
    ap.add_argument("--acc-steps", type=int, default=20,
                    help="macro-steps for the accumulated-defect rollout, which "
                         "starts from the developed state rather than the "
                         "freestream and so needs fewer than the march does")
    ap.add_argument("--skip-seam", action="store_true")
    ap.add_argument("--probe-max-seams", type=int, default=30,
                    help="above this many seams the probed compile is skipped; "
                         "the unprobed one runs at every rung so the "
                         "decertification counts are comparable")
    ap.add_argument("--long-steps", type=int, default=120,
                    help="W100: macro-steps for the long march from the "
                         "freestream. Tier 18 published 60; the classical "
                         "column with a bare assembly is not finite by 82 at "
                         "N=6, so anything under ~90 cannot see the finding "
                         "and anything at or over 110 is the claim")
    ap.add_argument("--only", default="",
                    help="comma-separated stages to run, e.g. "
                         "`--only long_march,r12`. The artifact is then SEEDED "
                         "from the existing out/w100/w100.json and only these "
                         "keys are replaced, so a partial re-run cannot delete "
                         "the stages it did not run")
    args = ap.parse_args(argv)
    #: The stage names `--only` accepts, in run order.
    STAGES = ("march", "defect", "fixed_physics", "stability", "seam", "compile",
              "fit", "r12", "long_march")
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    bad = only - set(STAGES)
    if bad:
        ap.error(f"unknown stage(s) {sorted(bad)}; the legal ones are {list(STAGES)}")

    def run(stage: str) -> bool:
        return (stage in only) if only else True
    os.makedirs(OUT, exist_ok=True)

    steps = 20 if args.quick else args.steps
    rungs = sl.ladder()
    if args.quick:
        rungs = [r for r in rungs if r.n_windows <= 6]
    if args.rungs:
        want = {int(x) for x in args.rungs.split(",")}
        rungs = [r for r in rungs if r.n_windows in want]

    assert os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE", (
        "KMP_DUPLICATE_LIB_OK must be TRUE before numpy and torch are both in "
        "this process; it is set at the top of this module and this asserts it "
        "rather than trusting it")

    path = os.path.join(OUT, "w100.json")
    # **Read the previous artifact BEFORE the first persist overwrites it.**
    # `--reuse-state` restores the march rows from here, and reading it after
    # the first write restored an empty list -- which is how a resumed run lost
    # its own power table once.
    previous = {}
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                previous = json.load(fh)
        except Exception:                                          # noqa: BLE001
            previous = {}

    # `--only` SEEDS the artifact from the previous one rather than starting
    # empty, because `persist` writes the whole dict: a partial re-run that
    # started empty would delete every stage it did not run, which is the same
    # class of accident as reading the artifact after the first write.
    art = dict(previous) if only else {}
    art.update({
        "case": "scaling_ladder", "date": time.strftime("%Y-%m-%d"),
        "steps": steps,
        "geometry": _f(sl.geometry_report()),
        "scaling": _f(wa.scaling_report()),
        "rungs": [r.label for r in rungs],
    })
    if only:
        art["partial_rerun"] = {"date": time.strftime("%Y-%m-%d"),
                                "stages": sorted(only),
                                "rungs": [r.label for r in rungs]}
    def persist():
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_f(art), fh, indent=1)
        print(f"  [artifact -> {path}]", flush=True)

    print("the ladder:")
    print(f"  {'rung':5s} {'tiling':8s} {'domain':12s} {'cells':>8s} "
          f"{'rotors':>7s} {'seams':>6s} {'overlaps':>9s}")
    for r in rungs:
        print(f"  {r.label:5s} {r.n_col}x{r.n_row:<6d} "
              f"{r.tiling.nx}x{r.tiling.ny:<8d} {r.n_cells:8d} "
              f"{r.n_rotors:7d} {r.n_seams:6d} {r.n_overlaps:9d}")
    persist()

    states = {}
    try:
        # -- 1. the marches ------------------------------------------------
        if run("march"):
            banner("1. the ladder, marched -- both experts, four "
                   "configurations each")
            art["march"] = []
            march_rungs = rungs
        else:
            # A stage that needs a developed state still needs one, so the
            # cached npz is loaded and nothing is re-marched.
            banner("1. skipped (--only): restoring cached states")
            march_rungs = []
            for r in rungs:
                cache = os.path.join(OUT, f"state_{r.label}.npz")
                if not os.path.isfile(cache):
                    print(f"  {r.label}: no {cache}; stages that need a "
                          "developed state will be skipped")
                    continue
                z = np.load(cache)
                for kind in ("reference", "poseidon"):
                    states[(r.label, kind)] = {
                        "all": (z[f"{kind}_all_u"], z[f"{kind}_all_v"]),
                        "free": (z[f"{kind}_free_u"], z[f"{kind}_free_v"]),
                        "prev": ((z[f"{kind}_prev_u"], z[f"{kind}_prev_v"])
                                 if f"{kind}_prev_u" in z else None),
                    }
                print(f"  {r.label}: restored {cache}")
        for r in march_rungs:
            cache = os.path.join(OUT, f"state_{r.label}.npz")
            done = False
            if args.reuse_state and os.path.isfile(cache):
                z = np.load(cache)
                for kind in ("reference", "poseidon"):
                    states[(r.label, kind)] = {
                        "all": (z[f"{kind}_all_u"], z[f"{kind}_all_v"]),
                        "free": (z[f"{kind}_free_u"], z[f"{kind}_free_v"]),
                        "prev": ((z[f"{kind}_prev_u"], z[f"{kind}_prev_v"])
                                 if f"{kind}_prev_u" in z else None),
                    }
                rows = [m for m in previous.get("march", [])
                        if m["rung"] == r.label]
                if not rows:
                    side = os.path.join(OUT, f"march_{r.label}.json")
                    if os.path.isfile(side):
                        with open(side, encoding="utf-8") as fh:
                            rows = json.load(fh)
                art["march"] += rows
                print(f"  {r.label}: reusing {cache}")
                done = True
            if done:
                continue
            print(f"  --- rung {r.label} ({r.n_col}x{r.n_row}, {r.n_windows} "
                  f"windows, {r.n_overlaps} overlaps) ---")
            for kind in ("reference", "poseidon"):
                print(f"   {kind}:")
                art["march"].append(_f(stage_march(r, kind, steps, states)))
                persist()
            z = {}
            for kind in ("reference", "poseidon"):
                st = states[(r.label, kind)]
                z[f"{kind}_all_u"], z[f"{kind}_all_v"] = st["all"]
                z[f"{kind}_free_u"], z[f"{kind}_free_v"] = st["free"]
                if st["prev"] is not None:
                    z[f"{kind}_prev_u"], z[f"{kind}_prev_v"] = st["prev"]
            np.savez_compressed(cache, **z)
            # the march rows beside the states, so a reuse restores the power
            # table and the wall times without re-marching
            with open(os.path.join(OUT, f"march_{r.label}.json"), "w",
                      encoding="utf-8") as fh:
                json.dump([m for m in art["march"] if m["rung"] == r.label],
                          fh, indent=1)

        # -- 2. the composed defect ----------------------------------------
        if run("defect"):
          banner("2-4. the composed defect at a common state, Pi, and W58")
          art["defect"] = []
          for r in rungs:
            st = states[(r.label, "reference")]["all"]
            art["defect"].append(_f(stage_defect(r, st, ("reference", "poseidon"),
                                                 art)))
            persist()

        if run("fixed_physics"):
          banner("2b. the same defect at ONE turbine -- interfaces without "
                 "the physics")
          art["fixed_physics"] = []
          for r in rungs:
            art["fixed_physics"].append(
                _f(fixed_physics_defect(r, solo_state(r, states, steps))))
            persist()

        # -- 5. the accumulated defect -------------------------------------
        if run("stability") and not args.skip_accumulated:
            banner("5. the rollout: is the composed system stable, and for how long")
            art["stability"] = []
            for r in rungs:
                st = states[(r.label, "reference")]["all"]
                try:
                    art["stability"].append(
                        _f(stage_stability(r, st, args.acc_steps)))
                except Exception as exc:                           # noqa: BLE001
                    art["stability"].append(
                        {"rung": r.label, "error": repr(exc),
                         "traceback": traceback.format_exc()[-1200:]})
                    print(f"  {r.label}: stability stage failed -- {exc}")
                persist()

        # -- 6-7. the seam -------------------------------------------------
        if run("seam") and not args.skip_seam:
            banner("6-7. tau/sigma/gamma on one wake seam, harness-tagged (W54), "
                   "and beta_min (W81)")
            art["seam"] = []
            for r in rungs:
                st = states[(r.label, "poseidon")]
                harness = HarnessParameters(
                    overlap_cells=wa.HALO,
                    chi_shape=f"partition-of-unity, ramp {r.tiling.ramp} cells",
                    exchange_dt=wa.MACRO_DT, exchange_cadence=1,
                    elliptic_placement="composition",
                    # W100: this column's march applies the projection to the
                    # ASSEMBLED field every macro-step (fused with the transport
                    # in one spectral pass), so the harness says so. Hand-built
                    # here rather than derived, so it is spelled the way
                    # `HarnessParameters.from_graph` spells it or two runs that
                    # did the same thing would not compare.
                    assembly_projection=("divergence-free projection, global, "
                                         "after-assembly, 1x per exchange"),
                    extra=(("tiling", f"{r.n_col}x{r.n_row}"),
                           ("n_windows", r.n_windows),
                           ("n_overlaps", r.n_overlaps),
                           ("transport", "global, once, on the assembled domain "
                                         "(W98)")))
                try:
                    art["seam"].append(_f(stage_seam(r, st["all"], st["prev"],
                                                     harness)))
                except Exception as exc:                           # noqa: BLE001
                    art["seam"].append({"rung": r.label, "error": repr(exc),
                                        "traceback": traceback.format_exc()[-1200:]})
                    print(f"  {r.label}: seam stage failed -- {exc}")
                persist()

        # -- 8. the compiles -----------------------------------------------
        if run("compile"):
          banner("8. compile_scheme at every rung")
          art["compile"] = []
          for r in rungs:
            st = states[(r.label, "reference")]["all"]
            art["compile"].append(_f(stage_compile(r, st, args.probe_max_seams,
                                                   art)))
            persist()

        # -- 10. W100: R12 at both declarations ----------------------------
        if run("r12"):
          banner("10. W100 -- L6/C2 and R12: the compile at both assemblies")
          art["r12"] = []
          for r in rungs:
            try:
                art["r12"].append(_f(stage_r12(r)))
            except Exception as exc:                           # noqa: BLE001
                art["r12"].append({"rung": r.label, "error": repr(exc),
                                   "traceback": traceback.format_exc()[-1200:]})
                print(f"  {r.label}: R12 stage failed -- {exc}")
            persist()

        # -- 11. W100: the long march, past the 60 steps Tier 18 published --
        if run("long_march"):
          banner(f"11. W100 -- the march continued to {args.long_steps} "
                 "macro-steps, with and without the projected assembly")
          art["long_march"] = []
          for r in rungs:
            try:
                art["long_march"].append(
                    _f(stage_long_march(r, args.long_steps)))
            except Exception as exc:                           # noqa: BLE001
                art["long_march"].append(
                    {"rung": r.label, "error": repr(exc),
                     "traceback": traceback.format_exc()[-1200:]})
                print(f"  {r.label}: long march failed -- {exc}")
            persist()

        # -- 9. the fit ----------------------------------------------------
        if run("fit"):
            art["fit"] = _f(stage_fit(art))
            art["gate"] = _f(verdict_block(art))
    finally:
        persist()
    return 0


def verdict_block(art):
    """The gate, stated against the criterion rather than described.

    **F1 is a falsification criterion, so the test is one-sided.**  It fails if
    composition error is SUPER-linear in interface count, which means the
    interval has to sit strictly above 1 before anything is falsified.  Three
    readings, and the pickup names all three:

        upper bound < 1      sub-linear      -- Claim B confirmed on this problem
        interval spans 1     linear          -- survivable, and stated as such
        lower bound > 1      SUPER-linear    -- the pathmap's stated reason to
                                                stop and rethink

    Gating on "upper bound <= 1" instead would report the middle row as a
    failure, which is the criterion tightened after the fact.
    """
    banner("THE GATE -- F1: composition error no faster than linear in interfaces")
    fit = art.get("fit", {})
    headline = [
        ("blend_spread_l2_reference", "classical, reference-free"),
        ("blend_spread_l2_poseidon", "checkpoint, reference-free"),
        ("composed_defect_l2_reference", "classical, against the monolith"),
    ]
    out = {"criterion": ("F1 fails if composition error grows SUPER-linearly in "
                         "interface count. Sub-linear confirms Claim B, linear is "
                         "survivable, super-linear is the stated reason to stop "
                         "(f1-pathmap-and-end-goal 5)"),
           "columns": {}, "reading": {}}
    any_super = False
    seen = False
    for key, label in headline:
        f = fit.get(key)
        if not f or f.get("exponent") is None:
            continue
        seen = True
        reading = ("SUPER-LINEAR" if f.get("super_linear") else
                   "sub-linear" if f.get("sub_linear") else
                   "linear, or indistinguishable from it")
        any_super = any_super or bool(f.get("super_linear"))
        out["columns"][key] = {"label": label, "exponent": f["exponent"],
                               "ci95": f.get("ci95"), "stderr": f.get("stderr"),
                               "r2": f.get("r2"),
                               "local_slopes": f.get("local_slopes"),
                               "reading": reading}
        ci = f.get("ci95")
        cis = (f"95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else
               "no interval: fewer than three rungs carry an interface count")
        print(f"  {label:34s} p = {f['exponent']:+.3f}  {cis}   {reading}")
        if f.get("local_slopes"):
            print("    local slopes " + ", ".join(
                f"{int(sl['from'])}->{int(sl['to'])}: {sl['slope']:+.3f}"
                for sl in f["local_slopes"]))
    out["passes"] = None if not seen else (not any_super)
    out["any_super_linear"] = any_super
    print()
    if out["passes"] is None:
        print("  no fit: the gate is not evaluated")
    elif out["passes"]:
        print("  F1 IS NOT FALSIFIED. No column's 95% interval lies above 1, so no")
        print("  super-linear growth is visible over the full 24x range on this")
        print("  problem with these two experts.")
        print("  Read it as F1 asks and no further. A criterion that was not")
        print("  falsified is not a theorem: this is five rungs, one geometry, one")
        print("  Reynolds number, and the largest graph is 24 agents against rung")
        print("  9's 15-20. It licenses building Phase C; it does not establish")
        print("  Claim B.")
    else:
        print("  F1 IS FALSIFIED in at least one column: the 95% interval lies")
        print("  strictly above 1.")
        print("  f1-pathmap-and-end-goal 5 calls this a reason to stop and rethink")
        print("  rather than press on, and case-study-ladder-to-f1 4 does not build")
        print("  CS-16 on it.")
    return out


if __name__ == "__main__":
    sys.exit(main())
