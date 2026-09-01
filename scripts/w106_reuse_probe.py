"""Tier 20 -- is a substitution certificate a property of the expert, or of the
state it was probed at?

Driver for `atlas/cases/reuse_probe.py`, the EIGHTH real case study.  Everything
`tier0-measurements` section 20 quotes comes from here; artifact
`out/w106/w106.json`, cached states `out/w106/state_<rung>.npz`.

    python scripts/w106_reuse_probe.py                    # everything, ~40 min
    python scripts/w106_reuse_probe.py --quick            # N6 only, 3 states
    python scripts/w106_reuse_probe.py --reuse-state      # skip the marches
    python scripts/w106_reuse_probe.py --only grid,floor  # one stage

The question, and what it decides
---------------------------------

`case-study-ladder-to-f1` section 4 names CS-8 as the only rung that tests the
FOUNDATION-MODEL claim rather than the coupling claim, and sharpens it past
*"does the expert work in a new scenario"* to:

    Is a certificate a property of the expert, or of the state it was probed at?

A twenty-agent car carries roughly twenty conformance certificates and forty
seam certificates.  **If each is state-specific the plug-in claim is per-design
rather than per-expert**, and the economic argument for a reusable expert
library -- search wide and cheap -- collapses back to re-certifying every
design.  That is the claim being measured, and it is measurable with the
instruments Tier 19 finished building.

What it measures, in order
---------------------------

  1. one march per rung, the R10-compliant classical column -- `wake_array
     .exposed_reference_solver` with `assembly.ProjectedAssembly` -- snapshotted
     at macro-steps 0, 10, 30, 60, 110, each snapshot health-checked for
     finiteness, band and divergence (Tier 19's N=12 tail is still settling at
     120, so the number is reported rather than trusted)
  2. the GRID: every comparable seam at every probe state, both experts, probed
     about a COMMON seam base, reduced to the five quantities a substitution
     certificate discriminates on
  3. control ZERO: at the freestream every ``_full`` seam holds the same field
     on the same port, so the four regime seams must return the SAME operator.
     If they do not, the regime labels are reading the port and not the flow
  4. control FLOOR: the identical (seam, state) pair re-probed with freshly
     constructed experts from the state reloaded off disk -- the reproducibility
     floor every spread below is quoted against
  5. tau and sigma on a reduced cross, so the DERIVED beta_min of W81 can be
     asked the same question as the thresholds
  6. the compiles, with W105 asserted on every graph: `ProjectedAssembly`
     present, elliptic part EXPOSED, R10 / R10b / R12 recorded
  7. the analysis: spread across seams at fixed state, spread across states at
     fixed seam, spread across geometries at fixed seam and state, each against
     the floor, each with a verdict of `travels` / `undecided` /
     `state-dependent`

Three things this driver is careful about, each because a prior tier paid for it
--------------------------------------------------------------------------------

**No probe is taken on a diverging trajectory.**  Tier 18 published a state
nobody had marched far enough and section 19.6 published a repair nobody had
marched far enough, on the same day.  Every probe state here comes off the one
column measured stable to 120 macro-steps, and `reuse_probe.probe_state_health`
refuses the probe rather than the write-up if it is not.

**The projection is asserted, not assumed.**  W105: a graph can clear R10 and
R10b and never apply the elliptic part at all, and the column that does is
stable and not incompressible.  `reuse_probe.assert_projected` runs on every
graph this driver builds, including the ones it only compiles.

**A spread is not a result until the floor is measured.**  Every movement below
is quoted as a multiple of the movement the same quantity shows when the
identical pair is probed twice.
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

from atlas.cases import reuse_probe as rp                            # noqa: E402
from atlas.cases import scaling_ladder as sl                         # noqa: E402
from atlas.cases import wake_array as wa                             # noqa: E402
from atlas.composition import (                                      # noqa: E402
    beta_min_from_tolerance,
    certify_substitution,
)
from atlas.compiler import compile_scheme                            # noqa: E402
from atlas.multiphysics import (                                     # noqa: E402
    MultiphysicsError,
    lag_distance,
    seam_defect_split,
    tight_couple,
)
from atlas.ports import ResponseHalf                                 # noqa: E402
from atlas.probe import ProbeBudget, assemble_seam                   # noqa: E402
from atlas.scheme import Budget                                      # noqa: E402
from atlas.transfer import InterfaceSpace, SeamTransfer              # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "out", "w106")

#: The freestream band in cells -- `w93`'s number and `w100`'s, unchanged, and
#: held fixed here for the same reason: without the inlet and both laterals at
#: (U_INF, 0) the disks drain the box.
BAND = 8

#: The seams probed at each rung, by ID.  A rule picks them (`reuse_probe
#: .seam_menu` classifies every fluid-fluid x-seam from the layout alone) and
#: this narrows to the ones that make a COMPARABLE set: within the ``full``
#: group all four regimes at N12, within the ``bypass`` group the clean/wake
#: pair plus its replicate.  Crossing the groups would be comparing a 33-mode
#: operator with a 25-mode one.
SEAMS = {
    "N12": ("x0r1_full", "x2r1_full", "x2r0_full", "x2r2_full",
            "x0r0_bypass", "x1r0_bypass", "x1r2_bypass"),
    # the cross-geometry set: every one of these IDs also exists at N12, which
    # is what makes the graph size the only thing that differs
    "N6": ("x0r1_full", "x0r0_bypass", "x1r0_bypass", "x1r1_bypass"),
}

#: The cells re-probed for the reproducibility floor: one per port group per
#: rung, at the developed state, plus the freestream where the answer is known.
FLOOR_CELLS = {
    "N12": (("x2r0_full", 110), ("x1r0_bypass", 110), ("x0r1_full", 0)),
    "N6": (("x0r1_full", 110),),
}

#: Where tau and sigma are measured.  A cross rather than the grid: every seam
#: at the developed state (the SEAM factor) and the two headline seams at every
#: state (the STATE factor).  The split runs a tight_couple and a
#: finite-difference Jacobian per cell, so the grid would double the run.
TAU_SEAMS_AT_DEVELOPED = {"N12": SEAMS["N12"], "N6": ()}
TAU_STATE_SEAMS = {"N12": ("x2r0_full", "x1r0_bypass"), "N6": ()}


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


def _fmt(x, spec=".4g"):
    if x is None:
        return "--"
    if isinstance(x, bool):
        return str(x)
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return str(x)


def banner(s):
    print()
    print("=" * 78)
    print(s)
    print("=" * 78, flush=True)


def persist(art, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_f(art), fh, indent=1, sort_keys=False)
    os.replace(tmp, path)
    print(f"  [artifact -> {path}]", flush=True)


# ---------------------------------------------------------------------------
# 1. the march -- the ONE trajectory every probe state comes off
# ---------------------------------------------------------------------------


def _forcing(r, u, active):
    """The disks' body force, lifted from `w100_scaling_ladder._forcing`.

    A transcription rather than a re-derivation, for the reason CS-7 gives about
    its own transcription of `w93`: the probe states have to be the SAME states
    that ladder marched, or the cross-tier comparison in section 20 compares two
    experiments.  Including the derived smearing thickness ``Delta_d = <U_d> dt``.
    """
    import importlib
    dk = importlib.import_module("atlas_windfarm_reference.disk")
    x_c = (np.arange(r.tiling.nx) + 0.5) * wa.DX
    y_c = (np.arange(r.tiling.ny) + 0.5) * wa.DX
    fx = np.zeros(r.shape)
    rec = {}
    for rot in r.tiling.rotors:
        rows = np.abs(y_c - rot.y_centre) <= 0.5 * wa.ROTOR_D + 1e-9
        i_up = int(np.argmin(np.abs(x_c - (rot.x_plane - 0.25))))
        ud = float(np.mean(u[rows, i_up]))
        rec[rot.rotor_id] = ud
        if rot.rotor_id in active:
            thick = max(ud, 0.05) * wa.MACRO_DT
            d = dk.ActuatorDisk(thickness=thick)
            st = d(max(ud, 1e-3))
            fx = fx + d.body_force_field(x_c, y_c, wa.DX, wa.DX, st.thrust,
                                         x0=rot.x_plane - 0.5 * thick,
                                         y0=rot.y_centre - 0.5 * wa.ROTOR_D)
    return fx, rec


def _band(r, u, v):
    u[:, :BAND] = wa.U_INF
    v[:, :BAND] = 0.0
    u[:BAND, :] = wa.U_INF
    v[:BAND, :] = 0.0
    u[-BAND:, :] = wa.U_INF
    v[-BAND:, :] = 0.0
    return u, v


def exposed_step(r, u, v, fx, ex, assembly):
    """ONE composed macro-step of the R10-compliant classical column.

    The elliptic part is OUT of the agent (`wake_array.exposed_reference_solver`
    replaces `_project` with the identity) and the composition layer applies it
    once, to the ASSEMBLED field, through the declared
    `assembly.ProjectedAssembly`.  That is R10 + R10b + R12 together and it is
    the only classical arrangement in this vault that reaches macro-step 120.

    ``wake_array.assemble_conservative`` is the one call that both blends and
    projects, so the record and the driver cannot disagree about which happened
    -- which is what W100 was.
    """
    t = r.tiling
    us, vs = t.cut(u), t.cut(v)
    fs = t.cut(fx)
    u1, v1 = ex.step_batch(us, vs, wa.MACRO_DT, bc0=None,
                           force=(fs, np.zeros_like(fs)))
    return wa.assemble_conservative(t, u1, v1, assembly)


def stage_march(r, steps, snap_at, band=3.0):
    """March the rung once and keep the probe states, health-checked.

    Returns ``(snapshots, record)``.  ``snapshots`` maps macro-step to
    ``(u, v)``; step 0 is the freestream itself, which is control ZERO and is
    kept without a step being taken.
    """
    ex = wa.exposed_reference_solver(wa.NU_REF)
    assembly = wa.projected_assembly(r.tiling)
    rotors = {x.rotor_id for x in r.tiling.rotors}
    u, v = np.ones(r.shape), np.zeros(r.shape)
    snaps: dict[int, tuple] = {}
    rec = {"rung": r.label, "n_windows": r.n_windows,
           "n_overlaps": r.n_overlaps, "steps": steps,
           "snap_at": list(snap_at), "column": "exposed + ProjectedAssembly",
           "health": {}, "div_rms": [], "u_max": []}
    t0 = time.time()
    div0 = None
    for k in range(steps + 1):
        if k in snap_at:
            h = rp.probe_state_health(u, v, k, r.tiling, band=band,
                                      div_ref=div0)
            if div0 is None:
                div0 = h["div_rms"]
            rec["health"][k] = h
            snaps[k] = (u.copy(), v.copy())
            print(f"  {r.label:4s} snapshot @{k:4d}  u_max={_fmt(h['u_max'])} "
                  f"div={_fmt(h['div_rms'])} settled={h['settled_horizon']} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if k == steps:
            break
        fx, _ = _forcing(r, u, rotors)
        u, v = exposed_step(r, u, v, fx, ex, assembly)
        u, v = _band(r, u, v)
        rec["u_max"].append(float(np.max(np.abs(u))))
        rec["div_rms"].append(float(wa.divergence_rms(u, v)))
        if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
            raise rp.ProbeStateRefused(
                f"{r.label} exposed+projected march is not finite at macro-step "
                f"{k + 1}; Tier 19 measured this column stable to 120")
    rec["wall_s"] = time.time() - t0
    fin = rec["div_rms"]
    q = len(fin) // 4
    if q >= 2:
        third, fourth = float(np.mean(fin[2 * q:3 * q])), float(np.mean(fin[3 * q:]))
        rec["div_tail_trend"] = (fourth / third) if third > 0 else None
    else:
        rec["div_tail_trend"] = None
    return snaps, rec


def save_states(r, snaps):
    path = os.path.join(OUT, f"state_{r.label}.npz")
    os.makedirs(OUT, exist_ok=True)
    payload = {}
    for k, (u, v) in snaps.items():
        payload[f"u{k}"] = u
        payload[f"v{k}"] = v
    np.savez_compressed(path, **payload)
    print(f"  [states -> {path}]", flush=True)
    return path


def load_states(r, snap_at):
    path = os.path.join(OUT, f"state_{r.label}.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    out = {}
    for k in snap_at:
        if f"u{k}" in z:
            out[k] = (np.array(z[f"u{k}"]), np.array(z[f"v{k}"]))
    return out or None


# ---------------------------------------------------------------------------
# 2. one cell of the grid
# ---------------------------------------------------------------------------


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
    """The COMMON interface state both experts are linearized about.

    W74's class: `probe_base` is one per expert, the interface variable is one
    variable, and a sum of Jacobians is a Jacobian only if every term was taken
    at the same point.  Taking the mean of the two declared bases is what CS-7
    did and it is what makes a difference between the two blocks the EXPERT
    rather than the base.
    """
    conn = graph.connection(seam_id)
    tr = transfer_for(graph, seam_id)
    coeffs = []
    for agent_id, name in (conn.a, conn.b):
        base_V = np.asarray(experts[agent_id].probe_base(name), dtype=float)
        coeffs.append(tr.prolongations[agent_id].adjoint(tr.space) @ base_V)
    return 0.5 * (coeffs[0] + coeffs[1]), coeffs


def probe_cell(r, choice, step, state, health, tau=False, prev_state=None):
    """One (seam, state) cell: two operators, one certificate, one reading.

    Both graphs are built fresh from the same state, both are asserted to carry
    a `ProjectedAssembly` (**W105**), and both operators are assembled about the
    same ``seam_base``.  The only thing that differs between them is the expert.
    """
    u, v = state
    ex_r = sl.make_experts(u, v, r, "reference_exposed")
    ex_p = sl.make_experts(u, v, r, "poseidon")
    g_r, _ = rp.build(u, v, r, kind="reference_exposed", experts=ex_r)
    g_p, _ = rp.build(u, v, r, kind="poseidon", experts=ex_p)
    conn = g_p.connection(choice.seam_id)
    tr = transfer_for(g_p, choice.seam_id)
    base_M, _sides = seam_base_M(g_p, ex_p, choice.seam_id)
    tag = f"CS-8 {r.label} {choice.seam_id} @macro-step {step} ({choice.regime})"
    op_r = assemble_seam(g_r, g_r.connection(choice.seam_id), tr,
                         budget=ProbeBudget(),
                         expected_null_dim=conn.expected_null_dim,
                         probe_state=tag, seam_base=base_M)
    op_p = assemble_seam(g_p, conn, tr, budget=ProbeBudget(),
                         expected_null_dim=conn.expected_null_dim,
                         probe_state=tag, seam_base=base_M)
    target = next(a for a, _n in (conn.a, conn.b) if a.startswith("F"))
    blk_r, blk_p = op_r.blocks[target], op_p.blocks[target]
    cert = certify_substitution(
        target, g_r.agent(target).capabilities, g_p.agent(target).capabilities,
        blk_r.S, blk_p.S, beta=op_r.beta,
        passivity_old=blk_r.passivity_lambda_min,
        passivity_new=blk_p.passivity_lambda_min,
        block_norm=float(np.linalg.norm(blk_r.S, 2)))
    reading = rp.read_certificate(r.label, choice, step, target, op_r, op_p,
                                  cert, state=dict(health or {}))
    reading.notes.append(cert.beta_min_source)
    if tau:
        _add_tau(reading, r, choice, g_r, g_p, ex_r, ex_p, conn, prev_state)
    return reading, op_r, op_p


def _add_tau(reading, r, choice, g_r, g_p, ex_r, ex_p, conn, prev_state):
    """tau, sigma and the DERIVED beta_min at one cell.

    The same construction CS-7's `stage_seam` runs, so section 19.8's numbers and
    section 20's are the same measurement at different probe states.  gamma is
    absent for the reason CS-7 records: a halo scheme poses no interface solve,
    so there is no solve infidelity localized at a seam.
    """
    ids = [conn.a[0], conn.b[0]]
    names = {conn.a[0]: conn.a[1], conn.b[0]: conn.b[1]}
    halves = {a: ResponseHalf.EFFORT for a in ids}
    trace0 = np.mean([np.asarray(ex_p[a].probe_base(names[a])) for a in ids],
                     axis=0)
    lagged = trace0
    if prev_state is not None:
        prev_ex = sl.make_experts(*prev_state, r, "poseidon")
        lagged = np.mean([np.asarray(prev_ex[a].probe_base(names[a]))
                          for a in ids], axis=0)
    reading.notes.append(f"lag_distance = {lag_distance(lagged, trace0):.6g}")
    rc = {a: (lambda t, a=a: ex_r[a].respond(names[a], t)) for a in ids}
    pc = {a: (lambda t, a=a: ex_p[a].respond(names[a], t)) for a in ids}
    tc = tight_couple(lambda lam: sum(np.asarray(rc[a](lam)).ravel() for a in ids),
                      trace0)
    reading.notes.append(
        f"referent converged={bool(tc.converged)} in {int(tc.iterations)} "
        f"iterations, residual {float(tc.residual_norm):.4g}")
    try:
        sd = seam_defect_split(choice.seam_id, dict(pc), rc, halves, trace0,
                               lagged, measure=wa.DX, jacobian="fd")
    except MultiphysicsError as exc:
        # **Not a bug, and not a reason to lose the rung.**  `seam_defect_split`
        # refuses a relative defect when no power crosses the REFERENCE
        # interface -- L4's empty-seam case -- and at a deep-wake seam the
        # reference pair can land there while the certificate quantities beside
        # it are perfectly well defined.  So tau is recorded as UNDEFINED with
        # the reason attached, which is a measurement about the seam, and the
        # thresholds this tier is actually about are unaffected.
        reading.notes.append(f"tau UNDEFINED: {type(exc).__name__}: {exc}")
        reading.tau_undefined = f"{type(exc).__name__}: {exc}"
        return
    if sd.converged:
        reading.tau_total = float(sum(float(x) for x in sd.tau.values()))
        reading.sigma = float(sd.sigma)
        eps, src = beta_min_from_tolerance(reading.tau_total, reading.sigma)
        reading.eps_tol = eps
        reading.notes.append(f"eps_tol source: {src}")
    else:
        reading.notes.append("seam_defect_split did not converge; tau absent")


def stage_grid(r, snaps, march_rec, tau=True, on_row=None, have=None):
    """Every comparable seam at every probe state, both experts.

    ``on_row`` is called after EACH cell rather than after the rung.  A grid cell
    is half a minute and a rung is twenty of them, and a run that loses a rung's
    work because it was interrupted in the twentieth is a run that has to start
    again -- which is `run-artifacts-and-warnings`' standing rule and was learnt
    here the first time this driver was interrupted.
    """
    menu = {s.seam_id: s for s in rp.seam_menu(r.tiling)}
    wanted = [menu[i] for i in SEAMS[r.label] if i in menu]
    rows = []
    have = set(have or ())
    steps = sorted(snaps)
    for choice in wanted:
        for k in steps:
            if (r.label, choice.seam_id, k) in have:
                print(f"  {r.label:4s} {choice.seam_id:14s} @{k:<4d} already in "
                      "the artifact, skipped", flush=True)
                continue
            want_tau = tau and (
                (k == steps[-1] and choice.seam_id in TAU_SEAMS_AT_DEVELOPED.get(r.label, ()))
                or choice.seam_id in TAU_STATE_SEAMS.get(r.label, ()))
            prev = None
            if want_tau:
                earlier = [s for s in steps if s < k]
                prev = snaps[earlier[-1]] if earlier else None
            t0 = time.time()
            reading, _op_r, _op_p = probe_cell(
                r, choice, k, snaps[k], march_rec["health"].get(k),
                tau=want_tau, prev_state=prev)
            reading.notes.append(f"wall {time.time() - t0:.1f}s")
            rows.append(reading.as_dict())
            if on_row is not None:
                on_row(rows[-1])
            print(f"  {r.label:4s} {choice.seam_id:14s} {choice.regime:16s} "
                  f"@{k:4d}  beta={_fmt(reading.beta)} "
                  f"|Delta|={_fmt(reading.delta_norm)} "
                  f"Xi={_fmt(reading.Xi_block)} "
                  f"vis>{_fmt(reading.visible_above)} "
                  f"fail>{_fmt(reading.fails_above)} "
                  f"tau={_fmt(reading.tau_total)} ({time.time() - t0:.0f}s)",
                  flush=True)
    return rows


# ---------------------------------------------------------------------------
# 3. control FLOOR -- the same cell, twice
# ---------------------------------------------------------------------------


def stage_floor(r, snaps, march_rec):
    """The reproducibility floor of the WHOLE pipeline, not of the solver.

    The state is reloaded off disk, the experts are constructed fresh, the graphs
    are rebuilt and the operators are re-probed.  That is the floor a
    re-certification would actually see -- `ExpertCapabilities
    .reproducibility_floor` is a declaration about the solver and this is a
    measurement of the pipeline, and only the second one is what a spread has to
    beat.
    """
    menu = {s.seam_id: s for s in rp.seam_menu(r.tiling)}
    disk = load_states(r, list(snaps))
    rows = []
    for seam_id, k in FLOOR_CELLS.get(r.label, ()):
        if seam_id not in menu or k not in snaps:
            continue
        state_disk = (disk or snaps)[k]
        a, _, _ = probe_cell(r, menu[seam_id], k, snaps[k],
                             march_rec["health"].get(k))
        b, _, _ = probe_cell(r, menu[seam_id], k, state_disk,
                             march_rec["health"].get(k))
        delta = {f: (None if (getattr(a, f) is None or getattr(b, f) is None)
                     else abs(float(getattr(a, f)) - float(getattr(b, f))))
                 for f in rp.CertificateReading.DISCRIMINATING}
        rows.append({
            "rung": r.label, "seam_id": seam_id, "step": k,
            "regime": menu[seam_id].regime, "segment": menu[seam_id].segment,
            "first": {f: getattr(a, f) for f in rp.CertificateReading.DISCRIMINATING},
            "second": {f: getattr(b, f) for f in rp.CertificateReading.DISCRIMINATING},
            "abs_delta": delta,
            "state_from_disk": disk is not None,
            "verdict_stable": a.verdict == b.verdict,
        })
        print(f"  {r.label:4s} FLOOR {seam_id:14s} @{k:<4d} "
              + "  ".join(f"{f}={_fmt(delta[f], '.3e')}"
                          for f in ("beta", "delta_norm", "Xi_block",
                                    "fails_above")), flush=True)
    return rows


def floor_of(floor_rows, field, segment=None, rung=None):
    """The floor for one field: the worst re-probe disagreement seen."""
    vals = [row["abs_delta"].get(field) for row in floor_rows
            if row["abs_delta"].get(field) is not None
            and (segment is None or row["segment"] == segment)
            and (rung is None or row["rung"] == rung)]
    return max(vals) if vals else None


# ---------------------------------------------------------------------------
# 4. control ZERO -- the freestream, where the answer is known
# ---------------------------------------------------------------------------


def stage_zero(rows):
    """At the freestream every seam of a group must return the SAME operator.

    Every window holds the same uniform field, every ``full`` port is the same
    128-cell face with the same 33 modes, and the solver is deterministic -- so
    the four regime seams are the same probe of the same expert at the same
    state, written four ways.  A nonzero spread here would mean the regime
    labels are picking up the port geometry and the whole SEAM factor would be
    uninterpretable.

    Reported as an ABSOLUTE range per field per group, and compared with zero
    rather than with a tolerance, because the claim is bit-for-bit.
    """
    out = {}
    for rung in sorted({row["rung"] for row in rows}):
        at0 = [row for row in rows if row["rung"] == rung and row["step"] == 0]
        for seg in sorted({row["segment"] for row in at0}):
            grp = [row for row in at0 if row["segment"] == seg]
            if len(grp) < 2:
                continue
            fields = {}
            for f in rp.CertificateReading.DISCRIMINATING:
                fields[f] = rp.spread([row["discriminating"][f] for row in grp])
            worst = max((v["abs_range"] for v in fields.values()
                         if v["abs_range"] is not None), default=None)
            out[f"{rung}:{seg}"] = {
                "rung": rung, "segment": seg,
                "seams": [row["seam_id"] for row in grp],
                "regimes": sorted({row["regime"] for row in grp}),
                "fields": fields,
                "worst_abs_range": worst,
                "identical": (worst is not None and worst == 0.0),
            }
            print(f"  ZERO {rung:4s} {seg:7s} {len(grp)} seams "
                  f"({', '.join(sorted({row['regime'] for row in grp}))}) "
                  f"worst range {_fmt(worst, '.3e')} "
                  f"identical={out[f'{rung}:{seg}']['identical']}", flush=True)
    return out


# ---------------------------------------------------------------------------
# 5. the compiles, and the W105 assertion
# ---------------------------------------------------------------------------


def stage_compile(r, state):
    """The compile at one rung, both columns, with the projection asserted."""
    u, v = state
    out = {"rung": r.label, "columns": {}}
    for kind in ("reference_exposed", "poseidon"):
        try:
            graph, _ = rp.build(u, v, r, kind=kind)
            proj = rp.assert_projected(graph)
            res = compile_scheme(
                graph, budget=Budget(allow_probe=False),
                probe_state=f"CS-8 {r.label}, developed array")
            caps = graph.agent("F00").capabilities
            groups: dict[str, int] = {}
            for d in res.decisions.decertifications:
                key = f"{d.layer}/{d.rule}"
                groups[key] = groups.get(key, 0) + 1
            row = {
                "verdict": res.verdict.value,
                "n_refusals": len(res.decisions.refusals),
                "n_decertifications": len(res.decisions.decertifications),
                "refusals_by_rule": sorted({f"{d.layer}/{d.rule}"
                                            for d in res.decisions.refusals}),
                "decertifications_by_rule": groups,
                "R10_R10b_R12": sorted({f"{d.layer}/{d.rule}"
                                        for d in res.decisions
                                        if d.rule in ("R10", "R10b", "R12", "C2")}),
                "elliptic_subsolve": str(getattr(caps.elliptic_subsolve, "value",
                                                 caps.elliptic_subsolve)),
                "assembly": type(graph.partition_of_unity).__name__,
                "assembly_projection": type(proj.projection).__name__,
                "projection_scope": str(getattr(proj.projection, "scope", None)),
                "projection_stage": str(getattr(proj.projection, "stage", None)),
                "projection_cadence": getattr(proj.projection, "cadence", None),
                "W105_asserted": True,
            }
        except Exception as exc:                                # noqa: BLE001
            row = {"error": f"{type(exc).__name__}: {exc}", "W105_asserted": False}
        out["columns"][kind] = row
        print(f"  {r.label:4s} {kind:18s} "
              f"{row.get('verdict', row.get('error'))}"
              f"  refusals={row.get('n_refusals')} "
              f"decerts={row.get('n_decertifications')} "
              f"elliptic={row.get('elliptic_subsolve')} "
              f"assembly={row.get('assembly')}", flush=True)
    return out


# ---------------------------------------------------------------------------
# 6. the analysis -- how much moved, against what floor
# ---------------------------------------------------------------------------


def stage_analysis(rows, floor_rows):
    """Spread across seams, across states, across geometries -- each vs the floor.

    Three factors, one statistic, and the statistic is always the same: the
    absolute range of the quantity over the factor, divided by the absolute
    range the same quantity shows when the identical pair is probed twice.
    """
    out = {"per_factor": {}, "headline": {}, "verdict_stability": {}}
    fields = list(rp.CertificateReading.DISCRIMINATING)

    def cells(**kw):
        return [row for row in rows
                if all(row.get(k) == v for k, v in kw.items())]

    # -- factor SEAM: fixed rung, fixed state, fixed port group ---------------
    seam_factor = {}
    for rung in sorted({row["rung"] for row in rows}):
        for seg in sorted({row["segment"] for row in rows if row["rung"] == rung}):
            for step in sorted({row["step"] for row in rows
                                if row["rung"] == rung and row["segment"] == seg}):
                grp = cells(rung=rung, segment=seg, step=step)
                if len(grp) < 2:
                    continue
                key = f"{rung}:{seg}:@{step}"
                seam_factor[key] = {
                    "rung": rung, "segment": seg, "step": step,
                    "n_seams": len(grp),
                    "seams": [g["seam_id"] for g in grp],
                    "fields": {f: rp.spread([g["discriminating"][f] for g in grp])
                               for f in fields},
                }
    out["per_factor"]["seam"] = seam_factor

    # -- factor STATE: fixed rung, fixed seam --------------------------------
    state_factor = {}
    for rung in sorted({row["rung"] for row in rows}):
        for seam in sorted({row["seam_id"] for row in rows if row["rung"] == rung}):
            grp = sorted(cells(rung=rung, seam_id=seam), key=lambda g: g["step"])
            if len(grp) < 2:
                continue
            state_factor[f"{rung}:{seam}"] = {
                "rung": rung, "seam_id": seam, "regime": grp[0]["regime"],
                "segment": grp[0]["segment"],
                "steps": [g["step"] for g in grp],
                "fields": {f: rp.spread([g["discriminating"][f] for g in grp])
                           for f in fields},
                "developed_only": {
                    f: rp.spread([g["discriminating"][f] for g in grp
                                  if g["step"] > 0]) for f in fields},
            }
    out["per_factor"]["state"] = state_factor

    # -- factor GEOMETRY: fixed seam, fixed state, across rungs --------------
    geom_factor = {}
    for seam in sorted({row["seam_id"] for row in rows}):
        for step in sorted({row["step"] for row in rows}):
            grp = cells(seam_id=seam, step=step)
            rungs = sorted({g["rung"] for g in grp})
            if len(rungs) < 2:
                continue
            geom_factor[f"{seam}:@{step}"] = {
                "seam_id": seam, "step": step, "rungs": rungs,
                "regime": grp[0]["regime"], "segment": grp[0]["segment"],
                "fields": {f: rp.spread([g["discriminating"][f] for g in grp])
                           for f in fields},
            }
    out["per_factor"]["geometry"] = geom_factor

    # -- factor REPLICATE: two seams the RULE calls the SAME regime ----------
    #
    # `x2r0_full` and `x2r2_full` are both `deep-wake`: same station, same port,
    # same upstream rotor count, different row.  Whatever separates them is what
    # the regime LABEL does not pin down, so this is the yardstick the floor
    # cannot be -- the floor says the pipeline is exact, and this says how much
    # of the seam factor survives after the taxonomy has done its work.
    rep_factor = {}
    for rung in sorted({row["rung"] for row in rows}):
        for regime in sorted({row["regime"] for row in rows
                              if row["rung"] == rung}):
            for step in sorted({row["step"] for row in rows}):
                grp = cells(rung=rung, regime=regime, step=step)
                if len({g["seam_id"] for g in grp}) < 2:
                    continue
                rep_factor[f"{rung}:{regime}:@{step}"] = {
                    "rung": rung, "regime": regime, "step": step,
                    "seams": sorted(g["seam_id"] for g in grp),
                    "fields": {f: rp.spread([g["discriminating"][f] for g in grp])
                               for f in fields},
                }
    out["per_factor"]["replicate"] = rep_factor

    # -- the floors ----------------------------------------------------------
    floors = {f: floor_of(floor_rows, f) for f in fields}
    floors_by_segment = {
        seg: {f: floor_of(floor_rows, f, segment=seg) for f in fields}
        for seg in sorted({row["segment"] for row in floor_rows})}
    out["floor"] = {"worst_over_all": floors, "by_segment": floors_by_segment,
                    "n_cells": len(floor_rows)}

    # -- the headline: median spread per factor per field, against the floor --
    def median_spread(bucket, field, **filt):
        vals = [v["fields"][field]["abs_range"] for v in bucket.values()
                if v["fields"][field]["abs_range"] is not None
                and all(v.get(k) == x for k, x in filt.items())]
        return float(np.median(vals)) if vals else None

    def median_relative(bucket, field, **filt):
        """Range over the level the quantity sits at -- the readable form.

        The floor is exactly zero on this pipeline, so *how many floors* is
        infinite for every quantity that moved at all and separates nothing.
        What separates them is how big the movement is compared with the number
        itself: beta moving 5% and ``visible_above`` moving 1800% are the same
        `state-dependent` verdict and not the same finding.
        """
        vals = []
        for v in bucket.values():
            if any(v.get(k) != x for k, x in filt.items()):
                continue
            sp = v["fields"][field]
            if sp["abs_range"] is None or not sp.get("mean_abs"):
                continue
            vals.append(sp["abs_range"] / sp["mean_abs"])
        return float(np.median(vals)) if vals else None

    for factor, bucket in (("seam", seam_factor), ("state", state_factor),
                           ("geometry", geom_factor),
                           ("replicate", rep_factor)):
        head = {}
        for f in fields:
            # **The freestream is control ZERO, not a level of these factors.**
            # At step 0 every seam of a group, every same-regime replicate and
            # every rung hold the identical field on the identical port, so
            # those cells are identically zero BY CONSTRUCTION and including
            # them in a median halves it -- which is a presentation that makes
            # the taxonomy look better than the measurement says it is.  The
            # STATE factor is the exception and keeps step 0, because the
            # freestream is a genuine level of that factor and is most of what
            # the factor measures.
            bucket_dev = ({k: v for k, v in bucket.items() if v.get("step") != 0}
                          if factor in ("seam", "replicate", "geometry")
                          else bucket)
            med = median_spread(bucket_dev, f)
            head[f] = {
                "median_abs_range": med,
                "median_range_over_level": median_relative(bucket_dev, f),
                **rp.travel_verdict({"abs_range": med}, floors.get(f)),
            }
        out["headline"][factor] = head

    # -- does the VERDICT move, which is the decision-relevant question -------
    #
    # Two forms, and only the second one is evidence.  The certificate as this
    # driver calls it is given NO tolerance, so by W81's own design it returns
    # `admit-uncertified` and the two thresholds instead of a verdict -- a
    # constant, and a constant is not a stability result.  The second form
    # sweeps the whole admissible axis and derives the verdict at each cell from
    # the stored thresholds (`reuse_probe.verdict_at`), which is the question
    # asked properly.
    verdicts = {}
    for rung in sorted({row["rung"] for row in rows}):
        grp = cells(rung=rung)
        seen = sorted({g["verdict"] for g in grp})
        verdicts[rung] = {
            "as_called": {
                "distinct": seen, "n_cells": len(grp),
                "stable": len(seen) == 1,
                "note": ("no tolerance supplied, so W81 returns the thresholds "
                         "instead of a verdict: this is a constant by "
                         "construction and is not evidence of stability"),
            },
            "swept": {},
        }
        for bm in rp.BETA_MIN_AXIS:
            got = {rp.verdict_at(g["discriminating"]["visible_above"],
                                 g["discriminating"]["fails_above"], bm)
                   for g in grp}
            verdicts[rung]["swept"][repr(bm)] = {
                "distinct": sorted(x for x in got if x is not None),
                "stable": len({x for x in got if x is not None}) == 1,
            }

    # the margin the verdict is decided by, which is the number that says how
    # close the invariance came to failing
    margin = rp.spread([row["discriminating"].get("delta_over_beta")
                        for row in rows])
    below = [row for row in rows
             if (row["discriminating"].get("delta_over_beta") or 0.0) < 1.0]
    out["decision_margin"] = {
        "delta_over_beta": margin,
        "verdict_is_refuse_iff_ratio_above_one": True,
        "cells_below_one": [f"{r['rung']}:{r['seam_id']}@{r['step']}"
                            for r in below],
        "closest_cell": (min(rows, key=lambda r: (
            r["discriminating"].get("delta_over_beta") or float("inf")))
            if rows else None),
    }
    out["verdict_stability"] = verdicts
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=110,
                    help="macro-steps to march (probe states are a subset)")
    ap.add_argument("--rungs", default=",".join(rp.PROBE_RUNGS))
    ap.add_argument("--only", default="",
                    help="comma-separated stages: march,grid,floor,zero,compile,analysis")
    ap.add_argument("--reuse-state", action="store_true")
    ap.add_argument("--no-tau", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="N6 only, probe states 0/10/30, for a smoke run")
    args = ap.parse_args(argv)

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    want = (lambda s: (not only) or s in only)

    steps = args.steps
    snap_at = list(rp.PROBE_STEPS)
    rung_labels = [s.strip() for s in args.rungs.split(",") if s.strip()]
    if args.quick:
        rung_labels = ["N6"]
        snap_at = [0, 10, 30]
        steps = 30
    snap_at = [k for k in snap_at if k <= steps]

    rungs = {r.label: r for r in sl.ladder()}
    chosen = [rungs[label] for label in rung_labels]

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "w106.json")
    art: dict = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                art = json.load(fh)
        except json.JSONDecodeError:
            art = {}
    art.setdefault("case", "CS-8 reuse_probe -- is a certificate the expert's or the state's?")
    art["date"] = time.strftime("%Y-%m-%d")
    art["argv"] = sys.argv[1:]
    art["steps"] = steps
    art["snap_at"] = snap_at
    art["design"] = rp.design_report(chosen)
    art.setdefault("march", {})
    art.setdefault("grid", [])
    art.setdefault("floor", [])
    art.setdefault("compile", {})
    persist(art, path)

    states: dict[str, dict] = {}

    banner("1. the trajectory -- exposed reference solver + ProjectedAssembly")
    for r in chosen:
        snaps = load_states(r, snap_at) if args.reuse_state else None
        if snaps and set(snaps) >= set(snap_at):
            print(f"  {r.label:4s} reusing cached states {sorted(snaps)}",
                  flush=True)
            rec = art["march"].get(r.label, {"rung": r.label, "health": {}})
            rec["health"] = {int(k): v for k, v in rec.get("health", {}).items()}
            if not rec["health"]:
                div0 = None
                for k in sorted(snaps):
                    h = rp.probe_state_health(*snaps[k], k, r.tiling,
                                              div_ref=div0)
                    div0 = div0 if div0 is not None else h["div_rms"]
                    rec["health"][k] = h
        elif want("march"):
            snaps, rec = stage_march(r, steps, snap_at)
            save_states(r, snaps)
        else:
            print(f"  {r.label:4s} no cached state and march not requested",
                  flush=True)
            continue
        states[r.label] = {"snaps": snaps, "rec": rec}
        art["march"][r.label] = rec
        persist(art, path)

    if want("grid"):
        banner("2. the grid -- every comparable seam at every probe state")
        rows = list(art.get("grid", []))
        have = {(row["rung"], row["seam_id"], row["step"]) for row in rows}
        art["grid"] = rows

        def keep(row):
            """Persist after every cell, not after every rung."""
            rows.append(row)
            art["grid"] = rows
            persist(art, path)

        for r in chosen:
            if r.label not in states:
                continue
            try:
                stage_grid(r, states[r.label]["snaps"], states[r.label]["rec"],
                           tau=not args.no_tau, on_row=keep, have=have)
            except Exception:                                   # noqa: BLE001
                traceback.print_exc()
            persist(art, path)

    if want("floor"):
        banner("3. control FLOOR -- the identical pair, probed twice")
        frows = [row for row in art.get("floor", [])
                 if row["rung"] not in {r.label for r in chosen}]
        for r in chosen:
            if r.label not in states:
                continue
            try:
                frows += stage_floor(r, states[r.label]["snaps"],
                                     states[r.label]["rec"])
            except Exception:                                   # noqa: BLE001
                traceback.print_exc()
            art["floor"] = frows
            persist(art, path)

    if want("compile"):
        banner("4. the compiles, with W105 asserted on every graph")
        for r in chosen:
            if r.label not in states:
                continue
            k = sorted(states[r.label]["snaps"])[-1]
            art["compile"][r.label] = stage_compile(r, states[r.label]["snaps"][k])
            persist(art, path)

    if want("zero") and art.get("grid"):
        banner("5. control ZERO -- the freestream, where the answer is known")
        art["zero_control"] = stage_zero(art["grid"])
        persist(art, path)

    if want("analysis") and art.get("grid"):
        banner("6. the analysis -- how much moved, against the floor")
        art["analysis"] = stage_analysis(art["grid"], art.get("floor", []))
        persist(art, path)
        report(art)

    print(f"\nartifact: {path}")
    return art


def report(art):
    """The tables section 20 quotes, printed so the log alone is readable."""
    an = art.get("analysis") or {}
    fields = list(rp.CertificateReading.DISCRIMINATING)
    print()
    print("floor (worst re-probe disagreement, over all floor cells)")
    for f in fields:
        print(f"   {f:16s} {_fmt((an.get('floor') or {}).get('worst_over_all', {}).get(f), '.4e')}")
    print()
    print("headline: median range over each factor, absolute and relative")
    print(f"   {'field':16s} {'factor':10s} {'range':>12s} {'of level':>12s}  verdict")
    for factor in ("seam", "state", "geometry", "replicate"):
        head = (an.get("headline") or {}).get(factor, {})
        for f in fields:
            row = head.get(f, {})
            rel = row.get("median_range_over_level")
            print(f"   {f:16s} {factor:10s} "
                  f"{_fmt(row.get('median_abs_range'), '.4e'):>12s} "
                  f"{(_fmt(100.0 * rel, '.3g') + '%' if rel is not None else '--'):>12s}"
                  f"  {row.get('verdict')}")
    print()
    dm = (an.get("decision_margin") or {}).get("delta_over_beta") or {}
    print(f"decision margin ||Delta||/beta over every cell: "
          f"{_fmt(dm.get('min'))} to {_fmt(dm.get('max'))} "
          f"(refuse on the admissible axis iff > 1)")
    print()
    print("verdict, swept over the admissible beta_min axis")
    for rung, row in (an.get("verdict_stability") or {}).items():
        called = row["as_called"]
        print(f"   {rung:5s} as called: {called['distinct']} "
              f"({called['n_cells']} cells) -- a constant by construction")
        for bm, got in row["swept"].items():
            print(f"        beta_min={bm:>8s}  {got['distinct']}  "
                  f"stable={got['stable']}")


if __name__ == "__main__":
    main()
