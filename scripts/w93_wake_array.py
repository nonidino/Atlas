"""Tier 18 -- the attribution machinery, on a real pretrained expert in a real wake.

Driver for `atlas/cases/wake_array.py`.  Everything [[tier0-measurements]] 18
quotes comes from here; artifact `out/w93/w93.json`.

    python scripts/w93_wake_array.py            # everything
    python scripts/w93_wake_array.py --quick    # short march, no compile

What it measures, in order
--------------------------

  1. the array marches, and the power table -- three turbines at 3.5 D in an L,
     against a turbine-free control and a single-turbine control
  2. `probe.support_reach` on Poseidon-T, which is the first time W75's gate has
     been pointed at a learned operator (W93)
  3. `poseidon.elliptic_signature` and `probe.operator_content` on the same block
  4. `probe.assemble_seam` on the wake seam and on a rotor seam, with the seam
     base the march supplies (W92's "a run has it for free")
  5. `multiphysics.seam_defect_split` -- the calibration ladder on Poseidon, then
     tau for Poseidon substituted for the reference solver, then the band tau
     inherits from the referent's own viscosity (W95)
  6. `composition.certify_substitution` at both seams, swept in beta_min (W76)
  7. `compile_scheme`, at `elliptic_subsolve` unknown and at the value step 2
     measures
  8. the consequence in the currency the case study is about: the wake loss the
     two experts predict for the same array

Nothing here is a scheme.  The march is a measurement instrument that produces
the STATE every probe is taken at, the way `tier0_window_ns` runs a monolith it
would never ship.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np                                                   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.cases import wake_array as wa                             # noqa: E402
from atlas.cases.poseidon import elliptic_signature                  # noqa: E402
from atlas.composition import certify_substitution                   # noqa: E402
from atlas.compiler import compile_scheme                            # noqa: E402
from atlas.capability import EllipticSubsolve                        # noqa: E402
from atlas.multiphysics import (                                     # noqa: E402
    lag_distance,
    numerical_jacobian,
    seam_defect_split,
    tight_couple,
)
from atlas.ports import ResponseHalf                                 # noqa: E402
from atlas.probe import (                                            # noqa: E402
    OPERATOR_CONTENT_FLOOR,
    ProbeBudget,
    assemble_seam,
    base_sensitivity,
    operator_content,
    support_reach,
)
from atlas.scheme import Budget                                      # noqa: E402
from atlas.transfer import InterfaceSpace, SeamTransfer              # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "out", "w93")

#: The seam every headline number is measured on: the open flow beside R2, at the
#: plane 3.5 D downstream of R1, so it is inside R1's developed wake and it is
#: fluid-to-fluid, where the port's own nu cancels exactly.
WAKE_SEAM = "x1r0_bypass"
#: The other one: field-to-lumped, where it does not.
ROTOR_SEAM = "R2_up"

#: W0 4.2's three fits for the checkpoint's viscosity, in this case's units.
#: The lambda = 0.25 D fit is NEGATIVE (-1.4e-5) and is not a viscosity, so the
#: band is bracketed by the two positive ones and their geometric mean.
NU_P_FITS = {"lambda=0.125D (r2=0.998)": 4.9e-4,
             "geometric mean": float(np.sqrt(4.9e-4 * 1.8e-5)),
             "lambda=0.5D (r2=0.27)": 1.8e-5}


def _f(x):
    """JSON-safe."""
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
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
    print("=" * 78)


# ---------------------------------------------------------------------------
# 1. the march
# ---------------------------------------------------------------------------

#: Freestream band, in cells: the domain's own outer boundary.  The farm sits in
#: an unbounded stream, so the inlet and the two lateral edges are held at
#: ``(U_INF, 0)`` and the outlet is left free.  Without it the disks drain the
#: box: measured, every turbine's inflow falls monotonically to 0.06 by step 45
#: with nothing replenishing the momentum they remove.
BAND = 8


def disk_rows(y_centre):
    y = (np.arange(wa.NY) + 0.5) * wa.DX
    return np.abs(y - y_centre) <= 0.5 * wa.ROTOR_D + 1e-9


def march(active, n_steps, kind="poseidon", nu_solver=None, record_prev=False,
          thickness=None, project=True, snapshots=None, snap_every=2):
    """Advance the array; return the final state and each turbine's inflow history.

    The disk's smearing thickness is DERIVED rather than chosen:
    ``Delta_d = <U_d> * dt``.  A body-force impulse over one macro-step is
    ``f dt = (T / (A Delta_d)) dt`` and momentum theory says a parcel crossing the
    disk must lose ``T / (A <U_d>)``; the two agree only at that thickness.  With
    `disk.py`'s default 0.1 D against this expert's fixed ``dt = 0.2`` the impulse
    is roughly 2x too large; measured over four macro-steps, min u goes from
    **+0.4313 to +0.2073**.  R4 forbids shrinking the exchange interval below
    ``max_i dt_i`` and a frozen checkpoint's ``dt_native`` is not a dial, so the
    disk's thickness is the only thing left to move -- which makes it a property
    of the EXPERT rather than of the rotor.  See `stage_disk_controls`, which also
    measures the projection: without it the same four steps reverse the flow
    through the disk plane.

    **W98 lives in the four lines that step the checkpoint.**  Transport and
    pressure are global operations and this loop used to buy both from
    `step_many`, which does them PER WINDOW and PERIODICALLY -- so a wake left a
    window's outflow edge and re-entered its own inflow edge, and a disk's
    pressure response was smeared over its own 4 D box.  Measured on a lone
    turbine, that manufactured a 25% velocity deficit 3.5 D UPSTREAM of it, where
    no cause exists.  `wake_array.transport_and_project` is the argument; the
    residual after it is 7% and belongs to the checkpoint (W93), not the march.
    """
    import importlib
    dk = importlib.import_module("atlas_windfarm_reference.disk")
    x_c = (np.arange(wa.NX) + 0.5) * wa.DX
    y_c = (np.arange(wa.NY) + 0.5) * wa.DX
    tiling = wa.DEFAULT_TILING

    if kind == "poseidon":
        ex = wa.scaled_expert()
    else:
        ex = wa.reference_solver(nu_solver or wa.NU_REF)

    u = np.ones((wa.NY, wa.NX))
    v = np.zeros((wa.NY, wa.NX))
    prev = None
    hist = []
    for _step in range(n_steps):
        if snapshots is not None and _step % snap_every == 0:
            snapshots.append((u[::2, ::2].copy(), v[::2, ::2].copy()))
        fx = np.zeros((wa.NY, wa.NX))
        rec = {}
        for rot in wa.ROTORS:
            rows = disk_rows(rot.y_centre)
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
        hist.append(rec)
        if record_prev:
            prev = (u.copy(), v.copy())
        if kind == "poseidon":
            # The checkpoint advances the FLUCTUATION in the co-moving frame and
            # does nothing else: no per-window translation (galilean=False), no
            # per-window projection. Transport and pressure are the composition
            # layer's, they are GLOBAL, and they are applied once below -- W98.
            ufs, vfs = tiling.cut(u - wa.U_INF), tiling.cut(v)
            u1, v1 = ex.step_many(ufs, vfs, wa.MACRO_DT, galilean=False,
                                  project=False)
            # the checkpoint's own mean drift, removed -- but the INCOMING window
            # mean is kept. A periodic box with no force conserves it, and the
            # deficit a disk puts there has to survive long enough to be carried
            # out of the domain instead of being deleted every step.
            u1 += (ufs.mean(axis=(1, 2)) - u1.mean(axis=(1, 2)))[:, None, None]
            v1 += (vfs.mean(axis=(1, 2)) - v1.mean(axis=(1, 2)))[:, None, None]
            uf, vf = tiling.assemble(u1, v1)
            # `step_many` DROPS `force` when galilean is False, and silently
            # (W99), so the disks are applied here. The partition of unity sums
            # to one, so applying fx after assembly is applying fs before it.
            uf = uf + fx * wa.MACRO_DT
            uf, vf = wa.transport_and_project(uf, vf, wa.MACRO_DT, wa.U_INF,
                                              project=project)
            u, v = wa.U_INF + uf, vf
        else:
            # WindowNS needs neither: its own projection is embedded, which is
            # exactly what R10 refuses to decompose and exactly what makes it
            # usable here without one. It also holds a real Dirichlet ring, so a
            # window's inflow is its NEIGHBOUR's field rather than its own
            # outflow -- which is why the reference solver never had W98.
            us, vs = tiling.cut(u), tiling.cut(v)
            fs = tiling.cut(fx)
            u1, v1 = ex.step_batch(us, vs, wa.MACRO_DT, bc0=None,
                                   force=(fs, np.zeros_like(fs)))
            u, v = tiling.assemble(u1, v1)
        u[:, :BAND] = wa.U_INF
        v[:, :BAND] = 0.0
        u[:BAND, :] = wa.U_INF
        v[:BAND, :] = 0.0
        u[-BAND:, :] = wa.U_INF
        v[-BAND:, :] = 0.0
    return u, v, hist, prev


def stage_disk_controls():
    """The two things the march does that are NOT free choices, measured.

    Both are consequences of the expert being frozen at a fixed native step, and
    both were established by watching the run fail before they were derived.
    """
    banner("1b. the two disk-coupling controls")
    import importlib
    dk = importlib.import_module("atlas_windfarm_reference.disk")
    rows = []
    cases = [
        ("derived thickness, projection ON  (as shipped)", None, True),
        ("disk.py default 0.1 D, projection ON", dk.ActuatorDisk().thickness, True),
        ("derived thickness, projection OFF", None, False),
    ]
    for label, thick, proj in cases:
        u, v, hist, _ = march({"R1", "R2", "R3"}, 4, "poseidon",
                              thickness=thick, project=proj)
        rows.append({"case": label, "thickness": thick, "project": proj,
                     "u_min": float(u.min()), "u_max": float(u.max()),
                     "Ud_R1": float(hist[-1]["R1"])})
        print(f"  {label:44s} u_min={u.min():+.4f}  u_max={u.max():+.4f}")
    print("  A momentum sink must not reverse the flow it acts on: the impulse per")
    print("  parcel is T*dt/(A*Delta_d) and momentum theory allows T/(A*<U_d>), so")
    print("  the two agree only at Delta_d = <U_d>*dt. R4 forbids shrinking dt, and")
    print("  a frozen checkpoint's dt_native is not a dial, so the disk thickness")
    print("  is a property of the EXPERT.")
    return {"rows": rows, "n_steps": 4}


def power_of(ud):
    import importlib
    dk = importlib.import_module("atlas_windfarm_reference.disk")
    return float(dk.ActuatorDisk()(max(float(ud), 1e-6)).power)


def settled(hist, key, n=8):
    tail = [h[key] for h in hist[-n:]]
    return float(np.mean(tail)), float(np.max(tail) - np.min(tail))


def stage_power(n_steps, kind="poseidon", nu_solver=None):
    banner(f"1. the array, marched -- {kind}")
    runs = {}
    t0 = time.time()
    for label, active in (("free", set()),
                          ("solo_R1", {"R1"}),
                          ("R1_off", {"R2", "R3"}),
                          ("all", {"R1", "R2", "R3"})):
        u, v, hist, prev = march(active, n_steps, kind, nu_solver,
                                 record_prev=(label == "all"))
        runs[label] = {"state": (u, v), "hist": hist, "prev": prev}
        line = "  ".join(f"{r.rotor_id} Ud={settled(hist, r.rotor_id)[0]:.4f}"
                         for r in wa.ROTORS)
        print(f"  {label:8s}  {line}   umin={u.min():+.4f} umax={u.max():+.4f}")
    print(f"  ({time.time() - t0:.1f}s for four runs of {n_steps} macro-steps)")

    out = {"n_steps": n_steps, "expert": kind, "turbines": {}}
    ud_free = {r.rotor_id: settled(runs["free"]["hist"], r.rotor_id)
               for r in wa.ROTORS}
    out["turbine_free_control"] = {
        k: {"Ud": v[0], "spread": v[1], "drift_from_unity": v[0] - 1.0}
        for k, v in ud_free.items()}
    print("\n  turbine-free control -- the array's own noise floor:")
    for k, v in ud_free.items():
        print(f"    {k}: <Ud> = {v[0]:.6f}  ({100 * (v[0] - 1):+.3f}% from freestream, "
              f"spread {v[1]:.2e})")

    ud_solo, _ = settled(runs["solo_R1"]["hist"], "R1")
    p_iso = power_of(ud_solo)
    print(f"\n  isolated turbine (R1 alone): <Ud> = {ud_solo:.4f}, momentum theory "
          f"says U_inf(1-a) = {1 - 1/3:.4f}  [{100*(ud_solo/(2/3) - 1):+.2f}%]")
    print(f"  P_isolated = {p_iso:.5f}")

    print("\n  power table:")
    print(f"    {'turbine':8s} {'Ud(all)':>9s} {'Ud(ctl)':>9s} {'P(all)':>9s} "
          f"{'P(ctl)':>9s} {'loss':>9s}   control")
    controls = {"R1": "solo_R1", "R2": "R1_off", "R3": "R1_off"}
    total = 0.0
    for rot in wa.ROTORS:
        rid = rot.rotor_id
        ud_a, sp_a = settled(runs["all"]["hist"], rid)
        ud_c, _ = settled(runs[controls[rid]]["hist"], rid)
        pa, pc = power_of(ud_a), power_of(ud_c)
        total += pa
        loss = 100.0 * (1.0 - pa / pc)
        print(f"    {rid:8s} {ud_a:9.4f} {ud_c:9.4f} {pa:9.5f} {pc:9.5f} "
              f"{loss:+8.2f}%   {controls[rid]}")
        out["turbines"][rid] = {
            "Ud_array": ud_a, "Ud_spread": sp_a, "Ud_control": ud_c,
            "P_array": pa, "P_control": pc, "wake_loss_pct": loss,
            "control_run": controls[rid],
            "kind": "blockage" if rid == "R1" else "wake loss",
            "x_plane": rot.x_plane, "y_centre": rot.y_centre,
        }
    eff = total / (len(wa.ROTORS) * p_iso)
    out["isolated"] = {"Ud": ud_solo, "P": p_iso,
                       "momentum_theory_Ud": 2.0 / 3.0,
                       "rel_error": ud_solo / (2.0 / 3.0) - 1.0}
    out["array_efficiency"] = eff
    out["array_power"] = total
    print(f"\n  array efficiency  sum P / (3 P_isolated) = {eff:.4f}  "
          f"({100 * (1 - eff):.2f}% lost to wakes and blockage)")
    return out, runs


# ---------------------------------------------------------------------------
# 2-3. what the two elliptic instruments read on a frozen checkpoint
# ---------------------------------------------------------------------------


def stage_reach(experts_p, experts_r):
    banner("2-3. support_reach, elliptic_signature and operator_content on Poseidon-T")
    out = {}
    port = wa.port_name("xhi", "full")
    for tag, ex in (("poseidon", experts_p["F01"]), ("reference", experts_r["F01"])):
        base = ex.probe_base(port)
        rows = []
        for amp in (1.0, 1e-1, 1e-2, 1e-3):
            sr = support_reach(ex.respond, port, base, amplitude=amp)
            rows.append({"amplitude": amp, **sr.as_dict()})
            print(f"  {tag:9s} amp={amp:7.0e}  nonzero={sr.nonzero:3d}/{sr.n}  "
                  f"fraction={sr.fraction:.4f}  reach={sr.reach:3d}  "
                  f"is_global={sr.is_global}  -> {sr.consistent_with}")
        out[tag] = {"support_reach": rows}

    caps = wa.fluid_capabilities(experts_p["F01"])
    product = int(caps.stencil_radius) * int(caps.substeps_per_macro_step)
    declared = caps.required_halo()
    measured = max(r["reach"] for r in out["poseidon"]["support_reach"])
    print(f"\n  W93. stencil_radius x substeps                    = {product} cells")
    print(f"       measured reach (support_reach, same quantity) = {measured} cells "
          f"({measured / product:.0f}x)")
    print(f"       the graph's overlap                           = {wa.HALO} cells, so "
          f"the halo rule PASSED on the product")
    print(f"       required_halo() returns                       = {declared} "
          f"-- undecidable, since the record does not say the step is explicit")
    out["W93"] = {"radius_times_substeps": product, "measured_reach": measured,
                  "ratio": measured / product, "overlap_cells": wa.HALO,
                  "halo_rule_passed_on_the_product": wa.HALO >= product,
                  "required_halo_after_fix": declared}

    # one block, on the declared basis, at the physical base
    for tag, ex in (("poseidon", experts_p["F01"]), ("reference", experts_r["F01"])):
        base = ex.probe_base(port)
        B = wa.fourier_basis(wa.N)
        eps = 1e-2
        z = ex.respond(port, base)
        S = wa.DX * (B.T @ np.column_stack(
            [(ex.respond(port, base + eps * B[:, k]) - z) / eps
             for k in range(B.shape[1])]))
        es = elliptic_signature(S)
        om = operator_content(S)
        sv = np.linalg.svd(S, compute_uv=False)
        print(f"\n  {tag}: ||S||={np.linalg.norm(S):.5g}  beta={sv[-1]:.4g}  "
              f"kappa={es['kappa']:.4g}  asym={es['asymmetry']:.4g}  omega={om:.4g} "
              f"({'above' if om >= OPERATOR_CONTENT_FLOOR else 'below'} the floor)")
        print(f"     elliptic_signature: {es['verdict']}")
        out[tag].update({"block_norm": float(np.linalg.norm(S)),
                         "beta": float(sv[-1]), "kappa": es["kappa"],
                         "asymmetry": es["asymmetry"], "operator_content": om,
                         "elliptic_signature": es["verdict"]})
        out[tag]["S"] = S.tolist()

    Xi = (np.linalg.norm(np.array(out["poseidon"]["S"]), 2)
          / np.linalg.norm(np.array(out["reference"]["S"]), 2))
    out["Xi_poseidon_over_reference"] = float(Xi)
    print(f"\n  Xi = ||Lambda_poseidon|| / ||Lambda_reference|| = {Xi:.4f}")
    for t in ("poseidon", "reference"):
        out[t].pop("S")
    return out


# ---------------------------------------------------------------------------
# 4. the assembled seam operators, at the base the run supplies
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
    """The consistent seam base, from the state the march produced (W92).

    Both sides' own `probe_base` reduced to M and averaged: the two rings are
    HALO cells apart in a wake, so they are NOT the same number, and averaging
    them is the cheapest consistent point rather than the right one.  What it
    buys is that `base_disagreement` reports the spread instead of reading zero.
    """
    conn = graph.connection(seam_id)
    tr = transfer_for(graph, seam_id)
    coeffs = []
    for agent_id, name in (conn.a, conn.b):
        base_V = np.asarray(experts[agent_id].probe_base(name), dtype=float)
        R = tr.prolongations[agent_id].adjoint(tr.space)
        coeffs.append(R @ base_V)
    return 0.5 * (coeffs[0] + coeffs[1]), coeffs


def stage_seams(graph_p, experts_p, graph_r, experts_r):
    banner("4. assemble_seam at the wake seam and a rotor seam")
    out = {}
    for seam_id in (WAKE_SEAM, ROTOR_SEAM, "x0r0_bypass", "y0c1"):
        tr = transfer_for(graph_p, seam_id)
        base_M, sides = seam_base_M(graph_p, experts_p, seam_id)
        rows = {}
        for tag, graph in (("poseidon", graph_p), ("reference", graph_r)):
            if seam_id == ROTOR_SEAM and tag == "reference":
                continue
            t0 = time.time()
            op = assemble_seam(graph, graph.connection(seam_id), tr,
                               budget=ProbeBudget(),
                               expected_null_dim=graph.connection(seam_id).expected_null_dim,
                               probe_state="developed wake, 3-turbine L array",
                               seam_base=base_M)
            d = op.as_dict()
            d["wall_s"] = time.time() - t0
            rows[tag] = d
            print(f"  {seam_id:14s} {tag:9s} dim M={op.dim_M:2d} beta={op.beta:.4g} "
                  f"kappa={op.kappa:.4g} null={op.null_dim}/{op.expected_null_dim} "
                  f"pi={op.passivity_defect:.3g} omega={op.operator_content:.4g}")
            shares = {k: b.share for k, b in op.blocks.items()}
            print(f"                 {'':9s} shares={ {k: round(v, 4) for k, v in shares.items()} }  "
                  f"one_sided={op.one_sided:.4g}")
            print(f"                 {'':9s} base: {op.derived_probe_state}")
        out[seam_id] = rows
        out[seam_id]["seam_base_M_spread"] = float(np.max(np.abs(sides[0] - sides[1])))
    print()
    # the base is not free here, and the check that says so
    port = wa.port_name("xhi", "full")
    B = wa.fourier_basis(wa.N)[:, :5]
    for tag, ex in (("poseidon", experts_p["F01"]), ("reference", experts_r["F01"])):
        bs = base_sensitivity(ex.respond, port, ex.probe_base(port), B, 1e-2)
        print(f"  base_sensitivity ({tag}): relative_change="
              f"{bs['relative_change']:.4g} at a {bs['shift']:.3g} shift  "
              f"affine={bs['affine']}")
        out.setdefault("base_sensitivity", {})[tag] = _f(bs)
    return out


# ---------------------------------------------------------------------------
# 5. attribution
# ---------------------------------------------------------------------------


def seam_callables(graph, experts, seam_id):
    conn = graph.connection(seam_id)
    return {agent_id: (lambda t, a=agent_id, n=name: experts[a].respond(n, t))
            for agent_id, name in (conn.a, conn.b)}


def stage_attribution(graph_p, experts_p, graph_r, experts_r, state, prev_state,
                      quick=False):
    banner("5. seam_defect_split -- tau for Poseidon-T, against a reference PAIR")
    out = {}
    conn = graph_p.connection(WAKE_SEAM)
    ids = [conn.a[0], conn.b[0]]
    names = {conn.a[0]: conn.a[1], conn.b[0]: conn.b[1]}
    halves = {a: ResponseHalf.EFFORT for a in ids}

    # the physical trace this seam carries, and the one it carried a step earlier
    trace0 = np.mean([np.asarray(experts_p[a].probe_base(names[a])) for a in ids],
                     axis=0)
    pu, pv = prev_state
    prev_experts = wa.make_experts(pu, pv, "poseidon")
    lagged = np.mean([np.asarray(prev_experts[a].probe_base(names[a])) for a in ids],
                     axis=0)
    lag = lag_distance(lagged, trace0)
    print(f"  seam {WAKE_SEAM}: dim V = {trace0.size}, "
          f"trace mean = {trace0.mean():.5f}, lag over one macro-step = {lag:.4e}")
    out["seam"] = WAKE_SEAM
    out["dim_V"] = int(trace0.size)
    out["trace_mean"] = float(trace0.mean())
    out["lag_distance"] = float(lag)

    ref_calls = seam_callables(graph_r, experts_r, WAKE_SEAM)
    pos_calls = seam_callables(graph_p, experts_p, WAKE_SEAM)
    tc_ref = tight_couple(
        lambda lam: sum(np.asarray(ref_calls[a](lam)).ravel() for a in ids), trace0)
    print(f"  referent on the WindowNS pair: converged={tc_ref.converged} in "
          f"{tc_ref.iterations} iterations, residual {tc_ref.residual_norm:.4e} "
          f"(from {tc_ref.history[0]:.4e})")
    out["referent_on_reference_pair"] = {
        "converged": bool(tc_ref.converged), "iterations": int(tc_ref.iterations),
        "residual_norm": float(tc_ref.residual_norm),
        "history": [float(h) for h in tc_ref.history]}

    # -- 5a(i). the positive control on the REFERENCE pair ------------------
    print("\n  5a(i). calibration on the reference pair: actual = a scaled WindowNS "
          "side.")
    print("         tau must recover the injected error and the unswapped side must "
          "read zero.")
    cal = []
    for alpha in (1.05, 1.25, 2.0):
        actual = dict(ref_calls)
        actual[ids[0]] = (lambda t, a=ids[0], s=alpha: s * np.asarray(ref_calls[a](t)))
        sd = seam_defect_split(WAKE_SEAM, actual, ref_calls, halves, trace0,
                               lagged, measure=wa.DX, jacobian="fd")
        cal.append({"alpha": alpha, **sd.as_dict()})
        print(f"         alpha={alpha:.2f}: tau[{ids[0]}]={sd.tau[ids[0]]:.6f}  "
              f"tau[{ids[1]}]={sd.tau[ids[1]]:.3e}  converged={sd.converged}  "
              f"subadd={sd.subadditive}")
    out["calibration_reference_pair"] = cal

    # -- 5a(ii). the same thing, ON THE CHECKPOINT --------------------------
    if not quick:
        print("\n  5a(ii). can the referent be built from the CHECKPOINT ALONE?")
        print("      W85 priced the referent in solves; on a learned expert the "
              "question is prior to the price.")
        t0 = time.time()
        resid = lambda lam: sum(np.asarray(pos_calls[a](lam)).ravel() for a in ids)
        J = numerical_jacobian(resid, trace0, step=1e-2)
        sv = np.linalg.svd(J, compute_uv=False)
        rank_eff = int(np.sum(sv > 1e-8 * sv[0]))
        print(f"      dense FD Jacobian on the Poseidon pair: {2 * (trace0.size + 1)} "
              f"forward passes, {time.time() - t0:.1f}s")
        print(f"      J is {J.shape}: sigma_max={sv[0]:.4e} sigma_min={sv[-1]:.4e} "
              f"kappa={sv[0] / max(sv[-1], 1e-300):.4e}  effective rank "
              f"{rank_eff}/{J.shape[0]} at a 1e-8 relative cut")
        out["jacobian_poseidon_pair"] = {
            "shape": list(J.shape), "sigma_max": float(sv[0]),
            "sigma_min": float(sv[-1]),
            "kappa": float(sv[0] / max(sv[-1], 1e-300)),
            "effective_rank": rank_eff,
            "singular_values_head": [float(x) for x in sv[:8]],
            "singular_values_tail": [float(x) for x in sv[-8:]]}
        # Is the small end of J a derivative or is it noise?  Measure it: the
        # SAME Jacobian at a second probe step, and the relative disagreement is
        # the floor below which a singular direction means nothing.
        J2 = numerical_jacobian(resid, trace0, step=5e-3)
        repro = float(np.linalg.norm(J - J2) / np.linalg.norm(J))
        noise_cut = repro * float(sv[0])
        below = int(np.sum(sv < noise_cut))
        print(f"      the same J at a 2x smaller probe step differs by "
              f"{repro:.3e} relative, so {below}/{len(sv)} singular directions "
              f"sit below the probe's own reproducibility ({noise_cut:.3e})")
        out["jacobian_poseidon_pair"].update({
            "reproducibility_relative": repro, "noise_cut": noise_cut,
            "directions_below_noise": below})

        U, sig, Vt = np.linalg.svd(J)
        keep = sig >= noise_cut

        def tsvd_step(r, U=U, sig=sig, Vt=Vt, keep=keep):
            rr = np.asarray(r, dtype=float).ravel()
            return -(Vt[keep].T @ ((U[:, keep].T @ rr) / sig[keep]))

        tct = tight_couple(resid, trace0, jacobian=tsvd_step)
        print(f"      truncated at that cut ({int(keep.sum())} directions kept): "
              f"converged={tct.converged} in {tct.iterations} it, residual "
              f"{tct.residual_norm:.4e} (from {tct.history[0]:.4e})")
        out["referent_truncated_svd"] = {
            "kept": int(keep.sum()), "converged": bool(tct.converged),
            "iterations": int(tct.iterations),
            "residual_norm": float(tct.residual_norm),
            "history": [float(h) for h in tct.history[:8]]}

        damp_rows = []
        for damping in (1.0, 0.5, 0.2):
            tcd = tight_couple(resid, trace0, jacobian=J, damping=damping)
            damp_rows.append({"damping": damping, "converged": bool(tcd.converged),
                              "iterations": int(tcd.iterations),
                              "residual_norm": float(tcd.residual_norm),
                              "history": [float(h) for h in tcd.history[:6]]})
            print(f"      damping={damping}: converged={tcd.converged} in "
                  f"{tcd.iterations} it, residual {tcd.residual_norm:.4e} "
                  f"(from {tcd.history[0]:.4e})")
        out["referent_damping_sweep"] = damp_rows
        tc = tight_couple(resid, trace0, jacobian=J)
        print(f"      referent: converged={tc.converged} in {tc.iterations} "
              f"iterations, residual {tc.residual_norm:.4e} "
              f"(from {tc.history[0]:.4e})")
        print(f"      history: {['%.3e' % h for h in tc.history[:8]]}")
        out["referent_on_poseidon_pair"] = {
            "converged": bool(tc.converged), "iterations": int(tc.iterations),
            "residual_norm": float(tc.residual_norm),
            "history": [float(h) for h in tc.history],
            "jacobian_step": 1e-2,
        }
        rows = []
        for alpha in (1.05, 1.25, 2.0):
            actual = dict(pos_calls)
            actual[ids[0]] = (lambda t, a=ids[0], s=alpha:
                              s * np.asarray(pos_calls[a](t)))
            sd = seam_defect_split(WAKE_SEAM, actual, pos_calls, halves, trace0,
                                   lagged, measure=wa.DX, jacobian=J)
            rows.append({"alpha": alpha, **sd.as_dict()})
            if not sd.converged:
                print(f"      alpha={alpha:.2f}: NO REFERENT -- {sd.note[:100]}")
                continue
            print(f"      alpha={alpha:.2f}: tau[{ids[0]}]={sd.tau[ids[0]]:.6f}  "
                  f"tau[{ids[1]}]={sd.tau[ids[1]]:.3e}  sigma={sd.sigma:.4e}  "
                  f"converged={sd.converged}  subadditive={sd.subadditive}")
        out["calibration"] = rows

    # -- 5b. the measurement ------------------------------------------------
    print("\n  5b. reference = the WindowNS pair at nu = %.4g; actual = Poseidon-T "
          "on one side, then both." % wa.NU_REF)
    cases = {
        "reference_pair (control)": dict(ref_calls),
        f"poseidon on {ids[0]} only": {**ref_calls, ids[0]: pos_calls[ids[0]]},
        f"poseidon on {ids[1]} only": {**ref_calls, ids[1]: pos_calls[ids[1]]},
        "poseidon on both": dict(pos_calls),
    }
    rows = {}
    for label, actual in cases.items():
        sd = seam_defect_split(WAKE_SEAM, actual, ref_calls, halves, trace0,
                               lagged, measure=wa.DX, jacobian="fd")
        rows[label] = sd.as_dict()
        if not sd.converged:
            print(f"      {label:28s} REFERENT DID NOT CONVERGE -- {sd.note[:70]}")
            continue
        print(f"      {label:28s} tau[{ids[0]}]={sd.tau[ids[0]]:.5f}  "
              f"tau[{ids[1]}]={sd.tau[ids[1]]:.5f}  sigma={sd.sigma:.4e}  "
              f"total={sd.total:.5f}  subadd={sd.subadditive}  "
              f"P_ref={sd.power_reference:.4e}  lam*={sd.trace_reference:.5f}")
    out["measurement"] = rows

    # -- 5c. the band the referent's own viscosity puts on tau (W95) --------
    print("\n  5c. W95: the checkpoint's viscosity is not a number, so the referent "
          "is a choice.")
    print(f"      {'nu_p fit':28s} {'nu_solver':>10s} {'Re_h(U)':>8s} {'Re_h(max)':>10s} "
          f"{'valid':>6s} {'tau[%s]' % ids[0]:>10s}")
    band = []
    for label, nu_p in NU_P_FITS.items():
        nu_s = nu_p * wa.S_LEN * 2.0
        ex_r = wa.make_experts(*state, "reference", nu_solver=nu_s)
        graph_rn, _ = wa.build(*state, kind="reference", experts=ex_r)
        rc = seam_callables(graph_rn, ex_r, WAKE_SEAM)
        cell_re = wa.DX * wa.U_INF / nu_s
        umax = float(np.max(np.hypot(*state)))
        cell_re_state = wa.DX * umax / nu_s
        ok = ex_r[ids[0]].reference_validity()
        sd = seam_defect_split(WAKE_SEAM, {**rc, ids[0]: pos_calls[ids[0]]}, rc,
                               halves, trace0, lagged, measure=wa.DX, jacobian="fd")
        tau = sd.tau.get(ids[0], float("nan")) if sd.converged else float("nan")
        print(f"      {label:28s} {nu_s:10.3e} {cell_re:8.2f} {cell_re_state:10.2f} "
              f"{'PASS' if ok else 'FAIL':>6s} {tau:10.5f}")
        band.append({"fit": label, "nu_p": nu_p, "nu_solver": nu_s,
                     "cell_reynolds_freestream": cell_re,
                     "cell_reynolds_state_max": cell_re_state,
                     "reference_validity": bool(ok),
                     "tau": None if tau != tau else float(tau),
                     "converged": sd.converged})
    taus = [b["tau"] for b in band if b["tau"] is not None]
    spread = (max(taus) / min(taus)) if taus and min(taus) > 0 else None
    out["referent_band"] = {"rows": band, "spread_ratio": spread}
    if spread:
        print(f"      the referent's viscosity is under-determined by 27x in cell "
              f"Reynolds number; tau moves {spread:.2f}x across it")
    return out


# ---------------------------------------------------------------------------
# 6. the substitution certificate
# ---------------------------------------------------------------------------


def stage_substitution(graph_p, experts_p, graph_r, experts_r, seams):
    banner("6. certify_substitution -- can the certificate see the swap? (W76)")
    out = {}
    for seam_id in (WAKE_SEAM, ROTOR_SEAM):
        tr = transfer_for(graph_p, seam_id)
        base_M, _ = seam_base_M(graph_p, experts_p, seam_id)
        conn = graph_p.connection(seam_id)
        # The rotor is the same object in both graphs, so `graph_r` is the whole
        # array with WindowNS in every fluid window and nothing else changed --
        # which is exactly the "before" the certificate compares against.
        target = next(a for a, _n in (conn.a, conn.b) if a.startswith("F"))
        op_ref = assemble_seam(graph_r, graph_r.connection(seam_id), tr,
                               seam_base=base_M,
                               expected_null_dim=conn.expected_null_dim)
        op_new = assemble_seam(graph_p, conn, tr, seam_base=base_M,
                               expected_null_dim=conn.expected_null_dim)
        S_old = op_ref.blocks[target].S
        S_new = op_new.blocks[target].S
        old_caps = graph_r.agent(target).capabilities
        new_caps = graph_p.agent(target).capabilities
        rows = []
        for beta_min in (0.0, 1e-12, 0.01, 0.1, 0.25, 0.5):
            cert = certify_substitution(
                target, old_caps, new_caps, S_old, S_new,
                beta=op_ref.beta, beta_min=beta_min,
                passivity_old=op_ref.blocks[target].passivity_lambda_min,
                passivity_new=op_new.blocks[target].passivity_lambda_min,
                block_norm=float(np.linalg.norm(S_old, 2)))
            rows.append({"beta_min": beta_min, **cert.as_dict()})
        c0 = rows[1]
        print(f"  {seam_id}: swap {target} WindowNS -> Poseidon-T")
        print(f"     ||Delta|| = {c0['delta_norm']:.5g}   beta = {c0['beta']:.5g}   "
              f"||S_i|| = {c0['block_norm']:.5g}   share = "
              f"{op_ref.blocks[target].share:.4f}   one_sided = {op_ref.one_sided:.4g}")
        for r in rows:
            print(f"     beta_min={r['beta_min']:<6g} verdict={r['verdict']:<18s} "
                  f"passes={str(r['passes']):5s} blind={r['blind']}")
        print(f"     {c0['message'][:200]}")
        out[seam_id] = {"target": target, "rows": rows,
                        "share": op_ref.blocks[target].share,
                        "one_sided": op_ref.one_sided}
    return out


# ---------------------------------------------------------------------------
# 7. the compiles
# ---------------------------------------------------------------------------


def stage_compile(state):
    banner("7. compile_scheme")
    out = {}
    t0 = time.time()
    graph, _ = wa.build(*state, kind="poseidon",
                        elliptic=EllipticSubsolve.UNKNOWN)
    res = compile_scheme(graph, probe_state="developed wake, 3-turbine L array")
    print(res.report())
    print(f"  ({time.time() - t0:.1f}s)")
    out["unknown"] = {
        "verdict": res.verdict.value,
        "refusals": [d.as_dict() for d in res.decisions.refusals],
        "decertifications": [d.as_dict() for d in res.decisions.decertifications],
        "stamp": {h.name: st.value for h, st in res.envelope.values.items()},
        "tau_undefined_seams": list(res.tau_undefined_seams),
        "unmeasured": list(res.unmeasured),
    }
    for tag, ell in (("embedded (what support_reach measures)",
                      EllipticSubsolve.EMBEDDED),
                     ("none (what poseidon.py declares by default)",
                      EllipticSubsolve.NONE)):
        g2, _ = wa.build(*state, kind="poseidon", elliptic=ell)
        r2 = compile_scheme(g2, budget=Budget(allow_probe=False))
        print(f"\n  elliptic_subsolve = {ell.value} -- {tag}: {r2.verdict.value}, "
              f"{len(r2.decisions.refusals)} refusals")
        for d in r2.decisions.refusals:
            print(f"      REFUSE {d.layer}/{d.rule} [{d.subject}] {d.message[:140]}")
        out[ell.value] = {
            "verdict": r2.verdict.value,
            "refusals": [d.as_dict() for d in r2.decisions.refusals],
            "note": "probe disabled: this compile is about the L2 verdict",
        }
    return out


# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="short march, skip the calibration ladder and the compile")
    ap.add_argument("--steps", type=int, default=45)
    ap.add_argument("--reuse-state", action="store_true",
                    help="load the marched state from out/w93/state.npz instead of "
                         "re-marching; the power table is then read from the "
                         "cached artifact")
    args = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    art = {"case": "wake_array", "date": time.strftime("%Y-%m-%d"),
           "geometry": _f({
               "S_LEN_D": wa.S_LEN, "dx_D": wa.DX, "macro_dt": wa.MACRO_DT,
               "N": wa.N, "NX": wa.NX, "NY": wa.NY,
               "domain_D": [wa.NX * wa.DX, wa.NY * wa.DX],
               "halo_cells": wa.HALO, "stride_D": wa.STRIDE * wa.DX,
               "rotor_D": wa.ROTOR_D, "rotor_cells": wa.ROTOR_CELLS,
               "rotors": {r.rotor_id: [r.x_plane, r.y_centre] for r in wa.ROTORS},
               "modes": {n: wa.modes_for(n) for n in (128, 96, 32)},
               "nu_ref": wa.NU_REF,
               "scaling": wa.scaling_report(),
           })}
    print("geometry:", json.dumps(art["geometry"], indent=2)[:1200])

    def persist():
        path = os.path.join(OUT, "w93.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_f(art), fh, indent=1)
        print(f"\nartifact -> {path}")

    try:
        steps = 20 if args.quick else args.steps
        cache = os.path.join(OUT, "state.npz")
        if args.reuse_state and os.path.isfile(cache):
            z = np.load(cache)
            state = (z["u"], z["v"])
            prev_state = (z["pu"], z["pv"])
            old = os.path.join(OUT, "w93.json")
            if os.path.isfile(old):
                with open(old, encoding="utf-8") as fh:
                    prev_art = json.load(fh)
                for k in ("power_poseidon", "power_reference", "consequence"):
                    if k in prev_art:
                        art[k] = prev_art[k]
            print(f"  reusing the marched state from {cache} "
                  f"({z['n_steps']} macro-steps)")
        else:
            art["power_poseidon"], runs = stage_power(steps, "poseidon")
            persist()
            state = runs["all"]["state"]
            prev_state = runs["all"]["prev"]
            np.savez_compressed(cache, u=state[0], v=state[1],
                                pu=prev_state[0], pv=prev_state[1],
                                n_steps=steps)

        experts_p = wa.make_experts(*state, "poseidon")
        experts_r = wa.make_experts(*state, "reference")
        graph_p, _ = wa.build(*state, kind="poseidon", experts=experts_p)
        graph_r, _ = wa.build(*state, kind="reference", experts=experts_r)

        art["disk_controls"] = _f(stage_disk_controls())
        persist()
        art["reach"] = _f(stage_reach(experts_p, experts_r))
        persist()
        art["seams"] = _f(stage_seams(graph_p, experts_p, graph_r, experts_r))
        persist()
        art["attribution"] = _f(stage_attribution(
            graph_p, experts_p, graph_r, experts_r, state, prev_state,
            quick=args.quick))
        persist()
        art["substitution"] = _f(stage_substitution(
            graph_p, experts_p, graph_r, experts_r, art["seams"]))
        persist()

        banner("8. the consequence, in the currency the case study is about")
        if not (args.reuse_state and "power_reference" in art):
            art["power_reference"], _ = stage_power(steps, "reference")
        pp, pr = art["power_poseidon"], art["power_reference"]
        print("\n  wake loss, the two experts, same array, same controls:")
        print(f"    {'turbine':8s} {'Poseidon':>10s} {'WindowNS':>10s} {'difference':>12s}")
        diffs = {}
        for rid in pp["turbines"]:
            a = pp["turbines"][rid]["wake_loss_pct"]
            b = pr["turbines"][rid]["wake_loss_pct"]
            diffs[rid] = a - b
            print(f"    {rid:8s} {a:9.2f}% {b:9.2f}% {a - b:11.2f} pp")
        ea, eb = pp["array_efficiency"], pr["array_efficiency"]
        print(f"    {'ARRAY':8s} {100*(1-ea):9.2f}% {100*(1-eb):9.2f}% "
              f"{100*((1-ea)-(1-eb)):11.2f} pp")
        art["consequence"] = {"wake_loss_difference_pp": diffs,
                              "array_loss_poseidon_pct": 100 * (1 - ea),
                              "array_loss_reference_pct": 100 * (1 - eb),
                              "array_loss_difference_pp": 100 * ((1 - ea) - (1 - eb))}
        persist()

        if not args.quick:
            art["compile"] = _f(stage_compile(state))
    finally:
        persist()
    return 0


if __name__ == "__main__":
    sys.exit(main())
