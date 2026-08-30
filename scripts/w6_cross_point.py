"""W6 -- the cross-point, treated rather than declared, on a real 4-agent junction.

The row has stood `in progress` since Tier 1 with a definition of done that is one
sentence: *"A written rule in `port-algebra-atlas-0.1`, plus **a test on a 4-agent
junction showing beta does not collapse**."*  The rule was written in the
2026-08-27 amendment and made conditional on the decomposition -- no action under
overlapping + partition of unity, primal cross-point degrees of freedom under
non-overlapping.  **The test was never run**, on any graph, and
[[tier0-measurements]] §7 and §9.8 both record the cross-point as *declared, not
treated*.

The junction is real and it has been sitting in the case study since §1: four
`reference.WindowNS` windows tiling one square **share their seam layer**, so all
four meet at one cell, and that cell belongs to all four seams at once.  §1 calls
it *"a genuine cross-point, which is declared rather than hidden"*.

What is measured
----------------

Per-seam ``beta`` is not the quantity the cross-point can damage: each seam's own
operator knows nothing about the others.  The **global** interface operator does
-- one matrix over all four seams' multipliers at once -- and that is what is
assembled here, by probing every global mode and reading the flux on every seam.

    beta_seam    min over the four seams of sigma_min(S_seam)          [the old number]
    beta_global  sigma_min of the assembled 4-seam operator            [the new one]

If the cross-point is a real difficulty, the second collapses relative to the
first and the collapse has a **structure**: the offending directions live at the
cross-point cell.  If it is not, they agree and the rule's "no action" branch is
right for a reason rather than by assertion.

Both decompositions are run, because the rule is conditional on exactly that:

    shared-layer (halo 1)   the four windows share the cross-point CELL
    overlapping (halo 21)   each window's artificial boundary is somewhere else,
                            and there is no common cell at all

Run:  python scripts/w6_cross_point.py [--out out/w6]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from atlas.cases import window_ns as W                            # noqa: E402

#: Modes per seam.  The declared interface space, as everywhere else.
M_SEAM = W.M_EFF

SEAM_ORDER = ("sx0", "sx1", "sy0", "sy1")


def _j(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def say(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------------------
# the junction geometry
# ---------------------------------------------------------------------------


def cross_point_cell(tiling):
    """The global cell where all four windows meet, if there is one.

    Under a shared-layer tiling the four windows' artificial faces coincide at a
    single cell and that cell is the cross-point.  Under a wider halo each
    window's artificial boundary is somewhere else and **there is no common
    cell** -- which is the geometric content of the rule's "no action under
    overlapping" branch, and the reason it is checked here rather than assumed.
    """
    o = W.MONO_N - tiling.n
    faces = {}
    for k, (ox, oy) in enumerate(tiling.offsets):
        faces[tiling.names[k]] = {
            "xhi": ox + tiling.n - 1, "xlo": ox,
            "yhi": oy + tiling.n - 1, "ylo": oy,
        }
    # the seam columns/rows each pair shares
    col_x = faces["W00"]["xhi"] if faces["W00"]["xhi"] == faces["W10"]["xlo"] else None
    row_y = faces["W00"]["yhi"] if faces["W00"]["yhi"] == faces["W01"]["ylo"] else None
    return (row_y, col_x) if (col_x is not None and row_y is not None) else None


def seam_index(tiling, seam):
    """(agent_a, face_a, agent_b, face_b) and the global cells of the seam ring."""
    (a, fa), (b, fb) = W.SEAMS[seam]
    ka = tiling.names.index(a)
    oxa, oya = tiling.offsets[ka]
    n = tiling.n
    if fa in ("xhi", "xlo"):
        col = oxa + (n - 1 if fa == "xhi" else 0)
        cells = [(oya + i, col) for i in range(n)]
    else:
        row = oya + (n - 1 if fa == "yhi" else 0)
        cells = [(row, oxa + i) for i in range(n)]
    return (a, fa, b, fb), cells


# ---------------------------------------------------------------------------
# the global operator
# ---------------------------------------------------------------------------


def respond_multi(agent, traces: dict) -> dict:
    """{face: trace} -> {face: flux}, in ONE solve. The off-diagonal probe.

    **This is the measurement `boundary_response` cannot express, and that is
    W6's real blocker.** The declared signature is
    ``Callable[[str, ndarray], ndarray]`` -- *"trace on V_i -> flux on V_i"* --
    so it perturbs one face and returns **that same face's** flux. A cross-point
    coupling is precisely the off-diagonal block
    ``d(flux on face B) / d(trace on face A)`` for two faces of ONE agent, and no
    sequence of single-port calls can produce it: each call restarts from the
    base state.

    So the vault has been declaring cross-points and refusing on them without
    ever being able to measure one. This function is the generalized probe,
    written in the harness rather than the package because adding it to the
    record is a `PortAmendment`-sized decision (**W64**), not a convenience.
    """
    r_u, r_v = agent.u0.copy(), agent.v0.copy()
    for face, trace in traces.items():
        ring, _interior = agent._ring_index(face)
        if W._FACE_GEOM[face][1] == "x":
            r_u[ring] = r_u[ring] + trace
        else:
            r_v[ring] = r_v[ring] + trace
    u1, v1 = agent._solver.step_batch(
        agent.u0[None], agent.v0[None], agent.dt,
        bc0=(r_u[None], r_v[None]), bc1=None)
    agent.n_calls += 1
    out = {}
    for face in traces:
        ring, interior = agent._ring_index(face)
        w = u1[0] if W._FACE_GEOM[face][1] == "x" else v1[0]
        out[face] = agent.nu * (w[ring] - w[interior]) / agent.h
    return out


def faces_of(tiling, agent_id):
    """Every seam face this agent carries, with the seam each belongs to."""
    out = {}
    for s in SEAM_ORDER:
        (a, fa), (b, fb) = W.SEAMS[s]
        if a == agent_id:
            out[fa] = s
        if b == agent_id:
            out[fb] = s
    return out


def assemble_global_multiport(experts, tiling, eps=1e-6, m_seam=M_SEAM):
    """The 4-seam operator WITH its off-diagonal blocks, via `respond_multi`.

    Column ``(s, k)`` perturbs mode ``k`` of seam ``s`` on both incident agents
    and reads the flux on **every** face of **every** agent -- so a trace on one
    seam can show up in another seam's residual through an agent they share.
    That path is the cross-point, and it is invisible to a per-seam probe.
    """
    B = W.fourier_basis(tiling.n, m_seam)
    dim = len(SEAM_ORDER) * m_seam
    agent_faces = {a: faces_of(tiling, a) for a in tiling.names}

    def flux_all(per_agent):
        """per_agent: {agent: {face: trace}}. Returns the stacked seam residuals."""
        got = {}
        for a, fmap in agent_faces.items():
            traces = {f: per_agent.get(a, {}).get(f, np.zeros(tiling.n))
                      for f in fmap}
            got[a] = respond_multi(experts[a], traces)
        out = []
        for s in SEAM_ORDER:
            (a, fa), (b, fb) = W.SEAMS[s]
            out.append(W.H * (B.T @ (got[a][fa] + got[b][fb])))
        return np.concatenate(out)

    zero = flux_all({})
    cols = []
    for s in SEAM_ORDER:
        (a, fa), (b, fb) = W.SEAMS[s]
        for k in range(m_seam):
            t = eps * B[:, k]
            pert = {a: {fa: t}, b: {fb: t}}
            cols.append((flux_all(pert) - zero) / eps)
    return np.column_stack(cols)


def block_structure(S, m_seam=M_SEAM):
    """How much of the global operator is OFF the seam-diagonal.

    Zero means the four seams do not see each other at all and a per-seam probe
    loses nothing; large means they do, and a per-seam probe is measuring the
    wrong operator.
    """
    n_s = len(SEAM_ORDER)
    diag = 0.0
    for i in range(n_s):
        sl = slice(i * m_seam, (i + 1) * m_seam)
        diag += float(np.linalg.norm(S[sl, sl])) ** 2
    total = float(np.linalg.norm(S)) ** 2
    off = max(total - diag, 0.0)
    return {"off_diagonal_fraction": float(np.sqrt(off / max(total, 1e-300))),
            "norm_total": float(np.sqrt(total)),
            "norm_off_diagonal": float(np.sqrt(off))}


def assemble_global(experts, tiling, eps=1e-6, m_seam=M_SEAM):
    """The 4-seam interface operator: perturb one global mode, read every seam.

    Column ``(s, k)`` is the flux response, on ALL four seams, to mode ``k`` of
    seam ``s``.  A per-seam probe never forms the off-diagonal blocks, and the
    off-diagonal blocks are the only place a cross-point can show up: two seams
    that share a cell respond to each other's traces through it.
    """
    B = W.fourier_basis(tiling.n, m_seam)
    n_s = len(SEAM_ORDER)
    dim = n_s * m_seam

    def flux_all(pert):
        """pert: {agent: (du, dv) on the window}. Returns the stacked seam flux."""
        out = []
        for s in SEAM_ORDER:
            (a, fa, b, fb), _cells = seam_index(tiling, s)
            fa_flux = experts[a].respond(f"{fa}:MECH", pert.get((a, fa),
                                                               np.zeros(tiling.n)))
            fb_flux = experts[b].respond(f"{fb}:MECH", pert.get((b, fb),
                                                               np.zeros(tiling.n)))
            out.append(B.T @ (fa_flux + fb_flux) * W.H)
        return np.concatenate(out)

    zero = flux_all({})
    cols = []
    for si, s in enumerate(SEAM_ORDER):
        (a, fa, b, fb), _ = seam_index(tiling, s)
        for k in range(m_seam):
            trace = eps * B[:, k]
            # The SAME trace is imposed on both sides of the seam: that is what
            # a single multiplier means, and it is what makes the assembled
            # operator the Steklov-Poincare sum rather than two unrelated blocks.
            cols.append((flux_all({(a, fa): trace, (b, fb): trace}) - zero) / eps)
    S = np.column_stack(cols)
    assert S.shape == (dim, dim)
    return S


def cross_point_modes(tiling, m_seam=M_SEAM):
    """The global directions that move the cross-point cell, if it exists.

    Each seam's trace at the cross-point is a linear functional of that seam's
    modes -- the row of the Fourier basis at the cell's index along the seam.
    Stacking the four gives a 4 x dim map ``V``: ``V @ lambda`` is the four
    values the four seams assign to **one physical cell**, which is exactly the
    thing a cross-point makes multi-valued.
    """
    cp = cross_point_cell(tiling)
    if cp is None:
        return None, None
    B = W.fourier_basis(tiling.n, m_seam)
    rows = []
    for si, s in enumerate(SEAM_ORDER):
        _ids, cells = seam_index(tiling, s)
        if cp not in cells:
            rows.append(np.zeros(len(SEAM_ORDER) * m_seam))
            continue
        j = cells.index(cp)
        r = np.zeros(len(SEAM_ORDER) * m_seam)
        r[si * m_seam:(si + 1) * m_seam] = B[j, :]
        rows.append(r)
    return cp, np.vstack(rows)


def multivaluedness(experts, tiling, u_ref, v_ref, m_seam=M_SEAM):
    """**The defect a cross-point actually causes, which is not a beta collapse.**

    Each seam carries its own multiplier -- its own ``m_seam`` Fourier modes over
    its own ring -- and the cross-point cell lies on all four of them.  So the
    assembled scheme assigns that ONE physical cell **four independent values**,
    and nothing in the assembly makes them agree.  That is what "declare primal
    (single-valued) corner degrees of freedom" removes, and it is measurable
    without solving anything: project the reference's own trace onto each seam's
    declared space, reconstruct, and read the four values off at the cell.

    They differ because each seam truncates a *different* function to 16 modes.
    The spread is the multi-valuedness the primal treatment exists to remove, and
    it is a property of the DECLARED interface space rather than of the expert --
    so it is the same for every expert on this tiling, and it does not go away by
    probing better.
    """
    cp = cross_point_cell(tiling)
    if cp is None:
        return None
    B = W.fourier_basis(tiling.n, m_seam)
    # **Grouped by component, and that is not a detail.** An x-normal seam
    # carries `u` and a y-normal one carries `v`; comparing across the two is
    # comparing different physical quantities and produces a spread of 200% that
    # means nothing. The cross-point question is whether the seams that carry the
    # SAME component agree about the same cell.
    groups = {"u": [], "v": []}
    for s in SEAM_ORDER:
        _ids, cells = seam_index(tiling, s)
        j = cells.index(cp)
        (a, fa), _b = W.SEAMS[s]
        axis = W._FACE_GEOM[fa][1]
        comp = u_ref if axis == "x" else v_ref
        raw = np.array([comp[r, c] for (r, c) in cells])
        coeff = np.linalg.lstsq(B, raw, rcond=None)[0]
        recon = B @ coeff
        groups["u" if axis == "x" else "v"].append(
            {"seam": s, "value": float(recon[j]),
             "exact": float(comp[cp]),
             "scale": float(np.linalg.norm(raw) / np.sqrt(len(raw)))})
    out = {"cross_point_cell": cp, "by_component": {}}
    worst = 0.0
    for comp, rows in groups.items():
        vals = np.array([r["value"] for r in rows])
        scale = float(np.mean([r["scale"] for r in rows])) or 1.0
        rel = float((vals.max() - vals.min()) / scale)
        worst = max(worst, rel)
        out["by_component"][comp] = {
            "seams": [r["seam"] for r in rows],
            "values": [r["value"] for r in rows],
            "exact": rows[0]["exact"],
            "spread_abs": float(vals.max() - vals.min()),
            "spread_relative": rel,
        }
    out["worst_relative_spread"] = worst
    out["note"] = ("seams sharing a component and a cell, disagreeing about its "
                   "value. A property of the DECLARED interface space -- each seam "
                   "truncates a different function to 16 modes -- not of the "
                   "expert, so no better probe removes it")
    return out


def primal_projector(V):
    """The primal treatment: force the cross-point to be SINGLE-VALUED.

    ``V @ lambda`` are the four values the four seams give one cell.  Requiring
    them equal is three independent constraints, spanned by the differences of
    consecutive rows of ``V``.  The primal multiplier space is the orthogonal
    complement of those, and this returns the projector onto it -- which is what
    "declare primal cross-point degrees of freedom" means as linear algebra.
    """
    D = np.vstack([V[i] - V[i + 1] for i in range(V.shape[0] - 1)])
    D = D[np.linalg.norm(D, axis=1) > 1e-14]
    if D.size == 0:
        return np.eye(V.shape[1]), 0
    Q, _r = np.linalg.qr(D.T)
    return np.eye(V.shape[1]) - Q @ Q.T, Q.shape[1]


# ---------------------------------------------------------------------------


def run(tiling, u, v, label, eps=1e-6):  # noqa: PLR0914
    experts = W.make_experts(u, v, tiling, expose_elliptic=True)
    cp, V = cross_point_modes(tiling)
    # The MULTI-PORT operator: the only assembly in which a cross-point can
    # appear at all, because the coupling lives in blocks `boundary_response`
    # cannot return. See `respond_multi`.
    S = assemble_global_multiport(experts, tiling)
    sv = np.linalg.svd(S, compute_uv=False)
    beta_global = float(sv[-1])
    blocks = block_structure(S)

    # per-seam beta, the number the vault has been quoting
    B = W.fourier_basis(tiling.n, M_SEAM)
    per = {}
    for s in SEAM_ORDER:
        (a, fa, b, fb), _ = seam_index(tiling, s)
        z = (experts[a].respond(f"{fa}:MECH", np.zeros(tiling.n))
             + experts[b].respond(f"{fb}:MECH", np.zeros(tiling.n)))
        cols = []
        for k in range(M_SEAM):
            t = eps * B[:, k]
            r = (experts[a].respond(f"{fa}:MECH", t)
                 + experts[b].respond(f"{fb}:MECH", t))
            cols.append(W.H * (B.T @ (r - z)) / eps)
        per[s] = float(np.linalg.svd(np.column_stack(cols), compute_uv=False)[-1])
    beta_seam = float(min(per.values()))

    out = {
        "label": label, "halo": tiling.halo, "n": tiling.n,
        "cross_point_cell": cp,
        "beta_per_seam": per,
        "beta_seam_min": beta_seam,
        "beta_global": beta_global,
        "collapse_factor": beta_seam / max(beta_global, 1e-300),
        "spectrum_tail": [float(x) for x in sv[-6:]],
        "block_structure": blocks,
    }
    say(f"  {label:26s} cross-point cell={cp}  beta_seam_min={beta_seam:.4e}  "
        f"beta_global={beta_global:.4e}  collapse={out['collapse_factor']:.2f}x")
    say(f"  {'':26s} off-diagonal fraction of the global operator: "
        f"{blocks['off_diagonal_fraction']:.4e}")

    if V is not None:
        P, n_constraints = primal_projector(V)
        # The primal space: restrict S to the complement of the disagreement
        # directions and re-measure. This is "declare primal cross-point DOFs".
        Q, _r = np.linalg.qr(P)
        keep = np.linalg.matrix_rank(P, tol=1e-10)
        basis = Q[:, :keep]
        S_primal = basis.T @ S @ basis
        sv_p = np.linalg.svd(S_primal, compute_uv=False)
        out["primal"] = {
            "n_constraints": int(n_constraints),
            "dim_before": int(S.shape[0]),
            "dim_after": int(keep),
            "beta_primal": float(sv_p[-1]),
            "recovery_factor": float(sv_p[-1] / max(beta_global, 1e-300)),
        }
        say(f"  {'':26s} primal: {S.shape[0]} -> {keep} dofs "
            f"({n_constraints} constraints), beta={sv_p[-1]:.4e}, "
            f"recovery={out['primal']['recovery_factor']:.3f}x")

        mv = multivaluedness(experts, tiling, u, v)
        out["multivaluedness"] = mv
        if mv is not None:
            for comp, g in mv["by_component"].items():
                say(f"  {'':26s} cross-point, component {comp}: seams "
                    f"{g['seams']} give {[f'{x:.5f}' for x in g['values']]}, "
                    f"exact {g['exact']:.5f}, spread "
                    f"{100 * g['spread_relative']:.3f}% of the trace scale")

        # how much of the smallest singular direction lives at the cross-point
        _u, _s, vt = np.linalg.svd(S)
        w = vt[-1]
        num = float(np.linalg.norm(V @ w))
        den = float(np.linalg.norm(V, ord=2) * np.linalg.norm(w))
        out["null_direction_at_cross_point"] = num / max(den, 1e-300)
        say(f"  {'':26s} the smallest singular direction's cross-point content: "
            f"{out['null_direction_at_cross_point']:.4f}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_ROOT, "out", "w6"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    d = np.load(os.path.join(_ROOT, "out", "tier0b", "s0_state.npz"))
    u, v = d["u"], d["v"]

    say("W6 -- the cross-point, on a real 4-agent junction")
    res = {"configurations": []}
    # shared layer: 2*128 - 255 = 1, so the four windows share ONE cell
    res["configurations"].append(
        run(W.Tiling(n=128, ramp=1), u, v, "shared-layer (halo 1)"))
    # the working overlapping tiling: no common cell at all
    res["configurations"].append(
        run(W.DEFAULT_TILING, u, v, "overlapping (halo 21)"))

    shared, over = res["configurations"]
    res["verdict"] = {
        "shared_layer_has_a_cross_point_cell": shared["cross_point_cell"] is not None,
        "overlapping_has_a_cross_point_cell": over["cross_point_cell"] is not None,
        "beta_collapses_on_shared_layer": shared["collapse_factor"] > 10.0,
        "beta_collapses_on_overlapping": over["collapse_factor"] > 10.0,
    }
    say("")
    for k, val in res["verdict"].items():
        say(f"  {k}: {val}")

    path = os.path.join(a.out, "w6.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=_j)
    say(f"wrote {path}")
    return res


if __name__ == "__main__":
    main()
