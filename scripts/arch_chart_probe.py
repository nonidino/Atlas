"""arch_chart_probe -- is the workbench layout's own coordinate pair a conformal chart?

The chart operator's version 0 ([[chart-operator-architecture]] section 2.2) claims,
as an [AI Inference] to be checked, that the layout's harmonic coordinates
(`atlas/workbench/layout.py`: *along*, 0 on the domain's start edge and 1 on its
end edge with no flux through the sides; *across*, 0 on one side and 1 on the
other with no flux through the ends) are, piece by piece, a conformal map onto
the unit square composed with an axis scaling.  If that holds, each piece's
chart has

    orthogonal gradients          grad(xi) . grad(eta) = 0,
    a constant length ratio       |grad eta| / |grad xi| = M_p, the piece's modulus,
    positive Jacobian             J = det DF > 0 (no fold),

and M_p should be close to 1 if the layout "keeps every modulus near one", which
version 0 also claimed.

**What is measured, per piece.**  The global along/across fields are solved by the
layout's own code (`layout.coordinates`), the pieces are cut by the layout's own
rule (`layout._cut`, equal cell counts), and each piece's chart is the global pair
rescaled affinely onto [0,1]^2.  Gradients are taken at grid vertices from the
four surrounding cells (the bilinear gradient), so the fold test is exact for the
dual mesh: a vertex folds when the image of its four cell centres is not a
positively oriented quadrilateral.

- J and its extremes, normalised by the piece's area so that the mean of 1/J over
  the piece is 1;
- the metric's anisotropy, the ratio of DF^{-1}'s singular values;
- the orthogonality defect |cos angle(grad xi, grad eta)|;
- the length ratio |grad eta| / |grad xi| and its spread across the piece;
- the conformal modulus four ways: (pred) the global modulus times the piece's
  extents, M * du / dv; (grad) the median length ratio; (res) the restricted
  coordinates' Dirichlet energies, sqrt(E_eta / E_xi); (own) an independent solve
  of the piece as its own quadrilateral, 1 / E_xi with xi harmonic on the piece,
  0 and 1 on its two cut sides, no flux elsewhere.  For a chart that is conformal
  up to scaling all four agree, and E_xi * E_eta = 1 (extremal-length duality).

**The cases.**  `s-channel` as built (4 along, 1 across) and three other counts;
`bend-3`'s quarter annulus cut by the layout (its example draws its windows by
hand, so the layout is applied to its domain here), where the exact answer is
known: along = angle / (pi/2), across = ln(r/r0) / ln(r1/r0), a conformal
log-polar map, with global modulus (pi/2) / ln(2.5) = 1.7143.

Headless: it imports the workbench's modules and starts no server.  It is not a
timing.  Writes ``out/arch/chart_probe.json``, ``out/arch/chart_probe.txt`` and
``out/arch/chart_fields_<case>.npz`` (the fields, for the proposal's figures).

    set PYTHONIOENCODING=utf-8
    python scripts/arch_chart_probe.py
"""

from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np                                                     # noqa: E402
import scipy.sparse as sp                                              # noqa: E402
import scipy.sparse.linalg as spla                                     # noqa: E402

from atlas.workbench import geometry as geo                            # noqa: E402
from atlas.workbench import layout                                     # noqa: E402
from atlas.workbench import spec as wspec                              # noqa: E402

OUT = os.path.join(HERE, "out", "arch")

#: (example, pieces along, pieces across, note)
CASES = [
    ("s-channel", 4, 1, "as built"),
    ("s-channel", 8, 1, "twice the pieces along"),
    ("s-channel", 4, 2, "cut across as well"),
    ("s-channel", 16, 2, "equal-modulus counts"),
    ("bend-3", 3, 1, "the layout applied to the drawn quarter ring"),
    ("bend-3", 3, 2, "cut across as well"),
]

BEND_EXACT_M = (math.pi / 2.0) / math.log(100.0 / 40.0)


# ---------------------------------------------------------------------------
# the domain's faces
# ---------------------------------------------------------------------------


def interior_faces(act: np.ndarray) -> np.ndarray:
    """Pairs (a, b) of flat indices of adjacent domain cells, each face once."""
    ny, nx = act.shape
    jj, ii = np.nonzero(act[:, :-1] & act[:, 1:])
    a1, b1 = jj * nx + ii, jj * nx + ii + 1
    jj, ii = np.nonzero(act[:-1, :] & act[1:, :])
    a2, b2 = jj * nx + ii, (jj + 1) * nx + ii
    return np.column_stack([np.concatenate([a1, a2]), np.concatenate([b1, b2])])


def sides(domain, start: str, end: str) -> tuple[list[str], list[str]]:
    """The edges on each side between the start and the end, as `layout` finds them."""
    ring = layout._ring(domain)
    s, t = ring.index(start), ring.index(end)
    n = len(ring)
    one = [ring[(s + k) % n] for k in range(1, (t - s) % n)]
    two = [ring[(t + k) % n] for k in range(1, (s - t) % n)]
    return one, two


# ---------------------------------------------------------------------------
# energies and the piece's own modulus
# ---------------------------------------------------------------------------


def energy(values, cells_in, faces, bnd_cell, bnd_val, bnd_on, outside_half=True):
    """The Dirichlet energy of a cell field over a cell set (unit spacing, 2-D, so it
    is dimensionless): interior faces (difference squared); faces to a domain cell
    outside the set, up to the face at the linear midpoint (difference squared over
    2) when ``outside_half``; boundary faces holding a value at half a cell,
    2 (value - held) squared."""
    a, b = faces[:, 0], faces[:, 1]
    ia, ib = cells_in[a], cells_in[b]
    d2 = (values[a] - values[b]) ** 2
    e = float(np.sum(d2[ia & ib]))
    if outside_half:
        e += 0.5 * float(np.sum(d2[ia ^ ib]))
    sel = bnd_on & cells_in[bnd_cell]
    e += 2.0 * float(np.sum((values[bnd_cell[sel]] - bnd_val[sel]) ** 2))
    return e


def own_modulus(n_cells_total, piece, faces, zero_faces, one_faces, zero_bnd, one_bnd):
    """Solve the piece's own quadrilateral problem: harmonic on the piece's cells, 0 on
    ``zero_*`` faces and 1 on ``one_*`` faces (each at half a cell), no flux on every
    other face.  Returns its energy E; the piece's modulus between those two sides is
    1 / E."""
    cells = np.flatnonzero(piece)
    pos = np.full(n_cells_total, -1, dtype=np.int64)
    pos[cells] = np.arange(cells.size)
    n = cells.size
    a, b = faces[:, 0], faces[:, 1]
    both = piece[a] & piece[b]
    pa, pb = pos[a[both]], pos[b[both]]
    rows = np.concatenate([pa, pb, pa, pb])
    cols = np.concatenate([pa, pb, pb, pa])
    vals = np.concatenate([np.ones(pa.size), np.ones(pa.size), -np.ones(pa.size),
                           -np.ones(pa.size)])
    diag = np.zeros(n)
    rhs = np.zeros(n)
    for cell_list, g in ((zero_faces, 0.0), (one_faces, 1.0), (zero_bnd, 0.0), (one_bnd, 1.0)):
        if cell_list.size:
            np.add.at(diag, pos[cell_list], 2.0)
            np.add.at(rhs, pos[cell_list], 2.0 * g)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n)) + sp.diags(diag)
    x = spla.spsolve(A.tocsc(), rhs)
    e = float(np.sum((x[pa] - x[pb]) ** 2))
    for cell_list, g in ((zero_faces, 0.0), (one_faces, 1.0), (zero_bnd, 0.0), (one_bnd, 1.0)):
        if cell_list.size:
            e += 2.0 * float(np.sum((x[pos[cell_list]] - g) ** 2))
    return e


# ---------------------------------------------------------------------------
# vertex gradients
# ---------------------------------------------------------------------------


def vertex_gradients(U: np.ndarray, V: np.ndarray, act: np.ndarray, dx: float):
    """At every grid vertex whose four cells are in the domain: (j, i) of the vertex,
    the bilinear gradients of U and V there, the four cells' flat indices, and the
    fold indicator of the image of the four cell centres."""
    ny, nx = act.shape
    j, i = np.meshgrid(np.arange(1, ny), np.arange(1, nx), indexing="ij")
    sw, se = (j - 1, i - 1), (j - 1, i)
    nw, ne = (j, i - 1), (j, i)
    ok = act[sw] & act[se] & act[nw] & act[ne]
    j, i = j[ok], i[ok]
    sw, se, nw, ne = (j - 1, i - 1), (j - 1, i), (j, i - 1), (j, i)

    def grad(F):
        fx = ((F[se] + F[ne]) - (F[sw] + F[nw])) / (2.0 * dx)
        fy = ((F[nw] + F[ne]) - (F[sw] + F[se])) / (2.0 * dx)
        return fx, fy
    ux, uy = grad(U)
    vx, vy = grad(V)
    # the image quadrilateral sw -> se -> ne -> nw, its four corner cross products
    P = [np.column_stack([U[c], V[c]]) for c in (sw, se, ne, nw)]
    cross = []
    for k in range(4):
        p0, p1, p2 = P[k - 1], P[k], P[(k + 1) % 4]
        e1, e2 = p1 - p0, p2 - p1
        cross.append(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    cross = np.column_stack(cross)
    flat = np.column_stack([sw[0] * nx + sw[1], se[0] * nx + se[1], nw[0] * nx + nw[1],
                            ne[0] * nx + ne[1]])
    return j, i, ux, uy, vx, vy, flat, cross


def singular_ratio(a, b, c, d):
    """For 2 x 2 matrices [[a, b], [c, d]], s_max / s_min."""
    f2 = a * a + b * b + c * c + d * d
    det = np.abs(a * d - b * c)
    disc = np.sqrt(np.maximum(f2 * f2 - 4.0 * det * det, 0.0))
    s1 = np.sqrt(0.5 * (f2 + disc))
    s2 = np.sqrt(np.maximum(0.5 * (f2 - disc), 1e-300))
    return s1 / s2


def q(x, p):
    return float(np.percentile(x, p)) if x.size else float("nan")


# ---------------------------------------------------------------------------
# the corners of the whole domain
# ---------------------------------------------------------------------------


#: the corner angle is read over this many cells of each edge: the grid cannot see a
#: drawn spline's direction below a cell, and a Catmull-Rom end can turn within one
#: (the s-channel's walls leave their end points heading about 18 degrees below the
#: direction they take a few cells on)
CHORD_CELLS = 5.0


def _chord(poly: np.ndarray, from_start: bool) -> np.ndarray:
    """The direction from an end of a polyline to its point CHORD_CELLS along it."""
    p = poly if from_start else poly[::-1]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    k = int(min(np.searchsorted(s, CHORD_CELLS), len(p) - 1))
    return p[k] - p[0]


def corner_angles(domain, start, end, one, two):
    """Interior angle, in degrees, at each corner where an end edge meets a side edge,
    from the drawn edges' polylines, read over CHORD_CELLS cells."""
    polys = dict(geo.drawn_edges(domain))
    act = geo.domain_mask(domain)
    out = []
    for e_end in (start, end):
        for e_side in one + two:
            pa, pb = np.asarray(polys[e_end]), np.asarray(polys[e_side])
            for ka, kb in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
                if np.hypot(*(pa[ka] - pb[kb])) < 1e-6:
                    corner = pa[ka]
                    ta = _chord(pa, ka == 0)
                    tb = _chord(pb, kb == 0)
                    ang = math.degrees(math.acos(float(np.clip(
                        np.dot(ta, tb) / (np.linalg.norm(ta) * np.linalg.norm(tb)), -1, 1))))
                    # interior or exterior?  test the bisector's point
                    bis = ta / np.linalg.norm(ta) + tb / np.linalg.norm(tb)
                    p = corner + 3.0 * bis / max(np.linalg.norm(bis), 1e-12)
                    ii, jj = int(p[0]), int(p[1])
                    inside = 0 <= jj < act.shape[0] and 0 <= ii < act.shape[1] and act[jj, ii]
                    interior = ang if inside else 360.0 - ang
                    # the conformal chart near a corner of interior angle alpha:
                    # J ~ r^(2 - pi/alpha)
                    alpha = math.radians(interior)
                    out.append({"end_edge": e_end, "side_edge": e_side,
                                "at_cells": [float(corner[0]), float(corner[1])],
                                "interior_angle_deg": round(interior, 2),
                                "conformal_J_exponent": round(2.0 - math.pi / alpha, 4)})
    return out


# ---------------------------------------------------------------------------
# one case
# ---------------------------------------------------------------------------


def run_case(key: str, n_along: int, n_across: int, note: str) -> dict:
    spec = wspec.example_case(key)
    d = spec.domain
    act = np.asarray(geo.domain_mask(d))
    ny, nx, dx = d.ny, d.nx, float(d.dx)
    along, across, start, end = layout.coordinates(d, across=True)
    one, two = sides(d, start, end)

    # the pieces, by the layout's own rule, and a check against layout.pieces
    cells = np.flatnonzero(act.ravel())
    cols = layout._cut(cells, along.ravel()[cells], n_along)
    col = np.full(act.size, -1)
    row = np.full(act.size, -1)
    for c, cc in enumerate(cols):
        col[cc] = c
        rows = layout._cut(cc, across.ravel()[cc], n_across)
        for r, rr in enumerate(rows):
            row[rr] = r
    spec.layout = wspec.Layout(cut="along", along=n_along, across=n_across)
    check = layout.pieces(spec)
    pid = col * n_across + row
    pid[~act.ravel()] = -1
    for k, (_name, m) in enumerate(check):
        assert np.array_equal(np.flatnonzero(m.ravel()), np.flatnonzero(pid == k)), \
            "the probe's pieces differ from layout.pieces"

    faces = interior_faces(act)
    bf = geo.boundary_faces(d)
    bnd_cell, bnd_edge = bf.cell, bf.edge.astype(str)
    U, V = along.ravel(), across.ravel()

    # global: the domain's modulus, two ways, and the duality product
    u_on = np.isin(bnd_edge, [start, end])
    u_val = np.where(bnd_edge == end, 1.0, 0.0)
    v_on = np.isin(bnd_edge, one + two)
    v_val = np.where(np.isin(bnd_edge, two), 1.0, 0.0)
    everywhere = act.ravel().copy()
    E_u = energy(U, everywhere, faces, bnd_cell, u_val, u_on)
    E_v = energy(V, everywhere, faces, bnd_cell, v_val, v_on)
    M_total = 1.0 / E_u

    # vertex gradients of the global pair
    j, i, ux, uy, vx, vy, flat, cross = vertex_gradients(along, across, act, dx)
    det_uv = ux * vy - uy * vx
    sgn = float(np.sign(np.median(det_uv)))
    fold = np.any(cross * sgn <= 0.0, axis=1)
    gu, gv = np.hypot(ux, uy), np.hypot(vx, vy)
    cos_uv = (ux * vx + uy * vy) / np.maximum(gu * gv, 1e-300)
    ratio_uv = gv / np.maximum(gu, 1e-300)
    has_bnd = np.zeros(act.size, dtype=bool)
    has_bnd[bnd_cell] = True
    wall = np.any(has_bnd[flat], axis=1)                   # a cell of the vertex has a wall
    vp = pid[flat]
    clean_piece = np.all(vp == vp[:, :1], axis=1)

    glob = {
        "cells": int(act.sum()), "grid": [nx, ny], "dx_m": dx,
        "start_edge": start, "end_edge": end, "side_one": one, "side_two": two,
        "modulus_from_along_energy_1_over_E_u": M_total,
        "modulus_from_across_energy_E_v": E_v,
        "duality_product_E_u_E_v": E_u * E_v,
        "vertices": int(j.size), "vertices_folded": int(fold.sum()),
        "vertices_det_nonpositive": int(np.sum(det_uv * sgn <= 0)),
        "abs_cos_median": q(np.abs(cos_uv), 50), "abs_cos_p95": q(np.abs(cos_uv), 95),
        "abs_cos_max": float(np.max(np.abs(cos_uv))),
        "abs_cos_p95_away_from_walls": q(np.abs(cos_uv[~wall]), 95),
        "length_ratio_over_M_p5": q(ratio_uv / M_total, 5),
        "length_ratio_over_M_median": q(ratio_uv / M_total, 50),
        "length_ratio_over_M_p95": q(ratio_uv / M_total, 95),
        "length_ratio_over_M_p5_away_from_walls": q(ratio_uv[~wall] / M_total, 5),
        "length_ratio_over_M_p95_away_from_walls": q(ratio_uv[~wall] / M_total, 95),
        "corners": corner_angles(d, start, end, one, two),
    }
    if key == "bend-3":
        glob["exact_modulus"] = BEND_EXACT_M

    # the pieces
    per = []
    for c in range(n_along):
        in_col = col == c
        u_lo = 0.0 if c == 0 else 0.5 * (U[col == c - 1].max() + U[in_col].min())
        u_hi = 1.0 if c == n_along - 1 else 0.5 * (U[in_col].max() + U[col == c + 1].min())
        for r in range(n_across):
            k = c * n_across + r
            piece = pid == k
            in_r = in_col & (row == r)
            v_lo = 0.0 if r == 0 else 0.5 * (V[in_col & (row == r - 1)].max() + V[in_r].min())
            v_hi = 1.0 if r == n_across - 1 else \
                0.5 * (V[in_r].max() + V[in_col & (row == r + 1)].min())
            du, dv = u_hi - u_lo, v_hi - v_lo
            area = float(piece.sum()) * dx * dx

            # the chart's derivative at the piece's own vertices
            sel = clean_piece & (vp[:, 0] == k)
            a_, b_, c_, d_ = ux[sel] / du, uy[sel] / du, vx[sel] / dv, vy[sel] / dv
            det = (a_ * d_ - b_ * c_) * sgn
            J_hat = 1.0 / (det * area) if det.size else det       # mean of 1/J_hat is 1
            aniso = singular_ratio(a_, b_, c_, d_)
            lr = np.hypot(c_, d_) / np.maximum(np.hypot(a_, b_), 1e-300)
            cs = np.abs(cos_uv[sel])
            w = wall[sel]
            ok = det > 0
            # the four moduli
            xi = (U - u_lo) / du
            eta = (V - v_lo) / dv
            xi_on = u_on & (col[bnd_cell] == c)
            xi_val = (u_val - u_lo) / du
            eta_on = v_on & (pid[bnd_cell] == k)
            eta_val = (v_val - v_lo) / dv
            E_xi = energy(xi, piece, faces, bnd_cell, xi_val, xi_on)
            E_eta = energy(eta, piece, faces, bnd_cell, eta_val, eta_on)
            # own solve: side A = faces to column c-1 (or the start edge), side C =
            # faces to column c+1 (or the end edge); for eta, faces to row r-1 / r+1
            # of this column (or side one / side two)
            fa, fb = faces[:, 0], faces[:, 1]

            def faces_to(pred_other):
                x1 = fa[piece[fa] & pred_other(fb)]
                x2 = fb[piece[fb] & pred_other(fa)]
                return np.concatenate([x1, x2])
            z_f = faces_to(lambda o: col[o] == c - 1)
            o_f = faces_to(lambda o: col[o] == c + 1)
            z_b = bnd_cell[(bnd_edge == start) & piece[bnd_cell]]
            o_b = bnd_cell[(bnd_edge == end) & piece[bnd_cell]]
            E_xi_own = own_modulus(act.size, piece, faces, z_f, o_f, z_b, o_b)
            z_f = faces_to(lambda o: (col[o] == c) & (row[o] == r - 1))
            o_f = faces_to(lambda o: (col[o] == c) & (row[o] == r + 1))
            z_b = bnd_cell[np.isin(bnd_edge, one) & piece[bnd_cell]]
            o_b = bnd_cell[np.isin(bnd_edge, two) & piece[bnd_cell]]
            E_eta_own = own_modulus(act.size, piece, faces, z_f, o_f, z_b, o_b)
            # where J is least
            if ok.any():
                kk = int(np.argmin(np.where(ok, J_hat, np.inf)))
                where_min = [float(i[sel][kk]), float(j[sel][kk])]
                min_on_wall = bool(w[kk])
            else:
                where_min, min_on_wall = [float("nan")] * 2, False
            per.append({
                "piece": f"F{c}{r}", "cells": int(piece.sum()),
                "along_range": [u_lo, u_hi], "across_range": [v_lo, v_hi],
                "vertices": int(sel.sum()), "vertices_away_from_walls": int((~w).sum()),
                "folds": int(np.sum(fold[sel])), "det_nonpositive": int(np.sum(~ok)),
                "J_hat_min": q(J_hat[ok], 0), "J_hat_p1": q(J_hat[ok], 1),
                "J_hat_max": q(J_hat[ok], 100),
                "J_ratio_max_over_min": q(J_hat[ok], 100) / max(q(J_hat[ok], 0), 1e-300),
                "J_ratio_p99_over_p1": q(J_hat[ok], 99) / max(q(J_hat[ok], 1), 1e-300),
                "J_hat_min_away_from_walls": q(J_hat[ok & ~w], 0),
                "J_ratio_away_from_walls": q(J_hat[ok & ~w], 100) / max(q(J_hat[ok & ~w], 0),
                                                                         1e-300),
                "J_min_at_cells": where_min, "J_min_on_wall_vertex": min_on_wall,
                "anisotropy_median": q(aniso, 50), "anisotropy_p95": q(aniso, 95),
                "anisotropy_max": q(aniso, 100),
                "abs_cos_median": q(cs, 50), "abs_cos_p95": q(cs, 95), "abs_cos_max": q(cs, 100),
                "abs_cos_p95_away_from_walls": q(cs[~w], 95),
                "length_ratio_p5": q(lr, 5), "length_ratio_median": q(lr, 50),
                "length_ratio_p95": q(lr, 95),
                "length_ratio_rel_spread_p5_p95": (q(lr, 95) - q(lr, 5)) / max(q(lr, 50), 1e-300),
                "length_ratio_rel_spread_away_from_walls":
                    (q(lr[~w], 95) - q(lr[~w], 5)) / max(q(lr[~w], 50), 1e-300),
                "modulus_pred_M_du_over_dv": M_total * du / dv,
                "modulus_grad_median_ratio": q(lr, 50),
                "modulus_res_sqrt_Eeta_over_Exi": math.sqrt(E_eta / E_xi),
                "duality_res_Exi_Eeta": E_xi * E_eta,
                "modulus_own_1_over_Exi": 1.0 / E_xi_own,
                "modulus_own_Eeta": E_eta_own,
                "duality_own_Exi_Eeta": E_xi_own * E_eta_own,
            })

    os.makedirs(OUT, exist_ok=True)
    tag = f"{key}_{n_along}x{n_across}"
    np.savez_compressed(os.path.join(OUT, f"chart_fields_{tag}.npz"), along=along,
                        across=across, piece=pid.reshape(act.shape), act=act, dx=dx,
                        vj=j, vi=i, det_uv=det_uv, cos_uv=cos_uv, ratio_uv=ratio_uv,
                        fold=fold, wall=wall)
    return {"case": key, "along": n_along, "across": n_across, "note": note,
            "global": glob, "pieces": per}


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def text(results: list[dict]) -> str:
    L = []
    w = L.append
    w("arch_chart_probe -- the layout's along/across pair as a chart, piece by piece")
    w("=" * 78)
    for res in results:
        g = res["global"]
        w("")
        w(f"{res['case']}  {res['along']} along x {res['across']} across  ({res['note']})")
        w(f"  cells {g['cells']}, start {g['start_edge']}, end {g['end_edge']}")
        line = (f"  global modulus: 1/E_u = {g['modulus_from_along_energy_1_over_E_u']:.4f}, "
                f"E_v = {g['modulus_from_across_energy_E_v']:.4f}, "
                f"E_u E_v = {g['duality_product_E_u_E_v']:.4f}")
        if "exact_modulus" in g:
            line += f"  (exact {g['exact_modulus']:.4f})"
        w(line)
        w(f"  vertices {g['vertices']}, folded {g['vertices_folded']}, "
          f"det<=0 {g['vertices_det_nonpositive']}")
        w(f"  |cos| median {g['abs_cos_median']:.2e}, p95 {g['abs_cos_p95']:.2e}, "
          f"max {g['abs_cos_max']:.3f}; p95 away from walls "
          f"{g['abs_cos_p95_away_from_walls']:.2e}")
        w(f"  |grad v|/|grad u| / M: p5 {g['length_ratio_over_M_p5']:.3f}, median "
          f"{g['length_ratio_over_M_median']:.3f}, p95 {g['length_ratio_over_M_p95']:.3f}; "
          f"away from walls {g['length_ratio_over_M_p5_away_from_walls']:.3f}"
          f"-{g['length_ratio_over_M_p95_away_from_walls']:.3f}")
        for cn in g["corners"]:
            w(f"  corner {cn['end_edge']}/{cn['side_edge']}: interior angle "
              f"{cn['interior_angle_deg']:.1f} deg, conformal J ~ r^"
              f"{cn['conformal_J_exponent']:+.3f}")
        w("  piece  cells  M_pred  M_grad  M_res  M_own  E.E(own)  J/|A| min  "
          "Jmax/Jmin  (no wall)  aniso med/max   |cos| p95  lr spread  folds")
        for p in res["pieces"]:
            w(f"  {p['piece']:<5} {p['cells']:>5}  {p['modulus_pred_M_du_over_dv']:6.3f}  "
              f"{p['modulus_grad_median_ratio']:6.3f} {p['modulus_res_sqrt_Eeta_over_Exi']:6.3f} "
              f"{p['modulus_own_1_over_Exi']:6.3f}  {p['duality_own_Exi_Eeta']:7.4f}  "
              f"{p['J_hat_min']:9.3e}  {p['J_ratio_max_over_min']:9.2f}  "
              f"({p['J_ratio_away_from_walls']:6.2f})  "
              f"{p['anisotropy_median']:5.2f}/{p['anisotropy_max']:6.2f}  "
              f"{p['abs_cos_p95']:9.2e}  {p['length_ratio_rel_spread_p5_p95']:8.3f}  "
              f"{p['folds']:>4}")
    return "\n".join(L) + "\n"


def main() -> None:
    results = [run_case(*c) for c in CASES]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "chart_probe.json"), "w", encoding="utf-8") as fh:
        json.dump({"probe": "arch_chart_probe", "results": results}, fh, indent=1)
    t = text(results)
    with open(os.path.join(OUT, "chart_probe.txt"), "w", encoding="utf-8") as fh:
        fh.write(t)
    sys.stdout.write(t)


if __name__ == "__main__":
    main()
