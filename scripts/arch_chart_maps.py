"""arch_chart_maps -- which chart map for a piece: conformal, Winslow, or transfinite?

Extends `arch_chart_probe.py` (same cases, same layout coordinates) for decision (b) of
the architecture conversation.  Two questions.

**1. Equal-value cuts.**  The first probe cut the layout's pieces at equal cell counts
and found moduli from 0.6 to 6.3.  Its [AI Inference]: because the along/across pair is
conformal up to an axis scaling, cutting at equal *values* -- along in [c/n, (c+1)/n),
across in [r/m, (r+1)/m) -- gives every piece the modulus M * (1/n) / (1/m).  Choosing
n = round(M m) should give every piece a modulus near one.  Measured here: each piece's
predicted modulus M du / dv against its own independent solve (harmonic on the piece,
0 and 1 on its two cut sides, no flux on the others), and its chart's J and metric.

**2. One piece, three charts.**  Every harmonic chart solves the same equation -- the
two reference coordinates are harmonic on the piece -- and they differ only in the
**boundary correspondence**:

- *conformal* (the layout's own pair): xi fixed on two opposite sides, no flux on the
  other two, and the reverse for eta.  The sides slide freely, so the map is conformal,
  and near a corner of interior angle alpha its Jacobian goes like r^(pi/alpha - 2) in
  the reference-to-physical direction: singular unless alpha = 90 degrees;
- *Winslow* (harmonic, arc length): xi and eta fixed on all four sides, each side
  parametrised by its normalised arc length.  Harmonic onto the convex square, so
  injective in the continuum (Rado-Kneser-Choquet).  Near a straight corner the
  harmonic extension of linear data is linear, so the map is locally affine: J is
  bounded and positive at any corner angle below 180 degrees;
- *transfinite* (Gordon-Hall / Coons): the explicit blend of the same four
  arc-length-parametrised side curves.  No solve, no injectivity guarantee: checked by
  min J > 0.

Measured on three pieces: the s-channel's end piece (the 68 and 148 degree corners), one
of its middle pieces, and a bend-3 piece (right-angle corners), all from the equal-value
cut.  For each chart: J (normalised by the piece's area, so its reference-space mean is
one), max/min and p99/p1 of J, the anisotropy of DF, folds, and the **metric the operator
would receive**, K_hat = J DF^-1 DF^-T (dimensionless in 2-D; a constant diag(1/M, M)
for a conformal chart of modulus M).  The conformal and Winslow charts are inverse maps
solved on the piece's cells, read at grid vertices as in the first probe; the
transfinite chart is a forward map evaluated on a 96 x 96 reference grid.  Extremes are
comparable; distributions are sampled in different measures (physical vs reference).

Headless: imports the workbench's modules and the first probe, starts no server, is not a
timing.  Writes ``out/arch/chart_maps.json``, ``out/arch/chart_maps.txt`` and
``out/arch/chart_maps_fields.npz`` (the three charts of the s-channel end piece and the
equal-value pieces, for the proposal's figures).

    set PYTHONIOENCODING=utf-8
    python scripts/arch_chart_maps.py
"""

from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

import numpy as np                                                     # noqa: E402
import scipy.sparse as sp                                              # noqa: E402
import scipy.sparse.linalg as spla                                     # noqa: E402

import arch_chart_probe as P                                           # noqa: E402
from atlas.workbench import geometry as geo                            # noqa: E402
from atlas.workbench import layout                                     # noqa: E402
from atlas.workbench import spec as wspec                              # noqa: E402

OUT = os.path.join(HERE, "out", "arch")

#: (example, target modulus per piece, pieces across)
EQUAL_VALUE = [
    ("s-channel", 1.0, 1),
    ("s-channel", 1.0, 2),
    ("bend-3", 1.0, 1),
    ("bend-3", 1.0, 3),
]

#: pieces compared under three charts: (example, pieces along, across, piece, label)
COMPARE = [
    ("s-channel", 9, 1, (0, 0), "s-channel end piece (68 and 148 degree corners)"),
    ("s-channel", 9, 1, (4, 0), "s-channel middle piece"),
    ("bend-3", 2, 1, (0, 0), "bend-3 piece (right-angle corners)"),
]

N_REF = 96


def q(x, p):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, p)) if x.size else float("nan")


# ---------------------------------------------------------------------------
# geometry: the drawn sides as polylines, and arc length along them
# ---------------------------------------------------------------------------


def chain(edges: dict, names: list[str]) -> np.ndarray:
    """The drawn edges ``names``, joined end to end into one polyline."""
    pts = np.asarray(edges[names[0]], dtype=float)
    for k, name in enumerate(names[1:]):
        nxt = np.asarray(edges[name], dtype=float)
        if k == 0:
            ends_ = [np.hypot(*(pts[-1] - nxt[0])), np.hypot(*(pts[-1] - nxt[-1])),
                     np.hypot(*(pts[0] - nxt[0])), np.hypot(*(pts[0] - nxt[-1]))]
            if min(ends_) == ends_[2] or min(ends_) == ends_[3]:
                pts = pts[::-1]
        if np.hypot(*(pts[-1] - nxt[0])) <= np.hypot(*(pts[-1] - nxt[-1])):
            pts = np.vstack([pts, nxt[1:]])
        else:
            pts = np.vstack([pts, nxt[::-1][1:]])
    return pts


def arclength(poly: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(poly, axis=0).T))])


def project(poly: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Arc-length parameter along ``poly`` of each point's nearest point, and the distance."""
    s = arclength(poly)
    a = poly[:-1]
    ab = poly[1:] - a
    L2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300)
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    out_s = np.empty(len(pts))
    out_d = np.empty(len(pts))
    for k0 in range(0, len(pts), 256):
        p = pts[k0:k0 + 256]
        ap = p[:, None, :] - a[None, :, :]
        t = np.clip(np.einsum("pij,ij->pi", ap, ab) / L2[None, :], 0.0, 1.0)
        near = a[None, :, :] + t[:, :, None] * ab[None, :, :]
        dist = np.hypot(p[:, None, 0] - near[:, :, 0], p[:, None, 1] - near[:, :, 1])
        m = np.argmin(dist, axis=1)
        out_s[k0:k0 + 256] = s[m] + t[np.arange(len(p)), m] * np.sqrt(L2[m])
        out_d[k0:k0 + 256] = dist[np.arange(len(p)), m]
    return out_s, out_d


def at_arclength(poly: np.ndarray, s_query: np.ndarray) -> np.ndarray:
    s = arclength(poly)
    return np.column_stack([np.interp(s_query, s, poly[:, 0]), np.interp(s_query, s, poly[:, 1])])


def sub_polyline(poly: np.ndarray, s0: float, s1: float, n: int = 400) -> np.ndarray:
    return at_arclength(poly, np.linspace(s0, s1, n))


# ---------------------------------------------------------------------------
# the domain, its coordinates, and the sides oriented from the start edge
# ---------------------------------------------------------------------------


class Domain:
    def __init__(self, key: str):
        self.key = key
        spec = wspec.example_case(key)
        self.d = d = spec.domain
        self.act = np.asarray(geo.domain_mask(d))
        self.ny, self.nx, self.dx = d.ny, d.nx, float(d.dx)
        self.along, self.across, self.start, self.end = layout.coordinates(d, across=True)
        self.one, self.two = P.sides(d, self.start, self.end)
        self.faces = P.interior_faces(self.act)
        bf = geo.boundary_faces(d)
        self.bnd_cell = bf.cell
        self.bnd_edge = bf.edge.astype(str)
        dirs = np.asarray(geo.DIRECTIONS, dtype=float)
        ci, cj = self.bnd_cell % self.nx, self.bnd_cell // self.nx
        self.bnd_mid = np.column_stack([ci + 0.5, cj + 0.5]) + 0.5 * dirs[bf.direction]
        edges = dict(geo.drawn_edges(d))
        start_poly = np.asarray(edges[self.start], dtype=float)
        side1 = chain(edges, self.one)
        side2 = chain(edges, self.two)
        # orient each side from the start edge towards the end edge
        for name in ("side1", "side2"):
            poly = side1 if name == "side1" else side2
            d0 = min(np.hypot(*(poly[0] - start_poly[0])), np.hypot(*(poly[0] - start_poly[-1])))
            d1 = min(np.hypot(*(poly[-1] - start_poly[0])),
                     np.hypot(*(poly[-1] - start_poly[-1])))
            if d1 < d0:
                poly = poly[::-1]
            if name == "side1":
                side1 = poly
            else:
                side2 = poly
        # the start edge from side one's corner to side two's
        if np.hypot(*(start_poly[0] - side1[0])) > np.hypot(*(start_poly[-1] - side1[0])):
            start_poly = start_poly[::-1]
        self.side1, self.side2, self.start_poly = side1, side2, start_poly
        U = self.along.ravel()
        V = self.across.ravel()
        self.U, self.V = U, V
        self.M = None

    def modulus(self) -> float:
        if self.M is None:
            u_on = np.isin(self.bnd_edge, [self.start, self.end])
            u_val = np.where(self.bnd_edge == self.end, 1.0, 0.0)
            self.M = 1.0 / P.energy(self.U, self.act.ravel().copy(), self.faces,
                                    self.bnd_cell, u_val, u_on)
        return self.M

    def wall_s_at_along(self, side: int, u_value: float) -> float:
        """Arc length along a side where the along coordinate reaches ``u_value``."""
        names = self.one if side == 1 else self.two
        poly = self.side1 if side == 1 else self.side2
        on = np.isin(self.bnd_edge, names)
        s, _ = project(poly, self.bnd_mid[on])
        u = self.U[self.bnd_cell[on]]
        order = np.argsort(s)
        s, u = s[order], u[order]
        u_run = np.maximum.accumulate(u)
        return float(np.interp(u_value, u_run, s))


# ---------------------------------------------------------------------------
# 1. equal-value cuts
# ---------------------------------------------------------------------------


def piece_ids(D: Domain, n_along: int, n_across: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    act = D.act.ravel()
    col = np.full(act.size, -1)
    row = np.full(act.size, -1)
    col[act] = np.clip(np.floor(D.U[act] * n_along), 0, n_along - 1).astype(int)
    row[act] = np.clip(np.floor(D.V[act] * n_across), 0, n_across - 1).astype(int)
    pid = np.where(act, col * n_across + row, -1)
    return col, row, pid


def conformal_metrics(D: Domain, n_along, n_across, col, row, pid, c, r, grads):
    """The layout's chart on piece (c, r), its own modulus, and its metric."""
    j, i, ux, uy, vx, vy, flat, cross = grads
    k = c * n_across + r
    piece = pid == k
    du, dv = 1.0 / n_along, 1.0 / n_across
    vp = pid[flat]
    sel = np.all(vp == k, axis=1)
    a_, b_, c_, d_ = ux[sel] / du, uy[sel] / du, vx[sel] / dv, vy[sel] / dv
    det = a_ * d_ - b_ * c_
    sgn = float(np.sign(np.median(det))) if det.size else 1.0
    det = det * sgn
    area = float(piece.sum()) * D.dx * D.dx
    J = 1.0 / (det * area)                          # physical area per unit reference area
    has_bnd = np.zeros(D.act.size, dtype=bool)
    has_bnd[D.bnd_cell] = True
    wall = np.any(has_bnd[flat[sel]], axis=1)
    Jw = J * area                                   # un-normalised, for K_hat
    K11 = Jw * (a_ * a_ + b_ * b_)
    K22 = Jw * (c_ * c_ + d_ * d_)
    K12 = Jw * (a_ * c_ + b_ * d_)
    aniso = P.singular_ratio(a_, b_, c_, d_)
    fold = np.any(cross[sel] * sgn <= 0.0, axis=1)
    # own modulus: 0 on the faces toward column c-1 (or the start edge), 1 toward c+1
    fa, fb = D.faces[:, 0], D.faces[:, 1]

    def faces_to(pred):
        x1 = fa[piece[fa] & pred(fb)]
        x2 = fb[piece[fb] & pred(fa)]
        return np.concatenate([x1, x2])
    z_f = faces_to(lambda o: (col[o] == c - 1))
    o_f = faces_to(lambda o: (col[o] == c + 1))
    z_b = D.bnd_cell[(D.bnd_edge == D.start) & piece[D.bnd_cell]]
    o_b = D.bnd_cell[(D.bnd_edge == D.end) & piece[D.bnd_cell]]
    E_xi = P.own_modulus(D.act.size, piece, D.faces, z_f, o_f, z_b, o_b)
    z_f = faces_to(lambda o: (col[o] == c) & (row[o] == r - 1))
    o_f = faces_to(lambda o: (col[o] == c) & (row[o] == r + 1))
    z_b = D.bnd_cell[np.isin(D.bnd_edge, D.one) & piece[D.bnd_cell]]
    o_b = D.bnd_cell[np.isin(D.bnd_edge, D.two) & piece[D.bnd_cell]]
    E_eta = P.own_modulus(D.act.size, piece, D.faces, z_f, o_f, z_b, o_b)
    return {
        "piece": f"F{c}{r}", "cells": int(piece.sum()),
        "modulus_pred_M_du_over_dv": D.modulus() * du / dv,
        "modulus_own_1_over_Exi": 1.0 / E_xi, "modulus_own_Eeta": E_eta,
        "duality_own": E_xi * E_eta,
        "J_min": q(J, 0), "J_max": q(J, 100), "J_ratio": q(J, 100) / max(q(J, 0), 1e-300),
        "J_ratio_p99_p1": q(J, 99) / max(q(J, 1), 1e-300),
        "J_ratio_away_from_walls": q(J[~wall], 100) / max(q(J[~wall], 0), 1e-300),
        "aniso_median": q(aniso, 50), "aniso_max": q(aniso, 100),
        "K11_p5_p95": [q(K11, 5), q(K11, 95)], "K22_p5_p95": [q(K22, 5), q(K22, 95)],
        "K12_abs_p95": q(np.abs(K12), 95),
        "K11_p5_p95_away_from_walls": [q(K11[~wall], 5), q(K11[~wall], 95)],
        "K22_p5_p95_away_from_walls": [q(K22[~wall], 5), q(K22[~wall], 95)],
        "folds": int(fold.sum()), "vertices": int(sel.sum()),
    }


def run_equal_value(key: str, target: float, n_across: int) -> dict:
    D = Domain(key)
    M = D.modulus()
    n_along = max(1, int(round(M * n_across / target)))
    col, row, pid = piece_ids(D, n_along, n_across)
    grads = P.vertex_gradients(D.along, D.across, D.act, D.dx)
    per = [conformal_metrics(D, n_along, n_across, col, row, pid, c, r, grads)
           for c in range(n_along) for r in range(n_across)]
    return {"case": key, "global_modulus": M, "target_piece_modulus": target,
            "along": n_along, "across": n_across, "pieces": per,
            "pid": pid.reshape(D.act.shape)}


# ---------------------------------------------------------------------------
# 2. one piece, three charts
# ---------------------------------------------------------------------------


def cut_curve(D: Domain, piece: np.ndarray, nxt: np.ndarray, s1_end: np.ndarray,
              s2_end: np.ndarray) -> np.ndarray:
    """The cut between ``piece`` and ``nxt`` as a polyline from its point on side one
    to its point on side two: the midpoints of the faces between them, ordered by the
    across value and smoothed over five faces, with the two wall points as its ends."""
    fa, fb = D.faces[:, 0], D.faces[:, 1]
    m1 = piece[fa] & nxt[fb]
    m2 = piece[fb] & nxt[fa]
    a = np.concatenate([fa[m1], fb[m2]])
    b = np.concatenate([fb[m1], fa[m2]])
    ca = np.column_stack([a % D.nx + 0.5, a // D.nx + 0.5])
    cb = np.column_stack([b % D.nx + 0.5, b // D.nx + 0.5])
    mid = 0.5 * (ca + cb)
    v = 0.5 * (D.V[a] + D.V[b])
    order = np.argsort(v)
    mid = mid[order]
    k = 5
    if len(mid) > k:
        pad = np.vstack([np.repeat(mid[:1], k // 2, 0), mid, np.repeat(mid[-1:], k // 2, 0)])
        ker = np.ones(k) / k
        mid = np.column_stack([np.convolve(pad[:, 0], ker, "valid"),
                               np.convolve(pad[:, 1], ker, "valid")])
    return np.vstack([s1_end, mid, s2_end])


def solve_dirichlet(D: Domain, piece: np.ndarray, face_cells: np.ndarray,
                    face_vals: np.ndarray) -> np.ndarray:
    """Harmonic on the piece's cells (unit spacing, 5 points), fixed to ``face_vals`` at
    half a cell on each listed face (a face is listed by its inside cell); no other
    faces are open."""
    cells = np.flatnonzero(piece)
    pos = np.full(D.act.size, -1, dtype=np.int64)
    pos[cells] = np.arange(cells.size)
    n = cells.size
    a, b = D.faces[:, 0], D.faces[:, 1]
    both = piece[a] & piece[b]
    pa, pb = pos[a[both]], pos[b[both]]
    rows = np.concatenate([pa, pb, pa, pb])
    cols = np.concatenate([pa, pb, pb, pa])
    vals = np.concatenate([np.ones(pa.size), np.ones(pa.size), -np.ones(pa.size),
                           -np.ones(pa.size)])
    diag = np.zeros(n)
    rhs = np.zeros(n)
    np.add.at(diag, pos[face_cells], 2.0)
    np.add.at(rhs, pos[face_cells], 2.0 * face_vals)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n)) + sp.diags(diag)
    x = spla.spsolve(A.tocsc(), rhs)
    out = np.full(D.act.size, np.nan)
    out[cells] = x
    return out


def inverse_chart_metrics(D: Domain, piece: np.ndarray, XI: np.ndarray, ETA: np.ndarray,
                          corners: np.ndarray) -> dict:
    """J, anisotropy, folds and K_hat of an inverse chart (xi, eta)(x) on a piece."""
    xi2 = XI.reshape(D.act.shape)
    eta2 = ETA.reshape(D.act.shape)
    act_p = piece.reshape(D.act.shape)
    j, i, ux, uy, vx, vy, flat, cross = P.vertex_gradients(xi2, eta2, act_p, D.dx)
    det = ux * vy - uy * vx
    sgn = float(np.sign(np.median(det)))
    det = det * sgn
    area = float(piece.sum()) * D.dx * D.dx
    ok = det > 0
    J = np.where(ok, 1.0 / np.where(ok, det, 1.0) / area, np.nan)
    Jw = J * area
    K11 = Jw * (ux * ux + uy * uy)
    K22 = Jw * (vx * vx + vy * vy)
    K12 = Jw * (ux * vx + uy * vy)
    aniso = P.singular_ratio(ux, uy, vx, vy)
    fold = np.any(cross * sgn <= 0.0, axis=1)
    pts = np.column_stack([i, j]).astype(float)          # vertex positions in cells
    dcorner = np.min(np.hypot(pts[:, None, 0] - corners[None, :, 0],
                              pts[:, None, 1] - corners[None, :, 1]), axis=1)
    near = dcorner <= 3.0
    kmin = int(np.nanargmin(np.where(ok, J, np.inf)))
    kmax = int(np.nanargmax(np.where(ok, J, -np.inf)))
    return summary(J, aniso, K11, K22, K12, fold, ~ok, near,
                   float(dcorner[kmin]), float(dcorner[kmax]), int(i.size))


def summary(J, aniso, K11, K22, K12, fold, nonpos, near, dmin, dmax, n) -> dict:
    far = ~near
    return {
        "samples": n, "folds": int(np.sum(fold)), "J_nonpositive": int(np.sum(nonpos)),
        "J_min": q(J, 0), "J_max": q(J, 100),
        "J_ratio": q(J, 100) / max(q(J, 0), 1e-300),
        "J_ratio_p99_p1": q(J, 99) / max(q(J, 1), 1e-300),
        "J_ratio_beyond_3_cells_of_corners": q(J[far], 100) / max(q(J[far], 0), 1e-300),
        "J_min_distance_to_corner_cells": dmin, "J_max_distance_to_corner_cells": dmax,
        "aniso_median": q(aniso, 50), "aniso_p95": q(aniso, 95), "aniso_max": q(aniso, 100),
        "K11_p5_median_p95": [q(K11, 5), q(K11, 50), q(K11, 95)],
        "K22_p5_median_p95": [q(K22, 5), q(K22, 50), q(K22, 95)],
        "K12_abs_p95": q(np.abs(K12), 95),
        "K11_rel_spread_p5_p95": (q(K11, 95) - q(K11, 5)) / max(q(K11, 50), 1e-300),
        "K22_rel_spread_p5_p95": (q(K22, 95) - q(K22, 5)) / max(q(K22, 50), 1e-300),
    }


def tfi(xa, xc, xb, xd, n: int):
    """Coons patch on an n x n grid of cell centres of [0,1]^2: x_A(eta) at xi = 0,
    x_C(eta) at xi = 1, x_B(xi) at eta = 0, x_D(xi) at eta = 1, each an arc-length
    parametrised polyline."""
    t = (np.arange(n) + 0.5) / n
    XI, ETA = np.meshgrid(t, t, indexing="ij")
    A = at_arclength(xa, t * arclength(xa)[-1])
    C = at_arclength(xc, t * arclength(xc)[-1])
    B = at_arclength(xb, t * arclength(xb)[-1])
    Dd = at_arclength(xd, t * arclength(xd)[-1])
    p00, p01 = xa[0], xa[-1]                       # (xi, eta) = (0, 0), (0, 1)
    p10, p11 = xc[0], xc[-1]                       # (1, 0), (1, 1)
    X = np.empty((n, n, 2))
    for k in range(2):
        X[..., k] = ((1 - XI) * A[None, :, k] + XI * C[None, :, k]
                     + (1 - ETA) * B[:, None, k] + ETA * Dd[:, None, k]
                     - ((1 - XI) * (1 - ETA) * p00[k] + (1 - XI) * ETA * p01[k]
                        + XI * (1 - ETA) * p10[k] + XI * ETA * p11[k]))
    return X


def forward_chart_metrics(X: np.ndarray, corners: np.ndarray) -> dict:
    n = X.shape[0]
    h = 1.0 / n
    x_xi = np.gradient(X[..., 0], h, axis=0)
    y_xi = np.gradient(X[..., 1], h, axis=0)
    x_eta = np.gradient(X[..., 0], h, axis=1)
    y_eta = np.gradient(X[..., 1], h, axis=1)
    det = x_xi * y_eta - x_eta * y_xi
    sgn = float(np.sign(np.median(det)))
    det = det * sgn
    area = float(np.sum(det)) * h * h
    J = det / area
    # K_hat = J DF^-1 DF^-T; DF^-1 = [[y_eta, -x_eta], [-y_xi, x_xi]] / det
    a_, b_ = y_eta / det, -x_eta / det           # grad xi  (row one of DF^-1)
    c_, d_ = -y_xi / det, x_xi / det             # grad eta (row two)
    K11 = det * (a_ * a_ + b_ * b_)
    K22 = det * (c_ * c_ + d_ * d_)
    K12 = det * (a_ * c_ + b_ * d_)
    aniso = P.singular_ratio(x_xi, x_eta, y_xi, y_eta)
    pts = X.reshape(-1, 2)
    dcorner = np.min(np.hypot(pts[:, None, 0] - corners[None, :, 0],
                              pts[:, None, 1] - corners[None, :, 1]), axis=1).reshape(n, n)
    near = dcorner <= 3.0
    ok = det > 0
    kmin = np.unravel_index(int(np.argmin(np.where(ok, J, np.inf))), J.shape)
    kmax = np.unravel_index(int(np.argmax(np.where(ok, J, -np.inf))), J.shape)
    return summary(np.where(ok, J, np.nan), aniso, K11, K22, K12, np.zeros(1, bool), ~ok,
                   near, float(dcorner[kmin]), float(dcorner[kmax]), int(n * n))


def run_compare(key: str, n_along: int, n_across: int, cr: tuple[int, int], label: str):
    D = Domain(key)
    c, r = cr
    assert n_across == 1, "the comparison uses whole-width pieces (walls as two sides)"
    col, row, pid = piece_ids(D, n_along, n_across)
    k = c * n_across + r
    piece = pid == k
    u_lo, u_hi = c / n_along, (c + 1) / n_along
    # the piece's four sides as arc-length polylines, oriented for the chart:
    # A (xi = 0) from side one to side two; C (xi = 1) likewise; B (eta = 0) along side
    # one from A to C; D (eta = 1) along side two from A to C
    s1_lo = 0.0 if c == 0 else D.wall_s_at_along(1, u_lo)
    s1_hi = arclength(D.side1)[-1] if c == n_along - 1 else D.wall_s_at_along(1, u_hi)
    s2_lo = 0.0 if c == 0 else D.wall_s_at_along(2, u_lo)
    s2_hi = arclength(D.side2)[-1] if c == n_along - 1 else D.wall_s_at_along(2, u_hi)
    xb = sub_polyline(D.side1, s1_lo, s1_hi)
    xd = sub_polyline(D.side2, s2_lo, s2_hi)
    if c == 0:
        xa = D.start_poly
    else:
        prev = pid == (c - 1) * n_across + r
        xa = cut_curve(D, prev, piece, xb[0], xd[0])
    if c == n_along - 1:
        xc = np.asarray(dict(geo.drawn_edges(D.d))[D.end], dtype=float)
        if np.hypot(*(xc[0] - xb[-1])) > np.hypot(*(xc[-1] - xb[-1])):
            xc = xc[::-1]
    else:
        nxt = pid == (c + 1) * n_across + r
        xc = cut_curve(D, piece, nxt, xb[-1], xd[-1])
    corners = np.array([xa[0], xa[-1], xc[0], xc[-1]])

    # conformal: the layout's pair on the piece
    xi_c = np.where(piece, (D.U - u_lo) / (u_hi - u_lo), np.nan)
    eta_c = np.where(piece, D.V, np.nan)
    conf = inverse_chart_metrics(D, piece, xi_c, eta_c, corners)

    # Winslow: Dirichlet on all four sides, by normalised arc length
    cells, vals_xi, vals_eta = [], [], []
    on = piece[D.bnd_cell]
    e = D.bnd_edge
    mids = D.bnd_mid
    Lb, Ld = arclength(xb)[-1], arclength(xd)[-1]
    La, Lc = arclength(xa)[-1], arclength(xc)[-1]
    sel = on & np.isin(e, D.one)
    s, _ = project(xb, mids[sel])
    cells.append(D.bnd_cell[sel]); vals_xi.append(s / Lb); vals_eta.append(np.zeros(sel.sum()))
    sel = on & np.isin(e, D.two)
    s, _ = project(xd, mids[sel])
    cells.append(D.bnd_cell[sel]); vals_xi.append(s / Ld); vals_eta.append(np.ones(sel.sum()))
    for is_a, poly, L in ((True, xa, La), (False, xc, Lc)):
        if (is_a and c == 0) or ((not is_a) and c == n_along - 1):
            sel = on & (e == (D.start if is_a else D.end))
            s, _ = project(poly, mids[sel])
            cells.append(D.bnd_cell[sel])
            vals_xi.append(np.full(sel.sum(), 0.0 if is_a else 1.0))
            vals_eta.append(s / L)
        else:
            other = pid == ((c - 1) if is_a else (c + 1)) * n_across + r
            fa, fb = D.faces[:, 0], D.faces[:, 1]
            m1 = piece[fa] & other[fb]
            m2 = piece[fb] & other[fa]
            inside = np.concatenate([fa[m1], fb[m2]])
            outside = np.concatenate([fb[m1], fa[m2]])
            mid = 0.5 * (np.column_stack([inside % D.nx + 0.5, inside // D.nx + 0.5])
                         + np.column_stack([outside % D.nx + 0.5, outside // D.nx + 0.5]))
            s, _ = project(poly, mid)
            cells.append(inside)
            vals_xi.append(np.full(inside.size, 0.0 if is_a else 1.0))
            vals_eta.append(s / L)
    cells = np.concatenate(cells)
    xi_w = solve_dirichlet(D, piece, cells, np.concatenate(vals_xi))
    eta_w = solve_dirichlet(D, piece, cells, np.concatenate(vals_eta))
    wins = inverse_chart_metrics(D, piece, xi_w, eta_w, corners)

    # transfinite
    X = tfi(xa, xc, xb, xd, N_REF)
    tf = forward_chart_metrics(X, corners)

    fields = {"act": piece.reshape(D.act.shape), "xi_conformal": xi_c.reshape(D.act.shape),
              "eta_conformal": eta_c.reshape(D.act.shape),
              "xi_winslow": xi_w.reshape(D.act.shape), "eta_winslow": eta_w.reshape(D.act.shape),
              "tfi_X": X, "side_A": xa, "side_B": xb, "side_C": xc, "side_D": xd}
    return {"case": key, "along": n_along, "across": n_across, "piece": f"F{c}{r}",
            "label": label, "cells": int(piece.sum()),
            "side_lengths_cells": {"A": La, "B": Lb, "C": Lc, "D": Ld},
            "conformal": conf, "winslow": wins, "transfinite": tf}, fields


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def text(eq: list[dict], cmp_: list[dict]) -> str:
    L = []
    w = L.append
    w("arch_chart_maps -- equal-value cuts, and one piece under three charts")
    w("=" * 78)
    w("")
    w("1. EQUAL-VALUE CUTS: along in [c/n, (c+1)/n), across in [r/m, (r+1)/m), n = round(M m)")
    for res in eq:
        w("")
        w(f"{res['case']}: global modulus M = {res['global_modulus']:.4f}; "
          f"{res['along']} along x {res['across']} across; predicted piece modulus "
          f"M m / n = {res['global_modulus'] * res['across'] / res['along']:.4f}")
        w("  piece  cells  M_pred  M_own  E.E(own)  Jmax/Jmin (no wall)  aniso med/max  "
          "K11 p5-p95 (no wall)   K22 p5-p95 (no wall)   |K12| p95  folds")
        for p in res["pieces"]:
            k1 = p["K11_p5_p95_away_from_walls"]
            k2 = p["K22_p5_p95_away_from_walls"]
            w(f"  {p['piece']:<5} {p['cells']:>5}  {p['modulus_pred_M_du_over_dv']:6.3f} "
              f"{p['modulus_own_1_over_Exi']:6.3f}  {p['duality_own']:7.4f}  "
              f"{p['J_ratio']:8.2f} ({p['J_ratio_away_from_walls']:6.2f})  "
              f"{p['aniso_median']:5.2f}/{p['aniso_max']:6.2f}  "
              f"{k1[0]:6.3f}-{k1[1]:6.3f}          {k2[0]:6.3f}-{k2[1]:6.3f}        "
              f"{p['K12_abs_p95']:8.4f}  {p['folds']:>4}")
    w("")
    w("2. ONE PIECE, THREE CHARTS (conformal = layout pair; Winslow = harmonic with")
    w("   arc-length data on all four sides; transfinite = Coons patch of the same sides)")
    for res in cmp_:
        w("")
        sl = res["side_lengths_cells"]
        w(f"{res['label']}: {res['case']} {res['along']}x{res['across']} piece {res['piece']}, "
          f"{res['cells']} cells; side lengths A {sl['A']:.1f}, B {sl['B']:.1f}, "
          f"C {sl['C']:.1f}, D {sl['D']:.1f} cells")
        w("  chart        folds J<=0   Jmax/Jmin  p99/p1  (>3 cells from corners)  "
          "Jmin@ Jmax@ (cells from corner)  aniso med/p95/max   "
          "K11 p5/med/p95        K22 p5/med/p95        |K12| p95")
        for name in ("conformal", "winslow", "transfinite"):
            m = res[name]
            k1, k2 = m["K11_p5_median_p95"], m["K22_p5_median_p95"]
            w(f"  {name:<12} {m['folds']:>5} {m['J_nonpositive']:>5}  {m['J_ratio']:10.2f} "
              f"{m['J_ratio_p99_p1']:7.2f}  ({m['J_ratio_beyond_3_cells_of_corners']:8.2f})   "
              f"          {m['J_min_distance_to_corner_cells']:5.1f} "
              f"{m['J_max_distance_to_corner_cells']:5.1f}          "
              f"{m['aniso_median']:5.2f}/{m['aniso_p95']:5.2f}/{m['aniso_max']:6.2f}   "
              f"{k1[0]:5.3f}/{k1[1]:5.3f}/{k1[2]:6.3f}   {k2[0]:5.3f}/{k2[1]:5.3f}/{k2[2]:6.3f}   "
              f"{m['K12_abs_p95']:7.4f}")
    return "\n".join(L) + "\n"


def main() -> None:
    eq = [run_equal_value(*c) for c in EQUAL_VALUE]
    cmp_, fields = [], {}
    for case in COMPARE:
        res, f = run_compare(*case)
        cmp_.append(res)
        tag = f"{res['case']}_{res['along']}x{res['across']}_{res['piece']}"
        for kk, vv in f.items():
            fields[f"{tag}__{kk}"] = vv
    for res in eq:
        fields[f"equal_value_{res['case']}_{res['along']}x{res['across']}__pid"] = res.pop("pid")
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "chart_maps_fields.npz"), **fields)
    with open(os.path.join(OUT, "chart_maps.json"), "w", encoding="utf-8") as fh:
        json.dump({"probe": "arch_chart_maps", "equal_value": eq, "compare": cmp_}, fh, indent=1)
    t = text(eq, cmp_)
    with open(os.path.join(OUT, "chart_maps.txt"), "w", encoding="utf-8") as fh:
        fh.write(t)
    sys.stdout.write(t)


if __name__ == "__main__":
    main()
