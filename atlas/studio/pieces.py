"""Pieces: the domain cut into four-sided pieces, each charted, certified and given an
expert, coupled by the chart operator's host (plan §5).

**Cutting.**  Automatic quad layout of an arbitrary domain is a known hard problem, so
the first version covers the domains whose layout is a strip (the plan's
**[AI Inference]**) and says plainly when it cannot cut one:

* ``grid``: a rectangle, cut on a grid and at every material interface of its
  rectangular regions, so that each piece has one material; affine charts;
* ``polar``: an annular sector (two radial sides, two arcs), cut at equal ``log r``
  and equal angle; the log-polar map, the exact conformal chart (`benchmarks.bend3`);
* ``strip``: any four-sided domain (its four sharpest corners), cut at equal values of
  its harmonic coordinate along the walls and charted conformally, the end pieces by
  Winslow where the card's envelope refuses the conformal chart: the algorithm of
  `benchmarks.s_channel`, for any strip;
* ``insert``: a rectangle with one round insert, the O-H layout of
  `benchmarks.insert_round` (a core of squares, two rings, two side columns).

The cutters repeat the constructions of `atlas/chartop/benchmarks.py` on the case's own
outline, so the three held-out benchmarks cut here are that module's pieces, chart for
chart (tested).  That module is pinned by registered evaluations: nothing here edits
it, and the constructions are written out again rather than refactored out of it.

**Hand cutting** (`split`): a cut joins a point on one side of a piece to the point at
the same fraction of the opposite side, and continues through every piece across that
side, so every seam stays conforming (a chord of the layout).  `merge` removes one, and
`move_corner` drags an inside corner where straight cuts meet.

**Charts** per piece: TFI (the Coons patch), Laplacian (Winslow's harmonic chart) or
conformal (`atlas/studio/conformal.py`, new), or the cutter's own.  **Certificates**
(`certificate.certify`, Proposition 3): ``det J > 0`` at every cell corner and a
simple, positively oriented boundary; the smallest ``det J`` (relative to its mean)
is shown, and a folded piece is refused with it.

**Coupling** (`solve`): the pieces become `PieceProblem`s with the case's materials and
step, and `host.Layout` couples them with the classical expert; `host.monolithic` is
the same discretisation solved whole, and the two agree to round-off when every seam
conforms (the host's Theorem 2; tested).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from atlas.chartop import certificate as cert
from atlas.chartop import chart as ch
from atlas.chartop import host as hs
from atlas.chartop.piece import PieceProblem

KIND, M = "sine-hat", 16               # the port basis the experts were trained with
METHODS = ("tfi", "laplacian", "conformal")
METHOD_LABELS = {"tfi": "TFI", "laplacian": "Laplacian", "conformal": "Conformal",
                 "affine": "affine", "conformal-log-polar": "conformal (log-polar)",
                 "conformal-harmonic": "conformal (harmonic)", "winslow-arc-length": "Laplacian (Winslow)",
                 "transfinite": "TFI", "conformal-numerical": "conformal (numerical)"}


class CutError(ValueError):
    """A domain or a cut this module cannot make; the message says why and what to do."""


@dataclass
class Piece:
    sides: list                       # four chartop Sides, counter-clockwise, loop order
    material: str
    label: str
    own: str                          # the cutter's chart method
    own_chart: ch.Chart | None = None # the cutter's chart at the cut's n
    method: str = "own"               # "own", or one of METHODS
    expert: str = "classical"
    #: side index -> the hand-cut chord it lies on (`split`), so `merge` can remove it
    chords: dict = field(default_factory=dict)
    #: charts already made, by (method, n)
    cache: dict = field(default_factory=dict, repr=False, compare=False)

    def corners(self) -> np.ndarray:
        return np.array([s.start for s in self.sides])


@dataclass
class Cut:
    cutter: str
    params: dict
    n: int
    pieces: list
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# the case's geometry, in metres
# ---------------------------------------------------------------------------


def _materials(spec) -> dict:
    from atlas.workbench import registry
    lib = registry.family(spec.physics.family).material_library()
    return {**{k: dict(v) for k, v in lib.items()}, **{k: dict(v) for k, v in spec.materials.items()}}


def material_at(spec, x: float, y: float) -> str | None:
    """The material of the cell holding the point (the regions' order decides)."""
    from atlas.workbench import geometry as wgeo
    d = spec.domain
    i, j = int(math.floor(x / d.dx)), int(math.floor(y / d.dx))
    i, j = min(max(i, 0), d.nx - 1), min(max(j, 0), d.ny - 1)
    owner = wgeo.region_owner(spec.regions, d.nx, d.ny)
    k = int(owner[j, i])
    return spec.regions[k].material if k >= 0 else None


def boundary_edges(spec) -> list:
    """Every named boundary edge of the domain as a polyline in metres."""
    from atlas.workbench import geometry as wgeo
    d = spec.domain
    dx = float(d.dx)
    if d.outline is None:
        W, H = d.nx * dx, d.ny * dx
        return [("bottom", np.array([[0.0, 0.0], [W, 0.0]])), ("right", np.array([[W, 0.0], [W, H]])),
                ("top", np.array([[0.0, H], [W, H]])), ("left", np.array([[0.0, 0.0], [0.0, H]]))]
    return [(name, np.asarray(poly, float) * dx) for name, poly in wgeo.drawn_edges(d, step=0.02)]


def _dist_to_polyline(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    a, ab = Q[:-1], np.diff(Q, axis=0)
    L2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300)
    out = np.full(len(P), np.inf)
    for k in range(len(a)):
        t = np.clip(((P - a[k]) @ ab[k]) / L2[k], 0.0, 1.0)
        out = np.minimum(out, np.hypot(*(a[k] + t[:, None] * ab[k] - P).T))
    return out


def make_bc(spec):
    """bc(i, s, points) for the host from the case's boundary conditions: a side whose
    every node lies on edges with one fixed temperature is Dirichlet, one on insulated
    edges (or a heat flux of zero) insulated; a side may run along several edges with
    the same condition.  A side on no named edge, across two different conditions, or
    on a nonzero heat flux is refused: the host takes fixed values and insulated sides."""
    edges = dict()
    for name, poly in boundary_edges(spec):
        edges.setdefault(name, []).append(poly)
    dx = float(spec.domain.dx)
    # a fiftieth of a cell: a curved wall's nodes sit on chords of its drawn curve (8e-6 m
    # off the S-channel's splines at 5 mm cells), far closer than any two edges come
    tol = 0.02 * dx
    groups: dict = {}
    for b in spec.boundaries:
        if b.kind in ("fixed-temperature", "fixed-potential", "electrode"):
            key = ("dirichlet", float(b.value))
        elif b.kind in ("insulated", "no-current") or (b.kind == "heat-flux" and not b.value):
            key = ("insulated", None)
        else:
            key = ("refused", f"{b.edge} ({b.kind})")
        for poly in edges.get(b.edge, []):
            Q = poly
            if b.start is not None and b.stop is not None and spec.domain.outline is None:
                s0, s1 = float(b.start) * dx, float(b.stop) * dx      # a stretch of a grid edge
                a, z = Q[0], Q[-1]
                L = float(np.hypot(*(z - a)))
                Q = np.array([a + (z - a) * (s0 / L), a + (z - a) * (s1 / L)])
            groups.setdefault(key, []).append(Q)

    def bc(i, s, P):
        P = np.asarray(P, float)
        for key, polys in groups.items():
            d = np.min([_dist_to_polyline(P, Q) for Q in polys], axis=0)
            if np.all(d <= tol):
                if key[0] == "refused":
                    raise CutError(f"piece {i}, side {s} lies on {key[1]}: the chart operator's "
                                   f"host takes fixed values and insulated sides")
                if key[0] == "dirichlet":
                    return ("dirichlet", np.full(len(P), key[1]))
                return ("insulated", None)
        raise CutError(f"piece {i}, side {s} lies on no single boundary condition: cut it so "
                       f"its outer sides follow the domain's edges, one condition each")
    return bc


# ---------------------------------------------------------------------------
# cutters
# ---------------------------------------------------------------------------


def _piece(sides, spec, label, own, chart=None, material=None) -> Piece:
    if material is None:
        c = np.mean([s(0.5) for s in sides], axis=0)
        X = chart.X if chart is not None else None
        mid = X[X.shape[0] // 2, X.shape[1] // 2] if X is not None else c
        material = material_at(spec, float(mid[0]), float(mid[1]))
    return Piece(sides, material or "", label, own, chart)


def _affine(x0, x1, y0, y1, n):
    def rect(xi, eta):
        return x0 + (x1 - x0) * xi, y0 + (y1 - y0) * eta
    return ch.Chart(ch.from_map(rect, n), "affine")


def _rect_sides(x0, x1, y0, y1):
    return [ch.line((x0, y0), (x1, y0)), ch.line((x1, y0), (x1, y1)),
            ch.line((x1, y1), (x0, y1)), ch.line((x0, y1), (x0, y0))]


def cut_grid(spec, n: int = 64, cols: int = 2, rows: int = 1) -> Cut:
    """A rectangle on a cols x rows grid, cut again at every edge of a rectangular
    region inside it (each piece one material)."""
    d = spec.domain
    if d.outline is not None or d.holes:
        raise CutError("the grid cut takes a plain rectangle; this domain has a drawn outline "
                       "or holes")
    dx = float(d.dx)
    W, H = d.nx * dx, d.ny * dx
    xs = set(np.linspace(0.0, W, cols + 1).round(15))
    ys = set(np.linspace(0.0, H, rows + 1).round(15))
    for r in spec.regions:
        if getattr(r, "shape", "rect") == "rect" and r.points is None:
            for v in (r.x0 * dx, (r.x0 + r.nx) * dx):
                if 0 < v < W:
                    xs.add(round(v, 15))
            for v in (r.y0 * dx, (r.y0 + r.ny) * dx):
                if 0 < v < H:
                    ys.add(round(v, 15))
        else:
            raise CutError(f"region {r.id} is not a rectangle: the grid cut follows rectangular "
                           f"regions only; cut by hand, or use the insert cut for a round one")
    xs, ys = sorted(xs), sorted(ys)
    pieces = []
    for q in range(len(ys) - 1):
        for p in range(len(xs) - 1):
            x0, x1, y0, y1 = xs[p], xs[p + 1], ys[q], ys[q + 1]
            c = _affine(x0, x1, y0, y1, n)
            pieces.append(_piece(_rect_sides(x0, x1, y0, y1), spec, f"block {p},{q}", "affine", c,
                                 material_at(spec, 0.5 * (x0 + x1), 0.5 * (y0 + y1))))
    return Cut("grid", {"cols": cols, "rows": rows}, n, pieces,
               [f"{len(pieces)} rectangles; cuts at {len(xs) - 2} x and {len(ys) - 2} y lines"])


def _outline_edges(spec):
    from atlas.workbench import shapes
    o = spec.domain.outline
    if o is None:
        raise CutError("this domain has no drawn outline")
    dx = float(spec.domain.dx)
    return [np.asarray(e, float) * dx for e in shapes.edges_sampled(o.points, o.edges, o.bulge, step=0.02)], o


def cut_polar(spec, n: int = 64, n_r: int = 3, n_th: int = 5) -> Cut:
    """An annular sector: two radial lines and two arcs about one centre."""
    E, o = _outline_edges(spec)
    if len(o.edges) != 4 or sorted(o.edges) != ["arc", "arc", "line", "line"]:
        raise CutError("the polar cut takes an annular sector: two straight radial sides and two "
                       "arcs")
    dx = float(spec.domain.dx)
    pts = [np.asarray(p, float) * dx for p in o.points]
    lines = [k for k, e in enumerate(o.edges) if e == "line"]
    (a, b), (c_, d_) = [(pts[k], pts[(k + 1) % 4]) for k in lines]
    A = np.array([b - a, -(d_ - c_)]).T
    if abs(np.linalg.det(A)) < 1e-14:
        raise CutError("the two straight sides are parallel: not an annular sector")
    s_, _t = np.linalg.solve(A, c_ - a)
    ctr = a + s_ * (b - a)
    # the exact centre when the outline's points say it in cells (the benchmark's 4 dx)
    ctr = np.round(ctr / dx, 9) * dx
    rad = sorted({round(float(np.hypot(*(p - ctr))), 12) for p in pts})
    if len(rad) != 2:
        raise CutError("the sector's corners do not lie on two circles about one centre")
    r0, r1 = rad
    ang = sorted({math.atan2(p[1] - ctr[1], p[0] - ctr[0]) for p in pts})
    th0, th1 = ang[0], ang[-1]
    rho0, rho1 = math.log(r0), math.log(r1)
    dr, dth = (rho1 - rho0) / n_r, (th1 - th0) / n_th
    cx, cy = float(ctr[0]), float(ctr[1])
    pieces = []
    for bb in range(n_th):
        for aa in range(n_r):
            def F(xi, eta, a=aa, b=bb):
                rr = np.exp(rho0 + (a + xi) * dr)
                th = th0 + (b + eta) * dth
                return cx + rr * np.cos(th), cy + rr * np.sin(th)
            c = ch.Chart(ch.from_map(F, n), "conformal-log-polar", {"modulus": dr / dth})
            # the sides are the map's own boundary, exact arcs and radii in the map's own
            # parameter, so another chart method starts from the true curves
            sides = [ch.side_from_function(lambda t, F=F: np.stack(F(t, 0.0 * t), -1), arclength_param=False),
                     ch.side_from_function(lambda t, F=F: np.stack(F(1.0 + 0.0 * t, t), -1), arclength_param=False),
                     ch.side_from_function(lambda t, F=F: np.stack(F(1.0 - t, 1.0 + 0.0 * t), -1), arclength_param=False),
                     ch.side_from_function(lambda t, F=F: np.stack(F(0.0 * t, 1.0 - t), -1), arclength_param=False)]
            pieces.append(_piece(sides, spec, f"bend {aa},{bb}", "conformal-log-polar", c))
    return Cut("polar", {"n_r": n_r, "n_th": n_th, "centre": [cx, cy], "radii": [r0, r1]}, n, pieces,
               [f"{n_r} x {n_th} pieces at equal log r and angle; log-polar charts (exact conformal)"])


def _sharpest_corners(E):
    """The four outline vertices where the direction turns most (a strip's corners)."""
    turns = []
    for k in range(len(E)):
        a = E[k - 1][-1] - E[k - 1][-2]
        b = E[k][1] - E[k][0]
        cr = a[0] * b[1] - a[1] * b[0]
        dt = a[0] * b[0] + a[1] * b[1]
        turns.append(abs(math.atan2(cr, dt)))
    return sorted(sorted(range(len(E)), key=lambda k: -turns[k])[:4])


def _polyline_at(P: np.ndarray, s_query) -> np.ndarray:
    s = ch.arclength(P)
    return np.stack([np.interp(s_query, s, P[:, 0]), np.interp(s_query, s, P[:, 1])], -1)


def _rect_q1(X: np.ndarray):
    """Q1 stiffness of the identity conductivity on a structured mesh (as benchmarks)."""
    import scipy.sparse as sp
    from atlas.chartop import fe
    NY, NX = X.shape[0] - 1, X.shape[1] - 1
    d_xi = 0.5 * ((X[:-1, 1:] - X[:-1, :-1]) + (X[1:, 1:] - X[1:, :-1]))
    d_et = 0.5 * ((X[1:, :-1] - X[:-1, :-1]) + (X[1:, 1:] - X[:-1, 1:]))
    a11, a12, a22, J = ch.metric_tensor(np.stack([d_xi, d_et], axis=-1))
    if np.any(J <= 0):
        raise CutError("the strip's fine mesh folds: this domain is not a strip the cut can map")
    kxx, kyy, kxy, _ms = fe.Q1
    N1 = NX + 1
    i, j = np.meshgrid(np.arange(NX), np.arange(NY))
    nodes = np.stack([j * N1 + i, j * N1 + i + 1, (j + 1) * N1 + i + 1, (j + 1) * N1 + i],
                     axis=-1).reshape(-1, 4)
    ke = (a11.reshape(-1)[:, None, None] * kxx + a22.reshape(-1)[:, None, None] * kyy
          + a12.reshape(-1)[:, None, None] * kxy)
    rows = np.repeat(nodes, 4, axis=1).ravel()
    cols = np.tile(nodes, (1, 4)).ravel()
    Nn = (NX + 1) * (NY + 1)
    return sp.csr_matrix((ke.ravel(), (rows, cols)), shape=(Nn, Nn))


def _dirichlet_solve(K, fixed_idx, fixed_val):
    import scipy.sparse.linalg as spla
    N = K.shape[0]
    free = np.setdiff1d(np.arange(N), fixed_idx)
    u = np.zeros(N)
    u[fixed_idx] = fixed_val
    K = K.tocsr()
    u[free] = spla.splu(K[free][:, free].tocsc()).solve(-K[free][:, fixed_idx] @ fixed_val)
    return u


def cut_strip(spec, n: int = 64, along: int | None = None, NX: int = 1152, NY: int = 128,
              envelope=None) -> Cut:
    """A four-sided domain cut across at equal values of its harmonic coordinate; the
    construction of `benchmarks.s_channel` for any strip (the walls are the two longer
    of the sides between its four sharpest corners)."""
    E, o = _outline_edges(spec)
    cs = _sharpest_corners(E)
    groups = []
    for q in range(4):
        k0, k1 = cs[q], cs[(q + 1) % 4]
        ks = list(range(k0, k1)) if k1 > k0 else list(range(k0, len(E))) + list(range(0, k1))
        groups.append(np.vstack([E[ks[0]]] + [E[k][1:] for k in ks[1:]]))
    lengths = [float(ch.arclength(g)[-1]) for g in groups]
    walls = sorted(range(4), key=lambda q: -lengths[q])[:2]
    if (walls[0] - walls[1]) % 2:
        raise CutError("the domain's two longest sides are not opposite: it is not a strip")
    # the walls in loop order, the first one starting at the end with the smaller x
    w0 = min(walls, key=lambda q: float(groups[q][0][0]))
    bottom = groups[w0]
    right = groups[(w0 + 1) % 4]
    top = groups[(w0 + 2) % 4]
    left = groups[(w0 + 3) % 4]
    sb, st = ch.arclength(bottom), ch.arclength(top[::-1])
    sl, sr = ch.arclength(left[::-1]), ch.arclength(right)
    xi = np.arange(NX + 1) / NX
    et = np.arange(NY + 1) / NY
    B = _polyline_at(bottom, xi * sb[-1])
    T = _polyline_at(top[::-1], xi * st[-1])
    L = _polyline_at(left[::-1], et * sl[-1])
    R = _polyline_at(right, et * sr[-1])
    XI, ET = np.meshgrid(xi, et)
    X = np.empty((NY + 1, NX + 1, 2))
    p00, p10, p11, p01 = B[0], B[-1], T[-1], T[0]
    for c_ in range(2):
        X[..., c_] = ((1 - ET) * B[None, :, c_] + ET * T[None, :, c_]
                      + (1 - XI) * L[:, None, c_] + XI * R[:, None, c_]
                      - ((1 - XI) * (1 - ET) * p00[c_] + XI * (1 - ET) * p10[c_]
                         + XI * ET * p11[c_] + (1 - XI) * ET * p01[c_]))
    K = _rect_q1(X)
    N1 = NX + 1
    idx = np.arange((NX + 1) * (NY + 1)).reshape(NY + 1, NX + 1)
    lft, rgt = idx[:, 0], idx[:, -1]
    U = _dirichlet_solve(K, np.concatenate([lft, rgt]),
                         np.concatenate([np.zeros(NY + 1), np.ones(NY + 1)])).reshape(NY + 1, N1)
    bot, tp = idx[0, :], idx[-1, :]
    V = _dirichlet_solve(K, np.concatenate([bot, tp]),
                         np.concatenate([np.zeros(N1), np.ones(N1)])).reshape(NY + 1, N1)
    Eu = float(U.ravel() @ (K @ U.ravel()))
    Ev = float(V.ravel() @ (K @ V.ravel()))
    Mod = 1.0 / Eu
    along = along or max(1, int(round(Mod)))
    du = 1.0 / along

    def wall_point(row: int, u: float) -> np.ndarray:
        uu = np.maximum.accumulate(U[row])
        return np.array([np.interp(u, uu, X[row, :, 0]), np.interp(u, uu, X[row, :, 1])])

    def cut_curve(u: float):
        pts, vs = [], []
        for j in range(NY + 1):
            uu = np.maximum.accumulate(U[j])
            pts.append([np.interp(u, uu, X[j, :, 0]), np.interp(u, uu, X[j, :, 1])])
            vs.append(np.interp(u, uu, V[j]))
        pts, vs = np.array(pts), np.array(vs)
        vs[0], vs[-1] = 0.0, 1.0
        order = np.argsort(vs)
        return pts[order], vs[order]

    def side_u(row: int, u0: float, u1: float, rev: bool) -> ch.Side:
        t = np.linspace(0.0, 1.0, 2049)
        P = np.array([wall_point(row, u0 + (u1 - u0) * tt) for tt in t])
        s = ch.Side(P, t)
        return s.reversed() if rev else s

    cuts = {}
    for c_ in range(1, along):
        P, vs = cut_curve(c_ * du)
        cuts[c_] = ch.Side(P, vs)
    pieces, report = [], []
    for c_ in range(along):
        u0, u1 = c_ * du, (c_ + 1) * du
        s0 = side_u(0, u0, u1, rev=False)
        s2 = side_u(NY, u0, u1, rev=True)
        s1 = cuts[c_ + 1] if c_ + 1 < along else ch.side_from_polyline(right)
        s3 = (cuts[c_].reversed() if c_ > 0 else ch.side_from_polyline(left))
        sides = [s0, s1, s2, s3]
        Xc = ch.harmonic(sides, n, aniso=Mod * du)
        chosen = ch.Chart(Xc, "conformal-harmonic", {"modulus": Mod * du})
        own = "conformal-harmonic"
        if c_ == 0 or c_ == along - 1:
            m_ = cert.measure(chosen)
            refused = (envelope is not None and cert.refusals(m_, envelope)) or \
                      (envelope is None and m_["logJ_range"] > math.log(10.0))
            report.append({"piece": c_, "refused": bool(refused)})
            if refused:
                wb = ch.side_from_polyline(np.array([s0(t) for t in np.linspace(0, 1, 2049)]))
                wt = ch.side_from_polyline(np.array([s2(t) for t in np.linspace(0, 1, 2049)]))
                Xw, info = ch.winslow([wb, s1, wt, s3], n)
                chosen = ch.Chart(Xw, "winslow-arc-length", info)
                own = "winslow-arc-length"
                sides = [wb, s1, wt, s3]
        pieces.append(_piece(sides, spec, f"channel {c_}", own, chosen))
    notes = [f"{along} pieces at equal harmonic coordinate (module {Mod:.4g}, E_u E_v = {Eu * Ev:.6f})"]
    notes += [f"end piece {r['piece']}: the conformal chart was refused by the envelope, so Winslow"
              for r in report if r["refused"]]
    return Cut("strip", {"along": along, "module": Mod}, n, pieces, notes)


def _arc(c, R, a0, a1, dense=2049) -> ch.Side:
    t = np.linspace(0.0, 1.0, dense)
    a = a0 + (a1 - a0) * t
    return ch.Side(np.stack([c[0] + R * np.cos(a), c[1] + R * np.sin(a)], -1), t)


def cut_insert(spec, n: int = 64, split: int = 3) -> Cut:
    """A rectangle with one round region: the O-H layout of `benchmarks.insert_round`
    (a core of split x split squares in the disc, its ring, the plate's ring out to a
    square box of twice the disc's radius, and two side columns)."""
    d = spec.domain
    if d.outline is not None or d.holes:
        raise CutError("the insert cut takes a rectangle with a round region inside")
    round_ = [r for r in spec.regions if r.shape == "curve" and r.points is None and r.nx == r.ny]
    if len(round_) != 1:
        raise CutError("the insert cut takes exactly one round region (a circle) in a rectangle")
    dx = float(d.dx)
    disc = round_[0]
    W, Hh = d.nx * dx, d.ny * dx
    c = np.array([disc.x0 + disc.nx / 2.0, disc.y0 + disc.ny / 2.0]) * dx
    R = (disc.nx / 2.0) * dx
    a, b = 0.5 * R, 2.0 * R
    if c[1] - b < -1e-12 or c[1] + b > Hh + 1e-12 or c[0] - b <= 0 or c[0] + b >= W:
        raise CutError("the box of twice the insert's radius must fit between the plate's top "
                       "and bottom and leave a column each side")
    cu = disc.material
    st = next((r.material for r in spec.regions if r is not disc), cu)
    deg = math.pi / 180.0
    S = split
    tq = np.tan((-45.0 + 90.0 * np.arange(S + 1) / S) * deg)
    pieces = []

    def ray_point(phi_deg, k, dist):
        rr = dist / math.cos((phi_deg - 90.0 * k) * deg)
        return c + rr * np.array([math.cos(phi_deg * deg), math.sin(phi_deg * deg)])

    def on_circle(phi_deg):
        return c + R * np.array([math.cos(phi_deg * deg), math.sin(phi_deg * deg)])

    for q in range(S):
        for p in range(S):
            x0, x1 = c[0] + a * tq[p], c[0] + a * tq[p + 1]
            y0, y1 = c[1] + a * tq[q], c[1] + a * tq[q + 1]
            pieces.append(_piece(_rect_sides(x0, x1, y0, y1), spec, f"disc core {p},{q}", "affine",
                                 _affine(x0, x1, y0, y1, n), cu))
    for ring, mat in (("disc ring", cu), ("plate ring", st)):
        for k in range(4):
            for l in range(S):
                lo = 90.0 * k - 45.0 + 90.0 * l / S
                hi = 90.0 * k - 45.0 + 90.0 * (l + 1) / S
                if ring == "disc ring":
                    pin_lo, pin_hi = ray_point(lo, k, a), ray_point(hi, k, a)
                    sides = [ch.line(pin_lo, on_circle(lo)), _arc(c, R, lo * deg, hi * deg),
                             ch.line(on_circle(hi), pin_hi), ch.line(pin_hi, pin_lo)]
                else:
                    pb_lo, pb_hi = ray_point(lo, k, b), ray_point(hi, k, b)
                    sides = [ch.line(on_circle(lo), pb_lo), ch.line(pb_lo, pb_hi),
                             ch.line(pb_hi, on_circle(hi)), _arc(c, R, hi * deg, lo * deg)]
                Xw, info = ch.winslow(sides, n)
                pieces.append(_piece(sides, spec, f"{ring} {k},{l}", "winslow-arc-length",
                                     ch.Chart(Xw, "winslow-arc-length", info), mat))
    ys = c[1] + b * tq
    for label, x0, x1 in (("left column", 0.0, c[0] - b), ("right column", c[0] + b, W)):
        for q in range(S):
            y0, y1 = ys[q], ys[q + 1]
            pieces.append(_piece(_rect_sides(x0, x1, y0, y1), spec, f"{label} {q}", "affine",
                                 _affine(x0, x1, y0, y1, n), st))
    return Cut("insert", {"split": split, "centre": c.tolist(), "radius": R}, n, pieces,
               [f"{len(pieces)} pieces: the O-H layout around the insert"])


CUTTERS = {"grid": cut_grid, "polar": cut_polar, "strip": cut_strip, "insert": cut_insert}


def auto_cut(spec, n: int = 64, **kw) -> Cut:
    """The cutter this domain's shape calls for, or a refusal that says why."""
    d = spec.domain
    tried = []
    if d.outline is None and not d.holes:
        if any(r.shape == "curve" and r.points is None for r in spec.regions):
            try:
                return cut_insert(spec, n, **{k: v for k, v in kw.items() if k == "split"})
            except CutError as exc:
                tried.append(f"insert: {exc}")
        try:
            return cut_grid(spec, n, **{k: v for k, v in kw.items() if k in ("cols", "rows")})
        except CutError as exc:
            tried.append(f"grid: {exc}")
    elif d.outline is not None and not d.holes:
        o = d.outline
        if len(o.edges) == 4 and sorted(o.edges) == ["arc", "arc", "line", "line"]:
            try:
                return cut_polar(spec, n, **{k: v for k, v in kw.items() if k in ("n_r", "n_th")})
            except CutError as exc:
                tried.append(f"polar: {exc}")
        try:
            from atlas.chartop.data import ENVELOPE
            return cut_strip(spec, n, along=kw.get("along"), envelope=ENVELOPE)
        except CutError as exc:
            tried.append(f"strip: {exc}")
    else:
        tried.append("a domain with holes")
    raise CutError("this domain cannot be cut automatically (" + "; ".join(tried) + "): cut it "
                   "by hand, or start from a plate, a bend, a channel or a plate with an insert")


# ---------------------------------------------------------------------------
# hand cutting: chords
# ---------------------------------------------------------------------------


def _same_curve(a: ch.Side, b: ch.Side, tol: float) -> bool:
    return (np.hypot(*(a.start - b.end)) < tol and np.hypot(*(a.end - b.start)) < tol
            and np.hypot(*(a(0.5) - b(0.5))) < tol)


def neighbour(pieces, i: int, s: int, tol: float):
    """(j, t) the piece across side s of piece i, and its side, or None."""
    me = pieces[i].sides[s]
    for j, p in enumerate(pieces):
        if j == i:
            continue
        for t, other in enumerate(p.sides):
            if _same_curve(me, other, tol):
                return j, t
    return None


def _sub(side: ch.Side, t0: float, t1: float) -> ch.Side:
    tt = np.linspace(t0, t1, 1025)
    return ch.side_from_polyline(side(tt), dense=1025)


def split(cut: Cut, i: int, s: int, t: float) -> Cut:
    """Cut piece ``i`` from the point at fraction ``t`` of its side ``s`` straight across
    to the opposite side, and on through every piece across those sides: a chord, so
    every seam stays conforming.  The new sides carry the chord's id for `merge`."""
    if not 0.02 < t < 0.98:
        raise CutError("cut nearer the middle of the side: within 2% of a corner")
    pieces = list(cut.pieces)
    diam = float(np.ptp(np.concatenate([p.corners() for p in pieces]), axis=0).max())
    tol = 1e-7 * diam
    used = {c for p in pieces for c in p.chords.values()}
    k = 1
    while f"c{k}" in used:
        k += 1
    cid = f"c{k}"
    # each piece's own fraction along its side 0 (axis 0) or side 1 (axis 1)
    todo = [(i, s % 2, t if s < 2 else 1.0 - t)]
    seen, replaced = set(), {}
    while todo:
        j, axis, frac = todo.pop()
        if j in seen:
            continue
        seen.add(j)
        p = pieces[j]
        S = p.sides
        tag = dict(p.chords)
        a, b = (0, 2) if axis == 0 else (1, 3)
        P0 = S[a](frac)
        P1 = S[b](1.0 - frac)
        cs = ch.line(P0, P1)
        if axis == 0:
            first = [_sub(S[0], 0.0, frac), cs, _sub(S[2], 1.0 - frac, 1.0), S[3]]
            first_tags = {q: tag[q] for q in (0, 2, 3) if q in tag} | {1: cid}
            second = [_sub(S[0], frac, 1.0), S[1], _sub(S[2], 0.0, 1.0 - frac), cs.reversed()]
            second_tags = {q: tag[q] for q in (0, 1, 2) if q in tag} | {3: cid}
        else:
            # the lower piece runs P0 -> P1 along the cut; the upper one P1 -> P0
            first = [S[0], _sub(S[1], 0.0, frac), cs, _sub(S[3], 1.0 - frac, 1.0)]
            first_tags = {q: tag[q] for q in (0, 1, 3) if q in tag} | {2: cid}
            second = [cs.reversed(), _sub(S[1], frac, 1.0), S[2], _sub(S[3], 0.0, 1.0 - frac)]
            second_tags = {q: tag[q] for q in (1, 2, 3) if q in tag} | {0: cid}
        replaced[j] = [Piece(first, p.material, p.label + "a", "tfi", None, "tfi", p.expert, first_tags),
                       Piece(second, p.material, p.label + "b", "tfi", None, "tfi", p.expert, second_tags)]
        for side, f in ((a, frac), (b, 1.0 - frac)):
            nb = neighbour(pieces, j, side, tol)
            if nb is not None:
                q, qs = nb
                # the shared side runs the other way in the neighbour
                g = 1.0 - f
                todo.append((q, qs % 2, g if qs < 2 else 1.0 - g))
    out = []
    for j, p in enumerate(pieces):
        out.extend(replaced.get(j, [p]))
    return Cut(cut.cutter + "+hand", dict(cut.params), cut.n, out,
               cut.notes + [f"chord {cid} cut through {len(replaced)} pieces"])


def chords(cut: Cut) -> list:
    """The hand-cut chords, by id."""
    return sorted({c for p in cut.pieces for c in p.chords.values()})


def merge(cut: Cut, cid: str) -> Cut:
    """Remove a hand-cut chord: every pair of pieces across it becomes one piece."""
    pieces = list(cut.pieces)
    diam = float(np.ptp(np.concatenate([p.corners() for p in pieces]), axis=0).max())
    tol = 1e-7 * diam
    on = [(j, s) for j, p in enumerate(pieces) for s, c in p.chords.items() if c == cid]
    if not on:
        raise CutError(f"no chord {cid}: only hand cuts can be merged")
    done, merged = set(), {}
    for j, sj in on:
        if j in done:
            continue
        nb = neighbour(pieces, j, sj, tol)
        if nb is None:
            raise CutError(f"chord {cid} lies on the boundary at {pieces[j].label}")
        k, sk = nb
        A, B = pieces[j], pieces[k]
        side = lambda P, q: P.sides[q % 4]
        sides = [side(A, sj + 2), _join(side(A, sj + 3), side(B, sk + 1)), side(B, sk + 2),
                 _join(side(B, sk + 3), side(A, sj + 1))]
        tags = {}
        for q, (P, r) in enumerate(((A, sj + 2), (A, sj + 3), (B, sk + 2), (B, sk + 3))):
            if (r % 4) in P.chords:
                tags[q] = P.chords[r % 4]
        lab = A.label[:-1] if A.label.endswith(("a", "b")) else A.label
        rot = _start_index(sides)
        sides = sides[rot:] + sides[:rot]
        tags = {(q - rot) % 4: c for q, c in tags.items()}
        merged[j] = Piece(sides, A.material, lab, "tfi", None, "tfi", A.expert, tags)
        merged[k] = None
        done.update({j, k})
    out = []
    for j, p in enumerate(pieces):
        if j in merged:
            if merged[j] is not None:
                out.append(merged[j])
        else:
            out.append(p)
    return Cut(cut.cutter + "+hand", dict(cut.params), cut.n, out,
               cut.notes + [f"chord {cid} removed: {len(merged) // 2} pairs merged"])


def _straight(side: ch.Side, tol: float) -> bool:
    a, b = side.start, side.end
    d = b - a
    L = float(np.hypot(*d))
    if L <= 0:
        return False
    cross = (side.P[:, 0] - a[0]) * d[1] - (side.P[:, 1] - a[1]) * d[0]
    return bool(np.abs(cross).max() <= tol * L)


def move_corner(cut: Cut, old, new) -> Cut:
    """Drag an interior corner of the layout to ``new``: every side meeting it, all of
    them straight cuts shared by two pieces, becomes the straight line from its other end
    to the new point, in both pieces alike (so the seams stay conforming).  A corner on
    the domain's boundary, or on a curved side, stays where it is."""
    pieces = list(cut.pieces)
    diam = float(np.ptp(np.concatenate([p.corners() for p in pieces]), axis=0).max())
    tol = 1e-7 * diam
    old, new = np.asarray(old, float), np.asarray(new, float)
    at = [(j, k) for j, p in enumerate(pieces) for k in range(4)
          if np.hypot(*(p.sides[k].start - old)) < 1e-6 * diam]
    if not at:
        raise CutError("no corner of a piece there")
    old = pieces[at[0][0]].sides[at[0][1]].start.copy()
    for j, k in at:
        for side in (k, (k + 3) % 4):              # the side leaving the corner and the one arriving
            if neighbour(pieces, j, side, tol) is None:
                raise CutError("that corner is on the domain's boundary: only an inside corner moves")
            if not _straight(pieces[j].sides[side], 1e-9):
                raise CutError("a curved side meets that corner: only corners where straight cuts meet move")
    out = list(pieces)
    for j, k in at:
        p = pieces[j]
        S = list(p.sides)
        leave, arrive = S[k], S[(k + 3) % 4]
        S[k] = ch.line(new, leave.end)
        S[(k + 3) % 4] = ch.line(arrive.start, new)
        out[j] = Piece(S, p.material, p.label, "tfi", None, p.method if p.method != "own" else "tfi",
                       p.expert, dict(p.chords))
    return Cut(cut.cutter + "+hand", dict(cut.params), cut.n, out,
               cut.notes + [f"a corner moved from ({old[0]:.4g}, {old[1]:.4g}) to ({new[0]:.4g}, {new[1]:.4g})"])


def _join(a: ch.Side, b: ch.Side) -> ch.Side:
    return ch.side_from_polyline(np.vstack([a.P, b.P[1:]]))


def _start_index(sides) -> int:
    """Which side to start the loop with: the one starting at the smallest x + y."""
    return min(range(4), key=lambda q: float(sides[q].start[0] + sides[q].start[1]))


# ---------------------------------------------------------------------------
# charts, certificates, the coupled solve
# ---------------------------------------------------------------------------


def chart_of(p: Piece, n: int) -> ch.Chart:
    m = p.own if p.method == "own" else p.method
    if p.method == "own" and p.own_chart is not None and p.own_chart.n == n:
        return p.own_chart
    if (m, n) not in p.cache:
        p.cache[(m, n)] = _make_chart(p, m, n)
    return p.cache[(m, n)]


def _make_chart(p: Piece, m: str, n: int) -> ch.Chart:
    if m in ("tfi", "transfinite"):
        return ch.Chart(ch.transfinite(p.sides, n), "transfinite")
    if m in ("laplacian", "winslow-arc-length"):
        try:
            X, info = ch.winslow(p.sides, n)
            return ch.Chart(X, "winslow-arc-length", info)
        except ch.WinslowDiverged:
            return ch.Chart(ch.harmonic(p.sides, n), "harmonic")
    if m in ("conformal", "conformal-numerical", "conformal-harmonic", "conformal-log-polar"):
        from .conformal import conformal_chart
        return conformal_chart(p.sides, n)
    if m == "affine":
        return ch.Chart(ch.transfinite(p.sides, n), "affine")
    raise CutError(f"no chart method {m!r}")


def certificate(c: ch.Chart, log_kappa_hat=None, fo=None, envelope=None) -> dict:
    """Proposition 3's certificate, and the envelope's quantities and refusals."""
    ce = cert.certify(c.X)
    jc = cert.corner_jacobians(c.X)
    # relative to the mean's size, so a folded piece reads negative
    rel = float(jc.min()) / max(abs(float(jc.mean())), 1e-300)
    out = {"ok": bool(ce.ok), "min_detJ_rel": rel,
           "nonpositive_corners": int(ce.nonpositive_corners), "simple": bool(ce.boundary_simple),
           "reason": ce.reason}
    try:
        m = cert.measure(c, log_kappa_hat, fo)
        out["measure"] = {k: (None if v is None else float(v)) for k, v in m.items()}
        if envelope is not None:
            out["refusals"] = cert.refusals(m, envelope)
    except Exception as exc:                      # a folded chart has no measure
        out["measure"] = None
        out["refusals"] = [f"no envelope measure: {type(exc).__name__}"]
    return out


def problems(cut: Cut, spec) -> list:
    """The pieces as the host's problems: each chart, material and step."""
    mats = _materials(spec)
    steady = spec.run.mode == "steady"
    dt = float(spec.run.macro_dt)
    T0 = float(spec.physics.params.get("T0", 300.0))
    out = []
    for p in cut.pieces:
        c = chart_of(p, cut.n)
        m = mats.get(p.material)
        if m is None or "k" not in m:
            raise CutError(f"{p.label}: its material {p.material!r} has no conductivity")
        el = c.element_fields()
        area = el["area"]
        if steady:
            out.append(PieceProblem(c, fo=None, kappa_ref=float(m["k"]), label=p.label, meta={"area": area}))
        else:
            fo = float(m["k"]) / (float(m["rho"]) * float(m["cp"])) * dt / area
            g = np.full((c.n + 1, c.n + 1), T0 / fo)
            out.append(PieceProblem(c, fo=fo, g=g, kappa_ref=float(m["k"]), label=p.label,
                                    meta={"area": area}))
    return out


def seams(layout) -> list:
    """Every seam and outer side as the host sees it, with its verdict."""
    out = []
    for e in layout.edges:
        i, s = e.owner
        P = layout.problems[i].chart
        side = {0: P.X[0, :], 1: P.X[:, -1], 2: P.X[-1, ::-1], 3: P.X[::-1, 0]}[s]
        if e.other is None:
            kind = e.bc[0] if e.bc is not None else "insulated"
            out.append({"pieces": [i], "side": s, "verdict": kind, "points": side[:: max(1, len(side) // 16)].tolist() + [side[-1].tolist()]})
        else:
            verdict = "admit" if e.conforming else "admit-uncertified"
            out.append({"pieces": [i, e.other[0]], "side": s, "verdict": verdict, "port": "THERM",
                        "why": ("conforming: the two pieces' nodes coincide (Theorem 2)" if e.conforming
                                else "mortar: the nodes differ, a nonconforming coupling Theorem 2 does not cover"),
                        "points": side[:: max(1, len(side) // 16)].tolist() + [side[-1].tolist()]})
    return out


def solve_classical(cut: Cut, spec, m: int | None = None) -> dict:
    """The coupled classical solve against the monolithic one, and the seams.

    ``m``: port modes per side.  With every side's trace space whole (``m = n``, hats
    and n - 1 sines span a side's n + 1 nodes) the coupled answer is the monolithic one
    to round-off on conforming seams (Theorem 2); the experts' 16 modes leave a
    truncation floor, which `layout_floor` reports beside it."""
    probs = problems(cut, spec)
    L = hs.Layout(probs, make_bc(spec), kind=KIND, m=m or M)
    cp = L.solve(hs.ClassicalExpert(), refuse_outside=False)
    out = {"pieces": len(probs), "seams": seams(L), "min_eig": cp.min_eig()}
    try:
        mono = hs.monolithic(L)
        diff = max(float(np.abs(a - b).max()) for a, b in zip(cp.fields, mono.fields))
        span = max(float(np.ptp(np.concatenate(mono.fields))), 1e-300)
        num, den = cp.energy_error(mono.fields)
        out.update(monolithic_nodes=mono.nodes, max_diff=diff, max_diff_rel=diff / span,
                   energy_rel=num / den if den > 0 else 0.0, span=span)
    except hs.Refused as exc:
        out["monolithic"] = f"refused: {exc}"
    out["_coupled"] = cp
    out["_layout"] = L
    return out


__all__ = ["CUTTERS", "Cut", "CutError", "METHODS", "Piece", "auto_cut", "certificate",
           "chart_of", "chords", "cut_grid", "cut_insert", "cut_polar", "cut_strip", "make_bc",
           "merge", "move_corner", "problems", "seams", "solve_classical", "split"]
