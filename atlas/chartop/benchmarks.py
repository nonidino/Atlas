"""The three held-out benchmarks, as layouts of four-sided pieces with conforming seams.

The workbench's examples `bend-3`, `s-channel` and `insert-round`
(`atlas/workbench/spec.py`, EXAMPLES) are the benchmarks of gates E3-E5, kept out of
training ([[chart-operator-training-and-cost]] section 6).  Their outlines and
materials are read from the workbench; the block layouts below are fixed by hand,
because the general layout for forks and rings is not built
([[chart-operator-design-decisions]] section 3.3).  Every seam is conforming: the two
pieces' nodes on a shared side coincide.

- **bend-3**: a quarter ring of steel, radii 0.2 and 0.5 m, the end along x at 400 K
  and the end along y at 300 K, the arcs insulated.  ``n_r x n_theta`` pieces cut at
  equal values of ``log r`` and ``theta``, charted by the log-polar map (conformal,
  every corner a right angle, constant real mu).  The steady solution is
  ``T = 400 - 100 theta / (pi/2)``, which bilinear elements on these charts reproduce
  exactly.
- **s-channel**: the drawn S-shaped channel (splines), the left end at 400 K, the
  right at 300 K, the walls insulated.  The layout's harmonic coordinates ``(u, v)``
  are solved on a fine Coons mesh of the whole channel; ``round(M)`` pieces are cut at
  equal values of ``u`` (decision 2) and charted conformally (the harmonic chart with
  the conformal boundary correspondence: walls by ``u``, cuts by ``v``).  The two end
  pieces, with their 68 and 148 degree corners, go by the chart rule: the conformal
  chart is tried first and refused by the envelope's ``log J`` range, so they are
  charted by Winslow with arc-length data, keeping the shared cut's ``v``
  parametrisation.  The steady solution is ``T = 400 - 100 u``.
- **insert-round**: a steel plate 0.4 x 0.25 m with a copper disc of radius
  62.5 mm, left edge 400 K, right edge 300 K, top and bottom insulated.  Eleven
  pieces: the disc as a core square and four ring pieces (corners 135, 135, 90, 90
  degrees), the plate as four ring pieces between the circle and a square box of
  half-width 0.125 m (corners 90, 90, 45, 45 degrees) and two side columns.  Ring
  pieces are charted by Winslow, the square and the columns by affine maps.  Every
  piece is one material, so ``log kappa_hat = 0`` on every piece and the contrast
  (400 / 45) enters through the host's ``kappa_ref`` only.

Each benchmark comes in two problems: **steady**, as the workbench defines it, and
**step**, one backward-Euler step of ``DT[name]`` seconds from 300 K everywhere with
the same boundary conditions (registered here, before any training), so that the
particular field and the Fourier number are exercised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import certificate as cert
from . import chart as ch
from . import fe
from .host import Layout
from .piece import PieceProblem

NAMES = ("bend-3", "s-channel", "insert-round")
#: the step benchmarks' time step, seconds (per-piece Fourier numbers 0.02-3)
DT = {"bend-3": 50.0, "s-channel": 100.0, "insert-round": 100.0}
T_OLD = 300.0
T_HOT, T_COLD = 400.0, 300.0


@dataclass
class Bench:
    name: str
    problem: str                     # "steady" or "step"
    problems: list
    bc: object
    meta: dict = field(default_factory=dict)

    def layout(self, kind: str = "sine-hat", m: int = 16) -> Layout:
        return Layout(self.problems, self.bc, kind=kind, m=m)


def _material(name: str) -> dict:
    from atlas.workbench import registry
    return dict(registry.family("conduction-2d").material_library()[name])


def _alpha(mat: dict) -> float:
    return mat["k"] / (mat["rho"] * mat["cp"])


def _finish(c: ch.Chart, mat: dict, problem: str, dt: float, label: str) -> PieceProblem:
    el = c.element_fields()
    area = el["area"]
    n = c.n
    if problem == "steady":
        return PieceProblem(c, fo=None, kappa_ref=mat["k"], label=label,
                            meta={"area": area})
    fo = _alpha(mat) * dt / area
    g = np.full((n + 1, n + 1), T_OLD / fo)
    return PieceProblem(c, fo=fo, g=g, kappa_ref=mat["k"], label=label,
                        meta={"area": area})


def _bc_by(predicates):
    """bc(i, s, points): the first predicate (test(points) -> bool, value) that holds
    gives a Dirichlet side; otherwise insulated."""
    def bc(i, s, P):
        for test, value in predicates:
            if test(P):
                return ("dirichlet", np.full(len(P), value))
        return ("insulated", None)
    return bc


# ---------------------------------------------------------------------------
# bend-3
# ---------------------------------------------------------------------------


def bend3(n: int = 64, problem: str = "steady", n_r: int = 3, n_th: int = 5) -> Bench:
    from atlas.workbench import spec as wspec
    sp_ = wspec.example_case("bend-3")
    dx = float(sp_.domain.dx)
    cx, cy = 4.0 * dx, 4.0 * dx
    r0, r1 = 40.0 * dx, 100.0 * dx
    rho0, rho1 = math.log(r0), math.log(r1)
    dr, dth = (rho1 - rho0) / n_r, (math.pi / 2) / n_th
    mat = _material("steel")
    probs = []
    for b in range(n_th):
        for a in range(n_r):
            def F(xi, eta, a=a, b=b):
                rr = np.exp(rho0 + (a + xi) * dr)
                th = (b + eta) * dth
                return cx + rr * np.cos(th), cy + rr * np.sin(th)
            c = ch.Chart(ch.from_map(F, n), "conformal-log-polar",
                         {"modulus": dr / dth})
            probs.append(_finish(c, mat, problem, DT["bend-3"], f"bend {a},{b}"))
    tol = 1e-6 * r1

    def hot(P):
        return bool(np.all(np.abs(P[:, 1] - cy) < tol))

    def cold(P):
        return bool(np.all(np.abs(P[:, 0] - cx) < tol))
    bc = _bc_by([(hot, T_HOT), (cold, T_COLD)])

    def exact(x, y):
        return T_HOT - (T_HOT - T_COLD) * np.arctan2(y - cy, x - cx) / (math.pi / 2)
    return Bench("bend-3", problem, probs, bc, {"exact": exact, "dx": dx,
                                               "pieces": (n_r, n_th)})


# ---------------------------------------------------------------------------
# s-channel
# ---------------------------------------------------------------------------


def _rect_q1(X: np.ndarray) -> sp.csr_matrix:
    """Q1 stiffness (identity physical conductivity) on a structured mesh X of shape
    (NY+1, NX+1, 2): element tensors J G^-1 from each element's centre Jacobian."""
    NY, NX = X.shape[0] - 1, X.shape[1] - 1
    d_xi = 0.5 * ((X[:-1, 1:] - X[:-1, :-1]) + (X[1:, 1:] - X[1:, :-1]))
    d_et = 0.5 * ((X[1:, :-1] - X[:-1, :-1]) + (X[1:, 1:] - X[:-1, 1:]))
    a11, a12, a22, J = ch.metric_tensor(np.stack([d_xi, d_et], axis=-1))
    if np.any(J <= 0):
        raise ValueError("the fine channel mesh folds")
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
    N = K.shape[0]
    free = np.setdiff1d(np.arange(N), fixed_idx)
    u = np.zeros(N)
    u[fixed_idx] = fixed_val
    K = K.tocsr()
    u[free] = spla.splu(K[free][:, free].tocsc()).solve(-K[free][:, fixed_idx] @ fixed_val)
    return u


def _channel_sides(dx: float):
    from atlas.workbench import shapes
    from atlas.workbench import spec as wspec
    o = wspec.example_case("s-channel").domain.outline
    edges = shapes.edges_sampled(o.points, o.edges, o.bulge, step=0.02)
    E = [np.asarray(e, float) * dx for e in edges]
    k = len(E) // 2                       # 7: six splines and a line on each half
    bottom = np.vstack([E[0]] + [e[1:] for e in E[1:k - 1]])
    right = E[k - 1]
    top = np.vstack([E[k]] + [e[1:] for e in E[k + 1:2 * k - 1]])
    left = E[2 * k - 1]
    return bottom, right, top, left


def _polyline_at(P: np.ndarray, s_query) -> np.ndarray:
    s = ch.arclength(P)
    return np.stack([np.interp(s_query, s, P[:, 0]), np.interp(s_query, s, P[:, 1])], -1)


def s_channel(n: int = 64, problem: str = "steady", NX: int = 1152, NY: int = 128,
              envelope: cert.Envelope | None = None) -> Bench:
    from atlas.workbench import spec as wspec
    dx = float(wspec.example_case("s-channel").domain.dx)
    bottom, right, top, left = _channel_sides(dx)
    # the fine Coons mesh of the whole channel: xi along (arc-length fraction of each
    # wall), eta across (fraction of each end)
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
    M = 1.0 / Eu
    along = max(1, int(round(M)))
    du = 1.0 / along
    mat = _material("steel")

    # the walls' u-correspondence and the cuts' v-parametrisation
    def wall_point(row: int, u: float) -> np.ndarray:
        uu = np.maximum.accumulate(U[row])
        return np.array([np.interp(u, uu, X[row, :, 0]), np.interp(u, uu, X[row, :, 1])])

    def cut_curve(u: float) -> np.ndarray:
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
        cuts[c_] = ch.Side(P, vs)          # bottom to top, parameter v
    probs, report = [], []
    for c_ in range(along):
        u0, u1 = c_ * du, (c_ + 1) * du
        s0 = side_u(0, u0, u1, rev=False)                            # bottom wall, left to right
        s2 = side_u(NY, u0, u1, rev=True)                            # top wall, right to left
        s1 = cuts[c_ + 1] if c_ + 1 < along else ch.side_from_polyline(right)
        s3 = (cuts[c_].reversed() if c_ > 0 else ch.side_from_polyline(left))
        # ends keep arc length; their corners must meet the walls' correspondence
        sides = [s0, s1, s2, s3]
        Xc = ch.harmonic(sides, n, aniso=M * du)
        chosen = ch.Chart(Xc, "conformal-harmonic", {"modulus": M * du})
        is_end = c_ == 0 or c_ == along - 1
        if is_end:
            m_ = cert.measure(chosen)
            refused = (envelope is not None and cert.refusals(m_, envelope)) or \
                      (envelope is None and m_["logJ_range"] > math.log(10.0))
            report.append({"piece": c_, "conformal": m_, "refused": bool(refused)})
            if refused:
                # Winslow with arc-length data on the walls and the end, v on the cut
                wb = ch.side_from_polyline(np.array([s0(t) for t in np.linspace(0, 1, 2049)]))
                wt = ch.side_from_polyline(np.array([s2(t) for t in np.linspace(0, 1, 2049)]))
                Xw, info = ch.winslow([wb, s1, wt, s3], n)
                chosen = ch.Chart(Xw, "winslow-arc-length", info)
        probs.append(_finish(chosen, mat, problem, DT["s-channel"], f"channel {c_}"))
    xl, xr = left[:, 0].mean(), right[:, 0].mean()
    tol = 1e-6 * float(np.ptp(X[..., 0]))
    bc = _bc_by([(lambda P: bool(np.all(np.abs(P[:, 0] - xl) < tol)), T_HOT),
                 (lambda P: bool(np.all(np.abs(P[:, 0] - xr) < tol)), T_COLD)])
    return Bench("s-channel", problem, probs, bc,
                 {"modulus": M, "duality_Eu_Ev": Eu * Ev, "along": along,
                  "end_piece_charts": report, "dx": dx, "fine_mesh": (NX, NY),
                  "harmonic_u": (X, U)})


# ---------------------------------------------------------------------------
# insert-round
# ---------------------------------------------------------------------------


def _arc(c, R, a0, a1, dense=2049) -> ch.Side:
    t = np.linspace(0.0, 1.0, dense)
    a = a0 + (a1 - a0) * t
    return ch.Side(np.stack([c[0] + R * np.cos(a), c[1] + R * np.sin(a)], -1), t)


def insert_round(n: int = 64, problem: str = "steady", split: int = 3) -> Bench:
    """The O-H layout, every ring cut into ``split`` pieces along its rays so that each
    piece has a modulus near one (decision 2): the core as split x split rectangles,
    4 split disc ring pieces, 4 split plate ring pieces and 2 split column pieces."""
    from atlas.workbench import spec as wspec
    sp_ = wspec.example_case("insert-round")
    dx = float(sp_.domain.dx)
    W, Hh = sp_.domain.nx * dx, sp_.domain.ny * dx
    c = np.array([80.0, 50.0]) * dx
    R = 25.0 * dx
    a = 12.5 * dx                      # the core square's half-width
    b = 50.0 * dx                      # the plate box's half-width
    cu, st = _material("copper"), _material("steel")
    deg = math.pi / 180.0
    S = split
    tq = np.tan((-45.0 + 90.0 * np.arange(S + 1) / S) * deg)     # where the rays meet a side
    probs = []

    def ray_point(phi_deg: float, k: int, dist: float) -> np.ndarray:
        """The point at angle phi on side k of the square of half-width dist."""
        rr = dist / math.cos((phi_deg - 90.0 * k) * deg)
        return c + rr * np.array([math.cos(phi_deg * deg), math.sin(phi_deg * deg)])

    def on_circle(phi_deg: float) -> np.ndarray:
        return c + R * np.array([math.cos(phi_deg * deg), math.sin(phi_deg * deg)])

    # the core: split x split rectangles between the lines c + a t_q, affine charts
    for q in range(S):
        for p in range(S):
            x0, x1 = c[0] + a * tq[p], c[0] + a * tq[p + 1]
            y0, y1 = c[1] + a * tq[q], c[1] + a * tq[q + 1]

            def rect(xi, eta, x0=x0, x1=x1, y0=y0, y1=y1):
                return x0 + (x1 - x0) * xi, y0 + (y1 - y0) * eta
            probs.append(_finish(ch.Chart(ch.from_map(rect, n), "affine"), cu, problem,
                                 DT["insert-round"], f"disc core {p},{q}"))
    # the rings: side k = 0 right, 1 top, 2 left, 3 bottom; rays phi_{k,l}
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
                probs.append(_finish(ch.Chart(Xw, "winslow-arc-length", info), mat,
                                     problem, DT["insert-round"], f"{ring} {k},{l}"))
    # the side columns, cut where the plate ring's rays meet the box: y = c_y + b t_q
    ys = c[1] + b * tq
    for label, x0, x1 in (("left column", 0.0, c[0] - b), ("right column", c[0] + b, W)):
        for q in range(S):
            y0, y1 = ys[q], ys[q + 1]

            def col_map(xi, eta, x0=x0, x1=x1, y0=y0, y1=y1):
                return x0 + (x1 - x0) * xi, y0 + (y1 - y0) * eta
            probs.append(_finish(ch.Chart(ch.from_map(col_map, n), "affine"), st, problem,
                                 DT["insert-round"], f"{label} {q}"))
    tol = 1e-6 * W
    bc = _bc_by([(lambda P: bool(np.all(np.abs(P[:, 0]) < tol)), T_HOT),
                 (lambda P: bool(np.all(np.abs(P[:, 0] - W) < tol)), T_COLD)])
    return Bench("insert-round", problem, probs, bc, {"dx": dx, "materials": {
        "copper": cu, "steel": st}})


def datum_shift(b: Bench, dT: float) -> Bench:
    """The same benchmark with every temperature moved by ``dT``: the Dirichlet data and,
    for a step, ``T_old`` (its source ``g = T_old / Fo`` moves by ``dT / Fo``; a source that
    the shift cancels to rounding is set to zero, so that no expert reads a normalised
    rounding pattern as its source).  The exact field moves by ``dT``, and so does the
    coupled answer of any trial space that reproduces the uniform state exactly
    (`forms`), so its error does not change.  Steady, only the data move."""
    probs = []
    for p in b.problems:
        g = p.g + dT * p.s0
        if p.s0 and float(np.abs(g).max()) <= 1e-12 * float(np.abs(p.g).max()):
            g = np.zeros_like(g)
        probs.append(replace(p, g=g))

    def bc(i, s, P, _bc=b.bc):
        kind, val = _bc(i, s, P)
        return (kind, val + dT) if kind == "dirichlet" else (kind, val)
    return Bench(b.name, b.problem, probs, bc, dict(b.meta, datum_shift=dT))


def build(name: str, n: int = 64, problem: str = "steady", **kw) -> Bench:
    if name == "bend-3":
        return bend3(n, problem, **kw)
    if name == "s-channel":
        return s_channel(n, problem, **kw)
    if name == "insert-round":
        return insert_round(n, problem, **kw)
    raise KeyError(name)
