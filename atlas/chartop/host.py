"""The tier-0 host: Gram assembly, one interface solve, reconstruction.

Given each piece's constraint modes ``H_i`` (for its port basis ``Q``) and particular
field ``u_p,i`` -- from the network or from an exact classical solve -- the host forms

    Lambda_i = V_i^T M_i V_i,  V_i = (H_i ; Q),
    b_i = H_i^T b_I + Q^T b_G - H_i^T A_I u_p - Q^T M_GI u_p,

assembles ``S = sum_i R_i^T kappa_i Lambda_i R_i`` and ``chi = sum_i R_i^T kappa_i b_i``,
fixes the Dirichlet coefficients, solves once, and rebuilds ``u_i = u_p,i + H_i R_i c``
with trace ``Q R_i c``.  No further call to the expert (SE2): one response per piece
per linear solve.  Theorem 1 (T25, `Atlas.Superelement.hostMatrix_posDef`): ``S`` is
positive semidefinite for any modes with exact traces, ``Lambda_i - Lambda_i^exact =
E_i^T A_I E_i``, and ``S >= S^exact``, so ``lambda_min`` can only rise.

**The interface unknowns** (corner hats plus side sines, decision D3): one value per
**vertex** of the layout -- a corner shared by any number of pieces, so cross-points
need nothing special -- and ``m - 1`` sine amplitudes per **edge**, in the orientation
of the edge's first piece.  A conforming neighbour, whose side nodes coincide with the
owner's, reads the amplitudes with the reversal sign ``(-1)^(j+1)``.

**The mortar.**  When the two pieces' nodes on a shared side do not coincide
(different resolutions or parametrisations), the second piece's sine amplitudes are
the L2 projection, in physical arc length along the side, of the owner's trace minus
the second piece's own vertex interpolant.  The vertex values stay shared.  That is
a nonconforming coupling, which Theorem 2 does not cover; it is reported, not gated.

**Typed ports.**  Every CO-D piece declares ``THERM`` (temperature with heat flux); a
seam is admitted only where both pieces declare it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import cKDTree

from . import certificate as cert
from . import fe
from . import ports as pt
from .piece import PieceProblem, exact_response

PORT_TYPE = "THERM"


@dataclass
class EdgeInfo:
    owner: tuple[int, int]
    other: tuple[int, int] | None
    conforming: bool
    bc: tuple | None = None          # ("dirichlet", values (n+1,)) or ("insulated", None)


@dataclass
class Response:
    """What an expert returns for one piece: interior modes (n-1)^2 x 4m and the
    interior particular field (n-1)^2; the kernel-exact forms (`forms`) add their level
    field (n-1)^2, the answer for the unit uniform source, which a march keeps with the
    modes."""
    H: np.ndarray
    up: np.ndarray
    level: np.ndarray | None = None


class Refused(Exception):
    pass


class ClassicalExpert:
    """Exact classical piece solves: the classical superelement (gate E3's reference)."""
    name = "classical"
    envelope = None

    def __init__(self):
        self.calls = 0
        self.factorisations = 0

    def respond(self, problems: list[PieceProblem], pb_of) -> list[Response]:
        out = []
        for p in problems:
            ex = exact_response(p, pb_of(p.n).Q)
            out.append(Response(ex.H, ex.up))
            self.calls += 1
            self.factorisations += 1
        return out


def _side_points(p: PieceProblem, s: int) -> np.ndarray:
    n = p.n
    pos = pt.side_positions(n, s)
    return p.chart.loop_points()[pos]


class Layout:
    """Pieces, their seams and their boundary conditions, with the global numbering.

    ``bc(i, s, points)`` is called for every side that no other piece shares and
    returns ``("dirichlet", values)`` with one value per side node, or
    ``("insulated", None)``."""

    def __init__(self, problems: list[PieceProblem], bc, kind: str = "sine-hat",
                 m: int = 16, tol: float = 1e-7, ports_of=None):
        self.problems = problems
        self.kind = kind
        self.m = m
        self.P = len(problems)
        self.ports = ports_of or (lambda i: (PORT_TYPE,))
        pts = [p.chart.loop_points() for p in problems]
        diam = float(np.ptp(np.concatenate(pts), axis=0).max())
        self.diam = diam
        self.tol = tol * diam
        self._vertices()
        self._edges(bc)
        self._numbering()

    # -- topology -----------------------------------------------------------------

    def pb(self, n: int) -> pt.PortBasis:
        return pt.basis(self.kind, n, self.m)

    def _vertices(self):
        corners = np.array([p.chart.corners() for p in self.problems]).reshape(-1, 2)
        tree = cKDTree(corners)
        vid = -np.ones(len(corners), int)
        nv = 0
        for k in range(len(corners)):
            if vid[k] >= 0:
                continue
            for q in tree.query_ball_point(corners[k], self.tol):
                vid[q] = nv
            nv += 1
        self.vertex_of = vid.reshape(self.P, 4)           # [piece, corner]
        self.nv = nv
        self.vertex_xy = np.zeros((nv, 2))
        self.vertex_xy[vid] = corners

    def _edges(self, bc):
        self.edges: list[EdgeInfo] = []
        self.edge_of = {}
        for i in range(self.P):
            for s in range(4):
                if (i, s) in self.edge_of:
                    continue
                a, b = self.vertex_of[i, s], self.vertex_of[i, (s + 1) % 4]
                Pi = _side_points(self.problems[i], s)
                partner = None
                for j in range(self.P):
                    for t in range(4):
                        if (j, t) == (i, s) or (j, t) in self.edge_of:
                            continue
                        if self.vertex_of[j, t] == b and self.vertex_of[j, (t + 1) % 4] == a:
                            Pj = _side_points(self.problems[j], t)
                            mid_i = Pi[len(Pi) // 2]
                            d = np.min(np.hypot(*(Pj - mid_i).T))
                            if d < 0.05 * np.hypot(*(Pi[-1] - Pi[0])) + 1e3 * self.tol:
                                partner = (j, t)
                                Pj_match = Pj
                if partner is None:
                    kind_v = bc(i, s, Pi)
                    e = EdgeInfo((i, s), None, True, kind_v)
                else:
                    j, t = partner
                    if not (set(self.ports(i)) & set(self.ports(j))):
                        raise Refused(f"pieces {i} and {j} share a side but no port type")
                    conf = (len(Pj_match) == len(Pi)
                            and float(np.max(np.hypot(*(Pj_match[::-1] - Pi).T))) < self.tol)
                    e = EdgeInfo((i, s), (j, t), conf, None)
                    if not conf and self.kind != "sine-hat":
                        raise Refused("the cosine arm needs conforming seams (no mortar)")
                k = len(self.edges)
                self.edges.append(e)
                self.edge_of[(i, s)] = k
                if partner is not None:
                    self.edge_of[partner] = k

    def _numbering(self):
        m = self.m
        per_edge = (m - 1) if self.kind == "sine-hat" else m
        self.per_edge = per_edge
        self.v_off = 0
        nvd = self.nv if self.kind == "sine-hat" else 0
        self.e_off = nvd
        self.N = nvd + len(self.edges) * per_edge
        self.R = [self._restriction(i) for i in range(self.P)]
        self._fixed()

    def edge_dofs(self, e: int) -> np.ndarray:
        return self.e_off + e * self.per_edge + np.arange(self.per_edge)

    def _restriction(self, i: int) -> np.ndarray:
        p = self.problems[i]
        pbi = self.pb(p.n)
        R = np.zeros((pbi.size, self.N))
        for k, col in enumerate(pbi.columns):
            if col.what == "vertex":
                R[k, self.v_off + self.vertex_of[i, col.index]] = 1.0
        for s in range(4):
            e = self.edge_of[(i, s)]
            info = self.edges[e]
            cols = pbi.side_columns(s)
            dofs = self.edge_dofs(e)
            if info.owner == (i, s):
                for kk, k in enumerate(cols):
                    R[k, dofs[kk]] = 1.0
            elif info.conforming:
                for kk, k in enumerate(cols):
                    R[k, dofs[kk]] = pbi.reversal_sign(k)
            else:
                R[cols] = self._mortar_rows(i, s, info, cols)
        return R

    def _mortar_rows(self, i, s, info: EdgeInfo, cols) -> np.ndarray:
        """Rows of piece i's side-s sine columns as combinations of global dofs: the
        L2 projection in physical arc length of the owner's trace on the shared side,
        minus piece i's own vertex interpolant."""
        oi, os_ = info.owner
        po, pi = self.problems[oi], self.problems[i]
        pbo, pbi = self.pb(po.n), self.pb(pi.n)
        Xo = _side_points(po, os_)                     # owner's direction
        Xi = _side_points(pi, s)[::-1]                 # piece i's nodes, owner's direction
        so = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(Xo, axis=0).T))])
        Lo = so[-1]
        # piece i's nodes located along the owner's polyline by projection
        si = _project_onto(Xo, so, Xi)
        si[0], si[-1] = 0.0, Lo
        K = 8 * max(po.n, pi.n) + 1
        sq = np.linspace(0.0, Lo, K)
        w = np.full(K, Lo / (K - 1))
        w[0] *= 0.5
        w[-1] *= 0.5
        posO = pt.side_positions(po.n, os_)
        posI = pt.side_positions(pi.n, s)[::-1]
        # the owner's trace on this side in terms of global dofs: only its two corner
        # hats and its own side columns are non-zero here, and all map to plain dofs
        RO = np.zeros((pbo.size, self.N))
        for k, col in enumerate(pbo.columns):
            if col.what == "vertex" and col.index in (os_, (os_ + 1) % 4):
                RO[k, self.v_off + self.vertex_of[oi, col.index]] = 1.0
        for kk, k in enumerate(pbo.side_columns(os_)):
            RO[k, self.edge_dofs(self.edge_of[(oi, os_)])[kk]] = 1.0
        PhiO = np.column_stack([np.interp(sq, so, pbo.Q[posO, k]) for k in range(pbo.size)])
        TO = PhiO @ RO                                  # K x N
        # piece i's vertex interpolant on this side (its two hats)
        PhiI = np.column_stack([np.interp(sq, si, pbi.Q[posI, k]) for k in range(pbi.size)])
        Ri_v = np.zeros((pbi.size, self.N))
        for k, col in enumerate(pbi.columns):
            if col.what == "vertex":
                Ri_v[k, self.v_off + self.vertex_of[i, col.index]] = 1.0
        target = TO - PhiI @ Ri_v
        S = PhiI[:, cols]
        G = S.T @ (w[:, None] * S)
        return np.linalg.solve(G, S.T @ (w[:, None] * target))

    def _fixed(self):
        fixed: dict[int, float] = {}
        for e, info in enumerate(self.edges):
            if info.other is not None or info.bc is None or info.bc[0] != "dirichlet":
                continue
            i, s = info.owner
            pbi = self.pb(self.problems[i].n)
            coef = pt.project_side(pbi, s, info.bc[1])
            R = self.R[i]
            for k, v in coef.items():
                dof = int(np.flatnonzero(R[k])[0]) if np.count_nonzero(R[k]) == 1 else None
                if dof is None:
                    raise Refused("a Dirichlet side whose coefficients are not plain dofs")
                if dof in fixed and abs(fixed[dof] - v) > 1e-9 * max(1.0, abs(v)):
                    raise Refused(f"vertex dof {dof} fixed twice: {fixed[dof]} and {v}")
                fixed[dof] = v * R[k, dof]
        self.fixed = fixed

    # -- the coupled solve --------------------------------------------------------------

    def check_envelope(self, envelope: cert.Envelope | None) -> list[str]:
        if envelope is None:
            return []
        out = []
        for i, p in enumerate(self.problems):
            m = cert.measure(p.chart, p.log_kappa_hat, None if p.steady else p.fo)
            for r in cert.refusals(m, envelope):
                out.append(f"piece {i} ({p.label}): {r}")
        return out

    def solve(self, expert, refuse_outside: bool = True) -> "Coupled":
        refusals = self.check_envelope(getattr(expert, "envelope", None))
        if refusals and refuse_outside:
            raise Refused("; ".join(refusals))
        calls0 = getattr(expert, "calls", 0)
        resp = expert.respond(self.problems, self.pb)
        calls = getattr(expert, "calls", 0) - calls0
        return self.couple(resp, calls=calls, refusals=refusals)

    def couple(self, resp: list[Response], calls: int | None = None,
               refusals: list[str] | None = None) -> "Coupled":
        N = self.N
        S = np.zeros((N, N))
        chi = np.zeros(N)
        grams, loads = [], []
        for i, (p, r) in enumerate(zip(self.problems, resp)):
            pbi = self.pb(p.n)
            M = p.matrix()
            loop, inner = fe.grid(p.n)
            V = fe.full_field(p.n, r.H, pbi.Q)
            Lam = fe.gram(M, V)
            b = p.load()
            Up = fe.full_field(p.n, r.up, np.zeros(4 * p.n))
            bt = V.T @ b - V.T @ (M @ Up)
            grams.append(Lam)
            loads.append(bt)
            Ri = self.R[i]
            S += p.kappa_ref * (Ri.T @ Lam @ Ri)
            chi += p.kappa_ref * (Ri.T @ bt)
        S = 0.5 * (S + S.T)
        fixed = np.array(sorted(self.fixed), int)
        free = np.setdiff1d(np.arange(N), fixed)
        c = np.zeros(N)
        c[fixed] = [self.fixed[d] for d in fixed]
        Sff = S[np.ix_(free, free)]
        rhs = chi[free] - S[np.ix_(free, fixed)] @ c[fixed]
        cf = sla.cho_solve(sla.cho_factor(Sff), rhs)
        c[free] = cf
        fields = []
        for i, (p, r) in enumerate(zip(self.problems, resp)):
            ci = self.R[i] @ c
            pbi = self.pb(p.n)
            fields.append(fe.full_field(p.n, r.up + r.H @ ci, pbi.Q @ ci))
        return Coupled(self, c, S, chi, free, fields, grams, loads, resp, calls,
                       refusals or [])


def _project_onto(P: np.ndarray, s: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Arc-length positions along polyline P (cumulative lengths s) of the nearest
    points to X."""
    a, ab = P[:-1], np.diff(P, axis=0)
    L2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-300)
    out = np.empty(len(X))
    for k, x in enumerate(X):
        t = np.clip(((x - a) * ab).sum(1) / L2, 0.0, 1.0)
        near = a + t[:, None] * ab
        d = np.hypot(*(near - x).T)
        q = int(np.argmin(d))
        out[k] = s[q] + t[q] * math.sqrt(L2[q])
    return out


@dataclass
class Coupled:
    layout: Layout
    c: np.ndarray
    S: np.ndarray
    chi: np.ndarray
    free: np.ndarray
    fields: list
    grams: list
    loads: list
    responses: list
    calls: int | None
    refusals: list = field(default_factory=list)

    def min_eig(self) -> float:
        Sff = self.S[np.ix_(self.free, self.free)]
        return float(np.linalg.eigvalsh(Sff)[0])

    def energy_error(self, ref_fields: list) -> tuple[float, float]:
        """(||u - u~||_M, ||u||_M) summed over pieces with each piece's kappa_ref."""
        num = den = 0.0
        for p, u, ut in zip(self.layout.problems, ref_fields, self.fields):
            M = p.matrix()
            e = u - ut
            num += p.kappa_ref * float(e @ (M @ e))
            den += p.kappa_ref * float(u @ (M @ u))
        return math.sqrt(max(num, 0.0)), math.sqrt(max(den, 0.0))


# ---------------------------------------------------------------------------
# the full-domain solve in the same discretisation (conforming layouts)
# ---------------------------------------------------------------------------


@dataclass
class Monolithic:
    fields: list
    nodes: int
    global_index: list


def monolithic(layout: Layout) -> Monolithic:
    """Merge coincident nodes across pieces into one bilinear mesh, assemble every
    piece's kappa_ref M_i, impose the Dirichlet sides' nodal data, solve once."""
    for e in layout.edges:
        if e.other is not None and not e.conforming:
            raise Refused("the monolithic reference needs conforming seams")
    pts, owners = [], []
    for i, p in enumerate(layout.problems):
        pts.append(p.chart.X.reshape(-1, 2))
        owners.append(np.full(len(pts[-1]), i))
    allp = np.concatenate(pts)
    tree = cKDTree(allp)
    gid = -np.ones(len(allp), int)
    ng = 0
    for k in range(len(allp)):
        if gid[k] >= 0:
            continue
        for q in tree.query_ball_point(allp[k], layout.tol):
            gid[q] = ng
        ng += 1
    offs = np.cumsum([0] + [len(x) for x in pts])
    G = [gid[offs[i]:offs[i + 1]] for i in range(layout.P)]
    rows, cols, vals = [], [], []
    rhs = np.zeros(ng)
    for i, p in enumerate(layout.problems):
        M = p.matrix().tocoo()
        rows.append(G[i][M.row]); cols.append(G[i][M.col]); vals.append(p.kappa_ref * M.data)
        np.add.at(rhs, G[i], p.kappa_ref * p.load())
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(ng, ng))
    dval = {}
    for e in layout.edges:
        if e.other is None and e.bc is not None and e.bc[0] == "dirichlet":
            i, s = e.owner
            pos = pt.side_positions(layout.problems[i].n, s)
            loop = fe.grid(layout.problems[i].n)[0]
            for q, v in zip(loop[pos], e.bc[1]):
                dval[int(G[i][q])] = float(v)
    dn = np.array(sorted(dval), int)
    fn = np.setdiff1d(np.arange(ng), dn)
    u = np.zeros(ng)
    u[dn] = [dval[d] for d in dn]
    Kff = K[fn][:, fn].tocsc()
    u[fn] = spla.splu(Kff).solve(rhs[fn] - K[fn][:, dn] @ u[dn])
    return Monolithic([u[G[i]] for i in range(layout.P)], ng, G)
