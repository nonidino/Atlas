"""CO-T on one piece: advection and diffusion of heat on the reference square.

The coolant plan's piece solver ([[chart-operator-stage-0b-and-1-plan]], Phase 1).  The
owner chose model (a) of its section 3 on 2026-11-18: side walls only, so the plate and
the coolant meet only at seams, and every piece is one material, all coolant or all
solid.  The flow is given, not solved.  Nothing here is learned: this is the classical
expert, the reference every later arm of the plan is judged against.

**The pulled-back step.**  Backward Euler for

    rho c (dT/dt + u . grad T) - div(k grad T) = q

on a piece of area |Omega|, pulled back through its chart and divided by k, is

    int grad T . A grad v  +  int (b . grad T) v  +  int (J~/Fo) T v  =  int J~ g v,
    A = J G^-1,  b = J DF^-1 u / alpha,  J~ = J/|Omega|,  Fo = alpha dt/|Omega|,
    g = q |Omega| / k + T_old / Fo,

with alpha = k / (rho c) and everything else as in `piece.py`.  Steady: 1/Fo = 0.  The
physical heat balance is k times this one, which the host applies when it couples
pieces of different materials.

**The flow is a stream function.**  ``psi`` holds, at the chart's nodes, the physical
stream function divided by alpha (dimensionless).  Its bilinear interpolant on the
reference square gives b = (d psi / d eta, -d psi / d xi): the contravariant Piola
transform of u / alpha, because pulling a stream function back through a chart and
taking its curl there is the Piola transform of its curl.  So b is divergence-free at
every point of every element, and the flux of b through any element edge is the
increment of psi along it.  Mass is therefore conserved element by element, and across
a seam exactly when the two pieces' psi increase alike along it (`transport_host`).

**The discretisation** is Galerkin with bilinear elements for all three terms (the
diffusion and the reaction are `fe.assemble`'s).  The advection matrix
C[k, l] = int (b . grad phi_l) phi_k is a trilinear form in (psi, phi_l, phi_k) whose
integrand has degree 2 in each reference direction, so 2 x 2 Gauss points integrate it
exactly (`ADV`).

**The loop identity.**  With B the loop's flux mass, the sum over the boundary edges
a -> b, counter-clockwise, of (psi_b - psi_a) times the edge's 1-D mass
[[1/3, 1/6], [1/6, 1/3]],

    C + C^T = B   exactly.

Proof: int (b . grad T) v + (b . grad v) T = int b . grad(T v) = loop integral of
(b . n) T v, since div b = 0; and on a counter-clockwise loop (b . n) ds = d psi, which
is constant on each edge.  So the symmetric part of the Galerkin piece matrix is its
diffusion and reaction plus B/2: positive on outflow edges, negative on inflow ones.
That is why a piece coupled through value and diffusive flux alone is not passive
(the family catalog, CO-T), and why `transport_host` moves half of each seam's B into
the junction (the ADVEC port).

**The adjoint.**  K^T = K(-psi) + B, and B lives on the loop, so the interior block of
K^T is that of the same piece with the flow reversed.  The adjoint constraint modes
G = -K_II^-T K_GI^T Q are therefore the trial modes of the reversed piece: one network
can give both by b -> -b (catalog section 1.3), which `reversed` makes testable.

**Peclet numbers.**  An element's cell Peclet number is h max(|b_1| / A_11,
|b_2| / A_22) at its centre; the piece's is n times the largest of them.  For a straight
piece of length L along the flow it is |u|_max L / alpha, the plan's Pe_piece.  The
owner's cap is 200 per piece (plan section 2), and Galerkin oscillates once a cell's
number exceeds 2 (plan section 4, item 5), so a piece at the cap needs n >= 100.
`envelope_refusals` names both.
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
from .piece import PieceProblem

#: the owner's Peclet cap per piece (plan section 2)
PE_MAX = 200.0
#: Galerkin's oscillation threshold on the cell Peclet number (plan section 4, item 5)
CELL_PE_MAX = 2.0
#: a side carries flow when the largest increment of psi along it exceeds this
#: fraction of the piece's largest psi increment over the loop (or is above an absolute
#: floor when the piece carries no flow at all)
FLOW_REL_TOL = 1e-9
PORTS_SOLID = ("THERM",)
PORTS_COOLANT = ("THERM", "ADVEC")


def _advection_tensor() -> np.ndarray:
    """T[m, k, l] = int over the unit cell of (d_y phi_m d_x phi_l - d_x phi_m d_y phi_l)
    phi_k, local nodes (0,0), (1,0), (1,1), (0,1), by 2 x 2 Gauss points (exact: the
    integrand has degree 2 in each direction).  The element's advection matrix is
    C_e[k, l] = sum_m psi_m T[m, k, l]; it does not depend on the element's size."""
    gp = (np.array([-1.0, 1.0]) / math.sqrt(3.0) + 1.0) / 2.0
    T = np.zeros((4, 4, 4))
    for x in gp:
        for y in gp:
            phi = np.array([(1 - x) * (1 - y), x * (1 - y), x * y, (1 - x) * y])
            dx = np.array([-(1 - y), 1 - y, y, -y])
            dy = np.array([-(1 - x), -x, x, 1 - x])
            # b . grad phi_l with b = (d_y psi, -d_x psi) and psi = sum_m psi_m phi_m
            T += 0.25 * (np.einsum("m,l,k->mkl", dy, dx, phi)
                         - np.einsum("m,l,k->mkl", dx, dy, phi))
    return T


ADV = _advection_tensor()
EDGE_MASS = np.array([[1.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 1.0 / 3.0]])


def _scatter(n: int, ke: np.ndarray) -> sp.csr_matrix:
    nodes = fe.element_nodes(n)
    rows = np.repeat(nodes, 4, axis=1).ravel()
    cols = np.tile(nodes, (1, 4)).ravel()
    N = (n + 1) ** 2
    return sp.csr_matrix((ke.ravel(), (rows, cols)), shape=(N, N))


def element_psi(psi: np.ndarray) -> np.ndarray:
    """(n^2, 4) element values of a nodal field, local nodes (0,0), (1,0), (1,1), (0,1),
    elements in [j, i] order (`fe.element_nodes`)."""
    psi = np.asarray(psi, float)
    return np.stack([psi[:-1, :-1], psi[:-1, 1:], psi[1:, 1:], psi[1:, :-1]],
                    axis=-1).reshape(-1, 4)


def advection(n: int, psi: np.ndarray) -> sp.csr_matrix:
    """C on the (n+1)^2 nodes: C[k, l] = int (b . grad phi_l) phi_k, b = curl psi."""
    psi = np.asarray(psi, float)
    if psi.shape != (n + 1, n + 1):
        raise ValueError(f"psi has shape {psi.shape}, the grid needs {(n + 1, n + 1)}")
    ke = np.einsum("em,mkl->ekl", element_psi(psi), ADV)
    return _scatter(n, ke)


def loop_increments(n: int, psi: np.ndarray) -> np.ndarray:
    """(4n,) psi at loop position k+1 minus psi at loop position k (counter-clockwise):
    the outward flux of b through each boundary edge.  Edge k belongs to side k // n."""
    loop = fe.grid(n)[0]
    pv = np.asarray(psi, float).ravel()[loop]
    return np.roll(pv, -1) - pv


def flux_mass(n: int, psi: np.ndarray, sides=(0, 1, 2, 3), weights=None) -> sp.csr_matrix:
    """B on the (n+1)^2 nodes, restricted to the given sides: per boundary edge a -> b,
    (psi_b - psi_a) [[1/3, 1/6], [1/6, 1/3]] on its two nodes.  ``weights`` (4n,)
    multiplies edge by edge (the host's split), default 1."""
    loop = fe.grid(n)[0]
    d = loop_increments(n, psi)
    w = np.ones(4 * n) if weights is None else np.asarray(weights, float)
    keep = np.zeros(4 * n, bool)
    for s in sides:
        keep[s * n:(s + 1) * n] = True
    k = np.flatnonzero(keep & (w != 0.0))
    a, b = loop[k], loop[(k + 1) % (4 * n)]
    coef = d[k] * w[k]
    rows = np.concatenate([a, a, b, b])
    cols = np.concatenate([a, b, a, b])
    vals = np.concatenate([coef * EDGE_MASS[0, 0], coef * EDGE_MASS[0, 1],
                           coef * EDGE_MASS[1, 0], coef * EDGE_MASS[1, 1]])
    N = (n + 1) ** 2
    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))


def centre_velocity(psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(b_1, b_2) at the element centres, [j, i]: b = (d psi / d eta, -d psi / d xi)."""
    psi = np.asarray(psi, float)
    n = psi.shape[0] - 1
    d_xi = 0.5 * ((psi[:-1, 1:] - psi[:-1, :-1]) + (psi[1:, 1:] - psi[1:, :-1])) * n
    d_eta = 0.5 * ((psi[1:, :-1] - psi[:-1, :-1]) + (psi[1:, 1:] - psi[:-1, 1:])) * n
    return d_eta, -d_xi


def cell_peclet(psi: np.ndarray, a11: np.ndarray, a22: np.ndarray) -> np.ndarray:
    """(n, n) h max(|b_1| / A_11, |b_2| / A_22) at the element centres."""
    b1, b2 = centre_velocity(psi)
    n = np.asarray(psi).shape[0] - 1
    return np.maximum(np.abs(b1) / np.asarray(a11), np.abs(b2) / np.asarray(a22)) / n


# ---------------------------------------------------------------------------
# the piece
# ---------------------------------------------------------------------------


@dataclass
class TransportPiece(PieceProblem):
    """A charted piece of one material carrying a given flow.

    ``psi``: the nodal stream function over alpha, (n+1, n+1) [j, i]; zero for a solid
    piece.  ``alpha``: the material's diffusivity k / (rho c), m^2/s, kept for the
    record (the operator reads only ``psi``).  ``kappa_ref`` is the conductivity k.
    ``fo``, ``g``, ``label`` and ``meta`` are `PieceProblem`'s.  A piece is one material,
    so ``log_kappa_hat`` must stay zero."""
    psi: np.ndarray | None = None
    alpha: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        n = self.chart.n
        if self.psi is None:
            self.psi = np.zeros((n + 1, n + 1))
        self.psi = np.asarray(self.psi, float)
        if self.psi.shape != (n + 1, n + 1):
            raise ValueError(f"psi has shape {self.psi.shape}, the chart needs {(n + 1, n + 1)}")
        if np.any(self.log_kappa_hat_e != 0.0):
            raise ValueError("a CO-T piece is one material: log_kappa_hat must be zero")

    # -- the flow -------------------------------------------------------------------

    def increments(self) -> np.ndarray:
        return loop_increments(self.n, self.psi)

    def flow_scale(self) -> float:
        return float(np.max(np.abs(self.increments()), initial=0.0))

    @property
    def flows(self) -> bool:
        """Whether the piece carries flow (any boundary edge with a flux)."""
        return self.flow_scale() > 0.0

    def side_flow(self, s: int) -> bool:
        """Whether side s carries flow, relative to the piece's own largest increment."""
        d = self.increments()[s * self.n:(s + 1) * self.n]
        sc = self.flow_scale()
        return sc > 0.0 and float(np.max(np.abs(d))) > FLOW_REL_TOL * sc

    def side_flux(self, s: int) -> float:
        """The outward flux of b through side s (its increment of psi)."""
        return float(np.sum(self.increments()[s * self.n:(s + 1) * self.n]))

    def ports(self) -> tuple:
        return PORTS_COOLANT if self.flows else PORTS_SOLID

    def peclet(self) -> float:
        """The piece's Peclet number: n times the largest cell Peclet number."""
        return self.n * self.cell_peclet_max()

    def cell_peclet_max(self) -> float:
        t = self.tensors()
        return float(np.max(cell_peclet(self.psi, t["a11"], t["a22"]), initial=0.0))

    def reversed(self) -> "TransportPiece":
        """The same piece with the flow reversed: its trial modes are this piece's
        adjoint modes."""
        return replace(self, psi=-self.psi, label=self.label + " (reversed)",
                       meta=dict(self.meta))

    # -- the operator -----------------------------------------------------------------

    def diffusion_matrix(self) -> sp.csr_matrix:
        """Diffusion and reaction: the symmetric part without the loop term B/2."""
        t = self.tensors()
        return fe.assemble(self.n, t["a11"], t["a12"], t["a22"], t["sigma"])

    def advection_matrix(self) -> sp.csr_matrix:
        return advection(self.n, self.psi)

    def matrix(self) -> sp.csr_matrix:
        """K, the Galerkin piece matrix (diffusion, reaction and advection) with natural
        conditions on the whole loop: the monolithic reference assembles these."""
        return (self.diffusion_matrix() + self.advection_matrix()).tocsr()

    def flux_mass(self, sides=(0, 1, 2, 3), weights=None) -> sp.csr_matrix:
        return flux_mass(self.n, self.psi, sides, weights)


def make_piece(chart: ch.Chart, psi, *, k: float, alpha: float, fo: float | None = None,
               g=None, label: str = "", meta: dict | None = None) -> TransportPiece:
    return TransportPiece(chart, fo=fo, g=g, kappa_ref=float(k), label=label,
                          meta=dict(meta or {}), psi=psi, alpha=float(alpha))


# ---------------------------------------------------------------------------
# the exact (classical) response
# ---------------------------------------------------------------------------


class ExactTransport:
    """One sparse LU of K_II; trial modes, adjoint modes and the particular field by
    triangular solves (the adjoint ones with the transposed factors)."""

    def __init__(self, K: sp.spmatrix, n: int):
        self.n = n
        loop, inner = fe.grid(n)
        K = K.tocsr()
        self.K = K
        self.K_II = K[inner][:, inner].tocsc()
        self.K_IG = K[inner][:, loop].tocsr()
        self.K_GI = K[loop][:, inner].tocsr()
        self.K_GG = K[loop][:, loop].toarray()
        self.lu = spla.splu(self.K_II)

    def modes(self, Q: np.ndarray) -> np.ndarray:
        """H = -K_II^-1 K_IG Q."""
        return -self.lu.solve(np.asarray(self.K_IG @ Q))

    def adjoint_modes(self, Q: np.ndarray) -> np.ndarray:
        """G = -K_II^-T K_GI^T Q."""
        return -self.lu.solve(np.asarray(self.K_GI.T @ Q), trans="T")

    def particular(self, b_I: np.ndarray) -> np.ndarray:
        return self.lu.solve(np.asarray(b_I, float))

    def port_matrix(self, Q: np.ndarray, H: np.ndarray | None = None,
                    G: np.ndarray | None = None) -> np.ndarray:
        """W^T K V with V = (H; Q), W = (G; Q), not symmetrised.  With exact modes it is
        the Schur complement Q^T (K_GG - K_GI K_II^-1 K_IG) Q."""
        H = self.modes(Q) if H is None else H
        G = self.adjoint_modes(Q) if G is None else G
        KV_I = self.K_II @ H + self.K_IG @ Q
        KV_G = self.K_GI @ H + self.K_GG @ Q
        return G.T @ KV_I + Q.T @ KV_G


@dataclass
class TransportResponse:
    """What a CO-T expert returns for one piece: trial modes H, adjoint modes G (None
    makes the host test with the trial modes: plain Galerkin, the first-order arm) and
    the particular field, all interior ((n-1)^2 rows)."""
    H: np.ndarray
    G: np.ndarray | None
    up: np.ndarray
    level: np.ndarray | None = None


def exact_response(p: TransportPiece, Q: np.ndarray) -> tuple[TransportResponse, ExactTransport]:
    ex = ExactTransport(p.matrix(), p.n)
    H = ex.modes(Q)
    G = ex.adjoint_modes(Q)
    up = ex.particular(p.load()[fe.grid(p.n)[1]])
    return TransportResponse(H, G, up), ex


class ClassicalTransportExpert:
    """Exact classical piece solves: the CO-T classical superelement."""
    name = "classical-cot"
    envelope = None

    def __init__(self, adjoint: bool = True):
        self.adjoint = adjoint
        self.calls = 0
        self.factorisations = 0

    def respond(self, problems, pb_of) -> list[TransportResponse]:
        out = []
        for p in problems:
            r, _ = exact_response(p, pb_of(p.n).Q)
            if not self.adjoint:
                r = TransportResponse(r.H, None, r.up)
            out.append(r)
            self.calls += 1
            self.factorisations += 1
        return out


# ---------------------------------------------------------------------------
# the envelope
# ---------------------------------------------------------------------------


def envelope_refusals(p: TransportPiece, pe_max: float = PE_MAX,
                      cell_pe_max: float = CELL_PE_MAX) -> list[str]:
    """Why a classical CO-T piece is refused, if it is: a chart that is not a
    homeomorphism (Proposition 3), a Peclet number above the cap, or a cell Peclet
    number at which Galerkin oscillates.  Empty when admitted."""
    out = []
    c = cert.certify(p.chart.X)
    if not c.ok:
        out.append(f"the chart is not certified: {c.reason}")
    pe = p.peclet()
    if pe > pe_max * (1.0 + 1e-12):
        out.append(f"Peclet number {pe:.4g} above the cap of {pe_max:g} per piece: "
                   "cut the channel into shorter pieces")
    cpe = p.cell_peclet_max()
    if cpe > cell_pe_max * (1.0 + 1e-12):
        need = int(math.ceil(pe / cell_pe_max * (1.0 - 1e-12)))
        out.append(f"cell Peclet number {cpe:.3g} above {cell_pe_max:g}: Galerkin "
                   f"oscillates; the grid needs n >= {need}")
    return out


# ---------------------------------------------------------------------------
# flows
# ---------------------------------------------------------------------------


def profile(s, kind: str = "parabolic"):
    """A channel's stream-function profile across it, f(0) = 0 and f(1) = 1: the
    plane Poiseuille profile 3 s^2 - 2 s^3 (velocity 6 s (1 - s)), or plug flow s."""
    s = np.asarray(s, float)
    if kind == "parabolic":
        return 3.0 * s ** 2 - 2.0 * s ** 3
    if kind == "plug":
        return s.copy()
    raise ValueError(f"profile {kind!r}: 'parabolic' or 'plug'")


def psi_nodes(chart: ch.Chart, psi_fn, alpha: float) -> np.ndarray:
    """A physical stream function psi_fn(x, y) (m^2/s), at the chart's nodes, over
    alpha.  One function for a whole layout makes every seam conserve mass."""
    X = chart.X
    return np.asarray(psi_fn(X[..., 0], X[..., 1]), float) / float(alpha)


def scale_to_peclet(pieces: list, target: float) -> float:
    """Scale every piece's psi by one factor so the largest piece Peclet number is
    ``target`` (Peclet numbers are linear in psi); returns the factor."""
    pe = max((p.peclet() for p in pieces), default=0.0)
    if pe <= 0.0:
        if target > 0.0:
            raise ValueError("no piece carries flow: cannot scale to a Peclet number")
        return 1.0
    f = target / pe
    for p in pieces:
        p.psi = p.psi * f
    return f
