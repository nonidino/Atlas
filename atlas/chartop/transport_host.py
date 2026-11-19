"""The CO-T host: Petrov-Galerkin coupling of transport pieces, with ADVEC seams.

The coolant plan's Phase 1 ([[chart-operator-stage-0b-and-1-plan]] section 5).  The
topology (vertices, edges, the interface numbering, Dirichlet sides) is the CO-D host's
own: `TransportLayout` subclasses `host.Layout`, which is pinned and is called, never
edited.  What changes is the coupling.

**Petrov-Galerkin (class G0').**  Each piece returns trial modes H, adjoint modes G and
a particular field u_p (`transport.TransportResponse`).  With V = (H; Q), W = (G; Q)
the host forms, per piece,

    Lambda_i = W^T K_i V,   chi_i = W^T (b_i - K_i U_p),

assembles S = sum_i k_i R_i^T Lambda_i R_i + J and chi likewise, fixes the Dirichlet
coefficients, solves once by LU (S is not symmetric) and rebuilds u_i = U_p + V R_i c.
For any trial and adjoint modes with exact traces, W~^T K V~ - Lambda = E_W^T K_II E_V
(the family catalog's G0' identity), so the port matrices, and the interface
coefficients, err quadratically.  With G = None the host tests with the trial modes
(plain Galerkin), whose error is first order: the comparison arm.

**ADVEC seams: where the advective term of a seam is kept.**  `transport` shows that a
Galerkin piece's symmetric part is its diffusion plus B/2 on its loop, B the flux mass:
positive where the flow leaves, negative where it enters.  A piece coupled by value and
diffusive flux alone (THERM) is therefore not passive once advection matters.  On
every edge segment the host moves theta B_segment out of the piece's own matrix,

    K_i^theta = K_i - B_i(theta_i),

and puts it back as the seam's junction J, the advective heat the ADVEC port carries:

- ``split="skew"`` (the default): theta = 1/2 on every seam that carries flow.  The
  piece's port flow is then the diffusive flux minus half the advective flux, the
  pairing whose product with T is the piece's dissipation, so its symmetric part is
  exactly its diffusion and reaction.
- ``split="upwind"``: theta = 1 where the flow enters the piece and 0 where it leaves
  (a Robin-Robin split, Achdou, Le Tallec, Nataf and Vidrascu, CMAME 184, 2000): the
  piece keeps |B|/2, a margin.
- ``split="galerkin"``: theta = 0, no ADVEC term: the THERM-only pairing, kept to show
  what it lacks.

On the layout's own boundary: an inflow segment must be held at a temperature (an inlet;
theta = 1/2 there touches only fixed coefficients), an outflow segment keeps its B/2
(theta = 0: the do-nothing outlet, B/2 >= 0 there).  So, for exact responses, every
piece's port matrix has a positive semidefinite symmetric part under "skew" and
"upwind" (the Schur complement of a matrix with a positive semidefinite symmetric part
keeps one), and the junction restores sum_i k_i K_i exactly: **the split changes no
coupled answer**, with exact or learned modes alike, because B lives on the traces,
which are exact.  It changes what each piece certifies, and the junction is where
the advective flux crosses.

**The checks a seam must pass** (refused with a named reason, before any solve):
- flow across a seam needs ADVEC on both pieces: a solid piece receives none;
- the heat-capacity flux k d psi must agree segment by segment on the two sides
  (mass conservation across the seam: the two pieces' stream functions increase alike);
- a seam carrying flow must be conforming (no mortar for ADVEC);
- flow may enter the layout only through a side held at a temperature.

**The positive control** is `host.monolithic` on the same pieces: one Galerkin solve of
sum_i k_i K_i on the merged mesh.  With every side's whole trace (m = n) the coupled
answer equals it to rounding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla

from . import fe
from . import ports as pt
from . import transport as tr
from .host import Layout, Refused, monolithic

SPLITS = ("skew", "upwind", "galerkin")
#: a segment carries flow when |k d psi| exceeds this fraction of the layout's largest
FLOW_TOL = 1e-9
#: the heat-capacity fluxes of a seam's two sides must agree to this fraction
CONSERVE_TOL = 1e-8


@dataclass
class SeamInfo:
    """What the host decided about one edge."""
    edge: int
    kind: str                    # "advective", "wall", "inlet", "outlet", "boundary"
    owner: tuple
    other: tuple | None
    flux: float = 0.0            # k-scaled flux out of the owner through it


class TransportLayout(Layout):
    """Transport pieces, their seams and boundary conditions; the coupled solve.

    ``bc(i, s, points)`` as for `host.Layout`: ``("dirichlet", values)`` or
    ``("insulated", None)`` (a natural side: zero diffusive flux; at an outlet, the
    do-nothing condition)."""

    def __init__(self, problems: list, bc, kind: str = "sine-hat", m: int = 16,
                 tol: float = 1e-7, split: str = "skew"):
        if split not in SPLITS:
            raise ValueError(f"split {split!r}: one of {SPLITS}")
        for i, p in enumerate(problems):
            if not isinstance(p, tr.TransportPiece):
                raise TypeError(f"piece {i} is not a TransportPiece")
        self.split = split
        super().__init__(problems, bc, kind=kind, m=m, tol=tol,
                         ports_of=lambda i: problems[i].ports())
        self._advec()

    # -- the ADVEC classification -------------------------------------------------

    def _kd(self, i: int, s: int) -> np.ndarray:
        """k times the increments of psi along side s of piece i: the heat-capacity
        flux out of the piece through each of the side's n segments."""
        p = self.problems[i]
        n = p.n
        return p.kappa_ref * p.increments()[s * n:(s + 1) * n]

    def _advec(self):
        scale = max((float(np.max(np.abs(p.kappa_ref * p.increments()), initial=0.0))
                     for p in self.problems), default=0.0)
        self.flow_scale = scale
        thr = FLOW_TOL * scale
        self.theta = [np.zeros(4 * p.n) for p in self.problems]
        self.seams: list[SeamInfo] = []
        junction_parts = []                       # (piece, side, weights over the loop)
        for e, info in enumerate(self.edges):
            i, s = info.owner
            pi = self.problems[i]
            n_i = pi.n
            kd_i = self._kd(i, s)
            sl_i = slice(s * n_i, (s + 1) * n_i)
            if info.other is None:
                inflow = kd_i < -thr
                outflow = kd_i > thr
                dirichlet = info.bc is not None and info.bc[0] == "dirichlet"
                if np.any(inflow) and not dirichlet:
                    raise Refused(f"piece {i} ({pi.label}) side {s}: coolant flows in through "
                                  "a side that is not held at a temperature (an inlet needs one)")
                if np.any(inflow):
                    self.theta[i][sl_i] = np.where(inflow, 0.5, 0.0)
                kind = ("inlet" if np.any(inflow) else
                        "outlet" if np.any(outflow) else "boundary")
                self.seams.append(SeamInfo(e, kind, (i, s), None, float(kd_i.sum())))
                continue
            j, t = info.other
            pj = self.problems[j]
            n_j = pj.n
            kd_j = self._kd(j, t)
            flowing = (np.max(np.abs(kd_i), initial=0.0) > thr
                       or np.max(np.abs(kd_j), initial=0.0) > thr)
            if not flowing:
                self.seams.append(SeamInfo(e, "wall", (i, s), (j, t), 0.0))
                continue
            for a, b in ((i, j), (j, i)):
                if "ADVEC" not in self.problems[a].ports():
                    raise Refused(f"pieces {i} and {j}: flow crosses their seam but piece "
                                  f"{a} ({self.problems[a].label}) is solid and declares "
                                  "no ADVEC port")
            if not info.conforming:
                raise Refused(f"pieces {i} and {j}: a seam that carries flow must be "
                              "conforming (ADVEC has no mortar)")
            mis = float(np.max(np.abs(kd_i + kd_j[::-1])))
            big = float(np.max(np.abs(kd_i)))
            if mis > CONSERVE_TOL * max(big, thr):
                raise Refused(f"pieces {i} and {j}: the heat-capacity flux k d(psi) across "
                              f"their seam disagrees by {mis / max(big, 1e-300):.3g} of its "
                              "size (mass is not conserved across it)")
            sl_j = slice(t * n_j, (t + 1) * n_j)
            if self.split == "skew":
                th_i = np.full(n_i, 0.5)
                th_j = np.full(n_j, 0.5)
            elif self.split == "upwind":
                th_i = np.where(kd_i < 0.0, 1.0, 0.0)
                th_j = np.where(kd_j < 0.0, 1.0, 0.0)
            else:
                th_i = np.zeros(n_i)
                th_j = np.zeros(n_j)
            self.theta[i][sl_i] = th_i
            self.theta[j][sl_j] = th_j
            for (a, side, th) in ((i, s, th_i), (j, t, th_j)):
                w = np.zeros(4 * self.problems[a].n)
                na = self.problems[a].n
                w[side * na:(side + 1) * na] = th
                junction_parts.append((a, w))
            self.seams.append(SeamInfo(e, "advective", (i, s), (j, t), float(kd_i.sum())))
        # the junction J: what the splits took out of the pieces on the seams, put back
        J = np.zeros((self.N, self.N))
        for a, w in junction_parts:
            if not np.any(w):
                continue
            p = self.problems[a]
            Q = self.pb(p.n).Q
            loop = fe.grid(p.n)[0]
            Bw = tr.flux_mass(p.n, p.psi, weights=w)
            Bl = Bw[loop][:, loop].toarray()
            Ra = self.R[a]
            J += p.kappa_ref * (Ra.T @ (Q.T @ Bl @ Q) @ Ra)
        self.junction = J
        self._K = {}

    # -- the operators ---------------------------------------------------------------

    def piece_matrix(self, i: int):
        """K_i^theta = K_i - B_i(theta_i): the piece's matrix with its seams' split."""
        if i not in self._K:
            p = self.problems[i]
            K = p.matrix()
            if np.any(self.theta[i]):
                K = (K - tr.flux_mass(p.n, p.psi, weights=self.theta[i])).tocsr()
            self._K[i] = K
        return self._K[i]

    def check_envelope(self, envelope=None) -> list[str]:
        out = []
        for i, p in enumerate(self.problems):
            for r in tr.envelope_refusals(p):
                out.append(f"piece {i} ({p.label}): {r}")
        return out

    def solve(self, expert, refuse_outside: bool = True) -> "TransportCoupled":
        refusals = self.check_envelope()
        if refusals and refuse_outside:
            raise Refused("; ".join(refusals))
        calls0 = getattr(expert, "calls", 0)
        resp = expert.respond(self.problems, self.pb)
        calls = getattr(expert, "calls", 0) - calls0
        return self.couple(resp, calls=calls, refusals=refusals)

    def couple(self, resp: list, calls: int | None = None,
               refusals: list | None = None) -> "TransportCoupled":
        N = self.N
        S = np.zeros((N, N))
        chi = np.zeros(N)
        mats, loads = [], []
        for i, (p, r) in enumerate(zip(self.problems, resp)):
            Q = self.pb(p.n).Q
            K = self.piece_matrix(i)
            V = fe.full_field(p.n, r.H, Q)
            W = V if r.G is None else fe.full_field(p.n, r.G, Q)
            Lam = W.T @ (K @ V)
            b = p.load()
            Up = fe.full_field(p.n, r.up, np.zeros(4 * p.n))
            bt = W.T @ (b - K @ Up)
            mats.append(Lam)
            loads.append(bt)
            Ri = self.R[i]
            S += p.kappa_ref * (Ri.T @ Lam @ Ri)
            chi += p.kappa_ref * (Ri.T @ bt)
        S += self.junction
        fixed = np.array(sorted(self.fixed), int)
        free = np.setdiff1d(np.arange(N), fixed)
        c = np.zeros(N)
        c[fixed] = [self.fixed[d] for d in fixed]
        Sff = S[np.ix_(free, free)]
        rhs = chi[free] - S[np.ix_(free, fixed)] @ c[fixed]
        c[free] = sla.lu_solve(sla.lu_factor(Sff), rhs)
        fields = []
        for i, (p, r) in enumerate(zip(self.problems, resp)):
            ci = self.R[i] @ c
            Q = self.pb(p.n).Q
            fields.append(fe.full_field(p.n, r.up + r.H @ ci, Q @ ci))
        return TransportCoupled(self, c, S, chi, free, fields, mats, loads, resp, calls,
                                refusals or [])

    # -- traces and the oracle ---------------------------------------------------------

    def trace_coefficients(self, fields: list) -> np.ndarray:
        """Global interface coefficients of given nodal fields: each edge's trace (its
        owner's) projected onto the port space as `ports.project_side` does it, corner
        values exact.  The oracle's coefficients when the fields are the reference."""
        c = np.zeros(self.N)
        for info in self.edges:
            i, s = info.owner
            p = self.problems[i]
            pbi = self.pb(p.n)
            loop = fe.grid(p.n)[0]
            pos = pt.side_positions(p.n, s)
            vals = np.asarray(fields[i], float)[loop[pos]]
            coef = pt.project_side(pbi, s, vals)
            R = self.R[i]
            for k, v in coef.items():
                nz = np.flatnonzero(R[k])
                if len(nz) != 1:
                    raise Refused("a side whose coefficients are not plain dofs")
                c[nz[0]] = v * R[k, nz[0]]
        return c

    def reconstruct(self, resp: list, c: np.ndarray) -> list:
        out = []
        for i, (p, r) in enumerate(zip(self.problems, resp)):
            ci = self.R[i] @ c
            Q = self.pb(p.n).Q
            out.append(fe.full_field(p.n, r.up + r.H @ ci, Q @ ci))
        return out

    # -- certificates -------------------------------------------------------------------

    def piece_passivity(self, i: int, resp=None) -> float:
        """The smallest eigenvalue of the symmetric part of piece i's port matrix
        W^T K_i^theta V, relative to its largest in modulus; exact responses by default."""
        p = self.problems[i]
        Q = self.pb(p.n).Q
        if resp is None:
            resp, _ = tr.exact_response(p, Q)
        K = self.piece_matrix(i)
        V = fe.full_field(p.n, resp.H, Q)
        W = V if resp.G is None else fe.full_field(p.n, resp.G, Q)
        L = W.T @ (K @ V)
        ev = np.linalg.eigvalsh(0.5 * (L + L.T))
        return float(ev[0] / max(np.max(np.abs(ev)), 1e-300))


def _outflow_bulk(layout: TransportLayout, fields: list) -> float | None:
    """The mixing-cup temperature over the layout's outflow segments: the sum of
    k d psi (T_a + T_b)/2 over the sum of k d psi (exact for bilinear traces, since
    b . n is constant along a segment)."""
    num = den = 0.0
    for sm in layout.seams:
        if sm.kind != "outlet":
            continue
        i, s = sm.owner
        p = layout.problems[i]
        n = p.n
        loop = fe.grid(n)[0]
        d = p.kappa_ref * p.increments()[s * n:(s + 1) * n]
        pos = pt.side_positions(n, s)
        T = np.asarray(fields[i], float)[loop[pos]]
        mid = 0.5 * (T[:-1] + T[1:])
        out = d > 0.0
        num += float(np.sum(d[out] * mid[out]))
        den += float(np.sum(d[out]))
    return num / den if den > 0.0 else None


@dataclass
class TransportCoupled:
    layout: TransportLayout
    c: np.ndarray
    S: np.ndarray
    chi: np.ndarray
    free: np.ndarray
    fields: list
    mats: list
    loads: list
    responses: list
    calls: int | None
    refusals: list = field(default_factory=list)

    def sym_min_eig(self) -> float:
        """The smallest eigenvalue of the symmetric part of the free interface matrix,
        relative to its norm: positive when the coupled system is positive real."""
        Sff = self.S[np.ix_(self.free, self.free)]
        ev = np.linalg.eigvalsh(0.5 * (Sff + Sff.T))
        return float(ev[0] / max(np.linalg.norm(Sff, 2), 1e-300))

    def outlet_bulk(self) -> float | None:
        return _outflow_bulk(self.layout, self.fields)

    def errors(self, ref_fields: list) -> dict:
        """Against reference fields on the same pieces: the relative error in the
        diffusion-and-reaction energy sum_i k_i e^T D_i e (the symmetric part without
        the loop term: an H^1 seminorm, plus the step's L^2 part), the largest nodal
        error over the reference's range, and the outlet's mixing-cup temperature."""
        num = den = 0.0
        emax = 0.0
        lo, hi = math.inf, -math.inf
        for p, u, ut in zip(self.layout.problems, ref_fields, self.fields):
            D = p.diffusion_matrix()
            e = np.asarray(u, float) - np.asarray(ut, float)
            num += p.kappa_ref * float(e @ (D @ e))
            den += p.kappa_ref * float(u @ (D @ u))
            emax = max(emax, float(np.max(np.abs(e))))
            lo, hi = min(lo, float(np.min(u))), max(hi, float(np.max(u)))
        rng = hi - lo
        tb, tr_ = self.outlet_bulk(), _outflow_bulk(self.layout, ref_fields)
        return {"energy_rel": math.sqrt(max(num, 0.0) / den) if den > 0 else 0.0,
                "max_rel": emax / rng if rng > 0 else emax,
                "range": rng,
                "outlet_bulk": tb, "outlet_bulk_ref": tr_,
                "outlet_bulk_err_rel": (abs(tb - tr_) / rng if (tb is not None and rng > 0)
                                        else None)}


def balance(layout: TransportLayout, fields: list, mono=None) -> dict:
    """The heat a field leaves unaccounted for, on the merged mesh.

    The assembled Galerkin residual r = sum_i k_i (K_i u_i - b_i) (each piece's own
    rows, natural conditions everywhere) is, on the nodes a Dirichlet side holds, the
    heat that enters there (the reactions), and elsewhere it should vanish.  Its sum
    over the nodes no Dirichlet side holds is reported against the heat throughput
    (the sources' and the step's loads, the reactions, and the advective heat carried
    in and out).  Zero to rounding for the monolithic solve and with whole traces;
    with truncated ports, what the ports could not carry."""
    mono = monolithic(layout) if mono is None else mono
    ng = mono.nodes
    res = np.zeros(ng)
    src = 0.0
    held = np.zeros(ng, bool)
    for i, (p, u) in enumerate(zip(layout.problems, fields)):
        Gi = mono.global_index[i]
        r = p.kappa_ref * (p.matrix() @ np.asarray(u, float) - p.load())
        np.add.at(res, Gi, r)
        src += p.kappa_ref * float(np.sum(np.abs(p.load())))
    for e in layout.edges:
        if e.other is None and e.bc is not None and e.bc[0] == "dirichlet":
            i, s = e.owner
            n = layout.problems[i].n
            loop = fe.grid(n)[0]
            pos = pt.side_positions(n, s)
            held[mono.global_index[i][loop[pos]]] = True
    adv = 0.0
    for sm in layout.seams:
        if sm.kind not in ("inlet", "outlet"):
            continue
        i, s = sm.owner
        p = layout.problems[i]
        n = p.n
        loop = fe.grid(n)[0]
        d = p.kappa_ref * p.increments()[s * n:(s + 1) * n]
        T = np.asarray(fields[i], float)[loop[pt.side_positions(n, s)]]
        adv += float(np.sum(np.abs(d * 0.5 * (T[:-1] + T[1:]))))
    tot = float(np.sum(res[~held]))
    scale = max(src + float(np.sum(np.abs(res[held]))) + adv, 1e-300)
    return {"residual_sum": tot, "relative": abs(tot) / scale,
            "reactions": float(np.sum(res[held])), "throughput": scale}
