"""Cell-centred finite volumes for one scalar on a case's grid.

The discretization four showcase families share: heat conduction (case 2), the
electric potential in a resistive plate (case 5), a pollutant carried down a
river (case 3), and the fluid and solid of a cooled block (case 8).  One
operator,

    cap du/dt  -  div(k grad u)  +  div(F u)  =  s,

on square cells of side ``dx``, per unit depth:

* **diffusion** across each face with the harmonic mean of the two cells'
  coefficients, ``G = 2 k_P k_Q / (k_P + k_Q)``.  That is the series resistance
  of the two half-cells, so a stack of layers whose interfaces lie on cell faces
  is solved **exactly**: the discrete heat flow through a layered wall is the
  closed form ``dT / sum_i L_i / k_i`` to round-off (a test pins it);
* **advection** by a given face flux density ``F`` (capacity times velocity),
  upwinded, so the operator is an M-matrix and an explicit step is monotone
  below its own stability limit;
* **boundaries** on edge segments: a fixed value (through the half-cell, with
  the value upwinded in if the flow enters), a given inward flux, no flux, an
  advective inlet, and an advective outlet.

**Subdomains.**  `assemble` builds the system on ANY set of cells, and says
what it does with a face that leaves the set.  That is the whole of what the
coupling styles differ in:

``neighbour``  the outside cell's value closes the face, through the full
               face conductance: the overlapping Schwarz restriction
               ``R_i A R_i^T``, with the coupling to the outside kept as a
               matrix ``C`` so a driver computes ``b - C u`` from the current
               global state (styles A and B);
``face``       the face is cut in two: the set sees a boundary through its own
               half-cell, whose far side is either a given face value (the
               Dirichlet side of Dirichlet-Neumann) or a given flow (the
               Neumann side) -- style C.

**The full domain is the same call on every cell.**  So a one-window
decomposition is the full-domain system to the bit, by construction, and the
tests check it rather than assume it.

**A drawn domain** (case file 0.4) is a mask of the grid's cells, ``active``:
faces between two domain cells are interior, and a face between a domain cell
and a grid cell outside the domain is a boundary face like the grid's own edge
faces -- through the domain cell's half-cell, with its own condition from
``void``.  The full domain is then the same call on every domain cell, so the
one-window control holds unchanged.  With ``active`` left None, nothing here
changes by a bit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import scipy.sparse as sp

# boundary kinds, as codes in the per-edge arrays
NO_FLUX, FIXED, FLUX, INLET, OUTLET = 0, 1, 2, 3, 4
KIND_CODES = {"no-flux": NO_FLUX, "fixed": FIXED, "flux": FLUX, "inlet": INLET,
              "outlet": OUTLET}
EDGES = ("left", "right", "bottom", "top")


@dataclass
class Field:
    """The problem on the whole grid.  Arrays are ``(ny, nx)``, row 0 at the bottom.

    ``fx`` is the advective flux density through the ``nx + 1`` x-faces of each
    row (positive towards +x) and ``fy`` through the ``ny + 1`` y-faces of each
    column (positive towards +y); None means no advection.  ``bc[edge]`` is a
    pair ``(kinds, values)`` of arrays along that edge (left and right: ``ny``
    long, bottom to top; bottom and top: ``nx`` long, left to right).
    """

    nx: int
    ny: int
    dx: float
    k: np.ndarray
    cap: np.ndarray | None = None
    source: np.ndarray | None = None
    fx: np.ndarray | None = None
    fy: np.ndarray | None = None
    bc: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    #: a drawn domain: which cells are in it (None: all of the grid) ...
    active: np.ndarray | None = None
    #: ... and its faces to the void inside the grid: ``(cell, outside, direction,
    #: kinds, values, labels)``, ``direction`` indexing `geometry.DIRECTIONS`
    void: tuple | None = None

    def __post_init__(self):
        shape = (self.ny, self.nx)
        self.k = np.asarray(self.k, dtype=float)
        if self.k.shape != shape:
            raise ValueError(f"k is {self.k.shape}, the grid is {shape}")
        if np.any(self.k < 0.0):
            raise ValueError("a diffusion coefficient is negative")
        if self.source is None:
            self.source = np.zeros(shape)
        for e in EDGES:
            n = self.ny if e in ("left", "right") else self.nx
            if e not in self.bc:
                self.bc[e] = (np.zeros(n, dtype=np.int8), np.zeros(n))
            kinds, vals = self.bc[e]
            self.bc[e] = (np.asarray(kinds, dtype=np.int8).reshape(n),
                          np.asarray(vals, dtype=float).reshape(n))
        if self.active is not None:
            self.active = np.asarray(self.active, dtype=bool)
            if self.active.shape != shape:
                raise ValueError(f"the domain mask is {self.active.shape}, the grid is {shape}")
            if self.void is None:
                z = np.zeros(0, dtype=np.int64)
                self.void = (z, z, np.zeros(0, dtype=np.int8), np.zeros(0, dtype=np.int8),
                             np.zeros(0), np.zeros(0, dtype=object))

    @property
    def n(self) -> int:
        return self.nx * self.ny

    def flat(self, j, i):
        return np.asarray(j) * self.nx + np.asarray(i)

    def cells(self) -> np.ndarray:
        """The domain's cells, as ascending flat indices."""
        if self.active is None:
            return np.arange(self.n)
        return np.flatnonzero(self.active.ravel())

    def void_cells(self) -> np.ndarray:
        """The grid's cells outside the domain (empty unless it is drawn)."""
        if self.active is None:
            return np.zeros(0, dtype=np.int64)
        return np.flatnonzero(~self.active.ravel())


def harmonic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``2ab / (a + b)``, zero where either is zero (an insulating cell)."""
    s = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0.0, 2.0 * a * b / np.where(s > 0.0, s, 1.0), 0.0)


def interior_faces(f: Field):
    """Every interior face once: (P, Q, G, F) with P below or left of Q and
    ``F`` the advective flow from P to Q through the face (per unit depth)."""
    nx, ny, dx = f.nx, f.ny, f.dx
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx - 1), indexing="ij")
    P = (jj * nx + ii).ravel()
    Gx = harmonic(f.k[:, :-1], f.k[:, 1:]).ravel()
    Fx = (f.fx[:, 1:-1].ravel() * dx) if f.fx is not None else np.zeros(P.size)
    jj, ii = np.meshgrid(np.arange(ny - 1), np.arange(nx), indexing="ij")
    Py = (jj * nx + ii).ravel()
    Gy = harmonic(f.k[:-1, :], f.k[1:, :]).ravel()
    Fy = (f.fy[1:-1, :].ravel() * dx) if f.fy is not None else np.zeros(Py.size)
    P, Q = np.concatenate([P, Py]), np.concatenate([P + 1, Py + nx])
    G, F = np.concatenate([Gx, Gy]), np.concatenate([Fx, Fy])
    if f.active is not None:
        a = f.active.ravel()
        both = a[P] & a[Q]
        P, Q, G, F = P[both], Q[both], G[both], F[both]
    return P, Q, G, F


def _void_inflow(f: Field, cell: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """The advective flow INTO the domain through each void face (per unit depth)."""
    out = np.zeros(cell.size)
    if f.fx is None and f.fy is None:
        return out
    j, i = cell // f.nx, cell % f.nx
    for k, (di, dj) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
        s = direction == k
        if not np.any(s):
            continue
        if di and f.fx is not None:
            face = i[s] + (1 if di > 0 else 0)
            out[s] = -di * f.fx[j[s], face] * f.dx
        if dj and f.fy is not None:
            face = j[s] + (1 if dj > 0 else 0)
            out[s] = -dj * f.fy[face, i[s]] * f.dx
    return out


def boundary_faces(f: Field, labels: bool = False):
    """Every domain-boundary face: (cell, kind, value, G_half, F_in) where
    ``F_in`` is the advective flow INTO the domain through it.  The grid's edge
    faces come first, edge by edge, then a drawn domain's faces to the void.
    With ``labels``, a sixth array names each face's edge (``left`` ... or a drawn
    edge's name)."""
    nx, ny, dx = f.nx, f.ny, f.dx
    cells, kinds, vals, g, fin, names = [], [], [], [], [], []
    a = f.active.ravel() if f.active is not None else None
    for e in EDGES:
        kd, vv = f.bc[e]
        if e == "left":
            c = np.arange(ny) * nx
            flow = f.fx[:, 0] * dx if f.fx is not None else np.zeros(ny)
        elif e == "right":
            c = np.arange(ny) * nx + nx - 1
            flow = -f.fx[:, -1] * dx if f.fx is not None else np.zeros(ny)
        elif e == "bottom":
            c = np.arange(nx)
            flow = f.fy[0, :] * dx if f.fy is not None else np.zeros(nx)
        else:
            c = (ny - 1) * nx + np.arange(nx)
            flow = -f.fy[-1, :] * dx if f.fy is not None else np.zeros(nx)
        if a is not None:
            on = a[c]
            c, kd, vv, flow = c[on], kd[on], vv[on], flow[on]
        cells.append(c)
        kinds.append(kd)
        vals.append(vv)
        g.append(2.0 * f.k.ravel()[c])
        fin.append(flow)
        names.append(np.full(c.size, e, dtype=object))
    if f.active is not None and f.void is not None and len(f.void[0]):
        vc, _vo, vd, vk, vv, vl = f.void
        cells.append(np.asarray(vc, dtype=np.int64))
        kinds.append(np.asarray(vk, dtype=np.int8))
        vals.append(np.asarray(vv, dtype=float))
        g.append(2.0 * f.k.ravel()[vc])
        fin.append(_void_inflow(f, np.asarray(vc), np.asarray(vd)))
        names.append(np.asarray(vl, dtype=object))
    out = (np.concatenate(cells), np.concatenate(kinds), np.concatenate(vals),
           np.concatenate(g), np.concatenate(fin))
    return out + (np.concatenate(names),) if labels else out


@dataclass
class LocalSystem:
    """``A u = b + (the cut faces' terms)`` on one set of cells.

    ``idx`` are the global indices of the set's cells, ascending.  For a
    ``neighbour`` cut, ``C`` couples the set's rows to OUTSIDE global cells, so
    the right-hand side is ``b - C @ u_global``.  For a ``face`` cut, the cut
    faces are listed with their half-cell conductance: the Dirichlet side adds
    ``g_half * lambda`` to ``b[face_rows]`` (and its ``A`` already holds
    ``g_half``); the Neumann side adds the given flow.
    """

    idx: np.ndarray
    A: sp.csr_matrix
    b: np.ndarray
    C: sp.csr_matrix | None
    face_rows: np.ndarray
    face_outside: np.ndarray       # the global cell across each cut face
    face_g: np.ndarray             # the set's own half-cell conductance
    face_mode: str

    @property
    def n(self) -> int:
        return int(self.idx.size)


def assemble(f: Field, cells: np.ndarray | None = None,
             cut: Literal["neighbour", "dirichlet", "neumann"] = "neighbour",
             diag_add: np.ndarray | None = None) -> LocalSystem:
    """The system on ``cells`` (global indices; all cells when None).

    ``cut`` says how a face leaving the set is closed (see the module
    docstring): ``neighbour`` (Schwarz), ``dirichlet`` (the face value is given:
    the set's ``A`` holds the half-cell conductance), or ``neumann`` (the flow
    through it is given: ``A`` holds nothing for it).  ``diag_add`` is added to
    the diagonal (a backward-Euler ``cap dx^2 / dt``), per global cell.

    Built from the same face lists whatever the set, in the same order, so the
    set of all cells gives the full-domain matrix itself.
    """
    N = f.n
    if cells is None:
        cells = f.cells()
    cells = np.asarray(np.unique(cells), dtype=np.int64)
    inside = np.zeros(N, dtype=bool)
    inside[cells] = True
    loc = np.full(N, -1, dtype=np.int64)
    loc[cells] = np.arange(cells.size)
    n = cells.size
    diag = np.zeros(n)
    b = f.source.ravel()[cells] * f.dx * f.dx
    if diag_add is not None:
        diag += np.asarray(diag_add, dtype=float).ravel()[cells]

    P, Q, G, F = interior_faces(f)
    out_P = G + np.maximum(F, 0.0)          # what P loses per unit of u_P
    in_P = G + np.maximum(-F, 0.0)          # what P gains per unit of u_Q
    out_Q = G + np.maximum(-F, 0.0)
    in_Q = G + np.maximum(F, 0.0)
    pin, qin = inside[P], inside[Q]
    rows, cols, vals = [], [], []
    # faces with both cells inside
    both = pin & qin
    rows += [loc[P[both]], loc[Q[both]]]
    cols += [loc[Q[both]], loc[P[both]]]
    vals += [-in_P[both], -in_Q[both]]
    np.add.at(diag, loc[P[both]], out_P[both])
    np.add.at(diag, loc[Q[both]], out_Q[both])
    # faces that leave the set: P inside, Q outside, and the reverse
    fr, fo, fg = [], [], []
    crow, ccol, cval = [], [], []
    for mine, other, out_me, in_me, k_mine, flow_out in (
            (P, Q, out_P, in_P, None, F), (Q, P, out_Q, in_Q, None, -F)):
        sel = inside[mine] & ~inside[other]
        if not np.any(sel):
            continue
        if cut == "neighbour":
            np.add.at(diag, loc[mine[sel]], out_me[sel])
            crow.append(loc[mine[sel]])
            ccol.append(other[sel])
            cval.append(-in_me[sel])
        else:
            if np.any(flow_out[sel] != 0.0):
                raise ValueError("a face cut for Dirichlet-Neumann carries advective "
                                 "flow; style C couples material interfaces, which are "
                                 "walls to the flow")
            g_half = 2.0 * f.k.ravel()[mine[sel]]
            if cut == "dirichlet":
                np.add.at(diag, loc[mine[sel]], g_half)
            fr.append(loc[mine[sel]])
            fo.append(other[sel])
            fg.append(g_half)
    # the domain's own boundary
    bc_cell, bc_kind, bc_val, bc_g, bc_in = boundary_faces(f)
    s = inside[bc_cell]
    bc_cell, bc_kind, bc_val, bc_g, bc_in = (bc_cell[s], bc_kind[s], bc_val[s], bc_g[s],
                                             bc_in[s])
    r = loc[bc_cell]
    fixed = bc_kind == FIXED
    np.add.at(diag, r[fixed], bc_g[fixed] + np.maximum(-bc_in[fixed], 0.0))
    np.add.at(b, r[fixed], (bc_g[fixed] + np.maximum(bc_in[fixed], 0.0)) * bc_val[fixed])
    flux = bc_kind == FLUX
    np.add.at(b, r[flux], bc_val[flux] * f.dx)
    inlet = bc_kind == INLET
    np.add.at(b, r[inlet], np.maximum(bc_in[inlet], 0.0) * bc_val[inlet])
    np.add.at(diag, r[inlet], np.maximum(-bc_in[inlet], 0.0))
    outlet = bc_kind == OUTLET
    np.add.at(diag, r[outlet], np.maximum(-bc_in[outlet], 0.0))
    # an advective flow through a no-flux face would be a leak nobody declared
    leak = (bc_kind == NO_FLUX) | (bc_kind == FLUX)
    if np.any(bc_in[leak] != 0.0):
        raise ValueError("the flow crosses a boundary declared closed (no-flux or a "
                         "given flux); give that edge an inlet or an outlet")

    rows.append(np.arange(n))
    cols.append(np.arange(n))
    vals.append(diag)
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n))
    A.sum_duplicates()
    A.sort_indices()
    C = None
    if cut == "neighbour" and crow:
        C = sp.csr_matrix((np.concatenate(cval), (np.concatenate(crow),
                                                   np.concatenate(ccol))), shape=(n, N))
        C.sum_duplicates()
        C.sort_indices()
    cat = (lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dtype=dt))  # noqa: E731
    return LocalSystem(cells, A, b, C, cat(fr, np.int64), cat(fo, np.int64),
                       cat(fg, float), cut)


# ---------------------------------------------------------------------------
# what crosses the boundary: the balance every family's check reads
# ---------------------------------------------------------------------------


def _face_inflow(f: Field, u: np.ndarray, c, kd, v, g, fin) -> np.ndarray:
    """The flow INTO the domain through each of the faces ``c`` (their kinds, values,
    half-cell conductances and advective inflows), for the global state ``u``."""
    q = np.zeros(c.size)
    fx_ = kd == FIXED
    q[fx_] = g[fx_] * (v[fx_] - u[c[fx_]]) + np.where(fin[fx_] > 0.0, fin[fx_] * v[fx_],
                                                      fin[fx_] * u[c[fx_]])
    fl = kd == FLUX
    q[fl] = v[fl] * f.dx
    il = kd == INLET
    q[il] = np.where(fin[il] > 0.0, fin[il] * v[il], fin[il] * u[c[il]])
    ol = kd == OUTLET
    q[ol] = np.minimum(fin[ol], 0.0) * u[c[ol]]
    return q


def boundary_face_inflow(f: Field, u: np.ndarray) -> np.ndarray:
    """The flow INTO the domain through every boundary face, in `boundary_faces`'
    order (the grid's edges, then the faces to the void): what `boundary_inflow`
    sums by edge, face by face."""
    u = np.asarray(u, dtype=float).ravel()
    return _face_inflow(f, u, *boundary_faces(f))


def boundary_inflow(f: Field, u: np.ndarray) -> dict[str, float]:
    """The flow INTO the domain through each edge, for the global state ``u``.

    Diffusive and advective together, computed from the same face terms
    `assemble` uses, so a solved steady state balances it against the sources
    to the solver's round-off.
    """
    u = np.asarray(u, dtype=float).ravel()
    out = {}

    def flows(c, kd, v, g, fin):
        return _face_inflow(f, u, c, kd, v, g, fin)

    if f.active is None:
        bc_cell, bc_kind, bc_val, bc_g, bc_in = boundary_faces(f)
        lens = {e: (f.ny if e in ("left", "right") else f.nx) for e in EDGES}
        start = 0
        for e in EDGES:
            sl = slice(start, start + lens[e])
            start += lens[e]
            out[e] = float(np.sum(flows(bc_cell[sl], bc_kind[sl], bc_val[sl], bc_g[sl],
                                        bc_in[sl])))
        return out
    # a drawn domain: the grid's edges (their domain cells only), then each drawn
    # edge's faces to the void, grouped by the edge's name
    bc_cell, bc_kind, bc_val, bc_g, bc_in, names = boundary_faces(f, labels=True)
    q = flows(bc_cell, bc_kind, bc_val, bc_g, bc_in)
    for e in EDGES:
        out[e] = float(np.sum(q[names == e]))
    for name in dict.fromkeys(names.tolist()):
        if name not in out:
            out[str(name)] = float(np.sum(q[names == name]))
    return out


def total_source(f: Field) -> float:
    return float(np.sum(f.source) * f.dx * f.dx)


def face_flows(f: Field, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(x-face flows ``(ny, nx-1)``, y-face flows ``(ny-1, nx)``) between cells,
    positive towards +x and +y: what an interface integral is summed from."""
    u = np.asarray(u, dtype=float).reshape(f.ny, f.nx)
    gx = harmonic(f.k[:, :-1], f.k[:, 1:])
    qx = gx * (u[:, :-1] - u[:, 1:])
    if f.fx is not None:
        fx = f.fx[:, 1:-1] * f.dx
        qx = qx + np.where(fx > 0.0, fx * u[:, :-1], fx * u[:, 1:])
    gy = harmonic(f.k[:-1, :], f.k[1:, :])
    qy = gy * (u[:-1, :] - u[1:, :])
    if f.fy is not None:
        fy = f.fy[1:-1, :] * f.dx
        qy = qy + np.where(fy > 0.0, fy * u[:-1, :], fy * u[1:, :])
    if f.active is not None:
        a = f.active
        qx = np.where(a[:, :-1] & a[:, 1:], qx, 0.0)
        qy = np.where(a[:-1, :] & a[1:, :], qy, 0.0)
    return qx, qy


__all__ = ["NO_FLUX", "FIXED", "FLUX", "INLET", "OUTLET", "KIND_CODES", "EDGES", "Field",
           "harmonic", "interior_faces", "boundary_faces", "LocalSystem", "assemble",
           "boundary_inflow", "boundary_face_inflow", "total_source", "face_flows"]
