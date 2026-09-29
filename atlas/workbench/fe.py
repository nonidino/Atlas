"""Q1 finite elements on a case's grid: plane-stress elasticity and conduction.

The discretization the two structural showcase families share: a bracket under
load (case 6) and a heated plate that expands (case 7).  Nodes at the cells'
corners, one bilinear element per cell, **a material per element** (from the
case's regions).

**The same element as the build repo's `ThermoStruct2D`** (`solvers/
thermostruct2d.py`): 2 x 2 Gauss points with unit weights, its shape functions
and strain-displacement matrix, and the plane-stress constitutive matrix

    D = E / (1 - nu^2) [[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]].

`ThermoStruct2D` holds one material for its whole mesh, which is why this module
exists: the showcase needs two.  On one material its stiffness, conduction and
mass matrices are that solver's own (`tests/test_workbench_cases.py` checks it
against the build repo when the checkout is present).  Everything is per metre
of thickness, as there.

Node ``(i, j)`` -- ``i`` along x, ``j`` along y -- is ``j (nx + 1) + i``; element
``(i, j)`` is cell ``j nx + i``, its corners counter-clockwise from the lower
left, `ThermoStruct2D`'s local order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

_GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)
_QP = [(a, b) for a in _GP for b in _GP]
_QW = [1.0, 1.0, 1.0, 1.0]


def _shape(xi: float, eta: float):
    """(N [4], dN_ref [4, 2]) at a reference point."""
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dN = 0.25 * np.array([[-(1 - eta), -(1 - xi)], [(1 - eta), -(1 + xi)],
                          [(1 + eta), (1 + xi)], [-(1 + eta), (1 - xi)]])
    return N, dN


def _B(dN: np.ndarray) -> np.ndarray:
    """Strain-displacement matrix [3, 8] for physical gradients dN [4, 2]."""
    B = np.zeros((3, 8))
    B[0, 0::2] = dN[:, 0]
    B[1, 1::2] = dN[:, 1]
    B[2, 0::2] = dN[:, 1]
    B[2, 1::2] = dN[:, 0]
    return B


def plane_stress(E: float, nu: float) -> np.ndarray:
    return E / (1.0 - nu ** 2) * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0],
                                           [0.0, 0.0, 0.5 * (1.0 - nu)]])


@dataclass(frozen=True)
class QuadGrid:
    nx: int
    ny: int
    dx: float

    @property
    def n_nodes(self) -> int:
        return (self.nx + 1) * (self.ny + 1)

    @property
    def n_elem(self) -> int:
        return self.nx * self.ny

    def node(self, i, j):
        return np.asarray(j) * (self.nx + 1) + np.asarray(i)

    def conn(self) -> np.ndarray:
        """[n_elem, 4] node ids, element e = j nx + i, counter-clockwise."""
        j, i = np.meshgrid(np.arange(self.ny), np.arange(self.nx), indexing="ij")
        i, j = i.ravel(), j.ravel()
        return np.stack([self.node(i, j), self.node(i + 1, j), self.node(i + 1, j + 1),
                         self.node(i, j + 1)], axis=1)

    def coords(self) -> np.ndarray:
        """[n_nodes, 2] node coordinates."""
        j, i = np.meshgrid(np.arange(self.ny + 1), np.arange(self.nx + 1), indexing="ij")
        return np.stack([i.ravel() * self.dx, j.ravel() * self.dx], axis=1)

    def edge_nodes(self, edge: str, start: int = 0, stop: int | None = None) -> np.ndarray:
        """The nodes of cells ``start .. stop`` along one edge (both ends' corners)."""
        n = self.ny if edge in ("left", "right") else self.nx
        stop = n if stop is None else stop
        k = np.arange(start, stop + 1)
        if edge == "left":
            return self.node(0, k)
        if edge == "right":
            return self.node(self.nx, k)
        if edge == "bottom":
            return self.node(k, 0)
        return self.node(k, self.ny)


def _element_matrices(h: float):
    """Per unit property on a square element of side h: the conduction matrix for
    k = 1, the mass matrix for rho cp = 1, and the Gauss data for elasticity."""
    J = 0.5 * h                                           # dx/dxi on a square
    det = J * J
    Kc = np.zeros((4, 4))
    Mc = np.zeros((4, 4))
    gauss = []
    for (xi, eta), w in zip(_QP, _QW):
        N, dN_ref = _shape(xi, eta)
        dN = dN_ref / J
        Kc += w * det * dN @ dN.T
        Mc += w * det * np.outer(N, N)
        gauss.append((w * det, N, _B(dN)))
    return Kc, Mc, gauss


def _scatter(ke: np.ndarray, dofs: np.ndarray, size: int) -> sp.csr_matrix:
    """``ke`` [n_elem, m, m] into a [size, size] matrix at ``dofs`` [n_elem, m]."""
    m = dofs.shape[1]
    rows = np.repeat(dofs, m, axis=1).ravel()
    cols = np.tile(dofs, (1, m)).ravel()
    return sp.csr_matrix(sp.coo_matrix((ke.ravel(), (rows, cols)), shape=(size, size)))


def elastic_dofs(conn: np.ndarray) -> np.ndarray:
    return np.stack([2 * conn, 2 * conn + 1], axis=-1).reshape(conn.shape[0], 8)


def assemble_elastic(g: QuadGrid, E: np.ndarray, nu: np.ndarray) -> sp.csr_matrix:
    """The plane-stress stiffness [2 n_nodes, 2 n_nodes], per metre of thickness,
    with ``E`` and ``nu`` per element."""
    _Kc, _Mc, gauss = _element_matrices(g.dx)
    mats = {}
    E, nu = np.asarray(E, float).ravel(), np.asarray(nu, float).ravel()
    ke = np.zeros((g.n_elem, 8, 8))
    for key in set(zip(E.tolist(), nu.tolist())):
        if key not in mats:
            D = plane_stress(*key)
            mats[key] = sum(w * B.T @ D @ B for w, _N, B in gauss)
        sel = (E == key[0]) & (nu == key[1])
        ke[sel] = mats[key]
    return _scatter(ke, elastic_dofs(g.conn()), 2 * g.n_nodes)


def assemble_thermal(g: QuadGrid, k: np.ndarray, rho_cp: np.ndarray):
    """(K_th, M_th) [n_nodes, n_nodes]: conduction and consistent mass, per element
    ``k`` and ``rho cp``."""
    Kc, Mc, _gauss = _element_matrices(g.dx)
    k, rc = np.asarray(k, float).ravel(), np.asarray(rho_cp, float).ravel()
    conn = g.conn()
    return (_scatter(k[:, None, None] * Kc[None], conn, g.n_nodes),
            _scatter(rc[:, None, None] * Mc[None], conn, g.n_nodes))


def thermal_load_matrix(g: QuadGrid, E: np.ndarray, nu: np.ndarray,
                        alpha: np.ndarray) -> sp.csr_matrix:
    """``G`` [2 n_nodes, n_nodes] with ``f_th = G (T - T_ref)``: the load of the
    thermal strain ``eps_0 = alpha dT (1, 1, 0)``, integrated as `ThermoStruct2D.
    solve_mechanical` does (``B^T D eps_0`` at each Gauss point, dT interpolated)."""
    _Kc, _Mc, gauss = _element_matrices(g.dx)
    E, nu, al = (np.asarray(a, float).ravel() for a in (E, nu, alpha))
    a1 = np.array([1.0, 1.0, 0.0])
    conn = g.conn()
    ge = np.zeros((g.n_elem, 8, 4))
    for key in set(zip(E.tolist(), nu.tolist(), al.tolist())):
        D = plane_stress(key[0], key[1])
        blk = sum(w * np.outer(B.T @ D @ a1, key[2] * N) for w, N, B in gauss)
        ge[(E == key[0]) & (nu == key[1]) & (al == key[2])] = blk
    rows = np.repeat(elastic_dofs(conn), 4, axis=1).ravel()
    cols = np.tile(conn, (1, 8)).ravel()
    return sp.csr_matrix(sp.coo_matrix((ge.ravel(), (rows, cols)),
                                       shape=(2 * g.n_nodes, g.n_nodes)))


def edge_force(g: QuadGrid, edge: str, start: int, stop: int | None,
               tx: float, ty: float) -> np.ndarray:
    """Nodal forces [2 n_nodes] of a uniform traction ``(tx, ty)`` (Pa, per metre of
    thickness: N/m^2 on the edge face) on an edge segment: half of each edge
    element's share to each of its two nodes, the consistent load."""
    f = np.zeros(2 * g.n_nodes)
    nodes = g.edge_nodes(edge, start, stop)
    a, b = nodes[:-1], nodes[1:]
    half = 0.5 * g.dx
    for n in (a, b):
        np.add.at(f, 2 * n, tx * half)
        np.add.at(f, 2 * n + 1, ty * half)
    return f


def rigid_modes(g: QuadGrid) -> np.ndarray:
    """Orthonormal basis of the planar rigid-body motions [2 n_nodes, 3]: two
    translations and a rotation, `ThermoStruct2D._rigid_modes`' construction."""
    p = g.coords()
    V = np.zeros((2 * g.n_nodes, 3))
    V[0::2, 0] = 1.0
    V[1::2, 1] = 1.0
    c = p.mean(0)
    V[0::2, 2] = -(p[:, 1] - c[1])
    V[1::2, 2] = p[:, 0] - c[0]
    q, _ = np.linalg.qr(V)
    return q


def rigid_modes_on(g: QuadGrid, nodes: np.ndarray) -> np.ndarray:
    """`rigid_modes` over a set of nodes only -- a drawn domain's (case file 0.4) --
    ``[2 len(nodes), 3]``, their degrees of freedom interleaved node by node."""
    p = g.coords()[np.asarray(nodes)]
    V = np.zeros((2 * len(p), 3))
    V[0::2, 0] = 1.0
    V[1::2, 1] = 1.0
    c = p.mean(0)
    V[0::2, 2] = -(p[:, 1] - c[1])
    V[1::2, 2] = p[:, 0] - c[0]
    q, _ = np.linalg.qr(V)
    return q


def element_stress(g: QuadGrid, u: np.ndarray, E: np.ndarray, nu: np.ndarray,
                   eps0: np.ndarray | None = None) -> np.ndarray:
    """(sigma_xx, sigma_yy, sigma_xy) [n_elem, 3] at each element's centre,
    ``D (eps(u) - eps0)``; ``eps0`` is the thermal strain per element (isotropic)."""
    _N, dN_ref = _shape(0.0, 0.0)
    B = _B(dN_ref / (0.5 * g.dx))
    eps = np.asarray(u).ravel()[elastic_dofs(g.conn())] @ B.T
    if eps0 is not None:
        eps[:, 0] -= eps0
        eps[:, 1] -= eps0
    E, nu = np.asarray(E, float).ravel(), np.asarray(nu, float).ravel()
    c = E / (1.0 - nu ** 2)
    sxx = c * (eps[:, 0] + nu * eps[:, 1])
    syy = c * (nu * eps[:, 0] + eps[:, 1])
    sxy = c * 0.5 * (1.0 - nu) * eps[:, 2]
    return np.stack([sxx, syy, sxy], axis=1)


def von_mises(s: np.ndarray) -> np.ndarray:
    """Plane stress: sqrt(sxx^2 - sxx syy + syy^2 + 3 sxy^2)."""
    return np.sqrt(s[:, 0] ** 2 - s[:, 0] * s[:, 1] + s[:, 1] ** 2 + 3.0 * s[:, 2] ** 2)


__all__ = ["QuadGrid", "plane_stress", "assemble_elastic", "assemble_thermal",
           "thermal_load_matrix", "edge_force", "rigid_modes", "element_stress", "von_mises",
           "elastic_dofs"]
