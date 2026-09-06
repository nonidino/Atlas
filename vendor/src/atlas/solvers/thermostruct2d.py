"""Conduction + plane-stress elasticity on the airframe shell (agent `c`).

Q1 bilinear finite elements on the shell mesh:

* **Conduction**, backward Euler. Unconditionally stable, which is the point --
  the shell's thermal timescale is slow and we want big steps (dt_model = 5e-2 s
  against the gas agents' 1e-3 s).
* **Elasticity**, quasi-static. The shell is thin and inertia is negligible
  against the ascent timescale, so each step is a linear SOLVE, not a march.

Both are assembled sparse. Conduction is solved with Jacobi-preconditioned CG
(SPD and well conditioned); elasticity is solved DIRECTLY, because the free-body
system needs its three rigid-body modes removed and the iterative route fails on
the real shell -- see `_solve_free`.

The mechanical solve takes the pressure loads from **both** gas sides (`b`
inside, `d` outside). That is exactly the b-c and c-d coupling Atlas will later
have to reproduce through its typed edges, so it is deliberately not simplified
to a one-sided load.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import cg

# 2x2 Gauss quadrature on the reference square
_GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)
_QP = np.array([(a, b) for a in _GP for b in _GP])
_QW = np.ones(4)


@dataclass(frozen=True)
class SolidMaterial:
    """Aluminium-lithium-like airframe alloy. Reference defaults, not a spec."""
    rho: float = 2700.0
    cp: float = 900.0
    k: float = 120.0
    E: float = 70.0e9
    nu: float = 0.33
    alpha: float = 23.0e-6
    emissivity: float = 0.3

    @property
    def diffusivity(self) -> float:
        return self.k / (self.rho * self.cp)


@dataclass(frozen=True, eq=False)
class ShellMesh:
    """Structured Q1 mesh. nodes [n_i+1, n_j+1, 2]; j = 0 is one face, j = -1 the
    other. `inner_face`/`outer_face` name which j edge sees which gas agent."""
    nodes: np.ndarray
    inner_j: int = 0
    outer_j: int = -1

    @property
    def shape(self) -> tuple[int, int]:
        return self.nodes.shape[0] - 1, self.nodes.shape[1] - 1

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0] * self.nodes.shape[1]

    def nid(self, i, j) -> np.ndarray:
        return i * self.nodes.shape[1] + j

    def element_nodes(self) -> np.ndarray:
        """[n_elem, 4] node ids, counter-clockwise."""
        ni, nj = self.shape
        i, j = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
        return np.stack([self.nid(i, j), self.nid(i + 1, j),
                         self.nid(i + 1, j + 1), self.nid(i, j + 1)], axis=-1).reshape(-1, 4)

    def element_coords(self) -> np.ndarray:
        """[n_elem, 4, 2]."""
        flat = self.nodes.reshape(-1, 2)
        return flat[self.element_nodes()]


def _shape_derivs(xy: np.ndarray, xi: float, eta: float):
    """Q1 shape-function gradients in physical space and the Jacobian determinant.

    xy [n_elem, 4, 2] -> (dN [n_elem, 4, 2], detJ [n_elem], N [4])."""
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dN_ref = 0.25 * np.array([[-(1 - eta), -(1 - xi)], [(1 - eta), -(1 + xi)],
                              [(1 + eta), (1 + xi)], [-(1 + eta), (1 - xi)]])
    J = np.einsum("kd,ekc->edc", dN_ref, xy)                 # [n_elem, 2, 2]
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    inv = np.empty_like(J)
    inv[:, 0, 0], inv[:, 1, 1] = J[:, 1, 1], J[:, 0, 0]
    inv[:, 0, 1], inv[:, 1, 0] = -J[:, 0, 1], -J[:, 1, 0]
    inv /= det[:, None, None]
    dN = np.einsum("kd,edc->ekc", dN_ref, inv)               # [n_elem, 4, 2]
    return dN, det, N


class ThermoStruct2D:
    def __init__(self, mesh: ShellMesh, mat: SolidMaterial = SolidMaterial()):
        self.mesh, self.mat = mesh, mat
        self.conn = mesh.element_nodes()
        self.xy = mesh.element_coords()
        self.K_th, self.M_th = self._assemble_thermal()
        self.K_me = self._assemble_mechanical()
        self._face_cache: dict = {}
        self._factors: dict = {}

    # -- assembly -----------------------------------------------------------
    def _assemble_thermal(self):
        n = self.mesh.n_nodes
        ne = self.conn.shape[0]
        Ke = np.zeros((ne, 4, 4))
        Me = np.zeros((ne, 4, 4))
        for (xi, eta), w in zip(_QP, _QW):
            dN, det, N = _shape_derivs(self.xy, xi, eta)
            Ke += w * self.mat.k * np.einsum("ekc,elc,e->ekl", dN, dN, det)
            Me += w * self.mat.rho * self.mat.cp * np.einsum("k,l,e->ekl", N, N, det)
        return self._scatter(Ke, n), self._scatter(Me, n)

    def _scatter(self, Ke, n, dofs_per_node: int = 1):
        conn = self.conn
        if dofs_per_node == 1:
            rows = np.repeat(conn, 4, axis=1).ravel()
            cols = np.tile(conn, (1, 4)).ravel()
        else:
            d = np.stack([2 * conn, 2 * conn + 1], axis=-1).reshape(conn.shape[0], -1)
            rows = np.repeat(d, 8, axis=1).ravel()
            cols = np.tile(d, (1, 8)).ravel()
        size = n * dofs_per_node
        return csr_matrix(coo_matrix((Ke.ravel(), (rows, cols)), shape=(size, size)))

    def _assemble_mechanical(self):
        E, nu = self.mat.E, self.mat.nu
        D = E / (1.0 - nu ** 2) * np.array([[1.0, nu, 0.0],
                                            [nu, 1.0, 0.0],
                                            [0.0, 0.0, 0.5 * (1.0 - nu)]])
        ne = self.conn.shape[0]
        Ke = np.zeros((ne, 8, 8))
        for (xi, eta), w in zip(_QP, _QW):
            dN, det, _ = _shape_derivs(self.xy, xi, eta)
            B = self._B(dN)
            Ke += w * np.einsum("eik,ij,ejl,e->ekl", B, D, B, det)
        self.D = D
        return self._scatter(Ke, self.mesh.n_nodes, dofs_per_node=2)

    @staticmethod
    def _B(dN):
        """Strain-displacement matrix, [n_elem, 3, 8]."""
        ne = dN.shape[0]
        B = np.zeros((ne, 3, 8))
        B[:, 0, 0::2] = dN[..., 0]
        B[:, 1, 1::2] = dN[..., 1]
        B[:, 2, 0::2] = dN[..., 1]
        B[:, 2, 1::2] = dN[..., 0]
        return B

    # -- boundary faces -----------------------------------------------------
    def _face(self, which: str):
        """(node pairs, lengths) along one j edge of the mesh."""
        if which in self._face_cache:
            return self._face_cache[which]
        ni, nj = self.mesh.shape
        j = 0 if which == "inner" else nj
        i = np.arange(ni)
        a, b = self.mesh.nid(i, j), self.mesh.nid(i + 1, j)
        p = self.mesh.nodes.reshape(-1, 2)
        L = np.linalg.norm(p[b] - p[a], axis=1)
        self._face_cache[which] = (a, b, L)
        return self._face_cache[which]

    def _robin(self, which: str, h, T_gas):
        """Robin surface terms: (K_contrib, f_contrib) for -k dT/dn = h (T - T_gas).

        Consistent 2-node line element mass matrix [[2,1],[1,2]] L/6."""
        a, b, L = self._face(which)
        n = self.mesh.n_nodes
        h = np.broadcast_to(np.asarray(h, dtype=float), a.shape)
        Tg = np.broadcast_to(np.asarray(T_gas, dtype=float), a.shape)
        m = np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0
        rows, cols, vals = [], [], []
        for r, nr in enumerate((a, b)):
            for c, nc in enumerate((a, b)):
                rows.append(nr); cols.append(nc); vals.append(h * L * m[r, c])
        K = csr_matrix(coo_matrix((np.concatenate(vals),
                                   (np.concatenate(rows), np.concatenate(cols))),
                                  shape=(n, n)))
        f = np.zeros(n)
        np.add.at(f, a, h * Tg * L / 2.0)
        np.add.at(f, b, h * Tg * L / 2.0)
        return K, f

    # -- solves -------------------------------------------------------------
    def step_thermal(self, T: np.ndarray, dt: float, h_in, T_gas_in, h_out, T_gas_out,
                     radiate: bool = True, T_inf: float = 250.0) -> np.ndarray:
        """Backward Euler: (M/dt + K + K_robin) T^{n+1} = M/dt T^n + f_robin.

        Radiation is linearized about the current wall temperature (h_rad =
        eps sigma (T^2 + Tinf^2)(T + Tinf)) and folded into the outer h."""
        Kb_i, fb_i = self._robin("inner", h_in, T_gas_in)
        h_o = np.asarray(h_out, dtype=float)
        if radiate:
            a, b, _ = self._face("outer")
            Tw = 0.5 * (T[a] + T[b])
            h_rad = self.mat.emissivity * 5.670374419e-8 * (Tw ** 2 + T_inf ** 2) * (Tw + T_inf)
            h_o = np.broadcast_to(h_o, a.shape) + h_rad
        Kb_o, fb_o = self._robin("outer", h_o, T_gas_out)

        A = self.M_th / dt + self.K_th + Kb_i + Kb_o
        rhs = self.M_th.dot(T) / dt + fb_i + fb_o
        return _cg(A, rhs, T)

    def solve_mechanical(self, T: np.ndarray, p_in, p_out, T_ref: float = 288.15,
                         clamp_nodes: np.ndarray | None = None):
        """Quasi-static plane stress with thermal expansion.

        Returns (u [n_nodes, 2], sigma [n_elem, 3] as (s_zz, s_yy, s_zy))."""
        mat = self.mat
        f = np.zeros(self.mesh.n_nodes * 2)

        # thermal body force: integral B^T D eps_th
        eps_th_coeff = mat.alpha
        for (xi, eta), w in zip(_QP, _QW):
            dN, det, N = _shape_derivs(self.xy, xi, eta)
            B = self._B(dN)
            dT = (T[self.conn] * N[None, :]).sum(1) - T_ref
            eps0 = np.zeros((self.conn.shape[0], 3))
            eps0[:, 0] = eps0[:, 1] = eps_th_coeff * dT
            fe = w * np.einsum("eik,ij,ej,e->ek", B, self.D, eps0, det)
            d = np.stack([2 * self.conn, 2 * self.conn + 1], axis=-1).reshape(-1, 8)
            np.add.at(f, d.ravel(), fe.ravel())

        # pressure traction on both faces (inward normal * p)
        for which, p, sign in (("inner", p_in, +1.0), ("outer", p_out, -1.0)):
            a, b, L = self._face(which)
            pv = np.broadcast_to(np.asarray(p, dtype=float), a.shape)
            nodes = self.mesh.nodes.reshape(-1, 2)
            tang = nodes[b] - nodes[a]
            nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1) / np.maximum(L, 1e-30)[:, None]
            trac = sign * pv[:, None] * nrm * L[:, None] * 0.5
            for nid in (a, b):
                np.add.at(f, 2 * nid, trac[:, 0])
                np.add.at(f, 2 * nid + 1, trac[:, 1])

        if clamp_nodes is None:
            # Free body: remove the three planar rigid-body modes by PROJECTION,
            # not by pinning nodes. Pinning is the obvious move and it is wrong
            # here -- any pin that also blocks a component of uniform thermal
            # expansion manufactures stress in a plate that should have none,
            # which is exactly the M1 free-expansion oracle.
            u = _solve_free(self.K_me, f, self._rigid_modes(), self._factors)
        else:
            fixed = sorted({int(d) for n in clamp_nodes for d in (2 * n, 2 * n + 1)})
            u = _solve_constrained(self.K_me, f, fixed)

        dN, det, N = _shape_derivs(self.xy, 0.0, 0.0)
        B = self._B(dN)
        d = np.stack([2 * self.conn, 2 * self.conn + 1], axis=-1).reshape(-1, 8)
        eps = np.einsum("eik,ek->ei", B, u[d])
        dT = (T[self.conn] * N[None, :]).sum(1) - T_ref
        eps0 = np.zeros_like(eps)
        eps0[:, 0] = eps0[:, 1] = mat.alpha * dT
        sigma = np.einsum("ij,ej->ei", self.D, eps - eps0)
        return u.reshape(-1, 2), sigma


    def _rigid_modes(self) -> np.ndarray:
        """Orthonormal basis of the planar rigid-body null space: two
        translations and one rotation, as [2*n_nodes, 3]."""
        p = self.mesh.nodes.reshape(-1, 2)
        n = p.shape[0]
        V = np.zeros((2 * n, 3))
        V[0::2, 0] = 1.0
        V[1::2, 1] = 1.0
        c = p.mean(0)
        V[0::2, 2] = -(p[:, 1] - c[1])
        V[1::2, 2] = (p[:, 0] - c[0])
        q, _ = np.linalg.qr(V)
        return q


def _cg(A, b, x0):
    M = 1.0 / np.maximum(A.diagonal(), 1e-30)
    x, info = cg(A, b, x0=x0, rtol=1e-10, maxiter=5000, M=_diag_prec(M))
    if info != 0:
        raise RuntimeError(f"CG failed to converge (info={info})")
    return x


def _diag_prec(dinv):
    from scipy.sparse.linalg import LinearOperator
    n = dinv.size
    return LinearOperator((n, n), matvec=lambda v: dinv * v)


def _solve_free(K, f, V, factor_cache: dict | None = None):
    """Solve K u = f for a body with no essential BCs.

    K is singular on the three planar rigid-body modes V. Solve the bordered
    system with Lagrange multipliers,

        [ K   V ] [u]   [f]
        [ V^T 0 ] [l] = [0]

    which is nonsingular, sparse, and DIRECT. An iterative deflation
    (K + c V V^T, Jacobi-preconditioned CG) works on a square test plate and
    fails outright on the real shell -- its elements are 20:1 slivers (20.3 mm x
    1 mm) and the conditioning that follows exhausts 20000 CG iterations. The
    stiffness matrix is fixed per mesh, so the factorization is computed once."""
    from scipy.sparse import bmat, csc_matrix
    from scipy.sparse.linalg import splu

    key = "free_lu"
    lu = (factor_cache or {}).get(key)
    if lu is None:
        Vs = csc_matrix(V)
        A = bmat([[K, Vs], [Vs.T, None]], format="csc")
        lu = splu(A)
        if factor_cache is not None:
            factor_cache[key] = lu
    rhs = np.concatenate([f, np.zeros(V.shape[1])])
    return lu.solve(rhs)[: f.size]


def _solve_constrained(K, f, fixed):
    from scipy.sparse.linalg import spsolve
    n = f.size
    free = np.setdiff1d(np.arange(n), np.asarray(fixed, dtype=int))
    u = np.zeros(n)
    if free.size:
        u[free] = spsolve(K[free][:, free].tocsc(), f[free])
    return u


def shell_mesh_from_block(nodes: np.ndarray) -> ShellMesh:
    return ShellMesh(nodes)


__all__ = ["SolidMaterial", "ShellMesh", "ThermoStruct2D", "shell_mesh_from_block"]
