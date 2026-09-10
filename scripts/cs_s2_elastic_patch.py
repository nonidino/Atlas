"""CS-S2 -- the classical control: linear elasticity on NeuberNet's own patch.

NeuberNet (Grossi, Beghini & Benedetti, *Communications Engineering* 2025) maps
the displacement on a circle of radius 5 R_n around a V-notch tip to the
elastic-plastic stress field inside it.  Read as an Atlas expert, that circle is
a seam, and the map restricted to the circle is a Dirichlet-to-Neumann operator.
Whether such an operator can be COMPACT along its own boundary is first a
question about the physics -- an elliptic solve couples every boundary point to
every other one through the interior -- and this module answers that half with
no learned expert anywhere in it: the same patch, the same 36 sensors, the same
gap-filling convention, and linear elasticity solved classically.

Geometry, read from `fem/s2_ansys_database_macro.mac` and
`database/s3_import_database.py` of github.com/grossIt/neubernet (read, not
copied):

  * lengths in units of the notch radius, R_n = 1;
  * the local frame sits at the notch TIP (APDL `LOCAL,12` / `LOCAL,13`), x
    radial away from the shaft axis, y axial, z hoop; at beta = 0 it coincides
    with the global frame, and beta = 0 is the only case used here;
  * the root circle (`LOCAL,11`) is centred one notch radius beyond the tip
    along the bisector, and the flanks are tangent to it at C-angles 90+alpha
    and -(90+alpha), so the notch opens towards +x;
  * the shaft axis is at x = -R (`D_notch/2 = R_midnotch` in notch radii);
  * the sub-model is the material inside |p| <= RL = 5;
  * sensors at -180, -170, ..., 170 deg on |p| = 5; those inside the opening
    carry no boundary data, and the pipeline fills them by linear interpolation
    across the gap (`np.interp` over the material arc).

Formulation: small-strain isotropic elasticity, axisymmetric with torsion (the
physics of ANSYS PLANE183 KEYOPT(3)=6), linear triangles, three-point
quadrature for the 1/rho terms.  For an isotropic material the in-plane pair
(u_rho, u_y) and the torsion u_theta decouple and are solved separately.
E = 1, so a traction is in units of E and a displacement in units of R_n --
which makes this operator directly comparable with NeuberNet's, whose trace is
u E / (sigma_y R_n) and whose flux is sigma / sigma_y.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

RL = 5.0
SENSOR_ANGLES = np.arange(-180.0, 180.0, 10.0)
COMPONENTS = ("ux", "uy", "ut")
#: three-point, degree-2 rule on the reference triangle; each weight is 1/3
QUAD = np.array([[2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
                 [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                 [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]])


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def flank_geometry(alpha_deg):
    """Tangent points, flank directions and inward (into-the-opening) normals."""
    a = math.radians(alpha_deg)
    T1 = np.array([1.0 - math.sin(a), math.cos(a)])
    T2 = np.array([1.0 - math.sin(a), -math.cos(a)])
    d1 = np.array([math.cos(a), math.sin(a)])
    d2 = np.array([math.cos(a), -math.sin(a)])
    n1 = np.array([math.sin(a), -math.cos(a)])
    n2 = np.array([math.sin(a), math.cos(a)])
    return T1, T2, d1, d2, n1, n2


def in_void(x, y, alpha_deg):
    """True inside the notch: the root disc, or the opening between the flanks."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if alpha_deg is None:
        return np.zeros(np.broadcast(x, y).shape, dtype=bool)
    T1, T2, _d1, _d2, n1, n2 = flank_geometry(alpha_deg)
    disc = (x - 1.0) ** 2 + y ** 2 < 1.0 - 1e-12
    wedge = ((n1[0] * (x - T1[0]) + n1[1] * (y - T1[1]) > 1e-12)
             & (n2[0] * (x - T2[0]) + n2[1] * (y - T2[1]) > 1e-12)
             & (x >= T1[0]))
    return disc | wedge


def void_half_angle(alpha_deg, r=RL):
    """Half-width, in degrees, of the opening as seen on the circle |p| = r."""
    a = math.radians(alpha_deg)
    return math.degrees(a + math.asin((1.0 - math.sin(a)) / r))


def material_sensors(alpha_deg, r=RL):
    """Sensor angles that lie in material, in arc order.

    Returns ``(theta_deg, phi_deg)``: theta in [-180, 180) as the pipeline
    writes it, and phi the unwrapped arc coordinate, increasing along the
    material arc from one flank to the other.  With no notch the ring is closed
    and phi = theta.
    """
    th = SENSOR_ANGLES.copy()
    if alpha_deg is None:
        return th, th.copy()
    x = r * np.cos(np.radians(th))
    y = r * np.sin(np.radians(th))
    keep = ~in_void(x, y, alpha_deg)
    th = th[keep]
    phi = np.mod(th, 360.0)
    order = np.argsort(phi)
    return th[order], phi[order]


def ring_basis(phi_eval, alpha_deg, r=RL):
    """Material-sensor hats along the ring, with the opening filled as the pipeline fills it.

    Column j is the boundary displacement produced by a unit value at material
    sensor j and zero at the others: the sensors inside the notch opening take
    the straight line across the gap (``np.interp`` over the material arc, as
    `database/s3_import_database.py` does), and between sensors the
    displacement is linear in angle.  So a flank-end sensor's hat does not stop
    at the flank; it continues along the gap line, which is what NeuberNet is
    shown.  The first CS-S2 check clamped it instead, and the two subjects then
    disagreed at the flank-end sensors only.  With no notch these are the
    periodic hats.
    """
    th_s, phi_s = material_sensors(alpha_deg, r)
    allphi = np.mod(SENSOR_ANGLES, 360.0)
    mat = np.array([int(np.flatnonzero(np.isclose(SENSOR_ANGLES, t))[0]) for t in th_s])
    void = np.setdiff1d(np.arange(SENSOR_ANGLES.size), mat)
    order = np.argsort(allphi)
    phi_eval = np.asarray(phi_eval, dtype=float)
    B = np.empty((phi_eval.size, th_s.size))
    for j in range(th_s.size):
        e = np.zeros(th_s.size)
        e[j] = 1.0
        full = np.zeros(SENSOR_ANGLES.size)
        full[mat] = e
        if void.size:
            full[void] = np.interp(allphi[void], np.mod(phi_s, 360.0), e, period=360.0)
        B[:, j] = np.interp(phi_eval, allphi[order], full[order], period=360.0)
    return B


# ---------------------------------------------------------------------------
# mesh and assembly
# ---------------------------------------------------------------------------


def mesh_patch(alpha_deg, r_out=RL, h_tip=0.04, grad=0.03, h_max=0.2):
    """Linear triangles on the material inside |p| <= r_out.  gmsh, OCC kernel."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.model.add("cs_s2_patch")
        occ = gmsh.model.occ
        dom = [(2, occ.addDisk(0.0, 0.0, 0.0, r_out, r_out))]
        if alpha_deg is not None:
            T1, T2, d1, d2, _n1, _n2 = flank_geometry(alpha_deg)
            L = 3.0 * r_out
            P = [T1, T1 + L * d1, T2 + L * d2, T2]
            pt = [occ.addPoint(float(p[0]), float(p[1]), 0.0) for p in P]
            ln = [occ.addLine(pt[i], pt[(i + 1) % 4]) for i in range(4)]
            wedge = occ.addPlaneSurface([occ.addCurveLoop(ln)])
            root = occ.addDisk(1.0, 0.0, 0.0, 1.0, 1.0)
            void, _ = occ.fuse([(2, root)], [(2, wedge)])
            dom, _ = occ.cut(dom, void)
        occ.synchronize()
        fld = gmsh.model.mesh.field
        fid = fld.add("MathEval")
        fld.setString(fid, "F", "%.17g + %.17g*Sqrt(x*x + y*y)" % (h_tip, grad))
        fld.setAsBackgroundMesh(fid)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h_max)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.model.mesh.generate(2)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        _et, enodes = gmsh.model.mesh.getElementsByType(2)
    finally:
        gmsh.finalize()
    tags = np.asarray(tags, dtype=np.int64)
    xy = np.asarray(coords, dtype=float).reshape(-1, 3)[:, :2]
    lut = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
    lut[tags] = np.arange(tags.size)
    tri = lut[np.asarray(enodes, dtype=np.int64).reshape(-1, 3)]
    used = np.unique(tri)
    remap = np.full(xy.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    return xy[used], remap[tri]


def lame(nu, E=1.0):
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


def assemble(nodes, tris, R, nu, E=1.0):
    """In-plane (2N x 2N) and torsion (N x N) stiffness, rho-weighted."""
    X = nodes[tris, 0]
    Y = nodes[tris, 1]
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    y1, y2, y3 = Y[:, 0], Y[:, 1], Y[:, 2]
    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    area = 0.5 * np.abs(det)
    b = np.stack([y2 - y3, y3 - y1, y1 - y2], axis=1) / det[:, None]
    c = np.stack([x3 - x2, x1 - x3, x2 - x1], axis=1) / det[:, None]
    lam, mu = lame(nu, E)
    D = np.array([[lam + 2 * mu, lam, lam, 0.0],
                  [lam, lam + 2 * mu, lam, 0.0],
                  [lam, lam, lam + 2 * mu, 0.0],
                  [0.0, 0.0, 0.0, mu]])
    T = tris.shape[0]
    Ke = np.zeros((T, 6, 6))
    Kt = np.zeros((T, 3, 3))
    for q in range(3):
        Lq = QUAD[q]
        rho = R + X @ Lq
        if np.any(rho <= 0.0):
            raise ValueError("the patch reaches the shaft axis")
        B = np.zeros((T, 4, 6))
        B[:, 0, 0::2] = b
        B[:, 1, 1::2] = c
        B[:, 2, 0::2] = Lq[None, :] / rho[:, None]
        B[:, 3, 0::2] = c
        B[:, 3, 1::2] = b
        w = (area * rho / 3.0)[:, None, None]
        Ke += w * np.einsum("tki,kl,tlj->tij", B, D, B)
        G = np.zeros((T, 2, 3))
        G[:, 0, :] = b - Lq[None, :] / rho[:, None]
        G[:, 1, :] = c
        Kt += w * mu * np.einsum("tki,tkj->tij", G, G)
    N = nodes.shape[0]
    dofs = np.empty((T, 6), dtype=np.int64)
    dofs[:, 0::2] = 2 * tris
    dofs[:, 1::2] = 2 * tris + 1
    K_in = sp.csr_matrix((Ke.ravel(), (np.repeat(dofs, 6, axis=1).ravel(),
                                       np.tile(dofs, (1, 6)).ravel())),
                         shape=(2 * N, 2 * N))
    K_t = sp.csr_matrix((Kt.ravel(), (np.repeat(tris, 3, axis=1).ravel(),
                                      np.tile(tris, (1, 3)).ravel())),
                        shape=(N, N))
    return K_in, K_t, (b, c, area)


def boundary_edges(tris):
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e.sort(axis=1)
    u, cnt = np.unique(e, axis=0, return_counts=True)
    return u[cnt == 1]


def homogeneous_field(nodes, R, nu, axial=0.0, shear=0.0, E=1.0):
    """The exact axisymmetric solution for a uniform far field.

    ``axial`` is sigma_yy (uniaxial tension along the shaft), ``shear`` is the
    torsional shear tau_theta_y at rho = R.  u_rho = -nu (s/E) rho,
    u_y = (s/E) y, u_theta = kappa rho y with tau_theta_y = mu kappa rho.
    """
    _lam, mu = lame(nu, E)
    rho = R + nodes[:, 0]
    y = nodes[:, 1]
    kappa = shear / (mu * R)
    u = np.stack([-nu * (axial / E) * rho, (axial / E) * y], axis=1)
    return u, kappa * rho * y


# ---------------------------------------------------------------------------
# the patch
# ---------------------------------------------------------------------------


class ElasticPatch:
    """Classical elastic sub-model, Dirichlet on its outer arc, free elsewhere."""

    def __init__(self, alpha_deg=30.0, R=50.0, nu=0.3, r_out=RL, h_tip=0.04,
                 grad=0.03, h_max=0.2, agent_id="ELASTIC"):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.tri as mtri

        self.alpha = alpha_deg
        self.R = float(R)
        self.nu = float(nu)
        self.r_out = float(r_out)
        self.agent_id = agent_id
        self.mesh = dict(h_tip=h_tip, grad=grad, h_max=h_max)
        self.nodes, self.tris = mesh_patch(alpha_deg, r_out, h_tip, grad, h_max)
        self.K_in, self.K_t, self.geo = assemble(self.nodes, self.tris, self.R, self.nu)
        N = self.nodes.shape[0]
        r = np.hypot(self.nodes[:, 0], self.nodes[:, 1])
        on_arc = np.abs(r - self.r_out) < 1e-6 * self.r_out
        ang = np.degrees(np.arctan2(self.nodes[:, 1], self.nodes[:, 0]))
        phi = ang if alpha_deg is None else np.mod(ang, 360.0)
        arc = np.where(on_arc)[0]
        arc = arc[np.argsort(phi[arc])]
        self.B = arc
        self.phi_B = phi[arc]
        self.I = np.setdiff1d(np.arange(N), arc)
        B2 = np.concatenate([2 * arc, 2 * arc + 1])
        I2 = np.concatenate([2 * self.I, 2 * self.I + 1])
        self._B2, self._I2 = B2, I2
        Kc = self.K_in.tocsc()
        self._lu_in = spla.splu(Kc[I2][:, I2].tocsc())
        self._K_IB_in = Kc[I2][:, B2].tocsr()
        Ktc = self.K_t.tocsc()
        self._lu_t = spla.splu(Ktc[self.I][:, self.I].tocsc())
        self._K_IB_t = Ktc[self.I][:, arc].tocsr()
        # consistent, rho-weighted boundary mass on the arc
        loc = np.full(N, -1, dtype=np.int64)
        loc[arc] = np.arange(arc.size)
        edges = boundary_edges(self.tris)
        edges = edges[(loc[edges[:, 0]] >= 0) & (loc[edges[:, 1]] >= 0)]
        rows, cols, vals = [], [], []
        for a, bnode in edges:
            ell = float(np.hypot(*(self.nodes[a] - self.nodes[bnode])))
            ra, rb = self.R + self.nodes[a, 0], self.R + self.nodes[bnode, 0]
            ia, ib = loc[a], loc[bnode]
            rows += [ia, ib, ia, ib]
            cols += [ia, ib, ib, ia]
            vals += [ell * (3 * ra + rb) / 12.0, ell * (ra + 3 * rb) / 12.0,
                     ell * (ra + rb) / 12.0, ell * (ra + rb) / 12.0]
        self._Mb_mat = sp.csc_matrix((vals, (rows, cols)), shape=(arc.size, arc.size))
        self._Mb = spla.splu(self._Mb_mat)
        self.theta_s, self.phi_s = material_sensors(alpha_deg, self.r_out)
        self.M = self.n = int(self.theta_s.size)
        th = np.radians(self.theta_s)
        self.points = np.stack([self.r_out * np.cos(th), self.r_out * np.sin(th)], axis=1)
        self.normals = np.stack([np.cos(th), np.sin(th)], axis=1)
        # the sensor basis on the arc: hats, continued across the notch opening
        # along the line the pipeline fills the opening with
        self.P = ring_basis(self.phi_B, alpha_deg, self.r_out)
        #: int phi_j rho ds, from the consistent boundary mass
        self.hat_mass = np.asarray(self._Mb_mat @ self.P).sum(axis=0)
        self._tri = mtri.Triangulation(self.nodes[:, 0], self.nodes[:, 1], self.tris)
        self._mtri = mtri

    # -- boundary maps ------------------------------------------------------

    @property
    def closed(self):
        return self.alpha is None

    def prolong(self, v_sensors):
        """Sensor values -> arc nodal values, in the basis `ring_basis` builds."""
        return self.P @ np.asarray(v_sensors, dtype=float)

    def restrict(self, v_nodes):
        kw = {"period": 360.0} if self.closed else {}
        return np.interp(self.phi_s, self.phi_B, v_nodes, **kw)

    # -- solves -------------------------------------------------------------

    def solve_nodal(self, ubx, uby, ubt):
        """Boundary nodal displacements -> fields, arc tractions (mass-inverted), arc reactions."""
        N = self.nodes.shape[0]
        nb = self.B.size
        u = np.zeros(2 * N)
        ub = np.concatenate([ubx, uby])
        u[self._B2] = ub
        u[self._I2] = self._lu_in.solve(-(self._K_IB_in @ ub))
        f = self.K_in @ u
        ut = np.zeros(N)
        ut[self.B] = ubt
        ut[self.I] = self._lu_t.solve(-(self._K_IB_t @ ubt))
        ft = self.K_t @ ut
        fB = np.stack([f[self._B2[:nb]], f[self._B2[nb:]], ft[self.B]], axis=1)
        t = np.stack([self._Mb.solve(fB[:, k]) for k in range(3)], axis=1)
        return u.reshape(N, 2), ut, t, fB

    def solve_sensors(self, s):
        """``s`` is (M, 3) sensor displacements [ux, uy, ut]."""
        return self.solve_nodal(self.prolong(s[:, 0]), self.prolong(s[:, 1]),
                                self.prolong(s[:, 2]))

    def sensor_traction(self, s):
        """POINTWISE traction at the sensor angles.  Kept for the record, not the port.

        A hat trace has slope jumps at the sensors, and a Dirichlet-to-Neumann
        response to a slope jump is logarithmically singular exactly there, so
        this sampling does not converge under mesh refinement -- measured at
        36-42% between 4890 and 19119 nodes in CS-S2's first run.  `sensor_flux`
        is the restriction that does.
        """
        _u, _ut, t, _f = self.solve_sensors(s)
        return np.stack([self.restrict(t[:, k]) for k in range(3)], axis=1)

    def sensor_flux(self, s):
        """Hat-weighted traction, ``int t phi_j rho ds / int phi_j rho ds``.

        The Galerkin dual of the hat trace and the port's flux: the reactions
        ARE ``int t N_a rho ds``, so ``P^T f`` integrates the traction against
        each sensor's hat with no pointwise sampling anywhere.
        """
        _u, _ut, _t, fB = self.solve_sensors(s)
        return (self.P.T @ fB) / self.hat_mass[:, None]

    def _as_sensors(self, port_name, trace):
        comp = port_name.split(":", 1)[1]
        tr = np.asarray(trace, dtype=float).ravel()
        s = np.zeros((self.M, 3))
        if comp == "all":
            for k in range(3):
                s[:, k] = tr[k * self.M:(k + 1) * self.M]
        else:
            s[:, COMPONENTS.index(comp)] = tr
        return comp, s

    def respond(self, port_name, trace):
        """The w173 instrument's interface: a LINEAR map, so any base is zero."""
        comp, s = self._as_sensors(port_name, trace)
        f = self.sensor_flux(s)
        return f.T.ravel() if comp == "all" else f[:, COMPONENTS.index(comp)]

    def respond_pointwise(self, port_name, trace):
        comp, s = self._as_sensors(port_name, trace)
        f = self.sensor_traction(s)
        return f.T.ravel() if comp == "all" else f[:, COMPONENTS.index(comp)]

    # -- fields -------------------------------------------------------------

    def element_stress(self, u_xy, u_t):
        """[xx, yy, zz, xy, yz, xz] per element, at the centroid."""
        b, c, _area = self.geo
        lam, mu = lame(self.nu)
        t = self.tris
        ux, uy, uth = u_xy[t, 0], u_xy[t, 1], u_t[t]
        rho = self.R + self.nodes[t, 0].mean(axis=1)
        e_rr = (b * ux).sum(1)
        e_yy = (c * uy).sum(1)
        e_tt = ux.mean(1) / rho
        g_ry = (c * ux).sum(1) + (b * uy).sum(1)
        s_rr = (lam + 2 * mu) * e_rr + lam * (e_yy + e_tt)
        s_yy = (lam + 2 * mu) * e_yy + lam * (e_rr + e_tt)
        s_tt = (lam + 2 * mu) * e_tt + lam * (e_rr + e_yy)
        g_rt = (b * uth).sum(1) - uth.mean(1) / rho
        g_ty = (c * uth).sum(1)
        return np.stack([s_rr, s_yy, s_tt, mu * g_ry, mu * g_ty, mu * g_rt], axis=1)

    def nodal_stress(self, u_xy, u_t):
        se = self.element_stress(u_xy, u_t)
        _b, _c, area = self.geo
        acc = np.zeros((self.nodes.shape[0], 6))
        wgt = np.zeros(self.nodes.shape[0])
        for k in range(3):
            np.add.at(acc, self.tris[:, k], se * area[:, None])
            np.add.at(wgt, self.tris[:, k], area)
        return acc / wgt[:, None]

    def interpolate(self, nodal, points):
        """Linear interpolation of nodal fields; NaN outside the mesh."""
        nodal = np.atleast_2d(nodal.T).T if nodal.ndim == 1 else nodal
        out = np.full((points.shape[0], nodal.shape[1]), np.nan)
        for k in range(nodal.shape[1]):
            it = self._mtri.LinearTriInterpolator(self._tri, nodal[:, k])
            v = it(points[:, 0], points[:, 1])
            out[:, k] = np.ma.filled(v.astype(float), np.nan)
        return out

    def stress_at_sensors_bc(self, s, points):
        u, ut, _t, _f = self.solve_sensors(s)
        return self.interpolate(self.nodal_stress(u, ut), points)


# ---------------------------------------------------------------------------
# self-test: the patch test, run as the module's own positive control
# ---------------------------------------------------------------------------


def self_test(R=50.0, nu=0.3):
    """A uniform far field on a notch-free disc must come back exactly.

    The axial field is linear in (x, y) and therefore in the P1 space: the
    interior displacement must match to solver precision, and the traction
    recovered from the reactions must match sigma . n up to the interpolation
    of a smooth function along the arc.  The torsion field u_theta = kappa rho y
    is quadratic and is not in the space, so its error is first order in h and
    is reported rather than asserted.
    """
    out = {}
    for h in (0.2, 0.1):
        ep = ElasticPatch(alpha_deg=None, R=R, nu=nu, h_tip=h, grad=0.0, h_max=h)
        u_ex, ut_ex = homogeneous_field(ep.nodes, R, nu, axial=1.0, shear=1.0)
        u, ut, t, fB = ep.solve_nodal(u_ex[ep.B, 0], u_ex[ep.B, 1], ut_ex[ep.B])
        _lam, mu = lame(nu)
        th = np.arctan2(ep.nodes[ep.B, 1], ep.nodes[ep.B, 0])
        rho_b = R + ep.nodes[ep.B, 0]
        ty_ex = 1.0 * np.sin(th)
        tt_ex = mu * (1.0 / (mu * R)) * rho_b * np.sin(th)
        out["h=%g" % h] = {
            "nodes": int(ep.nodes.shape[0]),
            "axial_u_max_err": float(np.max(np.abs(u - u_ex))),
            "axial_tx_max_err": float(np.max(np.abs(t[:, 0]))),
            "axial_ty_max_err": float(np.max(np.abs(t[:, 1] - ty_ex))),
            "torsion_ut_max_err": float(np.max(np.abs(ut - ut_ex))),
            "torsion_tt_max_err": float(np.max(np.abs(t[:, 2] - tt_ex))),
        }
        # the Galerkin flux of the exact field against the exact hats, by quadrature
        phq = np.linspace(-180.0, 180.0, 14401)[:-1]
        thq = np.radians(phq)
        eye = np.eye(ep.M)
        Hq = np.column_stack([np.interp(phq, ep.phi_s, eye[j], period=360.0)
                              for j in range(ep.M)])
        wq = (R + ep.r_out * np.cos(thq)) * ep.r_out * (2.0 * np.pi / phq.size)
        mq = (Hq * wq[:, None]).sum(axis=0)
        fy_ex = (Hq.T @ (np.sin(thq) * wq)) / mq
        ft_ex = (Hq.T @ ((R + ep.r_out * np.cos(thq)) / R * np.sin(thq) * wq)) / mq
        fG = (ep.P.T @ fB) / ep.hat_mass[:, None]
        out["h=%g" % h].update({
            "hat_mass_rel_err": float(np.max(np.abs(ep.hat_mass - mq) / mq)),
            "axial_flux_x_max_err": float(np.max(np.abs(fG[:, 0]))),
            "axial_flux_y_max_err": float(np.max(np.abs(fG[:, 1] - fy_ex))),
            "torsion_flux_max_err": float(np.max(np.abs(fG[:, 2] - ft_ex))),
        })
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(self_test(), indent=1))
