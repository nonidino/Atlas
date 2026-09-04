"""The NINTH real case study: what is the BOND for a two-way volumetric coupling?

`case-study-ladder-to-f1` schedules this as **CS-9**, Phase B's first row, and
calls it *"the single highest-leverage missing piece of vocabulary"* -- three of
the four Phase C subsystems (brake, tyre, battery) need it.  The question comes
from two ledger rows:

* **W70** closed by *reframing*.  A thermoelastic shell's conduction and
  elasticity are coupled through thermal strain, which is a **volume** term and
  not a surface bond -- but in `thermal_seam.py` that coupling is one expert's
  internals and one-way, so nothing needed a bond and the honest fix was to
  disclose the output (`VOLUMETRIC_COUPLING_SCOPE`).
* **W94** then found the case with no such escape.  An actuator disk against a
  frozen operator is **two-way volumetric**, and `atlas/ports.py` has five
  surface bonds and nothing else.

So: split `thermostruct2d.ThermoStruct2D` -- the same build-repo solver
`thermal_seam` uses -- into two agents, a **conduction** agent and an
**elasticity** agent, and require the thermal strain to cross as a *declared
bond* rather than as one solver's internals.  Exercise `PortAmendment`'s
six-field procedure (**W32**) for the first time; nothing has used it.

The control is free
-------------------

The unsplit `ThermoStruct2D` solve is right there, so there is no ground truth
to derive.  ``monolith()`` is the referent, ``split()`` is the graph, and the
gate is the stress field.

What comes out, in one paragraph
--------------------------------

The split is *exact*: carried in full, the volume term reproduces the monolith's
stress **bit for bit**, and lagged one macro-step it costs 7.1e-3 relative.  The
existing vocabulary can express the coupling **two** ways and both are wrong in
instructive directions: as a surface `MECH` traction it is wrong by three orders
of magnitude, and as a `GlobalField` it was numerically *exact* and turned off
every check the framework has -- until **W117** (2026-09-02) gave that route a
reader and it is now REFUSED at `L3/global-field`.  And the amendment **refuses**, on two of six
fields, with the refusal localized: thermal strain is a **bond** -- a conjugate
pair whose product is a power -- and is not a **port**, because a port is a bond
on an interface of co-dimension at least one between agents whose energies add,
and this one has co-dimension zero between agents whose energies do not add.
The free energy carries a bilinear cross term that belongs to neither agent and
is measured at 2680x the elastic energy; §6's residual sums dE_i/dt over agents
and therefore presumes an additive split that does not exist.

That is W70's conclusion reached again from the other side, and sharpened: W70
said *"a sixth port type is not the fix, because the object is not a bond"*.
Measured, the object **is** a bond and is not a **port**, and the distinction
says what to build instead -- an operator splitting with a splitting-error
bound, whose error this file measures, rather than a transmission condition.

The three routes, and what each one is
--------------------------------------

``volumetric``     the bond CS-9 wants.  ``build`` asks `ports.spec_for` for it
                   and the package REFUSES -- `NamedHoleError`, from
                   `holes.PORT_AMENDMENT`, at the line where a sixth type would
                   have to enter.  The refusal is the deliverable, and
                   `evaluate_amendment` is the six-field record behind it.
``surface-mech``   the best the closed vocabulary can do: the classical
                   equivalent-thermal-pressure reduction, ``t = beta dT n`` on
                   the body's own boundary, which is a real `MECH` bond on a real
                   surface.  Compiles.  Wrong by 1279x in stress, because the
                   exact identity ``G = G_surf + G_body`` leaves 9.6% of the load
                   as a body force no surface datum carries -- and the two halves
                   nearly cancel, so dropping one amplifies the displacement 233x.
                   The positive control passes: at a UNIFORM dT, ``G_body`` is
                   zero to 6.8e-17 of the load and this route is exact to 1.0e-13.
``global-field``   the other thing the closed vocabulary permits.  `GlobalField`
                   bypasses L3 entirely, so declaring the eigenstrain a global
                   field reproduces the monolith EXACTLY -- and no scale set, no
                   prolongation, no adjoint, no null space, no response half and
                   no tau, sigma or beta is ever asked for, because there is no
                   seam.  It was the silent-wrongness class in its purest form:
                   right answer, zero certificate, and nothing in the compile
                   said so.

                   **W117, 2026-09-02: the compile says so now.**  `GlobalField`
                   gained a `produced_by` field and `L3/global-field` reads it.
                   This route declares ``produced_by=("cond",)`` against
                   ``applies_to=("elas",)`` -- a proper subset applied outside
                   itself -- and is REFUSED.  The route is kept exactly as it was,
                   because a rule with nothing to fire on is not a rule: this is
                   the case study the check exists for, and `tests/
                   test_tier24_global_field_provenance.py` pins it against the six
                   legitimate global fields the same rule clears.

Two modes, and the second is what W94 actually asks about
---------------------------------------------------------

``one-way``   `ThermoStruct2D` as built.  ``step_thermal`` takes no displacement,
              so temperature drives deformation and deformation never returns.
              This is W70's shape, and it is the referent the task names.
``two-way``   the Biot / Gough-Joule term ``-T0 beta tr(eps_dot)`` restored in
              the energy equation.  `ThermoStruct2D` is NOT edited: the agent
              declines to call `step_thermal` and assembles the step itself from
              that solver's own ``M_th``, ``K_th`` and ``_robin`` blocks, which is
              `window_ns._no_projection_class` and
              `thermal_seam._explicit_shell_class`'s move without the subclass --
              the reverse term is a source, not a change to the operator.
              The reverse operator is ``G^T``: the SAME matrix
              transposed, which is what makes the pair a bond rather than two
              couplings, and it is exhibited here rather than asserted.  Measured
              on this trajectory the coupling number is 9.08e-3, the fixed point
              takes 7 iterations, and the reverse half moves T by 1.10 K and
              sigma by 3.8e-3 relative.

Geometry
--------

The same shell segment `thermal_seam` uses -- 0.20 m x 8 mm, Q1, aluminium-lithium
defaults -- at 48 x 6 elements, heated by a hot gas streak on the inner face
(Robin, h = 2000, a 600 K Gaussian of width 0.03 m) and cooled on the outer
(Robin plus radiation).  The streak is what makes the temperature field
genuinely two-dimensional: a uniform dT has ``G_body = 0`` exactly and every
question below collapses.  It is schematic, and a number measured here is a
number about the port algebra, not about an airframe.

Two things this file cannot say
-------------------------------

1. **The stress is a 121x cancellation.**  ``|D eps(u)|`` and ``|D eps_0|`` agree
   to three digits and sigma is the 0.8% residual.  Every relative error below
   is a relative error on that residual, which is the hard norm and the right
   one, and it is why the surface route's 9.6% load defect reads as 1279x.
2. **One material, one geometry, one linear constitutive law.**  The bilinear
   cross term is a fact about linear thermoelasticity; that *every* volumetric
   coupling has one is an [AI Inference] this file argues for and does not
   measure.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from ..capability import (
    BCChannel, ClaimType, Differentiable, Direction, EllipticSubsolve,
    ExpertCapabilities, MotionClass, TimeDiscretization, port_decl,
)
from ..graph import (Agent, CaseGraph, Connection, Decomposition, GlobalField,
                     MeasuredConstants)
from ..ports import PortType, ResponseHalf, spec_for
from ..transfer import Prolongation

DEFAULT_BUILD_REPO = os.path.join(os.path.expanduser("~"), "physics-foundation-model")

# ---------------------------------------------------------------------------
# the geometry and the drive -- schematic, and the streak is load-bearing
# ---------------------------------------------------------------------------

L_Z = 0.20          # along the shell, m
T_SHELL = 8.0e-3    # through the shell, m
NZ = 48             # elements along
NJ = 6              # elements through

T_REF = 288.15      # the stress-free reference temperature of solve_mechanical
T_INIT = 300.0      # the shell starts here
T_OUT = 300.0       # the outer ambient
H_IN = 2.0e3        # forced convection under the streak, W/(m^2 K)
H_OUT = 20.0        # natural convection outside
T_STREAK = 600.0    # the streak's amplitude above ambient, K
W_STREAK = 0.03     # the streak's Gaussian half-width, m

#: The conduction agent's macro step.  8 mm of aluminium diffuses in about 1.3 s
#: (t^2 / alpha with alpha = k / (rho cp) = 4.94e-5), and 0.03 m along the streak
#: in about 18 s, so 40 steps of 50 ms sits in the middle: the through-thickness
#: profile has equilibrated and the in-plane one has not.  That is where the
#: Laplacian of T is largest and therefore where the stress is.
DT_MACRO = 5.0e-2
N_STEPS = 40

#: The number of interface modes the surface routes carry.  Same 16-mode real
#: Fourier declaration as every other case study, for `thermal_seam`'s reason:
#: comparing experts through two different presentations confounds the expert
#: with the presentation.
M_EFF = 16

#: Explicit conduction sub-steps under ``split-step``.  Computed, never
#: hard-coded -- `CASE-STUDY-GUIDE` mistake 5.
def substeps_at(dt: float) -> int:
    alpha = 120.0 / (2700.0 * 900.0)
    dy = T_SHELL / NJ
    dt_stable = 0.4 * dy * dy / (2.0 * alpha)
    return max(1, int(np.ceil(dt / dt_stable)))


def build_repo() -> str:
    return os.environ.get("ATLAS_BUILD_REPO", DEFAULT_BUILD_REPO)


@lru_cache(maxsize=1)
def load_solvers(module_name: str = "atlas_build_solvers"):
    """Import the build repo's ``atlas`` package under a private name.

    Identical to `thermal_seam.load_solvers`, deliberately: both repositories
    have a top-level package called ``atlas``, and binding the whole build-repo
    package under a name that collides with nothing is the only way `grid.py`'s
    ``..config`` imports resolve.  Sharing the module name means the two case
    studies share one import, which is what keeps `ThermoStruct2D` identical
    between them.
    """
    pkg_dir = os.path.join(build_repo(), "src", "atlas")
    init = os.path.join(pkg_dir, "__init__.py")
    if not os.path.isfile(init):
        raise RuntimeError(
            f"the expert is not where this case study expects it: {init!r} does not "
            "exist. thermostruct2d lives in the build repo "
            "(github.com/nonidino/physics-foundation-model, src/atlas/solvers/); "
            "set ATLAS_BUILD_REPO to the checkout."
        )
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            module_name, init, submodule_search_locations=[pkg_dir]
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
    return importlib.import_module(module_name + ".solvers.thermostruct2d")


# ---------------------------------------------------------------------------
# the coupling operator, extracted from the expert rather than re-derived
# ---------------------------------------------------------------------------


@dataclass
class CouplingOperators:
    """The thermal-strain coupling, as three sparse matrices and one identity.

    ``G`` maps a nodal temperature rise to the thermal load ``solve_mechanical``
    assembles inside itself:  ``f_th = G (T - T_ref)``.  It is not new physics --
    it is that method's own quadrature loop, lifted out so the term can be
    *named*, which is what "carry it across as a declared bond" requires.  It is
    asserted against the expert in `verify_against_expert`: the residual of
    ``K u = (I - V V^T) G dT`` on the expert's own free-body solve is 7e-15.

    ``G_surf`` and ``G_body`` are its two halves under the divergence theorem,

        int_Omega beta dT grad(N_k) dV
            = oint_dOmega beta dT N_k n ds  -  int_Omega N_k beta grad(dT) dV
              \\____________ G_surf ______/     \\_________ G_body _________/

    and the identity ``G = G_surf + G_body`` holds to 1e-15 by construction.
    **That identity is the whole distinctness argument** (`PortAmendment` field
    5): ``G_surf`` is exactly what a surface `MECH` bond on the body's own
    boundary can deliver, ``G_body`` is exactly what it cannot, and ``G_body``
    vanishes identically when dT is uniform -- which is both the positive
    control and the reason W70's one-way case never noticed.

    ``H`` is the eigenstrain self-energy, ``int eps_0 : D : eps_0``, needed only
    for the stored elastic energy and for the second conjugate reading.  It is
    proportional to the conduction mass matrix and needs no assembly:
    ``a1^T D a1 = 2 E / (1 - nu)`` with ``a1 = (1, 1, 0)``.
    """

    G: Any
    G_surf: Any
    G_body: Any
    H: Any
    beta: float          # E alpha / (1 - nu), the thermal stress modulus, Pa/K
    delta: float         # T0 beta^2 / (rho cp D_plane), the Biot coupling number

    def verify_against_expert(self, ts, dT: np.ndarray) -> dict[str, float]:
        """Assert that G is the expert's own thermal load, not a re-derivation."""
        V = ts._rigid_modes()
        f = self.G.dot(dT)
        u, _sig = ts.solve_mechanical(dT + T_REF, 0.0, 0.0, T_ref=T_REF)
        resid = ts.K_me.dot(u.ravel()) - (f - V @ (V.T @ f))
        split = self.G.dot(dT) - (self.G_surf.dot(dT) + self.G_body.dot(dT))
        return {
            "G_vs_solve_mechanical": float(np.linalg.norm(resid) / np.linalg.norm(f)),
            "rigid_projection": float(np.linalg.norm(V.T @ u.ravel())),
            "G_surf_plus_G_body": float(np.linalg.norm(split) / np.linalg.norm(f)),
        }


def _scatter_G(rows, cols, vals, n_nodes: int):
    return csr_matrix(
        coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(2 * n_nodes, n_nodes),
        )
    )


@lru_cache(maxsize=4)
def _coupling_cached(nz: int, nj: int) -> tuple:
    """(mesh, expert, CouplingOperators) for one mesh size, built once."""
    TS = load_solvers()
    nodes = np.stack(
        np.meshgrid(np.linspace(0.0, L_Z, nz + 1),
                    np.linspace(0.0, T_SHELL, nj + 1), indexing="ij"), -1
    )
    mesh = TS.ShellMesh(nodes)
    mat = TS.SolidMaterial()
    ts = TS.ThermoStruct2D(mesh, mat)
    n = mesh.n_nodes
    conn = ts.conn
    ne = conn.shape[0]
    dofs = np.stack([2 * conn, 2 * conn + 1], axis=-1).reshape(ne, 8)
    a1 = np.array([1.0, 1.0, 0.0])
    beta = mat.E * mat.alpha / (1.0 - mat.nu)

    # -- G : the expert's own thermal-load quadrature ------------------------
    rows, cols, vals = [], [], []
    for (xi, eta), w in zip(TS._QP, TS._QW):
        dN, det, N = TS._shape_derivs(ts.xy, xi, eta)
        B = ts._B(dN)
        BDa = np.einsum("eia,ij,j->ea", B, ts.D, a1)
        for k in range(4):
            rows.append(dofs.ravel())
            cols.append(np.repeat(conn[:, k], 8))
            vals.append((w * mat.alpha * N[k] * BDa * det[:, None]).ravel())
    G = _scatter_G(rows, cols, vals, n)

    # -- G_body : -int N_k beta grad(dT) dV ----------------------------------
    rows, cols, vals = [], [], []
    for (xi, eta), w in zip(TS._QP, TS._QW):
        dN, det, N = TS._shape_derivs(ts.xy, xi, eta)
        for k in range(4):
            for l in range(4):
                c = -w * beta * N[k] * dN[:, l, :] * det[:, None]
                rows.append(2 * conn[:, k]); cols.append(conn[:, l]); vals.append(c[:, 0])
                rows.append(2 * conn[:, k] + 1); cols.append(conn[:, l]); vals.append(c[:, 1])
    G_body = _scatter_G(rows, cols, vals, n)

    # -- G_surf : +oint beta dT N_k n ds, 2-point Gauss per external edge -----
    ni, njj = mesh.shape
    nod = mesh.nodes.reshape(-1, 2)
    i, j = np.arange(ni), np.arange(njj)
    edges = [
        (mesh.nid(i, 0), mesh.nid(i + 1, 0)),
        (mesh.nid(i, njj), mesh.nid(i + 1, njj)),
        (mesh.nid(0, j), mesh.nid(0, j + 1)),
        (mesh.nid(ni, j), mesh.nid(ni, j + 1)),
    ]
    centroid = nod.mean(0)
    g = 1.0 / np.sqrt(3.0)
    rows, cols, vals = [], [], []
    for a, b in edges:
        pa, pb = nod[a], nod[b]
        tang = pb - pa
        Lg = np.linalg.norm(tang, axis=1)
        nrm = np.stack([tang[:, 1], -tang[:, 0]], axis=1) / np.maximum(Lg, 1e-30)[:, None]
        flip = np.sign(np.einsum("ij,ij->i", nrm, 0.5 * (pa + pb) - centroid))
        flip[flip == 0.0] = 1.0
        nrm = nrm * flip[:, None]
        for s in (-g, g):
            Nl = np.array([0.5 * (1.0 - s), 0.5 * (1.0 + s)])
            for k, nk in enumerate((a, b)):
                for l, nl in enumerate((a, b)):
                    c = beta * Nl[k] * Nl[l] * (0.5 * Lg)[:, None] * nrm
                    rows.append(2 * nk); cols.append(nl); vals.append(c[:, 0])
                    rows.append(2 * nk + 1); cols.append(nl); vals.append(c[:, 1])
    G_surf = _scatter_G(rows, cols, vals, n)

    H = (2.0 * mat.E * mat.alpha ** 2 / (1.0 - mat.nu) / (mat.rho * mat.cp)) * ts.M_th
    delta = T_INIT * beta ** 2 / (mat.rho * mat.cp * mat.E / (1.0 - mat.nu ** 2))
    return mesh, ts, CouplingOperators(G=G, G_surf=G_surf, G_body=G_body, H=H,
                                       beta=beta, delta=float(delta))


def coupling(nz: int = NZ, nj: int = NJ):
    """(mesh, ThermoStruct2D, CouplingOperators) -- the shared physical objects."""
    return _coupling_cached(nz, nj)


def streak(mesh, ts) -> np.ndarray:
    """The gas temperature along the inner face: ambient plus a Gaussian streak."""
    a, b, _L = ts._face("inner")
    p = mesh.nodes.reshape(-1, 2)
    z = 0.5 * (p[a, 0] + p[b, 0])
    return T_OUT + T_STREAK * np.exp(-((z - 0.5 * L_Z) / W_STREAK) ** 2)


# ---------------------------------------------------------------------------
# the two agents -- one solver, cut along the physics rather than the domain
# ---------------------------------------------------------------------------


@dataclass
class ConductionAgent:
    """`ThermoStruct2D`'s conduction half.  Owns T and nothing else.

    Its own surface port is real and it is the `THERM` one `thermal_seam`
    already exercises: the Robin channel ``-k dT/dn = h (T - T_gas)`` on the
    inner face, against a gas this case study does not model.  That port is
    OPEN -- §5.1's mechanism, and the drive enters through it.

    Under ``two_way`` the agent also accepts the Biot reverse term.  It arrives
    as ``G^T u_dot``, the transpose of the forward operator, which is why the
    pair is a bond: one matrix, read both ways.
    """

    agent_id: str = "cond"
    dt: float = DT_MACRO
    expose_elliptic: bool = False
    two_way: bool = False
    _mesh: Any = field(default=None, repr=False)
    _ts: Any = field(default=None, repr=False)
    _ops: Any = field(default=None, repr=False)
    _T: Any = field(default=None, repr=False)
    _Tg: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._mesh, self._ts, self._ops = coupling()
        self._T = np.full(self._mesh.n_nodes, T_INIT)
        self._Tg = streak(self._mesh, self._ts)

    # -- the agent's own march ---------------------------------------------
    def _robin_blocks(self, T: np.ndarray):
        Kb_i, fb_i = self._ts._robin("inner", H_IN, self._Tg)
        a, b, _L = self._ts._face("outer")
        Tw = 0.5 * (T[a] + T[b])
        h_rad = (self._ts.mat.emissivity * 5.670374419e-8
                 * (Tw ** 2 + T_OUT ** 2) * (Tw + T_OUT))
        Kb_o, fb_o = self._ts._robin("outer", np.broadcast_to(H_OUT, a.shape) + h_rad, T_OUT)
        return Kb_i + Kb_o, fb_i + fb_o

    def step(self, T: np.ndarray, dt: float | None = None,
             udot: np.ndarray | None = None) -> np.ndarray:
        """One conduction step, with the Biot source if the agent declares it.

        Under ``expose_elliptic`` the implicit solve is the composition layer's
        and the agent takes lumped-mass forward-Euler sub-steps at a cadence the
        diffusion number makes stable -- `thermal_seam._explicit_shell_class`
        exactly, and the declaration in `conduction_capabilities` follows this
        branch rather than the other way round. A record that says `exposed` and
        marches implicitly is a contradiction inside the declaration.
        """
        dt = self.dt if dt is None else dt
        biot = (None if (udot is None or not self.two_way)
                else T_INIT * self._ops.G.T.dot(udot))
        if self.expose_elliptic:
            Kb, fb = self._robin_blocks(T)
            K, f = self._ts.K_th + Kb, fb - (0.0 if biot is None else biot)
            m = np.maximum(np.asarray(self._ts.M_th.sum(axis=1)).ravel(), 1e-30)
            n = substeps_at(dt)
            h = dt / n
            for _ in range(n):
                T = T + h * (f - K.dot(T)) / m
            return T
        if biot is None:
            return self._ts.step_thermal(T, dt, H_IN, self._Tg, H_OUT, T_OUT,
                                         radiate=True, T_inf=T_OUT)
        from scipy.sparse.linalg import splu
        Kb, fb = self._robin_blocks(T)
        A = (self._ts.M_th / dt + self._ts.K_th + Kb).tocsc()
        rhs = self._ts.M_th.dot(T) / dt + fb - biot
        return splu(A).solve(rhs)

    def robin_power(self, T: np.ndarray) -> float:
        """The `THERM` port's power, integrated over the two faces, W per metre."""
        Kb, fb = self._robin_blocks(T)
        return float(np.ones(T.size) @ (fb - Kb.dot(T)))

    # -- the port callables ------------------------------------------------
    def face_T(self, T: np.ndarray | None = None) -> np.ndarray:
        a, b, _L = self._ts._face("inner")
        T = self._T if T is None else T
        return 0.5 * (T[a] + T[b])

    def outer_nodes(self) -> np.ndarray:
        """The outer face's node ids -- the `surface-mech` seam's own trace space."""
        a, b, _L = self._ts._face("outer")
        return np.unique(np.concatenate([a, b]))

    def respond(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        """The agent's boundary response, dispatched on the port it is asked for.

        ``inner:THERM``  (gas temperature on the inner face) -> (entropy flux).
        `thermal_seam.ShellAgent.respond` exactly, on the same solver: the trace
        enters the agent's own Robin channel as ``T_gas`` and the response is
        ``h (T_gas - T_wall)`` divided by the mean face temperature, which is
        `PORT_SPECS[THERM]`'s declared flow ``q_n / T`` and not the pseudo-bond.

        ``outer:MECH``  (boundary velocity) -> (the equivalent thermal traction
        ``beta dT`` on the outward normal, nodal).  **The response does not
        depend on the trace**, and that is a finding rather than a shortcut: a
        temperature field does not know its boundary is moving, so ``Lambda`` is
        identically zero on this side and the interface problem the
        ``surface-mech`` route poses is EMPTY rather than hard (spec §6.4(a),
        `CASE-STUDY-GUIDE` mistake 6).  The probe measures exactly that.
        """
        if port_name.endswith("MECH"):
            return self._ops.beta * (self._T[self.outer_nodes()] - T_REF)
        n_face = self.face_T().size
        T_gas = np.asarray(trace, dtype=float).reshape(n_face)
        T = self._ts.step_thermal(self._T, self.dt, H_IN, T_gas, H_OUT, T_OUT,
                                  radiate=True, T_inf=T_OUT)
        T_face = self.face_T(T)
        q = H_IN * (T_gas - T_face)
        return q / np.maximum(0.5 * (T_gas + T_face), 1.0)

    def base_trace(self, port_name: str = "inner:THERM") -> np.ndarray:
        """W74: the `THERM` port's effort is an ABSOLUTE temperature, so 0 K is
        not a state the agent is ever in. The `MECH` port's trace is a velocity,
        whose origin genuinely is zero -- the body at rest."""
        if port_name.endswith("MECH"):
            return np.zeros(self.outer_nodes().size)
        return np.asarray(self._Tg, dtype=float)

    def storage(self, state=None) -> float:
        """1/2 T^T M T, the conduction operator's own Lyapunov functional."""
        T = self._T if state is None else np.asarray(state)
        return float(0.5 * T @ self._ts.M_th.dot(T))

    def energy(self, T: np.ndarray) -> float:
        """int rho cp (T - T_ref) dV, J per metre -- the PHYSICAL energy R needs."""
        return float(np.ones(T.size) @ self._ts.M_th.dot(T - T_REF))

    def validity(self, state=None, cond=None) -> bool:
        return True


@dataclass
class ElasticityAgent:
    """`ThermoStruct2D`'s quasi-static plane-stress half.  Owns u and sigma.

    It has no time derivative at all, which is the structural fact this case
    study keeps running into: there is nothing to sub-step, so the
    ``split-step`` move `window_ns` and `thermal_seam` both make -- expose the
    elliptic part and take stable explicit sub-steps beside it -- **has no
    analogue here**.  A quasi-static agent is `EMBEDDED` or it is not an agent.
    See `R10_SCOPE`.

    Its own surface ports are the two pressure-loaded faces, and they are the
    `MECH` bond the vocabulary already has.  The ``surface-mech`` route uses the
    outer one; in the other two routes both are open.
    """

    agent_id: str = "elas"
    dt: float = DT_MACRO
    _mesh: Any = field(default=None, repr=False)
    _ts: Any = field(default=None, repr=False)
    _ops: Any = field(default=None, repr=False)
    _V: Any = field(default=None, repr=False)
    _u: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._mesh, self._ts, self._ops = coupling()
        self._V = self._ts._rigid_modes()
        self._u = np.zeros(2 * self._mesh.n_nodes)

    # -- the agent's own solve ---------------------------------------------
    def solve(self, load: np.ndarray) -> np.ndarray:
        TS = load_solvers()
        return TS._solve_free(self._ts.K_me, load, self._V, self._ts._factors)

    def stress(self, u: np.ndarray, dT: np.ndarray | None) -> np.ndarray:
        """sigma = D (eps(u) - eps_0).  ``dT=None`` is the surface route: an
        agent handed a traction has no eigenstrain to subtract, and that
        omission is a property of the bond it was handed, not a modelling
        choice."""
        TS = load_solvers()
        ts = self._ts
        dN, _det, N = TS._shape_derivs(ts.xy, 0.0, 0.0)
        B = ts._B(dN)
        d = np.stack([2 * ts.conn, 2 * ts.conn + 1], axis=-1).reshape(-1, 8)
        eps = np.einsum("eik,ek->ei", B, np.asarray(u).ravel()[d])
        if dT is not None:
            dTe = (dT[ts.conn] * N[None, :]).sum(1)
            e0 = np.zeros_like(eps)
            e0[:, 0] = e0[:, 1] = ts.mat.alpha * dTe
            eps = eps - e0
        return np.einsum("ij,ej->ei", ts.D, eps)

    def from_temperature(self, dT: np.ndarray):
        """The volumetric route: the whole eigenstrain field crosses."""
        u = self.solve(self._ops.G.dot(dT))
        return u, self.stress(u, dT)

    def from_surface_traction(self, dT: np.ndarray):
        """The `MECH` route: only ``G_surf`` crosses, and no eigenstrain with it."""
        u = self.solve(self._ops.G_surf.dot(dT))
        return u, self.stress(u, None)

    # -- energies, in the units the power residual needs --------------------
    def energy(self, u: np.ndarray, dT: np.ndarray) -> float:
        """1/2 int sigma : D^-1 : sigma dV, J per metre.

        Expanded, ``1/2 u^T K u - u^T G dT + 1/2 dT^T H dT``.  **The middle term
        is bilinear in the two agents' states and belongs to neither**, which is
        `PortAmendment` field 1's obstruction stated as arithmetic.
        """
        return float(0.5 * u @ self._ts.K_me.dot(u)
                     - u @ self._ops.G.dot(dT)
                     + 0.5 * dT @ self._ops.H.dot(dT))

    def cross_energy(self, u: np.ndarray, dT: np.ndarray) -> float:
        """-u^T G dT: the term no additive split of the free energy can place."""
        return float(-u @ self._ops.G.dot(dT))

    # -- the port callable --------------------------------------------------
    def respond(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        """(boundary velocity on the outer face) -> (traction), the `MECH` bond.

        Quasi-static, so a velocity trace is a displacement increment
        ``v dt``: prescribe it on the face's normal component, solve the
        constrained system, and return the reaction traction per unit length.
        `ResponseHalf.EFFORT`, which is the vault's `MECH` convention.
        """
        a, b, L = self._ts._face("outer")
        nodes = np.unique(np.concatenate([a, b]))
        v = np.asarray(trace, dtype=float).ravel()
        if v.size != nodes.size:
            v = np.interp(np.linspace(0.0, 1.0, nodes.size),
                          np.linspace(0.0, 1.0, v.size), v)
        n_all = self._mesh.n_nodes
        fixed = np.concatenate([2 * nodes, 2 * nodes + 1])
        u = np.zeros(2 * n_all)
        u[2 * nodes + 1] = v * self.dt          # normal (through-thickness) motion
        free = np.setdiff1d(np.arange(2 * n_all), fixed)
        from scipy.sparse.linalg import spsolve
        rhs = -self._ts.K_me.dot(u)[free]
        if free.size:
            u[free] = spsolve(self._ts.K_me[free][:, free].tocsc(), rhs)
        react = self._ts.K_me.dot(u)
        w = np.zeros(n_all)
        np.add.at(w, a, 0.5 * L)
        np.add.at(w, b, 0.5 * L)
        return react[2 * nodes + 1] / np.maximum(w[nodes], 1e-30)

    def base_trace(self) -> np.ndarray:
        """A velocity has an absolute origin and it is zero: the body at rest."""
        a, b, _L = self._ts._face("outer")
        return np.zeros(np.unique(np.concatenate([a, b])).size)

    def storage(self, state=None) -> float:
        """1/2 u^T K u, SPD on the complement of the rigid modes."""
        u = self._u if state is None else np.asarray(state)
        return float(0.5 * u @ self._ts.K_me.dot(u))

    def validity(self, state=None, cond=None) -> bool:
        return True


# ---------------------------------------------------------------------------
# the referent and the split -- one function each, so the comparison is visible
# ---------------------------------------------------------------------------


def monolith(n_steps: int = N_STEPS, dt: float = DT_MACRO, two_way: bool = False,
             tol: float = 1e-12, itmax: int = 60):
    """`ThermoStruct2D` unsplit: the referent the gate is graded against.

    ``two_way`` converges the Biot fixed point at every step, so the referent for
    the two-way mode is the tightly coupled pair of the same two solvers -- which
    is what `multiphysics.tight_couple` means by a referent at a multiphysics
    seam, computed here directly because the coupling is co-located and there is
    no seam to iterate on.
    """
    cond, elas = ConductionAgent(dt=dt, two_way=two_way), ElasticityAgent(dt=dt)
    _mesh, _ts, ops = coupling()
    T = np.full(cond._mesh.n_nodes, T_INIT)
    u = elas.solve(ops.G.dot(T - T_REF))
    hist, iters = [], []
    for _k in range(n_steps):
        if not two_way:
            T = cond.step(T, dt)
            u = elas.solve(ops.G.dot(T - T_REF))
            iters.append(1)
        else:
            T0, u0 = T, u
            Tn, un = T0, u0
            for it in range(itmax):
                Tn1 = cond.step(T0, dt, udot=(un - u0) / dt)
                un1 = elas.solve(ops.G.dot(Tn1 - T_REF))
                moved = np.linalg.norm(Tn1 - Tn) / max(np.linalg.norm(Tn1), 1e-30)
                Tn, un = Tn1, un1
                if moved < tol:
                    break
            T, u = Tn, un
            iters.append(it + 1)
        hist.append((T.copy(), u.copy()))
    sigma = elas.stress(u, T - T_REF)
    return dict(T=T, u=u, sigma=sigma, history=hist, iterations=iters,
                agents=(cond, elas), ops=ops)


def split(route: str = "volumetric", n_steps: int = N_STEPS, dt: float = DT_MACRO,
          lag: int = 0, two_way: bool = False):
    """The two-agent march, with the volume term crossing by one of three routes.

    ``lag = 0``  the elasticity agent sees the temperature the conduction agent
                 has just produced.  This is the SYNCHRONOUS split, and it is
                 the same arithmetic in the same order as the monolith, so it
                 reproduces it bit for bit.  That is a control on the plumbing
                 and not a measurement of coupling error -- **W106**: a floor of
                 exactly zero does not bound anything.
    ``lag = 1``  the elasticity agent sees the PREVIOUS macro-step's temperature,
                 which is what a two-agent exchange at a shared clock actually
                 does.  This is the splitting error, and it is the number a bond
                 would have to bound.
    """
    if route not in ROUTES:
        raise ValueError(f"route must be one of {sorted(ROUTES)}, got {route!r}")
    cond, elas = ConductionAgent(dt=dt, two_way=two_way), ElasticityAgent(dt=dt)
    _mesh, _ts, ops = coupling()
    T = np.full(cond._mesh.n_nodes, T_INIT)
    u = elas.solve(ops.G.dot(T - T_REF))
    seen: list[np.ndarray] = [T.copy()]
    u_before = u.copy()
    hist = []
    for _k in range(n_steps):
        # The staggered two-way pass: the conduction agent gets the displacement
        # rate the elasticity agent produced over the PREVIOUS macro-step, which
        # is the only rate a two-agent exchange has when the step begins. Using
        # the current one would be the converged fixed point, which is `monolith`.
        udot = (u - u_before) / dt if two_way else None
        T = cond.step(T, dt, udot=udot)
        seen.append(T.copy())
        dT = seen[max(0, len(seen) - 1 - lag)] - T_REF
        u_before = u
        if route == "surface-mech":
            u, sigma = elas.from_surface_traction(dT)
        else:
            u, sigma = elas.from_temperature(dT)
        hist.append((T.copy(), u.copy()))
    return dict(T=T, u=u, sigma=sigma, history=hist, agents=(cond, elas), ops=ops)


def power_residual(cond: ConductionAgent, elas: ElasticityAgent, ops: CouplingOperators,
                   Ta: np.ndarray, ua: np.ndarray, Tb: np.ndarray, ub: np.ndarray,
                   dt: float) -> dict[str, float]:
    """port-algebra §6's R(t) on the split graph, with the volume term named.

    Reports BOTH conjugate readings of the volumetric power, because they are not
    the same number and the difference is the finding:

        P_A = int sigma : eps_0_dot dV      = u^T G T_dot - dT^T H T_dot
        P_B = int beta dT tr(eps_dot) dV    = u_dot^T G dT

    and ``P_A + P_B = d/dt (u^T G dT - 1/2 dT^T H dT)`` exactly -- a TOTAL
    DERIVATIVE of an energy neither agent owns.  A surface bond has no such
    ambiguity, because two spatially disjoint agents have additive energies and
    the interface stores nothing.
    """
    dTa, dTb = Ta - T_REF, Tb - T_REF
    Tdot, udot = (Tb - Ta) / dt, (ub - ua) / dt
    P_A = float(ub @ ops.G.dot(Tdot) - dTb @ ops.H.dot(Tdot))
    P_B = float(udot @ ops.G.dot(dTb))
    shared = lambda u_, d_: float(u_ @ ops.G.dot(d_) - 0.5 * d_ @ ops.H.dot(d_))
    dE_th = (cond.energy(Tb) - cond.energy(Ta)) / dt
    dE_el = (elas.energy(ub, dTb) - elas.energy(ua, dTa)) / dt
    P_gamma = cond.robin_power(Tb)
    return {
        "E_th": cond.energy(Tb),
        "E_el": elas.energy(ub, dTb),
        "E_cross": elas.cross_energy(ub, dTb),
        "E_shared": shared(ub, dTb),
        "dE_th_dt": dE_th,
        "dE_el_dt": dE_el,
        "P_gamma": P_gamma,
        "P_omega_A": P_A,
        "P_omega_B": P_B,
        "conjugate_gap": P_A + P_B,
        "d_E_shared_dt": (shared(ub, dTb) - shared(ua, dTa)) / dt,
        # the elasticity agent's own balance: dE_el/dt = P_mech - P_omega, and
        # P_mech is zero on a traction-free body, so this is the volumetric
        # bond's power closing a subsystem balance on its own.
        "R_elas": dE_el + P_A,
        "R_elas_rel": abs(dE_el + P_A) / max(abs(P_A), 1e-300),
        # the whole graph, without and with the volume term declared.
        "R_global": dE_th + dE_el - P_gamma,
        "R_global_with_volume": dE_th + dE_el - P_gamma + P_A,
        "R_global_rel": abs(dE_th + dE_el - P_gamma) / max(abs(P_gamma), 1e-300),
        "R_global_with_volume_rel": (abs(dE_th + dE_el - P_gamma + P_A)
                                     / max(abs(P_gamma), 1e-300)),
        "P_omega_over_P_gamma": P_A / P_gamma if P_gamma else float("nan"),
        "E_el_over_E_th": elas.energy(ub, dTb) / cond.energy(Tb),
    }


# ---------------------------------------------------------------------------
# the PortAmendment submission -- W32's first exercise
# ---------------------------------------------------------------------------

ROUTES = ("volumetric", "surface-mech", "global-field")

#: The candidate sixth port type's name, used only to ask `ports.spec_for` for it
#: and be refused.  It is never added to `PortType`; adding it is the operation
#: the amendment exists to gate.
CANDIDATE_PORT = "THERMEL"


#: **The six fields, filled in for thermal strain.**  `holes.PORT_AMENDMENT`'s
#: `declared_interface` is the schema; this is the submission.  Each entry is
#: (what was submitted, what decides it, and the measurement that decides it),
#: and `evaluate_amendment` turns the measurements into a verdict per field.
#:
#: The amendment's own `inference_note` says the six-field checklist is *"a
#: proposed procedure with no evidence it is sufficient ... the first exercise
#: may well show a seventh field is needed"*.  It does -- see `FIELD_SEVEN`.
AMENDMENT_SUBMISSION: dict[str, dict[str, str]] = {
    "bond": {
        "submitted": (
            "effort  dT = T - T_ref  [K];  flow  beta tr(eps_dot)  [Pa/(K s)]; "
            "product beta dT tr(eps_dot)  [W/m^3].  beta = E alpha / (1 - nu) is "
            "the plane-stress thermal stress modulus"
        ),
        "decided_by": (
            "the declared interface says the product must be a power 'in W or "
            "W/m^2'.  A volumetric bond is W/m^3, which changes the MEASURE in "
            "§6's residual from ds to dV -- survivable on its own.  What is not "
            "survivable is that the pairing is not unique: two readings of the "
            "same coupling differ by the rate of an energy neither agent owns"
        ),
        "measurement": "conjugate_ratio, cross_over_elastic",
    },
    "mapping": {
        "submitted": (
            "effort CONSISTENT, flow CONSERVATIVE, both derived from the single "
            "declared prolongation exactly as on a surface (interface-transfer-theory "
            "§2.2).  The derivation nowhere uses the dimension of the carrier"
        ),
        "decided_by": "nothing about the adjoint pair depends on co-dimension",
        "measurement": "none required",
    },
    "transfer": {
        "submitted": (
            "P_i = I, R_i = I under the mass-matrix pairing, adjoint by "
            "construction, because both agents are discretized on the SAME mesh"
        ),
        "decided_by": (
            "it passes, and the vacuity is the finding.  There is no reduction "
            "because there is no cut: dim M = dim V = the whole state"
        ),
        "measurement": "dim_M_over_dim_V",
    },
    "dtn_reading": {
        "submitted": (
            "the conduction agent imposes dT and the elasticity agent returns "
            "beta tr(eps); under the two-way mode the reverse operator is G^T, "
            "the same matrix transposed, and the interface problem is non-empty"
        ),
        "decided_by": (
            "the reading is DEFINED and it is not AFFORDABLE.  The trace space "
            "is the whole state, so a finite-difference probe costs n+1 full "
            "solves to build an operator that IS the coupled solve -- and that is "
            "decidable from the declaration (co-dimension zero), not measured"
        ),
        "measurement": "probe_solves, dim_M_over_dim_V, two_way_iterations",
    },
    "distinctness": {
        "submitted": (
            "not a special case of MECH.  The exact identity G = G_surf + G_body "
            "splits the coupling into the part a surface traction on the body's "
            "own boundary can deliver and the part it cannot; G_body is the "
            "body force beta grad(dT) and vanishes IDENTICALLY when dT is uniform"
        ),
        "decided_by": (
            "run the surface route and grade it against the monolith, with the "
            "uniform-dT case as the positive control the reduction must pass"
        ),
        "measurement": "surface_route_stress_error, body_load_fraction, "
                       "uniform_control",
    },
    "exercise": {
        "submitted": (
            "this case study, and W94's actuator disk as a second instance in a "
            "different governing family"
        ),
        "decided_by": "one interface the existing five cannot type, which is this",
        "measurement": "none required",
    },
}


#: **The seventh field, which the first exercise found the procedure needs.**
FIELD_SEVEN = {
    "name": "support",
    "declared_interface": (
        "co-dimension of the bond's carrier, and -- if it is zero -- the argument "
        "that the two agents' free energies are ADDITIVE"
    ),
    "why": (
        "fields 1-6 are ALL satisfiable by a co-located pair, and not one of them "
        "notices that §6's residual has lost its premise.  R(t) sums dE_i/dt over "
        "agents, which presumes Psi = sum_i Psi_i(u_i).  Linear thermoelasticity's "
        "free energy carries a bilinear cross term -beta dT tr(eps) that belongs to "
        "neither half, measured here at 2680x the elastic energy, so no such "
        "decomposition exists.  A procedure that cannot refuse that is not gating "
        "the thing that matters"
    ),
    "would_have_refused": "thermal strain, at field 7, before any measurement",
}


#: What the framework has to say about a co-located split, recorded as scope
#: rather than as a field.  `VOLUMETRIC_COUPLING_SCOPE`'s precedent (W70): a
#: fourth unverifiable declaration is worse than a stated limit.
VOLUMETRIC_BOND_SCOPE = (
    "thermal strain is a BOND and is not a PORT. A port is a bond on an interface "
    "of co-dimension >= 1 between agents whose free energies add; this bond has "
    "co-dimension 0 between agents whose free energies do not add, and the cross "
    "term is 2680x the elastic energy. The framework's instrument for it is "
    "therefore an OPERATOR SPLITTING with a splitting-error bound -- measured here "
    "at 7.1e-3 relative stress per macro-step of lag, first order in dt -- and not "
    "a transmission condition. No sixth port type is proposed"
)

#: R10 refused this graph and its own derivation does not apply to it.  Recorded
#: here because the refusal was real, was not about the bond, and would otherwise
#: have been read as one.
#:
#: **W114 closed 2026-09-04 at CS-12** (`cases/wing_fsi.py`, `R10_HANDLE`), which
#: met the same defect at a co-dimension-1 FSI seam and supplied the premise
#: check the rule was missing: an EMBEDDED agent is refused only when the graph
#: contains another agent of the same ``governing_family``.  This graph's two
#: agents are conduction and elasticity, so it is no longer refused.  The text
#: below stays as the record of what the refusal was, which is the thing a reader
#: of the published numbers needs.
R10_SCOPE = (
    "L2/R10 refuses any graph with two agents one of which declares "
    "elliptic_subsolve=embedded, and its sentence is 'the graph decomposes the "
    "domain ... so the decomposition changes the operator rather than restricting "
    "it'. A CO-LOCATED split cuts no domain: both agents own all of Omega, and the "
    "quasi-static elasticity solve is over exactly the region it was over in the "
    "monolith. The rule is stated over elliptic_subsolve alone and never consults "
    "whether the decomposition cuts the agent, so it fires where its derivation "
    "does not reach. And there is no split-step escape on this side: a quasi-static "
    "solve has no time derivative to sub-step, so an elasticity agent is EMBEDDED "
    "or it is not an agent"
)


def request_volumetric_port():
    """Ask the package for the sixth port type, and be refused.

    This is the operation the amendment gates, performed at the line where it
    would have to happen.  `ports.spec_for` raises `NamedHoleError` naming
    `holes.PORT_AMENDMENT`; nothing here catches it, so a caller declaring a
    volumetric seam gets the refusal rather than a silently-typed connection.
    """
    return spec_for(CANDIDATE_PORT)


def evaluate_amendment(evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    """The six-field verdict, from measurements rather than from argument.

    ``evidence`` is the driver's measurement dict; with none supplied the fields
    that need a number report ``unmeasured`` rather than a guess.  Returns a
    per-field verdict plus the overall one, which is ``refuse`` if any field
    fails -- the amendment is a conjunction, per `holes.PORT_AMENDMENT`.
    """
    ev = dict(evidence or {})

    def num(key, default=None):
        v = ev.get(key, default)
        return None if v is None else float(v)

    fields: dict[str, dict[str, Any]] = {}

    ratio = num("conjugate_ratio")
    cross = num("cross_over_elastic")
    fields["bond"] = {
        "verdict": "unmeasured" if ratio is None or cross is None else (
            "fails" if (abs(ratio) > 2.0 or abs(cross) > 1.0) else "holds"),
        "why": (
            "the conjugate pair exists and its product is a power density, but the "
            "pairing is defined only up to the rate of a shared energy: the two "
            f"readings differ by {ratio if ratio is None else round(ratio, 1)}x, and "
            "the free energy's bilinear cross term is "
            f"{cross if cross is None else round(cross, 1)}x the elastic energy. §6's "
            "residual adds dE_i/dt over agents and there is no additive split to add"
        ),
    }

    fields["mapping"] = {
        "verdict": "holds",
        "why": ("the consistent/conservative pair is the adjoint of the declared "
                "prolongation and the derivation never mentions the carrier's "
                "dimension"),
    }

    frac = num("dim_M_over_dim_V")
    fields["transfer"] = {
        "verdict": "holds-vacuously" if frac is None or frac >= 1.0 else "holds",
        "why": ("P = R = I on a shared mesh, adjoint by construction -- and vacuous: "
                f"dim M / dim V = {frac if frac is None else round(frac, 3)}, so there "
                "is no reduction because there is no cut"),
    }

    solves = num("probe_solves")
    fields["dtn_reading"] = {
        "verdict": "unmeasured" if frac is None else (
            "fails" if frac >= 1.0 else "holds"),
        "why": ("which side is imposed and which returned is well defined, and under "
                "the two-way mode the interface problem is non-empty. It is not "
                f"affordable: dim M = dim V, so a probe is {solves if solves is None else int(solves)} "
                "full solves for an operator that IS the coupled solve. Decidable from "
                "the declaration -- co-dimension zero -- and not from any measurement"),
    }

    err = num("surface_route_stress_error")
    ctrl = num("uniform_control")
    fields["distinctness"] = {
        "verdict": "unmeasured" if err is None or ctrl is None else (
            "holds" if (err > 1.0 and ctrl < 1e-12) else "fails"),
        "why": ("G = G_surf + G_body is exact to 1e-15; dropping G_body -- 9.6% of the "
                f"load -- costs {err if err is None else round(err, 1)}x in stress, and "
                "the positive control passes: at a uniform dT, G_body is exactly zero "
                f"({ctrl}) and the surface reduction is exact. Not a special case of MECH"),
    }

    fields["exercise"] = {
        "verdict": "holds",
        "why": "this case study; W94's actuator disk is a second instance",
    }

    failed = [k for k, v in fields.items() if v["verdict"] == "fails"]
    return {
        "candidate": CANDIDATE_PORT,
        "fields": fields,
        "failed": failed,
        "verdict": "refuse" if failed else (
            "unmeasured" if any(v["verdict"] == "unmeasured" for v in fields.values())
            else "admit"),
        "seventh_field": FIELD_SEVEN,
        "scope": VOLUMETRIC_BOND_SCOPE,
    }


# ---------------------------------------------------------------------------
# the records
# ---------------------------------------------------------------------------

#: (T, q_n/T) in K and W/(m^2 K), s_e s_f = s_P with s_P in W/m^2 -- the same
#: declaration `thermal_seam` makes, on the same solver and the same face.
THERM_SCALES = {"temperature": T_STREAK + T_OUT,
                "entropy_flux": 1.0e5 / (T_STREAK + T_OUT),
                "power_area": 1.0e5}

#: (traction, velocity) in Pa and m/s.  The velocity scale is the shell's own
#: thermal expansion rate, alpha dT_scale L / t_diffusion, which is what a
#: quasi-static structural agent's boundary actually moves at.
MECH_SCALES = {"stress": 1.0e8, "velocity": 1.0e-5, "power_area": 1.0e3}


def _face_prolongation(agent_id: str, port_name: str, n_cells: int) -> Prolongation:
    """The same 16-mode real Fourier basis every other case study declares."""
    h = L_Z / n_cells
    y = (np.arange(n_cells) + 0.5) * h
    cols = [np.full(n_cells, 1.0 / np.sqrt(L_Z))]
    k = 1
    while len(cols) < M_EFF:
        w = 2.0 * np.pi * k * y / L_Z
        cols.append(np.sqrt(2.0 / L_Z) * np.cos(w))
        if len(cols) < M_EFF:
            cols.append(np.sqrt(2.0 / L_Z) * np.sin(w))
        k += 1
    return Prolongation(
        agent_id=agent_id, port_name=port_name,
        matrix=np.column_stack(cols[:M_EFF]), gram_V=h * np.eye(n_cells),
        label="16-mode real Fourier basis (probed-dtn-coupling 2.2)",
    )


def conduction_capabilities(expert: ConductionAgent, route: str) -> ExpertCapabilities:
    n_face = expert.face_T().size
    ports = [port_decl(
        name="inner:THERM", port_type=PortType.THERM,
        geometry=f"inner face of the shell, {n_face} line elements",
        direction=Direction.BIDIRECTIONAL,
        nondim=dict(THERM_SCALES),
        effective_resolution=M_EFF,
        motion_class=MotionClass.STATIC,
        response_half=ResponseHalf.FLOW,
        prolongation=_face_prolongation(expert.agent_id, "inner:THERM", n_face),
        note="the agent's own Robin channel, -k dT/dn = h (T - T_gas). OPEN: the "
             "gas is not modelled here and the drive enters through this port",
    )]
    if route == "surface-mech":
        n_mech = expert.outer_nodes().size
        ports.append(port_decl(
            name="outer:MECH", port_type=PortType.MECH,
            geometry=f"outer face of the shell, {n_mech} nodes",
            direction=Direction.BIDIRECTIONAL,
            nondim=dict(MECH_SCALES),
            effective_resolution=M_EFF,
            motion_class=MotionClass.STATIC,
            response_half=ResponseHalf.EFFORT,
            prolongation=_face_prolongation(expert.agent_id, "outer:MECH", n_mech),
            note="the equivalent thermal pressure beta dT n. Its response does NOT "
                 "depend on the imposed velocity -- the thermal field does not know "
                 "the boundary is moving -- so Lambda is identically zero here and "
                 "the interface problem is EMPTY, not hard (spec §6.4(a))",
        ))
    return ExpertCapabilities(
        expert_id=expert.agent_id,
        ports=ports,
        bc_channel=BCChannel.ROBIN,
        bc_time_varying=True,
        # Backward Euler over the whole shell. EMBEDDED in R10's exact sense --
        # and see R10_SCOPE for why the rule's derivation does not reach a
        # co-located split.
        elliptic_subsolve=(EllipticSubsolve.EXPOSED if expert.expose_elliptic
                           else EllipticSubsolve.EMBEDDED),
        time_discretization=(TimeDiscretization.EXPLICIT if expert.expose_elliptic
                             else TimeDiscretization.IMPLICIT),
        stencil_radius=1,
        substeps_per_macro_step=(substeps_at(expert.dt) if expert.expose_elliptic else 1),
        differentiable=Differentiable.NONE,
        dt_native=expert.dt,
        storage=expert.storage,
        equivariances=("translation-z",),
        validity=expert.validity,
        governing_family="heat-conduction-2d",
        lambda_ref="thermostruct2d.ThermoStruct2D.step_thermal, same mesh",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash="thermostruct2d/1-conduction-"
                    + ("explicit" if expert.expose_elliptic else "backward-euler"),
        boundary_response=expert.respond,
        probe_base=lambda _p, _e=expert: _e.base_trace(_p),
        reproducibility_floor=float(np.finfo(float).eps),
        deterministic=True,
        note="solvers/thermostruct2d.py, build repo, imported unmodified; the "
             "conduction half only",
    )


def elasticity_capabilities(expert: ElasticityAgent, route: str) -> ExpertCapabilities:
    a, b, _L = expert._ts._face("outer")
    n_face = np.unique(np.concatenate([a, b])).size
    return ExpertCapabilities(
        expert_id=expert.agent_id,
        ports=[port_decl(
            name="outer:MECH", port_type=PortType.MECH,
            geometry=f"outer face of the shell, {n_face} nodes",
            direction=Direction.BIDIRECTIONAL,
            nondim=dict(MECH_SCALES),
            effective_resolution=M_EFF,
            motion_class=MotionClass.STATIC,
            response_half=ResponseHalf.EFFORT,
            prolongation=_face_prolongation(expert.agent_id, "outer:MECH", n_face),
            note="quasi-static: the imposed velocity is a displacement increment "
                 "v dt and the response is the reaction traction",
        )],
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=True,
        # Quasi-static: a global linear solve with no time step to shrink at all,
        # and NO split-step variant exists -- there is nothing to sub-step.
        elliptic_subsolve=EllipticSubsolve.EMBEDDED,
        time_discretization=TimeDiscretization.IMPLICIT,
        stencil_radius=1,
        substeps_per_macro_step=1,
        differentiable=Differentiable.NONE,
        dt_native=expert.dt,
        storage=expert.storage,
        equivariances=("translation-z", "rigid-body"),
        validity=expert.validity,
        governing_family="plane-stress-elasticity-2d",
        lambda_ref="thermostruct2d.ThermoStruct2D.solve_mechanical, same mesh",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash="thermostruct2d/1-elasticity-quasistatic",
        boundary_response=expert.respond,
        probe_base=lambda _p, _e=expert: _e.base_trace(),
        reproducibility_floor=float(np.finfo(float).eps),
        deterministic=True,
        note="solvers/thermostruct2d.py, build repo, imported unmodified; the "
             "quasi-static plane-stress half only",
    )


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------


#: **Measured 2026-09-01** by `scripts/w94_thermal_strain.py`, and deliberately
#: **not wired into any graph** -- ``build`` defaults to ``measured=None`` and this
#: is a record rather than a declaration.  It is here to be read, cited and
#: reproduced, not to enter a bound.
#:
#: **Why it is not declared, which is the finding.**  ``tau = 0`` is honest and
#: free: both agents are the real solver, so each is its own reference and its
#: infidelity is zero by construction -- exactly `thermal_seam`'s case.  The
#: splitting error is not.  `MeasuredConstants` has a slot for ``sigma``, the
#: TRANSMISSION infidelity, and the master bound consumes it as one; the number
#: below is the defect of a **lagged operator splitting**, measured in relative
#: stress because there is no interface to take an interface power over, and
#: quoted with its lag because it is first order in the exchange interval.  Those
#: are different objects.  Putting one in the other's slot would be W56's class
#: with the sign reversed -- not a bound quoted from a constant nobody measured,
#: but a bound quoted from a constant somebody measured and that means something
#: else -- so ``sigma`` stays unmeasured on this graph and the compile says so.
#:
#: **The framework has no slot for a splitting error.**  That is W115's shape on
#: a different axis and it is recorded rather than worked around: a co-located
#: split's own measured defect is not declarable today.
SPLITTING_ERROR = 7.116244e-03          # relative Frobenius stress, lag = DT_MACRO

MEASURED_W94 = MeasuredConstants(
    tau=0.0,
    probe_state=(f"shell 0.20 m x 8 mm, {NZ} x {NJ} Q1, streak {T_STREAK} K over "
                 f"{W_STREAK} m, h_in {H_IN}, macro-step {N_STEPS} of {DT_MACRO} s"),
    scheme="split-volumetric",
    depth=0,
    source=("scripts/w94_thermal_strain.py stage referent; tau is 0 by construction "
            f"for a real solver against itself. The splitting error {SPLITTING_ERROR} "
            "(stage split, one-macro-step lag) is NOT declared as sigma: it is a "
            "splitting defect in relative stress, not a transmission infidelity in "
            "interface power, and the two are not the same constant"),
)


def build(route: str = "surface-mech", two_way: bool = False,
          expose_elliptic: bool = False, measured="default",
          experts: dict[str, Any] | None = None) -> tuple[CaseGraph, dict[str, Any]]:
    """The two-agent co-located graph, plus the experts backing it.

    ``route`` decides how the volume term is declared, and the three answers are
    the case study:

    * ``volumetric``    RAISES `NamedHoleError` from `ports.spec_for`.  The
                        package refuses the sixth port type at the line where it
                        would enter, which is the deliverable.
    * ``surface-mech``  a real `MECH` seam on the body's own outer face, carrying
                        the equivalent thermal pressure.  Compiles.
    * ``global-field``  the eigenstrain as a `GlobalField`, no connection at all.
                        Numerically exact and certified by nothing -- and since
                        **W117** it is REFUSED at `L3/global-field`, which reads
                        the `produced_by` declaration below.  It compiled when
                        this case study was written; that it no longer does is
                        the finding, not a regression.

    ``measured`` defaults to **None on every route**, and that is a decision
    rather than an omission.  `MEASURED_W94` exists and is not wired in: ``tau``
    is honestly zero and the splitting error is not a ``sigma``, and there is no
    field for what it *is* (**W116**).  Declaring the two routes ASYMMETRICALLY
    was the first version of this file's own mistake and it produced a headline
    that was a property of the declaration rather than of the routes -- so both
    routes now compile against the same (empty) constant set, and the comparison
    in the wiki page's §4.4 is the one that survives that.  Pass
    ``measured=MEASURED_W94`` to see the record enter a compile.
    """
    if route not in ROUTES:
        raise ValueError(f"route must be one of {sorted(ROUTES)}, got {route!r}")
    if route == "volumetric":
        # Not caught. A case study that wants a sixth port type gets the refusal,
        # not a silently-typed connection -- that is what a closed vocabulary is.
        request_volumetric_port()

    experts = experts or {
        "cond": ConductionAgent(expose_elliptic=expose_elliptic, two_way=two_way),
        "elas": ElasticityAgent(),
    }
    agents = [
        Agent("cond", conduction_capabilities(experts["cond"], route),
              domain="shell segment, 0.20 m x 8 mm -- ALL of it", role="thermal"),
        Agent("elas", elasticity_capabilities(experts["elas"], route),
              domain="shell segment, 0.20 m x 8 mm -- ALL of it", role="structure"),
    ]

    connections: list[Connection] = []
    global_fields: list[GlobalField] = []
    if route == "surface-mech":
        connections.append(Connection(
            seam_id="thermal-pressure",
            a=("cond", "outer:MECH"),
            b=("elas", "outer:MECH"),
            port_type=PortType.MECH,
            geometrically_coincident=True,
            derive_space=True,
            # n_0(Gamma) = 1, and it was DECLARED 0 first and refused.  Inspection
            # resolved it in a minute, which is the trade `CASE-STUDY-GUIDE` says to
            # make: the null direction is the CONSTANT mode, to 5.7e-13, and its
            # physical name is rigid-body translation normal to the face.  A uniform
            # normal velocity on the only constrained face of a FREE elastic body is
            # a rigid motion, so it produces no strain and no reaction, and the
            # probed operator cannot see it (s_min/s_max = 2.4e-15).
            #
            # That is a fourth row for the guide's n_0 table and it arrives for a
            # reason unrelated to the other three: not incompressibility, not
            # lumpedness, but the KINEMATICS of a free body.  A solid-solid MECH
            # seam gets 1 per unconstrained rigid direction the trace can excite.
            expected_null_dim=1,
            note="the classical equivalent-thermal-pressure reduction, t = beta dT n. "
                 "A real MECH bond on a real surface, and it carries G_surf only -- "
                 "the 9.6% of the load that is G_body has nowhere to go",
        ))
    if route == "global-field":
        global_fields.append(GlobalField(
            name="thermal-eigenstrain",
            applies_to=("elas",),
            # **W117.** The conduction agent produces it and the elasticity agent
            # consumes it: a proper subset applied outside itself, which is the
            # rule's refusal case and is exactly what this route exists to be.
            produced_by=("cond",),
            note="eps_0 = alpha (T - T_ref) I, entering the elasticity agent's update "
                 "directly. It is NEITHER uniform NOR external -- it is the other "
                 "agent's state -- and declaring it here bypasses L3 entirely: no "
                 "scale set, no prolongation, no adjoint, no null space, no response "
                 "half, no tau, no sigma, no beta. Numerically exact and certified by "
                 "nothing. This was the silent-wrongness class, declarable by anyone -- "
                 "and W117 gave it a reader: produced_by below makes the provenance "
                 "declarable, and L3/global-field refuses this route because the field "
                 "is one agent's state applied to another",
        ))

    # None, and `MEASURED_W94` is a record rather than a default -- see its own
    # comment. tau = 0 is true and declarable; the splitting error is not sigma,
    # and the compile is more useful reporting sigma UNMEASURED than reporting a
    # number that means something else.
    if measured == "default":
        measured = None

    return CaseGraph(
        name=f"thermal-strain-{route}" + ("-two-way" if two_way else ""),
        agents=agents,
        connections=connections,
        # NON-OVERLAPPING is wrong and OVERLAPPING is wrong. The two agents share
        # the WHOLE region, which is neither a shared surface nor a partial
        # overlap, and `Decomposition` has no member for it. Declared overlapping
        # with overlap = the whole domain, which is the closer of the two and is
        # said out loud rather than hidden: the partition of unity a co-located
        # split would need is not over subdomains at all.
        decomposition=Decomposition.OVERLAPPING,
        overlap=T_SHELL,
        overlap_cells=NJ,
        global_fields=global_fields,
        macro_dt=DT_MACRO,
        measured=measured,
        note="the ninth real case study, and the first CO-LOCATED split: two agents "
             "on the same mesh over the same region, cut along the physics rather "
             "than the domain. thermostruct2d.ThermoStruct2D, build repo, unmodified",
    ), experts
