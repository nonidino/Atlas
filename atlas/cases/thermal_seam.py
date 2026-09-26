"""A real MULTIPHYSICS seam: compressible gas against a thermoelastic shell.

The fifth real case study, and the first whose two sides do not solve the same
equations.  `CASE-STUDY-GUIDE` sets the bar for a fifth one at *"it can answer a
question none of the other four can"*; this one answers two:

**1. E3 has never been exercised against a genuine disagreement.** All four
existing real case studies are 2-D incompressible Navier-Stokes, so
`governing_family` -- the string comparison the spec calls load-bearing -- has
only ever been compared against itself.  Every graph that fails it in the test
suite is a fixture built to fail it.  Here the two sides are

    gas    compressible-navier-stokes-2d      (MUSCL/HLLC finite volume, SSP-RK2)
    shell  thermoelastic-shell-2d             (Q1 conduction + plane stress)

and the disagreement is a fact about the solvers, not a declaration chosen to
make a point.

**2. THERM's declared flow variable is not the one a thermal solver returns.**
`ports.PORT_SPECS[THERM]` declares the bond ``(T, q_n/T)`` -- entropy flux -- and
says so explicitly: *"the TRUE bond, not the pseudo-bond (T, q_n) most
co-simulation codes exchange."*  Every thermal solver in the build repo computes
``q_n``.  `ThermoStruct2D._robin` implements ``-k dT/dn = h (T - T_gas)`` and
`generate.py::_wall_flux` returns ``h`` and ``T``; nothing anywhere divides by
``T``.  So the physically natural callable returns the pseudo-bond, W66's failure
arrives on its own rather than by construction, and `flux_convention` selects
between them so the difference is measurable rather than argued.

**Where the objects come from.** Both experts are build-repo solvers, imported
verbatim and unmodified: `atlas/solvers/compressible2d.py` and
`atlas/solvers/thermostruct2d.py`, each graded against a closed-form oracle by
that repo's own M1 suite (Sod wave positions, the isentropic vortex, the erf
slab, free thermal expansion).  Neither was written with this package in mind,
which is the whole point of pointing this package at them.

**The geometry is a duct, and it is schematic.**  A 0.20 m x 0.02 m channel of
hot gas over a 0.20 m x 8 mm shell segment, 48 cells along the seam on both
sides.  A number measured here is a number about the port algebra and the
envelope stamp; it is not a number about a rocket.

**Two modes, for R10's reason and not for a new one.**

``as-built``    the shell declares ``elliptic_subsolve=embedded``, which is what
                backward Euler conduction and a quasi-static elasticity solve
                both are.  L2 refuses under R10, correctly.
``split-step``  the implicit conduction solve is the composition layer's and the
                agent takes stable explicit sub-steps -- the same move
                ``window_ns`` makes with its pressure projection, by subclassing
                rather than editing an expert that is not ours to change.
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

from ..capability import (
    BCChannel, ClaimType, Differentiable, Direction, EllipticSubsolve,
    ExpertCapabilities, MotionClass, TimeDiscretization, port_decl,
)
from ..graph import (Agent, CaseGraph, Connection, Decomposition, FluxMatching,
                     MeasuredConstants)
from ..ports import PortType, ResponseHalf
from ..transfer import Prolongation

DEFAULT_BUILD_REPO = os.path.join(os.path.expanduser("~"), "physics-foundation-model")

# ---------------------------------------------------------------------------
# geometry and the two states.  Fixed, because every number on the measurement
# page is quoted at one configuration and a moving one is not comparable.
# ---------------------------------------------------------------------------

#: **W70, decided 2026-08-30.**  The scope statement for the half of this expert
#: that is on no port, in `compiler.CROSS_POINT_COUPLING_SCOPE`'s form and for
#: the same reason: an unstated omission is worse than a stated one.
#:
#: The 2026-08-29 framing was "a thermoelastic body coupling is volumetric and
#: the port algebra has no volumetric bond, and a sixth port type is not the fix."
#: The first half of that is right and the second half was aimed at the wrong
#: target, because **the coupling is not between two agents at all** -- the
#: shell's conduction and its elasticity are one expert's internals, and no bond
#: of any kind belongs between them.  Nothing needs a port here.
#:
#: What is actually missing is smaller and worse: **the record describes a strict
#: subset of what the expert computes, and nothing in L1 says so.**  `mechanical`
#: returns a displacement and a stress field; no port carries them, no check
#: reads them, no hypothesis bounds them, and the certificate does not mention
#: that they exist.  A reader of the certificate cannot tell the difference
#: between an expert that computes only what is certified and one that does not.
#:
#: Left as a scope statement rather than a field because the field would be a
#: fourth unverifiable declaration (W69) -- an expert that omits an output from
#: its disclosure is exactly as undetectable as one that mis-declares
#: `response_half`, and buying disclosure with another unverifiable is not
#: obviously a trade worth making. Reopen it when a graph needs the stress.
VOLUMETRIC_COUPLING_SCOPE = (
    "the shell's elasticity is on no port and is never certified. It is not a "
    "missing bond -- the thermoelastic coupling is internal to one expert and "
    "one-way (step_thermal takes no mechanical argument), so the thermal "
    "subsystem is closed and E7's passivity against `storage` holds. What is "
    "uncertified is the OUTPUT: `mechanical` returns a stress field whose "
    "sensitivity to the seam trace is 4.8e-3 per unit at mode 15 against the "
    "certified flow's 2.75e-3, and which RISES with mode number while the "
    "displacement's falls 1200x. The interface truncation error that the "
    "certificate bounds for the flow points the wrong way for the stress (W70)"
)

N_SEAM = 48                  # cells along the seam, both sides -> conforming
NY_GAS = 8                   # gas cells across the duct
NJ_SHELL = 4                 # shell elements through the thickness
L_Z = 0.20                   # m, seam length
L_Y = 0.02                   # m, duct height
T_SHELL = 0.008              # m, shell thickness
H_SEAM = L_Z / N_SEAM        # face cell width, the V-space measure

M_EFF = 16                   # declared interface resolution, as everywhere else

GAMMA, R_GAS = 1.4, 287.0
T_HOT, P_0, U_IN = 900.0, 1.0e5, 60.0        # K, Pa, m/s
T_WALL_0 = 400.0                             # K, the shell's initial temperature
H_OUT, T_OUT = 20.0, 250.0                   # outer-face Robin, W/(m^2 K) and K
H_IN_NOMINAL = 500.0                         # inner-face Robin coefficient

#: The gas macro step.  The build repo runs its gas agents at 1e-3 s; this duct
#: uses 1e-4 because an explicit CFL-0.4 march at 900 K takes ~120 sub-steps to
#: cross it and a probe is 33 of them per column.  Either number is a genuine
#: multirate mismatch against the shell -- 500:1 here, 50:1 there -- so nothing
#: the seam measures depends on which was picked, and a probe that finishes does.
DT_GAS = 1.0e-4              # the gas solver's own macro step
DT_SHELL = 5.0e-2            # the shell's, 50x slower -- thermostruct2d's docstring

#: Explicit conduction sub-steps under ``split-step``.  ``dy^2/(2 alpha)`` with
#: ``alpha = k/(rho cp) = 4.94e-5`` and ``dy = 2 mm`` is 40 ms, so a 50 ms macro
#: step needs sub-stepping and a 1 ms one does not.  Computed, never hard-coded:
#: hard-coding a sub-step count is `CASE-STUDY-GUIDE` mistake 5 and cost two
#: orders of magnitude in tau the one time it was done (tier0-measurements 9.4).
def substeps_at(dt: float) -> int:
    alpha = 120.0 / (2700.0 * 900.0)
    dy = T_SHELL / NJ_SHELL
    dt_stable = 0.4 * dy * dy / (2.0 * alpha)
    return max(1, int(np.ceil(dt / dt_stable)))


def build_repo() -> str:
    return os.environ.get("ATLAS_BUILD_REPO", DEFAULT_BUILD_REPO)


@lru_cache(maxsize=1)
def build_repo_identity() -> str:
    """The build repo's commit, and a hash of its uncommitted diff when there is
    one: ``0a407b7`` or ``0a407b7+dirty:f06f8d898d46f3a0``.

    A number measured through the build repo's solvers is a number about the
    code that ran. Until Tier 86 that was always commit 0a407b7 and the modules
    wrote the string in by hand; Tier 86 changed the build repo, so it is read.

    ``ATLAS_BUILD_REPO_IDENTITY`` overrides it, for a copy of the tree shipped
    without its git history (a rented machine, W334): the identity is read where
    the history is, and the copy is checked against a manifest of that tree's
    files before anything runs."""
    override = os.environ.get("ATLAS_BUILD_REPO_IDENTITY")
    if override:
        return override
    import hashlib
    import subprocess

    def git(*a):
        try:
            return subprocess.run(["git", "-C", build_repo(), *a], capture_output=True,
                                  text=True, encoding="utf-8", timeout=60).stdout
        except (OSError, subprocess.SubprocessError):
            return ""
    commit = git("rev-parse", "--short", "HEAD").strip() or "unknown"
    diff = git("diff", "HEAD")
    if diff.strip():
        return "%s+dirty:%s" % (commit, hashlib.sha256(diff.encode("utf-8")).hexdigest()[:16])
    return commit


@lru_cache(maxsize=1)
def load_solvers(module_name: str = "atlas_build_solvers"):
    """Import the build repo's `atlas` package under a private name.

    Both repositories have a top-level package called ``atlas``.  `window_ns`
    loads the windfarm *subpackage* the same way; the solvers reach further up
    (``grid.py`` imports ``..config`` and ``..geometry.contours``) so the whole
    package is what has to be bound, under a name that collides with nothing.
    """
    pkg_dir = os.path.join(build_repo(), "src", "atlas")
    init = os.path.join(pkg_dir, "__init__.py")
    if not os.path.isfile(init):
        raise RuntimeError(
            f"the experts are not where this case study expects them: {init!r} does "
            "not exist. compressible2d and thermostruct2d live in the build repo "
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
    return (
        importlib.import_module(module_name + ".solvers.compressible2d"),
        importlib.import_module(module_name + ".solvers.thermostruct2d"),
        importlib.import_module(module_name + ".solvers.thermo"),
        importlib.import_module(module_name + ".solvers.grid"),
    )


def fourier_basis(n_cells: int = N_SEAM, m: int = M_EFF) -> np.ndarray:
    """The same 16-mode real Fourier prolongation the other four case studies use.

    Orthonormal in the ``h``-weighted pairing on the midpoint grid, so the V Gram
    is ``h I`` and the forced adjoint needs no solve.  Identical to
    `window_ns.fourier_basis` on purpose: the basis is a declaration, and
    comparing experts through two different declarations confounds the expert
    with the presentation.
    """
    length = n_cells * H_SEAM
    y = (np.arange(n_cells) + 0.5) * H_SEAM
    cols = [np.full(n_cells, 1.0 / np.sqrt(length))]
    k = 1
    while len(cols) < m:
        w = 2.0 * np.pi * k * y / length
        cols.append(np.sqrt(2.0 / length) * np.cos(w))
        if len(cols) < m:
            cols.append(np.sqrt(2.0 / length) * np.sin(w))
        k += 1
    return np.column_stack(cols[:m])


def face_prolongation(agent_id: str, port_name: str) -> Prolongation:
    return Prolongation(
        agent_id=agent_id,
        port_name=port_name,
        matrix=fourier_basis(),
        gram_V=H_SEAM * np.eye(N_SEAM),
        label="16-mode real Fourier basis (probed-dtn-coupling 2.2)",
    )


# ---------------------------------------------------------------------------
# the flux convention -- W66's whole subject
# ---------------------------------------------------------------------------

#: THERM's declared conjugate pair is ``(temperature, entropy flux q_n/T)``.
#: ``"entropy"`` returns that.  ``"heat"`` returns ``q_n``, which is what every
#: thermal solver in the build repo computes and what co-simulation codes
#: exchange -- the *pseudo-bond* `ports.PORT_SPECS[THERM].note` names.  The two
#: differ by a factor of T ~ 400 K, so this is not a small modelling choice; it
#: is the difference between a power and a quantity that is not one.
FLUX_CONVENTIONS = ("entropy", "heat")


def _as_flow(q_n: np.ndarray, T_face: np.ndarray, convention: str) -> np.ndarray:
    """Turn a heat flux into the declared flow variable, or decline to."""
    if convention == "entropy":
        return q_n / np.maximum(T_face, 1.0)
    if convention == "heat":
        return q_n
    raise ValueError(f"flux_convention must be one of {FLUX_CONVENTIONS}, got {convention!r}")


# ---------------------------------------------------------------------------
# agent 1 -- the gas
# ---------------------------------------------------------------------------


@dataclass
class GasAgent:
    """`compressible2d.Compressible2D` on a duct, seam on the ``jmin`` wall.

    The boundary channel is a genuine isothermal wall: ``BC("wall_noslip",
    {"T_wall": ...})`` sets the ghost density so the FACE temperature is the
    imposed one.  That is a Dirichlet channel in temperature, which is what makes
    the trace side of the THERM bond real rather than emulated.
    """

    agent_id: str = "gas"
    dt: float = DT_GAS
    flux_convention: str = "entropy"
    _sol: Any = field(default=None, repr=False)
    _U0: Any = field(default=None, repr=False)
    _blk: Any = field(default=None, repr=False)
    _cfg: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        C2, _TS, TH, GR = load_solvers()
        z = np.linspace(0.0, L_Z, N_SEAM + 1)
        y = np.linspace(0.0, L_Y, NY_GAS + 1)
        Z, Y = np.meshgrid(z, y, indexing="ij")
        self._blk = GR.Block("gas", np.stack([Z, Y], -1), np.zeros((N_SEAM, NY_GAS), bool))
        self._cfg = C2.GasConfig(gamma=GAMMA, R=R_GAS)
        W = np.zeros((N_SEAM, NY_GAS, 4))
        W[..., 0] = P_0 / (R_GAS * T_HOT)
        W[..., 1] = U_IN
        W[..., 3] = P_0
        self._U0 = TH.prim_to_cons(W, GAMMA)
        self._C2, self._TH = C2, TH

    # -- the solve ---------------------------------------------------------
    def _solver(self, T_wall: np.ndarray):
        C2 = self._C2
        return C2.Compressible2D(self._blk, self._cfg, bcs={
            "imin": C2.BC("freestream",
                          {"prim": np.array([P_0 / (R_GAS * T_HOT), U_IN, 0.0, P_0])}),
            "imax": C2.BC("outflow", {"p_inf": P_0}),
            "jmin": C2.BC("wall_noslip", {"T_wall": T_wall}),
            "jmax": C2.BC("wall_slip"),
        })

    def _wall_heat_flux(self, U, T_wall) -> tuple[np.ndarray, np.ndarray]:
        """(q_n into the wall [W/m^2], face temperature [K]).

        Conduction-limited ``h = k_gas/dn`` with Sutherland viscosity, which is
        `generate.py::_wall_flux` verbatim -- deliberately, so the number this
        case study probes is the one the build repo's own coupler exchanges.
        """
        TH, cfg, blk = self._TH, self._cfg, self._blk
        W = TH.cons_to_prim(U, cfg.gamma)
        T = W[..., 3] / (W[..., 0] * cfg.R)
        dn = 0.5 * blk.vol[:, 0] / np.maximum(blk.a_j[:, 0], 1e-30)
        mu = TH.sutherland(T[:, 0], cfg.mu_ref, cfg.T_mu_ref, cfg.sutherland_S)
        k_gas = mu * cfg.cp / cfg.Pr
        Tw = np.broadcast_to(np.asarray(T_wall, float), T[:, 0].shape)
        q = k_gas * (T[:, 0] - Tw) / np.maximum(dn, 1e-9)
        return q, 0.5 * (T[:, 0] + Tw)

    def respond(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        """(wall temperature on the seam) -> (declared THERM flow on the seam)."""
        T_wall = np.asarray(trace, dtype=float).reshape(N_SEAM)
        sol = self._solver(T_wall)
        U, _ = sol.advance(self._U0, self.dt)
        q, T_face = self._wall_heat_flux(U, T_wall)
        return _as_flow(q, T_face, self.flux_convention)

    def respond_integrated(self, port_name: str, trace: np.ndarray,
                           n_substeps: int) -> np.ndarray:
        """The same flow, AVERAGED over ``n_substeps`` of this agent's own clock.

        **R9 / W7, 2026-08-29.**  The one thing `respond` cannot do: it restarts
        from ``_U0`` every call, so calling it n times recomputes the first
        sub-step n times instead of marching n.  This marches, accumulating the
        agent's own sub-step quadrature of the interface flux.

        At ``n_substeps = 1`` it is `respond` exactly -- same initial state, same
        solver, same step -- which is what `conformance._test_integrated_response`
        checks, and it is the reason this callable is not a fourth unverifiable
        declaration.
        """
        n = max(1, int(n_substeps))
        T_wall = np.asarray(trace, dtype=float).reshape(N_SEAM)
        sol = self._solver(T_wall)
        U = self._U0
        acc = np.zeros(N_SEAM)
        for _ in range(n):
            U, _ = sol.advance(U, self.dt)
            q, T_face = self._wall_heat_flux(U, T_wall)
            acc = acc + _as_flow(q, T_face, self.flux_convention)
        return acc / n

    def base_trace(self) -> np.ndarray:
        return np.full(N_SEAM, T_WALL_0)

    def storage(self, state=None) -> float:
        """Internal energy of the gas, the natural Lyapunov functional here."""
        U = self._U0 if state is None else state
        return float(np.sum(U[..., 3] * self._blk.vol))

    def validity(self, state=None, cond=None) -> bool:
        return True


# ---------------------------------------------------------------------------
# agent 2 -- the shell
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def _explicit_shell_class():
    """`ThermoStruct2D` with the implicit conduction solve removed.

    Subclassing rather than editing, for `window_ns._no_projection_class`'s
    reason: the agent is not ours to change, and what the composition layer is
    allowed to do is DECLINE TO USE a part of it and supply that part itself.
    The assembly, the Robin terms and the mechanical solve are untouched; only
    the linear solve for ``T^{n+1}`` is replaced by lumped-mass forward Euler at
    a sub-cadence the diffusion number makes stable.
    """
    _C2, TS, _TH, _GR = load_solvers()

    class ExplicitThermoStruct(TS.ThermoStruct2D):
        def step_thermal(self, T, dt, h_in, T_gas_in, h_out, T_gas_out,
                         radiate=True, T_inf=250.0):
            Kb_i, fb_i = self._robin("inner", h_in, T_gas_in)
            h_o = np.asarray(h_out, dtype=float)
            if radiate:
                a, b, _ = self._face("outer")
                Tw = 0.5 * (T[a] + T[b])
                h_rad = (self.mat.emissivity * 5.670374419e-8
                         * (Tw ** 2 + T_inf ** 2) * (Tw + T_inf))
                h_o = np.broadcast_to(h_o, a.shape) + h_rad
            Kb_o, fb_o = self._robin("outer", h_o, T_gas_out)
            K = self.K_th + Kb_i + Kb_o
            f = fb_i + fb_o
            m = np.asarray(self.M_th.sum(axis=1)).ravel()      # lumped mass
            m = np.maximum(m, 1e-30)
            n = substeps_at(dt)
            h = dt / n
            for _ in range(n):
                T = T + h * (f - K.dot(T)) / m
            return T

    return ExplicitThermoStruct


@dataclass
class ShellAgent:
    """`thermostruct2d.ThermoStruct2D` on the shell segment, seam on the inner face.

    The boundary channel is genuinely ROBIN -- ``-k dT/dn = h (T - T_gas)``, the
    one place in this vault where an expert's own channel sits above `dirichlet`
    on the ladder and the declaration is a fact rather than a convenience.
    """

    agent_id: str = "shell"
    dt: float = DT_SHELL
    flux_convention: str = "entropy"
    expose_elliptic: bool = False
    h_in: float = H_IN_NOMINAL
    _ts: Any = field(default=None, repr=False)
    _mesh: Any = field(default=None, repr=False)
    _T0: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _C2, TS, _TH, _GR = load_solvers()
        nodes = np.stack(np.meshgrid(np.linspace(0.0, L_Z, N_SEAM + 1),
                                     np.linspace(0.0, T_SHELL, NJ_SHELL + 1),
                                     indexing="ij"), -1)
        self._mesh = TS.ShellMesh(nodes)
        cls = _explicit_shell_class() if self.expose_elliptic else TS.ThermoStruct2D
        self._ts = cls(self._mesh, TS.SolidMaterial())
        self._T0 = np.full(self._mesh.n_nodes, T_WALL_0)

    def _face_T(self, T: np.ndarray) -> np.ndarray:
        a, b, _L = self._ts._face("inner")
        return 0.5 * (T[a] + T[b])

    def respond(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        """(gas temperature on the seam) -> (declared THERM flow on the seam).

        The trace enters through the agent's own Robin channel as ``T_gas``; the
        response is the heat that crosses the inner face over one step,
        ``h (T_gas - T_wall)``, converted to the declared flow variable.
        """
        T_gas = np.asarray(trace, dtype=float).reshape(N_SEAM)
        T = self._ts.step_thermal(self._T0, self.dt, self.h_in, T_gas,
                                  H_OUT, T_OUT, T_inf=T_OUT)
        T_face = self._face_T(T)
        q = self.h_in * (T_gas - T_face)                  # W/m^2, into the shell
        return _as_flow(q, 0.5 * (T_gas + T_face), self.flux_convention)

    def respond_integrated(self, port_name: str, trace: np.ndarray,
                           n_substeps: int) -> np.ndarray:
        """The same flow, averaged over ``n_substeps`` of the shell's own clock.

        See `GasAgent.respond_integrated`.  At the native clocks this side's
        quadrature is a single step -- the shell IS the slow agent, so the
        exchange interval is its own -- and the method exists anyway, because R9
        requires *both* sides to be able to state their integral and a rule that
        only the fast side has to satisfy is not a matching condition.
        """
        n = max(1, int(n_substeps))
        T_gas = np.asarray(trace, dtype=float).reshape(N_SEAM)
        Tn = self._T0
        acc = np.zeros(N_SEAM)
        for _ in range(n):
            Tn = self._ts.step_thermal(Tn, self.dt, self.h_in, T_gas,
                                       H_OUT, T_OUT, T_inf=T_OUT)
            T_face = self._face_T(Tn)
            q = self.h_in * (T_gas - T_face)
            acc = acc + _as_flow(q, 0.5 * (T_gas + T_face), self.flux_convention)
        return acc / n

    def mechanical(self, T=None, p_in=P_0, p_out=0.5 * P_0):
        """The elasticity half. Not on any port -- see VOLUMETRIC_COUPLING_SCOPE.

        **Measured 2026-08-30 (W70).**  Three numbers decide what the omission
        costs, and they do not all point the same way:

        1. The coupling is **one-way**.  ``ThermoStruct2D.step_thermal`` takes
           ``(T, dt, h_in, T_gas_in, h_out, T_gas_out, radiate, T_inf)`` and no
           displacement, strain or stress, so temperature drives deformation and
           deformation never returns.  The thermal subsystem is closed, and the
           passivity argument taken against `storage` is therefore not wrong.
        2. The elastic strain energy is **negligible as an energy**:
           ``1/2 u^T K u`` is 2.0e-5 of ``1/2 T^T M T`` at the operating point
           (2.8e4 J against 1.4e9 J), rising only to 2.5e-5 at 1200 K.
        3. And the stress is **more sensitive to the seam trace than the
           certified port quantity is**.  Per unit trace perturbation,
           ``d|sigma|/|sigma|`` runs 8.5e-4 to 5.5e-3 against the port flow's
           2.75e-3 -- and it RISES with mode number (mode 0: 8.5e-4, mode 15:
           4.8e-3) while ``d|u|/|u|`` FALLS by 1200x over the same modes.

        (3) is the finding.  Displacement integrates and stress differentiates,
        so the high interface modes -- exactly the ones a 16-mode truncation
        discards -- drive the stress hardest and the displacement least.  The
        truncation argument that bounds the error in the certified flow is
        therefore **not** a bound on the uncertified stress; it points the wrong
        way for it.
        """
        T = self._T0 if T is None else T
        return self._ts.solve_mechanical(T, p_in, p_out)

    def base_trace(self) -> np.ndarray:
        return np.full(N_SEAM, T_HOT)

    def storage(self, state=None) -> float:
        """Thermal energy ``1/2 T^T M T`` -- the conduction operator's own
        Lyapunov functional, SPD by construction from the Q1 mass matrix."""
        T = self._T0 if state is None else np.asarray(state)
        return float(0.5 * T @ self._ts.M_th.dot(T))

    def validity(self, state=None, cond=None) -> bool:
        return True


# ---------------------------------------------------------------------------
# the records
# ---------------------------------------------------------------------------

#: (T, q_n/T) in K and W/(m^2 K): s_e s_f = s_P with s_P in W/m^2.
THERM_SCALES = {"temperature": T_HOT, "entropy_flux": 1.0e4 / T_HOT,
                "power_area": 1.0e4}


def gas_capabilities(expert: GasAgent) -> ExpertCapabilities:
    return ExpertCapabilities(
        expert_id=expert.agent_id,
        ports=[port_decl(
            name="wall:THERM", port_type=PortType.THERM,
            geometry="jmin wall of the duct, 48 cells",
            direction=Direction.BIDIRECTIONAL,
            nondim=dict(THERM_SCALES),
            effective_resolution=M_EFF,
            motion_class=MotionClass.STATIC,
            # W66 / L3-C9: the response is the entropy flux q_n/T -- THERM's
            # declared FLOW -- so the imposed trace is the temperature, the
            # effort. Under flux_convention="heat" the callable returns the
            # pseudo-bond instead and this declaration becomes a lie that
            # nothing numerical can catch; that is the measurement, not a bug.
            response_half=ResponseHalf.FLOW,
            prolongation=face_prolongation(expert.agent_id, "wall:THERM"),
            note="isothermal no-slip wall; the trace is T_wall and the response "
                 "is the conduction-limited wall flux generate.py exchanges",
        )],
        # BC("wall_noslip", {"T_wall": ...}) sets the ghost so the FACE
        # temperature is the imposed one. A temperature Dirichlet channel.
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=True,
        # MUSCL + SSP-RK2 + HLLC: an explicit finite-volume march. The domain of
        # dependence is two cells per sub-step and there is no solve of any kind.
        elliptic_subsolve=EllipticSubsolve.NONE,
        time_discretization=TimeDiscretization.EXPLICIT,
        stencil_radius=2,                     # NG = 2, MUSCL needs two ghosts
        substeps_per_macro_step=1,
        differentiable=Differentiable.NONE,
        dt_native=expert.dt,
        storage=expert.storage,
        equivariances=("translation-z",),
        validity=expert.validity,
        governing_family="compressible-navier-stokes-2d",
        # **W83, 2026-08-29.** The declaration that a REFERENT exists for tau
        # at this seam. E3 fails here and always will -- the two sides really do
        # solve different equations -- but tau never needed a shared governing
        # family, only a reference TRAJECTORY, and at a multiphysics seam that is
        # the tightly coupled pair. This agent's reference is itself: it is the
        # real solver, so its infidelity is zero by construction, and the point of
        # declaring it is that swapping in a learned expert then MEASURES that
        # expert's infidelity instead of emitting UNDEFINED.
        lambda_ref="compressible2d.Compressible2D, same duct and discretization",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash="compressible2d/2-hllc-minmod-ssprk2",
        boundary_response=expert.respond,
        # R9: the flux INTEGRAL over an exchange interval, on this agent's own
        # sub-steps. Required at a multirate seam that declares time-integrated
        # matching, and unrecoverable from `respond` (which restarts each call).
        boundary_response_integrated=expert.respond_integrated,
        # W74: this port's effort is an absolute temperature, so the probe's
        # default zero trace is 0 K. Declaring the operating point moves beta
        # on this seam by 12.6x.
        probe_base=lambda _port, _e=expert: _e.base_trace(),
        reproducibility_floor=float(np.finfo(float).eps),
        deterministic=True,
        note="solvers/compressible2d.py, build repo, unmodified; graded against "
             "the exact Riemann solution and the isentropic vortex by that "
             "repo's M1 suite",
    )


def shell_capabilities(expert: ShellAgent) -> ExpertCapabilities:
    exposed = expert.expose_elliptic
    return ExpertCapabilities(
        expert_id=expert.agent_id,
        ports=[port_decl(
            name="inner:THERM", port_type=PortType.THERM,
            geometry="inner face of the shell segment, 48 line elements",
            direction=Direction.BIDIRECTIONAL,
            nondim=dict(THERM_SCALES),
            effective_resolution=M_EFF,
            motion_class=MotionClass.STATIC,
            response_half=ResponseHalf.FLOW,
            prolongation=face_prolongation(expert.agent_id, "inner:THERM"),
            note="the agent's own Robin surface term, -k dT/dn = h (T - T_gas)",
        )],
        # `_robin` is a Robin condition and not an emulation of one. This is the
        # only expert in the vault whose channel is genuinely above `dirichlet`.
        bc_channel=BCChannel.ROBIN,
        bc_time_varying=True,
        # Backward Euler solves (M/dt + K + K_robin) T = rhs over the whole
        # shell, and the quasi-static elasticity solve is global with no time
        # step to shrink at all. Both are EMBEDDED in R10's exact sense.
        elliptic_subsolve=(EllipticSubsolve.EXPOSED if exposed
                           else EllipticSubsolve.EMBEDDED),
        time_discretization=(TimeDiscretization.EXPLICIT if exposed
                             else TimeDiscretization.IMPLICIT),
        stencil_radius=1,                     # Q1 elements, one ring
        substeps_per_macro_step=substeps_at(expert.dt) if exposed else 1,
        differentiable=Differentiable.NONE,
        dt_native=expert.dt,
        storage=expert.storage,
        equivariances=("translation-z",),
        validity=expert.validity,
        governing_family="thermoelastic-shell-2d",
        lambda_ref="thermostruct2d.ThermoStruct2D, same shell and discretization",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash=f"thermostruct2d/1-{'explicit' if exposed else 'backward-euler'}",
        boundary_response=expert.respond,
        boundary_response_integrated=expert.respond_integrated,
        # W74: this port's effort is an absolute temperature, so the probe's
        # default zero trace is 0 K. Declaring the operating point moves beta
        # on this seam by 12.6x.
        probe_base=lambda _port, _e=expert: _e.base_trace(),
        reproducibility_floor=float(np.finfo(float).eps),
        deterministic=True,
        note="solvers/thermostruct2d.py, build repo; conduction graded against "
             "the erf slab and free expansion against the M1 stress oracle",
    )


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------


#: **W83, measured 2026-08-29** by `scripts/w83_multiphysics_attribution.py`.
#: The first measured constants for a multiphysics graph in this vault, and they
#: exist because `tau` stopped being UNDEFINED across a family boundary.
#:
#:   tau   = 0 EXACTLY, for both agents. These are the real solvers, so each is
#:           its own reference and its infidelity is zero by construction. That
#:           is worth measuring rather than assuming: the same measurement run
#:           against a gas surrogate carrying a 5%, 25% and 100% conductivity
#:           error recovers tau = 0.05, 0.25 and 1.00 to the digit, per agent,
#:           with the unswapped side staying identically zero.
#:   sigma = 3.392278e-05, at the lag a real run actually carries -- the PREVIOUS
#:           macro-step's converged trace. It is not a property of the seam
#:           alone: sigma is linear in the lag distance TO LEADING ORDER, at
#:           3.44e-2 per K near the operating point, and the interface drifts
#:           9.83e-4 K per macro-step here, so 3.44e-2 x 9.83e-4 = 3.38e-5
#:           predicts the measured 3.392e-5. It is only asymptotically linear:
#:           interface power is BILINEAR (effort times flow), so there is a
#:           quadratic correction, and the measured ratios go 9.99 per decade at
#:           0.1-1 K, 4.97 against 5 at 1-5 K, and 3.91 against 4 at 5-20 K. A
#:           single slope in "per K" is right near the operating point and wrong
#:           away from it, which is why the lag is part of the provenance.
#:           Measured at the initial condition instead it is 1.0000 exactly,
#:           because the shell starts AT the lagged temperature and transmits no
#:           power there. A sigma without its lag is not a number.
#:
#: Both are in interface power (effort x flow), which is the only unit the two
#: sides share -- see `atlas/multiphysics.py` for why neither agent's own state
#: norm can carry this.
#: **W86, corrected 2026-08-29.**  Three things about the ``sigma`` above were
#: true and unstated, and the third made its provenance wrong.
#:
#: 1. It is a **uniform-shift** value, not a measurement at any lag a run
#:    carries: ``3.4501e-2 K^-1 x 9.83e-4 K = 3.3914e-5`` reproduces it to four
#:    digits.  The slope is a genuine seam property -- constant to five digits
#:    over five macro-steps -- and the lag is the assumed drift.
#: 2. The run's own sigma, walking the reference trajectory with the lag it
#:    actually carries, is **4.07e-6 to 3.13e-5 over five consecutive steps**, a
#:    7.70x range, all below this.  So the declared value is conservative here,
#:    which is the safe direction for a bound and is not the same as correct.
#: 3. ``source`` named a script stage that emits a *different* number:
#:    `w83.json`'s ``measured_sigma`` is 1.0773e-11, the sigma at ZERO lag, which
#:    is the check that sigma measures the lag and not a value to quote.  The
#:    stage that produces this one is `w87_referent_cost_and_lag.py`'s ``lag``.
#:
#: ``sigma_lag`` now carries the lag as a number so `solve.rollout` can compare it
#: against the lag the run derives from its own consecutive traces.  That
#: comparison can falsify and cannot confirm: sigma's argument is the lag
#: PROFILE, and at two steps whose scalar lags agree to 8% it differs by 1.48x.
MEASURED_W83 = MeasuredConstants(
    tau=0.0,
    sigma=3.392278e-05,
    sigma_lag=9.83e-04,
    probe_state="duct, T_hot=900 K, T_wall=400 K, matched clocks, interface at "
                "lambda* = 372.1377 K",
    scheme="split-step",
    depth=0,
    source="sigma slope 3.4501e-2 /K measured by scripts/w87_referent_cost_and_lag.py "
           "stage lag, at a uniform lag of 9.83e-4 K; tau by "
           "scripts/w83_multiphysics_attribution.py stage tau. Interface power norm",
)


def make_experts(mode: str = "as-built", flux_convention: str = "entropy",
                 clocks: str = "native") -> dict[str, Any]:
    if clocks not in ("native", "matched"):
        raise ValueError(f"clocks must be 'native' or 'matched', got {clocks!r}")
    # `matched` brings the SHELL down to the gas's clock, never the gas up to the
    # shell's. Backward Euler is unconditionally stable so a 0.1 ms shell step is
    # merely a smaller step; an explicit CFL-0.4 gas march at 50 ms is 60,000
    # sub-steps per probe column, which is not a measurement anyone runs.
    dt_shell = DT_SHELL if clocks == "native" else DT_GAS
    return {
        "gas": GasAgent(dt=DT_GAS, flux_convention=flux_convention),
        "shell": ShellAgent(dt=dt_shell, flux_convention=flux_convention,
                            expose_elliptic=(mode == "split-step")),
    }


def build(mode: str = "as-built", flux_convention: str = "entropy",
          clocks: str = "native", experts: dict[str, Any] | None = None,
          measured="default", flux_matching="default") -> tuple[CaseGraph, dict[str, Any]]:
    """The two-agent multiphysics graph, plus the experts backing it.

    ``mode``             ``as-built`` (shell elliptic embedded, L2/R10 refuses)
                         or ``split-step`` (exposed, as `window_ns`)
    ``flux_convention``  ``entropy`` (THERM's declared flow) or ``heat``
                         (the pseudo-bond every thermal code exchanges) -- W66
    ``clocks``           ``native`` (0.1 ms against 50 ms, E4 fails honestly) or
                         ``matched`` (the shell brought down to the gas's clock,
                         so E3 is measured with E4 out of the way)

    ``measured`` defaults to this graph's OWN measured constants as of
    2026-08-29 (`MEASURED_W83`: tau and sigma, in interface power).  Until then
    it defaulted to None with the note that "nothing has been measured for this
    graph and borrowing another graph's constants would be W56 on purpose" --
    which was true, and stopped being true when `tau` became measurable across a
    governing-family boundary.  Pass ``measured=None`` to get the pre-W83 compile
    back; nothing here borrows another graph's numbers.

    ``flux_matching`` defaults to `FluxMatching.TIME_INTEGRATED` at the NATIVE
    clocks and `POINTWISE` at the matched ones, because that is what each case
    actually is rather than a preference: across one clock the two conventions
    coincide exactly, and across 500:1 R9 requires the integral.  Pass
    ``FluxMatching.POINTWISE`` at the native clocks to get the pre-2026-08-29
    compile back -- a refusal, and the one this graph carried since it was
    written.  **What declaring it buys is measured and is smaller than it looks**:
    the conserved integral is matched, and on this seam the flux the pointwise
    convention loses is 1.0e-4 of the transported heat, against a lag defect over
    the same interval that is far larger.  See `w7_multirate_matching.py`.
    """
    if mode not in ("as-built", "split-step"):
        raise ValueError(mode)
    if flux_convention not in FLUX_CONVENTIONS:
        raise ValueError(flux_convention)
    if measured == "default":
        measured = MEASURED_W83 if clocks == "matched" else None
    if flux_matching == "default":
        flux_matching = (FluxMatching.TIME_INTEGRATED if clocks == "native"
                         else FluxMatching.POINTWISE)
    experts = experts or make_experts(mode, flux_convention, clocks)

    agents = [
        Agent("gas", gas_capabilities(experts["gas"]), domain="duct, 0.20 x 0.02 m"),
        Agent("shell", shell_capabilities(experts["shell"]),
              domain="shell segment, 0.20 m x 8 mm", role="structure"),
    ]
    connections = [Connection(
        seam_id="cht",
        a=("gas", "wall:THERM"),
        b=("shell", "inner:THERM"),
        port_type=PortType.THERM,
        geometrically_coincident=True,
        derive_space=True,
        # n_0(Gamma) = 0. A conjugate-heat-transfer seam constrains nothing the
        # way incompressibility constrains a MECH trace: a uniform temperature
        # shift produces a uniform flux change, so no direction of the trace
        # space is invisible to the response. `CASE-STUDY-GUIDE`'s table has no
        # row for this seam type, which is itself worth saying out loud.
        expected_null_dim=0,
        note="conjugate heat transfer: compressible gas against a conducting "
             "shell. The two sides solve different equations and say so",
    )]

    return CaseGraph(
        name=f"thermal-seam-{mode}-{flux_convention}-{clocks}",
        agents=agents,
        connections=connections,
        # Non-overlapping: the gas and the shell are different MATERIALS and
        # share a surface, not a region. There is no cell that belongs to both
        # and therefore no partition of unity to declare -- which puts this graph
        # on the substructuring branch, where W57 says no cut criterion exists.
        decomposition=Decomposition.NON_OVERLAPPING,
        macro_dt=max(a.capabilities.dt_native for a in agents),
        flux_matching=flux_matching,
        measured=measured,
        note="the fifth real case study and the first multiphysics one: "
             "compressible2d against thermostruct2d, both build-repo solvers "
             "imported unmodified. Geometry schematic; the seam is real",
    ), experts
