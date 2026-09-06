"""CS-11: a BOUND for the multirate lag defect.  (W90, W128)

A brake disc against its cooling duct.  The disc conducts -- slowly, implicitly,
with an `EMBEDDED` backward-Euler solve over its whole cross-section -- and the
duct convects, fast and explicitly, at the step its own CFL condition gives.
They meet at one `THERM` seam, and their clocks are **500:1 apart because the
physics puts them there**: a cast-iron wall has a thermal time constant of
seconds and a duct cell empties in tens of microseconds.

`thermal_seam` already carries a mismatch of exactly this size and compiles.
What it does not have, and what this case study exists to supply, is a **bound**.

W90, stated as the row states it
---------------------------------

    R9 covers the flux transient and is 62x too small; the lag over a long
    exchange interval is sigma, and nothing bounds it as a function of that
    interval.  R4 forbids shrinking the interval below max_i dt_i, so the
    obvious remedy is the one axis a frozen expert cannot move.

Three things follow, and each is a section of `scripts/w131_brake_thermal.py`.

1. **The defect is first order in the exchange interval and the constant is a
   seam property.**  `sigma(DT) <= s_seam * lambda_dot * DT + C2 * (lambda_dot *
   DT)^2`, with `s_seam = d sigma / d(uniform lag)` -- W86's transferable slope,
   the one thing about sigma that is a property of the seam rather than of the
   run -- and `lambda_dot` the interface trace's own rate of change, which is a
   property of the run and which only a run can supply.  Two measured constants
   and one run-derived rate, which is the same shape as `sigma_lag` and is why
   it belongs beside it rather than instead of it.

2. **The clock RATIO is not an independent variable, and the exchange INTERVAL
   is.**  This is the finding the sweep is built to isolate and it was not
   obvious in advance.  At fixed `DT` the ratio can be moved over a decade by
   refining the duct's own step and sigma does not move: the trace is stale for
   the interval regardless of how the fast agent chops it up.  What the ratio
   moves is R9's term -- the fast side's flux transient -- which is the smaller
   one.  So R4's floor is the whole of the multirate constraint on this defect,
   and a bound written in the interval transfers to a ratio nobody can afford
   to run.  **Scoped**: it holds for a fast agent that sub-steps internally at
   its own stability limit, which is what a classical explicit solver does.  A
   frozen checkpoint whose `dt_native` cannot be refined is the case where the
   ratio does re-enter, and this case study cannot see it.

3. **W17's `W > 1` is the remedy the row names, and it is admissible here.**
   R3 requires `bc_time_varying` on every agent at the interface and both of
   these have it, so a trace carried as a linear waveform across the interval is
   legal by rule rather than by hope.  It cuts the lag from `lambda_dot * DT` to
   the second difference, and the measured reduction is in the driver's
   `waveform` stage.  It is the one axis R4 leaves open, and it moves the
   interval's own exponent rather than its constant.

And W128, folded in
--------------------

CS-10 opened W128 because a design sensitivity taken from a rollout is a
function of the horizon and has no field on the record -- measured there at
`+0.411` on a 40-step rollout and `-0.152` on a 160-step one, a **sign change**,
with every value confirmed by its own finite difference.  This graph carries a
design knob of its own -- `U_DUCT`, the cooling duct's inlet velocity, which is
what a duct is sized for -- and the same law is measured on it here, on
different physics, with a **closed-form lumped prediction to check it against**:

    dT/dtheta (x) = T_eq' (1 - e^{-x/tau})  -  (T_0 - T_eq) e^{-x/tau} (x/tau) / H

whose short-horizon limit is proportional to `(T_gas - T_0)` and whose long-
horizon limit is `T_eq'`.  **The two have opposite signs whenever the disc is
released colder than the air cooling it**, which is a brake at the start of a
lap in a duct that has already been warmed, and it is the release state this
case study runs at.  So the sign change is not an accident of one objective:
it has a mechanism, the mechanism is written down before the run, and the
horizon at which the reported sensitivity is within a tolerance of its converged
value follows from the same two-term form.

How the gradient is taken, and the rule that says so
-----------------------------------------------------

By **central finite difference**, and that is R7's own branch rather than a
concession.  R7 reads the `differentiable` field: ``jvp`` takes an exact JVP
probe, and *"``none`` but deterministic and smooth -> finite differences with
epsilon set by the reproducibility floor"*.  Both agents here declare
``Differentiable.NONE`` beside ``deterministic=True`` and a
``reproducibility_floor`` of one ulp, which is the triple that selects it, and
the pipeline is bit-reproducible so the floor is real.  CS-10 is what makes this
affordable to trust: it swept a central difference against an exact reverse-mode
adjoint on a composed stack over five decades of step and found agreement to
``1.32e-6``, on a truncation branch that had not yet reached cancellation.  The
instrument is calibrated; this case study uses it and sweeps the step to show
the branch rather than asserting it.

**`CASE-STUDY-GUIDE` names a `Differentiable.FD` member and the enum has none**
-- its members are ``NONE``, ``JVP``, ``VJP`` and ``BOTH``.  Found by declaring
what the guide says and getting an `AttributeError`, which is the harmless end of
this class: the guide also lists `VJP` and `BOTH` nowhere, so a record that wants
to declare a reverse mode has no documented way to learn it can.  Corrected in
the guide by this case study.

What is reused, and what is new
--------------------------------

  the disc      `thermostruct2d.ThermoStruct2D` from the build repo, imported
                verbatim through `thermal_seam.load_solvers` and unmodified --
                the same expert CS-9 split and CS-5 put against a gas, here on
                a cast-iron disc section rather than an airframe shell.  Its
                `step_thermal` is backward Euler over the whole cross-section:
                `EMBEDDED` in R10's exact sense, which is what the ladder's
                CS-11 row asks for and which `L2/R10` duly refuses.
  the seam      `THERM`, the same declared bond `(T, q_n/T)` and the same
                `flux_convention` switch `thermal_seam` uses for W66.
  the referent  the **single-rate** solve: the same two agents, the same
                interface condition, exchanging at the DUCT's clock so that the
                lag is 500x smaller.  It is the same code path with one
                argument changed, so the zero-lag control is exact.

New here: the duct.

The duct, stated as the model it is
------------------------------------

A **low-Mach convective duct**: two-dimensional transport of gas temperature
along a cooling passage,

    dT/dt + u(y) dT/dx = d/dy[ alpha_eff(y) dT/dy ] + alpha_eff d2T/dx2

marched explicitly -- first-order upwind in x, central in y, sub-cycled at its
own CFL limit -- with the seam wall as a Dirichlet temperature and the far wall
adiabatic.  It is a real solver: it marches a PDE, its domain of dependence is
`stencil_radius * substeps` and finite, and its `boundary_response` is a solve
and not a formula.  It is **not** a build-repo expert, and saying so is the
scope statement `CASE-STUDY-GUIDE` asks for: one side of this seam is a
validated donor and the other is written here.  `tests/test_tier28_brake_thermal.py`
grades it against two closed-form oracles -- pure advection of a step, and the
erf solution of the half-space diffusion problem -- because an expert nobody
graded is a fixture wearing a solver's name.

**Why not `compressible2d`, which is right there.**  Because the referent has to
be affordable.  A compressible duct is acoustically limited at `dt = 3.5e-6 s`,
so a single-rate referent over the disc's own transient is `10^7` sub-steps,
and the one thing this case study cannot do without is the single-rate solve it
grades against.  A brake duct runs at `M = 0.12`; the acoustics are not the
physics, and dropping them is what makes the control exist.  The clock ratio is
still what the physics gives -- a convective cell transit against a conduction
time constant -- and it is still 500:1.

**alpha_eff carries a turbulence closure, and its constants are textbook.**  A
molecular-conductivity duct gives a wall coefficient near 15 W/(m^2 K) and a
disc time constant of fifteen minutes, which is not a brake.  `eddy_diffusivity`
is the classical linear-in-wall-distance mixing model, `nu_t = kappa u_tau d`,
with `kappa = 0.41`, a turbulent Prandtl number of `0.85` and Blasius' friction
law -- three constants of the literature and none fitted here.  What comes out
is a wall coefficient of 500 to 1600 W/(m^2 K) over the sweep and, unprompted, a
`U^0.83` scaling against Dittus-Boelter's `U^0.8`.  That agreement is a check on
the closure, not an input to it.

The friction face is a Robin condition and that is a modelling choice
----------------------------------------------------------------------

`ThermoStruct2D.step_thermal` offers two Robin faces and no volumetric source,
so the pad's friction heating enters as the outer face's `(h_pad, T_pad)`.  A
prescribed flux would be the textbook brake model; a Robin with a finite `h_pad`
is that with a mild negative feedback -- the friction power falls slightly as
the disc heats -- which is physically the right direction and is what the expert
can express without being modified.  `radiate=False`, because the outer face is
a contact interface rather than a radiating one, and because it keeps
`step_thermal` **affine in the trace**, which is what makes the probe's
linearization exact and the slope `s_seam` a slope rather than a secant.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Sequence

import numpy as np

from ..capability import (
    BCChannel, ClaimType, Differentiable, Direction, EllipticSubsolve,
    ExpertCapabilities, MotionClass, TimeDiscretization, port_decl,
)
from ..graph import (Agent, CaseGraph, Connection, Decomposition, FluxMatching,
                     MeasuredConstants)
from ..ports import PortType, ResponseHalf
from ..transfer import Prolongation
from .thermal_seam import load_solvers

# ---------------------------------------------------------------------------
# geometry, materials and the two clocks
# ---------------------------------------------------------------------------

N_SEAM = 32                  # cells along the seam, both sides -> conforming
NY_DUCT = 8                  # cells across the cooling duct
NJ_DISC = 6                  # elements through the disc wall

L_R = 0.08                   # m, radial extent of the cooled band -- the seam
H_DUCT = 0.016               # m, duct height
T_DISC = 0.0025              # m, disc wall thickness (a vented disc's wall)

H_SEAM = L_R / N_SEAM        # face cell width, the V-space measure
DX = L_R / N_SEAM
DY = H_DUCT / NY_DUCT

#: Declared interface resolution.  **Measured, and the measurement says the
#: question does not apply at this seam.**
#:
#: `effective_resolution` is defined as *"the expert's own measured spectral
#: cutoff, not a chosen number"*, and `wake_array.modes_for` turns a cutoff
#: WAVELENGTH into a mode count.  Probed here (driver stage `resolution`), the
#: seam response operator is **the identity to a condition number of 1.03**:
#: off-diagonal entries are 1% of the diagonal, the singular values fall from
#: 1.000 to 0.977 across all 32 of them, and the response to a unit cosine is
#: flat in mode number.  A conjugate-heat seam has no spectral cutoff, because
#: the flux at a cell is set by the temperature AT that cell -- the transport
#: along the seam is a 1% correction to it.
#:
#: The consequence is arithmetic and it is why this is declared at the grid: for
#: an operator with no small singular directions, a rank-m truncation discards
#: exactly the fraction of the modes it drops.  Measured, an 11-mode basis on
#: this 32-cell face loses **81% of the operator in norm** and even 31 modes
#: lose 17.8%.  `thermal_seam` declares 16 modes on 48 cells *"as everywhere
#: else"* and that is the number this would have inherited.
M_EFF = N_SEAM               # no cutoff exists to declare; the grid is the space

# --- the gas in the duct: air at the duct's inlet condition -----------------
RHO_G = 0.84                 # kg/m^3
CP_G = 1030.0                # J/(kg K)
K_G = 0.037                  # W/(m K)
MU_G = 2.6e-5                # Pa s
NU_G = MU_G / RHO_G
ALPHA_G = K_G / (RHO_G * CP_G)
PR_T = 0.85                  # turbulent Prandtl number
KAPPA = 0.41                 # von Karman

T_IN = 420.0                 # K, duct inlet air -- warmed on its way to the brake
U_DUCT = 40.0                # m/s, the DESIGN KNOB's nominal value
CFL = 0.4

# --- the disc --------------------------------------------------------------
H_IN = 2000.0                # W/(m^2 K), disc inner-face (duct-side) Robin
H_PAD = 250.0                # W/(m^2 K), friction-face contact conductance
T_PAD = 1150.0               # K, the pad's effective driving temperature
T_DISC_0 = 300.0             # K, the disc is released COLD -- see the docstring

#: The two native clocks, and the ratio they give.
#:
#:   the disc  `tau_disc = rho cp t / (h_pad + h_eff)` is about 7.5 s at this
#:             operating point, and 0.05 s resolves it in 150 steps.  Backward
#:             Euler is unconditionally stable, so this is an ACCURACY choice
#:             and it is declared as one.
#:   the duct  a cell empties in `dx / u = 5.7e-5 s` and the explicit stability
#:             limit is `2.05e-5 s`; 1e-4 s is the duct's own macro-step with
#:             five sub-steps inside it, which is what `substeps_per_macro_step`
#:             declares and what `required_halo` reads.
#:
#: **500:1, and it is the physics rather than a setting.**  `thermal_seam` has
#: the same number from entirely different solvers, which is the point: a
#: conduction-against-convection seam is where this ratio comes from.
DT_DISC = 0.05
DT_DUCT = 1.0e-4
CLOCK_RATIO = int(round(DT_DISC / DT_DUCT))          # 500

#: **W74's discipline, applied rather than rediscovered.**  The probe base
#: belongs to the SEAM, not to either expert.  Left to themselves the two sides
#: here would declare 300 K (the disc's release temperature, which is what the
#: duct sees) and 420 K (the inlet air, which is what the disc sees) -- each
#: right alone, 120 K apart, and wrong summed, which is precisely the
#: `thermal_seam` failure W74 names and which `L4/probe-base` decertifies.
#:
#: The seam's own value is the interface temperature the two responses balance
#: at, and on affine responses it is closed form -- `solve_interface`'s formula
#: evaluated at the release state:
#:
#:     lambda_0 = (h_gas T_in + h_in T_disc_0) / (h_gas + h_in)
#:
#: with ``h_gas = k_eff / (dy/2)`` the duct's own first-cell coefficient.  Both
#: records declare THIS, so `L4/probe-base` reads consistent -- and it is worth
#: saying that the base is only stationary because the probe is: the seam value
#: drifts from 337 K to above 550 K as the disc heats, so a certificate taken
#: here is a certificate at the release state and nowhere else.
def seam_base(u_bulk: float = U_DUCT, t_disc: float = T_DISC_0) -> float:
    h_gas = eddy_diffusivity(u_bulk)[0] * RHO_G * CP_G / (0.5 * DY)
    return float((h_gas * T_IN + H_IN * t_disc) / (h_gas + H_IN))


def solid_material():
    """Cast iron, as a `SolidMaterial`.  A declaration, not a solver change.

    `thermostruct2d.SolidMaterial`'s defaults are an aluminium-lithium airframe
    alloy, which is the material the expert was written for and is not what a
    brake disc is made of.  The dataclass is the expert's own parameterization
    and passing a different instance is using it, not modifying it.
    """
    _C2, TS, _TH, _GR = load_solvers()
    return TS.SolidMaterial(rho=7150.0, cp=460.0, k=48.0, E=110.0e9, nu=0.26,
                            alpha=11.0e-6, emissivity=0.6)


def modes_for(n_cells: int = N_SEAM, lambda_cut: float = 6.4) -> int:
    """`wake_array.modes_for`: a cutoff is a WAVELENGTH, not a mode count.

    Kept so the driver can report what a cutoff-derived declaration WOULD have
    been, against a seam that turns out not to have one.
    """
    return max(1, int(round(2.0 * n_cells / lambda_cut + 1.0)))


def fourier_basis(n_cells: int = N_SEAM, m: int = M_EFF) -> np.ndarray:
    """The same real Fourier prolongation every other case study declares.

    Orthonormal in the ``h``-weighted pairing on the midpoint grid, so the V
    Gram is ``h I``.  Identical in form to `thermal_seam.fourier_basis` on
    purpose: the basis is a declaration, and comparing two seams through two
    different declarations confounds the seam with the presentation.
    """
    length = n_cells * H_SEAM
    y = (np.arange(n_cells) + 0.5) * H_SEAM
    cols = [np.full(n_cells, 1.0 / np.sqrt(length))]
    k = 1
    while len(cols) < m:
        w = 2.0 * np.pi * k * y / length
        # **The Nyquist column is identically zero on a midpoint grid, and the
        # other case studies never saw it.**  At k = n/2 the cosine samples at
        # cos(pi (i + 1/2)), which is 0 for every cell, so `column_stack` would
        # return a rank-deficient "basis" and every projection built on it would
        # silently drop a direction.  `window_ns`, `thermal_seam` and
        # `wake_array` all declare m well below n and so never reach k = n/2.
        # The SINE at that k is the real Nyquist mode -- sin(pi (i + 1/2)) is
        # the alternating sequence -- so take it and skip the cosine.
        c = np.sqrt(2.0 / length) * np.cos(w)
        if np.linalg.norm(c) > 1e-12:
            cols.append(c)
        if len(cols) < m:
            sn = np.sqrt(2.0 / length) * np.sin(w)
            if np.linalg.norm(sn) > 1e-12:
                cols.append(sn / np.linalg.norm(sn) * np.sqrt(1.0 / H_SEAM))
        k += 1
        if k > n_cells:
            break
    return np.column_stack(cols[:m])


def face_prolongation(agent_id: str, port_name: str) -> Prolongation:
    return Prolongation(
        agent_id=agent_id,
        port_name=port_name,
        matrix=fourier_basis(),
        gram_V=H_SEAM * np.eye(N_SEAM),
        label=f"{M_EFF}-mode real Fourier basis on {N_SEAM} cells "
               "(full rank: the probed seam operator has no spectral cutoff)",
    )


# ---------------------------------------------------------------------------
# the flux convention -- W66's subject, inherited unchanged
# ---------------------------------------------------------------------------

FLUX_CONVENTIONS = ("entropy", "heat")


def _as_flow(q_n: np.ndarray, T_face: np.ndarray, convention: str) -> np.ndarray:
    """Turn a heat flux into THERM's declared flow variable, or decline to.

    ``entropy`` returns ``q_n / T``, the bond `ports.PORT_SPECS[THERM]` declares;
    ``heat`` returns ``q_n``, the pseudo-bond every thermal code exchanges.
    Identical to `thermal_seam._as_flow`, and re-exported rather than imported so
    the two case studies can diverge without one silently changing the other.
    """
    if convention == "entropy":
        return q_n / np.maximum(T_face, 1.0)
    if convention == "heat":
        return q_n
    raise ValueError(f"flux_convention must be one of {FLUX_CONVENTIONS}, "
                     f"got {convention!r}")


# ---------------------------------------------------------------------------
# agent 1 -- the cooling duct (fast, explicit)
# ---------------------------------------------------------------------------


def velocity_profile(u_bulk: float, ny: int = NY_DUCT,
                     h: float = H_DUCT) -> np.ndarray:
    """A 1/7-power turbulent duct profile, normalized to the declared bulk mean.

    Nearly plug, which is what a turbulent duct is, and the normalization is
    what makes ``u_bulk`` mean the mass flow rather than a centreline value --
    so the design knob is the thing a duct is actually sized for.
    """
    yc = (np.arange(ny) + 0.5) * (h / ny)
    d = np.minimum(yc, h - yc)
    p = (d / (0.5 * h)) ** (1.0 / 7.0)
    return u_bulk * p / p.mean()


def eddy_diffusivity(u_bulk: float, ny: int = NY_DUCT,
                     h: float = H_DUCT) -> np.ndarray:
    """alpha_mol + nu_t / Pr_t, with nu_t the classical mixing-length model.

    ``nu_t = kappa u_tau d (1 - 2d/h)``, ``u_tau = u sqrt(cf/2)`` and Blasius'
    ``cf = 0.079 Re^{-1/4}``.  Three textbook constants, none fitted here, and
    the wall coefficient it produces scales as ``U^0.83`` against
    Dittus-Boelter's ``U^0.8`` -- an agreement that is a check and not an input.
    """
    yc = (np.arange(ny) + 0.5) * (h / ny)
    d = np.minimum(yc, h - yc)
    re = max(u_bulk, 1e-6) * h / NU_G
    cf = 0.079 * re ** -0.25
    u_tau = u_bulk * np.sqrt(max(cf, 1e-9) / 2.0)
    return ALPHA_G + np.maximum(KAPPA * u_tau * d * (1.0 - 2.0 * d / h), 0.0) / PR_T


@dataclass
class DuctAgent:
    """Low-Mach convective transport of gas temperature along the cooling duct.

    The seam is the ``y = 0`` wall and the trace is its temperature; the far
    wall is adiabatic, the inlet is Dirichlet at `T_IN` and the outlet is
    zero-gradient.  ``u_bulk`` is the design knob and enters nowhere else.

    **Explicit, and its domain of dependence is finite and declared.**  Upwind
    advection plus a central diffusion stencil reaches one cell per sub-step, so
    ``stencil_radius = 1`` and the halo a lagged exchange needs is
    ``substeps_per_macro_step`` cells -- which is a number this agent computes
    from its own CFL condition rather than one it is told.
    """

    agent_id: str = "duct"
    dt: float = DT_DUCT
    u_bulk: float = U_DUCT
    flux_convention: str = "entropy"
    nx: int = N_SEAM
    ny: int = NY_DUCT
    #: The Courant number the internal sub-cycling runs at.  It exists so the
    #: **clock ratio can be swept at a FIXED exchange interval**, which is the
    #: control that separates the two multirate axes: lowering it refines the
    #: fast agent's own march without touching the interval the trace is stale
    #: over.  0.4 is the declared value and everything else is a sweep point.
    cfl: float = CFL
    _u: Any = field(default=None, repr=False)
    _a: Any = field(default=None, repr=False)
    _af: Any = field(default=None, repr=False)
    _T0: Any = field(default=None, repr=False)
    _scratch: Any = field(default=None, repr=False)
    _flux: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        u = velocity_profile(self.u_bulk, self.ny)
        a = eddy_diffusivity(self.u_bulk, self.ny)
        self._u = u[None, :]
        self._a = a[None, :]
        af = np.empty(self.ny + 1)
        af[1:-1] = 0.5 * (a[:-1] + a[1:])
        af[0], af[-1] = a[0], a[-1]
        self._af = af
        self._T0 = np.full((self.nx, self.ny), T_IN)
        self._scratch = np.empty((self.nx, self.ny))
        self._flux = np.zeros((self.nx, self.ny + 1))

    # -- the discretization ------------------------------------------------

    @property
    def k_wall(self) -> float:
        """The effective conductivity at the seam face, W/(m K)."""
        return float(self._af[0] * RHO_G * CP_G)

    @property
    def h_wall(self) -> float:
        """The film coefficient the first cell implies, W/(m^2 K)."""
        return self.k_wall / (0.5 * DY)

    @property
    def dt_stable(self) -> float:
        """The explicit stability limit.  Computed, never hard-coded.

        `CASE-STUDY-GUIDE` mistake 5 is hard-coding a sub-step count, measured
        at two orders of magnitude in tau the one time it was done.  This is a
        function of ``u_bulk``, which is the design knob, so hard-coding it here
        would make the sub-step count a function of a parameter nobody declared.
        """
        return self.cfl / (self._u.max() / DX
                           + 2.0 * self._a.max() * (1.0 / DX ** 2 + 1.0 / DY ** 2))

    def substeps_at(self, dt: float) -> int:
        return max(1, int(np.ceil(dt / self.dt_stable)))

    def _rhs(self, T: np.ndarray, T_wall: np.ndarray) -> np.ndarray:
        out = self._scratch
        u, a = self._u, self._a
        # upwind advection (u > 0 everywhere) and central x-diffusion; the inlet
        # is Dirichlet at T_IN and the outlet carries a zero-gradient ghost.
        right = np.empty_like(T)
        right[:-1] = T[1:]
        right[-1] = T[-1]
        out[1:] = (-u * (T[1:] - T[:-1]) / DX
                   + a * (right[1:] - 2.0 * T[1:] + T[:-1]) / DX ** 2)
        out[0] = (-u[0] * (T[0] - T_IN) / DX
                  + a[0] * (T[1] - 2.0 * T[0] + T_IN) / DX ** 2)
        # y-diffusion by face fluxes: seam wall Dirichlet, far wall adiabatic
        f = self._flux
        f[:, 1:-1] = self._af[1:-1] * (T[:, 1:] - T[:, :-1]) / DY
        f[:, 0] = self._af[0] * (T[:, 0] - T_wall) / (0.5 * DY)
        f[:, -1] = 0.0
        out += (f[:, 1:] - f[:, :-1]) / DY
        return out

    def advance(self, T: np.ndarray, dt: float,
                T_wall: np.ndarray) -> tuple[np.ndarray, int]:
        """March ``dt`` at the CFL limit.  Returns (T, sub-steps taken)."""
        t, n, h_max = 0.0, 0, self.dt_stable
        while t < dt - 1e-15:
            h = dt - t if dt - t < h_max else h_max
            T = T + h * self._rhs(T, T_wall)
            t += h
            n += 1
        return T, n

    def wall_heat_flux(self, T: np.ndarray,
                       T_wall: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(q_n into the wall [W/m^2], face temperature [K])."""
        q = self.k_wall * (T[:, 0] - T_wall) / (0.5 * DY)
        return q, 0.5 * (T[:, 0] + T_wall)

    # -- the port ----------------------------------------------------------

    def respond(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        """(seam wall temperature) -> (declared THERM flow on the seam)."""
        T_wall = np.asarray(trace, dtype=float).reshape(self.nx)
        T, _ = self.advance(self._T0, self.dt, T_wall)
        q, T_face = self.wall_heat_flux(T, T_wall)
        return _as_flow(q, T_face, self.flux_convention)

    def respond_integrated(self, port_name: str, trace: np.ndarray,
                           n_substeps: int) -> np.ndarray:
        """The same flow, AVERAGED over ``n_substeps`` of this agent's own clock.

        R9's own callable.  `respond` restarts from ``_T0`` every call, so
        calling it n times recomputes the first step n times instead of marching
        n; this marches, accumulating the agent's own quadrature of the interface
        flux.  At ``n_substeps = 1`` it equals `respond` exactly, which is what
        `conformance._test_integrated_response` checks and is what keeps this
        from being another unverifiable declaration.
        """
        n = max(1, int(n_substeps))
        T_wall = np.asarray(trace, dtype=float).reshape(self.nx)
        T = self._T0
        acc = np.zeros(self.nx)
        for _ in range(n):
            T, _ = self.advance(T, self.dt, T_wall)
            q, T_face = self.wall_heat_flux(T, T_wall)
            acc = acc + _as_flow(q, T_face, self.flux_convention)
        return acc / n

    def base_trace(self) -> np.ndarray:
        """The SEAM's base, not this agent's -- W74. See `seam_base`."""
        return np.full(self.nx, seam_base(self.u_bulk))

    def storage(self, state=None) -> float:
        """Thermal energy of the gas in the duct -- SPD by construction."""
        T = self._T0 if state is None else np.asarray(state)
        return float(RHO_G * CP_G * DX * DY * np.sum(T))

    def validity(self, state=None, cond=None) -> bool:
        """The closure's own envelope: a turbulent duct, and a real one.

        Blasius' friction law is fitted for ``4e3 < Re < 1e5`` and the eddy model
        below it describes nothing.  Declining outside that is the abstention
        this framework asks for, and it is the reason the design sweep is bounded
        rather than open.
        """
        re = self.u_bulk * H_DUCT / NU_G
        return 4.0e3 <= re <= 1.0e5


# ---------------------------------------------------------------------------
# agent 2 -- the disc (slow, implicit, EMBEDDED)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def _explicit_disc_class():
    """`ThermoStruct2D` with the implicit conduction solve removed.

    Subclassing rather than editing, for `window_ns._no_projection_class`'s
    reason and `thermal_seam._explicit_shell_class`'s: the agent is not ours to
    change, and what the composition layer may do is DECLINE TO USE a part of it
    and supply that part itself.  Only the linear solve for ``T^{n+1}`` is
    replaced, by lumped-mass forward Euler at a sub-cadence the diffusion number
    makes stable.
    """
    _C2, TS, _TH, _GR = load_solvers()

    class ExplicitDisc(TS.ThermoStruct2D):
        def step_thermal(self, T, dt, h_in, T_gas_in, h_out, T_gas_out,
                         radiate=False, T_inf=250.0):
            Kb_i, fb_i = self._robin("inner", h_in, T_gas_in)
            Kb_o, fb_o = self._robin("outer", h_out, T_gas_out)
            K = self.K_th + Kb_i + Kb_o
            f = fb_i + fb_o
            m = np.maximum(np.asarray(self.M_th.sum(axis=1)).ravel(), 1e-30)
            n = disc_substeps_at(dt)
            h = dt / n
            for _ in range(n):
                T = T + h * (f - K.dot(T)) / m
            return T

    return ExplicitDisc


def disc_substeps_at(dt: float) -> int:
    """Explicit conduction sub-steps under ``split-step``.  Computed, not fixed."""
    mat = solid_material()
    dy = T_DISC / NJ_DISC
    dt_stable = CFL * dy * dy / (2.0 * mat.diffusivity)
    return max(1, int(np.ceil(dt / dt_stable)))


@dataclass
class DiscAgent:
    """`thermostruct2d.ThermoStruct2D` on a radial section of the brake disc.

    The seam is the **inner** face (the cooling duct) and carries the agent's own
    Robin channel, ``-k dT/dn = h (T - T_gas)``.  The **outer** face is the
    friction interface: a second Robin at ``(h_pad, T_pad)`` standing in for the
    pad's heat input, with ``radiate=False`` because a contact interface does not
    radiate to ambient and because it keeps `step_thermal` affine in the trace.
    """

    agent_id: str = "disc"
    dt: float = DT_DISC
    flux_convention: str = "entropy"
    expose_elliptic: bool = False
    h_in: float = H_IN
    h_pad: float = H_PAD
    t_pad: float = T_PAD
    t_init: float = T_DISC_0
    _ts: Any = field(default=None, repr=False)
    _mesh: Any = field(default=None, repr=False)
    _T0: Any = field(default=None, repr=False)
    _face: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        _C2, TS, _TH, _GR = load_solvers()
        nodes = np.stack(np.meshgrid(np.linspace(0.0, L_R, N_SEAM + 1),
                                     np.linspace(0.0, T_DISC, NJ_DISC + 1),
                                     indexing="ij"), -1)
        self._mesh = TS.ShellMesh(nodes)
        cls = _explicit_disc_class() if self.expose_elliptic else TS.ThermoStruct2D
        self._ts = cls(self._mesh, solid_material())
        self._T0 = np.full(self._mesh.n_nodes, self.t_init)
        a, b, _L = self._ts._face("inner")
        self._face = (a, b)

    def face_T(self, T: np.ndarray) -> np.ndarray:
        a, b = self._face
        return 0.5 * (T[a] + T[b])

    def step(self, T: np.ndarray, dt: float, trace: np.ndarray) -> np.ndarray:
        pad = np.full(N_SEAM, self.t_pad)
        return self._ts.step_thermal(T, dt, self.h_in,
                                     np.asarray(trace, float).reshape(N_SEAM),
                                     self.h_pad, pad, radiate=False)

    # -- the port ----------------------------------------------------------

    def respond(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        """(duct gas temperature on the seam) -> (declared THERM flow)."""
        T_gas = np.asarray(trace, dtype=float).reshape(N_SEAM)
        T = self.step(self._T0, self.dt, T_gas)
        T_face = self.face_T(T)
        q = self.h_in * (T_gas - T_face)               # W/m^2, into the disc
        return _as_flow(q, 0.5 * (T_gas + T_face), self.flux_convention)

    def respond_integrated(self, port_name: str, trace: np.ndarray,
                           n_substeps: int) -> np.ndarray:
        """The same flow, averaged over ``n_substeps`` of the disc's own clock.

        The disc IS the slow agent, so at the native clocks its quadrature is a
        single step and this reduces to `respond`.  It exists anyway, because R9
        requires *both* sides to be able to state their integral and a rule only
        the fast side has to satisfy is not a matching condition.
        """
        n = max(1, int(n_substeps))
        T_gas = np.asarray(trace, dtype=float).reshape(N_SEAM)
        T = self._T0
        acc = np.zeros(N_SEAM)
        for _ in range(n):
            T = self.step(T, self.dt, T_gas)
            T_face = self.face_T(T)
            q = self.h_in * (T_gas - T_face)
            acc = acc + _as_flow(q, 0.5 * (T_gas + T_face), self.flux_convention)
        return acc / n

    def base_trace(self) -> np.ndarray:
        """The SEAM's base, not this agent's -- W74. See `seam_base`."""
        return np.full(N_SEAM, seam_base(t_disc=self.t_init))

    def storage(self, state=None) -> float:
        """``1/2 T^T M T`` -- the conduction operator's own Lyapunov functional."""
        T = self._T0 if state is None else np.asarray(state)
        return float(0.5 * T @ self._ts.M_th.dot(T))

    def validity(self, state=None, cond=None) -> bool:
        """Cast iron below its own limits: the model has no phase change in it."""
        if state is None:
            return True
        return float(np.max(np.asarray(state))) < 1400.0


# ---------------------------------------------------------------------------
# the interface condition -- CS-10's lesson, at a field-to-field THERM seam
# ---------------------------------------------------------------------------


def solve_interface(duct: DuctAgent, T_gas: np.ndarray,
                    disc: DiscAgent, T_disc: np.ndarray) -> np.ndarray:
    """The seam's own condition, solved in the seam's own variable.

    **This is CS-10's finding, applied where it was predicted to generalize.**
    That case study found that reading a lumped agent's constitutive law as an
    UPDATE diverges at the first macro-step, and that what is well posed is the
    port's own condition written in the port's own variable and solved there.
    The same reading applies at a field-to-field `THERM` seam and is cheaper
    still, because on the states at hand both responses are AFFINE in the trace:

        duct  q_duct = k_eff (T_gas[:,0] - lambda) / (dy/2)      into the wall
        disc  q_disc = h_in (lambda - T_face)                    into the disc

    and ``q_duct = q_disc`` has the closed form below.  No Newton iteration, no
    second field evaluation, and the residual is zero to floating point rather
    than to a tolerance.

    **The root does not depend on the flux convention and that is worth saying.**
    ``entropy`` divides both halves by the same face temperature, so it scales
    the residual and moves no root.  W66's convention question is about what the
    bond IS and about what `interface_power` measures; it is not about where the
    interface sits.
    """
    a = duct.k_wall / (0.5 * DY)
    return (a * T_gas[:, 0] + disc.h_in * disc.face_T(T_disc)) / (a + disc.h_in)


# ---------------------------------------------------------------------------
# the composed march
# ---------------------------------------------------------------------------


@dataclass
class BrakeMarch:
    """What one march returns.  Everything is per macro-step and nothing is
    reduced, because the horizon law reads J at every N and a march that stored
    only its endpoint would have to be run once per horizon."""

    t: np.ndarray                 # s, the time at the end of each interval
    trace: np.ndarray             # [n, N_SEAM], the seam trace held over it
    face_T: np.ndarray            # K, the seam-face disc temperature
    q_seam: np.ndarray            # W/m^2, the seam heat flux (into the disc)
    power: np.ndarray             # the interface power over the interval
    substeps: np.ndarray          # duct sub-steps per interval
    T_gas: Any = None
    T_disc: Any = None
    wall_s: float = 0.0

    def objective(self, n: int, frac: float = 0.25) -> float:
        """The settled seam temperature over the last ``frac`` of ``n`` steps.

        CS-10's convention exactly.  ``n`` is the HORIZON, and reading it off a
        single march is what makes the horizon law cost one march per design
        point rather than one per (design point, horizon) pair.
        """
        n = int(n)
        k = max(1, int(round(frac * n)))
        return float(np.mean(self.face_T[n - k:n]))


class BrakeRollout:
    """The composed march, with the exchange cadence as the one free variable.

    One code path serves the referent and every multirate column:

      ``exchange = DT_DUCT``   the **single-rate referent**.  Both agents step at
                               the duct's own clock and the interface condition
                               is re-solved every step, so the trace is stale for
                               one duct step rather than for 500 of them.
      ``exchange = DT_DISC``   the multirate column at the native 500:1.
      ``resolve_every = k``    the interface condition re-solved once every k
                               intervals and held in between -- the lag axis
                               `thermal_strain` and `ground_effect` both use, kept
                               separate from the interval so the two can be told
                               apart.

    The disc always takes exactly one of its own steps per exchange interval and
    the duct always sub-cycles its own; that is what makes ``exchange`` the
    exchange interval rather than a re-labelled time step.
    """

    def __init__(self, u_bulk: float = U_DUCT, exchange: float = DT_DISC,
                 resolve_every: int = 1, flux_convention: str = "entropy",
                 expose_elliptic: bool = False, t_init: float = T_DISC_0,
                 waveform: int = 1, cfl: float = CFL) -> None:
        if waveform not in (1, 2):
            raise ValueError("waveform is 1 (a held trace) or 2 (a linear one)")
        self.duct = DuctAgent(dt=DT_DUCT, u_bulk=u_bulk, cfl=cfl,
                              flux_convention=flux_convention)
        self.disc = DiscAgent(dt=DT_DISC, flux_convention=flux_convention,
                              expose_elliptic=expose_elliptic, t_init=t_init)
        self.exchange = float(exchange)
        self.resolve_every = max(1, int(resolve_every))
        self.waveform = waveform
        self.u_bulk = u_bulk

    # -- one interval -------------------------------------------------------

    def interval(self, T_gas, T_disc, lam, lam_next=None):
        """March both agents over one exchange interval at the given trace.

        ``lam_next`` is W17's ``W > 1``: when supplied the duct sees a trace
        that varies LINEARLY across the interval instead of one held constant,
        which is admissible here because R3's `bc_time_varying` holds on both
        agents.  The disc still sees the interval mean, because backward Euler
        over the interval integrates the Robin datum once and a linear datum's
        integral is its midpoint value -- so the waveform's effect is on the
        agent whose clock the interval is many of, which is the fast one, which
        is the side the lag is about.
        """
        if lam_next is None or self.waveform == 1:
            T_gas, n = self.duct.advance(T_gas, self.exchange, lam)
            lam_disc = lam
        else:
            n, t, h_max = 0, 0.0, self.duct.dt_stable
            while t < self.exchange - 1e-15:
                h = self.exchange - t if self.exchange - t < h_max else h_max
                s = (t + 0.5 * h) / self.exchange
                T_gas = T_gas + h * self.duct._rhs(
                    T_gas, lam + s * (lam_next - lam))
                t += h
                n += 1
            lam_disc = 0.5 * (lam + lam_next)
        T_disc = self.disc.step(T_disc, self.exchange, lam_disc)
        return T_gas, T_disc, n

    # -- the march ----------------------------------------------------------

    def run(self, steps: int, T_gas=None, T_disc=None,
            progress: Callable[[int, float], None] | None = None) -> BrakeMarch:
        import time
        duct, disc = self.duct, self.disc
        T_gas = duct._T0.copy() if T_gas is None else np.array(T_gas, float)
        T_disc = disc._T0.copy() if T_disc is None else np.array(T_disc, float)
        lam = solve_interface(duct, T_gas, disc, T_disc)
        lam_prev = lam.copy()

        ts, traces, faces, qs, powers, subs = [], [], [], [], [], []
        t0 = time.perf_counter()
        for s in range(steps):
            if s % self.resolve_every == 0:
                lam_prev, lam = lam, solve_interface(duct, T_gas, disc, T_disc)
            nxt = lam + (lam - lam_prev) if self.waveform == 2 else None
            T_gas, T_disc, n = self.interval(T_gas, T_disc, lam, nxt)
            face = disc.face_T(T_disc)
            q = disc.h_in * (lam - face)
            ts.append((s + 1) * self.exchange)
            traces.append(lam.copy())
            faces.append(float(face.mean()))
            qs.append(float(q.mean()))
            powers.append(float(np.sum(lam * q) * H_SEAM))
            subs.append(n)
            if not np.isfinite(T_gas).all() or not np.isfinite(T_disc).all():
                raise RuntimeError(
                    f"the brake rollout is not finite at macro-step {s}; raised "
                    "where it happened so the NaN is not carried into every "
                    "number after it")
            if progress is not None:
                progress(s, time.perf_counter() - t0)
        return BrakeMarch(
            t=np.array(ts), trace=np.array(traces), face_T=np.array(faces),
            q_seam=np.array(qs), power=np.array(powers),
            substeps=np.array(subs), T_gas=T_gas, T_disc=T_disc,
            wall_s=time.perf_counter() - t0)


def referent_rollout(**kw) -> BrakeRollout:
    """The **single-rate** column the bound is graded against.

    Both agents exchange at the duct's own clock, so the interface condition is
    re-solved 500 times inside what the multirate column spends holding one
    trace.  It is the same class with one argument changed, which is what makes
    the zero-lag control exact rather than approximate.
    """
    kw.setdefault("exchange", DT_DUCT)
    return BrakeRollout(**kw)


def composed_rollout(**kw) -> BrakeRollout:
    """The multirate column at the native 500:1."""
    kw.setdefault("exchange", DT_DISC)
    return BrakeRollout(**kw)


#: Macro-steps of the referent used to settle the duct before anything is
#: measured.  The duct's own residence time is ``L_R / U = 2 ms``, so 0.02 s is
#: ten flow-throughs and the field is stationary against a fixed wall to well
#: below every defect quoted; the disc has barely moved in that time, which is
#: the point -- the spin-up settles the FAST agent and leaves the slow one at
#: its release state.
N_SPIN = 200


@lru_cache(maxsize=8)
def settled_state(u_bulk: float = U_DUCT, t_init: float = T_DISC_0):
    """The duct field at its inlet condition against a disc at its release
    temperature.  Every column starts here, which is what makes a comparison
    between columns a comparison of the SCHEME rather than of the state."""
    r = referent_rollout(u_bulk=u_bulk, t_init=t_init)
    m = r.run(N_SPIN)
    return m.T_gas.copy(), m.T_disc.copy(), m


# ---------------------------------------------------------------------------
# the records
# ---------------------------------------------------------------------------

#: (T, q_n/T) in K and W/(m^2 K): s_e s_f = s_P with s_P in W/m^2.
THERM_SCALES = {"temperature": T_PAD, "entropy_flux": 1.0e5 / T_PAD,
                "power_area": 1.0e5}


def duct_capabilities(expert: DuctAgent) -> ExpertCapabilities:
    return ExpertCapabilities(
        expert_id=expert.agent_id,
        ports=[port_decl(
            name="wall:THERM", port_type=PortType.THERM,
            geometry=f"seam wall of the cooling duct, {N_SEAM} cells",
            direction=Direction.BIDIRECTIONAL,
            nondim=dict(THERM_SCALES),
            effective_resolution=M_EFF,
            motion_class=MotionClass.STATIC,
            # The response is the entropy flux q_n/T -- THERM's declared FLOW --
            # so the imposed trace is the temperature, the effort. Under
            # flux_convention="heat" the callable returns the pseudo-bond and
            # this declaration becomes a lie nothing numerical can catch; that
            # is W66's measurement and not a bug.
            response_half=ResponseHalf.FLOW,
            prolongation=face_prolongation(expert.agent_id, "wall:THERM"),
            note="Dirichlet wall temperature in, wall heat flux out; the duct's "
                 "own near-wall gradient sets the film coefficient",
        )],
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=True,
        # An explicit march with no solve of any kind. The y-diffusion is a
        # face-flux difference, not an implicit step.
        elliptic_subsolve=EllipticSubsolve.NONE,
        time_discretization=TimeDiscretization.EXPLICIT,
        stencil_radius=1,                       # upwind + central: one cell
        substeps_per_macro_step=expert.substeps_at(expert.dt),
        # R7's finite-difference branch, and the declaration that selects it is
        # this one plus `deterministic` plus `reproducibility_floor`: *"none but
        # deterministic and smooth -> finite differences with epsilon set by the
        # reproducibility floor"*. There is no `FD` member to declare -- see the
        # "How the gradient is taken" paragraph in the module docstring.
        differentiable=Differentiable.NONE,
        dt_native=expert.dt,
        storage=expert.storage,
        equivariances=(),
        validity=expert.validity,
        # An algebraic-turbulence closure INSIDE a convection-diffusion problem
        # is not a different continuum problem, so it declares the family it
        # closes -- `wind_farm.actuator_disk`'s comment, applied.
        governing_family="convection-diffusion-2d",
        lambda_ref="the same duct discretization at half the cell size and half "
                   "the step; this agent is its own reference and its "
                   "infidelity is zero by construction",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash="brake-duct/1-upwind-central-euler",
        boundary_response=expert.respond,
        boundary_response_integrated=expert.respond_integrated,
        probe_base=lambda _port, _e=expert: _e.base_trace(),
        reproducibility_floor=float(np.finfo(float).eps),
        deterministic=True,
        note="written for this case study, not a build-repo donor. Graded "
             "against a pure-advection step and the erf half-space solution by "
             "tests/test_tier28_brake_thermal.py",
    )


def disc_capabilities(expert: DiscAgent) -> ExpertCapabilities:
    exposed = expert.expose_elliptic
    return ExpertCapabilities(
        expert_id=expert.agent_id,
        ports=[port_decl(
            name="duct:THERM", port_type=PortType.THERM,
            geometry=f"duct-side face of the disc section, {N_SEAM} line elements",
            direction=Direction.BIDIRECTIONAL,
            nondim=dict(THERM_SCALES),
            effective_resolution=M_EFF,
            motion_class=MotionClass.STATIC,
            response_half=ResponseHalf.FLOW,
            prolongation=face_prolongation(expert.agent_id, "duct:THERM"),
            note="the agent's own Robin surface term, -k dT/dn = h (T - T_gas)",
        )],
        bc_channel=BCChannel.ROBIN,
        bc_time_varying=True,
        # Backward Euler solves (M/dt + K + K_robin) T = rhs over the whole
        # section: EMBEDDED in R10's exact sense, and the ladder's CS-11 row
        # asks for exactly that. L2/R10 refuses it and the refusal is correct.
        elliptic_subsolve=(EllipticSubsolve.EXPOSED if exposed
                           else EllipticSubsolve.EMBEDDED),
        time_discretization=(TimeDiscretization.EXPLICIT if exposed
                             else TimeDiscretization.IMPLICIT),
        stencil_radius=1,                       # Q1 elements, one ring
        substeps_per_macro_step=disc_substeps_at(expert.dt) if exposed else 1,
        differentiable=Differentiable.NONE,      # R7's FD branch, as the duct
        dt_native=expert.dt,
        storage=expert.storage,
        equivariances=(),
        validity=expert.validity,
        governing_family="thermoelastic-shell-2d",
        lambda_ref="thermostruct2d.ThermoStruct2D, same section and "
                   "discretization; the real solver is its own reference",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash=f"thermostruct2d/1-{'explicit' if exposed else 'backward-euler'}",
        boundary_response=expert.respond,
        boundary_response_integrated=expert.respond_integrated,
        probe_base=lambda _port, _e=expert: _e.base_trace(),
        reproducibility_floor=float(np.finfo(float).eps),
        deterministic=True,
        note="solvers/thermostruct2d.py, build repo, unmodified; conduction "
             "graded against the erf slab by that repo's own M1 suite. Cast "
             "iron is passed as a SolidMaterial, which is the expert's own "
             "parameterization and not a change to it",
    )


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------

#: **W131, measured 2026-09-03** by `scripts/w131_brake_thermal.py`.
#:
#: The first sigma on this vault's record that is a FUNCTION of the exchange
#: interval rather than a value at one of them.  The fields below carry the
#: value at the native 500:1 exchange; the law it is one point of, and the
#: constants that make it evaluable at another interval, are in
#: `MULTIRATE_SIGMA_LAW` and are what W90 asked for.
#:
#:   tau   = 0 for both agents. Each is its own reference -- the disc is the
#:           real build-repo solver and the duct is graded against its own
#:           refinement -- so infidelity is zero by construction, and declaring
#:           it is what lets a later swap to a learned expert MEASURE that
#:           expert instead of emitting UNDEFINED.
#:   sigma = the transmission defect in interface power over one exchange
#:           interval at the native clocks, against the single-rate referent.
#:   sigma_lag = the trace drift over that same interval, as a number, so
#:           `multiphysics.check_sigma_lag` can falsify it against the lag a
#:           run derives from its own consecutive traces.
#:
#: Filled in by the driver's `constants` stage; left as placeholders here would
#: be W56 on purpose, so the values below are the measured ones and the driver
#: re-derives them on every run.
MEASURED_W131 = MeasuredConstants(
    tau=0.0,
    sigma=None,
    sigma_lag=None,
    probe_state=(f"brake duct, U={U_DUCT} m/s, T_in={T_IN} K, disc released at "
                 f"{T_DISC_0} K against a pad at {T_PAD} K"),
    scheme="split-step",
    depth=0,
    source="scripts/w131_brake_thermal.py, stage `sigma_law`; interface power norm",
)

#: **The bound W90 asked for.**  Filled by the driver's `sigma_law` stage and
#: quoted here so a reader of the case file gets the law rather than a pointer.
#:
#:     sigma(DT) <= s_seam * lambda_dot * DT  +  C2 * (lambda_dot * DT)^2
#:
#: with ``s_seam`` the seam's own slope of sigma in a uniform lag (W86's
#: transferable quantity, measured once by a probe), ``lambda_dot`` the trace's
#: rate of change (a property of the RUN, which nothing at compile time can
#: know), and ``C2`` the curvature interface power's bilinearity forces.
#:
#: The validity statement that goes with it, because a bound without one is the
#: thing this framework exists to refuse:
#:
#:   - it is measured over exchange intervals from `DT_DUCT` to 2000 `DT_DUCT`,
#:     i.e. over a clock ratio from 1 to 2000;
#:   - it is measured at ONE seam, one pair of experts, one operating point;
#:   - ``lambda_dot`` is bounded over a run and is not constant along it, so the
#:     bound is evaluated with the run's own max and is conservative by that
#:     much;
#:   - and it bounds the LAG term only. R9's flux-transient term is separate,
#:     smaller, and is the only place the clock ratio enters.
MULTIRATE_SIGMA_LAW: dict[str, Any] = {
    "form": "sigma(DT) = s_seam * lambda_dot * DT + C2 * (lambda_dot * DT)**2",
    "s_seam": None,
    "C2": None,
    "lambda_dot_max": None,
    "validated_over_ratio": (1, 2000),
    "source": "scripts/w131_brake_thermal.py, stage sigma_law",
}


def make_experts(mode: str = "as-built", flux_convention: str = "entropy",
                 clocks: str = "native", u_bulk: float = U_DUCT) -> dict[str, Any]:
    if clocks not in ("native", "matched"):
        raise ValueError(f"clocks must be 'native' or 'matched', got {clocks!r}")
    # `matched` brings the DISC down to the duct's clock, never the duct up to
    # the disc's: backward Euler is unconditionally stable so a 0.1 ms disc step
    # is merely a smaller step, while an explicit duct at 50 ms is 2442 unstable
    # ones. The asymmetry is R4's, and it is why the referent exists at all.
    dt_disc = DT_DISC if clocks == "native" else DT_DUCT
    return {
        "duct": DuctAgent(dt=DT_DUCT, u_bulk=u_bulk,
                          flux_convention=flux_convention),
        "disc": DiscAgent(dt=dt_disc, flux_convention=flux_convention,
                          expose_elliptic=(mode == "split-step")),
    }


def build(mode: str = "as-built", flux_convention: str = "entropy",
          clocks: str = "native", experts: dict[str, Any] | None = None,
          measured="default", flux_matching="default") -> tuple[CaseGraph, dict[str, Any]]:
    """The two-agent multirate graph, plus the experts backing it.

    ``mode``             ``as-built`` (the disc's conduction solve is EMBEDDED,
                         which is what backward Euler over a cross-section IS,
                         and `L2/R10` refuses) or ``split-step`` (exposed, as
                         `window_ns` and `thermal_seam` both do it).
    ``flux_convention``  ``entropy`` (THERM's declared flow) or ``heat`` (the
                         pseudo-bond every thermal code exchanges) -- W66.
    ``clocks``           ``native`` (500:1, E4 fails honestly and R9 decides the
                         graph) or ``matched`` (the disc brought down to the
                         duct's clock -- the single-rate referent, as a graph).

    ``flux_matching`` defaults to `TIME_INTEGRATED` at the native clocks and
    `POINTWISE` at the matched ones, because that is what each case is rather
    than a preference: across one clock the two coincide exactly, and across
    500:1 R9 requires the integral.  Pass `POINTWISE` at the native clocks to
    see the refusal R9 has issued to every multirate graph since this compiler
    existed.
    """
    if mode not in ("as-built", "split-step"):
        raise ValueError(mode)
    if flux_convention not in FLUX_CONVENTIONS:
        raise ValueError(flux_convention)
    if measured == "default":
        measured = MEASURED_W131
    if flux_matching == "default":
        flux_matching = (FluxMatching.TIME_INTEGRATED if clocks == "native"
                         else FluxMatching.POINTWISE)
    experts = experts or make_experts(mode, flux_convention, clocks)

    agents = [
        Agent("duct", duct_capabilities(experts["duct"]),
              domain=f"cooling duct, {L_R} x {H_DUCT} m"),
        Agent("disc", disc_capabilities(experts["disc"]),
              domain=f"disc section, {L_R} m x {T_DISC * 1e3:.1f} mm",
              role="structure"),
    ]
    connections = [Connection(
        seam_id="cht",
        a=("duct", "wall:THERM"),
        b=("disc", "duct:THERM"),
        port_type=PortType.THERM,
        geometrically_coincident=True,
        derive_space=True,
        # n_0(Gamma) = 0, per CASE-STUDY-GUIDE's table row for conjugate heat
        # transfer: a uniform temperature shift produces a uniform flux change,
        # so no direction of the trace space is invisible to the response.
        expected_null_dim=0,
        note="conjugate heat transfer across a brake disc's cooling face: a "
             "convecting duct against a conducting disc, 500:1 in clock",
    )]

    return CaseGraph(
        name=f"brake-thermal-{mode}-{flux_convention}-{clocks}",
        agents=agents,
        connections=connections,
        # Non-overlapping: the gas and the iron are different MATERIALS sharing
        # a surface, not a region. No cell belongs to both, so there is no
        # partition of unity -- the substructuring branch, where W57 says no cut
        # criterion exists and where master-error-bound 4 rather than 4.1 is the
        # sigma branch.
        decomposition=Decomposition.NON_OVERLAPPING,
        macro_dt=max(a.capabilities.dt_native for a in agents),
        flux_matching=flux_matching,
        measured=measured,
        note="CS-11, the twelfth real case study: a brake disc against its "
             "cooling duct at the 500:1 the physics gives. The question only "
             "it can answer is what BOUNDS the multirate lag defect",
    ), experts
