"""CS-10: a MOVING interface, and the first design parameter.  (W30, W97, F5)

A 2-D wing section over a moving floor at ride height `h`.  The flow expert
produces the field; the integrated aerodynamic load `L(h)` crosses a
**field-to-lumped `MECH` seam** to a lumped suspension expert whose whole
content is two lines of algebra with zero fitted parameters,

    k (h0 - h) = L(h),

and whose solution **moves the interface** every macro-step.  Aero load changes
ride height, ride height changes aero load: [[f1-pathmap-and-end-goal]] §1.1
names this as the coupling that *is* the physics, and every interface in this
vault before it declares ``motion_class=STATIC``.

Three things only this case study can deliver, and each is a section below.

1. **`InterfaceMotion` becomes MEASURED.**  `holes.INTERFACE_MOTION` names three
   measurements -- ``reprobe_count``, ``operator_drift``, and the unaccounted
   power fraction at the sweeping seam -- and has carried none of them since it
   was written.  W30 emitted the drift on *static* runs (2.4e-4 to 3.5e-3
   relative over K = 10 macro-steps, `tier0-measurements` §2.4) precisely so
   that a moving one would have something to be compared against.  This is that
   comparison.  L2 still **refuses** a non-static port and nothing here changes
   that: the refusal is correct and the measurement is what it was waiting for.

2. **W97 is closed or admitted permanently.**  A field-to-lumped `MECH` seam is
   one-sided by the cell Reynolds number, structurally: §2.2's normative effort
   is ``nu dw/dn`` and a lumped body's traction carries no ``nu``, so
   ``||S_lumped|| / ||S_fluid|| ~ Re_h`` and `SubstitutionCertificate.blind` is
   True over the whole admissible ``beta_min`` axis.  The candidate repair is
   §4.1's **conservative co-normal** ``nu dw/dn - (u.n) w``, which **W47 scoped
   out at fluid-fluid seams because the advective term cancels there** -- the two
   sides share a ring cell and carry opposite normals.  At a wing seam there is
   no second fluid side, so it does not cancel, and the term it adds is O(w^2)
   -- the same order as the lumped traction.  ``flux_mode`` runs all three forms
   on identical arithmetic so the ratio is a measurement rather than an argument.

3. **`h0` is a DESIGN PARAMETER and not a state**, which is what makes this F5's
   honest test.  `d(downforce)/dh0` is taken by reverse-mode autograd through
   every sub-exchange of every window's local solve, through the blend, through
   the global Leray projection, and through the spring at both ends -- the same
   tape `wind_farm_design` built for PoC 1a, pointed at a knob rather than at a
   layout -- and checked against a central finite difference swept over eight
   decades of step.

What is reused, and what is new
-------------------------------

  the agent      `wind_farm_design.torch_solver_class()` -- `reference.WindowNS`
                 with its elliptic part removed (R10's own prescription), its
                 stencils out-of-place, and its tape left attached.  **It is
                 shape-generic**: `_lam` is only read by `_project`, which the
                 exposed class overrides with the identity, so the same class
                 runs a square window and the rectangular monolith.
  the assembly   `assembly.ProjectedAssembly` -- the partition-of-unity blend
                 AND one global spectral Leray projection on the assembled
                 field, once per exchange.  CS-8's advice, and the arrangement
                 `L2/R10` has prescribed since Tier 0.
  the referent   the SAME solver on the undivided domain with the suspension
                 tightly coupled.  So at ``n_col = n_row = 1`` the composed step
                 is the referent step **bitwise**, which is the control CS-7 §2
                 says a scaling study is not worth running without.

New here: the wing, the spring, and the moving seam.

The wing, stated as the model it is
------------------------------------

A **porous inclined plate**: `disk.ActuatorDisk`'s quadratic momentum sink with
the disk's axis replaced by a plate normal.  Per chordwise station,

    w_k  = mean of the two sides' velocity . n  -  (plate velocity) . n
    F_k  = 1/2 rho C_N (c/S) |w_k| w_k n         (on the plate; -F_k on the fluid)

with ``C_N = 2`` the flat-plate normal-force coefficient -- numerically the same
as the disk's ``C_T' = 4a/(1-a)`` at ``a = 1/3``, which is not a coincidence but
is also not an argument.  It is **not thin-aerofoil theory**: the normal force
goes as ``sin^2 alpha`` rather than ``sin alpha``, there is no Kutta condition
and no bound circulation.  What it does have is *thickness* -- it blocks the gap
-- which is the mechanism ground effect actually runs on, and a load that is an
**exact discrete momentum exchange** rather than a model: the force on the plate
is minus the integral of the force on the fluid, to floating point, on any
lattice.  Nothing about the downforce is prescribed; ``w_k`` is read off the
solver's own field.

**The two sides are averaged and that is load-bearing.**  A station sampled
inside its own smearing reads its own induction, and an actuator line that does
so drives itself.  The self-induced velocity of a sheet is antisymmetric across
it, so the mean of the two sides is the external flow at leading order.  The
offset is `D_OFFSET` cells, comfortably outside `SIGMA_N`.

The floor is a ROLLING ROAD, and that is a scope statement
-----------------------------------------------------------

The bottom row is pinned at ``(U_INF, 0)`` -- a wall moving with the stream, so
there is no floor boundary layer at all.  That is why Formula 1 uses one, and it
is also a limit on what this case study can see: the ground effect here is the
**inviscid blockage/venturi branch**, and the viscous gap choke that ends the
real downforce curve is not in this model.  Whether `L(h)` turns over anyway is
measured in `scripts/w127_ground_effect.py` and not assumed either way.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Sequence

import numpy as np

from ..assembly import ConstraintProjection, GridPartitionOfUnity, ProjectedAssembly
from ..capability import (
    BCChannel,
    ClaimType,
    Differentiable,
    Direction,
    EllipticSubsolve,
    ExpertCapabilities,
    MotionClass,
    TimeDiscretization,
    port_decl,
)
from ..graph import (
    Agent,
    CaseGraph,
    Connection,
    Decomposition,
    GlobalField,
    MeasuredConstants,
)
from ..ports import PortType, ResponseHalf
from ..transfer import Prolongation
from .window_ns import load_reference
from .wind_farm_design import TORCH_DTYPE, _place, _TapeBackend, torch, torch_solver_class

__all__ = [
    "DX", "NW", "N_COL", "N_ROW", "HALO", "STRIDE", "RAMP", "U_INF", "NU",
    "MACRO_DT", "EXCHANGES", "CHORD", "ALPHA", "C_N", "X_LE", "H0_REF",
    "K_SPRING", "N_STATION", "D_OFFSET", "SIGMA_N", "FLUX_MODES",
    "GroundTiling", "DEFAULT_TILING", "SINGLE_TILING",
    "Wing", "Suspension", "FlowWindow",
    "flow_capabilities", "suspension_capabilities", "build",
    "GroundRollout", "MarchResult", "referent_rollout", "composed_rollout",
    "settled_field", "N_SPIN", "H_START", "H_FLOOR", "H_CEILING",
    "solver_for", "project_assembled", "divergence_rms", "projected_assembly",
    "MEASURED_STATIC", "MEASURED_MOVING",
]


# ---------------------------------------------------------------------------
# the geometry
# ---------------------------------------------------------------------------

#: The cell.  64 per unit length, and the chord is half a unit, so 32 cells on
#: the plate and 5 to 26 in the gap over the ride heights this study sweeps.
DX = 1.0 / 64.0

#: One window, in cells.  Square, because `WindowNS`'s Poisson symbol is -- and
#: the exposed class never builds one, which is why the MONOLITH may be
#: rectangular with the same class (see `solver_for`).
NW = 80

N_COL, N_ROW = 3, 2

#: Overlap in cells.  `required_halo` is ``stencil_radius * substeps`` = 2 x 4 =
#: 8 over one macro-step, and 16 leaves a factor of two.  Declared rather than
#: tuned: it is the quantity `_halo_rule` checks.
HALO = 16
STRIDE = NW - HALO                    # 64 cells = 1.0 length unit
RAMP = 8

NX = (N_COL - 1) * STRIDE + NW        # 208 cells = 3.25
NY = (N_ROW - 1) * STRIDE + NW        # 144 cells = 2.25

U_INF = 1.0

#: Cell Reynolds number ``DX U / NU`` = 3.906, just inside the 4 that
#: `reference.WindowNS`'s own docstring names as its skew-symmetric operating
#: point.  Chord Reynolds number is 125, which is low and is what a 208x144 grid
#: buys; the same trade `wake_array` makes at Re_D = 255.
NU = 4.0e-3

#: The macro-step, and the number of composition-layer exchanges inside it.
#: `EXCHANGES` is `substeps_per_macro_step`: R10b requires the composition layer
#: to apply the exposed elliptic part at the agent's OWN sub-step cadence, and
#: `MACRO_DT / EXCHANGES` = 0.003125 is one `WindowNS` sub-step at
#: ``cfl = 0.4`` up to ``u_max = 2.0``.  Hard-coding a cadence that does not
#: match is `CASE-STUDY-GUIDE` mistake 5, measured at two orders in `tau`, so
#: `GroundRollout` ASSERTS the agent took exactly one sub-step per exchange.
MACRO_DT = 0.0125
EXCHANGES = 4

#: Cells of inlet and TOP held at the freestream.  The BOTTOM is the moving
#: floor and is pinned one cell only -- an 8-cell bottom band would swallow the
#: wing at every ride height this study is about.
BAND = 8

#: Downstream padding for the spectral projection, in cells.  Without it the
#: FFT's periodic image puts the outlet next to the inlet and the wing's
#: momentum sink reaches round the box.
PAD_CELLS = 32


# ---------------------------------------------------------------------------
# the wing and the suspension
# ---------------------------------------------------------------------------

CHORD = 0.5                            # 32 cells
ALPHA = math.radians(20.0)             # plate rises downstream -> downforce
#: The plate's normal-force coefficient.  ``C_N = 2`` is the textbook flat plate
#: and gives a plate this grid barely sees -- measured, `L` rises 8.5% from
#: ``h = 0.8`` to ``h = 0.09`` and the feedback ratio is 0.02.  20 is a plate
#: that actually blocks the gap: `L` rises 33% over the same sweep and
#: ``u_max`` reaches 1.35.  It is the one number here chosen to make the
#: coupling measurable rather than derived, it is declared as such, and it is
#: bounded above by the explicit stability of its own quadratic sink --
#: ``dt C_N |w| / (sigma_n dx) = 0.8`` at ``C_N = 20`` and 1 at 25.  It also
#: sets the aerodynamic damping the interface equation is solved against:
#: ``C_d = dL/d(plate velocity) = C_N |w| cos^2(alpha) c``, measured at 3.0
#: against a restoring stiffness of ``k + |dL/dh| = 2.7``, so the ride height
#: relaxes on a timescale of ``C_d / (k + |L'|) = 1.1`` -- ninety macro-steps,
#: which is why the coupled marches here are 240 long and not 40.  It is also
#: what makes the naive staggered update diverge, and for a reason that is NOT
#: the frozen-field stiffness: moving a massless plate the whole way to its new
#: equilibrium inside one macro-step implies an interface velocity of order
#: ``17 U_inf``, and ``C_d`` times that is a load three orders above the one the
#: spring was balancing.  The port's flow half IS that velocity, and a lumped
#: partner with no mass places no bound on it.
C_N = 20.0
X_LE = 88 * DX                         # leading edge, 1.375

#: Chordwise stations.  One per cell over 32 cells, plus the ends.
N_STATION = 33

#: Where each station reads the flow, in cells either side along the plate
#: normal, and the Gaussian half-width of the stamped force.
D_OFFSET = 2.5
SIGMA_N = 1.5

#: The design parameter's reference value, and the spring rate.  Both are
#: DECLARED constants of the suspension expert rather than fitted ones, and they
#: were chosen from the measured `L(h)` curve for one property: the loaded ride
#: height sits at ``h* = 0.113``, where the aero stiffness is
#: ``|dL/dh| = 0.324`` against the spring's ``0.68``, so the **feedback ratio
#: ``|dL/dh| / k`` is 0.48**.  That is the number the lagged split contracts at
#: per macro-step, and it is deliberately close enough to 1 that raising ``h0``
#: -- which softens the platform, because the equilibrium moves out along a
#: flattening curve -- or lowering ``k`` walks the split across its own
#: stability boundary while the tightly-coupled referent still has a solution.
#: A case study whose feedback ratio is 0.02 has a moving interface and no
#: coupling to measure.
H0_REF = 0.32
K_SPRING = 2.5

#: The ride height a coupled march is RELEASED from, and the envelope outside
#: which the algebraic suspension is not a model of anything.  `H_FLOOR` is
#: `Suspension.validity`'s own bound -- the plate's sampling offset plus two
#: cells -- and `H_CEILING` is the top of the sweep.
#:
#: **`H_START` must be BELOW `H0_REF`**, and that is the suspension's own
#: `validity` predicate rather than a convenience: above ``h0`` a linear spring
#: is in TENSION and pulls the plate down alongside the aerodynamic load, which
#: is a state a two-line compression model was not written for.  Released from
#: 0.45 against ``h0 = 0.32`` the predicate declines at macro-step 0 and the
#: first load comes back NEGATIVE; that is the expert refusing, correctly, and
#: it is why the release height is 0.30.
H_START = 0.30
H_FLOOR = (D_OFFSET + 2.0) * DX
H_CEILING = 1.0

MECH_SCALES = {"stress": U_INF**2, "velocity": U_INF, "power_area": U_INF**3}

#: The three efforts §2.2, §4.1 and the momentum exchange itself define at this
#: seam.  W97 is decided by the ratio of the two blocks under each.
FLUX_MODES = ("diffusive", "conormal", "reaction")

#: `probed-dtn-coupling` §2.2's cutoff, as a WAVELENGTH.  The plate is 32 cells
#: and the flow scale it resolves is the gap, so the trustworthy band is
#: lambda >= 4 cells and ``m = 2n/lambda + 1``.
LAMBDA_CUT_CELLS = 4

_FACE_GEOM = {"xlo": (-1.0, "x"), "xhi": (+1.0, "x"),
              "ylo": (-1.0, "y"), "yhi": (+1.0, "y")}

_RING = {
    "xlo": ((slice(None), 0), (slice(None), 1)),
    "xhi": ((slice(None), -1), (slice(None), -2)),
    "ylo": ((0, slice(None)), (1, slice(None))),
    "yhi": ((-1, slice(None)), (-2, slice(None))),
}


def modes_for(n_cells: int) -> int:
    """``m_eff`` from a cutoff WAVELENGTH, not from a mode count (W0 4.2)."""
    return int(2 * n_cells // LAMBDA_CUT_CELLS + 1)


def fourier_basis(n_cells: int, m: int | None = None, dx: float = DX) -> np.ndarray:
    """(n_cells, m) real Fourier modes, orthonormal in the dx-weighted pairing."""
    m = modes_for(n_cells) if m is None else m
    length = n_cells * dx
    y = (np.arange(n_cells) + 0.5) * dx
    cols = [np.full(n_cells, 1.0 / np.sqrt(length))]
    k = 1
    while len(cols) < m:
        w = 2.0 * np.pi * k * y / length
        cols.append(np.sqrt(2.0 / length) * np.cos(w))
        if len(cols) < m:
            cols.append(np.sqrt(2.0 / length) * np.sin(w))
        k += 1
    return np.column_stack(cols[:m])


# ---------------------------------------------------------------------------
# the tiling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundTiling:
    """A tiling of the `NX x NY` domain by `nw x nw` windows, overlapping.

    **The cuts are placed so that no seam passes through the wing**, and that is
    a deliberate answer to **W124**: `wake_array.Rotor` is declared on an
    x-*adjacency*, so every x-seam of CS-6 and CS-7 passes exactly through a
    rotor disc and the tiling rule could not express anything else.  Here the
    wing spans cells 88 to 120 and the x-overlaps are [64, 80) and [128, 144),
    so the plate sits 8 cells clear of each.  `wing_window()` returns the single
    window that owns it, and `cuts_clear_of_wing()` is asserted in the tests so
    it cannot drift.
    """

    n_col: int = N_COL
    n_row: int = N_ROW
    nw: int = NW
    stride: int = STRIDE
    ramp: int = RAMP

    @classmethod
    def single(cls) -> "GroundTiling":
        """One window covering the whole domain: the ZERO-CUT control.

        The partition of unity is identically one, `cut` and `assemble` are the
        identity, and the composed step must equal the referent step **bitwise**
        rather than to a tolerance -- CS-7 §2's N = 1 rung, which is the control
        `tier0-measurements` §8 says is the one nobody runs.
        """
        return cls(n_col=1, n_row=1, nw=0, stride=0)

    # -- the domain -------------------------------------------------------

    @property
    def is_single(self) -> bool:
        return self.n_col == 1 and self.n_row == 1

    @property
    def wx(self) -> int:
        """Window width in cells (the whole domain for the single tiling)."""
        return NX if self.is_single else self.nw

    @property
    def wy(self) -> int:
        return NY if self.is_single else self.nw

    @property
    def nx(self) -> int:
        return NX

    @property
    def ny(self) -> int:
        return NY

    @property
    def n_windows(self) -> int:
        return self.n_col * self.n_row

    @property
    def offsets(self) -> list[tuple[int, int]]:
        return [(i * self.stride, j * self.stride)
                for j in range(self.n_row) for i in range(self.n_col)]

    @property
    def names(self) -> list[str]:
        return [f"F{i}{j}" for j in range(self.n_row) for i in range(self.n_col)]

    @property
    def halo(self) -> int:
        return 0 if self.is_single else self.wx - self.stride

    def index_of(self, name: str) -> int:
        return self.names.index(name)

    def artificial_faces(self, ox: int, oy: int) -> tuple[str, ...]:
        out = []
        if ox > 0:
            out.append("xlo")
        if ox + self.wx < self.nx:
            out.append("xhi")
        if oy > 0:
            out.append("ylo")
        if oy + self.wy < self.ny:
            out.append("yhi")
        return tuple(out)

    # -- the wing's window -------------------------------------------------

    def wing_cells(self) -> tuple[int, int]:
        """The plate's chordwise cell span, [lo, hi), on the global lattice.

        This is the PLATE, which is what W124 is about.  The stamping box is
        eight cells wider each way and `stamp_cells` returns that; the force it
        carries out there is below float64 -- eight cells past the last station
        is sixteen chordwise sigma -- so the two clearances answer different
        questions and both are reported.
        """
        lo = int(math.floor(X_LE / DX))
        hi = int(math.ceil((X_LE + CHORD * math.cos(ALPHA)) / DX))
        return lo, hi

    def stamp_cells(self) -> tuple[int, int]:
        """The stamping box's chordwise span, margins included."""
        lo, hi = self.wing_cells()
        mid = 0.5 * (lo + hi)
        half = int(round(0.5 * CHORD / DX)) + 8
        return int(round(mid - half)), int(round(mid + half))

    def wing_window(self) -> str:
        """The one window that owns the whole plate, at every ride height."""
        lo, hi = self.wing_cells()
        for k, (ox, oy) in enumerate(self.offsets):
            if ox <= lo and hi <= ox + self.wx and oy == 0:
                return self.names[k]
        raise ValueError("no single window owns the plate at this tiling")

    def cuts_clear_of_wing(self) -> int:
        """Cells between the plate and the nearest x-overlap.  W124's check.

        Positive means no seam passes through the wing.  Returns the minimum
        clearance over every overlap band; ``NX`` when there are no cuts.
        """
        lo, hi = self.wing_cells()
        if self.is_single:
            return self.nx
        best = self.nx
        for i in range(self.n_col - 1):
            a = (i + 1) * self.stride          # overlap band [a, a + halo)
            b = a + self.halo
            if hi <= a:
                best = min(best, a - hi)
            elif b <= lo:
                best = min(best, lo - b)
            else:
                best = min(best, -1)
        return best

    # -- cut, blend, contaminate ------------------------------------------

    def cut(self, f: np.ndarray) -> np.ndarray:
        return np.stack([f[oy:oy + self.wy, ox:ox + self.wx]
                         for ox, oy in self.offsets])

    @lru_cache(maxsize=16)
    def weights(self) -> list[np.ndarray]:
        """chi_i, a squared linear ramp vanishing AT each artificial edge."""
        raw = []
        r = max(self.ramp, 1)
        wxn, wyn = self.wx, self.wy
        ix = np.arange(wxn) + 0.5
        iy = np.arange(wyn) + 0.5
        for ox, oy in self.offsets:
            gx, gy = np.ones(wxn), np.ones(wyn)
            if ox > 0:
                gx = np.minimum(gx, np.clip(ix / r, 0.0, 1.0))
            if ox + wxn < self.nx:
                gx = np.minimum(gx, np.clip((wxn - ix) / r, 0.0, 1.0))
            if oy > 0:
                gy = np.minimum(gy, np.clip(iy / r, 0.0, 1.0))
            if oy + wyn < self.ny:
                gy = np.minimum(gy, np.clip((wyn - iy) / r, 0.0, 1.0))
            w = np.zeros((self.ny, self.nx))
            w[oy:oy + wyn, ox:ox + wxn] = np.minimum(gy[:, None], gx[None, :]) ** 2
            raw.append(w)
        tot = np.sum(raw, axis=0)
        tot = np.where(tot <= 0.0, 1.0, tot)
        return [w / tot for w in raw]

    def assemble(self, us: np.ndarray, vs: np.ndarray):
        au, av = np.zeros((self.ny, self.nx)), np.zeros((self.ny, self.nx))
        ws = self.weights()
        for k, (ox, oy) in enumerate(self.offsets):
            chi = ws[k][oy:oy + self.wy, ox:ox + self.wx]
            au[oy:oy + self.wy, ox:ox + self.wx] += chi * us[k]
            av[oy:oy + self.wy, ox:ox + self.wx] += chi * vs[k]
        return au, av

    def contaminated(self, d_cells: int) -> list[np.ndarray]:
        out = []
        ix = np.arange(self.wx) + 0.5
        iy = np.arange(self.wy) + 0.5
        for ox, oy in self.offsets:
            faces = self.artificial_faces(ox, oy)
            dxx = np.full(self.wx, np.inf)
            dyy = np.full(self.wy, np.inf)
            if "xlo" in faces:
                dxx = np.minimum(dxx, ix)
            if "xhi" in faces:
                dxx = np.minimum(dxx, self.wx - ix)
            if "ylo" in faces:
                dyy = np.minimum(dyy, iy)
            if "yhi" in faces:
                dyy = np.minimum(dyy, self.wy - iy)
            out.append(np.minimum(dyy[:, None], dxx[None, :]) <= d_cells)
        return out

    def partition_of_unity(self, d_cells: int = RAMP) -> GridPartitionOfUnity:
        idx, wts, bad = {}, {}, {}
        ws, cont = self.weights(), self.contaminated(d_cells)
        for k, (ox, oy) in enumerate(self.offsets):
            rows, cols = np.meshgrid(np.arange(oy, oy + self.wy),
                                     np.arange(ox, ox + self.wx), indexing="ij")
            idx[self.names[k]] = (rows * self.nx + cols).reshape(-1)
            wts[self.names[k]] = ws[k][oy:oy + self.wy, ox:ox + self.wx].reshape(-1)
            bad[self.names[k]] = cont[k].reshape(-1)
        return GridPartitionOfUnity(
            self.nx * self.ny, idx, wts, contaminated=bad,
            ramp_cells=self.ramp,
            profile=f"min(gy, gx)^2 over a {self.ramp}-cell linear ramp, normalized")


DEFAULT_TILING = GroundTiling()
SINGLE_TILING = GroundTiling.single()


# ---------------------------------------------------------------------------
# the composition layer's own operator
# ---------------------------------------------------------------------------


def divergence_rms(u: np.ndarray, v: np.ndarray) -> float:
    """``||div u||_rms`` with the WIDE centred operator the solver inverts."""
    du = np.zeros_like(u)
    dv = np.zeros_like(v)
    du[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * DX)
    dv[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2.0 * DX)
    d = du + dv
    return float(np.sqrt(np.mean(d[2:-2, 2:-2] ** 2)))


@lru_cache(maxsize=4)
def _wavenumbers(ny: int, nxp: int):
    kx = 2.0 * np.pi * np.fft.fftfreq(nxp, d=DX)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=DX)
    if nxp % 2 == 0:
        kx[nxp // 2] = 0.0
    if ny % 2 == 0:
        ky[ny // 2] = 0.0
    k2 = kx[None, :] ** 2 + ky[:, None] ** 2
    return kx[None, :], ky[:, None], np.where(k2 == 0.0, 1.0, k2)


def _taper(pad: int = PAD_CELLS) -> np.ndarray:
    return np.cos(0.5 * np.pi * (np.arange(1, pad + 1) / pad)) ** 2


def project_assembled(u: np.ndarray, v: np.ndarray):
    """One global spectral Leray projection on the assembled field, numpy.

    The freestream is removed and restored here rather than by the caller, for
    `wake_array.project_assembled`'s reason: a caller that has to remember will
    one day not.  ``GroundRollout.project`` is the torch twin and the two are
    pinned against each other in the tests.
    """
    nyy, nxx = u.shape
    pad = PAD_CELLS
    uf, vf = u - U_INF, v
    tp = _taper(pad)[None, :]
    bu = np.concatenate((uf, uf[:, -1:] * tp), axis=1)
    bv = np.concatenate((vf, vf[:, -1:] * tp), axis=1)
    kx, ky, k2 = _wavenumbers(nyy, nxx + pad)
    uh, vh = np.fft.fft2(bu), np.fft.fft2(bv)
    div = kx * uh + ky * vh
    uh = uh - kx * div / k2
    vh = vh - ky * div / k2
    return U_INF + np.fft.ifft2(uh).real[:, :nxx], np.fft.ifft2(vh).real[:, :nxx]


def leray_projection(tiling: GroundTiling = DEFAULT_TILING) -> ConstraintProjection:
    """The composition layer's operator, declared (L6/C2, R12).

    Its hypothesis is the outer band, and this geometry satisfies it in a way
    `wake_array`'s does not have to state: the top is freestream, the bottom is a
    ROLLING ROAD at the same ``(U_INF, 0)``, so the field is periodic-compatible
    in y for the same reason it is at both laterals there.
    """
    return ConstraintProjection(
        constraint="divergence-free",
        scope="global",
        stage="after-assembly",
        cadence=1,
        operator=project_assembled,
        residual=divergence_rms,
        shape=(tiling.ny, tiling.nx),
        note=("ground_effect.project_assembled: an exact spectral Leray projection "
              "on the assembled domain extended downstream by PAD_CELLS of tapered "
              "fluctuation. Applied ONCE per exchange to the blended field, beside "
              "agents whose own projection has been removed -- W100's arrangement, "
              "not a projection added on top of one"),
    )


def projected_assembly(tiling: GroundTiling = DEFAULT_TILING,
                       d_cells: int = RAMP) -> ProjectedAssembly:
    return ProjectedAssembly(tiling.partition_of_unity(d_cells),
                             leray_projection(tiling))


# ---------------------------------------------------------------------------
# the flow expert
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def solver_for(nx: int, ny: int, nu: float = NU, device: str = "cpu"):
    """`wind_farm_design.torch_solver_class()` at this geometry, tape attached.

    The exposed class overrides ``_project`` with the identity, and ``_lam`` --
    the one place `WindowNS` is fixed to a square array -- is read by nothing
    else, so **the same class runs an 80x80 window and the 208x144 monolith**.
    That is what makes the zero-cut control bitwise rather than approximate:
    `GroundTiling.single()` puts the referent's own solver in the composed code
    path with a partition of unity identically one.
    """
    if torch is None:                                            # pragma: no cover
        raise RuntimeError("torch is required for ground_effect")
    s = torch_solver_class()(nu=nu, length=nx * DX, n=nx, cfl=0.4,
                             transmission="dirichlet", backend="torch",
                             device=device)
    s.b = _TapeBackend(s.b)
    s.ny_cells = ny
    return s


@lru_cache(maxsize=4)
def embedded_monolith(nu: float = NU):
    """The undivided domain with `WindowNS`'s OWN pressure solve left in.

    A second, independent referent.  It is not the one the gate is written
    against -- it uses a different elliptic operator from the composed column, so
    the difference between them is not the cut -- but a load history that agrees
    with it is a load history the projection did not invent.
    """
    from .scaling_ladder import RectangularNS
    return RectangularNS(nu=nu, length=NX * DX, n=NX, ny=NY, cfl=0.4,
                         transmission="dirichlet")


# ---------------------------------------------------------------------------
# the wing
# ---------------------------------------------------------------------------


@dataclass
class Wing:
    """A porous inclined plate: the actuator disk with the axis rotated.

    Everything that varies with the ride height carries its derivative: the
    station positions, the sampling offsets, the Gaussian, and the discrete
    normalization.  The stamping box's integer corner comes from a detached
    value and is a window on the lattice rather than a parameter, which is
    `DiskBank.forcing`'s own device.
    """

    chord: float = CHORD
    alpha: float = ALPHA
    c_n: float = C_N
    x_le: float = X_LE
    n_station: int = N_STATION
    device: str = "cpu"

    def __post_init__(self) -> None:
        opt = dict(dtype=TORCH_DTYPE, device=self.device)
        self.t_hat = torch.tensor([math.cos(self.alpha), math.sin(self.alpha)], **opt)
        self.n_hat = torch.tensor([-math.sin(self.alpha), math.cos(self.alpha)], **opt)
        #: Station arclengths measured from the LEADING EDGE, so ``h`` is the
        #: height of the plate's LOWEST point above the floor.  That is what a
        #: ride height is, and it is also the only definition under which a
        #: sweep to small ``h`` does not quietly put the leading edge inside the
        #: pinned floor row -- which the first sweep did, at 0.29 cells.
        self.s = torch.linspace(0.0, self.chord, self.n_station, **opt)
        self.ds = float(self.chord / self.n_station)
        self.x_mid = self.x_le + 0.5 * self.chord * math.cos(self.alpha)
        self.y_mid_offset = 0.5 * self.chord * math.sin(self.alpha)
        #: half-width of the stamping box, in cells: the plate plus 6 sigma
        self.box_x = int(round(0.5 * self.chord / DX)) + 8
        self.box_y = int(round(0.5 * self.chord * math.sin(self.alpha) / DX)) + 10
        self._opt = opt

    # -- geometry ----------------------------------------------------------

    def stations(self, h):
        """(x, y) of every chordwise station at LEADING-EDGE height ``h``."""
        cx = self.x_le + self.s * self.t_hat[0]
        cy = h + self.s * self.t_hat[1]
        return cx, cy

    def sample(self, u, v, h, ny: int, nx: int):
        """The external flow at each station: the MEAN of the two sides.

        A station read inside its own smearing reads its own induction and an
        actuator line that does so drives itself.  The self-induced velocity of
        a sheet is antisymmetric across it, so this average is the external flow
        to leading order -- and it is what makes the model a closure of the
        solver's field rather than a feedback loop on its own output.
        """
        cx, cy = self.stations(h)
        off = D_OFFSET * DX
        px = torch.stack((cx + off * self.n_hat[0], cx - off * self.n_hat[0]))
        py = torch.stack((cy + off * self.n_hat[1], cy - off * self.n_hat[1]))
        gx = 2.0 * px / (nx * DX) - 1.0
        gy = 2.0 * py / (ny * DX) - 1.0
        grid = torch.stack((gx, gy), dim=-1)[None]               # [1, 2, S, 2]
        fld = torch.stack((u, v))[None]                          # [1, 2, ny, nx]
        s = torch.nn.functional.grid_sample(
            fld, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return s[0, 0], s[0, 1]                                  # [2, S] each

    # -- the load ----------------------------------------------------------

    def external_normal(self, u, v, h, ny: int, nx: int):
        """``w_ext``: the EXTERNAL flow's plate-normal component, per station.

        It does not depend on the plate's velocity, which is what lets the seam's
        interface equation be solved in three scalar Newton steps with no second
        field evaluation.
        """
        us, vs = self.sample(u, v, h, ny, nx)
        um, vm = 0.5 * (us[0] + us[1]), 0.5 * (vs[0] + vs[1])
        return um * self.n_hat[0] + vm * self.n_hat[1]

    def station_normal(self, u, v, h, v_plate, ny: int, nx: int):
        """``w_k``: the plate-normal velocity of the flow RELATIVE to the plate.

        ``v_plate`` is the plate's own VERTICAL velocity per station -- the port's
        FLOW half -- so a rigid heave is a constant vector and a probe may hand
        this a mode.
        """
        us, vs = self.sample(u, v, h, ny, nx)
        um, vm = 0.5 * (us[0] + us[1]), 0.5 * (vs[0] + vs[1])
        w_ext = um * self.n_hat[0] + vm * self.n_hat[1]
        return w_ext - v_plate * self.n_hat[1], um, vm, us, vs

    def traction(self, w):
        """The plate's VERTICAL traction per unit plate area, per station.

        ``1/2 rho C_N |w| w`` along ``n``, projected on ``y``.  Negative under a
        rising-downstream plate in a forward stream, which is downforce.
        """
        return 0.5 * self.c_n * torch.abs(w) * w * self.n_hat[1]

    def load(self, w):
        """Downforce per unit span, positive DOWN: ``-integral of the traction``."""
        return -(self.traction(w) * self.ds).sum()

    # -- the body force ----------------------------------------------------

    def forcing(self, u, v, h, v_plate, ny: int, nx: int):
        """``(fx, fy, w, load)`` -- the force on the FLUID and the plate's load.

        The station kernels are normalized DISCRETELY on the stamping box, so

            sum(f * dx * dy)  ==  -(the force on the plate)

        holds to floating point on any lattice -- `disk.body_force_field`'s gate
        W2 property, which is what makes ``load`` an exact momentum exchange
        rather than a second model.
        """
        w, um, vm, _, _ = self.station_normal(u, v, h, v_plate, ny, nx)
        # force ON THE PLATE, per unit plate area, along n
        fn = 0.5 * self.c_n * torch.abs(w) * w                   # [S]
        cx, cy = self.stations(h)

        b_x, b_y = self.box_x, self.box_y
        hd = (float(h.detach()) if torch.is_tensor(h) else float(h))             + self.y_mid_offset
        ix0 = int(np.clip(round(self.x_mid / DX - 0.5) - b_x, 0, nx - (2 * b_x + 1)))
        iy0 = int(np.clip(round(hd / DX - 0.5) - b_y, 0, ny - (2 * b_y + 1)))
        ox = torch.arange(2 * b_x + 1, device=u.device)
        oy = torch.arange(2 * b_y + 1, device=u.device)
        gx = (ix0 + ox).to(TORCH_DTYPE) * DX + 0.5 * DX          # [W]
        gy = (iy0 + oy).to(TORCH_DTYPE) * DX + 0.5 * DX          # [H]

        dxg = gx[None, None, :] - cx[:, None, None]              # [S, 1, W]
        dyg = gy[None, :, None] - cy[:, None, None]              # [S, H, 1]
        d_n = dxg * self.n_hat[0] + dyg * self.n_hat[1]
        d_t = dxg * self.t_hat[0] + dyg * self.t_hat[1]
        sig_n, sig_t = SIGMA_N * DX, 0.5 * self.ds
        ker = (torch.exp(-0.5 * (d_n / sig_n) ** 2)
               * torch.exp(-0.5 * (d_t / sig_t) ** 2))           # [S, H, W]
        ker = ker / (ker.sum(dim=(1, 2), keepdim=True) * DX * DX)

        # the FLUID gets minus the plate's force, per station, spread by its kernel
        amp = -(fn * self.ds)[:, None, None] * ker
        fx_box = (amp * self.n_hat[0]).sum(dim=0)
        fy_box = (amp * self.n_hat[1]).sum(dim=0)
        fx = _place([fx_box], [(ix0, iy0)], ny, nx)
        fy = _place([fy_box], [(ix0, iy0)], ny, nx)
        return fx, fy, w, -(fn * self.n_hat[1] * self.ds).sum()


# ---------------------------------------------------------------------------
# the suspension -- two lines of algebra, zero fitted parameters
# ---------------------------------------------------------------------------


@dataclass
class Suspension:
    """``k (h0 - h) = L``: a lumped MECH agent on the wing's mounting face.

    The `disk.ActuatorDisk` pattern exactly -- algebraic, stateless in the sense
    that matters (its only state is the ride height it is asked about), zero
    fitted parameters, and exact to machine precision at any probe step because
    its response is the derivative of a closed form.

    **Its MECH response is AFFINE in the trace and its base is on the record.**
    A spring relates force to POSITION, and the port's flow half is a velocity,
    so over one exchange interval

        h_new = h + dt <v_plate> ,     effort = k (h0 - h_new) / c

    which is affine.  W74's tell is exactly this, `probe.base_sensitivity` clears
    an affine response outright, and the base is declared rather than left at
    zero: the plate's current heave rate is a state the expert is in, and 0 is
    not it once the interface is moving.
    """

    agent_id: str = "SUSP"
    k: float = K_SPRING
    h0: float = H0_REF
    chord: float = CHORD
    #: **The EXCHANGE interval, not the macro-step.**  A probe assembles the
    #: interface operator at the cadence the scheme runs the seam at, and R10b
    #: fixes that at ``dt_native / substeps_per_macro_step``.  Declared at the
    #: macro-step instead, this block comes out 4x too large and the whole of
    #: section 4.5's ratio moves with it -- `CASE-STUDY-GUIDE` mistake 5 arriving
    #: on the lumped side, where nothing about the agent is wrong and the
    #: declaration is off by the number of exchanges.
    dt: float = MACRO_DT / EXCHANGES
    h: float = H0_REF
    h_dot: float = 0.0
    n_station: int = N_STATION
    n_calls: int = field(default=0, init=False)

    # -- the two lines -----------------------------------------------------

    def ride_height(self, load: float) -> float:
        """``h = h0 - L/k``.  The whole expert."""
        return float(self.h0 - load / self.k)

    def spring_force(self, h: float) -> float:
        return float(self.k * (self.h0 - h))

    # -- the port ----------------------------------------------------------

    def probe_base(self, name: str) -> np.ndarray:
        """The plate's CURRENT heave rate, per station.

        Not zeros.  The response is affine, so the base does not move the
        operator -- and that is a measurement (`base_sensitivity`), not a reason
        not to declare it.  W74's correction is that the base belongs to the
        SEAM, so `assemble_seam` is handed the same vector for both sides.
        """
        return np.full(self.n_station, self.h_dot)

    def respond(self, name: str, trace: np.ndarray) -> np.ndarray:
        """(port, trace) -> the spring's VERTICAL traction per unit plate area."""
        self.n_calls += 1
        trace = np.asarray(trace, dtype=float).reshape(-1)
        if name != "mount:MECH":
            raise ValueError(f"{self.agent_id} has no port {name!r}")
        h_new = self.h + self.dt * float(np.mean(trace))
        return np.full(trace.shape, self.spring_force(h_new) / self.chord)

    def storage(self, u: Any = None, v: Any = None) -> float:
        """``1/2 k (h0 - h)^2``: real, and the only energy the agent owns."""
        return 0.5 * self.k * (self.h0 - self.h) ** 2

    def validity(self, state: Any = None, cond: Any = None) -> bool:
        """The suspension's declared envelope: the wing is off the floor and
        the spring is in compression, which is what a two-line linear law is
        about.  A ride height at or below the plate's own smearing is outside
        anything this expert was written for and the honest answer is a
        declination."""
        h = self.h if state is None else float(state)
        return bool(0.0 < self.h0 - h <= self.h0 and h > (D_OFFSET + 2.0) * DX)


# ---------------------------------------------------------------------------
# the flow agent's boundary channel
# ---------------------------------------------------------------------------


@dataclass
class FlowWindow:
    """One window of the composed column, exposed as a boundary response.

    Two kinds of port.  The four artificial faces are ordinary fluid-fluid
    `MECH` seams and behave exactly as `window_ns`'s do -- an imposed normal
    velocity on the ring, ``nu dw/dn`` back.  ``wing:MECH`` is the
    field-to-lumped one and is the whole point: the trace is the plate's own
    VERTICAL velocity per chordwise station and the effort is the vertical
    traction, in one of `FLUX_MODES`.
    """

    agent_id: str
    u0: np.ndarray
    v0: np.ndarray
    shared_faces: tuple[str, ...]
    has_wing: bool = False
    #: the window's own corner on the global lattice.  The plate is declared in
    #: GLOBAL coordinates and a window sees it in its own frame, so both offsets
    #: have to be subtracted -- getting the x one wrong put the plate outside the
    #: window's array, `grid_sample`'s border clamp returned a constant, and the
    #: probed fluid block came back EXACTLY zero with every other diagnostic
    #: healthy.  That is the silent-wrongness class in miniature and it is why
    #: `tests/test_tier27_ground_effect.py` asserts the block is nonzero.
    ox: int = 0
    oy: int = 0
    h: float = H0_REF
    dt: float = MACRO_DT / EXCHANGES
    nu: float = NU
    flux_mode: str = "diffusive"
    device: str = "cpu"
    n_calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.flux_mode not in FLUX_MODES:
            raise ValueError(self.flux_mode)
        self.u0 = np.asarray(self.u0, dtype=float)
        self.v0 = np.asarray(self.v0, dtype=float)
        self.ny_c, self.nx_c = self.u0.shape
        self._solver = solver_for(self.nx_c, self.ny_c, self.nu, self.device)
        self.h_solver = self._solver.h
        self._wing = (Wing(x_le=X_LE - self.ox * DX, device=self.device)
                      if self.has_wing else None)
        opt = dict(dtype=TORCH_DTYPE, device=self.device)
        self._u = torch.as_tensor(self.u0, **opt)
        self._v = torch.as_tensor(self.v0, **opt)

    # -- face bookkeeping --------------------------------------------------

    @staticmethod
    def face_of(port_name: str) -> str:
        return port_name.split(":", 1)[0]

    def _step(self, u, v, force=None):
        u1, v1 = self._solver.step_batch(u[None], v[None], self.dt,
                                         bc0=None, force=force)
        return u1[0], v1[0]

    # -- the boundary channel ---------------------------------------------

    def probe_base(self, name: str) -> np.ndarray:
        """Zero on every port, and it is the ABSOLUTE convention on the wing.

        The four fluid faces write the trace into the ring as a perturbation,
        which is `window_ns`'s convention and makes zero both the origin and a
        state the expert is in.  The wing's trace is the plate's own vertical
        velocity, written ABSOLUTELY: zero is the plate at rest, which is a real
        physical state and is the one the seam base is declared at.  W74's class,
        and the two sides of the seam then disagree by the heave rate -- which is
        what `L4/probe-base` is for and is reported rather than hidden.
        """
        if name == "wing:MECH":
            return np.zeros(N_STATION)
        return np.zeros(self.nx_c if self.face_of(name) in ("ylo", "yhi") else self.ny_c)

    def _wing_effort(self, u, v, v_plate):
        """The vertical traction on the plate, in the declared flux mode."""
        wing = self._wing
        hh = torch.as_tensor(self.h, dtype=TORCH_DTYPE, device=u.device)
        w, um, vm, us, vs = wing.station_normal(u, v, hh, v_plate,
                                                self.ny_c, self.nx_c)
        if self.flux_mode == "reaction":
            return wing.traction(w)
        # nu dv/dn across the sheet: the two sides are D_OFFSET cells apart
        dvdn = (vs[0] - vs[1]) / (2.0 * D_OFFSET * DX)
        diff = self.nu * dvdn
        if self.flux_mode == "diffusive":
            return diff
        # probed-dtn-coupling 4.1's conservative co-normal, nu dw/dn - (u.n) w.
        # W47 scoped it out at fluid-fluid seams because the two sides share a
        # ring cell with opposite normals and the advective term cancels. Here
        # there is no second fluid side and it does not.
        w_ext = um * wing.n_hat[0] + vm * wing.n_hat[1]
        return diff - w_ext * vm

    def respond(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        """(port, trace) -> flux.  The only callable the probe needs."""
        self.n_calls += 1
        trace = np.asarray(trace, dtype=float).reshape(-1)
        opt = dict(dtype=TORCH_DTYPE, device=self.device)

        if port_name == "wing:MECH":
            if not self.has_wing:
                raise ValueError(f"{self.agent_id} carries no wing")
            if trace.shape[0] != N_STATION:
                raise ValueError(f"wing trace has {trace.shape[0]} stations, "
                                 f"expected {N_STATION}")
            vp = torch.as_tensor(trace, **opt)
            hh = torch.as_tensor(self.h, **opt)
            fx, fy, _, _ = self._wing.forcing(self._u, self._v, hh, vp,
                                              self.ny_c, self.nx_c)
            u1, v1 = self._step(self._u, self._v, force=(fx[None], fy[None]))
            return self._wing_effort(u1, v1, vp).detach().cpu().numpy()

        face = self.face_of(port_name)
        ring, interior = _RING[face]
        n_face = self.nx_c if face in ("ylo", "yhi") else self.ny_c
        if trace.shape[0] != n_face:
            raise ValueError(f"trace on {port_name} has {trace.shape[0]} cells, "
                             f"expected {n_face}")
        r_u, r_v = self.u0.copy(), self.v0.copy()
        if _FACE_GEOM[face][1] == "x":
            r_u[ring] = r_u[ring] + trace
        else:
            r_v[ring] = r_v[ring] + trace
        u1, v1 = self._step(torch.as_tensor(r_u, **opt), torch.as_tensor(r_v, **opt))
        w = (u1 if _FACE_GEOM[face][1] == "x" else v1).detach().cpu().numpy()
        return self.nu * (w[ring] - w[interior]) / self.h_solver

    # -- declared properties ----------------------------------------------

    def storage(self, u: Any = None, v: Any = None) -> float:
        if u is None:
            u, v = self.u0, self.v0
        u = np.asarray(u, dtype=float)
        v = self.v0 if v is None else np.asarray(v, dtype=float)
        return 0.5 * float(np.sum(u * u + v * v)) * self.h_solver**2

    def validity(self, state: Any = None, cond: Any = None) -> bool:
        """`reference.WindowNS`'s own envelope: finite, and a cell Reynolds
        number at or below the 4 the class documents, doubled for headroom."""
        if state is None:
            u, v = self.u0, self.v0
        else:
            u, v = state
        u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
        if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
            return False
        return bool(self.h_solver * float(np.max(np.hypot(u, v))) / self.nu <= 8.0)


# ---------------------------------------------------------------------------
# the capability records
# ---------------------------------------------------------------------------


def _prolongation(agent_id: str, name: str, n_cells: int) -> Prolongation:
    return Prolongation(
        agent_id=agent_id, port_name=name,
        matrix=fourier_basis(n_cells),
        gram_V=DX * np.eye(n_cells),
        label=f"{modes_for(n_cells)}-mode real Fourier basis, cutoff "
              f"{LAMBDA_CUT_CELLS} cells")


def flow_ports(expert: FlowWindow, motion: MotionClass) -> list:
    ports = []
    for f in expert.shared_faces:
        n_cells = expert.nx_c if f in ("ylo", "yhi") else expert.ny_c
        name = f"{f}:MECH"
        ports.append(port_decl(
            name=name, port_type=PortType.MECH,
            geometry=f"{f} ring of {expert.agent_id} ({n_cells} cells)",
            direction=Direction.BIDIRECTIONAL, nondim=dict(MECH_SCALES),
            effective_resolution=modes_for(n_cells),
            # A window face does not move: only the WING does. Declaring the
            # whole agent's motion would put the refusal on the wrong ports.
            motion_class=MotionClass.STATIC,
            response_half=ResponseHalf.EFFORT,
            prolongation=_prolongation(expert.agent_id, name, n_cells)))
    if expert.has_wing:
        ports.append(port_decl(
            name="wing:MECH", port_type=PortType.MECH,
            geometry=f"the plate's {N_STATION} chordwise stations inside "
                     f"{expert.agent_id}",
            direction=Direction.BIDIRECTIONAL, nondim=dict(MECH_SCALES),
            effective_resolution=modes_for(N_STATION),
            motion_class=motion,
            # the vertical traction on the plate: the MECH EFFORT, against an
            # imposed plate velocity (the FLOW)
            response_half=ResponseHalf.EFFORT,
            prolongation=_prolongation(expert.agent_id, "wing:MECH", N_STATION)))
    return ports


def flow_capabilities(expert: FlowWindow,
                      motion: MotionClass = MotionClass.STATIC,
                      substeps: int = EXCHANGES) -> ExpertCapabilities:
    return ExpertCapabilities(
        expert_id=expert.agent_id,
        ports=flow_ports(expert, motion),
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=True,
        # R10's own prescription, and the arrangement W100 selects: the elliptic
        # part is OUT of the agent and the composition layer applies it once, to
        # the assembled field.
        elliptic_subsolve=EllipticSubsolve.EXPOSED,
        stencil_radius=2,
        # R10b: the composition layer applies the exposed elliptic part at the
        # agent's OWN cadence. `GroundRollout` exchanges EXCHANGES times per
        # macro-step and asserts the agent took one sub-step per exchange.
        substeps_per_macro_step=substeps,
        time_discretization=TimeDiscretization.EXPLICIT,
        differentiable=Differentiable.NONE,
        dt_native=MACRO_DT,
        L_native=expert.nx_c * DX,
        storage=expert.storage,
        equivariances=(),
        validity=expert.validity,
        governing_family="incompressible-navier-stokes-2d",
        # A real solver is its own referent: its infidelity is zero by
        # construction, and declaring it is what lets the compile say tau = 0
        # rather than tau = UNDEFINED at the multiphysics-shaped wing seam.
        lambda_ref="itself: reference.WindowNS, exposed, at this cell",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash=f"windowns-exposed-nu{expert.nu:.6g}-n{expert.nx_c}"
                    f"x{expert.ny_c}-{expert.flux_mode}",
        boundary_response=expert.respond,
        probe_base=expert.probe_base,
        reproducibility_floor=np.finfo(float).eps,
        deterministic=True,
        note="reference.WindowNS with its elliptic part removed, torch float64, "
             "tape attached; the same class family PoC 1a differentiates")


def suspension_capabilities(expert: Suspension,
                            motion: MotionClass = MotionClass.STATIC
                            ) -> ExpertCapabilities:
    return ExpertCapabilities(
        expert_id=expert.agent_id,
        ports=[port_decl(
            name="mount:MECH", port_type=PortType.MECH,
            geometry=f"the wing's mounting face, {N_STATION} stations",
            direction=Direction.BIDIRECTIONAL, nondim=dict(MECH_SCALES),
            effective_resolution=modes_for(N_STATION),
            motion_class=motion,
            response_half=ResponseHalf.EFFORT,
            prolongation=_prolongation(expert.agent_id, "mount:MECH", N_STATION))],
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=True,
        # KNOWN, unlike a checkpoint's: two lines of algebra contain no solve.
        elliptic_subsolve=EllipticSubsolve.NONE,
        stencil_radius=0,
        substeps_per_macro_step=1,
        time_discretization=TimeDiscretization.IMPLICIT,
        differentiable=Differentiable.NONE,
        dt_native=MACRO_DT,
        L_native=CHORD,
        storage=expert.storage,
        equivariances=(),
        validity=expert.validity,
        # The SAME family as the flow it closes. A lumped algebraic closure
        # WITHIN a continuum problem is not a different continuum problem, and
        # declaring otherwise fails E3 at the seam -- `wind_farm.actuator_disk`
        # has a comment on exactly this, put there after getting it wrong once.
        governing_family="incompressible-navier-stokes-2d",
        lambda_ref="itself: a closed form has no infidelity to measure",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash=f"spring-k{expert.k:.6g}-h0{expert.h0:.6g}",
        boundary_response=expert.respond,
        probe_base=expert.probe_base,
        reproducibility_floor=np.finfo(float).eps,
        deterministic=True,
        note="k (h0 - h) = L. Two lines, zero fitted parameters, the "
             "disk.ActuatorDisk pattern")


# ---------------------------------------------------------------------------
# measured constants
# ---------------------------------------------------------------------------

MEASURED_STATIC = MeasuredConstants(
    tau=0.0, gamma=0.0, norm_A=1.0,
    probe_state=f"settled fixed-floor wake at h = {H0_REF}, dt = {MACRO_DT}, "
                f"nu = {NU}",
    scheme=f"exposed agents + ProjectedAssembly, halo {HALO}, ramp {RAMP}, "
           f"{EXCHANGES} exchanges per macro-step, motion OFF",
    depth=0,
    source="scripts/w127_ground_effect.py, out/w127/")

MEASURED_MOVING = MeasuredConstants(
    tau=0.0, gamma=0.0, norm_A=1.0,
    probe_state=f"settled coupled state at h0 = {H0_REF}, dt = {MACRO_DT}, "
                f"nu = {NU}",
    scheme=f"exposed agents + ProjectedAssembly, halo {HALO}, ramp {RAMP}, "
           f"{EXCHANGES} exchanges per macro-step, ride height solved every "
           f"macro-step",
    depth=0,
    source="scripts/w127_ground_effect.py, out/w127/")


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------


def make_experts(u_full: np.ndarray, v_full: np.ndarray,
                 tiling: GroundTiling = DEFAULT_TILING,
                 h: float = H0_REF, h0: float = H0_REF,
                 flux_mode: str = "diffusive",
                 nu: float = NU) -> dict[str, Any]:
    """Cut the field into windows and build the spring the wing hangs from."""
    u_full = np.asarray(u_full, dtype=float)
    v_full = np.asarray(v_full, dtype=float)
    if u_full.shape != (tiling.ny, tiling.nx):
        raise ValueError(f"field is {u_full.shape}, expected "
                         f"{(tiling.ny, tiling.nx)}")
    us, vs = tiling.cut(u_full), tiling.cut(v_full)
    wing_window = tiling.wing_window()
    out: dict[str, Any] = {}
    for k, name in enumerate(tiling.names):
        ox, oy = tiling.offsets[k]
        out[name] = FlowWindow(
            agent_id=name, u0=us[k], v0=vs[k],
            shared_faces=tiling.artificial_faces(ox, oy),
            has_wing=(name == wing_window), ox=ox, oy=oy,
            # the window's own frame: the plate's height above ITS floor
            h=h - oy * DX, nu=nu, flux_mode=flux_mode)
    out["SUSP"] = Suspension(h=h, h0=h0)
    return out


def connections(tiling: GroundTiling = DEFAULT_TILING) -> list[Connection]:
    """The fluid-fluid seams from the layout, plus the one that matters."""
    conns: list[Connection] = []
    idx = {n: tiling.offsets[k] for k, n in enumerate(tiling.names)}
    for j in range(tiling.n_row):
        for i in range(tiling.n_col - 1):
            a, b = f"F{i}{j}", f"F{i+1}{j}"
            conns.append(Connection(
                seam_id=f"x{i}{j}", a=(a, "xhi:MECH"), b=(b, "xlo:MECH"),
                port_type=PortType.MECH, derive_space=True,
                # FALSE and honestly so: in an overlapping decomposition the two
                # rings a seam pairs are HALO cells apart and no two coincide.
                geometrically_coincident=False,
                # n_0 = 0, not 1: the elliptic part is in the composition layer,
                # so Lambda_i is the pure advection-diffusion DtN and the
                # incompressibility constraint that would make a fluid-fluid
                # MECH seam rank-deficient is not inside it (window_ns, 2026-08-27).
                expected_null_dim=0,
                note="fluid-fluid, an ordinary artificial boundary"))
    for j in range(tiling.n_row - 1):
        for i in range(tiling.n_col):
            a, b = f"F{i}{j}", f"F{i}{j+1}"
            conns.append(Connection(
                seam_id=f"y{i}{j}", a=(a, "yhi:MECH"), b=(b, "ylo:MECH"),
                port_type=PortType.MECH, derive_space=True,
                geometrically_coincident=False, expected_null_dim=0,
                note="fluid-fluid, an ordinary artificial boundary"))
    conns.append(Connection(
        seam_id="wing", a=(tiling.wing_window(), "wing:MECH"),
        b=("SUSP", "mount:MECH"),
        port_type=PortType.MECH, derive_space=True,
        geometrically_coincident=True,
        # field <-> lumped: the lumped side responds in exactly the direction
        # incompressibility leaves open, so its true null space is 0 and
        # declaring 1 produces a false defect (CASE-STUDY-GUIDE mistake 3).
        expected_null_dim=0,
        note="THE seam: field-to-lumped MECH, W97's blind spot, and the one "
             "whose geometry is a function of the solution"))
    return conns


def build(u_full: np.ndarray,
          v_full: np.ndarray,
          motion: bool = False,
          tiling: GroundTiling = DEFAULT_TILING,
          h: float = H0_REF,
          h0: float = H0_REF,
          flux_mode: str = "diffusive",
          nu: float = NU,
          measured: MeasuredConstants | None = "default",
          experts: dict[str, Any] | None = None) -> tuple[CaseGraph, dict[str, Any]]:
    """The graph at a state, with the interface static or solution-dependent.

    ``motion=False``  the fixed-floor control: the plate is pinned at ``h`` and
                      the seam is an ordinary field-to-lumped `MECH` seam.  This
                      is the column W30's drift number is compared against.
    ``motion=True``   the plate's position is the spring's output, so the seam's
                      geometry is a function of the solution.  **L2 refuses**,
                      at `InterfaceMotion`, and that refusal is correct: no rule
                      exists.  What this case study supplies is the three
                      measurements the hole asks for and has never had.
    """
    mc = MotionClass.SOLUTION_DEPENDENT if motion else MotionClass.STATIC
    if measured == "default":
        measured = MEASURED_MOVING if motion else MEASURED_STATIC
    experts = experts or make_experts(u_full, v_full, tiling, h, h0, flux_mode, nu)
    agents = [Agent(n, flow_capabilities(experts[n], mc), domain=f"window {n}")
              for n in tiling.names]
    agents.append(Agent("SUSP", suspension_capabilities(experts["SUSP"], mc),
                        domain="the wing's mounting face", role="suspension"))
    return (
        CaseGraph(
            name=f"ground-effect-{tiling.n_col}x{tiling.n_row}-"
                 f"{'moving' if motion else 'fixed-floor'}",
            agents=agents,
            connections=connections(tiling),
            decomposition=Decomposition.OVERLAPPING,
            overlap=tiling.halo * DX,
            overlap_cells=tiling.halo,
            partition_of_unity=projected_assembly(tiling),
            global_fields=[GlobalField(
                "pressure",
                # W117: produced by every FLUID agent. The suspension is not in
                # the list because a spring does not have a pressure field, and
                # a proper subset applied outside itself is what L3/global-field
                # refuses -- `applies_to` is the same set.
                produced_by=tuple(tiling.names),
                applies_to=tuple(tiling.names),
                note="the elliptic part, in the composition layer where R10 puts "
                     "it: one global spectral Leray projection on the assembled "
                     "field, once per exchange"),
            ],
            cross_points=() if tiling.is_single else ("mid",),
            macro_dt=MACRO_DT,
            measured=measured,
            note=(f"{tiling.n_windows} exposed reference.WindowNS windows of "
                  f"{tiling.wx}x{tiling.wy} cells tiling {NX}x{NY} at h = 1/64, "
                  f"halo {tiling.halo}; a porous inclined plate at "
                  f"{math.degrees(ALPHA):.0f} deg over a rolling road; the "
                  f"referent is the same solver undivided with the spring "
                  f"tightly coupled")),
        experts,
    )


# ---------------------------------------------------------------------------
# the march -- one code path, and the referent is a tiling of it
# ---------------------------------------------------------------------------


@dataclass
class MarchResult:
    """What one march returns.  Every trace is per macro-step."""

    load: np.ndarray                  # downforce per unit span
    height: np.ndarray                # ride height
    h_dot: np.ndarray
    u_max: np.ndarray
    div_rms: np.ndarray
    substeps: np.ndarray              # the agent's own sub-steps per exchange
    inner_residual: np.ndarray        # the tight coupling's fixed-point residual
    u: Any = None
    v: Any = None
    h_final: float = 0.0
    wall_s: float = 0.0
    fields: dict = field(default_factory=dict)

    def settled(self, frac: float = 0.25) -> tuple[float, float]:
        """(mean load, mean height) over the last ``frac`` of the march.

        A quarter, not the last five samples: a developed wake meanders, and
        `atlas-proof-of-concept-1` section 10 measured five samples moving a
        gain by fifteen points.
        """
        n = max(1, int(round(frac * self.load.size)))
        return float(self.load[-n:].mean()), float(self.height[-n:].mean())


class GroundRollout:
    """The composed march, differentiable, with a PINNED window order.

    One code path serves three things, which is what makes the controls mean
    anything:

      ``GroundTiling()``          the composed column -- six windows, thirteen
                                  seams, blend and project every exchange
      ``GroundTiling.single()``   the REFERENT -- one window covering the whole
                                  domain, a partition of unity identically one,
                                  and the projection applied to a field nothing
                                  was blended into.  The two differ by the CUT
                                  and by nothing else.
      ``coupling='tight'``        the ride height and the load satisfy
                                  ``k(h0-h) = L`` simultaneously at every
                                  exchange; ``'lagged'`` updates it once per
                                  macro-step from the load already computed,
                                  which is what a two-agent exchange does.
    """

    def __init__(self, tiling: GroundTiling = DEFAULT_TILING,
                 k: float = K_SPRING, nu: float = NU, device: str = "cpu",
                 coupling: str = "lagged", motion: bool = True,
                 lag: int = 1, n_inner: int = 3,
                 checkpoint: bool = True) -> None:
        if coupling not in ("tight", "lagged", "staggered"):
            raise ValueError(coupling)
        self.lag = max(1, int(lag))
        self.tiling = tiling
        self.k = k
        self.nu = nu
        self.device = device
        self.coupling = coupling
        self.motion = motion
        self.n_inner = n_inner
        self.checkpoint = checkpoint
        self.ny, self.nx = tiling.ny, tiling.nx
        self.solver = solver_for(tiling.wx, tiling.wy, nu, device)
        self.wing = Wing(device=device)
        self.assembly = projected_assembly(tiling)
        self.dt_ex = MACRO_DT / EXCHANGES
        opt = dict(dtype=TORCH_DTYPE, device=device)

        offs = tiling.offsets
        self.n_win = len(offs)
        self._offsets = offs
        chi = np.empty((self.n_win, tiling.wy, tiling.wx))
        w = tiling.weights()
        for kk, (ox, oy) in enumerate(offs):
            chi[kk] = w[kk][oy:oy + tiling.wy, ox:ox + tiling.wx]
        self._chi = torch.as_tensor(chi, **opt)

        # the outer band: inlet and TOP at freestream, the FLOOR one cell
        m = np.zeros((self.ny, self.nx), dtype=bool)
        m[:, :BAND] = True
        m[-BAND:, :] = True
        m[0, :] = True                       # the rolling road
        self._band = torch.as_tensor(m, device=device)

        kx, ky, k2 = _wavenumbers(self.ny, self.nx + PAD_CELLS)
        self._kx = torch.as_tensor(kx, **opt)
        self._ky = torch.as_tensor(ky, **opt)
        self._k2 = torch.as_tensor(k2, **opt)
        self._tp = torch.as_tensor(_taper()[None, :], **opt)
        self.substep_log: list[int] = []

    # -- the four composition-layer operators ------------------------------

    def cut(self, f):
        wx, wy = self.tiling.wx, self.tiling.wy
        return torch.stack([f[oy:oy + wy, ox:ox + wx] for ox, oy in self._offsets])

    def blend(self, us, vs):
        cu, cv = self._chi * us, self._chi * vs
        return (_place([cu[i] for i in range(self.n_win)], self._offsets,
                       self.ny, self.nx),
                _place([cv[i] for i in range(self.n_win)], self._offsets,
                       self.ny, self.nx))

    def project(self, u, v):
        uf, vf = u - U_INF, v
        bu = torch.cat((uf, uf[:, -1:] * self._tp), dim=1)
        bv = torch.cat((vf, vf[:, -1:] * self._tp), dim=1)
        uh, vh = torch.fft.fft2(bu), torch.fft.fft2(bv)
        div = self._kx * uh + self._ky * vh
        uh = uh - self._kx * div / self._k2
        vh = vh - self._ky * div / self._k2
        return (U_INF + torch.fft.ifft2(uh).real[:, :self.nx],
                torch.fft.ifft2(vh).real[:, :self.nx])

    def band(self, u, v):
        z = torch.zeros((), dtype=u.dtype, device=u.device)
        return (torch.where(self._band, z + U_INF, u),
                torch.where(self._band, z, v))

    # -- the coupling ------------------------------------------------------

    def load_at(self, u, v, h, v_plate):
        """``L`` on THIS field, at this plate position and velocity."""
        w, _, _, _, _ = self.wing.station_normal(u, v, h, v_plate,
                                                 self.ny, self.nx)
        return self.wing.load(w)

    def solve_interface(self, u, v, h, h0, v_plate):
        """The seam's own equation, solved for the plate's vertical velocity.

        **This is the interface problem, and writing it in the port's own
        variables is what makes it well posed.**  The naive reading of
        ``k(h0-h) = L`` as an UPDATE -- move the plate straight to the height the
        spring wants -- is the staggered scheme, and it diverges here at the
        first macro-step: measured, the load's sensitivity to the plate position
        on the field at hand is ``|dL/dh|_u = 1.95``, five to nine times the
        quasi-static ``0.22`` to ``0.39`` that a designer would compute, and it
        exceeds the spring rate.  Under-relaxation cannot rescue it either --
        the fixed point is repelling with a POSITIVE derivative, so
        ``(1-w) + w g' > 1`` for every ``w > 0``.  This is the partitioned-FSI
        added-mass instability, arriving unprompted at the first moving
        interface in the vault.

        What is well posed is the `MECH` port's own condition: the total
        vertical force on a massless plate is zero, in the trace variable the
        port declares.  With ``v`` the plate's vertical velocity and
        ``h_new = h + dt v``,

            R(v) = k (h0 - h - dt v)  -  L(u, h, v)   =   0

        and both halves are exactly the two agents' ``boundary_response``s: the
        first is `Suspension.respond` and the second is the wing port's
        ``reaction`` effort.  ``dR/dv`` is analytic and needs no second field
        evaluation, because the external flow at the stations does not depend on
        ``v`` at all:

            dR/dv = -k dt  -  C_N sum_k |w_k| cos^2(alpha) ds

        so three Newton steps at a FIXED count -- fixed, because the tape has to
        have the same shape at every ``h0`` for a reverse-mode gradient to mean
        anything -- and the residual is returned so the count is defended rather
        than assumed.
        """
        wing = self.wing
        w_ext = wing.external_normal(u, v, h, self.ny, self.nx)
        ny_hat = wing.n_hat[1]
        vp = v_plate
        for _ in range(self.n_inner):
            w = w_ext - vp * ny_hat
            load = -(wing.traction(w) * wing.ds).sum()
            r = self.k * (h0 - h - self.dt_ex * vp) - load
            dr = -self.k * self.dt_ex - (self.wing.c_n * torch.abs(w)
                                         * ny_hat ** 2 * wing.ds).sum()
            vp = vp - r / dr
        w = w_ext - vp * ny_hat
        load = -(wing.traction(w) * wing.ds).sum()
        res = torch.abs(self.k * (h0 - h - self.dt_ex * vp) - load)
        return vp, res

    # -- one exchange, and one macro-step ---------------------------------

    def exchange(self, u, v, h, v_plate):
        fx, fy, w, load = self.wing.forcing(u, v, h, v_plate, self.ny, self.nx)
        us, vs = self.cut(u), self.cut(v)
        fxs, fys = self.cut(fx), self.cut(fy)
        u1, v1 = self.solver.step_batch(us, vs, self.dt_ex, bc0=None,
                                        force=(fxs, fys))
        self.substep_log.append(int(self.solver.last_substeps))
        au, av = self.blend(u1, v1)
        au, av = self.project(au, av)
        bu, bv = self.band(au, av)
        return bu, bv, load

    def macro_step(self, u, v, h, v_plate, h0, step: int = 0):
        """`EXCHANGES` exchanges, and the seam solved at the declared cadence.

        ``tight``     the interface equation is solved at EVERY exchange, on the
                      field that exchange is about to be taken from.  This is the
                      referent's coupling, and it is affordable because the
                      equation is scalar and needs no second field evaluation.
        ``lagged``    it is solved once every ``lag`` macro-steps and the plate's
                      velocity is held in between -- an operator splitting with a
                      declared lag, `thermal_strain`'s construction exactly.
        ``staggered`` the seam is not solved at all: the ride height is moved
                      straight to ``h0 - L/k``, which is the obvious reading of a
                      two-line algebraic suspension.  **It diverges**, and it is
                      kept because a rule with nothing to fire on is not a rule.
        """
        h_in = h
        load = None
        res = torch.zeros((), dtype=TORCH_DTYPE, device=u.device)
        solve_now = (self.motion and self.coupling == "tight")
        lagged_now = (self.motion and self.coupling == "lagged"
                      and step % self.lag == 0)
        for i in range(EXCHANGES):
            if solve_now or (lagged_now and i == 0):
                v_plate, res = self.solve_interface(u, v, h, h0, v_plate)
            u, v, load = self.exchange(u, v, h, v_plate)
            if self.motion and self.coupling != "staggered":
                h = h + self.dt_ex * v_plate
        if self.motion and self.coupling == "staggered":
            h = h0 - load / self.k
            v_plate = (h - h_in) / MACRO_DT
        return u, v, h, v_plate, load, res

    # -- the march ---------------------------------------------------------

    def run(self, h0, steps: int, u0=None, v0=None, h_init=None,
            grad: bool = False, keep_fields: Sequence[int] = (),
            progress: Callable[[int, float], None] | None = None) -> MarchResult:
        import time
        opt = dict(dtype=TORCH_DTYPE, device=self.device)
        if not torch.is_tensor(h0):
            h0 = torch.tensor(float(h0), **opt)
        u = (torch.full((self.ny, self.nx), U_INF, **opt) if u0 is None
             else torch.as_tensor(np.asarray(u0), **opt))
        v = (torch.zeros((self.ny, self.nx), **opt) if v0 is None
             else torch.as_tensor(np.asarray(v0), **opt))
        h = h0 if h_init is None else torch.as_tensor(float(h_init), **opt)
        v_plate = torch.zeros((), **opt)
        self.substep_log = []

        loads, heights, hdots, umax, divs, ress = [], [], [], [], [], []
        fields, keep = {}, set(keep_fields)
        t0 = time.perf_counter()
        for s in range(steps):
            if s in keep:
                fields[s] = (u.detach().cpu().numpy().copy(),
                             v.detach().cpu().numpy().copy(), float(h.detach()))
            if grad and self.checkpoint:
                u, v, h, v_plate, load, res = torch.utils.checkpoint.checkpoint(
                    self.macro_step, u, v, h, v_plate, h0, s,
                    use_reentrant=False)
            else:
                u, v, h, v_plate, load, res = self.macro_step(
                    u, v, h, v_plate, h0, s)
            hv = float(h.detach())
            if self.motion and not (H_FLOOR < hv < H_CEILING):
                raise RuntimeError(
                    f"the ride height left the suspension's declared envelope at "
                    f"macro-step {s}: h = {hv:.6g}, admissible "
                    f"({H_FLOOR:.4g}, {H_CEILING:.4g}). A two-line algebraic "
                    f"suspension has no mass and no damper, so it moves the whole "
                    f"way to the new equilibrium in one macro-step; released from "
                    f"a freestream field, whose impulsive load is 2.0x the settled "
                    f"one, it puts the plate through the floor at step 0. Raised "
                    f"where it happens rather than clamped, because a hard stop is "
                    f"a fitted parameter and this expert has none")
            loads.append(load)
            heights.append(h)
            hdots.append(v_plate)
            ress.append(res)
            umax.append(float(torch.max(torch.hypot(u, v)).detach()))
            divs.append(divergence_rms(u.detach().cpu().numpy(),
                                       v.detach().cpu().numpy()))
            if not torch.isfinite(u).all() or not torch.isfinite(v).all():
                raise RuntimeError(
                    f"ground-effect rollout is not finite at macro-step {s}; "
                    "raised where it happened so the NaN is not carried into "
                    "every number after it")
            if progress is not None:
                progress(s, time.perf_counter() - t0)
        if steps in keep or -1 in keep:
            fields[steps] = (u.detach().cpu().numpy().copy(),
                             v.detach().cpu().numpy().copy(), float(h.detach()))
        return MarchResult(
            load=torch.stack(loads).detach().cpu().numpy(),
            height=torch.stack(heights).detach().cpu().numpy(),
            h_dot=torch.stack(hdots).detach().cpu().numpy(),
            u_max=np.array(umax), div_rms=np.array(divs),
            substeps=np.array(self.substep_log),
            inner_residual=torch.stack(ress).detach().cpu().numpy(),
            u=u, v=v, h_final=float(h.detach()),
            wall_s=time.perf_counter() - t0, fields=fields)

    # -- the objective F5 is about ----------------------------------------

    def objective(self, h0, steps: int, u0=None, v0=None, h_init=None,
                  frac: float = 0.25, grad: bool = False):
        """The settled downforce, as a tensor, so it can be differentiated.

        ``h0`` is a DESIGN PARAMETER and not a state: it enters only through the
        spring, and the initial field is the same for every value of it, which is
        what makes ``dJ/dh0`` a derivative of the composed stack rather than of
        its initial condition.
        """
        opt = dict(dtype=TORCH_DTYPE, device=self.device)
        if not torch.is_tensor(h0):
            h0 = torch.tensor(float(h0), **opt)
        u = (torch.full((self.ny, self.nx), U_INF, **opt) if u0 is None
             else torch.as_tensor(np.asarray(u0), **opt))
        v = (torch.zeros((self.ny, self.nx), **opt) if v0 is None
             else torch.as_tensor(np.asarray(v0), **opt))
        h = h0 if h_init is None else torch.as_tensor(float(h_init), **opt)
        v_plate = torch.zeros((), **opt)
        n_avg = max(1, int(round(frac * steps)))
        first = steps - n_avg
        j = None
        for s in range(steps):
            if grad and self.checkpoint:
                u, v, h, v_plate, load, _ = torch.utils.checkpoint.checkpoint(
                    self.macro_step, u, v, h, v_plate, h0, s,
                    use_reentrant=False)
            else:
                u, v, h, v_plate, load, _ = self.macro_step(
                    u, v, h, v_plate, h0, s)
            if s >= first:
                j = load if j is None else j + load
        return j / n_avg


def referent_rollout(**kw) -> GroundRollout:
    """The unsplit, tightly-coupled column the gate is written against."""
    kw.setdefault("coupling", "tight")
    return GroundRollout(tiling=SINGLE_TILING, **kw)


def composed_rollout(**kw) -> GroundRollout:
    """The six-window column with the ride height lagged one macro-step."""
    kw.setdefault("coupling", "lagged")
    return GroundRollout(tiling=DEFAULT_TILING, **kw)


#: Macro-steps of fixed-floor spin-up before a coupled march is released.  The
#: load settles to within 0.3% of its plateau by 30 (measured); 40 is that with
#: a quarter in hand, and it is the SAME field for every column and every value
#: of `h0`, which is what makes `dJ/dh0` a derivative of the composed stack
#: rather than of its initial condition.
N_SPIN = 40


@lru_cache(maxsize=16)
def settled_field(h: float = H_START, steps: int = N_SPIN,
                  single: bool = True, nu: float = NU):
    """The fixed-floor field at ride height ``h``, marched from the freestream.

    A coupled march is released from THIS and never from a uniform stream.  The
    reason is a property of the expert rather than of the harness: a two-line
    algebraic suspension has no mass, so it moves the whole way to its new
    equilibrium in one macro-step, and the impulsive-start load is **2.0x** the
    settled one -- enough, at this spring rate, to put the plate through the
    floor at step 0.  `GroundRollout.run` raises there rather than clamping.
    """
    tiling = SINGLE_TILING if single else DEFAULT_TILING
    r = GroundRollout(tiling=tiling, coupling="tight", motion=False, nu=nu)
    res = r.run(h0=h, steps=steps, h_init=h)
    return (res.u.detach().cpu().numpy().copy(),
            res.v.detach().cpu().numpy().copy(), res)
