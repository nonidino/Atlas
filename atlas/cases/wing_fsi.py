"""CS-12: two-way FSI, and the first genuinely NEW governing family. (W114, W136)

A 2-D wing section in a stream.  The flow over ``Omega_fluid`` comes from the
exposed-`WindowNS`-plus-`ProjectedAssembly` column CS-7 selected and CS-10 built
on; the wing itself is **quasi-static plane-stress elasticity** over
``Omega_solid``, solved by `thermostruct2d.ThermoStruct2D.solve_mechanical`, the
real solver, unmodified.  They meet at the wetted surface ``Gamma``: a
co-dimension-1 `MECH` seam carrying **traction against surface velocity**, which
is the port algebra's own MECH bond and is what it was written for.

**Two-way, and both directions are field solves.**  The flow's traction deflects
the wing; the deflection moves the flow's plate.  That is CS-10's moving
interface again -- and the whole difference is that CS-10's partner was a
two-line algebraic spring with one degree of freedom and this one is a field
with `N_STATION` of them, so the interface problem stops being scalar and the
questions that were invisible at rank one become the case study.

What only this case study can answer
------------------------------------

[[case-study-ladder-to-f1]] §4's CS-12 row: *"does `R` close across a genuinely
NEW governing family?"*  Six of the nine real case studies before it couple
incompressible Navier-Stokes to incompressible Navier-Stokes; `thermal_seam`
and `brake_thermal` add conduction at ONE seam and one-way; `thermal_strain`
adds elasticity but co-located, with no interface at all.  This is the first
graph in the vault whose two agents solve **different continuum problems**, are
coupled **both ways**, and meet on a **surface**.

Three things it settles, and the first had to be settled BEFORE it was built:

1. **W114.**  A quasi-static structural agent is `EMBEDDED` -- there is no time
   derivative, so the `split-step` escape `window_ns` and `brake_thermal` both
   take does not exist -- and `L2/R10` refused every graph containing one.  Its
   derivation is about a decomposition that CUTS the agent, and this graph does
   not cut the structure: ``Omega_solid`` is one agent's whole domain and the
   elasticity solve is over exactly the region a monolithic FSI solve would run
   it over.  Settled by narrowing the rule rather than by recording a scope
   defect -- see `R10_HANDLE` and `atlas/compiler.py::_r10_elliptic`.

2. **The gate.**  ``R``, the global power residual, closes across the FSI seam
   over the marched rollout, and the aeroelastic response -- tip deflection, and
   the divergence boundary -- matches the referent.  The referent is the
   monolithic FSI solve at a one-window fluid tiling with the SAME structural
   solve, so the zero-cut control is bitwise and the split differs by the cut
   alone.  CS-10's device, unchanged.

3. **Added mass.**  A wing under aerodynamic load is the canonical partitioned
   FSI added-mass problem, and a *quasi-static* structure is its extreme case:
   the structure has **no mass at all**, so the textbook fluid/structure mass
   ratio is infinite and is not the variable.  What is the variable is the
   aeroelastic stiffness ratio ``mu = rho(S_e^-1 K_aero)``, and
   `scripts/w136_wing_fsi.py` sweeps it.

The construction, stated as the model it is
--------------------------------------------

  the fluid      `ground_effect`'s column verbatim -- the same `DX`, the same
                 208x144 domain, the same six 80x80 exposed `WindowNS` windows
                 with halo 16, the same `ProjectedAssembly`, the same macro-step
                 and the same four exchanges inside it.  **Nothing about the
                 fluid moves between CS-10 and CS-12**, which is what makes the
                 partner the variable.
  the wing       `ground_effect.Wing`'s porous inclined plate with the scalar
                 ride height replaced by a per-station deflection.  `C_N = 20`,
                 `alpha = 20 deg`, chord 0.5, 32 stations at the structural
                 elements' own midpoints.
  the structure  `ThermoStruct2D` on a 32x2 Q1 plane-stress mesh of the plate,
                 clamped at the leading edge and free at the trailing edge: a
                 cantilevered flap, which is what an F1 front-wing element is.
                 `T = T_ref` everywhere, so the thermal load is identically zero
                 and the solve is pure elasticity.
  the seam       `Gamma`, the wetted surface.  Effort = the plate-NORMAL
                 traction per unit area; flow = the wetted surface's NORMAL
                 velocity.  Their product is the interface power exactly, and the
                 normal pair is the one that makes it so: the plate's force is
                 along ``n``, `solve_mechanical` consumes a pressure along ``n``,
                 and the compliance in that pairing is symmetric by Betti.  The
                 VERTICAL pairing is the obvious alternative and it is wrong --
                 measured, ``C_yy`` is asymmetric by 2.5e-4 and the work-conjugate
                 displacement misses the geometric one by 9.3%, because a
                 vertical traction against a vertical displacement is not an
                 energy pair once the surface also moves chordwise.  Under the
                 normal pairing the same two numbers are 5.1e-11 and 5.7e-4.

**The structure's stiffness is declared in FLOW units and that is a scope
statement, not a convenience.**  `SolidMaterial`'s own ``E = 70 GPa`` against a
traction of order ``rho U^2 = 1`` deflects a plate by ``1e-11`` and there is no
coupling to measure.  `E_STAR` is ``E / (rho U_inf^2)``, the aeroelastic
stiffness parameter, and it is the design knob §0.4's horizon rule is applied to.
It is the one number here chosen to make the coupling measurable rather than
derived, exactly as `ground_effect`'s ``C_N = 20`` is, and it is declared as such.

The seam's displacement, and the one identification this makes
---------------------------------------------------------------

The seam carries a generalized displacement ``delta``, defined as the
work-conjugate of the traction the port carries:

    sum_s p_s delta_s ds  ==  f^T u        (f the expert's own nodal load vector)

so the surface compliance ``C = U^T K U / ds`` is **symmetric by Betti**
(measured: 5.1e-11 relative) rather than by hope, and it is built from the
expert's own `solve_mechanical` calls and the expert's own assembled ``K_me``.
Nothing about the load assembly is re-derived here.

``delta`` is then used as the plate line's vertical position in the fluid.  The
porous plate has **no thickness in the flow** -- it is a line of stations -- so
there is no second definition to be inconsistent with; what the identification
costs is the difference between the work-conjugate displacement and the mesh's
geometric mid-surface one, measured at **5.7e-4** of the compliance norm at
``t/c = 0.04``, and it is a property of the STRUCTURAL model's thickness, which
the fluid model does not resolve at all.  Reported, not hidden.

What is reused and what is new
-------------------------------

Reused, unmodified: `ground_effect.GroundTiling` (the plate spans the same
cells, so `cuts_clear_of_wing` is the same 8), `ground_effect.solver_for`,
`ground_effect.projected_assembly`, `ground_effect.fourier_basis`,
`ground_effect.modes_for`, `ground_effect.divergence_rms`, and the taper and
wavenumbers behind the global projection.

New: `FlexWing` (per-station deflection instead of a scalar ride height),
`WingStructure` (the elasticity agent and its surface operator), `FSIRollout`
(the march, with a VECTOR interface Newton where CS-10 had a scalar one), and
the graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Sequence

import numpy as np

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
from .ground_effect import (
    ALPHA,
    BAND,
    C_N,
    CHORD,
    DX,
    EXCHANGES,
    HALO,
    MACRO_DT,
    MECH_SCALES,
    NU,
    NX,
    NY,
    N_COL,
    N_ROW,
    NW,
    PAD_CELLS,
    RAMP,
    SIGMA_N,
    STRIDE,
    U_INF,
    X_LE,
    D_OFFSET,
    GroundTiling,
    _taper,
    _wavenumbers,
    divergence_rms,
    fourier_basis,
    modes_for,
    projected_assembly,
    solver_for,
)
from .thermal_strain import load_solvers
from .wind_farm_design import TORCH_DTYPE, _place, torch

__all__ = [
    "THICK", "NI_STRUCT", "NJ_STRUCT", "N_STATION", "E_STAR", "E_REF",
    "Y_MOUNT", "FLUX_MODES", "LAMBDA_CUT_CELLS",
    "WingTiling", "DEFAULT_TILING", "SINGLE_TILING",
    "FlexWing", "WingStructure", "FSIFlowWindow",
    "flow_capabilities", "structure_capabilities", "build",
    "FSIRollout", "FSIResult", "referent_rollout", "composed_rollout",
    "settled_field", "N_SPIN", "MEASURED_STATIC", "MEASURED_MOVING",
    "CUT_DEFECT_STATIC", "CUT_DEFECT_MOVING",
    "R10_HANDLE", "STRUCTURE_SCOPE",
]


# ---------------------------------------------------------------------------
# the structure's geometry, and the one number chosen rather than derived
# ---------------------------------------------------------------------------

#: The plate's thickness.  ``t / c = 0.04``: thin enough that a Q1 plane-stress
#: mesh two elements deep is a plate and not a block, thick enough that the
#: element aspect ratio stays at 3:1 rather than the 20:1 slivers `_solve_free`'s
#: docstring warns about.
THICK = 0.02

#: The structural mesh: elements along the chord and through the thickness.
#: `N_STATION` is `NI_STRUCT` and not one more, because the seam's degrees of
#: freedom are the wetted face's ELEMENTS -- a pressure lives on an element and a
#: nodal reading of it is rank deficient by one, which is exactly the trap
#: `CASE-STUDY-GUIDE` mistake 3 has on a different axis.
NI_STRUCT, NJ_STRUCT = 32, 2
N_STATION = NI_STRUCT

#: `probed-dtn-coupling` §2.2's cutoff as a WAVELENGTH, carried from
#: `ground_effect` unchanged: the plate is the same plate.
LAMBDA_CUT_CELLS = 4

#: The stiffness the compliance is BUILT at.  Everything downstream scales
#: exactly: linear elasticity is exactly linear in ``1/E`` at fixed ``nu``, so
#: ``S_e(E) = (E / E_REF) S_e(E_REF)`` holds to the bit -- asserted, and it is
#: what makes ``E`` differentiable through a march that never re-runs the FE
#: solver.
E_REF = 1.0e5

#: The design knob, in FLOW units: ``E_STAR = E / (rho U_inf^2)``.  At this value
#: the settled aerodynamic load deflects the trailing edge by **1.64 cells**
#: vertically (measured; 1.74 along the plate normal), which is the same order as
#: the 1.10 cells CS-10's interface travelled in ``K = 10`` macro-steps and is
#: what makes the moving-interface machinery measurable rather than nominal.
#: Chosen to make the coupling measurable, not derived from a material -- and
#: swept, because the sweep is deliverable 3.  ``mu = rho(S_e^-1 K_aero)`` is
#: 0.0772 here, so the divergence boundary is 13x softer.
E_STAR = 5.0e4

#: Poisson's ratio and the thermal expansion coefficient handed to the expert.
#: ``alpha = 0`` and ``T = T_ref`` make the thermal load identically zero, so
#: `solve_mechanical` runs as pure quasi-static elasticity.  That is the whole
#: point: CS-9 measured the thermal-strain BOND and this case study needs the
#: ELASTICITY, and turning the coupling off by a declared coefficient is more
#: honest than reaching into the solver.
POISSON = 0.33
T_REF = 288.15

#: The leading edge's fixed mount height, in cells and then in length units.  The
#: plate spans y cells 40 to 51 at zero deflection; the y-overlap band is
#: [64, 80), so the cut clears the wing by 13 cells the way the x-cuts clear it
#: by 8 (`GroundTiling.cuts_clear_of_wing`, W124's discipline in the other
#: direction).  The moving lower wall is 40 cells -- 1.25 chords -- below, which
#: is far enough that this is a wing in a stream and not a ground-effect study;
#: CS-10's own `L(h)` table moves 1% across that range.
Y_MOUNT_CELLS = 40
Y_MOUNT = Y_MOUNT_CELLS * DX

#: The three efforts a `MECH` seam can carry, `ground_effect.FLUX_MODES`
#: unchanged.  W97 closed at a field-to-lumped seam; whether the co-normal is
#: still the one that separates the two blocks at a field-to-FIELD seam is a
#: question only this case study can ask, and `flux_mode` is how it asks it.
FLUX_MODES = ("diffusive", "conormal", "reaction")

#: Envelope for `WingStructure.validity`.  Small strain is the constitutive law's
#: own hypothesis and a deflection above a tenth of the chord is outside it; the
#: predicate declines rather than the harness clamping, `Suspension.validity`'s
#: pattern exactly.
DELTA_MAX = 0.1 * CHORD


# ---------------------------------------------------------------------------
# the tiling -- `GroundTiling`, and the y clearance it does not check
# ---------------------------------------------------------------------------


class WingTiling(GroundTiling):
    """`ground_effect.GroundTiling` with the vertical clearance added.

    The parent places the x-cuts clear of the plate (W124) and says nothing about
    y, because CS-10's plate hugged a floor that was always in the bottom row.
    Here the plate sits at mid-height and the y-overlap band is a real hazard, so
    the check exists on this side and a test asserts it.
    """

    @classmethod
    def single(cls) -> "WingTiling":
        return cls(n_col=1, n_row=1, nw=0, stride=0)

    def wing_rows(self) -> tuple[int, int]:
        """The plate's y cell span at zero deflection, [lo, hi)."""
        lo = int(math.floor(Y_MOUNT / DX))
        hi = int(math.ceil((Y_MOUNT + CHORD * math.sin(ALPHA)) / DX)) + 1
        return lo, hi

    def rows_clear_of_wing(self) -> int:
        """Cells between the plate and the nearest y-overlap band."""
        lo, hi = self.wing_rows()
        if self.is_single:
            return self.ny
        best = self.ny
        for j in range(self.n_row - 1):
            a = (j + 1) * self.stride
            b = a + self.halo
            if hi <= a:
                best = min(best, a - hi)
            elif b <= lo:
                best = min(best, lo - b)
            else:
                best = min(best, -1)
        return best

    def wing_window(self) -> str:
        """The one window that owns the whole plate, in x AND in y."""
        lo, hi = self.wing_cells()
        rlo, rhi = self.wing_rows()
        for k, (ox, oy) in enumerate(self.offsets):
            if ox <= lo and hi <= ox + self.wx and oy <= rlo and rhi <= oy + self.wy:
                return self.names[k]
        raise ValueError("no single window owns the plate at this tiling")


DEFAULT_TILING = WingTiling(n_col=N_COL, n_row=N_ROW, nw=NW, stride=STRIDE, ramp=RAMP)
SINGLE_TILING = WingTiling.single()


# ---------------------------------------------------------------------------
# the wing -- `ground_effect.Wing` with a deflection instead of a ride height
# ---------------------------------------------------------------------------


@dataclass
class FlexWing:
    """A porous inclined plate whose stations carry their own vertical offsets.

    `ground_effect.Wing` is the same model with one degree of freedom: its
    ``stations(h)`` puts the whole plate at a single leading-edge height.  Here
    ``h`` is a VECTOR -- the mount height plus the structure's own generalized
    deflection at each station -- and everything downstream broadcasts.

    **The stations are the structural elements' midpoints.**  CS-10 put them at
    the chord's endpoints because nothing on the other side cared; here the other
    side is a mesh, and the seam's degrees of freedom have to be the ones the
    expert's own load assembly is defined on.
    """

    chord: float = CHORD
    alpha: float = ALPHA
    c_n: float = C_N
    x_le: float = X_LE
    y_mount: float = Y_MOUNT
    n_station: int = N_STATION
    device: str = "cpu"

    def __post_init__(self) -> None:
        opt = dict(dtype=TORCH_DTYPE, device=self.device)
        self.t_hat = torch.tensor([math.cos(self.alpha), math.sin(self.alpha)], **opt)
        self.n_hat = torch.tensor([-math.sin(self.alpha), math.cos(self.alpha)], **opt)
        #: element MIDPOINT arclengths from the leading edge
        self.ds = float(self.chord / self.n_station)
        self.s = torch.as_tensor(
            (np.arange(self.n_station) + 0.5) * self.ds, **opt)
        self.x_mid = self.x_le + 0.5 * self.chord * math.cos(self.alpha)
        self.y_mid_offset = 0.5 * self.chord * math.sin(self.alpha)
        self.box_x = int(round(0.5 * self.chord / DX)) + 8
        # abs(), because the box has to COVER the body's vertical extent and a
        # body leaning nose-DOWN spans just as much of it as one leaning up.
        # With the signed sine the box SHRANK as alpha went negative: a plate
        # dropping more than ~11 cells got box_y <= 0 and torch.arange raised
        # "upper bound and larger bound inconsistent with step sign", and one
        # dropping less than that got a box too small to hold its own kernel,
        # which renormalises over the clipped region -- so the force stayed
        # conserved but was squeezed into too narrow a band of y.  W235.
        self.box_y = int(round(0.5 * self.chord * abs(math.sin(self.alpha)) / DX)) + 10
        self._opt = opt

    # -- geometry ---------------------------------------------------------

    def stations(self, delta, h=None):
        """(x, y) of every station at the NORMAL deflection ``delta`` [S].

        The plate deflects along its own normal, which is the direction the
        structural response is defined in and the direction a bending plate
        actually moves.  CS-10's plate heaved vertically because a ride height
        is vertical; a deflection is not.

        **``h`` is the MOUNT height and it defaults to the declared one, which is
        what CS-12 runs at.**  Added 2026-09-04 for PoC 2, where the same wing
        rides on CS-10's suspension *and* bends on CS-12's structure, so the
        clamp's own height is a state rather than a constant.  It is strictly
        additive: every CS-12 call omits it and gets ``self.y_mount``, and the
        two degrees of freedom are genuinely different directions -- a ride
        height moves the stations along ``y`` and a deflection along ``n``, so
        neither can stand in for the other.
        """
        y0 = self.y_mount if h is None else h
        cx = self.x_le + self.s * self.t_hat[0] + delta * self.n_hat[0]
        cy = y0 + self.s * self.t_hat[1] + delta * self.n_hat[1]
        return cx, cy

    def sample(self, u, v, delta, ny: int, nx: int, h=None):
        """The external flow at each station: the MEAN of the two sides.

        `ground_effect.Wing.sample`'s argument verbatim -- a station read inside
        its own smearing reads its own induction, and the self-induced velocity
        of a sheet is antisymmetric across it.
        """
        cx, cy = self.stations(delta, h)
        off = D_OFFSET * DX
        px = torch.stack((cx + off * self.n_hat[0], cx - off * self.n_hat[0]))
        py = torch.stack((cy + off * self.n_hat[1], cy - off * self.n_hat[1]))
        gx = 2.0 * px / (nx * DX) - 1.0
        gy = 2.0 * py / (ny * DX) - 1.0
        grid = torch.stack((gx, gy), dim=-1)[None]
        fld = torch.stack((u, v))[None]
        s = torch.nn.functional.grid_sample(
            fld, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return s[0, 0], s[0, 1]

    def external_normal(self, u, v, delta, ny: int, nx: int, h=None):
        """``w_ext``: the external flow's plate-normal component, per station.

        It does not depend on the plate's velocity, which is what lets the seam's
        Newton system be assembled with one field evaluation per macro-step
        however many stations there are.
        """
        us, vs = self.sample(u, v, delta, ny, nx, h)
        um, vm = 0.5 * (us[0] + us[1]), 0.5 * (vs[0] + vs[1])
        return um * self.n_hat[0] + vm * self.n_hat[1]

    def station_normal(self, u, v, delta, w_plate, ny: int, nx: int, h=None):
        """``w_k``: the flow's plate-normal velocity RELATIVE to the plate.

        ``w_plate`` is the wetted surface's own NORMAL velocity -- the port's FLOW
        half -- so the subtraction is direct and carries no geometric factor.
        """
        us, vs = self.sample(u, v, delta, ny, nx, h)
        um, vm = 0.5 * (us[0] + us[1]), 0.5 * (vs[0] + vs[1])
        w_ext = um * self.n_hat[0] + vm * self.n_hat[1]
        return w_ext - w_plate, um, vm, us, vs

    # -- the load ---------------------------------------------------------

    def normal_traction(self, w):
        """``f_n = 1/2 rho C_N |w| w``: the load ON THE PLATE along ``n``.

        **This is the seam's EFFORT half.**  Its product with the surface's own
        normal velocity is the interface power exactly, with no geometric factor
        left over -- which is the whole reason the pairing is normal and not
        vertical.
        """
        return 0.5 * self.c_n * torch.abs(w) * w

    def vertical_traction(self, w):
        """The vertical component, for reporting downforce.  Not the port."""
        return self.normal_traction(w) * self.n_hat[1]

    def load(self, w):
        """Downforce per unit span, positive DOWN."""
        return -(self.vertical_traction(w) * self.ds).sum()

    # -- the body force ---------------------------------------------------

    def forcing(self, u, v, delta, w_plate, ny: int, nx: int, h=None):
        """``(fx, fy, w, load)`` -- the force on the FLUID and the plate's load.

        `ground_effect.Wing.forcing`'s discrete normalization, unchanged: the
        station kernels are normalized on their own stamping box, so the force on
        the fluid integrates to minus the force on the plate to floating point on
        any lattice.  The only difference is that the stamping box's integer
        corner is taken from the MEAN deflection rather than from a scalar ride
        height -- it is a window on the lattice and not a parameter, so a detached
        mean is the right thing to build it from.
        """
        w, um, vm, _, _ = self.station_normal(u, v, delta, w_plate, ny, nx, h)
        fn = self.normal_traction(w)
        cx, cy = self.stations(delta, h)

        b_x, b_y = self.box_x, self.box_y
        dm = (float(delta.detach().mean()) if torch.is_tensor(delta)
              else float(np.mean(delta)))
        y0 = self.y_mount if h is None else (
            float(h.detach()) if torch.is_tensor(h) else float(h))
        hd = y0 + dm * float(self.n_hat[1]) + self.y_mid_offset
        xm = self.x_mid + dm * float(self.n_hat[0])
        ix0 = int(np.clip(round(xm / DX - 0.5) - b_x, 0, nx - (2 * b_x + 1)))
        iy0 = int(np.clip(round(hd / DX - 0.5) - b_y, 0, ny - (2 * b_y + 1)))
        ox = torch.arange(2 * b_x + 1, device=u.device)
        oy = torch.arange(2 * b_y + 1, device=u.device)
        gx = (ix0 + ox).to(TORCH_DTYPE) * DX + 0.5 * DX
        gy = (iy0 + oy).to(TORCH_DTYPE) * DX + 0.5 * DX

        dxg = gx[None, None, :] - cx[:, None, None]
        dyg = gy[None, :, None] - cy[:, None, None]
        d_n = dxg * self.n_hat[0] + dyg * self.n_hat[1]
        d_t = dxg * self.t_hat[0] + dyg * self.t_hat[1]
        sig_n, sig_t = SIGMA_N * DX, 0.5 * self.ds
        ker = (torch.exp(-0.5 * (d_n / sig_n) ** 2)
               * torch.exp(-0.5 * (d_t / sig_t) ** 2))
        ker = ker / (ker.sum(dim=(1, 2), keepdim=True) * DX * DX)

        amp = -(fn * self.ds)[:, None, None] * ker
        fx_box = (amp * self.n_hat[0]).sum(dim=0)
        fy_box = (amp * self.n_hat[1]).sum(dim=0)
        fx = _place([fx_box], [(ix0, iy0)], ny, nx)
        fy = _place([fy_box], [(ix0, iy0)], ny, nx)
        return fx, fy, w, -(fn * self.n_hat[1] * self.ds).sum()


# ---------------------------------------------------------------------------
# the structure -- the real solver, and its surface operator
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _plate_mesh(ni: int = NI_STRUCT, nj: int = NJ_STRUCT,
                chord: float = CHORD, thick: float = THICK,
                alpha: float = ALPHA, y_mount: float = Y_MOUNT,
                x_le: float = X_LE):
    """The plate's Q1 mesh, in GLOBAL coordinates.

    Built rotated into (x, y) rather than in plate-local axes, so the
    displacements `solve_mechanical` returns are already the ones the fluid
    consumes and nothing has to remember to rotate them back.  ``j = 0`` is the
    pressure side and ``j = nj`` the suction side, which is what `ShellMesh`'s
    ``inner_face`` / ``outer_face`` name.
    """
    TS = load_solvers()
    ca, sa = math.cos(alpha), math.sin(alpha)
    s = np.linspace(0.0, chord, ni + 1)
    n = np.linspace(-0.5 * thick, 0.5 * thick, nj + 1)
    S, N = np.meshgrid(s, n, indexing="ij")
    nodes = np.stack([x_le + S * ca - N * sa, y_mount + S * sa + N * ca], axis=-1)
    return TS.ShellMesh(nodes=nodes)


@lru_cache(maxsize=32)
def _surface_operator(e_ref: float = E_REF, ni: int = NI_STRUCT,
                      nj: int = NJ_STRUCT,
                      thick: float = THICK) -> dict[str, Any]:
    """The wetted face's compliance and stiffness, from the expert's own solves.

    ``C[s, s'] = `` the generalized NORMAL deflection at station ``s`` produced
    by a unit normal traction at station ``s'``, with "generalized" meaning the
    WORK-conjugate of the traction:

        sum_s p_s delta_s ds  ==  f^T u .

    Built as ``C = U^T K_me U / ds`` where ``U`` is the matrix of the expert's own
    `solve_mechanical` answers to the unit station tractions and ``K_me`` is the
    expert's own assembled stiffness.  Two things follow and both matter:

      * ``C`` is **symmetric by Betti** rather than by reciprocity holding
        approximately -- measured 5.1e-11 relative, which is the direct sparse
        solve's own round-off at a condition number of 7.1e8.  The same
        construction in the VERTICAL pairing comes back asymmetric at 2.5e-4,
        because two load systems that push along ``n`` and are read along ``y``
        do not satisfy Betti's identity in the read variable.
      * nothing about the load assembly is re-derived here.  ``U`` comes from the
        expert and ``K_me`` comes from the expert; this function contains no
        physics at all, which is the property `CASE-STUDY-GUIDE` asks of a case.

    ``S_e = C^-1`` is the structure's Dirichlet-to-Neumann map on ``Gamma``: the
    traction it takes to hold the surface at a given deflection.  It is the field
    analogue of `ground_effect.Suspension`'s ``k``, and where that was rank one
    this is rank `N_STATION`.
    """
    TS = load_solvers()
    mesh = _plate_mesh(ni=ni, nj=nj, thick=thick)
    mat = TS.SolidMaterial(E=e_ref, nu=POISSON, alpha=0.0)
    ts = TS.ThermoStruct2D(mesh, mat)
    T = np.full(mesh.n_nodes, T_REF)
    clamp = np.array([mesh.nid(0, j) for j in range(nj + 1)])
    ds = CHORD / ni
    nx_hat, ny_hat = -math.sin(ALPHA), math.cos(ALPHA)
    sig_cols: list[np.ndarray] = []

    def solve(p_unit: np.ndarray) -> np.ndarray:
        # split evenly between the two faces so the resultant acts on the
        # mid-surface rather than putting a spurious couple through the thickness
        p = 0.5 * np.asarray(p_unit, dtype=float)
        u, sig = ts.solve_mechanical(T, p, -p, T_ref=T_REF, clamp_nodes=clamp)
        sig_cols.append(np.asarray(sig, dtype=float))
        return u.reshape(-1)

    eye = np.eye(ni)
    U = np.column_stack([solve(eye[k]) for k in range(ni)])
    #: **The stress map, and it is a MAP rather than a solve.**  Added 2026-09-04
    #: for PoC 2's stress ceiling.  ``sigma_map[k]`` is the element stress
    #: ``(s_xx, s_yy, s_xy)`` the expert returns under a UNIT normal traction at
    #: station ``k``, so for any station traction ``q`` the stress field is
    #: ``sum_k q_k sigma_map[k]`` -- linear elasticity, exactly, and therefore
    #: differentiable in the traction without re-entering the FE solver.
    #:
    #: **It does not carry ``E``, and that is physics rather than an omission.**
    #: A stress is set by the load and the geometry: raising ``E`` at a fixed
    #: traction shrinks the displacement and the strain in the same proportion
    #: and leaves ``D eps`` where it was.  What ``E`` moves is the DEFLECTION,
    #: which is the other ceiling.
    sigma_map = np.stack(sig_cols, axis=0)          # [ni, n_elem, 3]
    C = (U.T @ (ts.K_me @ U)) / ds
    C = 0.5 * (C + C.T)                      # the residual is round-off, not model
    S_e = np.linalg.inv(C)
    mid = nj // 2
    ids = np.array([mesh.nid(i, mid) for i in range(ni + 1)])
    gx = np.column_stack([0.5 * (U[2 * ids[:-1], k] + U[2 * ids[1:], k])
                          for k in range(ni)])
    gy = np.column_stack([0.5 * (U[2 * ids[:-1] + 1, k] + U[2 * ids[1:] + 1, k])
                          for k in range(ni)])
    geo = gx * nx_hat + gy * ny_hat
    return dict(
        mesh=mesh, ts=ts, clamp=clamp, ds=ds, e_ref=e_ref, thick=thick,
        sigma_map=sigma_map,
        C=C, S_e=0.5 * (S_e + S_e.T), U=U, geometric=geo,
        geometric_gap=float(np.abs(geo - C).max() / np.abs(C).max()),
        symmetry=float(np.abs(U.T @ (ts.K_me @ U) / ds - (U.T @ (ts.K_me @ U) / ds).T).max()
                       / np.abs(C).max()),
        cond=float(np.linalg.cond(C)),
    )


@dataclass
class WingStructure:
    """The elasticity agent: `ThermoStruct2D.solve_mechanical`, unmodified.

    A cantilevered flap -- clamped at the leading edge, free at the trailing
    edge -- which is what an F1 front-wing element is and is the standard
    aeroelastic idealization of one.

    **It is `EMBEDDED` and there is no `split-step` variant.**  Quasi-static
    elasticity has no time derivative, so there is nothing to sub-step: the move
    `window_ns` and `brake_thermal` both make -- expose the elliptic part and take
    stable explicit sub-steps beside it -- has no analogue.  This agent is
    `EMBEDDED` or it is not an agent, and W114 is the rule that had to be settled
    before the graph could be built.  See `R10_HANDLE`.

    **Its response is AFFINE in the trace and its base is on the record.**  The
    structure relates traction to DISPLACEMENT and the port's flow half is a
    velocity, so over one exchange interval

        delta_new = delta + dt <v>,      effort = S_e delta_new

    which is affine.  `probe.base_sensitivity` clears an affine response outright;
    the base is declared anyway, because the plate's current deflection rate is a
    state the expert is in and zero is not it once the wing is loaded.
    """

    agent_id: str = "STRUCT"
    e_star: float = E_STAR
    dt: float = MACRO_DT / EXCHANGES
    n_station: int = N_STATION
    #: The plate's thickness.  CS-12 runs at the declared `THICK` and never moves
    #: it; PoC 2 makes it the second structural design knob, so it is a field
    #: rather than a module constant.  Strictly additive -- the default is the
    #: value CS-12 was measured at.
    thick: float = THICK
    delta: np.ndarray = field(default=None, repr=False)
    delta_dot: np.ndarray = field(default=None, repr=False)
    n_calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._op = _surface_operator(thick=self.thick)
        if self.delta is None:
            self.delta = np.zeros(self.n_station)
        if self.delta_dot is None:
            self.delta_dot = np.zeros(self.n_station)
        self.ds = self._op["ds"]

    # -- the surface operator, exactly linear in E -------------------------

    @property
    def scale(self) -> float:
        return self.e_star / self._op["e_ref"]

    @property
    def stiffness(self) -> np.ndarray:
        """``S_e`` at this stiffness.  Exactly linear in ``E`` -- asserted."""
        return self._op["S_e"] * self.scale

    @property
    def compliance(self) -> np.ndarray:
        return self._op["C"] / self.scale

    def deflect(self, traction: np.ndarray) -> np.ndarray:
        """The expert's OWN answer: the deflection under a station traction.

        Calls `solve_mechanical`.  `FSIRollout` uses ``compliance @ traction``
        instead, which is the same number to the bit for a linear structure and
        is what makes the march differentiable in ``E``; this method is what that
        claim is asserted against.
        """
        op = self._op
        p = 0.5 * np.asarray(traction, dtype=float) / self.scale
        T = np.full(op["mesh"].n_nodes, T_REF)
        u, _ = op["ts"].solve_mechanical(T, p, -p, T_ref=T_REF,
                                         clamp_nodes=op["clamp"])
        return (op["U"].T @ (op["ts"].K_me @ u.reshape(-1))) / op["ds"]

    def stress(self, delta: np.ndarray | None = None) -> np.ndarray:
        """The expert's own stress field at a deflection, for the record."""
        op = self._op
        d = self.delta if delta is None else np.asarray(delta, dtype=float)
        p = 0.5 * (self.stiffness @ d) / self.scale
        T = np.full(op["mesh"].n_nodes, T_REF)
        _, sig = op["ts"].solve_mechanical(T, p, -p, T_ref=T_REF,
                                           clamp_nodes=op["clamp"])
        return sig

    # -- the port ---------------------------------------------------------

    def probe_base(self, name: str) -> np.ndarray:
        """The plate's CURRENT deflection rate, per station.  Not zeros."""
        return np.asarray(self.delta_dot, dtype=float).reshape(-1)

    def respond(self, name: str, trace: np.ndarray) -> np.ndarray:
        """(port, trace) -> the reaction traction the structure presents.

        The trace is the wetted surface's NORMAL velocity; the flux is the NORMAL
        traction per unit area it takes to hold the surface there after one
        exchange interval.  Both are the `MECH` bond's own halves and the response
        is the EFFORT one.
        """
        self.n_calls += 1
        if name != "wet:MECH":
            raise ValueError(f"{self.agent_id} has no port {name!r}")
        trace = np.asarray(trace, dtype=float).reshape(-1)
        if trace.shape[0] != self.n_station:
            raise ValueError(f"trace has {trace.shape[0]} stations, "
                             f"expected {self.n_station}")
        return self.stiffness @ (self.delta + self.dt * trace)

    def storage(self, u: Any = None, v: Any = None) -> float:
        """``1/2 delta^T S_e delta ds``: the strain energy, and the only energy
        this agent owns.  ``S_e`` is SPD, so the response is passive and E7's
        branch is real rather than declared."""
        d = self.delta if u is None else np.asarray(u, dtype=float).reshape(-1)
        return 0.5 * float(d @ (self.stiffness @ d)) * self.ds

    def validity(self, state: Any = None, cond: Any = None) -> bool:
        """Small strain, which is the constitutive law's own hypothesis."""
        d = self.delta if state is None else np.asarray(state, dtype=float)
        return bool(np.all(np.isfinite(d)) and np.max(np.abs(d)) <= DELTA_MAX)


# ---------------------------------------------------------------------------
# the fluid agent's boundary channel
# ---------------------------------------------------------------------------

_FACE_GEOM = {"xlo": (-1.0, "x"), "xhi": (+1.0, "x"),
              "ylo": (-1.0, "y"), "yhi": (+1.0, "y")}

_RING = {
    "xlo": ((slice(None), 0), (slice(None), 1)),
    "xhi": ((slice(None), -1), (slice(None), -2)),
    "ylo": ((0, slice(None)), (1, slice(None))),
    "yhi": ((-1, slice(None)), (-2, slice(None))),
}


@dataclass
class FSIFlowWindow:
    """One window of the composed column, exposed as a boundary response.

    `ground_effect.FlowWindow` with the wing port's trace and geometry made
    vectorial.  The four artificial faces are ordinary fluid-fluid `MECH` seams;
    ``wet:MECH`` is the FSI one, and both of its sides are now fields.

    The window's own corner is subtracted from BOTH plate coordinates.  Getting
    the x one wrong is **W130**, measured on CS-10: the plate sat outside the
    window's array, `grid_sample`'s border clamp returned a constant, and the
    probed fluid block came back exactly zero with every other diagnostic healthy.
    A test asserts the block is nonzero here for that reason.
    """

    agent_id: str
    u0: np.ndarray
    v0: np.ndarray
    shared_faces: tuple[str, ...]
    has_wing: bool = False
    ox: int = 0
    oy: int = 0
    delta: np.ndarray = field(default=None, repr=False)
    dt: float = MACRO_DT / EXCHANGES
    nu: float = NU
    flux_mode: str = "reaction"
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
        self._wing = (FlexWing(x_le=X_LE - self.ox * DX,
                               y_mount=Y_MOUNT - self.oy * DX,
                               device=self.device)
                      if self.has_wing else None)
        if self.delta is None:
            self.delta = np.zeros(N_STATION)
        opt = dict(dtype=TORCH_DTYPE, device=self.device)
        self._u = torch.as_tensor(self.u0, **opt)
        self._v = torch.as_tensor(self.v0, **opt)
        self._d = torch.as_tensor(np.asarray(self.delta, dtype=float), **opt)

    @staticmethod
    def face_of(port_name: str) -> str:
        return port_name.split(":", 1)[0]

    def _step(self, u, v, force=None):
        u1, v1 = self._solver.step_batch(u[None], v[None], self.dt,
                                         bc0=None, force=force)
        return u1[0], v1[0]

    def probe_base(self, name: str) -> np.ndarray:
        """Zero on every port, ABSOLUTE on the wetted one.

        The four fluid faces write the trace into the ring as a perturbation --
        `window_ns`'s convention, so zero is both the origin and a state the
        expert is in.  The wetted face's trace is the surface's vertical velocity,
        written ABSOLUTELY, so zero is the wing at rest: a real physical state,
        and the one the seam base is declared at.  The structure declares its own
        current deflection rate, so the two sides disagree once the wing is
        loading -- which `L4/probe-base` is for and which is reported rather than
        hidden (W74's class).
        """
        if name == "wet:MECH":
            return np.zeros(N_STATION)
        return np.zeros(self.nx_c if self.face_of(name) in ("ylo", "yhi") else self.ny_c)

    def _wet_effort(self, u, v, w_plate):
        """The normal traction on the plate, in the declared flux mode.

        The three forms are `ground_effect.FLUX_MODES` carried onto the normal
        pairing.  ``reaction`` is the exact discrete momentum exchange and is what
        the march uses; ``diffusive`` is `probed-dtn-coupling` §2.2's normative
        effort ``nu dw/dn``; ``conormal`` is §4.1's conservative form
        ``nu dw/dn - (u.n) w``, which **W47 scoped out at fluid-fluid seams**
        because the advective term cancels across a shared ring cell and which
        **W97 closed a field-to-lumped seam with**.  Whether it is still the one
        that separates the blocks at a field-to-FIELD seam is a question only this
        case study can ask.
        """
        wing = self._wing
        w, um, vm, us, vs = wing.station_normal(u, v, self._d, w_plate,
                                                self.ny_c, self.nx_c)
        if self.flux_mode == "reaction":
            return wing.normal_traction(w)
        dwdn = ((us[0] - us[1]) * wing.n_hat[0]
                + (vs[0] - vs[1]) * wing.n_hat[1]) / (2.0 * D_OFFSET * DX)
        diff = self.nu * dwdn
        if self.flux_mode == "diffusive":
            return diff
        w_ext = um * wing.n_hat[0] + vm * wing.n_hat[1]
        return diff - w_ext * w_ext

    def respond(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        """(port, trace) -> flux.  The only callable the probe needs."""
        self.n_calls += 1
        trace = np.asarray(trace, dtype=float).reshape(-1)
        opt = dict(dtype=TORCH_DTYPE, device=self.device)

        if port_name == "wet:MECH":
            if not self.has_wing:
                raise ValueError(f"{self.agent_id} carries no wing")
            if trace.shape[0] != N_STATION:
                raise ValueError(f"wet trace has {trace.shape[0]} stations, "
                                 f"expected {N_STATION}")
            vp = torch.as_tensor(trace, **opt)
            fx, fy, _, _ = self._wing.forcing(self._u, self._v, self._d, vp,
                                              self.ny_c, self.nx_c)
            u1, v1 = self._step(self._u, self._v, force=(fx[None], fy[None]))
            return self._wet_effort(u1, v1, vp).detach().cpu().numpy()

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

    def storage(self, u: Any = None, v: Any = None) -> float:
        if u is None:
            u, v = self.u0, self.v0
        u = np.asarray(u, dtype=float)
        v = self.v0 if v is None else np.asarray(v, dtype=float)
        return 0.5 * float(np.sum(u * u + v * v)) * self.h_solver**2

    def validity(self, state: Any = None, cond: Any = None) -> bool:
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


def flow_ports(expert: FSIFlowWindow, motion: MotionClass) -> list:
    ports = []
    for f in expert.shared_faces:
        n_cells = expert.nx_c if f in ("ylo", "yhi") else expert.ny_c
        name = f"{f}:MECH"
        ports.append(port_decl(
            name=name, port_type=PortType.MECH,
            geometry=f"{f} ring of {expert.agent_id} ({n_cells} cells)",
            direction=Direction.BIDIRECTIONAL, nondim=dict(MECH_SCALES),
            effective_resolution=modes_for(n_cells),
            motion_class=MotionClass.STATIC,
            response_half=ResponseHalf.EFFORT,
            prolongation=_prolongation(expert.agent_id, name, n_cells)))
    if expert.has_wing:
        ports.append(port_decl(
            name="wet:MECH", port_type=PortType.MECH,
            geometry=f"the wetted surface, {N_STATION} stations inside "
                     f"{expert.agent_id}",
            direction=Direction.BIDIRECTIONAL, nondim=dict(MECH_SCALES),
            effective_resolution=modes_for(N_STATION),
            motion_class=motion,
            response_half=ResponseHalf.EFFORT,
            prolongation=_prolongation(expert.agent_id, "wet:MECH", N_STATION),
            note="the plate-NORMAL traction on the wetted surface: the MECH "
                 "EFFORT, against an imposed surface normal velocity (the FLOW)"))
    return ports


def flow_capabilities(expert: FSIFlowWindow,
                      motion: MotionClass = MotionClass.STATIC,
                      substeps: int = EXCHANGES) -> ExpertCapabilities:
    return ExpertCapabilities(
        expert_id=expert.agent_id,
        ports=flow_ports(expert, motion),
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=True,
        elliptic_subsolve=EllipticSubsolve.EXPOSED,
        stencil_radius=2,
        substeps_per_macro_step=substeps,
        time_discretization=TimeDiscretization.EXPLICIT,
        differentiable=Differentiable.NONE,
        dt_native=MACRO_DT,
        L_native=expert.nx_c * DX,
        storage=expert.storage,
        equivariances=(),
        validity=expert.validity,
        governing_family="incompressible-navier-stokes-2d",
        lambda_ref="itself: reference.WindowNS, exposed, at this cell",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash=f"windowns-exposed-nu{expert.nu:.6g}-n{expert.nx_c}"
                    f"x{expert.ny_c}-{expert.flux_mode}",
        boundary_response=expert.respond,
        probe_base=expert.probe_base,
        reproducibility_floor=np.finfo(float).eps,
        deterministic=True,
        note="reference.WindowNS with its elliptic part removed, torch float64, "
             "tape attached; CS-10's fluid column unchanged")


def structure_capabilities(expert: WingStructure,
                           motion: MotionClass = MotionClass.STATIC
                           ) -> ExpertCapabilities:
    """The elasticity agent's record.

    **`governing_family` is genuinely different from the flow's**, which is what
    §4's CS-12 row is about and which makes `L1/E3` fail at the wetted seam.  Both
    sides therefore declare `lambda_ref`, and both are real solvers whose
    reference is themselves, so ``tau = 0`` rather than ``tau = UNDEFINED`` --
    `CASE-STUDY-GUIDE`'s multiphysics box exactly.
    """
    return ExpertCapabilities(
        expert_id=expert.agent_id,
        ports=[port_decl(
            name="wet:MECH", port_type=PortType.MECH,
            geometry=f"the wetted surface of the plate, {N_STATION} face elements",
            direction=Direction.BIDIRECTIONAL, nondim=dict(MECH_SCALES),
            effective_resolution=modes_for(N_STATION),
            motion_class=motion,
            response_half=ResponseHalf.EFFORT,
            prolongation=_prolongation(expert.agent_id, "wet:MECH", N_STATION),
            note="quasi-static: the imposed velocity is a displacement increment "
                 "v dt and the response is the reaction traction S_e delta")],
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=True,
        # Quasi-static: a global linear solve with no time step to shrink, and NO
        # split-step variant -- there is nothing to sub-step.  EMBEDDED in R10's
        # exact sense, and W114 is why that is not a refusal here: this graph does
        # not cut Omega_solid.  See R10_HANDLE.
        elliptic_subsolve=EllipticSubsolve.EMBEDDED,
        time_discretization=TimeDiscretization.IMPLICIT,
        stencil_radius=1,                       # Q1 elements, one ring
        substeps_per_macro_step=1,
        differentiable=Differentiable.NONE,
        dt_native=MACRO_DT,
        L_native=CHORD,
        storage=expert.storage,
        equivariances=(),
        validity=expert.validity,
        governing_family="plane-stress-elasticity-2d",
        lambda_ref="thermostruct2d.ThermoStruct2D.solve_mechanical, same mesh; "
                   "the real solver is its own reference",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash=f"thermostruct2d/1-elasticity-quasistatic-E{expert.e_star:.6g}",
        boundary_response=expert.respond,
        probe_base=expert.probe_base,
        reproducibility_floor=float(np.finfo(float).eps),
        deterministic=True,
        note="solvers/thermostruct2d.py, build repo, imported unmodified; the "
             "quasi-static plane-stress half, clamped at the leading edge, with "
             "alpha = 0 and T = T_ref so the thermal load is identically zero")


# ---------------------------------------------------------------------------
# what the case study records about itself
# ---------------------------------------------------------------------------

#: **W114, settled 2026-09-04 BEFORE the graph was built rather than during it.**
#:
#: `L2/R10` refused any graph with two agents one of which declares
#: ``elliptic_subsolve=embedded``, and its sentence is *"the graph decomposes the
#: domain ... so the decomposition changes the operator rather than restricting
#: it"*.  A quasi-static structural agent is EMBEDDED and has no `split-step`
#: escape -- no time derivative, nothing to sub-step -- so the rule refused every
#: FSI graph before this one, for a reason that is not about the coupling.
#:
#: The premise the rule needs is that the decomposition CUTS THE AGENT.  Here it
#: does not: ``Omega_solid`` is one agent's whole domain and the elasticity solve
#: runs over exactly the region a monolithic FSI solve would run it over.  Only
#: the FLUID is tiled, and every fluid agent declares ``exposed``.
#:
#: The handle, decidable from the declarations alone: an EMBEDDED agent is
#: refused only when the graph contains **another agent of the same
#: `governing_family`** -- which is the compile-time signature of "a larger region
#: of the same physics that this agent has been given a piece of".  Every graph
#: R10 was derived on is a window tiling and keeps its refusal; the two graphs
#: whose agents each own their own physics -- CS-9's co-located split and this
#: FSI seam -- stop being refused for a reason that never applied to them.
#:
#: It is a **proxy** and it is conservative in the safe direction: two agents of
#: one family on genuinely disjoint regions would still be refused, which is a
#: false refusal and not a silent admission.  See `atlas/compiler.py::_r10_elliptic`.
R10_HANDLE = (
    "L2/R10's premise is that the decomposition cuts the agent's own domain. In "
    "an FSI graph the fluid is tiled and the structure is not: Omega_solid is one "
    "agent's whole domain and the quasi-static elasticity solve is over exactly "
    "the region a monolithic FSI solve would run it over. Since 2026-09-04 R10 "
    "consults whether the graph contains another agent of the same "
    "governing_family before refusing an EMBEDDED one -- the compile-time "
    "signature of a decomposition that cut this agent's physics -- so a window "
    "tiling is still refused and a sole-of-its-family elliptic agent is not. "
    "W114, closed. The predicate is a proxy and it is conservative: two agents of "
    "one family on disjoint regions are still refused"
)

#: What the structural model is, so no number here is read as being about wings.
STRUCTURE_SCOPE = (
    "A 32x2 Q1 plane-stress cantilever, clamped at the leading edge, two elements "
    "through a thickness of 0.04 chords. Q1 elements lock in bending, so the "
    "plate is stiffer than its own beam theory and E_STAR is calibrated to a "
    "deflection rather than taken from a material -- which is legitimate because "
    "E_STAR is declared in flow units (E / rho U^2) and is a design knob, and "
    "which is why no stress figure here is a statement about an alloy. The seam's "
    "generalized displacement is the WORK-conjugate of its traction and is "
    "identified with the plate line's position in the fluid; the geometric "
    "mid-plane displacement differs from it by 1.3% of the compliance norm"
)

#: **W157, measured 2026-09-08 on THIS graph** -- `scripts/w157_frontwing_constants.py`,
#: artifact `out/w157/w157.json`.  Before it, `L`, `sigma` and `C_mu` were
#: measured only in tier 0, on `window_ns`, at another state and another scheme,
#: and were therefore *not* declared here -- so `W56`'s backstop forced
#: `admit-uncertified` on every compile of this graph.  [[poc2-novelty-audit]]
#: section 3 recorded that the compiler had consequently never certified
#: anything, and that the reason was **provenance rather than physics**.
#:
#: What each number is, and the one that is not this graph's own:
#:
#:   ``L``      paired march from the settled state, one column perturbed by a
#:              divergence-free streamfunction blob at 1e-3 of max|u|, least
#:              squares on log||e^n|| over 60 macro-steps.  Both columns land on
#:              the **contractive** branch, so ``T_pred`` is unbounded.  The
#:              error decays only 7% over the horizon, which is a slow rate
#:              honestly fitted (rms log residual 0.000) and not a strong one.
#:
#:   ``sigma``  the transmission error at the DECLARED ramp: the same macro-step
#:              run with the artificial ring held at t^n against the same step
#:              with the ring ramped to the single-window referent's.  The
#:              positive control is the referent through the identical harness,
#:              where there is no artificial face and sigma is **exactly 0**, and
#:              it caught a real defect first: replacing a window's WHOLE ring
#:              also overwrites the domain boundary, which reported
#:              sigma = 6.86e-5 on one window and the same 6.86e-5 on six.
#:
#:   ``C_mu``   **is tier 0's 1.2 and is deliberately not this graph's 0.56.**
#:              Swept over six partitions of unity x two columns, the implied
#:              constant here runs [0.0031, 0.5504] -- a spread of **179x**
#:              while sigma moves 965x.  A constant that moves 179x across the
#:              sample has not been bounded by it, so the sampled maximum is not
#:              adopted; 1.2 comes from W49's sixteen configurations and holds
#:              on all twelve here with 2.2x of margin.  Declaring the looser
#:              number is the conservative direction and the honest one.
#:              **The 179x is itself the finding**: Pi moves only 0.75 -> 0.70
#:              across the sweep while sigma moves 965x, so on this graph the
#:              INDICATOR does not carry sigma's variation -- the partition of
#:              unity does.  That is [[interaction-horizon]]'s case for Pi_w
#:              arriving on a second graph, and it is tracked as **W158**.
#: **W159, measured 2026-09-09.**  `L2/C2`'s own quantity on THIS graph, in
#: the chi-weighted form -- the one the derivation bounds the composed defect
#: by, and the one that needs a monolith.  The monolith exists here because
#: `solver_for` builds the same class at 80x80 and at 208x144, so ``E`` is the
#: identical code path over one window covering the domain.
#:
#: Three controls, and the first two are the ones that could have failed:
#: the identity ``A({E_i R_i u}) - E u = sum_i R_i^T chi_i D_i`` closes to
#: **1.0e-10 relative** with SIGNED D_i, which is what says the lift, the
#: weights, the cut and the forcing are all right; the zero-cut column on
#: `SINGLE_TILING`, marched the same 80 exchanges, gives **exactly 0**; and two
#: identical runs differ by **exactly 0**, so the floor is bitwise and the
#: bound is signal.
#:
#: Declared as the MAX over 80 exchanges (20 macro-steps from the settled
#: state), which is the conservative direction for a bound.  The spread is
#: 1.003x fixed-shape and 1.178x aeroelastic, so this is a genuinely flat
#: quantity here rather than a number sampled at one lucky state.
#:
#: **The bound is tight to 0.05%** (5.6225e-05 against a composed defect of
#: 5.6199e-05), reproducing tier 0's 0.2% on a graph it was not fitted on; the
#: max form is **229x** looser at 1.285e-02, which is the partition of unity's
#: whole contribution.
CUT_DEFECT_STATIC = 5.622488e-05
CUT_DEFECT_MOVING = 6.054872e-05

MEASURED_STATIC = MeasuredConstants(
    tau=0.0, gamma=0.0, norm_A=1.0,
    L=0.998719, L_stderr=0.000001,
    sigma=4.534925e-08,
    C_mu=1.2,
    cut_defect_bound=CUT_DEFECT_STATIC,
    cut_defect_bound_form="chi-weighted",
    probe_state=f"settled fixed-shape wake at delta = 0, E* = {E_STAR:.3g}, "
                f"dt = {MACRO_DT}, nu = {NU}; L over 60 macro-steps from "
                f"out/w141/settled.npz, sigma over one",
    scheme=f"exposed fluid agents + ProjectedAssembly, halo {HALO}, ramp {RAMP}, "
           f"{EXCHANGES} exchanges per macro-step, wing shape FROZEN",
    depth=0,
    source="tau/gamma/norm_A: scripts/w136_wing_fsi.py, out/w136/. "
           "L/sigma/C_mu: scripts/w157_frontwing_constants.py, out/w157/ "
           "(C_mu is W49's 1.2, validated here 12/12, not this graph's implied "
           "0.56 -- see the note above). "
           "cut_defect_bound: scripts/w159_frontwing_cut_defect.py, out/w159/, "
           "chi-weighted form against the single-window monolith, max over 80 "
           "exchanges")

MEASURED_MOVING = MeasuredConstants(
    tau=0.0, gamma=0.0, norm_A=1.0,
    L=0.998735, L_stderr=0.000002,
    sigma=3.173377e-07,
    C_mu=1.2,
    cut_defect_bound=CUT_DEFECT_MOVING,
    cut_defect_bound_form="chi-weighted",
    probe_state=f"settled aeroelastic state at E* = {E_STAR:.3g}, "
                f"dt = {MACRO_DT}, nu = {NU}; L over 60 macro-steps from "
                f"out/w141/settled.npz, sigma over one",
    scheme=f"exposed fluid agents + ProjectedAssembly, halo {HALO}, ramp {RAMP}, "
           f"{EXCHANGES} exchanges per macro-step, the seam's own Newton solved "
           f"at every exchange",
    depth=0,
    source="tau/gamma/norm_A: scripts/w136_wing_fsi.py, out/w136/. "
           "L/sigma/C_mu: scripts/w157_frontwing_constants.py, out/w157/ "
           "(C_mu is W49's 1.2, validated here 12/12, not this graph's implied "
           "0.56 -- see the note above). "
           "cut_defect_bound: scripts/w159_frontwing_cut_defect.py, out/w159/, "
           "chi-weighted form against the single-window monolith, max over 80 "
           "exchanges")


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------


def make_experts(u_full: np.ndarray, v_full: np.ndarray,
                 tiling: WingTiling = DEFAULT_TILING,
                 delta: np.ndarray | None = None,
                 e_star: float = E_STAR,
                 flux_mode: str = "reaction",
                 nu: float = NU) -> dict[str, Any]:
    """Cut the field into windows and build the structure the wing is."""
    u_full = np.asarray(u_full, dtype=float)
    v_full = np.asarray(v_full, dtype=float)
    if u_full.shape != (tiling.ny, tiling.nx):
        raise ValueError(f"field is {u_full.shape}, expected "
                         f"{(tiling.ny, tiling.nx)}")
    delta = np.zeros(N_STATION) if delta is None else np.asarray(delta, dtype=float)
    us, vs = tiling.cut(u_full), tiling.cut(v_full)
    wing_window = tiling.wing_window()
    out: dict[str, Any] = {}
    for k, name in enumerate(tiling.names):
        ox, oy = tiling.offsets[k]
        out[name] = FSIFlowWindow(
            agent_id=name, u0=us[k], v0=vs[k],
            shared_faces=tiling.artificial_faces(ox, oy),
            has_wing=(name == wing_window), ox=ox, oy=oy,
            delta=delta, nu=nu, flux_mode=flux_mode)
    out["STRUCT"] = WingStructure(e_star=e_star, delta=delta.copy())
    return out


def connections(tiling: WingTiling = DEFAULT_TILING) -> list[Connection]:
    """The fluid-fluid seams from the layout, plus the FSI one."""
    conns: list[Connection] = []
    for j in range(tiling.n_row):
        for i in range(tiling.n_col - 1):
            a, b = f"F{i}{j}", f"F{i+1}{j}"
            conns.append(Connection(
                seam_id=f"x{i}{j}", a=(a, "xhi:MECH"), b=(b, "xlo:MECH"),
                port_type=PortType.MECH, derive_space=True,
                geometrically_coincident=False,
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
        seam_id="wet", a=(tiling.wing_window(), "wet:MECH"),
        b=("STRUCT", "wet:MECH"),
        port_type=PortType.MECH, derive_space=True,
        geometrically_coincident=True,
        # n_0 = 0, and the reason is a FIFTH row for the table.  CS-9's
        # solid-solid MECH seam on a FREE body has n_0 = 1 because a uniform
        # normal velocity there is a rigid translation -- no strain, no reaction.
        # This body is CLAMPED at the leading edge, so a uniform surface velocity
        # bends it and the reaction is not zero in any direction of the trace
        # space.  It is the boundary condition and not the physics.
        expected_null_dim=0,
        # **W138, declared 2026-09-08.**  Both sides return their EFFORT in the
        # SAME direction -- the fluid the aerodynamic load ON the surface, the
        # structure the elastic reaction that holds it -- rather than each
        # against its own outward normal.  So the well-posed interface condition
        # is their DIFFERENCE, which is what `solve_interface` writes:
        # `S_e(delta + dt w) - f_aero = 0`.  Naming STRUCT keeps the structure's
        # block and negates the fluid's, which is the operator the residual is
        # actually differentiated through.
        #
        # **Checked, not asserted.**  The case's own analytic Jacobian
        # `dR/dw = dt S_e + diag(g)`, reproduced by a central difference to
        # 1.15e-16 relative, is positive definite at lambda_min = +3.469 where
        # the unoriented sum is indefinite at -3.488.
        effort_normal="STRUCT",
        note="THE seam: fluid-structure MECH across two governing families, "
             "two-way, and its geometry is a function of the solution"))
    return conns


def build(u_full: np.ndarray,
          v_full: np.ndarray,
          motion: bool = False,
          tiling: WingTiling = DEFAULT_TILING,
          delta: np.ndarray | None = None,
          e_star: float = E_STAR,
          flux_mode: str = "reaction",
          nu: float = NU,
          measured: MeasuredConstants | None = "default",
          experts: dict[str, Any] | None = None) -> tuple[CaseGraph, dict[str, Any]]:
    """The graph at a state, with the wetted surface frozen or deforming.

    ``motion=False``  the fixed-shape control: the plate is pinned at ``delta``
                      and the seam is an ordinary fluid-structure `MECH` seam.
                      This is the column the drift number is compared against.
    ``motion=True``   the wing's shape is the structure's output, so the seam's
                      geometry is a function of the solution.  **L2 refuses**, at
                      `InterfaceMotion`, and the refusal is correct and priced:
                      CS-10 supplied the three measurements the hole asks for and
                      this graph inherits them unchanged.
    """
    mc = MotionClass.SOLUTION_DEPENDENT if motion else MotionClass.STATIC
    if measured == "default":
        measured = MEASURED_MOVING if motion else MEASURED_STATIC
    experts = experts or make_experts(u_full, v_full, tiling, delta, e_star,
                                      flux_mode, nu)
    agents = [Agent(n, flow_capabilities(experts[n], mc), domain=f"fluid window {n}")
              for n in tiling.names]
    agents.append(Agent("STRUCT", structure_capabilities(experts["STRUCT"], mc),
                        domain="Omega_solid: the whole wing section",
                        role="structure"))
    return (
        CaseGraph(
            name=f"wing-fsi-{tiling.n_col}x{tiling.n_row}-"
                 f"{'deforming' if motion else 'fixed-shape'}",
            agents=agents,
            connections=connections(tiling),
            decomposition=Decomposition.OVERLAPPING,
            overlap=tiling.halo * DX,
            overlap_cells=tiling.halo,
            partition_of_unity=projected_assembly(tiling),
            global_fields=[GlobalField(
                "pressure",
                # W117: produced by every FLUID agent. The structure is not in the
                # list because an elastic body has no pressure field of the flow's
                # kind, and a proper subset applied outside itself is what
                # L3/global-field refuses -- `applies_to` is the same set.
                produced_by=tuple(tiling.names),
                applies_to=tuple(tiling.names),
                note="the elliptic part, in the composition layer where R10 puts "
                     "it: one global spectral Leray projection on the assembled "
                     "fluid field, once per exchange"),
            ],
            cross_points=() if tiling.is_single else ("mid",),
            macro_dt=MACRO_DT,
            measured=measured,
            note=(f"{tiling.n_windows} exposed reference.WindowNS windows of "
                  f"{tiling.wx}x{tiling.wy} cells tiling {NX}x{NY} at h = 1/64, "
                  f"halo {tiling.halo}; a {NI_STRUCT}x{NJ_STRUCT} Q1 plane-stress "
                  f"cantilever at E* = {e_star:.3g} carrying a porous inclined "
                  f"plate at {math.degrees(ALPHA):.0f} deg; the referent is the "
                  f"same pair undivided with the FSI seam tightly coupled")),
        experts,
    )


# ---------------------------------------------------------------------------
# the march -- one code path, and the referent is a tiling of it
# ---------------------------------------------------------------------------


@dataclass
class FSIResult:
    """What one march returns.  Every trace is per macro-step."""

    load: np.ndarray                  # downforce per unit span
    tip: np.ndarray                   # trailing-edge deflection
    delta: np.ndarray                 # [steps, N_STATION]
    delta_dot: np.ndarray             # [steps, N_STATION]
    strain_energy: np.ndarray
    interface_power: np.ndarray
    #: The half-step energy term, per macro-step, as a rate.  See `macro_step`.
    half_step: np.ndarray
    u_max: np.ndarray
    div_rms: np.ndarray
    substeps: np.ndarray
    inner_residual: np.ndarray
    newton_drop: np.ndarray
    #: The strain energy at RELEASE, so `R` differences the marched history
    #: against a real starting value rather than a one-sided stencil.
    energy_0: float = 0.0
    u: Any = None
    v: Any = None
    delta_final: Any = None
    wall_s: float = 0.0
    fields: dict = field(default_factory=dict)

    def settled(self, frac: float = 0.25) -> tuple[float, float]:
        """(mean load, mean tip deflection) over the last ``frac`` of the march."""
        n = max(1, int(round(frac * self.load.size)))
        return float(self.load[-n:].mean()), float(self.tip[-n:].mean())


class FSIRollout:
    """The composed march, differentiable in ``E*``, with a PINNED window order.

    One code path serves three things, exactly as `ground_effect.GroundRollout`
    does, which is what makes the controls mean anything:

      ``WingTiling()``          the composed column -- six windows, thirteen
                                seams, blend and project every exchange
      ``WingTiling.single()``   the REFERENT -- one window covering the whole
                                domain, a partition of unity identically one, and
                                the projection applied to a field nothing was
                                blended into.  The two differ by the CUT alone.
      ``coupling='tight'``      the wetted surface's own interface equation is
                                solved at EVERY exchange; ``'lagged'`` solves it
                                once every ``lag`` macro-steps; ``'staggered'``
                                does not solve it at all and reads the structure's
                                constitutive law as an UPDATE, which is the
                                partitioned-FSI added-mass scheme.
    """

    def __init__(self, tiling: WingTiling = DEFAULT_TILING,
                 e_star: float = E_STAR, nu: float = NU, device: str = "cpu",
                 coupling: str = "lagged", motion: bool = True,
                 lag: int = 1, n_inner: int = 3,
                 checkpoint: bool = True) -> None:
        if coupling not in ("tight", "lagged", "staggered"):
            raise ValueError(coupling)
        self.lag = max(1, int(lag))
        self.tiling = tiling
        self.nu = nu
        self.device = device
        self.coupling = coupling
        self.motion = motion
        self.n_inner = n_inner
        self.checkpoint = checkpoint
        self.ny, self.nx = tiling.ny, tiling.nx
        self.solver = solver_for(tiling.wx, tiling.wy, nu, device)
        self.wing = FlexWing(device=device)
        self.dt_ex = MACRO_DT / EXCHANGES
        opt = dict(dtype=TORCH_DTYPE, device=device)

        op = _surface_operator()
        self.e_ref = op["e_ref"]
        #: ``S_e`` at unit ``E``.  Linear elasticity is exactly linear in ``1/E``
        #: at fixed ``nu`` (asserted to the bit in the tests), so the whole march
        #: is differentiable in the stiffness through one constant matrix and the
        #: FE solver is never re-run inside a rollout.
        self.S_unit = torch.as_tensor(op["S_e"] / op["e_ref"], **opt)
        self.C_unit = torch.as_tensor(op["C"] * op["e_ref"], **opt)
        self.ds = op["ds"]
        self.e_star_default = float(e_star)

        offs = tiling.offsets
        self.n_win = len(offs)
        self._offsets = offs
        chi = np.empty((self.n_win, tiling.wy, tiling.wx))
        w = tiling.weights()
        for kk, (ox, oy) in enumerate(offs):
            chi[kk] = w[kk][oy:oy + tiling.wy, ox:ox + tiling.wx]
        self._chi = torch.as_tensor(chi, **opt)

        m = np.zeros((self.ny, self.nx), dtype=bool)
        m[:, :BAND] = True
        m[-BAND:, :] = True
        m[0, :] = True
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

    def stiffness(self, e_star):
        return self.S_unit * e_star

    def aero_traction(self, u, v, delta, w_plate):
        """The fluid's EFFORT half at the seam: normal traction per station."""
        w, _, _, _, _ = self.wing.station_normal(u, v, delta, w_plate,
                                                 self.ny, self.nx)
        return self.wing.normal_traction(w), w

    def solve_interface(self, u, v, delta, w_plate, e_star, interval=None):
        """The seam's own equation, solved for the surface's normal velocity.

        **§4.2's rule at a field-to-FIELD seam.**  The residual is the two agents'
        `boundary_response`s balanced in the port's own flow variable, and it is a
        VECTOR of `N_STATION` equations where CS-10's was one scalar:

            R(w) = S_e (delta + dt w)  -  f_aero(u, delta, w)  =  0

        The Jacobian is analytic and needs no second field evaluation, for
        CS-10's reason -- the external flow at the stations does not depend on the
        surface's own velocity:

            dR/dw = dt S_e  +  diag(C_N |w_rel|)

        and both terms are SPD, so ``J`` is SPD at every state and the Newton
        system is a `N_STATION`-square solve rather than a scalar division.  The
        second term is the **aerodynamic damping**, which is exactly the term
        §4.2 says the update form throws away and exactly what makes the
        partitioned scheme stable.  At rank one it was a number; here it is a
        diagonal, and the structural half is a dense SPD matrix with a condition
        number of 7.1e8, so this is the first seam in the vault where the
        interface problem is a real linear system.

        Three steps at a FIXED count -- fixed, because the tape has to have the
        same shape at every value of ``E*`` for a reverse-mode gradient to mean
        anything -- and the residual is returned at both ends so the count is
        defended rather than assumed.

        **``interval`` is the interval the answer is HELD over, and it is not
        always the exchange interval.**  A tight coupling re-solves every
        exchange, so it is ``dt_ex``; a coupling lagged ``k`` macro-steps holds
        the velocity for ``k`` of them, so it is ``k * MACRO_DT``.  Passing
        ``dt_ex`` in the lagged case is not a lag -- it is a scheme that solves
        for the velocity that reaches equilibrium in one exchange and then applies
        it for `EXCHANGES` of them, a built-in overshoot of exactly that factor.
        Measured: it diverges at macro-step 9 where the honest lagged scheme is
        stable, and the divergence would have been reported as the added-mass
        instability it is not.  `CASE-STUDY-GUIDE` mistake 5 -- a composition-layer
        cadence wearing an agent's label -- arriving on the interface solve.
        """
        wing = self.wing
        S_e = self.stiffness(e_star)
        dt = self.dt_ex if interval is None else interval
        w_ext = wing.external_normal(u, v, delta, self.ny, self.nx)
        wp = w_plate
        eye = torch.eye(N_STATION, dtype=u.dtype, device=u.device)
        r0 = None
        for _ in range(self.n_inner):
            w = w_ext - wp
            f_aero = wing.normal_traction(w)
            r = S_e @ (delta + dt * wp) - f_aero
            if r0 is None:
                r0 = torch.linalg.vector_norm(r)
            damp = wing.c_n * torch.abs(w)
            J = dt * S_e + damp[:, None] * eye
            wp = wp - torch.linalg.solve(J, r)
        w = w_ext - wp
        res = torch.linalg.vector_norm(
            S_e @ (delta + dt * wp) - wing.normal_traction(w))
        return wp, res, r0

    # -- one exchange, and one macro-step ---------------------------------

    def exchange(self, u, v, delta, w_plate):
        fx, fy, w, load = self.wing.forcing(u, v, delta, w_plate, self.ny, self.nx)
        us, vs = self.cut(u), self.cut(v)
        fxs, fys = self.cut(fx), self.cut(fy)
        u1, v1 = self.solver.step_batch(us, vs, self.dt_ex, bc0=None,
                                        force=(fxs, fys))
        self.substep_log.append(int(self.solver.last_substeps))
        au, av = self.blend(u1, v1)
        au, av = self.project(au, av)
        bu, bv = self.band(au, av)
        return bu, bv, load, self.wing.normal_traction(w)

    def macro_step(self, u, v, delta, w_plate, e_star, step: int = 0):
        """`EXCHANGES` exchanges, and the seam solved at the declared cadence."""
        load = None
        t_aero = None
        res = torch.zeros((), dtype=u.dtype, device=u.device)
        r0 = torch.zeros((), dtype=u.dtype, device=u.device)
        solve_now = (self.motion and self.coupling == "tight")
        lagged_now = (self.motion and self.coupling == "lagged"
                      and step % self.lag == 0)
        hold = self.dt_ex if solve_now else self.lag * MACRO_DT
        #: The interface WORK over this macro-step, accumulated exchange by
        #: exchange.  Sampling the power once per macro-step instead is an O(dt)
        #: error in a quantity R is then asked to close to O(dt^2), and it reads
        #: as a coupling defect: measured, it puts the residual at 7.2e-2 where
        #: the accumulated form puts it two orders lower on the same march.
        #: The power is applied EXCHANGES times per macro-step, so it has to be
        #: read EXCHANGES times per macro-step.
        work = torch.zeros((), dtype=u.dtype, device=u.device)
        #: The HALF-STEP energy term, accumulated beside the work.
        #:
        #: At a converged interface solve the fluid's effort equals the
        #: structure's reaction at the END of the exchange, ``f = S_e delta_new``,
        #: while the structure's energy gain over that exchange is
        #: ``S_e delta_mid . d(delta)``.  Their difference is exactly
        #: ``1/2 d(delta)^T S_e d(delta)`` -- the backward-Euler-against-midpoint
        #: difference of the energy accounting, and NOT a defect of the bond.
        #: It is accumulated rather than argued so `R` can be reported with it
        #: and without it, which is the only way to tell the two apart.
        half = torch.zeros((), dtype=u.dtype, device=u.device)
        S_e = self.stiffness(e_star)
        for i in range(EXCHANGES):
            if solve_now or (lagged_now and i == 0):
                w_plate, res, r0 = self.solve_interface(
                    u, v, delta, w_plate, e_star, interval=hold)
            u, v, load, t_aero = self.exchange(u, v, delta, w_plate)
            work = work + (t_aero * w_plate).sum() * self.ds * self.dt_ex
            if self.motion and self.coupling != "staggered":
                d_delta = self.dt_ex * w_plate
                half = half + 0.5 * (d_delta @ (S_e @ d_delta)) * self.ds
                delta = delta + d_delta
        if self.motion and self.coupling == "staggered":
            # The UPDATE form: move the structure straight to the deflection its
            # own constitutive law wants under the load just computed.  This is
            # the obvious reading of `S_e delta = t_aero` and it is the
            # partitioned-FSI added-mass scheme; §4.2 says it diverges and CS-10
            # measured it diverging at rank one.  Kept because a rule with nothing
            # to fire on is not a rule.
            new = torch.linalg.solve(self.stiffness(e_star), t_aero)
            w_plate = (new - delta) / MACRO_DT
            delta = new
        return (u, v, delta, w_plate, load, t_aero, res, r0,
                work / MACRO_DT, half / MACRO_DT)

    # -- the march ---------------------------------------------------------

    def run(self, e_star=None, steps: int = 40, u0=None, v0=None, delta0=None,
            grad: bool = False, keep_fields: Sequence[int] = (),
            progress: Callable[[int, float], None] | None = None) -> FSIResult:
        import time
        opt = dict(dtype=TORCH_DTYPE, device=self.device)
        e_star = self.e_star_default if e_star is None else e_star
        if not torch.is_tensor(e_star):
            e_star = torch.tensor(float(e_star), **opt)
        u = (torch.full((self.ny, self.nx), U_INF, **opt) if u0 is None
             else torch.as_tensor(np.asarray(u0), **opt))
        v = (torch.zeros((self.ny, self.nx), **opt) if v0 is None
             else torch.as_tensor(np.asarray(v0), **opt))
        delta = (torch.zeros(N_STATION, **opt) if delta0 is None
                 else torch.as_tensor(np.asarray(delta0), **opt))
        w_plate = torch.zeros(N_STATION, **opt)
        self.substep_log = []

        e0 = float((0.5 * (delta @ (self.stiffness(e_star) @ delta))
                    * self.ds).detach())
        loads, tips, deltas, ddots, energies, powers = [], [], [], [], [], []
        halves = []
        umax, divs, ress, drops = [], [], [], []
        fields, keep = {}, set(keep_fields)
        t0 = time.perf_counter()
        for s in range(steps):
            if s in keep:
                fields[s] = (u.detach().cpu().numpy().copy(),
                             v.detach().cpu().numpy().copy(),
                             delta.detach().cpu().numpy().copy())
            if grad and self.checkpoint:
                (u, v, delta, w_plate, load, t_aero, res, r0, p_mean, p_half
                 ) = torch.utils.checkpoint.checkpoint(
                    self.macro_step, u, v, delta, w_plate, e_star, s,
                    use_reentrant=False)
            else:
                (u, v, delta, w_plate, load, t_aero, res, r0, p_mean, p_half
                 ) = self.macro_step(u, v, delta, w_plate, e_star, s)
            dv = delta.detach()
            if self.motion and float(torch.max(torch.abs(dv))) > DELTA_MAX:
                raise RuntimeError(
                    f"the wing left the structural expert's declared envelope at "
                    f"macro-step {s}: max|delta| = "
                    f"{float(torch.max(torch.abs(dv))):.6g} against a small-strain "
                    f"bound of {DELTA_MAX:.6g}. Raised where it happens rather than "
                    f"clamped, because a hard stop is a fitted parameter and this "
                    f"expert has none -- `WingStructure.validity` is the same bound "
                    f"and declines there")
            loads.append(load)
            tips.append(delta[-1])
            deltas.append(delta)
            ddots.append(w_plate)
            energies.append(0.5 * (delta @ (self.stiffness(e_star) @ delta)) * self.ds)
            powers.append(p_mean)
            halves.append(p_half)
            ress.append(res)
            drops.append(res / torch.clamp(r0, min=1e-300))
            umax.append(float(torch.max(torch.hypot(u, v)).detach()))
            divs.append(divergence_rms(u.detach().cpu().numpy(),
                                       v.detach().cpu().numpy()))
            if not torch.isfinite(u).all() or not torch.isfinite(v).all():
                raise RuntimeError(
                    f"wing-fsi rollout is not finite at macro-step {s}; raised "
                    "where it happened so the NaN is not carried into every "
                    "number after it")
            if progress is not None:
                progress(s, time.perf_counter() - t0)
        if steps in keep or -1 in keep:
            fields[steps] = (u.detach().cpu().numpy().copy(),
                             v.detach().cpu().numpy().copy(),
                             delta.detach().cpu().numpy().copy())
        return FSIResult(
            load=torch.stack(loads).detach().cpu().numpy(),
            tip=torch.stack(tips).detach().cpu().numpy(),
            delta=torch.stack(deltas).detach().cpu().numpy(),
            delta_dot=torch.stack(ddots).detach().cpu().numpy(),
            strain_energy=torch.stack(energies).detach().cpu().numpy(),
            interface_power=torch.stack(powers).detach().cpu().numpy(),
            half_step=torch.stack(halves).detach().cpu().numpy(),
            u_max=np.array(umax), div_rms=np.array(divs),
            substeps=np.array(self.substep_log),
            inner_residual=torch.stack(ress).detach().cpu().numpy(),
            newton_drop=torch.stack(drops).detach().cpu().numpy(),
            u=u, v=v, delta_final=delta, energy_0=e0,
            wall_s=time.perf_counter() - t0, fields=fields)

    # -- the objective F5 is about ----------------------------------------

    def objective(self, e_star, steps: int, u0=None, v0=None, delta0=None,
                  frac: float = 0.25, grad: bool = False):
        """The settled downforce, as a tensor, so it can be differentiated.

        ``E*`` is a DESIGN PARAMETER and not a state: it enters only through the
        structure, and the initial field is the same for every value of it, which
        is what makes ``dJ/dE*`` a derivative of the composed stack rather than of
        its initial condition.
        """
        opt = dict(dtype=TORCH_DTYPE, device=self.device)
        if not torch.is_tensor(e_star):
            e_star = torch.tensor(float(e_star), **opt)
        u = (torch.full((self.ny, self.nx), U_INF, **opt) if u0 is None
             else torch.as_tensor(np.asarray(u0), **opt))
        v = (torch.zeros((self.ny, self.nx), **opt) if v0 is None
             else torch.as_tensor(np.asarray(v0), **opt))
        delta = (torch.zeros(N_STATION, **opt) if delta0 is None
                 else torch.as_tensor(np.asarray(delta0), **opt))
        w_plate = torch.zeros(N_STATION, **opt)
        n_avg = max(1, int(round(frac * steps)))
        first = steps - n_avg
        j = None
        for s in range(steps):
            if grad and self.checkpoint:
                (u, v, delta, w_plate, load, _t, _r, _r0, _p, _h
                 ) = torch.utils.checkpoint.checkpoint(
                    self.macro_step, u, v, delta, w_plate, e_star, s,
                    use_reentrant=False)
            else:
                (u, v, delta, w_plate, load, _t, _r, _r0, _p, _h
                 ) = self.macro_step(u, v, delta, w_plate, e_star, s)
            #: **W145.**  `run` checked the structural expert's declared envelope
            #: at every macro-step and this method, which is the one a gradient
            #: or a design search is taken through, checked nothing.  PoC 2's
            #: search walked its own graph out of a declared envelope on exactly
            #: this asymmetry and nothing fired.  Detached, so it cannot enter
            #: the tape or move a number; every column CS-12 published is inside
            #: the bound and is bitwise unchanged.
            dv = delta.detach()
            if self.motion and float(torch.max(torch.abs(dv))) > DELTA_MAX:
                raise RuntimeError(
                    f"the wing left the structural expert's declared envelope at "
                    f"macro-step {s}: max|delta| = "
                    f"{float(torch.max(torch.abs(dv))):.6g} against a small-strain "
                    f"bound of {DELTA_MAX:.6g}")
            if s >= first:
                j = load if j is None else j + load
        return j / n_avg


def referent_rollout(**kw) -> FSIRollout:
    """The unsplit, tightly-coupled column the gate is written against."""
    kw.setdefault("coupling", "tight")
    return FSIRollout(tiling=SINGLE_TILING, **kw)


def composed_rollout(**kw) -> FSIRollout:
    """The six-window column with the FSI seam lagged one macro-step."""
    kw.setdefault("coupling", "lagged")
    return FSIRollout(tiling=DEFAULT_TILING, **kw)


#: Macro-steps of FIXED-SHAPE spin-up before a coupled march is released.  The
#: same field for every column and every value of `E*`, which is what makes
#: ``dJ/dE*`` a derivative of the composed stack rather than of its initial
#: condition -- `ground_effect.N_SPIN`'s argument, unchanged.
#:
#: **120 and not 40, measured.**  The load is still climbing at 40 (0.2117, and
#: 0.4% per 8 macro-steps) and settles into its own shedding band by 120, at
#: 0.2273 with a peak-to-peak of 1.503% over the last quarter.  That band is the
#: LEVEL every difference this case study reports is quoted against.
N_SPIN = 120


@lru_cache(maxsize=8)
def settled_field(steps: int = N_SPIN, single: bool = True, nu: float = NU):
    """The undeflected field, marched from the freestream with the wing rigid.

    A coupled march is released from THIS and never from a uniform stream: the
    impulsive-start load is twice the settled one, and a quasi-static structure
    has no mass to ride it out with.
    """
    tiling = SINGLE_TILING if single else DEFAULT_TILING
    r = FSIRollout(tiling=tiling, coupling="tight", motion=False, nu=nu)
    res = r.run(steps=steps)
    return (res.u.detach().cpu().numpy().copy(),
            res.v.detach().cpu().numpy().copy(), res)
