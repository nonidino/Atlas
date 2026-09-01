"""The SIXTH real case study: a turbine array with real geometry, and the first
graph in this vault whose seams carry an ABSOLUTE trace.

`wind_farm_real.py` has the wind farm's topology and says outright that its
geometry is schematic -- six windows of one field standing in for a farm, so
"a number measured here is a number about the port algebra, not a wind-farm
result".  This file is the other half: real turbine positions, real spacing, a
real wake, and a real power loss, composed from

    six fluid windows   **Poseidon-T**, frozen, 20.8M parameters, fixed at
                        128x128 (`adapters.FrozenFluidExpert`)
    three rotors        `disk.ActuatorDisk`, algebraic, ZERO fitted parameters
    one reference       `reference.WindowNS` on the same 128-cell windows, as
                        the high-fidelity referent tau is measured against

and it exists to run the Tier 14-16 attribution machinery -- `assemble_seam`,
`operator_content`, `support_reach`, `seam_defect_split`, `certify_substitution`
-- against a real pretrained checkpoint inside a real composed wake, which none
of the five earlier case studies could do.

The geometry, and why the checkpoint chose it
---------------------------------------------

The checkpoint is fixed at 128x128 and `adapters.Scaling` fixes everything else
from it.  One macro-step must be one *native* expert lead (0.1), the freestream
must land inside the trained velocity distribution (``U_s = 2``), so a window of
``S_LEN`` rotor diameters forces ``dt_macro = S_LEN / 20``.  Choosing the window
therefore chooses the macro-step, the effective Reynolds number
(``Re = 1 / (2 nu_p S_LEN)``) AND how much lateral room a wake has to recover in
-- and those pull in opposite directions.  `poseidon.py`'s note is that with a
solver you choose the window to suit the halo and with a checkpoint you choose
the domain to suit the window; here it goes one step further, because **the
turbine spacing is a whole number of window strides by construction.**

    S_LEN   = 4 D per 128-cell window     dx = 1/32 D
    dt      = 0.2                          lead 0.1, exactly native
    halo    = 16 cells                     stride 112 cells = 3.5 D
    domain  = 352 x 240 cells = 11.0 x 7.5 D
    rotors  R1 (3.75, 1.75)  R2 (7.25, 1.75)  R3 (7.25, 5.25), D = 1 = 32 cells

3.5 D streamwise and 3.5 D lateral, which is a real closely-spaced layout, and
an L: R1 and R2 in line, R3 abreast of R2 one row over.  R1 and R3 see clean
inflow and R2 sees R1's wake, so the array carries its own controls.

**Re = 255 at the checkpoint's grid-scale viscosity, and the spec asks for
1e3-1e4.**  That is the price of the geometry and it is paid deliberately:
``Re = 1020 / S_LEN``, so the band and the lateral room a wake needs are in
direct conflict for a fixed-resolution operator.  W0 4.2 has the deeper reason
-- the checkpoint's viscosity is *not a number*, it fits 4.9e-4 at lambda =
0.125 D (r2 = 0.998) and nothing distinguishable from zero at 0.25 D and 0.5 D
-- so any Re quoted for it describes a spectral cutoff, not a fluid.

The trace is ABSOLUTE here, and that is the departure
------------------------------------------------------

`window_ns`, `channel_ns`, `poseidon` and `wind_farm_real` all write the trace
into the ring as a **perturbation**: ``r_u[ring] = r_u[ring] + trace``.  Under
that convention ``probe_base`` is identically zero on every fluid seam in this
vault, so `probe.base_disagreement` reads *consistent* everywhere while the two
sides are in fact linearized about their own stored states, which nothing
compares.  That is **W74's class with the base hidden inside the state instead
of declared**, and it is invisible until two sides of one seam hold genuinely
different states -- which a wake does, since the two rings this file couples sit
0.25 D apart across a velocity gradient.

Two things follow and both are load-bearing:

  * `respond` writes the trace INTO the ring and ``probe_base`` returns the
    ring's own physical value, so the linearization point is visible, comparable
    and decertifiable at `L4/probe-base`.  The arithmetic is identical to the
    perturbation convention at the same point; what changes is that the point is
    now on the record.
  * **`multiphysics.seam_defect_split` needs it.**  It measures the defect in
    interface power, ``effort x flow``, and a perturbation times a perturbation
    is not a power.  None of the four fluid case studies could have been
    measured in interface power without this change.

What it cannot do, stated before any number
--------------------------------------------

**The rotor seam is the finding, not the deliverable.**  An actuator disk is a
surface where the traction jumps, and a surface bond is what the port algebra
has.  Getting one needs the two fluid subdomains to ABUT at the disk plane --
a non-overlapping cut -- and `R2b` will not lift the transmission rung to
probed-DtN for an agent whose ``time_discretization`` is ``unknown``, which a
learned one-shot map's is (W61).  Under the *overlapping* decomposition the
checkpoint does support, the disk sits inside the overlap, both windows cover
it, and the coupling is **two-way volumetric** -- W70's shape, and the row
[[gap-worklist]] Tier 16 lists as ``open`` with the note "the two-way one has no
port and no bond".

So this file declares the rotor's two ports at the two nearest artificial rings,
0.25 D upstream and 0.22 D downstream of the disk plane, declares
``geometrically_coincident=False`` because they are not, and reports what the
compiler says.  In an overlapping decomposition **each window's artificial ring
lies inside its neighbour**, so a surface agent in the overlap meets the
DOWNSTREAM window's inflow ring on its upstream side -- which reads backwards
and is right.

The fluid-fluid wake seams have none of this trouble: they are ordinary
artificial boundaries between two windows of one fluid, and every headline
attribution number below is measured on one of those.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

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
from ..graph import Agent, CaseGraph, Connection, Decomposition, GlobalField, MeasuredConstants
from ..ports import PortType, ResponseHalf
from ..transfer import Prolongation
from .poseidon import EXPERT_RES, load_expert
from .window_ns import _no_projection_class, load_reference

# ---------------------------------------------------------------------------
# the geometry, which the checkpoint's fixed resolution chooses
# ---------------------------------------------------------------------------

#: Rotor diameters spanned by one 128-cell window.  Not a tuning knob: it fixes
#: the macro-step (``S_LEN / 20``) and the effective Reynolds number
#: (``1020 / S_LEN``) at the same time -- `adapters.Scaling`, W0 4.3.
S_LEN = 4.0
N = EXPERT_RES
DX = S_LEN / N                       # 1/32 D
MACRO_DT = S_LEN / 20.0              # 0.2, so the expert lead is exactly 0.1

U_INF = 1.0
ROTOR_D = 1.0                        # = 32 cells
ROTOR_CELLS = int(round(ROTOR_D / DX))

#: Overlap in cells.  See `required_halo` and W93: the DECLARED domain of
#: dependence of a Poseidon window is 2 cells and its MEASURED one is the whole
#: window, so no overlap satisfies the halo rule honestly.  16 is what the
#: geometry can afford at a 3.5 D spacing and the compile says what it costs.
HALO = 16
STRIDE = N - HALO                    # 112 cells = 3.5 D
N_COL, N_ROW = 3, 2
NX = (N_COL - 1) * STRIDE + N        # 352 cells = 11.0 D
NY = (N_ROW - 1) * STRIDE + N        # 240 cells =  7.5 D

#: The partition of unity's ramp, in cells.  It must outrun one macro-step's
#: Galilean translation (``U dt / dx`` = 6.4 cells), because `spectral_shift` is
#: periodic within the window and wraps that much material from one edge to the
#: other; the ramp is what gives the wrapped strip zero weight.
RAMP = 8

#: The checkpoint's own spectral cutoff, in cells.  W0 4.2 fits nu = 4.9e-4 at
#: lambda = 0.125 D with r2 = 0.998 and nothing separable from a 2%-per-step bias
#: at 0.25 D or 0.5 D, so the trustworthy band is lambda >= 0.25 D = 8 cells.
#: `effective_resolution` follows from it as a WAVELENGTH rather than a mode
#: count: carrying window_ns's 16 modes onto a 32-cell rotor face would claim
#: four times the resolution the checkpoint has.
LAMBDA_CUT_CELLS = 8

#: The reference solver's viscosity, in this case study's units.  W0 4.2's
#: grid-scale fit ``nu_p = 4.9e-4`` through ``nu = nu_p * S_LEN * U_s``.  It is
#: the ONLY one of W0's three fits that is (a) positive and (b) leaves the
#: reference solver inside its own validity predicate -- see `reference_validity`
#: and W95.
NU_P_GRID_SCALE = 4.9e-4
NU_REF = NU_P_GRID_SCALE * S_LEN * 2.0        # 3.92e-3, Re_D = 255

MECH_SCALES = {"stress": U_INF**2, "velocity": U_INF, "power_area": U_INF**3}
ROT_SCALES = {"torque": U_INF**2, "angular_velocity": U_INF, "power": U_INF**3}

#: Outward normal sign along the face's own axis, and the axis itself.
_FACE_GEOM = {"xlo": (-1.0, "x"), "xhi": (+1.0, "x"),
              "ylo": (-1.0, "y"), "yhi": (+1.0, "y")}

_RING = {
    "xlo": ((slice(None), 0), (slice(None), 1)),
    "xhi": ((slice(None), -1), (slice(None), -2)),
    "ylo": ((0, slice(None)), (1, slice(None))),
    "yhi": ((-1, slice(None)), (-2, slice(None))),
}


def modes_for(n_cells: int) -> int:
    """m_eff from the checkpoint's cutoff WAVELENGTH, not from a mode count.

    A real Fourier basis of ``m`` columns carries wavenumbers up to
    ``k_max = (m - 1) / 2``, whose wavelength is ``n_cells / k_max`` cells.
    Setting that to `LAMBDA_CUT_CELLS` gives ``m = 2 n / lambda_cut + 1``.
    """
    return int(2 * n_cells // LAMBDA_CUT_CELLS + 1)


def fourier_basis(n_cells: int, m: int | None = None, dx: float = DX) -> np.ndarray:
    """(n_cells, m) real Fourier modes, orthonormal in the dx-weighted pairing.

    A local copy of `window_ns.fourier_basis` rather than an import, because that
    one closes over `window_ns.H` (2/128, this vault's original cell size) and
    this case study's cell is 1/32 D.  The Gram is then ``dx I`` and the forced
    adjoint is ``dx P^T`` with no solve, which is the only property the transfer
    layer needs of it.
    """
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


@dataclass(frozen=True)
class Rotor:
    """One turbine: where its disk plane is, and which windows meet it."""

    rotor_id: str
    col: int                 # the x-plane index: the seam between col and col+1
    row: int
    y_centre: float          # in D

    @property
    def x_plane(self) -> float:
        """Mid-overlap, in D: the plane the momentum sink sits on."""
        return ((self.col + 1) * STRIDE + HALO / 2.0) * DX

    @property
    def cells(self) -> slice:
        """The rotor's cells in a row window's LOCAL face index."""
        lo = int(round((self.y_centre - 0.5 * ROTOR_D) / DX)) - self.row * STRIDE
        return slice(lo, lo + ROTOR_CELLS)

    @property
    def upstream_window(self) -> str:
        """The window whose INFLOW ring lies upstream of the disk plane.

        It is the DOWNSTREAM window, and that is not a slip.  In an overlapping
        decomposition each window's artificial ring sits inside its neighbour, so
        the col+1 window's ``xlo`` ring is at ``(col+1) * STRIDE`` -- half a halo
        upstream of the disk -- while the col window's ``xhi`` ring is half a halo
        downstream of it.
        """
        return f"F{self.col + 1}{self.row}"

    @property
    def downstream_window(self) -> str:
        return f"F{self.col}{self.row}"


ROTORS = (
    Rotor("R1", col=0, row=0, y_centre=1.75),
    Rotor("R2", col=1, row=0, y_centre=1.75),
    Rotor("R3", col=1, row=1, y_centre=(STRIDE * DX) + 1.75),
)


@dataclass(frozen=True)
class ArrayTiling:
    """The tiling of the farm domain by windows of exactly `N` cells, and where
    the rotors sit on it.

    **Generalized 2026-08-30 for CS-7.**  It used to read `N_COL`, `N_ROW`, `NX`,
    `NY` and `ROTORS` off this module, which made the 3x2 array with three
    turbines the only layout expressible.  `scaling_ladder.py` needs the same
    geometry at five sizes with the ramp, the overlap, the cell and the macro-step
    held fixed, and copying a tiling in order to change two integers is how two
    tilings drift apart.  So the layout is fields on this object and the
    module-level names are its default instance: `ArrayTiling()` reproduces what
    was here before to the bit, which `tests/test_tier18_wake_array.py` pins.
    """

    ramp: int = RAMP
    n_col: int = N_COL
    n_row: int = N_ROW
    rotors: tuple[Rotor, ...] = ROTORS

    # -- the domain the tiling covers -------------------------------------

    @property
    def nx(self) -> int:
        return (self.n_col - 1) * STRIDE + N

    @property
    def ny(self) -> int:
        return (self.n_row - 1) * STRIDE + N

    @property
    def n_windows(self) -> int:
        return self.n_col * self.n_row

    @property
    def offsets(self) -> list[tuple[int, int]]:
        return [(i * STRIDE, j * STRIDE)
                for j in range(self.n_row) for i in range(self.n_col)]

    @property
    def names(self) -> list[str]:
        return [f"F{i}{j}" for j in range(self.n_row) for i in range(self.n_col)]

    def index_of(self, name: str) -> int:
        return self.names.index(name)

    def coords(self, name: str) -> tuple[int, int]:
        """F<i><j> -> (i, j).  Single digits, so at most ten windows either way."""
        return int(name[1]), int(name[2])

    def artificial_faces(self, ox: int, oy: int) -> tuple[str, ...]:
        out = []
        if ox > 0:
            out.append("xlo")
        if ox + N < self.nx:
            out.append("xhi")
        if oy > 0:
            out.append("ylo")
        if oy + N < self.ny:
            out.append("yhi")
        return tuple(out)

    def cut(self, f: np.ndarray) -> np.ndarray:
        return np.stack([f[oy:oy + N, ox:ox + N] for ox, oy in self.offsets])

    @lru_cache(maxsize=16)
    def weights(self) -> list[np.ndarray]:
        raw = []
        r = max(self.ramp, 1)
        idx = np.arange(N) + 0.5
        nx, ny = self.nx, self.ny
        for ox, oy in self.offsets:
            wx, wy = np.ones(N), np.ones(N)
            if ox > 0:
                wx = np.minimum(wx, np.clip(idx / r, 0.0, 1.0))
            if ox + N < nx:
                wx = np.minimum(wx, np.clip((N - idx) / r, 0.0, 1.0))
            if oy > 0:
                wy = np.minimum(wy, np.clip(idx / r, 0.0, 1.0))
            if oy + N < ny:
                wy = np.minimum(wy, np.clip((N - idx) / r, 0.0, 1.0))
            w = np.zeros((ny, nx))
            w[oy:oy + N, ox:ox + N] = np.minimum(wy[:, None], wx[None, :]) ** 2
            raw.append(w)
        tot = np.sum(raw, axis=0)
        tot = np.where(tot <= 0.0, 1.0, tot)
        return [w / tot for w in raw]

    def assemble(self, us: np.ndarray, vs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        au, av = np.zeros((self.ny, self.nx)), np.zeros((self.ny, self.nx))
        for k, (ox, oy) in enumerate(self.offsets):
            chi = self.weights()[k][oy:oy + N, ox:ox + N]
            au[oy:oy + N, ox:ox + N] += chi * us[k]
            av[oy:oy + N, ox:ox + N] += chi * vs[k]
        return au, av

    def contaminated(self, d_cells: int) -> list[np.ndarray]:
        out = []
        ii = np.arange(N) + 0.5
        for ox, oy in self.offsets:
            faces = self.artificial_faces(ox, oy)
            dx_ = np.full(N, np.inf)
            dy_ = np.full(N, np.inf)
            if "xlo" in faces:
                dx_ = np.minimum(dx_, ii)
            if "xhi" in faces:
                dx_ = np.minimum(dx_, N - ii)
            if "ylo" in faces:
                dy_ = np.minimum(dy_, ii)
            if "yhi" in faces:
                dy_ = np.minimum(dy_, N - ii)
            out.append(np.minimum(dy_[:, None], dx_[None, :]) <= d_cells)
        return out

    def partition_of_unity(self, d_cells: int = RAMP) -> GridPartitionOfUnity:
        idx, wts, bad = {}, {}, {}
        ws, cont = self.weights(), self.contaminated(d_cells)
        nx = self.nx
        for k, (ox, oy) in enumerate(self.offsets):
            rows, cols = np.meshgrid(np.arange(oy, oy + N),
                                     np.arange(ox, ox + N), indexing="ij")
            idx[self.names[k]] = (rows * nx + cols).reshape(-1)
            wts[self.names[k]] = ws[k][oy:oy + N, ox:ox + N].reshape(-1)
            bad[self.names[k]] = cont[k].reshape(-1)
        return GridPartitionOfUnity(
            nx * self.ny, idx, wts, contaminated=bad,
            # W54: the harness parameters the emitted defect has to carry. Convex
            # is admissibility; the RAMP is accuracy, and it is a factor of 200.
            ramp_cells=self.ramp,
            profile=f"min(wy, wx)^2 over a {self.ramp}-cell linear ramp, normalized")

    # -- the rotors, and the ports they split a face into ------------------

    def rotor_at(self, col: int, row: int):
        for r in self.rotors:
            if r.col == col and r.row == row:
                return r
        return None

    def port_segments(self, window: str) -> dict[str, list[str]]:
        """face -> [segment, ...] for one window, from `rotors` and the tiling."""
        i, j = self.coords(window)
        ox, oy = i * STRIDE, j * STRIDE
        out: dict[str, list[str]] = {}
        for face in self.artificial_faces(ox, oy):
            split = any(
                (r.row == j)
                and ((face == "xlo" and r.col + 1 == i)
                     or (face == "xhi" and r.col == i))
                for r in self.rotors
            )
            out[face] = ["bypass", "rotor"] if split else ["full"]
        return out

    def segment_index(self, window: str, face: str, segment: str) -> np.ndarray:
        """The face cells this port owns, as a LOCAL index array into the ring."""
        if segment == "full":
            return np.arange(N)
        i, j = self.coords(window)
        rot = next(r for r in self.rotors
                   if r.row == j and ((face == "xlo" and r.col + 1 == i)
                                      or (face == "xhi" and r.col == i)))
        sl = rot.cells
        if segment == "rotor":
            return np.arange(sl.start, sl.stop)
        return np.concatenate([np.arange(0, sl.start), np.arange(sl.stop, N)])


DEFAULT_TILING = ArrayTiling()


# ---------------------------------------------------------------------------
# the composition layer's own operators -- and they are GLOBAL
# ---------------------------------------------------------------------------

#: Cells of tapered buffer appended DOWNSTREAM of the outlet before the global
#: spectral operators run.  One window, because a window is the only length in
#: this geometry the checkpoint did not choose arbitrarily.
PAD_CELLS = N


def _extend(f: np.ndarray, pad: int) -> np.ndarray:
    """Continue the fluctuation downstream, tapered to zero over `pad` cells."""
    t = np.cos(0.5 * np.pi * (np.arange(1, pad + 1) / pad))[None, :] ** 2
    return np.concatenate([f, f[:, -1:] * t], axis=1)


def transport_and_project(uf: np.ndarray, vf: np.ndarray, dt: float = MACRO_DT,
                          u_inf: float = U_INF, pad: int = PAD_CELLS,
                          project: bool = True):
    """Advect the assembled fluctuation by ``u_inf * dt`` and project it, ONCE.

    Both operations belong to the composition layer and to the WHOLE domain, and
    running either per window is **W98**:

      * **Pressure.**  `build` declares ``pressure`` a `GlobalField` -- "the
        composition layer's, and it has to be" -- and the march then asked
        `step_many` for ``project=True``, which runs an exact Leray projection on
        each 128-cell window SEPARATELY and PERIODICALLY.  A projection is
        elliptic, so a disk's momentum sink had its pressure response smeared
        over its own 4 D window and wrapped onto that window's upstream image.
        The declaration and the code disagreed and the code won.

      * **Transport.**  ``frame=(u_inf, 0)`` makes `step_many` translate every
        window by ``u_inf * dt / dx`` = 6.4 cells with `spectral_shift`, which is
        periodic ON THE WINDOW.  What leaves a window's outflow edge re-entered
        its own inflow edge, and after ``N / 6.4`` = 20 macro-steps a wake had
        circulated its window once and homogenized along x.

    Neither is a halo failure and a wider overlap fixes neither: W93 measured
    this checkpoint's support reach as the WHOLE window, so no overlap satisfies
    the halo rule, and a partition-of-unity ramp that gives the wrapped strip
    zero weight changes the manufactured deficit by 0.01 (measured).  The remedy
    is to run the two global operators globally.

    The domain is extended downstream by `pad` cells of fluctuation tapered to
    zero, which makes the extension periodic-compatible.  On it,

      * the Leray projection is exact and the inlet no longer sees the outlet's
        periodic image;
      * the translation is a phase factor -- exact, no interpolation and no
        diffusion -- and what wraps onto the inlet is the tapered zero, which is
        freestream, which is what an inlet in an unbounded stream supplies.

    ``uf`` and ``vf`` are the fluctuation about the freestream, ``u - u_inf``,
    and the fluctuation is what comes back.
    """
    ny, nx = uf.shape
    big_u, big_v = _extend(uf, pad), _extend(vf, pad)
    nxp = nx + pad
    kx = 2.0 * np.pi * np.fft.fftfreq(nxp, d=(nxp * DX) / nxp)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=(ny * DX) / ny)
    uh, vh = np.fft.fft2(big_u), np.fft.fft2(big_v)
    if project:
        # `_wavenumbers`' convention: the Nyquist mode is zeroed for a
        # DERIVATIVE, so that the divergence and the projection meant to cancel
        # it agree at every mode.  Three modes then have k = 0, not one, and all
        # three have to be guarded or the projection returns NaN.
        kxd, kyd = kx.copy(), ky.copy()
        if nxp % 2 == 0:
            kxd[nxp // 2] = 0.0
        if ny % 2 == 0:
            kyd[ny // 2] = 0.0
        KX, KY = kxd[None, :], kyd[:, None]
        k2 = np.where((KX ** 2 + KY ** 2) == 0.0, 1.0, KX ** 2 + KY ** 2)
        div = KX * uh + KY * vh
        uh, vh = uh - KX * div / k2, vh - KY * div / k2
    # the translation keeps its Nyquist mode: this is an interpolation, not a
    # derivative -- `spectral_shift`'s own distinction, at domain scale.
    phase = np.exp(-1j * kx[None, :] * u_inf * dt)
    big_u = np.real(np.fft.ifft2(uh * phase))
    big_v = np.real(np.fft.ifft2(vh * phase))
    return big_u[:, :nx], big_v[:, :nx]



# ---------------------------------------------------------------------------
# L6/C2 -- the projected assembly, which is a composition-layer STEP (W100)
# ---------------------------------------------------------------------------


def divergence_rms(u: np.ndarray, v: np.ndarray) -> float:
    """``||div u||_rms`` on the interior, with the WIDE centred operator.

    The same difference the reference solver's own projection inverts, so a
    field that solver calls divergence-free reads zero here and a field it does
    not calls nonzero.  Measuring with a narrow stencil instead would report a
    residual the scheme never had.

    **This is the quantity L6/C2 is about.**  Each `WindowNS` window returns a
    field that is divergence-free ON ITS OWN WINDOW, and a partition-of-unity
    blend of divergence-free fields is not divergence-free: with
    ``sum_i chi_i = 1``,

        div( sum_i chi_i u_i ) = sum_i grad(chi_i) . u_i
                               = grad(chi_1) . (u_1 - u_2)   [two windows]

    so the assembly manufactures divergence exactly where the local solves
    disagree, in proportion to how fast chi is turning over.
    """
    du = np.zeros_like(u)
    dv = np.zeros_like(v)
    du[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * DX)
    dv[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2.0 * DX)
    d = du + dv
    return float(np.sqrt(np.mean(d[2:-2, 2:-2] ** 2)))


def project_assembled(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One global Leray projection on the ASSEMBLED field, and nothing else.

    `transport_and_project` at zero translation: the same operator, on the same
    padded domain, with ``dt = 0`` so the translation is the identity and what
    is left is the projection alone.  ``u`` and ``v`` are the FULL field rather
    than the fluctuation -- the freestream is removed and restored here, because
    a caller that had to remember to do it is a caller that will one day not.

    The padding is `PAD_CELLS` and it is load-bearing, not cosmetic: on the raw
    domain the FFT's periodic image puts the outlet next to the inlet, and a
    projection is elliptic, so a disk's momentum sink would reach round the box.
    """
    uf, vf = transport_and_project(u - U_INF, v, dt=0.0, u_inf=0.0, project=True)
    return U_INF + uf, vf


def leray_projection(tiling: ArrayTiling = DEFAULT_TILING) -> ConstraintProjection:
    """The composition layer's own operator, declared (L6/C2, R12).

    Global, after the blend, once per exchange -- the three fields the compile
    decides R12 on.  The Poseidon column has applied exactly this since W98 and
    nothing on the graph said so; the classical column did not apply it at all
    and nothing on the graph said that either, which is how a reference column
    shipped on a trajectory that is not finite past macro-step 82 at six windows.

    **ONCE is the word that carries the measurement.**  Declared beside agents
    that still project internally, this operator is the SECOND application of the
    same pressure and the column is worse for it -- band left at 51 against the
    unprojected 74.  See `exposed_reference_solver`, which is the agent side of
    the same repair and the one the 120-step march selects.
    """
    return ConstraintProjection(
        constraint="divergence-free",
        scope="global",
        stage="after-assembly",
        cadence=1,
        operator=project_assembled,
        residual=divergence_rms,
        # both callables want the (ny, nx) grid; a partition of unity assembles
        # into a flat n_global vector, and this is what reconciles them
        shape=(tiling.ny, tiling.nx),
        note=("wake_array.project_assembled: an exact spectral Leray projection on "
              "the assembled domain extended downstream by PAD_CELLS of tapered "
              "fluctuation. It is the SAME operator the composition layer already "
              "applies for transport (W98) with the translation set to zero, so "
              "the two cannot drift apart. **Its hypothesis is the freestream "
              "band**: the projection is periodic in y and periodic-compatible in "
              "x only because `_extend` tapers the outflow and because the march "
              "holds the inlet and both laterals at (U_INF, 0). On a field that "
              "is not freestream at the laterals it is a projection onto a "
              "different kernel from the agents' own -- measured on a random "
              "perturbation field at N=1 it moves the state by 6% -- so it is "
              "the composition layer's operator for THIS geometry and does not "
              "travel to a case study with a different outer boundary. And the "
              "residual it is measured by is not its own: `divergence_rms` is a "
              "wide centred difference on the unpadded interior while the "
              "projection is spectral on the padded periodic domain, so the two "
              "disagree at the domain edge. At N=2, where the blend's commutator "
              "is one overlap wide, that disagreement is the LARGER of the two "
              "and the projected column reads a higher divergence than the bare "
              "one (0.0259 against 0.0132, both flat); from N=6 up the commutator "
              "dominates and it reads 4.7x lower"),
    )


def projected_assembly(tiling: ArrayTiling = DEFAULT_TILING,
                       d_cells: int = RAMP) -> ProjectedAssembly:
    """The assembly this case study declares: the blend AND the projection.

    Passed to `CaseGraph.partition_of_unity`, where every existing L6/C1 query
    delegates to the partition and R12 reads the projection off the same object.
    """
    return ProjectedAssembly(tiling.partition_of_unity(d_cells),
                             leray_projection(tiling))


def assemble_conservative(tiling: ArrayTiling, us: np.ndarray, vs: np.ndarray,
                          assembly: ProjectedAssembly | None = None):
    """One composition-layer assembly step on stacked per-window solutions.

    ``us``/``vs`` are ``(n_windows, N, N)`` as `ArrayTiling.cut` returns them, so
    a march can call this exactly where it called ``tiling.assemble`` and get the
    step the graph declares rather than the blend the graph does not.
    """
    assembly = assembly or projected_assembly(tiling)
    au, av = tiling.assemble(us, vs)
    return assembly.projection.apply(au, av)


# ---------------------------------------------------------------------------
# which ports each window carries, derived from the layout
# ---------------------------------------------------------------------------


def port_segments(window: str,
                  tiling: ArrayTiling = DEFAULT_TILING) -> dict[str, list[str]]:
    """``face -> [segment, ...]`` for one window, from the tiling's rotors.

    A face that a rotor plane meets splits into a ``rotor`` segment carrying the
    disk and a ``bypass`` segment carrying the open flow beside it; every other
    artificial face is one ``full`` segment.  This is `wind_farm_real`'s "the
    passenger list is per FACE, not per agent" one step further on: **a port is
    per face SEGMENT**, because a real rotor spans 1 D of a 4 D face and nothing
    in the port algebra says a port's V has to be a whole face.

    The body moved onto `ArrayTiling` when CS-7 needed the same rule at five
    sizes; this is the module-level name the Tier 18 tests and scripts call.
    """
    return tiling.port_segments(window)


def segment_index(window: str, face: str, segment: str,
                  tiling: ArrayTiling = DEFAULT_TILING) -> np.ndarray:
    """The face cells this port owns, as a LOCAL index array into the ring."""
    return tiling.segment_index(window, face, segment)


def port_name(face: str, segment: str) -> str:
    return f"{face}:{segment}:MECH"


def parse_port(name: str) -> tuple[str, str, str]:
    face, segment, kind = name.split(":")
    return face, segment, kind


# ---------------------------------------------------------------------------
# the fluid agent -- one window, either expert, the same port list
# ---------------------------------------------------------------------------


@dataclass
class FluidWindow:
    """One 128-cell window of the farm, exposed as a `boundary_response`.

    ``kind`` is ``"poseidon"`` (the frozen checkpoint) or ``"reference"``
    (`reference.WindowNS` at `NU_REF` on the same window).  The two declare an
    IDENTICAL port list on purpose: that is what makes one a legal substitution
    for the other under `composition.certify_substitution`, and a swap that
    changed the port list would be a graph edit rather than a substitution.

    The trace is ABSOLUTE -- see this module's docstring.  ``respond`` writes it
    into the ring cells the port owns and returns the Steklov-Poincare flux
    ``nu dw/dn`` on those same cells, which is `probed-dtn-coupling` 2.2's
    normative MECH effort and what W47 settled as the block-level form.
    """

    agent_id: str
    u0: np.ndarray
    v0: np.ndarray
    kind: str = "poseidon"
    dt: float = MACRO_DT
    #: Which tiling this window belongs to, and therefore which faces are
    #: artificial and which of them a rotor plane splits.  Defaults to the 3x2
    #: array, so every pre-CS-7 caller is unchanged.
    tiling: "ArrayTiling" = DEFAULT_TILING
    #: The viscosity in the PORT's flux convention, ``nu dw/dn``. It is a
    #: declaration about the port and it cancels exactly in any relative defect
    #: taken on a fluid-fluid seam, where both sides carry the same factor.
    nu: float = NU_REF
    #: The viscosity the REFERENCE SOLVER integrates, which is a different thing
    #: and is what W95 sweeps: the checkpoint's own viscosity is not a number, so
    #: the referent's is a choice and tau inherits the spread. Ignored when
    #: ``kind == "poseidon"``, which has no viscosity to set.
    nu_solver: float | None = None
    expert: Any = None
    n_calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.u0 = np.ascontiguousarray(np.asarray(self.u0, dtype=float))
        self.v0 = np.ascontiguousarray(np.asarray(self.v0, dtype=float))
        if self.u0.shape != (N, N):
            raise ValueError(
                f"Poseidon-T is fixed at {N}x{N}; got {self.u0.shape}. This is the "
                "constraint the array geometry is built around, not a configuration "
                "error"
            )
        self.segments = self.tiling.port_segments(self.agent_id)
        if self.kind == "poseidon":
            self.expert = self.expert or scaled_expert()
        elif self.kind == "reference":
            self.expert = self.expert or reference_solver(
                self.nu if self.nu_solver is None else self.nu_solver)
        elif self.kind == "reference_exposed":
            # W100: the same solver with its elliptic part handed to the
            # composition layer. A separate kind rather than a flag, because the
            # capability record it produces declares a DIFFERENT
            # `elliptic_subsolve` and R10 turns on that word.
            self.expert = self.expert or exposed_reference_solver(
                self.nu if self.nu_solver is None else self.nu_solver)
        else:
            raise ValueError(f"unknown fluid expert kind {self.kind!r}")

    # -- the step ---------------------------------------------------------

    def step(self, u: np.ndarray, v: np.ndarray):
        self.n_calls += 1
        if self.kind == "poseidon":
            return self.expert.step(np.ascontiguousarray(u),
                                    np.ascontiguousarray(v), self.dt)
        uu, vv = self.expert.step_batch(self.u0[None], self.v0[None], self.dt,
                                        bc0=(u[None], v[None]), bc1=None)
        return uu[0], vv[0]

    # -- the boundary channel ---------------------------------------------

    def probe_base(self, name: str) -> np.ndarray:
        """The ring's own physical normal velocity: the state the probe is at.

        Declared rather than left at zeros, because this port's trace is the
        absolute velocity and zero is a state no window in a wind farm is ever
        in.  `probe.base_disagreement` can then compare the two sides of a seam,
        which on a fluid seam in this vault it has never been able to do.
        """
        face, segment, _ = parse_port(name)
        ring, _ = _RING[face]
        w = self.u0 if _FACE_GEOM[face][1] == "x" else self.v0
        idx = self.tiling.segment_index(self.agent_id, face, segment)
        return np.asarray(w[ring], dtype=float)[idx]

    def respond(self, name: str, trace: np.ndarray) -> np.ndarray:
        face, segment, kind = parse_port(name)
        if kind != "MECH":
            raise ValueError(f"{self.agent_id} has no port kind {kind!r}")
        idx = self.tiling.segment_index(self.agent_id, face, segment)
        trace = np.asarray(trace, dtype=float).reshape(-1)
        if trace.shape[0] != idx.size:
            raise ValueError(
                f"trace on {name} has length {trace.shape[0]}, expected {idx.size}")
        ring, interior = _RING[face]
        r_u, r_v = self.u0.copy(), self.v0.copy()
        if _FACE_GEOM[face][1] == "x":
            col = r_u[ring].copy()
            col[idx] = trace
            r_u[ring] = col
        else:
            col = r_v[ring].copy()
            col[idx] = trace
            r_v[ring] = col
        u1, v1 = self.step(r_u, r_v)
        w = u1 if _FACE_GEOM[face][1] == "x" else v1
        return (self.nu * (np.asarray(w[ring]) - np.asarray(w[interior])) / DX)[idx]

    # -- declarations that are callables ----------------------------------

    def storage(self, u: Any = None, v: Any = None) -> float:
        if u is None:
            u, v = self.u0, self.v0
        u = np.asarray(u, dtype=float)
        v = self.v0 if v is None else np.asarray(v, dtype=float)
        return 0.5 * float(np.sum(u * u + v * v)) * DX**2

    def reference_validity(self, state: Any = None, cond: Any = None) -> bool:
        """The cell Reynolds number the reference discretization is valid at.

        ``h U / nu <= 8``, which is `wind_farm_real`'s own predicate and twice
        `window_ns`'s operating point.  It is declared only on the REFERENCE
        expert: the checkpoint has no viscosity to form the number from, which is
        W0 4.2 and is why its record leaves ``validity`` None and decertifies at
        every graph size rather than declaring a predicate nobody earned.

        **W95 lives here.**  ``h U / nu = 1 / (256 nu_p)`` is independent of
        ``S_LEN``, so it is a property of the CHECKPOINT: 7.97 at W0's grid-scale
        fit and 217 at its lambda = 0.5 D fit.  The reference solver is inside its
        own validity at exactly one point of the checkpoint's measured viscosity
        spectrum, and it is the grid-scale end -- not the wake's.
        """
        u, v = (self.u0, self.v0) if state is None else state
        u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
        if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
            return False
        nu_s = self.nu if self.nu_solver is None else self.nu_solver
        return bool(DX * float(np.max(np.hypot(u, v))) / nu_s <= 8.0)


@lru_cache(maxsize=2)
def scaled_expert(device: str = "cpu", threads: int = 8):
    """`FrozenFluidExpert` re-scaled so one window spans `S_LEN` rotor diameters.

    `poseidon.load_expert` builds it at `adapters.WINDFARM_SCALING`
    (``length=1``), which puts one window on one rotor and leaves a wake no room
    to recover.  The scaling is a declaration about geometry, not a property of
    the weights, so it is set here and `Scaling.check` reports what it costs.
    """
    ex = load_expert(device=device, threads=threads)
    ad = importlib.import_module("atlas_windfarm_reference.adapters")
    ex.scaling = ad.Scaling(length=S_LEN, velocity=2.0)
    return ex


@lru_cache(maxsize=4)
def reference_solver(nu: float = NU_REF):
    """`reference.WindowNS` on one `N`-cell window of side `S_LEN`."""
    ref = load_reference()
    return ref.WindowNS(nu=nu, length=S_LEN, n=N, cfl=0.4, transmission="dirichlet")


@lru_cache(maxsize=4)
def exposed_reference_solver(nu: float = NU_REF):
    """The same window with its elliptic part REMOVED -- R10's own prescription.

    **W100, 2026-08-31, and it is the arrangement the measurement selects.**
    `window_ns._no_projection_class` overrides `_project` with the identity, so
    the window returns a field that is NOT divergence-free and the composition
    layer supplies the projection once, globally, after the blend. That is what
    ``elliptic_subsolve=EXPOSED`` declares, what R10 has demanded of this column
    since Tier 0, and what the Poseidon column has done since W98.

    It matters because the obvious repair does not work. Leaving the projection
    inside the agent and adding a global one after the assembly applies the
    pressure TWICE -- each window has already answered the disk's momentum sink
    on its own subdomain -- and measured over 120 macro-steps from the freestream
    that is worse than doing nothing:

        N = 6, macro-step at which |u| leaves the band 3
          embedded, bare blend                       74
          embedded + global spectral projection      51
          embedded + global Neumann projection       33
          **EXPOSED + global spectral projection     stable to 120**

    and the Neumann variant holds the divergence at 0.0089 against the bare
    column's 0.69 while blowing up soonest, so **the divergence is not what ends
    the rollout** -- which is the part of section 19.6's account this corrects.
    """
    return _no_projection_class()(nu=nu, length=S_LEN, n=N, cfl=0.4,
                                 transmission="dirichlet")


def scaling_report(dt: float = MACRO_DT, nu_p: float = NU_P_GRID_SCALE) -> dict:
    """What this geometry costs, from the adapter's own checker."""
    load_reference()                       # registers the private package
    ad = importlib.import_module("atlas_windfarm_reference.adapters")
    out = dict(ad.Scaling(length=S_LEN, velocity=2.0).check(dt, nu_expert=nu_p))
    out["cell_reynolds_reference"] = DX * U_INF / (nu_p * S_LEN * 2.0)
    out["S_LEN"] = S_LEN
    return out


# ---------------------------------------------------------------------------
# the rotor -- a real ActuatorDisk, cellwise, returning a TRACTION
# ---------------------------------------------------------------------------


@dataclass
class RotorDisk:
    """`disk.ActuatorDisk` on a 32-cell face, as a `boundary_response`.

    Algebraic, stateless, zero fitted parameters, and exact to machine precision
    at any probe step -- its response is a derivative of a closed form.

    **It returns the thrust per unit disk AREA**, ``0.5 rho C_T' u^2``, which is
    the MECH effort and carries the declared ``stress`` scale ``U_INF^2``.
    `wind_farm_real.RotorAgent` returns `ActuatorDisk.force_density` instead --
    ``thrust / (area * thickness)``, a force per unit VOLUME, which is the
    traction divided by the declared strip thickness and therefore ``10x`` it at
    the default ``thickness = 0.1``.  That is W66's class one notch along: the
    callable does not return the declared variable, `check_scales` validates the
    scale SET and never sees the callable, and here the two differ by a declared
    geometric LENGTH rather than by W69's un-separable O(1) factor.

    Local induction, per `disk.py`: ``C_T' = 4a/(1-a)`` referenced to the disk's
    own inflow rather than to a freestream, because inside a wake there is no
    freestream to reference and using the domain inlet would delete the coupling
    this case study exists to measure.
    """

    agent_id: str
    u_ref: np.ndarray                    # the physical inflow, per cell
    dt: float = MACRO_DT
    a: float = 1.0 / 3.0
    n_calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        load_reference()
        mod = importlib.import_module("atlas_windfarm_reference.disk")
        self._mod = mod
        self._disk = mod.ActuatorDisk(a=self.a)
        self.u_ref = np.asarray(self.u_ref, dtype=float).reshape(-1)
        if self.u_ref.size != ROTOR_CELLS:
            raise ValueError(
                f"{self.agent_id}: u_ref has {self.u_ref.size} cells, expected "
                f"{ROTOR_CELLS} -- the rotor spans {ROTOR_D} D at dx = {DX}")
        self._state = self._disk(float(np.mean(self.u_ref)))

    # -- the bond ---------------------------------------------------------

    def traction(self, u: np.ndarray) -> np.ndarray:
        """``0.5 rho C_T' u^2`` per cell: thrust per unit disk area."""
        ctp = self._mod.c_t_prime(self._state.a)
        return 0.5 * self._disk.rho * ctp * np.asarray(u, dtype=float) ** 2

    def probe_base(self, name: str) -> np.ndarray:
        """The disk's own inflow, which IS the state its response is about.

        Not zeros: the disk's response is quadratic in the trace, so at a zero
        base its block is identically zero and the seam would probe as EMPTY --
        which is the correct answer for a turbine in still air and the wrong one
        for this graph.  W74, on a rotor.
        """
        if name == "shaft:ROT":
            return np.zeros(1)
        return self.u_ref.copy()

    def respond(self, name: str, trace: np.ndarray) -> np.ndarray:
        self.n_calls += 1
        trace = np.asarray(trace, dtype=float).reshape(-1)
        if name == "shaft:ROT":
            st = self._disk(max(float(np.mean(self.u_ref)), 1e-6))
            return np.array([st.torque])
        if name not in ("up:MECH", "down:MECH"):
            raise ValueError(f"{self.agent_id} has no port {name!r}")
        # The disk removes streamwise momentum, so the traction it presents to
        # the flow points upstream on its downstream face and downstream on its
        # upstream one. The sign is the outward normal's, not a convention.
        sign = -1.0 if name == "down:MECH" else +1.0
        return sign * self.traction(trace)

    def storage(self, u: Any = None, v: Any = None) -> float:
        """Kinetic energy of the disk-averaged inflow. Real, and small."""
        return 0.5 * float(np.mean(self.u_ref) ** 2)

    def validity(self, state: Any = None, cond: Any = None) -> bool:
        """`clamp_induction` is the disk's own declared envelope."""
        return not bool(self._state.clamped)

    @property
    def power(self) -> float:
        return float(self._disk(max(float(np.mean(self.u_ref)), 1e-6)).power)


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


def _prolongation(agent_id: str, name: str, n_cells: int) -> Prolongation:
    return Prolongation(
        agent_id=agent_id, port_name=name,
        matrix=fourier_basis(n_cells),
        gram_V=DX * np.eye(n_cells),
        label=(f"{modes_for(n_cells)}-mode real Fourier basis, truncated at the "
               f"checkpoint's own {LAMBDA_CUT_CELLS}-cell cutoff (W0 4.2)"),
    )


def fluid_ports(window: str, tiling: ArrayTiling = DEFAULT_TILING) -> list:
    """The port list, identical for the checkpoint and the reference solver."""
    ports = []
    for face, segments in sorted(tiling.port_segments(window).items()):
        for segment in segments:
            n_cells = int(tiling.segment_index(window, face, segment).size)
            name = port_name(face, segment)
            ports.append(port_decl(
                name=name, port_type=PortType.MECH,
                geometry=f"{face} ring of {window}, {segment} segment ({n_cells} cells)",
                direction=Direction.BIDIRECTIONAL, nondim=dict(MECH_SCALES),
                effective_resolution=modes_for(n_cells),
                motion_class=MotionClass.STATIC,
                # nu dw/dn, the MECH EFFORT -- probed-dtn-coupling 2.2's
                # normative form, against an imposed normal velocity (the FLOW).
                response_half=ResponseHalf.EFFORT,
                prolongation=_prolongation(window, name, n_cells),
            ))
    return ports


def fluid_capabilities(expert: FluidWindow,
                       elliptic: EllipticSubsolve = EllipticSubsolve.UNKNOWN,
                       reproducibility_floor: float | None = None,
                       stencil_radius: int = 2) -> ExpertCapabilities:
    """One record per fluid window.

    ``elliptic_subsolve`` defaults to ``UNKNOWN`` for the checkpoint, which is
    the honest value for a black box (W60) and is the one
    `conformance._test_elliptic_subsolve` can RESOLVE, by poking a delta (W75).
    It is a parameter so the same graph compiles both ways and the verdict flip
    is visible rather than argued.

    ``stencil_radius`` is the DECLARED domain of dependence and `poseidon.py`
    declares 2.  W93 is that nothing checks it and `support_reach` measures it:
    on this checkpoint the measured reach is the whole window.
    """
    is_ref = expert.kind in ("reference", "reference_exposed")
    if reproducibility_floor is None:
        reproducibility_floor = np.finfo(float).eps if is_ref else 1e-5
    return ExpertCapabilities(
        expert_id=expert.agent_id,
        ports=fluid_ports(expert.agent_id, expert.tiling),
        # W59 for the checkpoint: the ring of the INITIAL CONDITION is settable
        # and there is no boundary condition held through the step, because there
        # is no "through". WindowNS holds a real Dirichlet ring through its own
        # sub-steps, and BCChannel cannot tell the two apart.
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=bool(is_ref),
        elliptic_subsolve=(
            EllipticSubsolve.EXPOSED if expert.kind == "reference_exposed"
            else EllipticSubsolve.EMBEDDED if is_ref else elliptic),
        stencil_radius=stencil_radius,
        substeps_per_macro_step=1,
        # A learned one-shot map is neither explicit nor implicit; R2b reads this
        # and must not lift the rung to probed-DtN on it (W61).
        time_discretization=(TimeDiscretization.EXPLICIT if is_ref
                             else TimeDiscretization.UNKNOWN),
        differentiable=Differentiable.NONE,
        dt_native=expert.dt,
        L_native=S_LEN,
        storage=expert.storage,
        equivariances=(),
        validity=expert.reference_validity if is_ref else None,
        governing_family="incompressible-navier-stokes-2d",
        # There is no monolith for a checkpoint fixed at 128x128 -- at ANY
        # resolution, ever -- so the referent tau is measured against is a PAIR
        # even though both sides share a governing family. `lambda_ref` was
        # introduced for the multiphysics case (W88); a fixed-resolution learned
        # expert needs it at a single-physics seam for the same reason.
        lambda_ref=None if is_ref else "reference.WindowNS pair at nu = NU_REF",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash=(f"windowns-wake-array-nu"
                     f"{expert.nu if expert.nu_solver is None else expert.nu_solver:.6g}"
                     f"-n{N}" if is_ref
                     else "poseidon-t-camlab-ethz-frozen-res128"),
        boundary_response=expert.respond,
        probe_base=expert.probe_base,
        reproducibility_floor=float(reproducibility_floor),
        deterministic=True,
        note=("reference.WindowNS at the array's own geometry, the referent"
              if is_ref else
              "Poseidon-T, frozen, 20.8M parameters; absolute trace, so the "
              "probe base is on the record (W74's class)"),
    )


def rotor_capabilities(expert: RotorDisk) -> ExpertCapabilities:
    ports = [
        port_decl(
            name=name, port_type=PortType.MECH,
            geometry=f"{side} face of {expert.agent_id} ({ROTOR_CELLS} cells)",
            direction=Direction.BIDIRECTIONAL, nondim=dict(MECH_SCALES),
            effective_resolution=modes_for(ROTOR_CELLS),
            motion_class=MotionClass.STATIC,
            response_half=ResponseHalf.EFFORT,
            prolongation=_prolongation(expert.agent_id, name, ROTOR_CELLS),
        )
        for side, name in (("upstream", "up:MECH"), ("downstream", "down:MECH"))
    ]
    ports.append(port_decl(
        name="shaft:ROT", port_type=PortType.ROT, geometry="shaft",
        direction=Direction.OUT, nondim=dict(ROT_SCALES),
        effective_resolution=1, response_half=ResponseHalf.EFFORT,
        note="unconnected: no drivetrain. The extracted power leaves the system "
             "here, and under the port algebra that is an OPEN PORT with a "
             "measurable power flow rather than an absence",
    ))
    return ExpertCapabilities(
        expert_id=expert.agent_id,
        ports=ports,
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=True,
        # KNOWN, unlike the checkpoint's: the disk is disk.py's zero-parameter
        # closed form and contains no solve of any kind.
        elliptic_subsolve=EllipticSubsolve.NONE,
        stencil_radius=0,
        substeps_per_macro_step=1,
        time_discretization=TimeDiscretization.IMPLICIT,
        differentiable=Differentiable.NONE,
        dt_native=expert.dt,
        L_native=ROTOR_D,
        storage=expert.storage,
        equivariances=(),
        validity=expert.validity,
        # The SAME family as the flow it closes. An algebraic closure WITHIN a
        # continuum problem is not a different continuum problem, and declaring
        # otherwise fails E3 at every rotor face.
        governing_family="incompressible-navier-stokes-2d",
        lambda_ref="itself: a closed form has no infidelity to measure",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash=f"actuatordisk-a{expert.a:.6g}-traction",
        boundary_response=expert.respond,
        probe_base=expert.probe_base,
        reproducibility_floor=np.finfo(float).eps,
        deterministic=True,
        note="disk.ActuatorDisk, closed form, zero fitted parameters; returns the "
             "traction (thrust per unit AREA), not force_density",
    )


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------


def make_experts(u_full: np.ndarray, v_full: np.ndarray, kind: str = "poseidon",
                 dt: float = MACRO_DT, nu: float = NU_REF,
                 tiling: ArrayTiling = DEFAULT_TILING,
                 nu_solver: float | None = None) -> dict[str, Any]:
    """Cut the farm state into six windows and read each rotor's own inflow."""
    u_full = np.asarray(u_full, dtype=float)
    v_full = np.asarray(v_full, dtype=float)
    if u_full.shape != (tiling.ny, tiling.nx):
        raise ValueError(f"farm state is {u_full.shape}, expected "
                         f"{(tiling.ny, tiling.nx)}")
    us, vs = tiling.cut(u_full), tiling.cut(v_full)
    out: dict[str, Any] = {
        name: FluidWindow(agent_id=name, u0=us[k], v0=vs[k], kind=kind, dt=dt,
                          nu=nu, nu_solver=nu_solver, tiling=tiling)
        for k, name in enumerate(tiling.names)
    }
    for rot in tiling.rotors:
        # the disk's own inflow, read from the ring that lies UPSTREAM of its
        # plane -- which belongs to the downstream window, see Rotor.upstream_window
        w = out[rot.upstream_window]
        out[rot.rotor_id] = RotorDisk(
            agent_id=rot.rotor_id,
            u_ref=w.probe_base(port_name("xlo", "rotor")), dt=dt)
    return out


def connections(tiling: ArrayTiling = DEFAULT_TILING) -> list[Connection]:
    """The seams, derived from the layout rather than listed -- thirteen on the
    3x2 array, and whatever the tiling implies at any other size."""
    conns: list[Connection] = []
    rotor_at = {(r.col, r.row): r for r in tiling.rotors}

    def conn(seam_id, a, b, expected_null_dim, note):
        return Connection(
            seam_id=seam_id, a=a, b=b, port_type=PortType.MECH,
            derive_space=True,
            # FALSE, and honestly so: in an overlapping decomposition the two
            # artificial rings a seam pairs are HALO cells apart and no two of
            # them coincide. `poseidon.py` and `wind_farm_real.py` both declare
            # True on rings 20 cells apart; the label is one nothing checks.
            geometrically_coincident=False,
            expected_null_dim=expected_null_dim, note=note)

    for j in range(tiling.n_row):
        for i in range(tiling.n_col - 1):
            up, dn = f"F{i}{j}", f"F{i+1}{j}"
            rot = rotor_at.get((i, j))
            if rot is None:
                conns.append(conn(
                    f"x{i}r{j}_full",
                    (up, port_name("xhi", "full")), (dn, port_name("xlo", "full")),
                    0, "fluid-fluid artificial boundary, whole face"))
                continue
            conns.append(conn(
                f"x{i}r{j}_bypass",
                (up, port_name("xhi", "bypass")), (dn, port_name("xlo", "bypass")),
                0, "the open flow beside the rotor: the wake's outer region"))
            conns.append(conn(
                f"{rot.rotor_id}_up",
                (rot.rotor_id, "up:MECH"),
                (rot.upstream_window, port_name("xlo", "rotor")),
                0, "field-to-lumped: the disk reads its inflow from the ring "
                   f"{HALO / 2 * DX:.3g} D upstream of its plane"))
            conns.append(conn(
                f"{rot.rotor_id}_down",
                (rot.rotor_id, "down:MECH"),
                (rot.downstream_window, port_name("xhi", "rotor")),
                0, "field-to-lumped: the disk's wake, imposed on the ring "
                   f"{HALO / 2 * DX:.3g} D downstream of its plane"))
    for j in range(tiling.n_row - 1):
        for i in range(tiling.n_col):
            conns.append(conn(
                f"y{j}c{i}",
                (f"F{i}{j}", port_name("yhi", "full")),
                (f"F{i}{j + 1}", port_name("ylo", "full")),
                0, "fluid-fluid artificial boundary between two turbine rows"))
    return conns


def build(u_full: np.ndarray, v_full: np.ndarray, kind: str = "poseidon",
          dt: float = MACRO_DT, nu: float = NU_REF,
          tiling: ArrayTiling = DEFAULT_TILING,
          elliptic: EllipticSubsolve = EllipticSubsolve.UNKNOWN,
          measured: MeasuredConstants | None = None,
          experts: dict[str, Any] | None = None,
          assembly_projection: bool = True) -> tuple[CaseGraph, dict[str, Any]]:
    """The nine-agent turbine array: six fluid windows and three rotors.

    ``measured`` defaults to None and the reason is stronger than `channel_ns`'s.
    A constant measured on another expert is not this expert's -- and for this
    expert ``tau``, ``sigma`` and ``C_mu`` measured against a MONOLITH are not
    available at all, because a checkpoint fixed at 128x128 has no monolith at
    any resolution.  What replaces it is a reference PAIR; see `lambda_ref` on
    the record and `scripts/w93_wake_array.py` for the measurement.

    ``assembly_projection`` defaults to **True** since 2026-08-31 (W100): the
    assembly is a `ProjectedAssembly` -- the blend AND the global Leray
    projection that makes it conservative -- because without that projection the
    classical composed rollout is not finite past macro-step 82 at six windows.
    Passing False declares the assembly this case study shipped before that
    measurement, which is a bare partition of unity; it exists so the negative
    control is a DECLARATION the compile refuses to certify (L6/R12) rather than
    a line a driver quietly omits, which is what it was.
    """
    experts = experts or make_experts(u_full, v_full, kind, dt, nu, tiling)
    agents = [Agent(name, fluid_capabilities(experts[name], elliptic),
                    domain=f"window {name}")
              for name in tiling.names]
    agents += [Agent(r.rotor_id, rotor_capabilities(experts[r.rotor_id]),
                     domain=f"rotor at ({r.x_plane:.2f}, {r.y_centre:.2f}) D",
                     role="rotor")
               for r in tiling.rotors]
    return (
        CaseGraph(
            name=f"wake-array-{tiling.n_col}x{tiling.n_row}-{kind}",
            agents=agents,
            connections=connections(tiling),
            decomposition=Decomposition.OVERLAPPING,
            overlap=HALO * DX,
            overlap_cells=HALO,
            partition_of_unity=(projected_assembly(tiling) if assembly_projection
                                else tiling.partition_of_unity()),
            global_fields=[GlobalField(
                "pressure",
                note="the composition layer's, and it has to be: the checkpoint's "
                     "pressure channel is a PLACEHOLDER its loader pins to 0 "
                     "(adapters fact 1), so the field that mediates an actuator "
                     "disk's momentum sink is not among its outputs. Measured over "
                     "four macro-steps with three disks: min u = +0.4313 with the "
                     "projection and -0.7395 without it -- a reversed flow through "
                     "the disk plane, which no momentum sink may produce. The "
                     "projection is GLOBAL, on this domain and not on a window: "
                     "doing it per window is W98. **And it is applied AFTER the "
                     "assembly, which is W100**: a partition-of-unity blend of "
                     "divergence-free fields is not divergence-free, so the "
                     "operator has to see the assembled field or it repairs "
                     "nothing. That ordering is declared on the assembly rather "
                     "than here -- see `projected_assembly` and L6/C2")],
            cross_points=tuple(f"x{i}y{j}"
                               for j in range(tiling.n_row - 1)
                               for i in range(tiling.n_col - 1)),
            macro_dt=dt,
            measured=measured,
            note=(f"{len(tiling.rotors)} turbines at 3.5 D spacing in an L, on a "
                  f"{tiling.nx}x{tiling.ny}-cell "
                  f"({tiling.nx * DX:.1f} x {tiling.ny * DX:.1f} D) domain tiled by "
                  f"{tiling.n_windows} "
                  f"{N}-cell Poseidon-T windows of {S_LEN} D each. Real geometry, "
                  "real wake, real power loss -- and the rotor seam is a two-way "
                  "VOLUMETRIC coupling wearing a surface bond (W94)"),
        ),
        experts,
    )
