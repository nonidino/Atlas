"""PoC 1a: gradient-based wind-farm design THROUGH the composed graph.

This is not a ninth case study and it is not a rung of the ladder.  It is the
one thing [[prior-art-and-novelty-atlas-0.1]] lists as genuinely unpublished and
that nine tiers of measurement have claimed as a headline feature and spent on
nothing: **reverse-mode differentiation of a design objective back through a
composed rollout of frozen, independently-declared experts, across every
fluid-disk seam.**

What is reused, unchanged, and what is new
------------------------------------------

Everything in the forward model belongs to CS-7/CS-8 already:

  the agents      `wake_array.exposed_reference_solver` -- `reference.WindowNS`
                  with its elliptic part removed, which is R10's own
                  prescription and the only arrangement of the classical column
                  that survives 120 macro-steps (W100).
  the assembly    `wake_array.projected_assembly` -- an `assembly.ProjectedAssembly`,
                  i.e. the partition-of-unity blend AND one global Leray
                  projection on the assembled field, once, after the blend.
  the tiling      `scaling_ladder.rung(n_col, n_row).tiling` -- the same
                  128-cell windows at the same 16-cell overlap and the same
                  8-cell ramp.
  the disk        `disk.ActuatorDisk`'s local-induction closure, zero fitted
                  parameters, C_T' = 4a/(1-a), P = T <U_d>.

Three things are new and each is stated where it is done:

  1. **The column runs in torch instead of numpy.**  `WindowNS` was already
     written against `backend.py`'s array interface with a torch path, so this
     is a backend flag and not a re-implementation.  The ONE thing that had to
     move is `step_batch`'s last line, which calls `to_numpy` at the class
     boundary and would cut the tape; `_TapeBackend` below wraps the backend so
     that call is the identity.  Nothing in the build repo is edited, and
     `tests/test_tier21_wind_farm_design.py` pins the torch column BITWISE
     against the numpy one on a real `step_batch`.

  2. **The disks are placed continuously and can yaw.**  CS-7's disks sit on
     window faces at fixed x-planes and their body force is an exact
     cell-rectangle overlap, which is a staircase in the disk's position.  A
     design vector needs a placement whose derivative is not a staircase, so
     the force is projected with a Gaussian along the rotor axis and a smoothed
     top hat across it, then normalized DISCRETELY so the momentum sink is
     still exactly -T.  Gaussian projection is the standard actuator-disk /
     actuator-line smearing (Sorensen-Shen, Troldborg); it is not a new model.

  3. **Yaw is a closure and this file says so twice.**  A yawed disk here is a
     disk whose axis is rotated by gamma: it responds to the axis-normal
     inflow component U_n = u cos(gamma) + v sin(gamma) and pushes back along
     its own axis.  The lateral component -T sin(gamma) is what deflects the
     wake.  Nothing in the checkpoint or in `WindowNS` knows about a yawed
     rotor; the deflection that comes out is the composed model's response to a
     rotated momentum sink, and the cos^3(gamma) power law falls out of the
     same rotation rather than being imposed.  FLORIS does the same thing with
     an engineering deflection law and calls it a closure.  So does this.

What the gradient is taken through
-----------------------------------

One macro-step is

    fx, fy  <- the disks, reading their own inflow off the CURRENT global field
    us, vs  <- tiling.cut(u, v)                              [W windows]
    u1, v1  <- exposed_reference_solver.step_batch(...)      [~20 sub-steps]
    au, av  <- partition-of-unity blend                      [the assembly]
    au, av  <- one global Leray projection on the padded domain  [L6/C2]
    u,  v   <- inlet and both laterals held at freestream

and the rollout is 40-60 of those from the freestream.  `d J / d theta` is
reverse-mode autograd through all of it: through every sub-step of every
window's local solve, through the blend, through the spectral projection, and
through the disk closure at both ends -- the inflow read that sets the thrust
and the body force that the fluid sees.  The seam is inside the differentiated
path, which is the whole claim.

Cost, measured on the development box (22 cores, torch 2.7.1 CPU, float64)
--------------------------------------------------------------------------

Per macro-step at 12 windows: numpy forward 2.97 s, torch forward 1.05 s, torch
forward + backward 4.87 s.  So the torch column is ~2.8x FASTER than the numpy
one that CS-7 and CS-8 ran, and one gradient costs ~4.6 forward evaluations.
The rollout is checkpointed one macro-step at a time (`torch.utils.checkpoint`),
because the un-checkpointed tape for a 50-step rollout at 12 windows is ~10^2 GB
and this machine has no GPU.

What this file does NOT do
--------------------------

It does not claim accuracy.  The checkpoint over-dissipates wakes (OP-3) and
this column is the classical reference solver at Re_D = 255, so a power figure
here is a number about the composed model.  Every number this module produces is
"as measured by the composed model" and `results.md` is required to say so.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Callable, Sequence

import numpy as np

from . import wake_array as wa
from . import scaling_ladder as sl
from .window_ns import _no_projection_class

__all__ = [
    "TORCH_DTYPE", "BAND", "YAW_MAX", "S_MIN", "A_INDUCTION",
    "tape_solver", "torch_solver_class",
    "TapedPoseidon", "taped_poseidon", "PoseidonRollout",
    "EXPERT_KINDS", "rollout_for",
    "FarmCase", "case_k12", "case_k25", "case_for",
    "DiskBank", "Rollout", "RolloutResult",
    "objective", "value_and_grad", "value_and_grad_local",
    "project_design",
    "adam_optimise", "cma_es", "fd_gradient", "fd_check",
    "initial_design", "power_of", "design_to_dict", "dict_to_design",
]


# ---------------------------------------------------------------------------
# constants -- every one of them either CS-7's or declared here
# ---------------------------------------------------------------------------

#: float64 throughout.  float32 buys ~15% on this box (measured) and costs the
#: finite-difference check its two spare digits, which is the wrong trade for a
#: run whose whole point is that the gradient is right.  OP-6's batch-position
#: sensitivity is a float32 phenomenon in the CHECKPOINT; the classical column
#: does not have it, and pinning the batch layout (`Rollout` never reorders
#: `tiling.offsets`) is what makes that statement checkable.
try:                                            # torch is imported lazily
    import torch
    import torch.utils.checkpoint            # noqa: F401
    TORCH_DTYPE = torch.float64
except Exception:                               # pragma: no cover
    torch = None
    TORCH_DTYPE = None

#: Cells of inlet and lateral boundary held at the freestream, exactly as
#: `scripts/w100_scaling_ladder.py:_band`.  The outlet is left free.
BAND = 8

#: Yaw bound, radians.  30 degrees is the wake-steering literature's operating
#: limit (beyond it the actuator-disk momentum balance and the yaw actuator both
#: leave their envelope) and it is a BOX constraint, not a penalty.
YAW_MAX = math.radians(30.0)

#: Minimum turbine separation, rotor diameters.  2 D is a hard packing limit
#: rather than a realistic layout rule -- real farms are at 4-8 D -- and it is
#: deliberately loose so the optimiser is free to find the layout instead of
#: being told it.
S_MIN = 2.0

#: The disk's axial induction.  Fixed, not a design variable: the spec's design
#: vector is (x, y, gamma) and `disk.ActuatorDisk` is the zero-parameter expert
#: precisely because a is declared geometry.  a = 1/3 gives C_T' = 2.
A_INDUCTION = 1.0 / 3.0
C_T_PRIME = 4.0 * A_INDUCTION / (1.0 - A_INDUCTION)

#: Where the disk reads its own inflow, rotor diameters UPSTREAM along its own
#: axis.  `disk.disk_average`'s docstring is the reason: sampling inside the
#: strip reads a velocity the disk's own body force has already slowed, which is
#: the classic actuator-disk double count.  CS-7 used the same 0.25 D.
D_UPSTREAM = 0.25

#: Sample points across the rotor face for the disk average.  33 over 1 D at
#: dx = 1/32 D is one per cell.
N_DISK_SAMPLES = 33

#: Tip-speed ratio, for the ROT port only.  `disk.TIP_SPEED_RATIO`.
TIP_SPEED_RATIO = 7.5


# ---------------------------------------------------------------------------
# the agent: the same solver, with the tape left attached
# ---------------------------------------------------------------------------


class _TapeBackend:
    """`backend.TorchBackend` with `to_numpy` made the identity.

    `WindowNS.step_batch` ends with ``return self.b.to_numpy(u), ...`` -- a
    deliberate contract (backend.py: "everything outside WindowNS is numpy and
    stays numpy, so a GPU run changes one flag rather than the type of every
    array").  For an adjoint that call is exactly where the tape is cut, and it
    is the ONLY place it is cut: every other backend method is a torch op.

    So the repair is one method, and it is made HERE rather than in the build
    repo, for the same reason `window_ns._no_projection_class` subclasses rather
    than edits: the agent is not ours to change, and what the composition layer
    may do is decline to use one part of it.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def to_numpy(self, x: Any) -> Any:
        return x


@lru_cache(maxsize=1)
def torch_solver_class():
    """`WindowNS` with the elliptic part removed AND the stencils out-of-place.

    The base is `window_ns._no_projection_class()`, i.e. the EXPOSED solver R10
    prescribes and `wake_array.exposed_reference_solver` instantiates.  The three
    overrides below replace ``empty_like`` plus slice assignment with ``cat``:
    the same arithmetic in the same order, so the forward is bitwise identical
    (asserted in the tests), but the backward of a `cat` is a slice while the
    backward of an in-place slice write is a scatter.  Measured: 4.87 s against
    6.15 s per gradient macro-step at 12 windows, ~20%.

    The numpy path is untouched -- these run only when the backend is torch,
    because they are written in torch.
    """
    base = _no_projection_class()

    class TapedWindowNS(base):                                  # type: ignore[misc]
        def _ddx(self, f):
            c = 1.0 / (2.0 * self.h)
            mid = (f[:, :, 2:] - f[:, :, :-2]) * c
            lo = (f[:, :, 1:2] - f[:, :, 0:1]) * c
            hi = (f[:, :, -1:] - f[:, :, -2:-1]) * c
            return torch.cat((lo, mid, hi), dim=2)

        def _ddy(self, f):
            c = 1.0 / (2.0 * self.h)
            mid = (f[:, 2:, :] - f[:, :-2, :]) * c
            lo = (f[:, 1:2, :] - f[:, 0:1, :]) * c
            hi = (f[:, -1:, :] - f[:, -2:-1, :]) * c
            return torch.cat((lo, mid, hi), dim=1)

        def _poisson(self, rhs):
            """The parent's arithmetic with the per-sub-step GPU sync removed.

            The parent records `last_mismatch` as a python float, which is an
            `.item()` -- a device-to-host synchronisation -- and it runs once per
            SUB-STEP, twenty times a macro-step.  On CPU it is free; on CUDA it
            serialises the whole pipeline.  The diagnostic is kept, as a tensor,
            so nothing is lost and nothing is waited for.  The returned `phi` is
            bitwise the parent's -- asserted in the tests.
            """
            m = self.b.mean_keepdims(rhs, (1, 2))
            self.last_mismatch = m.detach().abs().max()
            r = rhs - m
            ph = self.b.dct2(r) / self._lam_safe
            ph[:, 0, 0] = 0.0
            return self.b.idct2(ph)

        def _lap(self, f):
            c = 1.0 / self.h ** 2
            core = (f[:, 1:-1, 2:] + f[:, 1:-1, :-2] + f[:, 2:, 1:-1]
                    + f[:, :-2, 1:-1] - 4.0 * f[:, 1:-1, 1:-1]) * c
            b, ny, nx = f.shape
            col = torch.zeros((b, ny - 2, 1), dtype=f.dtype, device=f.device)
            row = torch.zeros((b, 1, nx), dtype=f.dtype, device=f.device)
            return torch.cat((row, torch.cat((col, core, col), dim=2), row), dim=1)

    return TapedWindowNS


@lru_cache(maxsize=4)
def tape_solver(nu: float = wa.NU_REF, device: str = "cpu"):
    """The differentiable twin of `wake_array.exposed_reference_solver`.

    Same class family, same nu, same window, same cfl, same transmission -- the
    only differences are the backend and the tape.  `test_tier21` asserts the
    two agree to the bit on a real `step_batch`, which is what lets this file
    say it is marching CS-7's column rather than one that resembles it.
    """
    s = torch_solver_class()(nu=nu, length=wa.S_LEN, n=wa.N, cfl=0.4,
                             transmission="dirichlet", backend="torch",
                             device=device)
    s.b = _TapeBackend(s.b)
    return s


# ---------------------------------------------------------------------------
# the OTHER agent: the frozen checkpoint, with the tape left attached
# ---------------------------------------------------------------------------


class TapedPoseidon:
    """`FrozenFluidExpert` made differentiable, without editing the build repo.

    The wrapper in the build repo cuts the tape in three separate places and
    every one of them is deliberate -- its own docstring says "so that a gradient
    cannot be taken by accident":

      1. it takes numpy in and converts,
      2. it runs the forward inside ``with torch.no_grad()``,
      3. it calls ``.detach().cpu().numpy()`` on the output.

    Taking one on PURPOSE is what this class is for, and the repair is made here
    for the same reason `_TapeBackend` and `window_ns._no_projection_class` are
    made here: **the agent is not ours to change, and what the composition layer
    may do is decline to use one part of it and supply that part itself.**

    The arithmetic is `FrozenFluidExpert.step_many(galilean=False, project=False)`
    re-expressed in torch, in the same order, with the same normalization
    constants read off the adapter rather than re-typed --
    `tests/test_tier26_poseidon_design.py` asserts the two agree **bitwise** on a
    real batched call.  ``galilean=False`` is not a choice made here: it is the
    W98 repair, which moved transport and pressure out of the per-window call and
    into the composition layer, where the graph's own declaration already put
    them (`wake_array.transport_and_project`).

    **The parameters stay frozen.**  `FrozenFluidExpert.__init__` applied
    ``requires_grad_(False)`` to all 20.8 M of them and nothing here undoes it,
    so the backward pass reaches ``theta`` through the activations and
    accumulates **no** parameter gradient.  ``n_grad_params`` is asserted zero in
    the tests, because "frozen" is the load-bearing word in the claim this run
    exists to support.

    **float32 is the checkpoint's, not a choice.**  The weights are float32, so
    the model boundary is where this column's float64 stops.  Everything outside
    the boundary -- the encode, the decode, the mean restore, the blend, the
    projection, the disks -- stays float64.  The consequence is OP-6's, and it is
    the reason `fd_check` has to sweep ``h`` rather than trust the classical
    column's optimum: a function computed through a float32 forward pass has a
    relative noise floor around 1e-7, so a central difference below
    ``h ~ 1e-3`` is measuring cancellation rather than a derivative.
    """

    #: Whether the reduced-precision matmul paths have been turned OFF.  See
    #: `_pin_precision`.
    tf32_pinned = False

    @staticmethod
    def _pin_precision() -> dict:
        """Turn OFF TF32, and say so in the record.

        On an Ampere or later card torch runs float32 matmuls in **TF32** by
        default: 10 mantissa bits, so a relative error around 1e-3.  The
        classical column is float64 and never met this; the checkpoint is
        float32 and its every matmul is on that path.  Left on, it would move J
        in its fourth significant figure between CPU and GPU, and a
        finite-difference check of a function that moves in its fourth digit is
        measuring the arithmetic, not the derivative -- the same failure mode as
        `scatter_add`'s nondeterminism, one order of magnitude worse.

        So it is turned off here, once, where the float32 expert is built, and
        the state is recorded rather than assumed.
        """
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.backends.cudnn.benchmark = False
        except Exception:                                   # pragma: no cover
            pass
        TapedPoseidon.tf32_pinned = True
        return {"cuda_matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
                "float32_matmul_precision": torch.get_float32_matmul_precision()}

    def __init__(self, expert: Any, device: str = "cpu") -> None:
        self.precision = self._pin_precision()
        self.expert = expert
        self.device = device
        self.model = expert.model
        self.scaling = expert.scaling
        ad = _adapters()
        opt = dict(dtype=TORCH_DTYPE, device=device)
        self._mean = torch.as_tensor(ad.EXPERT_MEAN, **opt)[None, :, None, None]
        self._std = torch.as_tensor(ad.EXPERT_STD, **opt)[None, :, None, None]
        self._rho = float(ad.EXPERT_RHO)
        self._p = float(ad.EXPERT_P)
        self.n_params = int(sum(p.numel() for p in self.model.parameters()))
        self.n_grad_params = int(sum(p.numel() for p in self.model.parameters()
                                     if p.requires_grad))
        self.n_calls = 0

    def lead(self, dt: float) -> float:
        return float(self.scaling.lead(dt))

    def step_batch(self, uf, vf, dt: float):
        """``[B,128,128]`` fluctuation in, the advanced fluctuation out.

        No Galilean translation and no projection -- both are global and both
        belong to the composition layer (W98).  No ``force`` either, and that is
        deliberate rather than an omission: `step_many` **drops** ``force``
        silently when ``galilean`` is off (**W99**), so a caller that passed one
        here would get a forward pass with no body force and no error.  The disks
        are applied by `PoseidonRollout.macro_step`, after the assembly, where
        the partition of unity summing to one makes it identical to applying
        them per window.
        """
        b = uf.shape[0]
        s = self.scaling.velocity
        rho = torch.full_like(uf, self._rho)
        pr = torch.full_like(uf, self._p)
        fields = torch.stack([rho, uf / s, vf / s, pr], dim=1)
        x = ((fields - self._mean) / self._std).to(torch.float32)
        t = torch.full((b,), self.lead(dt), dtype=torch.float32,
                       device=x.device)
        out = self.model(pixel_values=x, time=t).output
        self.n_calls += b
        arr = out.to(TORCH_DTYPE) * self._std + self._mean
        return arr[:, 1] * s, arr[:, 2] * s


@lru_cache(maxsize=1)
def _adapters():
    import importlib
    wa.load_reference()
    return importlib.import_module("atlas_windfarm_reference.adapters")


@lru_cache(maxsize=4)
def taped_poseidon(device: str = "cpu", threads: int = 8) -> "TapedPoseidon":
    """The differentiable twin of `wake_array.scaled_expert`.

    Same checkpoint, same `Scaling(length=S_LEN, velocity=2.0)`, same native lead
    of 0.1 -- the only difference is that the tape survives the call.
    """
    return TapedPoseidon(wa.scaled_expert(device=device, threads=threads), device)


# ---------------------------------------------------------------------------
# the case: a rung, a turbine count, a box, a start
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FarmCase:
    """One design problem: which rung it lives on and what may move on it.

    ``n_col``/``n_row`` are the CS-7 ladder's, so the domain, the overlap, the
    ramp and the macro-step are the ones already measured rather than new
    geometry.  ``k_col``/``k_row`` are the TURBINE grid, which is independent of
    the window grid -- the disks are placed continuously and know nothing about
    where the cuts are, which is the point.
    """

    name: str
    n_col: int
    n_row: int
    k_col: int
    k_row: int
    steps: int = 50
    #: Macro-steps averaged at the end of the rollout to form J.  The state is
    #: quasi-steady rather than converged (spec section 5), so the objective is
    #: a short time average and not a single snapshot.
    avg_window: int = 5
    #: Clearance from every domain wall, rotor diameters.
    margin: float = 1.5
    #: Extra clearance from the OUTLET, where a turbine's own wake would leave
    #: the domain before it had formed.
    outlet_margin: float = 2.0
    yaw_max: float = YAW_MAX
    s_min: float = S_MIN
    #: Spacing-penalty weight.  Chosen so a 0.5 D violation costs 5 power units,
    #: which is ~40% of the K=12 array's total: firm, but a penalty rather than
    #: a wall, because the projection below is the wall.
    lam_spacing: float = 20.0

    @property
    def k(self) -> int:
        return self.k_col * self.k_row

    @property
    def tiling(self) -> wa.ArrayTiling:
        return sl.rung(self.n_col, self.n_row).tiling

    @property
    def shape(self) -> tuple[int, int]:
        t = self.tiling
        return (t.ny, t.nx)

    @property
    def n_windows(self) -> int:
        return self.n_col * self.n_row

    @property
    def extent(self) -> tuple[float, float]:
        """(Lx, Ly) in rotor diameters."""
        t = self.tiling
        return (t.nx * wa.DX, t.ny * wa.DX)

    @property
    def box(self) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi) for the flat design vector, shape (3K,)."""
        lx, ly = self.extent
        lo = np.empty(3 * self.k)
        hi = np.empty(3 * self.k)
        lo[0::3], hi[0::3] = self.margin, lx - self.outlet_margin
        lo[1::3], hi[1::3] = self.margin, ly - self.margin
        lo[2::3], hi[2::3] = -self.yaw_max, self.yaw_max
        return lo, hi

    def as_dict(self) -> dict[str, Any]:
        lx, ly = self.extent
        return {
            "name": self.name, "K": self.k, "k_col": self.k_col,
            "k_row": self.k_row, "rung": f"N{self.n_windows}",
            "n_col": self.n_col, "n_row": self.n_row,
            "n_windows": self.n_windows, "shape": list(self.shape),
            "domain_D": [lx, ly], "steps": self.steps,
            "avg_window": self.avg_window, "n_design": 3 * self.k,
            "yaw_max_deg": math.degrees(self.yaw_max), "s_min_D": self.s_min,
            "lam_spacing": self.lam_spacing,
            "dx_D": wa.DX, "macro_dt": wa.MACRO_DT, "nu": wa.NU_REF,
            "halo_cells": wa.HALO, "ramp_cells": self.tiling.ramp,
        }


def case_k12(steps: int = 50) -> FarmCase:
    """K = 12 on the ladder's N12 rung: 14.5 D x 11.0 D, 4 columns x 3 rows.

    N12 is the largest rung `w100_scaling_ladder.stage_long_march` marched to
    120 macro-steps with the exposed agents and the projected assembly, and it
    was stable there, so a 50-step rollout is inside a confirmed band.
    """
    return FarmCase("K12", n_col=4, n_row=3, k_col=4, k_row=3, steps=steps)


def case_k25(steps: int = 40) -> FarmCase:
    """K = 25 on the ladder's N24 rung: 21.5 D x 14.5 D, 5 x 5.

    **N24 was never marched past 20 macro-steps in the assembly work.**  So the
    driver runs a confirmation march at this rung before it optimises, and the
    result is reported whatever it says.

    40 macro-steps rather than K12's 50, and the reason is cost and is stated
    rather than hidden: this rung is twice the windows, so a gradient costs
    twice as much per macro-step, and 40 is the low end of the spec's own
    40-60 band.  `--stage sensitivity` measures what the difference between 40,
    50 and 60 is worth at K12, where all three are affordable, so the size of
    what this choice costs is on the record beside it.
    """
    return FarmCase("K25", n_col=6, n_row=4, k_col=5, k_row=5, steps=steps)


def case_for(name: str, steps: int | None = None) -> FarmCase:
    c = {"K12": case_k12, "K25": case_k25}[name.upper()]()
    return c if steps is None else replace(c, steps=steps)


def initial_design(case: FarmCase) -> np.ndarray:
    """The unoptimised layout: a regular grid, every turbine facing the wind.

    This is the thing the power gain is quoted AGAINST, so it is a rule and not
    a choice -- the turbine grid is centred laterally, starts `margin` + one
    spacing in from the inlet, and is spread evenly over what the box allows.
    """
    lx, ly = case.extent
    x_lo, x_hi = case.margin + 1.0, lx - case.outlet_margin - 1.5
    xs = (np.linspace(x_lo, x_hi, case.k_col) if case.k_col > 1
          else np.array([0.5 * (x_lo + x_hi)]))
    span = ly - 2.0 * (case.margin + 0.75)
    ys = (np.linspace(0.5 * ly - 0.5 * span, 0.5 * ly + 0.5 * span, case.k_row)
          if case.k_row > 1 else np.array([0.5 * ly]))
    theta = np.zeros((case.k, 3))
    m = 0
    for j in range(case.k_row):
        for i in range(case.k_col):
            theta[m] = (xs[i], ys[j], 0.0)
            m += 1
    return theta.reshape(-1)


def design_to_dict(theta: np.ndarray) -> list[dict[str, float]]:
    t = np.asarray(theta, dtype=float).reshape(-1, 3)
    return [{"x": float(a), "y": float(b), "yaw_deg": math.degrees(float(c))}
            for a, b, c in t]


def dict_to_design(rows: Sequence[dict]) -> np.ndarray:
    return np.array([[r["x"], r["y"], math.radians(r["yaw_deg"])] for r in rows]
                    ).reshape(-1)


# ---------------------------------------------------------------------------
# the disks: continuous placement, yaw, and a momentum sink that still sums to T
# ---------------------------------------------------------------------------


@dataclass
class DiskBank:
    """K yawed actuator disks on one domain, differentiable in (x, y, gamma).

    Closure, per disk, and every line of it is `disk.ActuatorDisk`'s except the
    rotation:

        U_n  = < u cos g + v sin g >   over the rotor face, 0.25 D upstream
        T    = 1/2 rho A C_T' U_n^2                       [C_T' = 4a/(1-a) = 2]
        P    = T U_n                                       [= tau omega]
        f    = -T * kernel(x, y, g) * (cos g, sin g)

    `U_n` is the AXIS-NORMAL inflow, so a yawed disk sees less of the stream and
    its power falls as cos^3(g) without that law being written down anywhere:
    U_n ~ U cos g and P ~ U_n^3.  The lateral force -T sin g is what steers the
    wake, and it is a real momentum source in the composed rollout rather than a
    displacement added to a wake profile afterwards.

    **The kernel.**  Gaussian along the axis with sigma_s = Delta_d / 2, smoothed
    top hat across it with a one-cell edge, evaluated on a local box and
    normalized on that box so that

        sum_cells f * dx * dy  ==  -T * (cos g, sin g)     exactly

    which is gate W2's property and is asserted in the tests.  Delta_d is the
    smearing thickness CS-7 derives, `<U_d> dt`, so the impulse a crossing parcel
    receives matches what momentum theory allows it to lose -- and because that
    makes it part of the model rather than a knob, the adjoint carries its
    dependence on the inflow too.  See `forcing`.
    """

    case: FarmCase
    device: str = "cpu"

    def __post_init__(self) -> None:
        ny, nx = self.case.shape
        self.ny, self.nx = ny, nx
        self.dx = wa.DX
        self.lx, self.ly = nx * wa.DX, ny * wa.DX
        opt = dict(dtype=TORCH_DTYPE, device=self.device)
        self._x_c = torch.arange(nx, **opt) * self.dx + 0.5 * self.dx
        self._y_c = torch.arange(ny, **opt) * self.dx + 0.5 * self.dx
        # rotor-face sample offsets, one per cell across 1 D
        self._t = torch.linspace(-0.5 * wa.ROTOR_D, 0.5 * wa.ROTOR_D,
                                 N_DISK_SAMPLES, **opt)
        #: half-width of the local stamping box, in cells.  1.25 D each way is
        #: >6 sigma_s along the axis at the largest Delta_d this case study
        #: reaches and > (0.5 D + 4 edge cells) across it, at any yaw.
        self.box_cells = int(round(1.25 * wa.ROTOR_D / self.dx))
        self._check_box(self.box_cells)

    def _check_box(self, b: int) -> None:
        """The box must fit inside the domain, and it is an error if it does not.

        Silently shrinking it would silently weaken the one claim the box has to
        support -- that what it cuts off is below float64 -- so a domain too
        narrow to hold a turbine's stamping box raises here rather than
        producing an out-of-range scatter thirty macro-steps later.
        """
        if 2 * b + 1 > min(self.ny, self.nx):
            raise ValueError(
                f"the disk's stamping box is {2 * b + 1} cells and the domain is "
                f"{self.nx} x {self.ny}: no placement leaves the Gaussian tail "
                "inside the grid")

    # -- the inflow read ---------------------------------------------------

    def inflow(self, u, v, theta):
        """`U_n` per disk: bilinear, so it is differentiable in the position.

        `disk.disk_average` takes ``argmin |x_c - x_face|`` and a boolean row
        mask, which is a step function of the disk's position and would give
        this whole objective a zero derivative in x almost everywhere.  Bilinear
        sampling on the rotor's own face is the same quantity with the staircase
        removed, and at gamma = 0 on a cell-aligned disk the two agree to the
        interpolation error.
        """
        t = theta.reshape(-1, 3)
        x, y, g = t[:, 0], t[:, 1], t[:, 2]
        cg, sg = torch.cos(g), torch.sin(g)
        # face centre 0.25 D upstream along the axis, then out along the span
        px = (x - D_UPSTREAM * cg)[:, None] - sg[:, None] * self._t[None, :]
        py = (y - D_UPSTREAM * sg)[:, None] + cg[:, None] * self._t[None, :]
        gx = 2.0 * px / self.lx - 1.0
        gy = 2.0 * py / self.ly - 1.0
        grid = torch.stack((gx, gy), dim=-1)[None]              # [1, K, S, 2]
        fld = torch.stack((u, v))[None]                          # [1, 2, ny, nx]
        s = torch.nn.functional.grid_sample(
            fld, grid, mode="bilinear", padding_mode="border", align_corners=False)
        us, vs = s[0, 0], s[0, 1]                                # [K, S]
        return (us * cg[:, None] + vs * sg[:, None]).mean(dim=1)

    # -- the body force ----------------------------------------------------

    def forcing(self, u, v, theta):
        """``(fx, fy, U_n, T, P)`` for the whole domain at this state.

        The kernel is stamped on a per-disk box whose integer corner is taken
        from ``theta.detach()``.  The box is a window on the lattice, not a
        parameter: within one rollout theta is fixed, so the corner is a
        constant and every quantity that varies with theta -- the offsets inside
        the box, the Gaussian, the normalization -- carries its derivative.
        """
        t = theta.reshape(-1, 3)
        x, y, g = t[:, 0], t[:, 1], t[:, 2]
        cg, sg = torch.cos(g), torch.sin(g)
        u_n = self.inflow(u, v, theta)
        u_pos = torch.clamp(u_n, min=1e-3)
        thrust = 0.5 * C_T_PRIME * u_pos ** 2                    # rho = A = 1
        power = thrust * u_pos

        # -- the smearing thickness, and it is NOT detached ------------------
        # `Delta_d = <U_d> dt` is CS-7's derivation, not a free knob: it is the
        # thickness at which the impulse a parcel receives crossing the strip
        # equals what momentum theory allows it to lose.  So its dependence on
        # the inflow is part of the model and the adjoint carries it.  Detaching
        # it was tried first and the finite-difference check found it at once --
        # a systematic 0.2% deficit in every component, which is exactly the
        # size of the path that had been cut.  **That is what OP-6's check is
        # for**, and it is on the record because it caught something.
        delta = torch.clamp(u_pos, min=0.05) * wa.MACRO_DT
        sigma_s = torch.clamp(0.5 * delta, min=2.0 * self.dx)
        sigma_n = 0.5 * self.dx

        # -- the local box ---------------------------------------------------
        b = self.box_cells
        self._check_box(b)
        ix0 = torch.clamp((x.detach() / self.dx - 0.5).round().long() - b,
                          0, self.nx - 1)
        iy0 = torch.clamp((y.detach() / self.dx - 0.5).round().long() - b,
                          0, self.ny - 1)
        ix0 = torch.clamp(ix0, max=self.nx - (2 * b + 1))
        iy0 = torch.clamp(iy0, max=self.ny - (2 * b + 1))
        off = torch.arange(2 * b + 1, device=x.device)
        cx = self._x_c[ix0[:, None] + off[None, :]]              # [K, W]
        cy = self._y_c[iy0[:, None] + off[None, :]]              # [K, W]

        dxg = cx[:, None, :] - x[:, None, None]                  # [K, 1, W]
        dyg = cy[:, :, None] - y[:, None, None]                  # [K, W, 1]
        s_ax = dxg * cg[:, None, None] + dyg * sg[:, None, None]
        s_sp = -dxg * sg[:, None, None] + dyg * cg[:, None, None]
        ker = (torch.exp(-0.5 * (s_ax / sigma_s[:, None, None]) ** 2)
               * torch.sigmoid((0.5 * wa.ROTOR_D - s_sp.abs()) / sigma_n))
        ker = ker / (ker.sum(dim=(1, 2), keepdim=True) * self.dx ** 2)

        amp = -thrust[:, None, None] * ker
        corners = [(int(a_), int(b_)) for a_, b_ in zip(ix0.tolist(), iy0.tolist())]
        ax, ay = amp * cg[:, None, None], amp * sg[:, None, None]
        k = amp.shape[0]
        fx = _place([ax[i] for i in range(k)], corners, self.ny, self.nx)
        fy = _place([ay[i] for i in range(k)], corners, self.ny, self.nx)
        return fx, fy, u_n, thrust, power

    # -- the ROT port ------------------------------------------------------

    @staticmethod
    def rot_port(u_n, power):
        """(torque, omega) with tau * omega == P: `disk.py`'s own split.

        The port is open -- nothing is connected to it -- so only the product
        enters the objective, and the split is carried so that PoC 1b has
        something to attach a generator to (`port-algebra` 5.1).
        """
        omega = TIP_SPEED_RATIO * torch.clamp(u_n, min=1e-3) / (0.5 * wa.ROTOR_D)
        return power / omega, omega


# ---------------------------------------------------------------------------
# the composition layer, in torch
# ---------------------------------------------------------------------------


def _place(parts, offsets, ny: int, nx: int):
    """Accumulate local patches into a global field in a FIXED order.

    `Tensor.scatter_add` is the obvious way to do this and it is the wrong one:
    on CUDA it is implemented with `atomicAdd`, so the summation order of
    overlapping contributions is whatever the scheduler produced that run, and
    the objective stops being bit-reproducible.  Nothing else in this PoC
    survives that -- a finite difference of a function that changes in the last
    digit between calls measures the noise, and CMA-ES ranking a population by a
    number that moves is a different algorithm.

    So the patches are padded to the full field and added in the order the
    caller supplies, which is `tiling.offsets` for the assembly and turbine
    index for the disks.  It costs one extra kernel per patch, it is
    deterministic on both devices, and it is the operational content of OP-6's
    "pin the batch layout".
    """
    out = None
    for pt, (ox, oy) in zip(parts, offsets):
        h, w = pt.shape[-2], pt.shape[-1]
        q = torch.nn.functional.pad(pt, (int(ox), nx - int(ox) - w,
                                         int(oy), ny - int(oy) - h))
        out = q if out is None else out + q
    return out


class Rollout:
    """The composed march, differentiable, with a PINNED batch layout.

    Every piece is `wake_array`'s, re-expressed in torch and pinned against the
    numpy original in `tests/test_tier21_wind_farm_design.py`:

      `cut`      `ArrayTiling.cut`               -- windows in `tiling.offsets`
                                                    order, ALWAYS, which is what
                                                    OP-6's "pin the batch layout"
                                                    means operationally.
      `blend`    `ArrayTiling.assemble`          -- the partition of unity, whose
                                                    weights come off the numpy
                                                    object rather than being
                                                    recomputed here.
      `project`  `wake_array.project_assembled`  -- the L6/C2 global Leray
                                                    projection on the domain
                                                    extended by PAD_CELLS of
                                                    tapered fluctuation.
      `band`     `w100_scaling_ladder._band`     -- inlet and both laterals at
                                                    freestream, outlet free.
    """

    def __init__(self, case: FarmCase, device: str = "cpu",
                 checkpoint: bool = True) -> None:
        self.case = case
        self.device = device
        self.checkpoint = checkpoint
        self.tiling = case.tiling
        self.ny, self.nx = case.shape
        self.solver = tape_solver(wa.NU_REF, device)
        self.disks = DiskBank(case, device)
        #: The declared assembly object.  It is not called from the hot loop --
        #: its projection is `project_assembled`, which is numpy -- but it is
        #: constructed so the thing this rollout mirrors is on the record and so
        #: the tests can compare against it rather than against a description.
        self.assembly = wa.projected_assembly(self.tiling)
        opt = dict(dtype=TORCH_DTYPE, device=device)

        offs = self.tiling.offsets
        self.n_win = len(offs)
        n = wa.N
        idx = np.empty((self.n_win, n, n), dtype=np.int64)
        chi = np.empty((self.n_win, n, n), dtype=np.float64)
        w = self.tiling.weights()
        for k, (ox, oy) in enumerate(offs):
            rows, cols = np.meshgrid(np.arange(oy, oy + n),
                                     np.arange(ox, ox + n), indexing="ij")
            idx[k] = rows * self.nx + cols
            chi[k] = w[k][oy:oy + n, ox:ox + n]
        self._idx = torch.as_tensor(idx.reshape(-1), device=device)
        self._chi = torch.as_tensor(chi, **opt)
        self._offsets = offs

        # the freestream band mask
        m = np.zeros((self.ny, self.nx), dtype=bool)
        m[:, :BAND] = True
        m[:BAND, :] = True
        m[-BAND:, :] = True
        self._band = torch.as_tensor(m, device=device)

        # the projection's wavenumbers, built once
        pad = wa.PAD_CELLS
        nxp = self.nx + pad
        kx = 2.0 * np.pi * np.fft.fftfreq(nxp, d=wa.DX)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.ny, d=wa.DX)
        if nxp % 2 == 0:
            kx[nxp // 2] = 0.0
        if self.ny % 2 == 0:
            ky[self.ny // 2] = 0.0
        k2 = kx[None, :] ** 2 + ky[:, None] ** 2
        self._kx = torch.as_tensor(kx[None, :], **opt)
        self._ky = torch.as_tensor(ky[:, None], **opt)
        self._k2 = torch.as_tensor(np.where(k2 == 0.0, 1.0, k2), **opt)
        taper = np.cos(0.5 * np.pi * (np.arange(1, pad + 1) / pad)) ** 2
        self._taper = torch.as_tensor(taper[None, :], **opt)
        self._pad = pad

    # -- the four composition-layer operators ------------------------------

    def cut(self, f):
        n = wa.N
        return torch.stack([f[oy:oy + n, ox:ox + n] for ox, oy in self._offsets])

    def blend(self, us, vs):
        cu, cv = self._chi * us, self._chi * vs
        return (_place([cu[k] for k in range(self.n_win)], self._offsets,
                       self.ny, self.nx),
                _place([cv[k] for k in range(self.n_win)], self._offsets,
                       self.ny, self.nx))

    def project(self, u, v):
        """`wake_array.project_assembled`, in torch.

        The freestream is removed and restored here for the same reason the
        numpy version does it: a caller that had to remember would one day not.
        """
        uf, vf = u - wa.U_INF, v
        bu = torch.cat((uf, uf[:, -1:] * self._taper), dim=1)
        bv = torch.cat((vf, vf[:, -1:] * self._taper), dim=1)
        uh, vh = torch.fft.fft2(bu), torch.fft.fft2(bv)
        div = self._kx * uh + self._ky * vh
        uh = uh - self._kx * div / self._k2
        vh = vh - self._ky * div / self._k2
        bu = torch.fft.ifft2(uh).real
        bv = torch.fft.ifft2(vh).real
        return wa.U_INF + bu[:, :self.nx], bv[:, :self.nx]

    def band(self, u, v):
        z = torch.zeros((), dtype=u.dtype, device=u.device)
        return (torch.where(self._band, z + wa.U_INF, u),
                torch.where(self._band, z, v))

    # -- one macro-step ----------------------------------------------------

    def macro_step(self, u, v, theta):
        fx, fy, u_n, thrust, power = self.disks.forcing(u, v, theta)
        us, vs = self.cut(u), self.cut(v)
        fxs, fys = self.cut(fx), self.cut(fy)
        u1, v1 = self.solver.step_batch(us, vs, wa.MACRO_DT, bc0=None,
                                        force=(fxs, fys))
        au, av = self.blend(u1, v1)
        au, av = self.project(au, av)
        u2, v2 = self.band(au, av)
        return u2, v2, power, u_n

    # -- the march ---------------------------------------------------------

    def run(self, theta, steps: int | None = None, grad: bool = False,
            keep_fields: Sequence[int] | None = None,
            progress: Callable[[int, float], None] | None = None):
        """March from the freestream and return `RolloutResult`.

        Checkpointed one macro-step at a time when ``grad`` is on: the tape for
        an un-checkpointed 50-step rollout at 12 windows is ~10^2 GB, and the
        recompute costs one extra forward, which the measured 4.6x
        gradient-to-forward ratio already includes.
        """
        steps = self.case.steps if steps is None else steps
        keep = set(keep_fields or ())
        u = torch.full((self.ny, self.nx), wa.U_INF, dtype=TORCH_DTYPE,
                       device=self.device)
        v = torch.zeros_like(u)
        avg = min(self.case.avg_window, steps)
        first = steps - avg
        p_trace, un_trace, fields = [], [], {}
        j_sum = None
        t0 = time.perf_counter()
        for s in range(steps):
            if s in keep:
                fields[s] = (u.detach().cpu().numpy().copy(),
                             v.detach().cpu().numpy().copy())
            if grad and self.checkpoint:
                u, v, power, u_n = torch.utils.checkpoint.checkpoint(
                    self.macro_step, u, v, theta, use_reentrant=False)
            else:
                u, v, power, u_n = self.macro_step(u, v, theta)
            if s >= first:
                j_sum = power.sum() if j_sum is None else j_sum + power.sum()
            p_trace.append(power.detach().cpu().numpy().copy())
            un_trace.append(u_n.detach().cpu().numpy().copy())
            if not torch.isfinite(u).all() or not torch.isfinite(v).all():
                raise RuntimeError(
                    f"{self.case.name} rollout is not finite at macro-step {s}; "
                    "raised where it happened so the NaN is not carried into "
                    "every number after it")
            if progress is not None:
                progress(s, time.perf_counter() - t0)
        if steps in keep or -1 in keep:
            fields[steps] = (u.detach().cpu().numpy().copy(),
                             v.detach().cpu().numpy().copy())
        j_power = j_sum / avg
        return RolloutResult(
            power=j_power, power_trace=np.array(p_trace),
            inflow_trace=np.array(un_trace), fields=fields,
            u=u, v=v, wall_s=time.perf_counter() - t0, steps=steps)




class PoseidonRollout(Rollout):
    """The same composed march with the FROZEN CHECKPOINT as the fluid agent.

    This is the half of [[prior-art-and-novelty-atlas-0.1]] section 2.1 that
    `Rollout` does not exercise.  Everything outside the fluid agent is
    unchanged and inherited rather than re-written: the tiling, the disks, the
    partition-of-unity blend, the freestream band, the objective, the optimiser,
    the constraint projection and `_place`'s fixed accumulation order are the
    same objects the classical column used.

    **Three things differ, and all three are the checkpoint's, not choices made
    here.**

    1. **The agent advances the FLUCTUATION, not the field.**  W0 experiment E1
       measured that this checkpoint does not preserve a uniform flow -- fed
       ``u = 1`` it returns a mean of 0.969 after one call and 0.281 after forty,
       because its pretraining distribution has zero mean flow by construction.
       So the mean is carried by the composition layer and the checkpoint is
       given ``u - U_INF``.

    2. **Transport and pressure are global, and are applied once, after the
       blend (W98).**  `step_many(galilean=True)` translates each window with a
       spectral shift that is periodic ON THE WINDOW, so a wake leaving a
       window's outflow edge re-enters its own inflow edge -- measured as a 25%
       velocity deficit 3.5 D UPSTREAM of a lone turbine, where nothing causes
       one.  `wake_array.transport_and_project` is the repair and
       `transport_project` below is its torch twin, pinned against it in the
       tests.  Note what this means for the comparison: the classical column's
       agent does its own advection (it is a Navier-Stokes solver) and needs
       only the Leray projection from the composition layer; the checkpoint
       needs the ADVECTION as well.  That is a property of the expert, and it is
       the reason this class overrides `project` rather than reusing it.

    3. **The disks are applied after the assembly (W99).**  `step_many` drops
       ``force`` silently when ``galilean`` is off, so passing one into the
       agent would produce a forward pass with no body force and no error.  The
       partition of unity sums to one on every cell, so ``+ f dt`` after the
       blend is arithmetically ``+ f dt`` before it, and this is what
       `scripts/w93_wake_array.py` already does for the same reason.

    **What is NOT changed, and is the point:** `assembly.ProjectedAssembly`'s
    internals, `wake_array.exposed_reference_solver`, and every declaration in
    `wake_array.build`.  The expert is swapped; nothing the expert is composed
    BY is touched.
    """

    kind = "poseidon"

    def __init__(self, case: FarmCase, device: str = "cpu",
                 checkpoint: bool = True, threads: int = 8) -> None:
        super().__init__(case, device=device, checkpoint=checkpoint)
        self.expert = taped_poseidon(device, threads)
        # `solver` is the CLASSICAL agent and this column does not use it.  It is
        # dropped rather than left in place so that a code path that reaches for
        # it raises instead of silently marching the wrong expert.
        self.solver = None
        # The transport phase.  `transport_and_project` keeps the Nyquist mode
        # for the TRANSLATION (an interpolation) and zeroes it for the
        # PROJECTION (a derivative) -- `adapters._wavenumbers`' own distinction,
        # at domain scale.  `Rollout.__init__` already built the zeroed pair, so
        # only the un-zeroed pair is new.
        nxp = self.nx + wa.PAD_CELLS
        self._kx_shift = 2.0 * np.pi * np.fft.fftfreq(nxp, d=wa.DX)[None, :]
        self._ky_shift = 2.0 * np.pi * np.fft.fftfreq(self.ny, d=wa.DX)[:, None]
        self.free_u, self.free_v = wa.U_INF, 0.0
        self._rebuild_phase()

    def _rebuild_phase(self) -> None:
        """The translation the composition layer owes a non-advecting expert.

        Split out rather than inlined because the freestream is settable: the
        demo points and scales it, and a phase built once at construction would
        keep translating the field by the freestream the rollout was BUILT with
        while the band imposed a different one -- a mismatch that looks like a
        physical result and is not.
        """
        ph = np.exp(-1j * wa.MACRO_DT
                    * (self._kx_shift * self.free_u + self._ky_shift * self.free_v))
        self._phase = torch.as_tensor(ph, dtype=torch.complex128,
                                      device=self.device)

    # -- the composition layer's two GLOBAL operators, in one transform ------

    def transport_project(self, uf, vf):
        """`wake_array.transport_and_project`, in torch, on the FLUCTUATION.

        One padded transform: project (Nyquist-zeroed wavenumbers), then
        translate by ``U_INF dt`` (Nyquist kept), then come back.  Same order and
        same conventions as the numpy original, which the tests pin it against.
        """
        bu = torch.cat((uf, uf[:, -1:] * self._taper), dim=1)
        bv = torch.cat((vf, vf[:, -1:] * self._taper), dim=1)
        uh, vh = torch.fft.fft2(bu), torch.fft.fft2(bv)
        div = self._kx * uh + self._ky * vh
        uh = uh - self._kx * div / self._k2
        vh = vh - self._ky * div / self._k2
        bu = torch.fft.ifft2(uh * self._phase).real
        bv = torch.fft.ifft2(vh * self._phase).real
        return bu[:, :self.nx], bv[:, :self.nx]

    # -- one macro-step -----------------------------------------------------

    def macro_step(self, u, v, theta):
        fx, fy, u_n, thrust, power = self.disks.forcing(u, v, theta)
        ufs, vfs = self.cut(u - self.free_u), self.cut(v - self.free_v)
        u1, v1 = self.expert.step_batch(ufs, vfs, wa.MACRO_DT)
        # The checkpoint's own mean drift, removed -- but the INCOMING window
        # mean is KEPT.  A periodic box with no force conserves it, and the
        # deficit a disk puts there has to survive long enough to be carried out
        # of the domain instead of being deleted every macro-step.
        u1 = u1 + (ufs.mean(dim=(1, 2)) - u1.mean(dim=(1, 2)))[:, None, None]
        v1 = v1 + (vfs.mean(dim=(1, 2)) - v1.mean(dim=(1, 2)))[:, None, None]
        au, av = self.blend(u1, v1)
        au = au + fx * wa.MACRO_DT
        av = av + fy * wa.MACRO_DT
        au, av = self.transport_project(au, av)
        u2, v2 = self.band(self.free_u + au, self.free_v + av)
        return u2, v2, power, u_n


#: The fluid experts this PoC can march, by the `kind=` name `wake_array` uses.
EXPERT_KINDS = ("reference_exposed", "poseidon")


def rollout_for(case: FarmCase, kind: str = "reference_exposed",
                device: str = "cpu", checkpoint: bool = True,
                threads: int = 8) -> Rollout:
    """Build the rollout for one fluid expert.  **This is the whole swap.**

    `atlas-proof-of-concept-1` section 9 says of the frozen-checkpoint run that
    "the swap is a ``kind=`` argument"; this function is that sentence, and the
    fact that it is four lines is the claim `expert-library-atlas-0.1` makes
    about a capability record being all the composition layer needs to know.
    """
    if kind in ("reference_exposed", "reference", "classical"):
        return Rollout(case, device=device, checkpoint=checkpoint)
    if kind == "poseidon":
        return PoseidonRollout(case, device=device, checkpoint=checkpoint,
                               threads=threads)
    raise ValueError(f"unknown fluid expert kind {kind!r}; "
                     f"this PoC marches {EXPERT_KINDS}")


@dataclass
class RolloutResult:
    power: Any                       # torch scalar: the time-averaged array power
    power_trace: np.ndarray          # [steps, K]
    inflow_trace: np.ndarray         # [steps, K]
    fields: dict[int, tuple[np.ndarray, np.ndarray]]
    u: Any
    v: Any
    wall_s: float
    steps: int

    @property
    def farm_power(self) -> float:
        return float(self.power.detach() if hasattr(self.power, "detach")
                     else self.power)

    def settling(self, n: int = 5) -> float:
        """Relative change in farm power over the last `n` macro-steps.

        The number the quasi-steady caveat is quoted with: it says how far from
        a converged rollout the state the objective is read at actually is.
        """
        tot = self.power_trace.sum(axis=1)
        if len(tot) < 2 * n:
            return float("nan")
        a, b = tot[-2 * n:-n].mean(), tot[-n:].mean()
        return float(abs(b - a) / max(abs(b), 1e-12))


# ---------------------------------------------------------------------------
# the objective
# ---------------------------------------------------------------------------


def spacing_penalty(theta, case: FarmCase):
    """`lam * sum_{j<k} relu(s_min - d_jk)^2`, differentiable everywhere it is
    nonzero.  Distances are between rotor CENTRES, in rotor diameters."""
    t = theta.reshape(-1, 3)
    p = t[:, :2]
    d = torch.cdist(p[None], p[None])[0]
    k = p.shape[0]
    eye = torch.eye(k, dtype=torch.bool, device=p.device)
    viol = torch.clamp(case.s_min - d, min=0.0).masked_fill(eye, 0.0)
    return case.lam_spacing * 0.5 * (viol ** 2).sum()


def objective(theta, rollout: Rollout, grad: bool = False, **kw):
    """`J = <sum_k P_k>_last-window  -  spacing penalty`.  Maximised.

    Returns ``(J, RolloutResult)`` with `J` a torch scalar carrying the tape when
    ``grad`` is on.
    """
    res = rollout.run(theta, grad=grad, **kw)
    j = res.power - spacing_penalty(theta, rollout.case)
    return j, res


def value_and_grad(theta_np: np.ndarray, rollout: Rollout, **kw):
    """`(J, dJ/dtheta, RolloutResult)` -- one adjoint through the whole rollout."""
    th = torch.as_tensor(np.asarray(theta_np, dtype=float), dtype=TORCH_DTYPE,
                         device=rollout.device).requires_grad_(True)
    j, res = objective(th, rollout, grad=True, **kw)
    g, = torch.autograd.grad(j, th)
    return float(j.detach()), g.detach().cpu().numpy().copy(), res


def value_and_grad_local(theta_np: np.ndarray, rollout: Rollout, **kw):
    """`dJ/dtheta` with the ROLLOUT frozen -- the ablation the claim needs.

    theta reaches J by two routes:

      **(a) the read.**  At the states the objective is averaged over, each disk
      samples its own inflow at a place that depends on its position and yaw,
      and turns it into a power.  Differentiating that needs no adjoint of
      anything -- it is one algebraic closure evaluated on a fixed field, and a
      surrogate wake model gives it to you for free.

      **(b) the rollout.**  Those states are what they are BECAUSE of where the
      disks were for the preceding forty-odd macro-steps.  Getting this route
      means differentiating back through every window's local solve, the blend,
      the projection and every fluid-disk seam.  It is the route nothing else
      has, and it is what this PoC is for.

    This function returns (a) alone: the march is run under `no_grad`, the
    states the objective reads are kept, and the closure is re-evaluated on them
    with theta live.  `stage_ablation` reports how much of the full gradient
    route (b) accounts for, which is the honest way to say whether the seam
    being inside the differentiated path is doing work or is decorative.
    """
    case = rollout.case
    steps = kw.pop("steps", None) or case.steps
    avg = min(case.avg_window, steps)
    keep = list(range(steps - avg, steps))
    with torch.no_grad():
        th_c = torch.as_tensor(np.asarray(theta_np, dtype=float),
                               dtype=TORCH_DTYPE, device=rollout.device)
        res = rollout.run(th_c, steps=steps, grad=False, keep_fields=keep, **kw)
    th = torch.as_tensor(np.asarray(theta_np, dtype=float), dtype=TORCH_DTYPE,
                         device=rollout.device).requires_grad_(True)
    total = None
    for sidx in keep:
        u = torch.as_tensor(res.fields[sidx][0], dtype=TORCH_DTYPE,
                            device=rollout.device)
        v = torch.as_tensor(res.fields[sidx][1], dtype=TORCH_DTYPE,
                            device=rollout.device)
        p = rollout.disks.forcing(u, v, th)[4].sum()
        total = p if total is None else total + p
    j = total / avg - spacing_penalty(th, case)
    g, = torch.autograd.grad(j, th)
    return float(j.detach()), g.detach().cpu().numpy().copy(), res


def value_only(theta_np: np.ndarray, rollout: Rollout, **kw):
    """`(J, RolloutResult)` with no tape at all -- the baselines' evaluation."""
    with torch.no_grad():
        th = torch.as_tensor(np.asarray(theta_np, dtype=float),
                             dtype=TORCH_DTYPE, device=rollout.device)
        j, res = objective(th, rollout, grad=False, **kw)
    return float(j), res


def power_of(theta_np: np.ndarray, rollout: Rollout, **kw) -> float:
    return value_only(theta_np, rollout, **kw)[1].farm_power


# ---------------------------------------------------------------------------
# constraints: the box, and the spacing set
# ---------------------------------------------------------------------------


def project_design(theta: np.ndarray, case: FarmCase, iters: int = 60) -> np.ndarray:
    """Project onto the box AND the minimum-spacing set, alternating.

    The spacing set is not convex, so this is Dykstra-flavoured rather than a
    projection with a guarantee: each pass pushes every violating pair apart
    along its own line by half the deficit, then re-clips to the box.  It
    terminates when no pair violates, and the driver ASSERTS that it did.
    """
    lo, hi = case.box
    t = np.array(theta, dtype=float).reshape(-1, 3)
    t = np.clip(t.reshape(-1), lo, hi).reshape(-1, 3)
    for _ in range(iters):
        p = t[:, :2]
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        if d.min() >= case.s_min - 1e-9:
            break
        bad = np.argwhere(d < case.s_min - 1e-9)
        shift = np.zeros_like(p)
        for a, b in bad:
            if a >= b:
                continue
            delta = p[a] - p[b]
            r = float(np.linalg.norm(delta))
            if r < 1e-9:
                delta = np.array([1e-3, 0.0])
                r = 1e-3
            push = 0.5 * (case.s_min - r) * delta / r
            shift[a] += push
            shift[b] -= push
        t[:, :2] = p + shift
        t = np.clip(t.reshape(-1), lo, hi).reshape(-1, 3)
    return t.reshape(-1)


def min_spacing(theta: np.ndarray) -> float:
    p = np.asarray(theta, dtype=float).reshape(-1, 3)[:, :2]
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


# ---------------------------------------------------------------------------
# the optimiser, and the two baselines
# ---------------------------------------------------------------------------


@dataclass
class OptTrace:
    """What every optimiser here returns, so the three are comparable.

    ``evals`` is the count that carries the headline: the number of composed
    ROLLOUTS, which is the unit all three methods pay in.  ``fwd_equiv`` is the
    same count in forward-equivalents -- a gradient step costs one forward plus
    one backward, and the backward is measured rather than assumed.
    """

    method: str
    j: list[float] = field(default_factory=list)
    best_j: list[float] = field(default_factory=list)
    evals: list[int] = field(default_factory=list)
    wall_s: list[float] = field(default_factory=list)
    theta: list[list[float]] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def best(self) -> tuple[float, np.ndarray]:
        i = int(np.argmax(self.j))
        return self.j[i], np.array(self.theta[i])

    def as_dict(self) -> dict:
        # `theta` is the WHOLE trajectory, not just its endpoints.  It was left
        # out of the first version of this method and the omission cost a
        # headline panel: the design vector is written to disk after every step,
        # but each write overwrote the file with a record that kept only the
        # last one, so the history existed at no point in time.  A trajectory
        # that is not serialised did not happen.
        return {"method": self.method, "j": self.j, "best_j": self.best_j,
                "evals": self.evals, "wall_s": self.wall_s,
                "grad_norm": self.grad_norm, "theta": self.theta,
                "theta_final": self.theta[-1],
                "theta_best": self.best()[1].tolist(), **self.extra}


def adam_optimise(theta0: np.ndarray, rollout: Rollout, steps: int = 40,
                  lr_pos: float = 0.12, lr_yaw: float = 0.035,
                  betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                  on_step: Callable[[int, OptTrace], None] | None = None,
                  fwd_equiv_per_grad: float = 1.0) -> OptTrace:
    """Adam on the design vector, projected after every step.

    Two learning rates because the design vector has two units: `lr_pos` is in
    rotor diameters per step and `lr_yaw` in radians per step (0.035 rad = 2.0
    degrees).  Adam normalizes the gradient's magnitude away, so these ARE the
    step sizes and the two would otherwise be set by the accident of how much
    bigger dJ/dx is than dJ/dgamma.
    """
    case = rollout.case
    th = project_design(np.asarray(theta0, dtype=float), case)
    m = np.zeros_like(th)
    v = np.zeros_like(th)
    lr = np.empty_like(th)
    lr[0::3] = lr_pos
    lr[1::3] = lr_pos
    lr[2::3] = lr_yaw
    tr = OptTrace("adam-autograd")
    tr.extra["lr_pos"] = lr_pos
    tr.extra["lr_yaw"] = lr_yaw
    tr.extra["fwd_equiv_per_grad"] = fwd_equiv_per_grad
    t0 = time.perf_counter()
    n_eval = 0
    for it in range(steps):
        j, g, res = value_and_grad(th, rollout)
        n_eval += 1
        tr.j.append(j)
        tr.best_j.append(max(tr.best_j[-1], j) if tr.best_j else j)
        tr.evals.append(n_eval)
        tr.wall_s.append(time.perf_counter() - t0)
        tr.theta.append([float(x) for x in th])
        tr.grad_norm.append(float(np.linalg.norm(g)))
        tr.extra.setdefault("settling", []).append(res.settling())
        tr.extra.setdefault("power_trace", []).append(
            res.power_trace.sum(axis=1).tolist())
        if on_step is not None:
            on_step(it, tr)
        m = betas[0] * m + (1 - betas[0]) * g
        v = betas[1] * v + (1 - betas[1]) * g * g
        mh = m / (1 - betas[0] ** (it + 1))
        vh = v / (1 - betas[1] ** (it + 1))
        th = th + lr * mh / (np.sqrt(vh) + eps)          # ASCENT: J is maximised
        th = project_design(th, case)
    tr.extra["fwd_equiv"] = [e * fwd_equiv_per_grad for e in tr.evals]
    return tr


def fd_gradient(theta: np.ndarray, rollout: Rollout, h: float = 1e-3,
                idx: Sequence[int] | None = None) -> tuple[np.ndarray, int]:
    """Central coordinate finite differences.  `(grad, n_evals)`.

    2n evaluations for the whole vector, and that count is the honest one: this
    is what a derivative-free method has to pay to get the same object the
    adjoint returns for one backward pass.
    """
    th = np.asarray(theta, dtype=float)
    ids = range(len(th)) if idx is None else list(idx)
    g = np.zeros_like(th)
    n = 0
    for i in ids:
        tp, tm = th.copy(), th.copy()
        tp[i] += h
        tm[i] -= h
        jp, _ = value_only(tp, rollout)
        jm, _ = value_only(tm, rollout)
        n += 2
        g[i] = (jp - jm) / (2.0 * h)
    return g, n


def fd_check(theta: np.ndarray, rollout: Rollout, n_probe: int = 6,
             h: float = 1e-3, seed: int = 0) -> dict:
    """OP-6's check: a few components of the adjoint against central differences.

    Reported as a per-component relative error and a cosine, on components
    chosen to cover all three variable kinds rather than at random, because a
    random draw over a 36-vector is 2/3 position by construction.
    """
    j0, g, _ = value_and_grad(theta, rollout)
    k = len(theta) // 3
    rng = np.random.default_rng(seed)
    picks = []
    per = max(1, n_probe // 3)
    for kind in (0, 1, 2):
        for t in rng.choice(k, size=min(per, k), replace=False):
            picks.append(3 * int(t) + kind)
    picks = sorted(set(picks))[:n_probe] if n_probe else []
    gfd, n_ev = fd_gradient(theta, rollout, h=h, idx=picks)
    rows = []
    for i in picks:
        denom = max(abs(g[i]), abs(gfd[i]), 1e-30)
        rows.append({"i": int(i), "kind": ["x", "y", "yaw"][i % 3],
                     "turbine": int(i // 3), "adjoint": float(g[i]),
                     "fd": float(gfd[i]),
                     "rel_err": float(abs(g[i] - gfd[i]) / denom)})
    a = np.array([r["adjoint"] for r in rows])
    b = np.array([r["fd"] for r in rows])
    cos = (float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
           if a.size and np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0 else None)
    return {"J": j0, "h": h, "n_fd_evals": n_ev, "components": rows,
            "cosine": cos,
            "max_rel_err": float(max((r["rel_err"] for r in rows), default=0.0)),
            "median_rel_err": float(np.median([r["rel_err"] for r in rows]))
            if rows else None,
            "grad_norm": float(np.linalg.norm(g))}


def cma_es(theta0: np.ndarray, rollout: "Rollout | None", budget: int,
           sigma0: float = 0.6, popsize: int | None = None, seed: int = 0,
           on_eval: Callable[[int, float, float], None] | None = None,
           wall_budget_s: float | None = None,
           case: FarmCase | None = None,
           evaluate: Callable[[list], list] | None = None) -> OptTrace:
    """Standard CMA-ES (Hansen), written out because `cma` is not installed here.

    It is the textbook (mu/mu_w, lambda)-CMA-ES with rank-mu and rank-one
    updates and the usual step-size control; no restarts, no active update, no
    surrogate.  The design vector is scaled so one unit of the search variable
    is one rotor diameter in position and `yaw_scale` radians in yaw, which is
    the same units the projected Adam uses -- otherwise the comparison would be
    partly a comparison of coordinate scalings.

    **Both methods get the same constraint handling.**  A candidate is put
    through `project_design` -- the same box clip and the same spacing
    projection the gradient method's step is put through -- the objective is
    evaluated at the PROJECTED point, and the distance moved by the projection
    is charged as a penalty so the search does not sit on the wall for free.
    Giving the gradient method a projection and the baseline only a penalty
    would have made part of the headline ratio a comparison of constraint
    handling, which is not what is being measured.
    """
    case = case or rollout.case
    if evaluate is None:
        if rollout is None:
            raise ValueError("cma_es needs either a rollout or an `evaluate`")
        def evaluate(batch, _r=rollout):
            return [value_only(t, _r)[0] for t in batch]
    n = len(theta0)
    yaw_scale = 0.20                       # rad per unit; ~11.5 deg
    scale = np.ones(n)
    scale[2::3] = yaw_scale

    def unscale(z):
        return theta0 + scale * z

    lam = popsize or (4 + int(3 * math.log(n)))
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w = w / w.sum()
    mueff = 1.0 / np.sum(w ** 2)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chin = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

    rng = np.random.default_rng(seed)
    xmean = np.zeros(n)
    sigma = sigma0
    pc = np.zeros(n)
    ps = np.zeros(n)
    B = np.eye(n)
    D = np.ones(n)
    C = np.eye(n)
    invsqrtC = np.eye(n)
    eigeneval = 0

    tr = OptTrace("cma-es")
    tr.extra.update({"popsize": lam, "sigma0": sigma0, "seed": seed,
                     "n_design": n, "yaw_scale": yaw_scale})
    t0 = time.perf_counter()
    n_eval = 0
    best_j, best_th = -np.inf, project_design(theta0, case)
    gen = 0
    while n_eval < budget:
        if wall_budget_s is not None and time.perf_counter() - t0 > wall_budget_s:
            tr.extra["stopped"] = "wall_budget"
            break
        n_draw = min(lam, budget - n_eval)
        if n_draw <= 0:
            break
        zs, ths, pens = [], [], []
        for _ in range(n_draw):
            z = xmean + sigma * (B @ (D * rng.standard_normal(n)))
            raw = unscale(z)
            feas = project_design(raw, case)
            zs.append(z)
            ths.append(feas)
            pens.append(float(np.sum((raw - feas) ** 2)))
        # ONE batch, so the whole generation can go to a pool of workers.  The
        # population of a (mu/mu_w, lambda)-CMA-ES is independent by
        # construction, which is the derivative-free method's own answer to the
        # cost this experiment is measuring -- so it is given it, and the
        # forward-evaluation count the headline quotes is unaffected either way.
        raw_js = evaluate(ths)
        js = [float(rj) - case.lam_spacing * pe for rj, pe in zip(raw_js, pens)]
        for th, j in zip(ths, js):
            n_eval += 1
            if j > best_j:
                best_j, best_th = j, th.copy()
            tr.j.append(j)
            tr.best_j.append(best_j)
            tr.evals.append(n_eval)
            tr.wall_s.append(time.perf_counter() - t0)
            tr.theta.append([float(x) for x in th])
            if on_eval is not None:
                on_eval(n_eval, j, best_j)
        if len(zs) < lam:
            tr.extra["stopped"] = "budget"
            break
        order = np.argsort(-np.array(js))            # maximisation
        zs = np.array(zs)[order]
        xold = xmean
        xmean = w @ zs[:mu]
        ps = ((1 - cs) * ps
              + math.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ (xmean - xold)) / sigma)
        gen += 1
        hsig = (np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * gen)) / chin
                < 1.4 + 2 / (n + 1))
        pc = ((1 - cc) * pc
              + hsig * math.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma)
        artmp = (zs[:mu] - xold) / sigma
        C = ((1 - c1 - cmu) * C
             + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
             + cmu * (artmp.T * w) @ artmp)
        sigma = sigma * math.exp((cs / damps) * (np.linalg.norm(ps) / chin - 1))
        if n_eval - eigeneval > lam / (c1 + cmu) / n / 10:
            eigeneval = n_eval
            C = np.triu(C) + np.triu(C, 1).T
            dd, B = np.linalg.eigh(C)
            dd = np.maximum(dd, 1e-20)
            D = np.sqrt(dd)
            invsqrtC = B @ np.diag(1 / D) @ B.T
    tr.extra["generations"] = gen
    tr.extra["n_eval"] = n_eval
    # NOT "best_j": `OptTrace.as_dict` splats `extra` last, so a scalar under
    # that name silently replaces the per-evaluation running-best LIST and the
    # headline loses the trace it is computed from.
    tr.extra["best_j_final"] = best_j
    tr.extra["theta_best"] = [float(x) for x in best_th]
    if not tr.theta:
        tr.theta.append([float(x) for x in best_th])
        tr.j.append(best_j)
        tr.best_j.append(best_j)
        tr.evals.append(0)
        tr.wall_s.append(0.0)
    return tr


def evals_to_tolerance(tr: OptTrace, target: float) -> int | None:
    """First evaluation index at which the running best reaches `target`."""
    for e, b in zip(tr.evals, tr.best_j):
        if b >= target:
            return e
    return None
