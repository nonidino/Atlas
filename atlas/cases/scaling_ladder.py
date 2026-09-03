"""The SEVENTH real case study: one geometry at five sizes, to measure how
composition error grows with interface count.

`wake_array.py` is the parent and this is the same physics: 128-cell Poseidon-T
windows of 4 D each at a 16-cell overlap, actuator disks at 3.5 D spacing in an
L, a `reference.WindowNS` column beside the checkpoint's.  What is new is that
**the size is the variable and everything else is nailed down**, because the
quantity this file exists to measure is a *derivative* --

    d(composed defect) / d(interface count)

-- and a derivative taken while two things move measures neither.  Five rungs:

    N     tiling   domain (cells)   rotors   seams   overlapping pairs
    1     1 x 1      128 x 128         0       0        0   <- control: zero defect
    2     2 x 1      240 x 128         1       3        1
    6     3 x 2      352 x 240         3      13       11   <- control: the parent
    12    4 x 3      464 x 352         5      27       29
    24    6 x 4      688 x 464        12      62       68

Why this is a criterion and not a curiosity
--------------------------------------------

`f1-pathmap-and-end-goal` §5's **F1** is the falsification criterion that fails
if composition error grows *super-linearly* in interface count, and §3.2 names
rung 9 -- fifteen to twenty coupled agents -- as *the rung that decides
everything*.  The sweep §5 schedules as rung 9's early warning had never been
run at any N, and the pathmap's own words for a bad result are *"a reason to
stop and rethink, not to press on"*.  `case-study-ladder-to-f1` §4 then schedules
nine further case studies on the answer.

So the deliverable is an exponent with an error bar, in both columns, over the
full 24x range -- not an adjective.

Why it is a new file and not a flag on `wake_array`
----------------------------------------------------

`ArrayTiling` is now parameterized by `n_col`, `n_row` and `rotors`, so the
*tiling* did survive parameterization (that generalization is in `wake_array.py`,
and `ArrayTiling()` still reproduces the 3x2 array to the bit).  Three things did
not:

1. **The rotor layout.**  Three turbines in a fixed L is a layout, not a rule.
   `rotor_motif` below tiles the parent's L with period (3 columns, 2 rows) and
   truncates, which reproduces `ROTORS` exactly at 3x2 and keeps every row's
   first column clean-inflow at every size.
2. **The per-N controls.**  N=1 must return exactly zero composed defect and
   N=6 must return the parent's own array loss; neither is expressible as a
   parameter and both are the point.
3. **The referent.**  For `WindowNS` it is the MONOLITH on the whole domain,
   which exists at any size and is what `E` in `D_i = E_i R_i - R_i E` means.
   For Poseidon-T there is no monolith at any resolution ever (**W95**), so the
   checkpoint's column is measured reference-free and the classical column is
   what calibrates that surrogate.  `RectangularNS` below is the monolith and it
   is the only new expert in this file.

What is held fixed, and why each one
-------------------------------------

    dx          S_LEN / 128 = 1/32 D    the checkpoint's resolution is not a dial
    dt_macro    0.2                     exactly one native expert lead
    overlap     16 cells                changing it changes Pi, a measured output
    ramp        8 cells                 W53 keeps ramp 8, so priors stay comparable
    nu_ref      3.92e-3                 W0 4.2's grid-scale fit, the referent's own
    disk        ActuatorDisk, a = 1/3   zero fitted parameters, unchanged

The domain therefore GROWS with N rather than being cut finer.  That is the F1
question as the pathmap poses it -- rung 9 is a bigger problem with more agents,
not a finer cut of one problem -- and it is why the driver reports the total,
the per-cell RMS and the sup norm separately: a total that grows like
sqrt(interfaces) with a flat RMS is errors adding in quadrature, which is a real
and benign result, and it must not be reported as if it were a decay.

What this file cannot say
--------------------------

**It measures one expert family at one Reynolds number on one geometry.**  Claim
B might be expert-dependent (`case-study-ladder-to-f1` §2.4), which is exactly
why both columns run at every N; and a trend over five points with the largest
at 24 agents does not establish anything at 200.  The honest reading of a
sub-linear result here is *"no super-linear growth is visible up to 24
interfaces on this problem"*, which is what F1 asks and is less than Claim B.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from ..capability import EllipticSubsolve
from ..graph import CaseGraph, MeasuredConstants
from . import wake_array as wa
from .wake_array import (          # re-exported: this file adds no second copy
    DX,
    HALO,
    MACRO_DT,
    N,
    NU_REF,
    RAMP,
    ROTOR_CELLS,
    ROTOR_D,
    S_LEN,
    STRIDE,
    U_INF,
    ArrayTiling,
    Rotor,
    fourier_basis,
    modes_for,
    transport_and_project,
)

__all__ = [
    "RUNGS", "LadderRung", "rotor_motif", "tiling_for", "rung", "interface_count",
    "RectangularNS", "reference_monolith", "build", "make_experts",
]


# ---------------------------------------------------------------------------
# the ladder
# ---------------------------------------------------------------------------

#: The five rungs, as ``(n_col, n_row)``.  Chosen so the window count spans 24x
#: while the aspect ratio stays between 1.0 and 1.9 -- a ladder that also
#: stretched the domain would be measuring the aspect ratio.
RUNGS: tuple[tuple[int, int], ...] = ((1, 1), (2, 1), (3, 2), (4, 3), (6, 4))

#: The rotor motif, in x-adjacency coordinates ``(i mod 3, j mod 2)``.  It is the
#: parent's L -- two turbines in line and one abreast, at 3.5 D both ways -- and
#: tiling it with period (3, 2) is what keeps the layout a RULE rather than a
#: list.  ``(0, 1)`` is absent on purpose: it leaves the first x-adjacency of
#: every odd row rotor-free, so **every row carries at least one clean-inflow
#: control**, which is the property that made R3 the parent's control.
ROTOR_MOTIF: frozenset[tuple[int, int]] = frozenset({(0, 0), (1, 0), (1, 1)})


def rotor_motif(n_col: int, n_row: int) -> tuple[Rotor, ...]:
    """The rotors for an ``n_col x n_row`` tiling, from `ROTOR_MOTIF`.

    A rotor lives on an x-adjacency ``(i, j)`` -- the plane between window
    columns ``i`` and ``i+1`` of row ``j`` -- and its ``y_centre`` is the row's
    own stride plus the parent's 1.75 D, so the disk's LOCAL face cells are the
    same 32 in every window and `Rotor.cells` needs no special case.

    At ``(3, 2)`` this returns `wake_array.ROTORS` exactly, which is asserted in
    `tests/test_tier19_scaling_ladder.py` and is the reason the N=6 rung can be a
    reproduction control at all.  At ``(4, 3)`` truncation gives **5** rotors
    rather than the 6 the plan sketched; the rule is kept and the count is
    reported, because a motif that is bent to hit a predicted number is not a
    rule.
    """
    out = []
    k = 1
    for j in range(n_row):
        for i in range(n_col - 1):
            if (i % 3, j % 2) not in ROTOR_MOTIF:
                continue
            out.append(Rotor(f"R{k}", col=i, row=j,
                             y_centre=j * STRIDE * DX + 1.75))
            k += 1
    return tuple(out)


def tiling_for(n_col: int, n_row: int, ramp: int = RAMP) -> ArrayTiling:
    """The tiling at one rung, with the parent's ramp and the motif's rotors."""
    return ArrayTiling(ramp=ramp, n_col=n_col, n_row=n_row,
                       rotors=rotor_motif(n_col, n_row))


@dataclass(frozen=True)
class LadderRung:
    """One rung: the tiling, and the counting that goes with it.

    ``overlap_pairs`` is the number of window pairs that share at least one cell
    both own with positive weight -- side neighbours AND diagonal ones, since a
    16-cell overlap makes the corner square common to four windows.  It is the
    denominator the composition defect is read against, because a composed defect
    is created at overlaps and not at seams: two windows meeting diagonally have
    no `Connection` and their local solves still disagree in the corner.

    ``n_seams`` is the graph's own connection count, which is what the compile
    sees, and the two are reported side by side rather than one being called
    *the* interface count.
    """

    n_col: int
    n_row: int
    tiling: ArrayTiling

    @property
    def n_windows(self) -> int:
        return self.n_col * self.n_row

    @property
    def label(self) -> str:
        return f"N{self.n_windows}"

    @property
    def n_rotors(self) -> int:
        return len(self.tiling.rotors)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.tiling.ny, self.tiling.nx)

    @property
    def n_cells(self) -> int:
        return self.tiling.nx * self.tiling.ny

    @property
    def n_seams(self) -> int:
        return len(wa.connections(self.tiling))

    @lru_cache(maxsize=None)
    def overlap_pairs(self) -> list[tuple[str, str]]:
        """Window pairs whose positive-weight supports intersect."""
        t = self.tiling
        sup = {}
        for k, name in enumerate(t.names):
            ox, oy = t.offsets[k]
            w = t.weights()[k]
            sup[name] = w[oy:oy + N, ox:ox + N] > 0.0
        boxes = dict(zip(t.names, t.offsets))
        out = []
        names = t.names
        for a_i in range(len(names)):
            for b_i in range(a_i + 1, len(names)):
                a, b = names[a_i], names[b_i]
                (ax, ay), (bx, by) = boxes[a], boxes[b]
                x0, x1 = max(ax, bx), min(ax + N, bx + N)
                y0, y1 = max(ay, by), min(ay + N, by + N)
                if x1 <= x0 or y1 <= y0:
                    continue
                ma = sup[a][y0 - ay:y1 - ay, x0 - ax:x1 - ax]
                mb = sup[b][y0 - by:y1 - by, x0 - bx:x1 - bx]
                if np.any(ma & mb):
                    out.append((a, b))
        return out

    @property
    def n_overlaps(self) -> int:
        return len(self.overlap_pairs())

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label, "n_windows": self.n_windows,
            "n_col": self.n_col, "n_row": self.n_row,
            "nx": self.tiling.nx, "ny": self.tiling.ny,
            "n_cells": self.n_cells,
            "domain_D": [self.tiling.nx * DX, self.tiling.ny * DX],
            "aspect": self.tiling.nx / self.tiling.ny,
            "n_rotors": self.n_rotors,
            "rotors": {r.rotor_id: [r.x_plane, r.y_centre]
                       for r in self.tiling.rotors},
            "n_seams": self.n_seams,
            "n_overlaps": self.n_overlaps,
        }


@lru_cache(maxsize=None)
def rung(n_col: int, n_row: int, ramp: int = RAMP) -> LadderRung:
    return LadderRung(n_col, n_row, tiling_for(n_col, n_row, ramp))


def ladder(ramp: int = RAMP) -> list[LadderRung]:
    return [rung(c, r, ramp) for c, r in RUNGS]


def interface_count(r: LadderRung, kind: str = "overlaps") -> int:
    return r.n_overlaps if kind == "overlaps" else r.n_seams


# ---------------------------------------------------------------------------
# the referent -- and it is the whole reason the classical column is here
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _rect_class():
    """`reference.WindowNS` on a RECTANGULAR domain.

    The expert is fixed at a square ``[B, n, n]`` in exactly one place: its
    ``__post_init__`` builds the Poisson symbol as ``ev[None, None, :] +
    ev[None, :, None]`` from a single 1-D eigenvalue array, and every other
    operator in the class is elementwise with one cell size.  So a rectangular
    monolith is that one line with two arrays instead of one, and nothing else.

    Subclassing rather than editing: the agent is not ours to change, and this is
    the same latitude `window_ns._no_projection_class` already takes.  Asserted
    bit-identical to `WindowNS` when ``ny == n``, which is the only claim this
    class makes -- see `tests/test_tier19_scaling_ladder.py`.

    **It is the monolith, so it is what makes `tau`, the composed defect and the
    L2/C2 bound MEAN anything in the classical column.**  ``D_i = E_i R_i - R_i E``
    has an ``E`` in it, and for a checkpoint frozen at 128x128 there is no such
    thing at any resolution (W95).  That asymmetry is not a courtesy to the
    classical solver; it is the reason `case-study-ladder-to-f1` §2 reorders the
    whole program around classical experts first.
    """
    ref = wa.load_reference()
    pressure = importlib.import_module("atlas_windfarm_reference.pressure")

    @dataclass
    class RectangularNS(ref.WindowNS):
        #: Cells in y.  ``n`` stays the x count, so ``h = length / n`` is the one
        #: cell size and the grid is square-celled on a rectangular domain.
        ny: int = 0

        def __post_init__(self):
            super().__post_init__()
            ny = int(self.ny or self.n)
            ev_x = pressure.wide_eigenvalues_neumann(self.n, self.h)
            ev_y = pressure.wide_eigenvalues_neumann(ny, self.h)
            lam = ev_y[None, :, None] + ev_x[None, None, :]
            self._lam = self.b.asarray(lam)
            self._lam_safe = self.b.asarray(np.where(lam == 0.0, 1.0, lam))
            self.ny_cells = ny

    return RectangularNS


#: Public alias, so a caller does not have to know the class is built lazily
#: behind the private package registration.
def RectangularNS(*args, **kwargs):                          # noqa: N802
    """`reference.WindowNS` on a rectangular domain -- see `_rect_class`."""
    return _rect_class()(*args, **kwargs)


@lru_cache(maxsize=8)
def reference_monolith(nx: int, ny: int, nu: float = NU_REF):
    """The undivided domain, same discretization, same cell, same transmission.

    ``transmission='dirichlet'`` matches `wake_array.reference_solver`, so the
    only difference between this and the composed run is the cut -- which is the
    entire point, and is what makes the N=1 rung return zero to the bit rather
    than to a tolerance.
    """
    return RectangularNS(nu=nu, length=nx * DX, n=nx, ny=ny, cfl=0.4,
                         transmission="dirichlet")


# ---------------------------------------------------------------------------
# the graph, at any rung
# ---------------------------------------------------------------------------


def make_experts(u_full: np.ndarray, v_full: np.ndarray, r: LadderRung,
                 kind: str = "poseidon", dt: float = MACRO_DT,
                 nu: float = NU_REF, nu_solver: float | None = None) -> dict[str, Any]:
    """The parent's own factory, pointed at this rung's tiling."""
    return wa.make_experts(u_full, v_full, kind=kind, dt=dt, nu=nu,
                           tiling=r.tiling, nu_solver=nu_solver)


def build(u_full: np.ndarray, v_full: np.ndarray, r: LadderRung,
          kind: str = "poseidon", dt: float = MACRO_DT, nu: float = NU_REF,
          elliptic: EllipticSubsolve = EllipticSubsolve.UNKNOWN,
          measured: MeasuredConstants | None = None,
          experts: dict[str, Any] | None = None,
          assembly_projection: bool = True) -> tuple[CaseGraph, dict[str, Any]]:
    """The rung's graph: ``n_windows`` fluid agents and ``n_rotors`` disks.

    Delegates to `wake_array.build`, which is the point of generalizing
    `ArrayTiling`: one declaration, five sizes, and no branch anywhere on N.

    ``assembly_projection`` is the ladder's own variable since 2026-08-31
    (**W100**), and it is the only one besides the tiling: True declares the
    projected assembly of L6/C2 and False declares the bare blend the classical
    column shipped with.  The ladder is what showed the difference, because the
    difference does not appear at N = 1 -- with one window ``chi`` is identically
    one, ``grad(chi)`` is zero, and the two assemblies are the same operator.

    **N = 1 has no artificial faces, so it has no ports, and the record refuses
    before the compiler is reached**: `capability.ExpertCapabilities` raises
    `RecordError("expert 'F00' declares no ports")` at construction.  That is the
    correct answer and it is worth reporting rather than skipping -- a single
    window is not a composition, and the refusal arrives one layer EARLIER than
    `L3`'s "no connections declared: this is not a composition", which is the
    message a reader would expect.  The driver catches it and records it, because
    *"the framework will not express a one-agent composition"* is the declaration-
    side half of the control whose measurement half is a zero composed defect.
    """
    graph, experts = wa.build(u_full, v_full, kind=kind, dt=dt, nu=nu,
                              tiling=r.tiling, elliptic=elliptic,
                              measured=measured, experts=experts,
                              assembly_projection=assembly_projection)
    graph.name = f"scaling-ladder-{r.label}-{r.n_col}x{r.n_row}-{kind}"
    graph.note = (
        ("projected assembly (L6/C2): " if assembly_projection
         else "bare partition-of-unity assembly, no constraint projection (L6/C2 "
              "fails, R12): ") +
        f"CS-7 rung {r.label}: {r.n_windows} windows of {N} cells on a "
        f"{r.tiling.nx}x{r.tiling.ny}-cell ({r.tiling.nx * DX:.1f} x "
        f"{r.tiling.ny * DX:.1f} D) domain, {r.n_rotors} rotors, {r.n_seams} "
        f"declared seams and {r.n_overlaps} overlapping window pairs. dx, "
        f"dt_macro, overlap, ramp, nu_ref and the disk model are held fixed "
        f"across the whole ladder; the size is the only variable"
    )
    return graph, experts


def geometry_report(ramp: int = RAMP) -> dict[str, Any]:
    """The ladder's own table, for the artifact's header."""
    return {
        "held_fixed": {
            "dx_D": DX, "macro_dt": MACRO_DT, "overlap_cells": HALO,
            "stride_cells": STRIDE, "ramp_cells": ramp, "nu_ref": NU_REF,
            "window_cells": N, "S_LEN_D": S_LEN, "U_inf": U_INF,
            "rotor_D": ROTOR_D, "rotor_cells": ROTOR_CELLS, "induction_a": 1 / 3,
        },
        "rungs": [r.as_dict() for r in ladder(ramp)],
        "modes": {n: modes_for(n) for n in (N, N - ROTOR_CELLS, ROTOR_CELLS)},
    }
