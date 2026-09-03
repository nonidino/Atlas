"""The TENTH real case study: where to cut, decided by an algorithm.

`poc1-retrospective-and-hybrid-roadmap` 4 proposes this as **CS-9★** and
gives the reason it is inserted rather than appended: three of the four
end-goal capabilities of `f1-pathmap-and-end-goal` 1 are on the existing
Phase B/C schedule and the fourth -- *automatic, adaptive seam placement* --
is on no rung anywhere.  Every case study in this vault so far has chosen its
windows the same way: a person typing `N_COL` and `N_ROW`.

**W112** is the row.  What it asks for is one thing and it is small: run a
search over candidate decompositions, score each by a criterion under a stated
cost budget, and compare the cut the search picks against a hand-chosen tiling
this vault has already measured.

The premise the row states, and the correction it needs
--------------------------------------------------------

W112, and the retrospective 4 that proposes this case study, both say that
`interface-transfer-theory` 9's score

    Q(Gamma) = (1 / beta) * ||S - diag S|| / ||S||

*"has never been used to choose a cut"* and has never had a falsification.
**That is not true, and the record that contradicts it is in this vault.**
`tier0-measurements` 10.2 -- 2026-08-28, four days after Q was proposed --
scanned twelve cut placements of `tests/strip_model.py` and found Q ranks
against the measured composed defect at **-0.853**, that following it costs
1.407x the best available cut, that it has a **361x orbit** under an
admissible re-declaration of the same interface space, and that it is constant
to 8e-11 where the truth spreads 2.29x.  `probe.cut_score`'s own docstring has
said RETIRED since that day, and `compiler._cut_policy` replaced it with
**L2/C2**: minimize the chi-weighted restriction defect.

So this file does **not** re-run a falsification that has already happened.
What Tier 10 did was move ONE cut of a FIXED two-strip decomposition of a
linear model problem.  What has still never happened, and is what W112's own
*"done when"* clause actually asks for, is all four of these at once:

  1. a search over **decompositions** -- how many windows, where, how much
     overlap -- and not a scan of one cut of a fixed one;
  2. under a **stated cost budget**, which is what makes a placement question
     well-posed at all (without one the answer is always "one window");
  3. on the **real geometry** of a case study this vault has measured, with a
     real expert and a real developed flow, rather than on a model problem;
  4. against the **hand-chosen tiling** that geometry shipped with, so the
     algorithm is graded on whether it beats five days of a person choosing
     window boundaries.

Tier 10 falsified Q as a **ranking function on one cut**.  This file asks the
different question of whether ANY of the available criteria can **drive a
search** to a decomposition a person did not find -- and it carries Q through
that search beside its competitors, because a criterion that ranks backwards
on a strip model and is merely useless on a real one is a different finding
from a criterion that ranks backwards on both.

What is searched over
----------------------

A **candidate decomposition** of a fixed rectangular domain is a set of cut
positions per axis plus a halo:

    x_cuts = (x_1 < ... < x_p),  y_cuts = (y_1 < ... < y_q),  halo h

Window (i, j) spans ``[x_{i-1} - h, x_i + h) x [y_{j-1} - h, y_j + h)``,
clipped to the domain, with ``x_0 = 0`` and ``x_{p+1} = nx``.  The overlap
between adjacent windows is ``2h``.

**Two properties of that parameterization are the reason it is the right one.**

* CS-7's hand-chosen tilings are candidates in it.  `ArrayTiling(n_col=3,
  n_row=2)` -- the N=6 rung, and the wake array's own tiling -- is exactly
  ``x_cuts=(120, 232), y_cuts=(120,), halo=8`` on a 352 x 240 domain, and
  `from_array_tiling` builds it.  `tests/test_tier25_seam_placement.py`
  asserts the boxes, the partition of unity and the composed defect all agree
  with `wake_array`'s to the bit.  A search whose space did not contain the
  baseline could not be graded against it.
* The cuts move **independently**, so the search can put a seam where a person
  would not: off-centre, unevenly spaced, or a different count on each axis.
  That is the degree of freedom `tier0-measurements` 10.2's strip model had
  exactly one of and every case study in this vault has had none of.

The cost budget
----------------

Three are supplied and they do not agree, which is the point of stating one:

    n_windows     the agent count -- what `case-study-ladder-to-f1`'s rungs
                  are numbered by, and what a graph's compile cost scales with
    work_cells    sum of window areas -- what the solves actually cost, and it
                  grows with the halo at a fixed window count
    halo_cells    work_cells - nx*ny -- the cells paid for twice, which is the
                  only cost a decomposition has that a monolith does not

`search` ranks within a **pool**, and a pool is built by a predicate on `Cost`.
A budget is mandatory rather than defaulted: with no budget the argmin of every
criterion here is the one-window decomposition, whose composed defect is
exactly zero because there is nothing to compose.

**And an UPPER-BOUND budget is not enough, which the first run of this file
found the hard way.**  Under ``n_windows <= 6`` every criterion here returned
the same two-window decomposition, at 0.085x the hand-chosen tiling's defect --
because fewer seams is less defect, always, and the search had been handed a
question about agent count rather than about placement.  That is a true
statement about decomposition cost and a worthless one about cut placement, and
it is reported as its own result (`at_most_windows` below) rather than quietly
repaired.

The question this case study is actually about is **where**, so the gate's
budget is an **equality**: ``n_windows == 6``, the hand-chosen tiling's own
count, with the halo held at the hand-chosen value.  Can an algorithm place six
windows better than a person placed six windows?  The count and the halo are
then swept on their own axes, one variable at a time, which is CS-7's own
discipline and the reason its exponent means anything.

What is scored
---------------

Five numbers per candidate, and the separation between them is the finding
Tier 10 already made and this file carries onto real geometry:

    Q_ai_inference          `probe.cut_score` -- 9's score, unchanged, the
                            framework's own function on the framework's own
                            assembled S
    off_diagonal_mass       ||S - diag S||, UN-normalized.  10.2 measured this
                            ranking +0.853 where Q ranks -0.853, so Q's two
                            normalizations are carried separately here rather
                            than being described
    Q_star_chi_weighted     L2/C2: || sum_i chi_i |D_i| ||, the DERIVED
                            criterion.  Needs the monolith, so it is a
                            diagnostic and cannot drive a search
    Q_hat_reference_free    neighbours' disagreement on the overlap, which is
                            D_i - D_j there exactly.  Costs nothing beyond the
                            local solves the composed step already does, so it
                            is the ONLY criterion here a real search can use
    composed_defect         the truth: || A({E_i R_i u}) - E u || over one
                            exchange interval, and again over K of them

**The truth is available here and would not be in production**, because both
the windows and the monolith are `reference.WindowNS` on a rectangle -- the
same expert at two sizes, which is what makes ``D_i = E_i R_i - R_i E``
meaningful at all.  A frozen checkpoint has no monolith at any resolution
(W95), so on that column a search would have only Q_hat.  That asymmetry is
why the classical column is the one this file searches.

The controls
-------------

Four, and each is a thing that would invalidate the run if it failed.

    ZERO       the one-window candidate has no interface: chi is identically
               one, cutting and assembling are the identity, and the composed
               defect must be exactly zero, bitwise.  `tier0-measurements` 8's
               lesson and CS-7's N=1 rung, repeated because it is free.
    REPRODUCE  `from_array_tiling(ArrayTiling())` must reproduce wake_array's
               own boxes, weights and composed defect exactly.  Without it the
               gate compares a new instrument's number against an old one's.
    NOISE      one candidate scored twice, cold, in the same process.  Every
               ranking difference smaller than the spread is not a ranking
               difference.  `w16_cut_policy`'s CONSTANT_TOL is the same idea
               and it had to be measured rather than guessed there too.
    KNOWN      a geometry whose best cut is known before the run.  The `solo`
               state has ONE rotor in an otherwise near-freestream domain, so
               the wake is a single localized feature and the answer is *do
               not cut along it*.  A search that puts a seam down the wake
               centreline on this state is broken whatever it reports.

CS-9's shared domain
---------------------

`ShellDecomposition` is the second half, and it is a different question rather
than the same one on a smaller mesh.  `case-study-thermal-strain-atlas-0.1`
splits one PDE on one domain into a conduction agent and an elasticity agent,
and its result is that the thermal-strain coupling is a **bond and not a
port**: the interface between the two families has co-dimension ZERO, so there
is no Gamma between them to place.  **The family axis therefore contributes no
candidate space at all, and that is CS-9's own finding restated rather than a
limitation of this file.**

What is placeable on that domain is a **spatial** cut of the shell -- and
because the two families share the domain, a spatial cut cuts both.  So the
question CS-9's geometry poses, and no single-physics geometry can, is:

> **do the two families want the same cut?**

The shell is heated by a Gaussian streak at ``z = L_Z / 2`` of width
``W_STREAK``, so the conduction problem has one sharp localized feature and a
known preference.  Elasticity is quasi-static -- a global elliptic solve with
rigid modes -- and has no local feature at all.  If they disagree, one Q per
seam cannot express the decision, and no scalarization of one seam operator
can, whatever basis it is written in.

No new experts and no new physics
----------------------------------

`RectDecomposition` drives `scaling_ladder.RectangularNS` (and its
no-projection subclass, which is `window_ns._no_projection_class`'s move on a
rectangle).  `ShellDecomposition` restricts `ThermoStruct2D`'s own assembled
``K_th``, ``M_th`` and ``K_me`` to overlapping node sets and pins the
artificial ends -- the same solver `thermal_strain` uses, with a restriction
operator in front of it.  Nothing here integrates a new equation.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from ..probe import cut_score
from . import wake_array as wa
from . import scaling_ladder as sl
from .wake_array import DX, HALO, MACRO_DT, NU_REF, RAMP, fourier_basis, modes_for

__all__ = [
    "Cost", "SeamSpec", "RectDecomposition", "uniform", "from_array_tiling",
    "CandidateScore", "score_candidate", "enumerate_candidates", "search",
    "cut_shear", "CRITERIA",
    "SearchResult", "basis_orbit", "spearman", "exchange_interval",
    "ShellSegmentation", "shell_candidates", "score_shell", "shell_search",
    "DOMAINS", "HAND_CHOSEN",
]


# ---------------------------------------------------------------------------
# the domains, and the hand-chosen tilings that are the gate
# ---------------------------------------------------------------------------

#: The two CS-7 rungs whose tilings a person chose, keyed by the artifact that
#: carries their developed state.  `n_col` / `n_row` are what
#: `scaling_ladder.RUNGS` declares and what `case-study-scaling-ladder-atlas-0.1`
#: 2's table publishes; the domain follows from them through
#: `ArrayTiling.nx / .ny`, so nothing here is a second copy of a number.
HAND_CHOSEN: dict[str, dict[str, Any]] = {
    "N6": {"n_col": 3, "n_row": 2, "state": "out/w100/state_N6.npz",
           "solo": "out/w100/solo_N6.npz",
           "note": "the wake array's own tiling (CS-6) and CS-7's N=6 rung"},
    "N12": {"n_col": 4, "n_row": 3, "state": "out/w100/state_N12.npz",
            "solo": "out/w100/solo_N12.npz",
            "note": "CS-7's N=12 rung"},
}

#: ``label -> (nx, ny)``, derived from the tiling rather than restated.
DOMAINS: dict[str, tuple[int, int]] = {
    k: (wa.ArrayTiling(n_col=v["n_col"], n_row=v["n_row"], rotors=()).nx,
        wa.ArrayTiling(n_col=v["n_col"], n_row=v["n_row"], rotors=()).ny)
    for k, v in HAND_CHOSEN.items()
}


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cost:
    """What a candidate decomposition costs, in the three currencies that differ.

    They differ because a decomposition has two dials and they trade against
    each other: more windows is more agents and less work each, a wider halo is
    the same agent count and more work.  A budget stated in one of these admits
    candidates a budget stated in another refuses, which is why `search` takes a
    predicate on this object rather than a scalar.
    """

    n_windows: int
    work_cells: int
    domain_cells: int

    @property
    def halo_cells(self) -> int:
        """Cells paid for more than once -- the decomposition's whole surcharge."""
        return self.work_cells - self.domain_cells

    @property
    def redundancy(self) -> float:
        return self.work_cells / max(self.domain_cells, 1)

    def as_dict(self) -> dict[str, Any]:
        return {"n_windows": self.n_windows, "work_cells": self.work_cells,
                "domain_cells": self.domain_cells, "halo_cells": self.halo_cells,
                "redundancy": self.redundancy}


@dataclass(frozen=True)
class SeamSpec:
    """One seam of a candidate: which two windows, and the face each exposes.

    ``a_face`` is the ARTIFICIAL face of window ``a`` that looks toward ``b``
    and ``b_face`` is ``b``'s looking back.  They sit at different global
    positions -- separated by the overlap -- which is what an overlapping
    decomposition means and is `wake_array.connections`' own convention.
    """

    seam_id: str
    a: int
    b: int
    axis: str            # 'x' or 'y'
    a_face: str          # 'xhi' or 'yhi'
    b_face: str          # 'xlo' or 'ylo'
    n_cells: int         # the face length, common to both sides (conforming)

    def as_dict(self) -> dict[str, Any]:
        return {"seam_id": self.seam_id, "a": self.a, "b": self.b,
                "axis": self.axis, "a_face": self.a_face, "b_face": self.b_face,
                "n_cells": self.n_cells, "dim_M": modes_for(self.n_cells)}


# ---------------------------------------------------------------------------
# the candidate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RectDecomposition:
    """An overlapping rectangular decomposition of a fixed ``ny x nx`` domain.

    ``x_cuts`` and ``y_cuts`` are the INTERIOR cut positions in cells, strictly
    increasing, strictly inside the domain.  ``halo`` extends each window past
    its cuts on both sides, so adjacent windows overlap by ``2 * halo`` and the
    partition of unity has somewhere to ramp.  ``ramp`` is how many cells that
    takes, and it is `wake_array.RAMP` by default because **W53 keeps ramp 8**
    and every constant this vault has measured is measured at it.

    The empty cut tuple is legal and is the ZERO control: one window, no
    interface, and a composed defect that must be exactly zero.
    """

    nx: int
    ny: int
    x_cuts: tuple[int, ...] = ()
    y_cuts: tuple[int, ...] = ()
    #: Per-side extension past a cut.  The OVERLAP is twice this, so
    #: ``halo=8`` reproduces `wake_array.HALO` = 16.
    halo: int = HALO // 2
    ramp: int = RAMP

    def __post_init__(self) -> None:
        for name, cuts, n in (("x_cuts", self.x_cuts, self.nx),
                              ("y_cuts", self.y_cuts, self.ny)):
            c = list(cuts)
            if any(not (0 < v < n) for v in c):
                raise ValueError(f"{name}={cuts} has a cut outside (0, {n})")
            if any(c[k] >= c[k + 1] for k in range(len(c) - 1)):
                raise ValueError(f"{name}={cuts} is not strictly increasing")
        if self.halo < 0 or self.ramp < 1:
            raise ValueError(f"halo={self.halo}, ramp={self.ramp}")

    # -- geometry ---------------------------------------------------------

    @staticmethod
    def _spans(n: int, cuts: Sequence[int], halo: int) -> list[tuple[int, int]]:
        edges = [0, *cuts, n]
        return [(max(0, edges[k] - halo), min(n, edges[k + 1] + halo))
                for k in range(len(edges) - 1)]

    @property
    def x_spans(self) -> list[tuple[int, int]]:
        return self._spans(self.nx, self.x_cuts, self.halo)

    @property
    def y_spans(self) -> list[tuple[int, int]]:
        return self._spans(self.ny, self.y_cuts, self.halo)

    @property
    def n_col(self) -> int:
        return len(self.x_cuts) + 1

    @property
    def n_row(self) -> int:
        return len(self.y_cuts) + 1

    @property
    def n_windows(self) -> int:
        return self.n_col * self.n_row

    @property
    def boxes(self) -> list[tuple[int, int, int, int]]:
        """``(x0, x1, y0, y1)`` per window, row-major -- `ArrayTiling.offsets`' order."""
        return [(x0, x1, y0, y1)
                for (y0, y1) in self.y_spans for (x0, x1) in self.x_spans]

    @property
    def names(self) -> list[str]:
        return [f"W{i}{j}" for j in range(self.n_row) for i in range(self.n_col)]

    @property
    def shapes(self) -> list[tuple[int, int]]:
        return [(y1 - y0, x1 - x0) for x0, x1, y0, y1 in self.boxes]

    def artificial_faces(self, k: int) -> tuple[str, ...]:
        """The faces of window ``k`` that are cuts rather than domain boundary."""
        x0, x1, y0, y1 = self.boxes[k]
        out = []
        if x0 > 0:
            out.append("xlo")
        if x1 < self.nx:
            out.append("xhi")
        if y0 > 0:
            out.append("ylo")
        if y1 < self.ny:
            out.append("yhi")
        return tuple(out)

    # -- the partition of unity -------------------------------------------

    @lru_cache(maxsize=64)
    def weights(self) -> list[np.ndarray]:
        """`ArrayTiling.weights`' rule, on windows that need not be the same size.

        ``min(wy, wx) ** 2`` over a linear ramp at ARTIFICIAL faces only, then
        normalized by the sum -- so the ramp is quadratic, the partition is
        convex, and a domain boundary keeps full weight because it is a real
        boundary and not a cut.  Reproduced from `ArrayTiling` rather than
        imported because that one closes over a single window size; asserted
        bit-identical on the hand-chosen tiling in
        `tests/test_tier25_seam_placement.py`.
        """
        r = max(self.ramp, 1)
        raw = []
        for k, (x0, x1, y0, y1) in enumerate(self.boxes):
            w_, h_ = x1 - x0, y1 - y0
            faces = self.artificial_faces(k)
            ix, iy = np.arange(w_) + 0.5, np.arange(h_) + 0.5
            wx, wy = np.ones(w_), np.ones(h_)
            if "xlo" in faces:
                wx = np.minimum(wx, np.clip(ix / r, 0.0, 1.0))
            if "xhi" in faces:
                wx = np.minimum(wx, np.clip((w_ - ix) / r, 0.0, 1.0))
            if "ylo" in faces:
                wy = np.minimum(wy, np.clip(iy / r, 0.0, 1.0))
            if "yhi" in faces:
                wy = np.minimum(wy, np.clip((h_ - iy) / r, 0.0, 1.0))
            w = np.zeros((self.ny, self.nx))
            w[y0:y1, x0:x1] = np.minimum(wy[:, None], wx[None, :]) ** 2
            raw.append(w)
        tot = np.sum(raw, axis=0)
        tot = np.where(tot <= 0.0, 1.0, tot)
        return [w / tot for w in raw]

    def local_weights(self) -> list[np.ndarray]:
        """The same weights, restricted to each window's own box."""
        return [w[y0:y1, x0:x1]
                for w, (x0, x1, y0, y1) in zip(self.weights(), self.boxes)]

    def cut(self, f: np.ndarray) -> list[np.ndarray]:
        return [f[y0:y1, x0:x1] for x0, x1, y0, y1 in self.boxes]

    def assemble(self, locals_: Sequence[np.ndarray]) -> np.ndarray:
        out = np.zeros((self.ny, self.nx))
        for chi, (x0, x1, y0, y1), loc in zip(self.local_weights(), self.boxes,
                                              locals_):
            out[y0:y1, x0:x1] += chi * loc
        return out

    # -- the seams --------------------------------------------------------

    @lru_cache(maxsize=64)
    def seams(self) -> list[SeamSpec]:
        """Side-adjacent window pairs, in a fixed order so a seam id is stable.

        Diagonal neighbours are NOT seams and they DO overlap -- CS-7's
        ``overlap_pairs`` counts them and `wake_array.connections` does not.
        The difference is deliberate on both sides: a composed defect is created
        wherever two supports meet, and a `Connection` is declared only where a
        face is shared.  The scores below are aggregated over seams and the
        defect is measured over the whole grid, so the two counts are reported
        rather than reconciled.
        """
        out = []
        nc, nr = self.n_col, self.n_row
        ys, xs = self.y_spans, self.x_spans
        for j in range(nr):
            for i in range(nc - 1):
                a, b = j * nc + i, j * nc + i + 1
                out.append(SeamSpec(f"S-x{i}{j}", a, b, "x", "xhi", "xlo",
                                    ys[j][1] - ys[j][0]))
        for j in range(nr - 1):
            for i in range(nc):
                a, b = j * nc + i, (j + 1) * nc + i
                out.append(SeamSpec(f"S-y{i}{j}", a, b, "y", "yhi", "ylo",
                                    xs[i][1] - xs[i][0]))
        return out

    def overlap_pairs(self) -> list[tuple[int, int]]:
        """Window pairs whose POSITIVE-weight supports intersect, diagonals included."""
        ws = self.weights()
        sup = [w > 0.0 for w in ws]
        out = []
        for a in range(self.n_windows):
            for b in range(a + 1, self.n_windows):
                if np.any(sup[a] & sup[b]):
                    out.append((a, b))
        return out

    # -- cost -------------------------------------------------------------

    @property
    def cost(self) -> Cost:
        return Cost(n_windows=self.n_windows,
                    work_cells=int(sum(h * w for h, w in self.shapes)),
                    domain_cells=self.nx * self.ny)

    # -- reporting --------------------------------------------------------

    @property
    def label(self) -> str:
        return (f"{self.n_col}x{self.n_row}"
                f"|x{'-'.join(map(str, self.x_cuts)) or 'none'}"
                f"|y{'-'.join(map(str, self.y_cuts)) or 'none'}"
                f"|h{self.halo}")

    @property
    def is_uniform(self) -> bool:
        """Are the cuts evenly spaced -- i.e. is this a tiling a person would type?"""
        for cuts, n in ((self.x_cuts, self.nx), (self.y_cuts, self.ny)):
            m = len(cuts) + 1
            want = tuple(int(round(k * n / m)) for k in range(1, m))
            if tuple(cuts) != want:
                return False
        return True

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label, "nx": self.nx, "ny": self.ny,
            "x_cuts": list(self.x_cuts), "y_cuts": list(self.y_cuts),
            "halo": self.halo, "overlap": 2 * self.halo, "ramp": self.ramp,
            "n_col": self.n_col, "n_row": self.n_row,
            "n_windows": self.n_windows, "n_seams": len(self.seams()),
            "n_overlap_pairs": len(self.overlap_pairs()),
            "boxes": [list(b) for b in self.boxes],
            "shapes": [list(s) for s in self.shapes],
            "is_uniform": self.is_uniform,
            "cost": self.cost.as_dict(),
        }


def uniform(nx: int, ny: int, n_col: int, n_row: int,
            halo: int = HALO // 2, ramp: int = RAMP) -> RectDecomposition:
    """The evenly-spaced candidate -- what a person types when they type a tiling."""
    return RectDecomposition(
        nx=nx, ny=ny,
        x_cuts=tuple(int(round(k * nx / n_col)) for k in range(1, n_col)),
        y_cuts=tuple(int(round(k * ny / n_row)) for k in range(1, n_row)),
        halo=halo, ramp=ramp)


def from_array_tiling(t: wa.ArrayTiling) -> RectDecomposition:
    """The bridge, and the REPRODUCE control.

    `ArrayTiling` puts windows of exactly `wake_array.N` cells at a stride of
    `wake_array.STRIDE`, so its overlap is ``N - STRIDE = HALO`` and its
    interior cut sits at the midpoint of each overlap.  Written in this file's
    parameterization that is ``halo = HALO / 2`` and a cut at
    ``i * STRIDE + N - HALO / 2``.

    `tests/test_tier25_seam_placement.py` asserts the resulting boxes, weights
    and one-interval composed defect all agree with `wake_array`'s own to the
    bit.  A search graded against a hand-chosen baseline has to be able to
    express that baseline, and this is the assertion that it does.
    """
    h = HALO // 2
    return RectDecomposition(
        nx=t.nx, ny=t.ny,
        x_cuts=tuple(i * wa.STRIDE + wa.N - h for i in range(t.n_col - 1)),
        y_cuts=tuple(j * wa.STRIDE + wa.N - h for j in range(t.n_row - 1)),
        halo=h, ramp=t.ramp)


# ---------------------------------------------------------------------------
# the solvers -- the same expert at two sizes, which is what makes D_i mean
# something
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _rect_exposed_class():
    """`scaling_ladder.RectangularNS` with the elliptic part removed.

    `window_ns._no_projection_class`'s move on a rectangle, and it is the
    arrangement CS-7 5 selected after marching the alternatives to 120
    macro-steps: `L2/R10` takes the elliptic part OUT of the agent and the
    composition layer applies it once, to the assembled field.

    It is what this file probes and steps for a second reason, and it is the
    one `w16_cut_policy` stage C already stated: with no projection inside
    either the window or the monolith, ``D_i = E_i R_i - R_i E`` contains the
    restriction and nothing else -- no projection, no macro-step splitting.  A
    criterion is being graded on how well it predicts the defect a CUT causes,
    so anything else in the comparison is contamination.
    """
    rect = sl._rect_class()

    class NoProjectionRectangularNS(rect):
        def _project(self, u, v):
            return u, v

    return NoProjectionRectangularNS


@lru_cache(maxsize=256)
def solver(nx: int, ny: int, nu: float = NU_REF, exposed: bool = True):
    """One rectangular `WindowNS` at ``ny x nx`` cells of side `wake_array.DX`.

    Cached on the shape: a search evaluates hundreds of candidates and most of
    them reuse window shapes, and the constructor builds a Poisson symbol.
    """
    cls = _rect_exposed_class() if exposed else sl._rect_class()
    return cls(nu=nu, length=nx * DX, n=nx, ny=ny, cfl=0.4,
               transmission="dirichlet")


def exchange_interval(u: np.ndarray, v: np.ndarray, nu: float = NU_REF,
                      cfl: float = 0.4) -> float:
    """The largest step every solver here takes in exactly ONE sub-step.

    `WindowNS.step_batch` picks ``n_sub`` from the advective CFL and the viscous
    limit, and a defect measured over a step the monolith sub-steps differently
    from the windows contains the time splitting as well as the cut.  A window's
    ``u_max`` is a max over a subset of the domain's, so it is never larger, and
    a step that gives the MONOLITH one sub-step gives every window one too.
    `score_candidate` asserts that rather than assuming it.
    """
    umax = float(np.max(np.hypot(u, v))) + 1e-12
    adv = cfl * DX / umax
    vis = 0.2 * DX ** 2 / max(nu, 1e-12)
    return float(min(adv, vis))


def _step(sv, us: np.ndarray, vs: np.ndarray, dt: float,
          bc: tuple[np.ndarray, np.ndarray] | None = None):
    """One batched step with a zero body force and a held Dirichlet ring."""
    z = np.zeros(us.shape)
    return sv.step_batch(us, vs, dt, bc0=(us, vs) if bc is None else bc,
                         bc1=None, force=(z, z))


def local_steps(dec: RectDecomposition, u: np.ndarray, v: np.ndarray, dt: float,
                nu: float = NU_REF) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    """``E_i R_i (u, v)`` for every window, batched over equal shapes.

    Returns the per-window fields and the largest ``n_sub`` any window used, so
    the caller can assert the one-sub-step condition instead of trusting it.
    """
    by_shape: dict[tuple[int, int], list[int]] = {}
    for k, s in enumerate(dec.shapes):
        by_shape.setdefault(s, []).append(k)
    lu: list[np.ndarray | None] = [None] * dec.n_windows
    lv: list[np.ndarray | None] = [None] * dec.n_windows
    n_sub = 0
    us_all, vs_all = dec.cut(u), dec.cut(v)
    for (h_, w_), ks in by_shape.items():
        sv = solver(w_, h_, nu)
        us = np.stack([us_all[k] for k in ks])
        vs = np.stack([vs_all[k] for k in ks])
        ou, ov = _step(sv, us, vs, dt)
        n_sub = max(n_sub, sv.last_substeps)
        for n, k in enumerate(ks):
            lu[k], lv[k] = np.asarray(ou[n]), np.asarray(ov[n])
    return lu, lv, n_sub


@lru_cache(maxsize=8)
def _monolith(nx: int, ny: int, nu: float = NU_REF):
    return solver(nx, ny, nu, exposed=True)


def monolith_step(nx: int, ny: int, u: np.ndarray, v: np.ndarray, dt: float,
                  nu: float = NU_REF) -> tuple[np.ndarray, np.ndarray, int]:
    """``E (u, v)`` -- the undivided domain, same class, same cell, same ring."""
    sv = _monolith(nx, ny, nu)
    ou, ov = _step(sv, u[None], v[None], dt)
    return np.asarray(ou[0]), np.asarray(ov[0]), sv.last_substeps


# ---------------------------------------------------------------------------
# the defect, and the three criteria that try to predict it
# ---------------------------------------------------------------------------


def composed_defect(dec: RectDecomposition, u: np.ndarray, v: np.ndarray,
                    dt: float, nu: float = NU_REF) -> dict[str, Any]:
    """One exchange interval: the truth, L2/C2's bound, and the surrogate.

    Returns, all relative to the joint ``(u, v)`` norm -- one number for the
    graph rather than one per component, because ``v`` is an order smaller than
    ``u`` in a through-flow and normalizing separately reports the transverse
    component's smallness as a defect (`w16_cut_policy` stage C's own note):

        defect              || A({E_i R_i x}) - E x ||         the truth
        Q_star_chi_weighted || sum_i chi_i |D_i| ||            L2/C2, needs E
        Q_star_max          || max_i |D_i| ||                  the weight-free form
        Q_hat               || max_{i,j} |E_i R_i x - E_j R_j x| || on overlaps

    and the identity residual, which must be machine zero: `tier0-measurements`
    10.1 proves ``A - E = sum_i R_i^T chi_i D_i`` exactly, for any local maps,
    and a run where that does not close is measuring something else.
    """
    lu, lv, n_sub_w = local_steps(dec, u, v, dt, nu)
    ru, rv, n_sub_m = monolith_step(dec.nx, dec.ny, u, v, dt, nu)
    ws, boxes = dec.local_weights(), dec.boxes

    acc = {k: 0.0 for k in
           ("defect", "wbound", "mbound", "qhat", "ident", "scale")}
    per_component = {}
    for tag, loc, ref, x0 in (("u", lu, ru, u), ("v", lv, rv, v)):
        lhs = dec.assemble(loc) - ref
        rhs = np.zeros_like(lhs)
        wb = np.zeros_like(lhs)
        mb = np.zeros_like(lhs)
        glob, own = [], []
        for k, (x0b, x1b, y0b, y1b) in enumerate(boxes):
            sl_ = (slice(y0b, y1b), slice(x0b, x1b))
            chi = ws[k]
            D = loc[k] - ref[sl_]
            rhs[sl_] += chi * D
            wb[sl_] += chi * np.abs(D)
            mb[sl_] = np.maximum(mb[sl_], np.abs(D) * (chi > 0.0))
            g = np.zeros_like(lhs)
            g[sl_] = loc[k]
            o = np.zeros(lhs.shape, dtype=bool)
            o[sl_] = chi > 0.0
            glob.append(g)
            own.append(o)
        qh = np.zeros_like(lhs)
        for a, b in dec.overlap_pairs():
            both = own[a] & own[b]
            qh = np.maximum(qh, np.abs(glob[a] - glob[b]) * both)
        acc["defect"] += float(np.sum(lhs ** 2))
        acc["wbound"] += float(np.sum(wb ** 2))
        acc["mbound"] += float(np.sum(mb ** 2))
        acc["qhat"] += float(np.sum(qh ** 2))
        acc["ident"] = max(acc["ident"], float(np.max(np.abs(lhs - rhs))))
        acc["scale"] = max(acc["scale"], float(np.max(np.abs(lhs))))
        per_component[tag] = {
            "defect_abs": float(np.linalg.norm(lhs)),
            "cellwise_bound_violation": float(np.max(np.abs(lhs) - mb)),
        }

    joint = float(np.sqrt(np.sum(u ** 2) + np.sum(v ** 2)))
    out = {
        "defect": float(np.sqrt(acc["defect"])) / joint,
        "Q_star_chi_weighted": float(np.sqrt(acc["wbound"])) / joint,
        "Q_star_max": float(np.sqrt(acc["mbound"])) / joint,
        "Q_hat_reference_free": float(np.sqrt(acc["qhat"])) / joint,
        "identity_residual_abs": acc["ident"],
        "identity_residual_rel": acc["ident"] / max(acc["scale"], 1e-300),
        "max_abs_defect": acc["scale"],
        "cellwise_bound_violation": max(
            c["cellwise_bound_violation"] for c in per_component.values()),
        "n_sub_windows": n_sub_w,
        "n_sub_monolith": n_sub_m,
        "per_component": per_component,
    }
    out["bound_tightness"] = out["defect"] / max(out["Q_star_chi_weighted"], 1e-300)
    out["surrogate_ratio"] = (out["Q_hat_reference_free"]
                              / max(out["Q_star_max"], 1e-300))
    return out


def marched_defect(dec: RectDecomposition, u: np.ndarray, v: np.ndarray,
                   dt: float, n_intervals: int, nu: float = NU_REF) -> float:
    """The same defect after ``n_intervals`` exchange intervals of both columns.

    `tier0-measurements` 10.2 ranked at one interval AND at eight, and this
    vault's own standing lesson is that a number true as far as it was marched
    is a number true as far as it was marched.  A criterion that predicts the
    one-step defect and not the marched one has not predicted anything a
    rollout cares about.

    **No projection is applied between intervals**, in either column, for the
    reason `_rect_exposed_class` states: the comparison is of restrictions.
    """
    cu, cv = u.copy(), v.copy()
    mu, mv = u.copy(), v.copy()
    for _ in range(n_intervals):
        lu, lv, _ = local_steps(dec, cu, cv, dt, nu)
        cu, cv = dec.assemble(lu), dec.assemble(lv)
        mu, mv, _ = monolith_step(dec.nx, dec.ny, mu, mv, dt, nu)
    num = float(np.sqrt(np.sum((cu - mu) ** 2) + np.sum((cv - mv) ** 2)))
    den = float(np.sqrt(np.sum(mu ** 2) + np.sum(mv ** 2)))
    return num / max(den, 1e-300)


# ---------------------------------------------------------------------------
# the probe -- Q's own inputs, from the framework's own function
# ---------------------------------------------------------------------------

_RING = {"xlo": ((slice(None), 0), (slice(None), 1)),
         "xhi": ((slice(None), -1), (slice(None), -2)),
         "ylo": ((0, slice(None)), (1, slice(None))),
         "yhi": ((-1, slice(None)), (-2, slice(None)))}

#: The finite-difference step `probe.probe_block` uses when an expert declares
#: no reproducibility floor.  Restated rather than imported because it is
#: computed inline there; `tests/test_tier25_seam_placement.py` pins the two
#: together so they cannot drift.
PROBE_STEP = 1e-2


def _block(dec: RectDecomposition, k: int, face: str, u: np.ndarray,
           v: np.ndarray, dt: float, nu: float = NU_REF,
           step: float = PROBE_STEP) -> np.ndarray:
    """``P^* Lambda_k P`` for one side of one seam, all modes in one batched call.

    The port is `wake_array.FluidWindow`'s: an ABSOLUTE normal-velocity trace
    written into the face's ring cells, and the Steklov-Poincare flux
    ``nu dw/dn`` on those same cells returned -- `probed-dtn-coupling` 2.2's
    normative MECH effort.  The prolongation is the same real Fourier basis
    `wake_array._prolongation` declares, so the Gram is ``dx I`` and the forced
    adjoint is ``dx P^T`` with no solve.

    The base trace is the window's OWN ring velocity, not zero: **W74**, and on
    this port the difference is the whole operating point, since an absolute
    velocity of zero is a state no window in a wind farm is ever in.
    """
    x0, x1, y0, y1 = dec.boxes[k]
    uu, vv = u[y0:y1, x0:x1], v[y0:y1, x0:x1]
    ring, interior = _RING[face]
    normal_u = face.startswith("x")
    w0 = uu if normal_u else vv
    base = np.asarray(w0[ring], dtype=float)
    n = base.size
    P = fourier_basis(n)
    m = P.shape[1]

    # base + every mode, as one batch: column 0 is the base probe, whose
    # response is subtracted from all the others (probe._columns_finite_difference)
    traces = np.column_stack([base] + [base + step * P[:, c] for c in range(m)])
    bu = np.repeat(uu[None], m + 1, axis=0)
    bv = np.repeat(vv[None], m + 1, axis=0)
    tgt = bu if normal_u else bv
    for c in range(m + 1):
        sub = tgt[c]
        if normal_u:
            sub[ring[0], ring[1]] = traces[:, c]
        else:
            sub[ring[0], ring[1]] = traces[:, c]
    sv = solver(x1 - x0, y1 - y0, nu)
    ou, ov = _step(sv, bu, bv, dt)
    out = ou if normal_u else ov
    flux = np.stack([
        nu * (np.asarray(out[c][ring]) - np.asarray(out[c][interior])) / DX
        for c in range(m + 1)])
    cols = (flux[1:] - flux[0]) / step                     # (m, n)
    return (DX * P.T) @ cols.T                             # (m, m)


def seam_scores(dec: RectDecomposition, u: np.ndarray, v: np.ndarray, dt: float,
                nu: float = NU_REF) -> list[dict[str, Any]]:
    """``S = P_a^* Lambda_a P_a + P_b^* Lambda_b P_b`` and its scalarizations, per seam.

    ``beta`` is ``sigma_min`` and ``Q`` is `probe.cut_score` -- the framework's
    own retired-but-kept function, called here rather than reimplemented, so
    that what is graded below is 9's score and not a paraphrase of it.
    """
    rows = []
    for s in dec.seams():
        Sa = _block(dec, s.a, s.a_face, u, v, dt, nu)
        Sb = _block(dec, s.b, s.b_face, u, v, dt, nu)
        S = Sa + Sb
        sv = np.linalg.svd(S, compute_uv=False)
        beta = float(sv[-1])
        smax = float(sv[0])
        off = float(np.linalg.norm(S - np.diag(np.diag(S))))
        rows.append({
            **s.as_dict(),
            "beta": beta, "sigma_max": smax,
            "kappa": smax / max(beta, 1e-300),
            "norm_S": float(np.linalg.norm(S)),
            "off_diagonal_mass": off,
            "off_diagonal_fraction": off / max(float(np.linalg.norm(S)), 1e-300),
            "Q_ai_inference": cut_score(S, beta),
            "S": S,
        })
    return rows


def aggregate_seams(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per-seam scores to one number per candidate -- BOTH ways, on purpose.

    **W58 is the reason both are reported.**  `tier0-measurements` 19.5 found
    that a max over neighbour pairs and a norm over the whole grid *"diverge as
    the tiling grows whatever the geometry does"* -- by a factor of three across
    CS-7's ladder.  A search compares candidates with different seam counts, so
    the aggregation rule is not a presentational choice here: it is part of the
    criterion, and a criterion that ranks one way under `max` and another under
    `sum` has not stated itself.
    """
    if not rows:
        return {"n_seams": 0}
    out: dict[str, Any] = {"n_seams": len(rows)}
    for key in ("Q_ai_inference", "off_diagonal_mass", "off_diagonal_fraction",
                "beta", "kappa", "norm_S"):
        vals = np.asarray([r[key] for r in rows if r[key] is not None], dtype=float)
        if vals.size == 0:
            continue
        out[f"{key}__max"] = float(vals.max())
        out[f"{key}__min"] = float(vals.min())
        out[f"{key}__mean"] = float(vals.mean())
        out[f"{key}__sum"] = float(vals.sum())
        out[f"{key}__l2"] = float(np.linalg.norm(vals))
    return out


def cut_shear(dec: RectDecomposition, u: np.ndarray, v: np.ndarray) -> dict[str, Any]:
    """The normal shear of the STATE along the cut planes -- G5's slogan, as a number.

    `generalization-requirements` G5 proposes cutting *"never along a shear
    layer, wake centreline or reaction front"*, and `interface-transfer-theory`
    9 offers Q as the form that slogan could be measured in.  This is the
    other form, and it is the one the slogan literally says: the mean
    ``|d_n (u, v)|`` over the interior cut lines, taken from the field itself.

    **It reads the state and not the operator, and that is the point.**
    `tier0-measurements` 10.2's uniform-viscosity control found the decision a
    wind farm actually faces -- cut through the wake or beside it -- *"is not
    expressible in the operator alone"*, because every placement there had a
    provably identical ``S`` while the measured defect spread 2.29x.  A
    criterion computed from the state has no such blindness by construction.

    It is also **free**: no probe, no monolith, no local solve, no boundary
    response.  Two array gradients and a slice, on data a scheduler already
    has.  If it ranks, it ranks at a cost every other criterion here is
    thousands of times above.
    """
    duy, dux = np.gradient(u)
    dvy, dvx = np.gradient(v)
    vals: list[np.ndarray] = []
    for c in dec.x_cuts:
        vals.append(np.hypot(dux[:, c], dvx[:, c]) / DX)
    for c in dec.y_cuts:
        vals.append(np.hypot(duy[c, :], dvy[c, :]) / DX)
    if not vals:
        return {}
    flat = np.concatenate(vals)
    per_cut = np.array([float(x.mean()) for x in vals])
    return {
        # length-weighted over every cut plane: one number for the graph
        "cut_shear__mean": float(flat.mean()),
        # the worst single cut, which is the aggregation W58 says to report too
        "cut_shear__max": float(per_cut.max()),
        "cut_shear__per_cut": per_cut.tolist(),
        "cut_shear__peak_cellwise": float(flat.max()),
    }


def basis_orbit(S: np.ndarray, n_frames: int = 64, seed: int = 12345) -> dict[str, Any]:
    """Q's orbit under an admissible re-declaration of the SAME interface space.

    ``P -> P U`` for orthogonal ``U`` declares the same space -- ``range(PU) =
    range(P)`` -- so the scheme is unchanged and ``S -> U^T S U``.  ``beta`` and
    ``||S||`` are invariant; the off-diagonal mass is not.

    `tier0-measurements` 10.2 measured this on ONE seam of a linear strip model
    and found a 361x orbit.  It is measured here per seam of a real
    decomposition of a real flow, because 9's own flagged weakness is that the
    Fourier basis is a locality measure *by assumption*, and the assumption is
    weakest exactly where a placement search wants to go.
    """
    beta = float(np.linalg.svd(S, compute_uv=False)[-1])
    q0 = cut_score(S, beta)
    rng = np.random.default_rng(seed)
    qs = []
    for _ in range(n_frames):
        U, _r = np.linalg.qr(rng.normal(size=S.shape))
        Su = U.T @ S @ U
        qs.append(cut_score(Su, float(np.linalg.svd(Su, compute_uv=False)[-1])))
    sym = 0.5 * (S + S.T)
    _w, V = np.linalg.eigh(sym)
    S_eig = V.T @ S @ V
    beta_eig = float(np.linalg.svd(S_eig, compute_uv=False)[-1])
    q_eig = cut_score(S_eig, beta_eig)
    lo = min([q for q in qs if q is not None] + [q_eig])
    hi = max([q for q in qs if q is not None] + [q0])
    return {
        "Q_declared_fourier": q0,
        "Q_min_over_random_frames": float(min(qs)),
        "Q_max_over_random_frames": float(max(qs)),
        "Q_in_eigenbasis_of_sym_part": q_eig,
        "orbit_ratio": float(hi / max(lo, 1e-300)),
        "beta_declared": beta, "beta_in_eigenbasis": beta_eig,
        "beta_invariance_rel": abs(beta_eig - beta) / max(beta, 1e-300),
        "n_frames": n_frames,
    }


# ---------------------------------------------------------------------------
# a candidate's full score
# ---------------------------------------------------------------------------

#: The criteria a search may be driven by.  ``reference_free`` is the only one
#: of them a search in production could actually evaluate: the other three need
#: either a monolith (which a frozen expert does not have -- W95) or a probe.
CRITERIA = ("Q_ai_inference__max", "Q_ai_inference__sum",
            "off_diagonal_mass__max", "off_diagonal_mass__sum",
            "Q_hat_reference_free", "Q_star_chi_weighted",
            "cut_shear__mean", "cut_shear__max")


@dataclass
class CandidateScore:
    """Everything measured about one candidate decomposition."""

    dec: RectDecomposition
    defect: dict[str, Any]
    seams: list[dict[str, Any]] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)
    marched: dict[str, float] = field(default_factory=dict)
    #: State-side scores: read from the field at the cut, not from the seam
    #: operator and not from a monolith.  See `cut_shear`.
    state: dict[str, Any] = field(default_factory=dict)

    def scorable(self, criterion: str) -> bool:
        """Can this criterion express this candidate at all?

        A one-window decomposition has **no seams**, so every seam-derived
        criterion -- Q included -- has nothing to evaluate on it, while its
        composed defect is exactly zero and is therefore the best available.
        That is not a missing value to be imputed: it is a statement that the
        criterion cannot compare a decomposition against a non-decomposition,
        and `SearchResult` excludes such candidates from a ranking and reports
        how many it excluded rather than ranking them at infinity.
        """
        return (criterion in self.aggregate or criterion in self.defect
                or criterion in self.state)

    def value(self, criterion: str) -> float:
        if criterion in self.aggregate:
            return float(self.aggregate[criterion])
        if criterion in self.defect:
            return float(self.defect[criterion])
        if criterion in self.state:
            return float(self.state[criterion])
        raise KeyError(f"{criterion!r} is not a criterion this score carries")

    def as_dict(self, with_S: bool = False) -> dict[str, Any]:
        seams = [{k: val for k, val in r.items() if with_S or k != "S"}
                 for r in self.seams]
        d = dict(self.defect)
        d.pop("per_component", None)
        return {"decomposition": self.dec.as_dict(), "defect": d,
                "seams": seams, "aggregate": self.aggregate,
                "state": self.state, "marched": self.marched}


def score_candidate(dec: RectDecomposition, u: np.ndarray, v: np.ndarray,
                    dt: float, nu: float = NU_REF, probe: bool = True,
                    march: Sequence[int] = ()) -> CandidateScore:
    """Measure a candidate: the truth, the derived bound, the surrogate, and Q.

    Raises if either column takes more than one sub-step over ``dt``.  That is
    the one-exchange-interval condition `exchange_interval` computes, and it is
    asserted rather than assumed because a defect measured over a step the two
    columns sub-step differently is contaminated by the time splitting and
    would still look like a number.
    """
    d = composed_defect(dec, u, v, dt, nu)
    if d["n_sub_windows"] != 1 or d["n_sub_monolith"] != 1:
        raise ValueError(
            f"dt={dt} is not one exchange interval for {dec.label}: windows took "
            f"{d['n_sub_windows']} sub-steps and the monolith {d['n_sub_monolith']}. "
            "Use `exchange_interval` on this state")
    rows = seam_scores(dec, u, v, dt, nu) if probe else []
    sc = CandidateScore(dec=dec, defect=d, seams=rows,
                        aggregate=aggregate_seams(rows),
                        state=cut_shear(dec, u, v))
    for k in march:
        sc.marched[f"defect_{k}_intervals"] = marched_defect(dec, u, v, dt, k, nu)
    return sc


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------


#: Above this many cut sets on one axis, per-cut offsets are abandoned for a
#: common shift.  A search that enumerates ``|offsets|^k`` placements of ``k``
#: cuts stops being affordable at ``k = 4``, and the cap is stated here rather
#: than chosen inside the loop so that what the search did and did not look at
#: is a declared property of the run.
INDEPENDENT_CAP = 64


def _cut_sets(n: int, k: int, halo: int, offsets: Sequence[int],
              shifts: Sequence[int] = ()) -> list[tuple[int, ...]]:
    """``k`` interior cuts on an axis of ``n`` cells: even spacing, then moved.

    Enumerating every integer cut position is neither affordable nor
    informative -- adjacent positions differ by one cell.  Two families of
    alternative are generated around the evenly-spaced placement a person would
    type, and they answer different questions:

      * ``offsets`` moves **each cut independently**, which is the degree of
        freedom no tiling in this vault has ever had and the one a placement
        algorithm exists to use -- it is what lets a seam step off a wake
        centreline while its neighbours stay put;
      * ``shifts`` moves **the whole set together**, which is the cheap
        alternative and is all that is affordable once ``k`` is large.

    Above `INDEPENDENT_CAP` combinations the independent family is dropped and
    only the common shift survives.  That is a stated limit of the search, not
    a property of the criterion being graded, and `SearchResult.report` carries
    the candidate count so a reader can see how wide the net was.
    """
    if k == 0:
        return [()]
    base = [int(round(i * n / (k + 1))) for i in range(1, k + 1)]
    out: set[tuple[int, ...]] = set()

    def admit(c: tuple[int, ...]) -> None:
        if all(halo < v < n - halo for v in c) and all(
                c[i] + 2 * halo < c[i + 1] for i in range(len(c) - 1)):
            out.add(c)

    for s in (shifts or (0,)):
        admit(tuple(int(x + s) for x in base))
    if len(offsets) ** k <= INDEPENDENT_CAP:
        from itertools import product
        for combo in product(offsets, repeat=k):
            admit(tuple(int(x + o) for x, o in zip(base, combo)))
    return sorted(out)


def enumerate_candidates(nx: int, ny: int,
                         n_cols: Iterable[int], n_rows: Iterable[int],
                         halos: Iterable[int], offsets: Iterable[int],
                         ramp: int = RAMP, shifts: Iterable[int] = (),
                         keep: Callable[[RectDecomposition], bool] | None = None,
                         ) -> list[RectDecomposition]:
    """The candidate space: counts x cut placements x halo widths.

    ``keep`` filters before anything is scored, which is how a pool with a
    stated budget is built: `search` then ranks within a pool rather than
    across pools, because a defect ranking taken across candidates of different
    AGENT COUNTS is dominated by the count and says nothing about placement.
    """
    offs, shf = tuple(offsets), tuple(shifts)
    out, seen = [], set()
    for h in halos:
        for nc in n_cols:
            for nr in n_rows:
                for xs in _cut_sets(nx, nc - 1, h, offs, shf):
                    for ys in _cut_sets(ny, nr - 1, h, offs, shf):
                        d = RectDecomposition(nx=nx, ny=ny, x_cuts=xs, y_cuts=ys,
                                              halo=h, ramp=ramp)
                        if d.label in seen or (keep is not None and not keep(d)):
                            continue
                        seen.add(d.label)
                        out.append(d)
    return out


@dataclass
class SearchResult:
    """What one search returned, and what it is graded against."""

    criterion: str
    budget: str
    scores: list[CandidateScore]
    baseline: CandidateScore | None = None

    @property
    def pool(self) -> list[CandidateScore]:
        """The candidates this criterion can actually express -- see `scorable`."""
        return [s for s in self.scores if s.scorable(self.criterion)]

    @property
    def chosen(self) -> CandidateScore:
        return min(self.pool, key=lambda s: s.value(self.criterion))

    @property
    def best(self) -> CandidateScore:
        """The candidate with the lowest MEASURED defect -- what a perfect criterion picks.

        Taken over the same pool the criterion ranks, so the comparison is like
        for like: crediting a criterion with missing the one-window candidate
        it cannot see would be grading it on a question it was not asked.
        """
        return min(self.pool, key=lambda s: s.defect["defect"])

    @property
    def worst(self) -> CandidateScore:
        return max(self.pool, key=lambda s: s.defect["defect"])

    def report(self) -> dict[str, Any]:
        ch, be, wo = self.chosen, self.best, self.worst
        d_ch, d_be, d_wo = (ch.defect["defect"], be.defect["defect"],
                            wo.defect["defect"])
        pool = self.pool
        out = {
            "criterion": self.criterion, "budget": self.budget,
            "n_candidates": len(pool),
            "n_unscorable": len(self.scores) - len(pool),
            "chosen": ch.dec.label, "chosen_defect": d_ch,
            # the mechanism, so a reader can see WHY the chosen cut is better
            "chosen_shape": [ch.dec.n_col, ch.dec.n_row],
            "chosen_n_seams": len(ch.dec.seams()),
            "chosen_n_overlap_pairs": len(ch.dec.overlap_pairs()),
            "chosen_halo_cells": ch.dec.cost.halo_cells,
            "best_available": be.dec.label, "best_defect": d_be,
            "worst_defect": d_wo,
            # 0 = the criterion picked the best cut available, 1 = the worst
            "fraction_of_available_range_lost":
                (d_ch - d_be) / max(d_wo - d_be, 1e-300),
            "penalty_vs_best": d_ch / max(d_be, 1e-300),
            "spread_factor": d_wo / max(d_be, 1e-300),
            "rank_correlation_vs_defect": spearman(
                [s.value(self.criterion) for s in pool],
                [s.defect["defect"] for s in pool]),
        }
        if self.baseline is not None:
            d_hand = self.baseline.defect["defect"]
            out.update({
                "hand_chosen": self.baseline.dec.label,
                "hand_chosen_defect": d_hand,
                "hand_chosen_shape": [self.baseline.dec.n_col,
                                      self.baseline.dec.n_row],
                "hand_chosen_n_seams": len(self.baseline.dec.seams()),
                "hand_chosen_n_overlap_pairs": len(
                    self.baseline.dec.overlap_pairs()),
                # < 1 means the search beat the person
                "chosen_over_hand_chosen": d_ch / max(d_hand, 1e-300),
                "best_over_hand_chosen": d_be / max(d_hand, 1e-300),
                "hand_chosen_rank": 1 + sum(
                    1 for s in pool if s.defect["defect"] < d_hand),
                "hand_chosen_criterion_rank": 1 + sum(
                    1 for s in pool
                    if s.value(self.criterion) < self.baseline.value(self.criterion)
                ) if self.baseline.scorable(self.criterion) else None,
            })
        return out


def search(scores: Sequence[CandidateScore], criterion: str, budget: str = "",
           baseline: CandidateScore | None = None) -> SearchResult:
    return SearchResult(criterion=criterion, budget=budget,
                        scores=list(scores), baseline=baseline)


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Rank correlation, or None when either side is constant.

    A criterion that takes one value over the whole scan induces no ordering,
    and `numpy.argsort`'s arbitrary tie-break would report a correlation for it
    anyway.  `w16_cut_policy` says "constant" instead and so does this, for the
    same reason: on the uniform-viscosity control, saying *"Q correlates at
    -0.3"* about a quantity that is constant to eleven digits is the error.
    """
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.size < 2 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------------------------------------------------------------------------
# CS-9's shared domain: does the OTHER family want the same cut?
# ---------------------------------------------------------------------------

#: Overlap, in elements along the shell, for a shell segmentation.  The mesh is
#: `thermal_strain.NZ` = 48 elements long, so 2 is proportionally close to
#: CS-7's 16 cells in 128.
SHELL_HALO = 2


@dataclass(frozen=True)
class ShellSegmentation:
    """A spatial decomposition of CS-9's shell, cutting BOTH families at once.

    `case-study-thermal-strain-atlas-0.1` splits one PDE on one domain into a
    conduction agent and an elasticity agent and finds the coupling between
    them is a **bond and not a port**, because their interface has co-dimension
    zero.  So the family axis offers no Gamma to place, and the only placeable
    cut on that domain is a spatial one -- which, because the domain is shared,
    necessarily cuts both families at the same z.

    ``z_cuts`` are element indices in ``(0, NZ)``.  Segment k owns elements
    ``[z_{k-1} - halo, z_k + halo)`` and the nodes they touch.
    """

    n_z: int
    z_cuts: tuple[int, ...] = ()
    halo: int = SHELL_HALO

    def __post_init__(self) -> None:
        c = list(self.z_cuts)
        if any(not (0 < v < self.n_z) for v in c):
            raise ValueError(f"z_cuts={self.z_cuts} outside (0, {self.n_z})")
        if any(c[k] >= c[k + 1] for k in range(len(c) - 1)):
            raise ValueError(f"z_cuts={self.z_cuts} is not strictly increasing")

    @property
    def spans(self) -> list[tuple[int, int]]:
        edges = [0, *self.z_cuts, self.n_z]
        return [(max(0, edges[k] - self.halo), min(self.n_z, edges[k + 1] + self.halo))
                for k in range(len(edges) - 1)]

    @property
    def n_segments(self) -> int:
        return len(self.z_cuts) + 1

    @property
    def label(self) -> str:
        return f"seg{self.n_segments}|z{'-'.join(map(str, self.z_cuts)) or 'none'}|h{self.halo}"

    @property
    def cost(self) -> Cost:
        return Cost(n_windows=self.n_segments,
                    work_cells=int(sum(b - a for a, b in self.spans)),
                    domain_cells=self.n_z)

    def as_dict(self) -> dict[str, Any]:
        return {"label": self.label, "n_z": self.n_z,
                "z_cuts": list(self.z_cuts), "halo": self.halo,
                "n_segments": self.n_segments, "spans": [list(s) for s in self.spans],
                "cost": self.cost.as_dict()}


def _shell_bits():
    """The mesh, the solver and the coupling operators CS-9 already builds."""
    ts_mod = importlib.import_module("atlas.cases.thermal_strain")
    mesh, ts, ops = ts_mod.coupling()
    return ts_mod, mesh, ts, ops


@lru_cache(maxsize=1)
def _node_z():
    """Every node's z coordinate, and the element-column it belongs to.

    The mesh is structured -- `NZ` columns by `NJ` rows of Q1 elements -- so a
    node's element-column index is what a spatial cut is expressed in, and it
    is recovered from the coordinate rather than from an assumed ordering.
    """
    _m, mesh, _ts, _ops = _shell_bits()
    p = mesh.nodes.reshape(-1, 2)
    z = p[:, 0]
    from .thermal_strain import L_Z, NZ
    col = np.clip((z / L_Z * NZ).round().astype(int), 0, NZ)
    return z, col


def _segment_nodes(seg: ShellSegmentation) -> list[np.ndarray]:
    """The node ids each segment owns, from the element span it covers."""
    _z, col = _node_z()
    return [np.flatnonzero((col >= a) & (col <= b)) for a, b in seg.spans]


def _shell_weights(seg: ShellSegmentation) -> list[np.ndarray]:
    """A partition of unity over nodes, ramped across each overlap.

    The same rule `RectDecomposition.weights` uses -- a linear ramp at
    artificial ends only, squared, then normalized -- written on the node's z
    column instead of a cell index.
    """
    _z, col = _node_z()
    n = col.size
    raw = []
    for k, (a, b) in enumerate(seg.spans):
        w = np.zeros(n)
        own = (col >= a) & (col <= b)
        r = max(2 * seg.halo, 1)
        f = np.ones(n)
        if a > 0:
            f = np.minimum(f, np.clip((col - a) / r, 0.0, 1.0))
        if b < seg.n_z:
            f = np.minimum(f, np.clip((b - col) / r, 0.0, 1.0))
        w[own] = f[own] ** 2
        raw.append(w)
    tot = np.sum(raw, axis=0)
    tot = np.where(tot <= 0.0, 1.0, tot)
    return [w / tot for w in raw]


def _restricted_thermal_step(seg: ShellSegmentation, T: np.ndarray,
                             dt: float) -> np.ndarray:
    """One conduction step per segment, Dirichlet on the artificial ends, assembled.

    ``M_th`` and ``K_th`` are `ThermoStruct2D`'s own assembled matrices; this
    restricts them to a segment's nodes and pins the ring to the incoming field
    -- overlapping Schwarz on the operator the agent already has.  Nothing new
    is integrated, which is the constraint `poc1-retrospective-and-hybrid-roadmap`
    4.1 puts on this case study.
    """
    from scipy.sparse.linalg import spsolve
    ts_mod, _mesh, ts, _ops = _shell_bits()
    cond = _shell_cond()
    Kb, fb = cond._robin_blocks(T)
    A = (ts.M_th / dt + ts.K_th + Kb).tocsr()
    rhs = ts.M_th.dot(T) / dt + fb
    _z, col = _node_z()
    ws = _shell_weights(seg)
    out = np.zeros_like(T)
    for k, (idx, (a, b)) in enumerate(zip(_segment_nodes(seg), seg.spans)):
        ring = np.zeros(idx.size, dtype=bool)
        if a > 0:
            ring |= col[idx] <= a
        if b < seg.n_z:
            ring |= col[idx] >= b
        free = idx[~ring]
        loc = T.copy()
        if free.size:
            sub = A[free][:, free].tocsc()
            r = rhs[free] - A[free][:, idx[ring]].dot(T[idx[ring]])
            loc[free] = spsolve(sub, r)
        out += ws[k] * loc
    return out


def _restricted_elastic_solve(seg: ShellSegmentation, dT: np.ndarray,
                              u_lag: np.ndarray) -> np.ndarray:
    """One elasticity solve per segment, artificial ends pinned to LAGGED data.

    Quasi-static, so there is no time step and nothing to sub-step: what makes
    this a decomposition rather than a re-solve of the same problem is the
    **transmission datum**, and it is the same one conduction uses -- the field
    at ``t^n``.  Pinning the ring to the *current* monolith solution would make
    every segment solve exact (an elliptic problem with exact Dirichlet data
    returns the exact solution) and the measured defect would be round-off.
    That is a real trap and this file fell into it on its first run: the
    elasticity defect came back at 1e-11 for every cut, which is not a small
    number, it is no measurement at all.

    ``u_lag`` is therefore the displacement field the previous temperature
    produced, and the defect this returns is the lagged-interface error the cut
    actually causes -- the elastic analogue of `wake_array`'s held Dirichlet
    ring, and of `general-coupling-scheme` R3's ``bc_time_varying``.

    **One segment is the monolith**, and it is taken as such: with no
    artificial end there is no ring, ``K_me`` is singular on the rigid modes,
    and the agent's own `_solve_free` is the only correct answer.  Handing that
    case to a constrained solve would return a rigid-body mode and report it as
    a decomposition error.
    """
    from scipy.sparse.linalg import spsolve
    _mod, _mesh, ts, ops = _shell_bits()
    load = ops.G.dot(dT)
    elas = _shell_elas()
    if seg.n_segments == 1:
        return elas.solve(load)
    _z, col = _node_z()
    ws = _shell_weights(seg)
    K = ts.K_me.tocsr()
    out = np.zeros_like(u_lag)
    for k, (idx, (a, b)) in enumerate(zip(_segment_nodes(seg), seg.spans)):
        ring = np.zeros(idx.size, dtype=bool)
        if a > 0:
            ring |= col[idx] <= a
        if b < seg.n_z:
            ring |= col[idx] >= b
        # Every node outside the segment is pinned to the lagged field too: the
        # local problem is posed on the segment and the rest never moves.
        free = np.concatenate([2 * idx[~ring], 2 * idx[~ring] + 1])
        fixed = np.setdiff1d(np.arange(u_lag.size), free)
        loc = u_lag.copy()
        if free.size:
            r = load[free] - K[free][:, fixed].dot(loc[fixed])
            loc[free] = spsolve(K[free][:, free].tocsc(), r)
        out += np.repeat(ws[k], 2) * loc
    return out


@lru_cache(maxsize=1)
def _shell_cond():
    from .thermal_strain import ConductionAgent
    return ConductionAgent()


@lru_cache(maxsize=1)
def _shell_elas():
    from .thermal_strain import ElasticityAgent
    return ElasticityAgent()


@lru_cache(maxsize=1)
def shell_state(n_steps: int = 8):
    """A developed field and the one before it: ``(T_prev, T)``.

    Scoring a cut on the initial uniform field would be the uniform-viscosity
    control of `tier0-measurements` 10.2 by accident -- with no feature there is
    no placement question.  The streak is what makes the field two-dimensional
    and the cut decision real, and it needs a few steps to establish.

    ``T_prev`` is returned because the elasticity half needs a **lagged**
    transmission datum and quasi-statics supplies none of its own: the
    displacement the previous temperature produced is the field a segment's
    artificial end carries.  See `_restricted_elastic_solve`.
    """
    from .thermal_strain import DT_MACRO
    cond = _shell_cond()
    T = np.array(cond._T, dtype=float)
    T_prev = T.copy()
    for _ in range(n_steps):
        T_prev = T
        T = cond.step(T, DT_MACRO)
    return T_prev, T


def score_shell(seg: ShellSegmentation, state=None,
                dt: float | None = None) -> dict[str, Any]:
    """Both families' composed defect for ONE spatial cut of the shared domain.

    Returns the conduction defect and the elasticity defect separately, because
    **whether they agree is the question**.  A single Gamma serves both, so a
    criterion that scores the seam has to produce one number for a cut two
    physics have different opinions about -- and no scalarization of one seam
    operator can, whatever basis it is written in.

    Both halves are measured the same way: the composed step takes its
    artificial-end data from ``t^n`` and the referent is the same operator on
    the undivided domain.  That symmetry is what makes the two numbers
    comparable, and it is the reason the elasticity half carries a lagged field
    rather than the answer it is being graded against.
    """
    from .thermal_strain import DT_MACRO, L_Z, NZ, T_REF, W_STREAK
    T_prev, T = shell_state() if state is None else state
    dt = DT_MACRO if dt is None else dt
    cond, elas = _shell_cond(), _shell_elas()
    G = _shell_bits()[3].G

    T_ref = cond.step(T, dt)
    T_comp = _restricted_thermal_step(seg, T, dt)
    dT, dT_prev = T - T_REF, T_prev - T_REF
    u_ref = elas.solve(G.dot(dT))
    u_lag = elas.solve(G.dot(dT_prev))
    u_comp = _restricted_elastic_solve(seg, dT, u_lag)

    def rel(a, b):
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))

    # where the cuts sit relative to the streak, in units of its own width
    z_cut_m = [c / NZ * L_Z for c in seg.z_cuts]
    dist = [abs(z - 0.5 * L_Z) / W_STREAK for z in z_cut_m]
    return {
        **seg.as_dict(),
        "conduction_defect": rel(T_comp, T_ref),
        "elasticity_defect": rel(u_comp, u_ref),
        # what the lag alone costs with NO cut: the floor both defects sit on
        "elasticity_lag_only": rel(u_lag, u_ref),
        "z_cuts_m": z_cut_m,
        "cut_distance_from_streak_in_widths": dist,
        "nearest_cut_to_streak": (min(dist) if dist else None),
    }


def shell_candidates(n_z: int, n_segments: Iterable[int],
                     halo: int = SHELL_HALO) -> list[ShellSegmentation]:
    """Every admissible placement of ``k-1`` cuts, for each ``k`` asked for.

    The shell has 48 elements, so the space is small enough to enumerate
    exhaustively at two and three segments -- which removes the objection that a
    search found nothing because it looked in the wrong place.
    """
    out = []
    for k in n_segments:
        if k == 1:
            out.append(ShellSegmentation(n_z=n_z, z_cuts=(), halo=halo))
            continue
        from itertools import combinations
        lo, hi = halo + 1, n_z - halo
        for cuts in combinations(range(lo, hi), k - 1):
            if any(cuts[i] + 2 * halo >= cuts[i + 1] for i in range(len(cuts) - 1)):
                continue
            out.append(ShellSegmentation(n_z=n_z, z_cuts=cuts, halo=halo))
    return out


def shell_search(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    """Rank shell candidates by one family's defect and report the other's."""
    best = min(rows, key=lambda r: r[key])
    other = ("elasticity_defect" if key == "conduction_defect"
             else "conduction_defect")
    best_other = min(rows, key=lambda r: r[other])
    return {
        "ranked_by": key,
        "argmin": best["label"],
        "argmin_z_cuts": best["z_cuts"],
        key: best[key], other: best[other],
        "other_family_argmin": best_other["label"],
        "other_family_argmin_z_cuts": best_other["z_cuts"],
        "families_agree": best["label"] == best_other["label"],
        "cost_of_following_this_family_for_the_other":
            best[other] / max(best_other[other], 1e-300),
        "rank_correlation_between_families": spearman(
            [r["conduction_defect"] for r in rows],
            [r["elasticity_defect"] for r in rows]),
    }
