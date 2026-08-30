"""L6 -- assembly, and the certificate that is half checkable and half open.

end-to-end-architecture-spec §8.  The local solves come back overlapping and
disagreeing slightly; assembly is the rule that turns them into one field.  It is
the least examined layer in the framework and it carries a measured 19% of total
error -- a share that survives perfect agents, perfect windows and a perfect
interface solve.

Checkable today, at machine precision:

    sum_i R_i^T chi_i R_i == I

A partition of unity that fails this identity is producing a weighted average
that is not an average.  It is a one-line test, it is currently run nowhere, and
its failure is exactly the silent-wrongness class.  Refuse on failure.

**``condition`` was the open field until 2026-08-28, and it is now L6/C1.**  The
condition that the blend be at least as accurate as the local solves it blends is
a condition on ``chi`` alone, and the reason is the cellwise bias-variance
identity, which holds for ANY weights summing to one:

    |A(u) - u*|^2  =  sum_i chi_i |u_i - u*|^2  -  V_chi,
    V_chi := sum_i chi_i u_i^2 - (sum_i chi_i u_i)^2

``V_chi`` is the chi-weighted variance of the local values.  It needs no exact
solution, and it is non-negative for all data exactly when chi >= 0.  So:

    chi >= 0  and  sum_i chi_i = 1
      ==>  |A(u) - u*| <= max_i |u_i - u*| at EVERY cell, hence in every l^p norm

That is the missing statement, it is a theorem rather than an assumption, and its
only hypothesis is checkable from the declaration with no reference and no run.
``R11`` refuses a partition that fails it.  Measured 2026-08-28: the identity
closes to 1e-15 on the real four-window tiling, and a signed partition that passes
every check the framework had before it (identity residual 2.2e-16) blends outside
the hull of what it blends by 1.13e-4 and costs 166x on the composed macro-step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .holes import ASSEMBLY_CERTIFICATE, UNMEASURED_CONSTANTS, Unmeasured

__all__ = [
    "PartitionOfUnity", "GridPartitionOfUnity", "AssemblyCertificate",
    "AssemblyCondition", "certify", "blend_defect", "blend_variance",
    "sigma_halo_bound", "POU_TOL", "CONVEX_TOL", "C_MU_HALO",
]

#: The partition-of-unity identity is exact arithmetic, not an approximation.
#: This tolerance is a machine-precision guard scaled by problem size.
POU_TOL = 1e-10

#: The convexity test is also exact arithmetic: a weight is either negative or it
#: is not.  The tolerance exists only to absorb the roundoff of the normalization
#: ``chi_i = raw_i / sum_k raw_k``, which is the last operation every partition of
#: unity in this package performs.  Measured on the four-window tiling, convex
#: partitions return chi_min = 0.0 exactly and the signed control returns -0.598:
#: there is nothing between them to get wrong.
CONVEX_TOL = 1e-12

#: W49: the transmission-sensitivity constant for the OVERLAPPING branch of the
#: master bound, ``sigma <= C_mu * Pi * ||d_lambda||``.  Measured 2026-08-28 over
#: **16 configurations** of the four-window tiling: eleven varying the halo (1 to
#: 61 cells) and the partition of unity (Pi from 0.02 to 1.0), five varying the
#: Reynolds number (128 to 510) and the macro-step (0.025 to 0.10).  Across all of
#: them sigma spans a factor 4.3e4 and the implied constant stays in
#: **[0.228, 1.178], a spread of 5.2x**.  1.2 is the sampled maximum rounded up.
#:
#: It is a measurement with a stated scope rather than a fit: a constant that
#: moves 5x while the quantity it relates moves 4e4 is behaving like a constant,
#: which is exactly what section 4's substructuring form does NOT do on this
#: branch -- there the implied C_mu is 2.2e-8, which is a diagnosis of the
#: factorization rather than a value for it.
#:
#: Scope: one expert, one governing family, one time discretization.  Nothing
#: here establishes that 1.2 travels to a different agent or to multirate.
C_MU_HALO = 1.2


@dataclass
class PartitionOfUnity:
    """Restrictions and weights, in the form the identity is checkable from.

    ``restrictions[i]`` is R_i as an (n_i, n) matrix and ``weights[i]`` is the
    diagonal of chi_i on subdomain i.  A single-valued-flux assembly (the
    non-overlapping counterpart) has the same structure with the identity
    replaced by flux single-valuedness and the same hole; declare it by passing
    ``kind="single-valued-flux"`` and supplying the flux operator instead.
    """

    n_global: int
    restrictions: dict[str, np.ndarray]
    weights: dict[str, np.ndarray]
    kind: str = "partition-of-unity"
    #: W49. ``contaminated[i]`` marks, on subdomain i's own cells, the ones an
    #: artificial-boundary datum can have reached within one exchange interval --
    #: i.e. cells within ``stencil_radius_i * substeps_per_exchange_i`` of a face
    #: that is not a real boundary.  It is geometry, so the case study declares it;
    #: nothing here can derive it.  Absent, ``contaminated_weight`` returns None
    #: and the sigma bound decertifies rather than assuming the assembly is clean.
    contaminated: dict[str, np.ndarray] | None = None

    def identity_residual(self) -> float:
        """|| sum_i R_i^T chi_i R_i - I ||, the one-line test."""
        total = np.zeros((self.n_global, self.n_global))
        for key, R in self.restrictions.items():
            R = np.asarray(R, dtype=float)
            chi = np.asarray(self.weights[key], dtype=float).reshape(-1)
            total += R.T @ (chi[:, None] * R)
        return float(np.linalg.norm(total - np.eye(self.n_global), np.inf))

    def identity_norm(self) -> float:
        """|| sum_i R_i^T chi_i R_i ||_2.

        **This was `norm_A` until 2026-08-27 and it was vacuous.** For any
        partition of unity the sum IS the identity, so this returns 1.0 whenever
        the identity residual is zero -- it can only report a number when the
        thing it is checking has already failed.  Kept, renamed, as a companion
        to ``identity_residual``.
        """
        total = np.zeros((self.n_global, self.n_global))
        for key, R in self.restrictions.items():
            R = np.asarray(R, dtype=float)
            chi = np.asarray(self.weights[key], dtype=float).reshape(-1)
            total += R.T @ (chi[:, None] * R)
        return float(np.linalg.norm(total, 2))

    def norm_A(self) -> float:
        """||A||: the assembly operator's gain on the tuple of LOCAL errors.

        A is the map ``(u_1, ..., u_N) -> sum_i R_i^T chi_i u_i``, and the
        constant the bound needs is its norm from the direct sum with the l2
        norm ``||e||^2 = sum_i ||e_i||^2`` into the global l2 norm.  Cauchy-Schwarz
        cellwise gives

            ||A|| = max_j sqrt( sum_i chi_{i,j}^2 )

        which is <= 1 for any partition of unity, with equality exactly where a
        cell has a single owner.  **So overlap makes assembly a contraction**, and
        ||A|| < 1 is the assembly layer's own contribution to the L <= 1 branch --
        a statement the vacuous version could never make.

        The direct-sum norm is the convention, and it is stated because it is a
        choice: under ``max_i ||e_i||`` the same operator has norm 1 identically.

        **Measured 2026-08-27, and the answer is 1 by construction.** The maximum
        is attained at any cell with a single owner, where chi = 1 -- and every
        decomposition anybody builds has an interior.  So ``||A|| = 1`` for any
        partition of unity, this is a theorem rather than a measurement, and the
        master bound's ||A|| factor is not a free constant.  That closes it as an
        *unmeasured* quantity and leaves the assembly slot's real hole where it
        already was: ``condition``, and the blend defect it would constrain.
        """
        acc = np.zeros(self.n_global)
        for key, R in self.restrictions.items():
            R = np.asarray(R, dtype=float)
            chi = np.asarray(self.weights[key], dtype=float).reshape(-1)
            acc += (R.T @ chi) ** 2
        return float(np.sqrt(acc.max()))

    def contaminated_weight(self) -> float | None:
        """Pi = max_j sum_i chi_ij 1[cell j is contaminated for subdomain i].

        **W49's missing term.** `master-error-bound` section 4 bounds sigma
        through an interface SOLVE, where an operator mismatch is amplified by
        1/beta.  A halo scheme has no interface solve, and applied to one that
        bound overestimates by 4.6e7.  What actually carries the transmission
        error in a halo scheme is the weight the assembly gives to cells a stale
        artificial-boundary datum can have reached:

            sigma  <=  C_mu * Pi * ||d_lambda||

        Pi is 1 when the overlap is narrower than the agents' domain of
        dependence -- nothing is clean -- and 0 when the partition of unity
        vanishes over the whole contaminated band.  It is read off the
        declaration: chi, the stencil radius and the exchange interval.

        Measured over 11 configurations of the four-window tiling, the implied
        C_mu stayed in [0.297, 1.178] while sigma moved by 4.3e4.  See
        ``C_MU_HALO`` and `scripts/w49_sigma_halo.py`.
        """
        if self.contaminated is None:
            return None
        acc = np.zeros(self.n_global)
        for key, R in self.restrictions.items():
            R = np.asarray(R, dtype=float)
            chi = np.asarray(self.weights[key], dtype=float).reshape(-1)
            bad = np.asarray(self.contaminated[key], dtype=bool).reshape(-1)
            acc += R.T @ (chi * bad)
        return float(acc.max())

    def chi_min(self) -> float:
        """min_ij chi_ij: the whole of L6/C1's hypothesis, in one number.

        Negative is the refusal (R11).  It is not a tolerance question -- the
        partitions this package builds normalize to a sum of one and return
        exactly 0.0, and an extrapolatory one returns a number of order one.
        """
        return float(min(np.asarray(w, dtype=float).min() for w in self.weights.values()))

    def blend_variance(self, locals_: dict[str, np.ndarray]) -> np.ndarray:
        """V_chi = sum_i chi_i u_i^2 - (sum_i chi_i u_i)^2, cellwise.

        The margin by which the blend beats the chi-weighted average of what it
        blends, by the bias-variance identity in this module's docstring.  It
        needs no reference solution, which is what the slot's ``must_satisfy``
        demands, and it is non-negative for all data iff chi >= 0.
        """
        sq = np.zeros(self.n_global)
        for key, R in self.restrictions.items():
            R = np.asarray(R, dtype=float)
            chi = np.asarray(self.weights[key], dtype=float).reshape(-1)
            u = np.asarray(locals_[key], dtype=float).reshape(-1)
            sq += R.T @ (chi * u ** 2)
        return sq - self.assemble(locals_) ** 2

    def hull_escape(self, locals_: dict[str, np.ndarray]) -> np.ndarray:
        """max(A(u) - max_i u_i, min_i u_i - A(u)): how far outside the hull.

        Zero for a convex partition at every cell, by definition of a convex
        combination.  This is the DATA-side witness of the same condition
        ``chi_min`` states on the declaration side; both are reported because a
        disagreement between them would indict the implementation.
        """
        blend = self.assemble(locals_)
        lo = np.full(self.n_global, np.inf)
        hi = np.full(self.n_global, -np.inf)
        for key, R in self.restrictions.items():
            R = np.asarray(R, dtype=float)
            u = np.asarray(locals_[key], dtype=float).reshape(-1)
            sup = np.asarray(R.sum(axis=0) > 0)
            lifted = R.T @ u
            lo = np.where(sup, np.minimum(lo, lifted), lo)
            hi = np.where(sup, np.maximum(hi, lifted), hi)
        covered = np.isfinite(lo)
        esc = np.maximum(blend - hi, lo - blend)
        return np.where(covered, esc, 0.0)

    def assemble(self, locals_: dict[str, np.ndarray]) -> np.ndarray:
        out = np.zeros(self.n_global)
        for key, R in self.restrictions.items():
            R = np.asarray(R, dtype=float)
            chi = np.asarray(self.weights[key], dtype=float).reshape(-1)
            out += R.T @ (chi * np.asarray(locals_[key], dtype=float))
        return out

    def subdomains(self):
        return list(self.restrictions)

    def lift(self, key: str, values: np.ndarray) -> np.ndarray:
        """R_i^T u_i: one subdomain's field placed on the global grid, unweighted."""
        R = np.asarray(self.restrictions[key], dtype=float)
        return R.T @ np.asarray(values, dtype=float)

    def support(self, key: str) -> np.ndarray:
        R = np.asarray(self.restrictions[key], dtype=float)
        return np.asarray(R.sum(axis=0) > 0)


@dataclass
class GridPartitionOfUnity:
    """The same object for grids too large to write R_i as a dense matrix.

    R_i on a structured grid is a selection, so ``sum_i R_i^T chi_i R_i`` is
    *diagonal* and every quantity the certificate needs is an O(n) reduction
    rather than an n-by-n product.  The dense form above is kept because it is
    the definition; this is the one a real case study can afford -- a 255x255
    field would need a 65025-square matrix, which is 34 GB.

    ``indices[i]`` are the flat global indices subdomain i covers and
    ``weights[i]`` is chi_i on them.
    """

    n_global: int
    indices: dict[str, np.ndarray]
    weights: dict[str, np.ndarray]
    kind: str = "partition-of-unity"
    contaminated: dict[str, np.ndarray] | None = None

    def _accumulate(self, power: int = 1) -> np.ndarray:
        acc = np.zeros(self.n_global)
        for key, idx in self.indices.items():
            chi = np.asarray(self.weights[key], dtype=float).reshape(-1)
            np.add.at(acc, np.asarray(idx).reshape(-1), chi ** power)
        return acc

    def identity_residual(self) -> float:
        return float(np.abs(self._accumulate(1) - 1.0).max())

    def identity_norm(self) -> float:
        return float(np.abs(self._accumulate(1)).max())

    def norm_A(self) -> float:
        """max_j sqrt(sum_i chi_ij^2) -- see PartitionOfUnity.norm_A."""
        return float(np.sqrt(self._accumulate(2).max()))

    def contaminated_weight(self) -> float | None:
        """Pi -- see PartitionOfUnity.contaminated_weight."""
        if self.contaminated is None:
            return None
        acc = np.zeros(self.n_global)
        for key, idx in self.indices.items():
            chi = np.asarray(self.weights[key], dtype=float).reshape(-1)
            bad = np.asarray(self.contaminated[key], dtype=bool).reshape(-1)
            np.add.at(acc, np.asarray(idx).reshape(-1), chi * bad)
        return float(acc.max())

    def chi_min(self) -> float:
        return float(min(np.asarray(w, dtype=float).min() for w in self.weights.values()))

    def blend_variance(self, locals_: dict[str, np.ndarray]) -> np.ndarray:
        sq = np.zeros(self.n_global)
        for key, idx in self.indices.items():
            chi = np.asarray(self.weights[key], dtype=float).reshape(-1)
            u = np.asarray(locals_[key], dtype=float).reshape(-1)
            np.add.at(sq, np.asarray(idx).reshape(-1), chi * u ** 2)
        return sq - self.assemble(locals_) ** 2

    def hull_escape(self, locals_: dict[str, np.ndarray]) -> np.ndarray:
        blend = self.assemble(locals_)
        lo = np.full(self.n_global, np.inf)
        hi = np.full(self.n_global, -np.inf)
        for key, idx in self.indices.items():
            u = np.asarray(locals_[key], dtype=float).reshape(-1)
            flat = np.asarray(idx).reshape(-1)
            np.minimum.at(lo, flat, u)
            np.maximum.at(hi, flat, u)
        covered = np.isfinite(lo)
        esc = np.maximum(blend - hi, lo - blend)
        return np.where(covered, esc, 0.0)

    def assemble(self, locals_: dict[str, np.ndarray]) -> np.ndarray:
        out = np.zeros(self.n_global)
        for key, idx in self.indices.items():
            chi = np.asarray(self.weights[key], dtype=float).reshape(-1)
            np.add.at(out, np.asarray(idx).reshape(-1),
                      chi * np.asarray(locals_[key], dtype=float).reshape(-1))
        return out

    def subdomains(self):
        return list(self.indices)

    def lift(self, key: str, values: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_global)
        np.add.at(out, np.asarray(self.indices[key]).reshape(-1),
                  np.asarray(values, dtype=float).reshape(-1))
        return out

    def support(self, key: str) -> np.ndarray:
        out = np.zeros(self.n_global, dtype=bool)
        out[np.asarray(self.indices[key]).reshape(-1)] = True
        return out


#: L6/C1, stated once so the compiler, the certificate and the hole all quote the
#: same words rather than three paraphrases of them.
L6_C1 = (
    "L6/C1 -- the assembly accuracy condition. A partition-of-unity blend is at "
    "least as accurate as the local solves it blends, at every cell and in every "
    "l^p norm, if and only if the weights are a CONVEX partition: chi_ij >= 0 and "
    "sum_i chi_ij = 1. Proof, cellwise: with sum_i chi_i = 1 the identity "
    "|A(u)-u*|^2 = sum_i chi_i |u_i-u*|^2 - V_chi holds for any weights, where "
    "V_chi = sum_i chi_i u_i^2 - (sum_i chi_i u_i)^2 is the chi-weighted variance "
    "of the local VALUES and needs no exact solution. V_chi >= 0 for all data iff "
    "chi >= 0, and then A(u) is a convex combination, so |A(u)-u*| <= max_i "
    "|u_i-u*| pointwise. Both halves are checkable without the exact solution: "
    "chi_min from the declaration, V_chi and the hull escape from the run."
)


@dataclass
class AssemblyCondition:
    """L6/C1 evaluated: the hypothesis, its witnesses, and whether it holds.

    ``chi_min`` and ``identity_residual`` are the hypothesis, read off the
    declaration.  ``variance_margin`` and ``hull_escape`` are the data-side
    witnesses of the same statement, present only when local solves were
    supplied; they cannot fail while the hypothesis holds, and a disagreement
    between the two sides indicts this module rather than the case study.
    """

    chi_min: float | None = None
    identity_residual: float | None = None
    norm_A: float | None = None
    variance_margin: float | None = None        # min_j V_chi(j)
    hull_escape: float | None = None            # max_j distance outside the hull
    #: W49: the weight the assembly gives to cells a stale artificial-boundary
    #: datum can have reached. It is not part of L6/C1 -- a partition with Pi = 1
    #: is still admissible, just inaccurate -- but it is a property of chi, so it
    #: is measured here and consumed by the master bound's overlapping branch.
    contaminated_weight: float | None = None
    statement: str = L6_C1

    @property
    def convex(self) -> bool | None:
        if self.chi_min is None:
            return None
        return self.chi_min >= -CONVEX_TOL

    @property
    def identity(self) -> bool | None:
        if self.identity_residual is None:
            return None
        return self.identity_residual <= POU_TOL

    @property
    def holds(self) -> bool | None:
        """True only when both halves are affirmatively checked."""
        if self.convex is None or self.identity is None:
            return None
        return bool(self.convex and self.identity)

    def why(self) -> str:
        if self.holds is None:
            return "no assembly declared, so L6/C1 has nothing to check"
        if not self.identity:
            return f"the partition-of-unity identity fails at {self.identity_residual:.3e}"
        if not self.convex:
            return (f"the partition is not convex: min chi = {self.chi_min:.4g}. The "
                    "blend can leave the hull of the values it blends, and then no "
                    "bound relates it to them")
        return (f"convex (min chi = {self.chi_min:.4g}) and the identity holds "
                f"({self.identity_residual:.3e}), so the blend is at least as accurate "
                "as the local solves it blends at every cell")

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": "L6/C1",
            "holds": self.holds,
            "convex": self.convex,
            "identity": self.identity,
            "chi_min": self.chi_min,
            "identity_residual": self.identity_residual,
            "norm_A": self.norm_A,
            "variance_margin": self.variance_margin,
            "hull_escape": self.hull_escape,
            "contaminated_weight_W49": self.contaminated_weight,
            "why": self.why(),
            "statement": self.statement,
        }


@dataclass
class AssemblyCertificate:
    """The four fields of the slot. ``condition`` is L6/C1 since 2026-08-28."""

    pou_residual: float | None = None
    norm_A: float | Unmeasured = field(
        default_factory=lambda: UNMEASURED_CONSTANTS["norm_A"]
    )
    blend_defect: float | None = None
    condition: AssemblyCondition | None = None
    #: The definition `blend_defect` used until 2026-08-28, kept because it is
    #: what the first measurement reported.  It compares a norm of the blend's
    #: error against the MAX OF THE LOCAL NORMS, and convexity does not bound
    #: that: max-of-norms is smaller than norm-of-cellwise-max.  Measured, it is
    #: negative for a deliberately non-convex partition too, so it never
    #: discriminated.
    blend_defect_max_of_norms: float | None = None
    variance_margin: float | None = None
    kind: str = "partition-of-unity"
    note: str = field(
        default=(
            "condition is L6/C1 (2026-08-28): chi >= 0 with sum_i chi_i = 1, which is "
            "necessary and sufficient for the blend to be at least as accurate as the "
            "local solves it blends. It is checkable from the declaration with no "
            "reference solution, which is what the slot required. blend_defect is the "
            "reference-side diagnostic and is now measured against the cellwise max, "
            "which is the quantity convexity actually bounds."
        )
    )

    @property
    def identity_holds(self) -> bool | None:
        if self.pou_residual is None:
            return None
        return self.pou_residual <= POU_TOL

    @property
    def condition_holds(self) -> bool | None:
        return None if self.condition is None else self.condition.holds

    def as_dict(self) -> dict[str, Any]:
        from .verdict import _jsonable

        return {
            "kind": self.kind,
            "pou_residual": self.pou_residual,
            "identity_holds": self.identity_holds,
            "norm_A": _jsonable(self.norm_A),
            "blend_defect": self.blend_defect,
            "blend_defect_max_of_norms": self.blend_defect_max_of_norms,
            "variance_margin": self.variance_margin,
            "condition": None if self.condition is None else self.condition.as_dict(),
            "condition_holds": self.condition_holds,
            "hole": ASSEMBLY_CERTIFICATE.name,
            "note": self.note,
        }


def certify(
    pou: PartitionOfUnity | None,
    locals_: dict[str, np.ndarray] | None = None,
    reference: np.ndarray | None = None,
    overlap_mask: np.ndarray | None = None,
) -> AssemblyCertificate:
    """Emit the certificate, ``condition`` included.

    L6/C1 is evaluated from ``pou`` alone -- that is the point of it, and it is
    why the certificate can be issued at compile time on a graph that has not
    run.  ``locals_`` and ``reference`` add the data-side witnesses: the variance
    margin (no reference needed) and the blend defect (reference needed, and a
    diagnostic rather than a condition).

    **``blend_defect`` changed definition on 2026-08-28** to the cellwise max,
    which is the quantity convexity bounds.  The old max-of-norms number is kept
    under its own name because it is what the first measurement reported, and
    because the reason it was replaced is that it never discriminated: measured,
    it is negative for a deliberately non-convex partition as well.
    """
    if pou is None:
        return AssemblyCertificate(condition=AssemblyCondition())
    cert = AssemblyCertificate(
        pou_residual=pou.identity_residual(),
        norm_A=pou.norm_A(),
        kind=pou.kind,
    )
    cond = AssemblyCondition(
        chi_min=pou.chi_min(),
        identity_residual=cert.pou_residual,
        norm_A=float(cert.norm_A),
        contaminated_weight=pou.contaminated_weight(),
    )
    if locals_ is not None:
        v = pou.blend_variance(locals_)
        esc = pou.hull_escape(locals_)
        mask = (np.ones_like(v, dtype=bool) if overlap_mask is None
                else np.asarray(overlap_mask, bool))
        cond.variance_margin = float(v[mask].min()) if mask.any() else None
        cond.hull_escape = float(esc[mask].max()) if mask.any() else None
        cert.variance_margin = cond.variance_margin
    if locals_ is not None and reference is not None:
        cert.blend_defect = blend_defect(pou, locals_, reference, overlap_mask)
        cert.blend_defect_max_of_norms = blend_defect(
            pou, locals_, reference, overlap_mask, against="max-of-norms")
    cert.condition = cond
    return cert


def blend_variance(pou: PartitionOfUnity, locals_: dict[str, np.ndarray]) -> np.ndarray:
    """V_chi cellwise -- see the module docstring and ``PartitionOfUnity``."""
    return pou.blend_variance(locals_)


def blend_defect(
    pou: PartitionOfUnity,
    locals_: dict[str, np.ndarray],
    reference: np.ndarray,
    overlap_mask: np.ndarray | None = None,
    against: str = "cellwise-max",
) -> float:
    """alpha := ||A(u) - u*|| - ||m||, with m the CELLWISE max local error.

    Positive alpha means the blend is worse than the worst thing it blended at
    the cells where it blended it, and L6/C1 says that cannot happen for a
    convex partition -- ``|A(u)-u*|_j <= m_j`` holds at every cell, so it holds
    in every norm.

    ``against="max-of-norms"`` restores the definition used until 2026-08-28,
    which compared against ``max_i ||u_i - u*||``.  That is a different and
    smaller quantity, convexity does not bound it, and measured on the four-window
    tiling it came out negative for every partition tried including a
    deliberately non-convex one -- so it separated nothing.
    """
    ref = np.asarray(reference, dtype=float)
    mask = (np.ones_like(ref, dtype=bool) if overlap_mask is None
            else np.asarray(overlap_mask, bool))
    blended = pou.assemble(locals_)
    blended_err = float(np.linalg.norm((blended - ref)[mask]))

    if against == "max-of-norms":
        worst = 0.0
        for key in pou.subdomains():
            lifted = pou.lift(key, locals_[key])
            sel = mask & pou.support(key)
            if sel.any():
                worst = max(worst, float(np.linalg.norm((lifted - ref)[sel])))
        return blended_err - worst

    cellmax = np.zeros_like(ref)
    for key in pou.subdomains():
        sup = pou.support(key)
        err = np.abs(pou.lift(key, locals_[key]) - ref)
        cellmax = np.where(sup, np.maximum(cellmax, err), cellmax)
    return blended_err - float(np.linalg.norm(cellmax[mask]))


def sigma_halo_bound(
    pou: PartitionOfUnity,
    dlambda: float,
    C_mu: float = C_MU_HALO,
) -> float | None:
    """W49: sigma <= C_mu * Pi * ||d_lambda||, the OVERLAPPING branch.

    `master-error-bound` section 4's ``C_mu/beta * ||Lambda - Lambda~|| *
    ||lambda*||`` is the SUBSTRUCTURING branch: it routes the transmission error
    through an interface solve, and the 1/beta is that solve's amplifier.  A halo
    scheme poses no interface problem, so applied to one the factorization is not
    conservative but uninformative -- measured, it overestimates by 4.6e7.

    What carries the transmission error here instead is the stale
    artificial-boundary datum, whose reach is bounded by the agents' domain of
    dependence over one exchange interval and whose contribution to the assembled
    field is weighted by chi.  Both are in ``Pi``.

    Returns None when the partition does not declare its contaminated cells --
    the bound decertifies rather than assuming the assembly is clean, which is the
    same discipline the halo rule already applies to an undeclared overlap.
    """
    pi = pou.contaminated_weight()
    if pi is None:
        return None
    return float(C_mu) * pi * float(dlambda)
