"""L9 -- the typing of claims, and the two horizons.

end-to-end-architecture-spec §11, plus plug-in-composition-theorems §4.2.

**The decision, already made and implemented here rather than re-litigated:**
claims are typed trajectory inside the predictability horizon, statistical
outside; the statistical type is declared and refused, never approximated; and a
run whose L is unmeasured produces ``UNTYPED``.

Under that rule every result currently in the vault types as UNTYPED, because L
has never been measured.  That appearance is accurate.  It is the true state of
the theory and hiding it does not change it.

Two things this module implements that appear on no page before the spec:

  * **The horizon has three branches**, read off the master bound's specialization
    table by direct arithmetic -- exponential at L > 1, linear at L = 1, unbounded
    at L < 1.  Establishing incremental passivity is not only how L <= 1 is
    obtained structurally; it is also what converts a short exponential horizon
    into a long linear one, so the passivity certificate and the claim type are
    the same investment.

  * **The horizon is a table, not a scalar.**  The tolerance sits inside the
    logarithm and tolerance is a property of the quantity being claimed, so a run
    has one L, one defect, and as many horizons as it has quoted quantities.
    (**[AI Inference]** on the design call; the arithmetic itself is certain.)
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any

from .holes import Unmeasured, is_measured


class ClaimTypeTag(enum.Enum):
    T_CLAIM = "T-CLAIM"       # certified by the master bound, inside the horizon
    S_CLAIM = "S-CLAIM"       # statistical: type reserved, estimator absent, refused
    UNTYPED = "UNTYPED"       # L unknown, so the horizon is unknown, so no type is decidable


class HorizonBranch(enum.Enum):
    EXPONENTIAL = "exponential"     # L > 1: short
    LINEAR = "linear"               # L = 1: long
    UNBOUNDED = "unbounded"         # L < 1: no horizon at all
    UNDECIDABLE = "undecidable"     # L unmeasured


class StatisticalClaimRefused(NotImplementedError):
    """Emitting an S-CLAIM is a refusal until four missing fields can be filled."""


#: The interface an S-CLAIM must present. Reserved, so that when the theory
#: arrives it plugs in rather than restructures. None of the four has a
#: construction in this vault.
S_CLAIM_INTERFACE = {
    "functional": "the functional of the invariant measure being claimed",
    "averaging_window": "the window, and its ratio to the decorrelation time",
    "estimator": "an estimator with a sampling error, separate from model error",
    "gate": "what value, at what confidence, constitutes agreement with a reference",
}


@dataclass
class QuantityClaim:
    """One quoted quantity, with the tolerance it is held to."""

    name: str
    delta_tol: float
    value: float | None = None
    units: str = ""


@dataclass
class Horizon:
    """One row of the per-claim horizon table."""

    quantity: str
    branch: HorizonBranch
    t_pred: float | None
    n_pred: float | None = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "quantity": self.quantity,
            "branch": self.branch.value,
            "T_pred": self.t_pred,
            "N_pred": self.n_pred,
            "note": self.note,
        }


def horizon_for(
    quantity: QuantityClaim,
    L: float | Unmeasured | None,
    per_step_defect: float | None,
    dt: float | None = None,
    e0: float = 0.0,
) -> Horizon:
    """The three branches, by direct arithmetic on the specialization table.

    ``per_step_defect`` is tau + sigma + gamma for one macro-step.  It is not
    defaulted: without it there is no horizon, and inventing one is the failure
    this layer exists to prevent.
    """
    if not is_measured(L):
        return Horizon(
            quantity.name,
            HorizonBranch.UNDECIDABLE,
            None,
            note="L is unmeasured (W1: one paired rollout), so no horizon is decidable",
        )
    Lv = float(L)
    if per_step_defect is None or per_step_defect <= 0.0:
        return Horizon(
            quantity.name,
            HorizonBranch.UNDECIDABLE,
            None,
            note="the per-step defect tau + sigma + gamma has not been measured (W3)",
        )

    if Lv > 1.0:
        # T_pred ~ (1/lambda_1) ln(delta_tol / (tau + sigma + gamma)), with
        # lambda_1 = ln(L) / dt the leading Lyapunov exponent implied by the fit.
        ratio = quantity.delta_tol / per_step_defect
        if ratio <= 1.0:
            return Horizon(
                quantity.name,
                HorizonBranch.EXPONENTIAL,
                0.0,
                note="the per-step defect already exceeds the tolerance: the claim is "
                "invalid at every step, not merely late ones",
            )
        n_pred = math.log(ratio) / math.log(Lv)
        t_pred = n_pred * dt if dt else None
        return Horizon(
            quantity.name,
            HorizonBranch.EXPONENTIAL,
            t_pred,
            n_pred,
            note="exponential and short; this is the only branch written down before "
            "the end-to-end spec",
        )

    if abs(Lv - 1.0) <= 1e-12:
        n_pred = (quantity.delta_tol - e0) / per_step_defect
        t_pred = n_pred * dt if dt else None
        return Horizon(
            quantity.name,
            HorizonBranch.LINEAR,
            t_pred,
            n_pred,
            note="linear and long; the bound grows linearly, never exponentially. "
            "Reaching this branch is what a passivity certificate buys",
        )

    steady = per_step_defect / (1.0 - Lv)
    if steady < quantity.delta_tol:
        return Horizon(
            quantity.name,
            HorizonBranch.UNBOUNDED,
            None,
            note=f"contractive: bounded uniformly in time at {steady:.3e} < tolerance",
        )
    return Horizon(
        quantity.name,
        HorizonBranch.UNBOUNDED,
        0.0,
        note=f"contractive but the steady bound {steady:.3e} exceeds the tolerance: the "
        "claim is invalid at every step, not just late ones",
    )


def abstention_horizon(
    K: int,
    p: float | Unmeasured | None,
    dt: float | None,
) -> float | None:
    """T_abs ~ dt / (K p) -- the second horizon.

    It degrades **linearly in library size** and the predictability horizon does
    not depend on K at all, so for a large enough library the binding limit on a
    rollout is the framework rather than the physics.

    **[AI Inference], and the caveats are not small.**  Independence is wrong:
    agents in the same flow go out of distribution together, so the real horizon
    is longer than this when regimes are correlated and shorter when a single
    upstream excursion cascades.  The K-dependence is the claim; the constant is
    not.  And p has never been measured for any expert, because no expert
    declares validity.
    """
    if not is_measured(p) or dt is None:
        return None
    pv = float(p)
    if pv <= 0.0 or K <= 0:
        return None
    return dt / (K * pv)


@dataclass
class TypedClaim:
    """What L9 emits: a type, a horizon table, and the reason for both."""

    tag: ClaimTypeTag
    horizons: list[Horizon] = field(default_factory=list)
    t_abs: float | None = None
    t_usable: float | None = None
    reason: str = ""
    depth: int = 0                 # which decomposition depth these defects were measured at
    tags: list[str] = field(default_factory=list)

    def horizon(self, quantity: str) -> Horizon | None:
        for h in self.horizons:
            if h.quantity == quantity:
                return h
        return None

    def may_quote(self, quantity: str, at_time: float) -> tuple[bool, str]:
        """The single behaviour L9 exists to prevent: a metric past its own horizon."""
        if self.tag is ClaimTypeTag.UNTYPED:
            return False, (
                "UNTYPED: L is unmeasured, so the horizon is unknown and no trajectory "
                "metric can be quoted with a stated validity"
            )
        h = self.horizon(quantity)
        if h is None:
            return False, f"no horizon was computed for {quantity!r}; it was never declared"
        if h.branch is HorizonBranch.UNBOUNDED and h.t_pred is None:
            return True, "contractive: bounded uniformly in time"
        if h.t_pred is None:
            return False, f"horizon undecidable for {quantity!r}: {h.note}"
        if at_time <= h.t_pred:
            return True, f"inside the horizon ({at_time:.4g} <= {h.t_pred:.4g})"
        return False, (
            f"t = {at_time:.4g} is past this quantity's own horizon {h.t_pred:.4g}. "
            "The claim types S-CLAIM and is refused for publication, not for computation: "
            "the rollout may run, its trajectory metrics may not be quoted"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.tag.value,
            "depth": self.depth,
            "reason": self.reason,
            "T_abs": self.t_abs,
            "T_usable": self.t_usable,
            "horizons": [h.as_dict() for h in self.horizons],
            "tags": list(self.tags),
        }


def type_claim(
    quantities: list[QuantityClaim],
    L: float | Unmeasured | None,
    per_step_defect: float | None,
    dt: float | None = None,
    K: int = 1,
    p_decline: float | Unmeasured | None = None,
    depth: int = 0,
    e0: float = 0.0,
) -> TypedClaim:
    """Type a run's claims and build its horizon table."""
    horizons = [horizon_for(q, L, per_step_defect, dt, e0) for q in quantities]
    t_abs = abstention_horizon(K, p_decline, dt)

    if not is_measured(L):
        claim = TypedClaim(
            ClaimTypeTag.UNTYPED,
            horizons,
            t_abs,
            None,
            reason=(
                "L is unmeasured, so the predictability horizon is unknown, so no type is "
                "decidable. Every metric carries the UNTYPED tag"
            ),
            depth=depth,
        )
        claim.tags.append("UNTYPED: L unmeasured (W1)")
        return claim

    finite = [h.t_pred for h in horizons if h.t_pred is not None]
    t_pred = min(finite) if finite else None
    t_usable = min([t for t in (t_pred, t_abs) if t is not None], default=None)
    claim = TypedClaim(
        ClaimTypeTag.T_CLAIM,
        horizons,
        t_abs,
        t_usable,
        reason="a trajectory claim, certified by the master bound inside each "
        "quantity's own horizon",
        depth=depth,
    )
    if t_abs is not None and t_pred is not None and t_abs < t_pred:
        claim.tags.append(
            f"the binding horizon is abstention, not chaos: T_abs {t_abs:.4g} < "
            f"T_pred {t_pred:.4g}. T_abs degrades linearly in K and T_pred does not "
            "depend on K at all"
        )
    return claim


def refuse_statistical_claim(functional: str) -> StatisticalClaimRefused:
    """An S-CLAIM is declared and refused, never approximated."""
    return StatisticalClaimRefused(
        f"S-CLAIM for {functional!r} is refused. The type is reserved and its interface is "
        f"declared -- {sorted(S_CLAIM_INTERFACE)} -- and none of the four has a "
        "construction in this vault. There is no page developing shadowing under "
        "hyperbolicity, no ergodic-average error estimate, no gate set for statistical "
        "claims, and no definition of what tau means when the target is a measure rather "
        "than a trajectory. This page decides the typing discipline, not the statistical "
        "theory."
    )
