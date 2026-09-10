"""Closure, substitution, and the depth that every defect must carry.

plug-in-composition-theorems §1 and §2.  A plug-in architecture asserts three
things it had never stated: that a composed subsystem is itself a pluggable
expert (closure), that swapping one expert does not require re-certifying the
system (substitution), and that a capability record is true (conformance, which
lives in ``conformance.py``).

**Closure.**  A connected subgraph with its internal ports matched presents a
valid capability record, its transmission operator being the Schur complement of
the internal block -- so nesting is not a new mechanism, it is stopping the
existing elimination early.  Legal exactly when the internal interface problem is
nonsingular, and grouping-invariant by the quotient property, so a hierarchy and
the flat graph give the same composed operator and a build need not commit to a
nesting depth.

The record composes in 8 of 13 fields, fails in 3 -- ``L_native``, ``regime_law``
and ``validity``, all *operating-point* rather than *interface* fields -- and
propagates the SeamReference hole in 2.

**The attribution theorem, and it is the uncomfortable one.**  The master bound at
one step with zero initial error gives

    tau_composite  <=  tau_internal + sigma_internal + gamma_internal

so a subassembly's transmission and solve infidelity become the *agent* infidelity
of the expert one level up.  They are not removed by nesting and not
double-counted -- they are relabelled.  Therefore the three-way split is not
invariant under regrouping, and **every emitted defect must carry the depth it was
measured at**, or a nested run's numbers cannot be compared with a flat run's.

**Substitution.**  A swap moves the conditioning by at most the norm of the block
difference (Weyl), giving a certificate computable from two probes and no
rollout.  And there is no monotonicity theorem: a strictly better expert can
produce a strictly worse composition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .capability import BCChannel, ClaimType, ExpertCapabilities, PortDecl
from .holes import SEAM_REFERENCE
from .verdict import ADMIT, ADMIT_UNCERTIFIED, REFUSE, FailureClass, Verdict


class CompositionRefused(RuntimeError):
    """The subgraph does not present a legal expert."""


#: Field-by-field, from the theorem. The value is the composition rule or the
#: reason there is none. This table is the answer to "is the framework closed?".
CLOSURE_TABLE: dict[str, tuple[str, str]] = {
    "ports": ("closes", "the unmatched ports of the subgraph; internal ports are eliminated"),
    "transmission": ("closes", "the Schur complement of the internal block, given a "
                               "nonsingular internal interface problem"),
    "storage": ("closes", "power-preserving interconnection of passive parts is passive; "
                          "in operator form the Schur complement of a positive-real block "
                          "with respect to a positive-real principal block is positive real"),
    "dt_native": ("closes", "the max over constituents, by R4: a composite is as slow as "
                            "its slowest constituent"),
    "bc_channel": ("closes", "determined at each exposed port by the boundary agent there, "
                             "liftable by R2: a composite can always be probed"),
    "differentiable": ("closes", "the weakest of the constituents; a composite JVP costs "
                                 "one back-solve with the already-factored internal block"),
    "equivariances": ("closes", "the subgroup of the intersection that is ALSO a graph "
                                "automorphism of the subgraph -- a symmetry of the parts is "
                                "a symmetry of the whole only if it maps the internal "
                                "wiring to itself"),
    "claim_types": ("closes", "intersection"),
    "L_native": ("does not compose", "constituents with different window sizes leave the "
                                     "composite with no single value"),
    "regime_law": ("does not compose", "the composite's regime is a TUPLE of constituent "
                                       "regimes; scalarizing it is a modelling choice, not "
                                       "a derivation"),
    "validity": ("does not compose usably", "the conjunction is correct and is not "
                                            "evaluable before the internal solve, so a "
                                            "composite declares deferred and raises a "
                                            "declination from inside its own step"),
    "governing_family": ("propagates a hole", "the common value if all constituents agree; "
                                              "otherwise composite, and the SeamReference "
                                              "slot climbs the hierarchy"),
    "lambda_ref": ("propagates a hole", "requires a reference operator for the composite"),
}


@dataclass
class CompositeExpert:
    """A subassembly presented as a single expert, with what it could not inherit."""

    composite_id: str
    members: tuple[str, ...]
    capabilities: ExpertCapabilities
    schur: np.ndarray
    beta_int: float
    exposed: tuple[str, ...]
    internal: tuple[str, ...]
    depth: int
    must_redeclare: tuple[str, ...] = ("L_native", "regime_law", "validity")
    propagated_holes: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "composite_id": self.composite_id,
            "members": list(self.members),
            "depth": self.depth,
            "beta_int": self.beta_int,
            "exposed_dofs": list(self.exposed),
            "internal_dofs": list(self.internal),
            "must_redeclare": list(self.must_redeclare),
            "propagated_holes": list(self.propagated_holes),
            "notes": list(self.notes),
        }


def schur_complement(
    S: np.ndarray,
    exposed_idx: Sequence[int],
    internal_idx: Sequence[int],
    beta_int_floor: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """The composite's transmission operator, and the internal conditioning.

        Lambda_composite = S_EE - S_EI S_II^{-1} S_IE

    Schur complementation is transitive, so a hierarchy of nested composites and
    the flat graph produce the same composed interface operator: grouping is free
    to choose, and can be chosen for cost.

    **``beta_int_floor`` was called ``beta_min`` until 2026-08-30 and it is NOT
    the substitution certificate's ``beta_min`` (W81).**  It is a singularity
    guard on the internal block -- the point below which the subgraph does not
    present a legal expert at all -- and 1e-12 is the right default for that,
    because it is asking whether a matrix is invertible.  The certificate's
    ``beta_min`` is a *risk tolerance* on a different quantity, and the sibling
    default was the whole reason the row said "every seam in the vault is blind":
    two numbers with one name, one of which has a machine-epsilon default that
    means something entirely reasonable where it lives.
    """
    S = np.asarray(S, dtype=float)
    E = np.asarray(exposed_idx, dtype=int)
    I = np.asarray(internal_idx, dtype=int)
    if I.size == 0:
        return S[np.ix_(E, E)].copy(), float("inf")

    S_II = S[np.ix_(I, I)]
    svals = np.linalg.svd(S_II, compute_uv=False)
    beta_int = float(svals[-1]) if svals.size else 0.0
    if beta_int <= beta_int_floor:
        raise CompositionRefused(
            f"the subgraph's internal interface problem is singular (beta_int = "
            f"{beta_int:.3e}). A subassembly is a legal expert exactly when its own "
            "interface problem is well posed, and a badly conditioned internal interface "
            "is a bad plug twice: close to being an illegal expert at all, and amplifying "
            "everything above it"
        )
    S_EE = S[np.ix_(E, E)]
    S_EI = S[np.ix_(E, I)]
    S_IE = S[np.ix_(I, E)]
    return S_EE - S_EI @ np.linalg.solve(S_II, S_IE), beta_int


def compose(
    composite_id: str,
    members: Sequence[ExpertCapabilities],
    S: np.ndarray,
    exposed_idx: Sequence[int],
    internal_idx: Sequence[int],
    exposed_ports: Sequence[PortDecl],
    depth: int = 1,
    graph_automorphisms: Sequence[str] = (),
) -> CompositeExpert:
    """Present a connected, internally matched subgraph as one expert."""
    members = list(members)
    if not members:
        raise CompositionRefused("a composite needs at least one member")
    schur, beta_int = schur_complement(S, exposed_idx, internal_idx)

    families = {m.governing_family for m in members}
    propagated: list[str] = []
    if len(families) > 1:
        family: str | None = "composite"
        propagated.append(SEAM_REFERENCE.name)
    else:
        family = next(iter(families))

    dt_values = [m.dt_native for m in members if m.dt_native is not None]
    diff_rank = {"none": 0, "vjp": 1, "jvp": 1, "both": 2}
    weakest_diff = min(members, key=lambda m: diff_rank[m.differentiable.value]).differentiable

    # A symmetry of the parts is a symmetry of the whole only if it maps the
    # internal wiring to itself -- a condition no page states.
    shared = set(members[0].equivariances)
    for m in members[1:]:
        shared &= set(m.equivariances)
    equivariances = tuple(sorted(shared & set(graph_automorphisms))) if graph_automorphisms else ()

    claim_types = set(members[0].claim_types)
    for m in members[1:]:
        claim_types &= set(m.claim_types)

    caps = ExpertCapabilities(
        expert_id=composite_id,
        ports=list(exposed_ports),
        # A composite can always be probed, so R2 applies at the level above
        # exactly as below.
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=all(m.bc_time_varying for m in members),
        differentiable=weakest_diff,
        dt_native=max(dt_values) if dt_values else None,
        L_native=None,                 # does not compose; must be re-declared
        regime_law=None,               # does not compose; must be re-declared
        # H_composite = sum_i H_i: a power-preserving interconnection of passive
        # parts is passive, so the composite's storage is the sum of its parts'.
        storage=_composite_storage(members) if all(m.declares_storage for m in members) else None,
        equivariances=equivariances,
        validity=None,                 # deferred: evaluated at the internal solution
        governing_family=family,
        lambda_ref=None,
        claim_types=frozenset(claim_types),
        note="composite; validity is deferred and raised from inside its own step",
    )
    if family == "composite":
        caps.lambda_ref = None

    comp = CompositeExpert(
        composite_id=composite_id,
        members=tuple(m.expert_id for m in members),
        capabilities=caps,
        schur=schur,
        beta_int=beta_int,
        exposed=tuple(str(i) for i in exposed_idx),
        internal=tuple(str(i) for i in internal_idx),
        depth=depth,
        propagated_holes=tuple(propagated),
    )
    comp.notes.append(
        "the composite's own transmission and solve infidelity become its AGENT "
        "infidelity one level up; the three-way split is not invariant under regrouping, "
        "so every emitted defect carries its depth"
    )
    if not equivariances and graph_automorphisms:
        comp.notes.append(
            "no equivariance survived: the parts' shared symmetries are not graph "
            "automorphisms of the internal wiring"
        )
    return comp


def _composite_storage(members: Sequence[ExpertCapabilities]):
    """H_composite(u) = sum_i H_i(u_i), with u a mapping from member id to state."""
    parts = [(m.expert_id, m.storage) for m in members]

    def H(state: Any) -> float:
        total = 0.0
        for expert_id, fn in parts:
            if fn is None:
                continue
            local = state.get(expert_id) if isinstance(state, dict) else state
            total += float(fn(local))
        return total

    return H


def composite_tau_bound(
    tau_internal: float | None,
    sigma_internal: float | None,
    gamma_internal: float | None,
) -> float | None:
    """The attribution theorem: tau_C <= tau + sigma + gamma, internal, per step."""
    parts = [tau_internal, sigma_internal, gamma_internal]
    if any(p is None for p in parts):
        return None
    return float(sum(p for p in parts if p is not None))


#: **W81, closed 2026-08-30.**  What ``beta_min`` is, stated once, because the
#: framework never said and the row is that the omission is load-bearing.
#:
#: It is the smallest seam conditioning the composition is willing to operate at:
#: a *risk tolerance*, not a property of the seam.  Two things follow and both are
#: on `SubstitutionCertificate` below.
#:
#: **1. It has a derived candidate, and only one.**  ``eps_tol = min(tau, sigma)``
#: over the terms that carry a scale is already the interface tolerance the
#: compiler uses, and it is the only quantity in the framework with the right
#: units and the right meaning: the size of defect this composition already
#: accepts.  A swap that moves the seam by less than what the composition already
#: tolerates is not a change the certificate should be asked about.  W84's caveat
#: applies -- a composition of exact solvers has ``tau = 0`` and ``min`` degenerates
#: -- so the minimum is taken over the POSITIVE terms and is UNDEFINED when none
#: is positive, rather than collapsing to zero.
#:
#: **2. Where no tolerance is supplied, the certificate reports the two thresholds
#: instead of a verdict.**  ``blind`` and ``passes`` are both monotone in
#: ``beta_min``, so the whole verdict function is two numbers:
#:
#:     beta_min <  beta - ||S_i||       -> blind: no replacement of this agent
#:                                        could have failed, and a pass is empty
#:     beta_min >  beta - ||Delta||     -> refuse: this swap fails
#:     in between                       -> an informative pass
#:
#: The window is non-empty for every swap, since ``||Delta|| <= ||S_old|| +
#: ||S_new||`` and in particular a total failure of the agent has
#: ``||Delta|| = ||S_i||``.  So a certificate can always say *at what risk
#: tolerance this swap would be visible*, which is strictly more than a verdict at
#: a number nobody derived -- and it is what the row asks for on the branch where
#: beta_min stays a user parameter.
BETA_MIN_UNDERIVED = (
    "beta_min is a risk tolerance supplied by the caller: the smallest seam "
    "conditioning this composition is willing to operate at. Nothing in this "
    "framework derives it from the graph, and the certificate therefore reports "
    "the two thresholds that make it a verdict rather than assuming one. Pass "
    "eps_tol = min(tau, sigma) to use the interface tolerance the compile already "
    "carries (W81)"
)


def beta_min_from_tolerance(tau: float | None = None,
                            sigma: float | None = None,
                            terms: Sequence[float | None] = ()) -> tuple[float | None, str]:
    """**W81's derived candidate**: ``eps_tol = min(tau, sigma)`` over positive terms.

    Returns ``(value, source)``; ``value`` is ``None`` when no supplied term is a
    positive number, which is a legitimate outcome and not a zero.  **W84 is why
    the positivity filter is there**: ``tau = 0`` is a real measurement -- a
    composition of exact solvers has no agent infidelity -- and a plain ``min``
    over it sets the tolerance to zero, at which every certificate passes and
    every one of them is blind.  That is the 1e-12 sibling default reappearing
    from the other direction.
    """
    supplied = [("tau", tau), ("sigma", sigma)]
    supplied += [(f"term{i}", t) for i, t in enumerate(terms)]
    usable = [(k, float(v)) for k, v in supplied
              if v is not None and np.isfinite(v) and float(v) > 0.0]
    if not usable:
        zeroed = [k for k, v in supplied if v is not None and float(v) <= 0.0]
        return None, (
            "no defect term carries a positive scale"
            + (f" ({', '.join(zeroed)} measured at zero, which is W84's case)"
               if zeroed else "")
            + ", so eps_tol is UNDEFINED rather than zero: a tolerance of zero "
              "passes every swap and sees none of them"
        )
    key, val = min(usable, key=lambda kv: kv[1])
    return val, (f"derived: eps_tol = min over positive defect terms = {key} = "
                 f"{val:.6g}, the interface tolerance this composition already "
                 f"accepts (W81)")


@dataclass
class SubstitutionCertificate:
    """Two probes, no rollout. The plug-in guarantee stated as an inequality."""

    agent_id: str
    old_expert: str
    new_expert: str
    delta_norm: float
    beta: float
    #: **W81.**  Optional since 2026-08-30, and ``None`` is the honest value when
    #: no risk tolerance was supplied.  It does NOT default -- see
    #: `BETA_MIN_UNDERIVED` and `visible_above` / `fails_above`.
    beta_min: float | None
    same_port_list: bool
    passivity_preserved: bool | None = None
    #: ||S_i||_2, the swapped agent's own block on the seam. **W76, 2026-08-29.**
    #: Supplying it is what lets this certificate report its own blind spot; it
    #: is optional only so that pre-2026-08-29 callers keep working.
    block_norm: float | None = None
    #: **W81.**  Where `beta_min` came from: a caller's declaration, a derivation
    #: from the interface tolerance, or nowhere.
    beta_min_source: str = "supplied by the caller, underived"

    @property
    def margin(self) -> float | None:
        return None if self.beta_min is None else self.beta - self.beta_min

    @property
    def visible_above(self) -> float | None:
        """**W81.** The beta_min above which this certificate stops being blind.

        ``blind`` is ``||S_i|| < beta - beta_min``, so the test starts being
        informative at ``beta_min = beta - ||S_i||``.  Below that no replacement
        of this agent could fail, including one that ignores its boundary data
        entirely.  `None` when the block norm was not supplied, which is the same
        discipline `blind` keeps.
        """
        return None if self.block_norm is None else self.beta - self.block_norm

    @property
    def fails_above(self) -> float:
        """**W81.** The beta_min above which THIS swap is refused.

        ``passes`` is ``||Delta|| < beta - beta_min``, so it holds exactly below
        ``beta - ||Delta||``.  Together with `visible_above` this is the whole
        verdict function, and reporting the pair is what the certificate does
        when no tolerance is supplied.
        """
        return self.beta - self.delta_norm

    @property
    def informative_window(self) -> tuple[float, float] | None:
        """The beta_min interval on which this swap both passes and could have failed.

        Non-empty for any swap, because ``||Delta|| <= ||S_i||`` whenever the
        replacement's own block is no larger than the original's, and exactly
        degenerate for a total failure of the agent, where the two coincide.
        """
        lo = self.visible_above
        if lo is None:
            return None
        return (lo, self.fails_above)

    # -- what a verdict rests on: reported beside it, never charged against it --
    #
    # **W109 / W137 / W177, 2026-09-10 (Tier 41).**  Readings a certificate could
    # always have computed from its own fields and did not report.  Each was first
    # done by hand on a real seam -- W109's saturated refusals on the reuse probe,
    # W177's eight sides of the live checkpoint, W137's fluid side of `wing_fsi`
    # -- and none of them changes `verdict`.

    @property
    def decision_margin(self) -> float | None:
        """**W109.** ``(beta - beta_min) - ||Delta||``: how far this swap is from flipping.

        ``margin`` is the allowance; this is the swap measured against it --
        positive by how much it passes, negative by how much it is refused.  W109
        found `refuse` at 55 cells whose closest cell sat a quarter of one percent
        from flipping; the verdict cannot tell those apart and this can.  `None`
        when no beta_min was supplied, where `fails_above` is the same reading
        with the tolerance left free.
        """
        return None if self.margin is None else self.margin - self.delta_norm

    @property
    def decision_margin_over_beta(self) -> float | None:
        m = self.decision_margin
        if m is None or self.beta == 0.0:
            return None
        return m / self.beta

    @property
    def null_replacement_ratio(self) -> float | None:
        """**W76, measured by W177.** ``||Delta|| / ||S_i||``: how close the swap is to deleting the block.

        A replacement that ignores its boundary data removes ``S_i`` and nothing
        else, so ``||Delta|| = ||S_i||`` IS the null replacement.  On all eight
        agent-sides of `poseidon-t-2x2` this reads 0.95-1.01, so the three
        informative admits there admit an expert within 5% of absent.  A ratio
        near 1 refuses nothing -- an under-responding block can be harmless --
        but it says what an `admit` rests on.  `None` without a block norm.
        """
        if self.block_norm is None or self.block_norm <= 0.0:
            return None
        return self.delta_norm / self.block_norm

    @property
    def block_over_beta(self) -> float | None:
        """``||S_i|| / beta``: can a total failure of this agent be seen at all.

        `blind` is ``||S_i|| < beta - beta_min``, so below 1 this ratio names a
        band of tolerances in which no replacement of the agent could fail, and
        at or above 1 there is none.  It is the scale-free form of `visible_above`,
        which equals ``beta * (1 - block_over_beta)``.  It is NOT the block's share
        of the operator, ``||S_i|| / ||S||``: beta is the smallest singular value
        of the assembled operator and ``||S||`` its largest, so the two are
        different readings of one seam -- and at `wing_fsi`'s wetted seam they
        point opposite ways.  **W137, measured 2026-09-10**: under the declared
        (exact momentum exchange) effort the fluid's share is 1.44e-5, which W137
        read as blind by five orders, and ``||S_F|| / beta`` is 1.39, so a
        replacement that ignores its boundary data is REFUSED at every
        beta_min >= 0.  `None` without a block norm.
        """
        if self.block_norm is None or self.beta == 0.0:
            return None
        return self.block_norm / self.beta

    def _margin_clause(self) -> str:
        m = self.decision_margin
        if m is None:
            return ""
        rel = self.decision_margin_over_beta
        side = "passes" if m > 0.0 else "is refused"
        pct = "" if rel is None else f", {abs(rel):.3%} of beta"
        tail = ("" if self.null_replacement_ratio is None
                else f"; ||Delta||/||S_i|| = {self.null_replacement_ratio:.3g}")
        return f" [W109: this swap {side} by {abs(m):.4g}{pct}{tail}]"

    @property
    def blind(self) -> bool | None:
        """Could this test have failed at all?  **W76, measured 2026-08-29.**

        The largest move a swap of agent ``i`` can make is bounded by that
        agent's own block: a replacement that ignores its boundary data entirely
        removes ``S_i`` and nothing else, giving ``||Delta|| = ||S_i||``.  So if
        ``||S_i|| < beta - beta_min`` the inequality below cannot be violated by
        ANY replacement, and a pass carries no information.

        This is not a hypothetical.  On `window_ns` split-step's seam ``sx0``
        the downstream window is 16.7% of the assembled operator against the
        upstream window's 94.1%, and swapping it for an expert with no boundary
        dependence at all is certified ADMIT at every ``beta_min`` up to 0.20.
        The same swap at the balanced seam ``sy0`` is REFUSED from 0.10.

        **W81, 2026-08-30.** The framework used to derive no ``beta_min`` at all
        -- a caller's argument with no default here, beside a sibling default of
        1e-12 (the internal-block singularity guard, now renamed
        ``beta_int_floor``) at which every seam in the vault is blind.  It now
        derives one candidate, ``eps_tol = min(tau, sigma)``, and where none is
        supplied it reports `visible_above` and `fails_above` instead of a
        verdict.  A certificate that cannot fail is worse than no certificate,
        because it is reported as a pass.
        """
        if self.block_norm is None or self.margin is None:
            return None
        return bool(self.block_norm < self.margin)

    @property
    def passes(self) -> bool | None:
        """`None` when no tolerance was supplied: there is no verdict to give."""
        if self.margin is None:
            return None
        return self.same_port_list and self.delta_norm < self.margin

    @property
    def verdict(self) -> Verdict:
        if not self.same_port_list:
            return REFUSE
        if self.passes is None:
            # **W81.** No risk tolerance, so no verdict -- and reporting the
            # thresholds is a result rather than a failure to produce one.
            return ADMIT_UNCERTIFIED
        if self.passes:
            # A pass that could not have been a failure is not an admission.
            return ADMIT_UNCERTIFIED if self.blind else ADMIT
        if self.passivity_preserved:
            return ADMIT_UNCERTIFIED
        return REFUSE

    @property
    def message(self) -> str:
        if not self.same_port_list:
            return (
                "the replacement does not declare the same port list. That is a graph "
                "edit, not a substitution, and it re-opens the whole certification"
            )
        if self.passes is None:
            window = ""
            if self.visible_above is not None:
                window = (
                    f" This swap is INVISIBLE below beta_min = {self.visible_above:.4g} "
                    f"(= beta - ||S_i||, where no replacement of this agent could fail) "
                    f"and REFUSED above beta_min = {self.fails_above:.4g} "
                    f"(= beta - ||Delta||). Between them the pass is informative."
                )
            else:
                window = (
                    f" This swap is REFUSED above beta_min = {self.fails_above:.4g} "
                    f"(= beta - ||Delta|| = {self.beta:.4g} - {self.delta_norm:.4g}); "
                    "supply block_norm to learn the threshold below which it is blind."
                )
            return (
                "no beta_min was supplied, so this certificate reports the thresholds "
                "rather than a verdict." + window + " " + BETA_MIN_UNDERIVED
            )
        if self.passes and self.blind:
            return (
                f"||Delta|| = {self.delta_norm:.4g} < beta - beta_min = {self.margin:.4g}, "
                f"but this agent's whole block is only {self.block_norm:.4g}, which is "
                "also below the margin -- so NO replacement of it could have failed this "
                "test, including one that ignores its boundary data entirely. The pass "
                "carries no information about the swap. Raise beta_min above "
                f"{self.beta - self.block_norm:.4g} or certify this agent some other way"
            )
        if self.passes:
            return (
                f"||Delta|| = {self.delta_norm:.4g} < beta - beta_min = {self.margin:.4g}: "
                "the swap preserves admissibility without re-certifying the graph"
                + self._margin_clause()
            )
        base = (
            f"||Delta|| = {self.delta_norm:.4g} exceeds beta - beta_min = {self.margin:.4g}: "
            "the seam must be re-certified and sigma, L and the horizon re-derived. Note "
            "that a strictly better expert can fail this -- there is no monotonicity "
            "theorem, because the swap moves beta as well as tau and beta sits in the "
            "denominator of both sigma and L. No swap is accepted on the grounds that the "
            "new expert benchmarks better"
        ) + self._margin_clause()
        if self.passivity_preserved:
            return base + (
                ". Both experts are incrementally passive, so the L <= 1 branch survives "
                "the swap with no re-measurement: passivity is the only property in the "
                "framework with this closure"
            )
        return base

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent_id,
            "old": self.old_expert,
            "new": self.new_expert,
            "delta_norm": self.delta_norm,
            "beta": self.beta,
            "beta_min": self.beta_min,
            "beta_min_source": self.beta_min_source,
            "margin": self.margin,
            "visible_above": self.visible_above,
            "fails_above": self.fails_above,
            "informative_window": (None if self.informative_window is None
                                   else list(self.informative_window)),
            "passes": self.passes,
            "verdict": self.verdict.value,
            #: W109: the verdict's own distance from flipping, beside it
            "decision_margin": self.decision_margin,
            "decision_margin_over_beta": self.decision_margin_over_beta,
            #: W76/W177 and W137: what an admit rests on -- reported, not charged
            "null_replacement_ratio": self.null_replacement_ratio,
            "block_over_beta": self.block_over_beta,
            "passivity_preserved": self.passivity_preserved,
            "block_norm": self.block_norm,
            "blind": self.blind,
            "message": self.message,
        }


def certify_substitution(
    agent_id: str,
    old_caps: ExpertCapabilities,
    new_caps: ExpertCapabilities,
    S_old: np.ndarray,
    S_new: np.ndarray,
    beta: float,
    beta_min: float | None = None,
    passivity_old: float | None = None,
    passivity_new: float | None = None,
    block_norm: float | None = None,
    eps_tol: float | None = None,
    tau: float | None = None,
    sigma: float | None = None,
) -> SubstitutionCertificate:
    """Whether a swap preserves admissibility, from two probes of one agent.

    **W81.** ``beta_min`` is optional as of 2026-08-30 and has no default.  Three
    ways to supply it, in precedence order:

      * pass ``beta_min`` directly -- a declared risk tolerance;
      * pass ``eps_tol``, or ``tau`` and ``sigma``, and it is derived as the
        interface tolerance ``min(tau, sigma)`` over the terms that carry a
        positive scale (`beta_min_from_tolerance`);
      * pass none of them, and the certificate reports `visible_above` and
        `fails_above` -- the thresholds that make it a verdict -- rather than
        inventing one.

    The third is not a degraded mode.  It is the row's own second branch: where
    the number is a user's risk parameter, the honest output is the threshold
    above which the substitution becomes visible.
    """
    old_ports = sorted((p.port_type.value, p.name) for p in old_caps.ports)
    new_ports = sorted((p.port_type.value, p.name) for p in new_caps.ports)
    passivity_preserved = None
    if passivity_old is not None and passivity_new is not None:
        passivity_preserved = passivity_old <= 0.0 and passivity_new <= 0.0

    source = "supplied by the caller, underived"
    if beta_min is None:
        if eps_tol is not None and float(eps_tol) > 0.0:
            beta_min, source = float(eps_tol), (
                f"derived: eps_tol = {float(eps_tol):.6g}, the interface tolerance "
                "supplied by the compile (W81)")
        elif tau is not None or sigma is not None:
            beta_min, source = beta_min_from_tolerance(tau, sigma)
            if beta_min is None:
                source = "UNDEFINED -- " + source
        else:
            source = "not supplied: " + BETA_MIN_UNDERIVED

    return SubstitutionCertificate(
        agent_id=agent_id,
        old_expert=old_caps.expert_id,
        new_expert=new_caps.expert_id,
        delta_norm=float(np.linalg.norm(np.asarray(S_new) - np.asarray(S_old), 2)),
        beta=float(beta),
        beta_min=None if beta_min is None else float(beta_min),
        same_port_list=old_ports == new_ports,
        passivity_preserved=passivity_preserved,
        block_norm=None if block_norm is None else float(block_norm),
        beta_min_source=source,
    )


def beta_after_swap(beta: float, delta_norm: float) -> float:
    """Weyl's inequality for singular values: beta' >= beta - ||Delta||."""
    return float(beta - delta_norm)
