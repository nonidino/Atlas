"""The coupling compiler -- nine layers on a seven-hypothesis envelope.

end-to-end-architecture-spec, executed.  Given a case graph of declared agent
capability records plus port connections, this module derives the coupling scheme
rather than debating it, decides the envelope, and returns one of three verdicts
with every refusal citing its rule.

The layers are **ordered by dependency, not by execution**.  L1-L4 are
compile-time and run once; L5-L7 configure the per-macro-step loop; L8-L9 run at
emit time.  Every refusal is decidable before the first expensive solve except
two -- the probed null space at L4 and the partition-of-unity identity at L6 --
and both of those are cheap.

The compile does **not** stop at the first refusal.  It runs every layer it can
and returns the complete decision record, because a refusal that reports only the
first thing it found is a refusal nobody can act on, and because the envelope
stamp must be complete whatever the verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .admissibility import (
    ConditionResult,
    achievable_rung,
    apply_results,
    check_connection,
)
from .assembly import (
    C_MU_HALO,
    CONVEX_TOL,
    AssemblyCertificate,
    PartitionOfUnity,
    certify,
)
from .capability import (
    EllipticSubsolve,
    TimeDiscretization,
    BCChannel,
    ExpertCapabilities,
    MotionClass,
    Transmission,
)
from .claims import QuantityClaim, TypedClaim, type_claim
from .emit import BoundTerms, HarnessParameters, RunArtifact
from .envelope import EnvelopeStamp, Hypothesis, Status
from .graph import (
    CUT_DEFECT_FORMS,
    CaseGraph,
    Connection,
    CycleEnumerationBudget,
    Decomposition,
    FluxMatching,
)
from .holes import (
    ASSEMBLY_CERTIFICATE,
    INTERFACE_MOTION,
    PORT_AMENDMENT,
    SEAM_REFERENCE,
    TOPOLOGY_EVENT,
    UNMEASURED_CONSTANTS,
    HoleLedger,
    is_measured,
)
from .ports import advec_passenger_pressure
from .probe import (OPERATOR_CONTENT_FLOOR, ProbeBudget, SeamOperator, assemble_seam,
                    cut_score)
from .routing import RouteResult, RoutingRefused, route
from .scheme import Accelerator, Budget, Levels, Ordering, RULES, Scheme
from .transfer import (
    InterfaceSpace,
    SeamTransfer,
    dim_M,
    identity_prolongation,
)
from .verdict import (
    ADMIT,
    ADMIT_UNCERTIFIED,
    REFUSE,
    DecisionRecord,
    FailureClass,
    Verdict,
)


@dataclass
class CompileResult:
    """Everything the compile produced, whatever its verdict."""

    case: str
    verdict: Verdict
    scheme: Scheme | None
    envelope: EnvelopeStamp
    decisions: DecisionRecord
    artifact: RunArtifact
    seam_operators: dict[str, SeamOperator] = field(default_factory=dict)
    transfers: dict[str, SeamTransfer] = field(default_factory=dict)
    routes: dict[str, RouteResult] = field(default_factory=dict)
    holes: HoleLedger = field(default_factory=HoleLedger)
    refused_claims: list[str] = field(default_factory=list)
    tau_undefined_seams: list[str] = field(default_factory=list)
    unmeasured: list[str] = field(default_factory=list)

    @property
    def runnable(self) -> bool:
        return self.verdict is not REFUSE

    def report(self) -> str:
        parts = [self.artifact.report()]
        if self.scheme is not None:
            parts += ["", self.scheme.report()]
        if self.refused_claims:
            parts += ["", "refused claims:"] + [f"  - {c}" for c in self.refused_claims]
        return "\n".join(parts)


def compile_scheme(
    graph: CaseGraph,
    budget: Budget | None = None,
    probe_budget: ProbeBudget | None = None,
    references: dict[str, ExpertCapabilities] | None = None,
    quantities: Sequence[QuantityClaim] = (),
    depth: int = 0,
    probe_state: str = "unspecified",
) -> CompileResult:
    """Compile a coupling scheme from declarations. The whole public entry point.

    A new case study calls this and nothing else.  There is no hook here for
    case-specific coupling code, and adding one would defeat the purpose: the
    difference between a framework and a co-simulation harness built once per
    problem is that this function's output is derived from the declarations.
    """
    budget = budget or Budget()
    probe_budget = probe_budget or ProbeBudget()
    record = DecisionRecord()
    stamp = EnvelopeStamp()
    holes = HoleLedger()
    ctx = _Context(
        graph=graph,
        budget=budget,
        probe_budget=probe_budget,
        references=references or {},
        record=record,
        stamp=stamp,
        holes=holes,
        probe_state=probe_state,
        depth=depth,
    )

    _l1_records(ctx)
    _decide_rung(ctx)
    _l1_5_routing(ctx)
    _l2_decomposition(ctx)
    _l3_connections(ctx)
    _l3_global_fields(ctx)
    _l4_transmission(ctx)
    scheme = _l5_l7_scheme(ctx)
    # R13 -- a directed cycle is a fixed-point problem, not a list of seams.
    # After the scheme, because the condition is about the scheme's iteration.
    _r13_directed_cycle(ctx, scheme)
    certificate = _l6_assembly(ctx)
    claim = _l9_claims(ctx, quantities)
    _unmeasured_backstop(ctx)
    artifact = _l8_emit(ctx, scheme, certificate, claim)

    return CompileResult(
        case=graph.name,
        verdict=record.verdict,
        scheme=scheme,
        envelope=stamp,
        decisions=record,
        artifact=artifact,
        seam_operators=ctx.operators,
        transfers=ctx.transfers,
        routes=ctx.routes,
        holes=holes,
        refused_claims=ctx.refused_claims,
        tau_undefined_seams=ctx.tau_undefined_seams,
        unmeasured=ctx.unmeasured(),
    )


# ---------------------------------------------------------------------------


def _unmeasured_backstop(ctx: _Context) -> None:
    """**W56** -- `admit` requires every constant the bound quotes to be measured.

    `unmeasured` was reported on every compile and gated nothing.  That was
    invisible for as long as no graph could reach `admit`: every one carried at
    least the `G5/W16` decertification, so the worst verdict was already
    `admit-uncertified` and a missing constant could not change it.  The moment
    L2/C2 let a graph through, the gap became reachable -- and the first thing it
    produced was an `admit` on a graph whose `unmeasured` still listed ``C_mu``.

    Each individual constant has, or should have, its own decision at the site
    that needs it -- `_w49_sigma_branch` for ``C_mu``, `_choose_tolerance` for
    ``tau`` and ``sigma``, `_l9_claims` for ``L``.  **This is the backstop, not
    the rule**: if one of those sites is ever added without its check, the
    verdict stops being `admit` here rather than silently publishing a bound with
    a constant nobody measured.  Three call sites have now been found missing one
    (W45, W52, W56) and the fourth should not need finding.
    """
    missing = ctx.unmeasured()
    if not missing:
        return
    named = ", ".join(missing)
    ctx.record.decertify(
        "L8", "W56",
        f"{len(missing)} bound constant{'s' if len(missing) > 1 else ''} "
        f"unmeasured on this graph ({named}). The verdict cannot be `admit`: a "
        "bound quoted with a constant nobody measured is the silent-wrongness "
        "class, and `holes.Unmeasured` exists so that using one raises rather "
        "than returning a plausible number. Declare them on MeasuredConstants",
        subject="<graph>", quantity=named,
        unmeasured=list(missing),
    )


@dataclass
class _Context:
    graph: CaseGraph
    budget: Budget
    probe_budget: ProbeBudget
    references: dict[str, ExpertCapabilities]
    record: DecisionRecord
    stamp: EnvelopeStamp
    holes: HoleLedger
    probe_state: str
    depth: int

    transfers: dict[str, SeamTransfer] = field(default_factory=dict)
    provisional: set[str] = field(default_factory=set)
    operators: dict[str, SeamOperator] = field(default_factory=dict)
    routes: dict[str, RouteResult] = field(default_factory=dict)
    refused_claims: list[str] = field(default_factory=list)
    tau_undefined_seams: list[str] = field(default_factory=list)
    admissible_seams: set[str] = field(default_factory=set)
    transmission: Transmission | None = None
    decomposition: Decomposition | None = None

    def unmeasured(self) -> list[str]:
        """The constants this compile still lacks, each with its worklist item.

        A compile that produced a probed operator has measured beta and no longer
        lists it; nothing else on this list is produced by a compile, because
        every other constant needs a rollout or a reference.
        """
        names = ["L", "sigma", "tau", "C_mu"]
        if not self.operators:
            names.insert(1, "beta")
        if not all(a.capabilities.declares_validity for a in self.graph.agents):
            names.append("p_decline")
        if self.graph.partition_of_unity is None:
            names.append("norm_A")
        # W45: a constant the graph declares as measured is measured. Before this
        # the list was hard-coded and consulted nothing, so a measurement could
        # never reach a compile.
        supplied = self.graph.measured.supplied() if self.graph.measured else set()
        names = [n for n in dict.fromkeys(names) if n not in supplied]
        return [f"{n} ({UNMEASURED_CONSTANTS[n].worklist_item})" for n in names]

    def measured_L(self):
        """The fitted L if the graph declares one, else the Unmeasured sentinel."""
        m = self.graph.measured
        if m is not None and m.L is not None:
            return float(m.L)
        return UNMEASURED_CONSTANTS["L"]


# --- L1 -- expert declaration and the capability record --------------------


def _l1_records(ctx: _Context) -> None:
    """Where hypotheses become checkable. L1 relies on nothing and enables E3, E7."""
    rec, graph = ctx.record, ctx.graph

    for agent in graph.agents:
        caps = agent.capabilities
        subject = agent.agent_id

        missing = caps.missing_fields()
        if missing:
            rec.refuse(
                "L1", "record",
                f"record for {caps.expert_id!r} is missing fields a downstream layer "
                f"needs: {missing}",
                subject=subject, failure_class=FailureClass.STRUCTURAL, missing=missing,
            )

        if not caps.declares_storage:
            rec.decertify(
                "L1", "E7",
                "storage: none, so the L <= 1 branch of the master bound is unavailable "
                "and L must be fitted (W1). A declared storage function is not only the "
                "structural route to L <= 1; it is the only certificate that survives an "
                "expert swap without re-earning, which makes it the enabling condition "
                "of plug-in itself",
                subject=subject, quantity="L",
            )

        if not caps.declares_validity:
            rec.decertify(
                "L1", "C8",
                "validity: none, so the run cannot abstain and every bound is vacuous "
                "wherever the expert is out of distribution, with nothing saying where "
                "that is",
                subject=subject, quantity="p (declination rate)",
            )

    # Spec 3.4: a decertification at every K, with K reported, rather than a
    # refusal at a threshold nobody has measured. The threshold needs the
    # per-expert declination rate p (W13), which also sets T_abs.
    without_validity = [a.agent_id for a in graph.agents if not a.capabilities.declares_validity]
    if without_validity:
        thr = UNMEASURED_CONSTANTS["K_ood_threshold"]
        rec.decertify(
            "L1", "3.4",
            f"{len(without_validity)} of {graph.K} agents declare no validity predicate, "
            f"so the run cannot abstain and the exposure grows with K={graph.K}: "
            "P(some agent OOD) = 1 - (1-p)^K. This decertifies at every K rather than "
            "refusing at a threshold, because the threshold is K* = ceil(log(1-eps) / "
            f"log(1-p)) and p is unmeasured ({thr})",
            subject="<graph>", quantity="K_ood",
        )

    _stamp_E3(ctx)


def _stamp_E3(ctx: _Context) -> None:
    """E3 by string comparison: do the two sides of a seam share a governing family?"""
    rec, graph, stamp = ctx.record, ctx.graph, ctx.stamp
    undeclared = [a.agent_id for a in graph.agents if a.capabilities.governing_family is None]
    cross = graph.cross_family_seams()

    if cross:
        for c in cross:
            fa = graph.agent(c.a[0]).capabilities.governing_family
            fb = graph.agent(c.b[0]).capabilities.governing_family
            has_ref = all(
                graph.agent(x).capabilities.lambda_ref is not None for x in (c.a[0], c.b[0])
            )
            stamp.fails(
                Hypothesis.E3,
                f"seam {c.seam_id}: {c.a[0]} declares {fa!r}, {c.b[0]} declares {fb!r}",
            )
            ctx.holes.activate(
                SEAM_REFERENCE, c.seam_id, governing_families=[fa, fb], lambda_ref=has_ref
            )
            # **Corrected 2026-08-29.** E3's truth value is unchanged -- the two
            # sides really do declare different families -- but its CONSEQUENCE
            # was wrong, and it was the standing reason no multiphysics graph
            # could be certified. Read what tau is measured as, in
            # `scripts/tier0_window_ns.py`: the composed step given the TRUE
            # trace, against a reference trajectory. Nothing in that mentions a
            # governing equation. Sharing a family is what lets two agents share
            # a MONOLITHIC reference; tau needs only a REFERENCE PAIR, and at a
            # multiphysics seam one is constructible from the agents themselves
            # by converging the interface inside the macro-step
            # (`multiphysics.tight_couple`). Measured on thermal_seam: a gas
            # surrogate carrying a known 5%, 25% and 100% conductivity error is
            # recovered as tau = 0.05, 0.25 and 1.00 exactly, per agent, with the
            # unswapped side identically zero. `lambda_ref` is the declaration
            # that such a pair exists; it has been on the record since the
            # end-to-end spec and was checked for presence and never consumed.
            measured_tau = (None if graph.measured is None else graph.measured.tau)
            if has_ref and measured_tau is not None:
                rec.admit(
                    "L1", "E3",
                    f"multiphysics seam: {c.a[0]} and {c.b[0]} declare different "
                    f"governing families ({fa!r} against {fb!r}), so E3 fails and "
                    "always will -- they share no monolithic reference. **That does "
                    f"not make tau undefined, and here it is measured: {measured_tau:.4g} "
                    f"at {graph.measured.probe_state!r}.** Both sides declare "
                    "lambda_ref, so a reference PAIR exists; the trajectory tau is "
                    "scored against is the tightly coupled solve of that pair, and the "
                    "defect is measured in interface power, the one unit both sides of "
                    "the bond share. E3's failure costs the monolithic reference and "
                    "nothing else -- attribution survives it",
                    subject=c.seam_id, quantity="tau",
                )
            elif has_ref:
                rec.decertify(
                    "L1", "E3",
                    f"multiphysics seam: {c.a[0]} and {c.b[0]} declare different "
                    f"governing families ({fa!r} against {fb!r}), so they share no "
                    "monolithic reference and E3 fails. **tau is NOT undefined here**: "
                    "both sides declare lambda_ref, so a reference PAIR exists and tau "
                    "is measurable against the tightly coupled trajectory -- see "
                    "`multiphysics.seam_defect_split`, which attributes per agent and "
                    "measures in interface power, the one unit both sides of the bond "
                    "share. What is wrong is that it has not been RUN: this seam "
                    "carries a measurable tau and no measurement. Declare the result "
                    "on the graph's MeasuredConstants and this clears",
                    subject=c.seam_id, quantity="tau",
                )
            else:
                ctx.tau_undefined_seams.append(c.seam_id)
                rec.decertify(
                    "L1", "E3",
                    f"multiphysics seam: {c.a[0]} and {c.b[0]} declare different "
                    f"governing families ({fa!r} against {fb!r}) and carry no "
                    "lambda_ref, so there is no reference trajectory either side can "
                    "be scored against and tau is emitted as UNDEFINED for both. The "
                    "composition still RUNS correctly -- probing never mentions a "
                    "governing equation, so the transmission layer is robust to this "
                    "failure. It is the error ATTRIBUTION that has no referent. Note "
                    "this is now a statement about the missing REFERENCE and not "
                    "about the differing families: declare lambda_ref on both sides "
                    "and tau becomes measurable without either family moving",
                    subject=c.seam_id, quantity="tau",
                )
        ctx.holes.measure(
            SEAM_REFERENCE, "tau_undefined_seams", list(ctx.tau_undefined_seams),
            note="the field-5 measurement: the seams where tau has no referent",
        )
    elif undeclared:
        stamp.fails(
            Hypothesis.E3,
            f"governing_family undeclared on {undeclared}; the check is a string "
            "comparison and there is no string to compare",
        )
    else:
        stamp.holds(Hypothesis.E3, "all seams join agents declaring the same governing family")


def _decide_rung(ctx: _Context) -> None:
    """R1 lifted by R2, decided from the records alone -- and therefore before L2.

    The decomposition axis follows the transmission choice, and the axis decides
    whether cross-points exist at all. Deciding the rung after L2 would let a
    compile check cross-points against the declared axis and then silently switch
    to the one that has them, which is the silent-wrongness class the layer exists
    to catch. Going to the better transmission operator INTRODUCES a difficulty the
    present scheme does not have, and this is where that trade is made visible.
    """
    caps = [a.capabilities for a in ctx.graph.agents]
    ctx.transmission = achievable_rung(*caps) if caps else Transmission.DIRICHLET

    # R2b (W46): probed-DtN is a boundary-value-problem construction, and an
    # explicitly stepped agent poses no boundary-value problem over a macro-step.
    # R2 lifts the rung from what the expert ACCEPTS; this gates it on what the
    # expert IS. Without the gate the compiler recommends a scheme measured to be
    # 8.9x worse than doing nothing, with every diagnostic reading healthy.
    if ctx.transmission is Transmission.PROBED_DTN:
        explicit = [c.expert_id for c in caps
                    if c.time_discretization is TimeDiscretization.EXPLICIT]
        unknown = [c.expert_id for c in caps
                   if c.time_discretization is TimeDiscretization.UNKNOWN]
        if explicit:
            ctx.transmission = Transmission.DIRICHLET
            # An ADMIT, not a refusal: the compiler corrected the rung, so nothing
            # is silently wrong -- which is the split rule the three verdicts turn
            # on. It would be a refusal only if the scheme insisted on probed-DtN
            # anyway, and it cannot, because the rung is decided here.
            ctx.record.admit(
                "L4", "R2b/W46",
                f"R2 would lift the rung to probed-DtN, and {len(explicit)} agents declare "
                "time_discretization=explicit. The flux-balance condition sum_i Lambda_i "
                "lambda = chi is the interface condition of a boundary-value problem; an "
                "explicit macro-step poses none, so the condition is a STEADY one and the "
                "true trace does not satisfy it. Measured: solving it exactly moved the "
                "trace 41x past the truth and made the composed step 8.9x worse than not "
                "solving it, while beta, kappa, the passivity spectrum and the probe "
                "residual all read healthy. The rung is held at dirichlet; use an "
                "overlapping halo, or give the agent an implicit macro-step",
                subject=", ".join(explicit),
            )
        elif unknown:
            # **W61, found 2026-08-28 by the first expert that declares UNKNOWN.**
            # This used to decertify and leave the rung lifted, so a graph whose
            # agents' time discretization is unknown got a RUNNABLE probed-DtN
            # scheme -- the construction §5 measured at 8.9x worse than doing
            # nothing, with beta, kappa, the passivity spectrum and the probe
            # residual all reading healthy. That is the silent-wrongness class by
            # the three-verdict split rule's own definition, and the message
            # ending "It is not assumed to" was describing behaviour the code did
            # not have: the rung stayed lifted, which is assuming it.
            #
            # The rung is now held down, as in the `explicit` branch -- but the
            # decertification STAYS, and the difference from that branch is the
            # point. There we KNOW probed-DtN is wrong, so correcting it is an
            # admit. Here we know only that it is unestablished, so correcting it
            # is the conservative choice and the claim is not certified either way.
            ctx.transmission = Transmission.DIRICHLET
            ctx.record.decertify(
                "L4", "R2b/W46",
                f"{len(unknown)} agents declare time_discretization=unknown, so whether "
                "the probed-DtN interface condition applies to them is undecided -- and "
                "an undecided premise is not a favourable one. The rung is held at "
                "dirichlet rather than lifted: lifting it would pick the scheme that is "
                "measured 8.9x worse than doing nothing whenever the agents turn out to "
                "be explicit, and every diagnostic would read healthy while it did. "
                "Declare time_discretization to lift the decertification",
                subject=", ".join(unknown), quantity="sigma",
            )

    ctx.decomposition = (
        Decomposition.NON_OVERLAPPING
        if ctx.transmission is Transmission.PROBED_DTN
        else ctx.graph.decomposition
    )
    if ctx.decomposition is not ctx.graph.decomposition:
        ctx.record.admit(
            "L2", "R2/axis",
            f"the case declares a {ctx.graph.decomposition.value} decomposition, and R2 "
            f"lifts the achievable rung to {ctx.transmission.value}, which requires the "
            "non-overlapping view. The axis is switched here, BEFORE the cross-point "
            "check, because moving to the better transmission operator introduces a "
            "difficulty the overlapping scheme does not have",
            subject="<graph>",
        )


# --- L1.5 -- routing, the layer between L2 and L1 --------------------------


def _l1_5_routing(ctx: _Context) -> None:
    """Static routing, ranked by composability. The only mode admitted today."""
    rec, graph = ctx.record, ctx.graph
    for agent in graph.agents:
        if not agent.candidates:
            continue
        required = tuple({p.port_type for p in agent.capabilities.ports})
        try:
            result = route(agent.agent_id, agent.candidates, required)
        except RoutingRefused as exc:
            rec.refuse(
                "L1.5", "G15",
                str(exc), subject=agent.agent_id, failure_class=FailureClass.STRUCTURAL,
            )
            continue
        ctx.routes[agent.agent_id] = result
        if result.chosen is None:
            rec.decertify(
                "L1.5", "G15", result.note, subject=agent.agent_id, quantity="Xi",
            )
        else:
            rec.admit(
                "L1.5", "G15",
                f"routed to {result.chosen.expert_id}: {result.note}",
                subject=agent.agent_id,
            )


# --- L2 -- decomposition and cut placement ---------------------------------


def _l2_decomposition(ctx: _Context) -> None:
    rec, graph, stamp = ctx.record, ctx.graph, ctx.stamp

    # E1 -- is the graph fixed for the whole rollout?
    if graph.topology_events:
        stamp.fails(Hypothesis.E1, f"{len(graph.topology_events)} topology events declared")
        for ev in graph.topology_events:
            ctx.holes.activate(TOPOLOGY_EVENT, ev.event_id, t=ev.t)
            missing = []
            if ev.state_map is None:
                missing.append("state_map")
            if ev.ledger is None:
                missing.append("ledger")
            if missing:
                rec.refuse(
                    "L2", "TopologyEvent",
                    f"event {ev.event_id!r} declared without {missing}. Every event is a "
                    "restart with nonzero initial error, so the initial term of the bound "
                    "reactivates; and what has NO rule is conservation across the event, "
                    "so the architecture requires the ledger the missing rule will "
                    "constrain",
                    subject=ev.event_id, quantity="conservation ledger",
                )
            else:
                rec.decertify(
                    "L2", "E1",
                    f"event {ev.event_id!r} carries a state map and a ledger. The ledger "
                    "is emitted with NO constraint on its values, because no rule says "
                    "what they should be",
                    subject=ev.event_id, quantity="ledger",
                )
                ctx.holes.measure(TOPOLOGY_EVENT, "ledger", ev.ledger, constrained_by=None)
                ctx.holes.measure(TOPOLOGY_EVENT, "injected_e0", ev.e0_reset, constrained_by=None)
    else:
        stamp.holds(Hypothesis.E1, "no topology event declared; the graph is fixed")

    # E2, first half -- interface motion.
    moving = graph.moving_ports()
    if moving:
        for agent_id, port in moving:
            ctx.holes.activate(INTERFACE_MOTION, f"{agent_id}.{port.name}",
                               motion_class=port.motion_class.value)
            rec.refuse(
                "L2", "InterfaceMotion",
                f"port {agent_id}.{port.name} declares motion_class="
                f"{port.motion_class.value}. A moving interface changes the interface "
                "operator every step and invalidates a cached S, which is the entire "
                "economic argument for probing. No rule exists, and a moving interface "
                "silently invalidating a cached operator is the silent-wrongness class",
                subject=f"{agent_id}.{port.name}", quantity="operator drift",
            )
        stamp.fails(Hypothesis.E2, f"{len(moving)} ports declare non-static motion")
    # the second half of E2 -- declared transfer per seam -- is stamped at L3.

    # R10 -- an embedded elliptic sub-solve is not decomposable.
    _r10_elliptic(ctx)
    # The halo rule: a lagged exchange must outrun the agent's own domain of dependence.
    _halo_rule(ctx)
    # R13 runs after the scheme, not here: whether a cycle's gain has to
    # contract depends on whether the scheme SWEEPS the loop, and the
    # accelerator is not decided until L5.

    # Cross-points. Conditional on the decomposition axis.
    cps = graph.detected_cross_points()
    axis = ctx.decomposition or graph.decomposition
    if axis is Decomposition.OVERLAPPING:
        if cps:
            rec.admit(
                "L2", "I2/G1",
                f"{len(cps)} candidate cross-points, and no action required: under an "
                "overlapping decomposition the partition of unity handles the vertex. "
                "This is why tiles meeting four-at-a-corner have caused no trouble -- "
                "and section 12.2 confirmed it geometrically: under an overlapping "
                "tiling there is no common cell at all, so there is nothing to be "
                "multi-valued about. " + CROSS_POINT_COUPLING_SCOPE,
                subject="<graph>", cross_points=[list(c) for c in cps],
                cross_point_coupling="out-of-scope (W64)",
            )
    else:
        untreated = [c for c in cps if not set(c) & set(graph.primal_cross_point_dofs)]
        if untreated and not graph.primal_cross_point_dofs:
            # W64, decided 2026-08-29. The refusal STANDS and its stated reason
            # is replaced, because section 12.2 measured the old one and it was
            # wrong in all three of its claims: beta does NOT collapse
            # (1.00x on a real four-window junction), the conditioning is NOT
            # degraded, and the smallest singular direction carries 0.0097 of its
            # weight at the cross-point, so it is not a cross-point direction
            # either. The defect a cross-point actually causes is that the shared
            # cell is MULTI-VALUED: each incident seam carries its own multipliers
            # over its own ring and truncates a different function to the same
            # modes, so the cell gets one value per seam. Measured: 1.62% of the
            # trace scale in the streamwise component and 66.7% in the transverse
            # one, disagreeing about the sign.
            #
            # That quantity is decidable from the DECLARATION -- a cell in two
            # seams under a non-overlapping axis is multi-valued, full stop -- so
            # this refusal now rests on something the compiler can see, which the
            # old one did not. See `_cross_point_scope` for what is deliberately
            # NOT claimed.
            rec.refuse(
                "L2", "I2/G1",
                f"non-overlapping decomposition with {len(untreated)} cross-points and no "
                "primal set"
                + _cross_point_provenance(graph)
                + ". The shared cell is MULTI-VALUED: each incident seam carries "
                "its own multipliers over its own ring and truncates a different function "
                "to the same interface modes, so the cell receives one value per seam and "
                "they disagree -- measured at 1.62% of the trace scale in the streamwise "
                "component and 66.7% in the transverse one, with the sign disagreeing. "
                "The treatment is to make those degrees of freedom primal -- single-valued, "
                "in a small coarse problem -- and keep the rest dual, which is exactly what "
                "removes a multi-valued cell. It is a property of the DECLARED INTERFACE "
                "SPACE and not of the expert, so no better probe removes it. This refusal "
                "does NOT rest on beta: beta does not collapse at a real cross-point "
                "(measured 1.00x), and the off-diagonal coupling it would take a "
                "multi-port response to see is 1.8% of the operator norm and is not "
                "certified here at all (W64)",
                subject="<graph>", quantity="cross_point_multivaluedness",
                cross_points=[list(c) for c in untreated],
                measured_transverse_spread=0.667,
                measured_beta_collapse=1.00,
                measured_offdiagonal_fraction=0.018,
            )
        elif cps:
            rec.admit(
                "L2", "I2/G1",
                f"{len(cps)} cross-points, with primal degrees of freedom declared. "
                + CROSS_POINT_COUPLING_SCOPE,
                subject="<graph>", cross_point_coupling="out-of-scope (W64)",
            )

    _cut_policy(ctx)


def _cross_point_provenance(graph) -> str:
    """W162: say whether the cross-points were DECLARED or inferred, and how.

    The refusal below is severe -- it is a `refuse`, not a decertification --
    and until 2026-09-09 it did not say where its subject came from. On a tiling
    the adjacency triangle is a sound proxy for a shared corner. On a CIRCUIT it
    is not: three legs joined by three distinct planes are pairwise adjacent and
    share no point, and `cooling_loop.build(n_legs=3)` was refused for a
    multi-valued cell that does not exist. A rule that cannot be told it is
    wrong about its own subject is the failure this sentence exists to prevent.
    """
    if graph.cross_points_declared:
        return (
            ", from the graph's own `cross_points` declaration. This is not a "
            "proxy: the graph named these vertices"
        )
    return (
        ", inferred from the ADJACENCY and not declared. **This is a PROXY and "
        "it can be wrong here**: `detected_cross_points` reads every "
        "mutually-adjacent triple of agents as a shared vertex, which holds on "
        "a tiling -- three subdomains meeting pairwise really do meet at a "
        "corner -- and fails on a CIRCUIT, where three legs joined by three "
        "DISTINCT planes are pairwise adjacent and share no point at all. The "
        "triangle is then in the flow topology rather than in the geometry, and "
        "nothing in the adjacency tells the two apart. If this graph has no "
        "geometric cross-point, declare `cross_points=()` -- which since W162 "
        "(2026-09-09) is a declaration of NONE rather than silence -- and this "
        "refusal does not arise"
    )


#: **W64, decided 2026-08-29: cross-point COUPLING is out of scope, permanently.**
#:
#: The declared probe interface is ``BoundaryResponse = (port, trace) -> flux``:
#: one port in, *that same port's* flux out.  A cross-point's coupling is the
#: off-diagonal block ``d(flux on face B)/d(trace on face A)`` for two faces of
#: ONE agent, and no sequence of single-port calls produces it, because each call
#: restarts from the base state.  A per-seam assembly is therefore block-diagonal
#: by construction, and ``beta_global == min_s beta_s`` for a reason that has
#: nothing to do with the physics.
#:
#: **The record is NOT extended with a multi-port response, and this is the
#: decision rather than a deferral.**  What such a response would buy was
#: measured out-of-band first (`scripts/w6_cross_point.py`, section 12.2), and it
#: buys a diagnostic that changes no verdict:
#:
#:   * the off-diagonal coupling is real and **1.8%** of the global operator norm;
#:   * ``beta`` does **not** collapse -- 1.00x on a real four-window junction --
#:     so the quantity the old refusal cited is not affected by the block it
#:     could not see;
#:   * the defect a cross-point actually causes is **multi-valuedness** of the
#:     shared cell, which is a property of the declared interface space and is
#:     fixed by a primal (single-valued) corner degree of freedom -- a
#:     DECLARATION -- not by any measurement, however good the probe.
#:
#: A ``PortAmendment``-sized widening of the one interface every expert in the
#: library must implement, to obtain a 1.8% number that no rule consumes, is a
#: cost with no verdict attached to it.  A frozen checkpoint would have to
#: emulate it in its wrapper, which is where the emulation would then live
#: unvalidated.
#:
#: **What this costs, stated so it is never quietly forgotten.**  Cross-point
#: coupling is *diagnosed out of band and never certified*.  No envelope
#: hypothesis covers it, no bound constant measures it, and a graph that reaches
#: ``admit`` with a cross-point has said nothing whatever about the off-diagonal
#: block.  If a future measurement finds a graph where that block is not small,
#: this decision is the thing to reopen -- and `scripts/w6_cross_point.py` is
#: kept, runnable, for exactly that.
CROSS_POINT_COUPLING_SCOPE = (
    "cross-point coupling is diagnosed out-of-band (scripts/w6_cross_point.py) and "
    "never certified: the declared boundary_response cannot express an off-diagonal "
    "block, the measured coupling is 1.8% of the operator norm, and beta does not "
    "collapse. L2 refuses an untreated cross-point on MULTI-VALUEDNESS, which is "
    "decidable from the declaration, not on beta, which is not affected (W64)"
)


def _cut_policy(ctx: _Context) -> None:
    """**L2/C2** -- the decomposition criterion, derived 2026-08-28.

    This used to be an unconditional decertification citing `G5/W16`: *"cut where
    the exact operator is closest to local and the conditioning is comfortable"*,
    adopted as policy with its **[AI Inference]** status preserved.  It was the
    last thing between `window_ns` in `split-step` mode and `admit`, and it was a
    theory-status item rather than a missing measurement.

    What replaced it, and the two halves are not the same verdict:

    **Derived.**  ``D_i = E_i R_i - R_i E`` is the restriction defect, and the
    composed defect over one exchange interval is exactly ``sum_i R_i^T chi_i
    D_i``.  With L6/C1's convexity that is bounded cellwise by ``sum_i chi_i
    |D_i|``.  So the criterion is: **minimize the chi-weighted restriction
    defect** -- a theorem on hypotheses R11 already enforces, and the definition
    the phrase "closest to local" never had.

    **Falsified.**  The scalarization ``Q = (1/beta) ||S - diag S|| / ||S||``
    ranks cuts backwards (rank correlation -0.853 over twelve placements), is not
    a function of the decomposition at all (a 361x orbit under re-declaring the
    same interface space in another orthonormal frame), and is constant to 8e-11
    where the measured defect spreads 2.29x.  ``cut_score`` is retired as a
    criterion and scoped to the substructuring branch; see its docstring.

    **The verdict this rule issues.**  The bound is a theorem, so its
    *applicability* is checked from the declaration -- an overlapping axis and a
    convex partition of unity -- and its *value* is a measurement, ingested from
    the record exactly as L, tau, sigma and C_mu are (W45).  A graph that
    declares it gets an `admit`; one that does not gets a decertification naming
    it, which is a missing measurement and no longer an unearned inference.  A
    **non-overlapping** graph decertifies for a different reason and it is the
    honest one: the substructuring branch's cut criterion is still underived, and
    the quantity Q was reaching for lives there.  That is **W57**.
    """
    rec, graph = ctx.record, ctx.graph
    axis = ctx.decomposition or graph.decomposition
    declared = (graph.measured.cut_defect_bound
                if graph.measured is not None else None)

    if axis is not Decomposition.OVERLAPPING:
        # **L2/C3, derived and measured 2026-08-29 (W57).** The substructuring
        # branch has its own criterion now, and it is NOT L2/C2 re-scoped:
        # L2/C2's hypotheses are a partition of unity and an assembly, which
        # substructuring does not have. It comes from `master-error-bound` 4's
        # own factorization, stopped one step earlier than section 4 stops it.
        # With lam_dag solving S_M lam = chi_M on the declared interface space
        # and a* the exact trace's coordinates there,
        #
        #     S_M (lam_dag - a*) = chi_M - S_M a* =: -r
        #     ||lam_dag - a*|| <= ||r|| / beta            (an identity, then one bound)
        #
        # so Q_sub(Gamma) = ||S_M a* - chi_M|| / beta. Measured over 11
        # configurations of `tests/substructure_model.py`: ranks +0.579 to +1.000
        # against the true composed defect (POSITIVE in 11/11), tight to a factor
        # of 1.7, and following it costs at most 1.105x the best available cut.
        #
        # Two things this does NOT say, both measured:
        #   * section 4's own PRODUCT form, ||Lambda - Lambda~|| ||lam*|| / beta,
        #     ranks NEGATIVE in 9 of 11. The loose form is not merely loose, it
        #     is anti-correlated, and it is the shape the falsified Q had.
        #   * 1/beta helps HERE and inverted the ranking on the overlapping
        #     branch (section 10.2). Section 4's box said why and this is the
        #     measurement of it: q_tight beats the bare residual in 11/11.
        value = getattr(graph.measured, "cut_defect_bound", None)             if graph.measured is not None else None
        if value is None:
            rec.decertify(
                "L2", "C3/W57",
                "substructuring: the cut criterion L2/C3 = ||S_M a* - chi_M|| / beta "
                "is derived (master-error-bound 4's factorization, stopped before the "
                "operator-mismatch bound) and measured to rank cut placements at +0.579 "
                "to +1.000 over 11 configurations, tight to a factor of 1.7. Its VALUE "
                "is not declared on this graph, so the placement is unranked here. Note "
                "section 4's own product form ranks NEGATIVE in 9 of 11 and must not be "
                "used as a criterion; and the retired Q, rank -0.853, was that form's "
                "shape imported onto the overlapping branch",
                subject="<graph>", quantity="cut placement",
            )
        elif (getattr(graph.measured, "cut_defect_bound_form", None)
              not in (None, "substructuring-residual")):
            # **W58 on this branch.** L2/C2's three forms are quantities of the
            # overlapping branch and none of them is L2/C3's residual; a value
            # carried across is a number for a criterion it does not belong to.
            rec.decertify(
                "L2", "C3/W58",
                f"cut_defect_bound is declared at {float(value):.6g} with form "
                f"{graph.measured.cut_defect_bound_form!r}, which is one of L2/C2's "
                "OVERLAPPING-branch quantities, and this graph is non-overlapping. "
                "L2/C3 is ||S_M a* - chi_M|| / beta -- a different criterion on "
                "different hypotheses -- so declare cut_defect_bound_form as "
                "'substructuring-residual' or measure the right quantity",
                subject="<graph>", quantity="cut_defect_bound_form",
            )
        else:
            rec.admit(
                "L2", "C3",
                f"substructuring cut criterion L2/C3 declared: {float(value):.6g}. "
                "It is the residual the exact trace leaves in the approximate "
                "interface equation, amplified by 1/beta -- which is the amplifier "
                "on this branch by derivation, because substructuring has the "
                "interface solve section 4's factorization is about",
                subject="<graph>", quantity="cut_defect_bound",
                cut_defect_bound=float(value), branch="substructuring",
            )
        return

    pou = graph.partition_of_unity
    if pou is None:
        rec.decertify(
            "L2", "C2",
            "L2/C2 bounds the composed defect by the chi-weighted restriction defect "
            "sum_i chi_i |E_i R_i - R_i E|, and the graph declares no partition of "
            "unity, so there are no chi to weight with. The bound is not evaluable "
            "and the cut is uncharacterized",
            subject="<graph>", quantity="cut placement",
        )
        return

    # L2/C2 rests on L6/C1 -- the same convexity R11 enforces one layer up. It is
    # re-checked rather than assumed because the two rules are in different layers
    # and a graph can reach L2 before L6 has run.
    chi_min = None
    try:
        chi_min = float(pou.chi_min())
    except Exception:                                              # noqa: BLE001
        chi_min = None
    if chi_min is not None and chi_min < -CONVEX_TOL:
        rec.decertify(
            "L2", "C2",
            f"the partition of unity is not convex (min chi = {chi_min:.3e}), and "
            "L2/C2's bound is exactly the step convexity licenses. R11 refuses this "
            "graph at L6; the cut criterion is unavailable here for the same reason",
            subject="<graph>", quantity="cut placement",
        )
        return

    # **W58, closed 2026-08-30.** The disjointness condition that makes the
    # reference-free surrogate equal to the max form is a statement about the
    # `contaminated` geometry and nothing else, so it is decided here, from the
    # declaration W49's Pi already needs, with no solve of any kind. It is
    # DISCLOSED on every graph that declares the geometry and only ESCALATES to a
    # decertification where it changes what a declared number means -- which is
    # exactly when the reference-free form is the one that was measured.
    mult = None
    try:
        mult = pou.contaminated_multiplicity()
    except Exception:                                              # noqa: BLE001
        mult = None
    if mult is not None:
        rec.admit(
            "L2", "C2/W58",
            ("disclosure: " + mult["note"] + ". The chi-weighted and max forms of "
             "cut_defect_bound both need a monolithic reference and are unaffected by "
             "this; what it decides is whether the REFERENCE-FREE neighbour "
             "disagreement may be quoted as the bound"),
            subject="<graph>", quantity="cut_defect_bound_form", **mult,
        )

    if declared is None:
        rec.decertify(
            "L2", "C2",
            "L2/C2 applies -- overlapping axis, convex partition of unity -- but the "
            "graph declares no measured cut_defect_bound, so the criterion is a "
            "theorem with no value in it. Measure ||sum_i chi_i |E_i R_i u - R_i E u|||"
            " over one exchange interval and declare it on MeasuredConstants. This is "
            "a missing measurement, not an unearned inference",
            subject="<graph>", quantity="cut_defect_bound",
        )
        return

    # **W58's other half: the number without its form is unreadable.** The two
    # definitions differ by whether a monolith was run, and on the branch where
    # they are not provably equal that difference is the whole content of the
    # number. A value with no form is refused the admission rather than given it.
    form = getattr(graph.measured, "cut_defect_bound_form", None)
    if form is None:
        rec.decertify(
            "L2", "C2/W58",
            f"cut_defect_bound is declared at {declared:.4e} and the record does not "
            "say which of two inequivalent quantities it is: the chi-weighted "
            "restriction defect ||sum_i chi_i |D_i|||, which needs a monolithic "
            "reference exactly as tau does, or the reference-free neighbour "
            "disagreement, which does not. They are provably equal only when the "
            "agents' contaminated sets are pairwise disjoint. Declare "
            "cut_defect_bound_form as one of "
            + ", ".join(sorted(CUT_DEFECT_FORMS)),
            subject="<graph>", quantity="cut_defect_bound_form",
            cut_defect_bound=declared,
        )
        return
    if form not in CUT_DEFECT_FORMS:
        rec.decertify(
            "L2", "C2/W58",
            f"cut_defect_bound_form is {form!r}, which names no quantity this "
            "framework defines. The legal values are "
            + ", ".join(sorted(CUT_DEFECT_FORMS)),
            subject="<graph>", quantity="cut_defect_bound_form",
        )
        return
    if form == "substructuring-residual":
        rec.decertify(
            "L2", "C2/W58",
            "cut_defect_bound_form is the substructuring residual L2/C3, and this "
            "graph is OVERLAPPING, where L2/C2 applies. The two criteria are on "
            "different branches with different hypotheses -- L2/C2 needs a partition "
            "of unity and an assembly, which substructuring has neither of -- and a "
            "number from one is not a number for the other",
            subject="<graph>", quantity="cut_defect_bound_form",
        )
        return

    # The reference-free form is the bound only under the disjointness hypothesis,
    # so here and only here the geometry above changes the verdict.
    if form.startswith("neighbour-disagreement"):
        # **W58's second hypothesis, and it is the one that scales.** The
        # pairwise form is a MAX OVER PAIRS of a norm; L2/C2's bound is a norm
        # over the whole grid of a cellwise max. Those coincide when there is at
        # most one overlapping pair -- which is decidable here, from the
        # subdomain count -- and diverge on any larger tiling whatever the
        # contaminated sets do, because a max over pairs saturates as the tiling
        # grows while the bound accumulates.
        n_sub = len(pou.subdomains())
        if form == "neighbour-disagreement" and n_sub > 2:
            rec.decertify(
                "L2", "C2/W58",
                f"cut_defect_bound is declared at {declared:.4e} in the PAIRWISE "
                f"reference-free form on a {n_sub}-subdomain partition. That form "
                "is a max over neighbour pairs of a norm and L2/C2's bound is a "
                "norm over the whole grid of a cellwise max: they coincide when "
                "there is at most one overlapping pair and diverge above it, "
                "because a max over pairs saturates as the tiling grows while the "
                "bound accumulates. Declare "
                "'neighbour-disagreement-aggregated' instead "
                "(probe.aggregated_neighbour_disagreement) -- the same "
                "measurement aggregated the bound's way, at no extra cost",
                subject="<graph>", quantity="cut_defect_bound_form",
                cut_defect_bound=declared, n_subdomains=n_sub,
            )
            return
        if mult is None:
            rec.decertify(
                "L2", "C2/W58",
                f"cut_defect_bound is declared at {declared:.4e} in the "
                "REFERENCE-FREE form, which equals the bound only where the agents' "
                "contaminated sets are pairwise disjoint -- and the partition of unity "
                "declares no contaminated cells, so that hypothesis is undecidable "
                "here. It is decidable from geometry alone; declare `contaminated`, "
                "which W49's Pi needs anyway",
                subject="<graph>", quantity="cut_defect_bound_form",
            )
            return
        if not mult["disjoint"]:
            rec.decertify(
                "L2", "C2/W58",
                f"cut_defect_bound is declared at {declared:.4e} in the "
                f"REFERENCE-FREE form, and {mult['shared_cells']} of "
                f"{mult['contaminated_cells']} contaminated cells belong to more than "
                f"one subdomain (up to {mult['max_multiplicity']} at once). More than "
                "one D_i is nonzero there, so the neighbour disagreement is a "
                "SURROGATE for L2/C2's bound and not the bound. Measured on the "
                "four-window tiling, whose contaminated sets share 1120 cells, the two "
                "agreed to 1.00007 anyway -- which is a measurement, not a theorem, and "
                "does not transfer to a tiling with more interfaces. Either measure the "
                "chi-weighted form against a monolith or quote this as a surrogate",
                subject="<graph>", quantity="cut_defect_bound_form",
                cut_defect_bound=declared, **mult,
            )
            return

    rec.admit(
        "L2", "C2",
        f"cut criterion applies and is measured: the restriction defect over one "
        f"exchange interval is {declared:.4e}, in the {form} form -- "
        f"{CUT_DEFECT_FORMS[form]} -- bounding the composed defect cellwise by the "
        f"identity A({{E_i R_i u}}) - E u = sum_i R_i^T chi_i D_i together with "
        f"chi >= 0 (L6/C1). Derived, not adopted"
        + (". The reference-free form is quoted, and the contaminated sets are "
           "disjoint, so it IS the max form of the bound rather than a surrogate "
           "for it (W58)" if form.startswith("neighbour-disagreement") else ""),
        subject="<graph>",
        cut_defect_bound=declared,
        cut_defect_bound_form=form,
        provenance=(graph.measured.probe_state, graph.measured.scheme,
                    graph.measured.depth),
    )


# --- L3 -- the port algebra, and what makes a connection admissible --------


def _l3_connections(ctx: _Context) -> None:
    rec, graph, stamp = ctx.record, ctx.graph, ctx.stamp
    if not graph.connections:
        rec.decertify("L3", "graph", "no connections declared: this is not a composition",
                      subject="<graph>")
        # **W113, 2026-09-01.** This branch used to stamp E2 `unchecked`, and
        # `emit.validate` refuses that outright: E2 is one of the four static
        # hypotheses `envelope.illegally_unchecked` says are always decidable
        # from a declaration, so an unchecked E2 is *"a defect in the compiler,
        # not a property of the case"*.  The two rules contradicted each other
        # and a graph with no connections could not emit an artifact at all --
        # latent until `cases/thermal_strain.py`'s `global-field` route, the
        # first graph in this vault with zero seams.
        #
        # `holds`, and it is vacuous rather than generous.  E2 reads *"interfaces
        # are static, and transfer between the two sides is declared"*: over an
        # empty set of interfaces both clauses are true, and the L3
        # decertification above already says the thing that matters, which is
        # that nothing here is coupled.  E7 keeps `unchecked` because passivity
        # is a property of a probed block and `can_be_unchecked` is True for it.
        stamp.holds(Hypothesis.E2, "no seams: E2 holds vacuously over an empty "
                                   "interface set, and L3/graph carries the finding")
        stamp.unchecked(Hypothesis.E7, "no seams to check")
        return

    all_conforming = True
    any_transfer_missing = False
    e7_ok = True

    for conn in graph.connections:
        transfer = graph.seam_transfer(conn)
        if transfer is None and getattr(conn, "derive_space", False):
            transfer = _derive_transfer(ctx, conn)
        if transfer is not None:
            ctx.transfers[conn.seam_id] = transfer
            all_conforming &= transfer.conforming
        else:
            any_transfer_missing = True

        results = check_connection(graph, conn, transfer, ctx.budget.target_rung)
        worst = apply_results(rec, results, subject=conn.seam_id)
        if worst is not REFUSE:
            ctx.admissible_seams.add(conn.seam_id)
        for r in results:
            if r.condition == "C2/C3/C6" and r.verdict is REFUSE:
                e7_ok = False

    # E2, second half. Note that E2 and admissibility are NOT the same test, and
    # conflating them would misreport the stamp. E2 as the envelope states it is
    # "interfaces static and geometrically coincident"; the 2026-08-27 amendment
    # changed the CONNECTION RULE, not the hypothesis. So a seam can satisfy E2 --
    # static, coincident geometry -- and still be refused at L3 for want of the
    # declaration the amended rule requires. That is the wind farm exactly.
    unbacked = [
        c.seam_id
        for c in graph.connections
        if c.seam_id not in ctx.transfers and not c.geometrically_coincident
    ]
    if unbacked:
        stamp.fails(
            Hypothesis.E2,
            f"seams {unbacked} neither declare a common interface space and prolongation "
            "pair nor declare geometric coincidence, so neither branch of E2 is "
            "established",
        )
    elif Status.FAILS is not stamp[Hypothesis.E2]:
        coincident_only = [
            c.seam_id for c in graph.connections if c.seam_id not in ctx.transfers
        ]
        why = "every port is static and every seam declares a stable prolongation per side"
        if coincident_only:
            why = (
                f"every port is static, and seams {coincident_only} declare geometric "
                "coincidence -- which satisfies the hypothesis and does NOT satisfy the "
                "amended connection rule, so they are stamped here and refused at L3. "
                "Note that coincidence is a declaration nothing checks"
            )
        elif not all_conforming:
            why += (
                " (non-conforming, admitted through the declared common interface space; "
                "the [AI Inference] that probing accommodates this is unverified)"
            )
        stamp.holds(Hypothesis.E2, why)

    # E7 distinguishes "measured and wrong" from "never declared". A seam refused
    # for want of a declaration leaves the hypothesis UNCHECKED -- nothing was
    # checked -- while a declared pair that measurably fails adjointness makes it
    # FAIL. Collapsing the two would report a failure the run never established.
    if not e7_ok:
        stamp.unchecked(
            Hypothesis.E7,
            "a seam declares no adjoint reduction/prolongation pair, so power "
            "preservation is not established either way. The connection is refused at "
            "L3 -- a power leak is invisible in every residual the framework reports, "
            "which is why this is the one envelope hypothesis whose failure is a refusal "
            "rather than a decertification -- but the hypothesis itself is unchecked, not "
            "failed",
        )
    else:
        stamp.unchecked(
            Hypothesis.E7,
            "adjointness holds by construction from the single declared prolongation; "
            "passivity is unchecked until the probed block's symmetric part is inspected",
        )

    n_pressure = advec_passenger_pressure(
        [p for a in graph.agents for p in a.capabilities.ports]
    )
    if n_pressure:
        ctx.holes.activate(
            PORT_AMENDMENT, "<vocabulary>", advec_ports_with_extra_passengers=n_pressure
        )
        ctx.holes.measure(
            PORT_AMENDMENT, "advec_passenger_pressure_count", n_pressure,
            note="the leading indicator of vocabulary pressure, countable today",
        )



def _l3_global_fields(ctx: _Context) -> None:
    """W117: a global field whose value is another agent's state is a seam.

    `GlobalField` bypasses L3 by design -- gravity enters each agent's update
    directly and there is nothing to certify.  What the class could not express
    until 2026-09-02 is the difference between that and *routing a genuine
    two-agent coupling through it*, which is numerically exact, turns off every
    check a seam defines, and improves the envelope stamp by removing the
    quantities that would have failed.  `port-algebra-atlas-0.1` section 10.4
    states the hazard; this rule is the reader it did not have.

    The discriminator is `GlobalField.produced_by` and it is decidable from the
    declaration alone -- no probe, no solve, no measurement:

    * undeclared       -> DECERTIFY.  Nothing can be checked, and the default is
                          not "external" precisely so that silence is not a pass.
    * external, ``()`` -> ADMIT.  Section 5.1's case.
    * every agent      -> ADMIT.  A global operation over the whole state, owned
                          by the composition layer.  A split-step pressure solve
                          is this, and it is why the rule tests a PROPER subset
                          rather than merely "some agent produces it".
    * proper subset,
      applied outside  -> REFUSE.  One agent's state enters another agent's
                          update with no seam: a connection declared out of the
                          port algebra.  This is the silent-wrongness class, and
                          it is the one case the rule exists for.
    """
    rec, graph = ctx.record, ctx.graph
    if not graph.global_fields:
        return
    ids = {a.agent_id for a in graph.agents}
    for gf in graph.global_fields:
        subject = f"global-field:{gf.name}"
        applies = set(gf.applies_to) if gf.applies_to else set(ids)
        if gf.produced_by is None:
            rec.decertify(
                "L3", "global-field",
                f"global field {gf.name!r} does not declare its provenance, so whether "
                "it is an external field or another agent's state routed around the "
                "port algebra cannot be decided. Declare produced_by: () asserts "
                "external",
                subject=subject,
            )
            continue
        produced = set(gf.produced_by)
        stray = sorted(produced - ids)
        if stray:
            rec.refuse(
                "L3", "global-field",
                f"global field {gf.name!r} declares produced_by {stray}, which "
                "names no agent in this graph",
                subject=subject,
            )
            continue
        if not produced:
            rec.admit(
                "L3", "global-field",
                f"global field {gf.name!r} is external: no agent in this graph "
                "produces it, so there is no seam being bypassed",
                subject=subject,
            )
            continue
        if produced >= ids:
            rec.admit(
                "L3", "global-field",
                f"global field {gf.name!r} is produced by every agent -- a global "
                "operation over the whole state, owned by the composition layer "
                "rather than by any seam between two agents",
                subject=subject,
            )
            continue
        crossing = sorted(applies - produced)
        if crossing:
            rec.refuse(
                "L3", "global-field",
                f"global field {gf.name!r} is produced by {sorted(produced)} and "
                f"applies to {crossing}, so it carries one agent's state into "
                "another agent's update with no seam. That is a connection declared "
                "out of the port algebra: it is numerically whatever the physics is, "
                "and it is certified by nothing -- no scale set, no prolongation, no "
                "adjoint, no null space, no tau, sigma or beta. Declare it as a "
                "Connection with a port type, or state why the coupling is not one "
                "(port-algebra-atlas-0.1 section 10.4)",
                subject=subject,
            )
        else:
            rec.admit(
                "L3", "global-field",
                f"global field {gf.name!r} is produced by {sorted(produced)} and "
                "applies to no agent outside that set, so it crosses nothing",
                subject=subject,
            )


def _decomposition_cuts(graph: CaseGraph) -> tuple[set[str], set[str]]:
    """Which agents the decomposition actually cuts, from the declarations alone.

    **This is R10's premise check, factored out at W136 so that two rules resting
    on the same premise cannot come to disagree about it.**  W114 gave R10 the
    predicate: an agent's domain has been cut when the graph contains **another
    agent of the same `governing_family`**, which is the compile-time signature of
    *"a larger region of the same physics, of which this agent has been given a
    piece"*.  Its complement -- the sole agent of its family -- owns its whole
    region, so every boundary it has is a **physical** one and the decomposition
    has created no artificial face on it.

    It is a **proxy for "is this agent's domain cut", not a decision procedure**,
    and it is conservative in the safe direction: two agents of one family on
    genuinely disjoint regions are reported cut, which inspection resolves, where
    the other direction would be a silent admission.  What would replace it is a
    structured domain declaration on `Agent`, which does not exist --
    ``Agent.domain`` is free text.

    Returns ``(cut, uncut)`` as sets of agent ids.  Agents that declare no
    governing family are grouped together under ``""``, which is the behaviour
    R10 has had since W114 and is preserved here deliberately: two agents that
    both decline to say what they solve are not evidence that they solve
    different things.
    """
    families: dict[str, int] = {}
    for a in graph.agents:
        fam = a.capabilities.governing_family or ""
        families[fam] = families.get(fam, 0) + 1
    cut: set[str] = set()
    uncut: set[str] = set()
    for a in graph.agents:
        fam = a.capabilities.governing_family or ""
        (cut if families.get(fam, 0) > 1 else uncut).add(a.agent_id)
    return cut, uncut


def _r10_elliptic(ctx: _Context) -> None:
    """R10: decomposing an agent that embeds a global elliptic solve is silent.

    Added 2026-08-27 from the first real measurement. A projection method's
    pressure Poisson solve has an INFINITE domain of dependence over whatever
    domain it runs on, so cutting the domain cuts the operator -- and the error
    that follows is elliptic: flat in distance from the cut, flat in dt, and
    untouched by any halo, partition of unity or interface condition. It was 99.8%
    of the first measured defect and it was being read as agent infidelity.

    This is the silent-wrongness class exactly: no exception, no failed gate, and
    a number that looks like a bad expert. So it refuses.

    **W114, closed 2026-09-04 at CS-12.** The rule's own sentence names a premise
    -- *"the graph decomposes the domain"* -- that it never checked. It read
    ``elliptic_subsolve`` alone and refused every graph containing an EMBEDDED
    agent, including two where the decomposition does not cut that agent at all:
    `thermal_strain`'s co-located split, where both agents own all of Omega, and
    `wing_fsi`'s FSI seam, where the fluid is tiled and Omega_solid is one
    agent's whole domain. In both the elliptic solve runs over exactly the region
    a monolith would run it over, so nothing has been decomposed and the elliptic
    error the rule is about cannot arise. It fired anyway, and on a quasi-static
    structural agent there is no `split-step` escape to take -- no time
    derivative, nothing to sub-step -- so the refusal was terminal.

    The premise is now checked, from the declarations alone: an EMBEDDED agent is
    refused only when the graph contains **another agent of the same
    `governing_family`**, which is the compile-time signature of *"a larger region
    of the same physics, of which this agent has been given a piece"*. Every graph
    the rule was derived on is a tiling of one family and keeps its refusal
    unchanged.

    It is a **proxy for "is this agent's domain cut", not a decision procedure**,
    and it is deliberately conservative in the safe direction: two agents of one
    family on genuinely disjoint regions are still refused, which is a false
    refusal that inspection resolves, where the other direction would be a silent
    admission. What would replace it is a structured domain declaration on
    `Agent`, which does not exist -- ``Agent.domain`` is free text.

    **W160, closed 2026-09-09: the SAME scope defect a third time, in the
    ``undeclared`` branch.** That branch reasons *"an incompressible solver
    almost always contains a pressure solve"*, which is true of a **field**
    solver and vacuous for an algebraic closure with no field. Censused before it
    was narrowed, it fired on six agents across four unrelated graphs and every
    one of them was two lines of algebra: `ground_effect.Suspension` and its
    re-use in `front_wing` (a spring, k(h0-h)=L), `cooling_loop`'s four lumped
    coolant legs, `powertrain`'s actuator disk. Not one field solver anywhere in
    the vault was in the list -- the branch had a 0% hit rate on the class it is
    about. It is now narrowed by ``stencil_radius``, which is the compile-time
    signature already declared: zero means no spatial operator, hence no discrete
    Laplacian, hence no pressure Poisson problem to hide.

    So all three of R10's branches now check a **premise** rather than a label,
    and all three checks are proxies read off declarations that exist: the
    same-family count for *"is this agent's domain cut"*, `_decomposition_cuts`
    for the halo, and now ``stencil_radius`` for *"is there a field here at
    all"*. The cleared agents are named in an admission rather than dropped, on
    W136's discipline -- a rule that quietly stops looking at an agent is the
    same failure in the direction that does not announce itself.
    """
    rec, graph = ctx.record, ctx.graph
    if len(graph.agents) < 2:
        return
    cut, _uncut = _decomposition_cuts(graph)
    embedded, sole = [], []
    for a in graph.agents:
        if a.capabilities.elliptic_subsolve is not EllipticSubsolve.EMBEDDED:
            continue
        (embedded if a.agent_id in cut else sole).append(a.agent_id)
    incompressible_none = [a for a in graph.agents
                           if a.capabilities.elliptic_subsolve is EllipticSubsolve.NONE
                           and a.capabilities.governing_family
                           and "incompressible" in a.capabilities.governing_family]
    # W160, closed 2026-09-09. The branch below reasons "an incompressible
    # solver almost always contains a pressure solve". That is true of a FIELD
    # solver and vacuous for an algebraic closure, and `stencil_radius` is the
    # compile-time signature that separates them: zero means no spatial
    # operator, hence no discrete Laplacian, hence no pressure Poisson problem
    # for the declaration to be hiding.
    undeclared = [a.agent_id for a in incompressible_none
                  if a.capabilities.stencil_radius > 0]
    lumped = [a.agent_id for a in incompressible_none
              if a.capabilities.stencil_radius == 0]
    if embedded:
        rec.refuse(
            "L2", "R10",
            f"{len(embedded)} agents declare elliptic_subsolve=embedded and the graph "
            "decomposes the domain: each shares its governing_family with at least one "
            "other agent, so the family's region has been cut and each agent solves a "
            "global problem on its own piece of it. The decomposition changes the "
            "operator rather than restricting it. The resulting error does not decay "
            "with distance from the cut and no coupling scheme removes it -- measured "
            "at 99.8% of the total defect, mislabelled as agent infidelity. Declare "
            "elliptic_subsolve=exposed and run that solve in the composition layer, "
            "which is what probed-dtn-coupling 2.1 assumes when it says the elliptic "
            "part stays global. **What taking that repair is worth, measured on one "
            "graph at one state (W168, CS-S1 2026-09-09): 149x in coupling sweeps.** "
            "The same four windows over the same field need 74.81 sweeps per decade "
            "of interface residual with the pressure solve embedded and 0.50 with it "
            "exposed -- 466 sweeps against 7 to the same tolerance, 59.8 seconds "
            "against 0.62. A control with the projection dropped runs at 0.49, so the "
            "gain is the EXPOSURE and not the projection. This rule is not "
            "bookkeeping: it is the largest measured effect in this package, and it "
            "is a refusal rather than a decertification because the error it names is "
            "silent",
            subject=", ".join(embedded), quantity="tau",
        )
    if sole:
        rec.admit(
            "L2", "R10/sole-family",
            f"{len(sole)} agents declare elliptic_subsolve=embedded and are the ONLY "
            "agent of their governing_family in this graph, so the decomposition does "
            "not cut them: the elliptic solve runs over exactly the region a monolith "
            "would run it over, and R10's premise -- that the decomposition changes "
            "the operator rather than restricting it -- does not hold. R10 is right "
            "about the class it was derived on (a tiling of one family) and refused "
            "outside it until W114 closed 2026-09-04. The predicate is a PROXY for "
            "whether this agent's own domain is cut, and it is conservative: two "
            "agents of one family on disjoint regions would still be refused. What it "
            "does NOT clear is the agent's reach -- an embedded elliptic solve has an "
            "infinite domain of dependence inside its own region whether or not that "
            "region was cut, so the halo rule and support_reach still apply",
            subject=", ".join(sole), quantity="tau",
            r10_premise="uncut: sole agent of its governing_family",
        )
    unknown = [a.agent_id for a in graph.agents
               if a.capabilities.elliptic_subsolve is EllipticSubsolve.UNKNOWN]
    if unknown:
        rec.decertify(
            "L2", "R10/W60",
            f"{len(unknown)} agents declare elliptic_subsolve=unknown. R10 can neither "
            "fire nor be cleared: if the agent embeds a global solve, this graph has an "
            "undeclared R10 violation and tau absorbs it silently; if it does not, "
            "nothing is wrong. **The probe cannot settle it** -- elliptic_signature "
            "measures NON-NORMALITY and was read as measuring GLOBALITY, and the two "
            "come apart for a self-adjoint elliptic operator: a known-embedded backward "
            "Euler conduction solve reads kappa=1.0002, asymmetry=8.8e-5, which the "
            "statistic's own thresholds call `consistent with EXPOSED or NONE`, across a "
            "10,000x range of exchange interval. So this is a declaration nobody can "
            "check, and the honest verdict is a decertification naming it",
            subject=", ".join(unknown), quantity="tau",
        )
    if undeclared:
        rec.decertify(
            "L2", "R10",
            f"{len(undeclared)} agents declare an incompressible governing family, "
            "elliptic_subsolve=none, and a NONZERO stencil_radius. A field solver on a "
            "spatial stencil almost always contains a pressure solve; if it does, this "
            "graph has an undeclared R10 violation and tau will absorb it silently. "
            "The stencil is what makes the inference available: an agent with a "
            "discrete Laplacian to build has somewhere to hide a pressure Poisson "
            "problem, and W160 narrowed this branch to those agents on 2026-09-09",
            subject=", ".join(undeclared), quantity="tau",
        )
    if lumped:
        rec.admit(
            "L2", "R10/lumped",
            f"{len(lumped)} agents declare an incompressible governing family and "
            "elliptic_subsolve=none with stencil_radius=0, so there is no spatial "
            "operator, no discrete Laplacian, and no pressure Poisson problem for the "
            "declaration to be hiding. R10's undeclared branch reasons that an "
            "incompressible solver almost always contains a pressure solve, which is "
            "true of a FIELD solver and vacuous for an algebraic closure -- and the "
            "family declaration is nonetheless correct, because a lumped closure "
            "WITHIN a continuum problem is not a different continuum problem and "
            "declaring otherwise fails E3 at the seam. Narrowed at W160 2026-09-09, "
            "after the branch fired on six agents across four unrelated graphs and "
            "every one of them was algebraic: front_wing/ground_effect's SUSP (a "
            "two-line spring k(h0-h)=L), cooling_loop's four lumped coolant legs, and "
            "powertrain's actuator disk. This is a PROXY, like R10's own premise "
            "check: stencil_radius=0 is the compile-time signature of "
            "'no spatial operator', not a proof that no solve is hidden, and an agent "
            "that returns a field it computed somewhere else would pass it",
            subject=", ".join(lumped), quantity="tau",
            r10_premise="lumped: stencil_radius=0, no spatial operator",
        )


#: Accelerators that reach the interface solution by ITERATING on it, so a
#: cycle's composed gain has to contract for the iteration to have a limit.
#: `DIRECT_SCHUR` is not among them -- it solves the assembled system in one
#: shot and is order-free, which is R5's own sentence -- and neither is the
#: empty-operator case, which is not a coupling at all.
SWEEPING_ACCELERATORS: frozenset = frozenset({
    Accelerator.RICHARDSON, Accelerator.ROBIN_RICHARDSON,
    Accelerator.KRYLOV, Accelerator.NEWTON_KRYLOV,
})


def _r13_directed_cycle(ctx: _Context, scheme) -> None:
    """R13: a directed cycle is admissible when its composed gain contracts.

    **Added 2026-09-09, W163, and it is stated against TWO graphs on purpose.**
    A rule invented the same day the first cyclic graph appeared is a rule
    validated on one graph, which is the failure this whole direction exists to
    avoid, so the row was left open from 2026-09-08 until `powertrain` gave it a
    second circuit on unrelated physics.

    **The defect it repairs: the compiler could not tell a circuit from a
    chain.**  Compile `cooling_loop` with the return seam and without it and the
    entire difference in the decision record is the extra seam's own per-seam
    rows -- same verdict, same rule set, same per-agent decisions.  Nine layers
    of admissibility saw a directed cycle as a list of independent seams.
    `powertrain` reproduces it exactly: drop `mgu_bus`'s closing edge and
    nothing above L3 notices.  Meanwhile the measured consequence on CS-13 is
    1.5 K of order dependence at one sweep per macro-step, which is larger than
    several defects this framework does refuse over.

    **What a cycle is, and why order-dependence is the symptom rather than the
    disease.**  A chain has a topological order: every agent's inputs are
    available before it runs, one sweep is exact, and the scheme is a
    composition.  A cycle has none -- ``no agent's inputs are all available
    before the others have run``, in `cooling_loop`'s own words -- so one sweep
    is one step of a fixed-point iteration and the scheme's answer depends on
    where the sweep started.  A cycle therefore has ``2n`` equally defensible
    sweep orders and nothing in the declaration prefers one, which is
    `spec-wind-farm-wake` 5.1 stated as a list.  The repair is not to pick an
    order.  It is to require that the iteration those orders are all
    approximating actually has a fixed point to reach, which is a property of
    the loop map and not of the sweep.

    **The condition.**  Let ``G`` be the composed map of one traversal.  Then

        rho(G) < 1   -- a contraction.  Banach gives a unique fixed point and
                       convergence from any start, so the sweep order stops
                       mattering in the limit and the graph is a composition
                       again.  ADMIT.
        rho(G) = 1   -- no isolated fixed point.  The loop does not settle, and
                       that is a statement about the circuit rather than a
                       solver failure.  REFUSE.
        rho(G) > 1   -- the iteration diverges.  REFUSE.

    Both case studies already raise at exactly this condition inside their own
    closed forms, which is the check this rule lifts to compile time:
    `cooling_loop.LoopSolve._closed_form` raises when ``|1 - a_total| < 1e-15``
    (``B / (1 - A)`` has no value at unit gain) and `powertrain.Circuit.
    _closed_form` raises when the loop's total resistance is not positive (``I =
    sum(emf) / sum(R)`` is unbounded).  Measured, the two circuits contract
    comfortably: CS-13's composed gain is 0.9238, the product of the legs' own
    coefficients with the radiator's 0.925 doing the work; CS-14's loop is
    resistive at every element, which is the same statement in electrical form.

    **The gain is DECLARED, and that is a finding rather than an interface
    convenience.**  W163's row carried an [AI Inference] that the contraction
    would be *"decidable from the same declared response the probe already
    builds"*.  It was checked on both graphs and it is false:

      * `boundary_response` is ``(port, trace) -> flux`` on the SAME port, a
        Dirichlet-to-Neumann map.  A cycle's gain is a CROSS-port transfer --
        inlet datum to outlet datum -- and no capability field carries one.
      * On CS-13 the two are unrelated quantities.  Perturbing a leg's ADVEC
        trace moves the declared response by 2303.89 J/kg per unit mass flux;
        the loop's transport gain is ``a = 0.99926`` in TEMPERATURE and shows up
        only in the derivative with respect to the upstream state, which the
        port interface does not expose.  Reaching it means setting ``leg.t_in``
        on the expert object, which is reaching around the declaration and not
        reading it.
      * On CS-14 the ELEC response IS the element's series resistance, up to the
        declared terminal area -- so on that graph the ingredient is in the
        interface after all.

    Two circuits, two different answers, so a rule that computed the gain would
    be a rule that works on one of the two cyclic graphs in the package.  It
    asks instead, and decertifies rather than guessing when nothing is declared.
    That decertification is the honest state of both graphs today: the cycle is
    seen, its admissibility is not decidable from what has been declared, and
    the record says which.

    **What this does NOT do.**  It does not order the sweep, and it does not
    certify a particular scheme's answer at a finite number of sweeps -- L2/C2
    and the master bound own the truncation.  ``rho(G) < 1`` says the sequence
    has a limit and every sweep order shares it; how far a one-sweep-per-
    macro-step scheme sits from that limit is CS-13's measured 1.5 K, and it is
    a different quantity from this one.
    """
    rec, graph = ctx.record, ctx.graph
    try:
        cycles = graph.directed_cycles()
    except CycleEnumerationBudget as e:
        # A PARTIAL cycle list is worse than none: a rule reasoning over some of
        # a graph's cycles is silently reasoning about a different graph. So the
        # detector raises and this says so, in L2/C2's shape -- a criterion with
        # no value in it.
        rec.decertify(
            "L5", "R13",
            f"the directed cycles of this seam graph could not be enumerated: "
            f"{e}. R13's condition is therefore a theorem with no subject here. "
            "It is not evidence that the graph HAS many cycles -- the search is "
            "factorial in the worst case and this is a budget -- but it does "
            "mean nothing below has been checked against the graph's topology",
            subject="<graph>", quantity="loop_gain",
            detector="directed_cycles (budget exceeded)",
        )
        return
    if not cycles:
        return
    acc = getattr(scheme, "accelerator", None)
    sweeps = acc in SWEEPING_ACCELERATORS
    for cycle in cycles:
        route = " -> ".join(cycle) + f" -> {cycle[0]}"
        declared = graph.declared_loop_gain(cycle)
        if not sweeps:
            # **The rule is about an ITERATION and this scheme has none.** The
            # composition layer solves the assembled interface system in one
            # shot, so there is no sweep, no first agent to choose and no
            # fixed-point sequence to converge -- R5's own sentence about a
            # direct Schur solve being order-free, reaching the case R5 was
            # written before anyone had. The cycle is still REPORTED, because
            # W163's complaint is that the compiler could not tell a circuit
            # from a chain at all, and an admission that names the cycle and
            # says why it does not bind is the repair; silence would be the
            # defect again in a quieter form.
            gain = "undeclared" if declared is None else f"{declared.gain:.6g}"
            rec.admit(
                "L5", "R13",
                f"the seam graph contains a DIRECTED CYCLE ({route}) with a "
                f"composed loop gain of {gain}, and the scheme does not sweep "
                f"it: accelerator={acc.value if acc else 'none'} solves the "
                "assembled interface system in one shot, so there is no first "
                "agent to choose, no sweep order for the answer to depend on, "
                "and no fixed-point iteration for the gain to govern. R5 says "
                "the same thing about ordering -- a direct Schur solve is "
                "order-free -- and this is that sentence reaching the case R5 "
                "was written before anyone had. **What this does NOT clear is "
                "the cycle**: declare a sweeping accelerator and the gain "
                "becomes binding, which on this graph is what R13 would then "
                "decide on",
                subject="<graph>", quantity="loop_gain",
                cycle=list(cycle), accelerator=acc.value if acc else None,
                binding=False,
                loop_gain=None if declared is None else declared.gain,
            )
            continue
        if declared is None:
            rec.decertify(
                "L5", "R13",
                f"the seam graph contains a DIRECTED CYCLE ({route}) and the "
                "graph declares no loop gain for it, so the criterion is a "
                "theorem with no value in it. A cycle has no topological order: "
                "no agent's inputs are all available before the others have "
                "run, so one sweep is one step of a fixed-point iteration "
                "rather than an evaluation, and the 2n rotations of the cycle "
                "are 2n equally defensible schemes that nothing in the "
                "declaration prefers. Whether they share a limit is decided by "
                "the composed gain of one traversal and by nothing else. "
                "**The compiler cannot measure it**: boundary_response is a "
                "same-port Dirichlet-to-Neumann map and a cycle's gain is a "
                "cross-port transfer, and the two come apart -- on CS-13 the "
                "declared response moves 2303.89 J/kg per unit mass flux while "
                "the transport gain is 0.99926 in temperature and reachable "
                "only through the expert's internal state. Declare a "
                "graph.DeclaredLoopGain. Until then the per-seam rows below are "
                "the same rows this graph would get with the cycle CUT, which "
                "is the defect W163 opened: compiled with and without the "
                "closing seam, the decision records differ only by that seam's "
                "own rows",
                subject="<graph>", quantity="loop_gain",
                cycle=list(cycle), detector="directed_cycles (structural proxy)",
            )
            continue
        if not declared.contracts:
            rec.refuse(
                "L5", "R13",
                f"the directed cycle {route} declares a composed loop gain of "
                f"{declared.gain:.6g}, which is not a contraction. At a gain of "
                "exactly one the loop map has no isolated fixed point -- a "
                "circuit with no dissipative element does not settle, and that "
                "is a statement about the circuit rather than a solver failure "
                "-- and above one the iteration diverges. Either way the 2n "
                "sweep orders do not share a limit, so the composed answer is a "
                "property of where the sweep started and the graph is not a "
                "composition. Both case studies raise at exactly this "
                "condition inside their own closed forms: cooling_loop's "
                "B / (1 - A) has no value at unit gain, and powertrain's "
                "sum(emf) / sum(R) is unbounded at zero loop resistance",
                subject="<graph>", quantity="loop_gain",
                cycle=list(cycle), loop_gain=declared.gain,
                source=declared.source,
            )
            continue
        rec.admit(
            "L5", "R13",
            f"the directed cycle {route} declares a composed loop gain of "
            f"{declared.gain:.6g} < 1, so one traversal is a contraction: the "
            "loop map has a unique fixed point, every one of the 2n sweep "
            "orders converges to it from any start, and the cycle's lack of a "
            "topological order stops being a modelling choice in the limit. "
            f"Source: {declared.source or 'not stated'}. **What this does NOT "
            "certify is a finite-sweep answer** -- rho < 1 says the sequence "
            "has a limit and says nothing about how far one sweep per "
            "macro-step sits from it, which on CS-13 is a measured 1.5 K of "
            "order dependence. That distance is L2/C2's and the master bound's "
            "quantity, not this rule's. The cycle itself is found by a "
            "STRUCTURAL proxy -- the elementary directed cycles of the seam "
            "digraph, taken in each connection's own declared (a, b) order -- "
            "which returns exactly these two circuits and nothing on any tiling "
            "in the package, because a tiling orients its seams monotonically "
            "along the grid axes and its digraph is acyclic",
            subject="<graph>", quantity="loop_gain",
            cycle=list(cycle), loop_gain=declared.gain, source=declared.source,
        )


def _uncut_clause(excluded: list[str]) -> str:
    """W136: name the agents the halo requirement was NOT checked against.

    The exclusion is the whole content of W136's fix, so it travels with every
    outcome of the rule rather than only with the one it changed. A reader who
    sees `R10/halo` admit on a graph containing an implicit agent has to be able
    to tell the sound reason from the bug this replaced.
    """
    if not excluded:
        return ""
    return (
        ". The requirement was checked against the agents the decomposition CUTS "
        "and not against " + ", ".join(excluded) + ", each of which is the sole "
        "agent of its governing_family and therefore owns its whole region: every "
        "boundary it has is a physical one, and there is no artificial face for an "
        "overlap to outrun (W136)"
    )


def _halo_bounds(graph, cut: set[str]) -> str:
    """Which quantity the halo requirement is bounding on THIS graph.

    **W168, 2026-09-09.** The rule's sentence is about contaminated cells, which
    is an accuracy statement, and on an embedded agent the same number is a
    convergence threshold -- measured, and sharply. Naming the quantity is the
    whole content of the row: a reader handed "widen the overlap" cannot
    otherwise tell whether a narrower one costs them accuracy or costs them the
    fixed point.
    """
    embedded = [a.agent_id for a in graph.agents
                if a.agent_id in cut
                and a.capabilities.elliptic_subsolve in (
                    EllipticSubsolve.EMBEDDED, EllipticSubsolve.UNKNOWN)]
    return "accuracy and convergence" if embedded else "accuracy"


def _halo_bounds_clause(graph, cut: set[str]) -> str:
    """The measured sentence behind `_halo_bounds`, for the decision's message."""
    if _halo_bounds(graph, cut) == "accuracy":
        return (
            ". **What this bounds here is ACCURACY** (W168): every agent the "
            "decomposition cuts declares elliptic_subsolve=exposed, so no local "
            "step contains a global solve and a halo short of the reach carries "
            "stale data into the blended region without stopping the sweep from "
            "converging. Measured on CS-S1's exposed arrangement, the iteration "
            "converges at EVERY halo from 4 up -- contraction 0.089 at halo 4 "
            "against a declared reach of 20 -- so a narrow halo here costs "
            "accuracy and does not cost the fixed point"
        )
    return (
        ". **What this bounds here is ACCURACY *and* CONVERGENCE** (W168): at "
        "least one agent the decomposition cuts has its elliptic part embedded "
        "or undeclared, and an embedded global solve has an infinite domain of "
        "dependence inside its own region, for which the halo is what stands "
        "in. Measured on CS-S1's as-built arrangement over four windows, the "
        "sweep's contraction crosses 1 between halo 18 (1.00642) and halo 20 "
        "(0.980531) against a declared reach of exactly 20 -- so one step short "
        "of the requirement the iteration does not converge at all, rather than "
        "converging to something less accurate. The same graph with the "
        "elliptic part EXPOSED converges at every halo from 4 up, which is R10's "
        "own repair showing up in this rule's threshold"
    )


def _halo_rule(ctx: _Context) -> None:
    """The overlap must outrun the agent's own domain of dependence.

    An artificial boundary's influence travels ``stencil_radius`` cells per
    internal sub-step. Under a once-per-macro-step exchange nothing corrects it
    for ``substeps_per_macro_step`` of them, so the overlap must exceed the
    product or the blended region is contaminated. Re-assembling every sub-step
    reduces the requirement to ``stencil_radius``, which is the trade this priced.

    **W136, closed 2026-09-08: the rule had R10's scope defect one rule along.**
    Every word of the paragraph above is about an **artificial** boundary -- the
    face the decomposition created, whose stale datum the overlap has to outrun --
    and the rule never asked whether the agent it was decertifying has one. It
    read ``time_discretization`` and ``stencil_radius`` off every agent in the
    graph, so on `front_wing` and `wing_fsi` it decertified STRUCT for being
    implicit with a nonzero stencil. That reasoning is correct about STRUCT's
    domain of dependence and irrelevant to this graph: ``Gamma`` is a *physical*
    boundary of ``Omega_solid``, the structure is not tiled, and there is no
    overlap on that side for anything to outrun. The requirement is now checked
    only against the agents `_decomposition_cuts` reports cut -- **R10's own
    premise predicate, shared rather than restated**, so the two rules cannot
    drift apart about which agents the decomposition touches.

    The exclusion is reported in the decision's evidence rather than being
    silent: a rule that quietly stops looking at an agent is the same failure in
    the other direction, and it is the one that is not self-announcing.
    """
    rec, graph = ctx.record, ctx.graph
    axis = ctx.decomposition or graph.decomposition
    if axis is not Decomposition.OVERLAPPING:
        return
    cut, uncut = _decomposition_cuts(graph)
    excluded = sorted(uncut)
    needs = {a.agent_id: a.capabilities.required_halo() for a in graph.agents
             if a.agent_id in cut}
    if not needs:
        rec.admit(
            "L2", "R10/halo",
            "the decomposition cuts no agent -- each is the sole one of its "
            "governing_family, so every boundary it has is a physical boundary of its "
            "own region and the decomposition has created no artificial face for an "
            "overlap to outrun. The halo requirement has no subject here",
            subject="<graph>", uncut_agents=excluded,
        )
        return
    unknown = [k for k, v in needs.items() if v is None]
    if unknown:
        # W69, 2026-08-30: `required_halo` now returns None for two different
        # reasons and they want different sentences. An undeclared sub-step count
        # is a gap in the record; an implicit agent with a real stencil has a
        # dense one-step inverse, so its domain of dependence is the whole domain
        # and no product of declared integers is the right number.
        implicit = [k for k in unknown
                    if graph.agent(k).capabilities.time_discretization
                    is TimeDiscretization.IMPLICIT
                    and int(graph.agent(k).capabilities.stencil_radius) >= 1]
        # W93, 2026-08-30: the third reason, and W69's own derivation reaches it.
        # `radius * substeps` is an EXPLICIT agent's domain of dependence, and an
        # agent declaring `unknown` has not said its step is explicit.
        opaque = [k for k in unknown
                  if graph.agent(k).capabilities.time_discretization
                  is TimeDiscretization.UNKNOWN
                  and int(graph.agent(k).capabilities.stencil_radius) >= 1]
        undeclared = [k for k in unknown if k not in implicit and k not in opaque]
        parts = []
        if undeclared:
            parts.append(
                f"{len(undeclared)} agents ({', '.join(undeclared)}) do not declare "
                "substeps_per_macro_step"
            )
        if opaque:
            parts.append(
                f"{len(opaque)} agents ({', '.join(opaque)}) declare "
                "time_discretization=unknown with a nonzero stencil, so radius x "
                "substeps -- an EXPLICIT agent's domain of dependence -- is not "
                "licensed for them. Measured on a frozen neural operator declaring "
                "radius 2, substeps 1: probe.support_reach finds the response nonzero "
                "in every one of the 128 seam cells, 64 from the poke, a factor of 32 "
                "above the declaration. Measure it with probe.support_reach rather "
                "than declaring it"
            )
        if implicit:
            parts.append(
                f"{len(implicit)} agents ({', '.join(implicit)}) are implicit with a "
                "nonzero stencil, so one macro-step inverts an operator coupling every "
                "cell to every other and stencil_radius x substeps is an EXPLICIT "
                "agent's domain of dependence, not theirs. Measured on a conducting "
                "shell declaring radius 1, substeps 1: a delta on the seam still moves "
                "the response 5 cells out at the 1e-6 level"
            )
        rec.decertify(
            "L2", "R10/halo",
            "; ".join(parts) + ". The halo the exchange needs is undecidable here and "
            "it is not assumed adequate"
            + _uncut_clause(excluded),
            subject=", ".join(unknown), quantity="tau",
            uncut_agents=excluded,
        )
        return
    required = max(v for v in needs.values())
    have = graph.overlap_cells
    if have is None:
        rec.decertify(
            "L2", "R10/halo",
            f"the agents' domain of dependence needs an overlap of {required} cells and "
            "the graph declares its overlap in physical units only, so the check cannot "
            "run. Declare overlap_cells"
            + _uncut_clause(excluded),
            subject="<graph>", quantity="tau", uncut_agents=excluded,
        )
    elif have < required:
        rec.refuse(
            "L2", "R10/halo",
            f"overlap is {have} cells and the agents' domain of dependence over one "
            f"macro-step is {required} (stencil_radius x substeps). The blended region "
            "is contaminated by each window's own artificial boundary, and the partition "
            "of unity gives that boundary full weight. Widen the overlap, or exchange "
            "every sub-step, which drops the requirement to the stencil radius"
            + _halo_bounds_clause(graph, cut)
            + _uncut_clause(excluded),
            subject="<graph>", quantity="tau", uncut_agents=excluded,
            bounds=_halo_bounds(graph, cut),
        )
    else:
        rec.admit(
            "L2", "R10/halo",
            f"overlap {have} cells covers the {required}-cell domain of dependence"
            + _halo_bounds_clause(graph, cut)
            + _uncut_clause(excluded),
            subject="<graph>", uncut_agents=excluded,
            bounds=_halo_bounds(graph, cut),
        )


def _derive_transfer(ctx: _Context, conn: Connection) -> SeamTransfer | None:
    """Derive M from the declared effective resolutions, for the conforming case.

    This is not filling in a missing declaration by guessing.  dim M = min_i
    m_i_eff is a *rule*, and a side whose own effective resolution equals dim M
    has the identity as its prolongation by definition -- that is the conforming
    special case.  Anything else still requires a declared prolongation, because a
    non-conforming transfer is a modelling choice and not a derivation.
    """
    resolutions = ctx.graph.effective_resolutions(conn)
    if any(v is None for v in resolutions.values()):
        return None
    d = dim_M({k: int(v) for k, v in resolutions.items() if v is not None})
    if any(int(v) != d for v in resolutions.values()):
        return None
    space = InterfaceSpace(seam_id=conn.seam_id, dim=d, note="derived: dim M = min_i m_i_eff")
    prolongations = {}
    for agent_id, port_name in (conn.a, conn.b):
        port = ctx.graph.agent(agent_id).port(port_name)
        prolongations[agent_id] = port.prolongation or identity_prolongation(
            agent_id, port_name, d
        )
    port_a = ctx.graph.agent(conn.a[0]).port(conn.a[1])
    ctx.record.admit(
        "L3", "dim-M",
        f"interface space derived for this seam: dim M = min_i m_i_eff = {d}, with the "
        "identity prolongation on each side (the conforming special case). The multiplier "
        "space and the probe basis are the same space",
        subject=conn.seam_id,
    )
    return SeamTransfer(
        seam_id=conn.seam_id,
        port_type=conn.port_type,
        space=space,
        prolongations=prolongations,
        passengers=port_a.passengers,
    )


# --- L4 -- construction of the transmission operator ----------------------


def _l4_transmission(ctx: _Context) -> None:
    rec, graph = ctx.record, ctx.graph
    if not ctx.budget.allow_probe:
        rec.decertify(
            "L4", "budget",
            "probing disabled by budget: beta, kappa, the null-space dimension, the "
            "passivity spectrum, the optimal Robin coefficients and Xi are all "
            "unavailable, so the transmission term of the bound cannot be estimated",
            subject="<graph>", quantity="beta",
        )
        return

    for conn in graph.connections:
        transfer = ctx.transfers.get(conn.seam_id)
        provisional = False
        if transfer is None:
            transfer = _provisional_transfer(ctx, conn)
            provisional = transfer is not None
        if transfer is None:
            rec.decertify(
                "L4", "probe",
                "cannot probe this seam: no interface space is declared and none can be "
                "derived, so there is no basis to impose. The diagnostic that would have "
                "settled this seam costs one assembly and is unavailable for want of a "
                "declaration",
                subject=conn.seam_id, quantity="beta",
            )
            continue

        try:
            op = assemble_seam(
                graph, conn, transfer, ctx.probe_budget, ctx.references,
                expected_null_dim=getattr(conn, "expected_null_dim", None),
                probe_state=ctx.probe_state,
            )
        except Exception as exc:  # a probe that cannot run is reported, never guessed
            rec.decertify(
                "L4", "probe",
                f"probe failed on this seam: {exc}",
                subject=conn.seam_id, quantity="beta",
            )
            continue

        ctx.operators[conn.seam_id] = op
        if provisional:
            op.notes.append(
                "assembled on a PROVISIONAL interface space, for diagnosis only. It does "
                "not constitute an admissible connection"
            )
            rec.decertify(
                "L4", "probe/provisional",
                "probed on a provisional interface space so the diagnosis is available "
                "even though the connection is refused. The finding below is a "
                "measurement; the connection remains inadmissible until M and one "
                "prolongation per side are declared",
                subject=conn.seam_id,
            )

        _l4_verdicts(ctx, conn, op, provisional)


def _provisional_transfer(ctx: _Context, conn: Connection) -> SeamTransfer | None:
    """Build a diagnostic-only space so a refused seam can still be measured.

    The spec's central finding about the wind farm is that its negative composed
    result was available for the cost of one probe.  Refusing the connection and
    then declining to measure it would reproduce exactly the failure the finding
    is about, so a refused seam is still probed where a basis can be constructed,
    and the result is marked provisional.
    """
    resolutions = ctx.graph.effective_resolutions(conn)
    known = [int(v) for v in resolutions.values() if v is not None]
    if not known:
        return None
    d = min(known)
    space = InterfaceSpace(seam_id=conn.seam_id, dim=d, note="provisional, diagnosis only")
    prolongations = {}
    for agent_id, port_name in (conn.a, conn.b):
        port = ctx.graph.agent(agent_id).port(port_name)
        caps = ctx.graph.agent(agent_id).capabilities
        if caps.boundary_response is None:
            return None
        n = port.effective_resolution or d
        if port.prolongation is not None and port.prolongation.dim_M == d:
            prolongations[agent_id] = port.prolongation
        elif int(n) == d:
            prolongations[agent_id] = identity_prolongation(agent_id, port_name, d)
        else:
            return None
    port_a = ctx.graph.agent(conn.a[0]).port(conn.a[1])
    ctx.provisional.add(conn.seam_id)
    return SeamTransfer(
        seam_id=conn.seam_id, port_type=conn.port_type, space=space,
        prolongations=prolongations, passengers=port_a.passengers,
    )


def _l4_verdicts(
    ctx: _Context, conn: Connection, op: SeamOperator, provisional: bool = False
) -> None:
    rec = ctx.record

    if op.is_empty:
        ctx.refused_claims.append(
            f"the word 'coupled' on any output involving seam {conn.seam_id}"
        )
        rec.record(
            "L4", "6.4(a)", ADMIT_UNCERTIFIED,
            "the transmission operator is identically zero, so the interface residual is "
            "the same for EVERY trace. The interface problem is not ill-conditioned; it "
            "is empty, and every trace is equally consistent because no agent's output "
            "depends on any agent's input. Run it if you like -- an ensemble of "
            "independent local solves is a legitimate object -- but the word 'coupled' is "
            "REFUSED on the output. This is a refusal of a claim type, not of the run, "
            "and it is decidable from one cheap probe",
            failure_class=FailureClass.SILENT_WRONGNESS,
            subject=conn.seam_id, quantity="Xi", Xi=0.0,
        )
        return

    # **L4/operator-content, W68/W71, measured 2026-08-30.** Everything below
    # this point reads beta, kappa, the null count or the passivity defect off
    # the probed matrix. All four are healthy on a multiple of the identity, and
    # a multiple of the identity is what a probe returns when the interface
    # response is dominated by a boundary coefficient rather than by the
    # expert's operator. `thermal_seam` assembles to 4.8068 I to five digits --
    # +5.0001 I from the shell's film coefficient against -0.1933 I from the
    # gas's -- with the conduction physics 300x underneath. The composition is
    # not wrong and the numbers are not wrong; what is wrong is reading them as
    # statements about the solvers. So this decertifies rather than refusing,
    # and it names the quantity whose meaning changes.
    thin = {aid: b.operator_content for aid, b in op.blocks.items()
            if b.operator_content is not None
            and b.operator_content < OPERATOR_CONTENT_FLOOR}
    if thin:
        worst = min(thin.values())
        rec.decertify(
            "L4", "operator-content",
            f"{len(thin)} of {len(op.blocks)} blocks on this seam have an identity "
            f"defect below {OPERATOR_CONTENT_FLOOR:g} ("
            + ", ".join(f"{k} {v:.2e}" for k, v in sorted(thin.items()))
            + "): the probed block is a multiple of the identity, so it is the "
            "expert's BOUNDARY COEFFICIENT and not its operator. beta, kappa, the "
            "null count and the passivity defect are all well behaved on such a "
            "block and all four are then properties of that coefficient -- which "
            "means every bound carrying 1/beta is scaled by a declared constant "
            "rather than by a measured operator. The transverse physics is present "
            "and resolved (measured: an exponential tail over 5 cells) and sits "
            "300x below. Widen the exchange interval or raise the coupling until "
            "omega ~= C Bi (k_max sqrt(alpha dt))^2 clears the floor",
            subject=conn.seam_id, quantity="beta",
            operator_content=worst, floor=OPERATOR_CONTENT_FLOOR,
        )
    elif op.blocks:
        rec.admit(
            "L4", "operator-content",
            "every block on this seam carries resolvable operator content "
            "(identity defect "
            + ", ".join(f"{k} {b.operator_content:.3g}"
                        for k, b in sorted(op.blocks.items())
                        if b.operator_content is not None)
            + f"), so beta and kappa are statements about the experts' operators "
            f"rather than about a film coefficient",
            subject=conn.seam_id, quantity="beta",
        )

    # **L4/probe-base, W74's class, measured 2026-08-29.** Lambda_M = sum_i
    # P_i^* Lambda_i P_i is a sum of Jacobians, and a sum of Jacobians is a
    # Jacobian only if every term was taken at the same point. `probe_base` was
    # put on ExpertCapabilities -- one per EXPERT -- so each side of a seam names
    # its own, and nothing compared them. On `thermal_seam` the two records are
    # each honest and independently correct and they are 500 K apart: the gas
    # linearizes at the wall temperature it sees (400 K) and the shell at the gas
    # temperature it sees (900 K), on ONE interface variable.
    #
    # It decertifies rather than refusing, and the reason is the split rule
    # rather than mildness: the DISAGREEMENT is measured, but its consequence
    # depends on whether the responses are affine, and that is an unverified
    # hypothesis until `probe.base_sensitivity` is run. If they are affine the
    # base is free and the disagreement costs nothing -- the one case in this
    # vault where a measurement can PROMOTE the finding away.
    chk = op.base_check or {}
    if chk.get("agents") and not chk.get("consistent", True):
        rec.decertify(
            "L4", "probe-base",
            f"the two sides of this seam were linearized about DIFFERENT interface "
            f"states: the bases differ by {chk.get('spread', float('nan')):.4g} on M, "
            f"{100.0 * chk.get('relative_spread', float('nan')):.0f}% of the base norm. "
            "The seam operator is a sum of per-side Jacobians and that sum is a "
            "Jacobian only if the terms share a linearization point, so beta and "
            "kappa here are not statements about the interface map at any state. "
            "Measured on thermal_seam: sweeping a COMMON base over the physically "
            "admissible interval gives beta in [0.4200, 1.7689] and the mismatched "
            "probe reports 0.3757 -- below the entire range, so it is not merely the "
            "wrong point but no point at all. Pass assemble_seam a seam_base, or run "
            "probe.base_sensitivity: an affine response makes the base free and this "
            "decertification goes away",
            subject=conn.seam_id, quantity="beta",
            base_spread=chk.get("spread"), relative=chk.get("relative_spread"),
        )
    elif chk.get("agents"):
        rec.admit(
            "L4", "probe-base",
            "both sides of this seam are linearized about the same interface state, "
            "so the assembled operator is a Jacobian of the interface map",
            subject=conn.seam_id, quantity="beta",
        )

    # **L4/block-share, W76, measured 2026-08-29.** The per-block diagnostics the
    # probe has emitted since Tier 0 were read by no rule at all, and alpha_star
    # -- the measured symbol -- by nothing anywhere. What they carry is not a
    # second opinion on the assembled numbers; it is a bound on what any test
    # taken on the ASSEMBLED operator can see about one agent.
    #
    # A substitution certificate compares two assembled operators and passes when
    # ||Delta|| < beta - beta_min. Swapping agent i can move the seam by at most
    # ||S_i_old|| + ||S_i_new||, so a total failure of agent i -- a replacement
    # that ignores its boundary data entirely -- moves it by exactly ||S_i||. If
    # that is below the margin the test cannot fail, and the certificate is not
    # weak but blind. The threshold below is not calibrated: beta - ||S_i|| IS
    # the beta_min above which the test starts being informative.
    if op.beta is not None and op.blocks and len(op.blocks) > 1:
        shares = {a: b.share for a, b in op.blocks.items() if b.share is not None}
        norms = {a: float(np.linalg.norm(b.S, 2)) for a, b in op.blocks.items()}
        blind = {a: op.beta - n for a, n in norms.items() if op.beta - n > 0.0}
        ms = op.mode_shares
        mode_note = ""
        if ms is not None and ms.size:
            mode_note = (f" Per-mode, read off alpha_star, the dominant side holds "
                         f"median {float(np.median(ms)):.3f} and max {float(ms.max()):.3f} "
                         f"of the response, with {int((ms > 0.9).sum())} of {ms.size} "
                         f"modes above 0.90.")
        # What the ASSEMBLY hides, which is the part no other rule can reach.
        # A block can be orders more ill-conditioned than the operator it sums
        # into, and every §4 diagnostic the compiler reads is taken on the sum.
        # Measured on window_ns split-step's seam sx0: block kappa 4171 against
        # an assembled 1.13, and block beta 1.67e-5 against an assembled 0.349 --
        # 3700x and 21000x, between two instances of THE SAME SOLVER on a
        # symmetric tiling. The seam is one-sided because the flow is: the
        # downstream window barely responds to its inlet ring.
        kap = {a: b.kappa for a, b in op.blocks.items() if b.kappa is not None}
        hidden = {}
        if kap and op.kappa and op.kappa > 0:
            hidden = {a: k / op.kappa for a, k in kap.items() if k / op.kappa > 100.0}
        if hidden:
            rec.decertify(
                "L4", "block-share",
                "a block on this seam is far worse conditioned than the operator it "
                "assembles into, and every diagnostic downstream of L4 is taken on the "
                "sum: "
                + ", ".join(f"{a} kappa {kap[a]:.4g} against the assembled "
                            f"{op.kappa:.4g} ({r:.0f}x)" for a, r in sorted(hidden.items()))
                + ". The assembled numbers are not wrong and they are not sufficient: "
                  "beta_i is what the substructuring branch inverts (W57) and Xi_i is "
                  "the composability index, and neither is recoverable from the sum. "
                  "The mechanism here is one-sidedness -- block norms "
                + ", ".join(f"{a} {n:.4g}" for a, n in sorted(norms.items()))
                + f", assembled {float(np.linalg.norm(op.S, 2)):.4g}" + mode_note
                + " Substitution of the weak side is invisible below "
                + ", ".join(f"beta_min = {v:.4g} for {a}"
                            for a, v in sorted(blind.items())) + ".",
                subject=conn.seam_id, quantity="beta_i",
                shares=shares, block_kappa=kap, blind_below=blind,
            )
        elif blind:
            worst = max(blind, key=blind.get)
            rec.admit(
                "L4", "block-share",
                "disclosure, not a defect of this compile: the substitution certificate "
                "is BLIND at this seam. "
                + ", ".join(f"{a} becomes visible only once beta_min > {v:.4g}"
                            for a, v in sorted(blind.items()))
                + f" (assembled beta = {op.beta:.4g}, block norms "
                + ", ".join(f"{a} {n:.4g}" for a, n in sorted(norms.items()))
                + "). Replacing such an agent with one that UNDER-responds -- including "
                  "one that ignores its boundary data entirely, which is exactly the "
                  "bc_channel failure conformance calls the foundational hole -- moves "
                  "the seam by at most its own block norm, so that class of failure "
                  "cannot be caught. Since W81 the threshold above is the certificate's "
                  "own `visible_above`, `certify_substitution` takes beta_min with NO "
                  "default and reports `visible_above`/`fails_above` when none is given, "
                  "and the 1e-12 sibling default has been renamed `beta_int_floor` "
                  "because it guards a different quantity. The one derived candidate is "
                  "eps_tol = min(tau, sigma), which this graph must measure before it can "
                  "be used here. "
                  "This compile makes no substitution claim, so it is recorded and not "
                  "charged against the verdict; `composition.SubstitutionCertificate."
                  "blind` is where it has teeth, and it downgrades the pass there."
                + mode_note,
                subject=conn.seam_id, quantity="substitution",
                shares=shares, blind_below=blind, worst=worst,
            )
        else:
            rec.admit(
                "L4", "block-share",
                "every block on this seam carries more than the assembled beta, so a "
                "total failure of either agent moves the seam past any admissible "
                "beta_min and the substitution certificate can see it." + mode_note,
                subject=conn.seam_id, quantity="substitution", shares=shares,
            )

    excess = op.excess_null_directions
    if excess is None and op.null_dim:
        rec.decertify(
            "L4", "null-space",
            f"the probed operator has a {op.null_dim}-dimensional null space and the case "
            "declares no expected dimension, so the free correctness check cannot run. A "
            "probed operator whose measured null space is not the expected dimension "
            "indicts the PROBE, not the physics",
            subject=conn.seam_id, quantity="null-space dimension", null_dim=op.null_dim,
        )
    elif excess:
        rec.refuse(
            "L4", "null-space",
            f"the probed operator has {op.null_dim} null directions against an expected "
            f"{op.expected_null_dim}: {excess} in excess. Any excess null direction is a "
            "defect -- most often an untreated cross-point, whose dependent constraints "
            "show up as exactly this. The check is free and it must be run before "
            "anything is built on the matrix",
            subject=conn.seam_id, quantity="null-space dimension",
            null_dim=op.null_dim, expected=op.expected_null_dim,
        )
    elif op.expected_null_dim is not None and op.null_dim < op.expected_null_dim:
        rec.decertify(
            "L4", "null-space",
            f"the probed operator has {op.null_dim} null directions against an expected "
            f"{op.expected_null_dim}. A DEFICIT is not the cross-point signature and is "
            "not refused on the safe reading, but it still indicts one of the two: either "
            "the probe is not resolving a constraint the physics has, or the declared "
            "expectation is wrong. It is free to check and it should not pass silently",
            subject=conn.seam_id, quantity="null-space dimension",
            null_dim=op.null_dim, expected=op.expected_null_dim,
        )
    elif op.expected_null_dim is not None:
        rec.admit(
            "L4", "null-space",
            f"measured null-space dimension {op.null_dim} matches the declared "
            f"{op.expected_null_dim}",
            subject=conn.seam_id,
        )

    if op.passivity_defect is not None and op.passivity_defect > 0.0:
        rec.decertify(
            "L4", "E7/passivity",
            f"passivity defect {op.passivity_defect:.3e} on the assembled seam: the "
            "symmetric part has a negative mode, so the L <= 1 branch is unavailable "
            "and L falls back to fitted. The eigenvector names which interface mode is "
            "amplified",
            subject=conn.seam_id, quantity="L", defect=op.passivity_defect,
        )
        # A MEASURED negative mode is a genuine failure of E7, unlike a missing
        # declaration -- so this one does set FAILS, even from a provisional probe.
        ctx.stamp.set(
            Hypothesis.E7, Status.FAILS,
            f"seam {conn.seam_id}: passivity defect {op.passivity_defect:.3e}",
        )
    elif op.passivity_defect is not None and not provisional:
        ctx.stamp.holds(
            Hypothesis.E7,
            f"seam {conn.seam_id}: adjoint pair by construction and the probed symmetric "
            "part is positive semidefinite",
        )
    elif op.passivity_defect is not None:
        rec.decertify(
            "L4", "E7/passivity",
            "the probed symmetric part is positive semidefinite, but the probe ran on a "
            "PROVISIONAL interface space, so it certifies nothing about a transfer pair "
            "that was never declared. E7 stays unchecked",
            subject=conn.seam_id, quantity="passivity",
        )

    # A static port still owes the drift number: it is one extra assembly and it
    # prices the cached-S economics that the whole argument for probing rests on.
    # This compile assembles once, so the drift stays OUTSTANDING rather than
    # being recorded as a measurement whose value is None.
    ctx.holes.activate(INTERFACE_MOTION, conn.seam_id, motion_class="static")
    ctx.holes.measure(
        INTERFACE_MOTION, "reprobe_count", 1,
        note="one assembly this compile; the drift needs a second at t + K dt",
    )

    if op.cut_score is not None:
        rec.admit(
            "L4", "G5/cut-score",
            f"cut score {op.cut_score:.4g} (lower is better; [AI Inference] that this is "
            "the right scalarization). Cut quality is measurable before any rollout from "
            "quantities the probe already returns",
            subject=conn.seam_id, cut_score=op.cut_score, beta=op.beta, kappa=op.kappa,
        )


# --- L5, L7 -- the interface solve, the accelerator, and the clocks --------


def _l5_l7_scheme(ctx: _Context) -> Scheme:
    rec, graph, budget = ctx.record, ctx.graph, ctx.budget
    why: dict[str, str] = {}
    defaulted: list[str] = []

    # transmission: R1, lifted by R2.
    all_caps = [a.capabilities for a in graph.agents]
    transmission = ctx.transmission or Transmission.DIRICHLET
    probeable = all(c.can_be_probed for c in all_caps)
    why["Lambda_transmission"] = RULES["R2"] if probeable else RULES["R1"]

    # The decomposition axis was decided with the rung, before L2's cross-point check.
    decomposition = ctx.decomposition or graph.decomposition
    why["D_decomposition"] = (
        "probed-DtN requires the non-overlapping view, which INTRODUCES cross-points the "
        "overlapping scheme does not have"
        if transmission is Transmission.PROBED_DTN
        else "as declared"
    )

    # ordering: additive by rule, always.
    ordering = Ordering.ADDITIVE
    why["O_ordering"] = RULES["R5"]

    # accelerator: R6 first, then the budget.
    # R6 is decided by whether the assembled interface system has an empty block,
    # not by whether every block is empty: a direct solve over a system with a zero
    # block is singular, and iterating it converges instantly on those degrees of
    # freedom and means nothing.
    empty_seams = [s for s, op in ctx.operators.items() if op.is_empty]
    if empty_seams:
        accelerator = Accelerator.RICHARDSON
        why["K_accelerator"] = (
            "R6 refuses newton-krylov and direct-schur here: at least one seam's "
            "transmission operator is identically zero, so the system is EMPTY there, "
            "not stiff. Richardson at k = 1 survives and is a single pass"
        )
        rec.refuse(
            "L5", "R6",
            "newton-krylov and direct-schur are refused: " + RULES["R6"]
            + f". Seams {empty_seams} assemble to a zero block, so the interface system "
            "is singular there. Iterating an empty interface problem converges instantly "
            "and means nothing -- the degenerate-axis trap, generalized: an axis whose "
            "value cannot affect the answer must be refused when the configuration is "
            "read, not discovered by measuring it",
            subject="<graph>", quantity="gamma", empty_seams=empty_seams,
        )
    elif budget.allow_direct_schur and ctx.operators:
        accelerator = Accelerator.DIRECT_SCHUR
        why["K_accelerator"] = (
            "a direct Schur solve makes the solve-incompleteness term vanish, and is "
            "order-free so it satisfies R5 by construction rather than by discipline"
        )
    elif ctx.operators:
        accelerator = Accelerator.KRYLOV
        why["K_accelerator"] = (
            "Krylov on the interface: the convergence rate is governed by the "
            "CONDITIONING of the transmission operator, not by subdomain count"
        )
    else:
        accelerator = Accelerator.RICHARDSON
        why["K_accelerator"] = "no assembled operator, so no accelerator can be justified"
        defaulted.append("K_accelerator")

    levels = (
        Levels.TWO_PROBED_SCHUR
        if transmission is Transmission.PROBED_DTN and ctx.operators
        else Levels.ONE
    )
    why["C_levels"] = (
        RULES["R8"] if levels is not Levels.ONE
        else "one level; a classical coarse solve would require an agent that can run at "
             "a different window size without changing its regime"
    )

    # window: R3, R4, and the W* rule that cannot be computed.
    window, w_why, w_defaulted = _choose_window(ctx)
    why["W_window"] = w_why
    if w_defaulted:
        defaulted.append("W_window")

    # tolerance: relative to the dominant terms, never absolute.
    eps_tol, tol_why, tol_defaulted = _choose_tolerance(ctx)
    why["eps_tol"] = tol_why
    if tol_defaulted:
        defaulted.append("eps_tol")

    exchange, multirate, flux_matching = _clocks(ctx)
    why["exchange_interval"] = RULES["R4"]

    scheme = Scheme(
        decomposition=decomposition,
        transmission=transmission,
        ordering=ordering,
        accelerator=accelerator,
        levels=levels,
        window=window,
        eps_tol=eps_tol,
        overlap=None if decomposition is Decomposition.NON_OVERLAPPING else graph.overlap,
        exchange_interval=exchange,
        multirate=multirate,
        flux_matching=flux_matching,
        why=why,
        defaulted=tuple(defaulted),
    )
    return scheme


def _choose_window(ctx: _Context) -> tuple[int, str, bool]:
    rec, graph = ctx.record, ctx.graph
    can_vary = [a.agent_id for a in graph.agents if not a.capabilities.bc_time_varying]
    if can_vary:
        rec.admit(
            "L7", "R3",
            f"W = 1: {can_vary} declare bc_time_varying = false, so a longer window is "
            "refused. " + RULES["R3"] + ". And this is not only an admissibility rule: it "
            "is the precondition for reducing the temporal component of the transmission "
            "term at all, so these experts carry a floor on sigma that no scheme removes",
            subject="<graph>",
        )
        return 1, "R3: an agent at this interface cannot vary its ring within a macro-step", False
    return (
        1,
        "W* would be about 1/ln(L) clamped by R3, and L is unmeasured (W1), so W* is "
        "UNCOMPUTABLE. W = 1 is the honest default and this is the compiler saying it is "
        "defaulting rather than choosing",
        True,
    )


def _choose_tolerance(ctx: _Context) -> tuple[float | None, str, bool]:
    """The tolerance is set relative to the other terms, never absolutely.

    **The W45 ingest path reaches here as of 2026-08-28, and until then it did
    not.** W45 taught `unmeasured()` to consult `graph.measured` and stopped
    there, so a graph could declare a measured tau and sigma, drop both off the
    unmeasured list, and still be decertified here by a message asserting that
    both were unmeasured. Same class of bug as the one W45 closed, one call site
    further on: an emit path with no ingest path.
    """
    m = ctx.graph.measured
    tau = None if m is None else m.tau
    sigma = None if m is None else m.sigma
    if tau is not None and sigma is not None:
        # **Corrected 2026-08-29 (W84), exposed by multiphysics attribution.**
        # `min(tau, sigma)` degenerates when a term is exactly zero, and until tau
        # became measurable at a multiphysics seam no term ever was: tau was
        # UNDEFINED across a family boundary and nonzero within one. It is zero
        # here for a composition of EXACT solvers, whose reference pair is
        # itself -- a legitimate and useful state to be able to certify, and one
        # that would set the interface tolerance to 0 and refuse every achievable
        # solve. The rule's own justification names the fix: it exists so the
        # tolerance is not driven "far below" the defects that dominate, and a
        # term that is identically zero names no scale at all. So the minimum is
        # taken over the terms that carry one.
        terms = {n: float(v) for n, v in (("tau", tau), ("sigma", sigma))
                 if float(v) > 0.0}
        if not terms:
            ctx.record.decertify(
                "L5", "eps_tol",
                f"both tau ({float(tau):.4g}) and sigma ({float(sigma):.4g}) measured "
                "as exactly zero, so neither names a scale for the interface tolerance "
                "and eps_tol has nothing to be set relative to. That is not a failure "
                "of the run -- a composition of exact solvers at a converged interface "
                "is what it looks like -- but the rule cannot fire and says so rather "
                "than emitting a tolerance of 0 that no solve can meet",
                subject="<graph>", quantity="eps_tol",
                tau=float(tau), sigma=float(sigma),
            )
            return None, "both defect terms measured zero; no scale for eps_tol", True
        eps = min(terms.values())
        zeroed = [n for n in ("tau", "sigma") if n not in terms]
        ctx.record.admit(
            "L5", "eps_tol",
            f"eps_tol = {eps:.4g} = min over the defect terms that name a scale ("
            + ", ".join(f"{n} {v:.4g}" for n, v in sorted(terms.items()))
            + (f"; {' and '.join(zeroed)} measured exactly 0, which names no scale -- "
               "the agents ARE their reference pair, so there is no agent defect for "
               "the tolerance to hide behind and it falls to the remaining term"
               if zeroed else "")
            + f"), measured at {m.probe_state!r} under scheme "
            f"{m.scheme!r} at depth {m.depth}. The rule is that the interface tolerance "
            "is set relative to the dominant terms and refused if tighter than the "
            "smaller of the agent and transmission defects: converging the interface "
            "far below them spends compute to shrink the negligible term and removes "
            "the only alarm the run has. Both are now measured, so the rule applies "
            "rather than decertifying",
            subject="<graph>", quantity="eps_tol",
            tau=float(tau), sigma=float(sigma), eps_tol=eps,
        )
        return eps, f"min over the scale-bearing defect terms = {eps:.4g}, from the graph's measured constants", False

    missing = [n for n, v in (("tau", tau), ("sigma", sigma)) if v is None]
    ctx.record.decertify(
        "L5", "eps_tol",
        "the interface tolerance must be set relative to the dominant terms and refused "
        "if tighter than the smaller of the agent and transmission defects -- converging "
        "the interface far below them spends compute to shrink the negligible term and "
        f"removes the only alarm the run has. {' and '.join(missing)} "
        f"{'is' if len(missing) == 1 else 'are'} unmeasured (W3), so the rule cannot be "
        "applied and no tolerance is set. Declare them on the graph's MeasuredConstants "
        "and this becomes an admit",
        subject="<graph>", quantity="eps_tol",
    )
    return None, f"unset: the relative rule needs measured {' and '.join(missing)} (W3)", True


def _r10b_exchange_cadence(ctx: _Context, exchange: float | None) -> float | None:
    """R10b: an exposed elliptic part must be applied at the AGENT's own cadence.

    Added 2026-08-28 from a measurement.  R10 moves a global elliptic solve out of
    the agent and into the composition layer.  The composition layer then applies
    it once per exchange -- and if that is a different cadence from the one the
    agent's own sub-stepping would have used, the composed map is a DIFFERENT
    SPLITTING of the same equations from the reference it is compared against.

    Measured on the four-window `WindowNS` tiling, varying only the macro-step:

        dt      exchanges/step   agent's own sub-steps   tau        improvement
        0.025   10               5                       1.60e-4    1.8x
        0.025   5   (matched)    5                       8.00e-7    361x
        0.050   10  (matched)    10                      1.34e-6    217x
        0.100   10               20                      1.43e-4    2.1x
        0.100   20  (matched)    20                      2.12e-6    138x

    **A factor-of-two mismatch in either direction costs two orders of magnitude,
    and it is charged to tau** -- the agent term -- while being entirely a
    property of the harness.  That is `plug-in-composition-theorems` 1.4's
    attribution theorem again, and it is the second time this project has found a
    composition-layer defect wearing an agent's label.

    The rule is a rule and not a note because the failure is silent: every
    diagnostic stays healthy, sigma stays at 1e-7, and only the comparison against
    a monolithic reference -- which a real run does not have -- reveals it.
    """
    rec, graph = ctx.record, ctx.graph
    exposed = [a for a in graph.agents
               if a.capabilities.elliptic_subsolve is EllipticSubsolve.EXPOSED]
    if not exposed:
        return exchange

    undeclared = [a.agent_id for a in exposed
                  if a.capabilities.substeps_per_macro_step is None]
    if undeclared:
        rec.decertify(
            "L7", "R10b",
            f"{undeclared} expose their elliptic part, so the composition layer applies "
            "it, and they do not declare substeps_per_macro_step -- so the cadence it "
            "must be applied at is unknown. Measured, a factor-two cadence mismatch "
            "costs two orders of magnitude in tau and every other diagnostic stays "
            "healthy. The exchange interval is left at the macro-step rather than "
            "assumed correct",
            subject=", ".join(undeclared), quantity="tau",
        )
        return exchange

    per = {a.agent_id: a.capabilities.substeps_per_macro_step for a in exposed}
    counts = sorted(set(per.values()))
    if len(counts) > 1:
        rec.refuse(
            "L7", "R10b",
            f"agents that expose their elliptic part declare different sub-step counts "
            f"{per}. The composition layer applies the elliptic part once per exchange "
            "and cannot be at two cadences at once, so one of them would be composed at "
            "the wrong splitting -- and the resulting defect is charged to that agent "
            "rather than to the harness",
            subject=", ".join(per), quantity="tau", substeps=per,
        )
        return exchange

    n = counts[0]
    if exchange is None or n <= 0:
        return exchange
    matched = exchange / n
    rec.admit(
        "L7", "R10b",
        f"exchange interval {matched:.6g} = macro-step {exchange:.6g} / {n} sub-steps: "
        "the composition layer applies the exposed elliptic part at the same cadence "
        "the agents' own sub-stepping would have. Exchanging at the macro-step instead "
        "would make the composed step a different splitting from the reference, "
        "measured at two orders of magnitude in tau and invisible to every other "
        "diagnostic",
        subject="<graph>", quantity="exchange_interval",
        macro_dt=exchange, substeps=n, exchange_interval=matched,
    )
    return matched


def _clocks(ctx: _Context) -> tuple[float | None, bool, str]:
    rec, graph, stamp = ctx.record, ctx.graph, ctx.stamp
    steps = [v for v in graph.native_steps().values() if v is not None]
    exchange = max(steps) if steps else None
    multirate = graph.is_multirate()

    if not multirate:
        exchange = _r10b_exchange_cadence(ctx, exchange)
        stamp.holds(
            Hypothesis.E4,
            "a single macro-step clock: every agent declares the same native step"
            if steps else "no native steps declared to disagree",
        )
        return exchange, False, "single-clock"

    stamp.fails(
        Hypothesis.E4,
        f"multirate: native steps {sorted(set(steps))} differ across agents",
    )
    matching = _r9_flux_matching(ctx, exchange, steps)
    rec.decertify(
        "L7", "R9/order",
        "and the accounting the mechanism does not carry: the interface representation "
        "order CAPS the scheme order -- a second-order expert coupled through an "
        "interface held constant across the macro-step is a first-order method -- while "
        "stability is set by the coupling stiffness, which can force a macro-step SHORTER "
        "than any agent's native step. That is the opposite direction from R4 and is not "
        "implied by it. [AI Inference] the relevant stiffness is the probed operator's "
        "condition number, which the probe already prints. Untested",
        subject="<graph>", quantity="scheme order",
    )
    return exchange, True, matching


def _r9_flux_matching(ctx: _Context, exchange: float | None,
                      steps: Sequence[float]) -> str:
    """R9 at a multirate seam: refuse a pointwise match, admit an integrated one.

    **2026-08-29.**  This branch has refused every multirate graph since the
    compiler existed, and the refusal named its own exit -- *"until the scheme
    declares time-integrated matching with each side's own substep quadrature"* --
    while nothing could declare it.  `graph.FluxMatching` can now, and this is
    what the declaration has to survive.

    **The leak is measured, so the refusal is no longer an argument.**  On
    `thermal_seam`'s own 500:1 mismatch, holding the interface at the tightly
    coupled trace and marching the gas over one shell step, the flux the pointwise
    match hands the shell differs from the integral the gas actually transported
    by the amount `scripts/w7_multirate_matching.py` prints -- and at ratio 1 the
    two agree to machine precision, which is the control that says the instrument
    is measuring the clocks and not itself.

    Three things are checked, and each is a way the declaration can be empty:

    1. **the quadratures nest.**  ``DT / dt_i`` must be a whole number of each
       agent's own steps, or the two sums are over intervals with different
       endpoints and there is no common integral to match.
    2. **every side can compute its integral.**  `boundary_response` is one state
       in, one flux out, restarting from the agent's own state, so calling it
       ``n`` times recomputes the first sub-step ``n`` times rather than marching
       ``n``.  The integral is not recoverable from it, which is why
       `boundary_response_integrated` exists and why its absence refuses.
    3. **the composition uses it.**  `solve.coupled_step` reads the scheme's
       ``flux_matching`` and calls the integrated response when it says so;
       admitting on a declaration the run then ignores would be the exact
       silent-wrongness class this verdict split exists to separate.
    """
    rec, graph = ctx.record, ctx.graph
    native = sorted(set(steps))
    if graph.flux_matching is not FluxMatching.TIME_INTEGRATED:
        rec.refuse(
            "L7", "R9",
            "multirate coupling requires conservation on the TIME-INTEGRATED flux over "
            "the macro-step. Pointwise-in-time flux matching across different clocks is "
            "not conservative, and the residual is computed correctly and means nothing "
            "-- the same silent signature as a bitwise-zero residual. **Measured on "
            "thermal_seam's own 500:1 mismatch** (scripts/w7_multirate_matching.py): the "
            "pointwise match hands the slow side one instantaneous flux scaled over the "
            "whole interval, and at ratio 1 the two conventions agree to machine "
            "precision, so the discrepancy is the clocks and not the instrument. Declare "
            "flux_matching=TIME_INTEGRATED and supply boundary_response_integrated on "
            "each side",
            subject="<graph>", quantity="flux residual", native_steps=native,
        )
        return "pointwise (refused; declare time-integrated)"

    quad: dict[str, int] = {}
    ragged: list[str] = []
    silent: list[str] = []
    for a in graph.agents:
        dt_i = a.capabilities.dt_native
        if dt_i is None or exchange is None or dt_i <= 0.0:
            continue
        ratio = exchange / dt_i
        n = int(round(ratio))
        if n < 1 or abs(ratio - n) > 1e-9 * max(1.0, ratio):
            ragged.append(f"{a.agent_id} ({ratio:.6g})")
        quad[a.agent_id] = max(1, n)
        if a.capabilities.boundary_response_integrated is None:
            silent.append(a.agent_id)

    if ragged:
        rec.refuse(
            "L7", "R9/nesting",
            f"time-integrated matching is declared and the clocks do not nest: "
            f"exchange interval {exchange:.6g} is not a whole number of the native step "
            f"for {', '.join(ragged)}. The two sides would then sum over intervals with "
            "different endpoints, so there is no common integral for them to agree on "
            "and the condition names nothing. This is decidable from the declared clocks "
            "alone and is decided here rather than by a run",
            subject="<graph>", quantity="substep quadrature", native_steps=native,
        )
        return "time-integrated (refused; clocks do not nest)"

    if silent:
        rec.refuse(
            "L7", "R9/quadrature",
            f"time-integrated matching is declared and {silent} supply no "
            "boundary_response_integrated, so the integral the condition matches cannot "
            "be computed. boundary_response is one state in, one flux out and restarts "
            "from the agent's own state, so calling it n times recomputes the first "
            "sub-step n times rather than marching n -- the integral is not recoverable "
            "from it. A scheme claiming to match an integral with no way to compute one "
            "is a contradiction inside the declaration, which is boundary_response_jvp's "
            "rule applied to the same shape of claim",
            subject=", ".join(silent), quantity="substep quadrature",
        )
        return "time-integrated (refused; no integrated response)"

    rec.admit(
        "L7", "R9",
        f"time-integrated flux matching over an exchange interval of {exchange:.6g}, "
        "with each side quadrature'd on its own clock: "
        + ", ".join(f"{a} x{n}" for a, n in sorted(quad.items()))
        + f". The clocks nest, every side supplies boundary_response_integrated, and "
        "`solve.coupled_step` calls it. E4 still FAILS -- the agents really do run at "
        "different steps and no declaration changes that -- and what E4's failure costs "
        "is the pointwise condition, not the coupling: the conserved quantity over the "
        "interval is the integral, and that is what is matched",
        subject="<graph>", quantity="flux residual",
        native_steps=native, substep_quadrature=dict(sorted(quad.items())),
        exchange_interval=exchange,
    )
    span = max(quad.values()) if quad else 1
    rec.decertify(
        "L7", "R9/lag",
        f"and R9 is not the multirate defect, it is the smaller one. Matching the "
        "integral fixes what the pointwise convention loses to the fast side's flux "
        "TRANSIENT; it does nothing about the fast side's trace being STALE for the "
        f"whole interval, which here is {span} of that agent's own steps. Measured on "
        "thermal_seam at its own 500:1 mismatch, both in interface power over one "
        "exchange interval (scripts/w7_multirate_matching.py, stage lagvsleak): R9's "
        "term is 5.9972e-05 and the lag term is 3.7431e-03, a factor of **62.4**. So "
        "this rule refused every multirate graph for the whole life of the compiler "
        "over the term 62x smaller than the one it does not mention. The lag term is "
        "sigma, it is a function of the lag (W86), and a multirate exchange interval is "
        "where that lag is largest -- so declare sigma with the sigma_lag it was "
        "measured at and expect it to be the term that decides this graph",
        subject="<graph>", quantity="sigma",
        exchange_interval=exchange, substep_span=span,
    )
    return "time-integrated"


# --- L6 -- assembly --------------------------------------------------------


def _l6_assembly(ctx: _Context) -> AssemblyCertificate:
    """L6, and since 2026-08-28 the layer has a condition rather than a hole.

    **R11 -- a partition of unity must be convex.** `assembly.L6_C1` carries the
    statement and the proof; the short version is that the cellwise identity

        |A(u) - u*|^2 = sum_i chi_i |u_i - u*|^2 - V_chi

    holds for any weights summing to one, and V_chi -- the chi-weighted variance
    of the local VALUES, which needs no exact solution -- is non-negative for all
    data exactly when chi >= 0. So convexity is necessary and sufficient for the
    blend to be at least as accurate as the local solves it blends, and it is
    read off the declaration.

    A non-convex partition is the silent-wrongness class: it passes the identity
    check, it passes every alpha-style diagnostic the framework had before this
    (measured: alpha is NEGATIVE for a deliberately extrapolatory partition), and
    it costs 166x on the composed macro-step. So it refuses.
    """
    rec, graph, stamp = ctx.record, ctx.graph, ctx.stamp
    pou: PartitionOfUnity | None = graph.partition_of_unity
    # W100: the constraint the agents each enforce is derived from their declared
    # governing family here, so L6/C2 and R12 read the same string and the
    # certificate carries it whether or not a projection is declared.
    cert = certify(pou, constraint=_declared_constraint(graph))
    ctx.holes.activate(ASSEMBLY_CERTIFICATE, "<assembly>")

    if pou is None:
        rec.decertify(
            "L6", "E6",
            "no assembly declared, so the partition-of-unity identity cannot be checked "
            "and ||A|| is not computed. L is not computable without ||A||, so no bound "
            "can be claimed",
            subject="<assembly>", quantity="||A||",
        )
        stamp.unchecked(Hypothesis.E6, "no assembly declared")
        ctx.holes.measure(ASSEMBLY_CERTIFICATE, "pou_residual", None)
        ctx.holes.measure(ASSEMBLY_CERTIFICATE, "norm_A", None)
        ctx.holes.measure(ASSEMBLY_CERTIFICATE, "blend_defect", None)
        ctx.holes.measure(ASSEMBLY_CERTIFICATE, "condition", None)
        ctx.holes.measure(ASSEMBLY_CERTIFICATE, "conservative", None)
        return cert

    cond = cert.condition
    if cert.identity_holds is False:
        rec.refuse(
            "L6", "E6",
            f"the partition-of-unity identity fails at {cert.pou_residual:.3e}. A "
            "partition of unity that fails this identity is producing a weighted average "
            "that is not an average. The test is one line, it costs machine precision, "
            "and its failure is exactly the silent-wrongness class",
            subject="<assembly>", quantity="assembled state",
            pou_residual=cert.pou_residual,
        )
        stamp.fails(Hypothesis.E6, f"partition-of-unity identity residual {cert.pou_residual:.3e}")
    elif not cond.convex:
        rec.refuse(
            "L6", "R11",
            f"the partition of unity is not convex: min chi = {cond.chi_min:.4g}, and "
            f"||A|| = {float(cert.norm_A):.4g} rather than 1. L6/C1 is necessary as well "
            "as sufficient -- with a negative weight the chi-weighted variance V_chi can "
            "be negative, the blend leaves the hull of the values it blends, and the "
            "quantity that would bound its error against theirs (sum_i chi_i |u_i-u*|^2) "
            "is not even non-negative. Nothing else catches this: the identity holds to "
            "machine precision and the blend defect measures NEGATIVE, i.e. passing. "
            "Measured on the four-window tiling: hull escape 1.13e-4 and a composed "
            "macro-step 166x worse than the same graph with a convex partition",
            subject="<assembly>", quantity="assembled state",
            chi_min=cond.chi_min, norm_A=float(cert.norm_A),
        )
        stamp.fails(Hypothesis.E6, f"partition not convex: min chi = {cond.chi_min:.4g} (R11)")
    else:
        rec.admit(
            "L6", "L6/C1",
            f"{cond.why()}. ||A|| = {float(cert.norm_A):.4g}, which is 1 for any convex "
            "partition of unity and is a theorem rather than a measurement. The other "
            "half of E6 -- that the blend is at least as accurate as the local solves it "
            "blends -- now HAS a condition and this graph satisfies it: it follows from "
            "convexity at every cell and in every l^p norm, with no exact solution "
            "needed anywhere",
            subject="<assembly>",
        )
        stamp.holds(
            Hypothesis.E6,
            f"identity {cert.pou_residual:.3e} and L6/C1 convexity (min chi = "
            f"{cond.chi_min:.4g}): the blend is at least as accurate as what it blends",
        )
    _r12_conservative_assembly(ctx, cert)
    _w49_sigma_branch(ctx, cond)
    ctx.holes.measure(ASSEMBLY_CERTIFICATE, "pou_residual", cert.pou_residual)
    ctx.holes.measure(ASSEMBLY_CERTIFICATE, "norm_A", cert.norm_A)
    ctx.holes.measure(
        ASSEMBLY_CERTIFICATE, "condition", cond.as_dict()["holds"],
        constrained_by="L6/C1",
        note=cond.why(),
    )
    ctx.holes.measure(
        ASSEMBLY_CERTIFICATE, "blend_defect", cert.blend_defect,
        constrained_by="L6/C1: alpha <= 0 against the cellwise max, by convexity",
        note="the reference-side diagnostic. The condition itself is checked without it, "
             "which is what the slot required",
    )
    ctx.holes.measure(
        ASSEMBLY_CERTIFICATE, "conservative", cond.conservative,
        constrained_by="L6/C2",
        note=cond.why_C2(),
    )
    return cert



#: Governing families whose agents enforce a pointwise linear constraint on the
#: state they return.  The map is a string test rather than an analysis, exactly
#: as E3's cross-family check is, and it is short on purpose: a family only
#: belongs here if an agent that solves it returns a field satisfying the
#: constraint to solver tolerance, because that is the hypothesis L6/C2 needs.
CONSTRAINED_FAMILIES: tuple[tuple[str, str], ...] = (
    ("incompressible", "divergence-free"),
)


def _constrained_family(fam: str | None) -> str | None:
    """The constraint a single ``governing_family`` string carries, if any."""
    for needle, constraint in CONSTRAINED_FAMILIES:
        if needle in (fam or ""):
            return constraint
    return None


def _declared_constraint(graph) -> str | None:
    """What the agents each enforce on their own subdomain, from the families."""
    for a in graph.agents:
        constraint = _constrained_family(a.capabilities.governing_family)
        if constraint is not None:
            return constraint
    return None


def _enforcing_agents(graph, constraint: str | None = None) -> list[str]:
    """Agents whose OWN step enforces the constraint, so the blend can break it.

    ``embedded`` says the elliptic solve is inside the agent, so the field it
    returns satisfies the constraint on its own subdomain -- which is the
    hypothesis of L6/C2 and the arrangement that goes unstable.  ``unknown``
    cannot be cleared and is counted with it, on the same discipline R10 already
    applies to it.  ``exposed`` is the opposite: the agent hands the elliptic
    part out, so its local field is NOT individually constrained, there is
    nothing for the blend to destroy, and the composition layer's single global
    application is downstream of the blend by construction.

    **W161, closed 2026-09-09: an embedded solve is not automatically THIS
    constraint's solve.**  The predicate above read ``elliptic_subsolve`` alone
    and ignored ``governing_family``, so on the front wing it returned
    ``['STRUCT']`` -- a plane-stress elasticity agent whose embedded solve is a
    **stiffness** solve, not a pressure projection.  R12 then decertified the
    assembly for applying the divergence-free projection twice, naming an agent
    that had never applied it once.

    The scope is the constraint's own derivation and not a judgement call.
    L6/C2 is the commutator identity

        C(sum_i chi_i u_i) = sum_i [C, chi_i] u_i = sum_i grad(chi_i).u_i

    a sum over the subdomains of the **partition of unity**, whose hypothesis is
    ``C u_i = 0`` on each of them.  An agent that does not carry the family the
    constraint comes from has no ``C u_i = 0`` to state and, on this graph, no
    ``chi_i`` either: the partition's subdomains are the six fluid windows and
    STRUCT is not one of them.  So the filter is on the family that carries the
    constraint, which is where the identity's index set comes from.

    ``constraint`` is passed by the caller rather than recomputed so that the
    two cannot disagree about which constraint is in play; ``None`` keeps the
    old unfiltered behaviour for callers that have no constraint in hand.
    """
    return [a.agent_id for a in graph.agents
            if a.capabilities.elliptic_subsolve in (EllipticSubsolve.EMBEDDED,
                                                    EllipticSubsolve.UNKNOWN)
            and (constraint is None
                 or _constrained_family(a.capabilities.governing_family)
                 == constraint)]


def _r12_conservative_assembly(ctx: _Context, cert) -> None:
    """R12: an assembly must preserve the constraint its agents enforce.

    **Added 2026-08-31 from the measurement that ended a rollout.**  `assembly.L6_C2`
    carries the statement; the short version is that for a linear constraint C
    with ``C u_i = 0`` on every subdomain,

        C( sum_i chi_i u_i ) = sum_i [C, chi_i] u_i = sum_i grad(chi_i) . u_i

    which is ``grad(chi_1) . (u_1 - u_2)`` on two windows, and is nonzero at cells
    where every local field satisfies the constraint EXACTLY.  So the blend
    manufactures constraint residual out of the local solves' disagreement, in
    proportion to how fast chi turns over, and no overlap width and no convexity
    removes it -- L6/C1 bounds the blend's ERROR and is silent about a constraint.

    **The repair is to MOVE the projection, not to add one, and that distinction
    is the whole rule.**  120 macro-steps from the freestream at N = 6, disks
    live, reporting the macro-step at which |u| leaves the band 3:

        agents        composition layer        leaves the band at
        embedded      nothing                  74   (not finite by 82)
        embedded      global spectral Leray    51
        embedded      global Neumann Leray     33
        EXPOSED       nothing                  never -- and ||div u|| = 1.25
        EXPOSED       global spectral Leray    never, to 120, ||div u|| = 0.066

    Adding a projection to agents that already project **applies the pressure
    twice** -- each window has already answered the disk's momentum sink on its
    own subdomain -- and measured, that is worse than doing nothing.  The
    arrangement that survives is the one R10 has prescribed since Tier 0: take
    the elliptic part out of the agent, and let the composition layer apply it
    once, to the assembled field.

    **And the divergence is not the proximate cause**, which corrects the account
    section 19.6 first gave.  The Neumann variant above holds ``||div u||_rms`` at
    **0.0089** against the bare column's **0.69** and leaves the band **soonest**
    of the three.  L6/C2's residual is real, it is what the assembly injects, and
    it is not by itself what ends the rollout.

    The verdict ladder, and why it is not one verdict:

      * **no projection, and some agent enforces the constraint internally** --
        decertify.  The blend breaks the constraint and nothing restores it.
      * **a projection declared, and some agent STILL enforces it internally** --
        decertify.  That is the arrangement above: the constraint operator runs
        twice per macro-step, measured worse than either endpoint.  The message
        names R10 as the actual repair.
      * **declared per subdomain** -- refuse.  W98's measured failure: an operator
        that cannot see across a seam cannot repair what the blend broke across
        it.
      * **declared before the assembly** -- refuse.  That IS the unrepaired
        column: every window returns a constrained field and the blend breaks it
        again.  L6/C2 is about the order.
      * **declared, global, after the assembly, with the agents elliptic parts
        exposed** -- admit.  The measured-stable arrangement.

    **R12 does not clear R10, and R10 without R12 is not enough either.**  R10
    moves the elliptic part out of the agent; R12 says where the composition
    layer must then put it.  Neither alone is the scheme: exposed agents with no
    global projection run 120 macro-steps without leaving the band and with
    ``||div u||_rms`` at 1.25, which is stable and not incompressible.
    """
    rec, graph = ctx.record, ctx.graph
    cond = cert.condition
    if graph.partition_of_unity is None or len(graph.agents) < 2:
        return
    if ctx.decomposition is not Decomposition.OVERLAPPING:
        # A single-valued-flux or substructuring assembly is not a blend, so the
        # commutator identity has no chi to be taken with. It has its own
        # conservation question and this is not it.
        return
    constraint = _declared_constraint(graph)
    if constraint is None:
        return
    enforcing = _enforcing_agents(graph, constraint)
    proj = graph.assembly_projection
    if cond is not None and cond.constraint is None:
        cond.constraint = constraint

    if proj is None:
        if not enforcing:
            # Every agent exposes its elliptic part, so no local field is
            # individually constrained and there is nothing for the blend to
            # break. R10b has already fixed the cadence the composition layer
            # applies it at; L6/C2 is satisfied by the arrangement itself.
            rec.admit(
                "L6", "R12",
                f"the constraint is {constraint} and every agent declares "
                "elliptic_subsolve=exposed, so no local field is individually "
                "constrained and the composition layer's single global "
                "application is downstream of the blend by construction. L6/C2's "
                "hypothesis -- C u_i = 0 on every subdomain -- does not hold "
                "here, so its conclusion is not needed. **What this does NOT "
                "check is that the composition layer applies that elliptic part "
                "at all**: R10b fixes the cadence it must be applied at and "
                "assumes it happens, and nothing on the graph says it does. "
                "Measured 2026-08-31 on the CS-7 ladder, the same exposed agents "
                "with no global projection run 120 macro-steps without leaving "
                "the band and with ||div u||_rms at 1.25 at six windows -- "
                "stable, and not "
                "incompressible. Declaring an assembly.ProjectedAssembly is what "
                "makes it checkable, and W105 is the row",
                subject="<assembly>", quantity="assembled state",
                constraint=constraint,
            )
            return
        rec.decertify(
            "L6", "R12",
            f"{len(enforcing)} agents enforce {constraint} on their own subdomain "
            "and the assembly declares no projection, so the blend's commutator "
            "residual sum_i grad(chi_i).u_i is created on every overlap and "
            "nothing removes it. It is manufactured where the local solves "
            "DISAGREE, at cells where each of them satisfies the constraint "
            "exactly, so no overlap width and no convexity touches it -- L6/C1 "
            "bounds the blend's error against the local errors and is silent "
            "about a constraint. Measured on the CS-7 ladder 2026-08-31: the "
            "assembled ||div u||_rms grows with the graph, 0.0132 -> 0.0957 -> "
            "0.1507 -> 0.2502 from 1 to 68 overlapping pairs, and the composed "
            "rollout is not finite by macro-step 82 at N=6 while the monolith "
            "from the same state under the same forcing holds u_max at 1.33. "
            "Declare an assembly.ProjectedAssembly with a global, after-assembly "
            "constraint projection",
            subject="<assembly>", quantity="assembled state",
            constraint=constraint, enforcing=list(enforcing),
        )
        return

    if not proj.declared_legally:
        rec.decertify(
            "L6", "R12",
            f"the assembly declares a constraint projection this framework cannot "
            f"read: {proj.why()}",
            subject="<assembly>", quantity="assembled state",
            scope=proj.scope, stage=proj.stage, cadence=proj.cadence,
        )
        return

    if not proj.global_scope:
        rec.refuse(
            "L6", "R12",
            f"the assembly declares its {proj.constraint} projection at scope "
            f"{proj.scope!r}. {proj.why()}. The residual the blend creates lives "
            "on the overlap BETWEEN two subdomains, so an operator restricted to "
            "one of them cannot see the quantity it is meant to remove; and an "
            "elliptic operator run per window on a periodic window is W98, "
            "measured at a manufactured 25% velocity deficit 3.5 D upstream of a "
            "lone turbine",
            subject="<assembly>", quantity="assembled state",
            scope=proj.scope, constraint=proj.constraint,
        )
        return

    if not proj.after_assembly:
        rec.refuse(
            "L6", "R12",
            f"the assembly declares its {proj.constraint} projection at stage "
            f"{proj.stage!r}. {proj.why()}. Projecting each local field and then "
            "blending is exactly the arrangement the measurement ends: every "
            "window returns a constrained field, the blend breaks the constraint "
            "on every overlap, and the rollout is not finite by macro-step 82 at "
            "six windows. L6/C2 is a statement about the ORDER of two operators "
            "the graph already has",
            subject="<assembly>", quantity="assembled state",
            stage=proj.stage, constraint=proj.constraint,
        )
        return

    if proj.cadence < 1:
        rec.refuse(
            "L6", "R12",
            f"the assembly declares its {proj.constraint} projection at cadence "
            f"{proj.cadence}, so it is declared and not applied. A declaration "
            "that does not run is worse than none: the compile would issue L6/C2 "
            "on it",
            subject="<assembly>", quantity="assembled state",
            cadence=proj.cadence,
        )
        return

    if proj.constraint != constraint:
        rec.decertify(
            "L6", "R12",
            f"the agents enforce {constraint} and the assembly projects onto "
            f"ker({proj.constraint}). The projection is global and after the "
            "blend, which is the arrangement L6/C2 asks for, but it is a "
            "projection onto a different kernel and nothing here establishes "
            "that it removes the residual the blend creates",
            subject="<assembly>", quantity="assembled state",
            constraint=constraint, projected=proj.constraint,
        )
        return

    if enforcing:
        rec.decertify(
            "L6", "R12",
            f"the assembly projects onto ker({constraint}) globally and after the "
            f"blend, which is the arrangement L6/C2 asks for -- and {len(enforcing)} "
            "agents ALSO enforce the constraint inside their own step, so the "
            "constraint operator runs twice per macro-step. Measured 2026-08-31 "
            "over 120 macro-steps at six windows, that is worse than either "
            "endpoint: the same agents with no global projection leave the band at "
            "74, with a global spectral projection at 51 and with a global Neumann "
            "one at 33, while the same graph with the agents elliptic parts EXPOSED "
            "and one global projection holds all 120. Each window has already "
            "answered the momentum sink on its own subdomain and the global solve "
            "answers it again. **The repair is R10, not this**: declare "
            "elliptic_subsolve=exposed and let this projection be the only one",
            subject="<assembly>", quantity="assembled state",
            constraint=constraint, enforcing=list(enforcing),
            scope=proj.scope, stage=proj.stage, cadence=proj.cadence,
        )
        return

    ell = "with every agent's elliptic part exposed"
    rec.admit(
        "L6", "R12",
        f"{cond.why_C2()}. The blend does not inherit {constraint} -- "
        "C(sum_i chi_i u_i) = sum_i grad(chi_i).u_i, nonzero wherever the local "
        "solves disagree -- and the composition layer supplies the projection "
        f"that restores it, {ell} -- so it runs ONCE per macro-step, which is what "
        "makes this the arrangement that survives. Measured on the CS-7 ladder "
        "2026-08-31 over 120 macro-steps from the freestream: this column holds at "
        "every rung, while the same agents with the elliptic part left inside them "
        "are not finite by macro-step 82 at six windows and are WORSE, not better, "
        "if a global projection is added on top of their own",
        subject="<assembly>", quantity="assembled state",
        constraint=constraint, scope=proj.scope, stage=proj.stage,
        cadence=proj.cadence,
        residual_blend=cond.constraint_residual_blend,
        residual_projected=cond.constraint_residual_projected,
    )


def _w49_sigma_branch(ctx: _Context, cond) -> None:
    """W49: which branch of the master bound's sigma term this graph is on.

    `master-error-bound` section 4 routes the transmission error through an
    interface SOLVE, and the 1/beta in it is that solve's amplifier. An
    overlapping scheme poses no interface problem, so on that branch the
    factorization is not conservative but uninformative -- measured 2026-08-27,
    it overestimates by 4.6e7. Section 4.1 now carries the overlapping form,
    ``sigma <= C_mu * Pi * ||d_lambda||``, and ``Pi`` is a property of the
    partition of unity, so it is emitted from here.
    """
    if ctx.decomposition is not Decomposition.OVERLAPPING:
        return
    pi = cond.contaminated_weight
    if pi is None:
        ctx.record.decertify(
            "L6", "W49",
            "the decomposition is overlapping, so sigma is bounded by "
            "master-error-bound 4.1's product C_mu * Pi * ||d_lambda|| rather than "
            "4's substructuring form -- and Pi, the weight the assembly gives to "
            "cells a stale artificial-boundary datum can have reached, is not "
            "declared. Declare `contaminated` on the partition of unity. Assuming "
            "the assembly is clean is what the halo rule already refuses to do",
            subject="<assembly>", quantity="sigma",
        )
        return

    # **W56, found 2026-08-28 the first time any graph reached `admit`.**  This
    # quoted the module default `C_MU_HALO` unconditionally, for graphs that
    # declare no C_mu -- so the compile reported the constant on `unmeasured` and
    # published a bound using it in the same breath.  It is the W45 / W52 bug for
    # the third time: the ingest path exists and one more call site did not use
    # it.  `holes.Unmeasured` was built so that `float(L)` RAISES rather than
    # returning a plausible number; quoting 1.2 here for a graph that never
    # measured it is the same act with the guard bypassed.
    m = ctx.graph.measured
    c_mu = None if m is None else m.C_mu
    if c_mu is None:
        ctx.record.decertify(
            "L6", "W49",
            f"Pi = {pi:.4g} is measured, and C_mu is not declared on this graph, so "
            "master-error-bound 4.1's sigma <= C_mu * Pi * ||d_lambda|| has no value "
            f"for its leading constant. {C_MU_HALO} is what it measured on "
            "`reference.WindowNS` over 16 configurations and it is NOT a default for "
            "another expert -- W55 is open precisely because one solver is one "
            "solver. Declare C_mu on MeasuredConstants",
            subject="<assembly>", quantity="sigma",
        )
        return
    ctx.record.admit(
        "L6", "W49",
        f"Pi = {pi:.4g}: the assembly gives that much weight to cells a stale "
        f"artificial-boundary datum can reach, so sigma <= {c_mu} * {pi:.4g} * "
        "||d_lambda|| on the overlapping branch (master-error-bound 4.1). Pi = 1 "
        "would mean the overlap is narrower than the agents' domain of dependence "
        "and nothing in the blend is clean; Pi = 0 would mean the partition "
        "vanishes over the whole contaminated band",
        subject="<assembly>", quantity="sigma", contaminated_weight=pi, C_mu=c_mu,
    )


# --- L9 -- typing the claims ----------------------------------------------


def _l9_claims(ctx: _Context, quantities: Sequence[QuantityClaim]) -> TypedClaim:
    rec, graph, stamp = ctx.record, ctx.graph, ctx.stamp
    L = ctx.measured_L()
    m = graph.measured
    per_step = None
    if m is not None and m.tau is not None and m.sigma is not None:
        per_step = float(m.tau) + float(m.sigma) + float(m.gamma or 0.0)
    claim = type_claim(
        list(quantities),
        L,
        per_step_defect=per_step,
        dt=ctx.budget.macro_dt or graph.macro_dt,
        K=graph.K,
        p_decline=UNMEASURED_CONSTANTS["p_decline"],
        depth=ctx.depth,
    )
    if not is_measured(L):
        stamp.unchecked(
            Hypothesis.E5,
            "L is unmeasured, so the predictability horizon is unknown and the claim "
            "cannot be typed",
        )
        rec.decertify(
            "L9", "E5",
            "UNTYPED: L has never been measured (W1, one paired rollout), so the horizon "
            "is unknown, so no claim type is decidable and every metric carries the tag. "
            "This is the state of every result currently in the vault, and the appearance "
            "is accurate",
            subject="<run>", quantity="claim type",
        )
    else:
        branch = ("unbounded (L < 1: the composed map contracts)" if L < 1.0
                  else "linear (L = 1)" if L == 1.0
                  else "exponential (L > 1)")
        stamp.holds(
            Hypothesis.E5,
            f"L = {L:.6g} measured at {m.probe_state!r} under scheme {m.scheme!r} "
            f"(depth {m.depth}); horizon branch: {branch}",
        )
        rec.admit(
            "L9", "E5",
            f"claims are typed from a MEASURED L = {L:.6g}"
            + (f" +/- {m.L_stderr:.2g}" if m.L_stderr else "")
            + f", source {m.source!r}. Horizon branch: {branch}. The constant is local to "
            "the state, scheme and depth it was measured at, all three of which are "
            "carried on the record rather than assumed to transfer",
            subject="<run>", quantity="claim type",
        )
    return claim


# --- L8 -- emit ------------------------------------------------------------


def _l8_emit(
    ctx: _Context,
    scheme: Scheme,
    certificate: AssemblyCertificate,
    claim: TypedClaim,
) -> RunArtifact:
    interface: dict[str, Any] = {
        seam: op.as_dict() for seam, op in ctx.operators.items()
    }
    artifact = RunArtifact(
        case=ctx.graph.name,
        verdict=ctx.record.verdict,
        envelope=ctx.stamp,
        decisions=ctx.record,
        scheme=scheme.as_dict() if scheme else None,
        bound_terms=BoundTerms(
            depth=ctx.depth,
            tau=UNMEASURED_CONSTANTS["tau"],
            sigma=UNMEASURED_CONSTANTS["sigma"],
            gamma=None,
            L=UNMEASURED_CONSTANTS["L"],
            # **W54.** Derived from the graph rather than declared, so it cannot
            # be forgotten and cannot disagree with what the compile actually ran.
            harness=HarnessParameters.from_graph(ctx.graph),
            note="every defect carries the depth it was measured at -- a subassembly's "
                 "transmission and solve infidelity become its AGENT infidelity one level "
                 "up, so the three-way split is not invariant under regrouping -- and "
                 "since W54 the harness parameters it is a function of, because four "
                 "composition-layer defects have worn an agent's label and the depth "
                 "caught none of them",
        ),
        interface=interface,
        assembly=certificate,
        claim=claim,
        slots=ctx.holes,
        unmeasured=ctx.unmeasured(),
        depth=ctx.depth,
        note="; ".join(ctx.refused_claims) if ctx.refused_claims else "",
    )
    artifact.validate()
    return artifact
