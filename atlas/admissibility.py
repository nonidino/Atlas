"""L3 -- the port algebra, and what makes a connection admissible.

end-to-end-architecture-spec §5, in its 2026-08-27 amended form.

A port is a power socket with a declared shape.  "Match" once meant two
conditions -- same type, coincident geometry -- and the work of several pages was
discovering that it means eight.  Then the amendment collapsed three of the eight
into one:

    C2 (geometric compatibility), C3 (mapping class) and C6 (adjointness) are a
    single check: a common interface space M is declared for the seam and each
    side declares a stable, implementable prolongation P_i : M -> V_i.

C2 becomes the special case P_i = identity, so the framework's single most
restrictive line for plug-in use turns out to have been a restriction of the
connection rule rather than of the coupling construction.  C3 is derived rather
than declared.  C6 is the case dim M = 1.  The failure verdicts are unchanged and
the refusal now cites a *missing declaration* rather than an unbuilt mechanism.

Only one envelope hypothesis's failure is a refusal rather than a decertification
and it is E7, because a power leak is invisible in every residual the framework
reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .capability import BCChannel, ExpertCapabilities, Transmission
from .graph import CaseGraph, Connection
from .ports import PortType, check_response_half, compose_scales, spec_for
from .transfer import SeamTransfer, check_lumped_port_declares_transfer
from .verdict import ADMIT, ADMIT_UNCERTIFIED, REFUSE, DecisionRecord, FailureClass, Verdict

#: Adjointness is machine precision by construction. This tolerance exists to
#: catch a hand-written reduction, not to accommodate one.
ADJOINTNESS_TOL = 1e-10

#: A prolongation whose norm exceeds this is not "stable" in the sense the
#: revised connection rule requires. The number is a guard, not a theorem, and
#: the compiler reports the measured norm alongside any refusal it causes.
PROLONGATION_NORM_CEILING = 1e8


@dataclass
class ConditionResult:
    condition: str
    verdict: Verdict
    message: str
    failure_class: FailureClass = FailureClass.NONE
    evidence: dict[str, Any] | None = None


def check_connection(
    graph: CaseGraph,
    connection: Connection,
    transfer: SeamTransfer | None,
    required_rung: Transmission | None = None,
) -> list[ConditionResult]:
    """Run the admissibility checklist on one connection.

    Returns one result per condition, in checklist order.  The caller turns them
    into decisions; keeping the checklist pure makes it testable on a connection
    in isolation, which is how a plug-in library would use it.
    """
    results: list[ConditionResult] = []
    a_id, a_port_name = connection.a
    b_id, b_port_name = connection.b
    caps_a = graph.agent(a_id).capabilities
    caps_b = graph.agent(b_id).capabilities
    port_a = caps_a.port(a_port_name)
    port_b = caps_b.port(b_port_name)

    results.append(_c1_vocabulary(connection, port_a, port_b))
    results.append(_c2367_declared_transfer(graph, connection, transfer, caps_a, caps_b))
    results.append(_c4_nondimensionalization(connection, port_a, port_b))
    results.append(_c5_power_pairing(connection))
    results.append(_c9_response_half(connection, port_a, port_b))
    results.append(_c7_rung(connection, caps_a, caps_b, required_rung))
    results.append(_c8_validity(connection, caps_a, caps_b))
    return results


# ---------------------------------------------------------------------------


def _c1_vocabulary(connection: Connection, port_a, port_b) -> ConditionResult:
    """C1 -- both declare the port type, from the closed vocabulary."""
    if port_a.port_type is not connection.port_type or port_b.port_type is not connection.port_type:
        return ConditionResult(
            "C1",
            REFUSE,
            f"seam declares {connection.port_type.value} but the two ports declare "
            f"{port_a.port_type.value} and {port_b.port_type.value}",
            FailureClass.STRUCTURAL,
        )
    if connection.port_type is PortType.ADVEC and set(port_a.passengers) != set(port_b.passengers):
        return ConditionResult(
            "C1",
            REFUSE,
            f"ADVEC passenger lists disagree: {sorted(port_a.passengers)} against "
            f"{sorted(port_b.passengers)}. A multibond's passenger list is part of the "
            "type; a disagreement is an unnamed sixth port type in disguise, and it "
            "belongs in a PortAmendment",
            FailureClass.STRUCTURAL,
        )
    return ConditionResult(
        "C1", ADMIT, f"both sides declare {connection.port_type.value}", evidence={
            "passengers": list(port_a.passengers)
        }
    )


def _c2367_declared_transfer(
    graph: CaseGraph,
    connection: Connection,
    transfer: SeamTransfer | None,
    caps_a: ExpertCapabilities,
    caps_b: ExpertCapabilities,
) -> ConditionResult:
    """C2 + C3 + C6, collapsed: one declared object, one conformance test.

    The single check the amendment leaves: a common interface space is declared
    for the seam and each side declares a stable, implementable prolongation.
    """
    if transfer is None:
        return ConditionResult(
            "C2/C3/C6",
            REFUSE,
            "no common interface space and prolongation pair declared for this seam. "
            "This refusal cites a missing declaration, not an unbuilt mechanism: declare "
            "M and one P_i per side and the mapping class, the reduction and adjointness "
            "are all derived. Geometric coincidence is the special case P_i = identity",
            FailureClass.SILENT_WRONGNESS,
        )

    evidence: dict[str, Any] = {
        "dim_M": transfer.space.dim,
        "conforming": transfer.conforming,
    }

    lumped_problem = check_lumped_port_declares_transfer(connection.port_type, transfer.space)
    if lumped_problem:
        return ConditionResult("C2/C3/C6", REFUSE, lumped_problem, FailureClass.STRUCTURAL, evidence)

    # dim M = min_i m_i_eff. A declaration finer than the coarser side can respond
    # to buys nothing, because that side's operator returns noise on those modes.
    resolutions = graph.effective_resolutions(connection)
    known = {k: v for k, v in resolutions.items() if v is not None}
    evidence["effective_resolutions"] = resolutions
    if known and transfer.space.dim > min(known.values()):
        return ConditionResult(
            "C2/C3/C6",
            REFUSE,
            f"dim M = {transfer.space.dim} exceeds min_i m_i_eff = {min(known.values())}. "
            "The multiplier space and the probe basis are the same space; representing "
            "the trace more finely than the coarser expert can respond to buys nothing "
            "and breaks the inf-sup condition that beta measures",
            FailureClass.SILENT_WRONGNESS,
            evidence,
        )

    for agent_id, caps in ((connection.a[0], caps_a), (connection.b[0], caps_b)):
        P = transfer.prolongations.get(agent_id)
        if P is None:
            return ConditionResult(
                "C2/C3/C6",
                REFUSE,
                f"no prolongation declared for {agent_id} on this seam",
                FailureClass.SILENT_WRONGNESS,
                evidence,
            )
        norm = P.norm()
        evidence[f"norm_P[{agent_id}]"] = norm
        if not P.is_stable(PROLONGATION_NORM_CEILING):
            return ConditionResult(
                "C2/C3/C6",
                REFUSE,
                f"prolongation for {agent_id} is not stable: ||P|| = {norm:.3e}",
                FailureClass.SILENT_WRONGNESS,
                evidence,
            )
        # Implementable: the agent must be able to accept P mu as boundary data
        # and return a flux. That is a property of the record, not of the transfer.
        if caps.boundary_response is None:
            return ConditionResult(
                "C2/C3/C6",
                REFUSE,
                f"prolongation for {agent_id} is declared but not implementable: the "
                "record supplies no boundary_response, so the agent cannot accept "
                "P mu as boundary data and return a flux",
                FailureClass.STRUCTURAL,
                evidence,
            )
        resid = P.adjointness_residual(transfer.space)
        evidence[f"adjointness_residual[{agent_id}]"] = resid
        if P.uses_hand_written_reduction():
            evidence[f"hand_written_reduction[{agent_id}]"] = True
            if resid > ADJOINTNESS_TOL:
                return ConditionResult(
                    "C2/C3/C6",
                    REFUSE,
                    f"{agent_id} supplies a hand-written reduction whose adjointness "
                    f"residual is {resid:.3e}. A non-adjoint pair leaks power at the "
                    "interface and silently disables the passivity theorem: the "
                    "congruence P^* Lambda P is what carries positive-realness, and "
                    "R Lambda P with R != P^* destroys it",
                    FailureClass.SILENT_WRONGNESS,
                    evidence,
                )
        elif resid > ADJOINTNESS_TOL:
            return ConditionResult(
                "C2/C3/C6",
                REFUSE,
                f"derived adjoint for {agent_id} fails its own identity test at "
                f"{resid:.3e}; the Gram matrices are inconsistent with the declared "
                "prolongation",
                FailureClass.SILENT_WRONGNESS,
                evidence,
            )

    if not transfer.conforming:
        return ConditionResult(
            "C2/C3/C6",
            ADMIT,
            "non-conforming transfer, admitted through a declared common interface "
            "space. Adds a consistency term sigma_nc to sigma, which is not reduced by "
            "iteration and belongs to the coupling rather than to the agents. "
            "[AI Inference] that probing accommodates non-conforming interfaces is "
            "unverified and cheap to falsify: probe two agents at deliberately "
            "mismatched resolutions and watch beta",
            evidence=evidence,
        )
    return ConditionResult(
        "C2/C3/C6",
        ADMIT,
        "conforming: M = V_A = V_B and each prolongation is the identity",
        evidence=evidence,
    )


def _c4_nondimensionalization(connection: Connection, port_a, port_b) -> ConditionResult:
    """C4 -- the two scale sets compose, and each preserves the power bond."""
    for label, port in ((connection.a[0], port_a), (connection.b[0], port_b)):
        check = port.scale_check()
        if not check.complete:
            return ConditionResult(
                "C4",
                REFUSE,
                f"{label}: {check.detail}",
                FailureClass.STRUCTURAL,
                {"missing_scales": list(check.missing)},
            )
        if not check.ok:
            return ConditionResult(
                "C4",
                REFUSE,
                f"{label}: {check.detail}",
                FailureClass.SILENT_WRONGNESS,
                {"power_identity_residual": check.power_identity_residual},
            )
    ok, ratios, why = compose_scales(
        connection.port_type, port_a.nondim, port_b.nondim, port_a.passengers
    )
    if not ok:
        return ConditionResult("C4", REFUSE, why, FailureClass.STRUCTURAL)
    return ConditionResult(
        "C4",
        ADMIT,
        "scale sets are complete, satisfy s_e * s_f = s_P, and compose at the port",
        evidence={"conversion_ratios": ratios},
    )


def _c5_power_pairing(connection: Connection) -> ConditionResult:
    """C5 -- the power-preserving pairing, declared with an orientation.

    e_A = e_B and f_A = -f_B on the seam.  Without it there is no Dirac
    interconnection, and the passivity theorem has no hypothesis.  What is
    checkable at compile time is that an orientation exists and is unambiguous;
    that the values satisfy it is the per-port residual, measured at runtime.
    """
    if not connection.orientation:
        return ConditionResult(
            "C5",
            REFUSE,
            "no orientation declared for the seam normal. Get this wrong once and "
            "every port residual is meaningless",
            FailureClass.SILENT_WRONGNESS,
        )
    return ConditionResult(
        "C5",
        ADMIT,
        f"power-preserving pairing with orientation: {connection.orientation}",
        evidence={"enforced": connection.enforced},
    )


def _c9_response_half(connection: Connection, port_a, port_b) -> ConditionResult:
    """C9 (W66) -- both sides return the SAME half of the conjugate pair.

    C4 checks the declared scale *set* and never sees the callable; C5 checks
    that an orientation exists.  Neither establishes that what
    ``boundary_response`` actually returns is the variable the scale set says it
    is, and until 2026-08-29 nothing did.

    The check is on the *declaration*, deliberately.  Both value-based routes
    were measured and neither discriminates: a positive rescale leaves every
    field of E7 numerically unchanged, and a nondimensionalized scale set makes
    all three scales 1 so magnitude carries no information.  See
    `ports.ResponseHalf`.

    Refuses only the case it can be sure about -- two sides declaring different
    halves, whose assembled operator sums an effort and a flow.  An undeclared
    side decertifies: it is the pre-existing silence, now audible.
    """
    check = check_response_half(
        connection.port_type, port_a.response_half, port_b.response_half
    )
    evidence = {
        "response_half": check.response_half.value,
        "trace_half": check.trace_half.value,
        "declared": check.declared,
    }
    if check.declared and not check.agree:
        return ConditionResult(
            "C9", REFUSE, check.detail, FailureClass.SILENT_WRONGNESS, evidence
        )
    if not check.declared:
        return ConditionResult(
            "C9/W66",
            ADMIT_UNCERTIFIED,
            check.detail
            + " Measured: window_ns compiles to `admit` with one side of every "
              "seam returning the flow where the pair says effort.",
            FailureClass.UNVERIFIED_HYPOTHESIS,
            evidence,
        )
    return ConditionResult("C9", ADMIT, check.detail, evidence=evidence)


def _c7_rung(
    connection: Connection,
    caps_a: ExpertCapabilities,
    caps_b: ExpertCapabilities,
    required: Transmission | None,
) -> ConditionResult:
    """C7 -- rung compatibility. R1, lifted by R2.

    Not a refusal: if the achievable rung is below what the case needs, sigma
    will be large and the run should say so, not stop.
    """
    achievable = achievable_rung(caps_a, caps_b)
    evidence = {
        "bc_channel": {caps_a.expert_id: caps_a.bc_channel.value, caps_b.expert_id: caps_b.bc_channel.value},
        "achievable": achievable.value,
        "probeable": {caps_a.expert_id: caps_a.can_be_probed, caps_b.expert_id: caps_b.can_be_probed},
    }
    if required is not None and achievable.rung < required.rung:
        return ConditionResult(
            "C7",
            ADMIT_UNCERTIFIED,
            f"the case asks for {required.value} and the agents support at most "
            f"{achievable.value}; sigma will be large and the run reports it rather "
            "than stopping",
            FailureClass.UNVERIFIED_HYPOTHESIS,
            evidence,
        )
    return ConditionResult("C7", ADMIT, f"achievable rung: {achievable.value}", evidence=evidence)


def achievable_rung(*caps: ExpertCapabilities) -> Transmission:
    """R1 lifted by R2.

    R1: the interface is as weak as its weakest agent.
    R2: probed-DtN is the exception -- it requires only that every side accept a
        Dirichlet trace, because the higher rung is built OUTSIDE the expert from
        Dirichlet-in / flux-out probes. This is what decouples the achievable rung
        from the expert's own interface and keeps the ladder open to black boxes.
    """
    if all(c.can_be_probed for c in caps):
        return Transmission.PROBED_DTN
    weakest = min(c.bc_channel.rung for c in caps)
    return {
        BCChannel.NONE.rung: Transmission.DIRICHLET,
        BCChannel.DIRICHLET.rung: Transmission.DIRICHLET,
        BCChannel.NEUMANN.rung: Transmission.DIRICHLET_NEUMANN,
        BCChannel.ROBIN.rung: Transmission.ROBIN,
        BCChannel.VENTCELL.rung: Transmission.VENTCELL,
    }[weakest]


def _c8_validity(
    connection: Connection,
    caps_a: ExpertCapabilities,
    caps_b: ExpertCapabilities,
) -> ConditionResult:
    """C8 -- validity overlap, and the abstention hook it needs somewhere to live.

    The connection is *legal*; whether it is *valid* is decided per step, at
    runtime, by the predicate.  C8 is the condition that gets worse as the
    framework gets better: a clean contract makes more connections legal without
    making them valid, and with K experts the probability that at least one is
    out of distribution approaches 1.
    """
    undeclared = [c.expert_id for c in (caps_a, caps_b) if not c.declares_validity]
    if undeclared:
        return ConditionResult(
            "C8",
            ADMIT_UNCERTIFIED,
            f"no validity predicate on {undeclared}: the run cannot abstain, so every "
            "bound is vacuous wherever the expert is out of distribution and nothing "
            "says where that is. The declination rate p is undefined, so the abstention "
            "horizon T_abs cannot be reported either",
            FailureClass.UNVERIFIED_HYPOTHESIS,
            {"undeclared_validity": undeclared},
        )
    return ConditionResult(
        "C8",
        ADMIT,
        "both sides declare validity; the connection carries an abstention hook and a "
        "declination halts the rollout, typing the claim on the interval before it",
    )


def apply_results(
    record: DecisionRecord,
    results: list[ConditionResult],
    subject: str,
    layer: str = "L3",
) -> Verdict:
    """Fold checklist results into the decision record and return the worst verdict."""
    worst = ADMIT
    for r in results:
        record.record(
            layer,
            r.condition,
            r.verdict,
            r.message,
            failure_class=r.failure_class,
            subject=subject,
            **(r.evidence or {}),
        )
        worst = worst.worse_of(r.verdict)
    return worst
