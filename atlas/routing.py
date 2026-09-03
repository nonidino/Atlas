"""L1.5 -- expert-to-subdomain routing, the layer the spec flagged as missing.

end-to-end-architecture-spec §0.1 flags routing as "a real missing layer between
L2 and L1", out of scope and flagged-not-filled.  plug-in-composition-theorems §5
fills it with a criterion: the engineering classification was right about the
mechanism and wrong about the criterion -- *what to route by* is a theory
question, and the vault had already answered it twice without noticing.

    route(subdomain, regime) = argmax over the feasible set of Xi,
                               ties broken by min tau

    feasible = { ports match  AND  validity(state, regime)  AND
                 certificate valid at this regime }

and an empty feasible set is a **refusal**, not a nearest-neighbour fallback.

**Why composability and not accuracy.**  Ranking a library by benchmark accuracy
selects for exactly the property that is invisible to accuracy benchmarks, and
the substitution lemma shows the two axes can point in opposite directions.
Routing by accuracy is that substitution failure committed automatically, at
every subdomain, by the compiler.

Static routing -- decided once at compile time from the declared initial regime --
is the only mode admitted today.  Dynamic routing is a topology mutation and
inherits that slot's refusal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .capability import ExpertCapabilities
from .holes import TOPOLOGY_EVENT, NamedHoleError
from .ports import PortType


class RoutingRefused(RuntimeError):
    """No expert in the library is feasible for this subdomain and regime."""


@dataclass
class RouteCandidate:
    expert: ExpertCapabilities
    Xi: float | None = None
    tau: float | None = None
    certificate_valid_here: bool = True
    feasible: bool = True
    why_infeasible: str = ""


@dataclass
class RouteResult:
    subdomain: str
    chosen: ExpertCapabilities | None
    candidates: list[RouteCandidate] = field(default_factory=list)
    mode: str = "static"
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "subdomain": self.subdomain,
            "chosen": None if self.chosen is None else self.chosen.expert_id,
            "mode": self.mode,
            "note": self.note,
            "candidates": [
                {
                    "expert": c.expert.expert_id,
                    "Xi": c.Xi,
                    "tau": c.tau,
                    "feasible": c.feasible,
                    "why_infeasible": c.why_infeasible,
                }
                for c in self.candidates
            ],
        }


def required_port_types(required: Sequence[PortType]) -> set[PortType]:
    return set(required)


def route(
    subdomain: str,
    candidates: Sequence[ExpertCapabilities],
    required_ports: Sequence[PortType],
    state: Any = None,
    regime: Any = None,
    Xi: dict[str, float] | None = None,
    tau: dict[str, float] | None = None,
    certificate_valid: Callable[[ExpertCapabilities, Any], bool] | None = None,
) -> RouteResult:
    """Static routing. Refuses on an empty feasible set."""
    Xi = Xi or {}
    tau = tau or {}
    needed = required_port_types(required_ports)
    rows: list[RouteCandidate] = []

    for caps in candidates:
        row = RouteCandidate(expert=caps, Xi=Xi.get(caps.expert_id), tau=tau.get(caps.expert_id))
        declared = {p.port_type for p in caps.ports}
        if not needed.issubset(declared):
            row.feasible = False
            row.why_infeasible = (
                f"ports do not match: needs {sorted(t.value for t in needed)}, declares "
                f"{sorted(t.value for t in declared)}"
            )
        elif caps.validity is not None and not _validity_holds(caps, state, regime):
            row.feasible = False
            row.why_infeasible = "validity predicate declines this state and regime"
        elif certificate_valid is not None and not certificate_valid(caps, regime):
            row.feasible = False
            row.certificate_valid_here = False
            row.why_infeasible = (
                "no conformance certificate valid at this regime; a certificate is valid "
                "at a state and a regime, never globally, and one quoted outside its "
                "probe regime is the same error as a metric quoted past its horizon"
            )
        rows.append(row)

    feasible = [r for r in rows if r.feasible]
    if not feasible:
        raise RoutingRefused(
            f"routing refused for {subdomain!r}: the feasible set is empty. An empty "
            "feasible set is a refusal, not a nearest-neighbour fallback -- falling back "
            "to the closest expert is how an out-of-regime expert enters a composition "
            "with no signal.\n"
            + "\n".join(f"  {r.expert.expert_id}: {r.why_infeasible}" for r in rows)
        )

    unranked = [r for r in feasible if r.Xi is None]
    if unranked:
        return RouteResult(
            subdomain,
            None,
            rows,
            note=(
                "cannot rank: the composability index Xi is unmeasured for "
                f"{[r.expert.expert_id for r in unranked]}. Xi comes from one probe per "
                "expert; ranking by benchmark accuracy instead is the substitution "
                "failure committed automatically at every subdomain, so this returns no "
                "choice rather than a wrong one"
            ),
        )

    best = max(feasible, key=lambda r: (r.Xi, -(r.tau if r.tau is not None else float("inf"))))
    return RouteResult(
        subdomain,
        best.expert,
        rows,
        note=f"ranked by composability Xi = {best.Xi:.4g}, ties broken by min tau",
    )


def _validity_holds(caps: ExpertCapabilities, state: Any, regime: Any) -> bool:
    try:
        return bool(caps.validity(state, regime))  # type: ignore[misc]
    except TypeError:
        try:
            return bool(caps.validity(state))      # type: ignore[misc]
        except Exception:
            return True
    except Exception:
        return True


def route_dynamically(*_a: Any, **_k: Any):
    """Dynamic routing is a topology mutation, and inherits its refusal.

    **[AI Inference]** from plug-in-composition-theorems §5.2, and the strongest
    unification claimed there and the least supported: dynamic routing, abstention
    fallback and topology mutation are argued to be three names for one operation.
    If that holds the framework needs one mechanism rather than three -- and it is
    the one that has no rule.
    """
    raise NamedHoleError(
        TOPOLOGY_EVENT,
        "dynamic routing re-selects an expert mid-rollout, which changes the graph and "
        "reconciles as a restart with nonzero initial error. Static routing is the only "
        "mode admitted today",
    )
