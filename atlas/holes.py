"""The five named holes, and the unmeasured constants.

end-to-end-architecture-spec §12.  There are five things this architecture
cannot do and does not know how to do.  Each gets a slot rather than an
assumption, specified by five fields --- and the fifth is the one that makes a
hole useful rather than merely honest:

    1. what is missing            stated as the hypothesis whose removal creates it
    2. the declared interface     the signature a solution must present
    3. default behaviour today    always a refusal or a decertification
    4. what a solution must satisfy
    5. the measurement required now   the number the missing rule would constrain

Nothing here fills a hole.  ``NamedHole.solve`` raises; the slots exist so the
layers around them can be built against a declared interface, and so the field-5
measurement is emitted from every run that touches the slot.

The unmeasured constants (L, beta, sigma, ...) are a separate mechanism in the
same spirit: ``Unmeasured`` is a value that refuses to be used in arithmetic,
so a bound cannot be quoted from a constant nobody measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .verdict import ADMIT, ADMIT_UNCERTIFIED, REFUSE, FailureClass, Verdict, _jsonable


class NamedHoleError(NotImplementedError):
    """Raised when a caller asks a named slot to supply the rule it does not have."""

    def __init__(self, hole: "NamedHole", what: str = "") -> None:
        detail = f" ({what})" if what else ""
        super().__init__(
            f"{hole.name} is a named hole, not a rule{detail}. "
            f"Gap {hole.gap}: {hole.what_is_missing} "
            f"Default today: {hole.default_verdict.value}. "
            f"See end-to-end-architecture-spec {hole.section}."
        )
        self.hole = hole


class UnmeasuredError(ArithmeticError):
    """Raised when an unmeasured constant is used as if it had a value."""


@dataclass(frozen=True)
class Unmeasured:
    """A constant the framework needs and has never measured.

    It deliberately does not behave like a number.  An unmeasured L is the reason
    every result in the vault types as UNTYPED (spec §11.1); silently
    substituting a plausible default is the failure this class exists to make
    impossible.
    """

    symbol: str
    worklist_item: str
    why: str = ""

    def _refuse(self, *_a: Any, **_k: Any) -> float:
        raise UnmeasuredError(
            f"{self.symbol} has never been measured ({self.worklist_item}"
            + (f": {self.why}" if self.why else "")
            + "). Refusing to supply a value."
        )

    __float__ = _refuse
    __int__ = _refuse
    __add__ = __radd__ = _refuse
    __sub__ = __rsub__ = _refuse
    __mul__ = __rmul__ = _refuse
    __truediv__ = __rtruediv__ = _refuse
    __pow__ = __rpow__ = _refuse
    __lt__ = __gt__ = __le__ = __ge__ = _refuse

    def __bool__(self) -> bool:
        return False

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"UNMEASURED({self.symbol}, {self.worklist_item})"


#: The constants the master bound needs.  Every one is `open` on the gap
#: worklist; the compiler reports which ones a given run still lacks.
UNMEASURED_CONSTANTS: dict[str, Unmeasured] = {
    "L": Unmeasured("L", "W1", "the stability constant; fit from one paired rollout"),
    "beta": Unmeasured("beta", "W2", "sigma_min of the assembled interface operator; needs a probe"),
    "sigma": Unmeasured("sigma", "W3", "transmission infidelity; needs two assemblies"),
    "tau": Unmeasured("tau", "W3", "agent infidelity; needs a validated classical reference"),
    "norm_A": Unmeasured("norm_A", "W28", "the assembly norm; enters L multiplicatively"),
    "C_mu": Unmeasured("C_mu", "W3", "sensitivity of the composed local solve to its interface datum"),
    "p_decline": Unmeasured("p", "W36", "per-agent per-macro-step declination rate; needs W13"),
    "K_ood_threshold": Unmeasured(
        "K_ood",
        "unset",
        "the graph size at which P(some agent OOD) approaches 1; spec L1 requires a "
        "refusal above it and does not set it",
    ),
}


def is_measured(value: Any) -> bool:
    """True when a constant carries a usable number rather than an Unmeasured marker."""
    return value is not None and not isinstance(value, Unmeasured)


@dataclass
class HoleMeasurement:
    """Field 5: the number the missing rule will constrain, emitted now."""

    hole: str
    name: str
    value: Any
    units: str = ""
    constrained_by: str | None = None   # None is the point: nothing constrains it yet
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "hole": self.hole,
            "name": self.name,
            "value": _jsonable(self.value),
            "units": self.units,
            "constrained_by": self.constrained_by,
            "note": self.note,
        }


@dataclass(frozen=True)
class NamedHole:
    """A hole with a declared interface and no solution."""

    name: str
    gap: str
    section: str
    what_is_missing: str
    declared_interface: dict[str, str]
    default_verdict: Verdict
    failure_class: FailureClass
    must_satisfy: tuple[str, ...]
    measurements_required: tuple[str, ...]
    inference_note: str = ""

    def solve(self, *_args: Any, **_kw: Any) -> Any:
        raise NamedHoleError(self)

    def touched(self, subject: str, **evidence: Any) -> dict[str, Any]:
        """Record that a configuration reached this slot, for the emit contract."""
        return {
            "hole": self.name,
            "gap": self.gap,
            "subject": subject,
            "verdict": self.default_verdict.value,
            "measurements_required": list(self.measurements_required),
            "evidence": _jsonable(evidence),
        }


SEAM_REFERENCE = NamedHole(
    name="SeamReference",
    gap="O2 / G2",
    section="§12.1",
    what_is_missing=(
        "E3 assumes a single global evolution operator whose restrictions the agents "
        "approximate. Across a fluid-structure or fluid-chemistry seam no such operator "
        "exists, so agent infidelity tau has no referent."
    ),
    declared_interface={
        "tau_referent": "reference object against which each side's tau is defined",
        "scope": "interface_only | interface_and_interior",
        "cost": "compute estimate",
        "availability": "bool  # G19: a reference may not exist at all",
    },
    default_verdict=ADMIT_UNCERTIFIED,
    failure_class=FailureClass.UNVERIFIED_HYPOTHESIS,
    must_satisfy=(
        "reduces to the existing definition when both sides share a governing family",
        "measurable without a monolithic solve of the whole system",
        "supplies the symbol an optimized transmission condition is derived from, "
        "or explains why that symbol is not needed",
    ),
    measurements_required=(
        "tau_undefined_seams",
        "one_sided_operator_surrogate",
    ),
    inference_note=(
        "[AI Inference] spec §6.4(b): probing never mentions a governing equation, so the "
        "INTERFACE half of agent infidelity is definable today from a one-sided reference "
        "operator; only the INTERIOR half has no referent. The hole is real and smaller "
        "than stated."
    ),
)

INTERFACE_MOTION = NamedHole(
    name="InterfaceMotion",
    gap="O4 / G17",
    section="§12.2",
    what_is_missing=(
        "E2 assumes static geometry. Any interface that moves changes the interface "
        "operator every step and invalidates a cached S, which is the entire economic "
        "argument for probed-DtN coupling."
    ),
    declared_interface={
        "motion_class": "static | prescribed | solution_dependent",
        "Gamma_of_t": "callable | none  # prescribed only",
        "staleness": "predicate(S_cached, t) -> bool",
        "reprobe_cost": "m+1 solves per affected block",
        "conservation": "how flux is accounted for on a sweeping interface",
    },
    default_verdict=REFUSE,
    failure_class=FailureClass.SILENT_WRONGNESS,
    must_satisfy=(
        "flux accounting across a sweeping interface must conserve; the interface does "
        "work on the region it sweeps and that term is absent from the power residual",
        "the staleness predicate must be cheaper than the re-probe it gates",
        "solution_dependent motion makes the interface solve a system in (lambda, Gamma), "
        "a structural change to L5 rather than a parameter change",
    ),
    measurements_required=(
        "reprobe_count",
        "operator_drift",
        "unaccounted_power_fraction",
    ),
)

TOPOLOGY_EVENT = NamedHole(
    name="TopologyEvent",
    gap="O5 / G9",
    section="§12.3",
    what_is_missing=(
        "Staging, docking and contact make/break change which agents exist. That "
        "reconciles with a fixed-graph bound as a restart with nonzero initial error. "
        "What has no rule is conservation ACROSS the event."
    ),
    declared_interface={
        "graph_before": "G-",
        "graph_after": "G+",
        "state_map": "M : u- -> u+",
        "edge_reinstantiation": "which ports are created, destroyed, rewired",
        "ledger": "per conserved functional, the jump dC_k = C_k(u+) - C_k(u-)",
        "e0_reset": "the restart error injected at the event time",
    },
    default_verdict=REFUSE,
    failure_class=FailureClass.SILENT_WRONGNESS,
    must_satisfy=(
        "says which functionals must have zero jump and which are legitimately "
        "discontinuous (a jettisoned stage takes its mass and momentum with it)",
        "interacts correctly with the pooling hierarchy, whose clustering computed "
        "before the event is not valid after it",
        "specifies what happens to a cached S on every port touching a mutated node",
    ),
    measurements_required=("ledger", "injected_e0", "macro_steps_since_last_event"),
    inference_note=(
        "[AI Inference] plug-in-composition-theorems §5.2: dynamic routing, abstention "
        "fallback and topology mutation may be one operation, which would make this slot "
        "the mechanism every mid-rollout change routes through."
    ),
)

ASSEMBLY_CERTIFICATE = NamedHole(
    name="AssemblyCertificate",
    gap="O3 / G12",
    section="§12.4",
    what_is_missing=(
        "CLOSED 2026-08-28 as L6/C1, and enforced as R11. The condition that the blend be "
        "at least as accurate as the local solves it blends is a condition on chi alone: "
        "chi >= 0 with sum_i chi_i = 1. It follows from the cellwise identity "
        "|A(u)-u*|^2 = sum_i chi_i |u_i-u*|^2 - V_chi, which holds for any weights "
        "summing to one, and whose chi-weighted variance V_chi is non-negative for all "
        "data exactly when chi >= 0. Convexity is therefore necessary as well as "
        "sufficient, and it is read off the declaration with no reference solution."
    ),
    declared_interface={
        "pou_residual": "norm of (sum_i R_i^T chi_i R_i - I)  # CHECKABLE TODAY",
        "norm_A": "the assembly norm; enters L multiplicatively. 1 for any CONVEX "
                  "partition of unity; measured 1.7038 on a signed one",
        "blend_defect": "alpha := ||A(u_i) - u_exact|| - || max_i |u_i - u_exact| || on "
                        "the overlap. A DIAGNOSTIC, not the condition: measured, it reads "
                        "as passing for a deliberately non-convex partition",
        "condition": "L6/C1 -- chi_ij >= 0 and sum_i chi_ij = 1. FILLED",
    },
    default_verdict=ADMIT,
    failure_class=FailureClass.UNVERIFIED_HYPOTHESIS,
    must_satisfy=(
        "checkable without the exact solution, or the certificate is a diagnostic rather "
        "than a condition -- met: chi_min and the identity residual are properties of the "
        "declaration, and V_chi and the hull escape are computable from a run with no "
        "reference",
    ),
    measurements_required=("pou_residual", "norm_A", "blend_defect", "condition"),
    inference_note=(
        "The [AI Inference] this slot carried until 2026-08-28 guessed alpha <= "
        "C(delta, chi) * max_i ||grad u_i|| with C growing as the overlap shrinks. It was "
        "the wrong shape: no inequality on alpha can be the condition, because alpha "
        "compares the blend against the local solves and a blend can beat every one of "
        "them while still leaving their hull. Measured -- both alpha definitions read as "
        "passing on the partition that R11 refuses. The overlap-width dependence the "
        "guess reached for is real but lives in the accuracy, not the admissibility: the "
        "same tiling measures tau = 5.0e-5 at zero ramp and 2.5e-7 at full-overlap ramp, "
        "all of them convex and all of them admissible."
    ),
)

PORT_AMENDMENT = NamedHole(
    name="PortAmendment",
    gap="O6 / G16",
    section="§12.5",
    what_is_missing=(
        "Five port types are closed for the cases considered. Completeness is unprovable "
        "in the abstract, so the target is a documented extension procedure, and that "
        "reframing is itself an open design question."
    ),
    declared_interface={
        "bond": "the conjugate pair (e, f) with e*f a power, in W or W/m^2",
        "mapping": "the default mapping class for the flow and for the effort",
        "transfer": "the reduction/prolongation pair, with an adjointness proof",
        "dtn_reading": "which side is imposed and which is returned, so the probe is defined",
        "distinctness": "an argument that the type is NOT a special case of an existing one",
        "exercise": "at least one interface in one case study the existing five cannot type",
    },
    default_verdict=REFUSE,
    failure_class=FailureClass.STRUCTURAL,
    must_satisfy=(
        "keeps the O(K) scaling intact; an amendment requiring a bespoke adapter per "
        "expert pair defeats the purpose and is refused on that ground alone",
    ),
    measurements_required=("advec_passenger_pressure_count",),
    inference_note=(
        "[AI Inference] the six-field checklist is a proposed procedure with no evidence "
        "it is sufficient. It has never been exercised, and the first exercise (ADVEC "
        "passengers) may well show a seventh field is needed."
    ),
)

NAMED_HOLES: dict[str, NamedHole] = {
    h.name: h
    for h in (
        SEAM_REFERENCE,
        INTERFACE_MOTION,
        TOPOLOGY_EVENT,
        ASSEMBLY_CERTIFICATE,
        PORT_AMENDMENT,
    )
}


@dataclass
class HoleLedger:
    """Every slot this run touched, and the field-5 numbers it owes."""

    activations: list[dict[str, Any]] = field(default_factory=list)
    measurements: list[HoleMeasurement] = field(default_factory=list)

    def activate(self, hole: NamedHole, subject: str, **evidence: Any) -> None:
        self.activations.append(hole.touched(subject, **evidence))

    def measure(self, hole: NamedHole, name: str, value: Any, **kw: Any) -> None:
        self.measurements.append(HoleMeasurement(hole=hole.name, name=name, value=value, **kw))

    def touched(self) -> set[str]:
        return {a["hole"] for a in self.activations}

    def outstanding(self) -> dict[str, list[str]]:
        """Measurements a touched slot requires that this run has not supplied."""
        supplied: dict[str, set[str]] = {}
        for m in self.measurements:
            supplied.setdefault(m.hole, set()).add(m.name)
        out: dict[str, list[str]] = {}
        for name in sorted(self.touched()):
            hole = NAMED_HOLES[name]
            missing = [r for r in hole.measurements_required if r not in supplied.get(name, set())]
            if missing:
                out[name] = missing
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "activations": _jsonable(self.activations),
            "measurements": [m.as_dict() for m in self.measurements],
            "outstanding": self.outstanding(),
        }
