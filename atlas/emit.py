"""L8 -- the emitted diagnostic set. The emit line is the contract.

end-to-end-architecture-spec §10, extended by plug-in-composition-theorems §1.4.

The framework's most reliable failure has not been computing the wrong thing.  It
has been **reporting the term that was easy to compute** -- and every quantity
here is a by-product of machinery the coupled solve already builds.  The one
sentence that is the practical content of the master bound: *the framework
measures the smallest term, controls the second-largest, and has never reported
the other two.*  This layer is the whole of the fix.

Eight groups, and five of them are new with the spec.  One field is newer still:
**every defect carries the depth it was measured at**, because a subassembly's
transmission and solve infidelity become its agent infidelity one level up, so a
tau reported at depth 2 and a sigma reported at depth 1 are not independent terms
and must not be compared as though they were.

A run that cannot produce the envelope stamp is refused, because a result whose
reader cannot tell whether the bound applies is the artifact the specification
exists to stop being produced.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .assembly import AssemblyCertificate
from .claims import TypedClaim
from .envelope import EnvelopeStamp
from .holes import HoleLedger, Unmeasured, is_measured
from .verdict import DecisionRecord, Verdict, _jsonable


class EmitRefused(RuntimeError):
    """A run artifact that cannot carry a stamp is not an artifact."""


@dataclass(frozen=True)
class HarnessParameters:
    """**W54, closed 2026-08-30.** What the composition layer did, beside the depth.

    A defect has always carried the depth it was measured at, because a
    subassembly's transmission infidelity becomes its agent infidelity one level
    up.  That tag is necessary and it is not sufficient, and the evidence is a
    tally rather than an argument: **four composition-layer defects have now worn
    an agent's label**, and the depth caught none of them --

      1. the decomposed pressure solve, 99.8% of the first composed step's error,
         reading as agent infidelity (Tier 0);
      2. a partition of unity at full weight right up to the artificial edge,
         a factor of 200 in ``tau`` against a ramped one (section 9.1);
      3. the exchange cadence, two orders in ``tau`` with every probe diagnostic
         unchanged, from hard-coding ``substeps_per_macro_step`` (mistake 5);
      4. **W98** -- transport and pressure run per window and periodically, which
         manufactured a 25% velocity deficit 3.5 D UPSTREAM of a lone turbine.

    Every one of the four is a *function of a harness parameter* and none of them
    is a property of any agent.  So the emitted defect carries them: change one,
    and the attribution is a different measurement wearing the same name.

    ``elliptic_placement`` is where the elliptic part is applied -- ``agent``
    when the expert keeps it (an EMBEDDED subsolve), ``composition`` when the
    composition layer supplies it (an EXPOSED one, R10b), ``mixed`` when the
    graph does both.  ``chi_shape`` names the partition of unity's profile,
    because "convex" is admissibility and the *ramp* is accuracy (L6/C1's box).
    """

    overlap_cells: int | None = None
    chi_shape: str | None = None
    exchange_dt: float | None = None
    exchange_cadence: int | None = None
    elliptic_placement: str | None = None
    #: **W100, the fifth instance and the first to end a rollout.** Whether the
    #: assembly applies a constraint projection to the ASSEMBLED field, as
    #: ``"<constraint> projection, <scope>, <stage>, <cadence>x per exchange"``,
    #: or ``"none"``.  It is a harness parameter of exactly the W54 class: it
    #: belongs to no agent and it is invisible in every diagnostic the framework
    #: had.  Measured over 120 macro-steps at six windows, moving it decides
    #: whether the column exists: with the agents elliptic parts EXPOSED it is
    #: stable to 120, and with them embedded the same declaration makes the column
    #: WORSE than no projection at all (band left at 51 against 74), because the
    #: constraint operator is then applied twice.
    assembly_projection: str | None = None
    extra: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_graph(cls, graph: Any) -> "HarnessParameters":
        """Read the harness off a `CaseGraph`, so it is derived and not declared.

        Duck-typed rather than imported, to keep `emit` below `graph` in the
        dependency order -- the same reason `_jsonable` lives in `verdict`.
        """
        pou = getattr(graph, "partition_of_unity", None)
        chi = None
        if pou is not None:
            chi = getattr(pou, "kind", "partition-of-unity")
            ramp = getattr(pou, "ramp_cells", None)
            if ramp is not None:
                chi = f"{chi}, ramp {ramp} cells"
        placements = set()
        for agent in getattr(graph, "agents", ()) or ():
            caps = getattr(agent, "capabilities", None)
            ell = getattr(caps, "elliptic_subsolve", None)
            val = getattr(ell, "value", ell)
            if val in ("embedded", "unknown"):
                placements.add("agent")
            elif val == "exposed":
                placements.add("composition")
        cadences = {getattr(a.capabilities, "substeps_per_macro_step", None)
                    for a in (getattr(graph, "agents", ()) or ())}
        cadences.discard(None)
        proj = getattr(pou, "projection", None)
        return cls(
            overlap_cells=getattr(graph, "overlap_cells", None),
            chi_shape=chi,
            exchange_dt=getattr(graph, "macro_dt", None),
            exchange_cadence=(int(max(cadences)) if cadences else None),
            elliptic_placement=(
                "mixed" if len(placements) > 1
                else (next(iter(placements)) if placements else None)),
            # W100: derived from the assembly the graph declares, so a run that
            # applied the projection cannot emit a defect that says it did not,
            # and a run that did not cannot emit one that says it did.
            assembly_projection=(
                "none" if proj is None else
                f"{proj.constraint} projection, {proj.scope}, {proj.stage}, "
                f"{proj.cadence}x per exchange"),
        )

    def key(self) -> tuple:
        """The tuple two runs must share before their attributions are comparable."""
        return (self.overlap_cells, self.chi_shape, self.exchange_dt,
                self.exchange_cadence, self.elliptic_placement,
                self.assembly_projection, self.extra)

    def as_dict(self) -> dict[str, Any]:
        return {
            "overlap_cells": self.overlap_cells,
            "chi_shape": self.chi_shape,
            "exchange_dt": self.exchange_dt,
            "exchange_cadence": self.exchange_cadence,
            "elliptic_placement": self.elliptic_placement,
            "assembly_projection": self.assembly_projection,
            **{str(k): _jsonable(v) for k, v in self.extra},
        }


@dataclass
class BoundTerms:
    """The four constants of the master bound, each tagged with its depth.

    Unmeasured is a legitimate value here and is the honest one for most of them
    today.  What is not legitimate is a number with no depth -- **or, since W54,
    a measured number with no harness.**
    """

    depth: int = 0
    tau: float | Unmeasured | None = None
    sigma: float | Unmeasured | None = None
    gamma: float | Unmeasured | None = None
    L: float | Unmeasured | None = None
    sigma_nc: float | None = None          # the non-conforming interface-space component
    sigma_time: float | None = None        # the temporal-representation component
    #: **W54.**  The composition layer's own settings, which every one of the four
    #: mislabelled defects was a function of.  Required whenever any of tau,
    #: sigma or gamma is a measured number; `RunArtifact.validate` refuses
    #: otherwise, because the whole content of the row is that a defect quoted
    #: without its harness is not attributable.
    harness: HarnessParameters | None = None
    note: str = ""

    @property
    def per_step_defect(self) -> float | None:
        parts = [self.tau, self.sigma, self.gamma]
        if not all(is_measured(p) for p in parts):
            return None
        return float(sum(float(p) for p in parts))  # type: ignore[arg-type]

    @property
    def measured_terms(self) -> list[str]:
        """Which of the three defect terms carry an actual number."""
        return [k for k in ("tau", "sigma", "gamma")
                if is_measured(getattr(self, k))]

    def as_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "tau": _jsonable(self.tau),
            "sigma": _jsonable(self.sigma),
            "gamma": _jsonable(self.gamma),
            "L": _jsonable(self.L),
            "sigma_nc": self.sigma_nc,
            "sigma_time": self.sigma_time,
            "harness": None if self.harness is None else self.harness.as_dict(),
            "per_step_defect": self.per_step_defect,
            "note": self.note,
        }


@dataclass
class ConservationReport:
    """Per-port residuals and the one global power residual.

    The power residual is the cheapest possible physical-plausibility monitor and
    it needs no reference data: a nonzero value means the coupling is creating or
    destroying energy, and the per-port breakdown localizes where.
    """

    per_port: dict[str, float] = field(default_factory=dict)
    power_residual: float | None = None
    unaccounted_at_moving_seams: float | None = None
    #: **W86.**  The lag each seam actually carried over the last macro-step, and
    #: how it compares against the lag the graph's quoted ``sigma`` was measured
    #: at.  Derived from the run's own consecutive traces rather than declared:
    #: `rollout` already carried them from step to step and nothing read them.
    #: Empty when fewer than two steps ran, because a lag needs two traces.
    lag: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "per_port_residual": dict(self.per_port),
            "power_residual": self.power_residual,
            "unaccounted_at_moving_seams": self.unaccounted_at_moving_seams,
            "lag": dict(self.lag),
        }


@dataclass
class RunArtifact:
    """Everything one run emits. Built even when the verdict is refuse."""

    case: str
    verdict: Verdict
    envelope: EnvelopeStamp
    decisions: DecisionRecord
    scheme: dict[str, Any] | None = None

    bound_terms: BoundTerms = field(default_factory=BoundTerms)
    interface: dict[str, Any] = field(default_factory=dict)
    conservation: ConservationReport = field(default_factory=ConservationReport)
    assembly: AssemblyCertificate = field(default_factory=AssemblyCertificate)
    claim: TypedClaim | None = None
    slots: HoleLedger = field(default_factory=HoleLedger)

    unmeasured: list[str] = field(default_factory=list)
    conformance: list[dict[str, Any]] = field(default_factory=list)
    substitutions: list[dict[str, Any]] = field(default_factory=list)
    declination_rates: dict[str, Any] = field(default_factory=dict)
    depth: int = 0
    note: str = ""

    def validate(self) -> None:
        """A run that cannot emit the stamp is refused."""
        if self.envelope is None:
            raise EmitRefused(
                f"{self.case}: no envelope stamp. A result whose reader cannot tell "
                "whether the bound applies to it is the artifact this specification "
                "exists to stop being produced"
            )
        # **W54.** A measured defect with no harness is not attributable, and
        # this is the point at which it would leave the process.
        if self.bound_terms.measured_terms and self.bound_terms.harness is None:
            raise EmitRefused(
                f"{self.case}: {', '.join(self.bound_terms.measured_terms)} carries a "
                "measured value and no harness parameters. Every composition-layer "
                "defect this project has mislabelled -- the decomposed pressure solve, "
                "the partition of unity at full weight, the exchange cadence, and W98's "
                "per-window transport -- was a function of the overlap, chi's shape, the "
                "exchange cadence or the elliptic placement, and the depth tag caught "
                "none of them. Set BoundTerms.harness (HarnessParameters.from_graph "
                "derives it)"
            )
        illegal = self.envelope.illegally_unchecked()
        if illegal:
            raise EmitRefused(
                f"{self.case}: {[h.value for h in illegal]} left unchecked, but they are "
                "static properties of a declaration and are decidable at compile time. "
                "That is a defect in the compiler, not a property of the case"
            )

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "case": self.case,
            "verdict": self.verdict.value,
            "depth": self.depth,
            "scheme": self.scheme,
            "bound_terms": self.bound_terms.as_dict(),
            "interface": _jsonable(self.interface),
            "conservation": self.conservation.as_dict(),
            "assembly": self.assembly.as_dict(),
            "envelope": self.envelope.as_dict(),
            "envelope_stamp": list(self.envelope.tuple()),
            "claim": None if self.claim is None else self.claim.as_dict(),
            "slots": self.slots.as_dict(),
            "decisions": self.decisions.as_list(),
            "unmeasured_constants": list(self.unmeasured),
            "conformance": _jsonable(self.conformance),
            "substitutions": _jsonable(self.substitutions),
            "declination_rates": _jsonable(self.declination_rates),
            "note": self.note,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=False)

    def write(self, path: str) -> str:
        """Persist the artifact. Always safe to call, especially on a refusal."""
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())
        return path

    def report(self) -> str:
        lines = [
            f"=== {self.case} ===",
            f"verdict: {self.verdict.value}   depth: {self.depth}",
            "",
            self.envelope.summary(),
        ]
        if self.claim is not None:
            lines.append(f"claim: {self.claim.tag.value} -- {self.claim.reason}")
        if self.unmeasured:
            lines.append(f"unmeasured constants: {', '.join(self.unmeasured)}")
        touched = sorted(self.slots.touched())
        if touched:
            lines.append(f"named holes touched: {', '.join(touched)}")
        outstanding = self.slots.outstanding()
        if outstanding:
            lines.append(f"slot measurements still owed: {outstanding}")
        for label, rows in (
            ("refusals", self.decisions.refusals),
            ("decertifications", self.decisions.decertifications),
        ):
            if not rows:
                continue
            groups = self.decisions.grouped(rows)
            lines.append("")
            lines.append(f"{label} ({len(rows)} across {len(groups)} rules):")
            for d, subjects in groups:
                where = ""
                if subjects:
                    shown = ", ".join(subjects[:6])
                    more = f" +{len(subjects) - 6} more" if len(subjects) > 6 else ""
                    where = f"  [{shown}{more}]"
                lines.append(f"  {d.layer}/{d.rule}{where}")
                lines.append(f"      {d.message}")
        return "\n".join(lines)


#: The emit contract, as a checklist. Used by the tests to assert that the
#: artifact carries every group -- the contract is the point, not the numbers.
EMIT_GROUPS: dict[str, str] = {
    "bound_terms": "tau, sigma, gamma, L -- with a depth tag on every one, and "
                   "since W54 the harness parameters every measured one is a "
                   "function of: overlap, chi's shape, exchange cadence, elliptic "
                   "placement",
    "interface": "beta, kappa, null-space dimension, passivity defect and its "
                 "eigenvector, Xi, the per-mode optimal Robin coefficient",
    "conservation": "per-port residual and the global power residual",
    "assembly": "||A||, the partition-of-unity identity residual, the certificate fields",
    "envelope": "the seven-hypothesis stamp",
    "claim": "the claim type, and the per-claim horizon table",
    "slots": "re-probe count and staleness, mutation ledger rows, the tau-UNDEFINED "
             "seam list",
    "decisions": "the refusal and decertification record, each entry citing its rule",
}
