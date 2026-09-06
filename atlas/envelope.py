"""The spine -- the seven-hypothesis envelope as a machine-checkable stamp.

end-to-end-architecture-spec §1.  theory-closure-audit §3 states the seven
hypotheses under which the master error bound is a theorem.  This module makes
them a stamp emitted by every run, with three values per hypothesis:
``holds`` (checked), ``fails`` (checked), ``unchecked``.

Read the ``unchecked`` column.  Only three hypotheses can be unchecked at all --
E5, E6, E7 -- and those are exactly the three the audit records as unverified for
the wind farm.  **E1 to E4 are static properties of a declaration and are
decidable at compile time, before any compute is spent.**  That asymmetry is the
reason the envelope belongs in the compiler rather than in a review checklist.

A rollout that emits this stamp alongside its numbers is a rollout whose reader
can tell, without reading any theory page, whether the bound applies to it.
A run that cannot produce a stamp is refused.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Iterable


class Hypothesis(enum.Enum):
    E1 = "E1"
    E2 = "E2"
    E3 = "E3"
    E4 = "E4"
    E5 = "E5"
    E6 = "E6"
    E7 = "E7"


class Status(enum.Enum):
    HOLDS = "holds"
    FAILS = "fails"
    UNCHECKED = "unchecked"


@dataclass(frozen=True)
class HypothesisSpec:
    key: Hypothesis
    statement: str
    owned_by: tuple[str, ...]
    checked_how: str
    can_be_unchecked: bool


ENVELOPE: dict[Hypothesis, HypothesisSpec] = {
    Hypothesis.E1: HypothesisSpec(
        Hypothesis.E1,
        "the interaction graph is fixed for the whole rollout",
        ("L2", "L7"),
        "static: does the declaration contain a TopologyEvent?",
        can_be_unchecked=False,
    ),
    Hypothesis.E2: HypothesisSpec(
        Hypothesis.E2,
        "interfaces are static, and transfer between the two sides is declared",
        ("L2", "L3"),
        "static: motion class per port; a declared, stable prolongation per side",
        can_be_unchecked=False,
    ),
    Hypothesis.E3: HypothesisSpec(
        Hypothesis.E3,
        "a single global evolution operator exists, so tau has a referent",
        ("L1", "L4", "L9"),
        "static: do all agents at a seam declare the same governing family?",
        can_be_unchecked=False,
    ),
    Hypothesis.E4: HypothesisSpec(
        Hypothesis.E4,
        "a single macro-step clock, or proved-conservative multirate",
        ("L7",),
        "static: are all dt_native equal? dynamic: is flux matched time-integrated?",
        can_be_unchecked=False,
    ),
    Hypothesis.E5: HypothesisSpec(
        Hypothesis.E5,
        "the claim is a trajectory claim inside the predictability horizon",
        ("L9",),
        "requires a fitted L (W1); then arithmetic",
        can_be_unchecked=True,
    ),
    Hypothesis.E6: HypothesisSpec(
        Hypothesis.E6,
        "assembly bounded with ||A|| known, and the blend at least as accurate "
        "as what it blends",
        ("L6",),
        "half checkable: the partition-of-unity identity is a machine-precision "
        "test. half open: the accuracy condition is the AssemblyCertificate slot",
        can_be_unchecked=True,
    ),
    Hypothesis.E7: HypothesisSpec(
        Hypothesis.E7,
        "the interconnection is power-preserving and every reduction/prolongation "
        "pair is adjoint",
        ("L3", "L1"),
        "adjointness is a numerical identity test; passivity is the symmetric "
        "part's spectrum on the probed block",
        can_be_unchecked=True,
    ),
}


@dataclass
class EnvelopeStamp:
    """The seven values, with the evidence that produced each.

    Every entry starts ``UNCHECKED``.  A layer that checks a hypothesis must say
    so explicitly; nothing here defaults to ``holds``, because a hypothesis that
    holds by default is the assumption this whole mechanism exists to remove.
    """

    values: dict[Hypothesis, Status] = field(
        default_factory=lambda: {h: Status.UNCHECKED for h in Hypothesis}
    )
    evidence: dict[Hypothesis, list[str]] = field(
        default_factory=lambda: {h: [] for h in Hypothesis}
    )

    def set(self, h: Hypothesis, status: Status, why: str) -> None:
        """Record a check. A ``fails`` is never overwritten by a later ``holds``.

        One seam failing E3 fails E3 for the run; a second seam that is fine does
        not repair it.
        """
        current = self.values[h]
        if current is Status.FAILS and status is not Status.FAILS:
            self.evidence[h].append(f"(also: {why})")
            return
        self.values[h] = status
        self.evidence[h].append(why)

    def holds(self, h: Hypothesis, why: str) -> None:
        self.set(h, Status.HOLDS, why)

    def fails(self, h: Hypothesis, why: str) -> None:
        self.set(h, Status.FAILS, why)

    def unchecked(self, h: Hypothesis, why: str) -> None:
        self.set(h, Status.UNCHECKED, why)

    # -- reading -----------------------------------------------------------

    def __getitem__(self, h: Hypothesis) -> Status:
        return self.values[h]

    @property
    def failing(self) -> list[Hypothesis]:
        return [h for h in Hypothesis if self.values[h] is Status.FAILS]

    @property
    def unchecked_list(self) -> list[Hypothesis]:
        return [h for h in Hypothesis if self.values[h] is Status.UNCHECKED]

    @property
    def bound_applies(self) -> bool:
        """The master bound is a theorem only when all seven hold."""
        return all(s is Status.HOLDS for s in self.values.values())

    def illegally_unchecked(self) -> list[Hypothesis]:
        """Hypotheses left unchecked that the spec says are always decidable.

        E1 to E4 are static properties of a declaration.  Leaving one unchecked
        is a defect in the compiler, not a property of the case, and this is the
        self-test that catches it.
        """
        return [
            h
            for h in self.unchecked_list
            if not ENVELOPE[h].can_be_unchecked
        ]

    def tuple(self) -> tuple[str, ...]:
        return tuple(self.values[h].value for h in Hypothesis)

    def as_dict(self) -> dict[str, Any]:
        return {
            h.value: {
                "status": self.values[h].value,
                "statement": ENVELOPE[h].statement,
                "owned_by": list(ENVELOPE[h].owned_by),
                "evidence": self.evidence[h],
            }
            for h in Hypothesis
        }

    def summary(self) -> str:
        return " ".join(f"{h.value}:{self.values[h].value}" for h in Hypothesis)

    def report(self) -> str:
        lines = ["envelope stamp (E1..E7)"]
        for h in Hypothesis:
            lines.append(f"  {h.value} {self.values[h].value:<10} {ENVELOPE[h].statement}")
            for e in self.evidence[h]:
                lines.append(f"        - {e}")
        return "\n".join(lines)
