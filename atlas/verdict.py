"""The three-verdict system and the decision record.

end-to-end-architecture-spec §0.2. Every gate in this package returns one of
three verdicts, and every decision cites the rule that produced it.

    admit              run, and the master bound applies with all constants reported
    admit-uncertified  run, but no bound is claimed; the claim carries the tag
                       naming which hypothesis is unverified
    refuse             do not run; report the rule number and the quantity that
                       would have been silently wrong

The split rule, from the spec: refuse the silent-wrongness class, decertify the
unverified-hypothesis class.  It is an [AI Inference] (spec §14 item 1) --- a
judgement about which failures are recoverable, not a derived criterion --- and
it is load-bearing everywhere in this package.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Iterable


class Verdict(enum.Enum):
    ADMIT = "admit"
    ADMIT_UNCERTIFIED = "admit-uncertified"
    REFUSE = "refuse"

    @property
    def severity(self) -> int:
        return {"admit": 0, "admit-uncertified": 1, "refuse": 2}[self.value]

    def worse_of(self, other: "Verdict") -> "Verdict":
        return self if self.severity >= other.severity else other

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.value


ADMIT = Verdict.ADMIT
ADMIT_UNCERTIFIED = Verdict.ADMIT_UNCERTIFIED
REFUSE = Verdict.REFUSE


class FailureClass(enum.Enum):
    """Which of the spec's two classes a failure belongs to.

    The class decides the verdict, not the other way round; recording it makes
    the split rule auditable rather than an author's choice per call site.
    """

    SILENT_WRONGNESS = "silent-wrongness"
    UNVERIFIED_HYPOTHESIS = "unverified-hypothesis"
    STRUCTURAL = "structural"       # loud on failure; schema and type errors
    NONE = "none"                   # an admit


@dataclass(frozen=True)
class Decision:
    """One gate outcome, with its citation.

    ``rule`` is mandatory and is the spec's own identifier --- a layer/condition
    ("L3/C5"), an admissibility rule ("R9"), an envelope hypothesis ("E7"), a
    named slot ("InterfaceMotion"), or a worklist item ("W1").  A decision with
    no citation is a decision nobody can check.
    """

    layer: str
    rule: str
    verdict: Verdict
    message: str
    failure_class: FailureClass = FailureClass.NONE
    subject: str | None = None          # agent, seam or port this is about
    quantity: str | None = None         # the number that would have been wrong
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "rule": self.rule,
            "verdict": self.verdict.value,
            "failure_class": self.failure_class.value,
            "subject": self.subject,
            "quantity": self.quantity,
            "message": self.message,
            "evidence": _jsonable(self.evidence),
        }

    def __str__(self) -> str:  # pragma: no cover - display only
        where = f" [{self.subject}]" if self.subject else ""
        return f"{self.verdict.value.upper():<18} {self.layer}/{self.rule}{where}: {self.message}"


class DecisionRecord:
    """The accumulated decisions of a compile, and the verdict they imply.

    The record is the run's ``decisions`` emit group (spec §10.2).  It is built
    even when the overall verdict is ``refuse`` --- a refusal that does not say
    which rule fired and what would have been wrong is the artifact this whole
    specification exists to stop being produced.
    """

    def __init__(self) -> None:
        self._decisions: list[Decision] = []

    def add(self, decision: Decision) -> Decision:
        self._decisions.append(decision)
        return decision

    def record(
        self,
        layer: str,
        rule: str,
        verdict: Verdict,
        message: str,
        failure_class: FailureClass = FailureClass.NONE,
        subject: str | None = None,
        quantity: str | None = None,
        **evidence: Any,
    ) -> Decision:
        return self.add(
            Decision(
                layer=layer,
                rule=rule,
                verdict=verdict,
                message=message,
                failure_class=failure_class,
                subject=subject,
                quantity=quantity,
                evidence=evidence,
            )
        )

    def refuse(self, layer: str, rule: str, message: str, **kw: Any) -> Decision:
        kw.setdefault("failure_class", FailureClass.SILENT_WRONGNESS)
        return self.record(layer, rule, REFUSE, message, **kw)

    def decertify(self, layer: str, rule: str, message: str, **kw: Any) -> Decision:
        kw.setdefault("failure_class", FailureClass.UNVERIFIED_HYPOTHESIS)
        return self.record(layer, rule, ADMIT_UNCERTIFIED, message, **kw)

    def admit(self, layer: str, rule: str, message: str, **kw: Any) -> Decision:
        return self.record(layer, rule, ADMIT, message, **kw)

    def extend(self, decisions: Iterable[Decision]) -> None:
        self._decisions.extend(decisions)

    def rescope(self, start: int, region: str) -> None:
        """Re-issue the decisions recorded since ``start`` as decisions about ``region``.

        **W189, 2026-09-10.**  A rule written for one graph-scoped subject runs
        once per region by emitting its own decisions and then re-issuing them
        here: the sentence is unchanged and prefixed with the region, a
        graph-scoped subject -- ``<graph>`` or ``<assembly>`` -- becomes that
        region's, and the evidence gains ``region``.  Agent and seam subjects stay
        as they are, because they already name something inside the region.

        One mechanism for every per-region rule, so a region-scoped decision
        cannot drift from the graph-scoped decision it is.  W136 factored
        `_decomposition_cuts` out so two rules resting on one premise could not
        come to disagree about it; this is that move applied to scope.  It is
        also what makes the equivalence control a structural fact rather than a
        hope: a graph declared per region reaches the same rules with the same
        sentences, and the only difference is which region each is about.
        """
        from dataclasses import replace

        for i in range(start, len(self._decisions)):
            d = self._decisions[i]
            subject = d.subject
            if subject in (None, "<graph>"):
                subject = f"<region:{region}>"
            elif subject == "<assembly>":
                subject = f"<assembly:{region}>"
            self._decisions[i] = replace(
                d, subject=subject, message=f"region {region!r}: {d.message}",
                evidence={**d.evidence, "region": region})

    @property
    def verdict(self) -> Verdict:
        v = ADMIT
        for d in self._decisions:
            v = v.worse_of(d.verdict)
        return v

    @property
    def decisions(self) -> list[Decision]:
        return list(self._decisions)

    def of_verdict(self, verdict: Verdict) -> list[Decision]:
        return [d for d in self._decisions if d.verdict is verdict]

    @property
    def refusals(self) -> list[Decision]:
        return self.of_verdict(REFUSE)

    @property
    def decertifications(self) -> list[Decision]:
        return self.of_verdict(ADMIT_UNCERTIFIED)

    def cited_rules(self) -> list[str]:
        seen: list[str] = []
        for d in self._decisions:
            tag = f"{d.layer}/{d.rule}"
            if tag not in seen:
                seen.append(tag)
        return seen

    def as_list(self) -> list[dict[str, Any]]:
        return [d.as_dict() for d in self._decisions]

    def __len__(self) -> int:
        return len(self._decisions)

    def __iter__(self):
        return iter(self._decisions)

    def report(self, only_failures: bool = False) -> str:
        rows = [d for d in self._decisions if not only_failures or d.verdict is not ADMIT]
        return "\n".join(str(d) for d in rows)

    def grouped(self, decisions: Iterable["Decision"] | None = None) -> list[tuple["Decision", list[str]]]:
        """Collapse identical decisions that differ only in their subject.

        One rule firing on fifteen seams is one finding, not fifteen. The full
        record keeps every row -- it is the audit trail -- but a reader needs the
        rule and the list of subjects it fired on.
        """
        rows = list(self._decisions if decisions is None else decisions)
        groups: dict[tuple[str, str, str, str], tuple[Decision, list[str]]] = {}
        for d in rows:
            key = (d.layer, d.rule, d.verdict.value, d.message)
            if key not in groups:
                groups[key] = (d, [])
            if d.subject is not None:
                groups[key][1].append(d.subject)
        return list(groups.values())


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion so a decision record can always be written out.

    Persisting the record must never fail because one evidence field held an
    array or a callable --- the record is most valuable exactly on the runs that
    went wrong.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        try:
            return _jsonable(tolist())
        except Exception:  # pragma: no cover - defensive
            pass
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:  # pragma: no cover - defensive
            pass
    return repr(obj)
