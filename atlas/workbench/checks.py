"""A case's sanity checks: what is measured, and the tolerance it is held to.

Every family declares its checks as data (`CheckSpec`) in its own module, with
the date the tolerance was registered and why it is that number, **before any
run of that family**.  The run fills in a `CheckResult` per check.  The
tolerance is never computed from the run and never loosened after it: a check
that fails is shown failing, with the measured value beside the tolerance
(the vault's memory "Pre-registered criteria drift in code").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CheckSpec:
    key: str
    title: str
    #: "balance" (a conservation law closes), "reference" (agreement with the
    #: full domain or a closed form), or "control" (an exact identity, such as
    #: two arms that must agree to the bit)
    kind: str
    #: the measured value must be at most this; None for an exact control
    tolerance: float | None
    registered: str
    measure: str
    why: str

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "title": self.title, "kind": self.kind,
                "tolerance": self.tolerance, "registered": self.registered,
                "measure": self.measure, "why": self.why}


@dataclass
class CheckResult:
    spec: CheckSpec
    #: the measured value (None when the arm it needs did not run)
    value: float | None
    #: True, False, or None when it could not be measured
    passed: bool | None
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        return {True: "pass", False: "FAIL", None: "not measured"}[self.passed]

    def as_dict(self) -> dict[str, Any]:
        return {**self.spec.as_dict(), "value": self.value, "passed": self.passed,
                "verdict": self.verdict, "detail": self.detail, "extra": self.extra}


def judge(spec: CheckSpec, value: float | None, detail: str = "",
          **extra: Any) -> CheckResult:
    """A measured value against its registered tolerance (at most, inclusive)."""
    if value is None:
        return CheckResult(spec, None, None, detail or "not measured in this run", extra)
    if spec.tolerance is None:
        raise ValueError(f"check {spec.key!r} is an exact control; use exact()")
    return CheckResult(spec, float(value), bool(float(value) <= spec.tolerance), detail,
                       extra)


def exact(spec: CheckSpec, holds: bool | None, detail: str = "", **extra: Any) -> CheckResult:
    """An exact control: it holds or it does not."""
    return CheckResult(spec, None, holds, detail, extra)


__all__ = ["CheckSpec", "CheckResult", "judge", "exact"]
