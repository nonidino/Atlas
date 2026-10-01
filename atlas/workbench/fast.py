"""The Fast example per simulation type (demo item 1.4): its bars, the card's lines,
and the judgement of a run against them.

**The bars are registered in each family module** (``FAST``), before the family's
first timed run of a Fast example, and never loosened
(`demo-fast-examples-plan` §1).  The fastest decomposed arm's speed ratio
``s = t_full / t_arm`` (each the mean time of the family's ``step`` alone, as the
workbench already reports it) must reach ``speed``, and the arm must agree with
the full domain to ``agreement`` of the field's scale.  A family registers one
bar per mechanism it may use, in the plan's order of trial (§3), and a mechanism's
agreement bar is the plan's for its class:

``P``  parallel pieces that fit in cache: the same algebra, so round-off, 1e-9
``M``  multirate, each piece at its own stable step: the scheme differs, 1e-3,
       with the family's balance check kept at its own tolerance
``S``  substructuring, independent factorizations and one interface solve: the
       same algebra, 1e-9
``X``  a split by physics: at most the number of physics, so a ceiling (O3)

**O3** (the owner, 2026-09-30): a family that cannot reach the bar loads its
fastest honest setup, and its card names what limits it.  **The card**
(§5) carries one line naming the mechanism, with the run's own numbers in it, and
the fixed line below it.
"""

from __future__ import annotations

from dataclasses import dataclass

#: O3, on every speed card (the owner's words, `demo-fast-examples-plan` §5).
FIXED_LINE = ("These are classical solvers, cut into pieces. The gain grows as the shape "
              "gets more complicated, in three dimensions, and when the pieces are "
              "learned experts, which is what Atlas is built for.")

MECHANISMS = {
    "P": "parallel pieces that fit in cache",
    "M": "multirate: each piece at its own stable time step",
    "S": "substructuring: independent factorizations, one interface solve",
    "X": "a split by physics",
}

#: The plan's registration date, quoted by every family's ``FAST``.
REGISTERED = ("2026-09-30, before the Fast example's first timed run "
              "(demo-fast-examples-plan section 1)")


@dataclass(frozen=True)
class FastBars:
    """One mechanism's bars for one family, registered before its first timed run."""

    mechanism: str                 # P, M, S or X
    speed: float                   # the least t_full / t_arm of the fastest decomposed arm
    agreement: float | None        # the most |arm - full| over the scale; None: the
    #                                family's own checks decide (a lagged split)
    agreement_of: str              # what the agreement is measured on, in words
    registered: str = REGISTERED
    ceiling: str | None = None     # O3: what limits a mechanism that cannot reach `speed`

    def __post_init__(self) -> None:
        if self.mechanism not in MECHANISMS:
            raise ValueError(f"unknown mechanism {self.mechanism!r}")
        if not self.speed > 1.0:
            raise ValueError("a speed bar at or under 1 is not 'faster'")


def agreement(checks) -> float | None:
    """A run's agreement with the full domain, read from its checks as a run record
    holds them: the family's own reference check (the multirate one first, the farm's
    power last), or 0 where the family's control is the decomposed arm equal to the
    full domain bit for bit (the sound); None for a family whose agreement is its own
    controls (a split by physics)."""
    by = {c.get("key"): c for c in checks}
    for k in ("reference_multirate", "reference", "power"):
        c = by.get(k)
        if c and c.get("value") is not None:
            return float(c["value"])
    c = by.get("bitwise")
    if c and c.get("passed") and "full domain" in (c.get("title") or ""):
        return 0.0
    return None


def judge(bars: FastBars, speedup: float | None, agreement: float | None) -> dict:
    """A run against its bars: each bar met or not, read from the run's own numbers."""
    fast = speedup is not None and speedup >= bars.speed
    if bars.agreement is None:
        close = None
    else:
        close = agreement is not None and agreement <= bars.agreement
    return {"mechanism": bars.mechanism, "speed_bar": bars.speed, "speedup": speedup,
            "fast": fast, "agreement_bar": bars.agreement, "agreement": agreement,
            "close": close, "ceiling": bars.ceiling,
            "met": bool(fast and close is not False)}


def mechanism_line(mechanism: str, **n) -> str:
    """The card's line naming the mechanism, filled with the run's own numbers
    (`demo-fast-examples-plan` §5).  Every number is passed in; none is assumed."""
    if mechanism == "P":
        return (f"Faster because the {n['pieces']} pieces run at once on {n['threads']} "
                f"threads, each a small working set.")
    if mechanism == "M":
        return (f"Faster because the {n['slow']} steps {n['ratio']:g} times longer than "
                f"the {n['fast']}: each piece takes the largest step that is stable for it.")
    if mechanism == "S":
        return (f"Faster because the {n['pieces']} pieces are factored at once; the answer "
                f"is the undivided one to round-off.")
    if mechanism == "X":
        return (f"A split by physics can at most halve the time; this one takes "
                f"{n['fraction']:.2f} of it.")
    raise ValueError(mechanism)


__all__ = ["FIXED_LINE", "MECHANISMS", "REGISTERED", "FastBars", "agreement", "judge",
           "mechanism_line"]
