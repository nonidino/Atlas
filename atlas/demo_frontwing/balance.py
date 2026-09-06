"""Beat 5 -- the global power residual, live, as one needle.

`R(t)` is the single diagnostic that is defined identically at every seam in
every case study in this project, from a two-agent wind farm to this nine-seam
assembly (`port-algebra-atlas-0.1` section 6).  It asks one question:

> **is the composition creating or destroying energy at its seams?**

Every port is an effort-flow pair whose product is power, so the energy the
receiving subsystems gain over a macro-step has to equal the power that crossed
the interface into them, and the difference is computable from the coupling
values alone.  Here there are two receivers, so the energy is the plate's strain
energy plus the spring's, and the power is the interface power over the whole
plate, which both seams share.

Three accountings, and the gap between them is the point
--------------------------------------------------------

| what it counts | what it reads on the recorded 480-step march |
|---|---|
| **static** -- the energy change alone, with no interface power at all | `1.0000` (this is the normalising level) |
| **with the motion terms** -- the interface power the two seams actually carry | `5.93e-2` |
| **plus the half-step term** -- the mid-point correction the scheme implies | `3.30e-2` |

and once the transient has passed, `3.32e-8`.  So the needle is a real
measurement rather than a decoration: turn the motion terms off and it pegs, put
them back and it drops by a factor of 17, and by four more orders once settled.

The one that is easy to get wrong
---------------------------------

**Both receivers' energies at RELEASE, and the spring's is not zero.**  The
plate starts flat, so its strain energy at release is identically zero, which
makes it easy to carry one initial value and forget the other -- and the
suspension is already holding the car up at release, by construction, so its
energy is the largest single number in the balance.  Taking the value *after*
the first macro-step as the value *before* it put the first step's residual at
**1.81** against a static accounting's 1.00: the gate failed, on a march whose
every other step closed to `5e-8`.
"""
from __future__ import annotations

import numpy as np


class Balance:
    """A running power-residual accounting over a live march.

    `level` is the running maximum of `|dE/dt|` over this session, which is what
    the three residuals are normalised by -- the same normalisation the recorded
    march uses, except that there the maximum is over the whole 480 steps and
    here it grows as the march does.  The screen says so; a normalisation that
    is still growing is not a normalisation anybody should read three digits off.
    """

    def __init__(self, dt: float, keep: int = 400) -> None:
        self.dt = float(dt)
        self.keep = int(keep)
        self.reset()

    def reset(self, energy0: float | None = None) -> None:
        self.e_prev: float | None = energy0
        self.level = 0.0
        self.n = 0
        self.hist: list[dict] = []
        self.worst = dict(static=0.0, with_motion=0.0, corrected=0.0)

    def observe(self, strain: float, spring: float, power: float,
                half: float) -> dict | None:
        """One macro-step's contribution. Returns the row, or None on the first.

        The first call only establishes the baseline: with one energy sample
        there is no difference to take, and inventing one from a one-sided
        stencil is exactly the error the module docstring describes.
        """
        e_now = float(strain) + float(spring)
        if self.e_prev is None:
            self.e_prev = e_now
            return None
        de = (e_now - self.e_prev) / self.dt
        self.e_prev = e_now
        self.n += 1
        self.level = max(self.level, abs(de))
        lvl = self.level if self.level > 0 else 1.0
        row = dict(
            n=self.n,
            dE=de, power=float(power), half=float(half),
            static=abs(de) / lvl,
            with_motion=abs(de - float(power)) / lvl,
            corrected=abs(de - (float(power) - float(half))) / lvl,
        )
        for k in self.worst:
            self.worst[k] = max(self.worst[k], row[k])
        self.hist.append(row)
        if len(self.hist) > self.keep:
            del self.hist[:self.keep // 4]
        return row

    def payload(self, tail: int = 160) -> dict:
        rows = self.hist[-tail:]
        last = self.hist[-1] if self.hist else None
        #: the settled reading -- the last quarter of what is on hand. W140's
        #: discipline: a quantity read over a WINDOW is quoted with the window.
        q = self.hist[max(0, len(self.hist) - max(1, len(self.hist) // 4)):]
        settled = (max(r["corrected"] for r in q) if q else None)
        return dict(
            n=self.n, level=self.level,
            last=last, worst=dict(self.worst), settled=settled,
            window=len(q),
            series=[[r["static"], r["with_motion"], r["corrected"]]
                    for r in rows],
            #: the recorded 480-step march, so a viewer can tell a transient
            #: from the answer without waiting eight minutes
            recorded=dict(steps=480, static=1.0, with_motion=5.930484763405167e-2,
                          corrected=3.30463486270487e-2,
                          settled=3.322685749021862e-08,
                          factor=16.862027977386113),
        )


def strain_energy(delta, S_e0, ds) -> float:
    """The plate's strain energy, the same quadratic form `run` accumulates."""
    d = np.asarray(delta, dtype=float)
    return float(0.5 * (d @ (np.asarray(S_e0, dtype=float) @ d)) * float(ds))


def spring_energy(k: float, h0: float, h: float) -> float:
    return float(0.5 * float(k) * (float(h0) - float(h)) ** 2)
