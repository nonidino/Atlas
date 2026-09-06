"""Beat 3 -- freeze one seam and watch the answer move.

The thesis of this whole project is that **the couplings are the physics**, not a
correction applied to it.  That is easy to assert and cheap to check: take the
composed graph, turn one seam off by taking its partner to a rigid limit, march
the same number of steps from the same field, and read the downforce.  Whatever
the number moves by is what that seam was worth.

Two limits, and each one is a parent case study:

  * **k -> infinity** removes the suspension.  The wing is bolted at a fixed
    height and the graph becomes CS-12: an elastic wing on a mount that cannot
    move.
  * **E* -> infinity** removes the structure.  The wing cannot bend and the graph
    becomes CS-10: a rigid plate on a spring over a rolling road.

So this beat is also the assembly's own gate, run live: if freezing a seam did
*not* reproduce that seam's parent, the composition would not be an assembly of
the two things it claims to be.  The static version of the same check is in
`poc2-frontwing-results` section 1, where the two limits agree with their parents
to `3.4e-10` and `6.4e-8`.

The trap this module exists to avoid
------------------------------------

**The frozen column has to sit at the height the LIVE design settles to.**
Pinning it at the reference-load release height `h0 - LOAD_REF/k` is only the
same thing when the design's own settled load happens to be near `LOAD_REF`.  At
the reference design the two agree to `4e-4`; at the constrained optimum -- which
carries 19% more load on a softer spring -- they differ by **37%**, and that 37%
would be read as the seam's worth.  Measured, it inflated the suspension seam
from 8.16% to 10.42%.

This module removes the trap rather than guarding it: it marches the live design
**first**, in the same call, so it does not have to be told the settled height --
it has just measured it.  A caller that already believes it knows the value may
pass `h_hint`, and the result reports both, because a hint that disagrees with
the measurement is worth seeing.  (It usually does disagree: the caller's
instantaneous ride height is not the height a fresh rollout settles to.)
"""
from __future__ import annotations

from typing import Callable

from ..cases import front_wing as F

#: Each limit, the knob that reaches it, and the parent it should reproduce.
LIMITS = (
    ("no suspension", "k", 1.0e9,
     "CS-12 alone: the wing still bends, but it is bolted at a fixed height",
     "the ride height stops responding to load, so the wing cannot settle "
     "deeper into ground effect as it loads up"),
    ("no structure", "e_star", 1.0e9,
     "CS-10 alone: the wing still rides, but it is a rigid plate",
     "the plate cannot deflect, so the gap under it is whatever the ride "
     "height alone makes it"),
)


def _settled(rollout, steps, u, v):
    res = rollout.run(steps=steps, u0=u, v0=v)
    s = res.settled()
    return dict(load=s["load"], h=s["h"], tip=s["tip"], vm_max=s["vm_max"],
                delta_max=s["delta_max"], ok=True)


def freeze_each_seam(design: dict, steps: int, u, v, h_hint: float | None = None,
                     tiling=None,
                     progress: Callable[[str, int, int], None] | None = None
                     ) -> dict:
    """March the live design and each frozen limit, and price each seam.

    The height the frozen suspension is pinned at is **measured here**, from the
    live column this function marches first -- see the module docstring.
    `h_hint` is optional and is reported beside the measurement rather than used.
    `progress(tag, done, total)` is called between rollouts so a caller with a
    screen can say what it is doing.
    """
    tiling = tiling or F.SINGLE_TILING
    total = 1 + len(LIMITS)

    def _ro(d):
        return F.FrontWingRollout(tiling=tiling, coupling="tight", motion=True,
                                  design=d)

    if progress:
        progress("both seams live", 0, total)
    live = _settled(_ro(dict(design)), steps, u, v)

    #: **measured, not passed.** The live column has just settled and its own
    #: height is the operating point every frozen column has to sit at.
    h_live = float(live["h"])

    frozen: dict[str, dict] = {}
    for i, (tag, knob, limit, parent, plain) in enumerate(LIMITS):
        if progress:
            progress(tag, i + 1, total)
        d = dict(design)
        d[knob] = limit
        if knob == "k":
            #: pin the frozen column at the height the LIVE design settles to,
            #: plus the sag the rigid spring still has under the reference load
            d["h0"] = float(h_live) + F.LOAD_REF / limit
        try:
            rec = _settled(_ro(d), steps, u, v)
        except RuntimeError as exc:
            #: an expert DECLINING is a result, not a failure of the ablation --
            #: at the optimum the rigid-wing column makes MORE downforce and
            #: drives itself through the suspension's declared floor
            rec = dict(ok=False, why=str(exc)[:240])
        rec.update(parent=parent, plain=plain, knob=knob, limit=limit)
        if rec.get("ok"):
            rec["worth"] = abs(rec["load"] - live["load"]) / abs(live["load"])
        frozen[tag] = rec
    if progress:
        progress("done", total, total)
    return dict(live=live, frozen=frozen, steps=steps, design=dict(design),
                h_live=h_live,
                h_hint=None if h_hint is None else float(h_hint),
                h_hint_error=(None if h_hint is None
                              else abs(float(h_hint) - h_live) / abs(h_live)),
                tiling=tiling.__class__.__name__,
                n_windows=len(tiling.names))
