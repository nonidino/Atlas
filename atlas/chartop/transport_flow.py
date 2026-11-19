"""The coolant's flow on a layout's pieces: a laminar stream function for any channel
the pieces tile (the coolant plan, [[chart-operator-stage-0b-and-1-plan]] section 3: "the
flow is given, not solved").

**The construction.**  On each connected channel (the coolant pieces joined by the seams
between them), with one inlet face and one outlet face on its boundary, let v solve

    div grad v = 0 in the channel,   v = 0 on the wall to the right of the flow,
    v = 1 on the wall to its left,   d v / d n = 0 on the inlet and the outlet,

the channel's harmonic across-coordinate, on the pieces' own bilinear elements (each
piece's pulled-back Laplacian, `fe.assemble` with ``J G^-1``; one sparse solve on the
merged mesh, `host.monolithic`).  The stream function is

    psi = Q f(v),   f(s) = 3 s^2 - 2 s^3,   Q = U w_in,

U the mean speed and w_in the inlet's width.  Whatever v is, psi is constant along each
wall (v is), so no flow crosses a wall; the flux through the channel is exactly Q; and
u = Q f'(v) curl v vanishes on the walls, where f'(0) = f'(1) = 0, as a laminar flow
does.  In a straight channel v is linear across it, which bilinear elements reproduce
exactly, so psi is the plane Poiseuille profile there.  At a sharp inner corner of a
bend |grad v| is singular, but f'(v) vanishes there faster, so the velocity stays
bounded.  **[AI Inference]:** in a bend it is a laminar-like profile, not the Stokes
solution, which is what "given, not solved" allows.

**Which wall is which.**  Each piece's sides run counter-clockwise, so the channel's
outer sides chain counter-clockwise round it: inlet, the right-hand wall (looking
downstream), outlet, the left-hand wall.  A channel with more than one inlet or outlet
face, or with an island, is refused by name.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import chart as ch
from . import fe
from . import transport as tr
from .host import Layout, Refused, monolithic
from .piece import PieceProblem


class FlowError(ValueError):
    """A channel whose flow this module cannot build; the message says why."""


def _side_length(P: np.ndarray) -> float:
    return float(np.sum(np.hypot(*np.diff(np.asarray(P, float), axis=0).T)))


def _components(L: Layout) -> list[list[int]]:
    adj = defaultdict(set)
    for e in L.edges:
        if e.other is not None:
            adj[e.owner[0]].add(e.other[0])
            adj[e.other[0]].add(e.owner[0])
    seen, out = set(), []
    for i in range(L.P):
        if i in seen:
            continue
        comp, stack = [], [i]
        seen.add(i)
        while stack:
            a = stack.pop()
            comp.append(a)
            for b in adj[a]:
                if b not in seen:
                    seen.add(b)
                    stack.append(b)
        out.append(sorted(comp))
    return out


def stream_functions(charts: list, face, speed: float, profile: str = "parabolic",
                     tol: float = 1e-7) -> tuple[list[np.ndarray], list[dict]]:
    """Physical stream functions (m^2/s) at every chart's nodes, the charts being one
    layout's coolant pieces.  ``face(i, s, points)`` names an outer side of the coolant:
    ``"inlet"``, ``"outlet"`` or ``"wall"``.  Returns the nodal arrays and, per channel,
    what was built (its pieces, its inlet width and flux)."""
    if not charts:
        return [], []
    probs = [PieceProblem(c, fo=None, kappa_ref=1.0, label=f"coolant {i}") for i, c in enumerate(charts)]
    outer = {}

    def record(i, s, P):
        outer[(i, s)] = (face(i, s, P), np.asarray(P, float))
        return ("insulated", None)
    L0 = Layout(probs, record, tol=tol)
    for e in L0.edges:
        if e.other is not None and not e.conforming:
            raise FlowError("the coolant's pieces must meet conformingly for its flow to be built")
    psi = [None] * len(charts)
    info = []
    for comp in _components(L0):
        sides = [(i, s) for (i, s) in outer if i in comp]
        start = {}
        for (i, s) in sides:
            a = int(L0.vertex_of[i, s])
            if a in start:
                raise FlowError("a channel whose boundary touches itself (an island or a pinch): not built")
            start[a] = (i, s)
        kinds = {k: outer[k][0] for k in sides}
        n_in = [k for k in sides if kinds[k] == "inlet"]
        n_out = [k for k in sides if kinds[k] == "outlet"]
        if not n_in or not n_out:
            raise FlowError(f"a channel of {len(comp)} coolant pieces has "
                            f"{'no inlet' if not n_in else 'no outlet'}: the flow has no direction")
        # walk the outer sides counter-clockwise from an inlet side
        order, k = [], n_in[0]
        for _ in range(len(sides) + 1):
            order.append(k)
            i, s = k
            nxt = int(L0.vertex_of[i, (s + 1) % 4])
            k = start.get(nxt)
            if k is None:
                raise FlowError("the channel's outer sides do not close into one loop")
            if k == order[0]:
                break
        if len(order) != len(sides):
            raise FlowError("the channel has more than one boundary loop (an island): not built")
        # rotate so the loop starts at the first inlet side after a non-inlet side
        lab = [kinds[k] for k in order]
        r = next(j for j in range(len(lab)) if lab[j] == "inlet" and lab[j - 1] != "inlet")
        order, lab = order[r:] + order[:r], lab[r:] + lab[:r]
        runs = []
        for k, t in zip(order, lab):
            if runs and runs[-1][0] == t:
                runs[-1][1].append(k)
            else:
                runs.append([t, [k]])
        if [t for t, _ in runs] != ["inlet", "wall", "outlet", "wall"]:
            raise FlowError("a channel must have one inlet face and one outlet face, with a wall "
                            f"on each side between them; its outer sides run {[t for t, _ in runs]}")
        right = set(runs[1][1])
        left = set(runs[3][1])
        w_in = sum(_side_length(outer[k][1]) for k in runs[0][1])
        sub = {g: j for j, g in enumerate(comp)}
        cprobs = [probs[g] for g in comp]

        def bc(i, s, P, comp=comp):
            g = comp[i]
            if (g, s) in right:
                return ("dirichlet", np.zeros(len(P)))
            if (g, s) in left:
                return ("dirichlet", np.ones(len(P)))
            return ("insulated", None)
        Lc = Layout(cprobs, bc, tol=tol)
        v = monolithic(Lc).fields
        Q = float(speed) * w_in
        for j, g in enumerate(comp):
            n = charts[g].n
            vv = np.clip(np.asarray(v[j], float).reshape(n + 1, n + 1), 0.0, 1.0)
            psi[g] = Q * tr.profile(vv, profile)
        info.append({"pieces": comp, "inlet_width": w_in, "flux": Q,
                     "inlet_sides": [list(k) for k in runs[0][1]],
                     "outlet_sides": [list(k) for k in runs[2][1]]})
    return psi, info


__all__ = ["FlowError", "stream_functions"]
