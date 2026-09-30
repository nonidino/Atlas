"""What a new shape or a newly chosen physics needs before it can run.

The owner's first scenario (2026-09-30): the river family chosen, a smooth blob drawn
as the river, and four problems that made no sense to someone who had only drawn a
shape -- 13,539 cells with no material, an outfall at the river's default 200 m in a
grid still cut at the wind farm's cell size (11 m by 7.5 m), no inlet, no outlet.  A case should be runnable the moment its shape is drawn, with
every assumption said and changeable.  So:

* `fill_defaults` gives a case what it lacks, and only what it lacks, when its domain
  is drawn or its physics chosen: a material under every cell; a river's inlet and
  outlet on its leftmost and rightmost edges, and its outfall in the water; a plate's
  electrodes and circuit; a structure's clamp and load; a hot end and a cold end; a
  time step under the stability limit.  Each is logged and shown to the person.
* The check marks each problem one of these repairs with its key (`Issue.fix`), and
  the page offers the repair as a button beside the problem (`apply_fix`).

Nothing the case already has is changed: a repair runs only for a thing the check
finds missing, and each is an edit like any other (undone in one step).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from . import geometry as geo
from . import registry
from . import shapes
from .spec import Attachment, CaseSpec, Region, check


def _family(c: CaseSpec):
    try:
        return registry.family(c.physics.family)
    except KeyError:
        return None


# ---------------------------------------------------------------------------
# where the ends of a shape are
# ---------------------------------------------------------------------------


def _side(c: CaseSpec, which: str) -> int | None:
    """The index in ``c.boundaries`` of the boundary on the domain's ``which`` side
    ("left" or "right"): on a drawn outline, the edge whose points lie furthest that
    way on average; on the plain grid, the first segment of that grid edge."""
    d = c.domain
    if d.outline is not None:
        o = d.outline
        rings = shapes.edges_sampled(o.points, o.kinds(), o.bulges())
        #: an edge a hole took whole bounds nothing: never an end
        live = set(geo.boundary_faces(d).edge.tolist()) if d.holes else None
        best, score = None, None
        for i, b in enumerate(c.boundaries):
            if not b.edge.startswith("outline:"):
                continue
            if live is not None and b.edge not in live:
                continue
            k = int(b.edge.split(":")[1])
            if k >= len(rings):
                continue
            x = float(np.mean(rings[k][:, 0]))
            s = x if which == "left" else -x
            if score is None or s < score:
                best, score = i, s
        return best
    for i, b in enumerate(c.boundaries):
        if b.edge == which:
            return i
    return None


def _grid_default(c: CaseSpec, which: str) -> tuple[str, float | None] | None:
    """The condition the family puts on the grid's ``which`` edge: the one its solver
    fixes, or the one a new case of it starts with."""
    fam = _family(c)
    if fam is None:
        return None
    for f in fam.fixed_boundaries:
        if f.edge == which:
            return f.kind, None
    for edge, kind, value in fam.default_boundaries:
        if edge == which:
            return kind, value
    return None


def _where(c: CaseSpec, i: int) -> str:
    b = c.boundaries[i]
    if b.edge.startswith("outline:"):
        return f"edge {b.edge.split(':')[1]} of the outline"
    return f"the grid's {b.edge} edge"


def _set_end(c: CaseSpec, which: str, kind: str, value: float | None,
             bid: str | None = None) -> str | None:
    i = _side(c, which)
    if i is None:
        return None
    b = c.boundaries[i]
    b.kind, b.value = kind, value
    if bid is not None and bid not in {x.id for x in c.boundaries}:
        b.id = bid
    return _where(c, i)


# ---------------------------------------------------------------------------
# the repairs, one per key the check may put on a problem
# ---------------------------------------------------------------------------


def _materials(c: CaseSpec) -> str | None:
    """A material under every cell: one region covering the whole grid, first in the
    stack, so every region drawn later lies on top of it."""
    fam = _family(c)
    if fam is None or "regions" not in fam.layers or not fam.material_props:
        return None
    d = c.domain
    owner = geo.region_owner(c.regions, d.nx, d.ny)
    if not np.any((owner < 0) & geo.domain_mask(d)):
        return None
    lib = fam.material_library()
    solid = [m for m in c.materials if not c.materials[m].get("flows")]
    solid += [m for m in lib if not lib[m].get("flows") and m not in solid]
    m = solid[0] if solid else "material-1"
    if m not in c.materials:
        c.materials[m] = dict(lib.get(m) or {p.name: p.default for p in fam.material_props})
    taken = {r.id for r in c.regions}
    rid, k = "base", 2
    while rid in taken:
        rid, k = f"base-{k}", k + 1
    c.regions.insert(0, Region(id=rid, material=m, x0=0, y0=0, nx=d.nx, ny=d.ny))
    return f"{m} under the whole domain (Materials)"


def _river_inlet(c: CaseSpec) -> str | None:
    if any(b.kind == "river-inlet" for b in c.boundaries):
        return None
    where = _set_end(c, "left", "river-inlet", None)
    return where and f"the river's inlet on {where}, the leftmost (Boundaries)"


def _river_outlet(c: CaseSpec) -> str | None:
    if any(b.kind == "river-outlet" for b in c.boundaries):
        return None
    where = _set_end(c, "right", "river-outlet", None)
    return where and f"its outlet on {where}, the rightmost (Boundaries)"


def _electrodes(c: CaseSpec) -> str | None:
    if any(b.kind == "electrode" for b in c.boundaries):
        return None
    left = _set_end(c, "left", "electrode", None, "E1")
    right = _set_end(c, "right", "electrode", None, "E2")
    if not (left and right):
        return None
    return f"electrodes E1 on {left} and E2 on {right} (Boundaries)"


def _circuit(c: CaseSpec) -> str | None:
    """A 12 V battery with 0.5 ohm inside and a 1 ohm resistor, from one electrode to
    the other through the plate: the showcase circuit's values."""
    if any(a.kind == "battery" for a in c.attachments):
        return None
    ids = list(dict.fromkeys(b.id for b in c.boundaries if b.kind == "electrode"))
    if len(ids) < 2:                     # the current must pass through the plate
        return None
    first, last = ids[0], ids[-1]
    taken = {a.id for a in c.attachments} | {n for a in c.attachments for n in (a.a, a.b)}

    def free(prefix):
        k = 1
        while f"{prefix}{k}" in taken:
            k += 1
        taken.add(f"{prefix}{k}")
        return f"{prefix}{k}"
    node, bid, rid = free("n"), free("B"), free("R")
    # the showcase circuit's loop: + terminal, the resistor, one electrode, the
    # plate, the other electrode, back to the - terminal
    c.attachments.append(Attachment(id=bid, kind="battery", value=12.0, internal=0.5,
                                    a=node, b=last))
    c.attachments.append(Attachment(id=rid, kind="resistor", value=1.0, a=node, b=first))
    return (f"a 12 V battery and a 1 ohm resistor wired from {first} to {last} "
            f"(Circuit)")


def _clamp(c: CaseSpec) -> str | None:
    if any(b.kind == "clamped" for b in c.boundaries):
        return None
    where = _set_end(c, "left", "clamped", None)
    return where and f"{where}, the leftmost, clamped (Boundaries)"


def _load(c: CaseSpec) -> str | None:
    if any(b.kind in ("load-x", "load-y") and b.value for b in c.boundaries):
        return None
    kind, value = _grid_default(c, "right") or ("load-y", -5.0e6)
    if kind not in ("load-x", "load-y"):
        kind, value = "load-y", -5.0e6
    where = _set_end(c, "right", kind, value)
    return where and f"{where}, the rightmost, loaded ({kind} {value:g} Pa) (Boundaries)"


def _temperatures(c: CaseSpec) -> str | None:
    """A hot end and, where the family has one, a cold end: the family's own left and
    right conditions on a rectangle, put on the shape's leftmost and rightmost edges."""
    if any(b.kind == "fixed-temperature" for b in c.boundaries):
        return None
    out = []
    for which in ("left", "right"):
        want = _grid_default(c, which)
        if want and want[0] == "fixed-temperature":
            where = _set_end(c, which, "fixed-temperature", want[1])
            if where:
                out.append(f"{where} held at {want[1]:g} K")
    return "; ".join(out) + " (Boundaries)" if out else None


def _outfall(c: CaseSpec) -> str | None:
    """The river's outfall in the water: the domain's cell nearest a point a fifth of
    the way from its inlet to its outlet (or its middle, without them)."""
    if c.physics.family != "transport-2d":
        return None
    d = c.domain
    sx, sy = c.physics.get("source_x"), c.physics.get("source_y")
    act = geo.domain_mask(d)
    i, j = int(math.floor(sx / d.dx)), int(math.floor(sy / d.dx))
    if 0 <= i < d.nx and 0 <= j < d.ny and act[j, i]:
        return None
    ys, xs = np.nonzero(act)
    if xs.size == 0:
        return None
    tx, ty = float(xs.mean()) + 0.5, float(ys.mean()) + 0.5
    ends = [_side(c, "left"), _side(c, "right")]
    kinds = {b.kind: n for n, b in enumerate(c.boundaries)}
    if d.outline is not None and "river-inlet" in kinds and "river-outlet" in kinds:
        o = d.outline
        rings = shapes.edges_sampled(o.points, o.kinds(), o.bulges())

        def mid(n):
            e = c.boundaries[n].edge
            return (np.mean(rings[int(e.split(":")[1])], axis=0)
                    if e.startswith("outline:") else None)
        a, b = mid(kinds["river-inlet"]), mid(kinds["river-outlet"])
        if a is not None and b is not None:
            tx, ty = a[0] + 0.2 * (b[0] - a[0]), a[1] + 0.2 * (b[1] - a[1])
    elif d.outline is None and all(e is not None for e in ends):
        tx = 0.2 * d.nx                                  # the plain river runs along x
    k = int(np.argmin((xs + 0.5 - tx) ** 2 + (ys + 0.5 - ty) ** 2))
    x, y = (xs[k] + 0.5) * d.dx, (ys[k] + 0.5) * d.dx
    c.physics.params["source_x"], c.physics.params["source_y"] = float(x), float(y)
    return f"the outfall at ({x:g} m, {y:g} m), in the water near the inlet (Physics)"


def _round_down(x: float) -> float:
    """``x`` to two significant figures, never above it."""
    if x <= 0.0:
        return x
    e = math.floor(math.log10(x)) - 1
    return math.floor(x / 10.0 ** e) * 10.0 ** e


def _time_step(c: CaseSpec) -> str | None:
    """A macro-step 90% of the explicit step's stability limit, when it is over it."""
    try:
        if c.physics.family == "transport-2d":
            from .families.plume import explicit_limit
            lim = explicit_limit(c)
        elif c.physics.family == "acoustics-2d":
            from .families.acoustics import stable_dt
            lim = stable_dt(c)
        else:
            return None
    except Exception:                                    # the check says why
        return None
    if not (lim > 0.0) or c.run.macro_dt <= lim:
        return None
    c.run.macro_dt = _round_down(0.9 * lim)
    return (f"a macro-step of {c.run.macro_dt:g} s, under the stability limit "
            f"{lim:.3g} s (Physics)")


def _windows(c: CaseSpec) -> str | None:
    """Windows cut from the shape (the automatic layout, `layout.py`), as many as the
    style couples: one for a field on a circuit or a split by physics, two for
    Dirichlet-Neumann, two or more for the overlapping styles."""
    from . import layout as lay
    from .spec import Layout, Window
    d = c.domain
    if c.physics.family == "incompressible-2d" and d.outline is None and not d.holes:
        # the wind farm on the whole grid: the scaling ladder's own rectangles, 3 x 2
        # overlapping by twice the ramp, the tiling its compile recognises
        try:
            tiles = geo.tile(d.nx, d.ny, 3, 2, 2 * c.coupling.ramp_cells)
        except ValueError:
            tiles = None
        if tiles:
            c.layout = None
            c.windows = [Window(id=n, x0=b[0], y0=b[1], nx=b[2], ny=b[3]) for n, b in tiles]
            return f"{len(tiles)} rectangular windows, 3 x 2 (Windows)"
    style = c.coupling.style
    n = {"D": 1, "split": 1, "C": 2}.get(style, max(2, min(len(c.windows) or 2, 6)))
    before = (c.layout, list(c.windows))
    tries = [Layout(cut="along", along=n, across=1)]
    if style == "C" and _two_media(c):
        # Dirichlet-Neumann across the interface the materials (or the coolant and
        # the block) already have, as a newly drawn domain's windows are
        tries.insert(0, Layout(cut="materials"))
    for layout in tries:
        c.layout = layout
        try:
            lay.refresh(c)
        except lay.LayoutError:
            continue
        k = len(c.windows)
        how = ("one per material" if layout.cut == "materials" else
               "cut automatically from the shape")
        return f"{k} window{'s' * (k != 1)} {how} (Windows)"
    c.layout, c.windows = before
    return None


def _two_media(c: CaseSpec) -> bool:
    """Whether the domain holds two media a style-C cut can follow: two materials, or,
    for the cooled block, coolant beside solid."""
    d = c.domain
    act = geo.domain_mask(d)
    if c.physics.family == "conjugate-heat-2d":
        from .families.cooling import coolant_mask
        try:
            wet = coolant_mask(c)
        except Exception:                                # the check says why
            return False
        return bool(wet.any() and (act & ~wet).any())
    owner = geo.region_owner(c.regions, d.nx, d.ny)
    ks = np.unique(owner[act])
    return len({c.regions[k].material for k in ks.tolist() if k >= 0}) >= 2


def _swallowed(c: CaseSpec) -> str | None:
    """A condition on an edge a hole took whole, moved to the nearest edge that still
    bounds the domain and carries only the family's default.  The two boundaries swap
    edges, so the condition keeps its id: an electrode stays wired to its circuit."""
    fam = _family(c)
    if fam is None or fam.drawn_derived or not fam.boundary_kinds:
        return None
    d = c.domain
    live = set(geo.boundary_faces(d).edge.tolist())
    parts: dict[str, list[np.ndarray]] = {}
    for name, poly in geo.live_edges(d):
        parts.setdefault(name, []).append(poly)
    full = dict(geo.drawn_edges(d))
    done = []
    for b in c.boundaries:
        if not b.drawn or b.edge in live or b.kind == fam.drawn_default or b.edge not in full:
            continue
        src = full[b.edge]
        best, dist = None, None
        for other in c.boundaries:
            if (not other.drawn or other.edge not in live or other.kind != fam.drawn_default
                    or other.edge not in parts):
                continue
            pts = np.concatenate(parts[other.edge])
            dd = float(np.min(np.hypot(pts[:, None, 0] - src[None, :, 0],
                                       pts[:, None, 1] - src[None, :, 1])))
            if dist is None or dd < dist:
                best, dist = other, dd
        if best is None:
            continue
        gone = b.edge
        b.edge, best.edge = best.edge, gone
        live.discard(b.edge)                      # taken: the next moves elsewhere
        done.append(f"the {b.kind} {b.id} moved from {gone}, which the hole removed, to "
                    f"{b.edge}")
    return "; ".join(done) + " (Boundaries)" if done else None


#: key -> (the button's label, the repair)
FIXES: dict[str, tuple[str, Callable[[CaseSpec], str | None]]] = {
    "windows": ("Cut the windows automatically", _windows),
    "materials": ("Fill every cell with a material", _materials),
    "river-inlet": ("Make the leftmost edge the inlet", _river_inlet),
    "river-outlet": ("Make the rightmost edge the outlet", _river_outlet),
    "electrodes": ("Make the leftmost and rightmost edges electrodes", _electrodes),
    "circuit": ("Wire a battery and a resistor", _circuit),
    "clamp": ("Clamp the leftmost edge", _clamp),
    "load": ("Load the rightmost edge", _load),
    "temperatures": ("Hold the ends at the family's temperatures", _temperatures),
    "outfall": ("Move it into the water", _outfall),
    "time-step": ("Use a stable macro-step", _time_step),
    "swallowed": ("Move it to the nearest edge left", _swallowed),
}


#: repairs that move something the person set rather than give what is missing: the
#: problems list offers them, and a newly drawn shape never makes them unasked
ASKED_ONLY = frozenset({"swallowed"})


def fix_label(key: str) -> str:
    return FIXES[key][0]


def apply_fix(c: CaseSpec, key: str) -> str | None:
    """Repair one problem in place; what was done, or None when nothing was needed.
    A circuit needs its electrodes first, so that repair brings them along."""
    done = []
    if key == "circuit":
        e = _electrodes(c)
        if e:
            done.append(e)
    r = FIXES[key][1](c)
    if r:
        done.append(r)
    return "; ".join(done) or None


def new_case(fid: str) -> CaseSpec:
    """A new case of the physics ``fid``, ready to run before any shape is drawn: the
    whole grid, the physics' example's setup (`spec.adapt_to_family`), and what it
    then lacks filled in.  A blank case was a wind farm cut into no windows, an error
    from its first moment."""
    return new_case_from(fid)[0]


def new_case_from(fid: str) -> tuple[CaseSpec, str | None, list[str]]:
    """`new_case`; the example it had to start from instead of the whole grid (None when
    it did not); and what it was given to start (`fill_defaults`), for the page to say."""
    from . import layout as lay
    from .spec import EXAMPLES, adapt_to_family, blank_case, example_case
    c = blank_case()
    if fid != c.physics.family:
        adapt_to_family(c, fid)
    if c.layout is not None:
        lay.refresh(c)
    filled = fill_defaults(c)
    if any(i.severity == "error" for i in check(c)):
        # a physics whose whole grid cannot run as it stands (a cooled block needs its
        # channel drawn) starts from its first example instead
        key = next((k for k, ex in EXAMPLES.items() if ex.family == fid), None)
        if key is not None:
            return example_case(key), key, []
    return c, None, filled


def fill_defaults(c: CaseSpec, rounds: int = 4) -> list[str]:
    """Everything the check can repair, repaired, in place: what a newly drawn shape
    or a newly chosen physics needs to run.  Each repair can reveal the next (the
    river's materials make its time step checkable), so the check runs again until
    nothing more is offered.  Returns what was done, in order."""
    done: list[str] = []
    tried: set[str] = set()
    for _ in range(rounds):
        keys = []
        for i in check(c):
            if (i.fix and i.fix in FIXES and i.fix not in ASKED_ONLY and i.fix not in tried
                    and i.fix not in keys):
                keys.append(i.fix)
        if not keys:
            break
        for key in keys:
            tried.add(key)
            r = apply_fix(c, key)
            if r:
                done.append(r)
    return done


__all__ = ["FIXES", "ASKED_ONLY", "fix_label", "apply_fix", "fill_defaults", "new_case",
           "new_case_from"]
