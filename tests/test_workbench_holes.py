"""Holes that cross the domain's edge or reach past the grid (2026-09-30).

The owner: "Holes should be allowed to intersect the boundary's edge. Also, they
should also be allowed to extend beyond the geometry's edge, as long as the geometry
itself is a closed shape."  Two rules of `spec._check_drawn` refused both; they are
lifted.  The domain is its cells and a hole only removes cells, so the solvers do not
change; what changed is the bookkeeping that refers back to the drawing
(`geometry.live_edges`).  These tests pin:

* a hole across the outline and past the grid leaves the case ready, and every face
  takes the condition of an edge that still bounds the domain;
* where no hole crosses anything, every face keeps its label to the bit;
* the one-window control on a domain with a crossing hole, in every family, bit for
  bit wherever the discretization is identical, and the family's own checks where
  it is not;
* a plate loaded on what a hole leaves of its edge balances; a conduction case whose
  right edge a hole cuts reports the heat through each piece;
* a condition on an edge a hole took whole is a problem with a one-click fix; a hole
  that cuts the domain in two is refused; cutting across reads holes from the mask.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from atlas.workbench import fv  # noqa: E402
from atlas.workbench import geometry as geo  # noqa: E402
from atlas.workbench import layout as lay  # noqa: E402
from atlas.workbench import shapes as S  # noqa: E402
from atlas.workbench.runner import adapter_for, arms_for  # noqa: E402
from atlas.workbench.spec import (EXAMPLES, Outline, Window, check,  # noqa: E402
                                  derive_boundaries, example_case)
from atlas.workbench.starter import apply_fix, fill_defaults  # noqa: E402


def _errors(spec):
    return [i.message for i in check(spec) if i.severity == "error"]


def _cut(spec, hole: Outline):
    """A hole cut as the page cuts one: its edges take the family's own condition, and
    windows that follow the domain are cut again."""
    from atlas.workbench.editor import rebase_boundaries
    h = len(spec.domain.holes)
    spec.domain.holes.append(hole)
    rebase_boundaries(spec, f"hole{h}", len(hole.points), None)
    derive_boundaries(spec)
    if spec.layout is not None:
        lay.refresh(spec)
    return spec


def _march(spec, steps, arms=None, threads=2):
    mod = adapter_for(spec.physics.family)
    arms = arms or arms_for(mod, spec)[0]
    r = mod.build(spec, arms=arms, threads=threads)
    st = {a: r.initial(a) for a in r.arms}
    hist = {a: [] for a in r.arms}
    same = True
    for _ in range(steps):
        for a in r.arms:
            prev = st[a]
            st[a] = r.step(a, prev)
            hist[a].append(r.observe(a, st[a], prev))
        if "parallel" in st:
            same &= r.bitwise_equal(st["serial"], st["parallel"])
    metrics, checks = r.compare(st, hist, {"steps": steps,
                                           "first_difference": None if same else 1})
    notes = r.notes(steps) if hasattr(r, "notes") else []
    r.close()
    return r, st, metrics, {c.spec.key: c for c in checks}, notes


def _channel_with_a_hole():
    """The S-channel (hot at its left end, cold at its right), a hole of radius 10 cells
    across its cold end and on past the grid's right edge (the grid is 200 wide)."""
    return _cut(example_case("s-channel"), Outline.of(S.circle(192.0, 55.0, 10.0)))


# ---------------------------------------------------------------------------
# the rules, and the faces
# ---------------------------------------------------------------------------


def test_a_hole_across_the_outline_and_past_the_grid_leaves_the_case_ready():
    s = _channel_with_a_hole()
    hole = s.domain.holes[0]
    x0, _y0, x1, _y1 = hole.bbox()
    assert x1 > s.domain.nx and x0 < 190.0                 # past the grid, across the end
    assert _errors(s) == []
    assert not any("reaches past" in m or "not inside" in m for m in
                   [i.message for i in check(s)])


def test_faces_take_only_edges_that_still_bound_the_domain():
    """The outline's parts inside the hole and the hole's parts outside the outline or
    past the grid are no one's edge any more: every face near the crossing takes its
    condition from what is left."""
    s = _channel_with_a_hole()
    d = s.domain
    hole, ring = d.holes[0].ring(), d.outline.ring()
    for name, poly in geo.live_edges(d):
        mid = 0.5 * (poly[:-1] + poly[1:])
        if name.startswith("outline:"):
            assert not shapes_contains(hole, mid).any(), name
        else:
            assert shapes_contains(ring, mid).all(), name
            assert np.all((mid[:, 0] <= d.nx) & (mid[:, 1] <= d.ny)), name
    bf = geo.boundary_faces(d)
    names = set(bf.edge.tolist())
    assert {"hole0:1", "hole0:2"} <= names or any(n.startswith("hole0") for n in names)
    # every face of the cold end lies outside the hole
    step = np.asarray(geo.DIRECTIONS, dtype=float)[bf.direction]
    mid = np.column_stack([bf.cell % d.nx + 0.5 + 0.5 * step[:, 0],
                           bf.cell // d.nx + 0.5 + 0.5 * step[:, 1]])
    cold = bf.edge == "outline:6"
    assert cold.any() and not shapes_contains(hole, mid[cold]).any()


def shapes_contains(ring, pts):
    return S.contains(ring, pts)


@pytest.mark.parametrize("key", [k for k in EXAMPLES
                                 if example_case(k).domain.outline is not None
                                 or example_case(k).domain.holes])
def test_where_no_hole_crosses_anything_every_face_keeps_its_label(key):
    """`live_edges` gives back `drawn_edges`' own arrays wherever nothing is cut, so
    every drawn example's faces, conditions and lengths are what they were, bit for
    bit (a plate's hole inside it crosses nothing)."""
    d = example_case(key).domain
    for step in (1.0, 0.25):
        a, b = geo.drawn_edges(d, step), geo.live_edges(d, step)
        assert [n for n, _p in a] == [n for n, _p in b]
        assert all(p is q or np.array_equal(p, q) for (_n, p), (_m, q) in zip(a, b))


def test_a_hole_that_cuts_the_domain_in_two_is_refused():
    s = _cut(example_case("wall-2"), Outline.of(S.rectangle(70.0, -5.0, 10.0, 60.0)))
    errs = _errors(s)
    assert any("cuts the domain into 2 separate pieces" in e for e in errs), errs


def test_a_condition_on_an_edge_a_hole_took_whole_is_a_problem_with_a_fix():
    """The river's inlet is its left end; a hole that swallows the end leaves an inlet
    that bounds nothing.  The check says so with a fix; a newly drawn shape does not
    make it unasked (it moves the person's own condition); the fix moves the inlet,
    id and all, to the nearest edge left."""
    s = example_case("river-bend")
    inlet = next(b for b in s.boundaries if b.kind == "river-inlet").model_copy()
    _cut(s, Outline.of(S.circle(0.0, 50.0, 30.0)))
    issues = [i for i in check(s) if i.fix == "swallowed"]
    assert len(issues) == 1 and "river-inlet" in issues[0].message
    assert "removed" in issues[0].message
    before = s.model_dump()
    fill_defaults(s)
    assert [b.edge for b in s.boundaries] == [b["edge"] for b in before["boundaries"]]
    done = apply_fix(s, "swallowed")
    assert done and "moved" in done
    moved = next(b for b in s.boundaries if b.kind == "river-inlet")
    assert moved.id == inlet.id and moved.edge != inlet.edge
    assert moved.edge in set(geo.boundary_faces(s.domain).edge.tolist())
    assert not [i for i in check(s) if i.fix == "swallowed"]


def test_cutting_across_reads_holes_from_the_mask():
    """A hole that bites into the outline is a notch, not a hole: cutting across is
    allowed.  A hole inside the domain leaves no single pair of sides."""
    notch = example_case("s-channel")
    notch.layout = None                  # the coordinates alone, not a layout to grow
    # the channel's top wall at x = 100 cells is at 55 + 14 = 69: a bite out of it
    _cut(notch, Outline.of(S.circle(100.0, 69.0, 6.0)))
    act = geo.domain_mask(notch.domain)
    assert act.sum() < geo.domain_mask(example_case("s-channel").domain).sum()
    assert lay.topological_holes(act) == 0
    _along, across, _s, _e = lay.coordinates(notch.domain, across=True)
    assert np.nanmin(across) < 0.1 and np.nanmax(across) > 0.9
    inner = example_case("plate-hole")
    assert lay.topological_holes(geo.domain_mask(inner.domain)) == 1
    with pytest.raises(lay.LayoutError):
        lay.coordinates(inner.domain, across=True)


# ---------------------------------------------------------------------------
# the one-window control, per family, on a domain with a crossing hole
# ---------------------------------------------------------------------------


def _one_window(s):
    s.layout = None
    s.windows = [Window(id="ALL", shape="cells",
                        runs=geo.mask_to_runs(geo.domain_mask(s.domain)))]
    return s


def test_one_window_over_a_river_with_a_crossing_hole_is_the_full_river():
    s = _cut(example_case("river-bend"), Outline.of(S.circle(150.0, 50.0 + 22.0 * np.sin(
        2.0 * np.pi * 150.0 / 240.0) - 20.0, 8.0)))          # a bite out of a bank
    assert _errors(s) == []
    s = _one_window(s)
    assert _errors(s) == []
    _r, st, _m, _c, _n = _march(s, 15, arms=("serial", "full"))
    assert np.array_equal(st["serial"].u, st["full"].u)


def test_one_window_over_a_plate_with_a_crossing_hole_is_the_full_plate():
    s = _cut(example_case("plate-hole"), Outline.of(S.circle(160.0, 60.0, 10.0)))
    assert _errors(s) == []
    s = _one_window(s)
    assert _errors(s) == []
    _r, st, _m, _c, _n = _march(s, 1, arms=("serial", "full"))
    assert np.array_equal(st["serial"].u, st["full"].u)


def test_one_window_over_a_channel_with_a_crossing_hole_is_the_full_channel():
    s = _one_window(_channel_with_a_hole())
    assert _errors(s) == []
    _r, st, _m, _c, _n = _march(s, 1, arms=("serial", "full"))
    assert np.array_equal(st["serial"].u, st["full"].u)


def test_the_split_of_an_arc_with_a_crossing_hole_is_the_unsplit_solver():
    s = _cut(example_case("bimetal-arc"), Outline.of(S.circle(4.0 + 100.0 / np.sqrt(2.0),
                                                              4.0 + 100.0 / np.sqrt(2.0),
                                                              8.0)))
    assert _errors(s) == []
    _r, _st, _m, checks, _n = _march(s, 3)
    assert checks["reference"].passed is True and checks["lag"].passed is True
    assert checks["energy"].passed


def test_the_sounds_pieces_with_a_crossing_hole_are_the_full_domain():
    s = _cut(example_case("sound-lens"), Outline.of(S.circle(150.0, 100.0, 12.0)))
    assert _errors(s) == []
    _r, _st, _m, checks, _n = _march(s, 60)
    assert checks["bitwise"].passed is True and checks["energy"].passed


def test_one_window_over_a_farm_with_a_hole_through_its_top_is_its_full_domain():
    from atlas.workbench.families import windfarm as wf
    s = _cut(example_case("farm-hill"), Outline.of(S.circle(120.0, 240.0, 16.0)))
    assert _errors(s) == []
    d = s.domain
    s.windows = [Window(id="W", x0=0, y0=0, nx=d.nx, ny=d.ny)]
    s.layout = None
    s.coupling.elliptic, s.coupling.assembly = "embedded", "blend"
    run = wf.build(s, arms=("serial", "full"), threads=1)
    try:
        a, b = run.initial("serial"), run.initial("full")
        for _ in range(2):
            a, b = run.step("serial", a), run.step("full", b)
            assert np.array_equal(a.u, b.u) and np.array_equal(a.v, b.v)
    finally:
        run.close()


@pytest.mark.parametrize("key, hole", [
    ("ring-film", (4.0 + 70.0, 4.0, 8.0)),           # a bite out of the end E1 sits on
    ("cooled-winding", (100.0, 60.0, 6.0)),          # a bite out of the block's top
])
def test_the_iterated_families_pass_their_checks_with_a_crossing_hole(key, hole):
    """Style D and the two-physics seam are not the full domain to the bit (they
    iterate to a tolerance): their own registered checks hold instead."""
    s = _cut(example_case(key), Outline.of(S.circle(*hole)))
    assert _errors(s) == []
    _r, _st, _m, checks, _n = _march(s, 1)
    for c in checks.values():
        assert c.passed is not False, (key, c.as_dict())


# ---------------------------------------------------------------------------
# the plan's two cases
# ---------------------------------------------------------------------------


def test_a_plate_loaded_on_what_a_hole_leaves_of_its_edge_balances():
    """Clamped on the left, pulled on the right, a hole across the right edge: the load
    is carried by the faces left, and the supports balance it to 1e-6."""
    s = _cut(example_case("plate-hole"), Outline.of(S.circle(160.0, 40.0, 12.0)))
    assert _errors(s) == []
    bf = geo.boundary_faces(s.domain)
    assert 0 < int(np.sum(bf.edge == "right")) < s.domain.ny     # the edge is in two
    _r, _st, _m, checks, _n = _march(s, 1, arms=("serial", "full"))
    assert checks["force"].passed and checks["force"].value < 1e-6
    assert checks["reference"].passed


def test_a_conduction_case_whose_right_edge_a_hole_cuts_reports_each_piece():
    """A copper-and-steel wall whose right edge (held at 300 K) a hole cuts in two: the
    energy balances to 1e-6, and the heat through each surviving piece is reported,
    the pieces summing to the edge's total."""
    s = _cut(example_case("wall-2"), Outline.of(S.circle(160.0, 20.0, 8.0)))
    assert _errors(s) == []
    r, st, metrics, checks, notes = _march(s, 1)
    assert checks["balance"].passed and checks["balance"].value < 1e-6
    pieces = metrics["full"]["heat_through_edges"]
    right = {k: v for k, v in pieces.items() if k.startswith("right")}
    assert set(right) == {"right (1 of 2)", "right (2 of 2)"}
    assert all(v < 0.0 for v in right.values())                # heat leaves through both
    total = fv.boundary_inflow(r.f, st["full"].u)
    assert sum(right.values()) == pytest.approx(total["right"], rel=1e-12, abs=1e-12)
    assert sum(pieces.values()) == pytest.approx(sum(total.values()), abs=1e-9)
    assert any("right (1 of 2)" in n for n in notes)


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------


@pytest.fixture()
def wb(tmp_path, monkeypatch):
    from atlas.workbench import runner
    from atlas.workbench.app import Workbench
    monkeypatch.setattr(runner.CaseRun, "start", lambda self: None)
    return Workbench(cases_dir=str(tmp_path))


def test_in_the_page_a_hole_dragged_across_the_edge_and_past_the_grid_stays_ready(wb):
    """The done-when of item 1.3: a circle hole across the outline's edge and on past
    the grid's leaves the case ready."""
    wb.type_sel.value = "conduction-2d"
    ed = wb.geo_editor
    ed._switch(layer="domain", tool="draw")
    d = wb.spec.domain
    ed.add_drawn(Outline.of(S.rectangle(16.0, 16.0, d.nx - 32.0, d.ny - 32.0)))
    ed = wb.geo_editor
    ed.state["draw_as"] = "hole"
    ed._switch(layer="domain", tool="draw")
    ed.add_drawn(Outline.of(S.circle(d.nx - 8.0, d.ny / 2.0, 30.0)))
    assert wb.spec.domain.holes, wb.log_lines[:3]
    assert [i.message for i in wb.issues() if i.severity == "error"] == []
    assert wb.status_btn.name.startswith("✓ Ready")
