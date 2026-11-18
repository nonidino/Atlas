"""Drawn geometry in the workbench (case file 0.4): shapes with straight, arc and
spline edges for the domain's outline, its holes, regions and windows, resolved
to the cells whose centres they contain.

The owner's request, 2026-09-29: "draw nonregular geometry edges, and then have
NONRECTANGULAR windows".  These tests pin:

* the shapes against their closed forms (`shapes.py`);
* the case file: 0.3 files load unchanged, drawn cases round-trip;
* the masks, the distance-ramp partition of unity and its certificate;
* the one-window control on a drawn domain, bit for bit, steady and transient;
* both drawn examples against the full domain, their energy, the bitwise
  threaded control, and their compile verdicts;
* the check's refusals, and drawing and reshaping, sent as the studio's page sends
  them (its protocol, `tests/studio_driver.py`).
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.workbench import fv  # noqa: E402
from atlas.workbench import geometry as geo  # noqa: E402
from atlas.workbench import shapes as S  # noqa: E402
from atlas.workbench.families import conduction as cd  # noqa: E402
from atlas.workbench.spec import (Boundary, CaseSpec, Outline, Window, check,  # noqa: E402
                                  example_case)
from atlas.workbench.tiling import MaskTiling  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import studio_driver as D  # noqa: E402

DEG = math.pi / 180.0


def _errors(spec):
    return [i.message for i in check(spec) if i.severity == "error"]


def _march(spec, steps=None, arms=("serial", "parallel", "full")):
    r = cd.build(spec, arms=arms, threads=2)
    st = {a: r.initial(a) for a in r.arms}
    hist = {a: [] for a in r.arms}
    same = True
    for _ in range(steps or spec.run.steps):
        for a in r.arms:
            prev = st[a]
            st[a] = r.step(a, prev)
            hist[a].append(r.observe(a, st[a], prev))
        if "parallel" in st:
            same &= r.bitwise_equal(st["serial"], st["parallel"])
    r.close()
    return r, st, hist, same


# ---------------------------------------------------------------------------
# the shapes
# ---------------------------------------------------------------------------


def test_a_circle_of_arcs_is_on_its_radius_and_encloses_pi_r_squared():
    pts, ks, bs = S.circle(50.0, 40.0, 20.0)
    ring = S.sample(pts, ks, bs, step=0.1)
    assert np.max(np.abs(np.hypot(ring[:, 0] - 50, ring[:, 1] - 40) - 20)) < 1e-9
    assert 0.0 < math.pi * 400 - S.signed_area(ring) < 0.02     # inscribed polygon
    for k in range(4):                                         # each handle on the circle
        m = S.edge_midpoint(pts, ks, bs, k)
        assert abs(math.hypot(m[0] - 50, m[1] - 40) - 20) < 1e-9


def test_a_ring_sector_has_the_area_of_its_closed_form():
    pts, ks, bs = S.annulus_sector(0.0, 0.0, 40.0, 100.0, 0.0, math.pi / 2)
    ring = S.sample(pts, ks, bs, step=0.05)
    want = math.pi / 4 * (100 ** 2 - 40 ** 2)
    assert abs(S.signed_area(ring) - want) / want < 1e-4
    r = np.hypot(ring[:, 0], ring[:, 1])
    assert r.min() > 40 - 1e-9 and r.max() < 100 + 1e-9
    assert not S.self_intersects(ring)


def test_a_dragged_handle_sets_the_sagitta_and_a_split_does_not_move_the_curve():
    assert S.bulge_from_point((0, 0), (10, 0), (5, -5)) == pytest.approx(1.0)   # semicircle
    assert np.allclose(S.arc_midpoint((0, 0), (10, 0), 1.0), (5, -5))
    pts, ks, bs = S.circle(0.0, 0.0, 10.0)
    p2, k2, b2, origin = S.split(pts, ks, bs, 1)
    assert len(p2) == 5 and origin == [0, 1, 1, 2, 3]
    after = S.sample(p2, k2, b2, step=0.05)
    assert np.max(np.abs(np.hypot(after[:, 0], after[:, 1]) - 10)) < 1e-9
    # a straight edge bent to within half a cell of its chord stays straight
    _p, k3, _b = S.bend(pts, ["line"] * 4, [0.0] * 4, 0, (5.0, 5.0 + 0.1))
    assert k3[0] == "line"


def test_a_spline_passes_through_its_vertices():
    pts, ks, bs = S.ellipse(0.0, 0.0, 30.0, 15.0, 8)
    ring = S.sample(pts, ks, bs, step=0.1)
    assert max(float(np.min(np.hypot(ring[:, 0] - p[0], ring[:, 1] - p[1]))) for p in pts) < 1e-9
    assert abs(S.signed_area(ring) - math.pi * 450) / (math.pi * 450) < 0.01


def test_a_shape_that_crosses_itself_is_named_for_the_crossing():
    """Positive control and negative control: a figure-of-eight's signed area is
    zero, so the crossing must be tested before the area."""
    assert S.problems([(0, 0), (10, 10), (10, 0), (0, 10)]) == ["the outline crosses itself"]
    assert S.problems([(0, 0), (10, 0), (10, 10), (0, 10)]) == []
    assert S.problems([(0, 0), (10, 0)], ["arc", "arc"], [1.0, 1.0]) == []    # two semicircles
    assert len(S.problems([(0, 0), (10, 0)])) == 1


# ---------------------------------------------------------------------------
# the case file
# ---------------------------------------------------------------------------


def test_a_drawn_case_round_trips_and_its_window_box_is_derived():
    s = example_case("bend-3")
    assert CaseSpec.from_json(s.to_json()) == s
    w = s.windows[0]
    # its box: from x = 4 + 34 cos 44 deg = 28.46, and from y = 4 - 106 sin 5 deg < 0,
    # which is clipped to the grid's edge
    assert w.shape == "curve" and (w.x0, w.y0) == (28, 0)
    with pytest.raises(ValueError):
        Boundary(id="x", edge="outline", kind="insulated")          # not an edge name
    with pytest.raises(ValueError):
        Window(id="x", shape="curve")                               # drawn, with no outline


def test_every_case_without_a_drawn_shape_is_plain():
    """The rule that keeps every earlier case on its own arithmetic."""
    for key in ("wall-2", "plate-insert", "farm-12", "plume-2", "cooled-block"):
        assert geo.is_plain(example_case(key)), key
    for key in ("bend-3", "insert-round"):
        assert not geo.is_plain(example_case(key)), key


# ---------------------------------------------------------------------------
# masks, weights and the boundary of a drawn domain
# ---------------------------------------------------------------------------


def test_the_drawn_domain_is_its_cells_and_its_faces_know_their_edge():
    s = example_case("bend-3")
    act = geo.domain_mask(s.domain)
    area = math.pi / 4 * (100 ** 2 - 40 ** 2)
    assert abs(act.sum() - area) / area < 0.01
    vf = geo.void_faces(s.domain)
    counts = {n: int((vf.edge == n).sum()) for n in dict.fromkeys(vf.edge.tolist())}
    assert set(counts) == {"outline:0", "outline:1", "outline:2", "outline:3"}
    # the long outer arc has the most faces, the two straight ends the fewest
    assert counts["outline:1"] > counts["outline:3"] > counts["outline:0"]


def test_the_distance_ramp_partition_of_unity_is_certified():
    s = example_case("bend-3")
    t = MaskTiling(geo.domain_mask(s.domain), geo.window_masks(s), 8)
    cert = t.certify()
    assert cert.holds and cert.identity_residual <= 1e-12 and cert.chi_min >= 0.0
    an = geo.analyse_case(s)
    assert not an.ramp_only.any() and not an.uncovered.any()
    assert an.cross_points == [] and an.nested == []


def test_a_window_ramps_from_its_artificial_faces_and_not_from_the_domains():
    act = np.ones((20, 40), dtype=bool)
    left = np.zeros_like(act)
    left[:, :24] = True
    w = geo.ramp_weight(left, act, 8)
    assert w[10, 0] == 1.0                                    # the domain's own edge: no ramp
    assert w[10, 23] == pytest.approx((0.5 / 8) ** 2)         # next to the cut
    assert w[10, 15] == 1.0 and w[10, 16] < 1.0               # 8.5 cells from the cut


# ---------------------------------------------------------------------------
# the solvers on a drawn domain
# ---------------------------------------------------------------------------


def _one_window(mode):
    s = example_case("bend-3")
    s.windows = [Window(id="ALL", shape="curve", outline=s.domain.outline)]
    s.run.mode = mode
    s.run.macro_dt = 60.0
    return s


@pytest.mark.parametrize("mode", ["steady", "transient"])
def test_one_window_over_a_drawn_domain_is_the_full_domain_to_the_bit(mode):
    s = _one_window(mode)
    assert _errors(s) == []
    _r, st, _h, _same = _march(s, steps=2 if mode == "steady" else 3,
                               arms=("serial", "full"))
    assert np.array_equal(st["serial"].u, st["full"].u)


def test_the_bend_agrees_with_the_full_domain_and_closes_its_energy():
    s = example_case("bend-3")
    assert _errors(s) == []
    r, st, hist, same = _march(s, steps=2)
    assert r.certificate.holds
    span = 100.0
    assert float(np.max(np.abs(st["serial"].u - st["full"].u))) / span < 1e-6
    assert same                                              # threaded = serial, every step
    for arm, rows in hist.items():
        assert max(row["balance"] for row in rows) < 1e-6, arm
    f = r.field(st["full"])
    act = geo.domain_mask(s.domain)
    assert np.all(np.isnan(f[~act])) and np.all(np.isfinite(f[act]))


def test_the_round_insert_iterates_across_a_circle_to_the_full_domain():
    s = example_case("insert-round")
    assert _errors(s) == []
    r, st, hist, _same = _march(s, steps=1, arms=("serial", "full"))
    assert (r.d_id, r.n_id) == ("insert", "plate")           # the disc would float
    assert float(np.max(np.abs(st["serial"].u - st["full"].u))) / 100.0 < 1e-6
    assert all(row["converged"] for row in hist["serial"])
    for arm, rows in hist.items():
        assert max(row["balance"] for row in rows) < 1e-6, arm
    assert r.sd.face_rows.size > 150                         # the circle's staircase of faces


def test_a_drawn_domains_heat_is_close_to_the_continuum_ring():
    """Information, not a registered check: the staircase's own error against the
    ring's continuum heat flow k dT ln(ro/ri) / theta, recorded as measured
    (-0.97% on 2026-09-29) and pinned loosely so a broken boundary is seen."""
    s = example_case("bend-3")
    _r, st, hist, _s = _march(s, steps=1, arms=("full",))
    q = hist["full"][-1]["heat_in"]
    q_cont = 45.0 * 100.0 * math.log(100 / 40) / (math.pi / 2)
    assert abs(q - q_cont) / q_cont < 0.03


def test_the_runner_measures_a_drawn_case_on_the_domain_only():
    from atlas.workbench import runner
    run = runner.CaseRun(example_case("bend-3"), arms=("serial", "full"), steps=1)
    run.run_blocking()
    fd = run.results["field_difference"]["serial"]
    assert np.isfinite(fd["rms"]) and fd["max"] < 1e-6 * 100.0


# ---------------------------------------------------------------------------
# the compile
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["bend-3", "insert-round"])
def test_the_drawn_examples_compile_to_r10_as_their_rectangles_do(key):
    """Iterated one-physics pieces are decertified by R10 after the scheme whatever
    their shape, as their rectangles are (W348, 2026-09-30; refused at L2 before);
    every seam is admitted, and the bend declares no cross-point (only neighbours
    overlap)."""
    from atlas.workbench.compile import compile_case
    s = compile_case(example_case(key))
    assert s.refused_before is None and s.verdict == "admit-uncertified"
    assert [(r["layer"], r["verdict"]) for r in s.other if r["rule"] == "R10"] == [
        ("L5", "admit-uncertified")]
    assert not any(r["verdict"] == "refuse" for r in s.other)
    assert s.seams and all(sv.verdict in ("admit", "admit-uncertified") for sv in s.seams)
    assert s.cross_points == ()


def test_a_curved_seams_faces_are_ordered_along_it():
    from atlas.workbench.compile import curve_order
    s = example_case("insert-round")
    f = cd.field_from_case(s)
    sys_ = fv.assemble(f, cd.cells_of(s, s.windows[1]), "dirichlet")
    faces = curve_order(sys_, np.arange(sys_.face_rows.size), f.nx)
    ins = sys_.idx[sys_.face_rows[faces]]
    out = sys_.face_outside[faces]
    mid = 0.5 * (np.column_stack([ins % f.nx, ins // f.nx])
                 + np.column_stack([out % f.nx, out // f.nx]))
    steps = np.hypot(*np.diff(mid, axis=0).T)
    assert steps.max() <= math.sqrt(2) + 1e-9                # never jumps across the circle


# ---------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------


def test_the_check_refuses_what_a_drawn_case_gets_wrong():
    base = example_case("bend-3")
    s = base.copy_deep()
    s.boundaries = [b for b in s.boundaries if b.edge != "outline:1"]
    assert any("outline:1 has no boundary" in m for m in _errors(s))
    s = base.copy_deep()
    s.boundaries.append(Boundary(id="again", edge="outline:1", kind="insulated"))
    assert any("are all on outline:1" in m for m in _errors(s))
    s = base.copy_deep()
    s.boundaries.append(Boundary(id="grid", edge="left", kind="insulated"))
    assert any("on the grid's left edge" in m for m in _errors(s))
    s = base.copy_deep()
    s.domain.holes = [Outline.of(S.circle(5.0, 100.0, 3.0))]    # across the ring's end
    s.boundaries += [Boundary(id=f"h{k}", edge=f"hole0:{k}", kind="insulated")
                     for k in range(4)]
    # it was refused as "not inside the domain's outline"; since 2026-09-30 a hole may
    # cross the outline (the owner: "Holes should be allowed to intersect the
    # boundary's edge"), and this one bites a few cells out of the ring's cold end
    assert _errors(s) == []
    removed = int(geo.domain_mask(base.domain).sum() - geo.domain_mask(s.domain).sum())
    assert 0 < removed < 40
    s = base.copy_deep()
    s.domain.outline = Outline.of(S.circle(54.0, 54.0, 60.0))    # past the grid
    s.boundaries = []
    assert any("reaches past the 108 x 108 grid" in m for m in _errors(s))


def test_every_family_runs_on_drawn_shapes_and_the_refusal_remains(monkeypatch):
    """The owner, 2026-09-29: "make the smooth domain/spline/other stuff available for
    all cases".  Every family that runs declares drawn shapes; the check's refusal
    stays for a family that would not."""
    import dataclasses

    from atlas.workbench import registry
    for fid in registry.available_ids():
        assert registry.family(fid).drawn_shapes, fid
    s = example_case("wake-array-3")
    s.windows.append(Window(id="C1", shape="curve", outline=Outline.of(S.circle(100, 100, 30))))
    assert not any("runs on rectangles only" in m for m in _errors(s))
    fam = registry.family("incompressible-2d")
    blocked = dataclasses.replace(fam, drawn_shapes=False, drawn_why="a test says so")
    monkeypatch.setattr(registry, "FAMILIES", tuple(blocked if f.id == fam.id else f
                                                    for f in registry.FAMILIES))
    assert any("runs on rectangles only" in m and "a test says so" in m
               for m in _errors(s))


# ---------------------------------------------------------------------------
# drawing and reshaping, as the studio's page sends them
# ---------------------------------------------------------------------------


@pytest.fixture
def s(tmp_path):
    """A page's session on the plate with an insert."""
    return D.session(tmp_path, "plate-insert")


def _ring(cx, cy, r):
    """A circle as a path of four quarter arcs (the bulge of a quarter arc is
    tan(pi/8)), in cells."""
    pts = [(cx + r, cy), (cx, cy + r), (cx - r, cy), (cx, cy - r)]
    return pts, ["arc"] * 4, [math.tan(math.pi / 8)] * 4


def test_an_outline_drawn_becomes_the_domain(s):
    nx, ny = s.spec.domain.nx, s.spec.domain.ny
    r, _ = D.draw_domain(s, [(0, 0), (nx, 0), (nx, ny // 2), (nx // 2, ny // 2), (nx // 2, ny),
                             (0, ny)])
    assert r["ok"], r
    d = s.spec.domain
    assert d.outline is not None and len(d.outline.points) == 6
    assert geo.domain_mask(d).sum() == nx * ny - (nx // 2) * (ny - ny // 2)
    assert sorted(b.edge for b in s.spec.boundaries) == [f"outline:{k}" for k in range(6)]


def test_reshape_bends_splits_and_smooths_and_the_boundaries_follow(s):
    """An edge straightened, bent into an arc again, split by a new point (its condition
    held on both halves) and made smooth through a point: each one edit of the drawn
    path, and the engine's outline follows."""
    nx, ny = s.spec.domain.nx, s.spec.domain.ny
    pts, edges, bulge = _ring(nx / 2, ny / 2, ny / 2 - 4)
    r, _ = D.draw_domain(s, pts, edges=edges, bulge=bulge)
    assert r["ok"], r
    assert s.spec.domain.outline.kinds() == ["arc"] * 4

    def path_edit(*ops, label):
        r, pushes = D.edit(s, *ops, label=label)
        assert r["ok"], (label, r, D.notices(pushes))

    i = next(k for k, f in enumerate(D.features(s)) if f["role"] == "add")
    at = ["geometry", "features", i]
    path_edit(D.set_op(at + ["edges", 0], "line"), D.set_op(at + ["bulge", 0], 0.0),
              label="straightened edge 1")
    assert s.spec.domain.outline.kinds()[0] == "line"
    path_edit(D.set_op(at + ["edges", 0], "arc"), D.set_op(at + ["bulge", 0], 0.2),
              label="bent edge 1")
    assert s.spec.domain.outline.kinds()[0] == "arc"
    n_b = len(s.spec.boundaries)
    f = D.features(s)[i]
    (x0, y0), (x1, y1) = f["points"][2], f["points"][3]
    path_edit({"op": "insert", "path": at + ["points"], "index": 3,
               "value": [(x0 + x1) / 2, (y0 + y1) / 2]},
              {"op": "insert", "path": at + ["edges"], "index": 3, "value": "line"},
              {"op": "insert", "path": at + ["bulge"], "index": 3, "value": 0.0},
              {"op": "insert", "path": at + ["handles"], "index": 3, "value": None},
              D.set_op(at + ["edges", 2], "line"), D.set_op(at + ["bulge", 2], 0.0),
              label="a point on edge 3")
    assert len(s.spec.domain.outline.points) == 5 and len(s.spec.boundaries) == n_b + 1
    path_edit(D.set_op(at + ["edges", 0], "spline"), D.set_op(at + ["edges", 1], "spline"),
              label="smooth through point 2")
    k = s.spec.domain.outline.kinds()
    assert k[0] == "spline" and k[1] == "spline"


def test_drawn_windows_keep_their_place_when_a_rectangle_moves(s):
    square = Outline(points=[(8, 8), (72, 8), (72, 40), (8, 40)], edges=["line"] * 4,
                     bulge=[0.0] * 4)
    w = Window(id="W3", shape="curve", outline=square).model_dump(mode="json")
    r, _ = D.edit(s, {"op": "insert", "path": ["windows"], "value": w}, label="drew W3")
    assert r["ok"], r
    assert [w.shape for w in s.spec.windows] == ["rect", "rect", "curve"]
    drawn = s.spec.windows[2].outline
    x0 = s.spec.windows[1].x0                                  # F10, x0 72 of 160
    r, _ = D.edit(s, D.set_op(["windows", 1, "x0"], x0 - 8), label="F10 moved")
    assert r["ok"] and s.spec.windows[1].x0 == x0 - 8
    assert [w.shape for w in s.spec.windows] == ["rect", "rect", "curve"]
    assert s.spec.windows[2].outline == drawn


def test_a_circle_and_a_rectangle_drawn_as_regions(s):
    """Each tool adds one shape; a region keeps the material it was drawn in: a circle
    is four arcs, a rectangle on whole cells its cells."""
    dx = s.spec.domain.dx
    r, _ = D.add_feature(s, D.circle("f8", 40, 48, 12, dx, material="copper", name="Disc"))
    assert r["ok"], r
    reg = s.spec.regions[-1]
    assert reg.shape == "curve" and reg.outline.kinds() == ["arc"] * 4 and reg.material == "copper"
    r, _ = D.add_feature(s, D.rect("f9", 96, 16, 24, 24, dx, material="copper", name="Block"))
    assert r["ok"], r
    reg = s.spec.regions[-1]
    assert (reg.shape, reg.x0, reg.y0, reg.nx, reg.ny) == ("rect", 96, 16, 24, 24)


def test_an_outline_drawn_after_a_hole_is_an_outline(s):
    """In the retired editor, after Cut a hole the next shape drawn was still a hole,
    so Draw the outline cut a second one.  A cut and an outline are each what they are
    drawn as."""
    sq = [(64, 32), (96, 32), (96, 64), (64, 64)]
    r, _ = D.add_feature(s, D.path("f9", sq, s.spec.domain.dx, role="cut", name="Hole"))
    assert r["ok"], r
    assert len(s.spec.domain.holes) == 1 and s.spec.domain.outline is None
    r, _ = D.draw_domain(s, [(0, 0), (150, 0), (150, 96), (0, 96)])
    assert r["ok"], r
    assert len(s.spec.domain.holes) == 1 and s.spec.domain.outline is not None


def test_a_hole_is_cut_deleted_and_restored(s):
    dx = s.spec.domain.dx
    r, _ = D.add_feature(s, D.circle("f9", 120, 48, 10, dx, role="cut", name="Hole"))
    assert r["ok"], r
    assert len(s.spec.domain.holes) == 1
    assert sum(b.edge.startswith("hole0:") for b in s.spec.boundaries) == 4
    i = next(k for k, f in enumerate(D.features(s)) if f["id"] == "f9")
    r, _ = D.edit(s, {"op": "remove", "path": ["geometry", "features", i]}, label="deleted")
    assert r["ok"] and not s.spec.domain.holes
    assert not any(b.edge.startswith("hole") for b in s.spec.boundaries)
    D.ok(s, "edit.undo")
    assert len(s.spec.domain.holes) == 1
    # back to the whole grid: the drawing's own rectangle, nothing cut from it
    nx, ny = s.spec.domain.nx, s.spec.domain.ny
    keep = [f for f in D.features(s) if f["role"] == "region"]
    grid = D.rect("f1", 0, 0, nx, ny, dx, role="add", name="Plate")
    r, _ = D.edit(s, D.set_op(["geometry", "features"], [grid] + keep), label="the grid")
    assert r["ok"], r
    assert s.spec.domain.outline is None and not s.spec.domain.holes
    assert sorted(b.edge for b in s.spec.boundaries) == ["bottom", "left", "right", "top"]


def test_an_edge_of_a_drawn_window_set_to_an_arc(s):
    """An edge of a drawn window, drawn straight, as an arc of a typed bulge, or smooth:
    one edit of the window's outline."""
    D.ok(s, "case.example", key="bend-3")
    assert {k for w in s.spec.windows for k in w.outline.kinds()} == {"line", "arc"}
    assert s.spec.windows[0].outline.kinds()[0] == "line"
    r, _ = D.edit(s, D.set_op(["windows", 0, "outline", "edges", 0], "arc"),
                  D.set_op(["windows", 0, "outline", "bulge", 0], 0.25), label="edge 1: arc")
    assert r["ok"], r
    o = s.spec.windows[0].outline
    assert o.kinds()[0] == "arc" and o.bulges()[0] == 0.25
