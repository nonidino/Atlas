"""Windows generated from the geometry (case file 0.5, `atlas/workbench/layout.py`).

The owner's question, 2026-09-29: "it doesn't make sense for the windows
themselves to be rectangles, does it? Why doesn't their shape itself ADAPT to the
geometry of the curve smoothly?"  These tests pin the answer against what the
geometry predicts:

* on a quarter ring the domain's own "along" coordinate is its angle and its
  "across" coordinate a function of the radius, so equal-count cuts fall on the
  radii at 30 and 60 degrees and on the arc of equal areas, r = sqrt((ri^2 + ro^2)/2);
* an L-shaped domain is cut around its corner, from one arm's end to the other's;
* cut by material, the pieces are the materials' cells;
* the windows regenerate when the domain changes, and become the case's own when
  one is edited by hand;
* the generated S-channel runs to the full domain, threaded equal to serial.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.workbench import geometry as geo  # noqa: E402
from atlas.workbench import layout as L  # noqa: E402
from atlas.workbench.families import conduction as cd  # noqa: E402
from atlas.workbench.spec import CaseSpec, Layout, Outline, check, example_case  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from workbench_ui import walk  # noqa: E402


def _polar(mask, c=(4.0, 4.0)):
    jj, ii = np.nonzero(mask)
    x, y = ii + 0.5 - c[0], jj + 0.5 - c[1]
    return np.degrees(np.arctan2(y, x)), np.hypot(x, y)


def _errors(spec):
    return [i.message for i in check(spec) if i.severity == "error"]


# ---------------------------------------------------------------------------
# the domain's own coordinates, on a quarter ring
# ---------------------------------------------------------------------------


def test_a_rings_coordinates_are_its_angle_and_a_function_of_its_radius():
    s = example_case("bend-3")
    assert set(L.ends(s.domain)) == {"outline:0", "outline:2"}       # the two straight ends
    along, across, _a, _b = L.coordinates(s.domain, across=True)
    act = geo.domain_mask(s.domain)
    th, r = _polar(act)
    a = along[act]
    assert min(np.max(np.abs(a - th / 90.0)), np.max(np.abs(a - (1 - th / 90.0)))) < 0.01
    want = np.log(r / 40.0) / np.log(100.0 / 40.0)
    b = across[act]
    assert min(np.max(np.abs(b - want)), np.max(np.abs(b - (1 - want)))) < 0.01


def test_equal_count_cuts_of_a_ring_fall_on_its_radii_and_its_arc():
    s = example_case("bend-3")
    s.layout = Layout(along=3, across=2)
    parts = L.pieces(s)
    assert [n for n, _p in parts] == ["F00", "F01", "F10", "F11", "F20", "F21"]
    # the three sectors (both lanes of each): bounded by the radii at 30 and 60 degrees
    sectors: dict[str, list] = {}
    for n, m in parts:
        sectors.setdefault(n[1], []).append(m)
    bounds = []
    for c in sorted(sectors):
        th, _r = _polar(np.logical_or.reduce(sectors[c]))
        bounds += [th.min(), th.max()]
    assert np.allclose(sorted(bounds)[1:-1], [30, 30, 60, 60], atol=1.0)
    # each lane stays on its side of the arc that halves a sector's area
    rc = math.sqrt((40 ** 2 + 100 ** 2) / 2)
    for _n, m in parts:
        _t, r = _polar(m)
        split = r.max() if r.max() < rc + 1.5 else r.min()
        assert abs(split - rc) < 1.5


def test_an_l_shape_is_cut_around_its_corner():
    s = example_case("plate-insert")
    s.domain.outline = Outline(points=[(0, 0), (160, 0), (160, 40), (40, 40), (40, 96), (0, 96)])
    s.boundaries = []
    s.layout = Layout(along=4)
    parts = [m for _n, m in L.pieces(s)]
    first, last = parts[0], parts[-1]
    tips = [(np.nonzero(p)[0].max(), np.nonzero(p)[1].max()) for p in (first, last)]
    # one end piece holds the top of the upright arm, the other the far end of the foot
    assert {bool(t[0] > 80) for t in tips} == {True, False}
    assert {bool(t[1] > 140) for t in tips} == {True, False}
    counts = [int(p.sum()) for p in parts]
    assert max(counts) - min(counts) <= 1                            # equal cell counts


def test_cut_by_material_the_pieces_are_the_materials():
    s = example_case("insert-round")
    s.layout = Layout(cut="materials")
    wins, reach = L.generate(s)
    assert [w.id for w in wins] == ["steel", "copper"] and reach == 0     # style C: no growth
    disc = geo.region_mask(s.regions[1], s.domain.nx, s.domain.ny)
    assert np.array_equal(geo.window_mask(wins[1], s.domain.nx, s.domain.ny), disc)


def test_style_c_takes_two_pieces_that_meet_along_faces():
    s = example_case("bend-3")
    s.coupling.style = "C"
    s.layout = Layout(along=3)
    with pytest.raises(L.LayoutError, match="exactly two"):
        L.generate(s)
    s.layout = Layout(along=2)
    wins, _r = L.generate(s)
    a, b = (geo.window_mask(w, 108, 108) for w in wins)
    assert not (a & b).any() and geo.faces_between(a, b) > 0
    assert np.array_equal(a | b, geo.domain_mask(s.domain))


def test_generation_is_deterministic_and_stored_as_cells():
    s = example_case("bend-3")
    s.layout = Layout(along=3)
    w1, _ = L.generate(s)
    w2, _ = L.generate(s)
    assert [w.runs for w in w1] == [w.runs for w in w2]
    assert all(w.shape == "cells" for w in w1)
    s.windows = w1
    assert CaseSpec.from_json(s.to_json()) == s


def test_a_domain_in_pieces_or_with_holes_says_what_it_cannot_do():
    s = example_case("bend-3")
    s.layout = Layout(along=3, across=2)
    s.domain.holes = [Outline.of(__import__("atlas.workbench.shapes", fromlist=["circle"])
                                 .circle(50.0, 50.0, 6.0))]
    with pytest.raises(L.LayoutError, match="without holes"):
        L.pieces(s)
    s.layout = Layout(along=3)
    assert len(L.pieces(s)) == 3                                     # along still works


# ---------------------------------------------------------------------------
# the generated example, and following the domain in the editor
# ---------------------------------------------------------------------------


def test_the_s_channel_follows_its_shape_and_agrees_with_the_full_domain():
    s = example_case("s-channel")
    assert _errors(s) == []
    assert [w.id for w in s.windows] == ["F00", "F10", "F20", "F30"]
    act = geo.domain_mask(s.domain)
    from scipy.ndimage import label
    xs = []
    for w in s.windows:
        m = geo.window_mask(w, s.domain.nx, s.domain.ny) & act
        assert label(m)[1] == 1                                       # one piece each
        xs.append(np.nonzero(m)[1].mean())
    assert xs == sorted(xs)                                           # in order along it
    r = cd.build(s, threads=2)
    st = {a: r.initial(a) for a in r.arms}
    for a in r.arms:
        st[a] = r.step(a, st[a])
    r.close()
    assert float(np.max(np.abs(st["serial"].u - st["full"].u))) / 100.0 < 1e-6
    assert np.array_equal(st["serial"].u, st["parallel"].u)


@pytest.fixture
def wb():
    from atlas.workbench.app import Workbench
    w = Workbench(spec=example_case("s-channel"))
    w.show("geometry")
    return w


def test_reshaping_the_domain_regenerates_the_windows(wb):
    ed = wb.geo_editor
    ed._switch(layer="domain", tool="resize", snap=1)
    ed = wb.geo_editor
    before = [w.runs for w in wb.spec.windows]
    rows = ed._handle_rows["vtx"]
    data = dict(ed.src_vtx.data)
    k = [i for i, r in enumerate(rows) if r[0] == "domain:outline" and r[1] == 3][0]
    ys = list(data["y"])
    ys[k] -= 10
    ed._on_vertices("data", data, dict(data, y=ys))
    assert wb.spec.layout is not None
    assert [w.runs for w in wb.spec.windows] != before
    assert _errors(wb.spec) == []


def test_editing_a_window_by_hand_makes_the_windows_the_cases_own(wb):
    ed = wb.geo_editor
    ed._switch(layer="windows")
    ed = wb.geo_editor
    ed.select("windows:0")
    ed.delete_selected()
    assert wb.spec.layout is None and len(wb.spec.windows) == 3
    assert "no longer follow the shape" in wb.log_lines[0]
    wb.geo_editor.set_layout(True, "along", 3, 2)
    assert wb.spec.layout is not None and len(wb.spec.windows) == 6
    ed = wb.geo_editor
    ed.sync()
    assert len(ed.src_win_gen.data["xs"]) >= 6                       # the cuts, drawn


def test_drawing_a_domain_makes_the_rectangles_give_way(wb):
    wb.dispatch("file:example:plate-insert")
    ed = wb.geo_editor
    ed._switch(layer="domain", tool="shape")
    ed = wb.geo_editor
    ed.add_ready("circle")                                            # a round domain
    assert wb.spec.layout is not None and wb.spec.layout.along == 2
    assert [w.shape for w in wb.spec.windows] == ["cells", "cells"]
    assert _errors(wb.spec) == [] or all("boundary" in m or "fixed" in m
                                         for m in _errors(wb.spec))


def test_a_generated_window_is_not_resized_or_moved_by_hand(wb):
    """It follows the shape: no corners, no diamond, no place to type; a move asked
    for in code is refused."""
    import panel as pn
    ed = wb.geo_editor
    ed._switch(layer="windows")
    ed = wb.geo_editor
    ed.select("windows:0")
    assert wb.spec.windows[0].shape == "cells"
    assert ed._boxes() == [] and ed._handle_rows["move"] == []
    assert not [w for w in walk(ed.inspector) if isinstance(w, pn.widgets.IntInput)]
    before = [w.runs for w in wb.spec.windows]
    ed.translate_shape("windows:0", 8.0, 0.0)
    assert [w.runs for w in wb.spec.windows] == before and wb.spec.layout is not None
    assert "is not drawn here" in wb.log_lines[0]
