"""A river that forks into two outlets (2026-09-30).

The owner: "In the stream pollutant case, it doesn't let me have a geometry with a
bifurcation, and two outlets."  Reproduced first (demo-finish-plan section 2.2): a Y
ran, but the starter gave it one outlet and said nothing of the branch that carried no
water, and the automatic windows, run between the Y's two branch tips, split its stem
lengthwise.  These tests pin the repair:

* a branched shape is found, and only a branched one: the harmonic coordinate between
  its two ends leaves a dead region (`layout.dead_regions`), which no drawn example, no
  rectangle and no L has;
* a branched river's windows follow its own flow, a band across both branches two
  windows; any other branched shape is cut by recursive spectral bisection;
* a branch with no water is a warning with a fix, and a newly drawn fork gets its
  second outlet at once, said in the note;
* the `river-fork` example: every check, registered before its first run, the flow's
  continuity among them; each outlet's share; the one-window control, bit for bit;
  its compile's verdict.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from atlas.workbench import geometry as geo  # noqa: E402
from atlas.workbench import layout as L  # noqa: E402
from atlas.workbench.families import plume as pl  # noqa: E402
from atlas.workbench.spec import (EXAMPLES, Boundary, Layout, Outline, Window,  # noqa: E402
                                  check, default_boundaries, example_case)
from atlas.workbench.starter import apply_fix  # noqa: E402

#: the Y drawn in the page on step 1 (2026-09-30): the stem from the left, two branches
Y = [(24, 104), (152, 104), (328, 32), (328, 72), (200, 120), (328, 168), (328, 208),
     (152, 136), (24, 136)]


def _parts(mask):
    from scipy.ndimage import label
    return label(mask)[1]


def _issues(spec, severity=("error", "warning")):
    return [i for i in check(spec) if i.severity in severity]


# ---------------------------------------------------------------------------
# finding a branch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", [k for k in EXAMPLES
                                 if example_case(k).domain.outline is not None
                                 or example_case(k).domain.holes or k == "wall-2"])
def test_no_example_is_branched(key):
    """No drawn example, and no plain one, has a cell the coordinate between its two
    ends does not reach, but the fork: so every other example's windows are cut as
    they were."""
    d = example_case(key).domain
    act = geo.domain_mask(d)
    along, _a, _s, _e = L.coordinates(d)
    dead = L.dead_regions(L.cell_speed(along, act), act, min_cells=1)
    if key == "river-fork":
        assert dead
    else:
        assert dead == [], key


def test_the_y_and_an_l_and_a_rectangle():
    s = example_case("plate-insert")
    for outline, branched in (
            (Outline(points=Y, edges=["line"] * 9, bulge=[0.0] * 9), True),
            (Outline(points=[(0, 0), (160, 0), (160, 40), (40, 40), (40, 96), (0, 96)]),
             False),
            (Outline(points=[(8, 8), (152, 8), (152, 88), (8, 88)]), False)):
        s.domain.nx, s.domain.ny = 352, 240
        s.domain.outline = outline
        act = geo.domain_mask(s.domain)
        along, _a, _s, _e = L.coordinates(s.domain)
        assert bool(L.dead_regions(L.cell_speed(along, act), act)) is branched


# ---------------------------------------------------------------------------
# the windows
# ---------------------------------------------------------------------------


def test_a_forked_river_is_cut_along_its_own_flow():
    s = example_case("river-fork")
    assert "river's own flow" in s.layout.how
    names = [w.id for w in s.windows]
    assert names == ["F00", "F10", "F20", "F21", "F30", "F31"]      # two bands in two
    for _wid, m in geo.window_masks(s):
        assert _parts(m) == 1
    # every window lies in one branch or in the stem: the two windows of a band are the
    # band's two branches, and none crosses the land between them
    masks = dict(geo.window_masks(s))
    assert not (masks["F20"] & masks["F21"]).any() and not (masks["F30"] & masks["F31"]).any()


def test_a_branched_shape_with_no_flow_is_cut_by_bisection():
    """A Y of steel, hot on its stem and cold on both branch ends: no flow to follow,
    so recursive spectral bisection; every piece in one part, and the same pieces every
    time."""
    s = example_case("s-channel")
    s.domain.nx, s.domain.ny = 352, 240
    s.domain.outline = Outline(points=Y, edges=["line"] * 9, bulge=[0.0] * 9)
    s.boundaries = default_boundaries(s)
    s.boundaries[8] = Boundary(id="hot", edge="outline:8", kind="fixed-temperature",
                               value=400.0)
    for k, bid in ((2, "cold"), (5, "cold2")):
        s.boundaries[k] = Boundary(id=bid, edge=f"outline:{k}", kind="fixed-temperature",
                                   value=300.0)
    s.regions[0].nx, s.regions[0].ny = 352, 240
    s.layout = Layout(cut="along", along=4)
    parts, how = L.pieces_and_how(s)
    assert "bisection" in how and len(parts) == 4
    assert all(_parts(m) == 1 for _n, m in parts)
    assert np.array_equal(np.logical_or.reduce([m for _n, m in parts]),
                          geo.domain_mask(s.domain))
    again, _h = L.pieces_and_how(s)
    assert all(np.array_equal(a, b) for (_x, a), (_y, b) in zip(parts, again))
    L.refresh(s)
    assert not [i.message for i in _issues(s, ("error",))]


# ---------------------------------------------------------------------------
# a branch with no water
# ---------------------------------------------------------------------------


def test_a_branch_with_no_outlet_is_a_warning_with_a_fix():
    s = example_case("river-fork")
    k = next(i for i, b in enumerate(s.boundaries) if b.edge == "outline:12")
    s.boundaries[k] = Boundary(id=s.boundaries[k].id, edge="outline:12", kind="bank")
    warn = [i for i in _issues(s) if i.fix == "dead-branch"]
    assert len(warn) == 1 and "carries no water" in warn[0].message
    assert "an outlet on outline:12" in apply_fix(s, "dead-branch")
    assert s.boundaries[k].kind == "river-outlet"
    assert not [i for i in _issues(s) if i.fix == "dead-branch"]


@pytest.fixture()
def wb(tmp_path, monkeypatch):
    from atlas.workbench import runner
    from atlas.workbench.app import Workbench
    monkeypatch.setattr(runner.CaseRun, "start", lambda self: None)
    return Workbench(cases_dir=str(tmp_path))


def test_the_owners_fork_drawn_in_the_page_gets_both_outlets_and_follows_its_flow(wb):
    """The done-when of item 1.2, headless: the river chosen, a Y drawn with its stem on
    the left.  The inlet goes on the stem, an outlet on each branch (the second said in
    the note), and the windows follow the flow: nothing is left to fix."""
    wb.type_sel.value = "transport-2d"
    wb.geo_editor._switch(layer="domain", tool="draw")
    wb.geo_editor.add_drawn(Outline(points=Y, edges=["line"] * 9, bulge=[0.0] * 9))
    s = wb.spec
    kinds = {b.edge: b.kind for b in s.boundaries}
    assert kinds["outline:8"] == "river-inlet"
    assert kinds["outline:2"] == kinds["outline:5"] == "river-outlet"
    assert not _issues(s)
    assert "river's own flow" in s.layout.how
    assert any("end of a branch that carried no water" in line for line in wb.log_lines[:2])
    from workbench_ui import texts
    wb.geo_editor._switch(layer="windows")
    assert "Cut along the river's own flow" in texts(wb.geo_editor.inspector)


def test_the_owners_fork_sends_the_plume_down_both_branches(wb):
    """The served page's first walk (2026-09-30) found the drawn Y's outfall at its
    fork: the starting values placed it before the river had its ends, at the middle
    of its cells, and a release on the dividing streamline went down one branch alone
    (100% and 0% of the pollutant).  The outfall now goes mid-river a fifth of the way
    down the flow, and the plume reaches both outlets."""
    wb.type_sel.value = "transport-2d"
    wb.geo_editor._switch(layer="domain", tool="draw")
    wb.geo_editor.add_drawn(Outline(points=Y, edges=["spline"] * 9, bulge=[0.0] * 9))
    s = wb.spec
    assert not _issues(s)
    i, j = pl.outfall_cell(s)
    phi = pl.river_flow(s).phi
    # placed at 0.8 while the river had one outlet; its second puts more of the drop
    # in the stem, so the same cell reads a little lower now: still the stem, upstream
    assert 0.7 < phi[j, i] < 0.9 and i < 152
    r = pl.build(s, arms=("full",), threads=1)
    st = r.initial("full")
    for _ in range(s.run.steps):
        st = r.step("full", st)
    shares = r.outlet_shares(st.u)
    r.close()
    assert len(shares) == 2
    assert all(v["pollutant_share"] > 0.3 for v in shares.values()), shares


# ---------------------------------------------------------------------------
# the example
# ---------------------------------------------------------------------------


def test_the_forks_checks_were_registered_before_its_first_run():
    assert {c.key: c.tolerance for c in pl.CHECKS} == {
        "mass": 1e-9, "reference": 1e-10, "bitwise": None, "continuity": 1e-12}
    c = next(c for c in pl.CHECKS if c.key == "continuity")
    assert "before the forked river's first run" in c.registered


def test_the_fork_runs_every_check_and_reports_each_outlet():
    s = example_case("river-fork")
    assert not _issues(s)
    r = pl.build(s, arms=("serial", "parallel", "full"), threads=2)
    st = {a: r.initial(a) for a in r.arms}
    hist = {a: [] for a in r.arms}
    same = True
    for _ in range(600):
        for a in r.arms:
            prev = st[a]
            st[a] = r.step(a, prev)
            hist[a].append(r.observe(a, st[a], prev))
        same &= r.bitwise_equal(st["serial"], st["parallel"])
    m, checks = r.compare(st, hist, {"steps": 600, "first_difference": None if same else 1})
    notes = r.notes(600)
    r.close()
    for c in checks:
        assert c.passed is not False, c.as_dict()
    out = m["full"]["outlets"]
    assert set(out) == {"B-outline-5", "B-outline-12"}
    assert sum(o["water_share"] for o in out.values()) == pytest.approx(1.0, abs=1e-12)
    # the wider branch takes more of the water
    assert out["B-outline-5"]["water_share"] > out["B-outline-12"]["water_share"] > 0.3
    assert any("leaves by 2 outlets" in n for n in notes)


def test_one_window_over_the_forked_river_is_the_full_river_to_the_bit():
    s = example_case("river-fork")
    s.layout = None
    s.windows = [Window(id="ALL", shape="cells",
                        runs=geo.mask_to_runs(geo.domain_mask(s.domain)))]
    assert not [i for i in _issues(s, ("error",))]
    r = pl.build(s, arms=("serial", "full"), threads=1)
    a, b = r.initial("serial"), r.initial("full")
    for _ in range(50):
        a, b = r.step("serial", a), r.step("full", b)
        assert np.array_equal(a.u, b.u)
    r.close()


def test_the_fork_compiles_with_a_seam_to_each_branch():
    from atlas.workbench.compile import compile_case
    summ = compile_case(example_case("river-fork"))
    assert summ.verdict == "admit-uncertified" and summ.agents == 6
    assert {sv.seam for sv in summ.seams} == {"F00|F10", "F10|F20", "F10|F21", "F20|F30",
                                               "F21|F31"}
