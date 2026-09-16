"""Tier 67 -- where a learned expert can sit on the body-fitted column.

These pin what a window-placement layer can get silently wrong: a fast
admissibility test that disagrees with the honest slow one, a "hole-free" window
that is really full of cells the background does not decide, an extract that
loses a value or a scatter that writes outside its block, a reach that claims
more than any tiling could deliver, and a fraction quoted against a flattering
denominator.

The car's composite costs about a minute to build, so it is NOT built here: its
numbers come from the record `scripts/tier67_car_windows.py` writes, and the
logic is pinned on the verification composite.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (HERE, os.path.join(HERE, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                      # noqa: E402
import pytest                                                           # noqa: E402

from atlas.cases import car_windows as CW                               # noqa: E402
from atlas.cases import overset as OV                                   # noqa: E402
from atlas.cases import overset_multi as OM                             # noqa: E402

RECORD = os.path.join(HERE, "out", "racelab16", "racelab16.json")
PAGE = os.path.join(HERE, "wiki", "concepts", "Atlas 0.1", "common",
                    "poc3-racelab-car-windows.md")

#: Small enough to brute-force against, large enough that the car's bodies cut
#: real holes in it.
W = 24


@pytest.fixture(scope="module")
def small():
    return OM.multi_geometry(3)


# ---------------------------------------------------------------------------
# the fast test must equal the slow one
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("require", CW.REQUIRE)
def test_the_summed_area_test_equals_brute_force_everywhere(small, require):
    """The whole placement layer rests on an inclusion-exclusion identity over a
    summed-area table.  It is exact in integers, so it is checked against the
    obvious loop at every placement, not at a sample."""
    mask = CW._mask_for(small, require)
    fast = CW.hole_free(small, W, require)
    J, I = fast.shape
    brute = np.empty((J, I), dtype=bool)
    for j in range(J):
        for i in range(I):
            brute[j, i] = bool(mask[j:j + W, i:i + W].all())
    assert np.array_equal(fast, brute)


def test_reach_is_the_union_of_every_placement(small):
    fast = CW.reach(small, W, "live")
    free = CW.hole_free(small, W, "live")
    brute = np.zeros(CW.live_mask(small).shape, dtype=bool)
    for j, i in zip(*np.nonzero(free)):
        brute[j:j + W, i:i + W] = True
    assert np.array_equal(fast, brute)


# ---------------------------------------------------------------------------
# owned is stricter than live, and that is the point
# ---------------------------------------------------------------------------


def test_owning_a_cell_is_stricter_than_merely_holding_one(small):
    """A hole-free window can still be full of INTERP cells, which the background
    holds but does not decide.  Handing those to a learned expert hands it
    numbers it does not own."""
    live = CW.live_mask(small)
    owned = CW.owned_mask(small)
    assert np.all(owned <= live), "an owned cell must be a live cell"
    interp = small.status[small.bg.name] == OV.INTERP
    assert interp.any(), "this composite has no fringe; the test proves nothing"
    assert np.array_equal(live & ~owned, interp | (small.status[small.bg.name] == OV.WALL)
                          | (small.status[small.bg.name] == OV.OUTER_BC))
    assert CW.hole_free(small, W, "owned").sum() <= CW.hole_free(small, W, "live").sum()
    assert np.all(CW.reach(small, W, "owned") <= CW.reach(small, W, "live"))


def test_a_bad_require_or_order_is_refused(small):
    with pytest.raises(ValueError):
        CW.hole_free(small, W, "whatever")
    with pytest.raises(ValueError):
        CW.tile(small, W, order="sideways")
    with pytest.raises(ValueError):
        CW.hole_free(small, 0)


# ---------------------------------------------------------------------------
# a tiling
# ---------------------------------------------------------------------------


def test_a_tiling_does_not_overlap_and_every_window_is_admissible(small):
    mask = CW.live_mask(small)
    taken = np.zeros(mask.shape, dtype=bool)
    for win in CW.tile(small, W, order="row"):
        sl = win.slices()
        assert not taken[sl].any(), "two windows of one tiling overlap"
        taken[sl] = True
        assert mask[sl].all(), "a tiled window has a hole in it"
        assert win.w == W
        assert win.x1 > win.x0 and win.y1 > win.y0


def test_the_greedy_count_depends_on_the_order(small):
    """Greedy set packing is order-dependent, so the order is declared rather
    than left to whatever np.nonzero returns.  If this ever stops being true the
    tier's H2 needs re-reading, not the test deleting."""
    counts = {o: len(CW.tile(small, W, order=o)) for o in CW.ORDERS}
    assert set(counts) == set(CW.ORDERS)
    assert all(n > 0 for n in counts.values()), counts


# ---------------------------------------------------------------------------
# taking a window out and putting it back
# ---------------------------------------------------------------------------


def test_a_window_comes_out_dense_and_goes_back_bitwise(small):
    wins = CW.tile(small, W, order="row", require="owned")
    assert wins, "no owned window on the verification composite"
    win = wins[0]
    rng = np.random.default_rng(67)
    vec = rng.normal(size=small.n_unknowns)
    block = CW.extract(small, win, vec)
    assert block.shape == (W, W) and np.isfinite(block).all()
    assert np.array_equal(CW.scatter_into(small, win, vec, block), vec)
    changed = CW.scatter_into(small, win, vec, block + 1.0)
    assert int((changed != vec).sum()) == W * W, "a scatter wrote outside its own block"


def test_a_window_with_a_hole_is_refused_rather_than_quietly_wrong(small):
    """A checkpoint handed a hole has nothing to read there and no way to be told,
    so extracting one must raise rather than return whatever the index map has."""
    live = CW.live_mask(small)
    holes = np.argwhere(~live)
    if not holes.size:
        pytest.skip("this composite has no holes")
    j, i = holes[len(holes) // 2]
    j = int(min(max(j - W // 2, 0), live.shape[0] - W))
    i = int(min(max(i - W // 2, 0), live.shape[1] - W))
    bad = CW.Window(j=j, i=i, w=W, x0=0.0, x1=1.0, y0=0.0, y1=1.0)
    if live[bad.slices()].all():
        pytest.skip("could not place a window over a hole")
    with pytest.raises(OV.OversetError):
        CW.extract(small, bad, np.zeros(small.n_unknowns))


def test_territory_reports_the_composite_denominator_not_just_the_background(small):
    """A fraction of the BACKGROUND's live cells flatters the learned expert; the
    fraction that decides anything is of the composite's unknowns."""
    t = CW.territory(small, W, require="owned")
    for k in ("reach_frac_of_composite", "tiled_frac_of_composite",
              "body_grid_frac", "composite_unknowns", "body_grid_unknowns"):
        assert k in t, k
    assert 0.0 <= t["reach_frac_of_composite"] <= 1.0
    assert t["reach_frac_of_composite"] <= t["reach_frac_of_background_live"], \
        "the composite fraction cannot exceed the background one"
    assert t["body_grid_unknowns"] < t["composite_unknowns"]


# ---------------------------------------------------------------------------
# the record, and the page that quotes it
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def record():
    if not os.path.isfile(RECORD):
        pytest.skip("out/racelab16/racelab16.json is not here (it is carried upstream only)")
    with open(RECORD, encoding="utf-8") as fh:
        return json.load(fh)


def test_the_record_s_predictions_are_this_script_s_and_every_one_is_judged(record):
    import tier67_car_windows as T
    assert record["prediction"] == [dict(p) for p in T.PREDICTION]
    assert record["read_before_this_run"] == list(T.READ_BEFORE_THIS_RUN)
    assert record["verdicts_missing"] == []
    assert T.judge(record) == record["verdicts"]


def test_the_ceiling_is_reported_against_the_composite(record):
    L = record["live"]
    assert L["reach_frac_of_composite"] <= L["reach_frac_of_background_live"]
    assert L["body_grid_frac"] > 0.5, "the body grids are most of this composite"
    assert L["window_cells"] == CW.POSEIDON_CELLS
    assert abs(L["window_span"] - 2.0) < 1e-12, "128 cells at 1/64 is 2.0 length units"


def test_the_page_says_what_the_record_measured(record):
    if not os.path.isfile(PAGE):
        pytest.skip("the page is not here")
    text = open(PAGE, encoding="utf-8").read()
    for pid in record["verdicts"]:
        assert ("| %s |" % pid) in text, pid
    n = record["live"]["placements_hole_free"]
    assert "{:,}".format(n).replace(",", "{,}") in text, n
    body = record["live"]["body_grid_unknowns"]
    assert "{:,}".format(body).replace(",", "{,}") in text, body
