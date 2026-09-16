"""Tier 66 -- the body-fitted composite on one raster.

These pin what a picture of an overset composite can get silently wrong: a
renderer that stitches the grids in the wrong order, one that quietly draws the
background where a body grid owns the answer, one that draws fluid inside a car,
one whose picture stops agreeing with the solve, and a control that would pass
without touching a grid at all.

The car's own composite costs about a minute to build, so it is NOT built here:
its numbers are asserted from the record `scripts/tier66_car_render.py` writes,
and the renderer's LOGIC is pinned on the small verification composite, which is
the same `MultiOverset` machinery at 1/16 the cost.
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

from atlas.cases import car_render as CR                                # noqa: E402
from atlas.cases import car_union as CU                                 # noqa: E402
from atlas.cases import overset as OV                                   # noqa: E402
from atlas.cases import overset_multi as OM                             # noqa: E402

RECORD = os.path.join(HERE, "out", "racelab15", "racelab15.json")
PAGE = os.path.join(HERE, "wiki", "concepts", "Atlas 0.1", "common",
                    "poc3-racelab-car-render.md")


@pytest.fixture(scope="module")
def small():
    """The verification composite: a wheel, a panel, an aerofoil and a road patch.

    Level 3 because that is what Tiers 62 and 64 pin it at, and because level 1
    does not build at all: at h = 1/16 the wheel's wall lands within two grid
    rows of the road, which `overset_multi` refuses as a gap no interpolation
    can close.
    """
    return OM.multi_geometry(3)


@pytest.fixture(scope="module")
def raster(small):
    return CR.CompositeRaster(small, 48, 24)


# ---------------------------------------------------------------------------
# the operator: the solver's own, asked over a rectangle
# ---------------------------------------------------------------------------


def test_the_masked_probe_and_the_strict_one_are_the_same_search(small, raster):
    """Tier 66 made `probe_matrix` a wrapper so the picture and the devices'
    strips cannot come to disagree.  On points every grid can hold, the two must
    return the identical operator; on a point none can, one raises and the other
    reports.

    The points are taken from the raster's own drawn set rather than guessed, so
    the test cannot fail for the uninteresting reason that a hand-picked
    coordinate landed inside a body."""
    drawn = np.flatnonzero(~raster.mask.ravel())[::11][:12]
    px, py = raster.px[drawn], raster.py[drawn]
    M, missing, source = CU.probe_matrix_masked(small, px, py)
    assert not missing.any()
    strict = CU.probe_matrix(small, px, py)
    assert np.array_equal(M.indices, strict.indices)
    assert np.array_equal(M.data, strict.data)
    assert np.all(source >= 0)


def test_a_point_no_grid_holds_is_reported_by_one_and_fatal_to_the_other(raster, small):
    idx = np.flatnonzero(raster.mask.ravel())
    if not idx.size:
        pytest.skip("this composite leaves no raster point unreachable")
    px, py = raster.px[idx[:1]], raster.py[idx[:1]]
    _M, missing, source = CU.probe_matrix_masked(small, px, py)
    assert missing.all() and source[0] == -1
    with pytest.raises(OV.OversetError) as exc:
        CU.probe_matrix(small, px, py)
    assert "no usable stencil" in str(exc.value)


def test_the_mask_is_where_no_grid_holds_the_pixel(raster):
    assert raster.mask.shape == (raster.ny, raster.nx)
    assert raster.mask.any(), "a composite with bodies must mask something"
    assert not raster.mask.all()
    assert np.all(raster.source[raster.mask] == -1)
    assert np.all(raster.source[~raster.mask] >= 0)
    rep = raster.report()
    assert rep["masked"] + rep["drawn"] == rep["points"] == raster.nx * raster.ny
    assert sum(rep["per_grid"].values()) == rep["drawn"]


# ---------------------------------------------------------------------------
# the controls
# ---------------------------------------------------------------------------


def test_a_linear_field_comes_back_exact_and_a_quadratic_does_not(raster):
    """The donor weights reproduce linear fields, so a plane laid on every node
    must come back through the raster at rounding.  The quadratic beside it is
    what stops that from being vacuous: a renderer that ignored the grids and
    evaluated the plane analytically would pass the first check and fail this
    one."""
    c = CR.linear_control(raster)
    assert c["linear_max_err"] < 1e-11, c
    assert c["quadratic_max_err"] > 1e3 * max(c["linear_max_err"], 1e-16), c


def test_a_pixel_is_the_same_number_as_a_direct_probe(raster, small):
    rng = np.random.default_rng(66)
    vec = rng.normal(size=small.n_unknowns)
    drawn = np.flatnonzero(~raster.mask.ravel())[::7][:24]
    M, missing, _s = CU.probe_matrix_masked(small, raster.px[drawn], raster.py[drawn])
    assert not missing.any()
    assert np.array_equal((M @ vec), (raster.M @ vec)[drawn])


def test_the_operator_does_not_depend_on_the_state_or_on_the_build(raster, small):
    again = CR.CompositeRaster(small, raster.nx, raster.ny)
    assert np.array_equal(raster.mask, again.mask)
    assert np.array_equal(raster.source, again.source)
    assert np.array_equal(raster.M.indices, again.M.indices)
    assert np.array_equal(raster.M.indptr, again.M.indptr)
    assert np.array_equal(raster.M.data.view(np.uint64), again.M.data.view(np.uint64))


def test_sample_masks_the_car_and_checks_what_it_is_given(raster, small):
    rng = np.random.default_rng(1)
    vec = rng.normal(size=small.n_unknowns)
    a = raster.sample(vec)
    assert a.shape == (raster.ny, raster.nx)
    assert np.all(np.isnan(a[raster.mask]))
    assert np.all(np.isfinite(a[~raster.mask]))
    with pytest.raises(ValueError):
        raster.sample(vec[:-1])
    F = raster.fields(vec, vec[::-1].copy())
    assert set(F) == {"u", "v", "speed"}
    assert "p" in raster.fields(vec, vec, vec)
    assert np.allclose(F["speed"][~raster.mask],
                       np.hypot(F["u"], F["v"])[~raster.mask], equal_nan=False)


def test_a_raster_refuses_a_degenerate_request(small):
    with pytest.raises(ValueError):
        CR.CompositeRaster(small, 1, 10)
    with pytest.raises(ValueError):
        CR.CompositeRaster(small, 8, 8, extent=(1.0, 1.0, 0.0, 2.0))


# ---------------------------------------------------------------------------
# the picture
# ---------------------------------------------------------------------------


def test_the_png_draws_the_car_in_a_colour_the_field_cannot_make(raster, small):
    pytest.importorskip("PIL")
    from PIL import Image
    import io

    rng = np.random.default_rng(3)
    vec = rng.normal(size=small.n_unknowns)
    F = raster.fields(vec, vec[::-1].copy())
    png = CR.field_png(F["speed"], 0.0, 2.0)
    im = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
    assert im.shape == (raster.ny, raster.nx, 3)
    car = np.all(im == np.asarray(CR.CAR_RGB, dtype=np.uint8), axis=-1)
    # row 0 of the array is y = 0 and row 0 of the image is the top
    assert np.array_equal(car, raster.mask[::-1])
    lut = CR._lut("speed")
    assert not np.any(np.all(lut == np.asarray(CR.CAR_RGB, dtype=np.uint8), axis=-1))
    with pytest.raises(ValueError):
        CR.field_png(F["speed"], 1.0, 1.0)


def test_the_car_is_far_enough_from_the_field_to_read_as_an_object():
    """Exact inequality is not the claim.  The overlay's first colour was a
    near-black grey, which the map never produces exactly and which the eye
    could not tell from the slow air in the wake -- found by opening the PNG,
    not by any assertion.  This pins the distance instead: a mid grey scores
    3.6 and a slate 6.2, both of which would vanish into the field."""
    assert CR.car_colour_margin() >= CR.CAR_RGB_MIN_MARGIN, CR.car_colour_margin()
    assert CR.car_colour_margin((128, 128, 132)) < 10.0   # the control: a colour that fails


# ---------------------------------------------------------------------------
# the record, and the page that quotes it
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def record():
    if not os.path.isfile(RECORD):
        pytest.skip("out/racelab15/racelab15.json is not here (it is carried upstream only)")
    with open(RECORD, encoding="utf-8") as fh:
        return json.load(fh)


def test_the_record_s_predictions_are_this_script_s_and_every_one_is_judged(record):
    import tier66_car_render as T
    assert record["prediction"] == [dict(p) for p in T.PREDICTION]
    assert record["read_before_this_run"] == list(T.READ_BEFORE_THIS_RUN)
    assert record["verdicts_missing"] == []
    assert T.judge(record) == record["verdicts"]


def test_the_car_s_mask_is_the_car_at_every_raster(record):
    """G1, asserted from the record rather than rebuilt: the mask comes from the
    donor search and the outlines come from the drawing, and they must name the
    same car."""
    by = record["raster"]["by_raster"]
    assert len(by) >= 3
    for key, r in by.items():
        m = r["mask_vs_solids"]
        assert m["masked_not_inside"] == 0, (key, m)
        assert m["inside_not_masked"] == 0, (key, m)
        assert m["masked"] == m["both"] == m["inside_solid"], (key, m)


def test_the_background_alone_would_leave_the_interesting_part_blank(record):
    b = record["blank"]
    assert b["background_blank"] >= 3000, b
    assert b["background_can_draw"] + b["background_blank"] == b["drawable"]


def test_a_frame_is_cheap_against_a_step_measured_beside_it(record):
    c = record["state"]["cost"]
    assert c["ratio"] < 0.01, c
    assert c["apply_s"] > 0.0 and c["step_s"] > 0.0


def test_the_page_says_what_the_record_measured(record):
    if not os.path.isfile(PAGE):
        pytest.skip("the page is not here")
    text = open(PAGE, encoding="utf-8").read()
    for pid in record["verdicts"]:
        assert ("| %s |" % pid) in text, pid
    blank = record["blank"]["background_blank"]
    assert "{:,}".format(blank).replace(",", "{,}") in text, blank
    rows = record["raster"]["by_raster"]["320x180"]
    assert "{:,}".format(rows["masked"]).replace(",", "{,}") in text, rows["masked"]
