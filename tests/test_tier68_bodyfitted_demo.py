"""Tier 68 -- the body-fitted car as a demo column, and the cache that starts it.

These pin what a cached operator and a released column can get silently wrong: a
cache that is served although the geometry moved, a validator that never fires,
a control that passes vacuously on a raster drawing nothing, a column released
from a state settled for a different machine, and a field or a referent that is
missing with no reason given.

The car's composite costs about 90 s to build, so it is NOT built here: the
cache and the validator are pinned on the verification composite, the column's
declarations are pinned directly, and the car's own numbers come from the record
`scripts/tier68_bodyfitted_demo.py` writes.
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
from atlas.cases import overset_multi as OM                             # noqa: E402
from atlas.demo_racelab import bodyfitted as BF                         # noqa: E402

RECORD = os.path.join(HERE, "out", "racelab17", "racelab17.json")
PAGE = os.path.join(HERE, "wiki", "concepts", "Atlas 0.1", "common",
                    "poc3-racelab-bodyfitted-demo.md")


@pytest.fixture(scope="module")
def small():
    return OM.multi_geometry(3)


@pytest.fixture(scope="module")
def built(small, tmp_path_factory):
    d = tmp_path_factory.mktemp("t68")
    p = str(d / "raster.npz")
    r, info = CR.cached_raster(small, 40, 24, path=p)
    return small, r, info, p


# ---------------------------------------------------------------------------
# the cache
# ---------------------------------------------------------------------------


def test_a_miss_builds_and_a_hit_reproduces_it_bitwise(built):
    small, r1, info1, p = built
    assert info1["hit"] is False and info1["build_s"] > 0.0
    r2, info2 = CR.cached_raster(small, 40, 24, path=p)
    assert info2["hit"] is True and info2["why"] == "hit"
    assert np.array_equal(r1.M.data.view(np.uint64), r2.M.data.view(np.uint64))
    assert np.array_equal(r1.M.indices, r2.M.indices)
    assert np.array_equal(r1.M.indptr, r2.M.indptr)
    assert np.array_equal(r1.mask, r2.mask)
    assert np.array_equal(r1.source, r2.source)


def test_the_key_moves_with_the_raster(small):
    ext = (float(small.bg.X.min()), float(small.bg.X.max()),
           float(small.bg.Y.min()), float(small.bg.Y.max()))
    a = CR.raster_cache_key(small, 40, 24, ext)
    assert a == CR.raster_cache_key(small, 40, 24, ext)
    assert a != CR.raster_cache_key(small, 41, 24, ext)
    assert a != CR.raster_cache_key(small, 40, 25, ext)
    assert a != CR.raster_cache_key(small, 40, 24, (ext[0], ext[1] * 0.9, ext[2], ext[3]))


@pytest.mark.parametrize("what", ["weights", "too_many_masked", "too_few_masked"])
def test_the_validator_fires_on_a_cache_that_is_wrong(built, what, tmp_path):
    """**A validator that never fires is decoration.**  The cache does not trust
    its key -- a key only covers the inputs someone remembered to hash -- so each
    way of being wrong is planted and must be caught."""
    small, r, _info, p = built
    z = dict(np.load(p, allow_pickle=True))
    key = str(z["key"])
    if what == "weights":
        z["indices"] = (z["indices"] + 7) % int(z["shape"][1])
    elif what == "too_many_masked":
        m = z["mask"].ravel().copy()
        m[np.flatnonzero(~m)[:1]] = True          # ONE pixel too many
        z["mask"] = m.reshape(z["mask"].shape)
    else:
        m = z["mask"].ravel().copy()
        m[np.flatnonzero(m)[:1]] = False          # ONE pixel too few
        z["mask"] = m.reshape(z["mask"].shape)
    bad = str(tmp_path / "bad.npz")
    np.savez(bad, **z)
    got, why = CR.load_raster(small, bad, key)
    assert got is None, f"a {what} cache was served as sound"
    assert why and why != "hit"


def test_a_wrong_key_is_refused_before_anything_else(built):
    small, _r, _info, p = built
    got, why = CR.load_raster(small, p, "de" * 16)
    assert got is None and "key differs" in why


def test_a_missing_file_is_a_miss_and_not_an_error(small, tmp_path):
    got, why = CR.load_raster(small, str(tmp_path / "nope.npz"), "x" * 32)
    assert got is None and why == "no cache file"


def test_the_linear_control_cannot_pass_vacuously(built):
    """A raster that draws nothing used to reduce over an empty selection and
    raise; it must report inf, because a picture of nothing passes no control."""
    small, r, _info, _p = built
    blind = CR.CompositeRaster.__new__(CR.CompositeRaster)
    blind.__dict__.update(r.__dict__)
    blind.mask = np.ones_like(r.mask)
    c = CR.linear_control(blind)
    assert c["linear_max_err"] == float("inf")
    assert c["quadratic_max_err"] == float("inf")
    assert c.get("drawn") == 0


# ---------------------------------------------------------------------------
# what the column declares
# ---------------------------------------------------------------------------


def test_the_column_releases_from_a_settled_state_not_the_prefix():
    """**A release state belongs to its machine.**  The prefix is device-free, so
    releasing there puts the rotor outside its envelope from step 1."""
    cfg = BF.BodyFittedConfig()
    assert cfg.release_t == 12.0
    assert cfg.prefix.endswith("prefix_t8.npz")
    assert cfg.release_t > 8.0, "the release must be settled PAST the device-free prefix"


def test_the_referent_is_off_and_refused_with_a_reason():
    cfg = BF.BodyFittedConfig()
    assert cfg.referent is False
    col = BF.BodyFittedColumn()
    col.coverage = {"windows_learned": 0}
    st = col.referent_state()
    assert st["on"] is False and st["available"] is False
    assert "identically zero" in st["why"]
    col.coverage = {"windows_learned": 1}
    st2 = col.referent_state()
    assert st2["available"] is True and "halves" in st2["why"]


def test_every_field_the_column_cannot_draw_is_named_with_a_reason():
    """Section 4.3's rule, applied to fields: what is not offered is greyed out
    WITH the reason, never silently absent."""
    assert "vorticity" in BF.FIELD_HOLES and "derivative" in BF.FIELD_HOLES["vorticity"]
    assert "temperature" in BF.FIELD_HOLES and "lumped" in BF.FIELD_HOLES["temperature"]
    assert "error" in BF.FIELD_HOLES
    for name, why in BF.FIELD_HOLES.items():
        assert name not in BF.FIELDS, f"{name} is both offered and named as a hole"
        assert len(why) > 40, f"{name}'s reason is too short to be a reason"
    assert set(BF.FIELDS) == {"speed", "u", "v", "p"}


def test_the_learned_hole_names_the_fraction_and_the_reason():
    assert "61.0%" in BF.LEARNED_HOLE
    assert "128" in BF.LEARNED_HOLE and "curvilinear" in BF.LEARNED_HOLE
    assert "W288" in BF.LEARNED_HOLE


def test_a_column_that_has_not_been_built_refuses_to_step():
    with pytest.raises(RuntimeError):
        BF.BodyFittedColumn().step()


def test_steps_per_second_ignores_the_refactoring_step():
    """The first step after a release refactors the incomplete LU and is about
    five times a typical one, so including it would report a rate the column
    never runs at."""
    col = BF.BodyFittedColumn()
    col.steps = [{"wall_s": 3.5, "ilu_s": 3.0}, {"wall_s": 0.5, "ilu_s": 0.0},
                 {"wall_s": 0.5, "ilu_s": 0.0}]
    assert col.steps_per_second() == pytest.approx(2.0)
    col.steps = [{"wall_s": 3.5, "ilu_s": 3.0}]
    assert col.steps_per_second() is None


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def record():
    if not os.path.isfile(RECORD):
        pytest.skip("out/racelab17/racelab17.json is not here (carried upstream only)")
    with open(RECORD, encoding="utf-8") as fh:
        return json.load(fh)


def test_the_record_s_predictions_are_this_script_s_and_every_one_is_judged(record):
    import tier68_bodyfitted_demo as T
    assert record["prediction"] == [dict(p) for p in T.PREDICTION]
    assert record["read_before_this_run"] == list(T.READ_BEFORE_THIS_RUN)
    assert record["verdicts_missing"] == []
    assert T.judge(record) == record["verdicts"]


def test_the_released_column_starts_inside_its_envelope(record):
    s = record["settle"]
    assert s["declined_steps"] == 0, s
    assert s["band"]["ratio_lo"] <= s["ratio_first"] <= s["band"]["ratio_hi"], s


def test_the_page_says_what_the_record_measured(record):
    if not os.path.isfile(PAGE):
        pytest.skip("the page is not here")
    text = open(PAGE, encoding="utf-8").read()
    for pid in record["verdicts"]:
        assert ("| %s |" % pid) in text, pid
