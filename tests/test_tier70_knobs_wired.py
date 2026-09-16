"""Tier 70 -- rake and the duct area wired, and the knob layer criterion 2 needs.

These pin the things a knob layer can get silently wrong once it can actually
move something: a knob that changes the NOMINAL car under every cached field and
record, a dashboard that accepts a value for a control that controls nothing, a
geometry change that re-cuts the car inside a frame, and a demo that marches one
car while its sliders show another.
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

import pytest                                                           # noqa: E402

from atlas.cases import car_knobs as CK                                 # noqa: E402
from atlas.cases import car_solids as CS                                # noqa: E402
from atlas.cases import racelab as RL                                   # noqa: E402

RECORD = os.path.join(HERE, "out", "racelab19", "racelab19.json")
PAGE = os.path.join(HERE, "wiki", "concepts", "Atlas 0.1", "common",
                    "poc3-racelab-knobs-wired.md")

#: The car every committed record and cache was built on.
NOMINAL = "a1f67a8e279280a2e6b98ca77a01017afed294b4a1e1388fda04af09e5e77a8f"


# ---------------------------------------------------------------------------
# the check that matters most
# ---------------------------------------------------------------------------


def test_wiring_the_knobs_did_not_move_the_nominal_car():
    """**A knob must not change the car it was wired on.**  Every settled field,
    raster cache and record of Tiers 62 to 68 is keyed on this fingerprint, and
    `rake` spent its whole life declared-but-dead, so the car they were all
    built on is the rake = 0 car whatever the default said."""
    assert RL.geometry_fingerprint() == NOMINAL
    assert RL.CarParams().rake == 0.0
    assert CS.SOLIDS_DEFAULT["duct_area"] == 1.0
    # and the knobs really do move it, so the equality above is not vacuous
    assert RL.geometry_fingerprint(RL.CarParams(rake=0.10)) != NOMINAL


# ---------------------------------------------------------------------------
# the state a screen would drive
# ---------------------------------------------------------------------------


def test_every_offerable_knob_reports_what_responded():
    st = CK.KnobState()
    for k in CK.KNOBS:
        if k in CK.must_not_be_shown():
            continue
        target = k.hi if st.values[k.name] != k.hi else k.lo
        r = st.set(k.name, target)
        assert r["responded"], f"{k.name} moved and nothing responded"
        assert r["knob"] == k.name and r["to"] == target


def test_a_knob_that_reaches_nothing_is_refused_where_the_value_enters():
    """Section 3.3's rule is worth nothing if a dashboard can still set it."""
    st = CK.KnobState()
    hidden = CK.must_not_be_shown()
    assert {k.name for k in hidden} == {"road_speed", "battery_power"}
    for k in hidden:
        with pytest.raises(ValueError) as exc:
            st.set(k.name, k.hi)
        assert "reaches nothing" in str(exc.value)
        assert st.values[k.name] == k.default, "a refused knob must not have moved"


def test_a_value_outside_its_declared_range_is_refused():
    st = CK.KnobState()
    with pytest.raises(ValueError):
        st.set("rake", 99.0)
    with pytest.raises(ValueError):
        st.set("duct_area", 0.0)
    with pytest.raises(KeyError):
        st.set("wing_mirrors", 1.0)
    assert st.values["rake"] == 0.0


def test_a_geometry_knob_is_marked_as_needing_a_regrid():
    """A composite rebuild costs 66-90 s, so a geometry knob cannot re-cut the
    car inside a frame; section 4.1 asks for a visible recompiling state."""
    st = CK.KnobState()
    assert st.set("rake", 0.15)["needs_regrid"] is True
    assert st.set("duct_area", 1.2)["needs_regrid"] is True
    assert st.set("ambient_t", 310.0)["needs_regrid"] is False
    assert st.set("coolant_mdot", 0.2)["needs_regrid"] is False
    assert set(CK.REGRID) == {"ride_height", "rake", "diffuser_deg",
                              "front_flap_deg", "rear_wing_deg", "duct_area"}


def test_the_state_builds_the_objects_the_march_would_use():
    st = CK.KnobState()
    st.set("rake", 0.15)
    st.set("duct_area", 0.5)
    p = st.car_params()
    assert p.rake == 0.15
    assert st.solids_rule()["duct_area"] == 0.5
    # and the rule is a COPY: moving a knob must not edit the package default
    assert CS.SOLIDS_DEFAULT["duct_area"] == 1.0


def test_the_demo_column_says_which_car_it_is_marching():
    """A demo that marched the old car while the sliders showed the new one is
    the defect this project keeps finding."""
    from atlas.demo_racelab import bodyfitted as BF

    col = BF.BodyFittedColumn()
    r = col.set_knob("rake", 0.15)
    assert r["needs_regrid"] is True
    assert r["marching"] == "the car BEFORE this change"
    assert col.pending_regrid == ["rake"]
    r2 = col.set_knob("ambient_t", 310.0)
    assert r2["needs_regrid"] is False and r2["marching"] == "current"
    assert col.pending_regrid == ["rake"], "a lumped knob must not add a regrid"


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def record():
    if not os.path.isfile(RECORD):
        pytest.skip("out/racelab19/racelab19.json is not here (carried upstream only)")
    with open(RECORD, encoding="utf-8") as fh:
        return json.load(fh)


def test_the_record_s_predictions_are_this_script_s_and_every_one_is_judged(record):
    import tier70_knobs_wired as T
    assert record["prediction"] == [dict(p) for p in T.PREDICTION]
    assert record["read_before_this_run"] == list(T.READ_BEFORE_THIS_RUN)
    assert record["verdicts_missing"] == []
    assert T.judge(record) == record["verdicts"]


def test_the_record_shows_the_nominal_car_held(record):
    n = record["nominal"]
    assert n["unchanged"] is True
    assert n["fingerprint"] == NOMINAL
    assert n["rigid"] is True


def test_the_page_says_what_the_record_measured(record):
    if not os.path.isfile(PAGE):
        pytest.skip("the page is not here")
    text = open(PAGE, encoding="utf-8").read()
    for pid in record["verdicts"]:
        assert ("| %s |" % pid) in text, pid
