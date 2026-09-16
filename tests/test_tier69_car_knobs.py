"""Tier 69 -- section 3.3's parameters, and what each one actually reaches.

These pin the things a "knob layer" can get silently wrong: a reach test that
passes on a broken probe, a knob declared wired that reaches nothing, a repair
that changes the default behaviour it was supposed to preserve, and a subsystem
that reads a module constant instead of the neighbour it is wired to.
"""

from __future__ import annotations

import json
import math
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (HERE, os.path.join(HERE, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest                                                           # noqa: E402

from atlas.cases import car_knobs as CK                                 # noqa: E402
from atlas.cases import powertrain as PT                                # noqa: E402

RECORD = os.path.join(HERE, "out", "racelab18", "racelab18.json")
PAGE = os.path.join(HERE, "wiki", "concepts", "Atlas 0.1", "common",
                    "poc3-racelab-knobs.md")


# ---------------------------------------------------------------------------
# the test that decides every other one
# ---------------------------------------------------------------------------


def test_moved_refuses_nan_because_nan_is_not_equal_to_itself():
    """**The detector committed the defect it detects.**  The first reach test
    called a knob wired whenever its readings differed, and `nan != nan` is True
    in Python -- so a probe returning nan at both ends, because it called an API
    that did not exist, reported the knob as moving."""
    nan = float("nan")
    assert nan != nan                      # the language fact this rests on
    assert CK.moved(nan, nan) is False
    assert CK.moved(nan, 1.0) is False
    assert CK.moved(1.0, nan) is False
    assert CK.moved(float("inf"), 1.0) is False
    assert CK.moved(1.0, 2.0) is True
    assert CK.moved(1.0, 1.0) is False
    assert CK.moved("x", "y") is False


# ---------------------------------------------------------------------------
# the declarations
# ---------------------------------------------------------------------------


def test_every_knob_declares_a_range_a_target_and_a_verdict():
    assert len(CK.KNOBS) == 11, "section 3.3 lists eleven once brakes and stiffness are omitted"
    for k in CK.KNOBS:
        assert k.how in CK.HOW, (k.name, k.how)
        assert k.lo < k.hi, k.name
        assert k.lo <= k.default <= k.hi, (k.name, k.default)
        assert k.reaches, k.name
        assert k.probe, k.name
        if k.how != "wired":
            assert len(k.note) > 40, f"{k.name} is not wired and must say why, at length"


def test_a_knob_that_reaches_nothing_is_never_offered_as_working():
    """Section 3.3: a parameter that cannot reach its subsystem is omitted and
    the omission recorded.  So the two lists must not overlap."""
    wired = {k.name for k in CK.honestly_wired()}
    hidden = {k.name for k in CK.must_not_be_shown()}
    assert wired.isdisjoint(hidden)
    assert "rake" in hidden and "road_speed" in hidden and "duct_area" in hidden
    assert "battery_power" in hidden
    assert wired == {"ride_height", "diffuser_deg", "front_flap_deg", "rear_wing_deg"}


def test_the_omitted_knobs_from_amendment_13_1_are_kept_in_one_place():
    assert set(CK.OMITTED) == {"brake_duty", "brake_duct", "plate_stiffness"}
    for name, why in CK.OMITTED.items():
        assert len(why) > 20, name


def test_by_how_refuses_a_category_that_does_not_exist():
    with pytest.raises(ValueError):
        CK.by_how("probably")


# ---------------------------------------------------------------------------
# the geometry probe
# ---------------------------------------------------------------------------


def test_ride_height_moves_bodies_and_rake_moves_none():
    """The tier's headline: `rake` is declared as a geometry knob and appears
    exactly once in the package -- its own declaration."""
    assert len(CK.bodies_moved("ride_height", 0.15, 0.60)) >= 4
    assert CK.bodies_moved("rake", 0.0, 0.15) == []
    assert CK.bodies_moved("diffuser_deg", 0.0, 15.0) == ["DIFF"]
    assert CK.bodies_moved("front_flap_deg", 0.0, 30.0) == ["FW_FLAP"]
    assert CK.bodies_moved("rear_wing_deg", 0.0, 30.0) == ["RW_FLAP"]


def test_a_geometry_knob_at_one_value_moves_nothing_against_itself():
    """The probe's own control: same value, no movement, or the probe is noise."""
    assert CK.bodies_moved("ride_height", 0.22, 0.22) == []
    assert CK.bodies_moved("front_flap_deg", 21.8, 21.8) == []


# ---------------------------------------------------------------------------
# the machine and its battery
# ---------------------------------------------------------------------------


def test_the_machine_now_reads_the_battery_it_is_wired_to():
    """Before this tier `current_at` and `validity` read the MODULE constant, so
    a state-of-charge knob moved `BatteryLeg.emf` and left the machine drawing
    against a battery that was no longer there."""
    assert "v_oc" in PT.MachineAgent.__dataclass_fields__
    assert "r_total" in PT.MachineAgent.__dataclass_fields__
    m, low = PT.MachineAgent(), PT.MachineAgent(v_oc=0.80)
    assert m.current_at(20.0) != low.current_at(20.0)
    # the envelope predicate moves with it, which is the part W282 rests on
    assert m.validity([10.0]) is False
    assert low.validity([10.0]) is True


def test_the_repair_preserves_the_default_exactly():
    """The defaults ARE the module constants, so nothing that existed changes."""
    m = PT.MachineAgent()
    assert m.v_oc == PT.V_OC
    assert m.r_total == PT.R_TOTAL
    assert m.current_at(20.0) == (PT.K_E * 20.0 - PT.V_OC) / PT.R_TOTAL
    assert m.validity([20.0]) == bool(PT.K_E * 20.0 > PT.V_OC)


def test_the_battery_was_always_parameterised_and_still_is():
    assert PT.BatteryLeg().emf() == -PT.V_OC
    assert PT.BatteryLeg(v_oc=0.80).emf() == -0.80


# ---------------------------------------------------------------------------
# the cooling probe leaves nothing behind
# ---------------------------------------------------------------------------


def test_the_cooling_probe_restores_the_module_constants():
    """It reaches the loop by REBINDING a module constant, so it must put them
    back -- a constant left moved would make every later measurement wrong."""
    from atlas.cases import cooling_loop as CL

    before = (CL.MDOT, CL.T_AMB)
    lo = CK.loop_return(mdot=0.05)
    hi = CK.loop_return(mdot=0.30)
    assert (CL.MDOT, CL.T_AMB) == before
    assert CK.moved(lo, hi)
    assert math.isfinite(lo) and math.isfinite(hi)


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def record():
    if not os.path.isfile(RECORD):
        pytest.skip("out/racelab18/racelab18.json is not here (carried upstream only)")
    with open(RECORD, encoding="utf-8") as fh:
        return json.load(fh)


def test_the_record_s_predictions_are_this_script_s_and_every_one_is_judged(record):
    import tier69_car_knobs as T
    assert record["prediction"] == [dict(p) for p in T.PREDICTION]
    assert record["read_before_this_run"] == list(T.READ_BEFORE_THIS_RUN)
    assert record["verdicts_missing"] == []
    assert T.judge(record) == record["verdicts"]


def test_this_tier_added_no_artifact_to_the_moved_set(record):
    """L1 failed against the Tier 45 capture, which accumulates every deliberate
    change since; the number this tier answers for is what it ADDED."""
    c = record["control"]
    assert c["tested"] is True
    assert c["added_by_this_tier"] == [], c["added_by_this_tier"]
    assert c["compared"] == 40


def test_the_page_says_what_the_record_measured(record):
    if not os.path.isfile(PAGE):
        pytest.skip("the page is not here")
    text = open(PAGE, encoding="utf-8").read()
    for pid in record["verdicts"]:
        assert ("| %s |" % pid) in text, pid
