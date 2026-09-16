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
    the omission recorded.  So the two lists must not overlap.

    **Tier 69 measured four wired and this asserted four; Tier 70 wired `rake`
    and `duct_area` and it asserts six.**  The diagnosis is kept rather than
    deleted: `rake` WAS declared and dead, appearing once in the package, and
    `duct_area` WAS a set of literals nothing scaled -- that is what
    [[poc3-racelab-knobs]] records and what Tier 70's row repairs.
    """
    wired = {k.name for k in CK.honestly_wired()}
    hidden = {k.name for k in CK.must_not_be_shown()}
    assert wired.isdisjoint(hidden)
    assert "road_speed" in hidden, "the column is nondimensional; nothing reads a road speed"
    assert "battery_power" in hidden, "this circuit is a generator and accepts no demand"
    assert wired == {"ride_height", "rake", "diffuser_deg", "front_flap_deg",
                     "rear_wing_deg", "duct_area", "coolant_mdot", "ambient_t"}


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


def test_each_geometry_knob_moves_the_bodies_it_claims():
    """Tier 69 found `rake` moving NOTHING -- declared in CarParams and appearing
    exactly once in the package.  Tier 70 pitched the floor group with it."""
    assert len(CK.bodies_moved("ride_height", 0.15, 0.60)) >= 4
    assert CK.bodies_moved("diffuser_deg", 0.0, 15.0) == ["DIFF"]
    assert CK.bodies_moved("front_flap_deg", 0.0, 30.0) == ["FW_FLAP"]
    assert CK.bodies_moved("rear_wing_deg", 0.0, 30.0) == ["RW_FLAP"]
    assert CK.bodies_moved("rake", 0.0, 0.15) == [
        "DIFF", "DIFF_EXIT", "FLOOR", "FLOOR_LE", "FLOOR_STEP"]


def test_rake_is_a_rigid_pitch_about_the_floors_leading_edge():
    """A per-plate rise WITHOUT the angle would be a staircase pretending to be
    a ramp.  The pivot must not move, the rear must rise, and every plate's
    incidence must gain the same pitch."""
    import math

    from atlas.cases import racelab as RL

    x0, length = RL.rake_span()
    lo, hi = CK.built_bodies(rake=0.0), CK.built_bodies(rake=0.15)
    # FLOOR_LE sits at the pivot: its height must not move
    assert lo["FLOOR_LE"][1] == pytest.approx(hi["FLOOR_LE"][1], abs=1e-12)
    # DIFF is downstream: it rises by rake * (x - x0) / length
    dx = hi["DIFF"][0] - x0
    # `built_bodies` rounds to 9 decimals, so the tolerance is ABSOLUTE at that
    # scale; a relative one tighter than the probe reads fails on the rounding
    assert hi["DIFF"][1] - lo["DIFF"][1] == pytest.approx(0.15 * dx / length, abs=2e-9)
    # and every pitched plate gains the SAME angle
    pitch = math.degrees(math.atan(0.15 / length))
    for key in ("DIFF", "FLOOR", "FLOOR_LE", "FLOOR_STEP"):
        assert hi[key][3] - lo[key][3] == pytest.approx(pitch, abs=2e-9), key


def test_wiring_rake_did_not_move_the_nominal_car():
    """**The default had to move 0.10 -> 0.0 with it.**  While rake reached
    nothing, every car ever built here was the rake = 0 car -- the drawing, the
    solids, the settled fields, the raster cache and every record of Tiers 62 to
    68 -- so wiring it at 0.10 would have pitched the floor under all of them,
    and `geometry_fingerprint` is what those caches are keyed on."""
    from atlas.cases import racelab as RL

    assert RL.CarParams().rake == 0.0
    assert RL.geometry_fingerprint() == RL.geometry_fingerprint(RL.CarParams(rake=0.0))
    assert RL.geometry_fingerprint() != RL.geometry_fingerprint(RL.CarParams(rake=0.10))


def test_duct_area_scales_the_openings_and_one_is_the_drawn_duct():
    from atlas.cases import car_solids as CS

    assert CS.SOLIDS_DEFAULT["duct_area"] == 1.0, "1.0 must be the drawn duct"
    base = CK.duct_spans(1.0)
    assert CK.duct_spans(0.3) < base < CK.duct_spans(1.5)
    # scaled about the centre, so a span is area x nominal where it fits
    inlet = [r for r in CK.duct_report(0.3) if r["role"] == "inlet"][0]
    assert inlet["achieved_span"] == pytest.approx(0.3 * inlet["nominal_span"], rel=1e-12)
    # an opening that would run off its panel is SHIFTED, keeping its span
    outlet = [r for r in CK.duct_report(1.5) if r["role"] == "outlet"][0]
    assert outlet["achieved"][1] <= 1.0 and outlet["achieved"][0] >= 0.0
    assert outlet["achieved_span"] == pytest.approx(1.5 * outlet["nominal_span"], rel=1e-12)
    with pytest.raises(ValueError):
        CS.scaled_opening(0.2, 0.4, 0.0)


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


def test_the_cooling_probe_moves_no_module_constant():
    """**Tier 69 measured this knob as reaching the loop only by REBINDING
    `cooling_loop.MDOT`, and Tier 71 threaded `LoopSettings` instead.**  The
    diagnosis is kept: that route was global state under a marching thread, and
    the dashboard proved it was not merely inelegant -- the page reported the
    loop responding while the running circuit's return temperature never moved.

    The probe must now reach the loop WITHOUT touching a module constant, so
    this asserts the constants are untouched by a probe that genuinely moves the
    answer -- which is a stronger statement than restoring them afterwards."""
    from atlas.cases import cooling_loop as CL

    before = (CL.MDOT, CL.T_AMB)
    lo = CK.loop_return(mdot=0.05)
    hi = CK.loop_return(mdot=0.30)
    assert (CL.MDOT, CL.T_AMB) == before
    assert CK.moved(lo, hi)
    assert math.isfinite(lo) and math.isfinite(hi)
    # and the settings are what carry it
    st = CK.loop_settings(mdot=0.30, t_amb=310.0)
    assert st.mdot == 0.30 and st.t_amb == 310.0
    assert CL.LoopSettings().mdot == CL.MDOT, "the default must be the constant it replaced"


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
