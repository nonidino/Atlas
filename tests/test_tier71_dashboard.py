"""Tier 71 -- the dashboard, and the coolant knob that finally reaches the march.

These pin the things that made W291 worth fixing rather than labelling: a knob
that reports a response the march does not have, a capability record that goes
on claiming the machine that used to run, a refactor that changes what it was
supposed to preserve -- and the annotation trap that let the page load, answer
its API, and never receive a single frame.
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
from atlas.cases import cooling_loop as CL                              # noqa: E402
from atlas.cases import integration_union as IU                         # noqa: E402

RECORD = os.path.join(HERE, "out", "racelab20", "racelab20.json")
PAGE = os.path.join(HERE, "wiki", "concepts", "Atlas 0.1", "common",
                    "poc3-racelab-dashboard.md")
STATIC = os.path.join(HERE, "atlas", "demo_racelab", "static", "bodyfitted.html")


# ---------------------------------------------------------------------------
# the operating point
# ---------------------------------------------------------------------------


def test_the_defaults_are_the_constants_they_replaced():
    """The refactor is additive or it is a behaviour change wearing a refactor's
    clothes.  Every leg that existed must run at exactly what it ran at."""
    d = CL.LoopSettings()
    assert d.mdot == CL.MDOT
    assert d.t_amb == CL.T_AMB
    assert d.ua_rad == CL.UA_RAD
    assert d.w_pump == CL.W_PUMP
    assert d.cp == CL.CP_COOLANT
    assert d.mass_flux == CL.MASS_FLUX


def test_the_knob_reaches_the_solve_without_moving_a_module_constant():
    """**This is W291.**  The knob used to reach the loop only by rebinding
    `cooling_loop.MDOT` -- global state under a marching thread, which the
    dashboard proved was not merely inelegant."""
    before = (CL.MDOT, CL.T_AMB)
    cold = CL.LoopSolve(block=IU.MountedBlock(q_machine=0.0),
                        settings=CL.LoopSettings(t_amb=273.0)).solve().t_return
    hot = CL.LoopSolve(block=IU.MountedBlock(q_machine=0.0),
                       settings=CL.LoopSettings(t_amb=318.0)).solve().t_return
    assert (CL.MDOT, CL.T_AMB) == before, "a solve moved a module constant"
    assert CK.moved(cold, hot)
    assert hot > cold


def test_a_full_reach_report_leaves_every_constant_alone():
    before = (CL.MDOT, CL.T_AMB, CL.UA_RAD, CL.W_PUMP, CL.CP_COOLANT)
    CK.reach_report()
    assert (CL.MDOT, CL.T_AMB, CL.UA_RAD, CL.W_PUMP, CL.CP_COOLANT) == before


def test_the_weight_hash_moves_with_the_knob():
    """A record that does not move when the physics does is a record claiming
    the machine that used to run."""
    a, b = CL.LoopSettings(mdot=0.05), CL.LoopSettings(mdot=0.30)
    assert a.tag() != b.tag()
    assert CL.LoopSettings(t_amb=273.0).tag() != CL.LoopSettings(t_amb=318.0).tag()
    assert CL.LoopSettings().tag() == CL.LoopSettings(mdot=CL.MDOT, t_amb=CL.T_AMB).tag()


def test_every_leg_carries_its_operating_point():
    legs = CL.make_legs(4, CL.LoopSettings(mdot=0.22))
    assert legs, "no legs"
    for name, leg in legs.items():
        assert leg.s.mdot == 0.22, name
    # and the default build is the module's
    for name, leg in CL.make_legs(4).items():
        assert leg.s.mdot == CL.MDOT, name


def test_the_sweep_study_carries_one_too():
    """It has its own `_legs`, which is how the first version of this refactor
    broke: the study was patched as though it were `LoopSolve`."""
    st = CL.SweepStudy(settings=CL.LoopSettings(mdot=0.22))
    for name, leg in st._legs().items():
        assert leg.s.mdot == 0.22, name


def test_the_knob_state_hands_over_an_operating_point():
    s = CK.KnobState()
    s.set("ambient_t", 310.0)
    s.set("coolant_mdot", 0.2)
    st = s.loop_settings()
    assert st.t_amb == 310.0 and st.mdot == 0.2


# ---------------------------------------------------------------------------
# the trap that made the page look alive and deliver nothing
# ---------------------------------------------------------------------------


def test_fastapi_is_imported_at_module_level_in_the_server():
    """**The 403 regression guard.**

    `bodyfitted_server` carries `from __future__ import annotations`, so
    `sock: WebSocket` is the STRING "WebSocket" and FastAPI resolves it with
    `get_type_hints` against the MODULE's globals.  Imported inside
    `create_app`, the name does not resolve there, the parameter is not
    recognised as the socket, and every upgrade to /ws is refused with 403 --
    while `/` returns the page and `/api/meta` answers 200.  The page loads,
    looks alive, and never receives a frame.

    Moving the import back inside the function would restore that silently, so
    this asserts where it lives rather than trusting anyone to remember.
    """
    pytest.importorskip("fastapi")
    import atlas.demo_racelab.bodyfitted_server as S

    assert "WebSocket" in vars(S), "WebSocket must resolve in the module's globals"
    assert "FastAPI" in vars(S)
    src = open(S.__file__, encoding="utf-8").read()
    head = src.split("def create_app", 1)[0]
    assert "from fastapi import" in head, "the fastapi import must be above create_app"


def test_the_page_exists_and_names_what_it_will_not_show():
    text = open(STATIC, encoding="utf-8").read()
    assert "MARCHING THE CAR BEFORE THESE CHANGES" in text
    assert "can never run a learned expert" in text
    assert "Not drawn, and why" in text
    assert "Not offered" in text
    # the commit button, and the fact that a slider does not re-cut
    assert 'id="commit"' in text
    assert "not re-cut until you commit" in text


# ---------------------------------------------------------------------------
# the march itself
# ---------------------------------------------------------------------------


def test_car_union_carries_and_can_change_its_operating_point():
    """A cooling knob must move a RUNNING march, which is the whole point."""
    import inspect

    from atlas.cases import car_union as CU

    sig = inspect.signature(CU.CarUnion.__init__)
    assert "loop_settings" in sig.parameters
    assert hasattr(CU.CarUnion, "set_loop_settings")


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def record():
    if not os.path.isfile(RECORD):
        pytest.skip("out/racelab20/racelab20.json is not here (carried upstream only)")
    with open(RECORD, encoding="utf-8") as fh:
        return json.load(fh)


def test_the_record_s_predictions_are_this_script_s_and_every_one_is_judged(record):
    import tier71_dashboard as T
    assert record["prediction"] == [dict(p) for p in T.PREDICTION]
    assert record["read_before_this_run"] == list(T.READ_BEFORE_THIS_RUN)
    assert record["verdicts_missing"] == []
    assert T.judge(record) == record["verdicts"]


def test_this_tier_added_nothing_to_the_moved_set(record):
    c = record["control"]
    if not c.get("tested"):
        pytest.skip(c.get("why", "control not run"))
    assert c["added_by_this_tier"] == [], c["added_by_this_tier"]


def test_a_commit_lands_on_a_different_car_and_an_honest_release(record):
    """A release is honest when it either names THIS car's file or refuses to
    march at all -- never when it silently reuses another car's field."""
    m = record["commit"]
    assert m["changed"] is True
    assert m["release_is_honest"] is True
    assert m["pending_after_commit"] == []
    # the commit really did change the composite, which is why the old car's
    # state cannot be reused: 377,267 unknowns against 377,176
    assert m["unknowns_changed_by"] != 0
    if m["needs_spin_up"]:
        assert m["declined_to_march"] is True, "a car with no spun-up field must not march"
        assert m["prefix_refused"], "the prefix must have been refused, with a reason"


def test_the_page_says_what_the_record_measured(record):
    if not os.path.isfile(PAGE):
        pytest.skip("the page is not here")
    text = open(PAGE, encoding="utf-8").read()
    for pid in record["verdicts"]:
        assert ("| %s |" % pid) in text, pid
