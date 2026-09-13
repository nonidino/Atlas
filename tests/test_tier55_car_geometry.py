"""The car's shape is DATA, and the checks that guard it discriminate.

A geometry that only exists as literals inside a Python function can only be
changed by someone willing to read that function.  These pin the two claims
that make it editable instead: the JSON is the source of truth, and the
checker fails the things it says it fails.
"""

from __future__ import annotations

import copy
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
_S = os.path.join(HERE, "scripts")
if _S not in sys.path:
    sys.path.insert(0, _S)

import pytest                                                       # noqa: E402

from atlas.cases import racelab as RL                               # noqa: E402


@pytest.fixture(scope="module")
def doc():
    return RL.load_geometry()


def test_the_geometry_file_exists_and_is_what_the_car_is_built_from(doc):
    assert os.path.isfile(RL.GEOMETRY_JSON)
    assert doc["plates"] and doc["wheels"]
    objs, flat = RL.car_bodies()
    ids = {b.body_id for b in flat}
    for e in doc["plates"]:
        assert e["id"] in ids, "%s is in the file and not in the car" % e["id"]


def test_editing_the_file_moves_the_car(doc):
    """The point of the whole exercise: the JSON is the source of truth."""
    d = copy.deepcopy(doc)
    target = next(e for e in d["plates"] if "alpha_from" not in e
                  and "y_plus_ride" not in e and "y_plus_duct" not in e)
    target["x"] += 7.0
    _objs, flat = RL.car_bodies(geometry=d)
    moved = next(b for b in flat if b.body_id == target["id"])
    base = next(b for b in RL.car_bodies()[1] if b.body_id == target["id"])
    assert abs((moved.x_le - base.x_le) - 7.0) < 1e-9


def test_a_knob_still_owns_the_angles_it_owned(doc):
    """`alpha_from` entries must follow CarParams, not the stored number."""
    d = copy.deepcopy(doc)
    diff = next(e for e in d["plates"] if e["id"] == "DIFF")
    assert diff.get("alpha_from") == "diffuser_deg"
    diff["alpha"] = -99.0                       # a number the build must ignore
    p = RL.CarParams(diffuser_deg=7.5)
    _objs, flat = RL.car_bodies(p, geometry=d)
    b = next(x for x in flat if x.body_id == "DIFF")
    assert abs(b.alpha_deg - 7.5) < 1e-9, "the knob did not win"


def test_the_duct_stays_exactly_DEVICE_CELLS_tall_wherever_the_band_moves(doc):
    _objs, flat = RL.car_bodies()
    up = next(b for b in flat if b.body_id == "DUCT_UP")
    lo = next(b for b in flat if b.body_id == "DUCT_LO")
    assert abs((up.y_le - lo.y_le) - RL.DEVICE_CELLS) < 1e-9
    assert abs(lo.y_le - RL.DUCT_Y0) < 1e-9


def test_every_plate_builds_exactly_as_the_file_draws_it(doc):
    """The editor and the march must show the SAME car.

    Three plates take their incidence from a `CarParams` knob rather than from
    the file.  When those defaults disagreed with the drawing the model built a
    flap at 30 degrees that had been drawn at 21.8, a diffuser at 11 that had
    been drawn flat, and a rear flap at 32 that had been drawn at 43.6 -- and
    nothing said so, because the editor drew the stored number and the march
    used the knob.  This is the test that would have caught it.
    """
    _objs, flat = RL.car_bodies()
    by = {b.body_id: b for b in flat}
    p = RL.CarParams()
    for e in doc["plates"]:
        b = by[e["id"]]
        if "y_plus_ride" in e:
            y = e["y_plus_ride"] + p.ride_height
        elif "y_plus_duct" in e:
            y = RL.DUCT_Y0 + e["y_plus_duct"]
        else:
            y = e["y"]
        assert abs(b.x_le - e["x"]) < 1e-6, e["id"]
        assert abs(b.y_le - y) < 1e-6, e["id"]
        assert abs(b.chord - e["chord"]) < 1e-6, e["id"]
        assert abs(b.alpha_deg - e["alpha"]) < 0.05, (
            "%s is DRAWN at %.2f and BUILDS at %.2f"
            % (e["id"], e["alpha"], b.alpha_deg))


def test_the_wing_the_structure_couples_to_is_CS12_s_chord():
    """`WING_BODY_ID` is what the front wing's structure hangs off, and CS-12
    built that structure with 32 stations at 0.25 m.  Redrawing the car must
    not quietly shorten it."""
    from atlas.cases import ground_effect as GE
    _objs, flat = RL.car_bodies()
    fw = next(b for b in flat if b.body_id == RL.WING_BODY_ID)
    assert fw.n_station == 32
    assert abs(fw.chord * RL.DX - GE.CHORD) < 1e-9


# ---------------------------------------------------------------------------
# the checker
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def CE():
    import car_editor
    return car_editor


def test_the_saved_geometry_passes_its_own_checks(CE, doc):
    rep = CE.check(copy.deepcopy(doc))
    assert rep["errors"] == [], rep["errors"]
    assert any("window layout resolves" in m for m in rep["ok"])


def test_the_checker_catches_a_body_through_the_tiling_seam(CE, doc):
    d = copy.deepcopy(doc)
    # the tallest free-standing body plate, whichever it is called today
    tall = max((e for e in d["plates"]
                if e["group"] == "body" and "y_plus_ride" not in e
                and "y_plus_duct" not in e), key=lambda e: e["y"])
    tall["y"] = 130.0
    rep = CE.check(d)
    assert any("tiling seam" in m for m in rep["errors"]), rep


def test_the_checker_catches_a_plate_inside_a_wheel(CE, doc):
    d = copy.deepcopy(doc)
    # the longest floor plate, whichever it is called today
    flr = max((e for e in d["plates"] if e["group"] == "floor"),
              key=lambda e: e["chord"])
    flr["x"] = 120.0
    rep = CE.check(d)
    assert any("inside a wheel" in m for m in rep["errors"]), rep


def test_the_checker_catches_a_tyre_swallowing_the_turbine_cut(CE, doc):
    d = copy.deepcopy(doc)
    next(w for w in d["wheels"] if w["id"] == "WHEEL_R")["x"] = 380.0
    rep = CE.check(d)
    assert any("turbine cut" in m for m in rep["errors"]), rep


def test_the_checker_would_pass_a_broken_car_if_it_did_not_discriminate(CE, doc):
    """The control for the three above: the SAME car, unbroken, passes.

    Without this the three tests above would also pass on a checker that
    failed everything.
    """
    assert CE.check(copy.deepcopy(doc))["errors"] == []


def test_the_editor_ships_its_page_and_writes_exactly_one_file(CE):
    assert os.path.isfile(CE.PAGE), "the editor has no page"
    src = open(os.path.join(_S, "car_editor.py"), encoding="utf-8").read()
    assert src.count("json.dump(") == 1, "the editor writes more than one file"
    html = open(CE.PAGE, encoding="utf-8").read()
    for must in ('id="save"', 'id="check"', 'id="undo"', "backdrop"):
        assert must in html, "the editor page has no %s" % must
