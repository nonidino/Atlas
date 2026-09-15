"""Tier 63 -- the duct opened by the solids rule.

These pin what cutting an opening into a welded shell can get silently wrong:
an opening that leaves a square edge, one that bridges back when the shell is
filleted, an opening named on a plate the car does not have or on an aerofoil,
a rule change that quietly rewrites the previous tier's sealed car, and a
record the page misquotes.
"""

from __future__ import annotations

import copy
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (HERE, os.path.join(HERE, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                  # noqa: E402
import pytest                                                       # noqa: E402

from atlas.cases import car_solids as CS                            # noqa: E402
from atlas.cases import racelab as RL                               # noqa: E402

RECORD = os.path.join(HERE, "out", "racelab12", "racelab12.json")
PAGE = os.path.join(HERE, "wiki", "concepts", "Atlas 0.1", "common", "poc3-racelab-duct-openings.md")


def _doc(**solids):
    doc = copy.deepcopy(RL.load_geometry())
    doc["solids"] = dict(solids)
    return doc


def _plate(pid):
    objs, _flat = RL.car_bodies()
    return next(o.body for o in objs if isinstance(o, RL.PlateBody) and o.body.body_id == pid)


def test_the_rule_opens_an_inlet_and_an_outlet_by_default():
    ops = {o["plate"]: o for o in CS.SOLIDS_DEFAULT["openings"]}
    assert ops["CHASSIS"]["role"] == "inlet" and ops["POD_UP"]["role"] == "outlet"
    assert (ops["CHASSIS"]["from"], ops["CHASSIS"]["to"]) == (0.25, 0.75)
    assert (ops["POD_UP"]["from"], ops["POD_UP"]["to"]) == (0.80, 1.00)


def test_an_opening_removes_exactly_its_stretch_and_leaves_round_caps():
    from shapely.geometry import Point
    b = _plate("CHASSIS")
    rule = CS.solids_rule(_doc())
    whole = CS._plate_polygon(b, rule, openings=False)
    cut = CS._plate_polygon(b, rule)
    assert cut.geom_type == "MultiPolygon" and len(cut.geoms) == 2
    t = rule["panel_thickness_cells"]
    # the opening's middle is open, and the two pieces end in caps of the panel's half-thickness
    L = b.chord
    mid = Point(b.x_le + 0.5 * L * np.cos(b.alpha), b.y_le + 0.5 * L * np.sin(b.alpha))
    assert not cut.contains(mid) and whole.contains(mid)
    # a stretch of half the chord goes, and the two new ends each get back a
    # half-disc cap -- a square cut would have lost the whole rectangle
    lost = whole.area - cut.area
    assert lost == pytest.approx(0.5 * L * t - np.pi * (0.5 * t) ** 2, rel=0.01)


def test_the_openings_split_the_body_shell_into_three_solids():
    solids, rec = CS.car_solids(geometry=_doc(fillet_by_member={"FW_MAIN": 6.0, "COCKPIT": 4.5, "TAIL_DECK": 6.0}),
                                check=False)
    shells = [s for s in solids if s.kind == "shell"]
    holding = {m: [s.name for s in shells if m in s.members] for m in ("NOSE", "COCKPIT", "TAIL_DECK", "CHASSIS")}
    assert holding["NOSE"] != holding["COCKPIT"] != holding["TAIL_DECK"]
    assert len(holding["CHASSIS"]) == 2          # its two halves, either side of the inlet
    assert [o["plate"] for o in rec["openings"]] == ["CHASSIS", "POD_UP"]


def test_an_opening_on_a_plate_the_car_does_not_have_or_on_an_aerofoil_is_refused():
    with pytest.raises(ValueError):
        CS.car_solids(geometry=_doc(openings=[{"plate": "NO_SUCH", "from": 0.1, "to": 0.2}]), check=False)
    with pytest.raises(ValueError):
        CS.car_solids(geometry=_doc(openings=[{"plate": "FW_MAIN", "from": 0.1, "to": 0.2}]), check=False)
    with pytest.raises(ValueError):
        CS.car_solids(geometry=_doc(openings=[{"plate": "CHASSIS", "from": 0.6, "to": 0.2}]), check=False)


def test_tier_62_s_script_still_builds_the_sealed_pod_it_recorded():
    import tier62_car_solids as T62
    assert T62.TIER62_SOLIDS == {"openings": []}
    assert T62.GEOMETRY is T62._tier62_geometry
    doc = T62._tier62_geometry()
    assert CS.solids_rule(doc)["openings"] == []


# ---------------------------------------------------------------------------
# the record, and the page that quotes it
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def record():
    if not os.path.isfile(RECORD):
        pytest.skip("out/racelab12/racelab12.json is not here (it is carried upstream only)")
    with open(RECORD, encoding="utf-8") as fh:
        return json.load(fh)


def test_the_record_s_predictions_are_this_script_s_and_every_one_is_judged(record):
    import tier63_duct_openings as T
    assert record["prediction"] == [dict(p) for p in T.PREDICTION]
    assert record["read_before_this_run"] == list(T.READ_BEFORE_THIS_RUN)
    assert record["verdicts_missing"] == []
    assert T.judge(record) == record["verdicts"]


def test_the_record_ran_the_rule_with_the_openings(record):
    assert [o["plate"] for o in record["car"]["openings"]] == ["CHASSIS", "POD_UP"]
    assert record["compare"]["same_drawing"] is True


def test_the_page_says_what_the_record_measured(record):
    if not os.path.isfile(PAGE):
        pytest.skip("the page is not here")
    text = open(PAGE, encoding="utf-8").read()
    for pid in record["verdicts"]:
        assert ("| %s |" % pid) in text, pid
    u = record["march"]["summary"]["turbine_u_mean_window"]
    assert ("%+.3f" % u) in text or ("%.3f" % u) in text
