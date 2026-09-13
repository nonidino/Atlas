"""Tier 56 -- the car the user drew, re-sized, and the arms NOT run.

What this file pins:

1. **The record describes the car that is built.**  `out/racelab5/racelab5.json`
   carries the fingerprint of the car it measured, and the settled field the
   demo releases from carries the same one.  Edit the car and this fails --
   which is the point: every marched number in `out/racelab4` went on
   describing a car that no longer existed, and nothing said so.
2. **The gate refused, and it refused for the reason the probe predicted.**  No
   horizon admits the arms: the verify was declined, and `arms` refused to run
   rather than committing ninety minutes to a sizing already known to fail.
3. **W244: the sizing has no admissible answer.**  The duct flow falls far more
   over the horizon than on either earlier car, and the machine sized for the
   minimum is outside its clamp at the release state -- the verify says so.
4. **W245: the fluid leaves its envelope at the OUTFLOW BOUNDARY, not at a
   body.**  The per-step trace reads the model's own envelope and the fastest
   cell's location; attributing it to the nearest body 155 cells away was this
   tier's own first reading, and is refused by construction now.
5. **The script cannot overwrite a committed record by default** -- ``--out``
   is required -- and **refuses a settled field for a different car**, with a
   control that it accepts the right one.
6. **The demo is sized and released for this car**, read against the record.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_S = os.path.join(_ROOT, "scripts")
if _S not in sys.path:
    sys.path.insert(0, _S)

import numpy as np                                                    # noqa: E402
import pytest                                                         # noqa: E402

from atlas.cases import racelab as RL                                 # noqa: E402

ART = os.path.join(_ROOT, "out", "racelab5", "racelab5.json")
FIELD = os.path.join(_ROOT, "out", "racelab5", "cache", "settled.npz")
PAGE = os.path.join(_ROOT, "wiki", "concepts", "Atlas 0.1", "common",
                    "poc3-racelab-drawn-car.md")


def _art():
    if not os.path.isfile(ART):
        pytest.skip("out/racelab5/racelab5.json is absent; rebuild it with "
                    "python scripts/tier54_traced_car.py --out out/racelab5")
    with open(ART, encoding="utf-8") as fh:
        return json.load(fh)


def _page():
    if not os.path.isfile(PAGE):
        pytest.skip("the case study page is absent")
    with open(PAGE, encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# 1. a record belongs to a car
# ---------------------------------------------------------------------------


def test_the_fingerprint_repeats_and_ignores_a_caption():
    import copy
    doc = copy.deepcopy(RL.load_geometry())
    doc["plates"][0]["label"] = "a different caption"
    assert RL.geometry_fingerprint() == RL.geometry_fingerprint()
    assert RL.geometry_fingerprint(geometry=doc) == RL.geometry_fingerprint()


def test_the_fingerprint_sees_a_billionth_of_a_cell_and_a_knob():
    """The controls for the test above: a fingerprint that ignored everything
    would pass it."""
    import copy
    doc = copy.deepcopy(RL.load_geometry())
    e = next(p for p in doc["plates"] if "alpha_from" not in p
             and "y_plus_ride" not in p and "y_plus_duct" not in p)
    e["x"] += 1e-9
    assert RL.geometry_fingerprint(geometry=doc) != RL.geometry_fingerprint()
    assert (RL.geometry_fingerprint(RL.CarParams(diffuser_deg=7.5))
            != RL.geometry_fingerprint())


def test_the_record_describes_the_car_that_is_built():
    """Fails when the car is edited, and says how to re-measure it."""
    a = _art()
    assert a["geometry"]["fingerprint"] == RL.geometry_fingerprint(), (
        "out/racelab5 was measured on a DIFFERENT car from the one "
        "car_geometry.json builds now; re-run "
        "python scripts/tier54_traced_car.py --out out/racelab5")
    assert a["spinup"]["geometry_fingerprint"] == a["geometry"]["fingerprint"]


def test_the_settled_field_carries_the_same_car():
    if not os.path.isfile(FIELD):
        pytest.skip("out/racelab5/cache/settled.npz is absent; --stages spinup")
    with np.load(FIELD) as d:
        assert str(d["geometry"]) == RL.geometry_fingerprint()


def test_the_prediction_was_registered_before_the_first_stage():
    import tier54_traced_car as T
    a = _art()
    assert a["prediction"] == list(T.PREDICTION_DRAWN)
    first_run = a["runs"][0]
    assert first_run["stages"][0] == "spinup"
    assert a["prediction_recorded_at"] <= first_run["started_at"]


# ---------------------------------------------------------------------------
# 2-4. the gate refused, and why
# ---------------------------------------------------------------------------


def test_the_arms_were_refused_rather_than_run_on_a_declined_sizing():
    a = _art()
    assert a["verify"]["admitted"] is False
    assert "arms" not in a, "the arms ran on a sizing the verify declined"
    fails = [f for f in a.get("failures", []) if f["stage"] == "arms"]
    assert fails and "verify did not admit" in fails[0]["why"]


def test_W244_the_minimum_sizing_is_outside_the_clamp_at_release():
    """The machine sized for the horizon's minimum is declined at macro-step 0,
    by the ROTOR, at the UPPER clamp: the release inflow is too far above the
    minimum for one constant to span."""
    a = _art()
    s, v = a["size"], a["verify"]
    fall = s["fall_fraction"]
    assert fall > 0.10, fall
    t54 = os.path.join(_ROOT, "out", "racelab4", "racelab4.json")
    if os.path.isfile(t54):
        with open(t54, encoding="utf-8") as fh:
            assert fall > 5 * json.load(fh)["size"]["fall_fraction"]
    assert v["host_inflow"] == s["U_DUCT_to_size_for"] == s["u_rotor_min"]
    assert "macro-step 0" in v["why_declined"] and "ROTOR" in v["why_declined"]
    assert "0.4" in v["why_declined"], "not the upper clamp"


def test_the_horizon_is_chosen_from_the_probe_at_the_arms_sizing():
    """Probe 1 is sized for the release state and its first decline is the
    machine, which the arms do not run; so probe 2 marches the arms' sizing and
    the horizon comes from it."""
    a = _art()
    s = a["size"]
    p1 = s["probe_at_the_release_sizing"]
    assert set(p1["first_declined"]) - {"FLUID"}
    assert "probe_at_the_arms_sizing" in s
    assert s["horizon_chosen_from"].startswith("probe 2")


def test_W245_the_fluid_leaves_its_envelope_on_the_outflow_column():
    a = _art()
    t = a["trace"]
    assert t["first_FLUID_decline"] is not None
    rows = t["rows"]
    fluid = [r for r in rows if "FLUID" in r["declined"]]
    assert fluid, "the trace saw no fluid decline"
    on_outflow = sum(1 for r in fluid if r["at_cell"][0] == RL.RNX - 1)
    assert on_outflow / len(fluid) > 0.95, on_outflow
    # the model's own cell Reynolds number, not a re-derivation
    assert all(r["cell_reynolds"] > 8.0 for r in fluid)
    # and it is NOT attributed to a body
    reread = a.get("ablate", {}).get("trace_summary_reread", {}).get("is", {})
    if reread:
        where = reread["where_the_fastest_cell_is_while_FLUID_declines"]
        assert set(where) == {"the outflow column"}, where


def test_a_far_body_is_not_where_the_fastest_cell_is():
    import tier54_traced_car as T
    _objs, flat = RL.car_bodies()
    assert T._where(flat, RL.RNX - 1, 118, RL.RNX, RL.RNY) == "the outflow column"
    assert T._where(flat, 600, 200, RL.RNX, RL.RNY) == "open flow"
    # the control: a cell ON a body is attributed to it
    b = next(x for x in flat if x.body_id == "FLOOR")
    ix, iy = int(b.x_le + 20), int(round(b.y_le))
    assert T._where(flat, ix, iy, RL.RNX, RL.RNY) == "FLOOR"


def test_the_ablation_control_agrees_bitwise_with_the_trace():
    a = _art()
    ab = a.get("ablate")
    if not ab:
        pytest.skip("stage ablate has not run")
    assert ab["control_agrees_bitwise_with_stage_trace"] is True
    assert "none" in ab["arms"]


# ---------------------------------------------------------------------------
# 5. the script's guards
# ---------------------------------------------------------------------------


def test_the_script_will_not_run_without_an_output_directory():
    import tier54_traced_car as T
    with pytest.raises(SystemExit):
        T.main(["--stages", "compare"])


def test_the_script_refuses_a_settled_field_for_another_car(tmp_path):
    import tier54_traced_car as T
    old = (T.OUT, T.CACHE, T.NAME)
    try:
        T.configure(str(tmp_path / "racelab_probe"))
        os.makedirs(T.CACHE)
        u = np.ones((2, 3))
        np.savez(os.path.join(T.CACHE, "settled"), u=u, v=u,
                 geometry=np.array("0" * 64))
        with pytest.raises(RuntimeError, match="DIFFERENT car"):
            T.settled()
        np.savez(os.path.join(T.CACHE, "settled"), u=u, v=u)
        with pytest.raises(RuntimeError, match="DIFFERENT car"):
            T.settled()
        # the control: its own car is accepted
        np.savez(os.path.join(T.CACHE, "settled"), u=u, v=u,
                 geometry=np.array(RL.geometry_fingerprint()))
        uu, _vv = T.settled()
        assert uu.shape == (2, 3)
    finally:
        T.OUT, T.CACHE, T.NAME = old


def test_a_probe_that_declines_before_the_floor_has_no_horizon():
    """Tier 54's chooser returned its floor there, and Tier 56's first run
    reported "horizon 60" from a probe that had declined at macro-step 0."""
    import tier54_traced_car as T
    base = {"outside_the_envelope_steps": 5}
    assert T._horizon_from({**base, "first_outside_macro_step": 0}) is None
    assert T._horizon_from({**base, "first_outside_macro_step":
                            T.HORIZON_FLOOR + T.HORIZON_MARGIN - 1}) is None
    assert (T._horizon_from({**base, "first_outside_macro_step": 554})
            == 554 - T.HORIZON_MARGIN)
    assert T._horizon_from({"outside_the_envelope_steps": 0,
                            "first_outside_macro_step": None}) == T.HORIZON


def test_a_probe_keeps_its_whole_trajectory():
    a = _art()
    p = a["size"]["probe_at_the_release_sizing"]
    if "trace" not in p:
        pytest.skip("this record's probes predate whole-trajectory recording")
    assert len(p["trace"]["u_max"]) == p.get("steps", 600) or \
        len(p["trace"]["u_max"]) == 600


# ---------------------------------------------------------------------------
# 6. the demo, for this car
# ---------------------------------------------------------------------------


def test_the_demo_is_sized_for_this_car_s_release_state():
    from atlas.demo_racelab import engine as E
    a = _art()
    assert E.U_DUCT == a["spinup"]["u_rotor_at_the_release_state"]
    assert E.RaceConfig().host_inflow == E.U_DUCT


def test_the_demo_releases_this_car_from_its_own_field():
    from atlas.demo_racelab import engine as E
    if not os.path.isfile(FIELD):
        pytest.skip("out/racelab5/cache/settled.npz is absent")
    _u, _v, note, mine = E.find_release(_ROOT, RL.geometry_fingerprint())
    assert mine is True, note
    assert "out/racelab5" in note


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------


def test_the_page_names_what_the_tier_did_not_do():
    assert "What this tier did NOT do, named" in _page()


def test_the_page_quotes_the_record_s_own_numbers():
    a = _art()
    page = _page()
    t = a["trace"]
    assert str(t["first_FLUID_decline"]) in page
    assert "%.2f" % t["u_max_max"] in page
    assert "%.1f" % (100 * a["size"]["fall_fraction"]) in page
    assert str(a["size"]["probe_at_the_release_sizing"]
               ["first_outside_macro_step"]) in page
