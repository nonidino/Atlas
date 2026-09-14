"""Tier 58 -- the drawn car's two faults, and candidate fixes measured on copies.

What this file pins:

1. **The drawing is not touched.**  Every arm is a deep copy of the geometry;
   building one leaves `load_geometry()` and the car's fingerprint unchanged.
2. **A settled field belongs to an arm** -- its car, its box and its windows --
   and a field for any other arm is refused, with the control that the right one
   is accepted.
3. **The extra readings are readings.**  `arm_march` takes the same trajectory as
   Tier 56's `traced_march`, bitwise, and the speed-without-the-outflow-column
   is judged by the model's own FLUID predicate, checked against
   `RaceRollout.validity_report` on both sides of the bound.
4. **The summariser counts what it says**: a spike on the last column is in the
   model's reading and out of the column-free one; the machine's longest inside
   run; None is "never", not zero.
5. **The prediction is judged in code under the ids it was registered with**, and
   nothing is judged at unregistered lengths.
6. **The record** (`out/racelab6/racelab6.json`): the prediction in it is the one
   in the script, character for character; every arm is the arm this script
   builds now; the control is Tier 56's car bitwise; and the four findings --
   the car's own flow never leaves the envelope in any arm, only the outflow
   column does and the wake sets when (W258); the rear box does not carry the
   machine's fall (W256); at the arms' sizing the machine is outside only at
   the start (W259); a drawing change moves the machine (W257).
7. **The page quotes the record's own numbers.**
"""

from __future__ import annotations

import copy
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

ART = os.path.join(_ROOT, "out", "racelab6", "racelab6.json")
T56_FIELD = os.path.join(_ROOT, "out", "racelab5", "cache", "settled.npz")
PAGE = os.path.join(_ROOT, "wiki", "concepts", "Atlas 0.1", "common",
                    "poc3-racelab-car-fixes.md")
DRAWN_FINGERPRINT = ("a1f67a8e279280a2e6b98ca77a01017a"
                     "fed294b4a1e1388fda04af09e5e77a8f")


def _art():
    if not os.path.isfile(ART):
        pytest.skip("out/racelab6/racelab6.json is absent; rebuild it with "
                    "python scripts/tier58_car_fixes.py --out out/racelab6")
    with open(ART, encoding="utf-8") as fh:
        return json.load(fh)


def _page():
    if not os.path.isfile(PAGE):
        pytest.skip("the case study page is absent")
    with open(PAGE, encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture
def T():
    import tier58_car_fixes as mod
    return mod


@pytest.fixture
def isolated_out(tmp_path):
    """Point Tier 54's record I/O at a temporary directory, and put it back."""
    import tier54_traced_car as T54
    old = (T54.OUT, T54.CACHE, T54.NAME)
    T54.configure(str(tmp_path / "racelab_probe"))
    try:
        yield T54
    finally:
        T54.OUT, T54.CACHE, T54.NAME = old


# ---------------------------------------------------------------------------
# 1. the drawing is not touched
# ---------------------------------------------------------------------------


def test_an_arm_is_a_copy_and_the_drawing_is_untouched(T):
    before = copy.deepcopy(RL.load_geometry())
    doc = T.arm_geometry("no_rear_box")
    doc["plates"][0]["x"] = -999.0
    assert RL.load_geometry() == before
    assert RL.geometry_fingerprint() == DRAWN_FINGERPRINT
    kept = {e["id"] for e in doc["plates"]}
    assert not set(T.REAR_BOX) & kept
    assert len(kept) == len(before["plates"]) - len(T.REAR_BOX)
    assert {e["id"] for e in T.arm_geometry("no_endplate")["plates"]} == \
        {e["id"] for e in before["plates"]} - {"RW_ENDPLATE"}


def test_an_arm_that_drops_a_plate_the_car_does_not_have_is_refused(T, monkeypatch):
    monkeypatch.setitem(T.ARMS, "ghost", {"drop": ("NO_SUCH_PLATE",),
                                          "nx": RL.RNX, "layout": "derived",
                                          "role": "a test"})
    with pytest.raises(SystemExit, match="NO_SUCH_PLATE"):
        T.arm_geometry("ghost")


def test_the_layouts_the_arms_march(T):
    t0, _i = RL.layout()
    t, _info, _doc, _flat = T.arm_tiling("drawn")
    assert (t.cols, t.rows, t.nx) == (t0.cols, t0.rows, t0.nx)
    moved, info_m, _d, _f = T.arm_tiling("drawn_moved_layout")
    fix, info_f, _d2, _f2 = T.arm_tiling("no_rear_box")
    plate, _i3, _d3, _f3 = T.arm_tiling("no_endplate")
    assert moved.cols == fix.cols == plate.cols != t0.cols
    assert info_m["device_planes"] == info_f["device_planes"] == [294.0, 406.0]
    lb, info_l, _d4, _f4 = T.arm_tiling("long_box")
    assert lb.nx == 800 and lb.covers()
    assert info_l["device_planes"] == [294.0, 406.0]


def test_the_script_will_not_run_without_an_output_directory(T):
    with pytest.raises(SystemExit):
        T.main(["--stages", "summary"])


def test_an_unknown_arm_or_stage_is_refused(T, tmp_path):
    with pytest.raises(SystemExit, match="unknown arm"):
        T.main(["--out", str(tmp_path / "x"), "--arms", "no_wheels"])
    with pytest.raises(SystemExit, match="unknown stage"):
        T.main(["--out", str(tmp_path / "x"), "--stages", "arms,race"])


# ---------------------------------------------------------------------------
# 2. a settled field belongs to an arm
# ---------------------------------------------------------------------------


def test_a_settled_field_for_another_arm_is_refused(T, isolated_out):
    t, _info, doc, _flat = T.arm_tiling("drawn")
    mine = T.arm_identity("drawn", t, doc, T.N_SPIN)
    u = np.ones((2, 3))
    T.save_arm_field("drawn", mine, u, u)
    uu, _vv = T.load_arm_field("drawn", mine)                 # the control
    assert uu.shape == (2, 3)
    tl, _il, docl, _fl = T.arm_tiling("long_box")
    for other in (T.arm_identity("drawn", tl, docl, T.N_SPIN),       # the box
                  T.arm_identity("drawn", t, T.arm_geometry("no_rear_box"),
                                 T.N_SPIN),                           # the car
                  T.arm_identity("drawn", t, doc, T.N_SPIN - 1)):     # the spin
        with pytest.raises(RuntimeError, match="DIFFERENT arm"):
            T.load_arm_field("drawn", other)
    os.makedirs(isolated_out.CACHE, exist_ok=True)
    np.savez(os.path.join(isolated_out.CACHE, "drawn_settled"), u=u, v=u)
    with pytest.raises(RuntimeError, match="DIFFERENT arm"):
        T.load_arm_field("drawn", mine)


# ---------------------------------------------------------------------------
# 3. the extra readings are readings
# ---------------------------------------------------------------------------


def test_the_fluid_predicate_is_the_model_s_on_both_sides_of_the_bound(T):
    import torch
    from atlas.cases import wing_fsi as W
    r = RL.RaceRollout(join_coupling="lagged", enforce=False)
    h, nu = float(r.solver.h), float(r.nu)
    bound = 8.0 * nu / h
    for speed, breach in ((bound * (1.0 - 1e-9), False),
                          (bound * (1.0 + 1e-9), True)):
        u = torch.full((r.ny, r.nx), speed, dtype=W.TORCH_DTYPE)
        v = torch.zeros((r.ny, r.nx), dtype=W.TORCH_DTYPE)
        rep = r.validity_report(u, v)
        assert rep["FLUID"]["valid"] is (not breach)
        assert T.fluid_breach(speed, h, nu) is breach


def test_arm_march_takes_tier56_s_trajectory_bitwise(T):
    if not os.path.isfile(T56_FIELD):
        pytest.skip("out/racelab5/cache/settled.npz is absent")
    import torch
    import tier54_traced_car as T54
    torch.set_num_threads(T.THREADS)
    d = np.load(T56_FIELD)
    t, _info, doc, _flat = T.arm_tiling("drawn")
    host = 0.5317254889754462
    mine, _s, _m = T.arm_march(d["u"], d["v"], host, 2, t, doc, "test")
    theirs, _s2 = T54.traced_march(d["u"], d["v"], host, 2, label="test")
    for a, b in zip(mine, theirs):
        for k in b:
            assert a[k] == b[k], k
        assert a["u_max_without_the_outflow_column"] <= a["u_max"]


# ---------------------------------------------------------------------------
# 4. the summariser
# ---------------------------------------------------------------------------


def _row(step, u_max, no_col, declined=(), at_x=5, nx=10, u_rotor=0.5):
    return {"step": step, "u_max": u_max, "at_cell": [at_x, 3],
            "where": "the outflow column" if at_x == nx - 1 else "open flow",
            "declined": list(declined), "u_rotor": u_rotor,
            "u_max_without_the_outflow_column": no_col,
            "u_max_without_the_last_8_columns": no_col,
            "outflow_column_abs_v_max": 0.1, "load": 1.0, "drag": 0.5,
            "induction": 0.1, "current": 0.1, "omega": 1.0}


def test_a_spike_on_the_last_column_is_in_one_reading_and_not_the_other(T):
    nx, h, nu = 10, 1.0 / 64.0, 4.0e-3
    bound = 8.0 * nu / h
    rows = [_row(0, 1.0, 1.0, nx=nx),
            _row(1, bound + 0.5, 1.9, ("FLUID",), at_x=nx - 1, nx=nx),
            _row(2, bound + 0.6, 1.9, ("FLUID", "ROTOR"), at_x=nx - 1, nx=nx),
            _row(3, 1.0, 1.0, ("MGU",), nx=nx), _row(4, 1.0, 1.0, nx=nx),
            _row(5, 1.0, 1.0, nx=nx), _row(6, 1.0, 1.0, nx=nx)]
    meta = {"nx": nx, "h": h, "nu": nu, "host_inflow": 0.5, "wall_s": 1.0,
            "s_per_macro_step": 1.0 / 7}
    s = T.summarise(rows, meta, None)
    assert s["FLUID"] == {"first": 1, "last": 2, "declined_steps": 2}
    assert s["fluid"]["u_max"]["steps_over_the_bound"] == 2
    assert s["fluid"]["without_the_outflow_column"]["steps_over_the_bound"] == 0
    assert s["fluid"]["without_the_outflow_column"]["first_over_the_bound"] is None
    assert s["fluid"]["fastest_cell_on_the_outflow_column_while_FLUID_declines"] == 1.0
    assert s["machine"]["declined_steps"] == 2                 # steps 2 and 3
    assert s["machine"]["longest_inside_run"] == 3             # steps 4 to 6
    assert s["machine"]["longest_inside_run_starts_at"] == 4
    compact = T.compact(rows)
    assert compact["declined"] == ["", "F", "FR", "M", "", "", ""]


def test_none_means_never_not_zero(T):
    assert T._between(None, 20, None)
    assert T._between(123, 20, None)
    assert not T._between(10, 20, None)
    assert T._within(None, None, 5)
    assert not T._within(None, 3, 5)
    assert T._runs([]) == (0, None)
    assert T._runs([3, 4, 5, 9, 10]) == (3, 3)


# ---------------------------------------------------------------------------
# 5. the prediction
# ---------------------------------------------------------------------------


def test_every_registered_prediction_is_judged_under_its_own_id(T):
    ids = [pid for pid, _text in T.PREDICTION]
    assert len(ids) == len(set(ids))
    assert list(T.judge({}, registered=True)) == ids
    judged = T.judge({}, registered=False)
    assert list(judged) == ids
    assert all(v["verdict"] is None for v in judged.values())


def test_nothing_is_judged_without_its_measurement(T):
    assert all(v["verdict"] is None for k, v in T.judge({}, True).items())


# ---------------------------------------------------------------------------
# 6. the record
# ---------------------------------------------------------------------------


def test_the_prediction_in_the_record_is_the_one_in_the_script(T):
    """Registered before any stage ran, and not edited since: the texts in the
    source are character-for-character the texts in the record."""
    a = _art()
    assert a["prediction_recorded_at"] <= min(r["started_at"] for r in a["runs"])
    assert [(p["id"], p["text"]) for p in a["prediction"]] == list(T.PREDICTION)
    assert a["read_before_this_run"] == list(T.READ_BEFORE_THIS_RUN)
    assert a["lengths_are_the_registered_ones"] is True
    assert a["lengths"] == {"spin": T.N_SPIN, "horizon": T.HORIZON}
    assert not a.get("failures")


def test_the_record_belongs_to_the_arms_this_script_builds(T):
    a = _art()
    assert list(a["arms"]) == list(T.ARM_ORDER)
    for name in T.ARM_ORDER:
        t, _info, doc, _flat = T.arm_tiling(name)
        assert a["arms"][name]["identity"] == T.arm_identity(name, t, doc, T.N_SPIN)


def test_the_verdicts_in_the_record_are_the_code_s(T):
    a = _art()
    recomputed = T.judge(a, registered=True)
    recorded = a["summary"]["predictions"]
    assert list(recomputed) == list(recorded)
    for pid, v in recomputed.items():
        assert v["verdict"] == recorded[pid]["verdict"], pid
    assert {k: v["verdict"] for k, v in recorded.items()} == {
        "C1": True, "F1": True, "F2": False, "F3": True, "M1": True,
        "M2": False, "M3": False, "M4": True, "W1": True, "A1": True,
        "K1": False}


def test_C1_the_control_is_tier56_s_car_bitwise():
    a = _art()
    d = a["arms"]["drawn"]
    assert d["spinup"]["reproduces_tier56_settled_bitwise"] is True
    assert d["probe1"]["control"]["equal_steps"] == 600
    assert d["probe1"]["control"]["bitwise"] is True
    assert d["probe2"]["control"]["equal_steps"] == 60
    assert d["probe2"]["control"]["bitwise"] is True
    assert d["probe2"]["control"]["sized_for_the_same_value"] is True
    assert a["layouts"]["drawn"]["equals_racelab_layout"] is True
    assert a["layouts"]["drawn"]["geometry_fingerprint"] == DRAWN_FINGERPRINT


def test_W1_the_window_predicts_every_verdict_it_was_checked_against(T):
    a = _art()
    b = a["band"]
    lo, hi = b["machine_inside"]
    assert 0.97313 < lo < 0.97314 and 1.14075 < hi < 1.14076
    assert b["similarity_spread_across_hosts"] < 1e-12
    for k in ("release_sizing", "arms_sizing"):
        assert b["check_against_tier56"][k]["agreement"] == 1.0
    assert b["check_against_tier56"]["arms_sizing"]["rotor_declined_steps"] == \
        list(range(9))
    assert b["check_against_tier56"]["arms_sizing"]["mgu_declined_steps"] == []
    for name in T.ARM_ORDER:
        for probe in ("probe1", "probe2"):
            s = (a["arms"][name].get(probe) or {}).get("summary")
            if s:
                assert s["window_check"]["steps_where_they_disagree"] == 0


def test_W258_the_car_s_own_flow_never_leaves_the_envelope_in_any_arm(T):
    """Only the outflow column breaks the bound -- in every arm, at both sizings,
    over every macro-step -- and the control is that the model's own reading,
    which includes that column, does break it."""
    a = _art()
    n = 0
    for name in T.ARM_ORDER:
        for probe in ("probe1", "probe2"):
            s = (a["arms"][name].get(probe) or {}).get("summary")
            if not s:
                continue
            f = s["fluid"]
            assert f["without_the_outflow_column"]["steps_over_the_bound"] == 0
            assert f["without_the_last_8_columns"]["steps_over_the_bound"] == 0
            assert s["FLUID"]["declined_steps"] > 0                  # the control
            assert f["fastest_cell_on_the_outflow_column_while_FLUID_declines"] == 1.0
            n += 1
    assert n == 10


def test_W258_the_wake_sets_when_the_boundary_breaks():
    a = _art()
    first = {n: a["arms"][n]["probe1"]["summary"]["FLUID"]["first"]
             for n in a["arms"]}
    assert first == {"drawn": 56, "drawn_moved_layout": 56, "no_rear_box": 394,
                     "long_box": 265, "no_endplate": 294}


def test_W256_the_rear_box_does_not_carry_the_machine_s_fall():
    """Tier 56's ablation said it did, from the intact car's settled field."""
    a = _art()
    fall = {n: a["arms"][n]["probe1"]["summary"]["u_rotor"]["fall_fraction"]
            for n in a["arms"]}
    assert all(0.14 < f < 0.19 for f in fall.values())
    assert fall["no_rear_box"] > fall["drawn_moved_layout"]
    first = {n: a["arms"][n]["probe1"]["summary"]["machine"]["first"]
             for n in a["arms"]}
    assert all(15 <= f <= 25 for f in first.values())


def test_W259_at_the_arms_sizing_the_machine_is_outside_only_at_the_start(T):
    a = _art()
    for name in ("drawn_moved_layout", "no_rear_box", "long_box", "no_endplate"):
        s = a["arms"][name]["probe2"]["summary"]
        assert s["steps"] == 600
        assert s["machine"]["first"] == 0
        assert s["machine"]["last"] < 12
        assert s["machine"]["declined_at_or_after_step_%d" % T.AFTER_THE_START] == 0
        assert s["MGU"]["declined_steps"] == 0
        assert 0.9731 < s["u_rotor"]["ratio_to_the_sizing_min"] < 0.9752
    d = a["arms"]["drawn"]["probe2"]["summary"]
    assert (d["machine"]["first"], d["machine"]["last"]) == (0, 8)


def test_W257_a_drawing_change_moves_the_machine():
    a = _art()
    L = a["layouts"]
    assert L["drawn"]["layout"]["device_planes"] == [280.0, 392.0]
    for n in ("drawn_moved_layout", "no_rear_box", "no_endplate", "long_box"):
        assert L[n]["layout"]["device_planes"] == [294.0, 406.0]
    u0 = a["arms"]["drawn"]["spinup"]["u_rotor_at_the_release_state"]
    u1 = a["arms"]["drawn_moved_layout"]["spinup"]["u_rotor_at_the_release_state"]
    assert u1 / u0 - 1.0 < -0.06
    cc = L["_cross_costs"]["cars"]
    # the control: the drawn car under its own layout is the DP's own number
    assert cc["drawn"]["drawn's (planes 280, 392)"] == pytest.approx(
        L["drawn"]["layout"]["banded_force_fraction"], rel=1e-12)
    assert cc["no_rear_box"]["the fixes' (planes 294, 406)"] == pytest.approx(
        L["no_rear_box"]["layout"]["banded_force_fraction"], rel=1e-12)
    assert cc["drawn"]["relative_gap"] > 0 > cc["no_rear_box"]["relative_gap"]


# ---------------------------------------------------------------------------
# 7. the page
# ---------------------------------------------------------------------------


def test_the_page_names_what_the_tier_did_not_do():
    assert "What this tier did NOT do, named" in _page()


def test_the_page_quotes_the_record_s_own_numbers():
    a = _art()
    page = _page()
    for n in a["arms"]:
        s = a["arms"][n]["probe1"]["summary"]
        assert "$%d$" % s["FLUID"]["first"] in page, n
        assert "%.1f" % (100 * s["u_rotor"]["fall_fraction"]) in page, n
    lo, hi = a["band"]["machine_inside"]
    assert "%.4f" % lo in page and "%.4f" % hi in page
    assert "%.4f" % a["arms"]["no_rear_box"]["spinup"][
        "u_rotor_at_the_release_state"] in page
