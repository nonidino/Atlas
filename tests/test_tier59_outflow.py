"""Tier 59 -- W258 repaired at the outlet, and W259's start.

What this file pins:

1. **The outlet condition is what `racelab.OUTFLOW` declares.**  ``"convective"``
   moves the last column and nothing else, toward its interior neighbour, by the
   clipped Courant number, only where the flow leaves, and out of place;
   ``"pinned"`` returns its input untouched; anything else is refused.
2. **The pinned column is the old column, bitwise** -- against Tier 56's recorded
   trace rows, not against itself.
3. **Every expert behind the outlet gets the same condition**: `MixedRollout`
   with every window classical is `RaceRollout`, bitwise, on the repaired column.
4. **A record names its column.**  Tier 56's driver marches the column its
   record was measured on, and a settled field for another column is refused.
5. **The record** (`out/racelab7/racelab7.json`), once it exists.
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

ART = os.path.join(_ROOT, "out", "racelab7", "racelab7.json")
T56_FIELD = os.path.join(_ROOT, "out", "racelab5", "cache", "settled.npz")
T56_JSON = os.path.join(_ROOT, "out", "racelab5", "racelab5.json")


def _art():
    if not os.path.isfile(ART):
        pytest.skip("out/racelab7/racelab7.json is absent; rebuild it with "
                    "python scripts/tier59_outflow_and_start.py --out out/racelab7")
    with open(ART, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def rollouts():
    """One rollout per outlet condition, on the drawn car's layout."""
    return {m: RL.RaceRollout(join_coupling="lagged", enforce=False, outflow=m)
            for m in RL.OUTFLOW_MODES}


def _field(r, seed=0):
    import torch
    from atlas.cases import wing_fsi as W
    g = np.random.default_rng(seed)
    u = 1.0 + 0.3 * g.standard_normal((r.ny, r.nx))
    v = 0.3 * g.standard_normal((r.ny, r.nx))
    return (torch.as_tensor(u, dtype=W.TORCH_DTYPE),
            torch.as_tensor(v, dtype=W.TORCH_DTYPE))


# ---------------------------------------------------------------------------
# 1. the outlet condition
# ---------------------------------------------------------------------------


def test_the_default_is_the_repaired_column_and_the_modes_are_declared():
    assert RL.OUTFLOW == "convective"
    assert RL.OUTFLOW_MODES == ("convective", "pinned")
    assert RL.RaceRollout().outflow == "convective"
    with pytest.raises(ValueError, match="outflow"):
        RL.RaceRollout(outflow="open")


def test_pinned_returns_its_input_untouched(rollouts):
    r = rollouts["pinned"]
    u, v = _field(r)
    uu, vv = r.relax_outflow(u, v)
    assert uu is u and vv is v


def test_convective_moves_the_last_column_only_and_by_the_declared_rule(rollouts):
    import torch
    r = rollouts["convective"]
    u, v = _field(r, seed=1)
    u0, v0 = u.clone(), v.clone()
    uu, vv = r.relax_outflow(u, v)
    # out of place: the inputs are not written
    assert torch.equal(u, u0) and torch.equal(v, v0)
    # everything but the last column is untouched, bitwise
    assert torch.equal(uu[:, :-1], u[:, :-1]) and torch.equal(vv[:, :-1], v[:, :-1])
    # the last column, by hand
    c = r.dt_ex / float(r.solver.h)
    ul, un = u[:, -1].numpy(), u[:, -2].numpy()
    vl, vn = v[:, -1].numpy(), v[:, -2].numpy()
    cour = np.clip(ul * c, 0.0, 1.0)
    out = ul > 0.0
    np.testing.assert_array_equal(uu[:, -1].numpy(),
                                  np.where(out, ul - cour * (ul - un), ul))
    np.testing.assert_array_equal(vv[:, -1].numpy(),
                                  np.where(out, vl - cour * (vl - vn), vl))
    # the control: the rule did move something
    assert not torch.equal(uu[:, -1], u[:, -1])


def test_inflow_at_the_outlet_is_left_alone_and_the_courant_number_clips(rollouts):
    import torch
    r = rollouts["convective"]
    u, v = _field(r, seed=2)
    u[:5, -1] = -0.5                        # flow ENTERING through the outlet
    u[5:10, -1] = 1.0e6                     # a Courant number far above one
    assert float(u[5, -1]) * r.dt_ex / float(r.solver.h) > 1.0
    uu, vv = r.relax_outflow(u, v)
    assert torch.equal(uu[:5, -1], u[:5, -1]) and torch.equal(vv[:5, -1], v[:5, -1])
    # C clipped to 1 takes the neighbour's value -- to rounding, because
    # a - (a - b) is not b in floating point when a is a million
    assert torch.allclose(uu[5:10, -1], u[5:10, -2], rtol=0.0, atol=1e-9)
    assert torch.allclose(vv[5:10, -1], v[5:10, -2], rtol=0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# 2-3. the controls
# ---------------------------------------------------------------------------


def test_the_pinned_column_is_tier56_s_column_bitwise():
    if not (os.path.isfile(T56_FIELD) and os.path.isfile(T56_JSON)):
        pytest.skip("Tier 56's settled field or record is absent")
    import torch
    import tier54_traced_car as T54
    torch.set_num_threads(2)
    d = np.load(T56_FIELD)
    with open(T56_JSON, encoding="utf-8") as fh:
        t56 = json.load(fh)
    host = float(t56["trace"]["host_inflow"])
    rows, _ = T54.traced_march(d["u"], d["v"], host, 2, label="t59",
                               outflow="pinned")
    for i in range(2):
        for k in ("u_max", "u_rotor", "induction", "current", "load", "drag"):
            assert rows[i][k] == t56["trace"]["rows"][i][k], (i, k)
    # the control: the repaired column does NOT reproduce them
    rep, _ = T54.traced_march(d["u"], d["v"], host, 2, label="t59",
                              outflow="convective")
    assert any(rep[i]["u_max"] != t56["trace"]["rows"][i]["u_max"]
               or rep[i]["u_rotor"] != t56["trace"]["rows"][i]["u_rotor"]
               for i in range(2))


def test_every_expert_behind_the_outlet_gets_the_same_condition():
    if not os.path.isfile(T56_FIELD):
        pytest.skip("Tier 56's settled field is absent")
    import torch
    from atlas.cases import racelab_switch as SW
    from atlas.cases import wing_fsi as W
    torch.set_num_threads(2)
    d = np.load(T56_FIELD)
    t, _i = RL.layout()

    def run(cls):
        r = cls(tiling=t, join_coupling="lagged", enforce=False,
                host_inflow=0.53, outflow="convective")
        u = torch.as_tensor(d["u"], dtype=W.TORCH_DTYPE)
        v = torch.as_tensor(d["v"], dtype=W.TORCH_DTYPE)
        r.refresh(u, 0, which=("J1", "J3"))
        for s in range(2):
            with torch.no_grad():
                u, v, _l, _d = r.macro_step(u, v, s)
        return u, v

    ua, va = run(RL.RaceRollout)
    ub, vb = run(SW.MixedRollout)
    assert torch.equal(ua, ub) and torch.equal(va, vb)


# ---------------------------------------------------------------------------
# 4. a record names its column
# ---------------------------------------------------------------------------


def test_tier56_s_driver_marches_the_column_each_record_was_measured_on(tmp_path):
    import tier54_traced_car as T54
    import tier58_car_fixes as T58
    assert T54.RUNS["racelab4"]["outflow"] == "pinned"
    assert T54.RUNS["racelab5"]["outflow"] == "pinned"
    assert T58.OUTFLOW == "pinned"
    old = (T54.OUT, T54.CACHE, T54.NAME, T54.OUTFLOW)
    try:
        T54.configure(str(tmp_path / "racelab5"))
        assert T54.OUTFLOW == "pinned"
        T54.configure(str(tmp_path / "a_record_nobody_registered"))
        assert T54.OUTFLOW == RL.OUTFLOW
    finally:
        T54.OUT, T54.CACHE, T54.NAME, T54.OUTFLOW = old


def test_the_settle_stage_leaves_an_old_record_exactly_as_it_was():
    import tier54_traced_car as T54
    old = (T54.OUT, T54.CACHE, T54.NAME, T54.OUTFLOW)
    try:
        T54.configure(os.path.join(_ROOT, "out", "racelab5"))
        out = T54.stage_settle({})
        assert out["marched"] is False and "declares no settle" in out["why"]
        assert T54.RUNS["racelab8"]["settle"] == 120
        assert T54.RUNS["racelab8"]["outflow"] == "convective"
    finally:
        T54.OUT, T54.CACHE, T54.NAME, T54.OUTFLOW = old


def test_the_release_inflow_is_the_settled_state_s_once_it_is_settled():
    import tier54_traced_car as T54
    res = {"spinup": {"u_rotor_at_the_release_state": 0.53}}
    assert T54.release_u_rotor(res) == 0.53
    res["settle"] = {"marched": False}
    assert T54.release_u_rotor(res) == 0.53
    res["settle"] = {"marched": True, "u_rotor_at_the_settled_state": 0.457}
    assert T54.release_u_rotor(res) == 0.457


def test_every_old_driver_names_the_pinned_column():
    """Each record before Tier 59 was measured on the pinned column, and its
    driver says so at every march -- read from the source, because running
    them costs hours."""
    for name in ("tier51_racelab_graph.py", "tier52_racelab_switch.py",
                 "tier53_racelab_rerun.py"):
        with open(os.path.join(_S, name), encoding="utf-8") as fh:
            src = fh.read()
        assert 'OUTFLOW = "pinned"' in src, name
        calls = src.count("RL.march(") + src.count("RL.settled_field(")
        assert calls and src.count("outflow=OUTFLOW") >= calls, name


# ---------------------------------------------------------------------------
# 5. the record
# ---------------------------------------------------------------------------


def test_the_predictions_in_the_record_are_the_ones_in_the_script():
    import tier59_outflow_and_start as T
    a = _art()
    assert [(p["id"], p["text"]) for p in a["prediction"]] == list(T.PREDICTION)
    assert ([(p["id"], p["text"]) for p in a["prediction_start"]]
            == list(T.PREDICTION_START))
    runs = sorted(r["started_at"] for r in a["runs"])
    assert a["prediction_recorded_at"] <= runs[0]
    # stage start's block was registered by the run that first asked for it,
    # after stage repair had finished
    start_run = next(r for r in a["runs"] if "start" in r["stages"])
    assert a["prediction_start_recorded_at"] >= start_run["started_at"]
    assert a["read_before_this_run"] == list(T.READ_BEFORE_THIS_RUN)
    assert not a.get("failures")


def test_the_verdicts_in_the_record_are_the_code_s():
    import tier59_outflow_and_start as T
    a = _art()
    again = T.judge(a)
    rec = a["summary"]["predictions"]
    assert {k: v["verdict"] for k, v in again.items()} == \
        {k: v["verdict"] for k, v in rec.items()}
    assert {k: v["verdict"] for k, v in rec.items()} == {
        "R1": True, "R2": True, "R3": False, "R4": True, "R5": False,
        "R6": True, "T1": True, "T2": True, "T3": True}


def test_R1_the_pinned_column_through_this_file_is_tier58_s():
    a = _art()
    c = a["arms"]["drawn_pinned"]
    assert c["spinup"]["reproduces_tier56_settled_bitwise"] is True
    assert c["probe1"]["control"]["equal_steps"] == 60
    assert c["probe1"]["control"]["bitwise"] is True


def test_W258_repaired_the_outlet_no_longer_breaks_the_bound():
    """At both sizings, on every macro-step -- and the control is the pinned
    column in the same 800-cell box, which still breaks at 265."""
    a = _art()
    d = a["arms"]["drawn"]
    for probe in ("probe1", "probe2"):
        s = d[probe]["summary"]
        assert s["steps"] == 600
        assert s["FLUID"]["declined_steps"] == 0
        assert s["fluid"]["u_max"]["steps_over_the_bound"] == 0
    assert a["arms"]["long_repaired"]["probe1"]["summary"]["FLUID"][
        "declined_steps"] == 0
    assert a["arms"]["long_pinned"]["probe1"]["summary"]["FLUID"]["first"] == 265


def test_W259_repaired_the_settled_state_admits_the_whole_horizon():
    a = _art()
    st = a["start"]
    assert st["settle"]["steps"] == 120
    assert st["settle"]["host_inflow"] == a["arms"]["drawn"]["probe2"][
        "summary"]["host_inflow"]
    for probe in ("probe1", "probe2"):
        s = st[probe]["summary"]
        assert s["steps"] == 600
        assert s["machine"]["declined_steps"] == 0
        assert s["FLUID"]["declined_steps"] == 0
    lo, hi = a["band"]["machine_inside"]
    s2 = st["probe2"]["summary"]
    assert lo < s2["u_rotor"]["ratio_to_the_sizing_min"]
    assert s2["u_rotor"]["ratio_to_the_sizing_max"] < hi
    # the control: from the unsettled state, the same rule declined at the start
    un = a["arms"]["drawn"]["probe2"]["summary"]["machine"]
    assert (un["first"], un["declined_steps"]) == (0, 3)


# ---------------------------------------------------------------------------
# 6. the drawn car's arms, at last (out/racelab8, Tier 56's driver)
# ---------------------------------------------------------------------------

ART8 = os.path.join(_ROOT, "out", "racelab8", "racelab8.json")


def _art8():
    if not os.path.isfile(ART8):
        pytest.skip("out/racelab8/racelab8.json is absent; rebuild it with "
                    "python scripts/tier54_traced_car.py --out out/racelab8 "
                    "--stages spinup,size,settle,verify,arms,slow,gate,compare "
                    "--compare-with out/racelab4/racelab4.json")
    with open(ART8, encoding="utf-8") as fh:
        return json.load(fh)


def test_racelab8_s_prediction_was_registered_before_its_first_stage():
    import tier54_traced_car as T54
    a = _art8()
    assert a["prediction"] == list(T54.PREDICTION_REPAIRED)
    first = min(a["runs"], key=lambda r: r["started_at"])
    assert first["stages"][0] == "spinup"
    assert a["prediction_recorded_at"] <= first["started_at"]
    assert a["geometry"]["outflow"] == "convective"
    assert a["geometry"]["fingerprint"] == RL.geometry_fingerprint()
    assert not a.get("failures")


def test_racelab8_reproduces_tier59_s_settled_state_and_is_admitted():
    """The driver's own spin-up, sizing and settle land on Tier 59's numbers,
    and the first admissible horizon on the drawn car is CS-19's 600."""
    a, b = _art8(), _art()
    assert a["spinup"]["u_rotor_at_the_release_state"] == \
        b["arms"]["drawn"]["spinup"]["u_rotor_at_the_release_state"]
    assert a["size_before_the_settle"]["U_DUCT_to_size_for"] == \
        b["start"]["settle"]["host_inflow"]
    assert a["size_before_the_settle"]["safe_horizon"] is None     # the control
    assert a["settle"]["u_rotor_at_the_settled_state"] == \
        b["start"]["settle"]["u_rotor_at_the_end"]
    assert a["size"]["U_DUCT_to_size_for"] == \
        b["start"]["probe2"]["summary"]["host_inflow"]
    assert a["size"]["safe_horizon"] == 600
    v = a["verify"]
    assert v["admitted"] is True and v["steps"] == 600
    assert v["host_inflow"] == a["size"]["U_DUCT_to_size_for"]
    assert v["current_is_positive_throughout"] is True


def test_racelab8_every_arm_is_inside_every_envelope_for_the_whole_horizon():
    a = _art8()
    arms = a["arms"]
    assert arms["horizon"] == 600 and arms["enforce"] is True
    assert set(arms["arms"]) == {"referent", "repeat", "all_lagged", "null_J3",
                                 "null_J1"}
    for tag, arm in arms["arms"].items():
        assert arm["outside_the_envelope_steps"] == 0, tag
    assert a["gate"]["every_arm_inside_the_envelope"] is True


def test_racelab8_the_gate_passes_with_every_threshold_inherited():
    a = _art8()
    g = a["gate"]["verdicts"]
    for clause in ("P1_compile", "P2_J3_receiving_balance", "P3_J1_parametric",
                   "P4_J2_receiving_balance", "P5_repeat_floor",
                   "P7_body_force_conserves"):
        assert g[clause] == "pass", clause
    assert g["P6_macro_step_cost"]["lagged"] == "pass"
    assert g["P6_macro_step_cost"]["tight"] == "fail"
    assert g["P6_macro_step_cost"]["ceiling_s"] == RL.GATE[
        "P6_macro_step_cost"]["ceiling_s"]
    arms = a["arms"]
    assert arms["G1_J3"]["residual_with_the_term"] <= RL.GATE[
        "P2_J3_receiving_balance"]["tol_with"]
    assert arms["G1_J3"]["residual_without_the_term_null_arm"] >= RL.GATE[
        "P2_J3_receiving_balance"]["tol_without"]
    assert arms["G3_J1"]["tracking_residual"] <= RL.GATE[
        "P3_J1_parametric"]["tol_with"]
    assert arms["G3_J1"]["null_arm_fluid_is_bitwise_identical"] is True
    assert arms["repeat_floor"] == {"bitwise_in_every_crossing_quantity": True,
                                    "bitwise_in_the_field": True}


def test_the_demo_is_sized_for_racelab8_s_verified_sizing_and_released_from_it():
    from atlas.demo_racelab import engine as E
    a = _art8()
    assert E.U_DUCT == a["verify"]["host_inflow"]
    assert E.RaceConfig().host_inflow == E.U_DUCT
    assert E.RELEASE_TIERS[0] == "racelab8"
    if not os.path.isfile(os.path.join(_ROOT, "out", "racelab8", "cache",
                                       "settled.npz")):
        pytest.skip("out/racelab8/cache/settled.npz is absent")
    _u, _v, note, mine = E.find_release(_ROOT, RL.geometry_fingerprint())
    assert mine is True and "out/racelab8" in note, note
