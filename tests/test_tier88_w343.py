"""Tier 88, continued -- W343: the d-g seam made two-way.

The predictions for the two-way run were registered before it marched
(`w336_episode_residuals.PREDICTIONS_W343`). The first half of this file shows
each can hold and can fail on synthetic records; the second pins the records
(out/w336 and out/w321 now, Tier 88's in out/w336_t88 and out/w321_t88).
"""
from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pytest

from test_tier87_episode_residuals import _write_records

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "w336")
T88 = os.path.join(ROOT, "out", "w336_t88")
RECORD = os.path.join(ROOT, "out", "w321", "episode.npz")
T88_RECORD = os.path.join(ROOT, "out", "w321_t88", "episode.npz")


def _driver():
    path = os.path.join(ROOT, "scripts", "w336_episode_residuals.py")
    spec = importlib.util.spec_from_file_location("w336_driver_w343", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# 1. the registered predictions can hold and can fail
# ===========================================================================


def _write_w343(root, good):
    g = good
    d, t = os.path.join(root, "out"), os.path.join(root, "t88")
    os.makedirs(d)
    os.makedirs(t)
    _write_records(d, good)

    def load(where, n):
        with open(os.path.join(where, n), encoding="utf-8") as fh:
            return json.load(fh)

    def put(where, n, obj):
        with open(os.path.join(where, n), "w", encoding="utf-8") as fh:
            json.dump(obj, fh)

    au = load(d, "audit.json")
    for s in au["steps"]:
        s["seam_dg_mass_rel"] = -0.01 if g else -0.02
        s["chamber_p_mean"] = 7.84e6 if g else 7.9e6
        s["throat_mdot_e_over_declared"] = 1.00005 if g else 1.001
        s["rigid"][4] = 1000.0 + 10.0 * s["t"]
    put(d, "audit.json", au)
    d2 = load(d, "dt_2p5ms.json")
    for s in d2["steps"]:
        s["seam_dg_mass_rel"] = -0.009 if g else -0.002
    put(d, "dt_2p5ms.json", d2)
    rec = dict(np.load(os.path.join(d, "episode.npz")))
    rec["rigid"][:, 4] = 1000.0 + 10.0 * np.round(0.005 * np.arange(rec["rigid"].shape[0]), 6)
    rec["loads_body"] = np.tile([0.0, 0.0, 838.0, 0.0], (rec["rigid"].shape[0], 1))
    np.savez(os.path.join(d, "episode.npz"), **rec)
    old = dict(rec)
    old["rigid"] = rec["rigid"].copy()
    if not g:
        old["rigid"][-1, 4] = old["rigid"][0, 4] + 1.2 * (rec["rigid"][-1, 4] - rec["rigid"][0, 4])
    old["loads_body"] = np.tile([0.0, 0.0, 838.5 if g else 900.0, 0.0], (rec["rigid"].shape[0], 1))
    np.savez(os.path.join(t, "episode.npz"), **old)
    steps = [dict(t=s["t"], rigid=list(s["rigid"]), chamber_p_mean=7.84e6,
                  injector_mdot_over_declared=1.0, seam_be_mass_rel=3e-4, seam_dg_mass_rel=-0.021,
                  engine_heat_lost={"jmin": 600.0, "jmax": 600.0}) for s in au["steps"]]
    put(t, "audit.json", dict(steps=steps))
    ctl = json.loads(json.dumps(steps[:2]))
    if not g:
        ctl[1]["rigid"][1] *= 1.0 + 1e-9
    put(d, "control_w343.json", dict(steps=ctl))
    gates = {"G%d" % i: dict(value=0, passed=True) for i in range(17)}
    if not g:
        gates["G3"]["passed"] = False
    put(root, "gates.json", dict(gates=gates))
    return d, t


@pytest.mark.parametrize("good", [True, False])
def test_every_w343_prediction_can_hold_and_can_fail(tmp_path, good):
    W = _driver()
    d, t = _write_w343(str(tmp_path), good)
    res = W.evaluate_w343(d, t88_dir=t, record=os.path.join(d, "episode.npz"),
                          t88_record=os.path.join(t, "episode.npz"),
                          gates=os.path.join(str(tmp_path), "gates.json"))
    assert sorted(res) == sorted(W.PREDICTIONS_W343)
    held = {k: res[k]["held"] for k in res}
    assert all(v is good for v in held.values()), held


def test_w343_is_revertible_and_restorable():
    W = _driver()
    M = W._mods()
    gen = M["gen"]
    try:
        assert gen.CoupledEpisode.D_G_TWO_WAY is True
        assert W.revert(M, "W343") == ["W343"]
        assert gen.CoupledEpisode.D_G_TWO_WAY is False
    finally:
        W.restore_fixes()
    assert gen.CoupledEpisode.D_G_TWO_WAY is True


# ===========================================================================
# 2. the records: the two-way run against Tier 88's one-way run
# ===========================================================================


def _rec(name, where=OUT):
    path = os.path.join(where, name)
    if not os.path.exists(path):
        pytest.skip("%s not present" % os.path.relpath(path, ROOT))
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _dg(run, lo=0.03, hi=0.06):
    return float(np.mean([abs(s["seam_dg_mass_rel"]) for s in run["steps"]
                          if lo < s["t"] <= hi + 1e-9]))


def _dg_study(where):
    """d|g's mean |mass created| over 30-60 ms at 10, 5 and 2.5 ms."""
    au = _rec("audit.json", where)
    au = dict(au, steps=[s for s in au["steps"] if s["t"] <= 0.06 + 1e-9])
    return [_dg(_rec("dt_10ms.json", where)), _dg(au), _dg(_rec("dt_2p5ms.json", where))]


def test_the_registered_predictions_read_as_recorded():
    """Five of six held. S4 failed on its own wording: its throat clause asked
    for 1e-4 of the DECLARED flow, and Tier 88's throat already ran at 1.00089
    of it late in the run -- the engine did not move (next test)."""
    _rec("audit.json")
    if not os.path.exists(RECORD):
        pytest.skip("out/w321/episode.npz not present")
    res = _driver().evaluate_w343()
    held = {k: res[k]["held"] for k in res}
    assert held == {"S1": True, "S2": True, "S3": True, "S4": False, "S5": True, "S6": True}, held
    assert res["S4"]["got"]["chamber_p"][0] == pytest.approx(res["S4"]["got"]["chamber_p"][1], rel=1e-12)
    assert 8e-4 < res["S4"]["got"]["throat"] - 1.0 < 1e-3


def test_the_engine_does_not_see_the_seam():
    new, old = _rec("audit.json"), _rec("audit.json", T88)
    assert len(new["steps"]) == len(old["steps"]) == 29
    for a, b in zip(new["steps"], old["steps"]):
        for k in ("injector_mdot_over_declared", "throat_mdot_b_over_declared",
                  "throat_mdot_e_over_declared", "exit_mdot_over_declared",
                  "chamber_p_mean", "thrust_exit_face_mean"):
            assert a[k] == pytest.approx(b[k], rel=1e-12), k
    # the throat S4 compared with the declared flow: the same in both runs
    late = [np.mean([s["throat_mdot_e_over_declared"] for s in r["steps"][-10:]]) for r in (new, old)]
    assert late[0] == pytest.approx(late[1], rel=1e-12)
    assert 1.0008 < late[1] < 1.001


def test_dg_is_a_floor_at_about_half_its_one_way_size():
    """The two-way seam halves d|g and leaves a floor: 1.13% at 5 ms and at
    2.5 ms, where the one-way seam was 2.13% and 2.08%."""
    ten, five, half = _dg_study(OUT)
    assert 0.0105 < five < 0.0120 and 0.0105 < half < 0.0120, (five, half)
    assert abs(five / half - 1.0) < 0.02
    assert ten > 3.0 * five  # the 10 ms step still carries a lag on top
    # the old diagnosis, kept as the control: one way, a floor at 2.1%
    ten0, five0, half0 = _dg_study(T88)
    assert 0.020 < five0 < 0.022 and 0.020 < half0 < 0.022, (five0, half0)
    assert five < 0.6 * five0 and half < 0.6 * half0


def test_the_two_way_seam_rings_every_other_step_and_decays():
    """d and g are wired from each other's start-of-step states, so the loop
    closes over two macro steps. The late series alternates step by step and
    its swing shrinks; the one-way seam's late series is flat."""
    S = [s["seam_dg_mass_rel"] for s in _rec("audit.json")["steps"]]
    late = S[9:]  # from 50 ms, past the plume's arrival
    d = np.diff(late)
    assert all(d[i] * d[i + 1] < 0 for i in range(len(d) - 1)), np.round(100 * np.asarray(late), 2)
    assert abs(d[-1]) < 0.5 * abs(d[0])
    assert np.ptp(late[-10:]) > 2e-3
    one_way = [s["seam_dg_mass_rel"] for s in _rec("audit.json", T88)["steps"]][-10:]
    assert np.ptp(one_way) < 5e-4


def test_the_flight_does_not_move():
    for p in (RECORD, T88_RECORD):
        if not os.path.exists(p):
            pytest.skip("%s not present" % os.path.relpath(p, ROOT))
    a, b = np.load(RECORD), np.load(T88_RECORD)
    gain = [float(z["rigid"][-1][4] - z["rigid"][0][4]) for z in (a, b)]
    assert gain[0] == pytest.approx(gain[1], rel=1e-5)
    drag = [float(z["loads_body"][-1][2]) for z in (a, b)]
    assert drag[0] == pytest.approx(drag[1], rel=5e-3)
    assert drag[0] != drag[1]  # the seam does reach the body, at 0.15%


def test_the_control_is_tier88s_run():
    """S6: the one-way seam put back on today's code reproduces Tier 88's
    first two audited steps exactly."""
    co, old = _rec("control_w343.json"), _rec("audit.json", T88)
    assert co["reverted"] == ["W343"]
    assert len(co["steps"]) == 2
    for c, p in zip(co["steps"], old["steps"]):
        assert c["rigid"] == p["rigid"]
        for k in ("injector_mdot_over_declared", "seam_be_mass_rel", "seam_dg_mass_rel"):
            assert c[k] == p[k], k
        assert c["engine_heat_lost"] == p["engine_heat_lost"]


def test_the_engine_alone_is_tier88s_so_its_finer_grids_carry_over():
    """d-g cannot reach the engine run alone: coarsen 4 comes out equal to
    Tier 88's in every reading on a different box, so the coarsen-2 and
    coarsen-1 engine records in out/w336 are Tier 88's, carried over."""
    new, old = _rec("grid_engine_c4.json"), _rec("grid_engine_c4.json", T88)
    skip = {"machine", "build_repo", "wall_seconds", "steps"}
    assert {k: v for k, v in new.items() if k not in skip} == {k: v for k, v in old.items() if k not in skip}
    for a, b in zip(new["steps"], old["steps"]):
        assert {k: v for k, v in a.items() if k != "wall_s"} == {k: v for k, v in b.items() if k != "wall_s"}
    assert new["build_repo"] != old["build_repo"]
    for c in ("c2", "c1"):
        name = "grid_engine_%s.json" % c
        assert _rec(name) == _rec(name, T88)
