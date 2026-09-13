"""Tier 53 -- CS-19's arms re-run on the repaired column, and PoC 3's demo.

What this file pins:

1. **The re-run changes exactly three things** -- the machine's sizing, the
   release state, and `enforce` -- and nothing else: the horizon, the settle
   fraction, the five arms, every threshold in `racelab.GATE` and
   `vehicle_march.receiver_balances` are the ones CS-19 used.
2. **Every arm is inside the envelope**, which is the whole point of the
   re-run: CS-19's were all outside.
3. **CS-20's diagnosis of P3 is TESTED, not restated.** That tier said the
   CS-19 failure was Jensen's term over an unsettled window. These arms release
   from a settled field, so if the diagnosis is right the residual falls with
   the window's variance.
4. **Two threads is bitwise the one-thread answer and four is not** -- the
   field is identical at every count, but a domain sum's reduction order is not,
   and the two traces that move are the ones J3's balance is built from (W227).
5. **The demo carries the lead ratio, the envelope stamp and the family table**,
   because a dashboard that hid any of the three would be the artifact this
   project exists to not produce.

The artifact tests need ``out/racelab3/racelab3.json``; rebuild it with
``python scripts/tier53_racelab_rerun.py`` (about ninety minutes -- four tight
arms at 600 macro-steps).
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np                                                    # noqa: E402
import pytest                                                         # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from atlas.cases import ground_effect as GE                           # noqa: E402
from atlas.cases import racelab as RL                                 # noqa: E402
from atlas.cases import racelab_switch as SW                          # noqa: E402

ART = os.path.join(_ROOT, "out", "racelab3", "racelab3.json")
CS19 = os.path.join(_ROOT, "out", "racelab", "racelab.json")
PAGE = os.path.join(_ROOT, "wiki", "concepts", "Atlas 0.1", "common",
                    "poc3-racelab-demo.md")
DEMO = os.path.join(_ROOT, "atlas", "demo_racelab")


def _art():
    if not os.path.isfile(ART):
        pytest.skip("out/racelab3/racelab3.json is absent; rebuild it with "
                    "python scripts/tier53_racelab_rerun.py")
    with open(ART, encoding="utf-8") as fh:
        return json.load(fh)


def _cs19():
    if not os.path.isfile(CS19):
        pytest.skip("out/racelab/racelab.json is absent")
    with open(CS19, encoding="utf-8") as fh:
        return json.load(fh)


def _page():
    if not os.path.isfile(PAGE):
        pytest.skip("the case study page is absent")
    with open(PAGE, encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# 1. the re-run is the same experiment with three things changed
# ---------------------------------------------------------------------------


def test_the_rerun_keeps_CS19_s_horizon_and_arms():
    import tier53_racelab_rerun as T
    old = _cs19()
    assert T.HORIZON == old["march"]["horizon"]
    assert T.SETTLE_FRAC == old["march"]["settle_frac"]
    assert [a for a, _ in T.ARMS] == list(old["march"]["arms"])


def test_the_rerun_uses_the_same_gate_thresholds():
    """Every threshold is `racelab.GATE`'s, which is CS-18's.  A re-run with a
    different bar is not a re-run."""
    a = _art()
    if "arms" not in a:
        pytest.skip("stage arms has not run")
    v = a["arms"]["verdicts"]
    assert (v["P2_J3_receiving_balance"]["tol_with"]
            == RL.GATE["P2_J3_receiving_balance"]["tol_with"])
    assert (v["P3_J1_parametric"]["tol_with"]
            == RL.GATE["P3_J1_parametric"]["tol_with"])
    assert (v["P6_macro_step_cost"]["ceiling_s"]
            == RL.GATE["P6_macro_step_cost"]["ceiling_s"])


def test_every_arm_is_inside_the_envelope():
    """CS-19's were all outside; that is what this tier exists to fix."""
    a = _art()
    if "arms" not in a:
        pytest.skip("stage arms has not run")
    assert a["arms"]["enforce"] is True
    for tag, arm in a["arms"]["arms"].items():
        assert arm["outside_the_envelope_steps"] == 0, tag
    assert a["arms"]["every_arm_inside_the_envelope"] is True
    old = _cs19()
    assert old["envelope"]["every_arm_outside_the_envelope"] is True


def test_the_repeat_floor_is_still_bitwise():
    a = _art()
    if "arms" not in a:
        pytest.skip("stage arms has not run")
    assert a["arms"]["repeat_floor"]["bitwise_in_every_crossing_quantity"]
    assert a["arms"]["repeat_floor"]["bitwise_in_the_field"]


def test_J1_s_null_still_leaves_the_fluid_bitwise_identical():
    """The cleanest control on CS-19's page, and it has to survive the repair."""
    a = _art()
    if "arms" not in a:
        pytest.skip("stage arms has not run")
    g = a["arms"]["G3_J1"]
    assert g["null_arm_fluid_is_bitwise_identical"] is True
    assert g["null_arm_ua_is_exactly_UA_RAD"] is True


def test_J3_s_null_still_fails_at_exactly_one():
    a = _art()
    if "arms" not in a:
        pytest.skip("stage arms has not run")
    g = a["arms"]["G1_J3"]
    assert g["null_arm_work_on_the_fluid"] == 0.0
    assert g["null_arm_shaft_still_claims"] != 0.0
    assert g["residual_without_the_term_null_arm"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. CS-20's diagnosis of P3, tested
# ---------------------------------------------------------------------------


def test_P3_s_residual_tracks_the_window_s_variance():
    """**The test that could have refuted CS-20.**

    That tier said CS-19's P3 failure was Jensen's term -- the clause averages a
    concave map over the settle window, so it measures the variance times the
    curvature.  If that is right the residual must fall with the variance, and
    these arms release from a settled field instead of a freestream.
    """
    a = _art()
    if "arms" not in a:
        pytest.skip("stage arms has not run")
    j = a["arms"]["G3_J1"]["jensen"]
    assert j["pointwise_residual_max"] < 1e-12, (
        "the identity does not hold pointwise, so the clause is not measuring "
        "what CS-20 said it measures")
    #: the averaged residual IS the Jensen term, whatever their common size
    assert j["averaged_over_jensen"] == pytest.approx(1.0, rel=0.25), j


def test_the_settle_window_is_steadier_than_CS19_s():
    a = _art()
    old = _cs19()
    if "arms" not in a:
        pytest.skip("stage arms has not run")
    new_res = a["arms"]["G3_J1"]["tracking_residual"]
    old_res = old["march"]["G3_J1"]["tracking_residual"]
    assert new_res < old_res, (
        "releasing from a settled field did not steady the window, which "
        "would make CS-20's diagnosis wrong", new_res, old_res)


# ---------------------------------------------------------------------------
# 3. W227 -- a domain sum is not bitwise above two threads
# ---------------------------------------------------------------------------


def test_two_threads_is_bitwise_and_four_is_not():
    a = _art()
    if "threads" not in a:
        pytest.skip("stage threads has not run")
    rows = {r["threads"]: r for r in a["threads"]["rows"]}
    for n, r in rows.items():
        assert r["field_bitwise"] is True, (
            "the FIELD moved with the thread count, which would be a much "
            "larger problem than W227", n)
    assert rows[2]["all_traces_bitwise"] is True
    assert rows[4]["all_traces_bitwise"] is False
    assert set(rows[4]["traces_that_differ"]) == {"power_on_the_fluid",
                                                  "power_core"}
    for k, d in rows[4]["traces_that_differ"].items():
        assert d["relative"] < 1e-14, (k, d)
    assert a["threads"]["chosen"] in a["threads"]["bitwise_identical_to_one_thread"]


def test_the_traces_that_move_are_the_ones_J3_s_balance_is_built_from():
    """That is why W227 is a row and not a curiosity."""
    a = _art()
    if "threads" not in a:
        pytest.skip("stage threads has not run")
    rows = {r["threads"]: r for r in a["threads"]["rows"]}
    moved = set(rows[4]["traces_that_differ"])
    assert "power_on_the_fluid" in moved or "power_rotor" in moved
    assert "power_core" in moved


# ---------------------------------------------------------------------------
# 4. the demo
# ---------------------------------------------------------------------------


def test_the_demo_imports_and_declares_its_lead_ratio():
    from atlas.demo_racelab import engine as E
    assert E.LEAD_RATIO == pytest.approx(
        (GE.MACRO_DT / GE.EXCHANGES) / 0.1)
    assert round(1.0 / E.LEAD_RATIO) == 32


def test_the_demo_runs_with_the_envelope_check_off_and_stamps():
    """A dashboard that stops reports nothing; what it must never do is show a
    number from outside an envelope without saying so."""
    from atlas.demo_racelab import engine as E
    cfg = E.RaceConfig()
    assert cfg.enforce is False
    assert cfg.host_inflow is not None, (
        "the demo must march the REPAIRED column or every frame is stamped")
    assert cfg.coupling == "lagged", (
        "tight was never an interactive column: CS-19 measured it at 1.7 s a "
        "macro-step against the 0.5 s ceiling")


def test_the_demo_s_geometry_is_the_model_s_own():
    from atlas.demo_racelab import engine as E
    eng = E.Engine()
    g = eng.geometry()
    t, info = RL.layout()
    assert len(g["windows"]) == t.n_windows
    _objs, flat = RL.car_bodies()
    assert len(g["bodies"]) == len(flat)
    assert g["occupancy_in_a_cut"] == info["banded_force_fraction"]


def test_the_demo_refuses_certified_as_a_march_mode():
    from atlas.demo_racelab import engine as E
    eng = E.Engine()
    n = eng.names[0]
    eng.post({"kind": "mode", "window": n, "mode": "certified"})
    eng._drain()
    assert eng.assignment[n] == SW.Mode.CLASSICAL.value
    assert "certified_note" in eng.notes
    assert "fixed point" in eng.notes["certified_note"]


def test_the_demo_names_every_family_without_a_learned_option():
    from atlas.demo_racelab import server as SV
    assert set(SW.NO_LEARNED_OPTION) == {
        "plane-stress-elasticity-2d", "heat-conduction-2d",
        "incompressible-thermal-transport-1d", "lumped-dc-circuit"}
    assert SV.FRAME_HZ > 0


def test_the_page_says_the_learned_switch_is_asked_for_a_thirty_second():
    """W225: the screen must carry the lead ratio beside the switch."""
    html = open(os.path.join(DEMO, "static", "index.html"),
                encoding="utf-8").read()
    assert "lead" in html.lower()
    assert "leadnote" in html
    assert "OUTSIDE THE MODEL" in html
    readme = open(os.path.join(DEMO, "README.md"), encoding="utf-8").read()
    assert "1/32" in readme


def test_the_demo_does_not_claim_the_learned_expert_pays():
    readme = open(os.path.join(DEMO, "README.md"), encoding="utf-8").read().lower()
    for banned in ("poseidon pays", "the learned expert wins",
                   "faster and more accurate"):
        assert banned not in readme
    assert "is not evidence that a learned expert pays" in readme


def test_the_page_names_what_the_tier_did_not_do():
    assert "What this tier did NOT do, named" in _page()
