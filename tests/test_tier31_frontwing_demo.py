"""Tier 31 -- the PoC 2 demo's five beats, and the prose that describes them.

`tests/test_tier30_front_wing.py` grades the assembly.  This file grades the
**demo over it**, and the split matters: nothing here may re-measure physics, and
everything here is about whether the screen says something true.

Four kinds of test, in the order they would catch a regression:

  1. **the beats work** -- each of the three on-demand ones runs end to end and
     returns the shape the page renders;
  2. **the beats say the right thing** -- beat 1's verdict map actually changes
     when the declaration does, and stays put when it does not;
  3. **the recorded numbers are read, not remembered** -- every figure the
     explainer quotes is checked against `out/w141/w141.json`, so a stale
     sentence fails here rather than being read off a screen by somebody;
  4. **the prose does not overclaim** -- the vocabulary this project has decided
     it has not earned is asserted absent.

The fourth is the one worth defending.  A demo's prose is the part most likely to
drift, and it is the part no other test in this repository looks at.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from atlas.cases import front_wing as F
from atlas.demo_frontwing import ablation as AB
from atlas.demo_frontwing import explain as EX
from atlas.demo_frontwing import race as RA
from atlas.demo_frontwing import substitution as SUB
from atlas.demo_frontwing.balance import Balance, spring_energy, strain_energy
from atlas.demo_frontwing.engine import (
    RULE_NOTE,
    DemoConfig,
    Engine,
    load_recorded,
    seam_verdicts,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ARTIFACT = os.path.join(_ROOT, "out", "w141", "w141.json")


@pytest.fixture(scope="module")
def art():
    if not os.path.isfile(_ARTIFACT):
        pytest.skip("out/w141/w141.json is not here")
    with open(_ARTIFACT, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def field():
    p = os.path.join(_ROOT, "out", "w141", "settled.npz")
    if os.path.isfile(p):
        d = np.load(p)
        return d["u"], d["v"]
    return np.full((144, 208), F.U_INF), np.zeros((144, 208))


# ---------------------------------------------------------------------------
# beat 1 -- the substitution
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def subs(field):
    u, v = field
    return SUB.evaluate(u, v, dict(F.DESIGN_REF), F.H0_REF,
                        tiling=F.DEFAULT_TILING)


def test_the_substitution_changes_the_verdict_map_and_the_incumbent_does_not(subs):
    """Beat 1's whole claim in one assertion.

    Same graph, same field, same design, three DESCRIPTIONS of the fluid expert.
    The incumbent and the honest Poseidon-T record leave the colour map alone;
    declaring the elliptic part turns every seam red.  If this ever stops being
    true the demo is showing a rigged failure and must be pulled.
    """
    rows = subs["rows"]
    inc = rows["windowns"]
    dec = rows["poseidon_declared"]
    prb = rows["poseidon_probed"]
    assert inc["n_red"] == dec["n_red"] == 2, (inc["n_red"], dec["n_red"])
    assert prb["n_red"] == len(subs["seam_order"]) == 9, prb["n_red"]
    assert prb["n_amber"] == 0


def test_the_honest_poseidon_record_adds_reasons_without_adding_a_refusal(subs):
    """The middle row is the one to sit on, and it has to be legible as such.

    Nothing is refused that was not refused before -- and three rules that said
    nothing about the incumbent now name every fluid window.  That is the shape
    of "the record is honest and the compiler still cannot decide".
    """
    dec = subs["rows"]["poseidon_declared"]
    fluid = [s for s in subs["seam_order"] if s not in ("wet", "mount")]
    assert len(fluid) == 7
    for s in fluid:
        added = dec["added"][s]
        assert added["refusals"] == [], (s, added)
        assert set(added["decertifications"]) == {
            "L2/R10/W60", "L2/R10/halo", "L4/R2b/W46"}, (s, added)


def test_the_refusal_needs_a_cut_and_the_referent_column_says_so(field):
    """W114's premise check, which this beat would misrepresent without.

    On the single-window referent the fluid family has one member, so nothing has
    been decomposed and R10 correctly stays quiet however the expert is declared.
    A viewer must not read that as the substitution passing, so the payload
    carries the sentence and this asserts both halves.
    """
    u, v = field
    one = SUB.evaluate(u, v, dict(F.DESIGN_REF), F.H0_REF,
                       tiling=F.SINGLE_TILING)
    assert one["n_fluid"] == 1 and one["decomposed"] is False
    assert one["rows"]["poseidon_probed"]["n_red"] == 2
    assert "premise does not hold" in one["premise"]
    assert "six windows" in one["premise"]


def test_every_rule_the_substitution_can_surface_has_a_note(subs):
    """A panel that shows `L2/R10/halo` and nothing else is a panel nobody can
    act on. Every rule any candidate adds at any seam must have a one-line
    explanation, or the beat is showing a code to somebody who cannot read it."""
    seen = set()
    for row in subs["rows"].values():
        for a in row["added"].values():
            seen |= set(a["refusals"]) | set(a["decertifications"])
    missing = sorted(r for r in seen if r not in RULE_NOTE)
    assert not missing, f"no RULE_NOTE for: {missing}"


def test_the_substituted_record_matches_the_one_the_real_wrapper_builds():
    """The demo's Poseidon-T declaration is not invented for the demo.

    `atlas/cases/poseidon.py` builds the same record from the live checkpoint;
    this asserts the fields that reach a rule agree, so the beat cannot drift
    into showing a description of a model nobody has.
    """
    from atlas.cases import poseidon as P
    import inspect

    src = inspect.getsource(P.poseidon_capabilities)
    want = {
        "stencil_radius": "stencil_radius=2,",
        "substeps_per_macro_step": "substeps_per_macro_step=1,",
        "time_discretization": "time_discretization=TimeDiscretization.UNKNOWN,",
        "differentiable": "differentiable=Differentiable.NONE,",
    }
    for field_, needle in want.items():
        assert needle in src, (
            f"{field_} in the demo's declaration no longer matches "
            f"poseidon_capabilities; expected to find {needle!r}")
    d = SUB.BY_KEY["poseidon_declared"].fields
    assert int(d["stencil_radius"]) == 2
    assert int(d["substeps_per_macro_step"]) == 1


# ---------------------------------------------------------------------------
# beat 3 -- the ablation
# ---------------------------------------------------------------------------


def test_the_ablation_measures_its_own_operating_point_rather_than_taking_one(field):
    """The trap `ablation.py` exists to remove.

    A frozen suspension pinned at the wrong height measures a transit, not the
    seam. The function marches the live column first and uses THAT height, so a
    caller passing a stale hint cannot corrupt the answer -- and the hint's error
    is reported rather than swallowed.
    """
    u, v = field
    out = AB.freeze_each_seam(dict(F.DESIGN_REF), 12, u, v, h_hint=0.99,
                              tiling=F.SINGLE_TILING)
    assert out["h_live"] == pytest.approx(out["live"]["h"], rel=0, abs=0)
    assert abs(out["h_live"] - 0.99) > 0.5
    assert out["h_hint"] == 0.99
    assert out["h_hint_error"] > 1.0


def test_both_frozen_limits_are_named_after_the_case_study_they_recover(field):
    """Freezing a seam must land on a parent, or the word `assembly` is not
    earned. The static version of this is the gate in section 1 of the results
    page; here it only has to be true of the labels and the knobs."""
    limits = {tag: (knob, lim) for tag, knob, lim, _p, _w in AB.LIMITS}
    assert limits["no suspension"] == ("k", 1.0e9)
    assert limits["no structure"] == ("e_star", 1.0e9)
    parents = " ".join(p for _t, _k, _l, p, _w in AB.LIMITS)
    assert "CS-12" in parents and "CS-10" in parents


# ---------------------------------------------------------------------------
# beat 4 -- the race
# ---------------------------------------------------------------------------


def test_the_demos_objective_agrees_with_the_drivers(field):
    """A demo racing a different objective from the one the results page reports
    is racing a different problem. Both are `FrontWingRollout.objective` on the
    single-window column at the reference design, so they must agree to
    round-off -- and this is the test that notices if one of them is changed."""
    u, v = field
    ro = F.FrontWingRollout(tiling=F.SINGLE_TILING, coupling="tight",
                            motion=True, design=dict(F.DESIGN_REF))
    J, gs, gd, ts, td = ro.objective(dict(F.DESIGN_REF), steps=6, u0=u, v0=v,
                                     grad=False)
    mine = RA.evaluate(dict(F.DESIGN_REF), 6, u, v, want_grad=False)
    assert mine["J"] == pytest.approx(float(J), rel=1e-12)
    assert mine["true_sigma"] == pytest.approx(float(ts), rel=1e-12)
    assert mine["L"] == pytest.approx(float(F.penalised(J, gs, gd, F.PENALTY)),
                                      rel=1e-12)


def test_the_race_warms_both_columns_before_either_clock_starts(field):
    """`stage_cost` published a 9.47 adjoint premium that a warmed re-measurement
    put at 5.4-5.8, and an unwarmed three-step race reads 25x for the same
    reason: the first call in each column pays for a dispatch cache, a
    factorisation and the tape's first allocation. The warm-up is reported
    separately, and it must be nonzero."""
    u, v = field
    out = RA.race(u, v, steps=4, n_grad=2, n_pop=3)
    assert out["warm_s"] > 0.0
    assert out["done"] is True
    assert out["gradient"]["n"] == 2 and out["population"]["n"] == 3
    #: the warm-up is NOT inside the race's own clock
    assert out["wall_s"] >= out["gradient"]["wall_s"]


def test_the_race_reports_wall_clock_and_labels_the_rollout_count(field):
    """W143. The rollout ratio prices a gradient iterate at one forward
    evaluation, which it is not, so both are returned and the page quotes the
    wall-clock one first. This asserts the payload carries both rather than
    only the flattering one."""
    u, v = field
    out = RA.race(u, v, steps=4, n_grad=2, n_pop=6)
    c = out.get("comparison")
    assert c is not None, out
    for k in ("ratio_wall", "ratio_rollouts", "quality_ratio",
              "gradient_wall_to_target", "population_wall_to_target"):
        assert k in c, k


def test_a_declined_design_scores_the_sentinel_and_does_not_stop_the_search():
    """A population sampler WILL propose designs outside an expert's envelope,
    and losing the run to one of them is losing the run to a correct answer."""
    assert RA.DECLINED_L <= -1.0e3
    bad = dict(F.DESIGN_REF, h0=F.DESIGN_BOX["h0"][0], k=F.DESIGN_BOX["k"][0])
    #: not asserted to decline -- that is a physics claim and belongs in tier 30.
    #: What is asserted is the CONTRACT: if it declines, the caller gets a
    #: finite score and a reason rather than an exception.
    out = RA.evaluate(bad, 4, np.full((144, 208), F.U_INF),
                      np.zeros((144, 208)), want_grad=False)
    assert np.isfinite(out["L"])
    if out["declined"]:
        assert out["L"] == RA.DECLINED_L
        assert out["feasible"] is False


# ---------------------------------------------------------------------------
# beat 5 -- the balance
# ---------------------------------------------------------------------------


def test_the_balance_refuses_to_difference_one_sample():
    """The first call establishes a baseline and returns nothing. Inventing a
    difference from a one-sided stencil is exactly the error that put the first
    macro-step's residual at 1.81 against a static accounting's 1.00."""
    b = Balance(dt=0.0125)
    assert b.observe(0.0, 1.0, 0.0, 0.0) is None
    row = b.observe(0.1, 1.0, 0.0, 0.0)
    assert row is not None and row["n"] == 1


def test_the_balance_carries_BOTH_receivers_energies():
    """The plate starts flat so its strain energy at release is identically
    zero, which is what makes it easy to carry one initial value and not the
    other -- and the spring is already loaded at release, so its energy is the
    largest single number in the balance."""
    S = np.eye(3)
    assert strain_energy(np.zeros(3), S, 1.0) == 0.0
    assert spring_energy(2.5, 0.30, 0.24) == pytest.approx(0.5 * 2.5 * 0.06 ** 2)
    b = Balance(dt=1.0)
    b.observe(0.0, 5.0, 0.0, 0.0)
    row = b.observe(0.0, 5.0, 0.0, 0.0)
    #: no energy change and no power: the balance closes exactly
    assert row["corrected"] == 0.0


def test_the_balance_quotes_the_window_it_read_the_settled_value_over():
    """W140's discipline: a quantity read over a WINDOW is quoted with the
    window, and with the evidence that the window is what it is called."""
    b = Balance(dt=1.0)
    for i in range(20):
        b.observe(float(i), 0.0, 1.0, 0.0)
    p = b.payload()
    assert p["window"] >= 1 and p["window"] <= p["n"]
    assert p["settled"] is not None
    assert p["recorded"]["steps"] == 480


# ---------------------------------------------------------------------------
# the recorded numbers -- read, never remembered
# ---------------------------------------------------------------------------


def test_the_loader_reads_the_artifact_rather_than_a_copy_of_it(art):
    """`wall-clock numbers rot`: a ms/step figure in this project's own docs was
    wrong by 4x the next day on unchanged code. So every recorded figure on
    screen comes out of the artifact at load time, and this asserts the loader
    is reading the same file the driver wrote."""
    rec = load_recorded()
    assert rec["available"] is True
    assert rec["residual"]["settled"] == art["residual"]["settled"]
    assert rec["design"]["best_J"] == art["design"]["gradient"]["best_feasible"]["J"]
    assert (rec["design"]["comparison"]["ratio_wall"]
            == art["design"]["comparison"]["ratio_wall"])
    assert (rec["controls"]["k_to_infinity"]
            == art["controls"]["k_to_infinity"]["rel"])


def test_the_loader_says_so_rather_than_inventing_a_number_when_it_is_absent(tmp_path):
    rec = load_recorded(str(tmp_path / "nothing.json"))
    assert rec["available"] is False
    assert "why" in rec and rec["why"]


def test_declined_and_infeasible_are_different_numbers_and_stay_that_way(art):
    """66 of the population's 200 evaluations were infeasible and 39 were
    DECLINED by an expert; the other 27 were inside the model and over a
    ceiling. Reporting 66 as declines would overstate W145 by a factor of 1.7,
    so the loader carries both and the panel shows both rows."""
    rec = load_recorded()["design"]
    hist = art["design"]["population"]["history"]
    assert rec["declined_pop"] == sum(1 for h in hist if h.get("L", 0.0) <= -999.0)
    assert rec["infeasible_pop"] == sum(1 for h in hist if not h.get("feasible"))
    assert rec["declined_pop"] < rec["infeasible_pop"], (
        "if these ever coincide the panel's two rows are redundant, but they "
        "must still be derived separately")
    #: and the denominator is the history length, not the requested budget --
    #: the gradient column records one more evaluation than it takes steps
    assert rec["evals_grad"] == len(art["design"]["gradient"]["history"])


def test_the_envelope_beat_quotes_the_two_columns_the_artifact_actually_has(art):
    """Beat 2's whole demonstration is 19.3% against 27.5%, and both have to
    come from the file. The enforced column is `design`; the pre-W145 behaviour
    is `design_unenforced`, kept rather than overwritten."""
    rec = load_recorded()
    enf, un = rec["design"], rec["design_unenforced"]
    assert enf["start_J"] == un["start_J"], "the two columns must share a start"
    gain_enf = enf["best_J"] / enf["start_J"] - 1.0
    gain_un = un["best_J"] / un["start_J"] - 1.0
    assert 0.15 < gain_enf < 0.25, gain_enf
    assert gain_un > gain_enf, (gain_un, gain_enf)
    #: the enforced column is stopped by the ENVELOPE, not by a ceiling, and the
    #: unenforced one is not stopped at all
    assert enf["declined_grad"] > 0 and un["declined_grad"] == 0
    assert enf["active"]["sigma"] is False
    assert un["active"]["sigma"] is True


def test_every_number_the_explainer_quotes_is_in_the_artifact(art):
    """The prose carries figures, and prose drifts. Each of these is checked
    against the file it came from, so a stale sentence fails here rather than
    being read off a screen by somebody who believes it."""
    prose = " ".join(EX._all_prose())
    d, un = art["design"], art["design_unenforced"]
    checks = [
        (f"{(d['gradient']['best_feasible']['J'] / d['start_eval']['J'] - 1) * 100:.1f}%",
         "the enforced improvement"),
        (f"{(un['gradient']['best_feasible']['J'] / un['start_eval']['J'] - 1) * 100:.1f}%",
         "the unenforced improvement"),
        (f"{art['residual']['settled']:.2e}".replace("e-08", "e-8"),
         "the settled power residual"),
        (str(art["residual"]["steps"]), "the residual's horizon"),
    ]
    for needle, what in checks:
        assert needle in prose, f"{what}: {needle!r} is not in the explainer"
    #: the two ablation figures, to two decimals
    w = art["ablation"]["at_optimum"]
    for key, val in w["worth_at_reference"].items():
        if "suspension" in key:
            assert f"{val * 100:.2f}%" in prose, key
    for key, val in w["worth"].items():
        if "suspension" in key:
            assert f"{val * 100:.2f}%" in prose, key


def test_the_explainer_does_not_use_the_vocabulary_this_project_has_not_earned():
    """The one test in this repository that reads the prose. A demo's copy is the
    part most likely to drift into an overclaim and the part nothing else looks
    at."""
    prose = " ".join(EX._all_prose()).lower()
    hits = [w for w in EX.FORBIDDEN if w in prose]
    assert not hits, f"the explainer says: {hits}"


def test_the_explainer_says_out_loud_that_nothing_is_certified():
    """Every panel is amber or red and the page must not let that be missed."""
    titles = " ".join(x["title"] for x in EX.NOT_CLAIMED).lower()
    assert "certified" in titles
    body = " ".join(x["plain"] for x in EX.NOT_CLAIMED).lower()
    assert "not one joint" in body or "none ever has been" in body
    assert "classical solver" in body


def test_the_explainer_covers_every_beat_the_page_renders():
    keys = {b["key"] for b in EX.BEATS}
    assert keys == {"substitution", "envelope", "ablation", "race", "balance"}
    for b in EX.BEATS:
        for k in ("title", "plain", "technical", "matters", "watch"):
            assert b.get(k), (b["key"], k)
    assert len(EX.WHY_NOVEL) == 5
    assert [x["n"] for x in EX.WHY_NOVEL] == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# the engine, and the one switch that can make it lie
# ---------------------------------------------------------------------------


def test_the_engine_enforces_both_envelopes_on_every_path_by_default():
    """W145. `run` enforced them and `objective` did not; the demo's own march
    and its optimiser both drive `macro_step` directly, so the check lives in
    `_absorb`, which every path funnels through."""
    import inspect

    from atlas.demo_frontwing import engine as E
    src = inspect.getsource(E.Engine._absorb)
    assert "check_envelopes" in src
    assert DemoConfig().enforce is True, "enforcement is the default"

    #: **and the three other `objective`s the same asymmetry was found in.**
    #: Asserted by PROPERTY rather than by name: `front_wing` carries the shared
    #: `check_envelopes` method, and `wing_fsi` and `ground_effect` each carry
    #: the rule inline in their own loop.  Three implementations of one rule is
    #: the exact shape that produced W145 -- one path had the check and another
    #: did not -- so this test asks each `objective` whether it CAN raise on a
    #: declared envelope, which is the thing that must stay true however they
    #: are written.
    from atlas.cases import ground_effect as G
    from atlas.cases import wing_fsi as W

    for cls, needle in ((F.FrontWingRollout, "check_envelopes"),
                        (W.FSIRollout, "declared envelope"),
                        (G.GroundRollout, "declared envelope")):
        src = inspect.getsource(cls.objective)
        assert needle in src, (cls, needle)
        #: the guard is on `self.motion`, so a fixed-shape column -- which has
        #: no interface to move and so no envelope to leave -- is not
        #: gratuitously refused. In `front_wing` that guard is one level down,
        #: inside the shared method.
        guarded = src if "self.motion" in src else inspect.getsource(
            cls.check_envelopes)
        assert "raise RuntimeError" in guarded, cls
        assert "self.motion" in guarded, cls


def test_turning_enforcement_off_records_what_the_check_would_have_said():
    """The switch does not remove the check. It records the verdict and marches
    on, so the panel can show the number the model declines to stand behind
    beside the reason it declines."""
    import inspect

    from atlas.demo_frontwing import engine as E
    src = inspect.getsource(E.Engine._absorb)
    assert "self.cfg.enforce" in src
    assert "self.outside" in src
    assert "raise" in src, "with enforcement ON it must still raise"


def test_the_engine_holds_the_march_for_the_beats_that_time_themselves():
    """Beat 4 reports a wall-clock ratio; a 12-frames-per-second march in the
    background would be measured instead of the search. Measured: the race went
    from 4% of its budget in 25 seconds to finishing."""
    import inspect

    from atlas.demo_frontwing import engine as E
    assert "quiet=True" in inspect.getsource(E.Engine.request_race)
    assert "quiet=True" in inspect.getsource(E.Engine.request_ablation)
    #: and the SHORT one does not need it
    assert "quiet=True" not in inspect.getsource(E.Engine.request_substitution)
    assert "self._quiet" in inspect.getsource(E.Engine._loop)


def test_the_demo_and_the_driver_group_the_decisions_the_same_way(field):
    """A demo that groups the compiler's decisions differently from the driver is
    showing a different thing from the one the results page reports."""
    import importlib.util

    u, v = field
    spec = importlib.util.spec_from_file_location(
        "_w141_driver", os.path.join(_ROOT, "scripts", "w141_poc2_frontwing.py"))
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)
    from atlas.compiler import compile_scheme

    g, _e = F.build(u, v, motion=True, tiling=F.DEFAULT_TILING)
    r = compile_scheme(g)
    assert seam_verdicts(g, r) == drv.seam_verdicts(g, r)


def test_the_race_budget_keeps_the_recorded_runs_own_ratio():
    """Picking the two budgets independently would mean picking them until the
    answer came out the way the panel would prefer, which is the one thing a
    comparison that exists to be honest cannot do."""
    c = DemoConfig()
    recorded = 200 / 30
    live = c.race_pop / c.race_grad
    assert abs(live / recorded - 1.0) < 0.02, (live, recorded)
