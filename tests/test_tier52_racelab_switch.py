"""Tier 52 -- the envelope check (W222), and RaceLab phase 2: the switch.

What this file pins:

1. **The envelope is CONSULTED.**  `MachineAgent.validity`, the disk's induction
   clamp and the fluid window's cell-Reynolds bound all existed before Tier 52
   and none was read; `RaceRollout._absorb` reads all three, once, at the end of
   every macro-step, and `enforce` defaults ON.  The test that matters is the
   PAIR: the CS-19 machine declines on a given field and the host-sized one does
   not, so the check separates two columns that differ in one declaration.
2. **Three states and never two.**  A predicate that cannot be evaluated reports
   ``None``, not ``False`` and not ``True``.
3. **`machine_for_host` is a similarity, not a fit** -- at the reference inflow
   it reproduces `machine_for_rotor` bitwise, and at the duct's inflow it
   returns the reference induction exactly.
4. **The scaling is over-determined by the window's geometry**, so the
   composition layer's own exchange cadence asks the checkpoint for a thirty-
   second of its native lead.  That is arithmetic and the test asserts it.
5. **The error does NOT shrink with the macro step.**  `prior-art-and-novelty`
   §4 names that assumption as one partitioned-coupling theory makes and a
   frozen expert lacks; here it is measured false, with a sharp minimum AT the
   native lead.
6. **Four of five families have no learned option**, and the switch says so with
   a reason rather than hiding it.
7. **Nothing claims the learned expert pays.**

The live pieces need Poseidon-T in the local HF cache and skip without it.
The artifact tests need ``out/racelab2/racelab2.json``; rebuild it with
``python scripts/tier52_racelab_switch.py`` -- see that file's header.
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

from atlas.cases import ground_effect as GE                           # noqa: E402
from atlas.cases import powertrain as PT                              # noqa: E402
from atlas.cases import racelab as RL                                 # noqa: E402
from atlas.cases import racelab_switch as SW                          # noqa: E402
from atlas.cases import vehicle_march as VM                           # noqa: E402
from atlas.cases import wake_array as WA                              # noqa: E402

ART = os.path.join(_ROOT, "out", "racelab2", "racelab2.json")
PAGE = os.path.join(_ROOT, "wiki", "concepts", "Atlas 0.1", "common",
                    "case-study-racelab-switch-atlas-0.1.md")
U_DUCT = 0.6691530373612168


def _art():
    if not os.path.isfile(ART):
        pytest.skip("out/racelab2/racelab2.json is absent; rebuild it with "
                    "python scripts/tier52_racelab_switch.py")
    with open(ART, encoding="utf-8") as fh:
        return json.load(fh)


def _page():
    if not os.path.isfile(PAGE):
        pytest.skip("the case study page is absent")
    with open(PAGE, encoding="utf-8") as fh:
        return fh.read()


def _settled():
    p = os.path.join(_ROOT, "out", "racelab2", "cache", "settled.npz")
    if not os.path.isfile(p):
        pytest.skip("no settled field; run --stages spinup")
    d = np.load(p)
    return d["u"], d["v"]


# ---------------------------------------------------------------------------
# 1. the envelope is consulted, and enforce is ON by default
# ---------------------------------------------------------------------------


def test_enforce_is_on_by_default():
    """A check that is off by default is not a check."""
    r = RL.RaceRollout()
    assert r.enforce is True
    import inspect
    sig = inspect.signature(RL.march)
    assert sig.parameters["enforce"].default is True


def test_the_predicates_the_check_reads_all_existed_before_this_tier():
    """The repair is that they are CONSULTED, not that they were written.

    `MachineAgent.validity` declares the motoring condition in its own
    docstring, and Tier 51 walked past it for 600 macro-steps.
    """
    els = VM.machine_for_rotor(RL.HOST_ROTOR_WIDTH)
    assert callable(els["MGU"].validity)
    assert els["MGU"].validity(np.full(1, 30.0)) is True
    assert els["MGU"].validity(np.full(1, 1.0)) is False
    assert callable(WA.RotorDisk(agent_id="P", u_ref=np.ones(WA.ROTOR_CELLS)).validity)


def test_the_report_has_three_states_and_never_two():
    """`None` is *not consultable* and is neither a pass nor a fail."""
    r = RL.RaceRollout(joins=("J1", "J2"))          # no J3, so no shaft
    rep = r.validity_report(None, None)
    assert rep["MGU"]["valid"] is None
    assert rep["ROTOR"]["valid"] is None
    assert rep["FLUID"]["valid"] is None, "no field handed in is not a pass"
    for k, row in rep.items():
        if not k.startswith("_"):
            assert row["valid"] in SW.Mode.__members__.values() or \
                row["valid"] in (True, False, None)
    assert "MGU" in rep["_unconsultable"]


def test_the_check_separates_two_columns_that_differ_in_one_declaration():
    """**The pair is the test.**  From ONE settled field, the machine CS-19
    declared must decline and the host-sized one must not."""
    import torch
    torch.set_num_threads(1)
    u, v = _settled()
    with pytest.raises(RL.EnvelopeDeclined) as exc:
        RL.march(u, v, steps=2, join_coupling="lagged")
    assert "MGU" in exc.value.report["_declined"]
    m = RL.march(u, v, steps=2, join_coupling="lagged", host_inflow=U_DUCT)
    assert m.notes["outside_the_envelope_steps"] == 0
    assert m.notes["envelope_at_the_end"]["MGU"]["valid"] is True


def test_enforce_off_still_runs_the_check_and_stamps():
    """OFF does not remove the check; it records what it would have said."""
    import torch
    torch.set_num_threads(1)
    u, v = _settled()
    m = RL.march(u, v, steps=2, join_coupling="lagged", enforce=False)
    assert m.notes["enforce"] is False
    assert m.notes["outside_the_envelope_steps"] >= 1
    assert "MGU" in m.notes["outside_the_envelope"]["declined"]


# ---------------------------------------------------------------------------
# 2. the repair is a similarity
# ---------------------------------------------------------------------------


def test_machine_for_host_is_the_identity_at_its_reference_inflow():
    a = VM.machine_for_rotor(RL.HOST_ROTOR_WIDTH)
    b = RL.machine_for_host(RL.U_HOST_REF, RL.U_HOST_REF)
    for k in a:
        for f in ("resistance", "k_e", "k_t", "r_total"):
            va = getattr(a[k], f, None)
            if va is not None:
                assert va == getattr(b[k], f), (k, f)


def test_machine_for_host_returns_the_reference_induction_exactly():
    """A similarity preserves the electrical solution and changes only its
    size; a fit would not."""
    ref = VM.operating_point(np.full(WA.ROTOR_CELLS, RL.U_HOST_REF),
                             width=RL.HOST_ROTOR_WIDTH,
                             scale=RL.HOST_ROTOR_WIDTH)[0]
    fix = VM.operating_point(np.full(WA.ROTOR_CELLS, U_DUCT),
                             width=RL.HOST_ROTOR_WIDTH,
                             scale=RL.HOST_ROTOR_WIDTH,
                             elements=RL.machine_for_host(U_DUCT))[0]
    assert fix.induction == pytest.approx(ref.induction, abs=1e-12)
    assert fix.rotor_valid is True


def test_the_crossover_is_where_the_closed_form_says():
    els = VM.machine_for_rotor(RL.HOST_ROTOR_WIDTH)
    slope = 2.0 * 7.5 * float(els["MGU"].k_e) / RL.HOST_ROTOR_WIDTH
    u_star = float(PT.V_OC) / slope
    assert u_star == pytest.approx(0.89333, rel=1e-3)
    below = VM.operating_point(np.full(WA.ROTOR_CELLS, u_star * 0.99),
                               width=RL.HOST_ROTOR_WIDTH,
                               scale=RL.HOST_ROTOR_WIDTH)[0]
    above = VM.operating_point(np.full(WA.ROTOR_CELLS, u_star * 1.05),
                               width=RL.HOST_ROTOR_WIDTH,
                               scale=RL.HOST_ROTOR_WIDTH)[0]
    assert below.current < 0.0 < above.current


# ---------------------------------------------------------------------------
# 3. the spin-up, which the check found
# ---------------------------------------------------------------------------


def test_the_release_state_is_one_the_experts_admit():
    a = _art()
    if "spinup" not in a:
        pytest.skip("stage spinup has not run")
    s = a["spinup"]
    assert s["release_state_is_admitted"] is True
    assert s["macro_steps_outside"] > 0, (
        "if the freestream release never left an envelope then the spin-up is "
        "not what this stage says it is"
    )
    assert s["first_macro_step_outside"] == 0
    assert s["last_macro_step_outside"] < s["steps"]


# ---------------------------------------------------------------------------
# 4. the scaling is over-determined by the geometry
# ---------------------------------------------------------------------------


def test_one_window_spans_two_length_units_and_that_is_not_a_knob():
    t, _i = RL.layout()
    assert t.wx * RL.DX == pytest.approx(2.0)


def test_the_composition_cadence_asks_for_a_thirty_second_of_the_native_lead():
    a = _art()
    if "scaling" not in a:
        pytest.skip("stage scaling has not run")
    s = a["scaling"]
    assert s["exchange"]["lead_over_native"] == pytest.approx(1.0 / 32.0)
    assert s["macro_step"]["lead_over_native"] == pytest.approx(1.0 / 8.0)
    assert s["native"]["lead_over_native"] == pytest.approx(1.0)
    #: and it is arithmetic, not a measurement
    assert GE.MACRO_DT / GE.EXCHANGES / (2.0 / 2.0) == pytest.approx(0.003125)


# ---------------------------------------------------------------------------
# 5. the error does not shrink with the macro step
# ---------------------------------------------------------------------------


def test_the_error_has_a_minimum_AT_the_native_lead():
    """`prior-art-and-novelty` §4 names the assumption partitioned-coupling
    theory makes and a frozen expert lacks -- *that error shrinks with the
    macro step* -- and this is it measured false."""
    a = _art()
    if "windows" not in a:
        pytest.skip("stage windows has not run")
    rows = a["windows"]["error_vs_lead"]["rows"]
    by_lead = {r["n_macro"]: r["median"] for r in rows}
    assert 8 in by_lead, "the native lead is eight macro-steps on this tiling"
    native = by_lead[8]
    assert native == min(by_lead.values()), (
        "the minimum is not at the native lead", by_lead)
    #: and the error GROWS as the step shrinks below native
    assert by_lead[4] > native and by_lead[2] > native and by_lead[1] > native
    assert by_lead[1] > 1.5 * native


def test_the_per_window_error_needs_no_global_referent():
    """Section 5.4's third clause: it is a one-step comparison on the same
    input state, so every window has its own number."""
    a = _art()
    if "windows" not in a:
        pytest.skip("stage windows has not run")
    o = a["windows"]["one_macro_step"]
    t, _i = RL.layout()
    assert set(o["per_window"]) == set(t.names)
    assert o["n_windows"] == t.n_windows
    assert o["min"] <= o["median"] <= o["max"]


# ---------------------------------------------------------------------------
# 6. four of five families have no learned option
# ---------------------------------------------------------------------------


def test_four_of_five_families_have_no_learned_option():
    a = _art()
    if "families" not in a:
        pytest.skip("stage families has not run")
    f = a["families"]
    assert f["n_families"] == 5
    assert f["families_with_a_learned_option"] == list(SW.LEARNED_FAMILIES)
    assert len(f["families_without"]) == 4


def test_every_family_without_an_option_says_why():
    """Requirements §4.3: greyed out WITH A REASON, not hidden."""
    for fam, why in SW.NO_LEARNED_OPTION.items():
        assert isinstance(why, str) and len(why) > 40, fam
    a = _art()
    if "families" not in a:
        pytest.skip("stage families has not run")
    for agent, row in a["families"]["per_agent"].items():
        if not row["switch_available"]:
            assert row["why_not"], agent
            assert row["modes"] == ["classical"]
        else:
            assert set(row["modes"]) == set(SW.MODES)


# ---------------------------------------------------------------------------
# 7. the certified mode, and what it is not
# ---------------------------------------------------------------------------


def test_certified_is_not_a_per_macro_step_mode_and_says_so():
    """Defect correction certifies a FIXED POINT; an explicit time step has
    none to certify.  Assigning it to a marching window raises rather than
    quietly running something else."""
    t, _i = RL.layout()
    with pytest.raises(ValueError, match="steady-state"):
        SW.MixedRollout(tiling=t, assignment={t.names[0]: "certified"})


def test_the_certified_mode_carries_its_null_replacement():
    a = _art()
    if "certified" not in a:
        pytest.skip("stage certified has not run")
    for w, row in a["certified"]["windows"].items():
        assert "with_the_checkpoint" in row and "null_identity" in row
        assert row["null_identity"]["use_learned"] is False
        assert isinstance(row["what_the_checkpoint_bought_in_classical_calls"],
                          int)


# ---------------------------------------------------------------------------
# 8. the mixed march, and the vocabulary
# ---------------------------------------------------------------------------


def test_a_mixed_march_is_scored_against_the_all_classical_column():
    a = _art()
    if "assign" not in a:
        pytest.skip("stage assign has not run")
    cols = a["assign"]["columns"]
    assert cols["classical"]["rms_field_error_vs_all_classical"] == 0.0
    assert cols["classical"]["speed_ratio_vs_all_classical"] == 1.0
    for tag, row in cols.items():
        assert sum(row["ledger"].values()) == RL.layout()[0].n_windows


def test_the_learned_columns_left_the_fluid_expert_s_own_bound():
    """They do not merely degrade: the check says they leave a declared
    envelope, and that is recorded rather than hidden."""
    a = _art()
    if "assign" not in a:
        pytest.skip("stage assign has not run")
    cols = a["assign"]["columns"]
    assert cols["classical"]["outside_steps"] == 0
    assert cols["learned"]["outside_steps"] > 0


def test_the_page_does_not_claim_the_learned_expert_pays():
    page = _page().lower()
    for banned in ("poseidon pays", "the learned expert wins",
                   "faster and more accurate", "replaces the classical"):
        assert banned not in page


def test_the_page_names_what_the_tier_did_not_do():
    assert "What this tier did NOT do, named" in _page()


def test_NeuberNet_is_not_loaded_or_referenced_as_code():
    import re
    n = "neuber" + "net"
    banned = re.compile(
        r"import\s+%s|from\s+%s|%s\s*\.\w|%s[/\\]|cache[/\\]%s"
        % (n, n, n, n, n), re.IGNORECASE)
    for path in (os.path.join(_ROOT, "atlas", "cases", "racelab_switch.py"),
                 os.path.join(_ROOT, "scripts", "tier52_racelab_switch.py"),
                 os.path.abspath(__file__)):
        with open(path, encoding="utf-8") as fh:
            hit = banned.search(fh.read())
        assert hit is None, f"{path} reaches it at {hit.group(0)!r}"
