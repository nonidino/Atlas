"""Tier 48 -- what the kill tests measured, pinned against the artifact.

[[learned-contribution-kill-tests]] and [[defect-correction-learned-operator]] are
written from ``out/w202/w202.json``.  These tests assert the numbers both pages
quote, so a page that drifts from the record fails rather than lying quietly, and
so the record itself cannot be regenerated into something else without a failing
test.  Nothing here loads an expert; the artifact is JSON.

The claims, in the order the pages make them:

1. the reproduction control -- the driver's maps ARE CS-7's, bitwise;
2. the classical settled state, its Theta, and the fixed point that is not
   isolated (a march whose residual is machine zero a third of the cold distance
   away from the reference);
3. a learned START buys nothing -- transit-limited relaxation;
4. defect correction at N=2: the ordering P < Pc < Z < cold in classical calls,
   and every arm inside the certificate;
5. the cost ledger, and the crossing that decides the gate's accounting clause;
6. the relative-energy certificate: valid per step, floor above the checkpoint's
   own error at step 20, and vacuous over a transit;
7. self-consistency: a real lower bound, blind to the identity map;
8. the price of one fine-tuning iteration.

Rebuild the artifact with ``python scripts/w202_kill_tests.py``; the tests skip
without it.
"""

from __future__ import annotations

import json
import os

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT = os.path.join(_ROOT, "out", "w202", "w202.json")
THEORY = os.path.join(_ROOT, "wiki", "concepts", "Atlas 0.1", "common",
                      "defect-correction-learned-operator.md")
KILLS = os.path.join(_ROOT, "wiki", "concepts", "Atlas 0.1", "common",
                     "learned-contribution-kill-tests.md")


@pytest.fixture(scope="module")
def art():
    if not os.path.exists(ARTIFACT):
        pytest.skip("out/w202/w202.json is not on disk: run scripts/w202_kill_tests.py")
    with open(ARTIFACT, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# 1. the reproduction control
# ---------------------------------------------------------------------------


def test_the_driver_maps_are_cs7s_bitwise(art):
    r = art["repro"]["N2"]
    assert r["poseidon_vs_composed_step_maxabs"] == 0.0
    assert r["monolith_vs_long_march_advance_maxabs"] == 0.0
    # and the checkpoint is not doing nothing: it moves the state beyond the
    # composition layer that wraps it
    assert r["poseidon_minus_null_expert_rms"] > 1e-3
    assert r["passed"]


# ---------------------------------------------------------------------------
# 2. the referent, and the fixed point that is not isolated
# ---------------------------------------------------------------------------


def test_the_classical_reference_is_deterministic_and_settles(art):
    rec = art["reference"]["N2"]
    assert rec["replay_difference"] == 0.0
    assert rec["final_residual"] < 1e-11
    assert 1.0 < rec["theta"]["theta_min"] <= rec["theta"]["theta"] < 10.0


def test_the_classical_fixed_point_is_not_isolated_when_the_outflow_is_state(art):
    """The finding that is about no candidate: a residual certificate is a
    statement about a DECLARED problem."""
    arms = art["warm"]["N2"]["arms"]
    ring = arms["ring_control_unpinned"]
    assert ring["outflow_ring_pinned"] is False
    # converged exactly -- and a third of the cold distance from the reference
    assert ring["final_residual"] < 1e-12
    assert ring["final_error"] > 0.25 * art["warm"]["N2"]["cold_distance"]
    # the same start, with the column declared, reaches the reference
    pinned = arms["poseidon_settled"]
    assert pinned["outflow_ring_pinned"] is True
    assert pinned["final_error"] < 1e-4 * art["warm"]["N2"]["cold_distance"] * 10


# ---------------------------------------------------------------------------
# 3. a learned start buys nothing
# ---------------------------------------------------------------------------


def test_a_learned_start_is_closer_and_costs_more_classical_steps(art):
    w = art["warm"]["N2"]
    arms = w["arms"]
    tight = "2e-05"
    cold, learned = arms["cold"], arms["poseidon_settled"]
    null, noise = arms["null_expert_settled"], arms["matched_noise"]
    # it starts 2.5x closer than the cold march
    assert learned["start_distance"] < 0.45 * cold["start_distance"]
    # and needs MORE classical steps to the tight level
    assert learned["steps_to"][tight] > cold["steps_to"][tight]
    # the composition layer without the checkpoint starts further away and gets
    # there sooner: the contribution is not merely "closer is better"
    assert null["start_distance"] > learned["start_distance"]
    assert null["steps_to"][tight] < learned["steps_to"][tight]
    # and a start with only the right SIZE is the worst of the four
    assert noise["steps_to"][tight] >= max(a["steps_to"][tight]
                                           for a in (cold, learned, null))


# ---------------------------------------------------------------------------
# 4. defect correction at N=2
# ---------------------------------------------------------------------------


def _arms(art, rung):
    return art["dec"][f"N{rung}"]["arms"]


def test_the_checkpoint_removes_classical_calls_and_degrades_when_corrupted(art):
    d = art["dec"]["N2"]
    a = _arms(art, 2)
    cold = d["cold"]["phi_calls"]
    p, pc, z = a["P_a0.5"], a["Pc_a0.5"], a["Z_a0.5"]
    assert p["status"] == "converged"
    assert p["phi_calls"] < pc["phi_calls"] < z["phi_calls"] < cold
    # the null expert and the unshrunk checkpoint both stop on the stall rule
    assert a["Z_a0.5"]["fallback_from"] is not None
    assert a["P_a0"]["fallback_from"] is not None


def test_every_arm_returns_a_state_inside_the_certificate(art):
    d = art["dec"]["N2"]
    bound = d["theta"]["theta"] * d["r_stop"]
    for key, a in _arms(art, 2).items():
        if "_a" not in key:                       # tolerate a stale pre-module key
            continue
        assert a["final_error"] <= bound, (key, a["final_error"], bound)
    # and the checkpoint's arm is inside it by less than 10x: the bound is not
    # vacuous where it matters
    p = _arms(art, 2)["P_a0.5"]
    assert p["certificate_bound"] / p["final_error"] < 10.0


def test_the_stalled_checkpoint_fails_on_the_slowest_modes(art):
    """The mechanism the theory page marks [AI Inference], as measured."""
    a = _arms(art, 2)
    stalled = a["P_a0"]["pre_fallback_error_structure"]
    converging = a["P_a0.8"]["pre_fallback_error_structure"]
    assert stalled["lowest_modes_share"] > 0.5
    assert stalled["lowest_modes_share"] > 3.0 * converging["lowest_modes_share"]
    # the composition layer alone fails somewhere else entirely
    assert a["Z_a0"]["pre_fallback_error_structure"]["lowest_modes_share"] < 0.1


# ---------------------------------------------------------------------------
# 5. the cost ledger
# ---------------------------------------------------------------------------


def test_the_checkpoint_crosses_below_a_classical_step_and_coarsening_crosses_further(art):
    rows = {r["rung"]: r["ratio_to_F"] for r in art["cost"]["rows"]}
    assert rows["N2"]["P"] > 1.0 and rows["N6"]["P"] > 1.0
    assert rows["N12"]["P"] < 1.0
    # the classical competitor is cheaper than the checkpoint on every rung -- and
    # the gap NARROWS as the graph grows, 17x at two windows against 9x at twelve,
    # which is the one trend on this host that runs the learned side's way
    for rung in rows:
        assert rows[rung]["C"] < rows[rung]["P"]
    gaps = [rows[r]["P"] / rows[r]["C"] for r in ("N2", "N6", "N12")]
    assert gaps[0] > gaps[1] > gaps[2] > 1.0, gaps


# ---------------------------------------------------------------------------
# 6. the relative-energy certificate
# ---------------------------------------------------------------------------


def test_the_energy_certificate_is_valid_per_step_and_vacuous_over_a_transit(art):
    e = art["energy"]
    for row in e["rows"]:
        for arm in ("poseidon", "persistence"):
            assert row[arm]["eta"] >= row[arm]["true_error"], (row["step"], arm)
            assert row[arm]["eta_over_true"] < 10.0
    late = e["rows"][-1]
    # its floor on exact classical data is above the checkpoint's own error there
    assert late["classical"]["eta"] > late["poseidon"]["true_error"]
    # and one transit of the domain multiplies it by ten billion
    assert e["growth_one_transit_log10"] > 10.0


# ---------------------------------------------------------------------------
# 7. self-consistency
# ---------------------------------------------------------------------------


def test_self_consistency_proves_failure_and_cannot_see_a_null_replacement(art):
    s = art["selfcons"]
    assert s["lower_bound_holds"]
    assert 0.05 < s["lower_bound_tightness"] < 1.0
    assert s["delta_sc_poseidon"] > 10.0 * s["scheme_gap_2dt_vs_2xdt"]
    assert s["identity"]["delta_sc"] == 0.0
    assert s["identity"]["true_error_double_step"] > 1e-2


# ---------------------------------------------------------------------------
# 8. the price of adaptation
# ---------------------------------------------------------------------------


def test_one_fine_tuning_iteration_is_priced_and_affordable(art):
    f = art["finetune"]
    assert f["forward_backward_s"] > f["forward_s"]
    assert f["hours_for_2000_iterations"] < 2.0


# ---------------------------------------------------------------------------
# 9. out of sample: the gate, at six windows
# ---------------------------------------------------------------------------


def _cost_ratio(art, rung, operator):
    rows = {r["rung"]: r["ratio_to_F"] for r in art["cost"]["rows"]}
    # the detuned and corrupted checkpoints are the same network and are charged
    # at the checkpoint's own ratio
    return rows[f"N{rung}"]["P" if operator in ("Pc", "Pw") else operator]


def _total_cost(art, rung, key):
    a = _arms(art, rung)[key]
    return a["phi_calls"] + _cost_ratio(art, rung, a["operator"]) * a["psi_calls"]


def test_G2_and_G4_pass_at_six_windows(art):
    """Every arm inside the certificate, and the certified state is settled."""
    d = art["dec"]["N6"]
    bound = d["theta"]["theta"] * d["r_stop"]
    for key, a in _arms(art, 6).items():
        assert a["final_error"] <= bound, (key, a["final_error"], bound)
    p = _arms(art, 6)["P_a0.5"]
    assert p["certificate_bound"] / p["final_error"] < 10.0
    for key in ("P_a0.5", "C_a0.5"):
        m = _arms(art, 6)[key]["marched"]
        assert m["within"], (key, m)
        assert m["max_deviation"] <= m["two_theta_r"]


def test_G1_fails_out_of_sample(art):
    """In sample the checkpoint removed a fifth of the classical calls beyond both
    nulls; at six windows it does not."""
    d = art["dec"]["N6"]
    a = _arms(art, 6)
    nulls = min(d["cold"]["phi_calls"], a["Z_a0.5"]["phi_calls"])
    assert a["P_a0.5"]["phi_calls"] > 0.8 * nulls
    # and in sample it did pass, which is what makes this out-of-sample
    d2 = art["dec"]["N2"]
    a2 = _arms(art, 2)
    nulls2 = min(d2["cold"]["phi_calls"], a2["Z_a0.5"]["phi_calls"])
    assert a2["P_a0.5"]["phi_calls"] <= 0.8 * nulls2


def test_G5_fails_and_classical_coarsening_is_what_pays(art):
    d = art["dec"]["N6"]
    cold = d["cold"]["phi_calls"]
    p_cost = _total_cost(art, 6, "P_a0.5")
    c_cost = min(_total_cost(art, 6, k) for k in ("C_a0", "C_a0.5"))
    assert p_cost > cold / 1.5           # the checkpoint does not pay for itself
    assert c_cost < cold / 1.5           # the classical competitor does
    assert c_cost < p_cost


def test_G6_the_corrupted_checkpoint_does_not_cost_more_classical_calls(art):
    """Loud on failure has two halves here, and the second one fails: a checkpoint
    with 3% Gaussian noise on every weight needs FEWER classical calls than the
    clean one.

    **2026-09-12, Tier 50 (W205): this cell is the MINIMUM of twelve.**  The
    corruption was swept over twelve directions at this magnitude and the counts
    run 71 to 140 about a clean checkpoint at 99, so the 71 pinned here is an
    extreme value and not an effect size, and this clause was a one-draw
    Bernoulli trial.  The assertion is left as it is because it is true of this
    artifact and it is what Tier 48 measured; what does not survive is the
    reading that a corrupted checkpoint IS the better operator.  See
    [[corrupted-checkpoint-and-jacobian-fidelity]].
    """
    a = _arms(art, 6)
    assert a["Pw_a0.5"]["inside_certificate"]
    assert a["Pw_a0.5"]["phi_calls"] < a["P_a0.5"]["phi_calls"]


def test_the_slow_mode_mechanism_reproduces_out_of_sample(art):
    stalled = _arms(art, 6)["P_a0"]
    assert stalled["stop_reason"] == "stalled"
    es = stalled["pre_fallback_error_structure"]
    assert es["lowest_modes_share"] > 0.5
    assert es["window_mean_share"] > 0.2


# ---------------------------------------------------------------------------
# the pages quote the record
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.path.isfile(THEORY), reason="the page is not here")
def test_the_theory_page_quotes_the_arm_the_gate_turns_on(art):
    p = _arms(art, 2)["P_a0.5"]
    text = open(THEORY, encoding="utf-8").read()
    assert f"${p['phi_calls']}$" in text
    assert "pre-registered" in text.lower()
    # the gate's six clauses are all on the page
    for clause in ("G1", "G2", "G3", "G4", "G5", "G6"):
        assert clause in text


@pytest.mark.skipif(not os.path.isfile(KILLS), reason="the page is not here")
def test_the_kill_test_page_carries_a_verdict_for_every_candidate(art):
    text = open(KILLS, encoding="utf-8").read()
    for n in range(1, 15):
        assert f"**F{n}**" in text or f"F{n} ·" in text, n


# ---------------------------------------------------------------------------
# 9. the tables themselves, row by row
# ---------------------------------------------------------------------------


def _rung_of(heading: str) -> int | None:
    """``8.1`` is in sample, ``8.2.1`` is six windows, ``8.2.2`` is twelve."""
    return {"8.1": 2, "8.2.1": 6, "8.2.2": 12}.get(heading)


def _cell_num(cell: str) -> float:
    """A table cell down to its number: strip math, bolding and separators."""
    s = cell.replace("$", "").replace(r"\mathbf", "").replace("{", "")
    s = s.replace("}", "").replace(",", "").replace("**", "").strip()
    return float(s)


def _arm_key(name_cell: str, alpha_cell: str) -> str | None:
    """``| $P_c$, detuned | $0.5$ |`` -> ``Pc_a0.5``; the cold march has no key."""
    import re as _re
    if "cold march" in name_cell:
        return None
    m = _re.search(r"\$([A-Z])(?:_([a-z]))?\$", name_cell)
    if not m:
        return None
    op = m.group(1) + (m.group(2) or "")
    a = alpha_cell.replace("$", "").replace(r"\mathbf", "")
    a = a.replace("{", "").replace("}", "").strip()
    try:
        float(a)
    except ValueError:
        return None
    return f"{op}_a{a}"


def _measured_tables(text: str):
    """Yield ``(heading, rung, header, rows)`` for every arm table in section 8."""
    import re as _re
    heading, rung = None, None
    header, rows = None, []
    for line in text.splitlines():
        h = _re.match(r"^#+\s+(8(?:\.\d+)*)\s", line)
        if h:
            if header and rows and rung:
                yield heading, rung, header, rows
            heading, rung = h.group(1), _rung_of(h.group(1))
            header, rows = None, []
            continue
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        low = [c.lower() for c in cells]
        sep = set(line) <= set("|- :")
        if "classical calls" in low and "cheap calls" in low:
            header, rows = low, []
        elif sep and rows:
            # a separator after rows have started is the NEXT table's rule line:
            # close this one rather than swallowing the clause table below it.
            if header:
                yield heading, rung, header, rows
            header, rows = None, []
        elif header and not sep:
            if len(cells) != len(header):
                yield heading, rung, header, rows       # a narrower table began
                header, rows = None, []
            else:
                rows.append(cells)
    if header and rows and rung:
        yield heading, rung, header, rows


@pytest.mark.skipif(not os.path.isfile(THEORY), reason="the page is not here")
def test_every_row_of_every_measured_table_is_the_artifacts_row(art):
    """The pages' tables are the record, cell by cell -- not a transcription of it.

    Each arm table in section 8 is matched back to ``dec.N<rung>.arms``: the
    classical-call and cheap-call columns must be the artifact's counts, and the
    cold row must be the artifact's cold march.  A re-run that changes a count, or
    a hand edit to a table, fails here.
    """
    text = open(THEORY, encoding="utf-8").read()
    tables = list(_measured_tables(text))
    assert [t[0] for t in tables], "no measured table found in section 8"
    checked = 0
    for heading, rung, header, rows in tables:
        d = art["dec"][f"N{rung}"]
        i_phi, i_psi = header.index("classical calls"), header.index("cheap calls")
        for cells in rows:
            key = _arm_key(cells[0], cells[1])
            if key is None:
                if "cold march" in cells[0]:
                    assert _cell_num(cells[i_phi]) == d["cold"]["phi_calls"], (heading,)
                    assert _cell_num(cells[i_psi]) == 0
                    checked += 1
                continue
            assert key in d["arms"], (heading, key)
            arm = d["arms"][key]
            assert _cell_num(cells[i_phi]) == arm["phi_calls"], (heading, key, "phi")
            assert _cell_num(cells[i_psi]) == arm["psi_calls"], (heading, key, "psi")
            checked += 1
    assert checked >= 13, checked


# ---------------------------------------------------------------------------
# 10. the gate at twelve windows, and the clause the theory was meant to own
# ---------------------------------------------------------------------------


def test_the_gate_fails_out_of_sample_at_twelve_windows(art):
    """G1, G2, G5 and G6 fail at N=12; G4 passes. G3 therefore fails.

    Section 7 fixed every setting on N=2 and asked for all clauses at six AND
    twelve windows.  This pins the verdict so that a re-run which quietly turns a
    fail into a pass has to explain itself here first.
    """
    g = art["accounting"]["N12"]["gate"]
    assert g["G1_non_null"] is False
    assert g["G2_non_vacuous"] is False
    assert g["G4_marched"] is True
    assert g["G5_accounted"] is False
    assert g["G6_loud"] is False
    # and the in-sample rung is where G1 and G2 passed
    g2 = art["accounting"]["N2"]["gate"]
    assert g2["G1_non_null"] is True and g2["G2_non_vacuous"] is True
    assert g2["G5_accounted"] is False
    # G4 was not measured in sample: absent, not failed
    assert g2.get("G4_marched") is None


def test_the_certificate_constant_under_bounds_two_arms_at_twelve(art):
    """W208: Theta from one march does not bound the ratio along other approaches.

    The constant each arm needs is its returned error over its returned residual.
    At two and six windows the worst arm stays under the Theta the cold march
    reported; at twelve it does not, and the two arms that exceed it are the coarse
    classical solver and the CORRUPTED checkpoint -- so the bound fails hardest on
    the arm the loudness clause was written for.
    """
    worst, theta = {}, {}
    for rung in ("N2", "N6", "N12"):
        d = art["dec"][rung]
        theta[rung] = d["theta"]["theta"]
        worst[rung] = max(v["final_error"] / v["residual"]
                          for v in d["arms"].values() if "operator" in v)
    assert worst["N2"] < theta["N2"], (worst["N2"], theta["N2"])
    assert worst["N6"] < theta["N6"], (worst["N6"], theta["N6"])
    assert worst["N12"] > theta["N12"], (worst["N12"], theta["N12"])
    # the estimate falls with rung size while the requirement rises
    assert theta["N12"] < theta["N6"]
    assert worst["N12"] > worst["N6"]
    outside = sorted(art["accounting"]["N12"]["gate"]["G2_arms_outside"])
    assert outside == ["C_a0", "Pw_a0.5"], outside
    # the corrupted checkpoint misses by the larger margin
    arms = art["dec"]["N12"]["arms"]
    miss = {k: arms[k]["final_error"] / arms[k]["certificate_bound"] for k in outside}
    assert miss["Pw_a0.5"] > miss["C_a0"] > 1.0, miss
    assert miss["Pw_a0.5"] > 1.10, miss


def test_the_corrupted_checkpoint_is_the_cheaper_operator_on_both_out_of_sample_rungs(art):
    """G6's second half, measured twice, with the margin widening.

    In sample the classical calls ordered as the checkpoint's integrity did.  Out of
    sample the corrupted copy needs fewer classical calls than the clean one at six
    windows and at twelve, and at twelve it is also cheaper in total.

    **2026-09-12, Tier 50 (W205): one draw per rung, and the draw was the good
    one.**  Twelve directions at six windows give 71 to 140 about a clean 99, so
    the title's "is the cheaper operator" is not supported as a general claim --
    the typical direction saves about half what this cell does, and two of
    twelve are worse than not corrupting at all.  The assertions stay: they are
    true of this artifact.  See [[corrupted-checkpoint-and-jacobian-fidelity]].
    """
    for rung, clean, corrupt in (("N6", "P_a0.5", "Pw_a0.5"), ("N12", "P_a0.5", "Pw_a0.5")):
        rows = art["accounting"][rung]["arms"]
        assert rows[corrupt]["phi_calls"] < rows[clean]["phi_calls"], rung
    n6 = art["accounting"]["N6"]["arms"]
    n12 = art["accounting"]["N12"]["arms"]
    gap6 = n6["P_a0.5"]["phi_calls"] - n6["Pw_a0.5"]["phi_calls"]
    gap12 = n12["P_a0.5"]["phi_calls"] - n12["Pw_a0.5"]["phi_calls"]
    assert gap12 > gap6, (gap6, gap12)
    # at twelve the corrupted copy is cheaper in total cost too
    assert (n12["Pw_a0.5"]["cost_in_classical_calls"]
            < n12["P_a0.5"]["cost_in_classical_calls"])
    # and at six it is not -- it pays for the saved classical calls in cheap ones
    assert (n6["Pw_a0.5"]["cost_in_classical_calls"]
            > n6["P_a0.5"]["cost_in_classical_calls"])


def test_the_classical_competitor_improves_faster_than_the_checkpoint(art):
    """The entry condition of section 4, as three numbers per column.

    Coarsening's saving grows with the graph faster than the checkpoint's does, which
    is why the slot is real and the checkpoint does not enter it.
    """
    coarse, learned = [], []
    for rung in ("N2", "N6", "N12"):
        rows = art["accounting"][rung]["arms"]
        coarse.append(min(v["saving_ratio"] for v in rows.values() if v["operator"] == "C"))
        learned.append(rows["P_a0.5"]["saving_ratio"])
    best_coarse = []
    for rung in ("N2", "N6", "N12"):
        rows = art["accounting"][rung]["arms"]
        best_coarse.append(max(v["saving_ratio"] for v in rows.values() if v["operator"] == "C"))
    assert best_coarse[0] < best_coarse[1] < best_coarse[2], best_coarse
    assert learned[0] < learned[1] < learned[2], learned
    # the checkpoint only reaches break-even at twelve, where coarsening is at 7x
    assert learned[2] > 1.0 and best_coarse[2] > 6.5, (learned[2], best_coarse[2])
    assert best_coarse[2] / learned[2] > 5.0


def test_the_accounting_stage_and_the_module_price_an_arm_identically(art):
    """The page's cost column and the tested framework function are one formula.

    ``stage_account`` computes cost inline; ``break_even`` computes it from a
    result object.  If those ever diverge, the pages would quote a number no test
    exercises.
    """
    import sys as _sys
    _sys.path.insert(0, _ROOT)
    from atlas.defect_correction import DefectCorrectionResult, break_even

    for rung in ("N2", "N6", "N12"):
        for key, row in art["accounting"][rung]["arms"].items():
            res = DefectCorrectionResult(state=None, status="converged", residual=0.0,
                                         phi_calls=row["phi_calls"],
                                         psi_calls=row["psi_calls"])
            be = break_even(art["dec"][rung]["cold"]["phi_calls"], res, row["c_psi"])
            assert be["corrected_cost"] == pytest.approx(
                row["cost_in_classical_calls"], rel=1e-12), (rung, key)
            assert be["saving_ratio"] == pytest.approx(row["saving_ratio"], rel=1e-12)


def test_the_in_sample_arm_block_is_exactly_the_twelve_arms_of_the_grid(art):
    """No debris: the arms block is what the driver's grid writes, and nothing else.

    An earlier code path left four keys with no operator and no call counts in the
    N=2 block; they were not quoted anywhere, which is exactly why they could have
    survived into a commit.
    """
    arms = art["dec"]["N2"]["arms"]
    expected = {f"{op}_a{a}" for op in ("C", "Z", "P", "Pc") for a in ("0", "0.5", "0.8")}
    assert set(arms) == expected, sorted(set(arms) ^ expected)
    for key, v in arms.items():
        assert v.get("phi_calls") is not None, key
        assert "inside_certificate" in v, key
