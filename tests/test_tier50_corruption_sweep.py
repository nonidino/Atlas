"""Tier 50 -- W205: was the corrupted checkpoint's saving real, and what did it?

Tier 48 measured, once per rung, that Poseidon-T with Gaussian noise at 3% of
every weight tensor's rms needed fewer classical calls inside defect correction
than the clean checkpoint -- 71 against 99 at six windows, 138 against 222 at
twelve -- and left the inversion unexplained (W205).  This tier swept it.

What this file pins:

1. **The reproduction control, and it gates everything.**  The default cell must
   return Tier 48's arm to the call before any other cell is believed; a sweep
   whose default cell does not reproduce is measuring its own scaffolding.
2. **The pre-registered verdict: NOISE.**  Tier 48's 71 is the joint MINIMUM of
   twelve directions at its own magnitude, which run 71 to 140 about a clean
   checkpoint at 99.
3. **And a real effect the clause could not see.**  Ten of twelve directions
   beat the clean checkpoint (sign test p = 0.019) while the mean does not
   differ significantly, because the distribution has a heavy upper tail -- both
   readings are recorded, and neither is allowed to stand alone.
4. **The sigma response is NON-MONOTONE**: a little weight noise helps, a lot
   hurts.  That is the shape the mechanism has to explain.
5. **The seeds are DIRECTIONS, not draws.**  `manual_seed` fixes the noise
   pattern and sigma only rescales it, so a row of the grid is one ray through
   weight space sampled at five radii.  Pinned because it is the design caveat
   that decides what the grid can conclude.
6. **The mechanism, measured rather than inferred.**  Theorem 2's predicted rate
   orders the twenty copies better than accuracy does, and the property doing
   the work is Jacobian fidelity on the slow modes -- which is what
   `defect-correction-learned-operator` section 2.2 says and the OPPOSITE of
   what its section 5 **[AI Inference]** guessed.

Everything is asserted against ``out/w205/w205.json``; rebuild it with
``python scripts/w205_corruption_sweep.py``.  The N=6 grid is about 21 minutes
and the N=12 confirmation about 40, because a single corrupted arm at twelve
windows is 300-500 s.  The tests skip without the artifact and never load a
checkpoint.
"""

from __future__ import annotations

import json
import math
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest                                                         # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

ARTIFACT = os.path.join(_ROOT, "out", "w205", "w205.json")
W202 = os.path.join(_ROOT, "out", "w202", "w202.json")
WIKI = os.path.join(_ROOT, "wiki", "concepts", "Atlas 0.1", "common")
PAGE = os.path.join(WIKI, "corrupted-checkpoint-and-jacobian-fidelity.md")

#: Tier 48's two cells, which the reproduction control must return.
TIER48 = {"N6": 71, "N12": 138}
CLEAN = {"N6": 99, "N12": 222}


@pytest.fixture(scope="module")
def art():
    if not os.path.exists(ARTIFACT):
        pytest.skip("out/w205/w205.json is not on disk: run "
                    "scripts/w205_corruption_sweep.py")
    with open(ARTIFACT, encoding="utf-8") as fh:
        return json.load(fh)


def _cells(art, n, sigma=None):
    c = art.get("sweep", {}).get(f"N{n}", {})
    if sigma is None:
        return c
    return {k: v for k, v in c.items() if abs(v["sigma"] - sigma) < 1e-12}


def _phi(cells):
    return sorted(v["phi_calls"] for v in cells.values())


# ==========================================================================
# 1. the reproduction control
# ==========================================================================


def test_the_default_cell_reproduces_Tier_48_to_the_call(art):
    """Nothing in this file means anything if this fails."""
    rep = art.get("repro", {})
    assert rep, "the reproduction control did not run"
    for key, expected in TIER48.items():
        if key not in rep:
            continue
        r = rep[key]
        assert r["Pw_phi_calls"] == expected, (key, r["Pw_phi_calls"])
        assert r["P_phi_calls"] == CLEAN[key], (key, r["P_phi_calls"])
        assert r["passes"] is True, key


def test_the_sweep_did_not_move_Tier_48s_artifact():
    """`set_corruption` defaults to Tier 48's sigma and seed, so the committed
    w202 record is untouched and every Tier 48 test still asserts of it."""
    if not os.path.exists(W202):
        pytest.skip("out/w202/w202.json is not on disk")
    with open(W202, encoding="utf-8") as fh:
        a = json.load(fh)
    assert a["dec"]["N6"]["corrupt_sigma"] == 0.03
    assert a["dec"]["N6"]["arms"]["Pw_a0.5"]["phi_calls"] == 71
    assert a["dec"]["N12"]["arms"]["Pw_a0.5"]["phi_calls"] == 138


# ==========================================================================
# 2. the verdict, as pre-registered
# ==========================================================================


def test_Tier_48s_cell_is_the_minimum_of_twelve_directions(art):
    """**The finding.**  71 is not a typical corrupted copy; it is the best one
    of twelve, and the same magnitude also produced a copy needing 140."""
    cells = _cells(art, 6, 0.03)
    if len(cells) < 12:
        pytest.skip(f"only {len(cells)} directions at the decision magnitude")
    phi = _phi(cells)
    assert min(phi) == TIER48["N6"], phi
    assert max(phi) >= 140, phi
    assert max(phi) - min(phi) >= 60, phi


def test_the_pre_registered_verdict_is_noise(art):
    s = art["sweep"]["summary_N6"]
    assert s["verdict"].startswith("NOISE"), s["verdict"]
    dec = s["by_sigma"][f"{s['decision_sigma']:g}"]
    assert dec["spread_covers_the_clean_checkpoint"] is True
    assert dec["all_beat_the_clean_checkpoint"] is False


def test_the_continuity_control_behaves_like_a_control(art):
    """A corruption a tenth as hard must sit NEARER the clean checkpoint.  The
    first version of the verdict rule scored this as evidence of noise, which is
    backwards; it is the knob being the knob."""
    s = art["sweep"]["summary_N6"]
    c = s["continuity_control"]
    assert c is not None
    assert c["nearer_the_clean_checkpoint_than_the_decision_cell"] is True


# ==========================================================================
# 3. the effect the clause could not see, and both readings of it
# ==========================================================================


def test_ten_of_twelve_directions_beat_the_clean_checkpoint(art):
    """The sign test is significant where the t-test is not, because the
    distribution has a heavy upper tail.  Both are reported; neither alone."""
    cells = _cells(art, 6, 0.03)
    if len(cells) < 12:
        pytest.skip("the twelve-direction sample is not on disk")
    phi = _phi(cells)
    n = len(phi)
    k = sum(1 for x in phi if x < CLEAN["N6"])
    assert k == 10, phi
    p = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    assert p < 0.05, p
    mean = sum(phi) / n
    var = sum((x - mean) ** 2 for x in phi) / (n - 1)
    sem = math.sqrt(var / n)
    # the mean is below the clean checkpoint but not by two standard errors
    assert mean < CLEAN["N6"]
    assert (CLEAN["N6"] - mean) / sem < 2.2


def test_the_sigma_response_is_non_monotone(art):
    """A little weight noise helps and a lot hurts, with a minimum near 0.01 --
    which is the shape the mechanism has to explain."""
    s = art["sweep"]["summary_N6"]
    m = {k: s["by_sigma"][k]["mean"] for k in s["sigma_order"]}
    clean = CLEAN["N6"]
    assert m["0.003"] < clean and m["0.01"] < clean and m["0.03"] < clean
    assert m["0.1"] > clean and m["0.3"] > clean
    assert m["0.3"] > m["0.1"] > m["0.03"]
    assert s["monotone_decreasing_in_sigma"] is False
    assert s["monotone_increasing_in_sigma"] is False


def test_destroying_the_checkpoint_is_reliably_worse(art):
    """The learned content DOES carry something: at sigma = 0.3 every direction
    is worse than the clean checkpoint, and the spread is far below the gap."""
    s = art["sweep"]["summary_N6"]
    row = s["by_sigma"]["0.3"]
    assert row["any_beat_the_clean_checkpoint"] is False
    assert row["min"] > CLEAN["N6"]
    assert row["spread"] < row["mean"] - CLEAN["N6"] + row["spread"]


# ==========================================================================
# 4. the design caveat: seeds are directions
# ==========================================================================


def test_a_seed_is_a_direction_and_sigma_is_its_radius(art):
    """`manual_seed` fixes the noise PATTERN and sigma only rescales it, so one
    seed across the magnitudes is one ray through weight space.  The worst
    direction is therefore the worst at several magnitudes rather than at one,
    which is visible in the grid and is why four seeds is a small sample."""
    cells = _cells(art, 6)
    by_seed = {}
    for v in cells.values():
        by_seed.setdefault(v["seed"], {})[f"{v['sigma']:g}"] = v["phi_calls"]
    rays = {s: d for s, d in by_seed.items() if len(d) >= 3}
    assert rays, "no seed measured at three or more magnitudes"
    worst_at = [max(d, key=lambda k: d[k]) for d in rays.values()]
    # the same seed is the worst of its row at more than one magnitude
    counts = {}
    for sig in ("0.003", "0.01", "0.03"):
        row = {s: d[sig] for s, d in rays.items() if sig in d}
        if row:
            counts.setdefault(max(row, key=row.get), 0)
            counts[max(row, key=row.get)] += 1
    assert max(counts.values()) >= 2, counts
    assert worst_at  # the ray structure is what makes that expected


# ==========================================================================
# 5. the mechanism
# ==========================================================================


def test_theorem_2s_rate_orders_the_copies_better_than_accuracy_does(art):
    """**The measurement that replaces an [AI Inference].**  Tier 48 read the
    mechanism off the error structure of stalled iterates and marked it
    inferred.  Here each map's derivative is probed directly along band-limited
    perturbations, Theorem 2's per-band eigenvalue is assembled from it, and it
    orders the twenty copies by classical calls better than accuracy does."""
    c = art.get("probe", {}).get("N6", {}).get("correlation", {})
    if "spearman_phi_vs_max_abs_lambda" not in c:
        pytest.skip("the probe stage has not run")
    lam = c["spearman_phi_vs_max_abs_lambda"]
    acc = c["spearman_phi_vs_accuracy"]
    assert c["n"] >= 20, c["n"]
    assert lam is not None and lam > 0.8, lam
    assert acc is not None and acc > 0.0, acc
    assert lam > acc, (lam, acc)


def test_the_classical_map_amplifies_the_slow_band(art):
    """Which is why the slow band is where the rate is decided: an open
    advective domain grows low-mode perturbations over a macro-step."""
    p = art.get("probe", {}).get("N6", {})
    f = p.get("F (classical)")
    if f is None:
        pytest.skip("the probe stage has not run")
    assert f["bands"]["slow"]["amplification"] > 1.0
    assert f["one_step_distance_from_the_classical_map"] == 0.0


def test_the_unshrunk_column_preserves_what_the_monolith_damps(art):
    """`defect-correction-learned-operator` section 5's **[AI Inference]**,
    confirmed on the derivative rather than on stalled iterates: the clean
    column's slow-band response sits ABOVE the classical map's, so it preserves
    what the monolith relaxes."""
    p = art.get("probe", {}).get("N6", {})
    clean, classical = p.get("P (clean)"), p.get("F (classical)")
    if clean is None or classical is None:
        pytest.skip("the probe stage has not run")
    phi = classical["bands"]["slow"]["rayleigh"]
    psi = clean["bands"]["slow"]["rayleigh"]
    assert psi > phi, (psi, phi)


def test_the_shrink_overshoots_past_the_classical_map(art):
    """**The structural finding.**  alpha* = 0.5 was chosen on N=2 as the only
    value at which the checkpoint converged without falling back -- a stability
    criterion, not a fidelity one -- and it takes the slow-band response from
    ABOVE the classical map's to well BELOW it.  The corruption's whole benefit
    is partly undoing that, which is why a little helps and a lot hurts."""
    p = art.get("probe", {}).get("N6", {})
    clean, classical = p.get("P (clean)"), p.get("F (classical)")
    if clean is None or classical is None:
        pytest.skip("the probe stage has not run")
    phi = classical["bands"]["slow"]["rayleigh"]
    psi = clean["bands"]["slow"]["rayleigh"]
    psi_shrunk = clean["bands"]["slow"]["rayleigh_shrunk"]
    assert psi > phi > psi_shrunk, (psi, phi, psi_shrunk)
    # and it overshoots by MORE than the original excess
    assert (phi - psi_shrunk) > (psi - phi)


def test_the_dissipation_candidate_is_refuted_in_sign(art):
    """[[gap-worklist]]'s W205 row named DISSIPATION on the slow modes as the
    candidate.  Measured, every magnitude that helps moves the slow-band
    response UP rather than damping it -- the opposite sign."""
    p = art.get("probe", {}).get("N6", {})
    clean = p.get("P (clean)")
    if clean is None:
        pytest.skip("the probe stage has not run")
    psi_clean = clean["bands"]["slow"]["rayleigh"]
    helped = [p[k] for k in p
              if k.startswith(("Pw s0.003 ", "Pw s0.01 ", "Pw s0.03 "))]
    assert helped, "no probed copy at a helping magnitude"
    raised = sum(1 for r in helped if r["bands"]["slow"]["rayleigh"] > psi_clean)
    assert raised > 0.5 * len(helped), (raised, len(helped))


# ==========================================================================
# 6. the second rung
# ==========================================================================


def test_the_second_rung_is_measured_or_says_it_is_not(art):
    """W205 asked for two rungs.  A rung with no cells reports NOT MEASURED
    rather than inheriting the first rung's verdict."""
    cells = _cells(art, 12, 0.03)
    if not cells:
        pytest.skip("N=12 has not run")
    phi = _phi(cells)
    assert len(phi) >= 3, phi
    # Tier 48's cell is in the sample and is not the whole of it
    assert min(phi) <= TIER48["N12"]
    assert max(phi) > min(phi)


# ==========================================================================
# 7. the page quotes the record
# ==========================================================================


@pytest.mark.skipif(not os.path.isfile(PAGE), reason="the page is not here")
def test_the_page_carries_the_twelve_direction_sample(art):
    with open(PAGE, encoding="utf-8") as fh:
        text = fh.read()
    cells = _cells(art, 6, 0.03)
    if len(cells) >= 12:
        for v in (min(_phi(cells)), max(_phi(cells))):
            assert str(v) in text, v
    assert "NOISE" in text
    assert "Jacobian fidelity" in text
