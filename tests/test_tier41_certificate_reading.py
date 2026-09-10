"""Tier 41 -- what an `admit` rests on: the certificate reports it, and decides nothing new.

**W109**'s done-when is that the certificate reports its margin beside the
verdict, so *refused* and *refused by a quarter of one percent* can be told apart
without a re-probe.  `decision_margin` is that number, signed.

**W177** computed ``||Delta|| / ||S_i||`` by hand on the eight agent-sides of the
live checkpoint and found the three informative admits within 5% of deleting the
block.  `null_replacement_ratio` is that column, on the certificate itself.
``||S_i|| / beta`` is reported beside it as the scale-free form of
`visible_above`.

**Nothing here may move a verdict**, and the first assertions are the ones that
say so: the pre-Tier-41 truth table is restated here independently of
`composition.py`, and every verdict on a grid spanning every branch -- and every
verdict the census recorded, across its own beta_min sweep -- is reproduced.  A
rule that changed a verdict would have to rewrite these tests first, and that is
the point of writing them this way.
"""

from __future__ import annotations

import itertools
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.composition import SubstitutionCertificate  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W177 = os.path.join(_ROOT, "out", "w177", "w177.json")

DELTAS = (0.0, 0.05, 0.2, 0.39, 0.4, 0.8)
BETAS = (0.4, 1.0)
BETA_MINS = (None, 0.0, 0.01, 0.1, 0.35, 0.6)
#: chosen to tie with no margin the grid can produce
BLOCKS = (None, 0.07, 0.33, 2.0)


def _cert(delta, beta, beta_min, block, same=True, passive=None):
    return SubstitutionCertificate(
        agent_id="A", old_expert="old", new_expert="new",
        delta_norm=float(delta), beta=float(beta), beta_min=beta_min,
        same_port_list=same, passivity_preserved=passive, block_norm=block)


def _verdict_before_tier41(delta, beta, beta_min, block, same=True, passive=None):
    """The truth table as it stood at 39c5194, restated rather than imported."""
    if not same:
        return "refuse"
    if beta_min is None:
        return "admit-uncertified"
    margin = beta - beta_min
    if delta < margin:
        blind = None if block is None else bool(block < margin)
        return "admit-uncertified" if blind else "admit"
    if passive:
        return "admit-uncertified"
    return "refuse"


@pytest.fixture(scope="module")
def census():
    if not os.path.exists(W177):
        pytest.skip("run scripts/w177_campaign_census.py")
    with open(W177, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# 1. no verdict moves
# ---------------------------------------------------------------------------


def test_no_verdict_moves_on_a_grid_that_reaches_every_branch():
    seen = set()
    n = 0
    for d, b, bm, blk, same, pas in itertools.product(
            DELTAS, BETAS, BETA_MINS, BLOCKS, (True, False), (None, True)):
        c = _cert(d, b, bm, blk, same=same, passive=pas)
        want = _verdict_before_tier41(d, b, bm, blk, same=same, passive=pas)
        assert c.verdict.value == want, (d, b, bm, blk, same, pas)
        seen.add(want)
        n += 1
    assert n == 6 * 2 * 6 * 4 * 2 * 2
    #: and the grid is not trivially one-branch
    assert seen == {"admit", "admit-uncertified", "refuse"}


def test_W177_every_census_verdict_is_reproduced_across_its_own_sweep(census):
    rows = census["certificates"]["rows"]
    assert len(rows) == 8
    checked = 0
    for r in rows:
        c = _cert(r["delta_norm"], r["beta"], None, r["block_norm"],
                  same=r["same_port_list"])
        assert c.verdict.value == r["verdict"]
        for s in r["beta_min_sweep"]:
            c2 = _cert(r["delta_norm"], r["beta"], s["beta_min"], r["block_norm"],
                       same=r["same_port_list"])
            assert c2.verdict.value == s["verdict"], (r["seam"], r["agent"], s)
            checked += 1
    assert checked >= 8 * 6


# ---------------------------------------------------------------------------
# 2. W109 -- the margin, beside the verdict
# ---------------------------------------------------------------------------


def test_W109_the_margin_is_the_swap_measured_against_the_allowance():
    for d, b, bm, blk in itertools.product(DELTAS, BETAS, BETA_MINS, BLOCKS):
        c = _cert(d, b, bm, blk)
        if bm is None:
            assert c.decision_margin is None
            assert c.decision_margin_over_beta is None
            continue
        assert c.decision_margin == pytest.approx((b - bm) - d)
        assert (c.decision_margin > 0.0) == bool(c.passes)
        assert c.decision_margin_over_beta == pytest.approx(c.decision_margin / b)


def test_W109_refused_by_a_quarter_percent_now_reads_differently_from_saturated():
    """W109's own bracket: ||Delta||/beta in [1.00254, 1.38952], `refuse` at both."""
    near = _cert(1.00254, 1.0, 0.0, 2.0)
    far = _cert(1.38952, 1.0, 0.0, 2.0)
    assert near.verdict.value == far.verdict.value == "refuse"
    assert near.decision_margin_over_beta == pytest.approx(-0.00254)
    assert far.decision_margin_over_beta == pytest.approx(-0.38952)
    assert "is refused by" in near.message and "0.254% of beta" in near.message
    assert "38.952% of beta" in far.message
    #: a pass reports its margin the same way
    ok = _cert(0.2, 1.0, 0.1, 2.0)
    assert ok.verdict.value == "admit"
    assert "passes by 0.7, 70.000% of beta" in ok.message


def test_a_blind_pass_is_not_given_a_margin_it_does_not_have():
    """The blind branch says no replacement could have failed; a margin there
    would read as information the test does not carry."""
    c = _cert(0.01, 1.0, 0.1, 0.07)
    assert c.blind is True and c.verdict.value == "admit-uncertified"
    assert "W109" not in c.message


# ---------------------------------------------------------------------------
# 3. W76 / W177 -- the null-replacement ratio
# ---------------------------------------------------------------------------


def test_the_null_replacement_itself_reads_one():
    rng = np.random.default_rng(41)
    S_i = rng.normal(size=(16, 16))
    other = rng.normal(size=(16, 16))
    S_old = S_i + other
    S_new = other.copy()            # ignores its boundary data: S_i is removed
    c = _cert(np.linalg.norm(S_new - S_old, 2), np.linalg.svd(S_old, compute_uv=False)[-1],
              None, np.linalg.norm(S_i, 2))
    assert c.null_replacement_ratio == pytest.approx(1.0, rel=1e-12)
    #: and a faithful replacement reads near zero
    faithful = _cert(np.linalg.norm(1e-3 * S_i, 2), 1.0, None, np.linalg.norm(S_i, 2))
    assert faithful.null_replacement_ratio == pytest.approx(1e-3)
    assert _cert(0.2, 1.0, None, None).null_replacement_ratio is None


def test_W177_the_census_hand_column_is_now_the_certificate_s_own(census):
    ratios = []
    for r in census["certificates"]["rows"]:
        c = _cert(r["delta_norm"], r["beta"], None, r["block_norm"])
        assert c.null_replacement_ratio == pytest.approx(r["delta_over_block"], rel=1e-9)
        assert c.visible_above == pytest.approx(r["visible_above"], rel=1e-12, abs=1e-15)
        ratios.append(c.null_replacement_ratio)
    #: the checkpoint page's own figures, 0.947 and 1.013 as printed
    assert min(ratios) == pytest.approx(0.947, abs=6e-4)
    assert max(ratios) == pytest.approx(1.013, abs=6e-4)


# ---------------------------------------------------------------------------
# 4. ||S_i|| / beta, and what it is not
# ---------------------------------------------------------------------------


def test_block_over_beta_is_the_scale_free_form_of_visible_above():
    for d, b, bm, blk in itertools.product(DELTAS, BETAS, BETA_MINS, BLOCKS):
        c = _cert(d, b, bm, blk)
        if blk is None:
            assert c.block_over_beta is None
            continue
        assert c.visible_above == pytest.approx(b * (1.0 - c.block_over_beta))
        if bm is not None:
            assert c.blind == bool(c.block_over_beta < 1.0 - bm / b)


def test_the_reading_travels_in_the_emitted_record():
    d = _cert(0.2, 1.0, 0.1, 2.0).as_dict()
    assert d["verdict"] == "admit"
    for k in ("decision_margin", "decision_margin_over_beta",
              "null_replacement_ratio", "block_over_beta"):
        assert k in d
    assert d["decision_margin"] == pytest.approx(0.7)
    assert d["null_replacement_ratio"] == pytest.approx(0.1)
    assert d["block_over_beta"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 5. W137 -- the second graph, and the reading it needed
# ---------------------------------------------------------------------------
#
# W137 said the fluid's certificate at `wing_fsi`'s wetted seam is blind by five
# orders, because the fluid's block is 1.44e-5 of the assembled operator.  That
# number is the block's SHARE, ||S_F|| / ||S||.  The certificate reads
# ||S_F|| / beta, and beta is the assembled operator's SMALLEST singular value.
# `scripts/w137_certificate_visibility.py` measured both, on W136's own settled
# field, under all three effort conventions and both orientations.

W137J = os.path.join(_ROOT, "out", "w137", "w137.json")


@pytest.fixture(scope="module")
def visibility():
    if not os.path.exists(W137J):
        pytest.skip("run scripts/w137_certificate_visibility.py")
    with open(W137J, encoding="utf-8") as fh:
        return json.load(fh)


def test_W137_the_probe_is_the_object_W137_was_written_about(visibility):
    """Before anything is read off it: W136's stored block norms and its
    unoriented beta come back, and the assembled operator is exactly the SIGNED
    sum that W138's `effort_normal` declares."""
    assert set(visibility["modes"]) == {"diffusive", "conormal", "reaction"}
    for mode, m in visibility["modes"].items():
        st = m["w136_stored"]
        assert m["norm_fluid"] == pytest.approx(st["fluid"], rel=1e-6), mode
        assert m["norm_structure"] == pytest.approx(st["structure"], rel=1e-6), mode
        assert m["beta_unoriented"] == pytest.approx(st["beta"], rel=1e-6), mode
        assert m["share_fluid"] == pytest.approx(st["one_sided"], rel=1e-3), mode
        assert m["assembled_is_signed_sum"] is True
        assert m["reconstruction_rel_error"]["signed_sum"] < 1e-12
        assert m["op_beta"] == pytest.approx(m["beta_sigma_min"], rel=1e-12)


def test_W137_at_the_declared_effort_the_share_says_blind_and_the_certificate_sees(visibility):
    m = visibility["modes"]["reaction"]
    #: W137's own number, reproduced -- and the disparity its done-when names
    assert m["share_fluid"] < 2e-5
    assert m["structure_over_fluid"] > 1e4
    #: but the fluid's block is LARGER than beta, so no blind band exists
    assert m["beta_sigma_min"] < m["norm_fluid"]
    assert m["block_over_beta_fluid"] > 1.0
    null = [r for r in m["certificates"]
            if r["replacement"].startswith("null") and r["beta_min"] is not None]
    assert null and all(r["verdict"] == "refuse" and r["blind"] is False for r in null)
    under = [r for r in m["certificates"]
             if r["replacement"].startswith("under") and r["beta_min"] is not None]
    #: W137's done-when: an informative verdict on the soft side
    assert any(r["verdict"] == "admit" and r["blind"] is False for r in under)
    assert any(r["verdict"] == "refuse" for r in under)


def test_W137_its_reading_does_hold_under_the_two_efforts_the_graph_does_not_declare(visibility):
    """The control that says the check can return `blind` when blindness is there."""
    for mode in ("diffusive", "conormal"):
        m = visibility["modes"][mode]
        assert m["block_over_beta_fluid"] < 1.0, mode
        at0 = next(r for r in m["certificates"]
                   if r["replacement"].startswith("null") and r["beta_min"] == 0.0)
        assert at0["blind"] is True and at0["verdict"] == "admit-uncertified", mode


def test_W137_the_orientation_moves_beta_and_not_the_answer(visibility):
    m = visibility["modes"]["reaction"]
    #: W138's flipped lambda_min, arriving as the certificate's beta
    assert m["beta_sigma_min"] == pytest.approx(5.0538, abs=1e-3)
    assert m["beta_unoriented"] == pytest.approx(2.1658, abs=1e-3)
    #: visible before W138 as well -- the misreading was never the orientation
    assert m["block_over_beta_fluid_unoriented"] > 1.0


def test_W137_the_compiler_already_read_the_right_quantity(visibility):
    bs = visibility["compile"]["block_share_on_wet"]
    assert visibility["compile"]["graph_verdict"] == "admit"
    assert len(bs) == 1 and bs[0]["verdict"] == "admit"
    assert "the substitution certificate can see it" in bs[0]["message"]


def test_block_over_beta_carries_W137_s_reading_on_the_certificate_itself(visibility):
    m = visibility["modes"]["reaction"]
    c = _cert(m["norm_fluid"], m["beta_sigma_min"], 0.0, m["norm_fluid"])
    assert c.block_over_beta == pytest.approx(m["block_over_beta_fluid"])
    assert c.null_replacement_ratio == pytest.approx(1.0)
    assert c.verdict.value == "refuse" and c.blind is False
