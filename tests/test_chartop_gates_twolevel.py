"""Phase 4's registration (`atlas/chartop/gates_twolevel.py`) against the rework page's section
18 and the registrations it reads from: the shared bars are read, not copied; the new bars,
arms and predictions are the page's; the judges read what they say; the fresh test split
overlaps no other.  The owner accepted it as drafted on 2026-11-10; it is not yet pinned."""
from __future__ import annotations

import os
import re

import numpy as np
import pytest

from atlas.chartop import gates as G, gates_forms as GF, gates_prior as GP, gates_twolevel as GT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "wiki", "concepts", "Atlas 0.1", "atlas-0.1-proposal", "expert-architecture",
                    "chart-operator-stage-0-rework.md")


def _section18() -> str:
    text = open(PAGE, encoding="utf-8").read()
    start = text.index("## 18. Phase 4")
    return text[start:text.index("\n## ", start + 1)]


def test_it_is_accepted_with_the_owners_settled_choices():
    assert "2026-11-10" in GT.REGISTERED and "Accept as drafted (Recommended)" in GT.REGISTERED
    assert "not yet pinned" in GT.REGISTERED
    assert set(GT.SETTLED) == {"the corner treatment", "the NC bar", "the budget"}
    assert "Adopt B, n/8 (Recommended)" in GT.SETTLED["the corner treatment"]
    assert "As proposed (Recommended)" in GT.SETTLED["the NC bar"]
    assert "37,500 steps (Recommended)" in GT.SETTLED["the budget"]


def test_the_shared_bars_are_read_not_copied():
    assert GT.E4_ENERGY is GF.E4_ENERGY and GT.E4_MAX_OF_RANGE is GF.E4_MAX_OF_RANGE
    assert GT.HELD_OUT is GP.HELD_OUT and GT.REUSED is GP.REUSED
    assert GT.judge_e4_both_frames is GF.judge_e4_both_frames and GT.judge_k is GF.judge_k
    prior = {g.key: g for g in GP.GATES}
    for g in GT.GATES:
        if g.key in ("K", "D", "L", "E3", "E4", "SE1", "SE2", "DUAL", "KT"):
            assert g.bar == prior[g.key].bar and g.title == prior[g.key].title, g.key
    assert [g.key for g in GT.GATES] == ["K", "D", "L", "E3", "E4", "CX", "NC", "SE1", "SE2", "DUAL", "KT", "PR"]
    assert [g.key for g in GT.REPORTED] == ["E1", "SE3", "S11", "LEDGER"]


def test_the_new_numbers_are_the_pages():
    sec = _section18()
    assert (GT.N_C, GT.CORNERS, GT.TL0_CURVE, GT.STEPS, GT.SEEDS) == (16, 8, (4, 8, 16, 32), 37_500, (0, 1))
    assert (GT.CX_BAR, GT.NC_MEDIAN, GT.NC_P10, GT.NC_ROUNDING, GT.H2R_PRICE, GT.LR) == (1e-10, 0.3, 0.0, 1e-9,
                                                                                          0.306, 3e-4)
    for page in ("10^{-10}", "\\rho\\ge0.3", "10th percentile $\\ge0$", "10^{-9}", "0.306", "37,500",
                 "6,000,000–6,002,399", "3\\times10^{-4}", "e17f881b", "n_c=16", "n/8", "$n_c=4$, 8, 16 and 32"):
        assert page in sec, page
    assert GT.INIT_SHA256.startswith("e17f881b") and GT.INIT_SHA256.endswith("a35d")


def _plain(s: str) -> str:
    s = re.sub(r"\*\*\[AI Inference\]:\*\*|\[AI Inference\]", "", s)
    s = s.replace("\\rho", "rho").replace("§", "section").replace("–", "-")
    return re.sub(r"[\s$*]", "", s)


def test_the_predictions_are_the_pages_word_for_word():
    """Section 18's numbered predictions, once the notation is set aside."""
    sec = _section18()
    start = sec.index("**Predictions**")
    block = sec[start:sec.index("\n\n**", start + 1)]
    items = re.findall(r"^(\d)\. (.+)$", block, re.M)
    assert [i for i, _t in items] == ["1", "2", "3", "4", "5"]
    for (num, page), mod in zip(items, GT.PREDICTIONS):
        assert mod.startswith(num + ". ")
        assert _plain(mod[3:]) == _plain(page), num


def test_the_init_weights_are_ss_recorded_ones():
    path = os.path.join(ROOT, "out", "chartop", "runs", "forms", "weights-sha256.txt")
    rec = {line.split()[1].split("/")[0]: line.split()[0] for line in open(path, encoding="utf-8") if line.strip()}
    assert rec[GT.INIT_RUN] == GT.INIT_SHA256 and GT.HEAD_INIT == "loaded"


def test_the_fresh_test_split_overlaps_no_other():
    lo, hi = GT.TEST_SEEDS
    assert (lo, hi - lo) == (6_000_000, G.N_TEST)
    others = [(b, b + 1_000_000) for b in G.SEED_BASE.values()] + [(GP.TEST_SEEDS[0], GP.TEST_SEEDS[1])]
    assert all(hi <= a or lo >= b for a, b in others)
    assert "6,000,000 to 6,002,399" in {g.key: g for g in GT.GATES}["NC"].bar


def test_the_arm_config_is_a_train_arm():
    from atlas.chartop import train
    cfg = GT.arm_config()
    a = train.Arm("TL", **{k: v for k, v in cfg.items() if k != "m"})
    assert (a.coarse, a.corners, a.objective, a.steps, a.lr, a.head_init) == (16, 8, "relative", 37_500, 3e-4,
                                                                             "loaded")
    assert a.form == "split" and a.bubble == "corner" and a.strips and not a.strips_in_training
    with pytest.raises(ValueError):
        GT.arm_config("TL0")


def test_judge_cx():
    assert GT.judge_cx([1e-15, 1e-11], [None, 1e-12])["met"] is True
    assert GT.judge_cx([1e-15, 2e-10], [None])["met"] is False
    assert GT.judge_cx([1e-15], [2e-10])["met"] is False
    assert GT.judge_cx([], [])["met"] is False


def _e4(v=0.01):
    return {(b, p, f): v for b, p in GT.HELD_OUT for f in ("registered", "lowered")}


@pytest.mark.parametrize("rho, held", [([0.3, 0.4, 0.5, 0.31, np.nan], True), ([0.29, 0.3, 0.28], False),
                                       ([0.5, 0.6, -0.4, -0.3, 0.8, 0.9, 0.5, 0.6, 0.7, 0.8], False),
                                       ([0.5, 0.6, -0.4, 0.7, 0.8, 0.9, 0.5, 0.6, 0.7, 0.8], True)])
def test_judge_nc_reads_the_median_and_the_tenth_percentile(rho, held):
    v = GT.judge_nc(rho, _e4(), _e4())
    assert v["met"] is held


def test_judge_nc_reads_every_held_out_problem_in_both_frames():
    tl, tl0 = _e4(0.01), _e4(0.01)
    k = next(iter(tl))
    tl[k] = 0.01 * (1 + 2e-9)
    v = GT.judge_nc([0.5, 0.6], tl, tl0)
    assert v["met"] is False and v["worse_than_TL0"] == [k]
    tl[k] = 0.01 * (1 + 0.5e-9)
    assert GT.judge_nc([0.5, 0.6], tl, tl0)["met"] is True
    tl.pop(k)
    with pytest.raises(ValueError):
        GT.judge_nc([0.5], tl, tl0)


def test_the_text_names_every_gate():
    t = GT.text()
    for g in GT.GATES + GT.REPORTED:
        assert re.search(rf"^\s+{re.escape(g.key)}\s", t, re.M), g.key
