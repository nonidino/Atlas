"""The registration after Stage 0 (`atlas.chartop.gates_forms`).  The module is the one
registered: its hash, and that of Stage 0's `gates` whose bars it reads, are those pinned
in ``out/chartop/registered-forms.txt``.  The bars and arms stated in
[[chart-operator-kernel-exact-forms]] section 8, which the owner accepted on 2026-10-19,
agree with the ones in code ("pre-registered criteria drift in code", and always in the
permissive direction, unless something checks).  Stage 0's own registration is checked by
`test_chartop_gates.py`."""
from __future__ import annotations

import hashlib
import json
import os
import re

import numpy as np
import pytest

from atlas.chartop import forms, gates as G, gates_forms as F
from atlas.chartop import model as md

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "wiki", "concepts", "Atlas 0.1", "atlas-0.1-proposal",
                    "expert-architecture", "chart-operator-kernel-exact-forms.md")
PLAN = os.path.join(ROOT, "out", "chartop", "plan.json")
REG = os.path.join(ROOT, "out", "chartop", "registered-forms.txt")


def _hash(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def test_the_registered_modules_are_unchanged():
    assert os.path.exists(REG), "out/chartop/registered-forms.txt is missing: register before training"
    text = open(REG, encoding="ascii").read()
    for name in ("gates_forms.py", "gates.py"):
        m = re.search(r"atlas/chartop/" + re.escape(name) + r"\s+sha256 ([0-9a-f]{64})", text)
        assert m, f"the registration names no hash for {name}"
        assert m.group(1) == _hash(os.path.join(ROOT, "atlas", "chartop", name)), name
    assert F.REGISTERED in text and F.text() in text


def _section(number: str) -> str:
    text = open(PAGE, encoding="utf-8").read()
    start = text.index(f"\n## {number}. ")
    end = text.index("\n## ", start + 1)
    return text[start:end]


def _subsection(number: str) -> str:
    sec = _section(number.split(".")[0])
    start = sec.index(f"\n### {number} ")
    end = sec.find("\n### ", start + 1)
    return sec[start:] if end < 0 else sec[start:end]


def _rows(block: str) -> dict:
    """First cell -> the row's cells, for every table row in ``block``."""
    out = {}
    for line in block.splitlines():
        if line.startswith("|") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            out[cells[0]] = cells
    return out


def _row(block: str, key: str) -> list:
    hits = [cells for first, cells in _rows(block).items()
            if first == key or first.startswith(key + " ")]
    assert len(hits) == 1, f"{len(hits)} rows for {key!r}"
    return hits[0]


def _numbers(cell: str):
    """As `test_chartop_gates._numbers`: decimals, powers of ten, integers of 2+ digits."""
    dec = {float(x) for x in re.findall(r"(?<![\d.])\d+\.\d+(?![\d.])", cell)}
    pw = {10.0 ** -int(k) for k in re.findall(r"10\^\{-(\d+)\}", cell)}
    ints = {int(x) for x in re.findall(r"(?<![\d.\w{-])(\d{2,})(?![\d.\w^-])", cell)}
    return dec, pw, ints


def _count(cell: str) -> int:
    m = re.findall(r"(?<![\d,])(\d{1,3}(?:,\d{3})+)(?![\d,])", cell)
    assert len(m) == 1, f"one comma-grouped count expected in {cell!r}"
    return int(m[0].replace(",", ""))


def test_the_wiki_states_the_bars():
    t = _subsection("8.2")
    assert _numbers(_row(t, "**K**")[1]) == (set(), {F.K_E_K, F.K_E_K_INF}, set())
    assert _numbers(_row(t, "**D**")[1]) == (set(), {F.D_RATIO, F.D_FLOOR, F.D_MAX_OF_RANGE},
                                            {int(-F.DATUM_SHIFT_K)})
    assert F.D_FLOOR == F.D_MAX_OF_RANGE == 1e-9 and F.D_RATIO == 1e-8
    assert _numbers(_row(t, "**L**")[1]) == (set(), set(), {F.L_TEST_PIECES})
    assert _numbers(_row(t, "**E1**")[1]) == ({F.E1_MODES_MEDIAN, F.E1_MODES_P90}, set(),
                                             {F.N_LEARNED, F.N_COLUMNS})
    assert "$e_p$'s bars unchanged" in _row(t, "**E1**")[1]
    assert _numbers(_row(t, "**E4**")[1]) == ({F.E4_ENERGY, F.E4_MAX_OF_RANGE}, set(),
                                             {int(-F.DATUM_SHIFT_K)})
    assert "both frames" in _row(t, "**E4**")[1]
    assert _numbers(_row(t, "**E5**")[1])[2] >= {n for n, _b in F.E5_PRICED}
    assert "batches " + "/".join(str(b) for _n, b in F.E5_PRICED) in _row(t, "**E5**")[1]
    assert "Stage 0's bars, unchanged" in _row(t, "E3, SE1–SE3, DUAL, KT")[1]


def test_the_wiki_states_the_arms():
    t = _subsection("8.1")
    for key, arm in (("**S**", "S"), ("S′", "Sp"), ("**W**", "W")):
        cells = _row(t, key)
        assert _count(cells[3]) == F.ARMS[arm]["steps"]
        assert int(cells[4]) == len(F.ARMS[arm]["seeds"])
        if F.ARMS[arm]["strips"]:
            assert cells[2] == "as S" or ("corner bubble" in cells[2] and "strips" in cells[2])
        else:
            assert "square bubble, no strips" in cells[2]
    assert F.ARMS["S"]["seeds"] == (0,) and F.ARMS["Sp"]["seeds"] == F.ARMS["W"]["seeds"] == (0, 1)
    assert "S uses seed 0; S′ and W use 0 and 1" in t


@pytest.mark.parametrize("arm, key", [("S", "**S**"), ("W", "**W**")])
def test_the_parameter_counts_are_the_experts(arm, key):
    a, c = F.ARMS[arm], F.COMMON
    e = forms.KernelExpert(c["form"], c["backbone"], c["size"], c["kind"], c["m"],
                           p_size=c["p_size"], bubble=a["bubble"], strips=a["strips"])
    assert md.parameters(e) == _count(_row(_subsection("8.1"), key)[1])
    assert e.strips == a["strips"] and e.bubble == a["bubble"]


def test_the_budgets_are_stage_0s():
    plan = {a["name"]: a for a in json.load(open(PLAN, encoding="utf-8"))["arms"]}
    assert F.ARMS["S"]["steps"] == plan["B"]["steps"] == 150_000
    assert F.ARMS["Sp"]["steps"] == F.ARMS["W"]["steps"] == plan["Bp"]["steps"] == 37_500


def test_the_wiki_states_the_comparison_and_the_cost():
    c = _subsection("8.3")
    assert _numbers(c.split("\n\n")[1])[0] == {F.WIN, F.LOSE}
    assert (F.WIN, F.LOSE) == (G.WIN, G.LOSE) == (0.8, 1.25)
    assert "W over S′" in _row(c, "**CW**")[2]
    assert "verdict stands only if both pairings give it" in c
    cost = _subsection("8.5")
    assert f"\\${F.CEILING_USD:g} for the whole registration" in cost
    assert f"each box at most \\${F.BOX_MAX_USD_PER_HOUR:.2f} an hour" in cost


def test_the_learned_columns_are_the_forms():
    assert F.N_LEARNED == len(forms.learned_columns(F.KIND, F.M_PORTS)) == 36
    assert tuple(forms.STRIP_J) == F.STRIP_SINES
    assert F.N_COLUMNS == 64


def test_the_shared_bars_are_stage_0s():
    s0 = {g.key: g.bar for g in G.GATES}
    mine = {g.key: g.bar for g in F.GATES}
    for k in ("E3", "SE1", "SE2", "SE3", "DUAL", "KT"):
        assert mine[k] == s0[k]
    assert set(mine) == {"K", "D", "L", "E1", "E3", "E4", "E5", "SE1", "SE2", "SE3", "DUAL",
                         "KT", "S11"}
    assert (F.E1_MODES_MEDIAN, F.E1_MODES_P90, F.E1_PARTICULAR_MEDIAN, F.E1_PARTICULAR_P90) == \
        (0.01, 0.03, 0.02, 0.05)
    assert F.SEED_BASE is G.SEED_BASE and F.MU_BINS is G.MU_BINS


def test_e1s_denominator_is_the_learned_columns_own_energy():
    """The permissive alternative, dividing by tr Lambda, must not pass where this fails."""
    cols = forms.learned_columns(F.KIND, F.M_PORTS)
    other = [k for k in range(F.N_COLUMNS) if k not in cols]
    lam = np.ones(F.N_COLUMNS)
    lam[other] = 4.0                       # the strips carry most of the energy, as they do
    err = np.zeros(F.N_COLUMNS)
    err[list(cols)] = 1.5e-4               # e_H,L = 0.0122 > 0.01
    e = F.e_h_learned(err, lam, cols)
    assert e == pytest.approx(np.sqrt(1.5e-4))
    assert e > F.E1_MODES_MEDIAN
    assert np.sqrt(err.sum() / lam.sum()) < F.E1_MODES_MEDIAN     # tr Lambda would pass it
    err[other] = 1.0                       # the strip columns' error does not enter
    assert F.e_h_learned(err, lam, cols) == e
    with pytest.raises(ValueError):
        F.e_h_learned(err[:36], lam[:36], cols)
    with pytest.raises(ValueError):
        F.e_h_learned(err, lam, list(cols) + [cols[0]])


def test_judgements():
    assert F.judge_k([1e-16, 4e-16], [4e-16])["met"]
    assert not F.judge_k([1e-16, 2e-12], [4e-16])["met"]
    assert not F.judge_k([1e-16], [1.1e-12])["met"]
    # D: the ratio, judged only where both errors are above the floor
    assert F.judge_d(1e-3, 1e-3 * (1 + 5e-9), 1.0, 0.2, 0.2, 100.0)["met"]
    assert not F.judge_d(1e-3, 1e-3 * (1 + 2e-8), 1.0, 0.2, 0.2, 100.0)["met"]
    r = F.judge_d(1e-12, 3e-12, 1.0, 0.2, 0.2, 100.0)
    assert r["met"] and not r["ratio_judged"]
    assert not F.judge_d(1e-3, 1e-3, 1.0, 0.2, 0.2 + 2e-7, 100.0)["met"]
    # L: bit for bit, so one ulp fails
    H = np.linspace(0.0, 1.0, 12).reshape(3, 4)
    u = np.arange(3.0)
    assert F.judge_l([H, H.copy(), H.copy()], [u, u.copy(), u.copy()])["met"]
    assert not F.judge_l([H, H, np.nextafter(H, 2.0)], [u, u, u])["met"]
    assert not F.judge_l([H, H, H], [u, u, np.nextafter(u, 9.0)])["met"]
    with pytest.raises(ValueError):
        F.judge_l([H, H], [u, u])
    # E4: both frames
    ok = {"energy_rel": 0.01, "max_err": 0.9, "span": 100.0, "floor_max": 0.0}
    bad = {"energy_rel": 0.03, "max_err": 0.9, "span": 100.0, "floor_max": 0.0}
    assert F.judge_e4_both_frames(ok, ok)["met"]
    assert not F.judge_e4_both_frames(ok, bad)["met"] and not F.judge_e4_both_frames(bad, ok)["met"]
    # CW's measure: the steady problems as registered, the steps lowered
    er = {}
    for i, b in enumerate(F.HELD_OUT_BENCHMARKS):
        er[(b, "steady", "registered")], er[(b, "steady", "lowered")] = 1.0 + i, 99.0
        er[(b, "step", "registered")], er[(b, "step", "lowered")] = 99.0, 10.0 + i
    assert F.cw_measure(er) == pytest.approx((1 + 2 + 3 + 10 + 11 + 12) / 6)
    # the pairings must agree
    assert F.compare_pairings({0: 1.3, 1: 1.4}, {0: 1.0, 1: 1.0})["verdict"] == "alternative loses"
    assert F.compare_pairings({0: 0.7, 1: 1.0}, {0: 1.0, 1: 1.0})["verdict"] == "unresolved"
    assert F.compare_pairings({0: 1.0, 1: 1.1}, {0: 1.0, 1: 1.0})["verdict"] == "tie"
    with pytest.raises(ValueError):
        F.compare_pairings({0: 1.0}, {0: 1.0, 1: 1.0})
    # E1 on e_H,L: Stage 0's judge
    rng = np.random.default_rng(0)
    N = 1200
    mu, fq = rng.uniform(0, 0.65, N), rng.choice(F.FREQ_BINS, N)
    assert F.judge_e1(np.full(N, 0.005), np.full(N, 0.01), mu, fq)["met"]
    assert not F.judge_e1(np.full(N, 0.011), np.full(N, 0.01), mu, fq)["met"]


def test_text_names_every_gate_and_arm():
    t = F.text()
    for g in F.GATES + F.COMPARISONS:
        assert f"  {g.key} " in t
    for k in F.ARMS:
        assert f"  {k} " in t
    assert "$10.00" in t and "$0.70" in t
