"""Stage 0's registration (W355): the gate module is the one registered, and the bars
stated in the wiki agree with the ones in code ("pre-registered criteria drift in code",
and always in the permissive direction, unless something checks)."""
from __future__ import annotations

import hashlib
import os
import re

import numpy as np

from atlas.chartop import data, gates as G

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = os.path.join(ROOT, "out", "chartop", "registered.txt")
PAGE = os.path.join(ROOT, "wiki", "concepts", "Atlas 0.1", "atlas-0.1-proposal",
                    "expert-architecture", "chart-operator-stage-0.md")


def _hash(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def test_the_registered_module_is_unchanged():
    assert os.path.exists(REG), "out/chartop/registered.txt is missing: register before training"
    text = open(REG, encoding="ascii").read()
    m = re.search(r"gates\.py\s+sha256 ([0-9a-f]{64})", text)
    assert m, "the registration names no module hash"
    assert m.group(1) == _hash(os.path.join(ROOT, "atlas", "chartop", "gates.py"))


def _row(gate: str) -> str:
    text = open(PAGE, encoding="utf-8").read()
    for line in text.splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) >= 3 and cells[1] == gate:
            return cells[2]
    raise AssertionError(f"no row for {gate} in the wiki's registered-bars table")


def _numbers(cell: str):
    dec = {float(x) for x in re.findall(r"(?<![\d.])\d+\.\d+(?![\d.])", cell)}
    pw = {10.0 ** -int(k) for k in re.findall(r"10\^\{-(\d+)\}", cell)}
    ints = {int(x) for x in re.findall(r"(?<![\d.\w{-])(\d{2,})(?![\d.\w^-])", cell)}
    return dec, pw, ints


def test_the_wiki_states_the_registered_bars():
    assert _numbers(_row("E1")) == ({G.E1_MODES_MEDIAN, G.E1_MODES_P90, G.E1_PARTICULAR_MEDIAN,
                                     G.E1_PARTICULAR_P90}, set(), {G.E1_MIN_BIN})
    assert _numbers(_row("E3")) == (set(), {G.E3_ROUNDING}, set())
    assert _numbers(_row("E4")) == ({G.E4_ENERGY, G.E4_MAX_OF_RANGE}, set(), set())
    assert _numbers(_row("SE1")) == (set(), {G.SE1_IDENTITY, G.SE1_LAMBDA_MIN}, set())
    assert _numbers(_row("DUAL")) == (set(), {G.DUAL_CHAIN_ROUNDING}, set())
    assert _numbers(_row("KT")) == (set(), set(), {G.KAPPA_T_MAX_ITERATIONS})
    assert _numbers(_row("C1–C7")) == ({G.WIN, G.LOSE}, set(), set())


def test_the_bars_are_the_owners():
    """D2, 2026-10-18: the proposal accepted as it was made."""
    assert (G.E1_MODES_MEDIAN, G.E1_MODES_P90) == (0.01, 0.03)
    assert (G.E1_PARTICULAR_MEDIAN, G.E1_PARTICULAR_P90) == (0.02, 0.05)
    assert (G.E4_ENERGY, G.E4_MAX_OF_RANGE) == (0.02, 0.01)
    assert (G.WIN, G.LOSE) == (0.8, 1.25)


def test_one_source_for_seeds_and_bins():
    assert data.SEEDS is G.SEED_BASE
    assert G.MU_BINS[-1][1] == data.ENVELOPE.k_mu
    for m in (0.0, 0.1, 0.149, 0.15, 0.3, 0.44, 0.45, 0.6, 0.65):
        assert G.mu_bin_of(m) == data.mu_bin(m)


def test_judgements():
    rng = np.random.default_rng(0)
    N = 1200
    mu = rng.uniform(0, 0.65, N)
    fq = rng.choice(G.FREQ_BINS, N)
    good = G.judge_e1(np.full(N, 0.005), np.full(N, 0.01), mu, fq)
    assert good["met"] and good["bins_judged"] == 12
    bad = G.judge_e1(np.where(mu > 0.5, 0.05, 0.005), np.full(N, 0.01), mu, fq)
    assert not bad["met"]
    assert G.judge_e4(0.019, 0.9, 100.0)["met"] and not G.judge_e4(0.021, 0.9, 100.0)["met"]
    assert not G.judge_e4(0.01, 1.1, 100.0)["met"]
    assert G.compare(0.7, 1.0)["verdict"] == "alternative wins"
    assert G.compare(1.3, 1.0)["verdict"] == "alternative loses"
    assert G.compare(1.0, 1.0)["verdict"] == "tie"
    assert G.c5_choice({"0.3M": 1.2, "1M": 1.0, "3M": 0.99}) == "0.3M"
    assert G.c5_choice({"0.3M": 1.3, "1M": 1.0, "3M": 0.99}) == "1M"
    assert G.judge_se3(0.001, 0.01)["met"] and not G.judge_se3(0.02, 0.01)["met"]
