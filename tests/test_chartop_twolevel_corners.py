"""Branch 2c's check (`scripts/chartop_twolevel_corners.py`): the predictions' readings
against the code that judges them, on made-up records; the script's guards; and a smoke run
on seeds and a layout no registration draws, which reads only the controls.  Nothing here
reads a piece the check measures."""
from __future__ import annotations

import copy
import json
import math
import os
import re
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: F401  (loaded first: the script then leaves the BLAS variables alone)
import pytest

pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import chartop_twolevel_corners as T  # noqa: E402

PAGE = os.path.join(ROOT, "wiki", "concepts", "Atlas 0.1", "atlas-0.1-proposal", "expert-architecture",
                    "chart-operator-stage-0-rework.md")
SIX = [f"{b}-{p}" for b in ("bend-3", "s-channel", "insert-round") for p in ("steady", "step")]
K16 = {"B": T.arm_key(T.B8, 16), "A": T.arm_key(T.A3, 16)}


# ---------------------------------------------------------------------------
# made-up records on which every prediction holds
# ---------------------------------------------------------------------------


def _arm_row(extra: int) -> dict:
    row = {"e_H": 0.1, "cx": 1e-15, "cx_p": 1e-15, "check_violations": 0, "e_K": 1e-16,
           "e_K_inf": 1e-16, "extra": extra}
    if extra == 0:
        row["bit_for_bit_tl0"] = True
    return row


def _modes_records() -> dict:
    out = {}
    for n in T.NS:
        row = {"set": "validation", "n": n, "key": {"set": "validation", "seed": 1}, "U": {"e_H": 0.3}}
        for key, tr, _nc in T.arms(n):
            row[key] = _arm_row(0 if tr is None else 4)
        if n == 32:
            row["top"] = {"e_H": 1e-16, "treatments": {t: {"extra": 0, "bit_for_bit": True} for t in T.TREATMENTS}}
        out[f"modes-val-n{n}"] = {"config": {"part": "modes"}, "rows": [row]}
    return out


def _e4_records() -> dict:
    out = {}
    for unit in SIX:
        frames = {}
        for f in ("registered", "lowered"):
            variants = {}
            for key, tr, _nc in T.arms(64):
                exc = 0.02 if tr is None else (0.016 if tr.kind == "A" else 0.01)
                pc = {"cx": [1e-15], "cx_p": [None], "check_violations": [0],
                      "e_K": [1e-16] if f == "registered" else [],
                      "e_K_inf": [1e-16] if f == "registered" else [], "extra": [4], "bit_for_bit_tl0": [True]}
                vr = {"energy_rel": 0.019, "excess_rel": exc, "energy_abs": 1.0, "max_err": 0.1, "pieces": pc,
                      "split": {"j1-3": 1.0},
                      "where": {"share": {"corner, not right": 0.85 if tr is None else 0.3}}}
                if tr is not None:
                    vr["same_errors_as_tl0"] = unit.startswith("bend-3")
                variants[key] = vr
            frames[f] = {"floor_energy_rel": 0.001, "variants": variants,
                         "top": {t: {"extra": 0, "bit_for_bit": True} for t in T.TREATMENTS}}
        verdicts = {k: {"e4": {"met": unit != "s-channel-step"}} for k in variants}
        out[f"e4-{unit}"] = {"result": {"frames": frames, "verdicts": verdicts}}
    return out


def _cost_record(ratio=0.1, on_ac=True) -> dict:
    two = {"non_right": 2, f"{T.B8}_ratio": ratio, "TL0_ratio": 0.05}
    four = {"non_right": 4, f"{T.B8}_ratio": 0.5, "TL0_ratio": 0.05}
    return {"cost-laptop": {"rows": [two] * 4 + [four] * 3, "on_ac_throughout": on_ac}}


def _all() -> dict:
    return {**_modes_records(), **_e4_records(), **_cost_record()}


def _judge(tmp_path, recs) -> dict:
    for k, v in recs.items():
        (tmp_path / f"{k}.json").write_text(json.dumps(v), encoding="utf-8")
    return T.judge(str(tmp_path))["predictions"]


def test_every_prediction_holds_on_the_made_up_records(tmp_path):
    v = _judge(tmp_path, _all())
    assert {k: v[k]["held"] for k in v} == {str(i): True for i in range(1, 6)}


@pytest.mark.parametrize("breach", ["cx", "cx_p", "column", "K", "bit for bit", "top", "top e_H", "cx bench",
                                    "column bench", "K bench", "bit for bit bench", "bend-3", "top bench"])
def test_prediction_1_is_refuted_by_each_of_its_clauses(tmp_path, breach):
    recs = copy.deepcopy(_all())
    row = recs["modes-val-n64"]["rows"][0]
    top32 = recs["modes-val-n32"]["rows"][0]["top"]
    bench = recs["e4-insert-round-step"]["result"]["frames"]
    if breach == "cx":
        row[T.arm_key("A-K1", 16)]["cx"] = 2e-12
    elif breach == "cx_p":
        row[T.arm_key("B-n/16", 8)]["cx_p"] = 2e-12
    elif breach == "column":
        row[T.arm_key("A-K2", 16)]["check_violations"] = 1
    elif breach == "K":
        row[T.arm_key("B-n/8", 32)]["e_K_inf"] = 2e-12
    elif breach == "bit for bit":
        row[T.arm_key("A-K3", 8)] = dict(_arm_row(0), bit_for_bit_tl0=False)
    elif breach == "top":
        top32["treatments"]["B-n/8"]["extra"] = 3
    elif breach == "top e_H":
        top32["e_H"] = 2e-10
    elif breach == "cx bench":
        bench["lowered"]["variants"][K16["B"]]["pieces"]["cx"] = [2e-12]
    elif breach == "column bench":
        bench["lowered"]["variants"][K16["A"]]["pieces"]["check_violations"] = [0, 1]
    elif breach == "K bench":
        bench["registered"]["variants"][T.arm_key("A-K1", 16)]["pieces"]["e_K"] = [2e-12]
    elif breach == "bit for bit bench":
        bench["registered"]["variants"][T.arm_key("B-n/16", 32)]["pieces"]["bit_for_bit_tl0"] = [True, False]
    elif breach == "bend-3":
        recs["e4-bend-3-step"]["result"]["frames"]["lowered"]["variants"][K16["B"]]["same_errors_as_tl0"] = False
    elif breach == "top bench":
        bench["lowered"]["top"]["A-K2"]["bit_for_bit"] = False
    v = _judge(tmp_path, recs)
    assert v["1"]["held"] is False and v["1"]["failures_count"] == 1
    assert all(v[k]["held"] for k in ("2", "3", "4", "5"))


def test_prediction_1_reads_the_treatments_not_tl0(tmp_path):
    """TL0's own properties were Phase 2's prediction 1; this one reads the enlarged spaces."""
    recs = copy.deepcopy(_all())
    recs["modes-val-n64"]["rows"][0][T.arm_key(None, 16)]["cx"] = 1e-6
    assert _judge(tmp_path, recs)["1"]["held"] is True


def test_prediction_1_needs_all_six_problems(tmp_path):
    recs = _all()
    recs.pop("e4-bend-3-step")
    assert _judge(tmp_path, recs)["1"]["held"] is False


@pytest.mark.parametrize("excess, held", [(0.0176, True), (0.0177, False), (0.019, False)])
def test_prediction_2_is_the_excess_bar_in_both_frames(tmp_path, excess, held):
    recs = copy.deepcopy(_all())
    recs["e4-s-channel-steady"]["result"]["frames"]["lowered"]["variants"][K16["B"]]["excess_rel"] = excess
    assert _judge(tmp_path, recs)["2"]["held"] is held


def test_prediction_2_needs_e4_on_the_five_and_never_reads_the_sixth(tmp_path):
    recs = copy.deepcopy(_all())
    recs["e4-insert-round-step"]["result"]["verdicts"][K16["B"]]["e4"]["met"] = False
    assert _judge(tmp_path, recs)["2"]["held"] is False
    recs = copy.deepcopy(_all())
    recs["e4-s-channel-step"]["result"]["verdicts"][K16["B"]]["e4"]["met"] = True     # unmeetable, unread
    v = _judge(tmp_path, recs)["2"]
    assert v["held"] is True and "s-channel-step" not in v["met"]
    recs = copy.deepcopy(_all())                                # another arm's E4 is not read
    recs["e4-bend-3-step"]["result"]["verdicts"][K16["A"]]["e4"]["met"] = False
    assert _judge(tmp_path, recs)["2"]["held"] is True


@pytest.mark.parametrize("unit, frame", [("s-channel-steady", "registered"), ("s-channel-step", "lowered")])
def test_prediction_3_needs_a_above_b_in_every_frame(tmp_path, unit, frame):
    recs = copy.deepcopy(_all())
    var = recs[f"e4-{unit}"]["result"]["frames"][frame]["variants"]
    var[K16["A"]]["excess_rel"] = var[K16["B"]]["excess_rel"]              # equal is not above
    v = _judge(tmp_path, recs)
    assert v["3"]["held"] is False and v["2"]["held"] is True


@pytest.mark.parametrize("share, held", [(0.49, True), (0.5, False), (None, True)])
def test_prediction_4_is_under_half_in_the_non_right_corner_blocks(tmp_path, share, held):
    recs = copy.deepcopy(_all())
    vr = recs["e4-s-channel-step"]["result"]["frames"]["lowered"]["variants"][K16["B"]]
    if share is None:
        vr["split"], vr["where"] = None, None                 # below the split floor: no share
    else:
        vr["where"]["share"]["corner, not right"] = share
    assert _judge(tmp_path, recs)["4"]["held"] is held


@pytest.mark.parametrize("ratio, on_ac, held", [(0.149, True, True), (0.15, True, False), (0.05, False, None)])
def test_prediction_5_is_the_two_corner_median_and_needs_AC(tmp_path, ratio, on_ac, held):
    recs = {**_modes_records(), **_e4_records(), **_cost_record(ratio, on_ac)}
    v = _judge(tmp_path, recs)["5"]
    assert v["held"] is held and v["judged"] is on_ac and v["pieces"] == 4


# ---------------------------------------------------------------------------
# the prose against the code
# ---------------------------------------------------------------------------


def _plain(s: str) -> str:
    s = re.sub(r"\*\*\[AI Inference\]:\*\*|\[AI Inference\]", "", s)
    s = re.sub(r"10\^\{(-\d+)\}", r"1e\1", s).replace("–", "-").replace("\\%", "%")
    return re.sub(r"[\s$*]", "", s)


def _section(text: str, head: str) -> str:
    start = text.index(head)
    return text[start:text.index("\n## ", start + 1)]


def test_the_readings_carry_the_pages_numbers():
    """Each reading's bars appear in the script's text and in the page's section 16 row for
    the same prediction (LaTeX there); the script's prediction texts are section 15's, word
    for word once the notation is set aside."""
    text = open(PAGE, encoding="utf-8").read()
    sec = _section(text, "## 16. Branch 2c")
    rows = {m.group(1): m.group(0) for m in re.finditer(r"^\| (\d) \|.*$", sec, re.M)}
    assert sorted(rows) == [str(i) for i in range(1, 6)]
    script = {p["id"]: p["reading"] for p in T.PREDICTIONS}
    pairs = {"1": [("1e-12", "10^{-12}"), ("1e-9", "10^{-9}"), ("1e-10", "10^{-10}"), ("n = 32", "n=32"),
                   ("bend-3", "bend-3"), ("K = 1, 2, 3", "K=1$, 2, 3")],
             "2": [("0.0177", "0.0177"), ("n_c = 16", "n_c=16"), ("five problems", "five problems"),
                   ("both frames", "both frames")],
             "3": [("K = 3", "K=3"), ("n/8", "n/8"), ("both frames", "both frames")],
             "4": [("0.5", "0.5"), ("1e-9", "10^{-9}"), ("both frames", "both frames")],
             "5": [("< 0.15", "below 0.15"), ("exactly two", "exactly two"), ("AC", "AC"), ("n = 128", "n=128")]}
    for pid, items in pairs.items():
        for code, page in items:
            assert code in script[pid], (pid, code)
            assert page in rows[pid], (pid, page)
    sec15 = _plain(_section(text, "## 15. Branch 2c"))
    for p in T.PREDICTIONS:
        assert _plain(p["text"]).rstrip(".") in sec15, p["id"]


def test_the_thresholds_are_the_scripts_constants():
    assert (T.EXCESS_BAR, T.CORNER_SHARE_BAR, T.COST_BAR, T.TOL, T.TOP_EH) == (0.0177, 0.5, 0.15, 1e-12, 1e-10)
    assert abs(math.sqrt(0.02 ** 2 - 0.0093 ** 2) - T.EXCESS_BAR) < 5e-5    # section 13's 1.77 %
    assert (T.tc.CHECK_REL, T.tc.CHECK_ABS, T.tc.SPLIT_FLOOR) == (1e-9, 1e-12, 1e-9)
    assert T.MAIN_NC == 16 and T.CURVE_NCS == (8, 32) and T.NS == (32, 64, 128) and T.VAL_COUNT == 64
    assert list(T.TREATMENTS) == ["A-K1", "A-K2", "A-K3", "B-n/8", "B-n/16"]
    assert T.CURVE_TREATMENTS == ("A-K3", "B-n/8", "B-n/16") and (T.A3, T.B8) == ("A-K3", "B-n/8")
    assert T.COST_N == 128 and T.COST_COUNT == 16 and T.co.RIGHT_DEG == 2.0 and T.co.SINGULAR_R == 0.5
    assert T.fd.VAL_POSTHOC == 2_000_100


def test_the_arms():
    a64 = T.arms(64)
    by_nc = {}
    for key, tr, nc in a64:
        by_nc.setdefault(nc, []).append(key)
        assert key == T.arm_key(None if tr is None else tr.name, nc)
    assert sorted(by_nc) == [8, 16, 32]
    assert by_nc[16] == [T.arm_key(None, 16)] + [T.arm_key(t, 16) for t in T.TREATMENTS]
    for nc in (8, 32):
        assert by_nc[nc] == [T.arm_key(None, nc)] + [T.arm_key(t, nc) for t in T.CURVE_TREATMENTS]
    assert {nc for _k, _t, nc in T.arms(32)} == {8, 16} and {nc for _k, _t, nc in T.arms(16)} == {8}


# ---------------------------------------------------------------------------
# the guards
# ---------------------------------------------------------------------------


def test_nothing_registered_runs_before_the_owners_go(monkeypatch):
    assert "2026-11-10" in T.REGISTERED and "Register as drafted (Recommended)" in T.REGISTERED
    with pytest.raises(SystemExit):
        T.main(["--part", "modes", "--smoke"])                # the smoke run never writes the record
    monkeypatch.setattr(T, "REGISTERED", None)                # before the owner's go: nothing ran
    for part in ("modes", "e4", "cost"):
        with pytest.raises(SystemExit):
            T.main(["--part", part])


def test_the_registered_run_keeps_its_seeds_record_and_sizes(monkeypatch, tmp_path):
    monkeypatch.setattr(T, "REGISTERED", "registered (test)")
    for extra in (["--out", str(tmp_path)], ["--seed-base", "9000000"], ["--cost-n", "64"],
                  ["--bench-n", "32"], ["--allow-battery"]):
        with pytest.raises(SystemExit):
            T.main(["--part", "cost", *extra])


def test_the_cost_part_refuses_battery(tmp_path, monkeypatch):
    monkeypatch.setattr(T.tc, "power_state", lambda: {"on_ac": False, "raw": "1"})
    with pytest.raises(SystemExit):
        T.main(["--part", "cost", "--smoke", "--out", str(tmp_path)])


def test_a_record_of_another_configuration_is_refused(tmp_path):
    path = os.path.join(str(tmp_path), "modes-x.json")
    T.save(path, {"config": {"a": 1}, "rows": []})
    with pytest.raises(SystemExit):
        T.load_record(path, {"a": 2})
    assert T.load_record(path, {"a": 1})["rows"] == []


# ---------------------------------------------------------------------------
# the smoke run: the controls only
# ---------------------------------------------------------------------------


def test_the_smoke_run_holds_its_controls(tmp_path, monkeypatch):
    """Every code path at n = 16 (and the top at 32): the treatments' CX, the column check
    against TL0, K, the right-cornered pieces' bits, nothing added at the top, and the
    excess decomposition; a resumed run measures nothing again."""
    monkeypatch.setattr(T, "SMOKE_BENCH_N", 16)
    out = str(tmp_path)
    assert T.main(["--part", "modes", "--smoke", "--ns", "32", "--out", out]) == 0
    assert T.main(["--part", "e4", "--smoke", "--problem", "step", "--out", out]) == 0
    m = json.loads(open(os.path.join(out, "modes-val-n32.json"), encoding="utf-8").read())
    assert len(m["rows"]) == 2 and m["provenance"]["coarse_sha256"] and m["provenance"]["registered"] == T.REGISTERED
    assert [r["non_right"] for r in m["rows"]] == [[0, 1, 2, 3], []]
    for r in m["rows"]:
        assert r["top"]["e_H"] < 1e-10
        assert all(not x["extra"] and x["bit_for_bit"] for x in r["top"]["treatments"].values())
        for key in T._treatment_keys(r):
            t = r[key]
            assert t["cx"] < 1e-12 and t["check_violations"] == 0 and t["e_K_inf"] < 1e-12
            assert (t["extra"] > 0) == bool(r["non_right"])
            if not r["non_right"]:
                assert t["bit_for_bit_tl0"] is True
    e = json.loads(open(os.path.join(out, "e4-smoke-squares-step.json"), encoding="utf-8").read())["result"]
    assert e["n"] == 16
    for frame, fr in e["frames"].items():
        assert sum(1 for c in fr["non_right"] if c) == 8
        assert all(not x["extra"] and x["bit_for_bit"] for x in fr["top"].values())
        for key, vr in fr["variants"].items():
            assert vr["pythagoras_rel"] < 1e-9
            if key.startswith("TL0"):
                continue
            pc = vr["pieces"]
            assert max(x for x in pc["cx"] + pc["cx_p"] if x is not None) < 1e-12
            assert sum(pc["check_violations"]) == 0 and sum(pc["extra"]) > 0
            assert len(pc["bit_for_bit_tl0"]) == 17 and all(pc["bit_for_bit_tl0"])
            if frame == "registered":
                assert max(pc["e_K_inf"]) < 1e-12
    before = open(os.path.join(out, "modes-val-n32.json"), encoding="utf-8").read()
    assert T.main(["--part", "modes", "--smoke", "--ns", "32", "--out", out]) == 0
    after = json.loads(open(os.path.join(out, "modes-val-n32.json"), encoding="utf-8").read())
    assert [r["key"] for r in after["rows"]] == [r["key"] for r in json.loads(before)["rows"]]
