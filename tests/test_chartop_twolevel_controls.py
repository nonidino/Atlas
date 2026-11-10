"""Phase 2 of the Stage 0 rework (`scripts/chartop_twolevel_controls.py`): the predictions'
readings against the code that judges them, on made-up records; the script's guards; and a
smoke run on seeds and a layout no registration draws, which reads only the controls.
Nothing here reads a piece Phase 2 measures."""
from __future__ import annotations

import copy
import json
import os
import re
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: F401  (loaded first: the script then leaves the BLAS variables alone)
import pytest

pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import chartop_twolevel_controls as T  # noqa: E402

PAGE = os.path.join(ROOT, "wiki", "concepts", "Atlas 0.1", "atlas-0.1-proposal", "expert-architecture",
                    "chart-operator-stage-0-rework.md")
TL = [f"TL0-{c}" for c in T.NCS]


# ---------------------------------------------------------------------------
# made-up records on which every prediction holds
# ---------------------------------------------------------------------------


def _tl_row(low, e_H=0.1):
    return {"low": low, "e_H": e_H, "cx": 1e-15, "cx_p": 1e-15, "check_violations": 0,
            "e_K": 1e-16, "e_K_inf": 1e-16}


def _modes_records():
    out = {}
    for n in T.NS:
        row = {"set": "validation", "n": n, "key": {"set": "validation", "seed": 1},
               "U": {"low": 0.2, "e_H": 0.3}}
        for c in T.NCS:
            if c < n:
                row[f"TL0-{c}"] = _tl_row(0.05)
            elif c == n:
                row[f"TL0-{c}"] = dict(_tl_row(0.0, e_H=1e-16), cx=None, cx_p=None)
        out[f"modes-val-n{n}"] = {"config": {"part": "modes"}, "rows": [row]}
    return out


def _verdict(met=True, max_ok=True):
    fr = {"max_over_range": 0.001 if max_ok else 0.05, "bar_max_over_range": 0.01}
    return {"e4": {"met": met, "as registered": dict(fr), "lowered by 300 K": dict(fr)}}


def _e4_records():
    out = {}
    for b in ("bend-3", "s-channel", "insert-round"):
        for p in ("steady", "step"):
            frames = {}
            for f in ("registered", "lowered"):
                variants = {"U": {"energy_rel": 0.05, "pieces": {}}}
                for c in T.NCS:
                    pc = {"cx": [1e-15], "cx_p": [None], "check_violations": [0],
                          "e_K": [1e-16] if f == "registered" else [],
                          "e_K_inf": [1e-16] if f == "registered" else []}
                    variants[f"TL0-{c}"] = {"energy_rel": 0.01, "pieces": pc}
                variants["TL0-64 (control)"] = {"energy_rel": 0.001, "pieces": {}}
                frames[f] = {"floor_energy_rel": 0.001, "variants": variants}
            verdicts = {k: _verdict(max_ok=not (p == "step" and b == "s-channel")) for k in TL + ["U"]}
            out[f"e4-{b}-{p}"] = {"result": {"frames": frames, "verdicts": verdicts}}
    return out


def _cost_record(ratio=0.06, on_ac=True):
    return {"cost-laptop": {"rows": [{"n": 128, "ratio_16": ratio}] * 3 + [{"n": 64, "ratio_16": 0.5}],
                            "on_ac_throughout": on_ac}}


def _write(tmp_path, recs):
    for k, v in recs.items():
        (tmp_path / f"{k}.json").write_text(json.dumps(v), encoding="utf-8")
    return str(tmp_path)


def _all():
    return {**_modes_records(), **_e4_records(), **_cost_record()}


def _judge(tmp_path, recs):
    return T.judge(_write(tmp_path, recs))["predictions"]


def test_every_prediction_holds_on_the_made_up_records(tmp_path):
    v = _judge(tmp_path, _all())
    assert {k: v[k]["held"] for k in v} == {str(i): True for i in range(1, 7)}
    res = T.judge(str(tmp_path))
    assert res["smallest_nc_meeting_e4_on_all_six"] == "TL0-4"


@pytest.mark.parametrize("breach", ["cx", "cx_p", "column", "K", "top", "cx bench", "column bench",
                                    "K bench", "top bench"])
def test_prediction_1_is_refuted_by_each_of_its_clauses(tmp_path, breach):
    recs = copy.deepcopy(_all())
    row = recs["modes-val-n64"]["rows"][0]
    bench = recs["e4-insert-round-step"]["result"]["frames"]
    if breach == "cx":
        row["TL0-16"]["cx"] = 2e-12
    elif breach == "cx_p":
        row["TL0-8"]["cx_p"] = 2e-12
    elif breach == "column":
        row["TL0-4"]["check_violations"] = 1
    elif breach == "K":
        row["TL0-32"]["e_K_inf"] = 2e-12
    elif breach == "top":
        recs["modes-val-n32"]["rows"][0]["TL0-32"]["e_H"] = 2e-10
    elif breach == "cx bench":
        bench["lowered"]["variants"]["TL0-16"]["pieces"]["cx"] = [2e-12]
    elif breach == "column bench":
        bench["lowered"]["variants"]["TL0-4"]["pieces"]["check_violations"] = [0, 1]
    elif breach == "K bench":
        bench["registered"]["variants"]["TL0-8"]["pieces"]["e_K"] = [2e-12]
    elif breach == "top bench":
        bench["lowered"]["variants"]["TL0-64 (control)"]["energy_rel"] = 0.001 + 2e-10
    v = _judge(tmp_path, recs)
    assert v["1"]["held"] is False and v["1"]["failures_count"] == 1
    assert all(v[k]["held"] for k in ("2", "3", "4", "5", "6"))


def test_prediction_1_needs_all_six_problems(tmp_path):
    recs = _all()
    recs.pop("e4-bend-3-step")
    assert _judge(tmp_path, recs)["1"]["held"] is False


def test_prediction_2_reads_every_coarse_size_on_bend3_steady(tmp_path):
    recs = copy.deepcopy(_all())
    recs["e4-bend-3-steady"]["result"]["verdicts"]["TL0-32"]["e4"]["met"] = False
    v = _judge(tmp_path, recs)
    assert v["2"]["held"] is False and v["2"]["met"]["TL0-16"] is True


@pytest.mark.parametrize("ratio, held", [(0.5, True), (0.51, False)])
def test_prediction_3_is_half_at_every_resolution(tmp_path, ratio, held):
    recs = copy.deepcopy(_all())
    recs["modes-val-n128"]["rows"][0]["TL0-16"]["low"] = ratio * 0.2
    v = _judge(tmp_path, recs)
    assert v["3"]["held"] is held


@pytest.mark.parametrize("e16, e32, held", [(0.03, 0.01, True), (0.01, 0.03, True), (0.03, 0.03, False),
                                            (0.02, 0.5, True)])
def test_prediction_4_is_the_energy_bar_at_16_or_32_in_both_frames(tmp_path, e16, e32, held):
    recs = copy.deepcopy(_all())
    for f in ("registered", "lowered"):
        var = recs["e4-s-channel-steady"]["result"]["frames"][f]["variants"]
        var["TL0-16"]["energy_rel"], var["TL0-32"]["energy_rel"] = e16, e32
    assert _judge(tmp_path, recs)["4"]["held"] is held


def test_prediction_4_needs_both_frames(tmp_path):
    recs = copy.deepcopy(_all())
    for nc in ("TL0-16", "TL0-32"):
        recs["e4-insert-round-steady"]["result"]["frames"]["lowered"]["variants"][nc]["energy_rel"] = 0.021
    assert _judge(tmp_path, recs)["4"]["held"] is False


def test_prediction_5_is_refuted_when_every_step_meets_the_maximum_bar(tmp_path):
    recs = copy.deepcopy(_all())
    assert _judge(tmp_path, recs)["5"]["held"] is True
    recs["e4-s-channel-step"]["result"]["verdicts"]["TL0-16"] = _verdict(max_ok=True)
    assert _judge(tmp_path, recs)["5"]["held"] is False
    # a miss in one frame only is a miss
    v = _verdict(max_ok=True)
    v["e4"]["lowered by 300 K"]["max_over_range"] = 0.02
    recs["e4-bend-3-step"]["result"]["verdicts"]["TL0-16"] = v
    assert _judge(tmp_path, recs)["5"]["held"] is True


@pytest.mark.parametrize("ratio, on_ac, held", [(0.099, True, True), (0.1, True, False), (0.05, False, None)])
def test_prediction_6_is_the_n128_median_and_needs_AC(tmp_path, ratio, on_ac, held):
    recs = {**_modes_records(), **_e4_records(), **_cost_record(ratio, on_ac)}
    v = _judge(tmp_path, recs)
    assert v["6"]["held"] is held and v["6"]["judged"] is on_ac


# ---------------------------------------------------------------------------
# the prose against the code
# ---------------------------------------------------------------------------


def test_the_readings_carry_the_pages_numbers():
    """Each reading's bars appear in the script's registered text and in the page's
    section 12 row for the same prediction (LaTeX there)."""
    text = open(PAGE, encoding="utf-8").read()
    sec = text[text.index("## 12. Phase 2"):text.index("## See Also")]
    rows = {m.group(1): m.group(0) for m in re.finditer(r"^\| (\d) \|.*$", sec, re.M)}
    assert sorted(rows) == [str(i) for i in range(1, 7)]
    script = {p["id"]: p["reading"] for p in T.PREDICTIONS}
    pairs = {"1": [("1e-12", "10^{-12}"), ("1e-9", "10^{-9}"), ("1e-10", "10^{-10}"), ("n = 32", "n=32")],
             "2": [("4, 8, 16 and 32", "4$, 8, 16 and 32")],
             "3": [("0.5", "0.5"), ("{32, 64, 128}", "each $n$")],
             "4": [("0.02", "0.02"), ("n_c = 16 or at n_c = 32", "n_c=16$ or at $n_c=32")],
             "5": [("n_c = 16", "n_c=16"), ("at least one", "at least one")],
             "6": [("< 0.1", "below 0.1"), ("AC", "AC")]}
    for pid, items in pairs.items():
        for code, page in items:
            assert code in script[pid], (pid, code)
            assert page in rows[pid], (pid, page)
    # the script's texts are section 9's, word for word once the notation is set aside
    sec9 = _plain(text[text.index("## 9. Predictions"):text.index("## 10.")])
    for p in T.PREDICTIONS:
        assert _plain(p["text"]).rstrip(".") in sec9, p["id"]


def _plain(s: str) -> str:
    s = re.sub(r"\*\*\[AI Inference\]:\*\*|\[AI Inference\]", "", s)
    s = re.sub(r"10\^\{(-\d+)\}", r"1e\1", s).replace("–", "-")
    return re.sub(r"[\s$*]", "", s)


def test_the_thresholds_are_the_scripts_constants():
    assert (T.CHECK_REL, T.CHECK_ABS) == (1e-9, 1e-12)
    assert T.NCS == (4, 8, 16, 32) and T.NS == (32, 64, 128) and T.VAL_COUNT == 64
    assert T.fd.VAL_POSTHOC == 2_000_100 and T.B4_PIECES == 4 and T.B4_N == 128


# ---------------------------------------------------------------------------
# the guards
# ---------------------------------------------------------------------------


def test_the_registered_run_keeps_its_seeds_and_record(tmp_path):
    with pytest.raises(SystemExit):
        T.main(["--part", "modes", "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        T.main(["--part", "modes", "--seed-base", "9000000"])
    with pytest.raises(SystemExit):
        T.main(["--part", "modes", "--smoke"])                # the smoke run never writes the record


def test_the_cost_part_refuses_battery(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "power_state", lambda: {"on_ac": False, "raw": "1"})
    with pytest.raises(SystemExit):
        T.main(["--part", "cost", "--smoke", "--out", str(tmp_path)])


def test_a_record_of_another_configuration_is_refused(tmp_path):
    path = os.path.join(str(tmp_path), "modes-x.json")
    T.save(path, {"config": {"a": 1}, "rows": []})
    with pytest.raises(SystemExit):
        T.load_record(path, {"a": 2})
    assert T.load_record(path, {"a": 1})["rows"] == []


def test_the_regions_partition_the_elements():
    from atlas.chartop import benchmarks_squares as BS
    b = BS.build(BS.draw(T.SMOKE_LAYOUT_SEED), 16, "steady")
    seen = set()
    for p in b.problems:
        reg = T.element_regions(p)
        assert reg.shape == (16, 16) and reg.min() >= 0 and reg.max() < len(T.REGIONS)
        seen |= set(np.unique(reg).tolist())
    assert T.REGIONS.index("interior") in seen and T.REGIONS.index("corner, right") in seen


# ---------------------------------------------------------------------------
# the smoke run: the controls only
# ---------------------------------------------------------------------------


def test_the_smoke_run_holds_its_controls(tmp_path):
    out = str(tmp_path)
    assert T.main(["--part", "modes", "--smoke", "--ns", "16", "--out", out]) == 0
    assert T.main(["--part", "e4", "--smoke", "--problem", "steady", "--out", out]) == 0
    m = json.loads(open(os.path.join(out, "modes-val-n16.json"), encoding="utf-8").read())
    assert len(m["rows"]) == 2 and m["provenance"]["coarse_sha256"]
    for r in m["rows"]:
        assert r["TL0-16"]["e_H"] < 1e-10                     # n_c = n: exact
        for c in (4, 8):
            t = r[f"TL0-{c}"]
            assert t["cx"] < 1e-12 and t["check_violations"] == 0 and t["e_K_inf"] < 1e-12
    e = json.loads(open(os.path.join(out, "e4-smoke-squares-steady.json"), encoding="utf-8").read())["result"]
    for frame, fr in e["frames"].items():
        assert fr["control_none_bit_for_bit"]
        top = fr["variants"]["TL0-16 (control)"]
        assert abs(top["energy_rel"] - fr["floor_energy_rel"]) < 1e-10
        for k in ("TL0-4", "TL0-8"):
            vr = fr["variants"][k]
            assert max(vr["pieces"]["cx"]) < 1e-12 and sum(vr["pieces"]["check_violations"]) == 0
            if vr.get("split") is not None:
                assert vr["where"]["check_rel"] < 1e-10 and abs(sum(vr["where"]["share"].values()) - 1) < 1e-12
                assert vr["split_check_rel"] < 1e-10
        if frame == "registered":
            assert max(e["frames"][frame]["variants"]["TL0-4"]["pieces"]["e_K_inf"]) < 1e-12
    # a resumed run measures nothing again
    before = open(os.path.join(out, "modes-val-n16.json"), encoding="utf-8").read()
    assert T.main(["--part", "modes", "--smoke", "--ns", "16", "--out", out]) == 0
    after = json.loads(open(os.path.join(out, "modes-val-n16.json"), encoding="utf-8").read())
    assert [r["key"] for r in after["rows"]] == [r["key"] for r in json.loads(before)["rows"]]
