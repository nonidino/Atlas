"""The one evaluation of the pinned registration after section 11.14
(``scripts/chartop_evaluate_hybrid.py``), tested without reading a new problem: every test
that runs H, U or P-s0 runs them on the control's problems, which section 11.14 read post
hoc.  A new problem is built here only to read the rule's picks, which need no expert."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import chartop_evaluate_hybrid as E  # noqa: E402
import chartop_forms_diagnose as fd  # noqa: E402
from atlas.chartop import benchmarks_workbench as BW, gates_hybrid as GH  # noqa: E402

SCRIPT = os.path.join(ROOT, "scripts", "chartop_evaluate_hybrid.py")
P_S0 = os.path.join(ROOT, "out", "chartop", "runs", "prior", "P-s0", "best.pt")
needs_p_s0 = pytest.mark.skipif(not os.path.exists(P_S0), reason="P-s0's weights are not on this machine")
CONTROL_SPEC = ("bend-3", "steady", GH.CONTROL_VARIANT)


@pytest.fixture(scope="module")
def experts():
    if not os.path.exists(P_S0):
        pytest.skip("P-s0's weights are not on this machine")
    torch.set_num_threads(GH.PRICE_THREADS)
    return E.make_experts()


def test_the_registered_defaults():
    a = E.build_parser().parse_args([])
    assert (a.n, a.reps, a.runs, a.reruns, a.threads) == (GH.BENCHMARK_N, GH.PRICE_REPS, GH.PRICE_RUNS,
                                                          GH.PRICE_RERUNS_MAX, GH.PRICE_THREADS)
    assert a.problems is None and a.control is None and a.registration == E.REGISTRATION
    assert a.quiet_limit == 600.0
    assert E.specs_of(a) == [(n, p, E.WORKBENCH) for n, p in GH.NEW_PROBLEMS]


def test_the_wait_for_a_quiet_machine(monkeypatch):
    """It decides when a run starts: at once when quiet, after the readings when busy, and
    never past its limit."""
    seq = iter([0.4, 0.3, 0.05])
    monkeypatch.setattr(E, "other_core_max", lambda exclude, interval=1.0: next(seq))
    q = E.wait_for_quiet(limit_s=60.0, every_s=0.0)
    assert q["quiet"] and q["readings"] == [0.4, 0.3, 0.05]
    monkeypatch.setattr(E, "other_core_max", lambda exclude, interval=1.0: 0.5)
    q = E.wait_for_quiet(limit_s=0.0, every_s=0.0)
    assert not q["quiet"] and q["readings"] == [0.5]


def test_the_pin_holds_and_a_changed_file_is_refused(tmp_path):
    pins = E.check_pin()
    assert set(pins) == set(E.PINNED) and len(pins) == 10
    text = open(E.REGISTRATION, encoding="ascii").read()
    rel = "atlas/chartop/host.py"
    h = re.search(re.escape(rel) + r"\s+sha256 ([0-9a-f]{64})", text).group(1)
    bad = tmp_path / "registered-hybrid.txt"
    bad.write_text(text.replace(h, "0" * 64), encoding="ascii")
    with pytest.raises(SystemExit, match="host.py is not the registered file"):
        E.check_pin(str(bad))
    with pytest.raises(SystemExit, match="missing"):
        E.check_pin(str(tmp_path / "absent.txt"))


def test_the_one_evaluation_runs_once(tmp_path):
    out = str(tmp_path)
    stamp, d = E.begin(out, resume=False)
    assert os.path.isdir(d) and re.fullmatch(r"\d{8}-\d{6}", stamp)
    with pytest.raises(SystemExit, match="unfinished"):
        E.begin(out, resume=False)
    assert E.begin(out, resume=True) == (stamp, d)
    open(os.path.join(out, f"{E.STAMP_PREFIX}{stamp}.json"), "w").write("{}")
    with pytest.raises(SystemExit, match="has run"):
        E.begin(out, resume=True)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="nothing to resume"):
        E.begin(str(empty), resume=True)


def test_a_checkpoint_is_computed_once(tmp_path):
    calls = []
    assert E.checkpoint(str(tmp_path), "unit", lambda: calls.append(1) or {"x": np.float64(2.5)}) == {"x": 2.5}
    assert E.checkpoint(str(tmp_path), "unit", lambda: calls.append(1) or {"x": 9.0}) == {"x": 2.5}
    assert calls == [1]


def test_u_is_section_11_14s_u():
    a, b = E.build_u().state_dict(), fd.zero_heads().state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_the_picks_are_the_registrations():
    b = E.build_problem(CONTROL_SPEC, 16)
    assert E.expected_picks(CONTROL_SPEC, b) == [0, 1, 2, 12, 13, 14] == E.picks_of(CONTROL_SPEC, b)
    rec = json.load(open(os.path.join(ROOT, *GH.CLASSICAL_RECORD.split("/")), encoding="utf-8"))
    for r in rec["problems"]:
        spec = (r["name"], r["problem"], E.WORKBENCH)
        assert E.expected_picks(spec, BW.build(r["name"], 16, r["problem"])) == r["picks"]
    assert E.expected_picks(("bend-3", "steady", "none-such"), b) is None


def test_a_pick_that_differs_is_refused(monkeypatch):
    b = E.build_problem(CONTROL_SPEC, 16)
    monkeypatch.setattr(E, "expected_picks", lambda spec, b: [0])
    with pytest.raises(SystemExit, match="the registration recorded"):
        E.picks_of(CONTROL_SPEC, b)


def test_h_is_u_and_classical_everywhere_is_the_floor(experts):
    """On a control problem: the stand-in answers U's bits in both frames, and H with every
    piece classical is the classical floor exactly."""
    b = E.build_problem(CONTROL_SPEC, 32)
    every = list(range(len(b.problems)))
    res = E.coupled(b, {"H": (experts["Z"], every), "U": (experts["U"], [])})
    assert res["H answers U's bits"] is True
    for f in E.FRAMES:
        assert res["frames"][f]["arms"]["H"]["energy_rel"] == res["frames"][f]["floor"]
    assert set(res["verdicts"]) == {"H", "U"} and res["verdicts"]["H"]["met"]


@needs_p_s0
def test_the_control_reproduces_bit_for_bit_on_bend_3(experts):
    ctl = E.control(experts, GH.BENCHMARK_N, [("bend-3", "steady"), ("bend-3", "step")])
    assert all(v["H equals section 11.14's"] and v["P-s0 equals the one evaluation's"] for v in ctl.values())


def test_a_price_on_a_control_problem(experts):
    b = E.build_problem(CONTROL_SPEC, 32)
    p = E.price_problem(b, E.picks_of(CONTROL_SPEC, b), experts, reps=2)
    assert set(p["seconds"]) == set(GH.PRICE_ORDER) and all(len(v) == 2 for v in p["every_repetition"].values())
    assert 0.0 < p["price"] < 2.0 and p["classical_ms_per_piece"] > 0.0 and p["pieces"] == 15
    s = p["seconds"]
    assert p["price"] == GH.price(s["H"], s["classical"], s["coupled solve"])


def test_the_machine_state_and_the_power_log():
    st = E.machine_state({os.getpid()})
    assert set(st) == {"time", "plugged", "percent", "other_core_max"} and st["other_core_max"] >= 0.0
    ev = E.power_events(st["time"], st["time"])
    if sys.platform == "win32":
        assert isinstance(ev, list)


@needs_p_s0
def test_a_price_run_in_its_own_process(tmp_path):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", KMP_DUPLICATE_LIB_OK="TRUE")
    p = subprocess.run([sys.executable, SCRIPT, "--price-run", "1", "--threads", "4", "--price-dir",
                        str(tmp_path), "--n", "32", "--reps", "1", "--problems", ":".join(CONTROL_SPEC)],
                       env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    rec = json.load(open(tmp_path / "price-run-1-threads4.json", encoding="utf-8"))
    assert set(rec) >= {"start", "end", "power_events", "problems", "counts", "torch_threads"}
    assert rec["torch_threads"] == 4 and set(rec["counts"]) == {"counts", "failed"}
    assert list(rec["problems"]) == ["bend-3 steady"]


def _acc(h=0.001, u=0.001, p=0.05, floor=0.001, met=True, steady_top="plate-insert"):
    acc = {}
    for name, problem in GH.NEW_PROBLEMS:
        hh = h + (0.004 if (name == steady_top and problem == "steady") else 0.0)
        fr = {f: {"floor": floor, "arms": {"H": {"energy_rel": hh}, "U": {"energy_rel": u},
                                            "P-s0": {"energy_rel": p}}} for f in E.FRAMES}
        acc[f"{name} {problem}"] = {"frames": fr, "verdicts": {"H": {"met": met}, "U": {"met": met},
                                                               "P-s0": {"met": False}}}
    return acc


def test_the_predictions_are_judged_as_registered():
    pr = {"met": True, "geometric_mean": 0.8}
    out = E.judge_predictions(_acc(), pr)
    assert all(v["held"] for v in out.values()) and out["3"]["largest on"] == "plate-insert"
    assert not E.judge_predictions(_acc(u=0.0012), pr)["1"]["held"]         # 2e-4 over the floor
    assert E.judge_predictions(_acc(u=0.00109), pr)["1"]["held"]             # 0.9e-4 over it
    assert not E.judge_predictions(_acc(h=0.012), pr)["2"]["held"]
    assert not E.judge_predictions(_acc(steady_top="wall-2"), pr)["3"]["held"]
    assert not E.judge_predictions(_acc(), {"met": True, "geometric_mean": 0.89})["4"]["held"]
    assert not E.judge_predictions(_acc(), {"met": False, "geometric_mean": 0.8})["4"]["held"]
    assert not E.judge_predictions(_acc(p=0.0011), pr)["5"]["held"]          # below H somewhere
    assert not E.judge_predictions(_acc(met=False), pr)["5"]["held"]         # meets E4 no less often


def test_the_outcome():
    assert E.outcome({"met": True}, {"met": True}) == "both gates met"
    assert E.outcome({"met": False}, {"met": True}) == "E4-H not met"
    assert E.outcome({"met": True}, {"met": False}) == "E4-H met, PR not met"
    assert E.outcome({"met": True}, {"met": None}) == "E4-H met, PR not judged"
    assert {"both gates met", "E4-H not met", "E4-H met, PR not met"} == set(GH.OUTCOMES)


@needs_p_s0
def test_the_whole_run_on_a_control_problem(tmp_path, capsys):
    """End to end on one control problem at n = 32, one price run: the record and the table
    are written, nothing is judged as registered, and nothing lands in out/chartop."""
    before = set(os.listdir(E.OUT))
    rc = E.main(["--out", str(tmp_path), "--problems", ":".join(CONTROL_SPEC), "--control", "bend-3:steady",
                 "--n", "32", "--reps", "1", "--runs", "1", "--reruns", "0", "--quiet-limit", "0"])
    assert rc == 0
    recs = [f for f in os.listdir(tmp_path) if f.startswith(E.STAMP_PREFIX) and f.endswith(".json")]
    assert len(recs) == 1
    rec = json.load(open(tmp_path / recs[0], encoding="utf-8"))
    assert rec["registered_run"] is False and rec["predictions"] == {}
    assert rec["control"]["bend-3 steady"]["H equals section 11.14's"]
    assert rec["accuracy"]["bend-3 steady"]["H answers U's bits"] is True
    assert rec["price"]["PR"]["met"] is None and "reported_one_thread" in rec["price"]
    assert all("before_start" in r for r in rec["price"]["runs"])
    printed = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
    assert len([l for l in printed if not re.match(r"^\d\d:\d\d:\d\d ", l)]) <= 10
    assert set(os.listdir(E.OUT)) == before
