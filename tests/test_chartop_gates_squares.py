"""The registration after section 11.17 (`gates_squares`), against its draft
([[chart-operator-kernel-exact-forms]] section 11.18) and the owner's acceptance ("Accept as
drafted").  Every test that runs an expert other than the classical one runs it on seeds the
registration never draws (from `TEST_SEED`) or on section 11.17's problems, which have been
read; the registered layouts are only built, or measured classically."""
from __future__ import annotations

import ast
import json
import math
import os
import re
import subprocess
import sys
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from atlas.chartop import benchmarks_squares as BS  # noqa: E402
from atlas.chartop import chart as ch, data, gates as G, gates_forms as GF, gates_hybrid as GH  # noqa: E402
from atlas.chartop import gates_squares as GS, host as hs  # noqa: E402
from atlas.chartop.piece import PieceProblem  # noqa: E402
from atlas.chartop.reuse import ReuseExpert  # noqa: E402

RECORD = os.path.join(ROOT, *GS.CLASSICAL_RECORD.split("/"))
REGISTER = os.path.join(ROOT, "scripts", "chartop_register_squares.py")
PRICE = os.path.join(ROOT, "scripts", "chartop_price_squares.py")
#: seeds the registration never draws: one of each feature
TEST_SEED = 9_000_000
TEST_SEEDS = tuple(range(TEST_SEED, TEST_SEED + 5))


def _square(x0, x1, y0, y1, n=8, log_kappa=None):
    c = ch.Chart(ch.from_map(lambda xi, eta: (x0 + (x1 - x0) * xi, y0 + (y1 - y0) * eta), n), "affine")
    return PieceProblem(c, fo=None, kappa_ref=1.0, log_kappa_hat=log_kappa)


def test_the_owner_settled_the_drafts_recommendations():
    assert set(GS.SETTLED) == {"the square path", "the problems", "PR2's bar", "the load"}
    assert "H2r beside" in GS.SETTLED["the square path"] and "0.90" in GS.SETTLED["PR2's bar"]
    assert "not judged" in GS.SETTLED["the load"]
    assert "Accept as drafted" in GS.REGISTERED and "3f945f2" in GS.REGISTERED
    assert "not yet hash-pinned" in GS.REGISTERED


def test_the_owner_revised_the_price_protocol_after_the_first_dry_run():
    assert set(GS.REVISED) == {"the count", "the machine"}
    assert "Adopt it (Recommended)" in GS.REVISED["the count"] and "per problem" in GS.REVISED["the count"]
    assert "Box for both (Recommended)" in GS.REVISED["the machine"]
    assert "whole machine" in GS.REVISED["the machine"] and "accuracy on the laptop" in GS.REVISED["the machine"]
    assert "Adopt it (Recommended)" in GS.REGISTERED and "Box for both (Recommended)" in GS.REGISTERED


def test_the_owner_accepted_the_blas_revision_as_drafted():
    assert set(GS.REVISED_BLAS) == {"the BLAS threads", "the time bound", "the A/A check", "prediction 3"}
    assert "four" in GS.REVISED_BLAS["the BLAS threads"] and "35 ms" in GS.REVISED_BLAS["the time bound"]
    assert "wall-2" in GS.REVISED_BLAS["the A/A check"] and "kept as registered" in GS.REVISED_BLAS["prediction 3"]
    assert "400d2ed" in GS.REGISTERED and GS.REGISTERED.count("Accept as drafted") == 2
    assert GS.PREDICTIONS[2].startswith("3. PR2 is met: H2's geometric mean is 0.65-0.85.")     # kept


def test_every_e4_bar_is_gates_forms_and_u_is_section_11_14s():
    assert GS.judge_e4_both_frames is GF.judge_e4_both_frames
    assert (GS.E4_ENERGY, GS.E4_MAX_OF_RANGE) == (G.E4_ENERGY, G.E4_MAX_OF_RANGE)
    assert (GS.KIND, GS.M_PORTS, GS.BENCHMARK_N, GS.DATUM_SHIFT_K) == (GF.KIND, GF.M_PORTS, GF.BENCHMARK_N,
                                                                     GF.DATUM_SHIFT_K)
    assert GS.U_FORM is GH.U_FORM


def test_the_protocol_is_the_revised_draft():
    assert (GS.PRICE_BAR, GS.PRICE_THREADS, GS.PRICE_REPS, GS.PRICE_RUNS, GS.PRICE_RUNS_MAX) == (0.90, 4, 5, 3, 9)
    assert GS.PRICE_ORDER == ("classical", "H2", "H2r", "coupled solve")
    assert (GS.MAX_MS_PER_PIECE, GS.SAMPLE_S, GS.MAX_GAP_S, GS.DRY_RUNS) == (35.0, 5.0, 30.0, 6)
    assert not hasattr(GS, "PRICE_RERUNS_MAX") and not hasattr(GS, "DRY_RUNS_MIN")   # no run-wide count
    assert GS.BLAS_THREADS == GS.PRICE_THREADS == 4
    assert GS.BLAS_VARIABLES == ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
    assert GS.blas_env() == {v: "4" for v in GS.BLAS_VARIABLES}
    assert GS.PRICE_CPU == "AMD Ryzen 9 5900X 12-Core Processor" and GS.AA_TOLERANCE == 0.03
    assert GS.AA_CONTROLS == (("workbench", "wall-2", "steady"), ("workbench", "wall-2", "step"))
    assert (GS.SEED_BASE, GS.N_LAYOUTS, BS.S_SIDE) == (7_000_000, 8, 0.04)


def test_a_square_of_one_material_and_nothing_else():
    assert GS.is_square(_square(0.0, 0.04, 0.0, 0.04))
    assert GS.is_square(_square(0.31, 0.32, 0.07, 0.08))
    assert not GS.is_square(_square(0.0, 0.05, 0.0, 0.04))
    lk = np.zeros((9, 9))
    lk[0, 0] = 0.1
    assert not GS.is_square(_square(0.0, 0.04, 0.0, 0.04, log_kappa=lk))
    rh = ch.Chart(ch.from_map(lambda xi, eta: (0.04 * xi + 0.02 * eta, 0.04 * np.sqrt(0.75) * eta), 8), "affine")
    assert not GS.is_square(PieceProblem(rh, fo=None, kappa_ref=1.0))           # a rhombus: equal sides
    p = _square(0.0, 0.04, 0.0, 0.04)
    assert not GS.is_square(PieceProblem(ch.Chart(p.chart.X, "winslow-arc-length"), fo=None, kappa_ref=1.0))


def test_the_draw_is_the_seeds_and_holds_its_bounds():
    for seed in range(TEST_SEED, TEST_SEED + 300):
        sp = BS.draw(seed)
        assert sp == BS.draw(seed)
        assert sp["feature"] == BS.FEATURES[(seed - BS.SEED_BASE) % 5]
        W, H = sp["W"], sp["H"]
        assert BS.W_RANGE[0] <= W <= BS.W_RANGE[1] and BS.H_RANGE[0] <= H <= BS.H_RANGE[1]
        assert sp["plate"] != sp["feature_material"] and sp["plate"] in BS.MATERIALS
        x0, y0, w, h = sp["box"]
        assert 0 <= y0 and y0 + h <= H and 0 <= x0 and x0 + w <= W
        if sp["feature"] == "inclusion":
            assert x0 >= 1 and x0 + w <= W - 1 and h <= H - 1
        elif sp["feature"] == "spreader":
            assert x0 == 0 and w in (2, 3) and h <= H - 1
        elif sp["feature"] == "layer":
            assert (x0, w) == (0, W) and y0 + h == H and 1 <= h <= H - 1
        else:
            assert (w, h) == (2, 2) and x0 >= 1 and x0 + 2 <= W - 1
            assert (sp["feature_material"] is None) == (sp["feature"] == "hole")
    assert json.loads(json.dumps(BS.draw(TEST_SEED))) == BS.draw(TEST_SEED)


@pytest.mark.parametrize("seed", TEST_SEEDS)
def test_a_layout_is_squares_and_its_ring(seed):
    sp = BS.draw(seed)
    b = BS.build(sp, 16, "steady")
    L = b.layout(GS.KIND, GS.M_PORTS)
    # a hole: the box's 4 squares become the plate ring's 8 pieces; a round inclusion adds
    # the disc's 4 core squares and the disc ring's 8 pieces
    assert L.P == sp["W"] * sp["H"] + {"hole": 4, "round-inclusion": 16}.get(sp["feature"], 0)
    assert all(e.conforming for e in L.edges if e.other is not None)
    picks = GS.rule_pieces2(L)
    assert picks == [i for i, p in enumerate(b.problems) if p.chart.method.startswith("winslow")]
    assert len(picks) == (16 if sp["feature"] == "round-inclusion" else 8 if sp["feature"] == "hole" else 0)
    assert L.check_envelope(data.ENVELOPE) == []
    vals = np.concatenate([e.bc[1] for e in L.edges if e.other is None and e.bc[0] == "dirichlet"])
    assert np.isclose(vals.max(), 420.0) and np.isclose(vals.min(), 290.0)


def test_reuse_is_the_classical_answer_and_solves_once_a_class():
    """On a round inclusion the registration never draws: every response is the classical
    one to rounding; steady, all the squares share one solve, and in a step the plate's and
    the core's squares one each; the coupled answer is the classical one."""
    sp = BS.draw(TEST_SEED + 4)
    assert sp["feature"] == "round-inclusion"
    for problem, classes in (("steady", 1), ("step", 2)):
        b = BS.build(sp, 16, problem)
        L = b.layout(GS.KIND, GS.M_PORTS)
        rc = hs.ClassicalExpert().respond(L.problems, L.pb)
        R = ReuseExpert(GS.is_square)
        rr = R.respond(L.problems, L.pb)
        assert R.solves == len(GS.rule_pieces2(L)) + classes
        assert max(float(np.abs(a.H - z.H).max() / np.abs(a.H).max()) for a, z in zip(rc, rr)) < 1e-12
        mono = hs.monolithic(L)
        e_c, e_r = L.couple(rc).energy_error(mono.fields), L.couple(rr).energy_error(mono.fields)
        # absolute in the relative energy: a step's error here is itself round-off (~5e-13),
        # and a ratio of two round-offs says nothing
        assert abs(e_r[0] / e_r[1] - e_c[0] / e_c[1]) <= 1e-10 + 1e-8 * (e_c[0] / e_c[1])


def _counts(ms: dict, plugged=(True, True), gaps=(5.0,), max_ms=GS.MAX_MS_PER_PIECE) -> dict:
    return GS.run_counts2({"classical_ms_per_piece": ms, "plugged": list(plugged), "sample_gaps_s": list(gaps)},
                          max_ms=max_ms)


AA_OK, AA_FAILED = {"holds": True, "problems": {}}, {"holds": False, "problems": {"x": {"median": 1.2}}}


def test_the_validity_of_a_measurement():
    ok = _counts({"a": 34.0, "b": 5.0}, gaps=(5.1, 4.9))
    assert ok["run_holds"] and ok["failed"] == []                               # no lower edge
    assert all(v == {"counts": True, "failed": []} for v in ok["problems"].values())
    slow = _counts({"a": 36.0, "b": 5.0})               # the time check voids only its own problem
    assert slow["run_holds"] and not slow["problems"]["a"]["counts"] and slow["problems"]["b"]["counts"]
    assert _counts({"a": 41.0}, max_ms=60.0)["problems"]["a"]["counts"]        # the laptop's bound, then
    for run_wide in (dict(plugged=(True, False)), dict(plugged=()), dict(gaps=(5.0, 31.0)), dict(gaps=())):
        c = _counts({"a": 20.0, "b": 5.0}, **run_wide)    # AC and sleep void every measurement
        assert not c["run_holds"] and not any(v["counts"] for v in c["problems"].values())


def test_the_blas_threads_are_fixed_and_checked():
    four = [{"user_api": "blas", "internal_api": "openblas", "num_threads": 4},
            {"user_api": "openmp", "internal_api": "openmp", "num_threads": 24}]   # only BLAS pools are held
    assert GS.blas_threads_hold(four)
    assert not GS.blas_threads_hold(four + [{"user_api": "blas", "internal_api": "openblas", "num_threads": 24}])
    assert not GS.blas_threads_hold([p for p in four if p["user_api"] != "blas"])     # no BLAS seen: refused
    assert GS.on_price_cpu({"cpu": GS.PRICE_CPU}) and not GS.on_price_cpu({"cpu": "Intel(R) Core(TM) Ultra 7 155H"})
    assert not GS.on_price_cpu(None)


def test_the_aa_check():
    assert GS.aa_check({"x": [0.99, 1.02, 1.0]}, ["x"]) == {
        "holds": True, "worst": 0.0, "problems": {"x": {"median": 1.0, "holds": True}}}
    out = GS.aa_check({"x": [0.95, 0.96, 0.97], "y": [1.0, 1.0, 1.0]}, ["x", "y"])
    assert not out["holds"] and abs(out["worst"] - 0.04) < 1e-12 and out["problems"]["y"]["holds"]
    assert not GS.aa_check({"x": [1.0, 1.0]}, ["x"])["holds"]                   # short of three
    assert not GS.aa_check({}, ["x"])["holds"]
    with pytest.raises(ValueError):
        GS.aa_check({"x": [1.0, 1.0, 1.0]}, [])


def test_the_count_per_problem_and_when_the_runs_stop():
    runs = [_counts({"a": 70.0, "b": 20.0}), _counts({"a": 20.0, "b": 20.0}, plugged=(False,)),
            _counts({"a": 20.0, "b": 20.0}), _counts({"a": 20.0, "b": 61.0}), _counts({"a": 20.0, "b": 20.0})]
    assert GS.tally(runs) == {"a": 3, "b": 3} and GS.runs_suffice(runs) and not GS.runs_suffice(runs[:4])
    values = [{"a": 10 + i, "b": 20 + i} for i in range(1, 6)]
    assert GS.first_counted(runs, values) == {"a": [13, 14, 15], "b": [21, 23, 25]}
    assert GS.first_counted(runs[:2], values[:2]) == {"a": [], "b": [21]}      # short problems keep what they have
    assert not GS.dry_run_passes(runs, AA_OK)                                  # five runs, not six
    assert GS.dry_run_passes(runs + [_counts({"a": 99.0, "b": 99.0})], AA_OK)
    assert not GS.dry_run_passes(runs + [_counts({"a": 99.0, "b": 99.0})], AA_FAILED)   # the A/A check
    assert not GS.dry_run_passes(runs + [_counts({"a": 99.0, "b": 99.0})], None)
    assert not GS.dry_run_passes(runs[:4] + [_counts({"a": 99.0, "b": 20.0})] * 2, AA_OK)   # a has two


def test_the_first_dry_run_under_the_per_problem_count():
    """Section 11.18: applied to the first dry run, every problem has at least four counted
    measurements, plate-hole's steady exactly four.  The bar was set after seeing this, so it
    pins the page's claim; only a second dry run tests the bar."""
    rec = json.load(open(os.path.join(ROOT, "out", "chartop", "dryrun-squares-20261108-014441.json"), encoding="utf-8"))
    counts = []
    for r in rec["runs"]:
        assert all(f.startswith("the classical expert above") for f in r["counts"]["failed"])   # AC and sleep held
        counts.append(_counts({k: v["classical_ms_per_piece"] for k, v in r["problems"].items()},
                              plugged=[True] * r["samples"], gaps=[r["max_gap_s"]], max_ms=60.0))   # the bound then
    t = GS.tally(counts)
    assert len(t) == 12 and min(t.values()) == 4 and t["workbench plate-hole steady"] == 4
    assert [k for k, v in t.items() if v == 4] == ["workbench plate-hole steady"]
    assert GS.runs_suffice(counts) and len(counts) == GS.DRY_RUNS and rec["counted"] == [2, 4, 5, 6]
    aa = _aa_of(rec, counts)                       # section 11.18: the A/A check, as written, fails it
    assert not aa["holds"] and abs(aa["worst"] - 0.179) < 5e-4
    assert not GS.dry_run_passes(counts, aa)


def _aa_of(rec: dict, counts: list) -> dict:
    vals = [{k: v["price_H2"] for k, v in r["problems"].items()} for r in rec["runs"]]
    keys = [k for k, v in rec["runs"][0]["problems"].items() if v["squares"] == 0]
    assert len(keys) == 10 and not any("heated-strip" in k for k in keys)
    return GS.aa_check(GS.first_counted(counts, vals), keys)


def test_the_second_dry_run_and_its_diagnostic():
    """Section 11.18: on the box at the default BLAS threads, the count met the bar with nothing
    to spare (fast-heat steady exactly three, the third in run 5), and the A/A check, as written,
    fails it (worst 0.045).  The diagnostic run at four BLAS threads is within 0.007 of 1."""
    out = os.path.join(ROOT, "out", "chartop")
    rec = json.load(open(os.path.join(out, "dryrun-squares-20261108-170954.json"), encoding="utf-8"))
    assert rec["machine"]["cpu"] == GS.PRICE_CPU and rec["code"].startswith("4b06b48")
    counts = [r["counts"] for r in rec["runs"]]               # judged then by 60 ms, as recorded
    t = GS.tally(counts)
    assert GS.runs_suffice(counts) and min(t.values()) == 3 and t["workbench fast-heat steady"] == 3
    assert rec["run_of_third_count"]["workbench fast-heat steady"] == 5
    assert sum(not v["counts"] for c in counts for v in c["problems"].values()) == 19
    aa = _aa_of(rec, counts)
    assert not aa["holds"] and abs(aa["worst"] - 0.045) < 5e-4
    diag = json.load(open(os.path.join(out, "dryrun-squares-diagnostic-blas4-20261108-174234.json"), encoding="utf-8"))
    p = diag["runs"][0]["problems"]
    assert max(abs(v["price_H2"] - 1.0) for v in p.values() if v["squares"] == 0) < 0.0075
    assert 21.5 < min(v["classical_ms_per_piece"] for v in p.values()) < max(v["classical_ms_per_piece"]
                                                                           for v in p.values()) < 22.5


def test_the_runs_go_on_until_every_problem_has_three(monkeypatch):
    import chartop_price_squares as PS
    slow = {1: "a", 2: "a", 4: "b"}                          # the run, and its problem over the bound

    def fake(k, threads, specs, n, reps, out_dir):
        return {"run": k, "problems": {p: {"price_H2": 0.8 + 0.01 * k, "price_H2r": 0.4} for p in "ab"},
                "counts": _counts({p: 75.0 if slow.get(k) == p else 20.0 for p in "ab"})}
    monkeypatch.setattr(PS, "spawn", fake)
    out = PS.price_runs([], 16, 1, "unused", controls=())
    assert out["aa"] is None
    assert len(out["runs"]) == 5 and out["suffice"] and out["tally"] == {"a": 3, "b": 4}
    assert out["prices_H2"] == {"a": [0.8 + 0.01 * k for k in (3, 4, 5)], "b": [0.8 + 0.01 * k for k in (1, 2, 3)]}

    def void(k, *args):
        return {"run": k, "problems": {"a": {"price_H2": 0.8, "price_H2r": 0.4}},
                "counts": _counts({"a": 40.0}, plugged=(True, False))}
    monkeypatch.setattr(PS, "spawn", void)
    out = PS.price_runs([], 16, 1, "unused", controls=())
    assert len(out["runs"]) == GS.PRICE_RUNS_MAX and not out["suffice"] and out["prices_H2"] == {"a": []}


def test_the_evaluation_prices_the_aa_controls_beside_its_problems(monkeypatch):
    import chartop_price_squares as PS
    seen = []

    def fake(k, threads, specs, n, reps, out_dir, wall2=1.01):
        seen.append(list(specs))
        keys = [PS.spec_key(s) for s in specs]
        return {"run": k, "problems": {key: {"price_H2": wall2 if "wall-2" in key else 0.8, "price_H2r": 0.4}
                                       for key in keys}, "counts": _counts({key: 20.0 for key in keys})}
    monkeypatch.setattr(PS, "spawn", fake)
    new = [("squares", str(TEST_SEED), "steady")]
    out = PS.price_runs(new, 16, 1, "unused")                       # the controls are the default
    assert seen[0] == new + list(GS.AA_CONTROLS) and len(out["runs"]) == GS.PRICE_RUNS
    assert out["aa"]["holds"] and set(out["aa"]["problems"]) == {PS.spec_key(c) for c in GS.AA_CONTROLS}
    monkeypatch.setattr(PS, "spawn", lambda *a: fake(*a, wall2=1.05))  # identical work priced off by 0.05
    out = PS.price_runs(new, 16, 1, "unused")
    assert not out["aa"]["holds"]
    with pytest.raises(ValueError, match="A/A"):                     # and PR2 is then not judged
        GS.judge_pr2({(TEST_SEED, "steady"): out["prices_H2"][PS.spec_key(new[0])]}, [(TEST_SEED, "steady")],
                     aa=out["aa"])


def test_spawn_fixes_the_blas_threads(monkeypatch, tmp_path):
    import chartop_price_squares as PS
    got = {}

    def fake_run(cmd, env=None, **kw):
        got.update(env)
        return SimpleNamespace(returncode=1, stderr="stopped by the test")
    monkeypatch.setattr(PS.subprocess, "run", fake_run)
    with pytest.raises(SystemExit):
        PS.spawn(1, 4, [("workbench", "wall-2", "steady")], 16, 1, str(tmp_path))
    assert {v: got[v] for v in GS.BLAS_VARIABLES} == GS.blas_env()


def test_the_judges():
    keys = [(1, "steady"), (1, "step")]
    assert GS.judge_e4h2({k: {"met": True} for k in keys}, keys)["met"]
    out = GS.judge_e4h2({keys[0]: {"met": True}, keys[1]: {"met": False}}, keys)
    assert not out["met"] and out["not_met"] == ["1 step"]
    assert GS.judge_pr2({k: [0.8, 0.9, 0.85] for k in keys}, keys, aa=AA_OK)["met"]
    assert not GS.judge_pr2({k: [0.95, 0.91, 0.92] for k in keys}, keys, aa=AA_OK)["met"]
    with pytest.raises(ValueError):
        GS.judge_pr2({k: [0.8] for k in keys}, keys, aa=AA_OK)
    with pytest.raises(ValueError, match="A/A"):
        GS.judge_pr2({k: [0.8, 0.9, 0.85] for k in keys}, keys, aa=AA_FAILED)
    with pytest.raises(TypeError):                                   # the A/A verdict is required
        GS.judge_pr2({k: [0.8, 0.9, 0.85] for k in keys}, keys)
    assert GS.new_problems((7, 9)) == ((7, "steady"), (7, "step"), (9, "steady"), (9, "step"))


def test_the_text_names_the_gates_choices_predictions_and_outcomes():
    t = GS.text()
    for s in ("E4-H2", "PR2", "<= 0.9", "first 3 counted", "A/A", *GS.SETTLED.values(), *GS.REVISED.values(),
              *GS.REVISED_BLAS.values(), *GS.PREDICTIONS, *GS.OUTCOMES.values()):
        assert s in t


def test_no_expert_but_the_classical_reads_the_registered_layouts():
    tree = ast.parse(open(REGISTER, encoding="utf-8").read())
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            called.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    assert not called & {"LearnedExpert", "load_expert", "KernelExpert", "zero_heads", "without_networks",
                         "ReuseExpert", "make_arms"}
    assert "ClassicalExpert" in called
    imported = {a.name for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                for a in node.names}
    assert not imported & {"forms", "model", "train", "torch", "reuse"}


def test_the_layouts_were_measured_classically_and_are_meetable():
    assert os.path.exists(RECORD), "run scripts/chartop_register_squares.py"
    rec = json.load(open(RECORD, encoding="utf-8"))
    assert rec["experts_run"] == ["classical"]
    assert len(rec["layouts"]) == GS.N_LAYOUTS and tuple(rec["seeds"]) == GS.SEEDS
    assert rec["seeds"][0] >= GS.SEED_BASE and rec["seeds"] == sorted(rec["seeds"])
    for lay in rec["layouts"]:
        assert lay["spec"] == BS.draw(lay["seed"]) and lay["conforming"]
        for r in lay["problems"]:
            assert r["floor_meets_e4"] and r["refused"] == 0
        b = BS.build(lay["spec"], 16, "steady")
        assert lay["picks"] == GS.rule_pieces2(b.layout(GS.KIND, GS.M_PORTS)) and lay["pieces"] == len(b.problems)
    every = sorted(rec["seeds"] + [s["seed"] for s in rec["skipped"]])
    assert every == list(range(GS.SEED_BASE, GS.SEED_BASE + len(every)))   # no seed passed over unrecorded


def test_the_record_is_of_these_layouts():
    import chartop_register_prior as RP
    rec = json.load(open(RECORD, encoding="utf-8"))
    lay = min(rec["layouts"], key=lambda x: x["pieces"])
    b = BS.build(lay["spec"], GS.BENCHMARK_N, "steady")
    assert RP.fingerprint(b) == lay["problems"][0]["fingerprint"]


def test_a_price_on_a_read_problem():
    """heated-strip's 40 pieces are squares, so R2 leaves the classical solve none of them."""
    import chartop_price_squares as PS
    arms = PS.make_arms()
    b = PS.build_problem(("workbench", "heated-strip", "steady"), 16)
    picks = GS.rule_pieces2(b.layout(GS.KIND, GS.M_PORTS))
    assert picks == []
    p = PS.price_problem2(b, picks, arms, reps=1)
    assert set(p["seconds"]) == set(GS.PRICE_ORDER) and p["squares"] == 40
    assert p["price_H2"] > 0.0 and p["price_H2r"] > 0.0


def _price_run(tmp_path, blas: dict):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", KMP_DUPLICATE_LIB_OK="TRUE", **blas)
    return subprocess.run([sys.executable, PRICE, "--price-run", "1", "--threads", "4", "--price-dir", str(tmp_path),
                           "--n", "16", "--reps", "1", "--problems", "workbench:wall-2:steady"],
                          env=env, capture_output=True, text=True)


def test_a_price_run_refuses_to_time_off_the_fixed_blas_threads(tmp_path):
    p = _price_run(tmp_path, {v: "2" for v in GS.BLAS_VARIABLES})              # the failing control
    assert p.returncode != 0 and "refuses to time" in p.stderr
    assert not os.path.exists(tmp_path / "price-run-1-threads4.json")


def test_a_price_run_in_its_own_process(tmp_path):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", KMP_DUPLICATE_LIB_OK="TRUE", **GS.blas_env())
    p = subprocess.run([sys.executable, PRICE, "--price-run", "1", "--threads", "4", "--price-dir", str(tmp_path),
                        "--n", "16", "--reps", "1", "--problems", "workbench:wall-2:steady",
                        f"squares:{TEST_SEED + 3}:step"], env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    rec = json.load(open(tmp_path / "price-run-1-threads4.json", encoding="utf-8"))
    assert rec["samples"] >= 2 and set(rec["counts"]) == {"run_holds", "failed", "problems"}
    assert set(rec["counts"]["problems"]) == set(rec["problems"])
    assert {"load_mean", "load_max", "max_gap_s", "power_events", "battery_reported"} <= set(rec)
    m = rec["machine"]
    assert m["cpu"] and m["cpu"] != "unknown" and m["cpu_sources"]["cpu_count"] >= m["available_cpus"] >= 1
    assert m["torch"] == __import__("torch").__version__ and not {"host", "hostname", "node"} & set(m)
    assert rec["code"] is None or re.fullmatch(r"[0-9a-f]{40}(\+dirty)?", rec["code"])      # None off git
    pools = rec["thread_pools"]
    assert GS.blas_threads_hold(pools) and all(set(p) == {"user_api", "internal_api", "version", "num_threads"}
                                                for p in pools)          # no file path in a public record
    assert list(rec["problems"]) == ["workbench wall-2 steady", f"squares {TEST_SEED + 3} step"]
