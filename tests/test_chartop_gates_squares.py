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


def test_every_e4_bar_is_gates_forms_and_u_is_section_11_14s():
    assert GS.judge_e4_both_frames is GF.judge_e4_both_frames
    assert (GS.E4_ENERGY, GS.E4_MAX_OF_RANGE) == (G.E4_ENERGY, G.E4_MAX_OF_RANGE)
    assert (GS.KIND, GS.M_PORTS, GS.BENCHMARK_N, GS.DATUM_SHIFT_K) == (GF.KIND, GF.M_PORTS, GF.BENCHMARK_N,
                                                                     GF.DATUM_SHIFT_K)
    assert GS.U_FORM is GH.U_FORM


def test_the_protocol_is_the_draft():
    assert (GS.PRICE_BAR, GS.PRICE_THREADS, GS.PRICE_REPS, GS.PRICE_RUNS, GS.PRICE_RERUNS_MAX) == (0.90, 4, 5, 3, 3)
    assert GS.PRICE_ORDER == ("classical", "H2", "H2r", "coupled solve")
    assert (GS.MAX_MS_PER_PIECE, GS.SAMPLE_S, GS.MAX_GAP_S, GS.DRY_RUNS, GS.DRY_RUNS_MIN) == (60.0, 5.0, 30.0, 6, 5)
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


def test_the_validity_of_a_run():
    ok = {"classical_ms_per_piece": {"a": 41.0, "b": 5.0}, "plugged": [True, True], "sample_gaps_s": [5.1, 4.9]}
    assert GS.run_counts2(ok) == {"counts": True, "failed": []}                 # no lower edge
    for change in ({"classical_ms_per_piece": {"a": 61.0}}, {"plugged": [True, False]}, {"plugged": []},
                   {"sample_gaps_s": [5.0, 31.0]}, {"sample_gaps_s": []}):
        assert GS.run_counts2(dict(ok, **change))["counts"] is False
    assert GS.dry_run_passes(5) and GS.dry_run_passes(6) and not GS.dry_run_passes(4)
    assert not GS.dry_run_passes(5, runs=5)


def test_the_judges():
    keys = [(1, "steady"), (1, "step")]
    assert GS.judge_e4h2({k: {"met": True} for k in keys}, keys)["met"]
    out = GS.judge_e4h2({keys[0]: {"met": True}, keys[1]: {"met": False}}, keys)
    assert not out["met"] and out["not_met"] == ["1 step"]
    assert GS.judge_pr2({k: [0.8, 0.9, 0.85] for k in keys}, keys)["met"]
    assert not GS.judge_pr2({k: [0.95, 0.91, 0.92] for k in keys}, keys)["met"]
    with pytest.raises(ValueError):
        GS.judge_pr2({k: [0.8] for k in keys}, keys)
    assert GS.new_problems((7, 9)) == ((7, "steady"), (7, "step"), (9, "steady"), (9, "step"))


def test_the_text_names_the_gates_choices_predictions_and_outcomes():
    t = GS.text()
    for s in ("E4-H2", "PR2", "<= 0.9", *GS.SETTLED.values(), *GS.PREDICTIONS, *GS.OUTCOMES.values()):
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


def test_a_price_run_in_its_own_process(tmp_path):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", KMP_DUPLICATE_LIB_OK="TRUE")
    p = subprocess.run([sys.executable, PRICE, "--price-run", "1", "--threads", "4", "--price-dir", str(tmp_path),
                        "--n", "16", "--reps", "1", "--problems", "workbench:wall-2:steady",
                        f"squares:{TEST_SEED + 3}:step"], env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    rec = json.load(open(tmp_path / "price-run-1-threads4.json", encoding="utf-8"))
    assert rec["samples"] >= 2 and set(rec["counts"]) == {"counts", "failed"}
    assert {"load_mean", "load_max", "max_gap_s", "power_events"} <= set(rec)
    assert list(rec["problems"]) == ["workbench wall-2 steady", f"squares {TEST_SEED + 3} step"]
