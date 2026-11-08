"""The registration after section 11.14 (`gates_hybrid`), against its draft
([[chart-operator-kernel-exact-forms]] section 11.15) and the owner's acceptance ("Accept as
drafted"): the rule is section 11.14's, verbatim; every E4 bar is gates_forms'; the price bar
and its protocol are the draft's; the new problems are the six workbench scenarios as
drafted, built with `benchmarks`' own helpers, and measured with the classical solver only.
No expert but the classical one reads a new problem here."""
from __future__ import annotations

import ast
import inspect
import json
import math
import os
import re
import sys
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

from atlas.chartop import benchmarks as bm
from atlas.chartop import benchmarks_workbench as BW
from atlas.chartop import data, gates as G, gates_forms as GF, gates_hybrid as GH, gates_prior as GP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORD = os.path.join(ROOT, *GH.CLASSICAL_RECORD.split("/"))
DIAGNOSE = os.path.join(ROOT, "scripts", "chartop_forms_diagnose.py")
REGISTER = os.path.join(ROOT, "scripts", "chartop_register_hybrid.py")

#: section 11.15's table: pieces and the rule's picks
DRAFT = {"wall-2": (4, 2), "heated-strip": (40, 4), "plate-insert": (18, 6), "fast-heat": (18, 6),
         "bimetal-arc": (12, 4), "plate-hole": (58, 22)}
#: the charts each layout is made of
CHARTS = {"wall-2": {"affine": 4}, "heated-strip": {"affine": 40}, "plate-insert": {"affine": 18},
          "fast-heat": {"affine": 18}, "bimetal-arc": {"conformal-log-polar": 12},
          "plate-hole": {"winslow-arc-length": 12, "affine": 46}}


def _layout(name: str, n: int = 16, problem: str = "steady"):
    b = BW.build(name, n, problem)
    return b, b.layout(GH.KIND, GH.M_PORTS)


def _function_source(path: str, fname: str) -> str:
    src = open(path, encoding="utf-8").read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == fname:
            return ast.get_source_segment(src, node)
    raise AssertionError(f"{fname} is not in {path}")


def test_every_e4_bar_is_gates_forms():
    assert GH.judge_e4_both_frames is GF.judge_e4_both_frames
    assert (GH.E4_ENERGY, GH.E4_MAX_OF_RANGE) == (G.E4_ENERGY, G.E4_MAX_OF_RANGE) == (0.02, 0.01)
    assert GH.DATUM_SHIFT_K == GF.DATUM_SHIFT_K == -300.0
    assert (GH.KIND, GH.M_PORTS, GH.BENCHMARK_N) == (GF.KIND, GF.M_PORTS, GF.BENCHMARK_N)
    assert GH.BENCHMARK_N == 64


def test_the_owner_settled_the_drafts_recommendations():
    assert GH.PRICE_BAR == 0.90
    assert set(GH.SETTLED) == {"the problems", "PR's bar", "what is pinned"}
    assert "12 problems" in GH.SETTLED["the problems"] and "0.90" in GH.SETTLED["PR's bar"]
    for f in ("forms", "host", "benchmarks_workbench", "classical record"):
        assert f in GH.SETTLED["what is pinned"]
    assert "Accept as drafted" in GH.REGISTERED and "6d1b9ca" in GH.REGISTERED
    assert "not yet hash-pinned" in GH.REGISTERED


def test_the_price_protocol_is_the_draft():
    assert (GH.PRICE_THREADS, GH.PRICE_THREADS_REPORTED, GH.PRICE_REPS, GH.PRICE_RUNS,
            GH.PRICE_RERUNS_MAX) == (4, 1, 5, 3, 3)
    assert GH.PRICE_ORDER == ("classical", "H", "P-s0", "coupled solve")
    assert GH.VALID_MS_PER_PIECE == (30.0, 60.0) and GH.VALID_OTHER_CORE == 0.10
    assert GH.POWER_EVENT_IDS == (105, 506, 507)


def test_the_rule_is_section_11_14s_verbatim():
    """`rule_pieces` and its helper are the text section 11.14 fixed before measuring."""
    for fname in ("rule_pieces", "dirichlet_sides"):
        assert inspect.getsource(getattr(GH, fname)).rstrip() == _function_source(DIAGNOSE, fname)


def test_the_rule_on_a_layout_built_by_hand():
    def piece(method):
        return SimpleNamespace(chart=SimpleNamespace(method=method))
    edges = [SimpleNamespace(owner=(1, 0), other=None, bc=("dirichlet", np.zeros(3))),
             SimpleNamespace(owner=(2, 1), other=None, bc=("insulated", None)),
             SimpleNamespace(owner=(3, 2), other=(0, 0), bc=None)]
    L = SimpleNamespace(problems=[piece("affine"), piece("conformal-harmonic"),
                                  piece("winslow-arc-length"), piece("conformal-log-polar")],
                        edges=edges)
    assert GH.rule_pieces(L) == [1, 2]


def test_the_problems_are_the_drafted_set_and_the_control_is_section_11_14s():
    assert GH.NEW_NAMES == BW.NAMES == ("wall-2", "heated-strip", "plate-insert", "fast-heat",
                                        "bimetal-arc", "plate-hole")
    assert len(GH.NEW_PROBLEMS) == 12 and set(p for _, p in GH.NEW_PROBLEMS) == {"steady", "step"}
    assert GH.CONTROL == GP.HELD_OUT and len(GH.CONTROL) == 6
    assert GH.CONTROL_VARIANT == GP.HELD_OUT_VARIANT
    assert GH.U_FORM == {"form": "split", "backbone": "unet", "size": "1M", "kind": GF.KIND,
                         "m": GF.M_PORTS, "p_size": None, "bubble": "corner", "strips": True,
                         "seed": 0}


@pytest.mark.parametrize("name", BW.NAMES)
def test_the_layouts_are_the_drafts(name):
    """Pieces, picks and charts as section 11.15's table; every seam conforming; every
    piece but the ring's within aspect 3/4-4/3; one material a piece; no refusals."""
    b, L = _layout(name)
    n_pieces, n_picks = DRAFT[name]
    assert L.P == n_pieces
    assert all(e.conforming for e in L.edges if e.other is not None)
    assert len(GH.rule_pieces(L)) == n_picks
    methods = {}
    for p in b.problems:
        methods[p.chart.method] = methods.get(p.chart.method, 0) + 1
    assert methods == CHARTS[name]
    for p in b.problems:
        if p.chart.method.startswith("winslow"):
            continue
        C = p.chart.corners()
        s = [np.hypot(*(C[(k + 1) % 4] - C[k])) for k in range(4)]
        assert 0.75 - 1e-9 <= (s[0] + s[2]) / (s[1] + s[3]) <= 4.0 / 3.0 + 1e-9, p.label
        assert np.all(p.log_kappa_hat == 0.0)
    k = {"steel": 45.0, "copper": 400.0, "plate": 45.0, "insert": 400.0, "spreader": 400.0,
         "left": 45.0, "right": 45.0, "strip": 45.0}
    for p in b.problems:
        assert p.kappa_ref == k[p.label.split()[0]], p.label
    assert L.check_envelope(data.ENVELOPE) == []


def test_the_dimensions_are_the_workbench_scenarios():
    from atlas.workbench import spec as ws

    def box(case, rid):
        d = case.domain.dx
        r = [g for g in case.regions if g.id == rid][0]
        return (r.x0 * d, r.y0 * d, r.nx * d, r.ny * d)

    w = ws.example_case("wall-2")
    assert np.allclose((w.domain.nx * w.domain.dx, w.domain.ny * w.domain.dx),
                       (BW.WALL["length"], BW.WALL["height"]))
    assert np.isclose(box(w, "R1")[2], BW.WALL["steel"])
    s = ws.example_case("heated-strip")
    assert np.allclose((s.domain.nx * s.domain.dx, box(s, "steel")[3], box(s, "copper")[3]),
                       (BW.STRIP["length"], BW.STRIP["layer"], BW.STRIP["layer"]))
    for key, rid, ref in (("plate-insert", "insert", BW.INSERT), ("fast-heat", "spreader", BW.SPREADER)):
        c = ws.example_case(key)
        assert np.allclose((c.domain.nx * c.domain.dx, c.domain.ny * c.domain.dx),
                           (BW.PLATE["width"], BW.PLATE["height"]))
        assert np.allclose(box(c, rid), (ref["x0"], ref["y0"], ref["width"], ref["height"]))
    h = ws.example_case("plate-hole")
    d = h.domain.dx
    x0, y0, x1, y1 = h.domain.holes[0].bbox()
    assert np.allclose((h.domain.nx * d, h.domain.ny * d), (BW.HOLE["width"], BW.HOLE["height"]))
    assert np.allclose((0.5 * (x0 + x1) * d, 0.5 * (y0 + y1) * d, 0.5 * (x1 - x0) * d),
                       (0.5 * BW.HOLE["width"], 0.5 * BW.HOLE["height"], BW.HOLE["radius"]))
    a = ws.example_case("bimetal-arc")
    d = a.domain.dx

    def radii(outline):
        P = np.array(outline.points, float)
        return np.hypot(P[:, 0] - 4.0, P[:, 1] - 4.0) * d
    r = radii(a.domain.outline)
    cu = [g for g in a.regions if g.id == "copper"][0]
    assert np.allclose((r.min(), r.max(), radii(cu.outline).min()),
                       (BW.ARC["r_in"], BW.ARC["r_out"], BW.ARC["r_mid"]))


@pytest.mark.parametrize("name", BW.NAMES)
def test_the_ends_hold_the_held_out_profiles(name):
    """Dirichlet data only on the two ends: T_HOT + 20 cos(pi s) and T_COLD + 10 cos(2 pi s),
    420 and 380 K at the hot end's extremes, 310 K at both of the cold end's; every other
    side that no piece shares is insulated."""
    b, L = _layout(name)
    vals = [e.bc[1] for e in L.edges if e.other is None and e.bc[0] == "dirichlet"]
    allv = np.concatenate(vals)
    assert np.isclose(allv.max(), 420.0) and np.isclose(allv.min(), 290.0)
    hot = np.concatenate([v for v in vals if v.mean() > 350.0])
    cold = np.concatenate([v for v in vals if v.mean() <= 350.0])
    assert np.isclose(hot.min(), 380.0) and np.isclose(hot.max(), 420.0)
    assert np.isclose(cold.max(), 310.0) and np.isclose(cold.min(), 290.0)
    assert {e.bc[0] for e in L.edges if e.other is None} == {"dirichlet", "insulated"}


@pytest.mark.parametrize("name", BW.NAMES)
def test_dt_is_the_rules(name):
    assert BW.dt_by_rule(name, GH.BENCHMARK_N) == BW.DT[name]


def test_the_time_step_rule():
    assert BW.rule_dt([1.0 / 128.7]) == 100.0
    assert BW.rule_dt([1.0 / 150.0]) == 200.0          # log10 150 is nearer 200 than 100
    assert BW.rule_dt([1.0 / 7.14]) == 10.0
    assert BW.rule_dt([1.0, 2.0, 1.0 / 450.0]) == 1.0   # the median, not the mean


def test_the_step_is_one_backward_euler_step_from_300_k():
    b = BW.build("wall-2", 16, "step")
    for p in b.problems:
        alpha = p.kappa_ref / (dict(steel=7850.0 * 490.0, copper=8960.0 * 385.0)[p.label.split()[0]])
        assert math.isclose(p.fo, alpha * BW.DT["wall-2"] / p.meta["area"], rel_tol=1e-12)
        assert np.allclose(p.g, bm.T_OLD / p.fo)
    lowered = BW.datum_shift(b, GH.DATUM_SHIFT_K)
    assert all(np.all(p.g == 0.0) for p in lowered.problems)


def test_benchmarks_is_untouched():
    assert bm.NAMES == ("bend-3", "s-channel", "insert-round")
    with pytest.raises(KeyError):
        bm.build("wall-2")
    assert BW._finish is bm._finish and BW._held_out_bc is bm._held_out_bc and BW.Bench is bm.Bench


def test_no_expert_but_the_classical_reads_the_new_problems():
    src = open(REGISTER, encoding="utf-8").read()
    assert not re.search(r"\b(LearnedExpert|load_expert|KernelExpert|zero_heads|without_networks)\b", src)
    assert "hs.ClassicalExpert()" in src


def test_the_new_problems_were_measured_classically_and_are_meetable():
    assert os.path.exists(RECORD), "run scripts/chartop_register_hybrid.py"
    rec = json.load(open(RECORD, encoding="utf-8"))
    assert rec["experts_run"] == ["classical"]
    assert [(r["name"], r["problem"]) for r in rec["problems"]] == list(GH.NEW_PROBLEMS)
    assert rec["dropped"] == [] and all(r["floor_meets_e4"] for r in rec["problems"])
    assert rec["refusals"] == 0 and rec["fo_within_training"] and rec["dt_is_the_rules"]
    assert all(rec["control_picks_equal_section_11_14"].values())
    for r in rec["problems"]:
        for frame in ("registered", "lowered"):
            assert r["frames"][frame]["truncation_energy_rel"] < GH.E4_ENERGY
        b, L = _layout(r["name"], 16, r["problem"])
        assert r["pieces"] == L.P and r["picks"] == GH.rule_pieces(L)


def test_the_record_is_of_these_problems():
    """The record's fingerprint of the cheapest problem is the module's, so a layout
    changed after the measurement fails here."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import chartop_register_prior as RP
    rec = json.load(open(RECORD, encoding="utf-8"))
    r = [r for r in rec["problems"] if (r["name"], r["problem"]) == ("wall-2", "steady")][0]
    assert RP.fingerprint(BW.build("wall-2", GH.BENCHMARK_N, "steady")) == r["fingerprint"]


def test_the_price_and_its_judges():
    assert math.isclose(GH.price(1.0, 2.0, 0.5), 0.6)
    prices = {k: [0.8, 0.7, 0.9] for k in GH.NEW_PROBLEMS}
    assert math.isclose(GH.pr_measure(prices), 0.8)
    assert GH.judge_pr(prices)["met"]
    prices[GH.NEW_PROBLEMS[0]] = [0.95, 0.99, 0.97]
    m = GH.pr_measure(prices)
    assert math.isclose(m, math.exp((11 * math.log(0.8) + math.log(0.97)) / 12))
    assert GH.judge_pr({k: [0.91] * 3 for k in GH.NEW_PROBLEMS})["met"] is False
    assert GH.judge_pr({k: [0.90] * 3 for k in GH.NEW_PROBLEMS})["met"] is True
    with pytest.raises(ValueError):
        GH.judge_pr({k: [0.8, 0.8] for k in GH.NEW_PROBLEMS})
    ok = {"plugged_start": True, "plugged_end": True, "power_events": [], "other_core_max": 0.05,
          "classical_ms_per_piece": {"wall-2 steady": 41.0}}
    assert GH.run_counts(ok) == {"counts": True, "failed": []}
    for change in ({"plugged_end": False}, {"power_events": [506]}, {"other_core_max": 0.65},
                   {"classical_ms_per_piece": {"wall-2 steady": 61.0}}):
        assert GH.run_counts(dict(ok, **change))["counts"] is False
    assert GH.run_counts(dict(ok, power_events=[41]))["counts"] is True


def test_e4h_needs_every_new_problem():
    v = {k: {"met": True} for k in GH.NEW_PROBLEMS}
    assert GH.judge_e4h(v)["met"] and GH.judge_e4h(v)["count"] == "12 of 12"
    v[("plate-insert", "steady")] = {"met": False}
    out = GH.judge_e4h(v)
    assert not out["met"] and out["not_met"] == ["plate-insert steady"]
    with pytest.raises(ValueError):
        GH.judge_e4h({k: {"met": True} for k in GH.NEW_PROBLEMS[:-1]})


def test_the_text_names_the_gates_choices_predictions_and_outcomes():
    t = GH.text()
    for s in ("E4-H", "PR", "<= 0.9", "lowered by 300 K", *GH.SETTLED.values(), *GH.PREDICTIONS,
              *GH.OUTCOMES.values()):
        assert s in t
    assert len(GH.PREDICTIONS) == 5 and set(GH.OUTCOMES) == {"both gates met", "E4-H not met",
                                                              "E4-H met, PR not met"}
