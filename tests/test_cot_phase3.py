"""The coolant plan's Phase 3 scripts: the capacity probe's rule and verdict, the checks'
pieces (refinement, coverage, the synthetic experts, the rank correlation), the evaluation
layouts kept apart from the development ones, and the registration still a draft."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import cot_phase3_capacity as cap  # noqa: E402
import cot_phase3_checks as chk  # noqa: E402
from atlas.chartop import gates_cot as GC  # noqa: E402
from atlas.chartop import transport_data as td  # noqa: E402
from atlas.studio import cot_eval_layouts as EL  # noqa: E402
from atlas.studio import cot_layouts as CL  # noqa: E402
from atlas.studio import document as D  # noqa: E402


def test_the_probe_criteria_were_fixed_before_it_ran():
    assert cap.CRITERIA["cap_fraction"] == 0.30 and cap.CRITERIA["steps"] == 3000
    assert "the owner" in cap.CRITERIA["owner"] and cap.PE_MIN == pytest.approx(180.0)
    assert GC.CAP_FRACTION == cap.CRITERIA["cap_fraction"] and GC.CAP_STEPS == cap.CRITERIA["steps"]


def test_the_probe_rule_picks_fast_flowing_pieces_of_each_kind():
    seeds, metas = cap.select(n=32, per_kind=1, pe_min=0.5 * td.pe_cap(32))
    assert [m["kind"] for m in metas] == list(cap.KINDS)
    assert all(m["pe"] >= 0.5 * td.pe_cap(32) and m["n"] == 32 for m in metas)
    assert seeds == cap.select(n=32, per_kind=1, pe_min=0.5 * td.pe_cap(32))[0]


def test_the_verdict_reads_every_group():
    row = {"err": {"trial hats": {"median": 0.2}, "adjoint hats": {"median": 0.31}, "particular": {"median": 0.1}}}
    v = cap.verdict(row, 0.30)
    assert not v["met"] and v["groups_over_the_bar"] == ["adjoint hats"]
    row["err"]["adjoint hats"]["median"] = 0.3
    assert cap.verdict(row, 0.30)["met"]


def test_refinement_is_exact_for_bilinear_data():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((5, 5, 2))
    r = chk.refine(a)
    assert r.shape == (9, 9, 2) and np.array_equal(r[::2, ::2], a)
    # a bilinear function is reproduced at every new node
    t = np.linspace(0, 1, 5)
    X, Y = np.meshgrid(t, t)
    f = 1 + 2 * X - 3 * Y + 4 * X * Y
    t2 = np.linspace(0, 1, 9)
    X2, Y2 = np.meshgrid(t2, t2)
    assert np.allclose(chk.refine(f), 1 + 2 * X2 - 3 * Y2 + 4 * X2 * Y2, atol=1e-14)


def _manifest(n32, n64, n128, solid=0.15):
    N = n32 + n64 + n128
    cov = {"count": N, "by_n": {"32": n32, "64": n64, "128": n128},
           "by_kind": {"solid": int(solid * N), "through": int(0.55 * N), "turn": int(0.2 * N),
                       "wavy": N - int(solid * N) - int(0.55 * N) - int(0.2 * N)},
           "by_family": {"spline-quad": int(0.8 * N), "conformal-map": N - int(0.8 * N)},
           "steady": int(0.3 * N), "fo_decades": {"-2": 100, "-1": 100, "0": 100, "1": 100}}
    fl = N - cov["by_kind"]["solid"]
    cov["by_profile_flowing"] = {"laminar": int(0.6 * fl), "plug": int(0.2 * fl), "blend": fl - int(0.8 * fl)}
    cov["pe_tenths_of_cap_flowing"] = {k: [100] * 10 for k in ("32", "64", "128")}
    return {"coverage": cov}


def test_check1_passes_the_declared_mix_and_fails_a_short_one():
    assert chk.check1(_manifest(2500, 3500, 4000))["passed"]
    bad = chk.check1(_manifest(3000, 3500, 2900))
    assert not bad["passed"] and "fewer n = 128 pieces than n = 32" in bad["fails"]
    m = _manifest(2500, 3500, 4000)
    m["coverage"]["pe_tenths_of_cap_flowing"]["128"][9] = 10
    assert not chk.check1(m)["passed"]


def test_the_synthetic_experts_and_the_rank_correlation():
    s = chk.synthetic_set()
    assert len(s) == 40 and s[0][0] == "TL0" and s[1][0] == "classical"
    assert all(len(a) == 9 and (a >= 0).all() and (a <= 1).all() for _, a in s)
    assert chk.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert chk.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_check3_reads_the_control_from_phase2():
    r = chk.check3()
    assert r["passed"] and len(r["rows"]) == 6 and all(x["arm"] == "TL0-16" for x in r["rows"])


def test_the_evaluation_layouts_are_apart_from_the_development_ones():
    assert not set(EL.NAMES) & set(CL.NAMES)
    assert all(EL.SPEEDS[n][t] for n in EL.NAMES for t in EL.TIERS)
    a = D.dumps(*EL.build("eval-arch", "slow"))
    assert a == D.dumps(*EL.build("eval-arch", "slow"))
    assert a != D.dumps(*CL.build("dev-winding", "slow"))
    assert GC.EVAL_PROBLEMS == tuple((n, t) for n in EL.NAMES for t in EL.TIERS)


def test_the_floors_script_refuses_a_coarse_or_learned_arm(tmp_path, monkeypatch):
    import cot_phase3_eval_layouts as fl
    monkeypatch.setitem(sys.modules, "atlas.chartop.transport_torch", sys.modules.get("atlas.chartop.transport_torch")
                        or object())
    with pytest.raises(RuntimeError, match="only the classical"):
        fl.floors(str(tmp_path / "x"), str(tmp_path / "y"), names=())


def test_the_registration_is_still_a_draft():
    assert GC.REGISTERED is None
    assert [g.key for g in GC.GATES] == ["E4", "CL-A", "CAP", "SEEDS", "CX", "K", "D", "SAFE", "PR", "AMORT"]
    assert {g.kind for g in GC.GATES if g.key in ("PR", "AMORT")} == {"reported"}
    assert GC.n_c(32) == 16 and GC.n_c(128) == 16
