"""TL's price protocol (`scripts/chartop_price_twolevel.py`, gate PR of `gates_twolevel`): its
problems, its arms on one problem, a smoke dry run in subprocesses with R2's checks (BLAS
threads fixed, AC and sleep sampled, per-problem counting, the A/A arm), and its refusals.
Small resolutions and seeded random weights only; no registered number is read."""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: F401
import pytest

pytest.importorskip("torch")
pytest.importorskip("threadpoolctl")
pytest.importorskip("psutil")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import chartop_price_twolevel as PT  # noqa: E402
from atlas.chartop import gates_squares as GS, gates_twolevel as GT  # noqa: E402


def test_the_problems():
    specs = PT.pr_specs()
    assert specs[:6] == [("held-out", n, p) for n, p in GT.HELD_OUT]
    assert specs[6:] == [("squares", str(s), p) for s, p in GS.new_problems()] and len(specs) == 22
    assert len(PT.dry_run_specs()) == 12 and all(s[0] == "workbench" for s in PT.dry_run_specs())
    assert PT.PRICE_ORDER[0] == "classical" and "classical again" in PT.PRICE_ORDER
    assert PT.DRY_RUN_WEIGHTS_SHA256 == GT.INIT_SHA256


def test_one_problems_arms_and_prices():
    arms = PT.make_arms(None)
    assert arms["TL"].n_c == GT.N_C and arms["TL"].corners.name == "B-n/8"
    assert arms["TL0"].n_c == GT.N_C and arms["TL0"].corners.name == "B-n/8"
    b = PT.build_problem(("workbench", "plate-hole", "steady"), 32)      # n_c = 16 < n: corners added
    r = PT.price_problem(b, arms, reps=1)
    for a in PT.PRICED:
        assert np.isfinite(r[f"price_{a}"]) and r[f"price_{a}"] > 0, a
    assert set(r["seconds"]) == set(PT.PRICE_ORDER) and r["pieces"] == b.layout(GT.KIND, GT.M_PORTS).P
    assert arms["TL"].extra_columns > 0                   # the hole's ring pieces have non-right corners


def test_a_held_out_problem_builds():
    b = PT.build_problem(("held-out", "bend-3", "steady"), 16)
    assert b.layout(GT.KIND, GT.M_PORTS).P > 0


def test_a_smoke_dry_run(tmp_path):
    out = str(tmp_path)
    assert PT.main(["--dry-run", "--problems", "workbench:wall-2:steady", "squares:7000000:step",
                    "--n", "16", "--reps", "1", "--runs", "2", "--out", out]) == 0
    res_path = [f for f in os.listdir(out) if f.startswith("dryrun-twolevel-") and f.endswith(".json")]
    assert len(res_path) == 1
    res = json.load(open(os.path.join(out, res_path[0]), encoding="utf-8"))
    assert res["passes"] is None                           # not the registered settings: never "may be pinned"
    assert len(res["runs"]) == 2 and res["weights_sha256"] is None
    for r in res["runs"]:
        assert GS.blas_threads_hold(r["thread_pools"]) and r["torch_threads"] == GS.PRICE_THREADS
        assert set(r["problems"]) == {"workbench wall-2 steady", "squares 7000000 step"}
        for v in r["problems"].values():
            assert all(v[f"price_{a}"] > 0 for a in PT.PRICED)
    assert set(res["first_counted"]) == set(PT.PRICED) and res["aa"] is not None


def test_the_dry_run_refuses_other_weights(tmp_path):
    fake = tmp_path / "best.pt"
    fake.write_bytes(b"not S's weights")
    with pytest.raises(SystemExit):
        PT.main(["--dry-run", "--weights", str(fake), "--problems", "workbench:wall-2:steady", "--n", "16",
                 "--reps", "1", "--runs", "1", "--out", str(tmp_path)])


def test_a_price_run_refuses_free_blas(tmp_path, monkeypatch):
    monkeypatch.setattr(PT.PS, "thread_pools", lambda: [{"user_api": "blas", "num_threads": 24}])
    with pytest.raises(SystemExit):
        PT.price_run(1, 4, 1, [("workbench", "wall-2", "steady")], 16, str(tmp_path), None)
