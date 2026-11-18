"""P5: Single and Compare on the pieces (integrated-ui-plan §6, §8).

Done-when: a compare run of TL-s0 against the classical full domain on bend-3 shows the
same accuracy numbers the two-level record holds for that problem.  The record is
`out/chartop/evaluation-twolevel-20261111-004223-accuracy.json`, its benchmarks as the
workbench defines them (``benchmarks_seen``), bend-3 steady, the registered frame; it
was run with 8 torch threads on this machine, as this test sets.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest  # noqa: E402

from atlas.studio import experts as X  # noqa: E402
from atlas.studio.session import Session  # noqa: E402
from atlas.workbench.spec import example_case  # noqa: E402


def _session(key, **cut):
    s = Session(spec=example_case(key), key="x")
    s.pieces_cut(None, cut.pop("params", {}), cut.pop("n", 64), blocking=True)
    assert s.pieces_job.status == "done", s.pieces_job.error
    return s


def _no_raw_seconds(obj, path="result"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert not (k.endswith("_s") or "seconds" in k or k in ("times", "mean", "median")), f"{path}.{k}"
            _no_raw_seconds(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for v in obj:
            _no_raw_seconds(v, path)


def test_tl_s0_against_the_full_domain_on_bend3_holds_the_records_numbers():
    rows = {r["key"]: r for r in X.registry()}
    if not rows["TL-s0"]["offered"]:
        pytest.skip(f"TL-s0 is not offered here: {rows['TL-s0'].get('why')}")
    import torch
    torch.set_num_threads(8)
    rec = json.load(open(os.path.join(X.CHARTOP, "evaluation-twolevel-20261111-004223-accuracy.json"),
                         encoding="utf-8"))
    want = next(b for b in rec["benchmarks_seen"] if b["name"] == "bend-3" and b["problem"] == "steady")
    want = want["frames"]["registered"]["runs"]["TL-s0"]
    s = _session("bend-3", params={"n_r": 3, "n_th": 5})
    run = s.start_corun("compare", {"method": "full", "solver": "classical"},
                        {"method": "chart", "solver": "TL-s0"}, blocking=True)
    assert run.status == "done", run.error
    acc = run.result["accuracy"]
    assert acc["energy_rel"] == pytest.approx(want["energy_rel"], rel=1e-9)
    assert acc["max_err"] == pytest.approx(want["max_err"], rel=1e-9)
    assert acc["min_eig"] == pytest.approx(want["min_eig"], rel=1e-9)
    assert run.result["e4"]["passed"] is True
    assert run.result["price"]["ratio"] > 0 and run.result["price"]["turns"] == 5
    assert run.result["certificate"]["charts"] == [15, 15]
    assert run.result["seams"]["max_jump"] < 1e-9
    _no_raw_seconds(run.result)
    fr = run.frame(0)
    assert set(fr) == {"A", "B", "difference"} and fr["A"]["nx"] > 0


def test_single_classical_pieces_against_the_full_domain_is_the_floor():
    s = _session("bend-3", params={"n_r": 3, "n_th": 5}, n=32)
    run = s.start_corun("single", {"method": "chart", "solver": "classical"}, None, blocking=True)
    assert run.status == "done", run.error
    assert run.result["accuracy"]["energy_rel"] < 1e-9           # bend-3's traces fit 16 modes
    assert "price" not in run.result


def test_a_transient_compare_marches_each_window_from_its_own_field():
    """Three steps, each window from its own previous field: the run's error is the one a
    march by hand gives (the classical pieces at the experts' 16 port modes against the
    monolithic solve; 16 modes are a floor, not round-off, when a Dirichlet corner puts a
    steep layer into a seam's trace)."""
    from dataclasses import replace
    import numpy as np
    from atlas.chartop import host as hs
    from atlas.studio import pieces as P
    s = Session(spec=example_case("wall-2"), key="x")
    s.edit_ops([{"op": "set", "path": ["run", "mode"], "value": "transient"},
                {"op": "set", "path": ["run", "steps"], "value": 3},
                {"op": "set", "path": ["run", "macro_dt"], "value": 100.0}], "transient")
    s.pieces_cut("grid", {"cols": 2, "rows": 1}, 32, blocking=True)
    assert s.pieces_job.status == "done", s.pieces_job.error
    run = s.start_corun("compare", {"method": "full"}, {"method": "chart", "solver": "classical"}, blocking=True)
    assert run.status == "done", run.error
    assert run.steps == 3 and len(run.frames) == 3
    assert [r["step"] for r in run.result["series"]] == [1, 2, 3]
    probs = P.problems(s.cut, s.spec)
    L = hs.Layout(probs, P.make_bc(s.spec), kind=P.KIND, m=P.M)
    ua = ub = None
    for _ in range(3):
        L.problems = probs if ua is None else [replace(p, g=u.reshape(p.n + 1, p.n + 1) / p.fo) for p, u in zip(probs, ua)]
        ua = [np.asarray(f).ravel() for f in hs.monolithic(L).fields]
        L.problems = probs if ub is None else [replace(p, g=u.reshape(p.n + 1, p.n + 1) / p.fo) for p, u in zip(probs, ub)]
        ub = [np.asarray(f).ravel() for f in L.solve(hs.ClassicalExpert()).fields]
    mx = max(float(np.abs(a - b).max()) for a, b in zip(ua, ub))
    assert run.result["accuracy"]["max_err"] == pytest.approx(mx, rel=1e-9)


def test_a_refused_expert_is_named_before_anything_runs():
    s = _session("wall-2", params={"cols": 2, "rows": 1}, n=32)
    run = s.start_corun("compare", {"method": "full"}, {"method": "chart", "solver": "H2r"}, blocking=True)
    assert run.status == "failed" and "H2r refused" in run.error
