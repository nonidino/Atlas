"""Tier 89: W214 (the matched shrink), W345 (a coarse competitor across a coupled
seam) and W346 (classical decomposition against the monolith, by rotor count).

Two kinds of test.  The first builds the instruments and checks the property each
measurement rests on -- the transfer operators, the bitwise threaded column, the
map that must reproduce `LoopSolve`.  The second reads the committed records and
pins the numbers the wiki quotes, so a re-run that moves one fails here first.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))


def _record(name):
    path = os.path.join(HERE, "out", name, f"{name}.json")
    if not os.path.isfile(path):
        pytest.skip(f"{path} not present")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# W345's instruments
# ---------------------------------------------------------------------------


def _loop_maps():
    try:
        import w345_coarse_competitor as C
        return C, C.LoopMaps()
    except RuntimeError as exc:                      # the build repo is absent
        pytest.skip(str(exc))


def test_nodal_transfer_reproduces_a_bilinear_field():
    """Injection then bilinear prolongation is exact on a field that is bilinear
    in the node indices -- the property that makes the coarse block's settled
    state (linear through the thickness, uniform along the seam) exact."""
    C, M = _loop_maps()
    ni, nj = M.fine.n_seam + 1, M.fine.nj + 1
    i, j = np.meshgrid(np.arange(ni), np.arange(nj), indexing="ij")
    T = (300.0 + 0.7 * i + 2.5 * j + 0.01 * i * j).ravel()
    back = C.prolong_nodes(C.restrict_nodes(T, M.fine, M.coarse), M.fine, M.coarse)
    assert np.max(np.abs(back - T)) < 1e-10


def test_the_classical_map_is_loopsolve_bit_for_bit():
    """Q1's control, re-run: the script's Phi against LoopSolve's own loop body."""
    C, M = _loop_maps()
    art = {}
    C.persist = lambda a: None                       # do not touch the record
    C.stage_repro(M, art)
    assert art["repro"]["max_abs_diff"] == 0.0


def test_the_coarse_competitor_is_a_different_map():
    """A control on the control: C is not Phi, so a bitwise repro of Phi says
    nothing about C, and C's own fixed point has to be measured (stage bias)."""
    C, M = _loop_maps()
    w = M.w0()
    for _ in range(3):
        w = M.Phi(w)
    assert np.max(np.abs(M.C(w) - M.Phi(w))) > 1e-6


# ---------------------------------------------------------------------------
# W346's instrument
# ---------------------------------------------------------------------------


def test_threaded_column_is_the_serial_column_bit_for_bit():
    """The whole of W346's parallel arm rests on this: windows split across
    threads, every chunk forced to the batch's sub-step count, return exactly
    what the serial batch returns.  Local time stepping must NOT -- it is a
    different scheme, and a test that could not tell them apart would be blind."""
    try:
        import w346_rotor_count_speed as W
        R = W.Rung(2, threads=(2,))
    except RuntimeError as exc:
        pytest.skip(str(exc))
    u, v = R.freestream()
    for _ in range(3):
        u, v, _ = R.F(u, v)
    a = R.E(u.copy(), v.copy())
    b = R.Ep(u.copy(), v.copy(), 2)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    for p in R.pools.values():
        p.shutdown()


def test_umax_proxy_changes_only_the_substep_count():
    import w346_rotor_count_speed as W

    class B:
        def amax(self, x):
            return 1.0

        def hypot(self, a, b):
            return a

    p = W._Umax(B(), 7.0)
    assert p.amax(None) == 7.0 and p.hypot(3, 4) == 3


# ---------------------------------------------------------------------------
# the records the wiki quotes
# ---------------------------------------------------------------------------


def test_w214_controls_and_arms():
    a = _record("w214")
    arms = a["arms"]
    assert arms["N6"]["P_half"]["phi_calls"] == 99          # Tier 48, reproduced
    assert arms["N12"]["P_half"]["phi_calls"] == 222
    assert arms["N6"]["P_match"]["phi_calls"] == 82
    assert arms["N12"]["P_match"]["phi_calls"] == 270       # Tier 48's alpha = 0 arm
    assert arms["N6"]["Z_match"]["phi_calls"] == 126
    assert abs(a["probe"]["N6"]["alpha_P"] - 0.1431) < 1e-4
    assert a["probe"]["N12"]["alpha_P"] < 0.0                # the sign change
    assert all(v == 0.0 for v in a["probe"]["N6"]["reproduces_tier50_probe"].values())
    # the mechanism: the weakly shrunk arms' residual RISES within four iterations
    r = [row["residual"] for row in arms["N12"]["P_match"]["rows"]]
    assert r[2] > r[1] > r[0]
    ev = a["evaluation"]
    assert ev["M2"]["verdict"] == "FAILED" and ev["M6"]["verdict"] == "FAILED"


def test_w345_competitor_is_correct_and_not_cheap():
    a = _record("w345")
    assert a["repro"]["max_abs_diff"] == 0.0
    assert a["reference"]["cold_calls"] == 7411
    assert all(a["dc"][k]["inside_certificate"] for k in ("C_m40", "C_m400", "C_m4000"))
    c = a["cost"]["ratio_C"]
    best = min(a["dc"][k]["phi_calls"] + c * a["dc"][k]["psi_calls"]
               for k in ("C_m40", "C_m400", "C_m4000"))
    assert best > a["reference"]["cold_calls"]              # dearer than the cold march
    f = a["fsi"]
    assert f["loader_control"] is True
    assert f["reference"]["cold_calls"] == 901
    assert f["bias"]["load_rel_err"] < 0.005
    cf = f["cost"]["ratio_C"]
    best_f = min(f["dc"][k]["phi_calls"] + cf * f["dc"][k]["psi_calls"]
                 for k in ("C_m40", "C_m400"))
    assert best_f > f["reference"]["cold_calls"]
    # the seam is not the cause: coupled and frozen within a handful of calls
    assert abs(f["dc"]["C_m40"]["phi_calls"] - f["dc"]["C0_m40"]["phi_calls"]) <= 4


def test_cs12_settles():
    path = os.path.join(HERE, "out", "w345", "fsi_long.json")
    if not os.path.isfile(path):
        pytest.skip("fsi_long.json not present")
    with open(path, encoding="utf-8") as fh:
        z = json.load(fh)
    res = [r for _k, r in z["residual_every_25"]]
    assert res[-1] < 1e-5 * res[0]                          # 2.9e-3 -> 1.3e-9
    assert z["load_pp_by_quarter"][-1] < 1e-5


def test_w346_decomposition_speed_by_rotor_count():
    a = _record("w346")
    rungs = a["rungs"]
    assert a["control_maps_E"] is True
    assert all(all(r["bitwise"].values()) for r in rungs.values())
    # serial decomposition is never faster than the monolith, in either draw
    for k, r in rungs.items():
        assert r["cost"]["ratio"]["E"] > 0.9
        if "cost_replicate" in r:
            assert r["cost_replicate"]["ratio_min"]["E"] > 0.9
    # the threaded column beats the monolith by at least 3.4x from 5 to 21 rotors
    # in BOTH draws (the slower draw's best arm), and loses below 5 rotors
    for k in ("N12", "N24", "N48"):
        first = min(v for t, v in rungs[k]["cost"]["ratio"].items() if t.startswith("Ep"))
        second = min(v for t, v in rungs[k]["cost_replicate"]["ratio_min"].items()
                     if t.startswith("Ep"))
        assert max(first, second) < 1 / 3.4          # 3.499x at 5 rotors, draw 2
    for k in ("N2", "N6"):
        assert min(v for t, v in rungs[k]["cost"]["ratio"].items() if t.startswith("Ep")) > 0.7
    # accuracy: farm power within 10% of the monolith at every marched rung
    for k, r in rungs.items():
        if "accuracy" in r:
            assert abs(r["accuracy"]["E"]["power_rel_diff"]) < 0.10
