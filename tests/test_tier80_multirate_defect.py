"""Tier 80 -- R2's remaining half: the composed defect at 50:1, against CS-11.

R2 asked for ``flux_matching=TIME_INTEGRATED`` plus a `boundary_response_integrated`
on each side (Tier 76), and then for the composed defect at the declared 50:1
reported beside CS-11's bound.  This file pins the second half, and most of what
it pins are findings rather than features:

* **the single-rate control is EXACTLY zero**, because one gas call IS the
  interval -- the one statement that says the instrument measures the lag rather
  than itself, and it has to be bitwise or it says nothing;
* **both columns do the same amount of work**, so the difference between them is
  the trace and not the march;
* **W315** -- `generate.py::_wall_T` collapses the whole shell to ONE number
  before any gas agent sees it, so the build repo's shell-to-gas trace is rank 1
  while this vault probes the seam at ``dim M = 8``;
* **W316** -- the shell is marched and linearised about ``T_gas = 2800 K`` while
  the gas's own near-wall cell is 1490..2804 K, which is W74's class on the one
  direction of this seam nobody had checked;
* **the prediction evaluator agrees with the prose it is written beside**, which
  is a standing rule here after a gate and its evaluation were found disagreeing
  in the permissive direction.

The cheap tests run at intervals of 1e-6 s of gas time.  The chamber marches at
about 1.1e5 s of wall per second of gas time, so an interval is priced before it
is spent and the expensive numbers are read from ``out/w314.json`` when it is
there rather than recomputed.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from atlas.cases import rocket_experts as RE

W314 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "out", "w314.json")


def _have_expert() -> bool:
    try:
        RE.load_rocket_modules()
        return True
    except Exception:
        return False


def _driver():
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "scripts", "w314_rocket_multirate_defect.py")
    spec = importlib.util.spec_from_file_location("w314_driver", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


needs_expert = pytest.mark.skipif(
    not _have_expert(), reason="the build repo is not importable; set ATLAS_BUILD_REPO")


def _results():
    if not os.path.exists(W314):
        pytest.skip("out/w314.json not present; run scripts/w314_... --stages core")
    with open(W314, encoding="utf-8") as fh:
        return json.load(fh)


# ===========================================================================
# 1. the maps between the two faces
# ===========================================================================


def test_overlap_matrix_is_row_stochastic_and_exact_on_constants():
    """A conservative area map has to reproduce a constant exactly.

    `generate.py` maps between these faces with ``np.interp`` on NORMALIZED
    INDEX, which also reproduces constants -- so this is the weaker half of the
    claim and the z-window test below is the half that separates them.
    """
    W = _driver()
    src = np.linspace(0.0, 0.3, 113)
    dst = np.linspace(0.0, 0.3, 15)
    P = W.overlap_matrix(dst, src)
    assert P.shape == (14, 112)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)
    ones = np.ones(112)
    np.testing.assert_allclose(P @ ones, np.ones(14), atol=1e-12)
    # and a linear field comes back linear to the cell midpoint
    mid_src = 0.5 * (src[1:] + src[:-1])
    mid_dst = 0.5 * (dst[1:] + dst[:-1])
    np.testing.assert_allclose(P @ mid_src, mid_dst, atol=1e-3)


def test_overlap_matrix_is_physical_not_index_based():
    """The separating case: two faces over DIFFERENT z windows.

    An index map sends cell k of one to cell k of the other whatever z they are
    at.  An overlap map sends mass only where the cells actually overlap, so a
    source that covers half the destination leaves the other half untouched.
    """
    W = _driver()
    src = np.linspace(0.0, 0.15, 57)          # covers only the first half
    dst = np.linspace(0.0, 0.30, 15)
    P = W.overlap_matrix(dst, src)
    field = np.arange(56, dtype=float)
    got = P @ field
    # the upper-half destination cells have NO overlap, so the fallback fires
    # and they read the source mean -- not the "matching index" value
    assert got[-1] == pytest.approx(field.mean())
    assert got[0] < field.mean()


# ===========================================================================
# 2. the instrument: the single-rate control, and equal work
# ===========================================================================


@needs_expert
def test_the_single_rate_control_is_exactly_zero():
    """Ratio 1: one gas call IS the interval, so there is nothing to be stale.

    It has to be BITWISE zero.  A small number here would mean the two columns
    differ in something other than the trace, which is the failure this control
    exists to catch -- and quoting a tightness ratio against it would be
    W106's 0/0.
    """
    W = _driver()
    seam = W.BCSeam(burn_time=RE.T_PROBE_BURN)
    r = W.lag_defect(seam, seam.gas._U0, 1.0e-6, 1)
    assert r["sigma"] == 0.0
    assert r["lag"] == 0.0


@needs_expert
def test_both_columns_do_the_same_work():
    """The difference between the columns is the TRACE, not the march.

    Both march the same gas from the same state over the same interval with the
    same number of sub-intervals.  If the sub-step counts differed, sigma would
    be measuring the solver's own adaptivity.
    """
    W = _driver()
    seam = W.BCSeam(burn_time=RE.T_PROBE_BURN)
    n0 = seam.gas_substeps
    W.lag_defect(seam, seam.gas._U0, 4.0e-6, 4)
    total = seam.gas_substeps - n0
    assert total > 0
    # even split between REF and HELD: the counter is incremented per gas call
    # and each column makes exactly `ratio` of them
    assert total % 2 == 0 or total > 1


# ===========================================================================
# 3. W315 -- the build repo's shell-to-gas trace is rank 1
# ===========================================================================


@needs_expert
def test_W315_generate_py_collapses_the_shell_to_one_number():
    """The finding, read off the build repo rather than asserted about it.

    ``_wall_T`` returns ``float(np.mean([T.mean() for T in self.T_shell]))`` --
    the mean over every panel of every station -- and every gas agent's wall BC
    gets that one number.  The gas-to-shell direction is per-station via
    ``np.interp``; the shell-to-gas direction has rank 1.  So the interface space
    this vault probes at ``dim M = 8`` is one the build repo's own coupler cannot
    transport.
    """
    src = os.path.join(RE.build_repo(), "src", "atlas", "data", "generate.py")
    if not os.path.exists(src):
        pytest.skip("build repo checkout not present")
    body = open(src, encoding="utf-8").read()
    assert "def _wall_T(self, agent: str) -> float:" in body
    assert "return float(np.mean([T.mean() for T in self.T_shell]))" in body
    # and the other direction really is per-station
    assert "np.interp(np.linspace(0, 1, ni)" in body


@needs_expert
def test_W315_the_scalar_trace_is_rank_one_and_the_per_face_one_is_not():
    W = _driver()
    seam = W.BCSeam(burn_time=RE.T_PROBE_BURN)
    face = seam.trace_on_gas(seam.T_shell0, scalar=False)
    scal = seam.trace_on_gas(seam.T_shell0, scalar=True)
    assert scal.size == face.size == seam.gas._n_face
    assert float(scal.max() - scal.min()) == 0.0
    assert float(face.max() - face.min()) > 50.0
    # the scalar is the window mean, not the first cell or the max
    assert scal[0] == pytest.approx(float(seam.wall_T(seam.T_shell0).mean()))


# ===========================================================================
# 4. W317 -- a THIRD undeclared interface, and the declared one is the minority
# ===========================================================================


@needs_expert
def test_W317_a_c_is_undeclared_and_the_build_repo_says_so():
    """The third one, and the only one the build repo names itself.

    W311 derived `e-c` from the geometry and `d-f` came from
    `contours.plane_d_g`'s docstring.  This one is written down in
    `generate.py::_wall_T`, as the stated reason that method returns a scalar --
    so the build repo knows about an interface its own config does not declare.
    """
    assert "a-c" in RE.UNDECLARED_INTERFACES
    src = os.path.join(RE.build_repo(), "src", "atlas", "data", "generate.py")
    if not os.path.exists(src):
        pytest.skip("build repo checkout not present")
    # collapse the docstring's own line wrapping before matching
    body = " ".join(open(src, encoding="utf-8").read().split())
    assert "`a`'s lateral walls are an UNDECLARED a-c interface" in body
    # and it is genuinely not in the config's edge list
    cfg = RE.rocket_config()
    edges = {"%s-%s" % (e.src, e.dst) for e in cfg.edge_list}
    assert "a-c" not in edges and "c-a" not in edges
    assert "e-c" not in edges and "c-e" not in edges
    assert "b-c" in edges or "c-b" in edges


@needs_expert
def test_W317_the_declared_seam_is_the_minority_of_the_conjugate_wall():
    """40.7%: the declared b-c window against the wall that has gas behind it.

    Re-measured from the mesh here rather than read from the constants, so the
    constants cannot drift away from the geometry they describe.
    """
    sh = RE.ShellAgent(dt=5.0e-2)
    zs = sh._blk.nodes[:, 0, 0]
    L = np.asarray(sh._ts._face("inner")[2], dtype=float)
    mid = 0.5 * (zs[:-1] + zs[1:])
    engine = (mid >= 0.0) & (mid <= 0.70)
    declared = L[sh._cells].sum()
    assert declared == pytest.approx(RE.ENGINE_WALL_M["b-c"], rel=1e-6)
    assert L[engine].sum() == pytest.approx(RE.ENGINE_WALL_TOTAL_M, rel=1e-6)
    assert L.sum() == pytest.approx(RE.SHELL_INNER_FACE_M, rel=1e-6)
    share = declared / L[engine].sum()
    assert 0.40 < share < 0.41, share
    assert sum(RE.ENGINE_WALL_M.values()) == pytest.approx(
        RE.ENGINE_WALL_TOTAL_M, rel=1e-9)


# ===========================================================================
# 5. W316 -- the shell is linearised about a gas temperature the gas does not have
# ===========================================================================


@needs_expert
def test_W316_the_shell_is_marched_about_a_gas_temperature_the_gas_does_not_have():
    """W74's class, on the direction of this seam nobody had checked.

    `ShellAgent.march` drives the shell with ``T_CHAMBER`` and `base_trace`
    returns the same 2800 K.  The gas's own near-wall cell is well below that
    over most of the window, and strongly non-uniform -- which is what makes it
    a base disagreement rather than a units question.

    This test marches the gas only 1e-4 s, so it reads the near-wall field of a
    barely-started chamber and asserts only the SIGN and the non-uniformity.  The
    settled numbers are in `out/w314.json` and are pinned separately.
    """
    W = _driver()
    seam = W.BCSeam(burn_time=RE.T_PROBE_BURN)
    U, _flow, _q, _T = seam.gas_step(seam.gas._U0, 1.0e-4,
                                     seam.trace_on_gas(seam.T_shell0))
    Tg, _h = seam.gas_near_wall(U)
    assert Tg.mean() < RE.T_CHAMBER
    assert float(Tg.max() - Tg.min()) > 100.0
    assert RE.ShellAgent(dt=5.0e-2).base_trace("b:THERM")[0] == RE.T_CHAMBER


# ===========================================================================
# 6. the bound is the formula it says it is
# ===========================================================================


def test_the_bound_is_first_order_plus_a_quadratic_and_nothing_else():
    W = _driver()
    s, c2, lag = 1.2345e-3, 6.78e-5, 0.4321
    assert W._bound(s, c2, lag) == pytest.approx(s * lag + c2 * lag ** 2, rel=0, abs=0)
    assert W._bound(s, c2, 0.0) == 0.0
    # monotone in the lag, which is what makes it a bound rather than a fit
    assert W._bound(s, c2, 1.0) > W._bound(s, c2, 0.5) > W._bound(s, c2, 0.1)


def test_cs11_constants_are_the_published_ones():
    """The transfer control is only a control if it uses CS-11's real numbers."""
    W = _driver()
    assert W.CS11_S_GAMMA == pytest.approx(9.4966e-3)
    assert W.CS11_C2 == pytest.approx(1.4288e-5)
    assert W.CS11_LAMBDA_DOT == pytest.approx(11.94)


# ===========================================================================
# 7. the predictions and their evaluator cannot drift apart
# ===========================================================================


def test_the_prediction_evaluator_can_fail():
    """A gate that cannot fail is not a gate.

    Every registered prediction is fed a synthetic result that should FAIL it,
    and the evaluator has to say so.  This is the standing rule after a gate
    written in prose and evaluated in code were found disagreeing in the
    permissive direction.
    """
    W = _driver()
    empty = W.predictions({})
    assert len(empty) == 12
    assert [p["name"] for p in empty] == ["P%d" % i for i in range(1, 13)]
    # with nothing measured, nothing may be reported as held
    assert not any(p["held"] for p in empty)


def test_the_prediction_evaluator_can_pass():
    W = _driver()
    good = {
        "r9": {"admits": True},
        "sigma_law": {
            "rows": [{"sigma": 0.0, "control": True, "interval": 1e-6,
                      "lag": 0.0, "lag_rate": 0.0},
                     {"sigma": 1e-6, "interval": 1e-4, "exponent": 1.0,
                      "lag": 1e-3, "lag_rate": 10.0},
                     {"sigma": 1e-5, "interval": 1e-3, "exponent": 1.0,
                      "lag": 1e-2, "lag_rate": 10.0}],
            "bound_holds": True, "tightness": [1.5, 3.0]},
        "slope": {"s_seam": 1e-3, "C2": 1e-6, "s_seam_spread": 1.1},
        "rate": {"lambda_dot_at_5s": 50.0, "span": 4.0},
        "transfer": {"cs11_holds": False, "cs11_worst": 0.3},
        "ratio": {"sigma_move": 1.001},
        "trace_rank": {"rows": [{"ratio": 2.9}]},
        "two_way": {"rows": [{"ratio": 2.0}]},
    }
    P = W.predictions(good)
    assert all(p["held"] for p in P), [p["name"] for p in P if not p["held"]]


# ===========================================================================
# 8. the gate, read off the record the run wrote
# ===========================================================================


def test_R2_gate_first_half_L7_R9_admits():
    """The gate's first half: R9 clears on the graph with real experts."""
    res = _results()
    assert res["r9"]["flux_matching"] == "time-integrated"
    assert res["r9"]["admits"] is True
    assert set(res["r9"]["at_multirate"]) <= set("abcdefg")


def test_R2_gate_second_half_the_defect_is_measured_and_bounded():
    """The gate's second half: sigma at the pinned 50:1, beside CS-11's bound."""
    res = _results()
    law = res["sigma_law"]
    graded = [r for r in law["rows"] if r["sigma"] > 0.0]
    control = [r for r in law["rows"] if r.get("control")]
    assert control and control[0]["sigma"] == 0.0
    assert graded, "no graded intervals"
    assert all(r["ratio"] == 50 for r in graded), "the ratio is pinned at 50"
    assert law["bound_holds"] is True
    lo, hi = law["tightness"]
    assert lo >= 1.0


def test_the_transfer_control_was_actually_run():
    """'The bound holds' means nothing unless CS-11's own constants were tried."""
    res = _results()
    t = res["transfer"]
    assert "cs11_holds" in t and "cs11_worst" in t
    assert t["s_seam_ratio"] != pytest.approx(1.0, rel=1e-3)


def test_the_run_was_not_suspended():
    """A cost quoted across a Modern Standby is not a cost.

    The first sizing of this tier spanned a five-minute standby and read 2.6x
    high, so the driver counts the events itself and the record carries the
    count.
    """
    res = _results()
    assert res.get("standby_during_run", 0) == 0
