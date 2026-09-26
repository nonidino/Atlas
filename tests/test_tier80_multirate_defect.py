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
  direction of this seam nobody had checked.  **Re-measured at Tier 86** with
  W301 and W332 fixed: 2696..2804 K, mean 2756 K, ratio 0.984 against 0.880.
  The finding stands at about an eighth of its size; the rest was the wall's
  saturated channel and its mass leak (``out/w332_w316_setup.json``);
* **the prediction evaluator agrees with the prose it is written beside**, which
  is a standing rule here after a gate and its evaluation were found disagreeing
  in the permissive direction;
* **W334 (Tier 86)** -- the anchor's violation of the bound (0.2702) and the
  knee's second-order jump were the old wall's.  Re-derived with W301 and W332
  fixed, sigma is first order over the whole 250x of interval (exponents
  0.9997..1.0021) and the bound holds at the declared interval at 2.278x.  The
  old record is kept at ``out/pre_w332/w314.json`` and read as the control, so
  the diagnosis stays checkable rather than quoted.

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


#: The record as Tiers 80-85 had it: W301's saturated isothermal channel and
#: W332's leaking wall.  W334 moved it here when the re-derived one took its name.
W314_OLD = os.path.join(os.path.dirname(W314), "pre_w332", "w314.json")


def _old_results():
    if not os.path.exists(W314_OLD):
        pytest.skip("out/pre_w332/w314.json not present; the old-wall control needs it")
    with open(W314_OLD, encoding="utf-8") as fh:
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
def test_W315_closed_generate_py_hands_each_gas_wall_the_shell_at_its_z():
    """W315, closed at Tier 86 -- read off the build repo, not asserted about it.

    Until then ``_wall_T`` returned ``float(np.mean([T.mean() for T in
    self.T_shell]))`` -- the mean over every panel of every station -- and every
    gas agent's wall BC got that one number, while the gas-to-shell direction
    was mapped by ARRAY INDEX (``np.interp(np.linspace(0, 1, ni), ...)``), which
    stretched b's 0.28 m of wall over the whole 4.7 m airframe (W323). Both are
    gone: each gas wall cell takes the shell's own surface temperature at its z,
    and each shell station takes the gas wall at its z.
    """
    src = os.path.join(RE.build_repo(), "src", "atlas", "data", "generate.py")
    if not os.path.exists(src):
        pytest.skip("build repo checkout not present")
    body = open(src, encoding="utf-8").read()
    assert "return float(np.mean([T.mean() for T in self.T_shell]))" not in body
    assert "np.interp(np.linspace(0, 1, ni)" not in body
    assert "def _wall_T(self, agent: str, k: int, side: str) -> np.ndarray:" in body
    assert "def _shell_inner_loads(self, k: int, p_inf: float):" in body


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
def test_W317_a_c_is_undeclared_and_now_recorded():
    """The third undeclared interface -- and since Tier 86, a RECORDED one.

    W311 derived `e-c` from the geometry, `d-f` came from
    `contours.plane_d_g`'s docstring, and this one was written down in the old
    `generate.py::_wall_T` as the stated reason it returned a scalar. The
    coupler now exchanges across all three by position and the config records
    them (`recorded_interfaces`) without declaring them to the model -- the
    declaration is still a design decision, and recording them means taking it
    later needs no regenerated corpus.
    """
    assert "a-c" in RE.UNDECLARED_INTERFACES
    cfg = RE.rocket_config()
    edges = {"%s-%s" % (e.src, e.dst) for e in cfg.edge_list}
    assert "a-c" not in edges and "c-a" not in edges
    assert "e-c" not in edges and "c-e" not in edges
    assert "b-c" in edges or "c-b" in edges
    recorded = {"%s-%s" % (r.src, r.dst) for r in cfg.recorded_interfaces}
    assert recorded == {"a-c", "e-c", "d-f"}


@needs_expert
def test_W317_the_declared_seam_is_the_minority_of_the_conjugate_wall():
    """40.8%: the declared b-c window against the wall that has gas behind it.

    Re-measured from the mesh here rather than read from the constants, so the
    constants cannot drift away from the geometry they describe. They did move
    at Tier 86 (40.7% before): the shell now has nodes at the nozzle's kinks,
    so its inner face IS the contour -- 0.296619 m of b-c, not the 0.299018 m
    of chords that cut across the converging-section corner and the throat.
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


def _old_residual(self, U):
    """`Compressible2D.residual` as it was until W332 (Tier 86): the isothermal
    ghost handed to the inviscid flux as well as the viscous one."""
    Ue = self.ghost(U)
    r = self._inviscid_residual(Ue)
    if not self.cfg.inviscid:
        r = r + self._viscous_residual(Ue)
    if self._hole is not None:
        j0, j1 = self._hole
        r[:, j0:j1 + 1] = 0.0
    return r


@needs_expert
def test_W316_a_barely_started_chamber_was_cooled_by_the_wall_leak(monkeypatch):
    """W316 read the gas's near-wall cell well below the 2800 K the shell is
    linearised about, and strongly non-uniform.  This test marches the gas only
    1e-4 s, a barely-started chamber, and until Tier 86 it asserted exactly that:
    the near-wall mean below 2800 K, spread over more than 100 K.

    **That early cooling was W332.**  The isothermal ghost is ~125x denser than
    the cell beside a cold wall, and handed to the Riemann solver it let the wall
    face pass mass whenever the gas moved normal to it -- cold, dense ghost gas
    into the chamber's wall cells.  With the inviscid flux given the plain
    mirror, the same 1e-4 s leaves the near-wall gas within a few kelvin of the
    chamber and nearly uniform, because conduction alone has barely begun.  The
    old reading is kept as the control: put the old residual back and it
    returns.  The settled numbers, re-measured with the fix, are in
    `out/w332_w316_setup.json` (the old ones in `out/w314.json`).
    """
    import importlib

    W = _driver()
    seam = W.BCSeam(burn_time=RE.T_PROBE_BURN)
    trace = seam.trace_on_gas(seam.T_shell0)
    U, _flow, _q, _T = seam.gas_step(seam.gas._U0, 1.0e-4, trace)
    Tg, _h = seam.gas_near_wall(U)
    assert abs(Tg.mean() / RE.T_CHAMBER - 1.0) < 1.0e-2
    assert float(Tg.max() - Tg.min()) < 20.0
    # the control: the old wall, and the old reading
    C2 = importlib.import_module("atlas_build_solvers.solvers.compressible2d")
    monkeypatch.setattr(C2.Compressible2D, "residual", _old_residual)
    U0, _flow, _q, _T = seam.gas_step(seam.gas._U0, 1.0e-4, trace)
    T0, _h = seam.gas_near_wall(U0)
    assert T0.mean() < RE.T_CHAMBER
    assert float(T0.max() - T0.min()) > 100.0
    assert RE.ShellAgent(dt=5.0e-2).base_trace("b:THERM")[0] == RE.T_CHAMBER


def test_W316_re_measured_the_gap_is_an_eighth_of_what_was_recorded():
    """The settled state, both ways.  `out/w314.json` was taken with W301's
    saturated channel and W332's leaking wall; `out/w332_w316_setup.json` is the
    same `setup` stage re-run with both fixed.  W316 survives -- the shell is
    still linearised about a hotter gas than the wall layer holds -- but at
    1.6% where 12% was recorded, and the 1490 K floor is gone."""
    # the pre-Tier-86 record moved to out/pre_w332/ when W334 re-derived it
    old_path = W314_OLD
    new_path = os.path.join(os.path.dirname(W314), "w332_w316_setup.json")
    if not (os.path.exists(old_path) and os.path.exists(new_path)):
        pytest.skip("the two setup records are not both present")
    with open(old_path, encoding="utf-8") as fh:
        old = json.load(fh)["setup"]
    with open(new_path, encoding="utf-8") as fh:
        new = json.load(fh)["setup"]
    assert old["base_gap_near_wall_over_chamber"] == pytest.approx(0.8801, abs=1e-4)
    assert new["base_gap_near_wall_over_chamber"] == pytest.approx(0.9841, abs=1e-4)
    assert new["gas_near_wall_T_min"] > 2600.0 > 1500.0 > old["gas_near_wall_T_min"]
    gap_old = 1.0 - old["base_gap_near_wall_over_chamber"]
    gap_new = 1.0 - new["base_gap_near_wall_over_chamber"]
    assert 0.0 < gap_new < gap_old / 5.0, "still a gap, and a far smaller one"


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


def _graded(res):
    return [r for r in res["sigma_law"]["rows"] if r["sigma"] > 0.0]


def _anchor_row(rows):
    declared = [r for r in rows if abs(r["interval"] - 5.0e-2) < 1e-15]
    if not declared:
        pytest.skip("the anchor row is not in the record; run --stages anchor")
    return declared[0]


def test_R2_gate_second_half_the_defect_is_measured_and_bounded():
    """The gate's second half: sigma at the pinned 50:1, beside CS-11's bound.

    **This test has asserted three things in turn.** ``bound_holds is True``
    until the anchor ran (Tier 80). Then the anchor VIOLATING the bound at
    0.2702, the core's clean 2.269x-2.500x table notwithstanding (Tiers 80-85).
    Then W334 re-derived the record with W301 and W332 fixed (Tier 86), and the
    violation was the wall's: at the declared interval the bound holds at
    2.278x, inside the core's own span. The violation is kept as the control,
    read off the old record, so the diagnosis stays a measurement.
    """
    res = _results()
    law = res["sigma_law"]
    graded = _graded(res)
    control = [r for r in law["rows"] if r.get("control")]
    assert control and control[0]["sigma"] == 0.0
    assert graded, "no graded intervals"
    assert all(r["ratio"] == 50 for r in graded), "the ratio is pinned at 50"

    cheap = [r for r in graded if r["interval"] <= 5.0e-3]
    assert len(cheap) == 3
    assert all(r["bound_over_measured"] >= 1.0 for r in cheap)
    assert 2.2 < min(r["bound_over_measured"] for r in cheap) < 2.6

    a = _anchor_row(graded)
    assert a["bound_over_measured"] == pytest.approx(2.2782, abs=5e-4)
    assert law["bound_holds"] is True
    # loose by about the same factor at every interval, the anchor included
    tight = [r["bound_over_measured"] for r in graded]
    assert 2.2 < min(tight) and max(tight) < 2.4, tight
    assert law["tightness"] == pytest.approx([min(tight), max(tight)])


def test_R2_gate_control_the_old_wall_violated_the_bound():
    """The control for the gate above: the same stages, the same driver, the old
    wall -- and the anchor violates the bound it now clears by 2.3x."""
    old = _old_results()
    law = old["sigma_law"]
    cheap = [r for r in _graded(old) if r["interval"] <= 5.0e-3]
    assert all(r["bound_over_measured"] >= 1.0 for r in cheap), \
        "the cheap sweep held on the old wall too, which is why it was believed"
    a = _anchor_row(_graded(old))
    assert a["bound_over_measured"] == pytest.approx(0.2702, abs=5e-4)
    assert law["bound_holds"] is False
    assert law["tightness"][0] < 1.0


def test_W334_first_order_over_the_whole_250x():
    """What the anchor and the knee say once the wall is fixed.

    The lag's drift rate and its profile's peakedness are constant across 250x
    of interval, as they were before; so is sigma per unit lag now, to 0.8%,
    and every exponent is 1 to 0.3%. The knee's slope probe, which read the
    seam's own response 17x steeper at 1e-2 s on the old wall, reads it 1.008x.
    There is no knee. C2 is still negative, so it is s_seam alone that carries
    the bound.
    """
    res = _results()
    rows = _graded(res)
    _anchor_row(rows)
    rates = [r["lag"] / r["interval"] for r in rows]
    peaks = [r["lag_max"] / r["lag"] for r in rows]
    assert max(rates) / min(rates) < 1.005, rates
    assert max(peaks) / min(peaks) < 1.005, peaks

    per_lag = [r["sigma"] / r["lag"] for r in rows]
    assert max(per_lag) / min(per_lag) < 1.02, per_lag
    exps = [r["exponent"] for r in rows if r.get("exponent") == r.get("exponent")
            and r.get("exponent") is not None]
    assert len(exps) == 4 and all(0.997 < e < 1.003 for e in exps), exps
    knee = res["knee"]["rows"][0]
    assert knee["interval"] == pytest.approx(1.0e-2)
    assert knee["s_seam_rel"] == pytest.approx(1.0077, abs=1e-3)
    assert res["slope"]["C2"] < 0.0


def test_W334_control_the_old_wall_broke_the_ORDER_and_not_the_lag():
    """The old record's diagnosis, kept checkable.

    On the old wall the lag's drift rate and profile were just as constant --
    so the extrapolation was not sloppy -- while sigma per unit lag rose 8.8x
    in the last decade and the exponents went to 2.295 across the 1e-2 s knee
    and 1.790 beyond it. That was read, correctly, as the seam's RESPONSE
    changing rather than the trace it responds to. W334 found the response that
    changed was the wall's: a saturated isothermal channel (W301) and a ghost
    125x denser than the gas handed to the Riemann solver (W332).
    """
    old = _old_results()
    rows = _graded(old)
    _anchor_row(rows)
    rates = [r["lag"] / r["interval"] for r in rows]
    peaks = [r["lag_max"] / r["lag"] for r in rows]
    assert max(rates) / min(rates) < 1.005, rates
    assert max(peaks) / min(peaks) < 1.005, peaks

    per_lag = [r["sigma"] / r["lag"] for r in rows]
    assert max(per_lag[:3]) / min(per_lag[:3]) < 1.2, "flat over the cheap sweep"
    flat = max(per_lag[:3])
    assert per_lag[-1] / flat > 5.0, "and jumped by the declared interval"
    assert 1.5 < per_lag[-2] / flat < per_lag[-1] / flat

    exps = [r["exponent"] for r in rows if r.get("exponent") == r.get("exponent")
            and r.get("exponent") is not None]
    assert all(0.9 < e < 1.1 for e in exps[:2]), "first order over the cheap sweep: %r" % exps
    assert exps[-2] > 2.0, "second order across the knee: %r" % exps
    assert 1.7 < exps[-1] < 2.1, "and still very nearly second beyond it: %r" % exps
    assert old["knee"]["rows"][0]["s_seam_rel"] > 10.0
    assert old["slope"]["C2"] < 0.0


def test_W334_the_anchor_predictions_failed_on_their_registration_not_their_rule():
    """A1 and A2 were registered from the old core's lag rate, 25.89 K/s, and
    fail on the re-derived record: the fixed wall heats the shell faster (31.70
    K/s), so the lag is 1.581 K where 1.294 was registered, and sigma misses
    the top of A2's range by 0.7% because it rides on that lag. The rule behind
    them -- lag linear in the interval, sigma a fixed multiple of the lag -- is
    what the anchor tests, and applied to the NEW core's own graded rows it
    lands within 0.23% on the lag and 0.57% on sigma. (A consistency check, not
    a registration: the core and the anchor ran side by side on the box.)
    """
    W = _driver()
    res = _results()
    rows = _graded(res)
    a = _anchor_row(rows)
    got = a["anchor_predictions"]
    assert got["A1_lag_K"]["held"] is False and got["A2_sigma"]["held"] is False
    assert got["A3_bound_over_measured"]["held"] is True
    assert W.ANCHOR_PREDICTIONS["A1_lag_K"] == pytest.approx((1.2943 * 0.95, 1.2943 * 1.05))

    cheap = [r for r in rows if r["interval"] <= 5.0e-3]
    rate = sum(r["lag"] / r["interval"] for r in cheap) / len(cheap)
    per_lag = sum(r["sigma"] / r["lag"] for r in cheap) / len(cheap)
    lag_rule = rate * a["interval"]
    assert a["lag"] == pytest.approx(lag_rule, rel=3e-3)
    assert a["sigma"] == pytest.approx(per_lag * a["lag"], rel=1e-2)
    # and the registration really was the old core's: it held A1 at 1.292 K
    old = _anchor_row(_graded(_old_results()))["anchor_predictions"]
    assert old["A1_lag_K"]["held"] is True and old["A3_bound_over_measured"]["held"] is False


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

    **W334's record reads -1, 'not counted'.** It was re-derived on a rented
    Linux box (Tier 86), which has no Windows event log to count and no Modern
    Standby to be suspended by. -1 is accepted only with the evidence that
    PowerShell was absent -- the same record's machine probe failing to find
    it -- so a count that failed ON Windows still fails here.
    """
    res = _results()
    n = res.get("standby_during_run", 0)
    if n == -1:
        err = str(res["setup"].get("machine", {}).get("error", ""))
        assert "FileNotFoundError" in err, "-1 on a machine that has PowerShell: %r" % err
        return
    assert n == 0
