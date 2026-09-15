"""Tier 61 -- incompressible Navier-Stokes on body-fitted overset grids.

These pin what a flow solver on curved overlapping grids can get silently
wrong: a uniform stream that the overlap does not preserve, derivatives that are
not exact where they must be, a force with the wrong sign, a closed contour
that leaks, a pressure pinned into a blow-up, and a benchmark claimed rather
than measured.
"""

from __future__ import annotations

import json
import math
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (HERE, os.path.join(HERE, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                  # noqa: E402
import pytest                                                       # noqa: E402

from atlas.cases import overset as OV                               # noqa: E402
from atlas.cases import overset_ns as NS                            # noqa: E402

OUT = os.path.join(HERE, "out", "racelab10")
RECORD = os.path.join(OUT, "racelab10.json")
PAGE = os.path.join(HERE, "wiki", "concepts", "Atlas 0.1", "common",
                    "poc3-racelab-overset-flow.md")


def _small(box=None, **kw):
    bg = OV.CartesianGrid("bg", 64, 48, 1.0 / 16)
    body = OV.ogrid_annulus("body", 2.0, 1.5, 0.3, 0.75, 48, 7, beta=1.0, twist=0.4)
    ov = OV.Overset(bg, [body], hole_margin=0.15, width=3)
    return NS.OversetFlow(ov, 0.01, 0.01, box=box, **kw)


OUTFLOW = {"xlo": "dirichlet", "ylo": "dirichlet", "yhi": "dirichlet", "xhi": "outflow"}


# ---------------------------------------------------------------------------
# operators and rows
# ---------------------------------------------------------------------------


def test_first_derivatives_are_exact_on_a_linear_field_in_every_grid():
    f = _small(box=OUTFLOW)
    phi = 0.7 * f.X - 1.3 * f.Y + 2.0
    gx, gy = f.Dx @ phi, f.Dy @ phi
    assert np.abs(gx[f.disc] - 0.7).max() < 1e-11
    assert np.abs(gy[f.disc] + 1.3).max() < 1e-11
    # and nothing outside the discretisation rows
    assert np.abs(gx[~f.disc]).max() == 0.0


def test_every_unknown_has_exactly_one_kind_of_row():
    for box in (OUTFLOW, None):
        f = _small(box=box)
        kinds = (f.disc.astype(int) + f.wall + f.interp + f.edge_d + f.edge_out + f.outer_dirichlet)
        assert np.all(kinds == 1)
        assert f.has_outflow == (box is OUTFLOW)
        assert f.pressure_augmented == (box is None)


def test_the_momentum_rows_share_tier_60_s_laplacian():
    f = _small(box=OUTFLOW)
    c = f.ov.comps[0]
    R, C, V, _jj, _ii = f.ov.laplacian_triplets(c)
    rows = np.concatenate(R)
    L = f.Lap.tocsr()
    import scipy.sparse as sp
    ref = sp.coo_matrix((np.concatenate(V), (rows, np.concatenate(C))), shape=L.shape).tocsr()
    diff = (L[np.unique(rows)] - ref[np.unique(rows)])
    assert abs(diff).max() < 1e-12


# ---------------------------------------------------------------------------
# what a flow solver must preserve
# ---------------------------------------------------------------------------


def test_a_uniform_stream_passes_the_overlap_unchanged():
    """With the wall moving with the stream, uniform flow is an exact solution:
    the interpolation reproduces constants, every derivative of a constant is zero
    and the pressure increment is zero, so twenty steps must change nothing."""
    f = _small(box=OUTFLOW, wall_velocity=lambda g, t: (np.ones(g.ni), np.zeros(g.ni)),
               box_velocity=lambda x, y, t: (np.ones_like(x), np.zeros_like(x)))
    n = f.ov.n_unknowns
    f.set_state(np.ones(n), np.zeros(n))
    for _ in range(20):
        f.step()
    assert np.abs(f.U - 1.0).max() < 1e-12
    assert np.abs(f.V).max() < 1e-12
    assert np.abs(f.P).max() < 1e-12
    fr = f.forces("body")
    assert abs(fr["fx"]) < 1e-12 and abs(fr["fy"]) < 1e-12


def test_a_closed_contour_carries_no_net_flux_in_a_uniform_stream():
    f = _small(box=OUTFLOW)
    n = f.ov.n_unknowns
    f.set_state(np.full(n, 0.8), np.full(n, -0.3))
    assert abs(f.ring_flux("body", 3)) < 1e-12
    assert abs(f.box_flux(20, 44, 12, 36)) < 1e-12


def test_the_force_on_a_body_in_a_linear_pressure_points_down_the_gradient():
    """``p = x`` at rest pushes the body toward -x with force ``-area``: the sign of
    every drag this solver reports rests on this."""
    f = _small(box=OUTFLOW)
    n = f.ov.n_unknowns
    f.set_state(np.zeros(n), np.zeros(n), p=f.X.copy())
    fr = f.forces("body")
    area = math.pi * 0.3 ** 2
    assert fr["fx"] == pytest.approx(-area, rel=5e-3)
    assert abs(fr["fy"]) < 1e-3 * area


# ---------------------------------------------------------------------------
# the failure this tier found, and convergence
# ---------------------------------------------------------------------------


def test_a_dirichlet_only_box_no_longer_blows_up_from_its_pinned_corner():
    """The first version pinned the increment at one corner cell; with the overlap
    present the march blew up within ten steps.  Thirty steps must now stay close
    to the manufactured solution."""
    r = NS.mms_run(1, dt=0.01, T=0.3)
    assert r["steps"] == 30
    for g, e in r["grids"].items():
        assert e["u_max_err"] < 5e-3, g
    assert max(r["divergence"].values()) < 1e-2


def test_the_manufactured_solution_converges_between_two_levels():
    a = NS.mms_run(2, dt=0.004, T=0.08)
    b = NS.mms_run(3, dt=0.004, T=0.08)
    for g in ("bg", "body"):
        assert a["grids"][g]["u_max_err"] / b["grids"][g]["u_max_err"] > 2.8, g


def test_shedding_statistics_read_a_known_signal():
    t = np.arange(0, 60, 0.01)
    f = 0.2
    cl = 0.3 * np.sin(2 * math.pi * f * t) + 0.01
    cd = 1.35 + 0.01 * np.sin(4 * math.pi * f * t)
    s = NS.shedding_statistics(t, cd, cl, t_from=10.0, diameter=0.4)
    assert s["strouhal"] == pytest.approx(f * 0.4, rel=1e-4)
    assert s["cl_amplitude"] == pytest.approx(0.3, rel=1e-3)
    assert s["cd_mean"] == pytest.approx(1.35, abs=1e-3)
    assert s["cycles"] >= 9


def test_w271_the_compact_projection_is_first_order_in_time_and_the_exact_one_is_not():
    """W271, small enough for the suite: on the background alone the compact
    projection's successive time-step differences shrink at first order and the
    exact projection's at second."""
    import tier61_overset_flow as T
    compact = T._time_orders(lambda d: (T._mms_flow(2, d, with_body=False), lambda f: f.step()),
                             dts=(0.02, 0.01, 0.005), T=0.2)
    exact = T._time_orders(lambda d: (T._mms_flow(2, d, with_body=False, projection="exact"),
                                      lambda f: f.step()), dts=(0.02, 0.01, 0.005), T=0.2)
    assert compact["order_rms"][0] < 1.3
    assert exact["order_rms"][0] > 1.6
    assert "W271" in NS.__doc__ and NS.PROJECTIONS == ("compact", "exact")


# ---------------------------------------------------------------------------
# the record, and the page that quotes it
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def record():
    if not os.path.isfile(RECORD):
        pytest.skip("out/racelab10/racelab10.json is not here (it is carried upstream only)")
    with open(RECORD, encoding="utf-8") as fh:
        return json.load(fh)


def test_the_record_s_predictions_are_this_script_s_and_were_registered_in_order(record):
    import tier61_overset_flow as T
    assert record["prediction"] == [dict(p) for p in T.PREDICTION]
    assert record["prediction_small_dt"] == [dict(p) for p in T.PREDICTION_SMALL_DT]
    assert record["read_before_this_run"] == list(T.READ_BEFORE_THIS_RUN)
    assert record["read_before_diagnosis"] == list(T.READ_BEFORE_DIAGNOSIS)
    assert record["prediction_recorded_at"] < record["prediction_small_dt_recorded_at"]


def test_every_registered_prediction_has_a_verdict_and_the_judge_agrees(record):
    import tier61_overset_flow as T
    assert record["verdicts_missing"] == []
    ids = {p["id"] for p in T.PREDICTION} | {p["id"] for p in T.PREDICTION_SMALL_DT}
    assert set(record["verdicts"]) == ids
    assert T.judge(record) == record["verdicts"]


def test_w274_a_price_not_taken_on_mains_is_not_a_measurement_of_p1():
    """P1's prose says "on mains"; the judge once ignored the power state."""
    import tier61_overset_flow as T
    fast = {"median": {"step_s": 0.5}, "iterations_max": 10}
    on = {"on_mains": True}
    off = {"on_mains": False}
    assert T.judge({})["P1"] is None
    assert T.judge({"price": dict(fast, machine_before=on, machine_after=on)})["P1"] is True
    assert T.judge({"price": dict(fast, machine_before=off, machine_after=on)})["P1"] is None
    assert T.judge({"price": dict(fast, machine_before=on, machine_after=off)})["P1"] is None
    slow = {"median": {"step_s": 1.5}, "iterations_max": 10}
    assert T.judge({"price": dict(slow, machine_before=on, machine_after=on)})["P1"] is False


def test_the_time_study_failed_where_the_diagnosis_says_and_momentum_alone_is_second_order(record):
    v = record["verdicts"]
    assert v["T1"] is False and v["T2"] is False
    D = record["diagnose"]
    assert D["background_momentum_only_exact_pressure"]["order_max"][0] > 1.9
    assert D["background_as_built"]["order_max"][0] < 1.1
    assert D["background_exact_projection"]["order_rms"][0] > 1.9
    assert D["composite_exact_projection"]["failed_at_step"] is not None
    assert D["composite_compact_five_passes"]["order_max"][0] < 1.3


def test_the_cylinder_arms_are_the_declared_ones_and_their_statistics_are_in_the_record(record):
    import tier61_overset_flow as T
    cyl = record["cylinder"]
    assert set(cyl) == set(T.CYL_ARMS)
    for arm, spec in T.CYL_ARMS.items():
        assert cyl[arm]["spec"] == spec
        assert cyl[arm]["steps_done"] == int(round(T.CYL_T_END / spec["dt"]))
        assert cyl[arm]["statistics"]["cycles"] >= 15


def test_the_page_says_what_the_record_measured(record):
    if not os.path.isfile(PAGE):
        pytest.skip("the page is not here")
    text = open(PAGE, encoding="utf-8").read()
    st = record["cylinder"]["base"]["statistics"]
    assert "%.3f" % st["strouhal"] in text
    assert "%.3f" % st["cd_mean"] in text
    assert "%.3f" % st["cl_amplitude"] in text
    for pid, val in record["verdicts"].items():
        assert ("| %s |" % pid) in text
    assert "W271" in text
