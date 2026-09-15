"""Tier 60 -- body-fitted overset grids: the grids, the overlap, the pressure solve.

These pin what the foundation of the body-fitted column can get silently wrong:
a Jacobian of the wrong sign, interpolation that is not exact where it must be,
an overlap that leaks instead of refusing, a generator that folds or shrinks or
changes with its own resolution, and a convergence order that is claimed rather
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

RECORD = os.path.join(HERE, "out", "racelab9", "racelab9.json")
PAGE = os.path.join(HERE, "wiki", "concepts", "Atlas 0.1", "common",
                    "poc3-racelab-body-fitted-grids.md")


def _order(a, b):
    return math.log2(a / b)


@pytest.fixture(scope="module")
def pair():
    bg = OV.CartesianGrid("bg", 128, 96, 1.0 / 32)
    body = OV.ogrid_annulus("body", 2.0, 1.5, 0.3, 0.75, 96, 13, beta=1.0, twist=0.6)
    return bg, body


# ---------------------------------------------------------------------------
# grids and operators
# ---------------------------------------------------------------------------


def test_a_clockwise_ogrid_is_right_handed_and_a_counterclockwise_one_is_refused():
    """``i`` clockwise round the body and ``j`` outward is the orientation every
    formula in the module assumes; the other one has J < 0 and must not build."""
    g = OV.ogrid_annulus("g", 0.0, 0.0, 1.0, 2.0, 32, 5)
    assert np.all(g.J > 0)
    with pytest.raises(OV.GridQualityError):
        OV.CurvilinearGrid("mirror", g.x[:, ::-1], g.y[:, ::-1])


def test_every_laplacian_row_annihilates_constants_and_interpolation_rows_sum_to_zero(pair):
    bg, body = pair
    ov = OV.Overset(bg, [body], hole_margin=0.15, width=3)
    A, _b = ov.poisson(lambda x, y: np.zeros_like(x), lambda x, y: np.zeros_like(x),
                       lambda g: np.zeros(g.ni))
    rs = np.asarray(A.sum(axis=1)).ravel()
    diag = np.abs(A.diagonal())
    Sb, Sc = ov.status["bg"], ov.status["body"]
    inner = np.zeros_like(Sb, dtype=bool)
    inner[1:-1, 1:-1] = True
    rows = np.concatenate([ov.index["body"][Sc == OV.DISC], ov.index["body"][Sc == OV.WALL],
                           ov.index["bg"][(Sb == OV.DISC) & inner]])
    assert (np.abs(rs[rows]) / diag[rows]).max() < 1e-12
    interp = np.concatenate([ov.index["bg"][Sb == OV.INTERP], ov.index["body"][Sc == OV.INTERP]])
    assert interp.size > 0
    assert np.abs(rs[interp]).max() < 1e-12


def test_a_linear_field_s_gradient_is_exact_on_a_twisted_grid(pair):
    """Central differences of a linear function of the node coordinates are exact,
    and the metric identities turn them into exact physical derivatives -- on a
    grid whose lines are not orthogonal, where ``a12`` is not zero."""
    _bg, body = pair
    assert np.abs(body.a12).max() > 1e-3, "the control grid must actually be twisted"
    gx, gy = body.gradient(0.7 * body.x - 1.3 * body.y + 2.0)
    assert np.abs(gx - 0.7).max() < 1e-12
    assert np.abs(gy + 1.3).max() < 1e-12


def test_lagrange_weights_reproduce_polynomials_of_their_own_degree():
    t = np.linspace(-0.3, 2.3, 17)
    for w in (2, 3, 4):
        L = OV.lagrange_weights(t, w)
        nodes = np.arange(w)
        for deg in range(w):
            assert np.abs(L @ nodes ** deg - t ** deg).max() < 1e-12
        dL = OV._lagrange_dweights(t, w)
        assert np.abs(dL.sum(axis=1)).max() < 1e-12


# ---------------------------------------------------------------------------
# the overlap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("width", [2, 3, 4])
def test_interpolation_reproduces_linear_fields_exactly_at_every_width(pair, width):
    """The donor is located on the SAME width-w map its weights interpolate with,
    so the weights reproduce x and y -- hence any linear field -- to round-off,
    on a curved, twisted grid.  A bilinear location with wider weights would not."""
    bg, body = pair
    ov = OV.Overset(bg, [body], hole_margin=0.15, width=width)
    f = lambda xx, yy: 0.7 * xx - 1.3 * yy + 0.25                 # noqa: E731
    n_checked = 0
    for name, spec in ov.donors.items():
        for e in spec["entries"]:
            donor = bg if e["donor"] == "bg" else body
            DX = donor.X if donor is bg else donor.x
            DY = donor.Y if donor is bg else donor.y
            tx = (bg.X if name == "bg" else body.x)[e["j"], e["i"]]
            ty = (bg.Y if name == "bg" else body.y)[e["j"], e["i"]]
            val = np.sum(e["weights"] * f(DX.ravel()[e["flat"]], DY.ravel()[e["flat"]]), axis=1)
            assert np.abs(val - f(tx, ty)).max() < 1e-12
            n_checked += val.size
    assert n_checked == int(np.sum(ov.status["bg"] == OV.INTERP)) + body.ni


def test_every_donor_carries_its_own_equation(pair):
    """Explicit interpolation: an interpolation point never reads another one."""
    bg, body = pair
    ov = OV.Overset(bg, [body], hole_margin=0.15, width=3)
    for name, spec in ov.donors.items():
        for e in spec["entries"]:
            S = ov.status[e["donor"]].ravel()[e["flat"]]
            assert np.all(np.isin(S, (OV.DISC, OV.WALL))), f"{name} reads a non-equation point"


def test_a_body_grid_too_thin_for_its_overlap_is_refused_with_its_orphans_named(pair):
    bg, _body = pair
    thin = OV.ogrid_annulus("thin", 2.0, 1.5, 0.3, 0.42, 96, 5)
    with pytest.raises(OV.OversetError, match="orphan"):
        OV.Overset(bg, [thin], hole_margin=0.15)


def test_a_grid_leaving_the_box_and_two_intersecting_bodies_are_refused(pair):
    bg, _body = pair
    with pytest.raises(OV.OversetError, match="leaves the background box"):
        OV.Overset(bg, [OV.ogrid_annulus("edge", 0.5, 1.5, 0.3, 0.75, 96, 13)], hole_margin=0.15)
    with pytest.raises(OV.OversetError, match="inside"):
        OV.Overset(bg, [OV.ogrid_annulus("a", 1.6, 1.5, 0.3, 0.75, 96, 13),
                        OV.ogrid_annulus("b", 2.4, 1.5, 0.3, 0.75, 96, 13)], hole_margin=0.15)


def test_the_background_s_holes_are_exactly_the_body_and_its_margin(pair):
    bg, body = pair
    ov = OV.Overset(bg, [body], hole_margin=0.15, width=3)
    S = ov.status["bg"]
    r = np.hypot(bg.X - 2.0, bg.Y - 1.5)
    # the body outline is a 96-gon, so the cut radius is within its sag of 0.45
    sag = 0.3 * (1 - math.cos(math.pi / 96))
    assert np.all(S[r < 0.45 - sag - 1e-9] == OV.HOLE)
    assert not np.any(S[r > 0.45 + 1e-9] == OV.HOLE)
    assert np.all(S[r > 0.45 + 2 * bg.h] != OV.INTERP)


# ---------------------------------------------------------------------------
# convergence, on levels small enough for the suite
# ---------------------------------------------------------------------------


def _levels(**kw):
    return [OV.manufactured_poisson(level=L, **kw) for L in (1, 2, 3)]


def test_the_quadratic_ghost_makes_the_box_gradient_second_order():
    """The linear ghost ``2g - p0`` read the gradient at order 1.00 (Tier 60's
    smoke run); the quadratic ghost is what the flow solver's outlet needs."""
    r = _levels(case="box")
    e = [x["grids"]["bg"]["grad_max_err"] for x in r]
    assert _order(e[1], e[2]) > 1.9
    p = [x["grids"]["bg"]["max_err"] for x in r]
    assert _order(p[1], p[2]) > 1.9


def test_the_composite_converges_at_second_order_and_the_overlap_costs_a_constant():
    comp = _levels(case="composite")
    ann = _levels(case="annulus")
    for g in ("bg", "body"):
        e = [x["grids"][g]["max_err"] for x in comp]
        assert _order(e[1], e[2]) > 1.85, g
    assert comp[-1]["grids"]["body"]["max_err"] <= 3.0 * ann[-1]["grids"]["body"]["max_err"]
    for x in comp:
        assert x["solve"]["relative_residual"] < 1e-10


def test_bilinear_interpolation_costs_the_background_gradient_an_order_width_three_does_not():
    w3 = _levels(case="composite", width=3)
    w2 = _levels(case="composite", width=2)
    o3 = _order(w3[1]["grids"]["bg"]["grad_max_err"], w3[2]["grids"]["bg"]["grad_max_err"])
    o2 = _order(w2[1]["grids"]["bg"]["grad_max_err"], w2[2]["grids"]["bg"]["grad_max_err"])
    assert o2 < 1.7 < o3


# ---------------------------------------------------------------------------
# the generator
# ---------------------------------------------------------------------------


def test_the_generator_reproduces_the_analytic_annulus():
    g = OV.ogrid_from_outline("gen", OV.circle_outline(2.0, 1.5, 0.3), 128, 13, 0.45, beta=1.0)
    ref = OV.ogrid_annulus("ref", 2.0, 1.5, 0.3, 0.75, 128, 13, beta=1.0)
    sag = 0.3 * (1 - math.cos(math.pi / 4000))
    assert max(np.abs(g.x - ref.x).max(), np.abs(g.y - ref.y).max()) < 2 * sag + 1e-12


def test_the_generated_mapping_does_not_depend_on_the_resolution():
    """What the layer-marching first version failed: a finer grid must be the SAME
    mapping sampled more densely, or a convergence study refines nothing."""
    outline = OV.ellipse_outline(2.0, 1.5, 0.6, 0.15, 10.0)
    coarse = OV.ogrid_from_outline("c", outline, 128, 9, 0.3, beta=2.0)
    fine = OV.ogrid_from_outline("f", outline, 256, 17, 0.3, beta=2.0)
    assert np.abs(fine.x[::2, ::2] - coarse.x).max() < 1e-12
    assert np.abs(fine.y[::2, ::2] - coarse.y).max() < 1e-12


def test_a_concave_dent_tighter_than_the_grid_is_refused_not_repaired():
    with pytest.raises(OV.GridQualityError, match="fold"):
        OV.ogrid_from_outline("dent", OV.dented_circle_outline(2.0, 1.5, 0.3), 256, 13, 0.3,
                              beta=2.0, kappa=0.0)


def test_a_sharp_trailing_edge_and_a_thin_plate_generate_without_folding():
    for outline in (OV.naca4_outline(1.75, 1.5, 0.5, 0.12, 14.0),
                    OV.rounded_plate_outline(1.75, 1.45, 0.5, 0.05, 14.0)):
        g = OV.ogrid_from_outline("wing", outline, 256, 13, 0.3, beta=2.0)
        assert OV.grid_quality(g)["folded_cells"] == 0
        assert np.all(g.J > 0)


def test_the_plate_outline_puts_positive_alpha_s_trailing_edge_higher():
    p = OV.rounded_plate_outline(1.0, 1.0, 0.5, 0.02, 14.0)
    te = p[np.argmax(p[:, 0])]
    assert te[1] > 1.0 + 0.5 * math.sin(math.radians(14.0)) - 0.02


def _reference_points_in_polygon(px, py, poly):
    """The one-edge-at-a-time loop Tier 60's record was built with, kept as the
    reference the vectorised version must equal bitwise."""
    inside = np.zeros(px.shape, dtype=bool)
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    for a, b, c, d in zip(x1, y1, x2, y2):
        crosses = (b > py) != (d > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = a + (py - b) * (c - a) / (d - b)
        inside ^= crosses & (px < xc)
    return inside


def test_the_vectorised_point_in_polygon_is_bitwise_the_loop_it_replaced():
    """The speed-up (44.6 s to 12.6 s at the car's size) is allowed only because it
    changes no answer: every crossing is the same expression in the same order.
    Points ON vertices and edges are included, where a changed order would show."""
    rng = np.random.default_rng(3)
    polys = [OV.dented_circle_outline(0.0, 0.0, 1.0, n=97),
             OV.naca4_outline(-0.5, 0.0, 1.0, 0.12, 14.0, n=120),
             rng.standard_normal((31, 2))]
    for poly in polys:
        px = np.concatenate([rng.uniform(-1.5, 1.5, 3000), poly[:, 0],
                             0.5 * (poly[:, 0] + np.roll(poly[:, 0], -1))])
        py = np.concatenate([rng.uniform(-1.5, 1.5, 3000), poly[:, 1],
                             0.5 * (poly[:, 1] + np.roll(poly[:, 1], -1))])
        for cap in (4_000_000, 1000):                       # one chunk, and many
            got = OV.points_in_polygon(px, py, poly, max_elements=cap)
            assert np.array_equal(got, _reference_points_in_polygon(px, py, poly))


def test_gentle_normal_smoothing_is_what_keeps_a_tight_tip_converging():
    """G4's diagnosis, pinned at levels the suite can afford: on the thin ellipse's
    grid alone the L2->L3 order falls as the smoothing widens (Tier 60 measured
    1.97, 1.88 and 1.56 at kappa 0.25, 1 and 4)."""
    o = {}
    for kappa in (0.25, 1.0, 4.0):
        r = [OV.manufactured_poisson("annulus", L, body="ellipse", kappa=kappa)
             for L in (2, 3)]
        o[kappa] = _order(r[0]["grids"]["body"]["max_err"], r[1]["grids"]["body"]["max_err"])
    assert o[0.25] > o[1.0] > o[4.0]
    assert o[0.25] > 1.9


# ---------------------------------------------------------------------------
# the record, and the page that quotes it
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def record():
    if not os.path.isfile(RECORD):
        pytest.skip("out/racelab9/racelab9.json is not here (it is carried upstream only)")
    with open(RECORD, encoding="utf-8") as fh:
        return json.load(fh)


def test_the_record_s_predictions_are_this_script_s_and_were_registered_in_order(record):
    import tier60_body_fitted_grids as T
    assert record["prediction"] == [dict(p) for p in T.PREDICTION]
    assert record["prediction_adopted"] == [dict(p) for p in T.PREDICTION_ADOPTED]
    assert record["read_before_this_run"] == list(T.READ_BEFORE_THIS_RUN)
    assert record["read_before_diagnosis"] == list(T.READ_BEFORE_DIAGNOSIS)
    # the adopted default's prediction was registered after the first verdicts
    assert record["prediction_recorded_at"] < record["prediction_adopted_recorded_at"]


def test_the_verdicts_are_fifteen_held_and_g4_falsified(record):
    v = dict(record["verdicts"])          # a copy: the fixture is shared
    assert record["verdicts_missing"] == []
    assert v.pop("G4") is False
    assert set(v) == {"P1", "P2", "P3", "P4", "P5", "P6", "X1", "X2", "G1", "G2", "G3",
                      "T1", "T2", "D1", "D2"}
    assert all(val is True for val in v.values())


def test_the_judge_reads_the_record_the_same_way_again(record):
    import tier60_body_fitted_grids as T
    assert T.judge(record) == record["verdicts"]


def test_g4_s_diagnosis_changes_one_thing_per_arm_and_the_smoothing_is_monotone(record):
    import tier60_body_fitted_grids as T
    D = record["diagnose"]
    for label, _levels, kw in T.DIAGNOSE_ARMS:
        assert D[label]["kwargs"] == json.loads(json.dumps(kw))   # tuples are lists in JSON
    last = {k: D[k]["orders"]["body.max_err"][2] for k in ("kappa_0.25_alone", "kappa_4_alone")}
    g4 = record["generator"]["ellipse_composite"]["orders"]["body.max_err"][2]
    assert last["kappa_0.25_alone"] > 1.95 > g4 > last["kappa_4_alone"]
    # the outline's density is not the cause: identical to four figures at L4
    a = record["generator"]["ellipse_composite"]["levels"][-1]["grids"]["body"]["max_err"]
    b = D["polygon_64000_composite"]["levels"][-1]["grids"]["body"]["max_err"]
    assert abs(a - b) / a < 1e-3


def test_the_price_carries_its_power_state_and_the_rule_that_decided_it(record):
    import tier60_body_fitted_grids as T
    P = record["price"]
    assert P["rule"] == T.PRICE_RULE == record["price_rule"]
    assert P["machine_before"]["on_mains"] is not None
    assert P["machine_after"]["on_mains"] is not None
    s = P["splu"]["median_solve_s"]
    expected = ("every exchange" if s <= 0.05 else "once per macro-step" if s <= 0.2
                else "iterative, warm-started, or a smaller system")
    assert P["decision"] == expected
    assert record["build_cost"]["counts_match_the_price_stage"] is True


def test_the_page_says_what_the_record_measured():
    if not os.path.isfile(PAGE):
        pytest.skip("the page is not here")
    text = open(PAGE, encoding="utf-8").read()
    with open(RECORD, encoding="utf-8") as fh:
        rec = json.load(fh)
    P = rec["price"]
    assert "204{,}599" in text and P["splu"]["n"] == 204599
    assert "%.2f" % P["splu"]["factor_s"] in text
    assert "%.3f" % P["splu"]["median_solve_s"] in text
    assert "once per macro-step" in text and P["decision"] == "once per macro-step"
    g4 = rec["generator"]["ellipse_composite"]["orders"]["body.max_err"][2]
    assert "%.2f" % g4 in text
    assert "**falsified**" in text
    assert "$\\kappa = 0.25$" in text
    assert "on battery" in text and P["machine_before"]["on_mains"] is False


def test_the_generator_defaults_to_the_gentle_smoothing_and_the_record_pins_the_old():
    import inspect
    import tier60_body_fitted_grids as T
    assert OV.KAPPA == 0.25
    assert inspect.signature(OV.ogrid_from_outline).parameters["kappa"].default == OV.KAPPA
    # the registered stages ran at kappa 1, and a re-run must reproduce them
    assert inspect.signature(OV.manufactured_poisson).parameters["kappa"].default == 1.0
    assert T.GEN_KAPPA == 1.0
    assert inspect.signature(T.car_size_standin).parameters["kappa"].default == 1.0
