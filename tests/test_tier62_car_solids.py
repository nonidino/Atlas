"""Tier 62 -- the drawn car as solids on body-fitted grids.

These pin what several bodies and a road on overlapping grids can get silently
wrong: a subclass that joins Tier 60's grids differently, a donor that
extrapolates or reads an interpolation point, a gap no interpolation can close
accepted instead of refused, a road patch whose wall faces the wrong way, a
generated grid that changes when it is refined, a wheel that does not roll at
road speed, a start that is not an exact state, and a record the page misquotes.
"""

from __future__ import annotations

import copy
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

from atlas.cases import car_solids as CS                            # noqa: E402
from atlas.cases import overset as OV                               # noqa: E402
from atlas.cases import overset_multi as OM                         # noqa: E402
from atlas.cases import overset_ns as NS                            # noqa: E402

OUT = os.path.join(HERE, "out", "racelab11")
RECORD = os.path.join(OUT, "racelab11.json")
PAGE = os.path.join(HERE, "wiki", "concepts", "Atlas 0.1", "common", "poc3-racelab-car-solids.md")


def _table(ov):
    out = {}
    for name, spec in ov.donors.items():
        for e in spec["entries"]:
            for k in range(len(e["j"])):
                out[(name, int(e["j"][k]), int(e["i"][k]))] = (
                    e["donor"], tuple(int(v) for v in e["flat"][k]), np.asarray(e["weights"][k]))
    return out


@pytest.fixture(scope="module")
def multi_l3():
    return OM.multi_geometry(3)


# ---------------------------------------------------------------------------
# joining the grids
# ---------------------------------------------------------------------------


def test_the_subclass_joins_tier_60_s_grids_bit_for_bit():
    bg = OV.CartesianGrid("bg", 64, 48, 1.0 / 16)
    body = OV.ogrid_annulus("body", 2.0, 1.5, 0.3, 0.75, 48, 7, beta=1.0)
    a = OV.Overset(bg, [body], hole_margin=0.15, width=3)
    b = OM.MultiOverset(bg, [body], hole_margin=0.15, width=3)
    for g in ("bg", "body"):
        assert np.array_equal(a.status[g], b.status[g])
    ta, tb = _table(a), _table(b)
    assert set(ta) == set(tb)
    for k in ta:
        assert ta[k][:2] == tb[k][:2]
        assert np.array_equal(ta[k][2], tb[k][2])


def test_every_interpolation_equation_reproduces_a_linear_field(multi_l3):
    ov = multi_l3
    f = lambda x, y: 0.7 * x - 1.3 * y + 2.0                          # noqa: E731
    for name, spec in ov.donors.items():
        g = next(q for q in ov.grids if q.name == name)
        gx, gy = OM.MultiOverset._xy(g)
        for e in spec["entries"]:
            d = next(q for q in ov.grids if q.name == e["donor"])
            dx, dy = OM.MultiOverset._xy(d)
            got = np.sum(f(dx.ravel()[e["flat"]], dy.ravel()[e["flat"]]) * e["weights"], axis=1)
            assert np.abs(got - f(gx[e["j"], e["i"]], gy[e["j"], e["i"]])).max() < 1e-10


def test_every_donor_node_carries_its_own_equation_and_no_hole_is_an_unknown(multi_l3):
    ov = multi_l3
    for name, spec in ov.donors.items():
        for e in spec["entries"]:
            S = ov.status[e["donor"]].ravel()[e["flat"]]
            allowed = (OV.DISC,) if e["donor"] == "bg" else (OV.DISC, OV.WALL)
            assert np.all(np.isin(S, allowed)), (name, e["donor"])
    for g in ov.grids:
        assert np.all((ov.index[g.name] < 0) == (ov.status[g.name] == OV.HOLE))


def test_the_road_cuts_the_tyre_s_grid_and_the_patch_serves_it(multi_l3):
    ov = multi_l3
    rep = ov.report()["grids"]
    assert rep["wheel"]["holes_cut_by"].get("road", 0) > 0
    assert rep["wheel"]["donor_grids"].get("road", 0) > 0
    assert rep["road"]["donor_grids"].get("wheel", 0) > 0
    assert rep["panel"]["holes_cut_by"].get("wheel", 0) > 0


def _box():
    h = 1.0 / 64
    return OV.CartesianGrid("bg", 192, 97, h, y0=-0.5 * h)


def _wheel(name, cx, cy):
    return OV.ogrid_annulus(name, cx, cy, 0.3, 0.45, 384, 33, beta=3.0)


def _patch(cx):
    return OM.road_patch("road", cx - 0.24, cx + 0.24, 0.075, 177, 33, beta=3.0)


@pytest.mark.parametrize("comps, fragment", [
    (lambda: [_wheel("a", 1.0, 0.5), _wheel("b", 1.5, 0.5)], "overlap"),
    (lambda: [_wheel("w", 1.5, 0.3), _patch(1.5)], ""),
    (lambda: [_wheel("w", 1.5, 0.3012), _patch(1.5)], "two grid rows"),
    (lambda: [_wheel("w", 0.2, 0.6)], "inlet"),
])
def test_what_no_interpolation_can_join_is_refused_by_name(comps, fragment):
    with pytest.raises(OV.OversetError) as exc:
        OM.MultiOverset(_box(), comps(), hole_margin=0.04, road=0.0)
    assert fragment in str(exc.value)


def test_a_road_patch_is_right_handed_with_its_wall_normal_into_the_road():
    p = OM.road_patch("road", 1.0, 2.0, 0.1, 65, 17, beta=3.0)
    assert np.all(p.J > 0) and not p.periodic
    nx_, ny_ = p.wall_normal()
    assert np.allclose(nx_, 0.0) and np.allclose(ny_, -1.0)
    assert np.all(p.y[0] == 0.0)


# ---------------------------------------------------------------------------
# the generator's two new options
# ---------------------------------------------------------------------------


def test_the_rounded_trailing_edge_has_the_declared_thickness():
    xy = OM.aerofoil_outline(0.0, 0.0, 1.0, 0.12, 0.0, te_thickness=0.02, n=4000)
    near_te = xy[np.abs(xy[:, 0] - 1.0) < 1e-3]
    assert near_te[:, 1].max() == pytest.approx(0.01, abs=2e-4)
    assert near_te[:, 1].min() == pytest.approx(-0.01, abs=2e-4)
    assert xy[:, 0].max() == pytest.approx(1.01, abs=1e-6)


def test_clustering_and_room_are_properties_of_the_outline_not_the_resolution():
    outline = OM.aerofoil_outline(1.5, 0.3, 0.5, 0.12, 12.0, 0.01)
    kw = dict(beta=2.0, cluster=4.0, sigma_wall=0.02, kappa=0.5, room=0.4, room_floor=0.03)
    g1 = OV.ogrid_from_outline("a", outline, 128, 13, 0.1, **kw)
    g2 = OV.ogrid_from_outline("a", outline, 256, 25, 0.1, **kw)
    assert np.abs(g2.x[::2, ::2] - g1.x).max() < 1e-12
    assert np.abs(g2.y[::2, ::2] - g1.y).max() < 1e-12


def test_clustering_puts_columns_where_the_outline_turns():
    outline = OM.aerofoil_outline(0.0, 0.0, 1.0, 0.12, 0.0, 0.01)
    even = OV.ogrid_from_outline("a", outline, 256, 9, 0.1)
    dense = OV.ogrid_from_outline("a", outline, 256, 9, 0.1, cluster=4.0)
    def near_te(g):
        return int(np.sum(g.x[0] > 0.95))
    assert near_te(dense) > 1.5 * near_te(even)


def test_room_is_the_width_of_a_notch_and_infinite_off_a_convex_side():
    """A U with a slot 0.2 wide: from its inner walls the normal meets the far wall
    0.2 away; from the outside of the U it escapes."""
    u = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.6], [0.6, 0.6], [0.6, 0.2], [0.4, 0.2],
                  [0.4, 0.6], [0.0, 0.6]])
    body = OV.orient_clockwise(u)
    M = OV.resample_closed(body, 4096)
    L = float(np.sum(np.hypot(np.roll(M[:, 0], -1) - M[:, 0], np.roll(M[:, 1], -1) - M[:, 1])))
    tx, ty = OV._d_periodic(M[:, 0], 0), OV._d_periodic(M[:, 1], 0)
    m = np.hypot(tx, ty)
    reach = OV._room_along_normals(M, -ty / m, tx / m, body, L)
    inner = (np.abs(M[:, 0] - 0.4) < 1e-9) & (M[:, 1] > 0.3) & (M[:, 1] < 0.5)
    outer = (np.abs(M[:, 0] - 0.0) < 1e-9) & (M[:, 1] > 0.1) & (M[:, 1] < 0.5)
    assert inner.any() and outer.any()
    assert np.allclose(reach[inner], 0.2, atol=1e-6)
    assert np.all(np.isinf(reach[outer]))
    circle = OV.circle_outline(0.0, 0.0, 0.3, n=2000)
    g = OV.ogrid_from_outline("c", circle, 256, 9, 0.1, room=0.4, room_floor=0.03)
    assert g.meta["thickness_min"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# the car's rule
# ---------------------------------------------------------------------------


def test_the_rule_reads_the_file_s_block_and_refuses_what_it_does_not_know():
    from atlas.cases import racelab as RL
    doc = copy.deepcopy(RL.load_geometry())
    doc["solids"] = {"panel_thickness_cells": 4.0, "note": "a test"}
    rule = CS.solids_rule(doc)
    assert rule["panel_thickness_cells"] == 4.0 and rule["from_file"] == ["panel_thickness_cells"]
    doc["solids"] = {"panel_thickness": 4.0}
    with pytest.raises(ValueError):
        CS.solids_rule(doc)


@pytest.fixture(scope="module")
def solids_fixed():
    """The car's solids with the fillets the ladder chose in the record, unchecked:
    the shapes without the ladder's minute of trial grids."""
    from atlas.cases import racelab as RL
    doc = copy.deepcopy(RL.load_geometry())
    fillets = {"FW_MAIN": 6.0, "NOSE": 6.0}
    if os.path.isfile(RECORD):
        with open(RECORD, encoding="utf-8") as fh:
            rec = json.load(fh)
        if "car" in rec:
            fillets = {f["members"][0]: f["fillet_cells"] for f in rec["car"]["fillets"]}
    doc["solids"] = {"fillet_by_member": fillets}
    return CS.car_solids(geometry=doc, check=False)


def test_every_plate_is_in_exactly_one_solid_and_no_solid_reaches_a_wheel(solids_fixed):
    from shapely.geometry import Point, Polygon
    from atlas.cases import racelab as RL
    solids, rec = solids_fixed
    objs, _flat = RL.car_bodies()
    plates = [o.body.body_id for o in objs if isinstance(o, RL.PlateBody)]
    seen = [m for s in solids if s.kind == "shell" for m in s.members]
    assert sorted(seen) == sorted(plates)
    h = 1.0 / 64
    clear = CS.SOLIDS_DEFAULT["wheel_clearance_cells"] * h
    shells = [Polygon(s.outline) for s in solids if s.kind == "shell"]
    for w in (s for s in solids if s.kind == "wheel"):
        disc = Point(*w.centre).buffer(w.radius, 256)
        for poly in shells:
            assert poly.distance(disc) >= clear - 1e-6


def test_the_wheels_sit_two_percent_of_their_radius_above_the_road_and_roll_at_road_speed(solids_fixed):
    solids, _rec = solids_fixed
    wheels = [s for s in solids if s.kind == "wheel"]
    assert len(wheels) == 2
    for w in wheels:
        gap = w.centre[1] - w.radius
        assert gap == pytest.approx(0.02 * w.radius, rel=1e-9)
        assert w.omega * w.radius == pytest.approx(1.0)
        g = OV.ogrid_annulus(w.name, w.centre[0], w.centre[1], w.radius, w.radius + 0.1, 64, 5)
        s = {w.name: w}
        # the car's wall velocity after the ramp: omega x (r - c), tangential, road speed at the bottom
        u = -w.omega * (g.y[0] - w.centre[1])
        v = w.omega * (g.x[0] - w.centre[0])
        nx_, ny_ = g.wall_normal()
        assert np.abs(u * nx_ + v * ny_).max() < 1e-12
        bottom = int(np.argmin(g.y[0]))
        assert u[bottom] == pytest.approx(1.0, rel=1e-3) and abs(v[bottom]) < 5e-2
        assert s


def test_the_ramp_starts_at_the_stream_and_ends_at_the_walls_own_velocity():
    assert CS.ramp(0.0) == 0.0 and CS.ramp(CS.RAMP) == 1.0 and CS.ramp(10.0) == 1.0
    ts = np.linspace(0, CS.RAMP, 101)
    vals = np.array([CS.ramp(t) for t in ts])
    assert np.all(np.diff(vals) >= 0)


def test_the_verification_runs_at_the_car_s_grid_settings():
    shared = {"body_thickness", "body_beta", "cluster", "sigma_wall", "kappa", "room", "room_floor",
              "wheel_thickness", "wheel_beta", "patch_half_width_r", "patch_height_r"}
    for k in shared:
        assert OM.MULTI_GRID[k] == CS.GRID[k], k
    assert OM.MULTI_GRID["patch_spacing_h"] == pytest.approx(CS.GRID["patch_spacing_h"])
    assert OM.MULTI_GRID["te_thickness"] == CS.SOLIDS_DEFAULT["aerofoil_te_thickness"]
    assert OM.MULTI_HOLE_MARGIN == CS.GRID["hole_margin"]


def test_ilu_and_jacobi_reach_the_same_step(multi_l3):
    ov = multi_l3
    nu = 0.01
    runs = []
    for pc in ("jacobi", "ilu"):
        f = NS.OversetFlow(ov, nu, 0.001, box={k: "dirichlet" for k in NS.FACES},
                           box_velocity=lambda x, y, t: NS.mms_fields(x, y, t, nu)[:2],
                           wall_velocity=lambda g, t: NS.mms_fields(g.x[0], g.y[0], t, nu)[:2],
                           forcing=lambda x, y, t: NS.mms_fields(x, y, t, nu)[3:], chi=1.0, precond=pc)
        u0, v0, p0, _a, _b = NS.mms_fields(f.X, f.Y, 0.0, nu)
        um, vm, _p, _c, _d = NS.mms_fields(f.X, f.Y, -0.001, nu)
        f.set_state(u0, v0, p0, u_prev=um, v_prev=vm)
        f.step()
        runs.append(f)
    # both solves stop at a relative residual of 1e-10 on matrices conditioned very
    # differently, so the steps agree to the solver's tolerance, not to round-off:
    # measured 4.6e-7 in u, well under the discretisation's own error here
    assert np.abs(runs[0].U - runs[1].U).max() < 1e-5
    assert np.abs(runs[0].P - runs[1].P).max() < 1e-4


# ---------------------------------------------------------------------------
# the record, and the page that quotes it
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def record():
    if not os.path.isfile(RECORD):
        pytest.skip("out/racelab11/racelab11.json is not here (it is carried upstream only)")
    with open(RECORD, encoding="utf-8") as fh:
        return json.load(fh)


def test_the_record_s_predictions_are_this_script_s(record):
    import tier62_car_solids as T
    assert record["prediction"] == [dict(p) for p in T.PREDICTION]
    assert record["read_before_this_run"] == list(T.READ_BEFORE_THIS_RUN)


def test_every_registered_prediction_has_a_verdict_and_the_judge_agrees(record):
    import tier62_car_solids as T
    assert record["verdicts_missing"] == []
    assert T.judge(record) == record["verdicts"]


def test_the_page_says_what_the_record_measured(record):
    if not os.path.isfile(PAGE):
        pytest.skip("the page is not here")
    text = open(PAGE, encoding="utf-8").read()
    for pid in record["verdicts"]:
        assert ("| %s |" % pid) in text, pid
    assert "{:,}".format(record["car"]["n_unknowns"]).replace(",", "{,}") in text
