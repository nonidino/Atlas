"""Tier 64 -- the devices in the body-fitted duct, and the three joins.

These pin what moving the devices from one lattice onto overlapping grids can
get silently wrong: a force whose samples do not integrate to its thrust, a
ring read from a grid that does not hold the point, a machine re-sized by one
similarity and not the other, a join that is re-derived where it should be
held, a trace the porous column's balances cannot read, a restart that is not a
continuation, and a record the page misquotes.
"""

from __future__ import annotations

import copy
import json
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
from atlas.cases import car_union as CU                             # noqa: E402
from atlas.cases import cooling_loop as CL                          # noqa: E402
from atlas.cases import overset as OV                               # noqa: E402
from atlas.cases import overset_multi as OM                         # noqa: E402
from atlas.cases import overset_ns as NS                            # noqa: E402
from atlas.cases import racelab as RL                               # noqa: E402
from atlas.cases import vehicle_march as VM                         # noqa: E402

RECORD = os.path.join(HERE, "out", "racelab13", "racelab13.json")
PAGE = os.path.join(HERE, "wiki", "concepts", "Atlas 0.1", "common", "poc3-racelab-car-union.md")


# ---------------------------------------------------------------------------
# the force and the band
# ---------------------------------------------------------------------------


def _device(key="ROTOR", x=2.7503, y_lo=0.40, y_hi=0.90, h=1.0 / 64):
    return CU.DuctDevice(key, x, y_lo, y_hi, CU.RING_UPSTREAM_CELLS * h, CU.HALF_WIDTH_CELLS * h)


def test_the_force_integrates_to_one_on_its_lattice_and_on_any_shifted_background():
    h = 1.0 / 64
    for x in (2.7503, 2.75, 2.7421875, 1.0 / 3.0):
        d = _device(x=x)
        _x, _y, w = d.strip_points(h)
        assert abs(w.sum() - 1.0) < 1e-13
        # a uniform column of cell centres at any offset: the raised cosine's
        # shifted copies are a partition of unity
        xs = (np.arange(400) + 0.5) * h
        assert abs(np.sum(d.profile_x(xs)) * h - 1.0) < 1e-13
    # and a force outside its band or its support is zero
    d = _device()
    assert d.shape(d.x_plane, d.y_hi + 1e-9) == 0.0
    assert d.shape(d.x_plane + d.half_width, 0.5 * (d.y_lo + d.y_hi)) == 0.0


def test_the_duct_band_is_read_from_the_rule_and_follows_an_edit():
    b = CU.duct_band()
    assert (b["y_lo"], b["y_hi"]) == (19.5, 48.5)
    assert (b["x_from"], b["x_to"]) == (248.5, 416.5)
    doc = copy.deepcopy(RL.load_geometry())
    doc["solids"] = {"thickness_cells": {"DUCT_LO": 5.0}}
    assert CU.duct_band(doc)["y_lo"] == 20.5
    devs = CU.car_devices()
    assert [d.key for d in devs] == ["RAD", "ROTOR"]
    assert all(abs(d.width / (1.0 / 64) - 29.0) < 1e-12 for d in devs)


def test_the_open_band_is_read_off_the_solids_outlines():
    h = 1.0 / 64
    rect = lambda x0, x1, y0, y1: np.array([[x0, y0], [x0, y1], [x1, y1], [x1, y0]]) * h  # noqa: E731
    solids = [CS.Solid("lo", rect(10, 50, 16.5, 19.5), ["LO"], ["duct"], "shell", 120.0),
              CS.Solid("up", rect(10, 50, 48.5, 51.5), ["UP"], ["duct"], "shell", 120.0)]
    ivs = CU.open_band(solids, 30.0, 0.0, 70.0)
    assert ivs == [(0.0, 16.5), (19.5, 48.5), (51.5, 70.0)]


def test_the_machine_s_band_is_a_property_of_the_declaration_not_of_the_inflow():
    w = 29.0 / 64.0
    a = CU.machine_band(0.0876, w)
    b = CU.machine_band(0.05, w)
    assert abs(a["ratio_lo"] - b["ratio_lo"]) < 1e-6 and abs(a["ratio_hi"] - b["ratio_hi"]) < 1e-6
    assert 0.96 < a["ratio_lo"] < 0.99 and 1.10 < a["ratio_hi"] < 1.20
    assert abs(a["at_the_sizing_point"]["induction"] - b["at_the_sizing_point"]["induction"]) < 1e-9
    # at the reference inflow the host similarity is the width similarity exactly
    ref = RL.machine_for_host(RL.U_HOST_REF, scale=w)
    base = VM.machine_for_rotor(w)
    assert all(ref[n].resistance == base[n].resistance for n in base)
    assert (ref["MGU"].k_e, ref["MGU"].k_t, ref["MGU"].r_total) == (base["MGU"].k_e, base["MGU"].k_t,
                                                                     base["MGU"].r_total)


# ---------------------------------------------------------------------------
# on a composite
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def multi_l3():
    return OM.multi_geometry(3)


def _xy(ov):
    X = np.empty(ov.n_unknowns)
    Y = np.empty(ov.n_unknowns)
    for g in ov.grids:
        idx = ov.index[g.name]
        live = idx >= 0
        gx, gy = (g.X, g.Y) if isinstance(g, OV.CartesianGrid) else (g.x, g.y)
        X[idx[live]] = gx[live]
        Y[idx[live]] = gy[live]
    return X, Y


def test_the_interpolation_matrix_is_car_solids_probe(multi_l3):
    ov = multi_l3
    X, Y = _xy(ov)
    f = np.sin(2.3 * X) * np.cos(1.7 * Y) + 0.3 * X
    rng = np.random.default_rng(3)
    # points in open flow, round the wheel and over the panel's grid
    px = np.concatenate([rng.uniform(0.2, 2.8, 300), 0.9 + 0.33 * np.cos(np.linspace(0, 6, 40)),
                         np.linspace(1.3, 2.2, 40)])
    py = np.concatenate([rng.uniform(0.7, 1.1, 300), 0.306 + 0.33 * np.sin(np.linspace(0, 6, 40)) + 0.0,
                         np.full(40, 0.21)])
    keep = py > 0.01
    px, py = px[keep], py[keep]
    # a point inside a body has no stencil anywhere; car_solids.probe says so with a NaN
    held = np.isfinite(CS.probe(ov, f, px, py))
    assert held.sum() >= 0.95 * held.size
    px, py = px[held], py[held]
    Q = CU.probe_matrix(ov, px, py)
    assert np.abs(Q @ f - CS.probe(ov, f, px, py)).max() < 1e-13
    lin = 0.7 * X - 1.3 * Y + 2.0
    assert np.abs(Q @ lin - (0.7 * px - 1.3 * py + 2.0)).max() < 1e-12
    with pytest.raises(OV.OversetError):
        CU.probe_matrix(ov, np.array([0.9]), np.array([0.306]))      # the wheel's centre


def _flow(ov, precond="ilu", forcing=None):
    return NS.OversetFlow(ov, 0.004, 0.0125,
                          box={"xlo": "dirichlet", "ylo": "dirichlet", "yhi": "dirichlet", "xhi": "outflow"},
                          wall_velocity=lambda g, t: (np.ones(g.ni), np.zeros(g.ni)), precond=precond,
                          forcing=forcing)


def test_a_union_on_the_uniform_stream_forces_the_disk_s_thrust_and_the_balances_read_it(multi_l3):
    ov = multi_l3
    h = 1.0 / 64
    devs = [_device("RAD", x=2.55), _device("ROTOR", x=2.75)]
    D = CU.DuctDevices(ov, devs, h=h)
    assert all(abs(v) < 1e-13 for v in D.quadrature_identity().values())
    flow = _flow(ov)
    n = ov.n_unknowns
    flow.set_state(np.ones(n), np.zeros(n), u_prev=np.ones(n), v_prev=np.zeros(n))
    union = CU.CarUnion(flow, D, u_host=1.0, t_on=0.0125, n_per_coolant=2)
    assert flow.forcing == D.forcing
    union.step()                                  # before t_on: nothing applied, nothing recorded
    assert union.trace["t"] == [] and D.amplitude == {"RAD": 0.0, "ROTOR": 0.0}
    assert np.abs(flow.U - 1.0).max() < 1e-12
    for _ in range(3):
        union.step()
    s = union.state
    # the ring read the stream, so the machine sized for it sits at the reference induction
    ref = CU.machine_band(1.0, D.devices["ROTOR"].width)["at_the_sizing_point"]
    assert len(union.trace["t"]) == 3
    assert union.trace["induction"][0] == pytest.approx(ref["induction"], abs=1e-9)
    assert union.trace["thrust_rotor"][0] == pytest.approx(ref["thrust"], rel=1e-9)
    # the core was built once, at the stream, and holds its reference
    assert union._core.u_air_ref == pytest.approx(1.0, abs=1e-12)
    assert union.trace["ua"][0] == pytest.approx(CL.UA_RAD, rel=1e-12)
    assert union.loop.legs["RAD"] is union._core
    # a coolant step every two union steps
    assert len(union.coolant) == 1 and union.coolant[0]["union_step"] == 2
    # the force slowed the flow at its strip, and only there
    assert D.plane_mean(flow.U, "ROTOR") < 1.0
    assert s.thrust_rotor > 0.0 and s.thrust_core > 0.0
    m = union.union_march(0.0, 1.0)
    assert set(CU.TRACE_KEYS) <= set(m.trace)
    bal = VM.receiver_balances(m, frac=1.0)
    # in OPEN flow the strip slows the stream where it acts more than on a ring
    # upstream, so the force's work falls short of the shaft's claim
    assert 0.8 < bal["J3"]["ratio_with_the_term"] < 1.0
    assert bal["J2"]["coolant steps taken"] == 1
    assert union.trace["mgu_valid"][0] is True and union.trace["rotor_valid"][0] is True
    assert union.envelope["FLUID"]["valid"] is True


def test_a_kept_state_continues_the_march_it_was_taken_from(multi_l3, tmp_path):
    ov = multi_l3
    nu = 0.01
    kw = dict(box={k: "dirichlet" for k in NS.FACES},
              box_velocity=lambda x, y, t: NS.mms_fields(x, y, t, nu)[:2],
              wall_velocity=lambda g, t: NS.mms_fields(g.x[0], g.y[0], t, nu)[:2],
              forcing=lambda x, y, t: NS.mms_fields(x, y, t, nu)[3:], chi=1.0, precond="jacobi")
    a = NS.OversetFlow(ov, nu, 0.001, **kw)
    b = NS.OversetFlow(ov, nu, 0.001, **kw)
    for f in (a, b):
        u0, v0, p0, _x, _y = NS.mms_fields(f.X, f.Y, 0.0, nu)
        um, vm, _p, _c, _d = NS.mms_fields(f.X, f.Y, -0.001, nu)
        f.set_state(u0, v0, p0, u_prev=um, v_prev=vm)
    for _ in range(3):
        a.step()
    b.step()
    path = CU.save_state(b, str(tmp_path / "state.npz"))
    b.set_state(np.zeros(ov.n_unknowns), np.zeros(ov.n_unknowns))
    got = CU.load_state(b, path)
    assert got["t"] == pytest.approx(0.001)
    for _ in range(2):
        b.step()
    # with a preconditioner that keeps no state, a restart IS the continuation
    assert np.array_equal(a.U, b.U) and np.array_equal(a.V, b.V) and np.array_equal(a.P, b.P)
    with pytest.raises(ValueError):
        CU.save_state(b, str(tmp_path / "state.npy"))


# ---------------------------------------------------------------------------
# the record, and the page that quotes it
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def record():
    if not os.path.isfile(RECORD):
        pytest.skip("out/racelab13/racelab13.json is not here (it is carried upstream only)")
    with open(RECORD, encoding="utf-8") as fh:
        return json.load(fh)


def test_the_record_s_predictions_are_this_script_s_and_every_one_is_judged(record):
    import tier64_car_union as T
    assert record["prediction"] == [dict(p) for p in T.PREDICTION]
    assert record["read_before_this_run"] == list(T.READ_BEFORE_THIS_RUN)
    assert record["verdicts_missing"] == []
    assert T.judge(record) == record["verdicts"]


def test_the_page_says_what_the_record_measured(record):
    if not os.path.isfile(PAGE):
        pytest.skip("the page is not here")
    text = open(PAGE, encoding="utf-8").read()
    for pid in record["verdicts"]:
        assert ("| %s |" % pid) in text, pid
    import tier64_car_union as T
    S = T._registered(record)
    assert ("%.4f" % S["turbine_u_mean_window"]) in text
    assert ("%.5f" % S["balances"]["J3"]["u_plane / u_ring"]) in text
    # the two that failed are named as failures, not smoothed over
    assert text.count("**failed**") >= 2
