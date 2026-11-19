"""The coolant plan's Phase 2 in the studio: the channel cut (`atlas/studio/channel_cut.py`),
the development layouts (`cot_layouts.py`), the general flow and Danckwerts' inlet
(`cot.py`, `transport_flow`)."""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from atlas.chartop import certificate as cert  # noqa: E402
from atlas.chartop import transport as tr  # noqa: E402
from atlas.chartop.host import monolithic  # noqa: E402
from atlas.studio import cot  # noqa: E402
from atlas.studio import cot_layouts as CL  # noqa: E402
from atlas.studio import document as D  # noqa: E402
from atlas.studio import pieces as P  # noqa: E402


@pytest.fixture(scope="module")
def winding():
    spec, geom = CL.build("dev-winding", "slow")
    return spec, geom, CL.cut(spec, "dev-winding", "slow")


def _nodes(L, fields):
    out = {}
    for p, u in zip(L.problems, fields):
        for xy, v in zip(np.round(p.chart.X.reshape(-1, 2) * 1e9).astype(np.int64), np.asarray(u)):
            out[(int(xy[0]), int(xy[1]))] = float(v)
    return out


def test_a_winding_channel_cuts_into_conforming_certified_pieces(winding):
    spec, geom, cut = winding
    assert cut.cutter == "channels" and len(cut.pieces) == 32
    assert {p.material for p in cut.pieces} == {"aluminium", "water", "device"}
    assert sum(p.material == "water" for p in cut.pieces) == 8
    for p in cut.pieces:
        assert cert.certify(P.chart_of(p, cut.n).X).ok
    L = cot.layout(cut, spec)
    assert all(e.conforming for e in L.edges if e.other is not None)
    assert sum(1 for s in L.seams if s.kind == "advective") == 7


def test_whole_traces_on_curved_pieces_equal_the_monolithic_solve(winding):
    spec, geom, cut = winding
    L = cot.layout(cut, spec, m=cut.n)
    mono = monolithic(L)
    cp = L.solve(tr.ClassicalTransportExpert())
    top = max(float(np.max(np.abs(u))) for u in mono.fields)
    assert max(float(np.max(np.abs(a - b))) for a, b in zip(cp.fields, mono.fields)) <= 1e-10 * top


def test_the_finer_cut_is_the_same_mesh(winding):
    spec, geom, cut = winding
    fine = CL.cut(spec, "dev-winding", "slow", cols_factor=2, rows=2)
    assert len(fine.pieces) == 4 * len(cut.pieces) and fine.n == cut.n // 2
    La, Lb = cot.layout(cut, spec), cot.layout(fine, spec)
    A, B = _nodes(La, monolithic(La).fields), _nodes(Lb, monolithic(Lb).fields)
    assert set(A) == set(B)
    assert max(abs(A[k] - B[k]) for k in A) <= 1e-10 * max(abs(v) for v in A.values())


def test_danckwerts_inlet_holds_no_corner(winding):
    """The inlet is natural for the host: no coefficient on the left edge is fixed, and
    the coolant pieces at the mouth declare it, at 300 K."""
    spec, geom, cut = winding
    L = cot.layout(cut, spec)
    assert not L.fixed
    decl = [p.inlet for p in L.problems if p.inlet]
    assert len(decl) == 1 and list(decl[0].values()) == [CL.T_IN]
    # every watt the device makes leaves with the coolant (insulated plate, steady)
    mono = monolithic(L)
    q = sum(p.meta["q"] * p.meta["area"] for p in L.problems)
    water = cot._materials(spec)["water"]
    rise = cot.layout(cut, spec).solve(tr.ClassicalTransportExpert()).outlet_bulk() - CL.T_IN
    flux = [sm.flux for sm in L.seams if sm.kind == "inlet"]
    mdot_cp = -sum(flux)
    # no side is held, and the constant is in the coupled test space: exactly conserved
    assert mdot_cp * rise == pytest.approx(q, rel=1e-8)
    assert mdot_cp == pytest.approx(water["rho"] * water["cp"] * spec.physics.params["u_coolant"] * CL.WIDTH,
                                    rel=1e-9)
    del mono


def test_two_parallel_channels_each_carry_the_mean_speed():
    spec, _ = CL.build("dev-parallel", "slow")
    cut = CL.cut(spec, "dev-parallel", "slow")
    psis, info = cot.flow(cut, spec)
    assert len(info) == 2
    for ch_ in info:
        assert ch_["flux"] == pytest.approx(spec.physics.params["u_coolant"] * CL.WIDTH, rel=1e-9)
        assert ch_["inlet_width"] == pytest.approx(CL.WIDTH, rel=1e-9)


def test_the_layouts_are_built_the_same_every_time():
    a = D.dumps(*CL.build("dev-winding", "fast"))
    b = D.dumps(*CL.build("dev-winding", "fast"))
    assert a == b


@pytest.mark.parametrize("name,tier", [(n_, t_) for n_ in CL.NAMES for t_ in CL.TIERS])
def test_each_layout_sits_at_its_target_peclet_number(name, tier):
    spec, _ = CL.build(name, tier)
    probs = cot.problems(CL.cut(spec, name, tier), spec)
    n = CL.CUT[name]["n"][tier]
    assert max(p.peclet() for p in probs) == pytest.approx(CL.TARGET_PE[(tier, n)], rel=0.01)
    assert all(tr.envelope_refusals(p) == [] for p in probs)


def _plate_with(features, H=0.008):
    spec, _ = CL.build("dev-block", "slow")
    from atlas.studio import geometry_eval as GE
    g = D.Geometry(features=[D.Feature(id="f1", kind="rect", role="add", name="Plate", x=0, y=0,
                                       w=CL.W, h=H, material="aluminium")] + features)
    spec.regions = []
    applied, _res = GE.apply(spec, g, dx=CL.DX)
    applied.boundaries = spec.boundaries
    return applied


def test_refused_a_channel_that_does_not_cross_the_plate():
    ch_ = D.Feature(id="f2", kind="channel", role="region", material="water", w=CL.WIDTH,
                    points=[(0.0, 0.004), (0.010, 0.004)], edges=["line"])
    with pytest.raises(P.CutError, match="does not cross the plate"):
        P.cut_channels(_plate_with([ch_]), 32, cols=4)


def test_refused_a_solid_region_that_is_not_a_rectangle():
    ch_ = D.Feature(id="f2", kind="channel", role="region", material="water", w=CL.WIDTH,
                    points=[(0.0, 0.006), (CL.W, 0.006)], edges=["line"])
    blob = D.Feature(id="f3", kind="ellipse", role="region", material="device", cx=0.008, cy=0.002,
                     rx=0.001, ry=0.001)
    with pytest.raises(P.CutError, match="neither a coolant channel nor a rectangle"):
        P.cut_channels(_plate_with([ch_, blob]), 32, cols=4)


def test_refused_two_channels_whose_walls_cross():
    """Walls that cross are refused; walls that coincide everywhere (two channels drawn
    side by side) are one interface, so the two read as one wider channel."""
    a = D.Feature(id="f2", kind="channel", role="region", material="water", w=CL.WIDTH,
                  points=[(0.0, 0.004), (CL.W, 0.004)], edges=["line"])
    b = D.Feature(id="f3", kind="channel", role="region", material="water", w=CL.WIDTH,
                  points=[(0.0, 0.002), (0.002, 0.002), (CL.W - 0.002, 0.0065), (CL.W, 0.0065)],
                  edges=["line"] * 3)
    with pytest.raises(P.CutError, match="cross or touch"):
        P.cut_channels(_plate_with([a, b], H=0.010), 32, cols=4)
    c = D.Feature(id="f3", kind="channel", role="region", material="water", w=CL.WIDTH,
                  points=[(0.0, 0.006), (CL.W, 0.006)], edges=["line"])
    cut = P.cut_channels(_plate_with([a, c], H=0.010), 32, cols=4)
    assert sum(p.material == "water" for p in cut.pieces) == 8


def test_a_session_cuts_a_drawn_channel_automatically():
    from atlas.studio.session import Session
    spec, geom = CL.build("dev-winding", "slow")
    s = Session(spec=spec, key="x")
    s.geom = geom
    s.pieces_cut(None, {"cols": 8}, 32, blocking=True)
    assert s.pieces_job.status == "done", s.pieces_job.error
    sm = s.pieces_summary
    assert sm["cutter"] == "channels" and sm["physics"] == "CO-T" and len(sm["pieces"]) == 32
    assert sm["advective_seams"] == 7 and sm["refused"] is None
