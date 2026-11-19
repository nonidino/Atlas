"""The studio's CO-T path (the coolant plan's Phase 1, `atlas/studio/cot.py`): a
conjugate-heat case cut into pieces, the classical CO-T expert on the Pieces screen, and
Run's Compare against the full domain.

The case is the workbench's ``cooled-block`` (copper, a chip, a water channel along its
top) with the coolant slowed to 0.1 mm/s, so a grid cut with two columns keeps every
coolant piece under the owner's Peclet cap of 200 (the example's own 1 cm/s would need
about a hundred pieces along the channel: the refusal below names that).
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from atlas.chartop import transport as tr  # noqa: E402
from atlas.chartop.host import monolithic  # noqa: E402
from atlas.studio import cot  # noqa: E402
from atlas.studio.session import Refused, Session  # noqa: E402
from atlas.workbench.spec import example_case  # noqa: E402

SLOW = 1e-4


def _session(u=SLOW, n=64, cols=2):
    s = Session(spec=example_case("cooled-block"), key="x")
    s.edit_ops([{"op": "set", "path": ["physics", "params", "u_coolant"], "value": u}], "speed")
    s.pieces_cut(None, {"cols": cols}, n, blocking=True)
    assert s.pieces_job.status == "done", s.pieces_job.error
    return s


@pytest.fixture(scope="module")
def slow():
    return _session()


def test_the_summary_is_co_t(slow):
    sm = slow.pieces_summary
    assert sm["physics"] == "CO-T" and sm["class"] == "G0′" and sm["refused"] is None
    assert len(sm["pieces"]) == 12 and sm["advective_seams"] == 3
    water = [p for p in sm["pieces"] if p["material"] == "water"]
    assert len(water) == 4
    for p in water:
        assert p["flow"]["flows"] and p["flow"]["ports"] == ["THERM", "ADVEC"]
        assert 0.0 < p["flow"]["pe"] <= tr.PE_MAX and p["flow"]["cell_pe"] <= tr.CELL_PE_MAX
        assert p["envelope"]["refusals"] == [] and p["envelope"]["card"] == "CO-T"
    for p in sm["pieces"]:
        if p["material"] != "water":
            assert p["flow"]["pe"] == 0.0 and p["flow"]["ports"] == ["THERM"]
    kinds = [sd["kind"] for p in water for sd in p["sides"]]
    assert kinds.count("THERM+ADVEC") == 6            # three seams, seen from both sides
    flows = sorted(sd.get("flow") for p in water for sd in p["sides"] if sd["to"] is None and sd.get("flow"))
    assert "inlet" in flows and "outlet" in flows
    # the classical CO-T expert is the only one, and it is admitted everywhere here
    assert [r["key"] for r in slow.experts()] == ["classical"]
    assert all(p["admitted"] == {"classical": None} for p in sm["pieces"])


def test_the_mean_speed_and_the_profile(slow):
    """The coolant pieces carry the plane Poiseuille profile at the case's mean speed:
    the heat-capacity flux through the inlet is rho c U w per metre of depth."""
    probs = cot.problems(slow.cut, slow.spec)
    L = cot.layout(slow.cut, slow.spec)
    water = cot._materials(slow.spec)["water"]
    want = water["rho"] * water["cp"] * SLOW * 0.010
    inlet = [sm for sm in L.seams if sm.kind == "inlet"]
    assert len(inlet) == 1
    assert -inlet[0].flux == pytest.approx(want, rel=1e-12)
    assert sum(1 for p in probs if p.flows) == 4


def test_the_check_is_round_off_with_whole_traces(slow):
    slow.pieces_check(blocking=True)
    assert slow.pieces_job.status == "done", slow.pieces_job.error
    c = slow.pieces_summary["check"]
    assert c["full"]["max_diff_rel"] <= 1e-9 and c["full"]["balance_rel"] <= 1e-10
    assert c["full"]["refusals"] == [] and c["full"]["min_eig"] > 0.0
    assert c["ports16"]["max_diff_rel"] > 1e-6            # 16 modes are a floor here, reported
    assert c["ports16"]["outlet_bulk_err_rel"] is not None


def _no_raw_seconds(obj, path="result"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert not (k.endswith("_s") or "seconds" in k or k in ("times", "mean", "median")), f"{path}.{k}"
            _no_raw_seconds(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for v in obj:
            _no_raw_seconds(v, path)


def test_compare_against_the_full_domain(slow):
    run = slow.start_corun("compare", {"method": "full"}, {"method": "chart", "solver": "classical"},
                           blocking=True)
    assert run.status == "done", run.error
    r = run.result
    assert r["physics"] == "CO-T" and r["certificate"]["class"] == "G0′"
    assert r["certificate"]["charts"] == [12, 12] and r["certificate"]["min_eig"] > 0.0
    assert r["seams"]["advective"] == 3 and r["seams"]["max_jump"] < 1e-9
    assert r["e4"]["passed"] is None and "not registered" in r["e4"]["title"]
    assert r["price"]["ratio"] > 0 and r["price"]["turns"] == 5
    # B's error is the classical pieces' 16-mode floor: the same number the host gives directly
    L = cot.layout(slow.cut, slow.spec)
    mono = monolithic(L)
    cp = L.solve(tr.ClassicalTransportExpert())
    er = cp.errors(mono.fields)
    assert r["accuracy"]["energy_rel"] == pytest.approx(er["energy_rel"], rel=1e-9)
    assert r["outlet"]["bulk"] == pytest.approx(er["outlet_bulk"], rel=1e-12)
    assert r["balance"]["A"] <= 1e-10
    _no_raw_seconds(r)
    fr = run.frame(0)
    assert set(fr) == {"A", "B", "difference"} and fr["A"]["nx"] > 0


def test_single_classical_pieces(slow):
    run = slow.start_corun("single", {"method": "chart", "solver": "mixed"}, None, blocking=True)
    assert run.status == "done", run.error
    assert "price" not in run.result and run.result["accuracy"]["energy_rel"] > 0.0


def test_a_learned_expert_is_refused_by_name(slow):
    run = slow.start_corun("compare", {"method": "full"}, {"method": "chart", "solver": "TL0"},
                           blocking=True)
    assert run.status == "failed" and "classical expert only" in run.error


def test_a_fast_channel_is_refused_with_what_to_do():
    s = _session(u=1e-2, n=128)
    water = [p for p in s.pieces_summary["pieces"] if p["material"] == "water"]
    assert all(any("cut the channel into shorter pieces" in r for r in p["envelope"]["refusals"])
               for p in water)
    run = s.start_corun("compare", {"method": "full"}, {"method": "chart", "solver": "classical"},
                        blocking=True)
    assert run.status == "failed" and "above the cap of 200" in run.error


def test_a_coarse_grid_is_refused_for_its_cell_peclet_number():
    s = _session(n=32)
    msgs = [r for p in s.pieces_summary["pieces"] for r in p["envelope"]["refusals"]]
    assert msgs and all("cell Peclet" in m for m in msgs)


def test_a_curved_channel_is_refused_named():
    s = Session(spec=example_case("cooled-winding"), key="x")
    s.pieces_cut("grid", {}, 32, blocking=True)
    assert s.pieces_job.status == "failed"


def test_a_coolant_region_drawn_as_a_polygon_gets_the_same_flow(slow):
    """Phase 2 made the flow general (`transport_flow`): a channel drawn as a polygon has
    the rectangle's stream function, cut along its walls."""
    spec = slow.spec.model_copy(deep=True)
    from atlas.workbench.spec import Region
    r = spec.regions[2]
    spec.regions[2] = Region(id=r.id, material=r.material, shape="polygon", x0=0, y0=50,
                             nx=200, ny=10, points=[(0, 50), (200, 50), (200, 60), (0, 60)])
    from atlas.studio.pieces import cut_channels
    cut = cut_channels(spec, 64, cols=2)
    psis, info = cot.flow(cut, spec)
    assert len(info) == 1 and info[0]["flux"] == pytest.approx(SLOW * 0.010, rel=1e-9)


def test_other_kinds_are_still_refused():
    s = Session(spec=example_case("plume-2"), key="x")
    with pytest.raises(Refused, match="CO-T"):
        s.pieces_cut(None, {}, 32, blocking=True)


def test_the_bc_reads_stretches_of_grid_edges(slow):
    bc = cot.make_bc(slow.spec)
    dx = slow.spec.domain.dx
    inlet = np.array([[0.0, 52 * dx], [0.0, 58 * dx]])
    block = np.array([[0.0, 10 * dx], [0.0, 40 * dx]])
    outlet = np.array([[200 * dx, 52 * dx], [200 * dx, 58 * dx]])
    # the inlet is natural for the host: its coolant piece carries it (Danckwerts)
    assert bc(0, 3, inlet) == ("insulated", None) and cot.inlets(slow.spec)(inlet) == 300.0
    assert bc(0, 3, block) == ("insulated", None) and cot.inlets(slow.spec)(block) is None
    assert bc(0, 1, outlet) == ("insulated", None)
    probs = cot.problems(slow.cut, slow.spec)
    assert [p.inlet for p in probs if p.inlet] == [{3: 300.0}]
