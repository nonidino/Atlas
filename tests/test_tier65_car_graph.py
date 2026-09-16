"""Tier 65 -- the body-fitted column declared, and a volumetric device's port.

These pin what a declaration of an overset composite can get silently wrong: an
expert with nothing to declare, a port whose half is the wrong one, a device
whose two faces land on one agent, a bond that does not reproduce the march it
describes, a narrowing of a compiler premise that quietly moves another case,
and a record the page misquotes.
"""

from __future__ import annotations

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

from atlas import compiler as C                                     # noqa: E402
from atlas.capability import EllipticSubsolve, ExpertCapabilities, RecordError  # noqa: E402
from atlas.graph import Agent, CaseGraph, Decomposition, GraphError  # noqa: E402
from atlas.ports import PortType, ResponseHalf                      # noqa: E402
from atlas.cases import car_graph as CG                             # noqa: E402
from atlas.cases import integration_union as IU                     # noqa: E402

RECORD = os.path.join(HERE, "out", "racelab14", "racelab14.json")
PAGE = os.path.join(HERE, "wiki", "concepts", "Atlas 0.1", "common", "poc3-racelab-car-graph.md")


def _refusals(graph):
    res = C.compile_scheme(graph)
    return sorted({f"{d.layer}/{d.rule}" for d in res.decisions if d.verdict.value == "refuse"})


# ---------------------------------------------------------------------------
# what a record can and cannot say
# ---------------------------------------------------------------------------


def test_an_expert_with_nothing_to_declare_is_refused_by_the_record_layer():
    with pytest.raises(RecordError):
        ExpertCapabilities(expert_id="FLUID", ports=[])
    assert CG.no_port_record()["refused"] is True


def test_the_strip_port_is_the_devices_resolution_over_the_ducts_cells():
    caps = CG.composite_capabilities()
    assert [p.name for p in caps.ports] == ["core:MECH", "rotor:MECH"]
    assert CG.STRIP_SAMPLES == IU.DEVICE_CELLS == 32
    assert CG.STRIP_CELLS == 29
    for p in caps.ports:
        assert p.port_type is PortType.MECH
        # both sides of a MECH seam return the EFFORT in this package; a port that
        # returns the flow is refused at L3/C9, which is how this was found
        assert p.response_half is ResponseHalf.EFFORT
        assert str(CG.STRIP_CELLS) in p.geometry and str(CG.STRIP_SAMPLES) in p.geometry
    assert caps.elliptic_subsolve is EllipticSubsolve.EMBEDDED
    assert caps.substeps_per_macro_step == 1


def test_the_fluids_response_is_direct_forcing_about_the_band_it_is_written_at():
    caps = CG.composite_capabilities()
    band = np.full(CG.STRIP_SAMPLES, CG.U_BAND)
    at_base = np.asarray(caps.boundary_response("rotor:MECH", band), dtype=float)
    assert np.allclose(at_base, 0.0, atol=1e-18)          # no traction holds it where it is
    faster = np.asarray(caps.boundary_response("rotor:MECH", band + 0.01), dtype=float)
    assert np.all(faster > 0.0)                           # to speed the band up, push it
    assert np.allclose(caps.probe_base("rotor:MECH"), band)
    with pytest.raises(KeyError):
        caps.boundary_response("wet:MECH", band)
    with pytest.raises(ValueError):
        caps.boundary_response("rotor:MECH", band[:-1])


def test_a_seam_from_the_fluid_to_itself_is_refused_by_name():
    with pytest.raises(GraphError) as exc:
        CG.self_seam_graph()
    assert "itself" in str(exc.value)


# ---------------------------------------------------------------------------
# the compile
# ---------------------------------------------------------------------------


def test_the_body_fitted_column_refuses_the_clocks_and_nothing_else():
    g, aux = CG.build()
    assert len(g.agents) == 11 and len(g.connections) == 13
    assert g.decomposition is Decomposition.NON_OVERLAPPING
    assert _refusals(g) == ["L7/R9"]
    assert _refusals(CG.build(joins=())[0]) == []
    assert _refusals(CG.build(clocks="reconciled")[0]) == []


def test_no_seam_of_it_is_fluid_fluid_and_each_device_has_exactly_one():
    g, _aux = CG.build()
    fluid_side = [c for c in g.connections if "FLUID" in (c.a[0], c.b[0])]
    assert len(fluid_side) == 2
    assert {c.seam_id for c in fluid_side} == {"J1_core_strip", "J3_rotor_strip"}
    for c in fluid_side:
        assert c.port_type is PortType.MECH and c.expected_null_dim == 0
        assert c.a[0] in ("RAD", "ROTOR") and c.b[0] == "FLUID"


def test_the_cut_premise_ignores_an_agent_with_no_spatial_operator():
    """W285: the narrowing, on a two-agent fixture rather than on the car.

    An agent that declares a family and no spatial operator is not a piece of
    that family's region, so it cannot make a sole field solver read as cut --
    while two FIELD agents of one family still do.
    """
    from atlas.compiler import _decomposition_cuts
    field = CG.composite_capabilities()
    closure = CG.composite_capabilities()
    object.__setattr__(closure, "stencil_radius", 0)
    a = Agent("FLUID", field, domain="the composite")
    b = Agent("DISK", closure, domain="an algebraic closure")
    g = CaseGraph(name="t", agents=[a, b], connections=[],
                  decomposition=Decomposition.NON_OVERLAPPING)
    cut, uncut = _decomposition_cuts(g)
    assert cut == set() and uncut == {"FLUID", "DISK"}
    second = CG.composite_capabilities()
    g2 = CaseGraph(name="t2", agents=[a, Agent("FLUID2", second, domain="another piece")],
                   connections=[], decomposition=Decomposition.NON_OVERLAPPING)
    cut2, uncut2 = _decomposition_cuts(g2)
    assert cut2 == {"FLUID", "FLUID2"} and uncut2 == set()


def test_the_bond_is_the_marchs_own_arithmetic():
    """The turbine's declared effort times the band is the disk's own power."""
    from atlas.cases import ground_effect as GE
    from atlas.cases import vehicle_march as VM
    g, _aux = CG.build()
    rotor = next(a for a in g.agents if a.agent_id == "ROTOR")
    width = CG.STRIP_CELLS * GE.DX
    band = np.full(CG.STRIP_SAMPLES, CG.U_BAND)
    effort = np.asarray(rotor.capabilities.boundary_response("strip:MECH", band), dtype=float)
    els = __import__("atlas.cases.racelab", fromlist=["x"]).machine_for_host(
        0.08293773923977797, scale=width)
    _res, _e, disk = VM.operating_point(band, width=width, scale=width, elements=els)
    assert float(np.mean(effort)) == pytest.approx(disk.thrust / width, rel=1e-12)
    assert float(np.mean(effort) * np.mean(band) * width) == pytest.approx(disk.power, rel=1e-12)


# ---------------------------------------------------------------------------
# the record, and the page that quotes it
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def record():
    if not os.path.isfile(RECORD):
        pytest.skip("out/racelab14/racelab14.json is not here (it is carried upstream only)")
    with open(RECORD, encoding="utf-8") as fh:
        return json.load(fh)


def test_the_record_s_predictions_are_this_script_s_and_every_one_is_judged(record):
    import tier65_car_graph as T
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
    # the vault writes a thousands separator as LaTeX's `{,}`, so stripping the
    # comma alone leaves `12{}078` and never matches; Tier 62 pins the same
    # number the same way, and asserting the LaTeX form keeps the page in the
    # vault's convention rather than letting a bare 12078 satisfy it
    rows = record["record"]["composite"]["interpolation_rows"]
    assert "{:,}".format(rows).replace(",", "{,}") in text, rows
    assert ("%.4f" % record["power"]["rotor_ratio"]) in text
