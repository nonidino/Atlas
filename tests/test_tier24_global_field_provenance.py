"""Tier 24 -- W117, and whether a global field is hiding a seam.

`GlobalField` bypasses L3 by design: gravity enters each agent's update directly
and there is nothing to certify.  What the class could not express until
2026-09-02 is the difference between that and *routing a genuine two-agent
coupling through it*, which is numerically exact, turns off every check a seam
defines, and therefore **improves the envelope stamp by removing the quantities
that would have failed**.  `port-algebra-atlas-0.1` section 10.4 states the hazard;
CS-9 measured it; until W117 nothing read it -- `global_fields` was referenced
nowhere in the compiler, so the class had no reader at all.

Two groups:

  * **the rule** -- `L3/global-field` on the four cases of `produced_by`.  The
    undeclared case is the load-bearing one: the default is `None` and NOT `()`
    precisely so that silence is decertified rather than read as "external";
  * **the census** -- the rule run against every live declaration in the vault.
    This is W117a, the experiment the row was opened to run, and it is pinned
    here because its value is the whole decision: **the rule must fire on exactly
    the co-located substitute and clear the other six.**  If a later change makes
    it fire on a `pressure` declaration, either the rule is wrong or four of this
    vault's largest case studies are resting on the defect, and that is a finding
    nobody should have to rediscover by hand.
"""
from __future__ import annotations

import os

# Before numpy and torch are both in one process. The census test builds the
# torch-backed experts, and numpy.linalg.svd aborts outright without this.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

from atlas.capability import (BCChannel, ClaimType, Differentiable, Direction,
                              EllipticSubsolve, ExpertCapabilities, MotionClass,
                              TimeDiscretization, port_decl)
from atlas.compiler import compile_scheme
from atlas.graph import Agent, CaseGraph, Decomposition, GlobalField
from atlas.ports import PortType, ResponseHalf


def _caps(expert_id: str) -> ExpertCapabilities:
    return ExpertCapabilities(
        expert_id=expert_id,
        ports=[port_decl(
            name="face:MECH", port_type=PortType.MECH,
            geometry="a face", direction=Direction.BIDIRECTIONAL,
            nondim={"stress": 1.0, "velocity": 1.0, "power_area": 1.0},
            effective_resolution=4, motion_class=MotionClass.STATIC,
            response_half=ResponseHalf.EFFORT,
        )],
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=True,
        elliptic_subsolve=EllipticSubsolve.EXPOSED,
        time_discretization=TimeDiscretization.EXPLICIT,
        stencil_radius=1,
        substeps_per_macro_step=1,
        differentiable=Differentiable.NONE,
        dt_native=1.0e-2,
        storage=lambda *_a, **_k: 0.0,
        validity=lambda *_a, **_k: True,
        governing_family="toy",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash=f"toy/{expert_id}",
        deterministic=True,
    )


def _graph(field: GlobalField | None, ids=("A", "B")) -> CaseGraph:
    return CaseGraph(
        name="w117-probe",
        agents=[Agent(i, _caps(i), domain=i) for i in ids],
        connections=[],
        decomposition=Decomposition.OVERLAPPING,
        global_fields=[] if field is None else [field],
        macro_dt=1.0e-2,
    )


def _gf_decisions(graph: CaseGraph):
    return [d for d in compile_scheme(graph).decisions if d.rule == "global-field"]


# --------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------


def test_undeclared_provenance_is_decertified_not_admitted():
    """The default is None, and silence must not read as 'external'.

    This is the whole reason `produced_by` defaults to None rather than to the
    empty tuple: a silent default would certify every legacy declaration as an
    external field, which is exactly the thing the rule exists to catch.
    """
    d = _gf_decisions(_graph(GlobalField("p")))
    assert len(d) == 1
    assert d[0].verdict.value == "admit-uncertified"
    assert "does not declare its provenance" in d[0].message


def test_an_external_field_is_admitted():
    d = _gf_decisions(_graph(GlobalField("gravity", produced_by=())))
    assert len(d) == 1
    assert d[0].verdict.value == "admit"
    assert "external" in d[0].message


def test_a_field_produced_by_every_agent_is_a_global_operation_and_admitted():
    """A split-step pressure solve. The rule tests a PROPER subset for this reason."""
    d = _gf_decisions(_graph(GlobalField("pressure", produced_by=("A", "B"))))
    assert len(d) == 1
    assert d[0].verdict.value == "admit"
    assert "every agent" in d[0].message


def test_one_agents_state_applied_to_another_is_refused():
    """W117's case: a seam with the seam removed."""
    d = _gf_decisions(_graph(
        GlobalField("eigenstrain", applies_to=("B",), produced_by=("A",))))
    assert len(d) == 1
    assert d[0].verdict.value == "refuse"
    assert "with no seam" in d[0].message


def test_a_proper_subset_that_crosses_nothing_is_admitted():
    """Produced and consumed inside the same set: nothing is being carried."""
    d = _gf_decisions(_graph(
        GlobalField("local", applies_to=("A",), produced_by=("A",),),
        ids=("A", "B", "C")))
    assert len(d) == 1
    assert d[0].verdict.value == "admit"
    assert "crosses nothing" in d[0].message


def test_provenance_naming_an_absent_agent_is_refused():
    """Caught a real mis-declaration in channel_ns, whose agents are C-prefixed
    while the Tiling it reuses from window_ns names windows W-prefixed."""
    d = _gf_decisions(_graph(GlobalField("p", produced_by=("Z",))))
    assert len(d) == 1
    assert d[0].verdict.value == "refuse"
    assert "names no agent in this graph" in d[0].message


def test_a_graph_with_no_global_fields_gets_no_decision():
    assert _gf_decisions(_graph(None)) == []


# --------------------------------------------------------------------------
# W117a -- the census, pinned
# --------------------------------------------------------------------------


def _verdicts(graph) -> list[str]:
    return [d.verdict.value for d in _gf_decisions(graph)]


def test_w117a_the_rule_clears_every_legitimate_global_field_in_the_vault():
    """Six of seven. Each is a real declaration, not a fixture.

    `rocket` is gravity, `wind_farm` a declared absence, `window_ns` and
    `channel_ns` a split-step projection over agents that are ALL fluid windows,
    and `wind_farm_real` and `wake_array` the same projection over the fluid
    windows of a graph that also holds rotors -- which the rule clears because a
    rotor expert has no pressure input and no pressure state, so the field
    applies to no agent that does not produce it.
    """
    from atlas.cases import (channel_ns, rocket, wake_array, wind_farm,
                             wind_farm_real, window_ns)

    st = np.load("out/tier0b/s0_state.npz")
    u, v = st["u"], st["v"]
    wa = np.load("out/w93/state.npz")

    for name, graph in [
        ("rocket", rocket.build()),
        ("wind_farm", wind_farm.build()),
        ("window_ns", window_ns.build(u, v)[0]),
        ("channel_ns", channel_ns.build(u, v)[0]),
        ("wind_farm_real", wind_farm_real.build(u, v)),
        ("wake_array", wake_array.build(wa["u"], wa["v"])[0]),
    ]:
        assert _verdicts(graph) == ["admit"], f"{name} no longer clears the rule"


def test_w117a_the_rule_fires_on_the_co_located_substitute():
    """One of seven, and it is CS-9's `global-field` route.

    Before W117 this route COMPILED: numerically exact, certified by nothing, and
    carrying three fewer decertifications than the surface route that gets the
    answer wrong by 1279x. That is the trade section 10.4 calls the purest
    silent-wrongness instance this project has produced, and it is now refused.
    """
    from atlas.cases import thermal_strain

    graph, _ = thermal_strain.build(route="global-field")
    d = _gf_decisions(graph)
    assert [x.verdict.value for x in d] == ["refuse"]
    assert "carries one agent's state into another agent's update" in d[0].message
    assert compile_scheme(graph).decisions.refusals


def test_w117a_the_surface_route_is_untouched():
    """The rule is about global fields and must not disturb a declared seam."""
    from atlas.cases import thermal_strain

    graph, _ = thermal_strain.build(route="surface-mech")
    assert _gf_decisions(graph) == []


def test_the_volumetric_route_still_refuses_at_the_port_amendment():
    """W117 must not have moved where CS-9's own refusal comes from."""
    from atlas.cases import thermal_strain
    from atlas.holes import NamedHoleError

    with pytest.raises(NamedHoleError):
        thermal_strain.build(route="volumetric")
