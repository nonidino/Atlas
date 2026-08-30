"""L1.5 -- routing by composability, and the refusal on an empty feasible set."""

from __future__ import annotations

import numpy as np
import pytest

from atlas.capability import BCChannel, ExpertCapabilities, linear_response, port_decl
from atlas.holes import NamedHoleError
from atlas.ports import PortType
from atlas.routing import RoutingRefused, route, route_dynamically

SCALES = {"stress": 2.0, "velocity": 3.0, "power_area": 6.0}


def _expert(name, types=(PortType.MECH,), validity=None) -> ExpertCapabilities:
    ports = []
    for t in types:
        scales = SCALES if t is PortType.MECH else {
            "temperature": 2.0, "entropy_flux": 3.0, "power_area": 6.0
        }
        ports.append(port_decl(name=f"p:{t.value}", port_type=t, nondim=dict(scales),
                               effective_resolution=4))
    return ExpertCapabilities(
        expert_id=name,
        ports=ports,
        bc_channel=BCChannel.DIRICHLET,
        dt_native=1e-2,
        governing_family="fam",
        validity=validity,
        boundary_response=linear_response(np.eye(4)),
    )


class TestRouting:
    def test_it_ranks_by_composability_not_accuracy(self):
        """Ranking a library by benchmark accuracy selects for exactly the property
        accuracy benchmarks cannot see, and the two axes can point opposite ways."""
        accurate = _expert("accurate")
        composable = _expert("composable")
        result = route(
            "omega", [accurate, composable], [PortType.MECH],
            Xi={"accurate": 0.1, "composable": 0.9},
            tau={"accurate": 0.001, "composable": 0.05},   # accurate is 50x better on tau
        )
        assert result.chosen.expert_id == "composable"

    def test_tau_breaks_a_tie(self):
        a, b = _expert("a"), _expert("b")
        result = route("omega", [a, b], [PortType.MECH],
                       Xi={"a": 0.7, "b": 0.7}, tau={"a": 0.2, "b": 0.1})
        assert result.chosen.expert_id == "b"

    def test_an_empty_feasible_set_is_a_refusal_not_a_fallback(self):
        thermal_only = _expert("thermal", types=(PortType.THERM,))
        with pytest.raises(RoutingRefused) as exc:
            route("omega", [thermal_only], [PortType.MECH], Xi={"thermal": 0.9})
        assert "not a nearest-neighbour fallback" in str(exc.value)
        assert "ports do not match" in str(exc.value)

    def test_a_declining_validity_predicate_removes_a_candidate(self):
        ok = _expert("ok", validity=lambda s, c=None: True)
        declines = _expert("declines", validity=lambda s, c=None: False)
        result = route("omega", [ok, declines], [PortType.MECH],
                       Xi={"ok": 0.5, "declines": 0.99})
        assert result.chosen.expert_id == "ok"

    def test_an_unmeasured_Xi_returns_no_choice_rather_than_a_wrong_one(self):
        a, b = _expert("a"), _expert("b")
        result = route("omega", [a, b], [PortType.MECH])
        assert result.chosen is None
        assert "cannot rank" in result.note
        assert "substitution failure" in result.note

    def test_a_certificate_outside_its_regime_removes_a_candidate(self):
        a, b = _expert("a"), _expert("b")
        result = route(
            "omega", [a, b], [PortType.MECH], regime="Re=50000",
            Xi={"a": 0.9, "b": 0.5},
            certificate_valid=lambda caps, regime: caps.expert_id != "a",
        )
        assert result.chosen.expert_id == "b"
        row = [c for c in result.candidates if c.expert.expert_id == "a"][0]
        assert not row.certificate_valid_here

    def test_dynamic_routing_is_a_topology_mutation(self):
        with pytest.raises(NamedHoleError) as exc:
            route_dynamically()
        assert "TopologyEvent" in str(exc.value)
        assert "Static routing is the only mode admitted today" in str(exc.value)


class TestRoutingInsideTheCompiler:
    def test_a_declared_candidate_set_is_routed_at_compile_time(self):
        from atlas import compile_scheme
        from atlas.graph import Agent, CaseGraph, Connection

        chosen_a = _expert("agent-x-v1")
        alt_a = _expert("agent-x-v2")
        graph = CaseGraph(
            name="routed",
            agents=[
                Agent("x", chosen_a, candidates=(chosen_a, alt_a)),
                Agent("y", _expert("agent-y")),
            ],
            connections=[
                Connection("x-y", ("x", "p:MECH"), ("y", "p:MECH"), PortType.MECH,
                           derive_space=True, geometrically_coincident=True)
            ],
        )
        result = compile_scheme(graph)
        assert "x" in result.routes
        # Xi is unmeasured for the candidates, so routing declines to choose
        # rather than ranking them by something else.
        assert result.routes["x"].chosen is None
        assert any(d.rule == "G15" for d in result.decisions)
