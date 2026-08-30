"""The compiler: three verdicts, every refusal citing a rule, and the two traces.

The two compile traces at the bottom are the load-bearing tests.  They assert
that the compiler reproduces what end-to-end-architecture-spec §13 says the spec
does to the two case studies that exist -- one of which would have been refused
at compile time for the finding it was built to measure, and one of which does
not compile at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from atlas import compile_scheme
from atlas.cases import rocket, wind_farm
from atlas.capability import BCChannel, ExpertCapabilities, Transmission, port_decl
from atlas.claims import ClaimTypeTag
from atlas.emit import EMIT_GROUPS
from atlas.envelope import Hypothesis, Status
from atlas.graph import Decomposition
from atlas.holes import Unmeasured, UnmeasuredError
from atlas.ports import PortType
from atlas.scheme import Accelerator, Ordering
from atlas.verdict import ADMIT, ADMIT_UNCERTIFIED, REFUSE


def rules_cited(result) -> set[str]:
    return {f"{d.layer}/{d.rule}" for d in result.decisions}


class TestEveryDecisionCitesARule:
    def test_no_decision_is_uncited(self):
        for graph in (wind_farm.build(), rocket.build()):
            result = compile_scheme(graph)
            for d in result.decisions:
                assert d.layer and d.rule, f"uncited decision: {d.message[:60]}"
                assert d.message

    def test_a_refusal_names_the_quantity_that_would_have_been_wrong(self):
        result = compile_scheme(rocket.build())
        for d in result.decisions.refusals:
            assert d.quantity or d.failure_class.value == "structural", (
                f"{d.layer}/{d.rule} refuses without naming a quantity or being structural"
            )


class TestUnmeasuredConstants:
    def test_an_unmeasured_constant_refuses_to_be_a_number(self):
        L = Unmeasured("L", "W1")
        with pytest.raises(UnmeasuredError):
            float(L)
        with pytest.raises(UnmeasuredError):
            _ = L + 1
        assert not L

    def test_the_compile_reports_which_constants_it_lacks(self):
        result = compile_scheme(wind_farm.build())
        joined = " ".join(result.unmeasured)
        assert "L (W1)" in joined
        assert "tau (W3)" in joined


class TestWindFarmAsBuilt:
    """end-to-end-architecture-spec §13.1, as a test."""

    @pytest.fixture(scope="class")
    def result(self):
        return compile_scheme(wind_farm.build())

    def test_it_is_refused(self, result):
        assert result.verdict is REFUSE
        assert not result.runnable

    def test_the_stamp_matches_the_spec_trace(self, result):
        s = result.envelope
        assert s[Hypothesis.E1] is Status.HOLDS
        assert s[Hypothesis.E2] is Status.HOLDS
        assert s[Hypothesis.E3] is Status.HOLDS
        assert s[Hypothesis.E4] is Status.HOLDS
        assert s[Hypothesis.E5] is Status.UNCHECKED
        assert s[Hypothesis.E6] is Status.UNCHECKED
        assert s[Hypothesis.E7] is Status.UNCHECKED
        assert not s.bound_applies

    def test_L3_refuses_for_a_missing_declaration_not_an_unbuilt_mechanism(self, result):
        refusals = [d for d in result.decisions.refusals if d.rule == "C2/C3/C6"]
        assert refusals
        assert "missing declaration" in refusals[0].message
        assert "not an unbuilt mechanism" in refusals[0].message

    def test_the_probe_finds_the_interface_problem_empty(self, result):
        empty = [op for op in result.seam_operators.values() if op.is_empty]
        assert empty, "the frozen checkpoint's seams should probe to a zero operator"
        for op in empty:
            for block in op.blocks.values():
                if block.agent_id in ("I", "N", "F", "W", "Bp", "Bm"):
                    assert block.Xi == 0.0

    def test_the_word_coupled_is_refused_on_the_output(self, result):
        assert result.refused_claims
        assert all("coupled" in c for c in result.refused_claims)

    def test_R6_refuses_the_direct_and_krylov_solvers(self, result):
        assert "L5/R6" in rules_cited(result)
        assert result.scheme.accelerator is Accelerator.RICHARDSON

    def test_the_window_is_set_by_R3_not_defaulted(self, result):
        """W = 1 here is a RULE, not a default, and the two must not be confused.

        The frozen expert declares bc_time_varying false, so R3 forbids a longer
        window outright -- a ring held constant across the window is a W = 1
        scheme wearing a longer name. That is a different fact from "we could not
        compute W* because L is unmeasured", which is the boundary-capable case
        below, and the scheme records which one applies.
        """
        assert result.scheme.window == 1
        assert "W_window" not in result.scheme.defaulted
        assert "R3" in result.scheme.why["W_window"]
        assert "eps_tol" in result.scheme.defaulted

    def test_the_claim_is_untyped(self, result):
        assert result.artifact.claim.tag is ClaimTypeTag.UNTYPED

    def test_the_finding_was_available_at_compile_time(self, result):
        """The negative composed result was a refusal, not a measurement."""
        assert result.seam_operators, "the probe must run even on a refused graph"
        # Nothing in the compile requires a rollout.
        assert result.artifact.bound_terms.per_step_defect is None


class TestWindFarmBoundaryCapable:
    """The 'next' and 'target' rows of the coupling scheme's instantiation table."""

    def test_R2_lifts_the_rung_and_switches_the_axis(self):
        result = compile_scheme(wind_farm.build(boundary_capable=True, declare_transfer=True))
        assert result.scheme.transmission is Transmission.PROBED_DTN
        assert result.scheme.decomposition is Decomposition.NON_OVERLAPPING

    def test_the_better_operator_introduces_the_cross_point_difficulty(self):
        """The trade is made visible at L2 rather than discovered later."""
        result = compile_scheme(wind_farm.build(boundary_capable=True, declare_transfer=True))
        assert "L2/I2/G1" in {f"{d.layer}/{d.rule}" for d in result.decisions.refusals}
        assert result.verdict is REFUSE

    def test_with_primal_cross_point_dofs_it_admits_uncertified(self):
        graph = wind_farm.build(boundary_capable=True, declare_transfer=True)
        graph.primal_cross_point_dofs = tuple(a.agent_id for a in graph.agents)
        result = compile_scheme(graph)
        assert result.verdict is ADMIT_UNCERTIFIED
        assert result.runnable
        assert result.scheme.accelerator is Accelerator.DIRECT_SCHUR
        assert result.scheme.ordering is Ordering.ADDITIVE

    def test_the_window_defaults_only_when_W_star_is_uncomputable(self):
        """With bc_time_varying true, R3 no longer decides W -- and W* needs L."""
        graph = wind_farm.build(boundary_capable=True, declare_transfer=True)
        graph.primal_cross_point_dofs = tuple(a.agent_id for a in graph.agents)
        result = compile_scheme(graph)
        assert result.scheme.window == 1
        assert "W_window" in result.scheme.defaulted
        assert "UNCOMPUTABLE" in result.scheme.why["W_window"]

    def test_passivity_is_measured_from_the_probe(self):
        graph = wind_farm.build(boundary_capable=True, declare_transfer=True)
        graph.primal_cross_point_dofs = tuple(a.agent_id for a in graph.agents)
        result = compile_scheme(graph)
        assert result.envelope[Hypothesis.E7] is Status.HOLDS
        op = next(iter(result.seam_operators.values()))
        assert op.beta is not None and op.beta > 0
        assert op.kappa is not None
        assert op.cut_score is not None


class TestRocketAscent:
    """end-to-end-architecture-spec §13.2: the rocket does not compile."""

    @pytest.fixture(scope="class")
    def result(self):
        return compile_scheme(rocket.build())

    def test_it_does_not_compile(self, result):
        assert result.verdict is REFUSE

    def test_E3_fails_at_the_fluid_structure_seam(self, result):
        assert result.envelope[Hypothesis.E3] is Status.FAILS
        assert any(s.startswith("b-c") for s in result.tau_undefined_seams)

    def test_the_rung_choice_survives_E3s_failure(self, result):
        """Probing never mentions a governing equation, so it works at a seam
        where no single global evolution operator exists."""
        bc_seams = [s for s in result.seam_operators if s.startswith("b-c")]
        assert bc_seams
        for s in bc_seams:
            assert result.seam_operators[s].beta is not None

    def test_moving_interfaces_are_refused(self, result):
        refusals = {d.rule for d in result.decisions.refusals}
        assert "InterfaceMotion" in refusals
        assert result.envelope[Hypothesis.E2] is Status.FAILS

    def test_multirate_is_refused_under_pointwise_flux_matching(self, result):
        assert "R9" in {d.rule for d in result.decisions.refusals}
        assert result.envelope[Hypothesis.E4] is Status.FAILS
        assert result.scheme.multirate

    def test_it_trips_three_of_the_five_named_slots(self, result):
        touched = result.holes.touched()
        assert {"InterfaceMotion", "SeamReference", "AssemblyCertificate"} <= touched
        assert "TopologyEvent" not in touched      # staging is out of scope here

    def test_staging_trips_the_fourth(self):
        result = compile_scheme(rocket.build(with_staging=True))
        assert "TopologyEvent" in result.holes.touched()
        assert result.envelope[Hypothesis.E1] is Status.FAILS
        refusal = [d for d in result.decisions.refusals if d.rule == "TopologyEvent"]
        assert refusal and "ledger" in refusal[0].message


class TestTheEmitContract:
    def test_the_artifact_carries_every_group(self):
        result = compile_scheme(wind_farm.build())
        d = result.artifact.as_dict()
        for group in EMIT_GROUPS:
            assert group in d, f"the emit contract is missing the {group!r} group"

    def test_every_defect_carries_its_depth(self):
        result = compile_scheme(wind_farm.build(), depth=2)
        assert result.artifact.bound_terms.as_dict()["depth"] == 2
        assert result.artifact.claim.depth == 2

    def test_the_artifact_serializes_even_on_a_refusal(self, tmp_path):
        result = compile_scheme(rocket.build())
        assert result.verdict is REFUSE
        path = result.artifact.write(str(tmp_path / "run.json"))
        import json

        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["verdict"] == "refuse"
        assert len(data["envelope_stamp"]) == 7
        assert data["decisions"]

    def test_a_run_that_cannot_stamp_E1_to_E4_is_refused(self):
        from atlas.emit import EmitRefused, RunArtifact
        from atlas.envelope import EnvelopeStamp
        from atlas.verdict import DecisionRecord

        art = RunArtifact("bare", ADMIT, EnvelopeStamp(), DecisionRecord())
        with pytest.raises(EmitRefused):
            art.validate()


class TestZeroHandWrittenCouplingCode:
    def test_a_new_case_is_declarations_and_the_compiler_call(self):
        """The whole claim of the package, as an executable assertion.

        A three-agent chain nobody wrote coupling code for, compiled from
        declarations alone.
        """
        from atlas.capability import (
            Differentiable, TimeDiscretization, linear_response)
        from atlas.graph import Agent, CaseGraph, Connection

        m = 4

        def expert(name: str, faces: list[str], gain: float) -> ExpertCapabilities:
            rng = np.random.default_rng(abs(hash(name)) % 1000)
            A = rng.standard_normal((m, m)) * 0.1
            A = gain * (A + A.T + 3 * np.eye(m))
            return ExpertCapabilities(
                expert_id=name,
                ports=[
                    port_decl(
                        name=f"{f}:MECH",
                        port_type=PortType.MECH,
                        nondim={"stress": 2.0, "velocity": 3.0, "power_area": 6.0},
                        effective_resolution=m,
                    )
                    for f in faces
                ],
                bc_channel=BCChannel.DIRICHLET,
                bc_time_varying=True,
                # **W61.** This assertion is about the rung reaching probed-DtN,
                # and R2b gates that on the agent posing a boundary-value problem
                # over a macro-step. Before W61 an undeclared
                # `time_discretization` defaulted to `unknown` and the rung lifted
                # anyway with a decertification beside it; now an undecided
                # premise holds the rung down. So the declaration that used to be
                # implicit has to be made -- which is the point of the test, since
                # the claim under assertion is "a new case is DECLARATIONS and a
                # compile call" and this is one more declaration.
                time_discretization=TimeDiscretization.IMPLICIT,
                differentiable=Differentiable.JVP,
                dt_native=1e-2,
                storage=lambda u: float(np.dot(np.ravel(u), np.ravel(u))),
                validity=lambda state, cond=None: True,
                governing_family="toy",
                boundary_response=linear_response(A),
                boundary_response_jvp=lambda _p, _t, d, A=A: A @ np.asarray(d, float),
            )

        graph = CaseGraph(
            name="three-agent-chain",
            agents=[
                Agent("x", expert("x", ["y"], 1.0)),
                Agent("y", expert("y", ["x", "z"], 0.8)),
                Agent("z", expert("z", ["y"], 1.2)),
            ],
            connections=[
                Connection("x-y", ("x", "y:MECH"), ("y", "x:MECH"), PortType.MECH,
                           derive_space=True, geometrically_coincident=True),
                Connection("y-z", ("y", "z:MECH"), ("z", "y:MECH"), PortType.MECH,
                           derive_space=True, geometrically_coincident=True),
            ],
            macro_dt=1e-2,
        )
        result = compile_scheme(graph)
        assert result.verdict is ADMIT_UNCERTIFIED
        assert result.scheme.transmission is Transmission.PROBED_DTN
        assert len(result.seam_operators) == 2
        assert result.envelope[Hypothesis.E7] is Status.HOLDS
        # It is still not `admit`: L is unmeasured and the assembly has no
        # accuracy condition, and no amount of declaration fixes either.
        assert result.verdict is not ADMIT
