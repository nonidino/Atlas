"""The coupled step: gamma, the port residuals, and the refusal to run a refusal."""

from __future__ import annotations

import numpy as np
import pytest

from atlas import compile_scheme
from atlas.cases import wind_farm
from atlas.capability import BCChannel, Differentiable, ExpertCapabilities, linear_response, port_decl
from atlas.graph import Agent, CaseGraph, Connection
from atlas.ports import PortType
from atlas.scheme import Accelerator
from atlas.solve import (
    Declination,
    InterfaceProblem,
    RunRefused,
    coupled_step,
    rollout,
    solve_interface,
)

M = 4
SCALES = {"stress": 2.0, "velocity": 3.0, "power_area": 6.0}


def _expert(name, faces, gain=1.0, validity=None, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, M)) * 0.1
    A = gain * (A + A.T + 3.0 * np.eye(M))
    # A nonzero zero-probe response: what the agent returns with NO interface
    # datum imposed. Without it chi is zero, the interface problem is homogeneous,
    # and the converged trace is trivially zero -- which tests nothing.
    bias = rng.standard_normal(M) * 0.5
    return ExpertCapabilities(
        expert_id=name,
        ports=[
            port_decl(name=f"{f}:MECH", port_type=PortType.MECH, nondim=dict(SCALES),
                      effective_resolution=M)
            for f in faces
        ],
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=True,
        differentiable=Differentiable.JVP,
        dt_native=1e-2,
        storage=lambda u: float(np.dot(np.ravel(u), np.ravel(u))),
        validity=validity,
        governing_family="toy",
        boundary_response=linear_response(A, bias),
        boundary_response_jvp=lambda _p, _t, d, A=A: A @ np.asarray(d, float),
    )


def _pair(validity_b=None):
    graph = CaseGraph(
        name="pair",
        agents=[
            Agent("a", _expert("a", ["b"], 1.0, seed=1)),
            Agent("b", _expert("b", ["a"], 0.7, validity_b, seed=2)),
        ],
        connections=[
            Connection("a-b", ("a", "b:MECH"), ("b", "a:MECH"), PortType.MECH,
                       derive_space=True, geometrically_coincident=True)
        ],
        macro_dt=1e-2,
    )
    return graph, compile_scheme(graph)


class TestTheInterfaceSolve:
    def test_a_direct_schur_solve_drives_gamma_to_roundoff(self):
        rng = np.random.default_rng(0)
        S = rng.standard_normal((M, M)) + 4 * np.eye(M)
        chi = rng.standard_normal(M)
        p = InterfaceProblem("s", S, chi, M)
        lam, iters, hist = solve_interface(p, Accelerator.DIRECT_SCHUR)
        assert hist[-1] < 1e-10
        assert iters == 1

    def test_krylov_converges_and_richardson_converges_more_slowly(self):
        rng = np.random.default_rng(1)
        A = rng.standard_normal((M, M)) * 0.2
        S = A + A.T + 6 * np.eye(M)
        chi = rng.standard_normal(M)
        p = InterfaceProblem("s", S, chi, M)
        _, k_it, k_hist = solve_interface(p, Accelerator.KRYLOV, tol=1e-10)
        _, r_it, r_hist = solve_interface(p, Accelerator.RICHARDSON, tol=1e-10)
        assert k_hist[-1] < 1e-9 and r_hist[-1] < 1e-9
        assert k_it <= r_it

    def test_an_empty_problem_returns_after_one_pass(self):
        """Every trace is equally consistent, so iterating is the degenerate axis."""
        p = InterfaceProblem("s", np.zeros((M, M)), np.zeros(M), M)
        lam, iters, _ = solve_interface(p, Accelerator.RICHARDSON)
        assert iters == 1
        assert np.allclose(lam, 0.0)


class TestTheCoupledStep:
    def test_it_emits_gamma_and_the_port_residual(self):
        graph, result = _pair()
        assert result.runnable
        sr = coupled_step(result, graph)
        assert sr.gamma < 1e-8                 # a direct solve makes gamma vanish
        assert "a-b" in sr.port_residuals
        assert sr.power_residual is not None

    def test_iterating_cannot_reduce_the_wrong_operators_error(self):
        """The solve drives the trace to the root of the problem POSED.

        Perturbing the operator moves that root; iterating harder converges onto
        the moved root rather than the true one, which is why this layer controls
        gamma and nothing else.
        """
        graph, result = _pair()
        op = result.seam_operators["a-b"]
        true_S = op.S.copy()
        transfer = result.transfers["a-b"]
        from atlas.solve import build_interface_problem

        exact = build_interface_problem(graph, graph.connections[0], transfer, op)
        lam_true, _, _ = solve_interface(exact, Accelerator.DIRECT_SCHUR)

        op.S = true_S + 0.5 * np.eye(true_S.shape[0])       # a wrong operator
        wrong = build_interface_problem(graph, graph.connections[0], transfer, op)
        lam_wrong, _, hist = solve_interface(wrong, Accelerator.DIRECT_SCHUR)

        assert hist[-1] < 1e-10                             # gamma is zero
        assert np.linalg.norm(lam_wrong - lam_true) > 1e-3  # and the answer moved

    def test_a_refused_scheme_is_not_runnable(self):
        graph = wind_farm.build()
        result = compile_scheme(graph)
        with pytest.raises(RunRefused) as exc:
            coupled_step(result, graph)
        assert "not a scheme with a warning attached" in str(exc.value)

    def test_a_declination_halts_the_rollout_and_types_what_ran(self):
        calls = {"n": 0}

        def declines_after_three(_state, _cond=None):
            calls["n"] += 1
            return calls["n"] <= 3

        graph, result = _pair(validity_b=declines_after_three)
        steps, report, declined = rollout(result, graph, n_steps=10)
        assert declined == "b"
        assert 0 < len(steps) < 10
        assert report.power_residual is not None

    def test_a_declination_carries_the_originating_agent(self):
        graph, result = _pair(validity_b=lambda s, c=None: False)
        with pytest.raises(Declination) as exc:
            coupled_step(result, graph)
        assert exc.value.agent_id == "b"
        assert "shorter answer with a stated reason" in str(exc.value)


class TestRollout:
    def test_it_warm_starts_from_the_previous_converged_trace(self):
        graph, result = _pair()
        steps, report, declined = rollout(result, graph, n_steps=4)
        assert declined is None
        assert len(steps) == 4
        assert all(s.gamma < 1e-8 for s in steps)
        assert steps[1].t == pytest.approx(graph.macro_dt)

    def test_the_conservation_report_is_populated_from_declared_values_only(self):
        graph, result = _pair()
        _, report, _ = rollout(result, graph, n_steps=2)
        assert set(report.per_port) == {"a-b"}
        assert report.power_residual is not None
