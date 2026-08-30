"""L4's probe on the common interface space, and the envelope stamp."""

from __future__ import annotations

import numpy as np
import pytest

from atlas.capability import (
    BCChannel,
    Differentiable,
    ExpertCapabilities,
    linear_response,
    port_decl,
    zero_response,
)
from atlas.envelope import EnvelopeStamp, Hypothesis, Status
from atlas.ports import PortType
from atlas.probe import ProbeBudget, cut_score, probe_block, substitution_delta
from atlas.transfer import InterfaceSpace, Prolongation, identity_prolongation

M = 6
SCALES = {"stress": 2.0, "velocity": 3.0, "power_area": 6.0}


def _expert(
    name: str,
    response,
    bc=BCChannel.DIRICHLET,
    diff=Differentiable.NONE,
    jvp=None,
) -> ExpertCapabilities:
    return ExpertCapabilities(
        expert_id=name,
        ports=[
            port_decl(
                name="face:MECH",
                port_type=PortType.MECH,
                nondim=dict(SCALES),
                effective_resolution=M,
            )
        ],
        bc_channel=bc,
        differentiable=diff,
        dt_native=1e-2,
        governing_family="test-family",
        boundary_response=response,
        boundary_response_jvp=jvp,
    )


def _spd(seed: int, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, M)) * 0.1
    return scale * (A + A.T + 3.0 * np.eye(M))


class TestTheProbe:
    def test_it_recovers_the_operator_on_the_common_space(self):
        A = _spd(0)
        caps = _expert("e", linear_response(A))
        space = InterfaceSpace("s", dim=M)
        P = identity_prolongation("e", "face:MECH", M)
        block = probe_block(caps, caps.ports[0], space, P, "s")
        assert np.allclose(block.S, A, atol=1e-6)

    def test_it_subtracts_the_zero_probe_so_a_bias_cancels(self):
        """Any trace-independent bias cancels in the numerator.

        A systematic, state-dependent model error is not what breaks a probe;
        genuine non-reproducibility is.
        """
        A = _spd(1)
        bias = np.full(M, 42.0)
        caps = _expert("e", linear_response(A, bias))
        space = InterfaceSpace("s", dim=M)
        block = probe_block(caps, caps.ports[0], space, identity_prolongation("e", "p", M), "s")
        assert np.allclose(block.S, A, atol=1e-6)

    def test_the_jvp_route_is_exact(self):
        A = _spd(2)
        caps = _expert(
            "e", linear_response(A), diff=Differentiable.JVP,
            jvp=lambda _p, _t, d: A @ np.asarray(d, float),
        )
        assert caps.probe_route() == "jvp"
        assert caps.probe_class() == "probe-cheap"
        space = InterfaceSpace("s", dim=M)
        block = probe_block(caps, caps.ports[0], space, identity_prolongation("e", "p", M), "s")
        assert np.allclose(block.S, A, atol=1e-12)
        assert block.n_solves == M          # no zero probe needed

    def test_the_regression_route_runs_for_an_opaque_expert(self):
        A = _spd(3)
        caps = _expert("e", linear_response(A))
        caps.deterministic = False
        assert caps.probe_route() == "regression"
        assert caps.probe_class() == "probe-expensive"
        space = InterfaceSpace("s", dim=M)
        block = probe_block(
            caps, caps.ports[0], space, identity_prolongation("e", "p", M), "s",
            ProbeBudget(regression_oversample=3.0),
        )
        assert np.allclose(block.S, A, atol=1e-3)

    def test_the_probed_block_is_the_galerkin_compression(self):
        """P^* Lambda P, and positive-realness survives by congruence."""
        A = _spd(4)
        caps = _expert("e", linear_response(A))
        rng = np.random.default_rng(9)
        Pm = rng.standard_normal((M, 3))
        space = InterfaceSpace("s", dim=3)
        P = Prolongation("e", "p", Pm)
        block = probe_block(caps, caps.ports[0], space, P, "s")
        assert np.allclose(block.S, Pm.T @ A @ Pm, atol=1e-5)
        # A is SPD, so the compression is too: the congruence carries passivity
        # through a non-conforming transfer.
        assert block.passivity_defect == pytest.approx(0.0, abs=1e-9)


class TestTheEmptyInterfaceProblem:
    def test_no_boundary_channel_gives_exactly_zero(self):
        """The vault's sharpest finding, reproduced as an operator norm."""
        caps = _expert("frozen", zero_response, bc=BCChannel.NONE)
        space = InterfaceSpace("s", dim=M)
        block = probe_block(caps, caps.ports[0], space, identity_prolongation("f", "p", M), "s")
        assert block.is_empty
        assert block.Xi == 0.0
        assert any("empty rather than ill-conditioned" in n for n in block.notes)

    def test_composability_index_against_a_reference(self):
        A = _spd(5)
        ref = _expert("ref", linear_response(A))
        weak = _expert("weak", linear_response(0.25 * A))
        space = InterfaceSpace("s", dim=M)
        P = identity_prolongation("w", "p", M)
        block = probe_block(weak, weak.ports[0], space, P, "s", reference=ref)
        assert block.Xi == pytest.approx(0.25, rel=1e-3)


class TestPassivityAndConditioning:
    def test_a_negative_mode_is_found_without_a_rollout(self):
        A = np.diag([2.0, 1.0, -0.5, 1.0, 1.0, 1.0])
        caps = _expert("e", linear_response(A))
        space = InterfaceSpace("s", dim=M)
        block = probe_block(caps, caps.ports[0], space, identity_prolongation("e", "p", M), "s")
        assert block.passivity_defect == pytest.approx(0.5, abs=1e-5)
        # The eigenvector names WHICH interface mode is amplified.
        assert int(np.argmax(np.abs(block.passivity_eigvec))) == 2

    def test_arithmetic_noise_is_not_reported_as_a_passivity_defect(self):
        """A negative eigenvalue at the level of arithmetic noise is not a defect.

        Reporting one would put a passivity failure on every symmetric operator
        the probe ever assembles, which is how a diagnostic stops being read.
        """
        rng = np.random.default_rng(11)
        Q, _ = np.linalg.qr(rng.standard_normal((M, M)))
        A = Q @ np.diag([3.0, 2.0, 1.0, 1.0, 1.0, 0.0]) @ Q.T   # PSD up to roundoff
        caps = _expert("e", linear_response(A))
        space = InterfaceSpace("s", dim=M)
        block = probe_block(caps, caps.ports[0], space, identity_prolongation("e", "p", M), "s")
        assert block.passivity_defect == 0.0
        # The raw eigenvalue is kept, so the clipping is visible rather than silent.
        assert block.passivity_lambda_min is not None
        assert abs(block.passivity_lambda_min) < 1e-8

    def test_a_defect_above_the_noise_floor_still_reports(self):
        A = np.diag([2.0, 1.0, -1e-6, 1.0, 1.0, 1.0])
        caps = _expert("e", linear_response(A))
        space = InterfaceSpace("s", dim=M)
        block = probe_block(caps, caps.ports[0], space, identity_prolongation("e", "p", M), "s")
        assert block.passivity_defect > 0.0

    def test_the_optimal_robin_coefficient_is_read_off_the_diagonal(self):
        A = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        caps = _expert("e", linear_response(A))
        space = InterfaceSpace("s", dim=M)
        block = probe_block(caps, caps.ports[0], space, identity_prolongation("e", "p", M), "s")
        assert np.allclose(block.alpha_star, np.diag(A), atol=1e-5)

    def test_beta_ignores_the_constrained_null_direction(self):
        A = np.diag([3.0, 2.0, 1.0, 0.0, 4.0, 5.0])
        caps = _expert("e", linear_response(A))
        space = InterfaceSpace("s", dim=M)
        block = probe_block(caps, caps.ports[0], space, identity_prolongation("e", "p", M), "s")
        assert block.null_dim == 1
        assert block.beta == pytest.approx(1.0, abs=1e-5)

    def test_cut_score_is_lower_for_a_more_local_operator(self):
        local = np.diag([2.0, 2.0, 2.0])
        nonlocal_ = np.array([[2.0, 1.9, 1.8], [1.9, 2.0, 1.9], [1.8, 1.9, 2.0]])
        assert cut_score(local, 2.0) < cut_score(nonlocal_, float(np.linalg.svd(nonlocal_, compute_uv=False)[-1]))

    def test_substitution_delta_needs_two_probes_and_no_rollout(self):
        assert substitution_delta(np.eye(3), 1.5 * np.eye(3)) == pytest.approx(0.5)


class TestEnvelopeStamp:
    def test_everything_starts_unchecked(self):
        s = EnvelopeStamp()
        assert all(v is Status.UNCHECKED for v in s.values.values())
        assert not s.bound_applies

    def test_a_failure_is_never_overwritten_by_a_later_pass(self):
        s = EnvelopeStamp()
        s.fails(Hypothesis.E3, "seam 1 joins two governing families")
        s.holds(Hypothesis.E3, "seam 2 is fine")
        assert s[Hypothesis.E3] is Status.FAILS

    def test_E1_to_E4_may_never_be_left_unchecked(self):
        s = EnvelopeStamp()
        assert {h.value for h in s.illegally_unchecked()} == {"E1", "E2", "E3", "E4"}
        for h in (Hypothesis.E1, Hypothesis.E2, Hypothesis.E3, Hypothesis.E4):
            s.holds(h, "checked at compile time")
        assert s.illegally_unchecked() == []

    def test_the_bound_applies_only_when_all_seven_hold(self):
        s = EnvelopeStamp()
        for h in Hypothesis:
            s.holds(h, "checked")
        assert s.bound_applies
        s.unchecked(Hypothesis.E5, "L unmeasured")
        assert not s.bound_applies
