"""Closure, substitution, conformance, claim typing, and the named holes."""

from __future__ import annotations

import numpy as np
import pytest

from atlas.capability import (
    BCChannel,
    ClaimType,
    Differentiable,
    ExpertCapabilities,
    linear_response,
    port_decl,
    zero_response,
)
from atlas.claims import (
    ClaimTypeTag,
    HorizonBranch,
    QuantityClaim,
    abstention_horizon,
    horizon_for,
    refuse_statistical_claim,
    type_claim,
)
from atlas.composition import (
    CLOSURE_TABLE,
    CompositionRefused,
    beta_after_swap,
    certify_substitution,
    compose,
    composite_tau_bound,
    schur_complement,
)
from atlas.conformance import run_conformance
from atlas.holes import (
    ASSEMBLY_CERTIFICATE,
    NAMED_HOLES,
    INTERFACE_MOTION,
    HoleLedger,
    NamedHoleError,
)
from atlas.ports import PortType
from atlas.transfer import InterfaceSpace, identity_prolongation
from atlas.verdict import ADMIT, ADMIT_UNCERTIFIED, REFUSE

M = 5
SCALES = {"stress": 2.0, "velocity": 3.0, "power_area": 6.0}


def _expert(name, response, **kw) -> ExpertCapabilities:
    base = dict(
        bc_channel=BCChannel.DIRICHLET,
        dt_native=1e-2,
        governing_family="fam",
        boundary_response=response,
    )
    base.update(kw)
    return ExpertCapabilities(
        expert_id=name,
        ports=[
            port_decl(name="p:MECH", port_type=PortType.MECH, nondim=dict(SCALES),
                      effective_resolution=M)
        ],
        **base,
    )


def _spd(seed, scale=1.0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, M)) * 0.1
    return scale * (A + A.T + 3.0 * np.eye(M))


# ---------------------------------------------------------------------------
# Theorem 1 -- closure
# ---------------------------------------------------------------------------


class TestClosure:
    def test_the_composite_operator_is_a_schur_complement(self):
        S = _spd(0)
        comp, beta_int = schur_complement(S, [0, 1], [2, 3, 4])
        expected = S[:2, :2] - S[:2, 2:] @ np.linalg.solve(S[2:, 2:], S[2:, :2])
        assert np.allclose(comp, expected)
        assert beta_int > 0

    def test_grouping_does_not_change_the_answer(self):
        """Schur complementation is transitive, so a hierarchy and the flat graph
        produce the same composed operator: grouping can be chosen for cost."""
        S = _spd(1)
        flat, _ = schur_complement(S, [0, 1], [2, 3, 4])

        # Eliminate {4} first, then {2, 3} from the result.
        step1, _ = schur_complement(S, [0, 1, 2, 3], [4])
        nested, _ = schur_complement(step1, [0, 1], [2, 3])
        assert np.allclose(flat, nested, atol=1e-10)

    def test_a_singular_internal_problem_is_refused(self):
        S = np.eye(4)
        S[2:, 2:] = 0.0
        with pytest.raises(CompositionRefused) as exc:
            schur_complement(S, [0, 1], [2, 3])
        assert "well posed" in str(exc.value)

    def test_eight_fields_close_three_do_not_two_propagate(self):
        closes = [k for k, (v, _) in CLOSURE_TABLE.items() if v == "closes"]
        fails = [k for k, (v, _) in CLOSURE_TABLE.items() if v.startswith("does not")]
        holes = [k for k, (v, _) in CLOSURE_TABLE.items() if v == "propagates a hole"]
        assert len(closes) == 8
        assert set(fails) == {"L_native", "regime_law", "validity"}
        assert set(holes) == {"governing_family", "lambda_ref"}

    def test_the_composite_must_redeclare_its_operating_point(self):
        members = [_expert("a", linear_response(_spd(2))), _expert("b", linear_response(_spd(3)))]
        c = compose("C", members, _spd(4), [0, 1], [2, 3, 4], members[0].ports)
        assert set(c.must_redeclare) == {"L_native", "regime_law", "validity"}
        assert c.capabilities.L_native is None
        assert c.capabilities.regime_law is None
        assert c.capabilities.validity is None       # deferred, raised from inside the step

    def test_the_composite_can_always_be_probed(self):
        members = [_expert("a", linear_response(_spd(5))), _expert("b", linear_response(_spd(6)))]
        c = compose("C", members, _spd(7), [0, 1], [2, 3, 4], members[0].ports)
        assert c.capabilities.bc_channel is BCChannel.DIRICHLET

    def test_a_disagreeing_governing_family_propagates_the_seam_reference_hole(self):
        a = _expert("a", linear_response(_spd(8)), governing_family="fluid")
        b = _expert("b", linear_response(_spd(9)), governing_family="structure")
        c = compose("C", [a, b], _spd(10), [0, 1], [2, 3, 4], a.ports)
        assert c.capabilities.governing_family == "composite"
        assert "SeamReference" in c.propagated_holes

    def test_a_symmetry_survives_only_if_it_is_a_graph_automorphism(self):
        a = _expert("a", linear_response(_spd(11)), equivariances=("mirror", "rot90"))
        b = _expert("b", linear_response(_spd(12)), equivariances=("mirror",))
        c = compose("C", [a, b], _spd(13), [0, 1], [2, 3, 4], a.ports,
                    graph_automorphisms=("mirror",))
        assert c.capabilities.equivariances == ("mirror",)

        c2 = compose("C", [a, b], _spd(13), [0, 1], [2, 3, 4], a.ports,
                     graph_automorphisms=("rot90",))
        assert c2.capabilities.equivariances == ()

    def test_the_attribution_theorem_relabels_rather_than_removing(self):
        """A subassembly's transmission and solve infidelity BECOME its agent
        infidelity one level up. Not removed, not double counted -- relabelled."""
        assert composite_tau_bound(0.1, 0.2, 0.05) == pytest.approx(0.35)
        assert composite_tau_bound(0.1, None, 0.05) is None


# ---------------------------------------------------------------------------
# Theorem 2 -- substitution
# ---------------------------------------------------------------------------


class TestSubstitution:
    def test_the_certificate_is_two_probes_and_no_rollout(self):
        old = _expert("old", linear_response(_spd(20)))
        new = _expert("new", linear_response(_spd(20, 1.01)))
        cert = certify_substitution(
            "agent", old, new, _spd(20), _spd(20, 1.01), beta=2.0, beta_min=0.5
        )
        assert cert.passes
        assert cert.verdict is ADMIT

    def test_a_different_port_list_is_a_graph_edit_not_a_substitution(self):
        old = _expert("old", linear_response(_spd(21)))
        new = ExpertCapabilities(
            expert_id="new",
            ports=[port_decl(name="q:THERM", port_type=PortType.THERM,
                             nondim={"temperature": 2.0, "entropy_flux": 3.0,
                                     "power_area": 6.0}, effective_resolution=M)],
            bc_channel=BCChannel.DIRICHLET, dt_native=1e-2, governing_family="fam",
            boundary_response=linear_response(_spd(21)),
        )
        cert = certify_substitution("agent", old, new, _spd(21), _spd(21), 2.0, 0.5)
        assert not cert.passes
        assert cert.verdict is REFUSE
        assert "graph edit" in cert.message

    def test_a_better_expert_can_still_fail_the_certificate(self):
        """There is no monotonicity theorem: a swap moves beta as well as tau,
        and beta sits in the denominator of both sigma and L."""
        old = _expert("old", linear_response(_spd(22)))
        better = _expert("better", linear_response(_spd(23, 5.0)))
        cert = certify_substitution(
            "agent", old, better, _spd(22), _spd(23, 5.0), beta=2.0, beta_min=0.5
        )
        assert not cert.passes
        assert "no monotonicity theorem" in cert.message
        assert "benchmarks better" in cert.message

    def test_passivity_is_the_only_property_that_survives_a_swap(self):
        old = _expert("old", linear_response(_spd(24)))
        new = _expert("new", linear_response(_spd(25, 5.0)))
        cert = certify_substitution(
            "agent", old, new, _spd(24), _spd(25, 5.0), 2.0, 0.5,
            passivity_old=0.0, passivity_new=0.0,
        )
        assert not cert.passes
        assert cert.verdict is ADMIT_UNCERTIFIED   # not a refusal: L <= 1 survives
        assert "only property in the framework with this closure" in cert.message

    def test_weyl_bounds_how_far_beta_can_move(self):
        assert beta_after_swap(2.0, 0.3) == pytest.approx(1.7)


# ---------------------------------------------------------------------------
# Theorem 3 -- conformance
# ---------------------------------------------------------------------------


class TestConformance:
    def test_a_zero_response_fails_a_declared_boundary_channel(self):
        """A finding the vault made the slow way, caught at admission."""
        caps = _expert("liar", zero_response, bc_channel=BCChannel.DIRICHLET)
        cert = run_conformance(
            caps, InterfaceSpace("s", dim=M), identity_prolongation("liar", "p:MECH", M)
        )
        assert cert.verdict is REFUSE
        bad = [t for t in cert.contradictions if t.field_name == "bc_channel"]
        assert bad and bad[0].silent_if_false

    def test_a_declared_none_channel_is_confirmed_not_flagged(self):
        caps = _expert("frozen", zero_response, bc_channel=BCChannel.NONE)
        cert = run_conformance(
            caps, InterfaceSpace("s", dim=M), identity_prolongation("frozen", "p:MECH", M)
        )
        bc = [t for t in cert.tests if t.field_name == "bc_channel"][0]
        assert bc.verdict is ADMIT
        assert "disqualifying as a plug" in bc.message

    def test_a_false_passivity_declaration_is_caught_by_one_eigenvalue(self):
        """**Amended by W63: the eigenvalue must be the SEAM's, not the block's.**

        The intent is unchanged and it is the right intent -- a record that claims
        a storage function it does not have must be refused at admission. What
        changed is where the eigenvalue is taken. W48 closed with *"the
        conformance suite certifies on the seam and reports blocks as
        diagnostics"* and the suite refused on the block, which the first real run
        of it exposed: Poseidon-T's `xhi` block measures a passivity defect of
        3.218e-03 while its assembled seam measures 0.
        """
        A = np.diag([2.0, 1.0, -0.5, 1.0, 1.0])
        caps = _expert("claims-passive", linear_response(A), storage=lambda u: 1.0)
        cert = run_conformance(
            caps, InterfaceSpace("s", dim=M), identity_prolongation("c", "p:MECH", M),
            seam_defect=0.5,
        )
        assert cert.verdict is REFUSE
        bad = [t for t in cert.contradictions if t.field_name == "storage"]
        assert bad and "ASSEMBLED SEAM" in bad[0].message

    def test_w63_a_block_defect_alone_decertifies_rather_than_refusing(self):
        """The failure the suite committed on its own first run.

        Same record, same probe, no seam measurement supplied. The block's defect
        is reported -- it is a real diagnostic and §2.3's eigenvector still names
        which interface mode is amplified -- and it does **not** decide the
        verdict, because a block can be non-passive while the seam it assembles
        into is passive. Measured twice before this rule existed: kappa 19 against
        2260 (§2.3) and 1.196 against 7061 (§8.5).
        """
        A = np.diag([2.0, 1.0, -0.5, 1.0, 1.0])
        caps = _expert("claims-passive", linear_response(A), storage=lambda u: 1.0)
        cert = run_conformance(
            caps, InterfaceSpace("s", dim=M), identity_prolongation("c", "p:MECH", M)
        )
        st = [t for t in cert.tests if t.field_name == "storage"][0]
        assert st.verdict is ADMIT_UNCERTIFIED
        assert "verdict belongs to the" in st.message
        assert st.residual is not None and st.residual > 0.0   # the block IS reported

    def test_w63_a_passive_seam_certifies_a_record_whose_block_is_not(self):
        """The case that matters: Poseidon-T, in miniature.

        A block with a negative mode, a seam without one. Before W63 this record
        was refused; the theory page said all along that it should not be.
        """
        A = np.diag([2.0, 1.0, -0.5, 1.0, 1.0])
        caps = _expert("claims-passive", linear_response(A), storage=lambda u: 1.0)
        cert = run_conformance(
            caps, InterfaceSpace("s", dim=M), identity_prolongation("c", "p:MECH", M),
            seam_defect=0.0,
        )
        st = [t for t in cert.tests if t.field_name == "storage"][0]
        assert st.verdict is ADMIT
        assert "is not the verdict (W48)" in st.message

    def test_a_false_equivariance_is_caught_by_one_solve_per_generator(self):
        caps = _expert("e", linear_response(_spd(30)), equivariances=("mirror",))
        cert = run_conformance(
            caps, InterfaceSpace("s", dim=M), identity_prolongation("e", "p:MECH", M),
            equivariance_check=lambda g: 1e-2,
        )
        assert any(t.field_name == "equivariances" for t in cert.contradictions)

    def test_a_claimed_jvp_that_disagrees_is_loud(self):
        A = _spd(31)
        caps = _expert(
            "e", linear_response(A), differentiable=Differentiable.JVP,
            boundary_response_jvp=lambda _p, _t, d: 2.0 * (A @ np.asarray(d, float)),
        )
        cert = run_conformance(
            caps, InterfaceSpace("s", dim=M), identity_prolongation("e", "p:MECH", M)
        )
        t = [x for x in cert.tests if x.field_name == "differentiable"][0]
        assert t.verdict is REFUSE
        assert not t.silent_if_false

    def test_validity_is_never_certified_only_not_falsified(self):
        caps = _expert("e", linear_response(_spd(32)), validity=lambda s, c=None: True)
        cert = run_conformance(
            caps, InterfaceSpace("s", dim=M), identity_prolongation("e", "p:MECH", M),
            validity_suite=("state-a", "state-b"),
        )
        t = [x for x in cert.tests if x.field_name == "validity"][0]
        assert t.verdict is ADMIT_UNCERTIFIED
        assert "not falsified on suite" in t.message
        assert "single unverifiable declaration" in t.message
        assert cert.suite == ("state-a", "state-b")

    def test_validity_can_be_falsified(self):
        caps = _expert("e", linear_response(_spd(33)), validity=lambda s, c=None: True)
        cert = run_conformance(
            caps, InterfaceSpace("s", dim=M), identity_prolongation("e", "p:MECH", M),
            validity_falsified_on=("out-of-regime-state",),
        )
        assert cert.verdict is REFUSE

    def test_the_certificate_binds_to_a_weight_hash_and_a_probe_state(self):
        caps = _expert("e", linear_response(_spd(34)))
        caps.weight_hash = "sha256:deadbeef"
        cert = run_conformance(
            caps, InterfaceSpace("s", dim=M), identity_prolongation("e", "p:MECH", M),
            probe_state="Re=5000",
        )
        assert cert.weight_hash == "sha256:deadbeef"
        assert cert.probe_state == "Re=5000"
        assert cert.valid_at("Re=5000")
        assert not cert.valid_at("Re=50000")

    def test_without_a_probe_the_silent_fields_are_uncertified_not_passed(self):
        caps = _expert("e", linear_response(_spd(35)))
        cert = run_conformance(caps)
        bc = [t for t in cert.tests if t.field_name == "bc_channel"][0]
        assert bc.verdict is ADMIT_UNCERTIFIED


# ---------------------------------------------------------------------------
# L9 -- claim typing and the two horizons
# ---------------------------------------------------------------------------


class TestClaimTyping:
    def test_an_unmeasured_L_gives_UNTYPED(self):
        from atlas.holes import UNMEASURED_CONSTANTS

        claim = type_claim(
            [QuantityClaim("wake-deficit", 0.05)],
            UNMEASURED_CONSTANTS["L"], per_step_defect=1e-3, dt=1e-2,
        )
        assert claim.tag is ClaimTypeTag.UNTYPED
        ok, why = claim.may_quote("wake-deficit", at_time=1.0)
        assert not ok and "UNTYPED" in why

    def test_the_three_horizon_branches(self):
        q = QuantityClaim("q", 0.1)
        assert horizon_for(q, 1.5, 1e-3, dt=1e-2).branch is HorizonBranch.EXPONENTIAL
        assert horizon_for(q, 1.0, 1e-3, dt=1e-2).branch is HorizonBranch.LINEAR
        assert horizon_for(q, 0.5, 1e-3, dt=1e-2).branch is HorizonBranch.UNBOUNDED

    def test_passivity_converts_a_short_exponential_horizon_into_a_long_linear_one(self):
        q = QuantityClaim("q", 0.1)
        chaotic = horizon_for(q, 1.5, 1e-3, dt=1e-2)
        passive = horizon_for(q, 1.0, 1e-3, dt=1e-2)
        assert passive.n_pred > chaotic.n_pred

    def test_the_horizon_is_a_table_not_a_scalar(self):
        """Tolerance sits inside the logarithm and is a property of the quantity,
        so a run has one L and as many horizons as it has quoted quantities."""
        claim = type_claim(
            [QuantityClaim("tight", 1e-3), QuantityClaim("loose", 1e-1)],
            L=1.2, per_step_defect=1e-5, dt=1e-2,
        )
        assert len(claim.horizons) == 2
        assert claim.horizon("loose").t_pred > claim.horizon("tight").t_pred

    def test_a_metric_past_its_own_horizon_is_refused(self):
        claim = type_claim([QuantityClaim("q", 0.1)], L=1.5, per_step_defect=1e-3, dt=1e-2)
        h = claim.horizon("q")
        assert claim.may_quote("q", h.t_pred * 0.5)[0]
        ok, why = claim.may_quote("q", h.t_pred * 2.0)
        assert not ok
        assert "refused for publication, not for computation" in why

    def test_the_abstention_horizon_degrades_linearly_in_library_size(self):
        small = abstention_horizon(K=4, p=1e-3, dt=1e-2)
        large = abstention_horizon(K=20, p=1e-3, dt=1e-2)
        assert small == pytest.approx(5 * large)

    def test_the_binding_horizon_can_be_the_framework_not_the_physics(self):
        claim = type_claim(
            [QuantityClaim("q", 0.1)], L=1.001, per_step_defect=1e-6, dt=1e-2,
            K=20, p_decline=1e-2,
        )
        assert claim.t_abs < claim.horizon("q").t_pred
        assert claim.t_usable == claim.t_abs
        assert any("abstention, not chaos" in t for t in claim.tags)

    def test_a_statistical_claim_is_declared_and_refused(self):
        exc = refuse_statistical_claim("energy spectrum")
        assert "refused" in str(exc)
        assert "typing discipline, not the statistical theory" in str(exc)


# ---------------------------------------------------------------------------
# The five named holes
# ---------------------------------------------------------------------------


class TestNamedHoles:
    def test_there_are_five_and_none_of_them_solves(self):
        assert len(NAMED_HOLES) == 5
        for hole in NAMED_HOLES.values():
            with pytest.raises(NamedHoleError):
                hole.solve()

    def test_every_hole_declares_an_interface_and_requires_a_measurement(self):
        for hole in NAMED_HOLES.values():
            assert hole.declared_interface, f"{hole.name} has no declared interface"
            assert hole.measurements_required, f"{hole.name} requires no measurement"
            assert hole.must_satisfy, f"{hole.name} grades a solution against nothing"

    def test_the_assembly_certificates_condition_field_is_filled(self):
        """Superseded 2026-08-28. It used to assert the field STAYS EMPTY.

        `condition` is L6/C1 -- convexity of the partition -- and R11 enforces it.
        A bare `AssemblyCertificate()` still carries no condition, because nothing
        has been checked; `certify(pou)` fills it from the declaration alone.
        See `tests/test_tier9_assembly_condition.py` for the rule itself.
        """
        from atlas.assembly import AssemblyCertificate, PartitionOfUnity, certify

        assert AssemblyCertificate().condition is None
        assert "FILLED" in ASSEMBLY_CERTIFICATE.declared_interface["condition"]
        assert "L6/C1" in ASSEMBLY_CERTIFICATE.declared_interface["condition"]

        n = 6
        pou = PartitionOfUnity(n, {"a": np.eye(n)}, {"a": np.ones(n)})
        assert certify(pou).condition.holds is True

    def test_the_ledger_reports_what_a_touched_slot_still_owes(self):
        ledger = HoleLedger()
        ledger.activate(INTERFACE_MOTION, "seam-1", motion_class="static")
        assert ledger.outstanding()["InterfaceMotion"] == list(
            INTERFACE_MOTION.measurements_required
        )
        for name in INTERFACE_MOTION.measurements_required:
            ledger.measure(INTERFACE_MOTION, name, 0.0)
        assert ledger.outstanding() == {}


class TestAssemblyIdentity:
    def test_a_partition_of_unity_that_is_not_one_is_refused(self):
        from atlas.assembly import PartitionOfUnity, certify

        R1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        R2 = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        good = PartitionOfUnity(3, {"a": R1, "b": R2},
                                {"a": np.array([1.0, 0.5]), "b": np.array([0.5, 1.0])})
        assert certify(good).identity_holds

        bad = PartitionOfUnity(3, {"a": R1, "b": R2},
                               {"a": np.array([1.0, 0.9]), "b": np.array([0.5, 1.0])})
        assert certify(bad).identity_holds is False

    def test_norm_A_is_unmeasured_until_an_assembly_is_declared(self):
        from atlas.assembly import AssemblyCertificate
        from atlas.holes import Unmeasured

        assert isinstance(AssemblyCertificate().norm_A, Unmeasured)
