"""L3's declared object: one prolongation, and everything else derived."""

from __future__ import annotations

import numpy as np
import pytest

from atlas.holes import NamedHoleError
from atlas.ports import (
    MappingClass,
    PortType,
    Role,
    check_scales,
    compose_scales,
    spec_for,
)
from atlas.transfer import (
    InterfaceSpace,
    Prolongation,
    SeamTransfer,
    dim_M,
    identity_prolongation,
    lumped_prolongation,
    mapping_class,
)


class TestClosedVocabulary:
    def test_the_five_types_resolve(self):
        for t in PortType:
            assert spec_for(t).port_type is t

    def test_a_sixth_type_is_a_port_amendment_not_an_edit(self):
        with pytest.raises(NamedHoleError) as exc:
            spec_for("RADIATION")
        assert "amendment procedure" in str(exc.value)
        assert "never been exercised" in str(exc.value)


class TestScaleSets:
    def test_completeness_is_decidable_from_the_port_list_alone(self):
        chk = check_scales(PortType.MECH, {"stress": 1.0})
        assert not chk.complete
        assert set(chk.missing) == {"velocity", "power_area"}

    def test_the_power_identity_is_checked(self):
        good = check_scales(PortType.MECH, {"stress": 2.0, "velocity": 3.0, "power_area": 6.0})
        assert good.ok and good.power_identity_residual == pytest.approx(0.0)

        # Independently chosen effort and flow scales: dimensionally plausible,
        # and it silently destroys the power bond.
        bad = check_scales(PortType.MECH, {"stress": 2.0, "velocity": 3.0, "power_area": 5.0})
        assert not bad.ok
        assert "power identity" in bad.detail

    def test_advec_needs_one_pair_per_passenger(self):
        chk = check_scales(
            PortType.ADVEC,
            {"enthalpy": 2.0, "mass_flux": 3.0, "power_area": 6.0},
            passengers=("h0", "Yk"),
        )
        assert not chk.complete
        assert "Yk_effort" in chk.missing

    def test_two_scale_sets_either_compose_or_say_why_not(self):
        a = {"stress": 2.0, "velocity": 3.0, "power_area": 6.0}
        b = {"stress": 4.0, "velocity": 3.0, "power_area": 12.0}
        ok, ratios, why = compose_scales(PortType.MECH, a, b)
        assert ok and ratios["stress"] == pytest.approx(0.5)

        ok, _, why = compose_scales(PortType.MECH, a, {"stress": 4.0})
        assert not ok and "missing" in why


class TestTheForcedReduction:
    def test_the_reduction_is_the_adjoint_not_a_declaration(self):
        space = InterfaceSpace("s", dim=3)
        rng = np.random.default_rng(0)
        P = Prolongation("A", "p", rng.standard_normal((7, 3)))
        assert np.allclose(P.adjoint(space), P.matrix.T)
        assert P.adjointness_residual(space) < 1e-12

    def test_adjointness_holds_under_non_trivial_grams(self):
        rng = np.random.default_rng(1)
        G_M = np.diag(rng.uniform(0.5, 2.0, 4))
        G_V = np.diag(rng.uniform(0.5, 2.0, 9))
        space = InterfaceSpace("s", dim=4, gram=G_M)
        P = Prolongation("A", "p", rng.standard_normal((9, 4)), gram_V=G_V)
        assert P.adjointness_residual(space) < 1e-10

    def test_a_hand_written_reduction_is_caught(self):
        """The actuator disk was not missing a check. It was missing a declaration."""
        space = InterfaceSpace("s", dim=2)
        P = Prolongation("disk", "face", np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
        assert P.adjointness_residual(space) < 1e-12

        P.declared_reduction = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert P.uses_hand_written_reduction()
        assert P.adjointness_residual(space) > 1e-6

    def test_mapping_class_is_derived_and_takes_no_declaration(self):
        assert mapping_class(Role.EFFORT) is MappingClass.CONSISTENT
        assert mapping_class(Role.FLOW) is MappingClass.CONSERVATIVE
        # There is no argument by which a caller could reverse it.
        with pytest.raises(TypeError):
            mapping_class(Role.EFFORT, MappingClass.CONSERVATIVE)  # type: ignore[call-arg]

    def test_conservative_map_preserves_integrals_when_consistent_map_is_an_average(self):
        """The two mapping classes are the two halves of one adjoint pair."""
        space = InterfaceSpace("s", dim=1)
        # A consistent (constant-reproducing) prolongation onto four cells.
        P = Prolongation("A", "p", np.ones((4, 1)))
        R = P.adjoint(space)
        flow = np.array([0.25, 0.5, 0.1, 0.15])
        # The forced adjoint preserves the integral of the flow.
        assert (R @ flow).item() == pytest.approx(flow.sum())
        # And the prolongation reproduces the constant on the effort side.
        assert np.allclose(P.prolong(np.array([2.0]), space), 2.0)


class TestFieldToLumped:
    def test_lumped_is_the_case_dim_M_equals_one(self):
        weights = np.array([0.2, 0.3, 0.5])
        P = lumped_prolongation("disk", "face", weights)
        space = InterfaceSpace("s", dim=1)
        assert space.is_lumped
        assert P.adjointness_residual(space) < 1e-12
        # The reduction is forced to be the weighted integral, and no other
        # reduction preserves the bond.
        e = np.array([1.0, 2.0, 3.0])
        assert P.reduce(e, space).item() == pytest.approx(float(weights @ e))


class TestInterfaceSpaceDimension:
    def test_dim_M_is_the_min_over_sides(self):
        assert dim_M({"A": 32, "B": 16}) == 16

    def test_an_unset_resolution_is_an_error_not_a_default(self):
        with pytest.raises(Exception):
            dim_M({"A": 32, "B": None})  # type: ignore[dict-item]


class TestSeamTransfer:
    def test_geometric_coincidence_is_the_identity_special_case(self):
        space = InterfaceSpace("s", dim=5)
        t = SeamTransfer(
            "s", PortType.MECH, space,
            {"A": identity_prolongation("A", "p", 5), "B": identity_prolongation("B", "p", 5)},
        )
        assert t.conforming

    def test_non_conforming_is_expressible_and_still_adjoint(self):
        space = InterfaceSpace("s", dim=4)
        rng = np.random.default_rng(3)
        t = SeamTransfer(
            "s", PortType.MECH, space,
            {
                "A": identity_prolongation("A", "p", 4),
                "B": Prolongation("B", "p", rng.standard_normal((11, 4))),
            },
        )
        assert not t.conforming
        d = t.diagnostics()
        assert d["adjointness_residual[B]"] < 1e-10

    def test_a_seam_has_exactly_two_sides(self):
        space = InterfaceSpace("s", dim=2)
        with pytest.raises(Exception):
            SeamTransfer("s", PortType.MECH, space, {"A": identity_prolongation("A", "p", 2)})
