"""Tier 8: the rules the first real measurement forced, and the fields behind them.

Every test here corresponds to a number in `tier0-measurements`. They are written
against the *rule*, not against the case study, so they run without the build
repo checked out.
"""

from __future__ import annotations

import numpy as np
import pytest

from atlas import (
    REFUSE,
    Agent,
    BCChannel,
    CaseGraph,
    Connection,
    Decomposition,
    Differentiable,
    Direction,
    EllipticSubsolve,
    ExpertCapabilities,
    MeasuredConstants,
    PortType,
    TimeDiscretization,
    compile_scheme,
)
from atlas.assembly import GridPartitionOfUnity, PartitionOfUnity
from atlas.capability import linear_response, port_decl

M = 8


def _caps(agent_id, **kw):
    n = M
    rng = np.random.default_rng(3)
    A = rng.standard_normal((n, n)) * 0.05
    A = A + A.T + 2.0 * np.eye(n)
    defaults = dict(
        bc_channel=BCChannel.DIRICHLET,
        differentiable=Differentiable.NONE,
        dt_native=1e-2,
        governing_family="incompressible-navier-stokes-2d",
        boundary_response=linear_response(A, np.full(n, 0.05)),
        validity=lambda state, cond=None: True,
        storage=lambda u: 0.5 * float(np.dot(np.ravel(u), np.ravel(u))),
    )
    defaults.update(kw)
    return ExpertCapabilities(
        expert_id=agent_id,
        ports=[
            port_decl(name=f"{f}:MECH", port_type=PortType.MECH, geometry=f,
                      direction=Direction.BIDIRECTIONAL,
                      nondim={"stress": 1.0, "velocity": 1.0, "power_area": 1.0},
                      effective_resolution=M)
            for f in ("xlo", "xhi")
        ],
        **defaults,
    )


def _graph(decomposition=Decomposition.OVERLAPPING, overlap_cells=None,
           measured=None, **caps_kw):
    agents = [Agent(a, _caps(a, **caps_kw), domain=a) for a in ("A", "B")]
    conn = Connection(seam_id="s", a=("A", "xhi:MECH"), b=("B", "xlo:MECH"),
                      port_type=PortType.MECH, derive_space=True,
                      geometrically_coincident=True, expected_null_dim=0)
    return CaseGraph(name="t", agents=agents, connections=[conn],
                     decomposition=decomposition, overlap_cells=overlap_cells,
                     macro_dt=1e-2, measured=measured)


def _rules(result):
    return set(result.decisions.cited_rules())


# --- R10 -------------------------------------------------------------------


def test_r10_refuses_an_embedded_elliptic_subsolve():
    """The finding that was 99.8% of the first measured defect, as a gate."""
    r = compile_scheme(_graph(elliptic_subsolve=EllipticSubsolve.EMBEDDED))
    assert r.verdict is REFUSE
    hit = [d for d in r.decisions.refusals if d.rule == "R10"]
    assert hit, "an embedded elliptic sub-solve must be refused, not decertified"
    assert "elliptic_subsolve=embedded" in hit[0].message


def test_r10_admits_an_exposed_one():
    r = compile_scheme(_graph(elliptic_subsolve=EllipticSubsolve.EXPOSED))
    assert not [d for d in r.decisions.refusals if d.rule == "R10"]


def test_r10_decertifies_an_undeclared_incompressible_agent():
    """`none` on an incompressible solver is almost always a missed declaration."""
    r = compile_scheme(_graph(elliptic_subsolve=EllipticSubsolve.NONE))
    assert [d for d in r.decisions.decertifications if d.rule == "R10"]


def test_r10_is_silent_when_nothing_is_decomposed():
    """Nothing is cut, so there is nothing to cut wrongly.

    Called directly, so the guard itself is what is under test rather than the
    verdict a whole compile produces.

    **The note this docstring used to carry was a symptom (W113, 2026-09-01).**
    It said the public path *"cannot express a single agent"*, because a graph
    with no connections stamped E2 `unchecked` and `emit` refused it -- which
    `emit.validate`'s own message calls a defect in the compiler rather than a
    property of the case. E2 now holds vacuously over an empty interface set, so
    the public path does express it; `tests/test_tier23_thermal_strain.py` pins
    that. This test still calls the guard directly, deliberately: a compile would
    reach it through five other layers and could pass for the wrong reason.
    """
    from atlas.compiler import _Context, _r10_elliptic              # noqa: PLC2701
    from atlas.envelope import EnvelopeStamp
    from atlas.holes import HoleLedger
    from atlas.verdict import DecisionRecord

    caps = _caps("A", elliptic_subsolve=EllipticSubsolve.EMBEDDED)
    g = CaseGraph(name="one", agents=[Agent("A", caps)], connections=[],
                  decomposition=Decomposition.OVERLAPPING, macro_dt=1e-2)
    ctx = _Context(graph=g, budget=None, probe_budget=None, references={},
                   record=DecisionRecord(), stamp=EnvelopeStamp(),
                   holes=HoleLedger(), probe_state="none", depth=0)
    _r10_elliptic(ctx)
    assert not ctx.record.refusals


# --- R2b -------------------------------------------------------------------


def test_r2b_holds_the_rung_for_an_explicit_agent():
    """Flux balance is a boundary-value-problem condition; an explicit step poses none."""
    from atlas.scheme import Transmission

    r = compile_scheme(_graph(elliptic_subsolve=EllipticSubsolve.EXPOSED,
                              time_discretization=TimeDiscretization.EXPLICIT))
    assert r.scheme.transmission is Transmission.DIRICHLET
    assert r.scheme.decomposition is Decomposition.OVERLAPPING
    assert "L4/R2b/W46" in _rules(r)


def test_r2b_is_an_admit_not_a_refusal():
    """The compiler corrected the rung, so nothing is left silently wrong."""
    r = compile_scheme(_graph(elliptic_subsolve=EllipticSubsolve.EXPOSED,
                              time_discretization=TimeDiscretization.EXPLICIT))
    assert not [d for d in r.decisions.refusals if d.rule == "R2b/W46"]


def test_r2b_lets_an_implicit_agent_have_probed_dtn():
    from atlas.scheme import Transmission

    r = compile_scheme(_graph(elliptic_subsolve=EllipticSubsolve.EXPOSED,
                              time_discretization=TimeDiscretization.IMPLICIT))
    assert r.scheme.transmission is Transmission.PROBED_DTN


def test_r2b_decertifies_an_undeclared_discretization():
    r = compile_scheme(_graph(elliptic_subsolve=EllipticSubsolve.EXPOSED,
                              time_discretization=TimeDiscretization.UNKNOWN))
    assert [d for d in r.decisions.decertifications if d.rule == "R2b/W46"]


# --- the halo rule ---------------------------------------------------------


def test_required_halo_is_radius_times_substeps():
    """And only for an agent that has DECLARED its step explicit.

    ``time_discretization`` was left at its default here until W93, which is
    ``unknown`` -- so this test was asserting the product for an agent that had
    not said the product applies to it. That is the same omission mistake 0 in
    `CASE-STUDY-GUIDE` is about, inside a test of the rule the omission feeds.
    """
    caps = _caps("A", stencil_radius=2, substeps_per_macro_step=10,
                 time_discretization=TimeDiscretization.EXPLICIT)
    assert caps.required_halo() == 20


def test_required_halo_is_undecidable_without_a_substep_count():
    assert _caps("A", stencil_radius=2,
                 time_discretization=TimeDiscretization.EXPLICIT).required_halo() is None


def test_required_halo_is_undecidable_for_an_undeclared_discretization():
    """**W93.** ``radius * substeps`` is an EXPLICIT agent's domain of dependence.

    W69 established that for the ``implicit`` branch; the branch where the record
    says nothing was left returning the product, and ``unknown`` is what a frozen
    learned one-shot map declares. Measured on Poseidon-T: declared 2 cells,
    `probe.support_reach` measures 64.
    """
    assert _caps("A", stencil_radius=2, substeps_per_macro_step=10,
                 time_discretization=TimeDiscretization.UNKNOWN).required_halo() is None
    # a zero-stencil algebraic closure still couples nothing, under any label
    assert _caps("A", stencil_radius=0, substeps_per_macro_step=1,
                 time_discretization=TimeDiscretization.UNKNOWN).required_halo() == 0


def test_halo_rule_refuses_too_narrow_an_overlap():
    r = compile_scheme(_graph(overlap_cells=5,
                              elliptic_subsolve=EllipticSubsolve.EXPOSED,
                              time_discretization=TimeDiscretization.EXPLICIT,
                              stencil_radius=2, substeps_per_macro_step=10))
    hit = [d for d in r.decisions.refusals if d.rule == "R10/halo"]
    assert hit and "domain of dependence" in hit[0].message


def test_halo_rule_admits_a_sufficient_overlap():
    r = compile_scheme(_graph(overlap_cells=21,
                              elliptic_subsolve=EllipticSubsolve.EXPOSED,
                              time_discretization=TimeDiscretization.EXPLICIT,
                              stencil_radius=2, substeps_per_macro_step=10))
    assert not [d for d in r.decisions.refusals if d.rule == "R10/halo"]


def test_halo_rule_decertifies_rather_than_assuming_when_overlap_is_undeclared():
    r = compile_scheme(_graph(overlap_cells=None,
                              elliptic_subsolve=EllipticSubsolve.EXPOSED,
                              time_discretization=TimeDiscretization.EXPLICIT,
                              stencil_radius=2, substeps_per_macro_step=10))
    assert [d for d in r.decisions.decertifications if d.rule == "R10/halo"]


# --- W45: the ingest path --------------------------------------------------


def test_measured_constants_drop_off_the_unmeasured_list():
    base = compile_scheme(_graph(elliptic_subsolve=EllipticSubsolve.EXPOSED))
    assert any(u.startswith("L ") for u in base.unmeasured)

    m = MeasuredConstants(L=0.979644, L_stderr=4.1e-4, tau=1.34e-6, sigma=3.73e-8,
                          gamma=0.0, C_mu=1.0, probe_state="s", scheme="split-step",
                          depth=0, source="test")
    got = compile_scheme(_graph(elliptic_subsolve=EllipticSubsolve.EXPOSED, measured=m))
    assert not [u for u in got.unmeasured if u.startswith(("L ", "tau", "sigma", "C_mu"))]


def test_a_measured_L_stamps_E5_and_names_the_branch():
    from atlas.envelope import Hypothesis, Status

    m = MeasuredConstants(L=0.979644, tau=1.34e-6, sigma=3.73e-8, C_mu=1.0,
                          probe_state="developed wake", scheme="split-step")
    r = compile_scheme(_graph(elliptic_subsolve=EllipticSubsolve.EXPOSED, measured=m))
    assert r.envelope.values[Hypothesis.E5] is Status.HOLDS
    assert any("unbounded" in e for e in r.envelope.evidence[Hypothesis.E5])


def test_an_L_above_one_names_the_exponential_branch():
    from atlas.envelope import Hypothesis

    m = MeasuredConstants(L=1.014, tau=1e-4, sigma=1e-5, C_mu=1.0, scheme="x")
    r = compile_scheme(_graph(elliptic_subsolve=EllipticSubsolve.EXPOSED, measured=m))
    assert any("exponential" in e for e in r.envelope.evidence[Hypothesis.E5])


def test_unset_measured_fields_stay_unmeasured():
    """Unset means unmeasured, never a default."""
    m = MeasuredConstants(L=0.98)
    r = compile_scheme(_graph(elliptic_subsolve=EllipticSubsolve.EXPOSED, measured=m))
    assert any(u.startswith("C_mu") for u in r.unmeasured)
    assert not any(u.startswith("L ") for u in r.unmeasured)


# --- W28: the assembly constants ------------------------------------------


def test_norm_A_is_one_for_any_partition_of_unity():
    """A theorem, not a measurement: the max is attained at any single-owner cell."""
    n = 10
    R1, R2 = np.eye(n)[:7], np.eye(n)[4:]
    w1 = np.array([1, 1, 1, 1, 0.5, 0.5, 0.5])
    w2 = np.array([0.5, 0.5, 0.5, 1, 1, 1])
    pou = PartitionOfUnity(n, {"a": R1, "b": R2}, {"a": w1, "b": w2})
    assert pou.identity_residual() == pytest.approx(0.0, abs=1e-12)
    assert pou.norm_A() == pytest.approx(1.0, abs=1e-12)


def test_identity_norm_is_the_vacuous_one_it_replaced():
    """It returns ||I|| whenever the identity holds, which is the whole point."""
    n = 6
    pou = PartitionOfUnity(n, {"a": np.eye(n)}, {"a": np.ones(n)})
    assert pou.identity_norm() == pytest.approx(1.0)


def test_grid_partition_matches_the_dense_one():
    """The structured form exists because 255^2 needs a 34 GB dense matrix."""
    n = 10
    R1, R2 = np.eye(n)[:7], np.eye(n)[4:]
    w1 = np.array([1, 1, 1, 1, 0.5, 0.5, 0.5])
    w2 = np.array([0.5, 0.5, 0.5, 1, 1, 1])
    dense = PartitionOfUnity(n, {"a": R1, "b": R2}, {"a": w1, "b": w2})
    grid = GridPartitionOfUnity(n, {"a": np.arange(7), "b": np.arange(4, 10)},
                                {"a": w1, "b": w2})
    loc = {"a": np.arange(7.0), "b": np.arange(4.0, 10.0)}
    assert grid.identity_residual() == pytest.approx(dense.identity_residual(), abs=1e-12)
    assert grid.norm_A() == pytest.approx(dense.norm_A(), abs=1e-12)
    assert np.allclose(grid.assemble(loc), dense.assemble(loc))


def test_a_broken_partition_of_unity_is_caught():
    n = 6
    pou = PartitionOfUnity(n, {"a": np.eye(n)}, {"a": np.full(n, 0.9)})
    assert pou.identity_residual() == pytest.approx(0.1, abs=1e-12)
