"""Tier 26 -- CS-12, two-way FSI, a new governing family, and W114.

`atlas/cases/wing_fsi.py` and `scripts/w136_wing_fsi.py`.  Six groups:

  * **W114** -- the rule that had to be settled before the graph could be built.
    `L2/R10` now consults its own premise, an FSI graph compiles with zero
    refusals, and the CONTROL -- the same graph with the structure declaring the
    fluid's family -- is refused exactly as before, so the narrowing is a check
    and not an escape;
  * **the declaration** -- the plate lives inside one window in BOTH axes, the
    two agents declare genuinely different governing families, both declare
    `lambda_ref` so `tau` survives E3's failure, and the structural agent is
    `EMBEDDED` with no `split-step` variant;
  * **the structural operator** -- symmetric by Betti, built from the expert's
    own solves and the expert's own stiffness, exactly linear in `E`, and equal
    to `solve_mechanical`'s own answer.  Each is a control and NOT a floor
    (W106);
  * **the seam** -- the structural response reaches every station (the physics,
    not the architecture), and the fluid block is not identically zero (W130);
  * **the gate** -- the split reproduces the referent's tip-deflection history,
    the lag defect is first order, and the crossing is looked for;
  * **added mass** -- the update form diverges and the interface form does not,
    at a field-to-FIELD seam.

The artifact-backed tests re-derive the published numbers from
`out/w136/w136.json` and skip when it is absent.  The live tests build graphs and
probe seams, which costs about a minute; the marches are not re-run here.
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys

# Before numpy.  The build repo pulls torch in and torch's OpenMP beside numpy's
# MKL aborts the interpreter inside a dense solve -- `test_tier20`'s note, and it
# is not optional here either.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np                                                    # noqa: E402
import pytest                                                         # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas import compile_scheme                                      # noqa: E402
from atlas.capability import (                                        # noqa: E402
    EllipticSubsolve, MotionClass, TimeDiscretization,
)
from atlas.envelope import Hypothesis, Status                         # noqa: E402
from atlas.graph import Decomposition                                 # noqa: E402
from atlas.ports import PortType, ResponseHalf                        # noqa: E402
from atlas.verdict import Verdict                                     # noqa: E402

ARTIFACT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "out", "w136", "w136.json")

nrm = np.linalg.norm


def _artifact():
    if not os.path.exists(ARTIFACT):
        pytest.skip(f"no CS-12 artifact at {ARTIFACT}; run "
                    "scripts/w136_wing_fsi.py")
    with open(ARTIFACT, encoding="utf-8") as fh:
        return json.load(fh)


def _stage(name: str):
    a = _artifact()
    if name not in a:
        pytest.skip(f"stage {name!r} not in the artifact; run "
                    f"scripts/w136_wing_fsi.py --stages {name}")
    return a[name]


def _have_expert() -> bool:
    try:
        from atlas.cases import ground_effect as GE
        from atlas.cases import wing_fsi as WF
        GE.load_reference()
        WF.load_solvers()
        return GE.torch is not None
    except Exception:
        return False


needs_expert = pytest.mark.skipif(
    not _have_expert(),
    reason="reference.WindowNS + thermostruct2d (ATLAS_BUILD_REPO) and torch "
           "are all required")

if _have_expert():
    from atlas.cases import wing_fsi as WF                            # noqa: E402
    import torch                                                      # noqa: E402


def _flat_field():
    """A freestream field, which is all the DECLARATION tests need."""
    return (np.full((WF.NY, WF.NX), WF.U_INF), np.zeros((WF.NY, WF.NX)))


# ---------------------------------------------------------------------------
# 1. W114 -- the rule settled before the build
# ---------------------------------------------------------------------------


@needs_expert
def test_the_fsi_graph_compiles_with_no_refusals():
    """**W114.** Before 2026-09-04 `L2/R10` refused every graph containing a
    quasi-static structural agent, because it read `elliptic_subsolve` alone and
    never asked whether the decomposition cut that agent.  Here it does not: the
    fluid is tiled and `Omega_solid` is one agent's whole domain."""
    u, v = _flat_field()
    g, _e = WF.build(u, v, motion=False)
    r = compile_scheme(g)
    assert r.verdict is Verdict.ADMIT_UNCERTIFIED
    assert [f"{d.layer}/{d.rule}" for d in r.decisions.refusals] == []
    sole = [d for d in r.decisions if d.rule == "R10/sole-family"]
    assert len(sole) == 1 and sole[0].verdict is Verdict.ADMIT
    assert "STRUCT" in sole[0].subject


@needs_expert
def test_the_r10_narrowing_is_a_premise_check_and_not_an_escape():
    """A rule with nothing to fire on is not a rule.

    Declare the structure with the FLUID's `governing_family` -- which is what a
    window tiling looks like from R10's side -- and the refusal comes straight
    back.  This is the control that says the narrowing checks R10's own premise
    rather than switching it off.
    """
    u, v = _flat_field()
    g, _e = WF.build(u, v, motion=False)
    for a in g.agents:
        if a.agent_id == "STRUCT":
            caps = copy.copy(a.capabilities)
            object.__setattr__(caps, "governing_family",
                               "incompressible-navier-stokes-2d")
            a.capabilities = caps
    r = compile_scheme(g)
    assert "L2/R10" in [f"{d.layer}/{d.rule}" for d in r.decisions.refusals]


@needs_expert
def test_the_structural_agent_is_irreducibly_embedded():
    """There is no `split-step` variant on this side and that is structural: a
    quasi-static solve has no time derivative to sub-step, so the agent is
    EMBEDDED or it is not an agent.  W114 changed what that COSTS, not what it
    is."""
    u, v = _flat_field()
    _g, e = WF.build(u, v, motion=False)
    caps = WF.structure_capabilities(e["STRUCT"])
    assert caps.elliptic_subsolve is EllipticSubsolve.EMBEDDED
    assert caps.time_discretization is TimeDiscretization.IMPLICIT
    assert "cut" in WF.R10_HANDLE.lower()


@needs_expert
def test_the_halo_rule_has_the_same_scope_defect_one_rule_along():
    """**W136.** `L2/R10/halo` decertifies the structural agent for being
    implicit with a nonzero stencil, and there is no halo on this seam to be
    inadequate: `Gamma` is a PHYSICAL boundary of `Omega_solid`, the structure is
    not tiled, and no overlap exists on that side to outrun anything.  Same
    premise as R10's, same gap, one rule along -- recorded rather than fixed,
    because a decertification that over-fires is honest and noisy where a
    refusal that over-fires is terminal."""
    u, v = _flat_field()
    g, _e = WF.build(u, v, motion=False)
    r = compile_scheme(g)
    halo = [d for d in r.decisions if d.rule == "R10/halo"]
    assert halo and "STRUCT" in halo[0].subject
    assert halo[0].verdict is Verdict.ADMIT_UNCERTIFIED


@needs_expert
def test_the_deforming_graph_is_refused_at_interface_motion():
    """CS-10's refusal, inherited unchanged and correctly: a deflecting wing is a
    moving interface, no rule exists, and the refusal is priced rather than
    repaired."""
    u, v = _flat_field()
    g, _e = WF.build(u, v, motion=True)
    r = compile_scheme(g)
    assert r.verdict is Verdict.REFUSE
    rules = {d.rule for d in r.decisions.refusals}
    assert rules == {"InterfaceMotion"}
    assert r.envelope.values[Hypothesis.E2] is Status.FAILS


# ---------------------------------------------------------------------------
# 2. the declaration
# ---------------------------------------------------------------------------


@needs_expert
def test_the_plate_lives_inside_one_window_in_both_axes():
    """W124's discipline, on the axis CS-10 did not need.

    `GroundTiling` places the x-cuts clear of the plate and says nothing about
    y, because CS-10's plate hugged a floor that was always in the bottom row.
    Here the plate sits at mid-height and the y-overlap band is a real hazard.
    """
    t = WF.DEFAULT_TILING
    assert t.wing_window() == "F10"
    assert t.cuts_clear_of_wing() > 0
    assert t.rows_clear_of_wing() > 0
    lo, hi = t.wing_rows()
    for j in range(t.n_row - 1):
        a = (j + 1) * t.stride
        assert hi <= a or a + t.halo <= lo


@needs_expert
def test_the_two_agents_declare_genuinely_different_families():
    """§4's CS-12 row in one assertion.  Six of the nine real case studies before
    this one couple incompressible Navier-Stokes to incompressible
    Navier-Stokes; CS-10's spring declares the flow's own family deliberately,
    because an algebraic closure within a continuum problem is not a different
    continuum problem.  An elastic solid IS."""
    u, v = _flat_field()
    _g, e = WF.build(u, v, motion=False)
    fluid = WF.flow_capabilities(e[WF.DEFAULT_TILING.wing_window()])
    struct = WF.structure_capabilities(e["STRUCT"])
    assert fluid.governing_family == "incompressible-navier-stokes-2d"
    assert struct.governing_family == "plane-stress-elasticity-2d"
    # both declare lambda_ref, which is what keeps tau defined when E3 fails
    assert fluid.lambda_ref and struct.lambda_ref


@needs_expert
def test_e3_fails_at_the_wetted_seam_and_tau_survives_it():
    """`CASE-STUDY-GUIDE`'s multiphysics box, exercised: E3's failure costs the
    monolithic reference and nothing else, because a reference PAIR exists."""
    u, v = _flat_field()
    g, _e = WF.build(u, v, motion=False)
    r = compile_scheme(g)
    assert r.envelope.values[Hypothesis.E3] is Status.FAILS
    assert any("wet" in ev for ev in r.envelope.evidence[Hypothesis.E3])
    assert g.measured.tau == 0.0


@needs_expert
def test_both_sides_declare_the_same_response_half():
    """L3/C9 refuses a seam whose two sides disagree about which half of the
    conjugate pair they return.  Both return the EFFORT here -- the fluid the
    load on the surface, the structure the reaction it takes to hold it."""
    u, v = _flat_field()
    _g, e = WF.build(u, v, motion=False)
    fluid = WF.flow_capabilities(e[WF.DEFAULT_TILING.wing_window()])
    struct = WF.structure_capabilities(e["STRUCT"])
    assert fluid.port("wet:MECH").response_half is ResponseHalf.EFFORT
    assert struct.port("wet:MECH").response_half is ResponseHalf.EFFORT
    assert fluid.port("wet:MECH").port_type is PortType.MECH


@needs_expert
def test_the_wetted_seam_declares_zero_null_dimension():
    """A FIFTH row for the `n_0(Gamma)` table, and it differs from CS-9's by the
    BOUNDARY CONDITION rather than by the physics: a free elastic body's uniform
    normal velocity is a rigid translation and produces no reaction, so
    `n_0 = 1`; a CLAMPED one bends under the same trace, so `n_0 = 0`."""
    u, v = _flat_field()
    g, _e = WF.build(u, v, motion=False)
    wet = [c for c in g.connections if c.seam_id == "wet"][0]
    assert wet.expected_null_dim == 0
    assert wet.port_type is PortType.MECH


# ---------------------------------------------------------------------------
# 3. the structural operator -- controls, not floors
# ---------------------------------------------------------------------------


@needs_expert
def test_the_surface_compliance_is_symmetric_by_betti():
    """The pairing is NORMAL traction against NORMAL displacement, and that is
    what makes reciprocity hold in the read variable.  The obvious alternative --
    vertical against vertical -- is asymmetric at 2.5e-4, because two load
    systems that push along `n` and are read along `y` do not satisfy Betti's
    identity in `y`."""
    op = WF._surface_operator()
    C = op["C"]
    assert np.abs(C - C.T).max() / np.abs(C).max() < 1e-9
    assert op["symmetry"] < 1e-9


@needs_expert
def test_the_reduced_surface_operator_is_the_experts_own_answer():
    """`FSIRollout` never re-runs the FE solver inside a march; it applies
    ``compliance @ traction``.  For a LINEAR structure that is the same number,
    and the claim is asserted against `solve_mechanical` rather than argued."""
    st = WF.WingStructure()
    t = np.random.default_rng(0).normal(size=WF.N_STATION) * 0.3
    reduced = st.compliance @ t
    expert = st.deflect(t)
    assert nrm(reduced - expert) / nrm(expert) < 1e-9


@needs_expert
def test_the_structure_is_exactly_linear_in_its_stiffness():
    """Linear elasticity is exactly linear in ``1/E`` at fixed ``nu``, which is
    what makes ``E*`` differentiable through a march that never re-solves.  Not
    'to a tolerance': to the bit."""
    a = WF.WingStructure(e_star=WF.E_STAR)
    b = WF.WingStructure(e_star=2.0 * WF.E_STAR)
    assert np.array_equal(b.compliance * 2.0, a.compliance)


@needs_expert
def test_the_structural_stiffness_is_positive_definite():
    """`WingStructure.storage` is a real energy and E7's passive branch is
    earned rather than declared, because ``S_e`` is SPD -- which is also why the
    interface Newton system is."""
    op = WF._surface_operator()
    ev = np.linalg.eigvalsh(op["S_e"])
    assert ev.min() > 0.0
    st = WF.WingStructure(delta=np.full(WF.N_STATION, 1e-3))
    assert st.storage() > 0.0


@needs_expert
def test_the_structural_expert_declines_outside_small_strain():
    """The predicate is the constitutive law's own hypothesis, and the harness
    RAISES there rather than clamping: a hard stop is a fitted parameter and
    this expert has none."""
    st = WF.WingStructure()
    assert st.validity(np.zeros(WF.N_STATION))
    assert not st.validity(np.full(WF.N_STATION, 2.0 * WF.DELTA_MAX))


# ---------------------------------------------------------------------------
# 4. the seam
# ---------------------------------------------------------------------------


@needs_expert
def test_the_structural_response_reaches_every_station():
    """**The physics, not the architecture.**  W93 measured a frozen neural
    operator's response nonzero in all 128 seam cells and called it a fact about
    the architecture.  A quasi-static elliptic operator's inverse is dense, so
    the same reading here has the opposite cause -- and the declared
    ``stencil_radius * substeps`` is 1."""
    from atlas.probe import support_reach
    st = WF.WingStructure()
    r = support_reach(st.respond, "wet:MECH", st.probe_base("wet:MECH"),
                      amplitude=1e-2)
    assert r.nonzero == WF.N_STATION
    caps = WF.structure_capabilities(st)
    assert caps.stencil_radius * caps.substeps_per_macro_step == 1


@needs_expert
def test_the_probed_fluid_block_at_the_wetted_seam_is_not_zero():
    """**W130**, pinned.  The plate is declared in GLOBAL coordinates and a
    window sees it in its own frame; with an offset unsubtracted the plate sits
    outside the window's array, `grid_sample`'s border clamp returns a constant,
    and the probed block comes back EXACTLY zero -- byte-identical to a
    genuinely empty interface problem, with every other diagnostic healthy."""
    from atlas.probe import probe_block
    from atlas.transfer import InterfaceSpace
    u, v = _flat_field()
    g, e = WF.build(u, v, motion=False, flux_mode="reaction")
    fw = e[WF.DEFAULT_TILING.wing_window()]
    caps = WF.flow_capabilities(fw)
    port = caps.port("wet:MECH")
    space = InterfaceSpace(seam_id="wet", dim=port.effective_resolution)
    blk = probe_block(caps, port, space, port.prolongation, "wet")
    assert not np.allclose(blk.S, 0.0)
    assert nrm(blk.S, 2) > 1e-3


@needs_expert
def test_the_normal_pairing_is_the_power_and_the_vertical_one_is_not():
    """**Why the seam's conjugate pair is normal and not vertical.**

    The surface's force is along ``n`` and its velocity is along ``n``, so
    ``f_n w_n`` is the interface power exactly.  The obvious alternative -- the
    VERTICAL traction against the VERTICAL velocity, which is what CS-10's
    lumped seam carried because a ride height is vertical -- projects BOTH
    halves and under-reports the power by exactly ``cos^2(alpha)``, which is
    11.7% at this incidence.  Two correctly-declared EFFORT halves whose product
    is not the power: W132's family, in a form that is arithmetic rather than
    physical.
    """
    wing = WF.FlexWing()
    # a one-signed relative velocity: an antisymmetric one makes both powers
    # sum to machine zero and the comparison meaningless, which is the shape of
    # W106 -- a difference quoted against a quantity that is itself zero
    w = torch.linspace(0.2, 0.8, WF.N_STATION, dtype=WF.TORCH_DTYPE)
    f_n = wing.normal_traction(w)
    w_plate = torch.full((WF.N_STATION,), 0.1, dtype=WF.TORCH_DTYPE)
    p_normal = float((f_n * w_plate).sum())
    p_vertical = float((wing.vertical_traction(w)
                        * (w_plate * wing.n_hat[1])).sum())
    ny = float(wing.n_hat[1])
    assert abs(p_vertical - p_normal * ny ** 2) <= 1e-12 * abs(p_normal)
    assert abs(p_vertical / p_normal - math.cos(WF.ALPHA) ** 2) < 1e-12


@needs_expert
def test_the_agent_takes_exactly_one_substep_per_exchange():
    """`CASE-STUDY-GUIDE` mistake 5, guarded.

    R10b sets the exchange interval to ``dt_native / substeps_per_macro_step``
    so the composed step is the same splitting as the agent's own.  Declaring a
    cadence the agent does not actually take moves `tau` two orders with every
    other diagnostic healthy, and the whole defect is charged to the agent.
    """
    import os
    import numpy as _np
    z = os.path.join(os.path.dirname(ARTIFACT), "settled.npz")
    if not os.path.exists(z):
        pytest.skip("no settled field; run scripts/w136_wing_fsi.py --stages setup")
    d = _np.load(z)
    r = WF.FSIRollout(tiling=WF.SINGLE_TILING, coupling="tight", motion=True)
    res = r.run(steps=2, u0=d["u"], v0=d["v"])
    assert res.substeps.size == 2 * WF.EXCHANGES
    assert set(res.substeps.tolist()) == {1}


# ---------------------------------------------------------------------------
# 5. the gate, from the artifact
# ---------------------------------------------------------------------------


def test_the_split_reproduces_the_referent():
    g = _stage("gate")
    cols = g["columns"]
    assert g["steps"] >= 240
    # the lag alone, and the cut alone
    assert cols["lag 1, no cut"]["max_rel"] < 5e-2
    assert cols["tight, six windows"]["max_rel"] < 5e-2


def test_the_lag_defect_is_first_order_in_the_lag():
    g = _stage("gate")
    order = g["lag_order"]
    assert order is not None
    for o in order:
        assert 0.7 < o < 1.6, order


def test_the_crossing_was_looked_for():
    """Tier 23's standing rule as a GATE rather than as a discovery: a
    comparison of two configurations is not a result until it has been marched
    past the point where the curves could cross, and the crossing has to be
    looked for rather than assumed absent."""
    g = _stage("gate")
    for tag, col in g["columns"].items():
        assert "sign_changes" in col
        assert col["at_step"] <= g["steps"]


def test_the_zero_cut_control_is_reproducible():
    """A control, and NOT a floor (W106): the pipeline's own bitwise floor is
    exactly zero and bounds nothing, so no difference is quoted against it."""
    g = _stage("gate")
    assert g["reproducible_bitwise"] is True


def test_the_residual_closes_with_the_deformation_term_and_not_without():
    r = _stage("residual")
    assert r["with_motion"] < 0.2
    assert r["without_motion"] > 0.5
    assert r["factor"] > 5.0
    # and over the settled second half, where the half-step energy term has
    # nothing left to contribute
    assert r["settled"] < 1e-4


def test_the_transient_residual_is_the_half_step_accounting_and_not_the_bond():
    """At a converged interface solve the fluid's effort is the structure's
    reaction at the END of the exchange while the energy gain is taken at its
    MIDDLE, and the difference is exactly ``1/2 dd^T S_e dd``.  Subtracting the
    term the march accumulates for it removes most of the TRANSIENT residual --
    which turns the identification from an argument into a measurement.

    **The settled half of this assertion was removed at the 2026-09-04
    verification pass and its replacement is below (W140).**  It read
    ``corrected_settled < 0.1 * settled``, which held at the 120-step horizon
    the stage was first run at and fails at 480 -- not because the term is
    misidentified but because the term is second order in the interface's
    motion, so by step ~240 the raw residual has fallen to the term's own order
    and there is nothing left for it to remove.  The window the old assertion
    called `settled` was transient by a factor of 49.  What replaces it is
    `test_the_half_step_correction_is_a_transient_repair_and_says_so`, which
    asserts BOTH halves: that it works on the transient and that it does not
    work on the tail.
    """
    r = _stage("residual")
    assert r["corrected"] < 0.5 * r["with_motion"]
    # the transient, where the term is what is left -- quoted with its horizon
    q = r["quartiles_corrected"]
    assert q[0] < 0.2 * r["quartiles_with"][0]


# ---------------------------------------------------------------------------
# 6. added mass, from the artifact
# ---------------------------------------------------------------------------


def test_the_update_form_diverges_and_the_interface_form_does_not():
    """§4.2's rule at a field-to-FIELD seam.  CS-10 measured it at a lumped one
    and the mechanism does not depend on the partner being lumped: a structure
    with no mass moved the whole way to its own equilibrium inside one
    macro-step implies a surface velocity of order ``delta_eq / dt``."""
    a = _stage("addedmass")
    assert a["three"]["tight"]["ok"] is True
    assert a["three"]["lagged"]["ok"] is True
    assert a["three"]["staggered"]["ok"] is False
    # and the failure is CLASSIFIED: the scheme diverging, not the constitutive
    # envelope being left, which are different things that look identical from
    # outside a try block
    assert a["three"]["staggered"]["why"] in ("envelope", "blowup")
    assert a["load_ratio"] > 10.0


def test_the_divergence_boundary_is_outside_the_small_strain_envelope():
    """Not a defect of this construction: at aeroelastic divergence the
    equilibrium deflection is unbounded BY DEFINITION, so a linear structural
    model's own small-strain hypothesis is violated before the boundary is
    reached.  Reported as the scope statement it is."""
    a = _stage("addedmass")
    assert 0.0 < a["mu_at_reference"] < 1.0
    inside = [r["inside_envelope"] for r in a["ratios"].values() if r["mu"] > 0.9]
    assert inside and not any(inside)


def test_the_horizon_is_reported_with_the_gradient():
    """Spec §0.4: a quantity read off a rollout is reported with the horizon it
    was read at, and a report without one is REFUSED rather than decertified."""
    h = _stage("horizon")
    assert h["rows"] and "N" in h["rows"][-1]
    assert "n_sign" in h and "validity_limit" in h
    assert h["fd_agreement"] < 1e-2


# ---------------------------------------------------------------------------
# 7. the verification pass, 2026-09-04 -- horizons, and what moved on them
# ---------------------------------------------------------------------------


def test_the_residual_ordering_was_checked_step_by_step_and_not_on_maxima():
    """Tier 23's rule applied to an ORDERING rather than to a trajectory.

    "R closes with the deformation term and does not close without it" is a
    comparison of two configurations, so a maximum over the march does not
    settle it: two curves can swap at most of the steps underneath and still
    have their maxima in the published order.  The crossing is LOOKED FOR.
    """
    r = _stage("residual")
    cross = {c["lo"]: c for c in r["crossings"]}
    c = cross["with_motion"]
    assert c["hi"] == "without_motion"
    assert c["n_crossed"] == 0, c["steps"][:8]
    assert c["worst_ratio"] < 1.0


def test_the_residual_gate_was_marched_past_the_gates_own_horizon():
    """The residual was the SHORTEST-marched of this case study's three
    rollouts -- 120 macro-steps against the gate's 240 -- and it is the gate
    quantity.  Both horizons are kept, and the longer one is the headline."""
    r = _stage("residual")
    assert r["steps"] >= 480
    by = r["by_horizon"]
    assert "120" in by and "480" in by, sorted(by)
    # the closure STRENGTHENS with the horizon rather than reversing
    assert by["480"]["settled"] < by["120"]["settled"]


def test_the_half_step_correction_is_a_transient_repair_and_says_so():
    """W140.  Section 6's "removes 99% of the settled residual" was measured
    over a window a longer march shows was still transient by 49x.  Over a
    genuinely settled window the correction is worth nothing and is slightly
    the wrong way -- which dates the identification rather than retracting it."""
    r = _stage("residual")
    qw, qc = r["quartiles_with"], r["quartiles_corrected"]
    # the transient: the correction removes most of it
    assert qc[0] < 0.2 * qw[0]
    # the settled tail: it does not
    assert qc[-1] > 0.5 * qw[-1]
    # and the window section 6 called settled was not
    assert r["by_horizon"]["120"]["settled"] > 10.0 * qw[-1]


def test_every_stability_cell_is_marched_in_its_OWN_clock():
    """`undiverged` is not `stable`.  A cell at lag L gets N/L exchanges of its
    own interface equation out of an N-step march, so a map that marches every
    cell the same N compares 320 exchanges against 2.5."""
    a = _stage("addedmass")
    assert a["steps"] >= 320
    floor = a["min_lag_intervals"]
    assert floor >= 10
    for e, row in a["stability"].items():
        for lag, cell in row.items():
            assert cell["ok"] is True, (e, lag, cell["why"])
            assert cell["lag_intervals"] >= floor, (e, lag, cell["lag_intervals"])


def test_the_amplitude_growth_belongs_to_the_flow_and_not_to_the_coupling():
    """The discriminator a survival flag cannot supply.

    The tip oscillation's amplitude grows over the last two quarters -- and it
    grows by the SAME factor in the tightly coupled column, where the interface
    equation is solved at every exchange and there is no lag to be unstable in,
    and it FALLS as the coupling loosens.  A partitioned-coupling instability
    does the other thing.
    """
    a = _stage("addedmass")
    tight = a["three"]["tight"]["growth"]
    row = a["stability"]["50000"]
    assert tight > 1.0
    # the tight column is not quieter than the loosest lagged one
    assert row["32"]["growth"] < tight
    assert abs(row["1"]["growth"] - tight) / tight < 0.05
    # and it is still below the plate's own settled unsteadiness
    amp = a["three"]["tight"]["amp"]
    tip = abs(a["three"]["tight"]["tip"])
    assert amp[-1] / tip < _stage("setup")["unsteadiness"]


def test_the_published_drift_was_conservative_rather_than_wrong():
    """Section 7.2 quoted 5.0% at 80 macro-steps.  At 320 the same row spans
    0.37%: the release transient was still running, so the number on the page
    was an over-statement of the drift and not an under-statement."""
    a = _stage("addedmass")
    row = a["stability"]["50000"]
    tips = [row[l]["tip"] for l in row]
    spread = (max(tips) - min(tips)) / abs(max(tips, key=abs))
    assert spread < 0.01
    old = a["by_horizon"]["80"]["stability"]["50000"]
    old_tips = [old[l]["tip"] for l in old]
    old_spread = (max(old_tips) - min(old_tips)) / abs(max(old_tips, key=abs))
    assert old_spread > 3.0 * spread


def test_one_cell_touches_the_structural_experts_envelope():
    """And it is the cell the PoC 2 design box is drawn around: at E* = 3e4 the
    peak deflection reaches 0.0498 against a bound of 0.05, on the RELEASE
    transient rather than at the end."""
    a = _stage("addedmass")
    head = min(c["headroom"] for row in a["stability"].values()
               for c in row.values())
    assert 1.0 <= head < 1.05
