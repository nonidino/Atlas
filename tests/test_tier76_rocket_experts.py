"""Tier 76 -- the rocket's first real experts, and what probing them found.

`rocket.py` was a declaration exercise: every agent's ``boundary_response`` was
`linear_response` on a seeded random matrix.  This tier wires the two sides of
the b-c seam to the build repo's own solvers and runs R0's four gates against
them.

Three of the tests here pin findings rather than features, and they are the
point of the file:

* **W301, the saturated channel.**  `compressible2d`'s isothermal wall clamps its
  ghost temperature at 20 K, so below ``T_wall = (T_i + 20)/2`` the imposed trace
  does not reach the solver AT ALL.  Both gas agents in this vault -- the
  rocket's chamber and `thermal_seam`'s duct -- sit in that regime at their own
  declared probe bases, and the field is BITWISE identical across the range.
  The tests assert bitwise equality, because "small" and "did not happen" are
  different claims and only the second one is true.
* **W66's third leg**, on real solvers rather than on a fixture built to fail.
* **The counting control**, which exists so that R1's ledger is a measurement.

Anything needing the build repo is skipped, not failed, when it is absent.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np
import pytest

from atlas.cases import rocket
from atlas.cases import rocket_experts as RE
from atlas.cases import thermal_seam as TSC
from atlas.compiler import compile_scheme
from atlas.envelope import Hypothesis, Status
from atlas.graph import Decomposition
from atlas.ports import ResponseHalf
from atlas.probe import ProbeBudget, probe_base, support_reach

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

#: The gas cadence the declaration tests run at.  They decide a DECLARATION, not
#: a physical number, so the cadence is chosen for cost and stated -- the
#: chamber's own CFL sub-step is 2.99e-7 s and the YAML's clock is 3350 of them.
DT_GAS_TEST = 1.0e-6


def _have_expert() -> bool:
    try:
        RE.load_rocket_modules()
        return True
    except Exception:
        return False


needs_expert = pytest.mark.skipif(
    not _have_expert(),
    reason="the build repo is not importable; set ATLAS_BUILD_REPO",
)


# ===========================================================================
# 1. the graph as declared -- no build repo needed
# ===========================================================================


def test_fixture_level_reproduces_the_pre_tier76_compile():
    """`declarations='fixture'` must be the Tier-75 graph.

    Tier 76 corrects three declarations that were WRONG rather than absent (the
    clocks, the decomposition, the flux matching).  A correction that cannot be
    turned off is a correction nobody can check, so the old compile stays
    reachable and this test says its decisions did not move.

    **The harder version of this claim is checked elsewhere and it is a BYTE
    identity.**  `w189_artifact_control` compares the rocket's whole emitted
    artifact against a capture taken before this tier, and it is pinned to the
    fixture level, so the graph's own `name` and `note` are the fixture's
    verbatim rather than carrying a level tag.  That is why `build` spends no
    characters on one: 40 of 40 artifacts identical is worth more than a
    readable name, and this test would pass on a graph that had quietly
    rewritten every note it has.
    """
    r = compile_scheme(rocket.build(declarations="fixture"))
    assert str(r.verdict) == "refuse"
    assert len(r.decisions.refusals) == 10
    assert len(r.decisions.decertifications) == 49
    ref = Counter((d.layer, d.rule) for d in r.decisions.refusals)
    assert ref == {("L2", "InterfaceMotion"): 9, ("L7", "R9"): 1}
    dec = Counter((d.layer, d.rule) for d in r.decisions.decertifications)
    assert dec == {("L1", "E7"): 7, ("L1", "C8"): 7, ("L1", "3.4"): 1,
                   ("L1", "E3"): 7, ("L4", "R2b/W46"): 1, ("L2", "R10/halo"): 1,
                   ("L2", "C2"): 1, ("L3", "C8"): 19, ("L5", "eps_tol"): 1,
                   ("L7", "R9/order"): 1, ("L6", "E6"): 1, ("L9", "E5"): 1,
                   ("L8", "W56"): 1}


def test_interface_motion_still_refuses_nine_ports():
    """Stage C is not touched here, and it must not be touched by accident.

    The combustion front and the plume boundary refuse, and they SHOULD: a
    moving interface silently invalidates a cached operator, and no rule exists.
    A tier that quietly made this go away would have removed the product.
    """
    for level in rocket.DECLARATION_LEVELS:
        if level == "real" and not _have_expert():
            continue
        r = compile_scheme(rocket.build(declarations=level))
        im = [d for d in r.decisions.refusals if d.rule == "InterfaceMotion"]
        assert len(im) == 9, f"level {level}: {len(im)} InterfaceMotion refusals"


def test_clocks_match_the_declared_single_source_of_truth():
    """The fixture's clocks contradicted `config/atlas_0_1.yaml`; these are the
    YAML's, hard-coded so `atlas` imports without the build repo, and asserted
    against it whenever that checkout is present."""
    assert rocket.MACRO_DT_YAML == 5.0e-2
    assert rocket.CLOCKS_YAML["c"] / rocket.CLOCKS_YAML["b"] == pytest.approx(50.0)
    if not _have_expert():
        pytest.skip("no build repo to check the YAML against")
    cfg = RE.rocket_config()
    assert dict(cfg.dt_model) == rocket.CLOCKS_YAML
    assert float(cfg.dt_macro) == rocket.MACRO_DT_YAML


def test_decomposition_is_disjoint_above_fixture():
    """`domains.verify` asserts every active cell centre is in EXACTLY one
    agent's domain, at import. That is a partition, not an overlap."""
    assert rocket.build(declarations="fixture").decomposition is Decomposition.OVERLAPPING
    assert rocket.build(declarations="known").decomposition is Decomposition.NON_OVERLAPPING


def test_counting_control_is_labelled_as_an_instrument():
    """It attaches fiction to all seven agents on purpose; it must at least do
    what it says, or the ledger it feeds is meaningless."""
    g = rocket.build(declarations="fixture", counting_control=True)
    assert all(a.capabilities.storage is not None for a in g.agents)
    assert all(a.capabilities.validity is not None for a in g.agents)
    plain = rocket.build(declarations="fixture")
    assert all(a.capabilities.storage is None for a in plain.agents)


def test_correcting_the_families_moves_the_seams_and_frees_tau():
    """`L1/E3` reports 7 records before and 7 after, and conceals both changes.

    `data/generate.py` sets ``rxn = self.reaction if a == "a" else None``, so `a`
    is the only REACTING agent.  The fixture labelled `a`, `b` and `e` alike,
    which is wrong for two of the three and was **invisible** precisely because
    it made a-b and e-b agree -- an error that hides itself is the class this
    vault keeps finding, so it is pinned rather than described.

    Two things move and neither shows in a count:

    * the multiphysics seams go {b-c, c-d, e-f} -> {a-b, b-c, c-d}; the nozzle
      outflow and the plume stop being a family boundary, and the one place a
      reaction term is genuinely on one side only becomes one;
    * ``tau_undefined_seams`` goes from 7 to ZERO, because both sides of each
      surviving seam declare `lambda_ref`. E3 still FAILS -- the families do
      differ -- but tau becomes measurable, which is W83's move for a rocket.
    """
    fix = compile_scheme(rocket.build(declarations="fixture"))
    known = compile_scheme(rocket.build(declarations="known"))

    def seams(r):
        return sorted({d.subject.split(":")[0]
                       for d in r.decisions.decertifications if d.rule == "E3"})

    assert seams(fix) == ["b-c", "c-d", "e-f"]
    assert seams(known) == ["a-b", "b-c", "c-d"]
    # the count is the SAME, which is the point of pinning the sets
    assert len([d for d in fix.decisions.decertifications if d.rule == "E3"]) == 7
    assert len([d for d in known.decisions.decertifications if d.rule == "E3"]) == 7

    assert len(fix.tau_undefined_seams) == 7
    assert list(known.tau_undefined_seams) == []
    # and E3 still fails: declaring a referent does not make the families agree
    assert known.envelope[Hypothesis.E3] is Status.FAILS


def test_gas_substeps_are_not_declared_as_one():
    """R10's halo is radius x substeps. The gas agents march thousands of CFL
    sub-steps per macro step, and declaring 1 would under-report the domain of
    dependence by three orders -- W46's failure mode, too SMALL."""
    g = rocket.build(declarations="known")
    for a in g.agents:
        if a.agent_id in rocket.GAS_AGENTS:
            assert a.capabilities.substeps_per_macro_step > 1000, a.agent_id
        else:
            assert a.capabilities.substeps_per_macro_step == 1


# ===========================================================================
# 2. geometry -- taken from the build repo, never rebuilt
# ===========================================================================


@needs_expert
def test_meshes_come_from_the_build_repos_own_grid():
    assert RE.shell_panel().shape == (232, 8)          # config: 232 x 16, two panels
    assert RE.chamber_block().shape == (112, 80)


@needs_expert
def test_the_seam_is_non_conforming_and_caps_dim_M_at_eight():
    """`rocket.py` declares M_EFF = 12 on every port. The airframe's uniform
    20.3 mm mesh gives the chamber wall 14 cells, which resolves 8 modes at the
    4-cell cutoff every classical case in this vault uses -- so dim M is 8."""
    shell = RE.ShellAgent()
    gas = RE.ChamberGasAgent(dt=DT_GAS_TEST)
    assert shell.n_seam == 14
    assert gas.n_seam == 112
    assert RE.modes_for(14) == 8
    assert RE.modes_for(112) == 57
    assert min(RE.modes_for(14), RE.modes_for(112)) == 8 < rocket.M_EFF


@needs_expert
def test_interface_basis_is_orthonormal_in_the_arc_length_gram():
    """The seam cells are NOT uniform -- the converging section's Robin faces are
    1.166x longer -- so the closed-form cosines are not orthogonal and the basis
    is orthonormalised against the real Gram."""
    shell = RE.ShellAgent()
    w = shell.seam_weights()
    assert w.max() / w.min() > 1.1                     # genuinely non-uniform
    P = RE.fourier_basis(w, RE.modes_for(shell.n_seam))
    G = P.T @ np.diag(w) @ P
    assert np.abs(G - np.eye(P.shape[1])).max() < 1e-12


# ===========================================================================
# 3. R0 gate 4 -- the control that must PASS
# ===========================================================================


@needs_expert
def test_gate4_probe_reproduces_the_shells_assembled_operator():
    """The shell's conduction step is linear, so in the pseudo-bond convention
    its probed block must reproduce an independently assembled operator. If it
    does not, the probe harness is wrong, not the physics.

    Floor: CG rtol 1e-10 over fd_step 1e-2 = 1e-8.
    """
    sh = RE.ShellAgent(flux_convention="heat")
    gas = RE.ChamberGasAgent(dt=DT_GAS_TEST)
    sh.march(RE.T_CHAMBER, RE.gas_h_on_shell_seam(gas, sh, RE.T_SHELL_COLD))
    caps = RE.shell_capabilities(sh)
    step = ProbeBudget().fd_step or max(1e-2, 100.0 * caps.reproducibility_floor)
    n = sh.n_seam
    base = probe_base(caps, "b:THERM", n)
    f0 = np.asarray(sh.respond("b:THERM", base), float)
    cols = []
    for k in range(n):
        d = np.zeros(n)
        d[k] = 1.0
        cols.append((np.asarray(sh.respond("b:THERM", base + step * d), float) - f0) / step)
    J = np.column_stack(cols)
    J_ex = RE.shell_thermal_operator(sh, convention="heat")
    rel = np.linalg.norm(J - J_ex, "fro") / np.linalg.norm(J_ex, "fro")
    assert rel < 1e-6, rel


@needs_expert
def test_gate4b_the_declared_bond_is_not_affine_and_the_jacobian_still_matches():
    """W74's cause as an algebraic fact. THERM pairs (T, q_n/T), and dividing by
    T is what makes the response nonlinear in the effort -- the pseudo-bond would
    have been affine. Two correct choices with incompatible assumptions.

    The second assertion is the control's own control: if the entropy bond were
    affine after all, the first assertion would pass for the wrong reason.
    """
    sh = RE.ShellAgent(flux_convention="entropy")
    gas = RE.ChamberGasAgent(dt=DT_GAS_TEST)
    sh.march(RE.T_CHAMBER, RE.gas_h_on_shell_seam(gas, sh, RE.T_SHELL_COLD))
    caps = RE.shell_capabilities(sh)
    step = ProbeBudget().fd_step or 1e-2
    n = sh.n_seam
    base = probe_base(caps, "b:THERM", n)
    f0 = np.asarray(sh.respond("b:THERM", base), float)
    cols = []
    for k in range(n):
        d = np.zeros(n)
        d[k] = 1.0
        cols.append((np.asarray(sh.respond("b:THERM", base + step * d), float) - f0) / step)
    J = np.column_stack(cols)
    J_jac = RE.shell_thermal_operator(sh, convention="entropy")
    J_aff = RE.shell_thermal_operator(sh, convention="heat")
    assert np.linalg.norm(J - J_jac, "fro") / np.linalg.norm(J_jac, "fro") < 1e-4
    assert np.linalg.norm(J - J_aff, "fro") / np.linalg.norm(J_aff, "fro") > 0.5


@needs_expert
def test_probe_base_is_the_operating_point_not_zero_kelvin():
    """W74. THERM's effort is an absolute temperature, so the default zero trace
    asks a 2800 K gas for its flux against a 0 K wall."""
    sh = RE.ShellAgent()
    gas = RE.ChamberGasAgent(dt=DT_GAS_TEST)
    sh.march(RE.T_CHAMBER, RE.gas_h_on_shell_seam(gas, sh, RE.T_SHELL_COLD))
    gas.T_wall_base = 449.0
    assert probe_base(RE.shell_capabilities(sh), "b:THERM", sh.n_seam).min() > 1000.0
    assert probe_base(RE.chamber_gas_capabilities(gas), "c:THERM", gas.n_seam).min() > 300.0


# ===========================================================================
# 4. R0 gate 1 -- beta, the null space, omega
# ===========================================================================


@needs_expert
def test_gate1_seam_probes_to_a_finite_beta_and_the_declared_null_space():
    """Conjugate heat transfer constrains nothing the way incompressibility
    constrains a MECH trace: a uniform temperature shift produces a uniform flux
    change, so n_0(Gamma) = 0 and nothing should be invisible to the response."""
    shell, gas, prov = RE.make_bc_experts(dt_gas=DT_GAS_TEST)
    op, _tr = RE.seam_operator(RE.build_bc_graph(shell, gas),
                               probe_state=RE.probe_state_string(prov))
    assert op.dim_M == 8
    assert op.beta is not None and np.isfinite(op.beta) and op.beta > 0.0
    assert op.null_dim == 0 == op.expected_null_dim
    assert op.excess_null_directions == 0
    assert op.operator_content is not None
    # m + 1 per block: the base probe plus one column per interface mode.
    assert all(b.n_solves == op.dim_M + 1 for b in op.blocks.values())
    assert op.conforming is False          # 112 gas cells against 14 shell cells


@needs_expert
def test_the_two_sides_probe_bases_disagree_which_is_W74s_class():
    """The gas linearises about the wall it sees; the shell about the gas it
    sees. Each is honest alone and they are over a thousand kelvin apart, which
    is exactly what `base_disagreement` was written to report."""
    shell, gas, prov = RE.make_bc_experts(dt_gas=DT_GAS_TEST)
    op, _tr = RE.seam_operator(RE.build_bc_graph(shell, gas))
    assert op.base_check is not None
    assert op.base_check["consistent"] is False
    assert op.base_check["spread"] > 1000.0


# ===========================================================================
# 5. R0 gate 3 -- the control that must FAIL (W66's three legs)
# ===========================================================================


@needs_expert
@pytest.mark.parametrize("gas_half,shell_half,expect_c9", [
    (ResponseHalf.FLOW, ResponseHalf.FLOW, False),      # the truth
    (ResponseHalf.FLOW, ResponseHalf.EFFORT, True),     # one side wrong -> refuse
    (ResponseHalf.EFFORT, ResponseHalf.FLOW, True),     # the other side wrong
    (ResponseHalf.EFFORT, ResponseHalf.EFFORT, False),  # BOTH wrong -> silent
])
def test_gate3_W66_three_legs_against_real_physics(gas_half, shell_half, expect_c9):
    """The third leg is the finding and it is NEGATIVE: a matching pair of wrong
    halves passes silently. W66 established that on a fixture built to fail;
    this is the same result against two real solvers."""
    shell, gas, _prov = RE.make_bc_experts(dt_gas=DT_GAS_TEST)
    g = RE.build_bc_graph(shell, gas, shell_half=shell_half, gas_half=gas_half)
    r = compile_scheme(g)
    c9 = [d for d in r.decisions.refusals if d.rule == "C9"]
    assert bool(c9) is expect_c9


# ===========================================================================
# 6. the measured declarations -- reach, and W301's saturated channel
# ===========================================================================


@needs_expert
def test_shell_reach_is_global_and_corroborates_embedded():
    """Backward-Euler conduction is one global solve, so its domain of
    dependence is the whole panel. `support_reach` measures it rather than
    taking the declaration's word -- and here the declaration survives."""
    shell, gas, _prov = RE.make_bc_experts(dt_gas=DT_GAS_TEST)
    sr = support_reach(shell.respond, "b:THERM", shell.base_trace())
    assert sr.is_global
    assert sr.fraction == 1.0
    assert "embedded" in sr.consistent_with
    # globally supported AND strongly decaying: not a contradiction.
    assert sr.profile[1] < 0.05 * sr.profile[0]


@needs_expert
@pytest.mark.parametrize("which", ["rocket", "thermal_seam"])
def test_W301_closed_the_isothermal_wall_now_transmits_at_its_own_probe_base(which):
    """W301, closed at Tier 86 -- with the diagnosis kept.

    `compressible2d` still sets the isothermal ghost to
    ``T_g = max(2 T_wall - T_i, 20)``, a correct positivity guard, and at this
    probe base BOTH traces still clamp it. Until Tier 86 that made the field
    BITWISE identical for the two traces -- the Dirichlet channel was saturated
    and nothing said so. The wall face's conduction is now taken one-sided from
    T_wall (`Compressible2D._isothermal_wall_conduction`), so the same two
    traces move the field, and the ghost's clamp no longer decides what the gas
    feels.
    """
    if which == "rocket":
        agent = RE.ChamberGasAgent(dt=DT_GAS_TEST)
        j, lo, hi = -1, 300.0, 449.42
    else:
        agent = TSC.GasAgent()
        j, lo, hi = 0, 300.0, 440.0
    TH, cfg = agent._TH, agent._cfg
    n = agent.n_seam if which == "rocket" else TSC.N_SEAM

    # the diagnosis: both traces are below (T_i + 20)/2, so both clamp the ghost
    T_i = TH.temperature(agent._U0, cfg.gamma, cfg.R)[:, j]
    assert np.all(2.0 * lo - T_i < 20.0) and np.all(2.0 * hi - T_i < 20.0)

    def field(Tw):
        sol = agent._solver(np.full(n, float(Tw)))
        U, _k = sol.advance(agent._U0, agent.dt)
        W = TH.cons_to_prim(U, cfg.gamma)
        return (W[..., 3] / (W[..., 0] * cfg.R))[:, j].copy()

    a_T, b_T = field(lo), field(hi)
    assert not np.array_equal(a_T, b_T), "the trace reaches the solver"
    # a colder wall draws more heat, so the gas beside it ends colder
    assert np.all(a_T <= b_T) and np.any(a_T < b_T)


@needs_expert
@pytest.mark.parametrize("which", ["rocket", "thermal_seam"])
def test_W301_closed_support_reach_no_longer_reads_exact_zeros_n_minus_1(which):
    """The L1-visible signature W301 was found by, and that it is gone.

    A saturated channel made `support_reach.exact_zeros` exactly ``n - 1``: the
    poked cell responded through the algebraic ``-T_wall`` in the flux formula
    and every other cell was bit-zero, because the solver never saw the poke.
    Measured before Tier 86: 111/111 on the chamber wall and 47/47 on
    thermal_seam's. Now the poke enters the gas, and its neighbours respond.
    """
    if which == "rocket":
        agent, port = RE.ChamberGasAgent(dt=DT_GAS_TEST), "c:THERM"
        agent.T_wall_base = 449.42
        base = agent.base_trace()
    else:
        agent, port = TSC.GasAgent(), "wall:THERM"
        base = agent.base_trace()
    sr = support_reach(agent.respond, port, base)
    assert sr.nonzero > 1
    assert sr.exact_zeros < sr.n - 1
    assert sr.reach > 0
    # and the poked cell still dominates: the response decays away from it
    assert sr.profile[1] < 0.1 * sr.profile[0]


@needs_expert
def test_W301_a_genuine_operator_does_not_show_that_signature():
    """The control that makes the signature mean something: the shell is a real
    operator and does NOT read exact_zeros = n-1."""
    shell, gas, _prov = RE.make_bc_experts(dt_gas=DT_GAS_TEST)
    sr = support_reach(shell.respond, "b:THERM", shell.base_trace())
    assert sr.exact_zeros == 0
    assert sr.nonzero == sr.n


@needs_expert
def test_W301_above_the_clamp_the_trace_does_reach_the_solver():
    """The other half of the control. If the field never moved at ANY wall
    temperature the test above would be measuring a broken harness instead of a
    saturated boundary condition."""
    agent = RE.ChamberGasAgent(dt=DT_GAS_TEST)
    TH, cfg = agent._TH, agent._cfg

    def field(Tw):
        sol = agent._solver(np.full(agent.n_seam, float(Tw)))
        U, _k = sol.advance(agent._U0, agent.dt)
        W = TH.cons_to_prim(U, cfg.gamma)
        return (W[..., 3] / (W[..., 0] * cfg.R))[:, -1].copy()

    assert not np.array_equal(field(300.0), field(2000.0))


# ===========================================================================
# 7. omega does not catch it -- the statistic's measured limit
# ===========================================================================


@needs_expert
def test_operator_content_does_not_detect_the_saturated_block():
    """W68/W71's statistic reads the gas block as healthy.

    `operator_content` is ``||S - cI||/||S||``: how much of a block is NOT a
    scalar.  A saturated channel gives a block that is bitwise DIAGONAL, but its
    diagonal varies -- the film coefficient runs 156.8 to 315.1 W/(m^2 K) along
    this wall -- so omega sits well above its 1e-3 floor and no rule fires.
    So omega above the floor does NOT mean the block carries operator content,
    and `support_reach.exact_zeros` is the discriminator that does.
    """
    shell, gas, _prov = RE.make_bc_experts(dt_gas=DT_GAS_TEST)
    op, _tr = RE.seam_operator(RE.build_bc_graph(shell, gas))
    from atlas.probe import OPERATOR_CONTENT_FLOOR
    assert op.blocks["b"].operator_content > 10.0 * OPERATOR_CONTENT_FLOOR
    # Until Tier 86 the same block was bitwise DIAGONAL in the spatial domain
    # (exact_zeros == n - 1) while omega read it as healthy -- the measured limit
    # of the statistic. W301 is closed, so the block is no longer diagonal; the
    # limit it demonstrated stands, since omega never looked.
    sr = support_reach(gas.respond, "c:THERM", gas.base_trace())
    assert sr.exact_zeros < sr.n - 1


@needs_expert
def test_the_mixed_record_says_which_port_is_physics():
    """Agent `b` has six ports and exactly one is backed by a solver.

    A reader of the certificate has to be able to tell which, so the real
    response is dispatched by port name rather than broadcast over faces it was
    never computed for -- and the R9 integral, which has no stub of the right
    shape behind it, REFUSES on the other ports instead of returning a
    wrong-shaped array that would read as physics at the first caller.
    """
    g = rocket.build(declarations="real")
    caps = {a.agent_id: a.capabilities for a in g.agents}
    for aid, real_port, other, n_real in (("b", "c:THERM", "a:THERM", 112),
                                          ("c", "b:THERM", "d:THERM", 14)):
        c = caps[aid]
        assert "REAL" in c.note and real_port in c.note
        port = c.port(real_port)
        assert port.prolongation is not None
        assert port.prolongation.matrix.shape == (n_real, RE.bc_dim_M())
        # the real port answers at its own resolution ...
        base = np.asarray(probe_base(c, real_port, n_real), float)
        assert base.size == n_real
        assert c.boundary_response_integrated(real_port, base, 1).shape == (n_real,)
        # ... and every other port declines rather than guessing
        with pytest.raises(NotImplementedError):
            c.boundary_response_integrated(other, np.zeros(rocket.M_EFF), 1)


@needs_expert
def test_both_sides_declare_the_same_dim_M():
    """Declaring each side at its own m_eff leaves the prolongation matrices
    different widths and `L3/C2/C3/C6` refuses the seam for want of a consistent
    pair. That is the compiler catching a real inconsistency, and it is how
    `bc_dim_M` came to exist -- so the invariant is pinned here."""
    g = rocket.build(declarations="real")
    caps = {a.agent_id: a.capabilities for a in g.agents}
    widths = {caps["b"].port("c:THERM").prolongation.matrix.shape[1],
              caps["c"].port("b:THERM").prolongation.matrix.shape[1]}
    assert widths == {RE.bc_dim_M()} == {8}
    r = compile_scheme(g)
    assert not [d for d in r.decisions.refusals if d.rule == "C2/C3/C6"]
