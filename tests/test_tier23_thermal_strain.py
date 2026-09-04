"""Tier 23 -- CS-9, and whether thermal strain is a bond the port algebra can type.

`atlas/cases/thermal_strain.py` and `scripts/w94_thermal_strain.py`.  Six groups,
and the first two are the ones that make the rest readable:

  * **the operator** -- ``G`` is asserted to be `solve_mechanical`'s OWN thermal
    load rather than a re-derivation of it, and the surface/body decomposition
    ``G = G_surf + G_body`` is asserted exact.  If either fails, every number in
    the case study is measuring an arithmetic mistake;
  * **the control** -- at a uniform dT the body force is machine zero and the
    surface reduction is exact, which is the positive control the distinctness
    argument needs and the reason W70's one-way case never saw the problem;
  * **the gate** -- the synchronous split reproduces the monolith BIT FOR BIT,
    the lagged one costs a measured 7.1e-3, and that error is first order in dt;
  * **the routes** -- the volumetric route is REFUSED by the package at the line
    a sixth port type would enter, and the two routes the closed vocabulary does
    permit compile: one numerically exact and uncertified, one certified-shaped
    and three orders of magnitude wrong.  Two things the probe said about the
    surface route are pinned here as well: its interface problem is one-sided to
    MACHINE ZERO, and its null direction is the constant mode -- a free solid's
    rigid translation, which is a fourth row for the guide's n_0 table;
  * **the power** -- the volumetric bond's power closes the elasticity agent's
    own balance to first order, and the two conjugate readings of it differ by a
    total derivative of an energy neither agent owns;
  * **W113** -- a graph with no connections emits an artifact, which it could not
    do before this case study existed.

The artifact-backed tests re-derive the headline numbers from `out/w94/w94.json`
and skip when it is absent.  Everything else runs the physics live, which costs a
few seconds because the mesh is 343 nodes.
"""

from __future__ import annotations

import json
import os
import sys

# Before numpy. The build repo pulls torch in and torch's OpenMP beside numpy's
# MKL aborts the interpreter inside a dense solve -- `test_tier20`'s note, and it
# is not optional here either.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np                                                    # noqa: E402
import pytest                                                         # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas import compile_scheme                                      # noqa: E402
from atlas.capability import EllipticSubsolve                         # noqa: E402
from atlas.cases import thermal_strain as TS                          # noqa: E402
from atlas.envelope import Hypothesis, Status                         # noqa: E402
from atlas.graph import CaseGraph, Decomposition                      # noqa: E402
from atlas.holes import NamedHoleError, PORT_AMENDMENT                # noqa: E402
from atlas.ports import PortType, ResponseHalf                        # noqa: E402

ARTIFACT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "out", "w94", "w94.json")

nrm = np.linalg.norm


def _artifact():
    if not os.path.exists(ARTIFACT):
        pytest.skip(f"no CS-9 artifact at {ARTIFACT}; run "
                    "scripts/w94_thermal_strain.py")
    with open(ARTIFACT, encoding="utf-8") as fh:
        return json.load(fh)


def _have_expert() -> bool:
    try:
        TS.load_solvers()
        return True
    except Exception:
        return False


needs_expert = pytest.mark.skipif(
    not _have_expert(),
    reason="thermostruct2d is not importable; set ATLAS_BUILD_REPO",
)


# ===========================================================================
# 1. the operator -- G is the expert's, not a re-derivation of it
# ===========================================================================


@needs_expert
def test_G_is_the_experts_own_thermal_load():
    """The whole case study rests on ``f_th = G (T - T_ref)`` being the load
    `solve_mechanical` assembles inside itself. Asserted, not assumed."""
    mesh, ts, ops = TS.coupling()
    rng = np.random.default_rng(7)
    for dT in (np.full(mesh.n_nodes, 100.0),
               50.0 * rng.standard_normal(mesh.n_nodes)):
        v = ops.verify_against_expert(ts, dT)
        assert v["G_vs_solve_mechanical"] < 1e-12, v
        assert v["rigid_projection"] < 1e-12, v


@needs_expert
def test_the_surface_body_decomposition_is_exact():
    """``G = G_surf + G_body`` is the divergence theorem, and it has to hold to
    machine precision or the distinctness argument is measuring quadrature."""
    mesh, _ts, ops = TS.coupling()
    rng = np.random.default_rng(11)
    for dT in (np.full(mesh.n_nodes, 100.0),
               300.0 * np.exp(-((mesh.nodes.reshape(-1, 2)[:, 0] - 0.1) / 0.03) ** 2),
               50.0 * rng.standard_normal(mesh.n_nodes)):
        lhs = ops.G.dot(dT)
        rhs = ops.G_surf.dot(dT) + ops.G_body.dot(dT)
        assert nrm(lhs - rhs) / nrm(lhs) < 1e-12


@needs_expert
def test_the_reverse_coupling_is_the_forward_one_transposed():
    """A bond is one operator read both ways. The Biot source the two-way mode
    applies is ``G^T u_dot`` and the thermal load is ``G dT``: same matrix, and
    that reciprocity is what makes the pair conjugate rather than two couplings.

    Checked as the symmetry of the bilinear form, which is the statement that
    survives discretization: ``u . G dT`` computed both ways agrees exactly.
    """
    mesh, _ts, ops = TS.coupling()
    rng = np.random.default_rng(13)
    u = rng.standard_normal(2 * mesh.n_nodes)
    dT = rng.standard_normal(mesh.n_nodes)
    forward = float(u @ ops.G.dot(dT))
    reverse = float(dT @ ops.G.T.dot(u))
    assert forward == pytest.approx(reverse, rel=1e-14, abs=0.0)


# ===========================================================================
# 2. the control -- uniform dT is where a surface bond IS enough
# ===========================================================================


@needs_expert
def test_uniform_temperature_has_no_body_force_and_the_surface_route_is_exact():
    """The positive control the distinctness argument needs.

    The body force is ``beta grad(dT)``, so it vanishes identically at a uniform
    dT and the classical equivalent-thermal-pressure reduction is EXACT there.
    That is why W70's one-way shell -- whose interesting field was through the
    thickness of an 8 mm plate -- never met this, and it is what makes the 1279x
    on the transient a finding rather than an artefact of the comparison.
    """
    mesh, _ts, ops = TS.coupling()
    elas = TS.ElasticityAgent()
    dT = np.full(mesh.n_nodes, 100.0)
    assert nrm(ops.G_body.dot(dT)) / nrm(ops.G.dot(dT)) < 1e-14
    u_full = elas.solve(ops.G.dot(dT))
    u_surf = elas.solve(ops.G_surf.dot(dT))
    assert nrm(u_surf - u_full) / nrm(u_full) < 1e-10
    # and free expansion of a free body produces no stress, which is the build
    # repo's own M1 oracle reached through this file's stress recovery
    assert nrm(elas.stress(u_full, dT)) / nrm(np.einsum(
        "ij,ej->ei", _ts_D(), _eig(dT))) < 1e-12


def _ts_D():
    _mesh, ts, _ops = TS.coupling()
    return ts.D


def _eig(dT):
    from atlas.cases.thermal_strain import load_solvers
    TSmod = load_solvers()
    _mesh, ts, _ops = TS.coupling()
    _dN, _det, N = TSmod._shape_derivs(ts.xy, 0.0, 0.0)
    e = np.zeros((ts.conn.shape[0], 3))
    e[:, 0] = e[:, 1] = ts.mat.alpha * (dT[ts.conn] * N[None, :]).sum(1)
    return e


# ===========================================================================
# 3. the gate -- does the split reproduce the monolith's stress field?
# ===========================================================================


@needs_expert
def test_the_synchronous_split_reproduces_the_monolith_bit_for_bit():
    """The gate's first half, and it is a CONTROL rather than a measurement:
    carried in full, the volume term is the same arithmetic in the same order.
    W106 -- a floor of exactly zero does not bound anything, so this test's job
    is to catch a plumbing mistake, not to certify the coupling."""
    m = TS.monolith(n_steps=8)
    s = TS.split("global-field", n_steps=8, lag=0)
    assert np.array_equal(s["sigma"], m["sigma"])
    assert np.array_equal(s["T"], m["T"])


@needs_expert
def test_the_lagged_split_costs_a_measured_amount_first_order_in_dt():
    """The gate's second half and the number a bond would have had to bound.

    One macro-step of lag is what a two-agent exchange at a shared clock does,
    and halving dt halves the error -- so it is a SPLITTING error, not a bug.
    """
    errs = []
    for dt, ns in ((1.0e-1, 8), (5.0e-2, 16)):
        m = TS.monolith(n_steps=ns, dt=dt)
        s = TS.split("global-field", n_steps=ns, dt=dt, lag=1)
        errs.append(nrm(s["sigma"] - m["sigma"]) / nrm(m["sigma"]))
    assert errs[0] > errs[1] > 0.0
    assert np.log2(errs[0] / errs[1]) == pytest.approx(1.0, abs=0.15)


@needs_expert
def test_the_two_way_reverse_half_is_real_and_the_staggered_pass_captures_it():
    """W94 asks about a TWO-WAY volumetric coupling, so the reverse half has to
    exist before anything is being answered.

    Restored, it moves the temperature by about a kelvin and the stress by
    3.8e-3 -- small, and not zero, which is the point: `ThermoStruct2D` truncates
    it, and a one-way solver cannot say whether the truncation matters. The
    staggered split then recovers most of it in one pass.
    """
    # Marched to the horizon the headline quotes, not to a convenient short one:
    # the staggered pass's share of the coupling it captures RISES through the
    # transient (0.13 at 12 steps, 0.042 at 20, 0.017 at 40), so a test at 12
    # would be testing a different claim. `positive-controls-need-a-horizon`.
    m1 = TS.monolith(n_steps=TS.N_STEPS)
    m2 = TS.monolith(n_steps=TS.N_STEPS, two_way=True)
    s2 = TS.split("global-field", n_steps=TS.N_STEPS, lag=0, two_way=True)
    one_way_gap = nrm(m2["sigma"] - m1["sigma"]) / nrm(m1["sigma"])
    staggered_gap = nrm(s2["sigma"] - m2["sigma"]) / nrm(m2["sigma"])
    assert one_way_gap > 1e-4
    assert staggered_gap < 0.1 * one_way_gap
    assert max(m2["iterations"]) < 30      # a contraction, not a search


# ===========================================================================
# 4. the routes -- what the closed vocabulary can and cannot express
# ===========================================================================


def test_the_sixth_port_type_is_refused_by_the_package():
    """The deliverable. `ports.spec_for` raises `NamedHoleError` naming
    `holes.PORT_AMENDMENT`, and `build(route='volumetric')` does not catch it --
    a case study that wants a sixth type gets the refusal, not a silently-typed
    connection."""
    with pytest.raises(NamedHoleError) as exc:
        TS.request_volumetric_port()
    assert exc.value.hole is PORT_AMENDMENT
    with pytest.raises(NamedHoleError):
        TS.build(route="volumetric")
    assert TS.CANDIDATE_PORT not in {t.value for t in PortType}


@needs_expert
def test_the_surface_mech_route_compiles_and_is_three_orders_wrong():
    """The silent-wrongness demonstration in its first form: a real `MECH` bond
    on a real surface, admissible as far as L3 is concerned, carrying 90.4% of
    the load and producing a stress field 1279x the answer."""
    g, _e = TS.build(route="surface-mech")
    assert len(g.connections) == 1
    assert g.connections[0].port_type is PortType.MECH
    r = compile_scheme(g)
    # **Changed 2026-09-04, W114 closed at CS-12.** This assertion read
    # ``== ["L2/R10"]`` from 2026-09-01 to 2026-09-04, and `TS.R10_SCOPE` said in
    # prose what the assertion could not: the refusal was real, was not about the
    # bond, and fired where R10's own derivation does not reach, because a
    # co-located split cuts no domain. R10 now checks that premise -- an EMBEDDED
    # agent is refused only when another agent shares its governing_family -- and
    # this graph's two agents are conduction and elasticity, one of each. So the
    # refusal is gone and the route compiles with none, which does not change a
    # single measurement this case study published: every number in it was taken
    # from the marches, not from the compile, and §4.4's comparison of the three
    # routes was already symmetric.
    assert [f"{d.layer}/{d.rule}" for d in r.decisions.refusals] == []
    assert [d for d in r.decisions if d.rule == "R10/sole-family"]

    m = TS.monolith(n_steps=12)
    s = TS.split("surface-mech", n_steps=12)
    assert nrm(s["sigma"] - m["sigma"]) / nrm(m["sigma"]) > 100.0


@needs_expert
def test_the_surface_mech_interface_problem_is_empty_to_machine_zero():
    """`CASE-STUDY-GUIDE` mistake 6, measured rather than reasoned about.

    A temperature field does not know its boundary is moving, so the conduction
    agent's `MECH` response is constant in the trace and its block of the
    assembled operator is EXACTLY zero. The interface problem the surface route
    poses is *empty*, not hard -- a sharper instance than W97's rotor seam,
    whose one-sidedness is a ratio rather than a zero.
    """
    g, _e = TS.build(route="surface-mech")
    r = compile_scheme(g)
    so = r.seam_operators["thermal-pressure"]
    norms = {k: float(nrm(np.asarray(getattr(b, "matrix", getattr(b, "S", b)))))
             for k, b in (so.blocks or {}).items()}
    assert norms["cond"] == 0.0, norms
    assert norms["elas"] > 1e9, norms
    assert so.one_sided == 0.0


@needs_expert
def test_a_free_solid_solid_MECH_seam_has_one_null_direction_and_it_is_rigid():
    """The fourth row of the guide's n_0(Gamma) table, and the reason for it.

    A uniform normal velocity on the only constrained face of a FREE elastic
    body is a rigid translation: no strain, no reaction, so the probed operator
    cannot see it. Not incompressibility (the fluid-fluid row), not lumpedness
    (the rotor row) -- kinematics. Declaring 0 here refuses at `L4/null-space`,
    which is the cheap direction to be wrong in.
    """
    g, _e = TS.build(route="surface-mech")
    r = compile_scheme(g)
    so = r.seam_operators["thermal-pressure"]
    assert so.null_dim == 1 and so.expected_null_dim == 1
    assert not [d for d in r.decisions.refusals if d.rule == "null-space"]

    S = np.asarray(so.S)
    _u, sv, vt = np.linalg.svd(S)
    assert sv[-1] / sv[0] < 1e-12                       # a genuine null space
    P = g.agent("elas").capabilities.port("outer:MECH").prolongation.matrix
    phys = P @ vt[-1]
    assert nrm(phys - phys.mean()) / nrm(phys) < 1e-10  # and it is the CONSTANT mode


@needs_expert
def test_the_global_field_route_is_exact_and_carries_a_BETTER_stamp():
    """The silent-wrongness demonstration in its second and worse form.

    `GlobalField` bypasses L3 entirely, so the eigenstrain crosses in full and
    the answer is exact -- and because there is no seam, the checks that would
    have failed are never run. E3 goes from `fails` to `holds`, E7 from `fails`
    to `unchecked`, and four SEAM-level decertifications are traded for one line
    saying the graph is not a composition. **Declaring a coupling out of the
    port algebra improves its envelope stamp.**

    The trade is pinned in both directions, because it is not uniformly better
    and an earlier version of this test asserted that it was: `beta` is a
    property of a probed seam, so it becomes UNMEASURABLE on the seamless route.
    (That earlier claim came from declaring `MEASURED_W94` on one graph and not
    the other -- an asymmetry in the comparison rather than a property of the
    routes, which is why `build` now defaults `measured` to None on both.)
    """
    m = TS.monolith(n_steps=8)
    s = TS.split("global-field", n_steps=8)
    assert np.array_equal(s["sigma"], m["sigma"])

    g_gf, _ = TS.build(route="global-field")
    g_sm, _ = TS.build(route="surface-mech")
    assert len(g_gf.connections) == 0 and len(g_gf.global_fields) == 1
    r_gf, r_sm = compile_scheme(g_gf), compile_scheme(g_sm)

    assert r_sm.envelope[Hypothesis.E3] is Status.FAILS
    assert r_gf.envelope[Hypothesis.E3] is Status.HOLDS
    assert r_sm.envelope[Hypothesis.E7] is Status.FAILS
    assert r_gf.envelope[Hypothesis.E7] is Status.UNCHECKED

    # The trade, named. Four SEAM properties stop being checked; what replaces
    # them is one line saying the graph is not a composition.
    rules = lambda r: {f"{d.layer}/{d.rule}" for d in r.decisions.decertifications}
    only_mech = rules(r_sm) - rules(r_gf)
    only_gf = rules(r_gf) - rules(r_sm)
    assert only_mech == {"L1/E3", "L4/E7/passivity", "L4/block-share",
                         "L4/operator-content"}
    assert only_gf == {"L3/graph"}
    # and it is NOT uniformly better: beta is a property of a probed seam.
    assert set(r_gf.unmeasured) - set(r_sm.unmeasured) == {"beta (W2)"}
    assert not set(r_sm.unmeasured) - set(r_gf.unmeasured)


# ===========================================================================
# 5. the power residual -- the term is real, and R(t) cannot see it
# ===========================================================================


@needs_expert
def test_the_volumetric_power_closes_the_elasticity_agents_own_balance():
    """port-algebra §6's gate, on the subsystem where the term is O(1).

    ``dE_el/dt = P_mech - P_Omega`` and the body is traction-free, so the
    volumetric bond's power is the whole of the elasticity agent's balance --
    and the residual is first order in dt, which is the time differencing and
    not the bond.
    """
    # The driver's own operating point, t = 2.0 s. At t = 0.8 s the coarse end
    # of the sweep is not asymptotic yet (log2 ratios 0.70, 0.87, 0.94, rising),
    # so a cheaper sweep would be testing pre-asymptotic arithmetic.
    res = []
    for dt, ns in ((1.0e-1, 20), (5.0e-2, 40), (2.5e-2, 80)):
        m = TS.monolith(n_steps=ns, dt=dt)
        cond, elas = m["agents"]
        (Ta, ua), (Tb, ub) = m["history"][-2], m["history"][-1]
        res.append(TS.power_residual(cond, elas, m["ops"], Ta, ua, Tb, ub, dt))
    rel = [r["R_elas_rel"] for r in res]
    assert rel[0] > rel[1] > rel[2] > 0.0
    for i in range(len(rel) - 1):
        assert np.log2(rel[i] / rel[i + 1]) == pytest.approx(1.0, abs=0.1)


@needs_expert
def test_the_two_conjugate_readings_differ_by_a_total_derivative():
    """`PortAmendment` field 1's obstruction, as an identity rather than a claim.

    ``P_A + P_B = d/dt (u^T G dT - 1/2 dT^T H dT)`` exactly in continuous time,
    so the discrete gap converges. The two readings are not close to each other:
    the shared energy's rate dominates both.
    """
    gaps = []
    for dt, ns in ((1.0e-1, 20), (5.0e-2, 40), (2.5e-2, 80)):
        m = TS.monolith(n_steps=ns, dt=dt)
        cond, elas = m["agents"]
        (Ta, ua), (Tb, ub) = m["history"][-2], m["history"][-1]
        r = TS.power_residual(cond, elas, m["ops"], Ta, ua, Tb, ub, dt)
        gaps.append(abs(r["conjugate_gap"] - r["d_E_shared_dt"])
                    / abs(r["conjugate_gap"]))
        assert abs(r["P_omega_B"] / r["P_omega_A"]) > 100.0
    assert gaps[0] > gaps[-1]


@needs_expert
def test_the_cross_term_is_exactly_twice_the_strain_energy_on_a_free_body():
    """`PortAmendment` field 1's obstruction is not a large number, it is an
    identity.

    On a traction-free body ``K u = (I - V V^T) G dT`` and ``V^T u = 0``, so
    ``u^T K u = u^T G dT`` and the free energy's bilinear cross term is exactly
    ``-2 E_strain``. The joint term the two agents cannot divide is **twice** the
    term the elasticity agent could own, which is why no additive split of the
    energy exists and why R(t)'s per-agent sum has lost its premise.
    """
    m = TS.monolith(n_steps=12)
    _cond, elas = m["agents"]
    u, dT = m["u"], m["T"] - TS.T_REF
    e_strain = 0.5 * u @ elas._ts.K_me.dot(u)
    e_cross = elas.cross_energy(u, dT)
    assert e_cross / e_strain == pytest.approx(-2.0, abs=1e-9)
    # and the three pieces really are the stored elastic energy
    e_self = 0.5 * dT @ m["ops"].H.dot(dT)
    assert (e_strain + e_cross + e_self) == pytest.approx(elas.energy(u, dT),
                                                          rel=1e-12)


@needs_expert
def test_the_global_power_residual_is_blind_to_the_coupling_under_test():
    """§6 calls R(t) *"the cheapest possible physical-plausibility monitor"* and
    asks for it every macro-step. On this graph the coupling it would be
    monitoring is a millionth of the transported power, and the elastic energy a
    millionth of the thermal -- so R(t) is not the instrument here, and saying so
    is what keeps it from being quoted as if it were."""
    m = TS.monolith(n_steps=16)
    cond, elas = m["agents"]
    (Ta, ua), (Tb, ub) = m["history"][-2], m["history"][-1]
    r = TS.power_residual(cond, elas, m["ops"], Ta, ua, Tb, ub, TS.DT_MACRO)
    assert abs(r["P_omega_over_P_gamma"]) < 1e-4
    assert abs(r["E_el_over_E_th"]) < 1e-4
    assert abs(r["E_cross"] / r["E_el"]) > 100.0


# ===========================================================================
# 6. the amendment, and the compiler defect this case study exposed
# ===========================================================================


def test_the_amendment_refuses_and_names_which_fields():
    """A refusal that says WHICH field is the deliverable; one that says only
    that it refused is a refusal nobody can act on."""
    ev = {"conjugate_ratio": 3395.3, "cross_over_elastic": 2679.8,
          "dim_M_over_dim_V": 1.0, "probe_solves": 344.0,
          "surface_route_stress_error": 1279.0, "uniform_control": 6.8e-17}
    v = TS.evaluate_amendment(ev)
    assert v["verdict"] == "refuse"
    assert set(v["failed"]) == {"bond", "dtn_reading"}
    assert v["fields"]["distinctness"]["verdict"] == "holds"
    assert v["fields"]["transfer"]["verdict"] == "holds-vacuously"
    assert v["seventh_field"]["name"] == "support"
    # and the slot itself now carries it, so the package's own record of the
    # procedure and the binding wiki section cannot drift apart
    assert set(PORT_AMENDMENT.declared_interface) == {
        "bond", "mapping", "transfer", "dtn_reading", "distinctness", "exercise",
        "support",
    }
    assert "EXERCISED ONCE" in PORT_AMENDMENT.inference_note


def test_the_amendment_declines_rather_than_guessing_without_evidence():
    """`holes.Unmeasured`'s spirit: a field that needs a number and has none
    reports `unmeasured`, not a default that happens to agree with the answer."""
    v = TS.evaluate_amendment(None)
    assert v["verdict"] == "unmeasured"
    assert v["fields"]["bond"]["verdict"] == "unmeasured"
    assert v["fields"]["distinctness"]["verdict"] == "unmeasured"


def test_W113_a_graph_with_no_connections_can_emit_an_artifact():
    """**The compiler defect CS-9 exposed**, and the regression that pins it.

    `_l3_connections` used to stamp E2 `unchecked` when a graph had no seams,
    and `emit.validate` refuses an unchecked E2 outright -- so a graph with zero
    connections raised `EmitRefused` and produced no artifact at all. Latent
    until the `global-field` route, the first such graph in this vault.
    """
    from atlas.cases import wind_farm
    g = wind_farm.build()
    bare = CaseGraph(name="no-connections", agents=g.agents[:2], connections=[],
                     decomposition=Decomposition.OVERLAPPING)
    r = compile_scheme(bare)
    assert r.envelope[Hypothesis.E2] is Status.HOLDS
    assert r.artifact.as_dict()["case"] == "no-connections"
    assert any(d.rule == "graph" and d.layer == "L3"
               for d in r.decisions.decertifications)


# ===========================================================================
# 7. the declarations, and the artifact
# ===========================================================================


@needs_expert
def test_the_exposed_conduction_branch_marches_the_way_it_declares():
    """`expose_elliptic` changes the MARCH and not only the record.

    A capability record that says `exposed` / `explicit` / `substeps=7` while the
    agent quietly runs a backward-Euler solve is a contradiction inside the
    declaration -- the class `missing_fields` catches for `differentiable`, and
    nothing catches for this one. So it is tested: the explicit branch takes
    `substeps_at(dt)` lumped-mass sub-steps and lands within the first-order
    splitting difference of the implicit branch, not on top of it and not far
    away.
    """
    cond_i = TS.ConductionAgent()
    cond_e = TS.ConductionAgent(expose_elliptic=True)
    assert TS.substeps_at(TS.DT_MACRO) > 1
    Ti = np.full(cond_i._mesh.n_nodes, TS.T_INIT)
    Te = Ti.copy()
    for _ in range(TS.N_STEPS):
        Ti, Te = cond_i.step(Ti), cond_e.step(Te)
    assert not np.array_equal(Ti, Te)                    # a different scheme
    rel = nrm(Te - Ti) / nrm(Ti - TS.T_REF)
    assert rel < 1e-2, rel                               # and the same physics


@needs_expert
def test_the_elasticity_agent_is_irreducibly_embedded():
    """There is no ``split-step`` variant on this side and that is structural: a
    quasi-static solve has no time derivative to sub-step, so the agent is
    EMBEDDED or it is not an agent. That much is unchanged by W114; what changed
    on 2026-09-04 is that being EMBEDDED is no longer sufficient for R10 to
    refuse. `TS.R10_SCOPE` is the record of why it used to."""
    _g, experts = TS.build(route="global-field", expose_elliptic=True)
    caps = TS.elasticity_capabilities(experts["elas"], "global-field")
    assert caps.elliptic_subsolve is EllipticSubsolve.EMBEDDED
    cond = TS.conduction_capabilities(experts["cond"], "global-field")
    assert cond.elliptic_subsolve is EllipticSubsolve.EXPOSED
    assert "co-located" in TS.R10_SCOPE.lower() or "CO-LOCATED" in TS.R10_SCOPE


@needs_expert
def test_every_port_declares_a_complete_scale_set_and_a_response_half():
    """C4 and L3/C9 from the declaration alone -- mistake 0 and mistake 7."""
    for route in ("surface-mech", "global-field"):
        g, _e = TS.build(route=route)
        for a in g.agents:
            assert not a.capabilities.missing_fields(), (route, a.agent_id)
            for p in a.capabilities.ports:
                assert p.response_half is not ResponseHalf.UNDECLARED
                chk = p.scale_check()
                assert chk.ok, (route, a.agent_id, p.name, chk.detail)


def test_the_artifact_reproduces_the_headline_numbers():
    """Every number quoted on the wiki page comes from this file."""
    d = _artifact()
    ops = d["operators"]
    assert ops["checks"]["uniform-100K"]["body_load_fraction"] < 1e-14
    assert ops["checks"]["streak-transient"]["G_vs_solve_mechanical"] < 1e-12
    assert 0.09 < ops["checks"]["streak-transient"]["body_load_fraction"] < 0.10

    sp = d["split"]
    assert sp["lag_sweep"][0]["bitwise_identical"] is True
    assert sp["lag_sweep"][1]["rel_stress_error"] == pytest.approx(7.116e-3, rel=1e-3)
    for r in sp["log2_ratios"]:
        assert r == pytest.approx(1.0, abs=0.05)

    rt = d["routes"]
    assert rt["surface_mech"]["rel_stress_error"] > 1e3
    assert rt["global_field"]["rel_stress_error"] == 0.0
    assert rt["uniform_control"]["G_body_load_fraction"] < 1e-14

    seam = d["compile"]["surface-mech"]["seam"]
    assert seam["block_norms"]["cond"] == 0.0
    assert seam["null_dim"] == 1 == seam["expected_null_dim"]

    assert d["power"]["conjugate_ratio"] > 1e3
    assert d["power"]["cross_over_elastic"] > 1e3
    assert d["amendment"]["verdict"]["verdict"] == "refuse"
    assert set(d["amendment"]["verdict"]["failed"]) == {"bond", "dtn_reading"}
