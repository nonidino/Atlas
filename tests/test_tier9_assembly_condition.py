"""Tier 9: L6/C1, the assembly accuracy condition, and the R11 that enforces it.

`AssemblyCertificate.condition` was the last named hole blocking any graph from
`admit`.  It is closed by a condition on chi alone -- convexity -- because of the
cellwise bias-variance identity

    |A(u) - u*|^2  =  sum_i chi_i |u_i - u*|^2  -  V_chi
    V_chi = sum_i chi_i u_i^2 - (sum_i chi_i u_i)^2

which holds for ANY weights summing to one and whose V_chi is non-negative for
all data exactly when chi >= 0.

Every test here is written against the rule and the algebra, not against the case
study, so the file runs without the build repo checked out.  The numbers quoted
in the docstrings come from `scripts/l6_assembly_condition.py` and
`scripts/l6_macro_consequence.py` against `reference.WindowNS`; see
`tier0-measurements` section 9.
"""

from __future__ import annotations

import numpy as np
import pytest

from atlas import (
    ADMIT_UNCERTIFIED,
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
from atlas.ports import ResponseHalf
from atlas.assembly import (
    C_MU_HALO,
    CONVEX_TOL,
    AssemblyCondition,
    GridPartitionOfUnity,
    PartitionOfUnity,
    blend_defect,
    certify,
    sigma_halo_bound,
)
from atlas.capability import linear_response, port_decl
from atlas.envelope import Hypothesis, Status
from atlas.verdict import ADMIT

M = 8
N = 10


# ---------------------------------------------------------------------------
# partitions: one convex, one signed, on the same overlapping cover
# ---------------------------------------------------------------------------


def _cover():
    """Two subdomains of a 10-cell line overlapping on cells 4..6."""
    R1, R2 = np.eye(N)[:7], np.eye(N)[4:]
    return R1, R2


def convex_pou():
    R1, R2 = _cover()
    w1 = np.array([1.0, 1, 1, 1, 0.75, 0.5, 0.25])
    w2 = np.array([0.25, 0.5, 0.75, 1, 1, 1])
    return PartitionOfUnity(N, {"a": R1, "b": R2}, {"a": w1, "b": w2})


def signed_pou(excess=0.6):
    """Sums to one exactly and leans PAST one side on the overlap.

    Extrapolatory blending is a real technique and this is what it looks like as
    a declaration.  Every check the framework had before L6/C1 passes it.
    """
    R1, R2 = _cover()
    w1 = np.array([1.0, 1, 1, 1, 0.75 + excess, 0.5 + excess, 0.25 + excess])
    w2 = np.array([0.25 - excess, 0.5 - excess, 0.75 - excess, 1, 1, 1])
    return PartitionOfUnity(N, {"a": R1, "b": R2}, {"a": w1, "b": w2})


def _locals(f):
    return {"a": f(np.arange(7.0)), "b": f(np.arange(4.0, 10.0))}


# ---------------------------------------------------------------------------
# the algebra L6/C1 rests on
# ---------------------------------------------------------------------------


def test_the_bias_variance_identity_closes_for_any_weights():
    """|A(u)-u*|^2 = sum_i chi_i |u_i-u*|^2 - V_chi, cellwise, signed or not.

    The identity needs only sum_i chi_i = 1. It is what makes the condition a
    condition on chi rather than on the data.
    """
    rng = np.random.default_rng(0)
    for pou in (convex_pou(), signed_pou()):
        loc = {"a": rng.standard_normal(7), "b": rng.standard_normal(6)}
        star = rng.standard_normal(N)
        blend = pou.assemble(loc)
        v = pou.blend_variance(loc)
        wmean = np.zeros(N)
        for key in pou.subdomains():
            R = np.asarray(pou.restrictions[key], float)
            chi = np.asarray(pou.weights[key], float)
            e = np.asarray(loc[key], float) - R @ star
            wmean += R.T @ (chi * e ** 2)
        covered = np.asarray(sum(pou.support(k) for k in pou.subdomains())) > 0
        lhs = (blend - star) ** 2
        assert np.allclose(lhs[covered], (wmean - v)[covered], atol=1e-12)


def test_the_variance_is_non_negative_exactly_when_chi_is():
    """V_chi >= 0 for all data iff chi >= 0 -- the 'exactly' is the content."""
    rng = np.random.default_rng(1)
    conv, sgn = convex_pou(), signed_pou()
    worst_conv, worst_sgn = np.inf, np.inf
    for _ in range(200):
        loc = {"a": rng.standard_normal(7), "b": rng.standard_normal(6)}
        worst_conv = min(worst_conv, conv.blend_variance(loc).min())
        worst_sgn = min(worst_sgn, sgn.blend_variance(loc).min())
    assert worst_conv >= -1e-12, "a convex partition cannot produce a negative variance"
    assert worst_sgn < -1e-3, "a signed partition must be able to, or C1 is not necessary"


def test_a_convex_blend_never_leaves_the_hull_and_a_signed_one_does():
    """The data-side witness of the same statement chi_min makes on the record.

    ``hull_escape`` is a signed distance: negative is inside the hull with room
    to spare, positive is outside and is the failure.
    """
    loc = {"a": np.zeros(7), "b": np.ones(6)}
    assert convex_pou().hull_escape(loc).max() <= 1e-12
    assert signed_pou().hull_escape(loc).max() > 1e-2


def test_chi_min_reads_the_hypothesis_off_the_declaration():
    assert convex_pou().chi_min() == pytest.approx(0.25)
    assert signed_pou().chi_min() == pytest.approx(-0.35)


def test_norm_A_is_one_only_for_a_CONVEX_partition():
    """The `norm_A` docstring said 'any partition of unity'. Measured 1.7038 on a
    signed one on the four-window tiling, so the word convex is load-bearing."""
    assert convex_pou().norm_A() == pytest.approx(1.0, abs=1e-12)
    assert signed_pou().norm_A() > 1.0


# ---------------------------------------------------------------------------
# what the old checks did NOT catch, which is why R11 is a refusal
# ---------------------------------------------------------------------------


def test_the_identity_check_passes_a_signed_partition():
    """The one-line test the framework already had cannot see this at all."""
    assert signed_pou().identity_residual() == pytest.approx(0.0, abs=1e-12)


def test_NEITHER_blend_defect_detects_a_signed_partition():
    """The argument for R11 in one test: alpha cannot be the condition.

    Both alpha definitions read NEGATIVE -- i.e. passing -- on a partition whose
    blend demonstrably leaves the hull of the values it blends.  The reason is
    structural rather than a bad choice of norm: alpha compares the blend against
    the LOCAL solves, and when the local solves are individually bad and largely
    cancel, a blend can be much better than any of them and still be much worse
    than the convex blend of the same data.

    The construction is the real one in miniature: each local is worst near its
    OWN artificial edge, which is cell 6 for `a` and cell 4 for `b`.  Measured on
    the four-window tiling, the same pattern gives old alpha = -6.06e-3 for the
    signed partition against -6.84e-3 for the working one -- same sign, same
    order -- while the composed macro-step is 166x worse.
    """
    star = np.zeros(N)
    D, s = 1.0, 0.1
    a, b = np.zeros(7), np.zeros(6)
    a[4], a[5], a[6] = 0.0, s, D
    b[0], b[1], b[2] = D, s, 0.0
    loc = {"a": a, "b": b}
    mask = np.zeros(N, bool)
    mask[4:7] = True

    bad = signed_pou()
    assert blend_defect(bad, loc, star, mask, against="max-of-norms") < 0.0
    assert blend_defect(bad, loc, star, mask) < 0.0
    assert bad.hull_escape(loc)[mask].max() > 0.3
    assert bad.chi_min() < 0.0

    good = convex_pou()
    assert good.hull_escape(loc)[mask].max() <= 1e-12
    assert (np.linalg.norm(bad.assemble(loc)[mask])
            > 2.0 * np.linalg.norm(good.assemble(loc)[mask]))


def test_the_new_blend_defect_measures_against_the_cellwise_max():
    """Convexity bounds ||A(u)-u*|| by ||max_i |u_i-u*|||, not by max_i ||u_i-u*||."""
    rng = np.random.default_rng(3)
    pou = convex_pou()
    for _ in range(50):
        star = rng.standard_normal(N)
        loc = {"a": star[:7] + rng.standard_normal(7),
               "b": star[4:] + rng.standard_normal(6)}
        assert blend_defect(pou, loc, star) <= 1e-12


# ---------------------------------------------------------------------------
# the certificate object
# ---------------------------------------------------------------------------


def test_the_certificate_fills_condition_and_says_why():
    cert = certify(convex_pou())
    assert cert.condition is not None
    assert cert.condition.holds is True
    assert "convex" in cert.condition.why()
    assert cert.as_dict()["condition"]["name"] == "L6/C1"


def test_the_certificate_reports_a_signed_partition_as_failing():
    cert = certify(signed_pou())
    assert cert.condition.holds is False
    assert cert.condition.convex is False
    assert cert.condition.identity is True, "the identity half still holds; that is the point"


def test_the_condition_is_checkable_with_no_local_solves_at_all():
    """The slot required 'checkable without the exact solution'. It is checkable
    without any solution: chi_min and the identity are properties of the
    declaration, so the certificate can be issued at compile time."""
    cert = certify(convex_pou())
    assert cert.condition.holds is True
    assert cert.condition.variance_margin is None
    assert cert.blend_defect is None


def test_the_data_side_witnesses_appear_when_locals_are_supplied():
    loc = _locals(lambda x: x ** 2)
    cert = certify(convex_pou(), locals_=loc)
    assert cert.condition.variance_margin is not None
    assert cert.condition.variance_margin >= -1e-12
    assert cert.condition.hull_escape == pytest.approx(0.0, abs=1e-12)


def test_no_assembly_leaves_the_condition_undecided_rather_than_failed():
    cert = certify(None)
    assert cert.condition.holds is None
    assert cert.condition.convex is None


def test_an_undecided_condition_is_not_a_false():
    assert AssemblyCondition().holds is None
    assert AssemblyCondition(chi_min=0.1).holds is None
    assert AssemblyCondition(chi_min=0.1, identity_residual=0.0).holds is True


def test_convex_tol_absorbs_normalization_roundoff_and_nothing_else():
    R1, R2 = _cover()
    w1 = np.array([1.0, 1, 1, 1, 0.75, 0.5, 0.25 - 0.1 * CONVEX_TOL])
    w2 = np.array([0.25, 0.5, 0.75, 1, 1, 1])
    tiny = PartitionOfUnity(N, {"a": R1, "b": R2}, {"a": w1, "b": w2})
    assert certify(tiny).condition.convex is True
    w1b = w1.copy()
    w1b[-1] = -1e-6
    real = PartitionOfUnity(N, {"a": R1, "b": R2}, {"a": w1b, "b": w2})
    assert certify(real).condition.convex is False


def test_grid_and_dense_forms_agree_on_every_new_quantity():
    """A 255^2 grid cannot afford the dense R_i, so the two must not diverge."""
    dense = convex_pou()
    grid = GridPartitionOfUnity(N, {"a": np.arange(7), "b": np.arange(4, N)},
                                {"a": dense.weights["a"], "b": dense.weights["b"]})
    loc = _locals(lambda x: np.sin(x))
    assert grid.chi_min() == pytest.approx(dense.chi_min())
    assert np.allclose(grid.blend_variance(loc), dense.blend_variance(loc))
    assert np.allclose(grid.hull_escape(loc), dense.hull_escape(loc))
    assert grid.norm_A() == pytest.approx(dense.norm_A())


# ---------------------------------------------------------------------------
# R11 through the compiler
# ---------------------------------------------------------------------------


def _caps(agent_id, **kw):
    rng = np.random.default_rng(3)
    A = rng.standard_normal((M, M)) * 0.05
    A = A + A.T + 2.0 * np.eye(M)
    defaults = dict(
        bc_channel=BCChannel.DIRICHLET,
        differentiable=Differentiable.NONE,
        dt_native=1e-2,
        governing_family="incompressible-navier-stokes-2d",
        boundary_response=linear_response(A, np.full(M, 0.05)),
        validity=lambda state, cond=None: True,
        storage=lambda u: 0.5 * float(np.dot(np.ravel(u), np.ravel(u))),
        elliptic_subsolve=EllipticSubsolve.EXPOSED,
        time_discretization=TimeDiscretization.EXPLICIT,
        stencil_radius=2,
        substeps_per_macro_step=1,
    )
    defaults.update(kw)
    return ExpertCapabilities(
        expert_id=agent_id,
        ports=[port_decl(name=f"{f}:MECH", port_type=PortType.MECH, geometry=f,
                         direction=Direction.BIDIRECTIONAL,
                         nondim={"stress": 1.0, "velocity": 1.0, "power_area": 1.0},
                         effective_resolution=M,
                         response_half=ResponseHalf.EFFORT)
               for f in ("xlo", "xhi")],
        **defaults,
    )


def _graph(pou, measured=None):
    agents = [Agent(a, _caps(a), domain=a) for a in ("A", "B")]
    conn = Connection(seam_id="s", a=("A", "xhi:MECH"), b=("B", "xlo:MECH"),
                      port_type=PortType.MECH, derive_space=True,
                      geometrically_coincident=True, expected_null_dim=0)
    return CaseGraph(name="t", agents=agents, connections=[conn],
                     decomposition=Decomposition.OVERLAPPING, overlap_cells=8,
                     partition_of_unity=pou, macro_dt=1e-2, measured=measured)


def test_r11_admits_a_convex_partition_and_stamps_e6():
    r = compile_scheme(_graph(convex_pou()))
    assert not [d for d in r.decisions.refusals if d.rule == "R11"]
    assert r.envelope[Hypothesis.E6] is Status.HOLDS


def test_r11_refuses_a_signed_partition():
    r = compile_scheme(_graph(signed_pou()))
    assert r.verdict is REFUSE
    hit = [d for d in r.decisions.refusals if d.rule == "R11"]
    assert hit, "a non-convex partition of unity must be refused, not decertified"
    assert "not convex" in hit[0].message
    assert r.envelope[Hypothesis.E6] is Status.FAILS


def test_r11_refusal_carries_the_number_that_would_have_been_wrong():
    r = compile_scheme(_graph(signed_pou()))
    hit = [d for d in r.decisions.refusals if d.rule == "R11"][0]
    assert hit.quantity == "assembled state"
    assert hit.evidence["chi_min"] < 0.0
    assert hit.evidence["norm_A"] > 1.0


def test_e6_is_unchecked_not_failed_when_no_assembly_is_declared():
    """An absent declaration establishes nothing either way -- README decision 2."""
    r = compile_scheme(_graph(None))
    assert r.envelope[Hypothesis.E6] is Status.UNCHECKED
    assert not [d for d in r.decisions.refusals if d.rule == "R11"]


def test_a_convex_partition_no_longer_decertifies_at_l6():
    """Before 2026-08-28 this was a decertification on every graph in the package."""
    r = compile_scheme(_graph(convex_pou()))
    assert not [d for d in r.decisions.decertifications
                if d.layer == "L6" and d.rule == "AssemblyCertificate"]
    assert [d for d in r.decisions.of_verdict(ADMIT) if d.rule == "L6/C1"]


# ---------------------------------------------------------------------------
# eps_tol: the W45 ingest path, one call site further on
# ---------------------------------------------------------------------------


def _measured(**kw):
    d = dict(tau=1.3353e-06, sigma=3.7334e-08, probe_state="s", scheme="split-step")
    d.update(kw)
    return MeasuredConstants(**d)


def test_eps_tol_is_set_from_the_graphs_measured_tau_and_sigma():
    r = compile_scheme(_graph(convex_pou(), measured=_measured()))
    assert r.scheme.eps_tol == pytest.approx(3.7334e-08)
    assert [d for d in r.decisions.of_verdict(ADMIT) if d.rule == "eps_tol"]
    assert not [d for d in r.decisions.decertifications if d.rule == "eps_tol"]


def test_eps_tol_takes_the_smaller_of_the_two():
    r = compile_scheme(_graph(convex_pou(), measured=_measured(tau=1e-9)))
    assert r.scheme.eps_tol == pytest.approx(1e-9)


def test_eps_tol_decertifies_when_either_is_missing_and_names_which():
    r = compile_scheme(_graph(convex_pou(), measured=_measured(sigma=None)))
    hit = [d for d in r.decisions.decertifications if d.rule == "eps_tol"]
    assert hit and "sigma is unmeasured" in hit[0].message
    assert "tau" not in hit[0].message.split("removes the only alarm the run has.")[1]
    assert r.scheme.eps_tol is None


def test_eps_tol_decertifies_with_no_measured_constants_at_all():
    r = compile_scheme(_graph(convex_pou()))
    hit = [d for d in r.decisions.decertifications if d.rule == "eps_tol"]
    assert hit and "tau and sigma are unmeasured" in hit[0].message


def _convex_pou_with_contamination():
    """The same convex partition, declaring which cells a stale datum reaches."""
    pou = convex_pou()
    pou.contaminated = {"a": np.array([False] * 5 + [True] * 2),
                        "b": np.array([True] * 2 + [False] * 4)}
    return pou


def test_w49_decertifies_when_the_partition_does_not_declare_contamination():
    """An overlapping graph whose Pi is unknown cannot have its sigma bounded."""
    r = compile_scheme(_graph(convex_pou()))
    hit = [d for d in r.decisions.decertifications if d.rule == "W49"]
    assert hit and "Pi" in hit[0].message
    assert hit[0].quantity == "sigma"


def test_w49_emits_pi_when_the_partition_declares_it():
    """Pi from the declaration, C_mu from the record -- **both**, since W56.

    This test used to pass without a declared ``C_mu`` because the sigma bound
    quoted `assembly.C_MU_HALO` unconditionally.  That is the bug W56 closed: 1.2
    is what 16 configurations of `reference.WindowNS` measured, not a default for
    somebody else's solver, and W55 is open precisely because one solver is one
    solver.
    """
    r = compile_scheme(_graph(_convex_pou_with_contamination(),
                              measured=_measured(C_mu=1.2)))
    hit = [d for d in r.decisions.of_verdict(ADMIT) if d.rule == "W49"]
    assert hit
    assert hit[0].evidence["contaminated_weight"] == pytest.approx(
        _convex_pou_with_contamination().contaminated_weight())
    assert hit[0].evidence["C_mu"] == pytest.approx(1.2)
    assert not [d for d in r.decisions.decertifications if d.rule == "W49"]


def test_w49_decertifies_when_C_mu_is_not_declared():
    """**W56.** The sigma bound may not borrow another expert's leading constant."""
    r = compile_scheme(_graph(_convex_pou_with_contamination(),
                              measured=_measured()))
    hit = [d for d in r.decisions.decertifications if d.rule == "W49"]
    assert hit and "C_mu is not declared" in hit[0].message
    assert hit[0].quantity == "sigma"
    assert not [d for d in r.decisions.of_verdict(ADMIT) if d.rule == "W49"]


def test_a_fully_measured_convex_graph_has_one_decertification_left():
    """The end state, and **what the one remaining decertification changed into**.

    Written 2026-08-28 asserting `G5/W16` -- the cut-score decomposition policy's
    **[AI Inference]** status, a theory-status item that no run could close.
    Updated the same day when it was closed: the criterion is derived (`L2/C2`,
    the restriction-defect bound) and the scalarization it replaced is falsified.

    What stands here now is `C2` and it is a **different kind of thing**: this
    synthetic graph declares no measured ``cut_defect_bound``, so the criterion is
    a theorem with no value in it.  That is a missing measurement, and the very
    next test closes it by declaring one.  The distinction is the whole point of
    the change and is why this test kept its name.
    """
    r = compile_scheme(_graph(
        _convex_pou_with_contamination(),
        measured=_measured(L=0.98, gamma=0.0, norm_A=1.0, C_mu=1.2)))
    assert r.verdict is ADMIT_UNCERTIFIED
    assert {d.rule for d in r.decisions.decertifications} == {"C2"}
    hit = [d for d in r.decisions.decertifications if d.rule == "C2"][0]
    assert hit.quantity == "cut_defect_bound"
    assert "missing measurement, not an unearned inference" in hit.message


def test_declaring_the_cut_defect_bound_reaches_admit():
    """The first `admit` this package can issue, and what it costs to get there.

    Everything else about the graph is unchanged from the test above -- same
    agents, same convex partition of unity with its contamination declared, same
    L, tau, sigma, gamma, norm_A, C_mu.  The one addition is L2/C2's measured
    constant, and it takes the verdict from `admit-uncertified` to `admit` with
    **no decertifications at all**.
    """
    r = compile_scheme(_graph(
        _convex_pou_with_contamination(),
        measured=_measured(L=0.98, gamma=0.0, norm_A=1.0, C_mu=1.2,
                           cut_defect_bound=2.6268e-07,
                           cut_defect_bound_form="chi-weighted")))
    assert r.verdict is ADMIT
    assert list(r.decisions.decertifications) == []
    assert r.unmeasured == []
    hit = [d for d in r.decisions.of_verdict(ADMIT) if d.rule == "C2"]
    assert hit and hit[0].evidence["cut_defect_bound"] == pytest.approx(2.6268e-07)


def test_w56_an_unmeasured_constant_cannot_reach_admit():
    """**W56, the backstop.** Drop ONE constant from the `admit` graph above.

    Nothing else changes: the cut criterion is declared, the partition is convex
    and declares its contamination, L6/C1 holds.  A single missing ``L`` must be
    enough to stop the verdict, because a bound quoted with a constant nobody
    measured is the silent-wrongness class, and until this rule existed
    `unmeasured` was a field the compile reported and nothing read.

    ``L`` and not ``norm_A``, deliberately: ``norm_A`` drops off the list as soon
    as a partition of unity is declared, because ``||A|| = 1`` for a convex one is
    a **theorem** rather than a measurement (§9.1).  A constant a compile can
    derive is not a constant it is missing, and this test would not be about W56
    if it used one.
    """
    r = compile_scheme(_graph(
        _convex_pou_with_contamination(),
        measured=_measured(gamma=0.0, norm_A=1.0, C_mu=1.2,
                           cut_defect_bound=2.6268e-07)))
    assert r.verdict is ADMIT_UNCERTIFIED
    assert r.unmeasured == ["L (W1)"]
    hit = [d for d in r.decisions.decertifications if d.rule == "W56"]
    assert hit and "L (W1)" in hit[0].message
    assert hit[0].evidence["unmeasured"] == ["L (W1)"]


# ---------------------------------------------------------------------------
# W49: the sigma bound for the OVERLAPPING branch
# ---------------------------------------------------------------------------


def _pou_with_contamination(chi_a, chi_b, bad_a, bad_b):
    R1, R2 = _cover()
    return PartitionOfUnity(
        N, {"a": R1, "b": R2}, {"a": np.asarray(chi_a, float), "b": np.asarray(chi_b, float)},
        contaminated={"a": np.asarray(bad_a, bool), "b": np.asarray(bad_b, bool)},
    )


def test_pi_is_one_when_the_overlap_is_narrower_than_the_domain_of_dependence():
    """Nothing in the blend is clean, so the bound degrades to ||d_lambda||."""
    pou = _pou_with_contamination(
        [1.0, 1, 1, 1, 0.75, 0.5, 0.25], [0.25, 0.5, 0.75, 1, 1, 1],
        [False] * 4 + [True] * 3, [True] * 3 + [False] * 3)
    assert pou.contaminated_weight() == pytest.approx(1.0)


def test_pi_is_zero_when_the_partition_vanishes_on_the_contaminated_band():
    """The ideal a halo scheme is reaching for: weight only on clean cells."""
    pou = _pou_with_contamination(
        [1.0, 1, 1, 1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1, 1, 1],
        [False] * 4 + [True] * 3, [True] * 3 + [False] * 3)
    assert pou.contaminated_weight() == pytest.approx(0.0)


def test_pi_falls_as_the_ramp_widens_over_the_same_overlap():
    """The mechanism the section-4 form has no term for.

    14 cells; `a` covers 0..9 and `b` covers 4..13, so the overlap is 4..9.  With
    a domain of dependence of 2, `a` is contaminated on 8..9 (its artificial edge
    is 9) and `b` on 4..5.  Widening the ramp from 2 cells to 6 moves the weight
    off the contaminated cells and Pi falls 0.5 -> 0.2, with the overlap unchanged.
    """
    n = 14
    Ra, Rb = np.eye(n)[:10], np.eye(n)[4:]
    bad_a = np.array([False] * 8 + [True] * 2)
    bad_b = np.array([True] * 2 + [False] * 8)

    def mk(ramp_a, ramp_b):
        return PartitionOfUnity(n, {"a": Ra, "b": Rb},
                                {"a": np.array(ramp_a), "b": np.array(ramp_b)},
                                contaminated={"a": bad_a, "b": bad_b})

    steep = mk([1, 1, 1, 1, 1, 1, 1, 1, 0.5, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.5, 1, 1, 1, 1, 1])
    gentle = mk([1, 1, 1, 1, 1, 0.8, 0.6, 0.4, 0.2, 0.0],
                [0.0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1, 1, 1])
    assert steep.identity_residual() == pytest.approx(0.0, abs=1e-12)
    assert gentle.identity_residual() == pytest.approx(0.0, abs=1e-12)
    assert steep.contaminated_weight() == pytest.approx(0.5)
    assert gentle.contaminated_weight() == pytest.approx(0.2)


def test_pi_is_none_and_the_bound_decertifies_when_contamination_is_undeclared():
    """Same discipline as the halo rule: decertify rather than assume clean."""
    assert convex_pou().contaminated_weight() is None
    assert sigma_halo_bound(convex_pou(), dlambda=1.0) is None


def test_the_halo_sigma_bound_is_the_product_it_says_it_is():
    pou = _pou_with_contamination(
        [1.0, 1, 1, 1, 0.5, 0.25, 0.0], [0.5, 0.75, 1.0, 1, 1, 1],
        [False] * 4 + [True] * 3, [True] * 3 + [False] * 3)
    pi = pou.contaminated_weight()
    assert sigma_halo_bound(pou, 1e-3) == pytest.approx(C_MU_HALO * pi * 1e-3)
    assert sigma_halo_bound(pou, 1e-3, C_mu=1.0) == pytest.approx(pi * 1e-3)


def test_grid_and_dense_agree_on_pi():
    dense = _pou_with_contamination(
        [1.0, 1, 1, 1, 0.5, 0.25, 0.0], [0.5, 0.75, 1.0, 1, 1, 1],
        [False] * 4 + [True] * 3, [True] * 3 + [False] * 3)
    grid = GridPartitionOfUnity(N, {"a": np.arange(7), "b": np.arange(4, N)},
                                dense.weights, contaminated=dense.contaminated)
    assert grid.contaminated_weight() == pytest.approx(dense.contaminated_weight())


def test_the_certificate_carries_pi_when_the_partition_declares_it():
    pou = _pou_with_contamination(
        [1.0, 1, 1, 1, 0.5, 0.25, 0.0], [0.5, 0.75, 1.0, 1, 1, 1],
        [False] * 4 + [True] * 3, [True] * 3 + [False] * 3)
    cond = certify(pou).condition
    assert cond.contaminated_weight == pytest.approx(pou.contaminated_weight())
    assert cond.as_dict()["contaminated_weight_W49"] is not None


def test_pi_on_the_real_four_window_tiling_is_what_was_measured():
    """Regression on the numbers section 9 of `tier0-measurements` quotes.

    `Tiling` is pure numpy -- no solver, no build repo -- so the geometry half of
    the W49 measurement is reproducible here.  d = STENCIL_RADIUS because the
    split-step scheme exchanges every sub-step.
    """
    from atlas.cases.window_ns import STENCIL_RADIUS, Tiling

    expected = {(138, 1): 0.75, (138, 4): 0.29670, (138, 8): 0.095406,
                (138, 21): 0.020000, (128, 8): 1.0}
    for (n, ramp), want in expected.items():
        pou = Tiling(n=n, ramp=ramp).partition_of_unity(d_cells=STENCIL_RADIUS)
        assert pou.contaminated_weight() == pytest.approx(want, rel=1e-4), (n, ramp)


def test_the_halo_bound_holds_on_every_measured_configuration():
    """sigma <= C_mu * Pi * ||d_lambda|| on all 11 rows of the 2026-08-28 sweep.

    Pi and ||d_lambda|| are the measured values; the assertion is that the
    factorization holds with ONE constant.  Section 4's substructuring form on the
    same branch needs C_mu = 2.2e-8, i.e. it does not.
    """
    # (label, Pi, ||d_lambda||, measured sigma) -- scripts/w49_sigma_halo.py
    rows = [
        ("halo 1", 1.0000e+00, 3.3432e-04, 2.6825e-04),
        ("halo 11", 9.5406e-02, 1.0399e-06, 3.6115e-08),
        ("halo 21", 9.5406e-02, 1.0990e-06, 3.7334e-08),
        ("halo 31", 9.5406e-02, 1.1902e-06, 3.9398e-08),
        ("halo 41", 9.5406e-02, 1.3633e-06, 4.2226e-08),
        ("halo 61", 9.5406e-02, 1.7965e-06, 5.0971e-08),
        ("flat", 7.5000e-01, 1.3539e-04, 1.1963e-04),
        ("ramp 1", 7.5000e-01, 4.7421e-05, 2.7527e-05),
        ("ramp 4", 2.9670e-01, 3.6008e-06, 3.2437e-07),
        ("ramp 8", 9.5406e-02, 1.0990e-06, 3.7334e-08),
        ("ramp 21", 2.0000e-02, 6.5648e-07, 6.2645e-09),
    ]
    implied = []
    for label, pi, dlam, sigma in rows:
        assert C_MU_HALO * pi * dlam >= sigma, f"{label}: the bound must hold"
        implied.append(sigma / (pi * dlam))
    assert min(implied) > 0.29 and max(implied) < 1.18 + 1e-9
    assert max(implied) / min(implied) < 4.1, "C_mu must behave like a constant"


# ---------------------------------------------------------------------------
# R10b: the exposed elliptic part must be applied at the AGENT's cadence
# ---------------------------------------------------------------------------


def test_r10b_sets_the_exchange_interval_to_the_agents_substep_cadence():
    """Measured: a factor-two mismatch costs two orders of magnitude in tau.

    dt 0.025 with 10 exchanges against the agent's own 5: tau 1.6e-4.
    dt 0.025 with 5: tau 8.0e-7, and the improvement over the as-built scheme
    goes from 1.8x to 361x.
    """
    r = compile_scheme(_graph(_convex_pou_with_contamination()))
    hit = [d for d in r.decisions.of_verdict(ADMIT) if d.rule == "R10b"]
    assert hit, "an exposed elliptic part must have its cadence decided, not defaulted"
    assert hit[0].evidence["substeps"] == 1
    assert r.scheme.exchange_interval == pytest.approx(1e-2)


def test_r10b_divides_the_macro_step_by_the_declared_substep_count():
    agents = [Agent(a, _caps(a, substeps_per_macro_step=10), domain=a)
              for a in ("A", "B")]
    conn = Connection(seam_id="s", a=("A", "xhi:MECH"), b=("B", "xlo:MECH"),
                      port_type=PortType.MECH, derive_space=True,
                      geometrically_coincident=True, expected_null_dim=0)
    g = CaseGraph(name="t", agents=agents, connections=[conn],
                  decomposition=Decomposition.OVERLAPPING, overlap_cells=64,
                  partition_of_unity=_convex_pou_with_contamination(), macro_dt=1e-2)
    r = compile_scheme(g)
    assert r.scheme.exchange_interval == pytest.approx(1e-3)
    hit = [d for d in r.decisions.of_verdict(ADMIT) if d.rule == "R10b"][0]
    assert hit.evidence["substeps"] == 10


def test_r10b_is_silent_when_no_agent_exposes_its_elliptic_part():
    """Nothing was moved into the composition layer, so nothing needs a cadence."""
    r = compile_scheme(_graph_with(elliptic_subsolve=EllipticSubsolve.NONE))
    assert not [d for d in r.decisions.decisions if d.rule.startswith("R10b")]
    assert r.scheme.exchange_interval == pytest.approx(1e-2)


def test_r10b_decertifies_an_exposed_agent_with_no_declared_substep_count():
    r = compile_scheme(_graph_with(substeps_per_macro_step=None))
    hit = [d for d in r.decisions.decertifications if d.rule == "R10b"]
    assert hit and "substeps_per_macro_step" in hit[0].message
    assert hit[0].quantity == "tau"


def test_r10b_refuses_two_exposed_agents_at_different_cadences():
    """The composition layer applies the elliptic part once per exchange and
    cannot be at two cadences at once."""
    caps = [_caps("A", substeps_per_macro_step=4), _caps("B", substeps_per_macro_step=9)]
    agents = [Agent(n, c, domain=n) for n, c in zip(("A", "B"), caps)]
    conn = Connection(seam_id="s", a=("A", "xhi:MECH"), b=("B", "xlo:MECH"),
                      port_type=PortType.MECH, derive_space=True,
                      geometrically_coincident=True, expected_null_dim=0)
    g = CaseGraph(name="t", agents=agents, connections=[conn],
                  decomposition=Decomposition.OVERLAPPING, overlap_cells=64,
                  partition_of_unity=_convex_pou_with_contamination(), macro_dt=1e-2)
    r = compile_scheme(g)
    assert r.verdict is REFUSE
    hit = [d for d in r.decisions.refusals if d.rule == "R10b"]
    assert hit and hit[0].quantity == "tau"


def _graph_with(**caps_kw):
    agents = [Agent(a, _caps(a, **caps_kw), domain=a) for a in ("A", "B")]
    conn = Connection(seam_id="s", a=("A", "xhi:MECH"), b=("B", "xlo:MECH"),
                      port_type=PortType.MECH, derive_space=True,
                      geometrically_coincident=True, expected_null_dim=0)
    return CaseGraph(name="t", agents=agents, connections=[conn],
                     decomposition=Decomposition.OVERLAPPING, overlap_cells=64,
                     partition_of_unity=_convex_pou_with_contamination(),
                     macro_dt=1e-2)
