"""Tier 10: L2/C2, the decomposition criterion, and the cut score it retires.

`G5/W16` -- *"cut where the exact operator is closest to local and the
conditioning is comfortable"* -- was the last decertification standing between
`cases/window_ns.py` in `split-step` mode and `admit`, and it was a
theory-status item: identified, never derived, never tested.  Five configurations
agreed with it and agreement is not a derivation.

It is closed here in the only two ways it could be, and **both happened**:

**Derived.**  ``D_i = E_i R_i - R_i E`` is the restriction defect -- the failure
of the exact one-exchange-interval operator to commute with restriction to
subdomain ``i``, which is what "the exact operator is closest to local" denotes
and is the definition the phrase never had.  Given the partition of unity,

    A({E_i R_i u}) - E u  =  sum_i R_i^T chi_i D_i u          (identity)

and given ``chi >= 0`` besides -- L6/C1, which R11 already enforces --

    |A({E_i R_i u}) - E u|_j  <=  sum_i chi_ij |D_i u|_j  <=  max_i |D_i u|_j

at every cell and hence in every l^p norm.  The criterion is the middle term.

**Falsified.**  ``Q = (1/beta) ||S - diag S|| / ||S||`` is not a function of the
decomposition: re-declaring the same interface space in another orthonormal
frame leaves the scheme bit-identical and moves Q by 361x.  Scanned over cut
placement it ranks **backwards** (-0.853), and both of the operations it performs
on the off-diagonal mass invert the sign of a quantity that ranks correctly
(+0.853) before them.

Every test here is written against the rule and the algebra, not against
`reference.WindowNS`, so the file runs without the build repo checked out.  The
numbers quoted in the docstrings come from `scripts/w16_cut_policy.py` and
`scripts/tier0_window_ns.py`; see `tier0-measurements` section 10.
"""

from __future__ import annotations

import numpy as np
import pytest

from atlas import (
    ADMIT_UNCERTIFIED,
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
from atlas.assembly import CONVEX_TOL, GridPartitionOfUnity, PartitionOfUnity
from atlas.ports import ResponseHalf
from atlas.capability import linear_response, port_decl
from atlas.probe import (
    CUT_CONVEX_TOL,
    ProbeError,
    cut_score,
    neighbour_disagreement,
    restriction_defect_bound,
)
from atlas.verdict import ADMIT

from strip_model import (
    STENCIL_RADIUS,
    Physics,
    StripDecomposition,
    cellwise_defect_bound,
    composed_step,
    neighbour_disagreement as strip_disagreement,
    reference_step,
    rel_l2,
    restriction_defect,
    seam_operator,
    smooth_field,
)

N = 12


# ---------------------------------------------------------------------------
# the identity, as algebra
# ---------------------------------------------------------------------------


def _partition(n=N, overlap=4):
    """Two overlapping index sets on n cells with a convex ramped partition."""
    a = np.arange(0, n // 2 + overlap // 2)
    b = np.arange(n // 2 - overlap // 2, n)
    wa, wb = np.zeros(n), np.zeros(n)
    wa[a] = np.clip(np.linspace(1.0, 0.0, len(a) + 2)[1:-1] * 2.0, 0.0, 1.0)
    wb[b] = np.clip(np.linspace(0.0, 1.0, len(b) + 2)[1:-1] * 2.0, 0.0, 1.0)
    tot = wa + wb
    tot[tot <= 0] = 1.0
    return a, b, wa[a] / tot[a], wb[b] / tot[b]


def test_the_restriction_defect_identity_is_an_identity():
    """A({E_i R_i u}) - E u = sum_i R_i^T chi_i D_i, for ANY local maps.

    It needs only ``sum_i R_i^T chi_i R_i = I``.  Nothing about the maps being
    good, nothing about chi being non-negative, no PDE -- which is why it holds
    to 1e-12 on the strip model and to 3.6e-11 against `reference.WindowNS`, and
    why the *bound* below needs a hypothesis the identity does not.
    """
    rng = np.random.default_rng(0)
    a, b, ca, cb = _partition()
    u = rng.normal(size=N)
    Eu = rng.normal(size=N)                       # the exact step, whatever it is
    la, lb = rng.normal(size=len(a)), rng.normal(size=len(b))   # the local steps

    blend = np.zeros(N)
    blend[a] += ca * la
    blend[b] += cb * lb
    lhs = blend - Eu

    rhs = np.zeros(N)
    rhs[a] += ca * (la - Eu[a])
    rhs[b] += cb * (lb - Eu[b])

    assert np.allclose(lhs, rhs, atol=1e-14)


def test_convexity_is_what_turns_the_identity_into_a_bound():
    """chi >= 0 gives the cellwise bound; drop it and the bound is false.

    The same hypothesis, and the same failure mode, as L6/C1: a signed partition
    passes the identity (it is an identity) and violates the bound.  So L2/C2 is
    not a second assumption bolted on -- it is the assembly condition, read on
    the defect instead of on the solution.
    """
    a, b, ca, cb = _partition()
    # The two defects must DIFFER for the weights to matter: with |D_a| = |D_b|
    # every partition summing to one gives the same blend, signed or not, and
    # the test would pass for the wrong reason.
    Da = np.full(len(a), 1.0)
    Db = np.zeros(len(b))

    conv = np.zeros(N)
    conv[a] += ca * np.abs(Da)
    conv[b] += cb * np.abs(Db)
    assert np.all(conv <= 1.0 + 1e-14)            # <= max_i |D_i| = 1

    # the extrapolatory partition: sums to one, is not convex
    sa, sb = ca.copy(), cb.copy()
    ov = np.intersect1d(a, b)
    ia = np.searchsorted(a, ov)
    ib = np.searchsorted(b, ov)
    sa[ia] = 1.7
    sb[ib] = 1.0 - 1.7
    assert np.allclose(sa[ia] + sb[ib], 1.0)      # still a partition of unity

    signed = np.zeros(N)
    signed[a] += sa * np.abs(Da)
    signed[b] += sb * np.abs(Db)
    assert signed[ov].max() > 1.0 + 1e-9          # and the bound is violated


def test_restriction_defect_bound_returns_both_forms_and_the_worst_agent():
    locals_ = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([1.0, 1.0, 1.0])}
    ref = {"a": np.array([1.0, 1.0, 1.0]), "b": np.array([1.0, 1.0, 1.0])}
    w = {"a": np.array([0.5, 0.5, 0.5]), "b": np.array([0.5, 0.5, 0.5])}
    out = restriction_defect_bound(locals_, ref, w)
    # D_a = (0, 1, 2), D_b = 0.  chi-weighted: (0, .5, 1); max form: (0, 1, 2)
    assert out["cut_defect_bound_chi_weighted"] == pytest.approx(np.sqrt(1.25))
    assert out["cut_defect_bound_max"] == pytest.approx(np.sqrt(5.0))
    assert out["worst_agent"] == "a"
    assert out["cut_defect_bound"] == out["cut_defect_bound_chi_weighted"]
    assert restriction_defect_bound(locals_, ref, w, weighted=False)[
        "cut_defect_bound"] == pytest.approx(np.sqrt(5.0))


def test_restriction_defect_bound_refuses_a_signed_partition():
    """The bound rests on convexity, so it declines to compute on a signed one."""
    locals_ = {"a": np.ones(3), "b": np.ones(3)}
    ref = {"a": np.zeros(3), "b": np.zeros(3)}
    w = {"a": np.array([1.7, 0.5, 0.5]), "b": np.array([-0.7, 0.5, 0.5])}
    with pytest.raises(ProbeError, match="not non-negative"):
        restriction_defect_bound(locals_, ref, w)


def test_the_two_convexity_tolerances_agree():
    """L2/C2 and L6/C1 test the same inequality and must not drift apart."""
    assert CUT_CONVEX_TOL == CONVEX_TOL


# ---------------------------------------------------------------------------
# the falsification of Q, as algebra: it is not a function of the decomposition
# ---------------------------------------------------------------------------


def test_Q_moves_under_a_re_declaration_of_the_same_interface_space():
    """P -> P U declares the SAME space M, so the scheme cannot tell the difference.

    ``range(P U) = range(P)`` for orthogonal ``U``, and under the dirichlet rung
    the composed step never reads the interface basis at all.  ``S -> U^T S U``
    leaves beta and ||S|| invariant and moves the off-diagonal mass, so Q has an
    orbit of values on one decomposition while the thing it ranks has one.

    Measured on the strip model: declared (Fourier) 0.5016, random frames
    [0.419, 0.4895], eigenbasis of the symmetric part **0.001388** -- 361x.
    """
    rng = np.random.default_rng(3)
    A = rng.normal(size=(8, 8))
    S = A @ A.T + 4.0 * np.eye(8)                 # symmetric positive definite
    beta = float(np.linalg.svd(S, compute_uv=False)[-1])

    _w, V = np.linalg.eigh(S)
    S_eig = V.T @ S @ V
    beta_eig = float(np.linalg.svd(S_eig, compute_uv=False)[-1])

    assert beta_eig == pytest.approx(beta, rel=1e-10)          # beta invariant
    assert np.linalg.norm(S_eig) == pytest.approx(np.linalg.norm(S), rel=1e-10)
    assert cut_score(S_eig, beta_eig) < 1e-12 * cut_score(S, beta)


def test_Q_is_zero_in_the_eigenbasis_of_any_symmetric_seam():
    """And the split-step operator is symmetric to 0.002 (section 8.3).

    So on the graph the vault actually runs, every seam could be declared into a
    basis where Q reads zero.  A criterion that any admissible re-declaration can
    zero is not ranking decompositions.
    """
    rng = np.random.default_rng(7)
    B = rng.normal(size=(6, 6))
    S = B @ B.T + np.eye(6)
    _w, V = np.linalg.eigh(S)
    assert cut_score(V.T @ S @ V, float(np.linalg.svd(V.T @ S @ V,
                                                      compute_uv=False)[-1])) < 1e-12


# ---------------------------------------------------------------------------
# the falsification of Q, as measurement: it ranks backwards
# ---------------------------------------------------------------------------


def _small():
    """A cheap strip model: same rules, a quarter of the grid."""
    return Physics(nx=24, ny=24, nu0=8.0e-3)


def test_the_identity_closes_on_the_strip_model():
    phys = _small()
    dt = phys.stable_substep()
    dec = StripDecomposition(phys=phys, halo=6, ramp=4, cut=5)
    u = smooth_field(phys, seed=0)
    lhs = composed_step(dec, u, dt) - reference_step(dec, u, dt)
    D = restriction_defect(dec, u, dt)
    rhs = np.zeros_like(lhs)
    ws = dec.weights()
    for i in range(dec.n_strips):
        r = dec.rows(i)
        rhs[r, :] += ws[i][r, :] * D[i]
    scale = float(np.max(np.abs(lhs)))
    assert scale > 0.0
    assert float(np.max(np.abs(lhs - rhs))) / scale < 1e-9


def test_the_cellwise_bound_holds_and_the_chi_weighted_form_is_tighter():
    phys = _small()
    dt = phys.stable_substep()
    dec = StripDecomposition(phys=phys, halo=6, ramp=4, cut=5)
    u = smooth_field(phys, seed=1)
    lhs = np.abs(composed_step(dec, u, dt) - reference_step(dec, u, dt))
    D = restriction_defect(dec, u, dt)
    mx = cellwise_defect_bound(dec, D)
    wt = cellwise_defect_bound(dec, D, weighted=True)
    # Machine zero against the STATE's scale, not the bound's. Where D_i = 0 the
    # bound is exactly 0 while the blend computes chi_a x + chi_b x, which is x
    # to within one ulp and not bit-identical to it -- so the violation floor is
    # eps * ||u||_inf and no tighter. The full run reports 2.220e-16 for exactly
    # this reason and calls it machine zero.
    floor = 1e-14 * float(np.max(np.abs(u)))
    assert np.all(lhs <= mx + floor)
    assert np.all(lhs <= wt + floor)
    assert np.all(wt <= mx + floor)                 # convexity orders the two
    # and the ordering is not cosmetic: on the real tiling it is 220x
    assert np.linalg.norm(wt) < 0.5 * np.linalg.norm(mx)


def test_Q_ranks_cut_placements_backwards_while_the_derived_criterion_does_not():
    """The decisive measurement, at a size a test can afford.

    ``nu`` varies across the cut direction, so moving the cut moves it into
    stiffer or softer material.  Q goes as ``1/beta`` and beta goes as ``nu``, so
    Q prefers a cut where the diffusive coupling across it is STRONGEST -- and
    strong coupling is exactly what a lagged artificial boundary gets wrong.

    Full scan, 12 placements: rank correlation **-0.853** for Q against the
    measured composed defect and **+1.000** for the chi-weighted restriction
    defect, with Q's pick costing 92% of the available range.
    """
    phys = _small()
    dt = phys.stable_substep()
    rows = []
    for cut in (0, 3, 6, 9):
        dec = StripDecomposition(phys=phys, halo=6, ramp=4, cut=cut)
        u = smooth_field(phys, seed=0)
        S = seam_operator(dec, u, dt, m=8)
        beta = float(np.linalg.svd(S, compute_uv=False)[-1])
        D = restriction_defect(dec, u, dt)
        rows.append({
            "cut": cut,
            "Q": cut_score(S, beta),
            "Qstar": float(np.linalg.norm(
                cellwise_defect_bound(dec, D, weighted=True))),
            "defect": rel_l2(composed_step(dec, u, dt), reference_step(dec, u, dt)),
        })

    best = min(rows, key=lambda r: r["defect"])["cut"]
    assert min(rows, key=lambda r: r["Qstar"])["cut"] == best
    assert min(rows, key=lambda r: r["Q"])["cut"] != best

    def corr(key):
        rk = np.argsort(np.argsort([r[key] for r in rows])).astype(float)
        tr = np.argsort(np.argsort([r["defect"] for r in rows])).astype(float)
        return float(np.corrcoef(rk, tr)[0, 1])

    assert corr("Q") < 0.0                       # backwards
    assert corr("Qstar") == pytest.approx(1.0)   # exactly right


def test_Q_is_blind_where_the_operator_does_not_depend_on_the_cut():
    """nu uniform across the cut direction: every placement has the same S.

    Q is then constant to floating-point roundoff while the measured defect still
    varies, because the defect depends on the STATE at the cut and Q does not
    read the state.  On the full scan the defect spreads 2.29x under a Q that is
    constant to 8e-11.
    """
    phys = Physics(nx=24, ny=24, nu0=8.0e-3, nu_y_amp=0.0)
    dt = phys.stable_substep()
    qs, ds = [], []
    for cut in (0, 3, 6, 9):
        dec = StripDecomposition(phys=phys, halo=6, ramp=4, cut=cut)
        u = smooth_field(phys, seed=2)
        S = seam_operator(dec, u, dt, m=8)
        qs.append(cut_score(S, float(np.linalg.svd(S, compute_uv=False)[-1])))
        ds.append(rel_l2(composed_step(dec, u, dt), reference_step(dec, u, dt)))
    qs, ds = np.asarray(qs), np.asarray(ds)
    assert (qs.max() - qs.min()) / qs.mean() < 1e-8          # constant
    assert (ds.max() - ds.min()) / ds.mean() > 1e-3          # and the truth is not


# ---------------------------------------------------------------------------
# the reference-free surrogate, and the condition under which it is exact
# ---------------------------------------------------------------------------


def test_the_surrogate_equals_the_max_form_when_the_supports_are_disjoint():
    """halo >= 2 * reach makes the two strips' defect supports disjoint.

    Then at most one D_i is nonzero per cell, so ``|D_i - D_j| = max_i |D_i|``
    and the reference-free disagreement is not an approximation of the bound, it
    IS the bound.  Measured gap: **0.000e+00** at halo 4, 5, 6, 8, 10 against a
    reach of 2, and 6.6e-3 / 1.0e-3 at halo 2 and 3.
    """
    phys = _small()
    dt = phys.stable_substep()
    reach = STENCIL_RADIUS
    for halo, exact in ((2, False), (3, False), (6, True), (8, True)):
        dec = StripDecomposition(phys=phys, halo=halo, ramp=min(4, halo), cut=4)
        u = smooth_field(phys, seed=0)
        D = restriction_defect(dec, u, dt)
        mx = cellwise_defect_bound(dec, D)
        hat = strip_disagreement(dec, u, dt)
        gap = float(np.linalg.norm(hat - mx)) / max(float(np.linalg.norm(mx)), 1e-300)
        assert (halo >= 2 * reach) is exact
        if exact:
            assert gap == 0.0
        else:
            assert gap > 0.0


def test_neighbour_disagreement_needs_no_reference():
    """Its whole point: the arguments are local solves and a mask, and that is all."""
    locals_ = {"a": np.array([1.0, 2.0, 5.0]), "b": np.array([1.0, 2.0, 1.0])}
    mask = {("a", "b"): np.array([False, True, True])}
    assert neighbour_disagreement(locals_, mask) == pytest.approx(4.0)
    assert neighbour_disagreement(locals_, {("a", "b"): np.zeros(3, bool)}) == 0.0


# ---------------------------------------------------------------------------
# the compiler rule
# ---------------------------------------------------------------------------


M = 8


def _caps(name):
    rng = np.random.default_rng(3)
    A = rng.standard_normal((M, M)) * 0.05
    A = A + A.T + 2.0 * np.eye(M)
    return ExpertCapabilities(
        expert_id=name,
        ports=[port_decl(name=f"{f}:MECH", port_type=PortType.MECH, geometry=f,
                         direction=Direction.BIDIRECTIONAL,
                         nondim={"stress": 1.0, "velocity": 1.0,
                                 "power_area": 1.0},
                         effective_resolution=M,
                         response_half=ResponseHalf.EFFORT)
               for f in ("xlo", "xhi")],
        dt_native=1e-2,
        governing_family="incompressible-navier-stokes-2d",
        boundary_response=linear_response(A, np.full(M, 0.05)),
        elliptic_subsolve=EllipticSubsolve.EXPOSED,
        time_discretization=TimeDiscretization.EXPLICIT,
        bc_channel=BCChannel.DIRICHLET,
        differentiable=Differentiable.NONE,
        stencil_radius=2,
        substeps_per_macro_step=1,
        storage=lambda u: 0.5 * float(np.dot(np.ravel(u), np.ravel(u))),
        validity=lambda state, cond=None: True,
    )


def _pou(convex=True, contaminated=True):
    """Two subdomains of an 8-cell line overlapping on cells 3..4.

    Dense restrictions, matching `assembly.PartitionOfUnity`'s contract -- the
    index form is `GridPartitionOfUnity` and the last test in this file uses it,
    because the criterion has to work on the object a real tiling can afford.
    """
    n = 8
    R1, R2 = np.eye(n)[:5], np.eye(n)[3:]
    wa = np.array([1.0, 1.0, 1.0, 0.6, 0.3])
    wb = np.array([0.4, 0.7, 1.0, 1.0, 1.0])
    if not convex:
        wa = np.array([1.0, 1.0, 1.0, 1.7, 0.3])
        wb = np.array([-0.7, 0.7, 1.0, 1.0, 1.0])
    pou = PartitionOfUnity(n, {"a": R1, "b": R2}, {"a": wa, "b": wb})
    if contaminated:
        pou.contaminated = {"a": np.array([0, 0, 0, 1, 1], bool),
                            "b": np.array([1, 1, 0, 0, 0], bool)}
    return pou


def _graph(pou, decomposition=Decomposition.OVERLAPPING, **measured):
    m = dict(L=0.98, tau=1.3e-06, sigma=3.7e-08, gamma=0.0, norm_A=1.0, C_mu=1.2,
             probe_state="s", scheme="split-step")
    m.update(measured)
    return CaseGraph(
        name="t",
        agents=[Agent(a, _caps(a), domain=a) for a in ("A", "B")],
        connections=[Connection(seam_id="s", a=("A", "xhi:MECH"),
                                b=("B", "xlo:MECH"),
                                port_type=PortType.MECH, derive_space=True,
                                geometrically_coincident=True,
                                expected_null_dim=0)],
        decomposition=decomposition,
        overlap_cells=8, overlap=0.6,
        partition_of_unity=pou,
        macro_dt=1e-2,
        measured=MeasuredConstants(**m),
    )


def test_l2_c2_admits_an_overlapping_convex_graph_that_declares_the_bound():
    r = compile_scheme(_graph(_pou(), cut_defect_bound=2.6268e-07))
    hit = [d for d in r.decisions.of_verdict(ADMIT) if d.rule == "C2"]
    assert hit
    assert hit[0].evidence["cut_defect_bound"] == pytest.approx(2.6268e-07)
    assert not [d for d in r.decisions.decertifications if d.rule.startswith("C2")]
    assert r.verdict is ADMIT


def test_l2_c2_decertifies_a_graph_that_declares_no_bound_and_names_it():
    """A missing measurement, and the message says which -- not an inference."""
    r = compile_scheme(_graph(_pou()))
    hit = [d for d in r.decisions.decertifications if d.rule == "C2"]
    assert hit
    assert hit[0].quantity == "cut_defect_bound"
    assert "missing measurement, not an unearned inference" in hit[0].message
    assert r.verdict is ADMIT_UNCERTIFIED


def test_l2_c3_carries_the_substructuring_criterion():
    """The substructuring branch has its own criterion, **L2/C3**, since W57.

    > **Superseded 2026-08-29.**  This test used to assert the opposite -- that
    > the branch has *no* criterion and decertifies citing `C2/W57` with the
    > message *"OVERLAPPING branch only"*.  That was true and is not any more.
    > W57 closed by derivation plus measurement, so the assertion inverts; the
    > old text is kept here because a test that silently changes what it claims
    > is worse than one that says it changed.

    L2/C3 is `master-error-bound` 4's own factorization stopped one step earlier
    than section 4 stops it: ``Q_sub = ||S_M a* - chi_M|| / beta``.  A graph that
    declares its value is ranked; one that does not is decertified naming it.
    """
    r = compile_scheme(_graph(_pou(), decomposition=Decomposition.NON_OVERLAPPING,
                              cut_defect_bound=2.6268e-07))
    hit = [d for d in r.decisions.of_verdict(ADMIT) if d.rule == "C3"]
    assert hit
    assert hit[0].evidence["branch"] == "substructuring"
    assert hit[0].evidence["cut_defect_bound"] == pytest.approx(2.6268e-07)
    assert not [d for d in r.decisions.decertifications if d.rule == "C2/W57"]


def test_l2_c3_decertifies_the_substructuring_branch_when_undeclared():
    """Undeclared, the placement is unranked -- and the message carries the two
    things the measurement established that a reader would otherwise assume the
    other way round: section 4's PRODUCT form ranks negative, and the retired Q
    was that form's shape imported onto the wrong branch."""
    r = compile_scheme(_graph(_pou(), decomposition=Decomposition.NON_OVERLAPPING))
    hit = [d for d in r.decisions.decertifications if d.rule == "C3/W57"]
    assert hit
    assert "ranks NEGATIVE in 9 of 11" in hit[0].message
    assert hit[0].quantity == "cut placement"


def test_l2_c2_decertifies_when_there_is_no_partition_of_unity():
    r = compile_scheme(_graph(None, cut_defect_bound=2.6268e-07))
    hit = [d for d in r.decisions.decertifications if d.rule == "C2"]
    assert hit and "no partition of unity" in hit[0].message


def test_l2_c2_decertifies_a_non_convex_partition_and_says_r11_refuses_it():
    """L2/C2 rests on L6/C1, so a signed partition loses the criterion too.

    Two layers reach the same conclusion from the same inequality, and they are
    checked separately because a graph reaches L2 before L6 has run.
    """
    r = compile_scheme(_graph(_pou(convex=False), cut_defect_bound=2.6268e-07))
    hit = [d for d in r.decisions.decertifications if d.rule == "C2"]
    assert hit and "not convex" in hit[0].message
    assert [d for d in r.decisions.refusals if d.rule == "R11"]


def test_a_grid_partition_also_carries_the_criterion():
    """The index form a real tiling needs, not only the dense one."""
    idx = {"a": np.arange(0, 5), "b": np.arange(3, 8)}
    wts = {"a": np.array([1.0, 1.0, 1.0, 0.6, 0.3]),
           "b": np.array([0.4, 0.7, 1.0, 1.0, 1.0])}
    bad = {"a": np.array([0, 0, 0, 1, 1], bool), "b": np.array([1, 1, 0, 0, 0], bool)}
    r = compile_scheme(_graph(GridPartitionOfUnity(8, idx, wts, contaminated=bad),
                              cut_defect_bound=2.6268e-07))
    assert [d for d in r.decisions.of_verdict(ADMIT) if d.rule == "C2"]
    assert r.verdict is ADMIT
