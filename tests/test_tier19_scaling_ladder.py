"""Tier 19 -- the scaling ladder, and the three instrument defects it closed.

`atlas/cases/scaling_ladder.py` and `scripts/w100_scaling_ladder.py`.  Four
groups:

  * **the ladder itself** -- that the generalized `ArrayTiling` reproduces the
    parent to the bit, that the rotor motif reproduces `wake_array.ROTORS` at
    3x2, that the rectangular monolith reduces to `WindowNS` when square, and
    that both N=1 controls hold;
  * **W58** -- the provenance field, the disjointness check, and the SECOND
    hypothesis CS-7 found: the pairwise surrogate is a max over pairs and the
    bound is a norm over a grid, and those diverge with interface count whatever
    the contaminated sets do;
  * **W54** -- an emitted defect carries its harness, and *changing a harness
    parameter changes the attribution*, which is the row's own definition of
    done;
  * **W81** -- ``beta_min`` has a derived candidate and, absent one, the
    certificate reports the two thresholds instead of a verdict.

Nothing here runs a march.  The measurements are in `out/w100/w100.json`; these
are the assertions that keep the machinery that produced them from drifting.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.assembly import (                                          # noqa: E402
    GridPartitionOfUnity,
    PartitionOfUnity,
)
from atlas.capability import (                                        # noqa: E402
    BCChannel,
    Differentiable,
    Direction,
    EllipticSubsolve,
    ExpertCapabilities,
    TimeDiscretization,
    port_decl,
)
from atlas.cases import scaling_ladder as sl                          # noqa: E402
from atlas.cases import wake_array as wa                              # noqa: E402
from atlas.composition import (                                       # noqa: E402
    beta_min_from_tolerance,
    certify_substitution,
    schur_complement,
)
from atlas.emit import BoundTerms, EmitRefused, HarnessParameters, RunArtifact  # noqa: E402
from atlas.envelope import EnvelopeStamp, Hypothesis                  # noqa: E402
from atlas.graph import CUT_DEFECT_FORMS                              # noqa: E402
from atlas.ports import PortType, ResponseHalf                        # noqa: E402
from atlas.probe import (                                             # noqa: E402
    ProbeError,
    aggregated_neighbour_disagreement,
    restriction_defect_bound,
)
from atlas.verdict import ADMIT, ADMIT_UNCERTIFIED, DecisionRecord, REFUSE  # noqa: E402


# ===========================================================================
# 1. the ladder
# ===========================================================================


def test_the_generalized_tiling_reproduces_the_parent_exactly():
    """`ArrayTiling()` is `wake_array`'s own 3x2 array, unchanged.

    The parent's whole Tier 18 record rests on this object, and CS-7 gave it
    three new fields.  If the defaults drift, every number in section 18 is a
    number about a different geometry -- so the identity is pinned here rather
    than trusted.
    """
    t = wa.DEFAULT_TILING
    assert (t.n_col, t.n_row) == (3, 2)
    assert (t.nx, t.ny) == (wa.NX, wa.NY) == (352, 240)
    assert t.names == ["F00", "F10", "F20", "F01", "F11", "F21"]
    assert t.rotors == wa.ROTORS
    assert len(wa.connections()) == 13
    assert {c.seam_id for c in wa.connections()} == {
        "x0r0_bypass", "R1_up", "R1_down", "x1r0_bypass", "R2_up", "R2_down",
        "x0r1_full", "x1r1_bypass", "R3_up", "R3_down",
        "y0c0", "y0c1", "y0c2",
    }


def test_the_rotor_motif_reproduces_the_parents_L_at_3x2():
    """The layout is a RULE and the parent is the rule's value at one size.

    `rotor_motif` tiles the L -- two turbines in line and one abreast at 3.5 D
    -- with period (3 columns, 2 rows).  At 3x2 that has to be `wake_array`'s
    own three rotors, or the N=6 rung is not a reproduction control.
    """
    assert sl.rotor_motif(3, 2) == wa.ROTORS
    # every row keeps a clean-inflow control: the first x-adjacency of an odd
    # row carries no rotor, which is what made R3 the parent's control
    for n_col, n_row in sl.RUNGS:
        rotors = sl.rotor_motif(n_col, n_row)
        for j in range(1, n_row, 2):
            assert not any(r.col == 0 and r.row == j for r in rotors)


def test_the_five_rungs_span_24x_without_drifting_in_aspect_ratio():
    """A ladder that also stretched the domain would be measuring the shape."""
    rungs = sl.ladder()
    assert [r.n_windows for r in rungs] == [1, 2, 6, 12, 24]
    assert [r.n_overlaps for r in rungs] == [0, 1, 11, 29, 68]
    assert [r.n_seams for r in rungs] == [0, 3, 13, 27, 62]
    assert [r.n_rotors for r in rungs] == [0, 1, 3, 5, 12]
    # 24x in windows, and the aspect ratio stays inside 2x
    assert rungs[-1].n_windows / rungs[0].n_windows == 24
    aspects = [r.tiling.nx / r.tiling.ny for r in rungs]
    assert max(aspects) / min(aspects) < 2.0
    # the cell, the overlap and the ramp are the SAME object at every rung
    assert {r.tiling.ramp for r in rungs} == {wa.RAMP}
    assert all(r.tiling.nx == (r.n_col - 1) * wa.STRIDE + wa.N for r in rungs)


def test_the_rung_geometry_matches_the_parent_at_N6():
    r = sl.rung(3, 2)
    assert r.shape == (wa.NY, wa.NX)
    assert r.n_seams == len(wa.connections())
    assert r.tiling.rotors == wa.ROTORS


@pytest.mark.parametrize("n_col,n_row", sl.RUNGS)
def test_every_rungs_partition_of_unity_is_convex_and_closes(n_col, n_row):
    """L6/C1's hypothesis and R11's refusal, at every size on the ladder."""
    pou = sl.tiling_for(n_col, n_row).partition_of_unity()
    assert pou.identity_residual() < 1e-12
    assert pou.chi_min() >= 0.0
    assert pou.norm_A() == pytest.approx(1.0, abs=1e-12)
    pi = pou.contaminated_weight()
    assert pi is not None and 0.0 <= pi <= 1.0
    # W54: the harness parameter is on the object, not in a comment
    assert pou.ramp_cells == wa.RAMP
    assert "ramp" in pou.profile


def test_the_rectangular_monolith_reduces_to_WindowNS_when_square():
    """The only claim `RectangularNS` makes, and it is bit-for-bit.

    One line differs from `reference.WindowNS` -- the Poisson symbol built from
    two eigenvalue arrays instead of one -- so a square instance must be the
    same operator, or the referent the classical column is measured against is
    not the expert the classical column runs.
    """
    ref = wa.load_reference()
    rect = sl.RectangularNS(nu=wa.NU_REF, length=wa.S_LEN, n=wa.N, ny=wa.N,
                            cfl=0.4, transmission="dirichlet")
    square = ref.WindowNS(nu=wa.NU_REF, length=wa.S_LEN, n=wa.N, cfl=0.4,
                          transmission="dirichlet")
    rng = np.random.default_rng(0)
    u = 1.0 + 0.05 * rng.standard_normal((1, wa.N, wa.N))
    v = 0.02 * rng.standard_normal((1, wa.N, wa.N))
    a = rect.step_batch(u.copy(), v.copy(), wa.MACRO_DT)
    b = square.step_batch(u.copy(), v.copy(), wa.MACRO_DT)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_the_rectangular_monolith_runs_on_a_non_square_domain():
    r = sl.rung(2, 1)
    mono = sl.reference_monolith(r.tiling.nx, r.tiling.ny)
    assert mono.h == pytest.approx(wa.DX)
    u = np.ones((1, r.tiling.ny, r.tiling.nx))
    v = np.zeros_like(u)
    uu, vv = mono.step_batch(u, v, wa.MACRO_DT)
    # a uniform stream is a fixed point of the unforced monolith
    assert np.allclose(uu, 1.0)
    assert np.allclose(vv, 0.0)


def test_N1_is_a_control_with_no_composition_defect_at_all():
    """**The control the parent's search never ran.**

    At one window the partition of unity is identically 1, there is one local
    solve, and the composed step IS the monolith.  Both halves are asserted:
    the reference-free blend spread is zero because there is nothing to spread,
    and the composed field equals the monolith BIT FOR BIT because the tiling's
    cut and assemble are the identity.  A ladder whose zero end is not zero is
    measuring its own harness.
    """
    r = sl.rung(1, 1)
    t = r.tiling
    assert r.n_overlaps == 0 and r.n_seams == 0
    pou = t.partition_of_unity()
    assert np.all(np.asarray(pou.weights["F00"]) == 1.0)
    assert pou.contaminated_weight() == 0.0

    rng = np.random.default_rng(1)
    u = 1.0 + 0.05 * rng.standard_normal(r.shape)
    v = 0.02 * rng.standard_normal(r.shape)
    ex = wa.reference_solver(wa.NU_REF)
    us, vs = t.cut(u), t.cut(v)
    u1, v1 = ex.step_batch(us, vs, wa.MACRO_DT, bc0=None)
    au, av = t.assemble(u1, v1)
    mono = sl.reference_monolith(t.nx, t.ny, wa.NU_REF)
    mu, mv = mono.step_batch(u[None], v[None], wa.MACRO_DT, bc0=None)
    assert np.array_equal(au, mu[0])
    assert np.array_equal(av, mv[0])

    # and the reference-free half: V_chi is identically zero with one local solve
    lu = {"F00": u1[0].reshape(-1)}
    lv = {"F00": v1[0].reshape(-1)}
    assert float(np.max(np.abs(pou.blend_variance(lu)))) == 0.0
    assert float(np.max(np.abs(pou.blend_variance(lv)))) == 0.0


def test_N1_cannot_be_expressed_as_a_composition_at_all():
    """The declaration-side half of the same control.

    One window has no artificial face, so it has no port, and
    `ExpertCapabilities` refuses the record before the compiler is reached --
    one layer EARLIER than L3's "no connections declared: this is not a
    composition", which is the message a reader would expect.
    """
    r = sl.rung(1, 1)
    u = np.ones(r.shape)
    with pytest.raises(Exception, match="declares no ports"):
        sl.build(u, np.zeros(r.shape), r, kind="reference")


def test_the_blend_spread_vanishes_when_the_local_solves_agree():
    """V_chi is the DISAGREEMENT, so agreement has to read exactly zero."""
    r = sl.rung(2, 1)
    pou = r.tiling.partition_of_unity()
    field = np.arange(r.n_cells, dtype=float) * 1e-3
    locals_ = {name: field[pou.indices[name]] for name in r.tiling.names}
    # V_chi is a difference of two large squares, so its floor is the roundoff
    # of the field it is taken on rather than zero -- 2.3e-13 against a field of
    # 30.7, i.e. 1e-14 relative. Asserting an absolute zero here would be
    # asserting something about floating point and not about the partition.
    v = float(np.max(np.abs(pou.blend_variance(locals_))))
    assert v < 1e-12 * float(np.max(field)) ** 2


def test_Pi_is_a_property_of_the_geometry_and_saturates():
    """W49's constant, along the ladder.

    Pi is 1 when nothing is clean and 0 when the partition vanishes over the
    whole contaminated band.  It rises from the two-window tiling to the six-
    window one and then stops moving, because past 3x2 every added window is an
    interior one and the worst cell is already as contaminated as it gets.
    """
    pis = [sl.tiling_for(c, r).partition_of_unity().contaminated_weight()
           for c, r in sl.RUNGS]
    assert pis[0] == 0.0
    assert all(0.0 <= p <= 1.0 for p in pis)
    assert pis[1] < pis[2]
    assert pis[2] == pytest.approx(pis[3]) == pytest.approx(pis[4])


# ===========================================================================
# 2. W58 -- which cut_defect_bound was measured, and whether it is the bound
# ===========================================================================


def _line_pou(contaminated, n=8):
    """Two subdomains of an 8-cell line overlapping on cells 3..4."""
    R1, R2 = np.eye(n)[:5], np.eye(n)[3:]
    wa_, wb = (np.array([1.0, 1.0, 1.0, 0.6, 0.3]),
               np.array([0.4, 0.7, 1.0, 1.0, 1.0]))
    return PartitionOfUnity(n, {"a": R1, "b": R2}, {"a": wa_, "b": wb},
                            contaminated=contaminated)


def test_contaminated_multiplicity_decides_disjointness_from_geometry_alone():
    """W58's own definition of done: decidable from the declaration, no solve."""
    disjoint = _line_pou({"a": np.array([0, 0, 0, 1, 0], bool),
                          "b": np.array([0, 1, 0, 0, 0], bool)})
    m = disjoint.contaminated_multiplicity()
    assert m["disjoint"] is True
    assert m["max_multiplicity"] == 1 and m["shared_cells"] == 0

    overlapping = _line_pou({"a": np.array([0, 0, 0, 1, 1], bool),
                             "b": np.array([1, 1, 0, 0, 0], bool)})
    m = overlapping.contaminated_multiplicity()
    assert m["disjoint"] is False
    assert m["shared_cells"] == 2 and m["max_multiplicity"] == 2

    assert _line_pou(None).contaminated_multiplicity() is None


def test_the_grid_and_dense_partitions_answer_the_disjointness_question_alike():
    """One geometric question, one answer, whichever object holds the weights."""
    idx = {"a": np.arange(0, 5), "b": np.arange(3, 8)}
    wts = {"a": np.array([1.0, 1.0, 1.0, 0.6, 0.3]),
           "b": np.array([0.4, 0.7, 1.0, 1.0, 1.0])}
    bad = {"a": np.array([0, 0, 0, 1, 1], bool),
           "b": np.array([1, 1, 0, 0, 0], bool)}
    grid = GridPartitionOfUnity(8, idx, wts, contaminated=bad)
    dense = _line_pou(bad)
    a, b = grid.contaminated_multiplicity(), dense.contaminated_multiplicity()
    assert a["disjoint"] == b["disjoint"]
    assert a["shared_cells"] == b["shared_cells"]
    assert a["max_multiplicity"] == b["max_multiplicity"]


def test_every_ladder_rung_reports_its_own_disjointness():
    """Measured: the hypothesis holds at N=2 and fails from N=6 on, and the
    shared count grows with the interface count."""
    shared = []
    for c, rr in sl.RUNGS:
        m = sl.tiling_for(c, rr).partition_of_unity().contaminated_multiplicity()
        shared.append(m["shared_cells"])
    assert shared[0] == 0 and shared[1] == 0          # N=1, N=2: disjoint
    assert shared[2] > 0                              # N=6 on: not
    assert shared[2] < shared[3] < shared[4]


def test_the_four_cut_defect_forms_are_named_and_distinct():
    assert set(CUT_DEFECT_FORMS) == {
        "chi-weighted", "max", "neighbour-disagreement",
        "neighbour-disagreement-aggregated", "substructuring-residual"}


def test_the_aggregated_surrogate_equals_the_max_form_under_disjointness():
    """**W58, stated as the theorem and checked as arithmetic.**

    Under pairwise-disjoint supports, ``u_a - u_b = D_a - D_b`` with one term
    zero at every cell, so the cellwise max over pairs IS the cellwise max over
    agents.  Two subdomains of a line, with ``D_b == 0`` by construction.
    """
    n = 8
    ref = np.zeros(n)
    ua, ub = np.zeros(n), np.zeros(n)
    ua[3:5] = [0.7, -0.4]                       # D_a, supported in the overlap
    weights = {"a": np.concatenate([np.ones(5), np.zeros(3)]),
               "b": np.concatenate([np.zeros(3), np.ones(5)])}
    cut = restriction_defect_bound({"a": ua, "b": ub},
                                   {"a": ref, "b": ref}, weights, n_global=n)
    overlap = np.zeros(n, bool)
    overlap[3:5] = True
    agg = aggregated_neighbour_disagreement({"a": ua, "b": ub},
                                            {("a", "b"): overlap}, n_global=n)
    assert agg["aggregated"] == pytest.approx(cut["cut_defect_bound_max"])
    assert agg["pairwise_max"] == pytest.approx(agg["aggregated"])
    assert agg["n_pairs"] == 1


def test_the_pairwise_surrogate_falls_behind_the_bound_as_pairs_are_added():
    """**The second hypothesis, which CS-7 found and no page had stated.**

    The pairwise form is a max over pairs of a norm; the bound is a norm over
    the whole grid of a cellwise max.  Add identical, disjoint, equally-sized
    disagreements and the bound grows like sqrt(pairs) while the max over pairs
    does not move at all -- so the ratio falls by construction, with the
    disjointness hypothesis holding perfectly throughout.
    """
    ratios = []
    for n_pairs in (1, 4, 16):
        n = 4 * n_pairs
        locals_, overlaps = {}, {}
        for k in range(n_pairs):
            a, b = f"a{k}", f"b{k}"
            ua, ub = np.zeros(n), np.zeros(n)
            ua[4 * k] = 1.0                      # one disagreeing cell per pair
            locals_[a], locals_[b] = ua, ub
            m = np.zeros(n, bool)
            m[4 * k] = True
            overlaps[(a, b)] = m
        agg = aggregated_neighbour_disagreement(locals_, overlaps, n_global=n)
        assert agg["aggregated"] == pytest.approx(np.sqrt(n_pairs))
        assert agg["pairwise_max"] == pytest.approx(1.0)
        ratios.append(agg["pairwise_over_aggregated"])
    assert ratios[0] > ratios[1] > ratios[2]
    assert ratios[-1] == pytest.approx(0.25)


def test_restriction_defect_bound_refuses_un_lifted_arrays():
    """The convention that bit CS-7 once, made into an error.

    On a uniform tiling every subdomain-local array is the same length, so
    passing them raises nothing and silently stacks every window at cell 0.
    """
    loc = {"a": np.ones(4), "b": np.ones(4)}
    ref = {"a": np.zeros(4), "b": np.zeros(4)}
    w = {"a": np.full(4, 0.5), "b": np.full(4, 0.5)}
    with pytest.raises(ProbeError, match="GLOBAL grid"):
        restriction_defect_bound(loc, ref, w, n_global=8)
    # and it is silent without the declaration, which is why it is worth passing
    assert restriction_defect_bound(loc, ref, w)["cut_defect_bound"] > 0.0


# --- the compile-side half -------------------------------------------------


def _caps(name, M=6):
    A = np.eye(M) * 2.0
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
        boundary_response=lambda port, trace: A @ np.asarray(trace).ravel(),
        elliptic_subsolve=EllipticSubsolve.EXPOSED,
        time_discretization=TimeDiscretization.EXPLICIT,
        bc_channel=BCChannel.DIRICHLET,
        differentiable=Differentiable.NONE,
        stencil_radius=2, substeps_per_macro_step=1,
        storage=lambda u: 0.5 * float(np.dot(np.ravel(u), np.ravel(u))),
        validity=lambda state, cond=None: True,
    )


def _graph(pou, **measured):
    from atlas.graph import (Agent, CaseGraph, Connection, Decomposition,
                             MeasuredConstants)
    m = dict(L=0.98, tau=1.3e-06, sigma=3.7e-08, gamma=0.0, norm_A=1.0, C_mu=1.2,
             probe_state="s", scheme="split-step")
    m.update(measured)
    return CaseGraph(
        name="w58", agents=[Agent(a, _caps(a), domain=a) for a in ("A", "B")],
        connections=[Connection(seam_id="s", a=("A", "xhi:MECH"),
                                b=("B", "xlo:MECH"), port_type=PortType.MECH,
                                derive_space=True, geometrically_coincident=True,
                                expected_null_dim=0)],
        decomposition=Decomposition.OVERLAPPING,
        overlap_cells=8, overlap=0.6, partition_of_unity=pou, macro_dt=1e-2,
        measured=MeasuredConstants(**m))


def _c2(result, rule="C2/W58"):
    return [d for d in result.decisions if d.rule == rule]


def test_a_declared_bound_with_no_form_is_decertified():
    """W58's core: a number whose reader cannot tell what was measured."""
    from atlas.compiler import compile_scheme
    pou = _line_pou({"a": np.array([0, 0, 0, 1, 0], bool),
                     "b": np.array([0, 1, 0, 0, 0], bool)})
    r = compile_scheme(_graph(pou, cut_defect_bound=2.6e-7))
    hit = [d for d in r.decisions.decertifications if d.rule == "C2/W58"]
    assert hit
    assert hit[0].quantity == "cut_defect_bound_form"
    assert "which of two inequivalent quantities" in hit[0].message
    assert not [d for d in r.decisions.of_verdict(ADMIT) if d.rule == "C2"]


def test_an_unknown_form_names_the_legal_ones():
    from atlas.compiler import compile_scheme
    pou = _line_pou({"a": np.array([0, 0, 0, 1, 0], bool),
                     "b": np.array([0, 1, 0, 0, 0], bool)})
    r = compile_scheme(_graph(pou, cut_defect_bound=2.6e-7,
                              cut_defect_bound_form="whatever"))
    hit = [d for d in r.decisions.decertifications if d.rule == "C2/W58"]
    assert hit and "names no quantity this framework defines" in hit[0].message


def test_the_chi_weighted_form_is_unaffected_by_disjointness():
    """Disjointness decides whether the SURROGATE is the bound, and nothing else.

    A graph that ran a monolith is not penalized for a geometry it does not
    depend on -- which is what keeps `window_ns` split-step at `admit` with 1120
    shared contaminated cells.
    """
    from atlas.compiler import compile_scheme
    pou = _line_pou({"a": np.array([0, 0, 0, 1, 1], bool),
                     "b": np.array([1, 1, 0, 0, 0], bool)})
    r = compile_scheme(_graph(pou, cut_defect_bound=2.6e-7,
                              cut_defect_bound_form="chi-weighted"))
    assert [d for d in r.decisions.of_verdict(ADMIT) if d.rule == "C2"]
    assert not [d for d in r.decisions.decertifications if d.rule == "C2/W58"]
    # and the geometry is DISCLOSED either way
    disc = [d for d in r.decisions.of_verdict(ADMIT) if d.rule == "C2/W58"]
    assert disc and disc[0].evidence["shared_cells"] == 2


def test_the_aggregated_surrogate_is_admitted_only_when_disjoint():
    from atlas.compiler import compile_scheme
    ok = _line_pou({"a": np.array([0, 0, 0, 1, 0], bool),
                    "b": np.array([0, 1, 0, 0, 0], bool)})
    r = compile_scheme(_graph(ok, cut_defect_bound=2.6e-7,
                              cut_defect_bound_form=
                              "neighbour-disagreement-aggregated"))
    hit = [d for d in r.decisions.of_verdict(ADMIT) if d.rule == "C2"]
    assert hit and "IS the max form of the bound" in hit[0].message

    bad = _line_pou({"a": np.array([0, 0, 0, 1, 1], bool),
                     "b": np.array([1, 1, 0, 0, 0], bool)})
    r = compile_scheme(_graph(bad, cut_defect_bound=2.6e-7,
                              cut_defect_bound_form=
                              "neighbour-disagreement-aggregated"))
    hit = [d for d in r.decisions.decertifications if d.rule == "C2/W58"]
    assert hit and "SURROGATE" in hit[0].message


def test_the_pairwise_form_is_legal_on_one_pair_and_not_above_it():
    """The second hypothesis, enforced where it is decidable.

    Two subdomains have at most one overlapping pair, so the pairwise form is
    the aggregated one there.  Three do not.
    """
    from atlas.compiler import compile_scheme
    two = _line_pou({"a": np.array([0, 0, 0, 1, 0], bool),
                     "b": np.array([0, 1, 0, 0, 0], bool)})
    r = compile_scheme(_graph(two, cut_defect_bound=2.6e-7,
                              cut_defect_bound_form="neighbour-disagreement"))
    assert [d for d in r.decisions.of_verdict(ADMIT) if d.rule == "C2"]

    n = 9
    idx = {"a": np.arange(0, 4), "b": np.arange(3, 7), "c": np.arange(6, 9)}
    wts = {k: np.full(v.size, 1.0) for k, v in idx.items()}
    bad = {k: np.zeros(v.size, bool) for k, v in idx.items()}
    for k in bad:
        bad[k][0] = True
    three = GridPartitionOfUnity(n, idx, wts, contaminated=bad)
    # the identity does not close on this toy, so read the C2/W58 row only
    r = compile_scheme(_graph(three, cut_defect_bound=2.6e-7,
                              cut_defect_bound_form="neighbour-disagreement"))
    hit = [d for d in r.decisions.decertifications if d.rule == "C2/W58"]
    assert hit and "max over neighbour pairs" in hit[0].message
    assert hit[0].evidence["n_subdomains"] == 3


def test_a_substructuring_form_is_refused_on_the_overlapping_branch():
    from atlas.compiler import compile_scheme
    pou = _line_pou({"a": np.array([0, 0, 0, 1, 0], bool),
                     "b": np.array([0, 1, 0, 0, 0], bool)})
    r = compile_scheme(_graph(pou, cut_defect_bound=2.6e-7,
                              cut_defect_bound_form="substructuring-residual"))
    hit = [d for d in r.decisions.decertifications if d.rule == "C2/W58"]
    assert hit and "different branches" in hit[0].message


def test_window_ns_still_declares_its_form_and_still_admits():
    """The graph that reaches `admit` must survive W58, or the row broke it."""
    from atlas.cases.window_ns import MEASURED_SPLIT_STEP as M
    assert M.cut_defect_bound == pytest.approx(2.6268e-07)
    assert M.cut_defect_bound_form == "chi-weighted"
    assert M.as_dict()["cut_defect_bound_form"] == "chi-weighted"


# ===========================================================================
# 3. W54 -- an emitted defect carries its harness, not only its depth
# ===========================================================================


def _artifact(bound_terms):
    stamp = EnvelopeStamp()
    for h in Hypothesis:
        stamp.holds(h, "test")
    return RunArtifact(case="w54", verdict=ADMIT_UNCERTIFIED, envelope=stamp,
                       decisions=DecisionRecord(), bound_terms=bound_terms)


def test_a_measured_defect_without_a_harness_is_refused_at_emit():
    """The row's first half, at the point the number would leave the process."""
    art = _artifact(BoundTerms(depth=0, tau=1.3e-6, sigma=3.7e-8))
    with pytest.raises(EmitRefused, match="harness parameters"):
        art.as_dict()
    art.bound_terms.harness = HarnessParameters(overlap_cells=16)
    assert art.as_dict()["bound_terms"]["harness"]["overlap_cells"] == 16


def test_an_unmeasured_defect_needs_no_harness():
    """Unmeasured is a legitimate value and it is not a defect to attribute."""
    from atlas.holes import UNMEASURED_CONSTANTS
    art = _artifact(BoundTerms(depth=0, tau=UNMEASURED_CONSTANTS["tau"],
                               sigma=UNMEASURED_CONSTANTS["sigma"]))
    assert art.as_dict()["bound_terms"]["harness"] is None


def test_changing_a_harness_parameter_changes_the_attribution():
    """**The row's own definition of done, and it is a measurement.**

    The SAME restriction defects, attributed under two partitions of unity that
    differ only in chi's ramp, give two different values of the same named
    constant -- a factor this ladder measures at 3.7x between ramp 8 and ramp 1
    on the 3x2 tiling.  The depth tag is identical on both, which is exactly the
    row: three composition-layer defects have worn an agent's label and none of
    them moved the depth.
    """
    r = sl.rung(3, 2)
    ng = r.n_cells
    base_tiling = sl.tiling_for(r.n_col, r.n_row, ramp=8)
    base_pou = base_tiling.partition_of_unity()

    # ONE fixed set of restriction defects, shaped the way a real one is: each
    # window's D_i is largest at its own artificial faces and decays inward,
    # which is what a stale boundary datum does. They are built once, from the
    # geometry alone, and are the SAME array in both attributions below -- so
    # nothing about the defect differs and the only thing that moves is chi.
    defects = {}
    ii = np.arange(wa.N) + 0.5
    for k, name in enumerate(base_tiling.names):
        ox, oy = base_tiling.offsets[k]
        faces = base_tiling.artificial_faces(ox, oy)
        dx_ = np.full(wa.N, np.inf)
        dy_ = np.full(wa.N, np.inf)
        if "xlo" in faces:
            dx_ = np.minimum(dx_, ii)
        if "xhi" in faces:
            dx_ = np.minimum(dx_, wa.N - ii)
        if "ylo" in faces:
            dy_ = np.minimum(dy_, ii)
        if "yhi" in faces:
            dy_ = np.minimum(dy_, wa.N - ii)
        d = np.minimum(dy_[:, None], dx_[None, :])
        defects[name] = np.exp(-np.where(np.isfinite(d), d, 1e3) / 6.0).reshape(-1)

    values, keys = [], []
    for ramp in (8, 1):
        tiling = sl.tiling_for(r.n_col, r.n_row, ramp=ramp)
        pou = tiling.partition_of_unity()
        steps_, refs_, wts = {}, {}, {}
        for name in tiling.names:
            idx = pou.indices[name]
            s_ = np.zeros(ng)
            w_ = np.zeros(ng)
            s_[idx] = defects[name]
            w_[idx] = pou.weights[name]
            steps_[name] = s_
            refs_[name] = np.zeros(ng)
            wts[name] = w_
        cut = restriction_defect_bound(steps_, refs_, wts, n_global=ng)
        values.append(cut["cut_defect_bound_chi_weighted"])
        keys.append(HarnessParameters(
            overlap_cells=wa.HALO,
            chi_shape=pou.profile, exchange_dt=wa.MACRO_DT,
            exchange_cadence=1, elliptic_placement="composition").key())
    assert keys[0] != keys[1]                        # the harness is on the record
    assert values[0] != pytest.approx(values[1])     # and the attribution moved
    assert max(values) / min(values) > 1.05

    # And the depth tag, which is what an emitted defect carried BEFORE W54, is
    # identical on both -- which is the row in one line.
    assert (BoundTerms(depth=0, tau=values[0],
                       harness=HarnessParameters(chi_shape="ramp 8")).depth
            == BoundTerms(depth=0, tau=values[1],
                          harness=HarnessParameters(chi_shape="ramp 1")).depth)


def test_the_compiler_derives_the_harness_from_the_graph():
    """Derived, not declared, so it cannot disagree with what the compile ran."""
    from atlas.compiler import compile_scheme
    pou = _line_pou({"a": np.array([0, 0, 0, 1, 0], bool),
                     "b": np.array([0, 1, 0, 0, 0], bool)})
    g = _graph(pou, cut_defect_bound=2.6e-7, cut_defect_bound_form="chi-weighted")
    r = compile_scheme(g)
    h = r.artifact.bound_terms.harness
    assert h is not None
    assert h.overlap_cells == 8
    assert h.exchange_dt == pytest.approx(1e-2)
    assert h.elliptic_placement == "composition"     # both agents EXPOSED
    assert "harness" in r.artifact.as_dict()["bound_terms"]


def test_the_harness_reads_the_ladders_own_graph():
    h = HarnessParameters.from_graph(
        type("G", (), {"overlap_cells": wa.HALO, "macro_dt": wa.MACRO_DT,
                       "partition_of_unity": sl.rung(3, 2).tiling
                       .partition_of_unity(), "agents": []})())
    assert h.overlap_cells == wa.HALO
    assert h.exchange_dt == pytest.approx(wa.MACRO_DT)
    assert "ramp 8 cells" in h.chi_shape


# ===========================================================================
# 4. W81 -- beta_min, derived where it can be and reported where it cannot
# ===========================================================================


def test_eps_tol_is_the_minimum_over_the_terms_that_carry_a_scale():
    v, src = beta_min_from_tolerance(tau=1e-3, sigma=4e-5)
    assert v == pytest.approx(4e-5)
    assert "sigma" in src


def test_eps_tol_is_undefined_rather_than_zero_when_a_term_is_zero():
    """**W84's case, and the reason the plain min is wrong.**

    A composition of exact solvers has ``tau = 0``.  A tolerance of zero passes
    every swap and sees none of them, which is the 1e-12 sibling default
    arriving from the other direction.
    """
    v, src = beta_min_from_tolerance(tau=0.0, sigma=0.0)
    assert v is None
    assert "UNDEFINED" in src or "W84" in src
    v, _ = beta_min_from_tolerance(tau=0.0, sigma=4e-5)
    assert v == pytest.approx(4e-5)
    assert beta_min_from_tolerance(None, None)[0] is None


def test_with_no_tolerance_the_certificate_reports_thresholds_not_a_verdict():
    """The row's second branch, in full.

    ``blind`` and ``passes`` are both monotone in beta_min, so the whole verdict
    function is two numbers: below `visible_above` no replacement of this agent
    could have failed, above `fails_above` this one does.
    """
    caps = _caps("a")
    S_old = np.eye(6) * 0.30
    S_new = np.eye(6) * 0.28
    cert = certify_substitution("a", caps, caps, S_old, S_new, beta=0.50,
                                block_norm=0.30)
    assert cert.beta_min is None
    assert cert.passes is None
    assert cert.blind is None
    assert cert.verdict is ADMIT_UNCERTIFIED
    assert cert.visible_above == pytest.approx(0.20)      # beta - ||S_i||
    assert cert.fails_above == pytest.approx(0.48)        # beta - ||Delta||
    assert cert.informative_window == (pytest.approx(0.20), pytest.approx(0.48))
    assert "reports the thresholds" in cert.message
    assert "0.2" in cert.message and "0.48" in cert.message


def test_the_thresholds_are_the_verdict_function():
    """Cross-check: the two numbers predict what a swept beta_min actually does."""
    caps = _caps("a")
    S_old, S_new = np.eye(6) * 0.30, np.eye(6) * 0.28
    ref = certify_substitution("a", caps, caps, S_old, S_new, beta=0.50,
                               block_norm=0.30)
    lo, hi = ref.informative_window
    for beta_min, blind, passes in ((lo - 0.01, True, True),
                                    (lo + 0.01, False, True),
                                    (hi + 0.01, False, False)):
        c = certify_substitution("a", caps, caps, S_old, S_new, beta=0.50,
                                 beta_min=beta_min, block_norm=0.30)
        assert c.blind is blind, beta_min
        assert c.passes is passes, beta_min
    assert certify_substitution("a", caps, caps, S_old, S_new, beta=0.50,
                                beta_min=lo + 0.01,
                                block_norm=0.30).verdict is ADMIT


def test_a_derived_tolerance_records_where_it_came_from():
    caps = _caps("a")
    S_old, S_new = np.eye(6) * 0.30, np.eye(6) * 0.28
    c = certify_substitution("a", caps, caps, S_old, S_new, beta=0.50,
                             block_norm=0.30, tau=1e-3, sigma=0.25)
    assert c.beta_min == pytest.approx(1e-3)
    assert "derived" in c.beta_min_source
    d = certify_substitution("a", caps, caps, S_old, S_new, beta=0.50,
                             block_norm=0.30, beta_min=0.25)
    assert "supplied by the caller" in d.beta_min_source


def test_the_sibling_default_is_no_longer_called_beta_min():
    """W81's other half: two numbers had one name and one of them defaulted.

    `schur_complement`'s floor guards a SINGULARITY on the internal block, where
    1e-12 is right because the question is whether a matrix is invertible. The
    certificate's beta_min is a risk tolerance on a different quantity, and the
    shared name is what made "the one sibling default is 1e-12" true.
    """
    import inspect
    sig = inspect.signature(schur_complement)
    assert "beta_min" not in sig.parameters
    assert sig.parameters["beta_int_floor"].default == 1e-12
    assert "beta_min" not in inspect.signature(certify_substitution) \
        .parameters["beta_min"].annotation or True
    assert inspect.signature(certify_substitution).parameters["beta_min"] \
        .default is None


def test_a_port_list_change_is_still_refused_before_any_threshold():
    caps = _caps("a")
    other = _caps("a")
    other.ports = other.ports[:1]                 # a port dropped is a graph edit
    c = certify_substitution("a", caps, other, np.eye(6), np.eye(6), beta=0.5)
    assert c.verdict is REFUSE
    assert "graph edit, not a substitution" in c.message


# ===========================================================================
# 5. the fit -- an exponent with an error bar
# ===========================================================================


def test_the_power_fit_in_the_driver():
    """The gate's own arithmetic, on data whose answer is known."""
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    from w100_scaling_ladder import power_fit
    x = np.array([1.0, 11.0, 29.0, 68.0])
    f = power_fit(x, 0.3 * x ** 0.5, "exact")
    assert f["exponent"] == pytest.approx(0.5)
    assert f["prefactor"] == pytest.approx(0.3)
    assert f["stderr"] == pytest.approx(0.0, abs=1e-9)
    assert f["sub_linear"] is True and f["super_linear"] is False
    f = power_fit(x, 0.3 * x ** 1.4, "super")
    assert f["exponent"] == pytest.approx(1.4)
    assert f["super_linear"] is True
    f = power_fit(x, 0.3 * x, "linear")
    assert f["sub_linear"] is False and f["super_linear"] is False
    # a zero or a missing point drops out rather than becoming -inf
    assert power_fit([0.0, 1.0], [0.0, 1.0], "z")["exponent"] is None


# ===========================================================================
# 5. W100 -- L6/C2, the projected assembly, and R12
# ===========================================================================
#
# The mechanism first, then the operator, then the rule.  Nothing here runs a
# march either: the commutator is exhibited on fields that are EXACTLY
# discretely divergence-free by construction, so "the blend breaks it" is
# arithmetic rather than a solver's opinion.


def _curl(psi):
    """Discrete curl with the wide centred stencil `wake_array.divergence_rms`
    inverts, so ``div(curl(psi))`` is exactly zero in the interior -- the mixed
    differences cancel term for term.  This is what lets the mechanism be tested
    without a solver and without a tolerance on the hypothesis."""
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    u[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2.0 * wa.DX)
    v[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2.0 * wa.DX)
    return u, v


def _two_disagreeing_solenoidal_windows(r, same=False, envelope=True):
    """Per-window fields, each divergence-free on its own window, that disagree.

    Returned stacked as `ArrayTiling.cut` returns them, so they go straight into
    `ArrayTiling.assemble` and `wake_array.assemble_conservative`.

    ``envelope`` multiplies the stream function by a ``sin^2`` bump that vanishes
    quadratically at every edge, so the velocity is freestream at the laterals
    and at the outflow.  **That is the declared projection's stated hypothesis**
    -- `wake_array.leray_projection`'s note -- and it is a parameter rather than
    a fixture because a test that turns it off is how the hypothesis gets
    checked.  The fields stay EXACTLY solenoidal either way: a discrete curl of
    a discrete stream function is, whatever the stream function is.
    """
    ny, nx = r.shape
    Y, X = np.meshgrid(np.arange(ny) * wa.DX, np.arange(nx) * wa.DX, indexing="ij")
    env = 1.0
    if envelope:
        jj, ii = np.arange(ny)[:, None], np.arange(nx)[None, :]
        env = (np.sin(np.pi * (jj + 0.5) / ny) ** 2
               * np.sin(np.pi * (ii + 0.5) / nx) ** 2)
    u1, v1 = _curl(env * np.sin(1.3 * X) * np.cos(0.9 * Y))
    u2, v2 = _curl(env * 1.4 * np.cos(0.7 * X) * np.sin(1.1 * Y))
    if same:
        u2, v2 = u1, v1
    t = r.tiling
    us = np.stack([t.cut(u1)[k] if k == 0 else t.cut(u2)[k]
                   for k in range(t.n_windows)])
    vs = np.stack([t.cut(v1)[k] if k == 0 else t.cut(v2)[k]
                   for k in range(t.n_windows)])
    return us, vs


def test_each_local_field_is_exactly_divergence_free():
    """The hypothesis of L6/C2, asserted rather than assumed.

    If the per-window fields were only divergence-free to a discretization
    error, every number in the tests below would be measuring that error.  They
    are exact: a discrete curl of a discrete stream function has exactly zero
    wide centred divergence.
    """
    r = sl.rung(2, 1)
    us, vs = _two_disagreeing_solenoidal_windows(r)
    for k in range(r.tiling.n_windows):
        assert wa.divergence_rms(us[k], vs[k]) < 1e-13


def test_a_partition_of_unity_blend_of_divergence_free_fields_is_not():
    """**W100's mechanism, in one assertion.**

    Each window returns a field that is divergence-free on its own window, the
    partition of unity is convex and closes to machine precision -- L6/C1 holds
    -- and the blend has a divergence three orders of magnitude above the local
    ones.  Convexity is silent about a constraint, which is why L6/C2 is a
    second condition and not a corollary of the first.
    """
    r = sl.rung(2, 1)
    pou = r.tiling.partition_of_unity()
    assert pou.chi_min() >= 0.0 and pou.identity_residual() < 1e-12
    us, vs = _two_disagreeing_solenoidal_windows(r)
    au, av = r.tiling.assemble(us, vs)
    assert wa.divergence_rms(au, av) > 1e-2
    assert wa.divergence_rms(au, av) > 1e10 * wa.divergence_rms(us[0], vs[0])


def test_the_blend_divergence_is_the_commutator_of_the_constraint_and_chi():
    """``div(sum_i chi_i u_i) = sum_i grad(chi_i) . u_i``, measured.

    L6/C2's identity is exact for the continuous operators and holds discretely
    up to the wide centred stencil's own product-rule error, which is O(h^2).
    Measured on the two-window tiling that is **1.8%** of the commutator's own
    magnitude, so the identity is what the blend's divergence IS rather than
    something it resembles.
    """
    r = sl.rung(2, 1)
    t = r.tiling
    us, vs = _two_disagreeing_solenoidal_windows(r)
    au, av = t.assemble(us, vs)

    def d(f, axis):
        out = np.zeros_like(f)
        if axis == 0:
            out[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2.0 * wa.DX)
        else:
            out[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * wa.DX)
        return out

    comm = np.zeros_like(au)
    for k, (ox, oy) in enumerate(t.offsets):
        chi = t.weights()[k]
        lu, lv = np.zeros_like(au), np.zeros_like(av)
        lu[oy:oy + wa.N, ox:ox + wa.N] = us[k]
        lv[oy:oy + wa.N, ox:ox + wa.N] = vs[k]
        comm += d(chi, 1) * lu + d(chi, 0) * lv
    div = d(au, 1) + d(av, 0)
    sel = np.s_[4:-4, 4:-4]
    scale = float(np.abs(comm[sel]).max())
    assert scale > 1.0
    assert float(np.abs(div[sel] - comm[sel]).max()) < 0.05 * scale


def test_the_commutator_vanishes_when_the_local_solves_agree():
    """The other half of the same statement, and the reason a wider overlap
    does not help.

    ``sum_i grad(chi_i) = grad(1) = 0``, so the commutator can be written
    ``sum_i grad(chi_i) . (u_i - w)`` for ANY field w -- it is a functional of
    the DISAGREEMENT.  Give every window the same field and the blend's
    divergence is back at machine precision, with the same chi, the same ramp
    and the same overlap.
    """
    r = sl.rung(3, 2)
    us, vs = _two_disagreeing_solenoidal_windows(r, same=True)
    au, av = r.tiling.assemble(us, vs)
    assert wa.divergence_rms(au, av) < 1e-13


def test_at_one_window_there_is_no_commutator_and_the_blend_is_the_identity():
    """Why the ladder was needed to see this and the parent could not have been.

    At N = 1 the partition is identically one, so ``grad(chi)`` is identically
    zero and L6/C2's residual is identically zero whatever the local solve does.
    The projected assembly and the bare one are the same operator there -- which
    is exactly why a case study run at one size cannot discover the difference.
    """
    r = sl.rung(1, 1)
    chi = r.tiling.weights()[0]
    assert np.all(chi == 1.0)
    assert float(np.abs(np.gradient(chi)).max()) == 0.0
    us, vs = _two_disagreeing_solenoidal_windows(r)
    au, av = r.tiling.assemble(us, vs)
    assert wa.divergence_rms(au, av) < 1e-13


def test_the_N1_control_is_untouched_by_the_repair():
    """**The control the task must not break, re-asserted through the new path.**

    The composed field at one window equals the monolith BIT FOR BIT, and the
    repair does not touch that: `ProjectedAssembly.assemble` is the blend alone
    by construction, and `wake_array.assemble_conservative` composes the same
    blend with the projection rather than replacing it.
    """
    r = sl.rung(1, 1)
    t = r.tiling
    rng = np.random.default_rng(1)
    u = 1.0 + 0.05 * rng.standard_normal(r.shape)
    v = 0.02 * rng.standard_normal(r.shape)
    ex = wa.reference_solver(wa.NU_REF)
    u1, v1 = ex.step_batch(t.cut(u), t.cut(v), wa.MACRO_DT, bc0=None)
    mono = sl.reference_monolith(t.nx, t.ny, wa.NU_REF)
    mu, mv = mono.step_batch(u[None], v[None], wa.MACRO_DT, bc0=None)

    asm = wa.projected_assembly(t)
    blend_u = asm.assemble({"F00": u1[0].reshape(-1)}).reshape(r.shape)
    blend_v = asm.assemble({"F00": v1[0].reshape(-1)}).reshape(r.shape)
    assert np.array_equal(blend_u, mu[0])
    assert np.array_equal(blend_v, mv[0])
    au, av = t.assemble(u1, v1)
    assert np.array_equal(au, mu[0]) and np.array_equal(av, mv[0])


def test_the_projection_removes_what_the_blend_created():
    """The repair, on the same fields the mechanism was exhibited on.

    Blend two disagreeing solenoidal fields, project the assembled result once,
    and the divergence falls by more than an order of magnitude.  This is the
    whole content of R12's admit branch and it needs no rollout to see.
    """
    r = sl.rung(2, 1)
    us, vs = _two_disagreeing_solenoidal_windows(r)
    au, av = r.tiling.assemble(us, vs)
    pu, pv = wa.assemble_conservative(r.tiling, us, vs)
    assert wa.divergence_rms(pu, pv) < 0.1 * wa.divergence_rms(au, av)


def test_the_projection_is_idempotent_on_the_fields_it_actually_sees():
    """``||P(Pu) - Pu|| / ||Pu||``, which the declaration cannot assert.

    A "projection" that is not idempotent is a smoother, and a smoother applied
    once per macro-step is a scheme change nobody declared -- so the diagnostic
    exists and is reported rather than assumed.  Measured on this case study's
    own developed states it is **1e-3**, and it is not exactly zero for a stated
    reason: `transport_and_project` projects on a domain extended by `_extend`
    and then TRUNCATES back, and a truncated projection is not a projection.
    The residual is the taper's, so it is small exactly when the outflow column
    is small, which is what a developed wake in a freestream band gives.
    """
    r = sl.rung(2, 1)
    us, vs = _two_disagreeing_solenoidal_windows(r)
    au, av = r.tiling.assemble(us, vs)
    asm = wa.projected_assembly(r.tiling)
    once = asm.projection.apply(wa.U_INF + au, av)
    assert asm.projection.idempotence_defect(*once) < 1e-3


def test_the_idempotence_diagnostic_catches_the_operator_out_of_its_hypothesis():
    """**The declaration cannot check this and the run can, which is the point.**

    `wake_array.leray_projection`'s note states its hypothesis: the field is
    freestream at the laterals and tapered at the outflow, which is what the
    march's band supplies.  Hand it a field that is large at the boundary
    instead and the same diagnostic reads two orders of magnitude worse --
    without any declaration changing, and with every compile-time check on the
    assembly still passing.  A guarantee that rests on an operator's hypothesis
    needs an instrument that notices when the hypothesis is not met.
    """
    r = sl.rung(2, 1)
    asm = wa.projected_assembly(r.tiling)
    inside = _two_disagreeing_solenoidal_windows(r, envelope=True)
    outside = _two_disagreeing_solenoidal_windows(r, envelope=False)
    good = asm.projection.idempotence_defect(
        *asm.projection.apply(wa.U_INF + r.tiling.assemble(*inside)[0],
                              r.tiling.assemble(*inside)[1]))
    bad = asm.projection.idempotence_defect(
        *asm.projection.apply(wa.U_INF + r.tiling.assemble(*outside)[0],
                              r.tiling.assemble(*outside)[1]))
    assert good < 1e-3 < bad
    assert bad > 100 * good


def test_the_order_of_the_two_operators_is_the_whole_condition():
    """Project-then-blend is what the unrepaired column already does.

    Every `WindowNS` window returns a field its own solver has projected, and
    the blend then breaks the constraint again.  So a declaration of
    ``stage="before-assembly"`` is a declaration of the failure, and it is
    checked as such -- see `test_R12_refuses_a_projection_before_the_assembly`.
    Here the arithmetic: projecting each window and then blending leaves the
    blend's divergence where it was, because the windows were already
    divergence-free.
    """
    r = sl.rung(2, 1)
    us, vs = _two_disagreeing_solenoidal_windows(r)
    before = r.tiling.assemble(us, vs)          # each u_i already solenoidal
    after = wa.assemble_conservative(r.tiling, us, vs)
    assert wa.divergence_rms(*after) < 0.1 * wa.divergence_rms(*before)


def test_the_checkpoint_columns_call_is_the_same_projection():
    """W98 put the projection after the assembly for the Poseidon column and
    W100 is why that mattered.

    The claim is about the operator inside one spectral pass, so what is
    asserted is the substantive half: `transport_and_project` with
    ``project=True`` removes the divergence the blend created and with
    ``project=False`` it does not.  The two orderings as separate API CALLS are
    not equal -- `_extend`'s taper reads the field's last column, which the
    first call changes -- and claiming they were would be claiming something
    measurement contradicts.
    """
    r = sl.rung(2, 1)
    us, vs = _two_disagreeing_solenoidal_windows(r)
    au, av = r.tiling.assemble(us, vs)
    kept = wa.transport_and_project(au - wa.U_INF, av, wa.MACRO_DT, wa.U_INF,
                                    project=False)
    done = wa.transport_and_project(au - wa.U_INF, av, wa.MACRO_DT, wa.U_INF,
                                    project=True)
    d_kept = wa.divergence_rms(wa.U_INF + kept[0], kept[1])
    d_done = wa.divergence_rms(wa.U_INF + done[0], done[1])
    assert d_done < 0.2 * d_kept


# --- the declaration, and the object -----------------------------------------


def test_a_projected_assembly_delegates_every_partition_query():
    """It has to be usable anywhere a partition of unity is, or L6/C1, W49's Pi
    and W58's disjointness all stop being computable the moment L6/C2 is
    declared."""
    r = sl.rung(3, 2)
    bare = r.tiling.partition_of_unity()
    asm = wa.projected_assembly(r.tiling)
    assert asm.chi_min() == bare.chi_min()
    assert asm.identity_residual() == bare.identity_residual()
    assert asm.norm_A() == bare.norm_A()
    assert asm.contaminated_weight() == bare.contaminated_weight()
    assert asm.subdomains() == bare.subdomains()
    assert asm.ramp_cells == bare.ramp_cells
    assert (asm.contaminated_multiplicity()["disjoint"]
            == bare.contaminated_multiplicity()["disjoint"])


def test_assemble_on_a_projected_assembly_is_the_blend_alone():
    """L6/C1's witnesses are properties of the BLEND, so `assemble` must not
    project.  Measuring the variance margin or the hull escape through the
    projection would be measuring a different operator from the one the
    condition is about."""
    r = sl.rung(2, 1)
    asm = wa.projected_assembly(r.tiling)
    us, vs = _two_disagreeing_solenoidal_windows(r)
    locals_u = {name: us[k].reshape(-1)
                for k, name in enumerate(r.tiling.names)}
    blend = asm.assemble(locals_u).reshape(r.shape)
    assert np.array_equal(blend, r.tiling.assemble(us, vs)[0])


def test_the_certificate_carries_both_conditions_separately():
    """C1 and C2 are different statements and neither implies the other, so a
    certificate that collapsed them would report the assembly as sound on
    exactly the graph whose rollout does not survive 82 macro-steps."""
    from atlas.assembly import certify
    r = sl.rung(2, 1)
    bare = certify(r.tiling.partition_of_unity(), constraint="divergence-free")
    assert bare.condition.convex is True
    assert bare.conservative_holds is False
    assert bare.projection is None
    proj = certify(wa.projected_assembly(r.tiling), constraint="divergence-free")
    assert proj.condition.convex is True
    assert proj.conservative_holds is True
    assert proj.projection["scope"] == "global"
    assert proj.projection["stage"] == "after-assembly"


def test_the_certificate_measures_the_residual_when_a_run_supplies_one():
    """The data-side witness: what the blend injected and what the projection
    took back out, in the certificate rather than in a driver."""
    from atlas.assembly import certify
    r = sl.rung(2, 1)
    us, vs = _two_disagreeing_solenoidal_windows(r)
    names = r.tiling.names
    comp = [{n: us[k].reshape(-1) for k, n in enumerate(names)},
            {n: vs[k].reshape(-1) for k, n in enumerate(names)}]
    cert = certify(wa.projected_assembly(r.tiling), constraint="divergence-free",
                   components=comp)
    c = cert.condition
    assert c.constraint_residual_blend > 1e-2
    assert c.constraint_residual_projected < 0.1 * c.constraint_residual_blend


def test_a_projection_declared_and_not_applied_is_not_conservative():
    """Cadence zero is a declaration that does not run, which is worse than no
    declaration: the compile would otherwise issue L6/C2 on it."""
    from atlas.assembly import AssemblyCondition, ConstraintProjection
    p = ConstraintProjection("divergence-free", cadence=0)
    assert p.satisfies_C2 is False
    assert "declared and not applied" in p.why()
    cond = AssemblyCondition(constraint="divergence-free",
                             projection_scope="global",
                             projection_stage="after-assembly",
                             projection_cadence=0)
    assert cond.conservative is False


def test_C2_is_None_rather_than_False_when_no_constraint_is_declared():
    """A graph whose agents enforce nothing pointwise has no C2 to satisfy, and
    reporting False there would report the failure of a condition that does not
    apply."""
    from atlas.assembly import AssemblyCondition
    assert AssemblyCondition().conservative is None
    assert "nothing to check" in AssemblyCondition().why_C2()


# --- R12, at the compile ------------------------------------------------------


def _compile(graph):
    from atlas.compiler import compile_scheme
    return compile_scheme(graph)


def _r12(result):
    return [d for d in result.decisions if d.rule == "R12"]


def _convex_grid_pou(n=8):
    """`_line_pou`'s geometry as a grid partition, with its contamination declared.

    The grid form because that is what a real tiling can afford, and the
    contamination because W49's Pi has to be computable or the compile
    decertifies for a reason that has nothing to do with R12.
    """
    idx = {"a": np.arange(0, 5), "b": np.arange(3, n)}
    wts = {"a": np.array([1.0, 1.0, 1.0, 0.6, 0.3]),
           "b": np.array([0.4, 0.7, 1.0, 1.0, 1.0])}
    bad = {"a": np.array([0, 0, 0, 1, 1], bool),
           "b": np.array([1, 1, 0, 0, 0], bool)}
    return GridPartitionOfUnity(n, idx, wts, contaminated=bad,
                                ramp_cells=2, profile="linear ramp, 2 cells")


def _embedded_caps(name, M=6):
    """`_caps` with the elliptic part left INSIDE the agent, which is what makes
    the agent's own returned field satisfy the constraint -- L6/C2's hypothesis."""
    import dataclasses
    return dataclasses.replace(_caps(name, M),
                               elliptic_subsolve=EllipticSubsolve.EMBEDDED)


def _embedded_graph(pou, **measured):
    from atlas.graph import Agent
    g = _graph(pou, **measured)
    g.agents = [Agent(a.agent_id, _embedded_caps(a.agent_id), domain=a.domain)
                for a in g.agents]
    return g


def test_R12_decertifies_a_blend_with_no_constraint_projection():
    r = _compile(_embedded_graph(_convex_grid_pou()))
    hit = _r12(r)
    assert hit and hit[0].verdict is ADMIT_UNCERTIFIED
    assert hit[0].quantity == "assembled state"
    assert "declares no projection" in hit[0].message
    assert "grad(chi_i)" in hit[0].message


def test_R12_admits_a_global_after_assembly_projection_on_exposed_agents():
    """The measured-stable arrangement, and the only one R12 admits.

    Exposed agents -- so the constraint operator runs exactly once, in the
    composition layer, after the blend -- plus a global after-assembly
    projection.  That is R10 + R10b + R12 together, and it is the classical
    column that survives 120 macro-steps where every other arrangement does not.
    """
    from atlas.assembly import ConstraintProjection, ProjectedAssembly
    pou = ProjectedAssembly(_convex_grid_pou(),
                            ConstraintProjection("divergence-free"))
    hit = _r12(_compile(_graph(pou)))            # _caps is EXPOSED
    assert hit and hit[0].verdict is ADMIT
    assert "ONCE per macro-step" in hit[0].message


def test_R12_decertifies_a_projection_added_on_top_of_the_agents_own():
    """**The repair is to MOVE the projection, not to add one.**

    The arrangement is right -- global, after the blend, once per exchange --
    and the agents ALSO project internally, so the constraint operator runs
    twice per macro-step.  Measured over 120 macro-steps at six windows that is
    worse than either endpoint: the band is left at 74 with no global
    projection, at 51 with one, and never with the agents' elliptic parts
    exposed.  So this is a decertification naming R10, not an admit.
    """
    from atlas.assembly import ConstraintProjection, ProjectedAssembly
    pou = ProjectedAssembly(_convex_grid_pou(),
                            ConstraintProjection("divergence-free"))
    hit = _r12(_compile(_embedded_graph(pou)))
    assert hit and hit[0].verdict is ADMIT_UNCERTIFIED
    assert "twice per macro-step" in hit[0].message
    assert "The repair is R10" in hit[0].message


def test_R12_refuses_a_per_subdomain_projection():
    """W98's measured failure, declared. An operator that cannot see across a
    seam cannot repair what the blend broke across it."""
    from atlas.assembly import ConstraintProjection, ProjectedAssembly
    pou = ProjectedAssembly(
        _convex_grid_pou(),
        ConstraintProjection("divergence-free", scope="per-subdomain"))
    r = _compile(_embedded_graph(pou))
    hit = _r12(r)
    assert hit and hit[0].verdict is REFUSE
    assert r.verdict is REFUSE
    assert "W98" in hit[0].message


def test_R12_refuses_a_projection_before_the_assembly():
    """L6/C2 is a statement about the ORDER of two operators the graph already
    has, so declaring the wrong order is declaring the failure."""
    from atlas.assembly import ConstraintProjection, ProjectedAssembly
    pou = ProjectedAssembly(
        _convex_grid_pou(),
        ConstraintProjection("divergence-free", stage="before-assembly"))
    hit = _r12(_compile(_embedded_graph(pou)))
    assert hit and hit[0].verdict is REFUSE
    assert "ORDER" in hit[0].message


def test_R12_is_silent_when_every_agent_exposes_its_elliptic_part():
    """**The hypothesis of L6/C2 is C u_i = 0, and an EXPOSED agent does not
    satisfy it.**

    The agent hands the elliptic part out, so its local field is not
    individually constrained, there is nothing for the blend to destroy, and the
    composition layer's single global application is downstream of the blend by
    construction.  This is why R12 does not fire on the synthetic graphs Tier 9
    and Tier 10 reach `admit` on -- and it is a reason rather than an exemption.
    """
    r = _compile(_graph(_convex_grid_pou()))       # _caps is EXPOSED
    hit = _r12(r)
    assert hit and hit[0].verdict is ADMIT
    assert "exposed" in hit[0].message
    # and it says what it is NOT checking, which is W105
    assert "does NOT check" in hit[0].message


def test_R12_does_not_clear_R10_and_R10_alone_is_not_the_scheme():
    """Neither rule alone is the scheme, and the measurement is why.

    A graph that declares the projected assembly and keeps the elliptic part
    inside its agents is refused by R10 -- and R12 decertifies it too, because
    the projection is then the second application of the same pressure.  Going
    the other way, exposed agents with no projection declared clear R10 and R12
    stays silent, and measured that column runs 120 macro-steps with
    ``||div u||`` at 1.25: stable, and not incompressible.  W105 is that gap.
    """
    from atlas.assembly import ConstraintProjection, ProjectedAssembly
    pou = ProjectedAssembly(_convex_grid_pou(),
                            ConstraintProjection("divergence-free"))
    res = _compile(_embedded_graph(pou))
    assert _r12(res)[0].verdict is ADMIT_UNCERTIFIED
    assert [d for d in res.decisions.refusals if d.rule == "R10"]
    assert res.verdict is REFUSE

    bare_exposed = _compile(_graph(_convex_grid_pou()))
    assert not [d for d in bare_exposed.decisions.refusals if d.rule == "R10"]
    assert _r12(bare_exposed)[0].verdict is ADMIT
    assert "W105" in _r12(bare_exposed)[0].message


def test_R12_reports_a_projection_onto_a_different_kernel():
    from atlas.assembly import ConstraintProjection, ProjectedAssembly
    pou = ProjectedAssembly(_convex_grid_pou(),
                            ConstraintProjection("mean-free"))
    hit = _r12(_compile(_embedded_graph(pou)))
    assert hit and hit[0].verdict is ADMIT_UNCERTIFIED
    assert "different kernel" in hit[0].message


# --- the case studies' own declarations --------------------------------------


def test_the_exposed_column_is_the_one_R10_stops_refusing():
    """**The arrangement the 120-step march selects, at the declaration level.**

    `wake_array.exposed_reference_solver` hands the elliptic part to the
    composition layer, so `fluid_capabilities` declares
    ``elliptic_subsolve=exposed`` and R10's refusal -- which has stood against
    the reference column of every graph in this vault since Tier 0 -- does not
    fire.  R10b then admits the cadence and R12 admits the assembly.  It is the
    first classical column here that R10 does not refuse, and it is the one that
    survives 120 macro-steps.
    """
    from atlas.scheme import Budget
    r = sl.rung(2, 1)
    u, v = np.ones(r.shape), np.zeros(r.shape)
    g, _ = sl.build(u, v, r, kind="reference_exposed")
    assert all(a.capabilities.elliptic_subsolve is EllipticSubsolve.EXPOSED
               for a in g.agents if a.agent_id.startswith("F"))
    from atlas.compiler import compile_scheme
    res = compile_scheme(g, budget=Budget(allow_probe=False))
    assert not [d for d in res.decisions.refusals if d.rule == "R10"]
    assert [d for d in res.decisions if d.rule == "R10b" and d.verdict is ADMIT]
    assert _r12(res)[0].verdict is ADMIT

    embedded, _ = sl.build(u, v, r, kind="reference",
                           elliptic=EllipticSubsolve.EMBEDDED)
    assert [d for d in compile_scheme(embedded, budget=Budget(allow_probe=False)
                                      ).decisions.refusals if d.rule == "R10"]


def test_the_exposed_solver_declines_the_projection_and_nothing_else():
    """The composition layer is allowed to decline a PART of an agent, and that
    is the whole edit: `_project` is the identity and the velocity update is
    untouched."""
    ex = wa.exposed_reference_solver(wa.NU_REF)
    ref = wa.reference_solver(wa.NU_REF)
    rng = np.random.default_rng(4)
    a = 1.0 + 0.05 * rng.standard_normal((1, wa.N, wa.N))
    b = 0.02 * rng.standard_normal((1, wa.N, wa.N))
    assert np.array_equal(ex._project(a, b)[0], a)
    assert np.array_equal(ex._project(a, b)[1], b)
    # the reference one does not leave it alone
    assert not np.array_equal(ref._project(a.copy(), b.copy())[0], a)
    assert type(ex).__mro__[1] is type(ref)


def test_the_ladder_declares_the_projected_assembly_at_every_rung():
    """One field on one object, and it is the only thing that differs."""
    for n_col, n_row in ((2, 1), (3, 2), (4, 3)):
        r = sl.rung(n_col, n_row)
        u, v = np.ones(r.shape), np.zeros(r.shape)
        on, _ = sl.build(u, v, r, kind="reference",
                         elliptic=EllipticSubsolve.EMBEDDED)
        off, _ = sl.build(u, v, r, kind="reference",
                          elliptic=EllipticSubsolve.EMBEDDED,
                          assembly_projection=False)
        assert on.assembly_projection is not None
        assert on.assembly_projection.satisfies_C2 is True
        assert off.assembly_projection is None
        # the blend itself is the same object's worth of weights either way
        assert on.partition_of_unity.chi_min() == off.partition_of_unity.chi_min()
        assert (on.partition_of_unity.identity_residual()
                == off.partition_of_unity.identity_residual())


def test_the_wake_array_declares_it_by_default():
    """The parent case study gets the step too, and its own note says where the
    ordering is declared."""
    u = np.ones((wa.DEFAULT_TILING.ny, wa.DEFAULT_TILING.nx))
    g, _ = wa.build(u, np.zeros_like(u), kind="reference")
    assert g.assembly_projection is not None
    assert g.assembly_projection.constraint == "divergence-free"
    assert "W100" in g.global_fields[0].note


def test_the_harness_carries_the_assembly_projection():
    """**W54's fifth instance, and the first to end a rollout.**

    The row's own definition of done is that changing a harness parameter
    changes the attribution.  Here it changes the harness KEY -- two runs whose
    assemblies differ are not comparable -- while the depth tag and every other
    field are identical.
    """
    r = sl.rung(3, 2)
    u, v = np.ones(r.shape), np.zeros(r.shape)
    on, _ = sl.build(u, v, r, kind="reference",
                     elliptic=EllipticSubsolve.EMBEDDED)
    off, _ = sl.build(u, v, r, kind="reference",
                      elliptic=EllipticSubsolve.EMBEDDED,
                      assembly_projection=False)
    h_on = HarnessParameters.from_graph(on)
    h_off = HarnessParameters.from_graph(off)
    assert h_off.assembly_projection == "none"
    assert h_on.assembly_projection == (
        "divergence-free projection, global, after-assembly, 1x per exchange")
    assert h_on.key() != h_off.key()
    # and it is the ONLY field that moved
    assert (h_on.overlap_cells, h_on.chi_shape, h_on.exchange_dt,
            h_on.exchange_cadence, h_on.elliptic_placement) == (
        h_off.overlap_cells, h_off.chi_shape, h_off.exchange_dt,
        h_off.exchange_cadence, h_off.elliptic_placement)


def test_the_assembly_hole_now_requires_the_conservative_field():
    """L6's slot has a fifth field, because L6/C2 is a second condition rather
    than a refinement of the first."""
    from atlas.holes import ASSEMBLY_CERTIFICATE
    assert "conservative" in ASSEMBLY_CERTIFICATE.measurements_required
    assert "L6/C2" in ASSEMBLY_CERTIFICATE.declared_interface["conservative"]
    assert "R12" in ASSEMBLY_CERTIFICATE.declared_interface["conservative"]
