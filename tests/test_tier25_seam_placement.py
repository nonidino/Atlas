"""Tier 25 -- CS-9*, and whether an algorithm can place a seam.

`atlas/cases/seam_placement.py` and `scripts/w112_seam_placement.py`.  Six
groups, and the first two are what make the rest mean anything:

  * **the bridge** -- `from_array_tiling` reproduces `wake_array`'s own tiling
    exactly: the same boxes, the same partition of unity to the BIT, the same
    overlapping-pair count CS-7 publishes.  A search graded against a
    hand-chosen baseline it cannot express is graded against nothing, so this
    is a precondition of the whole case study rather than a convenience;
  * **the controls** -- ZERO (one window, no interface, a composed defect that
    is exactly `0.0` and not merely small), and the L2/C2 identity, which
    `tier0-measurements` 10.1 proves exactly and which must therefore close to
    machine precision on every decomposition this file can build.  A run where
    it does not is measuring something other than the cut;
  * **the geometry** -- cut sets, spans, costs, seam enumeration and the
    convexity of the partition, on decompositions with unequal window sizes,
    which no tiling in this vault has ever had;
  * **Q** -- that this file calls the framework's own `probe.cut_score` rather
    than a paraphrase of it, that beta is invariant under a re-declaration of
    the interface space and Q is not, and that the retired-criterion docstring
    still says RETIRED.  If Q were quietly re-implemented here the case study
    would be grading its own copy;
  * **the shell** -- CS-9's segmentation geometry, its partition of unity, and
    the one-segment control;
  * **the artifact** -- the headline numbers re-derived from `out/w112/w112.json`,
    cold, so that nothing in the wiki pages rests on a number this test suite
    cannot reproduce.  Skipped when the artifact is absent.

The live tests cost a few seconds each: a 352x240 monolith step and a batch of
window steps.  The probe tests cost more, so there are two of them.
"""

from __future__ import annotations

import json
import os
import sys

# Before numpy: the build repo pulls torch in, and torch's OpenMP beside
# numpy's MKL aborts the interpreter inside a dense solve.  `test_tier20`'s
# note, and it is not optional here either.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np                                                    # noqa: E402
import pytest                                                         # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas import probe as P                                          # noqa: E402
from atlas.cases import seam_placement as SP                          # noqa: E402
from atlas.cases import wake_array as WA                              # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT = os.path.join(_ROOT, "out", "w112", "w112.json")
STATE = os.path.join(_ROOT, "out", "w100", "state_N6.npz")

needs_state = pytest.mark.skipif(
    not os.path.exists(STATE),
    reason="run scripts/w100_scaling_ladder.py to produce out/w100/state_N6.npz")


@pytest.fixture(scope="module")
def state():
    d = np.load(STATE)
    return d["reference_all_u"], d["reference_all_v"]


@pytest.fixture(scope="module")
def hand_chosen():
    return SP.from_array_tiling(WA.ArrayTiling(n_col=3, n_row=2))


@pytest.fixture(scope="module")
def artifact():
    if not os.path.exists(ARTIFACT):
        pytest.skip("run scripts/w112_seam_placement.py")
    with open(ARTIFACT, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# the bridge -- the search space contains the baseline, and it is the same object
# ---------------------------------------------------------------------------


class TestBridge:
    def test_boxes_are_wake_arrays_own_windows(self, hand_chosen):
        t = WA.ArrayTiling(n_col=3, n_row=2)
        assert [tuple(b) for b in hand_chosen.boxes] == [
            (ox, ox + WA.N, oy, oy + WA.N) for ox, oy in t.offsets]

    def test_partition_of_unity_is_bitwise_identical(self, hand_chosen):
        t = WA.ArrayTiling(n_col=3, n_row=2)
        for a, b in zip(t.weights(), hand_chosen.weights()):
            assert np.array_equal(a, b), "the ramp rule drifted from ArrayTiling's"

    def test_overlap_pair_count_matches_the_published_ladder(self, hand_chosen):
        # `case-study-scaling-ladder-atlas-0.1` 2's table: N=6 has 11
        # overlapping window pairs, diagonals included.
        assert len(hand_chosen.overlap_pairs()) == 11

    def test_the_hand_chosen_tiling_is_not_evenly_spaced(self, hand_chosen):
        # 128-cell windows at a 112 stride do not divide 352 evenly, so the
        # baseline is NOT what `uniform` produces -- which is why
        # `score_geometry` adds it to the candidate list explicitly.
        assert not hand_chosen.is_uniform
        assert uniform_cuts(352, 3) != list(hand_chosen.x_cuts)

    def test_overlap_is_wake_arrays_halo(self, hand_chosen):
        assert 2 * hand_chosen.halo == WA.HALO
        assert hand_chosen.ramp == WA.RAMP

    def test_the_hand_chosen_cuts_sit_exactly_on_the_rotor_planes(self, hand_chosen):
        """The mechanism behind the whole gate, and it is structural.

        `wake_array.Rotor` lives on an x-ADJACENCY -- the plane between window
        columns ``i`` and ``i+1`` -- so a disk sits at the midpoint of an
        overlap by construction, and the cut this file's parameterization puts
        there is the same plane.  **Every x-seam of CS-6 and CS-7 therefore
        passes through a rotor disc**, which is exactly what
        `generalization-requirements` G5's slogan says not to do, and no page
        had noticed because no case study had ever moved a cut.
        """
        t = WA.ArrayTiling(n_col=3, n_row=2)
        planes = sorted({int(round(r.x_plane / WA.DX)) for r in t.rotors})
        assert planes == list(hand_chosen.x_cuts), (
            f"rotor planes {planes} vs hand-chosen x-cuts "
            f"{list(hand_chosen.x_cuts)}")


def uniform_cuts(n, k):
    return [int(round(i * n / k)) for i in range(1, k)]


# ---------------------------------------------------------------------------
# the controls
# ---------------------------------------------------------------------------


@needs_state
class TestControls:
    def test_one_window_has_exactly_zero_composed_defect(self, state):
        u, v = state
        one = SP.RectDecomposition(nx=u.shape[1], ny=u.shape[0])
        dt = SP.exchange_interval(u, v)
        r = SP.composed_defect(one, u, v, dt)
        # exactly, not to a tolerance: chi is identically one and cutting and
        # assembling are the identity.
        assert r["defect"] == 0.0
        assert one.seams() == []
        assert one.overlap_pairs() == []

    def test_the_l2c2_identity_closes(self, state, hand_chosen):
        u, v = state
        r = SP.composed_defect(hand_chosen, u, v, SP.exchange_interval(u, v))
        # A - Eu = sum_i R_i^T chi_i D_i, exactly (tier0-measurements 10.1)
        assert r["identity_residual_rel"] < 1e-9

    def test_the_cellwise_bound_is_not_violated(self, state, hand_chosen):
        u, v = state
        r = SP.composed_defect(hand_chosen, u, v, SP.exchange_interval(u, v))
        assert r["cellwise_bound_violation"] <= 1e-14 * r["max_abs_defect"] + 1e-15

    def test_the_chi_weighted_bound_is_tight_and_the_max_form_is_not(
            self, state, hand_chosen):
        u, v = state
        r = SP.composed_defect(hand_chosen, u, v, SP.exchange_interval(u, v))
        # 10.1 measured 0.998 and 0.0046 on a different tiling of a different
        # expert; the ORDERS are the claim, not the digits.
        assert 0.5 < r["bound_tightness"] <= 1.0 + 1e-12
        assert r["defect"] / r["Q_star_max"] < 0.1

    def test_the_reference_free_surrogate_tracks_the_max_form(
            self, state, hand_chosen):
        u, v = state
        r = SP.composed_defect(hand_chosen, u, v, SP.exchange_interval(u, v))
        assert abs(r["surrogate_ratio"] - 1.0) < 0.05

    def test_one_exchange_interval_is_one_sub_step_in_both_columns(
            self, state, hand_chosen):
        u, v = state
        r = SP.composed_defect(hand_chosen, u, v, SP.exchange_interval(u, v))
        assert r["n_sub_windows"] == 1 and r["n_sub_monolith"] == 1

    def test_score_candidate_refuses_a_step_that_is_not_one_interval(
            self, state, hand_chosen):
        u, v = state
        with pytest.raises(ValueError, match="not one exchange interval"):
            SP.score_candidate(hand_chosen, u, v, 4 * SP.exchange_interval(u, v),
                               probe=False)


# ---------------------------------------------------------------------------
# the geometry
# ---------------------------------------------------------------------------


class TestGeometry:
    def test_windows_cover_the_domain_and_overlap_by_twice_the_halo(self):
        d = SP.RectDecomposition(nx=352, ny=240, x_cuts=(100, 250),
                                 y_cuts=(90,), halo=8)
        xs = d.x_spans
        assert xs[0][0] == 0 and xs[-1][1] == 352
        for a, b in zip(xs, xs[1:]):
            assert a[1] - b[0] == 2 * d.halo

    def test_unequal_windows_are_expressible_and_the_partition_is_convex(self):
        d = SP.RectDecomposition(nx=352, ny=240, x_cuts=(64, 300), y_cuts=(60,),
                                 halo=8)
        assert len(set(d.shapes)) > 1, "this candidate should have unequal windows"
        ws = np.array(d.weights())
        assert np.all(ws >= 0.0)
        assert np.max(np.abs(ws.sum(axis=0) - 1.0)) < 1e-12

    def test_seam_count_is_side_adjacent_pairs_only(self):
        d = SP.uniform(352, 240, 3, 2)
        # 2 x-seams per row x 2 rows + 3 y-seams = 7; but 11 overlapping pairs,
        # because a diagonal pair overlaps and shares no face.
        assert len(d.seams()) == 7
        assert len(d.overlap_pairs()) == 11

    def test_cost_currencies_disagree(self):
        wide = SP.uniform(352, 240, 3, 2, halo=16)
        narrow = SP.uniform(352, 240, 4, 2, halo=4)
        assert narrow.cost.n_windows > wide.cost.n_windows
        assert narrow.cost.halo_cells < wide.cost.halo_cells

    def test_a_cut_outside_the_domain_is_refused(self):
        with pytest.raises(ValueError):
            SP.RectDecomposition(nx=100, ny=100, x_cuts=(0,))
        with pytest.raises(ValueError):
            SP.RectDecomposition(nx=100, ny=100, x_cuts=(50, 20))

    def test_enumeration_contains_the_uniform_candidate(self):
        cands = SP.enumerate_candidates(352, 240, (3,), (2,), (8,),
                                        (-24, 0, 24))
        assert SP.uniform(352, 240, 3, 2).label in {c.label for c in cands}

    def test_spearman_reports_constant_rather_than_a_number(self):
        assert SP.spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
        assert SP.spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
        assert SP.spearman([3.0, 2.0, 1.0], [1.0, 2.0, 3.0]) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Q -- the framework's own, and its basis dependence
# ---------------------------------------------------------------------------


class TestQ:
    def test_this_file_calls_the_frameworks_cut_score(self):
        assert SP.cut_score is P.cut_score

    def test_cut_score_is_still_retired(self):
        # The case study is about a criterion the vault has already retired;
        # if that word disappears from the docstring, this test is the place
        # the case study finds out.
        assert "RETIRED" in P.cut_score.__doc__

    def test_the_probe_step_matches_the_frameworks(self):
        # `probe.probe_block` computes `max(1e-2, 100 * reproducibility_floor)`
        # inline; PROBE_STEP restates the floor of it, and this pins them.
        assert SP.PROBE_STEP == 1e-2

    def test_beta_is_invariant_and_Q_is_not(self):
        # A real probed DtN operator is strongly DIAGONAL in the declared
        # Fourier basis -- that is what "the basis is a locality measure"
        # means -- so the orbit is measured on a matrix of that shape.  A dense
        # Gaussian one has no privileged basis to be rotated out of and its
        # orbit is near 1, which would test nothing.
        rng = np.random.default_rng(7)
        n = 12
        A = np.diag(np.linspace(1.0, 2.0, n)) + 0.05 * rng.normal(size=(n, n))
        # `tier0-measurements` 8.3 measured the split-step operator's asymmetry
        # at 0.002 -- essentially self-adjoint -- and 10.2's consequence is
        # that every seam of the graph this vault runs could be re-declared
        # into a frame where Q reads near zero. That is the mechanism, and it
        # is what is asserted rather than a threshold on the orbit width.
        S = 0.5 * (A + A.T) + 0.002 * rng.normal(size=(n, n))
        orb = SP.basis_orbit(S, n_frames=8)
        assert orb["beta_invariance_rel"] < 1e-10
        assert orb["Q_in_eigenbasis_of_sym_part"] < 0.05 * orb["Q_declared_fourier"], (
            "Q must collapse in the eigenbasis of the symmetric part -- the "
            "same interface space, re-declared, and interface-transfer-theory "
            "9's own flagged weakness")
        assert orb["orbit_ratio"] > 10.0

    @needs_state
    def test_the_block_is_the_frameworks_probe_block(self, state, hand_chosen):
        """`_block` must BE `probe.probe_block`, not resemble it.

        The case study grades interface-transfer-theory 9's Q, so the S it is
        computed from has to be the S the framework's own probe assembles.  On
        the hand-chosen tiling both are expressible -- the windows are exactly
        `wake_array`'s 128-cell ones -- so the two are run against each other
        here.  Everything is matched: the same exposed solver, the same one
        exchange interval, the same Fourier prolongation, the same absolute
        base trace, the same finite-difference step.
        """
        from atlas.probe import ProbeBudget, probe_block
        from atlas.transfer import InterfaceSpace

        u, v = state
        dt = SP.exchange_interval(u, v)
        # rotors removed: they split a face into `bypass`/`rotor` ports, and a
        # placement search has no rotor agents -- the disks' influence is in
        # the STATE it scores, which is the parent's developed flow.
        til = WA.ArrayTiling(n_col=3, n_row=2, rotors=())
        k, face = 0, "xhi"
        x0, x1, y0, y1 = hand_chosen.boxes[k]
        win = WA.FluidWindow(agent_id=til.names[k], u0=u[y0:y1, x0:x1],
                             v0=v[y0:y1, x0:x1], kind="reference_exposed",
                             dt=dt, tiling=til, nu=WA.NU_REF)
        caps = WA.fluid_capabilities(win)
        port = caps.port(f"{face}:full:MECH")
        prolong = WA._prolongation(win.agent_id, port.name, WA.N)
        space = InterfaceSpace(seam_id="cmp", dim=prolong.dim_M)
        ref = probe_block(caps, port, space, prolong, "cmp",
                          ProbeBudget(fd_step=SP.PROBE_STEP))
        mine = SP._block(hand_chosen, k, face, u, v, dt)
        assert mine.shape == ref.S.shape
        gap = np.linalg.norm(mine - ref.S) / np.linalg.norm(ref.S)
        assert gap < 1e-12, f"this file's probe differs from the framework's by {gap:.3e}"
        assert SP.cut_score(mine, float(np.linalg.svd(
            mine, compute_uv=False)[-1])) == pytest.approx(
            SP.cut_score(ref.S, ref.beta), rel=1e-9)

    @needs_state
    def test_the_probe_returns_a_well_conditioned_seam_operator(
            self, state, hand_chosen):
        u, v = state
        rows = SP.seam_scores(hand_chosen, u, v, SP.exchange_interval(u, v))
        assert len(rows) == 7
        for r in rows:
            assert r["beta"] > 0.0
            assert r["dim_M"] == WA.modes_for(r["n_cells"])
            assert r["Q_ai_inference"] == pytest.approx(
                r["off_diagonal_fraction"] / r["beta"], rel=1e-10)

    def test_aggregation_reports_both_rules(self):
        rows = [{"Q_ai_inference": 1.0, "off_diagonal_mass": 2.0,
                 "off_diagonal_fraction": 0.1, "beta": 1.0, "kappa": 1.0,
                 "norm_S": 1.0},
                {"Q_ai_inference": 3.0, "off_diagonal_mass": 4.0,
                 "off_diagonal_fraction": 0.2, "beta": 2.0, "kappa": 2.0,
                 "norm_S": 2.0}]
        agg = SP.aggregate_seams(rows)
        # W58: a max over pairs and a sum over the grid diverge as the tiling
        # grows, so both are on the record and neither is called "the" value.
        assert agg["Q_ai_inference__max"] == 3.0
        assert agg["Q_ai_inference__sum"] == 4.0


# ---------------------------------------------------------------------------
# CS-9's shared domain
# ---------------------------------------------------------------------------


class TestShell:
    def test_segmentation_geometry(self):
        s = SP.ShellSegmentation(n_z=48, z_cuts=(24,), halo=2)
        assert s.n_segments == 2
        assert s.spans == [(0, 26), (22, 48)]
        assert s.cost.halo_cells == 4

    def test_one_segment_is_the_control(self):
        s = SP.ShellSegmentation(n_z=48)
        assert s.n_segments == 1
        assert s.spans == [(0, 48)]
        assert s.cost.halo_cells == 0

    def test_candidates_are_exhaustive_and_admissible(self):
        cands = SP.shell_candidates(48, (1, 2))
        assert cands[0].n_segments == 1
        two = [c for c in cands if c.n_segments == 2]
        assert len(two) == 48 - 2 * 2 - 1
        assert all(0 < c.z_cuts[0] < 48 for c in two)

    def test_the_two_families_are_both_measured(self):
        pytest.importorskip("scipy")
        seg = SP.ShellSegmentation(n_z=48, z_cuts=(12,), halo=2)
        r = SP.score_shell(seg)
        assert r["conduction_defect"] >= 0.0
        assert r["elasticity_defect"] >= 0.0
        assert r["nearest_cut_to_streak"] > 0.0


# ---------------------------------------------------------------------------
# the artifact -- every headline re-derived cold
# ---------------------------------------------------------------------------


class TestArtifact:
    def test_the_controls_passed(self, artifact):
        c = artifact["controls"]
        assert c["ZERO"]["exactly_zero"] is True
        assert c["REPRODUCE"]["boxes_identical"] is True
        assert c["REPRODUCE"]["weights_bitwise_identical"] is True
        assert c["NOISE"]["defect_repeat_identical"] is True

    def test_the_noise_floor_is_reported_and_small(self, artifact):
        n = artifact["controls"]["NOISE"]
        # It bounds every ranking claim in the case study, so it must be a
        # number and it must be well below the spreads being ranked.
        assert 0.0 < n["Q_fd_step_relative_spread_max"] < 1e-2

    def test_the_search_beat_or_matched_the_hand_chosen_tiling(self, artifact):
        # The gate: SOME criterion, under SOME budget, on the N=6 geometry,
        # found a decomposition whose defect is within a factor of the
        # hand-chosen one. The factor itself is quoted in the wiki; this pins
        # that the comparison was made and is finite.
        s = artifact["geometries"]["N6"]["searches"]
        ratios = [r["chosen_over_hand_chosen"] for r in s.values()
                  if r.get("chosen_over_hand_chosen") is not None]
        assert ratios, "no search reported a comparison against the baseline"
        assert all(np.isfinite(r) for r in ratios)
        assert min(ratios) <= 1.0, (
            "no criterion found a cut at least as good as the hand-chosen one")

    def test_the_reference_free_criterion_is_the_one_that_ranks(self, artifact):
        s = artifact["geometries"]["N6"]["searches"]
        rf = [r for k, r in s.items() if k.endswith("Q_hat_reference_free")]
        assert rf
        for r in rf:
            assert r["rank_correlation_vs_defect"] is not None
            assert r["rank_correlation_vs_defect"] > 0.5

    def test_the_basis_orbit_is_recorded_on_real_seams(self, artifact):
        b = artifact["basis"]
        assert b["n_seams_measured"] > 0
        assert b["orbit_ratio_min"] > 1.0
        assert b["beta_invariance_worst"] < 1e-10

    def test_every_disagreement_is_listed(self, artifact):
        d = artifact["disagreements"]
        assert set(d) <= set(artifact["geometries"])
        for tag, block in d.items():
            assert len(block["rows"]) > 0
            for r in block["rows"]:
                assert r["geometry"] == tag
                assert "sign_disagrees" in r and "argmin_disagrees" in r

    def test_Q_ranks_backwards_on_the_halo_axis(self, artifact):
        """The one place Tier 10's sign inversion reproduces, and it is a new axis.

        On the placement pools Q ranks POSITIVELY, so 10.2's -0.853 does not
        carry over to where a cut goes.  On the OVERLAP it does: Q and both its
        factors rank exactly -1.000 over the halo sweep, preferring the
        narrowest overlap, which is the worst.
        """
        s = artifact["geometries"]["N6"]["searches"]
        halo = {k.split("::", 1)[1]: r for k, r in s.items()
                if r["pool"] == "HALO"}
        for crit in ("Q_ai_inference__max", "Q_ai_inference__sum",
                     "off_diagonal_mass__max", "off_diagonal_mass__sum"):
            assert halo[crit]["rank_correlation_vs_defect"] == pytest.approx(-1.0)
            assert halo[crit]["penalty_vs_best"] > 4.0
        # and the reference-free surrogate ranks +1 on the same four rungs
        assert halo["Q_hat_reference_free"]["rank_correlation_vs_defect"] == (
            pytest.approx(1.0))
        assert halo["Q_hat_reference_free"]["penalty_vs_best"] == pytest.approx(1.0)

    def test_the_one_interval_advantage_reverses_at_horizon(self):
        """The gate passes at one exchange interval and does not survive a march.

        This is the case study's own falsification and the number that bounds
        every other number in it, so it is pinned rather than described.
        """
        path = os.path.join(_ROOT, "out", "w112", "horizon.json")
        if not os.path.exists(path):
            pytest.skip("run scripts/w112_horizon.py")
        with open(path, encoding="utf-8") as fh:
            h = json.load(fh)
        rows = h["tracks"]["one_interval_winner"]["rows"]
        r = [x["over_hand_chosen"] for x in rows]
        assert r[0] < 0.30, "the one-interval win is the gate and it must be there"
        assert max(r) > 1.25, (
            "the chosen cut must be measurably WORSE than the hand-chosen one "
            "somewhere on the march -- that is the finding")
        crossings = sum(1 for i in range(1, len(r))
                        if (r[i - 1] - 1.0) * (r[i] - 1.0) < 0)
        assert crossings >= 1, "the ratio must cross 1.0 at least once"

    def test_the_two_families_want_different_cuts(self, artifact):
        """CS-9's result: one Gamma, two physics, opposite preferences."""
        if "shell" not in artifact:
            pytest.skip("run without --skip-shell")
        sh = artifact["shell"]
        k = sh["KNOWN_shell"]
        # conduction wants to cut AWAY from the streak, elasticity toward it
        assert k["distance_vs_conduction_defect_rank"] < -0.9
        assert k["distance_vs_elasticity_defect_rank"] > 0.9
        assert k["conduction_penalty_for_cutting_through_it"] > 100.0
        assert sh["by_conduction"]["families_agree"] is False
        assert sh["by_elasticity"]["families_agree"] is False
        # and over the whole candidate set the two are essentially uncorrelated
        assert abs(sh["by_conduction"]["rank_correlation_between_families"]) < 0.2

    def test_the_premise_correction_is_on_the_artifact(self, artifact):
        # The case study corrects W112's own statement of what has and has not
        # been measured. That correction travels with the numbers.
        assert "10.2" in artifact["premise_correction"]
        assert "-0.853" in artifact["premise_correction"]

    def test_the_shell_reports_both_families(self, artifact):
        if "shell" not in artifact:
            pytest.skip("run without --skip-shell")
        sh = artifact["shell"]
        assert sh["by_conduction"]["ranked_by"] == "conduction_defect"
        assert sh["by_elasticity"]["ranked_by"] == "elasticity_defect"
        assert "families_agree" in sh["by_conduction"]
        ctrl = sh["one_segment_control"]
        assert ctrl["n_segments"] == 1
# ---------------------------------------------------------------------------
# every number the wiki pages quote in prose, re-derived from the artifact
# ---------------------------------------------------------------------------


class TestQuotedNumbers:
    """`tier0-measurements` 21 and the case-study page quote figures in prose
    that no other test covers.  A number transcribed into a wiki page is a
    number that can drift from the artifact it came from, and the only repair
    for that is a reader that runs.  Each assertion below names the sentence it
    is protecting.
    """

    @staticmethod
    def _n6(artifact):
        g = artifact["geometries"]["N6"]
        return {s["decomposition"]["label"]: s for s in g["scores"]}, g

    def test_the_gate_ratios(self, artifact):
        # "0.2685x at N=6, 0.305x at N=12, 0.2949x on SOLO"
        want = {"N6": (0.2685, 138, 178), "N12": (0.3050, 48, 51),
                "SOLO": (0.2949, 30, 54)}
        for tag, (ratio, rank, n) in want.items():
            place = [r for r in artifact["geometries"][tag]["searches"].values()
                     if r["pool"] == "PLACE"]
            best = min(r["best_over_hand_chosen"] for r in place)
            assert best == pytest.approx(ratio, abs=5e-4), tag
            assert place[0]["hand_chosen_rank"] == rank, tag
            assert place[0]["n_candidates"] == n, tag

    def test_the_improvement_factor_and_the_shear_that_explains_it(self, artifact):
        # "worth 3.7x ... and the mean shear falls from 0.3333 to 0.0611"
        by, _g = self._n6(artifact)
        hand = by["3x2|x120-232|y120|h8"]
        win = by["3x2|x53-171|y168|h8"]
        assert (hand["defect"]["defect"] / win["defect"]["defect"]
                == pytest.approx(3.725, abs=5e-3))
        assert hand["state"]["cut_shear__mean"] == pytest.approx(0.3333, abs=5e-4)
        assert win["state"]["cut_shear__mean"] == pytest.approx(0.0611, abs=5e-4)

    def test_the_l2c2_instrument_checks(self, artifact):
        # "closes at 4.084e-12 ... violated by 2.220e-16 ... tight at 0.989
        #  and the max form is 179x loose"
        by, _g = self._n6(artifact)
        d = by["3x2|x120-232|y120|h8"]["defect"]
        assert d["identity_residual_rel"] == pytest.approx(4.084e-12, rel=1e-2)
        assert d["cellwise_bound_violation"] == pytest.approx(2.220e-16, rel=1e-2)
        assert d["bound_tightness"] == pytest.approx(0.989, abs=1e-3)
        assert (d["Q_star_max"] / d["defect"]) == pytest.approx(179.0, abs=1.0)

    def test_the_halo_sweep_table(self, artifact):
        # the four rows of 21.4's halo table, defect and Q both
        by, _g = self._n6(artifact)
        want = {4: (5.434e-6, 0.04066), 8: (2.126e-6, 0.07015),
                16: (1.267e-6, 0.09600), 24: (1.088e-6, 0.10685)}
        prev_defect, prev_q = None, None
        for h, (defect, q) in sorted(want.items()):
            s = by[f"3x2|x120-232|y120|h{h}"]
            assert s["defect"]["defect"] == pytest.approx(defect, rel=1e-3), h
            assert s["aggregate"]["Q_ai_inference__max"] == pytest.approx(q, rel=1e-3), h
            if prev_defect is not None:
                # the whole point: the defect FALLS and Q RISES with the overlap
                assert s["defect"]["defect"] < prev_defect
                assert s["aggregate"]["Q_ai_inference__max"] > prev_q
            prev_defect, prev_q = s["defect"]["defect"], s["aggregate"]["Q_ai_inference__max"]
        # "the narrowest is worse by 4.993x"
        assert (by["3x2|x120-232|y120|h4"]["defect"]["defect"]
                / by["3x2|x120-232|y120|h24"]["defect"]["defect"]
                ) == pytest.approx(4.993, abs=5e-3)

    def test_the_basis_orbit_statistics(self, artifact):
        # "120 real seams: 1.237x / 5.652x / 85.40x, 42.5% above 10x"
        b = artifact["basis"]
        assert b["n_seams_measured"] == 120
        assert b["orbit_ratio_min"] == pytest.approx(1.237, abs=5e-3)
        assert b["orbit_ratio_median"] == pytest.approx(5.652, abs=5e-3)
        assert b["orbit_ratio_max"] == pytest.approx(85.40, abs=5e-2)
        ratios = [r["orbit_ratio"] for r in b["rows"]]
        frac = sum(1 for x in ratios if x > 10.0) / len(ratios)
        assert frac == pytest.approx(0.425, abs=5e-3)

    def test_the_horizon_criterion_degradation(self, artifact):
        # 21.3's table: every criterion degrades over eight intervals
        h = artifact["horizon"]["N6"]
        assert h["one_interval_vs_marched"] == pytest.approx(0.719, abs=1e-3)
        c = h["criteria"]
        assert c["Q_star_chi_weighted"]["vs_one_interval"] == pytest.approx(
            0.9998, abs=5e-4)
        assert c["Q_star_chi_weighted"]["vs_8_intervals"] == pytest.approx(
            0.718, abs=1e-3)
        assert c["Q_hat_reference_free"]["vs_8_intervals"] == pytest.approx(
            0.681, abs=1e-3)
        for name, row in c.items():
            assert row["vs_8_intervals"] < row["vs_one_interval"], (
                f"{name} must degrade over the horizon -- that is the finding")

    def test_the_cs9_family_numbers(self, artifact):
        # 21.6's table
        if "shell" not in artifact:
            pytest.skip("run without --skip-shell")
        sh = artifact["shell"]
        assert sh["n_candidates"] == 785
        k = sh["KNOWN_shell"]
        assert k["distance_vs_conduction_defect_rank"] == pytest.approx(
            -0.9988, abs=5e-4)
        assert k["distance_vs_elasticity_defect_rank"] == pytest.approx(
            0.9991, abs=5e-4)
        assert k["conduction_penalty_for_cutting_through_it"] == pytest.approx(
            782.1, rel=1e-3)
        assert sh["by_elasticity"]["cost_of_following_this_family_for_the_other"
                                   ] == pytest.approx(795.4, rel=1e-3)
        assert sh["by_conduction"]["cost_of_following_this_family_for_the_other"
                                   ] == pytest.approx(3.601, rel=1e-3)
        assert sh["by_conduction"]["rank_correlation_between_families"
                                   ] == pytest.approx(-0.0259, abs=5e-4)

    def test_the_known_control_ratio(self, artifact):
        # "32.25x", and every criterion picking row 216
        k = artifact["geometries"]["SOLO"]["KNOWN"]
        assert k["wake_row"] == 53
        assert k["on_over_away"] == pytest.approx(32.25, rel=1e-3)
        assert k["argmin_defect_row"] == 216
        assert k["argmin_Q_row"] == 216
        assert k["argmin_Q_hat_row"] == 216

    def test_the_horizon_march_shape(self):
        # 21.3's own table: crosses at 10, peaks 1.5909 at 90, 0.9039 at 240
        path = os.path.join(_ROOT, "out", "w112", "horizon.json")
        if not os.path.exists(path):
            pytest.skip("run scripts/w112_horizon.py --steps 240")
        with open(path, encoding="utf-8") as fh:
            h = json.load(fh)
        rows = h["tracks"]["one_interval_winner"]["rows"]
        if len(rows) < 240:
            pytest.skip("horizon.json was produced with fewer than 240 steps")
        r = {x["step"]: x["over_hand_chosen"] for x in rows}
        assert r[1] == pytest.approx(0.2685, abs=5e-4)
        assert r[8] == pytest.approx(0.9455, abs=5e-4)
        assert r[32] == pytest.approx(1.3352, abs=5e-4)
        assert r[240] == pytest.approx(0.9039, abs=5e-4)
        peak = max(r, key=lambda k_: r[k_])
        assert peak == 90 and r[peak] == pytest.approx(1.5909, abs=5e-4)
        first_cross = next(k_ for k_ in sorted(r) if r[k_] > 1.0)
        assert first_cross == 10
        # and the eight-interval winner's own crossing, quoted at 95
        r8 = {x["step"]: x["over_hand_chosen"]
              for x in h["tracks"]["eight_interval_winner"]["rows"]}
        assert next(k_ for k_ in sorted(r8) if r8[k_] > 1.0) == 95
