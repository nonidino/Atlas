"""Tier 41 -- CS-S2: a bounded-receptive-field donor through Tier 40's instrument.

`scripts/cs_s2_bounded_donor.py` ran NeuberNet -- the one continuum donor in
[[expert-donor-survey]] whose input is a boundary displacement -- through
`w173_epsilon_halo.measure` and `w173_transverse_reach.depth_star` unchanged,
beside a classical elastic patch on the same disc and split-step `WindowNS` as
the positive control.

**The weights are not in this repository** (no licence; `~/.cache/neubernet`),
so every test here reads `out/cs_s2/cs_s2.json` and skips when it is absent.
An artifact that exists but is incomplete FAILS rather than skips: a partial
file is what the first run would have left behind had it written anything.

**The controls are asserted before any result**, in the order that a reading
depends on them: the classical patch passes its own patch test; its operator
converges under the Galerkin port and does not under the pointwise one (the
first run's defect, kept as a measurement); the classical operator is
reciprocal; the positive control reads exactly zero past a finite radius again,
in this session, to the digit Tier 40 recorded.

Then, in order: **W95** on this donor (section 2); **the third port defect**,
the hoop input's sign, from `out/cs_s2/torsion_control.json`, which states both
conventions so its labels stay true after the fix (section 3); **W180**, the
published forward jumping where its sign network changes its call, from
`out/cs_s2/sign_jump.json` (section 4); and **the outcome** against the three
the brief named, from the third run of the driver, made with the corrected
adapter and denormals flushed (section 5).  Each file skips when its artifact is
absent and fails when it is partial.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CS_S2 = os.path.join(_ROOT, "out", "cs_s2", "cs_s2.json")
W173 = os.path.join(_ROOT, "out", "w173", "w173.json")
W173T = os.path.join(_ROOT, "out", "w173", "transverse.json")


@pytest.fixture(scope="module")
def cs2():
    if not os.path.exists(CS_S2):
        pytest.skip("run scripts/cs_s2_bounded_donor.py (needs ~/.cache/neubernet)")
    with open(CS_S2, encoding="utf-8") as fh:
        d = json.load(fh)
    assert d.get("complete") is True, "cs_s2.json is a partial artifact"
    return d


def _E2(subject, r):
    return next(x["E2_rel"] for x in subject["linear"]["rows"] if x["r"] == r)


# ---------------------------------------------------------------------------
# 0. provenance
# ---------------------------------------------------------------------------


def test_provenance_is_recorded_and_says_there_is_no_licence(cs2):
    src = cs2["source"]
    assert src["licence"] is None
    assert src["commit"] == "cce93244dfc27db9dbe351760760a8142cee0bb0"
    assert set(cs2["files"]) == {"neubernet.pt", "yieldnet.pt", "signnet.pt",
                                 "definitions.py"}
    assert all(v["matches"] for v in cs2["files"].values())


def test_no_neubernet_file_was_copied_into_the_repository():
    names = {"neubernet.pt", "yieldnet.pt", "signnet.pt"}
    for top in ("atlas", "scripts", "out", "tests", "wiki"):
        for dirpath, _dirs, files in os.walk(os.path.join(_ROOT, top)):
            assert not names & set(files), dirpath
    assert not os.path.exists(os.path.join(_ROOT, "out", "cs_s2", "definitions.py"))


# ---------------------------------------------------------------------------
# 1. the controls
# ---------------------------------------------------------------------------


def test_the_classical_patch_passes_its_own_patch_test(cs2):
    st = cs2["classical"]["self_test"]
    for h in ("h=0.2", "h=0.1"):
        assert st[h]["axial_u_max_err"] < 1e-10
        assert st[h]["axial_flux_x_max_err"] < 1e-10
    #: the Galerkin flux converges as the mesh halves
    assert st["h=0.1"]["axial_flux_y_max_err"] < st["h=0.2"]["axial_flux_y_max_err"]
    assert st["h=0.1"]["torsion_flux_max_err"] < st["h=0.2"]["torsion_flux_max_err"]
    assert st["h=0.1"]["hat_mass_rel_err"] < st["h=0.2"]["hat_mass_rel_err"]


def test_the_galerkin_port_converges_where_the_pointwise_one_does_not(cs2):
    """A DtN response to a hat trace is log-singular at the knots.

    The first run sampled the flux pointwise there and the classical operator
    moved 36-42% between two meshes; this is that defect kept as a
    measurement, beside the port that replaced it.
    """
    g = cs2["classical"]["mesh_floor_galerkin"]
    p = cs2["classical"]["mesh_floor_pointwise"]
    assert set(g) == set(p) == {"ring:ux", "ring:uy", "ring:ut"}
    for port in g:
        assert g[port] < 0.1, port
        assert p[port] > 0.2, port
        assert g[port] < p[port] / 3.0, port


def test_the_classical_operator_is_reciprocal(cs2):
    """Betti, in the Galerkin basis: the control that says reciprocity is readable."""
    assert cs2["classical"]["full_operator"]["reciprocity"] < 1e-10
    for s in cs2["classical"]["subjects"]:
        assert s["reciprocity"] < 1e-10, s["label"]


def test_both_adapters_integrate_against_the_same_hats(cs2):
    assert cs2["classical"]["hat_mass_fe_vs_quadrature"] < 1e-2


def test_the_positive_control_reads_exactly_zero_again_in_this_session(cs2):
    """Tier 40's *exactly zero past r = 13* means zero FROM r = 14.

    E2_rel(13) is 1.4e-14 and E2_rel(14) is the first exact zero, in
    `out/w173/w173.json` and in this session's re-run alike.  The first version
    of this test pinned ``<= 13``, reading *past 13* as *from 13*, and failed on
    a control that had reproduced to the digit.  The radius is pinned exactly
    now, and the one nonzero row below it with it -- tighter, not looser.
    """
    pc = cs2["positive_control"]
    assert pc["exactly_zero_from_r"] is not None
    assert pc["exactly_zero_from_r"] == 14
    rows = {x["r"]: x["E2_rel"] for x in pc["along"]["linear"]["rows"]}
    assert 0.0 < rows[13] < 1e-12
    assert all(v == 0.0 for r, v in rows.items() if r >= 14)
    if os.path.exists(W173):
        with open(W173, encoding="utf-8") as fh:
            t40 = json.load(fh)
        ref = next(s for s in t40["subjects"]
                   if s["subject"] == "windowns-split-step" and s["agent"] == "C00")
        now = {x["r"]: x["E2_rel"] for x in pc["along"]["linear"]["rows"]}
        for x in ref["linear"]["rows"]:
            if x["r"] in now:
                assert now[x["r"]] == pytest.approx(x["E2_rel"], rel=1e-9, abs=1e-15)
    if os.path.exists(W173T):
        with open(W173T, encoding="utf-8") as fh:
            t40t = json.load(fh)
        ref = next(s for s in t40t["subjects"] if s["label"] == "windowns-split-step C00 xhi")
        ref64 = next(r for r in ref["rows"] if r["j0"] == 64)
        now64 = pc["transverse"]["rows"][0]
        assert now64["j0"] == 64
        assert (now64["depth_star_max"]["rows"][1]["depth_star"]
                == ref64["depth_star_max"]["rows"][1]["depth_star"])


# ---------------------------------------------------------------------------
# 2. W95 on this donor
# ---------------------------------------------------------------------------


def test_W95_the_donor_has_no_same_class_reference_pair(cs2):
    w = cs2["w95"]
    assert w["same_class_monolith"] is False
    assert "no larger or undecomposed domain" in w["why_not"]
    assert "exists in this repository" in w["classical_referent_elastic_branch"]
    assert "no solver here computes it" in w["classical_referent_plastic_branch"]


# ---------------------------------------------------------------------------
# 3. the third port defect: the hoop input, measured both ways round
# ---------------------------------------------------------------------------
#
# `scripts/cs_s2_torsion_control.py` states each hoop convention explicitly, so
# its labels stay true after the adapter was corrected.  The second run's
# convention check was made at a tension-only base; the directional stage then
# read the torsion far field at cosine -0.969 and -0.997, and two readings fit
# that -- SignNet deciding at its own boundary, or a sign convention.  Both turned
# out to be real, and these tests keep the diagnosis of each.

CS_S2_TORSION = os.path.join(_ROOT, "out", "cs_s2", "torsion_control.json")


@pytest.fixture(scope="module")
def torsion():
    if not os.path.exists(CS_S2_TORSION):
        pytest.skip("run scripts/cs_s2_torsion_control.py (needs ~/.cache/neubernet)")
    with open(CS_S2_TORSION, encoding="utf-8") as fh:
        d = json.load(fh)
    assert d.get("complete") is True, "torsion_control.json is a partial artifact"
    return d


def _conv(t, base, hoop):
    return next(r for r in t["convention_check"]["rows"]
                if r["base"] == base and r["hoop"] == hoop)


def _dir(t, base, hoop, direction):
    return next(r for r in t["directional"]["rows"]
                if r["base"] == base and r["hoop"] == hoop and r["direction"] == direction)


def test_the_control_is_on_the_drivers_patch_and_saw_the_shipped_adapter(torsion):
    assert all(torsion["setup"]["same_patch_as_driver"].values())
    #: run against the adapter as it stood when the second run was made
    assert torsion["setup"]["adapter_in_cs_s2_neubernet_equals"] == {"+1": True, "-1": False}
    for label in ("tension-elastic", "tension-plastic", "tension-torsion-plastic"):
        b = torsion["bases"][label]
        assert b["scale"] == pytest.approx(b["scale_in_driver"], rel=1e-12)


def test_the_second_runs_hoop_input_was_the_wrong_way_round(torsion):
    """At a nonzero-torsion elastic base -- SignNet's call firm, production mode
    exactly odd in torsion -- ``ROTY = +u_theta/rho`` NEGATES both hoop shears
    and the hoop flux and leaves the in-plane stresses where they were, and
    ``ROTY = -u_theta/rho`` agrees on all of them.  That is a convention, not a
    SignNet boundary: the call is the same size and opposite under the two."""
    for base in ("torsion-elastic", "tension-torsion-elastic"):
        wrong, right = _conv(torsion, base, "+1"), _conv(torsion, base, "-1")
        for comp in ("s_yz", "s_xz"):
            assert wrong["stress"][comp]["cosine"] < -0.998, (base, comp)
            assert wrong["stress"][comp]["rel_error"] > 1.9, (base, comp)
            assert wrong["stress"][comp]["sign_agreement_where_classical_above_10pct"] == 0.0
            assert right["stress"][comp]["cosine"] > 0.998, (base, comp)
            assert right["stress"][comp]["rel_error"] < 0.07, (base, comp)
            assert right["stress"][comp]["sign_agreement_where_classical_above_10pct"] == 1.0
        assert wrong["ring_flux"]["t_theta"]["cosine"] < -0.99
        assert right["ring_flux"]["t_theta"]["cosine"] > 0.99
        assert wrong["regime"]["sign_torsion"] == -right["regime"]["sign_torsion"]
    w = _conv(torsion, "tension-torsion-elastic", "+1")
    r = _conv(torsion, "tension-torsion-elastic", "-1")
    assert w["stress_in_plane"]["rel_error"] == pytest.approx(r["stress_in_plane"]["rel_error"],
                                                              rel=1e-9)
    assert r["stress_in_plane"]["rel_error"] < 0.06


def test_the_tension_only_check_could_not_have_seen_it(torsion):
    """Both subjects' hoop shears are identically zero at a tension-only base, so
    the check the second run passed carried no information about the sign."""
    for hoop in ("+1", "-1"):
        r = _conv(torsion, "tension-elastic", hoop)
        assert r["stress"]["s_yz"]["norm_classical"] == 0.0
        assert r["stress"]["s_xz"]["norm_classical"] == 0.0
        assert r["stress"]["s_yz"]["cosine"] is None
    assert (_conv(torsion, "tension-elastic", "+1")["stress_in_plane"]
            == _conv(torsion, "tension-elastic", "-1")["stress_in_plane"])


def test_the_hoop_check_has_teeth(torsion):
    """Mirrored query points read the correct convention's shears as wrong."""
    for base in ("torsion-elastic", "tension-torsion-elastic"):
        r = _conv(torsion, base, "-1")
        for comp in ("s_yz", "s_xz"):
            assert (r["control_mirrored_points"][comp]["rel_error"]
                    > 5.0 * r["stress"][comp]["rel_error"]), (base, comp)


def test_both_training_directions_are_faithful_where_both_loads_are_signed(torsion):
    for direction, tol in (("tension far field", 0.09), ("torsion far field", 0.13)):
        p = _dir(torsion, "tension-torsion-elastic", "-1", direction)["plus"]
        assert p["cosine"] > 0.99 and p["rel_error"] < tol, direction
    #: and off the training manifold the response is absent, either way round
    for hoop in ("+1", "-1"):
        one = _dir(torsion, "tension-torsion-elastic", hoop, "one sensor on u_y")["plus"]
        assert one["norm_ratio"] < 0.1 and abs(one["cosine"]) < 0.01


def test_at_zero_torsion_the_derivative_depends_on_which_way_it_is_taken(torsion):
    """The other reading, and it is real too.  Where SignNet's call differs
    between base + a v and base - a v the two one-sided differences disagree by
    a factor near 50; where it does not -- the plastic base at this step -- they
    agree, and with the corrected hoop the derivative is faithful."""
    el = _dir(torsion, "tension-elastic", "-1", "torsion far field")
    assert el["sign_torsion_at"]["base+av"] != el["sign_torsion_at"]["base-av"]
    lo, hi = sorted([el["plus"]["norm_ratio"], el["minus"]["norm_ratio"]])
    assert lo < 0.2 and hi > 8.0
    pl = _dir(torsion, "tension-plastic", "-1", "torsion far field")
    assert pl["sign_torsion_at"]["base+av"] == pl["sign_torsion_at"]["base-av"] == 1.0
    assert pl["plus_vs_minus_cosine"] > 0.999
    assert pl["plus"]["cosine"] > 0.99 and abs(pl["plus"]["norm_ratio"] - 1.0) < 0.02


def test_signnets_call_holds_at_small_steps_and_splits_at_larger_ones(torsion):
    rows = torsion["signnet"]["rows"]

    def call(base, hoop, a):
        return next(r["sign_torsion"] for r in rows
                    if r["base"] == base and r["hoop"] == hoop and r["a"] == a)

    for hoop in ("+1", "-1"):
        for a in (1e-4, 1e-3):
            assert call("tension-elastic", hoop, a) == call("tension-elastic", hoop, -a) == 1.0
        for a in (1e-2, 1e-1):
            assert call("tension-elastic", hoop, a) == -call("tension-elastic", hoop, -a)
        for a in (1e-4, 1e-3, 1e-2):
            assert call("tension-plastic", hoop, a) == call("tension-plastic", hoop, -a) == 1.0
        assert call("tension-plastic", hoop, 1e-1) == -call("tension-plastic", hoop, -1e-1)


def test_at_zero_tension_the_tension_derivative_reads_the_same_way(torsion):
    for hoop in ("+1", "-1"):
        p = _dir(torsion, "torsion-elastic", hoop, "tension far field")["plus"]
        assert p["norm_ratio"] > 5.0 and p["cosine"] < 0.3
    #: under pure torsion the tension sign is SignNet's own call, not the load's
    assert torsion["bases"]["torsion-elastic"]["regime"]["+1"]["sign_tension"] == -1.0


# ---------------------------------------------------------------------------
# 4. W180: where the sign network changes its call, the published forward jumps
# ---------------------------------------------------------------------------
#
# `scripts/cs_s2_sign_jump.py`: J(a) = ||F(base + a v) - F(base - a v)|| on the
# block v drives, SignNet's decision point bisected along v, and the forward
# measured across it.  Run on the corrected adapter, with denormals flushed.

CS_S2_JUMP = os.path.join(_ROOT, "out", "cs_s2", "sign_jump.json")


@pytest.fixture(scope="module")
def jump():
    if not os.path.exists(CS_S2_JUMP):
        pytest.skip("run scripts/cs_s2_sign_jump.py (needs ~/.cache/neubernet)")
    with open(CS_S2_JUMP, encoding="utf-8") as fh:
        d = json.load(fh)
    assert d.get("complete") is True, "sign_jump.json is a partial artifact"
    return d


def _series(j, base, direction):
    return next(s for s in j["series"] if s["base"] == base and s["direction"] == direction)


def test_W180_ran_on_the_corrected_adapter_with_denormals_flushed(jump):
    assert jump["hoop_sign"] == -1.0
    assert jump["torch"]["flush_denormal"] is True


def test_W180_the_forward_jumps_where_signnet_changes_its_call(jump):
    """At each zero-load base the call changes a finite step away from zero, and
    across an interval of relative width 2e-3 there the donor's flux moves by
    three to five orders more than linear elasticity does across the same one."""
    for base, direction, lo, hi in (("tension-elastic", "torsion far field", 1e-3, 1e-2),
                                    ("tension-plastic", "torsion far field", 1e-2, 1e-1),
                                    ("torsion-elastic", "tension far field", 1e-5, 1e-4)):
        dp = _series(jump, base, direction)["decision_point"]
        assert dp is not None, base
        assert lo < abs(dp["a_star"]) < hi, (base, dp["a_star"])
        assert dp["call_inside"] != dp["call_outside"], base
        assert dp["jump_block_norm"] > 1e3 * dp["classical_across_same_interval"], base
    #: at zero torsion the jump is a fixed fraction of the physical response to a unit step
    for base in ("tension-elastic", "tension-plastic"):
        dp = _series(jump, base, "torsion far field")["decision_point"]
        assert 0.07 < dp["jump_over_classical_S_v"] < 0.10, base


def test_W180_a_ladder_that_straddles_the_jump_reads_it_as_slope(jump):
    """J / 2a is flat while the step stays inside the decision point and blows up
    once it straddles it: a one-sided probe at the wrong amplitude reports the
    jump as a derivative."""
    rows = {r["a"]: r for r in _series(jump, "tension-elastic", "torsion far field")["rows"]}
    ratio = {a: r["J_over_2a_over_classical"] for a, r in rows.items()}
    assert ratio[1e-3] == pytest.approx(ratio[1e-4], rel=1e-2)
    #: inside, the elastic base's torsion branch carries a fifth of the physics
    assert 0.15 < ratio[1e-3] < 0.21
    assert ratio[1e-2] > 4.0
    assert rows[1e-2]["sign_torsion_plus"] != rows[1e-2]["sign_torsion_minus"]
    assert rows[1e-3]["sign_torsion_plus"] == rows[1e-3]["sign_torsion_minus"]
    z = {r["a"]: r["J_over_2a_over_classical"]
         for r in _series(jump, "torsion-elastic", "tension far field")["rows"]}
    assert z[1e-4] > 10.0 and 0.7 < z[1e-5] < 1.0


def test_W180_with_both_loads_signed_the_forward_is_smooth_and_faithful(jump):
    """The control: neither load at zero, SignNet's call never changes, and J / 2a
    sits within a few percent of linear elasticity until the float32 floor."""
    for direction, key in (("torsion far field", "sign_torsion"),
                           ("tension far field", "sign_tension")):
        s = _series(jump, "tension-torsion-elastic", direction)
        assert s["decision_point"] is None, direction
        for r in s["rows"]:
            assert r[key + "_plus"] == r[key + "_minus"], (direction, r["a"])
            if r["a"] >= 1e-4:
                assert 0.97 < r["J_over_2a_over_classical"] < 1.06, (direction, r["a"])


def test_W180_the_plastic_base_is_faithful_inside_its_decision_point(jump):
    rows = {r["a"]: r for r in _series(jump, "tension-plastic", "torsion far field")["rows"]}
    for a in (1e-2, 1e-3, 1e-4):
        assert 0.95 < rows[a]["J_over_2a_over_classical"] < 1.02, a
        assert rows[a]["sign_torsion_plus"] == rows[a]["sign_torsion_minus"], a


# ---------------------------------------------------------------------------
# 5. the outcome, pinned to the final artifact: not compact, beside the physics
# ---------------------------------------------------------------------------


def _fine(cs2):
    return {s["port"]: s for s in cs2["classical"]["subjects"]
            if s["subject"] == "elastic-patch-fine"}


def _nn(cs2, base, port):
    return next(s for s in cs2["neubernet"]["subjects"]
                if s["base"] == base and s["port"] == port)


def _rows(subject):
    return {x["r"]: x["E2_rel"] for x in subject["linear"]["rows"]}


def _dirrow(cs2, base, direction):
    return next(r for r in cs2["directional"]["rows"]
                if r["base"] == base and r["direction"] == direction)


def test_the_artifact_records_the_corrected_hoop_the_flush_and_all_three_defects(cs2):
    assert cs2["port"]["hoop_sign"] == -1.0
    assert cs2["torch"]["flush_denormal"] is True
    assert [d["run"] for d in cs2["port"]["defects"]] == [1, 1, 2]


def test_the_hoop_check_runs_with_every_build_and_agrees(cs2):
    rows = {r["base"]: r for r in cs2["convention_check_hoop"]["rows"]}
    assert set(rows) == {"torsion-elastic", "tension-torsion-elastic"}
    for base, r in rows.items():
        sc = r["stress_by_component"]
        assert sc["s_yz"]["cosine"] > 0.999 and sc["s_yz"]["rel_error"] < 0.03, base
        assert sc["s_xz"]["cosine"] > 0.998 and sc["s_xz"]["rel_error"] < 0.07, base
        assert r["ring_flux_by_component"]["t_theta"]["cosine"] > 0.99, base
        assert (r["control_mirrored_points_hoop_shears"]["rel_error"]
                > 5.0 * r["stress_hoop_shears"]["rel_error"]), base


def test_at_a_tension_only_base_the_donor_emits_hoop_output_the_physics_does_not(cs2):
    cc = cs2["convention_check"]
    sby = cc["stress_by_component"]
    for comp in ("s_yz", "s_xz"):
        assert sby[comp]["norm_classical"] == 0.0
        assert 0.0 < sby[comp]["norm_neubernet"] < 0.02 * sby["s_yy"]["norm_neubernet"]
    #: the per-component reading the second run printed is the same numbers, named
    named = [sby[c]["rel_error"] for c in ("s_xx", "s_yy", "s_zz", "s_xy")]
    assert named == pytest.approx(cc["stress_rel_error_by_component"], rel=1e-12)
    flux = cc["ring_flux_by_component"]
    assert flux["t_x"]["rel_error"] > 0.2 and flux["t_y"]["rel_error"] < 0.05
    assert flux["t_theta"]["norm_classical"] == 0.0 and flux["t_theta"]["norm_neubernet"] > 0.0


def test_outcome_along_the_ring_the_donor_is_less_compact_than_the_physics(cs2):
    """At every radius from 1 to 14, on every port and every base, the donor
    keeps more than one and a half times the norm the elastic physics keeps
    outside the band -- and the physics itself is global along the ring."""
    fine = _fine(cs2)
    for s in cs2["neubernet"]["subjects"]:
        nn, fe = _rows(s), _rows(fine[s["port"]])
        for r in range(1, 15):
            assert nn[r] > 1.5 * fe[r], (s["label"], r)
    E = _rows(_nn(cs2, "tension-elastic", "ring:ux"))
    F = _rows(fine["ring:ux"])
    assert E[2] == pytest.approx(0.811, abs=5e-4) and F[2] == pytest.approx(0.228, abs=5e-4)
    assert E[4] == pytest.approx(0.691, abs=5e-4) and F[4] == pytest.approx(0.138, abs=5e-4)
    #: global by physics too: every sensor moves under one poke
    assert fine["ring:ux"]["support_reach"]["nonzero"] == fine["ring:ux"]["n"] == 29


def test_outcome_rank_reach_and_reciprocity_contrast(cs2):
    fo = cs2["classical"]["full_operator"]
    assert fo["spectrum"]["effective_rank_99"] == 72
    assert fo["reciprocity"] < 1e-12
    for label, f in cs2["neubernet"]["full_operator"].items():
        assert f["spectrum"]["effective_rank_99"] <= 7, label
        assert f["reciprocity"] > 0.9, label
        assert 0.95 < f["rel_diff_vs_fine_patch"] < 1.3, label

    def rstar(s, t):
        return next(x for x in s["r_star_relative_2norm"] if x["target"] == t)

    fine = _fine(cs2)
    assert rstar(fine["ring:ux"], 0.1)["r_star"] == 9
    assert rstar(fine["ring:uy"], 0.1)["r_star"] == 5
    assert rstar(_nn(cs2, "tension-elastic", "ring:ux"), 0.1)["trivial"] is True
    assert rstar(_nn(cs2, "tension-elastic", "ring:uy"), 0.1)["r_star"] == 24
    #: W177's null-replacement signature, per port
    for base, ports in cs2["neubernet"]["against_classical"].items():
        for port, v in ports.items():
            assert 0.99 < v["rel_diff_vs_fine_patch"] < 1.11, (base, port)


def test_outcome_across_the_seam_the_donor_does_not_decay_where_the_physics_does(cs2):
    rows = cs2["transverse"]["rows"]
    nn = [p for p in rows if p["subject"] == "neubernet"]
    fe = [p for p in rows if p["subject"] in ("elastic-patch-fine", "elastic-patch-coarse")]
    assert len(nn) == 6 and len(fe) == 6
    for p in nn:
        assert all(r["depth_star"] is None for r in p["depth_star_max"]["rows"]), p["poke"]
        assert p["max_by_depth"][20] / p["max_by_depth"][5] > 0.9, (p["base"], p["poke"])
    for p in fe:
        assert p["depth_star_max"]["rows"][0]["depth_star"] <= 20, (p["subject"], p["poke"])
        assert p["max_by_depth"][20] / p["max_by_depth"][5] < 0.2, (p["subject"], p["poke"])
    #: the face value sits on a hat knot: it moves with the mesh, the depth ratio does not
    peak = {(p["subject"], p["poke"]): p["peak"] for p in fe}
    assert peak[("elastic-patch-fine", "180deg u_x")] > 1.5 * peak[("elastic-patch-coarse", "180deg u_x")]


def test_outcome_the_response_is_its_training_manifolds(cs2):
    for base in ("tension-elastic", "tension-plastic", "tension-torsion-elastic"):
        t = _dirrow(cs2, base, "tension far field")
        assert t["cosine"] > 0.995 and t["rel_error"] < 0.09, base
    tt = _dirrow(cs2, "tension-torsion-elastic", "torsion far field")
    assert tt["cosine"] > 0.99 and tt["rel_error"] < 0.13
    for base in ("tension-elastic", "tension-plastic", "torsion-elastic", "tension-torsion-elastic"):
        one = _dirrow(cs2, base, "one sensor on u_y")
        assert one["norm_ratio"] < 0.2 and abs(one["cosine"]) < 0.01, base
        assert _dirrow(cs2, base, "one sensor on u_theta")["norm_ratio"] < 0.02, base
    #: at a zero-load base the direction of that load straddles SignNet's decision point
    te = _dirrow(cs2, "tension-elastic", "torsion far field")
    assert te["sign_torsion"]["base+av"] == -1.0 and te["sign_torsion"]["base-av"] == 1.0
    assert te["norm_ratio"] > 8.0 and te["minus"]["norm_ratio"] < 0.2
    #: and at that base the donor already emits a hoop flux the physics does not
    assert 0.0 < te["f0_hoop_block_norm"] < 0.02 * te["f0_in_plane_block_norm"]


def test_the_operator_is_measured_at_the_bottom_of_the_amplitude_v(cs2):
    lad = cs2["amplitude_ladder"]["neubernet tension-elastic ring:uy"]
    d = {r["amplitude"]: r["rel_disagreement_vs_ref"] for r in lad["rows"]}
    assert d[1e-2] == 0.0
    assert d[1.0] > d[1e-1] > d[1e-3] and d[1e-5] > d[1e-4] > d[1e-3]
    assert d[1e-3] < 0.01
    ref = next(r for r in lad["rows"] if r["amplitude"] == 1e-2)
    assert ref["cancellation_floor_normF"] < 1e-3 * ref["normF"]
    fe = cs2["amplitude_ladder"]["elastic-patch fine ring:uy"]
    assert max(r["rel_disagreement_vs_ref"] for r in fe["rows"]) < 1e-13


def test_W176_the_second_familys_far_field_is_recorded(cs2):
    def ffs(s, r):
        return next(f for f in s["far_field_spectrum"] if f["r"] == r)

    nn = _nn(cs2, "tension-elastic", "ring:ux")
    fe = _fine(cs2)["ring:ux"]
    a, b = ffs(nn, 8), ffs(fe, 8)
    assert a["share_top4"] > 0.97 and a["effective_rank_99"] == 6
    assert b["share_top4"] > 0.9 and b["effective_rank_99"] == 14
    assert ffs(nn, 20)["singular_values"][0] / nn["linear"]["norm2"] == pytest.approx(0.394, abs=2e-3)
    assert ffs(fe, 20)["singular_values"][0] / fe["linear"]["norm2"] == pytest.approx(0.0251, abs=5e-4)
    sv = ffs(fe, 20)["singular_values"]
    assert sv[0] == pytest.approx(sv[1], rel=1e-3)
