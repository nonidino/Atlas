"""Tier 20 -- CS-8, and whether a certificate travels with the expert.

`atlas/cases/reuse_probe.py` and `scripts/w106_reuse_probe.py`.  Five groups:

  * **the design** -- the seam taxonomy is a rule over the LAYOUT, assignable
    before any march runs, it keeps the two port groups apart, and it provides a
    replicate of the deep-wake regime so the label carries its own error bar;
  * **W105** -- every graph this case study builds is asserted to carry an
    `assembly.ProjectedAssembly`, and one that does not is refused. The rule
    cannot check this yet; the case study can, and Tier 19 measured what happens
    when nothing does;
  * **the probe state** -- a state that is not finite, or outside the band, is
    refused before it is probed, and one past the settled horizon is FLAGGED
    rather than refused, because the flag is the measurement;
  * **the statistics** -- ``spread`` reports both forms and declines the relative
    one where the mean is not usable (``visible_above`` is negative at every
    seam Tier 19 measured), and ``travel_verdict`` is three-valued;
  * **control ZERO, run live** -- at the freestream two seams the taxonomy calls
    different regimes hold the same field on the same port, so they must return
    the SAME operator. That is the control which, had it failed, would have made
    the whole SEAM factor uninterpretable.

The last group reads `out/w106/w106.json` and **re-derives the headline analysis
from the raw grid rows**, which is what "reproduced cold from the artifact"
means here: the tables in section 20 are not transcribed from a log, they are a
function of the saved cells, and this test is that function evaluated again.
Those tests skip when the artifact is absent.
"""

from __future__ import annotations

import json
import math
import os
import sys

# **Before numpy, and it is not optional.**  This module's control ZERO builds a
# real expert, which pulls the build repo in, which pulls torch in -- and torch's
# OpenMP runtime beside numpy's MKL aborts the interpreter inside
# `transfer.adjoint`.  Measured here: without this line the run dies with
# `Fatal Python error: Aborted` in a numpy matmul, not in torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np                                                    # noqa: E402
import pytest                                                         # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.assembly import ProjectedAssembly                          # noqa: E402
from atlas.capability import EllipticSubsolve                         # noqa: E402
from atlas.cases import reuse_probe as rp                             # noqa: E402
from atlas.cases import scaling_ladder as sl                          # noqa: E402
from atlas.cases import wake_array as wa                              # noqa: E402
from atlas.probe import ProbeBudget, assemble_seam                    # noqa: E402
from atlas.transfer import InterfaceSpace, SeamTransfer               # noqa: E402

ARTIFACT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "out", "w106", "w106.json")


def _rung(label):
    return {r.label: r for r in sl.ladder()}[label]


def _artifact():
    if not os.path.exists(ARTIFACT):
        pytest.skip(f"no CS-8 artifact at {ARTIFACT}; run "
                    "scripts/w106_reuse_probe.py")
    with open(ARTIFACT, encoding="utf-8") as fh:
        return json.load(fh)


# ===========================================================================
# 1. the design -- a rule over the layout, not a list
# ===========================================================================


def test_the_regime_is_derived_from_the_layout_with_no_state():
    """The label has to be assignable before the march, or it is a description
    of the answer rather than a factor of the design."""
    t = _rung("N12").tiling
    assert rp.regime_of(t, "x0r1_full") == "near-freestream"
    assert rp.regime_of(t, "x2r1_full") == "shallow-wake"
    assert rp.regime_of(t, "x2r0_full") == "deep-wake"
    assert rp.regime_of(t, "x2r2_full") == "deep-wake"
    assert rp.regime_of(t, "x0r0_bypass") == "bypass-clean"
    assert rp.regime_of(t, "x1r0_bypass") == "bypass-wake"
    # every label used is one the module documents
    for choice in rp.seam_menu(t):
        assert choice.regime in rp.REGIMES


def test_all_four_flow_regimes_the_brief_asks_for_exist_at_N12():
    """A shallow wake, a deep multi-turbine wake, a bypass region and a
    near-freestream seam -- and N12 is the smallest rung carrying all four."""
    got = {c.regime for c in rp.seam_menu(_rung("N12").tiling)}
    assert {"near-freestream", "shallow-wake", "deep-wake"} <= got
    assert {"bypass-clean", "bypass-wake"} <= got
    # and N6 does not, which is why the state factor and the seam factor are
    # measured at different rungs
    got6 = {c.regime for c in rp.seam_menu(_rung("N6").tiling)}
    assert "deep-wake" not in got6


def test_the_deep_wake_regime_has_a_replicate_and_it_is_in_another_row():
    """Two seams the RULE calls the same regime, in different rows: their
    disagreement is the label's own error bar, and it is measured rather than
    assumed to be zero."""
    deep = [c for c in rp.seam_menu(_rung("N12").tiling)
            if c.regime == "deep-wake"]
    assert len(deep) == 2
    assert {c.seam_id for c in deep} == {"x2r0_full", "x2r2_full"}
    assert deep[0].row != deep[1].row
    assert deep[0].face_x_D == deep[1].face_x_D          # same station
    assert deep[0].upstream_rotors == deep[1].upstream_rotors == 2


def test_comparable_key_keeps_the_two_port_groups_apart():
    """A ``full`` face is 128 cells and 33 modes and a ``bypass`` face is 96 and
    25; a spread across the two would be a spread across different spaces, not a
    measurement of the state."""
    menu = {c.seam_id: c for c in rp.seam_menu(_rung("N12").tiling)}
    full, byp = menu["x2r0_full"], menu["x1r0_bypass"]
    assert (full.n_cells, full.dim_M) == (wa.N, wa.modes_for(wa.N))
    assert byp.n_cells == wa.N - int(round(wa.ROTOR_D / wa.DX))
    assert byp.dim_M == wa.modes_for(byp.n_cells)
    assert full.dim_M != byp.dim_M
    assert full.comparable_key != byp.comparable_key
    # and every seam of one regime shares its group's key
    for seg in ("full", "bypass"):
        keys = {c.comparable_key for c in menu.values() if c.segment == seg}
        assert len(keys) == 1


def test_the_menu_carries_only_fluid_fluid_x_seams():
    """A y-seam's normal component is ``v`` -- a different operator wearing the
    same port type -- and a rotor seam is blind at every admissible beta_min by
    the cell Reynolds number (**W97**), so it has no threshold to move."""
    ids = {c.seam_id for c in rp.seam_menu(_rung("N12").tiling)}
    declared = {c.seam_id for c in wa.connections(_rung("N12").tiling)}
    assert ids <= declared
    assert not any(i.startswith("y") for i in ids)
    assert not any("_up" in i or "_down" in i for i in ids)
    assert any(i.startswith("y") for i in declared)         # they do exist


def test_shared_seams_are_the_cross_geometry_set():
    """The same seam ID in two tilings is the same declared port on the same
    face at the same place in the layout; what differs is how much graph is
    around it."""
    shared = rp.shared_seams(_rung("N6").tiling, _rung("N12").tiling)
    assert {"x0r1_full", "x0r0_bypass", "x1r0_bypass"} <= set(shared)
    m6 = {c.seam_id: c for c in rp.seam_menu(_rung("N6").tiling)}
    m12 = {c.seam_id: c for c in rp.seam_menu(_rung("N12").tiling)}
    for sid in shared:
        # same regime, same port, same station -- only the graph size moves
        assert m6[sid].regime == m12[sid].regime
        assert m6[sid].comparable_key == m12[sid].comparable_key
        assert m6[sid].face_x_D == m12[sid].face_x_D


def test_the_probe_state_schedule_straddles_the_settled_horizon():
    """0 is control ZERO; 110 is inside the horizon Tier 19 marched this column
    over and outside the 60 Tier 18 published at."""
    assert rp.PROBE_STEPS[0] == 0
    assert max(rp.PROBE_STEPS) > rp.SETTLED_STEPS
    assert max(rp.PROBE_STEPS) <= 120
    assert "N24" not in rp.PROBE_RUNGS        # never marched at this horizon
    assert "N1" not in rp.PROBE_RUNGS         # one window has no seam


# ===========================================================================
# 2. W105 -- the projection is asserted, not assumed
# ===========================================================================


def test_a_graph_without_a_projected_assembly_is_refused():
    """**W105.** `R10` moves the elliptic part out of the agent, `R10b` fixes the
    cadence and assumes it happens, and `R12` is silent on an all-exposed graph.
    Measured, the gap is real: that column is stable and not incompressible.  So
    this case study refuses the graph rather than probing states it cannot vouch
    for."""
    r = _rung("N6")
    u, v = np.ones(r.shape), np.zeros(r.shape)
    bare, _ = sl.build(u, v, r, kind="reference_exposed",
                       assembly_projection=False)
    assert bare.assembly_projection is None
    with pytest.raises(rp.ProbeStateRefused) as exc:
        rp.assert_projected(bare)
    assert "W105" in str(exc.value)


def test_build_carries_the_projection_and_the_exposed_elliptic_part():
    """The two halves of Tier 19's repair, on the record together: `R10`'s
    (the elliptic part out of the agent) and `R12`'s (the composition layer
    applies it once, after the blend)."""
    r = _rung("N6")
    u, v = np.ones(r.shape), np.zeros(r.shape)
    for kind in ("reference_exposed", "poseidon"):
        g, _ = rp.build(u, v, r, kind=kind)
        pou = rp.assert_projected(g)
        assert isinstance(pou, ProjectedAssembly)
        assert isinstance(g.partition_of_unity, ProjectedAssembly)
        assert g.assembly_projection is not None
        assert g.assembly_projection.constraint == "divergence-free"
    g, _ = rp.build(u, v, r, kind="reference_exposed")
    assert (g.agent("F00").capabilities.elliptic_subsolve
            is EllipticSubsolve.EXPOSED)


def test_the_classical_column_probed_here_is_the_one_that_reaches_110():
    """CS-8 probes the EXPOSED classical column and not the embedded one.

    Tier 19 section 19.6: the embedded column is not finite past macro-step 82
    at six windows, and a probe taken on a diverging trajectory measures the
    divergence.  The two kinds declare different `elliptic_subsolve`, which is
    the word `R10` turns on, so this is a difference the compile can see."""
    r = _rung("N6")
    u, v = np.ones(r.shape), np.zeros(r.shape)
    exposed, _ = rp.build(u, v, r, kind="reference_exposed")
    embedded, _ = sl.build(u, v, r, kind="reference")
    assert (exposed.agent("F00").capabilities.elliptic_subsolve
            is EllipticSubsolve.EXPOSED)
    assert (embedded.agent("F00").capabilities.elliptic_subsolve
            is EllipticSubsolve.EMBEDDED)


# ===========================================================================
# 3. the probe state -- refused, or flagged, and the difference matters
# ===========================================================================


def test_a_non_finite_probe_state_is_refused():
    r = _rung("N6")
    u, v = np.ones(r.shape), np.zeros(r.shape)
    u[3, 3] = np.nan
    with pytest.raises(rp.ProbeStateRefused):
        rp.probe_state_health(u, v, 40, r.tiling)


def test_a_state_outside_the_band_is_refused_and_says_why():
    """Tier 18 published a state on a diverging trajectory and section 19.6
    published a repair on one, the same day.  This is the check that runs before
    the probe rather than after the write-up."""
    r = _rung("N6")
    u, v = np.ones(r.shape), np.zeros(r.shape)
    u[10, 10] = 12.0
    with pytest.raises(rp.ProbeStateRefused) as exc:
        rp.probe_state_health(u, v, 40, r.tiling, band=3.0)
    assert "band" in str(exc.value)


def test_a_state_past_the_settled_horizon_is_flagged_not_refused():
    """At N=12 the divergence tail was still rising by 4% at macro-step 120, so
    a state reached in more than `SETTLED_STEPS` steps carries its divergence in
    the record.  A flag is a measurement; a refusal would hide it."""
    r = _rung("N6")
    u, v = np.ones(r.shape), np.zeros(r.shape)
    early = rp.probe_state_health(u, v, 30, r.tiling)
    late = rp.probe_state_health(u, v, 110, r.tiling)
    assert early["settled_horizon"] is True
    assert late["settled_horizon"] is False
    assert late["finite"] is True                     # flagged, not refused
    assert "div_rms" in late and "div_ratio" in late


def test_the_divergence_flag_fires_on_a_ratio_and_not_on_a_magnitude():
    """The comparison is against the same column's own freestream value, because
    the absolute number is a property of the instrument -- section 19.6's own
    caveat that the divergence measure and the projection do not share a
    kernel."""
    r = _rung("N6")
    u, v = np.ones(r.shape), np.zeros(r.shape)
    quiet = rp.probe_state_health(u, v, 30, r.tiling, div_ref=1e-3)
    assert quiet["div_flag"] is False
    # A SMOOTH ramp, not a cell-to-cell ripple: `wake_array.divergence_rms` is a
    # WIDE centred difference, so a period-2 oscillation is in its kernel and
    # reads as exactly zero divergence. That is section 19.6's own caveat -- the
    # instrument and the operator do not share a kernel -- and it is worth
    # pinning, because a test written the obvious way passes an aliased field
    # through the check that exists to catch a divergent one.
    u2 = np.ones(r.shape) + 0.1 * np.arange(r.shape[1]) / r.shape[1]
    assert wa.divergence_rms(np.ones(r.shape) + 0.05 * (
        np.arange(r.shape[1]) % 2), v) == 0.0
    loud = rp.probe_state_health(u2, v, 30, r.tiling, div_ref=1e-6)
    assert loud["div_ratio"] > 10.0
    assert loud["div_flag"] is True


# ===========================================================================
# 4. the statistics
# ===========================================================================


def test_spread_reports_both_forms_and_declines_the_useless_one():
    """``visible_above`` is NEGATIVE at every seam Tier 19 measured, so a
    relative spread taken about a mean that crosses zero is meaningless.  The
    absolute range is what gets compared with the floor, which is what makes it
    the load-bearing one."""
    s = rp.spread([1.0, 2.0, 3.0])
    assert s["abs_range"] == 2.0
    assert s["max_over_min"] == 3.0
    assert s["range_over_mean"] == pytest.approx(1.0)
    zero_mean = rp.spread([-1.0, 1.0])
    assert zero_mean["abs_range"] == 2.0
    assert zero_mean["range_over_mean"] is None
    assert zero_mean["mean_abs"] == 1.0
    assert rp.spread([None, float("nan")])["n"] == 0
    assert rp.spread([])["abs_range"] is None


def test_travel_verdict_is_three_valued_and_names_the_undecided_case():
    floor = 1e-6
    assert rp.travel_verdict({"abs_range": 5e-7}, floor)["verdict"] == "travels"
    assert rp.travel_verdict({"abs_range": 5e-6}, floor)["verdict"] == "undecided"
    big = rp.travel_verdict({"abs_range": 1e-3}, floor)
    assert big["verdict"] == "state-dependent"
    assert big["ratio_to_floor"] == pytest.approx(1000.0)
    assert rp.travel_verdict({"abs_range": None}, floor)["verdict"] == "unmeasured"
    assert rp.travel_verdict({"abs_range": 1.0}, None)["verdict"] == "unmeasured"


def test_an_exactly_reproducible_reprobe_makes_any_movement_visible():
    """A deterministic expert re-probed identically can return a floor of
    exactly zero.  Then any nonzero range is a movement, and saying so is more
    honest than dividing by zero or silently substituting an epsilon."""
    out = rp.travel_verdict({"abs_range": 1e-9}, 0.0)
    assert out["verdict"] == "state-dependent"
    assert out["ratio_is_infinite"] is True
    assert out["ratio_to_floor"] is None
    assert rp.travel_verdict({"abs_range": 0.0}, 0.0)["verdict"] == "travels"


def test_the_discriminating_axes_are_declared_once():
    """The driver, the tests and the write-up have to mean the same thing by
    *the certificate's discriminating quantities*, so the list lives on the
    reading and nowhere else."""
    axes = rp.CertificateReading.DISCRIMINATING
    assert "delta_norm" in axes           # ||Lambda_expert - Lambda_ref||
    assert "Xi_block" in axes             # the composability index
    assert "visible_above" in axes and "fails_above" in axes   # the beta_min pair
    assert "beta" in axes and "block_norm" in axes
    row = rp.CertificateReading(rung="N6", seam_id="s", regime="deep-wake",
                                segment="full", step=0, target="F00",
                                beta=1.0, delta_norm=2.0)
    assert set(row.as_dict()["discriminating"]) == set(axes)


def test_verdict_at_is_W81s_verdict_function_written_down():
    """`blind` and `passes` are each monotone in ``beta_min``, so the whole
    verdict function is the two thresholds -- which is what W81 closed on.  That
    makes the axis sweepable at every cell of the grid with no re-probe, so
    *does the verdict move* is answered on the same footing as *do the numbers
    move* rather than only at the one tolerance a driver passed."""
    vis, fail = -0.10, 0.20
    assert rp.verdict_at(vis, fail, -0.5) == "blind"
    assert rp.verdict_at(vis, fail, -0.10) == "blind"      # boundary is blind
    assert rp.verdict_at(vis, fail, 0.0) == "admit"
    assert rp.verdict_at(vis, fail, 0.20) == "admit"       # boundary is admit
    assert rp.verdict_at(vis, fail, 0.21) == "refuse"
    assert rp.verdict_at(None, fail, 0.0) is None
    assert rp.verdict_at(vis, None, 0.0) is None


def test_the_admissible_axis_starts_at_zero_and_the_thresholds_can_be_negative():
    """Tier 19 section 19.8 measured BOTH thresholds negative at every rung, so
    the whole non-negative axis is `refuse` and the swap is never blind.  That is
    the informative case, and it is why `delta_over_beta` above 1 is the same
    statement as `refuse` at every admissible tolerance."""
    assert min(rp.BETA_MIN_AXIS) == 0.0
    beta, delta, block = 0.2147, 0.2154, 0.2154
    vis, fail = beta - block, beta - delta
    assert vis < 0 and fail < 0
    assert delta / beta > 1.0
    assert {rp.verdict_at(vis, fail, bm) for bm in rp.BETA_MIN_AXIS} == {"refuse"}


def test_the_decision_margin_is_on_the_declared_axes():
    """``||Delta||/beta`` is the one derived number that says how CLOSE the
    verdict came to answering differently, so it is a discriminating quantity
    rather than a note."""
    assert "delta_over_beta" in rp.CertificateReading.DISCRIMINATING


# ===========================================================================
# 5. control ZERO, run live -- the freestream, where the answer is known
# ===========================================================================


def _transfer_for(graph, seam_id):
    conn = graph.connection(seam_id)
    pro, dims = {}, []
    for agent_id, name in (conn.a, conn.b):
        p = graph.agent(agent_id).port(name).prolongation
        pro[agent_id] = p
        dims.append(p.dim_M)
    assert dims[0] == dims[1]
    return SeamTransfer(seam_id=seam_id, port_type=conn.port_type,
                        space=InterfaceSpace(seam_id=seam_id, dim=dims[0],
                                             note="derived"),
                        prolongations=pro)


def _op_at_freestream(r, graph, experts, seam_id):
    conn = graph.connection(seam_id)
    tr = _transfer_for(graph, seam_id)
    coeffs = []
    for agent_id, name in (conn.a, conn.b):
        base_V = np.asarray(experts[agent_id].probe_base(name), dtype=float)
        coeffs.append(tr.prolongations[agent_id].adjoint(tr.space) @ base_V)
    return assemble_seam(graph, conn, tr, budget=ProbeBudget(),
                         expected_null_dim=conn.expected_null_dim,
                         probe_state="control ZERO: freestream",
                         seam_base=0.5 * (coeffs[0] + coeffs[1]))


def test_control_zero_two_regimes_are_the_same_operator_at_the_freestream():
    """**The control that could have redirected the whole search.**

    At the freestream every window holds the same uniform field, every ``full``
    port is the same 128-cell face with the same 33 modes, and the reference
    solver is deterministic -- so a seam the taxonomy calls `near-freestream`
    and one it calls `deep-wake` are the same probe of the same expert at the
    same state, written twice.  A spread here would mean the regime labels are
    reading the PORT and not the FLOW, and every number in the seam factor would
    be uninterpretable.

    Asserted bit-for-bit rather than to a tolerance, which is what makes it a
    control rather than a check.
    """
    r = _rung("N12")
    u, v = np.ones(r.shape), np.zeros(r.shape)
    experts = sl.make_experts(u, v, r, "reference_exposed")
    graph, _ = rp.build(u, v, r, kind="reference_exposed", experts=experts)
    rp.assert_projected(graph)
    a = _op_at_freestream(r, graph, experts, "x0r1_full")     # near-freestream
    b = _op_at_freestream(r, graph, experts, "x2r0_full")     # deep-wake
    assert a.dim_M == b.dim_M
    assert np.array_equal(a.S, b.S)
    assert a.beta == b.beta
    assert a.kappa == b.kappa


# ===========================================================================
# 6. the artifact, re-derived cold
# ===========================================================================


def test_the_artifact_records_a_projection_on_every_compiled_graph():
    """**W105 asserted on every graph**, which is the prompt's own requirement
    and the thing the rules cannot yet check."""
    art = _artifact()
    compiles = art.get("compile") or {}
    if not compiles:
        pytest.skip("compile stage not present in the artifact")
    for rung, row in compiles.items():
        for kind, col in row["columns"].items():
            assert col.get("W105_asserted") is True, (rung, kind)
            assert col.get("assembly") == "ProjectedAssembly", (rung, kind)
            assert col.get("projection_stage") is not None
    exposed = [row["columns"]["reference_exposed"] for row in compiles.values()]
    assert all(c["elliptic_subsolve"] == "exposed" for c in exposed)


def test_every_probe_state_in_the_artifact_passed_its_health_check():
    art = _artifact()
    march = art.get("march") or {}
    if not march:
        pytest.skip("march stage not present in the artifact")
    for rung, rec in march.items():
        for step, health in rec.get("health", {}).items():
            assert health["finite"] is True, (rung, step)
            assert health["u_max"] <= 3.0, (rung, step)


def test_the_headline_analysis_reproduces_cold_from_the_saved_grid():
    """Every number in section 20 is a FUNCTION of the saved cells.

    This re-runs that function on `out/w106/w106.json`'s raw ``grid`` and
    ``floor`` rows and requires it to return what the artifact's ``analysis``
    block already holds.  It is what *reproduced cold from a saved artifact*
    means for a derived table: not that the log was transcribed correctly, but
    that the reduction is deterministic and still in the code.
    """
    art = _artifact()
    if not art.get("analysis") or not art.get("grid"):
        pytest.skip("analysis stage not present in the artifact")
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import w106_reuse_probe as drv

    again = drv.stage_analysis(art["grid"], art.get("floor", []))
    for factor in ("seam", "state", "geometry"):
        saved = art["analysis"]["headline"].get(factor, {})
        fresh = again["headline"].get(factor, {})
        assert set(saved) == set(fresh), factor
        for f, row in saved.items():
            assert row["verdict"] == fresh[f]["verdict"], (factor, f)
            if row["median_abs_range"] is None:
                assert fresh[f]["median_abs_range"] is None
            else:
                assert fresh[f]["median_abs_range"] == pytest.approx(
                    row["median_abs_range"], rel=1e-12), (factor, f)
    for f, val in art["analysis"]["floor"]["worst_over_all"].items():
        got = again["floor"]["worst_over_all"][f]
        if val is None:
            assert got is None
        else:
            assert got == pytest.approx(val, rel=1e-12)


def test_the_zero_control_in_the_artifact_actually_returned_zero():
    """If this fails the seam factor is uninterpretable and section 20 says so
    rather than reporting a spread."""
    art = _artifact()
    zero = art.get("zero_control")
    if not zero:
        pytest.skip("zero control not present in the artifact")
    for key, row in zero.items():
        assert row["worst_abs_range"] is not None, key
        assert row["worst_abs_range"] == 0.0, (
            f"{key}: two seams of the same port group disagree at the "
            f"freestream by {row['worst_abs_range']:.3e}; the regime labels are "
            "reading the port and not the flow")


def test_the_floor_is_measured_before_any_spread_is_called_a_result():
    art = _artifact()
    floor = art.get("floor")
    if not floor:
        pytest.skip("floor control not present in the artifact")
    assert len(floor) >= 1
    for row in floor:
        assert row["verdict_stable"] is True, row["seam_id"]
        for f in rp.CertificateReading.DISCRIMINATING:
            d = row["abs_delta"].get(f)
            assert d is None or math.isfinite(d)


def test_the_swept_verdict_is_reported_beside_the_called_one():
    """The certificate as the driver calls it is given no tolerance, so it
    returns `admit-uncertified` by construction (W81).  A constant is not a
    stability result, and the artifact has to say so rather than let a reader
    take it for one."""
    art = _artifact()
    vs = (art.get("analysis") or {}).get("verdict_stability")
    if not vs:
        pytest.skip("analysis stage not present in the artifact")
    for rung, row in vs.items():
        assert "as_called" in row and "swept" in row, rung
        assert "by construction" in row["as_called"]["note"]
        assert set(row["swept"]) == {repr(bm) for bm in rp.BETA_MIN_AXIS}


def test_the_replicate_factor_exists_and_is_a_same_regime_pair():
    """The floor is exact on this pipeline, so *how many floors* separates
    nothing.  The replicate is the yardstick that does: two seams the RULE calls
    the same regime, whose disagreement is what the label fails to pin down."""
    art = _artifact()
    rep = ((art.get("analysis") or {}).get("per_factor") or {}).get("replicate")
    if rep is None:
        pytest.skip("analysis stage not present in the artifact")
    assert rep, "no same-regime replicate was found in the grid"
    for key, row in rep.items():
        assert len(row["seams"]) >= 2, key
        assert len(set(row["seams"])) == len(row["seams"]), key
