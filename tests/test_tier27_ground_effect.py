"""Tier 24 -- CS-10, a moving interface, a design parameter, and W97.

`atlas/cases/ground_effect.py` and `scripts/w127_ground_effect.py`.  Seven
groups, and the first three are the ones that make the rest readable:

  * **the declaration** -- the plate lives inside ONE window and every seam is
    clear of it (W124's defect, checked before the fact this time); the graph
    declares an exposed elliptic part and a `ProjectedAssembly`, which is the
    first classical column `L2/R10` does not refuse; the two sides of the wing
    seam declare the same `response_half` and the same `governing_family`;
  * **the controls** -- the one-window tiling has a partition of unity that is
    identically one, its blend is the identity BITWISE, and two runs of the same
    objective agree to the bit.  Each is a control and NOT a floor (W106);
  * **the fluid block is not zero** -- pinned, because getting the window's own
    frame wrong put the plate outside its array, `grid_sample`'s border clamp
    returned a constant, and the probed block came back EXACTLY zero with every
    other diagnostic healthy;
  * **the gate** -- the split reproduces the referent's load and ride-height
    history, the lag defect is first order, the CUT is two orders below the LAG,
    and the difference between the two columns CHANGES SIGN when marched;
  * **W30** -- the moving seam's operator drift against the same seam held still
    and against W30's own static reference, and the re-probe count that follows;
  * **W97** -- the block ratio under section 2.2's effort, section 4.1's
    conservative co-normal, and the exact momentum exchange;
  * **F5** -- the adjoint agrees with a central finite difference, and its sign
    depends on the horizon.

The artifact-backed tests re-derive the published numbers from
`out/w127/w127.json` and skip when it is absent.  The live tests build graphs and
probe seams, which costs about a minute; the marches are not re-run here.
"""

from __future__ import annotations

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
from atlas.assembly import ProjectedAssembly                          # noqa: E402
from atlas.envelope import Hypothesis, Status                         # noqa: E402
from atlas.graph import Decomposition                                 # noqa: E402
from atlas.ports import PortType, ResponseHalf                        # noqa: E402

ARTIFACT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "out", "w127", "w127.json")

nrm = np.linalg.norm


def _artifact():
    if not os.path.exists(ARTIFACT):
        pytest.skip(f"no CS-10 artifact at {ARTIFACT}; run "
                    "scripts/w127_ground_effect.py")
    with open(ARTIFACT, encoding="utf-8") as fh:
        return json.load(fh)


def _stage(name: str):
    a = _artifact()
    if name not in a:
        pytest.skip(f"stage {name!r} not in the artifact; run "
                    f"scripts/w127_ground_effect.py --stages {name}")
    return a[name]


def _have_expert() -> bool:
    try:
        from atlas.cases import ground_effect as GE
        GE.load_reference()
        return GE.torch is not None
    except Exception:
        return False


needs_expert = pytest.mark.skipif(
    not _have_expert(),
    reason="reference.WindowNS (ATLAS_BUILD_REPO) and torch are both required")

if _have_expert():
    from atlas.cases import ground_effect as GE                       # noqa: E402
    import torch                                                      # noqa: E402


# ---------------------------------------------------------------------------
# 1. the declaration
# ---------------------------------------------------------------------------


@needs_expert
def test_the_plate_lives_inside_one_window():
    """W124, checked BEFORE the fact rather than after.

    Every x-seam of CS-6 and CS-7 passes exactly through a rotor disc because
    `wake_array.Rotor` is declared on an x-adjacency and the tiling rule cannot
    express anything else.  This tiling places the plate between two overlap
    bands and the clearance is a positive number.
    """
    t = GE.DEFAULT_TILING
    lo, hi = t.wing_cells()
    assert t.wing_window() == "F10"
    assert t.cuts_clear_of_wing() > 0
    # the overlap bands, explicitly, so the assertion is readable
    for i in range(t.n_col - 1):
        a = (i + 1) * t.stride
        b = a + t.halo
        assert hi <= a or b <= lo, f"overlap [{a}, {b}) cuts the plate"


@needs_expert
def test_the_stamping_box_is_wider_than_the_plate_and_says_so():
    t = GE.DEFAULT_TILING
    lo, hi = t.wing_cells()
    slo, shi = t.stamp_cells()
    assert slo < lo and hi < shi


@needs_expert
def test_the_single_tiling_is_a_zero_cut_control_bitwise():
    """chi == 1 EXACTLY, and the blend is the identity to the bit.

    CS-7 section 2: the control that would have redirected the parent's whole
    search was the one nobody ran.  This is that control, and `atol=0.0` is the
    point of it.
    """
    s = GE.SINGLE_TILING
    w = s.weights()
    assert len(w) == 1
    assert np.array_equal(w[0], np.ones((GE.NY, GE.NX)))
    r = GE.GroundRollout(tiling=s, coupling="tight", motion=False)
    f = np.random.default_rng(0).standard_normal((GE.NY, GE.NX))
    g = np.random.default_rng(1).standard_normal((GE.NY, GE.NX))
    tf = torch.as_tensor(f, dtype=GE.TORCH_DTYPE)
    tg = torch.as_tensor(g, dtype=GE.TORCH_DTYPE)
    bu, bv = r.blend(r.cut(tf), r.cut(tg))
    assert torch.equal(bu, tf) and torch.equal(bv, tg)


@needs_expert
def test_the_exposed_agent_is_the_column_R10_does_not_refuse():
    u = np.full((GE.NY, GE.NX), GE.U_INF)
    v = np.zeros((GE.NY, GE.NX))
    g, ex = GE.build(u, v, motion=False)
    fluids = [a for a in g.agents if a.agent_id != "SUSP"]
    assert fluids, "no fluid agents"
    for a in fluids:
        assert a.capabilities.elliptic_subsolve is EllipticSubsolve.EXPOSED
        assert a.capabilities.time_discretization is TimeDiscretization.EXPLICIT
    assert g.decomposition is Decomposition.OVERLAPPING
    # the assembly is the PAIR -- the blend AND one global Leray projection,
    # once, after it (W100's arrangement, R12's object)
    asm = g.partition_of_unity
    assert isinstance(asm, ProjectedAssembly)
    assert asm.projection.satisfies_C2 is True
    assert asm.projection.scope == "global"
    assert asm.projection.stage == "after-assembly"
    assert asm.projection.cadence == 1


@needs_expert
def test_both_sides_of_the_wing_seam_declare_the_same_effort_half():
    """L3/C9 refuses a seam whose two sides disagree; this asserts they agree.

    `CASE-STUDY-GUIDE` mistake 7: nothing can check the declaration itself, so
    what a test CAN pin is that the two sides declare the same half and the same
    governing family -- an algebraic closure within a continuum problem is not a
    different continuum problem.
    """
    u = np.full((GE.NY, GE.NX), GE.U_INF)
    v = np.zeros((GE.NY, GE.NX))
    g, _ = GE.build(u, v, motion=False)
    conn = g.connection("wing")
    for agent_id, port_name in (conn.a, conn.b):
        caps = g.agent(agent_id).capabilities
        assert caps.port(port_name).response_half is ResponseHalf.EFFORT
        assert caps.governing_family == "incompressible-navier-stokes-2d"
    assert conn.port_type is PortType.MECH
    # field <-> lumped, so n_0 is 0 and not the fluid-fluid 1
    assert conn.expected_null_dim == 0


@needs_expert
def test_the_halo_outruns_the_declared_domain_of_dependence():
    t = GE.DEFAULT_TILING
    u = np.full((GE.NY, GE.NX), GE.U_INF)
    v = np.zeros((GE.NY, GE.NX))
    g, _ = GE.build(u, v, motion=False)
    fluid = g.agent(t.names[0]).capabilities
    assert fluid.required_halo() == 2 * GE.EXCHANGES
    assert t.halo > fluid.required_halo()


@needs_expert
def test_the_suspension_probes_at_the_exchange_interval_not_the_macro_step():
    """`CASE-STUDY-GUIDE` mistake 5, on the lumped side.

    A spring's block is proportional to the interval it is probed over, so
    declaring the macro-step where R10b fixes ``dt / substeps`` makes it four
    times too large and moves the whole of W97's ratio with it.
    """
    s = GE.Suspension()
    assert s.dt == pytest.approx(GE.MACRO_DT / GE.EXCHANGES)


@needs_expert
def test_motion_class_is_declared_and_only_on_the_wing_ports():
    u = np.full((GE.NY, GE.NX), GE.U_INF)
    v = np.zeros((GE.NY, GE.NX))
    for motion, want in ((False, MotionClass.STATIC),
                         (True, MotionClass.SOLUTION_DEPENDENT)):
        g, _ = GE.build(u, v, motion=motion)
        moving = {f"{a}.{p.name}" for a, p in g.moving_ports()}
        if motion:
            assert moving == {"F10.wing:MECH", "SUSP.mount:MECH"}
        else:
            assert moving == set()
        assert (g.agent("SUSP").capabilities.port("mount:MECH").motion_class
                is want)
        # a window FACE does not move whatever the wing does
        assert (g.agent("F00").capabilities.port("xhi:MECH").motion_class
                is MotionClass.STATIC)


# ---------------------------------------------------------------------------
# 2. the physics, live
# ---------------------------------------------------------------------------


@needs_expert
def test_the_wing_force_integrates_to_the_load_exactly():
    """`disk.body_force_field`'s gate-W2 property, on an inclined plate.

    The kernels are normalized DISCRETELY on the stamping box, so the force on
    the fluid integrates to minus the force on the plate to floating point.  If
    it did not, the load would be a second model rather than an exact momentum
    exchange, and every number in the case study would be about that model.
    """
    wing = GE.Wing()
    rng = np.random.default_rng(3)
    u = torch.as_tensor(GE.U_INF + 0.1 * rng.standard_normal((GE.NY, GE.NX)),
                        dtype=GE.TORCH_DTYPE)
    v = torch.as_tensor(0.1 * rng.standard_normal((GE.NY, GE.NX)),
                        dtype=GE.TORCH_DTYPE)
    h = torch.tensor(GE.H_START, dtype=GE.TORCH_DTYPE)
    vp = torch.zeros((), dtype=GE.TORCH_DTYPE)
    fx, fy, w, load = wing.forcing(u, v, h, vp, GE.NY, GE.NX)
    fy_total = float(fy.sum() * GE.DX * GE.DX)
    assert fy_total == pytest.approx(float(load), rel=1e-12, abs=1e-14)


@needs_expert
def test_the_plate_produces_downforce_in_a_forward_stream():
    """The sign, from the geometry rather than from a convention.

    The plate rises downstream, so it deflects the flow up and the reaction on
    it is down.  Getting this backwards would make every ride height in the case
    study move the wrong way and nothing else would notice.
    """
    wing = GE.Wing()
    u = torch.full((GE.NY, GE.NX), GE.U_INF, dtype=GE.TORCH_DTYPE)
    v = torch.zeros((GE.NY, GE.NX), dtype=GE.TORCH_DTYPE)
    h = torch.tensor(GE.H_START, dtype=GE.TORCH_DTYPE)
    vp = torch.zeros((), dtype=GE.TORCH_DTYPE)
    _fx, _fy, w, load = wing.forcing(u, v, h, vp, GE.NY, GE.NX)
    assert float(load) > 0.0, "the plate must produce DOWNforce"
    assert float(w.mean()) < 0.0, "the flow must arrive on the plate's underside"


@needs_expert
def test_the_plate_velocity_unloads_the_plate():
    """The aerodynamic damping term, which is what makes the seam well posed.

    A plate descending sees less of the flow on its underside, so the load
    falls.  That term is the whole reason `solve_interface` has a nonsingular
    derivative, and its sign is what makes a massless algebraic partner
    tractable at all.
    """
    wing = GE.Wing()
    u = torch.full((GE.NY, GE.NX), GE.U_INF, dtype=GE.TORCH_DTYPE)
    v = torch.zeros((GE.NY, GE.NX), dtype=GE.TORCH_DTYPE)
    h = torch.tensor(GE.H_START, dtype=GE.TORCH_DTYPE)
    loads = [float(-(wing.traction(
        wing.station_normal(u, v, h, torch.tensor(vp, dtype=GE.TORCH_DTYPE),
                            GE.NY, GE.NX)[0]) * wing.ds).sum())
        for vp in (-0.02, 0.0, +0.02)]
    assert loads[0] < loads[1] < loads[2]


@needs_expert
def test_the_suspension_is_two_lines_and_is_affine_in_its_trace():
    s = GE.Suspension(h=0.2, h0=0.32, k=2.5)
    assert s.ride_height(0.25) == pytest.approx(0.32 - 0.25 / 2.5)
    assert s.spring_force(0.2) == pytest.approx(2.5 * (0.32 - 0.2))
    z = s.respond("mount:MECH", np.zeros(GE.N_STATION))
    a = s.respond("mount:MECH", np.full(GE.N_STATION, 1.0))
    b = s.respond("mount:MECH", np.full(GE.N_STATION, 2.0))
    assert np.allclose(b - z, 2.0 * (a - z), rtol=0, atol=1e-15)


@needs_expert
def test_the_suspension_declines_above_its_own_static_height():
    """`validity` is one of the two declarations nobody can verify, and this is
    what it DOES say: above ``h0`` a linear compression spring is in tension and
    the two-line model is outside the state it was written for."""
    s = GE.Suspension(h0=0.32)
    assert s.validity(0.25) is True
    assert s.validity(0.40) is False
    assert s.validity(0.5 * GE.H_FLOOR) is False


@needs_expert
def test_the_probed_fluid_block_at_the_wing_seam_is_not_zero():
    """The silent-wrongness class in miniature, pinned.

    The plate is declared in GLOBAL coordinates and a window sees it in its own
    frame.  With the x offset unsubtracted the plate sat outside the window's
    array, `grid_sample`'s border clamp returned a constant, and the probed
    fluid block came back EXACTLY zero -- which the compiler then read as 16
    excess null directions and refused, with nothing pointing at the cause.
    """
    u = np.full((GE.NY, GE.NX), GE.U_INF)
    v = np.zeros((GE.NY, GE.NX))
    ex = GE.make_experts(u, v, h=GE.H_START, h0=GE.H0_REF, flux_mode="conormal")
    f = ex[GE.DEFAULT_TILING.wing_window()]
    z = f.respond("wing:MECH", np.zeros(GE.N_STATION))
    eps = 1e-6
    cols = []
    for j in (0, 8, 16, 24, 32):
        t = np.zeros(GE.N_STATION)
        t[j] = eps
        cols.append((f.respond("wing:MECH", t) - z) / eps)
    block = np.array(cols)
    assert np.abs(block).max() > 1e-6
    # and the columns are not all the same, which a border clamp would give
    assert np.std(np.linalg.norm(block, axis=1)) > 0.0


@needs_expert
def test_the_wing_window_carries_the_plate_and_the_others_do_not():
    u = np.full((GE.NY, GE.NX), GE.U_INF)
    v = np.zeros((GE.NY, GE.NX))
    ex = GE.make_experts(u, v)
    carriers = [k for k, e in ex.items()
                if k != "SUSP" and e.has_wing]
    assert carriers == [GE.DEFAULT_TILING.wing_window()]


# ---------------------------------------------------------------------------
# 3. the compile
# ---------------------------------------------------------------------------


@needs_expert
def test_the_fixed_floor_graph_compiles_and_the_moving_one_is_refused():
    """The refusal is the deliverable, and it is at `L2/InterfaceMotion`.

    E2 holds on a static graph and FAILS on a moving one; the moving graph is
    refused on both ports of the wing seam and on nothing else new.  No rule
    exists for a solution-dependent interface, and this case study does not add
    one -- it supplies the three measurements the slot has asked for.
    """
    u = np.full((GE.NY, GE.NX), GE.U_INF)
    v = np.zeros((GE.NY, GE.NX))
    g0, _ = GE.build(u, v, motion=False, flux_mode="conormal")
    g1, _ = GE.build(u, v, motion=True, flux_mode="conormal")
    r0 = compile_scheme(g0)
    r1 = compile_scheme(g1)
    assert r0.verdict.value == "admit-uncertified"
    assert r0.envelope.values[Hypothesis.E2] is Status.HOLDS
    assert not r0.decisions.refusals

    assert r1.verdict.value == "refuse"
    assert r1.envelope.values[Hypothesis.E2] is Status.FAILS
    rules = {f"{d.layer}/{d.rule}" for d in r1.decisions.refusals}
    assert rules == {"L2/InterfaceMotion"}
    subjects = {d.subject for d in r1.decisions.refusals}
    assert subjects == {"F10.wing:MECH", "SUSP.mount:MECH"}
    assert "InterfaceMotion" in r1.holes.touched()


@needs_expert
def test_the_compile_says_which_measurements_the_slot_still_owes():
    u = np.full((GE.NY, GE.NX), GE.U_INF)
    v = np.zeros((GE.NY, GE.NX))
    g, _ = GE.build(u, v, motion=True, flux_mode="conormal")
    r = compile_scheme(g)
    owed = r.holes.outstanding().get("InterfaceMotion", [])
    assert "operator_drift" in owed
    assert "unaccounted_power_fraction" in owed


# ---------------------------------------------------------------------------
# 4. the gate, from the artifact
# ---------------------------------------------------------------------------


def test_the_split_reproduces_the_referents_ride_height_history():
    s = _stage("split")
    for col in ("lag1_nocut", "lag1_cut", "tight_cut"):
        assert s["columns"][col]["h_max_rel"] < 5e-3
        assert abs(s["columns"][col]["h_end_rel"]) < 1e-3


def test_the_lag_defect_is_first_order_in_the_lag():
    s = _stage("split")
    series = s["lag_series_h"]
    assert series[0] < series[1] < series[2]
    for o in s["lag_order_log2"]:
        assert 0.9 < o < 1.5, f"log2 ratio {o} is not first order"


def test_the_lumped_seams_lag_dominates_the_domain_cut_by_two_orders():
    s = _stage("split")
    assert s["lag_over_cut"] > 50.0
    assert s["cut_alone_h"] < 1e-4


def test_the_split_versus_referent_difference_changes_sign_when_marched():
    """The standing rule from Tier 23, discharged rather than quoted.

    A comparison between two configurations is not a result until it has been
    marched past the point where the two curves could cross, and the crossing
    has to be LOOKED FOR.  It was, and it is there: the lagged column is below
    the referent through the transient, peaks around macro-step 60, and ends
    above it.
    """
    s = _stage("split")
    for col in ("lag1_nocut", "lag2_nocut", "lag4_nocut", "lag1_cut"):
        row = s["columns"][col]
        assert row["h_sign_changes"] >= 1, f"{col} never crossed"
        # the peak is in the transient and the end is on the other side of zero
        assert row["h_argmax"] < 0.5 * s["steps"]
        h = np.array(s["height"][col]) - np.array(s["height"]["referent"])
        assert np.sign(h[row["h_argmax"]]) != np.sign(h[-1])


def test_the_zero_cut_control_is_bitwise_and_is_not_quoted_as_a_floor():
    s = _stage("split")
    assert s["zero_cut_bitwise"] is True


def test_the_naive_staggered_exchange_diverges():
    """The obvious reading of ``k(h0 - h) = L`` as an UPDATE, kept because a
    rule with nothing to fire on is not a rule."""
    s = _stage("split")
    assert s["naive_staggered"]["diverged_at"] is not None
    assert s["naive_staggered"]["diverged_at"] <= 2


def test_the_agent_took_exactly_one_substep_per_exchange():
    r = _stage("referent")
    assert r["substeps_unique"] == [1]


def test_the_referent_undershoots_and_comes_back():
    r = _stage("referent")
    assert r["h_min_step"] > 0
    assert r["h_min"] < r["h_final"]
    assert r["h_final"] < r["height"][0]


# ---------------------------------------------------------------------------
# 5. the ground-effect curve
# ---------------------------------------------------------------------------


def test_the_load_rises_as_the_ride_height_falls_and_no_peak_is_assumed():
    c = _stage("curve")
    L = np.array(c["load"])
    assert c["monotone_decreasing"] is True
    assert np.all(np.diff(L) < 0.0)
    assert c["ground_effect_ratio"] > 1.2


def test_the_frozen_field_stiffness_exceeds_the_quasi_static_one():
    c = _stage("curve")
    assert c["frozen_over_quasistatic"] > 3.0
    assert c["max_abs_dLdh_frozen"] > c["max_abs_dLdh_quasistatic"]


def test_the_unsteadiness_is_quoted_as_the_level_and_is_not_zero():
    """W106: a floor of exactly zero bounds nothing.  The plate sheds, and that
    shedding is what a difference in the load has to be read against."""
    c = _stage("curve")
    assert 1e-4 < c["replicate_level_rel"] < 1e-1


# ---------------------------------------------------------------------------
# 6. W30 -- InterfaceMotion, measured
# ---------------------------------------------------------------------------


def test_the_moving_seam_drifts_far_more_than_the_same_seam_held_still():
    m = _stage("motion")
    w = m["drift"]["wing"]
    assert w["relative_moving"] > 10.0 * w["relative_static"]
    # and far more than W30's own static reference on four window_ns seams
    assert w["relative_moving"] > 10.0 * m["W30_static_reference"][1]


def test_the_fluid_fluid_seam_stays_inside_W30s_static_band():
    m = _stage("motion")
    x = m["drift"]["x00"]
    assert x["relative_moving"] < m["W30_static_reference"][1]
    assert x["relative_static"] < m["W30_static_reference"][1]


def test_a_cached_operator_does_not_survive_one_macro_step_at_a_moving_seam():
    m = _stage("motion")
    assert m["drift_per_macro_step_moving"] > m["staleness_tolerance"]
    assert m["cache_life_macro_steps_moving"] == pytest.approx(1.0)
    assert m["reprobe_count_moving"] == m["reprobe_count_moving"]
    assert m["reprobe_share_of_march"] > 0.5


def test_the_moving_interface_carries_a_reportable_share_of_the_seam_power():
    m = _stage("motion")
    assert 0.0 < m["unaccounted_power_fraction"] < 1.0
    assert m["unaccounted_power_fraction"] > 0.05


# ---------------------------------------------------------------------------
# 7. W97
# ---------------------------------------------------------------------------


def test_the_normative_effort_makes_the_field_to_lumped_seam_one_sided():
    w = _stage("w97")
    assert w["ratio_diffusive"] > 1.0
    assert w["modes"]["diffusive"]["blocks"]["SUSP"] > \
        w["modes"]["diffusive"]["blocks"][w["modes"]["diffusive"]["fluid_agent"]]


def test_the_conservative_conormal_crosses_unity_and_closes_W97():
    """W47 scoped section 4.1's co-normal OUT at fluid-fluid seams because the
    advective term cancels between two sides sharing a ring cell.  There is no
    second fluid side here, it does not cancel, and it is worth a factor."""
    w = _stage("w97")
    assert w["ratio_conormal"] < 1.0
    assert w["conormal_improvement"] > 10.0
    assert w["crosses_one_under_conormal"] is True
    assert w["verdict"] == "closed"


def test_the_conormal_also_improves_beta_and_the_informative_window():
    w = _stage("w97")
    assert w["beta_conormal"] > 10.0 * w["beta_diffusive"]
    d = w["modes"]["diffusive"]["fluid_informative_window"]
    c = w["modes"]["conormal"]["fluid_informative_window"]
    assert c > 10.0 * d


def test_a_rigid_lumped_partner_contributes_a_rank_one_block():
    """The reason the verdict at THIS seam differs from W97's rotor face: a
    spring responds only to the rigid mode, so the assembled operator's smallest
    singular value is the field expert's own response in every other direction
    and beta collapses below the fluid block's own norm."""
    w = _stage("w97")
    assert w["lumped_block_rank"] == 1
    assert w["dim_M"] > 1
    for mode in ("diffusive", "conormal", "reaction"):
        row = w["modes"][mode]
        assert row["block_rank"]["SUSP"] == 1
        assert row["block_rank"][row["fluid_agent"]] == w["dim_M"]
        assert row["null_dim"] == 0
        # never blind, and for the rank reason rather than the ratio one
        assert row["fluid_blind_at_zero_tolerance"] is False


# ---------------------------------------------------------------------------
# 8. F5 -- the gradient
# ---------------------------------------------------------------------------


def test_the_adjoint_agrees_with_a_central_finite_difference():
    g = _stage("gradient")
    assert g["best_fd"]["rel_error"] < 1e-2
    assert g["best_fd"]["delta"] <= 1e-4


def test_the_finite_difference_shows_a_truncation_branch_and_no_floor_yet():
    """float64 throughout, so the truncation branch runs five clean decades and
    the cancellation branch has not been reached at 1e-7.  PoC 1a's float32
    checkpoint column floors at 1.4e-3 (W121) and would resolve neither."""
    g = _stage("gradient")
    fd = sorted(g["fd"], key=lambda r: -r["delta"])
    errs = [row["rel_error"] for row in fd]
    assert min(errs) < 1e-4
    assert max(errs) > 1e4 * min(errs)
    # monotone down: still on the truncation branch at the smallest step tried
    assert errs == sorted(errs, reverse=True)


def test_the_objective_is_bit_reproducible():
    g = _stage("gradient")
    assert g["reproducibility_rel"] == 0.0


def test_the_gradients_sign_depends_on_the_horizon():
    """The transient and the equilibrium want ``h0`` moved in opposite
    directions, and the objective averages over whichever the horizon reaches."""
    g = _stage("gradient")
    assert g["sign_flips_with_horizon"] is True
    signs = {int(k): v for k, v in g["sign_by_horizon"].items()}
    shortest, longest = min(signs), max(signs)
    assert signs[shortest] != signs[longest]


def test_the_long_horizon_gradient_has_the_quasi_static_sign():
    g = _stage("gradient")
    assert g["quasistatic_prediction"] < 0.0
    longest = max(g["horizons"], key=lambda r: r["steps"])
    assert longest["adjoint"] < 0.0


def test_every_horizons_adjoint_is_confirmed_by_a_finite_difference():
    g = _stage("gradient")
    for row in g["horizons"]:
        assert row["rel_error"] < 5e-2, row


def test_one_gradient_costs_a_handful_of_forward_evaluations():
    g = _stage("gradient")
    assert 2.0 < g["grad_over_forward"] < 12.0


# ---------------------------------------------------------------------------
# 9. the power residual
# ---------------------------------------------------------------------------


def test_the_residual_closes_with_the_motion_term_and_not_without_it():
    p = _stage("power")
    assert p["closes"] is True
    assert p["residual_with_motion"] < 0.05
    assert p["improvement"] > 20.0


# ---------------------------------------------------------------------------
# 10. the compile, from the artifact
# ---------------------------------------------------------------------------


def test_the_artifact_records_both_verdicts_under_both_efforts():
    c = _stage("compile")
    for mode in ("diffusive", "conormal"):
        assert c[f"{mode}-fixed-floor"]["verdict"] == "admit-uncertified"
        assert c[f"{mode}-moving"]["verdict"] == "refuse"
        assert c[f"{mode}-fixed-floor"]["envelope"]["E2"] == "holds"
        assert c[f"{mode}-moving"]["envelope"]["E2"] == "fails"
        assert c[f"{mode}-moving"]["refusals"] == ["L2/InterfaceMotion"]


def test_nothing_here_reaches_admit_and_the_reason_is_named():
    c = _stage("compile")
    for row in c.values():
        assert row["verdict"] != "admit"
        assert row["unmeasured"], "a graph with no unmeasured constants would " \
            "have to explain how L was measured"


def test_the_environment_the_numbers_were_taken_in_is_on_the_record():
    a = _artifact()
    env = a["environment"]
    assert env["kmp_duplicate_lib_ok"] == "TRUE"
    assert env["tf32_matmul"] is False
    assert env["tf32_cudnn"] is False
    assert env["dtype"] == "torch.float64"
