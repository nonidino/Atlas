"""Tier 51 -- PoC 3 RaceLab, phase 1: the car's geometry, its windows, its graph.

`POC3-RACELAB-REQUIREMENTS.md` section 9's phase 1.  What this file pins:

1. **The box was chosen from a MEASUREMENT**, not from the requirements'
   ``384x192`` guess, and the measurement is a ratio against the front-wing
   case timed in the same process.
2. **Every body is `wing_fsi.FlexWing`**, so the discrete conservation identity
   ``sum(f dA) == -F`` holds on this lattice to machine precision -- the check
   W94 exists for, now over thirteen bodies instead of one.
3. **The wheel's coefficient is DERIVED**, ``c_n = 3 C_D / 4``, and the ring of
   twelve chords reproduces the continuous integral it was derived from.
4. **The wheel does not roll, and that is measured.**  The rotation's
   projection onto the chord normals is machine-zero at one station a chord and
   first order in the chord length, which is the signature of a discretisation
   artifact and not of a physical effect.
5. **The layout is derived from the car** and is `ground_effect.GroundTiling`'s
   own machinery: the partition of unity sums to one everywhere, a uniform
   `RaceTiling` reproduces a `GroundTiling` bitwise, and the device bands are
   narrow enough that the ring a device reads is 7.5 cells from its plane --
   the front-wing case's distance to the digit.
6. **W124's discipline is unachievable on a car**, and that is a finding rather
   than a bug: every covering layout cuts the body.  The y-cut IS clear.
7. **The compile reproduces CS-18's verdicts on a different graph** -- the
   joined union refuses at `L7/R9` alone, the disjoint union refuses nothing --
   which is what makes the refusal a statement about the clocks rather than
   about the front wing's layout.
8. **The gate's numbers are the page's numbers**, because a criterion written
   in prose and evaluated in code drifts, and Tier 48 measured that it drifts
   permissive.

The live marches are slow (the tight column is 1.7 s a macro-step) so the tests
here march only a handful of steps.  The artifact tests need
``out/racelab/racelab.json``; rebuild it with
``python scripts/tier51_racelab_graph.py`` -- see that file's header for what
the arms cost.
"""

from __future__ import annotations

import json
import math
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np                                                    # noqa: E402
import pytest                                                         # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from atlas.compiler import compile_scheme                             # noqa: E402
from atlas.cases import ground_effect as GE                           # noqa: E402
from atlas.cases import integration_union as IU                       # noqa: E402
from atlas.cases import racelab as RL                                 # noqa: E402
from atlas.cases import vehicle_march as VM                           # noqa: E402
from atlas.cases import wing_fsi as W                                 # noqa: E402

ART = os.path.join(_ROOT, "out", "racelab", "racelab.json")
PAGE = os.path.join(_ROOT, "wiki", "concepts", "Atlas 0.1", "common",
                    "case-study-racelab-graph-atlas-0.1.md")


def _art():
    if not os.path.isfile(ART):
        pytest.skip("out/racelab/racelab.json is absent; rebuild it with "
                    "python scripts/tier51_racelab_graph.py")
    with open(ART, encoding="utf-8") as fh:
        return json.load(fh)


def _page():
    if not os.path.isfile(PAGE):
        pytest.skip("the case study page is absent")
    with open(PAGE, encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture(scope="module")
def car():
    return RL.car_bodies()


@pytest.fixture(scope="module")
def tiling():
    return RL.layout()


# ---------------------------------------------------------------------------
# 1. the car is thirteen bodies and every one of them is CS-12's plate
# ---------------------------------------------------------------------------


def test_the_car_is_a_car_and_not_a_rectangle(car):
    objs, flat = car
    ids = {getattr(o, "body_id", None) for o in objs}
    for want in ("FW_MAIN", "FW_FLAP", "NOSE", "FLOOR", "DIFF", "POD_UP",
                 "POD_LO", "DUCT_UP", "DUCT_LO", "RW_MAIN", "RW_FLAP",
                 "WHEEL_F", "WHEEL_R"):
        assert want in ids, f"the car has no {want}"
    #: the requirements' section 3.1 asks for a front wing, a floor, a
    #: diffuser, a sidepod, a radiator duct, a rear wing, two wheels and a
    #: moving ground.  The ground is `ground_effect`'s band condition and has
    #: no body.
    assert len(objs) == 13
    assert len(flat) == 11 + 2 * 12, "each wheel is twelve plate segments"


def test_the_front_wing_is_CS12_s_plate_at_CS12_s_chord(car):
    """The wetted seam pairs with a 32-station structure, so the plate has to
    have 32 stations at the chord that structure was built at."""
    _objs, flat = car
    fw = next(b for b in flat if b.body_id == RL.WING_BODY_ID)
    assert fw.n_station == W.N_STATION
    assert fw.chord * RL.DX == pytest.approx(GE.CHORD)


def test_every_body_s_force_integrates_to_minus_the_force_on_it(car):
    """W94's identity, over thirteen bodies instead of CS-18's two devices.

    `FlexWing.forcing` normalises each station's kernel on its own stamping
    box, so this is exact rather than approximate.  A discretization losing 3%
    of a body's force would read as a 3% interface residual at a join and be
    charged to the join.
    """
    import torch
    objs, _flat = car
    u = torch.full((RL.RNY, RL.RNX), GE.U_INF, dtype=W.TORCH_DTYPE)
    v = torch.zeros((RL.RNY, RL.RNX), dtype=W.TORCH_DTYPE)
    worst = 0.0
    for b in objs:
        fx, fy, _w, load, drag = b.forcing(u, v, RL.RNY, RL.RNX)
        got = float(fx.sum() * RL.DX * RL.DX)
        if abs(float(drag)) > 1e-12:
            worst = max(worst, abs(got + float(drag)) / abs(float(drag)))
        goty = float(fy.sum() * RL.DX * RL.DX)
        if abs(float(load)) > 1e-12:
            worst = max(worst, abs(goty - float(load)) / abs(float(load)))
    assert worst < RL.GATE["P7_body_force_conserves"]["tol"], worst


# ---------------------------------------------------------------------------
# 2. the wheel: a derived coefficient, and a rotation that is not physics
# ---------------------------------------------------------------------------


def test_the_wheel_s_coefficient_is_derived_and_not_the_plate_s(car):
    """``c_n = 3 C_D / 4`` falls out of equating the ring's drag to a cylinder's.

    The plate's own ``C_N = 20`` gave a wheel 22x too draggy and blew the march
    up at macro-step 2, so this is the number that had to be derived rather
    than inherited.
    """
    assert RL.WHEEL_CN == pytest.approx(3.0 * RL.WHEEL_CD / 4.0)
    assert RL.WHEEL_CN < GE.C_N / 20.0, "the plate's coefficient is 20; a "
    objs, _flat = car
    wheel = next(b for b in objs if isinstance(b, RL.WheelBody))
    assert wheel.c_n == pytest.approx(RL.WHEEL_CN)


def test_the_twelve_chord_ring_reproduces_the_cylinder_it_was_derived_from(car):
    """The derivation is continuous; the model is a 12-gon.  The gap is the
    discretisation and it has to be small or the derivation means nothing."""
    import torch
    objs, _flat = car
    wheel = next(b for b in objs if isinstance(b, RL.WheelBody))
    u = torch.full((RL.RNY, RL.RNX), GE.U_INF, dtype=W.TORCH_DTYPE)
    v = torch.zeros((RL.RNY, RL.RNX), dtype=W.TORCH_DTYPE)
    _fx, _fy, _w, _load, drag = wheel.forcing(u, v, RL.RNY, RL.RNX)
    assert float(drag) == pytest.approx(wheel.cylinder_drag(), rel=0.02)


def test_a_rolling_wheel_is_invisible_at_one_station_a_chord(car):
    """The rotation's projection is ``omega s`` from the chord's midpoint.

    At one station a chord the station IS the midpoint, so the projection is
    machine zero -- which is the continuum answer, reached exactly.
    """
    objs, _flat = car
    wheel = next(b for b in objs if isinstance(b, RL.WheelBody))
    one = RL.WheelBody("PROBE", wheel.xc, wheel.yc, wheel.r, n_seg=wheel.n_seg,
                       n_station=1, omega=wheel.rolling_omega())
    assert one.max_surface_normal_velocity() < 1e-12


def test_the_rotation_s_residual_is_first_order_in_the_chord_length(car):
    """Halving the chord halves it.  That is a discretisation artifact's
    signature and not a physical effect's, and it is why the declared car's
    wheels do not roll."""
    objs, _flat = car
    wheel = next(b for b in objs if isinstance(b, RL.WheelBody))
    vals = []
    for n_seg in (12, 24, 48):
        wk = RL.WheelBody("PROBE", wheel.xc, wheel.yc, wheel.r, n_seg=n_seg,
                          n_station=3, omega=wheel.rolling_omega())
        vals.append(wk.max_surface_normal_velocity())
    for a, b in zip(vals, vals[1:]):
        assert a / b == pytest.approx(2.0, rel=0.02)


def test_the_declared_car_does_not_roll(tiling):
    """A demo that showed the 12-gon's artifact would be showing a
    discretisation and calling it a rotating wall."""
    t, _i = tiling
    r = RL.RaceRollout(tiling=t)
    assert r.wheel_omega == 0.0
    for b in r.objects:
        if isinstance(b, RL.WheelBody):
            assert b.omega == 0.0


# ---------------------------------------------------------------------------
# 3. the tiling IS `GroundTiling`, with the layout handed in
# ---------------------------------------------------------------------------


def test_race_tiling_is_a_ground_tiling(tiling):
    t, _i = tiling
    assert isinstance(t, GE.GroundTiling)


def test_a_uniform_race_tiling_reproduces_a_ground_tiling_bitwise():
    """The whole claim of the subclass is that the partition of unity, the
    halo and the assembly are the parent's.  On a layout the parent can also
    express, the two have to agree to the bit or they are not the same code."""
    parent = W.DEFAULT_TILING
    child = RL.RaceTiling.of(
        [i * parent.stride for i in range(parent.n_col)],
        [j * parent.stride for j in range(parent.n_row)],
        nx=parent.nx, ny=parent.ny, wx=parent.wx, wy=parent.wy,
        ramp=parent.ramp)
    assert child.offsets == parent.offsets
    assert child.names == parent.names
    assert child.halo == parent.halo
    for a, b in zip(child.weights(), parent.weights()):
        assert np.array_equal(a, b)


def test_the_partition_of_unity_sums_to_one_everywhere(tiling):
    """A layout with a hole assembles to zero there and the march would be
    silently wrong rather than loud."""
    t, _i = tiling
    tot = np.sum(t.weights(), axis=0)
    assert tot.shape == (t.ny, t.nx)
    assert np.all(tot > 0.0), "some cell is in no window"
    assert np.allclose(tot, 1.0, atol=1e-12)


def test_the_layout_covers_and_overlaps(tiling):
    t, info = tiling
    assert t.covers()
    assert min(t.overlaps()) >= RL.HALO_MIN
    assert t.halo == RL.HALO_MIN
    assert 12 <= t.n_windows <= 20, "the requirements' section 4.1 target"
    assert info["nx"] == RL.RNX and info["ny"] == RL.RNY


def test_the_layout_is_derived_from_the_car_and_not_a_uniform_grid(tiling):
    """A uniform grid in x would have one stride; this has several, and they
    are where they are because the car is where it is."""
    t, _i = tiling
    strides = {t.cols[i + 1] - t.cols[i] for i in range(len(t.cols) - 1)}
    assert len(strides) > 1, "the columns are uniformly spaced after all"


# ---------------------------------------------------------------------------
# 4. the devices sit ON a cut, narrowly, because that is where they are declared
# ---------------------------------------------------------------------------


def test_each_device_sits_on_a_cut_of_its_own(tiling):
    t, _i = tiling
    sites = RL.device_sites(t)
    assert set(sites) == {"RAD", "ROTOR"}
    assert sites["RAD"].seam != sites["ROTOR"].seam
    bands = {tuple(b) for b in t.x_bands()}
    for d, s in sites.items():
        i = int(s.seam[1])
        band = (t.cols[i + 1], t.cols[i] + t.wx)
        assert band in bands


def test_the_ring_a_device_reads_is_as_close_to_its_plane_as_the_front_wing_s(
        tiling):
    """CS-18 section 7.1 diagnosed J3's whole 4% residual as this distance.

    The first layout this tier searched put it at forty cells, which would have
    made the receiving balance report the layout rather than the join.  The
    constraint that fixed it is `device_overlap_max`.
    """
    t, _i = tiling
    sites = RL.device_sites(t)
    for d, s in sites.items():
        spec = RL.RaceDeviceSpec("J1" if d == "RAD" else "J3", s, d, t)
        assert spec.ring_to_plane_cells == pytest.approx(7.5)
        assert spec.rows.size == IU.DEVICE_CELLS


def test_the_device_planes_are_inside_the_duct(tiling):
    t, _i = tiling
    _objs, flat = RL.car_bodies()
    duct = [b for b in flat if b.body_id.startswith("DUCT_")]
    lo = min(b.x_span()[0] for b in duct)
    hi = max(b.x_span()[1] for b in duct)
    sites = RL.device_sites(t)
    for d, s in sites.items():
        spec = RL.RaceDeviceSpec("J1" if d == "RAD" else "J3", s, d, t)
        x = spec.x_plane / RL.DX
        assert lo < x < hi, f"{d}'s plane at {x} is outside the duct"


# ---------------------------------------------------------------------------
# 5. W124 on a car: unachievable in x, met in y
# ---------------------------------------------------------------------------


def test_every_covering_layout_cuts_the_car(car, tiling):
    """The finding.  The front wing had ONE body in 208 cells and eight cells
    of clearance; the car's bodies cover 72 to 501 of 672 almost without a gap
    and the largest stride the halo allows is 112."""
    _objs, flat = car
    t, info = tiling
    assert info["min_cut_to_body_clearance_cells"] < 0
    #: and it is not this layout's fault: the car's own gaps are all shorter
    #: than a band, so no placement of the cuts avoids it
    spans = sorted(b.x_span() for b in flat if not b.cuttable)
    covered = []
    for lo, hi in spans:
        if covered and lo <= covered[-1][1]:
            covered[-1][1] = max(covered[-1][1], hi)
        else:
            covered.append([lo, hi])
    gaps = [covered[i + 1][0] - covered[i][1] for i in range(len(covered) - 1)]
    assert max(gaps, default=0.0) < RL.WX - RL.HALO_MIN, (
        "there is a gap wide enough to hide a cut in, so the finding is wrong")


def test_the_y_cut_is_clear_of_every_body(car, tiling):
    """In y the discipline IS met, and `wing_fsi.WingTiling.rows_clear_of_wing`
    is the check this mirrors."""
    _objs, flat = car
    t, info = tiling
    assert info["y_cut_to_body_clearance_cells"] > 0
    assert t.y_clearance(flat) > 0


def test_the_front_wing_is_owned_by_one_window(car, tiling):
    """The wetted and mount seams pair STRUCT and SUSP with ONE fluid window,
    so a plate inside an overlap would be a declaration that does not describe
    the march."""
    _objs, flat = car
    t, info = tiling
    fw = next(b for b in flat if b.body_id == RL.WING_BODY_ID)
    assert t.owns(fw) is not None
    for band_lo, band_hi in t.x_bands():
        lo, hi = fw.x_span()
        assert not (lo < band_hi and band_lo < hi), "a cut crosses the plate"


# ---------------------------------------------------------------------------
# 6. the graph
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def graph():
    u = np.full((RL.RNY, RL.RNX), GE.U_INF)
    v = np.zeros((RL.RNY, RL.RNX))
    return RL.build(u, v)


def test_the_graph_has_the_five_families_CS18_marched(graph):
    g, _aux = graph
    fams = {a.capabilities.governing_family for a in g.agents}
    assert fams == {"incompressible-navier-stokes-2d",
                    "plane-stress-elasticity-2d",
                    "heat-conduction-2d",
                    "incompressible-thermal-transport-1d",
                    "lumped-dc-circuit"}
    assert len(g.agents) == 26


def test_the_joined_union_refuses_at_R9_and_nothing_else(graph):
    """CS-18 section 4.5's table, on a graph that is not the front wing's."""
    g, _aux = graph
    res = compile_scheme(g)
    refusals = sorted({f"{d.layer}/{d.rule}" for d in res.decisions
                       if d.verdict.value == "refuse"})
    assert refusals == RL.GATE["P1_compile"]["expected_refusals_joined"]
    assert res.verdict.value == "refuse"


def test_the_disjoint_union_refuses_nothing():
    """W194's control.  The disjoint union has no multirate SEAM, so R9's
    premise does not hold and it must stop being refused."""
    u = np.full((RL.RNY, RL.RNX), GE.U_INF)
    v = np.zeros((RL.RNY, RL.RNX))
    g, _aux = RL.build(u, v, joins=())
    res = compile_scheme(g)
    refusals = sorted({f"{d.layer}/{d.rule}" for d in res.decisions
                       if d.verdict.value == "refuse"})
    assert refusals == RL.GATE["P1_compile"]["expected_refusals_disjoint"]
    assert res.verdict.value == "admit-uncertified"


def test_reconciling_the_clocks_stops_the_refusal():
    """The control that says the refusal is about the clocks: re-declare every
    agent's native step at the tiling's and R9 goes quiet, which is exactly
    what `vehicle-scale-and-sizing` section 1.3 says costs a 400x
    over-resolution of the coolant circuit."""
    u = np.full((RL.RNY, RL.RNX), GE.U_INF)
    v = np.zeros((RL.RNY, RL.RNX))
    g, _aux = RL.build(u, v, clocks="reconciled")
    res = compile_scheme(g)
    refusals = sorted({f"{d.layer}/{d.rule}" for d in res.decisions
                       if d.verdict.value == "refuse"})
    assert refusals == []


def test_the_graph_declares_the_partition_of_unity_and_the_pressure_field(graph):
    g, _aux = graph
    assert g.decomposition is not None
    assert g.overlap_cells == {IU.FLUID: RL.HALO_MIN}
    names = [f.name for f in g.global_fields]
    assert "pressure" in names


# ---------------------------------------------------------------------------
# 7. the march runs, and it is guarded
# ---------------------------------------------------------------------------


def test_a_short_march_runs_and_moves_every_join(tiling):
    t, _i = tiling
    u = np.full((RL.RNY, RL.RNX), GE.U_INF)
    v = np.zeros((RL.RNY, RL.RNX))
    m = RL.march(u, v, steps=3, tiling=t, join_coupling="lagged")
    assert m.steps == 3
    for k in ("u_rotor", "u_core", "ua", "current", "q_machine", "load",
              "drag"):
        a = np.asarray(m.trace[k])
        assert a.size == 3
        assert np.all(np.isfinite(a))
    assert float(np.asarray(m.trace["u_max"])[-1]) < RL.U_MAX_BAND


def test_the_march_bails_past_the_declared_band_rather_than_hanging():
    """`WindowNS` sizes its sub-step count from ``u_max``, so a diverging field
    reads as a hang.  The guard has to fire, and it has to name the step."""
    u = np.full((RL.RNY, RL.RNX), 10.0 * GE.U_INF)
    v = np.zeros((RL.RNY, RL.RNX))
    with pytest.raises(RuntimeError, match="past the declared band"):
        RL.march(u, v, steps=2, join_coupling="lagged")


def test_the_march_result_is_CS18_s_own_record_type(tiling):
    """`vehicle_march.receiver_balances` reads this march unchanged, which is
    what stops phase 1's numbers drifting from CS-18's."""
    t, _i = tiling
    u = np.full((RL.RNY, RL.RNX), GE.U_INF)
    v = np.zeros((RL.RNY, RL.RNX))
    m = RL.march(u, v, steps=2, tiling=t, join_coupling="lagged")
    assert isinstance(m, VM.UnionMarch)
    bal = VM.receiver_balances(m, 0.5)
    assert set(bal) == {"J1", "J2", "J3"}


def test_J1_s_null_leaves_the_fluid_alone(tiling):
    """J1 carries no power, so pinning UA must not touch the flow at all -- the
    cleanest control on CS-18's page, reproduced here."""
    t, _i = tiling
    u = np.full((RL.RNY, RL.RNX), GE.U_INF)
    v = np.zeros((RL.RNY, RL.RNX))
    a = RL.march(u, v, steps=3, tiling=t, join_coupling="lagged")
    b = RL.march(u, v, steps=3, tiling=t, join_coupling="lagged", null="J1")
    assert np.array_equal(a.u, b.u)
    assert np.array_equal(a.v, b.v)
    assert float(np.asarray(b.trace["ua"])[-1]) == pytest.approx(
        VM.CL.UA_RAD, abs=0.0)


def test_the_repeat_is_bitwise(tiling):
    t, _i = tiling
    u = np.full((RL.RNY, RL.RNX), GE.U_INF)
    v = np.zeros((RL.RNY, RL.RNX))
    a = RL.march(u, v, steps=3, tiling=t, join_coupling="lagged")
    b = RL.march(u, v, steps=3, tiling=t, join_coupling="lagged")
    assert np.array_equal(a.u, b.u)
    for k in VM.CROSSING_KEYS:
        if k in a.trace:
            assert np.array_equal(np.asarray(a.trace[k]),
                                  np.asarray(b.trace[k]))


# ---------------------------------------------------------------------------
# 8. the artifact, and the page that quotes it
# ---------------------------------------------------------------------------


def test_the_box_was_chosen_from_a_measurement():
    a = _art()
    s = a["size"]
    assert s["chosen"]["nx"] == RL.RNX and s["chosen"]["ny"] == RL.RNY
    assert len(s["candidates"]) >= 6, "one candidate is not a measurement"
    names = [c["name"] for c in s["candidates"]]
    assert any(n.startswith("control-frontwing") for n in names), (
        "a timing without a control in the same process is a wall-clock number")
    assert any("384x192" in n for n in names), (
        "the requirements' own candidate has to be among them")
    assert s["control_drift_over_the_run"] == pytest.approx(1.0, abs=0.15)


def test_the_conservation_identity_is_machine_precision_on_the_artifact():
    a = _art()
    assert (a["geometry"]["worst_conservation_residual_x"]
            < RL.GATE["P7_body_force_conserves"]["tol"])


def test_the_two_layout_profiles_agree_to_the_search_grid():
    """The geometric profile and the release-state force profile are different
    questions; the layout being the same under both is what says the answer
    does not turn on the force profile's blindness to a body at zero
    incidence."""
    a = _art()
    pc = a["windows"]["profile_control"]
    assert pc["same_column_count"]
    assert pc["max_offset_difference_cells"] <= pc["search_grid_step_cells"]


def test_the_gate_s_numbers_are_the_page_s_numbers():
    """Tier 48 measured that a criterion written in prose and evaluated in code
    drifts, and drifts permissive.  So every threshold has to appear on the
    page."""
    page = _page()
    wanted = [
        str(RL.GATE["P2_J3_receiving_balance"]["tol_with"]),
        str(RL.GATE["P2_J3_receiving_balance"]["tol_without"]),
        str(RL.GATE["P4_J2_receiving_balance"]["tol_without"]),
        str(RL.GATE["P6_macro_step_cost"]["ceiling_s"]),
    ]
    for w in wanted:
        assert w in page, f"the gate's {w} is not on the page"
    assert "1e-6" in page or "10^{-6}" in page, (
        "P3's and P4's tolerance is not on the page")


def test_the_page_names_what_the_tier_did_not_do():
    page = _page()
    assert "What this tier did NOT do, named" in page


def test_the_page_does_not_claim_a_learned_expert_pays():
    """The foundation-model half has no learned expert admitted with a nonzero
    contribution, and RaceLab must not imply otherwise (requirements section
    4.2).  Phase 1 has no learned expert at all."""
    page = _page()
    low = page.lower()
    for banned in ("poseidon pays", "faster than classical",
                   "the learned expert wins"):
        assert banned not in low


def test_NeuberNet_is_not_loaded_or_referenced_as_code():
    """Hard constraint 1: unlicensed, local-only, and nowhere near this tier.

    **The check is on USE, not on the word.**  A docstring saying the thing was
    not loaded is exactly what the vault's standard asks for, so a substring
    test on the name flags the honesty rather than the violation -- which is
    what the first version of this test did.  What is banned is an import, an
    attribute access, or a path into the local cache.
    """
    import re
    #: An import, an attribute access or a path into the local cache.  NOT the
    #: bare name: this test's own name carries it, and so does every honest
    #: sentence saying the thing was not loaded.
    #:
    #: **The pattern is assembled at run time on purpose.**  Written out as a
    #: literal it appears in this file's own source and the scan flags the
    #: scanner -- which is what the second version of this test did, and is the
    #: same class of self-reference as `vault_scan`'s orphan check flagging the
    #: page that QUOTES a corruption.
    n = "neuber" + "net"
    banned = re.compile(
        r"import\s+%s|from\s+%s|%s\s*\.\w|%s[/\\]|cache[/\\]%s"
        % (n, n, n, n, n), re.IGNORECASE)
    for path in (os.path.join(_ROOT, "atlas", "cases", "racelab.py"),
                 os.path.join(_ROOT, "scripts", "tier51_racelab_graph.py"),
                 os.path.abspath(__file__)):
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        hit = banned.search(text)
        assert hit is None, f"{path} reaches NeuberNet at {hit.group(0)!r}"


def test_the_march_arms_each_carry_their_null():
    """A gate a broken graph passes is not a gate.

    The null arm's residual is READ OFF the arm and not copied from the
    record's own constant: `receiver_balances` writes 1.0 there because
    withholding the force makes the work exactly zero, and an identity written
    down is not a control.  So the arm has to produce it.
    """
    a = _art()
    if "march" not in a:
        pytest.skip("stage march has not run")
    arms = a["march"]["arms"]
    assert "referent" in arms and "null_J1" in arms and "null_J3" in arms
    g1 = a["march"]["G1_J3"]
    assert g1["null_arm_work_on_the_fluid"] == 0.0, (
        "the null arm's device still did work on the fluid")
    assert g1["null_arm_shaft_still_claims"] != 0.0, (
        "the null arm withheld the whole device, so the balance reads 0/0 and "
        "cannot fail -- which is not a control")
    assert g1["residual_without_the_term_null_arm"] == pytest.approx(1.0)


def test_the_settling_curve_has_two_columns_that_disagree():
    """The full band and the fluid-only band settle in opposite directions at
    the short end, and a single trailing window would have hidden it."""
    a = _art()
    if "settle" not in a:
        pytest.skip("stage settle has not run")
    b = a["settle"]["bands"]
    norms = [b[k]["norm"] for k in ("through_160", "through_400",
                                    "through_800", "through_1600")]
    fluid = [b[k]["fluid_only_norm"] for k in ("through_160", "through_400",
                                               "through_800", "through_1600")]
    assert norms == sorted(norms, reverse=True), "the full band is not monotone"
    assert fluid[0] < fluid[1], (
        "the fluid band does not rise first, so the claim that a 160-step "
        "window is quiet for the wrong reason is wrong")
    assert fluid[-1] > 10 * a["settle"]["CS18_band_norm_for_comparison"], (
        "the car settles to CS-18's band after all, and W221 is not a gap")


def test_every_gate_clause_the_arms_can_judge_has_a_verdict():
    a = _art()
    if "march" not in a or "verdicts" not in a.get("march", {}):
        pytest.skip("stage march has not run")
    v = a["march"]["verdicts"]
    for clause in ("P2_J3_receiving_balance", "P3_J1_parametric",
                   "P5_repeat_floor"):
        assert v[clause]["verdict"] in ("pass", "fail")
    #: **P6 is expected to split**, and a verdict that cannot split is not a
    #: measurement: the lagged column is what a dashboard would march and the
    #: tight one is five solver calls an exchange.
    assert "verdict_lagged" in v["P6_macro_step_cost"]
    assert "verdict_tight" in v["P6_macro_step_cost"]


def test_the_thresholds_the_verdicts_used_are_the_gate_s_own():
    """A criterion written in prose and evaluated in code drifts permissive
    (Tier 48).  So the artifact records which number it actually compared
    against, and it has to be the declaration's."""
    a = _art()
    if "march" not in a or "verdicts" not in a.get("march", {}):
        pytest.skip("stage march has not run")
    v = a["march"]["verdicts"]
    assert (v["P2_J3_receiving_balance"]["tol_with"]
            == RL.GATE["P2_J3_receiving_balance"]["tol_with"])
    assert (v["P2_J3_receiving_balance"]["tol_without"]
            == RL.GATE["P2_J3_receiving_balance"]["tol_without"])
    assert (v["P3_J1_parametric"]["tol_with"]
            == RL.GATE["P3_J1_parametric"]["tol_with"])
    assert (v["P6_macro_step_cost"]["ceiling_s"]
            == RL.GATE["P6_macro_step_cost"]["ceiling_s"])


def test_the_seam_map_separates_what_a_seam_earned_from_what_it_was_handed():
    """`L7/R9`'s subject is <graph>: N red tiles from it are N views of ONE
    refusal.  PoC 2 published the other reading once and W177 corrected it."""
    a = _art()
    j = a["compile"]["joined"]
    assert j["graph_level_refusals"] == ["L7/R9"]
    assert j["seam_local_refusals"] == [], (
        "a seam earned a refusal on its own subject, which would change the "
        "reading of the map entirely")
    assert set(j["seam_verdict_counts"]) == {"refuse"}
    assert "refuse" not in j["seam_local_verdict_counts"]


def test_the_page_quotes_the_artifact_s_own_numbers():
    a = _art()
    page = _page()
    t, info = RL.layout()
    assert str(t.n_windows) in page
    assert str(RL.RNX) in page and str(RL.RNY) in page
    assert "%.1f" % (100 * info["banded_force_fraction"]) in page or \
           "%.0f" % (100 * info["banded_force_fraction"]) in page
