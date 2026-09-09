"""Tier 27 -- PoC 2, CS-10 and CS-12 assembled, and a constrained design search.

`atlas/cases/front_wing.py`, `scripts/w141_poc2_frontwing.py` and
`atlas/demo_frontwing/`.  Six groups:

  * **the assembly's own gate** -- each parent is a LIMIT of the joint interface
    system and both limits are asserted: `k -> inf` must reproduce CS-12's
    32-equation solve and `S_e -> inf` must reproduce CS-10's scalar one.  A
    composition that does not reduce to its parts is not an assembly;
  * **the declaration** -- both surface seams on one plate, W114's narrowing
    surviving the third agent, and the control that shows it is still a premise
    check;
  * **the parents are unmoved** -- the additive changes to `wing_fsi` are
    bitwise-neutral when their new arguments are omitted, which is what lets this
    be called an assembly rather than a fork;
  * **the design box is measured** -- the thickness trends have the physical sign
    inside the box and the stress trend INVERTS below it, which is where the
    lower bound comes from;
  * **the gate and the residual** -- the split reproduces the referent in BOTH
    degrees of freedom, the crossing is looked for, and R closes across both
    seams at once;
  * **the demo** -- its certification panel is the driver's, not a second one.

The artifact-backed tests re-derive the published numbers from
`out/w141/w141.json` and skip when it is absent.  The live tests build graphs and
solve interface systems, which costs about a minute; the marches are not re-run.
"""

from __future__ import annotations

import json
import math
import os
import sys

# Before numpy.  The build repo pulls torch in and torch's OpenMP beside numpy's
# MKL aborts the interpreter inside a dense solve -- `test_tier29`'s note, and it
# is not optional here either.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas import compile_scheme                                  # noqa: E402
from atlas.cases import front_wing as F                           # noqa: E402
from atlas.cases import ground_effect as G                        # noqa: E402
from atlas.cases import wing_fsi as W                             # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ART = os.path.join(_ROOT, "out", "w141", "w141.json")
_FIELD = os.path.join(_ROOT, "out", "w141", "settled.npz")
_CS12_FIELD = os.path.join(_ROOT, "out", "w136", "settled.npz")


def _artifact() -> dict:
    if not os.path.isfile(_ART):
        pytest.skip(f"{_ART} is absent; run scripts/w141_poc2_frontwing.py")
    with open(_ART, encoding="utf-8") as fh:
        return json.load(fh)


def _stage(name: str) -> dict:
    a = _artifact()
    if name not in a:
        pytest.skip(f"stage {name!r} has not been run")
    return a[name]


def _field():
    for p in (_FIELD, _CS12_FIELD):
        if os.path.isfile(p):
            d = np.load(p)
            return d["u"], d["v"]
    pytest.skip("no settled field on disk")


# ---------------------------------------------------------------------------
# 1. the assembly's own gate: each parent is a limit
# ---------------------------------------------------------------------------


def test_the_spring_going_rigid_reproduces_CS12s_interface_solve():
    """`k -> infinity` holds the mount, so the remaining 32 equations must be
    `wing_fsi.FSIRollout.solve_interface` -- the SAME solve CS-12 published, not
    a re-derivation of it."""
    u, v = _field()
    opt = dict(dtype=F.TORCH_DTYPE)
    ut, vt = torch.as_tensor(u, **opt), torch.as_tensor(v, **opt)
    S = F.N_STATION
    zeros, zero = torch.zeros(S, **opt), torch.zeros((), **opt)
    r = F.FrontWingRollout(tiling=F.SINGLE_TILING, coupling="tight")
    e = torch.tensor(F.E_STAR_REF, **opt)
    S_e, _ = r.structure(e, torch.tensor(F.TC_REF, **opt))
    h = torch.tensor(W.Y_MOUNT, **opt)
    kbig = torch.tensor(1.0e12, **opt)
    wd, vm, _res, _r0 = r.solve_seams(ut, vt, zeros, h, zeros, zero, S_e, kbig,
                                      h + F.LOAD_REF / kbig)
    rw = W.FSIRollout(tiling=W.SINGLE_TILING, coupling="tight")
    wp, _, _ = rw.solve_interface(ut, vt, zeros, zeros, e)
    scale = float(wp.abs().max())
    assert scale > 1e-3
    assert float((wd - wp).abs().max()) / scale < 1e-7
    assert abs(float(vm)) < 1e-8


def test_the_plate_going_rigid_reproduces_CS10s_scalar_newton():
    """`S_e -> infinity` holds the shape, so the remaining scalar equation must
    be CS-10's -- written out here rather than shared, so what is compared is the
    EQUATION and not a call into the same helper."""
    u, v = _field()
    opt = dict(dtype=F.TORCH_DTYPE)
    ut, vt = torch.as_tensor(u, **opt), torch.as_tensor(v, **opt)
    S = F.N_STATION
    zeros, zero = torch.zeros(S, **opt), torch.zeros((), **opt)
    r = F.FrontWingRollout(tiling=F.SINGLE_TILING, coupling="tight")
    S_e, _ = r.structure(torch.tensor(F.E_STAR_REF, **opt),
                         torch.tensor(F.TC_REF, **opt))
    h = torch.tensor(F.H0_REF - F.LOAD_REF / F.K_REF, **opt)
    k = torch.tensor(F.K_REF, **opt)
    h0 = torch.tensor(F.H0_REF, **opt)
    wd, vm, _res, _r0 = r.solve_seams(ut, vt, zeros, h, zeros, zero,
                                      S_e * 1.0e10, k, h0)
    wing, ny = r.wing, r.wing.n_hat[1]
    w_ext = wing.external_normal(ut, vt, zeros, r.ny, r.nx, h=h)
    vp = torch.zeros((), **opt)
    for _ in range(r.n_inner):
        w = w_ext - vp * ny
        load = -(wing.normal_traction(w) * ny * r.ds).sum()
        rr = k * (h0 - h - r.dt_ex * vp) - load
        dr = -k * r.dt_ex - (wing.c_n * torch.abs(w) * ny ** 2 * r.ds).sum()
        vp = vp - rr / dr
    assert abs(float(vp)) > 1e-6
    assert abs(float(vm) - float(vp)) / abs(float(vp)) < 1e-6
    assert float(wd.abs().max()) < 1e-8


def test_the_assembly_gate_is_recorded_as_passing():
    c = _stage("controls")
    assert c["pass"] is True
    assert c["k_to_infinity"]["rel"] < 1e-7
    assert c["S_to_infinity"]["rel"] < 1e-7
    assert c["reproducible_bitwise"] is True


# ---------------------------------------------------------------------------
# 2. the declaration
# ---------------------------------------------------------------------------


def test_both_surface_seams_are_declared_on_the_same_window():
    u, v = _field()
    g, _e = F.build(u, v, motion=False)
    seams = {c.seam_id: c for c in g.connections}
    assert "wet" in seams and "mount" in seams
    win = F.DEFAULT_TILING.wing_window()
    assert seams["wet"].a[0] == win and seams["mount"].a[0] == win
    # different agents on the far side, and different port names on the near one
    assert seams["wet"].b[0] == "STRUCT" and seams["mount"].b[0] == "SUSP"
    assert seams["wet"].a[1] != seams["mount"].a[1]
    # both n_0 = 0, for two DIFFERENT reasons -- the clamped-body row and the
    # field-to-lumped row of the guide's table
    assert seams["wet"].expected_null_dim == 0
    assert seams["mount"].expected_null_dim == 0


def test_W114s_narrowing_survives_the_third_agent():
    """`Suspension` declares the FLOW's governing family, so STRUCT is still the
    only plane-stress-elasticity-2d agent and R10's premise check still clears
    it.  This is not obvious from CS-12: adding an agent is exactly what could
    have re-triggered a rule that counts agents per family."""
    u, v = _field()
    g, _e = F.build(u, v, motion=False)
    fams = {}
    for a in g.agents:
        fams.setdefault(a.capabilities.governing_family, []).append(a.agent_id)
    assert fams["plane-stress-elasticity-2d"] == ["STRUCT"]
    assert "SUSP" in fams["incompressible-navier-stokes-2d"]
    r = compile_scheme(g)
    assert not list(r.decisions.refusals), [
        f"{d.layer}/{d.rule}" for d in r.decisions.refusals]


def test_the_R10_control_still_refuses_in_the_assembly():
    """A rule with nothing to fire on is not a rule.  Declare the structure with
    the fluid's family -- what a window tiling looks like from R10's side -- and
    the refusal has to come straight back."""
    u, v = _field()
    g, _e = F.build(u, v, motion=False,
                    struct_family="incompressible-navier-stokes-2d")
    r = compile_scheme(g)
    assert "L2/R10" in {f"{d.layer}/{d.rule}" for d in r.decisions.refusals}


def test_the_riding_graph_is_refused_at_interface_motion_on_both_seams():
    u, v = _field()
    g, _e = F.build(u, v, motion=True)
    r = compile_scheme(g)
    rules = [f"{d.layer}/{d.rule}" for d in r.decisions.refusals]
    assert rules and set(rules) == {"L2/InterfaceMotion"}
    # four ports: two on the wing window, one on each partner
    assert len(rules) == 4, rules


# ---------------------------------------------------------------------------
# 3. the parents are unmoved
# ---------------------------------------------------------------------------


def test_the_mount_height_argument_is_bitwise_neutral_when_omitted():
    """`FlexWing` gained an `h` argument for this assembly.  CS-12 omits it, so
    omitting it has to be bit-identical to passing the declared mount -- which is
    what lets CS-12's published numbers stand unchanged."""
    u, v = _field()
    opt = dict(dtype=W.TORCH_DTYPE)
    ut, vt = torch.as_tensor(u, **opt), torch.as_tensor(v, **opt)
    wing = W.FlexWing()
    d = torch.zeros(W.N_STATION, **opt)
    wp = torch.zeros(W.N_STATION, **opt)
    ny, nx = ut.shape
    a = wing.forcing(ut, vt, d, wp, ny, nx)
    for h in (W.Y_MOUNT, torch.tensor(W.Y_MOUNT, **opt)):
        b = wing.forcing(ut, vt, d, wp, ny, nx, h=h)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        assert torch.equal(a[3], b[3])


def test_the_surface_operator_at_the_declared_thickness_is_CS12s():
    """`_surface_operator` gained a `thick` argument and a stress map.  At the
    declared thickness it must still be the operator CS-12 measured -- the four
    numbers its own page quotes."""
    op = W._surface_operator()
    assert op["thick"] == W.THICK
    #: **Relative tolerances, and the reason is a distinction worth keeping.**
    #: There are TWO thread controls here and they act on different halves:
    #:
    #:   * `torch.set_num_threads` sets the march's threading. Measured: 20
    #:     macro-steps at 1 and at 8 agree to the BIT on the load, the ride
    #:     height, the tip deflection, the peak stress and u_max -- so the
    #:     driver's `--threads` flag is a clock and not a variable;
    #:   * `OMP_NUM_THREADS` sets numpy/MKL's, which is what builds `S_e` in the
    #:     BUILD REPO's solver. Measured across processes: `sigma_map` is
    #:     bitwise identical, and `S_e` -- a dense inverse at a condition number
    #:     of 7.1e8 -- differs by 1.3e-9 relative, which is round-off times the
    #:     conditioning.
    #:
    #: So `symmetry` (5.1e-11, and CS-12's own page calls it the direct sparse
    #: solve's round-off) and `cond` move in their last digits with the
    #: environment, while `geometric_gap` -- a property of the model rather than
    #: of the arithmetic -- does not. At the default environment all four
    #: reproduce CS-12's published values EXACTLY, which is what says these
    #: additive changes moved nothing; this test uses relative tolerances so it
    #: passes under a pinned `OMP_NUM_THREADS` as well.
    assert abs(op["symmetry"] - 5.08224474177674e-11) < 1e-4 * 5.08e-11
    assert op["symmetry"] < 1e-9
    assert abs(op["cond"] - 708025576.5006286) < 1e-6 * 7.08e8
    assert abs(op["geometric_gap"] - 0.000572967778934811) < 1e-12


def test_the_stress_map_is_the_experts_own_solve():
    """W139's replacement: linear elasticity is exactly linear in the traction,
    so the element stress under any station traction is a MAP and not a solve.
    Asserted against `solve_mechanical` itself."""
    op = W._surface_operator()
    q = np.linspace(0.3, 1.0, W.NI_STRUCT)
    mapped = np.einsum("k,kea->ea", q, op["sigma_map"])
    T = np.full(op["mesh"].n_nodes, W.T_REF)
    _u, direct = op["ts"].solve_mechanical(T, 0.5 * q, -0.5 * q, T_ref=W.T_REF,
                                           clamp_nodes=op["clamp"])
    assert np.abs(mapped - direct).max() / np.abs(direct).max() < 1e-11


def test_the_stress_does_not_depend_on_the_stiffness_and_the_deflection_does():
    """The two ceilings constrain different things, and that is physics rather
    than a modelling choice: a stress is set by the LOAD and the geometry, so
    raising E at a fixed traction shrinks the displacement and the strain in the
    same proportion and leaves D eps where it was."""
    op = W._surface_operator()
    q = np.full(W.NI_STRUCT, 0.4)
    sig = F.von_mises(np.einsum("k,kea->ea", q, op["sigma_map"])).max()
    d_soft = np.linalg.solve(op["S_e"] * (3.0e4 / W.E_REF), q)
    d_stiff = np.linalg.solve(op["S_e"] * (2.0e5 / W.E_REF), q)
    assert np.abs(d_soft).max() > 5.0 * np.abs(d_stiff).max()
    assert sig > 0.0     # and it carried no E at all


# ---------------------------------------------------------------------------
# 4. the design box is measured
# ---------------------------------------------------------------------------


def test_the_thickness_trends_have_the_physical_sign_inside_the_box():
    s = _stage("setup")
    lo, hi = s["design_box"]["tc"]
    inside = [(r, p) for r, p in zip(s["thickness"][1:], s["vm_exponent"])
              if r["tc"] >= lo]
    assert inside
    for r, p in inside:
        assert p < 0.0, (r["tc"], p)           # thicker is less stressed
    for r, p in zip(s["thickness"][1:], s["tip_exponent"]):
        assert p < 0.0                         # thicker is stiffer, everywhere


def test_the_stress_trend_inverts_below_the_box_and_that_is_why_the_box_starts_there():
    """Q1 elements lock in bending and the locking gets worse as the elements get
    more slender, so below about t/c = 0.026 the peak stress RISES with the
    thickness.  A search run through the inversion optimises an element
    formulation rather than a wing."""
    s = _stage("setup")
    assert s["vm_inversion"], "the inversion is the reason for the lower bound"
    worst = max(hi for _lo, hi in s["vm_inversion"])
    assert s["design_box"]["tc"][0] > worst


def test_the_operator_norm_is_not_the_quantity_to_read():
    """||S_e||_2 is non-monotone in the thickness while every response quantity
    is monotone: the norm is set by the stiffest mode, which on a
    two-element-deep Q1 mesh is a locked shear mode."""
    s = _stage("setup")
    signs = {p > 0 for p in s["norm_exponent"]}
    assert signs == {True, False}, s["norm_exponent"]
    assert all(p < 0 for p in s["tip_exponent"])


def test_the_ceilings_are_reachable_and_the_envelope_is_outside_them():
    """A ceiling the reference design already violates is not a constraint, and
    one the search cannot reach is not one either.  And `DELTA_CEIL` sits inside
    `WingStructure.validity`'s own bound deliberately: a search driven onto the
    deflection constraint must not simultaneously drive the expert out of its
    envelope, or the two failures are indistinguishable."""
    assert F.DELTA_CEIL < F.DELTA_MAX
    s = _stage("setup")
    assert s["delta_ceiling"] < s["delta_envelope"]


# ---------------------------------------------------------------------------
# 5. the gate and the residual
# ---------------------------------------------------------------------------


def test_the_split_reproduces_the_referent_in_BOTH_degrees_of_freedom():
    g = _stage("gate")
    cut = g["columns"]["tight, six windows"]
    assert cut["tip"]["max_rel"] < 5e-3
    assert cut["h"]["max_rel"] < 5e-3


def test_both_seams_lag_defects_are_first_order():
    g = _stage("gate")
    for key in ("tip", "h"):
        for o in g["lag_order"][key]:
            assert 0.7 < o < 1.6, (key, g["lag_order"][key])


def test_the_seams_dominate_the_cut_on_both_degrees_of_freedom():
    """CS-10 measured 97x at its field-to-lumped seam and CS-12 237x at its
    field-to-field one.  This graph carries both at once, and the fourth
    instance of `the seam is where the error is and the domain cut is not`."""
    g = _stage("gate")
    assert g["lag_over_cut"]["tip"] > 50.0
    assert g["lag_over_cut"]["h"] > 50.0


def test_the_crossing_was_looked_for_on_every_column():
    g = _stage("gate")
    for tag, col in g["columns"].items():
        for key in ("tip", "h"):
            assert "sign_changes" in col[key]
            assert col[key]["at_step"] <= g["steps"]


def test_R_closes_across_BOTH_seams_and_the_ordering_was_checked_step_by_step():
    """The receiving subsystems' own balance -- CS-9 section 6's rule -- with TWO
    receivers, so the energy is the structure's strain energy plus the spring's
    and the power is the interface power the two seams share."""
    r = _stage("residual")
    assert r["with_motion"] < 0.5 * r["without_motion"]
    cross = {c["lo"]: c for c in r["crossings"]}
    c = cross["with_motion"]
    assert c["hi"] == "without_motion"
    assert c["n_crossed"] == 0, c["steps"][:8]


def test_the_residual_settles_and_the_window_is_quoted_with_it():
    """W140's discipline on the page that opened it: a quantity read over a
    WINDOW carries the window and the evidence the window is what it is called."""
    r = _stage("residual")
    q = r["quartiles_with"]
    assert q[-1] < 1e-6
    assert q[-1] <= q[-2] * 3.0
    assert r["steps"] >= 480


# ---------------------------------------------------------------------------
# 6. the demo shows the compiler's own verdicts and not a second opinion
# ---------------------------------------------------------------------------


def test_the_demos_certification_panel_is_the_drivers():
    """A demo that groups the decisions differently from the driver is showing a
    different thing from the one the results page reports."""
    sys.path.insert(0, os.path.join(_ROOT, "scripts"))
    import importlib
    drv = importlib.import_module("w141_poc2_frontwing")
    from atlas.demo_frontwing.engine import seam_verdicts as demo_sv
    u, v = _field()
    for motion in (False, True):
        g, _e = F.build(u, v, motion=motion)
        r = compile_scheme(g)
        assert demo_sv(g, r) == drv.seam_verdicts(g, r)


def test_the_panel_is_red_on_the_surface_seams_and_amber_on_the_fluid_ones():
    from atlas.demo_frontwing.engine import seam_verdicts
    u, v = _field()
    g, _e = F.build(u, v, motion=True)
    sv = seam_verdicts(g, compile_scheme(g))
    assert sv["wet"]["colour"] == "red" and sv["mount"]["colour"] == "red"
    fluid = [s for k, s in sv.items() if s["kind"] == "fluid-fluid"]
    assert fluid and all(s["colour"] == "amber" for s in fluid)
    # nothing is green, and the reason is on the record
    assert not any(s["colour"] == "green" for s in sv.values())


def test_the_constants_are_measured_now_and_nothing_is_green_anyway():
    """**W157 changed this test's subject and that is the point of it.**

    It used to assert `unmeasured` was NON-empty -- L, sigma and C_mu were
    measured only in tier 0, on another graph at another state, so this graph
    declared none of them and W56's backstop forced `admit-uncertified`. That
    made "nothing is green" true for a **bookkeeping** reason, and
    `poc2-novelty-audit` section 3 recorded that the compiler had therefore
    never certified anything.

    `scripts/w157_frontwing_constants.py` measured all three here, so the
    backstop no longer fires -- and the graph is **still not green**. That is
    the stronger statement and the one worth defending: what stands now is five
    named rules rather than an empty ledger.
    """
    u, v = _field()
    g, _e = F.build(u, v, motion=False)
    r = compile_scheme(g)

    #: the ingest path works: every bound constant is declared, with provenance
    assert not (getattr(r, "unmeasured", ()) or ()), r.unmeasured
    for name in ("L", "sigma", "C_mu", "tau", "gamma", "norm_A"):
        assert getattr(g.measured, name) is not None, name
    assert g.measured.source and "w157" in g.measured.source.lower()

    #: and it is still not certified, for reasons that are now NAMED
    assert r.verdict.value == "admit-uncertified", r.verdict
    left = {d.rule for d in r.decisions if d.verdict.value != "admit"}
    assert left == {"C2", "R10", "R10/halo", "E7/passivity", "R12"}, left
    #: three of those five are recorded checker defects rather than findings
    #: about this assembly -- W136 (the halo rule over-fires on a physical
    #: boundary) and W138 (passivity is an orientation artefact). A test that
    #: let them quietly disappear would hide the two open rows that explain
    #: most of what this panel shows.
    assert {"R10/halo", "E7/passivity"} <= left


def test_the_verdict_does_not_move_over_the_design_box():
    """Measured, and it is what the demo must not overstate: the colour is a
    property of the DECLARATION, not of the design point."""
    c = _stage("compile")
    assert c["design_corners"] >= 16
    assert c["verdict_moves_with_design"] == 0


def test_the_stress_ramp_is_anchored_on_the_ceiling_and_not_on_the_frame():
    """A ramp that renormalises every frame makes a design at 20% of the ceiling
    look identical to one at 99% of it, which is the one thing the panel exists
    to distinguish."""
    from atlas.demo_frontwing.engine import stress_colour
    lo = stress_colour(0.2 * F.SIGMA_CEIL)
    hi = stress_colour(0.99 * F.SIGMA_CEIL)
    over = stress_colour(4.0 * F.SIGMA_CEIL)
    assert lo != hi
    assert over == stress_colour(F.SIGMA_CEIL)      # clamped, not renormalised


def test_the_demo_engine_marches_optimises_and_publishes_what_the_page_draws():
    """Headless: no server, no socket.  The engine is the demo's only stateful
    part, and a page that cannot be drawn from its payload is a demo that looks
    broken for a reason nobody can see."""
    if not os.path.isfile(_FIELD):
        pytest.skip("out/w141/settled.npz is absent")
    torch.set_num_threads(1)
    from atlas.demo_frontwing.engine import DemoConfig, Engine
    eng = Engine(DemoConfig(tiling="six", coupling="tight", horizon=3))
    try:
        for _ in range(2):
            eng._march_once()
        assert eng.load > 0.0 and eng.vm > 0.0
        assert float(eng.h) > 0.0

        before = dict(eng.design)
        eng._optimise_once()
        eng._drain()
        assert eng.design != before, "the optimiser did not move any knob"
        for k in F.DESIGN_KEYS:
            lo, hi = F.DESIGN_BOX[k]
            assert lo <= eng.design[k] <= hi, (k, eng.design[k])
        # both margin pairs are reported, and the smooth one is never BELOW the
        # true one -- logsumexp overshoots, which is the safe direction
        m = eng.margins
        assert m["smooth_sigma"] >= m["true_sigma"] - 1e-12
        assert m["smooth_delta"] >= m["true_delta"] - 1e-12

        eng._publish()
        p = eng.frame.payload
        for key in ("seq", "mode", "design", "box", "config", "readout",
                    "geometry", "seam_quantities", "certification", "trace",
                    "domain", "ramp"):
            assert key in p, key
        assert eng.frame.png and eng.frame.width == F.NX
        g = p["geometry"]
        assert g["n"] == F.N_STATION - 1
        assert all(len(q["p"]) == 4 and q["colour"].startswith("#")
                   for q in g["quads"])
        # a cantilever: the clamped end carries the stress and the tip does not
        assert g["quads"][0]["vm"] > 10.0 * g["quads"][-1]["vm"]
        r = p["readout"]
        assert r["vm_ceiling"] == F.SIGMA_CEIL
        assert r["delta_ceiling"] == F.DELTA_CEIL
        assert abs(r["stress_margin"] - (r["vm_max"] / F.SIGMA_CEIL - 1)) < 1e-9
    finally:
        eng.stop()


def test_the_design_box_thickness_stays_inside_what_the_flow_model_neglects():
    """The flow carries the plate as a LINE of stations with no thickness at
    all, so a plate the grid would resolve is one the flow model may not
    neglect.  The first design run drove to `t/c = 0.080`, which is 2.56 cells;
    the bound is now where the plate is about ONE cell thick."""
    lo, hi = F.DESIGN_BOX["tc"]
    cells_hi = hi * F.CHORD / F.DX
    assert cells_hi <= 1.6, cells_hi
    assert lo * F.CHORD / F.DX >= 1.0


def test_the_ceilings_are_inside_the_reachable_range():
    """A ceiling the search never reaches is not a constraint.  Both are set
    above the reference design's own value -- so the start is feasible -- and
    below what the downforce-seeking direction produces."""
    s = _stage("setup")
    assert F.SIGMA_CEIL < 6.0e2, "the loose ceiling was never reached"
    assert F.DELTA_CEIL < 0.040
    assert F.DELTA_CEIL < F.DELTA_MAX


# ---------------------------------------------------------------------------
# 8. the constrained design search, and whether the word is earned
# ---------------------------------------------------------------------------


def _design() -> dict:
    return _stage("design")


def test_the_reported_optimum_is_STRICTLY_feasible_on_the_TRUE_margins():
    """The smooth maximum is a gradient device, not a feasibility test.  W142:
    a `logsumexp` overshoots by temp*log(n), so a point feasible on the smooth
    margins can violate the real ceiling.  What the page reports has to be
    feasible on the margins the ceilings are actually written in."""
    best = _design()["gradient"]["best_feasible"]
    assert best["feasible"] is True
    assert best["true_sigma"] <= 0.0, best["true_sigma"]
    assert best["true_delta"] <= 0.0, best["true_delta"]
    pop = _design()["population"]["best_feasible"]
    assert pop["true_sigma"] <= 0.0 and pop["true_delta"] <= 0.0


def test_what_binds_the_primary_search_is_the_ENVELOPE_and_not_a_ceiling():
    """W145 changed the answer to "is the word constrained earned", and this test
    is where the old answer lived.

    Before the envelope check reached `objective`, the search walked out of the
    model and rode the stress ceiling to 0.016%.  With the check live it is
    stopped by `Suspension.validity`'s floor instead: NEITHER named ceiling is
    active, and the honest report is that the binding constraint is a validity
    bound.  `design_tight` is where a ceiling is what the search meets first.
    """
    a = _design()["active"]
    assert a["sigma"] is False, a
    assert a["delta"] is False, a
    #: near the stress ceiling but not on it -- a couple of per cent of headroom
    assert -0.10 < a["sigma_margin"] < -0.005, a["sigma_margin"]
    #: and the envelope is what did stop it: declines on both columns
    d = _design()
    n_dec = sum(1 for h in d["gradient"]["history"] if h.get("declined"))
    assert n_dec >= 3, n_dec
    #: the population loop records only the numeric fields, so a declined
    #: sample is identified by the sentinel `_evaluate` returns for one
    dec_pop = sum(1 for h in d["population"]["history"]
                  if h["L"] <= -999.0)
    assert dec_pop >= 10, dec_pop


def test_the_TIGHT_column_does_ride_its_ceiling():
    """The constrained-search capability, demonstrated on a column where the
    ceiling is what the search meets before the envelope.  Same box, same start,
    same weight, same budget, same seed -- one number changed."""
    t = _stage("design_tight")
    assert t["sigma_ceiling"] < F.SIGMA_CEIL, t["sigma_ceiling"]
    assert t["active"]["sigma"] is True, t["active"]
    assert abs(t["active"]["sigma_margin"]) < 0.02, t["active"]["sigma_margin"]
    #: and the penalty actually fires here, which it never does in `design`
    viol = [h for h in t["gradient"]["history"]
            if not h.get("declined") and h.get("true_sigma", -1) > 0]
    assert len(viol) >= 3, len(viol)
    #: while the envelope barely does
    assert sum(1 for h in t["gradient"]["history"] if h.get("declined")) <= 3


def test_the_deflection_ceiling_is_NOT_active_on_any_column():
    """The other half of the same honesty, and it survived W145 unchanged: the
    downforce-seeking direction runs AWAY from the deflection ceiling, because
    the porous plate has no thickness in the flow so thickening costs the
    aerodynamics nothing (section 2.1)."""
    for key in ("design", "design_tight"):
        a = _stage(key)["active"]
        assert a["delta"] is False, (key, a)
        assert a["delta_margin"] < -0.4, (key, a["delta_margin"])


def test_the_loose_run_is_kept_because_its_outcome_is_a_result():
    """`design_loose` is the first search: both ceilings inactive at an optimum
    on every box bound.  It is evidence about the model and is not deleted."""
    loose = _stage("design_loose")
    assert loose["ceilings"]["sigma"] > F.SIGMA_CEIL
    assert loose["ceilings"]["delta"] > F.DELTA_CEIL
    b = loose["gradient"]["best_feasible"]
    assert b["true_sigma"] < -0.5 and b["true_delta"] < -0.5, b
    assert "note" in loose and "box search" in loose["note"]


def test_the_baseline_is_CMA_ES_and_not_a_hand_rolled_sampler():
    """A comparison is only worth reporting against the strongest baseline
    available, and PoC 1a's is available.  The first draft of this stage rolled
    its own Gaussian sampler with a shrinking sigma, which would have flattered
    the gradient column."""
    assert _design()["population"]["optimiser"] == "cma-es"
    #: and the loose run, which predates the swap, must NOT claim otherwise
    assert _stage("design_loose")["population"].get("optimiser") != "cma-es"


def test_the_three_ratios_are_re_derivable_from_the_recorded_histories():
    """`_compare` is the whole comparison arithmetic and it reads only the two
    stored histories, so a correction to it needs no re-march -- which is why
    `stage_recompare` exists.  Re-running it here must reproduce what the
    artifact carries."""
    sys.path.insert(0, os.path.join(_ROOT, "scripts"))
    import importlib
    drv = importlib.import_module("w141_poc2_frontwing")
    d = _design()
    if "comparison" not in d:
        pytest.skip("run `--stages recompare`")
    fresh = drv._compare(d)
    for k, v in d["comparison"].items():
        if isinstance(v, float):
            assert abs(fresh[k] - v) < 1e-9 * max(1.0, abs(v)), k
        else:
            assert fresh[k] == v, k


def test_the_flattering_ratio_is_recorded_but_is_not_the_headline():
    """W143.  Dividing by the population's WHOLE budget credits the gradient for
    evaluations the baseline spent after it had already peaked.  Both forms are
    recorded and the page quotes the smaller one; and the wall-clock ratio is
    smaller still, because one Adam iterate costs a forward march AND a reverse
    sweep while a population evaluation costs the march alone."""
    d = _design()
    if "comparison" not in d:
        pytest.skip("run `--stages recompare`")
    c = d["comparison"]
    assert c["pop_evals_to_best"] < c["pop_evals_used"]
    assert c["ratio_to_pop_best"] < c["ratio_full_budget"]
    assert c["ratio_wall"] < c["ratio_to_pop_best"]
    #: the premium is a MEASURED end-to-end quantity, not `stage_cost`'s
    #: per-macro-step figure, and it must be above one or the adjoint is free
    assert c["grad_iterate_in_pop_evals"] > 1.0
    #: the legacy field is the flattering one and stays labelled as such
    assert abs(d["ratio"] - c["ratio_full_budget"]) < 1e-9


def test_the_exterior_penalty_never_fires_in_the_primary_column():
    """It cannot: no iterate reaches a ceiling, because the envelope declines the
    trajectory first.  So J_pen == L everywhere, the weight multiplies zero, and
    the weight sweep returns the same number three times -- which is the finding,
    not a bug.  This test is what keeps that from being read as a bug later."""
    d = _design()
    for h in d["gradient"]["history"]:
        if h.get("declined"):
            continue
        assert h["true_sigma"] <= 0.0, h
        assert abs(h["J"] - h["L"]) < 1e-12, h
    rows = _stage("penalty")["rows"]
    assert len(rows) >= 3
    js = {round(r["last"]["J"], 9) for r in rows.values()}
    assert len(js) == 1, js


def test_the_exterior_penaltys_optimum_sits_OUTSIDE_where_it_can_act():
    """Textbook, and visible on the two columns where an iterate can reach a
    ceiling: the converged point violates by an amount set by w, so the reported
    answer is the best strictly FEASIBLE iterate and never the final one."""
    prev = _stage("penalty_unenforced")["rows"]
    viol = {w: r["violation"] for w, r in prev.items()}
    assert viol["10.0"] > viol["40.0"] > viol["160.0"], viol
    assert viol["10.0"] > 0.02, viol
    #: and in the tight column the trajectory crosses the ceiling repeatedly,
    #: so the penalised objective is above the reported answer at every
    #: violating iterate -- which is why the reported answer is the best
    #: strictly feasible one and not the largest J on the trace
    t = _stage("design_tight")
    hist = [h for h in t["gradient"]["history"] if not h.get("declined")]
    viol = [h for h in hist if h["true_sigma"] > 0.0]
    assert len(viol) >= 3, len(viol)
    best = t["gradient"]["best_feasible"]["J"]
    assert max(h["J"] for h in viol) > best, (max(h["J"] for h in viol), best)


def test_the_search_improved_the_objective_it_was_given():
    """+19.3% on the primary column.  The pre-W145 run read +27.5% and 6.9 points
    of that were bought outside the model's declared envelope, so this bound is
    lower than it was and deliberately so."""
    d = _design()
    assert d["gradient"]["best_feasible"]["J"] > 1.15 * d["start_eval"]["J"]
    assert d["start_eval"]["feasible"] is True, "the start must be feasible"


def test_the_ablation_at_the_optimum_FALSIFIED_the_inference_it_replaced():
    """Section 7.2 wanted to say that a search which stiffens the wing shrinks
    what the elastic seam is worth.  Measured, it went the other way and on the
    other seam: the SUSPENSION seam goes from 0.49% of the downforce at the
    reference design to 8.16% at the optimum, because the optimum rides in deep
    ground effect where dL/dh is far larger.  This test holds the measured
    direction, so that re-running it somewhere else has to confront it."""
    ab = _stage("ablation")
    o = ab["at_optimum"]
    assert set(o["design"]) == set(F.DESIGN_KEYS)
    susp = "no suspension (CS-12 alone, k -> inf)"
    assert susp in o["worth"] and susp in o["worth_at_reference"]
    assert o["worth"][susp] > 10.0 * o["worth_at_reference"][susp], o["worth"]
    #: the frozen column has to sit at the height the LIVE design settles to,
    #: or it is measuring a transit rather than the seam
    assert abs(o["frozen"][susp]["h"] - o["live"]["h"]) < 1e-6, o
    #: and the elastic column could not be measured at all: a rigid wing there
    #: makes MORE downforce and drives itself through the floor
    elastic = "no structure (CS-10 alone, E* -> inf)"
    assert o["frozen"][elastic]["ok"] is False, o["frozen"][elastic]
    assert "envelope" in o["frozen"][elastic]["why"]


def test_the_artifact_records_the_box_and_ceilings_the_MODULE_actually_has():
    """The stages are re-runnable one at a time, which means a stage can be left
    behind by a change to `front_wing.py` -- and it was.  Narrowing the design
    box and lowering both ceilings after the first search left `setup` in the
    artifact still recording `t/c` up to 0.080 and a stress ceiling of 600,
    which the results page cites.  A stale stage is worse than a missing one,
    because it reads as a measurement.  Re-run `--stages setup` after any change
    to the box or the ceilings."""
    s = _stage("setup")
    assert s["sigma_ceiling"] == F.SIGMA_CEIL, "stale setup: re-run --stages setup"
    assert s["delta_ceiling"] == F.DELTA_CEIL, "stale setup: re-run --stages setup"
    for k, (lo, hi) in F.DESIGN_BOX.items():
        assert s["design_box"][k] == [lo, hi], (k, s["design_box"][k])
    #: and the design stage has to agree with it, since the search is what the
    #: box is FOR
    d = _design()
    assert d["ceilings"]["sigma"] == F.SIGMA_CEIL
    assert d["ceilings"]["delta"] == F.DELTA_CEIL
    for k, (lo, hi) in F.DESIGN_BOX.items():
        assert d["box"][k] == [lo, hi], (k, d["box"][k])


def test_every_knobs_adjoint_agrees_with_a_central_difference():
    """Four knobs, seven steps each, 56 rollouts.  The thickness is the one that
    matters most: its adjoint is NOT pure autograd but a central difference on
    the surface operator spliced into the exact adjoint through everything else,
    and a wrong sign or a missing chain-rule factor there would not crash."""
    f = _stage("fdsweep")
    assert set(f["knobs"]) == set(F.DESIGN_KEYS)
    for k, rec in f["knobs"].items():
        assert rec["best"]["rel_err"] < 1.0e-4, (k, rec["best"])
        #: and the sign, which is the failure a small relative error can hide
        #: only if both are wrong the same way
        assert rec["adjoint"] * rec["best"]["fd"] > 0.0, k
    assert f["worst_best"] < 1.0e-4


def test_the_finite_difference_sweep_is_not_claimed_to_be_a_FLOOR():
    """[[poc1-results-differentiable-design]] §5's classical column got a
    textbook curve either side of a minimum and could quote a floor.  Two of
    these four branches are NOT monotone and the other two had not bottomed out
    at the smallest step tested, so the page quotes a best-over-the-sweep and
    says so.  This test exists to fail if someone later reports the number as a
    floor without re-measuring."""
    f = _stage("fdsweep")
    assert any(not rec["monotone"] for rec in f["knobs"].values()), (
        "every branch is now monotone -- the page's caveat may be re-checked")
    #: the smallest step tested is still the best for three of the four, which
    #: is what "had not bottomed out" means
    smallest = min(f["fd_steps"])
    at_smallest = sum(1 for rec in f["knobs"].values()
                      if rec["best"]["rel"] == smallest)
    assert at_smallest >= 3, at_smallest


def test_the_two_bad_FD_knobs_are_exactly_the_two_that_set_the_release_height():
    """W144.  Each design is released at h = h0 - LOAD_REF/k, which is a function
    of `h0` and `k` and of NEITHER structural knob; and `FlexWing.forcing` stamps
    the plate's kernels on an INTEGER box whose corner is a step function of that
    height.  So the objective is piecewise smooth in exactly those two knobs, and
    a wide central difference measures the jump the adjoint is blind to.  This
    test is the discriminator: if a structural knob ever joins the bad set, the
    mechanism on the page is wrong."""
    f = _stage("fdsweep")
    widest = max(f["fd_steps"])
    err = {k: next(s["rel_err"] for s in rec["sweep"] if s["rel"] == widest)
           for k, rec in f["knobs"].items()}
    sets_height = {"h0", "k"}
    assert min(err[k] for k in sets_height) > 10.0 * max(
        err[k] for k in set(F.DESIGN_KEYS) - sets_height), err
    #: and the release height really is a function of those two alone
    a = F.FrontWingRollout.release_height(0.32, 2.5)
    assert a == F.FrontWingRollout.release_height(0.32, 2.5)
    assert F.FrontWingRollout.release_height(0.33, 2.5) != a
    assert F.FrontWingRollout.release_height(0.32, 2.6) != a


def test_the_ride_height_knobs_FD_branch_is_not_second_order():
    """A central difference of a smooth function converges at O(h^2): one decade
    of step buys two decades of error.  These do not, which is the measurement
    behind W144 -- and CS-10's own published table for the same knob fits an
    order of about ONE, which it read as a clean truncation branch."""
    f = _stage("fdsweep")
    for k in ("h0", "k"):
        sw = f["knobs"][k]["sweep"]
        slopes = [math.log(a["rel_err"] / b["rel_err"]) / math.log(a["step"] / b["step"])
                  for a, b in zip(sw, sw[1:])]
        mean = sum(slopes) / len(slopes)
        assert mean < 1.8, (k, mean, slopes)
        #: and it is ERRATIC rather than a consistent lower order
        assert max(slopes) - min(slopes) > 1.0, (k, slopes)


# ---------------------------------------------------------------------------
# 9. W145: an envelope enforced on one path and not the other
# ---------------------------------------------------------------------------


def _out_of_envelope_design() -> dict:
    """A design whose settled ride height is under `Suspension.validity`'s floor.

    Constructed rather than taken from the artifact, so the guard does not
    depend on a search having produced one.  h = h0 - L/k, so a soft spring at a
    low free height buys a ride height the box on h0 never bounded -- which is
    the whole mechanism of W145.
    """
    d = dict(F.DESIGN_REF)
    d["h0"], d["k"] = 0.2227, 1.5352
    assert F.FrontWingRollout.release_height(d["h0"], d["k"]) > F.G.H_FLOOR, (
        "the design must START inside the envelope and be walked out of it, or "
        "this tests the release height rather than the march")
    return d


def test_run_and_objective_decline_at_the_SAME_design_and_the_SAME_step():
    """W145.  `run` checked both experts' declared envelopes at every macro-step
    and `objective` checked neither -- and the design search calls `objective`.
    So the search optimised its way through the suspension's floor and nothing
    fired.  A framework that CAN decline and does not decline on the path a
    search uses is worse than one that cannot, because the search is the thing
    that goes looking for the edge."""
    u, v = _field()
    d = _out_of_envelope_design()
    msgs = []
    for call in ("run", "objective"):
        r = F.FrontWingRollout(tiling=F.DEFAULT_TILING, coupling="tight",
                               motion=True, design=d)
        with pytest.raises(RuntimeError) as exc:
            if call == "run":
                r.run(steps=10, u0=u, v0=v)
            else:
                r.objective(d, steps=10, u0=u, v0=v, grad=False)
        msgs.append(str(exc.value))
    #: same envelope, same macro-step -- one check, called from two places
    for m in msgs:
        assert "suspension expert's declared envelope" in m, m
    step = [m.split("macro-step ")[1].split(":")[0] for m in msgs]
    assert step[0] == step[1], (step, msgs)


def test_the_reference_design_marches_and_is_not_declined():
    """The guard above is only worth anything if the check is not simply always
    on."""
    u, v = _field()
    r = F.FrontWingRollout(tiling=F.DEFAULT_TILING, coupling="tight",
                           motion=True)
    res = r.run(steps=10, u0=u, v0=v)
    assert float(res.h[-1]) > F.G.H_FLOOR


def test_the_envelope_check_is_bitwise_neutral_where_it_does_not_fire():
    """The check reads DETACHED values, so it cannot enter the tape or move a
    number.  The reference design's J at N = 20 is on the results page."""
    u, v = _field()
    r = F.FrontWingRollout(tiling=F.DEFAULT_TILING, coupling="tight",
                           motion=True)
    J = float(r.objective(dict(F.DESIGN_REF), steps=20, u0=u, v0=v,
                          grad=False)[0])
    assert abs(J - 0.20810365) < 1e-8, J


def test_the_reported_optimum_stays_inside_the_declared_ENVELOPE():
    """The constraint the page reports is the stress ceiling; the envelope is not
    a constraint at all but a bound on where the model means anything, and an
    optimum outside it is not an answer.  `h = h0 - L/k`, so bounding `h0` --
    which the design box does -- does NOT bound `h`."""
    d = _design()
    best = d["gradient"]["best_feasible"]
    h = best["design"]["h0"] - best["L"] / best["design"]["k"]
    assert h > F.G.H_FLOOR, (
        f"the reported optimum settles at h = {h:.5g}, under the suspension's "
        f"declared floor of {F.G.H_FLOOR:.5g} -- see W145")
    pop = d["population"]["best_feasible"]
    h_p = pop["design"]["h0"] - pop["L"] / pop["design"]["k"]
    assert h_p > F.G.H_FLOOR, h_p


def test_the_pre_W145_run_is_kept_as_the_evidence():
    """It is what happened, and the difference between it and the enforced run is
    the measurement of what the missing check was worth."""
    prev = _stage("design_unenforced")
    b = prev["gradient"]["best_feasible"]
    h = b["design"]["h0"] - b["L"] / b["design"]["k"]
    assert h < F.G.H_FLOOR, h
    assert "W145" in prev.get("note", "")


def test_the_demos_paths_check_the_same_envelopes():
    """W145's third path.  `_march_once` and `_optimise_once` both drive
    `macro_step` directly, so neither inherited `run`'s checks: the live
    optimiser could walk the wing through the suspension's declared floor and
    the screen would show a downforce for a design the model declines.  Every
    path funnels through `_absorb`, so the check lives there."""
    import inspect
    from atlas.demo_frontwing import engine as E
    src = inspect.getsource(E.Engine._absorb)
    assert "check_envelopes" in src, "the demo bypasses the envelope again"
    #: and the two envelope messages BOTH contain the word "envelope", so the
    #: classifier has to test the more specific one first or every ride-height
    #: decline is labelled a structural one
    loop = inspect.getsource(E.Engine._loop)
    i_susp = loop.index('"suspension" in msg')
    i_env = loop.index('"envelope" in msg')
    assert i_susp < i_env, "the suspension decline is being mislabelled"
