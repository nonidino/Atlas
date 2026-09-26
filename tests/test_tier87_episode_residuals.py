"""Tier 87 -- W336: the rocket episode's residuals, and the error budget behind them.

`scripts/w336_episode_residuals.py` re-runs the marched rocket episode under a
ledger that captures the face fluxes the solver itself applied at every block
boundary, and reads the run's conservation off them. Before any of its numbers
mean anything, the instrument has to be shown to read a flow whose fluxes are
known by hand, to leave the numerics alone, and to be able to fail. That is
what the first half of this file pins. The second half pins the records.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os

import numpy as np
import pytest

from atlas.cases import rocket_experts as RE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "w336")
#: Tier 87's records as recorded. Tier 88 (2026-09-26) re-ran the audit on the
#: fixed code into out/w336 and moved these aside; the pins below are the
#: diagnosis, and tests/test_tier88_episode_fixes.py pins the fix.
PRE = os.path.join(ROOT, "out", "w336_pre_w337")
PRE_RECORD = os.path.join(ROOT, "out", "w321_pre_w337", "episode.npz")


def _driver():
    path = os.path.join(ROOT, "scripts", "w336_episode_residuals.py")
    spec = importlib.util.spec_from_file_location("w336_driver", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _have_expert() -> bool:
    try:
        RE.load_solvers()
        return True
    except Exception:
        return False


needs_expert = pytest.mark.skipif(
    not _have_expert(), reason="the build repo is not importable; set ATLAS_BUILD_REPO")


def _stream_block(W):
    """A 20 x 10 channel, 1.0 m by 0.5 m, a uniform stream at 100 m/s entering
    at imin and leaving at imax, slip walls on both j faces."""
    C2 = importlib.import_module("atlas_build_solvers.solvers.compressible2d")
    grid = importlib.import_module("atlas_build_solvers.solvers.grid")
    thermo = importlib.import_module("atlas_build_solvers.solvers.thermo")
    z, y = np.linspace(0.0, 1.0, 21), np.linspace(0.0, 0.5, 11)
    Z, Y = np.meshgrid(z, y, indexing="ij")
    blk = grid.Block("stream", np.stack([Z, Y], -1), np.zeros((20, 10), bool))
    prim = np.array([1.0, 100.0, 0.0, 1.0e5])
    s = C2.Compressible2D(blk, C2.GasConfig(gamma=1.4, R=287.0, inviscid=False), bcs={
        "imin": C2.BC("freestream", {"prim": prim}), "imax": C2.BC("outflow"),
        "jmin": C2.BC("wall_slip"), "jmax": C2.BC("wall_slip")})
    U0 = np.broadcast_to(thermo.prim_to_cons(prim, 1.4), (20, 10, 4)).copy()
    return s, U0, W


# ===========================================================================
# 1. the instrument reads what it should, and only reads
# ===========================================================================


@needs_expert
def test_the_ledger_reads_a_uniform_stream_by_hand():
    """rho u H dt in at imin, the same out at imax, nothing through the walls,
    and the block's content unchanged -- all to round-off."""
    W = _driver()
    M = W._mods()
    led = W.Ledger(M)
    led.install()
    try:
        s, U0, _ = _stream_block(W)
        U, n = s.advance(U0, 1.0e-4)
        acc = led.acc_for(s)
        assert acc["substeps"] == n > 0
        m_in = float(W._sum_side(acc, "imin")[0])
        m_out = float(W._sum_side(acc, "imax")[0])
        assert m_in == pytest.approx(1.0 * 100.0 * 0.5 * 1.0e-4, rel=1e-12)
        assert m_out == pytest.approx(-m_in, rel=1e-12)
        for side in ("jmin", "jmax"):
            assert np.abs(acc["inv"][side][:, 0]).max() < 1e-15
        res, scale, _d, _i = W.block_budget(s, acc, W._totals(s, U0), W._totals(s, U))
        assert np.all(np.abs(res) <= 1e-12 * scale)
    finally:
        led.uninstall()


@needs_expert
def test_the_ledger_does_not_change_the_numerics():
    """The wrappers call the originals and read their outputs; a march with the
    ledger installed is bitwise the march without it."""
    W = _driver()
    M = W._mods()
    s, U0, _ = _stream_block(W)
    U0 = U0.copy()
    U0[5:8, 3:6, 3] *= 1.05                         # something to march
    ref, _ = s.advance(U0, 5.0e-5)
    led = W.Ledger(M)
    led.install()
    try:
        got, _ = s.advance(U0, 5.0e-5)
    finally:
        led.uninstall()
    assert np.array_equal(ref, got)
    C2 = M["C2"]
    assert not getattr(C2.Compressible2D, "_w336_installed", False)
    assert not hasattr(C2.FLUXES["hllc"], "__wrapped__")
    assert C2.Compressible2D._inviscid_face_fluxes.__qualname__ == "Compressible2D._inviscid_face_fluxes"


@needs_expert
def test_a_budget_with_a_side_left_out_does_not_close():
    """The control that gives gate A1 its teeth: drop the inflow side from the
    account and the residual is exactly that side's inflow."""
    W = _driver()
    M = W._mods()
    led = W.Ledger(M)
    led.install()
    try:
        s, U0, _ = _stream_block(W)
        U, _ = s.advance(U0, 1.0e-4)
        acc = led.acc_for(s)
        kept = {k: (dict(v) if isinstance(v, dict) else v) for k, v in acc.items()}
        kept["inv"] = {k: (np.zeros_like(v) if k == "imin" else v) for k, v in acc["inv"].items()}
        kept["visc"] = {k: (np.zeros_like(v) if k == "imin" else v) for k, v in acc["visc"].items()}
        res, scale, _d, _i = W.block_budget(s, kept, W._totals(s, U0), W._totals(s, U))
        assert abs(res[0]) > 1e6 * 1e-12 * scale[0]
        assert res[0] == pytest.approx(float(W._sum_side(acc, "imin")[0]), rel=1e-9)
    finally:
        led.uninstall()


def _combustor_strip():
    """Tier 88: a 10 x 4 strip of burnt 3000 K gas at 8 MPa with the reaction
    ON and a 900 K inlet_massflow face on imin -- the rocket's injector, which
    no Riemann call computes any more (W337)."""
    C2 = importlib.import_module("atlas_build_solvers.solvers.compressible2d")
    grid = importlib.import_module("atlas_build_solvers.solvers.grid")
    thermo = importlib.import_module("atlas_build_solvers.solvers.thermo")
    z, y = np.linspace(0.0, 0.1, 11), np.linspace(-0.02, 0.02, 5)
    Z, Y = np.meshgrid(z, y, indexing="ij")
    blk = grid.Block("strip", np.stack([Z, Y], -1), np.zeros((10, 4), bool))
    gas = C2.GasConfig(gamma=1.22, R=361.0, inviscid=False)
    mdot = 376.87 / 0.08
    s = C2.Compressible2D(blk, gas, reaction=C2.ReactionConfig(A=2.0e6, T_act=6000.0, q_rxn=3.0e6),
                          bcs={"imin": C2.BC("inlet_massflow", {"mdot": mdot, "T": 900.0}),
                               "imax": C2.BC("outflow", {"p_inf": 7.9e6}),
                               "jmin": C2.BC("wall_noslip", {"T_wall": np.full(10, 400.0)}),
                               "jmax": C2.BC("wall_noslip")})
    W_ = np.zeros((10, 4, 5))
    W_[..., 0] = 8.0e6 / (361.0 * 3000.0)
    W_[..., 1] = mdot / W_[..., 0]
    W_[..., 3] = 8.0e6
    W_[..., 4] = 0.9
    return s, thermo.prim_to_cons(W_, 1.22), mdot


@needs_expert
def test_the_ledger_reads_the_prescribed_injector_flux():
    """Tier 88. The injector face's flux is prescribed after the Riemann call
    (W337), so a ledger that only wrapped the Riemann functions would book the
    wrong inflow. This one reads the arrays the residual differences: the
    injector passes exactly mdot H t, and a burning, walled block's budget --
    reaction, isothermal and adiabatic no-slip walls, outflow -- still closes."""
    W = _driver()
    M = W._mods()
    led = W.Ledger(M)
    led.install()
    try:
        s, U0, mdot = _combustor_strip()
        U, n = s.advance(U0, 2.0e-5)
        acc = led.acc_for(s)
        assert n > 1
        assert float(W._sum_side(acc, "imin")[0]) == pytest.approx(mdot * 0.04 * 2.0e-5, rel=1e-12)
        res, scale, _d, _i = W.block_budget(s, acc, W._totals(s, U0), W._totals(s, U))
        assert np.all(np.abs(res) <= 1e-12 * scale)
        assert abs(acc["react"][3]) > 0.0, "the reaction ran"
    finally:
        led.uninstall()


@needs_expert
def test_revert_puts_each_old_behaviour_back_and_restore_undoes_it():
    """The controls' machinery: each fix reverted shows its diagnosis, and
    `restore_fixes` brings every fix back."""
    W = _driver()
    M = W._mods()
    C2, TS, gen = M["C2"].Compressible2D, M["TS"].ThermoStruct2D, M["gen"]
    fixed = (C2._inlet_face_flux, C2._isothermal_wall_conduction, TS._outer_robin,
             gen.CoupledEpisode._wire_engine, gen.CoupledEpisode._wire_external)
    s, U, mdot = _combustor_strip()
    try:
        Fi, _ = s._inviscid_face_fluxes(s.ghost(U, thermal=False))
        assert np.allclose(Fi[0, :, 0] / s.block.a_i[0], mdot, rtol=1e-14)
        assert W.revert(M, "W337") == ["W337"]
        Fi, _ = s._inviscid_face_fluxes(s.ghost(U, thermal=False))
        assert np.all(Fi[0, :, 0] / s.block.a_i[0] > 1.1 * mdot)       # the Riemann overshoot
        assert W.revert(M, "all") == list(W.FIXES)
        assert TS._outer_robin is W._old_outer_robin
        assert gen.CoupledEpisode._wire_engine.__qualname__.startswith("_old_wiring")
        with pytest.raises(ValueError):
            W.revert(M, "W999")
    finally:
        W.restore_fixes()
    now = (C2._inlet_face_flux, C2._isothermal_wall_conduction, TS._outer_robin,
           gen.CoupledEpisode._wire_engine, gen.CoupledEpisode._wire_external)
    assert now == fixed


# ===========================================================================
# 2. the registered predictions and their evaluator cannot drift apart
# ===========================================================================


def _step(t, good, early=False):
    g = good
    return dict(
        t=t,
        telescoping_rel={"b0": [1e-14 if g else 1e-6] * 4},
        engine_wall_mass_rel=1e-16 if g else 1e-3,
        airframe_wall_mass_rel=1e-17 if g else 1e-3,
        shell={"0": dict(energy_residual_over_stored=1e-12 if g else 1e-6)},
        record_dev={"rigid": 1e-12 if g else 1e-6},
        injector_mdot_over_declared=1.001 if g else 1.10,
        seam_ab_mass_rel=(0.01 if early else 1e-3) if g else (0.01 if early else 0.05),
        seam_be_mass_rel=(0.01 if early else 1e-3) if g else (0.01 if early else 0.05),
        thrust_routes_rel=1e-3 if g else 0.1,
        thrust_cells_vs_face_rel=1e-4 if g else 0.1,
        thermal_engine={"jmin": dict(rel=1e-3 if g else 0.1), "jmax": dict(rel=1e-3 if g else 0.1)},
        vehicle_mass_loss_over_exit=1.001 if g else 1.1,
        thermal_airframe={"0": dict(rad_slip_over_outer=0.01 if g else 0.2)},
        rigid=_rigid(t, g))


def _rigid(t, good=True):
    """A rigid state whose lateral parts are round-off, like a symmetric flight."""
    y = 25000.0 + 1044.0 * t
    return [1e-21, y if good else y * (1.0 + 1e-6), 1.5707963267948966, -3e-18, 1000.0, 0.0,
            50000.0 - 377.0 * t]


def _write_records(d, good):
    def put(name, obj):
        with open(os.path.join(d, name), "w", encoding="utf-8") as fh:
            json.dump(obj, fh)
    audit = [_step(round(0.005 * k, 6), good, early=0.005 * k <= 0.06 + 1e-9) for k in range(1, 30)]
    put("audit.json", dict(steps=audit))
    rec = np.array([_rigid(round(0.005 * k, 6)) for k in range(0, 30)])
    rec[:, 0] = 7e-21                    # lateral round-off that differs, as it does
    rec[:, 3] = 5e-18
    np.savez(os.path.join(d, "episode.npz"), rigid=rec)
    put("control_wall.json", dict(steps=[dict(engine_wall_mass_rel=0.5)]))
    for name, dt, seam in (("dt_10ms.json", 0.01, 0.02 if good else 0.01),
                           ("dt_2p5ms.json", 0.0025, 0.005 if good else 0.01)):
        steps = [dict(t=round(dt * k, 6), seam_ab_mass_rel=seam, seam_be_mass_rel=seam,
                      rigid=[0.0, 0.0, 0.0, 0.0, 1000.00001 if good else 1001.0, 0.0, 0.0])
                 for k in range(1, int(round(0.06 / dt)) + 1)]
        put(name, dict(steps=steps))
    for c, thrust, heat, mdot in ((4, 1.00e6 if good else 0.8e6, 1.0, 1.0),
                                  (1, 1.02e6, 3.0 if good else 1.1, 1.01 if good else 1.2)):
        put("grid_engine_c%d.json" % c, dict(steps=[dict(
            throat_mdot_e_over_declared=mdot, thrust_exit_face_mean=thrust,
            engine_heat_lost={"jmin": heat, "jmax": heat})]))
    for c, drag in ((4, 800.0 if good else 600.0), (1, 820.0)):
        put("grid_external_c%d.json" % c, dict(steps=[dict(drag=drag)]))


@pytest.mark.parametrize("good", [True, False])
def test_every_registered_prediction_can_hold_and_can_fail(tmp_path, good):
    """Synthetic records built to hold every prediction, and built to fail every
    one: the evaluator has to say so, clause by clause. The standing rule after
    a gate and its evaluation were found disagreeing in the permissive
    direction (and again at Tier 81's R5)."""
    W = _driver()
    _write_records(str(tmp_path), good)
    res = W.evaluate(str(tmp_path), record=os.path.join(str(tmp_path), "episode.npz"))
    names = [k for k in W.PREDICTIONS]
    assert sorted(k for k in res if k in W.PREDICTIONS) == sorted(names)
    held = {k: res[k]["held"] for k in names}
    assert all(v is good for v in held.values()), held


def _write_records_t88(d, pre, good):
    """Tier 88's synthetic records: `_write_records`' set, with the fields the
    Tier 88 predictions read added -- built to hold each one, or to fail each
    one."""
    g = good
    _write_records(d, good)
    os.makedirs(pre, exist_ok=True)

    def load(n):
        with open(os.path.join(d, n), encoding="utf-8") as fh:
            return json.load(fh)

    def put(n, obj, where=d):
        with open(os.path.join(where, n), "w", encoding="utf-8") as fh:
            json.dump(obj, fh)

    ideal = dict(u_e=2300.0, p_e=5.0e5, M_e=2.4, p_c=8.0e6)
    mdot, A_e, p_inf = 376.87, 0.24, 2500.0
    alpha = np.arctan((0.12 - 0.04) / (0.70 - 0.40))
    F2 = (np.sin(alpha) / alpha * mdot * ideal["u_e"] + ideal["p_e"] * A_e) - p_inf * A_e

    def fill(s, dt_ms):
        early = s["t"] <= 0.06 + 1e-9
        k = {10: 4.0, 5: 2.0, 2.5: 1.0}[dt_ms] if g else 2.0      # lag: shrinks with the step
        s["seam_be_mass_rel"] = (2e-3 * k if early else 1e-3) if g else 0.03
        s["seam_ab_mass_rel"] = 0.01 if early else 1e-3
        rel = (1e-3 * k if early else 1e-3) if g else 0.01
        s["thermal_engine"] = {"jmin": dict(rel=rel), "jmax": dict(rel=-rel)}
        return s

    au = load("audit.json")
    au.update(ideal=ideal, mdot_declared=mdot, A_exit_z=A_e)
    for s in au["steps"]:
        fill(s, 5)
        s["injector_mdot_over_declared"] = 1.0 + (1e-9 if g else 2e-6)
        s["shell"] = {k: dict(energy_residual_over_stored=1e-12, q_rad_applied=3.0 + (1e-12 if g else 0.5),
                              q_rad_ambient=3.0) for k in ("0", "1")}
        s["chamber_p_mean"] = 8.0e6 * (1.01 if g else 1.10)
        s.update(p_inf=p_inf, throat_mdot_e_over_declared=1.0, exit_mach_massavg=2.4,
                 exit_u_massavg=2300.0, exit_p_areaavg=5.0e5,
                 thrust_exit_face_mean=float(F2 * (1.005 if g else 0.9)))
    put("audit.json", au)
    for name, dt_ms in (("dt_10ms.json", 10), ("dt_2p5ms.json", 2.5)):
        r = load(name)
        r["steps"] = [fill(s, dt_ms) for s in r["steps"]]
        put(name, r)
    for c in (4, 1):
        r = load("grid_engine_c%d.json" % c)
        for s in r["steps"]:
            s["injector_mdot_over_declared"] = 1.0 if g else 1.2
        put("grid_engine_c%d.json" % c, r)
    put("thermal_seam_diag.json", dict(sides={sd: dict(gas_conduction=600.0,
                                                       gas_work=6e-11 if g else 30.0)
                                              for sd in ("jmin", "jmax")}))
    old = [dict(rigid=_rigid(0.005 * k), injector_mdot_over_declared=1.1772,
                seam_be_mass_rel=-0.03, engine_heat_lost={"jmin": 626.2, "jmax": 626.2},
                shell={kk: dict(q_rad_applied=4.0, q_rad_ambient=3.05) for kk in ("0", "1")})
           for k in (1, 2)]
    put("audit.json", dict(steps=old), where=pre)
    ctl = json.loads(json.dumps(old))
    if not g:
        ctl[1]["rigid"][1] = np.nextafter(ctl[1]["rigid"][1], 1e9)      # one ulp of altitude
    put("control_old.json", dict(steps=ctl))


@pytest.mark.parametrize("good", [True, False])
def test_every_tier88_prediction_can_hold_and_can_fail(tmp_path, good):
    """Tier 88's predictions, registered before the fixed run, clause by clause:
    synthetic records built to hold each, and built to fail each."""
    W = _driver()
    d, pre = str(tmp_path / "out"), str(tmp_path / "pre")
    os.makedirs(d)
    _write_records_t88(d, pre, good)
    res = W.evaluate_t88(d, pre_dir=pre)
    assert sorted(res) == sorted(W.PREDICTIONS_T88)
    held = {k: res[k]["held"] for k in res}
    assert all(v is good for v in held.values()), held


def test_the_tier88_evaluator_reads_what_its_prose_says(tmp_path):
    """Where a clause could be read two ways, the evaluator takes the prose's.
    Q1 is EVERY step, not a mean; Q14's rigid state is bitwise, so one ulp of
    altitude fails it while every other reading agrees to 1e-12."""
    W = _driver()
    d, pre = str(tmp_path / "out"), str(tmp_path / "pre")
    os.makedirs(d)
    _write_records_t88(d, pre, True)
    with open(os.path.join(d, "audit.json"), encoding="utf-8") as fh:
        au = json.load(fh)
    au["steps"][3]["injector_mdot_over_declared"] = 1.0 + 5e-6       # one step off, the mean is not
    with open(os.path.join(d, "audit.json"), "w", encoding="utf-8") as fh:
        json.dump(au, fh)
    res = W.evaluate_t88(d, pre_dir=pre)
    assert res["Q1"]["held"] is False
    assert res["Q14"]["held"] is True
    with open(os.path.join(d, "control_old.json"), encoding="utf-8") as fh:
        co = json.load(fh)
    co["steps"][0]["rigid"][1] = float(np.nextafter(co["steps"][0]["rigid"][1], 1e9))
    with open(os.path.join(d, "control_old.json"), "w", encoding="utf-8") as fh:
        json.dump(co, fh)
    res = W.evaluate_t88(d, pre_dir=pre)
    assert res["Q14"]["held"] is False and res["Q14"]["got"]["worst_rel"] < 1e-12


def test_with_nothing_measured_nothing_is_reported_held(tmp_path):
    W = _driver()
    assert W.evaluate(str(tmp_path), record=os.path.join(str(tmp_path), "none.npz")) == {}
    assert W.evaluate_t88(str(tmp_path), pre_dir=str(tmp_path)) == {}


def test_lateral_round_off_is_not_a_state_deviation(tmp_path):
    """A4's first evaluator divided each component by its own recorded value and
    read O(1) on 1e-20-level lateral round-off. The re-implemented one measures
    each quantity against its own magnitude and still catches a real miss."""
    W = _driver()
    rec = np.array([_rigid(round(0.005 * k, 6)) for k in range(0, 3)])
    np.savez(os.path.join(str(tmp_path), "episode.npz"), rigid=rec)
    same = [dict(rigid=list(rec[1] * [2.0, 1, 1, -1.0, 1, 1, 1])), dict(rigid=list(rec[2]))]
    got = W.rigid_deviation(same, os.path.join(str(tmp_path), "episode.npz"))
    assert max(r["dev"] for r in got) < 1e-15
    assert all(r["flight_bitwise"] for r in got)
    miss = [dict(rigid=list(rec[1] * [1, 1 + 1e-7, 1, 1, 1, 1, 1]))]
    got = W.rigid_deviation(miss, os.path.join(str(tmp_path), "episode.npz"))
    assert got[0]["dev"] > 1e-9 and not got[0]["flight_bitwise"]


# ===========================================================================
# 3. the records (out/w336_pre_w337: Tier 87's, from the rented box)
# ===========================================================================


def _rec(name):
    path = os.path.join(PRE, name)
    if not os.path.exists(path):
        pytest.skip("out/w336_pre_w337/%s not present" % name)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def test_the_gates_hold_and_the_wall_gate_can_fail():
    """A1-A4 on the audited run, and the old-wall control failing A2's gate by
    ten orders of magnitude -- so a clean wall reading is a measurement."""
    W = _driver()
    _rec("audit.json")
    res = W.evaluate(PRE, record=PRE_RECORD)
    for k in ("A1", "A2", "A3", "A4"):
        assert res[k]["held"] is True, (k, res[k])
    assert res["A4"]["got"]["flight_bitwise_steps"] == 29
    cw = _rec("control_wall.json")["steps"][0]["engine_wall_mass_rel"]
    assert cw > 1e-2 > 1e-12


def test_W337_the_injector_delivers_more_than_it_declares():
    late = _rec("audit.json")["steps"][-10:]
    inj = np.mean([s["injector_mdot_over_declared"] for s in late])
    assert 1.17 < inj < 1.19
    # a property of the condition, not of the grid
    g4 = _rec("grid_engine_c4.json")["steps"][-1]["injector_mdot_over_declared"]
    g2 = _rec("grid_engine_c2.json")["steps"][-1]["injector_mdot_over_declared"]
    assert abs(g2 / g4 - 1.0) < 2e-3
    # and the vehicle's mass account inherits it
    vm = np.mean([s["vehicle_mass_loss_over_exit"] for s in late])
    assert 0.85 < vm < 0.89


def test_W339_the_thermal_seam_conserves_conduction_and_loses_the_wall_work():
    d = _rec("thermal_seam_diag.json")
    for side in ("jmin", "jmax"):
        s = d["sides"][side]
        assert abs(s["shell_engine"] / s["gas_conduction"] - 1.0) < 5e-3
        assert 0.04 < s["gas_work"] / s["gas_total_visc"] < 0.055
        assert abs(s["gas_inviscid"]) < 1e-6 * s["gas_total_visc"]


def test_W340_the_throat_floor_does_not_move_with_the_coupling_step():
    """A lag shrinks with the step and a floor does not. e|f shrinks at second
    order; b|e sits at 3% whatever the step. Tier 87 read the floor as the point
    interpolation between 20 and 24 cells; Tier 88 measured it with no lag at
    all and found the ghost's depth instead (test_the_throat_floor_was_the_
    ghosts_depth_not_the_interpolation)."""
    W = _driver()
    t = W.tables(PRE, record=PRE_RECORD)["coupling_step"]
    be, ef = t["seam_be"], t["seam_ef"]
    assert be["dt5"] > 0.02 and be["dt2p5"] > 0.02
    assert abs(be["dt2p5"] / be["dt5"] - 1.0) < 0.1
    assert ef["dt10"] > ef["dt5"] > ef["dt2p5"] and ef["dt2p5"] < 2e-4
    # and it halves with the grid, as a first-order error should
    b4 = _rec("grid_engine_c4.json")["steps"][-1]["seam_be_mass_rel"]
    b2 = _rec("grid_engine_c2.json")["steps"][-1]["seam_be_mass_rel"]
    assert abs(b2) < 0.7 * abs(b4)


def test_W341_the_plume_seams_break_by_the_gas_constant():
    W = _driver()
    t = W.tables(PRE, record=PRE_RECORD)["coupling_step"]
    for k in ("seam_df", "seam_gf"):
        assert all(0.15 < t[k][d] < 0.5 for d in ("dt10", "dt5", "dt2p5")), (k, t[k])


def test_W342_every_wall_flux_is_a_property_of_the_grid():
    h4 = _rec("grid_engine_c4.json")["steps"][-1]["engine_heat_lost"]["jmax"]
    h2 = _rec("grid_engine_c2.json")["steps"][-1]["engine_heat_lost"]["jmax"]
    assert 1.8 < h2 / h4 < 2.2
    x = {c: _rec("grid_external_c%d.json" % c)["steps"][-1] for c in (4, 2, 1)}
    # the airframe's heating runs the wrong way at the episode's grid
    assert x[4]["airframe_heat_lost"]["0"] < 0.0 < x[2]["airframe_heat_lost"]["0"] < x[1]["airframe_heat_lost"]["0"]
    # skin friction grows with every refinement; pressure drag converges slowly
    assert x[4]["drag_shear"] < x[2]["drag_shear"] < x[1]["drag_shear"]
    assert x[4]["drag_pressure"] > x[2]["drag_pressure"] > x[1]["drag_pressure"]


def test_the_nozzle_is_right_given_its_inflow():
    W = _driver()
    v = W.tables(PRE, record=PRE_RECORD)["validation"]
    assert 0.98 < v["measured_over_planar"] < 1.0
    assert 1.12 < v["chamber_p_over_declared"] < 1.14


def test_the_throat_floor_was_the_ghosts_depth_not_the_interpolation():
    """Tier 88. The pre-fix run's engine at three instants, each side's face
    flux evaluated at that same instant, so no lag: point interpolation (adc470b)
    and the briefed overlap average both lose 3% of the mass flow at the throat;
    the same overlap average two ghost layers deep loses under 1e-4."""
    path = os.path.join(ROOT, "out", "w340_throat_diag.json")
    if not os.path.exists(path):
        pytest.skip("out/w340_throat_diag.json not present; run scripts/w340_throat_diag.py")
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)["rows"]
    assert [r["frame"] for r in rows] == [10, 20, 29]
    for r in rows:
        assert -0.032 < r["remap"]["be"] < -0.029
        assert -0.032 < r["overlap"]["be"] < -0.029
        assert abs(r["layers"]["be"]) < 1e-4
        assert abs(r["layers"]["ab"]) <= abs(r["remap"]["ab"])
