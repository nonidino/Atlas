"""Tier 88 -- the four code defects W336 found (W337-W340), fixed, re-marched,
re-audited, and the records they touch re-derived.

Every test here pins a fix on the new records (out/w336, out/w321, out/w3xx)
and keeps the diagnosis as its control, read from the records kept beside them
(out/w336_pre_w337, out/w321_pre_w337, out/pre_w337) or from a run of today's
code with the old behaviour put back (`w336_episode_residuals.revert`, checked
bitwise against build repo adc470b). tests/test_tier87_episode_residuals.py
pins the diagnoses themselves and the instrument.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import re

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "w336")
PRE = os.path.join(ROOT, "out", "w336_pre_w337")
TWO = os.path.join(ROOT, "out", "w336_t88_twolayer")
RECORD = os.path.join(ROOT, "out", "w321", "episode.npz")
PRE_RECORD = os.path.join(ROOT, "out", "w321_pre_w337", "episode.npz")


def _driver():
    path = os.path.join(ROOT, "scripts", "w336_episode_residuals.py")
    spec = importlib.util.spec_from_file_location("w336_driver_t88", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rec(name, where=OUT):
    path = os.path.join(where, name)
    if not os.path.exists(path):
        pytest.skip("%s not present" % os.path.relpath(path, ROOT))
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _win(run, f, lo=0.03, hi=0.06):
    return float(np.mean([f(s) for s in run["steps"] if lo < s["t"] <= hi + 1e-9]))


def _dt_study(where, key):
    """(10 ms, 5 ms, 2.5 ms) of a per-step reading's mean |.| over 30-60 ms."""
    au = _rec("audit.json", where)
    au = dict(au, steps=[s for s in au["steps"] if s["t"] <= 0.06 + 1e-9])
    f = key if callable(key) else (lambda s: abs(s[key]))
    return [_win(_rec("dt_10ms.json", where), f), _win(au, f), _win(_rec("dt_2p5ms.json", where), f)]


# ===========================================================================
# the gates, and the controls that give them teeth
# ===========================================================================


def test_the_gates_hold_on_the_fixed_run():
    W = _driver()
    _rec("audit.json")
    if not os.path.exists(RECORD):
        pytest.skip("out/w321/episode.npz not present")
    res = W.evaluate(OUT, record=RECORD)
    for k in ("A1", "A2", "A3", "A4"):
        assert res[k]["held"] is True, (k, res[k])
    assert res["A4"]["got"]["flight_bitwise_steps"] == 29
    # W332's old ghost still passes 5.2% through the walls on today's code
    assert _rec("control_wall.json")["steps"][0]["engine_wall_mass_rel"] > 1e-2


def test_all_four_reverted_reproduce_the_pre_fix_flight():
    """Q14's control. Today's code with W337-W340 put back reproduces the
    pre-fix audit's first steps: altitude, attitude, v_y, omega and mass
    bitwise, every reading to 1.1e-13. The lateral x and v_x, round-off around
    zero (1e-20, 1e-17), differ in their last bits between the two boxes
    (Zen 4 then, Zen 3 now) -- which is why Q14, registered as "rigid state
    bitwise", reads FAILED: A4's first mistake, repeated."""
    W = _driver()
    co, pre = _rec("control_old.json"), _rec("audit.json", PRE)
    assert co["reverted"] == list(W.FIXES)
    for c, p in zip(co["steps"], pre["steps"]):
        assert [c["rigid"][i] for i in (1, 2, 4, 5, 6)] == [p["rigid"][i] for i in (1, 2, 4, 5, 6)]
        assert max(abs(c["rigid"][i]) for i in (0, 3)) < 1e-15
        for key in ("injector_mdot_over_declared", "seam_be_mass_rel", "seam_ab_mass_rel"):
            assert c[key] == pytest.approx(p[key], rel=1e-12, abs=0.0)
        for sd in ("jmin", "jmax"):
            assert c["engine_heat_lost"][sd] == pytest.approx(p["engine_heat_lost"][sd], rel=1e-12)
    t88 = W.evaluate_t88(OUT, pre_dir=PRE)
    assert t88["Q14"]["held"] is False
    assert t88["Q14"]["got"]["rigid_bitwise"] is False and t88["Q14"]["got"]["worst_rel"] < 1e-12


# ===========================================================================
# W337 -- the injector
# ===========================================================================


def test_W337_the_injector_delivers_exactly_what_it_declares():
    """Every step of every run on the fixed code: the audit, the coupling-step
    study and the engine alone at every grid present. Tier 87 read 1.1772."""
    runs = [_rec("audit.json"), _rec("dt_10ms.json"), _rec("dt_2p5ms.json")]
    runs += [_rec("grid_engine_c%d.json" % c) for c in (4, 2, 1)
             if os.path.exists(os.path.join(OUT, "grid_engine_c%d.json" % c))]
    for r in runs:
        worst = max(abs(s["injector_mdot_over_declared"] - 1.0) for s in r["steps"])
        assert worst < 1e-6, (r["stage"], worst)
    # the diagnosis, from the pre-fix audit and from the fix reverted today
    late = _rec("audit.json", PRE)["steps"][-10:]
    assert 1.17 < np.mean([s["injector_mdot_over_declared"] for s in late]) < 1.19
    assert _rec("control_old.json")["steps"][0]["injector_mdot_over_declared"] > 1.17


# ===========================================================================
# W338 -- the skin's radiation
# ===========================================================================


def test_W338_the_skin_radiates_to_the_ambient():
    S = _rec("audit.json")["steps"]
    for s in S:
        for v in s["shell"].values():
            assert abs(v["q_rad_applied"] - v["q_rad_ambient"]) <= 1e-9 * abs(v["q_rad_ambient"])
    # the diagnosis: against the air beside it, 0.95 J/m per step per panel
    W = _driver()
    slip = W.tables(PRE, record=PRE_RECORD)["audit_late"]["rad_slip_abs"]
    assert 0.8 < abs(slip) < 1.1


# ===========================================================================
# W339 -- work at a stationary wall
# ===========================================================================


def test_W339_the_walls_do_no_work_and_the_seam_closes_to_the_lag():
    """The diagnostic reads the wall face's viscous energy flux as the
    conduction exactly, and the shell receives it to 0.13%. With W339 reverted
    on today's code the 30 J/m of work is back (the pre-fix run read 29.6)."""
    d = _rec("thermal_seam_diag.json")
    for s in d["sides"].values():
        assert s["gas_work"] == 0.0
        assert s["gas_total_visc"] == s["gas_conduction"]
        assert 1.0005 < s["shell_engine"] / s["gas_conduction"] < 1.0025
    r = _rec("thermal_seam_diag_revert_W339.json")
    assert r["reverted"] == ["W339"]
    for s in r["sides"].values():
        assert 0.04 < s["gas_work"] / s["gas_total_visc"] < 0.055
        assert 1.0005 < s["shell_engine"] / s["gas_conduction"] < 1.0025
    # the audit's own reading, late in the run: under 0.5% per side (Q4)
    S = _rec("audit.json")["steps"][-10:]
    for sd in ("jmin", "jmax"):
        assert np.mean([abs(s["thermal_engine"][sd]["rel"]) for s in S]) < 5e-3


def test_W339_what_remains_of_the_thermal_seam_is_much_smaller_than_the_work():
    """The coupling-step study: 4.7% at every step before, 3e-4 to 5e-4 now.
    It shrinks with the step, but by 1.24 and 1.44 -- under the 1.5 Q5
    registered, so Q5 reads FAILED."""
    new = _dt_study(OUT, lambda s: 0.5 * sum(abs(s["thermal_engine"][sd]["rel"])
                                             for sd in ("jmin", "jmax")))
    old = _dt_study(PRE, lambda s: abs(s["thermal_engine"]["jmax"]["rel"]))
    assert max(new) < 1e-3 and min(old) > 0.04
    assert new[0] > new[1] > new[2]


# ===========================================================================
# W340 -- the plane seams
# ===========================================================================


def test_W340_the_throat_is_a_lag_now():
    """b|e over 30-60 ms at 10, 5 and 2.5 ms: 7.3e-2, 2.9e-2, 3.0e-2 before
    (a floor); now it shrinks by more than 5x per halving. So does the
    throat's momentum error, which was 1.7% of the thrust."""
    be = _dt_study(OUT, "seam_be_mass_rel")
    assert be[0] / be[1] > 1.5 and be[1] / be[2] > 1.5 and be[2] < 3e-3
    mom = _dt_study(OUT, "seam_be_zmom_rel")
    assert mom[0] / mom[1] > 1.5 and mom[1] / mom[2] > 1.5
    old = _dt_study(PRE, "seam_be_mass_rel")
    assert old[1] > 0.02 and old[2] > 0.02


def test_W340_two_layers_on_a_one_way_seam_made_a_floor_of_a_lag():
    """The first fixed build handed EVERY plane seam two layers. e's exit is an
    outflow boundary, so e|f went from a lag (1.2e-3, 3.2e-4, 9.9e-5) to a 1.8%
    floor; one layer on the one-way seams put it back."""
    first = _dt_study(TWO, "seam_ef_mass_rel")
    assert min(first) > 0.015 and first[1] / first[2] < 1.2
    now = _dt_study(OUT, "seam_ef_mass_rel")
    assert max(now) < 2e-3 and now[0] > now[1] > now[2]


def test_the_carried_over_records_are_the_same_on_both_builds():
    """The two builds differ only in the one-way seams' depth, which neither
    the engine alone, nor `d` alone, nor the engine's thermal seam can see. The
    coarsen-4 runs of both are bitwise equal, so the first build's coarsen-2
    and coarsen-1 runs (and the re-derivation) stand for the second."""
    def strip(r):
        return [{k: v for k, v in s.items() if k != "wall_s"} for s in r["steps"]]
    for n in ("grid_engine_c4.json", "grid_external_c4.json"):
        a, b = _rec(n, TWO), _rec(n)
        assert a["build_repo"] != b["build_repo"]
        assert strip(a) == strip(b), n
    assert _rec("thermal_seam_diag.json", TWO)["sides"] == _rec("thermal_seam_diag.json")["sides"]


# ===========================================================================
# what the fixes did not touch, and a new row
# ===========================================================================


def test_W341_the_plume_seams_still_break_by_the_gas_constant():
    for key in ("seam_df_mass_rel", "seam_gf_mass_over_outflows"):
        q = _dt_study(OUT, key)
        assert all(0.15 < x < 0.5 for x in q), (key, q)


def test_W342_every_wall_flux_is_still_a_property_of_the_grid():
    h4 = _rec("grid_engine_c4.json")["steps"][-1]["engine_heat_lost"]["jmax"]
    h2 = _rec("grid_engine_c2.json")["steps"][-1]["engine_heat_lost"]["jmax"]
    assert 1.8 < h2 / h4 < 2.2
    x = {c: _rec("grid_external_c%d.json" % c)["steps"][-1] for c in (4, 2, 1)}
    assert x[4]["airframe_heat_lost"]["0"] < 0.0 < x[2]["airframe_heat_lost"]["0"] < x[1]["airframe_heat_lost"]["0"]
    assert x[4]["drag_shear"] < x[2]["drag_shear"] < x[1]["drag_shear"]


def test_W343_d_g_is_coupled_one_way_across_a_subsonic_band():
    """Found reading the fixed run. d|g loses 2% of its mass flow at every
    coupling step (4% before the fixes). Frozen at one instant the handover
    is exact -- g receives d's outflow to round-off -- and g's own inflow faces
    come up short beside the plume, where g's first cells are subsonic and d,
    behind an outflow boundary, cannot hear them."""
    path = os.path.join(ROOT, "out", "w340_throat_diag.json")
    if not os.path.exists(path):
        pytest.skip("out/w340_throat_diag.json not present")
    with open(path, encoding="utf-8") as fh:
        dg = json.load(fh)["dg"]
    for r in dg:
        assert r["ghost_mass"] == pytest.approx(r["d_to_g"], rel=1e-12)
        assert -0.05 < r["created"] < -0.03
        assert min(r["g_first_column_mach"]) < 1.0 < max(r["g_first_column_mach"])
    q = _dt_study(OUT, "seam_dg_mass_rel")
    assert q[1] > 0.015 and q[1] / q[2] < 1.2


# ===========================================================================
# the re-derived Tier 76-85 records (W334's procedure)
# ===========================================================================

#: Leaves allowed to move by 1e-3 or more between the fixed and reverted
#: re-derivations (same box, same tree), each with the reason it can.
ROUND_OFF_OR_COUNT = (
    (r"reach.*\.(nonzero|fraction|reach)$|rows\[\d+\]\.(nonzero|fraction|reach)$",
     "a count of cells over a threshold"),
    (r"linearity\.heat\.rel_error$", "a CG floor near 6e-9"),
    (r"bc_mech\.", "W318: the MECH operator's singular values are 1e-11..1e-13 (kappa 8.8e8)"),
    (r"seams\[5\]\.|d-g\.(beta|kappa)$|beta_movement\.d-g$", "d-g's beta is 9e-17"),
    (r"g-f\.|beta_movement\.g-f$", "g-f's converged beta is noise at 1e-12 (W334)"),
    (r"a-b\.(one_sided|blocks\.a\.(share|norm_F))$|seams\[0\]\.one_sided$",
     "a's share of a-b is 1.5e-4"),
    (r"saturation\.thermal_seam\.scan\[\d+\]\.max_dT$", "thermal_seam's saturation scan (W339, 0.17%)"),
    (r"^rigid\.V_orthonormality$", "an orthonormality residual of 1.8e-15"),
    (r"^reach\[7\]\.peak$", "d's g:ADVEC poke peaks at 2.7e-12"),
    (r"fill\.(serial_s|speedup)$|^workers$", "timing and pool size"),
)


def _moved(old_dir, new_dir, name, floor=1e-3):
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import w334_compare_records as C
    paths = [os.path.join(ROOT, d, name + ".json") for d in (old_dir, new_dir)]
    if not all(os.path.exists(p) for p in paths):
        pytest.skip("%s not present in both %s and %s" % (name, old_dir, new_dir))
    a, b = (json.load(open(p, encoding="utf-8")) for p in paths)
    fa, fb = dict(C.flatten(a)), dict(C.flatten(b))
    out = []
    for k in sorted(set(fa) & set(fb)):
        x, y = fa[k], fb[k]
        if C.is_noise(k) or isinstance(x, bool) or not isinstance(x, (int, float)) \
                or not isinstance(y, (int, float)):
            continue
        if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
            continue
        if C.rel(float(x), float(y)) >= floor:
            out.append(k)
    return out


RECORDS = ("w300", "w301", "w302", "w303", "w304", "w305", "w306", "w310", "w312",
           "w312_underresolved", "w312_converged", "w312_gf", "w314")


@pytest.mark.parametrize("name", RECORDS)
def test_the_fixes_move_the_tier76_85_records_only_where_named(name):
    """Fixed against reverted on the same tree and box: every leaf that moved by
    1e-3 or more is a threshold count, a round-off-level quantity, or a timing."""
    unexplained = [k for k in _moved("out/t88_revert", "out/t88", name)
                   if not any(re.search(p, k) for p, _why in ROUND_OFF_OR_COUNT)]
    assert unexplained == [], unexplained


@pytest.mark.parametrize("name", RECORDS)
def test_the_reverted_control_reproduces_the_w334_records(name):
    """The attribution control: today's tree with the four fixes reverted,
    against W334's records (now in out/pre_w337). Nothing moves by 1e-3 but
    round-off-level quantities and timings -- so nothing else changed since
    W334 that these records can see, and the difference above is the fixes'."""
    unexplained = [k for k in _moved("out/pre_w337", "out/t88_revert", name)
                   if not any(re.search(p, k) for p, _why in ROUND_OFF_OR_COUNT)]
    assert unexplained == [], unexplained


def test_the_swapped_records_are_the_fixed_re_derivation():
    import hashlib
    for n in RECORDS:
        a, b = (os.path.join(ROOT, "out", d, n + ".json") for d in ("", "t88"))
        a = os.path.join(ROOT, "out", n + ".json")
        if not (os.path.exists(a) and os.path.exists(b)):
            pytest.skip("records not present")
        with open(a, "rb") as fa, open(b, "rb") as fb:
            assert hashlib.sha256(fa.read()).digest() == hashlib.sha256(fb.read()).digest(), n
