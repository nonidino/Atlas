"""Tier 49 -- rung 9: the union MARCHED, and its gate measured (CS-18).

Tier 46 built the joined union and compiled it; Tier 47 restated rung 9's gate
and measured each join's receiving balance at the RELEASE STATE, one step each,
and listed seven decisions a march needs first.  This makes the seven and
marches.

Stages, each persisted to ``out/tier49/tier49.json`` as it finishes:

  ``decisions``  the seven, each with the control that makes it evidence: the
                 clock table in seconds against the bare numbers `L7/R9`
                 compares (W201); the rotor sized for its host and the machine
                 sized for that rotor, beside Tier 47's failure and the
                 width-1 control that must reproduce Tier 46 (W199); the
                 devices' body force and its discrete-thrust identity (W94);
                 the multirate seams (W194); and the named holes (W200, W197).
  ``calib``      **the instrument, run before the gate was fixed**: a 40-step
                 march, its repeat, the settling report, and the cost.
  ``gate``       the seven arms at the 1200-step horizon, judged against
                 `vehicle_march.GATE`, which was written after `calib` and
                 before this ran.
  ``slow``       the coolant circuit and the block on THEIR clock, where J2's
                 receiver lives -- with the mount term and without it -- and
                 the block's own thermal time constant, which is what makes
                 the union's clocks unspannable by one march.
  ``r9``         W194: `L7/R9` scoped to the multirate seams, with the disjoint
                 union as the control that must stop refusing and
                 `thermal_seam` as the one that must still refuse.

The fluid is released from ``out/w141/settled.npz`` -- the same state Tier 47
measured its release-state balances at -- and marched.  Nothing is downloaded,
no checkpoint is loaded and no machine is rented.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import io                                                               # noqa: E402
import json                                                             # noqa: E402
import sys                                                              # noqa: E402
import time                                                             # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np                                                      # noqa: E402
import torch                                                            # noqa: E402

torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert torch.backends.cuda.matmul.allow_tf32 is False
assert torch.backends.cudnn.allow_tf32 is False

from atlas.cases import cooling_loop as CL                              # noqa: E402
from atlas.cases import integration_union as IU                         # noqa: E402
from atlas.cases import powertrain as PT                                # noqa: E402
from atlas.cases import vehicle_march as VM                             # noqa: E402
from atlas.cases import wake_array as WA                                # noqa: E402
from atlas.cases import wing_fsi as W                                   # noqa: E402

OUT = os.path.join(HERE, "out", "tier49")
CACHE = os.path.join(OUT, "cache")

#: **The horizon, fixed with the gate.**  1200 fluid macro-steps is 15 of the
#: tiling's time units -- 4.6 transits of the 3.25-unit domain -- and, at the
#: declared vehicle scale, 0.15 s and exactly 3 coolant steps.  The settle
#: window is the last quarter of it.
HORIZON = 1200
SETTLE_FRAC = 0.25
#: the instrument run, which fixed the gate's numbers
CALIB_STEPS = 40


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def _retry(fn, attempts=40, pause=0.25):
    """OneDrive holds a just-written file, so `os.replace` can raise WinError 5."""
    for k in range(attempts):
        try:
            return fn()
        except PermissionError:
            if k == attempts - 1:
                raise
            time.sleep(pause)


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return (x.tolist() if x.size <= 64
                else {"shape": list(x.shape), "first": float(x.flat[0]),
                      "last": float(x.flat[-1]),
                      "norm": float(np.linalg.norm(x))})
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if x is None or isinstance(x, (str, int, float)):
        return x
    return str(x)


def persist(res):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "tier49.json")
    tmp = path + ".tmp"

    def write():
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(clean(res), fh, indent=1)

    _retry(write)
    _retry(lambda: os.replace(tmp, path))
    return path


def load_res():
    path = os.path.join(OUT, "tier49.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def settled(name="w141"):
    d = np.load(os.path.join(HERE, "out", name, "settled.npz"))
    return d["u"], d["v"]


# ---------------------------------------------------------------------------
# the arms, cached -- a tight march is ~4 minutes and there are seven
# ---------------------------------------------------------------------------


def _cache_path(key: str) -> str:
    return os.path.join(CACHE, f"{key}.npz")


def _save_march(key: str, m: VM.UnionMarch) -> None:
    os.makedirs(CACHE, exist_ok=True)
    payload = {f"trace__{k}": np.asarray(v) for k, v in m.trace.items()}
    payload["u"] = m.u
    payload["v"] = m.v
    payload["block_T"] = m.block_T
    payload["join_residual"] = np.asarray(m.join_residual)
    payload["meta"] = np.frombuffer(json.dumps({
        "steps": m.steps, "join_coupling": m.join_coupling,
        "joins": list(m.joins), "coolant": clean(m.coolant),
        "t_in": m.t_in, "wall_s": m.wall_s,
        "thickness": clean(m.thickness), "notes": clean(m.notes),
    }).encode("utf-8"), dtype=np.uint8)
    # np.savez appends ".npz" unless the name already ends with it
    tmp = os.path.join(CACHE, f".{key}.tmp.npz")
    _retry(lambda: np.savez_compressed(tmp, **payload))
    _retry(lambda: os.replace(tmp, _cache_path(key)))


def _load_march(key: str) -> VM.UnionMarch | None:
    p = _cache_path(key)
    if not os.path.isfile(p):
        return None
    d = np.load(p, allow_pickle=False)
    meta = json.loads(bytes(d["meta"]).decode("utf-8"))
    return VM.UnionMarch(
        steps=meta["steps"], join_coupling=meta["join_coupling"],
        joins=tuple(meta["joins"]),
        trace={k[len("trace__"):]: d[k] for k in d.files
               if k.startswith("trace__")},
        coolant=meta["coolant"], u=d["u"], v=d["v"], block_T=d["block_T"],
        t_in=meta["t_in"], wall_s=meta["wall_s"], thickness=meta["thickness"],
        join_residual=d["join_residual"], notes=meta["notes"])


#: (key, joins, per-join coupling, null) -- the arm list, fixed with the gate.
ARMS: tuple[tuple[str, dict, str | None], ...] = (
    ("referent",   {"J1": "tight",  "J2": "tight",  "J3": "tight"},  None),
    ("repeat",     {"J1": "tight",  "J2": "tight",  "J3": "tight"},  None),
    ("all_lagged", {"J1": "lagged", "J2": "lagged", "J3": "lagged"}, None),
    ("J1_lagged",  {"J1": "lagged", "J2": "tight",  "J3": "tight"},  None),
    ("J3_lagged",  {"J1": "tight",  "J2": "tight",  "J3": "lagged"}, None),
    ("null_J3",    {"J1": "tight",  "J2": "tight",  "J3": "tight"},  "J3"),
    ("null_J1",    {"J1": "tight",  "J2": "tight",  "J3": "tight"},  "J1"),
)


def arm(key: str, coupling: dict, null: str | None, steps: int,
        u, v, force: bool = False) -> VM.UnionMarch:
    cached = None if force else _load_march(key)
    if cached is not None and cached.steps == steps:
        print(f"  [{key}] cached, {steps} steps", flush=True)
        return cached
    t0 = time.perf_counter()
    last = [t0]

    def prog(s, el):
        if s % 200 == 199 or s == 0:
            print(f"    [{key}] step {s + 1}/{steps}  {el:.0f} s", flush=True)
            last[0] = el

    m = VM.march_union(u, v, steps=steps, join_coupling=coupling, null=null,
                       progress=prog)
    print(f"  [{key}] {steps} steps in {time.perf_counter() - t0:.1f} s",
          flush=True)
    _save_march(key, m)
    return m


# ===========================================================================
# stage 1 -- the seven decisions
# ===========================================================================


def stage_decisions(u, v) -> dict:
    out: dict = {}

    # -- 1 -- the units (W201) -----------------------------------------------
    out["W201_units"] = {
        "clock_table": VM.clock_table(),
        "alternatives": VM.VEHICLE.alternatives(),
        "what_Tier_46_reconciled": (
            "ten agents' dt_native re-declared at the tiling's bare 0.0125"),
        "the_bare_numbers_understate_the_spread_by":
            VM.clock_table()["spread_in_seconds"]
            / VM.clock_table()["spread_in_bare_numbers"],
    }

    # -- 2 -- the rotor and the machine (W199) -------------------------------
    u_ring = IU.ring_velocity(u, IU.ROTOR_SITE)
    u_mean = float(np.mean(u_ring))
    rows = {}
    for label, width, scale in (
            ("control: width 1, the machine powertrain declares", 1.0, 1.0),
            ("Tier 47's failure: width 0.5, that same machine",
             VM.HOST_ROTOR_WIDTH, 1.0),
            ("the choice: width 0.5, the machine scaled by the width",
             VM.HOST_ROTOR_WIDTH, VM.ROTOR_SCALE)):
        res, els, disk = VM.operating_point(u_ring, width=width, scale=scale)
        mod = WA.RotorDisk(agent_id="_P", u_ref=u_ring)._mod
        best = max(float(mod.ActuatorDisk(a=a, area=width)(u_mean).torque)
                   for a in np.linspace(PT.A_MIN, PT.A_MAX, 800))
        r_tot = sum(e.resistance for e in els.values())
        p_ref = IU.calibrated_p_ref(PT.MachineAgent())[0]
        rows[label] = {
            "width": width, "machine_scale": scale,
            "omega": res.omega, "induction": res.induction,
            "current": res.current, "torque_rotor": res.torque_rotor,
            "torque_machine": res.torque_machine,
            "largest_torque_in_the_clamp": best,
            "demand_over_largest_supply": res.torque_machine / max(best, 1e-300),
            "rotor_valid": bool(res.rotor_valid),
            "mgu_valid": bool(res.mgu_valid),
            "thrust": disk.thrust, "shaft_power": disk.power,
            "torque_residual": res.torque_residual,
            "balance_residual": res.residual,
            "k_t": els["MGU"].k_t, "r_total": r_tot,
            "r_mgu": els["MGU"].resistance,
            "I2R_nondim": res.current ** 2 * r_tot,
            "q_machine_W_at_the_held_p_ref":
                res.current ** 2 * els["MGU"].resistance * p_ref,
        }
    ctrl = rows["control: width 1, the machine powertrain declares"]
    sized = rows["the choice: width 0.5, the machine scaled by the width"]
    out["W199_sizing"] = {
        "rotor_face_in_the_host": VM.HOST_ROTOR_WIDTH,
        "rotor_diameter_m": VM.VEHICLE.rotor_diameter_m,
        "rows": rows,
        "the_similarity_invariant": "k_e = k_t scale with the width and every "
                                    "resistance inversely, so the back-EMF and "
                                    "the demand/supply ratio are unchanged",
        "control_reproduces_Tier_46_omega": abs(ctrl["omega"] - 13.837) < 1e-3,
        "same_induction_as_the_control": abs(
            sized["induction"] - ctrl["induction"]) <= 1e-12,
        "same_demand_over_supply": abs(
            sized["demand_over_largest_supply"]
            - ctrl["demand_over_largest_supply"]) <= 1e-12,
        "shaft_power_ratio_sized_over_control":
            sized["shaft_power"] / ctrl["shaft_power"],
        "q_machine_ratio_sized_over_control":
            sized["q_machine_W_at_the_held_p_ref"]
            / ctrl["q_machine_W_at_the_held_p_ref"],
    }

    # -- 6 -- the shared operating point, and how stiff it is (W196) ---------
    #: **W196 with a magnitude.**  `joining-seam-cost` section 5.4 measured that a
    #: join's re-derived state reaches agents off its own seams and that no rule
    #: sees it.  The march says how hard it reaches: the machine sits just above
    #: the battery's open-circuit voltage, so ``I = (k_e lambda u / r - V_oc)/R``
    #: is a small difference of two large numbers, and the heat is its square.
    #: The elasticities are exact and are computed rather than differenced.
    sens = {}
    for label, width, scale in (
            ("width 1, the machine powertrain declares", 1.0, 1.0),
            ("width 0.5, the machine scaled by the width",
             VM.HOST_ROTOR_WIDTH, VM.ROTOR_SCALE)):
        els = VM.machine_for_rotor(scale)
        mgu = els["MGU"]
        r_tot = sum(x.resistance for x in els.values())
        lam = WA.RotorDisk(agent_id="_P", u_ref=u_ring)._mod.TIP_SPEED_RATIO
        gain = mgu.k_e * lam / (0.5 * width)          # d(back-EMF)/du
        emf = gain * u_mean
        cur = (emf - PT.V_OC) / r_tot
        sens[label] = {
            "d_backEMF_du": gain, "backEMF_at_the_settled_inflow": emf,
            "V_oc": PT.V_OC, "current": cur,
            "elasticity_of_the_current_in_the_inflow": emf / (emf - PT.V_OC),
            "elasticity_of_the_machines_heat_in_the_inflow":
                2.0 * emf / (emf - PT.V_OC),
        }
    a_ = sens["width 1, the machine powertrain declares"]
    b_ = sens["width 0.5, the machine scaled by the width"]
    out["W196_operating_point_stiffness"] = {
        "rows": sens,
        "the_similarity_preserves_the_elasticity_exactly": bool(abs(
            a_["elasticity_of_the_current_in_the_inflow"]
            - b_["elasticity_of_the_current_in_the_inflow"]) < 1e-9),
        "so": "re-sizing restored the operating point's EXISTENCE and left its "
              "CONDITIONING exactly where it was -- the back-EMF is the "
              "similarity's invariant, and the stiffness is a property of the "
              "back-EMF against V_oc",
    }

    # -- 5 -- the devices as body forces (W94) -------------------------------
    f = VM.DeviceForcing()
    checks = {}
    for dev in VM.DEVICES:
        thrust = 0.1093 if dev.join == "J3" else 0.30
        fld = f.field(thrust, dev, 0.92)              # asserts the identity
        checks[dev.site.device] = {
            "join": dev.join, "x_plane": dev.x_plane, "y0": dev.y0,
            "width": dev.width, "rows": [int(dev.rows[0]), int(dev.rows[-1])],
            "ring_read_at_column": dev.i_up,
            "downstream_ring_column": dev.i_down,
            "cells_between_the_ring_and_the_plane":
                dev.x_plane / W.DX - (dev.i_up + 0.5),
            "sum_f_dA": float(np.sum(fld) * W.DX * W.DX),
            "thrust": -thrust,
            "exact_to": abs(float(np.sum(fld) * W.DX * W.DX) + thrust),
        }
    f.last = {"ROTOR": {"u_disk": 0.92}, "RAD": {"u_disk": 1.0}}
    out["W94_body_force"] = {
        "devices": checks, "thickness": f.thickness_report(),
        "the_rule": "Delta_d = <U_d> dt, the distance a parcel travels while "
                    "the impulse is applied -- w100_scaling_ladder._forcing's, "
                    "unchanged -- with a declared floor of one cell, because on "
                    "this host the rule alone gives 0.18 of a cell",
    }

    # -- 7 -- the multirate seams (W194) -------------------------------------
    gj, _ = IU.build(u, v)
    gd, _ = IU.build(u, v, joins=())
    out["W194_multirate_seams"] = {
        "joined": VM.multirate_seams(gj),
        "disjoint": VM.multirate_seams(gd),
    }

    # -- 3 and 4 -- the named holes ------------------------------------------
    out["W200_named_hole"] = {
        "a_dissipation_field_was_added": False,
        "domain_boundaries_became_ports": False,
        "instead": "each join's receiver balance is written by hand, which is "
                   "what CS-9, CS-12 and the front-wing demo each did",
    }
    out["W197_named"] = {
        "the_orientation_was_decided": False,
        "instead": "a march applies the device as a BODY FORCE, so the power "
                   "path is the force's work on the fluid and does not go "
                   "through the two declared ports at all",
    }
    return out


# ===========================================================================
# stage 2 -- the instrument, before the gate was fixed
# ===========================================================================


def stage_calib(u, v) -> dict:
    t0 = time.perf_counter()
    a = arm("calib_a", {"J1": "tight", "J2": "tight", "J3": "tight"}, None,
            CALIB_STEPS, u, v)
    b = arm("calib_b", {"J1": "tight", "J2": "tight", "J3": "tight"}, None,
            CALIB_STEPS, u, v)
    floor = {}
    for k in VM.CROSSING_KEYS:
        if k == "t_wall":
            continue
        x, y = np.asarray(a.trace[k]), np.asarray(b.trace[k])
        floor[k] = {"max_abs_diff": float(np.max(np.abs(x - y))),
                    "bitwise_identical": bool(np.array_equal(x, y))}
    bal = VM.receiver_balances(a, SETTLE_FRAC)
    jr = np.asarray(a.join_residual)
    n = max(1, int(round(SETTLE_FRAC * a.steps)))
    band = {}
    for k in ("load", "u_rotor", "u_core", "induction"):
        seg = np.asarray(a.trace[k])[-n:]
        band[k] = {"mean": float(seg.mean()),
                   "peak_to_peak_relative": float(
                       (seg.max() - seg.min()) / max(abs(seg.mean()), 1e-300))}
    return {
        "steps": CALIB_STEPS,
        "wall_s": time.perf_counter() - t0,
        "s_per_macro_step_tight": a.wall_s / a.steps,
        "repeat_floor": floor,
        "repeat_floor_is_bitwise": all(r["bitwise_identical"]
                                       for r in floor.values()),
        "J3_balance": bal["J3"], "J1_balance": bal["J1"],
        "join_inner_residual_first": float(jr[:, 0].mean()) if jr.size else None,
        "join_inner_residual_last": float(jr[:, 1].mean()) if jr.size else None,
        "unsteadiness_band_over_the_settle_window": band,
        "note": "this stage fixed vehicle_march.GATE's numbers and ran before "
                "any 1200-step arm",
    }


# ===========================================================================
# stage 3 -- the arms, and the gate
# ===========================================================================


def stage_gate(u, v) -> dict:
    marches: dict[str, VM.UnionMarch] = {}
    for key, coupling, null in ARMS:
        marches[key] = arm(key, coupling, null, HORIZON, u, v)
        persist_partial(key, marches[key])

    ref = marches["referent"]
    out: dict = {"horizon": HORIZON, "settle_frac": SETTLE_FRAC,
                 "gate": VM.GATE, "prediction": VM.PREDICTION, "arms": {}}

    # -- the repeat floor (G6) ----------------------------------------------
    floor = {}
    for k in VM.CROSSING_KEYS:
        if k == "t_wall":
            continue
        x = np.asarray(ref.trace[k])
        y = np.asarray(marches["repeat"].trace[k])
        floor[k] = {"max_abs_diff": float(np.max(np.abs(x - y))),
                    "bitwise": bool(np.array_equal(x, y))}
    ref_vec = ref.crossing()
    rep_vec = marches["repeat"].crossing()
    floor_scalar = VM.composition_error(rep_vec, ref_vec)
    out["G6"] = {
        "per_quantity": floor, "bitwise_in_every_crossing_quantity":
            all(r["bitwise"] for r in floor.values()),
        "repeat_floor_in_the_crossing_norm": floor_scalar,
        "verdict": "pass" if all(r["bitwise"] for r in floor.values()) else "fail",
    }

    # -- per-arm summaries ---------------------------------------------------
    for key, m in marches.items():
        bal = VM.receiver_balances(m, SETTLE_FRAC)
        jr = np.asarray(m.join_residual)
        out["arms"][key] = {
            "join_coupling": m.join_coupling, "null": m.notes.get("null"),
            "wall_s": m.wall_s, "coolant_steps": len(m.coolant),
            "settled": m.settled(SETTLE_FRAC),
            "crossing_vector": dict(zip(VM.CROSSING_KEYS, m.crossing().tolist())),
            "J3_balance": bal["J3"], "J1_balance": bal["J1"],
            "join_inner_residual_last": (float(jr[:, 1].mean())
                                         if jr.size else None),
            "thickness": m.thickness,
        }

    # -- G1: J3's receiver balance over a march ------------------------------
    b = VM.receiver_balances(ref, SETTLE_FRAC)["J3"]
    null3 = VM.receiver_balances(marches["null_J3"], SETTLE_FRAC)["J3"]
    #: the mis-attribution control: charge the shaft against the OTHER device's
    #: work.  A balance that closes either way is not measuring the join.
    mis = (abs(b["the core's share of it"] - b["the shaft's claim, T <U_d>"])
           / max(abs(b["the shaft's claim, T <U_d>"]), 1e-300))
    g1 = VM.GATE["G1"]
    out["G1"] = {
        "with_the_term": b["residual_with_the_term"],
        "tol_with": g1["tol_with"],
        "null_J3_residual": (null3["residual_with_the_term"]
                             if null3["the shaft's claim, T <U_d>"] else 1.0),
        "mis_attributed_residual": mis,
        "tol_without": g1["tol_without"],
        "u_ring": b["u on the ring the disk reads"],
        "u_plane": b["u in the cell the sink sits in"],
        "u_plane_over_u_ring": b["u_plane / u_ring"],
        "the_residual_the_velocity_gap_alone_predicts":
            1.0 - b["u_plane / u_ring"],
        #: the gate asks for BOTH controls to fail, so both enter the verdict
        "verdict": _verdict(
            None if b["residual_with_the_term"] is None else (
                b["residual_with_the_term"] <= g1["tol_with"]
                and mis >= g1["tol_without"]
                and (null3["residual_with_the_term"] or 1.0) >= g1["tol_without"])),
    }

    # -- G3: J1's parametric check over a march ------------------------------
    j1 = VM.receiver_balances(ref, SETTLE_FRAC)["J1"]
    n1 = VM.receiver_balances(marches["null_J1"], SETTLE_FRAC)["J1"]
    ua_null = np.asarray(marches["null_J1"].trace["ua"])
    air_null = np.asarray(marches["null_J1"].trace["u_core"])
    g3 = VM.GATE["G3"]
    out["G3"] = {
        "tracking_residual": j1["tracking_residual"],
        "tol_with": g3["tol_with"],
        "UA_release_to_settled": [j1["UA at release"],
                                  j1["the join's term: UA follows the air"]],
        "air_release_to_settled": j1["air through the core, release -> settled"],
        "null_UA_range": [float(ua_null.min()), float(ua_null.max())],
        "null_UA_is_exactly_constant": bool(ua_null.max() == ua_null.min()),
        "null_air_moved_by": float(air_null.max() - air_null.min()),
        "the_air_s_mechanical_loss_unaccounted":
            j1["the air's mechanical loss across the core, unaccounted"],
        "verdict": _verdict(
            j1["tracking_residual"] is not None
            and j1["tracking_residual"] <= g3["tol_with"]
            and bool(ua_null.max() == ua_null.min())
            and float(air_null.max() - air_null.min()) > 0.0),
    }

    # -- G4 and G5: composition error ---------------------------------------
    vecs = {k: m.crossing() for k, m in marches.items()}
    e = {}
    for key in ("J1_lagged", "J3_lagged", "all_lagged"):
        e[key] = VM.composition_error(vecs[key], ref_vec)
    #: **J2's lag lives on the COOLANT clock**, so its arm is a replay of the
    #: referent's own fluid trace rather than a second fluid march -- legitimate
    #: because the coupling is one-way on that side, which
    #: `tests/test_tier49_union_march.py` pins bitwise.  Over the union march's
    #: own horizon the coolant circuit takes three steps, so J2's error is
    #: measured on `march_loop`'s horizon instead, where it has 1200.
    lp_ref = VM.march_loop(steps=1200, q_machine=float(np.mean(
        np.asarray(ref.trace["q_machine"])[-300:])), ua=float(np.mean(
            np.asarray(ref.trace["ua"])[-300:])), mounted=True, lag=1)
    lp_lag = VM.march_loop(steps=1200, q_machine=float(np.mean(
        np.asarray(ref.trace["q_machine"])[-300:])), ua=float(np.mean(
            np.asarray(ref.trace["ua"])[-300:])), mounted=True, lag=2)
    e["J2_lagged"] = _replay_error(ref, lp_ref, lp_lag)
    e["J2_lagged_on_the_union_horizon"] = _replay_error(
        ref, VM.replay_coolant(ref, q_lag=1), VM.replay_coolant(ref, q_lag=2))
    g4, g5 = VM.GATE["G4"], VM.GATE["G5"]
    resolved = {k: bool(v > g4["floor_multiple"] * max(floor_scalar, 1e-300))
                for k, v in e.items() if k != "all_lagged"}
    #: **G5 as pre-registered sums three joins, and the three-join form has no
    #: subject on this union**: J1's and J3's errors are measured on the fluid
    #: clock over 1200 macro-steps and J2's on the coolant clock over 1200
    #: coolant steps, which is a horizon 400 times longer in seconds.  Adding a
    #: number measured over 0.15 s to one measured over 60 s is not an addition
    #: of comparable things.  **Both forms are reported** -- the pre-registered
    #: three-join one and the well-posed two-join one over the joins that DO
    #: share a clock -- and the clause is not silently substituted, which is
    #: W183's and W198's precedent on this ladder.
    def _judge(e_all, s_sum):
        if s_sum <= 0.0:
            return "not measured", 0.0
        gp = abs(e_all - s_sum)
        if e_all > g5["compound_factor"] * s_sum:
            return "COMPOUND", gp
        if gp <= g5["add_tol"] * s_sum:
            return "ADD", gp
        return "NEITHER", gp

    per_join_sum = sum(e[k] for k in ("J1_lagged", "J2_lagged", "J3_lagged"))
    two_join_sum = sum(e[k] for k in ("J1_lagged", "J3_lagged"))
    add_verdict, gap = _judge(e["all_lagged"], two_join_sum)
    three_verdict, three_gap = _judge(e["all_lagged"], per_join_sum)
    #: **The repeat floor is bitwise ZERO, so G4's clause as pre-registered is
    #: vacuous**: ten times zero is zero and every nonzero difference clears it.
    #: Reported rather than quietly passed, with the floor that does bind given
    #: beside it -- the fluid's own unsteadiness over the settle window.  A
    #: difference smaller than the band the flow moves through on its own is not
    #: a signal even when it is perfectly reproducible.
    band = {}
    n_set = max(1, int(round(SETTLE_FRAC * ref.steps)))
    for k in VM.CROSSING_KEYS:
        if k == "t_wall":
            continue
        seg = np.asarray(ref.trace[k])[-n_set:]
        m_ = float(seg.mean())
        band[k] = float((seg.max() - seg.min()) / max(abs(m_), 1e-300))
    band_norm = float(np.linalg.norm([band[k] for k in band]))
    resolved_band = {k: bool(v > band_norm) for k, v in e.items()
                     if k != "all_lagged"}
    out["G4"] = {
        "errors": e, "repeat_floor": floor_scalar,
        "floor_multiple_required": g4["floor_multiple"],
        "the_clause_as_written_is_vacuous_because_the_floor_is_zero":
            bool(floor_scalar == 0.0),
        "the_floor_that_binds_instead": {
            "what": "the fluid's own unsteadiness over the settle window, per "
                    "crossing quantity, relative peak-to-peak",
            "per_quantity": band, "norm": band_norm,
            "errors_above_the_unsteadiness_band": resolved_band,
        },
        "resolved_above_the_floor": resolved,
        "per_component": {
            k: dict(zip(VM.CROSSING_KEYS,
                        ((vecs[k] - ref_vec) / np.where(np.abs(ref_vec) > 0,
                                                        np.abs(ref_vec), 1.0)
                         ).tolist()))
            for k in ("J1_lagged", "J3_lagged", "all_lagged")},
        "verdict": _verdict(all(resolved.values())),
        "not_resolved": [k for k, ok in resolved.items() if not ok],
    }
    #: **G5 as pre-registered compares NORMS, and a norm cannot see a sign.**
    #: The per-component table shows J1's and J3's errors pointing OPPOSITE
    #: ways, so the triangle inequality is slack and the sum of norms overstates
    #: the whole even when the underlying errors superpose exactly.  The
    #: question the clause was written to ask -- do the joins' errors compound --
    #: is answered by the VECTOR sum, so it is measured here beside the
    #: pre-registered form rather than instead of it.
    denom = np.where(np.abs(ref_vec) > 0, np.abs(ref_vec), 1.0)
    rel = {k: (vecs[k] - ref_vec) / denom
           for k in ("J1_lagged", "J3_lagged", "all_lagged")}
    predicted = rel["J1_lagged"] + rel["J3_lagged"]
    resid = predicted - rel["all_lagged"]
    sup = {
        "what": "the per-join error VECTORS added, against the error vector of "
                "the arm with both joins lagged",
        "components": list(VM.CROSSING_KEYS),
        "e_J1": rel["J1_lagged"].tolist(),
        "e_J3": rel["J3_lagged"].tolist(),
        "e_J1_plus_e_J3": predicted.tolist(),
        "e_both_lagged": rel["all_lagged"].tolist(),
        "residual_norm": float(np.linalg.norm(resid)),
        "relative_residual": float(np.linalg.norm(resid)
                                   / max(np.linalg.norm(rel["all_lagged"]), 1e-300)),
        "the_two_joins_errors_point_opposite_ways": bool(
            float(rel["J1_lagged"] @ rel["J3_lagged"]) < 0.0),
        "cosine": float(rel["J1_lagged"] @ rel["J3_lagged"]
                        / max(np.linalg.norm(rel["J1_lagged"])
                              * np.linalg.norm(rel["J3_lagged"]), 1e-300)),
    }
    out["G5_superposition"] = sup
    out["G5"] = {
        "e_all_lagged": e["all_lagged"],
        "the_well_posed_form": {
            "joins": ["J1", "J3"], "share_a_clock": True,
            "sum_of_the_single_join_errors": two_join_sum,
            "gap": gap, "gap_over_sum": gap / max(two_join_sum, 1e-300),
            "verdict": add_verdict,
        },
        "as_pre_registered_over_three_joins": {
            "sum_of_the_single_join_errors": per_join_sum,
            "gap": three_gap,
            "gap_over_sum": three_gap / max(per_join_sum, 1e-300),
            "verdict": three_verdict,
            "has_no_subject_because":
                "J1 and J3's errors are measured over 1200 fluid macro-steps "
                "(0.15 s) and J2's over 1200 coolant steps (60 s), a horizon "
                "400x longer in seconds. The three joins do not share a clock, "
                "so their errors are not addends of one sum",
        },
        "add_tol": g5["add_tol"], "compound_factor": g5["compound_factor"],
        "verdict": add_verdict,
    }
    return out


def _replay_error(ref: VM.UnionMarch, a: VM.LoopMarch, b: VM.LoopMarch) -> float:
    """J2's composition error: the block's own crossing data, lagged vs not."""
    if not a.rows or not b.rows:
        return 0.0
    ka = np.array([a.rows[-1]["t_wall_mean"], a.rows[-1]["q_block"]])
    kb = np.array([b.rows[-1]["t_wall_mean"], b.rows[-1]["q_block"]])
    return VM.composition_error(kb, ka)


def _verdict(ok) -> str:
    if ok is None:
        return "not measured"
    return "pass" if bool(ok) else "fail"


def persist_partial(key, m):
    res = load_res()
    res.setdefault("progress", {})[key] = {
        "steps": m.steps, "wall_s": m.wall_s,
        "coolant_steps": len(m.coolant)}
    persist(res)


# ===========================================================================
# stage 4 -- the slow half, on its own clock
# ===========================================================================


def stage_slow() -> dict:
    ref = _load_march("referent")
    if ref is None:
        raise RuntimeError("stage 'slow' needs stage 'gate' to have run")
    q = float(np.mean(np.asarray(ref.trace["q_machine"])[-300:]))
    ua = float(np.mean(np.asarray(ref.trace["ua"])[-300:]))

    #: **the block's own thermal time constant**, measured rather than quoted:
    #: march the loop to a settled wall temperature and read the 1/e time.
    long = VM.march_loop(steps=4000, q_machine=q, ua=ua, mounted=True)
    tw = np.array([r["t_wall_mean"] for r in long.rows])
    final = float(tw[-1])
    start = float(tw[0])
    span = final - start
    tau_idx = int(np.argmax(np.abs(tw - start) >= abs(span) * (1.0 - 1.0 / np.e))
                  ) if span != 0.0 else 0
    tau_s = tau_idx * CL.MACRO_DT

    with_term = VM.march_loop(steps=1200, q_machine=q, ua=ua, mounted=True)
    without = VM.march_loop(steps=1200, q_machine=q, ua=ua, mounted=False)
    g2 = VM.GATE["G2"]

    def bal_of(lm, n=300):
        rows = lm.rows[-n:]
        return {"relative": float(np.mean([r["block_first_law"]["relative"]
                                           for r in rows])),
                "t_wall": float(rows[-1]["t_wall_mean"]),
                "q_block": float(rows[-1]["q_block"]),
                "loop_residual": float(rows[-1]["loop_residual"]),
                "t_return": float(rows[-1]["t_return"])}

    a, b = bal_of(with_term), bal_of(without)
    #: the NULL is the block with no mount term at all -- its dry face carries
    #: `cooling_loop`'s own declared gas temperature, so the first law still
    #: closes.  What must fail is the balance written WITH the mount term in it
    #: and the mount term removed, which is Tier 47's `_with_and_without`.
    blk = IU.MountedBlock(q_machine=q)
    tc = np.full(CL.N_SEAM, CL.T_COOLANT_0)
    for _ in range(1200):
        blk.step(tc)
    bb = blk.energy_balance(tc)
    scale = max(abs(bb["q_outer"]), abs(bb["q_wall"]), 1e-30)
    without_the_term = abs(bb["stored_rate"] - (0.0 - bb["q_wall"])) / scale

    return {
        "q_machine_W": q, "ua_W_per_K": ua,
        "dt_s": CL.MACRO_DT,
        "block_thermal_time_constant_s": tau_s,
        "block_wall_start_K": start, "block_wall_settled_K": final,
        "fluid_macro_steps_to_span_the_block_s_transient":
            tau_s / VM.VEHICLE.seconds(W.MACRO_DT, "front_wing"),
        "coolant_steps_to_span_it": tau_s / CL.MACRO_DT,
        "with_the_mount_term": a,
        "the_block_with_its_own_declared_source": b,
        "the_mount_balance_without_the_mount_term": without_the_term,
        "tol_with": g2["tol_with"], "tol_without": g2["tol_without"],
        "G2_verdict": _verdict(a["relative"] <= g2["tol_with"]
                               and without_the_term >= g2["tol_without"]),
        "replay": {
            "lag_1": [r["t_wall_mean"] for r in
                      VM.replay_coolant(ref, q_lag=1).rows],
            "lag_3": [r["t_wall_mean"] for r in
                      VM.replay_coolant(ref, q_lag=3).rows],
        },
        #: the SECOND vehicle scale -- the out-of-sample cell for the multirate
        #: clauses.  The fluid trajectory does not depend on the coolant clock,
        #: so this is a replay and not a second march.
        "second_vehicle_scale": _second_scale(ref),
    }


def _second_scale(ref: VM.UnionMarch) -> dict:
    alt = VM.VehicleScale(l0_m=1.0, u0_ms=30.0)
    n = int(round(VM.clock_ratio(alt)))
    lp = VM.replay_coolant(ref, n_per_coolant=n)
    base = VM.replay_coolant(ref, n_per_coolant=VM.N_FLUID_PER_COOLANT)
    return {
        "scale": {"l0_m": alt.l0_m, "u0_ms": alt.u0_ms,
                  "chord_m": alt.chord_m, "t0_s": alt.t0_s},
        "n_fluid_per_coolant": n,
        "coolant_steps_over_the_same_fluid_horizon": len(lp.rows),
        "at_the_declared_scale": len(base.rows),
        "t_wall_final_K": (lp.rows[-1]["t_wall_mean"] if lp.rows else None),
        "t_wall_final_at_the_declared_scale_K":
            (base.rows[-1]["t_wall_mean"] if base.rows else None),
        "q_block_final_W": (lp.rows[-1]["q_block"] if lp.rows else None),
    }


# ===========================================================================
# stage 5 -- W194: L7/R9 scoped to the multirate seams
# ===========================================================================


def stage_r9(u, v) -> dict:
    from atlas.compiler import compile_scheme
    from atlas.cases import thermal_seam as TS
    from atlas.graph import FluxMatching

    def verdict_of(g):
        r = compile_scheme(g)
        return {"verdict": r.verdict.value,
                "refusals": sorted({f"{d.layer}/{d.rule}"
                                    for d in r.decisions.refusals}),
                "multirate_by_agents": bool(g.is_multirate()),
                "seams": VM.multirate_seams(g)}

    gj, _ = IU.build(u, v)
    gd, _ = IU.build(u, v, joins=())
    gjq, _ = IU.build(u, v, flux_matching=FluxMatching.TIME_INTEGRATED)
    out = {
        "joined_native": verdict_of(gj),
        "disjoint_native": verdict_of(gd),
        "joined_time_integrated": verdict_of(gjq),
    }
    #: **The control that must NOT move.**  `thermal_seam`'s 500:1 mismatch is
    #: carried BY its seam, so R9's premise is still met and the narrower rule
    #: must still fire.  Its DEFAULT build declares time-integrated matching and
    #: supplies integrated responses, so it admits for the reason Tier 17 built
    #: -- not because of anything here -- and the pointwise form beside it is the
    #: cell where the rule has to refuse.
    try:
        gt, _ = TS.build()
        out["control_thermal_seam_default"] = verdict_of(gt)
        gtp, _ = TS.build(flux_matching=FluxMatching.POINTWISE)
        out["control_thermal_seam_pointwise"] = verdict_of(gtp)
    except Exception as exc:                              # pragma: no cover
        out["control_thermal_seam_default"] = {"error": str(exc)}
    #: and the other direction: a single-clock graph must be untouched
    try:
        gr, _ = IU.build(u, v, clocks="reconciled")
        out["control_joined_reconciled_clocks"] = verdict_of(gr)
    except Exception as exc:                              # pragma: no cover
        out["control_joined_reconciled_clocks"] = {"error": str(exc)}
    return out


# ===========================================================================


STAGES = {"decisions": None, "calib": None, "gate": None, "slow": None,
          "r9": None}


def main(argv) -> None:
    want = [a for a in argv[1:] if not a.startswith("-")] or list(STAGES)
    bad = [s for s in want if s not in STAGES]
    if bad:
        raise SystemExit(f"unknown stage(s) {bad}; choose from {list(STAGES)}")
    t0 = time.perf_counter()
    res = load_res()
    res.setdefault("what", "rung 9's gate, over a march of the joined union")
    res["date"] = time.strftime("%Y-%m-%d")
    res["release_state"] = "out/w141/settled.npz"
    u, v = settled()

    if "decisions" in want:
        print("1. the seven decisions", flush=True)
        res["decisions"] = stage_decisions(u, v)
        ct = res["decisions"]["W201_units"]["clock_table"]
        print("   clocks, bare:", ct["bare_numbers"], " in seconds:",
              ct["seconds"], " spread", ct["spread_in_bare_numbers"], "->",
              ct["spread_in_seconds"], flush=True)
        sz = res["decisions"]["W199_sizing"]
        print("   sizing: same induction as the control:",
              sz["same_induction_as_the_control"],
              " same demand/supply:", sz["same_demand_over_supply"], flush=True)
        persist(res)

    if "calib" in want:
        print("2. the instrument (before the gate was fixed)", flush=True)
        res["calib"] = stage_calib(u, v)
        c = res["calib"]
        print("   %.3f s/macro-step tight; repeat bitwise %s; J3 residual %.4g"
              % (c["s_per_macro_step_tight"], c["repeat_floor_is_bitwise"],
                 c["J3_balance"]["residual_with_the_term"]), flush=True)
        persist(res)

    if "gate" in want:
        print("3. the arms, at %d macro-steps" % HORIZON, flush=True)
        res["gate_stage"] = stage_gate(u, v)
        g = res["gate_stage"]
        for k in ("G1", "G3", "G4", "G5", "G6"):
            print("   %s: %s" % (k, g[k].get("verdict")), flush=True)
        sp = g["G5_superposition"]
        print("   G5 superposition: the two joins' error vectors add to %.3g of "
              "the pair's own norm (cosine %.4f)"
              % (sp["relative_residual"], sp["cosine"]), flush=True)
        persist(res)

    if "slow" in want:
        print("4. the slow half, on its own clock", flush=True)
        res["slow"] = stage_slow()
        s = res["slow"]
        print("   block tau %.1f s = %.0f coolant steps = %.3g fluid steps; "
              "G2 %s" % (s["block_thermal_time_constant_s"],
                         s["coolant_steps_to_span_it"],
                         s["fluid_macro_steps_to_span_the_block_s_transient"],
                         s["G2_verdict"]), flush=True)
        persist(res)

    if "r9" in want:
        print("5. W194 -- R9 scoped to the multirate seams", flush=True)
        res["r9"] = stage_r9(u, v)
        for k, row in res["r9"].items():
            print("   %-26s %s  refusals %s  seams %d" % (
                k, row.get("verdict"), row.get("refusals"),
                row.get("seams", {}).get("n_seams", -1)), flush=True)
        persist(res)

    res["elapsed_seconds"] = time.perf_counter() - t0
    print("wrote", persist(res), "in %.1f s" % res["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    main(sys.argv)
