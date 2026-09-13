"""Tier 53 -- CS-19's arms re-run on the repaired column.

CS-19 measured five arms on a graph whose powertrain was outside its declared
envelope for every macro-step of every one of them, and CS-20 repaired that but
did not re-run them.  So the two RaceLab pages described two different vehicles.
This closes that.

**What changes, and it is exactly three things.**

  1. the machine is sized for the inflow its host delivers
     (`racelab.machine_for_host`, CS-20 section 2.2), so `MGU.validity` admits;
  2. the arms release from the SETTLED field CS-20's spin-up produced, not from
     a uniform freestream whose transient leaves the disk's clamp for 58
     macro-steps;
  3. ``enforce=True``, so the envelope is consulted every macro-step and an
     arm that leaves it stops instead of reporting a number from outside.

**What does NOT change**: the horizon (600 macro-steps), the settle fraction,
the five arms, the gate's thresholds (`racelab.GATE`, every one of them
inherited from CS-18), and `vehicle_march.receiver_balances`, which reads this
march exactly as it read CS-19's.  Anything that moves, moves because of the
three things above.

**Two threads, and that is a measurement rather than a default.**  The field is
bitwise identical at 1, 2, 4 and 8 threads, but `power_on_the_fluid` and
`power_core` -- the two domain sums J3's receiving balance is BUILT from --
differ by one ulp above two threads, because a full-domain reduction's order
depends on the thread count.  Two threads is bitwise the one-thread answer in
the field AND in every trace, and is about 20% faster.  Stage ``threads``
measures it; W227 records it.

Stages, each persisted to ``out/racelab3/racelab3.json``:

  ``threads``  the bitwise thread-independence measurement above.
  ``arms``     the five arms at 600 macro-steps, with each join's receiver
               balance read by `vehicle_march.receiver_balances`.
  ``slow``     J2 on the coolant clock at the repaired machine's heat.
  ``compare``  every clause and every settled quantity, CS-19 beside this run,
               read out of ``out/racelab/racelab.json`` rather than retyped.
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

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert torch.backends.cuda.matmul.allow_tf32 is False
assert torch.backends.cudnn.allow_tf32 is False

from atlas.cases import cooling_loop as CL                              # noqa: E402
from atlas.cases import ground_effect as GE                             # noqa: E402
from atlas.cases import integration_union as IU                         # noqa: E402
from atlas.cases import racelab as RL                                   # noqa: E402
from atlas.cases import vehicle_march as VM                             # noqa: E402

OUT = os.path.join(HERE, "out", "racelab3")
CACHE = os.path.join(OUT, "cache")
CS19 = os.path.join(HERE, "out", "racelab", "racelab.json")
SETTLED = os.path.join(HERE, "out", "racelab2", "cache", "settled.npz")

#: CS-19's, unchanged, because a horizon that moves makes the comparison a
#: comparison of horizons.
HORIZON = 600
SETTLE_FRAC = 0.25

#: **CS-19's measured inflow at one instant**, which is what CS-20 sized the
#: machine for.  Kept as the CONTROL: sizing here admits the release state and
#: declines at macro-step 312.
U_RELEASE = 0.6691530373612168

#: **The inflow the machine is actually sized for, and it is a RANGE and not a
#: value.**  ``u_rotor`` falls 6.3% over this horizon -- 0.67208 to 0.63000 --
#: because the car's flow does not settle (CS-19 section 6.2: the fluid band is
#: still 5.8% after 1600 lagged steps), and the machine's margin over its own
#: crossover is only 3.3% of ``u``.  Sizing for the release value therefore
#: buys 311 macro-steps.  Sizing for the horizon's MINIMUM puts the induction at
#: its reference value there and above it everywhere else.
#:
#: Measured, not guessed: a 600-step lagged march at ``U_RELEASE`` reaches a
#: minimum ``u_rotor`` of 0.63000, and one at this value is admitted for all
#: 600 macro-steps with the current positive throughout.
U_MIN_HORIZON = 0.63
U_DUCT = U_MIN_HORIZON

#: **Measured in stage ``threads``, not chosen.**  See the module docstring.
THREADS = 2

ARMS = (
    ("referent", dict(join_coupling="tight")),
    ("repeat", dict(join_coupling="tight")),
    ("all_lagged", dict(join_coupling="lagged")),
    ("null_J3", dict(join_coupling="tight", null="J3")),
    ("null_J1", dict(join_coupling="tight", null="J1")),
)

#: **The prediction, recorded before the arms ran.**
PREDICTION = (
    "P2 (J3's receiving balance) passes again and its residual MOVES, because "
    "the residual is the ring-to-plane velocity gap and the repaired machine "
    "changes the thrust that makes that gap.  Direction not predicted.",
    "Every arm is admitted for all 600 macro-steps.  The FIRST attempt at "
    "this tier sized the machine for CS-19's release-state inflow and was "
    "declined at macro-step 312; sizing for the horizon's minimum was verified "
    "on a lagged march before these arms were launched.",
    "P3 (J1's parametric check) **passes**, where CS-19 failed it at "
    "3.482e-6 against 1e-6.  CS-20 diagnosed that failure as Jensen's term -- "
    "the clause averages a concave map over the settle window, so it measures "
    "the window's VARIANCE times the curvature, and the averaged residual was "
    "Jensen's term to a ratio of 0.9995.  These arms release from a settled "
    "field instead of a freestream, so the settle window is far steadier and "
    "the variance is far smaller.  **If P3 still fails, the CS-20 diagnosis is "
    "wrong** and that is the most informative outcome available here.",
    "P5 (the repeat floor) is bitwise, as it was.",
    "P6 splits as it did: the lagged column under the 0.5 s ceiling and the "
    "tight one over it.",
    "The machine's heat falls by about two orders -- CS-19 measured 3982 W "
    "with the machine motoring at I = -0.5605, and the repaired column "
    "generates at I = +0.031, so I^2 is about 320x smaller.",
)


def _retry(fn, attempts=40, pause=0.25):
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
    path = os.path.join(OUT, "racelab3.json")
    tmp = path + ".tmp"

    def write():
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(clean(res), fh, indent=1)

    _retry(write)
    _retry(lambda: os.replace(tmp, path))
    return path


def load_res():
    path = os.path.join(OUT, "racelab3.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def cs19():
    if not os.path.isfile(CS19):
        raise RuntimeError("out/racelab/racelab.json is absent; this tier is a "
                           "comparison and has nothing to compare against")
    with open(CS19, encoding="utf-8") as fh:
        return json.load(fh)


def save_field(name, **arrays):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name + ".npz")
    tmp = os.path.join(CACHE, name + ".tmp.npz")
    _retry(lambda: np.savez(tmp[:-4], **arrays))
    _retry(lambda: os.replace(tmp, path))
    return path


def settled():
    if not os.path.isfile(SETTLED):
        raise RuntimeError("no settled field; run "
                           "`python scripts/tier52_racelab_switch.py "
                           "--stages spinup` first")
    d = np.load(SETTLED)
    return d["u"], d["v"]


# ---------------------------------------------------------------------------
# stage: the thread count, measured
# ---------------------------------------------------------------------------


def stage_threads() -> dict:
    """Which thread counts give the ONE-thread answer bitwise, and which do not.

    The field and every trace are compared, because they are different objects:
    a domain sum's reduction order depends on the thread count and a window
    solve's does not.
    """
    u0, v0 = settled()
    base = None
    rows = []
    for n in (1, 2, 4, 8):
        torch.set_num_threads(n)
        t0 = time.perf_counter()
        m = RL.march(u0, v0, steps=5, join_coupling="tight", host_inflow=U_DUCT)
        wall = time.perf_counter() - t0
        cur = ({k: np.asarray(v).copy() for k, v in m.trace.items()},
               m.u.copy(), m.v.copy())
        if base is None:
            base = cur
            rows.append({"threads": n, "s_per_macro_step": wall / 5,
                         "field_bitwise": True, "all_traces_bitwise": True,
                         "traces_that_differ": {}})
            continue
        diff = {}
        for k in base[0]:
            if not np.array_equal(base[0][k], cur[0][k]):
                d = float(np.abs(base[0][k] - cur[0][k]).max())
                s = float(np.abs(base[0][k]).max())
                diff[k] = {"max_abs": d,
                           "relative": d / max(s, 1e-30)}
        rows.append({
            "threads": n, "s_per_macro_step": wall / 5,
            "field_bitwise": bool(np.array_equal(base[1], cur[1])
                                  and np.array_equal(base[2], cur[2])),
            "all_traces_bitwise": not diff,
            "traces_that_differ": diff,
        })
    torch.set_num_threads(THREADS)
    ok = [r for r in rows if r["field_bitwise"] and r["all_traces_bitwise"]]
    return {
        "rows": rows,
        "bitwise_identical_to_one_thread": [r["threads"] for r in ok],
        "chosen": THREADS,
        "why": "the field is bitwise at every count, but a full-domain "
               "reduction's order is not -- and the two traces that move are "
               "the ones J3's receiving balance is built from.  Two threads is "
               "the one-thread answer in the field AND in every trace, and is "
               "the fastest count that is",
        "W227": "a derived scalar is not bitwise reproducible above two "
                "threads while the field it is derived from is.  The vault's "
                "repeat-floor claims are therefore thread-count-conditional "
                "and have never said so",
    }


# ---------------------------------------------------------------------------
# stage: the arms
# ---------------------------------------------------------------------------


def stage_arms() -> dict:
    torch.set_num_threads(THREADS)
    u0, v0 = settled()
    out: dict = {"horizon": HORIZON, "settle_frac": SETTLE_FRAC,
                 "threads": THREADS, "host_inflow": U_DUCT,
                 "released_from": "out/racelab2/cache/settled.npz",
                 "enforce": True, "arms": {}}
    marches = {}
    for tag, kw in ARMS:
        print("   arm %s ..." % tag, flush=True)
        t0 = time.perf_counter()
        m = RL.march(u0, v0, steps=HORIZON, host_inflow=U_DUCT, **kw)
        wall = time.perf_counter() - t0
        marches[tag] = m
        bal = VM.receiver_balances(m, SETTLE_FRAC)
        n_set = max(1, int(round(SETTLE_FRAC * HORIZON)))
        wr = -np.asarray(m.trace["power_rotor"])[-n_set:]
        ps = np.asarray(m.trace["shaft_power"])[-n_set:]
        ratio = wr / np.where(np.abs(ps) > 1e-30, ps, 1.0)
        out["arms"][tag] = {
            "wall_s": wall, "s_per_macro_step": wall / HORIZON,
            "settled": m.settled(SETTLE_FRAC), "balances": bal,
            "outside_the_envelope_steps": m.notes["outside_the_envelope_steps"],
            "coolant_steps": len(m.coolant),
            "J3_ratio_per_step_band": {
                "mean": float(np.mean(ratio)), "ptp": float(np.ptp(ratio)),
                "relative_ptp": (float(np.ptp(ratio) / abs(np.mean(ratio)))
                                 if float(np.mean(ratio)) else None)},
            "notes": m.notes,
        }
        save_field("arm_" + tag, u=m.u, v=m.v)
        print("      %.1f s (%.3f s/step), outside %d"
              % (wall, wall / HORIZON,
                 m.notes["outside_the_envelope_steps"]), flush=True)
        persist({**load_res(), "arms": out})

    ref, rep = marches["referent"], marches["repeat"]
    out["repeat_floor"] = {
        "bitwise_in_every_crossing_quantity": all(
            np.array_equal(np.asarray(ref.trace[k]), np.asarray(rep.trace[k]))
            for k in VM.CROSSING_KEYS if k in ref.trace),
        "bitwise_in_the_field": bool(np.array_equal(ref.u, rep.u)
                                     and np.array_equal(ref.v, rep.v)),
    }
    rb = out["arms"]["referent"]["balances"]
    nb = out["arms"]["null_J3"]["balances"]
    w_null = nb["J3"]["the join's term: work the rotor's body force does on the fluid"]
    p_null = nb["J3"]["the shaft's claim, T <U_d>"]
    out["G1_J3"] = {
        "residual_with_the_term": rb["J3"]["residual_with_the_term"],
        "residual_without_the_term_null_arm": (abs(w_null - p_null) / abs(p_null)
                                               if p_null else None),
        "null_arm_work_on_the_fluid": w_null,
        "null_arm_shaft_still_claims": p_null,
        "u_ring": rb["J3"]["u on the ring the disk reads"],
        "u_plane": rb["J3"]["u in the cell the sink sits in"],
        "u_plane_over_u_ring": rb["J3"]["u_plane / u_ring"],
        "what_the_velocity_gap_alone_predicts":
            (1.0 - rb["J3"]["u_plane / u_ring"]
             if rb["J3"]["u_plane / u_ring"] else None),
        "the_ratio_s_own_band": out["arms"]["referent"]["J3_ratio_per_step_band"],
    }
    out["G3_J1"] = {
        "tracking_residual": rb["J1"]["tracking_residual"],
        "ua_release_to_settled": [rb["J1"]["UA at release"],
                                  rb["J1"]["the join's term: UA follows the air"]],
        "air_release_to_settled":
            rb["J1"]["air through the core, release -> settled"],
        "null_arm_ua_settled": out["arms"]["null_J1"]["settled"]["ua"],
        "null_arm_ua_is_exactly_UA_RAD":
            out["arms"]["null_J1"]["settled"]["ua"] == CL.UA_RAD,
        "null_arm_fluid_is_bitwise_identical": bool(
            np.array_equal(marches["referent"].u, marches["null_J1"].u)
            and np.array_equal(marches["referent"].v, marches["null_J1"].v)),
    }
    #: **P3's diagnosis, tested rather than assumed.**  CS-20 said the CS-19
    #: failure was Jensen's term; if so it must shrink with the window's
    #: variance, and these arms release from a settled field.
    n_set = max(1, int(round(SETTLE_FRAC * HORIZON)))
    ua = np.asarray(ref.trace["ua"])
    uc = np.asarray(ref.trace["u_core"])
    p = IU.UA_EXPONENT
    pred_pt = float(ua[0]) * (uc / float(uc[0])) ** p
    rel_var = float(np.var(uc[-n_set:]) / np.mean(uc[-n_set:]) ** 2)
    out["G3_J1"]["jensen"] = {
        "pointwise_residual_max": float(np.max(np.abs(ua - pred_pt)
                                               / np.abs(pred_pt))),
        "relative_variance_of_u_core_over_the_window": rel_var,
        "jensen_term": abs(0.5 * p * (p - 1.0) * rel_var),
        "averaged_residual": rb["J1"]["tracking_residual"],
    }
    j = out["G3_J1"]["jensen"]
    j["averaged_over_jensen"] = (j["averaged_residual"] / j["jensen_term"]
                                 if j["jensen_term"] else None)

    g = RL.GATE
    out["verdicts"] = {
        "P2_J3_receiving_balance": {
            "with": out["G1_J3"]["residual_with_the_term"],
            "without": out["G1_J3"]["residual_without_the_term_null_arm"],
            "tol_with": g["P2_J3_receiving_balance"]["tol_with"],
            "tol_without": g["P2_J3_receiving_balance"]["tol_without"],
            "verdict": ("pass" if (
                out["G1_J3"]["residual_with_the_term"] is not None
                and out["G1_J3"]["residual_with_the_term"]
                <= g["P2_J3_receiving_balance"]["tol_with"]
                and out["G1_J3"]["residual_without_the_term_null_arm"]
                >= g["P2_J3_receiving_balance"]["tol_without"]) else "fail")},
        "P3_J1_parametric": {
            "with": out["G3_J1"]["tracking_residual"],
            "tol_with": g["P3_J1_parametric"]["tol_with"],
            "null_is_exactly_UA_RAD": out["G3_J1"]["null_arm_ua_is_exactly_UA_RAD"],
            "null_fluid_bitwise": out["G3_J1"]["null_arm_fluid_is_bitwise_identical"],
            "verdict": ("pass" if (
                out["G3_J1"]["tracking_residual"] is not None
                and out["G3_J1"]["tracking_residual"]
                <= g["P3_J1_parametric"]["tol_with"]
                and out["G3_J1"]["null_arm_ua_is_exactly_UA_RAD"]
                and out["G3_J1"]["null_arm_fluid_is_bitwise_identical"])
                else "fail")},
        "P5_repeat_floor": {
            "bitwise": out["repeat_floor"]["bitwise_in_every_crossing_quantity"],
            "verdict": ("pass" if out["repeat_floor"][
                "bitwise_in_every_crossing_quantity"] else "fail")},
        "P6_macro_step_cost": {
            "lagged_s": out["arms"]["all_lagged"]["s_per_macro_step"],
            "tight_s": out["arms"]["referent"]["s_per_macro_step"],
            "ceiling_s": g["P6_macro_step_cost"]["ceiling_s"],
            "verdict_lagged": ("pass" if out["arms"]["all_lagged"][
                "s_per_macro_step"] <= g["P6_macro_step_cost"]["ceiling_s"]
                else "fail"),
            "verdict_tight": ("pass" if out["arms"]["referent"][
                "s_per_macro_step"] <= g["P6_macro_step_cost"]["ceiling_s"]
                else "fail")},
    }
    out["every_arm_inside_the_envelope"] = all(
        a["outside_the_envelope_steps"] == 0 for a in out["arms"].values())
    #: **The control for the sizing, and it is the reason the sizing is what it
    #: is.**  Sizing for the release-state inflow -- which is what CS-20 did --
    #: is admitted at the release state and declines at macro-step 312, because
    #: u_rotor falls 6.3% over this horizon and the machine's margin over its
    #: own crossover is 3.3% of u.
    out["sizing_control"] = {
        "sized_for_the_release_value": U_RELEASE,
        "declined_at_macro_step": 312,
        "u_rotor_over_the_horizon_at_that_sizing": {
            "start": 0.67208, "min": 0.63000, "end": 0.63000,
            "fall_percent": 6.26},
        "induction_at_that_sizing": {"start": 0.12665, "min": 0.02000,
                                     "clamp_floor": 0.02},
        "current_at_that_sizing": {"start": 0.03168, "end": -0.02367,
                                   "crosses_sign_near_macro_step": 300},
        "sized_for_the_horizon_minimum": U_MIN_HORIZON,
        "verified_before_these_arms_ran": {
            "outside_steps": 0, "of": 600,
            "u_rotor_min": 0.61753, "induction_min": 0.04767,
            "current_min": 0.008691},
        "the_residual": "the sizing is still one Picard step: sized for "
                        "0.63000, the new trajectory's minimum is 0.61753, a "
                        "1.98% residual -- and the margin the sizing creates "
                        "absorbs it, the induction there being 0.0477 against "
                        "a clamp floor of 0.02.  That is what a margin is for",
        "what_this_supersedes": "CS-20 section 2.2's repair is measured at the "
                                "RELEASE STATE and is not a statement about a "
                                "horizon.  That page said so -- it reported the "
                                "Picard residual rather than iterating it away "
                                "-- and this is how big the omission was",
    }
    return out


def stage_slow(res) -> dict:
    q = float(res["arms"]["arms"]["referent"]["settled"]["q_machine"])
    ua = float(res["arms"]["arms"]["referent"]["settled"]["ua"])
    long = VM.march_loop(steps=4000, q_machine=q, ua=ua, mounted=True)
    tw = np.array([r["t_wall_mean"] for r in long.rows])
    start, final = float(tw[0]), float(tw[-1])
    span = final - start
    tau = (int(np.argmax(np.abs(tw - start) >= abs(span) * (1.0 - 1.0 / np.e)))
           if span != 0.0 else 0) * CL.MACRO_DT
    with_term = VM.march_loop(steps=1200, q_machine=q, ua=ua, mounted=True)
    own = VM.march_loop(steps=1200, q_machine=q, ua=ua, mounted=False)

    def bal(lm, n=300):
        rows = lm.rows[-n:]
        return {"relative": float(np.mean([r["block_first_law"]["relative"]
                                           for r in rows])),
                "t_wall_K": float(rows[-1]["t_wall_mean"]),
                "t_return_K": float(rows[-1]["t_return"]),
                "q_block_W": float(rows[-1]["q_block"])}

    blk = IU.MountedBlock(q_machine=q)
    tc = np.full(CL.N_SEAM, CL.T_COOLANT_0)
    for _ in range(1200):
        blk.step(tc)
    bb = blk.energy_balance(tc)
    scale = max(abs(bb["q_outer"]), abs(bb["q_wall"]), 1e-30)
    without = abs(bb["stored_rate"] - (0.0 - bb["q_wall"])) / scale
    gg = RL.GATE["P4_J2_receiving_balance"]
    a = bal(with_term)
    return {
        "q_machine_W": q, "ua": ua,
        "block_thermal_time_constant_s": tau,
        "block_wall_start_K": start, "block_wall_settled_K": final,
        "coolant_steps_to_span_it": tau / CL.MACRO_DT,
        "with_the_mount_term": a,
        "the_block_with_its_own_declared_source": bal(own),
        "the_mount_balance_without_the_mount_term": without,
        "tol_with": gg["tol_with"], "tol_without": gg["tol_without"],
        "P4_verdict": ("pass" if (a["relative"] <= gg["tol_with"]
                                  and without >= gg["tol_without"]) else "fail"),
    }


# ---------------------------------------------------------------------------
# stage: the comparison
# ---------------------------------------------------------------------------


def stage_compare(res) -> dict:
    """CS-19 beside this run, read out of its artifact rather than retyped."""
    old = cs19()
    new = res["arms"]
    o_arms, n_arms = old["march"]["arms"], new["arms"]
    keys = ("u_rotor", "u_core", "ua", "induction", "current", "q_machine",
            "load", "drag", "u_max", "thrust_rotor", "shaft_power")
    settled_cmp = {}
    for k in keys:
        a = o_arms["referent"]["settled"].get(k)
        b = n_arms["referent"]["settled"].get(k)
        settled_cmp[k] = {
            "CS19": a, "repaired": b,
            "ratio": (b / a if (a not in (None, 0.0) and b is not None) else None),
        }
    clauses = {}
    for c in ("P2_J3_receiving_balance", "P3_J1_parametric", "P5_repeat_floor",
              "P6_macro_step_cost"):
        clauses[c] = {
            "CS19": old["march"]["verdicts"].get(c),
            "repaired": new["verdicts"].get(c),
        }
    out = {
        "settled": settled_cmp,
        "clauses": clauses,
        "envelope": {
            "CS19_every_arm_outside": old["envelope"]["every_arm_outside_the_envelope"],
            "repaired_every_arm_inside": new["every_arm_inside_the_envelope"],
        },
        "what_changed": [
            "the machine is sized for the inflow its host delivers",
            "the arms release from a settled field, not a freestream",
            "enforce=True: the envelope is consulted every macro-step",
        ],
        "what_did_not": [
            "the horizon (600), the settle fraction, the five arms",
            "every threshold in racelab.GATE, all inherited from CS-18",
            "vehicle_march.receiver_balances, unchanged",
        ],
        "wall_times_are_not_comparable":
            "CS-19's arms ran at one thread in a different process hours "
            "earlier; this tier's run at two.  The BALANCES are the "
            "comparison, and they are thread-independent.  Within-run the "
            "tight/lagged ratio is the cost number to quote",
    }
    if "slow" in res:
        out["clauses"]["P4_J2_receiving_balance"] = {
            "CS19": {"with": old["slow"]["with_the_mount_term"]["relative"],
                     "without": old["slow"][
                         "the_mount_balance_without_the_mount_term"],
                     "verdict": old["slow"]["P4_verdict"]},
            "repaired": {"with": res["slow"]["with_the_mount_term"]["relative"],
                         "without": res["slow"][
                             "the_mount_balance_without_the_mount_term"],
                         "verdict": res["slow"]["P4_verdict"]},
        }
        out["block"] = {
            "CS19_tau_s": old["slow"]["block_thermal_time_constant_s"],
            "repaired_tau_s": res["slow"]["block_thermal_time_constant_s"],
            "CS19_wall_K": old["slow"]["with_the_mount_term"]["t_wall_K"],
            "repaired_wall_K": res["slow"]["with_the_mount_term"]["t_wall_K"],
        }
    return out


STAGES = ("threads", "arms", "slow", "compare")


def main(argv):
    want = STAGES
    for i, a in enumerate(argv):
        if a == "--stages" and i + 1 < len(argv):
            want = tuple(x.strip() for x in argv[i + 1].split(","))
    res = load_res()
    t0 = time.perf_counter()
    res["meta"] = {"tier": 53, "horizon": HORIZON, "threads": THREADS,
                   "host_inflow": U_DUCT, "prediction": list(PREDICTION)}

    if "threads" in want:
        print("1. which thread counts give the one-thread answer bitwise",
              flush=True)
        res["threads"] = stage_threads()
        for r in res["threads"]["rows"]:
            print("   %d threads: field %s, traces %s, %.3f s/step  %s"
                  % (r["threads"], r["field_bitwise"], r["all_traces_bitwise"],
                     r["s_per_macro_step"],
                     ",".join(r["traces_that_differ"]) or ""), flush=True)
        persist(res)

    if "arms" in want:
        print("2. CS-19's five arms on the repaired column, %d macro-steps"
              % HORIZON, flush=True)
        res["arms"] = stage_arms()
        v = res["arms"]["verdicts"]
        for k, row in v.items():
            print("   %-26s %s" % (k, row.get("verdict")
                                   or "%s/%s" % (row.get("verdict_lagged"),
                                                 row.get("verdict_tight"))),
                  flush=True)
        print("   every arm inside the envelope: %s"
              % res["arms"]["every_arm_inside_the_envelope"], flush=True)
        persist(res)

    if "slow" in want:
        print("3. J2 on the coolant clock", flush=True)
        res["slow"] = stage_slow(res)
        print("   P4 %s: %.3g with the mount term, %.6f without"
              % (res["slow"]["P4_verdict"],
                 res["slow"]["with_the_mount_term"]["relative"],
                 res["slow"]["the_mount_balance_without_the_mount_term"]),
              flush=True)
        persist(res)

    if "compare" in want:
        print("4. CS-19 beside the repaired column", flush=True)
        res["compare"] = stage_compare(res)
        for k, row in res["compare"]["settled"].items():
            print("   %-14s CS19 %14.6g  repaired %14.6g  x%.4g"
                  % (k, row["CS19"] if row["CS19"] is not None else float("nan"),
                     row["repaired"] if row["repaired"] is not None else float("nan"),
                     row["ratio"] if row["ratio"] is not None else float("nan")),
                  flush=True)
        persist(res)

    res["elapsed_seconds"] = time.perf_counter() - t0
    print("wrote", persist(res), "in %.1f s" % res["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    main(sys.argv)
