"""Tier 54 -- the traced car, re-sized and re-marched.

The car stopped being a hand-drawn arrangement of plates on 2026-09-13 and
became a TRACED silhouette: a CC0 Formula One side view rasterised, thresholded
and mapped isotropically into the cell box.  That moved every body, moved both
wheels, and moved the radiator duct's band from y 44..76 down to y 18..50 --
because a real Formula One body TAPERS behind the cockpit and the old band put
the duct's roof where a real engine cover has no metal.

**So every marched number in CS-19, CS-20 and Tier 53 describes a different
vehicle, and this tier re-measures them.**  What does NOT change: the settle
fraction, the five arms, the gate's thresholds (`racelab.GATE`, every one
inherited from CS-18), and `vehicle_march.receiver_balances`.  Anything that
moves, moves because the car did.

**The HORIZON does move, and that is a finding rather than a convenience.**
CS-19's 600 macro-steps are not available to this car: the traced body is
blockier than the hand-drawn one -- bigger wheels, an airbox, a floor that runs
between the wheels instead of through them -- so it accelerates the flow more,
and the window expert's declared cell-Reynolds bound of 8 is breached partway
along.  The arms therefore run at the longest horizon the car stays INSIDE its
declared envelopes for, chosen by the probe below, and every comparison with
Tier 53 is a comparison at a shorter horizon and says so.

**The machine has to be re-sized, and that is the first stage rather than a
constant.**  Tier 53's W228 found that sizing a machine for the inflow at ONE
INSTANT buys a few hundred macro-steps and then the margin is consumed: size
for the horizon's MINIMUM instead.  `U_DUCT` is therefore MEASURED here, not
written down -- stage ``size`` marches the horizon with ``enforce=False``,
records ``u_rotor`` every macro-step, and reports the minimum; stage ``verify``
checks a short enforced march at that value before the arms are committed to.

**The horizon is CHOSEN by the probe, and the first version of this script got
that wrong in a way worth recording.**  It read only the ``u_rotor`` band out of
the probe, gated the arms on an eighty-step enforced march, and lost a
twenty-five-minute referent arm to a decline at macro-step 552 -- while the
probe's own result already said the envelope would object on **46** macro-steps.
Two recorded lessons at once: *march a repair as far as the failure it repairs*,
and a gate that reads a different quantity from the one that predicts the
failure is not a gate.  So ``verify`` now marches the WHOLE horizon the arms
will use, and ``arms`` refuses to run at any other.

Stages, each persisted to ``out/racelab4/racelab4.json``:

  ``spinup``  settle the traced car's field from the freestream, enforce=False,
              and record what the check would have said.
  ``size``    the ``u_rotor`` band, the minimum to size for, AND the horizon the
              car stays inside its declared envelopes for.
  ``verify``  that whole horizon, ENFORCED, before the arms are committed to.
  ``arms``    the five arms at the verified horizon with enforce=True.
  ``compare`` this run beside Tier 53's, clause by clause.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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

from atlas.cases import racelab as RL                                   # noqa: E402
from atlas.cases import vehicle_march as VM                             # noqa: E402

OUT = os.path.join(HERE, "out", "racelab4")
CACHE = os.path.join(OUT, "cache")
TIER53 = os.path.join(HERE, "out", "racelab3", "racelab3.json")

HORIZON = 600
SETTLE_FRAC = 0.25
THREADS = 2                    # W227: bitwise the one-thread answer, and fastest
VERIFY_STEPS = 80

ARMS = (
    ("referent", dict(join_coupling="tight")),
    ("repeat", dict(join_coupling="tight")),
    ("all_lagged", dict(join_coupling="lagged")),
    ("null_J3", dict(join_coupling="tight", null="J3")),
    ("null_J1", dict(join_coupling="tight", null="J1")),
)

#: **The prediction, recorded before stage ``size`` ran.**
PREDICTION = (
    "u_rotor in the traced duct is DIFFERENT from Tier 53's 0.6692, because "
    "the duct moved from y 44..76 to y 18..50 -- nearer the floor, where the "
    "ground-effect flow is faster -- and the body above it changed shape. "
    "Direction not predicted.",
    "u_rotor still FALLS over the horizon, as it did in Tier 53 (6.3%), "
    "because CS-19 section 6.2's finding that this flow does not settle is a "
    "property of the decomposition and not of the old car's shape.",
    "Sizing for the horizon's MINIMUM admits all 600 macro-steps, as W228's "
    "procedure did for the old car.  If it does not, the procedure is "
    "insufficient rather than mis-applied, and that is the informative "
    "outcome.",
    "P5 (the repeat floor) is bitwise, because nothing about determinism "
    "depends on the car's shape.",
    "The aerodynamic coefficients MOVE A LOT -- the traced car has 22 plates "
    "against 11, an airbox, and a floor that no longer runs through both "
    "wheels -- so downforce and drag are not comparable with Tier 53's and "
    "this tier does not pretend they are.",
)


def _retry(fn, tries=6, delay=0.35):
    """OneDrive holds fresh files; os.replace can raise WinError 5."""
    for k in range(tries):
        try:
            return fn()
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(delay * (k + 1))


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
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
    path = os.path.join(OUT, "racelab4.json")
    tmp = path + ".tmp"

    def write():
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(clean(res), fh, indent=1)

    _retry(write)
    _retry(lambda: os.replace(tmp, path))
    return path


def load_res():
    path = os.path.join(OUT, "racelab4.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_field(name, **arrays):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name + ".npz")
    tmp = os.path.join(CACHE, name + ".tmp.npz")      # savez appends .npz
    _retry(lambda: np.savez(tmp[:-4], **arrays))
    _retry(lambda: os.replace(tmp, path))
    return path


def settled():
    path = os.path.join(CACHE, "settled.npz")
    if not os.path.isfile(path):
        raise RuntimeError("no settled field; run --stages spinup first")
    d = np.load(path)
    return d["u"], d["v"]


# ---------------------------------------------------------------------------


def stage_spinup() -> dict:
    """Settle the traced car from the freestream, with the check watching."""
    torch.set_num_threads(THREADS)
    t0 = time.perf_counter()
    n = 0

    def progress(s, _):
        nonlocal n
        n = s
        if s % 20 == 0:
            print("      spin %d/%d  %.0f s" % (s, RL.N_SPIN,
                                                time.perf_counter() - t0),
                  flush=True)

    u, v, rep = RL.settled_field(steps=RL.N_SPIN, host_inflow=None,
                                 progress=progress)
    wall = time.perf_counter() - t0
    save_field("settled", u=u, v=v)
    rep["wall_s"] = wall
    rep["s_per_macro_step"] = wall / RL.N_SPIN
    print("   settled in %.0f s (%.3f s/step); u_rotor at release %.6f"
          % (wall, wall / RL.N_SPIN, rep["u_rotor_at_the_release_state"]),
          flush=True)
    return rep


def stage_size(res) -> dict:
    """March the horizon UNENFORCED and read the u_rotor band."""
    torch.set_num_threads(THREADS)
    u0, v0 = settled()
    u_rel = res["spinup"]["u_rotor_at_the_release_state"]
    print("   probing the horizon at host_inflow=%.6f ..." % u_rel, flush=True)
    t0 = time.perf_counter()
    m = RL.march(u0, v0, steps=HORIZON, host_inflow=u_rel,
                 join_coupling="lagged", enforce=False)
    wall = time.perf_counter() - t0
    ur = np.asarray(m.trace["u_rotor"], dtype=float)
    out = {
        "probe_host_inflow": u_rel,
        "wall_s": wall,
        "s_per_macro_step": wall / HORIZON,
        "u_rotor_first": float(ur[0]),
        "u_rotor_last": float(ur[-1]),
        "u_rotor_min": float(ur.min()),
        "u_rotor_max": float(ur.max()),
        "u_rotor_argmin": int(ur.argmin()),
        "fall_fraction": float((ur.max() - ur.min()) / ur.max()),
        "outside_the_envelope_steps": m.notes["outside_the_envelope_steps"],
        "outside_the_envelope_first": m.notes.get("outside_the_envelope"),
        "U_DUCT_to_size_for": float(ur.min()),
        "note": "W228: size for the horizon's MINIMUM, not for one instant. "
                "This probe runs with enforce=False by design -- it is "
                "measuring what the envelope would object to, so it must not "
                "be stopped by it.",
    }
    # **The horizon is chosen from this probe and not inherited blind.**
    # The first version of this script read only the u_rotor band here, gated
    # the arms on an 80-step enforced verify, and lost a 25-minute referent arm
    # to a decline at macro-step 552 -- while THIS dict already said the
    # envelope would object on 46 macro-steps.  A gate that reads a different
    # quantity from the one that predicts the failure is not a gate.
    first = None
    ofs = out["outside_the_envelope_first"]
    if isinstance(ofs, dict):
        first = ofs.get("step")
    out["first_outside_macro_step"] = first
    if out["outside_the_envelope_steps"] == 0:
        out["safe_horizon"] = HORIZON
        out["horizon_is_CS19_s"] = True
    else:
        margin = 24
        out["safe_horizon"] = (max(60, int(first) - margin) if first is not None
                               else HORIZON // 2)
        out["horizon_is_CS19_s"] = False
        out["why_the_horizon_moved"] = (
            "the traced car leaves the FLUID envelope before CS-19's 600 "
            "macro-steps: the window expert's cell-Reynolds bound of 8 is "
            "breached at macro-step %s, on %d macro-steps of the probe. The "
            "arms run at %s instead, which is inside it with %d macro-steps "
            "of margin, and the comparison is therefore at a SHORTER horizon "
            "and says so."
            % (first, out["outside_the_envelope_steps"],
               out.get("safe_horizon"), margin))
    print("   u_rotor %.6f -> %.6f (min %.6f at %d), fall %.2f%%"
          % (out["u_rotor_first"], out["u_rotor_last"], out["u_rotor_min"],
             out["u_rotor_argmin"], 100 * out["fall_fraction"]), flush=True)
    return out


def stage_verify(res) -> dict:
    """A short ENFORCED march at the measured sizing, before the long one."""
    torch.set_num_threads(THREADS)
    u0, v0 = settled()
    ud = res["size"]["U_DUCT_to_size_for"]
    #: **The verify marches the WHOLE horizon the arms will use, not a token
    #: prefix.**  [[positive-controls-need-a-horizon]]: a repair verified for
    #: 80 macro-steps and committed to for 600 is not verified, and this script
    #: learned that by losing a referent arm at macro-step 552.
    steps = int(res["size"].get("safe_horizon") or VERIFY_STEPS)
    print("   verifying %d enforced macro-steps at host_inflow=%.6f ..."
          % (steps, ud), flush=True)
    t0 = time.perf_counter()
    admitted, why = True, None
    try:
        m = RL.march(u0, v0, steps=steps, host_inflow=ud,
                     join_coupling="lagged", enforce=True)
    except RL.EnvelopeDeclined as exc:
        admitted, why, m = False, str(exc), None
    wall = time.perf_counter() - t0
    out = {"steps": steps, "host_inflow": ud, "admitted": admitted,
           "why_declined": why, "wall_s": wall,
           "probe_said_outside_steps": res["size"]["outside_the_envelope_steps"],
           "probe_said_first_outside_at": res["size"].get(
               "first_outside_macro_step")}
    if m is not None:
        ind = np.asarray(m.trace["induction"], dtype=float)
        cur = np.asarray(m.trace["current"], dtype=float)
        out["induction_min"] = float(ind.min())
        out["induction_max"] = float(ind.max())
        out["current_min"] = float(cur.min())
        out["current_is_positive_throughout"] = bool((cur > 0).all())
    print("   admitted=%s  %s" % (admitted, why or ""), flush=True)
    return out


def stage_arms(res) -> dict:
    torch.set_num_threads(THREADS)
    u0, v0 = settled()
    ud = res["size"]["U_DUCT_to_size_for"]
    if not res.get("verify", {}).get("admitted"):
        raise RuntimeError("stage verify did not admit; refusing to commit "
                           "ninety minutes to a sizing that is already known "
                           "to be declined")
    horizon = int(res["verify"]["steps"])
    if horizon != res["size"].get("safe_horizon"):
        raise RuntimeError("the arms' horizon and the verified one disagree; "
                           "that is how a repair gets committed to on the "
                           "strength of a shorter march than it will run")
    out: dict = {"horizon": horizon, "horizon_is_CS19_s":
                 res["size"].get("horizon_is_CS19_s"),
                 "why_the_horizon_moved":
                     res["size"].get("why_the_horizon_moved"),
                 "settle_frac": SETTLE_FRAC,
                 "threads": THREADS, "host_inflow": ud,
                 "released_from": "out/racelab4/cache/settled.npz",
                 "enforce": True, "arms": {}}
    marches = {}
    for tag, kw in ARMS:
        print("   arm %s ..." % tag, flush=True)
        t0 = time.perf_counter()
        m = RL.march(u0, v0, steps=horizon, host_inflow=ud, **kw)
        wall = time.perf_counter() - t0
        marches[tag] = m
        bal = VM.receiver_balances(m, SETTLE_FRAC)
        n_set = max(1, int(round(SETTLE_FRAC * horizon)))
        wr = -np.asarray(m.trace["power_rotor"])[-n_set:]
        ps = np.asarray(m.trace["shaft_power"])[-n_set:]
        ratio = wr / np.where(np.abs(ps) > 1e-30, ps, 1.0)
        out["arms"][tag] = {
            "wall_s": wall, "s_per_macro_step": wall / horizon,
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
              % (wall, wall / horizon,
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
        "u_plane_over_u_ring": rb["J3"]["u_plane / u_ring"],
        "what_the_velocity_gap_alone_predicts":
            (1.0 - rb["J3"]["u_plane / u_ring"]
             if rb["J3"]["u_plane / u_ring"] else None),
        "the_ratio_s_own_band": out["arms"]["referent"]["J3_ratio_per_step_band"],
    }
    out["G3_J1"] = {
        "tracking_residual": rb["J1"]["tracking_residual"],
        "null_arm_ua_settled": out["arms"]["null_J1"]["settled"]["ua"],
        "null_arm_fluid_is_bitwise_identical": bool(
            np.array_equal(marches["referent"].u, marches["null_J1"].u)
            and np.array_equal(marches["referent"].v, marches["null_J1"].v)),
    }
    return out


def stage_compare(res) -> dict:
    """This run beside Tier 53's, on the clauses that are comparable."""
    if not os.path.isfile(TIER53):
        return {"skipped": "out/racelab3/racelab3.json is absent"}
    with open(TIER53, encoding="utf-8") as fh:
        t53 = json.load(fh)
    a53 = t53.get("arms", {}).get("arms", {})
    a54 = res.get("arms", {}).get("arms", {})
    out = {"note": "the aerodynamic coefficients are NOT comparable -- the car "
                   "is a different shape -- and are shown to say by how much, "
                   "not to be judged against a threshold"}
    for tag in ("referent", "all_lagged"):
        if tag in a53 and tag in a54:
            out[tag] = {
                "s_per_macro_step": [a53[tag]["s_per_macro_step"],
                                     a54[tag]["s_per_macro_step"]],
                "outside_steps": [a53[tag]["outside_the_envelope_steps"],
                                  a54[tag]["outside_the_envelope_steps"]],
            }
    g53, g54 = t53.get("arms", {}), res.get("arms", {})
    out["J3_residual"] = [g53.get("G1_J3", {}).get("residual_with_the_term"),
                          g54.get("G1_J3", {}).get("residual_with_the_term")]
    out["J1_tracking_residual"] = [g53.get("G3_J1", {}).get("tracking_residual"),
                                   g54.get("G3_J1", {}).get("tracking_residual")]
    out["host_inflow"] = [g53.get("host_inflow"), g54.get("host_inflow")]
    return out


STAGES = ("spinup", "size", "verify", "arms", "compare")


def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(STAGES))
    args = ap.parse_args(argv)
    want = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in want:
        if s not in STAGES:
            raise SystemExit("unknown stage %r; known: %s" % (s, STAGES))

    res = load_res()
    res.setdefault("prediction", list(PREDICTION))
    res.setdefault("geometry", {
        "source": "freesvg.org id 48844, CC0 / public domain",
        "method": "rasterised 1200x310 in a browser, thresholded to a "
                  "silhouette, profiles simplified with Douglas-Peucker and "
                  "welded into a chain, mapped ISOTROPICALLY",
        "map": "x = 94.0 + raster_x * 0.3760, y = (309 - raster_y) * 0.3760",
        "plates": 22, "flat_bodies": 46,
        "duct_band": [RL.DUCT_Y0, RL.DUCT_Y0 + RL.DEVICE_CELLS],
        "core_range": list(RL.DUCT_CORE_RANGE),
        "turbine_range": list(RL.TURBINE_RANGE),
    })
    persist(res)

    for s in want:
        print("== stage %s ==" % s, flush=True)
        t0 = time.perf_counter()
        if s == "spinup":
            res[s] = stage_spinup()
        elif s == "size":
            res[s] = stage_size(res)
        elif s == "verify":
            res[s] = stage_verify(res)
        elif s == "arms":
            res[s] = stage_arms(res)
        elif s == "compare":
            res[s] = stage_compare(res)
        res.setdefault("stage_wall_s", {})[s] = time.perf_counter() - t0
        print("   -> %s" % persist(res), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
