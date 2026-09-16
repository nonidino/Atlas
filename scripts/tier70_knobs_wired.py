"""Tier 70 -- rake and the duct area wired, and the knob layer criterion 2 needs.

    python scripts/tier70_knobs_wired.py --out out/racelab19 --stages wire,respond,nominal,summary

Tier 69 measured section 3.3's eleven parameters against what they claim to
reach and found four wired, three dead, one unwireable, two reaching only
through a module global and one reaching half of what it claims.  This wires the
two that were dead and repairable.

  ``wire``     the reach report again, with `rake` and `duct_area` in it.
  ``respond``  `KnobState.set`: every offerable knob moved, with the subsystems
               that RESPONDED measured rather than asserted -- and the four that
               reach nothing refused at the point a value enters.
  ``nominal``  the check that matters more than either: wiring a knob must not
               move the car every cached field and record was built on.
  ``summary``  every registered prediction judged in code.

Nothing here marches.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse                                                         # noqa: E402
import datetime as dt                                                   # noqa: E402
import json                                                             # noqa: E402
import math                                                             # noqa: E402
import sys                                                              # noqa: E402
import time                                                             # noqa: E402
import traceback                                                        # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (HERE, os.path.join(HERE, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from atlas.cases import car_knobs as CK                                 # noqa: E402
from atlas.cases import car_solids as CS                                # noqa: E402
from atlas.cases import racelab as RL                                   # noqa: E402
import tier60_body_fitted_grids as T60                                  # noqa: E402

#: The car every committed record and cache was built on, measured on HEAD
#: before this tier's change (Tier 69's `powertrain` commit, 566f302).
NOMINAL_FINGERPRINT = "a1f67a8e279280a2e6b98ca77a01017afed294b4a1e1388fda04af09e5e77a8f"

READ_BEFORE_THIS_RUN = (
    "Tier 69 (out/racelab18): four of section 3.3's eleven knobs reach their subsystem; `rake` "
    "appears exactly once in the package and moves 0 of the car's 51 bodies; the duct's openings "
    "are fixed literals nothing scales; `road_speed` reaches no field because the column is "
    "nondimensional; and `battery_power` has nothing to reach because this circuit is a generator "
    "with no demand input.",
    "Section 3.3: a parameter that cannot reach its subsystem is omitted and the omission "
    "recorded.  Section 4.1: a parameter that would change the decomposition is clamped or the "
    "window set is rebuilt with a VISIBLE recompiling state.",
    "racelab's own note beside diffuser_deg and front_flap_deg: a default that disagreed with the "
    "drawing meant the model built a flap at 30 degrees drawn at 21.8 -- the editor showed one "
    "car and the march ran another -- so the defaults now ARE the drawing.",
    "Tier 68: a composite rebuild costs 66 to 90 s, so a geometry knob cannot re-cut the car "
    "inside a frame.",
    "Explored before these predictions were written: rake wired as a rigid pitch of the floor "
    "group about its leading edge moves 5 bodies with FLOOR_LE at the pivot; duct_area scaling "
    "each opening about its centre takes the total achieved span from 0.210 to 1.050 across its "
    "range; and HEAD's geometry fingerprint is a1f67a8e, which equals the fingerprint at rake = 0 "
    "and NOT the one at rake = 0.10.",
    "NOT measured before these predictions: whether the wired defaults leave the nominal car's "
    "fingerprint exactly where it was; whether every offerable knob reports a non-empty response; "
    "whether the four unofferable ones are refused at the point a value enters; and what the "
    "tally becomes.",
)

PREDICTION = (
    {"id": "M1", "claim": "the nominal car is UNCHANGED: the fingerprint at the wired defaults is "
                          "the one every committed record and cache was built on",
     "why": "rake's default moved 0.10 -> 0.0 and duct_area's is 1.0, both of which are the car as "
            "drawn, so the build is the same build"},
    {"id": "M2", "claim": "the tally becomes six wired, two global, one partial, one dead and one "
                          "absent",
     "why": "rake and duct_area move from dead to wired and nothing else changes"},
    {"id": "M3", "claim": "every offerable knob reports at least one subsystem that RESPONDED when "
                          "it is moved across its declared range",
     "why": "that is what `wired` and `global` mean, measured rather than declared"},
    {"id": "M4", "claim": "each of the four unofferable knobs is refused AT THE POINT A VALUE "
                          "ENTERS, with its reason, and so is any value outside a declared range",
     "why": "section 3.3's rule is worth nothing if a dashboard can still set the value"},
    {"id": "M5", "claim": "rake is a rigid pitch: the pivot does not move, the rear rises by "
                          "rake times its fractional distance, and every pitched plate gains the "
                          "same angle",
     "why": "a per-plate rise without the angle would be a staircase pretending to be a ramp"},
)

OUT = NAME = None


def configure(out: str) -> None:
    global OUT, NAME
    OUT = os.path.abspath(out)
    NAME = os.path.basename(os.path.normpath(OUT))


def persist(obj) -> str:
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, NAME + ".json")
    tmp = p + ".tmp"

    def write():
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(T60.clean(obj), fh, indent=1)

    T60._retry(write)
    T60._retry(lambda: os.replace(tmp, p))
    return p


def load() -> dict:
    p = os.path.join(OUT, NAME + ".json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def say(*a) -> None:
    print(*a, flush=True)


def stage_wire(res) -> dict:
    r = CK.reach_report()
    out = {"by_how": r["by_how"], "tally": {k: len(v) for k, v in r["by_how"].items()},
           "rake_bodies": CK.bodies_moved("rake", 0.0, 0.15),
           "duct_spans": {"lo": CK.duct_spans(0.3), "nominal": CK.duct_spans(1.0),
                          "hi": CK.duct_spans(1.5)},
           "duct_report_lo": CK.duct_report(0.3), "duct_report_hi": CK.duct_report(1.5)}
    res["wire"] = out
    say("  tally: %s" % json.dumps(out["tally"]))
    say("  rake moves %d bodies: %s" % (len(out["rake_bodies"]), out["rake_bodies"]))
    say("  duct total span %.3f -> %.3f -> %.3f"
        % (out["duct_spans"]["lo"], out["duct_spans"]["nominal"], out["duct_spans"]["hi"]))
    persist(res)
    return out


def stage_respond(res) -> dict:
    st = CK.KnobState()
    rows = {}
    for k in CK.KNOBS:
        if k in CK.must_not_be_shown():
            continue
        target = k.hi if st.values[k.name] != k.hi else k.lo
        r = st.set(k.name, target)
        rows[k.name] = {"from": r["from"], "to": r["to"], "responded": r["responded"],
                        "n_responded": len(r["responded"]), "needs_regrid": r["needs_regrid"]}
        say("  %-16s -> %-8s responded %d: %s" % (
            k.name, r["to"], len(r["responded"]), ", ".join(r["responded"])[:48] or "NOTHING"))
    refused = {}
    st2 = CK.KnobState()
    for k in CK.must_not_be_shown():
        try:
            st2.set(k.name, k.hi)
            refused[k.name] = {"refused": False}
        except ValueError as exc:
            refused[k.name] = {"refused": True, "why": str(exc)[:200]}
    try:
        st2.set("rake", 99.0)
        range_refused = False
    except ValueError:
        range_refused = True
    out = {"moved": rows, "refused": refused, "out_of_range_refused": range_refused,
           "all_responded": all(v["n_responded"] > 0 for v in rows.values())}
    res["respond"] = out
    say("  refused at entry: %s; out-of-range refused: %s"
        % (sorted(k for k, v in refused.items() if v["refused"]), range_refused))
    persist(res)
    return out


def stage_nominal(res) -> dict:
    """Wiring a knob must not move the car every cache and record was built on."""
    fp = RL.geometry_fingerprint()
    x0, length = RL.rake_span()
    lo = CK.built_bodies(rake=0.0)
    hi = CK.built_bodies(rake=0.15)
    pitch = math.degrees(math.atan(0.15 / length))
    out = {
        "fingerprint": fp,
        "nominal_fingerprint": NOMINAL_FINGERPRINT,
        "unchanged": fp == NOMINAL_FINGERPRINT,
        "default_rake": float(RL.CarParams().rake),
        "default_duct_area": float(CS.SOLIDS_DEFAULT["duct_area"]),
        "fingerprint_at_rake_0_10": RL.geometry_fingerprint(RL.CarParams(rake=0.10)),
        "rake_span": [x0, length],
        "pivot_moved": abs(hi["FLOOR_LE"][1] - lo["FLOOR_LE"][1]),
        "diff_rise": hi["DIFF"][1] - lo["DIFF"][1],
        "diff_rise_expected": 0.15 * (hi["DIFF"][0] - x0) / length,
        "pitch_deg": pitch,
        "pitch_gained": {k: hi[k][3] - lo[k][3]
                         for k in ("DIFF", "FLOOR", "FLOOR_LE", "FLOOR_STEP")},
    }
    out["rigid"] = bool(
        out["pivot_moved"] <= 1e-9
        and abs(out["diff_rise"] - out["diff_rise_expected"]) <= 2e-9
        and all(abs(v - pitch) <= 2e-9 for v in out["pitch_gained"].values()))
    res["nominal"] = out
    say("  fingerprint %s (unchanged: %s)" % (fp[:12], out["unchanged"]))
    say("  pivot moved %.2e, DIFF rises %.6f vs %.6f expected, pitch %.6f deg, rigid: %s"
        % (out["pivot_moved"], out["diff_rise"], out["diff_rise_expected"], pitch, out["rigid"]))
    persist(res)
    return out


def judge(res) -> dict:
    v = {}
    W, R, N = res.get("wire"), res.get("respond"), res.get("nominal")
    if N:
        v["M1"] = bool(N["unchanged"])
        v["M5"] = bool(N["rigid"])
    if W:
        t = W["tally"]
        v["M2"] = bool(t.get("wired") == 6 and t.get("global") == 2
                       and t.get("partial") == 1 and t.get("dead") == 1
                       and t.get("absent") == 1)
    if R:
        v["M3"] = bool(R["all_responded"])
        v["M4"] = bool(R["out_of_range_refused"]
                       and all(x["refused"] for x in R["refused"].values())
                       and len(R["refused"]) == 4)
    return v


def stage_summary(res) -> dict:
    verdicts = judge(res)
    res["verdicts"] = verdicts
    res["verdicts_missing"] = sorted({p["id"] for p in PREDICTION} - set(verdicts))
    persist(res)
    for p in PREDICTION:
        say("%-3s %-6s %s" % (p["id"], verdicts.get(p["id"]), p["claim"][:92]))
    if res["verdicts_missing"]:
        say("NOT JUDGED:", res["verdicts_missing"])
    return verdicts


STAGES = ("wire", "respond", "nominal", "summary")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default=",".join(STAGES))
    args = ap.parse_args(argv)
    configure(args.out)
    res = load()
    if "prediction" not in res:
        res.update(tier=70, prediction=list(PREDICTION),
                   prediction_recorded_at=dt.datetime.now().isoformat(timespec="seconds"),
                   read_before_this_run=list(READ_BEFORE_THIS_RUN),
                   machine_state=T60.machine_state())
        persist(res)
    elif [p["id"] for p in res["prediction"]] != [p["id"] for p in PREDICTION]:
        raise SystemExit("the record's prediction differs from this file's")
    for st in [s.strip() for s in args.stages.split(",") if s.strip()]:
        if st not in STAGES:
            raise SystemExit(f"unknown stage {st!r}; stages are {STAGES}")
        say("stage", st, "...")
        t0 = time.perf_counter()
        try:
            {"wire": stage_wire, "respond": stage_respond,
             "nominal": stage_nominal, "summary": stage_summary}[st](res)
        except Exception:
            err = load()
            err.setdefault("stage_errors", {})[st] = traceback.format_exc()[-3000:]
            persist(err)
            say(f"stage {st} FAILED")
            raise
        res.setdefault("stage_wall_s", {})[st] = time.perf_counter() - t0
        res.get("stage_errors", {}).pop(st, None)
        if res.get("stage_errors") == {}:
            res.pop("stage_errors", None)
        persist(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
