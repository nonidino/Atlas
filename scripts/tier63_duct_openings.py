"""Tier 63 -- the duct opened: the solids rule cuts an inlet and an outlet, and the duct's flow is measured.

    python scripts/tier63_duct_openings.py --out out/racelab12 --stages car,march,compare,summary

Tier 62's march found the solid car's radiator duct flowing BACKWARDS (W278):
with solid walls the pod the drawing encloses is reached only through the
clearances round the wheels.  The user decided the rule should cut openings --
an inlet in the body shell ahead of the duct and an outlet behind it, sized by
editable numbers -- and that the duct's flow be measured before the radiator
core and the turbine are put in it.  This file does that, and nothing else:

  ``car``      the solids with the openings (`car_solids.SOLIDS_DEFAULT["openings"]`),
               the grids, the composite, a manufactured pressure on it and the
               uniform stream -- Tier 62's stage, on the rule as it now stands.
  ``march``    Tier 62's march, unchanged: from the stream to t = 16, statistics
               over [12, 16].
  ``compare``  the duct's flow against the porous column (Tier 59) and against
               Tier 62's sealed pod, and the car's forces against the sealed pod's.
  ``summary``  every registered prediction judged in code.

Nothing here touches RaceLab's column, its records, its gate or the demo.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse                                                         # noqa: E402
import copy                                                             # noqa: E402
import datetime as dt                                                   # noqa: E402
import json                                                             # noqa: E402
import sys                                                              # noqa: E402
import time                                                             # noqa: E402
import traceback                                                        # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(HERE, "scripts")
for _p in (HERE, SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from atlas.cases import car_solids as CS                                # noqa: E402
from atlas.cases import racelab as RL                                   # noqa: E402
import tier62_car_solids as T62                                         # noqa: E402

TIER62_RECORD = os.path.join(HERE, "out", "racelab11", "racelab11.json")
POROUS_RECORD = os.path.join(HERE, "out", "racelab8", "racelab8.json")


def geometry():
    """The rule as it now stands: `car_geometry.json` with no `solids` block, so
    `SOLIDS_DEFAULT` -- openings included -- is what runs."""
    return copy.deepcopy(RL.load_geometry())


READ_BEFORE_THIS_RUN = (
    "Tier 62's sealed pod over t in [12, 16]: the duct's mean streamwise velocity -0.109 at both "
    "device planes, reversed by t = 1.6; the car's drag 2.736 and vertical force +1.834 (the "
    "front wing +1.419, the body shell -0.901, the front wheel's drag 1.869); the porous column's "
    "settled rotor inflow +0.457.",
    "The openings chosen for the rule before anything was marched: an inlet removing 25% to 75% "
    "of CHASSIS's centreline (19.2 cells of a panel rising at 32 degrees ahead of the duct) and "
    "an outlet removing the last 20% of POD_UP (20.1 cells, ending just behind the duct's exit at "
    "x = 416.5). They split Tier 62's body shell into three solids: the nose with CHASSIS's front, "
    "the middle of the body (fillet 4.5 by the ladder, 8.2 cells added) and the tail with the rear "
    "wing and the diffuser (fillet 6, 67.6 cells added). The front wing is unchanged (fillet 6).",
    "The opened car's composite was built and nothing measured on it: 377,267 unknowns, no "
    "orphans, in 25 s after 60 s of solids.",
    "Nothing was marched on the opened car before these predictions were written.",
)

PREDICTION = (
    {"id": "D1", "claim": "the duct flows forward: its mean streamwise velocity at the turbine plane "
                          "over t in [12, 16] is positive",
     "why": "the inlet faces the stream on the chassis's slope and the outlet opens where the pod's "
            "top falls away behind the airbox, which should be the lower pressure"},
    {"id": "D2", "claim": "the duct carries a useful flow: that mean is at least 0.10, a fifth of the "
                          "porous column's 0.457",
     "why": "an inlet of 19 cells feeds a duct of 29 open cells and a pod that also leaks at the "
            "wheels; a fifth is a floor, not an estimate"},
    {"id": "D3", "claim": "and at most the porous column's 0.457",
     "why": "the porous plates let the whole pod carry flow; solid walls and two openings cannot "
            "do better"},
    {"id": "D4", "claim": "mass is conserved along the duct: over [12, 16] the core-plane and "
                          "turbine-plane fluxes differ by <= 2% of the core's",
     "why": "they agreed to 2e-5 on the sealed pod"},
    {"id": "D5", "claim": "the march is as sound as the sealed pod's: no blow-up with the largest "
                          "speed <= 4, the largest divergence over [12, 16] <= 0.2, the contour's net "
                          "flux <= 1% of its inflow, and the drag's spread <= 10% of its mean",
     "why": "Tier 62 read 3.82, 0.030, 0.14% and 1.1% on the same machinery"},
    {"id": "D6", "claim": "the openings do not undo the lift: the car's mean vertical force over "
                          "[12, 16] is still positive",
     "why": "the lift is the front wing's, which the openings do not touch"},
)


def stage_compare(res) -> dict:
    out: dict = {}
    M = res.get("march", {}).get("summary", {})
    if os.path.isfile(POROUS_RECORD):
        with open(POROUS_RECORD, encoding="utf-8") as fh:
            r8 = json.load(fh)
        out["porous_u_rotor_at_the_settled_state"] = r8["settle"]["u_rotor_at_the_settled_state"]
        out["porous_geometry_fingerprint"] = r8["geometry"]["fingerprint"]
    if os.path.isfile(TIER62_RECORD):
        with open(TIER62_RECORD, encoding="utf-8") as fh:
            r62 = json.load(fh)
        S62 = r62["march"]["summary"]
        out["sealed_turbine_u_mean"] = S62["turbine_u_mean_window"]
        out["sealed_fx_mean"] = S62["fx_mean_window"]
        out["sealed_fy_mean"] = S62["fy_mean_window"]
    if M:
        out["opened_turbine_u_mean"] = M["turbine_u_mean_window"]
        out["opened_core_u_mean"] = M["core_u_mean_window"]
        out["opened_fx_mean"] = M["fx_mean_window"]
        out["opened_fy_mean"] = M["fy_mean_window"]
        if "porous_u_rotor_at_the_settled_state" in out:
            out["turbine_ratio_to_porous"] = M["turbine_u_mean_window"] / out["porous_u_rotor_at_the_settled_state"]
    out["same_drawing"] = (out.get("porous_geometry_fingerprint") ==
                           res.get("car", {}).get("geometry_fingerprint"))
    res["compare"] = out
    T62.persist(res, T62.NAME + ".json")
    print("compare", json.dumps(T62.T60.clean(out)), flush=True)
    return out


def judge(res) -> dict:
    v: dict = {}
    M = res.get("march")
    if M and M.get("failed"):
        v["D5"] = False
    elif M and "summary" in M:
        S = M["summary"]
        u = S["turbine_u_mean_window"]
        v["D1"] = u > 0.0
        v["D2"] = u >= 0.10
        v["D3"] = u <= 0.457
        v["D4"] = S["duct_flux_mismatch_window"] <= 0.02
        v["D5"] = (S["steps_done"] == S["steps"] and S["u_max_max"] <= 4.0 and S["max_div_window"] <= 0.2
                   and S["contour_relative_window"] <= 0.01
                   and S["fx_spread_window"] <= 0.10 * abs(S["fx_mean_window"]))
        v["D6"] = S["fy_mean_window"] > 0.0
    return v


def stage_summary(res) -> dict:
    verdicts = judge(res)
    res["verdicts"] = verdicts
    res["verdicts_missing"] = sorted({p["id"] for p in PREDICTION} - set(verdicts))
    T62.persist(res, T62.NAME + ".json")
    for p in PREDICTION:
        print("%-3s %-6s %s" % (p["id"], verdicts.get(p["id"]), p["claim"][:100]), flush=True)
    return verdicts


STAGES = ("car", "march", "compare", "summary")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="car,march,compare,summary")
    args = ap.parse_args(argv)
    T62.configure(args.out)
    T62.GEOMETRY = geometry
    res = T62.load(T62.NAME + ".json")
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if "prediction" not in res:
        res.update(tier=63, prediction=list(PREDICTION),
                   prediction_recorded_at=dt.datetime.now().isoformat(timespec="seconds"),
                   read_before_this_run=list(READ_BEFORE_THIS_RUN),
                   rule=CS.solids_rule(geometry()))
        T62.persist(res, T62.NAME + ".json")
    elif [p["id"] for p in res["prediction"]] != [p["id"] for p in PREDICTION]:
        raise SystemExit("the record's prediction differs from this file's")
    for st in stages:
        if st not in STAGES:
            raise SystemExit(f"unknown stage {st!r}; stages are {STAGES}")
        t0 = time.perf_counter()
        try:
            {"car": T62.stage_car, "march": T62.stage_march, "compare": stage_compare,
             "summary": stage_summary}[st](res)
        except Exception:
            err = T62.load(T62.NAME + ".json")
            err.setdefault("stage_errors", {})[st] = traceback.format_exc()[-3000:]
            T62.persist(err, T62.NAME + ".json")
            print(f"stage {st} FAILED", flush=True)
            raise
        res.setdefault("stage_wall_s", {})[st] = time.perf_counter() - t0
        T62.persist(res, T62.NAME + ".json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
