"""Tier 69 -- section 3.3's parameters, and what each one actually reaches.

    python scripts/tier69_car_knobs.py --out out/racelab18 --stages reach,machine,control,summary

Criterion 2 of the requirements is *"they can move a parameter and watch every
coupled subsystem respond"*, and its purpose is to show the coupling is REAL --
that one knob propagates through every subsystem the graph says it reaches.
Section 3.3 gives the rule that makes it testable: a slider that moves a number
nothing reads is worse than no slider.

  ``reach``    every knob moved end to end, with the quantity it CLAIMS to
               reach measured at both ends.
  ``machine``  the one defect repaired here: the motor-generator read the
               module's battery voltage rather than the battery it is wired to,
               so a state-of-charge knob moved the battery and left the machine
               -- and W282's envelope predicate -- unchanged.
  ``control``  the price of that repair, measured the way Tier 45 measures a
               compiler change: every captured artifact recompiled and compared
               byte for byte.
  ``summary``  every registered prediction judged in code.

Nothing here marches, and nothing here touches the body-fitted column.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse                                                         # noqa: E402
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

import numpy as np                                                      # noqa: E402

from atlas.cases import car_knobs as CK                                 # noqa: E402
from atlas.cases import powertrain as PT                                # noqa: E402
import tier60_body_fitted_grids as T60                                  # noqa: E402

READ_BEFORE_THIS_RUN = (
    "Requirements section 3.3: every parameter must be honestly wired; a slider that moves a "
    "number nothing reads is worse than no slider, and a parameter that cannot reach its "
    "subsystem is omitted and the omission recorded.  Amendment 13.1 already omitted brake duty, "
    "brake duct and plate stiffness.",
    "racelab.CarParams carries five geometry knobs (ride_height, rake, diffuser_deg, "
    "front_flap_deg, rear_wing_deg) and the geometry file has knob-owned entries -- y_plus_ride, "
    "y_plus_duct, alpha_from -- resolved at build time.",
    "Tier 64's W282: the machine generates only between 0.9731 and 1.1408 of its sizing inflow, "
    "and MachineAgent.validity is the predicate that envelope is written on.",
    "Explored before these predictions were written (out/t69_reach.json): ride_height moves four "
    "bodies, diffuser_deg, front_flap_deg and rear_wing_deg one each; RAKE MOVES NOTHING and "
    "appears exactly once in the package, its own declaration; the duct's openings are fixed "
    "literals in car_solids.SOLIDS_DEFAULT so duct_area reaches nothing; ground_effect.U_INF is "
    "1.0 and the column is nondimensional so road_speed reaches no field; the powertrain has no "
    "demand input at all, so battery_power cannot be wired; coolant_mdot moves the loop's return "
    "from 310.156 to 310.865 K and ambient_t from 288.133 to 325.778 K, both only by rebinding a "
    "module constant; and MachineAgent had NO v_oc field, reading the module V_OC in current_at "
    "and validity, so a state-of-charge knob moved BatteryLeg.emf and nothing else.",
    "Also explored: the first version of this reach test called a knob wired whenever its two "
    "readings differed, and since nan != nan is True in Python a probe that returned nan at both "
    "ends -- because it called an API that did not exist -- reported the knob as MOVING.",
    "NOT measured before these predictions: whether giving the machine its own v_oc and r_total "
    "leaves every existing test and every captured compile artifact untouched; where the validity "
    "threshold moves to at a lower state of charge; and whether the repair accidentally wires "
    "anything else.",
)

PREDICTION = (
    {"id": "L1", "claim": "giving MachineAgent its own v_oc and r_total leaves EVERY captured "
                          "compile artifact byte-identical",
     "why": "the new fields default to the module constants the methods used to read, so every "
            "existing caller computes exactly what it computed before"},
    {"id": "L2", "claim": "the machine's loop current and validity now follow the battery: at "
                          "v_oc = 0.80 the decline threshold moves from 13.4 to 8.0 rad/s",
     "why": "validity is k_e * omega > v_oc, so the threshold is v_oc / k_e and k_e is 0.10"},
    {"id": "L3", "claim": "exactly four of section 3.3's eleven knobs reach their subsystem as "
                          "declared; three reach nothing, one has nothing to reach, two reach "
                          "only through a module global and one reaches half of what it claims",
     "why": "measured knob by knob in the reach stage; this pins the tally so a later tier cannot "
            "quietly change it without the number moving"},
    {"id": "L4", "claim": "the reach test's `moved` refuses a pair of nans",
     "why": "nan != nan is True, so the obvious comparison calls a broken probe a working knob"},
    {"id": "L5", "claim": "rake still reaches nothing after the repair",
     "why": "the repair is in the powertrain and rake is geometry; this is the control that says "
            "the tier fixed the thing it meant to and not something else"},
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


def stage_reach(res) -> dict:
    out = CK.reach_report()
    out["nan_pair_refused"] = bool(CK.moved(float("nan"), float("nan")) is False)
    res["reach"] = out
    for name, row in out["knobs"].items():
        extra = ""
        if "n_moved" in row:
            extra = "%d bodies" % row["n_moved"]
        elif "lo_value" in row:
            extra = "%.3f -> %.3f" % (row["lo_value"], row["hi_value"])
        say("  %-16s %-8s reaches=%-5s %s" % (name, row["how"], row["reaches_something"], extra))
    say("  tally: %s" % json.dumps({k: len(v) for k, v in out["by_how"].items()}))
    persist(res)
    return out


def stage_machine(res) -> dict:
    """The repair, and where the envelope's threshold moves to."""
    m = PT.MachineAgent()
    lowered = PT.MachineAgent(v_oc=0.80)
    out = {
        "has_v_oc": "v_oc" in PT.MachineAgent.__dataclass_fields__,
        "has_r_total": "r_total" in PT.MachineAgent.__dataclass_fields__,
        "default_current_at_20": float(m.current_at(20.0)),
        "lowered_current_at_20": float(lowered.current_at(20.0)),
        "default_threshold": float(PT.V_OC / PT.K_E),
        "lowered_threshold": float(0.80 / PT.K_E),
        "validity_at_10": {"default": bool(m.validity([10.0])),
                           "v_oc_0.80": bool(lowered.validity([10.0]))},
        "module_V_OC": float(PT.V_OC), "module_R_TOTAL": float(PT.R_TOTAL),
        "note": ("the same shaft speed is OUTSIDE the envelope on a full battery and INSIDE "
                 "on a depleted one, which is physically right and was not expressible before"),
    }
    out["default_preserved"] = bool(
        out["default_current_at_20"] == (PT.K_E * 20.0 - PT.V_OC) / PT.R_TOTAL)
    res["machine"] = out
    say("  default current_at(20) %.6f (preserved: %s); at v_oc 0.80 -> %.6f"
        % (out["default_current_at_20"], out["default_preserved"], out["lowered_current_at_20"]))
    say("  decline threshold %.3f -> %.3f rad/s; validity at 10: %s"
        % (out["default_threshold"], out["lowered_threshold"], out["validity_at_10"]))
    persist(res)
    return out


#: The artifacts that ALREADY differ from Tier 45's capture, measured on HEAD
#: before this tier's change to `powertrain`.
#:
#: **The capture is a Tier 45 baseline and it accumulates.**  W194 moved the
#: two union fixtures, W285 moved `front_wing`, and other tiers moved the rest;
#: comparing a 2026-09-16 tree against it therefore reports sixteen differences
#: that have nothing to do with this tier.  What THIS tier has to answer is
#: whether it ADDS to that set, so the set is declared and `added` is the number.
MOVED_BEFORE_THIS_TIER = (
    "front_wing", "front_wing-riding", "ground_effect", "ground_effect-moving",
    "reuse_probe-N6", "scaling_ladder-N12-reference_exposed",
    "scaling_ladder-N6-reference", "scaling_ladder-N6-reference_exposed",
    "tier44-union", "tier44-union-with-pou", "wake_array-bare-blend",
    "wake_array-poseidon", "wake_array-reference", "wake_array-reference_exposed",
    "wind_farm_real", "wind_farm_real-as-built",
)


def stage_control(res) -> dict:
    """Tier 45's control, re-run: every captured artifact, byte for byte."""
    import w189_artifact_control as CTRL

    before_path = os.path.join(HERE, "out", "w189", "control_before1.json")
    if not os.path.isfile(before_path):
        res["control"] = {"tested": False, "why": "the pre-change manifest is not on disk"}
        persist(res)
        return res["control"]
    with open(before_path, encoding="utf-8") as fh:
        before = json.load(fh)
    rows = {}
    moved_keys = []
    for key in sorted(before["rows"]):
        try:
            _g, _r, text, _t = CTRL.compile_artifact(key)
        except Exception as exc:
            rows[key] = {"skipped": f"{type(exc).__name__}: {str(exc)[:120]}"}
            continue
        same = CTRL.digest(text) == before["rows"][key]["sha256"]
        rows[key] = {"same": bool(same)}
        if not same:
            moved_keys.append(key)
    baseline = set(MOVED_BEFORE_THIS_TIER)
    out = {"tested": True, "n": len(rows),
           "compared": sum(1 for v in rows.values() if "same" in v),
           "skipped": sum(1 for v in rows.values() if "skipped" in v),
           "moved": moved_keys, "rows": rows,
           "moved_before_this_tier": sorted(baseline),
           "added_by_this_tier": sorted(set(moved_keys) - baseline),
           "healed_by_this_tier": sorted(baseline - set(moved_keys)),
           "note": ("the capture is Tier 45's and it accumulates every deliberate change "
                    "since; `added_by_this_tier` is the number this tier answers for, and "
                    "it was established by running this same stage against HEAD's "
                    "powertrain, which gave the identical sixteen")}
    res["control"] = out
    say("  %d artifacts: %d compared, %d skipped, %d moved vs the Tier 45 capture"
        % (out["n"], out["compared"], out["skipped"], len(moved_keys)))
    say("  ADDED BY THIS TIER: %s   healed: %s"
        % (out["added_by_this_tier"] or "none", out["healed_by_this_tier"] or "none"))
    persist(res)
    return out


def judge(res) -> dict:
    v = {}
    C = res.get("control")
    M = res.get("machine")
    R = res.get("reach")
    if C and C.get("tested"):
        v["L1"] = bool(not C["moved"] and C["compared"] > 0)
    if M:
        v["L2"] = bool(abs(M["default_threshold"] - 13.4) < 1e-9
                       and abs(M["lowered_threshold"] - 8.0) < 1e-9
                       and M["validity_at_10"]["default"] is False
                       and M["validity_at_10"]["v_oc_0.80"] is True)
    if R:
        n = {k: len(vv) for k, vv in R["by_how"].items()}
        v["L3"] = bool(n.get("wired") == 4 and n.get("dead") == 3
                       and n.get("absent") == 1 and n.get("global") == 2
                       and n.get("partial") == 1)
        v["L4"] = bool(R["nan_pair_refused"])
        rake = R["knobs"].get("rake", {})
        v["L5"] = bool(rake.get("reaches_something") is False
                       and rake.get("n_moved") == 0)
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


STAGES = ("reach", "machine", "control", "summary")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default=",".join(STAGES))
    args = ap.parse_args(argv)
    configure(args.out)
    res = load()
    if "prediction" not in res:
        res.update(tier=69, prediction=list(PREDICTION),
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
            {"reach": stage_reach, "machine": stage_machine,
             "control": stage_control, "summary": stage_summary}[st](res)
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
