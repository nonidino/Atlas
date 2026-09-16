"""Tier 71 -- the dashboard, and the coolant knob that finally reaches the march.

    python scripts/tier71_dashboard.py --out out/racelab20 --stages settings,control,commit,summary

Tiers 69 and 70 wired section 3.3's knobs and measured what each reaches.  The
dashboard was built on top and immediately proved the one thing the measurement
could not: the page reported the coolant loop RESPONDING while the marching
circuit's return temperature never moved, because the only route in was
rebinding a module constant that the running `CarUnion` had already read.  That
was W291, and this closes it.

  ``settings``  `cooling_loop.LoopSettings` threaded through the legs, the
                solve, the sweep study and `CarUnion` -- the knob reaching the
                loop with no module constant touched, and the leg's weight hash
                moving with it.
  ``control``   the price, measured the way Tier 45 measures one: every captured
                artifact recompiled and compared byte for byte.
  ``commit``    a geometry knob committed end to end: the car re-cut, a new
                fingerprint, and a release state that belongs to the new car.
  ``summary``   every registered prediction judged in code.
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
for _p in (HERE, os.path.join(HERE, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from atlas.cases import car_knobs as CK                                 # noqa: E402
from atlas.cases import cooling_loop as CL                              # noqa: E402
from atlas.cases import integration_union as IU                         # noqa: E402
import tier60_body_fitted_grids as T60                                  # noqa: E402

#: The artifacts that already differ from Tier 45's capture, established on HEAD
#: in Tier 69.  This tier asks only what it ADDS.
MOVED_BEFORE_THIS_TIER = (
    "front_wing", "front_wing-riding", "ground_effect", "ground_effect-moving",
    "reuse_probe-N6", "scaling_ladder-N12-reference_exposed",
    "scaling_ladder-N6-reference", "scaling_ladder-N6-reference_exposed",
    "tier44-union", "tier44-union-with-pou", "wake_array-bare-blend",
    "wake_array-poseidon", "wake_array-reference", "wake_array-reference_exposed",
    "wind_farm_real", "wind_farm_real-as-built",
)

NOMINAL_FINGERPRINT = "a1f67a8e279280a2e6b98ca77a01017afed294b4a1e1388fda04af09e5e77a8f"

READ_BEFORE_THIS_RUN = (
    "Tier 69's W291: coolant_mdot and ambient_t reached the loop only by rebinding "
    "cooling_loop.MDOT and T_AMB, which ten call sites read directly, and a leg's weight_hash is "
    "built from MDOT -- so moving the knob moved a capability record's identity with nothing "
    "noticing.",
    "Tier 70: KnobState.set reports the subsystems that responded; a geometry knob is flagged "
    "needs_regrid and does not re-cut the car, because a rebuild costs 66-90 s.",
    "The user's decision (2026-09-16): the dashboard rebuilds ON COMMIT, not on slider move.",
    "Explored before these predictions were written, ON THE RUNNING DASHBOARD: the page loaded "
    "and /api/meta answered while EVERY WebSocket upgrade was refused with 403, so no frame "
    "would ever have arrived -- the module carries `from __future__ import annotations`, so "
    "`sock: WebSocket` is a string FastAPI resolves against MODULE globals, and FastAPI had been "
    "imported inside create_app; and with that repaired the page marched at about 2 steps a "
    "second while the coolant return stood at exactly 313.633745 K across every step, with the "
    "ambient knob reporting that cooling_loop had responded.",
    "Also explored: LoopSettings threaded through the legs reproduces the rebinding probe's own "
    "numbers -- 310.155908 K at mdot 0.05 and 325.778299 K at t_amb 318 -- with no module "
    "constant moved, and the knob tally goes from 6 wired / 2 global to 8 wired / 0 global.",
    "NOT measured before these predictions: whether the LIVE march's return temperature follows "
    "the knob and by how much; whether threading the settings leaves every captured artifact "
    "where Tier 69 left it; and whether a geometry commit completes and lands on a car with a "
    "different fingerprint and a release state of its own.",
)

PREDICTION = (
    {"id": "N1", "claim": "the default LoopSettings is the module constants field for field, so "
                          "nothing that existed changes",
     "why": "every field defaults to the constant it replaces, which is the whole reason the "
            "refactor can be additive"},
    {"id": "N2", "claim": "a full reach report moves NO module constant, where Tier 69's rebound "
                          "two and put them back",
     "why": "the probe now passes settings instead of assigning to cooling_loop.MDOT"},
    {"id": "N3", "claim": "the leg's weight_hash moves with the coolant flow, so a capability "
                          "record cannot claim the machine that used to run",
     "why": "the hash is built from the leg's own settings tag rather than from the module"},
    {"id": "N4", "claim": "threading the settings adds NOTHING to Tier 45's moved set",
     "why": "the defaults are the constants the legs used to read, so every existing graph "
            "compiles to the bytes it compiled to before"},
    {"id": "N5", "claim": "a geometry commit lands on a DIFFERENT fingerprint and an HONEST "
                          "release: either a state named for that car, or a refusal to march "
                          "at all -- never the old car's field",
     "why": "the settled state's filename carries the fingerprint and load_state checks the "
            "unknown count, so the only two outcomes left are this car's field or none"},
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


def stage_settings(res) -> dict:
    d = CL.LoopSettings()
    before = (CL.MDOT, CL.T_AMB, CL.UA_RAD, CL.W_PUMP, CL.CP_COOLANT)
    rows = {}
    for mdot, t_amb in ((0.05, 300.0), (0.15, 300.0), (0.30, 300.0),
                        (0.15, 273.0), (0.15, 318.0)):
        st = CL.LoopSettings(mdot=mdot, t_amb=t_amb)
        r = CL.LoopSolve(block=IU.MountedBlock(q_machine=0.0), settings=st).solve()
        rows["mdot%g_tamb%g" % (mdot, t_amb)] = {
            "t_return": float(r.t_return), "residual": float(r.residual),
            "weight_tag": st.tag()}
    CK.reach_report()          # the whole thing, to see if it leaves a mark
    after = (CL.MDOT, CL.T_AMB, CL.UA_RAD, CL.W_PUMP, CL.CP_COOLANT)
    out = {
        "defaults_are_the_constants": bool(
            d.mdot == CL.MDOT and d.t_amb == CL.T_AMB and d.ua_rad == CL.UA_RAD
            and d.w_pump == CL.W_PUMP and d.cp == CL.CP_COOLANT),
        "constants_before": list(before), "constants_after": list(after),
        "constants_untouched": bool(before == after),
        "solves": rows,
        "weight_tags_differ": bool(CL.LoopSettings(mdot=0.05).tag()
                                   != CL.LoopSettings(mdot=0.30).tag()),
        "tally": {k: len(v) for k, v in CK.reach_report()["by_how"].items()},
    }
    res["settings"] = out
    say("  defaults are the constants: %s; constants untouched by a full reach report: %s"
        % (out["defaults_are_the_constants"], out["constants_untouched"]))
    for k, v in rows.items():
        say("    %-22s t_return %.6f K  tag %s" % (k, v["t_return"], v["weight_tag"]))
    say("  tally: %s" % json.dumps(out["tally"]))
    persist(res)
    return out


def stage_control(res) -> dict:
    import w189_artifact_control as CTRL

    before_path = os.path.join(HERE, "out", "w189", "control_before1.json")
    if not os.path.isfile(before_path):
        res["control"] = {"tested": False, "why": "the pre-change manifest is not on disk"}
        persist(res)
        return res["control"]
    with open(before_path, encoding="utf-8") as fh:
        before = json.load(fh)
    moved = []
    n = 0
    for key in sorted(before["rows"]):
        try:
            _g, _r, text, _t = CTRL.compile_artifact(key)
        except Exception:
            continue
        n += 1
        if CTRL.digest(text) != before["rows"][key]["sha256"]:
            moved.append(key)
    baseline = set(MOVED_BEFORE_THIS_TIER)
    out = {"tested": True, "compared": n, "moved": moved,
           "added_by_this_tier": sorted(set(moved) - baseline),
           "healed_by_this_tier": sorted(baseline - set(moved))}
    res["control"] = out
    say("  %d artifacts compared, %d moved; ADDED %s, healed %s"
        % (n, len(moved), out["added_by_this_tier"] or "none",
           out["healed_by_this_tier"] or "none"))
    persist(res)
    return out


def stage_commit(res) -> dict:
    """A geometry knob committed end to end."""
    from atlas.demo_racelab import bodyfitted as BF

    col = BF.BodyFittedColumn()
    stages = []
    t0 = time.perf_counter()
    rep0 = col.build(root=HERE, progress=lambda p: stages.append(p.get("stage")))
    build_s = time.perf_counter() - t0
    fp0 = rep0["fingerprint"]
    rel0 = rep0["released_from"]

    r = col.set_knob("rake", 0.12)
    pending = list(col.pending_regrid)

    t0 = time.perf_counter()
    rep1 = col.regrid(root=HERE, progress=lambda p: stages.append(p.get("stage")))
    commit_s = time.perf_counter() - t0

    rel1 = rep1["released_from"]
    out = {
        "build_s": build_s, "commit_s": commit_s,
        "fingerprint_before": fp0, "fingerprint_after": rep1["fingerprint_after"],
        "changed": fp0 != rep1["fingerprint_after"],
        "nominal_before": fp0 == NOMINAL_FINGERPRINT,
        "knob": {k: r[k] for k in ("knob", "from", "to", "responded",
                                   "needs_regrid", "marching")},
        "pending_after_knob": pending,
        "pending_after_commit": list(col.pending_regrid),
        "released_before": rel0, "released_after": rel1,
        "stages": stages,
        "n_unknowns_before": rep0["n_unknowns"], "n_unknowns_after": rep1["n_unknowns"],
        "unknowns_changed_by": rep1["n_unknowns"] - rep0["n_unknowns"],
        "prefix_refused": rep1.get("prefix_refused"),
        "needs_spin_up": bool(col.needs_spin_up),
        # A release is honest when it either names THIS car's file or declines
        # to march at all -- never when it silently reuses another car's field.
        "release_is_honest": bool(
            (rel1.get("file") and rep1["fingerprint_after"][:12] in str(rel1["file"]))
            or rel1.get("spun_up") is False),
    }
    if col.needs_spin_up:
        try:
            col.step()
            out["declined_to_march"] = False
        except RuntimeError as exc:
            out["declined_to_march"] = True
            out["decline_reason"] = str(exc)[:200]
    res["commit"] = out
    say("  build %.1f s, commit %.1f s" % (build_s, commit_s))
    say("  fingerprint %s -> %s (changed: %s)"
        % (fp0[:12], rep1["fingerprint_after"][:12], out["changed"]))
    say("  pending after knob %s, after commit %s"
        % (out["pending_after_knob"], out["pending_after_commit"]))
    say("  unknowns %d -> %d (%+d)" % (out["n_unknowns_before"], out["n_unknowns_after"],
                                        out["unknowns_changed_by"]))
    if out.get("prefix_refused"):
        say("  prefix refused: %s" % out["prefix_refused"][:120])
    say("  released: %s" % (out["released_after"].get("kind"),))
    say("  release honest: %s; needs spin-up: %s; declined to march: %s"
        % (out["release_is_honest"], out["needs_spin_up"], out.get("declined_to_march")))
    persist(res)
    return out


def judge(res) -> dict:
    v = {}
    S, C, M = res.get("settings"), res.get("control"), res.get("commit")
    if S:
        v["N1"] = bool(S["defaults_are_the_constants"])
        v["N2"] = bool(S["constants_untouched"])
        v["N3"] = bool(S["weight_tags_differ"])
    if C and C.get("tested"):
        v["N4"] = bool(not C["added_by_this_tier"])
    if M:
        v["N5"] = bool(M["changed"] and M["release_is_honest"]
                       and not M["pending_after_commit"])
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


STAGES = ("settings", "control", "commit", "summary")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default=",".join(STAGES))
    args = ap.parse_args(argv)
    configure(args.out)
    res = load()
    if "prediction" not in res:
        res.update(tier=71, prediction=list(PREDICTION),
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
            {"settings": stage_settings, "control": stage_control,
             "commit": stage_commit, "summary": stage_summary}[st](res)
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
