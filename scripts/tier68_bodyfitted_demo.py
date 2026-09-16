"""Tier 68 -- the body-fitted car as a demo column, and the cache that starts it.

    python scripts/tier68_bodyfitted_demo.py --out out/racelab17 --stages build,settle,frames,summary

Tiers 60-67 built a second column -- real walls, an open duct, the devices and
the joins marching in it, a declaration that compiles, an overlay and a measured
learned-expert territory -- and none of it had ever been on a screen.  This puts
it there, as a SECOND column beside the porous one (requirements 13.3).

  ``build``    the column assembled, with both caches: the probe operator
               (W287, closed here) and the settled release state.
  ``settle``   the evidence for the release state: releasing from the DEVICE-FREE
               prefix puts the rotor outside its machine's envelope on step one
               and keeps it there; releasing from a state settled WITH the
               devices on does not.
  ``frames``   every field the column offers, drawn, and every field it does not
               offer, named with a reason.
  ``summary``  every registered prediction judged in code.

Nothing here touches the porous column, its records, its gate or its demo.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse                                                         # noqa: E402
import datetime as dt                                                   # noqa: E402
import json                                                             # noqa: E402
import shutil                                                           # noqa: E402
import sys                                                              # noqa: E402
import time                                                             # noqa: E402
import traceback                                                        # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(HERE, "scripts")
for _p in (HERE, SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                      # noqa: E402

from atlas.cases import car_render as CR                                # noqa: E402
from atlas.cases import car_union as CU                                 # noqa: E402
from atlas.demo_racelab import bodyfitted as BF                         # noqa: E402
import tier60_body_fitted_grids as T60                                  # noqa: E402

#: How many steps the column is marched to judge the envelope.
N_STEPS = 30

READ_BEFORE_THIS_RUN = (
    "Requirements 13.3 (decided 2026-09-16): both columns switchable, the 61% drawn and "
    "labelled classical-only, the referent off by default, pressure added, temperature dropped "
    "on this column because the coolant loop is lumped and owns no region.",
    "Tier 66: the overlay is the solver's own probe operator over a raster; it costs 63 s to "
    "build at 320 x 180 and 0.0002 s to apply, and it is a function of the geometry alone -- "
    "W287 asked for it to be cached.",
    "Tier 67: a uniform 128 x 128 window reaches at most 33.4% of this column and the body "
    "grids hold 61.0% of the unknowns, so the learned switch can never be offered near the car.",
    "Tier 64: the machine generates only between 0.9731 and 1.1408 of its sizing inflow (W282); "
    "arm 2 was sized at u_host = 0.0829 and admitted, with every decline BEFORE its window, "
    "which began at t = 12 after releasing from the device-free prefix at t = 8.",
    "Explored before these predictions were written: a cold build costs 134.1 s (composite 66.0, "
    "raster 64.3, flow 2.6, devices 1.1) and the raster cache turns 64.3 s into 0.09 s, a 715x "
    "saving; the cache validator catches a wrong key, corrupted weights and a mask wrong by ONE "
    "pixel in either direction; released from the DEVICE-FREE prefix the rotor-plane ratio is "
    "1.2921 at step 1 and 1.2086 at step 60 and ALL 60 steps are outside the envelope, closing "
    "at about 0.0013 a step; released from a state settled to t = 12 with the devices on it is "
    "1.0348 and no step of 60 is outside.",
    "NOT measured before these predictions: what a build costs with BOTH caches warm; whether "
    "the settled state reloads deterministically; whether every OFFERED field's PNG carries the "
    "car exactly; whether the validator rejects a corrupted operator built on the CAR rather "
    "than on the verification composite; and whether a warm build does any settling at all.",
)

PREDICTION = (
    {"id": "K1", "claim": "with both caches warm the column builds in under a quarter of the "
                          "134.1 s cold build",
     "why": "the raster's 64.3 s becomes 0.09 s and the settle is skipped entirely, leaving the "
            "composite, the flow and the devices, which came to 69.7 s"},
    {"id": "K2", "claim": "a warm build does ZERO settling steps and still starts inside the "
                          "machine's envelope",
     "why": "the settled state was written at t = 12 by a march with the devices on, so there is "
            "nothing left to settle and the rotor is already in its band"},
    {"id": "K3", "claim": "released from the settled state, NO step of the run is outside the "
                          "envelope",
     "why": "Tier 64 admitted this arm over a window beginning at t = 12, which is where this "
            "state was written"},
    {"id": "K4", "claim": "every field the column OFFERS draws the car in exactly as many "
                          "car-coloured pixels as the raster masks",
     "why": "field_png writes the car's colour over the non-finite pixels the mask puts there, "
            "whatever field is being coloured"},
    {"id": "K5", "claim": "the cache validator rejects a corrupted operator built on the CAR, "
                          "not merely on the verification composite",
     "why": "the control it runs is the linear field laid on the live composite, which does not "
            "care how big the composite is"},
    {"id": "K6", "claim": "the column names a reason for every field it does not offer and for "
                          "the referent it does not run",
     "why": "section 4.3's rule is that what is not offered is greyed out WITH the reason; this "
            "applies it to fields and to the referent"},
    {"id": "K7", "claim": "the settled state reloads deterministically: two builds from it give "
                          "a first-step rotor ratio agreeing to 1e-9",
     "why": "the state is a saved vector and the step is the same arithmetic; nothing in the "
            "release is random"},
)

OUT = NAME = None
_COL = {}


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


def column():
    if "col" not in _COL:
        col = BF.BodyFittedColumn()
        t0 = time.perf_counter()
        rep = col.build(root=HERE)
        rep["wall_s"] = time.perf_counter() - t0
        _COL["col"] = col
        _COL["rep"] = rep
    return _COL["col"], _COL["rep"]


def stage_build(res) -> dict:
    col, rep = column()
    out = {k: rep.get(k) for k in
           ("composite_s", "raster_s", "flow_s", "devices_s", "settle_s", "total_s",
            "wall_s", "grids", "n_unknowns", "u_host_sized")}
    out["raster_cache"] = rep["raster"]
    out["released_from"] = rep["released_from"]
    out["coverage"] = rep["coverage"]
    out["cold_build_s"] = 134.1          # explored, on mains
    out["warm_over_cold"] = out["wall_s"] / 134.1
    res["build"] = out
    say("  build %.1f s (cold was 134.1) -- composite %.1f, raster %.2f (%s), settle %.1f"
        % (out["wall_s"], out["composite_s"] or 0, out["raster_s"] or 0,
           "hit" if rep["raster"]["hit"] else "MISS", out["settle_s"] or 0))
    say("  released from %s" % (out["released_from"],))
    persist(res)
    return out


def stage_settle(res) -> dict:
    """Does the released column sit inside its machine's envelope?"""
    col, _rep = column()
    band = CU.machine_band(col.union.u_host, col.union.width)

    # K7: the release is deterministic.  Measured by RELOADING the settled state
    # into this same flow and taking one step, twice -- not by building a second
    # column, which would cost another 90 s composite to learn the same thing.
    settled = col.build_report["released_from"].get("file")
    reload_deltas = None
    if settled:
        p = os.path.join(HERE, settled)
        ratios = []
        for _ in range(2):
            CU.load_state(col.flow, p)
            col.union.trace["u_rotor"].clear()
            col.union.step()
            ratios.append(float(col.union.trace["u_rotor"][-1]) / col.union.u_host)
        reload_deltas = abs(ratios[0] - ratios[1])
        CU.load_state(col.flow, p)            # release cleanly for the run below
        for k in col.union.trace:
            col.union.trace[k].clear()
        col.union.outside_steps = 0
        say("  reload determinism: %.4f vs %.4f, delta %.2e"
            % (ratios[0], ratios[1], reload_deltas))

    rows = []
    for _ in range(N_STEPS):
        r = col.step()
        tr = col.union.trace
        u_rot = float(tr["u_rotor"][-1]) if tr.get("u_rotor") else float("nan")
        env = col.union.envelope or {}
        rows.append({"i": r["i"], "t": r["t"], "u_rotor": u_rot,
                     "ratio": u_rot / col.union.u_host,
                     "declined": list(env.get("_declined") or []),
                     "wall_s": r["wall_s"], "ilu_s": r["ilu_s"]})
    declined = [r for r in rows if r["declined"]]
    out = {"n": N_STEPS, "u_host": float(col.union.u_host),
           "band": {k: float(v) for k, v in band.items() if isinstance(v, (int, float))},
           "ratio_first": rows[0]["ratio"], "ratio_last": rows[-1]["ratio"],
           "declined_steps": len(declined), "outside_steps": int(col.union.outside_steps),
           "steps_per_second": col.steps_per_second(),
           "reload_ratio_delta": reload_deltas,
           "rows": rows,
           "from_the_device_free_prefix": {
               "ratio_first": 1.2920886, "ratio_at_60": 1.2086,
               "declined_steps_of_60": 60,
               "note": "explored: releasing from prefix_t8 is outside the band on every one of "
                       "60 steps and closes at about 0.0013 a step, so about 300 steps -- three "
                       "minutes -- would pass before the column said anything trustworthy"}}
    res["settle"] = out
    say("  %d steps: ratio %.4f -> %.4f, band [%.4f, %.4f], declined %d"
        % (N_STEPS, out["ratio_first"], out["ratio_last"],
           out["band"]["ratio_lo"], out["band"]["ratio_hi"], out["declined_steps"]))
    persist(res)
    return out


def stage_frames(res) -> dict:
    col, _rep = column()
    from PIL import Image
    import io as _io

    masked = int(col.raster.mask.sum())
    fields = {}
    for name in BF.FIELDS:
        png, payload = col.frame(name)
        im = np.asarray(Image.open(_io.BytesIO(png)).convert("RGB"))
        car = int(np.all(im == np.asarray(CR.CAR_RGB, dtype=np.uint8), axis=-1).sum())
        fields[name] = {"bytes": len(png), "lo": payload["lo"], "hi": payload["hi"],
                        "clipped_pixels": payload["clipped_pixels"],
                        "field_max": payload["field_max"],
                        "car_pixels": car, "masked": masked,
                        "car_matches_mask": car == masked}
        say("  %-6s %6d bytes  car %d of %d masked  clipped %d"
            % (name, len(png), car, masked, payload["clipped_pixels"]))
    tel = col.telemetry()
    out = {"fields": fields, "masked": masked,
           "offered": list(BF.FIELDS), "holes": dict(BF.FIELD_HOLES),
           "referent": tel["referent"], "learned_hole": BF.LEARNED_HOLE,
           "telemetry_keys": sorted(tel)}
    # K5: the validator, on the CAR's own operator
    out["validator"] = _validator_on_the_car(col)
    res["frames"] = out
    persist(res)
    return out


def _validator_on_the_car(col) -> dict:
    """Plant a corrupted operator built on the CAR and check it is refused."""
    import tempfile

    src = col.build_report["raster"]["path"]
    if not os.path.isfile(src):
        return {"tested": False, "why": "the car's cache file is not on disk"}
    z = dict(np.load(src, allow_pickle=True))
    key = str(z["key"])
    z["indices"] = (z["indices"] + 11) % int(z["shape"][1])
    d = tempfile.mkdtemp()
    bad = os.path.join(d, "bad.npz")
    np.savez(bad, **z)
    got, why = CR.load_raster(col.ov, bad, key)
    shutil.rmtree(d, ignore_errors=True)
    return {"tested": True, "refused": got is None, "why": why}


def judge(res) -> dict:
    v = {}
    B = res.get("build")
    S = res.get("settle")
    F = res.get("frames")
    if B:
        v["K1"] = bool(B["warm_over_cold"] < 0.25)
        v["K2"] = bool((B.get("settle_s") or 0.0) == 0.0)
    if S:
        if "K2" in v:
            v["K2"] = bool(v["K2"] and S["band"]["ratio_lo"] <= S["ratio_first"]
                           <= S["band"]["ratio_hi"])
        v["K3"] = bool(S["declined_steps"] == 0)
        # NOT MEASURED stays out of the verdicts; a missing measurement must
        # not read as a passed one
        if S.get("reload_ratio_delta") is not None:
            v["K7"] = bool(S["reload_ratio_delta"] <= 1e-9)
    if F:
        v["K4"] = bool(all(f["car_matches_mask"] for f in F["fields"].values()))
        v["K5"] = bool(F["validator"].get("refused") is True)
        v["K6"] = bool(F["holes"] and all(len(w) > 40 for w in F["holes"].values())
                       and bool(F["referent"].get("why"))
                       and "61.0%" in F["learned_hole"])
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


STAGES = ("build", "settle", "frames", "summary")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default=",".join(STAGES))
    args = ap.parse_args(argv)
    configure(args.out)
    res = load()
    if "prediction" not in res:
        res.update(tier=68, prediction=list(PREDICTION),
                   prediction_recorded_at=dt.datetime.now().isoformat(timespec="seconds"),
                   read_before_this_run=list(READ_BEFORE_THIS_RUN),
                   machine=T60.machine_state())
        persist(res)
    elif [p["id"] for p in res["prediction"]] != [p["id"] for p in PREDICTION]:
        raise SystemExit("the record's prediction differs from this file's")
    for st in [s.strip() for s in args.stages.split(",") if s.strip()]:
        if st not in STAGES:
            raise SystemExit(f"unknown stage {st!r}; stages are {STAGES}")
        say("stage", st, "...")
        t0 = time.perf_counter()
        try:
            {"build": stage_build, "settle": stage_settle,
             "frames": stage_frames, "summary": stage_summary}[st](res)
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
