"""Tier 66 -- the body-fitted composite on one raster: the demo's field overlay.

    python scripts/tier66_car_render.py --out out/racelab15 --stages raster,blank,state,summary

The porous column's fluid is fourteen rectangles on one lattice and a picture of
it is an array slice.  The body-fitted column is twelve overlapping grids and
there is no array that holds the answer, so the demo cannot draw it at all until
something turns the composite into one rectangle of pixels.
`atlas/cases/car_render.py` is that something, and the point of this tier is that
it is the SOLVER'S OWN interpolation asked over a raster, not a second opinion:

  ``raster``  the operator built at three resolutions -- what it costs, what it
              masks, and the two controls (a linear field must come back exact,
              a quadratic must not).
  ``blank``   what drawing the background alone would lose, which is the reason
              the body grids have to be sampled at all.
  ``state``   Tier 64's cached state rendered: the mask against a second state,
              the operator against a second build, a pixel against a direct
              probe, a frame against a march step measured on the same machine
              in the same process, and the PNG the page would receive.
  ``summary`` every registered prediction judged in code.

Nothing here marches the car for its own sake -- `state` takes five steps, and
only to time a frame against one: the first step after a state is loaded
refactors the incomplete LU and is five times a typical one, so timing a single
step would divide by the wrong number.  Nothing here touches RaceLab's porous
column, its records, its gate or its demo.
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

from atlas.cases import car_render as CR                                # noqa: E402
from atlas.cases import car_solids as CS                                # noqa: E402
from atlas.cases import car_union as CU                                 # noqa: E402
import tier60_body_fitted_grids as T60                                  # noqa: E402
import tier62_car_solids as T62                                         # noqa: E402
import tier63_duct_openings as T63                                      # noqa: E402

#: Tier 64's own composite: the duct's openings ON.  Tier 62's script pins them
#: off so its sealed-pod record still reproduces, so the geometry has to be
#: swapped before the first `_car()` call, exactly as `tier64_car_union` does it.
T62.GEOMETRY = T63.geometry

PREFIX = os.path.join(HERE, "out", "racelab13", "cache", "prefix_t8.npz")

#: The rasters this tier measures.  640 x 360 was explored and is not re-run:
#: it costs 261.6 s to build for a picture the page scales down anyway.
RASTERS = ((160, 90), (320, 180), (480, 270))

#: Tier 64's median step on this machine, for orientation only.  The prediction
#: is judged against a step measured in THIS process (W-lesson: wall clock rots,
#: so a ratio has to be measured with both halves on the same machine at the
#: same time).
TIER64_MEDIAN_STEP_S = 0.6228487499993207

#: How many steps the cost check times.  More than one because the first step
#: after a state is loaded refactors the incomplete LU and is not a typical step.
STEP_SAMPLES = 5

READ_BEFORE_THIS_RUN = (
    "Tier 62-64: the body-fitted composite is 12 grids (a Cartesian background 241 x 672 at "
    "h = 0.015625 and eleven curvilinear) with 377,267 unknowns, and it marches at a median "
    "0.62-0.81 s a step (out/racelab13: prefix 0.8076, null arm 0.6228).",
    "Tier 65: the composite is ONE implicit solve -- 12,078 interpolation rows over 40 grid pairs "
    "-- so no grid can be stepped or drawn alone, and the fluid is one expert.",
    "car_union.probe_matrix: the devices' strips already read the composite at arbitrary points, "
    "taking the first body grid in ov.comps order that holds the point and the background "
    "otherwise, and RAISING when a point is held by nothing.",
    "Explored before these predictions were written (out/t66_explore.json), on this composite and "
    "Tier 64's cached prefix state at t = 8: a raster of 320 x 180 over the background's own "
    "extent leaves 2,785 points (4.84%) held by no grid; the masked set and the set inside the "
    "solids' outlines agreed EXACTLY there, 2,785 both ways and 0 either way; the operator cost "
    "15.9 / 63.9 / 261.6 s to build at 160x90 / 320x180 / 640x360 and 0.0001 / 0.0003 / 0.0019 s "
    "to apply; a linear field came back at 3.55e-15, 3.55e-15 and 2.66e-15 and a quadratic at "
    "5.42e-6, 4.46e-6 and 5.51e-6; and of the 54,815 drawn pixels at 320 x 180 the background "
    "owned 49,678 and the eleven body grids 5,137.",
    "NOT measured before these predictions: whether the mask is the car at a raster the "
    "exploration did not try; how much of the picture the BACKGROUND ALONE could not draw; "
    "whether the mask and the operator depend on the state or on the build; whether a raster "
    "pixel is the same number as a direct probe; what a frame costs beside a step measured in the "
    "same process; and what the PNG actually contains.",
)

PREDICTION = (
    {"id": "G1", "claim": "the masked set is EXACTLY the set inside the solids' outlines at every "
                          "raster tried, 0 either way, not just at the 320 x 180 the exploration "
                          "checked",
     "why": "the mask is where the donor search fails and the outlines are what the grids were cut "
            "from; neither depends on the raster, so agreeing once is agreeing always"},
    {"id": "G2", "claim": "drawing the background alone would leave at least 3,000 of the "
                          "320 x 180 raster's drawable pixels blank",
     "why": "the hole margin cuts the background a margin BEYOND each body, and the body grids "
            "claim 5,137 pixels in exactly that region"},
    {"id": "G3", "claim": "the mask and the source map do not depend on the state: identical for "
                          "Tier 64's prefix and for a random vector",
     "why": "the operator is built from the geometry and the donor search alone, and reads no "
            "solution at all"},
    {"id": "G4", "claim": "building the same raster twice gives a bitwise identical operator",
     "why": "the donor search is a fixed sequence of array operations with no randomness and no "
            "iteration count that depends on the machine"},
    {"id": "G5", "claim": "a raster pixel is the SAME NUMBER as a direct probe at that point, to "
                          "0.0 exactly and not merely to rounding",
     "why": "the raster is that operator applied at those points; there is no second path for the "
            "two to differ along"},
    {"id": "G6", "claim": "a frame costs under 1% of a march step, both measured in this process",
     "why": "a frame is one sparse matrix-vector product against a step that solves the momentum "
            "equations and a pressure Poisson on 377,267 unknowns"},
    {"id": "G7", "claim": "the PNG carries exactly as many car-coloured pixels as the mask has "
                          "masked ones, and the field's colour map never produces that colour",
     "why": "field_png writes the car's colour AFTER the colour map, over the non-finite pixels "
            "the mask put there"},
    {"id": "G8", "claim": "at a raster the exploration did not try, the linear control stays at "
                          "rounding and the quadratic control does NOT improve",
     "why": "both errors are set by the GRID spacing; refining the pixels cannot refine the grids, "
            "which is what makes the pair identify what is being measured"},
)

OUT = NAME = None
_CAR = {}


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


def car():
    """The composite, built once per process (it costs about a minute)."""
    if "ov" not in _CAR:
        t0 = time.perf_counter()
        C = T62._car()
        _CAR.update(ov=C["ov"], solids=C["solids"], build_s=time.perf_counter() - t0)
    return _CAR


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def stage_raster(res) -> dict:
    """The operator at three resolutions: cost, mask, and the two controls."""
    C = car()
    ov, solids = C["ov"], C["solids"]
    out = res.setdefault("raster", {})
    out["composite"] = {"grids": len(ov.grids), "components": len(ov.comps),
                        "n_unknowns": int(ov.n_unknowns), "build_s": C["build_s"],
                        "bg_shape": list(ov.bg.shape), "h": float(ov.bg.h)}
    out["by_raster"] = {}
    for nx, ny in RASTERS:
        r = CR.CompositeRaster(ov, nx, ny)
        rep = r.report()
        rep["controls"] = CR.linear_control(r)
        try:
            rep["mask_vs_solids"] = CR.mask_against_solids(r, solids)
        except ImportError:
            # matplotlib only; a clone without it records NOT MEASURED rather
            # than a False that would read as "the mask is not the car"
            rep["mask_vs_solids"] = None
        rep["apply_s"] = r.apply_seconds(np.zeros(ov.n_unknowns))
        out["by_raster"]["%dx%d" % (nx, ny)] = rep
        say("  %4dx%-4d masked %5d (%.2f%%)  build %6.1f s  apply %.5f s  "
            "lin %.2e quad %.2e  mask-vs-solids %s" % (
                nx, ny, rep["masked"], 100 * rep["masked_frac"], rep["build_s"],
                rep["apply_s"], rep["controls"]["linear_max_err"],
                rep["controls"]["quadratic_max_err"],
                "NOT MEASURED" if rep["mask_vs_solids"] is None else
                ("EXACT" if (rep["mask_vs_solids"]["masked_not_inside"] == 0
                             and rep["mask_vs_solids"]["inside_not_masked"] == 0) else "DIFFERS")))
        if (nx, ny) == (CR.NX, CR.NY):
            _CAR["raster"] = r
    persist(res)
    return out


def stage_blank(res) -> dict:
    """What the background ALONE could not draw -- the reason to sample the bodies.

    The control the demo would have shipped without this tier is 'draw the
    background', so it is measured rather than asserted: the same donor search,
    restricted to the background, over the same raster.
    """
    C = car()
    ov = C["ov"]
    r = _CAR.get("raster") or CR.CompositeRaster(ov, CR.NX, CR.NY)
    _CAR["raster"] = r
    bg_res = ov._cartesian_donors(r.px, r.py)
    bg_ok = np.asarray(bg_res["ok"], dtype=bool).reshape(r.ny, r.nx)
    drawable = ~r.mask
    blank = drawable & ~bg_ok
    out = {"drawable": int(drawable.sum()), "background_can_draw": int((drawable & bg_ok).sum()),
           "background_blank": int(blank.sum()),
           "background_blank_frac_of_drawable": float(blank.sum() / max(1, drawable.sum())),
           "note": ("pixels the composite can draw and the background alone cannot: the "
                    "boundary layers, the wheel clearances and the cooling duct")}
    res["blank"] = out
    say("  drawable %d, background alone can draw %d, BLANK %d (%.1f%%)" % (
        out["drawable"], out["background_can_draw"], out["background_blank"],
        100 * out["background_blank_frac_of_drawable"]))
    persist(res)
    return out


def stage_state(res) -> dict:
    """Tier 64's cached state rendered, and the five checks on the operator."""
    C = car()
    ov, solids = C["ov"], C["solids"]
    r = _CAR.get("raster") or CR.CompositeRaster(ov, CR.NX, CR.NY)
    _CAR["raster"] = r
    out = {}

    flow = CS.car_flow(ov, solids, precond="ilu")
    st = CU.load_state(flow, PREFIX)
    out["state"] = {"t": float(st["t"]), "k": int(st["k"]), "file": os.path.relpath(PREFIX, HERE)}

    F = r.fields(flow.U, flow.V, flow.P)
    ok = ~r.mask
    out["fields"] = {k: [float(np.nanmin(v[ok])), float(np.nanmax(v[ok]))] for k, v in F.items()}
    out["nonfinite_drawn"] = int((~np.isfinite(F["speed"][ok])).sum())

    # G3: the mask does not depend on the state
    rng = np.random.default_rng(66)
    other = rng.normal(size=ov.n_unknowns)
    F2 = r.fields(other, other[::-1].copy(), None)
    out["state_independence"] = {
        "mask_identical": bool(np.array_equal(r.mask, ~np.isfinite(F2["speed"]))),
        "note": "the mask is the operator's, so a different state must not move it"}

    # G4: the operator is deterministic
    r2 = CR.CompositeRaster(ov, r.nx, r.ny)
    out["determinism"] = {
        "indices_identical": bool(np.array_equal(r.M.indices, r2.M.indices)),
        "indptr_identical": bool(np.array_equal(r.M.indptr, r2.M.indptr)),
        "data_bitwise_identical": bool(np.array_equal(r.M.data.view(np.uint64),
                                                      r2.M.data.view(np.uint64))),
        "mask_identical": bool(np.array_equal(r.mask, r2.mask)),
        "source_identical": bool(np.array_equal(r.source, r2.source))}

    # G5: a pixel is a direct probe
    sel = np.flatnonzero(~r.mask.ravel())[::997][:64]
    Md, missing, _s = CU.probe_matrix_masked(ov, r.px[sel], r.py[sel])
    direct = Md @ flow.U
    pixels = (r.M @ flow.U)[sel]
    out["pixel_vs_direct"] = {
        "n": int(sel.size), "any_missing": bool(missing.any()),
        "max_abs_diff": float(np.max(np.abs(direct - pixels))) if sel.size else None}

    # G6: a frame against a step, both here and now.
    #
    # The FIRST step after a state is loaded refactors the incomplete LU
    # (`overset_ns` starts with `_ilu = None`), and it is 3.24 s against a
    # median of 0.62 s in Tier 64.  Timing one step would therefore divide by a
    # number five times too large and let this prediction pass for a reason that
    # has nothing to do with the renderer.  So several steps are taken and the
    # denominator is the median of the ones that did NOT refactor -- the
    # smallest honest step, which is the hardest test of the claim.
    apply_s = r.apply_seconds(flow.U, repeats=50)
    steps = []
    for _ in range(STEP_SAMPLES):
        t0 = time.perf_counter()
        rec = flow.step()                    # step() RETURNS its record; it keeps no log
        wall = time.perf_counter() - t0
        steps.append({"wall_s": wall,
                      "solver_step_s": float(rec.get("step_s", 0.0) or 0.0),
                      "ilu_s": float(rec.get("ilu_s", 0.0) or 0.0),
                      "iterations": rec.get("iterations")})
    plain = [s["wall_s"] for s in steps if s["ilu_s"] == 0.0]
    step_s = float(np.median(plain)) if plain else float(min(s["wall_s"] for s in steps))
    out["cost"] = {
        "apply_s": apply_s, "steps": steps,
        "first_step_s": steps[0]["wall_s"],
        "steps_that_refactored": int(sum(1 for s in steps if s["ilu_s"] > 0.0)),
        "steps_timed": len(steps), "step_s": step_s,
        "step_s_is": ("the median of the steps that did NOT refactor the incomplete LU"
                      if plain else "the fastest step; every one refactored"),
        "ratio": apply_s / step_s,
        "tier64_median_step_s": TIER64_MEDIAN_STEP_S,
        "note": "both halves measured in this process, on this machine, seconds apart"}

    # G7: the PNG
    lo, hi = 0.0, 2.0
    png = CR.field_png(F["speed"], lo, hi)
    out["png"] = _png_facts(png, r, lo, hi, F["speed"])
    p = os.path.join(OUT, "overlay_speed.png")
    os.makedirs(OUT, exist_ok=True)
    T60._retry(lambda: open(p, "wb").write(png))
    out["png"]["file"] = os.path.relpath(p, HERE)

    # G8: the controls at a raster the exploration did not try
    out["controls_480x270"] = res.get("raster", {}).get("by_raster", {}).get(
        "480x270", {}).get("controls")

    res["state"] = out
    say("  t=%.3f  speed %.3f..%.3f  nonfinite %d" % (
        out["state"]["t"], *out["fields"]["speed"], out["nonfinite_drawn"]))
    say("  mask state-independent %s, operator deterministic %s" % (
        out["state_independence"]["mask_identical"],
        all(out["determinism"].values())))
    say("  pixel vs direct probe: max |diff| = %r" % out["pixel_vs_direct"]["max_abs_diff"])
    say("  frame %.5f s / step %.4f s = %.4f%%  (first step %.4f s, %d of %d refactored)" % (
        apply_s, step_s, 100 * out["cost"]["ratio"], out["cost"]["first_step_s"],
        out["cost"]["steps_that_refactored"], out["cost"]["steps_timed"]))
    say("  png %d bytes, car pixels %d of %d masked" % (
        out["png"]["bytes"], out["png"]["car_pixels"], out["png"]["masked"]))
    persist(res)
    return out


def _png_facts(png: bytes, r, lo: float, hi: float, field=None) -> dict:
    from PIL import Image
    import io as _io
    im = np.asarray(Image.open(_io.BytesIO(png)).convert("RGB"))
    car = np.all(im == np.asarray(CR.CAR_RGB, dtype=np.uint8), axis=-1)
    lut = CR._lut("speed")
    lut_has_car = bool(np.any(np.all(lut == np.asarray(CR.CAR_RGB, dtype=np.uint8), axis=-1)))
    return {"bytes": len(png), "shape": list(im.shape), "lo": lo, "hi": hi,
            "car_pixels": int(car.sum()), "masked": int(r.mask.sum()),
            "car_pixels_match_mask": bool(np.array_equal(car, r.mask[::-1])),
            "colour_map_can_make_the_car_colour": lut_has_car,
            "car_colour_margin": CR.car_colour_margin(),
            "car_colour_min_margin": CR.CAR_RGB_MIN_MARGIN,
            # what the top of the scale throws away: the field runs past `hi`,
            # so the fastest pixels are clipped and the picture cannot be read
            # as a measurement above it
            "field_max": None if field is None else float(np.nanmax(field)),
            "clipped_pixels": None if field is None else int(np.nansum(field > hi))}


def judge(res) -> dict:
    v = {}
    R = res.get("raster", {}).get("by_raster")
    if R:
        M = [x["mask_vs_solids"] for x in R.values()]
        # NOT MEASURED stays out of the verdicts entirely; a False here would
        # read as "the mask is not the car", which is a different claim
        if all(m is not None for m in M):
            v["G1"] = all(m["masked_not_inside"] == 0 and m["inside_not_masked"] == 0 for m in M)
        k = "480x270"
        if k in R:
            c = R[k]["controls"]
            base = 4.46e-06                      # the 320 x 180 quadratic, explored
            v["G8"] = bool(c["linear_max_err"] < 1e-12 and c["quadratic_max_err"] > 0.5 * base)
    B = res.get("blank")
    if B:
        v["G2"] = bool(B["background_blank"] >= 3000)
    S = res.get("state", {})
    if S.get("state_independence"):
        v["G3"] = bool(S["state_independence"]["mask_identical"])
    if S.get("determinism"):
        v["G4"] = bool(all(S["determinism"].values()))
    if S.get("pixel_vs_direct"):
        d = S["pixel_vs_direct"]
        v["G5"] = bool(d["max_abs_diff"] == 0.0 and not d["any_missing"])
    if S.get("cost"):
        v["G6"] = bool(S["cost"]["ratio"] < 0.01)
    if S.get("png"):
        p = S["png"]
        # G7 is judged exactly as it was REGISTERED.  Opening the PNG showed the
    # registered form is too weak -- "the map never makes this colour" is
    # satisfied by a colour that looks identical to slow air -- but a
    # pre-registered criterion is not rewritten after seeing the data.  The
    # quantity that actually matters, `car_colour_margin`, is recorded beside
    # the verdict and pinned by a test instead.
    v["G7"] = bool(p["car_pixels"] == p["masked"] and p["car_pixels_match_mask"]
                       and not p["colour_map_can_make_the_car_colour"])
    return v


def stage_summary(res) -> dict:
    verdicts = judge(res)
    res["verdicts"] = verdicts
    res["verdicts_missing"] = sorted({p["id"] for p in PREDICTION} - set(verdicts))
    persist(res)
    for p in PREDICTION:
        say("%-3s %-6s %s" % (p["id"], verdicts.get(p["id"]), p["claim"][:96]))
    if res["verdicts_missing"]:
        say("NOT JUDGED:", res["verdicts_missing"])
    return verdicts


STAGES = ("raster", "blank", "state", "summary")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default=",".join(STAGES))
    args = ap.parse_args(argv)
    configure(args.out)
    res = load()
    if "prediction" not in res:
        res.update(tier=66, prediction=list(PREDICTION),
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
            {"raster": stage_raster, "blank": stage_blank,
             "state": stage_state, "summary": stage_summary}[st](res)
        except Exception:
            err = load()
            err.setdefault("stage_errors", {})[st] = traceback.format_exc()[-3000:]
            persist(err)
            say(f"stage {st} FAILED")
            raise
        res.setdefault("stage_wall_s", {})[st] = time.perf_counter() - t0
        # A stage that has now SUCCEEDED must not leave a failure claim standing
        # beside its own result.  Tier 65's record carries exactly that -- a
        # `stage_errors.power` traceback next to a complete `power` block -- and
        # a reader cannot tell from the record alone which one is current.  The
        # entry is dropped rather than renamed, because a clean rebuild never
        # produces it and a record that cannot be reproduced is not a record;
        # what the failure taught belongs in the log and on the page.
        res.get("stage_errors", {}).pop(st, None)
        if res.get("stage_errors") == {}:
            res.pop("stage_errors", None)
        persist(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
