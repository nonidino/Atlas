"""Tier 67 -- where a learned expert can sit on the body-fitted column.

    python scripts/tier67_car_windows.py --out out/racelab16 --stages territory,owned,window,summary

The requirements decided it (section 4.1's decided row): *the background keeps
rectangular windows, and the learned expert runs only there, because Poseidon-T
accepts nothing but a uniform 128 x 128 grid.*  Nobody had measured what "only
there" costs.  `atlas/cases/car_windows.py` works it out:

  ``territory``  every 128 x 128 placement on the car's background, the greedy
                 tilings under four declared orders, and the ceiling no tiling
                 can pass -- reported against the COMPOSITE's unknowns, not
                 flatteringly against the background's.
  ``owned``      the same, under the stricter test: a hole-free window can still
                 contain INTERP cells, which the background holds but does not
                 decide, and a learned expert must not be given those.
  ``window``     one window taken out and put back: dense, finite, bitwise.
  ``summary``    every registered prediction judged in code.

Nothing here marches, loads a checkpoint, or touches RaceLab's porous column.
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

from atlas.cases import car_windows as CW                               # noqa: E402
from atlas.cases import overset as OV                                   # noqa: E402
import tier60_body_fitted_grids as T60                                  # noqa: E402
import tier62_car_solids as T62                                         # noqa: E402
import tier63_duct_openings as T63                                      # noqa: E402

T62.GEOMETRY = T63.geometry

READ_BEFORE_THIS_RUN = (
    "The requirements' section 4.1 decided row: the background keeps rectangular windows and the "
    "learned expert runs only there, because Poseidon-T accepts nothing but a uniform 128 x 128 "
    "grid.  Its status names the learned background windows and the demo on them as NOT built.",
    "racelab_switch: `adapters.Scaling.length` is the tiling units one 128-cell window spans, and "
    "128 cells at dx = 1/64 is 2.0 length units, with no choice in it.  racelab's own lattice is "
    "240 x 672 at that spacing with WX = WY = 128.",
    "Tier 65: the composite is ONE implicit solve, 12,078 interpolation rows over 40 grid pairs, "
    "so a grid cannot be stepped alone.  Tier 66: the body grids own 5,137 of a 320 x 180 raster's "
    "pixels, the near-wall region and the duct.",
    "Explored before these predictions were written (out/t67_explore.json), on the car's own "
    "composite: the background is 241 x 672 at h = 1/64, so it is 3.766 x 10.500 length units and "
    "a 128-cell window spans 2.000; 9.2% of its cells are holes; 18,109 of 62,130 placements "
    "(29.15%) are free of holes; a greedy row-order tiling finds 4 windows covering 44.6% of the "
    "live background, the union of all placements reaches 85.7% of it, and the body-fitted grids "
    "hold 230,197 of the composite's 377,267 unknowns (61.0%).",
    "Explored on the verification composite only: requiring every cell be a DISC cell rather than "
    "merely non-HOLE removed 198 of 4,943 placements and left the tiling count unchanged at 14, "
    "and the four declared orders gave 14, 15, 15 and 15 windows.",
    "NOT measured before these predictions: any of the OWNED numbers on the car; whether the "
    "car's tiling count depends on the order; the ceiling expressed against the COMPOSITE's "
    "unknowns rather than the background's; and whether a window comes out of the car's composite "
    "dense, finite and bitwise reversible.",
)

PREDICTION = (
    {"id": "H1", "claim": "requiring every cell be OWNED rather than merely non-hole removes at "
                          "least 1,000 of the car's 18,109 admissible placements",
     "why": "the background's 2,153 INTERP cells are the fringe that hugs the car's holes, and a "
            "128 x 128 window that clears a hole by a little still straddles the fringe around it"},
    {"id": "H2", "claim": "the greedy tiling's count depends on the order: the four declared "
                          "orders do not all give the same number of windows on the car",
     "why": "greedy set packing is order-dependent, and it already is on the verification "
            "composite, where row gives 14 and the other three give 15"},
    {"id": "H3", "claim": "the learned expert's CEILING -- the union of every admissible "
                          "placement -- is under 35% of the composite's unknowns",
     "why": "the body-fitted grids hold 61.0% of them and a uniform window cannot accept a "
            "curvilinear grid at all, so the ceiling cannot exceed 39% however the tiling is cut"},
    {"id": "H4", "claim": "a window extracted from the car and written straight back leaves the "
                          "solution vector BITWISE identical, and a changed block changes exactly "
                          "128 x 128 entries and no others",
     "why": "extract and scatter_into index the same unknowns through the same index map"},
    {"id": "H5", "claim": "the summed-area admissibility test agrees with brute force on every "
                          "sampled placement of the car's background",
     "why": "the inclusion-exclusion identity is exact in integers; there is no tolerance in it"},
    {"id": "H6", "claim": "the stricter OWNED test does not collapse the tiling: it still finds at "
                          "least 3 windows on the car",
     "why": "the placements it removes are those hugging the fringe, and a greedy tiling's windows "
            "sit in open background where there is no fringe to straddle"},
    {"id": "H7", "claim": "the OWNED reach is a subset of the LIVE reach, cell for cell",
     "why": "a DISC cell is non-HOLE, so every owned-admissible placement is live-admissible, and "
            "reach is a union over placements"},
    {"id": "H8", "claim": "a window off the car is dense and finite, and every one of its cells is "
                          "a DISC cell under the owned test",
     "why": "admissibility is defined cell by cell, so it is the same statement read back"},
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
    if "ov" not in _CAR:
        t0 = time.perf_counter()
        C = T62._car()
        _CAR.update(ov=C["ov"], solids=C["solids"], build_s=time.perf_counter() - t0)
    return _CAR


def _orders(ov, require):
    out = {}
    for o in CW.ORDERS:
        wins = CW.tile(ov, CW.POSEIDON_CELLS, order=o, require=require)
        out[o] = {"n": len(wins), "windows": [w.__dict__ for w in wins]}
    return out


def stage_territory(res) -> dict:
    C = car()
    ov = C["ov"]
    t = CW.territory(ov, CW.POSEIDON_CELLS, order="row", require="live")
    t["by_order"] = {k: v["n"] for k, v in _orders(ov, "live").items()}
    t["orders_agree"] = len(set(t["by_order"].values())) == 1
    res["live"] = t
    say("  live : %d of %d placements (%.2f%%), tiling %s, reach %.1f%% of live bg, "
        "%.1f%% of the COMPOSITE" % (
            t["placements_hole_free"], t["placements"], 100 * t["placements_hole_free_frac"],
            t["by_order"], 100 * t["reach_frac_of_background_live"],
            100 * t["reach_frac_of_composite"]))
    say("  body grids hold %.1f%% of the composite's unknowns" % (100 * t["body_grid_frac"]))
    persist(res)
    return t


def stage_owned(res) -> dict:
    C = car()
    ov = C["ov"]
    t = CW.territory(ov, CW.POSEIDON_CELLS, order="row", require="owned")
    t["by_order"] = {k: v["n"] for k, v in _orders(ov, "owned").items()}
    t["orders_agree"] = len(set(t["by_order"].values())) == 1
    live = res.get("live", {})
    t["placements_lost_to_owning"] = (live.get("placements_hole_free", 0)
                                      - t["placements_hole_free"])
    # H7: owned reach must be inside live reach
    Rl = CW.reach(ov, CW.POSEIDON_CELLS, "live")
    Ro = CW.reach(ov, CW.POSEIDON_CELLS, "owned")
    t["owned_reach_inside_live_reach"] = bool(np.all(Ro <= Rl))
    t["owned_reach_cells"] = int(Ro.sum())
    t["live_reach_cells"] = int(Rl.sum())
    res["owned"] = t
    say("  owned: %d placements (%d fewer), tiling %s, reach %.1f%% of the COMPOSITE" % (
        t["placements_hole_free"], t["placements_lost_to_owning"], t["by_order"],
        100 * t["reach_frac_of_composite"]))
    say("  owned reach inside live reach:", t["owned_reach_inside_live_reach"])
    persist(res)
    return t


def stage_window(res) -> dict:
    """One window out and back, and the summed-area test against brute force."""
    C = car()
    ov = C["ov"]
    w = CW.POSEIDON_CELLS
    wins = CW.tile(ov, w, order="row", require="owned") or CW.tile(ov, w, order="row")
    if not wins:
        res["window"] = {"skipped": "no admissible window on this composite"}
        persist(res)
        return res["window"]
    win = wins[0]
    rng = np.random.default_rng(67)
    vec = rng.normal(size=ov.n_unknowns)
    block = CW.extract(ov, win, vec)
    back = CW.scatter_into(ov, win, vec, block)
    changed = CW.scatter_into(ov, win, vec, block + 1.0)
    st = ov.status[ov.bg.name][win.slices()]
    out = {
        "window": win.__dict__,
        "block_shape": list(block.shape),
        "finite": bool(np.isfinite(block).all()),
        "round_trip_bitwise": bool(np.array_equal(back, vec)),
        "changed_entries": int((changed != vec).sum()),
        "expected_entries": w * w,
        "all_cells_disc": bool(np.all(st == OV.DISC)),
        "status_counts": {int(k): int(v) for k, v in
                          zip(*np.unique(st, return_counts=True))},
    }
    # H5: the summed-area test against brute force, on a sample
    live = CW.live_mask(ov)
    free = CW.hole_free(ov, w, "live")
    rs = np.random.default_rng(6767)
    J, I = free.shape
    js = rs.integers(0, J, size=400)
    iss = rs.integers(0, I, size=400)
    brute = np.array([bool(live[j:j + w, i:i + w].all()) for j, i in zip(js, iss)])
    out["brute_force"] = {"sampled": int(js.size),
                          "agree": int((brute == free[js, iss]).sum()),
                          "disagree": int((brute != free[js, iss]).sum())}
    res["window"] = out
    say("  window (j=%d,i=%d) x %.3f..%.3f y %.3f..%.3f: dense %s, bitwise %s, "
        "changed %d of %d, all DISC %s" % (
            win.j, win.i, win.x0, win.x1, win.y0, win.y1, out["finite"],
            out["round_trip_bitwise"], out["changed_entries"], out["expected_entries"],
            out["all_cells_disc"]))
    say("  summed-area vs brute force on %d placements: %d agree, %d disagree" % (
        out["brute_force"]["sampled"], out["brute_force"]["agree"],
        out["brute_force"]["disagree"]))
    persist(res)
    return out


def judge(res) -> dict:
    v = {}
    L = res.get("live")
    O = res.get("owned")
    W = res.get("window")
    if L and O:
        v["H1"] = bool(O["placements_lost_to_owning"] >= 1000)
    if L:
        v["H2"] = bool(not L["orders_agree"])
        v["H3"] = bool(L["reach_frac_of_composite"] < 0.35)
    if O:
        v["H6"] = bool(min(O["by_order"].values()) >= 3)
        v["H7"] = bool(O["owned_reach_inside_live_reach"])
    if W and "skipped" not in W:
        v["H4"] = bool(W["round_trip_bitwise"]
                       and W["changed_entries"] == W["expected_entries"])
        v["H5"] = bool(W["brute_force"]["disagree"] == 0)
        v["H8"] = bool(W["finite"] and W["all_cells_disc"]
                       and W["block_shape"] == [CW.POSEIDON_CELLS, CW.POSEIDON_CELLS])
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


STAGES = ("territory", "owned", "window", "summary")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default=",".join(STAGES))
    args = ap.parse_args(argv)
    configure(args.out)
    res = load()
    if "prediction" not in res:
        res.update(tier=67, prediction=list(PREDICTION),
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
            {"territory": stage_territory, "owned": stage_owned,
             "window": stage_window, "summary": stage_summary}[st](res)
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
