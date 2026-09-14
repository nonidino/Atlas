"""Tier 58 -- the drawn car's two faults, and candidate fixes measured on copies.

    python scripts/tier58_car_fixes.py --out out/racelab6
    python scripts/tier58_car_fixes.py --out out/racelab6 --stages arms --arms long_box

Tier 56 (``out/racelab5``, [[poc3-racelab-drawn-car]]) could not run the arms on the
car the user drew, for two reasons it named and did not repair:

  W244  the duct flow falls a sixth over 600 macro-steps, so no constant machine
        sizing stays inside the disk's induction clamps; a 200-step ablation that
        started from the intact car's settled field pointed at the closed rear box.
  W245  the window expert's cell-Reynolds bound is breached from macro-step 56, on
        the OUTFLOW column, and not beside any body.

The user chose to measure fixes before redrawing anything.  **The drawing is not
touched.**  Every arm is a copy -- a geometry dict with plates removed, or a box
longer behind the car -- and every arm is spun up from the freestream in its own
right, so no arm inherits the intact car's wake, which was Tier 56 section 4.2's
objection to its own ablation.

**Three things were read out of Tier 56's record before this file was written, and
they shaped it.**  The record carries them under ``read_before_this_run`` rather
than presenting them as found here.

  1. From macro-step 150 on, every over-bound cell in Tier 56's five saved
     snapshots is on the outflow column alone.  So every probe here also records,
     every macro-step, the largest speed WITHOUT the outflow column and without the
     last eight columns.  The model's predicate is not changed and is read exactly
     as it is; the extra readings say WHERE the state it declines is.
  2. At the arms' sizing (Tier 56's probe 2) the rotor's induction sat on a clamp on
     9 of 600 macro-steps, all at the start, and most of the 553 steps that probe
     counted "outside" were the fluid's.  So every probe here records each expert's
     declines per macro-step, not the first decline of any of them.
  3. The window layout is DERIVED from the bodies, and removing the rear box, or the
     endplate alone, or lengthening the box, moves both device planes 14 cells
     downstream (the core's 280 -> 294, the turbine's 392 -> 406).  A fix would
     therefore change the layout as well as the car, and a control that varies two
     things is not one: arm ``drawn_moved_layout`` is the intact car on the moved
     layout, and every fix is compared with it.

Arms, in the order they run (`ARM_ORDER`):

  drawn               the control.  Its spin-up must reproduce
                      ``out/racelab5/cache/settled.npz`` BITWISE, its release-sizing
                      probe Tier 56's stage-trace rows bitwise over all 600 steps,
                      and the first 60 steps of its minimum-sizing probe Tier 56's
                      probe 2.
  drawn_moved_layout  the intact car on the layout the fixes derive: the layout's
                      effect, alone.
  no_rear_box         RW_ENDPLATE, RW_ENDPLATE_LO and DIFF_EXIT removed.
  long_box            the intact car in a box 800 cells long -- 128 more behind it --
                      on the layout derived for that box.
  no_endplate         RW_ENDPLATE alone removed.

Per arm: ``spinup`` (240 macro-steps from the freestream with the declared machine,
`racelab.settled_field` with the arm's own tiling and bodies -- Tier 56's recipe),
``probe1`` (600 macro-steps, lagged and unenforced, sized for the release state),
``probe2`` (600 at probe 1's minimum -- W228's sizing -- whenever probe 1's machine
declined at all; the control marches 60).  Stage ``band`` measures the machine's
admissible inflow window and checks it against Tier 56's recorded flags; stage
``layouts`` derives every arm's windows; stage ``summary`` sets the arms side by
side and judges every registered prediction in code.

**Nothing here changes a threshold, a clamp, a bound, the sizing rule or the car.**
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import copy                                                             # noqa: E402
import datetime as dt                                                   # noqa: E402
import json                                                             # noqa: E402
import sys                                                              # noqa: E402
import time                                                             # noqa: E402
import traceback                                                        # noqa: E402
from collections import Counter                                         # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(HERE, "scripts")
for _p in (HERE, SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                      # noqa: E402
import torch                                                            # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert torch.backends.cuda.matmul.allow_tf32 is False
assert torch.backends.cudnn.allow_tf32 is False

from atlas.cases import powertrain as PT                                # noqa: E402
from atlas.cases import racelab as RL                                   # noqa: E402
from atlas.cases import vehicle_march as VM                             # noqa: E402
from atlas.cases import wing_fsi as W                                   # noqa: E402

#: Tier 56's driver, imported for its record I/O, its stall guard and its
#: summariser of where a cell is -- imported rather than retyped, so the two
#: tiers cannot disagree about what "the outflow column" means.
import tier54_traced_car as T54                                         # noqa: E402

TIER56_JSON = os.path.join(HERE, "out", "racelab5", "racelab5.json")
TIER56_SETTLED = os.path.join(HERE, "out", "racelab5", "cache", "settled.npz")

HORIZON = T54.HORIZON                   # 600, CS-19's
N_SPIN = RL.N_SPIN                      # 240, Tier 56's
CONTROL_PROBE2_STEPS = 60
THREADS = T54.THREADS                   # W227: bitwise the one-thread answer
SNAPS = (0, 150, 300, 450, 599)
#: Tier 56's snapshots show a sawtooth in v over the last six columns beside the
#: spike on the last one; eight is `ground_effect.BAND`, the inflow band's width.
EXCLUDE_LAST = 8
#: The aerodynamic loads are averaged over the last quarter of probe 1.
TAIL = 150
#: A rotor that declines only in the first few macro-steps after release is a
#: release transient; this is where "after the start" begins for that count.
AFTER_THE_START = 24

REAR_BOX = ("RW_ENDPLATE", "RW_ENDPLATE_LO", "DIFF_EXIT")

ARMS = {
    "drawn": {
        "drop": (), "nx": RL.RNX, "layout": "derived",
        "role": "the control: the car as drawn, on its own derived layout -- "
                "Tier 56's car, re-spun through this file's path"},
    "drawn_moved_layout": {
        "drop": (), "nx": RL.RNX, "layout": "from:no_rear_box",
        "role": "the car as drawn on the layout the fixes derive: what moving "
                "the device planes does, alone"},
    "no_rear_box": {
        "drop": REAR_BOX, "nx": RL.RNX, "layout": "derived",
        "role": "the closed rear box removed -- the endplate, its lower edge and "
                "the diffuser exit"},
    "long_box": {
        "drop": (), "nx": 800, "layout": "derived",
        "role": "the car as drawn in a box 128 cells longer behind it"},
    "no_endplate": {
        "drop": ("RW_ENDPLATE",), "nx": RL.RNX, "layout": "derived",
        "role": "the rear wing's endplate alone removed"},
}
ARM_ORDER = ("drawn", "drawn_moved_layout", "no_rear_box", "long_box",
             "no_endplate")
#: Each arm is judged against the arm that differs from it in one thing.
REFERENCE = {"drawn_moved_layout": "drawn", "no_rear_box": "drawn_moved_layout",
             "long_box": "drawn_moved_layout",
             "no_endplate": "drawn_moved_layout"}

#: **Read out of Tier 56's record before this file was written.**
READ_BEFORE_THIS_RUN = (
    "Tier 56's five saved snapshots (out/racelab5/cache/trace_fields.npz): no "
    "cell over the cell-Reynolds bound at macro-step 0; 51, 81, 100 and 113 at "
    "macro-steps 150, 300, 450 and 599, and every one of them on the outflow "
    "column x = 671, where v is -1.99, -2.55, -2.86 and -3.00 against about -0.3 "
    "one to two cells upstream. Tier 56 section 4 already placed the fastest "
    "cell there; this sharpens it to every over-bound cell.",
    "Tier 56's probe 2 (sized for the minimum, 0.442762): the rotor's induction "
    "was on a clamp on 9 of 600 macro-steps, all with the inflow ratio above "
    "1.141 at the start; most of the 553 macro-steps it counted outside were "
    "the fluid's. Its section 3 reads the probe by its first decline.",
    "Derived layouts, computed before any march: the drawn car (0, 112, 160, "
    "272, 384, 496, 544), device planes 280 and 392; without the rear box, and "
    "without the endplate alone, (0, 112, 174, 286, 398, 496, 544), planes 294 "
    "and 406; the drawn car in an 800-cell box (0, 112, 174, 286, 398, 510, "
    "560, 672), planes 294 and 406.",
    "The machine's inflow window from a uniform ring at the W222 similarity, on "
    "a 0.005 grid: inside for u_rotor / sizing in [0.975, 1.140], a span of "
    "1.169; Tier 56's probe 1 inflow spans 0.5311 / 0.4428 = 1.20.",
)

#: **The prediction, registered 2026-09-13 before any stage of this file ran.**
#: Every entry is judged in code by `judge`, under the same id.
PREDICTION = (
    ("C1", "CONTROL. The drawn car re-spun through this file reproduces "
           "out/racelab5/cache/settled.npz bitwise; its release-sizing probe "
           "reproduces Tier 56's stage-trace rows bitwise over all 600 "
           "macro-steps; the first 60 macro-steps of its minimum-sizing probe "
           "reproduce Tier 56's probe 2 bitwise; and its derived layout is "
           "racelab.layout(). If any of these fails, no arm is comparable with "
           "Tier 56, and the arms remain comparable with each other."),
    ("F1", "W245 IS THE OUTFLOW COLUMN, in every arm. In each arm's probe 1: "
           "while FLUID declines, the fastest cell is on that arm's outflow "
           "column on at least 95% of the declined macro-steps; with the "
           "outflow column excluded the speed is inside the bound on at least "
           "90% of the 600 macro-steps; with the last 8 columns excluded, on "
           "all 600."),
    ("F2", "Changing the rear of the car does not move the breach: the first "
           "FLUID decline of no_rear_box and of no_endplate is within 15 "
           "macro-steps of drawn_moved_layout's."),
    ("F3", "The long box delays the breach and does not remove the mechanism: "
           "long_box's first FLUID decline is at least 60 macro-steps later "
           "than drawn_moved_layout's (the wake needs about 160 more macro-steps "
           "at the free stream to cross 128 more cells), and while it declines "
           "the fastest cell is on ITS outflow column, x = 799, on at least 95% "
           "of the declined macro-steps."),
    ("M1", "The layout alone: moving the device planes changes u_rotor at the "
           "release state by more than 1% (direction not predicted), and the "
           "fall of u_rotor over probe 1 stays within 4 percentage points of "
           "the drawn car's -- the fall belongs to the car, not to where the "
           "ring is read."),
    ("M2", "W244 IS THE REAR BOX. no_rear_box's fall of u_rotor over probe 1 is "
           "under half of drawn_moved_layout's, and its first ROTOR-or-MGU "
           "decline at the release sizing is later than macro-step 100, or "
           "absent."),
    ("M3", "The endplate alone lies between: no_endplate's fall, and its first "
           "machine decline at the release sizing, both lie between "
           "no_rear_box's and drawn_moved_layout's."),
    ("M4", "The long box does not help the machine: long_box's fall is within 4 "
           "percentage points of drawn_moved_layout's, and its first machine "
           "decline at the release sizing within 20 macro-steps of it."),
    ("W1", "The uniform-ring window agrees with the model's own per-step ROTOR "
           "and MGU verdicts on at least 98% of the macro-steps Tier 56 "
           "recorded, at both sizings."),
    ("A1", "Removing the rear box lowers the drag: the magnitude of "
           "no_rear_box's mean drag over probe 1's last 150 macro-steps is "
           "smaller than drawn_moved_layout's."),
    ("K1", "Cost: long_box's probe-1 seconds per macro-step are 1.10 to 1.35 "
           "times drawn_moved_layout's, and no_rear_box's within 5% of it."),
)


# ---------------------------------------------------------------------------
# the arms: copies of the car, and the layouts they derive
# ---------------------------------------------------------------------------


def arm_geometry(name: str) -> dict:
    """The arm's car as a geometry dict.  A DEEP copy: `load_geometry` returns
    the cached dict every other caller reads."""
    spec = ARMS[name]
    doc = copy.deepcopy(RL.load_geometry())
    ids = {e["id"] for e in doc["plates"]}
    missing = set(spec["drop"]) - ids
    if missing:
        raise SystemExit("arm %s drops plates the car does not have: %s"
                         % (name, sorted(missing)))
    doc["plates"] = [e for e in doc["plates"] if e["id"] not in set(spec["drop"])]
    return doc


def arm_tiling(name: str):
    """``(tiling, info, doc, flat)`` for an arm."""
    spec = ARMS[name]
    doc = arm_geometry(name)
    _objs, flat = RL.car_bodies(geometry=doc)
    how = spec["layout"]
    if how == "derived":
        t, info = RL.windows_from_geometry(flat, nx=int(spec["nx"]))
        info = dict(info)
        info["layout_source"] = "derived from this arm's own bodies"
    elif how.startswith("from:"):
        other = how.split(":", 1)[1]
        t_o, _info_o, _doc_o, _flat_o = arm_tiling(other)
        if int(t_o.nx) != int(spec["nx"]):
            raise SystemExit("arm %s borrows %s's layout for a box of another "
                             "length" % (name, other))
        t = RL.RaceTiling.of(t_o.cols, t_o.rows, nx=t_o.nx, ny=t_o.ny,
                             wx=t_o.wx, wy=t_o.wy)
        info = RL.layout_report(t, flat)
        info["layout_source"] = ("derived from arm %s's bodies and applied to "
                                 "this arm's" % other)
    else:                                                    # pragma: no cover
        raise SystemExit("unknown layout rule %r" % how)
    if not t.covers():
        raise SystemExit("arm %s's layout does not cover its box" % name)
    return t, info, doc, flat


def arm_identity(name: str, t, doc: dict, n_spin: int) -> dict:
    """What a settled field for this arm IS: the car, the box and the windows."""
    return {"arm": name, "dropped": sorted(ARMS[name]["drop"]),
            "geometry_fingerprint": RL.geometry_fingerprint(geometry=doc),
            "nx": int(t.nx), "ny": int(t.ny),
            "cols": [int(c) for c in t.cols], "rows": [int(r) for r in t.rows],
            "n_spin": int(n_spin)}


def _cache(name: str) -> str:
    return os.path.join(T54.CACHE, name)


def save_arm_field(name: str, ident: dict, u, v) -> str:
    return T54.save_field(name + "_settled", u=u, v=v,
                          identity=np.array(json.dumps(ident, sort_keys=True)))


def load_arm_field(name: str, ident: dict):
    """The arm's settled field -- refused if it was settled for anything else."""
    path = _cache(name + "_settled.npz")
    if not os.path.isfile(path):
        raise RuntimeError("no settled field for arm %s at %s; its spinup has "
                           "not run" % (name, T54._rel(path)))
    d = np.load(path)
    have = str(d["identity"]) if "identity" in d.files else None
    want = json.dumps(ident, sort_keys=True)
    if have != want:
        raise RuntimeError(
            "the settled field at %s was settled for a DIFFERENT arm, car, box or "
            "layout (%s) than arm %s is now (%s); re-run its spinup"
            % (T54._rel(path), (have or "no identity recorded")[:160], name,
               want[:160]))
    return d["u"], d["v"]


def write_json(path: str, obj) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"

    def write():
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(T54.clean(obj), fh)

    T54._retry(write)
    T54._retry(lambda: os.replace(tmp, path))
    return path


# ---------------------------------------------------------------------------
# the march, read every macro-step
# ---------------------------------------------------------------------------


def fluid_breach(value: float, h: float, nu: float) -> bool:
    """`RaceRollout.validity_report`'s FLUID predicate on one speed, verbatim:
    declined when the cell Reynolds number exceeds 8 or the speed the band."""
    return not (h * value / nu <= 8.0 and value <= RL.U_MAX_BAND)


def arm_march(u0, v0, host_inflow: float, steps: int, tiling, doc: dict,
              label: str, snaps=()):
    """Tier 56's `traced_march`, on an arm's own tiling and bodies, reading three
    more things after every macro-step: the largest speed without the outflow
    column, without the last `EXCLUDE_LAST` columns, and on the outflow column.

    The state is advanced exactly as `tier54_traced_car.traced_march` advances
    it -- the extra readings are taken from the field after the step and write
    nothing back -- and the control asserts that rather than this docstring.
    Fresh bodies for every march, so no march can inherit another's objects.
    """
    objs, _flat = RL.car_bodies(geometry=doc)
    r = RL.RaceRollout(tiling=tiling, objects=objs, join_coupling="lagged",
                       enforce=False, host_inflow=host_inflow)
    opt = dict(dtype=W.TORCH_DTYPE, device=r.device)
    u = torch.as_tensor(np.asarray(u0), **opt)
    v = torch.as_tensor(np.asarray(v0), **opt)
    r.refresh(u, 0, which=("J1", "J3"))
    flat = T54._flat_of(r.objects)
    cb = T54.watch(label, steps, every=50)
    rows, saved = [], {}
    t0 = time.perf_counter()
    for s in range(steps):
        with torch.no_grad():
            u, v, load, drag = r.macro_step(u, v, s)
            sp = torch.hypot(u, v)
            k = int(torch.argmax(sp))
            no_col = float(torch.max(sp[:, :-1]))
            no_band = float(torch.max(sp[:, :-EXCLUDE_LAST]))
            col = float(torch.max(sp[:, -1]))
            col_v = float(torch.max(torch.abs(v[:, -1])))
        iy, ix = divmod(k, r.nx)
        nb, dist = T54._nearest_body(flat, ix + 0.5, iy + 0.5)
        env = r.envelope or {}
        rows.append({
            "step": s, "u_max": float(sp.reshape(-1)[k]),
            "at_cell": [ix, iy], "where": T54._where(flat, ix, iy, r.nx, r.ny),
            "nearest_body": nb, "distance_cells": dist,
            "cell_reynolds": env.get("FLUID", {}).get("cell_reynolds"),
            "declined": list((r.outside or {}).get("declined", [])),
            "u_rotor": float(r.state.u_rotor),
            "induction": float(r.state.induction),
            "current": float(r.state.current),
            "omega": float(r.state.omega),
            "load": float(load), "drag": float(drag),
            "u_max_without_the_outflow_column": no_col,
            "u_max_without_the_last_%d_columns" % EXCLUDE_LAST: no_band,
            "outflow_column_u_max": col,
            "outflow_column_abs_v_max": col_v})
        if s in snaps:
            saved["u_%03d" % s] = u.detach().cpu().numpy().copy()
            saved["v_%03d" % s] = v.detach().cpu().numpy().copy()
        cb(s, 0.0)
    wall = time.perf_counter() - t0
    meta = {"wall_s": wall, "s_per_macro_step": wall / max(steps, 1),
            "nx": int(r.nx), "ny": int(r.ny), "h": float(r.solver.h),
            "nu": float(r.nu), "host_inflow": host_inflow, "steps": steps,
            "torch_threads": torch.get_num_threads()}
    return rows, saved, meta


def _runs(steps_list) -> tuple[int, int]:
    """The longest run of consecutive integers in a sorted list, and its start."""
    best, best_at, cur, cur_at, prev = 0, None, 0, None, None
    for s in steps_list:
        if prev is not None and s == prev + 1:
            cur += 1
        else:
            cur, cur_at = 1, s
        if cur > best:
            best, best_at = cur, cur_at
        prev = s
    return best, best_at


def summarise(rows, meta: dict, window: dict | None) -> dict:
    """Everything the summary and the page read, from one probe's rows."""
    n = len(rows)
    nx, h, nu = meta["nx"], meta["h"], meta["nu"]
    host = meta["host_inflow"]

    def declined(tag):
        return [r_["step"] for r_ in rows if tag in r_["declined"]]

    def block(steps_):
        return {"first": steps_[0] if steps_ else None,
                "last": steps_[-1] if steps_ else None,
                "declined_steps": len(steps_)}

    fl, ro, mg = declined("FLUID"), declined("ROTOR"), declined("MGU")
    machine = sorted(set(ro) | set(mg))
    inside = [r_["step"] for r_ in rows
              if not ({"ROTOR", "MGU"} & set(r_["declined"]))]
    run, run_at = _runs(inside)
    ur = np.array([r_["u_rotor"] for r_ in rows], dtype=float)
    um = np.array([r_["u_max"] for r_ in rows], dtype=float)
    key8 = "u_max_without_the_last_%d_columns" % EXCLUDE_LAST
    nc = np.array([r_["u_max_without_the_outflow_column"] for r_ in rows])
    n8 = np.array([r_[key8] for r_ in rows])
    cv = np.array([r_["outflow_column_abs_v_max"] for r_ in rows])
    load = np.array([r_["load"] for r_ in rows])
    drag = np.array([r_["drag"] for r_ in rows])
    fl_rows = [r_ for r_ in rows if "FLUID" in r_["declined"]]
    on_col = sum(1 for r_ in fl_rows if r_["at_cell"][0] == nx - 1)

    def over(arr):
        idx = [i for i, x in enumerate(arr) if fluid_breach(float(x), h, nu)]
        return {"max": float(arr.max()), "argmax": int(arr.argmax()),
                "first_over_the_bound": idx[0] if idx else None,
                "steps_over_the_bound": len(idx),
                "steps_inside_the_bound": n - len(idx),
                "every_50": [float(arr[i]) for i in range(0, n, 50)]}

    out = {
        "steps": n, "host_inflow": host,
        "wall_s": meta["wall_s"], "s_per_macro_step": meta["s_per_macro_step"],
        "FLUID": block(fl), "ROTOR": block(ro), "MGU": block(mg),
        "machine": dict(block(machine), **{
            "declined_at_or_after_step_%d" % AFTER_THE_START:
                sum(1 for s in machine if s >= AFTER_THE_START),
            "longest_inside_run": run, "longest_inside_run_starts_at": run_at}),
        "u_rotor": {
            "first": float(ur[0]), "last": float(ur[-1]),
            "min": float(ur.min()), "argmin": int(ur.argmin()),
            "max": float(ur.max()), "argmax": int(ur.argmax()),
            "fall_fraction": float((ur.max() - ur.min()) / ur.max()),
            "span": float(ur.max() / ur.min()),
            "ratio_to_the_sizing_min": float(ur.min() / host),
            "ratio_to_the_sizing_max": float(ur.max() / host),
            "every_50": [float(ur[i]) for i in range(0, n, 50)]},
        "fluid": {
            "bound_speed": 8.0 * nu / h, "h": h, "nu": nu,
            "u_max": over(um),
            "without_the_outflow_column": over(nc),
            "without_the_last_%d_columns" % EXCLUDE_LAST: over(n8),
            "outflow_column_abs_v_max_every_50":
                [float(cv[i]) for i in range(0, n, 50)],
            "fastest_cell_on_the_outflow_column_while_FLUID_declines":
                (on_col / len(fl_rows)) if fl_rows else None,
            "where_while_FLUID_declines": dict(Counter(
                r_["where"] for r_ in fl_rows).most_common(8)),
            "where_over_the_whole_probe": dict(Counter(
                r_["where"] for r_ in rows).most_common(8)),
        },
        "aero": {"load_first": float(load[0]), "drag_first": float(drag[0]),
                 "load_mean_last_%d" % TAIL: float(load[-TAIL:].mean()),
                 "drag_mean_last_%d" % TAIL: float(drag[-TAIL:].mean())},
    }
    if window and window.get("machine_inside"):
        lo, hi = window["machine_inside"]
        ratio = ur / host
        pred = [bool(lo <= x <= hi) for x in ratio]
        meas = [not ({"ROTOR", "MGU"} & set(r_["declined"])) for r_ in rows]
        out["window_check"] = {
            "window": [lo, hi],
            "steps_the_window_predicts_inside": int(sum(pred)),
            "steps_measured_inside": int(sum(meas)),
            "steps_where_they_disagree": int(sum(p != m for p, m in
                                                 zip(pred, meas))),
            "fits_one_sizing": bool(ur.max() / ur.min() <= hi / lo),
            "note": "a uniform ring's window read against this probe's inflow; "
                    "it ignores the thrust's feedback on the inflow, so it is a "
                    "diagnostic of the measured verdicts, not a substitute"}
    return out


def compact(rows) -> dict:
    """The whole trajectory of the numbers the page reads, small enough to
    commit.  Every row in full goes to the cache beside it."""
    key8 = "u_max_without_the_last_%d_columns" % EXCLUDE_LAST
    return {
        "u_rotor": [r_["u_rotor"] for r_ in rows],
        "u_max": [r_["u_max"] for r_ in rows],
        "u_max_without_the_outflow_column":
            [r_["u_max_without_the_outflow_column"] for r_ in rows],
        key8: [r_[key8] for r_ in rows],
        "induction": [r_["induction"] for r_ in rows],
        "drag": [r_["drag"] for r_ in rows],
        "load": [r_["load"] for r_ in rows],
        "declined": ["".join(t[0] for t in ("FLUID", "ROTOR", "MGU")
                             if t in r_["declined"]) for r_ in rows],
        "declined_legend": "F FLUID, R ROTOR, M MGU",
    }


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def _machine_inside(ratio: float, host: float) -> tuple[bool, bool, float]:
    els = RL.machine_for_host(host, scale=RL.HOST_ROTOR_WIDTH)
    ring = np.full(int(RL.DEVICE_CELLS), float(ratio) * host)
    res, e2, _disk = VM.operating_point(ring, width=RL.HOST_ROTOR_WIDTH,
                                        scale=RL.HOST_ROTOR_WIDTH, elements=els)
    return (bool(res.rotor_valid),
            bool(e2["MGU"].validity(np.full(1, res.omega))),
            float(res.induction))


def _edge(pred, lo: float, hi: float, iters: int = 48) -> float:
    """The ratio where ``pred`` flips between ``lo`` and ``hi``."""
    a, b = float(lo), float(hi)
    fa = pred(a)
    if fa == pred(b):                                        # pragma: no cover
        raise RuntimeError("no flip in [%r, %r]" % (a, b))
    for _ in range(iters):
        m = 0.5 * (a + b)
        if pred(m) == fa:
            a = m
        else:
            b = m
    return 0.5 * (a + b)


def stage_band() -> dict:
    """The machine's admissible inflow window, and a check of it against every
    per-step verdict Tier 56 recorded."""
    out: dict = {"ring": "uniform, DEVICE_CELLS cells", "hosts": {}}
    for host in (1.0, 0.5317254889754462, 0.44276205468922647):
        rot_lo = _edge(lambda x: _machine_inside(x, host)[0], 0.90, 1.0)
        rot_hi = _edge(lambda x: _machine_inside(x, host)[0], 1.0, 1.30)
        mgu_lo = _edge(lambda x: _machine_inside(x, host)[1], 0.90, 1.0)
        out["hosts"][repr(host)] = {"rotor_inside": [rot_lo, rot_hi],
                                    "mgu_inside_above": mgu_lo}
    base = out["hosts"][repr(1.0)]
    spread = max(abs(v["rotor_inside"][0] - base["rotor_inside"][0])
                 + abs(v["rotor_inside"][1] - base["rotor_inside"][1])
                 + abs(v["mgu_inside_above"] - base["mgu_inside_above"])
                 for v in out["hosts"].values())
    lo = max(base["rotor_inside"][0], base["mgu_inside_above"])
    hi = base["rotor_inside"][1]
    out.update({"machine_inside": [lo, hi], "span": hi / lo,
                "rotor_inside": base["rotor_inside"],
                "mgu_inside_above": base["mgu_inside_above"],
                "similarity_spread_across_hosts": spread})
    print("   window [%.6f, %.6f], span %.4f; similarity spread %.2e"
          % (lo, hi, hi / lo, spread), flush=True)

    # -- the check: the model's own verdicts, recorded by Tier 56
    with open(TIER56_JSON, encoding="utf-8") as fh:
        t56 = json.load(fh)
    rows = t56["trace"]["rows"]
    host = float(t56["trace"]["host_inflow"])
    rot_lo, rot_hi = base["rotor_inside"]
    mgu_lo = base["mgu_inside_above"]
    dis_r = [r_["step"] for r_ in rows
             if (rot_lo <= r_["u_rotor"] / host <= rot_hi)
             != ("ROTOR" not in r_["declined"])]
    dis_m = [r_["step"] for r_ in rows
             if (r_["u_rotor"] / host >= mgu_lo) != ("MGU" not in r_["declined"])]
    p2 = t56["size"]["probe_at_the_arms_sizing"]
    host2 = float(p2["host_inflow"])
    els = RL.machine_for_host(host2, scale=RL.HOST_ROTOR_WIDTH)
    tr = p2["trace"]
    rot_act = [bool(PT.A_MIN < a < PT.A_MAX) for a in tr["induction"]]
    mgu_act = [bool(els["MGU"].validity(np.full(1, w))) for w in tr["omega"]]
    ratio2 = [x / host2 for x in tr["u_rotor"]]
    dis_r2 = [i for i, x in enumerate(ratio2)
              if (rot_lo <= x <= rot_hi) != rot_act[i]]
    dis_m2 = [i for i, x in enumerate(ratio2) if (x >= mgu_lo) != mgu_act[i]]
    n1, n2 = len(rows), len(ratio2)
    out["check_against_tier56"] = {
        "release_sizing": {
            "source": "out/racelab5/racelab5.json trace.rows -- the model's own "
                      "declined lists", "host_inflow": host, "steps": n1,
            "rotor_disagreements": dis_r, "mgu_disagreements": dis_m,
            "agreement": 1.0 - (len(set(dis_r) | set(dis_m)) / n1)},
        "arms_sizing": {
            "source": "out/racelab5/racelab5.json size.probe_at_the_arms_sizing "
                      "-- ROTOR read as the induction strictly inside the "
                      "disk's clamps, MGU as the machine's own validity on the "
                      "recorded shaft speed", "host_inflow": host2, "steps": n2,
            "rotor_disagreements": dis_r2, "mgu_disagreements": dis_m2,
            "rotor_declined_steps": [i for i, ok in enumerate(rot_act) if not ok],
            "mgu_declined_steps": [i for i, ok in enumerate(mgu_act) if not ok],
            "agreement": 1.0 - (len(set(dis_r2) | set(dis_m2)) / n2)},
    }
    print("   agreement with Tier 56: %.4f at the release sizing, %.4f at the "
          "arms' sizing; rotor declined at the arms' sizing on %d steps"
          % (out["check_against_tier56"]["release_sizing"]["agreement"],
             out["check_against_tier56"]["arms_sizing"]["agreement"],
             len(out["check_against_tier56"]["arms_sizing"]["rotor_declined_steps"])),
          flush=True)
    return out


def stage_layouts() -> dict:
    out: dict = {}
    t_default, _i = RL.layout()
    cols = {}
    for name in ARM_ORDER:
        t, info, doc, flat = arm_tiling(name)
        spec = ARMS[name]
        removed = [{"id": e["id"], "x": e["x"], "y": e["y"], "chord": e["chord"],
                    "alpha": e["alpha"]}
                   for e in RL.load_geometry()["plates"] if e["id"] in spec["drop"]]
        cols[name] = tuple(t.cols)
        out[name] = {
            "role": spec["role"], "dropped": list(spec["drop"]), "removed": removed,
            "flat_bodies": len(flat),
            "geometry_fingerprint": RL.geometry_fingerprint(geometry=doc),
            "layout": {k: info.get(k) for k in (
                "layout_source", "cols", "rows", "nx", "ny", "n_windows",
                "x_bands", "overlaps", "device_planes", "banded_force_fraction",
                "bodies_cut", "min_cut_to_body_clearance_cells",
                "y_cut_to_body_clearance_cells", "covers")},
            "equals_racelab_layout": (tuple(t.cols) == tuple(t_default.cols)
                                      and tuple(t.rows) == tuple(t_default.rows)
                                      and t.nx == t_default.nx),
        }
        print("   %-19s nx %d cols %s planes %s" % (
            name, t.nx, list(t.cols), info.get("device_planes")), flush=True)
    out["_the_fixes_derive_one_layout"] = (
        cols["no_rear_box"] == cols["no_endplate"])
    out["_cross_costs"] = layout_cross_costs()
    return out


def layout_cross_costs() -> dict:
    """Every 672-cell car's banded occupancy under BOTH 672-cell layouts.

    `windows_from_geometry` minimises the car's occupancy inside the cut bands,
    and the device planes are wherever the minimising layout puts its two narrow
    bands.  So how far apart the two layouts are IN THAT OBJECTIVE, for the car
    as drawn, is how firmly the planes were chosen: a near-tie means the
    machine's ring is placed by a tie-break in a cut-placement objective.
    """
    t_drawn, _i, _d, _f = arm_tiling("drawn")
    t_fix, _i2, _d2, _f2 = arm_tiling("no_rear_box")
    layouts = {"drawn's (planes 280, 392)": t_drawn,
               "the fixes' (planes 294, 406)": t_fix}
    out: dict = {"objective": "the occupancy profile inside every x-band, as a "
                              "fraction of the car's total -- "
                              "`windows_from_geometry`'s own objective",
                 "cars": {}}
    for name in ("drawn", "no_rear_box", "no_endplate"):
        _objs, flat = RL.car_bodies(geometry=arm_geometry(name))
        phi = np.asarray(RL.body_profile(flat, RL.RNX), dtype=float)
        total = float(phi.sum())
        row = {}
        for label, t in layouts.items():
            banded = sum(float(phi[lo:min(hi, RL.RNX)].sum())
                         for lo, hi in t.x_bands())
            row[label] = banded / total
        a, b = row.values()
        row["relative_gap"] = (b - a) / a
        out["cars"][name] = row
    return out


def _control_rows(mine, theirs, keys) -> dict:
    n = min(len(mine), len(theirs))
    equal, first = 0, None
    for i in range(n):
        a, b = mine[i], theirs[i]
        same = all(a[k] == b[k] for k in keys)
        if "declined" in b:
            same = same and list(a["declined"]) == list(b["declined"])
        if "at_cell" in b:
            same = same and list(a["at_cell"]) == list(b["at_cell"])
        if same:
            equal += 1
        elif first is None:
            first = {"step": i, "mine": {k: a.get(k) for k in keys},
                     "tier56": {k: b.get(k) for k in keys}}
    return {"compared_steps": n, "equal_steps": equal,
            "bitwise": bool(n > 0 and equal == n), "first_difference": first,
            "keys": list(keys)}


def run_arm(res: dict, name: str, steps: dict) -> dict:
    """spinup, probe1 and probe2 for one arm, each persisted as it finishes and
    skipped when the record already holds it for the same arm."""
    torch.set_num_threads(THREADS)
    arm = res.setdefault("arms", {}).setdefault(name, {})
    t, info, doc, _flat = arm_tiling(name)
    ident = arm_identity(name, t, doc, steps["spin"])
    if arm.get("identity") not in (None, ident):
        raise RuntimeError("the record's arm %s was measured for %s, and the arm "
                           "is now %s; use a new --out" % (name, arm["identity"],
                                                           ident))
    arm["identity"] = ident
    arm["role"] = ARMS[name]["role"]
    arm.setdefault("machine_state", []).append(
        dict(T54.machine_state(), at=dt.datetime.now().isoformat(timespec="seconds")))
    window = res.get("band")
    is_control = name == "drawn"
    registered = (steps["spin"] == N_SPIN and steps["horizon"] == HORIZON)

    # -- spin-up ------------------------------------------------------------
    if "spinup" not in arm:
        print("   [%s] spin-up, %d macro-steps from the freestream ..."
              % (name, steps["spin"]), flush=True)
        objs, _f = RL.car_bodies(geometry=doc)
        t0 = time.perf_counter()
        u, v, rep = RL.settled_field(
            steps=steps["spin"], host_inflow=None, tiling=t, objects=objs,
            progress=T54.watch("spin-" + name, steps["spin"], every=40))
        wall = time.perf_counter() - t0
        save_arm_field(name, ident, u, v)
        rep = dict(rep)
        rep.update(wall_s=wall, s_per_macro_step=wall / steps["spin"],
                   written_to=T54._rel(_cache(name + "_settled.npz")))
        if is_control:
            if registered and os.path.isfile(TIER56_SETTLED):
                d = np.load(TIER56_SETTLED)
                rep["tier56_settled_fingerprint"] = (
                    str(d["geometry"]) if "geometry" in d.files else None)
                rep["reproduces_tier56_settled_bitwise"] = bool(
                    np.array_equal(u, d["u"]) and np.array_equal(v, d["v"]))
                if not rep["reproduces_tier56_settled_bitwise"]:
                    rep["max_abs_difference"] = float(max(
                        np.max(np.abs(u - d["u"])), np.max(np.abs(v - d["v"]))))
            else:
                rep["reproduces_tier56_settled_bitwise"] = None
                rep["not_compared_because"] = (
                    "not the registered spin-up length" if not registered
                    else "no Tier 56 settled field on this machine")
        arm["spinup"] = rep
        T54.persist(res)
        print("   [%s] u_rotor at release %.6f; %.3f s/step%s"
              % (name, rep["u_rotor_at_the_release_state"],
                 rep["s_per_macro_step"],
                 ("; reproduces Tier 56 bitwise: %s"
                  % rep.get("reproduces_tier56_settled_bitwise"))
                 if is_control else ""), flush=True)
    u0, v0 = load_arm_field(name, ident)
    u_rel = float(arm["spinup"]["u_rotor_at_the_release_state"])

    # -- probe 1: the release state's sizing --------------------------------
    if "probe1" not in arm:
        print("   [%s] probe 1, %d macro-steps at host_inflow=%.6f ..."
              % (name, steps["horizon"], u_rel), flush=True)
        rows, saved, meta = arm_march(u0, v0, u_rel, steps["horizon"], t, doc,
                                      name + "-p1", snaps=SNAPS)
        write_json(_cache(name + "_probe1_rows.json"), rows)
        if saved:
            T54.save_field(name + "_probe1_fields", **saved)
        p1 = {"summary": summarise(rows, meta, window),
              "trajectory": compact(rows),
              "rows_in": T54._rel(_cache(name + "_probe1_rows.json")),
              "snapshots_at": [s for s in SNAPS if s < steps["horizon"]]}
        if is_control:
            if registered:
                with open(TIER56_JSON, encoding="utf-8") as fh:
                    t56 = json.load(fh)
                p1["control"] = _control_rows(
                    rows, t56["trace"]["rows"],
                    ("u_max", "u_rotor", "induction", "current", "load", "drag"))
                p1["control"]["against"] = "out/racelab5/racelab5.json trace.rows"
            else:
                p1["control"] = {"bitwise": None, "not_compared_because":
                                 "not the registered horizon or spin-up"}
        arm["probe1"] = p1
        T54.persist(res)
        s1 = p1["summary"]
        print("   [%s] probe 1: machine first %s (%d steps), fall %.2f%%; FLUID "
              "first %s (%d steps); without the outflow column first over %s"
              % (name, s1["machine"]["first"], s1["machine"]["declined_steps"],
                 100 * s1["u_rotor"]["fall_fraction"], s1["FLUID"]["first"],
                 s1["FLUID"]["declined_steps"],
                 s1["fluid"]["without_the_outflow_column"]["first_over_the_bound"]),
              flush=True)

    # -- probe 2: W228's sizing, the horizon's minimum ----------------------
    if "probe2" not in arm:
        s1 = arm["probe1"]["summary"]
        ud = float(s1["u_rotor"]["min"])
        if s1["machine"]["declined_steps"] == 0:
            arm["probe2"] = {
                "marched": False, "host_inflow": ud,
                "why": "probe 1's machine declined on no macro-step, so the "
                       "release state's sizing already admits the horizon and "
                       "W228's re-sizing is not needed"}
        else:
            n2 = (min(CONTROL_PROBE2_STEPS, steps["horizon"]) if is_control
                  else steps["horizon"])
            print("   [%s] probe 2, %d macro-steps at host_inflow=%.6f (probe 1's "
                  "minimum) ..." % (name, n2, ud), flush=True)
            rows, _saved, meta = arm_march(u0, v0, ud, n2, t, doc, name + "-p2")
            write_json(_cache(name + "_probe2_rows.json"), rows)
            p2 = {"marched": True, "summary": summarise(rows, meta, window),
                  "trajectory": compact(rows),
                  "rows_in": T54._rel(_cache(name + "_probe2_rows.json"))}
            if is_control:
                if registered:
                    with open(TIER56_JSON, encoding="utf-8") as fh:
                        t56 = json.load(fh)
                    tp = t56["size"]["probe_at_the_arms_sizing"]
                    keys = ("u_rotor", "u_max", "induction", "current", "omega",
                            "load", "drag")
                    theirs = [{k: tp["trace"][k][i] for k in keys}
                              for i in range(len(tp["trace"]["u_rotor"]))]
                    p2["control"] = _control_rows(rows, theirs, keys)
                    p2["control"]["against"] = (
                        "out/racelab5/racelab5.json size.probe_at_the_arms_sizing")
                    p2["control"]["sized_for_the_same_value"] = (
                        ud == float(tp["host_inflow"]))
                    #: the control's own 600 are Tier 56's -- read, labelled,
                    #: never presented as marched here
                    p2["tier56_full_horizon_read_not_marched"] = {
                        "steps": len(tp["trace"]["u_rotor"]),
                        "first_outside_macro_step": tp["first_outside_macro_step"],
                        "outside_the_envelope_steps":
                            tp["outside_the_envelope_steps"]}
                else:
                    p2["control"] = {"bitwise": None, "not_compared_because":
                                     "not the registered horizon or spin-up"}
            arm["probe2"] = p2
            s2 = p2["summary"]
            print("   [%s] probe 2: machine declined on %d of %d (first %s, last "
                  "%s); FLUID first %s" % (
                      name, s2["machine"]["declined_steps"], n2,
                      s2["machine"]["first"], s2["machine"]["last"],
                      s2["FLUID"]["first"]), flush=True)
        T54.persist(res)
    return arm


# ---------------------------------------------------------------------------
# the summary, and the prediction judged in code
# ---------------------------------------------------------------------------


def _p(res, name, probe="probe1"):
    return (((res.get("arms") or {}).get(name) or {}).get(probe) or {}).get("summary")


def _within(a, b, tol):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def _between(x, a, b):
    """``x`` between ``a`` and ``b`` inclusive, with None meaning never (+inf)."""
    inf = float("inf")
    x, a, b = [inf if y is None else y for y in (x, a, b)]
    return min(a, b) <= x <= max(a, b)


def judge(res: dict, registered: bool) -> dict:
    """Every entry of `PREDICTION`, judged.  ``None`` means not measured."""
    out: dict = {}
    arms = res.get("arms") or {}

    def fall(n):
        s = _p(res, n)
        return None if s is None else s["u_rotor"]["fall_fraction"]

    def first_machine(n):
        return _p(res, n)["machine"]["first"]

    def first_fluid(n):
        return _p(res, n)["FLUID"]["first"]

    def have(*names):
        return all(_p(res, n) is not None for n in names)

    if not registered:
        return {pid: {"verdict": None, "why": "this run did not march the "
                      "registered lengths, so the prediction is not judged"}
                for pid, _t in PREDICTION}

    # C1
    d = arms.get("drawn") or {}
    flags = {
        "spinup_bitwise": (d.get("spinup") or {}).get(
            "reproduces_tier56_settled_bitwise"),
        "probe1_bitwise": ((d.get("probe1") or {}).get("control") or {}).get("bitwise"),
        "probe2_bitwise": ((d.get("probe2") or {}).get("control") or {}).get("bitwise"),
        "layout_is_racelab_layout": ((res.get("layouts") or {}).get("drawn") or {}).get(
            "equals_racelab_layout"),
    }
    out["C1"] = {"verdict": (None if any(v is None for v in flags.values())
                             else all(flags.values())), "numbers": flags}

    # F1
    per, ok, missing = {}, True, False
    for n in ARM_ORDER:
        s = _p(res, n)
        if s is None:
            missing = True
            continue
        frac = s["fluid"]["fastest_cell_on_the_outflow_column_while_FLUID_declines"]
        wc = s["fluid"]["without_the_outflow_column"]["steps_inside_the_bound"]
        w8 = s["fluid"]["without_the_last_%d_columns" % EXCLUDE_LAST][
            "steps_inside_the_bound"]
        a = (frac is None or frac >= 0.95)
        b = wc >= 0.90 * s["steps"]
        c = w8 == s["steps"]
        per[n] = {"on_the_outflow_column": frac, "inside_without_the_column": wc,
                  "inside_without_the_last_8": w8, "holds": bool(a and b and c)}
        ok = ok and a and b and c
    out["F1"] = {"verdict": None if missing else bool(ok), "numbers": per}

    # F2
    if have("no_rear_box", "no_endplate", "drawn_moved_layout"):
        ref = first_fluid("drawn_moved_layout")
        nums = {"drawn_moved_layout": ref, "no_rear_box": first_fluid("no_rear_box"),
                "no_endplate": first_fluid("no_endplate")}
        out["F2"] = {"verdict": bool(_within(nums["no_rear_box"], ref, 15)
                                     and _within(nums["no_endplate"], ref, 15)),
                     "numbers": nums}
    else:
        out["F2"] = {"verdict": None}

    # F3
    if have("long_box", "drawn_moved_layout"):
        ref = first_fluid("drawn_moved_layout")
        lb = first_fluid("long_box")
        frac = _p(res, "long_box")["fluid"][
            "fastest_cell_on_the_outflow_column_while_FLUID_declines"]
        later = (lb is None) or (ref is not None and lb >= ref + 60)
        out["F3"] = {"verdict": bool(later and (frac is None or frac >= 0.95)),
                     "numbers": {"drawn_moved_layout": ref, "long_box": lb,
                                 "on_its_outflow_column": frac}}
    else:
        out["F3"] = {"verdict": None}

    # M1
    if have("drawn", "drawn_moved_layout"):
        ua = arms["drawn"]["spinup"]["u_rotor_at_the_release_state"]
        ub = arms["drawn_moved_layout"]["spinup"]["u_rotor_at_the_release_state"]
        moved = abs(ub / ua - 1.0)
        dfall = abs(fall("drawn_moved_layout") - fall("drawn"))
        out["M1"] = {"verdict": bool(moved > 0.01 and dfall <= 0.04),
                     "numbers": {"u_rotor_release_drawn": ua,
                                 "u_rotor_release_moved": ub,
                                 "relative_change": ub / ua - 1.0,
                                 "fall_drawn": fall("drawn"),
                                 "fall_moved": fall("drawn_moved_layout")}}
    else:
        out["M1"] = {"verdict": None}

    # M2
    if have("no_rear_box", "drawn_moved_layout"):
        fm = first_machine("no_rear_box")
        out["M2"] = {"verdict": bool(fall("no_rear_box") < 0.5 * fall("drawn_moved_layout")
                                     and (fm is None or fm > 100)),
                     "numbers": {"fall_no_rear_box": fall("no_rear_box"),
                                 "fall_reference": fall("drawn_moved_layout"),
                                 "first_machine_decline_no_rear_box": fm,
                                 "first_machine_decline_reference":
                                     first_machine("drawn_moved_layout")}}
    else:
        out["M2"] = {"verdict": None}

    # M3
    if have("no_endplate", "no_rear_box", "drawn_moved_layout"):
        out["M3"] = {"verdict": bool(
            _between(fall("no_endplate"), fall("no_rear_box"),
                     fall("drawn_moved_layout"))
            and _between(first_machine("no_endplate"), first_machine("no_rear_box"),
                         first_machine("drawn_moved_layout"))),
            "numbers": {n: {"fall": fall(n), "first_machine_decline": first_machine(n)}
                        for n in ("no_rear_box", "no_endplate", "drawn_moved_layout")}}
    else:
        out["M3"] = {"verdict": None}

    # M4
    if have("long_box", "drawn_moved_layout"):
        out["M4"] = {"verdict": bool(
            abs(fall("long_box") - fall("drawn_moved_layout")) <= 0.04
            and _within(first_machine("long_box"),
                        first_machine("drawn_moved_layout"), 20)),
            "numbers": {n: {"fall": fall(n), "first_machine_decline": first_machine(n)}
                        for n in ("long_box", "drawn_moved_layout")}}
    else:
        out["M4"] = {"verdict": None}

    # W1
    b = (res.get("band") or {}).get("check_against_tier56")
    if b:
        a1, a2 = b["release_sizing"]["agreement"], b["arms_sizing"]["agreement"]
        out["W1"] = {"verdict": bool(a1 >= 0.98 and a2 >= 0.98),
                     "numbers": {"release_sizing": a1, "arms_sizing": a2}}
    else:
        out["W1"] = {"verdict": None}

    # A1
    if have("no_rear_box", "drawn_moved_layout"):
        key = "drag_mean_last_%d" % TAIL
        dn = _p(res, "no_rear_box")["aero"][key]
        dr = _p(res, "drawn_moved_layout")["aero"][key]
        out["A1"] = {"verdict": bool(abs(dn) < abs(dr)),
                     "numbers": {"no_rear_box": dn, "drawn_moved_layout": dr}}
    else:
        out["A1"] = {"verdict": None}

    # K1
    if have("long_box", "no_rear_box", "drawn_moved_layout"):
        ref = _p(res, "drawn_moved_layout")["s_per_macro_step"]
        rl = _p(res, "long_box")["s_per_macro_step"] / ref
        rn = _p(res, "no_rear_box")["s_per_macro_step"] / ref
        out["K1"] = {"verdict": bool(1.10 <= rl <= 1.35 and abs(rn - 1.0) <= 0.05),
                     "numbers": {"long_box_over_reference": rl,
                                 "no_rear_box_over_reference": rn},
                     "note": "wall-clock ratios inside one run on one machine"}
    else:
        out["K1"] = {"verdict": None}
    return out


def stage_summary(res: dict, registered: bool) -> dict:
    arms = res.get("arms") or {}
    table: dict = {}
    for n in ARM_ORDER:
        a = arms.get(n)
        if not a:
            table[n] = {"measured": False}
            continue
        s1, p2 = _p(res, n), a.get("probe2") or {}
        s2 = p2.get("summary")
        lay = ((res.get("layouts") or {}).get(n) or {}).get("layout") or {}
        key8 = "without_the_last_%d_columns" % EXCLUDE_LAST
        row = {
            "role": ARMS[n]["role"], "nx": lay.get("nx"),
            "device_planes": lay.get("device_planes"),
            "n_windows": lay.get("n_windows"),
            "u_rotor_at_release": (a.get("spinup") or {}).get(
                "u_rotor_at_the_release_state"),
            "spinup_s_per_macro_step": (a.get("spinup") or {}).get("s_per_macro_step"),
        }
        if s1:
            row["release_sizing"] = {
                "host_inflow": s1["host_inflow"],
                "machine_first": s1["machine"]["first"],
                "machine_declined_steps": s1["machine"]["declined_steps"],
                "u_rotor_fall": s1["u_rotor"]["fall_fraction"],
                "u_rotor_span": s1["u_rotor"]["span"],
                "FLUID_first": s1["FLUID"]["first"],
                "FLUID_declined_steps": s1["FLUID"]["declined_steps"],
                "on_the_outflow_column_while_declining": s1["fluid"][
                    "fastest_cell_on_the_outflow_column_while_FLUID_declines"],
                "without_the_outflow_column_first_over":
                    s1["fluid"]["without_the_outflow_column"]["first_over_the_bound"],
                "without_the_outflow_column_max":
                    s1["fluid"]["without_the_outflow_column"]["max"],
                "without_the_last_8_first_over": s1["fluid"][key8]["first_over_the_bound"],
                "without_the_last_8_max": s1["fluid"][key8]["max"],
                "u_max_max": s1["fluid"]["u_max"]["max"],
                "load_mean_last_150": s1["aero"]["load_mean_last_%d" % TAIL],
                "drag_mean_last_150": s1["aero"]["drag_mean_last_%d" % TAIL],
                "s_per_macro_step": s1["s_per_macro_step"],
                "window_fits_one_sizing": (s1.get("window_check") or {}).get(
                    "fits_one_sizing"),
            }
        if s2:
            row["arms_sizing"] = {
                "host_inflow": s2["host_inflow"], "steps": s2["steps"],
                "machine_first": s2["machine"]["first"],
                "machine_last": s2["machine"]["last"],
                "machine_declined_steps": s2["machine"]["declined_steps"],
                "machine_declined_at_or_after_step_%d" % AFTER_THE_START:
                    s2["machine"]["declined_at_or_after_step_%d" % AFTER_THE_START],
                "longest_inside_run": s2["machine"]["longest_inside_run"],
                "FLUID_first": s2["FLUID"]["first"],
            }
        elif p2:
            row["arms_sizing"] = {"marched": False, "why": p2.get("why")}
        # what the arm admits, as far as it was marched; None = not measured
        full1 = bool(s1 and s1["steps"] == HORIZON)
        full2 = bool(s2 and s2["steps"] == HORIZON)
        m_ok = None
        if full1 and s1["machine"]["declined_steps"] == 0:
            m_ok = True
        elif full2:
            m_ok = s2["machine"]["declined_steps"] == 0
        row["verdict"] = {
            "machine_inside_all_600_at_a_probed_sizing": m_ok,
            "fluid_inside_all_600_by_the_model": (
                None if not full1 else s1["FLUID"]["declined_steps"] == 0),
            "fluid_inside_all_600_without_the_outflow_column": (
                None if not full1 else
                s1["fluid"]["without_the_outflow_column"]["steps_over_the_bound"] == 0),
        }
        table[n] = row
    comparisons = {}
    for n, ref in REFERENCE.items():
        a, b = table.get(n, {}), table.get(ref, {})
        ra, rb = a.get("release_sizing"), b.get("release_sizing")
        if not (ra and rb):
            continue
        comparisons["%s vs %s" % (n, ref)] = {
            k: {"arm": ra.get(k), "reference": rb.get(k)}
            for k in ("machine_first", "machine_declined_steps", "u_rotor_fall",
                      "FLUID_first", "FLUID_declined_steps",
                      "without_the_outflow_column_max", "load_mean_last_150",
                      "drag_mean_last_150", "s_per_macro_step")}
        comparisons["%s vs %s" % (n, ref)]["u_rotor_at_release"] = {
            "arm": a.get("u_rotor_at_release"), "reference": b.get("u_rotor_at_release")}
    verdicts = judge(res, registered)
    print("   predictions: %s" % ", ".join(
        "%s=%s" % (k, v.get("verdict")) for k, v in verdicts.items()), flush=True)
    return {"table": table, "comparisons": comparisons, "predictions": verdicts,
            "comparable_with_tier56": verdicts.get("C1", {}).get("verdict")}


STAGES = ("band", "layouts", "arms", "summary")


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True,
                    help="output directory; the record is <out>/<basename>.json")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--arms", default=",".join(ARM_ORDER))
    ap.add_argument("--spin-steps", type=int, default=N_SPIN,
                    help="ONLY for a smoke test; a run that changes it is not judged")
    ap.add_argument("--horizon", type=int, default=HORIZON,
                    help="ONLY for a smoke test; a run that changes it is not judged")
    args = ap.parse_args(argv)
    want = [s.strip() for s in args.stages.split(",") if s.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for s in want:
        if s not in STAGES:
            raise SystemExit("unknown stage %r; known: %s" % (s, STAGES))
    for a in arms:
        if a not in ARMS:
            raise SystemExit("unknown arm %r; known: %s" % (a, ARM_ORDER))
    T54.configure(args.out)
    steps = {"spin": int(args.spin_steps), "horizon": int(args.horizon)}
    registered = steps["spin"] == N_SPIN and steps["horizon"] == HORIZON
    torch.set_num_threads(THREADS)

    res = T54.load_res()
    #: the prediction and its timestamp are written BEFORE the first stage
    res.setdefault("tier", 58)
    res.setdefault("prediction", [{"id": pid, "text": text}
                                  for pid, text in PREDICTION])
    res.setdefault("prediction_recorded_at",
                   dt.datetime.now().isoformat(timespec="seconds"))
    res.setdefault("read_before_this_run", list(READ_BEFORE_THIS_RUN))
    res.setdefault("arms_defined", {n: {k: (list(v) if isinstance(v, tuple) else v)
                                        for k, v in ARMS[n].items()}
                                    for n in ARM_ORDER})
    res.setdefault("reference_of", dict(REFERENCE))
    if res.get("lengths") not in (None, steps):
        raise SystemExit("this record was marched at %s and this run asks for %s; "
                         "use a new --out" % (res["lengths"], steps))
    res["lengths"] = steps
    res["lengths_are_the_registered_ones"] = registered
    res.setdefault("runs", []).append({
        "stages": want, "arms": arms, "argv": list(argv),
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        "torch": torch.__version__, "numpy": np.__version__,
        "torch_threads": torch.get_num_threads(),
        "machine_state": T54.machine_state()})
    T54.persist(res)

    for s in want:
        print("== stage %s ==" % s, flush=True)
        t0 = time.perf_counter()
        try:
            if s == "band":
                res["band"] = stage_band()
            elif s == "layouts":
                res["layouts"] = stage_layouts()
            elif s == "arms":
                for name in arms:
                    print("-- arm %s --" % name, flush=True)
                    try:
                        run_arm(res, name, steps)
                    except T54.StalledMarch as exc:
                        #: a blow-up is a measured outcome of that arm, and the
                        #: other arms still run
                        res["arms"][name]["stalled"] = str(exc)
                        res.setdefault("failures", []).append({
                            "stage": "arms", "arm": name, "type": "StalledMarch",
                            "why": str(exc),
                            "at": dt.datetime.now().isoformat(timespec="seconds")})
                        T54.persist(res)
                        print("   [%s] STALLED: %s" % (name, exc), flush=True)
            elif s == "summary":
                res["summary"] = stage_summary(res, registered)
        except BaseException as exc:
            res.setdefault("failures", []).append({
                "stage": s, "type": type(exc).__name__, "why": str(exc)[:1200],
                "after_s": time.perf_counter() - t0,
                "at": dt.datetime.now().isoformat(timespec="seconds"),
                "traceback": traceback.format_exc()[-3000:]})
            T54.persist(res)
            print("   STAGE %s FAILED: %s: %s" % (s, type(exc).__name__,
                                                  str(exc)[:400]), flush=True)
            raise
        res.setdefault("stage_wall_s", {})[s] = time.perf_counter() - t0
        print("   -> %s" % T54.persist(res), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
