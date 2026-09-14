"""Tier 59 -- W258 repaired at the outlet, and what is left of the car's two faults.

    python scripts/tier59_outflow_and_start.py --out out/racelab7
    python scripts/tier59_outflow_and_start.py --out out/racelab7 --stages start,summary

Tier 58 (`out/racelab6`, [[poc3-racelab-car-fixes]]) measured the drawn car's two
faults on five copies and found neither in the drawing:

  W258  only the OUTFLOW column leaves the cell-Reynolds bound -- the car's own
        flow never does -- and RaceLab's column pins that column: every window
        is built ``transmission="dirichlet"``, so nothing moves the domain's
        outlet but the global projection, and nothing relaxes what it adds.
  W259  at W228's sizing the machine is outside only for the first few
        macro-steps, after the pipeline swaps the spin-up's clamped declared
        machine for a sized one with six to thirty-three times its thrust
        coefficient.

The user asked for the repairs to be measured in whatever order the evidence
suggests, and made.  The outlet comes first, because it is the only fault with
a mechanism in the code, and because a wake that cannot leave the box is a
candidate for W256's unexplained fall of the duct flow as well.

**The repair is `racelab.OUTFLOW = "convective"`**: `WindowNS._convect_outflow`'s
Orlanski condition, applied by the composition layer to the domain's one open
face once per exchange, before the windows step.  ``"pinned"`` is the old
column, kept bitwise.

Stages, each persisted to ``<out>/<basename>.json``:

  ``control``  the pinned column, marched through this file, must reproduce
               Tier 56's settled field bitwise and the first 60 macro-steps of
               Tier 58's drawn arm bitwise.
  ``repair``   the drawn car on the repaired column -- spin-up, the release
               sizing and W228's -- and the box-length control on BOTH columns:
               the same car, the same seven columns and device planes, in a box
               128 cells longer.  An open boundary is one whose position does
               not shape the flow inside it, and that is the test.
  ``start``    W259, on the repaired column: the spin-up re-settled against the
               machine the release will run (Tier 52's own practice, one Picard
               step, which Tier 54 dropped), then probed.
  ``summary``  every registered prediction judged in code.

Nothing here changes a threshold, a clamp, a bound, the sizing rule or the car.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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
import torch                                                            # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert torch.backends.cuda.matmul.allow_tf32 is False
assert torch.backends.cudnn.allow_tf32 is False

from atlas.cases import racelab as RL                                   # noqa: E402

#: Tier 58's per-step reading and summariser, and Tier 56's record I/O and stall
#: guard -- imported, so the three tiers read a probe the same way.
import tier54_traced_car as T54                                         # noqa: E402
import tier58_car_fixes as T58                                          # noqa: E402

TIER56_SETTLED = T58.TIER56_SETTLED
TIER58_JSON = os.path.join(HERE, "out", "racelab6", "racelab6.json")

HORIZON = T58.HORIZON
N_SPIN = T58.N_SPIN
CONTROL_STEPS = 60
THREADS = T58.THREADS
SNAPS = T58.SNAPS

#: The drawn car's own seven columns, and two more to fill a box 800 cells long:
#: 544 -> 608 -> 672 are strides of 64, so every overlap is at least the halo
#: and the device planes stay at 280 and 392.
DRAWN_COLS = (0, 112, 160, 272, 384, 496, 544)
LONG_COLS = DRAWN_COLS + (608, 672)

ARMS = {
    "drawn_pinned": {
        "nx": RL.RNX, "cols": "derived", "outflow": "pinned",
        "probe1": CONTROL_STEPS, "probe2": None,
        "role": "the control: Tier 56's and Tier 58's column, marched through "
                "this file"},
    "drawn": {
        "nx": RL.RNX, "cols": "derived", "outflow": "convective",
        "probe1": HORIZON, "probe2": HORIZON,
        "role": "the car as drawn, on the repaired column"},
    "long_repaired": {
        "nx": 800, "cols": LONG_COLS, "outflow": "convective",
        "probe1": HORIZON, "probe2": None,
        "role": "the box-length control on the repaired column: the same car, "
                "columns and device planes, 128 cells more box behind it"},
    "long_pinned": {
        "nx": 800, "cols": LONG_COLS, "outflow": "pinned",
        "probe1": HORIZON, "probe2": None,
        "role": "the box-length control on the pinned column, for contrast"},
}
STAGE_ARMS = {"control": ("drawn_pinned",),
              "repair": ("drawn", "long_repaired", "long_pinned")}

#: **Read before this file was written**, and carried into the record.
READ_BEFORE_THIS_RUN = (
    "Pre-flight, 2026-09-13, from Tier 56's settled field (which the pinned "
    "column made): the pinned column through the new code reproduces Tier 56's "
    "first 3 trace rows bitwise; `MixedRollout` with every window classical "
    "equals `RaceRollout` bitwise on the repaired column; after 3 macro-steps "
    "the two columns differ by 0.20 at the outflow column and by 0.018 anywhere "
    "with x < 600.",
    "Over 40 macro-steps from that same field, the outflow column's largest "
    "|v| went 1.17 -> 1.30 -> 1.24 on the repaired column (the largest speed "
    "never on that column) and 1.07 -> 1.50 on the pinned one (on that column "
    "from macro-step 24); u_rotor fell 0.5311 -> 0.5120 on the repaired "
    "column and 0.5311 -> 0.5108 on the pinned one -- so the first 40 "
    "macro-steps of the fall are not the outlet.",
    "Tier 52 settled its release state WITH the machine it had sized "
    "(`settled_field(host_inflow=U_DUCT_TIER51)`, one Picard step); Tier 54's "
    "driver, which Tier 56 re-used, settles with the machine `powertrain` "
    "declares.",
)

#: **The prediction, registered 2026-09-13 before stage ``control`` or
#: ``repair`` ran**, judged in code by `judge` under the same ids.
PREDICTION = (
    ("R1", "CONTROL. The pinned column, marched through this file, reproduces "
           "Tier 56's settled field bitwise and the first 60 macro-steps of "
           "Tier 58's drawn arm bitwise -- u_rotor, u_max, induction, drag, "
           "load and the declined verdicts."),
    ("R2", "THE REPAIR REMOVES W258. On the repaired column the drawn car's "
           "FLUID envelope declines on no macro-step of 600 at either sizing, "
           "and the fastest cell is on the outflow column on fewer than 5% of "
           "the macro-steps."),
    ("R3", "THE OUTLET STOPS SHAPING THE FLOW. On the repaired column the "
           "drawn car's u_rotor over probe 1 in the 800-cell box stays within "
           "1% of the 672-cell box's at every macro-step, with the columns and "
           "device planes identical; on the pinned column the two boxes' first "
           "FLUID declines differ by at least 100 macro-steps."),
    ("R4", "THE CAR'S OWN FLOW MOVES LITTLE. On the repaired column the drawn "
           "car's u_rotor at release is within 2% of the pinned column's "
           "0.531725, and its largest speed away from the last eight columns "
           "over probe 1 is within 5% of the pinned column's 1.8189."),
    ("R5", "W256'S FALL IS THE OUTLET, AFTER ITS START. On the repaired column "
           "the fall of u_rotor over probe 1 is at least 3 percentage points "
           "smaller than the pinned column's 16.6%."),
    ("R6", "W259'S START IS NOT THE OUTLET. At W228's sizing on the repaired "
           "column the machine is still outside at the start -- its first "
           "decline at macro-step 0 -- and on no macro-step at or after 24."),
)


#: **Stage ``start``'s prediction, registered before it ran** -- the record
#: carries the time, under ``prediction_start_recorded_at`` -- and after stage
#: ``repair``'s record was read: on the repaired column the drawn
#: car at W228's sizing (0.456800) was outside on macro-steps 0-2 only, its
#: u_rotor falling 0.531 -> 0.460 over the first 100 macro-steps and by 0.26%
#: per 100 over the last 300.
PREDICTION_START = (
    ("T1", "THE START BELONGS TO THE RELEASE STATE. Settled for 120 macro-steps "
           "against the machine W228 sized from the unsettled state (0.456800, "
           "unenforced), then sized again by W228 on the settled state -- a "
           "probe at its own release sizing, and a probe at that probe's "
           "minimum -- the machine declines on no macro-step of the second "
           "probe's 600, against 3 from the unsettled state, and the fluid on "
           "none."),
    ("T2", "The sizing hardly moves: W228's minimum measured on the settled state "
           "is within 3% of 0.456800, and the second probe's inflow ratio at its "
           "lowest stays at least 0.5% above the window's lower edge."),
    ("T3", "At the settled state's own release sizing the machine is inside on "
           "at least 550 of 600 macro-steps -- the long fall was the flow "
           "adjusting to a machine it had not been settled against."),
)

#: How long the release state is settled against the machine the arms will
#: run.  Chosen from stage ``repair``'s W228 probe, where u_rotor had covered
#: 84% of its whole 600-step fall by macro-step 100; not tuned afterwards.
N_SETTLE = 120


# ---------------------------------------------------------------------------
# the arms
# ---------------------------------------------------------------------------


def arm_tiling(name: str):
    """``(tiling, info, doc)``: the drawn car, on the arm's box and columns."""
    spec = ARMS[name]
    doc = T58.arm_geometry("drawn")
    _objs, flat = RL.car_bodies(geometry=doc)
    if spec["cols"] == "derived":
        t, info = RL.windows_from_geometry(flat, nx=int(spec["nx"]))
        info = dict(info)
    else:
        t0, _i0 = RL.layout()
        t = RL.RaceTiling.of(spec["cols"], t0.rows, nx=int(spec["nx"]),
                             ny=t0.ny, wx=t0.wx, wy=t0.wy)
        info = RL.layout_report(t, flat)
    if not t.covers():
        raise SystemExit("arm %s's layout does not cover its box" % name)
    return t, info, doc


def arm_identity(name: str, t, doc: dict) -> dict:
    """What a settled field for this arm IS: the car, the box, the windows and
    -- new in this tier -- the column's outlet condition."""
    return {"arm": name, "dropped": [],
            "geometry_fingerprint": RL.geometry_fingerprint(geometry=doc),
            "nx": int(t.nx), "ny": int(t.ny),
            "cols": [int(c) for c in t.cols], "rows": [int(r) for r in t.rows],
            "n_spin": int(N_SPIN), "outflow": ARMS[name]["outflow"]}


def load_field(name: str, ident: dict):
    return T58.load_arm_field(name, ident)


def run_arm(res: dict, name: str) -> dict:
    """spinup, probe 1 and (if the arm asks for it) probe 2, each persisted as
    it finishes and skipped when the record already holds it for this arm."""
    torch.set_num_threads(THREADS)
    spec = ARMS[name]
    arm = res.setdefault("arms", {}).setdefault(name, {})
    t, info, doc = arm_tiling(name)
    ident = arm_identity(name, t, doc)
    if arm.get("identity") not in (None, ident):
        raise RuntimeError("the record's arm %s was measured for %s, and is now "
                           "%s; use a new --out" % (name, arm["identity"], ident))
    arm["identity"] = ident
    arm["role"] = spec["role"]
    arm["layout"] = {k: info.get(k) for k in ("cols", "rows", "nx", "ny",
                                              "n_windows", "device_planes")}
    arm.setdefault("machine_state", []).append(
        dict(T54.machine_state(), at=dt.datetime.now().isoformat(timespec="seconds")))
    window = res.get("band")
    outflow = spec["outflow"]

    if "spinup" not in arm:
        print("   [%s] spin-up on the %s column ..." % (name, outflow), flush=True)
        objs, _f = RL.car_bodies(geometry=doc)
        t0 = time.perf_counter()
        u, v, rep = RL.settled_field(
            steps=N_SPIN, host_inflow=None, tiling=t, objects=objs,
            outflow=outflow,
            progress=T54.watch("spin-" + name, N_SPIN, every=40))
        wall = time.perf_counter() - t0
        T58.save_arm_field(name, ident, u, v)
        rep = dict(rep)
        rep.update(wall_s=wall, s_per_macro_step=wall / N_SPIN)
        if name == "drawn_pinned":
            d = np.load(TIER56_SETTLED)
            rep["reproduces_tier56_settled_bitwise"] = bool(
                np.array_equal(u, d["u"]) and np.array_equal(v, d["v"]))
        arm["spinup"] = rep
        T54.persist(res)
        print("   [%s] u_rotor at release %.6f%s" % (
            name, rep["u_rotor_at_the_release_state"],
            ("; Tier 56 bitwise: %s" % rep.get("reproduces_tier56_settled_bitwise"))
            if name == "drawn_pinned" else ""), flush=True)
    u0, v0 = load_field(name, ident)
    u_rel = float(arm["spinup"]["u_rotor_at_the_release_state"])

    if "probe1" not in arm:
        n1 = int(spec["probe1"])
        print("   [%s] probe 1, %d macro-steps at %.6f ..." % (name, n1, u_rel),
              flush=True)
        rows, saved, meta = T58.arm_march(u0, v0, u_rel, n1, t, doc,
                                          name + "-p1", snaps=SNAPS,
                                          outflow=outflow)
        T58.write_json(os.path.join(T54.CACHE, name + "_probe1_rows.json"), rows)
        if saved:
            T54.save_field(name + "_probe1_fields", **saved)
        p1 = {"summary": T58.summarise(rows, meta, window),
              "trajectory": T58.compact(rows), "outflow": outflow}
        if name == "drawn_pinned":
            with open(TIER58_JSON, encoding="utf-8") as fh:
                tr = json.load(fh)["arms"]["drawn"]["probe1"]["trajectory"]
            mine = T58.compact(rows)
            keys = ("u_rotor", "u_max", "induction", "drag", "load", "declined")
            eq = [all(mine[k][i] == tr[k][i] for k in keys) for i in range(n1)]
            p1["control"] = {"against": "out/racelab6/racelab6.json "
                                        "arms.drawn.probe1.trajectory",
                             "compared_steps": n1, "equal_steps": int(sum(eq)),
                             "bitwise": bool(all(eq)), "keys": list(keys)}
        arm["probe1"] = p1
        T54.persist(res)
        s1 = p1["summary"]
        print("   [%s] probe 1: FLUID first %s (%d); machine first %s (%d); fall "
              "%.2f%%; on the outflow column while declining %s"
              % (name, s1["FLUID"]["first"], s1["FLUID"]["declined_steps"],
                 s1["machine"]["first"], s1["machine"]["declined_steps"],
                 100 * s1["u_rotor"]["fall_fraction"],
                 s1["fluid"]["fastest_cell_on_the_outflow_column_while_FLUID_declines"]),
              flush=True)

    if spec["probe2"] and "probe2" not in arm:
        s1 = arm["probe1"]["summary"]
        ud = float(s1["u_rotor"]["min"])
        n2 = int(spec["probe2"])
        print("   [%s] probe 2, %d macro-steps at %.6f ..." % (name, n2, ud),
              flush=True)
        rows, _s, meta = T58.arm_march(u0, v0, ud, n2, t, doc, name + "-p2",
                                       outflow=outflow)
        T58.write_json(os.path.join(T54.CACHE, name + "_probe2_rows.json"), rows)
        arm["probe2"] = {"summary": T58.summarise(rows, meta, window),
                         "trajectory": T58.compact(rows), "outflow": outflow}
        T54.persist(res)
        s2 = arm["probe2"]["summary"]
        print("   [%s] probe 2: machine declined on %d (first %s, last %s); FLUID "
              "first %s" % (name, s2["machine"]["declined_steps"],
                            s2["machine"]["first"], s2["machine"]["last"],
                            s2["FLUID"]["first"]), flush=True)
    return arm


def stage_start(res: dict) -> dict:
    """W259 on the repaired column: settle the release state against the machine
    W228 chose, then apply W228 again on the settled state.

    Tier 52 settled its release state WITH the machine it had sized; Tier 54's
    driver settles with the machine `powertrain` declares, whose disk sits on
    its lower clamp at the drawn car's release state, and the probe that follows
    switches in one with several times its thrust.  So the release state is
    re-settled here, for `N_SETTLE` macro-steps, at the sizing stage ``repair``
    measured from the unsettled state, and the two W228 probes are marched again
    FROM that settled state.  Every step of all three is kept.
    """
    torch.set_num_threads(THREADS)
    d = res["arms"]["drawn"]
    if "probe2" not in d:
        raise RuntimeError("stage start needs stage repair's W228 probe of arm "
                           "drawn")
    t, _info, doc = arm_tiling("drawn")
    ident = arm_identity("drawn", t, doc)
    window = res.get("band")
    out = res.setdefault("start", {})
    ud = float(d["probe2"]["summary"]["host_inflow"])
    sident = dict(ident, arm="drawn_settled", settled_at=ud, n_settle=N_SETTLE)

    if "settle" not in out:
        u0, v0 = load_field("drawn", ident)
        print("   settle %d macro-steps at W228's sizing %.6f ..." % (N_SETTLE, ud),
              flush=True)
        last = N_SETTLE - 1
        rows, saved, meta = T58.arm_march(u0, v0, ud, N_SETTLE, t, doc, "settle",
                                          snaps=(last,), outflow="convective")
        T58.save_arm_field("drawn_settled", sident,
                           saved["u_%03d" % last], saved["v_%03d" % last])
        out["settle"] = {
            "host_inflow": ud, "steps": N_SETTLE,
            "summary": T58.summarise(rows, meta, window),
            "trajectory": T58.compact(rows),
            "u_rotor_at_the_end": float(rows[-1]["u_rotor"])}
        T54.persist(res)
        print("   settled: u_rotor %.6f -> %.6f; machine declined on %d"
              % (rows[0]["u_rotor"], rows[-1]["u_rotor"],
                 out["settle"]["summary"]["machine"]["declined_steps"]), flush=True)
    u1, v1 = T58.load_arm_field("drawn_settled", sident)
    h1 = float(out["settle"]["u_rotor_at_the_end"])

    for key, host in (("probe1", h1), ("probe2", None)):
        if key in out:
            continue
        if host is None:
            host = float(out["probe1"]["summary"]["u_rotor"]["min"])
        print("   %s from the settled state, %d macro-steps at %.6f ..."
              % (key, HORIZON, host), flush=True)
        rows, _s, meta = T58.arm_march(u1, v1, host, HORIZON, t, doc,
                                       "settled-" + key, outflow="convective")
        T58.write_json(os.path.join(T54.CACHE, "settled_%s_rows.json" % key), rows)
        out[key] = {"summary": T58.summarise(rows, meta, window),
                    "trajectory": T58.compact(rows)}
        T54.persist(res)
        s = out[key]["summary"]
        print("   %s: machine declined on %d (first %s, last %s); FLUID %d; ratio "
              "range [%.4f, %.4f]" % (key, s["machine"]["declined_steps"],
                                      s["machine"]["first"], s["machine"]["last"],
                                      s["FLUID"]["declined_steps"],
                                      s["u_rotor"]["ratio_to_the_sizing_min"],
                                      s["u_rotor"]["ratio_to_the_sizing_max"]),
              flush=True)
    return out


# ---------------------------------------------------------------------------
# the prediction, judged
# ---------------------------------------------------------------------------


def _s(res, arm, probe="probe1"):
    return ((((res.get("arms") or {}).get(arm) or {}).get(probe) or {})
            .get("summary"))


def judge(res: dict) -> dict:
    out: dict = {}
    arms = res.get("arms") or {}
    with open(TIER58_JSON, encoding="utf-8") as fh:
        t58 = json.load(fh)
    pinned1 = t58["arms"]["drawn"]["probe1"]["summary"]

    # R1
    a = arms.get("drawn_pinned") or {}
    flags = {"settled_bitwise": (a.get("spinup") or {}).get(
                 "reproduces_tier56_settled_bitwise"),
             "probe_bitwise": ((a.get("probe1") or {}).get("control") or {}).get(
                 "bitwise")}
    out["R1"] = {"verdict": (None if None in flags.values()
                             else all(flags.values())), "numbers": flags}

    # R2
    s1, s2 = _s(res, "drawn"), _s(res, "drawn", "probe2")
    if s1 and s2:
        n = s1["steps"]
        traj = res["arms"]["drawn"]["probe1"]["trajectory"]
        on = sum(1 for a_, b_ in zip(traj["u_max"],
                                     traj["u_max_without_the_outflow_column"])
                 if a_ > b_)
        out["R2"] = {"verdict": bool(s1["FLUID"]["declined_steps"] == 0
                                     and s2["FLUID"]["declined_steps"] == 0
                                     and on < 0.05 * n),
                     "numbers": {"FLUID_declined_release_sizing":
                                     s1["FLUID"]["declined_steps"],
                                 "FLUID_declined_W228_sizing":
                                     s2["FLUID"]["declined_steps"],
                                 "steps_fastest_on_the_outflow_column": on,
                                 "steps": n}}
    else:
        out["R2"] = {"verdict": None}

    # R3
    a672 = (arms.get("drawn") or {}).get("probe1", {}).get("trajectory")
    a800 = (arms.get("long_repaired") or {}).get("probe1", {}).get("trajectory")
    sp = _s(res, "long_pinned")
    if a672 and a800 and sp:
        u1, u2 = np.asarray(a672["u_rotor"]), np.asarray(a800["u_rotor"])
        rel = float(np.max(np.abs(u2 / u1 - 1.0)))
        planes_same = (arms["drawn"]["layout"]["device_planes"]
                       == arms["long_repaired"]["layout"]["device_planes"])
        f672, f800 = pinned1["FLUID"]["first"], sp["FLUID"]["first"]
        gap = (None if f672 is None or f800 is None else abs(f800 - f672))
        out["R3"] = {"verdict": bool(rel <= 0.01 and planes_same
                                     and (gap is None and f672 != f800
                                          or gap is not None and gap >= 100)),
                     "numbers": {"repaired_max_relative_u_rotor_difference": rel,
                                 "device_planes_identical": planes_same,
                                 "pinned_FLUID_first_672": f672,
                                 "pinned_FLUID_first_800": f800}}
    else:
        out["R3"] = {"verdict": None}

    # R4
    if s1:
        ur = arms["drawn"]["spinup"]["u_rotor_at_the_release_state"]
        key8 = "without_the_last_%d_columns" % T58.EXCLUDE_LAST
        m8 = s1["fluid"][key8]["max"]
        p8 = pinned1["fluid"][key8]["max"]
        out["R4"] = {"verdict": bool(abs(ur / 0.5317254889754462 - 1.0) <= 0.02
                                     and abs(m8 / p8 - 1.0) <= 0.05),
                     "numbers": {"u_rotor_release": ur,
                                 "pinned_u_rotor_release": 0.5317254889754462,
                                 "max_without_last_8": m8,
                                 "pinned_max_without_last_8": p8}}
    else:
        out["R4"] = {"verdict": None}

    # R5
    if s1:
        f, fp = s1["u_rotor"]["fall_fraction"], pinned1["u_rotor"]["fall_fraction"]
        out["R5"] = {"verdict": bool(f <= fp - 0.03),
                     "numbers": {"fall_repaired": f, "fall_pinned": fp}}
    else:
        out["R5"] = {"verdict": None}

    # R6
    if s2:
        k = "declined_at_or_after_step_%d" % T58.AFTER_THE_START
        out["R6"] = {"verdict": bool(s2["machine"]["first"] == 0
                                     and s2["machine"][k] == 0),
                     "numbers": {"first": s2["machine"]["first"],
                                 "last": s2["machine"]["last"],
                                 "declined": s2["machine"]["declined_steps"],
                                 k: s2["machine"][k]}}
    else:
        out["R6"] = {"verdict": None}

    # T1 - T3: stage start
    st = res.get("start") or {}
    p1 = (st.get("probe1") or {}).get("summary")
    p2 = (st.get("probe2") or {}).get("summary")
    if p1 and p2:
        lo = res["band"]["machine_inside"][0]
        out["T1"] = {"verdict": bool(p2["steps"] == HORIZON
                                     and p2["machine"]["declined_steps"] == 0
                                     and p2["FLUID"]["declined_steps"] == 0),
                     "numbers": {"machine_declined": p2["machine"]["declined_steps"],
                                 "FLUID_declined": p2["FLUID"]["declined_steps"],
                                 "unsettled_machine_declined":
                                     s2["machine"]["declined_steps"] if s2 else None}}
        m = p2["host_inflow"]
        ud = st["settle"]["host_inflow"]
        rmin = p2["u_rotor"]["ratio_to_the_sizing_min"]
        out["T2"] = {"verdict": bool(abs(m / ud - 1.0) <= 0.03
                                     and rmin >= lo * 1.005),
                     "numbers": {"W228_on_the_settled_state": m, "first": ud,
                                 "ratio_min": rmin, "window_lower_edge": lo}}
        out["T3"] = {"verdict": bool(p1["steps"] - p1["machine"]["declined_steps"]
                                     >= 550),
                     "numbers": {"inside": p1["steps"]
                                 - p1["machine"]["declined_steps"],
                                 "machine": p1["machine"]}}
    else:
        for k in ("T1", "T2", "T3"):
            out[k] = {"verdict": None}
    return out


STAGES = ("control", "repair", "start", "summary")


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True,
                    help="output directory; the record is <out>/<basename>.json")
    ap.add_argument("--stages", default=",".join(STAGES))
    args = ap.parse_args(argv)
    want = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in want:
        if s not in STAGES:
            raise SystemExit("unknown stage %r; known: %s" % (s, STAGES))
    T54.configure(args.out)
    torch.set_num_threads(THREADS)

    res = T54.load_res()
    res.setdefault("tier", 59)
    res.setdefault("prediction", [{"id": pid, "text": text}
                                  for pid, text in PREDICTION])
    res.setdefault("prediction_recorded_at",
                   dt.datetime.now().isoformat(timespec="seconds"))
    res.setdefault("read_before_this_run", list(READ_BEFORE_THIS_RUN))
    res.setdefault("arms_defined", {n: {k: (list(v) if isinstance(v, tuple) else v)
                                        for k, v in ARMS[n].items()}
                                    for n in ARMS})
    if "band" not in res:
        with open(TIER58_JSON, encoding="utf-8") as fh:
            res["band"] = json.load(fh)["band"]
        res["band_source"] = ("out/racelab6/racelab6.json -- the machine's "
                              "window is a property of the machine and not of "
                              "the column, so it is read, not re-measured")
    res.setdefault("runs", []).append({
        "stages": want, "argv": list(argv),
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
        "machine_state": T54.machine_state()})
    T54.persist(res)

    for s in want:
        print("== stage %s ==" % s, flush=True)
        t0 = time.perf_counter()
        try:
            if s in STAGE_ARMS:
                for name in STAGE_ARMS[s]:
                    print("-- arm %s --" % name, flush=True)
                    try:
                        run_arm(res, name)
                    except T54.StalledMarch as exc:
                        res["arms"][name]["stalled"] = str(exc)
                        res.setdefault("failures", []).append({
                            "stage": s, "arm": name, "type": "StalledMarch",
                            "why": str(exc),
                            "at": dt.datetime.now().isoformat(timespec="seconds")})
                        T54.persist(res)
                        print("   [%s] STALLED: %s" % (name, exc), flush=True)
            elif s == "start":
                #: registered BEFORE the stage runs, with its own timestamp
                res.setdefault("prediction_start",
                               [{"id": pid, "text": text}
                                for pid, text in PREDICTION_START])
                res.setdefault("prediction_start_recorded_at",
                               dt.datetime.now().isoformat(timespec="seconds"))
                T54.persist(res)
                stage_start(res)
            elif s == "summary":
                res["summary"] = {"predictions": judge(res)}
                print("   predictions: %s" % ", ".join(
                    "%s=%s" % (k, v.get("verdict"))
                    for k, v in res["summary"]["predictions"].items()), flush=True)
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
