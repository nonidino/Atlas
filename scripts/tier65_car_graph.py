"""Tier 65 -- the body-fitted column declared, and a volumetric device's first port.

    python scripts/tier65_car_graph.py --out out/racelab14 --stages record,compile,power,summary

Tier 64 marched the radiator core and the recovery turbine in the car's duct and
measured the three joins; nothing about that march was DECLARED, so
`atlas.compiler` had never seen the body-fitted column and gate clause P1 could
not be stated on it.  `atlas/cases/car_graph.py` writes the declaration:

  ``record``   what a record can and cannot say about an overset composite: the
               zero-port refusal, the strip ports, the interpolation rows that
               make "step one grid alone" undefined, and the self-seam the
               porous column's shape would have suggested.
  ``compile``  the body-fitted column compiled -- joined, disjoint, with the
               clocks reconciled, and with one join at a time -- beside the
               POROUS column's compile, which this tier's narrowing of R10's
               premise must leave untouched.
  ``power``    the strip port's bond against the march: effort times flow times
               the band's width, against the power Tier 64 measured.
  ``summary``  every registered prediction judged in code.

Nothing here marches, and nothing here touches RaceLab's column or its records.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse                                                         # noqa: E402
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

from atlas import compiler as C                                         # noqa: E402
from atlas.graph import GraphError                                      # noqa: E402
from atlas.cases import car_graph as CG                                 # noqa: E402
from atlas.cases import car_solids as CS                                # noqa: E402
from atlas.cases import ground_effect as GE                             # noqa: E402
from atlas.cases import racelab as RL                                   # noqa: E402
import tier60_body_fitted_grids as T60                                  # noqa: E402

TIER64_RECORD = os.path.join(HERE, "out", "racelab13", "racelab13.json")

READ_BEFORE_THIS_RUN = (
    "Tier 64 (out/racelab13): the registered arm's J3 receiving residual 3.40e-4, the work the "
    "rotor's body force does on the fluid 8.3509e-5 against the shaft's claim 8.3537e-5, and the "
    "band's mean velocity at the turbine plane 0.083511 over the window.",
    "The porous column's graph (racelab.build): 26 agents, 36 seams, and a compile that refuses at "
    "L7/R9 and nothing else, with the disjoint union refusing nothing -- CS-18's table, reproduced "
    "by Tier 51 on the car's own tiling.",
    "Explored before these predictions were written, on the declaration this tier builds: a record "
    "with no ports is refused by the record layer itself ('expert FLUID declares no ports'); with "
    "the strip ports declared FLOW-returning the compile refused at L3/C9 (both sides of a MECH "
    "seam return the effort in this package) and at L1/record (no boundary_response); with both "
    "repaired it refused at L2/R10 and L7/R9; and R10's premise -- 'another agent of the same "
    "governing_family' -- was being met by the actuator disk, which declares "
    "incompressible-navier-stokes-2d with stencil_radius 0 and owns no region at all.",
    "Explored: with R10's premise narrowed by stencil_radius (W160's own narrowing, applied to the "
    "branch beside the one it fixed) the body-fitted column refuses at L7/R9 alone; the disjoint "
    "union and the clocks-reconciled version refuse nothing; one join at a time still refuses "
    "L7/R9; and the self-seam the porous column's shape would suggest is refused by the graph "
    "layer by name ('seam connects agent FLUID to itself').",
    "Explored on the composite itself: 377,267 unknowns, 12,078 interpolation rows (3.2% of them) "
    "with 108,702 donor entries over 40 distinct grid pairs -- the rows that couple one grid's "
    "unknowns to another's inside one matrix.",
    "Not measured before these predictions: whether the narrowing leaves the POROUS column's "
    "compile and the whole test suite untouched, what the strip bond's power comes to against the "
    "march's own number, and what the compile says seam by seam.",
)

PREDICTION = (
    {"id": "D1", "claim": "the narrowing leaves the porous column's compile exactly as it was: 26 "
                          "agents, 36 seams, refusals ['L7/R9'], the disjoint union refusing nothing",
     "why": "every fluid window declares stencil_radius 2, so the family's count over field agents "
            "is unchanged at 14; only field-less agents leave the count"},
    {"id": "D2", "claim": "the whole test suite passes with the narrowing in",
     "why": "the premise is only read for agents that declare an embedded elliptic sub-solve, and "
            "no case in the package has one beside a field-less agent of its own family but this"},
    {"id": "D3", "claim": "the body-fitted column compiles with refusals ['L7/R9'] and nothing else, "
                          "its disjoint union refuses nothing, and reconciling the clocks removes "
                          "the refusal",
     "why": "CS-18's table on a third graph: the three clocks are the refusal and the joins are not"},
    {"id": "D4", "claim": "the declaration is 11 agents and 13 seams against the porous column's 26 "
                          "and 36, and NO seam of it is fluid-fluid",
     "why": "the composite is one expert, so thirteen window-window seams, the wetted seam and the "
            "mount seam all vanish; what is left is one seam a device"},
    {"id": "D5", "claim": "the strip bond's power -- the record's effort at the march's band, times "
                          "that band, times its width -- is the shaft power Tier 64 measured to "
                          "within 1%",
     "why": "the effort is the thrust over the width by construction, so the product telescopes to "
            "T times the band's mean; what is left is the trace the record is written at"},
    {"id": "D6", "claim": "the two strip seams earn no refusal of their own: every decision whose "
                          "subject is one of them, its ports or its agents is admit or "
                          "admit-uncertified",
     "why": "both sides return the effort, the scales are MECH's, the prolongations are declared, "
            "and the null space of a strip's bond is zero"},
    {"id": "D7", "claim": "a graph that declares a seam from the fluid to itself is refused by name",
     "why": "the porous column's shape puts the device's two faces on two windows; nothing cuts the "
            "composite at the device plane, so the same declaration is a self-seam"},
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


def _compile(graph) -> dict:
    res = C.compile_scheme(graph)
    rows = [{"layer": d.layer, "rule": d.rule, "verdict": d.verdict.value,
             "subject": d.subject, "message": d.message[:300]} for d in res.decisions]
    return {"name": graph.name, "verdict": res.verdict.value,
            "n_agents": len(graph.agents), "n_seams": len(graph.connections),
            "counts": dict(Counter(r["verdict"] for r in rows)),
            "refusals": sorted({f"{r['layer']}/{r['rule']}" for r in rows if r["verdict"] == "refuse"}),
            "decertifications": sorted({f"{r['layer']}/{r['rule']}" for r in rows
                                        if r["verdict"] == "admit-uncertified"}),
            "rows": rows}


def _seam_local(graph, rows) -> dict:
    """What each seam earned on its OWN subject, its two ports and its two agents.

    A graph-level refusal painted across every seam is one refusal (W177), so
    only decisions whose subject is in the seam's own reach count here.
    """
    out = {}
    for c in graph.connections:
        reach = {c.seam_id, c.a[0], c.b[0], f"{c.a[0]}.{c.a[1]}", f"{c.b[0]}.{c.b[1]}", c.a[1], c.b[1]}
        local = "admit"
        rules = []
        for r in rows:
            if r["verdict"] == "admit" or r["subject"] not in reach:
                continue
            rules.append(f"{r['layer']}/{r['rule']}")
            if r["verdict"] == "refuse":
                local = "refuse"
            elif local != "refuse":
                local = "admit-uncertified"
        out[c.seam_id] = {"local_verdict": local, "local_rules": sorted(set(rules))}
    return out


def stage_record(res) -> dict:
    out: dict = {"no_port_record": CG.no_port_record(), "holes": list(CG.GRAPH_HOLES)}
    caps = CG.composite_capabilities()
    out["fluid_record"] = {
        "ports": [{"name": p.name, "type": p.port_type.value, "half": p.response_half.value,
                   "modes": p.effective_resolution, "geometry": p.geometry} for p in caps.ports],
        "elliptic_subsolve": caps.elliptic_subsolve.value,
        "time_discretization": caps.time_discretization.value,
        "stencil_radius": caps.stencil_radius,
        "substeps_per_macro_step": caps.substeps_per_macro_step,
        "governing_family": caps.governing_family, "weight_hash": caps.weight_hash,
        "band_written_at": CG.U_BAND}
    t0 = time.perf_counter()
    solids, _rec = CS.car_solids(geometry=copy.deepcopy(RL.load_geometry()))
    ov, _grec = CS.car_overset(solids)
    out["composite"] = CG.cross_grid_rows(ov)
    out["composite"]["build_s"] = time.perf_counter() - t0
    out["composite"]["grids"] = len(ov.grids)
    try:
        CG.self_seam_graph()
        out["self_seam"] = {"refused": False, "message": None}
    except GraphError as exc:
        out["self_seam"] = {"refused": True, "message": str(exc)[:200]}
    res["record"] = out
    persist(res)
    say("record", json.dumps(T60.clean({k: v for k, v in out.items()
                                        if k not in ("holes", "composite")}))[:900])
    say("composite", json.dumps(T60.clean({k: v for k, v in out["composite"].items()
                                           if k != "by_pair"})))
    return out


def stage_compile(res) -> dict:
    out: dict = {}
    for label, kw in (("joined", {}), ("disjoint", {"joins": ()}),
                      ("clocks_reconciled", {"clocks": "reconciled"}),
                      ("J1_only", {"joins": ("J1",)}), ("J3_only", {"joins": ("J3",)})):
        g, aux = CG.build(**kw)
        c = _compile(g)
        if label == "joined":
            c["seam_local"] = _seam_local(g, c["rows"])
            c["info"] = aux["info"]
        else:
            c.pop("rows")
        out[label] = c
        say(label, c["verdict"], c["refusals"], c["n_agents"], "agents", c["n_seams"], "seams")
    # the control that matters: the POROUS column, which the narrowing must not move
    u = np.full((RL.RNY, RL.RNX), GE.U_INF)
    v = np.zeros((RL.RNY, RL.RNX))
    gp, _auxp = RL.build(u, v)
    cp = _compile(gp)
    cp.pop("rows")
    gpd, _ = RL.build(u, v, joins=())
    cpd = _compile(gpd)
    cpd.pop("rows")
    out["porous_column"] = {"joined": cp, "disjoint": cpd}
    out["against_CS18"] = {
        "CS18_joined_refusals": ["L7/R9"],
        "body_fitted_matches": out["joined"]["refusals"] == ["L7/R9"],
        "porous_matches": cp["refusals"] == ["L7/R9"],
        "porous_disjoint_verdict": cpd["verdict"],
    }
    say("porous column", cp["verdict"], cp["refusals"], cp["n_agents"], "agents", cp["n_seams"], "seams")
    res["compile"] = out
    persist(res)
    return out


def stage_power(res) -> dict:
    """The strip bond against the march it describes."""
    out: dict = {}
    width = CG.STRIP_CELLS * GE.DX
    band = np.full(CG.STRIP_SAMPLES, CG.U_BAND)
    g, _aux = CG.build()
    rotor = next(a for a in g.agents if a.agent_id == "ROTOR")
    core = next(a for a in g.agents if a.agent_id == "RAD")
    eff_rotor = np.asarray(rotor.capabilities.boundary_response("strip:MECH", band), dtype=float)
    eff_core = np.asarray(core.capabilities.boundary_response("strip:MECH", band), dtype=float)
    p_rotor = float(np.mean(eff_rotor) * np.mean(band) * width)
    p_core = float(np.mean(eff_core) * np.mean(band) * width)
    out["declared"] = {"rotor_effort": float(np.mean(eff_rotor)),
                       "core_effort": float(np.mean(eff_core)),
                       "band": float(np.mean(band)), "width": width,
                       "rotor_power": p_rotor, "core_power": p_core}
    if os.path.isfile(TIER64_RECORD):
        with open(TIER64_RECORD, encoding="utf-8") as fh:
            r64 = json.load(fh)
        k = r64["arms"].get("registered")
        S = r64["arms"]["list"][(k or len(r64["arms"]["list"])) - 1]
        B = S["balances"]["J3"]
        out["marched"] = {
            "shaft_power": B["the shaft's claim, T <U_d>"],
            "work_of_the_force": B["the join's term: work the rotor's body force does on the fluid"],
            "core_share": B["the core's share of it"],
            "ring": B["u on the ring the disk reads"],
            "u_host": S["u_host"]}
        out["rotor_ratio"] = p_rotor / out["marched"]["shaft_power"]
    # the same bond, evaluated where the march evaluated it: step by step over the
    # window, rather than once at the state the record is written at
    arm_file = os.path.join(HERE, "out", "racelab13", "arm2.json")
    if os.path.isfile(arm_file) and "marched" in out:
        with open(arm_file, encoding="utf-8") as fh:
            arm = json.load(fh)
        tr = arm["union"]["trace"]
        t = np.asarray(tr["t"])
        win = (t >= 12.0 - 1e-9) & (t <= 16.0 + 1e-9)
        u = np.asarray(tr["u_rotor"])[win]
        powers = []
        for uk in u:
            e = np.asarray(rotor.capabilities.boundary_response("strip:MECH",
                                                                np.full(CG.STRIP_SAMPLES, uk)),
                           dtype=float)
            powers.append(float(np.mean(e) * uk * width))
        step_mean = float(np.mean(powers))
        out["step_by_step"] = {
            "declared_power_mean_over_the_window": step_mean,
            "marched_shaft_power_mean": out["marched"]["shaft_power"],
            "ratio": step_mean / out["marched"]["shaft_power"],
            "steps": int(u.size),
            "at_one_state_ratio": out["rotor_ratio"],
            "note": "the bond is the march's own arithmetic; evaluating it at the "
                    "state the record is written at is not evaluating it over the "
                    "window, and the gap between the two ratios is that difference"}
    fluid = next(a for a in g.agents if a.agent_id == "FLUID")
    at_base = np.asarray(fluid.capabilities.boundary_response("rotor:MECH", band), dtype=float)
    faster = np.asarray(fluid.capabilities.boundary_response("rotor:MECH", band * 1.01), dtype=float)
    out["fluid_response"] = {"at_the_band": float(np.max(np.abs(at_base))),
                             "one_percent_faster": float(np.mean(faster)),
                             "probe_base": float(np.mean(fluid.capabilities.probe_base("rotor:MECH")))}
    res["power"] = out
    persist(res)
    say("power", json.dumps(T60.clean(out))[:900])
    return out


def judge(res) -> dict:
    v: dict = {}
    comp = res.get("compile")
    if comp:
        p = comp["porous_column"]["joined"]
        pd = comp["porous_column"]["disjoint"]
        v["D1"] = (p["n_agents"] == 26 and p["n_seams"] == 36 and p["refusals"] == ["L7/R9"]
                   and pd["refusals"] == [])
        v["D3"] = (comp["joined"]["refusals"] == ["L7/R9"] and comp["disjoint"]["refusals"] == []
                   and comp["clocks_reconciled"]["refusals"] == [])
        v["D4"] = (comp["joined"]["n_agents"] == 11 and comp["joined"]["n_seams"] == 13
                   and p["n_agents"] == 26 and p["n_seams"] == 36)
        sl = comp["joined"].get("seam_local", {})
        v["D6"] = (len(sl) > 0 and all(s["local_verdict"] != "refuse"
                                       for k, s in sl.items()
                                       if k in ("J1_core_strip", "J3_rotor_strip")))
    S = res.get("suite")
    if S:
        v["D2"] = bool(S.get("failed") == 0 and S.get("passed", 0) > 1600)
    P = res.get("power")
    if P and "marched" in P:
        v["D5"] = abs(P["rotor_ratio"] - 1.0) <= 0.01
    R = res.get("record")
    if R:
        v["D7"] = bool(R["self_seam"]["refused"])
    return v


def stage_summary(res) -> dict:
    verdicts = judge(res)
    res["verdicts"] = verdicts
    res["verdicts_missing"] = sorted({p["id"] for p in PREDICTION} - set(verdicts))
    persist(res)
    for p in PREDICTION:
        say("%-3s %-6s %s" % (p["id"], verdicts.get(p["id"]), p["claim"][:100]))
    return verdicts


STAGES = ("record", "compile", "power", "summary")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="record,compile,power,summary")
    ap.add_argument("--suite", nargs=2, type=int, metavar=("PASSED", "FAILED"),
                    help="record the suite's tally, which is D2's evidence")
    args = ap.parse_args(argv)
    configure(args.out)
    res = load()
    if "prediction" not in res:
        res.update(tier=65, prediction=list(PREDICTION),
                   prediction_recorded_at=dt.datetime.now().isoformat(timespec="seconds"),
                   read_before_this_run=list(READ_BEFORE_THIS_RUN),
                   machine=T60.machine_state())
        persist(res)
    elif [p["id"] for p in res["prediction"]] != [p["id"] for p in PREDICTION]:
        raise SystemExit("the record's prediction differs from this file's")
    if args.suite:
        res["suite"] = {"passed": args.suite[0], "failed": args.suite[1],
                        "note": "python scripts/run_suite.py, with the narrowing in"}
        persist(res)
    for st in [s.strip() for s in args.stages.split(",") if s.strip()]:
        if st not in STAGES:
            raise SystemExit(f"unknown stage {st!r}; stages are {STAGES}")
        t0 = time.perf_counter()
        try:
            {"record": stage_record, "compile": stage_compile, "power": stage_power,
             "summary": stage_summary}[st](res)
        except Exception:
            err = load()
            err.setdefault("stage_errors", {})[st] = traceback.format_exc()[-3000:]
            persist(err)
            say(f"stage {st} FAILED")
            raise
        res.setdefault("stage_wall_s", {})[st] = time.perf_counter() - t0
        persist(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
