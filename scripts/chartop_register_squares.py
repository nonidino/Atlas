"""The registration after section 11.17 (`atlas.chartop.gates_squares`): the generated
layouts, drawn and measured with the classical solver only.

Accepted by the owner on 2026-11-08 ("Accept as drafted",
[[chart-operator-kernel-exact-forms]] section 11.18), with its four recommended choices.  No
learned expert, not U's form and not the reuse expert reads the layouts here; only the
classical superelement and the full-domain solve do, which are properties of the problems
and the port basis.

It walks the seeds from `benchmarks_squares.SEED_BASE` and keeps the first `N_LAYOUTS`
whose two problems (steady and step) are built, are not refused by the envelope (which also
bounds the Fourier numbers), and have classical floors that meet E4 in both frames.  Every
seed it skips is listed with the reason.  For each layout it records the specification, the
pieces, the squares and rule R2's picks, the charts, the time step, and per problem the
floors in both frames, the refusals, the Fourier numbers, the Dirichlet data's range and a
fingerprint.  It writes ``out/chartop/registration-squares.json`` and prints ten lines or
fewer.  Pinning is a separate step, run only with the owner's approval and after the price
protocol's dry run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

import chartop_evaluate as ev0  # noqa: E402
import chartop_register_prior as RP  # noqa: E402
import chartop_registry as registry  # noqa: E402
from atlas.chartop import benchmarks as bm, benchmarks_squares as BS, data, gates_squares as GS, host as hs  # noqa: E402

RECORD = os.path.join(ROOT, *GS.CLASSICAL_RECORD.split("/"))
MAX_SEEDS = 200


def measure(spec: dict, problem: str) -> dict:
    out = {"problem": problem, "frames": {}}
    for frame in ("registered", "lowered"):
        b = BS.build(spec, GS.BENCHMARK_N, problem)
        if frame == "lowered":
            b = bm.datum_shift(b, GS.DATUM_SHIFT_K)
        L = b.layout(GS.KIND, GS.M_PORTS)
        mono = hs.monolithic(L)
        cl = L.solve(hs.ClassicalExpert())
        un, ud = cl.energy_error(mono.fields)
        mx0, span = ev0.field_errors(mono.fields, cl.fields)
        out["frames"][frame] = {"truncation_energy_rel": un / ud, "floor_max": mx0, "span": span,
                                "floor_max_over_range": mx0 / span,
                                "refusals": L.check_envelope(data.ENVELOPE)}
        if frame == "registered":
            fos = [p.fo for p in b.problems if p.fo is not None]
            dvals = []
            for i, p in enumerate(b.problems):
                for s in range(4):
                    kind, val = b.bc(i, s, hs._side_points(p, s))
                    if kind == "dirichlet":
                        dvals += list(np.asarray(val, float))
            out.update(fingerprint=RP.fingerprint(b), fo_range=[min(fos), max(fos)] if fos else None,
                       dirichlet_range=[min(dvals), max(dvals)])
    f = out["frames"]
    verdict = GS.judge_e4_both_frames(
        *({"energy_rel": f[k]["truncation_energy_rel"], "max_err": f[k]["floor_max"],
           "span": f[k]["span"], "floor_max": f[k]["floor_max"]} for k in ("registered", "lowered")))
    out["floor_meets_e4"] = bool(verdict["met"])
    out["refused"] = sum(len(f[k]["refusals"]) for k in f)
    return out


def layout_summary(spec: dict) -> dict:
    b = BS.build(spec, GS.BENCHMARK_N, "steady")
    L = b.layout(GS.KIND, GS.M_PORTS)
    picks = GS.rule_pieces2(L)
    charts = {}
    for p in b.problems:
        charts[p.chart.method] = charts.get(p.chart.method, 0) + 1
    return {"pieces": L.P, "squares": L.P - len(picks), "picks": picks,
            "picks_labels": [str(b.problems[i].label) for i in picks], "charts": charts,
            "conforming": all(e.conforming for e in L.edges if e.other is not None),
            "dt": BS.dt_of(spec, GS.BENCHMARK_N)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.parse_args(argv)
    t0 = time.perf_counter()
    rec = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "experts_run": ["classical"],
           "registered": GS.REGISTERED, "layouts": [], "skipped": []}
    seed = GS.SEED_BASE
    while len(rec["layouts"]) < GS.N_LAYOUTS and seed < GS.SEED_BASE + MAX_SEEDS:
        spec = BS.draw(seed)
        try:
            lay = {"seed": seed, "spec": spec, **layout_summary(spec),
                   "problems": [measure(spec, p) for p in BS.PROBLEMS]}
        except Exception as e:                      # a chart that cannot be built
            rec["skipped"].append({"seed": seed, "feature": spec["feature"],
                                   "why": f"not built: {type(e).__name__}: {e}"[:200]})
            seed += 1
            continue
        why = [f"{r['problem']}: the floor fails E4" for r in lay["problems"] if not r["floor_meets_e4"]]
        why += [f"{r['problem']}: {r['refused']} refusals" for r in lay["problems"] if r["refused"]]
        if not lay["conforming"]:
            why.append("a seam is not conforming")
        if why:
            rec["skipped"].append({"seed": seed, "feature": spec["feature"], "why": "; ".join(why)})
        else:
            rec["layouts"].append(lay)
        seed += 1
    rec["seeds"] = [x["seed"] for x in rec["layouts"]]
    rec["elapsed_s"] = time.perf_counter() - t0
    registry.write_atomic(RECORD, json.dumps(ev0.strip(rec), indent=1).encode("utf-8"))
    print("SQUARES   seed     feature          W x H  pieces squares  dt s   floor E%: steady  step (lowered)")
    for x in rec["layouts"]:
        s, t = x["problems"]
        print(f"          {x['seed']}  {x['spec']['feature']:15s}  {x['spec']['W']} x {x['spec']['H']}  {x['pieces']:5d} "
              f"{x['squares']:6d}  {x['dt']:5g}   {100 * s['frames']['registered']['truncation_energy_rel']:13.3g} "
              f"{100 * t['frames']['lowered']['truncation_energy_rel']:7.3g}")
    print(f"          kept {len(rec['layouts'])} of {seed - GS.SEED_BASE} seeds; skipped: "
          f"{[(x['seed'], x['why'][:40]) for x in rec['skipped']] or 'none'}"[:240])
    print(f"          wrote {os.path.relpath(RECORD, ROOT)} ({rec['elapsed_s']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
