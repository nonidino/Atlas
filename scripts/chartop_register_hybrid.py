"""The registration after section 11.14 (`atlas.chartop.gates_hybrid`): the new problems,
measured with the classical solver only.

Accepted by the owner on 2026-11-07 ("Accept as drafted",
[[chart-operator-kernel-exact-forms]] section 11.15), with its three recommended choices.
No learned expert, and not U's form, reads the new problems here; only the classical
superelement and the full-domain solve do, which are properties of the problems and the
port basis, not of any network.

``--classical`` (the default) builds the twelve new problems (`benchmarks_workbench`).  For
each it records the pieces and their charts, the extremes of the pieces' aspects, the
envelope's refusals in both frames, the Fourier numbers against the training generator's
range, the Dirichlet data's range, the classical floor in both frames (E4's truncation term
and its maximum error over the range) with E4's verdict on it, the rule's picks, and a
fingerprint of the problem's definition.  A problem whose floor fails E4 in either frame
is listed as dropped.  It checks that each step's time step is the rule's
(`benchmarks_workbench.dt_by_rule`) and, as the control, that `gates_hybrid.rule_pieces`
picks section 11.14's pieces on its six problems.  It writes
``out/chartop/registration-hybrid.json`` and prints a table of ten lines or fewer.
Pinning is a separate step, run only with the owner's approval.
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
from atlas.chartop import benchmarks_workbench as BW, data, gates_hybrid as GH, host as hs  # noqa: E402

OUT = os.path.join(ROOT, "out", "chartop")
RECORD = os.path.join(ROOT, *GH.CLASSICAL_RECORD.split("/"))


def build(name: str, problem: str, frame: str = "registered"):
    b = BW.build(name, GH.BENCHMARK_N, problem)
    return b if frame == "registered" else BW.datum_shift(b, GH.DATUM_SHIFT_K)


def aspects(problems) -> dict:
    """Each non-Winslow piece's mean side length along xi over that along eta (from its
    corners), and the log-polar pieces' registered moduli."""
    asp, mod = [], []
    for p in problems:
        if p.chart.method.startswith("winslow"):
            continue
        C = p.chart.corners()
        s = [float(np.hypot(*(C[(k + 1) % 4] - C[k]))) for k in range(4)]
        asp.append((s[0] + s[2]) / (s[1] + s[3]))
        if "modulus" in (p.chart.info or {}):
            mod.append(float(p.chart.info["modulus"]))
    return {"aspect_range": [min(asp), max(asp)] if asp else None,
            "moduli": sorted({round(v, 6) for v in mod}) or None}


def measure(name: str, problem: str) -> dict:
    out = {"name": name, "problem": problem, "frames": {}}
    for frame in ("registered", "lowered"):
        b = build(name, problem, frame)
        L = b.layout(GH.KIND, GH.M_PORTS)
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
            picks = GH.rule_pieces(L)
            methods = {}
            for p in b.problems:
                methods[p.chart.method] = methods.get(p.chart.method, 0) + 1
            out.update(pieces=len(b.problems), fingerprint=RP.fingerprint(b),
                       kappa_ref=sorted({float(p.kappa_ref) for p in b.problems}),
                       fo_range=[min(fos), max(fos)] if fos else None,
                       dirichlet_range=[min(dvals), max(dvals)], charts=methods,
                       picks=picks, picks_labels=[str(b.problems[i].label) for i in picks],
                       dirichlet_owners=len({i for i, _ in GH.dirichlet_sides(L)}),
                       dt=b.meta["dt"] if problem == "step" else None, **aspects(b.problems))
    f = out["frames"]
    verdict = GH.judge_e4_both_frames(
        *({"energy_rel": f[k]["truncation_energy_rel"], "max_err": f[k]["floor_max"],
           "span": f[k]["span"], "floor_max": f[k]["floor_max"]} for k in ("registered", "lowered")))
    out["floor_meets_e4"] = bool(verdict["met"])
    return out


def control_picks() -> dict:
    """`gates_hybrid.rule_pieces` on section 11.14's six problems against the labels its
    record holds (the registered frame; the picks depend on the layout only)."""
    rec = json.load(open(os.path.join(ROOT, *GH.CONTROL_RECORD.split("/")), encoding="utf-8"))
    out = {}
    for name, problem in GH.CONTROL:
        b = RP.build(name, problem, GH.CONTROL_VARIANT)
        L = b.layout(GH.KIND, GH.M_PORTS)
        labels = [str(b.problems[i].label) for i in GH.rule_pieces(L)]
        out[f"{name} {problem}"] = labels == rec["problems"][f"{name} {problem} held-out"]["hard"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.parse_args(argv)
    t0 = time.perf_counter()
    rec = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "experts_run": ["classical"],
           "registered": GH.REGISTERED, "problems": []}
    for name, problem in GH.NEW_PROBLEMS:
        rec["problems"].append(measure(name, problem))
    rec["dt_by_rule"] = {name: BW.dt_by_rule(name, GH.BENCHMARK_N) for name in BW.NAMES}
    rec["dt_is_the_rules"] = all(rec["dt_by_rule"][k] == BW.DT[k] for k in BW.NAMES)
    rec["control_picks_equal_section_11_14"] = control_picks()
    fo_lo, fo_hi = data.FO_RANGE
    rec["fo_within_training"] = all(
        r["fo_range"] is None or (fo_lo <= r["fo_range"][0] and r["fo_range"][1] <= fo_hi)
        for r in rec["problems"])
    rec["refusals"] = sum(len(r["frames"][k]["refusals"]) for r in rec["problems"]
                          for k in ("registered", "lowered"))
    rec["dropped"] = [f"{r['name']} {r['problem']}" for r in rec["problems"] if not r["floor_meets_e4"]]
    rec["elapsed_s"] = time.perf_counter() - t0
    registry.write_atomic(RECORD, json.dumps(ev0.strip(rec), indent=1).encode("utf-8"))
    P = {(r["name"], r["problem"]): r for r in rec["problems"]}
    print("NEW       scenario       pieces picks  dt s   floor E%: steady  step (lowered)   Fo range       refusals")
    for name in BW.NAMES:
        s, t = P[(name, "steady")], P[(name, "step")]
        fs, ft = s["frames"]["registered"], t["frames"]["lowered"]
        fo = "-" if t["fo_range"] is None else f"{t['fo_range'][0]:.3g}-{t['fo_range'][1]:.3g}"
        nref = sum(len(r["frames"][k]["refusals"]) for r in (s, t) for k in ("registered", "lowered"))
        print(f"          {name:14s} {s['pieces']:5d} {len(s['picks']):5d} {t['dt']:6g}   "
              f"{100 * fs['truncation_energy_rel']:13.3g} {100 * ft['truncation_energy_rel']:7.3g}"
              f"             {fo:14s} {nref:3d}")
    print(f"          dt is the rule's: {rec['dt_is_the_rules']}; Fo within training's {data.FO_RANGE}: "
          f"{rec['fo_within_training']}; control picks equal 11.14's: "
          f"{all(rec['control_picks_equal_section_11_14'].values())}; dropped: {rec['dropped'] or 'none'}")
    print(f"          wrote {os.path.relpath(RECORD, ROOT)} ({rec['elapsed_s']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
