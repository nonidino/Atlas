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

``--pin``, run with the owner's approval ("Pin it", 2026-11-07), writes
``out/chartop/registered-hybrid.txt``: the sha256 of the drafted scope (`PINNED` and the
classical record), of the files the earlier registration pinned already (which must agree
with its ``registered-prior.txt``) and of the control's record, line endings normalised;
and, by value, the arm, the new problems with the rule's picks and floors, and the price's
protocol.  It refuses to overwrite a registration, and to pin a classical record that fails
its own controls or no longer describes these problems.
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


#: the drafted scope, settled by the owner: the gates module, the new problems, the code H runs
PINNED = {"gate module": "atlas/chartop/gates_hybrid.py",
          "new problems": "atlas/chartop/benchmarks_workbench.py",
          "H's form": "atlas/chartop/forms.py",
          "H's host": "atlas/chartop/host.py"}
#: pinned already by the earlier registration (``registered-prior.txt``); the hashes must agree
ALREADY_PINNED = {"reads E4 from": "atlas/chartop/gates_forms.py",
                  "and through it": "atlas/chartop/gates.py",
                  "the control from": "atlas/chartop/gates_prior.py",
                  "helpers from": "atlas/chartop/benchmarks.py"}
#: recorded beside, by hash: the record the control must reproduce
RECORDED_BESIDE = {"control record": GH.CONTROL_RECORD}
REGISTRATION = os.path.join(OUT, "registered-hybrid.txt")
PRIOR_REGISTRATION = os.path.join(OUT, "registered-prior.txt")


def pin(before: str) -> int:
    """Write ``out/chartop/registered-hybrid.txt`` (refusing to overwrite one, or to pin a
    classical record that fails its own controls or no longer describes these problems)."""
    import datetime as dt
    import re
    import subprocess
    if os.path.exists(REGISTRATION):
        raise SystemExit(f"{REGISTRATION} exists; a registration is not overwritten")
    rec = json.load(open(RECORD, encoding="utf-8"))
    if not (rec["experts_run"] == ["classical"] and rec["dropped"] == [] and rec["refusals"] == 0
            and rec["fo_within_training"] and rec["dt_is_the_rules"]
            and all(rec["control_picks_equal_section_11_14"].values())):
        raise SystemExit("the classical record does not pass its own controls; not pinning")
    for r in rec["problems"]:
        if RP.fingerprint(build(r["name"], r["problem"])) != r["fingerprint"]:
            raise SystemExit(f"{r['name']} {r['problem']} is not the problem the record measured")
    prior = open(PRIOR_REGISTRATION, encoding="ascii").read()
    for rel in ALREADY_PINNED.values():
        m = re.search(re.escape(rel) + r"\s+sha256 ([0-9a-f]{64})", prior)
        if not m or m.group(1) != RP.file_hash(os.path.join(ROOT, rel)):
            raise SystemExit(f"{rel} is not as the earlier registration pinned it")
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    env = data.ENVELOPE
    u = ", ".join(f"{k}={v!r}" for k, v in GH.U_FORM.items() if k != "seed")
    lines = ["THE CHART OPERATOR, THE REGISTRATION AFTER SECTION 11.14 (CO-D, THE FIXED RULE'S HYBRID H)",
             "",
             "written            %s" % dt.datetime.now().astimezone().isoformat(timespec="seconds"),
             "before             %s" % before,
             "repository at      %s (the commit adding this file follows it)" % head]
    for label, rel in PINNED.items():
        lines.append("%-18s %s  sha256 %s" % (label, rel, RP.file_hash(os.path.join(ROOT, rel))))
    lines.append("classical record   %s  sha256 %s" % (GH.CLASSICAL_RECORD, RP.file_hash(RECORD)))
    for label, rel in ALREADY_PINNED.items():
        lines.append("%-18s %s  sha256 %s  (as registered-prior.txt pinned it)"
                     % (label, rel, RP.file_hash(os.path.join(ROOT, rel))))
    for label, rel in RECORDED_BESIDE.items():
        lines.append("%-18s %s  sha256 %s" % (label, rel, RP.file_hash(os.path.join(ROOT, *rel.split("/")))))
    lines += ["registered as      %s" % GH.REGISTERED,
              "",
              "THE ARM, H, BY VALUE (no weights)",
              "  the rule           rule_pieces: the classical solve on every piece whose chart is a Winslow",
              "                     chart and on every piece that owns a Dirichlet side",
              "  everywhere else    U's form without its networks: forms.KernelExpert(%s)," % u,
              "                     built after torch.manual_seed(%d), zero heads, the networks replaced by"
              % GH.U_FORM["seed"],
              "                     a stand-in answering their zeros",
              "  then               the coupled solve (host.Layout.couple)",
              "",
              "THE NEW PROBLEMS: benchmarks_workbench's %s, problems %s, at n = %d"
              % (", ".join(BW.NAMES), " and ".join(BW.PROBLEMS), GH.BENCHMARK_N),
              "  the ends' data     hot 400 + 20 cos(pi s), cold 300 + 10 cos(2 pi s), s from 0 to 1 along each",
              "                     end; every other side insulated; the step one backward-Euler step from 300 K",
              "  the step's dt, s   %s" % "; ".join(f"{k} {v:g}" for k, v in BW.DT.items()),
              "  dimensions, m      wall %s; strip %s; plate %s; insert %s; spreader %s; arc %s; hole %s"
              % (BW.WALL, BW.STRIP, BW.PLATE, BW.INSERT, BW.SPREADER, BW.ARC, BW.HOLE),
              "  read only by the classical superelement and the full-domain solve; the rule's picks and",
              "  the classical floors:"]
    for r in rec["problems"]:
        f = r["frames"]
        lines.append("    %-13s %-6s %2d pieces, picks %s; floor %.4g%% as registered, %.4g%% lowered" % (
            r["name"], r["problem"], r["pieces"], r["picks"], 100 * f["registered"]["truncation_energy_rel"],
            100 * f["lowered"]["truncation_energy_rel"]))
    lines += ["  the control        %s, benchmarks' variant %r, as section 11.14 measured them"
              % (", ".join(f"{a} {b}" for a, b in GH.CONTROL), GH.CONTROL_VARIANT),
              "",
              "THE PRICE'S PROTOCOL, BY VALUE",
              "  %d torch threads (reported beside at %d); each time the best of %d, the arms in turn: %s"
              % (GH.PRICE_THREADS, GH.PRICE_THREADS_REPORTED, GH.PRICE_REPS, ", ".join(GH.PRICE_ORDER)),
              "  %d runs, each in its own process, the median per problem; a void run is run again, at most %d"
              % (GH.PRICE_RUNS, GH.PRICE_RERUNS_MAX),
              "  more times; a run counts only if on AC at its start and end, with no power event %s in"
              % (list(GH.POWER_EVENT_IDS),),
              "  its window, no other process above %g of a core, and the classical expert at %g-%g ms a piece"
              % (GH.VALID_OTHER_CORE, *GH.VALID_MS_PER_PIECE),
              "",
              "THE TRAINING ENVELOPE (data.ENVELOPE), as gates_forms",
              "  sup|mu| <= %g, max log J - min log J <= %.4g, |grad log J| + |grad mu| <= %g,"
              % (env.k_mu, env.l_J, env.s_grad),
              "  |log kappa_hat| <= %g, Fo in [%g, %g] or steady" % (env.l_kappa, env.fo_min, env.fo_max),
              "",
              GH.text(),
              ""]
    with open(REGISTRATION, "w", encoding="ascii", newline="\n") as fh:
        fh.write("\n".join(lines))
    print("registered ->", os.path.relpath(REGISTRATION, ROOT))
    for label, rel in list(PINNED.items()) + [("classical record", GH.CLASSICAL_RECORD)]:
        print("  %-18s %s %s" % (label, rel, RP.file_hash(os.path.join(ROOT, *rel.split("/")))[:16]))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pin", action="store_true",
                    help="write out/chartop/registered-hybrid.txt (only with the owner's approval)")
    ap.add_argument("--before", default="any expert but the classical one read the new problems "
                                        "(the one evaluation's script not yet written)")
    args = ap.parse_args(argv)
    if args.pin:
        return pin(args.before)
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
