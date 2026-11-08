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
fewer.

``--pin``, run with the owner's approval ("Pin it", 2026-11-08) after the price protocol's
third dry run passed, writes ``out/chartop/registered-squares.txt``: the sha256 (line endings
normalised) of the scope (`PINNED` and the classical record), of the files earlier
registrations pinned (`ALREADY_PINNED`, which must agree with every pin file that lists
them), and of what H2's arm is built with and the passing dry run (`RECORDED_BESIDE`); and,
by value, the arm, the new problems with their picks and floors, and the price's protocol.
It refuses to overwrite a registration; to pin a classical record that fails its own
controls or no longer describes these layouts; and to pin a protocol other than the one the
passing dry run ran (its commit, the gate module's docstring and `REGISTERED` aside).
`check_pin` is what the one evaluation runs first.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import os
import re
import subprocess
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


# ---------------------------------------------------------------------------
# the pin
# ---------------------------------------------------------------------------

#: the scope, settled by the owner: the gate module, the generator, H2r's expert, the price
#: protocol and the box it runs on (`gates_squares.REVISED`)
PINNED = {"gate module": "atlas/chartop/gates_squares.py",
          "new problems": "atlas/chartop/benchmarks_squares.py",
          "H2r's expert": "atlas/chartop/reuse.py",
          "price protocol": "scripts/chartop_price_squares.py",
          "box payload": "scripts/chartop_price_box_payload.py",
          "box runner": "scripts/chartop_price_box_run.sh"}
#: pinned already by earlier registrations; each hash must agree with every pin file that lists it
ALREADY_PINNED = {"reads E4 from": "atlas/chartop/gates_forms.py",
                  "and through it": "atlas/chartop/gates.py",
                  "U's form from": "atlas/chartop/gates_hybrid.py",
                  "layout helpers": "atlas/chartop/benchmarks.py",
                  "dry-run problems": "atlas/chartop/benchmarks_workbench.py",
                  "U's code": "atlas/chartop/forms.py",
                  "the host": "atlas/chartop/host.py"}
#: recorded beside, by hash: what H2's arm is built with that no registration owns, the
#: evaluation's utilities, and the dry run that passed
DRY_RUN = "out/chartop/dryrun-squares-20261108-181228.json"
RECORDED_BESIDE = {"U's builder": "scripts/chartop_evaluate_hybrid.py",
                   "the stand-in": "scripts/chartop_forms_diagnose.py",
                   "U's adapter": "atlas/chartop/expert.py",
                   "the envelope": "atlas/chartop/data.py",
                   "utilities": "scripts/chartop_evaluate.py",
                   "the dry run": DRY_RUN}
REGISTRATION = os.path.join(ROOT, "out", "chartop", "registered-squares.txt")
BEFORE = "any expert but the classical one read the new problems (the one evaluation's script not yet written)"


def every_path() -> list:
    return [*PINNED.values(), GS.CLASSICAL_RECORD, *ALREADY_PINNED.values(), *RECORDED_BESIDE.values()]


def _code_of(src: bytes, gate_module: bool) -> bytes:
    """The file as code: for the gate module, its AST without the module docstring and the
    `REGISTERED` assignment, which record the pin; any other file, its bytes."""
    if not gate_module:
        return src
    body = ast.parse(src).body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    body = [n for n in body if not (isinstance(n, ast.Assign)
                                    and any(getattr(t, "id", None) == "REGISTERED" for t in n.targets))]
    return ast.dump(ast.Module(body=body, type_ignores=[])).encode()


def same_as_run(rel: str, commit: str, root: str = ROOT) -> bool:
    """Whether ``rel`` is the file ``commit`` held, line endings normalised; the gate module
    apart from its docstring and `REGISTERED` (`_code_of`)."""
    then = subprocess.run(["git", "-C", ROOT, "-c", "core.autocrlf=false", "show", f"{commit}:{rel}"],
                          capture_output=True).stdout.replace(b"\r\n", b"\n")
    now = open(os.path.join(root, *rel.split("/")), "rb").read().replace(b"\r\n", b"\n")
    gate = rel == PINNED["gate module"]
    return bool(then) and _code_of(then, gate) == _code_of(now, gate)


def check_pin(path: str = REGISTRATION, root: str = ROOT) -> dict:
    """{file: sha256} for every file the pin records, or SystemExit if one is missing or changed.
    The one evaluation runs this first."""
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing: the registration is not pinned")
    text = open(path, encoding="ascii").read()
    out = {}
    for rel in every_path():
        m = re.search(re.escape(rel) + r"\s+sha256 ([0-9a-f]{64})", text)
        if not m:
            raise SystemExit(f"{path} does not pin {rel}")
        h = RP.file_hash(os.path.join(root, *rel.split("/")))
        if h != m.group(1):
            raise SystemExit(f"{rel} is not the registered file: sha256 {h[:16]}... against {m.group(1)[:16]}...")
        out[rel] = h
    return out


def _refusals_before_pin() -> list:
    """Why the scope may not be pinned; empty when it may."""
    why = []
    rec = json.load(open(RECORD, encoding="utf-8"))
    if not (rec["experts_run"] == ["classical"] and tuple(rec["seeds"]) == GS.SEEDS
            and len(rec["layouts"]) == GS.N_LAYOUTS):
        why.append("the classical record is not of the registered seeds, or ran another expert")
    for lay in rec["layouts"]:
        if lay["spec"] != BS.draw(lay["seed"]) or not lay["conforming"]:
            why.append(f"seed {lay['seed']}: not the generator's layout, or a seam is not conforming")
        for r in lay["problems"]:
            if not r["floor_meets_e4"] or r["refused"]:
                why.append(f"seed {lay['seed']} {r['problem']}: the floor fails E4 or a piece is refused")
            b = BS.build(lay["spec"], GS.BENCHMARK_N, r["problem"])
            if RP.fingerprint(b) != r["fingerprint"]:
                why.append(f"seed {lay['seed']} {r['problem']}: not the problem the record measured")
        if lay["picks"] != GS.rule_pieces2(BS.build(lay["spec"], GS.BENCHMARK_N, "steady").layout(GS.KIND, GS.M_PORTS)):
            why.append(f"seed {lay['seed']}: rule R2's picks are not the record's")
    pins = {f: open(os.path.join(ROOT, "out", "chartop", f), encoding="utf-8").read()
            for f in sorted(os.listdir(os.path.join(ROOT, "out", "chartop"))) if re.fullmatch(r"registered.*\.txt", f)}
    for rel in ALREADY_PINNED.values():
        found = [re.search(re.escape(rel) + r"\s+sha256 ([0-9a-f]{64})", t) for t in pins.values()]
        found = [m.group(1) for m in found if m]
        if not found or any(h != RP.file_hash(os.path.join(ROOT, rel)) for h in found):
            why.append(f"{rel} is not as the earlier registrations pinned it")
    run = json.load(open(os.path.join(ROOT, *DRY_RUN.split("/")), encoding="utf-8"))
    if not (run["passes"] is True and run["on_price_cpu"] and run["machine"]["cpu"] == GS.PRICE_CPU):
        why.append("the dry run recorded beside did not pass on the pricing CPU")
    code = str(run.get("code") or "")
    for rel in [*PINNED.values(), *ALREADY_PINNED.values(), *list(RECORDED_BESIDE.values())[:-1]]:
        if not re.fullmatch(r"[0-9a-f]{40}", code) or not same_as_run(rel, code):
            why.append(f"{rel} is not the file the passing dry run ran ({code[:7]})")
    return why


def pin(before: str = BEFORE, path: str = REGISTRATION) -> int:
    """Write the registration (refusing to overwrite one, or to pin what `_refusals_before_pin`
    refuses)."""
    if os.path.exists(path):
        raise SystemExit(f"{path} exists; a registration is not overwritten")
    why = _refusals_before_pin()
    if why:
        raise SystemExit("not pinning: " + "; ".join(why[:5]))
    rec = json.load(open(RECORD, encoding="utf-8"))
    run = json.load(open(os.path.join(ROOT, *DRY_RUN.split("/")), encoding="utf-8"))
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    h = lambda rel: RP.file_hash(os.path.join(ROOT, *rel.split("/")))  # noqa: E731
    env = data.ENVELOPE
    u = ", ".join(f"{k}={v!r}" for k, v in GS.U_FORM.items() if k != "seed")
    lines = ["THE CHART OPERATOR, THE REGISTRATION AFTER SECTION 11.17 (CO-D, RULE R2'S HYBRID H2: U'S FORM ON SQUARES)",
             "",
             "written            %s" % dt.datetime.now().astimezone().isoformat(timespec="seconds"),
             "before             %s" % before,
             "repository at      %s (the commit adding this file follows it)" % head]
    lines += ["%-18s %s  sha256 %s" % (label, rel, h(rel)) for label, rel in PINNED.items()]
    lines.append("classical record   %s  sha256 %s" % (GS.CLASSICAL_RECORD, h(GS.CLASSICAL_RECORD)))
    lines += ["%-18s %s  sha256 %s  (as the earlier registrations pinned it)" % (label, rel, h(rel))
              for label, rel in ALREADY_PINNED.items()]
    lines += ["%-18s %s  sha256 %s" % (label, rel, h(rel)) for label, rel in RECORDED_BESIDE.items()]
    lines += ["registered as      %s" % GS.REGISTERED,
              "",
              "THE ARM, H2, BY VALUE (no weights)",
              "  the rule           rule_pieces2: the classical solve on every piece is_square refuses; U's form",
              "                     on every piece whose chart is affine onto a square of one material: four",
              "                     equal sides and four right angles to %g of the side, log kappa_hat zero" % GS.SQUARE_TOL,
              "  U's form           forms.KernelExpert(%s)," % u,
              "                     built after torch.manual_seed(%d), zero heads, the networks replaced by" % GS.U_FORM["seed"],
              "                     a stand-in answering their zeros",
              "  then               the coupled solve (host.Layout.couple)",
              "  beside it, H2r     H2's picks; each square's classical response computed once per class",
              "                     (resolution, Fourier number to 12 significant digits, uniform source)",
              "",
              "THE NEW PROBLEMS: benchmarks_squares' layouts from seeds %s, problems %s, at n = %d"
              % (", ".join(str(s) for s in GS.SEEDS), " and ".join(BS.PROBLEMS), GS.BENCHMARK_N),
              "  read only by the classical superelement and the full-domain solve; the layouts, R2's",
              "  classical pieces and the classical floors (energy, as registered and lowered by %g K):"
              % -GS.DATUM_SHIFT_K]
    for lay in rec["layouts"]:
        s = lay["spec"]
        fl = "; ".join("%s %.4g%%, %.4g%%" % (r["problem"], 100 * r["frames"]["registered"]["truncation_energy_rel"],
                                             100 * r["frames"]["lowered"]["truncation_energy_rel"]) for r in lay["problems"])
        lines.append("    %d  %-32s %d x %d, %2d pieces, %d classical %s; dt %g s; %s"
                     % (lay["seed"], s["feature"][:32], s["W"], s["H"], lay["pieces"], len(lay["picks"]),
                        lay["picks"], lay["dt"], fl))
    lines += ["  skipped            %s" % "; ".join(f"{x['seed']} ({x['why'][:90]})" for x in rec["skipped"]),
              "",
              "THE PRICE'S PROTOCOL, BY VALUE",
              "  where              a rented whole machine (every GPU of its host), CPU %s" % GS.PRICE_CPU,
              "  threads            %d torch threads and %d BLAS threads (%s set before numpy loads)," % (
                  GS.PRICE_THREADS, GS.BLAS_THREADS, ", ".join(GS.BLAS_VARIABLES)),
              "                     checked with threadpoolctl before a run times anything",
              "  each time          the best of %d, the arms in turn: %s" % (GS.PRICE_REPS, ", ".join(GS.PRICE_ORDER)),
              "  a measurement      counts if its run held AC at every %g s sample with no gap over %g s, and its"
              % (GS.SAMPLE_S, GS.MAX_GAP_S),
              "                     problem's classical best time is at most %g ms a piece; load and Kernel-Power" % GS.MAX_MS_PER_PIECE,
              "                     events recorded, not judged",
              "  runs               each in its own process, every run kept, until every problem has %d counted, at" % GS.PRICE_RUNS,
              "                     most %d; each problem's price the median of its first %d counted"
              % (GS.PRICE_RUNS_MAX, GS.PRICE_RUNS),
              "  the A/A check      %s priced beside in every run; their H2 medians within %g of 1, or PR2 is"
              % (", ".join(" ".join(c[1:]) for c in GS.AA_CONTROLS), GS.AA_TOLERANCE),
              "                     not judged",
              "  the dry run        %s: passed (every problem %d of %d counted, the A/A check within %.3f," % (
                  DRY_RUN, min(run["tally"].values()), len(run["runs"]), run["aa"]["worst"]),
              "                     H2 %.3f, H2r %.3f), commit %s" % (run["geometric_mean_of_medians"]["H2"],
                                                                     run["geometric_mean_of_medians"]["H2r"], run["code"][:7]),
              "",
              "THE TRAINING ENVELOPE (data.ENVELOPE), as gates_forms",
              "  sup|mu| <= %g, max log J - min log J <= %.4g, |grad log J| + |grad mu| <= %g,"
              % (env.k_mu, env.l_J, env.s_grad),
              "  |log kappa_hat| <= %g, Fo in [%g, %g] or steady" % (env.l_kappa, env.fo_min, env.fo_max),
              "",
              GS.text(),
              ""]
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write("\n".join(lines))
    print("registered ->", os.path.relpath(path, ROOT))
    for label, rel in list(PINNED.items()) + [("classical record", GS.CLASSICAL_RECORD)]:
        print("  %-18s %s %s" % (label, rel, h(rel)[:16]))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pin", action="store_true",
                    help="write out/chartop/registered-squares.txt (only with the owner's approval)")
    ap.add_argument("--before", default=BEFORE)
    args = ap.parse_args(argv)
    if args.pin:
        return pin(args.before)
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
