"""The registration after the forms' one evaluation (`atlas.chartop.gates_prior`): the
held-out benchmark problems, measured with the classical solver only.

Accepted by the owner on 2026-11-07 ([[chart-operator-kernel-exact-forms]] section
11.7), with two choices settled in chat: a fresh test split at seed base 5,000,000,
and new held-out benchmark problems built and frozen before training.  No learned
expert -- trained or untrained -- reads the new problems here; only the classical
superelement and the full-domain solve do, which are properties of the problems and
the port basis, not of any network.

``--classical`` (the default) builds the six held-out problems (`benchmarks`'
``variant="held-out"``) and, as the control that the defaults did not move, the six
seen ones.  For each it records the pieces, the envelope's refusals, the Fourier
numbers' range against the training generator's, the Dirichlet data's range, the
classical floor (E4's truncation term and its maximum error over the range), and a
fingerprint of the problem's definition.  The seen problems' floors must equal the
one evaluation's record bit for bit.  It writes
``out/chartop/registration-prior-heldout.json`` and prints a table of ten lines or
fewer.  Pinning is a separate step, run only with the owner's approval.
"""

from __future__ import annotations

import argparse
import hashlib
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
import chartop_registry as registry  # noqa: E402
from atlas.chartop import benchmarks as bm, data, gates_forms as GF, host as hs  # noqa: E402

OUT = os.path.join(ROOT, "out", "chartop")
EVALUATION = os.path.join(OUT, "evaluation-forms-20261019-205923.json")


def build(name: str, problem: str, variant: str | None, frame: str = "registered") -> bm.Bench:
    kw = {"envelope": data.ENVELOPE} if name == "s-channel" else {}
    if variant:
        kw["variant"] = variant
    b = bm.build(name, GF.BENCHMARK_N, problem, **kw)
    return b if frame == "registered" else bm.datum_shift(b, GF.DATUM_SHIFT_K)


def fingerprint(b: bm.Bench) -> str:
    """sha256 over the problem's definition: each piece's label, kappa_ref and Fourier
    number, its node coordinates and its Dirichlet data, every float rounded to nine
    significant digits (so that libm's last bit, which differs between platforms, does
    not split one problem into two)."""
    h = hashlib.sha256()
    for i, p in enumerate(b.problems):
        h.update(p.label.encode("utf-8"))
        h.update(f"{float(p.kappa_ref):.9g}".encode())
        h.update(("steady" if p.fo is None else f"{p.fo:.9g}").encode())
        h.update(" ".join(f"{v:.9g}" for v in np.asarray(p.chart.X, float).ravel()).encode())
        for s in range(4):
            kind, val = b.bc(i, s, hs._side_points(p, s))
            h.update(kind.encode())
            if val is not None:
                h.update(" ".join(f"{v:.9g}" for v in np.asarray(val, float)).encode())
    return h.hexdigest()


def measure(name: str, problem: str, variant: str | None) -> dict:
    out = {"name": name, "problem": problem, "variant": variant, "frames": {}}
    for frame in ("registered", "lowered"):
        b = build(name, problem, variant, frame)
        L = b.layout(GF.KIND, GF.M_PORTS)
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
            out.update(pieces=len(b.problems), fingerprint=fingerprint(b),
                       kappa_ref=sorted({float(p.kappa_ref) for p in b.problems}),
                       fo_range=[min(fos), max(fos)] if fos else None,
                       dirichlet_range=[min(dvals), max(dvals)])
    return out


PINNED = {"gate module": "atlas/chartop/gates_prior.py",
          "reads the bars of": "atlas/chartop/gates_forms.py",
          "and through it": "atlas/chartop/gates.py",
          "held-out problems": "atlas/chartop/benchmarks.py"}
REGISTRATION = os.path.join(OUT, "registered-prior.txt")
RECORD = os.path.join(OUT, "registration-prior-heldout.json")


def file_hash(path: str) -> str:
    """sha256 with line endings normalised, as `scripts/chartop_register_forms.py` hashes."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def treatments() -> dict:
    """The registered treatments by value: the prior's definition and the numbers it gives
    at n = 64 (the trace of Sigma, and the learned block's largest and smallest eigenvalues
    before the floor), and the spectral path's size."""
    from atlas.chartop import forms as fm, model as md
    S = fm.trace_covariance("generator-field", GF.KIND, 64, GF.M_PORTS)
    L = list(fm.learned_columns(GF.KIND, GF.M_PORTS))
    lam = np.linalg.eigvalsh(S[np.ix_(L, L)])
    return {"mode_priors": list(fm.MODE_PRIORS), "mode_prior_floor": fm.MODE_PRIOR_FLOOR,
            "frequencies": dict(data.FREQUENCIES), "trace_sigma_n64": float(np.trace(S)),
            "learned_lambda_max_n64": float(lam.max()), "learned_lambda_min_n64": float(lam.min()),
            "spectral_k": md.SPECTRAL_K, "spectral_width": md.SPECTRAL_WIDTH}


def pin(before: str) -> int:
    """Write ``out/chartop/registered-prior.txt`` (refusing to overwrite one)."""
    import datetime as dt
    import subprocess
    from atlas.chartop import gates_prior as GP
    if os.path.exists(REGISTRATION):
        raise SystemExit(f"{REGISTRATION} exists; a registration is not overwritten")
    rec = json.load(open(RECORD, encoding="utf-8"))
    if not (rec["seen_floor_equals_evaluation"] and rec["held_out_fo_within_training"]
            and rec["held_out_refusals"] == 0):
        raise SystemExit("the classical record does not pass its own controls; not pinning")
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    env = data.ENVELOPE
    tm = treatments()
    lines = ["THE CHART OPERATOR, THE REGISTRATION AFTER THE FORMS' EVALUATION (CO-D, ARMS P AND PG)",
             "",
             "written            %s" % dt.datetime.now().astimezone().isoformat(timespec="seconds"),
             "before             %s" % before,
             "repository at      %s (the commit adding this file follows it)" % head]
    for label, rel in PINNED.items():
        lines.append("%-18s %s  sha256 %s" % (label, rel, file_hash(os.path.join(ROOT, rel))))
    lines += ["classical record   out/chartop/registration-prior-heldout.json  sha256 %s" % file_hash(RECORD),
              "registered as      %s" % GP.REGISTERED,
              "",
              "SPLITS (example i of a split has seed base + i)"]
    for k, v in GF.SEED_BASE.items():
        if k != "test":
            lines.append("  %-16s seed base %d, as gates_forms" % (k, v))
    lines += ["  test, fresh      %d examples, seeds %d..%d, drawn only at the one evaluation"
              % (GP.N_TEST, GP.TEST_SEEDS[0], GP.TEST_SEEDS[1]),
              "  (the test split at %d was evaluated once, 2026-10-19, and is not read again)"
              % GF.SEED_BASE["test"],
              "",
              "THE HELD-OUT PROBLEMS: benchmarks' variant %r of %s, problems %s, at n = %d"
              % (GP.HELD_OUT_VARIANT, ", ".join(GP.BENCHMARK_NAMES), " and ".join(GP.BENCHMARK_PROBLEMS),
                 GP.BENCHMARK_N),
              "  materials by role  %s" % "; ".join(f"{k}: " + ", ".join(f"{r} {m}" for r, m in v.items())
                                                   for k, v in bm.HELD_OUT_MATERIALS.items()),
              "  the ends' data     hot %g + %g cos(pi s), cold %g + %g cos(2 pi s), s from 0 to 1 along each end"
              % (bm.T_HOT, bm.HELD_OUT_HOT_AMPLITUDE, bm.T_COLD, bm.HELD_OUT_COLD_AMPLITUDE),
              "  read only by the classical superelement and the full-domain solve; classical floors:"]
    for r in rec["held_out"]:
        f = r["frames"]["registered"]
        lines.append("    %-13s %-6s energy %.4g%%, max %.3g of the range" % (
            r["name"], r["problem"], 100 * f["truncation_energy_rel"], f["floor_max_over_range"]))
    lines += ["  seen, beside       gates_forms' six problems, read post hoc in section 11; reported, not judged",
              "",
              "THE TREATMENTS, BY VALUE (checked by value by the test)",
              "  mode prior         %s: C(x, x') = 1/3 + 0.75 mean over ell of exp(-|x - x'|^2 / (2 ell^2)),"
              % tm["mode_priors"][0],
              "                     ell in %s; each block floored at %g of its largest eigenvalue"
              % (sorted(tm["frequencies"].values()), tm["mode_prior_floor"]),
              "                     at n = 64: tr Sigma = %.12g; learned block's eigenvalues %.12g to %.12g"
              % (tm["trace_sigma_n64"], tm["learned_lambda_max_n64"], tm["learned_lambda_min_n64"]),
              "  spectral path      %d cosine modes per direction on %d channels at the coarsest level"
              % (tm["spectral_k"], tm["spectral_width"]),
              "",
              "THE TRAINING ENVELOPE (data.ENVELOPE), as gates_forms",
              "  sup|mu| <= %g, max log J - min log J <= %.4g, |grad log J| + |grad mu| <= %g,"
              % (env.k_mu, env.l_J, env.s_grad),
              "  |log kappa_hat| <= %g, Fo in [%g, %g] or steady" % (env.l_kappa, env.fo_min, env.fo_max),
              "",
              GP.text(),
              "",
              "the measures and judgements are gates_forms', applied once at the evaluation; the arms'",
              "batches, learning rate and validation cadence are S''s, written into the plan after",
              "E5's pricing, before training.",
              ""]
    with open(REGISTRATION, "w", encoding="ascii", newline="\n") as fh:
        fh.write("\n".join(lines))
    print("registered ->", os.path.relpath(REGISTRATION, ROOT))
    for label, rel in PINNED.items():
        print("  %-18s %s %s" % (label, rel, file_hash(os.path.join(ROOT, rel))[:16]))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pin", action="store_true",
                    help="write out/chartop/registered-prior.txt (only with the owner's approval)")
    ap.add_argument("--before", default="any network of these arms was trained (no training run, "
                                        "no rental; E5 not yet priced)")
    args = ap.parse_args(argv)
    if args.pin:
        return pin(args.before)
    t0 = time.perf_counter()
    rec = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "experts_run": ["classical"],
           "held_out": [], "seen": []}
    for name in GF.HELD_OUT_BENCHMARKS:
        for problem in GF.BENCHMARK_PROBLEMS:
            rec["held_out"].append(measure(name, problem, "held-out"))
            rec["seen"].append(measure(name, problem, None))
    # the control: the seen problems' floors are the one evaluation's, bit for bit
    evalrec = json.load(open(EVALUATION, encoding="utf-8"))
    ref = {(r["name"], r["problem"]): r["frames"]["registered"]["truncation_energy_rel"]
           for r in evalrec["benchmarks"]}
    rec["seen_floor_equals_evaluation"] = all(
        r["frames"]["registered"]["truncation_energy_rel"] == ref[(r["name"], r["problem"])]
        for r in rec["seen"])
    fo_lo, fo_hi = data.FO_RANGE
    rec["held_out_fo_within_training"] = all(
        r["fo_range"] is None or (fo_lo <= r["fo_range"][0] and r["fo_range"][1] <= fo_hi)
        for r in rec["held_out"])
    rec["held_out_refusals"] = sum(len(r["frames"][f]["refusals"]) for r in rec["held_out"]
                                   for f in ("registered", "lowered"))
    rec["elapsed_s"] = time.perf_counter() - t0
    path = os.path.join(OUT, "registration-prior-heldout.json")
    registry.write_atomic(path, json.dumps(ev0.strip(rec), indent=1).encode("utf-8"))
    print("HELD-OUT  problem              floor E%   floor max/range  Fo range        data (K)       refusals")
    for r in rec["held_out"]:
        f = r["frames"]["registered"]
        fo = "-" if r["fo_range"] is None else f"{r['fo_range'][0]:.3g}-{r['fo_range'][1]:.3g}"
        print(f"          {r['name'] + ' ' + r['problem']:20s} {100 * f['truncation_energy_rel']:8.3g}  "
              f"{f['floor_max_over_range']:14.3g}  {fo:14s}  {r['dirichlet_range'][0]:.0f}-"
              f"{r['dirichlet_range'][1]:.0f}       {len(f['refusals'])}")
    print(f"          seen floors equal the evaluation's bit for bit: {rec['seen_floor_equals_evaluation']}; "
          f"held-out Fo within training's {data.FO_RANGE}: {rec['held_out_fo_within_training']}")
    print(f"          wrote {os.path.relpath(path, ROOT)} ({rec['elapsed_s']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
