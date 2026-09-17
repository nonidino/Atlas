"""W306 -- the sign structure `L4/E7/passivity` never reads, swept across the vault.

`probe._fill_diagnostics` sets

    passivity_defect = abs(lam_min) if lam_min < -pass_tol else 0.0

and **never reads `lam_max`**.  So two different things report the same defect:

    ONE-SIGNED NEGATIVE   every mode damped, the whole operator a global sign
                          away from passive.  The interface problem S lam = chi
                          is indifferent to that sign, because chi flips with S.
                          It is a DECLARATION inconsistency -- an undeclared or
                          wrong `effort_normal` -- not a physical defect.
    MIXED                 lam_min < 0 < lam_max: a genuinely amplified interface
                          mode, and the thing E7 is actually about.

Tier 77 found the rocket's b-c seam is the first kind.  **The question this
script exists to answer is whether it is the ONLY one**, because a rule change
that serves one case study is a fix and not a rule, and should be described as
one.

Every buildable case, every seam it probes, classified. ASCII output only.

    python scripts/w306_sign_structure.py --json out/w306.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

#: Named rather than discovered, for `w301_saturation_sweep`'s reason: a sweep
#: that quietly drops the cases it could not build reports a rate about the
#: cases it could.
CASES = [
    ("wind_farm", {}),
    ("thermal_seam", {}),
    ("rocket", {"declarations": "real"}),
    ("thermal_strain", {}),
    ("brake_thermal", {}),
    ("cooling_loop", {}),
    ("powertrain", {}),
    ("car_graph", {}),
]


def classify(S, tol_scale=None):
    """one-signed-positive | one-signed-negative | mixed | (near-)zero."""
    sym = 0.5 * (np.asarray(S, float) + np.asarray(S, float).T)
    ev = np.linalg.eigvalsh(sym)
    scale = float(np.abs(sym).max()) if tol_scale is None else float(tol_scale)
    tol = max(sym.shape) * np.finfo(float).eps * max(scale, 1.0)
    lo, hi = float(ev.min()), float(ev.max())
    if abs(lo) <= tol and abs(hi) <= tol:
        kind = "zero"
    elif lo > -tol:
        kind = "one-signed-positive"
    elif hi < tol:
        kind = "one-signed-negative"
    else:
        kind = "mixed"
    return kind, lo, hi, tol


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", metavar="PATH")
    a = ap.parse_args(argv)
    from atlas.compiler import compile_scheme

    out = {"script": "w306_sign_structure.py", "date": "2026-09-17"}
    rows, skipped = [], []
    print("=" * 104)
    print("W306 -- is the rocket the ONLY seam where a passivity defect is a sign")
    print("        convention rather than an amplified mode?")
    print("=" * 104)
    print("%-16s %-22s %-22s %-13s %-13s %s"
          % ("case", "seam", "sign structure", "lam_min", "lam_max", "defect"))
    print("-" * 104)
    for name, kwargs in CASES:
        try:
            mod = importlib.import_module("atlas.cases." + name)
            built = mod.build(**kwargs)
            graph = built[0] if isinstance(built, tuple) else built
            t0 = time.perf_counter()
            res = compile_scheme(graph)
            el = time.perf_counter() - t0
        except Exception as exc:                          # noqa: BLE001
            skipped.append({"case": name,
                            "why": "%s: %s" % (type(exc).__name__, str(exc)[:110])})
            print("%-16s SKIPPED  %s: %s"
                  % (name, type(exc).__name__, str(exc)[:60]))
            continue
        if not res.seam_operators:
            skipped.append({"case": name, "why": "compiles without probing a seam"})
            print("%-16s (no probed seams)" % name)
            continue
        for seam, op in sorted(res.seam_operators.items()):
            kind, lo, hi, tol = classify(op.S)
            rows.append({"case": name, "seam": seam, "kind": kind,
                         "lam_min": lo, "lam_max": hi, "tol": tol,
                         "defect": op.passivity_defect,
                         "beta": op.beta, "dim_M": op.dim_M,
                         "effort_normal": getattr(op, "effort_normal", "") or "",
                         "seconds": el})
            flag = ""
            if (op.passivity_defect or 0.0) > 0.0 and kind == "one-signed-negative":
                flag = "   <-- MISREPORTED: a sign, not a mode"
            print("%-16s %-22s %-22s %-13.5g %-13.5g %-10.5g%s"
                  % (name, seam[:22], kind, lo, hi,
                     op.passivity_defect or 0.0, flag))
    out["rows"], out["skipped"] = rows, skipped

    defect = [r for r in rows if (r["defect"] or 0.0) > 0.0]
    mis = [r for r in defect if r["kind"] == "one-signed-negative"]
    real = [r for r in defect if r["kind"] == "mixed"]
    print()
    print("=" * 104)
    print("  %d seams classified across %d cases; %d cases skipped and named."
          % (len(rows), len({r["case"] for r in rows}), len(skipped)))
    print("  %d report a passivity defect." % len(defect))
    print("    %d of those are MIXED        -- a genuinely amplified mode, which is"
          " what E7 is about" % len(real))
    print("    %d of those are ONE-SIGNED   -- a global sign the interface solve is"
          " indifferent to" % len(mis))
    for r in mis:
        print("        %s / %s  (effort_normal=%r)"
              % (r["case"], r["seam"], r["effort_normal"]))
    out["n_defect"], out["n_mixed"], out["n_one_signed"] = (
        len(defect), len(real), len(mis))
    print()
    if len(mis) <= 1:
        print("  <= 1 misreported seam: **W306 is real but NOT a vault-wide rule**,")
        print("  and a change to what L4/E7/passivity DECIDES is not justified by")
        print("  this evidence. Ship the diagnostic; say the scope out loud.")
    else:
        print("  More than one: the rule change has a case beyond the rocket.")

    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=str)
        print()
        print("artifact written to %s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
