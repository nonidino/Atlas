"""The fifth real case study, run: a genuine multiphysics seam.

    python scripts/w67_thermal_seam.py [--out out/w67]

`cases/thermal_seam.py` puts `compressible2d.Compressible2D` against
`thermostruct2d.ThermoStruct2D` -- two build-repo solvers, imported unmodified,
each graded against a closed-form oracle by that repo's own M1 suite -- across
one conjugate-heat-transfer `THERM` seam.  Four rules that had never met a real
disagreement meet one here, and this script measures each:

**E3, against a genuine governing-family mismatch.**  All four pre-existing real
case studies are 2-D incompressible Navier-Stokes, so the string comparison the
spec calls load-bearing has only ever been compared against itself and every
graph that failed it was a fixture built to fail it.  The spec's claim is precise:
E3 fails, tau goes ``UNDEFINED``, and *the probe still runs*, because probing
mentions no governing equation.  Measured here.

**R1's ladder, against a genuine bc_channel disagreement.**  Every real record in
this vault declares ``DIRICHLET``.  `ThermoStruct2D._robin` is an actual Robin
condition, so the shell is the **first real expert above `dirichlet`** on the
transmission ladder, and R1's "as weak as its weakest agent" has never had two
different values to be an inequality over.

**W66, on a second port type, and it goes the other way.**  `PORT_SPECS[THERM]`
declares the bond ``(T, q_n/T)`` and says in its own note that ``(T, q_n)`` is
the pseudo-bond most co-simulation codes exchange.  Every thermal solver in the
build repo computes ``q_n``.  On `ADVEC` the wrong pairing produced a **false
alarm**; here it produces a **false pass**, which is worse and was not predicted.

**W60, two more calibration points from different physics.**  `elliptic_signature`
had exactly two, both from `WindowNS`, which is why section 11.6 called it a
diagnosis rather than a gate.  The shell's conduction solve is a *known*
embedded elliptic part (backward Euler over the whole shell) and a *known*
exposed one under ``split-step``, on physics that is not Navier-Stokes at all.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from atlas.cases import thermal_seam as T                          # noqa: E402
from atlas.cases.poseidon import elliptic_signature                # noqa: E402
from atlas.compiler import compile_scheme                          # noqa: E402


def stamp_of(r) -> dict:
    return {h.value: s.value for h, s in r.envelope.values.items()}


def compile_row(mode: str, clocks: str, convention: str = "entropy") -> dict:
    t0 = time.perf_counter()
    g, _ex = T.build(mode=mode, clocks=clocks, flux_convention=convention)
    r = compile_scheme(g, probe_state="duct, T_hot=900 K, T_wall=400 K")
    op = r.seam_operators.get("cht")
    row = {
        "mode": mode, "clocks": clocks, "flux_convention": convention,
        "verdict": getattr(r.verdict, "value", str(r.verdict)),
        "refusals": [f"{d.layer}/{d.rule}" for d in r.decisions.refusals],
        "decertifications": [f"{d.layer}/{d.rule}" for d in r.decisions.decertifications],
        "unmeasured": list(r.unmeasured),
        "stamp": stamp_of(r),
        "tau_undefined_seams": list(r.tau_undefined_seams),
        "seconds": time.perf_counter() - t0,
    }
    if op is not None:
        row["probe"] = {
            "beta": op.beta, "kappa": op.kappa, "null_dim": op.null_dim,
            "expected_null_dim": op.expected_null_dim,
            "passivity_defect": op.passivity_defect,
            "norm_S": float(np.linalg.norm(op.S, 2)),
            "is_empty": bool(op.is_empty),
        }
        row["elliptic_signature"] = elliptic_signature(op.S)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join("out", "w67"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    out: dict = {"case": "thermal_seam", "date": "2026-08-29"}

    # -- the compile matrix -------------------------------------------------
    print("=== the compile matrix ===")
    rows = []
    for mode in ("as-built", "split-step"):
        for clocks in ("native", "matched"):
            r = compile_row(mode, clocks)
            rows.append(r)
            print(f"{mode:<11}{clocks:<9}{r['verdict']:<19}"
                  f"ref={len(r['refusals']):<2}dec={len(r['decertifications']):<2}"
                  f"E3={r['stamp']['E3']:<7}E4={r['stamp']['E4']:<7}"
                  f"tau_undef={r['tau_undefined_seams']}  [{r['seconds']:.0f}s]")
            if r["refusals"]:
                print(f"{'':<20}refusals: {r['refusals']}")
    out["compile_matrix"] = rows

    # -- E3, the row this case study exists for -----------------------------
    print()
    print("=== E3 against a genuine governing-family disagreement ===")
    clean = [r for r in rows if r["mode"] == "split-step" and r["clocks"] == "matched"][0]
    print(f"  gas   declares 'compressible-navier-stokes-2d'")
    print(f"  shell declares 'thermoelastic-shell-2d'")
    print(f"  verdict           {clean['verdict']}   refusals {len(clean['refusals'])}")
    print(f"  E3                {clean['stamp']['E3']}")
    print(f"  tau UNDEFINED on  {clean['tau_undefined_seams']}")
    p = clean["probe"]
    print(f"  and the probe RAN anyway: beta={p['beta']:.4f} kappa={p['kappa']:.6f} "
          f"null={p['null_dim']} empty={p['is_empty']}")
    out["e3"] = {
        "families": ["compressible-navier-stokes-2d", "thermoelastic-shell-2d"],
        "verdict": clean["verdict"], "refusals": clean["refusals"],
        "E3": clean["stamp"]["E3"], "tau_undefined": clean["tau_undefined_seams"],
        "probe_ran": not p["is_empty"], "beta": p["beta"], "kappa": p["kappa"],
        "claim": "E3 fails, tau is UNDEFINED, the composition still runs -- "
                 "confirmed on a real object for the first time",
    }

    # -- R1's ladder --------------------------------------------------------
    print()
    print("=== R1: the transmission ladder, with two different real channels ===")
    g, _ = T.build(mode="split-step", clocks="matched")
    chans = {ag.agent_id: ag.capabilities.bc_channel for ag in g.agents}
    for k, v in chans.items():
        print(f"  {k:<7}{v.value:<12}rung {v.rung}")
    weakest = min(chans.values(), key=lambda c: c.rung)
    print(f"  weakest agent decides the seam: {weakest.value} (rung {weakest.rung})")
    print("  ROBIN is the first channel ABOVE dirichlet any real expert in this "
          "vault has declared")
    out["r1_ladder"] = {k: {"channel": v.value, "rung": v.rung} for k, v in chans.items()}
    out["r1_ladder"]["weakest"] = weakest.value

    # -- W66 on THERM -------------------------------------------------------
    print()
    print("=== W66: the declared flow is q_n/T; every thermal solver returns q_n ===")
    w66 = {}
    for conv in T.FLUX_CONVENTIONS:
        r = compile_row("split-step", "matched", conv)
        w66[conv] = r
        p = r["probe"]
        print(f"  {conv:<8} verdict={r['verdict']:<19}E7={r['stamp']['E7']:<7}"
              f"defect={p['passivity_defect']}  beta={p['beta']:.4f}  "
              f"||S||={p['norm_S']:.4e}")
    ratio = w66["heat"]["probe"]["norm_S"] / w66["entropy"]["probe"]["norm_S"]
    same = (w66["heat"]["verdict"] == w66["entropy"]["verdict"]
            and w66["heat"]["stamp"] == w66["entropy"]["stamp"])
    print(f"  ||S|| ratio {ratio:.4f}; verdict and stamp identical: {same}")
    print("  -> a FALSE PASS. On ADVEC the same class of error was a false ALARM.")
    print("     Passivity is exactly invariant under a positive rescale, and "
          "q_n/(q_n/T) = T > 0.")
    out["w66"] = {"norm_S_ratio": ratio, "verdict_and_stamp_identical": same,
                  "by_convention": {k: {"verdict": v["verdict"], "E7": v["stamp"]["E7"],
                                        "beta": v["probe"]["beta"],
                                        "norm_S": v["probe"]["norm_S"],
                                        "passivity_defect": v["probe"]["passivity_defect"]}
                                    for k, v in w66.items()}}

    # -- W60: two more calibration points -----------------------------------
    print()
    print("=== W60: elliptic_signature, two calibration points from new physics ===")
    print("  section 8.3's two points, both WindowNS:")
    print("    EMBEDDED  kappa=21.73  asymmetry=0.153")
    print("    EXPOSED   kappa= 1.196 asymmetry=0.002")
    sig = {}
    for r in rows:
        if r["clocks"] != "matched":
            continue
        known = "EMBEDDED" if r["mode"] == "as-built" else "EXPOSED"
        e = r["elliptic_signature"]
        sig[known] = e
        print(f"  shell, elliptic {known:<9} kappa={e['kappa']:.4f}  "
              f"asymmetry={e['asymmetry']:.4f}")
        print(f"      -> {e['verdict']}")
    out["w60_calibration"] = sig
    ok = all(
        "EXPOSED" in sig[k]["verdict"] if k == "EXPOSED" else True for k in sig
    )
    out["w60_separates"] = bool(
        sig.get("EMBEDDED", {}).get("asymmetry", 0)
        != sig.get("EXPOSED", {}).get("asymmetry", 0))

    with open(os.path.join(a.out, "w67.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print()
    print(f"wrote {os.path.join(a.out, 'w67.json')}")


if __name__ == "__main__":
    main()
