"""W302 -- what each declaration is worth, ONE FACTOR AT A TIME.

R1's gate asks for the 49 decertifications to fall to a named residue with a
one-line reason for each survivor.  A count alone is not a result, and neither is
a count taken with five things changed at once: this vault has already paid for
"a control that varies two things is not one".

So every arm below moves exactly ONE declaration off the Tier-75 fixture, and the
ledger is the difference each one makes on its own.  The combined levels are run
last, and the gap between the sum of the parts and the whole is reported rather
than assumed to be zero.

ASCII output only -- the console is cp1252.

    python scripts/w302_declaration_ledger.py --json out/w302.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.cases import rocket
from atlas.compiler import compile_scheme
from atlas.graph import Decomposition, FluxMatching


def tally(graph):
    r = compile_scheme(graph)
    dec = Counter("%s/%s" % (d.layer, d.rule) for d in r.decisions.decertifications)
    ref = Counter("%s/%s" % (d.layer, d.rule) for d in r.decisions.refusals)
    return {"verdict": str(r.verdict),
            "n_decert": len(r.decisions.decertifications),
            "n_refuse": len(r.decisions.refusals),
            "decert": dict(dec), "refuse": dict(ref)}


def delta(base, arm):
    """What changed, per rule. Negative = records removed."""
    keys = set(base["decert"]) | set(arm["decert"])
    d = {k: arm["decert"].get(k, 0) - base["decert"].get(k, 0) for k in keys}
    keys_r = set(base["refuse"]) | set(arm["refuse"])
    r = {k: arm["refuse"].get(k, 0) - base["refuse"].get(k, 0) for k in keys_r}
    return ({k: v for k, v in sorted(d.items()) if v},
            {k: v for k, v in sorted(r.items()) if v})


# -- the arms. Each mutates the fixture graph in exactly one way. ------------


def arm_fixture():
    return rocket.build(declarations="fixture")


def arm_known_fields():
    """`time_discretization`, `stencil_radius`, `elliptic_subsolve`,
    `substeps_per_macro_step`, `lambda_ref` -- and NOTHING else."""
    g = rocket.build(declarations="known")
    g.decomposition = Decomposition.OVERLAPPING          # put back
    g.flux_matching = FluxMatching.POINTWISE             # put back
    g.macro_dt = rocket.MACRO_DT_FIXTURE                 # put back
    for a in g.agents:
        a.capabilities.dt_native = rocket.CLOCKS_FIXTURE[a.agent_id]
    return g


def arm_no_lambda_ref():
    """`known` minus `lambda_ref`, to split L4/R2b from L1/E3."""
    g = arm_known_fields()
    for a in g.agents:
        a.capabilities.lambda_ref = None
    return g


def arm_decomposition():
    g = rocket.build(declarations="fixture")
    g.decomposition = Decomposition.NON_OVERLAPPING
    return g


def arm_clocks():
    g = rocket.build(declarations="fixture")
    g.macro_dt = rocket.MACRO_DT_YAML
    for a in g.agents:
        a.capabilities.dt_native = rocket.CLOCKS_YAML[a.agent_id]
    return g


def arm_flux_matching():
    g = rocket.build(declarations="fixture")
    g.flux_matching = FluxMatching.TIME_INTEGRATED
    return g


def arm_storage_validity():
    """The COUNTING CONTROL. Trivially-true callables on all seven agents.

    This is an instrument and not a claim: a stub's storage is fiction. It exists
    so the record count those two fields control is measured rather than
    inferred, and no verdict from this arm is a verdict about the rocket.
    """
    return rocket.build(declarations="fixture", counting_control=True)


ARMS = [
    ("known fields (time_disc, stencil, elliptic, substeps, lambda_ref)",
     arm_known_fields),
    ("  ... minus lambda_ref", arm_no_lambda_ref),
    ("decomposition NON_OVERLAPPING (P14)", arm_decomposition),
    ("clocks from the YAML (50:1, not 100:1)", arm_clocks),
    ("flux_matching TIME_INTEGRATED", arm_flux_matching),
    ("storage + validity, all 7 [COUNTING CONTROL]", arm_storage_validity),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--with-real", action="store_true",
                    help="also compile declarations='real' (loads the build repo "
                         "and marches the shell; slower)")
    a = ap.parse_args(argv)

    out = {"script": "w302_declaration_ledger.py", "date": "2026-09-17"}
    print("=" * 78)
    print("W302 -- the declaration ledger, one factor at a time")
    print("=" * 78)
    base = tally(arm_fixture())
    out["fixture"] = base
    print("BASELINE (declarations='fixture', the Tier-75 graph):")
    print("  verdict %-10s  %d refusals, %d decertifications"
          % (base["verdict"], base["n_refuse"], base["n_decert"]))
    print("  refusals: %s" % base["refuse"])
    print()

    out["arms"] = {}
    for name, fn in ARMS:
        t = tally(fn())
        dd, dr = delta(base, t)
        out["arms"][name] = dict(t, delta_decert=dd, delta_refuse=dr)
        print("-" * 78)
        print("ARM: %s" % name)
        print("  verdict %-10s  %d refusals (%+d), %d decerts (%+d)"
              % (t["verdict"], t["n_refuse"], t["n_refuse"] - base["n_refuse"],
                 t["n_decert"], t["n_decert"] - base["n_decert"]))
        if dd:
            print("  decertifications moved: %s"
                  % ", ".join("%s %+d" % (k, v) for k, v in dd.items()))
        else:
            print("  decertifications moved: NONE")
        if dr:
            print("  REFUSALS moved:         %s"
                  % ", ".join("%s %+d" % (k, v) for k, v in dr.items()))
    print("-" * 78)

    # the combined levels
    print()
    print("=" * 78)
    print("COMBINED LEVELS")
    print("=" * 78)
    levels = [("known", lambda: rocket.build(declarations="known"))]
    if a.with_real:
        levels.append(("real", lambda: rocket.build(declarations="real")))
    for name, fn in levels:
        t = tally(fn())
        dd, dr = delta(base, t)
        out.setdefault("levels", {})[name] = dict(t, delta_decert=dd, delta_refuse=dr)
        print("LEVEL '%s': verdict %-10s  %d refusals (%+d), %d decerts (%+d)"
              % (name, t["verdict"], t["n_refuse"], t["n_refuse"] - base["n_refuse"],
                 t["n_decert"], t["n_decert"] - base["n_decert"]))
        print("  surviving decertifications, by rule:")
        for k, v in sorted(t["decert"].items()):
            print("    %-16s %d" % (k, v))
        print("  refusals, by rule:")
        for k, v in sorted(t["refuse"].items()):
            print("    %-24s %d" % (k, v))
        print()

    # additivity: is the whole the sum of the parts?
    if "levels" in out and "known" in out["levels"]:
        parts = Counter()
        for nm in ("known fields (time_disc, stencil, elliptic, substeps, lambda_ref)",
                   "decomposition NON_OVERLAPPING (P14)",
                   "clocks from the YAML (50:1, not 100:1)",
                   "flux_matching TIME_INTEGRATED"):
            for k, v in out["arms"][nm]["delta_decert"].items():
                parts[k] += v
        whole = out["levels"]["known"]["delta_decert"]
        print("ADDITIVITY -- the sum of the four parts against the whole 'known' level:")
        keys = set(parts) | set(whole)
        agree = True
        for k in sorted(keys):
            p, w = parts.get(k, 0), whole.get(k, 0)
            flag = "" if p == w else "   <-- NOT ADDITIVE"
            if p != w:
                agree = False
            print("    %-16s parts %+d   whole %+d%s" % (k, p, w, flag))
        print("  additive: %s" % agree)
        out["additive"] = agree

    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print()
        print("artifact written to %s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
