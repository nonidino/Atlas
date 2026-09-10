"""W174 -- is a refusal EARNED?  R10's halo threshold against a coupling that fails.

`poc2-novelty-audit` section 4 is the audit's sharpest open point and Tier 39 left
it standing:

    "Nobody has yet exhibited a seam the compiler refused that would in fact have
     DIVERGED, or a seam it admitted that HELD.  Until one of those exists, 'it
     judges correctly' is a statement about internal consistency, not about the
     world."

`L2/R10/halo` is the one rule in the package with a numeric threshold on one
declared integer -- the overlap must cover `stencil_radius * substeps` -- and
CS-S1 built a sweep that varies exactly that integer and marches the coupling to
its fixed point at each value.  So the confusion table can be filled in: compile
at each halo, march at each halo, and cross-tabulate.

    refuse & diverge   the refusal is EARNED at that row
    admit  & converge  the admission is earned
    refuse & converge  the rule OVER-fires: it costs a coupling that would work
    admit  & diverge   the rule UNDER-fires: silent wrongness, the class this
                       whole framework exists to refuse

Run in BOTH arrangements, because W168 measured that the halo is a convergence
condition only for the EMBEDDED agent -- so a table taken in one arrangement
would report the opposite of what the rule does in the other.

Outputs `out/w174/w174.json`, with the CS-S1 artifact's own rows carried beside
the fresh ones so the two cannot drift.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import io
import json
import sys
import time

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas import Budget, compile_scheme  # noqa: E402
from atlas.cases import neural_interface as NI  # noqa: E402
from atlas.cases import poseidon as PO  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "out", "w174")

HALOS = (4, 8, 12, 16, 18, 20, 21, 22, 24, 32, 48)


def halo_decision(graph):
    """What `L2/R10/halo` said on this graph, and what the whole compile said."""
    res = compile_scheme(graph, budget=Budget(allow_probe=False))
    rows = [d for d in res.decisions if d.rule == "R10/halo"]
    verdicts = sorted({d.verdict.value for d in rows})
    return {
        "halo_rule_verdicts": verdicts,
        "halo_rule_refuses": any(v == "refuse" for v in verdicts),
        "halo_rule_messages": [d.message[:280] for d in rows],
        "graph_verdict": res.verdict.value,
        "n_refusals": len(res.decisions.refusals),
        "refused_rules": sorted({f"{d.layer}/{d.rule}"
                                 for d in res.decisions.refusals}),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.perf_counter()
    u_full, v_full = NI.load_state()

    results = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "declared_reach": NI.DOMAIN_OF_DEPENDENCE,
        "halos": list(HALOS),
        "arrangements": {},
    }

    # **The 2x2, completed.**  CS-S1 swept two of the four cells and they differ
    # in BOTH variables -- "as-built" is (embedded, unprojected) and "split-step"
    # is (exposed, projected) -- so its bracket cannot attribute the divergence
    # to either one.  The two missing cells are run here.
    for expose, project, name in [
        (False, False, "embedded + no projection  [CS-S1's 'as-built']"),
        (False, True, "embedded + global projection  [NEW]"),
        (True, True, "exposed + global projection  [CS-S1's 'split-step']"),
        (True, False, "exposed + no projection  [NEW]"),
    ]:
        print("=== " + name)
        conv = NI.halo_convergence(u_full, v_full, HALOS,
                                   expose_elliptic=expose, project=project)
        by_halo = {r["halo"]: r for r in conv}
        rows = []
        for halo in HALOS:
            mono_n = 2 * PO.EXPERT_RES - halo
            if mono_n > min(u_full.shape[0], u_full.shape[1]):
                continue
            til = NI.HybridTiling(mono_n=mono_n, ramp=8)
            graph, _ = NI.build(u_full, v_full, tiling=til,
                                expose_elliptic=expose)
            dec = halo_decision(graph)
            c = by_halo.get(til.halo, {})
            row = {"halo": til.halo, "mono_n": mono_n,
                   "converging": c.get("converging"),
                   "contraction": c.get("contraction"),
                   "sweeps": c.get("sweeps"),
                   "diverged": c.get("diverged"),
                   **dec}
            # the four cells of the confusion table
            if row["converging"] is None:
                row["cell"] = "unmeasured"
            elif dec["halo_rule_refuses"] and not row["converging"]:
                row["cell"] = "refuse & diverge -- EARNED"
            elif dec["halo_rule_refuses"] and row["converging"]:
                row["cell"] = "refuse & converge -- OVER-fires"
            elif not dec["halo_rule_refuses"] and row["converging"]:
                row["cell"] = "admit & converge -- earned"
            else:
                row["cell"] = "admit & diverge -- UNDER-fires"
            rows.append(row)
            print("  halo %2d  rule %-8s  contraction %-10s  %s"
                  % (row["halo"],
                     "refuse" if dec["halo_rule_refuses"] else "admit",
                     ("%.5f" % row["contraction"]) if row["contraction"] is not None
                     and np.isfinite(row["contraction"]) else str(row["contraction"]),
                     row["cell"]))
        tally = {}
        for r in rows:
            tally[r["cell"]] = tally.get(r["cell"], 0) + 1
        results["arrangements"][name] = {
            "expose_elliptic": expose,
            "project": project,
            "rows": rows,
            "tally": tally,
            "threshold": NI.halo_threshold(conv),
        }
        print("  tally:", tally, " threshold:", NI.halo_threshold(conv))

    # Can the graph even SAY whether the composition layer projects?  Asked of
    # the constructor rather than argued: `project` is a parameter of the
    # exchange and not of `build`, so two arrangements that differ in it compile
    # to the same declarations and therefore to the same verdict.
    import inspect

    sig = list(inspect.signature(NI.build).parameters)
    results["project_is_declarable"] = {
        "build_parameters": sig,
        "project_in_build": "project" in sig,
        "halo_convergence_parameters": list(
            inspect.signature(NI.halo_convergence).parameters),
        "note": "the global projection is a property of the COMPOSITION LAYER and "
                "the graph has no field for it, so the compile cannot distinguish "
                "the two cells of the 2x2 that differ in it",
    }
    print("project declarable on the graph:",
          results["project_is_declarable"]["project_in_build"])

    # the CS-S1 artifact, carried beside so the two cannot drift
    w166 = os.path.join(HERE, "out", "w166", "w166.json")
    if os.path.exists(w166):
        with open(w166, encoding="utf-8") as f:
            prev = json.load(f)
        results["cs_s1_artifact"] = {
            k: prev["B_halo"][k]["threshold"] for k in prev["B_halo"]}
        print("CS-S1 artifact thresholds:", results["cs_s1_artifact"])

    results["elapsed_seconds"] = time.perf_counter() - t0
    path = os.path.join(OUT, "w174.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("wrote", path, "in %.1f s" % results["elapsed_seconds"])


if __name__ == "__main__":
    main()
