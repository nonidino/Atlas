"""W303 -- does the W138 convention question reach CONJUGATE HEAT TRANSFER too?

`probe.assemble_seam`'s own docstring already says what Tier 77 measured:

    A fluid-solid seam is usually **not** written [in the Steklov-Poincare
    convention]: the fluid returns *the load on the surface* and the solid *the
    reaction that holds it*, both positive in one shared direction, and the
    well-posed interface condition is their DIFFERENCE. ... Adding them
    assembles a matrix no scheme differentiates, and `L4/E7/passivity` then
    reports a defect that is the convention rather than an amplified mode.

It says it about a **fluid-solid MECH** seam, and `front_wing` is the case it
was measured on.  This script asks whether the same is true of a **THERM** seam,
where both sides return a heat flux that is positive in one shared direction --
into the solid -- exactly as that note describes.

Two seams, the same three questions each:

    1. do the two sides report against the same normal, or opposite ones?
    2. does the SUM condition have a root in the admissible interval? the
       DIFFERENCE?
    3. is the sum's symmetric part definite? the difference's?

`thermal_seam` is included **because its numbers are load-bearing** -- W68, W71,
W74, W83, W86 and W87 all quote them -- so whether the question reaches it is
not a curiosity.

ASCII output only -- the console is cp1252.

    python scripts/w303_cht_convention.py --json out/w303.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

DT_GAS_ROCKET = 1.0e-5


def _fmt(x, n=6):
    if x is None:
        return "None"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, str):
        return x
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return ("%." + str(n) + "g") % float(x)


def _transfer(graph, label):
    from atlas.compiler import _Context, _derive_transfer          # noqa: PLC2701
    from atlas.envelope import EnvelopeStamp
    from atlas.holes import HoleLedger
    from atlas.probe import ProbeBudget
    from atlas.verdict import DecisionRecord
    ctx = _Context(graph=graph, budget=None, probe_budget=ProbeBudget(),
                   references={}, record=DecisionRecord(), stamp=EnvelopeStamp(),
                   holes=HoleLedger(), probe_state=label, depth=0)
    return _derive_transfer(ctx, graph.connections[0])


def _rocket():
    from atlas.cases import rocket_experts as RE
    shell, gas, prov = RE.make_bc_experts(dt_gas=DT_GAS_ROCKET)
    graph = RE.build_bc_graph(shell, gas)
    tr = _transfer(graph, "w303-rocket")
    return {
        "label": "rocket b-c:THERM", "graph": graph, "tr": tr,
        "sides": {"b": (gas, "c:THERM"), "c": (shell, "b:THERM")},
        "fluid": "b", "solid": "c",
        "lo": RE.T_AMBIENT, "hi": RE.T_CHAMBER,
        "note": RE.probe_state_string(prov),
    }


def _thermal_seam():
    from atlas.cases import thermal_seam as T
    graph, experts = T.build(mode="split-step", clocks="matched")
    tr = _transfer(graph, "w303-thermal-seam")
    return {
        "label": "thermal_seam cht", "graph": graph, "tr": tr,
        "sides": {"gas": (experts["gas"], "wall:THERM"),
                  "shell": (experts["shell"], "inner:THERM")},
        "fluid": "gas", "solid": "shell",
        "lo": T.T_OUT, "hi": T.T_HOT,
        "note": "duct, T_hot=%g K, T_wall_0=%g K, matched clocks" % (T.T_HOT, T.T_WALL_0),
    }


def _uniform_mu(tr, P, value):
    M = P.effective_matrix()
    mu, _r, _rk, _sv = np.linalg.lstsq(M, np.full(M.shape[0], float(value)),
                                       rcond=None)
    return mu


def analyse(case, n_scan=9):
    tr = case["tr"]
    names = sorted(case["sides"])
    Ps = {a: tr.prolongations[a] for a in names}
    Rs = {a: Ps[a].adjoint(tr.space) for a in names}

    def halves(value):
        out = {}
        for a in names:
            expert, port = case["sides"][a]
            v = Ps[a].prolong(_uniform_mu(tr, Ps[a], value), tr.space)
            out[a] = Rs[a] @ np.asarray(expert.respond(port, v), float)
        return out

    lo, hi = case["lo"], case["hi"]
    print("  %s" % case["note"])
    print("     base [K]     sum (scalar)   difference (scalar)")
    rows = []
    f, s = case["fluid"], case["solid"]
    for v in np.linspace(lo, hi, n_scan):
        h = halves(v)
        one = np.ones_like(h[f])
        ssum = float(np.dot(h[f] + h[s], one) / np.sqrt(one.size))
        sdif = float(np.dot(h[f] - h[s], one) / np.sqrt(one.size))
        rows.append({"base_K": float(v), "sum": ssum, "difference": sdif})
        print("  %11.2f  %14.5e  %16.5e" % (v, ssum, sdif))

    def sign_changes(key):
        return [(a["base_K"], b["base_K"]) for a, b in zip(rows[:-1], rows[1:])
                if a[key] * b[key] < 0.0]

    br = {k: sign_changes(k) for k in ("sum", "difference")}
    print("  roots bracketed:  sum %s   difference %s"
          % (br["sum"] or "NONE", br["difference"] or "NONE"))

    # and the two assembled operators at the SAME base, so definiteness is
    # compared like for like rather than at two different points.
    from atlas.probe import probe_block
    base_val = 0.5 * (lo + hi)
    if br["difference"]:
        base_val = 0.5 * sum(br["difference"][0])
    blocks = {}
    for a in names:
        expert, port = case["sides"][a]
        caps = [ag.capabilities for ag in case["graph"].agents if ag.agent_id == a][0]
        mu = _uniform_mu(tr, Ps[a], base_val)
        blocks[a] = probe_block(caps, caps.port(port), tr.space, Ps[a],
                                case["graph"].connections[0].seam_id,
                                base_V=Ps[a].prolong(mu, tr.space)).S
    S_sum = blocks[f] + blocks[s]
    S_dif = blocks[f] - blocks[s]
    res = {"scan": rows, "brackets": br, "compared_at_K": float(base_val),
           "blocks": {}}
    print("  operators compared at a common base of %s K:" % _fmt(base_val, 6))
    for a in names:
        sym = 0.5 * (blocks[a] + blocks[a].T)
        ev = np.linalg.eigvalsh(sym)
        res["blocks"][a] = {"norm_F": float(np.linalg.norm(blocks[a], "fro")),
                            "sym_min": float(ev.min()), "sym_max": float(ev.max())}
        print("    block %-6s ||S||_F %-11s sym lambda in [%s, %s]"
              % (a, _fmt(res["blocks"][a]["norm_F"], 5),
                 _fmt(ev.min(), 4), _fmt(ev.max(), 4)))
    for tag, S in (("sum", S_sum), ("difference", S_dif)):
        sym = 0.5 * (S + S.T)
        ev = np.linalg.eigvalsh(sym)
        sv = np.linalg.svd(S, compute_uv=False)
        res[tag] = {"norm_F": float(np.linalg.norm(S, "fro")),
                    "sigma_min": float(sv.min()), "sym_lambda_min": float(ev.min()),
                    "definite": bool(ev.min() > 0.0)}
        print("    %-11s ||S||_F %-11s sigma_min %-11s sym lambda_min %-11s definite %s"
              % (tag, _fmt(res[tag]["norm_F"], 5), _fmt(sv.min(), 5),
                 _fmt(ev.min(), 5), res[tag]["definite"]))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", metavar="PATH")
    a = ap.parse_args(argv)
    out = {"script": "w303_cht_convention.py", "date": "2026-09-17",
           "build_repo_commit": "0a407b7"}
    print("=" * 84)
    print("W303 -- does W138's convention question reach conjugate heat transfer?")
    print("=" * 84)
    print("`assemble_seam`'s own note says a fluid-solid seam usually has both")
    print("sides positive in ONE shared direction, so the well-posed condition is")
    print("their DIFFERENCE and adding them makes L4/E7/passivity report the")
    print("convention. It says it of MECH. Both THERM sides here return heat INTO")
    print("THE SOLID, which is the same shape.")
    print()
    for builder in (_thermal_seam, _rocket):
        case = builder()
        print("-" * 84)
        print("CASE: %s" % case["label"])
        out[case["label"]] = analyse(case)
        print()
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=str)
        print("artifact written to %s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
