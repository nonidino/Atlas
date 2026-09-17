"""W301 -- the saturated-channel detector, and its FALSE-POSITIVE RATE.

Tier 76 found that `compressible2d`'s isothermal wall clamps its ghost at
``T_g = max(2 T_wall - T_i, 20)``, so below ``T_wall = (T_i + 20)/2`` the imposed
trace never reaches the solver -- and that BOTH gas agents in this vault sit in
that regime at their own declared probe bases.  It proposed a detector that
needs no new machinery:

    SupportReach.exact_zeros == n - 1

a field `probe.support_reach` has emitted since Tier 0 and no layer has ever
read.  **A detector is worth nothing until its false-positive rate is
measured**, because the signature is necessary and not sufficient: an expert
whose genuine response happened to be exactly diagonal would trip it.

So this sweeps every agent of every case graph that will build, pokes one seam
cell at the agent's own declared probe base, and reports which trip it.  Two
synthetic controls bracket the measurement: an exactly-diagonal responder, which
MUST trip, and a banded one, which must not.

Agents whose case needs a checkpoint, a cache or the build repo are SKIPPED and
named -- a sweep that silently omits the expensive half is a sweep whose rate
means nothing.

ASCII output only -- the console is cp1252.

    python scripts/w301_saturation_sweep.py --json out/w301.json
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

from atlas.capability import ExpertCapabilities, port_decl
from atlas.ports import PortType, ResponseHalf
from atlas.probe import probe_base, support_reach

#: Cases whose `build()` is cheap enough to sweep and needs no external asset.
#: Named explicitly rather than discovered, because a sweep that quietly drops
#: the cases it could not build reports a rate about the cases it could.
CASES = [
    ("thermal_seam", {}),
    ("rocket", {"declarations": "real"}),
    ("thermal_strain", {}),
    ("brake_thermal", {}),
    ("cooling_loop", {}),
    ("channel_ns", {}),
    ("front_wing", {}),
    ("wing_fsi", {}),
    ("ground_effect", {}),
    ("reuse_probe", {}),
    ("powertrain", {}),
    ("scaling_ladder", {}),
    ("wind_farm", {}),
    ("neural_interface", {}),
    ("poseidon", {}),
    ("wake_array", {}),
    ("window_ns", {}),
    ("car_graph", {}),
    ("racelab", {}),
    ("integration_union", {}),
    ("wind_farm_real", {}),
]


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


def _controls():
    """The two synthetic brackets. Without these the rate is uninterpretable."""
    n = 32

    def diagonal(_port, trace):
        return 3.0 * np.asarray(trace, float)

    def banded(_port, trace):
        t = np.asarray(trace, float)
        out = 3.0 * t.copy()
        out[1:] += 0.4 * t[:-1]
        out[:-1] += 0.4 * t[1:]
        return out

    rows = []
    for tag, fn, must_trip in (("control: exactly diagonal", diagonal, True),
                               ("control: 1-cell banded", banded, False)):
        sr = support_reach(fn, "p", np.full(n, 1.0))
        tripped = sr.exact_zeros == sr.n - 1
        rows.append({"agent": tag, "case": "-", "n": sr.n,
                     "nonzero": sr.nonzero, "exact_zeros": sr.exact_zeros,
                     "tripped": bool(tripped), "must_trip": must_trip,
                     "as_expected": bool(tripped == must_trip)})
    return rows


def _kind(fn) -> str:
    """Is this response a real callable, or `capability.linear_response`?

    **The distinction is the whole measurement.**  A fixture built as
    `linear_response(A)` with a diagonal or near-diagonal `A` trips the
    signature trivially and says nothing about whether the rule would fire on an
    expert.  Detected structurally, off the closure's qualified name, because
    nothing marks a fixture as one -- which is deliberate: `linear_response`'s
    own docstring says *"the probe cannot tell this from a neural expert, which
    is the point"*.
    """
    q = getattr(fn, "__qualname__", "") or ""
    return "synthetic" if "linear_response.<locals>" in q else "callable"


def _sweep_case(name, kwargs, budget_s):
    """Every PORT of every agent, not just the first.

    The first version of this sweep took ``caps.ports[0]`` and reported that the
    rocket's `b` did not trip -- because at the `real` level that agent's first
    port is `a:MECH`, still a seeded random matrix, while the one backed by
    `compressible2d` is `c:THERM`. A sweep that probes one port of a mixed
    record measures the port it happened to pick.
    """
    mod = importlib.import_module("atlas.cases." + name)
    built = mod.build(**kwargs)
    graph = built[0] if isinstance(built, tuple) else built
    rows = []
    t_case = time.perf_counter()
    for agent in graph.agents:
        caps: ExpertCapabilities = agent.capabilities
        if caps.boundary_response is None:
            rows.append({"case": name, "agent": agent.agent_id, "port": None,
                         "skipped": "declares no boundary_response"})
            continue
        for port in caps.ports:
            n = None
            pro = port.prolongation
            if pro is not None:
                n = int(pro.effective_matrix().shape[0])
            elif port.effective_resolution:
                n = int(port.effective_resolution)
            if not n:
                rows.append({"case": name, "agent": agent.agent_id,
                             "port": port.name,
                             "skipped": "no V dimension derivable from the record"})
                continue
            try:
                base = probe_base(caps, port.name, n)
                t0 = time.perf_counter()
                sr = support_reach(caps.boundary_response, port.name, base)
                el = time.perf_counter() - t0
            except Exception as exc:                  # noqa: BLE001
                rows.append({"case": name, "agent": agent.agent_id,
                             "port": port.name,
                             "skipped": "%s: %s" % (type(exc).__name__,
                                                    str(exc)[:90])})
                continue
            rows.append({
                "case": name, "agent": agent.agent_id, "port": port.name,
                "kind": _kind(caps.boundary_response),
                "n": sr.n, "nonzero": sr.nonzero, "exact_zeros": sr.exact_zeros,
                "reach": sr.reach, "peak": sr.peak, "fraction": sr.fraction,
                "tripped": bool(sr.exact_zeros == sr.n - 1),
                "is_global": bool(sr.is_global), "seconds": el,
                "bc_channel": caps.bc_channel.value,
            })
            if time.perf_counter() - t_case > budget_s:
                rows.append({"case": name, "agent": "<budget>",
                             "skipped": "case exceeded the per-case budget"})
                return rows
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--budget", type=float, default=90.0,
                    help="seconds per case before it is abandoned and named")
    a = ap.parse_args(argv)

    out = {"script": "w301_saturation_sweep.py", "date": "2026-09-17",
           "build_repo_commit": "0a407b7"}
    print("=" * 96)
    print("W301 -- does `exact_zeros == n-1` flag anything but the two known")
    print("        saturated channels? The detector's false-positive rate.")
    print("=" * 96)

    ctrl = _controls()
    out["controls"] = ctrl
    for r in ctrl:
        print("  %-28s n=%-4d nonzero=%-4d exact_zeros=%-4d tripped=%-6s expected=%s"
              % (r["agent"], r["n"], r["nonzero"], r["exact_zeros"],
                 r["tripped"], r["as_expected"]))
    if not all(r["as_expected"] for r in ctrl):
        print("  CONTROLS FAILED -- the rate below means nothing. Stopping.")
        return 1
    print()

    rows, skipped = [], []
    for name, kwargs in CASES:
        t0 = time.perf_counter()
        try:
            got = _sweep_case(name, kwargs, a.budget)
        except Exception as exc:                      # noqa: BLE001
            skipped.append({"case": name,
                            "why": "%s: %s" % (type(exc).__name__, str(exc)[:120])})
            print("  %-18s SKIPPED  %s: %s"
                  % (name, type(exc).__name__, str(exc)[:70]))
            continue
        el = time.perf_counter() - t0
        ok = [r for r in got if "skipped" not in r]
        sk = [r for r in got if "skipped" in r]
        rows.extend(ok)
        skipped.extend(sk)
        trip = [r for r in ok if r["tripped"]]
        print("  %-18s %2d agents probed, %2d skipped, %2d TRIPPED   (%5.1f s)"
              % (name, len(ok), len(sk), len(trip), el))
        for r in trip:
            print("        -> %s.%s on %s: n=%d nonzero=%d peak=%s channel=%s"
                  % (r["case"], r["agent"], r["port"], r["n"], r["nonzero"],
                     _fmt(r["peak"], 5), r["bc_channel"]))

    out["rows"], out["skipped"] = rows, skipped
    tripped = [r for r in rows if r["tripped"]]
    print()
    print("=" * 96)
    print("  %d PORTS probed across %d cases; %d skipped and named."
          % (len(rows), len({r["case"] for r in rows}), len(skipped)))
    print()
    print("  **The rate has to be read per KIND**, because a fixture built as")
    print("  `linear_response(A)` with a diagonal A trips the signature trivially")
    print("  and says nothing about whether the rule would fire on an expert.")
    for kind in ("callable", "synthetic"):
        k = [r for r in rows if r.get("kind") == kind]
        kt = [r for r in k if r["tripped"]]
        rate = (100.0 * len(kt) / len(k)) if k else float("nan")
        print()
        print("    %-10s %3d ports, %2d tripped  (%5.1f%%)" % (kind, len(k), len(kt), rate))
        for r in kt:
            print("       -> %-16s %-8s %-14s n=%-5d peak=%s channel=%s"
                  % (r["case"], r["agent"], r["port"], r["n"],
                     _fmt(r["peak"], 5), r["bc_channel"]))
        out["rate_" + kind] = {"n": len(k), "tripped": len(kt), "pct": rate}
    out["n_probed"], out["n_tripped"] = len(rows), len(tripped)

    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)) or ".", exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print()
        print("artifact written to %s" % a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
