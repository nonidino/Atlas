"""W189 -- the control that has to exist BEFORE the schema changes.

Tier 42 clause 2 and Tier 44 section 3.1 make it the first assertion: a schema
extension moves nothing on a graph nobody asked about, and the way to know that
is to compile every existing graph before the change, compile it again after,
and compare the emitted artifact BYTE FOR BYTE.  Anything weaker -- the same
verdict, the same rule set -- is a claim about a projection of the artifact and
would pass a change that rewrote every message.

So this script does three things and nothing else:

  capture <label>          build every constructible case graph (and the Tier 44
                           union fixtures), compile it, and write the full
                           `RunArtifact.to_json()` plus its sha256 to
                           ``out/w189/control_<label>.json`` and
                           ``out/w189/artifacts/<label>/<key>.json``.
  compare <a> <b>          byte-compare two captures, key by key.
  census                   the declaration census W188 requires before any
                           derivation is leaned on: for every graph that carries
                           a partition of unity, are its subdomains exactly the
                           agents of one overlapping region?

**The repeat floor is part of the control.**  A capture that is not
bit-reproducible across two processes cannot tell a behaviour change from noise,
so `capture` is run twice on the unchanged tree -- under two different
``PYTHONHASHSEED`` values, because `verdict._jsonable` turns a SET into a list
in iteration order and string hashing is randomised per process -- and those two
must agree before the "after" capture means anything.

No network: ``HF_HUB_OFFLINE`` is set before any import, so the Poseidon-backed
graphs load from the local cache or are recorded as unconstructible -- never
downloaded.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import hashlib
import importlib
import io
import json
import sys
import time
import traceback

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

OUT = os.path.join(HERE, "out", "w189")

#: Case modules that construct no `CaseGraph` at all.  Listed so the count of
#: twenty is honest: `seam_placement` is a search over decompositions and
#: `wind_farm_design` an optimiser over rollouts, and neither has a graph to
#: compile.
NO_GRAPH_MODULES = ("seam_placement", "wind_farm_design")

#: Variants that need torch and the Poseidon-T checkpoint from the local cache.
TORCH_KEYS = ("poseidon", "wake_array-poseidon")


def _npz(rel):
    import numpy as np

    return np.load(os.path.join(HERE, *rel.split("/")))


def _unwrap(g):
    """Most builders return (graph, experts); a few return the graph."""
    if isinstance(g, tuple):
        return next(x for x in g if hasattr(x, "agents"))
    return g


def _s0():
    d = _npz("out/tier0b/s0_state.npz")
    return d["u"], d["v"]


def _settled(name):
    d = _npz(f"out/{name}/settled.npz")
    return d["u"], d["v"]


def _cases(name):
    return importlib.import_module(f"atlas.cases.{name}")


def _rung(label):
    sl = _cases("scaling_ladder")
    return {r.label: r for r in sl.ladder()}[label]


def _ladder(kind, label="N6", **kw):
    import numpy as np

    sl = _cases("scaling_ladder")
    r = _rung(label)
    return sl.build(np.ones(r.shape), np.zeros(r.shape), r, kind=kind, **kw)


def _reuse(kind="reference_exposed", label="N6"):
    import numpy as np

    rp = _cases("reuse_probe")
    r = _rung(label)
    return rp.build(np.ones(r.shape), np.zeros(r.shape), r, kind=kind)


def _union(with_pou):
    import w171_region_axis as W

    return W.union_graph(with_pou=with_pou)


#: Every graph the control compiles, in a FIXED order -- a case module may cache
#: state at import, so the order is part of what is held constant.  Keys are
#: stable names; the builders are the calls the suite itself makes.
VARIANTS = [
    ("rocket", lambda: _cases("rocket").build()),
    ("rocket-staging", lambda: _cases("rocket").build(with_staging=True)),
    ("wind_farm", lambda: _cases("wind_farm").build()),
    ("wind_farm-boundary", lambda: _cases("wind_farm").build(
        boundary_capable=True, declare_transfer=True)),
    ("thermal_seam", lambda: _cases("thermal_seam").build()),
    ("thermal_seam-split-matched", lambda: _cases("thermal_seam").build(
        mode="split-step", clocks="matched")),
    ("brake_thermal", lambda: _cases("brake_thermal").build()),
    ("brake_thermal-split", lambda: _cases("brake_thermal").build(
        mode="split-step", clocks="native")),
    ("thermal_strain", lambda: _cases("thermal_strain").build()),
    ("thermal_strain-global-field", lambda: _cases("thermal_strain").build(
        route="global-field")),
    ("cooling_loop", lambda: _cases("cooling_loop").build()),
    ("cooling_loop-open", lambda: _cases("cooling_loop").build(close_loop=False)),
    ("cooling_loop-3leg", lambda: _cases("cooling_loop").build(n_legs=3)),
    ("powertrain", lambda: _cases("powertrain").build()),
    ("powertrain-open", lambda: _cases("powertrain").build(close_loop=False)),
    ("window_ns", lambda: _cases("window_ns").build(*_s0())),
    ("window_ns-as-built", lambda: _cases("window_ns").build(*_s0(), mode="as-built")),
    ("channel_ns", lambda: _cases("channel_ns").build(*_s0())),
    ("channel_ns-as-built", lambda: _cases("channel_ns").build(*_s0(), mode="as-built")),
    ("wind_farm_real", lambda: _cases("wind_farm_real").build(*_s0())),
    ("wind_farm_real-as-built", lambda: _cases("wind_farm_real").build(
        *_s0(), mode="as-built")),
    ("neural_interface", lambda: _cases("neural_interface").build(*_s0())),
    ("neural_interface-embedded", lambda: _cases("neural_interface").build(
        *_s0(), expose_elliptic=False)),
    ("front_wing", lambda: _cases("front_wing").build(*_settled("w141"), motion=False)),
    ("front_wing-riding", lambda: _cases("front_wing").build(
        *_settled("w141"), motion=True)),
    ("ground_effect", lambda: _cases("ground_effect").build(
        *_settled("w141"), motion=False)),
    ("ground_effect-moving", lambda: _cases("ground_effect").build(
        *_settled("w141"), motion=True)),
    ("wing_fsi", lambda: _cases("wing_fsi").build(*_settled("w136"), motion=False)),
    ("wing_fsi-moving", lambda: _cases("wing_fsi").build(*_settled("w136"), motion=True)),
    ("wake_array-reference", lambda: _cases("wake_array").build(
        *(lambda d: (d["u"], d["v"]))(_npz("out/w93/state.npz")), kind="reference")),
    ("wake_array-reference_exposed", lambda: _cases("wake_array").build(
        *(lambda d: (d["u"], d["v"]))(_npz("out/w93/state.npz")),
        kind="reference_exposed")),
    ("wake_array-bare-blend", lambda: _cases("wake_array").build(
        *(lambda d: (d["u"], d["v"]))(_npz("out/w93/state.npz")),
        kind="reference_exposed", assembly_projection=False)),
    ("scaling_ladder-N6-reference_exposed", lambda: _ladder("reference_exposed")),
    ("scaling_ladder-N6-reference", lambda: _ladder("reference")),
    ("scaling_ladder-N12-reference_exposed", lambda: _ladder("reference_exposed",
                                                             label="N12")),
    ("reuse_probe-N6", lambda: _reuse()),
    ("tier44-union", lambda: _union(False)),
    ("tier44-union-with-pou", lambda: _union(True)),
    ("poseidon", lambda: _cases("poseidon").build(*_s0())),
    ("wake_array-poseidon", lambda: _cases("wake_array").build(
        *(lambda d: (d["u"], d["v"]))(_npz("out/w93/state.npz")), kind="poseidon")),
]


def module_of(key: str) -> str:
    return key.split("-")[0]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compile_artifact(key: str):
    """Build and compile one variant; return (graph, result, artifact_json, timings)."""
    from atlas import compile_scheme

    builder = dict(VARIANTS)[key]
    t0 = time.perf_counter()
    graph = _unwrap(builder())
    t1 = time.perf_counter()
    result = compile_scheme(graph)
    t2 = time.perf_counter()
    text = result.artifact.to_json()
    return graph, result, text, {"build_s": t1 - t0, "compile_s": t2 - t1}


# ===========================================================================
# capture / compare
# ===========================================================================


def _retry(fn, attempts=40, pause=0.25):
    """Run a filesystem step, retrying on Windows sharing violations.

    **Found by the first repeat-floor capture, 2026-09-10.** The vault lives under
    OneDrive, and the sync client holds a just-written file open for a moment:
    ``os.replace`` onto it raised ``PermissionError: [WinError 5]`` after eight
    graphs and the capture died with the manifest half-written.  A persist step
    that can kill the run it is persisting is the wrong way round, so it retries
    and, past the budget, raises with the path named.
    """
    last = None
    for _ in range(attempts):
        try:
            return fn()
        except PermissionError as exc:                              # pragma: no cover
            last = exc
            time.sleep(pause)
    raise last


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"

    def dump():
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=1)

    _retry(dump)
    _retry(lambda: os.replace(tmp, path))


def capture(label: str, only=None, skip_torch=False) -> dict:
    """Compile every variant and persist after EACH one, so a failure keeps state."""
    path = os.path.join(OUT, f"control_{label}.json")
    art_dir = os.path.join(OUT, "artifacts", label)
    os.makedirs(art_dir, exist_ok=True)
    res = {
        "label": label,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "<random>"),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "no_graph_modules": list(NO_GRAPH_MODULES),
        "rows": {},
    }
    for key, _b in VARIANTS:
        if only and key not in only:
            continue
        if skip_torch and key in TORCH_KEYS:
            res["rows"][key] = {"skipped": "torch variants not requested"}
            continue
        row = {"module": module_of(key)}
        try:
            graph, result, text, t = compile_artifact(key)

            def keep(text=text, key=key):
                with open(os.path.join(art_dir, key + ".json"), "w",
                          encoding="utf-8") as fh:
                    fh.write(text)

            _retry(keep)
            row.update(
                graph=graph.name,
                sha256=digest(text),
                n_bytes=len(text.encode("utf-8")),
                verdict=result.verdict.value,
                n_decisions=len(result.decisions.decisions),
                hex_addresses=text.count(" at 0x"),
                **t,
            )
        except Exception as exc:                                    # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"[:600]
            row["traceback"] = traceback.format_exc()[-2000:]
        res["rows"][key] = row
        _write(path, res)
        status = row.get("sha256", "ERROR " + row.get("error", ""))[:16]
        print("  %-40s %-18s %s" % (key, row.get("verdict", "-"), status), flush=True)
    res["modules_constructed"] = sorted({r["module"] for r in res["rows"].values()
                                         if "sha256" in r})
    _write(path, res)
    return res


def compare(a: str, b: str) -> dict:
    ra = json.load(open(os.path.join(OUT, f"control_{a}.json"), encoding="utf-8"))
    rb = json.load(open(os.path.join(OUT, f"control_{b}.json"), encoding="utf-8"))
    rows, same, differ, missing = {}, [], [], []
    for key in dict(VARIANTS):
        x, y = ra["rows"].get(key, {}), rb["rows"].get(key, {})
        if "sha256" not in x or "sha256" not in y:
            missing.append(key)
            rows[key] = {"a": x.get("sha256") or x.get("error") or x.get("skipped"),
                         "b": y.get("sha256") or y.get("error") or y.get("skipped")}
            continue
        eq = x["sha256"] == y["sha256"]
        (same if eq else differ).append(key)
        rows[key] = {"identical": eq, "sha256_a": x["sha256"], "sha256_b": y["sha256"]}
    out = {"a": a, "b": b, "identical": same, "differ": differ,
           "not_compared": missing, "rows": rows}
    _write(os.path.join(OUT, f"compare_{a}_vs_{b}.json"), out)
    return out


# ===========================================================================
# census -- W188: price the derivation against every graph before leaning on it
# ===========================================================================


def census_row(graph) -> dict:
    from atlas.graph import Decomposition

    pou = graph.partition_of_unity
    inner = getattr(pou, "partition", pou)
    regions = graph.regions()
    axes = graph.region_axes()
    row = {
        "graph": graph.name,
        "declared_axis": graph.decomposition.value,
        "n_agents": len(graph.agents),
        "regions": {k: sorted(v) for k, v in regions.items()},
        "region_axes": {k: v.value for k, v in axes.items()},
        "n_overlapping_regions": sum(1 for v in axes.values()
                                     if v is Decomposition.OVERLAPPING),
        "n_non_overlapping_regions": sum(1 for v in axes.values()
                                         if v is Decomposition.NON_OVERLAPPING),
        "overlap": graph.overlap,
        "overlap_cells": graph.overlap_cells,
        "pou": None if pou is None else type(pou).__name__,
        "pou_inner": None if pou is None else type(inner).__name__,
        "projection": graph.assembly_projection is not None,
    }
    if pou is None:
        return row
    keys = sorted(pou.subdomains())
    ids = {a.agent_id for a in graph.agents}
    fams = {graph.agent(k).capabilities.governing_family or ""
            for k in keys if k in ids}
    row.update(
        pou_subdomains=keys,
        pou_keys_are_agent_ids=all(k in ids for k in keys),
        pou_families=sorted(fams),
    )
    match = [fam for fam, members in regions.items() if sorted(members) == keys]
    row["pou_keys_equal_one_regions_agents"] = match[0] if len(match) == 1 else None
    row["that_region_axis"] = (axes[match[0]].value if len(match) == 1 else None)
    return row


def census(skip_torch=False) -> dict:
    rows = {}
    for key, builder in VARIANTS:
        if skip_torch and key in TORCH_KEYS:
            rows[key] = {"skipped": "torch variants not requested"}
            continue
        try:
            rows[key] = census_row(_unwrap(builder()))
        except Exception as exc:                                    # noqa: BLE001
            rows[key] = {"error": f"{type(exc).__name__}: {exc}"[:400]}
        print("  %-40s %s" % (key, json.dumps(
            {k: rows[key].get(k) for k in ("pou", "pou_keys_are_agent_ids",
                                           "pou_keys_equal_one_regions_agents",
                                           "n_overlapping_regions", "error")})),
            flush=True)
    with_pou = {k: r for k, r in rows.items() if r.get("pou")}
    summary = {
        "graphs_censused": sum(1 for r in rows.values() if "graph" in r),
        "graphs_with_a_partition": len(with_pou),
        "keys_are_agent_ids": sum(1 for r in with_pou.values()
                                  if r.get("pou_keys_are_agent_ids")),
        "keys_are_exactly_one_regions_agents": sum(
            1 for r in with_pou.values() if r.get("pou_keys_equal_one_regions_agents")),
        "that_region_is_overlapping": sum(
            1 for r in with_pou.values() if r.get("that_region_axis") == "overlapping"),
        "exceptions": sorted(k for k, r in with_pou.items()
                             if not r.get("pou_keys_equal_one_regions_agents")),
        "max_overlapping_regions_in_one_graph": max(
            (r.get("n_overlapping_regions", 0) for r in rows.values()), default=0),
    }
    out = {"rows": rows, "summary": summary}
    _write(os.path.join(OUT, "census.json"), out)
    return out


def main(argv) -> None:
    if not argv:
        print(__doc__)
        return
    cmd = argv[0]
    skip_torch = "--skip-torch" in argv
    only = None
    for a in argv:
        if a.startswith("--only="):
            only = set(a.split("=", 1)[1].split(","))
    if cmd == "capture":
        label = argv[1]
        t0 = time.perf_counter()
        res = capture(label, only=only, skip_torch=skip_torch)
        errs = {k: r["error"] for k, r in res["rows"].items() if "error" in r}
        print("captured %d, errors %d, in %.1f s" % (
            sum(1 for r in res["rows"].values() if "sha256" in r), len(errs),
            time.perf_counter() - t0))
        for k, e in errs.items():
            print("  ERROR %s: %s" % (k, e[:200]))
    elif cmd == "compare":
        out = compare(argv[1], argv[2])
        print("identical %d, differ %d, not compared %d" % (
            len(out["identical"]), len(out["differ"]), len(out["not_compared"])))
        for k in out["differ"]:
            print("  DIFFER", k)
        for k in out["not_compared"]:
            print("  NOT COMPARED", k, out["rows"][k])
    elif cmd == "census":
        out = census(skip_torch=skip_torch)
        print(json.dumps(out["summary"], indent=1))
    else:
        raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    main(sys.argv[1:])
