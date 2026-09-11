"""W172 -- the joining seams exist, and what joining costs is measured.

Tier 39 counted the union rung 9 would form -- `front_wing`, `cooling_loop` and
`powertrain`, eighteen agents across five families -- and found forty ports,
forty per-agent prolongations, five scale sets and ZERO per-pair declarations
against 153 pairs.  Its own caveat: that is the union's EXISTING structure, the
joining seams did not exist, and "each new seam needs two new ports, one per
side -- O(1) per seam" was an inspection and not a measurement.
`atlas/cases/integration_union.py` builds the three joins; this measures them.

Six measurements, each beside the control that makes it evidence:

  0. **The counter.**  Tier 39's published table reproduced from the six graphs by
     the same definitions, and the disjoint union reproducing its union row,
     before anything is counted on a new graph.
  1. **The joined union compiles**, beside the disjoint union built from the same
     three `build()` calls.
  2. **Each join's marginal cost at two family counts** -- J1 at 4 and 5, J2 at 4
     and 5 (with and without J3), J3 at 3 and 5 (with and without J2) -- as the
     difference between two graphs that differ by that join alone.  Every column
     is derived from the two graphs and none is typed.
  3. **Two insertion orders**, A (cooling first) and B (powertrain first), stage
     by stage, separating a subsystem's existing structure from what adding it
     and joining it costs.
  4. **J3's rotor faces four ways**: re-declared for the host, native (the
     control), per pair in the host's measure, per pair in each side's own.
  5. **The clocks**: `L7/R9` on the disjoint and joined unions, pointwise,
     time-integrated and reconciled, with who must re-declare what.
  6. **Physics controls**: the operating point J3 moves, the heat J2 carries, the
     mounted block's first law, the loop gains, and the device-down passivity
     defect on `wake_array`'s own graph, which says whether it is inherited.

Writes ``out/w172/w172.json`` after every section.  Uses ``out/w141/settled.npz``
(the settled `front_wing` field) and ``out/w136/settled.npz`` (`wing_fsi`'s, for
the counter).  No checkpoint is loaded; `wake_array` is built at
``kind="reference"``.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import collections                                                     # noqa: E402
import dataclasses                                                     # noqa: E402
import enum                                                            # noqa: E402
import hashlib                                                         # noqa: E402
import io                                                              # noqa: E402
import json                                                            # noqa: E402
import sys                                                             # noqa: E402
import time                                                            # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np                                                     # noqa: E402

from atlas import compile_scheme                                       # noqa: E402
from atlas.graph import FluxMatching                                   # noqa: E402
from atlas.cases import integration_union as IU                        # noqa: E402

OUT = os.path.join(HERE, "out", "w172")

#: Tier 39's table as published (wiki/log.md, tier 39), column for column.
COUNT_KEYS = ("K", "families", "ports", "per_agent_prolongations", "scale_sets",
              "per_pair")
TIER39 = {
    "ground_effect": dict(K=7, families=1, ports=16, per_agent_prolongations=16,
                          scale_sets=1, per_pair=0),
    "thermal_seam": dict(K=2, families=2, ports=2, per_agent_prolongations=2,
                         scale_sets=1, per_pair=0),
    "wing_fsi": dict(K=7, families=2, ports=16, per_agent_prolongations=16,
                     scale_sets=1, per_pair=0),
    "front_wing": dict(K=8, families=2, ports=18, per_agent_prolongations=18,
                       scale_sets=1, per_pair=0),
    "cooling_loop": dict(K=5, families=2, ports=10, per_agent_prolongations=10,
                         scale_sets=2, per_pair=0),
    "powertrain": dict(K=5, families=2, ports=12, per_agent_prolongations=12,
                       scale_sets=3, per_pair=0),
}
TIER39_UNION = dict(K=18, families=5, ports=40, per_agent_prolongations=40,
                    scale_sets=5, per_pair=0, pairs=153)

SUB = IU.SUBSYSTEMS
UNIONS = {
    "fw": (("front_wing",), ()),
    "fw+cl": (("front_wing", "cooling_loop"), ()),
    "fw+cl|J1": (("front_wing", "cooling_loop"), ("J1",)),
    "fw+pt": (("front_wing", "powertrain"), ()),
    "fw+pt|J3": (("front_wing", "powertrain"), ("J3",)),
    "cl+pt": (("cooling_loop", "powertrain"), ()),
    "cl+pt|J2": (("cooling_loop", "powertrain"), ("J2",)),
    "all": (SUB, ()),
    "all|J1": (SUB, ("J1",)),
    "all|J3": (SUB, ("J3",)),
    "all|J1+J2": (SUB, ("J1", "J2")),
    "all|J1+J3": (SUB, ("J1", "J3")),
    "all|J2+J3": (SUB, ("J2", "J3")),
    "all|J1+J2+J3": (SUB, IU.JOINS),
}
#: (join, graph without it, graph with it) -- every other join held fixed.
MARGINAL = (
    ("J1", "fw+cl", "fw+cl|J1"),
    ("J1", "all|J2+J3", "all|J1+J2+J3"),
    ("J2", "cl+pt", "cl+pt|J2"),
    ("J2", "all|J1", "all|J1+J2"),
    ("J2", "all|J1+J3", "all|J1+J2+J3"),
    ("J3", "fw+pt", "fw+pt|J3"),
    ("J3", "all|J1", "all|J1+J3"),
    ("J3", "all|J1+J2", "all|J1+J2+J3"),
)
ORDERS = {
    "A": ("fw", "fw+cl", "fw+cl|J1", "all|J1", "all|J1+J2+J3"),
    "B": ("fw", "fw+pt", "fw+pt|J3", "all|J3", "all|J1+J2+J3"),
}
SPLITS = {IU.CORE_SITE.seam: IU.CORE_SITE.seam + "_bypass",
          IU.ROTOR_SITE.seam: IU.ROTOR_SITE.seam + "_bypass"}
DEVICE_SEAMS = ("J1_core_up", "J1_core_down", "J3_rotor_up", "J3_rotor_down")
GLOBAL_FIELDS = ("decomposition", "partition_of_unity", "overlap", "overlap_cells",
                 "global_fields", "cross_points", "loop_gains", "macro_dt",
                 "flux_matching", "measured")
#: The state a join reads from its partner, as `integration_union.build` records
#: it.  ``p_ref`` is not here: it is a reconciliation constant and counted as one.
STATE_KEYS = ("mgu_omega", "rotor_u_ref_mean", "rad_ua", "block_outer_h",
              "block_outer_T", "current_op", "q_machine", "t_case_ref")


# ===========================================================================
# plumbing
# ===========================================================================


def settled(name):
    d = np.load(os.path.join(HERE, "out", name, "settled.npz"))
    return d["u"], d["v"]


def _retry(fn, attempts=40, pause=0.25):
    for k in range(attempts):
        try:
            return fn()
        except PermissionError:
            if k == attempts - 1:
                raise
            time.sleep(pause)


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set, frozenset)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        if x.size <= 16:
            return x.tolist()
        return {"shape": list(x.shape), "norm": float(np.linalg.norm(x))}
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, enum.Enum):
        return x.value
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return str(x)


def persist(res):
    path = os.path.join(OUT, "w172.json")
    tmp = path + ".tmp"

    def write():
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(clean(res), fh, indent=1)

    _retry(write)
    _retry(lambda: os.replace(tmp, path))
    return path


def show(v):
    """A graph-global field as something two graphs can be compared on."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, enum.Enum):
        return v.value
    if isinstance(v, dict):
        return {str(k): show(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [show(x) for x in v]
    if hasattr(v, "gain") and hasattr(v, "agents"):
        return {"loop": list(v.agents), "gain": float(v.gain)}
    if hasattr(v, "produced_by") and hasattr(v, "applies_to"):
        return {"field": v.name, "produced_by": show(v.produced_by),
                "applies_to": show(v.applies_to)}
    if hasattr(v, "probe_state") and hasattr(v, "source"):
        return f"MeasuredConstants(source={v.source!r})"
    return type(v).__name__


# ===========================================================================
# what is counted
# ===========================================================================


def nondim_key(p):
    return (p.port_type.value, tuple(sorted((k, float(x)) for k, x in p.nondim.items())))


def count(g) -> dict:
    """Tier 39's columns, exactly, and what a union adds to read."""
    ports = [(a.agent_id, p) for a in g.agents for p in a.capabilities.ports]
    dt = {a.agent_id: a.capabilities.dt_native for a in g.agents}
    multirate = sorted(c.seam_id for c in g.connections if dt[c.a[0]] != dt[c.b[0]])
    at_multirate = sorted({x for c in g.connections if c.seam_id in multirate
                           for x in (c.a[0], c.b[0])})
    try:
        cycles = [list(c) for c in g.directed_cycles()]
    except Exception as exc:                        # the budget, reported
        cycles = [f"not enumerated: {type(exc).__name__}"]
    K = len(g.agents)
    return {
        "K": K,
        "families": len({a.capabilities.governing_family for a in g.agents}),
        "ports": len(ports),
        "per_agent_prolongations": sum(1 for _a, p in ports if p.prolongation is not None),
        "scale_sets": len({nondim_key(p) for _a, p in ports if p.nondim}),
        "scale_sets_by_type": len({p.port_type.value for _a, p in ports if p.nondim}),
        "per_pair": sum(1 for c in g.connections if c.space is not None or c.prolongations),
        "pairs": K * (K - 1) // 2,
        "seams": len(g.connections),
        "open_ports": [list(x) for x in g.open_ports()],
        "cut_axis_declared": sorted(c.seam_id for c in g.connections
                                    if c.cut_axis is not None),
        "effort_normal_declared": sorted(c.seam_id for c in g.connections
                                         if c.effort_normal),
        "multirate_seams": multirate,
        "agents_at_multirate_seams": at_multirate,
        "agents_without_integrated_response": sorted(
            a.agent_id for a in g.agents
            if a.capabilities.boundary_response_integrated is None),
        "agents_off_the_macro_step": sorted(
            a.agent_id for a in g.agents
            if a.capabilities.dt_native is not None and a.capabilities.dt_native != g.macro_dt),
        "directed_cycles": cycles,
        "regions": {k: sorted(v) for k, v in g.regions().items()},
    }


def _arr_sig(x):
    if x is None:
        return None
    a = np.ascontiguousarray(np.asarray(x, dtype=float))
    return (tuple(a.shape), hashlib.sha256(a.tobytes()).hexdigest()[:16])


def port_sig(p):
    """A port's DECLARATION, free text excluded."""
    P = p.prolongation
    S = p.interface_space
    return (p.port_type.value, p.direction.value,
            tuple(sorted((k, float(x)) for k, x in (p.nondim or {}).items())),
            p.effective_resolution, p.motion_class.value, p.response_half.value,
            tuple(p.passengers or ()),
            None if P is None else (_arr_sig(P.matrix), _arr_sig(P.gram_V),
                                    _arr_sig(P.nondim_diag)),
            None if S is None else (S.dim, _arr_sig(S.gram)))


def convention_of(P):
    """(cell measure, cutoff in cells): what a prolongation says about its host."""
    if P is None:
        return None
    n, m = P.n_V, P.dim_M
    return {"n_V": n, "dim_M": m, "cell_measure": round(float(P.G_V[0, 0]), 15),
            "cutoff_cells": None if m <= 1 else round(2.0 * n / (m - 1), 6)}


def _conv_key(P):
    c = convention_of(P)
    return None if c is None else (c["cell_measure"], c["cutoff_cells"])


def record_sig(caps) -> dict:
    d = caps.as_dict()
    d.pop("ports", None)
    return d


def responses_at_base(g) -> dict:
    """Every port's own base and its response there -- what a join can move."""
    out = {}
    for a in g.agents:
        caps = a.capabilities
        for p in caps.ports:
            n = p.prolongation.n_V if p.prolongation is not None else 1
            try:
                base = (np.zeros(n) if caps.probe_base is None
                        else np.asarray(caps.probe_base(p.name), dtype=float).ravel())
                resp = np.asarray(caps.boundary_response(p.name, base.copy()),
                                  dtype=float).ravel()
                out[(a.agent_id, p.name)] = (base, resp)
            except Exception as exc:
                out[(a.agent_id, p.name)] = ("error", repr(exc))
    return out


def _same(x, y):
    if (not isinstance(x, tuple) or not isinstance(y, tuple)
            or isinstance(x[0], str) or isinstance(y[0], str)):
        return False, False
    return (x[0].shape == y[0].shape and bool(np.array_equal(x[0], y[0])),
            x[1].shape == y[1].shape and bool(np.array_equal(x[1], y[1])))


def compile_summary(g) -> dict:
    t0 = time.perf_counter()
    r = compile_scheme(g)
    elapsed = time.perf_counter() - t0
    rows = [(d.layer, d.rule, d.verdict.value, d.subject or "")
            for d in r.decisions.decisions]
    local = [[d.layer, d.rule, d.verdict.value, d.subject, d.message[:240]]
             for d in r.decisions.decisions
             if (d.subject or "").split("_", 1)[0] in IU.JOINS
             or (d.subject or "").endswith("_bypass")
             if d.verdict.value != "admit"]
    return {"r": r, "verdict": r.verdict.value,
            "refusals": sorted({f"{d.layer}/{d.rule}" for d in r.decisions.refusals}),
            "refusal_subjects": sorted({f"{d.layer}/{d.rule}@{d.subject}"
                                        for d in r.decisions.refusals}),
            "n_decisions": len(rows), "rows": rows, "compile_s": elapsed,
            "non_admit_on_joining_and_split_seams": local}


def make_entry(u, v, subsystems, joins, **kw) -> dict:
    g, meta = IU.build(u, v, subsystems=subsystems, joins=joins, **kw)
    return {"graph": g, "meta": meta,
            "subsystems": tuple(meta["info"]["subsystems"]),
            "joins": tuple(meta["info"]["joins"]),
            "count": count(g), "compile": compile_summary(g), "_responses": None}


def responses(entry) -> dict:
    if entry["_responses"] is None:
        entry["_responses"] = responses_at_base(entry["graph"])
    return entry["_responses"]


def public(entry) -> dict:
    c = entry["compile"]
    return {"name": entry["graph"].name, "subsystems": list(entry["subsystems"]),
            "joins": list(entry["joins"]), "count": entry["count"],
            "verdict": c["verdict"], "refusals": c["refusals"],
            "refusal_subjects": c["refusal_subjects"], "n_decisions": c["n_decisions"],
            "compile_s": c["compile_s"],
            "non_admit_on_joining_and_split_seams":
                c["non_admit_on_joining_and_split_seams"],
            "info": {k: val for k, val in entry["meta"]["info"].items() if k != "parts"}}


def _per_subject(rows):
    d = collections.defaultdict(list)
    for layer, rule, verdict, subject in rows:
        d[subject].append((layer, rule, verdict))
    return {s: sorted(x) for s, x in d.items()}


def _region_of(g):
    return {a: fam for fam, ids in g.regions().items() for a in ids}


# ===========================================================================
# a join's cost: the difference between two graphs that differ by it
# ===========================================================================


def cost(lb: str, la: str, U: dict, joins_added) -> dict:
    eb, ea = U[lb], U[la]
    gb, ga = eb["graph"], ea["graph"]
    kb, ka = eb["count"], ea["count"]
    pb = {(a.agent_id, p.name): p for a in gb.agents for p in a.capabilities.ports}
    pa = {(a.agent_id, p.name): p for a in ga.agents for p in a.capabilities.ports}
    sb = {c.seam_id: c for c in gb.connections}
    sa = {c.seam_id: c for c in ga.connections}

    added_seams = sorted(set(sa) - set(sb))
    removed_seams = sorted(set(sb) - set(sa))
    joining = [s for s in added_seams if s.split("_", 1)[0] in IU.JOINS]
    redeclared = [s for s in added_seams if s not in joining]
    added_ports = sorted(set(pa) - set(pb))
    removed_ports = sorted(set(pb) - set(pa))
    common = sorted(set(pa) & set(pb))
    changed = [k for k in common if port_sig(pa[k]) != port_sig(pb[k])]
    text_only = [k for k in common if k not in changed
                 and (pa[k].geometry, pa[k].note) != (pb[k].geometry, pb[k].note)]
    open_b = set(gb.open_ports())

    joining_ports, sides = set(), []
    for s in joining:
        c = sa[s]
        for side, partner in ((c.a, c.b), (c.b, c.a)):
            joining_ports.add(side)
            if side not in pb:
                status = "new"
            elif side in changed:
                status = "re-declared"
            elif side in open_b:
                status = "open port, reused unchanged"
            else:
                status = "reused unchanged"
            P_side = c.prolongations.get(side[0]) or pa[side].prolongation
            P_partner = c.prolongations.get(partner[0]) or pa[partner].prolongation
            prior = sorted({_conv_key(p.prolongation) for (ag, _n), p in pb.items()
                            if ag == side[0] and p.port_type is pa[side].port_type
                            and p.prolongation is not None}, key=str)
            key, pkey = _conv_key(P_side), _conv_key(P_partner)
            if key != pkey:
                origin = "mismatched with the partner"
            elif key in prior:
                origin = "the agent's own convention"
            elif prior:
                origin = "taken from the partner, away from the agent's own"
            else:
                origin = "taken from the partner: no prior port of this type"
            sides.append({"seam": s, "agent": side[0], "port": side[1],
                          "status": status, "convention": convention_of(P_side),
                          "partner_convention": convention_of(P_partner),
                          "content": origin,
                          "seam_level_transfer": bool(c.space is not None
                                                      or c.prolongations)})
    split_ports = [k for k in added_ports if k not in joining_ports]

    pp_b = {c.seam_id for c in gb.connections if c.space is not None or c.prolongations}
    pp_a = {c.seam_id for c in ga.connections if c.space is not None or c.prolongations}
    ss_b = {nondim_key(p) for p in pb.values() if p.nondim}
    ss_a = {nondim_key(p) for p in pa.values() if p.nondim}

    agents_b = {a.agent_id: a for a in gb.agents}
    agents_a = {a.agent_id: a for a in ga.agents}
    records = {}
    for aid in sorted(set(agents_a) & set(agents_b)):
        db, da = record_sig(agents_b[aid].capabilities), record_sig(agents_a[aid].capabilities)
        diff = sorted(k for k in set(db) | set(da) if db.get(k) != da.get(k))
        if diff:
            records[aid] = {k: [db.get(k), da.get(k)] for k in diff}

    rb, ra = responses(eb), responses(ea)
    touched = {x[0] for x in joining_ports} | {x for s in redeclared
                                              for x in (sa[s].a[0], sa[s].b[0])}
    moved = []
    for k in common:
        base_same, resp_same = _same(rb.get(k), ra.get(k))
        if not (base_same and resp_same):
            moved.append({"agent": k[0], "port": k[1], "base_moved": not base_same,
                          "response_moved": not resp_same,
                          "agent_on_a_new_or_split_seam": k[0] in touched})

    globals_changed = {f: [show(getattr(gb, f)), show(getattr(ga, f))]
                       for f in GLOBAL_FIELDS
                       if show(getattr(gb, f)) != show(getattr(ga, f))}

    cb, ca = eb["compile"], ea["compile"]
    mb = collections.Counter((l, r, v) for l, r, v, _s in cb["rows"])
    ma = collections.Counter((l, r, v) for l, r, v, _s in ca["rows"])
    psb, psa = _per_subject(cb["rows"]), _per_subject(ca["rows"])
    seam_changes = {s: {"before": psb.get(s), "after": psa.get(s)}
                    for s in sorted(set(sb) & set(sa)) if psb.get(s) != psa.get(s)}
    for old, new in SPLITS.items():
        if old in sb and new in sa and old not in sa:
            seam_changes[f"{old} -> {new}"] = {
                "before": psb.get(old), "after": psa.get(new),
                "same_rules_and_verdicts": psb.get(old) == psa.get(new)}
    agent_changes = {aid: {"before": psb.get(aid), "after": psa.get(aid)}
                     for aid in sorted(set(agents_a) & set(agents_b))
                     if psb.get(aid) != psa.get(aid)}
    graph_changes = {s: {"before": psb.get(s), "after": psa.get(s)}
                     for s in sorted(set(psb) | set(psa))
                     if s.startswith("<") and psb.get(s) != psa.get(s)}

    region_b, region_a = _region_of(gb), _region_of(ga)
    regions = {a: [region_b.get(a), region_a.get(a)]
               for a in sorted(set(region_a) | set(region_b))
               if region_b.get(a) != region_a.get(a)}
    cyc_b = {tuple(c) for c in kb["directed_cycles"] if isinstance(c, list)}
    cyc_a = {tuple(c) for c in ka["directed_cycles"] if isinstance(c, list)}

    ib, ia = eb["meta"]["info"], ea["meta"]["info"]
    pairs = {k: (ib.get(k), ia.get(k)) for k in STATE_KEYS}
    state_new = {k: list(x) for k, x in pairs.items()
                 if x[0] is None and x[1] is not None}
    state = {k: list(x) for k, x in pairs.items()
             if x[0] is not None and x[1] is not None and x[0] != x[1]}

    declared = {j: IU.JOIN_LEDGER[j]["declared_constants"] for j in joins_added}
    reconciled = {j: IU.JOIN_LEDGER[j]["reconciliation_constants"] for j in joins_added}
    prol_changed = [k for k in changed
                    if (_arr_sig(pa[k].prolongation.matrix if pa[k].prolongation else None),
                        _arr_sig(pa[k].prolongation.gram_V if pa[k].prolongation else None))
                    != (_arr_sig(pb[k].prolongation.matrix if pb[k].prolongation else None),
                        _arr_sig(pb[k].prolongation.gram_V if pb[k].prolongation else None))]

    tally = {
        "joining_seams": len(joining),
        "joining_ports_new": sum(1 for x in sides if x["status"] == "new"),
        "joining_ports_redeclared": sum(1 for x in sides if x["status"] == "re-declared"),
        "joining_ports_reused": sum(1 for x in sides if "reused" in x["status"]),
        "joining_ports_content_from_partner": sum(1 for x in sides
                                                  if x["content"].startswith("taken")),
        "joining_ports_mismatched": sum(1 for x in sides
                                        if x["content"].startswith("mismatched")),
        "existing_seams_removed": len(removed_seams),
        "existing_seams_redeclared": len(redeclared),
        "split_ports_new": len(split_ports),
        "ports_removed": len(removed_ports),
        "ports_redeclared_off_the_joining_seams": sum(1 for k in changed
                                                     if k not in joining_ports),
        "prolongations_new": sum(1 for k in added_ports if pa[k].prolongation is not None),
        "prolongations_redeclared": len(prol_changed),
        "per_pair_new": len(pp_a - pp_b),
        "scale_sets_new": len(ss_a - ss_b),
        "declared_constants": sum(len(x) for x in declared.values()),
        "reconciliation_constants": sum(len(x) for x in reconciled.values()),
        "agent_records_changed": len(records),
        "ports_moved_at_their_own_base": len(moved),
        "ports_moved_on_agents_off_the_new_seams": sum(
            1 for m in moved if not m["agent_on_a_new_or_split_seam"]),
        "state_data_introduced": len(state_new),
        "state_data_rederived": len(state),
        "graph_global_fields_changed": len(globals_changed),
        "cut_axis_declarations_new": len(set(ka["cut_axis_declared"])
                                         - set(kb["cut_axis_declared"])),
        "effort_normal_declarations_new": len(set(ka["effort_normal_declared"])
                                              - set(kb["effort_normal_declared"])),
        "existing_seams_whose_decisions_changed": sum(
            1 for k, x in seam_changes.items()
            if "->" not in k or not x.get("same_rules_and_verdicts")),
        "refusals_new": len(set(ca["refusal_subjects"]) - set(cb["refusal_subjects"])),
        "directed_cycles_new": len(cyc_a - cyc_b),
        "agents_whose_region_changed": len(regions),
        "multirate_seams_new": len(set(ka["multirate_seams"]) - set(kb["multirate_seams"])),
        "agents_newly_at_a_multirate_seam": len(set(ka["agents_at_multirate_seams"])
                                                - set(kb["agents_at_multirate_seams"])),
    }
    literal = (tally["joining_ports_new"] == 2 * tally["joining_seams"]
               and tally["joining_ports_redeclared"] == 0
               and tally["joining_ports_reused"] == 0)
    beyond = {k: tally[k] for k in (
        "existing_seams_removed", "existing_seams_redeclared", "split_ports_new",
        "ports_removed", "ports_redeclared_off_the_joining_seams",
        "prolongations_redeclared", "per_pair_new", "scale_sets_new",
        "reconciliation_constants", "agent_records_changed",
        "ports_moved_at_their_own_base", "state_data_rederived",
        "graph_global_fields_changed", "cut_axis_declarations_new",
        "existing_seams_whose_decisions_changed", "refusals_new",
        "directed_cycles_new", "agents_whose_region_changed",
        "multirate_seams_new") if tally[k]}
    return {
        "from": lb, "to": la, "joins_added": list(joins_added),
        "families": [kb["families"], ka["families"]], "K": [kb["K"], ka["K"]],
        "tally": tally,
        "prediction": {
            "statement": "each new seam needs two new ports, one per side -- O(1) per seam",
            "two_new_ports_per_new_seam": literal,
            "and_nothing_else": literal and not beyond,
            "what_else": beyond,
        },
        "seams": {"joining": joining, "redeclared": redeclared, "removed": removed_seams},
        "joining_sides": sides,
        "ports": {"added": [list(k) for k in added_ports],
                  "removed": [list(k) for k in removed_ports],
                  "redeclared": [list(k) for k in changed],
                  "text_only": [list(k) for k in text_only],
                  "split": [list(k) for k in split_ports]},
        "per_pair_new": sorted(pp_a - pp_b),
        "scale_sets_new": sorted(str(x) for x in ss_a - ss_b),
        "agent_records_changed": records,
        "ports_moved_at_their_own_base": moved,
        "state_data_introduced": state_new,
        "state_data_rederived": state,
        "declared_constants": declared,
        "reconciliation_constants": reconciled,
        "graph_global_fields_changed": globals_changed,
        "compile": {
            "verdict": [cb["verdict"], ca["verdict"]],
            "refusals": [cb["refusals"], ca["refusals"]],
            "rule_verdicts_added": sorted([list(k) + [n] for k, n in (ma - mb).items()]),
            "rule_verdicts_removed": sorted([list(k) + [n] for k, n in (mb - ma).items()]),
            "existing_seam_decisions_changed": seam_changes,
            "agent_decisions_changed": agent_changes,
            "graph_and_region_decisions_changed": graph_changes,
            "joining_seam_decisions": {s: psa.get(s) for s in joining},
        },
        "directed_cycles_new": sorted(list(c) for c in cyc_a - cyc_b),
        "agents_whose_region_changed": regions,
    }


def per_join_summary(marginal: list) -> dict:
    out = {}
    for j in IU.JOINS:
        rows = [m for m in marginal if m["joins_added"] == [j]]
        keys = sorted(rows[0]["tally"]) if rows else []
        differ = {k: {f'{m["from"]} (F={m["families"][1]})': m["tally"][k] for m in rows}
                  for k in keys if len({m["tally"][k] for m in rows}) > 1}
        out[j] = {
            "contexts": [{"from": m["from"], "to": m["to"],
                          "families_after": m["families"][1], "K": m["K"][1],
                          "tally": m["tally"],
                          "two_new_ports_per_new_seam":
                              m["prediction"]["two_new_ports_per_new_seam"],
                          "and_nothing_else": m["prediction"]["and_nothing_else"]}
                         for m in rows],
            "family_counts": sorted({m["families"][1] for m in rows}),
            "identical_in_every_context": not differ,
            "columns_that_differ_between_contexts": differ,
        }
    return out


def stage_row(lb: str, la: str, U: dict, orig: dict, own: dict) -> dict:
    eb, ea = U[lb], U[la]
    kb, ka = eb["count"], ea["count"]
    gb, ga = eb["graph"], ea["graph"]
    added = [s for s in SUB if s in ea["subsystems"] and s not in eb["subsystems"]]
    jadd = [j for j in IU.JOINS if j in ea["joins"] and j not in eb["joins"]]
    row = {"from": lb, "to": la, "added_subsystems": added, "joins_added": jadd,
           "K": [kb["K"], ka["K"]], "families": [kb["families"], ka["families"]],
           "delta": {k: ka[k] - kb[k] for k in ("ports", "per_agent_prolongations",
                                                 "scale_sets", "per_pair", "seams")},
           "verdict": [eb["compile"]["verdict"], ea["compile"]["verdict"]],
           "refusals": [eb["compile"]["refusals"], ea["compile"]["refusals"]]}
    if added:
        structure = {s: dict(TIER39[s]) for s in added}
        own_seams = sum(len(orig[s].connections) for s in added)
        row["existing_structure_of_what_was_added"] = structure
        row["delta_is_exactly_the_added_structure"] = all(
            row["delta"][k] == sum(structure[s][k] for s in added)
            for k in ("ports", "per_agent_prolongations", "per_pair")
        ) and row["delta"]["seams"] == own_seams
        rb, ra = _region_of(gb), _region_of(ga)
        regions = {}
        for a in ga.agents:
            src = next((s for s in added
                        if any(x.agent_id == a.agent_id for x in own[s].agents)), None)
            before = _region_of(own[src]).get(a.agent_id) if src else rb.get(a.agent_id)
            if before != ra.get(a.agent_id):
                regions[a.agent_id] = {"in_its_own_graph_or_before": before,
                                       "in_the_union": ra.get(a.agent_id),
                                       "from": src or "already in the union"}
        row["union_reconciliation"] = {
            "graph_global_fields": {
                f: {"union_before": show(getattr(gb, f)),
                    **{f"{s} alone": show(getattr(orig[s], f)) for s in added},
                    "union_after": show(getattr(ga, f))}
                for f in GLOBAL_FIELDS
                if show(getattr(ga, f)) != show(getattr(gb, f))
                or any(show(getattr(orig[s], f)) != show(getattr(ga, f)) for s in added)},
            "cut_axis_declarations_new": sorted(set(ka["cut_axis_declared"])
                                                - set(kb["cut_axis_declared"])),
            "agents_without_integrated_response": [
                len(kb["agents_without_integrated_response"]),
                len(ka["agents_without_integrated_response"])],
            "agents_off_the_macro_step": [kb["agents_off_the_macro_step"],
                                          ka["agents_off_the_macro_step"]],
            "multirate_seams": [kb["multirate_seams"], ka["multirate_seams"]],
            "agents_whose_region_changed": regions,
        }
    if jadd:
        row["joining_cost"] = cost(lb, la, U, jadd)["tally"]
    return row


# ===========================================================================
# the measurements
# ===========================================================================


def counter_control(u, v) -> tuple[dict, dict]:
    from atlas.cases import (cooling_loop, front_wing, ground_effect, powertrain,
                             thermal_seam, wing_fsi)

    uw, vw = settled("w136")
    graphs = {
        "ground_effect": ground_effect.build(u, v, motion=False)[0],
        "thermal_seam": thermal_seam.build()[0],
        "wing_fsi": wing_fsi.build(uw, vw, motion=False)[0],
        "front_wing": front_wing.build(u, v, motion=False)[0],
        "cooling_loop": cooling_loop.build()[0],
        "powertrain": powertrain.build()[0],
    }
    measured = {k: {c: count(g)[c] for c in COUNT_KEYS} for k, g in graphs.items()}
    disjoint = IU.build(u, v, joins=())[0]
    cu = count(disjoint)
    mu = {c: cu[c] for c in COUNT_KEYS + ("pairs",)}
    res = {"published": {**TIER39, "union": TIER39_UNION},
           "measured": {**measured, "union": mu},
           "equal": all(measured[k] == TIER39[k] for k in TIER39) and mu == TIER39_UNION}
    return res, {k: graphs[k] for k in SUB}


def rotor_forms(u, v, U: dict) -> dict:
    out = {}
    for form in IU.ROTOR_PORT_FORMS:
        label = "all|J1+J2+J3" if form == "host" else f"all|J1+J2+J3|rotor-{form}"
        if label not in U:
            U[label] = make_entry(u, v, SUB, IU.JOINS, rotor_ports=form)
        e = U[label]
        g, r = e["graph"], e["compile"]["r"]
        seams = {}
        for sid in ("J3_rotor_up", "J3_rotor_down"):
            c = g.connection(sid)
            P = {aid: (c.prolongations.get(aid) or g.agent(aid).port(pn).prolongation)
                 for aid, pn in (c.a, c.b)}
            win = next(k for k in P if k != "ROTOR")
            m = min(P["ROTOR"].dim_M, P[win].dim_M)
            ratios = [float(np.linalg.norm(P["ROTOR"].matrix[:, k])
                            / np.linalg.norm(P[win].matrix[:, k])) for k in range(m)]
            seams[sid] = {
                "rotor_convention": convention_of(P["ROTOR"]),
                "window_convention": convention_of(P[win]),
                "trace_amplitude_ratio_rotor_over_window": sorted(
                    {round(x, 12) for x in ratios}),
                "same_trace_for_the_same_coefficient": bool(np.allclose(
                    P["ROTOR"].matrix[:, :m], P[win].matrix[:, :m], rtol=0.0,
                    atol=1e-12)),
                "decisions": [[d.layer, d.rule, d.verdict.value, d.message[:300],
                               clean(d.evidence)]
                              for d in r.decisions.decisions if d.subject == sid],
            }
        out[form] = {"graph": g.name, "verdict": e["compile"]["verdict"],
                     "refusals": e["compile"]["refusal_subjects"],
                     "seams": seams,
                     "cost_against_the_union_without_J3": cost(
                         "all|J1+J2", label, U, ["J3"])["tally"]}
    return out


def clocks(u, v) -> dict:
    variants = (("native", FluxMatching.POINTWISE),
                ("native", FluxMatching.TIME_INTEGRATED),
                ("reconciled", FluxMatching.POINTWISE))
    out = {}
    native_dt = {}
    for joins in ((), IU.JOINS):
        for clock, fm in variants:
            e = make_entry(u, v, SUB, joins, clocks=clock, flux_matching=fm)
            g, r = e["graph"], e["compile"]["r"]
            dts = {a.agent_id: a.capabilities.dt_native for a in g.agents}
            key = ("joined" if joins else "disjoint") + f"|{clock}|{fm.value}"
            if clock == "native" and fm is FluxMatching.POINTWISE:
                native_dt[bool(joins)] = dts
            k = e["count"]
            out[key] = {
                "graph": g.name, "verdict": e["compile"]["verdict"],
                "refusals": e["compile"]["refusal_subjects"],
                "macro_dt": g.macro_dt,
                "native_steps": sorted(set(v_ for v_ in dts.values() if v_ is not None)),
                "agents_off_the_macro_step": k["agents_off_the_macro_step"],
                "agents_without_integrated_response": k["agents_without_integrated_response"],
                "multirate_seams": k["multirate_seams"],
                "agents_at_multirate_seams": k["agents_at_multirate_seams"],
                "r9_decisions": [[d.layer, d.rule, d.verdict.value, d.message[:400]]
                                 for d in r.decisions.decisions
                                 if d.layer == "L7" and d.rule.startswith("R9")],
                "_dt": dts,
            }
    for key, row in out.items():
        base = native_dt[key.startswith("joined")]
        row["dt_native_redeclared_against_native"] = sorted(
            a for a, d in row.pop("_dt").items() if d != base.get(a))
    return out


def physics(u, U: dict) -> dict:
    from atlas.cases import cooling_loop as CL
    from atlas.cases import powertrain as PT

    ref = PT.CircuitSolve().solve()
    joined = U["all|J1+J2+J3"]["meta"]["info"]
    without_j3 = U["cl+pt|J2"]["meta"]["info"]
    op = joined["operating_point"]
    tc = np.full(CL.N_SEAM, CL.T_COOLANT_0)
    first_law = {
        "BlockAgent (declared t_source)": CL.BlockAgent().energy_balance(tc),
        "MountedBlock, q_machine = Q_SOURCE (J2 without J3)":
            IU.MountedBlock(q_machine=without_j3["q_machine"]).energy_balance(tc),
        "MountedBlock, q_machine at the joined operating point (J2 with J3)":
            IU.MountedBlock(q_machine=joined["q_machine"]).energy_balance(tc),
    }
    legs = CL.make_legs(4)
    with_core = dict(legs)
    core = IU.CoreRadiator(u_air=IU.ring_velocity(u, IU.CORE_SITE))
    with_core["RAD"] = core
    gains = {"cooling_loop": CL.composed_loop_gain(legs, CL.LOOP_ORDER),
             "cooling_loop_with_the_core_in_the_flow":
                 CL.composed_loop_gain(with_core, CL.LOOP_ORDER)}
    ua_ref = core.ua
    core.u_air = core.u_air * 1.1
    ua_fast = core.ua
    core.u_air = core.u_air / 1.1
    u_rotor = IU.ring_velocity(u, IU.ROTOR_SITE)
    return {
        "powertrain_reference_operating_point": {
            k: getattr(ref, k) for k in ("omega", "induction", "current", "p_mech",
                                         "mgu_valid", "rotor_valid", "residual")},
        "operating_point_in_the_host_flow_J3": {
            k: op[k] for k in ("omega", "induction", "current", "p_mech", "mgu_valid",
                               "rotor_valid", "residual", "torque_residual")},
        "rotor_inflow": {"mean": float(np.mean(u_rotor)), "min": float(np.min(u_rotor)),
                         "max": float(np.max(u_rotor)), "declared_by_powertrain": PT.U_REF},
        "core_inflow_mean": float(np.mean(core.u_air)),
        "J2_heat": {
            "p_ref_W_per_power_unit": joined["p_ref"],
            "q_machine_W_without_J3": without_j3["q_machine"],
            "q_machine_W_with_J3": joined["q_machine"],
            "ratio_with_over_without": joined["q_machine"] / without_j3["q_machine"],
            "t_case_ref_K_without_J3": without_j3["t_case_ref"],
            "t_case_ref_K_with_J3": joined["t_case_ref"],
            "Q_SOURCE": CL.Q_SOURCE,
        },
        "block_first_law_at_the_release_state": first_law,
        "loop_gains": gains,
        "loop_gain_unchanged_by_J1_at_the_build_state":
            gains["cooling_loop"] == gains["cooling_loop_with_the_core_in_the_flow"],
        "core_UA": {"at_build": ua_ref, "UA_RAD": CL.UA_RAD,
                    "air_plus_10_percent": ua_fast,
                    "relative_change": ua_fast / ua_ref - 1.0},
    }


def device_down_seams(U: dict) -> dict:
    r = U["all|J1+J2+J3"]["compile"]["r"]
    union = {s: [[d.layer, d.rule, d.verdict.value, d.message[:300], clean(d.evidence)]
                 for d in r.decisions.decisions
                 if d.subject == s and d.layer == "L4"]
             for s in DEVICE_SEAMS}
    out = {"union": union}
    try:
        from atlas.cases import wake_array as WA

        t0 = time.perf_counter()
        gw, _ = WA.build(np.ones((WA.NY, WA.NX)), np.zeros((WA.NY, WA.NX)),
                         kind="reference")
        rw = compile_scheme(gw)
        rotor_seams = sorted(c.seam_id for c in gw.connections
                             if c.a[0] in {x.rotor_id for x in WA.ROTORS})
        out["wake_array_reference_uniform_flow"] = {
            "graph": gw.name, "verdict": rw.verdict.value,
            "compile_s": time.perf_counter() - t0,
            "seams": {s: [[d.layer, d.rule, d.verdict.value, d.message[:300]]
                          for d in rw.decisions.decisions
                          if d.subject == s and d.layer == "L4"]
                      for s in rotor_seams},
        }
    except Exception as exc:
        out["wake_array_reference_uniform_flow"] = {"error": repr(exc)}
    return out


def main() -> None:
    t0 = time.perf_counter()
    os.makedirs(OUT, exist_ok=True)
    u, v = settled("w141")
    res = {"what": "W172 -- the joining seams, and the O(K) count across them",
           "date": time.strftime("%Y-%m-%d"),
           "state": "out/w141/settled.npz (front_wing's settled field)",
           "sites": {"rotor": dataclasses.asdict(IU.ROTOR_SITE),
                     "core": dataclasses.asdict(IU.CORE_SITE)},
           "ledger": IU.JOIN_LEDGER}

    res["counter_control"], orig = counter_control(u, v)
    print("0. counter control: Tier 39 reproduced =", res["counter_control"]["equal"],
          flush=True)
    persist(res)

    own = {s: IU.build(u, v, subsystems=(s,), joins=())[0] for s in SUB}
    U = {}
    for label, (subs, joins) in UNIONS.items():
        U[label] = make_entry(u, v, subs, joins)
        e = U[label]
        print("   %-14s K=%2d F=%d ports=%2d seams=%2d per_pair=%d verdict=%s %s"
              % (label, e["count"]["K"], e["count"]["families"], e["count"]["ports"],
                 e["count"]["seams"], e["count"]["per_pair"], e["compile"]["verdict"],
                 e["compile"]["refusals"]), flush=True)
    res["unions"] = {label: public(e) for label, e in U.items()}
    persist(res)

    res["marginal"] = [cost(lb, la, U, [j]) for j, lb, la in MARGINAL]
    res["per_join"] = per_join_summary(res["marginal"])
    print("2. per join:", flush=True)
    for j, s in res["per_join"].items():
        print("   %s families %s identical %s differ %s" % (
            j, s["family_counts"], s["identical_in_every_context"],
            sorted(s["columns_that_differ_between_contexts"])), flush=True)
        for c in s["contexts"]:
            t = c["tally"]
            print("      from %-12s F=%d seams %d new ports %d redeclared %d reused %d "
                  "split seams %d per-pair %d moved %d (off-seam %d) state +%d ~%d "
                  "literal %s"
                  % (c["from"], c["families_after"], t["joining_seams"],
                     t["joining_ports_new"], t["joining_ports_redeclared"],
                     t["joining_ports_reused"], t["existing_seams_redeclared"],
                     t["per_pair_new"], t["ports_moved_at_their_own_base"],
                     t["ports_moved_on_agents_off_the_new_seams"],
                     t["state_data_introduced"], t["state_data_rederived"],
                     c["and_nothing_else"]), flush=True)
    persist(res)

    res["orders"] = {k: [stage_row(a, b, U, orig, own) for a, b in zip(seq, seq[1:])]
                     for k, seq in ORDERS.items()}
    persist(res)

    res["rotor_forms"] = rotor_forms(u, v, U)
    for form, row in res["rotor_forms"].items():
        print("4. rotor %-15s verdict %s refusals %s ratio %s" % (
            form, row["verdict"], row["refusals"],
            row["seams"]["J3_rotor_up"]["trace_amplitude_ratio_rotor_over_window"]),
            flush=True)
    persist(res)

    res["clocks"] = clocks(u, v)
    for key, row in res["clocks"].items():
        print("5. %-40s %s %s off-macro %d silent %d multirate seams %d" % (
            key, row["verdict"], row["refusals"], len(row["agents_off_the_macro_step"]),
            len(row["agents_without_integrated_response"]), len(row["multirate_seams"])),
            flush=True)
    persist(res)

    res["physics"] = physics(u, U)
    print("6. physics:", json.dumps(clean({k: res["physics"][k] for k in (
        "operating_point_in_the_host_flow_J3", "J2_heat", "loop_gain_unchanged_by_J1_at_the_build_state")}))[:900],
        flush=True)
    persist(res)

    res["device_down_seams"] = device_down_seams(U)
    res["elapsed_seconds"] = time.perf_counter() - t0
    path = persist(res)
    print("wrote", path, "in %.1f s" % res["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    main()
