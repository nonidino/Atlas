"""cot_phase2_decisive -- the coolant plan's Phase 2: the decisive test, with no training.

[[chart-operator-stage-0b-and-1-plan]] section 5, Phase 2: TL0 for CO-T and the classical
alternatives, each priced, on the development layouts only (`atlas/studio/cot_layouts`);
"stop if any classical option meets E4 under the price bar.  Then the coolant needs no
network, and that is reported as CO-D's result was."

**Fixed before this script's first run** (the owner's answers in chat, 2026-11-19), and
written into the record word for word (`CRITERIA`):

- E4 ("Floor + CO-D margins"): an arm meets E4 on a development problem when, against the
  full-domain solve on the same mesh, its relative energy error (the diffusion norm) is at
  most the classical 16-mode expert's own plus 0.02, its largest nodal error at most the
  classical's plus 0.01 of the range, and its outlet's mixing-cup temperature error at most
  the classical's plus 0.01 of the range;
- the price bar ("0.90 of classical"): an option's price is the median time of its whole
  coupled solve (every piece's response and the host's coupling) over the classical
  expert's, on the same problem, timed in turns; its price over the development problems
  is the geometric mean; the bar is 0.90;
- the stop rule: Phase 2 stops, with no network, when some classical option meets E4 on
  every development problem at a price under the bar;
- the A/A check: the classical expert timed twice in every turn; a problem whose A/A ratio
  falls outside [0.90, 1.10] has no price judged (it is reported, not counted).

**The options** (each a whole coupled solve on the registered cut unless named):
- ``TL0-<n_c>``: TL0 for CO-T, the bilinear coarse level at n_c = 8, 16, 32 (and 64 at
  n = 128): "a raised n_c";
- ``MsFEM-16``, ``MsFEM-OS-16``: TL0 with the multiscale coarse basis, conforming and
  oversampled (Hou and Wu 1997; Hou, Wu and Cai 1999): "a streamline-adapted classical
  coarse basis";
- ``finer``: the classical expert on the cut with twice the columns and two rows per layer
  at half the grid, the same mesh: "finer cutting";
- ``finer-TL0-16``: TL0 at n_c = min(16, n/2) on that finer cut;
- ``classical``: the floor, price 1 by definition.

**Also measured** (plan section 4, the safety switch): TL0-16's dual residual on every
piece, exactly and by the two-level Schwarz estimate and by Jacobi, with each one's cost
over the classical piece solve's, and the Schwarz estimate's ratio to the exact one.

BLAS threads are fixed at 4 before numpy loads and checked with threadpoolctl.  Writes
``out/chartop/cot/phase2-decisive.json`` and ``.txt``.  ``--quick``: one problem, two
turns, small grids, for the tests; nothing is judged.

    python scripts/cot_phase2_decisive.py [--quick] [--out DIR]
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_v] = "4"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

from atlas.chartop import fe  # noqa: E402
from atlas.chartop import transport as tr  # noqa: E402
from atlas.chartop import transport_coarse as tco  # noqa: E402
from atlas.chartop import transport_switch as sw  # noqa: E402
from atlas.chartop.host import monolithic  # noqa: E402
from atlas.studio import cot  # noqa: E402
from atlas.studio import cot_layouts as CL  # noqa: E402

#: fixed before the first run (the owner, 2026-11-19); see the module docstring
E4_ENERGY_MARGIN, E4_MAX_MARGIN, E4_OUTLET_MARGIN = 0.02, 0.01, 0.01
PRICE_BAR = 0.90
AA_BAND = (0.90, 1.10)
TURNS = 5
CRITERIA = {
    "E4": "against the full-domain solve on the same mesh: energy error <= the classical 16-mode "
          "expert's + 0.02; max nodal error <= the classical's + 0.01 of the range; outlet "
          "mixing-cup temperature error <= the classical's + 0.01 of the range (the owner, "
          "2026-11-19: 'Floor + CO-D margins')",
    "price": "the median time of the whole coupled solve over the classical expert's on the same "
             "problem, timed in turns; over the development problems, the geometric mean; bar "
             "0.90 (the owner, 2026-11-19: '0.90 of classical')",
    "stop": "Phase 2 stops with no network when some classical option meets E4 on every "
            "development problem at a geometric-mean price under the bar",
    "A/A": "the classical expert timed twice per turn; outside [0.90, 1.10] the problem's prices "
           "are reported, not judged",
}


def _blas() -> dict:
    try:
        from threadpoolctl import threadpool_info
        pools = [{"api": p.get("user_api"), "internal": p.get("internal_api"),
                  "threads": p.get("num_threads")} for p in threadpool_info()]
    except Exception as exc:                      # threadpoolctl is optional
        pools = [{"error": f"{type(exc).__name__}"}]
    blas = [p for p in pools if p.get("api") == "blas"]
    return {"pools": pools, "fixed_at_4": bool(blas) and all(p["threads"] == 4 for p in blas)}


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True,
                              text=True, timeout=30).stdout.strip()
    except Exception:
        return "unknown"


def _write(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    for k in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.5 * (k + 1))
    os.replace(tmp, path)


def _gap(La, fa, Lb, fb) -> float | None:
    """The largest difference between two cuts' fields at the same nodes, the nodes matched
    by distance (within 1e-12 of the plate's size), or None if the node sets differ.  (The
    first run keyed nodes by coordinates rounded to 1 nm, and at n = 128 many coordinates
    sit on half a nanometre, so identical meshes read as different: fixed 2026-11-19, after
    that run; a diagnostic, no criterion.)"""
    from scipy.spatial import cKDTree
    A = np.concatenate([p.chart.X.reshape(-1, 2) for p in La.problems])
    B = np.concatenate([p.chart.X.reshape(-1, 2) for p in Lb.problems])
    ua = np.concatenate([np.asarray(u, float).ravel() for u in fa])
    ub = np.concatenate([np.asarray(u, float).ravel() for u in fb])
    tol = 1e-12 * float(np.ptp(A, axis=0).max())
    d, k = cKDTree(B).query(A)
    d2, _ = cKDTree(A).query(B)
    if d.max() > tol or d2.max() > tol:
        return None
    return float(np.max(np.abs(ua - ub[k])))


def arms_for(n: int, quick: bool) -> list:
    """(name, which cut, expert factory)."""
    ncs = [nc for nc in (8, 16, 32, 64) if nc < n] if not quick else [8]
    out = [(f"TL0-{nc}", "base", (lambda nc=nc: tco.TL0TransportExpert(n_c=nc))) for nc in ncs]
    if n >= 32:
        out += [("MsFEM-16", "base", lambda: tco.TL0TransportExpert(n_c=min(16, n // 2), basis="msfem")),
                ("MsFEM-OS-16", "base", lambda: tco.TL0TransportExpert(n_c=min(16, n // 2), basis="msfem-os"))]
    out += [("finer", "fine", lambda: tr.ClassicalTransportExpert()),
            ("finer-TL0-16", "fine", lambda: tco.TL0TransportExpert(n_c=min(16, n // 4)))]
    return out


def problem(name: str, tier: str, quick: bool, turns: int) -> dict:
    spec, _g = CL.build(name, tier)
    n = CL.CUT[name]["n"][tier] if not quick else 32
    cols = CL.CUT[name]["cols"]
    from atlas.studio.pieces import cut_channels
    base = cut_channels(spec, n, cols=cols)
    fine = cut_channels(spec, n // 2, cols=2 * cols, rows=2)
    Lb, Lf = cot.layout(base, spec), cot.layout(fine, spec)
    mono_b, mono_f = monolithic(Lb), monolithic(Lf)
    ref_gap = _gap(Lb, mono_b.fields, Lf, mono_f.fields)
    pe = max(p.peclet() for p in Lb.problems)
    # accuracy, once per arm
    floor = Lb.solve(tr.ClassicalTransportExpert()).errors(mono_b.fields)
    arms = arms_for(n, quick)
    acc = {"classical": floor}
    for aname, which, make in arms:
        L, mono = (Lb, mono_b) if which == "base" else (Lf, mono_f)
        acc[aname] = L.solve(make()).errors(mono.fields)
    e4 = {}
    for aname, er in acc.items():
        ok = (er["energy_rel"] <= floor["energy_rel"] + E4_ENERGY_MARGIN
              and er["max_rel"] <= floor["max_rel"] + E4_MAX_MARGIN
              and (er["outlet_bulk_err_rel"] or 0.0) <= (floor["outlet_bulk_err_rel"] or 0.0) + E4_OUTLET_MARGIN)
        e4[aname] = bool(ok)
    # price, in turns: the classical twice (A/A), every option, the full domain beside
    times = {k: [] for k in ["classical", "classical (A/A)"] + [a for a, _w, _m in arms] + ["full domain"]}
    plan = [("classical", "base", lambda: tr.ClassicalTransportExpert()),
            ("classical (A/A)", "base", lambda: tr.ClassicalTransportExpert())] + list(arms)
    for t in range(turns):
        order = plan if t % 2 == 0 else plan[::-1]
        for aname, which, make in order:
            L = Lb if which == "base" else Lf
            ex = make()
            t0 = time.perf_counter()
            resp = ex.respond(L.problems, L.pb)
            L.couple(resp)
            times[aname].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        monolithic(Lb)
        times["full domain"].append(time.perf_counter() - t0)
    med = {k: float(np.median(v)) for k, v in times.items()}
    aa = med["classical (A/A)"] / med["classical"]
    price = {k: med[k] / med["classical"] for k in med}
    aa_ok = AA_BAND[0] <= aa <= AA_BAND[1]
    # the switch's residual, on TL0-16's answers (or the largest coarse size below n)
    nc = min(16, n // 2)
    tl0 = tco.TL0TransportExpert(n_c=nc)
    sw_rows = []
    for p in Lb.problems:
        Q = Lb.pb(p.n).Q
        r = tl0.response(p, Q)
        t0 = time.perf_counter()
        tr.exact_response(p, Q)
        t_cl = time.perf_counter() - t0
        row = {"label": p.label, "flows": bool(p.flows)}
        for mode in ("exact", "schwarz", "jacobi"):
            t0 = time.perf_counter()
            et = sw.piece_eta(p, r, Q, mode, n_c=nc)
            row[mode] = et["total"]
            row[f"{mode}_cost"] = (time.perf_counter() - t0) / t_cl
        sw_rows.append(row)
    ratio = [r_["schwarz"] / r_["exact"] for r_ in sw_rows if r_["exact"] > 0]
    switch = {"n_c": nc, "pieces": len(sw_rows),
              "cost_over_classical_piece": {m: float(np.median([r_[f"{m}_cost"] for r_ in sw_rows]))
                                            for m in ("exact", "schwarz", "jacobi")},
              "schwarz_over_exact": [float(min(ratio)), float(max(ratio))] if ratio else None,
              "jacobi_over_exact": [float(min(r_["jacobi"] / r_["exact"] for r_ in sw_rows if r_["exact"] > 0)),
                                    float(max(r_["jacobi"] / r_["exact"] for r_ in sw_rows if r_["exact"] > 0))]}
    return {"layout": name, "tier": tier, "n": n, "cols": cols, "pieces": len(base.pieces),
            "fine_pieces": len(fine.pieces), "pe_max": pe, "u_coolant": spec.physics.params["u_coolant"],
            "nodes": mono_b.nodes, "references_agree": ref_gap,
            "floor": floor, "accuracy": acc, "e4": e4, "price": price, "aa": aa, "aa_ok": aa_ok,
            "turns": turns, "switch": switch}


def judge(problems: list) -> dict:
    opts = [k for k in problems[0]["accuracy"] if k != "classical"]
    out = {}
    for k in opts:
        e4_all = all(p["e4"][k] for p in problems)
        priced = [p["price"][k] for p in problems if p["aa_ok"]]
        gm = float(math.exp(np.mean(np.log(priced)))) if priced else None
        out[k] = {"e4_every_problem": e4_all, "e4_count": sum(p["e4"][k] for p in problems),
                  "price_geomean": gm, "priced_problems": len(priced),
                  "stops_phase2": bool(e4_all and gm is not None and gm < PRICE_BAR)}
    stop = [k for k, v in out.items() if v["stops_phase2"]]
    return {"options": out, "stop": bool(stop), "stopped_by": stop}


def run(quick: bool, out_dir: str) -> dict:
    t0 = time.time()
    rec = {"what": "CO-T Phase 2: the decisive test, no training (the coolant plan)",
           "written": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "commit": _commit(), "quick": quick, "criteria": CRITERIA, "blas": _blas(),
           "layouts": "development only (atlas/studio/cot_layouts.py)", "problems": []}
    try:
        from atlas.workbench import machine
        rec["machine"] = machine.machine_record()
    except Exception as exc:
        rec["machine"] = {"error": type(exc).__name__}
    todo = [("dev-block", "slow")] if quick else [(n_, t_) for n_ in CL.NAMES for t_ in CL.TIERS]
    for name, tier in todo:
        p = problem(name, tier, quick, 2 if quick else TURNS)
        rec["problems"].append(p)
        print(f"{name} {tier} (n {p['n']}, Pe {p['pe_max']:.0f}): floor {p['floor']['energy_rel']:.2e}; "
              + "; ".join(f"{k} {v['energy_rel']:.2e} E4 {'y' if p['e4'][k] else 'n'} "
                          f"price {p['price'][k]:.2f}" for k, v in p["accuracy"].items() if k != "classical")
              + f"; A/A {p['aa']:.3f}", flush=True)
    rec["verdict"] = None if quick else judge(rec["problems"])
    rec["seconds"] = round(time.time() - t0, 1)
    os.makedirs(out_dir, exist_ok=True)
    stem = "phase2-decisive-quick" if quick else "phase2-decisive"
    _write(os.path.join(out_dir, stem + ".json"), json.dumps(rec, indent=1, default=float) + "\n")
    _write(os.path.join(out_dir, stem + ".txt"), report(rec))
    return rec


def _f(x, fmt=".3g"):
    return "-" if x is None else format(x, fmt)


def report(rec: dict) -> str:
    L = [rec["what"], "", f"written  {rec['written']}", f"commit   {rec['commit']}",
         f"BLAS     fixed at 4: {rec['blas'].get('fixed_at_4')}", "",
         "CRITERIA (fixed before the first run)"]
    for k, v in rec["criteria"].items():
        L.append(f"  {k}: {v}")
    for p in rec["problems"]:
        L += ["", f"{p['layout']} {p['tier']}: n {p['n']}, {p['pieces']} pieces ({p['fine_pieces']} finer), "
                  f"Pe max {p['pe_max']:.1f}, u {p['u_coolant']} m/s, {p['nodes']} nodes; "
                  f"the two cuts' references agree to {_f(p['references_agree'], '.1e')}; A/A {p['aa']:.3f} "
                  f"({'counted' if p['aa_ok'] else 'NOT counted'})",
              "  arm              energy     max/range  outlet/range  E4   price"]
        for k, er in p["accuracy"].items():
            L.append(f"  {k:16s} {er['energy_rel']:.3e}  {er['max_rel']:.2e}   {_f(er['outlet_bulk_err_rel'], '.2e'):9s}    "
                     f"{'yes' if p['e4'][k] else 'no ':3s}  {p['price'][k]:.3f}")
        L.append(f"  full domain (one solve, for reference): price {p['price']['full domain']:.3f}")
        s = p["switch"]
        L.append(f"  switch (TL0-{s['n_c']} answers): cost over a classical piece solve: exact "
                 f"{s['cost_over_classical_piece']['exact']:.2f}, Schwarz {s['cost_over_classical_piece']['schwarz']:.2f}, "
                 f"Jacobi {s['cost_over_classical_piece']['jacobi']:.2f}; Schwarz/exact "
                 f"{_f(s['schwarz_over_exact'][0])}-{_f(s['schwarz_over_exact'][1])}" if s["schwarz_over_exact"] else "")
    if rec.get("verdict"):
        v = rec["verdict"]
        L += ["", "VERDICT (the stop rule)"]
        for k, o in v["options"].items():
            L.append(f"  {k:16s} E4 on {o['e4_count']} of {len(rec['problems'])}; price {_f(o['price_geomean'], '.3f')} "
                     f"over {o['priced_problems']} problems{'  STOPS PHASE 2' if o['stops_phase2'] else ''}")
        L.append(f"  stop: {v['stop']} {v['stopped_by']}")
    L += [f"  {rec['seconds']} s", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "chartop", "cot"))
    a = ap.parse_args(argv)
    rec = run(a.quick, a.out)
    print("verdict:", None if rec["verdict"] is None else {k: rec["verdict"][k] for k in ("stop", "stopped_by")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
