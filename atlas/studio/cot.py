"""CO-T on the studio's pieces: a conjugate-heat case cut into pieces and coupled by the
transport host (the coolant plan's Phase 1, [[chart-operator-stage-0b-and-1-plan]]).

The cut is the Pieces screen's own (`pieces.py`, geometry only).  What changes for a
conjugate-heat case (``conjugate-heat-2d``):

* **the pieces** are `transport.TransportPiece`s: each one material, all coolant (its
  material ``flows``) or all solid, with the material's ``heat`` as its source;
* **the flow** is given, not solved: on every channel the coolant pieces tile, the stream
  function ``Q f(v)`` of its harmonic across-coordinate v (`transport_flow`: the plane
  Poiseuille profile in a straight channel, a laminar-like one round a bend), at the
  case's mean speed ``u_coolant``, from the ``coolant-inlet`` face to the
  ``coolant-outlet`` face.  It is one function on the channel's merged nodes, so every
  seam across the channel conserves mass.  A channel with no inlet, two inlets or an
  island is refused, named.  (Phase 1 built straight rectangles only; Phase 2 made it
  general, for the development layouts drawn with the Channel tool and cut along their
  walls by `channel_cut`);
* **the boundaries**: ``coolant-inlet`` is the inlet in Danckwerts' form (Phase 2; Phase 1
  held it, which pinned the plate's corners at the channel's mouth to T_in, a point
  sink): the inflow carries T_in in by its advective flux, and nothing is imposed where
  the velocity vanishes.  ``coolant-outlet`` is the do-nothing outlet (natural: the
  host's insulated side, where the flow leaves), ``insulated`` and ``fixed-temperature``
  as for heat; a nonzero heat flux is refused, as the CO-D path refuses it;
* **the seams** are `transport_host.TransportLayout`'s: a wall carries ``THERM``, a seam
  the flow crosses ``THERM`` and ``ADVEC`` (the advective heat), each named;
* **the expert** is the classical CO-T superelement only (exact trial and adjoint
  modes): nothing learned exists for CO-T, and TL0 for CO-T is Phase 2's;
* **the envelope** is CO-T's: Proposition 3's certificate, the owner's cap of 200 on a
  piece's Peclet number (cut the channel shorter) and Galerkin's cell Peclet number of
  2 (a finer grid).  A piece outside it is refused, named, before anything is solved.

The conjugate-heat family runs steady here (its registry lists one mode), so a CO-T run
is one steady solve; the workbench's own runner keeps its plug flow and its style C,
and this path does not touch them.
"""

from __future__ import annotations

import math
import time
import traceback

import numpy as np

from atlas.chartop import transport as tr
from atlas.chartop import transport_host as th
from atlas.chartop.host import Refused, monolithic

from . import pieces as P

FAMILY = "conjugate-heat-2d"
EXPERT_ROWS = ({"key": "classical", "label": "Classical (CO-T)", "kind": "classical", "offered": True,
                "tip": "The piece's exact trial and adjoint modes, solved (Petrov-Galerkin, class G0'): "
                       "the floor every later CO-T expert is measured against"},)
G0P_NOTE = ("G0': the pieces are coupled by Petrov-Galerkin, testing with each piece's adjoint modes, "
            "so an expert's errors enter the interface quadratically; there is no energy and no "
            "positivity for every output, unlike G0. Each piece is passive under the ADVEC split; the "
            "coupled interface matrix is positive real (its symmetric part's smallest eigenvalue below).")


def is_cot(spec) -> bool:
    return spec.physics.family == FAMILY


def experts() -> list[dict]:
    return [dict(r) for r in EXPERT_ROWS]


def prices() -> dict:
    return {"classical": {"value": 1.0, "record": None, "set": "by definition"}}


# ---------------------------------------------------------------------------
# the flow
# ---------------------------------------------------------------------------


def _materials(spec) -> dict:
    return P._materials(spec)


def coolant(spec) -> set:
    return {k for k, v in _materials(spec).items() if float(v.get("flows", 0.0)) >= 0.5}


def _boundary_polys(spec) -> dict:
    """Every boundary as polylines in metres, by (kind, value) (a stretch of a grid edge
    with no stop runs to the edge's end)."""
    d = spec.domain
    dx = float(d.dx)
    edges = {}
    for name, poly in P.boundary_edges(spec):
        edges.setdefault(name, []).append(poly)
    out: dict = {}
    for b in spec.boundaries:
        for poly in edges.get(b.edge, []):
            Q = poly
            if d.outline is None and (b.start is not None or b.stop is not None):
                a, z = Q[0], Q[-1]
                L = float(np.hypot(*(z - a)))
                s0 = float(b.start or 0) * dx
                s1 = float(b.stop) * dx if b.stop is not None else L
                Q = np.array([a + (z - a) * (s0 / L), a + (z - a) * (s1 / L)])
            out.setdefault((b.kind, b.value), []).append(Q)
    return out


def face_kinds(spec):
    """face(i, s, points) for `transport_flow`: a coolant piece's outer side on a
    coolant-inlet boundary is the channel's inlet, on a coolant-outlet its outlet, and
    anything else (the plate beside it, an insulated edge) a wall."""
    polys = _boundary_polys(spec)
    tol = 0.02 * float(spec.domain.dx)
    inlets = [Q for (k, _v), qs in polys.items() if k == "coolant-inlet" for Q in qs]
    outlets = [Q for (k, _v), qs in polys.items() if k == "coolant-outlet" for Q in qs]

    def on(Pts, qs):
        return bool(qs) and bool(np.all(np.min([P._dist_to_polyline(Pts, Q) for Q in qs], axis=0) <= tol))

    def face(i, s, Pts):
        Pts = np.asarray(Pts, float)
        if on(Pts, inlets):
            return "inlet"
        if on(Pts, outlets):
            return "outlet"
        return "wall"
    return face


def flow(cut: P.Cut, spec) -> tuple[dict, list]:
    """The physical stream function (m^2/s) at every coolant piece's nodes, by piece
    index, from `transport_flow` (the channel's harmonic across-coordinate: plane
    Poiseuille in a straight channel) at the case's mean speed ``u_coolant``; and what
    was built, per channel.  Kept on the cut for the case's flow-relevant fields."""
    from atlas.chartop import transport_flow as tf
    wet = coolant(spec)
    U = float(spec.physics.params.get("u_coolant", 0.0))
    idx = [i for i, p in enumerate(cut.pieces) if p.material in wet]
    key = (U, tuple(sorted(wet)), tuple((b.edge, b.kind, b.start, b.stop) for b in spec.boundaries),
           tuple((p.method, p.material) for p in cut.pieces))
    memo = getattr(cut, "_cot_flow", None)
    if memo is not None and memo[0] == key:
        return memo[1], memo[2]
    if not idx:
        res = ({}, [])
    else:
        charts = [P.chart_of(cut.pieces[i], cut.n) for i in idx]
        try:
            psis, info = tf.stream_functions(charts, face_kinds(spec), U)
        except tf.FlowError as exc:
            raise P.CutError(f"the coolant's flow: {exc}") from None
        for c_ in info:
            c_["pieces"] = [idx[j] for j in c_["pieces"]]
        res = ({i: psis[j] for j, i in enumerate(idx)}, info)
    cut._cot_flow = (key, res[0], res[1])
    return res


# ---------------------------------------------------------------------------
# pieces, boundaries, layout
# ---------------------------------------------------------------------------


def inlets(spec):
    """inlet(points) -> the inlet temperature of a side lying on a coolant-inlet boundary,
    or None."""
    polys = _boundary_polys(spec)
    tol = 0.02 * float(spec.domain.dx)
    groups = [(float(v), qs) for (k, v), qs in polys.items() if k == "coolant-inlet"]

    def at(Pts):
        Pts = np.asarray(Pts, float)
        for T, qs in groups:
            if np.all(np.min([P._dist_to_polyline(Pts, Q) for Q in qs], axis=0) <= tol):
                return T
        return None
    return at


def problems(cut: P.Cut, spec) -> list:
    """The pieces as CO-T problems: chart, material, flow and source; steady.  A coolant
    piece's side on a coolant-inlet boundary is its inlet, in Danckwerts' form (the
    inflow carries T_in in by its advective flux; nothing is held where the velocity
    vanishes, so the plate is not pinned at the channel's mouth)."""
    from atlas.chartop import ports as pt
    if spec.run.mode != "steady":
        raise P.CutError("the conjugate-heat family runs steady here: set the run to steady")
    mats = _materials(spec)
    psis, _info = flow(cut, spec)
    t_in = inlets(spec)
    out = []
    for i, p in enumerate(cut.pieces):
        c = P.chart_of(p, cut.n)
        m = mats.get(p.material)
        if m is None or "k" not in m:
            raise P.CutError(f"{p.label}: its material {p.material!r} has no conductivity")
        k = float(m["k"])
        alpha = k / (float(m["rho"]) * float(m["cp"]))
        area = c.element_fields()["area"]
        q = float(m.get("heat", 0.0))
        psi = None if i not in psis else psis[i] / alpha
        g = np.full((c.n + 1, c.n + 1), q * area / k)
        inlet = {}
        if i in psis:
            loop_pts = c.loop_points()
            for s in range(4):
                T = t_in(loop_pts[pt.side_positions(c.n, s)])
                if T is not None:
                    inlet[s] = T
        out.append(tr.make_piece(c, psi, k=k, alpha=alpha, fo=None, g=g, label=p.label,
                                 meta={"area": area, "material": p.material, "q": q},
                                 inlet=inlet))
    return out


#: a coolant inlet is natural for the host: its piece carries the inflow (Danckwerts)
BC_KINDS = {"coolant-inlet": "insulated", "fixed-temperature": "dirichlet",
            "coolant-outlet": "insulated", "insulated": "insulated"}


def make_bc(spec):
    """bc(i, s, points) for the transport host from the case's boundaries (`pieces.make_bc`
    with the coolant's kinds; a stretch of a grid edge with no stop runs to its end)."""
    tol = 0.02 * float(spec.domain.dx)
    groups: dict = {}
    for (kind_, value), qs in _boundary_polys(spec).items():
        kind = BC_KINDS.get(kind_)
        if kind is None and kind_ == "heat-flux" and not value:
            kind = "insulated"
        if kind == "dirichlet":
            key = ("dirichlet", float(value))
        elif kind == "insulated":
            key = ("insulated", None)
        else:
            key = ("refused", f"a {kind_} boundary")
        groups.setdefault(key, []).extend(qs)

    def bc(i, s, Pts):
        Pts = np.asarray(Pts, float)
        for key, polys in groups.items():
            dd = np.min([P._dist_to_polyline(Pts, Q) for Q in polys], axis=0)
            if np.all(dd <= tol):
                if key[0] == "refused":
                    raise P.CutError(f"piece {i}, side {s} lies on {key[1]}: the CO-T host takes "
                                     "held temperatures, insulated sides, inlets and outlets")
                if key[0] == "dirichlet":
                    return ("dirichlet", np.full(len(Pts), key[1]))
                return ("insulated", None)
        raise P.CutError(f"piece {i}, side {s} lies on no single boundary condition: cut it so "
                         "its outer sides follow the domain's edges, one condition each")
    return bc


def layout(cut: P.Cut, spec, m: int | None = None, split: str = "skew") -> th.TransportLayout:
    return th.TransportLayout(problems(cut, spec), make_bc(spec), kind=P.KIND, m=m or P.M,
                              split=split)


def seams(L: th.TransportLayout) -> list:
    """Every seam and outer side with its verdict and its ports."""
    out = []
    by_edge = {sm.edge: sm for sm in L.seams}
    for k, e in enumerate(L.edges):
        i, s = e.owner
        X = L.problems[i].chart.X
        side = {0: X[0, :], 1: X[:, -1], 2: X[-1, ::-1], 3: X[::-1, 0]}[s]
        pts = side[:: max(1, len(side) // 16)].tolist() + [side[-1].tolist()]
        sm = by_edge.get(k)
        if e.other is None:
            kind = e.bc[0] if e.bc is not None else "insulated"
            out.append({"pieces": [i], "side": s, "verdict": kind, "points": pts,
                        "flow": sm.kind if sm else "boundary"})
        else:
            adv = sm is not None and sm.kind == "advective"
            out.append({"pieces": [i, e.other[0]], "side": s,
                        "verdict": "admit" if e.conforming else "admit-uncertified",
                        "port": "THERM+ADVEC" if adv else "THERM",
                        "why": ("the flow crosses it: temperature, diffusive and advective heat "
                                "(ADVEC) are exchanged; conforming" if adv else
                                "a wall: temperature and heat flux (THERM); "
                                + ("conforming" if e.conforming else "a mortar")),
                        "points": pts})
    return out


def _piece_flow(pr: tr.TransportPiece) -> dict:
    return {"pe": float(pr.peclet()), "cell_pe": float(pr.cell_peclet_max()),
            "flows": bool(pr.flows), "ports": list(pr.ports())}


def summarise(cut: P.Cut, spec, rows: list) -> dict:
    """The Pieces screen's summary for a CO-T layout (`piecejob.summarise`'s shape)."""
    from . import piecejob as PJ
    probs = problems(cut, spec)
    out = {"cutter": cut.cutter, "params": cut.params, "n": cut.n, "notes": list(cut.notes),
           "chords": P.chords(cut), "pieces": [], "seams": None, "refused": None,
           "physics": "CO-T", "class": "G0′",
           "flow": {"u_coolant": float(spec.physics.params.get("u_coolant", 0.0)),
                    "profile": "plane Poiseuille (parabolic)", "pe_cap": tr.PE_MAX,
                    "cell_pe_max": tr.CELL_PE_MAX}}
    for i, (pc, pr) in enumerate(zip(cut.pieces, probs)):
        Xg = pr.chart.X
        n = Xg.shape[0] - 1
        step = max(1, n // 8)
        sub = max(1, n // 32)
        lines = [Xg[j, ::sub] for j in range(0, n + 1, step)] + [Xg[::sub, k] for k in range(0, n + 1, step)]
        ring = np.vstack([Xg[0, :], Xg[:, -1], Xg[-1, ::-1], Xg[::-1, 0]])[:: max(1, n // 24)]
        ce = P.certificate(pr.chart)
        refusals = tr.envelope_refusals(pr)
        fl = _piece_flow(pr)
        out["pieces"].append({
            "index": i, "label": pc.label, "material": pc.material, "own": pc.own,
            "method": pc.method, "chart": pr.chart.method,
            "outline": PJ._round(ring), "lines": [PJ._round(l) for l in lines],
            "sides_xy": [PJ._round(Xg[0, ::sub]), PJ._round(Xg[::sub, -1]), PJ._round(Xg[-1, ::-1][::sub]),
                         PJ._round(Xg[::-1, 0][::sub])],
            "centre": PJ._round(Xg[n // 2, n // 2]),
            "certificate": {"ok": ce["ok"], "min_detJ_rel": ce["min_detJ_rel"], "reason": ce["reason"],
                            "nonpositive": ce["nonpositive_corners"], "simple": ce["simple"]},
            "skew": PJ.max_skew(Xg) if ce["ok"] else None,
            "envelope": {"measure": {"pe": fl["pe"], "cell_pe": fl["cell_pe"]}, "refusals": refusals,
                         "card": "CO-T"},
            "flow": fl, "fo": None, "kappa": float(pr.kappa_ref),
            "expert": pc.expert, "admitted": {r["key"]: ("; ".join(refusals) if refusals else None)
                                              for r in rows if r["offered"]},
            "square": False, "chords": pc.chords,
        })
    try:
        L = th.TransportLayout(probs, make_bc(spec), kind=P.KIND, m=P.M)
        out["seams"] = seams(L)
        sides = {}
        by_edge = {sm.edge: sm for sm in L.seams}
        for k, e in enumerate(L.edges):
            i, s = e.owner
            sm = by_edge.get(k)
            if e.other is None:
                sides[(i, s)] = {"to": None, "kind": e.bc[0] if e.bc else "insulated",
                                 "flow": sm.kind if sm else "boundary"}
            else:
                j, t = e.other
                port = "THERM+ADVEC" if (sm and sm.kind == "advective") else "THERM"
                sides[(i, s)] = {"to": j, "kind": port, "conforming": bool(e.conforming)}
                sides[(j, t)] = {"to": i, "kind": port, "conforming": bool(e.conforming)}
        for p in out["pieces"]:
            p["sides"] = [sides.get((p["index"], s), {"to": None, "kind": "?"}) for s in range(4)]
        out["advective_seams"] = sum(1 for sm in L.seams if sm.kind == "advective")
    except (Refused, P.CutError) as exc:
        out["refused"] = str(exc)
        for p in out["pieces"]:
            p["sides"] = [{"to": None, "kind": "?"} for _ in range(4)]
    ok = [p["certificate"]["ok"] for p in out["pieces"]]
    out["certified"] = [sum(ok), len(ok)]
    dets = [p["certificate"]["min_detJ_rel"] for p in out["pieces"] if p["certificate"]["min_detJ_rel"] is not None]
    out["min_detJ_rel"] = min(dets) if dets else None
    sk = [p["skew"] for p in out["pieces"] if p["skew"] is not None]
    out["max_skew"] = max(sk) if sk else None
    pes = [p["flow"]["pe"] for p in out["pieces"]]
    out["max_pe"] = max(pes) if pes else 0.0
    return out


def check(cut: P.Cut, spec) -> dict:
    """The coupled classical CO-T solve against `host.monolithic`: with the 16 port modes
    and with every side's whole trace (round-off on conforming seams)."""
    out = {}
    for key, m in (("ports16", P.M), ("full", cut.n)):
        L = layout(cut, spec, m=m)
        refusals = L.check_envelope()
        cp = L.solve(tr.ClassicalTransportExpert(), refuse_outside=False)
        mono = monolithic(L)
        er = cp.errors(mono.fields)
        diff = max(float(np.abs(a - b).max()) for a, b in zip(cp.fields, mono.fields))
        out[key] = {"max_diff_rel": diff / er["range"] if er["range"] > 0 else diff,
                    "energy_rel": er["energy_rel"], "outlet_bulk_err_rel": er["outlet_bulk_err_rel"],
                    "outlet_bulk": er["outlet_bulk"], "balance_rel": th.balance(L, cp.fields, mono)["relative"],
                    "monolithic_nodes": mono.nodes, "min_eig": cp.sym_min_eig(),
                    "refusals": refusals[:5]}
    return out


# ---------------------------------------------------------------------------
# Run: Single and Compare on CO-T pieces (corun's loop, steady)
# ---------------------------------------------------------------------------


def _arm(spec_arm: dict) -> tuple[str, str]:
    method = spec_arm.get("method", "full")
    solver = spec_arm.get("solver", "classical")
    if method not in ("full", "chart"):
        raise P.CutError(f"a window's method is the full domain or the chart operator, not {method!r}")
    if method == "chart" and solver not in ("classical", "mixed"):
        raise P.CutError(f"CO-T has the classical expert only (TL0 for CO-T is the coolant plan's "
                         f"Phase 2): {solver} is not built")
    label = "full domain" if method == "full" else "chart operator, classical (CO-T)"
    return method, label


def _errors(L, ref, fields) -> dict:
    num = den = 0.0
    per = []
    for p, u, v in zip(L.problems, ref, fields):
        D = p.diffusion_matrix()
        e = np.asarray(u, float) - np.asarray(v, float)
        a = p.kappa_ref * float(e @ (D @ e))
        b = p.kappa_ref * float(np.asarray(u) @ (D @ np.asarray(u)))
        num += a
        den += b
        per.append(math.sqrt(max(a, 0.0) / b) if b > 0 else 0.0)
    mx = max(float(np.abs(np.asarray(a) - np.asarray(b)).max()) for a, b in zip(ref, fields))
    span = float(np.ptp(np.concatenate([np.ravel(f) for f in ref]))) or 1.0
    tb, trf = th._outflow_bulk(L, fields), th._outflow_bulk(L, ref)
    return {"energy_rel": math.sqrt(max(num, 0.0) / den) if den > 0 else 0.0, "max_err": mx,
            "max_over_range": mx / span, "per_piece": per,
            "worst_piece": int(np.argmax(per)) if per else None,
            "outlet_bulk": tb, "outlet_bulk_ref": trf,
            "outlet_bulk_err_rel": (abs(tb - trf) / span if (tb is not None and trf is not None) else None)}


def run(job) -> None:
    """`corun.CORun._run` for a CO-T case: steady, A and B in turns, the same result's shape."""
    from atlas.workbench import machine
    from . import corun as CR
    s = job.session
    job.status = "running"
    try:
        spec, cut = s.spec, s.cut
        if cut is None:
            raise P.CutError("cut the domain into pieces first (Pieces)")
        arms = {k: _arm(v) for k, v in job.specs.items()}
        for k, v in job.specs.items():
            v["label"] = arms[k][1]
        L = layout(cut, spec)
        refused = L.check_envelope()
        if refused and any(a[0] == "chart" for a in arms.values()):
            raise P.CutError("the CO-T envelope refuses: " + "; ".join(refused[:3])
                             + (" …" if len(refused) > 3 else ""))
        job.steps = 1
        times = {k: [] for k in arms}
        out = {}
        for turn in range(CR.TURNS_STEADY):
            for k in (list(arms) if turn % 2 == 0 else list(arms)[::-1]):
                t0 = time.perf_counter()
                if arms[k][0] == "full":
                    fields, cp = monolithic(L).fields, None
                else:
                    cp = L.solve(tr.ClassicalTransportExpert())
                    fields = cp.fields
                times[k].append(time.perf_counter() - t0)
                out[k] = (fields, cp)
        if job.mode == "single":
            ref = monolithic(L).fields
            test = "A"
        else:
            ref = out["A"][0]
            test = "B"
        tf, tcp = out[test]
        acc = {"step": 1, **_errors(L, ref, tf)}
        if tcp is not None:
            acc["min_eig"] = tcp.sym_min_eig()
            acc["calls_per_piece"] = (tcp.calls or 0) / len(L.problems)
        fr = {"A": CR.raster(L.problems, out["A"][0])}
        if job.mode == "compare":
            fr["B"] = CR.raster(L.problems, out["B"][0])
            fr["difference"] = CR.raster(L.problems, [b - a for a, b in zip(out["A"][0], out["B"][0])])
        else:
            fr["full domain"] = CR.raster(L.problems, ref)
            fr["difference"] = CR.raster(L.problems, [a - r for r, a in zip(ref, out["A"][0])])
        with job.lock:
            job.frames.append(fr)
            job.step = 1
            job.version += 1
        jumps = CR.seam_jumps(L, tf)
        mono = monolithic(L)
        res = {"steps": 1, "steady": True, "physics": "CO-T", "accuracy": acc,
               "series": [{k: acc[k] for k in ("step", "energy_rel", "max_err")}],
               "pieces": [{"label": p.label, "expert": ("classical" if arms[test][0] == "chart" else "full domain")}
                          for p in cut.pieces],
               "seams": {"max_jump": max(jumps) if jumps else 0.0, "count": len(jumps),
                         "advective": sum(1 for sm in L.seams if sm.kind == "advective")},
               "balance": {k: th.balance(L, out[k][0], mono)["relative"] for k in arms},
               "outlet": {"bulk": acc["outlet_bulk"], "reference": acc["outlet_bulk_ref"],
                          "err_over_range": acc["outlet_bulk_err_rel"]},
               "e4": {"passed": None, "title": "E4 for CO-T: not registered yet",
                      "tip": ("CO-T's E4 measure and bars are fixed at the coolant plan's registration "
                              "(Phase 3); this run reports the energy error, the outlet's mixing-cup "
                              "temperature and the balance beside it, with no bar")}}
        if job.mode == "compare":
            ta, tb = np.median(times["A"]), np.median(times["B"])
            res["price"] = {"ratio": float(tb / ta) if ta > 0 else None,
                            "meaning": "B's time over A's, each the median of its turns on this machine: below 1, B is faster",
                            "aa_spread": float(np.percentile(times["A"], 90) / np.percentile(times["A"], 10)) if len(times["A"]) > 1 else None,
                            "turns": len(times["A"])}
        res["certificate"] = {"class": "G0′", "charts": [sum(1 for p in L.problems if P.certificate(p.chart)["ok"]), len(L.problems)],
                              "min_eig": acc.get("min_eig"), "note": G0P_NOTE}
        res["machine"] = machine.machine_record()
        res["threads"] = CR._threads()
        with job.lock:
            job.result = res
            job.version += 1
        job.status = "stopped" if job._stop.is_set() else "done"
    except (P.CutError, Refused) as exc:
        job.error = str(exc)
        job.status = "failed"
    except Exception as exc:
        job.error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        job.status = "failed"


__all__ = ["EXPERT_ROWS", "FAMILY", "check", "coolant", "experts", "face_kinds", "flow",
           "is_cot", "layout", "make_bc", "prices", "problems", "run", "seams", "summarise"]
