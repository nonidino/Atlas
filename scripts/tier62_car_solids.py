"""Tier 62 -- the drawn car as solids on body-fitted grids, and the first march.

    python scripts/tier62_car_solids.py --out out/racelab11 --stages controls,poisson,flow
    python scripts/tier62_car_solids.py --out out/racelab11 --stages car,march
    python scripts/tier62_car_solids.py --out out/racelab11 --stages compare,summary

Tier 61 verified a flow on curved overlapping grids round one body.  The car
needs several bodies whose grids cut each other, a road that cuts them, the
gap between a tyre and the road, and the car's plates turned into solids.
This file checks the first three on shapes with known answers and then puts
the car on them:

  ``controls``  `MultiOverset` against Tier 60's `Overset` on Tier 60's own
                geometry; linear fields through every interpolation point of
                the multi-body composite; every donor carrying its own
                equation; the refusals; the uniform stream on the car.
  ``poisson``   a manufactured pressure on a wheel two percent of its radius
                above a road patch, a thick panel within two background cells
                of the tyre, and an aerofoil above it -- three resolutions.
  ``flow``      Tier 61's manufactured flow on the same composite, with every
                wall moving, at a step small enough to leave the space error.
  ``car``       the solids by the user's rule (`car_solids`), the grids round
                them, the composite, and a manufactured pressure on it.
  ``march``     the car from the uniform stream, its walls brought to rest:
                forces on every part, the duct's flow, the flux through a
                contour round the car, the divergence and where it sits.
  ``compare``   the duct's flow against the immersed column's settled release
                state (Tier 59) -- two models of one drawing, not a referent.
  ``summary``   every registered prediction judged in code.

Nothing here touches RaceLab's column, its records, its gate or the demo.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse                                                         # noqa: E402
import datetime as dt                                                   # noqa: E402
import json                                                             # noqa: E402
import math                                                             # noqa: E402
import sys                                                              # noqa: E402
import time                                                             # noqa: E402
import traceback                                                        # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(HERE, "scripts")
for _p in (HERE, SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                      # noqa: E402

from atlas.cases import car_solids as CS                                # noqa: E402
from atlas.cases import overset as OV                                   # noqa: E402
from atlas.cases import overset_multi as OM                             # noqa: E402
from atlas.cases import overset_ns as NS                                # noqa: E402
import tier60_body_fitted_grids as T60                                  # noqa: E402

OUT = NAME = None

POISSON_LEVELS = (3, 4, 5)
FLOW_LEVELS = (3, 4)
FLOW_DT, FLOW_T = 0.00025, 0.05
#: the march: to this time, the statistics over this window, persisted this often
MARCH_T = 16.0
MARCH_WINDOW = (12.0, 16.0)
MARCH_PERSIST = 40
MARCH_PROBE_EVERY = 8
#: the device planes and the duct's band, in cells (`racelab.DEVICE_...`, Tier 59's record)
CORE_X, TURBINE_X = 280.0, 392.0
#: a contour round the car, in cells: inlet side, outlet side and top; the road closes it
CONTOUR = (40, 600, 150)

#: **Read before this file's predictions were written**, and carried into the record.
READ_BEFORE_THIS_RUN = (
    "The multi-body manufactured pressure at L3 and L4 with body grids 0.15 thick and a NACA "
    "trailing edge closed to a point: max-error orders wheel 2.17, panel 2.26, road patch 2.08, "
    "background 1.14 and aerofoil 1.12, both maxima in the fan behind the trailing edge. With the "
    "trailing edge rounded to 1% of the chord: 1.67 and 1.72; with the columns clustered where the "
    "outline turns (4): 1.76 and 1.82; with the normals also smoothed at the wall (0.02, kappa 0.5): "
    "1.94 and 2.01. Those settings became the car's, so these orders were chosen on this geometry.",
    "At L2 the multi-body composite had orphans (39 with body grids 0.10 thick, 1 at 0.15); L3 is "
    "the coarsest level used. With the car's final settings (bodies 0.10 thick, the smaller road "
    "patch) the composite was built, and nothing measured, at L3, L4 and L5: 46,613, 183,652 and "
    "728,356 unknowns. Two points between a tyre's polygon and its interpolant's curve were orphans "
    "until a stencil starting at a wall row was allowed a quarter row past it (WALL_REACH).",
    "The manufactured flow on the multi-body composite at L3 only, 20 steps at dt = 0.001, earlier "
    "settings: velocity errors 1.2e-5 to 8.8e-4, divergence at most 9.3e-3 (the aerofoil).",
    "The car's welded shells: an offset grid 0.10 thick folds round the front wing and the body "
    "shell with no fillet; the ladder picks the smallest fillet that neither folds, nor skews past "
    "70 degrees, nor leaves an orphan against the background alone. Trimming the thickened panels "
    "at a wheel's clearance left square corners; trimming the centreline instead leaves caps.",
    "The car's first steps. Impulsive start: divergence 2.3e3 and speed 23 at step 1, the second "
    "momentum solve did not converge. From rest with the stream, road and wheels ramped together: "
    "divergence growing at a road patch's end, 0.5 to 2.5 in six steps. Walls ramped from the "
    "stream to rest, wheel clearance 2 cells: divergence 28 by step 11 in the slits; clearance 4 "
    "cells and a smaller road patch: the march below.",
    "On the car's composite the uniform stream with every wall moving with it stayed uniform to "
    "2e-14 over three steps, and a manufactured pressure's max error was at most 6.3e-3 per grid.",
    "The car marched to t = 2 (ILU; the first factor at drop 1e-4 was singular, 1e-6 worked): "
    "largest divergence 7.3 during the ramp, then 4.6, 2.4, 1.1, 0.61, 0.36, 0.24, 0.17 at t = 0.5 "
    "to 2.0; largest speed 3.8, then 2.5; drag 35 during the ramp falling to 4.4; vertical force "
    "-1.10 at t = 2 and falling; the duct's mean streamwise velocity 0.154 at both device planes "
    "over the whole run; the contour's net flux at most 0.5% of its inflow; BiCGSTAB at most 33 "
    "iterations, 3 to 4 at the end; a median step of 1.68 s on battery. Jacobi had taken 100 to 300.",
    "The immersed column's settled release state (Tier 59, out/racelab8): the rotor's inflow 0.457, "
    "its minimum over the horizon 0.446, the largest speed 1.83.",
)

#: **The prediction, registered before any stage of this file ran**, judged in
#: code by `judge` under the same ids.
PREDICTION = (
    {"id": "V1", "claim": "on Tier 60's own geometry at L2 and L3, MultiOverset's point statuses are "
                          "Tier 60's exactly, its interpolation points and stencils are the same, "
                          "and its weights differ by at most 1e-15",
     "why": "with one body and no road the candidate order and the first stencil tried are Tier 60's"},
    {"id": "V2", "claim": "on the multi-body composite at L3 and L4 every interpolation equation "
                          "reproduces a linear field to 1e-10, and every donor node carries its own "
                          "equation",
     "why": "Newton on the same Lagrange map the weights come from, for every shifted stencil too"},
    {"id": "V3", "claim": "four configurations are refused: two bodies that overlap, a wheel on the "
                          "road, a gap under two grid rows, and a grid leaving through the inlet",
     "why": "each is a refusal written into MultiOverset's classification"},
    {"id": "V4", "claim": "the generator with clustering and room-limited thickness is one mapping: "
                          "at twice the columns and rows, sampled at every other, it is the coarser "
                          "grid to 1e-12",
     "why": "the density and the room are properties of the outline's master resampling"},
    {"id": "M1", "claim": "the multi-body manufactured pressure: max-error order >= 1.7 on every grid "
                          "from L3 to L4 and >= 1.8 from L4 to L5",
     "why": "the exploration read 1.94 to 2.07 from L3 to L4 at these settings with thicker body "
            "grids; L5 has not been run"},
    {"id": "M2", "claim": "the multi-body manufactured flow at dt = 0.00025 to t = 0.05: velocity "
                          "max-error order L3->L4 >= 1.6 on every grid",
     "why": "Tier 61's S4 read 2.52 and 1.81 on one body with the time error removed; the tyre's gap "
            "and the trailing edge are harder"},
    {"id": "M3", "claim": "the same runs: pressure rms-error order L3->L4 >= 1.0 on every grid",
     "why": "Tier 61 measured 1.22 and 1.39 for one body (W273), undiagnosed"},
    {"id": "M4", "claim": "on the car's composite, a manufactured pressure's max error is <= 1e-2 on "
                          "every grid and the uniform stream with every wall moving with it stays "
                          "uniform to 1e-12 over three steps",
     "why": "6.3e-3 and 2e-14 were read on the same construction before the rule's last changes"},
    {"id": "K1", "claim": "the car marches to t = 16 without a blow-up: every momentum solve "
                          "converges and the largest speed never exceeds 4",
     "why": "to t = 2 the speed peaked at 3.8 during the ramp and fell to 2.5"},
    {"id": "K2", "claim": "over t in [12, 16] the largest divergence anywhere stays <= 0.2",
     "why": "it fell to 0.17 by t = 2 and was still falling"},
    {"id": "K3", "claim": "over t in [12, 16] the net flux out of a contour round the car is <= 1% "
                          "of the inflow through it",
     "why": "0.5% at most over the ramp and the first two units"},
    {"id": "K4", "claim": "the duct carries flow: its mean streamwise velocity at the turbine plane "
                          "over t in [12, 16] lies in [0.08, 0.40] -- positive, and below the "
                          "porous column's settled 0.457",
     "why": "0.154 over the first two units; with solid walls the pod is fed only through the "
            "clearances at the wheels"},
    {"id": "K5", "claim": "mass is conserved along the duct: over t in [12, 16] the flux through the "
                          "core plane and through the turbine plane differ by <= 2% of the core's",
     "why": "they agreed to five figures over the first two units"},
    {"id": "K6", "claim": "the car makes downforce: the mean vertical force on all its solids over "
                          "t in [12, 16] is negative",
     "why": "-1.10 at t = 2, and falling"},
    {"id": "K7", "claim": "the forces settle: over t in [12, 16] the drag's peak-to-peak spread is "
                          "<= 10% of its mean",
     "why": "at Re 250 per unit the wheels shed, but most of the car's drag is its pressure on "
            "slow-moving solid faces"},
    {"id": "K8", "claim": "the momentum solves stay cheap after the ramp: at most 60 BiCGSTAB "
                          "iterations a component on any step with t > 1",
     "why": "33 at most during the ramp, 3 to 4 at t = 2; an iteration count does not depend on "
            "the power state"},
)


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


def configure(out: str) -> None:
    global OUT, NAME
    OUT = os.path.abspath(out)
    NAME = os.path.basename(os.path.normpath(OUT))


def path(name: str) -> str:
    return os.path.join(OUT, name)


def persist(obj, fname: str) -> str:
    os.makedirs(OUT, exist_ok=True)
    p = path(fname)
    tmp = p + ".tmp"

    def write():
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(T60.clean(obj), fh, indent=1)

    T60._retry(write)
    T60._retry(lambda: os.replace(tmp, p))
    return p


def load(fname: str) -> dict:
    p = path(fname)
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def say(*a) -> None:
    print(*a, flush=True)


def _orders(v):
    return [None if (v[k] is None or v[k + 1] is None or v[k + 1] <= 0)
            else math.log2(v[k] / v[k + 1]) for k in range(len(v) - 1)]


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------


def _donor_table(ov):
    table = {}
    for name, spec in ov.donors.items():
        for e in spec["entries"]:
            for k in range(len(e["j"])):
                table[(name, int(e["j"][k]), int(e["i"][k]))] = (
                    e["donor"], tuple(int(v) for v in e["flat"][k]), np.asarray(e["weights"][k]))
    return table


def _linear_exactness(ov) -> float:
    f = lambda x, y: 0.7 * x - 1.3 * y + 2.0                      # noqa: E731
    worst = 0.0
    for name, spec in ov.donors.items():
        g = next(q for q in ov.grids if q.name == name)
        gx, gy = OM.MultiOverset._xy(g)
        for e in spec["entries"]:
            d = next(q for q in ov.grids if q.name == e["donor"])
            dx, dy = OM.MultiOverset._xy(d)
            vals = f(dx.ravel()[e["flat"]], dy.ravel()[e["flat"]])
            got = np.sum(vals * e["weights"], axis=1)
            want = f(gx[e["j"], e["i"]], gy[e["j"], e["i"]])
            worst = max(worst, float(np.abs(got - want).max()))
    return worst


def _donors_usable(ov) -> bool:
    for name, spec in ov.donors.items():
        for e in spec["entries"]:
            S = ov.status[e["donor"]].ravel()[e["flat"]]
            ok = (S == OV.DISC) if e["donor"] == ov.bg.name else np.isin(S, (OV.DISC, OV.WALL))
            if not np.all(ok):
                return False
    return True


def _refused(fn) -> str | None:
    try:
        fn()
    except OV.OversetError as exc:
        return str(exc)[:200]
    return None


def stage_controls(res) -> dict:
    out: dict = {}
    for level in (2, 3):
        n16 = 2 ** (level - 1)
        h = 1.0 / (16 * n16)
        bg = OV.CartesianGrid("bg", int(round(4 / h)), int(round(3 / h)), h)
        body = OV.ogrid_annulus("body", 2.0, 1.5, 0.3, 0.75, 48 * n16, 6 * n16 + 1, beta=1.0)
        a = OV.Overset(bg, [body], hole_margin=0.15, width=3)
        b = OM.MultiOverset(bg, [body], hole_margin=0.15, width=3)
        ta, tb = _donor_table(a), _donor_table(b)
        same_points = set(ta) == set(tb)
        out[f"tier60_L{level}"] = {
            "statuses_identical": all(np.array_equal(a.status[g], b.status[g]) for g in ("bg", "body")),
            "same_interpolation_points": same_points,
            "same_stencils": bool(same_points and all(ta[k][:2] == tb[k][:2] for k in ta)),
            "max_weight_difference": (max(float(np.abs(ta[k][2] - tb[k][2]).max()) for k in ta)
                                      if same_points else None),
        }
        say("controls tier60 L%d" % level, out[f"tier60_L{level}"])
    for level in (3, 4):
        ov = OM.multi_geometry(level)
        out[f"multi_L{level}"] = {"n_unknowns": ov.n_unknowns, "build_s": ov.build_s,
                                  "linear_exactness": _linear_exactness(ov),
                                  "donors_carry_their_own_equation": _donors_usable(ov),
                                  "report": ov.report()}
        say("controls multi L%d" % level, {k: v for k, v in out[f"multi_L{level}"].items() if k != "report"})
    # refusals, at level 3
    h = 1.0 / 64

    def box():
        return OV.CartesianGrid("bg", 192, 97, h, y0=-0.5 * h)

    def wheel(name, cx, cy, r=0.3):
        return OV.ogrid_annulus(name, cx, cy, r, r + 0.15, 384, 33, beta=3.0)

    def patch(cx):
        return OM.road_patch("road", cx - 0.24, cx + 0.24, 0.075, 177, 33, beta=3.0)

    out["refusals"] = {
        "two_bodies_overlap": _refused(lambda: OM.MultiOverset(
            box(), [wheel("a", 1.0, 0.5), wheel("b", 1.5, 0.5)], hole_margin=0.04, road=0.0)),
        "a_wheel_on_the_road": _refused(lambda: OM.MultiOverset(
            box(), [wheel("w", 1.5, 0.3), patch(1.5)], hole_margin=0.04, road=0.0)),
        "a_gap_under_two_rows": _refused(lambda: OM.MultiOverset(
            box(), [wheel("w", 1.5, 0.3 + 0.0012), patch(1.5)], hole_margin=0.04, road=0.0)),
        "a_grid_through_the_inlet": _refused(lambda: OM.MultiOverset(
            box(), [wheel("w", 0.2, 0.6)], hole_margin=0.04, road=0.0)),
    }
    say("controls refusals", {k: (v is not None) for k, v in out["refusals"].items()})
    # the generator's two new options, sampled twice as finely, are the same mapping
    outline = OM.aerofoil_outline(1.5, 0.3, 0.5, 0.12, 12.0, 0.01)
    kw = dict(beta=2.0, cluster=4.0, sigma_wall=0.02, kappa=0.5, room=0.4, room_floor=0.03)
    g1 = OV.ogrid_from_outline("a", outline, 128, 13, 0.1, **kw)
    g2 = OV.ogrid_from_outline("a", outline, 256, 25, 0.1, **kw)
    out["generator_resolution_independence"] = float(max(np.abs(g2.x[::2, ::2] - g1.x).max(),
                                                          np.abs(g2.y[::2, ::2] - g1.y).max()))
    say("controls generator", out["generator_resolution_independence"])
    res["controls"] = out
    persist(res, NAME + ".json")
    return out


# ---------------------------------------------------------------------------
# the manufactured solutions
# ---------------------------------------------------------------------------


def stage_poisson(res) -> dict:
    out = {"levels": []}
    for level in POISSON_LEVELS:
        t0 = time.perf_counter()
        r = OM.manufactured_poisson_multi(level)
        r["wall_total_s"] = time.perf_counter() - t0
        out["levels"].append(r)
        say("poisson L%d n=%d %.1fs %s" % (level, r["n_unknowns"], r["wall_total_s"],
                                         {g: "%.2e" % e["max_err"] for g, e in r["grids"].items()}))
        res["poisson"] = out
        persist(res, NAME + ".json")
    out["orders"] = {f"{g}.{k}": _orders([lv["grids"][g][k] for lv in out["levels"]])
                     for g in out["levels"][0]["grids"] for k in ("max_err", "rms_err")}
    res["poisson"] = out
    persist(res, NAME + ".json")
    return out


def stage_flow(res) -> dict:
    out = {"dt": FLOW_DT, "T": FLOW_T, "levels": []}
    for level in FLOW_LEVELS:
        t0 = time.perf_counter()
        r = OM.mms_flow_multi(level, FLOW_DT, FLOW_T)
        r["wall_total_s"] = time.perf_counter() - t0
        out["levels"].append(r)
        say("flow L%d n=%d %.1fs %s div %s" % (level, r["n_unknowns"], r["wall_total_s"],
                                             {g: "%.2e/%.2e" % (e["u_max_err"], e["p_rms_err"])
                                              for g, e in r["grids"].items()},
                                             {g: "%.1e" % v for g, v in r["divergence"].items()}))
        res["flow"] = out
        persist(res, NAME + ".json")
    out["orders"] = {f"{g}.{k}": _orders([lv["grids"][g][k] for lv in out["levels"]])
                     for g in out["levels"][0]["grids"]
                     for k in ("u_max_err", "u_rms_err", "p_max_err", "p_rms_err")}
    res["flow"] = out
    persist(res, NAME + ".json")
    return out


# ---------------------------------------------------------------------------
# the car
# ---------------------------------------------------------------------------


_CAR: dict = {}


def _car():
    if "ov" not in _CAR:
        t0 = time.perf_counter()
        solids, rec = CS.car_solids()
        t_solids = time.perf_counter() - t0
        t0 = time.perf_counter()
        ov, grec = CS.car_overset(solids)
        _CAR.update(solids=solids, rec=rec, ov=ov, grec=grec, solids_s=t_solids,
                    overset_s=time.perf_counter() - t0)
    return _CAR


def stage_car(res) -> dict:
    from atlas.cases import racelab as RL
    C = _car()
    ov = C["ov"]
    out = {"rule": C["rec"]["rule"], "trimmed": C["rec"]["trimmed"],
           "filled_pockets": C["rec"]["filled_pockets"], "dropped_slivers": C["rec"]["dropped_slivers"],
           "clearance_cut_after_weld_cells": C["rec"]["clearance_cut_after_weld_cells"],
           "fillets": C["rec"]["fillets"], "solids": C["rec"]["solids"], "grids": C["grec"],
           "grid_settings": {k: v for k, v in CS.GRID.items()},
           "n_unknowns": ov.n_unknowns, "report": ov.report(), "solids_s": C["solids_s"],
           "overset_s": C["overset_s"], "geometry_fingerprint": RL.geometry_fingerprint()}
    # a manufactured pressure on the car's own composite
    bx, by = 5.0, 0.6
    for c in ov.comps:
        if not c.periodic:
            c.wall = "dirichlet"

    def wall_value(g):
        p, gx, gy, _l = OV._ms(g.x[0], g.y[0], bx, by)
        if g.wall == "dirichlet":
            return p
        nx_, ny_ = g.wall_normal()
        return nx_ * gx + ny_ * gy

    t0 = time.perf_counter()
    A, b = ov.poisson(lambda x, y: OV._ms(x, y, bx, by)[3], lambda x, y: OV._ms(x, y, bx, by)[0], wall_value)
    x, info = OV.solve_sparse(A, b, "splu")
    info.pop("_lu", None)
    for c in ov.comps:
        if not c.periodic:
            c.wall = "neumann"
    fields = ov.scatter(x)
    errs = {}
    for g in ov.grids:
        S = ov.status[g.name]
        X, Y = OM.MultiOverset._xy(g)
        live = S != OV.HOLE
        e = np.abs(fields[g.name] - OV._ms(X, Y, bx, by)[0])[live]
        errs[g.name] = {"max_err": float(e.max()), "rms_err": float(np.sqrt(np.mean(e ** 2)))}
    out["poisson"] = {"solve": info, "wall_s": time.perf_counter() - t0, "grids": errs}
    # the uniform stream, with every wall moving with it, is an exact discrete solution
    flow = NS.OversetFlow(ov, 0.004, 0.0125,
                          box={"xlo": "dirichlet", "ylo": "dirichlet", "yhi": "dirichlet", "xhi": "outflow"},
                          wall_velocity=lambda g, t: (np.ones(g.ni), np.zeros(g.ni)), precond="ilu")
    n = ov.n_unknowns
    flow.set_state(np.ones(n), np.zeros(n), u_prev=np.ones(n), v_prev=np.zeros(n))
    for _ in range(3):
        flow.step()
    out["uniform_stream_error"] = float(max(np.abs(flow.U - 1).max(), np.abs(flow.V).max()))
    say("car n=%d solids %.1fs overset %.1fs poisson %s uniform %.1e" % (
        ov.n_unknowns, C["solids_s"], C["overset_s"], {g: "%.1e" % e["max_err"] for g, e in errs.items()},
        out["uniform_stream_error"]))
    res["car"] = out
    persist(res, NAME + ".json")
    return out


def _duct_flux(flow, x_cells, n=64):
    from atlas.cases import racelab as RL
    rule = CS.solids_rule()
    half = 0.5 * float(rule["panel_thickness_cells"])
    y0 = RL.DUCT_Y0 + half
    y1 = RL.DUCT_Y0 + RL.DEVICE_CELLS - half
    h = CS.GRID["h"]
    ys = np.linspace(y0, y1, n) * h
    xs = np.full(n, x_cells * h)
    u = CS.probe(flow.ov, flow.U, xs, ys)
    v = CS.probe(flow.ov, flow.V, xs, ys)
    w = np.full(n, (ys[-1] - ys[0]) / (n - 1))
    w[0] = w[-1] = 0.5 * w[1]
    return {"flux": float(np.sum(u * w)), "u_mean": float(np.sum(u * w) / (ys[-1] - ys[0])),
            "u_min": float(np.nanmin(u)), "u_max": float(np.nanmax(u)), "v_max_abs": float(np.nanmax(np.abs(v))),
            "unprobed": int(np.sum(~np.isfinite(u)))}


def _contour_flux(flow):
    """Net volume flux out through a contour round the car: the background's
    cells, the inlet side, the top and the outlet side, the road closing it
    below (no flow through the road)."""
    bg = flow.ov.bg
    idx = flow.ov.index[bg.name]
    i0, i1, j1 = CONTOUR
    h = bg.h
    js = np.arange(0, j1 + 1)
    is_ = np.arange(i0, i1 + 1)

    def face(field, a, b):
        if np.any(a < 0) or np.any(b < 0):
            raise ValueError("the contour touches a hole")
        return 0.5 * (field[a] + field[b])

    left = face(flow.U, idx[js, i0], idx[js, i0 - 1])
    right = face(flow.U, idx[js, i1], idx[js, i1 + 1])
    top = face(flow.V, idx[j1, is_], idx[j1 + 1, is_])
    # the road row sits AT the road, so the side faces' bottom cell counts half
    wy = np.ones(js.size)
    wy[0] = 0.5
    net = h * (np.sum(right * wy) - np.sum(left * wy) + np.sum(top))
    inflow = h * np.sum(left * wy)
    return {"net": float(net), "inflow": float(inflow), "relative": float(net / inflow)}


def stage_march(res) -> dict:
    C = _car()
    ov, solids = C["ov"], C["solids"]
    fname = "march.json"
    flow = CS.car_flow(ov, solids, precond="ilu")
    members = CS.wall_members(ov, solids)
    owner = np.empty(ov.n_unknowns, dtype=object)
    for g in ov.grids:
        ix = ov.index[g.name]
        owner[ix[ix >= 0]] = g.name
    out = {"t_end": MARCH_T, "dt": flow.dt, "ramp": CS.RAMP, "n_unknowns": ov.n_unknowns,
           "setup_s": flow.setup_s, "pressure_factor_s": flow.p_factor_s,
           "machine_before": T60.machine_state(), "started_at": dt.datetime.now().isoformat()}
    steps = int(round(MARCH_T / flow.dt))
    tr = {k: [] for k in ("t", "fx", "fy", "max_div", "div_at", "u_max", "u_max_at", "iterations",
                          "step_s", "ilu_s", "ilu_note")}
    tr["by_solid"] = {s.name: {"fx": [], "fy": []} for s in solids}
    probes = {"t": [], "core": [], "turbine": [], "contour": [], "by_member": []}
    t_start = time.perf_counter()
    for s in range(steps):
        try:
            rec = flow.step()
        except Exception:
            out.update(trajectory=tr, probes=probes, steps_done=s, failed=traceback.format_exc()[-3000:])
            persist(out, fname)
            res["march"] = {"failed": out["failed"], "steps_done": s, "steps": steps, "file": fname}
            persist(res, NAME + ".json")
            raise
        div = np.abs(flow.Dx @ flow.U + flow.Dy @ flow.V)
        div[~flow.disc] = 0.0
        q = int(np.argmax(div))
        sp = np.hypot(flow.U, flow.V)
        r = int(np.argmax(sp))
        fx = fy = 0.0
        for sol in solids:
            f = flow.forces(sol.name)
            tr["by_solid"][sol.name]["fx"].append(f["fx"])
            tr["by_solid"][sol.name]["fy"].append(f["fy"])
            fx += f["fx"]
            fy += f["fy"]
        tr["t"].append(rec["t"])
        tr["fx"].append(fx)
        tr["fy"].append(fy)
        tr["max_div"].append(float(div[q]))
        tr["div_at"].append([owner[q], float(flow.X[q]), float(flow.Y[q])])
        tr["u_max"].append(float(sp[r]))
        tr["u_max_at"].append([owner[r], float(flow.X[r]), float(flow.Y[r])])
        tr["iterations"].append(rec["iterations"])
        tr["step_s"].append(rec["step_s"])
        tr["ilu_s"].append(rec["ilu_s"])
        tr["ilu_note"].append(rec["ilu_note"])
        if s % MARCH_PROBE_EVERY == 0 or s + 1 == steps:
            probes["t"].append(rec["t"])
            probes["core"].append(_duct_flux(flow, CORE_X))
            probes["turbine"].append(_duct_flux(flow, TURBINE_X))
            probes["contour"].append(_contour_flux(flow))
            probes["by_member"].append({sol.name: CS.forces_by_member(flow, sol, members.get(sol.name))
                                        for sol in solids})
        if (s + 1) % MARCH_PERSIST == 0 or s + 1 == steps:
            out.update(trajectory=tr, probes=probes, steps_done=s + 1, wall_s=time.perf_counter() - t_start)
            persist(out, fname)
            say("march step %d/%d t %.3f Fx %.4f Fy %.4f div %.2e at %s | u_max %.3f at %s | its %s | %.2fs" % (
                s + 1, steps, rec["t"], fx, fy, div[q], owner[q], sp[r], owner[r], rec["iterations"],
                rec["step_s"]))
    out["machine_after"] = T60.machine_state()
    out["finished_at"] = dt.datetime.now().isoformat()
    persist(out, fname)
    # what the record keeps: the trajectory stays in march.json, summarised here
    t = np.asarray(tr["t"])
    a, b = MARCH_WINDOW
    win = (t >= a - 1e-9) & (t <= b + 1e-9)
    fx_w, fy_w = np.asarray(tr["fx"])[win], np.asarray(tr["fy"])[win]
    summary = {
        "steps": steps, "steps_done": steps, "window": list(MARCH_WINDOW), "wall_s": out["wall_s"],
        "median_step_s": float(np.median(tr["step_s"])),
        "iterations_max": int(max(max(i) for i in tr["iterations"])),
        "iterations_max_after_t1": int(max(max(i) for i, tt in zip(tr["iterations"], t) if tt > 1.0)),
        "u_max_max": float(max(tr["u_max"])), "u_max_last": float(tr["u_max"][-1]),
        "max_div_window": float(np.max(np.asarray(tr["max_div"])[win])),
        "max_div_window_at": tr["div_at"][int(np.flatnonzero(win)[np.argmax(np.asarray(tr["max_div"])[win])])],
        "fx_mean_window": float(np.mean(fx_w)), "fy_mean_window": float(np.mean(fy_w)),
        "fx_spread_window": float(np.ptp(fx_w)), "fy_spread_window": float(np.ptp(fy_w)),
        "by_solid_window": {name: {"fx": float(np.mean(np.asarray(d["fx"])[win])),
                                   "fy": float(np.mean(np.asarray(d["fy"])[win]))}
                            for name, d in tr["by_solid"].items()},
    }
    pt = np.asarray(probes["t"])
    pw = (pt >= a - 1e-9) & (pt <= b + 1e-9)
    tur = [p for p, k in zip(probes["turbine"], pw) if k]
    cor = [p for p, k in zip(probes["core"], pw) if k]
    summary["turbine_u_mean_window"] = float(np.mean([p["u_mean"] for p in tur]))
    summary["core_u_mean_window"] = float(np.mean([p["u_mean"] for p in cor]))
    summary["duct_flux_mismatch_window"] = float(max(abs(p["flux"] - q["flux"]) / abs(q["flux"])
                                                     for p, q in zip(tur, cor)))
    summary["duct_unprobed_points"] = int(max(p["unprobed"] for p in tur + cor))
    summary["contour_relative_window"] = float(np.max(np.abs(
        [p["relative"] for p, k in zip(probes["contour"], pw) if k])))
    members_w = {}
    for p, k in zip(probes["by_member"], pw):
        if not k:
            continue
        for solid, parts in p.items():
            for m, f in parts.items():
                acc = members_w.setdefault(m, {"fx": [], "fy": []})
                acc["fx"].append(f["fx"])
                acc["fy"].append(f["fy"])
    summary["by_member_window"] = {m: {"fx": float(np.mean(v["fx"])), "fy": float(np.mean(v["fy"]))}
                                   for m, v in members_w.items()}
    res["march"] = {"summary": summary, "file": fname, "machine_before": out["machine_before"],
                    "machine_after": out["machine_after"]}
    persist(res, NAME + ".json")
    say("march", json.dumps(T60.clean(summary)))
    return summary


def stage_compare(res) -> dict:
    rec8 = os.path.join(HERE, "out", "racelab8", "racelab8.json")
    out: dict = {"against": "out/racelab8/racelab8.json, the drawn car on the repaired porous column"}
    if os.path.isfile(rec8):
        with open(rec8, encoding="utf-8") as fh:
            r8 = json.load(fh)
        out["porous_u_rotor_at_the_settled_state"] = r8["settle"]["u_rotor_at_the_settled_state"]
        out["porous_u_rotor_min_over_the_horizon"] = r8["size"]["U_DUCT_to_size_for"]
        out["porous_u_max_max"] = r8["verify"]["u_max_max"]
        out["porous_geometry_fingerprint"] = r8["geometry"]["fingerprint"]
    M = res.get("march", {}).get("summary")
    if M:
        out["body_fitted_turbine_u_mean"] = M["turbine_u_mean_window"]
        out["body_fitted_core_u_mean"] = M["core_u_mean_window"]
        if "porous_u_rotor_at_the_settled_state" in out:
            out["turbine_ratio"] = M["turbine_u_mean_window"] / out["porous_u_rotor_at_the_settled_state"]
    out["same_drawing"] = (out.get("porous_geometry_fingerprint") ==
                           res.get("car", {}).get("geometry_fingerprint"))
    res["compare"] = out
    persist(res, NAME + ".json")
    say("compare", json.dumps(T60.clean(out)))
    return out


# ---------------------------------------------------------------------------
# the verdicts
# ---------------------------------------------------------------------------


def judge(res) -> dict:
    v: dict = {}
    C = res.get("controls")
    if C:
        v["V1"] = all(
            C[f"tier60_L{L}"]["statuses_identical"] and C[f"tier60_L{L}"]["same_interpolation_points"]
            and C[f"tier60_L{L}"]["same_stencils"]
            and C[f"tier60_L{L}"]["max_weight_difference"] is not None
            and C[f"tier60_L{L}"]["max_weight_difference"] <= 1e-15 for L in (2, 3))
        v["V2"] = all(C[f"multi_L{L}"]["linear_exactness"] <= 1e-10
                      and C[f"multi_L{L}"]["donors_carry_their_own_equation"] for L in (3, 4))
        v["V3"] = len(C["refusals"]) == 4 and all(x is not None for x in C["refusals"].values())
        v["V4"] = C["generator_resolution_independence"] <= 1e-12
    P = res.get("poisson")
    if P and "orders" in P:
        o = P["orders"]
        v["M1"] = all((o[f"{g}.max_err"][0] or 0) >= 1.7 and (o[f"{g}.max_err"][1] or 0) >= 1.8
                      for g in P["levels"][0]["grids"])
    F = res.get("flow")
    if F and "orders" in F:
        o = F["orders"]
        grids = F["levels"][0]["grids"]
        v["M2"] = all((o[f"{g}.u_max_err"][0] or 0) >= 1.6 for g in grids)
        v["M3"] = all((o[f"{g}.p_rms_err"][0] or 0) >= 1.0 for g in grids)
    K = res.get("car")
    if K:
        v["M4"] = (all(e["max_err"] <= 1e-2 for e in K["poisson"]["grids"].values())
                   and K["uniform_stream_error"] <= 1e-12)
    M = res.get("march")
    if M and M.get("failed"):
        v["K1"] = False
    elif M and "summary" in M:
        S = M["summary"]
        v["K1"] = S["steps_done"] == S["steps"] and S["u_max_max"] <= 4.0
        v["K2"] = S["max_div_window"] <= 0.2
        v["K3"] = S["contour_relative_window"] <= 0.01
        v["K4"] = 0.08 <= S["turbine_u_mean_window"] <= 0.40
        v["K5"] = S["duct_flux_mismatch_window"] <= 0.02
        v["K6"] = S["fy_mean_window"] < 0.0
        v["K7"] = S["fx_spread_window"] <= 0.10 * abs(S["fx_mean_window"])
        v["K8"] = S["iterations_max_after_t1"] <= 60
    return v


def stage_summary(res) -> dict:
    verdicts = judge(res)
    res["verdicts"] = verdicts
    res["verdicts_missing"] = sorted({p["id"] for p in PREDICTION} - set(verdicts))
    persist(res, NAME + ".json")
    for p in PREDICTION:
        say("%-3s %-6s %s" % (p["id"], verdicts.get(p["id"]), p["claim"][:100]))
    return verdicts


STAGES = ("controls", "poisson", "flow", "car", "march", "compare", "summary")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="controls,poisson,flow,car,summary")
    args = ap.parse_args(argv)
    configure(args.out)
    res = load(NAME + ".json")
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if "prediction" not in res:
        res.update(tier=62, prediction=list(PREDICTION),
                   prediction_recorded_at=dt.datetime.now().isoformat(timespec="seconds"),
                   read_before_this_run=list(READ_BEFORE_THIS_RUN))
        persist(res, NAME + ".json")
    elif [p["id"] for p in res["prediction"]] != [p["id"] for p in PREDICTION]:
        raise SystemExit("the record's prediction differs from this file's")
    for st in stages:
        if st not in STAGES:
            raise SystemExit(f"unknown stage {st!r}; stages are {STAGES}")
        t0 = time.perf_counter()
        try:
            {"controls": stage_controls, "poisson": stage_poisson, "flow": stage_flow, "car": stage_car,
             "march": stage_march, "compare": stage_compare, "summary": stage_summary}[st](res)
        except Exception:
            err = load(NAME + ".json")
            err.setdefault("stage_errors", {})[st] = traceback.format_exc()[-3000:]
            persist(err, NAME + ".json")
            say(f"stage {st} FAILED")
            raise
        res.setdefault("stage_wall_s", {})[st] = time.perf_counter() - t0
        persist(res, NAME + ".json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
