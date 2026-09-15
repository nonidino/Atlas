"""Tier 61 -- incompressible Navier-Stokes on body-fitted overset grids, verified.

    python scripts/tier61_overset_flow.py --out out/racelab10 --stages mms_space,mms_time
    python scripts/tier61_overset_flow.py --out out/racelab10 --stages cylinder --arm base
    python scripts/tier61_overset_flow.py --out out/racelab10 --stages cylinder --arm dt_half
    python scripts/tier61_overset_flow.py --out out/racelab10 --stages cylinder --arm body_fine
    python scripts/tier61_overset_flow.py --out out/racelab10 --stages price,summary

Tier 60 built and verified the grids, the overlap and the composite pressure
system on an elliptic problem.  This tier marches a flow on them
(`atlas.cases.overset_ns`) and checks it three ways before any car part is put
on it:

  ``mms_space``  a manufactured solution with a MOVING wall, marched to t = 0.4
                 at three resolutions: the error against the exact velocity and
                 pressure, per grid, with and without the box's edge cells.
  ``mms_time``   the same solution at four time steps on one grid; orders from
                 successive differences, so the spatial error cancels.
  ``cylinder``   the circular cylinder at Re = 100, started impulsively and
                 marched to shedding: Strouhal number, drag and lift against
                 the published values, the volume flux through a closed contour
                 in each grid, and two refinements -- half the time step, and a
                 body grid twice as fine each way -- run as separate processes,
                 each into its own file.
  ``price``      the step's cost at the car-size stand-in of Tier 60.
  ``summary``    every registered prediction judged in code.

Nothing here touches RaceLab's column, its records, its gate or the car.
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

from atlas.cases import overset as OV                                   # noqa: E402
from atlas.cases import overset_ns as NS                                # noqa: E402
import tier60_body_fitted_grids as T60                                  # noqa: E402

OUT = NAME = None

#: The manufactured solution's space study: levels 2-4 at one small step.  Level
#: 1 is left out on purpose -- see READ_BEFORE_THIS_RUN: at h = 1/16 a step
#: under h^2 wakes the collocated pressure-correction's spurious pressure mode.
MMS_SPACE_LEVELS = (2, 3, 4)
MMS_SPACE_DT = 0.002
MMS_T = 0.4
MMS_TIME_LEVEL = 3
MMS_TIME_DTS = (0.04, 0.02, 0.01, 0.005)

#: The cylinder arms: the base, and one refinement each way.
CYL_T_END = 120.0
CYL_T_STATS = 70.0
CYL_ARMS = {
    "base": {"h": 1.0 / 32, "ni": 256, "nj": 33, "dt": 0.0125},
    "dt_half": {"h": 1.0 / 32, "ni": 256, "nj": 33, "dt": 0.00625},
    "body_fine": {"h": 1.0 / 32, "ni": 512, "nj": 65, "dt": 0.0125},
}
PRICE_STEPS, PRICE_WARMUP = 30, 10

#: **Read before this file was written**, and carried into the record.
READ_BEFORE_THIS_RUN = (
    "The first march of the manufactured solution blew up within ten steps: with "
    "every box face prescribing velocity the pressure increment was pinned to zero "
    "at one corner cell, the interpolation equations make that pure-Neumann problem "
    "slightly incompatible, and the pin turned the mismatch into a point source. "
    "The same march was stable with an outflow face (no pin) and on the background "
    "alone. Fixed by a uniform compatibility unknown with a zero-mean row.",
    "The background's Dirichlet-edge condition for the increment was first "
    "phi_edge = phi_inner (first order); with it the background pressure's max "
    "error converged at about first order. Replaced by the second-order one-sided "
    "3 phi_edge - 4 phi_in + phi_in2 = 0, and the outflow's zero gradient likewise.",
    "Exploration after both fixes, the manufactured solution to t = 0.4 at "
    "dt = 0.002, max-norm orders L2->L3: background velocity 1.74 and body 1.99; "
    "pressure rms background 1.65 and body 1.88; background pressure max 0.82, "
    "sitting in the box's CORNER cell (0.04, 0.01). Level 1 at that step was far "
    "worse (velocity 6.7e-2, pressure 0.53): dt = 0.002 is below h^2 = 0.0039 "
    "there, the regime in which a collocated pressure-correction's stabilisation "
    "vanishes. At dt = 0.01, L1 read velocity 2.4e-3.",
    "Exploration of the cylinder (h = 1/32, 256 x 33, dt = 0.0125): 106,496 "
    "unknowns, set up in 0.9 s with the pressure factored in 0.82 s; 400 steps in "
    "59 s on mains, 13-22 BiCGSTAB iterations a component; the divergence fell "
    "from 2.2e-2 to 1.4e-3 and the drag coefficient from 1.56 to 1.21 by t = 5, "
    "before any shedding.",
    "Benchmark values read before the bands below were written (2-D, Re = 100): "
    "Ding (2004) Cd 1.325, lift amplitude 0.28, St 0.164, and a virtual-"
    "interpolation-point method Cd 1.328, 0.31, 0.164, both body-conforming "
    "(from arXiv:1401.6513, Table 3); Tseng and Ferziger (2003) Cd 1.42, 0.29, "
    "0.164 and Uhlmann (2005) Cd 1.453, 0.339, 0.169, both immersed-boundary "
    "(quoted in PMC9887058, Table 3). Immersed boundaries read the drag high.",
    "A dry run of this file's mechanics with tiny settings, into a scratch "
    "directory, after P1 was written: at the car-size stand-in, the second step "
    "after an impulsive start cost 0.80 s with 41 BiCGSTAB iterations. P1 was "
    "left as written.",
)

#: **What stages ``diagnose`` and ``space_small_dt`` were written after**: the
#: manufactured-solution stages had run and T1, T2 and S1 had failed.
READ_BEFORE_DIAGNOSIS = (
    "mms_time on the L3 composite: successive time-step differences shrank at "
    "velocity max order 0.89 and 0.94, rms 1.09 and 1.04; pressure rms 1.44 and "
    "0.86 -- first order where BDF2 was predicted second.",
    "mms_space L3->L4: background velocity max order 0.78 with the max in the box "
    "corner's neighbour (0.02, 0.01), body 1.64; pressure rms background 1.39, body "
    "2.50; body divergence 9.4e-6 -> 1.6e-5.",
    "Exploration on the background ALONE (L2, all-Dirichlet, T = 0.4, dt 0.02 -> "
    "0.01 -> 0.005): as built, velocity max order 0.98; without the rotational term "
    "0.95; backward Euler every step 0.99 with differences 7x larger; without "
    "advection (Stokes) 0.94 -- so the order is neither advection nor the "
    "rotational term. BDF2 momentum with the EXACT pressure gradient and no "
    "projection read 1.96. The same scheme with the pressure operator replaced by "
    "the exact divergence-of-correction read max 1.59, rms 1.93.",
    "The exact operator on the COMPOSITE (L3, dt 0.02) blew up at step 38; at L2 "
    "the velocity error grew eightfold every five steps from the overlap (1.67, "
    "1.14), not as a checkerboard (alternating fraction under 0.03), and the march "
    "blew up at step 31. Repeating the compact projection within the step: 2, 3 "
    "and 5 passes left the velocity order at 1.01, 1.06 and 1.10, and halved the "
    "error constant at five.",
    "Where the differences sit: on the background alone and on the composite the "
    "velocity differences were first order everywhere -- rms 1.9e-4 -> 9.4e-5 "
    "beyond 6.6 cells of the box edge -- not a boundary layer.",
)

#: **Registered before stage ``space_small_dt`` ran.**
PREDICTION_SMALL_DT = (
    {"id": "S4", "claim": "with the first-order time error pushed two orders below the spatial "
                          "one (T = 0.05; dt = 0.001 at L2, 0.00025 at L3 and L4), the "
                          "manufactured solution's velocity max-error order L3->L4 is >= 1.7 "
                          "on both grids and its pressure rms order >= 1.5 on both",
     "why": "the time study's constant puts the temporal error near 7e-5 at dt = 0.002 and "
            "T = 0.4, above L4's spatial error, which is what S1 and S2 read; at T = 0.05 "
            "and dt = 0.00025 it is near 1e-6"},
)
SMALL_DT = {2: 0.001, 3: 0.00025, 4: 0.00025}
SMALL_DT_T = 0.05

#: **The prediction, registered before any stage of this file ran**, judged in
#: code by `judge` under the same ids.
PREDICTION = (
    {"id": "S1", "claim": "manufactured solution, L3->L4: velocity max-error order >= 1.7 on "
                          "both grids",
     "why": "second-order operators, third-order interpolation; L2->L3 read 1.74 and 1.99, "
            "and at dt = 0.002 the temporal error should still sit below L4's spatial one"},
    {"id": "S2", "claim": "manufactured solution, L3->L4: pressure rms-error order >= 1.5 on both "
                          "grids, and the pressure max error away from the box's edge cells "
                          "converges at >= 1.5 on both",
     "why": "the rotational correction keeps the pressure's boundary layer thin, and the "
            "corner cell that held the max at L3 is excluded by construction"},
    {"id": "S3", "claim": "manufactured solution: the body grid's largest divergence falls by "
                          "at least 2x per level, L2->L3 and L3->L4",
     "why": "an approximate projection leaves O(h^2) divergence behind, not O(1)"},
    {"id": "T1", "claim": "manufactured solution on the L3 grid, successive time-step differences: "
                          "velocity order >= 1.8 between the two finest pairs",
     "why": "BDF2 with an exact start; the rotational scheme's velocity is second order "
            "in time with Dirichlet data"},
    {"id": "T2", "claim": "the same study: pressure rms order >= 1.3 between the two finest pairs",
     "why": "the analysis gives the pressure O(dt^(3/2)) near a boundary"},
    {"id": "C1", "claim": "cylinder at Re = 100, the base arm over t in [70, 120]: Strouhal number "
                          "in [0.160, 0.175], mean drag coefficient in [1.30, 1.42], lift "
                          "amplitude in [0.28, 0.36]",
     "why": "the body-conforming references read 0.164, 1.325-1.328 and 0.28-0.31; a 5% "
            "blockage raises drag and frequency a few percent, and the band stops "
            "short of the immersed-boundary values' high drag"},
    {"id": "C2", "claim": "the base arm sheds regularly: at least 15 whole lift cycles in the "
                          "window, with the period's spread under 1% of its mean",
     "why": "at Re = 100 shedding is strictly periodic once established, and a small "
            "asymmetric start should establish it well before t = 70"},
    {"id": "C3", "claim": "mass through the overlap: over the window, the mean absolute net "
                          "volume flux out of a background rectangle enclosing the body grid is "
                          "<= 1e-3 of the free-stream flux through its height, and through a "
                          "ring of the body grid <= 1e-3 of the free stream times the ring's "
                          "diameter",
     "why": "interpolation is not conservative, but at third order across a well-resolved "
            "overlap the leak should sit far below a percent"},
    {"id": "C4", "claim": "half the time step changes the Strouhal number by < 1% and the mean "
                          "drag by < 2%",
     "why": "BDF2 at dt = 0.0125 is about 200 steps a shedding period"},
    {"id": "C5", "claim": "a body grid twice as fine each way changes the Strouhal number by < 1% "
                          "and the mean drag by < 2%",
     "why": "the base grid's wall spacing, 0.0022, puts about 18 rows inside a boundary "
            "layer of thickness D / sqrt(Re)"},
    {"id": "P1", "claim": "at Tier 60's car-size stand-in one flow step costs <= 1.0 s on mains, "
                          "with at most 40 BiCGSTAB iterations a component",
     "why": "the cylinder's 106k unknowns stepped in 0.13 s; the stand-in has about 205k and "
            "eight walls"},
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
# stages
# ---------------------------------------------------------------------------


def _mms_record(r: dict) -> dict:
    """What a manufactured-solution run keeps: everything but its fields."""
    return {k: v for k, v in r.items() if k != "fields"}


def stage_mms_space(res) -> dict:
    out = {"dt": MMS_SPACE_DT, "T": MMS_T, "levels": []}
    for level in MMS_SPACE_LEVELS:
        t0 = time.perf_counter()
        r = NS.mms_run(level, dt=MMS_SPACE_DT, T=MMS_T)
        r = _mms_record(r)
        r["wall_total_s"] = time.perf_counter() - t0
        out["levels"].append(r)
        say("mms_space L%d n=%d %.1f s  %s" % (level, r["n_unknowns"], r["wall_total_s"],
                                            {g: "%.2e/%.2e" % (e["u_max_err"], e["p_rms_err"])
                                             for g, e in r["grids"].items()}))
        res["mms_space"] = out
        persist(res, NAME + ".json")
    orders = {}
    for g in out["levels"][0]["grids"]:
        for key in ("u_max_err", "u_rms_err", "p_max_err", "p_rms_err",
                    "p_max_err_off_edges", "u_max_err_off_edges"):
            orders[f"{g}.{key}"] = _orders([lv["grids"][g][key] for lv in out["levels"]])
        orders[f"{g}.divergence"] = _orders([lv["divergence"][g] for lv in out["levels"]])
    out["orders"] = orders
    res["mms_space"] = out
    persist(res, NAME + ".json")
    return out


def stage_mms_time(res) -> dict:
    runs = []
    for d in MMS_TIME_DTS:
        t0 = time.perf_counter()
        r = NS.mms_run(MMS_TIME_LEVEL, dt=d, T=MMS_T)
        runs.append(r)
        say("mms_time dt=%.4f steps=%d %.1f s" % (d, r["steps"], time.perf_counter() - t0))
    diffs = {"u": [], "p": []}
    for a, b in zip(runs[:-1], runs[1:]):
        du = np.hypot(a["fields"]["u"] - b["fields"]["u"], a["fields"]["v"] - b["fields"]["v"])
        dp = a["fields"]["p"] - b["fields"]["p"]
        diffs["u"].append({"max": float(du.max()), "rms": float(np.sqrt(np.mean(du ** 2)))})
        diffs["p"].append({"max": float(np.abs(dp).max()), "rms": float(np.sqrt(np.mean(dp ** 2)))})
    out = {"level": MMS_TIME_LEVEL, "T": MMS_T, "dts": list(MMS_TIME_DTS),
           "runs": [_mms_record(r) for r in runs], "successive_differences": diffs,
           "orders": {f"{q}.{n}": _orders([x[n] for x in diffs[q]])
                      for q in ("u", "p") for n in ("max", "rms")}}
    res["mms_time"] = out
    persist(res, NAME + ".json")
    say("mms_time orders", json.dumps(T60.clean(out["orders"])))
    return out


def _mms_flow(level, dt_, *, with_body=True, chi=1.0, **kw):
    nu = 0.01
    n16 = 2 ** (level - 1)
    h = 1.0 / (16 * n16)
    bg = OV.CartesianGrid("bg", int(4 / h), int(3 / h), h)
    comps = ([OV.ogrid_annulus("body", 2.0, 1.5, 0.3, 0.75, 48 * n16, 6 * n16 + 1, beta=1.0)]
             if with_body else [])
    ov = OV.Overset(bg, comps, hole_margin=0.15 if with_body else 0.0, width=3)
    f = NS.OversetFlow(ov, nu, dt_, box={k: "dirichlet" for k in NS.FACES},
                       box_velocity=lambda x, y, t: NS.mms_fields(x, y, t, nu)[:2],
                       wall_velocity=lambda g, t: NS.mms_fields(g.x[0], g.y[0], t, nu)[:2],
                       forcing=lambda x, y, t: NS.mms_fields(x, y, t, nu)[3:], chi=chi, **kw)
    u0, v0, p0, _a, _b = NS.mms_fields(f.X, f.Y, 0.0, nu)
    um, vm, _p, _c, _d = NS.mms_fields(f.X, f.Y, -dt_, nu)
    f.set_state(u0, v0, p0, u_prev=um, v_prev=vm)
    return f


def _momentum_only_step(f):
    """BDF2 momentum with the EXACT pressure gradient and no projection -- the
    control that says whether the time integration itself is second order."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    nu, dt_ = f.nu, f.dt
    d = f.disc
    t1 = f.t + dt_
    us, vs = 2 * f.U - f.Um1, 2 * f.V - f.Vm1
    M = (f.K_fixed + f.K_visc + (1.5 / dt_) * f.P_disc + sp.diags(us * d) @ f.Dx
         + sp.diags(vs * d) @ f.Dy).tocsc()
    _u, _v, _p, fu, fv = NS.mms_fields(f.X, f.Y, t1, nu)
    B, e, g, w = (NS.MMS[k] for k in ("B", "e", "g", "omega"))
    s = math.sin(w * t1)
    px = -B * e * np.sin(e * f.X) * np.sin(g * f.Y) * s
    py = B * g * np.cos(e * f.X) * np.cos(g * f.Y) * s
    bu = np.zeros_like(f.U)
    bv = np.zeros_like(f.V)
    bu[d] = ((2 * f.U - 0.5 * f.Um1) / dt_ - px + fu)[d]
    bv[d] = ((2 * f.V - 0.5 * f.Vm1) / dt_ - py + fv)[d]
    ed = f.edge_d
    bu[ed], bv[ed] = NS.mms_fields(f.X[ed], f.Y[ed], t1, nu)[:2]
    lu = spla.splu(M)
    f.Um1, f.Vm1 = f.U, f.V
    f.U, f.V = lu.solve(bu), lu.solve(bv)
    f.t = t1


def _time_orders(arm_fn, dts=(0.02, 0.01, 0.005), T=0.4):
    finals = []
    for d in dts:
        f, stepper = arm_fn(d)
        steps = int(round(T / d))
        done = 0
        try:
            for _ in range(steps):
                stepper(f)
                done += 1
        except Exception as exc:
            return {"failed_at_step": done + 1, "dt": d, "message": str(exc)[:200]}
        finals.append(f)
    du = [np.hypot(a.U - b.U, a.V - b.V) for a, b in zip(finals[:-1], finals[1:])]
    mx = [float(x.max()) for x in du]
    rms = [float(np.sqrt(np.mean(x ** 2))) for x in du]
    return {"dts": list(dts), "T": T, "diff_max": mx, "diff_rms": rms,
            "order_max": _orders(mx), "order_rms": _orders(rms)}


def stage_diagnose(res) -> dict:
    """The time-order diagnosis, recorded -- written after T1 and T2 failed, and
    every arm first seen in an exploration (READ_BEFORE_DIAGNOSIS)."""
    step = lambda f: f.step()                                   # noqa: E731
    arms = {
        "background_as_built": lambda d: (_mms_flow(2, d, with_body=False), step),
        "background_no_rotational_term": lambda d: (_mms_flow(2, d, with_body=False, chi=0.0), step),
        "background_momentum_only_exact_pressure":
            lambda d: (_mms_flow(2, d, with_body=False), _momentum_only_step),
        "background_exact_projection": lambda d: (_mms_flow(2, d, with_body=False, projection="exact"), step),
        "composite_compact_one_pass": lambda d: (_mms_flow(3, d), step),
        "composite_compact_five_passes": lambda d: (_mms_flow(3, d, passes=5), step),
        "composite_exact_projection": lambda d: (_mms_flow(3, d, projection="exact"), step),
    }
    out = {}
    for label, fn in arms.items():
        t0 = time.perf_counter()
        out[label] = _time_orders(fn)
        out[label]["wall_s"] = time.perf_counter() - t0
        say("diagnose %-44s %s" % (label, json.dumps(T60.clean(
            {k: out[label].get(k) for k in ("order_max", "order_rms", "failed_at_step")}))))
        res["diagnose"] = out
        persist(res, NAME + ".json")
    return out


def stage_space_small_dt(res) -> dict:
    out = {"T": SMALL_DT_T, "dts": {str(k): v for k, v in SMALL_DT.items()}, "levels": []}
    for level, d in SMALL_DT.items():
        t0 = time.perf_counter()
        r = _mms_record(NS.mms_run(level, dt=d, T=SMALL_DT_T))
        r["wall_total_s"] = time.perf_counter() - t0
        out["levels"].append(r)
        say("space_small_dt L%d dt=%g %.1f s %s" % (level, d, r["wall_total_s"],
                                                   {g: "%.2e/%.2e" % (e["u_max_err"], e["p_rms_err"])
                                                    for g, e in r["grids"].items()}))
        res["space_small_dt"] = out
        persist(res, NAME + ".json")
    orders = {}
    for g in out["levels"][0]["grids"]:
        for key in ("u_max_err", "u_rms_err", "p_max_err", "p_rms_err", "u_max_err_off_edges",
                    "p_max_err_off_edges"):
            orders[f"{g}.{key}"] = _orders([lv["grids"][g][key] for lv in out["levels"]])
    out["orders"] = orders
    res["space_small_dt"] = out
    persist(res, NAME + ".json")
    return out


def stage_cylinder(res, arm: str) -> dict:
    spec = CYL_ARMS[arm]
    fname = f"cylinder_{arm}.json"
    D = NS.CYLINDER["diameter"]
    out = {"arm": arm, "spec": spec, "t_end": CYL_T_END, "t_stats": CYL_T_STATS,
           "machine_before": T60.machine_state(), "started_at": dt.datetime.now().isoformat()}
    flow = NS.cylinder_flow(h=spec["h"], ni=spec["ni"], nj=spec["nj"], dt=spec["dt"])
    cx, cy = NS.CYLINDER["centre"]
    h = spec["h"]
    r_out = 0.5 * D + NS.CYLINDER["thickness"]
    i0, i1 = int(math.floor((cx - r_out - 0.1) / h)), int(math.ceil((cx + r_out + 0.1) / h))
    j0, j1 = int(math.floor((cy - r_out - 0.1) / h)), int(math.ceil((cy + r_out + 0.1) / h))
    height = (j1 - j0 + 1) * h
    ring_j = spec["nj"] // 2
    c = flow.ov.comps[0]
    ring_r = float(np.hypot(c.x[ring_j] - cx, c.y[ring_j] - cy).mean())
    out.update(n_unknowns=flow.ov.n_unknowns, setup_s=flow.setup_s,
               pressure_factor_s=flow.p_factor_s, rectangle_cells=[i0, i1, j0, j1],
               rectangle_height=height, ring_row=ring_j, ring_radius=ring_r)
    steps = int(round(CYL_T_END / spec["dt"]))
    tr = {k: [] for k in ("t", "cd", "cl", "cd_pressure", "cl_pressure", "iterations",
                          "step_s", "max_div", "u_max")}
    flux = {"t": [], "rectangle": [], "ring": []}
    t_start = time.perf_counter()
    for s in range(steps):
        try:
            rec = flow.step()
        except Exception:
            out.update(trajectory=tr, flux=flux, steps_done=s, failed=traceback.format_exc()[-3000:])
            persist(out, fname)
            raise
        f = flow.forces(c.name)
        tr["t"].append(rec["t"])
        tr["cd"].append(2.0 * f["fx"] / D)
        tr["cl"].append(2.0 * f["fy"] / D)
        tr["cd_pressure"].append(2.0 * f["fx_pressure"] / D)
        tr["cl_pressure"].append(2.0 * f["fy_pressure"] / D)
        tr["iterations"].append(max(rec["iterations"]))
        tr["step_s"].append(rec["step_s"])
        tr["max_div"].append(rec["max_div"])
        tr["u_max"].append(rec["u_max"])
        if s % 8 == 0:
            flux["t"].append(rec["t"])
            flux["rectangle"].append(flow.box_flux(i0, i1, j0, j1))
            flux["ring"].append(flow.ring_flux(c.name, ring_j))
        if (s + 1) % 800 == 0 or s + 1 == steps:
            out.update(trajectory=tr, flux=flux, steps_done=s + 1, wall_s=time.perf_counter() - t_start)
            persist(out, fname)
            say("cylinder %s step %d/%d t %.2f Cd %.4f Cl %+.4f its %d div %.2e" % (
                arm, s + 1, steps, rec["t"], tr["cd"][-1], tr["cl"][-1], tr["iterations"][-1],
                rec["max_div"]))
    st = NS.shedding_statistics(tr["t"], tr["cd"], tr["cl"], CYL_T_STATS, D)
    tt = np.asarray(flux["t"])
    w = tt >= CYL_T_STATS
    out["statistics"] = st
    out["flux_window"] = {
        "rectangle_mean_abs": float(np.abs(np.asarray(flux["rectangle"])[w]).mean()),
        "rectangle_scale": height * 1.0,
        "ring_mean_abs": float(np.abs(np.asarray(flux["ring"])[w]).mean()),
        "ring_scale": 2.0 * ring_r * 1.0,
    }
    out["machine_after"] = T60.machine_state()
    out["finished_at"] = dt.datetime.now().isoformat()
    persist(out, fname)
    say("cylinder", arm, json.dumps(T60.clean(st)))
    return out


def stage_price(res) -> dict:
    out = {"machine_before": T60.machine_state()}
    bg, grids = T60.car_size_standin()
    ov = OV.Overset(bg, grids, hole_margin=0.1, width=3)
    flow = NS.OversetFlow(ov, 0.004, 0.0125,
                          box={"xlo": "dirichlet", "ylo": "dirichlet", "yhi": "dirichlet",
                               "xhi": "outflow"})
    u = np.ones(ov.n_unknowns)
    u[flow.wall] = 0.0
    flow.set_state(u, np.zeros(ov.n_unknowns))
    out.update(n_unknowns=ov.n_unknowns, setup_s=flow.setup_s, pressure_factor_s=flow.p_factor_s)
    recs = []
    for s in range(PRICE_WARMUP + PRICE_STEPS):
        rec = flow.step()
        if s >= PRICE_WARMUP:
            recs.append(rec)
    keys = ("step_s", "momentum_s", "pressure_s", "assemble_s")
    out["steps"] = [{k: r[k] for k in keys + ("iterations", "max_div")} for r in recs]
    out["median"] = {k: float(np.median([r[k] for r in recs])) for k in keys}
    out["iterations_max"] = int(max(max(r["iterations"]) for r in recs))
    out["machine_after"] = T60.machine_state()
    res["price"] = out
    persist(res, NAME + ".json")
    say("price", json.dumps(T60.clean(out["median"])), "its max", out["iterations_max"])
    return out


# ---------------------------------------------------------------------------
# the verdicts
# ---------------------------------------------------------------------------


def _last(o):
    return o[-1] if o else None


def judge(res) -> dict:
    v: dict = {}
    S = res.get("mms_space")
    if S and "orders" in S:
        o = S["orders"]
        v["S1"] = all((_last(o[f"{g}.u_max_err"]) or 0) >= 1.7 for g in ("bg", "body"))
        v["S2"] = all((_last(o[f"{g}.p_rms_err"]) or 0) >= 1.5
                      and (_last(o[f"{g}.p_max_err_off_edges"]) or 0) >= 1.5 for g in ("bg", "body"))
        v["S3"] = all((x or 0) >= 1.0 for x in o["body.divergence"])
    Tm = res.get("mms_time")
    if Tm:
        o = Tm["orders"]
        v["T1"] = all((x or 0) >= 1.8 for x in o["u.max"][-2:])
        v["T2"] = all((x or 0) >= 1.3 for x in o["p.rms"][-2:])
    cyl = res.get("cylinder", {})
    base = cyl.get("base")
    if base and base.get("statistics", {}).get("strouhal") is not None:
        st = base["statistics"]
        v["C1"] = (0.160 <= st["strouhal"] <= 0.175 and 1.30 <= st["cd_mean"] <= 1.42
                   and 0.28 <= st["cl_amplitude"] <= 0.36)
        v["C2"] = st["cycles"] >= 15 and st["period_std"] <= 0.01 * st["period_mean"]
        fw = base["flux_window"]
        v["C3"] = (fw["rectangle_mean_abs"] <= 1e-3 * fw["rectangle_scale"]
                   and fw["ring_mean_abs"] <= 1e-3 * fw["ring_scale"])
        for key, arm in (("C4", "dt_half"), ("C5", "body_fine")):
            other = cyl.get(arm)
            if other and other.get("statistics", {}).get("strouhal") is not None:
                so = other["statistics"]
                v[key] = (abs(so["strouhal"] - st["strouhal"]) < 0.01 * st["strouhal"]
                          and abs(so["cd_mean"] - st["cd_mean"]) < 0.02 * st["cd_mean"])
    P = res.get("price")
    # P1's claim is a cost ON MAINS.  The first version of this line judged the
    # number whatever the power state -- the prose and the code disagreeing in the
    # permissive direction -- and the stage was about to run on battery.  A price
    # not taken on mains, or not taken at all, is not a measurement of P1: None.
    v["P1"] = None
    if P:
        mains = (P.get("machine_before", {}).get("on_mains") is True
                 and P.get("machine_after", {}).get("on_mains") is True)
        if mains:
            v["P1"] = P["median"]["step_s"] <= 1.0 and P["iterations_max"] <= 40
    SD = res.get("space_small_dt")
    if SD and "orders" in SD:
        o = SD["orders"]
        v["S4"] = all((_last(o[f"{g}.u_max_err"]) or 0) >= 1.7
                      and (_last(o[f"{g}.p_rms_err"]) or 0) >= 1.5 for g in ("bg", "body"))
    return v


def stage_summary(res) -> dict:
    cyl = {}
    for arm in CYL_ARMS:
        rec = load(f"cylinder_{arm}.json")
        if rec:
            cyl[arm] = {k: val for k, val in rec.items() if k != "trajectory"}
            cyl[arm]["trajectory_file"] = f"cylinder_{arm}.json"
    res["cylinder"] = cyl
    verdicts = judge(res)
    res["verdicts"] = verdicts
    registered = {p["id"] for p in PREDICTION}
    if "prediction_small_dt" in res:
        registered |= {p["id"] for p in PREDICTION_SMALL_DT}
    res["verdicts_missing"] = sorted(registered - set(verdicts))
    persist(res, NAME + ".json")
    for p in PREDICTION + (PREDICTION_SMALL_DT if "prediction_small_dt" in res else ()):
        say("%-3s %-6s %s" % (p["id"], verdicts.get(p["id"]), p["claim"][:100]))
    return verdicts


STAGES = ("mms_space", "mms_time", "diagnose", "space_small_dt", "cylinder", "price", "summary")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="mms_space,mms_time,price,summary")
    ap.add_argument("--arm", default="base", choices=sorted(CYL_ARMS))
    args = ap.parse_args(argv)
    configure(args.out)
    res = load(NAME + ".json")
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if "prediction" not in res:
        if stages == ["cylinder"]:
            raise SystemExit("register the prediction first: run any other stage before an arm")
        res.update(tier=61, prediction=list(PREDICTION),
                   prediction_recorded_at=dt.datetime.now().isoformat(timespec="seconds"),
                   read_before_this_run=list(READ_BEFORE_THIS_RUN),
                   cylinder_arms=CYL_ARMS, cylinder_t_end=CYL_T_END, cylinder_t_stats=CYL_T_STATS)
        persist(res, NAME + ".json")
    elif [p["id"] for p in res["prediction"]] != [p["id"] for p in PREDICTION]:
        raise SystemExit("the record's prediction differs from this file's")
    for st in stages:
        if st not in STAGES:
            raise SystemExit(f"unknown stage {st!r}; stages are {STAGES}")
        if st in ("diagnose", "space_small_dt") and "prediction_small_dt" not in res:
            res.update(prediction_small_dt=list(PREDICTION_SMALL_DT),
                       prediction_small_dt_recorded_at=dt.datetime.now().isoformat(timespec="seconds"),
                       read_before_diagnosis=list(READ_BEFORE_DIAGNOSIS))
            persist(res, NAME + ".json")
        t0 = time.perf_counter()
        try:
            if st == "cylinder":
                stage_cylinder(res, args.arm)
            else:
                {"mms_space": stage_mms_space, "mms_time": stage_mms_time,
                 "diagnose": stage_diagnose, "space_small_dt": stage_space_small_dt,
                 "price": stage_price, "summary": stage_summary}[st](res)
        except Exception:
            err = load(NAME + ".json") if st != "cylinder" else res
            err.setdefault("stage_errors", {})[st if st != "cylinder" else f"cylinder_{args.arm}"] = \
                traceback.format_exc()[-3000:]
            if st != "cylinder":
                persist(err, NAME + ".json")
            say(f"stage {st} FAILED")
            raise
        if st != "cylinder":
            res.setdefault("stage_wall_s", {})[st] = time.perf_counter() - t0
            persist(res, NAME + ".json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
