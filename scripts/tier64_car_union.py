"""Tier 64 -- the radiator core and the turbine in the body-fitted duct, and the three joins.

    python scripts/tier64_car_union.py --out out/racelab13 --stages controls,prefix
    python scripts/tier64_car_union.py --out out/racelab13 --stages arms
    python scripts/tier64_car_union.py --out out/racelab13 --stages slow,compare,summary

Tier 63 opened the solid car's duct and measured its flow, forward and weak
(W281); the user accepted it for now.  This file puts the devices into that duct
(`car_union`) and marches the joins:

  ``controls``  the devices' pieces with known answers: the duct's open band at
                every station read, the force's quadrature, the interpolation
                matrices, the machine's generating band and its similarity, the
                uniform stream.
  ``prefix``    the car from the stream to t = 8 with no devices -- Tier 63's
                march, repeated in a new process and compared step by step --
                and its state kept.  Every arm starts from it.
  ``arms``      the devices switched on at t = 8 and marched to t = 16, the
                machine sized for the previous arm's minimum inflow over
                [12, 16] (`racelab`'s W228 rule, iterated) until an arm stays
                inside every declared envelope over that window.
  ``null``      the same window with no devices at all, released from the same
                state: the pressure-jump formula's own floor where the arms read
                it, and what the restart itself costs.  Added after the prefix.
  ``ring``      what a ring reads, point by point, on the state the arms release
                from, and how much of the duct's air is moving backwards.  Added
                after arm 1.
  ``slow``      J2 on the coolant clock at the registered arm's heat
                (`tier53_racelab_rerun.stage_slow`, imported).
  ``compare``   the registered arm against Tier 63 (no devices) and the porous
                column's referent (Tier 59).
  ``summary``   every registered prediction judged in code.

Nothing here touches RaceLab's column, its records, its gate or the demo.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse                                                         # noqa: E402
import datetime as dt                                                   # noqa: E402
import gc                                                               # noqa: E402
import json                                                             # noqa: E402
import sys                                                              # noqa: E402
import time                                                             # noqa: E402
import traceback                                                        # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(HERE, "scripts")
for _p in (HERE, SCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                      # noqa: E402
import torch                                                            # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert torch.backends.cuda.matmul.allow_tf32 is False
assert torch.backends.cudnn.allow_tf32 is False

from atlas.cases import car_solids as CS                                # noqa: E402
from atlas.cases import car_union as CU                                 # noqa: E402
from atlas.cases import cooling_loop as CL                              # noqa: E402
from atlas.cases import ground_effect as GE                             # noqa: E402
from atlas.cases import integration_union as IU                         # noqa: E402
from atlas.cases import overset as OV                                   # noqa: E402
from atlas.cases import racelab as RL                                   # noqa: E402
from atlas.cases import vehicle_march as VM                             # noqa: E402
import tier62_car_solids as T62                                         # noqa: E402
import tier63_duct_openings as T63                                      # noqa: E402

TIER63_RECORD = os.path.join(HERE, "out", "racelab12", "racelab12.json")
TIER63_MARCH = os.path.join(HERE, "out", "racelab12", "march.json")
POROUS_RECORD = os.path.join(HERE, "out", "racelab8", "racelab8.json")

#: the devices switch on here; every arm starts from the prefix's state at it
T_ON = 8.0
T_END = 16.0
WINDOW = (12.0, 16.0)
#: arms at most; the W228 rule is iterated until one is admitted
K_MAX = 4
#: the first arm's sizing: Tier 63's own minimum of the duct's flow over [12, 16]
#: at the turbine plane, read out of its record, not typed
PROBE_EVERY = 8
PERSIST_EVERY = 40
#: the pressure lines, in cells from each plane
LINES = (-24.0, -8.0, 8.0, 24.0)

READ_BEFORE_THIS_RUN = (
    "Tier 63's march (out/racelab12): the duct's mean streamwise velocity at the turbine plane "
    "0.585 at t = 0.5, 0.335 at 1, 0.147 at 2, 0.114 at 3, 0.123 at 4, 0.119 at 6, 0.107 at 8, "
    "0.098 at 10, 0.092 at 12 and 0.0876 at 16; over [12, 16] mean 0.0895, minimum 0.0876, maximum "
    "0.0921; the core and turbine planes' fluxes agreeing to 9.5e-5; drag 2.913 and vertical force "
    "+2.940; largest speed 3.37, largest divergence over the window 0.031, contour 0.16%; median "
    "step 1.25 s (on battery at its start, on mains at its end; the laptop slept 15:32 to 17:10).",
    "The porous column's referent arm (out/racelab8, 600 macro-steps from its settled state): J3's "
    "receiving residual 0.0292 with u_plane / u_ring 0.968 (ring 7.5 cells upstream, porous duct "
    "walls); J1's tracking residual 9.8e-8; J2's first law 1.08e-8 over its one coolant step; the "
    "slow stage's P4 5.6e-9 with the mount term and 0.998 without; q_machine 1.265 W at a host "
    "inflow of 0.4457 (the minimum over its horizon) and induction 0.0516; back-EMF 1.357 against "
    "the battery's 1.34.",
    "The build repo's disk.py: ActuatorDisk gives thrust 1/2 rho A C_T' u_d^2 and power = thrust u_d "
    "with u_d the ring's mean. So the shaft's claim is T <u_ring>, and the force's work is T times "
    "the mean velocity where it acts; in a solid duct continuity holds the two means together.",
    "The machine's generating band, measured on the similarity with the duct's open width (29 "
    "cells, 0.4531): the rotor has an operating point and the machine generates for inflows from "
    "0.973138 to 1.140759 of the sizing inflow, the same at sizing inflows 0.0876, 0.08 and 0.07. "
    "At the sizing point the induction is 0.1139 and, at 0.08, the current 4.31e-5 and q_machine "
    "0.040 W (p_ref held at 42187.5); 0.265 W at 1.05 and 0.69 W at 1.10. Below the band the rotor "
    "sits at its lower induction and the machine motors; above it the induction reaches the clamp "
    "0.4, and nothing raised up to five times the sizing inflow.",
    "The devices on the opened car's composite, nothing marched: the duct's open band is 19.5 to "
    "48.5 cells at x = 256, 272, 280, 288, 304, 368, 384, 392 and 400; DUCT_LO and DUCT_UP are "
    "shells of their own; the strip quadrature's weights sum to 1 within 2.2e-16; the background's "
    "samples of each force integrate to 1.0 over all its points and to 0.724 over its "
    "discretisation points, the rest carried near the walls by the two wall grids (207 forced "
    "points each); a ring's 32 points come 6 from each wall grid and 20 from the background; the "
    "interpolation matrix matches car_solids.probe to 1.3e-15 and reproduces linear fields to "
    "2.7e-15; one forcing call takes 5 ms against a 1.25 s step.",
    "[AI Inference] A lumped estimate made before any march: Poiseuille friction along the "
    "168-cell duct at u = 0.0876 is about 0.054, the core's 1/2 K <u^2> about 0.0055, so the core "
    "alone should lower the duct's flow by about a tenth; the turbine at its sizing point adds "
    "1/2 C_T' u^2, about 0.002, and its thrust rises some thirty-fold across the band.",
    "Nothing with a device in it was marched on the body-fitted car before these predictions.",
)

PREDICTION = (
    {"id": "C1", "claim": "the strip quadrature integrates each device's force to one within 1e-12, "
                          "and the background's samples of it integrate to one within 1e-12 over "
                          "all its points",
     "why": "midpoint weights of a raised cosine over whole periods, and its shifted copies are a "
            "partition of unity"},
    {"id": "C2", "claim": "the interpolation matrices agree with car_solids.probe within 1e-13 on a "
                          "smooth field and reproduce linear fields within 1e-12",
     "why": "the same donors and weights, summed in a different order"},
    {"id": "C3", "claim": "the duct's open band is [19.5, 48.5] cells within 1e-9 at every station "
                          "read: both rings, both planes and every pressure line",
     "why": "DUCT_LO and DUCT_UP are straight panels three cells thick with no other solid near"},
    {"id": "C4", "claim": "the machine's generating band is the same at three sizing inflows within "
                          "1e-6, the sizing point's induction is the reference's within 1e-9 at "
                          "each, and at the reference inflow machine_for_host is machine_for_rotor "
                          "exactly",
     "why": "the similarity makes the band a property of the powertrain's declaration"},
    {"id": "C5", "claim": "on the uniform stream the rings read one and the force-weighted plane "
                          "means are one, within 1e-12",
     "why": "interpolation reproduces a constant and the weights sum to one"},
    {"id": "U0", "claim": "the prefix repeats Tier 63's march: at every step to t = 8 its forces, "
                          "largest speed and largest divergence, and at every probe its duct and "
                          "contour fluxes, agree with Tier 63's record within 1e-6 relative",
     "why": "the same code, geometry and machine; not necessarily bitwise, because a BLAS reduction's "
            "order can move and the wheels shed"},
    {"id": "U1", "claim": "arm 1, sized for Tier 63's own window minimum, is NOT admitted over "
                          "[12, 16], and its declined steps there include MGU's: the machine motors",
     "why": "the core alone should take about a tenth off the flow, and the band's lower edge is "
            "2.7% below the sizing"},
    {"id": "U2", "claim": "an arm is admitted over [12, 16] by arm 3 at the latest, each arm sized "
                          "for the previous arm's minimum inflow over the window",
     "why": "the turbine's thrust climbs steeply across the band and governs the flow near its "
            "lower edge, so the second re-sizing moves by less than the band"},
    {"id": "U3", "claim": "in the admitted arm every declined step lies before t = 12, and each is "
                          "ROTOR's (the induction at its clamp), never MGU's",
     "why": "before the window the duct runs faster than its minimum, above the band, where the "
            "machine still generates"},
    {"id": "U4", "claim": "J3 in the admitted arm: the receiving residual over [12, 16] is at most "
                          "0.005, and charging the shaft against the core's work instead gives at "
                          "least 0.5",
     "why": "continuity in a solid duct holds the plane's mean velocity at the ring's; what is left "
            "is the ring's 32-point midpoint rule on a profile that vanishes at both walls (about "
            "5e-4) and a one-step lag. The porous column read 0.0292. The second clause is the "
            "core's work over the shaft's claim, K <u^2> / (C_T' <u>^2), about 2.8 at the sizing "
            "induction 0.114 and 1 near induction 0.25 -- so it also measures where in its band the "
            "arm settled, and it stops discriminating if the turbine pulls as hard as the core"},
    {"id": "U5", "claim": "J1 in the admitted arm: at every step of the window UA is UA_RAD (u/u_ref)^0.8 "
                          "within 1e-12 relative; receiver_balances' tracking residual over the "
                          "window is above P3's 1e-6, and equals the window's Jensen gap within 1e-12",
     "why": "per step it is algebra; receiver_balances compares the mean of UA with UA at the mean "
            "air, and the air still moves over this window"},
    {"id": "U6", "claim": "J2 in the admitted arm: its coolant step closes the block's first law "
                          "within 1e-6; the slow recipe at the arm's heat and UA passes both of P4's "
                          "clauses; and the machine's mean heat over the window is below the porous "
                          "column's 1.265 W",
     "why": "the first law is the block solver's; the heat falls with the cube of the inflow ratio"},
    {"id": "U7", "claim": "the fluid feels the forces: over [12, 16] the pressure jump across each "
                          "device, net of the friction over the adjacent 16-cell stretch, is T / W "
                          "within 20% for the core and 40% for the turbine",
     "why": "in a developed duct flow a force uniform across the band is balanced by a pressure "
            "jump alone"},
    {"id": "U8", "claim": "the devices slow the duct: the admitted arm's mean velocity at the turbine "
                          "plane over [12, 16] is 3% to 20% below Tier 63's 0.0895",
     "why": "the lumped estimate's tenth for the core, plus the turbine"},
    {"id": "U9", "claim": "the admitted arm marches soundly: no blow-up and the largest speed at most "
                          "4 over [8, 16]; over [12, 16] the largest divergence at most 0.2, the "
                          "contour's net flux at most 1% of its inflow, the core and turbine planes' "
                          "fluxes within 2%, and at most 60 BiCGSTAB iterations a component",
     "why": "Tier 63 read 3.37, 0.031, 0.16%, 9.5e-5 and 7 on the same machinery; the devices are "
            "forces of a few thousandths"},
    {"id": "U10", "claim": "the admitted arm's drag and vertical force over [12, 16] are within 2% of "
                           "Tier 63's",
     "why": "the devices' thrusts are about a thousandth of the car's drag and act inside the duct"},
    {"id": "U11", "claim": "the joins cost at most 5% of the median fluid step",
     "why": "the rings and the work are sparse products, the operating point a bisection, the "
            "forcing 5 ms"},
)


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

OUT = NAME = None


def configure(out: str) -> None:
    global OUT, NAME
    OUT = os.path.abspath(out)
    NAME = os.path.basename(os.path.normpath(OUT))
    T62.configure(out)
    T62.GEOMETRY = T63.geometry


def persist(obj, fname: str) -> str:
    return T62.persist(obj, fname)


def load(fname: str) -> dict:
    return T62.load(fname)


def say(*a) -> None:
    print(*a, flush=True)


def prefix_path() -> str:
    return os.path.join(OUT, "cache", "prefix_t8.npz")


_DEV: dict = {}


def _devices():
    C = T62._car()
    if "D" not in _DEV:
        _DEV["D"] = CU.DuctDevices(C["ov"], CU.car_devices(T63.geometry()))
    return _DEV["D"]


def _tier63_window_min() -> float:
    with open(TIER63_MARCH, encoding="utf-8") as fh:
        m = json.load(fh)
    t = np.asarray(m["probes"]["t"])
    u = np.asarray([p["u_mean"] for p in m["probes"]["turbine"]])
    w = (t >= WINDOW[0] - 1e-9) & (t <= WINDOW[1] + 1e-9)
    return float(u[w].min())


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------


def stage_controls(res) -> dict:
    C = T62._car()
    ov, solids = C["ov"], C["solids"]
    D = _devices()
    h = GE.DX
    out: dict = {"n_unknowns": ov.n_unknowns, "devices_build_s": D.build_s,
                 "band": CU.duct_band(T63.geometry())}
    # C1
    out["quadrature_identity"] = D.quadrature_identity()
    out["discrete_integrals"] = D.discrete_integrals()
    # C2
    X = np.empty(ov.n_unknowns)
    Y = np.empty(ov.n_unknowns)
    for g in ov.grids:
        idx = ov.index[g.name]
        live = idx >= 0
        gx, gy = (g.X, g.Y) if isinstance(g, OV.CartesianGrid) else (g.x, g.y)
        X[idx[live]] = gx[live]
        Y[idx[live]] = gy[live]
    smooth = np.sin(2.3 * X) * np.cos(1.7 * Y) + 0.3 * X
    lin = 0.7 * X - 1.3 * Y + 2.0
    c2 = {}
    for k, d in D.devices.items():
        x, y, _w = d.strip_points()
        c2[k] = {"against_probe": float(np.max(np.abs(D.Q_strip[k] @ smooth - CS.probe(ov, smooth, x, y)))),
                 "linear": float(np.max(np.abs(D.Q_strip[k] @ lin - (0.7 * x - 1.3 * y + 2.0)))),
                 "smooth_interpolation_error": float(np.max(np.abs(
                     D.Q_strip[k] @ smooth - (np.sin(2.3 * x) * np.cos(1.7 * y) + 0.3 * x))))}
        xr, yr, _w = d.ring_points()
        c2[k]["ring_against_probe"] = float(np.max(np.abs(D.Q_ring[k] @ smooth - CS.probe(ov, smooth, xr, yr))))
        c2[k]["ring_linear"] = float(np.max(np.abs(D.Q_ring[k] @ lin - (0.7 * xr - 1.3 * yr + 2.0))))
    out["interpolation"] = c2
    # C3
    stations = {}
    for k, d in D.devices.items():
        for label, xc in [("ring", d.x_ring / h), ("plane", d.x_plane / h)] + \
                [(f"line{off:+g}", d.x_plane / h + off) for off in LINES]:
            if k == "ROTOR" and xc > CU.duct_band(T63.geometry())["x_to"] - 1.0:
                continue
            ivs = CU.open_band(solids, xc, 0.0, 70.0)
            duct = [iv for iv in ivs if iv[0] < 34.0 < iv[1]]
            stations[f"{k}.{label}@{xc:g}"] = duct[0] if duct else None
    out["open_band"] = stations
    # C4
    w = D.devices["ROTOR"].width
    bands = {f"{u:g}": CU.machine_band(u, w) for u in (_tier63_window_min(), 0.08, 0.07)}
    out["machine_band"] = bands
    ref = RL.machine_for_host(RL.U_HOST_REF, scale=w)
    base = VM.machine_for_rotor(w)
    out["similarity_at_the_reference"] = all(
        getattr(ref[n], a) == getattr(base[n], a) for n in base for a in ("resistance",)) and all(
        getattr(ref["MGU"], a) == getattr(base["MGU"], a) for a in ("k_e", "k_t", "r_total"))
    # the reference induction: the porous column's machine_for_rotor at its own u_ref
    res_ref, _e, _d = VM.operating_point(np.full(CU.RING_SAMPLES, RL.U_HOST_REF), width=w, scale=w,
                                         elements=VM.machine_for_rotor(w))
    out["reference_induction"] = float(res_ref.induction)
    # C5
    ones = np.ones(ov.n_unknowns)
    out["uniform"] = {k: {"ring": float(np.max(np.abs(D.ring(ones, k) - 1.0))),
                          "plane": float(abs(D.plane_mean(ones, k) - 1.0))} for k in D.devices}
    say("controls", json.dumps(T62.T60.clean({kk: vv for kk, vv in out.items()
                                               if kk not in ("discrete_integrals", "machine_band")})))
    say("machine band", {u: (b["ratio_lo"], b["ratio_hi"], b["at_the_sizing_point"]["induction"])
                         for u, b in bands.items()})
    res["controls"] = out
    persist(res, NAME + ".json")
    return out


# ---------------------------------------------------------------------------
# the marches
# ---------------------------------------------------------------------------


def _owner(ov):
    owner = np.empty(ov.n_unknowns, dtype=object)
    for g in ov.grids:
        ix = ov.index[g.name]
        owner[ix[ix >= 0]] = g.name
    return owner


def _lines(flow, D) -> dict:
    return {k: {f"{off:+g}": D.line_mean(flow.P, k, off) for off in LINES
                if not (k == "ROTOR" and off > 8.0)} for k in D.devices}


def _jumps(lines: dict) -> dict:
    """The pressure jump across each device net of the adjacent stretch's friction:
    the core against the stretch downstream of it (the one upstream is within an
    entrance length of the duct's mouth), the turbine against the stretch upstream
    (the one downstream reaches the duct's exit)."""
    c, r = lines["RAD"], lines["ROTOR"]
    return {"RAD": (c["-8"] - c["+8"]) - (c["+8"] - c["+24"]),
            "ROTOR": (r["-8"] - r["+8"]) - (r["-24"] - r["-8"])}


def _run(flow, union, t_end: float, fname: str, extra: dict) -> dict:
    """March to ``t_end``, recording Tier 62's trajectory and probes, and the
    union's trace if there is one."""
    C = T62._car()
    ov, solids = C["ov"], C["solids"]
    D = _devices()
    owner = _owner(ov)
    out = dict(extra)
    out.update({"t_start": flow.t, "t_end": t_end, "dt": flow.dt, "n_unknowns": ov.n_unknowns,
                "machine_before": T62.T60.machine_state(), "started_at": dt.datetime.now().isoformat()})
    steps = int(round((t_end - flow.t) / flow.dt))
    tr = {k: [] for k in ("t", "fx", "fy", "max_div", "div_at", "u_max", "u_max_at", "iterations",
                          "step_s", "ilu_s", "ilu_note")}
    tr["by_solid"] = {s.name: {"fx": [], "fy": []} for s in solids}
    probes = {"t": [], "core": [], "turbine": [], "contour": [], "lines": [], "amplitude": []}
    t_start = time.perf_counter()
    for s in range(steps):
        try:
            if union is not None:
                rec = union.step()
                forces = union.last_forces
            else:
                rec = flow.step()
                forces = {sol.name: flow.forces(sol.name) for sol in solids}
        except Exception:
            out.update(trajectory=tr, probes=probes, steps_done=s, failed=traceback.format_exc()[-3000:])
            if union is not None:
                out["union"] = {"trace": union.trace, "coolant": union.coolant}
            persist(out, fname)
            raise
        div = np.abs(flow.Dx @ flow.U + flow.Dy @ flow.V)
        div[~flow.disc] = 0.0
        q = int(np.argmax(div))
        sp_ = np.hypot(flow.U, flow.V)
        r = int(np.argmax(sp_))
        fx = fy = 0.0
        for sol in solids:
            f = forces[sol.name]
            tr["by_solid"][sol.name]["fx"].append(f["fx"])
            tr["by_solid"][sol.name]["fy"].append(f["fy"])
            fx += f["fx"]
            fy += f["fy"]
        tr["t"].append(rec["t"])
        tr["fx"].append(fx)
        tr["fy"].append(fy)
        tr["max_div"].append(float(div[q]))
        tr["div_at"].append([owner[q], float(flow.X[q]), float(flow.Y[q])])
        tr["u_max"].append(float(sp_[r]))
        tr["u_max_at"].append([owner[r], float(flow.X[r]), float(flow.Y[r])])
        tr["iterations"].append(rec["iterations"])
        tr["step_s"].append(rec["step_s"])
        tr["ilu_s"].append(rec["ilu_s"])
        tr["ilu_note"].append(rec["ilu_note"])
        if s % PROBE_EVERY == 0 or s + 1 == steps:
            probes["t"].append(rec["t"])
            probes["core"].append(T62._duct_flux(flow, T62.CORE_X))
            probes["turbine"].append(T62._duct_flux(flow, T62.TURBINE_X))
            probes["contour"].append(T62._contour_flux(flow))
            probes["lines"].append(_lines(flow, D))
            probes["amplitude"].append(dict(D.amplitude))
        if (s + 1) % PERSIST_EVERY == 0 or s + 1 == steps:
            out.update(trajectory=tr, probes=probes, steps_done=s + 1, steps=steps,
                       wall_s=time.perf_counter() - t_start)
            if union is not None:
                out["union"] = {"trace": union.trace, "coolant": union.coolant,
                                "outside_steps": union.outside_steps, "outside_first": union.outside_first}
            persist(out, fname)
            extra_txt = ""
            if union is not None and union.trace["t"]:
                st = union.state
                extra_txt = " | u_rotor %.5f a %.4f I %.2e q %.3eW mgu %s rotor %s" % (
                    st.u_rotor, st.induction, st.current, st.q_machine, union.trace["mgu_valid"][-1],
                    union.trace["rotor_valid"][-1])
            say("%s step %d/%d t %.3f Fx %.4f Fy %.4f div %.2e u_max %.3f its %s %.2fs%s" % (
                fname, s + 1, steps, rec["t"], fx, fy, div[q], sp_[r], rec["iterations"], rec["step_s"],
                extra_txt))
    out["machine_after"] = T62.T60.machine_state()
    out["finished_at"] = dt.datetime.now().isoformat()
    persist(out, fname)
    return out


def stage_prefix(res) -> dict:
    C = T62._car()
    flow = CS.car_flow(C["ov"], C["solids"], precond="ilu")
    run = _run(flow, None, T_ON, "prefix.json", {"what": "the car from the stream, no devices"})
    CU.save_state(flow, prefix_path())
    D = _devices()
    floor = _jumps(_lines(flow, D))
    # against Tier 63's record
    cmp: dict = {}
    if os.path.isfile(TIER63_MARCH):
        with open(TIER63_MARCH, encoding="utf-8") as fh:
            m63 = json.load(fh)
        n = len(run["trajectory"]["t"])
        t63 = m63["trajectory"]
        for key in ("fx", "fy", "u_max", "max_div"):
            a = np.asarray(run["trajectory"][key])
            b = np.asarray(t63[key][:n])
            cmp[key] = {"max_relative": float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))),
                        "bitwise": bool(np.array_equal(a, b))}
        cmp["iterations_identical"] = run["trajectory"]["iterations"] == t63["iterations"][:n]
        # the probes taken at the same steps: every eighth, and not the prefix's
        # own last one, which Tier 63 (a longer march) did not take there
        at63 = {round(t, 9): i for i, t in enumerate(m63["probes"]["t"])}
        pairs = [(i, at63[round(t, 9)]) for i, t in enumerate(run["probes"]["t"]) if round(t, 9) in at63]
        cmp["probes_compared"] = len(pairs)
        cmp["probes_expected"] = len(range(0, n, PROBE_EVERY))
        for key, sub in (("core", "flux"), ("turbine", "flux"), ("contour", "net")):
            a = np.asarray([run["probes"][key][i][sub] for i, _j in pairs])
            b = np.asarray([m63["probes"][key][j][sub] for _i, j in pairs])
            cmp[f"{key}.{sub}"] = {"max_relative": float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))),
                                   "bitwise": bool(np.array_equal(a, b))}
        cmp["probe_times_identical"] = cmp["probes_compared"] == cmp["probes_expected"]
    out = {"file": "prefix.json", "state": os.path.relpath(prefix_path(), HERE), "t": flow.t,
           "steps": run["steps"], "wall_s": run["wall_s"],
           "median_step_s": float(np.median(run["trajectory"]["step_s"])),
           "machine_before": run["machine_before"], "machine_after": run["machine_after"],
           "against_tier63": cmp, "pressure_jump_formula_without_devices": floor,
           "turbine_u_mean_at_t_on": run["probes"]["turbine"][-1]["u_mean"]}
    say("prefix", json.dumps(T62.T60.clean(out)))
    res["prefix"] = out
    persist(res, NAME + ".json")
    return out


def _window(t, a: float | None = None, b: float | None = None):
    """The registered window's mask.  The bounds are read at CALL time, so a
    caller that moves `WINDOW` moves every stage with it."""
    a = WINDOW[0] if a is None else a
    b = WINDOW[1] if b is None else b
    t = np.asarray(t)
    return (t >= a - 1e-9) & (t <= b + 1e-9)


def _arm_summary(run: dict, union: CU.CarUnion) -> dict:
    tr = run["trajectory"]
    t = np.asarray(tr["t"])
    win = _window(t)
    ut = union.trace
    tu = np.asarray(ut["t"])
    wu = _window(tu)
    S: dict = {"u_host": union.u_host, "steps": run["steps"], "steps_done": run["steps_done"],
               "wall_s": run["wall_s"], "median_step_s": float(np.median(tr["step_s"])),
               "median_union_s": float(np.median(ut["union_s"])),
               "iterations_max": int(max(max(i) for i in tr["iterations"])),
               "u_max_max": float(max(tr["u_max"])),
               "max_div_window": float(np.max(np.asarray(tr["max_div"])[win])),
               "fx_mean_window": float(np.mean(np.asarray(tr["fx"])[win])),
               "fy_mean_window": float(np.mean(np.asarray(tr["fy"])[win])),
               "fx_spread_window": float(np.ptp(np.asarray(tr["fx"])[win]))}
    pt = np.asarray(run["probes"]["t"])
    pw = _window(pt)
    tur = [p for p, k in zip(run["probes"]["turbine"], pw) if k]
    cor = [p for p, k in zip(run["probes"]["core"], pw) if k]
    S["turbine_u_mean_window"] = float(np.mean([p["u_mean"] for p in tur]))
    S["core_u_mean_window"] = float(np.mean([p["u_mean"] for p in cor]))
    S["duct_flux_mismatch_window"] = float(max(abs(p["flux"] - q["flux"]) / abs(q["flux"])
                                               for p, q in zip(tur, cor)))
    S["contour_relative_window"] = float(np.max(np.abs(
        [p["relative"] for p, k in zip(run["probes"]["contour"], pw) if k])))
    ur = np.asarray(ut["u_rotor"])
    S["u_rotor_min_window"] = float(ur[wu].min())
    S["u_rotor_max_window"] = float(ur[wu].max())
    S["u_rotor_min_horizon"] = float(ur.min())
    S["u_rotor_first"] = float(ur[0])
    S["ratio_min_window"] = S["u_rotor_min_window"] / union.u_host
    S["ratio_max_window"] = S["u_rotor_max_window"] / union.u_host
    declined = {}
    for key, name in (("mgu_valid", "MGU"), ("rotor_valid", "ROTOR"), ("rad_valid", "RAD"),
                      ("fluid_valid", "FLUID")):
        flags = np.asarray([v is False for v in ut[key]])
        declined[name] = {"window": int(np.sum(flags & wu)), "before_window": int(np.sum(flags & (tu < WINDOW[0] - 1e-9))),
                          "last_t": (float(tu[flags].max()) if flags.any() else None)}
    S["declined"] = declined
    S["admitted"] = bool(run["steps_done"] == run["steps"] and all(v["window"] == 0 for v in declined.values()))
    S["outside_first"] = union.outside_first
    S["envelope_at_the_end"] = union.envelope
    # the balances, by the porous column's own function
    m = union.union_march(*WINDOW)
    S["balances"] = VM.receiver_balances(m, frac=1.0)
    w_core = -float(np.mean(m.trace["power_core"]))
    p_shaft = float(np.mean(m.trace["shaft_power"]))
    S["J3_against_the_core_s_work"] = abs(w_core - p_shaft) / abs(p_shaft) if p_shaft else None
    ratio = -np.asarray(m.trace["power_rotor"]) / np.asarray(m.trace["shaft_power"])
    S["J3_ratio_per_step"] = {"mean": float(ratio.mean()), "min": float(ratio.min()), "max": float(ratio.max())}
    ua = np.asarray(m.trace["ua"])
    uc = np.asarray(m.trace["u_core"])
    uref = float(union._core.u_air_ref)
    pred = CL.UA_RAD * (uc / uref) ** IU.UA_EXPONENT
    S["J1_identity_max_relative"] = float(np.max(np.abs(ua - pred) / np.abs(ua)))
    mean_pow = float(np.mean((uc / uref) ** IU.UA_EXPONENT))
    pow_mean = float((np.mean(uc) / uref) ** IU.UA_EXPONENT)
    S["J1_jensen_gap"] = abs(mean_pow - pow_mean) / pow_mean
    S["J1_air_cv_window"] = float(np.std(uc) / np.mean(uc))
    S["q_machine_W_window_mean"] = float(np.mean(m.trace["q_machine"]))
    S["ua_window_mean"] = float(np.mean(ua))
    S["coolant"] = union.coolant
    S["induction_window"] = [float(np.min(np.asarray(ut["induction"])[wu])),
                             float(np.max(np.asarray(ut["induction"])[wu]))]
    # the pressure jumps
    lines_w = [p for p, k in zip(run["probes"]["lines"], pw) if k]
    amp_w = [p for p, k in zip(run["probes"]["amplitude"], pw) if k]
    width = union.width
    pj = {}
    for key in ("RAD", "ROTOR"):
        jumps = np.asarray([_jumps(ln)[key] for ln in lines_w])
        tw = np.asarray([a[key] for a in amp_w]) / width
        pj[key] = {"jump_mean": float(jumps.mean()), "T_over_W_mean": float(tw.mean()),
                   "ratio": float(jumps.mean() / tw.mean()) if tw.mean() else None}
    S["pressure_jump"] = pj
    S["union_cost_fraction"] = S["median_union_s"] / S["median_step_s"]
    return S


def stage_arms(res) -> dict:
    C = T62._car()
    ov, solids = C["ov"], C["solids"]
    D = _devices()
    if not os.path.isfile(prefix_path()):
        raise SystemExit("run the prefix first")
    arms = res.setdefault("arms", {"list": [], "k_max": K_MAX, "rule": (
        "W228, iterated: arm 1 sized for Tier 63's own minimum of the duct's flow over the window; "
        "arm k sized for arm k-1's minimum of u_rotor over the window; stop at the first arm "
        "inside every declared envelope at every step of the window")})
    arms["list"] = []
    u_host = _tier63_window_min()
    arms["first_sizing_from_tier63"] = u_host
    for k in range(1, K_MAX + 1):
        flow = CS.car_flow(ov, solids, precond="ilu")
        st = CU.load_state(flow, prefix_path())
        D.amplitude = {key: 0.0 for key in D.devices}
        union = CU.CarUnion(flow, D, u_host, solids=[s.name for s in solids], t_on=T_ON)
        say(f"arm {k}: sized for {u_host:.6f}, from the prefix at t = {st['t']:.4f}")
        run = _run(flow, union, T_END, f"arm{k}.json", {"arm": k, "u_host": u_host})
        S = _arm_summary(run, union)
        S["file"] = f"arm{k}.json"
        S["machine_before"] = run["machine_before"]
        S["machine_after"] = run["machine_after"]
        arms["list"].append(S)
        persist(res, NAME + ".json")
        say(f"arm {k}", json.dumps(T62.T60.clean({kk: vv for kk, vv in S.items()
                                                  if kk not in ("coolant", "envelope_at_the_end", "balances")})))
        # the flow's incomplete and complete factors are hundreds of megabytes;
        # let them go before the next arm builds its own
        del union, flow, run
        gc.collect()
        if S["admitted"]:
            arms["registered"] = k
            break
        u_host = S["u_rotor_min_window"]
    else:
        arms["registered"] = None
    persist(res, NAME + ".json")
    return arms


def stage_null(res) -> dict:
    """The same window with NO devices, released from the same state.

    **Added after the prefix**, and it judges nothing: the prefix measured the
    pressure-jump formula's own floor at $t = 8$ -- the difference between two
    16-cell stretches of duct with no device in either -- as -0.0067 at the
    core's station, which is the size of the jump the core is expected to make.
    A stretch of duct that is not uniform cannot be its own friction reference,
    so this arm gives the floor AT THE WINDOW, at the same times as the arms,
    and with it the devices' own jump by difference.  It also prices the restart:
    with no devices this march is Tier 63's, and what it does not reproduce
    bitwise is what releasing from a kept state costs (the incomplete factor is
    refreshed on a different schedule).
    """
    C = T62._car()
    flow = CS.car_flow(C["ov"], C["solids"], precond="ilu")
    st = CU.load_state(flow, prefix_path())
    say(f"the device-free arm, from the prefix at t = {st['t']:.4f}")
    run = _run(flow, None, T_END, "null_arm.json", {"what": "the same window, no devices"})
    tr = run["trajectory"]
    t = np.asarray(tr["t"])
    win = _window(t)
    pt = np.asarray(run["probes"]["t"])
    pw = _window(pt)
    lines_w = [p for p, k in zip(run["probes"]["lines"], pw) if k]
    out = {"file": "null_arm.json", "wall_s": run["wall_s"],
           "median_step_s": float(np.median(tr["step_s"])),
           "turbine_u_mean_window": float(np.mean([p["u_mean"] for p, k in zip(run["probes"]["turbine"], pw) if k])),
           "fx_mean_window": float(np.mean(np.asarray(tr["fx"])[win])),
           "fy_mean_window": float(np.mean(np.asarray(tr["fy"])[win])),
           "u_rotor_like_min_window": float(np.min([p["u_mean"] for p, k in zip(run["probes"]["turbine"], pw) if k])),
           "pressure_jump_floor_window": {k: float(np.mean([_jumps(ln)[k] for ln in lines_w]))
                                          for k in ("RAD", "ROTOR")},
           "machine_before": run["machine_before"], "machine_after": run["machine_after"]}
    # what the restart costs, against Tier 63's own record at the same steps
    if os.path.isfile(TIER63_MARCH):
        with open(TIER63_MARCH, encoding="utf-8") as fh:
            m63 = json.load(fh)
        n0 = int(round(T_ON / flow.dt))
        n = len(tr["t"])
        cmp: dict = {}
        for key in ("fx", "fy", "u_max"):
            a = np.asarray(tr[key])
            b = np.asarray(m63["trajectory"][key][n0:n0 + n])
            cmp[key] = {"max_relative": float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))),
                        "bitwise": bool(np.array_equal(a, b))}
        at63 = {round(v, 9): i for i, v in enumerate(m63["probes"]["t"])}
        pairs = [(i, at63[round(v, 9)]) for i, v in enumerate(run["probes"]["t"]) if round(v, 9) in at63]
        a = np.asarray([run["probes"]["turbine"][i]["u_mean"] for i, _j in pairs])
        b = np.asarray([m63["probes"]["turbine"][j]["u_mean"] for _i, j in pairs])
        cmp["turbine.u_mean"] = {"max_relative": float(np.max(np.abs(a - b) / np.abs(b))),
                                 "bitwise": bool(np.array_equal(a, b)), "compared": len(pairs)}
        out["against_tier63_after_the_restart"] = cmp
    res["null_arm"] = out
    persist(res, NAME + ".json")
    say("null arm", json.dumps(T62.T60.clean(out)))
    return out


def stage_ring(res) -> dict:
    """What a ring actually reads, measured on the prefix's own state.

    **Added after arm 1**, and it judges nothing.  U5 registered that the
    radiator's conductance follows ``UA_RAD (u / u_ref)^0.8`` in the air the
    trace reports, and the two disagreed by more than a percent.  They are not
    the same average: `integration_union.CoreRadiator.ua` reads ``mean|u|`` over
    its 32 cells and the trace's ``u_core`` is the SIGNED mean, so any reversed
    flow in the ring separates them.  This reads the ring point by point at
    ``t = 8``, and turns the recorded ``UA`` back into ``mean|u|`` over the whole
    of the registered arm, so the size of the reversal is a measurement rather
    than an inference.
    """
    C = T62._car()
    D = _devices()
    flow = CS.car_flow(C["ov"], C["solids"], precond="ilu")
    st = CU.load_state(flow, prefix_path())
    h = GE.DX
    S = _registered(res)
    u_host = S["u_host"] if S else _tier63_window_min()
    union = CU.CarUnion(flow, D, u_host, t_on=0.0)
    union.refresh(flow.U)                      # the registered arm's own first refresh
    out: dict = {"at_t": st["t"], "u_air_ref": float(union._core.u_air_ref),
                 "u_host": u_host, "note": "the state every arm releases from, no device applied yet"}
    for k in ("RAD", "ROTOR"):
        x, y, _w = D.devices[k].ring_points()
        u = D.ring(flow.U, k)
        v = D.Q_ring[k] @ flow.V
        out[k] = {"y_cells": [round(float(q) / h, 3) for q in y],
                  "u": [float(q) for q in u], "v_max_abs": float(np.max(np.abs(v))),
                  "mean": float(np.mean(u)), "abs_mean": float(np.mean(np.abs(u))),
                  "min": float(np.min(u)), "max": float(np.max(u)),
                  "negative_samples": int(np.sum(u < 0.0)),
                  "abs_over_signed": float(np.mean(np.abs(u)) / np.mean(u))}
    # the reversal over the registered arm, read back out of UA
    if S:
        with open(os.path.join(OUT, S["file"]), encoding="utf-8") as fh:
            arm = json.load(fh)
        tr = arm["union"]["trace"]
        t = np.asarray(tr["t"])
        ua = np.asarray(tr["ua"])
        uc = np.asarray(tr["u_core"])
        absmean = out["u_air_ref"] * (ua / CL.UA_RAD) ** (1.0 / IU.UA_EXPONENT)
        gap = absmean / uc - 1.0
        win = _window(t)
        out["reversal_over_the_registered_arm"] = {
            "abs_over_signed_first": float(absmean[0] / uc[0]),
            "abs_over_signed_last": float(absmean[-1] / uc[-1]),
            "abs_over_signed_window_mean": float(np.mean(absmean[win] / uc[win])),
            "gap_grows": bool(gap[-1] > gap[0]),
            "at": [[float(t[i]), float(gap[i])] for i in range(0, t.size, 80)],
        }
    res["ring"] = out
    persist(res, NAME + ".json")
    say("ring", json.dumps(T62.T60.clean({k: (v if k not in ("RAD", "ROTOR") else
                                              {kk: vv for kk, vv in v.items() if kk not in ("u", "y_cells")})
                                          for k, v in out.items()})))
    return out


def _registered(res) -> dict | None:
    A = res.get("arms")
    if not A or not A.get("list"):
        return None
    k = A.get("registered")
    return A["list"][k - 1] if k else A["list"][-1]


def stage_slow(res) -> dict:
    import tier53_racelab_rerun as T53
    S = _registered(res)
    if S is None:
        raise SystemExit("run the arms first")
    q, ua = S["q_machine_W_window_mean"], S["ua_window_mean"]
    out = T53.stage_slow({"arms": {"arms": {"referent": {"settled": {"q_machine": q, "ua": ua}}}}})
    out["from_the_arm"] = res["arms"].get("registered")
    out["recipe"] = "scripts/tier53_racelab_rerun.py::stage_slow, imported"
    res["slow"] = out
    persist(res, NAME + ".json")
    say("slow", json.dumps(T62.T60.clean(out)))
    return out


def stage_compare(res) -> dict:
    S = _registered(res)
    out: dict = {"registered_arm": res.get("arms", {}).get("registered")}
    if os.path.isfile(TIER63_RECORD):
        with open(TIER63_RECORD, encoding="utf-8") as fh:
            r63 = json.load(fh)
        M = r63["march"]["summary"]
        out["tier63"] = {k: M[k] for k in ("turbine_u_mean_window", "core_u_mean_window", "fx_mean_window",
                                           "fy_mean_window", "u_max_max", "max_div_window",
                                           "contour_relative_window", "median_step_s")}
    if os.path.isfile(POROUS_RECORD):
        with open(POROUS_RECORD, encoding="utf-8") as fh:
            r8 = json.load(fh)
        ref = r8["arms"]["arms"]["referent"]
        out["porous_referent"] = {
            "J3_residual": ref["balances"]["J3"]["residual_with_the_term"],
            "J3_u_plane_over_u_ring": ref["balances"]["J3"]["u_plane / u_ring"],
            "J1_tracking_residual": ref["balances"]["J1"]["tracking_residual"],
            "J2_first_law": ref["balances"]["J2"]["block first law, relative"],
            "q_machine_W": ref["settled"]["q_machine"], "induction": ref["settled"]["induction"],
            "host_inflow": r8["arms"]["host_inflow"], "u_rotor": ref["settled"]["u_rotor"]}
    if S and "tier63" in out:
        T = out["tier63"]
        out["duct_slowed_by"] = 1.0 - S["turbine_u_mean_window"] / T["turbine_u_mean_window"]
        out["drag_change"] = S["fx_mean_window"] / T["fx_mean_window"] - 1.0
        out["vertical_force_change"] = S["fy_mean_window"] / T["fy_mean_window"] - 1.0
    if S and "prefix" in res:
        fl = res["prefix"]["pressure_jump_formula_without_devices"]
        out["pressure_jump_floor_at_t_on_over_T_over_W"] = {
            k: (fl[k] / S["pressure_jump"][k]["T_over_W_mean"]) if S["pressure_jump"][k]["T_over_W_mean"] else None
            for k in ("RAD", "ROTOR")}
    if S and "null_arm" in res:
        N = res["null_arm"]
        out["against_the_device_free_arm"] = {
            "turbine_u_mean": N["turbine_u_mean_window"],
            "duct_slowed_by": 1.0 - S["turbine_u_mean_window"] / N["turbine_u_mean_window"],
            "drag_change": S["fx_mean_window"] / N["fx_mean_window"] - 1.0,
            "vertical_force_change": S["fy_mean_window"] / N["fy_mean_window"] - 1.0,
            "restart_cost": N.get("against_tier63_after_the_restart"),
        }
        #: the registered clause U7 charges the whole formula to the devices; this
        #: is the same formula with the device-free arm's value at the SAME times
        #: subtracted, reported beside it and judging nothing
        out["pressure_jump_less_the_device_free_arm"] = {
            k: {"jump_less_floor": S["pressure_jump"][k]["jump_mean"] - N["pressure_jump_floor_window"][k],
                "T_over_W": S["pressure_jump"][k]["T_over_W_mean"],
                "ratio": ((S["pressure_jump"][k]["jump_mean"] - N["pressure_jump_floor_window"][k])
                          / S["pressure_jump"][k]["T_over_W_mean"]) if S["pressure_jump"][k]["T_over_W_mean"] else None}
            for k in ("RAD", "ROTOR")}
    res["compare"] = out
    persist(res, NAME + ".json")
    say("compare", json.dumps(T62.T60.clean(out)))
    return out


# ---------------------------------------------------------------------------
# the verdicts
# ---------------------------------------------------------------------------


def judge(res) -> dict:
    v: dict = {}
    K = res.get("controls")
    if K:
        v["C1"] = (all(abs(x) <= 1e-12 for x in K["quadrature_identity"].values())
                   and all(abs(d["background_all_points"] - 1.0) <= 1e-12 for d in K["discrete_integrals"].values()))
        v["C2"] = all(c["against_probe"] <= 1e-13 and c["ring_against_probe"] <= 1e-13
                      and c["linear"] <= 1e-12 and c["ring_linear"] <= 1e-12 for c in K["interpolation"].values())
        v["C3"] = (len(K["open_band"]) >= 9 and all(
            iv is not None and abs(iv[0] - 19.5) <= 1e-9 and abs(iv[1] - 48.5) <= 1e-9
            for iv in K["open_band"].values()))
        bands = list(K["machine_band"].values())
        v["C4"] = (all(abs(b["ratio_lo"] - bands[0]["ratio_lo"]) <= 1e-6
                       and abs(b["ratio_hi"] - bands[0]["ratio_hi"]) <= 1e-6 for b in bands)
                   and all(abs(b["at_the_sizing_point"]["induction"] - K["reference_induction"]) <= 1e-9
                           for b in bands)
                   and bool(K["similarity_at_the_reference"]))
        v["C5"] = all(u["ring"] <= 1e-12 and u["plane"] <= 1e-12 for u in K["uniform"].values())
    P = res.get("prefix")
    if P and P.get("against_tier63"):
        c = P["against_tier63"]
        keys = ("fx", "fy", "u_max", "max_div", "core.flux", "turbine.flux", "contour.net")
        v["U0"] = all(k in c and c[k]["max_relative"] <= 1e-6 for k in keys) and bool(c.get("probe_times_identical"))
    A = res.get("arms")
    if A and A.get("list"):
        arms = A["list"]
        a1 = arms[0]
        v["U1"] = (not a1["admitted"]) and a1["declined"]["MGU"]["window"] > 0
        reg = A.get("registered")
        v["U2"] = reg is not None and reg <= 3
        S = _registered(res)
        if reg is None:
            # nothing admitted: the balance clauses are judged on no registered arm
            for pid in ("U3", "U4", "U5", "U6", "U7", "U8", "U9", "U10", "U11"):
                v[pid] = None
        else:
            d = S["declined"]
            v["U3"] = (all(x["window"] == 0 for x in d.values()) and d["MGU"]["before_window"] == 0
                       and d["RAD"]["before_window"] == 0 and d["FLUID"]["before_window"] == 0)
            B = S["balances"]
            v["U4"] = (B["J3"]["residual_with_the_term"] is not None and B["J3"]["residual_with_the_term"] <= 0.005
                       and S["J3_against_the_core_s_work"] is not None and S["J3_against_the_core_s_work"] >= 0.5)
            tr = B["J1"]["tracking_residual"]
            v["U5"] = (S["J1_identity_max_relative"] <= 1e-12 and tr is not None and tr > 1e-6
                       and abs(tr - S["J1_jensen_gap"]) <= 1e-12)
            cool = S["coolant"]
            first_law_ok = bool(cool) and all(r["block_first_law"]["relative"] <= 1e-6 for r in cool)
            sl = res.get("slow")
            slow_ok = (None if sl is None else sl.get("P4_verdict") == "pass")
            v["U6"] = (None if slow_ok is None else
                       (first_law_ok and slow_ok and S["q_machine_W_window_mean"] < 1.265))
            pj = S["pressure_jump"]
            v["U7"] = (pj["RAD"]["ratio"] is not None and abs(pj["RAD"]["ratio"] - 1.0) <= 0.20
                       and pj["ROTOR"]["ratio"] is not None and abs(pj["ROTOR"]["ratio"] - 1.0) <= 0.40)
            Cm = res.get("compare", {})
            if "duct_slowed_by" in Cm:
                v["U8"] = 0.03 <= Cm["duct_slowed_by"] <= 0.20
                v["U10"] = abs(Cm["drag_change"]) <= 0.02 and abs(Cm["vertical_force_change"]) <= 0.02
            v["U9"] = (S["steps_done"] == S["steps"] and S["u_max_max"] <= 4.0 and S["max_div_window"] <= 0.2
                       and S["contour_relative_window"] <= 0.01 and S["duct_flux_mismatch_window"] <= 0.02
                       and S["iterations_max"] <= 60)
            v["U11"] = S["union_cost_fraction"] <= 0.05
    return v


def stage_summary(res) -> dict:
    verdicts = judge(res)
    res["verdicts"] = verdicts
    res["verdicts_missing"] = sorted({p["id"] for p in PREDICTION} - set(verdicts))
    persist(res, NAME + ".json")
    for p in PREDICTION:
        say("%-3s %-6s %s" % (p["id"], verdicts.get(p["id"]), p["claim"][:100]))
    return verdicts


STAGES = ("controls", "prefix", "arms", "null", "ring", "slow", "compare", "summary")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="controls,summary")
    args = ap.parse_args(argv)
    configure(args.out)
    res = load(NAME + ".json")
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if "prediction" not in res:
        res.update(tier=64, prediction=list(PREDICTION),
                   prediction_recorded_at=dt.datetime.now().isoformat(timespec="seconds"),
                   read_before_this_run=list(READ_BEFORE_THIS_RUN),
                   settings={"t_on": T_ON, "t_end": T_END, "window": list(WINDOW), "k_max": K_MAX,
                             "ring_upstream_cells": CU.RING_UPSTREAM_CELLS,
                             "half_width_cells": CU.HALF_WIDTH_CELLS, "quad_per_cell": CU.QUAD_PER_CELL,
                             "device_x_cells": CU.DEVICE_X_CELLS, "lines": list(LINES),
                             "n_per_coolant": VM.N_FLUID_PER_COOLANT},
                   rule=CS.solids_rule(T63.geometry()))
        persist(res, NAME + ".json")
    elif [p["id"] for p in res["prediction"]] != [p["id"] for p in PREDICTION]:
        raise SystemExit("the record's prediction differs from this file's")
    for st in stages:
        if st not in STAGES:
            raise SystemExit(f"unknown stage {st!r}; stages are {STAGES}")
        t0 = time.perf_counter()
        try:
            {"controls": stage_controls, "prefix": stage_prefix, "arms": stage_arms, "null": stage_null,
             "ring": stage_ring, "slow": stage_slow, "compare": stage_compare,
             "summary": stage_summary}[st](res)
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
