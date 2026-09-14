"""Tier 51 -- PoC 3 RaceLab, phase 1: the car's graph, marched (CS-19).

`POC3-RACELAB-REQUIREMENTS.md` section 9's phase 1: the car's geometry, a
static window decomposition DERIVED from it, the assembled `CaseGraph`, and a
headless march of the whole thing at the declared vehicle scale.  Nothing here
is a server, a dashboard, an expert switch or a third dimension.

Stages, each persisted to ``out/racelab/racelab.json`` as it finishes:

  ``size``      section 10's first named risk, answered before anything was
                built on top of it: a macro-step timed at eight candidate
                boxes with the FRONT-WING CASE AS A CONTROL IN THE SAME
                PROCESS, so what is reported is a ratio and not a wall-clock
                number that rots.  Plus the thread-count control and the
                sub-step sensitivity, because `WindowNS` sizes its sub-step
                count from ``u_max``.
  ``geometry``  the car: every body's release-state force, the wheel's
                coefficient against the cylinder drag it was DERIVED to
                reproduce, the discrete body-force conservation identity on
                this lattice, and the wheel-rotation measurement -- whose
                answer is that a normal-only closure cannot see a rotating
                surface in the CONTINUUM, and that what a twelve-sided
                polygon leaves behind is 17% of the free stream, which is
                why the declared car's wheels do not roll.
  ``windows``   the decomposition: the layout, the two profiles that must
                agree, the window-count frontier, and what requiring the
                front wing to be clear of every cut actually costs.
  ``compile``   the graph's verdicts seam by seam, with the DISJOINT union as
                the control that says which of them the joins caused.
  ``calib``     the instrument, run BEFORE the gate below was fixed: a short
                march, its bitwise repeat, the settling report and the cost.
  ``march``     the arms at the fixed horizon -- the tight referent, its
                repeat, all-lagged, and one null per join -- and each join's
                receiver balance read by `vehicle_march.receiver_balances`,
                unchanged, so nothing can drift from CS-18's own numbers.
  ``slow``      J2's receiver on the coolant circuit's OWN clock, with the
                mount term and without it, as CS-18 section 7.3 measured it.

Nothing is downloaded, no checkpoint is loaded, no machine is rented, and
NeuberNet is not loaded.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import io                                                               # noqa: E402
import json                                                             # noqa: E402
import sys                                                              # noqa: E402
import time                                                             # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np                                                      # noqa: E402
import torch                                                            # noqa: E402

#: **One thread, measured rather than inherited.**  The vault's note that one
#: thread beats eight was taken on small tensors and a 672x240 field is not
#: small, so stage ``size`` re-measures it; the answer is 4.25x and it is the
#: largest single factor in this tier's cost.
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert torch.backends.cuda.matmul.allow_tf32 is False
assert torch.backends.cudnn.allow_tf32 is False

from atlas import compiler as C                                         # noqa: E402
from atlas.cases import cooling_loop as CL                              # noqa: E402
from atlas.cases import ground_effect as GE                             # noqa: E402
from atlas.cases import integration_union as IU                        # noqa: E402
from atlas.cases import racelab as RL                                   # noqa: E402
from atlas.cases import vehicle_march as VM                             # noqa: E402
from atlas.cases import wake_array as WA                               # noqa: E402
from atlas.cases import wing_fsi as W                                   # noqa: E402

OUT = os.path.join(HERE, "out", "racelab")
CACHE = os.path.join(OUT, "cache")
#: W258: this record was measured on the column whose outlet is pinned, and
#: `racelab.OUTFLOW` became the repaired one later (Tier 59); every march here
#: names the column its record describes.
OUTFLOW = "pinned"

#: **The horizon, chosen from stage ``settle``'s curve.**  600 fluid
#: macro-steps is 7.5 of the tiling's time units, at the declared vehicle
#: scale 0.075 s, and **0.71 transits of the 10.5-unit domain** against the
#: 4.6 transits CS-18 marched.  The settle window is the last quarter, 150
#: macro-steps.
#:
#: **It is short and the page says so.**  A tight arm is 1.7 s a macro-step
#: here -- the box is 3.7x the cells, the car is thirteen bodies where the
#: front wing was one, and the tight column is five solver calls an
#: exchange -- so four tight arms at 600 steps is about an hour and at
#: CS-18's 4.6 transits would be nine.  Stage ``settle`` measures what the
#: longer horizon would have bought: the fluid band falls from 19% at 400
#: steps to 5.8% at 1600, so it buys a factor of three and not an order.
HORIZON = 600
SETTLE_FRAC = 0.25
CALIB_STEPS = 40

#: The ceiling section 10's first risk is written against.
MACRO_STEP_CEILING_S = 0.5


# ---------------------------------------------------------------------------
# persistence -- OneDrive holds a just-written file, so os.replace can raise
# ---------------------------------------------------------------------------


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
        return (x.tolist() if x.size <= 64
                else {"shape": list(x.shape), "first": float(x.flat[0]),
                      "last": float(x.flat[-1]),
                      "norm": float(np.linalg.norm(x))})
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if x is None or isinstance(x, (str, int, float)):
        return x
    return str(x)


def persist(res):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "racelab.json")
    tmp = path + ".tmp"

    def write():
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(clean(res), fh, indent=1)

    _retry(write)
    _retry(lambda: os.replace(tmp, path))
    return path


def load_res():
    path = os.path.join(OUT, "racelab.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_field(name: str, **arrays):
    """`np.savez` appends `.npz`, so the temp name must ALREADY end in it."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name + ".npz")
    tmp = os.path.join(CACHE, name + ".tmp.npz")
    _retry(lambda: np.savez(tmp[:-4], **arrays))
    _retry(lambda: os.replace(tmp, path))
    return path


def load_field(name: str):
    path = os.path.join(CACHE, name + ".npz")
    return np.load(path) if os.path.isfile(path) else None


def machine_state() -> dict:
    """What else was running, and on what power, when a timing was taken.

    The vault has paid for this twice: a milliseconds-per-step table that was
    wrong by four the next time anyone measured it because the box was in a
    different power state, and a `ps` incantation that missed detached Windows
    processes.  This tier paid for it a third time -- the same configuration
    timed in three processes gave three different fastest thread counts, and
    the machine moved from battery to mains between them.  So the state is
    recorded WITH every timing rather than remembered afterwards.
    """
    import subprocess
    out: dict = {}
    try:
        r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe",
                            "/FO", "CSV"], capture_output=True, text=True,
                           timeout=30)
        rows = [ln for ln in r.stdout.splitlines()[1:] if ln.strip()]
        out["python_processes_running"] = len(rows)
        out["python_processes"] = rows[:8]
    except Exception as exc:                                 # pragma: no cover
        out["python_processes_running"] = None
        out["tasklist_failed"] = str(exc)
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Battery).BatteryStatus"],
            capture_output=True, text=True, timeout=30)
        st = r.stdout.strip()
        out["battery_status_raw"] = st
        out["on_mains"] = (st == "2")
    except Exception as exc:                                 # pragma: no cover
        out["on_mains"] = None
        out["battery_query_failed"] = str(exc)
    return out


# ---------------------------------------------------------------------------
# stage 1 -- the size, measured before anything is built on it
# ---------------------------------------------------------------------------


def _time_layout(nx, ny, wx, wy, offsets, steps=5, warm=2, nu=GE.NU):
    """One macro-step of cut / step / blend / project / band, timed.

    The composition layer alone, with a plausible body force walked through the
    force path: this is what a box costs before any car is in it, which is the
    number the size decision turns on.  `RaceRollout` adds the car on top and
    stage ``calib`` measures THAT.
    """
    from atlas.cases.wind_farm_design import TORCH_DTYPE, _place
    opt = dict(dtype=TORCH_DTYPE, device="cpu")
    t = RL.RaceTiling.of(sorted({o[0] for o in offsets}),
                         sorted({o[1] for o in offsets}),
                         nx=nx, ny=ny, wx=wx, wy=wy)
    solver = GE.solver_for(wx, wy, nu, "cpu")
    ws = t.weights()
    chi = np.empty((t.n_windows, wy, wx))
    for k, (ox, oy) in enumerate(t.offsets):
        chi[k] = ws[k][oy:oy + wy, ox:ox + wx]
    chi_t = torch.as_tensor(chi, **opt)
    kx, ky, k2 = GE._wavenumbers(ny, nx + GE.PAD_CELLS)
    kx_t, ky_t, k2_t = (torch.as_tensor(kx, **opt), torch.as_tensor(ky, **opt),
                        torch.as_tensor(k2, **opt))
    tp = torch.as_tensor(GE._taper()[None, :], **opt)
    m = np.zeros((ny, nx), dtype=bool)
    m[:, :GE.BAND] = True
    m[-GE.BAND:, :] = True
    m[0, :] = True
    band = torch.as_tensor(m, device="cpu")
    holes = int(np.sum(np.sum(ws, axis=0) <= 0.0))
    u = torch.full((ny, nx), GE.U_INF, **opt)
    v = torch.zeros((ny, nx), **opt)
    fx = torch.zeros((ny, nx), **opt)
    fy = torch.zeros((ny, nx), **opt)
    fx[ny // 3:ny // 3 + 3, nx // 4:nx // 4 + 30] = -0.5
    subs = []

    def macro(u, v):
        for _ in range(GE.EXCHANGES):
            us = torch.stack([u[oy:oy + wy, ox:ox + wx] for ox, oy in t.offsets])
            vs = torch.stack([v[oy:oy + wy, ox:ox + wx] for ox, oy in t.offsets])
            fxs = torch.stack([fx[oy:oy + wy, ox:ox + wx] for ox, oy in t.offsets])
            fys = torch.stack([fy[oy:oy + wy, ox:ox + wx] for ox, oy in t.offsets])
            u1, v1 = solver.step_batch(us, vs, GE.MACRO_DT / GE.EXCHANGES,
                                       bc0=None, force=(fxs, fys))
            subs.append(int(solver.last_substeps))
            cu, cv = chi_t * u1, chi_t * v1
            au = _place([cu[i] for i in range(t.n_windows)], t.offsets, ny, nx)
            av = _place([cv[i] for i in range(t.n_windows)], t.offsets, ny, nx)
            uf, vf = au - GE.U_INF, av
            bu = torch.cat((uf, uf[:, -1:] * tp), dim=1)
            bv = torch.cat((vf, vf[:, -1:] * tp), dim=1)
            uh, vh = torch.fft.fft2(bu), torch.fft.fft2(bv)
            div = kx_t * uh + ky_t * vh
            uh, vh = uh - kx_t * div / k2_t, vh - ky_t * div / k2_t
            au = GE.U_INF + torch.fft.ifft2(uh).real[:, :nx]
            av = torch.fft.ifft2(vh).real[:, :nx]
            z = torch.zeros((), dtype=au.dtype)
            u = torch.where(band, z + GE.U_INF, au)
            v = torch.where(band, z, av)
        return u, v

    for _ in range(warm):
        u, v = macro(u, v)
    t0 = time.perf_counter()
    for _ in range(steps):
        u, v = macro(u, v)
    dt = (time.perf_counter() - t0) / steps
    return {"nx": nx, "ny": ny, "wx": wx, "wy": wy, "n_windows": t.n_windows,
            "min_overlap": min(t.overlaps()), "holes": holes,
            "overlap_ratio": t.n_windows * wx * wy / (nx * ny),
            "s_per_macro_step": dt, "substeps_max": max(subs),
            "metres": [nx * RL.DX * 0.50, ny * RL.DX * 0.50]}


def _grid(nx, ny, wx, wy, halo=16):
    sx, sy = wx - halo, wy - halo
    ox = list(range(0, nx - wx + 1, sx))
    oy = list(range(0, ny - wy + 1, sy))
    if ox[-1] + wx < nx:
        ox.append(nx - wx)
    if oy[-1] + wy < ny:
        oy.append(ny - wy)
    return [(a, b) for b in oy for a in ox]


def stage_size() -> dict:
    """Risk 1: is the domain too slow to be interactive?  Measured, with a
    control in the same process and the answer quoted as a ratio."""
    cands = [
        ("control-frontwing-208x144-w80", 208, 144, 80, 80, 16),
        ("requirements-384x192-w80", 384, 192, 80, 80, 16),
        ("384x192-w128x96", 384, 192, 128, 96, 16),
        ("p128-464x240", 464, 240, 128, 128, 16),
        ("p128-576x240", 576, 240, 128, 128, 16),
        ("CHOSEN-672x240-derived-layout", None, None, None, None, None),
        ("p128-800x240", 800, 240, 128, 128, 16),
        ("full-length-car-1024x240", 1024, 240, 128, 128, 16),
    ]
    t_layout, _i = RL.layout()
    rows = []
    for nm, nx, ny, wx, wy, halo in cands:
        if nx is None:
            r = _time_layout(RL.RNX, RL.RNY, RL.WX, RL.WY, t_layout.offsets)
        else:
            r = _time_layout(nx, ny, wx, wy, _grid(nx, ny, wx, wy, halo))
        r["name"] = nm
        rows.append(r)
        print("   %-32s %2dw %4dx%-4d %.4f s" % (nm, r["n_windows"], r["nx"],
                                                 r["ny"], r["s_per_macro_step"]),
              flush=True)
    base = rows[0]["s_per_macro_step"]
    for r in rows:
        r["ratio_to_the_front_wing_control"] = r["s_per_macro_step"] / base
    # the control re-timed at the end: drift over the run, visible
    again = _time_layout(208, 144, 80, 80, _grid(208, 144, 80, 80))
    # threads: the same case at 1, 4 and 8
    threads = []
    for n in (1, 8, 4):
        torch.set_num_threads(n)
        r = _time_layout(RL.RNX, RL.RNY, RL.WX, RL.WY, t_layout.offsets,
                         steps=4, warm=1)
        r["threads"] = n
        threads.append(r)
    torch.set_num_threads(1)
    best = min(threads, key=lambda r: r["s_per_macro_step"])
    worst = max(threads, key=lambda r: r["s_per_macro_step"])
    chosen = next(r for r in rows if r["name"].startswith("CHOSEN"))
    return {
        "candidates": rows,
        "machine_state": machine_state(),
        "control_s_per_macro_step": base,
        "control_retimed_at_the_end": again["s_per_macro_step"],
        "control_drift_over_the_run": again["s_per_macro_step"] / base,
        "threads": threads,
        "threads_chosen": best["threads"],
        "threads_worst_over_best": (worst["s_per_macro_step"]
                                    / best["s_per_macro_step"]),
        "chosen": chosen,
        "ceiling_s": MACRO_STEP_CEILING_S,
        "headroom_under_the_ceiling": MACRO_STEP_CEILING_S / chosen["s_per_macro_step"],
        "note": "every figure is a ratio against the front-wing control timed in "
                "THIS process; the absolute seconds are this laptop on this day "
                "and are recorded only so the ratio can be reconstructed",
    }


# ---------------------------------------------------------------------------
# stage 2 -- the car
# ---------------------------------------------------------------------------


def stage_geometry() -> dict:
    objs, flat = RL.car_bodies()
    u = torch.full((RL.RNY, RL.RNX), GE.U_INF, dtype=W.TORCH_DTYPE)
    v = torch.zeros((RL.RNY, RL.RNX), dtype=W.TORCH_DTYPE)
    per = []
    tot_fx = torch.zeros_like(u)
    tot_fy = torch.zeros_like(u)
    for b in objs:
        fx, fy, _w, load, drag = b.forcing(u, v, RL.RNY, RL.RNX)
        tot_fx, tot_fy = tot_fx + fx, tot_fy + fy
        per.append({
            "body": b.body_id,
            "kind": "wheel" if isinstance(b, RL.WheelBody) else "plate",
            "x_span": list(b.x_span()) if isinstance(b, RL.WheelBody)
                      else list(b.body.x_span()),
            "y_span": list(b.y_span()) if isinstance(b, RL.WheelBody)
                      else list(b.body.y_span()),
            "peak_abs_fx": float(fx.abs().max()),
            "downforce": float(load), "drag": float(drag),
        })

    # -- the wheel's coefficient against the cylinder drag it was derived for --
    wheel = next(b for b in objs if isinstance(b, RL.WheelBody))
    w_drag = next(p["drag"] for p in per if p["body"] == wheel.body_id)
    cyl = wheel.cylinder_drag()
    wheel_check = {
        "C_D_declared": RL.WHEEL_CD,
        "c_n_derived": RL.WHEEL_CN,
        "c_n_the_plates_use": GE.C_N,
        "measured_drag": w_drag,
        "cylinder_drag_0.5_C_D_U2_D": cyl,
        "relative_error": abs(w_drag - cyl) / cyl,
        "segments": wheel.n_seg,
        "what_the_uncalibrated_ring_gave":
            w_drag * GE.C_N / RL.WHEEL_CN,
        "factor_the_calibration_removed": GE.C_N / RL.WHEEL_CN,
    }

    # -- the discrete conservation identity, on THIS lattice -----------------
    #: `FlexWing.forcing` normalises each station's kernel on its own stamping
    #: box, so the force on the fluid integrates to minus the force on the body
    #: to floating point.  Asserted here per body, because a discretization
    #: losing 3% would read as a 3% interface residual at a join.
    ident = []
    for b in objs:
        fx, fy, _w, load, drag = b.forcing(u, v, RL.RNY, RL.RNX)
        got_x = float(fx.sum() * RL.DX * RL.DX)
        got_y = float(fy.sum() * RL.DX * RL.DX)
        ident.append({
            "body": b.body_id,
            "integral_fx": got_x, "minus_drag": -float(drag),
            "residual_x": abs(got_x + float(drag))
                          / max(abs(float(drag)), 1e-30),
            "integral_fy": got_y, "load": float(load),
            "residual_y": abs(got_y - float(load))
                          / max(abs(float(load)), 1e-30),
        })

    # -- the wheel's rotation: computed, then measured ----------------------
    #: The closure carries a NORMAL traction and a rolling surface's velocity
    #: is TANGENTIAL, so in the continuum ``u_s . n = 0`` and the rotation
    #: cannot enter any term.  `WheelBody` COMPUTES the projection rather than
    #: assuming it, so what is reported here is what the model does: the
    #: polygon's chords are radial only at their midpoints, and the residual is
    #: the discretisation's and not the physics'.
    t_layout, _i = RL.layout()
    wheel_r = next(b for b in objs if isinstance(b, RL.WheelBody))
    #: **The mechanism, isolated.**  The artifact is ``omega s`` at a
    #: station ``s`` from its chord's midpoint, so it must be EXACTLY zero
    #: at one station a chord -- the midpoint -- and grow with the station
    #: count.  Measured over four station counts, which is what separates a
    #: discretisation artifact from a physical effect.
    by_station = []
    for ns in (1, 2, 3, 4):
        wk = RL.WheelBody("PROBE", wheel_r.xc, wheel_r.yc, wheel_r.r,
                          n_seg=wheel_r.n_seg, n_station=ns,
                          omega=wheel_r.rolling_omega())
        by_station.append({"n_station": ns,
                           "max_surface_normal_velocity":
                               wk.max_surface_normal_velocity()})
    by_seg = []
    for nseg in (12, 24, 48):
        wk = RL.WheelBody("PROBE", wheel_r.xc, wheel_r.yc, wheel_r.r,
                          n_seg=nseg, n_station=3,
                          omega=wheel_r.rolling_omega())
        by_seg.append({"n_seg": nseg, "max_surface_normal_velocity":
                       wk.max_surface_normal_velocity()})
    fields = {}
    for tag, omega in (("rolling", None), ("static", 0.0)):
        r = RL.RaceRollout(tiling=t_layout, joins=("J1", "J2", "J3"),
                           join_coupling="lagged", wheel_omega=omega,
                           outflow=OUTFLOW)
        uu = torch.full((r.ny, r.nx), GE.U_INF, dtype=W.TORCH_DTYPE)
        vv = torch.zeros((r.ny, r.nx), dtype=W.TORCH_DTYPE)
        r.refresh(uu, 0)
        for s in range(8):
            uu, vv, _l, _d = r.macro_step(uu, vv, s)
        fields[tag] = (uu.detach().cpu().numpy().copy(),
                       vv.detach().cpu().numpy().copy())
    du = fields["rolling"][0] - fields["static"][0]
    dv = fields["rolling"][1] - fields["static"][1]
    wheel_r.omega = wheel_r.rolling_omega()
    art = wheel_r.max_surface_normal_velocity()
    rotation = {
        "omega_rolling": wheel_r.rolling_omega(),
        "omega_static": 0.0,
        "segments": wheel_r.n_seg,
        "max_surface_normal_velocity_on_the_polygon": art,
        "as_a_fraction_of_U_inf": art / GE.U_INF,
        "in_the_continuum_it_is": 0.0,
        "by_station_count": by_station,
        "by_segment_count": by_seg,
        "machine_zero_at_one_station_per_chord":
            by_station[0]["max_surface_normal_velocity"] < 1e-12,
        "first_order_in_the_chord_length": [
            by_seg[i]["max_surface_normal_velocity"]
            / by_seg[i + 1]["max_surface_normal_velocity"]
            for i in range(len(by_seg) - 1)],
        "declared_car_rolls": False,
        "max_abs_difference_in_u": float(np.abs(du).max()),
        "max_abs_difference_in_v": float(np.abs(dv).max()),
        "relative_difference_in_u": float(np.abs(du).max()
                                          / np.abs(fields["static"][0]).max()),
        "bitwise_identical": bool(np.array_equal(fields["rolling"][0],
                                                 fields["static"][0])
                                  and np.array_equal(fields["rolling"][1],
                                                     fields["static"][1])),
        "why": "a rolling surface's velocity is tangential and the closure "
               "carries a normal traction only, so u_s . n = 0 in the "
               "continuum; on a 12-gon a chord's normal is radial only at its "
               "midpoint, so what is left is the polygon's artifact.  The "
               "rotating-surface boundary condition the requirements' section "
               "3.1 asks for is NOT expressible in this model, and the "
               "measurement says so rather than the prose asserting it",
        "macro_steps": 8,
    }

    # -- the rolling ground, which was already there -------------------------
    ground = {
        "how": "ground_effect's band condition, inherited unchanged: row 0 of "
               "the domain is held at u = U_inf, v = 0 every exchange",
        "so": "the road moves at the free stream in the car's frame, which IS "
              "a moving ground; nothing was written for it",
        "what_it_costs": "no floor boundary layer and no viscous gap choke, so "
                         "CS-10's measured ground-effect curve is monotone with "
                         "no turnover -- poc2-frontwing-results' own note",
        "band_cells": GE.BAND,
    }
    return {"bodies": per, "n_bodies": len(flat), "n_objects": len(objs),
            "wheel_calibration": wheel_check,
            "conservation_identity": ident,
            "worst_conservation_residual_x": max(
                r["residual_x"] for r in ident),
            "wheel_rotation": rotation, "moving_ground": ground,
            "notes": list(RL.CAR_NOTES),
            "car_extent_cells": [min(b.x_span()[0] for b in flat),
                                 max(b.x_span()[1] for b in flat),
                                 min(b.y_span()[0] for b in flat),
                                 max(b.y_span()[1] for b in flat)],
            "domain_cells": [RL.RNX, RL.RNY],
            "scale": {"l0_m": VM.VEHICLE.l0_m, "u0_ms": VM.VEHICLE.u0_ms,
                      "t0_s": VM.VEHICLE.t0_s,
                      "cell_m": VM.VEHICLE.cell_m,
                      "domain_m": [RL.RNX * RL.DX * VM.VEHICLE.l0_m,
                                   RL.RNY * RL.DX * VM.VEHICLE.l0_m]}}


# ---------------------------------------------------------------------------
# stage 3 -- the windows, derived
# ---------------------------------------------------------------------------


def stage_windows() -> dict:
    _objs, flat = RL.car_bodies()
    t, info = RL.layout()
    phi_occ = RL.body_profile(flat, kind="occupancy")
    phi_frc = RL.force_profile()
    # the control: the same search under the other profile
    t2, i2 = RL.windows_from_geometry(flat, phi=phi_frc, profile="release-force")
    # the control: what requiring the front wing clear of every cut costs
    t3, i3 = RL.windows_from_geometry(flat, phi=phi_occ, clear_bodies=())
    frontier = RL.layout_frontier(flat, phi=phi_occ)
    sites = RL.device_sites(t)
    specs = {d: RL.RaceDeviceSpec("J1" if d == "RAD" else "J3", s, d, t)
             for d, s in sites.items()}
    return {
        "chosen": info,
        "profile_control": {
            "occupancy_cols": info["cols"],
            "release_force_cols": i2["cols"],
            "the_two_profiles_agree": list(info["cols"]) == list(i2["cols"]),
            "same_column_count": len(info["cols"]) == len(i2["cols"]),
            "max_offset_difference_cells": (
                max(abs(a - b) for a, b in zip(info["cols"], i2["cols"]))
                if len(info["cols"]) == len(i2["cols"]) else None),
            "search_grid_step_cells": info["step"],
            "offsets_that_differ": [
                [a, b] for a, b in zip(info["cols"], i2["cols"]) if a != b]
            if len(info["cols"]) == len(i2["cols"]) else None,
            "why_it_matters": "at the release state a body at zero incidence "
                              "carries exactly zero force, and the car's floor "
                              "is at zero incidence, so the force profile is "
                              "blind to it; the two agreeing says the layout "
                              "does not turn on that blindness",
            "release_force_banded_fraction": i2["banded_force_fraction"],
        },
        "front_wing_clear_control": {
            "with_the_constraint_cols": info["cols"],
            "without_it_cols": i3["cols"],
            "with_the_constraint_banded_fraction": info["banded_force_fraction"],
            "without_it_banded_fraction": i3["banded_force_fraction"],
            "what_the_constraint_costs":
                info["banded_force_fraction"] / i3["banded_force_fraction"],
            "why": "the wet and mount seams pair STRUCT and SUSP with ONE fluid "
                   "window, so the plate has to be owned by one window or the "
                   "declaration does not describe the march",
        },
        "frontier": frontier,
        "w124": {
            "min_cut_to_body_clearance_cells":
                info["min_cut_to_body_clearance_cells"],
            "clear_of_every_body": info["min_cut_to_body_clearance_cells"] > 0,
            "y_cut_clearance_cells": info["y_cut_to_body_clearance_cells"],
            "y_cut_is_clear": info["y_cut_to_body_clearance_cells"] > 0,
            "bodies_cut": info["bodies_cut"],
            "distinct_bodies_cut": len({b["body"] for b in info["bodies_cut"]}),
            "front_wing_cut": any(b["body"] in ("FW_MAIN", "FW_FLAP")
                                  for b in info["bodies_cut"]),
            "finding": "the front wing had ONE body in 208 cells and eight cells "
                       "of clearance; the car has 13 bodies covering 72 to 501 "
                       "of 672 almost without a gap, and the largest stride the "
                       "halo allows is 112, so EVERY covering layout cuts the "
                       "car.  W124's discipline is unachievable here and the "
                       "objective changes from avoiding the bodies to choosing "
                       "where to cut them",
        },
        "devices": {d: {"seam": s.seam, "left": s.left, "right": s.right,
                        "first_cell": s.first_cell,
                        "plane_cells": specs[d].x_plane / RL.DX,
                        "rows": [int(specs[d].rows[0]), int(specs[d].rows[-1])],
                        "ring_to_plane_cells": specs[d].ring_to_plane_cells,
                        "front_wing_case_ring_to_plane_cells": 7.5}
                    for d, s in sites.items()},
    }


# ---------------------------------------------------------------------------
# stage 4 -- the compile
# ---------------------------------------------------------------------------


def _verdict_rows(graph):
    res = C.compile_scheme(graph)
    rows = [{"layer": d.layer, "rule": d.rule, "verdict": d.verdict.value,
             "subject": d.subject, "failure_class": d.failure_class.value,
             "message": d.message[:400]}
            for d in res.decisions]
    from collections import Counter
    return res, rows, dict(Counter(r["verdict"] for r in rows))


def _is_graph_level(subject) -> bool:
    s = subject or ""
    return (s in ("<graph>", "<assembly>", "<run>")
            or s.startswith("<region:") or s.startswith("<assembly:"))


def _seam_verdicts(graph, res):
    """Each seam's colour, and WHAT IT EARNED against what it was handed.

    **A graph-level refusal painted across every seam is one refusal.**
    `L7/R9`'s subject is ``<graph>``: the compiler emits a single decision and a
    per-seam map paints it everywhere so that it can be seen.  Thirty-six red
    tiles from it are thirty-six views of one verdict, not thirty-six verdicts,
    and PoC 2 published the other reading once before W177 corrected it (see
    the note under beat 4 of `atlas/demo_frontwing/README.md`).  So each seam
    carries two colours: ``verdict`` including everything that reaches it, and
    ``local_verdict`` from the decisions whose subject IS this seam, one of its
    two ports or one of its two agents.
    """
    reach = {}
    for c in graph.connections:
        reach[c.seam_id] = {c.seam_id, c.a[0], c.b[0],
                            f"{c.a[0]}.{c.a[1]}", f"{c.b[0]}.{c.b[1]}",
                            c.a[1], c.b[1]}
    out = {}
    for c in graph.connections:
        worst, local, rules, local_rules = "admit", "admit", [], []
        for d in res.decisions:
            if d.verdict.value == "admit":
                continue
            is_local = d.subject in reach[c.seam_id]
            if not (is_local or _is_graph_level(d.subject)):
                continue
            tag = f"{d.layer}/{d.rule}"
            rules.append(tag)
            if d.verdict.value == "refuse":
                worst = "refuse"
            elif worst != "refuse":
                worst = "admit-uncertified"
            if is_local:
                local_rules.append(tag)
                if d.verdict.value == "refuse":
                    local = "refuse"
                elif local != "refuse":
                    local = "admit-uncertified"
        out[c.seam_id] = {"verdict": worst, "rules": sorted(set(rules)),
                          "local_verdict": local,
                          "local_rules": sorted(set(local_rules))}
    return out


def stage_compile(u, v) -> dict:
    out = {}
    g, aux = RL.build(u, v)
    res, rows, counts = _verdict_rows(g)
    seams = _seam_verdicts(g, res)
    from collections import Counter
    out["joined"] = {
        "name": g.name, "verdict": res.verdict.value,
        "n_agents": len(g.agents), "n_seams": len(g.connections),
        "families": dict(Counter(a.capabilities.governing_family
                                 for a in g.agents)),
        "decision_counts": counts,
        "refusals": sorted({f"{r['layer']}/{r['rule']}" for r in rows
                            if r["verdict"] == "refuse"}),
        "decertifications": sorted({f"{r['layer']}/{r['rule']}" for r in rows
                                    if r["verdict"] == "admit-uncertified"}),
        "decisions": rows,
        "seam_verdicts": seams,
        "seam_verdict_counts": dict(Counter(s["verdict"]
                                            for s in seams.values())),
        "seam_local_verdict_counts": dict(Counter(
            s["local_verdict"] for s in seams.values())),
        "graph_level_refusals": sorted({
            f"{d.layer}/{d.rule}" for d in res.decisions
            if d.verdict.value == "refuse" and _is_graph_level(d.subject)}),
        "seam_local_refusals": sorted({
            r for s in seams.values() for r in s["local_rules"]
            if s["local_verdict"] == "refuse"}),
        "how_to_read_the_seam_map":
            "L7/R9 is a GRAPH-level rule: it emits one decision whose "
            "subject is <graph>, and the map paints it across every seam "
            "so it can be seen.  N red tiles from it are N views of ONE "
            "refusal.  `local_verdict` is what each seam earned on its "
            "own subject, its two ports and its two agents.  PoC 2 "
            "published the other reading once, and W177 corrected it",
        "info": aux["info"],
    }
    # -- the control: the disjoint union, which every join is a difference from
    gd, _ = RL.build(u, v, joins=())
    resd, rowsd, countsd = _verdict_rows(gd)
    out["disjoint"] = {
        "name": gd.name, "verdict": resd.verdict.value,
        "n_agents": len(gd.agents), "n_seams": len(gd.connections),
        "decision_counts": countsd,
        "refusals": sorted({f"{r['layer']}/{r['rule']}" for r in rowsd
                            if r["verdict"] == "refuse"}),
        "decertifications": sorted({f"{r['layer']}/{r['rule']}" for r in rowsd
                                    if r["verdict"] == "admit-uncertified"}),
    }
    # -- the control: the clocks reconciled, which must stop L7/R9 refusing
    gr, _ = RL.build(u, v, clocks="reconciled")
    resr, rowsr, countsr = _verdict_rows(gr)
    out["clocks_reconciled"] = {
        "name": gr.name, "verdict": resr.verdict.value,
        "refusals": sorted({f"{r['layer']}/{r['rule']}" for r in rowsr
                            if r["verdict"] == "refuse"}),
    }
    ms = VM.multirate_seams(g)
    out["multirate"] = ms
    out["caused_by_the_joins"] = {
        "refusals_joined": out["joined"]["refusals"],
        "refusals_disjoint": out["disjoint"]["refusals"],
        "only_when_joined": sorted(set(out["joined"]["refusals"])
                                   - set(out["disjoint"]["refusals"])),
        "only_when_disjoint": sorted(set(out["disjoint"]["refusals"])
                                     - set(out["joined"]["refusals"])),
    }
    out["against_CS18"] = {
        "CS18_joined_union_refusals": ["L7/R9"],
        "CS18_disjoint_union_verdict": "admit-uncertified",
        "racelab_matches_on_the_joined_union":
            out["joined"]["refusals"] == ["L7/R9"],
        "racelab_matches_on_the_disjoint_union":
            out["disjoint"]["verdict"] == "admit-uncertified",
        "note": "the car's graph is a different graph -- 26 agents against 18, "
                "14 fluid windows against 6, a different tiling -- so agreeing "
                "on the verdict is a statement that the REASON for the verdict "
                "is structural and not a property of the front wing's layout",
    }
    return out


# ---------------------------------------------------------------------------
# stage 5 -- the instrument, before the gate
# ---------------------------------------------------------------------------


def _march(u, v, tag, **kw):
    key = tag
    cached = load_field(key)
    if cached is not None and os.environ.get("RACELAB_NO_CACHE") != "1":
        return None, cached
    t0 = time.perf_counter()
    last = [t0]

    def progress(s, el):
        if time.perf_counter() - last[0] > 30.0:
            last[0] = time.perf_counter()
            print("      %s: macro-step %d / %s, %.0f s" %
                  (tag, s, kw.get("steps", "?"), el), flush=True)

    m = RL.march(u, v, progress=progress, outflow=OUTFLOW, **kw)
    return m, None


def stage_calib(u, v) -> dict:
    """The instrument: a short march, its repeat, the settling and the cost."""
    t0 = time.perf_counter()
    m1 = RL.march(u, v, steps=CALIB_STEPS, join_coupling="tight", outflow=OUTFLOW)
    t_tight = (time.perf_counter() - t0) / CALIB_STEPS
    t0 = time.perf_counter()
    m2 = RL.march(u, v, steps=CALIB_STEPS, join_coupling="tight", outflow=OUTFLOW)
    t_repeat = (time.perf_counter() - t0) / CALIB_STEPS
    t0 = time.perf_counter()
    ml = RL.march(u, v, steps=CALIB_STEPS, join_coupling="lagged", outflow=OUTFLOW)
    t_lagged = (time.perf_counter() - t0) / CALIB_STEPS
    bitwise = all(np.array_equal(np.asarray(m1.trace[k]), np.asarray(m2.trace[k]))
                  for k in m1.trace)
    n = max(1, CALIB_STEPS // 4)
    band = {}
    for k in ("u_core", "ua", "u_rotor", "current", "q_machine", "load",
              "drag", "u_max"):
        a = np.asarray(m1.trace[k])[-n:]
        mu = float(np.mean(a))
        band[k] = {"mean": mu, "ptp": float(np.ptp(a)),
                   "relative_ptp": float(np.ptp(a) / abs(mu)) if mu else None}
    bal = VM.receiver_balances(m1, SETTLE_FRAC)
    res = np.asarray(m1.join_residual)
    return {
        "steps": CALIB_STEPS,
        "s_per_macro_step_tight": t_tight,
        "s_per_macro_step_tight_repeat": t_repeat,
        "s_per_macro_step_lagged": t_lagged,
        "tight_over_lagged": t_tight / t_lagged,
        "solver_calls_per_exchange_tight_over_lagged":
            (RL.RaceRollout(joins=("J1", "J2", "J3")).n_join_inner + 2) / 1.0,
        "repeat_floor_is_bitwise": bool(bitwise),
        "settle_band_over_the_last_quarter": band,
        "band_norm": float(np.linalg.norm(
            [v["relative_ptp"] for v in band.values()
             if v["relative_ptp"] is not None])),
        "J3_balance_at_this_horizon": bal["J3"],
        "inner_fixed_point_residual_first_to_last":
            [float(res[:, 0].max()), float(res[:, 1].max())] if res.size else None,
        "substeps_max": m1.notes["substeps_max"],
        "u_max_last": float(np.asarray(m1.trace["u_max"])[-1]),
        "u_max_band": RL.U_MAX_BAND,
    }



# ---------------------------------------------------------------------------
# stage 5b -- how long the car takes to settle, and what it never reaches
# ---------------------------------------------------------------------------


SETTLE_STEPS = 1600
BAND_KEYS = ("u_core", "ua", "u_rotor", "current", "q_machine", "load",
             "drag", "u_max")
FLUID_KEYS = ("u_core", "u_rotor", "load", "drag", "u_max")


def _band(trace, lo, hi) -> dict:
    row = {}
    for k in BAND_KEYS:
        a = np.asarray(trace[k])[lo:hi]
        mu = float(np.mean(a))
        row[k] = float(np.ptp(a) / abs(mu)) if mu else None
    row["norm"] = float(np.linalg.norm([v for v in row.values()
                                        if v is not None]))
    row["fluid_only_norm"] = float(np.linalg.norm(
        [row[k] for k in FLUID_KEYS]))
    return row


def stage_settle(u, v) -> dict:
    """A long LAGGED march, and the settle band over four trailing windows.

    **The horizon is chosen from this curve and not from CS-18's number.**  That
    tier marched 1200 steps of a 3.25-unit domain with ONE body in it -- 4.6
    transits -- and its settle band had a norm of 1.008e-3.  This domain is
    10.5 units and holds thirteen bluff bodies whose wakes shed, so the two
    numbers are not comparable and assuming they were would have put a
    "settled" label on a flow that is not.
    """
    t0 = time.perf_counter()
    m = RL.march(u, v, steps=SETTLE_STEPS, join_coupling="lagged", outflow=OUTFLOW)
    wall = time.perf_counter() - t0
    out = {"steps": SETTLE_STEPS, "wall_s": wall,
           "s_per_macro_step": wall / SETTLE_STEPS,
           "coupling": "lagged", "bands": {}}
    for end in (160, 400, 800, 1600):
        if end > SETTLE_STEPS:
            continue
        win = max(1, end // 4)
        out["bands"]["through_%d" % end] = _band(m.trace, end - win, end)
    out["CS18_band_norm_for_comparison"] = 1.008e-3
    out["transits_of_the_domain_at_the_horizon"] = (
        HORIZON * GE.MACRO_DT / (RL.RNX * RL.DX))
    out["transits_CS18_marched"] = 4.6
    out["one_transit_in_macro_steps"] = (RL.RNX * RL.DX) / GE.MACRO_DT
    b = out["bands"]["through_%d" % SETTLE_STEPS]
    out["finding"] = (
        "the fluid quantities still move by %.1f%% over the last quarter of "
        "1600 lagged macro-steps, against CS-18's 0.1%% -- so 'settled' here "
        "means the near field around the bodies has stopped changing fast, "
        "not that the flow is steady, and every residual below is read "
        "against that band rather than against zero" % (100 * b["fluid_only_norm"]))
    out["q_over_u_rotor_band_ratio"] = (b["q_machine"] / b["u_rotor"]
                                        if b["u_rotor"] else None)
    return out


# ---------------------------------------------------------------------------
# stage 6 -- the arms
# ---------------------------------------------------------------------------


ARMS = (
    ("referent", dict(join_coupling="tight")),
    ("repeat", dict(join_coupling="tight")),
    ("all_lagged", dict(join_coupling="lagged")),
    ("null_J3", dict(join_coupling="tight", null="J3")),
    ("null_J1", dict(join_coupling="tight", null="J1")),
)


def stage_march(u, v) -> dict:
    out: dict = {"horizon": HORIZON, "settle_frac": SETTLE_FRAC, "arms": {}}
    marches = {}
    for tag, kw in ARMS:
        print("   arm %s ..." % tag, flush=True)
        t0 = time.perf_counter()
        m = RL.march(u, v, steps=HORIZON, outflow=OUTFLOW, **kw)
        wall = time.perf_counter() - t0
        marches[tag] = m
        bal = VM.receiver_balances(m, SETTLE_FRAC)
        #: **A balance is a RATIO, so its own band is the control.**  The
        #: work and the shaft claim co-vary, and reading the residual
        #: against the band of either component would charge it with
        #: unsteadiness that cancels between them.  This is the
        #: per-macro-step ratio's peak-to-peak over the settle window.
        n_set = max(1, int(round(SETTLE_FRAC * HORIZON)))
        wr = -np.asarray(m.trace["power_rotor"])[-n_set:]
        ps = np.asarray(m.trace["shaft_power"])[-n_set:]
        ratio = wr / np.where(np.abs(ps) > 1e-30, ps, 1.0)
        out["arms"][tag] = {
            "wall_s": wall, "s_per_macro_step": wall / HORIZON,
            "settled": m.settled(SETTLE_FRAC),
            "balances": bal,
            "J3_ratio_per_step_band": {
                "mean": float(np.mean(ratio)),
                "ptp": float(np.ptp(ratio)),
                "relative_ptp": float(np.ptp(ratio) / abs(np.mean(ratio)))
                if float(np.mean(ratio)) else None},
            "settle_band": _band(m.trace, HORIZON - n_set, HORIZON),
            "coolant_steps": len(m.coolant),
            "substeps_max": m.notes["substeps_max"],
            "notes": m.notes,
        }
        save_field("arm_" + tag, u=m.u, v=m.v)
        print("      %.1f s (%.3f s/step)" % (wall, wall / HORIZON), flush=True)
        persist({**load_res(), "march": out})

    ref, rep = marches["referent"], marches["repeat"]
    out["repeat_floor"] = {
        "bitwise_in_every_crossing_quantity": all(
            np.array_equal(np.asarray(ref.trace[k]), np.asarray(rep.trace[k]))
            for k in VM.CROSSING_KEYS if k in ref.trace),
        "bitwise_in_the_field": bool(np.array_equal(ref.u, rep.u)
                                     and np.array_equal(ref.v, rep.v)),
        "wall_ratio": out["arms"]["repeat"]["wall_s"]
                      / out["arms"]["referent"]["wall_s"],
    }
    cv = {t: VM.crossing_vector(
              VM.JoinState(**{k: float(marches[t].settled(SETTLE_FRAC)[k])
                              for k in ("u_core", "ua", "u_rotor", "current",
                                        "q_machine")}),
              float(np.mean(marches[t].trace["t_wall"][-1])))
          for t in marches}
    out["composition_error"] = {
        "all_lagged_against_the_tight_referent":
            VM.composition_error(cv["all_lagged"], cv["referent"]),
        "crossing_keys": list(VM.CROSSING_KEYS),
        "per_component": {
            k: float((cv["all_lagged"][i] - cv["referent"][i])
                     / (cv["referent"][i] if cv["referent"][i] else 1.0))
            for i, k in enumerate(VM.CROSSING_KEYS)},
        "the_floor_that_binds": "the flow's own unsteadiness over the settle "
                                "window, in stage calib's settle band",
    }
    # -- the null arms, beside the referent ---------------------------------
    rb = out["arms"]["referent"]["balances"]
    nb = out["arms"]["null_J3"]["balances"]
    #: **Read off the null arm, not copied from the referent's record.**
    #: `receiver_balances` writes 1.0 there because withholding the force
    #: makes the work exactly zero, which is an identity -- but an
    #: identity written down is not a control, so the arm is asked.
    w_null = nb["J3"]["the join's term: work the rotor's body force does on the fluid"]
    p_null = nb["J3"]["the shaft's claim, T <U_d>"]
    out["G1_J3"] = {
        "residual_with_the_term": rb["J3"]["residual_with_the_term"],
        "residual_without_the_term_null_arm": (
            abs(w_null - p_null) / abs(p_null) if p_null else None),
        "null_arm_work_on_the_fluid": w_null,
        "null_arm_shaft_still_claims": p_null,
        "the_record_s_own_constant_for_this": 1.0,
        "the_ratio_s_own_band_over_the_settle_window":
            out["arms"]["referent"]["J3_ratio_per_step_band"],
        "the_flow_s_own_band_over_the_settle_window":
            out["arms"]["referent"]["settle_band"],
        "u_ring": rb["J3"]["u on the ring the disk reads"],
        "u_plane": rb["J3"]["u in the cell the sink sits in"],
        "u_plane_over_u_ring": rb["J3"]["u_plane / u_ring"],
        "what_the_velocity_gap_alone_predicts":
            1.0 - rb["J3"]["u_plane / u_ring"]
            if rb["J3"]["u_plane / u_ring"] else None,
        "null_arm_moved": {
            "u_rotor": [out["arms"]["referent"]["settled"]["u_rotor"],
                        out["arms"]["null_J3"]["settled"]["u_rotor"]],
            "induction": [out["arms"]["referent"]["settled"]["induction"],
                          out["arms"]["null_J3"]["settled"]["induction"]],
            "current": [out["arms"]["referent"]["settled"]["current"],
                        out["arms"]["null_J3"]["settled"]["current"]],
            "q_machine": [out["arms"]["referent"]["settled"]["q_machine"],
                          out["arms"]["null_J3"]["settled"]["q_machine"]]},
    }
    out["G3_J1"] = {
        "tracking_residual": rb["J1"]["tracking_residual"],
        "ua_release_to_settled": [rb["J1"]["UA at release"],
                                  rb["J1"]["the join's term: UA follows the air"]],
        "air_release_to_settled": rb["J1"]["air through the core, release -> settled"],
        "null_arm_ua_settled": out["arms"]["null_J1"]["settled"]["ua"],
        "null_arm_ua_is_exactly_UA_RAD":
            out["arms"]["null_J1"]["settled"]["ua"] == CL.UA_RAD,
        "null_arm_fluid_is_bitwise_identical": bool(
            np.array_equal(marches["referent"].u, marches["null_J1"].u)
            and np.array_equal(marches["referent"].v, marches["null_J1"].v)),
        "the_air_s_mechanical_loss_across_the_core_unaccounted":
            rb["J1"]["the air's mechanical loss across the core, unaccounted"],
    }
    out["rotor_power_paths"] = VM.rotor_power_paths(marches["referent"],
                                                    SETTLE_FRAC)
    out["cost"] = {
        "tight_over_lagged_wall": out["arms"]["referent"]["s_per_macro_step"]
                                  / out["arms"]["all_lagged"]["s_per_macro_step"],
        "solver_calls_per_exchange": {"tight": 5, "lagged": 1},
    }
    # -- the gate, judged.  The thresholds are read out of `racelab.GATE`
    # -- never retyped -- because a criterion written in prose and
    # evaluated in code drifts, and Tier 48 measured that it drifts
    # permissive.
    g = RL.GATE
    r2 = out["G1_J3"]
    r3 = out["G3_J1"]
    out["verdicts"] = {
        "P2_J3_receiving_balance": {
            "with": r2["residual_with_the_term"],
            "without": r2["residual_without_the_term_null_arm"],
            "tol_with": g["P2_J3_receiving_balance"]["tol_with"],
            "tol_without": g["P2_J3_receiving_balance"]["tol_without"],
            "verdict": ("pass" if (
                r2["residual_with_the_term"] is not None
                and r2["residual_with_the_term"]
                <= g["P2_J3_receiving_balance"]["tol_with"]
                and r2["residual_without_the_term_null_arm"] is not None
                and r2["residual_without_the_term_null_arm"]
                >= g["P2_J3_receiving_balance"]["tol_without"])
                else "fail")},
        "P3_J1_parametric": {
            "with": r3["tracking_residual"],
            "tol_with": g["P3_J1_parametric"]["tol_with"],
            "null_is_exactly_UA_RAD": r3["null_arm_ua_is_exactly_UA_RAD"],
            "null_fluid_bitwise": r3["null_arm_fluid_is_bitwise_identical"],
            "verdict": ("pass" if (
                r3["tracking_residual"] is not None
                and r3["tracking_residual"]
                <= g["P3_J1_parametric"]["tol_with"]
                and r3["null_arm_ua_is_exactly_UA_RAD"]
                and r3["null_arm_fluid_is_bitwise_identical"])
                else "fail")},
        "P5_repeat_floor": {
            "bitwise": out["repeat_floor"][
                "bitwise_in_every_crossing_quantity"],
            "verdict": ("pass" if out["repeat_floor"][
                "bitwise_in_every_crossing_quantity"] else "fail")},
        "P6_macro_step_cost": {
            "lagged_s": out["arms"]["all_lagged"]["s_per_macro_step"],
            "tight_s": out["arms"]["referent"]["s_per_macro_step"],
            "ceiling_s": g["P6_macro_step_cost"]["ceiling_s"],
            "verdict_lagged": ("pass" if out["arms"]["all_lagged"][
                "s_per_macro_step"] <= g["P6_macro_step_cost"]["ceiling_s"]
                else "fail"),
            "verdict_tight": ("pass" if out["arms"]["referent"][
                "s_per_macro_step"] <= g["P6_macro_step_cost"]["ceiling_s"]
                else "fail")},
    }
    return out



# ---------------------------------------------------------------------------
# stage 8 -- the envelope the march did not check, and P3's diagnosis
# ---------------------------------------------------------------------------


def stage_envelope(res) -> dict:
    """**The arms ran outside the rotor's declared envelope and nothing said so.**

    `SizedCircuitSolve.solve` returns `rotor_valid`; `RaceRollout.refresh`
    records it into `JoinState`; and **nothing reads it** -- not the march, not
    the compile.  PoC 2 paid for this once already (W145): a march that drives
    `macro_step` directly can walk a design through an expert's declared floor
    and report a number for it.  The same hole is here, on the powertrain.

    The operating point is an exact function of the inflow ring, so each arm's
    validity is re-derived from its own settled `u_rotor` rather than re-run.
    """
    out: dict = {}
    #: where the machine stops generating: I = (k_e omega - V_oc) / R_total
    #: with omega = lambda u / r, so I changes sign at one inflow.
    grid = [0.55, 0.60, 0.65, 0.6692, 0.70, 0.80, 0.8933, 0.90, 0.9225, 1.00]
    rows = []
    for u in grid:
        ring = np.full(WA.ROTOR_CELLS, u)
        r_, _els, disk = VM.operating_point(ring, width=RL.HOST_ROTOR_WIDTH,
                                            scale=RL.HOST_ROTOR_WIDTH)
        rows.append({"u": u, "induction": float(r_.induction),
                     "omega": float(r_.omega), "current": float(r_.current),
                     "thrust": float(disk.thrust), "shaft_power": float(disk.power),
                     "rotor_valid": bool(r_.rotor_valid)})
    out["sweep"] = rows
    gen = [r_["u"] for r_ in rows if r_["current"] > 0.0]
    out["generates_above_u"] = min(gen) if gen else None
    out["crossover_u_closed_form"] = None
    try:
        els = VM.machine_for_rotor(RL.HOST_ROTOR_WIDTH)
        k_e = float(els["MGU"].k_e)
        r_tot = float(sum(getattr(e, "resistance", 0.0) for e in els.values()))
        #: omega = lambda u / r with r = A/2, so k_e omega = (2 lambda k_e / A) u
        lam = float(WA.TIP_SPEED_RATIO) if hasattr(WA, "TIP_SPEED_RATIO") else 7.5
        slope = 2.0 * lam * k_e / RL.HOST_ROTOR_WIDTH
        v_oc = slope * 0.9225 - rows[-2]["current"] * r_tot
        out["k_e"] = k_e
        out["R_total"] = r_tot
        out["d(k_e omega)/du"] = slope
        out["V_oc_implied"] = v_oc
        out["crossover_u_closed_form"] = v_oc / slope
    except Exception as exc:                                 # pragma: no cover
        out["closed_form_failed"] = str(exc)

    #: each arm's validity, re-derived from its own settled inflow
    arms = {}
    for tag, arm in res.get("march", {}).get("arms", {}).items():
        u = float(arm["settled"]["u_rotor"])
        ring = np.full(WA.ROTOR_CELLS, u)
        r_, _els, disk = VM.operating_point(ring, width=RL.HOST_ROTOR_WIDTH,
                                            scale=RL.HOST_ROTOR_WIDTH)
        arms[tag] = {"u_rotor": u, "rotor_valid": bool(r_.rotor_valid),
                     "induction": float(r_.induction),
                     "induction_is_the_clamp_floor":
                         abs(float(r_.induction) - 0.02) < 1e-12,
                     "current": float(r_.current),
                     "current_sign": ("generating" if r_.current > 0
                                      else "motoring")}
    out["arms"] = arms
    out["every_arm_outside_the_envelope"] = bool(
        arms and not any(a["rotor_valid"] for a in arms.values()))
    out["CS18_u_rotor"] = 0.9225
    out["CS18_rotor_valid"] = True
    out["why"] = (
        "CS-18's rotor sat in OPEN FLOW at u = 0.9225, about 3% above the "
        "inflow at which this machine stops generating.  The car's duct puts "
        "the turbine downstream of a radiator core whose drag is 0.143, so it "
        "sees 0.669 -- 27% lower -- and the machine, which sits just above the "
        "battery's open-circuit voltage, crosses sign.  CS-18 section 7.2 "
        "measured that elasticity at 63.3 and said no rule reads it (W196, "
        "W211).  This is that, realised: one duct of re-siting took the "
        "operating point out of the envelope and no part of the framework "
        "noticed")

    #: **the repair, priced but not taken.**  The same similarity W199 used to
    #: size the machine for a host-sized rotor, applied one level on: size it
    #: for the inflow its host actually delivers.
    try:
        u_duct = float(res["march"]["arms"]["referent"]["settled"]["u_rotor"])
        margin_CS18 = (slope * 0.9225) / v_oc
        need = margin_CS18 * v_oc / (slope * u_duct)
        out["repair_not_taken"] = {
            "what": "scale k_e = k_t by this factor so the machine sits at the "
                    "same margin above its own crossover that CS-18's did",
            "CS18_margin_k_e_omega_over_V_oc": margin_CS18,
            "k_e_scale_needed": need,
            "k_e_now": k_e, "k_e_after": k_e * need,
            "why_not_taken": "it is a second vehicle decision of exactly the "
                             "kind vehicle-scale-and-sizing section 0.1 says is "
                             "the user's call and cheap to overrule, and taking "
                             "it silently would be making that call",
        }
        ring = np.full(WA.ROTOR_CELLS, u_duct)
        els2 = VM.machine_for_rotor(RL.HOST_ROTOR_WIDTH)
        els2["MGU"].k_e = k_e * need
        els2["MGU"].k_t = k_e * need
        r2, _e2, d2 = VM.operating_point(ring, width=RL.HOST_ROTOR_WIDTH,
                                         scale=RL.HOST_ROTOR_WIDTH,
                                         elements=els2)
        out["repair_not_taken"]["control_it_restores_validity"] = {
            "rotor_valid": bool(r2.rotor_valid),
            "induction": float(r2.induction),
            "current": float(r2.current),
        }
    except Exception as exc:                                 # pragma: no cover
        out["repair_failed"] = str(exc)
    return out


def stage_jensen(u, v) -> dict:
    """**Why P3 fails its inherited tolerance, and it is not the join.**

    `receiver_balances` averages `ua` and `u_core` over the settle window and
    then compares ``mean(UA)`` against ``UA_ref (mean(u)/u_ref)^0.8``.  The map
    is CONCAVE, so averaging first and mapping second is not mapping first and
    averaging second: the gap is Jensen's, one half the curvature times the
    window's variance, and it is a property of how settled the flow is rather
    than of the join.  The POINTWISE residual is the identity the clause is
    actually about, and it has to be machine zero.
    """
    m = RL.march(u, v, steps=120, join_coupling="lagged", outflow=OUTFLOW)
    ua = np.asarray(m.trace["ua"], dtype=float)
    uc = np.asarray(m.trace["u_core"], dtype=float)
    ua0, u0 = float(ua[0]), float(uc[0])
    pred_pt = ua0 * (uc / u0) ** IU.UA_EXPONENT
    pointwise = np.abs(ua - pred_pt) / np.abs(pred_pt)
    n = max(1, len(ua) // 4)
    a_bar, u_bar = float(np.mean(ua[-n:])), float(np.mean(uc[-n:]))
    pred_avg = ua0 * (u_bar / u0) ** IU.UA_EXPONENT
    averaged = abs(a_bar - pred_avg) / abs(pred_avg)
    p = IU.UA_EXPONENT
    rel_var = float(np.var(uc[-n:]) / u_bar ** 2)
    jensen = abs(0.5 * p * (p - 1.0) * rel_var)
    return {
        "steps": 120, "settle_window": n,
        "pointwise_residual_max": float(pointwise.max()),
        "pointwise_residual_mean": float(pointwise.mean()),
        "averaged_residual": averaged,
        "relative_variance_of_u_core_over_the_window": rel_var,
        "jensen_term_half_p_p_minus_1_var": jensen,
        "averaged_over_jensen": (averaged / jensen) if jensen else None,
        "tol_with": RL.GATE["P3_J1_parametric"]["tol_with"],
        "reading": "the identity holds POINTWISE to machine precision; what "
                   "fails the inherited tolerance is the average of a concave "
                   "map over a window that has not settled, and CS-18's window "
                   "was settled to 1e-3 where this one is at 2e-2",
    }


# ---------------------------------------------------------------------------
# stage 7 -- the slow half, where J2's receiver lives
# ---------------------------------------------------------------------------


def stage_slow(res) -> dict:
    """J2's receiver on the coolant circuit's OWN clock -- CS-18 section 7.3.

    **Three columns and not two, because the obvious null closes.**  A block
    with no mount term carries `cooling_loop`'s own declared gas temperature on
    its dry face, so its first law still shuts: CS-18 measured that middle
    column at 8.88e-11.  What has to FAIL is the balance written WITH the mount
    term in it and the term withheld, which is Tier 47's `_with_and_without`.
    The recipe here is that tier's, line for line, so the numbers compare.
    """
    q = ua = None
    try:
        q = float(res["march"]["arms"]["referent"]["settled"]["q_machine"])
        ua = float(res["march"]["arms"]["referent"]["settled"]["ua"])
    except (KeyError, TypeError):
        pass
    out: dict = {"q_machine_from_the_march_W": q, "ua_from_the_march": ua}
    if q is None:
        out["not_measured_because"] = "stage march has not run"
        return out

    #: the block's own thermal time constant, measured rather than quoted
    long = VM.march_loop(steps=4000, q_machine=q, ua=ua, mounted=True)
    tw = np.array([r["t_wall_mean"] for r in long.rows])
    start, final = float(tw[0]), float(tw[-1])
    span = final - start
    tau_idx = (int(np.argmax(np.abs(tw - start) >= abs(span) * (1.0 - 1.0 / np.e)))
               if span != 0.0 else 0)
    tau_s = tau_idx * CL.MACRO_DT

    with_term = VM.march_loop(steps=1200, q_machine=q, ua=ua, mounted=True)
    own_source = VM.march_loop(steps=1200, q_machine=q, ua=ua, mounted=False)

    def bal_of(lm, n=300):
        rows = lm.rows[-n:]
        return {"relative": float(np.mean([r["block_first_law"]["relative"]
                                           for r in rows])),
                "t_wall_K": float(rows[-1]["t_wall_mean"]),
                "q_block_W": float(rows[-1]["q_block"]),
                "loop_residual": float(rows[-1]["loop_residual"]),
                "t_return_K": float(rows[-1]["t_return"])}

    a, b = bal_of(with_term), bal_of(own_source)
    #: **the null**: the mount balance with the mount's outer heat withheld,
    #: on a block that has one.  Everything that reads the term stays.
    blk = IU.MountedBlock(q_machine=q)
    tc = np.full(CL.N_SEAM, CL.T_COOLANT_0)
    for _ in range(1200):
        blk.step(tc)
    bb = blk.energy_balance(tc)
    scale = max(abs(bb["q_outer"]), abs(bb["q_wall"]), 1e-30)
    without_the_term = abs(bb["stored_rate"] - (0.0 - bb["q_wall"])) / scale

    g = RL.GATE["P4_J2_receiving_balance"]
    out.update({
        "dt_s": CL.MACRO_DT,
        "block_thermal_time_constant_s": tau_s,
        "block_wall_start_K": start, "block_wall_settled_K": final,
        "coolant_steps_to_span_it": tau_s / CL.MACRO_DT,
        "fluid_macro_steps_to_span_it":
            tau_s / VM.VEHICLE.seconds(GE.MACRO_DT, "front_wing"),
        "with_the_mount_term": a,
        "the_block_with_its_own_declared_source": b,
        "the_mount_balance_without_the_mount_term": without_the_term,
        "tol_with": g["tol_with"], "tol_without": g["tol_without"],
        "ratio_without_over_with": (without_the_term / a["relative"]
                                    if a["relative"] else None),
        "P4_verdict": ("pass" if (a["relative"] <= g["tol_with"]
                                  and without_the_term >= g["tol_without"])
                       else "fail"),
        "horizon_s": 1200 * CL.MACRO_DT,
        "clock_ratio": VM.clock_ratio(),
        "fluid_macro_steps_per_coolant_step": VM.N_FLUID_PER_COOLANT,
        "the_march_s_horizon_in_coolant_steps":
            HORIZON / VM.N_FLUID_PER_COOLANT,
    })
    return out


# ---------------------------------------------------------------------------


STAGES = ("size", "geometry", "windows", "compile", "calib", "settle",
          "march", "slow", "envelope", "jensen")


def main(argv):
    want = STAGES
    for i, a in enumerate(argv):
        if a == "--stages" and i + 1 < len(argv):
            want = tuple(x.strip() for x in argv[i + 1].split(","))
    res = load_res()
    t0 = time.perf_counter()
    res["meta"] = {
        "tier": 51, "phase": 1, "horizon": HORIZON,
        "settle_frac": SETTLE_FRAC,
        "torch_threads": torch.get_num_threads(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "domain_cells": [RL.RNX, RL.RNY],
        "window_cells": [RL.WX, RL.WY],
        "machine_state_at_start": machine_state(),
    }
    u = np.full((RL.RNY, RL.RNX), GE.U_INF)
    v = np.zeros((RL.RNY, RL.RNX))

    if "size" in want:
        print("1. the size, measured before anything is built on it", flush=True)
        res["size"] = stage_size()
        s = res["size"]
        print("   chosen %.4f s/macro-step = %.2fx the front-wing control; "
              "headroom %.2fx under the %.1f s ceiling"
              % (s["chosen"]["s_per_macro_step"],
                 s["chosen"]["ratio_to_the_front_wing_control"],
                 s["headroom_under_the_ceiling"], s["ceiling_s"]), flush=True)
        print("   threads: %d chosen, %.2fx over the worst; control drift %.3fx"
              % (s["threads_chosen"], s["threads_worst_over_best"],
                 s["control_drift_over_the_run"]), flush=True)
        persist(res)

    if "geometry" in want:
        print("2. the car", flush=True)
        res["geometry"] = stage_geometry()
        g = res["geometry"]
        print("   %d bodies; wheel drag %.4f against a cylinder's %.4f (%.2f%%); "
              "worst conservation residual %.3g"
              % (g["n_bodies"], g["wheel_calibration"]["measured_drag"],
                 g["wheel_calibration"]["cylinder_drag_0.5_C_D_U2_D"],
                 100 * g["wheel_calibration"]["relative_error"],
                 g["worst_conservation_residual_x"]), flush=True)
        print("   wheel rotation bitwise invisible: %s"
              % g["wheel_rotation"]["bitwise_identical"], flush=True)
        persist(res)

    if "windows" in want:
        print("3. the windows, derived from the car", flush=True)
        res["windows"] = stage_windows()
        w = res["windows"]
        print("   %d windows, cols %s, halo %d; %.1f%% of the car inside a cut"
              % (w["chosen"]["n_windows"], w["chosen"]["cols"],
                 w["chosen"]["halo"],
                 100 * w["chosen"]["banded_force_fraction"]), flush=True)
        print("   the two profiles agree: %s; W124 clear of every body: %s; "
              "y-cut clear: %s"
              % (w["profile_control"]["the_two_profiles_agree"],
                 w["w124"]["clear_of_every_body"], w["w124"]["y_cut_is_clear"]),
              flush=True)
        persist(res)

    if "compile" in want:
        print("4. the compile", flush=True)
        res["compile"] = stage_compile(u, v)
        c = res["compile"]
        print("   joined: %s, %d agents, %d seams, refusals %s"
              % (c["joined"]["verdict"], c["joined"]["n_agents"],
                 c["joined"]["n_seams"], c["joined"]["refusals"]), flush=True)
        print("   disjoint control: %s, refusals %s"
              % (c["disjoint"]["verdict"], c["disjoint"]["refusals"]), flush=True)
        print("   clocks reconciled: %s, refusals %s"
              % (c["clocks_reconciled"]["verdict"],
                 c["clocks_reconciled"]["refusals"]), flush=True)
        persist(res)

    if "calib" in want:
        print("5. the instrument, before the gate", flush=True)
        res["calib"] = stage_calib(u, v)
        c = res["calib"]
        print("   %.3f s/macro-step tight, %.3f lagged (%.2fx); repeat bitwise %s"
              % (c["s_per_macro_step_tight"], c["s_per_macro_step_lagged"],
                 c["tight_over_lagged"], c["repeat_floor_is_bitwise"]), flush=True)
        print("   the flow's own band over the settle window: %.3g"
              % c["band_norm"], flush=True)
        persist(res)

    if "settle" in want:
        print("5b. how long the car takes to settle", flush=True)
        res["settle"] = stage_settle(u, v)
        st = res["settle"]
        for k, b in st["bands"].items():
            print("   %-14s norm %.4g, fluid-only %.4g"
                  % (k, b["norm"], b["fluid_only_norm"]), flush=True)
        print("   the horizon is %.2f transits; CS-18 marched %.1f"
              % (st["transits_of_the_domain_at_the_horizon"],
                 st["transits_CS18_marched"]), flush=True)
        persist(res)

    if "march" in want:
        print("6. the arms, at %d macro-steps" % HORIZON, flush=True)
        res["march"] = stage_march(u, v)
        m = res["march"]
        print("   G1 (J3) residual with the term %.4g, null arm %.4g"
              % (m["G1_J3"]["residual_with_the_term"],
                 m["G1_J3"]["residual_without_the_term_null_arm"]), flush=True)
        print("   G3 (J1) tracking residual %.4g; null fluid bitwise %s"
              % (m["G3_J1"]["tracking_residual"],
                 m["G3_J1"]["null_arm_fluid_is_bitwise_identical"]), flush=True)
        print("   repeat floor bitwise: %s"
              % m["repeat_floor"]["bitwise_in_every_crossing_quantity"],
              flush=True)
        persist(res)

    if "slow" in want:
        print("7. the slow half, on its own clock", flush=True)
        res["slow"] = stage_slow(res)
        s = res["slow"]
        print("   P4 %s: %s with the mount term, %s with the term removed, "
              "%s on the block's own source"
              % (s.get("P4_verdict"),
                 s.get("with_the_mount_term", {}).get("relative"),
                 s.get("the_mount_balance_without_the_mount_term"),
                 s.get("the_block_with_its_own_declared_source", {}).get(
                     "relative")), flush=True)
        print("   the block's thermal time constant: %s s = %s coolant steps "
              "= %s fluid macro-steps"
              % (s.get("block_thermal_time_constant_s"),
                 s.get("coolant_steps_to_span_it"),
                 s.get("fluid_macro_steps_to_span_it")), flush=True)
        persist(res)

    if "envelope" in want:
        print("8. the envelope the march did not check", flush=True)
        res["envelope"] = stage_envelope(res)
        e = res["envelope"]
        print("   the machine generates only above u = %s; the duct delivers "
              "%s" % (e.get("generates_above_u"),
                      e.get("arms", {}).get("referent", {}).get("u_rotor")),
              flush=True)
        print("   every arm outside the envelope: %s"
              % e.get("every_arm_outside_the_envelope"), flush=True)
        rp = e.get("repair_not_taken", {})
        print("   the repair, priced not taken: scale k_e by %s -> valid %s"
              % (rp.get("k_e_scale_needed"),
                 rp.get("control_it_restores_validity", {}).get("rotor_valid")),
              flush=True)
        persist(res)

    if "jensen" in want:
        print("9. why P3 fails, and it is not the join", flush=True)
        res["jensen"] = stage_jensen(u, v)
        j = res["jensen"]
        print("   pointwise residual %.3g; averaged %.3g; Jensen term %.3g "
              "(ratio %.2f)"
              % (j["pointwise_residual_max"], j["averaged_residual"],
                 j["jensen_term_half_p_p_minus_1_var"],
                 j["averaged_over_jensen"] or float("nan")), flush=True)
        persist(res)

    res["elapsed_seconds"] = time.perf_counter() - t0
    print("wrote", persist(res), "in %.1f s" % res["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    main(sys.argv)
