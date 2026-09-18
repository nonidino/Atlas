r"""W314 -- R2's remaining half: the composed defect at 50:1, against CS-11's bound.

R2 asked for two things.  The first -- declare ``flux_matching=TIME_INTEGRATED``
and supply ``boundary_response_integrated`` -- was done at Tier 76 and is checked
here as the gate's first half (stage ``r9``).  The second, *the composed defect
at 50:1 reported beside CS-11's bound*, is what this script measures.

Three things make the rocket a different case from the brake disc CS-11 measured
the bound on, and each is a stage here.

1.  **CS-11's bound has two constants it calls SEAM properties and one rate it
    calls a RUN property.**  If that split is real the constants must be
    re-measured on this seam and the form must still hold.  If it is not, the
    bound is a description of one brake.  Stage ``transfer`` evaluates CS-11's
    OWN constants against the rocket's measured defect, which is the control
    that makes "the bound holds" non-vacuous.

2.  **CS-11 measured at a settled state and the rocket's shell has none.**  Tier
    76 renamed ``settle()`` to ``march()`` for exactly this reason.  So the
    bound's run-derived rate carries a burn time here, the way beta already
    does, and stage ``rate`` measures how much that costs.

3.  **The clock ratio is pinned at 50 in every row.**  CS-11's central finding is
    that sigma is a function of the exchange INTERVAL and not of the ratio; it
    showed this by pinning the interval and moving the ratio.  This script runs
    the complement -- pin the RATIO at the declared 50 and move the interval --
    and runs CS-11's own control as well, so the two statements are checked
    against each other on a second seam.

**The cost is measured, not estimated, and it is the reason for the stage
split.**  The chamber marches at 111496 s of wall per second of gas time on this
machine at 1.4 GHz (three samples, spread 1.041, none spanning a Modern Standby
-- the first attempt at this number spanned a five-minute one and read 2.6x
high).  One exchange interval costs TWO gas marches of it, so the declared
5e-2 s costs 3.10 hours and is stage ``anchor``, run separately and on mains.
Everything below 5e-3 is the ``core`` and is about 50 minutes.

    python scripts/w314_rocket_multirate_defect.py --stages core --json out/w314.json
    python scripts/w314_rocket_multirate_defect.py --stages anchor --json out/w314.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from atlas.cases import rocket_experts as RE
from atlas.multiphysics import ResponseHalf, interface_power

STAGES = ("setup", "r9", "rate", "slope", "sigma_law", "ratio", "trace_rank",
          "two_way", "transfer", "anchor", "constants")

#: The core is everything affordable in one sitting; `anchor` is the declared
#: exchange interval and costs 3.10 h on its own.
CORE = ("setup", "r9", "rate", "slope", "sigma_law", "ratio", "trace_rank",
        "two_way", "transfer", "constants")

#: Stages that march the seam and so need `setup`'s settled state.  The others
#: are a compile, a shell march and arithmetic, and forcing them through a ten
#: minute settle they never read would be paying for nothing.
NEEDS_SETUP = ("slope", "sigma_law", "ratio", "trace_rank", "two_way", "anchor")

#: Exchange intervals, in seconds.  The RATIO is pinned at 50 in every row --
#: the gas's step is ``interval / 50`` -- so every row is "at 50:1" and the
#: declared row is the last.  `RATIO_ONE` is the single-rate control.
INTERVALS = (2.0e-4, 1.0e-3, 5.0e-3)
ANCHOR_INTERVAL = 5.0e-2
DECLARED_RATIO = 50

#: Uniform trace shifts for the slope, in K.  Three decades, because interface
#: power is BILINEAR and a single-decade fit cannot see the curvature that makes
#: the first-order law an approximation rather than an identity.  Centred lower
#: than CS-11's because the rocket's lag over an affordable interval is smaller.
SHIFTS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)

#: The interval the slope is measured at, and the two it is CHECKED at.  CS-11
#: measured s_seam at one interval and used it as a constant; whether it IS one
#: was never tested.  It is here, because a bound that extrapolates in the
#: interval has to know that its constant does not.
SLOPE_INTERVAL = 2.0e-4
SLOPE_CHECK_INTERVALS = (5.0e-5, 1.0e-3)
SLOPE_CHECK_SHIFTS = (0.1, 1.0)

#: CS-11's own published constants, for the transfer control.  From
#: `case-study-brake-thermal-atlas-0.1` section 3.
CS11_S_GAMMA = 9.4966e-3        # per K
CS11_C2 = 1.4288e-5             # per K^2
CS11_LAMBDA_DOT = 11.94         # K/s
CS11_TIGHTNESS = (1.426, 2.178)

#: The gas is pre-marched to a common state ONCE and every column starts from
#: it.  One chamber flow-through is 0.30 m at 50 m/s, so 6e-3 s is about one;
#: the residual is checked along the way rather than asserted.
SETTLE = 6.0e-3

#: CFL numbers for CS-11's own control -- the ratio moved at a FIXED interval by
#: refining the fast agent's own stability step.  `GasConfig.cfl` defaults 0.4.
CFLS = (0.4, 0.2, 0.1)

#: **What the CORE predicts for the ANCHOR, registered 2026-09-17 before the
#: anchor ran and after the core's three graded rows were in hand.**
#:
#: The whole point of a bound written in the interval is that it can be checked
#: cheaply and spent expensively.  The core measured sigma at 2e-4, 1e-3 and
#: 5e-3 and found it first order with sigma/lag = 4.89e-4 .. 5.39e-4 and a lag
#: exactly proportional to the interval; the anchor is a further 10x out.  So
#: these are the numbers the extrapolation commits to, and the anchor is the one
#: arm that can refute them:
#:
#:   A1  lag at DT = 5e-2 is 1.2943 K within 5%      (lag / DT was 25.89 K/s,
#:                                                    flat to 0.1% over 25x)
#:   A2  sigma is in [5.5e-4, 8.0e-4]                (sigma/lag times that lag)
#:   A3  bound/measured is in [2.0, 2.8]             (the core's 2.269 .. 2.500,
#:                                                    widened by half its span)
#:
#: If A1 holds and A2 fails, sigma is NOT first order out to the declared
#: interval and the bound's usefulness stops somewhere between 5e-3 and 5e-2 --
#: which would be the most interesting outcome available here, and is why the
#: anchor is worth 1.8 hours.
ANCHOR_PREDICTIONS = {
    "A1_lag_K": (1.2943 * 0.95, 1.2943 * 1.05),
    "A2_sigma": (5.5e-4, 8.0e-4),
    "A3_bound_over_measured": (2.0, 2.8),
}

PS = ["powershell", "-NoProfile", "-Command"]


# ---------------------------------------------------------------------------
# the machine, because a cost quoted without it is the wall-clock-rot mistake
# ---------------------------------------------------------------------------


def power_state() -> dict:
    import subprocess
    q = ("$b=Get-CimInstance Win32_Battery; $c=Get-CimInstance Win32_Processor; "
         "Write-Output ('{0}|{1}|{2}|{3}' -f $b.BatteryStatus, "
         "$b.EstimatedChargeRemaining, $c.CurrentClockSpeed, $c.MaxClockSpeed)")
    try:
        out = subprocess.run(PS + [q], capture_output=True, text=True,
                             timeout=30).stdout.strip().split("|")
        return dict(battery_status=int(out[0]), charge_pct=int(out[1]),
                    mhz_now=int(out[2]), mhz_max=int(out[3]))
    except Exception as exc:                                  # pragma: no cover
        return dict(error=repr(exc))


def standby_count(minutes: int) -> int:
    import subprocess
    q = ("Get-WinEvent -FilterHashtable @{LogName='System'; Id=506,507; "
         "StartTime=(Get-Date).AddMinutes(-%d)} -ErrorAction SilentlyContinue "
         "| Measure-Object | Select-Object -ExpandProperty Count" % minutes)
    try:
        out = subprocess.run(PS + [q], capture_output=True, text=True,
                             timeout=30).stdout.strip()
        return int(out) if out.isdigit() else -1
    except Exception:                                         # pragma: no cover
        return -1


# ---------------------------------------------------------------------------
# the two faces, and the map between them
# ---------------------------------------------------------------------------


def overlap_matrix(dst_nodes: np.ndarray, src_nodes: np.ndarray) -> np.ndarray:
    """Row-stochastic ``(n_dst, n_src)`` map by physical z overlap.

    The same construction `gas_h_on_shell_seam` uses, written once and used in
    BOTH directions -- conservative area weighting rather than `generate.py`'s
    normalized-index ``np.interp``, because the two faces cover the same physical
    z window and an index map between different cell counts does not.
    """
    dst = np.asarray(dst_nodes, dtype=float)
    src = np.asarray(src_nodes, dtype=float)
    W = np.zeros((dst.size - 1, src.size - 1))
    for k in range(dst.size - 1):
        lo, hi = min(dst[k], dst[k + 1]), max(dst[k], dst[k + 1])
        ov = np.clip(np.minimum(np.maximum(src[1:], src[:-1]), hi)
                     - np.maximum(np.minimum(src[1:], src[:-1]), lo), 0.0, None)
        s = ov.sum()
        W[k] = ov / s if s > 0 else 1.0 / (src.size - 1)
    return W


class BCSeam:
    """The b-c seam driven directly: chamber gas against airframe shell.

    `ChamberGasAgent.respond` restarts from its own ``_U0`` every call, which is
    what makes `boundary_response_integrated` necessary (R9's own reasoning).  A
    defect measurement needs a march that CARRIES state, so this drives the two
    solvers the way `generate.py` does and reads the same wall flux.
    """

    def __init__(self, burn_time: float = RE.T_PROBE_BURN,
                 dt_setup: float = 1.0e-5, cfl: float | None = None):
        self.gas = RE.ChamberGasAgent(dt=dt_setup)
        self.shell = RE.ShellAgent(dt=5.0e-2)
        h_cold = RE.gas_h_on_shell_seam(self.gas, self.shell, RE.T_SHELL_COLD)
        self.marched = self.shell.march(RE.T_CHAMBER, h_cold, burn_time=burn_time)
        self.gas.T_wall_base = self.marched["T_wall_mean_K"]
        self.h_cold = h_cold
        if cfl is not None:
            self.gas._cfg = self.gas._C2.GasConfig(gamma=RE.GAMMA_GAS,
                                                   R=RE.R_GAS, cfl=float(cfl))
        self.cfl = float(self.gas._cfg.cfl)
        zg = self.gas._blk.nodes[:, 0, 0]
        zs = self.shell._blk.nodes[:, 0, 0]
        c = self.shell._cells
        win = np.concatenate([zs[c], zs[c[-1] + 1:c[-1] + 2]])
        self.P = overlap_matrix(zg, win)        # shell window -> gas face
        self.R = overlap_matrix(win, zg)        # gas face -> shell window
        self.w_gas = RE.face_lengths(self.gas._blk, -1)
        self.T_shell0 = self.shell._T0.copy()
        self.gas_substeps = 0
        self.gas_marched_s = 0.0

    # -- the two sides -----------------------------------------------------
    def wall_T(self, T_nodes: np.ndarray) -> np.ndarray:
        """The shell's seam-window face temperature, 14 cells."""
        return self.shell._face_T(T_nodes)[self.shell._cells]

    def trace_on_gas(self, T_nodes: np.ndarray, scalar: bool = False) -> np.ndarray:
        """The wall the GAS sees.

        ``scalar=True`` is `generate.py::_wall_T`: one number, the mean over
        every shell panel, handed to every gas agent's wall BC.  That is what the
        build repo's own coupler transports, and it is rank 1.
        """
        w = self.wall_T(T_nodes)
        if scalar:
            return np.full(self.gas._n_face, float(w.mean()))
        return self.P @ w

    def gas_step(self, U, dt: float, lam: np.ndarray):
        """One gas march of ``dt`` against the wall ``lam``; the flow it returns."""
        sol = self.gas._solver(lam)
        U, n = sol.advance(U, dt)
        self.gas_substeps += int(n)
        self.gas_marched_s += float(dt)
        q, T_face = self.gas._wall_heat_flux(U, lam)
        flow = RE._as_flow(q, T_face, self.gas.flux_convention)
        return U, flow, q, T_face

    def gas_near_wall(self, U):
        """(near-wall gas temperature, conduction-limited h) on the gas's face."""
        TH, cfg, blk = self.gas._TH, self.gas._cfg, self.gas._blk
        W = TH.cons_to_prim(U, cfg.gamma)
        T = W[..., 3] / (W[..., 0] * cfg.R)
        dn = 0.5 * blk.vol[:, -1] / np.maximum(blk.a_j[:, -1], 1e-30)
        mu = TH.sutherland(T[:, -1], cfg.mu_ref, cfg.T_mu_ref, cfg.sutherland_S)
        return T[:, -1], mu * cfg.cp / cfg.Pr / np.maximum(dn, 1e-9)

    def shell_step(self, T_nodes, dt: float, h14, Tgas14):
        """One backward-Euler shell step under the gas's own h and temperature."""
        full_h = np.full(self.shell._n_face, RE.H_INNER_BACKGROUND)
        full_h[self.shell._cells] = h14
        full_T = np.full(self.shell._n_face, float(self.shell.T_background))
        full_T[self.shell._cells] = Tgas14
        return self.shell._ts.step_thermal(T_nodes, dt, full_h, full_T,
                                           RE.H_OUTER, RE.T_AMBIENT,
                                           T_inf=RE.T_AMBIENT)

    def power(self, lam: np.ndarray, flow: np.ndarray) -> float:
        return interface_power(lam, flow, ResponseHalf.FLOW, self.w_gas)

    # -- the settled start -------------------------------------------------
    def settle(self, seconds: float, checks: int = 3) -> tuple[np.ndarray, list]:
        """Pre-march the gas to a common state, reporting how settled it is.

        The chamber is a fed reservoir with an outflow, so it HAS a steady state;
        the shell does not, which is why one of these is a settle and the other
        is a march with a stated horizon.  The residual is measured rather than
        asserted, because "settled" with no number attached is the thing this
        vault keeps finding.
        """
        lam = self.trace_on_gas(self.T_shell0)
        U = self.gas._U0.copy()
        step = seconds / max(1, checks)
        curve = []
        for k in range(checks):
            prev = U
            U, _flow, _q, _T = self.gas_step(U, step, lam)
            d = float(np.linalg.norm(U - prev) / max(np.linalg.norm(U), 1e-300))
            curve.append(dict(t=(k + 1) * step, rel_change=d))
            print("    settle %7.4f s of gas time: relative state change %.5e"
                  % ((k + 1) * step, d), flush=True)
        return U, curve


# ---------------------------------------------------------------------------
# the deliverable
# ---------------------------------------------------------------------------


def lag_defect(seam: BCSeam, U0, interval: float, ratio: int,
               scalar_trace: bool = False, two_way: bool = False) -> dict:
    """sigma over one exchange interval, against the single-rate referent.

    Both columns march the SAME gas from the SAME state over the SAME interval
    and sub-step it the SAME number of times.  The only difference is the trace
    it is handed: the referent's own path from a tightly coupled rollout at the
    gas's clock, or its value at the start of the interval held constant.  That
    isolates the staleness and nothing else -- in particular it does not confound
    the lag with either agent's time-discretization error, which is what a
    comparison against a finer-stepped column would do.

    **The functional is the GAS's own wall flux in both columns.**  CS-11 found
    that reading the slope through one side while reading the defect through the
    other made its bound over-predict by 5.7x to 8.6x and look exactly like a
    lag-profile effect; interface power agrees between the sides only at a
    converged trace.

    **What drives the SHELL in the referent, and why the default is the
    declaration.**  `ShellAgent.march` drives the shell with ``T_CHAMBER`` and
    `ShellAgent.base_trace` returns the same 2800 K, so that is the shell's
    declared trace and the state this measurement starts from was produced under
    it.  Driving the referent with the gas's own near-wall temperature instead --
    measured here at 1490..2804 K, mean 2464 K -- is the physically two-way
    coupling, and it puts a STEP into the shell's boundary condition at t = 0
    that has nothing to do with the exchange interval.  So the default follows
    the declaration and ``two_way=True`` is run as a control at one interval,
    with the gap between them reported: that gap is what the base disagreement
    costs this tier's quantity, rather than an unstated choice inside it.
    """
    n = max(1, int(ratio))
    dt = interval / n
    Ts = seam.T_shell0.copy()
    lam0 = seam.trace_on_gas(Ts, scalar=scalar_trace)
    h_dec = np.asarray(seam.h_cold, dtype=float)
    T_dec = np.full(seam.shell.n_seam, float(RE.T_CHAMBER))

    U, acc = U0.copy(), 0.0
    lam = lam0
    for _ in range(n):
        lam = seam.trace_on_gas(Ts, scalar=scalar_trace)
        U, flow, _q, _Tf = seam.gas_step(U, dt, lam)
        acc += seam.power(lam, flow)
        if two_way:
            Tg, h = seam.gas_near_wall(U)
            Ts = seam.shell_step(Ts, dt, seam.R @ h, seam.R @ Tg)
        else:
            Ts = seam.shell_step(Ts, dt, h_dec, T_dec)
    p_ref = acc / n
    lam_end = lam

    U, acc = U0.copy(), 0.0
    for _ in range(n):
        U, flow, _q, _Tf = seam.gas_step(U, dt, lam0)
        acc += seam.power(lam0, flow)
    p_held = acc / n

    lag = float(np.linalg.norm(lam_end - lam0) / np.sqrt(lam0.size))
    return dict(interval=interval, ratio=n, dt_gas=dt,
                sigma=abs(p_held - p_ref) / abs(p_ref), lag=lag,
                lag_max=float(np.max(np.abs(lam_end - lam0))),
                p_ref=float(p_ref), p_held=float(p_held),
                lag_rate=lag / interval, scalar_trace=bool(scalar_trace),
                two_way=bool(two_way))


def sigma_uniform(seam: BCSeam, U0, interval: float, ratio: int,
                  lam: np.ndarray, shift: float) -> tuple[float, float]:
    """The interface-power defect from displacing the trace UNIFORMLY by `shift`.

    W86's construction, and the one part of sigma that is a property of the SEAM
    rather than of the run.  Evaluated through the same functional `lag_defect`
    reads, which is the part CS-11 had to fix.
    """
    def power_of(trace):
        n = max(1, int(ratio))
        U, acc = U0.copy(), 0.0
        for _ in range(n):
            U, flow, _q, _Tf = seam.gas_step(U, interval / n, trace)
            acc += seam.power(trace, flow)
        return acc / n

    pa = power_of(lam)
    pb = power_of(lam + shift)
    return abs(pb - pa) / abs(pa), pa


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def stage_setup(out: dict, a) -> None:
    print("=== the two agents, their clocks, and the ratio the YAML gives ===")
    t0 = time.perf_counter()
    seam = BCSeam(burn_time=a.burn)
    cfg = RE.rocket_config()
    dt_gas_declared = float(cfg.dt_model["b"])
    dt_shell = float(cfg.dt_model["c"])
    print("  chamber block %s, %d wall cells; shell seam window %d cells"
          % (str(seam.gas._blk.shape), seam.gas._n_face, seam.shell.n_seam))
    print("  declared dt_model: b = %.4g s, c = %.4g s  ->  ratio %.0f"
          % (dt_gas_declared, dt_shell, dt_shell / dt_gas_declared))
    print("  dt_macro = %.4g s, which R4 floors at max_i dt_native = %.4g s"
          % (float(cfg.dt_macro), dt_shell))
    print("  shell MARCHED %.2f s of burn to %.4f K mean (%.4f K max); validity %s"
          % (seam.marched["burn_time_s"], seam.marched["T_wall_mean_K"],
             seam.marched["T_wall_max_K"],
             "holds" if seam.marched["valid_at_base"] else "DECLINES"))
    print("  overlap maps: P %s (shell window -> gas face), R %s (the transpose)"
          % (str(seam.P.shape), str(seam.R.shape)))
    print("  row sums: P %.12f..%.12f  R %.12f..%.12f  (row-stochastic)"
          % (seam.P.sum(1).min(), seam.P.sum(1).max(),
             seam.R.sum(1).min(), seam.R.sum(1).max()))
    print()
    print("  pre-marching the gas to a common start, %g s of gas time:" % a.settle)
    U0, curve = seam.settle(a.settle)
    print()
    #: **W74's class, on the side nobody has checked.**  `ShellAgent.march` drives
    #: the shell with ``T_CHAMBER`` on the seam and `ShellAgent.base_trace`
    #: returns the same 2800 K -- so the shell is linearised about a gas
    #: temperature of 2800 K.  What the gas actually supplies at its own wall is
    #: its NEAR-WALL cell temperature, and the two are only the same if the
    #: boundary layer is thin.  Measured rather than assumed, both ways.
    Tg, hg = seam.gas_near_wall(U0)
    base_gap = float(Tg.mean() / RE.T_CHAMBER)
    print("  base consistency, gas -> shell:")
    print("    the shell is marched and linearised about T_gas = %.1f K"
          % RE.T_CHAMBER)
    print("    the gas's own near-wall cell is  %.3f .. %.3f K, mean %.3f K"
          % (Tg.min(), Tg.max(), Tg.mean()))
    print("    ratio near-wall / chamber = %.4f" % base_gap)
    print("    h: march used %.3f W/(m^2 K) mean; settled gas gives %.3f"
          % (seam.h_cold.mean(), float((seam.R @ hg).mean())))
    wall = seam.wall_T(seam.T_shell0)
    print("  the wall the gas sees: %.3f .. %.3f K (spread %.3f K)"
          % (wall.min(), wall.max(), wall.max() - wall.min()))
    print()
    ps = power_state()
    print("  machine: battery_status=%s charge=%s%% clock=%s/%s MHz"
          % (ps.get("battery_status"), ps.get("charge_pct"),
             ps.get("mhz_now"), ps.get("mhz_max")))
    rate = seam.gas_marched_s and (time.perf_counter() - t0) / seam.gas_marched_s
    print("  gas march rate so far: %.0f s of wall per second of gas time" % rate)
    out["_seam"] = seam
    out["_U0"] = U0
    out["setup"] = dict(
        gas_shape=list(seam.gas._blk.shape), n_face_gas=seam.gas._n_face,
        n_seam_shell=int(seam.shell.n_seam),
        dt_gas_declared=dt_gas_declared, dt_shell=dt_shell,
        declared_ratio=dt_shell / dt_gas_declared,
        dt_macro=float(cfg.dt_macro), cfl=seam.cfl,
        settle_seconds=a.settle, settle_curve=curve,
        marched={k: v for k, v in seam.marched.items() if k != "curve"},
        h_cold_mean=float(seam.h_cold.mean()),
        gas_near_wall_T_mean=float(Tg.mean()),
        gas_near_wall_T_min=float(Tg.min()),
        gas_near_wall_T_max=float(Tg.max()),
        shell_linearised_about_K=float(RE.T_CHAMBER),
        base_gap_near_wall_over_chamber=base_gap,
        h_settled_mean=float((seam.R @ hg).mean()),
        wall_spread_K=float(wall.max() - wall.min()),
        machine=ps, standby_last_hour=standby_count(60),
        setup_wall_s=time.perf_counter() - t0)


def stage_r9(out: dict, a) -> None:
    print("=== the gate's FIRST half: what L7/R9 decides on the real graph ===")
    from atlas.compiler import compile_scheme
    t0 = time.perf_counter()
    graph, _experts = RE.build_rocket_real(dt_scale=a.dt_scale)
    res = compile_scheme(graph)
    rows = []
    for d in res.decisions.decisions:
        if not d.rule.startswith("R9"):
            continue
        rows.append(dict(layer=d.layer, rule=d.rule, verdict=d.verdict.value,
                         subject=d.subject, message=d.message))
        print("  %-3s %-14s %-20s %s" % (d.layer, d.rule, d.verdict.value,
                                         d.message[:96]))
    admits = [r for r in rows if r["rule"] == "R9" and r["verdict"] == "admit"]
    print()
    print("  flux_matching declared: %s" % graph.flux_matching.value)
    print("  macro_dt %.4g; agents at a multirate seam: %s"
          % (graph.macro_dt, sorted(graph.agents_at_multirate_seams())))
    print("  L7/R9 ADMITS: %s" % bool(admits))
    out["r9"] = dict(rows=rows, admits=bool(admits),
                     flux_matching=graph.flux_matching.value,
                     macro_dt=float(graph.macro_dt),
                     verdict=res.verdict.value,
                     at_multirate=sorted(graph.agents_at_multirate_seams()),
                     wall_s=time.perf_counter() - t0)


def stage_rate(out: dict, a) -> None:
    print("=== the run-derived constant, and the burn time it carries ===")
    print("  CS-11 measured lambda_dot at a SETTLED state.  The rocket's shell")
    print("  has none -- Tier 76 renamed settle() to march() for that reason --")
    print("  so the rate is a function of when in the burn it is taken.")
    print()
    rows = []
    print("  %10s %14s %14s %14s" % ("burn [s]", "wall mean [K]", "d(wall)/dt",
                                     "rel to 5 s"))
    ref = None
    for bt in a.burns:
        sh = RE.ShellAgent(dt=5.0e-2)
        gas = RE.ChamberGasAgent(dt=1.0e-5)
        h = RE.gas_h_on_shell_seam(gas, sh, RE.T_SHELL_COLD)
        m = sh.march(RE.T_CHAMBER, h, burn_time=bt)
        T1 = sh._T0.copy()
        full_h = np.full(sh._n_face, RE.H_INNER_BACKGROUND)
        full_h[sh._cells] = h
        full_T = np.full(sh._n_face, float(sh.T_background))
        full_T[sh._cells] = RE.T_CHAMBER
        T2 = sh._ts.step_thermal(T1, sh.dt, full_h, full_T, RE.H_OUTER,
                                 RE.T_AMBIENT, T_inf=RE.T_AMBIENT)
        f1 = sh._face_T(T1)[sh._cells]
        f2 = sh._face_T(T2)[sh._cells]
        rate = float(np.linalg.norm(f2 - f1) / np.sqrt(f1.size) / sh.dt)
        if ref is None and abs(bt - RE.T_PROBE_BURN) < 1e-12:
            ref = rate
        rows.append(dict(burn=bt, wall_mean=m["T_wall_mean_K"], rate=rate))
        print("  %10.2f %14.4f %14.5e %14s"
              % (bt, m["T_wall_mean_K"], rate, "-"), flush=True)
    ref = ref or rows[-1]["rate"]
    for r in rows:
        r["rel_to_5s"] = r["rate"] / ref
    lo = min(r["rate"] for r in rows)
    hi = max(r["rate"] for r in rows)
    print()
    print("  lambda_dot over the swept burn times: %.5e .. %.5e K/s, a factor "
          "of %.3f" % (lo, hi, hi / max(lo, 1e-300)))
    print("  CS-11's own lambda_dot was %.2f K/s at a settled state."
          % CS11_LAMBDA_DOT)
    out["rate"] = dict(rows=rows, lambda_dot_at_5s=ref, lo=lo, hi=hi,
                       span=hi / max(lo, 1e-300),
                       cs11_lambda_dot=CS11_LAMBDA_DOT,
                       over_cs11=ref / CS11_LAMBDA_DOT)


def stage_slope(out: dict, a) -> None:
    print("=== s_seam and C2: the constants CS-11 calls SEAM properties ===")
    seam, U0 = out["_seam"], out["_U0"]
    lam = seam.trace_on_gas(seam.T_shell0)
    rows = []
    print("  measured at DT = %.4g s, ratio %d" % (SLOPE_INTERVAL, DECLARED_RATIO))
    print("  %11s %14s %15s %12s" % ("shift [K]", "sigma", "sigma/shift", "log ratio"))
    prev = None
    for s in SHIFTS:
        sg, p0 = sigma_uniform(seam, U0, SLOPE_INTERVAL, DECLARED_RATIO, lam, s)
        r = (np.log(sg / prev[1]) / np.log(s / prev[0])) if prev else float("nan")
        rows.append(dict(shift=s, sigma=sg, slope=sg / s, exponent=r))
        print("  %11.3f %14.5e %15.5e %12.4f" % (s, sg, sg / s, r), flush=True)
        prev = (s, sg)
    s_seam = rows[0]["slope"]
    big = rows[-1]
    c2 = (big["sigma"] - s_seam * big["shift"]) / big["shift"] ** 2
    print()
    print("  s_seam = %.6e per K (smallest shift, the linear branch)" % s_seam)
    print("  C2     = %.6e per K^2 (closes the gap at %g K)" % (c2, big["shift"]))
    print()
    print("  and whether s_seam IS a constant in the interval, which CS-11 used")
    print("  it as and never tested:")
    checks = []
    print("  %11s %14s %14s" % ("DT [s]", "s_seam", "rel to base"))
    for dt_ex in SLOPE_CHECK_INTERVALS:
        sub = []
        for s in SLOPE_CHECK_SHIFTS:
            sg, _p = sigma_uniform(seam, U0, dt_ex, DECLARED_RATIO, lam, s)
            sub.append(sg / s)
        val = float(sub[0])
        checks.append(dict(interval=dt_ex, s_seam=val, rel=val / s_seam,
                           slopes=sub))
        print("  %11.4g %14.5e %14.4f" % (dt_ex, val, val / s_seam), flush=True)
    spread = max([s_seam] + [c["s_seam"] for c in checks]) / \
        min([s_seam] + [c["s_seam"] for c in checks])
    print()
    print("  s_seam moves by %.4fx over %.0fx of exchange interval."
          % (spread, max(SLOPE_CHECK_INTERVALS) / min(SLOPE_CHECK_INTERVALS)))
    out["slope"] = dict(rows=rows, s_seam=s_seam, C2=c2, checks=checks,
                        s_seam_spread=float(spread),
                        interval=SLOPE_INTERVAL, ratio=DECLARED_RATIO)


def _bound(s_seam, c2, lag):
    return s_seam * lag + c2 * lag ** 2


def _sigma_table(out, rows, title):
    s_seam = out.get("slope", {}).get("s_seam")
    c2 = out.get("slope", {}).get("C2")
    if s_seam is None:
        return
    print()
    print("  %s -- from the slope stage's constants and each interval's own" % title)
    print("  lag. Nothing here is fitted to this table.")
    print("  %10s %14s %14s %12s" % ("DT [s]", "measured", "bound", "bound/meas"))
    for r in rows:
        b = _bound(s_seam, c2, r["lag"])
        r["bound"] = b
        r["bound_over_measured"] = (b / r["sigma"] if r["sigma"] > 0
                                    else float("inf"))
        print("  %10.4g %14.5e %14.5e %12.4f"
              % (r["interval"], r["sigma"], b, r["bound_over_measured"]))


def stage_sigma_law(out: dict, a) -> None:
    print("=== sigma as a function of the exchange interval, RATIO PINNED at 50 ===")
    print("  CS-11 pinned the interval and moved the ratio.  This is the")
    print("  complement: every row is at the declared 50:1 and the interval")
    print("  moves, so the two statements check each other on a second seam.")
    print()
    seam, U0 = out["_seam"], out["_U0"]
    rows = []
    print("  %10s %8s %13s %13s %13s %9s"
          % ("DT [s]", "ratio", "lag [K]", "sigma", "sigma/lag", "log ratio"))

    ctl = lag_defect(seam, U0, INTERVALS[0] / DECLARED_RATIO, 1)
    ctl["control"] = True
    rows.append(ctl)
    print("  %10.4g %8d %13.5e %13.5e %13s %9s   <- single-rate CONTROL"
          % (ctl["interval"], 1, ctl["lag"], ctl["sigma"], "-", "-"), flush=True)

    prev = None
    intervals = list(INTERVALS) + ([ANCHOR_INTERVAL] if a.with_anchor else [])
    for dt_ex in intervals:
        t0 = time.perf_counter()
        r = lag_defect(seam, U0, dt_ex, DECLARED_RATIO)
        r["wall_s"] = time.perf_counter() - t0
        r["control"] = False
        e = (np.log(r["sigma"] / prev["sigma"]) / np.log(dt_ex / prev["interval"])
             if prev and prev["sigma"] > 0 else float("nan"))
        r["exponent"] = e
        rows.append(r)
        print("  %10.4g %8d %13.5e %13.5e %13.5e %9.3f   [%.0f s]"
              % (dt_ex, DECLARED_RATIO, r["lag"], r["sigma"],
                 r["sigma"] / max(r["lag"], 1e-300), e, r["wall_s"]), flush=True)
        prev = r
        out["sigma_law"] = dict(rows=rows)
        _persist(out, a)

    _sigma_table(out, rows, "the BOUND")
    graded = [r for r in rows if r["sigma"] > 0.0 and "bound" in r]
    control = [r for r in rows if r["sigma"] == 0.0]
    print()
    for r in control:
        print("  ratio 1 is the single-rate control and its defect is %.1e --"
              % r["sigma"])
        print("  EXACTLY zero, because one gas call IS the interval.  It carries")
        print("  no tightness ratio; quoting 0/0 as one would be W106's mistake.")
    if graded:
        holds = all(r["bound_over_measured"] >= 1.0 for r in graded)
        tight = min(r["bound_over_measured"] for r in graded)
        worst = max(r["bound_over_measured"] for r in graded)
        print("  the bound holds at every graded interval: %s" % holds)
        print("  loose by %.3fx to %.3fx over %.0fx of exchange interval"
              % (tight, worst,
                 graded[-1]["interval"] / graded[0]["interval"]))
        print("  CS-11's own tightness on the brake seam was %.3fx to %.3fx."
              % CS11_TIGHTNESS)
        out["sigma_law"] = dict(
            rows=rows, bound_holds=bool(holds), tightness=[tight, worst],
            lag_rate=float(np.mean([r["lag_rate"] for r in graded])))
    else:
        out["sigma_law"] = dict(rows=rows)


def stage_ratio(out: dict, a) -> None:
    print("=== CS-11's OWN control: the RATIO at a fixed exchange interval ===")
    print("  The interval sweep moves two things at once.  Here the interval is")
    print("  pinned and the gas's own stability step is refined through its CFL")
    print("  number, so the ratio moves with the staleness untouched.")
    print()
    #: **The settled state is REUSED across the CFL arms, not re-settled.**  The
    #: CFL number changes how the gas steps, not what state it is in, so
    #: re-settling under each CFL would vary two things and the control would
    #: not be one.  It is also the difference between 5 minutes and 78.
    U0 = out["_U0"]
    rows = []
    print("  %6s %10s %14s %13s %12s"
          % ("CFL", "substeps", "dt_stable [s]", "sigma", "relative"))
    base = None
    for c in CFLS:
        seam = BCSeam(burn_time=a.burn, cfl=c)
        n0 = seam.gas_substeps
        r = lag_defect(seam, U0, SLOPE_INTERVAL, DECLARED_RATIO)
        used = seam.gas_substeps - n0
        dts = SLOPE_INTERVAL * 2.0 / max(used, 1)
        base = base if base is not None else r["sigma"]
        rows.append(dict(cfl=c, substeps=used, dt_stable=dts, sigma=r["sigma"],
                         relative=r["sigma"] / max(base, 1e-300),
                         ratio=SLOPE_INTERVAL / max(dts, 1e-300)))
        print("  %6.2f %10d %14.5e %13.6e %12.6f"
              % (c, used, dts, r["sigma"], r["sigma"] / max(base, 1e-300)),
              flush=True)
        out["ratio"] = dict(rows=rows)
        _persist(out, a)
    span = max(r["ratio"] for r in rows) / min(r["ratio"] for r in rows)
    move = max(r["relative"] for r in rows) / min(r["relative"] for r in rows)
    print()
    print("  %.1fx of clock ratio at a fixed interval moves sigma by %.6fx."
          % (span, move))
    print("  CS-11 measured 8x of ratio moving sigma by 1.000160x.")
    out["ratio"] = dict(rows=rows, ratio_span=float(span), sigma_move=float(move))


def stage_trace_rank(out: dict, a) -> None:
    print("=== what the BUILD REPO's coupler actually transports on this seam ===")
    seam, U0 = out["_seam"], out["_U0"]
    #: Read from the build repo rather than asserted, so this is a measurement
    #: of that file and not a claim about it.
    src = os.path.join(RE.build_repo(), "src", "atlas", "data", "generate.py")
    body = open(src, encoding="utf-8").read() if os.path.exists(src) else ""
    scalar_return = "return float(np.mean([T.mean() for T in self.T_shell]))" in body
    print("  generate.py exists at that path: %s; _wall_T returns a scalar mean "
          "over every panel: %s" % (os.path.exists(src), scalar_return))
    print("  generate.py::_wall_T returns ONE number -- the mean over every shell")
    print("  panel -- and hands it to every gas agent's wall BC.  The gas->shell")
    print("  direction is per-station (np.interp on normalized index); the")
    print("  shell->gas direction is rank 1.")
    print()
    lam_field = seam.trace_on_gas(seam.T_shell0, scalar=False)
    lam_scalar = seam.trace_on_gas(seam.T_shell0, scalar=True)
    spread = float(lam_field.max() - lam_field.min())
    print("  the wall the gas would see, per station: %.4f .. %.4f K (spread "
          "%.4f K)" % (lam_field.min(), lam_field.max(), spread))
    print("  the wall generate.py hands it:           %.4f K, rank 1"
          % lam_scalar[0])
    print()
    print("  and what that costs the defect, at two intervals:")
    rows = []
    print("  %10s %15s %15s %10s"
          % ("DT [s]", "sigma per-face", "sigma scalar", "ratio"))
    for dt_ex in (SLOPE_INTERVAL, 1.0e-3):
        rf = [r for r in out.get("sigma_law", {}).get("rows", [])
              if abs(r["interval"] - dt_ex) < 1e-15 and not r.get("scalar_trace")]
        f = rf[0]["sigma"] if rf else lag_defect(
            seam, U0, dt_ex, DECLARED_RATIO)["sigma"]
        s = lag_defect(seam, U0, dt_ex, DECLARED_RATIO, scalar_trace=True)
        rows.append(dict(interval=dt_ex, sigma_face=f, sigma_scalar=s["sigma"],
                         lag_face=(rf[0]["lag"] if rf else None),
                         lag_scalar=s["lag"],
                         ratio=s["sigma"] / max(f, 1e-300)))
        print("  %10.4g %15.5e %15.5e %10.4f"
              % (dt_ex, f, s["sigma"], s["sigma"] / max(f, 1e-300)), flush=True)
        out["trace_rank"] = dict(rows=rows)
        _persist(out, a)
    out["trace_rank"] = dict(rows=rows, wall_spread_K=spread,
                             wall_scalar_K=float(lam_scalar[0]),
                             wall_min_K=float(lam_field.min()),
                             wall_max_K=float(lam_field.max()),
                             generate_py_source=src)


def stage_two_way(out: dict, a) -> None:
    print("=== what the base disagreement costs THIS tier's quantity ===")
    st = out.get("setup", {})
    print("  The shell is marched and linearised about T_gas = %.1f K; the gas's"
          % RE.T_CHAMBER)
    print("  own near-wall cell is %.1f .. %.1f K, mean %.1f (ratio %.4f)."
          % (st.get("gas_near_wall_T_min", float("nan")),
             st.get("gas_near_wall_T_max", float("nan")),
             st.get("gas_near_wall_T_mean", float("nan")),
             st.get("base_gap_near_wall_over_chamber", float("nan"))))
    print("  Tier 77 measured the other direction of this at 876%% of the base")
    print("  norm and it moved beta by 114x.  This is the same class, one")
    print("  direction along, priced on sigma instead of on beta.")
    print()
    seam, U0 = out["_seam"], out["_U0"]
    rows = []
    print("  %10s %15s %15s %10s"
          % ("DT [s]", "declared trace", "two-way", "ratio"))
    for dt_ex in (SLOPE_INTERVAL, 1.0e-3):
        have = [r for r in out.get("sigma_law", {}).get("rows", [])
                if abs(r["interval"] - dt_ex) < 1e-15
                and not r.get("scalar_trace") and not r.get("two_way")]
        d = (have[0]["sigma"] if have
             else lag_defect(seam, U0, dt_ex, DECLARED_RATIO)["sigma"])
        t = lag_defect(seam, U0, dt_ex, DECLARED_RATIO, two_way=True)
        rows.append(dict(interval=dt_ex, sigma_declared=d,
                         sigma_two_way=t["sigma"], lag_two_way=t["lag"],
                         ratio=t["sigma"] / max(d, 1e-300)))
        print("  %10.4g %15.5e %15.5e %10.4f"
              % (dt_ex, d, t["sigma"], t["sigma"] / max(d, 1e-300)), flush=True)
        out["two_way"] = dict(rows=rows)
        _persist(out, a)
    span = max(r["ratio"] for r in rows) / min(r["ratio"] for r in rows)
    print()
    print("  the two-way referent moves sigma by %.4fx to %.4fx, and that "
          "spread is %.4fx" % (min(r["ratio"] for r in rows),
                               max(r["ratio"] for r in rows), span))
    out["two_way"] = dict(rows=rows, spread=float(span))


def stage_transfer(out: dict, a) -> None:
    print("=== the control that makes 'the bound holds' mean something ===")
    print("  CS-11 calls s_seam and C2 SEAM properties and lambda_dot a RUN")
    print("  property.  If its own constants happened to bound the rocket too,")
    print("  the rocket's measurement would be evidence of nothing.")
    print()
    rows = [r for r in out.get("sigma_law", {}).get("rows", [])
            if r["sigma"] > 0.0]
    if not rows:
        print("  (no graded rows; run sigma_law first)")
        return
    sl = out.get("slope", {})
    print("  %10s %13s %13s %10s %13s %10s"
          % ("DT [s]", "measured", "rocket bnd", "ratio", "CS-11 bnd", "ratio"))
    tab = []
    for r in rows:
        br = _bound(sl["s_seam"], sl["C2"], r["lag"])
        bc = _bound(CS11_S_GAMMA, CS11_C2, r["lag"])
        tab.append(dict(interval=r["interval"], sigma=r["sigma"],
                        bound_rocket=br, rel_rocket=br / r["sigma"],
                        bound_cs11=bc, rel_cs11=bc / r["sigma"]))
        print("  %10.4g %13.5e %13.5e %10.4f %13.5e %10.4f"
              % (r["interval"], r["sigma"], br, br / r["sigma"],
                 bc, bc / r["sigma"]))
    worst = min(t["rel_cs11"] for t in tab)
    holds = all(t["rel_cs11"] >= 1.0 for t in tab)
    print()
    print("  CS-11's own constants bound the rocket's defect: %s (worst %.4f)"
          % (holds, worst))
    print("  s_seam: rocket %.6e vs CS-11 %.6e, a factor of %.4f"
          % (sl["s_seam"], CS11_S_GAMMA, sl["s_seam"] / CS11_S_GAMMA))
    out["transfer"] = dict(rows=tab, cs11_holds=bool(holds),
                           cs11_worst=float(worst),
                           s_seam_ratio=sl["s_seam"] / CS11_S_GAMMA,
                           C2_ratio=sl["C2"] / CS11_C2)


def stage_anchor(out: dict, a) -> None:
    print("=== the ANCHOR: the declared exchange interval, paid for ===")
    print("  DT = %g s at ratio %d -- the config's own dt_macro against its own"
          % (ANCHOR_INTERVAL, DECLARED_RATIO))
    print("  dt_model['b'].  Two gas marches of %g s each; the cost IS the"
          % ANCHOR_INTERVAL)
    print("  measurement and it is why this is a separate stage.")
    print()
    seam, U0 = out["_seam"], out["_U0"]
    t0 = time.perf_counter()
    r = lag_defect(seam, U0, ANCHOR_INTERVAL, DECLARED_RATIO)
    r["wall_s"] = time.perf_counter() - t0
    r["control"] = False
    sl = out.get("slope", {})
    if sl:
        r["bound"] = _bound(sl["s_seam"], sl["C2"], r["lag"])
        r["bound_over_measured"] = r["bound"] / r["sigma"]
    print("  lag   %.6e K  (max %.6e K)" % (r["lag"], r["lag_max"]))
    print("  sigma %.6e" % r["sigma"])
    if "bound" in r:
        print("  bound %.6e   ->  bound/measured %.4f"
              % (r["bound"], r["bound_over_measured"]))
    print("  cost  %.0f s (%.2f h), %d gas sub-steps"
          % (r["wall_s"], r["wall_s"] / 3600.0, seam.gas_substeps))
    print("  machine: %s" % power_state())
    print()
    print("  and the three the CORE registered before this ran:")
    got = {"A1_lag_K": r["lag"], "A2_sigma": r["sigma"],
           "A3_bound_over_measured": r.get("bound_over_measured")}
    checks = {}
    for name, (lo, hi) in ANCHOR_PREDICTIONS.items():
        v = got[name]
        ok = v is not None and lo <= v <= hi
        checks[name] = dict(lo=lo, hi=hi, got=v, held=bool(ok))
        print("    %-24s %-6s got %s, registered [%g, %g]"
              % (name, "HELD" if ok else "FAILED",
                 "None" if v is None else "%.6g" % v, lo, hi))
    r["anchor_predictions"] = checks
    out["anchor"] = dict(row=r, machine=power_state(),
                         standby_during=standby_count(
                             int(r["wall_s"] / 60) + 2))
    #: Fold it into the law's own table, so the sweep and the anchor are ONE
    #: table rather than a table and a footnote -- and so the order column and
    #: the tightness range include the row the gate actually asks for.
    law = out.setdefault("sigma_law", {}).setdefault("rows", [])
    law[:] = [x for x in law if abs(x["interval"] - ANCHOR_INTERVAL) > 1e-15]
    prev = [x for x in law if x["sigma"] > 0.0]
    if prev:
        p = max(prev, key=lambda x: x["interval"])
        r["exponent"] = float(np.log(r["sigma"] / p["sigma"])
                              / np.log(r["interval"] / p["interval"]))
    law.append(r)
    law.sort(key=lambda x: x["interval"])
    graded = [x for x in law if x["sigma"] > 0.0 and "bound_over_measured" in x]
    if graded:
        out["sigma_law"]["bound_holds"] = all(
            x["bound_over_measured"] >= 1.0 for x in graded)
        out["sigma_law"]["tightness"] = [
            min(x["bound_over_measured"] for x in graded),
            max(x["bound_over_measured"] for x in graded)]
        out["sigma_law"]["lag_rate"] = float(
            np.mean([x["lag_rate"] for x in graded]))
        print("  with the anchor folded in, the bound holds at every graded "
              "interval: %s" % out["sigma_law"]["bound_holds"])
        print("  and is loose by %.4fx to %.4fx over %.0fx of interval"
              % (out["sigma_law"]["tightness"][0],
                 out["sigma_law"]["tightness"][1],
                 graded[-1]["interval"] / graded[0]["interval"]))


def stage_constants(out: dict, a) -> None:
    print("=== the constants this tier publishes, and the state they carry ===")
    sl = out.get("slope", {})
    law = out.get("sigma_law", {})
    rate = out.get("rate", {})
    print("  s_seam     = %.6e per K" % sl.get("s_seam", float("nan")))
    print("  C2         = %.6e per K^2" % sl.get("C2", float("nan")))
    print("  lambda_dot = %.6e K/s at a %g s burn"
          % (rate.get("lambda_dot_at_5s", float("nan")), a.burn))
    print("  bound      = s_seam * lag + C2 * lag^2,  lag <= lambda_dot * DT")
    st = out.get("setup", {})
    print()
    print("  probe state: chamber p_c=%.3g Pa T_c=%.1f K; shell marched %.2f s "
          "to %.4f K; ratio pinned at %d; CFL %.2f; gas settled %g s"
          % (RE.P_CHAMBER, RE.T_CHAMBER,
             st.get("marched", {}).get("burn_time_s", float("nan")),
             st.get("marched", {}).get("T_wall_mean_K", float("nan")),
             DECLARED_RATIO, st.get("cfl", float("nan")),
             st.get("settle_seconds", float("nan"))))
    out["constants"] = dict(
        s_seam=sl.get("s_seam"), C2=sl.get("C2"),
        lambda_dot=rate.get("lambda_dot_at_5s"), burn_time=a.burn,
        bound_holds=law.get("bound_holds"), tightness=law.get("tightness"),
        ratio=DECLARED_RATIO, cfl=st.get("cfl"),
        settle_seconds=st.get("settle_seconds"))


# ---------------------------------------------------------------------------
# predictions, registered as numbers and evaluated by the same code
# ---------------------------------------------------------------------------


def predictions(out: dict) -> list[dict]:
    """Registered BEFORE any arm ran; see the tier's log entry for the wording.

    Each is a callable on the results dict so the prose and the evaluation
    cannot drift apart in the permissive direction, which is a standing rule
    here after it happened.
    """
    law = out.get("sigma_law", {})
    rows = [r for r in law.get("rows", []) if r["sigma"] > 0.0]
    ctl = [r for r in law.get("rows", []) if r.get("control")]
    sl = out.get("slope", {})
    tr = out.get("transfer", {})
    rt = out.get("rate", {})
    ra = out.get("ratio", {})
    tk = out.get("trace_rank", {})

    def _p(name, claim, got, ok):
        return dict(name=name, claim=claim, got=got, held=bool(ok))

    P = []
    P.append(_p("P1", "L7/R9 ADMITS on the real graph",
                out.get("r9", {}).get("admits"),
                out.get("r9", {}).get("admits") is True))
    P.append(_p("P2", "the ratio-1 control is EXACTLY zero, not small",
                (ctl[0]["sigma"] if ctl else None),
                bool(ctl) and ctl[0]["sigma"] == 0.0))
    exps = [r["exponent"] for r in rows if np.isfinite(r.get("exponent", np.nan))]
    P.append(_p("P3", "sigma is first order in the interval: every exponent "
                      "in [0.90, 1.10]", exps,
                bool(exps) and all(0.90 <= e <= 1.10 for e in exps)))
    P.append(_p("P4", "the bound with ROCKET constants holds at every graded "
                      "interval", law.get("bound_holds"),
                law.get("bound_holds") is True))
    P.append(_p("P5", "lambda_dot here exceeds CS-11's 11.94 K/s and lies in "
                      "[20, 200] K/s", rt.get("lambda_dot_at_5s"),
                rt.get("lambda_dot_at_5s") is not None
                and 20.0 <= rt["lambda_dot_at_5s"] <= 200.0))
    P.append(_p("P6", "CS-11's OWN constants do NOT bound the rocket's defect",
                tr.get("cs11_worst"), tr.get("cs11_holds") is False))
    P.append(_p("P7", "4x of clock ratio at a fixed interval moves sigma by "
                      "less than 1.05x", ra.get("sigma_move"),
                ra.get("sigma_move") is not None and ra["sigma_move"] < 1.05))
    P.append(_p("P8", "lambda_dot at 0.5 s and 5.0 s of burn differ by more "
                      "than 2x", rt.get("span"),
                rt.get("span") is not None and rt["span"] > 2.0))
    P.append(_p("P9", "the tightness is within an order of magnitude of "
                      "CS-11's, i.e. in [1, 10]", law.get("tightness"),
                law.get("tightness") is not None
                and 1.0 <= law["tightness"][0] and law["tightness"][1] <= 10.0))
    P.append(_p("P10", "s_seam is constant in the interval to better than 1.5x",
                sl.get("s_seam_spread"),
                sl.get("s_seam_spread") is not None
                and sl["s_seam_spread"] < 1.5))
    P.append(_p("P11", "the build repo's rank-1 trace changes the defect by "
                       "more than 1.2x",
                [r["ratio"] for r in tk.get("rows", [])],
                bool(tk.get("rows"))
                and any(abs(np.log(max(r["ratio"], 1e-300))) > np.log(1.2)
                        for r in tk["rows"])))
    #: **P12 was registered AFTER the base gap was measured (0.8801) and BEFORE
    #: its effect on sigma was.**  Saying which is which is the point of
    #: registering them at all.
    tw = out.get("two_way", {})
    P.append(_p("P12", "the two-way referent moves sigma by more than 1.5x at "
                       "some interval",
                [r["ratio"] for r in tw.get("rows", [])],
                bool(tw.get("rows"))
                and any(max(r["ratio"], 1.0 / max(r["ratio"], 1e-300)) > 1.5
                        for r in tw["rows"])))
    return P


# ---------------------------------------------------------------------------


def _signature(a) -> dict:
    """What a later invocation has to match before it may merge into the record.

    The anchor is a separate three-hour invocation, so its numbers land in the
    same file as the core's.  Merging results taken at a DIFFERENT operating
    point would be an invisible mixing of two runs, so the point is recorded and
    checked rather than assumed.
    """
    return dict(burn=a.burn, settle=a.settle, ratio=DECLARED_RATIO,
                cfl=CFLS[0], slope_interval=SLOPE_INTERVAL,
                intervals=list(INTERVALS), anchor=ANCHOR_INTERVAL)


def _load(a) -> dict:
    """Earlier stages' results, so a second invocation ADDS rather than replaces.

    Without this, ``--stages anchor --json out/w314.json`` writes a file holding
    only setup and anchor and the core's three hours are gone.
    """
    if not a.json or not os.path.exists(a.json):
        return {}
    try:
        with open(a.json, encoding="utf-8") as fh:
            old = json.load(fh)
    except Exception as exc:
        print("  could not read %s (%r); starting fresh" % (a.json, exc))
        return {}
    sig = old.get("_signature")
    if sig is not None and sig != _signature(a):
        print("  %s was written at a DIFFERENT operating point; not merging"
              % a.json)
        print("    on disk: %s" % sig)
        print("    now:     %s" % _signature(a))
        return {}
    print("  merging into %s, which already has: %s"
          % (a.json, ", ".join(sorted(k for k in old if not k.startswith("_")
                                      and k not in ("predictions",
                                                    "wall_seconds",
                                                    "standby_during_run")))))
    return old


def _persist(out: dict, a) -> None:
    if not a.json:
        return
    path = os.path.abspath(a.json)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = {k: v for k, v in out.items() if not k.startswith("_")}
    body["_signature"] = _signature(a)
    tmp = path + ".tmp"
    for attempt in range(5):
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(body, fh, indent=2, default=float)
            os.replace(tmp, path)
            return
        except PermissionError:                               # OneDrive holds it
            time.sleep(0.6 * (attempt + 1))
    print("  WARNING: could not persist to %s" % path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", default="core",
                    help="'core', 'all', or a comma list from " + ",".join(STAGES))
    ap.add_argument("--json", metavar="PATH", default=None)
    ap.add_argument("--burn", type=float, default=RE.T_PROBE_BURN)
    ap.add_argument("--burns", type=float, nargs="+",
                    default=[0.5, 1.0, 5.0, 10.0])
    ap.add_argument("--settle", type=float, default=SETTLE)
    ap.add_argument("--dt-scale", type=float, default=1.0e-3,
                    help="the graph's probe cadence for the R9 stage only")
    ap.add_argument("--with-anchor", action="store_true",
                    help="include the 5e-2 s row in sigma_law (3.1 h)")
    a = ap.parse_args(argv)

    want = (list(CORE) if a.stages == "core" else
            list(STAGES) if a.stages == "all" else
            [s.strip() for s in a.stages.split(",")])
    bad = [s for s in want if s not in STAGES]
    if bad:
        ap.error("unknown stage(s): %s" % bad)
    if "setup" not in want and any(s in NEEDS_SETUP for s in want):
        want = ["setup"] + want

    print("=" * 88)
    print("W314 -- the composed defect at 50:1, against CS-11's bound")
    print("stages: %s" % ", ".join(want))
    print("=" * 88)
    out: dict = _load(a)
    t0 = time.perf_counter()
    for s in want:
        print()
        globals()["stage_" + s](out, a)
        _persist(out, a)

    print()
    print("=" * 88)
    print("PREDICTIONS, registered before any arm ran")
    print("=" * 88)
    P = predictions(out)
    for p in P:
        print("  %-4s %-5s %s" % (p["name"], "HELD" if p["held"] else "FAILED",
                                  p["claim"]))
        print("       got: %s" % (p["got"],))
    held = sum(1 for p in P if p["held"])
    print()
    print("  %d of %d held." % (held, len(P)))
    out["predictions"] = P
    out["wall_seconds"] = time.perf_counter() - t0
    out["standby_during_run"] = standby_count(int(out["wall_seconds"] / 60) + 2)
    print("  wall %.0f s; Modern Standby events during the run: %d"
          % (out["wall_seconds"], out["standby_during_run"]))
    _persist(out, a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
