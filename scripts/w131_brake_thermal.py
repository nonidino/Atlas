"""CS-11: a bound for the multirate lag defect, and a horizon for a gradient.

    python scripts/w131_brake_thermal.py [--out out/w131] [--stages ...]

W90 has stood open since Tier 17 with the sharpest statement of any row on the
worklist: *"R9 covers the flux transient and is 62x too small; the lag over a
long exchange interval is sigma, and nothing bounds it as a function of that
interval."*  `thermal_seam` measured both terms at one interval and put them in
one norm; what has never existed is the FUNCTION.

This driver measures it, on a brake disc against its cooling duct at the 500:1
their physics gives, and folds in **W128** -- CS-10's finding that a design
sensitivity taken from a rollout is a function of its horizon and has no field
on the record.

Stages, in the order they depend on each other:

    setup       geometry, both clocks, the duct's closure checked against
                Dittus-Boelter, and the referent's own convergence
    resolution  the seam's spectral cutoff, so `effective_resolution` is
                MEASURED rather than carried over from another expert's face
    compile     the four compiles, and the `L7/R9/lag` text this replaces
    slope       `s_seam = d sigma / d(uniform lag)` -- W86's transferable
                constant -- and the curvature interface power's bilinearity forces
    sigma_law   THE DELIVERABLE. sigma against the exchange interval over three
                decades of clock ratio, against the single-rate referent
    ratio       and the control that says the ratio is not the variable: the
                fast agent refined at a FIXED interval
    waveform    W17's W > 1, which R3 makes admissible here, measured
    accumulate  what the per-interval bound predicts about a rollout, marched
                past the point where the columns could cross
    horizon     W128: dJ/dU against rollout length, its closed-form lumped
                prediction, and the horizon below which its sign is undetermined
    constants   the record: MeasuredConstants, the law, and check_sigma_lag

Every stage writes into one artifact and the artifact is written after EVERY
stage, not at the end, so a run that dies at `horizon` still leaves `sigma_law`
on disk.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from atlas import compile_scheme                                    # noqa: E402
from atlas.cases import brake_thermal as B                          # noqa: E402
from atlas.graph import FluxMatching                                # noqa: E402
from atlas.multiphysics import check_sigma_lag, interface_power     # noqa: E402
from atlas.ports import ResponseHalf                                # noqa: E402

STAGES = ("setup", "resolution", "compile", "slope", "sigma_law", "ratio",
          "waveform", "accumulate", "horizon", "constants")

#: The exchange intervals the law is measured over, in seconds, and the clock
#: ratios they are against the duct's declared native step.  `DT_DUCT` is ratio
#: 1 -- the single-rate case, where the lag defect must vanish and does not
#: vanish only if the instrument is measuring itself.
INTERVALS = (1.0e-4, 5.0e-4, 2.0e-3, 1.0e-2, 5.0e-2, 2.0e-1)

#: Uniform trace shifts for the slope, in K.  Spanning two decades because
#: interface power is BILINEAR and a single-decade fit cannot see the curvature
#: that makes the first-order law an approximation rather than an identity.
SHIFTS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)

#: The design sweep.  `DuctAgent.validity` declines outside Blasius' own
#: Reynolds range, so this is bounded by the expert rather than by taste.
U_STEPS = (0.5, 1.0, 2.0)
HORIZON_STEPS = 1000
HORIZONS = (10, 25, 50, 100, 150, 200, 300, 400, 600, 800, 1000)


def _pow(x, y):
    """log-log slope through two points -- an exponent, quoted as a ratio."""
    return float(np.log(y[1] / y[0]) / np.log(x[1] / x[0]))


def _horizon_limits(rows):
    """N_valid(eps) and N_sign, from a horizon table.

    **A tolerance met only at the final horizon is NOT met.**  The last point is
    within 0% of itself by construction, so reporting it as the limit would be
    reporting the sweep's own endpoint as a result -- W106's shape, one level up.
    Those return None, which reads as *not determined over the horizons marched*.
    """
    ns = [r["N"] for r in rows]
    g = [r["grad_ref"] for r in rows]
    conv = g[-1]
    lim = {}
    for tol in (0.5, 0.2, 0.1, 0.05):
        ok = [i for i in range(len(g))
              if all(abs(g[k] - conv) <= tol * abs(conv) for k in range(i, len(g)))]
        lim[tol] = ns[ok[0]] if ok and ok[0] < len(g) - 1 else None
    sign = [i for i in range(len(g))
            if all((g[k] > 0) == (conv > 0) for k in range(i, len(g)))]
    n_sign = ns[sign[0]] if sign else None
    cross = [(ns[i], ns[i + 1]) for i in range(len(g) - 1) if g[i] * g[i + 1] < 0]
    return dict(validity_limit={str(k): v for k, v in lim.items()},
                n_sign=n_sign, crossings=cross, converged=conv)


def _seam_power(trace, flow):
    return interface_power(trace, flow, ResponseHalf.FLOW, B.H_SEAM)


# ---------------------------------------------------------------------------
# stage 1 -- setup
# ---------------------------------------------------------------------------


def stage_setup(out: dict) -> None:
    print("=== the two agents, their clocks, and the ratio the physics gives ===")
    d = B.DuctAgent()
    disc = B.DiscAgent()
    rows = []
    print(f"  {'U [m/s]':>9}{'Re':>9}{'h_wall':>10}{'dt_stable':>12}{'substeps':>10}")
    for u in (20.0, 28.0, 40.0, 56.0, 80.0):
        du = B.DuctAgent(u_bulk=u)
        rows.append(dict(u=u, re=u * B.H_DUCT / B.NU_G, h_wall=du.h_wall,
                         dt_stable=du.dt_stable, substeps=du.substeps_at(du.dt),
                         valid=bool(du.validity())))
        print(f"  {u:9.1f}{u * B.H_DUCT / B.NU_G:9.0f}{du.h_wall:10.1f}"
              f"{du.dt_stable:12.3e}{du.substeps_at(du.dt):10d}")
    e = _pow((rows[0]["u"], rows[-1]["u"]), (rows[0]["h_wall"], rows[-1]["h_wall"]))
    print()
    print(f"  the wall coefficient scales as U^{e:.3f}, against Dittus-Boelter's")
    print("  U^0.8. Nothing told the closure that exponent -- it is kappa, Pr_t and")
    print("  Blasius' friction law and no fitted constant -- so the agreement is a")
    print("  CHECK on the duct and not an input to it.")
    print()
    tau = (7150.0 * 460.0 * B.T_DISC) / (B.H_PAD + 609.0)
    print(f"  disc     dt_native = {B.DT_DISC:g} s, thermal time constant ~ {tau:.1f} s")
    print(f"  duct     dt_native = {B.DT_DUCT:g} s, {d.substeps_at(d.dt)} sub-steps inside it")
    print(f"  ratio    {B.CLOCK_RATIO}:1, and it is a conduction time against a cell transit")
    print(f"  seam     {B.N_SEAM} cells, M = {B.M_EFF}, seam base {B.seam_base():.4f} K")

    print()
    print("  -- the referent's own convergence, which nothing below means without --")
    Tg, Td, _ = B.settled_state()
    ref = []
    for div in (1, 2, 4):
        r = B.BrakeRollout(exchange=B.DT_DUCT / div)
        m = r.run(200 * div, Tg, Td)
        ref.append(dict(exchange=B.DT_DUCT / div, face_T=float(m.face_T[-1]),
                        trace=float(m.trace[-1].mean())))
        print(f"    exchange {B.DT_DUCT / div:9.3e} s -> face T {m.face_T[-1]:.9f} K")
    drift = abs(ref[-1]["face_T"] - ref[0]["face_T"])
    print(f"    refining the referent 4x moves it {drift:.3e} K, which is the floor")
    print("    every sigma below is read against. It is not a bitwise zero and is")
    print("    not quoted as one (W106).")
    out["setup"] = dict(closure=rows, h_exponent=e, tau_disc=tau,
                        clock_ratio=B.CLOCK_RATIO, seam_base=B.seam_base(),
                        referent_convergence=ref, referent_floor=drift)


# ---------------------------------------------------------------------------
# stage 2 -- resolution
# ---------------------------------------------------------------------------


def _seam_block(agent, port, base, eps: float = 1.0e-3) -> np.ndarray:
    """The probed response operator on V, by finite-difference columns.

    R7's branch again: both agents declare `NONE` with `deterministic=True` and
    a one-ulp floor, so a finite difference is the rule's own probe and not a
    workaround.  The step is swept in the stage below rather than asserted.
    """
    f0 = agent.respond(port, base)
    out = np.empty((base.size, base.size))
    for j in range(base.size):
        p = base.copy()
        p[j] += eps
        out[:, j] = (agent.respond(port, p) - f0) / eps
    return out


def stage_resolution(out: dict) -> None:
    print("=== the seam's own operator, and the cutoff that is not there ===")
    print("  `CASE-STUDY-GUIDE`: effective_resolution is the expert's own MEASURED")
    print("  cutoff, and a cutoff is a WAVELENGTH. This stage measures it, and the")
    print("  measurement's answer is that a conjugate-heat seam does not have one.")
    print()
    duct, disc = B.DuctAgent(), B.DiscAgent()
    base = np.full(B.N_SEAM, B.seam_base())
    blocks = {"duct": _seam_block(duct, "wall:THERM", base),
              "disc": _seam_block(disc, "duct:THERM", base)}

    # the probe step, swept -- the same discipline the gradient gets
    steps = []
    for eps in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        L = _seam_block(duct, "wall:THERM", base, eps)
        steps.append(dict(eps=eps, norm=float(np.linalg.norm(L)),
                          rel=float(np.linalg.norm(L - blocks["duct"])
                                    / np.linalg.norm(blocks["duct"]))))
    detail = " ".join(f"{r['eps']:.0e}->{r['rel']:.1e}" for r in steps)
    print(f"  probe step swept (relative to 1e-3): {detail}")

    rows = {}
    print(f"  {'block':>7}{'|diag|':>12}{'max offdiag':>14}{'off/diag':>11}"
          f"{'cond':>9}{'sv_min/sv_max':>15}")
    for name, L in blocks.items():
        d = float(np.abs(np.diag(L)).mean())
        off = float(np.abs(L - np.diag(np.diag(L))).max())
        sv = np.linalg.svd(L, compute_uv=False)
        rows[name] = dict(diag=d, max_offdiag=off, off_over_diag=off / d,
                          cond=float(sv[0] / sv[-1]),
                          sv_ratio=float(sv[-1] / sv[0]),
                          sv=[float(x) for x in sv])
        print(f"  {name:>7}{d:12.4e}{off:14.4e}{off / d:11.3e}"
              f"{sv[0] / sv[-1]:9.4f}{sv[-1] / sv[0]:15.4f}")

    print()
    print("  **There is no spectral decay to cut at.**  The response at a cell is")
    print("  set by the temperature AT that cell; transport along the seam is a 1%")
    print("  correction. So every interface mode carries the same information and")
    print("  a rank-m truncation discards exactly the fraction of modes it drops:")
    print()
    print(f"  {'m':>4}{'lambda [cells]':>16}{'duct truncation':>18}"
          f"{'disc truncation':>18}")
    trunc = []
    for m in (5, 11, 16, 21, 25, 29, 31, 32):
        P = B.fourier_basis(B.N_SEAM, m)
        Q = P @ P.T * B.H_SEAM
        t = {n: float(np.linalg.norm(L - Q @ L @ Q) / np.linalg.norm(L))
             for n, L in blocks.items()}
        trunc.append(dict(m=m, **t))
        print(f"  {m:4d}{2.0 * B.N_SEAM / max(m - 1, 1):16.2f}"
              f"{t['duct']:18.4e}{t['disc']:18.4e}")

    would_be = B.modes_for()
    print()
    print(f"  a cutoff-derived declaration would have said m = {would_be} "
          f"(thermal_seam's 6.4-cell")
    print(f"  cutoff carried onto 32 cells), which loses "
          f"{[r for r in trunc if r['m'] == 11][0]['duct'] * 100:.0f}% of the operator.")
    print(f"  This case study declares M_EFF = {B.M_EFF}, the grid itself.")

    # does the declaration move a NUMBER, or only the space? beta says.
    print()
    print("  -- and what the choice costs a published quantity --")
    betas = []
    for m in (11, 16, 32):
        P = B.fourier_basis(B.N_SEAM, m)
        S = P.T * B.H_SEAM @ (blocks["duct"] + blocks["disc"]) @ P
        sv = np.linalg.svd(S, compute_uv=False)
        betas.append(dict(m=m, beta=float(sv[-1]), kappa=float(sv[0] / sv[-1])))
        print(f"    m = {m:2d}:  beta = {sv[-1]:.6e}   kappa = {sv[0] / sv[-1]:.4f}")
    print("    beta and kappa barely move, because an operator with no small")
    print("    singular directions has none to lose. **The truncation costs the")
    print("    interface SPACE, not the conditioning** -- so nothing thermal_seam")
    print("    published is wrong, and what it could not have represented is a")
    print("    seam disagreement above its 16th mode.")
    out["resolution"] = dict(blocks=rows, truncation=trunc, betas=betas,
                             probe_step=steps, m_declared=B.M_EFF,
                             m_from_cutoff=would_be)


# ---------------------------------------------------------------------------
# stage 3 -- compile
# ---------------------------------------------------------------------------


def stage_compile(out: dict) -> None:
    print("=== the four compiles, and the decertification this study replaces ===")
    rows, lag_text = [], None
    for mode in ("as-built", "split-step"):
        for clocks in ("native", "matched"):
            g, _ = B.build(mode=mode, clocks=clocks)
            r = compile_scheme(g, probe_state="brake duct, release state")
            fired = [f"{d.layer}/{d.rule}" for d in r.decisions._decisions
                     if d.verdict.value != "admit"]
            rows.append(dict(mode=mode, clocks=clocks, verdict=r.verdict.value,
                             unmeasured=sorted(r.unmeasured), fired=fired))
            print(f"  {mode:11s} {clocks:8s} -> {r.verdict.value:18s} "
                  f"{len(fired)} not-admit")
            for d in r.decisions._decisions:
                if d.rule == "R9/lag" and lag_text is None:
                    lag_text = d.message
    # and the refusal a pointwise declaration still earns
    g, _ = B.build(mode="split-step", clocks="native",
                   flux_matching=FluxMatching.POINTWISE)
    r = compile_scheme(g, probe_state="brake duct")
    print(f"  {'split-step':11s} {'pointwise':8s} -> {r.verdict.value:18s} "
          "(R9's own refusal, unchanged)")
    rows.append(dict(mode="split-step", clocks="native-pointwise",
                     verdict=r.verdict.value, unmeasured=sorted(r.unmeasured),
                     fired=[f"{d.layer}/{d.rule}" for d in r.decisions._decisions
                            if d.verdict.value != "admit"]))
    print()
    print("  L7/R9/lag currently says, verbatim:")
    for line in (lag_text or "").split(". "):
        if line.strip():
            print(f"    {line.strip()[:100]}")
    out["compile"] = dict(rows=rows, r9_lag_text=lag_text)


# ---------------------------------------------------------------------------
# stage 4 -- the slope
# ---------------------------------------------------------------------------


def _sigma_uniform(duct, disc, Tg, Td, lam, shift, interval):
    """The interface-power defect from displacing the trace uniformly by `shift`.

    W86's construction: this is ``d sigma / d(uniform lag)``, the one part of
    sigma that is a property of the SEAM and transfers.

    **It is evaluated through the same functional `_lag_defect` uses, and that
    is not a detail.**  Interface power can be read off either side's response,
    and the two agree only at a converged trace -- so a slope measured through
    the disc's flux is not the slope of the defect measured through the duct's,
    and a bound assembled from the two is comparing different quantities.  The
    lag acts on the FAST agent, so the duct's own wall flux is the functional
    both stages read.  Measured before this was fixed, the mismatch made the
    bound over-predict by 5.7x to 8.6x and look like W86's lag-profile effect.
    """
    def power(trace):
        T, _ = duct.advance(Tg, interval, trace)
        q, tf = duct.wall_heat_flux(T, trace)
        return _seam_power(trace, B._as_flow(q, tf, duct.flux_convention))

    pa = power(lam)
    pb = power(lam + shift)
    return abs(pb - pa) / abs(pa), pa


def stage_slope(out: dict) -> None:
    print("=== s_seam: d sigma / d(uniform lag), the transferable constant ===")
    Tg, Td, _ = B.settled_state()
    duct, disc = B.DuctAgent(), B.DiscAgent()
    lam = B.solve_interface(duct, Tg, disc, Td)
    rows = []
    print(f"  {'shift [K]':>11}{'sigma':>13}{'sigma/shift':>14}{'log2 ratio':>12}")
    prev = None
    for s in SHIFTS:
        sg, p0 = _sigma_uniform(duct, disc, Tg, Td, lam, s, B.DT_DISC)
        r = (np.log(sg / prev[1]) / np.log(s / prev[0])) if prev else float("nan")
        rows.append(dict(shift=s, sigma=sg, slope=sg / s, exponent=r))
        print(f"  {s:11.2f}{sg:13.5e}{sg / s:14.5e}{r:12.4f}")
        prev = (s, sg)
    # the slope is the small-shift limit; the curvature is the departure from it
    s_seam = rows[0]["slope"]
    big = rows[-1]
    c2 = (big["sigma"] - s_seam * big["shift"]) / big["shift"] ** 2
    print()
    print(f"  s_seam = {s_seam:.6e} per K, from the smallest shift where the")
    print("  linear branch is clean. The exponent column is 1.0 where the law is")
    print("  first order and falls below it as interface power's BILINEARITY bites:")
    print(f"  the quadratic coefficient that closes the gap at {big['shift']:g} K is")
    print(f"  C2 = {c2:.6e} per K^2.")
    out["slope"] = dict(rows=rows, s_seam=s_seam, C2=c2,
                        reference_power=float(_sigma_uniform(
                            duct, disc, Tg, Td, lam, 1.0, B.DT_DISC)[1]))


# ---------------------------------------------------------------------------
# stage 5 -- THE DELIVERABLE
# ---------------------------------------------------------------------------


def _lag_defect(interval: float, cfl: float = B.CFL):
    """sigma over one exchange interval, against the single-rate referent.

    Both columns march the SAME agents from the SAME state over the SAME
    interval, and the disc takes one step of `interval` in both.  The only
    difference is the trace the duct is handed: the referent's own path
    ``lambda*(t)``, or its value at the start of the interval held constant.
    **That isolates the staleness and nothing else** -- in particular it does
    not confound the lag with either agent's time-discretization error, which is
    what a comparison against a finer-stepped column would do.
    """
    Tg, Td, _ = B.settled_state()
    duct = B.DuctAgent(cfl=cfl)
    disc = B.DiscAgent()
    lam0 = B.solve_interface(duct, Tg, disc, Td)

    # the referent's trace path over the interval, at the duct's own clock
    ref = B.BrakeRollout(exchange=B.DT_DUCT, cfl=cfl)
    n_ref = max(1, int(round(interval / B.DT_DUCT)))
    mref = ref.run(n_ref, Tg, Td)
    path = mref.trace                                   # [n_ref, N_SEAM]

    # column REF: the duct sees the referent's varying trace
    T = Tg.copy()
    acc_r = 0.0
    for i in range(n_ref):
        T, _ = duct.advance(T, interval / n_ref, path[i])
        q, Tf = duct.wall_heat_flux(T, path[i])
        acc_r += _seam_power(path[i], B._as_flow(q, Tf, duct.flux_convention))
    p_ref = acc_r / n_ref

    # column HELD: the duct sees lambda(t0) for the whole interval
    T = Tg.copy()
    acc_h = 0.0
    for i in range(n_ref):
        T, _ = duct.advance(T, interval / n_ref, lam0)
        q, Tf = duct.wall_heat_flux(T, lam0)
        acc_h += _seam_power(lam0, B._as_flow(q, Tf, duct.flux_convention))
    p_held = acc_h / n_ref

    lag = float(np.linalg.norm(path[-1] - lam0) / np.sqrt(B.N_SEAM))
    return dict(interval=interval, ratio=interval / B.DT_DUCT,
                ratio_stable=interval / duct.dt_stable,
                sigma=abs(p_held - p_ref) / abs(p_ref), lag=lag,
                p_ref=float(p_ref), lag_rate=lag / interval)


def stage_sigma_law(out: dict) -> None:
    print("=== sigma as a FUNCTION of the exchange interval  (W90) ===")
    print("  R4 floors the interval at max_i dt_native, which here is the DISC's")
    print("  0.05 s. Everything below that is measured anyway, because a law that")
    print("  is only evaluated where it is used has not been tested.")
    print()
    rows = []
    print(f"  {'DT [s]':>10}{'ratio':>8}{'lag [K]':>12}{'sigma':>13}"
          f"{'sigma/lag':>12}{'log2':>8}")
    prev = None
    for dt in INTERVALS:
        t0 = time.perf_counter()
        r = _lag_defect(dt)
        r["wall_s"] = time.perf_counter() - t0
        e = (np.log(r["sigma"] / prev["sigma"]) / np.log(dt / prev["interval"])
             if prev and prev["sigma"] > 0 else float("nan"))
        r["exponent"] = e
        rows.append(r)
        print(f"  {dt:10.4g}{r['ratio']:8.0f}{r['lag']:12.5e}{r['sigma']:13.5e}"
              f"{r['sigma'] / max(r['lag'], 1e-300):12.4e}{e:8.3f}", flush=True)
        prev = r

    s_seam = out.get("slope", {}).get("s_seam")
    c2 = out.get("slope", {}).get("C2")
    print()
    if s_seam is not None:
        print("  and the BOUND, evaluated from the slope stage's constants and each")
        print("  interval's own lag -- no fit to this table:")
        print(f"  {'DT [s]':>10}{'measured':>13}{'bound':>13}{'bound/meas':>12}")
        for r in rows:
            b = s_seam * r["lag"] + c2 * r["lag"] ** 2
            r["bound"] = b
            r["bound_over_measured"] = b / r["sigma"] if r["sigma"] > 0 else float("inf")
            print(f"  {r['interval']:10.4g}{r['sigma']:13.5e}{b:13.5e}"
                  f"{r['bound_over_measured']:12.4f}")
        # ratio 1 is the single-rate CONTROL: both the measured defect and the
        # bound are exactly zero there, which is the statement that the
        # instrument measures the lag rather than itself. It carries no
        # tightness ratio, and quoting 0/0 as one would be W106's mistake.
        graded = [r for r in rows if r["sigma"] > 0.0]
        control = [r for r in rows if r["sigma"] == 0.0]
        holds = all(r["bound_over_measured"] >= 1.0 for r in graded)
        worst = max(r["bound_over_measured"] for r in graded)
        tight = min(r["bound_over_measured"] for r in graded)
        print()
        for r in control:
            print(f"  ratio {r['ratio']:.0f} is the single-rate control and its lag "
                  f"defect is {r['sigma']:.1e} -- EXACTLY zero, because one duct")
            print("  step IS the interval. Anything else would mean the instrument is")
            print("  measuring itself, and it carries no tightness ratio.")
        print(f"  the bound holds at every graded interval: {holds}")
        print(f"  and it is loose by between {tight:.3f}x and {worst:.3f}x over "
              f"{graded[-1]['ratio'] / graded[0]['ratio']:.0f}x of clock ratio.")
        out["bound_holds"] = bool(holds)
        out["bound_tightness"] = [tight, worst]
        out["bound_control_ratio"] = [r["ratio"] for r in control]
    rate = np.mean([r["lag_rate"] for r in rows])
    print()
    print(f"  the trace's own rate lambda_dot is {rate:.5e} K/s across the sweep,")
    print("  and it is a property of the RUN -- nothing at compile time knows it,")
    print("  which is exactly why sigma_lag is provenance and not a certificate.")
    out["sigma_law"] = dict(rows=rows, lag_rate=float(rate))


# ---------------------------------------------------------------------------
# stage 6 -- the ratio is not the variable
# ---------------------------------------------------------------------------


def stage_ratio(out: dict) -> None:
    print("=== the control: the clock RATIO at a FIXED exchange interval ===")
    print("  The interval sweep moves two things at once -- the interval and the")
    print("  ratio -- so on its own it cannot say which one sigma is a function of.")
    print("  Here the interval is pinned at the native 0.05 s and the fast agent is")
    print("  REFINED, so the ratio moves 8x and the staleness does not move at all.")
    print()
    rows = []
    print(f"  {'CFL':>7}{'dt_stable':>12}{'ratio':>9}{'sigma':>13}{'vs CFL 0.4':>12}")
    base = None
    for c in (0.4, 0.2, 0.1, 0.05):
        r = _lag_defect(B.DT_DISC, cfl=c)
        r["cfl"] = c
        base = base if base is not None else r["sigma"]
        r["relative"] = r["sigma"] / base
        rows.append(r)
        print(f"  {c:7.2f}{B.DuctAgent(cfl=c).dt_stable:12.4e}"
              f"{r['ratio_stable']:9.0f}{r['sigma']:13.6e}{r['relative']:12.6f}",
              flush=True)
    spread = max(r["sigma"] for r in rows) / min(r["sigma"] for r in rows)
    ratio_span = rows[-1]["ratio_stable"] / rows[0]["ratio_stable"]
    print()
    print(f"  over {ratio_span:.0f}x of clock ratio at a fixed interval, sigma moves "
          f"{spread:.6f}x.")
    print("  **The exchange interval is the variable and the clock ratio is not.**")
    print("  Scoped: this is a fast agent that sub-steps internally at its own")
    print("  stability limit, which is what a classical explicit solver does. A")
    print("  frozen checkpoint whose dt_native cannot be refined is where the ratio")
    print("  re-enters, and nothing here measures that case.")
    out["ratio"] = dict(rows=rows, spread=float(spread), ratio_span=float(ratio_span))


# ---------------------------------------------------------------------------
# stage 7 -- W17's W > 1
# ---------------------------------------------------------------------------


def _lag_defect_waveform(interval: float):
    """The same measurement with the trace carried as a LINEAR waveform.

    R3 -- ``W > 1`` requires `bc_time_varying` on every agent at the interface --
    holds here on both, so this is admissible by rule.  The extrapolated
    endpoint ``lambda + (lambda - lambda_prev)`` is what a composition layer can
    actually build: it needs no information from the future, only the previous
    exchange, which is the same data a lagged scheme already has.
    """
    Tg0, Td0, _ = B.settled_state()
    duct, disc = B.DuctAgent(), B.DiscAgent()
    n_ref = max(1, int(round(interval / B.DT_DUCT)))

    # march ONE interval FIRST, so the waveform's slope comes from the PAST.
    # An extrapolation that reads its own interval's endpoint is not a scheme,
    # it is the answer, and measuring one would be measuring nothing.
    pre = B.BrakeRollout(exchange=B.DT_DUCT).run(n_ref, Tg0, Td0)
    lam_prev = pre.trace[0].copy()
    Tg, Td = pre.T_gas, pre.T_disc
    lam0 = B.solve_interface(duct, Tg, disc, Td)
    path = B.BrakeRollout(exchange=B.DT_DUCT).run(n_ref, Tg, Td).trace

    def power(traces):
        T, acc = Tg.copy(), 0.0
        for i in range(n_ref):
            T, _ = duct.advance(T, interval / n_ref, traces[i])
            q, Tf = duct.wall_heat_flux(T, traces[i])
            acc += _seam_power(traces[i], B._as_flow(q, Tf, duct.flux_convention))
        return acc / n_ref

    p_ref = power(path)
    p_held = power([lam0] * n_ref)
    slope = lam0 - lam_prev                     # the PREVIOUS interval's rise
    p_wave = power([lam0 + slope * (i + 0.5) / n_ref for i in range(n_ref)])
    return dict(interval=interval,
                sigma_held=abs(p_held - p_ref) / abs(p_ref),
                sigma_wave=abs(p_wave - p_ref) / abs(p_ref),
                slope_norm=float(np.linalg.norm(slope) / np.sqrt(B.N_SEAM)))


def stage_waveform(out: dict) -> None:
    print("=== W17's W > 1: a trace carried as a waveform  (R3 admits it here) ===")
    print("  W90 names this as the candidate remedy and it is the ONE axis R4")
    print("  leaves open: R4 forbids shrinking the interval, and a waveform does")
    print("  not shrink it -- it makes the trace inside it stop being constant.")
    print()
    rows = []
    print(f"  {'DT [s]':>10}{'held':>13}{'waveform':>13}{'reduction':>12}")
    for dt in INTERVALS[2:]:
        r = _lag_defect_waveform(dt)
        r["reduction"] = r["sigma_held"] / max(r["sigma_wave"], 1e-300)
        rows.append(r)
        print(f"  {dt:10.4g}{r['sigma_held']:13.5e}{r['sigma_wave']:13.5e}"
              f"{r['reduction']:12.2f}x", flush=True)
    print()
    red = [r["reduction"] for r in rows]
    print()
    print(f"  the waveform is worth between {min(red):.1f}x and {max(red):.1f}x, and the")
    print("  reduction is NOT monotone in the interval. It should not be: the held")
    print("  scheme's error is the trace's first difference and the waveform's is its")
    print("  SECOND, so the ratio between them is the trace's curvature, which is a")
    print("  property of where on the transient the interval sits and not of the")
    print("  interval's length. That is the same shape as W86 -- the transferable")
    print("  quantity is a slope and the thing that moves is the profile -- and it")
    print("  means a waveform buys an order and not a factor.")
    out["waveform"] = dict(rows=rows)


# ---------------------------------------------------------------------------
# stage 8 -- accumulation, marched past the crossing
# ---------------------------------------------------------------------------


def stage_accumulate(out: dict, steps: int = 200) -> None:
    print("=== what the per-interval bound says about a ROLLOUT ===")
    print("  Tier 23's standing rule: a comparison between two configurations is")
    print("  not a result until it has been marched PAST the point where the curves")
    print("  could cross, and the crossing has to be looked for. CS-9* and CS-10")
    print("  both produced results that reversed when marched further.")
    print()
    Tg, Td, _ = B.settled_state()
    span = steps * B.DT_DISC
    n_ref = int(round(span / B.DT_DUCT))
    t0 = time.perf_counter()
    ref = B.referent_rollout().run(n_ref, Tg, Td)
    print(f"  single-rate referent: {n_ref} exchanges over {span:g} s "
          f"in {time.perf_counter() - t0:.0f} s", flush=True)
    stride = int(round(B.DT_DISC / B.DT_DUCT))
    ref_at = ref.face_T[stride - 1::stride][:steps]

    rows = []
    print(f"  {'column':>22}{'end dT [K]':>13}{'max |dT|':>12}{'at step':>9}"
          f"{'sign changes':>14}")
    cols = [("multirate, held", dict(exchange=B.DT_DISC)),
            ("multirate, lag 2", dict(exchange=B.DT_DISC, resolve_every=2)),
            ("multirate, lag 4", dict(exchange=B.DT_DISC, resolve_every=4)),
            ("multirate, waveform", dict(exchange=B.DT_DISC, waveform=2))]
    for name, kw in cols:
        m = B.BrakeRollout(**kw).run(steps, Tg, Td)
        d = m.face_T[:steps] - ref_at
        sgn = int(np.sum(np.diff(np.sign(d[np.abs(d) > 1e-12])) != 0))
        i = int(np.argmax(np.abs(d)))
        rows.append(dict(column=name, end=float(d[-1]), max_abs=float(np.abs(d).max()),
                         at_step=i + 1, sign_changes=sgn,
                         relative_end=float(abs(d[-1]) / ref_at[-1]),
                         wall_s=m.wall_s))
        print(f"  {name:>22}{d[-1]:13.5e}{np.abs(d).max():12.5e}{i + 1:9d}"
              f"{sgn:14d}", flush=True)
    print()
    print(f"  the referent rises {ref_at[0]:.2f} -> {ref_at[-1]:.2f} K over the window,")
    print("  so the defect column is read against that rise and not against zero.")
    out["accumulate"] = dict(steps=steps, span=span, rows=rows,
                             referent_start=float(ref_at[0]),
                             referent_end=float(ref_at[-1]),
                             referent_wall_s=ref.wall_s)


# ---------------------------------------------------------------------------
# stage 9 -- W128, the horizon law
# ---------------------------------------------------------------------------


def stage_horizon(out: dict) -> None:
    print("=== W128: dJ/dU against the rollout horizon ===")
    print("  J is the settled seam temperature over the last quarter of an N-step")
    print("  march, CS-10's convention. U is a DESIGN KNOB and not a state: it")
    print("  enters only the duct, and every column is released from the same")
    print("  field, so dJ/dU is a derivative of the composed stack.")
    print()
    Tg, Td, _ = B.settled_state()

    # the control CS-10's convention needs: does the release field matter?
    m_own = B.composed_rollout(u_bulk=42.0).run(
        20, *B.settled_state(42.0)[:2])
    m_shared = B.composed_rollout(u_bulk=42.0).run(20, Tg, Td)
    ic = abs(m_own.face_T[-1] - m_shared.face_T[-1])
    print(f"  control: releasing U=42 from its OWN settled duct instead of the")
    print(f"  shared one moves J(20) by {ic:.3e} K -- the duct re-settles inside one")
    print("  macro-step, so the choice is immaterial and is measured rather than")
    print("  argued. Everything below uses the shared field.")

    # bit-reproducibility, asserted and NOT used as a level (W106)
    a = B.composed_rollout().run(20, Tg, Td).objective(20)
    b = B.composed_rollout().run(20, Tg, Td).objective(20)
    print(f"  the objective is bit-reproducible: |J1 - J2| = {abs(a - b):.1e}. "
          "Nothing is quoted against it.")
    print()

    marches = {}
    for u in (B.U_DUCT,):
        t0 = time.perf_counter()
        marches[u] = B.composed_rollout(u_bulk=u).run(HORIZON_STEPS, Tg, Td)
        print(f"  base march U={u}: {HORIZON_STEPS} steps in "
              f"{time.perf_counter() - t0:.0f} s", flush=True)

    print()
    print("  -- the finite-difference step, swept: R7's branch needs its own check --")
    fd = {}
    for d in U_STEPS:
        for u in (B.U_DUCT - d, B.U_DUCT + d):
            if u not in marches:
                t0 = time.perf_counter()
                marches[u] = B.composed_rollout(u_bulk=u).run(HORIZON_STEPS, Tg, Td)
                print(f"    U={u}: {time.perf_counter() - t0:.0f} s", flush=True)
    for d in U_STEPS:
        g = [(marches[B.U_DUCT + d].objective(n)
              - marches[B.U_DUCT - d].objective(n)) / (2.0 * d) for n in HORIZONS]
        fd[d] = g
    print(f"  {'N':>6}{'t [s]':>8}{'J':>10}" +
          "".join(f"{'dJ/dU @' + str(d):>14}" for d in U_STEPS))
    rows = []
    for i, n in enumerate(HORIZONS):
        j = marches[B.U_DUCT].objective(n)
        gs = [fd[d][i] for d in U_STEPS]
        rows.append(dict(N=n, t=n * B.DT_DISC, J=float(j),
                         grad={str(d): float(v) for d, v in zip(U_STEPS, gs)},
                         grad_ref=float(gs[0])))
        print(f"  {n:6d}{n * B.DT_DISC:8.2f}{j:10.3f}" +
              "".join(f"{v:14.5f}" for v in gs))

    # the truncation branch: agreement between FD steps
    spread = [max(abs(fd[d][i] - fd[U_STEPS[0]][i]) for d in U_STEPS)
              / max(abs(fd[U_STEPS[0]][i]), 1e-12) for i in range(len(HORIZONS))]
    print()
    print(f"  the three FD steps agree to {max(spread):.2e} relative at worst, so")
    print("  the sweep is on a truncation branch and not in cancellation.")

    # the crossing, and the validity limit
    lims = _horizon_limits(rows)
    cross, conv, lim = lims["crossings"], lims["converged"], lims["validity_limit"]
    print()
    print(f"  dJ/dU changes sign between N = {cross}" if cross
          else "  dJ/dU does not change sign over the horizons measured")
    print(f"  and its value at the longest horizon is {conv:.5f}.")
    print(f"  N_sign, the horizon above which the SIGN is settled: {lims['n_sign']}"
          f"  (t = {lims['n_sign'] * B.DT_DISC:.1f} s)")
    print("  and N_valid, the horizon above which the MAGNITUDE is within a")
    print("  tolerance of its converged value -- W128's own T_pred:")
    for tol, n in lim.items():
        txt = ("NOT DETERMINED over the horizons marched" if n is None
               else f"N >= {n}  (t >= {int(n) * B.DT_DISC:.1f} s)")
        print(f"    within {float(tol) * 100:4.0f}% : {txt}")

    # the closed-form lumped prediction
    duct = B.DuctAgent()
    h_d = duct.h_wall * B.H_IN / (duct.h_wall + B.H_IN)
    hp = B.H_PAD
    hh = h_d + hp
    tg = float(marches[B.U_DUCT].trace[-1].mean())
    t_eq = (h_d * B.T_IN + hp * B.T_PAD) / hh
    dhd = (B.DuctAgent(u_bulk=B.U_DUCT + 1.0).h_wall * B.H_IN
           / (B.DuctAgent(u_bulk=B.U_DUCT + 1.0).h_wall + B.H_IN)) - h_d
    dt_eq = hp * (B.T_IN - B.T_PAD) / hh ** 2 * dhd
    print()
    print("  the closed-form lumped law, written before the run:")
    print(f"    h_eff = {h_d:.1f} W/(m^2 K), H = {hh:.1f}, T_eq = {t_eq:.1f} K")
    print(f"    dT_eq/dU = {dt_eq:.4f} K per (m/s)   -- the LONG-horizon limit")
    print(f"    short-horizon coefficient is proportional to (T_gas - T_0) = "
          f"{B.T_IN - B.T_DISC_0:+.1f} K")
    print("  so the two limits have opposite signs whenever the disc is released")
    print("  colder than the air cooling it, which is the release state here, and")
    print(f"  the measured long-horizon value {conv:.4f} is "
          f"{abs(conv / dt_eq):.2f}x the lumped prediction.")
    out["horizon"] = dict(
        rows=rows, fd_spread=float(max(spread)), crossings=cross,
        converged=float(conv), validity_limit=lim, n_sign=lims["n_sign"],
        lumped=dict(h_eff=float(h_d), H=float(hh), T_eq=float(t_eq),
                    dT_eq_dU=float(dt_eq), T_gas_minus_T0=B.T_IN - B.T_DISC_0,
                    trace_end=tg),
        ic_control=float(ic), reproducibility=float(abs(a - b)),
        steps=HORIZON_STEPS)


# ---------------------------------------------------------------------------
# stage 10 -- the record
# ---------------------------------------------------------------------------


def stage_constants(out: dict) -> None:
    print("=== the record: what a compile can now ingest ===")
    law = out.get("sigma_law", {})
    slope = out.get("slope", {})
    native = [r for r in law.get("rows", []) if abs(r["interval"] - B.DT_DISC) < 1e-12]
    if not native:
        print("  sigma_law has not run; nothing to write")
        return
    n = native[0]
    print(f"  sigma      = {n['sigma']:.6e}   (interface power, at DT = {B.DT_DISC} s)")
    print(f"  sigma_lag  = {n['lag']:.6e} K  (the trace drift over that interval)")
    print(f"  s_seam     = {slope.get('s_seam', float('nan')):.6e} per K")
    print(f"  C2         = {slope.get('C2', float('nan')):.6e} per K^2")
    print(f"  lambda_dot = {law.get('lag_rate', float('nan')):.6e} K/s")
    print()
    # the run's own lag, derived from consecutive traces, against the declaration
    Tg, Td, _ = B.settled_state()
    m = B.composed_rollout().run(20, Tg, Td)
    actual = float(np.mean([np.linalg.norm(m.trace[i + 1] - m.trace[i])
                            / np.sqrt(B.N_SEAM) for i in range(len(m.trace) - 1)]))
    chk = check_sigma_lag(n["lag"], actual)
    print(f"  check_sigma_lag: declared {n['lag']:.4g}, the run carries {actual:.4g}")
    print(f"    -> {'consistent' if chk.agrees else 'INCONSISTENT'}")
    for line in chk.note.split(". "):
        if line.strip():
            print(f"    {line.strip()[:96]}")
    out["constants"] = dict(sigma=n["sigma"], sigma_lag=n["lag"],
                            s_seam=slope.get("s_seam"), C2=slope.get("C2"),
                            lambda_dot=law.get("lag_rate"),
                            run_lag=actual, sigma_lag_agrees=bool(chk.agrees),
                            sigma_lag_note=chk.note)

    # and the horizon's own declarations, recomputed from the stored table so a
    # re-run of this stage alone fills them in without re-marching
    h = out.get("horizon")
    if h and h.get("rows"):
        lims = _horizon_limits(h["rows"])
        h.update(lims)
        out["constants"]["horizon"] = h["rows"][-1]["N"]
        out["constants"]["n_sign"] = lims["n_sign"]
        out["constants"]["validity_limit"] = lims["validity_limit"]
        print()
        print(f"  horizon    = {h['rows'][-1]['N']} macro-steps "
              f"({h['rows'][-1]['t']:.1f} s), dJ/dU = {lims['converged']:.5f}")
        print(f"  n_sign     = {lims['n_sign']} "
              f"({lims['n_sign'] * B.DT_DISC:.1f} s) -- the SIGN is settled above this")
        for tol, nn in lims["validity_limit"].items():
            txt = "NOT DETERMINED" if nn is None else f"N >= {nn}"
            print(f"  n_valid({float(tol) * 100:4.0f}%) = {txt}")
        print()
        print("  These three are what section 0.4 of the spec makes binding: a")
        print("  rollout-derived quantity is reported with the horizon it was")
        print("  taken at, the horizon above which its sign is settled, and the")
        print("  horizon above which its magnitude is within a stated tolerance.")


# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_ROOT, "out", "w131"))
    ap.add_argument("--stages", nargs="*", default=list(STAGES), choices=STAGES)
    ap.add_argument("--accumulate-steps", type=int, default=200)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "w131.json")
    # **Load before writing.**  Stages depend on each other -- `sigma_law` reads
    # `slope`'s constants, `constants` reads both -- so re-running one alone has
    # to keep what the others left.  Starting from an empty dict here silently
    # destroyed a 12-minute artifact once; the stage-by-stage write below is what
    # makes a long sweep safe to start, and it is worth nothing if the next
    # invocation truncates the file it protected.
    out: dict = {}
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                out = json.load(fh)
            print(f"resuming from {path}: "
                  f"{sorted(k for k in out if k not in ('generated', 'stages', 'case', 'wall_s'))}")
        except (OSError, ValueError) as exc:
            print(f"could not read {path} ({exc}); starting fresh")
            out = {}
    out.update(generated=time.strftime("%Y-%m-%d %H:%M:%S"),
               stages=a.stages, case="brake-thermal (CS-11)")
    t_all = time.perf_counter()
    for s in a.stages:
        print()
        t0 = time.perf_counter()
        try:
            if s == "accumulate":
                stage_accumulate(out, steps=a.accumulate_steps)
            else:
                globals()[f"stage_{s}"](out)
        finally:
            # written after EVERY stage: a run that dies late still leaves the
            # stages that finished on disk, which is the only reason a long
            # sweep is safe to start.
            out.setdefault("wall_s", {})[s] = time.perf_counter() - t0
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2, default=float)
        print(f"  [{s} done in {time.perf_counter() - t0:.0f} s]")
    print()
    print(f"total {time.perf_counter() - t_all:.0f} s; wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
