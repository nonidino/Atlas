"""Tier 72 -- the fixed point the implicit step has, and the certified mode on it.

    python scripts/tier72_certified_step.py --out out/racelab21 --stages cylinder,car,learned,control,summary

Section 12's criterion 4 -- *flip to certified and watch the error go to the
classical answer, and see what that cost* -- has never been startable.  W226 said
why: defect correction certifies a FIXED POINT, and the porous column's
`WindowNS` step is explicit, so one call to phi already IS the answer.  On
2026-09-14 the user chose W226's second option, an implicit fluid time step, and
rejected amending section 4.2 to a steady-state mode.

**The step they chose already exists.**  `overset_ns.OversetFlow.step` is
implicit in viscosity and in advection with BDF2 in time, and its momentum
system M x = b is solved ITERATIVELY.  So this tier asks the question that has
to be answered before anything is built on that: does the body-fitted column's
step have a fixed point defect correction can reach, and what does reaching it
cost?

  ``cylinder``  the mechanism on a small composite: does a fixed point exist for
                each candidate sweep, is it reachable, does every cheap map land
                on the classical answer, and does certifying the momentum solve
                certify the whole step.
  ``car``       the same on the real 12-grid car, whose cells are 27 times longer
                than wide and skewed 60 degrees -- the geometry that decides it.
  ``learned``   Poseidon-T as the cheap map on a background window (W288's
                territory).  Reports None, not False, if the checkpoint cannot be
                loaded or run.
  ``control``   the porous column's refusal still stands, and the null element is
                the classical sweep march bitwise.
  ``summary``   every registered prediction judged in code.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse                                                         # noqa: E402
import datetime as dt                                                   # noqa: E402
import json                                                             # noqa: E402
import sys                                                              # noqa: E402
import time                                                             # noqa: E402
import traceback                                                        # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (HERE, os.path.join(HERE, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                      # noqa: E402

from atlas import defect_correction as DC                               # noqa: E402
from atlas.cases import certified_step as CT                            # noqa: E402
from atlas.cases import overset_ns as NS                                # noqa: E402
import tier60_body_fitted_grids as T60                                  # noqa: E402

#: The near-wall spacing Tier 62 measured on the car's body grids, and the
#: constants that decide whether an explicit evaluation of the implicit
#: right-hand side can ever reach its own fixed point.
WALL_SPACING = 0.0014
PICARD_AMPLIFICATION = (0.0125 / 1.5) * 0.004 / WALL_SPACING ** 2       # 17.0

READ_BEFORE_THIS_RUN = (
    "W226 as the user decided it (2026-09-14): the certified mode gets an IMPLICIT time step, "
    "so that a step has a fixed point to certify; section 4.2 is NOT amended to a steady-state "
    "mode, and that option is not to be re-proposed.",
    "overset_ns's own docstring: the step is implicit in viscosity AND in advection (linearised "
    "about u* = 2u^k - u^(k-1)) with BDF2 in time, and it says the implicit advection 'is the "
    "kind of step the live certified mode the user chose will need'.",
    "W271: the solver is first order in time in practice despite BDF2, because the projection's "
    "compact pressure Laplacian and the divergence-of-gradient it corrects are different "
    "operators. That is a statement about the DISCRETISATION's accuracy against the PDE, not "
    "about the algebraic system's fixed point, and section 5.4's rule 2 already carries it.",
    "W288: the body-fitted grids hold 61.0% of the composite's unknowns and a uniform 128x128 "
    "window cannot accept a curvilinear patch, so a learned expert reaches at most 33.4%.",
    "Explored before these predictions were written, ON THE CYLINDER at h = 1/16 (27,716 "
    "unknowns): x* from BiCGSTAB satisfies ||phi(x*) - x*||/||x*|| at 4.0e-11 (picard), 7.2e-12 "
    "(jacobi) and 1.6e-11 (ilu); picard DIVERGES at 19.4 a sweep while ilu contracts at 0.0121 "
    "and jacobi at 0.624; the null, jacobi, detuned and deliberately-wrong cheap maps all "
    "finish with an error to the classical answer between 1.8e-10 and 5.9e-10 while their "
    "inner-call counts run 8, 26, 11 and 86; and a certified step lands within 4.6e-10 "
    "relative of the classical step, taking 0.231 s against 0.169 s.",
    "Also found while exploring, and repaired before these predictions: `contraction` first "
    "took the minimum residual over n sweeps as 'the floor', which calls a sweep still falling "
    "at sweep n floored -- it mislabelled Jacobi; and detuned_psi crashed with 'Factor is "
    "exactly singular', the zero-pivot trap _momentum_precond already carries a retry for.",
    "NOT measured before these predictions: ANY of it on the car's own composite, where the "
    "cells are 27:1 and skewed 60 degrees and the incomplete factor has met a zero pivot "
    "before; and whether Poseidon-T can serve as the cheap map for a LINEAR-SOLVE fixed point.",
)

PREDICTION = (
    {"id": "P1", "claim": "the implicit step HAS a fixed point, and it is the classical answer: "
                          "for every sweep, on the cylinder and on the car, "
                          "||phi(x*) - x*|| / ||x*|| < 1e-8",
     "why": "phi_P(x) - x = P(b - M x) and BiCGSTAB returns x* with ||b - M x*|| <= 1e-10 ||b||, "
            "so the gap is ||P r|| on a residual already at the solver's tolerance. This is "
            "EXISTENCE and says nothing about reachability, which is P2 and P3"},
    {"id": "P2", "claim": "picard -- P = (dt/a0) I, the explicit evaluation of the implicit "
                          "right-hand side -- DIVERGES on the car, growing by more than 5 a sweep",
     "why": "the viscous amplification at the car's near-wall spacing is "
            f"(dt/a0) nu / h^2 = {PICARD_AMPLIFICATION:.1f} before the Laplacian's stencil "
            "factor; the cylinder, whose wall spacing is coarser, already read 19.4"},
    {"id": "P3", "claim": "ilu -- the step's own incomplete factor -- CONTRACTS on the car, at a "
                          "rate below 0.5 a sweep",
     "why": "the incomplete factor approximates M^{-1}, so I - PM is small; the cylinder read "
            "0.0121, and 0.5 is deliberately a weaker bar because the car's cells are 27:1 and "
            "skewed and its factor has met a zero pivot before"},
    {"id": "P4", "claim": "the null element is the classical sweep march BITWISE, on the car",
     "why": "a constant psi makes the inner problem's solution phi(w_k) after one cheap call, so "
            "the iterates are the sweep march itself -- defect_correction's own fact 2, never "
            "checked on this column"},
    {"id": "P5", "claim": "Theorem 1 holds where it is easiest to break: a deliberately WRONG "
                          "cheap map lands on the same classical answer as the null element, "
                          "within a factor of 10, while costing at least twice the inner calls",
     "why": "consistency does not depend on the cheap map's accuracy -- a wrong psi can cost "
            "classical calls, it cannot change what is returned. The cylinder read 1.8e-10 "
            "against 4.5e-10 at 86 inner calls against 8"},
    {"id": "P6", "claim": "certifying the momentum solve certifies the STEP: the certified "
                          "step's state is within 1e-6 relative of the classical step's, on the car",
     "why": "the projection is one factored solve and the interpolation one sparse apply; "
            "neither iterates, so once the momentum answers agree the rest of the step is "
            "deterministic. The bar is looser than P1's because the projection can amplify"},
    {"id": "P7", "claim": "and it does NOT pay: on the car the certified step is SLOWER than the "
                          "classical one, for every cheap map that can be built",
     "why": "BiCGSTAB reaches rtol in 15-20 iterations a component, and defect correction needs "
            "several outer sweeps each costing a matvec and a triangular solve, plus its inner "
            "calls. Tiers 48 and 50 already measured the learned expert not paying in this "
            "project's slot, and section 4.2 says RaceLab must not imply otherwise"},
    {"id": "P8", "claim": "Poseidon-T is not a cheap map for THIS fixed point: as psi it does "
                          "not reduce the outer phi-call count below the null element's",
     "why": "phi is a linear-solve sweep whose Jacobian is I - PM; Poseidon-T is a time-advance "
            "map, a different operator, so its Jacobian fidelity -- which defect_correction's "
            "fact 3 says sets the rate -- is low. Consistent with P5: it still CONVERGES, it "
            "just does not help"},
    {"id": "P9", "claim": "the porous column's refusal still stands: MixedRollout with a "
                          "CERTIFIED window still raises, so the fixed point is a property of "
                          "IMPLICITNESS and not of this tier's code",
     "why": "WindowNS's step is explicit and nothing in this tier touches it. Without this the "
            "tier would be claiming a universal fact where it has measured a conditional one"},
)


OUT = NAME = None
_COL = None                     # the built car column, kept between stages


def configure(out: str) -> None:
    global OUT, NAME
    OUT = os.path.abspath(out)
    NAME = os.path.basename(os.path.normpath(OUT))


def persist(obj) -> str:
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, NAME + ".json")
    tmp = p + ".tmp"

    def write():
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(T60.clean(obj), fh, indent=1)

    T60._retry(write)
    T60._retry(lambda: os.replace(tmp, p))
    return p


def load() -> dict:
    p = os.path.join(OUT, NAME + ".json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def say(*a) -> None:
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# the arms, run on whatever flow they are given
# ---------------------------------------------------------------------------


def _existence(flow) -> dict:
    """P1: is x* a fixed point of each sweep?  Existence, not reachability."""
    out = {}
    for sweep in CT.SWEEPS:
        s = CT.MomentumSystem(flow, sweep=sweep)
        xstar, its, cwall = s.classical()
        gap = float(DC.vector_rms(s.phi(xstar) - xstar))
        scale = float(DC.vector_rms(xstar))
        out[sweep] = {"gap": gap, "scale": scale,
                      "relative_gap": gap / scale if scale > 0 else float("nan"),
                      "bicgstab_iterations": its, "classical_s": cwall,
                      "algebraic_residual": s.algebraic_residual(xstar),
                      "rhs_norm": s.rhs_norm()}
    return out


def _reachability(flow, n: int = 30) -> dict:
    """P2 and P3: which sweeps can actually reach the fixed point they have."""
    out = {}
    for sweep in CT.SWEEPS:
        s = CT.MomentumSystem(flow, sweep=sweep)
        c = CT.contraction(s, n=n)
        c.pop("residuals", None)          # kept out of the record; the shape is in rate/floor
        out[sweep] = c
    return out


def _cheap_maps(flow, sweep: str = "ilu") -> dict:
    """P4, P5: every cheap map against ONE x*, and the null element bitwise."""
    ref_sys = CT.MomentumSystem(flow, sweep=sweep)
    xstar, cits, cwall = ref_sys.classical()

    arms = {}
    for name, factory in (("null", None),
                          ("jacobi", CT.jacobi_psi),
                          ("detuned", CT.detuned_psi),
                          ("wrong", lambda sy: CT.wrong_psi())):
        s = CT.MomentumSystem(flow, sweep=sweep)
        psi = factory(s) if factory is not None else None
        if factory is not None and psi is None:
            arms[name] = {"available": False,
                          "reason": "the incomplete factor is exactly singular at this "
                                    "drop tolerance, even after the tighter retry"}
            continue
        w, rep = CT.certified_momentum(s, psi, reference=xstar, classical_its=cits,
                                       classical_wall_s=cwall)
        d = rep.as_dict()
        d.pop("rows", None)
        d["available"] = True
        arms[name] = d

    # P4: the null element IS the classical sweep march, bitwise
    s_a = CT.MomentumSystem(flow, sweep=sweep)
    r_stop = s_a.flow.rtol * max(s_a.rhs_norm(), 1.0) / np.sqrt(s_a.n)
    w_dc, _ = CT.certified_momentum(s_a, CT.null_psi(), r_stop=r_stop)
    s_b = CT.MomentumSystem(flow, sweep=sweep)
    march = DC.classical_march(s_b.phi, s_b.w0, r_stop=r_stop)
    return {"arms": arms, "r_stop": float(r_stop),
            "classical_iterations": cits, "classical_s": cwall,
            "null_is_classical_march_bitwise": bool(np.array_equal(w_dc, march.state)),
            "null_dc_phi_calls": int(arms["null"]["outer_iterations"]),
            "classical_march_phi_calls": int(march.phi_calls)}


def _whole_step(flow, sweep: str = "ilu") -> dict:
    """P6 and P7: the step taken twice from one state, and what it cost.

    Run TWICE and the second reading kept: the first step after a state is
    loaded refactors the incomplete factor and is several times a typical one,
    so a first reading prices the refactor and not the step.
    """
    out = {}
    for name, factory in (("null", None), ("detuned", CT.detuned_psi)):
        CT.certified_step_report(flow, sweep=sweep, psi_factory=factory)   # warm
        r = CT.certified_step_report(flow, sweep=sweep, psi_factory=factory)
        if r["certified"] is not None:
            r["certified"].pop("rows", None)
        r["pays"] = bool(r["certified_s"] < r["classical_s"])
        r["cost_ratio"] = (r["certified_s"] / r["classical_s"]
                           if r["classical_s"] > 0 else float("nan"))
        out[name] = r
    return out


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def stage_cylinder(res) -> dict:
    flow = NS.cylinder_flow(h=1.0 / 16, ni=128, nj=25, dt=0.025)
    flow.precond = "ilu"
    for _ in range(3):
        flow.step()
    out = {"n_unknowns": int(flow.ov.n_unknowns), "k": int(flow.k),
           "existence": _existence(flow), "reachability": _reachability(flow),
           "cheap_maps": _cheap_maps(flow), "whole_step": _whole_step(flow)}
    res["cylinder"] = out
    _say_column("cylinder", out)
    persist(res)
    return out


def _column(res) -> object:
    global _COL
    if _COL is None:
        from atlas.demo_racelab import bodyfitted as BF
        t0 = time.perf_counter()
        col = BF.BodyFittedColumn()
        rep = col.build(root=HERE)
        res.setdefault("car", {})["build_s"] = time.perf_counter() - t0
        res["car"]["build_report"] = {k: v for k, v in rep.items()
                                      if isinstance(v, (int, float, str, bool))}
        say("  built the car in %.1f s" % res["car"]["build_s"])
        _COL = col
    return _COL


def stage_car(res) -> dict:
    col = _column(res)
    flow = col.flow
    out = res.setdefault("car", {})
    out.update(n_unknowns=int(flow.ov.n_unknowns), t=float(flow.t), k=int(flow.k),
               wall_spacing=WALL_SPACING,
               picard_amplification_predicted=PICARD_AMPLIFICATION,
               existence=_existence(flow), reachability=_reachability(flow),
               cheap_maps=_cheap_maps(flow), whole_step=_whole_step(flow))
    _say_column("car", out)
    persist(res)
    return out


def _say_column(label, out) -> None:
    say("  %s: %d unknowns" % (label, out["n_unknowns"]))
    for sweep, e in out["existence"].items():
        say("    exists  %-7s rel gap %.3e   bicgstab its %s"
            % (sweep, e["relative_gap"], e["bicgstab_iterations"]))
    for sweep, c in out["reachability"].items():
        say("    reach   %-7s contracts %-5s rate %.4g  window %s  floor %.2e  diverged %s"
            % (sweep, c["contracts"], c["rate"], c["rate_window"], c["floor"], c["diverged"]))
    for name, a in out["cheap_maps"]["arms"].items():
        if not a.get("available"):
            say("    psi     %-7s UNAVAILABLE: %s" % (name, a["reason"]))
            continue
        say("    psi     %-7s %-20s outer %3d  inner %4d  err->classical %.3e"
            % (name, a["status"], a["outer_iterations"], a["inner_cheap_calls"],
               a["error_to_classical"]))
    say("    null == classical march bitwise: %s"
        % out["cheap_maps"]["null_is_classical_march_bitwise"])
    for name, w in out["whole_step"].items():
        say("    step    %-7s rel gap %.3e   classical %.3f s  certified %.3f s  pays %s"
            % (name, w["relative_velocity_gap"], w["classical_s"], w["certified_s"], w["pays"]))


def stage_learned(res) -> dict:
    """P8: Poseidon-T as the cheap map.  None, not False, when not measured."""
    out = {"attempted": True, "available": None, "reason": None}
    try:
        from atlas.cases import car_windows as CW
        from atlas.cases.poseidon import load_expert
        col = _column(res)
        flow = col.flow
        wins = CW.tile(flow.ov, order="row")
        if not len(wins):
            out.update(available=False, reason="no admissible 128x128 background window")
            res["learned"] = out
            persist(res)
            return out
        t0 = time.perf_counter()
        expert = load_expert(device="cpu", threads=1)
        out["load_s"] = time.perf_counter() - t0
        wins = list(wins)[:2]
        out["windows"] = [w.name if hasattr(w, "name") else str(i)
                          for i, w in enumerate(wins)]
        out["n_windows_used"] = len(wins)

        sysm = CT.MomentumSystem(flow, sweep="ilu")
        xstar, cits, cwall = sysm.classical()
        calls = [0]

        def poseidon_psi(w):
            w = np.asarray(w, dtype=float)
            u, v = w[0].copy(), w[1].copy()
            for win in wins:
                bu = CW.extract(flow.ov, win, u)
                bv = CW.extract(flow.ov, win, v)
                au, av = expert.step(np.ascontiguousarray(bu),
                                     np.ascontiguousarray(bv), flow.dt)
                u = CW.scatter_into(flow.ov, win, u, np.asarray(au))
                v = CW.scatter_into(flow.ov, win, v, np.asarray(av))
            calls[0] += 1
            return np.stack([u, v])

        t0 = time.perf_counter()
        one = poseidon_psi(sysm.w0)
        out["one_call_s"] = time.perf_counter() - t0
        out["one_call_finite"] = bool(np.all(np.isfinite(one)))

        s = CT.MomentumSystem(flow, sweep="ilu")
        w, rep = CT.certified_momentum(s, poseidon_psi, alpha=0.5, k_max=12, m_max=12,
                                       reference=xstar, classical_its=cits,
                                       classical_wall_s=cwall)
        d = rep.as_dict()
        d.pop("rows", None)
        out.update(available=True, report=d, psi_calls_made=calls[0])
        null_outer = res.get("car", {}).get("cheap_maps", {}).get(
            "arms", {}).get("null", {}).get("outer_iterations")
        out["null_outer_iterations"] = null_outer
        out["reduces_outer_calls"] = (None if null_outer is None
                                      else bool(rep.outer_iterations < null_outer))
        say("    poseidon %s outer %d (null %s)  inner %d  err->classical %.3e  %.2f s/call"
            % (rep.status, rep.outer_iterations, null_outer, rep.inner_cheap_calls,
               rep.error_to_classical, out["one_call_s"]))
    except Exception as exc:                                    # pragma: no cover
        out.update(available=None,
                   reason="%s: %s" % (type(exc).__name__, str(exc)[:300]),
                   traceback=traceback.format_exc()[-1500:])
        say("    poseidon NOT MEASURED: %s" % out["reason"])
    res["learned"] = out
    persist(res)
    return out


def stage_control(res) -> dict:
    """P9: the porous column still refuses, so this tier's claim stays conditional."""
    from atlas.cases import racelab_switch as RS
    out = {}
    try:
        RS.MixedRollout(assignment=RS.Mode.CERTIFIED.value)
        out["raises"] = False
        out["message"] = None
    except Exception as exc:
        out["raises"] = True
        out["exception"] = type(exc).__name__
        out["message"] = str(exc)[:300]
    out["explicit_step_named"] = bool(out.get("message")
                                      and "explicit" in out["message"].lower())
    say("    porous CERTIFIED raises: %s (%s)" % (out["raises"], out.get("exception")))
    res["control"] = out
    persist(res)
    return out


def judge(res) -> dict:
    v = {}
    cyl, car = res.get("cylinder"), res.get("car")
    both = [c for c in (cyl, car) if c and c.get("existence")]
    if len(both) == 2:
        v["P1"] = all(e["relative_gap"] < 1e-8
                      for c in both for e in c["existence"].values())
    if car and car.get("reachability"):
        p = car["reachability"]["picard"]
        v["P2"] = bool(p["diverged"] and p["rate"] > 5.0)
        i = car["reachability"]["ilu"]
        v["P3"] = bool(i["contracts"] and i["rate"] < 0.5)
    if car and car.get("cheap_maps"):
        cm = car["cheap_maps"]
        v["P4"] = bool(cm["null_is_classical_march_bitwise"])
        n, wr = cm["arms"].get("null"), cm["arms"].get("wrong")
        if n and wr and n.get("available") and wr.get("available"):
            en, ew = n["error_to_classical"], wr["error_to_classical"]
            ok_err = (np.isfinite(en) and np.isfinite(ew) and en > 0 and ew > 0
                      and 0.1 <= ew / en <= 10.0)
            ok_cost = wr["inner_cheap_calls"] >= 2 * n["inner_cheap_calls"]
            v["P5"] = bool(ok_err and ok_cost)
    if car and car.get("whole_step"):
        ws = car["whole_step"]
        v["P6"] = all(w["relative_velocity_gap"] < 1e-6 for w in ws.values())
        v["P7"] = all(not w["pays"] for w in ws.values())
    L = res.get("learned")
    if L and L.get("available") is True and L.get("reduces_outer_calls") is not None:
        v["P8"] = bool(not L["reduces_outer_calls"])
    C = res.get("control")
    if C:
        v["P9"] = bool(C["raises"])
    return v


def stage_summary(res) -> dict:
    verdicts = judge(res)
    res["verdicts"] = verdicts
    res["verdicts_missing"] = sorted({p["id"] for p in PREDICTION} - set(verdicts))
    persist(res)
    for p in PREDICTION:
        say("%-3s %-6s %s" % (p["id"], verdicts.get(p["id"]), p["claim"][:88]))
    if res["verdicts_missing"]:
        say("NOT JUDGED (None, not False):", res["verdicts_missing"])
    held = sum(1 for x in verdicts.values() if x)
    say("%d of %d judged predictions held" % (held, len(verdicts)))
    return verdicts


STAGES = ("cylinder", "car", "learned", "control", "summary")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default=",".join(STAGES))
    args = ap.parse_args(argv)
    configure(args.out)
    res = load()
    if "prediction" not in res:
        res.update(tier=72, prediction=list(PREDICTION),
                   prediction_recorded_at=dt.datetime.now().isoformat(timespec="seconds"),
                   read_before_this_run=list(READ_BEFORE_THIS_RUN),
                   machine_state=T60.machine_state())
        persist(res)
    elif [p["id"] for p in res["prediction"]] != [p["id"] for p in PREDICTION]:
        raise SystemExit("the record's prediction differs from this file's")
    for st in [s.strip() for s in args.stages.split(",") if s.strip()]:
        if st not in STAGES:
            raise SystemExit(f"unknown stage {st!r}; stages are {STAGES}")
        say("stage", st, "...")
        t0 = time.perf_counter()
        try:
            {"cylinder": stage_cylinder, "car": stage_car, "learned": stage_learned,
             "control": stage_control, "summary": stage_summary}[st](res)
        except Exception:
            err = load()
            err.setdefault("stage_errors", {})[st] = traceback.format_exc()[-3000:]
            persist(err)
            say(f"stage {st} FAILED")
            raise
        res.setdefault("stage_wall_s", {})[st] = time.perf_counter() - t0
        res.get("stage_errors", {}).pop(st, None)
        if res.get("stage_errors") == {}:
            res.pop("stage_errors", None)
        persist(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
