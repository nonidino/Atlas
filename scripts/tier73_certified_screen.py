"""Tier 73 -- the certified mode on the screen, and what it costs over a march.

    python scripts/tier73_certified_screen.py --out out/racelab22 --stages switch,trajectory,page,summary

Tier 72 built the certified mode's mechanism and priced ONE step.  Section 12's
criterion 4 asks for something a person does: *flip to certified and watch the
error go to the classical answer, and see what that cost.*  This puts the switch
on the body-fitted dashboard and measures the two things a single step cannot
answer -- whether a certified MARCH tracks the classical one, and whether the
rest of the demo still works while it is on.

  ``switch``      the live column flipped classical -> certified -> verify, with
                  section 5.2's row recorded on every step, and a knob moved
                  while certified to see that criterion 2 survives criterion 4.
  ``trajectory``  two columns released from the SAME settled state, one
                  classical and one certified, marched side by side -- with a
                  positive control, because a gap that is small only because
                  nothing moved is not a result.
  ``page``        the page carries what the record measured.
  ``summary``     every registered prediction judged in code.
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
from atlas.demo_racelab import bodyfitted as BF                         # noqa: E402
import tier60_body_fitted_grids as T60                                  # noqa: E402

#: Steps per arm.  Enough for a median with a spread on each, and short enough
#: that two columns can be built and marched inside a few minutes.
N_STEPS = 24

READ_BEFORE_THIS_RUN = (
    "Tier 72: the body-fitted column's implicit step has a fixed point (its momentum system's "
    "own answer), the step's incomplete factor reaches it at 0.007719 a sweep on the car, and a "
    "certified step landed 1.163e-12 relative from the classical step. P8 was NOT measured: "
    "Poseidon-T refused the call at a lead of 0.0250 against its native 0.1 (W294).",
    "W295: this column is ONE fluid expert (Tier 65), so there is no per-window step whose "
    "answer could be certified separately, and section 5.2's per-window certified telemetry has "
    "nothing to attach to.",
    "Explored before these predictions were written, ON THE RUNNING DASHBOARD at port 8014: the "
    "switch flips, `learned` is refused showing BOTH blocks, and with verify on the page drew "
    "outer iterations 2, inner cheap calls 2, residual 6.213e-9, error to classical 6.204e-9, "
    "status converged, classical step 0.654 s (n=20), certified step 0.717 s (n=20), certified "
    "+ verify step 1.034 s (n=19), cost of the mode 1.10x and cost with verify 1.58x.",
    "Two defects found by opening the page and not by any test: the panel read "
    "`arms[comparing]`, so while the mode was classical it drew the CLASSICAL arm's 0.516 s "
    "under a 'certified step' label; and the cost row said '20 this arm' counting the classical "
    "arm's steps. Both were repaired before these predictions were registered.",
    "Also found while writing the tests: `cylinder_flow` starts impulsively, and the certified "
    "sweep does NOT converge through that transient -- it returns fallback_unconverged and the "
    "next step blows up. The real column never marches from rest either, which is W293.",
    "NOT measured before these predictions: whether a certified MARCH tracks the classical one "
    "over many steps rather than one; whether every step of such a march converges; whether the "
    "certificate's residual bounds the measured error step after step; and whether a knob still "
    "reaches its subsystem while the certified mode is on.",
)

PREDICTION = (
    {"id": "Q1", "claim": f"every one of {N_STEPS} certified steps on the live column converges, "
                          "and with verify on the measured error to the classical answer stays "
                          "below 1e-6 on every one of them",
     "why": "the fixed point is the momentum system's own answer and the settled car reached it "
            "in 2 outer iterations; what is untested is whether that holds step after step "
            "rather than once"},
    {"id": "Q2", "claim": "the certificate is never violated: on every verified step the measured "
                          "error is at most 10 times the reported residual",
     "why": "the certificate is ||w - w*|| <= Theta r with Theta a property of the CLASSICAL map; "
            "the one step observed on the page read error 6.204e-9 against residual 6.213e-9, so "
            "Theta is about 1 there. Ten is the falsifiable line, and Theta is NOT guaranteed to "
            "be 1 away from that state"},
    {"id": "Q3", "claim": "a certified march TRACKS the classical one: after "
                          f"{N_STEPS} steps from the same settled state the two velocity fields "
                          "differ by less than 1e-6 relative, while the field itself has moved "
                          "by more than 1e-3 over the same steps",
     "why": "each step is certified to about 1e-8 and the accumulation over 24 steps is the "
            "thing no single-step measurement can see. **The second clause is the positive "
            "control**: a gap that is small only because nothing moved is not a result, and "
            "this vault has already published one of those"},
    {"id": "Q4", "claim": "the mode's own price and the verification's price are different "
                          "numbers, both above 1, and the verification is the larger",
     "why": "verify runs a full classical solve beside the certified one every step. The page "
            "read 1.10x and 1.58x before this run, so this is a re-measurement with replicates "
            "rather than a discovery -- what it adds is the spread"},
    {"id": "Q5", "claim": "criterion 2 survives criterion 4: moving the ambient temperature knob "
                          "while the certified mode is ON still moves the marching coolant return",
     "why": "the mode changes the fluid's momentum solve and nothing in the circuit, so it "
            "should not reach the loop -- but Tier 71's whole lesson was a knob that REPORTED a "
            "response the march did not have, and nothing has checked this combination"},
    {"id": "Q6", "claim": "the classical mode is bit-for-bit what it was before the switch "
                          "existed: with the mode classical the flow's momentum_solver is None, "
                          "and a step is identical to one taken with the attribute never set",
     "why": "the switch is a seam, not a rewrite. If flipping back to classical left anything "
            "installed, every classical number this demo reports would be about a different "
            "solver than the one it names"},
)


OUT = NAME = None
_COL = None


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


def _build(label: str) -> BF.BodyFittedColumn:
    t0 = time.perf_counter()
    col = BF.BodyFittedColumn()
    col.build(root=HERE)
    say("  built %s in %.1f s (%d unknowns, t = %.3f)"
        % (label, time.perf_counter() - t0, col.flow.ov.n_unknowns, col.flow.t))
    return col


def _column(res) -> BF.BodyFittedColumn:
    global _COL
    if _COL is None:
        _COL = _build("the column")
    return _COL


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def stage_switch(res) -> dict:
    col = _column(res)
    out = {"n_unknowns": int(col.flow.ov.n_unknowns)}

    # Q6's control FIRST, before anything is installed: a classical step taken
    # with the attribute never touched is the baseline everything else moves from
    assert col.flow.momentum_solver is None
    col.set_mode("classical")
    for _ in range(N_STEPS):
        col.step()
    out["classical_solver_is_none"] = col.flow.momentum_solver is None

    col.set_mode("certified")
    col.set_verify(False)
    plain = []
    for _ in range(N_STEPS):
        r = col.step()
        plain.append(dict(r["certified"]))
    out["certified_plain"] = {
        "steps": len(plain),
        "all_converged": all(p["status"] in ("converged", "fallback_converged")
                             for p in plain),
        "statuses": sorted({p["status"] for p in plain}),
        "outer": [p["outer_iterations"] for p in plain],
        "inner": [p["inner_cheap_calls"] for p in plain],
        "residual_max": max(p["residual"] for p in plain),
        "error_reported_with_verify_off": [p["error_to_classical"] for p in plain
                                           if p["error_to_classical"] is not None],
    }

    col.set_verify(True)
    ver = []
    for _ in range(N_STEPS):
        r = col.step()
        ver.append(dict(r["certified"]))
    errs = [p["error_to_classical"] for p in ver]
    res_ = [p["residual"] for p in ver]
    ratios = [e / r for e, r in zip(errs, res_) if r and e is not None]
    out["certified_verified"] = {
        "steps": len(ver),
        "all_converged": all(p["status"] in ("converged", "fallback_converged")
                             for p in ver),
        "statuses": sorted({p["status"] for p in ver}),
        "error_max": max(errs), "error_min": min(errs),
        "residual_max": max(res_),
        "error_over_residual_max": max(ratios) if ratios else None,
        "classical_iterations_seen": sorted({tuple(p["classical_iterations"])
                                             for p in ver}),
        "outer": [p["outer_iterations"] for p in ver],
    }
    out["cost"] = col.cost()

    # Q5 -- criterion 2 while criterion 4 is on
    before = col.telemetry().get("coolant_return_t")
    knob = col.set_knob("ambient_t", 318.0)
    for _ in range(3):
        col.step()
    after = col.telemetry().get("coolant_return_t")
    col.set_knob("ambient_t", 300.0)
    for _ in range(2):
        col.step()
    out["knob_while_certified"] = {
        "knob": "ambient_t", "mode": col.mode,
        "responded": list(knob.get("responded") or []),
        "coolant_return_before": before, "coolant_return_after": after,
        # nan != nan is True; a broken probe must not read as a working knob
        "moved": bool(before is not None and after is not None
                      and np.isfinite(before) and np.isfinite(after)
                      and before != after),
        "delta": (after - before) if (before is not None and after is not None) else None,
    }

    col.set_mode("classical")
    out["back_to_classical_solver_is_none"] = col.flow.momentum_solver is None
    res["switch"] = out
    _say_switch(out)
    persist(res)
    return out


def _say_switch(out) -> None:
    p, v = out["certified_plain"], out["certified_verified"]
    say("  certified, verify off : %d steps, all converged %s, outer %s, residual<=%.3e"
        % (p["steps"], p["all_converged"], sorted(set(p["outer"])), p["residual_max"]))
    say("    errors reported with verify OFF: %d (must be 0)"
        % len(p["error_reported_with_verify_off"]))
    say("  certified, verify on  : %d steps, all converged %s, error %.3e..%.3e, "
        "error/residual <= %.4f"
        % (v["steps"], v["all_converged"], v["error_min"], v["error_max"],
           v["error_over_residual_max"] or float("nan")))
    c = out["cost"]
    say("  cost: mode %s, with verify %s"
        % (None if c["ratio_mode_only"] is None else round(c["ratio_mode_only"], 3),
           None if c["ratio_with_verify"] is None else round(c["ratio_with_verify"], 3)))
    for a in ("classical", "certified", "certified+verify"):
        r = c["arms"][a]
        if r["n"]:
            say("    %-18s n=%2d  median %.3f s  (%.3f-%.3f)"
                % (a, r["n"], r["median_s"], r["lo_s"], r["hi_s"]))
    k = out["knob_while_certified"]
    say("  knob while certified: %s -> moved %s (%.6f K)"
        % (k["knob"], k["moved"], k["delta"] if k["delta"] is not None else float("nan")))


def stage_trajectory(res) -> dict:
    """Q3: does a certified MARCH track the classical one?

    **Each arm is re-spun rather than continued.**  Two columns are built and
    released from the same settled state, because continuing one march into the
    other measures the continuation and not the mode.
    """
    a = _build("arm A (classical)")
    start_u = a.flow.U.copy()
    start_v = a.flow.V.copy()
    a.set_mode("classical")
    for _ in range(N_STEPS):
        a.step()
    ua, va = a.flow.U.copy(), a.flow.V.copy()

    b = _build("arm B (certified)")
    same_start = bool(np.array_equal(b.flow.U, start_u)
                      and np.array_equal(b.flow.V, start_v))
    b.set_mode("certified")
    b.set_verify(False)
    for _ in range(N_STEPS):
        b.step()
    ub, vb = b.flow.U.copy(), b.flow.V.copy()

    def rms(u, v):
        return float(DC.vector_rms(np.stack([u, v])))

    gap = rms(ua - ub, va - vb)
    scale = rms(ua, va)
    # the POSITIVE CONTROL: how far the field moved over the same steps.  A gap
    # that is small only because nothing happened is not a result.
    moved = rms(ua - start_u, va - start_v)
    out = {"steps": N_STEPS, "same_start": same_start,
           "gap": gap, "scale": scale, "relative_gap": gap / scale if scale else None,
           "moved": moved, "relative_move": moved / scale if scale else None,
           "gap_over_move": gap / moved if moved else None,
           "fingerprint_a": a._fingerprint[:12], "fingerprint_b": b._fingerprint[:12]}
    res["trajectory"] = out
    say("  same start: %s" % same_start)
    say("  after %d steps: gap %.3e, field moved %.3e, gap/move %.3e"
        % (N_STEPS, gap, moved, out["gap_over_move"] or float("nan")))
    say("  relative gap %.3e   relative move %.3e"
        % (out["relative_gap"], out["relative_move"]))
    persist(res)
    return out


def stage_page(res) -> dict:
    """Criterion 5: the numbers on the page come from somewhere recorded."""
    static = os.path.join(HERE, "atlas", "demo_racelab", "static", "bodyfitted.html")
    text = open(static, encoding="utf-8").read()
    labels = ("outer iterations", "inner cheap calls", "residual",
              "error to classical", "cost of the mode", "cost with verify",
              "BiCGSTAB iterations")
    out = {"labels_present": {k: (k in text) for k in labels},
           "sends_mode": 'kind: "mode"' in text,
           "sends_verify": 'kind: "verify"' in text,
           "reads_compared_arm_by_name": "A[cost.comparing]" not in text,
           "names_the_not_per_window_reason": "notPerWindow" in text,
           "learned_refusal_names_both_blocks": bool(
               "128" in BF.LEARNED_REFUSED and "61.0%" in BF.LEARNED_REFUSED
               and "0.1" in BF.LEARNED_REFUSED
               and "out of distribution" in BF.LEARNED_REFUSED)}
    out["all_labels"] = all(out["labels_present"].values())
    res["page"] = out
    say("  labels %s, sends mode/verify %s/%s, arms by name %s, both blocks named %s"
        % (out["all_labels"], out["sends_mode"], out["sends_verify"],
           out["reads_compared_arm_by_name"], out["learned_refusal_names_both_blocks"]))
    persist(res)
    return out


def judge(res) -> dict:
    v = {}
    S, T, P = res.get("switch"), res.get("trajectory"), res.get("page")
    if S:
        p, ver = S["certified_plain"], S["certified_verified"]
        v["Q1"] = bool(p["all_converged"] and ver["all_converged"]
                       and ver["error_max"] < 1e-6)
        r = ver.get("error_over_residual_max")
        v["Q2"] = bool(r is not None and r <= 10.0)
        c = S["cost"]
        m, w = c.get("ratio_mode_only"), c.get("ratio_with_verify")
        v["Q4"] = bool(m is not None and w is not None and m > 1.0 and w > m)
        v["Q5"] = bool(S["knob_while_certified"]["moved"])
        v["Q6"] = bool(S["classical_solver_is_none"]
                       and S["back_to_classical_solver_is_none"]
                       and not p["error_reported_with_verify_off"])
    if T:
        v["Q3"] = bool(T["same_start"] and T["relative_gap"] is not None
                       and T["relative_gap"] < 1e-6
                       and T["relative_move"] is not None
                       and T["relative_move"] > 1e-3)
    if P:
        v["page"] = bool(P["all_labels"] and P["sends_mode"] and P["sends_verify"]
                         and P["reads_compared_arm_by_name"])
    return v


def stage_summary(res) -> dict:
    verdicts = judge(res)
    res["verdicts"] = verdicts
    res["verdicts_missing"] = sorted({p["id"] for p in PREDICTION} - set(verdicts))
    persist(res)
    for p in PREDICTION:
        say("%-3s %-6s %s" % (p["id"], verdicts.get(p["id"]), p["claim"][:86]))
    if res["verdicts_missing"]:
        say("NOT JUDGED (None, not False):", res["verdicts_missing"])
    judged = {k: x for k, x in verdicts.items() if k.startswith("Q")}
    say("%d of %d judged predictions held" % (sum(1 for x in judged.values() if x),
                                              len(judged)))
    return verdicts


def _return_t(col):
    t = col.telemetry().get("coolant_return_t")
    return None if t is None else float(t)


def stage_knobhorizon(res) -> dict:
    """Q5's diagnosis, recorded BESIDE its verdict and not in place of it.

    Q5 was judged False and stays judged False.  This says what it actually
    measured.  `vehicle_march.N_FLUID_PER_COOLANT` is **400**: the coolant
    circuit sub-cycles once every four hundred fluid steps, and Q5's probe took
    **three**.  The loop never ran, so the return temperature was `None` and
    nothing could have moved it -- in EITHER mode.

    Q5's `why` argued about the certified mode; its failure had nothing to do
    with the mode, and **the tier registered it with no control**: the same
    probe in the classical mode was never run, and it is what would have shown
    that at once.  Both are run here -- the short probe in the classical mode
    (the control that must fail identically) and a probe marched past a real
    sub-cycle boundary in the certified mode (the measurement Q5 wanted).
    """
    import atlas.cases.vehicle_march as VM

    per = int(VM.N_FLUID_PER_COOLANT)
    col = _build("the column for the horizon probe")
    out = {"n_fluid_per_coolant": per, "q5_probe_steps": 3,
           "why_q5_read_nothing": (
               f"the coolant circuit sub-cycles once every {per} fluid steps and Q5's "
               f"probe took 3, so the loop never ran and the return was None in either "
               f"mode; `moved` reported False where it should have reported None")}

    # -- the control Q5 never had: the SAME short probe, classically ------
    col.set_mode("classical")
    for _ in range(3):
        col.step()
    b0 = _return_t(col)
    col.set_knob("ambient_t", 318.0)
    for _ in range(3):
        col.step()
    a0 = _return_t(col)
    out["classical_short_probe"] = {
        "before": b0, "after": a0,
        "moved": _moved(b0, a0),
        "note": "the control: if this also reads None, the failure is the probe's horizon"}
    col.set_knob("ambient_t", 300.0)

    # -- the measurement Q5 wanted: past a real sub-cycle boundary --------
    say("  marching to the first coolant sub-cycle (%d steps) ..." % per)
    t0 = time.perf_counter()
    while col.union.steps_on < per + 2:
        col.step()
    b1 = _return_t(col)
    say("    reached step %d, return %s (%.0f s)"
        % (col.union.steps_on, b1, time.perf_counter() - t0))

    col.set_mode("certified")
    knob = col.set_knob("ambient_t", 318.0)
    target = col.union.steps_on + per + 2
    say("  knob moved while CERTIFIED; marching to the next sub-cycle ...")
    t0 = time.perf_counter()
    while col.union.steps_on < target:
        col.step()
    a1 = _return_t(col)
    say("    reached step %d, return %s (%.0f s)"
        % (col.union.steps_on, a1, time.perf_counter() - t0))

    out["certified_full_horizon"] = {
        "mode": col.mode, "responded": list(knob.get("responded") or []),
        "before": b1, "after": a1, "moved": _moved(b1, a1),
        "delta": (a1 - b1) if (b1 is not None and a1 is not None) else None,
        "coolant_rows": len(col.union.coolant),
    }
    out["diagnosis_holds"] = bool(
        out["classical_short_probe"]["moved"] is None
        and out["certified_full_horizon"]["moved"] is True)
    res["knobhorizon"] = out
    say("  classical short probe moved: %s (None = could not be read)"
        % out["classical_short_probe"]["moved"])
    say("  certified full horizon moved: %s, delta %s K"
        % (out["certified_full_horizon"]["moved"],
           out["certified_full_horizon"]["delta"]))
    say("  diagnosis holds: %s" % out["diagnosis_holds"])
    persist(res)
    return out


def _moved(before, after):
    """Three states and never two: None for *could not be read*.

    `nan != nan` is True, so a broken probe reads as a working knob unless both
    readings are required finite -- and a reading that is absent is **not** a
    knob that did not move.  Q5's instrument collapsed those two into False.
    """
    if before is None or after is None:
        return None
    if not (np.isfinite(before) and np.isfinite(after)):
        return None
    return bool(before != after)


STAGES = ("switch", "trajectory", "page", "knobhorizon", "summary")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default=",".join(STAGES))
    args = ap.parse_args(argv)
    configure(args.out)
    res = load()
    if "prediction" not in res:
        res.update(tier=73, prediction=list(PREDICTION), n_steps=N_STEPS,
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
            {"switch": stage_switch, "trajectory": stage_trajectory,
             "page": stage_page, "knobhorizon": stage_knobhorizon,
             "summary": stage_summary}[st](res)
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
