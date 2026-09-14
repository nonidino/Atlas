"""Tier 52 -- the envelope check (W222), and PoC 3 RaceLab phase 2: the switch.

Two things, and the first is why the second is trustworthy.

**The envelope check (W222).**  CS-19 marched 600 macro-steps with the machine
motoring and nothing said so: `MachineAgent.validity` already declared the
condition -- *"a generator can only push current into the battery while its
back-EMF exceeds the open-circuit voltage"* -- the disk declared its induction
clamp, the fluid window declared its cell-Reynolds bound, and **none of the
three was consulted**.  `RaceRollout._absorb` now consults all of them, once, at
the end of every macro-step, and `enforce` defaults ON.

**Phase 2, the switch.**  Per-window classical / learned / certified, with the
per-window one-step error that needs no global referent and the per-call cost
with each column at its own best thread count.

Stages, each persisted to ``out/racelab2/racelab2.json``:

  ``envelope``   the repair and its controls: `machine_for_host` at ``x = 1``
                 must reproduce `machine_for_rotor` exactly, the crossover
                 sweep, and the declined/repaired pair from ONE settled field.
  ``spinup``     the settled release state, and how many macro-steps of the
                 transient the experts decline -- which is the second thing the
                 check found, because CS-19 released every arm from a uniform
                 freestream and did not say so.
  ``scaling``    the over-determined scaling: one window spans 2.0 length units
                 by geometry, so the composition layer's own exchange cadence
                 asks the checkpoint for 1/32 of its native lead.
  ``windows``    section 5.2: per-window one-step error against the classical
                 expert on the same input state, the error-vs-lead curve, and
                 the per-call cost with each column at its own best threads.
  ``certified``  defect correction per window, with the identity as psi -- W76's
                 null replacement -- as the control that says what the
                 checkpoint contributed.
  ``families``   section 4.3: which families have a learned option and which do
                 not, with the reason for each.
  ``assign``     section 5.3: mixed marches against the all-classical column.

Nothing is downloaded (the hub is offline and Poseidon-T comes from the local
cache), no machine is rented, and NeuberNet is not loaded.
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

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert torch.backends.cuda.matmul.allow_tf32 is False
assert torch.backends.cudnn.allow_tf32 is False

from atlas import compiler as C                                         # noqa: E402
from atlas.cases import ground_effect as GE                             # noqa: E402
from atlas.cases import powertrain as PT                                # noqa: E402
from atlas.cases import racelab as RL                                   # noqa: E402
from atlas.cases import racelab_switch as SW                            # noqa: E402
from atlas.cases import vehicle_march as VM                             # noqa: E402
from atlas.cases import wake_array as WA                                # noqa: E402

OUT = os.path.join(HERE, "out", "racelab2")
CACHE = os.path.join(OUT, "cache")
#: W258: this record was measured on the column whose outlet is pinned, and
#: `racelab.OUTFLOW` became the repaired one later (Tier 59); every march here
#: names the column its record describes.
OUTFLOW = "pinned"

#: **The inflow the machine is sized for, and where the number comes from.**
#: CS-19 measured the turbine's settled ring velocity at 0.6691530373612168 with
#: the machine `powertrain` declares -- the configuration `MGU.validity`
#: declines.  `machine_for_host` is referred to that measurement, which makes
#: the sizing ONE Picard step from Tier 51's answer rather than a fixed point,
#: and stage ``spinup`` reports the residual instead of iterating in silence.
U_DUCT_TIER51 = 0.6691530373612168

SPIN_STEPS = 240
MIXED_STEPS = 40


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
    path = os.path.join(OUT, "racelab2.json")
    tmp = path + ".tmp"

    def write():
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(clean(res), fh, indent=1)

    _retry(write)
    _retry(lambda: os.replace(tmp, path))
    return path


def load_res():
    path = os.path.join(OUT, "racelab2.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_field(name, **arrays):
    """`np.savez` appends `.npz`, so the temp name must ALREADY end in it."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name + ".npz")
    tmp = os.path.join(CACHE, name + ".tmp.npz")
    _retry(lambda: np.savez(tmp[:-4], **arrays))
    _retry(lambda: os.replace(tmp, path))
    return path


def load_field(name):
    path = os.path.join(CACHE, name + ".npz")
    return np.load(path) if os.path.isfile(path) else None


def machine_state() -> dict:
    import subprocess
    out: dict = {}
    try:
        r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe",
                            "/FO", "CSV"], capture_output=True, text=True,
                           timeout=30)
        rows = [ln for ln in r.stdout.splitlines()[1:] if ln.strip()]
        out["python_processes_running"] = len(rows)
        out["python_processes"] = rows[:8]
    except Exception as exc:                                     # pragma: no cover
        out["python_processes_running"] = None
        out["tasklist_failed"] = str(exc)
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Battery).BatteryStatus"],
            capture_output=True, text=True, timeout=30)
        out["battery_status_raw"] = r.stdout.strip()
        out["on_mains"] = (r.stdout.strip() == "2")
    except Exception as exc:                                     # pragma: no cover
        out["on_mains"] = None
    return out


# ---------------------------------------------------------------------------
# stage 1 -- the envelope check, and the repair (W222)
# ---------------------------------------------------------------------------


def stage_envelope() -> dict:
    """The check, the repair, and the three controls that make it evidence."""
    out: dict = {}

    # -- control 1: the similarity at x = 1 must reproduce the parent EXACTLY
    a = VM.machine_for_rotor(RL.HOST_ROTOR_WIDTH)
    b = RL.machine_for_host(RL.U_HOST_REF, RL.U_HOST_REF)
    same = {}
    for k in a:
        for f in ("resistance", "k_e", "k_t", "r_total"):
            va, vb = getattr(a[k], f, None), getattr(b[k], f, None)
            if va is not None:
                same["%s.%s" % (k, f)] = bool(va == vb)
    out["control_x_equals_1_reproduces_machine_for_rotor"] = {
        "all_bitwise": all(same.values()), "per_field": same,
        "why": "the similarity is ABOUT the reference inflow, so at that inflow "
               "it must be the identity or it is a fit and not a similarity",
    }

    # -- the crossover, closed form and swept
    els = VM.machine_for_rotor(RL.HOST_ROTOR_WIDTH)
    k_e = float(els["MGU"].k_e)
    r_tot = float(els["MGU"].r_total)
    lam = 7.5
    slope = 2.0 * lam * k_e / RL.HOST_ROTOR_WIDTH
    out["crossover"] = {
        "k_e": k_e, "R_total": r_tot, "V_oc": float(PT.V_OC),
        "d_back_emf_d_u": slope,
        "u_at_which_the_machine_stops_generating": float(PT.V_OC) / slope,
        "formula": "I = (k_e omega - V_oc) / R_total with omega = lambda u / r, "
                   "so the current changes sign at u = V_oc / (2 lambda k_e / A)",
    }
    sweep = []
    for u in (0.55, 0.60, 0.6692, 0.70, 0.80, 0.8933, 0.9225, 1.00):
        ring = np.full(WA.ROTOR_CELLS, u)
        r_, _e, disk = VM.operating_point(ring, width=RL.HOST_ROTOR_WIDTH,
                                          scale=RL.HOST_ROTOR_WIDTH)
        sweep.append({"u": u, "induction": float(r_.induction),
                      "current": float(r_.current),
                      "rotor_valid": bool(r_.rotor_valid),
                      "MGU_validity": bool(els["MGU"].validity(np.full(1, r_.omega)))})
    out["sweep_with_the_declared_machine"] = sweep

    # -- the repair at the duct's inflow, and the induction control
    els2 = RL.machine_for_host(U_DUCT_TIER51)
    ring = np.full(WA.ROTOR_CELLS, U_DUCT_TIER51)
    ring_ref = np.full(WA.ROTOR_CELLS, RL.U_HOST_REF)
    r_ref, _e, _d = VM.operating_point(ring_ref, width=RL.HOST_ROTOR_WIDTH,
                                       scale=RL.HOST_ROTOR_WIDTH)
    r_bad, _e, _d = VM.operating_point(ring, width=RL.HOST_ROTOR_WIDTH,
                                       scale=RL.HOST_ROTOR_WIDTH)
    r_fix, _e, _d = VM.operating_point(ring, width=RL.HOST_ROTOR_WIDTH,
                                       scale=RL.HOST_ROTOR_WIDTH, elements=els2)
    out["repair"] = {
        "u_ref": RL.U_HOST_REF, "u_host": U_DUCT_TIER51,
        "x": U_DUCT_TIER51 / RL.U_HOST_REF,
        "k_e_before": k_e, "k_e_after": float(els2["MGU"].k_e),
        "R_total_before": r_tot, "R_total_after": float(els2["MGU"].r_total),
        "derivation": "k_e ~ 1/x from the back-EMF being invariant under "
                      "omega ~ x; R ~ 1/x^3 from the torque having to follow "
                      "the disk's x^2 with k_t ~ 1/x -- W199's own two "
                      "conditions, applied to the inflow instead of the width",
        "at_u_ref_declared_machine": {
            "induction": float(r_ref.induction), "current": float(r_ref.current),
            "rotor_valid": bool(r_ref.rotor_valid)},
        "at_u_host_declared_machine": {
            "induction": float(r_bad.induction), "current": float(r_bad.current),
            "rotor_valid": bool(r_bad.rotor_valid)},
        "at_u_host_host_sized_machine": {
            "induction": float(r_fix.induction), "current": float(r_fix.current),
            "rotor_valid": bool(r_fix.rotor_valid)},
        "induction_matches_the_reference": bool(
            abs(r_fix.induction - r_ref.induction) < 1e-12),
        "induction_gap": abs(float(r_fix.induction) - float(r_ref.induction)),
    }
    return out


# ---------------------------------------------------------------------------
# stage 2 -- the spin-up: a release state the experts stand behind
# ---------------------------------------------------------------------------


def stage_spinup() -> dict:
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    u, v, rep = RL.settled_field(steps=SPIN_STEPS, host_inflow=U_DUCT_TIER51,
                                 outflow=OUTFLOW)
    rep["wall_s"] = time.perf_counter() - t0
    save_field("settled", u=u, v=v)
    rep["picard"] = {
        "sized_for": U_DUCT_TIER51,
        "settled_at": rep["u_rotor_at_the_release_state"],
        "relative_residual": abs(rep["u_rotor_at_the_release_state"]
                                 - U_DUCT_TIER51) / U_DUCT_TIER51,
        "why_one_step": "the machine is sized for an inflow that its own thrust "
                        "then changes, so the sizing is a fixed point.  ONE "
                        "Picard step is taken from CS-19's measurement and the "
                        "residual is reported rather than iterated away",
    }
    # the control: the SAME settled field with the machine CS-19 declared
    d = load_field("settled")
    try:
        RL.march(d["u"], d["v"], steps=2, join_coupling="lagged", outflow=OUTFLOW)
        rep["control_tier51_machine_from_this_field"] = {
            "declined": False,
            "why": "the check did not fire, which would mean the repair is "
                   "not what separates the two columns"}
    except RL.EnvelopeDeclined as exc:
        rep["control_tier51_machine_from_this_field"] = {
            "declined": True, "who": list(exc.report["_declined"]),
            "message": str(exc)[:300]}
    m = RL.march(d["u"], d["v"], steps=4, join_coupling="lagged",
                 host_inflow=U_DUCT_TIER51, outflow=OUTFLOW)
    rep["repaired_marches_clean"] = {
        "outside_steps": m.notes["outside_the_envelope_steps"],
        "enforce": m.notes["enforce"],
        "envelope_at_the_end": m.notes["envelope_at_the_end"],
        "settled": m.settled(0.5),
    }
    return rep


# ---------------------------------------------------------------------------
# stage 3 -- the scaling, over-determined
# ---------------------------------------------------------------------------


def stage_scaling(stack) -> dict:
    return stack.scaling_report()


# ---------------------------------------------------------------------------
# stage 4 -- section 5.2, the per-window numbers
# ---------------------------------------------------------------------------


def stage_windows(stack, u, v) -> dict:
    out: dict = {}
    out["one_macro_step"] = SW.one_step_errors(stack, u, v, 1)
    out["error_vs_lead"] = SW.error_vs_lead(stack, u, v, (1, 2, 4, 8, 16))
    out["per_call_cost"] = SW.per_call_cost(stack, u, v, n_macro=8)
    return out


# ---------------------------------------------------------------------------
# stage 5 -- the certified mode, with W76's null replacement as the control
# ---------------------------------------------------------------------------


def stage_certified(stack, u, v, windows=("F40", "F01")) -> dict:
    out: dict = {"windows": {}}
    for w in windows:
        row = {}
        for tag, use in (("with_the_checkpoint", True), ("null_identity", False)):
            t0 = time.perf_counter()
            r = SW.certified_window(stack, u, v, w, use_learned=use)
            r["wall_s"] = time.perf_counter() - t0
            row[tag] = r
        a, b = row["with_the_checkpoint"], row["null_identity"]
        row["what_the_checkpoint_bought_in_classical_calls"] = (
            b["phi_calls"] - a["phi_calls"])
        row["classical_calls"] = {"with": a["phi_calls"], "null": b["phi_calls"]}
        out["windows"][w] = row
    out["note"] = (
        "psi is the checkpoint shrunk by alpha, and the null replaces it with "
        "the IDENTITY -- W76's null replacement.  Whatever the checkpoint "
        "column does that the identity column also does is not the "
        "checkpoint's.  Tier 48 measured that difference at 13 classical calls "
        "against a 69-call spread from perturbing the same weights by 3%")
    out["theorem_1"] = (
        "the limit is the classical map's own fixed point whatever psi does, "
        "so what is measured here is the PRICE of getting there and never the "
        "accuracy of the answer")
    return out


# ---------------------------------------------------------------------------
# stage 6 -- section 4.3, which families have a learned option
# ---------------------------------------------------------------------------


def stage_families(u, v) -> dict:
    g, _aux = RL.build(u, v)
    tab = SW.family_switch_table(g)
    res = C.compile_scheme(g)
    tab["graph_verdict"] = res.verdict.value
    return tab


# ---------------------------------------------------------------------------
# stage 7 -- section 5.3, mixed marches
# ---------------------------------------------------------------------------


def stage_assign(stack, u, v) -> dict:
    return SW.switch_report(
        u, v, ["upper-learned", "wake-learned", "learned"], stack=stack,
        steps=MIXED_STEPS, host_inflow=U_DUCT_TIER51, enforce=False,
        join_coupling="lagged", outflow=OUTFLOW)


STAGES = ("envelope", "spinup", "scaling", "windows", "certified",
          "families", "assign")


def main(argv):
    want = STAGES
    for i, a in enumerate(argv):
        if a == "--stages" and i + 1 < len(argv):
            want = tuple(x.strip() for x in argv[i + 1].split(","))
    res = load_res()
    t0 = time.perf_counter()
    res["meta"] = {
        "tier": 52, "phase": 2,
        "torch_threads_at_start": torch.get_num_threads(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "u_duct_tier51": U_DUCT_TIER51,
        "spin_steps": SPIN_STEPS, "mixed_steps": MIXED_STEPS,
        "machine_state_at_start": machine_state(),
    }

    if "envelope" in want:
        print("1. the envelope check and its repair (W222)", flush=True)
        res["envelope"] = stage_envelope()
        e = res["envelope"]
        print("   x=1 reproduces machine_for_rotor bitwise: %s"
              % e["control_x_equals_1_reproduces_machine_for_rotor"]["all_bitwise"],
              flush=True)
        print("   the machine stops generating below u = %.4f"
              % e["crossover"]["u_at_which_the_machine_stops_generating"],
              flush=True)
        r = e["repair"]
        print("   at the duct's u=%.4f: declared machine valid=%s (I=%+.4f), "
              "host-sized valid=%s (I=%+.4f)"
              % (r["u_host"], r["at_u_host_declared_machine"]["rotor_valid"],
                 r["at_u_host_declared_machine"]["current"],
                 r["at_u_host_host_sized_machine"]["rotor_valid"],
                 r["at_u_host_host_sized_machine"]["current"]), flush=True)
        print("   induction matches the reference: %s (gap %.3g)"
              % (r["induction_matches_the_reference"], r["induction_gap"]),
              flush=True)
        persist(res)

    if "spinup" in want:
        print("2. the spin-up, and the release state the experts admit", flush=True)
        res["spinup"] = stage_spinup()
        s = res["spinup"]
        print("   %d macro-steps outside (steps %s to %s), all %s"
              % (s["macro_steps_outside"], s["first_macro_step_outside"],
                 s["last_macro_step_outside"],
                 list(s["which_experts_declined"])), flush=True)
        print("   release state admitted: %s; Picard residual %.4g"
              % (s["release_state_is_admitted"],
                 s["picard"]["relative_residual"]), flush=True)
        print("   control -- the CS-19 machine on this field declines: %s"
              % s["control_tier51_machine_from_this_field"]["declined"],
              flush=True)
        persist(res)

    d = load_field("settled")
    if d is None:
        print("   (no settled field; run --stages spinup first)", flush=True)
        u = np.full((RL.RNY, RL.RNX), GE.U_INF)
        v = np.zeros((RL.RNY, RL.RNX))
    else:
        u, v = d["u"], d["v"]

    stack = None
    if any(s in want for s in ("scaling", "windows", "certified", "assign")):
        stack = SW.WindowStack()

    if "scaling" in want:
        print("3. the scaling, over-determined by the window's geometry", flush=True)
        res["scaling"] = stage_scaling(stack)
        s = res["scaling"]
        print("   one window spans %.3f length units -> scaling time %.4f"
              % (s["window_span_length_units"], s["scaling_time"]), flush=True)
        for tag in ("exchange", "macro_step", "native"):
            print("   %-11s lead %.6g = %.4g x native"
                  % (tag, s[tag]["lead"], s[tag]["lead_over_native"]), flush=True)
        persist(res)

    if "windows" in want:
        print("4. section 5.2 -- the per-window numbers", flush=True)
        res["windows"] = stage_windows(stack, u, v)
        w = res["windows"]
        o = w["one_macro_step"]
        print("   one macro-step: median error %.4f (min %.4f max %.4f)"
              % (o["median"], o["min"], o["max"]), flush=True)
        for r in w["error_vs_lead"]["rows"]:
            print("      %2d macro-steps (lead %.4gx native): median %.4f  "
                  "classical/learned %.3f"
                  % (r["n_macro"], r["lead_over_native"], r["median"],
                     r["classical_over_learned"] or float("nan")), flush=True)
        c = w["per_call_cost"]
        print("   each column at its own best threads: classical/learned = %.3f"
              % (c["ratio_each_at_its_own_best"] or float("nan")), flush=True)
        persist(res)

    if "certified" in want:
        print("5. the certified mode, with W76's null beside it", flush=True)
        res["certified"] = stage_certified(stack, u, v)
        for wname, row in res["certified"]["windows"].items():
            a, b = row["with_the_checkpoint"], row["null_identity"]
            print("   %s: checkpoint %s in %d classical calls; identity %s in "
                  "%d -- the checkpoint bought %d"
                  % (wname, a["status"], a["phi_calls"], b["status"],
                     b["phi_calls"],
                     row["what_the_checkpoint_bought_in_classical_calls"]),
                  flush=True)
        persist(res)

    if "families" in want:
        print("6. section 4.3 -- which families have a learned option", flush=True)
        res["families"] = stage_families(u, v)
        print("   " + res["families"]["the_fact"], flush=True)
        persist(res)

    if "assign" in want:
        print("7. section 5.3 -- mixed marches against the all-classical column",
              flush=True)
        res["assign"] = stage_assign(stack, u, v)
        for tag, row in res["assign"]["columns"].items():
            print("   %-14s ledger %s  %.4f s/step  speed %.3fx  rms err %.4g"
                  % (tag, row["ledger"], row["s_per_macro_step"],
                     row["speed_ratio_vs_all_classical"],
                     row["rms_field_error_vs_all_classical"]), flush=True)
        persist(res)

    res["elapsed_seconds"] = time.perf_counter() - t0
    print("wrote", persist(res), "in %.1f s" % res["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")
    main(sys.argv)
