"""Tier 54 -- the traced car, re-sized and re-marched.  Tier 56 -- the drawn car.

    python scripts/tier54_traced_car.py --out out/racelab5
    python scripts/tier54_traced_car.py --out out/racelab5 --stages arms,slow,gate

**``--out`` is REQUIRED, and that is the first thing this file says.**  It used
to write to ``out/racelab4`` unconditionally, so re-running it for a new car
would have overwritten Tier 54's committed record of the car before it.  A
default that clobbers a committed result is a default nobody should be one
keystroke away from.  The record is ``<out>/<basename of out>.json`` --
``out/racelab4/racelab4.json`` for Tier 54, ``out/racelab5/racelab5.json`` for
Tier 56.

The car stopped being a hand-drawn arrangement of plates on 2026-09-13 and
became a TRACED silhouette: a CC0 Formula One side view rasterised, thresholded
and mapped isotropically into the cell box.  That moved every body, moved both
wheels, and moved the radiator duct's band from y 44..76 down to y 18..50 --
because a real Formula One body TAPERS behind the cockpit and the old band put
the duct's roof where a real engine cover has no metal.  **Later the same day
the user drew the car by hand** (`atlas/cases/car_geometry.json`, Tier 55), and
every marched number in ``out/racelab4`` then described a vehicle that no
longer existed.  Tier 56 re-runs this file into ``out/racelab5`` for it.

What does NOT change between cars: the settle fraction, the five arms, the
gate's thresholds (`racelab.GATE`, every one inherited from CS-18), and
`vehicle_march.receiver_balances`.  Anything that moves, moves because the car
did.

**The HORIZON can move, and that is a finding rather than a convenience.**
CS-19's 600 macro-steps were not available to the traced car: it breached the
window expert's declared cell-Reynolds bound of 8 at macro-step 554.  The arms
therefore run at the longest horizon the car stays INSIDE its declared
envelopes for, chosen by the probe below, and every comparison with an earlier
tier at a different horizon says so.

**The machine has to be re-sized, and that is a stage rather than a constant.**
Tier 53's W228 found that sizing a machine for the inflow at ONE INSTANT buys a
few hundred macro-steps and then the margin is consumed: size for the horizon's
MINIMUM instead.  `U_DUCT` is therefore MEASURED here, not written down -- stage
``size`` marches the horizon with ``enforce=False``, records ``u_rotor`` every
macro-step, and reports the minimum; stage ``verify`` marches the chosen
horizon ENFORCED at that value before the arms are committed to.

**The horizon is CHOSEN by the probe, and the first version of this script got
that wrong in a way worth recording.**  It read only the ``u_rotor`` band out of
the probe, gated the arms on an eighty-step enforced march, and lost a
twenty-five-minute referent arm to a decline at macro-step 552 -- while the
probe's own result already said the envelope would object on **46** macro-steps.
Two recorded lessons at once: *march a repair as far as the failure it repairs*,
and a gate that reads a different quantity from the one that predicts the
failure is not a gate.  So ``verify`` marches the WHOLE horizon the arms will
use, and ``arms`` refuses to run at any other.

**Tier 56 applies the same lesson once more, to the probe itself.**  The probe
runs at the RELEASE state's sizing, and the arms do not: they run at the
horizon's minimum.  A decline of the MACHINE in the probe is a decline of a
machine the arms never use, and choosing the horizon from it would cut the arms
short for a reason the re-sizing removes.  So when the probe's first decline
names anything other than the fluid, a second unenforced probe is marched AT
the sizing the arms will use and the horizon is chosen from that one.  Both
probes are recorded, and so is the horizon the Tier 54 procedure alone would
have chosen.

**A settled field belongs to a geometry** (`racelab.geometry_fingerprint`).  The
spin-up stores the fingerprint of the car it settled around inside
``settled.npz``, and every stage that releases from it refuses to run when the
car built NOW is a different car.  The car is edited with a mouse, and a
directory name does not change when it is.

Stages, each persisted to ``<out>/<name>.json``:

  ``spinup``  settle the car's field from the freestream, enforce=False, and
              record what the check would have said.
  ``size``    the ``u_rotor`` band, the minimum to size for, AND the horizon the
              car stays inside its declared envelopes for.
  ``verify``  that whole horizon, ENFORCED, before the arms are committed to.
  ``arms``    the five arms at the verified horizon with enforce=True.
  ``slow``    J2 on the coolant clock -- P4 with its null, Tier 53's recipe
              imported rather than retyped.  Tier 54 did not run it, so its P4
              passed without a control.
  ``gate``    P1 and P7 measured at the release state, and all seven clauses
              judged with every threshold read out of `racelab.GATE`.
  ``compare`` this run beside an earlier one (``--compare-with``), clause by
              clause, with the two horizons named.

A stage that raises is RECORDED before the exception propagates: a run that
dies at macro-step 552 of an arm leaves the step, the reason and the arms that
finished in the record rather than only in a scrolled-away console.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import datetime as dt                                                   # noqa: E402
import json                                                             # noqa: E402
import sys                                                              # noqa: E402
import time                                                             # noqa: E402
import traceback                                                        # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np                                                      # noqa: E402
import torch                                                            # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
assert torch.backends.cuda.matmul.allow_tf32 is False
assert torch.backends.cudnn.allow_tf32 is False

from atlas.cases import racelab as RL                                   # noqa: E402
from atlas.cases import vehicle_march as VM                             # noqa: E402

#: Set by `configure`, from ``--out``.  There is no default on purpose -- see
#: the module docstring.
OUT: str = ""
CACHE: str = ""
NAME: str = ""
TIER53 = os.path.join(HERE, "out", "racelab3", "racelab3.json")

HORIZON = 600
SETTLE_FRAC = 0.25
THREADS = 2                    # W227: bitwise the one-thread answer, and fastest
VERIFY_STEPS = 80
#: Macro-steps of margin between the probe's first decline and the horizon.
HORIZON_MARGIN = 24
HORIZON_FLOOR = 60

ARMS = (
    ("referent", dict(join_coupling="tight")),
    ("repeat", dict(join_coupling="tight")),
    ("all_lagged", dict(join_coupling="lagged")),
    ("null_J3", dict(join_coupling="tight", null="J3")),
    ("null_J1", dict(join_coupling="tight", null="J1")),
)

#: **Tier 54's prediction, recorded before its stage ``size`` ran.**
PREDICTION = (
    "u_rotor in the traced duct is DIFFERENT from Tier 53's 0.6692, because "
    "the duct moved from y 44..76 to y 18..50 -- nearer the floor, where the "
    "ground-effect flow is faster -- and the body above it changed shape. "
    "Direction not predicted.",
    "u_rotor still FALLS over the horizon, as it did in Tier 53 (6.3%), "
    "because CS-19 section 6.2's finding that this flow does not settle is a "
    "property of the decomposition and not of the old car's shape.",
    "Sizing for the horizon's MINIMUM admits all 600 macro-steps, as W228's "
    "procedure did for the old car.  If it does not, the procedure is "
    "insufficient rather than mis-applied, and that is the informative "
    "outcome.",
    "P5 (the repeat floor) is bitwise, because nothing about determinism "
    "depends on the car's shape.",
    "The aerodynamic coefficients MOVE A LOT -- the traced car has 22 plates "
    "against 11, an airbox, and a floor that no longer runs through both "
    "wheels -- so downforce and drag are not comparable with Tier 53's and "
    "this tier does not pretend they are.",
)

#: **Tier 56's prediction, for the car the user drew -- recorded 2026-09-13,
#: before its stage ``spinup`` ran.**
PREDICTION_DRAWN = (
    "u_rotor at the release state is DIFFERENT from the traced car's 0.667914. "
    "The duct band is the same (y 18..50), but both device planes moved 14 "
    "cells upstream (the core's to x = 280, the turbine's to x = 392) and the "
    "body around the duct is a different shape.  Direction not predicted.",
    "The drawn car leaves the FLUID envelope EARLIER than the traced car's "
    "macro-step 554, so its horizon is SHORTER than Tier 54's 530.  The reason "
    "offered: it stands four plates nearly normal to the stream -- RW_ENDPLATE "
    "(61.5 cells at -89.5 degrees), DIFF_EXIT (27.0 at -90.0), ROLL_HOOP (21.1 "
    "at 88.6) and FW_TRAILING (10.6 at -98.4) -- and a porous plate across the "
    "stream accelerates the flow past its ends more than an inclined plate of "
    "the same chord does.",
    "u_rotor still FALLS over the horizon (CS-19 section 6.2: this flow does "
    "not settle), so sizing for the release state would decline and sizing "
    "for the horizon's minimum is again what admits the arms.",
    "P1 passes unchanged (L7/R9 and nothing else joined; nothing disjoint), "
    "because the refusal is about the clocks and not the car; P7 holds at "
    "machine precision at the release state.",
    "P2 passes, and its residual is NOT the ring-to-plane velocity gap to five "
    "figures: W231 is undiagnosed and the ratio has already moved twice "
    "(1.00001, 1.68, 1.098).  No direction predicted.",
    "P3 passes with its null leaving the fluid BITWISE identical; P4 passes and "
    "its null on the coolant clock -- run for the first time since Tier 53 -- "
    "fails as a null must, at or above 0.01.",
    "P5 is bitwise.  P6 splits as always: the lagged column under the 0.5 s "
    "ceiling and the tight one over it.  The drawn car has 51 flat bodies "
    "against the traced car's 46, so the lagged column is at most a few "
    "percent slower than Tier 54's 0.3955 s.",
)

#: What each record's car IS, in words, beside the fingerprint that says it
#: exactly.  Keyed by the record's name so a new output directory cannot
#: silently inherit another car's description or another tier's prediction.
RUNS = {
    "racelab4": {
        "prediction": PREDICTION,
        "described_as": {
            "source": "freesvg.org id 48844, CC0 / public domain",
            "method": "rasterised 1200x310 in a browser, thresholded to a "
                      "silhouette, profiles simplified with Douglas-Peucker "
                      "and welded into a chain, mapped ISOTROPICALLY",
            "map": "x = 94.0 + raster_x * 0.3760, y = (309 - raster_y) * "
                   "0.3760",
        },
    },
    "racelab5": {
        "prediction": PREDICTION_DRAWN,
        "described_as": {
            "source": "atlas/cases/car_geometry.json, drawn by hand by the "
                      "user in scripts/car_editor.py (Tier 55)",
            "method": "27 plates and 2 wheels placed with the mouse; three "
                      "plates moved by the model for reasons it forces (the "
                      "duct rebuilt DEVICE_CELLS tall about the drawn "
                      "centreline, FW_MAIN at CS-12's chord, knob-owned "
                      "angles set to the drawn ones -- W241)",
        },
    },
}


def configure(out: str) -> None:
    """Point every path in this file at one output directory."""
    global OUT, CACHE, NAME
    OUT = os.path.abspath(out)
    CACHE = os.path.join(OUT, "cache")
    NAME = os.path.basename(os.path.normpath(OUT))


def _rel(path: str) -> str:
    try:
        return os.path.relpath(path, HERE).replace(os.sep, "/")
    except ValueError:                                  # another drive
        return path


def _retry(fn, tries=6, delay=0.35):
    """OneDrive holds fresh files; os.replace can raise WinError 5."""
    for k in range(tries):
        try:
            return fn()
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(delay * (k + 1))


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if x is None or isinstance(x, (str, int, float)):
        return x
    return str(x)


def _json_path() -> str:
    if not OUT:
        raise RuntimeError("configure(out) was not called")
    return os.path.join(OUT, NAME + ".json")


def persist(res):
    os.makedirs(OUT, exist_ok=True)
    path = _json_path()
    tmp = path + ".tmp"

    def write():
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(clean(res), fh, indent=1)

    _retry(write)
    _retry(lambda: os.replace(tmp, path))
    return path


def load_res():
    path = _json_path()
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_field(name, **arrays):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name + ".npz")
    tmp = os.path.join(CACHE, name + ".tmp.npz")      # savez appends .npz
    _retry(lambda: np.savez(tmp[:-4], **arrays))
    _retry(lambda: os.replace(tmp, path))
    return path


def machine_state() -> dict:
    """What else was running, and on what power, when a timing was taken.

    `tier51_racelab_graph.machine_state`, copied rather than imported: importing
    that script sets the torch thread count to 1 as a side effect, and this
    file's timings are taken at two.
    """
    import subprocess
    out: dict = {}
    try:
        r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe",
                            "/FO", "CSV"], capture_output=True, text=True,
                           timeout=30)
        rows = [ln for ln in r.stdout.splitlines()[1:] if ln.strip()
                and "No tasks" not in ln]
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
        out["on_mains"] = (st == "2") if st else None
    except Exception as exc:                                 # pragma: no cover
        out["on_mains"] = None
        out["battery_query_failed"] = str(exc)
    return out


# ---------------------------------------------------------------------------
# the car, and the field that belongs to it
# ---------------------------------------------------------------------------


def geometry_record() -> dict:
    """The car this process builds, measured from the model rather than typed."""
    doc = RL.load_geometry()
    _objs, flat = RL.car_bodies()
    t, info = RL.layout()
    return {
        "fingerprint": RL.geometry_fingerprint(),
        "fingerprint_is": "sha256 over the BUILT bodies at the default "
                          "CarParams (racelab.geometry_fingerprint)",
        "geometry_file": _rel(RL.GEOMETRY_JSON),
        "plates": len(doc["plates"]),
        "wheels": len(doc["wheels"]),
        "flat_bodies": len(flat),
        "duct_band": [RL.DUCT_Y0, RL.DUCT_Y0 + RL.DEVICE_CELLS],
        "core_range": list(RL.DUCT_CORE_RANGE),
        "turbine_range": list(RL.TURBINE_RANGE),
        "n_windows": t.n_windows,
        "cuts": list(info.get("cols", [])),
        "device_planes": list(info.get("device_planes", [])),
        "banded_force_fraction": info.get("banded_force_fraction"),
        "described_as": RUNS.get(NAME, {}).get(
            "described_as", "no description registered for this record name"),
    }


def settled():
    """The release state, and the fingerprint of the car it was settled around.

    **Refuses a field that belongs to a different car.**  A field written
    before fingerprints existed carries none, and that is refused too: an
    unknown provenance is not a matching one.
    """
    path = os.path.join(CACHE, "settled.npz")
    if not os.path.isfile(path):
        raise RuntimeError("no settled field at %s; run --stages spinup first"
                           % _rel(path))
    d = np.load(path)
    have = str(d["geometry"]) if "geometry" in d.files else None
    want = RL.geometry_fingerprint()
    if have != want:
        raise RuntimeError(
            "the settled field at %s was settled around a DIFFERENT car "
            "(fingerprint %s) from the one this process builds (%s). A settled "
            "field belongs to a geometry; re-run --stages spinup."
            % (_rel(path), (have or "none recorded")[:16], want[:16]))
    return d["u"], d["v"]


class StalledMarch(RuntimeError):
    """A macro-step that took many times the running median."""


def watch(label: str, total: int, every: int = 25, factor: float = 10.0,
          floor_s: float = 20.0):
    """A progress callback that prints, and bails on a stall.

    **A stalled march is a blow-up.**  `WindowNS` sizes its sub-steps from
    ``u_max``, so a diverging field reads as a hang -- and with
    ``enforce=False`` nothing else stops it.  The callback runs after each
    macro-step, so it cannot interrupt a step in progress; it fires on the
    first step that took ``factor`` times the running median and at least
    ``floor_s``, and the per-step print in the log is what shows a hang that
    has not finished its step yet.
    """
    state = {"t": time.perf_counter(), "dts": [], "t0": time.perf_counter()}

    def cb(s, _elapsed):
        now = time.perf_counter()
        dt_s = now - state["t"]
        state["t"] = now
        dts = state["dts"]
        if len(dts) >= 10:
            med = float(np.median(dts[-50:]))
            if dt_s > max(factor * med, floor_s):
                raise StalledMarch(
                    "%s: macro-step %d took %.1f s against a running median of "
                    "%.3f s -- a stalled march is a blow-up" % (label, s, dt_s,
                                                                med))
        dts.append(dt_s)
        if s == 0 or (s + 1) % every == 0 or s + 1 == total:
            print("      %s %d/%d  %.0f s  (%.3f s/step)"
                  % (label, s + 1, total, now - state["t0"],
                     (now - state["t0"]) / (s + 1)), flush=True)
    return cb


# ---------------------------------------------------------------------------


def stage_spinup() -> dict:
    """Settle the car from the freestream, with the check watching."""
    torch.set_num_threads(THREADS)
    t0 = time.perf_counter()
    u, v, rep = RL.settled_field(steps=RL.N_SPIN, host_inflow=None,
                                 progress=watch("spin", RL.N_SPIN, every=20))
    wall = time.perf_counter() - t0
    fp = RL.geometry_fingerprint()
    save_field("settled", u=u, v=v, geometry=np.array(fp))
    rep["wall_s"] = wall
    rep["s_per_macro_step"] = wall / RL.N_SPIN
    rep["geometry_fingerprint"] = fp
    rep["written_to"] = _rel(os.path.join(CACHE, "settled.npz"))
    print("   settled in %.0f s (%.3f s/step); u_rotor at release %.6f"
          % (wall, wall / RL.N_SPIN, rep["u_rotor_at_the_release_state"]),
          flush=True)
    return rep


def _probe(u0, v0, host_inflow, label) -> dict:
    """One unenforced march of the whole CS-19 horizon, read three ways.

    **The whole trajectory is kept, not its endpoints.**  Tier 56's first run
    recorded the first, last, minimum and maximum of each probe and nothing
    between, and so could not say at which macro-step the fluid first breached
    its bound once the probe showed that it had -- the vault's own
    [[serialise-the-trajectory]] lesson, paid for again.
    """
    t0 = time.perf_counter()
    m = RL.march(u0, v0, steps=HORIZON, host_inflow=host_inflow,
                 join_coupling="lagged", enforce=False,
                 progress=watch(label, HORIZON, every=50))
    wall = time.perf_counter() - t0
    ur = np.asarray(m.trace["u_rotor"], dtype=float)
    um = np.asarray(m.trace["u_max"], dtype=float)
    first = m.notes.get("outside_the_envelope")
    return {
        "trace": {k: [float(x) for x in np.asarray(m.trace[k], dtype=float)]
                  for k in ("u_rotor", "u_max", "induction", "current",
                            "omega", "load", "drag")},
        "host_inflow": host_inflow,
        "wall_s": wall,
        "s_per_macro_step": wall / HORIZON,
        "u_rotor_first": float(ur[0]),
        "u_rotor_last": float(ur[-1]),
        "u_rotor_min": float(ur.min()),
        "u_rotor_max": float(ur.max()),
        "u_rotor_argmin": int(ur.argmin()),
        "fall_fraction": float((ur.max() - ur.min()) / ur.max()),
        "u_max_first": float(um[0]),
        "u_max_max": float(um.max()),
        "u_max_argmax": int(um.argmax()),
        "outside_the_envelope_steps": m.notes["outside_the_envelope_steps"],
        "outside_the_envelope_first": first,
        "first_outside_macro_step": (first.get("step")
                                     if isinstance(first, dict) else None),
        "first_declined": (list(first.get("declined", []))
                           if isinstance(first, dict) else []),
    }


def _horizon_from(probe: dict) -> int | None:
    """The longest horizon the probe stayed inside for, less the margin.

    ``None`` when there is NO admissible horizon at that probe's sizing: a
    first decline earlier than the floor plus the margin.  Tier 54's version
    returned the floor there, so Tier 56's first run reported "horizon 60"
    from a probe that had declined at macro-step 0 -- a horizon the probe had
    already refuted, which the verify then declined at the same step.
    """
    if probe["outside_the_envelope_steps"] == 0:
        return HORIZON
    first = probe["first_outside_macro_step"]
    if first is None:
        return None
    h = int(first) - HORIZON_MARGIN
    return min(HORIZON, h) if h >= HORIZON_FLOOR else None


def stage_size(res) -> dict:
    """March the horizon UNENFORCED, read the u_rotor band, choose the horizon."""
    torch.set_num_threads(THREADS)
    u0, v0 = settled()
    u_rel = res["spinup"]["u_rotor_at_the_release_state"]
    print("   probe 1 at the RELEASE state's sizing, host_inflow=%.6f ..."
          % u_rel, flush=True)
    p1 = _probe(u0, v0, u_rel, "probe1")
    out = {
        "probe_host_inflow": u_rel,
        "probe_at_the_release_sizing": p1,
        "note": "W228: size for the horizon's MINIMUM, not for one instant. "
                "The probes run with enforce=False by design -- they measure "
                "what the envelope would object to, so they must not be "
                "stopped by it.",
    }
    # Tier 54's flat keys, kept so a record reads the same whichever tier
    # wrote it
    for k in ("wall_s", "s_per_macro_step", "u_rotor_first", "u_rotor_last",
              "u_rotor_min", "u_rotor_max", "u_rotor_argmin", "fall_fraction",
              "outside_the_envelope_steps", "outside_the_envelope_first",
              "first_outside_macro_step"):
        out[k] = p1[k]
    out["U_DUCT_to_size_for"] = float(p1["u_rotor_min"])
    out["horizon_by_the_tier54_procedure"] = _horizon_from(p1)

    chooser, why_probe = p1, "probe 1"
    if p1["outside_the_envelope_steps"] and set(p1["first_declined"]) - {"FLUID"}:
        #: **The probe's first decline names the machine, and the arms do not
        #: run that machine.**  Probe 1 is sized for the release state; the
        #: arms are sized for the minimum.  Choosing the horizon from a decline
        #: the re-sizing removes would read a different quantity from the one
        #: that predicts the arms' failure -- so march the sizing the arms use.
        ud = out["U_DUCT_to_size_for"]
        print("   probe 1's first decline is %s, not the fluid; probe 2 at "
              "the ARMS' sizing, host_inflow=%.6f ..."
              % (p1["first_declined"], ud), flush=True)
        p2 = _probe(u0, v0, ud, "probe2")
        out["probe_at_the_arms_sizing"] = p2
        out["probe_2_u_rotor_min_below_the_sizing_by"] = (
            (ud - p2["u_rotor_min"]) / ud if ud else None)
        chooser, why_probe = p2, "probe 2 (at the arms' sizing)"

    out["horizon_chosen_from"] = why_probe
    out["safe_horizon"] = _horizon_from(chooser)
    out["horizon_is_CS19_s"] = out["safe_horizon"] == HORIZON
    f = chooser["outside_the_envelope_first"] or {}
    if out["safe_horizon"] is None:
        out["no_admissible_horizon"] = (
            "%s declines at macro-step %s (%s) -- earlier than the floor of %d "
            "plus the margin of %d -- so there is NO horizon this sizing admits "
            "that the arms could be run at: %s"
            % (why_probe, f.get("step"),
               ", ".join(f.get("declined", [])) or "nothing named",
               HORIZON_FLOOR, HORIZON_MARGIN, str(f.get("why", ""))[:300]))
    elif not out["horizon_is_CS19_s"]:
        out["why_the_horizon_moved"] = (
            "%s leaves a declared envelope before CS-19's %d macro-steps: "
            "first at macro-step %s, declined by %s, on %d macro-steps of the "
            "probe -- %s. The arms run at %d instead, %d macro-steps short of "
            "that first decline, and every comparison at a different horizon "
            "says so."
            % (why_probe, HORIZON, f.get("step"),
               ", ".join(f.get("declined", [])) or "nothing named",
               chooser["outside_the_envelope_steps"],
               str(f.get("why", ""))[:300], out["safe_horizon"],
               int(f["step"]) - out["safe_horizon"]))
    print("   u_rotor %.6f -> %.6f (min %.6f at %d), fall %.2f%%; horizon %s "
          "from %s" % (out["u_rotor_first"], out["u_rotor_last"],
                       out["u_rotor_min"], out["u_rotor_argmin"],
                       100 * out["fall_fraction"], out["safe_horizon"],
                       why_probe), flush=True)
    return out


def stage_verify(res) -> dict:
    """The WHOLE chosen horizon, ENFORCED, at the measured sizing."""
    torch.set_num_threads(THREADS)
    u0, v0 = settled()
    ud = res["size"]["U_DUCT_to_size_for"]
    #: **The verify marches the WHOLE horizon the arms will use, not a token
    #: prefix.**  [[positive-controls-need-a-horizon]]: a repair verified for
    #: 80 macro-steps and committed to for 600 is not verified, and this script
    #: learned that by losing a referent arm at macro-step 552.
    if res["size"].get("safe_horizon") is None:
        #: **Nothing to verify, and saying so is the result.**  Falling back
        #: to a default length here would verify a horizon the probe has
        #: already refuted.
        why = res["size"].get("no_admissible_horizon",
                              "stage size chose no horizon")
        print("   NOT verified: %s" % why, flush=True)
        return {"steps": None, "host_inflow": ud, "admitted": False,
                "why_declined": why, "marched": False, "wall_s": 0.0}
    steps = int(res["size"]["safe_horizon"])
    print("   verifying %d enforced macro-steps at host_inflow=%.6f ..."
          % (steps, ud), flush=True)
    t0 = time.perf_counter()
    admitted, why, m = True, None, None
    try:
        m = RL.march(u0, v0, steps=steps, host_inflow=ud,
                     join_coupling="lagged", enforce=True,
                     progress=watch("verify", steps, every=50))
    except RL.EnvelopeDeclined as exc:
        admitted, why, m = False, str(exc), None
    wall = time.perf_counter() - t0
    out = {"steps": steps, "host_inflow": ud, "admitted": admitted,
           "why_declined": why, "wall_s": wall,
           "coupling": "lagged",
           "the_arms_couple": sorted({kw.get("join_coupling") for _t, kw in ARMS}),
           "coupling_note": "the verify marches the LAGGED column and four of "
                            "the five arms are TIGHT; the arms run enforced, "
                            "so a tight arm that leaves the envelope stops and "
                            "is recorded rather than published",
           "probe_said_outside_steps": res["size"]["outside_the_envelope_steps"],
           "probe_said_first_outside_at": res["size"].get(
               "first_outside_macro_step")}
    if m is not None:
        ind = np.asarray(m.trace["induction"], dtype=float)
        cur = np.asarray(m.trace["current"], dtype=float)
        um = np.asarray(m.trace["u_max"], dtype=float)
        out["induction_min"] = float(ind.min())
        out["induction_max"] = float(ind.max())
        out["current_min"] = float(cur.min())
        out["current_is_positive_throughout"] = bool((cur > 0).all())
        out["u_max_max"] = float(um.max())
        out["u_max_argmax"] = int(um.argmax())
    print("   admitted=%s  %s" % (admitted, why or ""), flush=True)
    return out


def stage_arms(res) -> dict:
    torch.set_num_threads(THREADS)
    u0, v0 = settled()
    ud = res["size"]["U_DUCT_to_size_for"]
    if not res.get("verify", {}).get("admitted"):
        raise RuntimeError("stage verify did not admit; refusing to commit "
                           "ninety minutes to a sizing that is already known "
                           "to be declined")
    horizon = int(res["verify"]["steps"])
    if horizon != res["size"].get("safe_horizon"):
        raise RuntimeError("the arms' horizon and the verified one disagree; "
                           "that is how a repair gets committed to on the "
                           "strength of a shorter march than it will run")
    out: dict = {"horizon": horizon, "horizon_is_CS19_s":
                 res["size"].get("horizon_is_CS19_s"),
                 "why_the_horizon_moved":
                     res["size"].get("why_the_horizon_moved"),
                 "settle_frac": SETTLE_FRAC,
                 "threads": THREADS, "host_inflow": ud,
                 "released_from": _rel(os.path.join(CACHE, "settled.npz")),
                 "geometry_fingerprint": RL.geometry_fingerprint(),
                 "machine_state_at_start": machine_state(),
                 "enforce": True, "arms": {}}
    marches = {}
    for tag, kw in ARMS:
        print("   arm %s ..." % tag, flush=True)
        t0 = time.perf_counter()
        try:
            m = RL.march(u0, v0, steps=horizon, host_inflow=ud,
                         progress=watch(tag, horizon, every=50), **kw)
        except Exception as exc:
            out["failed_arm"] = {"arm": tag, "type": type(exc).__name__,
                                 "why": str(exc)[:1200],
                                 "after_s": time.perf_counter() - t0}
            persist({**load_res(), "arms": out})
            raise
        wall = time.perf_counter() - t0
        marches[tag] = m
        bal = VM.receiver_balances(m, SETTLE_FRAC)
        n_set = max(1, int(round(SETTLE_FRAC * horizon)))
        wr = -np.asarray(m.trace["power_rotor"])[-n_set:]
        ps = np.asarray(m.trace["shaft_power"])[-n_set:]
        ratio = wr / np.where(np.abs(ps) > 1e-30, ps, 1.0)
        um = np.asarray(m.trace["u_max"], dtype=float)
        out["arms"][tag] = {
            "wall_s": wall, "s_per_macro_step": wall / horizon,
            "settled": m.settled(SETTLE_FRAC), "balances": bal,
            "outside_the_envelope_steps": m.notes["outside_the_envelope_steps"],
            "coolant_steps": len(m.coolant),
            "u_max_max": float(um.max()),
            "J3_ratio_per_step_band": {
                "mean": float(np.mean(ratio)), "ptp": float(np.ptp(ratio)),
                "relative_ptp": (float(np.ptp(ratio) / abs(np.mean(ratio)))
                                 if float(np.mean(ratio)) else None)},
            "notes": m.notes,
        }
        save_field("arm_" + tag, u=m.u, v=m.v)
        print("      %.1f s (%.3f s/step), outside %d"
              % (wall, wall / horizon,
                 m.notes["outside_the_envelope_steps"]), flush=True)
        persist({**load_res(), "arms": out})

    ref, rep = marches["referent"], marches["repeat"]
    out["repeat_floor"] = {
        "bitwise_in_every_crossing_quantity": all(
            np.array_equal(np.asarray(ref.trace[k]), np.asarray(rep.trace[k]))
            for k in VM.CROSSING_KEYS if k in ref.trace),
        "bitwise_in_the_field": bool(np.array_equal(ref.u, rep.u)
                                     and np.array_equal(ref.v, rep.v)),
    }
    rb = out["arms"]["referent"]["balances"]
    nb = out["arms"]["null_J3"]["balances"]
    w_null = nb["J3"]["the join's term: work the rotor's body force does on the fluid"]
    p_null = nb["J3"]["the shaft's claim, T <U_d>"]
    out["G1_J3"] = {
        "residual_with_the_term": rb["J3"]["residual_with_the_term"],
        "residual_without_the_term_null_arm": (abs(w_null - p_null) / abs(p_null)
                                               if p_null else None),
        "u_plane_over_u_ring": rb["J3"]["u_plane / u_ring"],
        "what_the_velocity_gap_alone_predicts":
            (1.0 - rb["J3"]["u_plane / u_ring"]
             if rb["J3"]["u_plane / u_ring"] else None),
        "the_ratio_s_own_band": out["arms"]["referent"]["J3_ratio_per_step_band"],
    }
    out["G3_J1"] = {
        "tracking_residual": rb["J1"]["tracking_residual"],
        "null_arm_ua_settled": out["arms"]["null_J1"]["settled"]["ua"],
        "null_arm_fluid_is_bitwise_identical": bool(
            np.array_equal(marches["referent"].u, marches["null_J1"].u)
            and np.array_equal(marches["referent"].v, marches["null_J1"].v)),
    }
    return out


def stage_slow(res) -> dict:
    """J2 on the coolant clock, P4 with its null -- Tier 53's recipe, imported.

    Imported and not retyped, so the numbers compare with Tier 53's and cannot
    drift from them: a copy of a recipe is a second recipe.
    """
    sd = os.path.join(HERE, "scripts")
    if sd not in sys.path:
        sys.path.insert(0, sd)
    import tier53_racelab_rerun as T53
    out = T53.stage_slow(res)
    out["recipe"] = "scripts/tier53_racelab_rerun.py::stage_slow, imported"
    out["from_the_referent_arm_of"] = _rel(_json_path())
    return out


def stage_gate(res) -> dict:
    """P1 and P7 at the release state, and every clause judged.

    **Every threshold is read out of `racelab.GATE`, never retyped** -- a
    criterion written in prose and evaluated in code drifts, and Tier 48
    measured that it drifts permissive.  A clause whose measurement is absent
    is ``None``, not ``fail`` and not ``pass``.
    """
    from atlas import compiler as C
    from atlas.cases import ground_effect as GE
    from atlas.cases import cooling_loop as CL
    from atlas.cases import wing_fsi as W
    torch.set_num_threads(THREADS)
    g = RL.GATE
    u0, v0 = settled()
    out: dict = {"measured_at": "the release state, " +
                 _rel(os.path.join(CACHE, "settled.npz"))}

    # -- P1: the joined union, and the disjoint union as its control --------
    t0 = time.perf_counter()

    def refusals(graph):
        r = C.compile_scheme(graph)
        return r.verdict.value, sorted({"%s/%s" % (d.layer, d.rule)
                                        for d in r.decisions
                                        if d.verdict.value == "refuse"})

    gj, _ = RL.build(u0, v0)
    vj, rj = refusals(gj)
    gd, _ = RL.build(u0, v0, joins=())
    vd, rd = refusals(gd)
    gr, _ = RL.build(u0, v0, clocks="reconciled")
    vr, rr = refusals(gr)
    p1 = g["P1_compile"]
    out["P1"] = {
        "joined": {"verdict": vj, "refusals": rj, "n_agents": len(gj.agents),
                   "n_seams": len(gj.connections)},
        "disjoint": {"verdict": vd, "refusals": rd},
        "clocks_reconciled_control": {"verdict": vr, "refusals": rr},
        "expected_refusals_joined": p1["expected_refusals_joined"],
        "expected_refusals_disjoint": p1["expected_refusals_disjoint"],
        "wall_s": time.perf_counter() - t0,
        "verdict": ("pass" if (rj == p1["expected_refusals_joined"]
                               and rd == p1["expected_refusals_disjoint"])
                    else "fail"),
    }

    # -- P7: every body's force integrates to minus the force on it ---------
    tu = torch.as_tensor(u0, dtype=W.TORCH_DTYPE)
    tv = torch.as_tensor(v0, dtype=W.TORCH_DTYPE)
    objs, _flat = RL.car_bodies()
    rows, worst = [], 0.0
    for b in objs:
        fx, fy, _w, load, drag = b.forcing(tu, tv, RL.RNY, RL.RNX)
        gx = float(fx.sum() * RL.DX * RL.DX)
        gy = float(fy.sum() * RL.DX * RL.DX)
        rx = (abs(gx + float(drag)) / abs(float(drag))
              if abs(float(drag)) > 1e-12 else None)
        ry = (abs(gy - float(load)) / abs(float(load))
              if abs(float(load)) > 1e-12 else None)
        for r_ in (rx, ry):
            if r_ is not None:
                worst = max(worst, r_)
        rows.append({"body": b.body_id, "drag": float(drag),
                     "load": float(load), "residual_x": rx, "residual_y": ry})
    out["P7"] = {"per_body": rows, "worst": worst,
                 "tol": g["P7_body_force_conserves"]["tol"],
                 "evaluated_as": "tests/test_tier51_racelab_graph.py::"
                                 "test_every_body_s_force_integrates_to_minus_"
                                 "the_force_on_it, at the release state "
                                 "instead of the freestream",
                 "verdict": ("pass" if worst < g["P7_body_force_conserves"]["tol"]
                             else "fail")}

    # -- the table -----------------------------------------------------------
    arms = res.get("arms") or {}
    a = arms.get("arms") or {}
    v: dict = {"P1_compile": out["P1"]["verdict"],
               "P7_body_force_conserves": out["P7"]["verdict"]}
    r2 = arms.get("G1_J3")
    if r2 and r2.get("residual_with_the_term") is not None \
            and r2.get("residual_without_the_term_null_arm") is not None:
        c = g["P2_J3_receiving_balance"]
        v["P2_J3_receiving_balance"] = (
            "pass" if (r2["residual_with_the_term"] <= c["tol_with"]
                       and r2["residual_without_the_term_null_arm"]
                       >= c["tol_without"]) else "fail")
    else:
        v["P2_J3_receiving_balance"] = None
    r3 = arms.get("G3_J1")
    if r3 and r3.get("tracking_residual") is not None:
        c = g["P3_J1_parametric"]
        v["P3_J1_parametric"] = (
            "pass" if (r3["tracking_residual"] <= c["tol_with"]
                       and r3["null_arm_ua_settled"] == CL.UA_RAD
                       and r3["null_arm_fluid_is_bitwise_identical"])
            else "fail")
    else:
        v["P3_J1_parametric"] = None
    s4 = res.get("slow")
    if s4 and "with_the_mount_term" in s4:
        c = g["P4_J2_receiving_balance"]
        v["P4_J2_receiving_balance"] = (
            "pass" if (s4["with_the_mount_term"]["relative"] <= c["tol_with"]
                       and s4["the_mount_balance_without_the_mount_term"]
                       >= c["tol_without"]) else "fail")
    else:
        v["P4_J2_receiving_balance"] = None
    rf = arms.get("repeat_floor")
    v["P5_repeat_floor"] = (None if not rf else
                            ("pass" if rf["bitwise_in_every_crossing_quantity"]
                             else "fail"))
    if "all_lagged" in a and "referent" in a:
        ceil = g["P6_macro_step_cost"]["ceiling_s"]
        v["P6_macro_step_cost"] = {
            "lagged": ("pass" if a["all_lagged"]["s_per_macro_step"] <= ceil
                       else "fail"),
            "tight": ("pass" if a["referent"]["s_per_macro_step"] <= ceil
                      else "fail"),
            "lagged_s": a["all_lagged"]["s_per_macro_step"],
            "tight_s": a["referent"]["s_per_macro_step"],
            "ceiling_s": ceil}
    else:
        v["P6_macro_step_cost"] = None
    out["verdicts"] = v
    out["every_arm_inside_the_envelope"] = (
        all(x.get("outside_the_envelope_steps") == 0 for x in a.values())
        if a else None)
    out["horizon"] = arms.get("horizon")
    out["unit_note"] = "GE.U_INF=%r, UA_RAD=%r" % (GE.U_INF, CL.UA_RAD)
    return out


def stage_compare(res, against: str) -> dict:
    """This run beside an earlier one, on the clauses that are comparable."""
    if not os.path.isfile(against):
        return {"skipped": "%s is absent" % _rel(against)}
    with open(against, encoding="utf-8") as fh:
        old = json.load(fh)
    a_old = old.get("arms", {}).get("arms", {})
    a_new = res.get("arms", {}).get("arms", {})
    h_old = old.get("arms", {}).get("horizon")
    h_new = res.get("arms", {}).get("horizon")
    out = {"against": _rel(against),
           "horizon": [h_old, h_new],
           "same_horizon": (h_old == h_new) if (h_old and h_new) else None,
           "note": "the aerodynamic coefficients are NOT comparable -- the car "
                   "is a different shape -- and are shown to say by how much, "
                   "not to be judged against a threshold"}
    if h_old and h_new and h_old != h_new:
        out["horizon_note"] = (
            "the two records were marched to DIFFERENT horizons (%s against "
            "%s macro-steps), so every settled quantity and every balance "
            "below is a comparison of settle windows of different lengths "
            "and ends, not only of cars" % (h_old, h_new))
    for tag in ("referent", "all_lagged"):
        if tag in a_old and tag in a_new:
            out[tag] = {
                "s_per_macro_step": [a_old[tag]["s_per_macro_step"],
                                     a_new[tag]["s_per_macro_step"]],
                "outside_steps": [a_old[tag]["outside_the_envelope_steps"],
                                  a_new[tag]["outside_the_envelope_steps"]],
            }
    if "referent" in a_old and "referent" in a_new:
        out["settled_referent"] = {
            k: [a_old["referent"]["settled"].get(k),
                a_new["referent"]["settled"].get(k)]
            for k in ("u_rotor", "u_core", "ua", "induction", "current",
                      "q_machine", "load", "drag")}
    g_old, g_new = old.get("arms", {}), res.get("arms", {})
    out["J3_residual"] = [g_old.get("G1_J3", {}).get("residual_with_the_term"),
                          g_new.get("G1_J3", {}).get("residual_with_the_term")]
    out["J3_velocity_gap_predicts"] = [
        g_old.get("G1_J3", {}).get("what_the_velocity_gap_alone_predicts"),
        g_new.get("G1_J3", {}).get("what_the_velocity_gap_alone_predicts")]
    out["J1_tracking_residual"] = [g_old.get("G3_J1", {}).get("tracking_residual"),
                                   g_new.get("G3_J1", {}).get("tracking_residual")]
    out["host_inflow"] = [g_old.get("host_inflow"), g_new.get("host_inflow")]
    out["u_rotor_fall_fraction"] = [old.get("size", {}).get("fall_fraction"),
                                    res.get("size", {}).get("fall_fraction")]
    out["first_outside_macro_step"] = [
        old.get("size", {}).get("first_outside_macro_step"),
        res.get("size", {}).get("first_outside_macro_step")]
    out["geometry"] = [old.get("geometry", {}).get("plates"),
                       res.get("geometry", {}).get("plates")]
    return out


# ---------------------------------------------------------------------------
# where a car leaves the fluid envelope, and which body takes it there
# ---------------------------------------------------------------------------


def _flat_of(objects) -> list:
    flat = []
    for o in objects:
        if isinstance(o, RL.PlateBody):
            flat.append(o.body)
        else:
            flat.extend(o.bodies)
    return flat


#: A body further than this from the fastest cell is not "where it is".  The
#: first reading of Tier 56's trace named `RW_ENDPLATE` on all 544 declined
#: steps -- because it is the most downstream body, 155 cells from a cell that
#: sits on the outflow column.  A nearest neighbour is always found; that does
#: not make it near.
NEAR_CELLS = 8.0


def _where(flat, ix: int, iy: int, nx: int, ny: int) -> str:
    """A boundary if the cell is on one, a body if one is within NEAR_CELLS,
    and otherwise ``"open flow"``."""
    edges = [name for hit, name in ((ix == nx - 1, "the outflow column"),
                                    (ix == 0, "the inflow column"),
                                    (iy == 0, "the road row"),
                                    (iy == ny - 1, "the top row")) if hit]
    if edges:
        return " and ".join(edges)
    nb, dist = _nearest_body(flat, ix + 0.5, iy + 0.5)
    return nb if dist <= NEAR_CELLS else "open flow"


def _nearest_body(flat, x: float, y: float) -> tuple[str, float]:
    """The body whose segment passes closest to a point, in cells."""
    best, dist = None, float("inf")
    for b in flat:
        ax, ay, bx, by = b.x_le, b.y_le, b.x_te, b.y_te
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0.0 else max(0.0, min(1.0, ((x - ax) * dx
                                                     + (y - ay) * dy) / L2))
        d = float(np.hypot(x - (ax + t * dx), y - (ay + t * dy)))
        if d < dist:
            best, dist = b.body_id, d
    return best, dist


def traced_march(u0, v0, host_inflow, steps, objects=None, label="trace",
                 snaps=()) -> tuple[list, dict]:
    """March by hand, lagged and unenforced, and read the MODEL's own envelope
    after every macro-step -- which expert declined, the cell Reynolds number it
    computed, and where in the field the fastest cell is.

    Nothing about the predicate is re-derived here: the declined list and the
    cell Reynolds number are `RaceRollout._absorb`'s, read off the rollout.
    `racelab.march` could not be used, because it does not hand its rollout to
    a progress callback.  The fluid trajectory is the same one `march` takes
    with ``join_coupling="lagged"``: the coolant loop it also sub-cycles reads
    the fluid and writes nothing back into it.
    """
    from atlas.cases import wing_fsi as W
    t, _i = RL.layout()
    r = RL.RaceRollout(tiling=t, objects=objects, join_coupling="lagged",
                       enforce=False, host_inflow=host_inflow)
    opt = dict(dtype=W.TORCH_DTYPE, device=r.device)
    u = torch.as_tensor(np.asarray(u0), **opt)
    v = torch.as_tensor(np.asarray(v0), **opt)
    r.refresh(u, 0, which=("J1", "J3"))
    flat = _flat_of(r.objects)
    cb = watch(label, steps, every=50)
    rows, saved = [], {}
    for s in range(steps):
        with torch.no_grad():
            u, v, load, drag = r.macro_step(u, v, s)
            sp = torch.hypot(u, v)
            k = int(torch.argmax(sp))
        iy, ix = divmod(k, r.nx)
        nb, dist = _nearest_body(flat, ix + 0.5, iy + 0.5)
        env = r.envelope or {}
        rows.append({
            "step": s, "u_max": float(sp.reshape(-1)[k]),
            "at_cell": [ix, iy], "where": _where(flat, ix, iy, r.nx, r.ny),
            "nearest_body": nb, "distance_cells": dist,
            "cell_reynolds": env.get("FLUID", {}).get("cell_reynolds"),
            "declined": list((r.outside or {}).get("declined", [])),
            "u_rotor": float(r.state.u_rotor),
            "induction": float(r.state.induction),
            "current": float(r.state.current),
            "load": float(load), "drag": float(drag)})
        if s in snaps:
            saved["u_%03d" % s] = u.detach().cpu().numpy().copy()
            saved["v_%03d" % s] = v.detach().cpu().numpy().copy()
        cb(s, 0.0)
    return rows, saved


def _summarise_trace(rows) -> dict:
    from collections import Counter

    def first(tag):
        return next((r_["step"] for r_ in rows if tag in r_["declined"]), None)

    fluid_rows = [r_ for r_ in rows if "FLUID" in r_["declined"]]
    um = np.array([r_["u_max"] for r_ in rows])
    return {
        "steps": len(rows),
        "first_FLUID_decline": first("FLUID"),
        "first_ROTOR_decline": first("ROTOR"),
        "first_MGU_decline": first("MGU"),
        "FLUID_declined_steps": len(fluid_rows),
        "u_max_first": float(um[0]), "u_max_last": float(um[-1]),
        "u_max_max": float(um.max()), "u_max_argmax": int(um.argmax()),
        "u_max_every_50": [float(um[i]) for i in range(0, len(um), 50)],
        "where_the_fastest_cell_is_while_FLUID_declines": dict(Counter(
            _where_of(r_) for r_ in fluid_rows).most_common(8)),
        "where_the_fastest_cell_is_over_the_whole_march": dict(Counter(
            _where_of(r_) for r_ in rows).most_common(8)),
        "near_means_within_cells": NEAR_CELLS,
        "fastest_cell_at_the_last_step": {
            k: rows[-1].get(k) for k in ("at_cell", "where", "nearest_body",
                                         "distance_cells", "u_max",
                                         "cell_reynolds")},
    }


def _where_of(row) -> str:
    """A row's location, re-derived for rows written before `where` existed."""
    if "where" in row:
        return row["where"]
    ix, iy = row["at_cell"]
    if ix == RL.RNX - 1:
        return "the outflow column"
    return (row["nearest_body"] if row["distance_cells"] <= NEAR_CELLS
            else "open flow")


def stage_trace(res, steps: int) -> dict:
    """The drawn car's own march, with the envelope read every macro-step."""
    torch.set_num_threads(THREADS)
    u0, v0 = settled()
    ud = res["spinup"]["u_rotor_at_the_release_state"]
    snaps = tuple(sorted({0, steps // 4, steps // 2, (3 * steps) // 4,
                          steps - 1}))
    rows, saved = traced_march(u0, v0, ud, steps, label="trace", snaps=snaps)
    if saved:
        save_field("trace_fields", **saved)
    out = _summarise_trace(rows)
    out.update({"host_inflow": ud, "sizing": "the release state's",
                "rows": rows,
                "snapshots_saved_at": list(snaps),
                "snapshots_in": _rel(os.path.join(CACHE, "trace_fields.npz"))})
    print("   FLUID first declines at %s on %d of %d macro-steps; u_max %.4f -> "
          "%.4f; fastest cell while declining: %s"
          % (out["first_FLUID_decline"], out["FLUID_declined_steps"], steps,
             out["u_max_first"], out["u_max_last"],
             out["where_the_fastest_cell_is_while_FLUID_declines"]),
          flush=True)
    return out


def stage_ablate(res, steps: int, drops: list) -> dict:
    """The same march with bodies REMOVED, one set at a time, and the car intact
    beside them as the control.

    **The window layout is held at the drawn car's** -- only the objects the
    fluid is forced with change -- so each arm differs from the control in the
    bodies it carries and in nothing else.  The control is marched through the
    same code path with nothing dropped, and it must agree bitwise with stage
    ``trace`` over the steps both took; that is asserted rather than assumed.
    """
    import copy
    torch.set_num_threads(THREADS)
    u0, v0 = settled()
    ud = res["spinup"]["u_rotor_at_the_release_state"]
    doc = RL.load_geometry()
    ids = {e["id"] for e in doc["plates"]}
    out: dict = {"steps": steps, "host_inflow": ud,
                 "layout": "held at the drawn car's", "arms": {}}
    for drop in [()] + [tuple(d) for d in drops]:
        bad = set(drop) - ids
        if bad:
            raise SystemExit("no such plate: %s" % sorted(bad))
        tag = "none" if not drop else "-".join(drop)
        d2 = copy.deepcopy(doc)
        d2["plates"] = [e for e in d2["plates"] if e["id"] not in set(drop)]
        objs, _flat = RL.car_bodies(geometry=d2)
        print("   ablate %s ..." % tag, flush=True)
        rows, _saved = traced_march(u0, v0, ud, steps, objects=objs,
                                    label="ablate-" + tag)
        s = _summarise_trace(rows)
        s["dropped"] = list(drop)
        s["u_max_trace"] = [r_["u_max"] for r_ in rows]
        out["arms"][tag] = s
        persist({**load_res(), "ablate": out})
    tr = res.get("trace", {}).get("rows")
    if tr:
        n = min(len(tr), steps)
        out["control_agrees_bitwise_with_stage_trace"] = all(
            tr[i]["u_max"] == out["arms"]["none"]["u_max_trace"][i]
            for i in range(n))
        #: **stage trace's first summary is re-read, and the correction is
        #: recorded rather than made silently.**  It named the NEAREST body
        #: for the fastest cell -- `RW_ENDPLATE`, 155 cells away from a cell
        #: on the outflow column.
        fixed = _summarise_trace(tr)
        out["trace_summary_reread"] = {
            "why": "the first summary attributed the fastest cell to the "
                   "nearest body however far away it was; a body further than "
                   "NEAR_CELLS is not where the cell is",
            "was": {k: res["trace"].get(k) for k in (
                "where_the_fastest_cell_is_while_FLUID_declines",
                "where_the_fastest_cell_is_over_the_whole_march")},
            "is": {k: fixed[k] for k in (
                "where_the_fastest_cell_is_while_FLUID_declines",
                "where_the_fastest_cell_is_over_the_whole_march",
                "near_means_within_cells")},
        }
    return out


STAGES = ("spinup", "size", "verify", "arms", "slow", "gate", "compare",
          "trace", "ablate")


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True,
                    help="output directory; the record is <out>/<basename>.json. "
                         "REQUIRED: the old default overwrote Tier 54's "
                         "committed record")
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--compare-with", default=TIER53,
                    help="the earlier record stage compare reads")
    ap.add_argument("--trace-steps", type=int, default=HORIZON,
                    help="macro-steps stage trace marches")
    ap.add_argument("--ablate-steps", type=int, default=300,
                    help="macro-steps each arm of stage ablate marches")
    ap.add_argument("--drop", default="",
                    help="stage ablate's arms: plate-id sets separated by ';', "
                         "ids within a set by ','")
    args = ap.parse_args(argv)
    want = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in want:
        if s not in STAGES:
            raise SystemExit("unknown stage %r; known: %s" % (s, STAGES))
    configure(args.out)

    res = load_res()
    #: the prediction and its timestamp are written BEFORE the first stage, so
    #: the record shows the order rather than asserting it
    run = RUNS.get(NAME)
    res.setdefault("prediction", list(run["prediction"]) if run else [
        "NO PREDICTION was registered for a record named %r before this run"
        % NAME])
    res.setdefault("prediction_recorded_at",
                   dt.datetime.now().isoformat(timespec="seconds"))
    if "spinup" in want or "geometry" not in res:
        res["geometry"] = geometry_record()
    res.setdefault("runs", []).append({
        "stages": want, "started_at": dt.datetime.now().isoformat(
            timespec="seconds"),
        "argv": list(argv), "machine_state": machine_state()})
    persist(res)

    for s in want:
        print("== stage %s ==" % s, flush=True)
        t0 = time.perf_counter()
        try:
            if s == "spinup":
                res[s] = stage_spinup()
            elif s == "size":
                res[s] = stage_size(res)
            elif s == "verify":
                res[s] = stage_verify(res)
            elif s == "arms":
                res[s] = stage_arms(res)
            elif s == "slow":
                res[s] = stage_slow(res)
            elif s == "gate":
                res[s] = stage_gate(res)
            elif s == "compare":
                res[s] = stage_compare(res, os.path.abspath(args.compare_with))
            elif s == "trace":
                res[s] = stage_trace(res, args.trace_steps)
            elif s == "ablate":
                drops = [[x.strip() for x in grp.split(",") if x.strip()]
                         for grp in args.drop.split(";") if grp.strip()]
                if not drops:
                    raise SystemExit("stage ablate needs --drop")
                res[s] = stage_ablate(res, args.ablate_steps, drops)
        except BaseException as exc:
            #: persist the failure BEFORE it propagates -- keep what an arm
            #: that finished already wrote (stage_arms persists per arm)
            latest = load_res()
            latest.setdefault("failures", []).append({
                "stage": s, "type": type(exc).__name__, "why": str(exc)[:1200],
                "after_s": time.perf_counter() - t0,
                "at": dt.datetime.now().isoformat(timespec="seconds"),
                "traceback": traceback.format_exc()[-3000:]})
            persist(latest)
            print("   STAGE %s FAILED: %s: %s" % (s, type(exc).__name__,
                                                  str(exc)[:400]), flush=True)
            raise
        res.setdefault("stage_wall_s", {})[s] = time.perf_counter() - t0
        if s == "arms":
            #: stage_arms persisted its own partial record per arm; merge so
            #: the in-memory copy does not overwrite anything it wrote
            latest = load_res()
            latest.update({k: res[k] for k in res if k != "failures"})
            res = latest
        print("   -> %s" % persist(res), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
