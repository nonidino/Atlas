"""PoC 2's demo engine: a live front-wing march you can watch and optimise.

This sits ON TOP of `atlas.cases.front_wing` and changes nothing in it.
Everything here is a call into that module; nothing about the physics, the
composition layer or the interface system lives in this file.  `atlas/demo/`'s
engine is the template and three of its rules are inherited verbatim:

  * **the engine measures itself and this file quotes no fixed speed.**  A
    wall-clock number is a claim about a machine in a power state -- the PoC 1a
    demo shipped a table that was wrong by 4x the next time anyone looked -- so
    `Engine.eta_s` reports a moving average of what steps have actually cost
    here and the screen marks it as an estimate until the first real one lands;
  * **the optimiser's rollout IS the live march**, warm-started and short, so the
    field animates inside the gradient step rather than freezing until it lands.
    That is a different estimator from `w141_poc2_frontwing.py`'s and the screen
    says so;
  * **nothing here owns a verdict.**  The certification panel is
    `compile_scheme`'s own output, grouped per seam by
    `w141_poc2_frontwing.seam_verdicts`, and this file's only job is to run the
    compile on a worker and hand the answer to the page.

What is new, and it is the point of this demo
---------------------------------------------

**A per-seam certified indicator, driven by the compiler and not by a rule of
thumb.**  Every one of the nine seams gets its own verdict -- `admit`,
`admit-uncertified` or `refuse` -- from the decisions whose subject reaches it,
and the page draws green/amber/red from that.  Two things it must not overstate,
both measured in `w141`'s `compile` stage and both said on screen:

1. **The verdict is a property of the DECLARATION, not of the design point.**
   Over the sixteen corners of the design box, zero produce a different per-seam
   verdict map.  So the colours change when the interface is declared to move --
   the two surface seams go red at `L2/InterfaceMotion` and the seven
   fluid-fluid ones stay amber -- and they do NOT flicker as a knob turns.  The
   compile is still re-run on every design change, on a worker thread, because
   the honest way to show that is to run it rather than to assume it.
2. **What moves with the design is the quantity underneath.**  The one-sidedness
   of each seam, the interface residual and the two constraint margins are live,
   and they are what a designer would actually watch.
"""
from __future__ import annotations

import io
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from ..cases import front_wing as F
from ..cases import wing_fsi as W
from ..compiler import compile_scheme
from . import ablation as AB
from . import race as RA
from . import substitution as SUB
from .balance import Balance, spring_energy, strain_energy

# ---------------------------------------------------------------------------
# the colour ramp for the flow field -- `atlas/demo/engine.py`'s, unchanged
# ---------------------------------------------------------------------------

U_LO, U_HI = 0.0, 1.6
RAMP_STOPS = (
    (0.00, (12, 20, 48)), (0.25, (26, 76, 128)), (0.50, (44, 148, 160)),
    (0.72, (150, 200, 130)), (0.88, (240, 214, 110)), (1.00, (250, 250, 235)),
)

#: The von Mises ramp is SEPARATE and it is anchored on the CEILING, not on the
#: frame's own maximum.  A ramp that renormalises every frame makes a design at
#: 20% of the ceiling look identical to one at 99% of it, which is the one thing
#: this panel exists to distinguish.
STRESS_STOPS = (
    (0.00, (40, 54, 84)), (0.35, (60, 130, 170)), (0.62, (140, 190, 120)),
    (0.85, (238, 190, 80)), (1.00, (226, 74, 60)),
)


def _lut(stops) -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        for k in range(1, len(stops)):
            a, ca = stops[k - 1]
            b, cb = stops[k]
            if t <= b:
                f = 0.0 if b == a else (t - a) / (b - a)
                lut[i] = [round(ca[j] + f * (cb[j] - ca[j])) for j in range(3)]
                break
        else:
            lut[i] = stops[-1][1]
    return lut


_FIELD_LUT = _lut(RAMP_STOPS)
_STRESS_LUT = _lut(STRESS_STOPS)


def stress_colour(vm: float, ceiling: float = None) -> str:
    ceiling = F.SIGMA_CEIL if ceiling is None else ceiling
    t = float(np.clip(vm / max(ceiling, 1e-30), 0.0, 1.0))
    r, g, b = _STRESS_LUT[int(t * 255)]
    return f"#{r:02x}{g:02x}{b:02x}"


def field_png(u: np.ndarray, v: np.ndarray, stride: int = 1) -> bytes:
    """Colour-map the speed and encode it as a PNG.

    Row 0 of the array is y = 0 and an image's row 0 is the top, so the array is
    flipped here; the client draws it with y up, which is how a wing over a floor
    is read.
    """
    from PIL import Image
    a = np.hypot(u, v)[::stride, ::stride]
    t = np.clip((a - U_LO) / (U_HI - U_LO), 0.0, 1.0)
    rgb = _FIELD_LUT[(t * 255.0).astype(np.uint8)][::-1]
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# the configuration
# ---------------------------------------------------------------------------


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _ema(prev, x, a: float = 0.25):
    return float(x) if prev is None else float((1 - a) * prev + a * x)


@dataclass
class DemoConfig:
    #: the composed column by default; `single` is the referent, and switching
    #: between them live is what makes the cut's cost visible rather than quoted
    tiling: str = "six"
    coupling: str = "tight"
    lag: int = 1
    #: the optimiser's rollout length.  SHORT and warm-started: see the module
    #: docstring, and the screen says which estimator it is
    horizon: int = 12
    lr: float = 0.05
    penalty: float = 40.0
    stride: int = 1
    spin: int = 40
    #: **Beat 2, and the only switch on this screen that can make it lie.**
    #: True is the model: both experts' declared validity envelopes are checked
    #: on every march, and a design that leaves one is DECLINED.  False
    #: reproduces the behaviour every path but `run` had until W145 -- the
    #: optimiser walks the wing through the suspension's floor and the screen
    #: shows a downforce for a design the model refuses to stand behind.  It is
    #: offered because the difference is the demonstration, and every number it
    #: produces is stamped OUTSIDE THE MODEL wherever it appears.
    enforce: bool = True
    #: the on-demand beats.  Small enough to finish while somebody watches, and
    #: the screen always shows the recorded full-scale figure beside the live one.
    #:
    #: **The race budget is scaled, not chosen.**  The recorded run is 30 Adam
    #: steps against 200 CMA-ES evaluations -- a ratio of 1 to 6.67 -- and the
    #: live one keeps that ratio (14 to 93) and shortens the horizon instead.
    #: Picking the two budgets independently would mean picking them until the
    #: answer came out the way this panel would prefer, which is the one thing a
    #: comparison that exists to be honest cannot do.  At a budget below about
    #: nine gradient iterates the population column wins on wall-clock, because
    #: Adam has not turned yet; that is a real property of the estimator and the
    #: screen says it rather than tuning it away.
    ablation_steps: int = 60
    race_steps: int = 16
    race_grad: int = 14
    race_pop: int = 93

    def clamped(self) -> "DemoConfig":
        return DemoConfig(
            tiling=self.tiling if self.tiling in ("six", "single") else "six",
            coupling=(self.coupling
                      if self.coupling in ("tight", "lagged", "split") else "tight"),
            lag=int(_clamp(self.lag, 1, 32)),
            horizon=int(_clamp(self.horizon, 2, 60)),
            lr=float(_clamp(self.lr, 1e-3, 0.3)),
            penalty=float(_clamp(self.penalty, 0.0, 500.0)),
            stride=int(_clamp(self.stride, 1, 4)),
            spin=int(_clamp(self.spin, 0, 400)),
            enforce=bool(self.enforce),
            ablation_steps=int(_clamp(self.ablation_steps, 8, 240)),
            race_steps=int(_clamp(self.race_steps, 4, 120)),
            race_grad=int(_clamp(self.race_grad, 2, 40)),
            race_pop=int(_clamp(self.race_pop, 4, 400)))


@dataclass
class Frame:
    seq: int = 0
    png: bytes = b""
    width: int = 0
    height: int = 0
    payload: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# the certification panel
# ---------------------------------------------------------------------------

GRAPH_SUBJECTS = ("<graph>", "<assembly>", "<run>", "<scheme>")


def seam_verdicts(graph, result) -> dict:
    """Every seam's own verdict, from the compiler's own decisions.

    Kept identical to `scripts/w141_poc2_frontwing.seam_verdicts` on purpose --
    a demo that groups the decisions differently from the driver is showing a
    different thing from the one the results page reports.  A test asserts the
    two agree on this graph.
    """
    from ..verdict import Verdict
    out: dict[str, dict] = {}
    for c in graph.connections:
        ports = {f"{c.a[0]}.{c.a[1]}", f"{c.b[0]}.{c.b[1]}"}
        agents = {c.a[0], c.b[0]}
        rows = {"refuse": [], "admit-uncertified": [], "global": []}
        for d in result.decisions.decisions:
            if d.verdict is Verdict.ADMIT:
                continue
            subs = {s.strip() for s in str(d.subject or "").split(",")}
            tag = f"{d.layer}/{d.rule}"
            if subs & ({c.seam_id} | ports | agents):
                rows[d.verdict.value].append(tag)
            elif subs & set(GRAPH_SUBJECTS):
                rows["global"].append(tag)
        verdict = ("refuse" if rows["refuse"]
                   else "admit-uncertified"
                   if (rows["admit-uncertified"] or rows["global"]) else "admit")
        out[c.seam_id] = dict(
            verdict=verdict,
            colour={"refuse": "red", "admit-uncertified": "amber",
                    "admit": "green"}[verdict],
            kind=("fluid-structure" if c.seam_id == "wet"
                  else "field-lumped" if c.seam_id == "mount"
                  else "fluid-fluid"),
            a=f"{c.a[0]}.{c.a[1]}", b=f"{c.b[0]}.{c.b[1]}",
            refusals=sorted(set(rows["refuse"])),
            decertifications=sorted(set(rows["admit-uncertified"])),
            global_decertifications=sorted(set(rows["global"])))
    return out


#: What each rule means, in one line, because a panel that shows `L2/R10/halo`
#: and nothing else is a panel nobody can act on.
RULE_NOTE = {
    "L2/InterfaceMotion": "the interface's geometry is a function of the "
                          "solution and no rule certifies a cached operator "
                          "across that motion. CS-10 priced it: at a 2% "
                          "staleness tolerance the cache does not survive one "
                          "exchange",
    "L2/R10": "an agent declares an incompressible family with no elliptic "
              "sub-solve, or embeds one the decomposition cut. Narrowed at "
              "W114 to check its own premise",
    "L2/R10/halo": "the halo an exchange needs is undecidable from the record. "
                   "Two ways to get here and both are on this screen: an "
                   "IMPLICIT agent with a nonzero stencil inverts an operator "
                   "coupling every cell to every other, and an agent declaring "
                   "time_discretization=unknown -- what a frozen learned "
                   "operator declares -- has not said its step is explicit, so "
                   "radius x substeps is not licensed for it either (W93: "
                   "measured 128 cells against a declared 2). W136: at the wet "
                   "seam it over-fires, because that seam is a PHYSICAL "
                   "boundary with no overlap for anything to outrun",
    "L2/R10/W60": "elliptic_subsolve is declared unknown, so R10 can neither "
                  "fire nor be cleared -- and the probe cannot settle it. "
                  "elliptic_signature measures NON-NORMALITY and was read as "
                  "measuring GLOBALITY: a known-embedded conduction solve reads "
                  "kappa = 1.0002 across a 10,000x range of exchange interval, "
                  "which the statistic's own thresholds call consistent with "
                  "exposed. A declaration nobody can check",
    "L4/R2b/W46": "the transmission rung is held at Dirichlet rather than "
                  "lifted to probed-DtN. The flux-balance condition is the "
                  "interface condition of a boundary-value problem, and an "
                  "agent that poses none -- explicit, or of an undecided kind "
                  "-- does not satisfy it. Measured: solving it exactly moved "
                  "the trace 41x past the truth and made the composed step 8.9x "
                  "worse than not solving it, while beta, kappa, the passivity "
                  "spectrum and the probe residual all read healthy",
    "L2/C2": "the cut is admissible and is not certified optimal",
    "L4/E7/passivity": "the symmetric part of the assembled response is "
                       "indefinite. W138: at this seam that is an ORIENTATION "
                       "artefact -- the two blocks are added where the "
                       "interface residual subtracts them -- and flipping one "
                       "sign takes the defect to exactly zero",
    "L5/eps_tol": "the accelerator's tolerance rests on a defect term that is "
                  "identically zero here",
    "L6/R12": "the partition of unity's contaminated set bounds sigma and the "
              "bound is not certified",
    "L6/W49": "C_mu is unmeasured",
    "L9/E5": "the run-level claim has no measured L (W1)",
    "L8/W56": "at least one bound constant is unmeasured, which has forced "
              "admit-uncertified on every graph in this package since W56",
}


# ---------------------------------------------------------------------------
# the recorded run, so a live number is never the only number on screen
# ---------------------------------------------------------------------------

#: `out/w141/w141.json` -- the driver's own artifact, every stage of the
#: full-scale run. The demo NEVER writes it and never falls back to a hard-coded
#: copy of a number in it: if it is absent the screen says the recorded column is
#: unavailable, which is the honest state, rather than quoting a constant that
#: has drifted from the run it came from. (`wall-clock numbers rot` -- the PoC 1a
#: demo shipped a table that was wrong by 4x the next day on unchanged code.)
ARTIFACT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "out", "w141", "w141.json")


def load_recorded(path: str | None = None) -> dict:
    """The full-scale numbers the live panels are shown beside.

    Only the handful the screen actually quotes are lifted out, so the payload
    stays small and so that adding a number to the screen means adding it here
    -- where the path it came from is written down beside it.
    """
    import json

    p = path or ARTIFACT
    if not os.path.isfile(p):
        return dict(available=False, path=p,
                    why="out/w141/w141.json is not here, so every panel shows "
                        "its live measurement only. Run "
                        "scripts/w141_poc2_frontwing.py to produce it.")
    try:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError) as exc:                    # pragma: no cover
        return dict(available=False, path=p, why=f"unreadable: {exc}")

    def _dig(*keys, default=None):
        cur: Any = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    out: dict[str, Any] = dict(available=True, path=p,
                               generated=d.get("generated"))
    out["residual"] = {k: _dig("residual", k) for k in
                       ("steps", "level", "without_motion", "with_motion",
                        "corrected", "settled", "factor")}
    for col in ("design", "design_tight", "design_unenforced", "design_loose"):
        if col not in d:
            continue
        out[col] = dict(
            start_J=_dig(col, "start_eval", "J"),
            best_J=_dig(col, "gradient", "best_feasible", "J"),
            active=_dig(col, "active"),
            comparison=_dig(col, "comparison"),
            n_grad=_dig(col, "n_grad"), n_pop=_dig(col, "n_pop"),
            steps=_dig(col, "steps"),
            #: **`declined` and `infeasible` are different numbers and the panel
            #: must not merge them.** A design is DECLINED when an expert says
            #: it is outside what that expert is valid for -- W145's story. It
            #: is merely INFEASIBLE when it is inside the model and over a
            #: ceiling. On the primary column: 39 declined, 66 infeasible, so
            #: 27 of the population's rejections were ordinary constraint
            #: violations and reporting 66 as declines would overstate W145 by
            #: a factor of 1.7.
            #: The population history carries no `declined` key -- it is
            #: detected by the sentinel L the driver returns for one.
            evals_grad=len(_dig(col, "gradient", "history") or []),
            evals_pop=len(_dig(col, "population", "history") or []),
            declined_grad=sum(1 for h in (_dig(col, "gradient", "history") or [])
                              if h.get("declined")),
            declined_pop=sum(1 for h in (_dig(col, "population", "history") or [])
                             if h.get("L", 0.0) <= -999.0),
            infeasible_pop=sum(1 for h in (_dig(col, "population", "history") or [])
                               if not h.get("feasible")),
            sigma_ceiling=_dig(col, "ceilings", "sigma", default=F.SIGMA_CEIL),
        )
    out["ablation"] = dict(
        steps=_dig("ablation", "steps"),
        worth_at_reference=_dig("ablation", "at_optimum", "worth_at_reference"),
        worth_at_optimum=_dig("ablation", "at_optimum", "worth"),
        frozen_at_optimum=_dig("ablation", "at_optimum", "frozen"),
    )
    out["cost"] = dict(
        adjoint_over_forward=_dig("cost", "adjoint_over_forward"),
        adjoint_over_bare_march=_dig("cost", "adjoint_over_bare_march"),
    )
    #: the assembly's own gate: each parent case study recovered as a limit of
    #: this graph. These are the numbers that earn the word "assembly"
    out["controls"] = dict(
        k_to_infinity=_dig("controls", "k_to_infinity", "rel"),
        S_to_infinity=_dig("controls", "S_to_infinity", "rel"),
        stress_map=_dig("controls", "stress_map", "rel"),
        bitwise=_dig("controls", "reproducible_bitwise"),
        passed=_dig("controls", "pass"),
    )
    return out


# ---------------------------------------------------------------------------
# Tier 33's measured influence reach -- the bars under beat 4's three verdicts
# ---------------------------------------------------------------------------

#: `out/w153/w153.json`, the Tier 33 driver's own artifact. Same rule as
#: `ARTIFACT` above and for the same reason: the demo never writes it, never
#: re-runs it, and **never falls back to a hard-coded copy of a number in it.**
#: If it is absent the reach panel says so and the three verdicts stand on their
#: own, which is the state the beat was in before this was measured.
HORIZON_ARTIFACT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "out", "w153", "w153.json")

#: How each measured agent in `w153.json` reaches the screen: which row of
#: `side.rows` carries its Pi_w, which case carries its decay profile, and which
#: of beat 4's three DECLARATIONS it is the measurement of.
#:
#: **The mapping is deliberately not one-to-one and the panel must not pretend
#: it is.** `poseidon_declared` and `poseidon_probed` are two descriptions of one
#: model, so they share one measured row -- which is the beat's own point, seen
#: from the other side. And the embedded classical column is the measurement of
#: no declaration on this screen: it is the control that separates "global" from
#: "global because it is learned", and it carries no candidate key.
_REACH_ROWS = (
    dict(agent="WindowNS", variant="exposed",
         case=("gateB", "cases", "windowns_exposed"),
         label="the classical solver that is running",
         plain="a local step: information moves a fixed number of cells per "
               "sub-step and no further",
         candidates=("windowns",)),
    dict(agent="WindowNS", variant="embedded",
         case=("gateB", "cases", "windowns_embedded"),
         label="the same solver, carrying its own pressure solve",
         plain="one whole-domain solve inside the step, so a poke anywhere is "
               "felt everywhere -- and it is still a classical solver",
         candidates=()),
    dict(agent="Poseidon-T", variant="as declared",
         case=("gateC",),
         label="Poseidon-T",
         plain="a frozen pretrained operator: global by construction, and "
               "further out than the classical solve that is also global",
         candidates=("poseidon_declared", "poseidon_probed")),
)


def load_horizon(path: str | None = None) -> dict:
    """Tier 33's three measured reaches, for the panel under beat 4.

    Reads the same handful of figures the screen quotes and no more, so that
    adding one to the screen means adding it here, beside the path in the file
    it came from.

    The distinction the panel exists to draw: **the verdict is a property of the
    declaration; the reach is a property of the model.** Both are on screen and
    they are read from different places -- the verdict from `compile_scheme` on
    the live graph, the reach from this artifact.
    """
    import json

    p = path or HORIZON_ARTIFACT
    if not os.path.isfile(p):
        return dict(available=False, path=p,
                    why="out/w153/w153.json is not here, so the three verdicts "
                        "are shown without the measured reach underneath them. "
                        "Run scripts/w153_influence_envelope.py to produce it.")
    try:
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
    except (OSError, ValueError) as exc:                    # pragma: no cover
        return dict(available=False, path=p, why=f"unreadable: {exc}")

    side = (d.get("side") or {}).get("rows") or []
    by_agent = {(r.get("agent"), r.get("variant")): r for r in side}

    rows: list[dict] = []
    for spec in _REACH_ROWS:
        src = by_agent.get((spec["agent"], spec["variant"]))
        if src is None:
            return dict(available=False, path=p,
                        why=f"{spec['agent']} ({spec['variant']}) is not in "
                            "side.rows of this artifact, so the ranking would "
                            "be incomplete and is not shown.")
        cur: Any = d
        for k in spec["case"]:
            if not isinstance(cur, dict) or k not in cur:
                cur = None
                break
            cur = cur[k]
        states = (cur or {}).get("states") or []
        far = [s.get("far_field") or {} for s in states]
        if not far:
            return dict(available=False, path=p,
                        why=f"{spec['agent']} carries no probed states in this "
                            "artifact, so its reach is not shown.")
        #: **`exactly_zero` is asserted across EVERY probed state, not one.**
        #: `d_eff` moves between 19 and 20 over the three spin-up states while
        #: the far-field block is bitwise zero in all of them, so the bitwise
        #: claim is the one that is quoted and the horizon is the DECLARED
        #: `d_ref` it was tested at rather than a per-state fit.
        zero = all(bool(f.get("exactly_zero")) for f in far)
        #: **The far-field FROBENIUS is deliberately not what the panel shows.**
        #: It is a raw magnitude in each agent's own units, and it ranks the
        #: three the OPPOSITE way from the reach: 76.07 for the embedded
        #: classical solver against 9.78 for Poseidon-T, while Poseidon-T is the
        #: one whose influence does not decay. Showing it beside the bars would
        #: invite exactly the misreading the bars exist to prevent. What is
        #: shown instead is `profile_max` at b = d -- the response normalised by
        #: its own peak, which is what the bar is measuring: **decay, not size.**
        #: Sampled at TWO distances, not one, because "still 26% at b = 20" and
        #: "still 61% at b = 20" do not say the thing that actually separates
        #: these two rows: the embedded solver keeps decaying (0.264 -> 0.124)
        #: and the checkpoint does not (0.612 -> 0.600). One sample would let
        #: the panel call both of them flat, which is true of only one.
        FAR_B = 60
        d_ref = (cur or {}).get("d_ref")

        def _profile_at(k: int):
            vals = [s["profile_max"][k] for s in states
                    if isinstance(s.get("profile_max"), list)
                    and len(s["profile_max"]) > k]
            return max(vals) if vals else None

        at_d = _profile_at(int(d_ref)) if d_ref is not None else None
        at_far = _profile_at(FAR_B)
        rows.append(dict(
            agent=spec["agent"], variant=spec["variant"],
            label=spec["label"], plain=spec["plain"],
            candidates=list(spec["candidates"]),
            pi=src.get("Pi"), pi_w=src.get("Pi_w"),
            far_zero=zero,
            far_cells=far[0].get("n_far_cells"),
            #: the worst over the probed states, so a quiet one cannot flatter
            profile_at_d=at_d,
            profile_at_far=at_far,
            far_b=FAR_B,
            far_frobenius=max(float(f.get("frobenius") or 0.0) for f in far),
            #: `d` is the distance from the cut the far field was taken beyond,
            #: carried for EVERY row so the panel never writes the number in.
            #: `horizon` is the same value promoted to a claim, and only when
            #: the far field is bitwise zero at it.
            d=(cur or {}).get("d_ref"),
            horizon=(cur or {}).get("d_ref") if zero else None,
            states=len(states),
        ))

    ratio = (d.get("gateA") or {}).get("Pi_w_over_Pi") or {}
    return dict(
        available=True, path=p, generated=d.get("generated"),
        #: what the screen prints. The absolute path is kept for a diagnostic
        #: but is the wrong thing to show: it names one machine's checkout, and
        #: in the standalone bundle it names a different one, so a reader
        #: comparing the screen against the repository sees a path that is not
        #: in the repository.
        rel="out/w153/w153.json",
        rows=rows,
        #: the reduction control, which is what licenses adopting the refinement
        #: at all: where the overlap is narrower than the domain of dependence
        #: nothing decays inside it and the refined envelope returns the old
        #: indicator exactly, so no past verdict can have moved silently.
        reduces_exactly=dict(min=ratio.get("min"), max=ratio.get("max")),
        exchanges="one per macro-step",
    )


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------


class Engine:
    """One persistent coupled state, marched or optimised on a worker thread."""

    def __init__(self, cfg: DemoConfig | None = None) -> None:
        self.cfg = (cfg or DemoConfig()).clamped()
        self.design = dict(F.DESIGN_REF)
        self.mode = "run"
        self.notice = ""
        self.frame = Frame()
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seq = 0
        self.step_ms = None
        self.fwd_ms = None
        self.opt_ms = None
        self.iteration = 0
        self.history: list[dict] = []
        self.field_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "out", "w141", "settled.npz")
        self._op_cache: dict[float, dict] = {}
        #: beat 5 -- the global power residual, accumulated as the march runs
        self.balance = Balance(F.MACRO_DT)
        #: beat 2 -- what the last decline was, kept after the reset so the
        #: screen can say WHICH expert said no and at which macro-step
        self.decline: dict | None = None
        self._build(reset=True)
        # the certification panel, computed on its own worker
        self.cert: dict = {"state": "pending", "seams": {}, "verdict": None}
        self._cert_lock = threading.Lock()
        self._cert_thread: threading.Thread | None = None
        self._cert_want = 0
        self._cert_done = -1
        #: beats 1, 3 and 4 -- each is seconds to minutes, so each runs on its
        #: own worker and streams progress rather than blocking the march
        self.tasks: dict[str, dict] = {}
        #: >0 while a beat that MEASURES A CLOCK is running, which holds the
        #: live march so the beat is timing itself rather than the contention
        self._quiet = 0
        self._task_lock = threading.Lock()
        self._task_threads: dict[str, threading.Thread] = {}
        self.recorded = load_recorded()
        #: beat 4's measured-reach bars, from Tier 33's artifact rather than
        #: from this run -- absent, the beat shows its three verdicts alone
        self.horizon = load_horizon()
        self.request_compile()

    # -- state -------------------------------------------------------------

    def _build(self, reset: bool) -> None:
        tiling = (F.SINGLE_TILING if self.cfg.tiling == "single"
                  else F.DEFAULT_TILING)
        self.ro = F.FrontWingRollout(tiling=tiling, coupling=self.cfg.coupling,
                                     lag=self.cfg.lag, motion=True,
                                     design=self.design)
        if not reset:
            return
        opt = dict(dtype=F.TORCH_DTYPE)
        if os.path.isfile(self.field_path):
            d = np.load(self.field_path)
            self.u = torch.as_tensor(d["u"], **opt)
            self.v = torch.as_tensor(d["v"], **opt)
            self.notice = "released from the settled fixed-shape field"
        else:
            self.u = torch.full((self.ro.ny, self.ro.nx), F.U_INF, **opt)
            self.v = torch.zeros((self.ro.ny, self.ro.nx), **opt)
            self.notice = ("no settled field on disk: released from the "
                           "freestream, which is not the state any number on "
                           "the results page was measured at")
        self.delta = torch.zeros(F.N_STATION, **opt)
        kn = self.ro.knobs()
        self.h = F.FrontWingRollout.release_height(kn["h0"], kn["k"]).clone()
        self.w_delta = torch.zeros(F.N_STATION, **opt)
        self.v_mount = torch.zeros((), **opt)
        self.load = self.drag = self.vm = 0.0
        self.trace: list[dict] = []
        self.iteration = 0
        self.outside: dict | None = None
        #: the balance restarts with the march: its baseline is the energy at
        #: RELEASE, and carrying one from a previous design would difference two
        #: different systems
        self.balance.reset()

    # -- the message queue --------------------------------------------------

    def post(self, kind: str, payload: Any = None) -> None:
        self._q.put((kind, payload))

    def _drain(self) -> None:
        while True:
            try:
                kind, payload = self._q.get_nowait()
            except queue.Empty:
                return
            self._apply(kind, payload)

    def _apply(self, kind: str, payload: Any) -> None:
        if kind == "mode":
            self.mode = payload
        elif kind == "design":
            d = dict(self.design)
            for k, v in (payload or {}).items():
                if k in F.DESIGN_BOX:
                    d[k] = float(_clamp(float(v), *F.DESIGN_BOX[k]))
            if d != self.design:
                self.design = d
                self._build(reset=False)
                # the ride height is a STATE, so a change of h0 or k does not
                # teleport it -- the suspension moves it, which is the coupling
                self.request_compile()
        elif kind == "config":
            cur = asdict(self.cfg)
            cur.update({k: v for k, v in (payload or {}).items() if k in cur})
            new = DemoConfig(**cur).clamped()
            reset = new.tiling != self.cfg.tiling
            self.cfg = new
            self._build(reset=reset)
            self.request_compile()
        elif kind == "reset":
            self._build(reset=True)
            self.request_compile()
        elif kind == "compile":
            self.request_compile()
        elif kind == "substitution":
            self.request_substitution()
        elif kind == "ablation":
            self.request_ablation((payload or {}).get("steps"))
        elif kind == "race":
            self.request_race()

    # -- the certification panel, on its own worker -------------------------

    def request_compile(self) -> None:
        with self._cert_lock:
            self._cert_want += 1
            want = self._cert_want
            self.cert = dict(self.cert, state="running")
        if self._cert_thread is None or not self._cert_thread.is_alive():
            self._cert_thread = threading.Thread(
                target=self._compile_loop, daemon=True)
            self._cert_thread.start()
        return want

    def _compile_loop(self) -> None:
        while not self._stop.is_set():
            with self._cert_lock:
                want, done = self._cert_want, self._cert_done
                design = dict(self.design)
            if want == done:
                return
            t0 = time.perf_counter()
            try:
                payload = self._compile_now(design)
                payload["wall_s"] = time.perf_counter() - t0
            except Exception as exc:                        # pragma: no cover
                payload = dict(state="error", error=str(exc)[:300], seams={})
            with self._cert_lock:
                self._cert_done = want
                self.cert = payload

    def _compile_now(self, design: dict) -> dict:
        u = self.u.detach().cpu().numpy()
        v = self.v.detach().cpu().numpy()
        h = float(self.h.detach())
        out = {}
        for tag, motion in (("riding", True), ("fixed-shape", False)):
            g, _e = F.build(u, v, motion=motion, design=design, h=h,
                            tiling=self.ro.tiling)
            r = compile_scheme(g)
            out[tag] = dict(verdict=r.verdict.value,
                            seams=seam_verdicts(g, r),
                            unmeasured=sorted(getattr(r, "unmeasured", ()) or ()))
        live = out["riding"]
        return dict(state="ok", verdict=live["verdict"], seams=live["seams"],
                    unmeasured=live["unmeasured"],
                    fixed_shape=out["fixed-shape"]["seams"],
                    fixed_shape_verdict=out["fixed-shape"]["verdict"],
                    design=design, notes=dict(RULE_NOTE))

    # -- beats 1, 3 and 4, each on its own worker ---------------------------

    def _set_task(self, name: str, **fields) -> None:
        with self._task_lock:
            cur = dict(self.tasks.get(name) or {})
            cur.update(fields)
            self.tasks[name] = cur

    def task(self, name: str) -> dict:
        with self._task_lock:
            return dict(self.tasks.get(name) or {"state": "idle"})

    def _spawn(self, name: str, fn, quiet: bool = False) -> bool:
        """Run `fn` on a worker named `name`, one at a time.

        A second request while one is running is **refused rather than queued**:
        these tasks read the live field, and a queued one would run against a
        state the user has since changed without any way to know it had.

        `quiet` suspends the live march for the duration, and it is **not a
        speed optimisation**. Beat 1 reports a wall-clock ratio between two
        search columns; a 12-frames-per-second march in the background steals
        from both, unevenly, and the number it produces would be a measurement
        of the contention rather than of the search. Beat 3's three rollouts
        have the same problem in a milder form. Measured: the race went from
        about 4% of its budget in 25 seconds to finishing, because the march was
        taking roughly three quarters of the machine. The screen says the march
        is held.
        """
        th = self._task_threads.get(name)
        if th is not None and th.is_alive():
            return False
        self._set_task(name, state="running", progress=0.0, error=None,
                       started=time.time(), quiet=quiet)

        def _run():
            t0 = time.perf_counter()
            if quiet:
                self._quiet += 1
            try:
                payload = fn()
                payload["wall_s"] = time.perf_counter() - t0
                self._set_task(name, state="done", progress=1.0, **payload)
            except Exception as exc:                        # pragma: no cover
                self._set_task(name, state="error", progress=1.0,
                               error=f"{type(exc).__name__}: {exc}"[:300],
                               wall_s=time.perf_counter() - t0)
            finally:
                if quiet:
                    self._quiet -= 1

        self._task_threads[name] = threading.Thread(target=_run, daemon=True)
        self._task_threads[name].start()
        return True

    def request_substitution(self) -> bool:
        """Beat 4 -- compile the graph once per candidate expert.

        Declaration-level and therefore fast: a few compiles, no marching, no
        weights. It reads the live field only so the graph is built at the state
        on screen; the verdicts do not depend on it, which is a measured claim
        (`compile` stage: zero of sixteen box corners change the map) and not an
        assumption this panel is making.
        """
        u = self.u.detach().cpu().numpy()
        v = self.v.detach().cpu().numpy()
        design = dict(self.design)
        h = float(self.h.detach())
        tiling = self.ro.tiling

        def _fn():
            r = SUB.evaluate(u, v, design, h, tiling=tiling, motion=True)
            r["notes"] = dict(RULE_NOTE)
            return r

        return self._spawn("substitution", _fn)

    def request_ablation(self, steps: int | None = None) -> bool:
        """Beat 3 -- freeze each seam in turn and price it.

        Three full rollouts, so it is the slowest thing on this screen and it
        streams progress. It starts from the field on screen and pins the frozen
        suspension at the height the LIVE design settles to -- which is the whole
        subtlety, and `ablation.py` says what it costs to get wrong.
        """
        u = self.u.detach().cpu().numpy()
        v = self.v.detach().cpu().numpy()
        design = dict(self.design)
        h_live = float(self.h.detach())
        n = int(steps or self.cfg.ablation_steps)

        def _fn():
            def _prog(tag, done, total):
                self._set_task("ablation", progress=done / max(1, total),
                               doing=tag)
            #: the settled height is MEASURED inside, from the live column
            #: that call marches first; the engine's instantaneous height goes
            #: in as a hint only, and the panel shows how far apart they are
            return AB.freeze_each_seam(design, n, u, v, h_hint=h_live,
                                       tiling=F.SINGLE_TILING, progress=_prog)

        return self._spawn("ablation", _fn, quiet=True)

    def request_race(self) -> bool:
        """Beat 1 -- the gradient and the population, on one clock."""
        u = self.u.detach().cpu().numpy()
        v = self.v.detach().cpu().numpy()
        cfg = self.cfg
        start = dict(self.design)

        def _fn():
            def _emit(snap):
                total = cfg.race_grad + cfg.race_pop
                done = snap["gradient"]["n"] + snap["population"]["n"]
                self._set_task("race", progress=min(1.0, done / max(1, total)),
                               live=snap)
            return RA.race(u, v, steps=cfg.race_steps, n_grad=cfg.race_grad,
                           n_pop=cfg.race_pop, weight=cfg.penalty,
                           start=start, emit=_emit,
                           should_stop=lambda: self._stop.is_set())

        return self._spawn("race", _fn, quiet=True)

    # -- the march ----------------------------------------------------------

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self, join_s: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(join_s)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._drain()
            try:
                if self._quiet > 0:
                    #: a beat that measures a clock has the machine. Frames keep
                    #: going out so the progress bar moves; the physics does not
                    self._publish()
                    time.sleep(0.12)
                    continue
                if self.mode in ("optimize", "step"):
                    self._optimise_once()
                    if self.mode == "step":
                        self.mode = "paused"
                elif self.mode == "run":
                    self._march_once()
                else:
                    self._publish()
                    time.sleep(0.05)
            except RuntimeError as exc:
                # an expert DECLINING is not the same as the scheme diverging,
                # and the two look identical from outside a try block -- CS-12
                # section 7.1's rule, on screen
                msg = str(exc)
                #: **`suspension` is tested FIRST**, because both envelope
                #: messages contain the word "envelope" and testing that first
                #: labelled every ride-height decline as a structural one
                kind = ("the suspension expert declined" if "suspension" in msg
                        else "the structural expert declined" if "envelope" in msg
                        else "the scheme went non-finite" if "finite" in msg
                        else "the march stopped")
                self.notice = f"{kind}: {msg[:220]}"
                #: beat 2 -- kept ACROSS the reset, because the reset is what
                #: makes it invisible otherwise: the state that caused it is
                #: gone by the time the next frame is drawn
                self.decline = dict(
                    kind=kind, expert=("suspension" if "suspension" in msg
                                       else "structure" if "envelope" in msg
                                       else "scheme"),
                    step=self.iteration, message=msg[:400],
                    design=dict(self.design), at=time.time(),
                    was_optimising=self.mode in ("optimize", "step"))
                self.mode = "paused"
                self._build(reset=True)
                self._publish()
                time.sleep(0.3)
            except Exception as exc:                        # pragma: no cover
                self.notice = f"simulation error: {exc}"
                self.mode = "paused"
                time.sleep(0.2)

    def _step_once(self, grad: bool = False):
        kn = self.ro.knobs()
        return self.ro.macro_step(self.u, self.v, self.delta, self.h,
                                  self.w_delta, self.v_mount,
                                  kn["e_star"], kn["tc"], kn["k"], kn["h0"],
                                  self.iteration)

    def _march_once(self) -> None:
        t0 = time.perf_counter()
        with torch.no_grad():
            (u, v, delta, h, wd, vm_, load, drag, f, vm,
             res, r0, p_mean, p_half) = self._step_once()
        self._absorb(u, v, delta, h, wd, vm_, load, drag, f, vm, res,
                     p_mean, p_half)
        self.step_ms = (time.perf_counter() - t0) * 1000.0
        self.fwd_ms = _ema(self.fwd_ms, self.step_ms)
        self._publish()

    def _absorb(self, u, v, delta, h, wd, vm_, load, drag, f, vm, res,
                power=None, half=None) -> None:
        #: **W145, on the demo's own paths.** Both `_march_once` and
        #: `_optimise_once` drive `macro_step` directly, so neither went through
        #: `run`'s envelope checks -- the live optimiser could walk the wing
        #: through the suspension's declared floor and the screen would show a
        #: downforce for a design the model declines. Every path funnels through
        #: `_absorb`, so the check goes here, once. `_loop` already catches the
        #: `RuntimeError` and puts the reason on screen.
        #:
        #: `cfg.enforce` is beat 2's switch, and turning it OFF is the ONLY way
        #: to reach the pre-W145 behaviour from this screen. It does not remove
        #: the check; it records what the check WOULD have said and marches on,
        #: so the panel can show the number the model declines to stand behind
        #: beside the reason it declines. `outside` stays set until the design
        #: comes back inside, because a design that left the envelope at step 4
        #: is still outside it at step 40.
        try:
            self.ro.check_envelopes(self.iteration, delta.detach(),
                                    float(h.detach()),
                                    float(self.design["h0"]))
            self.outside = None
        except RuntimeError:
            if self.cfg.enforce:
                raise
            import sys
            self.outside = dict(step=self.iteration,
                                why=str(sys.exc_info()[1])[:240])
        self.u, self.v = u.detach(), v.detach()
        self.delta = delta.detach()
        self.h = h.detach()
        self.w_delta = wd.detach()
        self.v_mount = vm_.detach()
        self.load = float(load.detach())
        self.drag = float(drag.detach())
        self.vm = float(vm.detach())
        self.traction = f.detach()
        self.residual = float(res.detach())
        self.iteration += 1
        #: beat 5 -- one macro-step's contribution to the global power balance.
        #: Both receivers' energies, and the spring's is the larger of the two at
        #: release; see `balance.py` for why forgetting it reads 1.81.
        if power is not None and half is not None:
            kn = self.ro.knobs()
            S_e0, _m = self.ro.structure(kn["e_star"], kn["tc"])
            self.balance.observe(
                strain_energy(self.delta.cpu().numpy(),
                              S_e0.detach().cpu().numpy(), float(self.ro.ds)),
                spring_energy(float(self.design["k"]), float(self.design["h0"]),
                              float(self.h)),
                float(power.detach() if torch.is_tensor(power) else power),
                float(half.detach() if torch.is_tensor(half) else half))
        self.trace.append(dict(i=self.iteration, load=self.load, drag=self.drag,
                               h=float(self.h), tip=float(self.delta[-1]),
                               vm=self.vm))
        if len(self.trace) > 600:
            del self.trace[:200]

    def _optimise_once(self) -> None:
        """One Adam step, differentiated through the next `horizon` macro-steps.

        The rollout IS the live march: it starts from the state on screen and
        the state it ends at is the state that stays, so the field animates
        through the gradient rather than freezing until it lands.  That makes it
        a warm-started short-horizon estimator and NOT the one
        `w141_poc2_frontwing.py` measures with -- the screen says so.
        """
        t0 = time.perf_counter()
        opt = dict(dtype=F.TORCH_DTYPE)
        leaves = {k: torch.tensor(float(self.design[k]), requires_grad=True, **opt)
                  for k in F.DESIGN_KEYS}
        u, v = self.u, self.v
        delta, h = self.delta, self.h
        wd, vm_ = self.w_delta, self.v_mount
        loads, vms, dmax = [], [], []
        for s in range(self.cfg.horizon):
            out = self.ro.macro_step(u, v, delta, h, wd, vm_,
                                     leaves["e_star"], leaves["tc"],
                                     leaves["k"], leaves["h0"],
                                     self.iteration + s)
            (u, v, delta, h, wd, vm_, load, drag, f, vm, res, r0,
             p_mean, p_half) = out
            loads.append(load); vms.append(vm)
            dmax.append(torch.max(torch.abs(delta)))
        J = torch.stack(loads).mean()
        sig, dl = torch.stack(vms), torch.stack(dmax)
        #: the SMOOTH margins drive the step and the TRUE ones are what the
        #: readout calls feasible -- `logsumexp` overshoots the maximum it
        #: smooths, and a constraint is not the place to leave that unmeasured
        g_sig = (F._softmax_over(sig, F.SIGMA_CEIL * F.SOFTMAX_FRAC)
                 / F.SIGMA_CEIL - 1.0)
        g_del = (F._softmax_over(dl, F.DELTA_CEIL * F.SOFTMAX_FRAC)
                 / F.DELTA_CEIL - 1.0)
        L = F._penalised(J, g_sig, g_del, self.cfg.penalty)
        grads = torch.autograd.grad(L, list(leaves.values()), allow_unused=True)
        self._absorb(u, v, delta, h, wd, vm_, load, drag, f, vm, res,
                     p_mean, p_half)
        self.margins = dict(
            smooth_sigma=float(g_sig), smooth_delta=float(g_del),
            true_sigma=float(sig.max()) / F.SIGMA_CEIL - 1.0,
            true_delta=float(dl.max()) / F.DELTA_CEIL - 1.0)
        # Adam, in the box's own unit cube so one learning rate serves knobs
        # whose scales differ by four orders
        st = getattr(self, "_adam", None)
        if st is None or st["n"] != len(leaves):
            st = self._adam = dict(m=np.zeros(len(leaves)),
                                   v=np.zeros(len(leaves)), t=0,
                                   n=len(leaves))
        span = np.array([F.DESIGN_BOX[k][1] - F.DESIGN_BOX[k][0]
                         for k in F.DESIGN_KEYS])
        g = np.array([0.0 if x is None else float(x) for x in grads]) * span
        st["t"] += 1
        st["m"] = 0.9 * st["m"] + 0.1 * g
        st["v"] = 0.999 * st["v"] + 0.001 * g * g
        mh = st["m"] / (1 - 0.9 ** st["t"])
        vh = st["v"] / (1 - 0.999 ** st["t"])
        x = np.array([(self.design[k] - F.DESIGN_BOX[k][0]) / span[i]
                      for i, k in enumerate(F.DESIGN_KEYS)])
        x = np.clip(x + self.cfg.lr * mh / (np.sqrt(vh) + 1e-8), 0.0, 1.0)
        self.post("design", {k: F.DESIGN_BOX[k][0] + x[i] * span[i]
                             for i, k in enumerate(F.DESIGN_KEYS)})
        self.history.append(dict(t=st["t"], J=float(J),
                                 g_sigma=float(g_sig), g_delta=float(g_del),
                                 **dict(self.design)))
        if len(self.history) > 400:
            del self.history[:150]
        self.step_ms = (time.perf_counter() - t0) * 1000.0
        self.opt_ms = _ema(self.opt_ms, self.step_ms)
        self._publish()

    # -- what the page draws -------------------------------------------------

    def geometry(self) -> dict:
        """The plate's own quads and their von Mises, in DOMAIN coordinates.

        Built from `FlexWing.stations(delta, h)` -- the same call the physics
        makes -- so the picture is the model's geometry and not a second one.
        """
        wing = self.ro.wing
        d = self.delta.detach()
        h = self.h.detach()
        cx, cy = wing.stations(d, h)
        cx = cx.cpu().numpy(); cy = cy.cpu().numpy()
        nx_h, ny_h = float(wing.n_hat[0]), float(wing.n_hat[1])
        half = 0.5 * self.design["tc"] * F.CHORD
        vm = self._von_mises_stations()
        quads = []
        for k in range(len(cx) - 1):
            x0, y0, x1, y1 = cx[k], cy[k], cx[k + 1], cy[k + 1]
            quads.append(dict(
                p=[[x0 + half * nx_h, y0 + half * ny_h],
                   [x1 + half * nx_h, y1 + half * ny_h],
                   [x1 - half * nx_h, y1 - half * ny_h],
                   [x0 - half * nx_h, y0 - half * ny_h]],
                vm=float(vm[k]), colour=stress_colour(vm[k])))
        return dict(quads=quads, chord_x=[float(cx[0]), float(cx[-1])],
                    chord_y=[float(cy[0]), float(cy[-1])],
                    thickness=2 * half, n=len(quads))

    def _von_mises_stations(self) -> np.ndarray:
        """Per chordwise station: the max over the two elements through the
        thickness.  `sigma_map` is the expert's own answer to a unit station
        traction, so this re-enters no solver."""
        tc = float(self.design["tc"])
        key = round(tc, 12)
        op = self._op_cache.get(key)
        if op is None:
            op = self._op_cache[key] = F.surface_operator(tc)
            if len(self._op_cache) > 24:
                self._op_cache.pop(next(iter(self._op_cache)))
        q = getattr(self, "traction", None)
        if q is None:
            return np.zeros(F.N_STATION)
        sig = np.einsum("k,kea->ea", q.cpu().numpy(), op["sigma_map"])
        vm = F.von_mises(sig)
        return vm.reshape(F.N_STATION, -1).max(axis=1)

    def seam_quantities(self) -> dict:
        """What actually MOVES with the design, under a verdict that does not.

        Both are one-sidedness ratios -- the partner's block against the fluid's
        -- which is the quantity `SubstitutionCertificate` is blind in proportion
        to (W97, W137).  They are computed from the design and the current
        traction rather than probed, because a probe is seconds and this panel is
        per frame; the driver's `compile` stage probes them properly.
        """
        kn = self.ro.knobs()
        S_e, _m = self.ro.structure(kn["e_star"], kn["tc"])
        q = getattr(self, "traction", None)
        w = float(np.abs(q.cpu().numpy()).mean()) if q is not None else 0.0
        # the fluid's own aerodynamic damping at the seam, diag(C_N |w|)
        fluid = F.C_N * max(w, 1e-30) ** 0.5
        struct = float(torch.linalg.matrix_norm(S_e, 2))
        return {
            "wet": dict(label="structure / fluid, operator norm",
                        value=struct / max(fluid, 1e-30),
                        note="W137: the fluid's share of this seam is the "
                             "reciprocal, and SubstitutionCertificate is blind "
                             "in proportion to it"),
            "mount": dict(label="spring / fluid, rate",
                          value=float(self.design["k"]) / max(fluid, 1e-30),
                          note="W97 closed the same reading at CS-10's seam by "
                               "repairing an effort convention"),
        }

    def eta_s(self):
        ms = self.opt_ms if self.mode in ("optimize", "step") else self.fwd_ms
        return None if ms is None else ms / 1000.0

    def _publish(self) -> None:
        self._seq += 1
        u = self.u.detach().cpu().numpy()
        v = self.v.detach().cpu().numpy()
        png = field_png(u, v, self.cfg.stride)
        with self._cert_lock:
            cert = dict(self.cert)
        tip = float(self.delta[-1]) if self.delta is not None else 0.0
        dmax = float(torch.max(torch.abs(self.delta)))
        payload = dict(
            seq=self._seq, mode=self.mode, notice=self.notice,
            iteration=self.iteration,
            design=dict(self.design), box={k: list(v) for k, v in
                                           F.DESIGN_BOX.items()},
            config=asdict(self.cfg),
            readout=dict(
                downforce=self.load, drag=self.drag,
                lift_to_drag=(self.load / self.drag) if self.drag else 0.0,
                ride_height=float(self.h), tip=tip, delta_max=dmax,
                vm_max=self.vm, vm_ceiling=F.SIGMA_CEIL,
                delta_ceiling=F.DELTA_CEIL,
                delta_envelope=F.DELTA_MAX,
                stress_margin=self.vm / F.SIGMA_CEIL - 1.0,
                deflection_margin=dmax / F.DELTA_CEIL - 1.0,
                interface_residual=getattr(self, "residual", 0.0),
                u_max=float(np.hypot(u, v).max())),
            geometry=self.geometry(),
            seam_quantities=self.seam_quantities(),
            certification=cert,
            #: beat 5 -- the power balance, and beat 2's two fields. `enforce`
            #: is repeated at the top level so no panel has to reach into the
            #: config to know whether the numbers beside it are inside the model
            balance=self.balance.payload(),
            enforce=bool(self.cfg.enforce),
            quiet=self._quiet > 0,
            outside=self.outside,
            decline=self.decline,
            tasks={k: self.task(k) for k in ("substitution", "ablation", "race")},
            trace=self.trace[-240:],
            history=self.history[-160:],
            step_ms=self.step_ms, eta_s=self.eta_s(),
            domain=dict(nx=int(u.shape[1]), ny=int(u.shape[0]), dx=F.DX,
                        stride=self.cfg.stride),
            ramp=dict(lo=U_LO, hi=U_HI,
                      stops=[[a, list(c)] for a, c in RAMP_STOPS],
                      stress=[[a, list(c)] for a, c in STRESS_STOPS]),
        )
        self.frame = Frame(seq=self._seq, png=png, width=u.shape[1],
                           height=u.shape[0], payload=payload)
