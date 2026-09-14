"""Tier 52 -- PoC 3 RaceLab, phase 2: the switch.

`POC3-RACELAB-REQUIREMENTS.md` section 9's phase 2: **per-window mode selection,
classical / learned / certified, with per-window one-step error and per-call
cost**, and a headless path that runs any mode assignment and reports section
5.2 and section 5.3.  The dashboard is phase 3; nothing here draws anything.

The three modes, and what each one is
-------------------------------------

  ``CLASSICAL``  `reference.WindowNS` with its elliptic part exposed -- the
                 column CS-19 marched, and the referent every other mode is
                 scored against.
  ``LEARNED``    Poseidon-T's frozen checkpoint on that window, at the scaling
                 the window's own geometry forces.
  ``CERTIFIED``  Poseidon-T inside `atlas.defect_correction`, whose limit is
                 the classical map's own fixed point whatever the learned map
                 does (Theorem 1).  **It is the one accuracy claim in the whole
                 demo that rests on a proof rather than a measurement.**

The scaling is OVER-DETERMINED, and that is the first measurement
----------------------------------------------------------------

`adapters.Scaling` declares ``length`` as the tiling units one 128-cell window
spans.  That is geometry: 128 cells at ``dx = 1/64`` is **2.0** length units and
there is no choice in it.  Then ``time = length / velocity`` and
``lead = dt / time`` follow, and the checkpoint's native lead is ``0.1``.  So:

  ============================  ==========  ===================================
  cadence                       lead        against the native 0.1
  ============================  ==========  ===================================
  one exchange (`dt_ex`)        0.003125    **1/32** of native
  one macro-step                0.0125      **1/8** of native
  eight macro-steps             0.1         native
  ============================  ==========  ===================================

`wake_array`'s column exchanges once per macro-step at the native lead; this
one exchanges four times per macro-step at an eighth of it, because
`ground_effect`'s tiling marches eight times finer in time per unit space.
**A learned expert dropped into this composition layer is asked for a step
thirty-two times shorter than the one it was trained to take**, which is the
thing it is worst at, and `error_vs_lead` measures what that costs rather than
asserting it.

What this file must not do
--------------------------

Tiers 48 and 50 measured that in this project's certified slot the learned
expert does not pay, and that its learned content is worth about 13 classical
calls against a 69-call spread from perturbing its own weights by 3%.  The
requirements' section 4.2 is explicit: **RaceLab shows the mechanism working and
reports the price, including when the price is bad.**  Nothing here is allowed
to imply otherwise, and `tests/test_tier52_racelab_switch.py` asserts the
vocabulary.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

import numpy as np

from .. import defect_correction as DC
from . import ground_effect as GE
from . import racelab as RL
from . import wake_array as WA
from . import wing_fsi as W

__all__ = [
    "Mode", "MODES", "LEARNED_FAMILIES", "NO_LEARNED_OPTION",
    "WindowStack", "one_step_errors", "error_vs_lead", "per_call_cost",
    "certified_window", "family_switch_table", "assignment_from",
    "MixedRollout", "march_mixed", "switch_report",
]


class Mode(str, Enum):
    """What runs in one window this macro-step."""

    CLASSICAL = "classical"
    LEARNED = "learned"
    CERTIFIED = "certified"


MODES = tuple(m.value for m in Mode)

#: The one governing family this project has a shippable learned expert for.
LEARNED_FAMILIES = ("incompressible-navier-stokes-2d",)

#: **Four of five families have no learned option, and the switch must say so
#: rather than hide it** (requirements section 4.3).  A viewer should be able to
#: learn this from the demo, because it is a true and important fact about the
#: field; `physics-simulation-datasets` section 3.4 records why.
NO_LEARNED_OPTION: dict[str, str] = {
    "plane-stress-elasticity-2d":
        "no shippable learned structural expert exists in this project. "
        "NeuberNet is unlicensed and local-only and must not appear in a "
        "bundle (requirements section 2.1); anything else would have to be "
        "trained from scratch and licensed by us",
    "heat-conduction-2d":
        "no learned conduction operator is vendored, and the block's solver is "
        "a 2-D FE conduction step with no public checkpoint at this geometry",
    "incompressible-thermal-transport-1d":
        "the coolant legs are one-dimensional advection with closed-form "
        "decay; there is no operator to learn that is cheaper than the "
        "closed form",
    "lumped-dc-circuit":
        "the circuit is algebraic -- `CircuitSolve` is a bisection on two "
        "scalar equations -- so a learned surrogate would be replacing "
        "microseconds of arithmetic",
}


# ===========================================================================
# the window stack: one input state, three maps
# ===========================================================================


@dataclass
class WindowStack:
    """The three experts on one tiling's windows, sharing one input state.

    **The point of the class is that all three see the SAME state.**  Section
    5.4's third clause: *"per-window one-step error needs no referent at all
    and is the number to trust most"* -- which is only true if the learned and
    certified maps are handed exactly what the classical one was handed, rather
    than a state that has already drifted.
    """

    tiling: Any = None
    nu: float = GE.NU
    device: str = "cpu"
    threads_classical: int = 4
    threads_learned: int = 4
    velocity_scale: float = 2.0
    load_expert: bool = True

    def __post_init__(self) -> None:
        import torch
        if self.tiling is None:
            self.tiling, _i = RL.layout()
        t = self.tiling
        self.solver = GE.solver_for(t.wx, t.wy, self.nu, self.device)
        self.dt_ex = GE.MACRO_DT / GE.EXCHANGES
        self._opt = dict(dtype=W.TORCH_DTYPE, device=self.device)
        #: **The span is geometry and not a knob.**  One window is `wx` cells at
        #: `dx`, so it spans `wx * dx` tiling length units, and the scaling's
        #: `length` is that number.  Declaring anything else would be declaring
        #: a different window.
        self.span = t.wx * RL.DX
        self.ex = None
        self.scaling = None
        self.native_lead = None
        if self.load_expert:
            WA.load_reference()
            ad = importlib.import_module("atlas_windfarm_reference.adapters")
            self._ad = ad
            self.ex = WA.scaled_expert(threads=self.threads_learned)
            self.scaling = ad.Scaling(length=self.span,
                                      velocity=self.velocity_scale)
            self.ex.scaling = self.scaling
            self.native_lead = float(getattr(ad, "EXPERT_NATIVE_DT", 0.1))
        self.calls = {m: 0 for m in MODES}

    # -- the scaling, reported rather than assumed -------------------------

    def scaling_report(self) -> dict[str, Any]:
        """What this window's geometry forces, and where it sits against the
        checkpoint's trained envelope."""
        if self.scaling is None:
            return {"available": False,
                    "why": "the expert was not loaded"}
        out: dict[str, Any] = {
            "available": True,
            "window_cells": self.tiling.wx,
            "window_span_length_units": self.span,
            "scaling_length": self.scaling.length,
            "scaling_velocity": self.scaling.velocity,
            "scaling_time": self.scaling.time,
            "native_lead": self.native_lead,
            "why_length_is_not_a_knob":
                "one window is wx cells at dx, so it spans wx*dx length units; "
                "the scaling's `length` IS that number and declaring another "
                "would be declaring a different window",
        }
        for tag, dt in (("exchange", self.dt_ex),
                        ("macro_step", GE.MACRO_DT),
                        ("native", self.native_lead * self.scaling.time)):
            lead = self.scaling.lead(dt)
            out[tag] = {
                "dt": dt, "lead": lead,
                "lead_over_native": lead / self.native_lead,
                "macro_steps": dt / GE.MACRO_DT,
                **self.scaling.check(dt),
            }
        out["the_composition_layer_asks_for"] = out["exchange"]["lead_over_native"]
        return out

    # -- cutting -----------------------------------------------------------

    def cut(self, u, v):
        import torch
        t = self.tiling
        U = torch.as_tensor(np.asarray(u), **self._opt)
        V = torch.as_tensor(np.asarray(v), **self._opt)
        us = torch.stack([U[oy:oy + t.wy, ox:ox + t.wx] for ox, oy in t.offsets])
        vs = torch.stack([V[oy:oy + t.wy, ox:ox + t.wx] for ox, oy in t.offsets])
        return us, vs

    # -- the three maps ----------------------------------------------------

    def classical(self, us, vs, n_macro: int = 1, force=None):
        """`WindowNS` for ``n_macro`` macro-steps of `EXCHANGES` sub-steps."""
        import torch
        torch.set_num_threads(self.threads_classical)
        a, b = us, vs
        f = force if force is not None else None
        for _ in range(int(n_macro) * GE.EXCHANGES):
            a, b = self.solver.step_batch(a, b, self.dt_ex, bc0=None, force=f)
        self.calls["classical"] += int(n_macro) * GE.EXCHANGES
        return a, b

    def learned(self, us, vs, dt: float):
        """Poseidon-T over every window at once, at lead ``dt``.

        **The window mean is restored, and that is `w100_scaling_ladder`'s own
        line.**  The checkpoint drifts its own mean; a periodic box with no
        force conserves it, and a deficit a body puts there has to survive long
        enough to be carried out of the domain.
        """
        import torch
        if self.ex is None:
            raise RuntimeError("the learned expert was not loaded")
        torch.set_num_threads(self.threads_learned)
        ufs = (us - GE.U_INF).detach().cpu().numpy()
        vfs = vs.detach().cpu().numpy()
        a, b = self.ex.step_many(ufs, vfs, float(dt), galilean=False,
                                 project=False)
        a = a + (ufs.mean(axis=(1, 2)) - a.mean(axis=(1, 2)))[:, None, None]
        b = b + (vfs.mean(axis=(1, 2)) - b.mean(axis=(1, 2)))[:, None, None]
        self.calls["learned"] += int(us.shape[0])
        return (torch.as_tensor(GE.U_INF + a, **self._opt),
                torch.as_tensor(b, **self._opt))


# ===========================================================================
# section 5.2 -- the per-window numbers, which need no global referent
# ===========================================================================


def _rel_err(a, b, ra, rb) -> np.ndarray:
    """Per-window relative error of ``(a, b)`` against the referent ``(ra, rb)``.

    The denominator is the referent's own FLUCTUATION -- the field minus the
    free stream -- because dividing by the field itself would put a window of
    nearly uniform flow at an error of nearly zero however wrong it was, which
    is most of the domain here.
    """
    import torch
    num = torch.sqrt(((a - ra) ** 2 + (b - rb) ** 2).sum(dim=(1, 2)))
    den = torch.sqrt(((ra - GE.U_INF) ** 2 + rb ** 2).sum(dim=(1, 2)))
    return (num / torch.clamp(den, min=1e-30)).detach().cpu().numpy()


def one_step_errors(stack: WindowStack, u, v, n_macro: int = 1) -> dict[str, Any]:
    """Section 5.2's headline: each window's learned error against the classical
    expert **on the same input state**, with both wall times.

    This is exact, cheap, and needs no global referent, which is why section 5.4
    calls it the number to trust most.
    """
    us, vs = stack.cut(u, v)
    dt = n_macro * GE.MACRO_DT
    t0 = time.perf_counter()
    ca, cb = stack.classical(us.clone(), vs.clone(), n_macro)
    t_cl = time.perf_counter() - t0
    t0 = time.perf_counter()
    pa, pb = stack.learned(us, vs, dt)
    t_po = time.perf_counter() - t0
    e = _rel_err(pa, pb, ca, cb)
    names = list(stack.tiling.names)
    return {
        "n_macro": n_macro, "dt": dt,
        "lead": (stack.scaling.lead(dt) if stack.scaling else None),
        "lead_over_native": ((stack.scaling.lead(dt) / stack.native_lead)
                             if stack.scaling else None),
        "per_window": {names[k]: float(e[k]) for k in range(len(names))},
        "min": float(e.min()), "median": float(np.median(e)),
        "max": float(e.max()), "mean": float(e.mean()),
        "worst_window": names[int(np.argmax(e))],
        "best_window": names[int(np.argmin(e))],
        "classical_s": t_cl, "learned_s": t_po,
        "classical_over_learned": t_cl / t_po if t_po else None,
        "n_windows": len(names),
    }


def error_vs_lead(stack: WindowStack, u, v,
                  n_macros: Sequence[int] = (1, 2, 4, 8, 16)) -> dict[str, Any]:
    """The curve the composition cadence makes unavoidable.

    The composition layer exchanges `EXCHANGES` times a macro-step, so a
    learned expert inside it is asked for a lead of ``dt_ex`` -- a thirty-second
    of the checkpoint's native step on this tiling.  This measures the error and
    the cost across leads so that number has a curve behind it rather than an
    assertion.
    """
    rows = [one_step_errors(stack, u, v, n) for n in n_macros]
    best = min(rows, key=lambda r: r["median"])
    return {
        "rows": rows,
        "best_median_at_n_macro": best["n_macro"],
        "best_median": best["median"],
        "best_lead_over_native": best["lead_over_native"],
        "note": "the composition layer's own cadence is one EXCHANGE, which is "
                "1/%d of a macro-step and so 1/%d of the checkpoint's native "
                "lead on this tiling"
                % (GE.EXCHANGES, GE.EXCHANGES * 8),
    }


def per_call_cost(stack: WindowStack, u, v, n_macro: int = 8,
                  thread_counts: Sequence[int] = (1, 2, 4, 8),
                  repeats: int = 2) -> dict[str, Any]:
    """Each column at its OWN best thread count, which is the only fair ratio.

    **Measured, because the two columns do not agree.**  CS-19 measured the
    car's full march fastest at one thread; the bare window solve and a
    transformer forward pass are different objects.  Pinning both to the count
    that suits one of them is the same defect as quoting a wall-clock number
    with no control, and on this host it moves the headline by 1.7x.
    """
    import torch
    us, vs = stack.cut(u, v)
    dt = n_macro * GE.MACRO_DT
    rows = []
    for n in thread_counts:
        stack.threads_classical = stack.threads_learned = n
        if stack.ex is not None:
            WA.scaled_expert.cache_clear()
            stack.ex = WA.scaled_expert(threads=n)
            stack.ex.scaling = stack.scaling
        stack.classical(us.clone(), vs.clone(), 1)             # warm
        c = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            stack.classical(us.clone(), vs.clone(), n_macro)
            c.append(time.perf_counter() - t0)
        p = []
        if stack.ex is not None:
            stack.learned(us, vs, dt)                          # warm
            for _ in range(repeats):
                t0 = time.perf_counter()
                stack.learned(us, vs, dt)
                p.append(time.perf_counter() - t0)
        rows.append({"threads": n,
                     "classical_s": float(np.median(c)),
                     "learned_s": (float(np.median(p)) if p else None),
                     "repeats": repeats})
    bc = min(rows, key=lambda r: r["classical_s"])
    bl = (min((r for r in rows if r["learned_s"] is not None),
              key=lambda r: r["learned_s"]) if stack.ex is not None else None)
    torch.set_num_threads(4)
    return {
        "rows": rows, "n_macro": n_macro,
        "classical_best": bc, "learned_best": bl,
        "ratio_each_at_its_own_best": (bc["classical_s"] / bl["learned_s"]
                                       if bl else None),
        "reading": "greater than one means the learned column is faster than "
                   "the classical column it replaces, at that many macro-steps",
    }


# ===========================================================================
# the certified mode -- the one claim that rests on a proof
# ===========================================================================


def certified_window(stack: WindowStack, u, v, window: str,
                     r_stop: float = 1e-6, k_max: int = 12,
                     alpha: float = 0.5, use_learned: bool = True
                     ) -> dict[str, Any]:
    """Defect correction on ONE window, with the classical-only column beside it.

    ``phi`` is one classical macro-step on this window and the state sought is
    its fixed point; ``psi`` is Poseidon-T, shrunk by ``alpha``, or the identity
    when ``use_learned`` is False -- which is **W76's null replacement** and the
    control that says what the checkpoint contributed.

    **Theorem 1 is what makes this mode worth having**: the limit is the
    classical map's own fixed point whatever the learned map does.  What is
    measured here is only the PRICE of getting there.
    """
    import torch
    t = stack.tiling
    k = t.names.index(window)
    us, vs = stack.cut(u, v)
    w0 = np.stack([us[k].detach().cpu().numpy(), vs[k].detach().cpu().numpy()])

    def phi(w):
        a = torch.as_tensor(w[0][None], **stack._opt)
        b = torch.as_tensor(w[1][None], **stack._opt)
        a, b = stack.classical(a, b, 1)
        return np.stack([a[0].detach().cpu().numpy(),
                         b[0].detach().cpu().numpy()])

    if use_learned and stack.ex is not None:
        def psi_raw(w):
            a = torch.as_tensor(w[0][None], **stack._opt)
            b = torch.as_tensor(w[1][None], **stack._opt)
            a, b = stack.learned(a, b, GE.MACRO_DT)
            return np.stack([a[0].detach().cpu().numpy(),
                             b[0].detach().cpu().numpy()])
        psi = DC.shrink(psi_raw, alpha)
    else:
        psi = DC.shrink(lambda w: w, alpha)

    t0 = time.perf_counter()
    res = DC.defect_correct(phi, psi, w0, r_stop=r_stop, k_max=k_max)
    wall = time.perf_counter() - t0
    stack.calls["certified"] += 1
    out = res.as_dict()
    out.update(window=window, wall_s=wall, alpha=alpha,
               use_learned=bool(use_learned and stack.ex is not None),
               r_stop=r_stop, k_max=k_max)
    return out


# ===========================================================================
# section 4.3 -- which families have a learned option, and which do not
# ===========================================================================


def family_switch_table(graph) -> dict[str, Any]:
    """Per agent: is the switch available, and if not, WHY.

    Requirements section 4.3: the windows with no learned option show the switch
    **greyed out with a reason, not hidden** -- *"a viewer should learn from the
    dashboard that four of five families have no learned option, that is a true
    and important fact about the field."*  Phase 3 draws this; phase 2 measures
    it, so the page and the screen cannot disagree.
    """
    rows = {}
    for a in graph.agents:
        fam = a.capabilities.governing_family
        avail = fam in LEARNED_FAMILIES
        rows[a.agent_id] = {
            "family": fam,
            "switch_available": avail,
            "modes": list(MODES) if avail else [Mode.CLASSICAL.value],
            "why_not": None if avail else NO_LEARNED_OPTION.get(
                fam, "no learned expert is declared for this family"),
        }
    fams: dict[str, int] = {}
    for r in rows.values():
        fams[r["family"]] = fams.get(r["family"], 0) + 1
    with_opt = [f for f in fams if f in LEARNED_FAMILIES]
    return {
        "per_agent": rows,
        "families": fams,
        "n_families": len(fams),
        "families_with_a_learned_option": with_opt,
        "families_without": sorted(set(fams) - set(with_opt)),
        "n_agents_switchable": sum(1 for r in rows.values()
                                   if r["switch_available"]),
        "n_agents_total": len(rows),
        "the_fact": "%d of %d governing families on this graph have a "
                    "shippable learned option, and %d of %d agents"
                    % (len(with_opt), len(fams),
                       sum(1 for r in rows.values() if r["switch_available"]),
                       len(rows)),
    }


# ===========================================================================
# section 5.3 -- a mixed march, and what it costs against all-classical
# ===========================================================================


def assignment_from(spec, tiling=None) -> dict[str, str]:
    """A per-window mode map from a preset name, a list, or a dict."""
    if tiling is None:
        tiling, _i = RL.layout()
    names = list(tiling.names)
    if isinstance(spec, dict):
        bad = set(spec) - set(names)
        if bad:
            raise ValueError(f"no such window: {sorted(bad)}")
        out = {n: Mode.CLASSICAL.value for n in names}
        out.update({k: Mode(v).value for k, v in spec.items()})
        return out
    if isinstance(spec, (list, tuple)):
        return {n: Mode(m).value for n, m in zip(names, spec)}
    spec = str(spec)
    if spec in MODES:
        return {n: spec for n in names}
    if spec == "wake-learned":
        #: the windows downstream of the car, where a learned expert is least
        #: likely to be asked about a body it has never seen
        return {n: (Mode.LEARNED.value if int(n[1]) >= 4
                    else Mode.CLASSICAL.value) for n in names}
    if spec == "upper-learned":
        #: row 1 -- above the car, where there is almost no body at all
        return {n: (Mode.LEARNED.value if n[2] == "1"
                    else Mode.CLASSICAL.value) for n in names}
    raise ValueError(f"unknown assignment preset {spec!r}")


class MixedRollout(RL.RaceRollout):
    """`RaceRollout` with a per-window expert, chosen by `assignment`.

    **The learned windows are stepped at the EXCHANGE cadence**, because that is
    what the composition layer runs at, and on this tiling that is a lead of
    1/32 of the checkpoint's native step.  That is not a choice this class makes
    -- it is what putting a learned expert inside `FSIRollout`'s composition
    layer means -- and `error_vs_lead` is what prices it.

    **`CERTIFIED` is not a per-macro-step mode and this class says so.**  Defect
    correction certifies a FIXED POINT: it finds ``w`` with ``phi(w) = w``.  An
    explicit time step has no fixed point to certify -- one call to ``phi``
    already IS the answer -- so there is nothing for the iteration to do inside
    an exchange.  The certified mode is therefore a steady-state solve per
    window (`certified_window`), which is what Tier 48 and CS-S3 measured, and
    assigning it here raises rather than quietly running something else.
    """

    def __init__(self, *a, assignment=None, stack: WindowStack = None, **kw):
        super().__init__(*a, **kw)
        self.assignment = assignment_from(assignment or Mode.CLASSICAL.value,
                                          self.tiling)
        bad = [n for n, m in self.assignment.items()
               if m == Mode.CERTIFIED.value]
        if bad:
            raise ValueError(
                "CERTIFIED is a steady-state mode and not a per-macro-step one: "
                "defect correction certifies a fixed point, and an explicit "
                "time step has none to certify. Use `certified_window` for "
                f"windows {sorted(bad)}, and see this class's docstring")
        self.stack = stack
        self._learned_idx = [i for i, n in enumerate(self.tiling.names)
                             if self.assignment[n] == Mode.LEARNED.value]
        self._classical_idx = [i for i, n in enumerate(self.tiling.names)
                               if self.assignment[n] == Mode.CLASSICAL.value]
        if self._learned_idx and self.stack is None:
            raise ValueError("a learned window needs a WindowStack")
        self.mode_calls = {m: 0 for m in MODES}

    def _step_windows(self, us, vs, fxs, fys):
        """One exchange, with each window's own expert."""
        import torch
        out_u = us.clone()
        out_v = vs.clone()
        if self._classical_idx:
            i = self._classical_idx
            a, b = self.solver.step_batch(us[i], vs[i], self.dt_ex, bc0=None,
                                          force=(fxs[i], fys[i]))
            out_u[i], out_v[i] = a, b
            self.substep_log.append(int(self.solver.last_substeps))
            self.mode_calls["classical"] += len(i)
        if self._learned_idx:
            i = self._learned_idx
            a, b = self.stack.learned(us[i], vs[i], self.dt_ex)
            #: **W99: `step_many` DROPS `force` silently.**  The partition of
            #: unity sums to one, so applying the body force after the window
            #: solve is applying it before -- `w100_scaling_ladder`'s own line.
            out_u[i] = a + fxs[i] * self.dt_ex
            out_v[i] = b + fys[i] * self.dt_ex
            self.mode_calls["learned"] += len(i)
        return out_u, out_v

    def _advance(self, u, v):
        import torch
        #: W258: the outlet's condition is the composition layer's, so it is
        #: applied whichever expert steps the windows behind it
        u, v = self.relax_outflow(u, v)
        fx, fy, load, drag = self.car_forcing(u, v)
        fdev = torch.as_tensor(self._fx_dev, **self._opt_t)
        fx = fx + fdev
        us, vs = self.cut(u), self.cut(v)
        fxs, fys = self.cut(fx), self.cut(fy)
        u1, v1 = self._step_windows(us, vs, fxs, fys)
        au, av = self.blend(u1, v1)
        au, av = self.project(au, av)
        bu, bv = self.band(au, av)
        return bu, bv, load, drag, fdev


def march_mixed(u0, v0, steps: int = 40, assignment=Mode.CLASSICAL.value,
                stack: WindowStack = None, tiling=None, **kw):
    """`racelab.march`'s loop with `MixedRollout`, returning the same record."""
    import torch
    if tiling is None:
        tiling, _i = RL.layout()
    r = MixedRollout(tiling=tiling, assignment=assignment, stack=stack, **kw)
    opt = dict(dtype=W.TORCH_DTYPE, device=r.device)
    u = torch.as_tensor(np.asarray(u0), **opt)
    v = torch.as_tensor(np.asarray(v0), **opt)
    r.refresh(u, 0, which=tuple(j for j in ("J1", "J3") if j in r.joins))
    keys = ("u_rotor", "u_core", "ua", "current", "q_machine", "load", "drag",
            "u_max")
    trace = {k: [] for k in keys}
    t0 = time.perf_counter()
    for s in range(steps):
        u, v, load, drag = r.macro_step(u, v, s)
        st = r.state
        for k in ("u_rotor", "u_core", "ua", "current", "q_machine"):
            trace[k].append(float(getattr(st, k)))
        trace["load"].append(float(load.detach()))
        trace["drag"].append(float(drag.detach()))
        trace["u_max"].append(float(torch.max(torch.hypot(u, v)).detach()))
    wall = time.perf_counter() - t0
    return {
        "steps": steps, "wall_s": wall, "s_per_macro_step": wall / steps,
        "macro_steps_per_s": steps / wall,
        "assignment": dict(r.assignment),
        "ledger": {m: sum(1 for x in r.assignment.values() if x == m)
                   for m in MODES},
        "mode_calls": dict(r.mode_calls),
        "trace": {k: np.asarray(v_) for k, v_ in trace.items()},
        "u": u.detach().cpu().numpy().copy(),
        "v": v.detach().cpu().numpy().copy(),
        "outside_the_envelope": r.outside_first,
        "outside_steps": r.outside_steps,
        "enforce": r.enforce,
    }


def switch_report(u0, v0, assignments: Sequence[Any], stack: WindowStack = None,
                  steps: int = 40, tiling=None, **kw) -> dict[str, Any]:
    """Section 5.3: every assignment against the all-classical march.

    The all-classical column is run FIRST and every other column is scored
    against it -- which is section 5.4's first clause, *"there IS a referent for
    the composed answer"*.  What it does not buy is section 5.4's second: the
    composed classical answer carries its own composition defect against a
    single-domain monolith, and that second referent is not run here.
    """
    base = march_mixed(u0, v0, steps=steps, assignment=Mode.CLASSICAL.value,
                       stack=stack, tiling=tiling, **kw)
    rows = {"classical": {**{k: v_ for k, v_ in base.items()
                             if k not in ("u", "v", "trace")},
                          "rms_field_error_vs_all_classical": 0.0,
                          "speed_ratio_vs_all_classical": 1.0,
                          "settled": {k: float(np.mean(v_[-max(1, steps // 4):]))
                                      for k, v_ in base["trace"].items()}}}
    for spec in assignments:
        m = march_mixed(u0, v0, steps=steps, assignment=spec, stack=stack,
                        tiling=tiling, **kw)
        du = m["u"] - base["u"]
        dv = m["v"] - base["v"]
        den = np.sqrt(np.mean((base["u"] - GE.U_INF) ** 2 + base["v"] ** 2))
        rms = float(np.sqrt(np.mean(du ** 2 + dv ** 2)) / max(den, 1e-30))
        tag = spec if isinstance(spec, str) else "custom"
        rows[tag] = {
            **{k: v_ for k, v_ in m.items() if k not in ("u", "v", "trace")},
            "rms_field_error_vs_all_classical": rms,
            "speed_ratio_vs_all_classical":
                base["s_per_macro_step"] / m["s_per_macro_step"],
            "settled": {k: float(np.mean(v_[-max(1, steps // 4):]))
                        for k, v_ in m["trace"].items()},
        }
    return {
        "steps": steps, "columns": rows,
        "referent": "the all-classical composed march at the same horizon",
        "not_run": "the single-domain monolith -- section 5.4's SECOND "
                   "referent, against which the composed classical answer "
                   "carries its own composition defect. CS-19 did not run it "
                   "either and neither tier claims it",
    }
