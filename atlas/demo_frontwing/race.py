"""Beat 1 -- the gradient and the population, raced on one clock.

Two searches, the same objective, the same box, the same penalty, the same
start, running side by side and reporting against **one wall clock**:

  * **the gradient column** -- Adam on the reverse-mode adjoint of the whole
    composed rollout. One backward pass through both seams returns the
    derivative of settled downforce with respect to all four design knobs at
    once;
  * **the population column** -- CMA-ES on exactly the same objective, which is
    what an engineer without gradients actually does today.

Why this beat is here, and the honesty it is carrying
-----------------------------------------------------

Differentiability, not speed, is the enabling property of the whole approach
(`f1-pathmap-and-end-goal` section 1.2).  So the comparison worth showing is not
"is the composed model fast" but "does having a gradient change the search", and
that has a specific number attached which **this project has been quoting too
favourably, twice over** (W143):

  1. **the denominator was CMA-ES's whole budget** rather than the evaluation
     count at which it actually peaked -- 200 against 191, and on PoC 1 the gap
     was wider;
  2. **nobody divided by the adjoint premium.** A gradient iterate costs
     more than a forward evaluation, because the tape has to be replayed. On
     this graph, measured, it is **4.9 forward evaluations per gradient
     iterate**, and a comparison in *rollouts* silently prices that at one.

Correct both and the win is **1.8x to 4.3x in wall-clock**, not the 18x-22x in
rollouts that a naive count reports.  That is a real advantage and a much
smaller one, and this panel quotes the wall-clock figure first, in large type,
with the rollout count beside it in small type and labelled.

What the live race is, and is not
---------------------------------

The live race is **short**: a handful of gradient steps against a matched
evaluation budget, so it finishes while somebody is watching.  It is not the
measurement on the results page, which is 30 Adam steps against 200 CMA-ES
evaluations on a 120-macro-step objective from a settled field, and takes about
half an hour.  Both are on screen, the recorded one is labelled recorded, and
the live one never overwrites it.

The quality result is the one to read: on the full-scale run the two columns
land within **1.003x to 1.008x of each other in the objective**.  The gradient
does not find a better design.  It finds an equally good one sooner.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from ..cases import front_wing as F

#: The Adam settings the driver uses, so a live race is the same estimator at a
#: shorter horizon rather than a different one.
LR, B1, B2, EPS = 0.06, 0.9, 0.999, 1e-8

#: What one declined evaluation is worth to a search. Large and negative, so a
#: design outside an expert's envelope can never win, and finite, so the search
#: carries on -- a population sampler WILL propose designs outside an envelope
#: and losing the run to one of them is losing the run to a correct answer.
DECLINED_L = -1.0e3


def _span() -> np.ndarray:
    return np.array([F.DESIGN_BOX[k][1] - F.DESIGN_BOX[k][0]
                     for k in F.DESIGN_KEYS])


def to_unit(d: dict) -> np.ndarray:
    return np.array([(d[k] - F.DESIGN_BOX[k][0])
                     / (F.DESIGN_BOX[k][1] - F.DESIGN_BOX[k][0])
                     for k in F.DESIGN_KEYS])


def from_unit(x) -> dict:
    return {k: float(F.DESIGN_BOX[k][0]
                     + np.clip(x[i], 0.0, 1.0)
                     * (F.DESIGN_BOX[k][1] - F.DESIGN_BOX[k][0]))
            for i, k in enumerate(F.DESIGN_KEYS)}


def evaluate(design: dict, steps: int, u, v, want_grad: bool,
             weight: float = F.PENALTY) -> dict:
    """One objective evaluation, with or without the adjoint.

    Deliberately the same shape as `scripts/w141_poc2_frontwing._evaluate`, and
    a test asserts the two agree on one design to round-off -- a demo whose
    objective differs from the driver's is racing a different problem from the
    one the results page reports.

    **Driven by the smooth margins, judged by the true ones.** `logsumexp`
    overshoots the maximum it smooths, so a design the penalty calls infeasible
    may be feasible in fact; the gradient needs the smooth pair because a hard
    max has a chattering subgradient, and `feasible` needs the true pair because
    that is the actual question.
    """
    import torch

    ro = F.FrontWingRollout(tiling=F.SINGLE_TILING, coupling="tight",
                            motion=True, design=design)
    opt = dict(dtype=F.TORCH_DTYPE)
    keys = F.DESIGN_KEYS if want_grad else ()
    leaves = {k: torch.tensor(float(design[k]), requires_grad=True, **opt)
              for k in keys}
    try:
        J, gs, gd, ts, td = ro.objective(
            dict(design, **leaves) if leaves else design,
            steps=steps, u0=u, v0=v, grad=bool(keys))
    except RuntimeError as exc:
        msg = str(exc)
        why = ("envelope" if "envelope" in msg
               else "blowup" if "finite" in msg else "other")
        return dict(J=0.0, L=DECLINED_L, grad=None, feasible=False,
                    declined=why, error=msg[:200], true_sigma=1.0,
                    true_delta=1.0)
    L = F.penalised(J, gs, gd, weight)
    grad = None
    if leaves:
        gg = torch.autograd.grad(L, list(leaves.values()), allow_unused=True)
        grad = np.array([0.0 if x is None else float(x) for x in gg])
    return dict(J=float(J), L=float(L), grad=grad,
                true_sigma=float(ts), true_delta=float(td),
                feasible=bool(float(ts) <= 0.0 and float(td) <= 0.0),
                declined=None)


def _better(best, cand):
    if not cand.get("feasible"):
        return best
    if best is None or cand["J"] > best["J"]:
        return dict(cand)
    return best


def race(u, v, steps: int = 24, n_grad: int = 8, n_pop: int = 40,
         weight: float = F.PENALTY, seed: int = 0,
         start: dict | None = None,
         emit: Callable[[dict], None] | None = None,
         should_stop: Callable[[], bool] | None = None) -> dict:
    """Run both columns and report them against one clock.

    `emit` is called after every evaluation in either column with the running
    state, so a screen can animate the race rather than show its result.  The
    two columns run **in turn, one evaluation each**, so neither has the machine
    to itself while the other waits -- which is the same discipline the PoC 1a
    demo's measurement panel uses, and the reason the wall-clock ratio here
    means anything at all.
    """
    start = dict(start or F.DESIGN_REF)
    span = _span()

    # -- the gradient column, as a coroutine of one Adam step per call --------
    gx = to_unit(start)
    gm, gv2, gt = np.zeros_like(gx), np.zeros_like(gx), 0
    g_hist: list[dict] = []
    g_best = None
    g_wall = 0.0

    def grad_step():
        nonlocal gx, gm, gv2, gt, g_best, g_wall
        t0 = time.perf_counter()
        d = from_unit(gx)
        e = evaluate(d, steps, u, v, want_grad=True, weight=weight)
        g_wall += time.perf_counter() - t0
        gt += 1
        rec = dict(t=gt, design=d, J=e["J"], L=e["L"],
                   feasible=e["feasible"], declined=e.get("declined"),
                   wall_s=g_wall)
        g_hist.append(rec)
        g_best = _better(g_best, dict(rec, **{"J": e["J"]}))
        if e["grad"] is not None:
            g = e["grad"] * span
            gm[:] = B1 * gm + (1 - B1) * g
            gv2[:] = B2 * gv2 + (1 - B2) * g * g
            mh = gm / (1 - B1 ** gt)
            vh = gv2 / (1 - B2 ** gt)
            gx = np.clip(gx + LR * mh / (np.sqrt(vh) + EPS), 0.0, 1.0)
        else:
            #: a declined iterate has no gradient, so back off toward the last
            #: answered point rather than stepping blind
            gx = np.clip(0.5 * (gx + to_unit(g_hist[0]["design"])), 0.0, 1.0)
        return rec

    # -- the population column ------------------------------------------------
    p_hist: list[dict] = []
    p_best = None
    p_wall = 0.0
    try:
        import cma
        es = cma.CMAEvolutionStrategy(
            list(to_unit(start)), 0.30,
            {"bounds": [0.0, 1.0], "seed": seed + 1, "verbose": -9,
             "maxfevals": n_pop})
        optimiser = "cma-es"
    except Exception:                                       # pragma: no cover
        es, optimiser = None, "isotropic (mu,lambda) ES -- cma not installed"
        rng = np.random.default_rng(seed + 1)
        pm, psig = to_unit(start), 0.30

    #: `es.ask()` is stateful and `es.tell` needs that same list back, so the
    #: generation is cached rather than regenerated. One list, one turn.
    pending: list = []
    pending_vals: list[float] = []
    ask_cache: list = []

    def pop_step():
        nonlocal p_best, p_wall, pending, pending_vals, ask_cache, pm, psig
        if not pending:
            if es is not None:
                ask_cache = list(es.ask())
            else:                                            # pragma: no cover
                ask_cache = [
                    np.clip(pm + psig * rng.standard_normal(len(pm)), 0.0, 1.0)
                    for _ in range(8)]
            pending = list(ask_cache)
            pending_vals = []
        xi = pending.pop(0)
        t0 = time.perf_counter()
        d = from_unit(np.asarray(xi))
        e = evaluate(d, steps, u, v, want_grad=False, weight=weight)
        p_wall += time.perf_counter() - t0
        pending_vals.append(-e["L"])          # CMA-ES minimises
        rec = dict(n=len(p_hist) + 1, design=d, J=e["J"], L=e["L"],
                   feasible=e["feasible"], declined=e.get("declined"),
                   wall_s=p_wall)
        p_hist.append(rec)
        if e["feasible"] and (p_best is None or e["J"] > p_best["J"]):
            p_best = dict(rec)
        if not pending:
            if es is not None:
                es.tell(ask_cache, pending_vals)
            else:                                            # pragma: no cover
                keep = max(1, len(pending_vals) // 2)
                idx = np.argsort(pending_vals)[:keep]
                pm = np.mean([ask_cache[i] for i in idx], axis=0)
                psig *= 0.92
        return rec

    # -- warm BOTH columns before either clock starts -------------------------
    #
    #: **Not optional, and it is the whole reason this measurement is worth
    #: anything.**  The first call in each column pays for a torch dispatch
    #: cache, the structural operator's factorisation and the tape's first
    #: allocation.  Unwarmed, the same three-step race that reads a 5.1x adjoint
    #: premium when warm reads **25x**, because the gradient column ran first and
    #: absorbed the whole cost for both.  `stage_cost` made exactly this mistake
    #: and published a 9.47 that a warmed re-measurement put at 5.4-5.8.
    t_warm = time.perf_counter()
    evaluate(start, steps, u, v, want_grad=False, weight=weight)
    evaluate(start, steps, u, v, want_grad=True, weight=weight)
    warm_s = time.perf_counter() - t_warm

    # -- one evaluation each, in turn, until both budgets are spent -----------
    t_all = time.perf_counter()
    while len(g_hist) < n_grad or len(p_hist) < n_pop:
        if should_stop is not None and should_stop():
            break
        if len(g_hist) < n_grad:
            grad_step()
            if emit:
                emit(_snapshot(g_hist, p_hist, g_best, p_best, g_wall, p_wall,
                               n_grad, n_pop, optimiser, steps, start))
        if should_stop is not None and should_stop():
            break
        if len(p_hist) < n_pop:
            pop_step()
            if emit:
                emit(_snapshot(g_hist, p_hist, g_best, p_best, g_wall, p_wall,
                               n_grad, n_pop, optimiser, steps, start))
    snap = _snapshot(g_hist, p_hist, g_best, p_best, g_wall, p_wall,
                     n_grad, n_pop, optimiser, steps, start)
    snap["wall_s"] = time.perf_counter() - t_all
    snap["warm_s"] = warm_s
    snap["done"] = True
    return snap


def _snapshot(g_hist, p_hist, g_best, p_best, g_wall, p_wall,
              n_grad, n_pop, optimiser, steps, start) -> dict:
    """The comparison, derived the way W143 says it must be.

    Three ratios, and the order they are quoted in is the finding:

      * `ratio_wall` -- wall-clock to the gradient column's best, against
        wall-clock to the population column's best. **This is the headline.**
      * `ratio_rollouts` -- evaluations, which prices a gradient iterate at one
        forward evaluation and is therefore flattering. Quoted small and
        labelled.
      * `quality_ratio` -- what the two columns actually found. Near 1 is the
        honest result: the gradient does not find a better design.
    """
    def _first_at_least(hist, target):
        for i, r in enumerate(hist, 1):
            if r.get("feasible") and r["J"] >= target:
                return i, r["wall_s"]
        return None, None

    gJ = g_best["J"] if g_best else None
    pJ = p_best["J"] if p_best else None
    out = dict(
        gradient=dict(history=g_hist[-200:], best=g_best, wall_s=g_wall,
                      n=len(g_hist), budget=n_grad,
                      declined=sum(1 for r in g_hist if r.get("declined"))),
        population=dict(history=p_hist[-400:], best=p_best, wall_s=p_wall,
                        n=len(p_hist), budget=n_pop, optimiser=optimiser,
                        declined=sum(1 for r in p_hist if r.get("declined"))),
        steps=steps, start=dict(start), done=False)
    if gJ is None or pJ is None:
        return out
    target = min(gJ, pJ)
    gn, gw = _first_at_least(g_hist, target)
    pn, pw = _first_at_least(p_hist, target)
    out["comparison"] = dict(
        target_J=target,
        gradient_evals_to_target=gn, population_evals_to_target=pn,
        gradient_wall_to_target=gw, population_wall_to_target=pw,
        ratio_rollouts=(pn / gn) if (gn and pn) else None,
        ratio_wall=(pw / gw) if (gw and pw and gw > 0) else None,
        quality_ratio=(gJ / pJ) if pJ else None,
        best_gradient_J=gJ, best_population_J=pJ,
    )
    return out
