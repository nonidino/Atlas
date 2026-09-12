"""Defect correction with a cheap operator: a slot where a learned expert changes
the RATE of a classical computation and never the ANSWER.

Tier 48, [[defect-correction-learned-operator]].  Every substitution this vault has
tried asked a learned expert to BE part of the answer -- an agent at a seam, a
starting guess -- and so owed it a certificate it could not be issued: a checkpoint
fixed at one resolution has no same-class reference pair, hence no tau, no sigma,
no eps_tol and no beta_min (W95).  This module asks something else of it.

The iteration
-------------

The classical composed macro-step ``phi`` has a settled state ``w* = phi(w*)``.  A
cheap map ``psi`` -- the learned composed column -- is used only to solve for the
next iterate:

    d_k      = phi(w_k) - psi(w_k)                   (one classical call)
    w_{k+1}  = the fixed point of  x -> psi(x) + d_k  (cheap calls only)

which is Stetter's defect correction (Numer. Math. 29, 1978; Boehmer, Hemker and
Stetter, Computing Suppl. 5, 1984) with ``I - psi`` as the approximate operator and
``I - phi`` as the target.

Three facts, each with a test on an exact linear problem in
``tests/test_tier48_defect_correction.py``:

1. **Consistency, whatever psi is.**  If the iterates converge, the limit satisfies
   ``phi(w) = w``: pass to the limit in
   ``(I - psi)(w_{k+1}) - (I - psi)(w_k) = -(I - phi)(w_k)``.  Nothing about the
   cheap map's accuracy enters.  A wrong psi can cost classical calls; it cannot
   change what is returned.
2. **The null element is the classical march, exactly.**  A constant psi makes the
   inner problem's solution ``phi(w_k)`` after one cheap call, so the iterates ARE
   the classical march, bit for bit.  W76's null replacement therefore has an
   algebraic meaning here and is measured in the same currency as everything else.
3. **The rate is the cheap map's Jacobian fidelity.**  Linearised at ``w*``,
   ``e_{k+1} = (I - J_psi^{-1} J_phi) e_k`` with ``J = I - D(map)``.  The cheap map
   moves the iteration operator, not only the starting distance, which is why
   CS-S1's one-sweep ceiling on a predictor does not bind in this slot.

What is certified
-----------------

The returned state ``w`` carries the classical residual ``r = ||phi(w) - w||``,
computed, never estimated.  If ``phi`` contracts with constant ``L < 1`` on a ball
holding ``w`` and ``w*`` (Banach), ``||w - w*|| <= r / (1 - L)``; linearised,
``||w - w*|| <= Theta r`` with ``Theta = ||(I - D phi)^{-1}||``.  ``Theta`` is a
property of the CLASSICAL map in its regime.  It is measured, not assumed, by
`theta_from_march`, and the bound is reported beside the measured error.

**Its hypothesis that is easy to break**: the fixed point must be isolated.  A
map that carries boundary data as state has a family of fixed points and
``Theta`` is infinite along it -- measured on `wake_array`'s monolith, whose
outflow ring is held at its input value (``scripts/w202_kill_tests.py``).

When it fails loudly
--------------------

The residual is monitored every outer iteration.  If it does not FALL -- ratio to
the previous one above ``stall_ratio``, which is one by default -- for
``stall_patience`` consecutive iterations, or an inner march blows up, the
iteration stops and the classical march resumes from the best iterate.  The answer
is then the classical march's, and the result says so in its ``status``; the cost
of the detour is in its call counts.

**The rule detects failure, not slowness, on purpose.**  A threshold below one
misfires on the null element itself: a constant cheap map makes every outer
iteration one classical step, and a classical march contracting at 0.9 per step
-- `wake_array`'s monolith reads 0.894 on its tail -- would be declared stalled
while doing exactly what it is supposed to.  A slow cheap map that still converges
is not a failure; what it costs is reported by `break_even`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

State = np.ndarray
Map = Callable[[State], State]


def vector_rms(w: np.ndarray) -> float:
    """rms of a stacked field ``(n_components, ...)``: sqrt of the per-cell mean of
    the sum of squared components.  For a velocity ``(u, v)`` this is the rms of
    the vector magnitude."""
    w = np.asarray(w, dtype=float)
    if w.ndim < 2:
        return float(np.sqrt(np.mean(w * w)))
    return float(np.sqrt(np.sum(np.mean(w * w, axis=tuple(range(1, w.ndim))))))


@dataclass
class DefectCorrectionResult:
    """The returned state, its certificate input, and what it cost."""

    state: np.ndarray
    status: str
    residual: float
    phi_calls: int
    psi_calls: int
    rows: list[dict[str, Any]] = field(default_factory=list)
    fallback_from: int | None = None
    fallback_rows: list[dict[str, Any]] = field(default_factory=list)
    #: The iterate at which the corrected iteration stopped, before any fallback:
    #: the state a failure's MECHANISM is read from.
    pre_fallback_state: np.ndarray | None = None
    #: Why the corrected iteration stopped -- ``converged``, ``stalled``,
    #: ``diverged``, ``diverged_inner`` or ``max_outer`` -- kept because a fallback
    #: overwrites ``status``.
    stop_reason: str | None = None

    @property
    def converged(self) -> bool:
        return self.status in ("converged", "fallback_converged")

    def cost(self, c_psi: float) -> float:
        """Total cost in classical calls: ``phi_calls + c_psi * psi_calls``."""
        return float(self.phi_calls + c_psi * self.psi_calls)

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "stop_reason": self.stop_reason,
                "residual": self.residual,
                "phi_calls": self.phi_calls, "psi_calls": self.psi_calls,
                "fallback_from": self.fallback_from, "rows": self.rows,
                "fallback_rows": self.fallback_rows}


def defect_correct(phi: Map, psi: Map, w0: State, *, r_stop: float,
                   k_max: int = 20, m_max: int = 40, inner_frac: float = 0.1,
                   stall_ratio: float = 1.0, stall_patience: int = 2,
                   fallback: bool = True, fallback_steps: int = 5000,
                   divergence_bound: float | None = None,
                   norm: Callable[[np.ndarray], float] = vector_rms,
                   reference: State | None = None,
                   on_row: Callable[[dict[str, Any]], None] | None = None,
                   ) -> DefectCorrectionResult:
    """Find ``w`` with ``||phi(w) - w|| <= r_stop`` by defect correction with ``psi``.

    ``reference``, when given, is used only to REPORT the error of each iterate;
    nothing in the iteration reads it.  ``inner_frac`` ends an inner march when its
    step falls below that fraction of the current classical residual, so the inner
    solve tightens as the outer one converges.
    """
    w = np.array(w0, dtype=float, copy=True)
    rows: list[dict[str, Any]] = []
    phi_calls = psi_calls = 0
    status = "max_outer"
    best_res, best_w = float("inf"), w.copy()
    prev_res: float | None = None
    stall = 0
    k = 0
    for k in range(k_max):
        fw = np.asarray(phi(w), dtype=float)
        phi_calls += 1
        res = norm(fw - w)
        row = {"k": k, "residual": res, "phi_calls": phi_calls, "psi_calls": psi_calls}
        if reference is not None:
            row["error"] = norm(w - reference)
        rows.append(row)
        if on_row is not None:
            on_row(row)
        if not np.isfinite(res) or (divergence_bound is not None and res > divergence_bound):
            status = "diverged"
            break
        if res < best_res:
            best_res, best_w = res, w.copy()
        if res <= r_stop:
            status = "converged"
            done = DefectCorrectionResult(w, status, res, phi_calls, psi_calls, rows)
            done.stop_reason = status
            return done
        if prev_res is not None and res > stall_ratio * prev_res:
            stall += 1
        else:
            stall = 0
        prev_res = res
        if stall >= stall_patience:
            status = "stalled"
            break
        pw = np.asarray(psi(w), dtype=float)
        psi_calls += 1
        x = fw
        inner = 0
        inner_ok = True
        for _m in range(m_max):
            # x <- phi(w_k) + (psi(x) - psi(w_k)).  The same iteration as
            # psi(x) + d_k with d_k = phi(w_k) - psi(w_k), written so the cheap
            # map enters only through a DIFFERENCE: a constant psi then returns
            # phi(w_k) exactly, and a bias psi carries cancels before it is added.
            xn = fw + (np.asarray(psi(x), dtype=float) - pw)
            psi_calls += 1
            inner += 1
            if not np.all(np.isfinite(xn)) or (
                    divergence_bound is not None and norm(xn) > divergence_bound):
                inner_ok = False
                break
            step = norm(xn - x)
            x = xn
            if step <= inner_frac * res:
                break
        rows[-1]["inner_steps"] = inner
        if not inner_ok:
            status = "diverged_inner"
            break
        w = x

    result = DefectCorrectionResult(w, status, rows[-1]["residual"] if rows else float("nan"),
                                    phi_calls, psi_calls, rows)
    result.pre_fallback_state = w.copy()
    result.stop_reason = status
    if not fallback:
        return result
    # The classical march from the best iterate.  The answer is the march's, and it
    # is certified by the same residual.
    result.fallback_from = k
    w = best_w
    for j in range(fallback_steps):
        fw = np.asarray(phi(w), dtype=float)
        result.phi_calls += 1
        res = norm(fw - w)
        if j % 25 == 0 or res <= r_stop:
            frow = {"j": j, "phase": "fallback", "residual": res,
                    "phi_calls": result.phi_calls, "psi_calls": result.psi_calls}
            if reference is not None:
                frow["error"] = norm(w - reference)
            result.fallback_rows.append(frow)
            # a fallback at a large rung is minutes of classical marching, and an
            # instrument that says nothing until it finishes reads as a stall
            if on_row is not None:
                on_row(frow)
        if res <= r_stop:
            result.state, result.residual, result.status = w, res, "fallback_converged"
            return result
        if not np.isfinite(res):
            break
        w = fw
    result.state, result.residual = w, res
    result.status = "fallback_unconverged"
    return result


def shrink(psi: Map, alpha: float) -> Map:
    """The cheap map pulled toward the null element: ``(1 - alpha) psi``.

    The iteration reads the cheap map only through differences, so ``alpha = 1`` is
    a constant map and reproduces the classical march exactly, and ``alpha = 0`` is
    the cheap map itself.  Linearised, ``J = alpha I + (1 - alpha) J_psi``, whose
    eigenvalues cannot fall below ``alpha``: a cheap map that is closer to singular
    than the classical one on some mode -- the way a column that conserves what the
    classical map dissipates is -- is regularised on exactly that mode.  The
    contribution a learned map makes is then measured against ``alpha = 1`` at the
    same ``alpha`` it needed.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
    scale = 1.0 - float(alpha)
    return lambda w: scale * np.asarray(psi(w), dtype=float)


def classical_march(phi: Map, w0: State, *, r_stop: float, max_steps: int = 5000,
                    norm: Callable[[np.ndarray], float] = vector_rms,
                    reference: State | None = None) -> DefectCorrectionResult:
    """The null element as a function: the classical march, in the same currency."""
    w = np.array(w0, dtype=float, copy=True)
    rows = []
    for j in range(max_steps):
        fw = np.asarray(phi(w), dtype=float)
        res = norm(fw - w)
        row = {"k": j, "residual": res, "phi_calls": j + 1, "psi_calls": 0}
        if reference is not None:
            row["error"] = norm(w - reference)
        rows.append(row)
        if res <= r_stop:
            return DefectCorrectionResult(w, "converged", res, j + 1, 0, rows)
        w = fw
    return DefectCorrectionResult(w, "max_outer", res, max_steps, 0, rows)


def theta_from_march(errors: Sequence[float], residuals: Sequence[float],
                     tail: tuple[int, int] | None = None) -> dict[str, float]:
    """``Theta`` read off a converged classical march: ``error / residual``.

    ``errors[j]`` is the distance of the j-th iterate to the settled state and
    ``residuals[j]`` its one-step residual.  The default window is the MIDDLE HALF
    of the march, which leaves out the transient at one end and the reference's own
    round-off floor at the other; ``tail`` overrides it.  Returned as the largest
    ratio (the constant a certificate should use) and the smallest (how much the
    reading moves).  Iterates whose error is at the reference's floor are excluded
    as well, because a ratio of two round-off numbers is not a stability constant.

    **This is an estimate along ONE approach to the settled state, and it has been
    measured under-bounding others.**  The constant a certificate needs is an
    operator norm -- a supremum over directions -- and a march samples the single
    direction it took.  Tier 48 (W208) checked the estimate against every arm of a
    defect-correction sweep on three graph sizes: it held at two windows (worst arm
    needed 3.45 against a reported 3.55) and at six (3.34 against 3.67), and FAILED
    at twelve, where the worst arm needed 3.81 against a reported 3.35 -- the
    estimate falling with size while the requirement rose.  Report the ratio each
    state actually needs beside the bound it was given, and do not treat a pass as
    proof that the constant is valid off the march.
    """
    e = np.asarray(errors, dtype=float)
    r = np.asarray(residuals, dtype=float)
    n = min(len(e), len(r))
    lo, hi = tail if tail is not None else (n // 4, (3 * n) // 4)
    sel = [(e[j] / r[j]) for j in range(lo, min(hi, n)) if r[j] > 0.0 and e[j] > 1e-12]
    if not sel:
        return {"theta": float("nan"), "theta_min": float("nan"), "n": 0}
    return {"theta": float(max(sel)), "theta_min": float(min(sel)), "n": len(sel)}


def certificate(residual: float, theta: float) -> float:
    """``Theta * ||phi(w) - w||``: the bound on the distance to the settled state.

    Only as good as ``theta``; see ``theta_from_march`` for the direction in which
    an estimated one was measured to fail (W208).
    """
    return float(theta * residual)


def break_even(cold_phi_calls: int, result: DefectCorrectionResult,
               c_psi: float) -> dict[str, float]:
    """The contribution, as a ratio: cold classical calls over the corrected cost.

    Above one, the cheap map paid for itself at the SAME certified residual.  The
    per-call cost ratio ``c_psi`` must be measured in the same process as the
    classical call it is a ratio to.
    """
    cost = result.cost(c_psi)
    return {"cold_cost": float(cold_phi_calls), "corrected_cost": cost,
            "saving_ratio": float(cold_phi_calls) / cost if cost > 0 else float("inf"),
            "phi_calls": result.phi_calls, "psi_calls": result.psi_calls,
            "c_psi": float(c_psi)}


def exchange_rate(q_phi: float, q_psi: float, c_psi: float, inner_mean: float) -> float:
    """The entry condition as one number, for geometric convergence on both sides.

    A cold march contracting at ``q_phi`` per classical call needs
    ``ln(r0/r)/ln(1/q_phi)`` calls; defect correction contracting at ``q_psi`` per
    outer iteration pays ``1 + c_psi (1 + inner_mean)`` classical-equivalents per
    iteration.  Returns cold cost over corrected cost for the same reduction:
    ``ln(1/q_psi) / (ln(1/q_phi) (1 + c_psi (1 + inner_mean)))``.  Above one the
    cheap map pays.  Transit-limited convergence is not geometric, and the
    measured ``break_even`` is the number to quote; this is the reading of it.
    """
    if not (0.0 < q_phi < 1.0) or not (0.0 < q_psi < 1.0):
        return float("nan")
    return float(np.log(1.0 / q_psi)
                 / (np.log(1.0 / q_phi) * (1.0 + c_psi * (1.0 + inner_mean))))


__all__ = ["DefectCorrectionResult", "break_even", "certificate", "classical_march",
           "defect_correct", "exchange_rate", "shrink", "theta_from_march", "vector_rms"]
