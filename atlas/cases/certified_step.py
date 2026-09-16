"""The certified mode on the body-fitted column: the fixed point an implicit step
has and an explicit one does not.

PoC 3, Tier 72.  [[poc3-racelab-certified-step]].

W226, and what the user decided
-------------------------------

Defect correction certifies a FIXED POINT: `atlas.defect_correction` finds ``w``
with ``phi(w) = w``.  The porous column's `WindowNS` step is EXPLICIT -- one call
to ``phi`` already IS the answer -- so there is nothing for an iteration to do
inside a macro-step, and `racelab_switch.MixedRollout` raises rather than quietly
running something else.  That is W226, and on 2026-09-14 the user chose its second
option: **an implicit fluid time step**, not an amendment of section 4.2 to a
steady-state mode.

**That step already exists.**  `overset_ns.OversetFlow.step` is implicit in
viscosity AND in advection (linearised about ``u* = 2u^k - u^(k-1)``), with BDF2 in
time, and it says so in its own docstring.  Its momentum system

    M x = b,        M = K_fixed + K_visc + (a0/dt) P_disc + u* Dx + v* Dy

is solved ITERATIVELY, by BiCGSTAB to ``rtol``.  Its solution ``x* = M^{-1} b`` is
the fixed point of the preconditioned sweep

    phi_P(x) = x + P (b - M x)

for any non-singular ``P``: ``phi_P(x) = x``  iff  ``M x = b``.  So the certified
mode's ``phi`` is one sweep, the state it certifies is the momentum solve's own
answer, and the limit is the CLASSICAL answer whatever the cheap map does
(Theorem 1, `defect_correction`'s consistency).

**The rest of the step is direct.**  The projection is one factored solve
(``p_lu``), the interpolation one sparse apply; neither iterates.  So certifying
the momentum solve certifies the step, which is what `certified_step` measures
rather than assumes.

Which ``P``, and why it is not a free choice
--------------------------------------------

``picard`` -- ``P = (dt/a0) I`` -- is the tempting one, because ``phi`` is then
literally "evaluate the implicit right-hand side explicitly", a time-advance map
of the same shape as a learned expert's.  **It diverges here, and the reason is
the geometry the body-fitted column exists for.**  A body grid clusters its rows
toward the wall at a spacing of about 0.0014, and

    (dt/a0) nu / h^2 = 0.008333 * 0.004 / 0.0014^2 = 17.0

before the Laplacian's own stencil factor.  A sweep that amplifies by 17 has a
fixed point it can never reach.  ``ilu`` -- the step's own incomplete factor --
is the sweep that contracts, and `contraction` measures its rate rather than
assuming it.

**Measuring a rate at the tail measures the round-off floor, not the rate.**
`contraction` therefore reports the ratio over a stated early window and the
floor it stopped at, and refuses to call a flat tail a rate.  (This module's
first probe made exactly that mistake and read 0.9987 for a sweep contracting at
about 0.3.)

What is reported, and what is only reported
-------------------------------------------

Section 5.2 asks the certified mode for **outer iterations, inner cheap calls and
the residual**.  All three come out of `DefectCorrectionResult`.  Two more are
carried because a viewer of section 12's criterion 4 is watching *the error go to
the classical answer*:

- ``algebraic_residual`` -- ``||b - M w||``, the residual of the system itself,
  which is preconditioner-free and comparable between sweeps.  The iteration's own
  ``residual`` is ``||P(b - M w)||``, which is a different number in a different
  currency, and conflating the two would make one sweep look better than another
  for no reason but its preconditioner.
- ``error_to_classical`` -- ``||w - x*||`` against BiCGSTAB's answer.  It is
  passed to `defect_correct` as ``reference``, which REPORTS it and never reads
  it; nothing in the iteration may consult the answer it claims to reach.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import scipy.sparse.linalg as spla

from .. import defect_correction as DC

__all__ = ["SWEEPS", "MomentumSystem", "contraction", "certified_momentum",
           "CertifiedMomentumSolver", "null_psi", "jacobi_psi", "detuned_psi",
           "wrong_psi", "certified_step_report"]

#: The candidate ``P`` in ``phi_P(x) = x + P (b - M x)``.  ``picard`` is kept
#: BECAUSE it diverges: a module that offered only the sweep that works would
#: hide the reason the working one is needed.
SWEEPS = ("picard", "jacobi", "ilu")


def _stack(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The two components as one ``(2, n)`` state, which is what
    `defect_correction.vector_rms` reads as a vector field."""
    return np.stack([np.asarray(u, dtype=float), np.asarray(v, dtype=float)])


class MomentumSystem:
    """One step's momentum system, and the maps whose fixed point is its answer.

    Built from ``flow._assemble_momentum()`` -- the SAME assembly `step` uses, not
    a replica.  A replica would be a second definition of the car, free to drift
    from the first, which this vault has already paid for twice.
    """

    def __init__(self, flow, *, sweep: str = "ilu", precond=None) -> None:
        if sweep not in SWEEPS:
            raise ValueError(f"sweep must be one of {SWEEPS}, got {sweep!r}")
        self.flow = flow
        self.sweep = sweep
        self.M, self.bu, self.bv, self.us, self.vs, self.a0, self.t1 = \
            flow._assemble_momentum()
        self.n = self.bu.size
        self._diag = self.M.diagonal()
        if np.any(self._diag == 0):                              # pragma: no cover
            raise AssertionError("a momentum row has a zero diagonal")
        self._apply_ilu = None
        self.ilu_s = 0.0
        self.ilu_note = None
        self.reused_precond = False
        if sweep == "ilu":
            # **The flow's own preconditioner, not a second one.**  Building a
            # fresh incomplete factor here was wrong twice over: SuperLU's factor
            # is EXACTLY SINGULAR on the car at the step's drop tolerance, and
            # `_momentum_precond` is where the retry and the Jacobi fallback for
            # that already live; and factoring afresh every call, where `step`
            # refreshes every ``ilu_every`` steps and reuses between, prices the
            # certified mode against a classical mode that is not paying the same
            # bill.  Same factor, same cadence, one definition.
            if precond is not None:
                self._apply_ilu = precond.matvec
                self.reused_precond = True
            else:
                saved = flow.precond
                flow.precond = "ilu"
                try:
                    p, t_ilu, note = flow._momentum_precond(self.M)
                finally:
                    flow.precond = saved
                self._apply_ilu = p.matvec
                self.ilu_s = t_ilu
                self.ilu_note = note
        self.phi_calls = 0

    # -- the preconditioner ---------------------------------------------------

    def apply_P(self, r: np.ndarray) -> np.ndarray:
        """``P r`` for this sweep."""
        if self.sweep == "picard":
            return (self.flow.dt / self.a0) * r
        if self.sweep == "jacobi":
            return r / self._diag
        return self._apply_ilu(r)

    # -- the maps -------------------------------------------------------------

    @property
    def w0(self) -> np.ndarray:
        """The starting state: the extrapolated advecting velocity, which is
        exactly what `_momentum_solve` hands BiCGSTAB as ``x0``.  The certified
        solve therefore starts where the classical one starts."""
        return _stack(self.us, self.vs)

    def phi(self, w: np.ndarray) -> np.ndarray:
        """One preconditioned sweep, both components.  Its fixed point is
        ``x* = M^{-1} b`` for ANY non-singular ``P``."""
        w = np.asarray(w, dtype=float)
        self.phi_calls += 1
        xu = w[0] + self.apply_P(self.bu - self.M @ w[0])
        xv = w[1] + self.apply_P(self.bv - self.M @ w[1])
        return _stack(xu, xv)

    def algebraic_residual(self, w: np.ndarray) -> float:
        """``||b - M w||`` over both components -- the residual of the SYSTEM,
        free of the preconditioner, and so comparable between sweeps."""
        w = np.asarray(w, dtype=float)
        ru = self.bu - self.M @ w[0]
        rv = self.bv - self.M @ w[1]
        return float(np.sqrt(np.sum(ru * ru) + np.sum(rv * rv)))

    def rhs_norm(self) -> float:
        return float(np.sqrt(np.sum(self.bu ** 2) + np.sum(self.bv ** 2)))

    def classical(self) -> tuple[np.ndarray, list[int], float]:
        """BiCGSTAB's answer ``x*``, its iteration counts and its wall time --
        through the flow's own `_momentum_solve`, so this is the classical mode's
        number and not a second implementation of it."""
        precond, _t_ilu, _note = self.flow._momentum_precond(self.M)
        t0 = time.perf_counter()
        ut, vt, its = self.flow._momentum_solve(self.M, self.bu, self.bv,
                                                self.us, self.vs, precond)
        return _stack(ut, vt), list(its), time.perf_counter() - t0


# ---------------------------------------------------------------------------
# is the fixed point REACHABLE?  -- a rate, measured where a rate exists
# ---------------------------------------------------------------------------


def contraction(sys: MomentumSystem, n: int = 30, *, window: int = 8,
                plateau_ratio: float = 0.95, plateau_patience: int = 2
                ) -> dict[str, Any]:
    """The sweep's contraction rate, read where a rate exists.

    **A ratio taken at the tail measures the round-off floor, not the rate.**
    Once the residual has stopped falling, consecutive ratios go to one and a
    contracting sweep reports itself as stalled -- this module's first probe read
    0.9987 for a sweep contracting at 0.0024.  But the converse trap is just as
    easy: taking the MINIMUM residual over ``n`` sweeps as "the floor" calls a
    sweep that is still falling at sweep ``n`` floored, which mislabelled Jacobi
    on the cylinder.  A floor is where the residual stops falling, so it is found
    from the RATIOS:

    - a sweep whose residual grew, or went non-finite, has diverged; the reported
      rate is then its growth factor, which is the number that matters about it;
    - otherwise the plateau is the first place ``plateau_patience`` consecutive
      ratios exceed ``plateau_ratio``, and the rate is the geometric mean over the
      first ``window`` sweeps before it.

    ``rate_is_floor`` is True only when that leaves no window at all.
    """
    w = sys.w0
    res = [sys.algebraic_residual(w)]
    for _ in range(n):
        w = sys.phi(w)
        r = sys.algebraic_residual(w)
        res.append(r)
        if not np.isfinite(r):
            break
    finite = [r for r in res if np.isfinite(r)]
    diverged = (not np.isfinite(res[-1])) or (len(finite) > 1 and finite[-1] > res[0])
    ratios = [res[i + 1] / res[i] if (np.isfinite(res[i + 1]) and res[i] > 0)
              else float("inf") for i in range(len(res) - 1)]

    if diverged:
        hi = min(window, len(finite) - 1)
        lo = 0
        rate = ((finite[hi] / finite[lo]) ** (1.0 / max(hi - lo, 1))
                if hi > lo and finite[lo] > 0 else float("inf"))
        plateau = hi
        rate_is_floor = False
    else:
        plateau = len(ratios)
        run = 0
        for i, q in enumerate(ratios):
            if q > plateau_ratio:
                run += 1
                if run >= plateau_patience:
                    plateau = i - plateau_patience + 1
                    break
            else:
                run = 0
        lo, hi = 0, min(window, max(plateau, 0))
        rate_is_floor = hi <= lo
        rate = ((res[hi] / res[lo]) ** (1.0 / max(hi - lo, 1))
                if not rate_is_floor and res[lo] > 0 else float("nan"))

    return {"sweep": sys.sweep, "residuals": [float(r) for r in res],
            "r0": float(res[0]), "r_end": float(res[-1]),
            "rate": float(rate), "rate_window": [int(lo), int(hi)],
            "rate_is_floor": bool(rate_is_floor),
            "plateau_at": int(plateau),
            "floor": float(min(finite)) if finite else float("nan"),
            "diverged": bool(diverged),
            "contracts": bool((not diverged) and np.isfinite(rate) and rate < 1.0)}


# ---------------------------------------------------------------------------
# the cheap maps
# ---------------------------------------------------------------------------


def null_psi() -> DC.Map:
    """W76's null replacement: a constant map.  `defect_correct` then returns
    ``phi(w_k)`` after one inner call, so the iterates ARE the classical sweep
    march, bit for bit.  This is the control every other cheap map is measured
    against, and it is the only one whose cost is known in advance."""
    return lambda w: np.zeros_like(np.asarray(w, dtype=float))


def jacobi_psi(sys: MomentumSystem) -> DC.Map:
    """A genuinely cheaper classical sweep: the same map with ``P = D^{-1}``.
    One divide a row against the incomplete factor's triangular solves."""
    def psi(w):
        w = np.asarray(w, dtype=float)
        xu = w[0] + (sys.bu - sys.M @ w[0]) / sys._diag
        xv = w[1] + (sys.bv - sys.M @ w[1]) / sys._diag
        return _stack(xu, xv)
    return psi


def detuned_psi(sys: MomentumSystem, drop_tol: float = 1e-2,
                fill: float = 2.0) -> DC.Map | None:
    """A cheaper incomplete factor -- the honest candidate for a cheap map that
    might actually pay: same shape as ``phi``, dropped harder.

    **Returns None when the factor cannot be built.**  SuperLU's incomplete
    factor can meet an exactly zero pivot after dropping -- it did on the
    cylinder at ``drop_tol = 1e-2``, and `_momentum_precond` already carries a
    retry for the same trap on the car.  A cheap map that does not exist is a
    fact for the record to carry, not a traceback in the middle of a march.
    """
    ilu = None
    for drop, fl in ((drop_tol, fill), (0.01 * drop_tol, 1.5 * fill)):
        try:
            ilu = spla.spilu(sys.M.tocsc(), drop_tol=drop, fill_factor=fl)
            break
        except RuntimeError:
            ilu = None
    if ilu is None:
        return None

    def psi(w):
        w = np.asarray(w, dtype=float)
        xu = w[0] + ilu.solve(sys.bu - sys.M @ w[0])
        xv = w[1] + ilu.solve(sys.bv - sys.M @ w[1])
        return _stack(xu, xv)
    return psi


def wrong_psi(seed: int = 0, scale: float = 0.3) -> DC.Map:
    """A cheap map that is WRONG on purpose -- the control for Theorem 1.

    Consistency does not depend on the cheap map's accuracy, so a deliberately
    wrong one must still land on the classical answer, and cost more to get
    there.  A cheap map that changed the answer would refute the one accuracy
    claim in this demo that rests on a proof rather than a measurement.
    """
    rng = np.random.default_rng(seed)
    cache: dict[int, np.ndarray] = {}

    def psi(w):
        w = np.asarray(w, dtype=float)
        if w.shape not in cache:
            cache[w.shape] = rng.standard_normal(w.shape)
        return scale * (w * cache[w.shape])
    return psi


# ---------------------------------------------------------------------------
# the certified solve
# ---------------------------------------------------------------------------


@dataclass
class CertifiedReport:
    """What section 5.2 asks the certified mode to put on a screen."""

    status: str
    outer_iterations: int
    inner_cheap_calls: int
    residual: float
    algebraic_residual: float
    error_to_classical: float
    classical_iterations: list[int]
    wall_s: float
    classical_wall_s: float
    sweep: str
    #: Which cheap map actually RAN, and why, when it is not the one asked for.
    #: A map that could not be built must not answer under its own name.
    psi_ran: str = "null"
    psi_requested: str | None = None
    psi_unavailable_reason: str | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    theta: float | None = None
    certificate: float | None = None

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["rows"] = list(self.rows)
        return d


def certified_momentum(sys: MomentumSystem, psi: DC.Map | None = None, *,
                       psi_name: str | None = None,
                       psi_requested: str | None = None,
                       psi_unavailable_reason: str | None = None,
                       r_stop: float | None = None, k_max: int = 40,
                       m_max: int = 40, alpha: float = 0.0,
                       reference: np.ndarray | None = None,
                       classical_its: list[int] | None = None,
                       classical_wall_s: float = float("nan"),
                       **dc_kw) -> tuple[np.ndarray, CertifiedReport]:
    """Solve this step's momentum system by defect correction, and report the cost.

    ``reference`` is BiCGSTAB's ``x*``.  It is handed to `defect_correct` only so
    each iterate's error can be REPORTED; nothing in the iteration reads it.
    ``r_stop`` defaults to the flow's own ``rtol`` scaled by ``||b||``, so the
    certified answer is asked for the same accuracy the classical one is.
    """
    psi_ran = "null" if psi is None else (psi_name or "cheap")
    if psi is None:
        psi = null_psi()
    if alpha:
        psi = DC.shrink(psi, alpha)
    if r_stop is None:
        r_stop = sys.flow.rtol * max(sys.rhs_norm(), 1.0) / np.sqrt(sys.n)
    t0 = time.perf_counter()
    res = DC.defect_correct(sys.phi, psi, sys.w0, r_stop=r_stop, k_max=k_max,
                            m_max=m_max, reference=reference, **dc_kw)
    wall = time.perf_counter() - t0
    w = res.state
    err = (float(DC.vector_rms(w - reference)) if reference is not None
           else float("nan"))
    rep = CertifiedReport(
        status=res.status,
        outer_iterations=res.phi_calls,
        inner_cheap_calls=res.psi_calls,
        residual=float(res.residual),
        algebraic_residual=sys.algebraic_residual(w),
        error_to_classical=err,
        classical_iterations=list(classical_its or []),
        wall_s=wall,
        classical_wall_s=float(classical_wall_s),
        sweep=sys.sweep,
        psi_ran=psi_ran,
        psi_requested=psi_requested,
        psi_unavailable_reason=psi_unavailable_reason,
        rows=list(res.rows),
    )
    return w, rep


class CertifiedMomentumSolver:
    """A momentum solve that can be installed on ``flow.momentum_solver``.

    `step` then runs unchanged in every other respect, so the certified step is
    the classical step with a different solve.  Each step's report is appended to
    `reports`, which is the per-step telemetry section 5.2 asks for.
    """

    def __init__(self, flow, *, sweep: str = "ilu",
                 psi_factory: Callable[[MomentumSystem], DC.Map] | None = None,
                 alpha: float = 0.0, k_max: int = 40, m_max: int = 40,
                 compare_classical: bool = True) -> None:
        self.flow = flow
        self.sweep = sweep
        self.psi_factory = psi_factory
        self.alpha = alpha
        self.k_max = k_max
        self.m_max = m_max
        self.compare_classical = compare_classical
        self.reports: list[CertifiedReport] = []

    def __call__(self, M, bu, bv, us, vs, precond):
        # rebuilt from the flow rather than from the arguments so that ONE
        # assembly is in play; the arguments are what `step` just built from it
        sys = MomentumSystem(self.flow, sweep=self.sweep, precond=precond)
        ref, cits, cwall = (None, None, float("nan"))
        if self.compare_classical:
            ref, cits, cwall = sys.classical()
        requested = (getattr(self.psi_factory, "__name__", "cheap")
                     if self.psi_factory is not None else None)
        psi = self.psi_factory(sys) if self.psi_factory is not None else None
        reason = None
        if self.psi_factory is not None and psi is None:
            # it could not be BUILT.  The null map is what will run, and the
            # report must say so rather than answer under the asked-for name.
            reason = ("the cheap map could not be built here (an incomplete "
                      "factor exactly singular after the retry); the NULL map "
                      "ran instead")
        w, rep = certified_momentum(sys, psi, psi_name=requested,
                                    psi_requested=requested,
                                    psi_unavailable_reason=reason,
                                    alpha=self.alpha, k_max=self.k_max,
                                    m_max=self.m_max, reference=ref,
                                    classical_its=cits, classical_wall_s=cwall)
        self.reports.append(rep)
        return w[0], w[1], [rep.outer_iterations, rep.inner_cheap_calls]


def certified_step_report(flow, *, sweep: str = "ilu",
                          psi_factory: Callable[[MomentumSystem], DC.Map] | None = None,
                          alpha: float = 0.0) -> dict[str, Any]:
    """One step taken twice from the same state -- classically and certified --
    and the distance between the two states.

    This is section 12's criterion 4 reduced to a number: the certified step must
    land on the classical step's answer.  It is measured on the WHOLE state the
    step leaves behind (u, v and p after the projection), not only on the
    momentum solve, because the claim is about the step.
    """
    keep = (flow.U.copy(), flow.V.copy(),
            None if flow.Um1 is None else flow.Um1.copy(),
            None if flow.Vm1 is None else flow.Vm1.copy(),
            flow.P.copy(), flow.t, flow.k)

    def restore():
        flow.U, flow.V = keep[0].copy(), keep[1].copy()
        flow.Um1 = None if keep[2] is None else keep[2].copy()
        flow.Vm1 = None if keep[3] is None else keep[3].copy()
        flow.P = keep[4].copy()
        flow.t, flow.k = keep[5], keep[6]

    flow.momentum_solver = None
    t0 = time.perf_counter()
    rec_c = flow.step()
    classical_s = time.perf_counter() - t0
    cU, cV, cP = flow.U.copy(), flow.V.copy(), flow.P.copy()

    restore()
    # ``compare_classical`` OFF: a certified step that fetches its own reference
    # by running the classical solve would be timed as classical PLUS certified,
    # and the price it reports would be an artefact of the instrument.  The gap
    # that matters is measured below, between the two STATES.
    solver = CertifiedMomentumSolver(flow, sweep=sweep, psi_factory=psi_factory,
                                     alpha=alpha, compare_classical=False)
    flow.momentum_solver = solver
    t0 = time.perf_counter()
    rec_k = flow.step()
    certified_s = time.perf_counter() - t0
    kU, kV, kP = flow.U.copy(), flow.V.copy(), flow.P.copy()
    flow.momentum_solver = None
    restore()

    du = float(DC.vector_rms(_stack(kU - cU, kV - cV)))
    scale = float(DC.vector_rms(_stack(cU, cV)))
    rep = solver.reports[-1] if solver.reports else None
    return {"sweep": sweep, "alpha": alpha,
            "psi_ran": None if rep is None else rep.psi_ran,
            "psi_requested": None if rep is None else rep.psi_requested,
            "psi_unavailable_reason": (None if rep is None
                                       else rep.psi_unavailable_reason),
            "velocity_gap": du, "velocity_scale": scale,
            "relative_velocity_gap": du / scale if scale > 0 else float("nan"),
            "pressure_gap": float(np.sqrt(np.mean((kP - cP) ** 2))),
            "pressure_scale": float(np.sqrt(np.mean(cP ** 2))),
            "classical_s": classical_s, "certified_s": certified_s,
            "classical_iterations": rec_c["iterations"],
            "certified": None if rep is None else rep.as_dict(),
            "max_div_classical": rec_c["max_div"], "max_div_certified": rec_k["max_div"]}
