"""L5 to L7 at runtime -- the coupled macro-step, and the rollout.

general-coupling-scheme §4.1, driven entirely by a compiled scheme.  This module
closes the loop that the compiler opens: without it the emit contract is a
promise, and the whole point of L8 is that it is a contract.

    lambda      <- the previous step's converged trace, the best available guess
    G(lambda)   <- for each agent, impose P_i lambda, take back the flux,
                   reduce with the forced adjoint P_i^*, and sum
    lambda*     <- solved by the accelerator the compiler chose
    u^{n+1}     <- assembled by the declared partition of unity
    emit        <- gamma, the per-port residuals, the power residual, the stamp

There is no physics here and there is no case-specific branch.  Every call into
an expert goes through ``boundary_response``, which is the same interface the
probe uses -- hand over a trace, take back a flux -- so a new case study adds
nothing to this file either.

**The one thing this module refuses to do** is run a configuration the compiler
refused.  A scheme that was refused is not a scheme with a warning attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .assembly import PartitionOfUnity
from .compiler import CompileResult
from .emit import ConservationReport
from .graph import CaseGraph, Connection
from .multiphysics import check_sigma_lag, lag_distance
from .probe import SeamOperator
from .scheme import Accelerator, Scheme
from .transfer import SeamTransfer
from .verdict import REFUSE


class RunRefused(RuntimeError):
    """The compiler refused this configuration; it is not runnable."""


class Declination(RuntimeError):
    """An agent declined. The rollout halts and the claim is typed on what ran."""

    def __init__(self, agent_id: str, step: int, t: float) -> None:
        super().__init__(
            f"agent {agent_id!r} declined at step {step} (t = {t:.6g}). The rollout halts "
            "and the claim is typed on the interval before it: a declination is not a "
            "failure, it is a shorter answer with a stated reason. Continuing past one is "
            "running with a hypothesis known to be false"
        )
        self.agent_id = agent_id
        self.step = step
        self.t = t


@dataclass
class StepResult:
    """One macro-step, with everything the emit contract asks of it."""

    step: int
    t: float
    trace: np.ndarray
    iterations: int
    gamma: float                       # ||G(lambda*)||: solve incompleteness
    residual_history: list[float] = field(default_factory=list)
    port_residuals: dict[str, float] = field(default_factory=dict)
    power_residual: float | None = None
    locals_: dict[str, np.ndarray] = field(default_factory=dict)
    assembled: np.ndarray | None = None
    declined_by: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "t": self.t,
            "iterations": self.iterations,
            "gamma": self.gamma,
            "port_residuals": dict(self.port_residuals),
            "power_residual": self.power_residual,
            "declined_by": self.declined_by,
        }


@dataclass
class InterfaceProblem:
    """The residual the whole composition is arranged around.

    ``G(lambda) = S lambda - chi``, assembled on the common interface space from
    the declared prolongations.  ``chi`` is the zero-probe response: what the
    agents return with no interface datum imposed.
    """

    seam_id: str
    S: np.ndarray
    chi: np.ndarray
    dim: int

    def residual(self, lam: np.ndarray) -> np.ndarray:
        return self.S @ np.asarray(lam, dtype=float) - self.chi

    @property
    def is_empty(self) -> bool:
        return bool(np.allclose(self.S, 0.0))


def build_interface_problem(
    graph: CaseGraph,
    connection: Connection,
    transfer: SeamTransfer,
    operator: SeamOperator,
) -> InterfaceProblem:
    """Assemble the seam's residual from the probed operator and the zero probe."""
    chi = np.zeros(transfer.space.dim)
    for agent_id, port_name in (connection.a, connection.b):
        caps = graph.agent(agent_id).capabilities
        P = transfer.prolongations[agent_id]
        zero_trace = np.zeros(P.n_V)
        flux0 = np.asarray(caps.boundary_response(port_name, zero_trace), dtype=float)
        chi -= P.reduce(flux0, transfer.space)
    return InterfaceProblem(connection.seam_id, operator.S, chi, transfer.space.dim)


# ---------------------------------------------------------------------------
# the accelerators -- each one's error is a bound, not a preference
# ---------------------------------------------------------------------------


def solve_interface(
    problem: InterfaceProblem,
    accelerator: Accelerator,
    lam0: np.ndarray | None = None,
    tol: float | None = None,
    max_iter: int = 200,
) -> tuple[np.ndarray, int, list[float]]:
    """Drive the returned trace toward the root of the problem that was posed.

    This layer controls gamma and nothing else. It cannot reduce the error from
    posing the interface problem with the wrong operator: iterating harder
    converges onto the root of whatever question was asked.
    """
    lam = np.zeros(problem.dim) if lam0 is None else np.asarray(lam0, dtype=float).copy()
    history: list[float] = [float(np.linalg.norm(problem.residual(lam)))]

    if problem.is_empty:
        # Every trace is equally consistent. Returning the initial guess after one
        # pass is the honest behaviour, and the compiler has already refused the
        # word "coupled" on anything built from it.
        return lam, 1, history

    if accelerator is Accelerator.DIRECT_SCHUR:
        lam = np.linalg.lstsq(problem.S, problem.chi, rcond=None)[0]
        history.append(float(np.linalg.norm(problem.residual(lam))))
        return lam, 1, history

    target = tol if tol is not None else 1e-10 * max(1.0, float(np.linalg.norm(problem.chi)))

    if accelerator in (Accelerator.KRYLOV, Accelerator.NEWTON_KRYLOV):
        lam, iters, history = _gmres_like(problem, lam, target, max_iter)
        return lam, iters, history

    # Richardson: the classical Schwarz sweep, whose rate is the contraction
    # factor and which advances information one neighbour per iteration.
    scale = float(np.linalg.norm(problem.S, 2))
    step = 1.0 / scale if scale > 0 else 1.0
    for k in range(1, max_iter + 1):
        r = problem.residual(lam)
        lam = lam - step * r
        history.append(float(np.linalg.norm(problem.residual(lam))))
        if history[-1] <= target:
            return lam, k, history
    return lam, max_iter, history


def _gmres_like(
    problem: InterfaceProblem,
    lam: np.ndarray,
    target: float,
    max_iter: int,
) -> tuple[np.ndarray, int, list[float]]:
    """Conjugate-gradient-style Krylov iteration on the interface.

    The rate is governed by the CONDITIONING of the transmission operator, not by
    subdomain count -- which is the whole reason a dense assembled operator
    couples the graph in one solve where a local exchange advances one neighbour
    per sweep.
    """
    history = [float(np.linalg.norm(problem.residual(lam)))]
    A = problem.S.T @ problem.S
    b = problem.S.T @ problem.chi
    r = b - A @ lam
    p = r.copy()
    rs = float(r @ r)
    for k in range(1, max_iter + 1):
        Ap = A @ p
        denom = float(p @ Ap)
        if denom == 0.0:
            break
        alpha = rs / denom
        lam = lam + alpha * p
        r = r - alpha * Ap
        rs_new = float(r @ r)
        history.append(float(np.linalg.norm(problem.residual(lam))))
        if history[-1] <= target:
            return lam, k, history
        p = r + (rs_new / rs) * p
        rs = rs_new
    return lam, len(history) - 1, history


# ---------------------------------------------------------------------------
# the coupled step
# ---------------------------------------------------------------------------


def coupled_step(
    result: CompileResult,
    graph: CaseGraph,
    step: int = 0,
    t: float = 0.0,
    lam0: dict[str, np.ndarray] | None = None,
    state: Any = None,
) -> StepResult:
    """One macro-step of the composed system, driven by the compiled scheme."""
    if result.verdict is REFUSE:
        raise RunRefused(
            f"{result.case} was refused at compile time. A refused scheme is not a scheme "
            "with a warning attached.\n"
            + "\n".join(f"  {d}" for d in result.decisions.refusals[:5])
        )
    scheme = result.scheme
    assert scheme is not None

    lam0 = lam0 or {}
    traces: dict[str, np.ndarray] = {}
    port_residuals: dict[str, float] = {}
    gamma_sq = 0.0
    iterations = 0
    history: list[float] = []
    locals_: dict[str, np.ndarray] = {}
    power_terms: list[float] = []

    for conn in graph.connections:
        transfer = result.transfers.get(conn.seam_id)
        operator = result.seam_operators.get(conn.seam_id)
        if transfer is None or operator is None:
            continue

        # C8 at runtime: the connection is legal, and whether it is VALID is
        # decided per step by the predicate.
        for agent_id, _ in (conn.a, conn.b):
            caps = graph.agent(agent_id).capabilities
            if caps.validity is not None and not _validity(caps, state):
                raise Declination(agent_id, step, t)

        problem = build_interface_problem(graph, conn, transfer, operator)
        lam, iters, hist = solve_interface(
            problem, scheme.accelerator, lam0.get(conn.seam_id), scheme.eps_tol
        )
        traces[conn.seam_id] = lam
        iterations = max(iterations, iters)
        history.extend(hist)
        gamma_sq += float(np.linalg.norm(problem.residual(lam))) ** 2

        f_a, f_b = _port_fluxes(graph, conn, transfer, lam, scheme)
        port_residuals[conn.seam_id] = _port_residual(f_a, f_b)
        power_terms.append(_seam_power(lam, f_a, f_b, transfer, conn))

        for agent_id, port_name in (conn.a, conn.b):
            caps = graph.agent(agent_id).capabilities
            P = transfer.prolongations[agent_id]
            locals_[f"{agent_id}@{conn.seam_id}"] = np.asarray(
                caps.boundary_response(port_name, P.prolong(lam, transfer.space)), dtype=float
            )

    assembled = None
    pou: PartitionOfUnity | None = graph.partition_of_unity
    if pou is not None:
        keyed = {k: locals_[k] for k in pou.restrictions if k in locals_}
        if len(keyed) == len(pou.restrictions):
            assembled = pou.assemble(keyed)

    return StepResult(
        step=step,
        t=t,
        trace=np.concatenate([traces[k] for k in sorted(traces)]) if traces else np.zeros(0),
        iterations=iterations,
        gamma=float(np.sqrt(gamma_sq)),
        residual_history=history,
        port_residuals=port_residuals,
        power_residual=float(sum(power_terms)) if power_terms else None,
        locals_=locals_,
        assembled=assembled,
    )


def _validity(caps: Any, state: Any) -> bool:
    try:
        return bool(caps.validity(state, None))
    except TypeError:
        return bool(caps.validity(state))


def _substep_quadrature(graph: CaseGraph, agent_id: str, exchange: float | None) -> int:
    """How many of this agent's own steps make up one exchange interval.

    Derived from the declared clocks, never declared: ``DT / dt_native``.  L7/R9
    has already refused the graph if that is not a whole number, so rounding here
    cannot silently paper over clocks that do not nest.
    """
    dt_i = graph.agent(agent_id).capabilities.dt_native
    if not dt_i or not exchange or dt_i <= 0.0:
        return 1
    return max(1, int(round(exchange / dt_i)))


def _port_fluxes(
    graph: CaseGraph,
    conn: Connection,
    transfer: SeamTransfer,
    lam: np.ndarray,
    scheme: Scheme | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Each side's flux, reduced to the common space by the forced adjoint.

    **R9, 2026-08-29.**  When the compiled scheme says ``time-integrated``, each
    side is asked for the flux AVERAGED over the exchange interval on its own
    sub-steps rather than the value at the end of it.  L7 has already refused any
    graph that declares the matching and cannot supply the callable, so reaching
    here without one would be a compile that admitted a scheme it cannot run --
    the assertion is there to make that loud rather than to handle it.
    """
    integrated = scheme is not None and scheme.flux_matching == "time-integrated"
    out = []
    for agent_id, port_name in (conn.a, conn.b):
        caps = graph.agent(agent_id).capabilities
        P = transfer.prolongations[agent_id]
        trace_V = P.prolong(lam, transfer.space)
        if integrated:
            assert caps.boundary_response_integrated is not None, (
                f"{agent_id} has no boundary_response_integrated and the scheme is "
                "time-integrated; L7/R9/quadrature should have refused this compile"
            )
            n = _substep_quadrature(graph, agent_id,
                                    scheme.exchange_interval if scheme else None)
            flux_V = np.asarray(
                caps.boundary_response_integrated(port_name, trace_V, n), dtype=float)
        else:
            flux_V = np.asarray(caps.boundary_response(port_name, trace_V), dtype=float)
        out.append(P.reduce(flux_V, transfer.space))
    return out[0], out[1]


def _port_residual(f_a: np.ndarray, f_b: np.ndarray) -> float:
    """The per-port residual, reported everywhere rather than on selected edges.

    What leaves one agent through a port is what enters the other: the pairing
    is the definition of a port connection, not a label applied to some of them.
    """
    denom = 0.5 * (float(np.linalg.norm(f_a)) + float(np.linalg.norm(f_b)))
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(f_a + f_b)) / denom


def _seam_power(
    lam: np.ndarray,
    f_a: np.ndarray,
    f_b: np.ndarray,
    transfer: SeamTransfer,
    conn: Connection,
) -> float:
    """The power crossing one seam, for the global residual.

    Because the exchanged variables ARE a power bond, the energy the coupling
    creates or destroys is computable from the coupling values alone -- the
    cheapest possible physical-plausibility monitor, and it needs no reference
    data.
    """
    G = transfer.space.G
    return float(lam @ (G @ (f_a + f_b)))


def rollout(
    result: CompileResult,
    graph: CaseGraph,
    n_steps: int,
    dt: float | None = None,
    state: Any = None,
) -> tuple[list[StepResult], ConservationReport, str | None]:
    """Roll the composed system forward, halting on a declination.

    Returns the steps that ran, the conservation report over them, and the agent
    that declined if one did.  A declination is not an exception the caller must
    handle away: the claim is typed on the interval that ran.
    """
    dt = dt or graph.macro_dt or 1.0
    steps: list[StepResult] = []
    lam: dict[str, np.ndarray] = {}
    declined: str | None = None
    # **W86.** The lag the run actually carries, derived from the traces this
    # loop was already carrying from one step to the next. `sigma` is a function
    # of that lag and the certificate quotes one number for it, so a run that
    # never computes its own lag cannot tell whether the constant it is quoting
    # belongs to it.
    lag_seen: dict[str, float] = {}

    for n in range(n_steps):
        try:
            sr = coupled_step(result, graph, step=n, t=n * dt, lam0=lam, state=state)
        except Declination as exc:
            declined = exc.agent_id
            break
        steps.append(sr)
        offset = 0
        for seam in sorted(sr.port_residuals):
            transfer = result.transfers.get(seam)
            if transfer is None:
                continue
            d = transfer.space.dim
            new = sr.trace[offset:offset + d]
            if seam in lam:
                lag_seen[seam] = lag_distance(lam[seam], new)
            lam[seam] = new
            offset += d

    report = ConservationReport()
    if steps:
        last = steps[-1]
        report.per_port = dict(last.port_residuals)
        report.power_residual = last.power_residual
    declared_lag = getattr(graph.measured, "sigma_lag", None) if graph.measured else None
    report.lag = {seam: check_sigma_lag(declared_lag, d).as_dict()
                  for seam, d in sorted(lag_seen.items())}
    return steps, report, declined
