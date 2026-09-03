"""Error attribution at a seam whose two sides solve different equations.

**The blocker this removes, measured 2026-08-29.**  ``E3`` asks whether the two
sides of a seam declare the same ``governing_family``, by string comparison, and
when they do not the compiler emits ``tau`` as ``UNDEFINED`` for both sides.  That
has been the standing reason a multiphysics graph cannot be certified: the
composition *runs* -- probing never mentions a governing equation, so the
transmission layer is robust to E3 -- and the error attribution has no meaning.

Read what ``tau`` actually is, in `scripts/tier0_window_ns.py`:

    u_ref = the reference trajectory
    u_t   = the composed step given the TRUE trace   ->  tau = ||u_t - u_ref||
    u_c   = the composed step with the LAGGED trace  ->  sigma = ||u_c - u_t||

**Nothing in that mentions a governing equation.**  What it requires is a
*reference trajectory*, and at a multiphysics seam one exists and is constructible
from the agents themselves: the **tightly coupled** solve, in which the interface
condition is converged inside the macro-step instead of lagged across it.  That
is the exact analogue of the single-physics monolith, which is likewise not
ground truth but "the same expert applied without the cut".

So ``E3`` was gating the wrong thing.  Sharing a governing family is what lets two
agents share a *monolithic* reference; it is not what ``tau`` needs.  What ``tau``
needs is a **reference pair**, and `lambda_ref` -- a field on the record since the
end-to-end spec, documented as *"tau at a multiphysics seam"* -- was the
declaration that one exists.  It was checked for presence and never consumed.

Measured on `thermal_seam` with the gas replaced by a surrogate carrying a known
conductivity error, against the reference pair:

    surrogate      tau        recovered
    k_gas x1.05    5.0000e-02   the injected 5%
    k_gas x1.25    2.5000e-01   the injected 25%
    k_gas x2.00    1.0000e+00   the injected 100%

exactly, per agent, with ``tau`` on the *unswapped* side staying identically zero.
Attribution works and it never needed the families to match.

The second problem, and it is the one that makes this a module rather than a rule
-----------------------------------------------------------------------------

The master bound adds ``tau + sigma + gamma`` as scalars, which presumes a single
norm.  At a multiphysics seam there is no single norm: the gas's state is a field
of conserved variables and the shell's is a temperature in kelvin, and there is no
scalar sum of the two.  Worse, an agent's own state norm can be *blind* -- over
one 1e-4 s step the gas's interior is **bit-identical** for every wall temperature
from 352 K to 450 K while its wall flux varies smoothly by 10%, because the
isothermal wall enters through a ghost state the interior has not yet felt.  A
defect measured in that norm would read exactly zero.

The port algebra already carries the common currency.  Every entry of
``PORT_SPECS`` pairs its two halves so that **effort times flow is a power** --
that is why `THERM`'s bond is $(T,\\ q_n/T)$ and not the pseudo-bond $(T,\\ q_n)$,
the choice W66 rests on.  Power is a unit neither side owns, both sides agree on,
and every port type already has.  So the defect at a multiphysics seam is measured
in **interface power**, and then it is commensurable and it can be summed.

Measured, same surrogates, and the master bound's shape survives:

    surrogate      tau_P        sigma_P      total_P      tau+sigma
    reference      0.0000e+00   8.6966e-02   0.0000e+00   8.6966e-02
    k_gas x1.05    5.0000e-02   9.6339e-02   5.5025e-02   1.4634e-01
    k_gas x1.25    2.5000e-01   1.3920e-01   2.8049e-01   3.8920e-01
    k_gas x2.00    1.0000e+00   3.8449e-01   1.2106e+00   1.3845e+00

sub-additive on every row.

**What this rests on, stated rather than buried.**  Computing a power needs to
know which half of the conjugate pair the expert returns, which is
``response_half`` -- one of W69's three unverifiable declarations.  So
multiphysics attribution inherits that dependency, and `interface_power` refuses
rather than guessing when the half is ``UNDECLARED``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .ports import PortType, ResponseHalf, spec_for


class MultiphysicsError(RuntimeError):
    """The attribution could not be run as configured."""


# ---------------------------------------------------------------------------
# the common currency
# ---------------------------------------------------------------------------


def interface_power(
    trace: np.ndarray,
    response: np.ndarray,
    half: ResponseHalf,
    measure: np.ndarray | float = 1.0,
) -> float:
    """Integrated ``effort x flow`` over the seam: the one norm both sides share.

    ``half`` says which half of the conjugate pair ``response`` IS, so that the
    trace supplies the other.  It is refused rather than assumed when
    ``UNDECLARED``: no property of the returned numbers distinguishes an effort
    from a flow (W66), and guessing here would put a sign and a unit error into
    every attribution downstream.

    The measure is the V-space cell measure, so this is a quadrature of the bond
    over the interface and carries the port's ``power_area`` units.
    """
    if half is ResponseHalf.UNDECLARED:
        raise MultiphysicsError(
            "interface_power needs response_half: the bond's two halves are not "
            "distinguishable from the returned numbers, so which one an expert "
            "returns has to be declared (W66). Attribution inherits that "
            "dependency rather than hiding it"
        )
    t = np.asarray(trace, dtype=float).ravel()
    r = np.asarray(response, dtype=float).ravel()
    if t.shape != r.shape:
        raise MultiphysicsError(
            f"trace has length {t.size} and response {r.size}; the bond is pointwise"
        )
    effort, flow = (t, r) if half is ResponseHalf.FLOW else (r, t)
    return float(np.sum(effort * flow * np.asarray(measure, dtype=float)))


def bond_of(port_type: PortType) -> tuple[str, str]:
    """The declared (effort, flow) names for a port type, from the port algebra."""
    spec = spec_for(port_type)
    return (spec.effort, spec.flow)


# ---------------------------------------------------------------------------
# the lag -- whose property it is, measured
# ---------------------------------------------------------------------------


def lag_distance(previous: np.ndarray, current: np.ndarray) -> float:
    """How far the interface moved over one macro-step: ``max |lam_n - lam_(n-1)|``.

    **W86, 2026-08-29.**  A lagged composition imposes the PREVIOUS step's trace,
    so this is exactly the lag ``sigma`` is a function of, and the run has both
    traces in hand -- `solve.rollout` already carries ``lam[seam]`` from one step
    to the next.  Nothing had ever computed it.

    Max rather than a norm because the lag enters the response pointwise, and a
    mean would hide a lag that is large somewhere and zero elsewhere -- which is
    the case that breaks the uniform-shift picture below.
    """
    p = np.asarray(previous, dtype=float).ravel()
    c = np.asarray(current, dtype=float).ravel()
    if p.shape != c.shape:
        raise MultiphysicsError(
            f"the two traces have lengths {p.size} and {c.size}; a lag is a "
            "difference between consecutive states of one interface"
        )
    return float(np.max(np.abs(c - p)))


@dataclass
class SigmaLagCheck:
    """Whether a quoted ``sigma`` was measured at the lag the run actually carries.

    Necessary and **not sufficient**, and the gap is measured rather than
    conceded: on `thermal_seam`, two consecutive macro-steps whose lag distances
    agree to 8% (1.087e-3 against 1.174e-3 K) carry sigmas that differ by
    **1.48x**, because sigma's argument is the lag PROFILE and this is a scalar
    summary of it.  So a mismatch falsifies the quoted constant and a match
    establishes only that the provenance is not obviously wrong.
    """

    declared: float | None
    actual: float
    ratio: float | None
    agrees: bool | None
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {"declared_lag": self.declared, "actual_lag": self.actual,
                "ratio": self.ratio, "agrees": self.agrees, "note": self.note}


#: How far the run's lag may sit from the declared one before the quoted sigma is
#: reported as measured somewhere else.  A factor of two, and it is a **tolerance
#: on provenance rather than on a bound**: sigma moved 7.70x over five consecutive
#: macro-steps of `thermal_seam` while the lag moved 3.43x, so nothing tighter
#: would mean anything and nothing looser would catch a constant quoted from a
#: different regime.
SIGMA_LAG_TOLERANCE = 2.0


def check_sigma_lag(declared: float | None, actual: float) -> SigmaLagCheck:
    """Compare a quoted sigma's lag against the one a run carries."""
    if declared is None:
        return SigmaLagCheck(
            None, actual, None, None,
            f"sigma carries no declared lag, so the run's own {actual:.4g} cannot be "
            "compared against anything. A sigma quoted without its lag is not a "
            "number (W86): it moved 7.70x over five consecutive macro-steps of the "
            "one seam this has been measured on",
        )
    if declared <= 0.0:
        return SigmaLagCheck(
            declared, actual, None, None,
            f"the declared lag is {declared:.4g}, which names no scale to compare "
            "against; sigma at zero lag is zero by construction and is a check that "
            "the measurement is measuring the lag, not a value to quote",
        )
    ratio = actual / declared
    ok = (1.0 / SIGMA_LAG_TOLERANCE) <= ratio <= SIGMA_LAG_TOLERANCE
    return SigmaLagCheck(
        declared, actual, ratio, ok,
        (f"the run's lag is {actual:.4g} against the {declared:.4g} sigma was measured "
         f"at, a factor of {ratio:.3g}. "
         + ("Within the provenance tolerance -- which establishes that the constant "
            "was not quoted from a different regime, and NOT that it is right here: "
            "two steps whose lags agree to 8% carry sigmas 1.48x apart, because the "
            "argument is the lag profile and this is a scalar summary of it"
            if ok else
            "Outside the provenance tolerance, so the quoted sigma was measured at a "
            "lag this run does not carry and the bound it feeds is not this run's")),
    )


# ---------------------------------------------------------------------------
# the reference trajectory: the tightly coupled pair
# ---------------------------------------------------------------------------


@dataclass
class TightCoupling:
    """The converged interface state, and what it cost to get there.

    ``residual(lam)`` is the interface condition -- for a two-sided seam the sum
    of the two flows, which is zero when the bond balances.  The iteration is
    Newton on the probed operator when one is supplied and a damped secant
    otherwise, because a probed ``S`` is exactly the Jacobian this needs and the
    composition has usually already paid for it.

    This is a **measurement instrument, not a scheme.**  Converging the interface
    inside the macro-step is what the composition is trying to avoid; running it
    once buys the reference trajectory that ``tau`` is defined against, in the
    same way `tier0_window_ns` runs a monolith it would never ship.
    """

    trace: np.ndarray
    residual_norm: float
    iterations: int
    converged: bool
    history: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"residual_norm": self.residual_norm, "iterations": self.iterations,
                "converged": self.converged, "history": list(self.history),
                "trace_mean": float(np.mean(self.trace))}


def numerical_jacobian(
    residual: Callable[[np.ndarray], np.ndarray],
    trace: np.ndarray,
    step: float | None = None,
) -> np.ndarray:
    """The interface residual's Jacobian in V, by finite differences.

    ``n + 1`` residual evaluations, so ``2(n + 1)`` solves for a two-sided seam.
    That is expensive and it is the honest price of a *converged* referent: a
    scalar secant on the mean residual moves only a uniform shift of the trace,
    and on `thermal_seam` it stalls five orders in at 3.04e-3 because the part of
    the residual that varies along the seam is untouchable by a uniform move.
    Measured: with this Jacobian the same solve reaches 1e-11 in 3 iterations.

    **W85, settled 2026-08-29, and the answer is no.**  The probed ``S`` is the
    same object compressed to M and it is already paid for, so it is the obvious
    cheap substitute.  Measured on `thermal_seam` over five operating points --
    ``dt`` across 4x, the film coefficient across 100x, ``operator_content`` from
    2.00e-05 to 3.55e-04 -- the coarse-space Newton `seam_jacobian` builds from it
    **never converges, in any of them**, and the reason is not conditioning:

        residual in range(P)      8.3e-14 ... 1.0e-11     machine zero
        residual orthogonal to it 1.04e-04 ... 5.4e-04    exactly the plateau

    A step of the form ``P S^-1 R r`` lives in ``range(P)``, and ``dim M = 16``
    against ``dim V = 48`` leaves 32 directions the step cannot reach.  The
    iteration drives what it can see to machine zero in 6 iterations and stops.
    **No threshold on ``operator_content`` can fix a subspace**, and none is
    licensed by the data either: over a 17.8x range of omega the outcome does not
    move at all.

    What ``S`` is worth is a **warm start** -- 24 to 32 solves against 106 to 116,
    landing 5.2e-05 K from the converged trace -- and only at a *consistent seam
    base*.  At the per-expert bases the compiler still defaults to (W80), the same
    operator leaves 2.6e-02 in range(P) and lands 3.0e-03 away: 245x worse in
    residual, 58x in trace.  That is W74's class getting a numerical consequence
    for the first time, rather than a certificate one.
    """
    lam = np.asarray(trace, dtype=float).ravel()
    n = lam.size
    h = step if step is not None else 1e-4 * max(1.0, float(np.mean(np.abs(lam))))
    r0 = np.asarray(residual(lam), dtype=float).ravel()
    J = np.empty((r0.size, n))
    for k in range(n):
        d = lam.copy()
        d[k] += h
        J[:, k] = (np.asarray(residual(d), dtype=float).ravel() - r0) / h
    return J


def seam_jacobian(
    S: np.ndarray,
    P: np.ndarray,
    R: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """A Newton step built from the seam's probed operator: ``r -> -P S^-1 R r``.

    Pass the result as ``tight_couple(jacobian=...)`` to use the operator the
    compiler already assembled instead of paying for a dense finite-difference
    one.  ``P`` is the prolongation ``M -> V`` and ``R`` its forced adjoint.

    **It does not converge, and the docstring of `numerical_jacobian` has the
    numbers (W85).**  The step lives in ``range(P)``, so the residual's component
    orthogonal to a ``dim M``-dimensional subspace of ``V`` is untouchable at any
    iteration count -- measured at 1.04e-04 against an in-subspace residual driven
    to 1e-13.  This exists because a warm start at a quarter the cost is worth
    having and because the alternative to exposing it was leaving a plausible
    substitution untested; it is not a default and `tight_couple` will not pick
    it.
    """
    S = np.asarray(S, dtype=float)
    P = np.asarray(P, dtype=float)
    R = np.asarray(R, dtype=float)

    def step(r: np.ndarray) -> np.ndarray:
        rhs = -(R @ np.asarray(r, dtype=float).ravel())
        try:
            x = np.linalg.solve(S, rhs)
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(S, rhs, rcond=None)[0]
        return P @ x

    return step


def tight_couple(
    residual: Callable[[np.ndarray], np.ndarray],
    trace0: np.ndarray,
    jacobian: np.ndarray | str | Callable[[np.ndarray], np.ndarray] | None = "fd",
    tol: float = 1e-10,
    max_iter: int = 50,
    damping: float = 1.0,
) -> TightCoupling:
    """Converge the interface condition inside one macro-step.

    Returns the reference trace.  ``tau`` and ``sigma`` are both defined against
    the trajectory this produces, so a run that does not converge produces no
    attribution rather than a bad one -- ``converged`` is checked by the caller
    and `seam_defect_split` refuses on it.

    ``jacobian`` is ``"fd"`` (the default: a dense finite-difference Jacobian),
    an array in V, a **callable** ``r -> step`` such as `seam_jacobian` returns,
    or ``None`` for the scalar secant.  A dense array of the wrong shape is
    refused rather than silently falling through to the secant -- which is what
    passing a ``dim M`` operator used to do.
    """
    lam = np.asarray(trace0, dtype=float).ravel().copy()
    r = np.asarray(residual(lam), dtype=float).ravel()
    hist = [float(np.linalg.norm(r))]
    solve_step: Callable[[np.ndarray], np.ndarray] | None = None
    J = None
    if isinstance(jacobian, str):
        if jacobian != "fd":
            raise MultiphysicsError(f"unknown jacobian mode {jacobian!r}; use 'fd'")
        J = numerical_jacobian(residual, lam)
    elif callable(jacobian):
        solve_step = jacobian
    elif jacobian is not None:
        J = np.asarray(jacobian, dtype=float)
        if J.shape != (lam.size, lam.size):
            raise MultiphysicsError(
                f"the supplied Jacobian is {J.shape} and the trace is {lam.size}; a "
                "seam operator on M is not a Jacobian on V. Wrap it with "
                "`seam_jacobian(S, P, R)`, which is explicit about the subspace it "
                "steps in -- and read that function's docstring first, because it does "
                "not converge (W85)"
            )
    n_it = 0
    for n_it in range(1, max_iter + 1):
        if hist[-1] <= tol * max(1.0, float(np.linalg.norm(lam))):
            break
        if solve_step is not None:
            step = np.asarray(solve_step(r), dtype=float).ravel()
        elif J is not None:
            try:
                step = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(J, -r, rcond=None)[0]
        else:
            # No operator supplied: a scalar secant on the mean residual, which
            # is enough for a seam whose response is monotone in the trace and
            # is reported as such rather than dressed up as a Newton solve.
            eps = 1e-4 * max(1.0, float(np.mean(np.abs(lam))))
            r1 = np.asarray(residual(lam + eps), dtype=float).ravel()
            slope = float(np.mean(r1 - r) / eps)
            if slope == 0.0:
                break
            step = np.full_like(lam, -float(np.mean(r)) / slope)
        lam = lam + damping * step
        r = np.asarray(residual(lam), dtype=float).ravel()
        hist.append(float(np.linalg.norm(r)))
    ok = hist[-1] <= tol * max(1.0, float(np.linalg.norm(lam)))
    return TightCoupling(trace=lam, residual_norm=hist[-1], iterations=n_it,
                         converged=bool(ok), history=hist)


# ---------------------------------------------------------------------------
# the split
# ---------------------------------------------------------------------------


@dataclass
class SeamDefect:
    """``tau`` per agent and ``sigma``, at a seam, in interface power.

    ``tau[a]`` is agent ``a``'s infidelity *given the true trace* -- the reference
    pair is run at the same trace, so anything left is the agent.  ``sigma`` is
    what lagging the trace costs on top.  Both are relative to the reference
    interface power, so both are dimensionless and summable, which is the whole
    reason the norm is a power.
    """

    seam_id: str
    tau: dict[str, float]
    sigma: float
    total: float
    power_reference: float
    trace_reference: float
    converged: bool
    note: str = ""

    @property
    def tau_worst(self) -> tuple[str, float] | None:
        if not self.tau:
            return None
        k = max(self.tau, key=lambda a: self.tau[a])
        return k, self.tau[k]

    @property
    def subadditive(self) -> bool:
        """The master bound's shape: the whole is no worse than the parts.

        Not an identity and not asserted as one -- ``total`` is measured
        independently and this reports whether the inequality held.
        """
        return self.total <= sum(self.tau.values()) + self.sigma + 1e-12

    def as_dict(self) -> dict[str, Any]:
        return {"seam": self.seam_id, "tau": dict(self.tau), "sigma": self.sigma,
                "total": self.total, "power_reference": self.power_reference,
                "trace_reference": self.trace_reference,
                "converged": self.converged, "subadditive": self.subadditive,
                "tau_worst": self.tau_worst, "note": self.note}


def seam_defect_split(
    seam_id: str,
    actual: dict[str, Callable[[np.ndarray], np.ndarray]],
    reference: dict[str, Callable[[np.ndarray], np.ndarray]],
    halves: dict[str, ResponseHalf],
    trace0: np.ndarray,
    lagged_trace: np.ndarray,
    measure: np.ndarray | float = 1.0,
    jacobian: np.ndarray | str | None = "fd",
    tol: float = 1e-10,
) -> SeamDefect:
    """``tau`` and ``sigma`` at a seam, without asking what equations either side solves.

    ``actual`` and ``reference`` map agent id -> ``trace -> response``.  The
    reference pair defines the trajectory; the actual pair is what is being
    certified.  Supplying the same callables for both is legitimate and gives
    ``tau = 0``, which is the correct answer for a composition of exact solvers
    and is worth being able to state.

    The three quantities, each one measurement:

        reference   converge the REFERENCE pair            -> lam*, P_ref
        tau_a       run ACTUAL agent a at lam*             -> |P_a - P_ref| / P_ref
        sigma       run the ACTUAL pair at its own lam,
                    then at the lagged trace               -> |P_lag - P_conv| / P_ref
    """
    ids = sorted(reference)
    if set(ids) != set(actual):
        raise MultiphysicsError(
            f"the actual and reference pairs name different agents: "
            f"{sorted(actual)} against {ids}"
        )
    missing = [a for a in ids if a not in halves]
    if missing:
        raise MultiphysicsError(f"no response_half declared for {missing}")

    def residual_of(pair):
        def r(lam):
            return sum(np.asarray(pair[a](lam), dtype=float).ravel() for a in ids)
        return r

    def powers(pair, lam):
        """The per-agent bond power VECTOR, not its sum.

        The sum is zero at a converged interface -- that is what balance means --
        so it is exactly the wrong scale to divide by. What crosses the seam is
        the per-side magnitude, and at balance the two agree.
        """
        return np.array(
            [interface_power(lam, pair[a](lam), halves[a], measure) for a in ids]
        )

    ref_solve = tight_couple(residual_of(reference), trace0, jacobian, tol=tol)
    lam_star = ref_solve.trace
    if not ref_solve.converged:
        return SeamDefect(
            seam_id=seam_id, tau={}, sigma=float("nan"), total=float("nan"),
            power_reference=float("nan"), trace_reference=float(np.mean(lam_star)),
            converged=False,
            note=(f"the reference pair did not converge in {ref_solve.iterations} "
                  f"iterations (residual {ref_solve.residual_norm:.3e}); there is no "
                  "reference trajectory, so tau and sigma have no referent here and "
                  "are not reported. This is E3's verdict arrived at HONESTLY, by "
                  "failing to build the referent rather than by comparing strings"),
        )

    P_ref = powers(reference, lam_star)
    scale = float(np.max(np.abs(P_ref)))
    if scale == 0.0:
        raise MultiphysicsError(
            f"seam {seam_id}: no power crosses the reference interface, so a relative "
            "defect is undefined. That is the empty-seam case L4 already refuses"
        )

    # Exactly tier0's three runs, with the state norm replaced by the bond power:
    #   P_t  the ACTUAL agents at the TRUE trace          -> tau
    #   P_c  the ACTUAL agents at the LAGGED trace        -> sigma against P_t
    #   P_ref the REFERENCE agents at the true trace      -> total against P_c
    # Because sigma is taken against P_t and not against the actual pair's own
    # converged state, `total <= tau + sigma` is the triangle inequality and not
    # a hope. `subadditive` still reports it, because an identity nobody checks
    # is how a wrong norm would hide.
    P_t = powers(actual, lam_star)
    P_c = powers(actual, np.asarray(lagged_trace, dtype=float).ravel())

    tau = {a: abs(float(P_t[i] - P_ref[i])) / scale for i, a in enumerate(ids)}
    return SeamDefect(
        seam_id=seam_id,
        tau=tau,
        sigma=float(np.max(np.abs(P_c - P_t))) / scale,
        total=float(np.max(np.abs(P_c - P_ref))) / scale,
        power_reference=scale,
        trace_reference=float(np.mean(lam_star)),
        converged=True,
        note=("measured in interface power (effort x flow), the one unit both sides "
              "of a multiphysics seam share. Each agent's own state norm is not "
              "commensurable with the other's and can be blind: thermal_seam's gas "
              "state is bit-identical across 100 K of wall temperature over one "
              "macro-step while its wall flux moves 10%"),
    )
