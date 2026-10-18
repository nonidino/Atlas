"""Nonlinear conduction through CO-D's coefficient channel (v2's addition (a)): no new
network, version 1's guarantee for every inner solve.

**Frozen coefficients** ([[chart-operator-general-form]] section 2.4; the Kacanov
iteration, Heid and Wihler, *Math. Comp.* 89, 2020).  For ``-div(kappa(T) grad T) = 0``
iterate ``-div(kappa(T^k) grad T^{k+1}) = 0``: each step is a linear problem with a
heterogeneous coefficient, which is exactly what the Stage 0 expert was trained for.
Per piece and step, ``kappa^k`` is evaluated from ``T^k`` at the nodes (the network's
``log kappa_hat`` channel, centred on its geometric mean, which becomes the piece's
``kappa_ref``) and at the element centres (the energy's coefficient, from the mean of
the element's four nodal values).  The classical reference runs the same iteration
with the full-domain solve in the same discretisation, so both converge to the same
discrete nonlinear problem's solution if the expert is exact.

**The Kirchhoff transform**, in one material: ``Theta = int_300^T kappa(s) ds`` makes
the steady problem linear in ``Theta`` (``kappa(T) grad T = grad Theta``), with the
Dirichlet data transformed; reported for the one-material benchmarks as a
cross-check of a different discretisation of the same PDE.
"""

from __future__ import annotations

import math

import numpy as np

from . import host as hs
from .benchmarks import Bench
from .piece import PieceProblem, element_from_nodal

T_REF = 300.0


def kappa_of(T, k0: float, beta: float):
    return k0 * (1.0 + beta * (np.asarray(T, float) - T_REF))


def frozen_problems(bench: Bench, fields: list, betas: list, k0s: list) -> list[PieceProblem]:
    """The pieces with kappa frozen at the nodal temperatures ``fields``."""
    out = []
    for p, u, beta, k0 in zip(bench.problems, fields, betas, k0s):
        n = p.n
        Tn = np.asarray(u, float).reshape(n + 1, n + 1)
        kn = kappa_of(Tn, k0, beta)
        ke = kappa_of(element_from_nodal(Tn), k0, beta)
        if np.any(kn <= 0) or np.any(ke <= 0):
            raise ValueError("kappa(T) is not positive on this temperature field")
        lref = float(np.mean(np.log(kn)))
        out.append(PieceProblem(p.chart, log_kappa_hat=np.log(kn) - lref,
                                log_kappa_hat_e=np.log(ke) - lref, fo=None, g=np.zeros((n + 1, n + 1)),
                                kappa_ref=math.exp(lref), label=p.label))
    return out


def iterate(bench: Bench, betas: list, expert=None, kind: str = "sine-hat", m: int = 16,
            tol: float = 1e-6, maxit: int = 50, start: list | None = None) -> dict:
    """Frozen-coefficient iteration from ``start`` (default: the linear solution).
    ``expert=None`` uses the full-domain classical solve; otherwise tier 0 with the
    expert, one call per piece per outer iteration.  Convergence: the largest nodal
    change over the solution's range below ``tol``."""
    k0s = [p.kappa_ref for p in bench.problems]

    def solve(problems):
        L = hs.Layout(problems, bench.bc, kind=kind, m=m)
        if expert is None:
            return hs.monolithic(L).fields
        return L.solve(expert).fields
    fields = start if start is not None else solve(bench.problems)
    span = float(max(np.max(u) for u in fields) - min(np.min(u) for u in fields))
    hist = []
    calls0 = getattr(expert, "calls", 0)
    converged = False
    for it in range(1, maxit + 1):
        new = solve(frozen_problems(bench, fields, betas, k0s))
        change = max(float(np.max(np.abs(a - b))) for a, b in zip(new, fields)) / span
        hist.append(change)
        fields = new
        if change < tol:
            converged = True
            break
    return {"fields": fields, "iterations": len(hist), "converged": converged,
            "history": hist, "span": span,
            "network_calls": (getattr(expert, "calls", 0) - calls0) if expert is not None else 0}


def kirchhoff(bench: Bench, beta: float, linear_fields: list, t_hot: float, t_cold: float) -> list:
    """One material: map the linear solution through Theta.  The linear problem with
    Dirichlet data 0 and 1 is phi = (T_lin - t_cold) / (t_hot - t_cold); Theta = Theta_hot phi
    (Theta(t_cold) = 0 when t_cold = 300 K), then T = Theta^-1."""
    k0 = 1.0          # Theta / k0: the material constant cancels
    th = lambda T: (T - T_REF) + 0.5 * beta * (T - T_REF) ** 2
    th_hot, th_cold = th(t_hot), th(t_cold)
    out = []
    for u in linear_fields:
        phi = (np.asarray(u) - t_cold) / (t_hot - t_cold)
        Th = th_cold + (th_hot - th_cold) * phi
        # solve (T - 300) + beta/2 (T - 300)^2 = Th for the root near the linear one
        if abs(beta) < 1e-15:
            out.append(T_REF + Th)
            continue
        disc = 1.0 + 2.0 * beta * Th
        out.append(T_REF + (-1.0 + np.sqrt(disc)) / beta)
    return out
