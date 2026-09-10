"""The seven-axis coupling scheme, and the rules that bound it.

general-coupling-scheme §1.  Every coupling method the vault has considered --
one-pass exchange, halo Schwarz, flux-BC Schwarz, a thrust fixed point, waveform
relaxation, the probed Schur solve -- is one point in a seven-dimensional space:

    Sigma = ( D, Lambda_tilde, O, K, C, W, eps_tol )

Each axis maps to exactly one term of the master bound, so **choosing a scheme is
choosing an error budget**.  The scheme is not designed per case; it is compiled
from the declarations by ``compiler.compile_scheme`` under the rules R1-R9, which
live here so they can be cited by number in a refusal.

Note what does not appear on any axis: agent infidelity tau.  It is a property of
the agents, not of the coupling, and no choice of scheme reduces it.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

from .capability import Transmission
from .graph import Decomposition


class Ordering(enum.Enum):
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    RESTRICTED_ADDITIVE = "restricted-additive"


class Accelerator(enum.Enum):
    RICHARDSON = "richardson"
    #: **W167, added 2026-09-09.** Richardson preconditioned by the measured
    #: Robin coefficient ``diag(S) = alpha_star``, damped by ``1/rho(D^-1 S)``.
    #: The probe has emitted `alpha_star` since Tier 0 and nothing had ever used
    #: it as a transmission condition, which is what it is; CS-S1 measured both
    #: halves of why -- undamped it diverges at ``rho = 2.976`` while CUTTING
    #: the condition number 22.686 -> 7.345, and damped it beats plain
    #: Richardson 594 -> 172 expert calls to ``tol = 1e-6``.
    ROBIN_RICHARDSON = "robin-richardson"
    KRYLOV = "krylov"
    NEWTON_KRYLOV = "newton-krylov"
    DIRECT_SCHUR = "direct-schur"


class Levels(enum.Enum):
    ONE = "one"
    TWO_COARSE = "two(coarse)"
    TWO_PROBED_SCHUR = "two(probed-schur)"


#: The nine admissibility rules, by number, so a refusal can cite one.
RULES: dict[str, str] = {
    "R1": "the interface is as weak as its weakest agent: "
          "rung(transmission) <= min_i rung(bc_channel_i)",
    "R2": "probing is the exception that makes R1 survivable: probed-DtN requires only "
          "bc_channel >= dirichlet, because the higher rung is built outside the expert",
    "R3": "a window W > 1 requires bc_time_varying on every agent at that interface; a "
          "ring held constant across the window is a W = 1 scheme wearing a longer name",
    "R4": "the exchange interval cannot go below max_i dt_native. The window is the one "
          "axis a frozen expert can move along: W can be increased, never decreased",
    "R5": "multiplicative ordering is forbidden when any composition-layer guarantee "
          "depends on equivalent treatment of agents; a direct Schur solve satisfies this "
          "automatically, being order-free",
    "R6": "newton-krylov and direct-schur require a nonzero transmission operator; "
          "otherwise the interface system is not ill-conditioned but empty",
    "R7": "the probing method follows differentiable: jvp -> exact JVP; deterministic and "
          "smooth -> finite differences with the step set by the reproducibility floor; "
          "otherwise regularized regression over more probes than modes",
    "R8": "two-level via a classical coarse solve requires an agent that can run at a "
          "different window size without changing its regime; via a probed Schur "
          "complement it requires nothing extra, since every call stays at the native size",
    "R9": "under multirate, conservation is enforced on the TIME-INTEGRATED flux over the "
          "macro-step, never pointwise in time. Pointwise matching across different clocks "
          "is not conservative and the residual is computed correctly and means nothing",
    "R10": "an agent that embeds a global elliptic sub-solve is not decomposable: the "
           "solve is global over whatever domain it runs on, so cutting the domain cuts "
           "the operator and the error is flat in distance from the cut and flat in dt",
    "R2b": "probed-DtN requires an implicitly stepped agent. sum_i Lambda_i lambda = chi "
           "is the interface condition of a boundary-value problem, and an explicit "
           "macro-step poses none",
    "R10b": "when the composition layer owns an agent's elliptic part (R10), it must "
            "apply it at the agent's OWN sub-step cadence. A different cadence makes "
            "the composed step a different splitting of the same equations, and the "
            "difference is charged to tau while being a property of the harness",
    "R11": "a partition of unity must be CONVEX -- chi >= 0 as well as summing to one. "
           "Convexity is necessary and sufficient for the blend to be at least as "
           "accurate as the local solves it blends (L6/C1), and nothing else detects "
           "its failure: the identity still holds and the blend defect still reads as "
           "passing",
}


@dataclass
class Scheme:
    """Sigma. Every field carries the rule that set it, so nothing is a preference."""

    decomposition: Decomposition
    transmission: Transmission
    ordering: Ordering
    accelerator: Accelerator
    levels: Levels
    window: int
    eps_tol: float | None

    overlap: float | None = None
    exchange_interval: float | None = None
    multirate: bool = False
    flux_matching: str = "pointwise"      # "pointwise" | "time-integrated"
    why: dict[str, str] = field(default_factory=dict)
    defaulted: tuple[str, ...] = ()       # axes set by a default rather than chosen

    def bound_term(self, axis: str) -> str:
        """Which term of the master bound an axis controls."""
        return {
            "decomposition": "rho, and whether cross-points exist",
            "transmission": "sigma",
            "ordering": "rho, and composability with other guarantees",
            "accelerator": "gamma",
            "levels": "rho at large N",
            "window": "the gamma count, and the method's share of L",
            "eps_tol": "gamma",
        }[axis]

    def as_dict(self) -> dict[str, Any]:
        return {
            "D_decomposition": self.decomposition.value,
            "Lambda_transmission": self.transmission.value,
            "O_ordering": self.ordering.value,
            "K_accelerator": self.accelerator.value,
            "C_levels": self.levels.value,
            "W_window": self.window,
            "eps_tol": self.eps_tol,
            "overlap": self.overlap,
            "exchange_interval": self.exchange_interval,
            "multirate": self.multirate,
            "flux_matching": self.flux_matching,
            "defaulted": list(self.defaulted),
            "why": dict(self.why),
        }

    def report(self) -> str:
        lines = ["scheme Sigma"]
        for k, v in self.as_dict().items():
            if k in ("why", "defaulted"):
                continue
            reason = self.why.get(k, "")
            flag = " (DEFAULTED)" if k in self.defaulted else ""
            lines.append(f"  {k:<22} {v}{flag}" + (f"   <- {reason}" if reason else ""))
        return "\n".join(lines)


@dataclass
class Budget:
    """What the compile is allowed to spend, and what it must not spend it on."""

    allow_probe: bool = True
    max_probe_solves: int | None = None
    allow_direct_schur: bool = True
    target_rung: Transmission | None = None
    macro_dt: float | None = None
    delta_tol: float | None = None
