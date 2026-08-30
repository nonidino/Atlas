"""L4 -- constructing the transmission operator by probing.

probed-dtn-coupling, re-expressed on the common interface space of
interface-transfer-theory §3.

The Dirichlet-to-Neumann map is *defined* by what it takes and what it returns,
so an expert that accepts a trace and returns a flux already implements it --
you have simply never asked it for the answer in that form.  Probing builds the
operator from outside, in the composition layer, on data the expert has already
returned:

    S_i psi_k = d_n ( E_i[psi_k] - E_i[0] )     k = 1..m,   m+1 solves per block

and on the common space that reads

    Lambda_M_i = P_i^* Lambda_i P_i

which is a Galerkin compression.  Three properties follow and each discharges
something: positive-realness survives by congruence (which is precisely why a
non-adjoint pair silently disables the passivity theorem), the seam operator is
the sum over sides, and beta is read off the assembled matrix rather than proved.

The probe never mentions a governing equation.  That is why it survives the
failure of E3 at a multiphysics seam while agent infidelity tau does not: the
transmission layer is robust to E3, the error *attribution* is not.

Everything in §4 of probed-dtn-coupling comes from one assembly, and this module
emits all of it: beta, kappa, the null-space dimension, the passivity defect and
its eigenvector, the per-mode optimal Robin coefficient, and the composability
index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .capability import ExpertCapabilities, PortDecl
from .transfer import InterfaceSpace, Prolongation, SeamTransfer


#: **W68/W71, measured 2026-08-30.**  Below this identity defect a probed block
#: carries no resolvable operator content and every spectral statistic taken on
#: it -- kappa, the asymmetry, `poseidon.elliptic_signature` -- is reading a
#: boundary coefficient rather than the expert's operator.
#:
#: Calibrated on four blocks measured on the same instrument, all four with
#: ||S|| stable to 1.0003x across eps from 1e-2 to 1e-5, and all four at the
#: declared probe base (W74 -- the thermal pair moved when that was fixed):
#:
#:     window_ns as-built  block W00   omega = 0.5268   elliptic EMBEDDED
#:     window_ns split-step block W00  omega = 0.0467   elliptic EXPOSED
#:     thermal_seam        block shell omega = 1.97e-5  elliptic EMBEDDED
#:     thermal_seam        block gas   omega = 1.91e-5  elliptic NONE
#:
#: The floor sits between 0.0467 and 1.97e-5 -- a 2370x separation, with 47x of
#: margin above and 51x below -- and the probe's own noise floor for omega is
#: 1.9e-6, so the thin blocks are resolved small numbers and not noise.
OPERATOR_CONTENT_FLOOR = 1.0e-3


def operator_content(S: np.ndarray) -> float:
    """omega = ||S - cI|| / ||S||, c = tr(S)/n:  how much of the block is NOT a scalar.

    **What it is for.**  A probed block is supposed to be the expert's DtN map.
    It can instead be the expert's *boundary condition*: if the interface
    response is dominated by a film coefficient, ``S = h I`` exactly, and every
    downstream statistic is then a property of ``h`` and of nothing else.  There
    is no way to see that in ``beta``, ``kappa``, the null count or the passivity
    defect, because a multiple of the identity is a perfectly healthy operator by
    all four measures -- it is simply not the operator anybody meant to measure.

    **The measurement, 2026-08-30.**  `thermal_seam`'s seam assembles to
    ``4.8068 I`` to five digits, from ``+5.0001 I`` (shell) and ``-0.1933 I``
    (gas): the difference of two film coefficients, with the solvers' transverse
    physics entering only at the 2.3e-4 level.  In the spatial domain the gas
    block is *exactly* diagonal -- a delta at one seam cell moves that cell's
    flux by 9.3e-2 and every other cell by bit-zero -- and the shell block has a
    real 5-cell exponentially decaying tail, 0.471 at the pole and 1.6e-3 one
    cell out.  So the shell's operator content is not absent; it is 300x below
    the boundary condition sitting on top of it.

    **The closed form.**  Sweeping the exchange interval over 7 decades and the
    film coefficient over 7 more, on the shell block alone,

        omega ~= C * Bi * (k_max * ell)^2,   Bi = h d / k,   ell = sqrt(alpha dt)

    with C measured in [0.120, 0.307] over Pi from 1.0e-5 to 1.0, and omega
    saturating at 0.68 past Pi ~ 1 where the expansion stops holding.  The two
    groups are the two ways the operator can hide: **Bi** small means the
    interface resistance swamps the internal one, **k_max ell** small means the
    exchange interval is too short for heat to travel one interface wavelength.
    Either one alone is enough, which is why sweeping only ``dt`` -- as W60's
    calibration did, over 10,000x -- moved nothing: it held Bi = 0.033 fixed.

    Frobenius, not spectral: the question is how much of the whole matrix is
    off-scalar, not how much the largest direction is.
    """
    n = S.shape[0]
    if S.size == 0 or n != S.shape[1]:
        return float("nan")
    total = float(np.linalg.norm(S))
    if total == 0.0:
        return 0.0
    c = float(np.trace(S)) / n
    return float(np.linalg.norm(S - c * np.eye(n)) / total)


#: **W75, measured 2026-08-29.**  A response whose nonzero support covers at
#: least this fraction of the seam is read as having an INFINITE domain of
#: dependence -- the defining property of `EllipticSubsolve.EMBEDDED`.
#:
#: The mechanism needs no calibration and that is the point: one implicit
#: macro-step inverts ``(M/dt + K)``, and the inverse of a sparse SPD matrix is
#: DENSE, so every seam cell responds to every seam cell.  An explicit
#: sub-stepped march has a finite domain of dependence and its response is
#: EXACTLY zero -- not small -- beyond ``radius * substeps``.  Five known points,
#: two of them the same shell under two solvers with everything else held fixed:
#:
#:     shell implicit,   dt = 0.05   nonzero  48/48   1.000   declared embedded
#:     shell explicit,   dt = 0.05   nonzero   9/48   0.188   declared none
#:     gas explicit                  nonzero   1/48   0.021   declared none
#:     window_ns as-built            nonzero 136/138  0.986   declared embedded
#:     window_ns split-step          nonzero  29/138  0.210   declared exposed
#:
#: 0.986 against 0.210 at the nominal cadence.  **The margin closes as the
#: exchange interval grows** -- the same shell run explicitly at dt = 1 s reaches
#: 0.81, because an explicit march with enough sub-steps to be stable eventually
#: covers the seam too.  So this gate wants a SHORT interval, which is exactly
#: where `poseidon.elliptic_signature` has no signal, and the two instruments are
#: complements rather than alternatives (W75).
SUPPORT_GLOBAL_FRACTION = 0.9


@dataclass
class SupportReach:
    """How far a delta on the seam is felt.  **W75's gate, 2026-08-29.**

    Two solves, no thresholds on any spectral quantity, and it measures the
    property `EllipticSubsolve` is *defined* by -- "a solve with an INFINITE
    domain of dependence" -- rather than a correlate of it.

    **Why this exists and `elliptic_signature` was not enough.**  W68 gave that
    statistic a precondition, and W75 asked whether clearing the precondition
    would let it decide the field.  It does not.  Run against the same shell
    under two solvers, at four cadences, once ``omega`` is above the floor:

        dt = 1     implicit kappa 1.0214   explicit kappa 1.0210
        dt = 10    implicit kappa 1.2311   explicit kappa 1.2649
        dt = 100   implicit kappa 3.1966   explicit kappa 9.6688

    identical at dt = 1 and **ranked backwards** at dt = 100, with the asymmetry
    flat at 5e-6 throughout.  The spectral route does not decide this field at
    any cadence.  The spatial one decides it at the first.

    **The one way it fails, and it fails conservatively in the wrong direction.**
    A genuinely dense response whose tail UNDERFLOWS to zero far from the poke
    reads as compact.  Measured here the far tail is 1.7e-11 on 48 cells, eleven
    orders above underflow, but a seam long enough for ``exp(-d/ell)`` to reach
    1e-308 would be misread -- as ``none``, which switches R10 off.  So the
    profile is kept, not just the count, and a reach that is large but not
    saturating should be read rather than thresholded.
    """

    reach: int
    n: int
    nonzero: int
    peak: float
    profile: list[float] = field(default_factory=list)

    @property
    def exact_zeros(self) -> int:
        return self.n - self.nonzero

    @property
    def fraction(self) -> float:
        return self.nonzero / self.n if self.n else float("nan")

    @property
    def is_global(self) -> bool:
        """Infinite domain of dependence: the response has no compact support."""
        return self.fraction >= SUPPORT_GLOBAL_FRACTION

    @property
    def consistent_with(self) -> tuple[str, ...]:
        """The `EllipticSubsolve` values this reach corroborates.

        It cannot split ``exposed`` from ``none``: both have removed the global
        solve from the agent's own step, which is the whole of what the reach
        can see.  That split is not one R10 or L2 needs -- both branches leave
        the agent decomposable -- so the measurement decides exactly the
        distinction the rules consume, and no more.
        """
        return ("embedded",) if self.is_global else ("exposed", "none")

    def as_dict(self) -> dict[str, Any]:
        return {"reach": self.reach, "n": self.n, "nonzero": self.nonzero,
                "exact_zeros": self.exact_zeros, "fraction": self.fraction,
                "peak": self.peak, "is_global": self.is_global,
                "consistent_with": list(self.consistent_with),
                "profile": list(self.profile)}


def support_reach(
    respond: BoundaryResponse,
    port_name: str,
    base: np.ndarray,
    amplitude: float = 1.0,
    profile_cells: int = 8,
) -> SupportReach:
    """Poke one seam cell and see which cells move.  Two solves.

    The poke is a delta rather than a smooth mode on purpose: a Fourier mode is
    global by construction, so probing with one cannot distinguish a global
    operator from a local one.  That is the whole reason the spectral route
    could not settle `elliptic_subsolve`, restated as a fact about the basis
    instead of about the statistic.
    """
    base = np.asarray(base, dtype=float).ravel()
    n = base.size
    f0 = np.asarray(respond(port_name, base), dtype=float).ravel()
    j0 = n // 2
    d = np.zeros(n)
    d[j0] = float(amplitude)
    g = np.abs(np.asarray(respond(port_name, base + d), dtype=float).ravel() - f0)
    nz = np.nonzero(g)[0]
    peak = float(g.max()) if g.size else 0.0
    reach = int(np.max(np.abs(nz - j0))) if nz.size else 0
    prof = [float(g[j0 + k]) for k in range(min(profile_cells, n - j0))]
    return SupportReach(reach=reach, n=n, nonzero=int(nz.size), peak=peak, profile=prof)


#: **W74's class, measured 2026-08-29.**  Above this relative change the probed
#: block depends on WHERE it was linearized, so ``beta`` is a local number and
#: the base is no longer free.  Below it the response is affine in the trace and
#: the base provably cannot matter -- the one check in this vault whose passing
#: PROMOTES rather than merely failing to contradict.
#:
#: Calibrated on an exactly affine fixture (`capability.linear_response`, which
#: returns 0 to machine precision) against the two real experts, per 1% of base
#: moved: gas 6.1e-3, shell 1.4e-2.  Three orders of margin either side.
BASE_SENSITIVITY_FLOOR = 1.0e-6


def base_sensitivity(
    respond: BoundaryResponse,
    port_name: str,
    base: np.ndarray,
    directions: np.ndarray,
    step: float,
    shift: float = 0.01,
) -> dict[str, float]:
    """Does the probed block depend on the point it was linearized about?

    **W74 closed the instance; this closes the class.**  The instance was that
    the base was hard-wired to the zero trace, which on a port whose effort is a
    temperature in kelvin is 0 K.  The class is that **nothing measures whether
    the choice matters at all**, and it is two extra block probes to find out.

    The shift is relative when the base is nonzero and absolute in units of the
    probe step when it is zero, because a zero base carries no scale of its own
    -- which is itself worth noticing, since it means an undeclared base cannot
    even be interrogated without inventing one.
    """
    base = np.asarray(base, dtype=float).ravel()
    scale = float(np.linalg.norm(base)) / max(np.sqrt(base.size), 1.0)
    delta = shift * scale if scale > 0.0 else shift * 100.0 * step
    moved = base + delta

    def block(b: np.ndarray) -> np.ndarray:
        zero = np.asarray(respond(port_name, b), dtype=float)
        cols = [(np.asarray(respond(port_name, b + step * directions[:, k]), dtype=float)
                 - zero) / step for k in range(directions.shape[1])]
        return np.column_stack(cols)

    S0 = block(base)
    S1 = block(moved)
    n0 = float(np.linalg.norm(S0))
    rel = float(np.linalg.norm(S1 - S0) / n0) if n0 > 0 else float("nan")
    return {"relative_change": rel, "shift": float(delta), "base_scale": scale,
            "affine": bool(rel == rel and rel < BASE_SENSITIVITY_FLOOR)}


def base_disagreement(bases_M: dict[str, np.ndarray]) -> dict[str, Any]:
    """Are the two sides of one seam linearized about the SAME interface state?

    **The W74 class defect, and it is a property of the field rather than of any
    record that fills it in.**  ``probe_base`` was put on `ExpertCapabilities`,
    which is one per expert -- so each side of a seam names its own.  The
    interface variable is ONE variable: both sides see the same trace, and

        Lambda_M = sum_i P_i^* Lambda_i P_i

    is a sum of Jacobians, which is a Jacobian only if every term was taken at
    the same point.  Nothing checked that, and on `thermal_seam` the two records
    are honest, independently correct, and **500 K apart**: the gas linearizes at
    ``T_WALL_0 = 400`` (the wall temperature it sees) and the shell at
    ``T_HOT = 900`` (the gas temperature it sees).

    The consequence is not a shifted number, it is a number that is not on the
    curve.  Sweeping a COMMON base over the physically admissible interval --
    the interface temperature is trapped between the two reservoirs, 250 K and
    900 K -- gives ``beta`` in [0.4200, 1.7689], and the mismatched-base probe
    reports **0.3757, below the whole range**.  At the state the coupled step
    actually reaches, ``lambda* = 371.97 K`` where the two fluxes balance,
    ``beta = 1.2381``: 3.3x the certificate's number, after W74 had already
    moved it 12.6x.

    So the field is on the wrong object.  A base belongs to a SEAM.
    """
    ids = sorted(bases_M)
    out: dict[str, Any] = {"agents": ids, "consistent": True, "spread": 0.0}
    if len(ids) < 2:
        return out
    ref = np.asarray(bases_M[ids[0]], dtype=float).ravel()
    worst = 0.0
    scale = max(float(np.linalg.norm(ref)), 1e-300)
    for a in ids[1:]:
        v = np.asarray(bases_M[a], dtype=float).ravel()
        if v.size != ref.size:
            out["consistent"] = False
            out["spread"] = float("inf")
            out["note"] = f"{a} declares a base of length {v.size}, {ids[0]} of {ref.size}"
            return out
        worst = max(worst, float(np.linalg.norm(v - ref)))
    out["spread"] = worst
    out["relative_spread"] = worst / scale
    out["consistent"] = bool(worst <= 1e-9 * scale)
    return out


def probe_base(caps: ExpertCapabilities, port_name: str, n: int) -> np.ndarray:
    """The trace the probe linearizes about.  **W74, 2026-08-30.**

    Zeros unless the record declares a ``probe_base``, which is what every
    pre-2026-08-30 record does, so every measurement taken before this existed is
    reproduced bit-for-bit.
    """
    fn = getattr(caps, "probe_base", None)
    if fn is None:
        return np.zeros(n)
    base = np.asarray(fn(port_name), dtype=float).ravel()
    if base.size != n:
        raise ProbeError(
            f"{caps.expert_id}: probe_base({port_name!r}) returned {base.size} values, "
            f"expected {n} -- the base must live in V, beside the trace"
        )
    return base


class ProbeError(RuntimeError):
    """The probe could not be run as configured."""


@dataclass
class ProbeBudget:
    """What the probe is allowed to spend, and how it is allowed to spend it."""

    max_modes: int | None = None          # cap on dim M actually probed
    fd_step: float | None = None          # epsilon; default is set from the noise floor
    regression_oversample: float = 2.0    # m' / m for the opaque-expert fallback
    regression_ridge: float = 1e-8        # the Tikhonov parameter, stated not hidden
    seed: int = 0
    allow_assembly: bool = True           # False forces matrix-free, disabling §4 diagnostics


@dataclass
class ProbedBlock:
    """One agent's contribution to a seam operator, plus everything it certifies."""

    agent_id: str
    seam_id: str
    S: np.ndarray                     # the block on M:  P^* Lambda P
    route: str
    n_solves: int
    dim_M: int

    beta: float | None = None         # sigma_min on the constrained subspace
    kappa: float | None = None
    null_dim: int | None = None
    null_tol: float | None = None
    passivity_defect: float | None = None      # pi_i, zeroed below the noise floor
    passivity_lambda_min: float | None = None  # the raw eigenvalue, never clipped
    passivity_tol: float | None = None
    passivity_eigvec: np.ndarray | None = None # which interface mode is amplified
    alpha_star: np.ndarray | None = None       # per-mode optimal Robin coefficient
    #: W68/W71. ||S - cI||/||S||: how much of this block is NOT a scalar. Below
    #: OPERATOR_CONTENT_FLOOR the block is a boundary coefficient and every
    #: spectral statistic taken on it is about that coefficient.
    operator_content: float | None = None
    Xi: float | None = None                    # composability index
    #: W76. ||S_i||_2 / ||S||_2 on the assembled seam: this agent's share of the
    #: interface response. It bounds what ANY test of this agent taken on the
    #: assembled operator can see -- see `SeamOperator.one_sided`.
    share: float | None = None
    #: W74's class. The trace this block was linearized about, reduced to M, so
    #: the two sides of a seam can be compared for consistency.
    base_M: np.ndarray | None = None
    #: The same base in the agent's own V, summarized: (mean, min, max) in the
    #: port's physical units. M coefficients are the right space to compare in
    #: and the wrong one to READ -- a 400 K uniform trace reduces to 11.18 in a
    #: measure-weighted Fourier basis, which tells a reader nothing.
    base_summary: tuple[float, float, float] | None = None
    probe_state: str = "unspecified"
    notes: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Lambda == 0: the interface problem is not ill-conditioned, it is empty.

        Every trace is equally consistent, because no agent's output depends on
        any agent's input.  A run in that configuration is a legitimate object --
        an ensemble of independent local solves -- but it is not a composition.
        """
        return bool(np.allclose(self.S, 0.0))

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent_id,
            "seam": self.seam_id,
            "route": self.route,
            "n_solves": self.n_solves,
            "dim_M": self.dim_M,
            "beta": self.beta,
            "kappa": self.kappa,
            "null_dim": self.null_dim,
            "passivity_defect": self.passivity_defect,
            "passivity_lambda_min": self.passivity_lambda_min,
            "Xi": self.Xi,
            "alpha_star": None if self.alpha_star is None else self.alpha_star.tolist(),
            "operator_content": self.operator_content,
            "share": self.share,
            "empty": self.is_empty,
            "probe_state": self.probe_state,
            "notes": list(self.notes),
        }


@dataclass
class SeamOperator:
    """The assembled seam operator, Lambda_M = sum_i P_i^* Lambda_i P_i."""

    seam_id: str
    S: np.ndarray
    blocks: dict[str, ProbedBlock]
    dim_M: int

    beta: float | None = None
    kappa: float | None = None
    null_dim: int | None = None
    expected_null_dim: int | None = None
    passivity_defect: float | None = None
    passivity_lambda_min: float | None = None
    passivity_eigvec: np.ndarray | None = None
    alpha_star: np.ndarray | None = None
    operator_content: float | None = None
    cut_score: float | None = None
    conforming: bool = True
    #: W74's class. What `base_disagreement` found across this seam's two sides.
    base_check: dict[str, Any] | None = None
    probe_state: str = "unspecified"
    notes: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return bool(np.allclose(self.S, 0.0))

    @property
    def mode_shares(self) -> np.ndarray | None:
        """Per interface mode, the largest side's share of the response.

        Read off ``alpha_star``, which in a Fourier interface basis IS the
        measured symbol mode by mode.  **W76: this is the first rule in the
        vault that reads that field**, which the probe has emitted since Tier 0
        and no layer had ever consumed.
        """
        cols = [np.abs(b.alpha_star) for b in self.blocks.values()
                if b.alpha_star is not None]
        if len(cols) < 2:
            return None
        A = np.vstack(cols)
        tot = A.sum(axis=0)
        return A.max(axis=0) / np.maximum(tot, 1e-300)

    @property
    def one_sided(self) -> float | None:
        """The smallest block's share of the assembled operator.

        **W76's derivable consequence, measured 2026-08-29.**  A substitution
        certificate compares two assembled operators, so the perturbation it can
        possibly see when agent ``i`` is swapped is bounded by that agent's own
        block:

            ||Delta|| = ||S_i_new - S_i_old||  <=  ||S_i_old|| + ||S_i_new||

        and `composition.SubstitutionCertificate` passes whenever
        ``||Delta|| < beta - beta_min``.  So if a block's norm is below that
        margin, **every** replacement of that agent passes, whatever the
        replacement does.  The test is not weak there, it is blind.

        Measured on `window_ns` split-step, whose two vertical seams are
        one-sided: on ``sx0`` the two blocks are 94.1% and 16.7% of the seam,
        and replacing the 16.7% agent with an expert that IGNORES its boundary
        data entirely is certified ADMIT at every ``beta_min`` up to 0.20.  The
        same swap on the balanced seam ``sy0`` (50.6% / 49.5%) is REFUSED from
        ``beta_min = 0.10`` and moves beta by 46%.  The blind spot is the block
        norm, and it is a number the probe already had.
        """
        vals = [b.share for b in self.blocks.values() if b.share is not None]
        return min(vals) if vals else None

    @property
    def excess_null_directions(self) -> int | None:
        """Any excess null direction is a defect -- the cross-point detector.

        Stated as an equality the null-count identity is a conjecture; stated as
        'any excess null direction is a defect' it is safe, and it is the form
        implemented here.
        """
        if self.null_dim is None or self.expected_null_dim is None:
            return None
        return max(0, self.null_dim - self.expected_null_dim)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seam": self.seam_id,
            "dim_M": self.dim_M,
            "beta": self.beta,
            "kappa": self.kappa,
            "null_dim": self.null_dim,
            "expected_null_dim": self.expected_null_dim,
            "excess_null_directions": self.excess_null_directions,
            "passivity_defect": self.passivity_defect,
            # probed-dtn-coupling 4.2 requires the raw eigenvalue beside the
            # floored defect so the clipping is visible rather than silent. The
            # block emitted it and the seam did not, which meant the assembled
            # operator -- the one the scheme is built on -- reported only the
            # clipped number. Found by W2's first real assembly, 2026-08-27.
            "passivity_lambda_min": self.passivity_lambda_min,
            "operator_content": self.operator_content,
            "cut_score": self.cut_score,
            "conforming": self.conforming,
            "empty": self.is_empty,
            "one_sided": self.one_sided,
            "mode_shares": None if self.mode_shares is None else self.mode_shares.tolist(),
            "base_check": self.base_check,
            "declared_probe_state": self.probe_state,
            "derived_probe_state": self.derived_probe_state,
            "probe_state_agrees": self.probe_state_agrees,
            "blocks": {k: v.as_dict() for k, v in self.blocks.items()},
            "notes": list(self.notes),
        }

    # -- W77: the state a certificate is valid at, derived rather than declared -

    @property
    def derived_probe_state(self) -> str:
        """What the probe can SAY about its own state, from the base it used.

        **W77, closed 2026-08-29.**  ``probe_state`` is a free-form string the
        caller hands in, and W74 is the proof it can be wrong while everything
        else is right: it read ``"duct, T_hot=900 K, T_wall=400 K"`` on every
        `thermal_seam` certificate ever emitted while the probe was linearizing
        at **0 K**.  The architecture's own rule -- *a certificate is valid at a
        state, never globally* -- was carried by a label nothing checked.

        It never needed to be declared.  The probe knows the base it used, and
        the base IS the state the linearization is about, so this is derived
        from the measurement and cannot disagree with it.  The declared string
        is kept beside it as documentation and `probe_state_agrees` says whether
        the two are even about the same seam.

        Unlike the three unverifiable declarations of W69 this ADDS no fourth:
        it removes a candidate, because the field it replaces was never checked
        against anything.
        """
        parts = []
        for aid, b in sorted(self.blocks.items()):
            if b.base_summary is None:
                parts.append(f"{aid}@undeclared")
                continue
            mean, lo, hi = b.base_summary
            parts.append(f"{aid}@{mean:.6g}" if lo == hi
                         else f"{aid}@mean {mean:.6g} in [{lo:.6g}, {hi:.6g}]")
        chk = self.base_check or {}
        tag = "" if chk.get("consistent", True) else \
            f" INCONSISTENT(spread {chk.get('spread', float('nan')):.4g})"
        return "; ".join(parts) + tag

    @property
    def probe_state_agrees(self) -> bool | None:
        """Whether the declared label is consistent with the derived one.

        Only one thing is decidable without re-running the experts: a declared
        state cannot be right if the bases it names disagree across the seam.
        So this reports ``False`` exactly when the seam is internally
        inconsistent and the label claims a single state anyway -- which is the
        `thermal_seam` case, and it is the check that would have caught W74.
        """
        if self.probe_state in ("unspecified", ""):
            return None
        chk = self.base_check or {}
        if not chk:
            return None
        return bool(chk.get("consistent", True))


# ---------------------------------------------------------------------------
# the three probe routes -- R7: the probing method follows ``differentiable``
# ---------------------------------------------------------------------------


def _columns_jvp(
    caps: ExpertCapabilities,
    port_name: str,
    directions: np.ndarray,
    base: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Exact forward-mode JVP: no epsilon, no reproducibility contamination."""
    jvp = caps.boundary_response_jvp
    if jvp is None:
        raise ProbeError(
            f"{caps.expert_id} declares differentiable={caps.differentiable.value} but supplies "
            "no boundary_response_jvp; the declaration and the record disagree"
        )
    base = probe_base(caps, port_name, directions.shape[0]) if base is None else base
    cols = [np.asarray(jvp(port_name, base, directions[:, k]), dtype=float)
            for k in range(directions.shape[1])]
    return np.column_stack(cols), directions.shape[1]


def _columns_finite_difference(
    caps: ExpertCapabilities,
    port_name: str,
    directions: np.ndarray,
    step: float,
    base: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Finite differences with the base probe subtracted.

    The base probe is what removes every trace-independent bias in the agent.
    A systematic, state-dependent model error cancels in the numerator; what does
    not cancel is genuine non-reproducibility, and that is the quantity the step
    is set against.

    **W74, 2026-08-30 -- the base used to be hard-wired to zero.**  For an affine
    expert that is exactly right and the choice is free: the derivative is the
    same everywhere, so the zero trace removes the bias and costs nothing.  For a
    nonlinear one the derivative is taken WHERE the base is, and the zero trace
    of a port whose variable has an absolute origin is not a state the expert has
    ever been in.  Measured on `thermal_seam`, whose THERM effort is a
    temperature in kelvin: the zero trace is 0 K, and against the expert's own
    operating point ``beta`` moves 4.8062 -> 0.3807, a factor of **12.6**.  See
    `capability.ExpertCapabilities.probe_base`.
    """
    respond = caps.boundary_response
    if respond is None:
        raise ProbeError(f"{caps.expert_id} supplies no boundary_response; nothing to probe")
    n = directions.shape[0]
    base = probe_base(caps, port_name, n) if base is None else base
    zero = np.asarray(respond(port_name, base), dtype=float)
    cols = []
    for k in range(directions.shape[1]):
        plus = np.asarray(respond(port_name, base + step * directions[:, k]), dtype=float)
        cols.append((plus - zero) / step)
    return np.column_stack(cols), directions.shape[1] + 1


def _columns_regression(
    caps: ExpertCapabilities,
    port_name: str,
    directions: np.ndarray,
    budget: ProbeBudget,
) -> tuple[np.ndarray, int]:
    """Regularized least squares over m' > m random probes, for an opaque expert.

    Reproducibility error averages down as 1/sqrt(m') instead of amplifying as
    1/epsilon, at the cost of more calls and a Tikhonov parameter that has to be
    stated -- so it is stated, in the budget, rather than buried here.
    """
    respond = caps.boundary_response
    if respond is None:
        raise ProbeError(f"{caps.expert_id} supplies no boundary_response; nothing to probe")
    n, m = directions.shape
    m_prime = max(m + 1, int(np.ceil(budget.regression_oversample * m)))
    rng = np.random.default_rng(budget.seed)
    zero = np.asarray(respond(port_name, np.zeros(n)), dtype=float)
    X = directions @ rng.standard_normal((m, m_prime))
    Y = np.column_stack(
        [np.asarray(respond(port_name, X[:, j]), dtype=float) - zero for j in range(m_prime)]
    )
    # Solve  A X = Y  for A, ridge-regularized, then restrict to the probe basis.
    G = X @ X.T + budget.regression_ridge * np.eye(n)
    A = np.linalg.solve(G.T, (X @ Y.T)).T
    return A @ directions, m_prime + 1


# ---------------------------------------------------------------------------
# the probe
# ---------------------------------------------------------------------------


def probe_block(
    caps: ExpertCapabilities,
    port: PortDecl,
    space: InterfaceSpace,
    prolongation: Prolongation,
    seam_id: str,
    budget: ProbeBudget | None = None,
    reference: ExpertCapabilities | None = None,
    probe_state: str = "unspecified",
    base_V: np.ndarray | None = None,
) -> ProbedBlock:
    """Assemble one side's block on the common interface space M.

    Imposing the prolonged mode P mu_k and reducing the returned flux with the
    forced adjoint P^* gives P^* Lambda P directly -- so a non-conforming
    interface costs no probes beyond the ones already budgeted, the multiplier
    space and the probe basis being the same space.
    """
    budget = budget or ProbeBudget()
    dim = space.dim if budget.max_modes is None else min(space.dim, budget.max_modes)
    if dim < space.dim:
        space = InterfaceSpace(seam_id=space.seam_id, dim=dim, basis=space.basis)

    modes_M = np.eye(space.dim)
    directions_V = np.column_stack(
        [prolongation.prolong(modes_M[:, k], space) for k in range(space.dim)]
    )

    route = caps.probe_route()
    step = budget.fd_step or max(1e-2, 100.0 * caps.reproducibility_floor)
    if base_V is None:
        base_V = probe_base(caps, port.name, directions_V.shape[0])
    else:
        base_V = np.asarray(base_V, dtype=float).ravel()
    if route == "jvp":
        flux_V, n_solves = _columns_jvp(caps, port.name, directions_V, base_V)
    elif route == "finite-difference":
        flux_V, n_solves = _columns_finite_difference(
            caps, port.name, directions_V, step, base_V
        )
    else:
        flux_V, n_solves = _columns_regression(caps, port.name, directions_V, budget)

    R = prolongation.adjoint(space)
    S = R @ flux_V

    block = ProbedBlock(
        agent_id=caps.expert_id,
        seam_id=seam_id,
        S=S,
        route=route,
        n_solves=n_solves,
        dim_M=space.dim,
        probe_state=probe_state,
        base_M=R @ base_V if base_V.size == R.shape[1] else None,
        base_summary=(float(base_V.mean()), float(base_V.min()), float(base_V.max()))
        if base_V.size else None,
    )
    _fill_diagnostics(block, S)

    if reference is not None:
        ref_block = probe_block(
            reference, port, space, prolongation, seam_id, budget, None, probe_state
        )
        denom = float(np.linalg.norm(ref_block.S, 2))
        block.Xi = float(np.linalg.norm(S, 2) / denom) if denom > 0 else None
        block.notes.append(
            "Xi is the composability index ||Lambda_expert|| / ||Lambda_ref||: how much of "
            "the true boundary response the agent actually reproduces."
        )
    if block.is_empty:
        block.Xi = 0.0
        block.notes.append(
            "Lambda == 0. The agent's output does not depend on imposed boundary data, so "
            "the interface problem is empty rather than ill-conditioned and every trace is "
            "equally consistent."
        )
    return block


def _fill_diagnostics(block: ProbedBlock, S: np.ndarray, tol: float | None = None) -> None:
    """Everything §4 of probed-dtn-coupling asks for, from the one assembly."""
    if S.size == 0:
        return
    svals = np.linalg.svd(S, compute_uv=False)
    smax = float(svals[0]) if svals.size else 0.0
    scale = max(smax, 1e-300)
    tol = tol if tol is not None else max(S.shape) * np.finfo(float).eps * scale
    block.null_tol = float(tol)
    block.null_dim = int(np.sum(svals <= tol))

    nonzero = svals[svals > tol]
    if nonzero.size:
        # beta is sigma_min on the constrained subspace: the null directions are
        # constrained out, so the smallest surviving singular value is the one
        # that sits in the denominator of both sigma and L.
        block.beta = float(nonzero[-1])
        block.kappa = float(nonzero[0] / nonzero[-1])
    else:
        block.beta = 0.0
        block.kappa = float("inf")

    sym = 0.5 * (S + S.T)
    evals, evecs = np.linalg.eigh(sym)
    lam_min = float(evals[0])
    # A negative eigenvalue at the level of arithmetic noise is not a physical
    # defect, and reporting it as one would put a passivity failure on every
    # symmetric operator the probe ever assembles. The raw eigenvalue is kept
    # alongside so the clipping is visible rather than silent.
    pass_tol = max(S.shape) * np.finfo(float).eps * scale
    block.passivity_lambda_min = lam_min
    block.passivity_tol = float(pass_tol)
    block.passivity_defect = abs(lam_min) if lam_min < -pass_tol else 0.0
    block.passivity_eigvec = evecs[:, 0].copy()

    if S.shape[0] == S.shape[1]:
        block.operator_content = operator_content(S)
        # In a Fourier interface basis the diagonal IS the measured symbol, mode
        # by mode -- so the optimal Robin coefficient is read off rather than
        # derived, which for a frozen expert is the only available route.
        block.alpha_star = np.diag(S).copy()


def assemble_seam(
    graph: Any,
    connection: Any,
    transfer: SeamTransfer,
    budget: ProbeBudget | None = None,
    references: dict[str, ExpertCapabilities] | None = None,
    expected_null_dim: int | None = None,
    probe_state: str = "unspecified",
    seam_base: np.ndarray | None = None,
) -> SeamOperator:
    """Lambda_M = sum_i P_i^* Lambda_i P_i, with every diagnostic the sum affords.

    ``seam_base`` is the interface state on M that BOTH sides are linearized
    about.  Leaving it None keeps each side on its own declared ``probe_base``,
    which is what every record written before 2026-08-29 does and what makes
    those measurements reproduce bit-for-bit -- and `base_disagreement` then
    reports whether the two sides actually agreed, which on `thermal_seam` they
    do not, by 500 K.  See `base_disagreement` for why a base belongs to a seam.
    """
    budget = budget or ProbeBudget()
    references = references or {}
    blocks: dict[str, ProbedBlock] = {}
    for agent_id, port_name in (connection.a, connection.b):
        caps = graph.agent(agent_id).capabilities
        port = caps.port(port_name)
        prolong = transfer.prolongations[agent_id]
        base_V = None
        if seam_base is not None:
            base_V = prolong.prolong(np.asarray(seam_base, dtype=float).ravel(),
                                     transfer.space)
        blocks[agent_id] = probe_block(
            caps,
            port,
            transfer.space,
            prolong,
            connection.seam_id,
            budget,
            references.get(agent_id),
            probe_state,
            base_V,
        )
    S = sum(b.S for b in blocks.values())
    total = float(np.linalg.norm(S, 2))
    for b in blocks.values():
        b.share = float(np.linalg.norm(b.S, 2) / total) if total > 0 else None

    op = SeamOperator(
        seam_id=connection.seam_id,
        S=S,
        blocks=blocks,
        dim_M=transfer.space.dim,
        expected_null_dim=expected_null_dim,
        conforming=transfer.conforming,
        probe_state=probe_state,
        base_check=base_disagreement(
            {a: b.base_M for a, b in blocks.items() if b.base_M is not None}
        ),
    )
    proxy = ProbedBlock(
        agent_id="<seam>", seam_id=connection.seam_id, S=S, route="assembled",
        n_solves=0, dim_M=transfer.space.dim,
    )
    _fill_diagnostics(proxy, S)
    op.beta = proxy.beta
    op.kappa = proxy.kappa
    op.null_dim = proxy.null_dim
    op.passivity_defect = proxy.passivity_defect
    op.passivity_lambda_min = proxy.passivity_lambda_min
    op.passivity_eigvec = proxy.passivity_eigvec
    op.alpha_star = proxy.alpha_star
    op.operator_content = proxy.operator_content
    op.cut_score = cut_score(S, op.beta)
    return op


def cut_score(S: np.ndarray, beta: float | None) -> float | None:
    """The RETIRED cut-placement criterion, kept as a substructuring diagnostic.

        Q = (1/beta) * ||S - diag S|| / ||S||

    **Falsified 2026-08-28 as a criterion on decompositions.  Do not rank cuts
    with it.**  The derived criterion is `cut_defect_bound` below (L2/C2); this
    stays because the quantity is real, its scope is now stated, and deleting the
    thing a finding is about would leave the finding unreproducible.

    Three measurements, in `scripts/w16_cut_policy.py`:

    1. **It is not a function of the decomposition.**  Replacing the declared
       prolongation ``P`` by ``P U`` for orthogonal ``U`` declares the SAME
       interface space -- ``range(P U) = range(P)`` -- so the scheme is
       unchanged, and ``S -> U^T S U`` leaves ``beta`` and ``||S||`` invariant
       while moving the off-diagonal mass.  On one seam of the strip model Q runs
       over ``[0.419, 0.4895]`` on random frames, ``0.5016`` on the declared
       Fourier one and **0.001388 in the eigenbasis of the symmetric part**: a
       361x orbit on one decomposition, and the docstring's own worry about the
       basis "doing unjustified work" is the whole of it.
    2. **It ranks backwards.**  Over twelve cut placements of one problem, rank
       correlation against the measured composed defect is **-0.853**, and
       following it costs 1.407x the best cut -- 92% of the available range.  The
       mechanism is in the two factors separately: the UN-normalized off-diagonal
       mass ranks **+0.853**, i.e. correctly, and both things Q does to it invert
       the sign.  Dividing by ``||S||`` removes the magnitude that matters, and
       ``1/beta`` is the SUBSTRUCTURING branch's amplifier -- the one W49 already
       scoped `master-error-bound` §4 to -- imported into an overlapping scheme
       whose sigma bound has no beta in it at all.
    3. **It is blind where the answer is not in the operator.**  With a
       cut-independent operator, Q is constant to 8e-11 while the measured defect
       spreads 2.29x over the same cuts.

    Where it is still meaningful: an interface *solve*, where ``1/beta`` really is
    the amplifier, on a fixed declared basis, comparing two cuts of the same
    geometry.  That is the substructuring branch, and no criterion there is
    derived yet -- **W57**.
    """
    if beta is None or beta <= 0.0:
        return None
    total = float(np.linalg.norm(S))
    if total == 0.0:
        return None
    off = float(np.linalg.norm(S - np.diag(np.diag(S))))
    return off / total / beta


#: L2/C2's tolerance for "this partition is convex".  The same number
#: `assembly.CONVEX_TOL` uses, restated here so the two rules cannot drift apart
#: without a test noticing -- `test_tier10_cut_policy` asserts they agree.
CUT_CONVEX_TOL = 1e-12


def restriction_defect_bound(
    local_steps: dict[str, np.ndarray],
    reference_step: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
    weighted: bool = True,
) -> dict[str, float]:
    """**L2/C2** -- the derived decomposition criterion.

    ``D_i = E_i R_i - R_i E`` is the failure of the exact one-exchange-interval
    operator to commute with restriction to subdomain ``i``.  It is what the
    phrase *"the exact operator is closest to local"* denotes, and giving it a
    definition is most of what G5/W16 was missing.

    **The identity.**  With ``sum_i R_i^T chi_i R_i = I`` -- the partition of
    unity L6 already checks -- the composed defect over one exchange interval is

        A({E_i R_i u}) - E u  =  sum_i R_i^T chi_i D_i u

    exactly, with no approximation anywhere.  Measured residual: 1.95e-12 on the
    strip model over 16 runs, 3.6e-11 and 5.1e-12 on `reference.WindowNS`'s four
    windows.

    **The bound.**  Add ``chi >= 0`` -- L6/C1, which R11 already enforces -- and
    the triangle inequality gives, at every cell,

        |A({E_i R_i u}) - E u|_j  <=  sum_i chi_ij |D_i u|_j  <=  max_i |D_i u|_j

    and therefore in every l^p norm.  **So the cut criterion is: minimize the
    chi-weighted restriction defect.**  It is a theorem on the same hypotheses
    L6/C1 already needs, which is the sense in which this derives the locality
    half of the policy rather than adopting it.

    Both forms are returned because they are not close to each other and the
    difference is the partition of unity's whole contribution.  On the real
    four-window tiling the chi-weighted bound is **tight to 0.2%** (2.6268e-07
    against a measured 2.6216e-07) while the max form is 220x loose -- chi is
    small exactly where ``D`` is large, which is what the ramp is for.

    ``weighted=False`` returns the max form, whose one advantage is that it does
    not need the weights to be right, only non-negative.
    """
    ids = list(local_steps)
    if set(ids) != set(weights) or not ids:
        raise ProbeError(
            "restriction_defect_bound needs one weight array per local solve; got "
            f"{sorted(local_steps)} against {sorted(weights)}"
        )
    per_agent: dict[str, float] = {}
    acc = None
    best = None
    for a in ids:
        chi = np.asarray(weights[a], dtype=float).reshape(-1)
        if float(chi.min()) < -CUT_CONVEX_TOL:
            raise ProbeError(
                f"weights for {a} are not non-negative (min {float(chi.min()):.3e}); "
                "L2/C2's bound rests on L6/C1 and R11 refuses this partition anyway"
            )
        d = np.abs(np.asarray(local_steps[a], dtype=float).reshape(-1)
                   - np.asarray(reference_step[a], dtype=float).reshape(-1))
        per_agent[a] = float(np.linalg.norm(d))
        w = chi * d
        m = d * (chi > 0.0)
        acc = w if acc is None else acc + w
        best = m if best is None else np.maximum(best, m)
    field = acc if weighted else best
    return {
        "cut_defect_bound": float(np.linalg.norm(field)),
        "cut_defect_bound_chi_weighted": float(np.linalg.norm(acc)),
        "cut_defect_bound_max": float(np.linalg.norm(best)),
        "per_agent": per_agent,
        "worst_agent": max(per_agent, key=per_agent.get),
    }


def neighbour_disagreement(
    local_steps: dict[str, np.ndarray],
    overlaps: dict[tuple[str, str], np.ndarray],
) -> float:
    """L2/C2's **reference-free** surrogate: what the agents disagree about.

    ``restriction_defect_bound`` needs ``E u``, the monolithic solve a
    decomposition exists precisely to avoid, so a criterion resting on it is a
    diagnostic and not a condition -- the distinction `AssemblyCertificate`'s
    ``must_satisfy`` drew and L6/C1 had to satisfy.  This is the computable half.

    On the overlap ``R_i E u = R_j E u``, so

        E_i R_i u - E_j R_j u  =  D_i - D_j     (there, exactly)

    the common mode cancels and what survives is the part of the restriction
    defect **the cut itself creates**.  It costs nothing: both local solves are
    already computed by the composed step.

    **It equals the max form of the bound exactly when the agents' contaminated
    sets are pairwise disjoint**, because then at most one ``D_i`` is nonzero at
    each cell -- and that is decidable at compile time from the ``contaminated``
    geometry the partition of unity already declares for W49's ``Pi``.  Measured
    on the strip model: the gap is **0.000e+00** at every halo satisfying
    ``halo >= 2 * stencil_radius * substeps_per_exchange`` and 6.6e-3 / 1.0e-3
    below it.  Where the condition fails it is not a licence to ignore the
    surrogate, only to stop calling it exact: on the four-window tiling, whose
    contaminated sets share 1120 cells, it still agrees to **1.00007**.

    ``overlaps[(a, b)]`` is the boolean mask, on the two agents' shared index
    space, of the cells both own with positive weight.
    """
    worst = 0.0
    for (a, b), mask in overlaps.items():
        m = np.asarray(mask, dtype=bool).reshape(-1)
        if not m.any():
            continue
        ua = np.asarray(local_steps[a], dtype=float).reshape(-1)[m]
        ub = np.asarray(local_steps[b], dtype=float).reshape(-1)[m]
        worst = max(worst, float(np.linalg.norm(ua - ub)))
    return worst


def operator_drift(S_then: np.ndarray, S_now: np.ndarray) -> float:
    """||S(t + K dt) - S(t)||: the InterfaceMotion slot's field-5 measurement.

    Emitted even on static runs, because it is one extra assembly and it prices
    the cached-S economics that the whole argument for probing rests on.
    """
    return float(np.linalg.norm(np.asarray(S_now) - np.asarray(S_then), 2))


def substitution_delta(S_old: np.ndarray, S_new: np.ndarray) -> float:
    """||Delta|| for the substitution certificate. Two probes, no rollout."""
    return float(np.linalg.norm(np.asarray(S_new) - np.asarray(S_old), 2))


@dataclass
class ReferenceTraceCheck:
    """W46's named measurement: does the REFERENCE's own trace reduce the residual?

    An interface condition is a statement the true solution satisfies.  So the
    cheapest possible falsification is to substitute the reference's own trace
    and see whether the residual falls.  It costs one extra call, it needs no
    rollout, and it is the check that would have caught 2026-08-27's finding at
    design time instead of after a scheme was built on it.

    **Measured, and it fails for flux balance under BOTH flux conventions and at
    every time step tried.** `sum_i Lambda_i lambda = chi` at a shared-layer seam
    gives, on the exact field, exactly

        F_A + F_B  =  nu (2 w_Gamma - w_A,in - w_B,in) / h  =  -nu h d2w/dn2

    -- a discrete SECOND DERIVATIVE, not a jump.  It vanishes only where the
    solution is linear across the seam, so its residual on the true solution is
    O(h) and nonzero wherever the solution is curved.  Ratios measured on
    `reference.WindowNS`: 1.002 (exposed agent), 1.028-1.040 (embedded), at
    dt = 0.05, 0.01 and 0.002.  **Time-integrating the flux does not help**, and
    the identity above says why: the defect is spatial, not temporal.
    """

    lagged: float
    reference: float
    solved: float | None = None
    note: str = ""

    @property
    def ratio(self) -> float:
        return self.reference / self.lagged if self.lagged else float("inf")

    @property
    def passes(self) -> bool:
        """The reference's trace must not make the residual WORSE."""
        return self.ratio <= 1.0

    @property
    def overshoot(self) -> float | None:
        """||solved move|| / ||true move||, when both were measured."""
        return None if self.solved is None else self.solved

    def as_dict(self) -> dict[str, Any]:
        return {"lagged": self.lagged, "reference": self.reference,
                "ratio": self.ratio, "passes": self.passes,
                "solved": self.solved, "note": self.note}


def reference_trace_check(residual, lagged_trace, reference_trace,
                          solved_trace=None, note: str = "") -> ReferenceTraceCheck:
    """Run W46's check. ``residual`` maps a trace to a residual norm.

    Three calls, or four with ``solved_trace``.  A `False` verdict does not say
    the solver is broken; it says the EQUATION is not one the true solution
    solves, which is a different and much worse problem.
    """
    return ReferenceTraceCheck(
        lagged=float(residual(lagged_trace)),
        reference=float(residual(reference_trace)),
        solved=None if solved_trace is None else float(residual(solved_trace)),
        note=note,
    )


def shared_layer_flux_sum(w_ring, w_a_in, w_b_in, nu: float, h: float):
    """F_A + F_B at a seam whose two sides pin the SAME cell layer.

    Both sides read ``w_ring``, so with outward normals the sum is
    ``nu (2 w_ring - w_A,in - w_B,in) / h``, which is ``-nu h`` times the centred
    second difference of ``w`` across the seam.  Returned so the identity is
    checkable rather than asserted: it is why flux balance cannot be the
    interface condition of a shared-layer decomposition, and it is verified
    bit-exactly against `reference.WindowNS`'s own monolithic solution on all
    four seams of the 2x2 tiling.
    """
    w_ring = np.asarray(w_ring, dtype=float)
    return nu * (2.0 * w_ring - np.asarray(w_a_in, dtype=float)
                 - np.asarray(w_b_in, dtype=float)) / h
