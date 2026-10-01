"""L1 -- the expert declaration and the capability record.

end-to-end-architecture-spec §3, in its 2026-08-27 amended form.

An expert is a frozen black box with a label on the outside.  Everything the
framework is allowed to conclude about a composition is inherited from labels,
never from inspecting the box -- so the label has to carry every property any
downstream guarantee depends on.

The amendment matters and this module implements the amended record, not the
original.  ``mapping``, ``reduction``, ``prolongation`` and ``adjointness`` were
four independent per-port declarations; they are one.  A port declares an
interface space and one prolongation, and the other three are derived.  This is
strictly better than a per-type default, because a default can be overridden and
a derivation cannot: the classic backwards-mapping bug needs two declarations
that can disagree, and there is now only one.  Declaring the dropped fields is
therefore itself an error, and ``PortDecl`` refuses them.

The record carries no physics.  ``boundary_response`` is the only callable the
probe needs, and it is the expert's own response to imposed boundary data.
"""

from __future__ import annotations

import enum
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .holes import UNMEASURED_CONSTANTS, Unmeasured
from .ports import (PortType, ResponseHalf, RingActuation, ScaleCheck,
                    check_scales, spec_for)
from .transfer import InterfaceSpace, Prolongation


class RecordError(ValueError):
    """A capability record that cannot be read as declared."""


class BCChannel(enum.Enum):
    """The rung of the transmission ladder an expert's own interface can accept.

    Ordered, because rule R1 is an inequality over this order: the interface is
    as weak as its weakest agent.  R2 is the exception that keeps the ladder open
    to black boxes -- probed-DtN needs only ``dirichlet``, because the higher rung
    is built outside the expert.
    """

    NONE = "none"
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    ROBIN = "robin"
    VENTCELL = "ventcell"

    @property
    def rung(self) -> int:
        return ["none", "dirichlet", "neumann", "robin", "ventcell"].index(self.value)


class Transmission(enum.Enum):
    """The transmission axis of the scheme. Ordered by rung."""

    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    DIRICHLET_NEUMANN = "dirichlet-neumann"
    ROBIN = "robin"
    OPTIMIZED_ROBIN = "optimized-robin"
    VENTCELL = "ventcell"
    PROBED_DTN = "probed-DtN"

    @property
    def rung(self) -> int:
        return [
            "dirichlet",
            "neumann",
            "dirichlet-neumann",
            "robin",
            "optimized-robin",
            "ventcell",
            "probed-DtN",
        ].index(self.value)


class Differentiable(enum.Enum):
    NONE = "none"
    JVP = "jvp"
    VJP = "vjp"
    BOTH = "both"

    @property
    def has_jvp(self) -> bool:
        return self in (Differentiable.JVP, Differentiable.BOTH)

    @property
    def has_vjp(self) -> bool:
        return self in (Differentiable.VJP, Differentiable.BOTH)


class EllipticSubsolve(enum.Enum):
    """Whether the agent's step contains a solve with an INFINITE domain of dependence.

    Measured 2026-08-27, and it is the finding that cost the most.  A projection
    method's pressure Poisson solve is global over whatever domain it is run on.
    Decompose the domain and you have decomposed that solve -- and the resulting
    error is elliptic, so it does **not** decay with distance from the cut, does
    **not** shrink with the time step, and no halo, partition of unity or
    interface condition removes it.  On the first real case study it was 99.8% of
    the defect and was being read as agent infidelity.

    ``embedded``  the agent solves it internally, per window.  A decomposition
                  silently changes the operator, and L2 refuses.
    ``exposed``   the agent offers the elliptic part separately, so the
                  composition layer can run it globally.  This is what
                  probed-dtn-coupling 2.1 assumes when it says the elliptic part
                  "stays where it is".
    ``none``      no elliptic sub-solve: nothing to decompose.
    ``unknown``   a black box whose internals nobody can see.  **Added 2026-08-29
                  (W60), and only after the measurement route was shown to
                  fail.**  W60 opened saying a checkpoint's record cannot be
                  honest -- ``none`` asserts something unmeasured *and switches
                  R10 off*, ``embedded`` asserts something unmeasured *and makes
                  L2 refuse* -- and left the resolution to
                  `poseidon.elliptic_signature`, to be "promoted from a diagnosis
                  to a gate once it has more than two calibration points".

                  **The extra calibration points arrived and they falsify the
                  promotion.**  `cases/thermal_seam.py`'s shell is a *known*
                  embedded elliptic part -- backward Euler over the whole shell,
                  plus a quasi-static elasticity solve with no time step to
                  shrink at all -- and the statistic reads
                  ``kappa = 1.0002, asymmetry = 8.8e-5``, which its own
                  thresholds call *"consistent with EXPOSED or NONE"*.  It stays
                  wrong across a 10,000x range of exchange interval
                  (asymmetry 2.7e-6 to 7.1e-6 against an embedded calibration
                  point of 0.153, a factor of 5e4).

                  The diagnosis is precise: the statistic measures
                  **non-normality**, and it was read as measuring **globality**.
                  Those coincide for a pressure projection, where the elliptic
                  part is a constraint that makes the interface operator strongly
                  non-normal, and they come apart for a **self-adjoint** elliptic
                  operator -- ``(M/dt + K)`` is SPD, so its Schur complement on a
                  seam is near-normal *however global it is*.  So there is a whole
                  class of embedded elliptic solves the measurement cannot see,
                  and for a black box there is no honest value and no prospect of
                  measuring one.  Hence an enum value, which decertifies.

                  It decertifies rather than refusing for the three-verdict
                  split's own reason: an undeclared internal is an unverified
                  hypothesis, not a measured contradiction.  And it is strictly
                  more conservative than ``none``, which is what W61 asks of a
                  default -- ``none`` admits silently and this does not.
    """

    NONE = "none"
    EMBEDDED = "embedded"
    EXPOSED = "exposed"
    UNKNOWN = "unknown"


class TimeDiscretization(enum.Enum):
    """How the agent advances a macro-step, and it decides whether probed-DtN applies.

    **W46, measured 2026-08-27.** The Steklov-Poincare formulation
    ``sum_i Lambda_i lambda = chi`` is the interface condition of a *boundary-value
    problem*. Discretize an unsteady problem implicitly and each macro-step IS one
    -- an elliptic solve for the new state -- so flux balance is exactly right.
    Discretize it explicitly and there is no boundary-value problem at all: the
    new state is an explicit function of the old one, the domain of dependence is
    finite, and flux balance is a *steady* condition that the true trace does not
    satisfy.

    Measured on `reference.WindowNS`: substituting the reference solution's own
    trace left the flux residual unchanged (2.6421e-5 -> 2.6520e-5), and solving
    the condition exactly moved the trace 41x past the truth -- growing to 150x as
    dt was refined, which is the signature of a steady condition imposed inside an
    unsteady step. So the rung is gated on this field rather than on bc_channel
    alone.
    """

    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    UNKNOWN = "unknown"


class MotionClass(enum.Enum):
    STATIC = "static"
    PRESCRIBED = "prescribed"
    SOLUTION_DEPENDENT = "solution_dependent"


class ClaimType(enum.Enum):
    TRAJECTORY = "trajectory"
    STATISTICAL = "statistical"


class Direction(enum.Enum):
    BIDIRECTIONAL = "bidirectional"
    IN = "in"
    OUT = "out"


#: Fields the amendment removed. Declaring one is an error, not a preference:
#: it re-creates the two-declarations-that-can-disagree bug the amendment closed.
DROPPED_PORT_FIELDS = ("mapping", "reduction", "adjointness", "conservative")


@dataclass
class PortDecl:
    """One port on one expert, in the amended form.

    ``interface_space`` and ``prolongation`` may be left None on the expert's own
    record when the seam supplies them -- a port declaration lives with the expert
    and a seam is a property of a connection.  L3 refuses a connection for which
    neither the port nor the seam supplies them.
    """

    name: str
    port_type: PortType
    geometry: str = ""
    direction: Direction = Direction.BIDIRECTIONAL
    interface_space: InterfaceSpace | None = None
    prolongation: Prolongation | None = None
    motion_class: MotionClass = MotionClass.STATIC
    nondim: dict[str, float] = field(default_factory=dict)
    passengers: tuple[str, ...] = ()
    effective_resolution: int | None = None   # m_i_eff; sets dim M with the other side
    #: W66. Which half of the conjugate pair ``boundary_response`` RETURNS.
    #: Undeclared by default, which decertifies at L3/C9 rather than assuming a
    #: convention: no property of the returned numbers distinguishes an effort
    #: from a flow, and both cheaper routes were measured and failed. See
    #: ports.ResponseHalf for the two measurements.
    response_half: ResponseHalf = ResponseHalf.UNDECLARED
    #: W152. How many components the ring this port sits on actually
    #: carries, and how many of them a control on this port may drive.
    #: `Lambda` and the matching condition want the conjugate pair and are
    #: unaffected; a CONTROL is bounded by `actuation`, and Tier 32
    #: measured that bound at 7.6% against 97.7% of the available sigma
    #: reduction on one seam. Undeclared by default -- see ports.RingActuation.
    ring_components: int | None = None
    actuation: RingActuation = RingActuation.UNDECLARED
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.port_type, PortType):
            self.port_type = PortType(str(self.port_type))
        if not isinstance(self.response_half, ResponseHalf):
            self.response_half = ResponseHalf(str(self.response_half))
        if not isinstance(self.actuation, RingActuation):
            self.actuation = RingActuation(str(self.actuation))
        if self.ring_components is not None and self.ring_components < 1:
            raise RecordError(
                f"port {self.name!r}: ring_components = "
                f"{self.ring_components}; a ring carries at least one"
            )
        if self.port_type is not PortType.ADVEC and self.passengers:
            raise RecordError(
                f"port {self.name!r}: only ADVEC carries passengers; {self.port_type.value} "
                f"declared {list(self.passengers)}"
            )
        if self.port_type is PortType.ADVEC and not self.passengers:
            self.passengers = ("h0",)

    @property
    def spec(self):
        return spec_for(self.port_type)

    @property
    def actuated_components(self) -> int | None:
        """How many ring components a control on this port may drive.

        ``None`` when undeclared, which is not the same as 1: a caller that
        needs the number has to notice the absence rather than inherit the
        narrow answer by default.
        """
        if self.actuation is RingActuation.CONJUGATE_PAIR:
            return 1
        if self.actuation is RingActuation.FULL_RING:
            return self.ring_components
        return None

    def scale_check(self) -> ScaleCheck:
        return check_scales(self.port_type, self.nondim, self.passengers)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "port_type": self.port_type.value,
            "geometry": self.geometry,
            "direction": self.direction.value,
            "motion_class": self.motion_class.value,
            "passengers": list(self.passengers),
            "effective_resolution": self.effective_resolution,
            "response_half": self.response_half.value,
            "ring_components": self.ring_components,
            "actuation": self.actuation.value,
            "actuated_components": self.actuated_components,
            "declares_interface_space": self.interface_space is not None,
            "declares_prolongation": self.prolongation is not None,
        }


def port_decl(**kw: Any) -> PortDecl:
    """Construct a PortDecl, refusing the four fields the amendment dropped."""
    offenders = [k for k in DROPPED_PORT_FIELDS if k in kw]
    if offenders:
        raise RecordError(
            f"port {kw.get('name')!r} declares {offenders}, which the 2026-08-27 amendment "
            "removed. Declare an interface space and one prolongation; the mapping class, "
            "the reduction and adjointness are derived from them. A default can be "
            "overridden and a derivation cannot."
        )
    return PortDecl(**kw)


#: The signature the probe needs from an expert, and the only one it needs.
#: trace on V_i  ->  flux on V_i.  It mentions no governing equation, which is why
#: probing survives the failure of E3 while tau does not.
BoundaryResponse = Callable[[str, np.ndarray], np.ndarray]


@dataclass
class ExpertCapabilities:
    """The complete L1 record.

    Five fields beyond general-coupling-scheme §2 are added by the end-to-end
    spec and every one of them exists to make a downstream check decidable:
    ``governing_family`` (E3 by string comparison), ``lambda_ref`` (tau at a
    multiphysics seam), ``claim_types`` (L9), per-port ``motion_class`` (E2), and
    the interface-space / prolongation pair that replaced four fields.
    """

    expert_id: str
    ports: list[PortDecl]

    bc_channel: BCChannel = BCChannel.NONE
    bc_time_varying: bool = False
    #: R10. See EllipticSubsolve: an embedded global sub-solve is not decomposable.
    elliptic_subsolve: EllipticSubsolve = EllipticSubsolve.NONE
    #: **W348, 2026-09-30.**  True when every datum the embedded elliptic solve
    #: uses on this agent's artificial boundary arrives through its ports, and the
    #: composition re-solves the agent with it until the interface agrees: a
    #: converged coupling then closes the local problem with the neighbours'
    #: values, not with data of the agent's own.  Tier 0's windows did not -- their
    #: pressure Poisson problem took homogeneous Neumann data on the cut whatever
    #: the neighbours held, which is R10's 99.8%.  The default keeps R10's
    #: refusal; True moves the agent's case after the scheme (`_r10_scheme`).
    elliptic_data_from_ports: bool = False
    #: R2b / W46. Probed-DtN needs an implicit macro-step; see TimeDiscretization.
    time_discretization: TimeDiscretization = TimeDiscretization.UNKNOWN
    #: Cells the agent's stencil reaches per internal sub-step. With
    #: ``substeps_per_macro_step`` it gives the halo a lagged exchange needs:
    #: information from the artificial boundary reaches ``radius * substeps``
    #: cells inward before the next exchange corrects it.
    stencil_radius: int = 1
    #: How many internal sub-steps the agent takes per macro-step. ``None`` means
    #: undeclared, and the halo rule is then undecidable rather than assumed safe.
    substeps_per_macro_step: int | None = None
    differentiable: Differentiable = Differentiable.NONE
    dt_native: float | None = None
    L_native: float | None = None
    regime_law: Callable[..., Any] | None = None
    storage: Callable[..., float] | None = None
    equivariances: tuple[str, ...] = ()
    validity: Callable[..., bool] | None = None

    governing_family: str | None = None
    lambda_ref: Any | None = None
    claim_types: frozenset[ClaimType] = frozenset({ClaimType.TRAJECTORY})

    weight_hash: str | None = None
    boundary_response: BoundaryResponse | None = None
    #: (port, trace, direction) -> directional derivative of the flux.
    #: Required when ``differentiable`` claims a JVP; the conformance suite checks
    #: it against a finite difference, and a disagreement there is loud, not silent.
    boundary_response_jvp: Callable[[str, np.ndarray, np.ndarray], np.ndarray] | None = None
    #: **R9 / W7, 2026-08-29.**  ``(port, trace, n_substeps) -> the flux AVERAGED
    #: over ``n_substeps`` of this agent's own clock``, which is
    #: ``(1/DT) int_t^{t+DT} F dt`` with ``DT = n_substeps * dt_native``.
    #:
    #: Required at a multirate seam when the graph declares
    #: ``FluxMatching.TIME_INTEGRATED``, and required rather than optional for
    #: `boundary_response_jvp`'s reason: a scheme that claims to match an integral
    #: with no way to compute one is a contradiction inside the declaration, and
    #: L7 decides it rather than the run discovering it.
    #:
    #: **Why `boundary_response` cannot stand in for it.**  That callable is one
    #: state in, one flux out, and it restarts from the agent's own state every
    #: time -- so calling it ``n`` times does not march ``n`` sub-steps, it
    #: recomputes the same first sub-step ``n`` times.  The integral is not
    #: recoverable from the interface this record already has, which is why this
    #: is a field and not a helper.
    #:
    #: The trace is a single value, not a waveform: at ``W = 1`` the composition
    #: holds it constant across the interval, and the flux still varies because
    #: the AGENT's state evolves under it.  That is the whole of the leak R9 is
    #: about, and it is separable from W17's ``W > 1``, which is about the trace
    #: varying too.
    boundary_response_integrated: Callable[[str, np.ndarray, int], np.ndarray] | None = None
    #: **W74, 2026-08-30.**  ``port -> the trace the probe should linearize about``.
    #:
    #: The probe used to hard-wire the zero trace, and for an affine expert that
    #: is exactly right: the derivative is the same everywhere, so subtracting the
    #: zero probe removes every trace-independent bias and costs nothing.  The
    #: assumption is invisible because it is *free* on a port whose variable is a
    #: perturbation -- a velocity, a wake deficit -- where zero is both the origin
    #: and a state the expert is in all the time.
    #:
    #: It stops being free when the port variable has an **absolute origin**.
    #: `thermal_seam`'s THERM effort is a temperature in kelvin, so the zero trace
    #: is 0 K: the gas is asked for its wall flux against a 0 K wall, and the
    #: shell conducts against a 0 K gas.  Measured against the experts' own
    #: operating point, ``beta`` on that seam moves **4.8062 -> 0.3807, a factor
    #: of 12.6**, and the seam operator's norm moves 19.23 -> 1.52.  The verdict
    #: and the whole decertification set are unchanged, so the compiler's
    #: CONCLUSIONS were robust; every NUMBER the certificate carried was not.
    #:
    #: And the nonlinearity is not incidental.  It is not radiation -- setting
    #: ``radiate=False`` changes nothing to six digits -- it is the declared bond
    #: itself: `PORT_SPECS[THERM]` pairs $(T, q_n/T)$, chosen over the pseudo-bond
    #: $(T, q_n)$ precisely so effort times flow is a power, and dividing by $T$
    #: is what makes the response nonlinear in the effort.  The pseudo-bond would
    #: have been affine.  Two correct choices with incompatible assumptions.
    #:
    #: Undeclared means zeros, which is what every record written before this
    #: field existed does, so every measurement taken before it reproduces
    #: bit-for-bit.  It is not a fifth unverifiable declaration (W69): a base the
    #: expert cannot accept raises, and a base far from the operating point shows
    #: up in `probe.operator_content` and in the eps window rather than passing
    #: silently.
    probe_base: Callable[[str], np.ndarray] | None = None
    reproducibility_floor: float = 1e-6      # sets the finite-difference probe step
    deterministic: bool = True
    note: str = ""

    # Measured or fitted later; never defaulted to a plausible number.
    L_fitted: float | Unmeasured = field(default_factory=lambda: UNMEASURED_CONSTANTS["L"])

    def __post_init__(self) -> None:
        if not self.ports:
            raise RecordError(f"expert {self.expert_id!r} declares no ports")
        names = [p.name for p in self.ports]
        if len(set(names)) != len(names):
            raise RecordError(f"expert {self.expert_id!r} has duplicate port names: {names}")
        if isinstance(self.claim_types, (set, list, tuple)):
            self.claim_types = frozenset(
                c if isinstance(c, ClaimType) else ClaimType(str(c)) for c in self.claim_types
            )
        if self.weight_hash is None:
            # A certificate binds to a weight hash: a retrained expert is a
            # different expert and inherits nothing. A placeholder derived from
            # the id makes that binding visible rather than absent.
            self.weight_hash = "unhashed:" + hashlib.sha256(self.expert_id.encode()).hexdigest()[:16]

    # -- lookups -----------------------------------------------------------

    def port(self, name: str) -> PortDecl:
        for p in self.ports:
            if p.name == name:
                return p
        raise RecordError(f"expert {self.expert_id!r} has no port {name!r}")

    def ports_of_type(self, port_type: PortType) -> list[PortDecl]:
        return [p for p in self.ports if p.port_type is port_type]

    def required_halo(self) -> int | None:
        """Cells of overlap a once-per-macro-step exchange needs to stay faithful.

        ``radius * substeps``: one sub-step advances the artificial boundary's
        influence ``radius`` cells, and nothing corrects it until the next
        exchange.  Re-assembling every sub-step instead reduces the requirement to
        ``radius`` -- which is the trade the compiler prices, and the reason this
        is a method rather than a constant.

        **Corrected 2026-08-30 (W69).**  That product is an *explicit* agent's
        domain of dependence.  Discretize implicitly and one macro-step inverts
        an operator that couples every cell to every other, so the reach is the
        whole domain however small the assembly stencil is -- and this method
        consulted ``stencil_radius`` and ``substeps_per_macro_step`` and never
        ``time_discretization``, which sits in the same record.  A rule with two
        inputs and a third it does not consult.

        Found by measuring rather than by reading: `thermal_seam`'s shell
        declares ``stencil_radius=1, substeps=1`` -- honestly, they describe its
        Q1 element -- so this returned **1**, while a delta on the seam is still
        moving the response **5** cells away at the 1e-6 level and, being a dense
        inverse, everywhere at some level.  `_halo_rule` REFUSES an overlap below
        this number, so the wrong value buys a passing halo check.

        It was latent: the shell is on a non-overlapping graph, and the one
        implicit agent on an overlapping graph in this vault is
        `wind_farm_real`'s actuator disk, whose ``stencil_radius`` is 0.  Which
        is also why the guard is on ``radius >= 1`` rather than on the label
        alone -- a zero-stencil algebraic closure couples nothing, so there is no
        dense inverse to be global, and its reach is 0 under either
        discretization.  The guard follows the derivation, not the verdict.

        **Corrected again 2026-08-30 (W93), by the same derivation on a third
        case.**  W69's sentence is *"that product is an EXPLICIT agent's domain
        of dependence"*, and it fixed the branch where the record says
        ``implicit``.  It left the branch where the record says nothing.  An
        agent declaring ``time_discretization = unknown`` has not said its step
        is explicit, so ``radius * substeps`` is not licensed for it either --
        and ``unknown`` is exactly what a frozen learned one-shot map declares,
        under R2b, because it is neither.

        Measured on Poseidon-T at a real wake seam: the record declares
        ``stencil_radius = 2, substeps = 1``, this returned **2**, and
        `probe.support_reach` -- which measures the same quantity, by poking a
        delta rather than by reading a field -- finds the response **nonzero in
        every one of the 128 seam cells at every amplitude from 1 to 1e-3**, 64
        cells from the poke.  The declaration was 32x short and `_halo_rule` was
        admitting a 16-cell overlap on it.  A neural operator's receptive field
        is global by construction, which is a fact about the architecture and not
        about the physics; the honest report is that the halo is undecidable from
        the record, and `conformance` measures it.
        """
        if self.substeps_per_macro_step is None:
            return None
        if (self.time_discretization in (TimeDiscretization.IMPLICIT,
                                         TimeDiscretization.UNKNOWN)
                and int(self.stencil_radius) >= 1):
            return None
        return int(self.stencil_radius) * int(self.substeps_per_macro_step)

    @property
    def declares_storage(self) -> bool:
        return self.storage is not None

    @property
    def declares_validity(self) -> bool:
        return self.validity is not None

    @property
    def can_be_probed(self) -> bool:
        """R2: probed-DtN requires only that the expert accept a Dirichlet trace."""
        return self.boundary_response is not None and self.bc_channel.rung >= BCChannel.DIRICHLET.rung

    def probe_route(self) -> str:
        """R7 -- the probing method follows ``differentiable``."""
        if self.differentiable.has_jvp:
            return "jvp"
        if self.deterministic:
            return "finite-difference"
        return "regression"

    def probe_class(self) -> str:
        """probed-dtn-coupling §5.3: an axis invisible to any accuracy benchmark."""
        return {
            "jvp": "probe-cheap",
            "finite-difference": "probe-affordable",
            "regression": "probe-expensive",
        }[self.probe_route()]

    # -- schema completeness ----------------------------------------------

    def missing_fields(self) -> list[str]:
        """Fields a downstream layer needs that this record does not carry.

        L1's refusal is 'a record missing a field a downstream layer needs', so
        this is the list that refusal is built on.  It is deliberately not a
        list of *all* unset fields: an unset field with a defined default is a
        declaration, and an unset field with no default is a hole.
        """
        missing: list[str] = []
        if self.dt_native is None:
            missing.append("dt_native")
        if self.governing_family is None:
            missing.append("governing_family")
        if self.boundary_response is None:
            missing.append("boundary_response")
        if self.differentiable.has_jvp and self.boundary_response_jvp is None:
            # A record that claims a derivative and cannot supply one is a
            # contradiction inside the declaration itself, and it is decidable at
            # L1 rather than being discovered when the probe tries to use it.
            missing.append("boundary_response_jvp (claimed by differentiable)")
        for p in self.ports:
            if p.effective_resolution is None:
                missing.append(f"ports[{p.name}].effective_resolution")
        return missing

    def as_dict(self) -> dict[str, Any]:
        out = {
            "expert_id": self.expert_id,
            "weight_hash": self.weight_hash,
            "bc_channel": self.bc_channel.value,
            "bc_time_varying": self.bc_time_varying,
            "elliptic_subsolve": self.elliptic_subsolve.value,
            "time_discretization": self.time_discretization.value,
            "stencil_radius": self.stencil_radius,
            "substeps_per_macro_step": self.substeps_per_macro_step,
            "required_halo": self.required_halo(),
            "differentiable": self.differentiable.value,
            "dt_native": self.dt_native,
            "L_native": self.L_native,
            "governing_family": self.governing_family,
            "declares_probe_base": self.probe_base is not None,
            "lambda_ref": None if self.lambda_ref is None else "declared",
            "claim_types": sorted(c.value for c in self.claim_types),
            "storage": self.declares_storage,
            "validity": self.declares_validity,
            "equivariances": list(self.equivariances),
            "regime_law": self.regime_law is not None,
            "probe_route": self.probe_route(),
            "probe_class": self.probe_class(),
            "ports": [p.as_dict() for p in self.ports],
        }
        if self.elliptic_data_from_ports:
            # W348: written only when declared, so every record from before it
            # is byte-identical (the W189 control)
            out["elliptic_data_from_ports"] = True
        return out


def zero_response(_port: str, trace: np.ndarray) -> np.ndarray:
    """The response of an expert with no boundary channel: identically zero.

    An expert with ``bc_channel: none`` has no place to put a boundary ring, so
    its response to imposed boundary data does not depend on that data.  The
    probe then measures Lambda == 0, S == 0, and the interface problem is not
    ill-conditioned but empty.  This function exists so that fact is produced by
    the machinery rather than special-cased.
    """
    return np.zeros_like(np.asarray(trace, dtype=float))


def linear_response(matrix: np.ndarray, bias: np.ndarray | None = None) -> BoundaryResponse:
    """A declared linear boundary response, for fixtures and reference operators.

    Used to exercise the probe without any physics.  The probe cannot tell this
    from a neural expert, which is the point: it hands over a trace and takes
    back a flux.
    """
    A = np.asarray(matrix, dtype=float)
    b = None if bias is None else np.asarray(bias, dtype=float)

    def respond(_port: str, trace: np.ndarray) -> np.ndarray:
        out = A @ np.asarray(trace, dtype=float)
        return out if b is None else out + b

    return respond
