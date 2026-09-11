"""The case graph -- the whole of what a new case study writes.

A case study is a list of agents, each carrying a declared capability record,
and a list of port connections.  Nothing else.  There is no place in this module
for coupling code, and adding a case study adds no code anywhere else in the
package: the scheme is compiled from the declarations by ``compiler.compile_scheme``.

Two things are declared here that are properties of the *decomposition* rather
than of any agent:

  * the decomposition axis, overlapping or non-overlapping, which decides whether
    cross-points are a problem at all (interface-transfer-theory §7);
  * topology events and global fields, which bypass the port mechanism -- gravity
    and any uniform external field enter each agent's update directly and are a
    third category alongside edges and internal state.
"""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

from .capability import ExpertCapabilities, MotionClass, PortDecl
from .ports import PortType
from .transfer import InterfaceSpace, Prolongation, SeamTransfer, TransferDeclarationError


class GraphError(ValueError):
    """A case graph that cannot be read as declared."""


class Decomposition(enum.Enum):
    OVERLAPPING = "overlapping"
    NON_OVERLAPPING = "non-overlapping"


class FluxMatching(enum.Enum):
    """Which flux the two sides of a seam are required to agree on.

    **R9, and the declaration it has been refusing for want of, 2026-08-29.**
    `compiler._clocks` has refused every multirate graph since the compiler
    existed, with the exit named in the refusal itself -- *"until the scheme
    declares time-integrated matching with each side's own substep quadrature"* --
    and no way to declare it.  This is that way.

    ``POINTWISE`` matches the flux at the end of the exchange interval.  It is the
    default because it is what a composition does when nobody thinks about it, and
    across one clock it is right: one sub-step *is* the interval.

    ``TIME_INTEGRATED`` matches ``int_t^{t+DT} F dt``, each side quadrature'd on
    its own sub-steps.  Declaring it is not free -- L7 requires every agent at a
    multirate seam to supply `boundary_response_integrated`, on the
    `boundary_response_jvp` precedent: a record that claims an integral and has no
    way to compute one is a contradiction inside the declaration.
    """

    POINTWISE = "pointwise"
    TIME_INTEGRATED = "time-integrated"


@dataclass
class Agent:
    """One node: an expert instance placed on a subdomain."""

    agent_id: str
    capabilities: ExpertCapabilities
    domain: str = ""
    role: str = ""
    candidates: tuple[ExpertCapabilities, ...] = ()   # for static routing (L1.5 / G15)

    def port(self, name: str) -> PortDecl:
        return self.capabilities.port(name)


@dataclass
class Connection:
    """One edge: two ports of the same type, plus the seam's declared transfer.

    ``space`` and ``prolongations`` are the seam-level form of the amended
    declaration.  Either the connection supplies them, or both ports do; L3
    refuses a connection for which neither does.
    """

    seam_id: str
    a: tuple[str, str]                 # (agent_id, port_name)
    b: tuple[str, str]
    port_type: PortType
    space: InterfaceSpace | None = None
    prolongations: dict[str, Prolongation] = field(default_factory=dict)
    orientation: str = ""              # normal points from the first-named agent to the second
    #: **W138, 2026-09-08.**  Which agent's outward normal BOTH sides report
    #: their EFFORT against.  `response_half` names *which half* a port returns
    #: and nothing named *which way its normal points*, so ``Lambda_M = sum_i
    #: P_i^* Lambda_i P_i`` added two blocks that the interface residual
    #: subtracts, and `L4/E7/passivity` read the symmetric part of a sum of two
    #: oppositely-oriented responses.
    #:
    #: Empty --- the default, and what every graph written before 2026-09-08
    #: means --- is the **Steklov-Poincare** convention: each side reports the
    #: traction against its OWN outward normal, the two normals at Gamma are
    #: opposite, each block is separately positive semidefinite, and the
    #: assembled operator is their SUM.  That is what a tiling cut does and the
    #: seven fluid-fluid seams of `front_wing` measure it: every block has
    #: ``lambda_min`` near ``+0.25`` and the sum near ``+0.49``.
    #:
    #: Naming an agent says the two efforts are **co-oriented** --- both positive
    #: in one shared direction, that agent's outward normal --- so the other
    #: side's block is its own-outward block NEGATED, and the assembled operator
    #: is the DIFFERENCE with the named side kept.  A field-solid seam where the
    #: fluid returns *the load on the surface* and the solid *the reaction that
    #: holds it* is this case, and it is the one the residual is written in.
    #:
    #: **It is checkable and it was checked, not asserted.**  On `front_wing`'s
    #: `wet` seam the case's own ``solve_seams`` residual has the analytic
    #: Jacobian ``dt S_e + diag(g)`` --- structure MINUS fluid in the shared
    #: normal --- and a central difference reproduces it to ``1.15e-16``
    #: relative.  That Jacobian is positive definite at ``lambda_min = +3.469``
    #: where the SUM the probe was assembling is indefinite at ``-3.488``.  The
    #: scheme differentiates the oriented operator; the probe was reading a
    #: different matrix.
    effort_normal: str = ""
    enforced: bool = True              # enforce the port residual, or measure only
    #: Opt in to having the compiler build M from the rule dim M = min_i m_i_eff,
    #: with the identity prolongation on each side. Only legal for the conforming
    #: case: a non-conforming transfer is a modelling choice, not a derivation.
    derive_space: bool = False
    #: A declaration that the two interface discretizations coincide. It is what
    #: E2's coincidence half is stamped from, and **nothing checks it** -- it is a
    #: label like any other, and the conformance argument applies to it. It does
    #: NOT make the connection admissible: since the 2026-08-27 amendment,
    #: admissibility needs a declared interface space and prolongation even in the
    #: coincident case, where the prolongation is the identity.
    geometrically_coincident: bool = False
    #: **W171.** The decomposition axis of the cut THIS seam is, when it is one.
    #:
    #: ``None`` -- the default, and what every graph written before 2026-09-10
    #: means -- inherits the graph's own ``decomposition``.  Naming an axis says
    #: *this seam is a cut of this kind*, which is what a union graph needs: a
    #: tiling and a circuit in one compile have two axes and one field per graph
    #: cannot say so.
    #:
    #: **It is a declaration and not a derivation, and that is a correction.**
    #: Tier 40 proposed deriving it from ``geometrically_coincident`` and
    #: measured the proposal on four graphs.  Censused over all eleven
    #: constructible case graphs it does not separate the axis: a same-family
    #: coincident seam appears under BOTH axes, because coincidence is a
    #: property of the two sides' *discretizations* -- E2's coincidence half --
    #: and the axis is a property of their *domains*.  `window_ns` and
    #: `wind_farm_real` are overlapping tilings declaring coincident seams and
    #: are right to.  Nothing else derived separates it either: `rocket` is
    #: overlapping with no partition of unity and no ``overlap_cells``.
    #:
    #: Like ``geometrically_coincident``, nothing checks it.  A seam that
    #: declares an axis its region's other seams contradict is reported by
    #: `CaseGraph.region_axis_conflicts` rather than resolved.
    cut_axis: "Decomposition | None" = None
    #: The null-space dimension the case expects, e.g. one for incompressibility.
    #: Leaving it None disables the free correctness check rather than defaulting it.
    expected_null_dim: int | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.port_type, PortType):
            self.port_type = PortType(str(self.port_type))
        if self.a[0] == self.b[0]:
            raise GraphError(f"seam {self.seam_id!r} connects agent {self.a[0]!r} to itself")
        if not self.orientation:
            self.orientation = f"n points {self.a[0]} -> {self.b[0]}"
        if self.effort_normal and self.effort_normal not in self.agents:
            raise GraphError(
                f"seam {self.seam_id!r} declares effort_normal="
                f"{self.effort_normal!r}, which is not one of its two agents "
                f"{self.agents}. The field names the agent whose outward normal "
                "BOTH sides' effort is reported against, so it has to be a side "
                "of this seam"
            )

    @property
    def effort_signs(self) -> dict[str, float]:
        """Each side's sign in the assembled operator (W138).

        With no ``effort_normal`` both sides are ``+1``: each reports against its
        own outward normal and the assembly is the Steklov-Poincare SUM.  Naming
        an agent keeps that side at ``+1`` and negates the other, because the
        other is reporting against a normal opposite to its own outward one.
        """
        a, b = self.agents
        if not self.effort_normal:
            return {a: 1.0, b: 1.0}
        return {x: (1.0 if x == self.effort_normal else -1.0) for x in (a, b)}

    @property
    def agents(self) -> tuple[str, str]:
        return (self.a[0], self.b[0])

    def side(self, agent_id: str) -> tuple[str, str]:
        if agent_id == self.a[0]:
            return self.a
        if agent_id == self.b[0]:
            return self.b
        raise GraphError(f"agent {agent_id!r} is not on seam {self.seam_id!r}")

    def other(self, agent_id: str) -> str:
        return self.b[0] if agent_id == self.a[0] else self.a[0]


@dataclass
class GlobalField:
    """A uniform external field. It bypasses L3 entirely.

    Gravity and its kin enter each agent's update directly rather than through an
    edge, and contribute to P_ext in the power residual and to nothing else.

    **W117, 2026-09-02: that bypass is the one route that improves an envelope
    stamp by deleting the seam.**  Declaring another agent's state a global field
    is numerically exact and turns off every check a seam defines -- no scale set,
    no prolongation, no adjoint, no null space, no tau, sigma or beta -- which
    `port-algebra-atlas-0.1` section 10.4 calls the silent-wrongness class in its
    purest form.  Until W117 the class had **no reader at all**: `global_fields`
    was referenced nowhere in the compiler, and this dataclass recorded `name`,
    `applies_to` and `note`, none of which says *where the value comes from* --
    the one fact a rule would have to test.

    ``produced_by`` is that fact, and its three cases are the whole rule:

    * ``None``  -- UNDECLARED.  The default is deliberately not ``()``: a silent
      default would read every legacy declaration as external and certify the
      thing the field exists to catch.  `L3/global-field` decertifies it.
    * ``()``    -- EXTERNAL.  No agent in this graph produces it: gravity, an
      ambient field, or a declared absence.  This is what section 5.1 wrote the
      class for.
    * agent ids -- PRODUCED.  If they are *all* of the graph's agents the field is
      a global operation over the whole state, owned by the composition layer --
      a split-step pressure solve is the vault's four instances of this.  If they
      are a **proper subset** and the field ``applies_to`` an agent outside it,
      the field carries one agent's state into another's update with no seam,
      and that is a connection declared out of the port algebra.
    """

    name: str
    applies_to: tuple[str, ...] = ()   # empty means every agent
    note: str = ""
    #: **W117.** Which agents in this graph produce the field's value. `None` is
    #: undeclared and is decertified; `()` asserts the field is external. See the
    #: class docstring -- this is the discriminator `L3/global-field` reads.
    produced_by: tuple[str, ...] | None = None


@dataclass
class DeclaredTopologyEvent:
    """A declared graph mutation. Refused unless it carries a state map AND a ledger.

    The refusal is in the compiler, not here; this class only records what was
    declared, so the refusal can say which of the two is missing.
    """

    event_id: str
    t: float
    graph_before: tuple[str, ...]
    graph_after: tuple[str, ...]
    state_map: Any | None = None
    ledger: dict[str, float] | None = None
    edge_reinstantiation: dict[str, str] = field(default_factory=dict)
    e0_reset: float | None = None


class CycleEnumerationBudget(RuntimeError):
    """`directed_cycles` ran past its search budget. **W163.**

    Raised rather than returning a partial list -- see `directed_cycles`.
    """


@dataclass(frozen=True)
class DeclaredLoopGain:
    """The composed gain of one traversal of a directed cycle, declared.

    **W163, 2026-09-09.**  A directed cycle in the seam graph is a fixed-point
    problem the compiler cannot see: compile `cooling_loop` with the return seam
    and without it and the decision record differs only by the extra seam's own
    rows -- same verdict, same rule set, same per-agent decisions -- while the
    measured consequence of the cycle is 1.5 K of order dependence at one sweep
    per macro-step.  `powertrain` reproduces that on unrelated physics.

    **The gain has to be DECLARED because it cannot be probed**, and that is a
    finding rather than a convenience.  W163's row carried an [AI Inference]
    that the contraction would be "decidable from the same declared response the
    probe already builds".  Checked on both graphs, it is not:

      * `ExpertCapabilities.boundary_response` is ``(port, trace) -> flux`` on
        the SAME port -- a Dirichlet-to-Neumann map.  A cycle's gain is a
        CROSS-port transfer, from what the agent is fed on its inlet to what it
        presents at its outlet, and no capability field carries one.
      * On CS-13 the two derivatives are unrelated: perturbing an ADVEC leg's
        port trace moves the response by 2303.89 J/kg per unit mass flux, while
        the loop's transport gain is ``a = 0.99926`` in temperature and appears
        only in the derivative with respect to the UPSTREAM STATE, which the
        port interface does not expose.  Reaching it means setting ``leg.t_in``
        on the expert object, which is reaching around the declaration.
      * On CS-14 the ELEC response IS the element's resistance up to the
        declared terminal area -- so on that graph the ingredient is in the
        interface after all.  Two graphs, two answers, which is exactly why the
        rule may not compute it: a mechanism that works on one of the two
        circuits in the package is not a mechanism.

    So the compiler asks, and refuses to guess.  ``gain`` is the spectral radius
    of the composed loop map over one traversal; below one the sweep converges
    to the cycle's fixed point from any start and the order of the sweep stops
    mattering in the limit, at one the map has no isolated fixed point, and
    above one it diverges.  Both case studies already raise at exactly this
    condition inside their own closed forms.
    """

    agents: tuple[str, ...]           # the cycle, in traversal order
    gain: float                       # rho of the composed one-traversal map
    source: str = ""                  # where the number was measured
    note: str = ""

    @property
    def contracts(self) -> bool:
        return bool(self.gain < 1.0)


#: **W58.**  The three legal values of `MeasuredConstants.cut_defect_bound_form`,
#: with what each one is and what it costs.  There is no default: the whole row
#: is that a number carrying neither form is unreadable.
CUT_DEFECT_FORMS: dict[str, str] = {
    "chi-weighted": (
        "|| sum_i chi_i |D_i| ||, D_i = E_i R_i - R_i E. L2/C2's own quantity, and "
        "the one the derivation bounds the composed defect by. Needs a MONOLITHIC "
        "reference, exactly as tau does, so it is a diagnostic and not a condition"
    ),
    "max": (
        "|| max_i |D_i| ||, the same bound with the partition of unity dropped. "
        "Also needs the monolith; 220x looser on the four-window tiling, and its one "
        "advantage is that it does not need the weights to be right"
    ),
    "neighbour-disagreement": (
        "MAX OVER PAIRS of ||E_i R_i u - E_j R_j u|| on the overlap. "
        "REFERENCE-FREE -- the common mode cancels there, so what survives is the "
        "part of the restriction defect the cut itself creates, and both local "
        "solves are already computed by the composed step. Equal to the max form "
        "on ONE overlapping pair with disjoint contaminated sets, and on nothing "
        "else: a max over pairs saturates as the tiling grows while the bound "
        "accumulates, so on a many-window tiling it under-estimates the bound "
        "for a reason unrelated to disjointness"
    ),
    "neighbour-disagreement-aggregated": (
        "|| max over pairs |E_i R_i u - E_j R_j u| ||, the same reference-free "
        "measurement aggregated the way the bound is: a cellwise max over pairs, "
        "then one norm over the whole grid. This is the form that is comparable "
        "with the max form at any size, and it equals it exactly when the "
        "contaminated sets are pairwise disjoint -- which IS decidable from the "
        "declaration, and is usually false"
    ),
    "substructuring-residual": (
        "L2/C3's ||S_M a* - chi_M|| / beta, which is a different criterion on a "
        "different branch and is not comparable with the three above"
    ),
}


@dataclass
class MeasuredConstants:
    """Bound constants a measurement has supplied, with the provenance that makes
    them usable.

    **W45, opened by the first real measurement and closed here.** Before this the
    compiler hard-coded its unmeasured list and consulted nothing, so a constant
    could be measured, written down and published, and the compile would still
    report it missing and type every claim UNTYPED. There was an emit path and no
    ingest path.

    Every field is optional and unset means unmeasured -- never a default. The
    provenance fields are not decoration: a constant measured at one state, one
    scheme and one depth is not the same constant somewhere else, so a value
    without them cannot be checked against the run that uses it.
    """

    L: float | None = None
    L_stderr: float | None = None
    tau: float | None = None
    sigma: float | None = None
    gamma: float | None = None
    C_mu: float | None = None
    norm_A: float | None = None
    p_decline: float | None = None
    #: **L2/C2**, added 2026-08-28 when the decomposition criterion was derived.
    #: ``|| sum_i chi_i |D_i| ||`` over ONE exchange interval, relative, with
    #: ``D_i = E_i R_i - R_i E`` the restriction defect.  It is the cut's own
    #: contribution to the composed defect, and declaring it is what lets L2 stop
    #: decertifying the decomposition policy -- exactly the W45 ingest path, one
    #: constant further on.  Its provenance is the same ``probe_state`` /
    #: ``scheme`` / ``depth`` triple as the rest: a bound measured at one state
    #: and one exchange cadence is not the same bound at another.
    cut_defect_bound: float | None = None
    #: **W58, closed 2026-08-30.**  WHICH of the two definitions the number above
    #: was measured by, as a value from `CUT_DEFECT_FORMS`.  The quantity has two
    #: inequivalent readings -- ``|| sum_i chi_i |D_i| ||``, which needs a
    #: monolithic reference exactly as ``tau`` does, and the reference-free
    #: neighbour disagreement, which does not -- and they are **provably equal
    #: only when the agents' contaminated sets are pairwise disjoint**.  On the
    #: four-window tiling they are not, and the surrogate agreed to 1.00007
    #: anyway; agreement without the hypothesis is luck, and luck does not
    #: transfer to a bigger tiling.  Declaring the value without the form is what
    #: L2/C2 now decertifies: it is a number whose reader cannot tell what was
    #: measured, which is the silent-wrongness class.
    cut_defect_bound_form: str | None = None
    #: **W86, measured 2026-08-29.**  The lag distance ``sigma`` was measured at,
    #: as a NUMBER rather than a phrase inside ``probe_state``.  §16.5 states the
    #: reason in its own words -- *"a sigma quoted without its lag is not a
    #: number"* -- and then quoted one, in prose nothing can read.
    #:
    #: What the measurement says this field can and cannot buy, walking
    #: `thermal_seam`'s reference trajectory for five macro-steps:
    #:
    #:   the SLOPE is a property of the seam.  ``d sigma / d(uniform lag)`` is
    #:   3.45001e-2 to 3.45013e-2 across all five steps -- constant to five
    #:   digits, so it transfers.
    #:
    #:   the LAG is a property of the RUN.  The interface drifts 5.17e-4 to
    #:   1.77e-3 K per step, a 3.43x range, non-monotone, and nothing at compile
    #:   time knows it because nothing has stepped yet.  So W77's fix does NOT
    #:   transfer: the probe had already run when `probe_state` was derived from
    #:   it, and the trajectory has not.
    #:
    #:   and sigma is not a function of the SCALAR lag at all.  At drifts within
    #:   8% of each other (1.087e-3 against 1.174e-3 K) sigma differs by 1.48x,
    #:   and over the five steps it ranges 7.70x.  A uniform-shift extrapolation
    #:   ``slope x lag`` over-predicts the run's own sigma by 1.20x to 4.38x and
    #:   does not track it.  sigma's argument is the lag PROFILE, and a scalar
    #:   does not determine it.
    #:
    #: So this field is **provenance, not a certificate**: comparing it against
    #: the lag a run actually carries can falsify a quoted sigma and can never
    #: confirm one.  `multiphysics.check_sigma_lag` says so at the point of use,
    #: and `solve.rollout` derives the run's own lag rather than trusting this.
    sigma_lag: float | None = None

    probe_state: str = "unspecified"
    scheme: str = "unspecified"
    depth: int = 0
    source: str = ""

    def supplied(self) -> set[str]:
        return {k for k in ("L", "tau", "sigma", "gamma", "C_mu", "norm_A",
                            "p_decline", "cut_defect_bound")
                if getattr(self, k) is not None}

    def as_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in
             ("L", "L_stderr", "tau", "sigma", "sigma_lag", "gamma", "C_mu",
              "norm_A", "p_decline", "cut_defect_bound")}
        d.update(cut_defect_bound_form=self.cut_defect_bound_form,
                 probe_state=self.probe_state, scheme=self.scheme,
                 depth=self.depth, source=self.source)
        return d


@dataclass
class CaseGraph:
    """A complete case study, as declarations.

    This is the whole input to the compiler.  A new case study is an instance of
    this class and nothing else.
    """

    name: str
    agents: list[Agent]
    connections: list[Connection]
    decomposition: Decomposition = Decomposition.OVERLAPPING
    #: delta, for the overlapping case.  **W189:** a ``dict`` keyed by REGION
    #: files one overlap per overlapping region -- see `per_region` -- and a
    #: bare number is the graph's own, which is what every graph written before
    #: 2026-09-10 meant and still means.
    overlap: float | dict[str, float] | None = None
    #: The same overlap in CELLS. The halo rule (R10) compares it against the
    #: agents' domain of dependence, which is counted in cells, and a physical
    #: delta cannot be converted without the agents' grid spacing -- so it is
    #: declared rather than derived, and the check decertifies when it is absent.
    #: **W189:** per region as a ``dict``, exactly like ``overlap``; the halo
    #: rule then checks each overlapping region against its own count.
    overlap_cells: int | dict[str, int] | None = None
    global_fields: list[GlobalField] = field(default_factory=list)
    topology_events: list[DeclaredTopologyEvent] = field(default_factory=list)
    #: Declared vertices where 3+ subdomains meet.
    #:
    #: **W162, 2026-09-09: three states, not two.** ``None`` is UNDECLARED and
    #: falls through to `detected_cross_points`' adjacency proxy; a non-empty
    #: tuple names them; and ``()`` is the declaration that this graph HAS
    #: none, which was previously unsayable -- an empty tuple is falsy, so it
    #: fell through to detection exactly as an absent declaration did. The
    #: distinction is not cosmetic: on a circuit the proxy is wrong, and
    #: `cooling_loop.build(n_legs=3)` was refused at `L2/I2/G1` for a
    #: multi-valued shared cell that does not exist, with no way for the graph
    #: to say so.
    cross_points: tuple[str, ...] | None = None
    primal_cross_point_dofs: tuple[str, ...] = () # the FETI-DP / BDDC treatment, if applied
    #: **W163.** The composed gain of each directed cycle in the seam graph, one
    #: entry per cycle, matched to `directed_cycles()` by agent SET so that a
    #: declaration does not have to guess which rotation the detector reports.
    #: Empty means undeclared, and the rule decertifies rather than assuming.
    loop_gains: tuple[DeclaredLoopGain, ...] = ()
    macro_dt: float | None = None
    #: R9. Which flux the seams match; see `FluxMatching`. Only consulted when
    #: the graph is multirate, because across one clock the two agree exactly.
    flux_matching: FluxMatching = FluxMatching.POINTWISE
    #: The assembly.  A bare `assembly.PartitionOfUnity` or
    #: `assembly.GridPartitionOfUnity` is the blend alone; an
    #: `assembly.ProjectedAssembly` is the blend PLUS the composition layer's
    #: constraint projection, which is what L6/C2 requires and R12 checks (W100).
    #: One field rather than two, because the two halves are one step and a graph
    #: that could declare them apart could declare a projection its assembly does
    #: not apply.
    #:
    #: **W189, 2026-09-10: one per OVERLAPPING REGION, or one for the graph.**  A
    #: ``dict`` keyed by region -- a governing_family with more than one agent,
    #: exactly `regions()`' keys -- files each partition under the region whose
    #: cover it weights; a bare object is the graph's own, which is what every
    #: graph written before this meant and still means.  The shape was censused
    #: over all forty constructible graphs before it was chosen: every one of the
    #: 22 that carries a partition has exactly ONE overlapping region, and the
    #: one graph with two overlapping regions (`rocket`) carries none -- so the
    #: map is sparse, and a non-overlapping or uncut region has nothing to file.
    #: The key is DECLARED rather than derived, because neither direction
    #: derives: in 12 of the 22 the region also holds agents the partition does
    #: not blend (a suspension, rotor disks -- lumped closures of the fluid's own
    #: family), and in 2 the partition's subdomains are not agent ids at all
    #: (`channel_ns`, W191).  `region_declaration_report` checks what it can.
    partition_of_unity: Any | dict[str, Any] | None = None   # see assembly.PartitionOfUnity
    #: W45: measured bound constants, with provenance. Unset fields stay unmeasured.
    measured: MeasuredConstants | None = None
    note: str = ""

    def __post_init__(self) -> None:
        ids = [a.agent_id for a in self.agents]
        if len(set(ids)) != len(ids):
            raise GraphError(f"duplicate agent ids in case {self.name!r}: {ids}")
        seams = [c.seam_id for c in self.connections]
        if len(set(seams)) != len(seams):
            raise GraphError(f"duplicate seam ids in case {self.name!r}: {seams}")
        known = set(ids)
        for c in self.connections:
            for agent_id, port_name in (c.a, c.b):
                if agent_id not in known:
                    raise GraphError(f"seam {c.seam_id!r} names unknown agent {agent_id!r}")
                self.agent(agent_id).port(port_name)   # raises if the port is undeclared

    # -- lookups -----------------------------------------------------------

    def agent(self, agent_id: str) -> Agent:
        for a in self.agents:
            if a.agent_id == agent_id:
                return a
        raise GraphError(f"case {self.name!r} has no agent {agent_id!r}")

    def connection(self, seam_id: str) -> Connection:
        for c in self.connections:
            if c.seam_id == seam_id:
                return c
        raise GraphError(f"case {self.name!r} has no seam {seam_id!r}")

    @property
    def K(self) -> int:
        """The number of agents. T_abs degrades linearly in this."""
        return len(self.agents)

    @property
    def assembly_projection(self):
        """The constraint projection the assembly declares, or None (W100).

        Duck-typed off `partition_of_unity` rather than a second field, so a
        graph cannot declare a projection its assembly does not apply.  L6/C2
        and R12 ask this one question and this is where they ask it.

        **W189.**  A graph that files its partitions per region has one
        projection per region and no single answer, so this RAISES rather than
        returning None -- which would read as "no projection declared" on a
        graph whose regions may each declare one.  Ask `assembly_projection_for`.
        """
        if isinstance(self.partition_of_unity, dict):
            raise GraphError(
                f"case {self.name!r} files its partitions of unity per region "
                f"({sorted(self.partition_of_unity)}), so it has one assembly "
                "projection per region and no graph-level one: ask "
                "assembly_projection_for(region). Returning None here would say "
                "no projection is declared on a graph whose regions may each "
                "declare one (W189)")
        return getattr(self.partition_of_unity, "projection", None)

    def connections_of(self, agent_id: str) -> list[Connection]:
        return [c for c in self.connections if agent_id in c.agents]

    def neighbours(self, agent_id: str) -> set[str]:
        return {c.other(agent_id) for c in self.connections_of(agent_id)}

    def open_ports(self) -> list[tuple[str, str]]:
        """Declared ports nothing is connected to.

        An unconnected port is not an absence; it is an open port with a
        measurable power flow through it, and that is the mechanism by which the
        roadmap grows -- each new subsystem is a port connection.
        """
        connected = set()
        for c in self.connections:
            connected.add(c.a)
            connected.add(c.b)
        out = []
        for a in self.agents:
            for p in a.capabilities.ports:
                if (a.agent_id, p.name) not in connected:
                    out.append((a.agent_id, p.name))
        return out

    # -- W171: the decomposition axis, per REGION rather than per graph ----

    def seam_axis(self, connection: Connection) -> "Decomposition | None":
        """The axis of the cut this seam IS, or ``None`` when it is not a cut.

        **W171, derived from Tier 40's measurement rather than declared.**  The
        row's own `[AI Inference]` said the natural carrier is the AGENT, and
        that was measured FALSE: a cut is a **relation among several agents**,
        not a property of one -- two agents of one family cut two different ways
        is one fact about one region and is unrepresentable on either agent --
        and an agent that is not cut has no axis at all.

        The carrier that works is the **connection**, and the field is already
        declared.  Two clauses, in order:

        1.  **Different `governing_family` on the two sides: not a cut.**  Gamma
            is a physical boundary between two regions, not a face a
            decomposition created, so neither axis applies and the overlapping
            mechanism has no subject there.  This is `_decomposition_cuts`'
            own predicate (W114), read on the seam instead of on the agent.
        2.  **Same family: the seam's own ``cut_axis``, or the graph's.**
            Tier 40 proposed deriving this from ``geometrically_coincident`` and
            measured the proposal on four graphs.  Censused over all eleven
            constructible case graphs, per seam, it **does not separate the
            axis** -- ``same_family=True, coincident=True`` appears under both
            axes, and so does ``same_family=False, coincident=True`` -- because
            coincidence is a property of the two sides' *discretizations* and
            the axis is a property of their *domains*.  So it is declared, and
            ``None`` inherits the graph's, which is what every graph written
            before W171 means.

        Nothing checks the declaration, exactly as nothing checks
        ``geometrically_coincident``; `region_axis_conflicts` reports a region
        whose own seams disagree rather than resolving it.
        """
        fa = self.agent(connection.a[0]).capabilities.governing_family or ""
        fb = self.agent(connection.b[0]).capabilities.governing_family or ""
        if fa != fb:
            return None
        return connection.cut_axis or self.decomposition

    def regions(self) -> dict[str, list[str]]:
        """The CUT regions, keyed by governing family, each a list of agent ids.

        A region is a family with more than one agent in this graph -- exactly
        `_decomposition_cuts`' cut set, grouped instead of flattened.  A sole
        agent of its family owns its whole domain and is in no region.
        """
        by_family: dict[str, list[str]] = {}
        for a in self.agents:
            by_family.setdefault(a.capabilities.governing_family or "", []).append(
                a.agent_id)
        return {fam: ids for fam, ids in by_family.items() if len(ids) > 1}

    def region_axes(self) -> dict[str, "Decomposition"]:
        """One axis per cut region, derived from that region's own seams.

        A region whose intra-family seams disagree is reported through
        `region_axis_conflicts` rather than resolved here; a region with no
        intra-family seam to derive from falls back to the graph's declared
        ``decomposition``, which is what every single-axis graph means.
        """
        out: dict[str, "Decomposition"] = {}
        for fam, ids in self.regions().items():
            members = set(ids)
            seen = {
                self.seam_axis(c)
                for c in self.connections
                if c.a[0] in members and c.b[0] in members
            } - {None}
            out[fam] = seen.pop() if len(seen) == 1 else self.decomposition
        return out

    def region_axis_conflicts(self) -> dict[str, list[str]]:
        """Regions whose own seams do not agree about the axis.

        Reported, never resolved.  One region cut two ways is a declaration
        defect and the compiler should say so rather than pick.
        """
        out: dict[str, list[str]] = {}
        for fam, ids in self.regions().items():
            members = set(ids)
            seen: dict[str, list[str]] = {}
            for c in self.connections:
                if c.a[0] in members and c.b[0] in members:
                    axis = self.seam_axis(c)
                    if axis is not None:
                        seen.setdefault(axis.value, []).append(c.seam_id)
            if len(seen) > 1:
                out[fam] = sorted(s for v in seen.values() for s in v)
        return out

    def agent_axis(self, agent_id: str) -> "Decomposition | None":
        """The axis of the region this agent is in; ``None`` when it is not cut."""
        fam = self.agent(agent_id).capabilities.governing_family or ""
        return self.region_axes().get(fam)

    def decomposition_axes(self) -> frozenset:
        """Every axis present in this graph.

        One element for every graph written before W171, which is the control
        that says the derivation did not move any of them.  Two is a union graph
        -- a tiling and a circuit in one compile -- which is what rung 9 needs
        and what `decomposition` as a single field could not express.
        """
        return frozenset(self.region_axes().values())

    # -- W189: the overlapping mechanism, per REGION rather than per graph --

    #: The three fields that may be declared per region.  Not a dataclass field
    #: (it carries no annotation): a constant of the schema, which the compiler
    #: and `emit` read rather than restating.
    PER_REGION_FIELDS = ("partition_of_unity", "overlap", "overlap_cells")

    def per_region(self, name: str) -> bool:
        """Is ``name`` -- one of `PER_REGION_FIELDS` -- declared per region?

        **W189.**  A ``dict`` is the per-region form, keyed by region; anything
        else, ``None`` included, is the graph-scoped form every graph written
        before W189 uses.  Asking about any other field raises, because a silent
        ``False`` would read as "graph-scoped" for a field that has no per-region
        form at all.
        """
        if name not in self.PER_REGION_FIELDS:
            raise GraphError(
                f"{name!r} has no per-region form; the fields that do are "
                f"{self.PER_REGION_FIELDS} (W189)")
        return isinstance(getattr(self, name), dict)

    def declares_per_region(self) -> bool:
        """Does this graph declare ANY of the three per region?

        The switch every region-scoped rule reads.  A graph that declares none
        takes exactly the path it always took, which is what keeps every
        artifact written before W189 byte-identical -- and that is measured, over
        forty graphs, not assumed.
        """
        return any(self.per_region(n) for n in self.PER_REGION_FIELDS)

    def region_value(self, name: str, region: str) -> Any:
        """The value of ``name`` filed under ``region``, or None.

        ``None`` for the graph-scoped form too, deliberately.  A partition the
        graph declares as its own is not ANY region's, and Tier 44's union is the
        case that shows why: its one partition covered only the tiling and had to
        be declared as the whole graph's.  Answering a region query with it
        would be guessing which region it was meant for.
        """
        value = getattr(self, name)
        return value.get(region) if isinstance(value, dict) else None

    def partition_for(self, region: str) -> Any:
        """The partition of unity filed under ``region`` (W189)."""
        return self.region_value("partition_of_unity", region)

    def overlap_for(self, region: str) -> Any:
        """The overlap filed under ``region`` (W189)."""
        return self.region_value("overlap", region)

    def overlap_cells_for(self, region: str) -> Any:
        """The overlap-cell count filed under ``region`` (W189)."""
        return self.region_value("overlap_cells", region)

    def assembly_projection_for(self, region: str) -> Any:
        """The constraint projection the partition filed under ``region`` declares."""
        return getattr(self.partition_for(region), "projection", None)

    def region_declaration_report(self) -> dict[str, dict[str, dict[str, Any]]]:
        """What every per-region key names, reported and never resolved. **W189.**

        For each field declared per region and each key in it: whether the key
        is an OVERLAPPING cut region of this graph and, if not, which of three
        ways it is not -- no agent of that family (``no-such-region``), a family
        with one agent, which owns its whole domain so nothing was cut
        (``not-a-cut-region``), or a region whose own seams declare it
        NON-overlapping, where a substructuring cut shares no cells and there is
        no cover to weight (``non-overlapping-region``).  The axis read is the
        DECLARED one, `region_axes`, and not R2's lift: a partition is certified
        on its declaration whatever view the interface problem is posed in.

        A partition also carries its subdomain NAMES, and they are read against
        the agents, which is decidable only where they ARE agent ids:

          * ``foreign``   -- subdomains that are agents of ANOTHER family, whose
                             state the blend would weight into this region;
          * ``unmatched`` -- subdomains that name no agent, so whether the
                             partition covers this region cannot be decided;
          * ``unblended`` -- agents of the region the partition does not blend.
                             Disclosure, not a defect: censused before this was
                             written, 12 of the 22 partitions in the package leave
                             out a lumped closure of the fluid's own family (a
                             suspension, rotor disks) that W114's family proxy
                             puts in the region.

        ``unmatched`` was not hypothetical when this was written: `channel_ns`'s
        partition names `window_ns`'s windows (W00..W11) while its agents are
        C00..C11.  Nothing read the names, so nothing noticed (W191).

        A graph that declares nothing per region returns an empty dict.
        """
        family_of = {a.agent_id: a.capabilities.governing_family or ""
                     for a in self.agents}
        counts: dict[str, int] = {}
        for fam in family_of.values():
            counts[fam] = counts.get(fam, 0) + 1
        regions = self.regions()
        axes = self.region_axes()
        out: dict[str, dict[str, dict[str, Any]]] = {}
        for name in self.PER_REGION_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, dict):
                continue
            rows: dict[str, dict[str, Any]] = {}
            for key, item in value.items():
                row: dict[str, Any] = {
                    "region_agents": list(regions.get(key, [])),
                    "axis": axes[key].value if key in axes else None,
                }
                n = counts.get(key, 0)
                if n == 0:
                    row["status"] = "no-such-region"
                elif n == 1:
                    row["status"] = "not-a-cut-region"
                    row["sole_agent"] = next(a for a, f in family_of.items()
                                             if f == key)
                elif axes.get(key) is not Decomposition.OVERLAPPING:
                    row["status"] = "non-overlapping-region"
                else:
                    row["status"] = "region"
                if name == "partition_of_unity" and hasattr(item, "subdomains"):
                    subs = [str(s) for s in item.subdomains()]
                    row["subdomains"] = subs
                    row["foreign"] = {s: family_of[s] for s in subs
                                      if s in family_of and family_of[s] != key}
                    row["unmatched"] = sorted(s for s in subs if s not in family_of)
                    row["unblended"] = sorted(set(regions.get(key, [])) - set(subs))
                rows[str(key)] = row
            out[name] = rows
        return out

    # -- structural properties the compiler gates on -----------------------
    # -- structural properties the compiler gates on -----------------------

    def governing_families(self) -> dict[str, str | None]:
        return {a.agent_id: a.capabilities.governing_family for a in self.agents}

    def cross_family_seams(self) -> list[Connection]:
        """Seams whose two sides declare different governing families.

        E3 is decidable at compile time and it is a string comparison, not an
        analysis.
        """
        out = []
        for c in self.connections:
            fa = self.agent(c.a[0]).capabilities.governing_family
            fb = self.agent(c.b[0]).capabilities.governing_family
            if fa is not None and fb is not None and fa != fb:
                out.append(c)
        return out

    def native_steps(self) -> dict[str, float | None]:
        return {a.agent_id: a.capabilities.dt_native for a in self.agents}

    def is_multirate(self) -> bool:
        steps = [v for v in self.native_steps().values() if v is not None]
        return len(set(steps)) > 1

    def moving_ports(self) -> list[tuple[str, PortDecl]]:
        out = []
        for a in self.agents:
            for p in a.capabilities.ports:
                if p.motion_class is not MotionClass.STATIC:
                    out.append((a.agent_id, p))
        return out

    @property
    def cross_points_declared(self) -> bool:
        """W162: has this graph SAID anything about its cross-points?

        Distinct from having any.  ``()`` is a declaration of none and ``None``
        is silence, and the two used to be the same value.
        """
        return self.cross_points is not None

    def detected_cross_points(self) -> list[tuple[str, ...]]:
        """Structural proxy for vertices where three or more subdomains meet.

        Declared cross points are authoritative, **including a declaration that
        there are none** (W162): ``cross_points=()`` returns an empty list and
        the proxy does not run, where ``cross_points=None`` is silence and it
        does.

        Absent a declaration this returns the 3-cliques of the connection graph,
        which is a *proxy*: three pairwise-connected subdomains normally share a
        vertex, but the graph alone cannot prove it and a geometric
        decomposition can produce a cross-point with no triangle.  The compiler
        records it as a proxy and leans on the real detector -- the probe's
        null-space count, where any excess null direction beyond the declared
        count is a defect.

        **And it fails in the other direction on a CIRCUIT**, which is W162.
        Three legs of a coolant loop are pairwise adjacent and share no point:
        their three seams are three distinct planes, and the triangle is in the
        FLOW topology rather than in the geometry.  A tiling's triangle and a
        circuit's triangle are the same object in the adjacency and different
        objects in space, and nothing in the adjacency tells them apart -- so
        the escape has to be a declaration, and now there is one.
        """
        if self.cross_points is not None:
            return [(cp,) for cp in self.cross_points]
        adjacency = {a.agent_id: self.neighbours(a.agent_id) for a in self.agents}
        triangles: list[tuple[str, ...]] = []
        for x, y, z in itertools.combinations(sorted(adjacency), 3):
            if y in adjacency[x] and z in adjacency[x] and z in adjacency[y]:
                triangles.append((x, y, z))
        return triangles

    def directed_cycles(self, max_steps: int = 200_000
                        ) -> list[tuple[str, ...]]:
        """Elementary directed cycles of the seam digraph. **W163.**

        Each connection contributes one edge ``a[0] -> b[0]``, in the order the
        connection itself declares.  A cycle is a closed walk visiting each
        agent once, reported from its lowest-named agent so that one cycle has
        one representation.

        **This is a PROXY for "the agents form a feedback loop", and it was
        censused before it was trusted.**  Over every graph in the package it
        returns exactly one cycle on `cooling_loop` (COLD -> PASS -> HOT -> RAD)
        and exactly one on `powertrain` (BATT -> INV -> MGU -> BUS), zero on
        both of their ``close_loop=False`` controls, and zero on `front_wing`,
        `ground_effect`, `wing_fsi` and `thermal_seam`.  A tiling survives it
        because a tiling orients its seams monotonically along the grid axes --
        ``xhi -> xlo``, ``yhi -> ylo`` -- so its digraph is a grid poset, which
        is acyclic.  What the proxy cannot see is a graph whose ``(a, b)`` order
        is arbitrary rather than meaningful; there is none here, and the rule's
        message says the detector is structural rather than physical.

        **``max_steps`` is a budget and not a cap on the output**, and the
        distinction is the whole of it.  The number of elementary cycles in a
        digraph is factorial in the worst case -- found by stress-testing this
        function rather than by reading it, on a complete digraph where it does
        not return at ten nodes -- and every graph in this package is sparse
        enough to finish in microseconds.  Past the budget it RAISES rather than
        returning what it has, because a partial list is worse than none: a rule
        reasoning over some of a graph's cycles is silently reasoning about a
        different graph, which is the failure class this whole compiler exists
        to refuse.  `_r13_directed_cycle` catches it and decertifies.
        """
        adjacency: dict[str, list[str]] = {}
        for c in self.connections:
            adjacency.setdefault(c.a[0], []).append(c.b[0])
        found: list[tuple[str, ...]] = []
        seen: set[frozenset] = set()
        budget = [int(max(1, max_steps))]

        def walk(start: str, node: str, path: list[str]) -> None:
            for nxt in adjacency.get(node, ()):
                budget[0] -= 1
                if budget[0] <= 0:
                    raise CycleEnumerationBudget(
                        f"enumerating the elementary directed cycles of "
                        f"{self.name!r} exceeded {max_steps} search steps over "
                        f"{len(adjacency)} agents and {len(self.connections)} "
                        "seams. The count of elementary cycles is factorial in "
                        "the worst case, so this is a budget and not a bug -- "
                        "but a PARTIAL list is worse than none, because a rule "
                        "reasoning over some of a graph's cycles is silently "
                        "reasoning about a different graph. Declare the cycles "
                        "the graph has, or raise max_steps deliberately"
                    )
                if nxt == start and len(path) >= 2:
                    key = frozenset(path)
                    if key not in seen:
                        seen.add(key)
                        found.append(tuple(path))
                elif nxt not in path and nxt > start:
                    walk(start, nxt, path + [nxt])

        for a in sorted(adjacency):
            walk(a, a, [a])
        return found

    def declared_loop_gain(self, cycle: tuple[str, ...]
                           ) -> "DeclaredLoopGain | None":
        """The declaration for this cycle, matched by agent SET (W163).

        By set rather than by sequence because a cycle has no distinguished
        first agent -- that is the property the whole rule is about -- so a
        declaration written in one rotation must match a detection reported in
        another.
        """
        want = frozenset(cycle)
        for g in self.loop_gains:
            if frozenset(g.agents) == want:
                return g
        return None

    def seam_transfer(self, connection: Connection) -> SeamTransfer | None:
        """Assemble the seam's declared transfer from the connection or the ports.

        Returns None when the declaration is incomplete; the caller (L3) turns
        that into a refusal citing a *missing declaration* rather than an unbuilt
        mechanism.
        """
        space = connection.space
        prolongations = dict(connection.prolongations)
        for agent_id, port_name in (connection.a, connection.b):
            port = self.agent(agent_id).port(port_name)
            if space is None and port.interface_space is not None:
                space = port.interface_space
            if agent_id not in prolongations and port.prolongation is not None:
                prolongations[agent_id] = port.prolongation
        if space is None or len(prolongations) != 2:
            return None
        port_a = self.agent(connection.a[0]).port(connection.a[1])
        try:
            return SeamTransfer(
                seam_id=connection.seam_id,
                port_type=connection.port_type,
                space=space,
                prolongations=prolongations,
                passengers=port_a.passengers,
            )
        except TransferDeclarationError:
            return None

    def effective_resolutions(self, connection: Connection) -> dict[str, int | None]:
        return {
            agent_id: self.agent(agent_id).port(port_name).effective_resolution
            for agent_id, port_name in (connection.a, connection.b)
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "K": self.K,
            "decomposition": self.decomposition.value,
            "overlap": self.overlap,
            "overlap_cells": self.overlap_cells,
            "measured": None if self.measured is None else self.measured.as_dict(),
            "agents": [a.capabilities.as_dict() for a in self.agents],
            "connections": [
                {
                    "seam_id": c.seam_id,
                    "a": list(c.a),
                    "b": list(c.b),
                    "port_type": c.port_type.value,
                    "orientation": c.orientation,
                    "effort_normal": c.effort_normal,
                    "enforced": c.enforced,
                }
                for c in self.connections
            ],
            "global_fields": [g.name for g in self.global_fields],
            "topology_events": [e.event_id for e in self.topology_events],
            "open_ports": [list(p) for p in self.open_ports()],
        }

    def __iter__(self) -> Iterator[Agent]:
        return iter(self.agents)
