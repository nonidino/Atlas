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
    """

    name: str
    applies_to: tuple[str, ...] = ()   # empty means every agent
    note: str = ""


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
    overlap: float | None = None                  # delta, for the overlapping case
    #: The same overlap in CELLS. The halo rule (R10) compares it against the
    #: agents' domain of dependence, which is counted in cells, and a physical
    #: delta cannot be converted without the agents' grid spacing -- so it is
    #: declared rather than derived, and the check decertifies when it is absent.
    overlap_cells: int | None = None
    global_fields: list[GlobalField] = field(default_factory=list)
    topology_events: list[DeclaredTopologyEvent] = field(default_factory=list)
    cross_points: tuple[str, ...] = ()            # declared vertices where 3+ subdomains meet
    primal_cross_point_dofs: tuple[str, ...] = () # the FETI-DP / BDDC treatment, if applied
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
    partition_of_unity: Any | None = None         # see assembly.PartitionOfUnity
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
        """
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

    def detected_cross_points(self) -> list[tuple[str, ...]]:
        """Structural proxy for vertices where three or more subdomains meet.

        Declared cross points are authoritative.  Absent a declaration this
        returns the 3-cliques of the connection graph, which is a *proxy*: three
        pairwise-connected subdomains normally share a vertex, but the graph
        alone cannot prove it and a geometric decomposition can produce a
        cross-point with no triangle.  The compiler records it as a proxy and
        leans on the real detector -- the probe's null-space count, where any
        excess null direction beyond the declared count is a defect.
        """
        if self.cross_points:
            return [(cp,) for cp in self.cross_points]
        adjacency = {a.agent_id: self.neighbours(a.agent_id) for a in self.agents}
        triangles: list[tuple[str, ...]] = []
        for x, y, z in itertools.combinations(sorted(adjacency), 3):
            if y in adjacency[x] and z in adjacency[x] and z in adjacency[y]:
                triangles.append((x, y, z))
        return triangles

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
