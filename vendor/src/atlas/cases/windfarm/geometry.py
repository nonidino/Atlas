"""The eight agents and the fifteen typed interfaces, as data.

Phase W1. Everything here is transcribed from `spec-wind-farm-wake-atlas-0.1`
sections 3 and 5 and verified without a network anywhere in sight; the point of
the phase is that a geometry bug should be impossible to carry into a
measurement. `verify()` runs at import, following the Phase-1 scaffold's rule
that a broken partition fails the moment anything imports Atlas rather than
later, in one test nobody ran.

Coordinates are nondimensional: x streamwise and y lateral, both in rotor
diameters, origin at turbine 1. The domain is `[-6, 18] x [-4, 4]`.

Why the partition is expressed as half-open boxes
-------------------------------------------------
The rocket's `contains` used closed sets and then took care to test only cell
centres, which can never land on a shared face. Here every interface is
axis-aligned and lands exactly on the token lattice, so closed sets would make
`(0, 1)` belong to both `I` and `N` and there would be no way to state
disjointness as an identity. The boxes below are half-open on exactly the faces
they share with a downstream neighbour, which makes

    exactly one agent contains every point of Omega

true pointwise, with no tolerance and no sampling caveat. `contains_closed` is
kept for the questions that genuinely are about closures.

The notch
---------
`N` and `W` are their bounding boxes minus the rotor strip that sits inside
them. At the continuum level that is exact. At `N`'s token spacing of 0.25 the
notch is 0.1 wide -- **sub-token** -- so a boolean cell-centre mask would not
see it at all and the rotor's cells would be counted twice. Agent masks are
therefore *fractional*: `AgentGrid.frac` is the area fraction of each cell lying
inside the agent, and the discrete coverage identity `sum over agents of
frac*area == |Omega|` holds to quadrature accuracy. This is recorded rather than
smoothed over: it is a real consequence of the spec's own token spacing.

Generalizing to N turbines
---------------------------
Everything below was originally two literal tuples, `AGENTS` and `INTERFACES`,
hand-transcribed for exactly two turbines at `x = 0` and `x = 7`. `build_case`
now generates that same topology for an arbitrary number of turbines in a line,
spaced `TURBINE_SPACING` D apart, by literally repeating the pattern the
original file's own comments describe: `I -> R1 -> N -> F -> R2 -> ... -> W`,
with a near/far wake pair inserted between every consecutive pair of rotors and
collapsing into the single exit-wake `W` only after the last one.

The module-level `AGENTS`, `INTERFACES`, `OPEN_PORTS`, `UNDECLARED_BY_DESIGN`
and `DOMAIN` below are simply `build_case(2)`'s output -- so the N=2 path is
not "generalized code that happens to reduce to the old numbers", it is the
literal old numbers, produced by the one function that also produces every
other N. `owner()`, `materialized_pairs()` and `verify()` keep their original
signatures and, called with no arguments, their original exact behaviour;
they now also accept explicit `agent`/`priority`/`domain`/... keywords so
`build_geometry_case` can run the identical checks against a different N.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ...ports.types import PortType, port

EPS = 1e-9

#: `spec-wind-farm-wake-atlas-0.1` section 2.4
DOMAIN = (-6.0, 18.0, -4.0, 4.0)          # x0, x1, y0, y1 -- filled in below
X_T1, X_T2 = 0.0, 7.0                     # turbine stations
ROTOR_HALFSPAN = 0.5                      # rotors span |y| <= 0.5, so A = 1
DISK_THICKNESS = 0.1                      # Delta_d, spec section 3
WAKE_HALFWIDTH = 2.0                      # wake corridor |y| <= 2
NEAR_FAR_SPLIT = 3.0                      # near/far wake regime boundary
TURBINE_SPACING = 7.0                     # spec's inter-turbine spacing, in D
NEAR_WAKE_LEN = 3.0                       # x-extent of a near-wake segment (N)
EXIT_BUDGET = 11.0                        # near+far+exit budget after the LAST turbine


# --------------------------------------------------------------------------
# regions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    """An axis-aligned box, half-open on the faces named in `open_faces`.

    `open_faces` is any subset of `{'xhi', 'yhi', 'ylo'}` -- the face is
    excluded from the box, so the neighbour on that side owns it."""
    x0: float
    x1: float
    y0: float
    y1: float
    open_faces: frozenset[str] = frozenset()

    def contains(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        m = x >= self.x0
        m &= (x < self.x1) if "xhi" in self.open_faces else (x <= self.x1)
        m &= (y > self.y0) if "ylo" in self.open_faces else (y >= self.y0)
        m &= (y < self.y1) if "yhi" in self.open_faces else (y <= self.y1)
        return m

    def contains_closed(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return ((x >= self.x0 - EPS) & (x <= self.x1 + EPS)
                & (y >= self.y0 - EPS) & (y <= self.y1 + EPS))

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


@dataclass(frozen=True)
class AgentRegion:
    """One agent: a box, minus any boxes carved out of it, on its own lattice."""
    name: str
    box: Box
    holes: tuple[Box, ...]
    spacing: tuple[float, float]        # token size (dx, dy) in D
    expert: str                         # 'fluid' | 'disk'
    role: str                           # chi_wake / chi_disk one-hot label
    tokens_declared: int                # spec section 3, for the +-10% gate

    def contains(self, x, y):
        m = self.box.contains(x, y)
        for h in self.holes:
            m &= ~h.contains(x, y)
        return m

    def contains_closed(self, x, y):
        return self.box.contains_closed(x, y)

    @property
    def area(self) -> float:
        return self.box.area - sum(h.area for h in self.holes)


@dataclass(frozen=True)
class Interface:
    """One declared typed edge and the curve that realizes it.

    The normal points from `src` to `dst`, always -- guide section 0.4. Get it
    wrong once and every residual in the case study is meaningless, so
    `verify()` tests it by stepping off the curve in both directions and asking
    `owner()` who is there, which cannot be fooled by a sign convention written
    down in a comment."""
    src: str
    dst: str
    axis: str                                  # 'x' -> curve is x = pos; 'y' -> y = pos
    pos: float
    spans: tuple[tuple[float, float], ...]     # branches along the other coordinate
    ports: tuple[str, ...]
    tangential: bool = False                   # for the record: a shear interface

    @property
    def key(self) -> tuple[str, str]:
        return (self.src, self.dst)

    def normal_using(self, owner_fn) -> tuple[float, float]:
        return (self._sign_using(owner_fn), 0.0) if self.axis == "x" else (0.0, self._sign_using(owner_fn))

    def _sign_using(self, owner_fn) -> float:
        """+1 when `dst` lies on the increasing side of the curve."""
        lo, hi = self.spans[0]
        mid = 0.5 * (lo + hi)
        p = (self.pos, mid) if self.axis == "x" else (mid, self.pos)
        probe = np.array([p[0] + (1e-6 if self.axis == "x" else 0.0),
                          p[1] + (0.0 if self.axis == "x" else 1e-6)])
        return 1.0 if owner_fn(probe[0:1], probe[1:2])[0] == self.dst else -1.0

    @property
    def normal(self) -> tuple[float, float]:
        return self.normal_using(owner)

    @property
    def _sign(self) -> float:
        return self._sign_using(owner)

    @property
    def length(self) -> float:
        return float(sum(hi - lo for lo, hi in self.spans))

    def sample(self, n_per_branch: int = 61) -> np.ndarray:
        """Points strictly interior to each branch (endpoints excluded: they are
        the corners where three agents meet, and belong to no single edge)."""
        pts = []
        for lo, hi in self.spans:
            s = np.linspace(lo, hi, n_per_branch + 2)[1:-1]
            c = np.full_like(s, self.pos)
            pts.append(np.stack([c, s], axis=1) if self.axis == "x"
                       else np.stack([s, c], axis=1))
        return np.concatenate(pts, axis=0)

    def port_types(self) -> tuple[PortType, ...]:
        return tuple(port(p) for p in self.ports)


_MA = ("MECH", "ADVEC")
_M = ("MECH",)


# --------------------------------------------------------------------------
# the generator: N turbines in a line, TURBINE_SPACING apart
# --------------------------------------------------------------------------
#
# Section 3's agent table for N=2 is exactly `I -> R1 -> N -> F -> R2 -> W`
# (plus the two bypass strips, run the full domain length). For N turbines the
# same pattern repeats: after each rotor `R_i` (i < N) comes a near-wake
# segment and a far-wake segment identical in shape to `N`/`F`, ending at the
# next rotor `R_{i+1}`; only after the *last* rotor does the near+far pattern
# collapse into the single `W` (exit-wake) region, which keeps the same 11D
# budget of near+far+exit that `W` has today after turbine 2 (`18 - 7 == 11`,
# i.e. `EXIT_BUDGET`).
#
# The first segment keeps the legacy names `N`/`F` (not `N1`/`F1`) so that
# `build_case(2)` reproduces today's literal `AGENTS`/`INTERFACES` tuples
# name-for-name, box-for-box, order-for-order -- checked directly (not just
# via `verify()`) by `tests/atlas/windfarm/test_geometry_generator.py`.
# Segments after the first turbine (only present for N >= 3) are named
# `N2`/`F2`, `N3`/`F3`, ...


def _segment_name(base: str, i: int) -> str:
    """`base='N', i=1` -> `'N'` (legacy name); `base='N', i=2` -> `'N2'`; etc."""
    return base if i == 1 else f"{base}{i}"


def build_case(n_turbines: int = 2, spacing: float = TURBINE_SPACING):
    """Build `(domain, agents, interfaces, open_ports, undeclared, extras)` for
    `n_turbines` turbines in a line, `spacing` D apart, turbine 1 at x=0.

    `build_case(2)` is the single source of truth for the original hand-written
    2-turbine geometry: this is the *only* place the topology is expressed."""
    if n_turbines < 1:
        raise ValueError(f"n_turbines must be >= 1, got {n_turbines}")
    positions = [i * spacing for i in range(n_turbines)]
    x1 = positions[-1] + EXIT_BUDGET
    domain = (-6.0, x1, -4.0, 4.0)

    agents: list[AgentRegion] = []
    interfaces: list[Interface] = []
    gs_fluid_order: list[str] = ["I"]
    field_owner_remap: dict[str, str] = {}
    undeclared: dict[tuple[str, str], float] = {}

    agents.append(AgentRegion("I", Box(-6.0, positions[0], -4.0, 4.0, frozenset({"xhi"})), (),
                              (0.5, 0.5), "fluid", "inflow", 192))
    first_downstream = "N" if n_turbines > 1 else "W"
    interfaces.append(Interface("I", "R1", "x", positions[0], ((-0.5, 0.5),), _MA))
    interfaces.append(Interface("I", first_downstream, "x", positions[0],
                                ((-2.0, -0.5), (0.5, 2.0)), _MA))
    interfaces.append(Interface("I", "B+", "x", positions[0], ((2.0, 4.0),), _MA))
    interfaces.append(Interface("I", "B-", "x", positions[0], ((-4.0, -2.0),), _MA))

    for i, x in enumerate(positions, start=1):
        rname = f"R{i}"
        rbox = Box(x, x + DISK_THICKNESS, -ROTOR_HALFSPAN, ROTOR_HALFSPAN)
        agents.append(AgentRegion(rname, rbox, (), (DISK_THICKNESS, 0.05), "disk", "disk", 20))

        if i < n_turbines:
            x_next = positions[i]
            near_name, far_name = _segment_name("N", i), _segment_name("F", i)
            near_x1 = x + NEAR_WAKE_LEN
            near_box = Box(x, near_x1, -WAKE_HALFWIDTH, WAKE_HALFWIDTH)
            far_box = Box(near_x1, x_next, -WAKE_HALFWIDTH, WAKE_HALFWIDTH, frozenset({"xhi"}))
            agents.append(AgentRegion(near_name, near_box, (rbox,), (0.25, 0.25),
                                      "fluid", "wake_near", 192))
            agents.append(AgentRegion(far_name, far_box, (), (0.25, 0.25),
                                      "fluid", "wake_far", 256))
            gs_fluid_order += [near_name, far_name]
            field_owner_remap[rname] = near_name
            undeclared[(rname, near_name)] = 2 * DISK_THICKNESS

            interfaces.append(Interface(rname, near_name, "x", x + DISK_THICKNESS,
                                        ((-0.5, 0.5),), _MA))
            interfaces.append(Interface(near_name, far_name, "x", near_x1,
                                        ((-2.0, 2.0),), _MA))
            interfaces.append(Interface(near_name, "B+", "y", WAKE_HALFWIDTH,
                                        ((x, near_x1),), _M, tangential=True))
            interfaces.append(Interface(near_name, "B-", "y", -WAKE_HALFWIDTH,
                                        ((x, near_x1),), _M, tangential=True))
            interfaces.append(Interface(far_name, "B+", "y", WAKE_HALFWIDTH,
                                        ((near_x1, x_next),), _M, tangential=True))
            interfaces.append(Interface(far_name, "B-", "y", -WAKE_HALFWIDTH,
                                        ((near_x1, x_next),), _M, tangential=True))
            next_name = f"R{i + 1}"
            interfaces.append(Interface(far_name, next_name, "x", x_next,
                                        ((-0.5, 0.5),), _MA))
            downstream_open = "W" if i + 1 == n_turbines else _segment_name("N", i + 1)
            interfaces.append(Interface(far_name, downstream_open, "x", x_next,
                                        ((-2.0, -0.5), (0.5, 2.0)), _MA))
        else:
            w_box = Box(x, x1, -WAKE_HALFWIDTH, WAKE_HALFWIDTH)
            agents.append(AgentRegion("W", w_box, (rbox,), (0.5, 0.5),
                                      "fluid", "wake_exit", 176))
            gs_fluid_order.append("W")
            field_owner_remap[rname] = "W"
            undeclared[(rname, "W")] = 2 * DISK_THICKNESS
            interfaces.append(Interface(rname, "W", "x", x + DISK_THICKNESS,
                                        ((-0.5, 0.5),), _MA))
            interfaces.append(Interface("W", "B+", "y", WAKE_HALFWIDTH, ((x, x1),),
                                        _M, tangential=True))
            interfaces.append(Interface("W", "B-", "y", -WAKE_HALFWIDTH, ((x, x1),),
                                        _M, tangential=True))

    # `tokens_declared` for B+/B- is the spec's N=2 table value (144), which is
    # only meaningful for the N=2 bypass length of 18 D. For other N the strip
    # runs the whole (N-dependent) domain length, so the declared count is
    # scaled by that length -- the same 8 tokens/D the N=2 table implies -- and
    # the +-10% gate in `verify()` still checks something real instead of being
    # vacuously satisfied or spuriously failing.
    bypass_len = x1 - positions[0]
    bypass_declared = int(round(144 * bypass_len / 18.0))
    agents.append(AgentRegion("B+", Box(positions[0], x1, WAKE_HALFWIDTH, 4.0,
                                        frozenset({"ylo"})), (),
                              (0.5, 0.5), "fluid", "bypass", bypass_declared))
    agents.append(AgentRegion("B-", Box(positions[0], x1, -4.0, -WAKE_HALFWIDTH,
                                        frozenset({"yhi"})), (),
                              (0.5, 0.5), "fluid", "bypass", bypass_declared))
    gs_fluid_order += ["B+", "B-"]

    open_ports = tuple((f"R{i}", "ROT") for i in range(1, n_turbines + 1))
    extras = {
        "positions": tuple(positions),
        "gs_order": tuple(gs_fluid_order),
        "field_owner_remap": field_owner_remap,
        "rotor_names": tuple(f"R{i}" for i in range(1, n_turbines + 1)),
        "n_turbines": n_turbines,
    }
    return domain, tuple(agents), tuple(interfaces), open_ports, undeclared, extras


#: `F` is open at `x = 3`? No -- `N` owns `x = 3`, so `F` must exclude it. A
#: half-open box cannot exclude its *low* face, so the exclusion is stated here
#: and enforced by `F` being reached only after `N` in `owner()`. Recorded
#: explicitly because a silent ordering dependency is exactly the kind of thing
#: that survives into a wrong flux integral.
#:
#: Generalized: every rotor claims its strip first (order among rotors does not
#: matter, they never overlap), then `I`, then every near-wake / exit-wake
#: segment (each needs its own rotor already claimed, to cut the notch), then
#: the bypass strips, then every far-wake segment last (a far-wake segment has
#: no hole of its own; it only needs to lose ties at its shared low face, which
#: the near-wake segment immediately upstream -- claimed earlier in this list --
#: already owns).
def _priority_for(n_turbines: int) -> tuple[str, ...]:
    rotors = tuple(f"R{i}" for i in range(1, n_turbines + 1))
    near_and_exit = tuple(_segment_name("N", i) for i in range(1, n_turbines)) + ("W",)
    far = tuple(_segment_name("F", i) for i in range(1, n_turbines))
    return rotors + ("I",) + near_and_exit + ("B+", "B-") + far


(DOMAIN, AGENTS, INTERFACES, OPEN_PORTS, UNDECLARED_BY_DESIGN, _EXTRAS) = build_case(2)
AGENT = {a.name: a for a in AGENTS}

#: `spec-wind-farm-wake-atlas-0.1` section 2.4, restated: same numbers as
#: before, now produced by `build_case(2)` instead of hand-typed.
assert DOMAIN == (-6.0, 18.0, -4.0, 4.0)

_PRIORITY = _priority_for(2)
assert _PRIORITY == ("R1", "R2", "I", "N", "W", "B+", "B-", "F")


def owner(x, y, *, agent: dict | None = None, priority: tuple[str, ...] | None = None) -> np.ndarray:
    """Name of the agent owning each point, '' outside Omega. The partition is
    exact, so this is single-valued everywhere.

    With no keyword arguments this is exactly the original N=2 function, over
    the original module-level `AGENT`/`_PRIORITY`. `build_geometry_case` passes
    an explicit `agent`/`priority` for other N."""
    agent = AGENT if agent is None else agent
    priority = _PRIORITY if priority is None else priority
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = np.full(x.shape, "", dtype=object)
    for name in priority:
        m = agent[name].contains(x, y) & (out == "")
        out[m] = name
    return out


# --------------------------------------------------------------------------
# grids
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class AgentGrid:
    """Token lattice of one agent, with fractional occupancy."""
    name: str
    nx: int
    ny: int
    x_c: np.ndarray                  # [nx]
    y_c: np.ndarray                  # [ny]
    frac: np.ndarray                 # [nx, ny] area fraction inside the agent
    cell_area: float

    @property
    def n_tokens(self) -> int:
        return int(np.count_nonzero(self.frac > 0.0))

    @property
    def area(self) -> float:
        return float(self.frac.sum() * self.cell_area)

    def points(self) -> np.ndarray:
        X, Y = np.meshgrid(self.x_c, self.y_c, indexing="ij")
        return np.stack([X.ravel(), Y.ravel()], axis=1)


def _overlap_1d(lo_a, hi_a, lo_b, hi_b):
    return np.clip(np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b), 0.0, None)


def build_grid(a: AgentRegion) -> AgentGrid:
    """Token lattice with *exact* fractional occupancy.

    Every region here is a box minus boxes, so the overlap of a cell with the
    agent has a closed form and there is no reason to sample for it. The first
    version did sample, at 8x8 per cell, and reported agent `N`'s area as
    11.90625 against a true 11.9 -- the 0.1-wide notch does not land on eighths
    of a 0.25 cell. That is 0.05% and would never have shown up as anything but
    a slightly wrong flux integral much later."""
    dx, dy = a.spacing
    nx = int(round((a.box.x1 - a.box.x0) / dx))
    ny = int(round((a.box.y1 - a.box.y0) / dy))
    if abs(nx * dx - (a.box.x1 - a.box.x0)) > 1e-9 or abs(ny * dy - (a.box.y1 - a.box.y0)) > 1e-9:
        raise AssertionError(f"agent {a.name}: box {a.box} is not an integer number of {a.spacing} tokens")
    x_c = a.box.x0 + (np.arange(nx) + 0.5) * dx
    y_c = a.box.y0 + (np.arange(ny) + 0.5) * dy
    xlo, xhi = x_c[:, None] - dx / 2, x_c[:, None] + dx / 2
    ylo, yhi = y_c[None, :] - dy / 2, y_c[None, :] + dy / 2
    area = (_overlap_1d(xlo, xhi, a.box.x0, a.box.x1)
            * _overlap_1d(ylo, yhi, a.box.y0, a.box.y1))
    for h in a.holes:
        area = area - (_overlap_1d(xlo, xhi, h.x0, h.x1) * _overlap_1d(ylo, yhi, h.y0, h.y1))
    frac = np.clip(area, 0.0, None) / (dx * dy)
    return AgentGrid(a.name, nx, ny, x_c, y_c, frac, dx * dy)


GRIDS: dict[str, AgentGrid] = {a.name: build_grid(a) for a in AGENTS}


#: The two unconnected shaft ports. They carry the extracted power out of the
#: model, so it appears in P_ext instead of vanishing (spec section 5).
#: (restated: identical to the original literal for N=2)
assert OPEN_PORTS == (("R1", "ROT"), ("R2", "ROT"))

#: Shared boundary that exists geometrically but is deliberately not declared,
#: spec section 3.1: each rotor's two lateral faces, |y| = 0.5 over the strip's
#: 0.1 thickness, sitting inside the wake agent's notch. Named here so that the
#: undeclared-interface assertion can subtract a *documented* omission and still
#: fail on an undocumented one.
assert UNDECLARED_BY_DESIGN == {("R1", "N"): 2 * DISK_THICKNESS, ("R2", "W"): 2 * DISK_THICKNESS}


# --------------------------------------------------------------------------
# graph
# --------------------------------------------------------------------------


def adjacency(agents: tuple[AgentRegion, ...] | None = None,
             interfaces: tuple[Interface, ...] | None = None) -> dict[str, set[str]]:
    agents = AGENTS if agents is None else agents
    interfaces = INTERFACES if interfaces is None else interfaces
    adj: dict[str, set[str]] = {a.name: set() for a in agents}
    for e in interfaces:
        adj[e.src].add(e.dst)
        adj[e.dst].add(e.src)
    return adj


def graph_diameter(agents: tuple[AgentRegion, ...] | None = None,
                   interfaces: tuple[Interface, ...] | None = None) -> tuple[int, tuple[str, str]]:
    adj = adjacency(agents, interfaces)
    best, pair = 0, ("", "")
    for s in adj:
        dist = {s: 0}
        q = deque([s])
        while q:
            n = q.popleft()
            for m in adj[n]:
                if m not in dist:
                    dist[m] = dist[n] + 1
                    q.append(m)
        if len(dist) != len(adj):
            missing = sorted(set(adj) - set(dist))
            raise AssertionError(f"agent graph is disconnected: {s} cannot reach {missing}")
        for t, d in dist.items():
            if d > best:
                best, pair = d, (s, t)
    return best, pair


#: Message-passing depth. The spec quotes a diameter of 4 via the path
#: `I -> R1 -> N -> F -> R2`, but that path is not a geodesic -- `I -> N -> F ->
#: R2` is one hop shorter, and BFS gives 3. `n_mp = 4` is kept anyway: it
#: matches the Phase-1 scaffold, and one round more than the diameter is
#: conservative, not wrong. Recorded so the discrepancy is a decision.
N_MP = 4


# --------------------------------------------------------------------------
# shared boundaries, measured rather than declared
# --------------------------------------------------------------------------


def _segments_1d(a: tuple[float, float], b: tuple[float, float]):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return (lo, hi) if hi - lo > EPS else None


def materialized_pairs(n_probe: int = 4001, *,
                       agents: tuple[AgentRegion, ...] | None = None,
                       domain: tuple[float, float, float, float] | None = None,
                       owner_fn=None) -> dict[tuple[str, str], float]:
    """Every pair of agents that actually shares boundary of positive length,
    found by probing both sides of every candidate line rather than by reading
    the edge table -- so an interface the table forgot still shows up.

    With no keyword arguments this is exactly the original N=2 function."""
    agents = AGENTS if agents is None else agents
    domain = DOMAIN if domain is None else domain
    owner_fn = owner if owner_fn is None else owner_fn
    x0, x1, y0, y1 = domain
    out: dict[tuple[str, str], float] = {}
    lines = []
    for a in agents:
        for v in (a.box.x0, a.box.x1):
            lines.append(("x", v))
        for v in (a.box.y0, a.box.y1):
            lines.append(("y", v))
        for h in a.holes:
            lines += [("x", h.x0), ("x", h.x1), ("y", h.y0), ("y", h.y1)]
    seen = set()
    for axis, pos in lines:
        if (axis, round(pos, 12)) in seen:
            continue
        seen.add((axis, round(pos, 12)))
        if axis == "x":
            s = np.linspace(y0, y1, n_probe)
            lo = owner_fn(np.full_like(s, pos - 1e-7), s)
            hi = owner_fn(np.full_like(s, pos + 1e-7), s)
            step = (y1 - y0) / (n_probe - 1)
        else:
            s = np.linspace(x0, x1, n_probe)
            lo = owner_fn(s, np.full_like(s, pos - 1e-7))
            hi = owner_fn(s, np.full_like(s, pos + 1e-7))
            step = (x1 - x0) / (n_probe - 1)
        for A, B in zip(lo, hi):
            if A and B and A != B:
                out[tuple(sorted((A, B)))] = out.get(tuple(sorted((A, B))), 0.0) + step
    return out


# --------------------------------------------------------------------------
# gate W1
# --------------------------------------------------------------------------


def verify(strict: bool = True, *,
          domain: tuple[float, float, float, float] | None = None,
          agents: tuple[AgentRegion, ...] | None = None,
          interfaces: tuple[Interface, ...] | None = None,
          grids: dict[str, "AgentGrid"] | None = None,
          undeclared: dict[tuple[str, str], float] | None = None,
          open_ports: tuple[tuple[str, str], ...] | None = None,
          owner_fn=None,
          n_mp: int | None = None) -> dict:
    """Gate W1 (guide section 2.2). Returns the measured numbers; raises on any
    assertion that fails.

    With no keyword arguments this is exactly the original N=2 gate, over the
    original module-level geometry. `build_geometry_case` calls this with an
    explicit `domain`/`agents`/`interfaces`/`grids`/`undeclared`/`open_ports`/
    `owner_fn` for other N, running the identical checks."""
    domain = DOMAIN if domain is None else domain
    agents = AGENTS if agents is None else agents
    interfaces = INTERFACES if interfaces is None else interfaces
    grids = GRIDS if grids is None else grids
    undeclared = UNDECLARED_BY_DESIGN if undeclared is None else undeclared
    open_ports = OPEN_PORTS if open_ports is None else open_ports
    owner_fn = owner if owner_fn is None else owner_fn
    n_mp = N_MP if n_mp is None else n_mp

    x0, x1, y0, y1 = domain
    report: dict = {}

    # 1. disjointness and coverage, on the continuum, off the lattice so that no
    #    probe lands on a shared face by accident.
    xs = np.linspace(x0, x1, 1201)[:-1] + 0.00137
    ys = np.linspace(y0, y1, 401)[:-1] + 0.00071
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    hits = np.zeros(XX.shape, dtype=np.int64)
    for a in agents:
        hits += a.contains(XX, YY).astype(np.int64)
    if strict and not np.all(hits == 1):
        bad = np.stack([XX[hits != 1], YY[hits != 1]], axis=1)[:5]
        raise AssertionError(
            f"{int((hits != 1).sum())} sample points lie in "
            f"{'no' if (hits == 0).any() else 'more than one'} agent, e.g. {bad}")
    report["partition_exact"] = True

    # 2. areas: the declared regions tile Omega, and the token lattices agree
    total = sum(a.area for a in agents)
    domain_area = (x1 - x0) * (y1 - y0)
    report["area_sum"] = total
    report["domain_area"] = domain_area
    if strict and abs(total - domain_area) > 1e-9:
        raise AssertionError(f"agent areas sum to {total}, domain is {domain_area}")
    for a in agents:
        g = grids[a.name]
        if strict and abs(g.area - a.area) > 1e-6 * max(a.area, 1.0):
            raise AssertionError(
                f"agent {a.name}: token lattice covers {g.area:.6f}, region is {a.area:.6f}")
    report["grid_area_matches_region"] = True

    # 3. every interface lies on BOTH agents' boundaries, and the normal points
    #    src -> dst. One test does both: step off the curve either way and ask
    #    who owns the point.
    for e in interfaces:
        pts = e.sample()
        nx, ny = e.normal_using(owner_fn)
        d = 1e-7
        back = owner_fn(pts[:, 0] - d * nx, pts[:, 1] - d * ny)
        fwd = owner_fn(pts[:, 0] + d * nx, pts[:, 1] + d * ny)
        if strict and not np.all(back == e.src):
            raise AssertionError(
                f"edge {e.src}-{e.dst}: behind the curve is {set(back) - {e.src}}, not {e.src}")
        if strict and not np.all(fwd == e.dst):
            raise AssertionError(
                f"edge {e.src}-{e.dst}: ahead of the curve is {set(fwd) - {e.dst}}, not {e.dst}")
        if strict and e.length <= 0:
            raise AssertionError(f"edge {e.src}-{e.dst}: zero-length interface curve")
    report["normals_src_to_dst"] = True
    report["interface_lengths"] = {f"{e.src}-{e.dst}": e.length for e in interfaces}

    # 4. the materialized pair set equals the declared one, both directions.
    declared = {tuple(sorted(e.key)) for e in interfaces}
    found = materialized_pairs(agents=agents, domain=domain, owner_fn=owner_fn)
    extra = set(found) - declared
    missing = declared - set(found)
    if strict and (extra or missing):
        raise AssertionError(f"agent-pair mismatch: undeclared {sorted(extra)}, "
                             f"declared-but-absent {sorted(missing)}")
    report["n_declared_edges"] = len(declared)
    report["n_materialized_pairs"] = len(found)

    #    ... and the *lengths* agree, except for the documented omissions.
    length_gaps = {}
    for e in interfaces:
        k = tuple(sorted(e.key))
        gap = found[k] - e.length
        allowed = undeclared.get(e.key, 0.0)
        length_gaps[f"{e.src}-{e.dst}"] = gap
        if strict and abs(gap - allowed) > 2e-2:
            raise AssertionError(
                f"edge {e.src}-{e.dst}: shares {found[k]:.4f} of boundary but declares "
                f"{e.length:.4f}; documented omission is {allowed:.4f}")
    report["undeclared_boundary_length"] = length_gaps

    # 5. graph
    diameter, pair = graph_diameter(agents, interfaces)
    report["diameter"] = diameter
    report["diameter_pair"] = pair
    report["n_mp"] = n_mp
    if strict and n_mp < diameter:
        raise AssertionError(f"n_mp={n_mp} is below the graph diameter {diameter}")

    # 6. token counts against the spec table
    counts = {a.name: grids[a.name].n_tokens for a in agents}
    report["tokens"] = counts
    report["tokens_declared"] = {a.name: a.tokens_declared for a in agents}
    report["tokens_total"] = sum(counts.values())
    for a in agents:
        got, want = counts[a.name], a.tokens_declared
        if strict and abs(got - want) > 0.10 * want:
            raise AssertionError(f"agent {a.name}: {got} tokens, spec says {want} (>10%)")
    report["ports_used"] = sorted({p for e in interfaces for p in e.ports})
    report["open_ports"] = list(open_ports)
    return report


REPORT = verify()


# --------------------------------------------------------------------------
# arbitrary-N cases (Atlas 0.1 N-turbine sweep)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GeometryCase:
    """Everything `CoupledSystem` needs for one N-turbine layout, in one place.

    `n_turbines=2` reproduces the module-level `AGENTS`/`INTERFACES`/`DOMAIN`
    exactly (checked by `tests/atlas/windfarm/test_geometry_generator.py`), so
    switching `CoupledSystem` to build from a `GeometryCase` instead of the bare
    module globals is behaviour-preserving at N=2 by construction, not by
    coincidence."""
    n_turbines: int
    spacing: float
    domain: tuple[float, float, float, float]
    agents: tuple[AgentRegion, ...]
    agent: dict[str, AgentRegion]
    grids: dict[str, AgentGrid]
    interfaces: tuple[Interface, ...]
    open_ports: tuple[tuple[str, str], ...]
    undeclared_by_design: dict[tuple[str, str], float]
    priority: tuple[str, ...]
    gs_order: tuple[str, ...]
    field_owner_remap: dict[str, str]
    rotor_names: tuple[str, ...]
    positions: tuple[float, float]
    report: dict | None

    def owner(self, x, y) -> np.ndarray:
        return owner(x, y, agent=self.agent, priority=self.priority)

    def materialized_pairs(self, n_probe: int = 4001) -> dict[tuple[str, str], float]:
        return materialized_pairs(n_probe, agents=self.agents, domain=self.domain,
                                  owner_fn=self.owner)


def build_geometry_case(n_turbines: int = 2, spacing: float = TURBINE_SPACING,
                        strict: bool = True) -> GeometryCase:
    """The generalized entry point: build and (by default) verify the full
    N-turbine agent/interface graph, running gate W1's checks against it.

    `build_geometry_case(2)` runs the exact same `verify()` logic against data
    that is *equal* (not merely equivalent) to the original module-level
    `AGENTS`/`INTERFACES` -- see the equality assertions right after
    `build_case(2)` above, and the dedicated structural-equality test."""
    domain, agents, interfaces, open_ports, undeclared, extras = build_case(n_turbines, spacing)
    agent = {a.name: a for a in agents}
    grids = {a.name: build_grid(a) for a in agents}
    priority = _priority_for(n_turbines)

    def owner_fn(x, y):
        return owner(x, y, agent=agent, priority=priority)

    report = verify(strict, domain=domain, agents=agents, interfaces=interfaces,
                    grids=grids, undeclared=undeclared, open_ports=open_ports,
                    owner_fn=owner_fn) if strict else None

    return GeometryCase(
        n_turbines=n_turbines, spacing=spacing, domain=domain, agents=agents,
        agent=agent, grids=grids, interfaces=interfaces, open_ports=open_ports,
        undeclared_by_design=undeclared, priority=priority,
        gs_order=extras["gs_order"], field_owner_remap=extras["field_owner_remap"],
        rotor_names=extras["rotor_names"], positions=extras["positions"], report=report,
    )


__all__ = [
    "DOMAIN", "X_T1", "X_T2", "ROTOR_HALFSPAN", "DISK_THICKNESS", "WAKE_HALFWIDTH",
    "NEAR_FAR_SPLIT", "TURBINE_SPACING", "NEAR_WAKE_LEN", "EXIT_BUDGET",
    "Box", "AgentRegion", "AgentGrid", "Interface",
    "AGENTS", "AGENT", "GRIDS", "INTERFACES", "OPEN_PORTS", "UNDECLARED_BY_DESIGN",
    "N_MP", "owner", "build_grid", "adjacency", "graph_diameter",
    "materialized_pairs", "verify", "REPORT",
    "build_case", "GeometryCase", "build_geometry_case",
]
