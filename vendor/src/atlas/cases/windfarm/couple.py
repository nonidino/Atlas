"""Phase W5 -- the eight-agent system, coupled and running end to end.

`impl-wind-farm-guide` section 6. Nothing here is trained; the fluid expert is
the frozen checkpoint of `adapters` and the rotor is the zero-parameter disk of
`disk`.

Why this is not the Phase-1 scaffold
------------------------------------
The guide says to reuse the Phase-1 scaffold verbatim and swap the identity
experts for the real ones. That is not possible without rewriting the scaffold:
`atlas.geometry.domains` builds every agent grid from `atlas.solvers.grid`,
whose blocks come from the rocket's nozzle contours, and the agent set, edge
list and conditioning all come from `config/atlas_0_1.yaml`, which describes the
rocket. The scaffold is a *rocket* scaffold, not a case-study-agnostic one.

Recorded rather than worked around, because it is a finding about the framework
and it bears directly on falsification criterion F3: the reusable layer turned
out to be the **port algebra** (`atlas.ports`), which this case study did reuse
without change, and not the model scaffold. What W5 builds instead is the
coupling itself -- which is the part the guide's own file layout calls
`couple.py`, and the part the gates measure.

The window problem, and what it forces
--------------------------------------
The expert is a fixed 128x128 operator on a square window. The agents are not
square and not the same size: `I` is 6x8, the bypasses are 18x2, the rotor
strips are 0.1x1. One window per agent is therefore impossible -- a non-square
agent mapped onto a square window is stretched, and anisotropic stretching is
not a symmetry of Navier-Stokes, while sizing the window to the largest agent
would put `Re_eff = T_s / (nu_p L^2)` through the floor.

So every fluid agent is **tiled** with overlapping windows, and that has a
consequence worth stating plainly: **an agent is no longer atomic.** The tiles
inside one agent exchange through the same band mechanism the declared
interfaces use, but those intra-agent seams are *not* ports -- they are an
artefact of the expert's fixed window size, they are not in the spec's edge
table, and no residual reported here should be read as if they were. The frozen
expert's input shape has imposed a decomposition finer than the one Atlas
declares.

Ownership, and the one place it disagrees with the spec
-------------------------------------------------------
Every global cell is owned by exactly one agent (`geometry.owner`) and written
back by exactly one tile -- the nearest tile belonging to that owner. A tile
reads its whole 128x128 patch, including cells it does not own; that periphery
*is* its interface band, which is why the R2 mechanism needs no separate code
path here.

The exception is the rotor strips. `R1` and `R2` are `disk` agents: algebraic,
holding `(a, C_T', T, P)` and no field at all. But the fluid inside the strip is
still fluid and still has to be advanced. Those cells are therefore written back
by the surrounding wake agent's tiles (`N` for `R1`, `W` for `R2`) even though
the spec's token partition carves them out as a notch. The token partition and
the field representation disagree there, deliberately and in exactly one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field

import numpy as np

from ...ports.residual import PowerLedger, summarize
from .adapters import EXPERT_RES, FrozenFluidExpert, Scaling, divergence, leray_project
from .analytic import Field
from .disk import ActuatorDisk, DiskState, disk_average
from . import geometry
from .geometry import (
    AGENT, AGENTS, DISK_THICKNESS, DOMAIN, INTERFACES, ROTOR_HALFSPAN, TURBINE_SPACING,
    X_T1, X_T2, GeometryCase, build_geometry_case, materialized_pairs, owner,
)
from .metrics import ThrustResidual, thrust_residual
from .pressure import (ProjectionReport, divergence_bc, pressure_neumann,
                       project_outflow)
from .probe import Window, pressure_from_velocity

#: The W5 configuration decided at the 2026-08-21 W0 revisit: expert lead 0.5.
#: Below this the rollout drifts, above it the wake is over-dissipated.
DT_MACRO = 0.25
COUPLE_SCALING = Scaling(length=2.0, velocity=4.0)

#: `band >= U_inf * dt` -- the interface Courant condition measured at W3's
#: second band sweep. At dt = 0.25 the minimum is 0.25 D (16 cells); the tiling
#: below gives at least that on every side.
MIN_BAND = DT_MACRO


# --------------------------------------------------------------------------
# tiling
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Tile:
    """One 128x128 expert window, and the agent responsible for it."""
    index: int
    agent: str
    window: Window

    @property
    def centre(self) -> tuple[float, float]:
        L = self.window.length
        return self.window.x0 + L / 2, self.window.y0 + L / 2


def _centres(lo: float, hi: float, length: float, band: float) -> np.ndarray:
    """Window centres covering `[lo, hi]` with at least `band` of overhang.

    Inset by `length/2 - band` so the end windows stick out past the agent by
    exactly `band`; interior spacing is then whatever is needed to keep the
    overlap at or above `band`."""
    inset = length / 2.0 - band
    a, b = lo + inset, hi - inset
    if b <= a:
        return np.array([0.5 * (lo + hi)])
    stride = length - band
    n = max(2, int(np.ceil((b - a) / stride)) + 1)
    return np.linspace(a, b, n)


def build_tiles_monolithic(length: float = COUPLE_SCALING.length,
                           band: float = 0.5, agents=None) -> list[Tile]:
    """The W9 baseline: **the same tiles, with the agents taken away.**

    [[impl-wind-farm-guide]] section 7.3 asks for "the frozen fluid expert over
    the **undivided** domain". Taken literally that is not constructible -- the
    expert is a fixed 128x128 operator and the domain is 24 x 8, so something is
    tiled either way. W5 already recorded why: the checkpoint's input shape
    imposes a decomposition finer than the one Atlas declares.

    So the baseline keeps the tiling and removes only the partition. Every tile
    sits at exactly the position `build_tiles` puts it, and the single
    difference is that `build_ownership_monolithic` assigns each cell to its
    nearest tile **without asking which agent owns it**, so no cell-to-tile
    boundary follows a declared interface and no port is ever materialized.

    **The first version of this tiled the domain box instead, and that was a
    confound worth recording.** It produced 102 tiles against the partitioned
    run's 124, at entirely different positions, so the seams fell in different
    places -- and a difference between the two runs then mixed "declaring
    agents" with "placing windows somewhere else". Since seam artefacts are
    exactly the kind of error W9 is trying to attribute, that difference could
    not be read. Same tiles is the control that isolates one variable."""
    return [Tile(t.index, MONOLITHIC_AGENT, t.window)
            for t in build_tiles(length, band, agents=agents)]


def build_ownership_monolithic(g: GlobalField, tiles: list[Tile]):
    """Every cell to its nearest tile. No agent test, because there is one agent."""
    X, Y = np.meshgrid(g.x_c, g.y_c, indexing="xy")
    own = np.full(X.shape, MONOLITHIC_AGENT, dtype=object)
    best = np.full(X.shape, -1, dtype=np.int64)
    dist = np.full(X.shape, np.inf)
    for t in tiles:
        w = t.window
        L = w.length
        inside = (X >= w.x0) & (X < w.x0 + L) & (Y >= w.y0) & (Y < w.y0 + L)
        cx, cy = t.centre
        d = np.maximum(np.abs(X - cx), np.abs(Y - cy))
        take = inside & (d < dist)
        best[take] = t.index
        dist[take] = d[take]
    if (best < 0).any():
        raise AssertionError(f"{int((best < 0).sum())} cells covered by no tile")
    return own, best


def build_tiles(length: float = COUPLE_SCALING.length, band: float = 0.5,
                agents=None) -> list[Tile]:
    """Tile every fluid agent. The `disk` agents get none: they hold no field.

    `agents` defaults to the module-level (N=2) `AGENTS`; `CoupledSystem`
    passes its `GeometryCase.agents` for other N, which has the identical
    shape for every `N`/`F`/`W` segment regardless of turbine count -- so this
    function's logic did not need to change at all, only its input."""
    if band < MIN_BAND - 1e-12:
        raise ValueError(
            f"band {band} is below U_inf*dt = {MIN_BAND}. Measured at W3: a band the "
            "flow crosses within one macro-step is overrun and stops transmitting "
            "(19.6% error in <U_d> at a quarter of the required width).")
    agents = AGENTS if agents is None else agents
    tiles: list[Tile] = []
    for a in agents:
        if a.expert != "fluid":
            continue
        for xc in _centres(a.box.x0, a.box.x1, length, band):
            for yc in _centres(a.box.y0, a.box.y1, length, band):
                tiles.append(Tile(len(tiles), a.name,
                                  Window(x0=xc - length / 2, y0=yc - length / 2,
                                         length=length)))
    return tiles


# --------------------------------------------------------------------------
# the global field and who writes what
# --------------------------------------------------------------------------


@dataclass
class GlobalField:
    """The domain at the expert's own resolution, indexed [y, x]."""
    dx: float
    x_c: np.ndarray
    y_c: np.ndarray
    u: np.ndarray
    v: np.ndarray

    @property
    def shape(self):
        return self.u.shape


def build_global(length: float = COUPLE_SCALING.length, u_inf: float = 1.0,
                 domain=None) -> GlobalField:
    x0, x1, y0, y1 = DOMAIN if domain is None else domain
    dx = length / EXPERT_RES
    nx = int(round((x1 - x0) / dx))
    ny = int(round((y1 - y0) / dx))
    x_c = x0 + (np.arange(nx) + 0.5) * dx
    y_c = y0 + (np.arange(ny) + 0.5) * dx
    return GlobalField(dx, x_c, y_c,
                       np.full((ny, nx), float(u_inf)), np.zeros((ny, nx)))


#: The single pseudo-agent of the W9 monolithic baseline. Deliberately not a
#: name in `AGENTS`, so anything that looks an agent up by name fails loudly
#: rather than silently treating the baseline as a partitioned run.
MONOLITHIC_AGENT = "ALL"

#: The rotor strips hold no field of their own, so their cells are carried by
#: the wake agent around them. The single place the token partition and the
#: field representation disagree, named here rather than left implicit.
FIELD_OWNER_REMAP = {"R1": "N", "R2": "W"}


def build_ownership(g: GlobalField, tiles: list[Tile], owner_fn=None, field_owner_remap=None):
    """`(owner_agent, tile_index)` for every global cell.

    A cell is written by the nearest tile *of its own agent*. Cells whose agent
    has no tile containing them would be unwritable; that is an error in the
    tiling, not something to paper over, so it raises.

    `owner_fn`/`field_owner_remap` default to the module-level (N=2)
    `geometry.owner`/`FIELD_OWNER_REMAP`; `CoupledSystem` passes its
    `GeometryCase`'s bound owner and `field_owner_remap` for other N."""
    owner_fn = owner if owner_fn is None else owner_fn
    field_owner_remap = FIELD_OWNER_REMAP if field_owner_remap is None else field_owner_remap
    X, Y = np.meshgrid(g.x_c, g.y_c, indexing="xy")
    own = owner_fn(X, Y)
    for k, v in field_owner_remap.items():
        own[own == k] = v
    best = np.full(own.shape, -1, dtype=np.int64)
    dist = np.full(own.shape, np.inf)
    for t in tiles:
        w = t.window
        L = w.length
        inside = ((X >= w.x0) & (X < w.x0 + L) & (Y >= w.y0) & (Y < w.y0 + L)
                  & (own == t.agent))
        cx, cy = t.centre
        d = np.maximum(np.abs(X - cx), np.abs(Y - cy))       # Chebyshev: prefer centred
        take = inside & (d < dist)
        best[take] = t.index
        dist[take] = d[take]
    missing = (own != "") & (best < 0)
    if missing.any():
        i, j = np.nonzero(missing)
        raise AssertionError(
            f"{missing.sum()} cells have no tile of their own agent, e.g. "
            f"({g.x_c[j[0]]:.3f}, {g.y_c[i[0]]:.3f}) owned by {own[i[0], j[0]]!r}")
    return own, best


# --------------------------------------------------------------------------
# gather / scatter
# --------------------------------------------------------------------------


def _slice_of(g: GlobalField, t: Tile):
    w = t.window
    i0 = int(round((w.x0 - g.x_c[0] + g.dx / 2) / g.dx))
    j0 = int(round((w.y0 - g.y_c[0] + g.dx / 2) / g.dx))
    return i0, j0


def gather(g: GlobalField, tiles: list[Tile]):
    """Read every tile's 128x128 patch out of the global field.

    Indexed by POSITION in `tiles`, not by `Tile.index` -- the fixed point
    gathers a two-tile subset, and indexing a subset array by a global tile id
    is an out-of-bounds waiting to happen. `scatter` takes the same list and the
    same positional convention.

    Outside the domain the field is extended by *edge replication*, which is the
    composite outer boundary condition of spec section 9 as closely as a periodic
    operator allows: a slip lateral wall (`du_x/dy = 0`) and a convective outlet
    both look like "no gradient across the edge"."""
    n = EXPERT_RES
    U = np.empty((len(tiles), n, n))
    V = np.empty((len(tiles), n, n))
    ny, nx = g.shape
    for pos, t in enumerate(tiles):
        i0, j0 = _slice_of(g, t)
        ii = np.clip(np.arange(i0, i0 + n), 0, nx - 1)
        jj = np.clip(np.arange(j0, j0 + n), 0, ny - 1)
        U[pos] = g.u[np.ix_(jj, ii)]
        V[pos] = g.v[np.ix_(jj, ii)]
    return U, V


def scatter(g: GlobalField, tiles: list[Tile], U: np.ndarray, V: np.ndarray,
            best: np.ndarray) -> None:
    """Write each tile's owned cells back. `U`, `V` are positional, as `gather`
    returns them; ownership is still by `Tile.index`, which is what `best`
    holds."""
    n = EXPERT_RES
    ny, nx = g.shape
    for pos, t in enumerate(tiles):
        i0, j0 = _slice_of(g, t)
        i1, j1 = min(i0 + n, nx), min(j0 + n, ny)
        a0, b0 = max(i0, 0), max(j0, 0)
        if a0 >= i1 or b0 >= j1:
            continue
        sub = best[b0:j1, a0:i1] == t.index
        if not sub.any():
            continue
        tu = U[pos][b0 - j0:j1 - j0, a0 - i0:i1 - i0]
        tv = V[pos][b0 - j0:j1 - j0, a0 - i0:i1 - i0]
        g.u[b0:j1, a0:i1][sub] = tu[sub]
        g.v[b0:j1, a0:i1][sub] = tv[sub]


def taper_1d(n: int, band_cells: float) -> np.ndarray:
    """Smoothstep taper: 0 at both window edges, 1 in the interior.

    `t^2 (3 - 2t)` rather than a linear ramp because it is C1 at both ends, so
    the blended field has no kink where the taper reaches 1 -- a linear ramp
    trades a jump at the seam for a gradient jump slightly inside it, which the
    dissipation term of the power ledger would pick up as readily.

    Smoothstep also happens to form an **exact** partition of unity across a
    pairwise overlap: `S(t) + S(1-t) = 1` identically. Adjacent tiles overlap by
    exactly `band` and each tapers over `band`, so in the two-tile case the
    weights sum to one before normalization and the normalization is a no-op.
    Where three or more tiles meet, the normalization does the work."""
    s = np.arange(n) + 0.5
    t = np.clip(np.minimum(s, n - s) / max(band_cells, 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def tile_weight(n: int, band_cells: float) -> np.ndarray:
    """The 2-D taper, as a separable product."""
    w = taper_1d(n, band_cells)
    return w[:, None] * w[None, :]


def scatter_pou(g: GlobalField, tiles: list[Tile], U: np.ndarray, V: np.ndarray,
                own: np.ndarray, band_cells: float) -> None:
    """Blend overlapping tiles with a partition of unity, instead of picking one.

    **What this fixes, and why it was worth fixing before anything else.**
    `scatter` gives each cell to its *nearest* tile and discards what every
    other tile predicted there. Tiles overlap by 0.5 D -- 32 cells -- so a large
    fraction of the domain has two or four independent expert predictions
    available and exactly one of them is kept. Because the choice is a hard
    nearest-tile switch, neighbouring cells on opposite sides of an ownership
    boundary come from *different expert calls*, and the assembled field has a
    genuine discontinuity there. Measured on the t = 20 field:

        adjacent cell pairs      mean |du|     p99
        same tile                1.81e-3       1.18e-2
        different tile           5.03e-3       4.56e-2
        different agent          7.48e-3       1.08e-1

    The seams carry 3-4x the interior's cell-to-cell variation, and at declared
    interfaces the 99th percentile reaches 10% of freestream. **That jump is an
    artefact of the hard choice, not of the physics**, and a continuous weight
    removes it by construction rather than by correction.

    **Ownership is still respected.** The weighted sum runs only over tiles of
    the cell's own agent, so declared interfaces stay real interfaces and the
    port residuals still measure something. Blending across agents would drive
    every port residual to zero by erasing the distinction the framework exists
    to test -- a metric that reads perfect because it stopped looking.

    So this targets the 9,584 tile-seam pairs, which W5 established are *not*
    ports -- they are an artefact of the checkpoint's fixed window size -- and
    deliberately leaves the 1,024 agent-seam pairs to the declared R2 band
    mechanism, which is the contract under test."""
    n = EXPERT_RES
    ny, nx = g.shape
    W = tile_weight(n, band_cells)
    num_u = np.zeros(g.shape)
    num_v = np.zeros(g.shape)
    den = np.zeros(g.shape)
    for pos, t in enumerate(tiles):
        i0, j0 = _slice_of(g, t)
        i1, j1 = min(i0 + n, nx), min(j0 + n, ny)
        a0, b0 = max(i0, 0), max(j0, 0)
        if a0 >= i1 or b0 >= j1:
            continue
        sub = own[b0:j1, a0:i1] == t.agent
        if not sub.any():
            continue
        w = np.where(sub, W[b0 - j0:j1 - j0, a0 - i0:i1 - i0], 0.0)
        num_u[b0:j1, a0:i1] += w * U[pos][b0 - j0:j1 - j0, a0 - i0:i1 - i0]
        num_v[b0:j1, a0:i1] += w * V[pos][b0 - j0:j1 - j0, a0 - i0:i1 - i0]
        den[b0:j1, a0:i1] += w
    live = den > 0.0

    # A cell whose agent is being scattered but which no tile of that agent
    # weights is unwritable, and silently leaving it at its previous value is a
    # stale-state bug that reads as a slow drift.
    #
    # The obvious version of this check does not work, and the first draft
    # shipped it: accumulating a `touched` mask inside the loop and testing
    # `touched & ~live` can never fire, because a cell is only touched where the
    # weight was added, and the taper is strictly positive at every cell centre
    # strictly inside a window -- so `touched` is a subset of `live` by
    # construction. A guard that cannot fire is worse than no guard, because it
    # looks like protection. The set to compare against is the *ownership map*,
    # restricted to the agents this call is responsible for, since a
    # Gauss-Seidel sweep scatters one agent group at a time.
    should = np.zeros(g.shape, dtype=bool)
    for name in {t.agent for t in tiles}:
        should |= (own == name)
    missing = should & ~live
    if missing.any():
        i, j = np.nonzero(missing)
        raise AssertionError(
            f"{int(missing.sum())} owned cells got zero blend weight, e.g. "
            f"({g.x_c[j[0]]:.3f}, {g.y_c[i[0]]:.3f})")
    g.u[live] = num_u[live] / den[live]
    g.v[live] = num_v[live] / den[live]


def inflow_profile(y: np.ndarray, t: float, u_inf: float = 1.0,
                   gust: bool = False) -> np.ndarray:
    """Spec section 9's inlet: uniform `U = u_inf`, or the gust ramp.

    `U(t) = 1 + 0.15 tanh((t - 20) / 2)` is the guide's section 7.1 case, and it
    exists because a composed system that settles to a steady wake has not
    tested temporal coupling at all."""
    u = u_inf * (1.0 + 0.15 * np.tanh((t - 20.0) / 2.0)) if gust else u_inf
    return np.full_like(np.asarray(y, dtype=np.float64), float(u))


def apply_outer_bc(g: GlobalField, t: float, band: float, u_inf: float = 1.0,
                   gust: bool = False) -> None:
    """The *outer* boundary of the union of all agents -- not an interface.

    Spec section 9. Three conditions, and the first is load-bearing:

    * **inlet, x = -6: Dirichlet.** Without it the inflow is whatever the
      windows produce, the rotors' deceleration propagates upstream through the
      edge-replicated boundary, and the entire domain slows down together.
      Measured before this was added: at t = 1.5 turbine 2's `<U_d>` had fallen
      to 0.674, tracking turbine 1 almost exactly -- at a time when turbine 1's
      wake has covered a fifth of the 7 D between them and cannot have reached
      it. That is a boundary condition failure wearing a wake's clothes.
    * **lateral, |y| = 4: slip.** `u_y = 0`, `du_x/dy = 0`.
    * **outlet, x = 18: convective outflow**, which edge replication already
      approximates -- so it is left alone rather than imposed twice.

    The inlet is pinned over a band rather than a single column, for the same
    reason interfaces need one: at `dt` the flow crosses `U dt`, and a condition
    thinner than that is overrun within a step."""
    ncol = max(1, int(round(band / g.dx)))
    g.u[:, :ncol] = inflow_profile(g.y_c, t, u_inf, gust)[:, None]
    g.v[:, :ncol] = 0.0
    nrow = max(1, int(round(0.5 * band / g.dx)))
    g.v[:nrow, :] = 0.0                      # slip: no through-flow at the walls
    g.v[-nrow:, :] = 0.0


def window_slice(g: GlobalField, w: Window):
    """Index arrays for an arbitrary window, clipped at the domain edge."""
    n = w.n
    i0 = int(round((w.x0 - g.x_c[0] + g.dx / 2) / g.dx))
    j0 = int(round((w.y0 - g.y_c[0] + g.dx / 2) / g.dx))
    ny, nx = g.shape
    ii = np.clip(np.arange(i0, i0 + n), 0, nx - 1)
    jj = np.clip(np.arange(j0, j0 + n), 0, ny - 1)
    return np.ix_(jj, ii)


def tile_open_faces(g: GlobalField, tiles: list[Tile],
                    length: float = 2.0) -> np.ndarray:
    """Which of each tile's four faces is a **true domain boundary**.

    Order is `(x-lo, x-hi, y-lo, y-hi)`. The distinction the 2026-08-24
    refinement test forced: an open-boundary condition is a statement about the
    *domain*, and applying it to an interior tile seam throws away the
    neighbour's data -- which on a seam is not merely available but is the right
    answer -- and replaces it with an extrapolation that is inconsistent wherever
    streamwise structure crosses. Measured on a fully interior tile, one
    macro-step, rotor force off: `characteristic` gave `max|u|` of 1.17, 3.10,
    11.87 at 48, 190 and 379 sub-steps; with these flags it gives 1.2182,
    1.2182, 1.2183. Converged, and equal to the `dirichlet` control, because for
    an interior tile that is what the condition should reduce to.

    Only **7 of 124** tiles have a face on the real outlet."""
    eps = 1e-9
    x_hi = g.x_c.max() + g.dx / 2
    out = np.zeros((len(tiles), 4), dtype=bool)
    for i, t in enumerate(tiles):
        # Only the outlet. The inlet is prescribed, and the walls are slip -- for
        # both, the ring read out of the assembled field already carries the
        # domain's own boundary condition, so taking it hard is both correct and
        # stable. An open condition is only ever right where the domain actually
        # opens.
        out[i, 1] = t.window.x0 + length >= x_hi - eps
    return out


def global_project_outflow(g: GlobalField) -> ProjectionReport:
    """Project the assembled field with the domain's **real** boundary conditions.

    The adopted path, and the fix for the finding W5 recorded on 2026-08-21.
    `global_project` below is exact and is still the wrong operator, because the
    FFT that makes it exact also makes the domain periodic in `x`: what leaves
    the outlet arrives at the inlet. A wind farm's inlet is prescribed and its
    outlet is open, and the difference is not cosmetic -- it is the entire
    upstream-induction error. Measured at t = 15 under the periodic projection:
    a 26% deficit one diameter upstream of turbine 1 and 13% three diameters
    upstream, against roughly 10% and 2% for a real disk, which inverted
    `P_2/P_1` to 1.16.

    `pressure.project_outflow` carries the mechanism and the reasoning. What
    matters here is the accounting: the correction is zero on the inlet and wall
    faces by construction, so this cannot violate the boundary condition it was
    given, and the outlet is the one face free to pass the imbalance out of the
    domain.

    Returns the report rather than swallowing it -- `div_face_after` is a gate
    number and `net_flux_out - net_flux_in` is the quantity that was structurally
    pinned to zero before and no longer is."""
    g.u, g.v, rep = project_outflow(g.u, g.v, g.dx)
    return rep


def global_project(g: GlobalField) -> None:
    """Leray projection of the **assembled** field, periodic -- the CONTROL.

    Superseded by `global_project_outflow` and kept deliberately, the way W3
    kept R1: the 2026-08-21 rollout numbers were produced by this operator, and
    a fix is only readable against the thing it replaced. `CoupleConfig.pressure
    = 'periodic'` selects it.

    Its own docstring, unchanged, because the reasoning below is still correct
    as far as it goes -- it is a genuine improvement on the per-tile projection
    it replaced, and it is still periodic in `x`, which is the part that had to
    go.

    This is the fix for the failure mode W3 already named: *a periodic pressure
    cannot carry a mean gradient*. The Leray projection leaves the `k = 0` mode
    alone by construction, so a tile containing a rotor loses mean momentum
    every step with nothing to balance it, and the deceleration accumulates
    until the flow through the disk reverses -- measured, `<U_d> = -0.33` by
    t = 5.

    In a real incompressible flow the actuator disk does *not* slow the fluid
    within the strip: it imposes a pressure jump, and the pressure field
    decelerates the flow gradually over the whole streamtube, most of it
    upstream. Producing that requires a pressure solve over a region large
    enough to hold the streamtube -- a 2 D tile is not, the domain is. So the
    projection is done once on the assembled field.

    Mass conservation is a global statement anyway: spec section 7.1 (C1) asks
    for it over every agent *and every union of agents*, which a per-tile
    projection cannot deliver however exact it is on each tile."""
    ny, nx = g.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=g.dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=g.dx)
    if nx % 2 == 0:
        kx[nx // 2] = 0.0
    if ny % 2 == 0:
        ky[ny // 2] = 0.0
    KX, KY = kx[None, :], ky[:, None]
    k2 = KX * KX + KY * KY
    k2 = np.where(k2 == 0.0, 1.0, k2)
    uh, vh = np.fft.fft2(g.u), np.fft.fft2(g.v)
    div = KX * uh + KY * vh
    g.u = np.real(np.fft.ifft2(uh - KX * div / k2))
    g.v = np.real(np.fft.ifft2(vh - KY * div / k2))


# --------------------------------------------------------------------------
# the rotors
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Rotor:
    name: str
    x_face: float
    disk: ActuatorDisk

    @property
    def thickness(self) -> float:
        return self.disk.thickness

    @property
    def x_back(self) -> float:
        return self.x_face + self.thickness


_STRIP_CACHE: dict = {}


def strip_fraction(g: GlobalField, r: Rotor) -> np.ndarray:
    """Per-cell area fraction inside the rotor strip, cached.

    A boolean mask is not good enough. The strip is 0.1 D thick and the lattice
    is 0.015625 D, so it spans 6.4 cells -- a mask rounds that to 6 or 7 and
    loses or gains 6% of the thrust, which then shows up as an interface
    residual and gets blamed on the expert. `ActuatorDisk.body_force_field`
    weights by exact rectangle overlap so the discrete integral is `-T` on any
    lattice, and gate W2 tests precisely that; this reuses it rather than
    reimplementing a second, worse version."""
    key = (id(g), r.name)
    m = _STRIP_CACHE.get(key)
    if m is None:
        unit = r.disk.body_force_field(g.x_c, g.y_c, g.dx, g.dx, 1.0,
                                       x0=r.x_face, y0=-ROTOR_HALFSPAN)
        m = unit / r.disk.force_density(1.0) * -1.0     # back out the pure fraction
        _STRIP_CACHE[key] = m
    return m


def rotor_force(g: GlobalField, r: Rotor, thrust: float) -> np.ndarray:
    """`f_x` on the global lattice, integrating to exactly `-thrust`."""
    return -r.disk.force_density(thrust) * strip_fraction(g, r)


def disk_velocity(g: GlobalField, r: Rotor) -> float:
    """Disk-averaged streamwise velocity on the rotor's UPSTREAM face.

    Spec section 4.2: the local-induction form reads its own inflow, which is
    what makes turbine 2 respond to turbine 1's wake instead of to the domain
    inlet."""
    i = int(np.argmin(np.abs(g.x_c - (r.x_face - g.dx))))
    m = np.abs(g.y_c) <= ROTOR_HALFSPAN
    return float(np.mean(g.u[m, i]))


@dataclass
class CVBalance:
    """Streamwise momentum budget of a CLOSED control volume, all four faces.

    W3's `slab_balance` spans the full height of a periodic window, so its
    lateral faces coincide and cancel identically and it never has to evaluate
    them. That does not survive into the coupled system: the rotor's control
    volume is a 2 D tile inside an 8 D domain, the flow crosses `y = +-1`, and
    the transverse flux `u_y u_x` is a first-class term. Ignoring it reported a
    pre-projection `r_T` of 24-37% -- which was the neglected lateral flux, not
    the expert."""
    conv_x: float
    conv_y: float
    pressure: float
    viscous: float
    d_momentum_dt: float
    forcing: float
    #: Compatibility defect of the Neumann pressure solve, in the same units as
    #: the terms above. The error bar on `pressure`, carried rather than
    #: discarded: `integral laplacian p dA` must equal `contour integral dp/dn ds`
    #: for the continuous problem, so what is left over is how far the input
    #: field is from admitting a pressure at all. A residual quoted without it
    #: is a residual quoted without its instrument's accuracy.
    p_mismatch: float = float("nan")

    @property
    def phi_x(self) -> float:
        return self.conv_x + self.conv_y + self.pressure + self.viscous

    @property
    def residual(self) -> float:
        """`dM/dt + Phi_x - integral f`, zero at every instant for a true flow."""
        return self.d_momentum_dt + self.phi_x - self.forcing


def cv_balance(u0, v0, u1, v1, window: Window, xl: float, xr: float,
               yl: float, yr: float, dt: float, nu: float,
               fx: np.ndarray | None = None,
               pressure_mode: str = "neumann") -> CVBalance:
    """Streamwise momentum balance over `[xl, xr] x [yl, yr]`.

    Fluxes at the midpoint in time, so they are second-order consistent with the
    centred `dM/dt`; the pressure is recovered from the midpoint velocity.

    `pressure_mode` is the W5 fix and it is the difference between a readable
    residual and an unreadable one:

    `'neumann'`   the window is treated as what it is -- a piece of a larger
                  domain -- and `p` comes from a Poisson solve whose boundary
                  data is the normal momentum equation on the window face.
    `'periodic'`  the pre-fix operator, `probe.pressure_from_velocity`. Exact on
                  W3's genuinely periodic probe window and wrong here. Measured
                  against an analytic Taylor-Green pressure on a non-periodic
                  sub-window it errs by 56-70%, and the error *grows* under grid
                  refinement -- the signature of an inconsistent operator rather
                  than an inaccurate one. Kept as the control, because that
                  comparison is the evidence the fix was needed.

    The viscous derivatives move with the pressure. Spectral differentiation
    carries the same wrap-around assumption as the spectral solve, so in
    `'neumann'` mode they are second-order finite differences: centred inside
    and one-sided on the edge, which is where a control-volume face sits."""
    if pressure_mode not in ("neumann", "periodic"):
        raise ValueError(pressure_mode)
    L, h = window.length, window.dx
    il, ir = window.index_of(xl), window.index_of(xr)
    jl, ju = int(np.argmin(np.abs(window.y_c - yl))), int(np.argmin(np.abs(window.y_c - yr)))
    um, vm = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    if pressure_mode == "neumann":
        p, mismatch = pressure_neumann(um, vm, fx, None, h, dudt=(u1 - u0) / dt,
                                       nu=nu, return_mismatch=True)
        dudx = np.gradient(um, h, axis=1)
        dudy = np.gradient(um, h, axis=0)
    else:
        p = pressure_from_velocity(um, vm, fx, None, L)
        mismatch = float("nan")
        dudx = _ddx_arr(um, L, axis=1)
        dudy = _ddx_arr(um, L, axis=0)
    ys = slice(jl, ju)
    xs = slice(il, ir)

    # x faces: columns; the quadrature weight along them is dy = h
    conv_x = h * (float(np.sum(um[ys, ir] ** 2)) - float(np.sum(um[ys, il] ** 2)))
    pres = h * (float(np.sum(p[ys, ir])) - float(np.sum(p[ys, il])))
    visc_x = -nu * h * (float(np.sum(dudx[ys, ir])) - float(np.sum(dudx[ys, il])))
    # y faces: rows, carrying transverse momentum flux u_y u_x
    conv_y = h * (float(np.sum((vm * um)[ju, xs])) - float(np.sum((vm * um)[jl, xs])))
    visc_y = -nu * h * (float(np.sum(dudy[ju, xs])) - float(np.sum(dudy[jl, xs])))

    cell = h * h
    m0 = cell * float(np.sum(u0[ys, xs]))
    m1 = cell * float(np.sum(u1[ys, xs]))
    forcing = cell * float(np.sum(fx[ys, xs])) if fx is not None else 0.0
    return CVBalance(conv_x, conv_y, pres, visc_x + visc_y, (m1 - m0) / dt, forcing,
                     p_mismatch=float(mismatch) * cell * (ir - il) * (ju - jl))


def _ddx_arr(f, length, axis):
    n = f.shape[axis]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
    if n % 2 == 0:
        k[n // 2] = 0.0
    shape = [1, 1]
    shape[axis] = n
    return np.real(np.fft.ifft(1j * k.reshape(shape) * np.fft.fft(f, axis=axis), axis=axis))


def rotor_window(r: Rotor, length: float = COUPLE_SCALING.length) -> Window:
    """A window centred on the rotor, for the momentum balance and the projection.

    Not one of the tiles. The tiles are placed to cover agents, so the nearest
    one to a rotor can have it well off-centre -- and a control volume clipped by
    the tile edge silently drops part of the strip: measured, 27% of the applied
    force went missing that way. Centring the measurement window on the rotor
    also makes W5's `r_T` directly comparable to W3's, which used exactly this
    geometry."""
    return Window(x0=r.x_face + r.thickness / 2 - length / 2,
                  y0=-length / 2, length=length)


# --------------------------------------------------------------------------
# the thrust projection -- spec section 7.3, adapted to a velocity decoder
# --------------------------------------------------------------------------


def _divfree_direction(window: Window, xl: float, xr: float) -> np.ndarray:
    """The min-norm correction direction, projected divergence free.

    Spec section 7.3 does this in stream-function space, where the correction
    preserves `div u = 0` identically because `div curl = 0` whatever the
    correction is. This checkpoint has no stream-function head (W0), so the same
    guarantee is obtained by projecting the direction itself: a Leray-projected
    field is divergence free, and adding any multiple of it leaves the divergence
    untouched.

    The direction is the Leray projection of the indicator of the control
    volume in `x`. That is the gradient of the *unsteady* term of the balance,
    which is the part linear in `u`; the convective and pressure terms
    contribute the quadratic part, handled below by solving the exact scalar
    quadratic rather than by linearizing.

    **This direction is degenerate, and the fixed pressure is what revealed it.**
    The indicator varies only in `x` and points along `x`, so in Fourier every
    mode of it is pure gradient: `w_hat = d_hat - k (k.d_hat)/k^2` kills
    everything except `k = 0`. What comes back is a **uniform** field -- measured,
    `max|w_u|` equals its mean to every digit. A uniform velocity added to an
    incompressible flow is a Galilean shift, and once the pressure is recovered
    with the correct Neumann data (which contains `-du/dt`) the momentum balance
    is invariant under it: the pressure flux moves by `-6.19e-2 lambda` and
    `dM/dt` by `+6.19e-2 lambda`, cancelling to five figures. The measured
    sensitivity is `dg/dlambda = 1.0e-6` against a residual of `1.0e-1`, so the
    correction needed is `lambda = 9.9e4` and the field it produces peaks at
    770 times freestream.

    Under the *periodic* pressure this looked like it worked, and the reason is
    exactly the defect that was fixed: a periodic pressure cannot carry a mean
    gradient, so it could not perform the cancellation, and `beta` was
    spuriously nonzero. **The same missing degree of freedom made the target
    wrong and made the direction appear viable.** A correction that actually
    constrains this residual has to be localized near the disk rather than
    uniform over the window; recorded as the open item it is, per the guide's
    rule that a failed gate stops the phase."""
    n = window.n
    x = window.x_c[None, :] * np.ones((n, 1))
    d = np.where((x >= xl) & (x <= xr), 1.0, 0.0)
    wu, wv = leray_project(d, np.zeros_like(d), window.length)
    nrm = float(np.sqrt(np.sum(wu ** 2 + wv ** 2)))
    return wu / max(nrm, 1e-30), wv / max(nrm, 1e-30)


@dataclass
class Projection:
    """What one thrust projection did. `pre` is the number that matters."""
    rotor: str
    pre: float
    post: float
    lam: float
    norm: float
    alpha: float
    #: Compatibility defect of this window's Neumann pressure solve, as a
    #: fraction of `T`. `pre` is only meaningful when this is small: it is the
    #: part of the momentum budget the pressure recovery could not account for,
    #: so a `pre` below it is measuring the instrument.
    mismatch: float = float("nan")
    #: `dg/dlambda` -- how much the momentum residual actually responds to the
    #: correction direction. Reported because a projection is only meaningful
    #: when this is O(1); at 1e-6 the direction is in the constraint's null
    #: space and `post` grades the arithmetic rather than the flow.
    sensitivity: float = float("nan")
    #: True when the correction was computed but refused as degenerate.
    refused: bool = False


def project_thrust(u0, v0, u1, v1, window: Window, r: Rotor, thrust: float,
                   dt: float, nu: float, fx: np.ndarray, half: float = 0.5,
                   y_half: float = 0.9, pressure_mode: str = "neumann"):
    """Make the fluid side's momentum balance match the disk's thrust exactly.

    **Only valid where the pressure is.** The mechanism itself works -- the
    correction is divergence free by construction and drives the measured
    residual to 1e-15 -- but what it drives to zero is `g`, and `g` contains a
    pressure flux recovered by `pressure_from_velocity`, which assumes a
    periodic window. W3's probe *was* a periodic window, so there this is sound.
    A rotor window inside the coupled domain is not, and with a body force
    present the recovered pressure is wrong by enough that `g` reads 100-450% of
    `T` while the same operator on a turbine-free volume reads 3%. Correcting
    to that target is what reversed the flow.

    Kept, with `CoupleConfig.project` defaulting to False, because the fix is a
    Neumann Poisson solve on the window rather than a different projection, and
    because the measurement it wraps -- the *pre*-projection residual -- is the
    number the guide says matters.

    `g(lambda) = dM/dt + Phi_x - integral f` evaluated at `u1 + lambda w` is a
    *quadratic* in `lambda` -- the convective fluxes are quadratic in the
    velocity, the pressure and viscous terms linear. Rather than deriving the
    gradient and Hessian, the three coefficients are recovered by evaluating `g`
    at three values of `lambda`, which is exact for a quadratic and costs three
    integrals and no network calls.

    The sampling scale matters. Probing at a fixed small `eps` puts the second
    difference `(g+ + g- - 2 g0)` at the far end of a catastrophic cancellation
    and left the post-projection residual at 1.5e-8, just outside the gate. So a
    cheap first pass estimates the root, and the quadratic is then fitted at
    `0, lambda_est, 2 lambda_est` -- three points that straddle it. Still one
    closed-form solve, still no iteration on the constraint."""
    xl, xr = r.x_face - half, r.x_back + half
    yl, yr = -y_half, y_half
    wu, wv = _divfree_direction(window, xl, xr)

    def balance(lam):
        return cv_balance(u0, v0, u1 + lam * wu, v1 + lam * wv,
                          window, xl, xr, yl, yr, dt, nu, fx,
                          pressure_mode=pressure_mode)

    def g(lam):
        return balance(lam).residual

    b0 = balance(0.0)
    g0 = b0.residual
    mism = abs(b0.p_mismatch) / max(abs(thrust), 1e-30)
    eps = max(1e-4, 1e-3 * float(np.sqrt(np.mean(u1 ** 2))))
    beta0 = (g(eps) - g(-eps)) / (2.0 * eps)
    lam_est = -g0 / beta0 if abs(beta0) > 1e-30 else 0.0
    if not np.isfinite(lam_est) or abs(lam_est) < 1e-14:
        return u1, v1, Projection(r.name, abs(g0) / max(abs(thrust), 1e-30),
                                  abs(g0) / max(abs(thrust), 1e-30), 0.0, 0.0, 0.0,
                                  mismatch=mism)

    # A correction that moves the field further than the field itself is not a
    # min-norm correction, it is a different flow. This guard exists because the
    # run that motivated it reported post-projection r_T = 7.5e-11 -- passing the
    # gate by four orders of magnitude -- while <U_d> stood at 770 times
    # freestream. `post` measures whether the scalar solve found its root, and a
    # root of a nearly-degenerate constraint is found very precisely and means
    # nothing. Same lesson as W3's rigid-advection control and W5's reversed-flow
    # PASS, in the third place it has now appeared.
    scale = float(np.sqrt(np.mean(u1 ** 2)))
    if abs(lam_est) * float(np.abs(wu).max()) > 0.5 * scale:
        return u1, v1, Projection(
            r.name, pre=abs(g0) / max(abs(thrust), 1e-30),
            post=float("nan"), lam=float(lam_est), norm=0.0, alpha=0.0,
            mismatch=mism, sensitivity=float(beta0), refused=True)

    g1, g2 = g(lam_est), g(2.0 * lam_est)
    alpha = (g2 - 2.0 * g1 + g0) / (2.0 * lam_est ** 2)
    beta = (4.0 * g1 - g2 - 3.0 * g0) / (2.0 * lam_est)
    if abs(alpha) < 1e-14:
        lam = -g0 / beta if abs(beta) > 1e-30 else 0.0
    else:
        disc = beta * beta - 4.0 * alpha * g0
        if disc < 0.0:
            lam = -beta / (2.0 * alpha)
        else:
            r1 = (-beta + np.sqrt(disc)) / (2.0 * alpha)
            r2 = (-beta - np.sqrt(disc)) / (2.0 * alpha)
            lam = r1 if abs(r1) <= abs(r2) else r2
    u2, v2 = u1 + lam * wu, v1 + lam * wv
    return u2, v2, Projection(r.name, pre=abs(g0) / max(abs(thrust), 1e-30),
                              post=abs(g(lam)) / max(abs(thrust), 1e-30),
                              lam=float(lam),
                              norm=float(abs(lam) * np.sqrt(np.sum(wu ** 2 + wv ** 2))),
                              alpha=float(alpha), mismatch=mism,
                              sensitivity=float(beta0))


# --------------------------------------------------------------------------
# the coupled system
# --------------------------------------------------------------------------


@dataclass
class CoupleConfig:
    dt: float = DT_MACRO
    a: float = 1.0 / 3.0
    band: float = 0.5
    u_inf: float = 1.0
    schedule: str = "jacobi"           # 'jacobi' | 'gauss_seidel'
    theta: float = 0.5                 # fixed-point relaxation, spec section 8.3
    tol: float = 1e-4
    max_iter: int = 6                  # gate W5: convergence in <= 6
    #: How the pressure is obtained -- the W5 fix. `'outflow'` gives the domain
    #: its real boundary conditions: Neumann at the pinned inlet and the slip
    #: walls, Dirichlet at the open outlet. `'periodic'` is the FFT Leray
    #: projection the 2026-08-21 rollout used, kept as the control the fix has to
    #: be read against. The choice propagates to the momentum balance's pressure
    #: as well, so a run is periodic or it is not, in one place rather than two.
    pressure: str = "outflow"
    #: Enforce C2 at the rotor. This was blocked, and the block was never in the
    #: correction -- it was in the target. What the correction drives to zero is
    #: a residual containing a pressure flux, and on a periodic sub-window
    #: holding a body force that flux is wrong by 100-450% of `T`, so correcting
    #: to it reversed the flow through the disk. Under `pressure = 'outflow'` the
    #: target is a real momentum residual. Refused outright under `'periodic'`
    #: rather than silently producing the old failure.
    project: bool = False
    leray: bool = True
    #: How overlapping tiles are assembled. `'nearest'` gives each cell to its
    #: nearest tile and discards the rest, which is what creates the seam
    #: discontinuities; `'pou'` blends them with a partition of unity. Kept as a
    #: switch rather than replaced outright so the seam statistics can be read
    #: against the thing they were measured on.
    blend: str = "pou"
    #: Which operator runs inside an agent. `'solver'` swaps the frozen
    #: checkpoint for `reference.SolverExpert`, holding the tiling, framing and
    #: forcing identical -- the positive control that makes composition error
    #: measurable at all, since W9 put it 6.5x below the checkpoint's own error.
    expert_kind: str = "frozen"
    #: W9's baseline: one agent over the whole domain, no partition and no
    #: ports. Everything else -- tiles, band, expert, disk, pressure, fixed
    #: point -- identical, so the difference against a partitioned run is the
    #: cost of decomposition and nothing else. See `build_tiles_monolithic`.
    monolithic: bool = False
    accelerate: bool = True            # IQN-ILS (scalar secant) per guide 6.4
    chunk: int = 32
    gust: bool = False
    disk_thickness: float | None = None
    nu: float = None                   # defaults to 1/Re_eff of the scaling
    #: Number of turbines in the line, spaced `turbine_spacing` D apart,
    #: turbine 1 at x=0. `n_turbines=2` (the default) is the original,
    #: hand-designed case study and is untouched by this option's existence --
    #: it drives `geometry.build_geometry_case(2, ...)`, whose output is equal
    #: (not merely equivalent) to the module-level `geometry.AGENTS` etc. See
    #: `wiki` `results-w6-w11-wind-farm.md` for the committed N=2 thesis
    #: numbers this must reproduce.
    n_turbines: int = 2
    turbine_spacing: float = TURBINE_SPACING

    #: Declared symmetry the composition layer must impose on the expert, one of
    #: "none" or "mirror-y". Not a solver option and not tuning: OP-6 measured the
    #: frozen checkpoint returning a 2.42e-2 mirror asymmetry from a *uniform*
    #: inflow in one call, 2.4e4 times gate W7's threshold, with no agents or
    #: tiling in the loop. Equivariance is a property of the operator, so W7 is
    #: unreachable unless the composition layer supplies it. "mirror-y" wraps the
    #: expert in the Reynolds average over {e, M_y} -- exactly 2x the forward
    #: passes, nothing trained. See `atlas.invariants.symmetry`.
    symmetry: str = "none"

    #: What each expert window is solved as. `'periodic'` wraps -- what leaves
    #: the right edge re-enters at the left -- which is what both the frozen
    #: checkpoint and `SpectralNS` do. `'dirichlet'` swaps in `reference.WindowNS`
    #: and takes the window's boundary ring from the surrounding field.
    #:
    #: This is OP-5's candidate (1), and it is a **precondition** of `schwarz`
    #: below rather than a peer of it. A Schwarz fixed point is the global
    #: solution only if each local solve is the exact restriction of the global
    #: operator; a periodic box is not the restriction of an open channel, so
    #: iterating around periodic windows converges to the fixed point of the
    #: wrong map -- the tiles agree with each other more precisely while staying
    #: wrong the same way, and the interface residual that was the only visible
    #: symptom is what the iteration removes.
    #:
    #: Not available to the frozen checkpoint, which was pretrained on periodic
    #: data and has no boundary channel at all. That asymmetry is a result about
    #: frozen experts, not a limitation of this switch.
    window_bc: str = "periodic"
    #: Schwarz iterations of the *field* exchange per macro-step. `1` is the
    #: one-pass exchange the system has always done.
    #:
    #: **Under `window_bc='periodic'` any value above 1 is refused, and the
    #: reason is measured rather than argued.** The assembled sweep is a constant
    #: map in the iteration index: `gather` reads each window out of the global
    #: field at `t^n`, the operator returns one field, and re-running it
    #: reproduces the previous result *bitwise* -- verified directly. There is no
    #: boundary channel to iterate on, so a periodic Schwarz loop is not merely
    #: weak, it is exactly a no-op costing `schwarz` times the expert calls.
    schwarz: int = 1
    #: Stop the Schwarz loop when the interface residual -- the largest change in
    #: the assembled field over overlap cells, relative to `u_inf` -- falls below
    #: this. Read it against the expert's own per-call noise floor (~3e-2 for the
    #: checkpoint, W0): iterating below the floor of the operator being iterated
    #: is measuring arithmetic.
    schwarz_tol: float = 1e-4
    #: Where the rotor body force enters the window solve. `'impulse'` adds
    #: `f*dt` after the step, which is all a frozen one-shot operator can do.
    #: `'rhs'` integrates it inside the sub-stepping, which is what `ChannelNS`
    #: -- the baseline this is all measured against -- does. A switch rather than
    #: a silent change of default, so "non-periodic" and "properly forced" can be
    #: varied one at a time instead of arriving together and being inseparable.
    force_mode: str = "impulse"
    #: What the Dirichlet ring imposes. `'characteristic'` pins the velocity only
    #: where flow **enters** a window and lets it leave under zero normal
    #: gradient; `'dirichlet'` pins the whole ring.
    #:
    #: Pinning an outflow boundary is ill-posed for an advection-dominated flow,
    #: and here the consequence is concrete and measured: with the whole ring
    #: pinned, a forced window drives its wake back to freestream at the outflow
    #: cells (0.9864 against a 0.7970 minimum inside), so **the deficit cannot
    #: leave the window** -- and across 124 tiles that annihilates the wake at
    #: every seam, which is the defect the case study is trying to remove.
    #: Kept as a switch because pure Dirichlet is what classical alternating
    #: Schwarz prescribes and is the right control to read the other against.
    window_transmission: str = "characteristic"
    #: Array backend for the windowed solve. `'numpy'` is the reference path and
    #: the default; `'torch'` with `device='cuda'` is the same arithmetic on a
    #: GPU. Profiling put 95% of a macro-step in the batched stencils and a CPU
    #: thread pool saturates at 2.8x (memory-bandwidth bound), so this is the
    #: only lever that changes the order of magnitude. See `backend.py`.
    #: Applies to `expert_kind='solver'` only.
    backend: str = "numpy"
    #: Torch device, shared by BOTH experts: the `WindowNS` backend above when
    #: `backend='torch'`, and the frozen checkpoint's forward pass when
    #: `expert_kind='frozen'`. The frozen path ignores `backend` entirely -- it
    #: is a network, always torch -- but it very much does not ignore this.
    device: str = "cpu"

    @property
    def pressure_mode(self) -> str:
        """`cv_balance`'s name for the same decision.

        Two names because they answer different questions: `pressure` says what
        the *domain* is (open at the outlet, or wrapped), `pressure_mode` says
        what boundary condition a *control volume's* pressure solve carries. One
        field drives both so a run cannot end up periodic in the projection and
        non-periodic in the measurement -- which is a state that would look like
        a working fix and produce a residual belonging to neither."""
        return "neumann" if self.pressure == "outflow" else "periodic"

    def __post_init__(self):
        if self.schedule not in ("jacobi", "gauss_seidel"):
            raise ValueError(self.schedule)
        if self.pressure not in ("outflow", "periodic"):
            raise ValueError(self.pressure)
        if self.blend not in ("nearest", "pou"):
            raise ValueError(self.blend)
        if self.expert_kind not in ("frozen", "solver"):
            raise ValueError(self.expert_kind)
        if self.symmetry not in ("none", "mirror-y"):
            raise ValueError(self.symmetry)
        if self.window_bc not in ("periodic", "dirichlet"):
            raise ValueError(self.window_bc)
        if self.window_transmission not in ("characteristic", "dirichlet"):
            raise ValueError(self.window_transmission)
        if self.force_mode not in ("impulse", "rhs"):
            raise ValueError(self.force_mode)
        if self.schwarz < 1:
            raise ValueError(f"schwarz must be >= 1, got {self.schwarz}")
        if self.window_bc == "dirichlet" and self.expert_kind != "solver":
            raise ValueError(
                "window_bc='dirichlet' needs expert_kind='solver'. The frozen "
                "checkpoint was pretrained on the periodic unit square and takes one "
                "field in and one field out -- it has no channel through which a "
                "boundary condition could be imposed, so the flag could only be "
                "silently ignored. That the fix for OP-5's dominant term is "
                "unavailable to a frozen expert is the finding, not a gap here.")
        if self.schwarz > 1 and self.window_bc != "dirichlet":
            raise ValueError(
                "schwarz > 1 needs window_bc='dirichlet', and this is a measured fact "
                "rather than a design preference. With periodic windows the assembled "
                "sweep is a CONSTANT MAP in the iteration index: every tile's input is "
                "read out of the global field at t^n and the operator returns one "
                "field, so a second pass reproduces the first BITWISE. Iterating would "
                "cost schwarz times the expert calls and change nothing. A boundary "
                "channel has to exist before iterating on it can mean anything.")
        if self.project and self.pressure != "outflow":
            raise ValueError(
                "project=True needs pressure='outflow'. The thrust projection drives "
                "the momentum residual to zero, and under a periodic pressure that "
                "residual reads 100-450% of T at a rotor against 3% on a turbine-free "
                "volume -- so the correction is aimed at the instrument's error and "
                "reverses the flow through the disk. Blocked at the config rather than "
                "left to be rediscovered by watching a rollout go backwards.")
        if self.disk_thickness is None:
            object.__setattr__(self, "disk_thickness",
                               max(DISK_THICKNESS, self.u_inf * self.dt))                 if False else setattr(self, "disk_thickness",
                                      max(DISK_THICKNESS, self.u_inf * self.dt))
        if self.n_turbines < 1:
            raise ValueError(f"n_turbines must be >= 1, got {self.n_turbines}")
        if self.disk_thickness < self.u_inf * self.dt - 1e-12:
            raise ValueError(
                f"disk_thickness {self.disk_thickness} is thinner than U_inf*dt = "
                f"{self.u_inf * self.dt}. The flow crosses the strip inside one macro-step, "
                "so the impulse T*dt/(A*Delta_d) removes more momentum than the fluid "
                "carries and the disk velocity reverses -- measured, <U_d> = -0.22 at t = 5. "
                "Same Courant condition as the interface band, on the body force instead.")


#: Graph order for the Gauss-Seidel sweep: upstream to downstream, bypasses last
#: because they are fed by the wake agents through the shear interfaces.
GS_ORDER = ("I", "N", "F", "W", "B+", "B-")


@dataclass
class StepRecord:
    step: int
    t: float
    iters: dict
    converged: bool
    thrust: dict
    u_disk: dict
    power: dict
    r_T_pre: dict
    r_T_post: dict
    r_T_null: float
    r_T_null_wake: float
    mass_linf: float
    mass_l2: float
    mass_intra: float | None
    energy: float
    u_max: float
    finite: bool
    mean_drift: float
    #: Conservative (face) divergence after the projection -- the number C1
    #: actually asks about, because summed over any union of cells it telescopes
    #: to that union's net boundary flux. Machine zero on the `outflow` path.
    mass_face: float | None = None
    #: Net mass flux out minus in. Structurally zero under a periodic
    #: projection, which is exactly the constraint that had nowhere to put the
    #: rotors' blockage; free under `outflow`, so a nonzero value here is the
    #: fix working rather than a leak.
    flux_imbalance: float | None = None
    #: Compatibility defect of the rotor window's Neumann pressure solve, as a
    #: fraction of `T`. The error bar `r_T_pre` must be read against.
    p_mismatch: dict | None = None
    #: Rotors whose thrust projection was refused as degenerate this step. A
    #: refused projection leaves C2 measured rather than enforced, which is a
    #: BLOCKED gate item, never a passing one.
    projection_refused: dict | None = None
    #: Schwarz iterations taken on the last fixed-point pass, and the interface
    #: residual they reached -- split into tile seams and declared-port cells,
    #: because those are different populations and the case study has never let
    #: them be quoted as one number. `None` under the one-pass default, where
    #: there is no second iterate to difference against.
    schwarz_iters: int | None = None
    schwarz_seam: float | None = None
    schwarz_port: float | None = None


class CoupledSystem:
    """Eight agents, fifteen declared ports, two frozen experts, nothing trained."""

    def __init__(self, cfg: CoupleConfig | None = None,
                 expert: FrozenFluidExpert | None = None):
        self.cfg = cfg or CoupleConfig()
        self.scaling = COUPLE_SCALING
        self.nu = self.cfg.nu if self.cfg.nu is not None else 1.0 / 255.0
        if expert is not None:
            self.expert = expert
        elif self.cfg.expert_kind == "solver":
            from .reference import SolverExpert
            self.expert = SolverExpert(nu=self.nu, scaling=self.scaling,
                                       window_bc=self.cfg.window_bc,
                                       force_mode=self.cfg.force_mode,
                                       window_transmission=self.cfg.window_transmission,
                                       backend=self.cfg.backend,
                                       device=self.cfg.device)
        else:
            # `device` is shared with the WindowNS backend: on a rented box the
            # frozen checkpoint is the thing that most wants the GPU (it is a
            # network forward over 124 windows), and leaving it on CPU there
            # would look like a GPU that bought nothing.
            self.expert = FrozenFluidExpert(scaling=self.scaling,
                                            device=self.cfg.device)
        if self.cfg.symmetry == "mirror-y":
            # Wrapped *outside* everything the expert does -- Galilean framing,
            # spectral shift, explicit impulse -- because the average has to be
            # over the whole operator the system actually calls. Symmetrizing
            # some inner part of it would leave the rest free to break the
            # symmetry and would still report a group average.
            from atlas.invariants.symmetry import MIRROR_Y, symmetrize
            self.expert = symmetrize(self.expert, MIRROR_Y)
        # The N-turbine agent/interface graph. `n_turbines=2` (the default)
        # reproduces the module-level `geometry.AGENTS`/`INTERFACES` exactly
        # (see `geometry.build_geometry_case`'s docstring and
        # `tests/atlas/windfarm/test_geometry_generator.py`), so this is
        # behaviour-preserving at N=2 by construction.
        self.case: GeometryCase = build_geometry_case(self.cfg.n_turbines,
                                                       self.cfg.turbine_spacing)
        if self.cfg.monolithic:
            self.tiles = build_tiles_monolithic(self.scaling.length, self.cfg.band,
                                                agents=self.case.agents)
            self.g = build_global(self.scaling.length, self.cfg.u_inf, domain=self.case.domain)
            self.own, self.best = build_ownership_monolithic(self.g, self.tiles)
        else:
            self.tiles = build_tiles(self.scaling.length, self.cfg.band,
                                     agents=self.case.agents)
            self.g = build_global(self.scaling.length, self.cfg.u_inf, domain=self.case.domain)
            self.own, self.best = build_ownership(self.g, self.tiles, owner_fn=self.case.owner,
                                                  field_owner_remap=self.case.field_owner_remap)
        apply_outer_bc(self.g, 0.0, self.cfg.band, self.cfg.u_inf, self.cfg.gust)
        th = self.cfg.disk_thickness
        self.rotors = [Rotor(name, x, ActuatorDisk(a=self.cfg.a, thickness=th))
                       for name, x in zip(self.case.rotor_names, self.case.positions)]
        self.thrust = {r.name: 0.0 for r in self.rotors}
        self.step_index = 0
        self.t = 0.0
        self.proj: ProjectionReport | None = None      # last projection's report
        self._u_inlet = self.g.u[:, 0].copy()          # the prescribed inlet column
        self.history: list[StepRecord] = []
        self._by_agent = {}
        for t in self.tiles:
            self._by_agent.setdefault(t.agent, []).append(t)
        self._mask_seam, self._mask_port = self._overlap_masks()
        self.last_schwarz = {"iters": 1, "seam": None, "port": None, "trace": []}

    def _overlap_masks(self):
        """Where an interface residual means something, split two ways.

        The case study has always insisted these are different populations and
        the split has to survive into this measurement. **Tile seams** are cells
        two or more windows of the *same* agent predict -- an artefact of the
        expert's fixed input shape, not a declared interface, and the thing
        `scatter_pou` blends. **Port cells** are cells covered by tiles of more
        than one agent: the neighbourhood of a declared R2 interface, which is
        the contract actually under test and which blending deliberately leaves
        alone.

        The prediction worth recording before the numbers arrive: the tile
        overlap is `0.5 D` against an advective distance of `U_inf*dt = 0.25 D`,
        so seams sit a factor of two *inside* the one-iteration condition, while
        the band rule pins the port width at exactly `U_inf*dt` -- i.e. *on* it.
        Iterating should therefore do measurably more at ports than at seams. If
        it comes out the other way round, the reasoning is wrong somewhere and
        that is worth more than the fix."""
        n = EXPERT_RES
        ny, nx = self.g.shape
        cover = np.zeros(self.g.shape, dtype=np.int16)
        agents = np.zeros(self.g.shape, dtype=object)
        first = np.empty(self.g.shape, dtype=object)
        first[:] = None
        multi = np.zeros(self.g.shape, dtype=bool)
        for t in self.tiles:
            i0, j0 = _slice_of(self.g, t)
            i1, j1 = min(i0 + n, nx), min(j0 + n, ny)
            a0, b0 = max(i0, 0), max(j0, 0)
            if a0 >= i1 or b0 >= j1:
                continue
            sl = (slice(b0, j1), slice(a0, i1))
            cover[sl] += 1
            seen = first[sl]
            multi[sl] |= (seen != None) & (seen != t.agent)      # noqa: E711
            first[sl] = np.where(seen == None, t.agent, seen)    # noqa: E711
        del agents
        return (cover >= 2) & ~multi, multi

    # -- pieces -----------------------------------------------------------

    def _rotor_tiles(self, r: Rotor) -> list[Tile]:
        out = []
        for t in self.tiles:
            w = t.window
            if (w.x0 <= r.x_face <= w.x0 + w.length
                    and w.y0 <= 0.0 <= w.y0 + w.length):
                out.append(t)
        return out

    def _advance(self, tiles: list[Tile], fx_global: np.ndarray | None,
                 bc_g: "GlobalField | None" = None):
        """Gather, force, step, project, for a list of tiles.

        `bc_g` is the Schwarz transmission field -- the current iterate's
        estimate of the domain at `t + dt`. Each tile's *ring* is read out of it
        while the tile's *interior initial condition* still comes from `self.g`
        at `t^n`, and keeping those two separate is the whole content of the
        iteration. `None` reproduces the one-pass behaviour exactly."""
        U, V = gather(self.g, tiles)
        force = None
        if fx_global is not None:
            F = np.zeros_like(U)
            n = EXPERT_RES
            ny, nx = self.g.shape
            for pos, t in enumerate(tiles):
                i0, j0 = _slice_of(self.g, t)
                ii = np.clip(np.arange(i0, i0 + n), 0, nx - 1)
                jj = np.clip(np.arange(j0, j0 + n), 0, ny - 1)
                F[pos] = fx_global[np.ix_(jj, ii)]
            force = (F, np.zeros_like(F))
        # one frame for the whole domain -- see `step_many`'s `frame` argument
        kw = {}
        if bc_g is not None:
            Ub, Vb = gather(bc_g, tiles)
            kw["bc"] = (Ub, Vb)
        if self.cfg.window_bc == "dirichlet":
            # Which faces are genuinely open. Without this every tile is given an
            # open outlet on whichever faces the flow happens to leave through,
            # including interior seams where the neighbour's data is the answer.
            kw["open_faces"] = tile_open_faces(self.g, tiles)
        U1, V1 = self.expert.step_many(U, V, self.cfg.dt, project=False,
                                       force=force, chunk=self.cfg.chunk,
                                       frame=(self.cfg.u_inf, 0.0), **kw)
        return U, V, U1, V1

    def _scatter(self, tiles, U, V) -> None:
        """Assemble a sweep's tiles into the global field, by the configured rule."""
        if self.cfg.blend == "pou":
            scatter_pou(self.g, tiles, U, V, self.own, self.cfg.band / self.g.dx)
        else:
            scatter(self.g, tiles, U, V, self.best)

    def _sweep(self, fx: np.ndarray):
        """One macro-step of every agent, Schwarz-iterated if configured.

        `cfg.schwarz == 1` is the historical one-pass exchange and takes the
        `_sweep_once` path unchanged, byte for byte.

        Above 1 this is **additive** Schwarz: every tile is advanced from the
        same `t^n` interior state, reading its transmission ring from the
        previous iterate's assembled field, and all tiles are updated together.
        Multiplicative (sequential) Schwarz converges in fewer iterations and is
        refused on purpose. It would reinstall exactly the sweep-order path
        dependence that gate W7's reattribution ruled out, and -- the harder
        reason -- it would break `symmetry.SymmetrizedExpert`'s exactness, since
        mirror-paired tiles visited at different points of the iteration no
        longer see equivalent data. Two composition-layer guarantees that cannot
        both hold; the framework keeps the one that is exact.

        The interface residual is recorded per iteration and split into tile
        seams and declared-port cells, because the case study has never let those
        two populations be quoted as one number."""
        cfg = self.cfg
        if cfg.schwarz == 1:
            self.last_schwarz = {"iters": 1, "seam": None, "port": None, "trace": []}
            return self._sweep_once(fx, None)

        base_u, base_v = self.g.u.copy(), self.g.v.copy()
        prev_u = prev_v = None
        bc_g = None
        trace = []
        seam = port = None
        k = 0
        for k in range(cfg.schwarz):
            # the interior initial condition is ALWAYS t^n; only the ring moves
            self.g.u, self.g.v = base_u.copy(), base_v.copy()
            last_U, last_V = self._sweep_once(fx, bc_g)
            if prev_u is not None:
                d = np.maximum(np.abs(self.g.u - prev_u),
                               np.abs(self.g.v - prev_v)) / cfg.u_inf
                seam = float(d[self._mask_seam].max()) if self._mask_seam.any() else 0.0
                port = float(d[self._mask_port].max()) if self._mask_port.any() else 0.0
                trace.append({"iter": k, "seam": seam, "port": port})
            prev_u, prev_v = self.g.u.copy(), self.g.v.copy()
            bc_g = GlobalField(self.g.dx, self.g.x_c, self.g.y_c, prev_u, prev_v)
            if seam is not None and max(seam, port) < cfg.schwarz_tol:
                break
        self.last_schwarz = {"iters": k + 1, "seam": seam, "port": port, "trace": trace}
        return last_U, last_V

    def _sweep_once(self, fx: np.ndarray, bc_g: "GlobalField | None" = None):
        """One full macro-step of every agent, in the configured order.

        This is *the* operator: the fixed point below iterates on exactly this,
        and `step` keeps whatever it converged to. An earlier version iterated
        the two rotor tiles alone and then did a separate full sweep -- so the
        fixed point converged to a different problem than the one that was then
        solved, and `T` stopped satisfying its own formula: at t = 1.5 it sat at
        0.624 while `<U_d>` was 0.598, which the disk expert says should give
        0.357. That inconsistency is what drove the runaway, not the timestep
        and not the strip thickness."""
        cfg = self.cfg
        last_U = last_V = None
        if cfg.schedule == "jacobi":
            _, _, U1, V1 = self._advance(self.tiles, fx, bc_g)
            self._scatter(self.tiles, U1, V1)
            last_U, last_V = U1, V1
        else:
            for name in self.case.gs_order:
                grp = self._by_agent.get(name, [])
                if not grp:
                    continue
                _, _, U1, V1 = self._advance(grp, fx, bc_g)
                self._scatter(grp, U1, V1)
                last_U, last_V = U1, V1        # last group: the same diagnostic
        if cfg.pressure == "periodic":
            if cfg.leray:
                global_project(self.g)
            apply_outer_bc(self.g, self.t + cfg.dt, cfg.band, cfg.u_inf, cfg.gust)
        else:
            # Boundary condition FIRST, projection second, which is the reverse
            # of the periodic path and is required rather than tidier. The
            # projection's inlet is Neumann, so it cannot alter the mass flux
            # through the inlet face -- it inherits whatever `apply_outer_bc`
            # imposed and leaves it exactly. Run the other way round, the
            # boundary overwrite is applied *after* the projection and puts a
            # fresh divergence back into the inlet band, which is the one place
            # C1 is then guaranteed to be violated.
            apply_outer_bc(self.g, self.t + cfg.dt, cfg.band, cfg.u_inf, cfg.gust)
            self._u_inlet = self.g.u[:, 0].copy()
            if cfg.leray:
                self.proj = global_project_outflow(self.g)
        return last_U, last_V

    def _fixed_point(self):
        """Spec section 8.3: resolve `T <-> <U_d>` within the macro-step.

        Both rotors are advanced together on the full sweep -- block-Jacobi over
        the two scalars, each accelerated by the secant, which is what IQN-ILS
        degenerates to for a scalar interface. The field left behind by the
        converged iteration *is* the macro-step's answer, so no extra sweep is
        needed after it.

        Non-convergence is a gate, not a bug to smother. Under plain relaxation
        this converges monotonically in 8-9 iterations; the secant does it in
        3-4. Monotone convergence is the substantive part: the frozen expert
        responds monotonically to a body force, so falsification criterion F2 is
        not triggered."""
        base_u, base_v = self.g.u.copy(), self.g.v.copy()
        names = [r.name for r in self.rotors]
        T = {}
        for r in self.rotors:
            T[r.name] = self.thrust[r.name] or r.disk(
                u_disk=self.cfg.u_inf * (1 - self.cfg.a)).thrust
        prev_T = {k: None for k in names}
        prev_R = {k: None for k in names}
        traces = {k: [] for k in names}
        last_U = last_V = None

        for it in range(self.cfg.max_iter + 1):
            self.g.u, self.g.v = base_u.copy(), base_v.copy()
            fx = np.zeros(self.g.shape)
            for r in self.rotors:
                fx += rotor_force(self.g, r, T[r.name])
            last_U, last_V = self._sweep(fx)

            rel_max = 0.0
            nxt = {}
            for r in self.rotors:
                ud = disk_velocity(self.g, r)
                G = r.disk(u_disk=ud).thrust
                R = G - T[r.name]
                rel = abs(R) / max(abs(G), 1e-30)
                rel_max = max(rel_max, rel)
                mode = "relax"
                if (self.cfg.accelerate and prev_T[r.name] is not None
                        and abs(R - prev_R[r.name]) > 1e-14):
                    nxt[r.name] = T[r.name] - R * (T[r.name] - prev_T[r.name]) / (R - prev_R[r.name])
                    mode = "secant"
                else:
                    nxt[r.name] = (1 - self.cfg.theta) * T[r.name] + self.cfg.theta * G
                traces[r.name].append({"iter": it, "T": float(T[r.name]),
                                       "u_disk": float(ud), "T_new": float(G),
                                       "rel": float(rel), "mode": mode})
                prev_T[r.name], prev_R[r.name] = T[r.name], R
            if rel_max < self.cfg.tol:
                break
            T = {k: float(v) for k, v in nxt.items()}

        # keep the converged iterate's field and the T that produced it
        self.thrust = {k: float(traces[k][-1]["T"]) for k in names}
        return traces, last_U, last_V, (base_u, base_v)

    def _mass_residual(self, U=None, V=None) -> dict:
        """Divergence, split into the part the projection controls and the part it does not.

        The level-4 path of W0. Inside a window the Leray projection is spectral
        and exact, so the intra-tile divergence is machine zero. The *assembled*
        field is a different object: neighbouring tiles were projected
        independently and their outputs do not match across a seam, so the
        divergence there is not controlled by anything.

        Reporting one number for both would hide that. The spec's 1e-8 is
        available on the stream-function path and is **not** available here;
        this is the documented alternative the W0 fallback allows."""
        if self.cfg.pressure == "outflow":
            # The boundary-aware operator the projection actually inverts, and
            # over the WHOLE domain -- the four boundary cells included. The old
            # diagnostic trimmed four cells off each edge, which is exactly
            # where a boundary-condition error lives, so it could not have seen
            # one. Nothing is trimmed now because nothing needs to be.
            d = divergence_bc(self.g.u, self.g.v, self.g.dx, self._u_inlet)
        else:
            d = (np.gradient(self.g.u, self.g.dx, axis=1)
                 + np.gradient(self.g.v, self.g.dx, axis=0))[4:-4, 4:-4]
        out = {"assembled_linf": float(np.abs(d).max()),
               "assembled_l2": float(np.sqrt(np.mean(d ** 2)))}
        if U is not None:
            per = [float(np.abs(divergence(U[i], V[i], self.scaling.length)).max())
                   for i in range(U.shape[0])]
            out["intra_tile_linf"] = float(np.max(per))
        return out

    def _null_cv(self, u_before, v_before) -> dict:
        """The same balance operator on control volumes with **no body force in them**.

        The W4 discipline applied to W5: a residual is only readable against its
        own floor. Two floors, because they separate two different things:

        `inflow`  quiet, nearly uniform flow upstream of everything. Tests the
                  operator itself.
        `wake`    2.5 D downstream of rotor 1 -- strongly disturbed flow, sheared
                  and unsteady, but **no force term and no pressure jump**. Tests
                  whether the operator survives a hard flow, which is what
                  separates "the expert does not conserve momentum" from "the
                  pressure recovery fails where the body force is".

        Both normalized by the same thrust as `r_T`, so all three are directly
        comparable, and recorded every step so the comparison is never lost."""
        out = {}
        x_t1 = self.case.positions[0]
        first_rotor = self.case.rotor_names[0]
        for name, xc in (("inflow", -4.0), ("wake", x_t1 + 2.5)):
            w = Window(x0=xc - COUPLE_SCALING.length / 2,
                       y0=-COUPLE_SCALING.length / 2, length=COUPLE_SCALING.length)
            sl = window_slice(self.g, w)
            b = cv_balance(u_before[sl], v_before[sl], self.g.u[sl], self.g.v[sl],
                           w, xc - 0.55, xc + 0.55, -0.9, 0.9, self.cfg.dt, self.nu,
                           pressure_mode=self.cfg.pressure_mode)
            out[name] = float(abs(b.residual) / max(abs(self.thrust.get(first_rotor, 0.0)), 1e-30))
        return out

    # -- the macro-step ---------------------------------------------------

    def step(self) -> StepRecord:
        cfg = self.cfg
        traces, last_U, last_V, (u_before, v_before) = self._fixed_point()
        converged = all(t[-1]["rel"] < cfg.tol for t in traces.values())

        fx = np.zeros(self.g.shape)
        for r in self.rotors:
            fx += rotor_force(self.g, r, self.thrust[r.name])

        # -- thrust projection at the disk faces, pre-projection recorded ----
        pre, post, mismatch, refused = {}, {}, {}, {}
        for r in self.rotors:
            w = rotor_window(r)
            sl = window_slice(self.g, w)
            u0w, v0w = u_before[sl], v_before[sl]
            u1w, v1w = self.g.u[sl].copy(), self.g.v[sl].copy()
            u2, v2, pr = project_thrust(u0w, v0w, u1w, v1w, w, r,
                                        self.thrust[r.name], cfg.dt, self.nu, fx[sl],
                                        pressure_mode=cfg.pressure_mode)
            if cfg.project:
                # a divergence-free perturbation, so it is ADDED over the window
                # footprint rather than written over it -- nothing another tile
                # owns gets clobbered
                self.g.u[sl] += (u2 - u1w)
                self.g.v[sl] += (v2 - v1w)
            pre[r.name] = pr.pre
            post[r.name] = pr.post if cfg.project else None
            mismatch[r.name] = pr.mismatch
            if pr.refused:
                # Not silently skipped: a refused projection means C2 is measured
                # rather than enforced on this step, and the gate has to see that.
                post[r.name] = None
                refused[r.name] = True

        nulls = self._null_cv(u_before, v_before)
        mass = self._mass_residual(last_U, last_V)
        self.step_index += 1
        self.t += cfg.dt
        rec = StepRecord(
            step=self.step_index, t=self.t,
            iters={k: len(v) for k, v in traces.items()},
            converged=bool(converged),
            thrust={k: float(v) for k, v in self.thrust.items()},
            u_disk={r.name: disk_velocity(self.g, r) for r in self.rotors},
            power={r.name: float(r.disk(u_disk=disk_velocity(self.g, r)).power)
                   for r in self.rotors},
            r_T_pre=pre, r_T_post=post, r_T_null=nulls['inflow'], r_T_null_wake=nulls['wake'],
            mass_linf=mass["assembled_linf"], mass_l2=mass["assembled_l2"],
            mass_intra=mass.get("intra_tile_linf"),
            mass_face=(float(self.proj.div_face_after) if self.proj else None),
            flux_imbalance=(float(self.proj.net_flux_out - self.proj.net_flux_in)
                            if self.proj else None),
            p_mismatch=mismatch, projection_refused=refused or None,
            energy=float(0.5 * np.mean(self.g.u ** 2 + self.g.v ** 2)),
            u_max=float(np.abs(self.g.u).max()),
            finite=bool(np.all(np.isfinite(self.g.u)) and np.all(np.isfinite(self.g.v))),
            mean_drift=float(self.expert.last_mean_drift[0]),
            schwarz_iters=self.last_schwarz["iters"],
            schwarz_seam=self.last_schwarz["seam"],
            schwarz_port=self.last_schwarz["port"],
        )
        self.history.append(rec)
        return rec

    def run(self, t_end: float, on_step=None):
        n = int(round(t_end / self.cfg.dt))
        for _ in range(n):
            rec = self.step()
            if on_step is not None:
                on_step(rec)
            if not rec.finite:
                break
        return self.history


__all__ = [
    "DT_MACRO", "COUPLE_SCALING", "MIN_BAND", "Tile", "build_tiles", "GlobalField",
    "build_global", "FIELD_OWNER_REMAP", "build_ownership", "gather", "scatter",
    "MONOLITHIC_AGENT", "build_tiles_monolithic", "build_ownership_monolithic",
    "taper_1d", "tile_weight", "scatter_pou", "tile_open_faces",
    "inflow_profile", "apply_outer_bc", "global_project", "global_project_outflow",
    "CVBalance", "cv_balance", "window_slice",
    "Rotor", "rotor_force", "strip_fraction", "rotor_window", "disk_velocity", "Projection", "project_thrust",
    "CoupleConfig", "GS_ORDER", "StepRecord", "CoupledSystem",
    "CoupledField", "symmetry_of", "energy_per_agent",
]


# --------------------------------------------------------------------------
# the bridge from the coupled system to the W4 metrics
# --------------------------------------------------------------------------


class CoupledField(Field):
    """A `Field` view of the coupled system, so W4's metrics can read it.

    **This is the piece that was missing, and its absence is worth naming.**
    W4 built the whole metric suite -- fifteen per-port residuals, the mirror
    symmetry residual, the Jensen/BPA corridors, the global power residual
    `R(t)` -- and validated every one of them against fields with known answers,
    exactly as the guide's section 5 demands. But every one of those metrics
    takes an *analytic* `Field`, an object with a `velocity(x, y)` method, while
    the coupled system produces a lattice. So the metrics and the system they
    were built for had **no way to be connected at all**, and gates W7, W8, W10
    and W11 were unreachable for that reason alone rather than for any physical
    one. One adapter class unblocks four gates.

    Three methods, and the second is the one that only became possible on
    2026-08-22:

    * `velocity` -- bilinear sampling of the assembled field, clamped at the
      domain edge. Bilinear rather than nearest because `symmetry_residual`
      samples at `+y` and `-y`, and on an even-sized lattice neither lands on a
      cell centre; nearest-neighbour would round the two to cells at different
      distances from the axis and report a mirror asymmetry that is purely the
      sampling. Gate W7 asks for `< 1e-6`, so an artefact at the level of half a
      cell would swamp the measurement.
    * `pressure` -- a real one, from `pressure_neumann` over the whole domain
      with the momentum equation's own Neumann data. Before the pressure solve
      existed this had to be declared absent (`has_pressure = False`), which is
      why `Field` carries that flag: a metric that needs pressure must be able
      to find out it has none rather than silently read zeros. It is now `True`.
    * `body_force` -- the two rotor strips at their converged thrusts, so the
      power ledger's `P_ext` term is the one the run actually applied.
    """

    has_pressure = True

    def __init__(self, system: "CoupledSystem"):
        self.g = system.g
        self.system = system
        self._p = None

    # -- sampling ----------------------------------------------------------

    def _bilinear(self, arr, x, y):
        g = self.g
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        fi = (x - g.x_c[0]) / g.dx
        fj = (y - g.y_c[0]) / g.dx
        ny, nx = arr.shape
        i0 = np.clip(np.floor(fi).astype(int), 0, nx - 2)
        j0 = np.clip(np.floor(fj).astype(int), 0, ny - 2)
        tx = np.clip(fi - i0, 0.0, 1.0)
        ty = np.clip(fj - j0, 0.0, 1.0)
        a = arr[j0, i0] * (1 - tx) + arr[j0, i0 + 1] * tx
        b = arr[j0 + 1, i0] * (1 - tx) + arr[j0 + 1, i0 + 1] * tx
        return a * (1 - ty) + b * ty

    def velocity(self, x, y):
        return self._bilinear(self.g.u, x, y), self._bilinear(self.g.v, x, y)

    def pressure(self, x, y):
        if self._p is None:
            self._p = pressure_neumann(self.g.u, self.g.v, self.force_field(),
                                       None, self.g.dx, nu=self.system.nu)
        return self._bilinear(self._p, x, y)

    # -- the forcing the run actually applied -------------------------------

    def force_field(self) -> np.ndarray:
        fx = np.zeros(self.g.shape)
        for r in self.system.rotors:
            fx += rotor_force(self.g, r, self.system.thrust[r.name])
        return fx

    def body_force(self, x, y):
        """The disk force **evaluated analytically**, not sampled off the lattice.

        `metrics._integrate_rect` point-samples this at Gauss nodes. The lattice
        field `force_field` returns is exact in a different sense -- each cell
        holds the *area fraction* of the strip inside it, which makes the
        discrete sum exactly `-T` on any lattice (gate W2) -- but those are cell
        averages, not point values, and bilinearly interpolating them and then
        Gauss-integrating does not recover the integral. Measured on a field
        with a known answer: the power ledger's work term came out `0.2708`
        against an exact `0.2963`, an 8.6% error, entirely in this term.

        This is the W4 entry's "fractional occupancy is not a quadrature weight"
        in its second incarnation, and the resolution is the same both times --
        use the closed form where one exists. The strip is a rectangle and the
        force is piecewise constant on it, so a point evaluation is exact."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        fx = np.zeros_like(x)
        for r in self.system.rotors:
            inside = ((x >= r.x_face) & (x <= r.x_back)
                      & (np.abs(y) <= ROTOR_HALFSPAN))
            fx = fx - np.where(inside, r.disk.force_density(
                self.system.thrust[r.name]), 0.0)
        return fx, np.zeros_like(x)

    # -- what the power ledger needs ---------------------------------------

    def extracted_power(self) -> float:
        """`sum T <U_d>` over the rotors -- the `P_ext` of spec section 7.4."""
        total = 0.0
        for r in self.system.rotors:
            total += float(r.disk(u_disk=disk_velocity(self.g, r)).power)
        return total


def symmetry_of(system: "CoupledSystem") -> float:
    """Gate W7's mirror residual, read straight off a coupled run.

    Separate from `CoupledField` so the failure mode is legible: W7 does not
    need pressure, forcing, or any of the rest, and a symmetry number that
    depended on a pressure solve would be harder to trust than one that does
    not."""
    from .metrics import symmetry_residual
    return symmetry_residual(CoupledField(system))


def energy_per_agent(field: "CoupledField") -> dict:
    """Kinetic energy per agent, on the same quadrature the power ledger uses.

    `metrics.power_ledger` takes `d_energy_dt` per agent and defaults it to
    zero, which its own docstring says is exact only for the steady analytic
    fields W4 validated against and **must be supplied by the caller for
    anything else**. The coupled system is not steady -- OP-2 records that its
    upstream induction is still climbing at t = 20 -- so a ledger run on a
    snapshot with the default omits a real term.

    Measured cost of omitting it: the residual read 88% of extracted power at
    t = 5, which looks like a catastrophic conservation failure and is mostly
    the missing `dE/dt`. Computed here on the *identical* quadrature
    `_area_integral` uses, so the difference of two of these is consistent with
    the flux and dissipation terms it will be added to rather than merely close
    to them."""
    from .metrics import _area_integral
    agents = field.system.case.agents if hasattr(field, "system") else AGENTS
    return {a.name: _area_integral(field, a.name, "energy", 1.0) for a in agents}
