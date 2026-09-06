"""The seven agents and the seven typed edges, plus the build-time assertions.

`ATLAS_01_AGENTS` and `ATLAS_01_EDGES` are built from `config/atlas_0_1.yaml` at
import, and the consistency assertions run at import too (phase-1 spec 3.2) --
not in tests only. A geometry change that breaks the partition should fail the
moment anything imports Atlas, not later, in one test nobody ran.

Grid representation is deliberately cell-centred (`z_c`, `y_c`, `area`, `active`)
rather than node-based. Two of the seven agents are two-panel maps (the shell
`c`, the body-fitted atmosphere `d`) whose y coordinate jumps across the vehicle,
which a single [n_z+1, n_y+1] node array cannot express without special cases.
Cell centres and areas are all the tokenizer, the patch layout, and any later
flux integration actually need.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..config import AgentCfg, AtlasConfig, GeometryCfg, load_config
from .contours import InterfaceCurve, build_interface, nozzle_inner, plume_half, shell_outer

EPS = 1e-12


# --------------------------------------------------------------------------
# grids
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class GridSpec:
    """One agent's discretization.

    z_c    [n_z]        cell-centre axial coordinate
    y_c    [n_z, n_y]   cell-centre transverse coordinate (z-dependent for the
                        body-fitted maps)
    area   [n_z, n_y]   cell area, m^2 per unit depth
    active [n_z, n_y]   cell lies inside the agent's physical domain
    in_hole[n_z, n_y]   cell lies inside ANOTHER agent's domain carved out of
                        this one's bounding box (only agent `g`, whose hole is
                        the plume). Distinct from `~active`: an inactive cell
                        outside a nozzle wall is exterior to the flow, and its
                        patch is KEPT because that patch carries the wall
                        interface; a hole cell belongs to a different agent, and
                        a patch entirely inside a hole is dropped.
    """
    n_z: int
    n_y: int
    z_c: np.ndarray
    y_c: np.ndarray
    area: np.ndarray
    active: np.ndarray
    in_hole: np.ndarray

    @property
    def points(self) -> np.ndarray:
        """[n_z*n_y, 2] cell centres, row-major (z-major)."""
        z = np.broadcast_to(self.z_c[:, None], self.y_c.shape)
        return np.stack([z.ravel(), self.y_c.ravel()], axis=1)

    @property
    def active_points(self) -> np.ndarray:
        return self.points[self.active.ravel()]


def build_grid(a: AgentCfg, cfg: AtlasConfig) -> GridSpec:
    """The model's cells ARE the Phase-0 finite-volume cells.

    Derived from `solvers.grid.build_blocks` rather than re-deriving the y map
    here. Two independent derivations drifted at exactly the places that matter:
    a block's cell centroid averages the contour at the two NODE stations, while
    a closed-form y(z_centre) evaluates it at the midpoint, and those differ
    wherever the contour has a kink -- 1.7 mm on the 8 mm shell at the throat.
    One derivation, one answer, and the "no interpolation between solver and
    model" claim is true by construction instead of by coincidence."""
    from ..solvers.grid import build_blocks           # local: solvers import geometry

    blocks = build_blocks(a, cfg)
    z_c = blocks[0].centroid[:, 0, 0].copy()
    y_c = np.concatenate([b.centroid[..., 1] for b in blocks], axis=1)
    area = np.concatenate([b.vol for b in blocks], axis=1)
    in_hole = np.concatenate([b.blanked for b in blocks], axis=1)
    if (y_c.shape[0], y_c.shape[1]) != tuple(a.grid):
        raise AssertionError(f"agent {a.id}: blocks give {y_c.shape}, config says {a.grid}")
    return GridSpec(a.grid[0], a.grid[1], z_c, y_c, area, ~in_hole, in_hole)


# --------------------------------------------------------------------------
# specs
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class AgentSpec:
    name: str                       # 'a' ... 'g'
    kind: str                       # reacting | confined_gas | solid | external_gas | free_jet
    tokenizer: str                  # parameter-sharing group (d, f, g share one)
    fields: tuple[str, ...]
    grid: GridSpec
    patch: tuple[int, int]
    dt_model: float
    expert: str                     # reacting_flow | external_flow | thermostruct | rigid_body
    cond_names: tuple[str, ...]
    y_map: str                      # uniform | shell | wall_stretched

    @property
    def two_panel(self) -> bool:
        """True when the y index runs over two disjoint sheets separated by the
        vehicle -- no derivative or patch may straddle the seam."""
        return self.y_map in ("shell", "wall_stretched")

    @property
    def n_patch(self) -> tuple[int, int]:
        return self.grid.n_z // self.patch[0], self.grid.n_y // self.patch[1]


@dataclass(frozen=True, eq=False)
class EdgeSpec:
    src: str
    dst: str
    types: tuple[str, ...]
    type_ids: tuple[int, ...]
    curve: InterfaceCurve

    @property
    def key(self) -> tuple[str, str]:
        return (self.src, self.dst)


def build_agents(cfg: AtlasConfig) -> tuple[AgentSpec, ...]:
    out = []
    for a in cfg.agents:
        pz, py = cfg.patch_size(a)
        if a.grid[0] % pz or a.grid[1] % py:
            raise ValueError(f"agent {a.id}: grid {a.grid} not divisible by patch {(pz, py)}")
        out.append(AgentSpec(
            name=a.id, kind=a.kind, tokenizer=a.tokenizer, fields=a.fields,
            grid=build_grid(a, cfg), patch=(pz, py),
            dt_model=float(cfg.dt_model[a.id]), expert=a.expert,
            cond_names=cfg.cond_names(a), y_map=a.y_map,
        ))
    return tuple(out)


def build_edges(cfg: AtlasConfig) -> tuple[EdgeSpec, ...]:
    return tuple(
        EdgeSpec(
            src=e.src, dst=e.dst, types=e.types,
            type_ids=tuple(cfg.type_index(t) for t in e.types),
            curve=build_interface(e.interface, cfg.geometry),
        )
        for e in cfg.edge_list
    )


# --------------------------------------------------------------------------
# continuum domain membership -- the authority the grids are checked against
# --------------------------------------------------------------------------


def contains(agent: str, z, y, geo: GeometryCfg) -> np.ndarray:
    """Is (z, y) inside agent `agent`'s declared domain? Closed sets: points on a
    shared face belong to both neighbours, which is why the overlap assertion
    tests cell CENTRES (never on a face) and the coverage assertion tests an
    off-lattice sample."""
    z = np.asarray(z, dtype=np.float64)
    y = np.abs(np.asarray(y, dtype=np.float64))
    h_in = nozzle_inner(z, geo)
    h_out = shell_outer(z, geo)
    t = geo.shell_thickness
    far = geo.farfield_halfwidth
    hp = geo.plume_halfwidth
    pz0, pz1 = geo.plume_z
    if agent == "a":
        return (z >= geo.z_injector) & (z <= 0.12) & (y <= h_in)
    if agent == "b":
        return (z >= 0.12) & (z <= geo.z_throat) & (y <= h_in)
    if agent == "e":
        return (z >= geo.z_throat) & (z <= geo.z_exit) & (y <= h_in)
    if agent == "c":
        return (z >= geo.shell_z[0]) & (z <= geo.shell_z[1]) & (y >= h_in) & (y <= h_in + t)
    if agent == "d":
        return (z >= geo.atmos_front_z[0]) & (z <= geo.atmos_front_z[1]) & (y >= h_out) & (y <= far)
    if agent == "f":
        return (z >= pz0) & (z <= pz1) & (y <= hp)
    if agent == "g":
        return (z >= pz0) & (z <= pz1) & (y >= hp) & (y <= far)
    raise KeyError(agent)


def _unmodelled(cfg: AtlasConfig, z, y) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    y = np.abs(np.asarray(y, dtype=np.float64))
    out = np.zeros(z.shape, dtype=bool)
    for r in cfg.unmodelled:
        out |= (z >= r.z[0]) & (z <= r.z[1]) & (y <= r.y_halfwidth)
    return out


# --------------------------------------------------------------------------
# build-time assertions
# --------------------------------------------------------------------------


def verify(agents: tuple[AgentSpec, ...], edges: tuple[EdgeSpec, ...], cfg: AtlasConfig) -> None:
    geo = cfg.geometry
    ids = [a.name for a in agents]

    # 1. every active cell centre lies in exactly one agent's declared domain.
    for a in agents:
        pts = a.grid.active_points
        hits = np.zeros(pts.shape[0], dtype=np.int64)
        for other in ids:
            hits += contains(other, pts[:, 0], pts[:, 1], geo).astype(np.int64)
        if not np.all(hits == 1):
            bad = pts[hits != 1][:3]
            raise AssertionError(
                f"agent {a.name}: {int((hits != 1).sum())} active cells are in "
                f"{'no' if (hits == 0).any() else 'more than one'} declared domain, e.g. {bad}"
            )

    # 2. the agents plus the declared unmodelled regions cover the bounding box.
    #    Off-lattice sample so no probe lands exactly on a shared face.
    zs = np.linspace(geo.atmos_front_z[0], geo.plume_z[1], 601)[:-1] + 0.00137
    ys = np.linspace(-geo.farfield_halfwidth, geo.farfield_halfwidth, 401)[:-1] + 0.00071
    ZZ, YY = np.meshgrid(zs, ys, indexing="ij")
    cov = _unmodelled(cfg, ZZ, YY)
    for other in ids:
        cov |= contains(other, ZZ, YY, geo)
    if not cov.all():
        gap = np.stack([ZZ[~cov], YY[~cov]], axis=1)[:3]
        raise AssertionError(f"{int((~cov).sum())} sample points covered by no agent, e.g. {gap}")

    # 3. every edge curve is non-degenerate and lies on BOTH agents' boundaries.
    trees = {a.name: cKDTree(a.grid.active_points) for a in agents}
    diag = {a.name: float(np.hypot(
        a.patch[0] * (a.grid.z_c[1] - a.grid.z_c[0]),
        a.patch[1] * float(np.median(np.abs(np.diff(a.grid.y_c, axis=1)))),
    )) for a in agents}
    for e in edges:
        if e.curve.length <= 0.0:
            raise AssertionError(f"edge {e.src}-{e.dst}: zero-length interface curve")
        probes = np.concatenate([b.points for b in e.curve.branches], axis=0)
        tol = cfg.tol_patch_diag * max(diag[e.src], diag[e.dst])
        for side in (e.src, e.dst):
            d, _ = trees[side].query(probes)
            if d.max() > tol:
                raise AssertionError(
                    f"edge {e.src}-{e.dst}: curve strays {d.max():.4f} m from agent "
                    f"{side}'s active cells (tol {tol:.4f} m)"
                )

    # 4. b-g must stay removed (edge-generation-atlas-0.1). Guard both orders.
    pairs = {(e.src, e.dst) for e in edges} | {(e.dst, e.src) for e in edges}
    if ("b", "g") in pairs:
        raise AssertionError("edge b-g is back: it is a two-hop shortcut and was removed by design")

    # 5. the declared set is the seven agents and seven edges of the master plan.
    if sorted(ids) != list("abcdefg"):
        raise AssertionError(f"expected agents a..g, got {sorted(ids)}")
    if len(edges) != 7:
        raise AssertionError(f"expected 7 declared edges, got {len(edges)}")


# --------------------------------------------------------------------------
# the Atlas 0.1 instances
# --------------------------------------------------------------------------

CONFIG = load_config()
ATLAS_01_AGENTS: tuple[AgentSpec, ...] = build_agents(CONFIG)
ATLAS_01_EDGES: tuple[EdgeSpec, ...] = build_edges(CONFIG)
verify(ATLAS_01_AGENTS, ATLAS_01_EDGES, CONFIG)


__all__ = [
    "GridSpec", "AgentSpec", "EdgeSpec", "build_grid", "build_agents", "build_edges",
    "contains", "verify", "ATLAS_01_AGENTS", "ATLAS_01_EDGES", "CONFIG",
]
