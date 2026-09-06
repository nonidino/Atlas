"""Atlas 0.1 config loading.

`atlas_0_1.yaml` is the single source of truth (phase-1 spec 3.7). This module
parses it into frozen dataclasses and nothing else -- no defaults live here that
are not in the YAML, so "grep the YAML" is a complete answer to "what number is
the model using".

Everything is frozen and hashable, which is what lets `geometry.layout` and
`model.edges` memoize on `cfg.fingerprint()` instead of recomputing static
geometry every forward pass (phase-1 spec 3.3, and the pitfall list).

    from atlas.config import load_config
    cfg = load_config()                 # the packaged atlas_0_1.yaml
    cfg = load_config("my_variant.yaml")
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).with_name("atlas_0_1.yaml")


@dataclass(frozen=True)
class ModelCfg:
    d_model: int
    n_layers: int
    n_mp_layers: int
    n_heads: int
    n_fourier: int
    d_edge: int
    d_ffn_mult: int
    n_rigid: int
    use_bottleneck: bool
    decoder_out_scale: float

    @property
    def d_head(self) -> int:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model {self.d_model} not divisible by n_heads {self.n_heads}")
        return self.d_model // self.n_heads


@dataclass(frozen=True)
class GeometryCfg:
    """Body-fixed planar frame, metres. z downstream, y transverse about y=0."""
    chamber_halfheight: float
    throat_halfheight: float
    exit_halfheight: float
    z_injector: float
    z_converge: tuple[float, float]
    z_diverge: tuple[float, float]
    z_throat: float
    z_exit: float
    shell_thickness: float
    shell_z: tuple[float, float]
    farfield_halfwidth: float
    atmos_front_z: tuple[float, float]
    atmos_stretch_ratio: float
    plume_halfwidth: float
    plume_z: tuple[float, float]
    throat_round_radius: float = 0.0


@dataclass(frozen=True)
class AgentCfg:
    id: str
    name: str
    kind: str
    tokenizer: str
    expert: str
    grid: tuple[int, int]          # (n_z, n_y) CELLS
    patch: str                     # key into cfg.patch
    z: tuple[float, float]
    y_map: str                     # uniform | shell | wall_stretched
    fields: tuple[str, ...]
    cond: str                      # key into cfg.cond
    y_halfwidth: float | None = None
    wall: str | None = None        # name of a wall contour masking inactive cells
    hole: str | None = None        # name of a region carved out (another agent's domain)

    @property
    def n_fields(self) -> int:
        return len(self.fields)


@dataclass(frozen=True)
class EdgeCfg:
    src: str
    dst: str
    types: tuple[str, ...]
    interface: str                 # constructor name in geometry/contours.py


@dataclass(frozen=True)
class RecordedIfaceCfg:
    """A geometrically real interface that is RECORDED in the corpus but not
    declared to the model (decision D4).

    The asymmetry is the whole point: declaring an edge later, having never
    recorded it, means regenerating the corpus (~1000 core-hours); recording one
    that is never declared costs a few percent of storage. It also makes Phase 4's
    A4 declared-vs-dense ablation runnable, which without these records it is not."""
    src: str
    dst: str
    face: str              # key into generate.IFACE_FACES
    note: str = ""


@dataclass(frozen=True)
class RegionCfg:
    name: str
    z: tuple[float, float]
    y_halfwidth: float


@dataclass(frozen=True)
class AtlasConfig:
    model: ModelCfg
    geometry: GeometryCfg
    patch: dict            # 'gas'/'solid' -> (pz, py); plus tol_patch_diag
    K_match: int
    dt_model: dict
    dt_macro: float
    train: dict
    cond: dict             # 'gas'/'solid' -> tuple of scalar names
    agents: tuple[AgentCfg, ...]
    edge_list: tuple[EdgeCfg, ...]
    edge_types: tuple[str, ...]
    unmodelled: tuple[RegionCfg, ...]
    recorded_interfaces: tuple[RecordedIfaceCfg, ...] = ()
    _raw: str = ""         # verbatim YAML text, for the fingerprint

    # -- lookups ------------------------------------------------------------
    def agent(self, name: str) -> AgentCfg:
        for a in self.agents:
            if a.id == name:
                return a
        raise KeyError(f"no agent {name!r}; have {[a.id for a in self.agents]}")

    @property
    def agent_ids(self) -> tuple[str, ...]:
        return tuple(a.id for a in self.agents)

    def patch_size(self, agent: AgentCfg | str) -> tuple[int, int]:
        a = self.agent(agent) if isinstance(agent, str) else agent
        return tuple(self.patch[a.patch])

    @property
    def tol_patch_diag(self) -> float:
        return float(self.patch["tol_patch_diag"])

    def cond_names(self, agent: AgentCfg | str) -> tuple[str, ...]:
        a = self.agent(agent) if isinstance(agent, str) else agent
        return tuple(self.cond[a.cond])

    def type_index(self, t: str) -> int:
        return self.edge_types.index(t)

    # -- identity -----------------------------------------------------------
    def fingerprint(self) -> str:
        """Stable content hash. Memoization keys derive from this, never from
        object identity -- two `load_config()` calls must hit the same cache.

        Hashes the parsed CONTENT, not the YAML text: a programmatic override
        (`dataclasses.replace(cfg, ...)`, which tests use) has to miss the layout
        and graph caches, and a text-only change (a comment) has to hit them."""
        d = {k: v for k, v in asdict(self).items() if k != "_raw"}
        return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _tup(x):
    return tuple(x) if isinstance(x, (list, tuple)) else x


def load_config(path: str | Path | None = None) -> AtlasConfig:
    p = Path(path) if path is not None else DEFAULT_CONFIG
    raw = p.read_text(encoding="utf-8")
    d = yaml.safe_load(raw)

    geo = {k: _tup(v) for k, v in d["geometry"].items()}

    agents = tuple(
        AgentCfg(
            id=aid,
            name=a["name"],
            kind=a["kind"],
            tokenizer=a["tokenizer"],
            expert=a["expert"],
            grid=tuple(a["grid"]),
            patch=a["patch"],
            z=tuple(a["z"]),
            y_map=a["y_map"],
            fields=tuple(a["fields"]),
            cond=a["cond"],
            y_halfwidth=a.get("y_halfwidth"),
            wall=a.get("wall"),
            hole=a.get("hole"),
        )
        for aid, a in d["agents"].items()
    )

    edge_list = tuple(
        EdgeCfg(src=e["src"], dst=e["dst"], types=tuple(e["types"]), interface=e["interface"])
        for e in d["edge_list"]
    )

    return AtlasConfig(
        model=ModelCfg(**d["model"]),
        geometry=GeometryCfg(**geo),
        patch={k: (tuple(v) if isinstance(v, list) else v) for k, v in d["patch"].items()},
        K_match=int(d["edges"]["K_match"]),
        dt_model=dict(d["dt_model"]),
        dt_macro=float(d["dt_macro"]),
        train=dict(d["train"]),
        cond={k: tuple(v) for k, v in d["cond"].items()},
        agents=agents,
        edge_list=edge_list,
        edge_types=tuple(d["edge_types"]),
        unmodelled=tuple(
            RegionCfg(name=r["name"], z=tuple(r["z"]), y_halfwidth=float(r["y_halfwidth"]))
            for r in d.get("unmodelled", [])
        ),
        recorded_interfaces=tuple(
            RecordedIfaceCfg(src=r["src"], dst=r["dst"], face=r["face"],
                             note=r.get("note", ""))
            for r in d.get("recorded_interfaces", [])
        ),
        _raw=raw,
    )


__all__ = [
    "AtlasConfig", "ModelCfg", "GeometryCfg", "AgentCfg", "EdgeCfg", "RegionCfg",
    "RecordedIfaceCfg",
    "load_config", "DEFAULT_CONFIG",
]
