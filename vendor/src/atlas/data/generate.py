"""Coupled episode generation -- all solvers advanced together.

A training episode is **one coupled run of all solvers**, not seven solo runs
stitched together, because Atlas has to learn interface behaviour and interface
behaviour only exists in a coupled run. Coupling is loose (Gauss-Seidel) at the
macro step, which is exactly what the surrogate will imitate (spec §3.4):

    for macro_step in range(N):
        advance a, b, e            # walls at c's current surface temperature
        advance d, f, g            # freestream from atmosphere(h); f's inlet from e
        interface fluxes
        thermostruct step
        loads -> trajectory
        snapshot (agents, INTERFACE FLUXES, rigid state, conditioning)

Fidelity knobs (`dt_macro`, `n_macro`, `coarsen`) change what is simulated, never
how honestly: a cheaper episode is a physically correct run of a shorter window
on a coarser grid, and `/meta` records `full_fidelity` so a reduced run can never
be mistaken for a production one. See the implementation log for why the
production setting is not reachable on CPU.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import AtlasConfig, load_config
from ..geometry.contours import nozzle_inner, shell_outer
from ..solvers import atmosphere, thermo, trajectory
from ..solvers.compressible2d import BC, Compressible2D, GasConfig, ReactionConfig
from ..solvers.grid import Block, build_blocks
from ..solvers.thermostruct2d import ShellMesh, SolidMaterial, ThermoStruct2D
from . import normalize, schema
from .sweep import T_INJECT, SweepPoint, q_rxn_for

GAS_AGENTS = ("a", "b", "e", "d", "f", "g")
ENGINE_AGENTS = ("a", "b", "e")
EXTERNAL_AGENTS = ("d", "f", "g")


@dataclass(frozen=True)
class EpisodeTiming:
    """The resolved time plan of one episode (decision D1).

    The corpus-completion plan's route B' rests on separating two things the
    original spec conflated in the phrase "200 snapshots at dt_macro":

    * `dt_macro` is an ARCHITECTURE decision -- it sets the surrogate's step size,
      the multi-rate subcycling ratio, and therefore the speedup claim. It does
      not change.
    * `dt_snap` is a STORAGE decision. Making it smaller costs disk, not solver
      time, because the solver already marches at dt_CFL << dt_snap either way.

    Reading option B as "10 ms episodes" drags dt_macro down with it and reduces
    the surrogate's step to ~3x the solver's, which deletes the reason Atlas
    exists. Shrinking only the HORIZON leaves every ratio intact."""
    dt_macro: float
    dt_snap: float
    n_macro: int
    n_sub: int                     # snapshots per macro step
    coarsen: int

    @property
    def T_ep(self) -> float:
        return self.dt_macro * self.n_macro

    @property
    def n_snapshots(self) -> int:
        return self.n_macro * self.n_sub

    @property
    def full_fidelity(self) -> bool:
        """Full fidelity means the SPEC's episode: 10 s of flight on the declared
        grids. A B'-reduced run is physically correct and honestly labelled, never
        silently promoted."""
        return self.coarsen == 1 and self.T_ep >= 10.0 and self.n_snapshots >= 200


@dataclass
class EpisodeSpec:
    point: SweepPoint
    n_macro: int = 200
    dt_macro: float | None = None      # defaults to cfg.dt_macro
    dt_snap: float | None = None       # defaults to dt_macro: one snapshot per macro step
    coarsen: int = 1                   # divide every grid dimension by this
    inviscid: bool = False
    riemann: str = "hllc"
    snapshot_policy: str = "hold_last"
    couple_every: str = "snapshot"     # snapshot | model_cadence -- see coupling_cadences
    farfield: str = "prescribed"       # D3: prescribed | characteristic -- see below
    reduction: str = ""                # provenance tag, e.g. "B-prime"

    def timing(self, cfg: AtlasConfig) -> EpisodeTiming:
        dt_macro = self.dt_macro if self.dt_macro is not None else cfg.dt_macro
        dt_snap = self.dt_snap if self.dt_snap is not None else dt_macro
        if dt_snap > dt_macro * (1.0 + 1e-9):
            raise ValueError(f"dt_snap {dt_snap} exceeds dt_macro {dt_macro}")
        n_sub = int(round(dt_macro / dt_snap))
        if abs(n_sub * dt_snap - dt_macro) > 1e-9 * dt_macro:
            raise ValueError(
                f"dt_macro {dt_macro} is not an integer multiple of dt_snap {dt_snap}; "
                "a snapshot that straddles a macro boundary has no hold-last value")
        if self.snapshot_policy != "hold_last":
            raise ValueError(
                f"snapshot_policy {self.snapshot_policy!r}: only 'hold_last' is "
                "implemented. Interpolating `c` and the rigid state across a macro "
                "step manufactures states the solver never computed, and the "
                "surrogate would learn the interpolation artifact (D1).")
        return EpisodeTiming(dt_macro=dt_macro, dt_snap=dt_snap, n_macro=self.n_macro,
                             n_sub=n_sub, coarsen=self.coarsen)

    # D3 was proposed as the fix for the Blasius shortfall and the measurement
    # rejected it. delta_99/Blasius on the flat plate, 28x24: prescribed 0.846,
    # characteristic at the top boundary 0.846 (identical), characteristic at
    # inflow and top 0.842, characteristic everywhere 0.650 with the station
    # spread up 5x. The residual edge acceleration (u_e/u_inf = 1.033) is the same
    # in every variant, so it is NOT the far-field treatment -- open action item 8
    # stays open, minus this explanation. The BC is kept and graded by the
    # free-stream-preservation oracle, because it is the right condition for a
    # genuinely non-reflecting far field; adopting it for the corpus needs a
    # far-field-sensitivity study on agent `d` (halve and double
    # farfield_halfwidth, require the solution to be invariant), not a hypothesis.

    @classmethod
    def b_prime(cls, point: SweepPoint, cfg: AtlasConfig, **kw) -> "EpisodeSpec":
        """The adopted production setting (D1): 0.2 s of flight, dt_macro unchanged,
        200 snapshots at 1 ms. 4 macro steps -- enough for the loads->motion->
        freestream loop to close measurably, not enough for max-Q."""
        return cls(point=point, n_macro=4, dt_macro=cfg.dt_macro, dt_snap=1.0e-3,
                   reduction="B-prime", **kw)


IFACE_FACES = {
    # declared edges: (agent, block, face) on the SOURCE side
    ("a", "b"): ("a", 0, "imax"),
    ("e", "b"): ("e", 0, "imin"),
    ("b", "c"): ("b", 0, "jmax"),
    ("c", "d"): ("d", 0, "jmax"),
    ("e", "f"): ("e", 0, "imax"),
    ("d", "g"): ("d", 0, "imax"),
    ("g", "f"): ("f", 0, "jmax"),
    # recorded-but-undeclared (D4): named faces, resolved in _iface_for
    ("d", "f"): "f_inlet_outside_exit",
    ("a", "c"): "a_lateral_walls",
    ("c", "f"): "shell_aft",
}


IFACE_DST_FACES = {
    # The DESTINATION side of each conservation-typed edge, so gate G5 (interface
    # flux consistency) can be computed at all. Coupling here is loose, meaning
    # each side sees the other lagged by one exchange; if that lag exceeds the
    # Phase-3 conservation tolerance, the hard constraint fights its own training
    # data and the symptom is a conservation loss that plateaus for no visible
    # reason. Recording only the source side makes that undiagnosable.
    ("e", "f"): ("f", 0, "imin", "core"),
    ("d", "g"): ("g", 0, "imin", "unblanked"),
}


def edge_cadences(cfg: AtlasConfig) -> dict[str, float]:
    """dt_exch = max(dt_p, dt_q) per interface -- the phase-3 exchange-cadence rule
    (D5), applied to the recorded interfaces as well as the declared ones.

    a-b and e-b land at 1e-3 s, the conservation edges at 5e-3 s, and anything
    touching the shell at 5e-2 s. This is the cadence the SURROGATE's multi-rate
    stepper will run at; see `coupling_cadences` for the one the solver uses."""
    dtm = cfg.dt_model
    out = {}
    for src, dst in IFACE_FACES:
        out[f"{src}-{dst}"] = max(float(dtm[src]), float(dtm[dst]))
    return out


def coupling_cadences(cfg: AtlasConfig, dt_snap: float, dt_macro: float,
                      mode: str = "snapshot") -> dict[str, float]:
    """How often the GENERATOR exchanges across each interface.

    `mode="snapshot"` (default): as often as the solver steps, because the teacher
    has no multi-rate constraint -- with one exception that is physics rather than
    choice. Any edge touching agent `c` cannot refresh faster than `c` advances,
    and `c` advances once per macro step, so those edges hold for dt_macro / dt_snap
    consecutive snapshots. That is a real hold, and it is exactly where D5's stamps
    have to be masked.

    `mode="model_cadence"`: exchange at `edge_cadences`, i.e. make the teacher
    imitate the surrogate's stepper. Measured to be a bad idea: at 5 ms the plume
    inlet misses the entire nozzle start-up (live/held mass flux 7.4x). Kept so the
    comparison can be rerun rather than re-argued."""
    model = edge_cadences(cfg)
    if mode == "model_cadence":
        return model
    if mode != "snapshot":
        raise ValueError(f"couple_every {mode!r}: expected 'snapshot' or 'model_cadence'")
    return {k: (dt_macro if "c" in k.split("-") else dt_snap) for k in model}


def _coarsen_block(b: Block, k: int) -> Block:
    if k <= 1:
        return b
    return Block(b.name, b.nodes[::k, ::k].copy(), b.blanked[::k, ::k].copy())


def _face_state(U: np.ndarray, side: str) -> np.ndarray:
    return {"imin": U[:1], "imax": U[-1:], "jmin": U[:, :1], "jmax": U[:, -1:]}[side]


def _remap(state: np.ndarray, y_src: np.ndarray, y_dst: np.ndarray) -> np.ndarray:
    """Interpolate a face state onto a neighbour's face stations.

    Grids differ across most interfaces (b has 80 transverse cells, e has 96), so
    a copy is not available. Interpolation happens ONLY on the coupling faces --
    never on the stored fields, which stay on the model's own cells."""
    s_src = (y_src - y_src.min()) / max(y_src.ptp(), 1e-30)
    s_dst = np.clip((y_dst - y_dst.min()) / max(y_dst.ptp(), 1e-30), 0.0, 1.0)
    out = np.empty((y_dst.size, state.shape[-1]))
    for c in range(state.shape[-1]):
        out[:, c] = np.interp(s_dst, s_src, state[:, c])
    return out


def _remap_conservative(state: np.ndarray, y_src_edges: np.ndarray,
                        y_dst_edges: np.ndarray) -> np.ndarray:
    """Area-weighted remap of a face state onto a coarser (or finer) face.

    `state` is [N_src, C] on cells bounded by `y_src_edges` [N_src+1]; the result
    is [N_dst, C] on cells bounded by `y_dst_edges` [N_dst+1], with

        out_k = (1 / |dst_k|) * integral over dst_k of the piecewise-constant source

    so sum(out_k * |dst_k|) equals sum(state_j * |src_j|) exactly wherever the two
    spans coincide. `_remap`'s pointwise interpolation does not have that property,
    and at the e-f interface (96 source cells against 8 destination cells) the
    difference is the whole G5 finding."""
    ys = np.asarray(y_src_edges, dtype=float)
    yd = np.asarray(y_dst_edges, dtype=float)
    if ys[0] > ys[-1]:
        ys, state = ys[::-1], state[::-1]
    flip = yd[0] > yd[-1]
    if flip:
        yd = yd[::-1]
    # cumulative integral of the piecewise-constant source, evaluated at any y
    seg = np.cumsum(state * np.diff(ys)[:, None], axis=0)
    cum = np.concatenate([np.zeros((1, state.shape[1])), seg], axis=0)
    lo = np.clip(yd[:-1], ys[0], ys[-1])
    hi = np.clip(yd[1:], ys[0], ys[-1])

    def integral(y):
        j = np.clip(np.searchsorted(ys, y, side="right") - 1, 0, ys.size - 2)
        return cum[j] + state[j] * (y - ys[j])[:, None]

    width = np.maximum(hi - lo, 1e-30)[:, None]
    out = (integral(hi) - integral(lo)) / width
    # destination cells outside the source span keep the nearest source value
    empty = (hi - lo) <= 1e-15
    if empty.any():
        near = np.clip(np.searchsorted(ys, 0.5 * (yd[:-1] + yd[1:])) - 1, 0, state.shape[0] - 1)
        out[empty] = state[near[empty]]
    return out[::-1] if flip else out


def gas_face_record(s, U: np.ndarray, side: str, T_ref: float,
                    mask: np.ndarray | None = None) -> np.ndarray:
    """The eight interface channels on one face of one gas block.

    Module-level so the solo generator writes records that are byte-comparable
    with the coupled corpus: an interface sample must mean the same thing whether
    it came from a coupled episode or a single-agent sweep, or Phase 2 would be
    training one head on two silently different definitions."""
    cfg, blk = s.cfg, s.block
    W = thermo.cons_to_prim(U, cfg.gamma)
    if side in ("imin", "imax"):
        sl = (0, slice(None)) if side == "imin" else (-1, slice(None))
        n = blk.n_i[0 if side == "imin" else -1]
        L = blk.a_i[0 if side == "imin" else -1]
    else:
        sl = (slice(None), 0) if side == "jmin" else (slice(None), -1)
        n = blk.n_j[:, 0 if side == "jmin" else -1]
        L = blk.a_j[:, 0 if side == "jmin" else -1]
    w = W[sl]
    rho, u, v, p = w[..., 0], w[..., 1], w[..., 2], w[..., 3]
    un = u * n[..., 0] + v * n[..., 1]
    T = p / (rho * cfg.R)
    E = p / (cfg.gamma - 1.0) + 0.5 * rho * (u ** 2 + v ** 2)
    mu = thermo.sutherland(T, cfg.mu_ref, cfg.T_mu_ref, cfg.sutherland_S)
    q = mu * cfg.cp / cfg.Pr * (T - T_ref) / np.maximum(
        0.5 * blk.vol[sl] / np.maximum(L, 1e-30), 1e-9)
    rec = np.stack([rho * un, rho * u * un + p * n[..., 0],
                    rho * v * un + p * n[..., 1], (E + p) * un, q, p, T, L])
    return rec if mask is None else rec[:, np.asarray(mask, dtype=bool)]


class CoupledEpisode:
    """One episode: build every solver, wire the declared interfaces, march."""

    def __init__(self, spec: EpisodeSpec, cfg: AtlasConfig | None = None):
        self.spec = spec
        self.cfg = cfg = cfg if cfg is not None else load_config()
        self.p = spec.point
        self.timing = spec.timing(cfg)
        self.dt_macro = self.timing.dt_macro
        self.dt_snap = self.timing.dt_snap
        self.farfield_characteristic = spec.farfield == "characteristic"
        # D4/D5: every geometrically real interface is recorded; the declared
        # seven are a subset, tagged in the file rather than selected here.
        declared = [(e.src, e.dst) for e in cfg.edge_list]
        extra = [(r.src, r.dst) for r in cfg.recorded_interfaces]
        self.iface_keys = tuple(declared + [k for k in extra if k not in declared])
        self.declared_keys = frozenset(f"{a}-{b}" for a, b in declared)
        self.dt_exch_model = edge_cadences(cfg)
        self.dt_exch = coupling_cadences(cfg, self.dt_snap, self.dt_macro,
                                         spec.couple_every)
        self.t_exch: dict[str, float] = {}
        self._exch: dict[str, object] = {}
        self.geo = cfg.geometry
        self.scales = normalize.refs_for_episode(cfg, self.p)
        self.mat = SolidMaterial()

        gamma, R = self.p.gamma_gas, self.p.R_gas
        self.gas_cfg = GasConfig(gamma=gamma, R=R, riemann=spec.riemann,
                                 inviscid=spec.inviscid)
        self.air_cfg = GasConfig(gamma=atmosphere.GAMMA_AIR, R=atmosphere.R_AIR,
                                 riemann=spec.riemann, inviscid=spec.inviscid)
        self.reaction = self._reaction_for_residence(gamma, R)

        self.blocks = {a.id: tuple(_coarsen_block(b, spec.coarsen)
                                   for b in build_blocks(a, cfg)) for a in cfg.agents}
        self._build_solvers()
        self._init_state()

    def _reaction_for_residence(self, gamma: float, R: float) -> ReactionConfig:
        """Pick the pre-exponential from the RESIDENCE TIME of agent `a`.

        A fixed A is a trap: the induction time scales as exp(T_act/T_inject), so
        one hardcoded value either fails to ignite (Y stays near 0 through the
        whole reaction zone) or burns in the first cell. Sizing A so the
        induction time is ~10% of the residence time makes the flame sit inside
        agent `a` across the whole sweep, which is what the a-b interface has to
        carry."""
        geo = self.geo
        rho_inj = self.p.p_c / (R * T_INJECT)
        u_inj = self.scales.mdot / (2.0 * geo.chamber_halfheight) / rho_inj
        t_res = 0.12 / max(u_inj, 1.0)                  # agent `a` spans z in [0, 0.12]
        self.residence = t_res
        T_act = 6000.0
        A = (1.0 / (0.1 * t_res)) * float(np.exp(T_act / T_INJECT))
        return ReactionConfig(A=A, T_act=T_act,
                              q_rxn=q_rxn_for(self.p.T_c, T_INJECT, gamma, R))

    # ------------------------------------------------------------------ setup
    def _build_solvers(self):
        self.sol: dict[str, list[Compressible2D]] = {}
        for a in GAS_AGENTS:
            gc = self.gas_cfg if a in ENGINE_AGENTS or a == "f" else self.air_cfg
            rxn = self.reaction if a == "a" else None
            self.sol[a] = [Compressible2D(b, gc, reaction=rxn, bcs={})
                           for b in self.blocks[a]]
        shell = self.blocks["c"]
        self.shell_mesh = [ShellMesh(b.nodes) for b in shell]
        self.shell = [ThermoStruct2D(m, self.mat) for m in self.shell_mesh]

    def _init_state(self):
        p = self.p
        gamma, R = p.gamma_gas, p.R_gas
        T_inf, p_inf, rho_inf, c_inf = (float(x) for x in atmosphere.properties(p.h0))
        self.v_flight = p.M_inf * c_inf
        rho_c = p.p_c / (R * p.T_c)

        self.U: dict[str, list[np.ndarray]] = {}
        for a in ENGINE_AGENTS:
            nv = 5 if a == "a" else 4
            for b, _ in zip(self.blocks[a], [0]):
                nz, ny = b.shape
                W = np.zeros((nz, ny, nv))
                W[..., 0] = rho_c
                W[..., 1] = 50.0
                W[..., 3] = p.p_c
                if nv == 5:
                    W[..., 4] = 0.0
                self.U[a] = [thermo.prim_to_cons(W, gamma)]
        for a in EXTERNAL_AGENTS:
            gam = gamma if a == "f" else atmosphere.GAMMA_AIR
            self.U[a] = []
            for b in self.blocks[a]:
                nz, ny = b.shape
                W = np.zeros((nz, ny, 4))
                W[..., 0] = rho_inf
                W[..., 1] = self.v_flight
                W[..., 3] = p_inf
                self.U[a].append(thermo.prim_to_cons(W, gam))
        self.T_shell = [np.full(m.n_nodes, 288.15) for m in self.shell_mesh]
        self.sigma = [np.zeros((m.shape[0] * m.shape[1], 3)) for m in self.shell_mesh]
        self.u_shell = [np.zeros((m.n_nodes, 2)) for m in self.shell_mesh]

        self.rigid = trajectory.initial_state(h0=p.h0, v0=self.v_flight)
        self.loads = np.zeros(4)
        self.t = 0.0
        self.substeps = 0
        self.t_macro = 0.0

    # ------------------------------------------------------- wall temperatures
    def _wall_T(self, agent: str) -> float:
        """Mean shell surface temperature seen by a gas agent. Scalar in 0.1: the
        per-station map is available (`T_shell`) but the gas agents' wall BC takes
        a per-face array only on the faces that touch the shell, and `a`'s lateral
        walls are an UNDECLARED a-c interface (see the implementation log)."""
        return float(np.mean([T.mean() for T in self.T_shell]))

    def _atm(self):
        h = float(self.rigid[1])
        T, p_, rho, c = (float(x) for x in atmosphere.properties(h))
        return T, p_, rho, c

    # -------------------------------------------------------------- one macro
    def _exchange(self, key: str, fn, mirror: tuple[str, ...] = ()):
        """Return the payload the DESTINATION of interface `key` is running on.

        Coupling is loose (Gauss-Seidel) and exchanges at the cadence rule
        dt_exch = max(dt_p, dt_q) (phase-3 spec 3.3), so with dt_snap = 1 ms the
        b-c and c-d edges are held for up to 50 snapshots while a-b and e-b
        refresh every one. Recomputing everything every sub-step would be a
        corpus the multi-rate stepper cannot reproduce at inference; holding is
        what the stepper actually does.

        `t_exch` records when each payload was last really computed, which is the
        only thing that lets the Phase-2 interface loss tell a held constant from
        a live signal (decision D5). `mirror` stamps the same exchange onto other
        interfaces carrying the same payload -- the mean shell temperature reaches
        agent `a` through the undeclared a-c interface and agent `b` through b-c,
        and they are one exchange, not two."""
        dt_e = self.dt_exch[key]
        last = self.t_exch.get(key)
        if last is None or self.t + 1e-12 >= last + dt_e:
            self._exch[key] = fn()
            self.t_exch[key] = self.t
            for m in mirror:
                self.t_exch[m] = self.t
        return self._exch[key]

    def _wire_engine(self):
        geo = self.geo
        Ua, Ub, Ue = self.U["a"][0], self.U["b"][0], self.U["e"][0]
        Tw = self._exchange("b-c", lambda: self._wall_T("engine"), mirror=("a-c",))
        ya = self.blocks["a"][0].centroid[-1, :, 1]
        yb0 = self.blocks["b"][0].centroid[0, :, 1]
        yb1 = self.blocks["b"][0].centroid[-1, :, 1]
        ye0 = self.blocks["e"][0].centroid[0, :, 1]

        mdot_flux = self.scales.mdot / (2.0 * geo.chamber_halfheight)

        def _ab():
            # b carries no progress variable; hand `a` its own edge Y back, so the
            # downstream boundary neither injects unburnt gas nor quenches the zone.
            b_to_a = _remap(_face_state(Ub, "imin")[0], yb0, ya)
            b_to_a = np.concatenate([b_to_a, _face_state(Ua, "imax")[0, :, 4:]], axis=-1)
            a_to_b = _remap(_face_state(Ua, "imax")[0, :, :4], ya, yb0)
            return b_to_a, a_to_b

        def _eb():
            return (_remap(_face_state(Ue, "imin")[0], ye0, yb1),
                    _remap(_face_state(Ub, "imax")[0], yb1, ye0))

        b_to_a, a_to_b = self._exchange("a-b", _ab)
        e_to_b, b_to_e = self._exchange("e-b", _eb)

        self.sol["a"][0].bcs = {
            "imin": BC("inlet_massflow", {"mdot": mdot_flux, "T": T_INJECT}),
            "imax": BC("prescribed", {"state": b_to_a[None]}),
            "jmin": BC("wall_noslip", {"T_wall": Tw}),
            "jmax": BC("wall_noslip", {"T_wall": Tw}),
        }
        self.sol["b"][0].bcs = {
            "imin": BC("prescribed", {"state": a_to_b[None]}),
            "imax": BC("prescribed", {"state": e_to_b[None]}),
            "jmin": BC("wall_noslip", {"T_wall": Tw}),
            "jmax": BC("wall_noslip", {"T_wall": Tw}),
        }
        _, p_inf, _, _ = self._atm()
        self.sol["e"][0].bcs = {
            "imin": BC("prescribed", {"state": b_to_e[None]}),
            "imax": BC("outflow", {"p_inf": p_inf}),
            "jmin": BC("wall_noslip", {"T_wall": Tw}),
            "jmax": BC("wall_noslip", {"T_wall": Tw}),
        }

    def _wire_external(self):
        T_inf, p_inf, rho_inf, c_inf = self._atm()
        free = np.array([rho_inf, self.v_flight, 0.0, p_inf])
        Tw = self._exchange("c-d", lambda: self._wall_T("external"))
        Ue = self.U["e"][0]
        far = BC("farfield", {"prim": free}) if self.farfield_characteristic else \
            BC("freestream", {"prim": free})

        for k, b in enumerate(self.blocks["d"]):
            wall_side = "jmin" if k == 1 else "jmax"      # panel 1 is the upper sheet
            far_side = "jmax" if k == 1 else "jmin"
            self.sol["d"][k].bcs = {
                "imin": far,
                "imax": BC("outflow", {"p_inf": p_inf}),
                wall_side: BC("wall_noslip", {"T_wall": Tw}),
                far_side: far,
            }

        # f's inlet: the nozzle exit over |y| <= exit half-height, ambient
        # elsewhere. That "elsewhere" IS the undeclared d-f interface -- the
        # plume's upstream face is wider than the nozzle exit -- and D4 records it.
        yf = self.blocks["f"][0].centroid[0, :, 1]
        core = np.abs(yf) <= self.geo.exit_halfheight

        ye_edges = self.blocks["e"][0].nodes[-1, :, 1]
        yf_edges = self.blocks["f"][0].nodes[0, :, 1]

        def _ef():
            # CONSERVATIVE remap: e-f is a conservation-typed edge and Phase 3
            # imposes hard mass matching on it, so the payload must carry the same
            # integral the nozzle expels, not a point sample of its profile.
            inlet = np.broadcast_to(thermo.prim_to_cons(free, atmosphere.GAMMA_AIR),
                                    (yf.size, 4)).copy()
            if core.any():
                k = np.flatnonzero(core)
                edges = np.concatenate([yf_edges[k], yf_edges[k[-1] + 1:k[-1] + 2]])
                inlet[core] = _remap_conservative(_face_state(Ue, "imax")[0],
                                                  ye_edges, edges)
            return inlet

        def _gf():
            # the g-f shear layer, both directions: f sees g's cells just outboard
            # of the plume, g sees f's state inside its blanked band.
            Ug = self.U["g"][0]
            j0, j1 = self.sol["g"][0]._hole
            g_lo = Ug[:, j0 - 1] if j0 > 0 else Ug[:, 0]
            g_hi = Ug[:, j1 + 1] if j1 + 1 < Ug.shape[1] else Ug[:, -1]
            yg = self.blocks["g"][0].centroid[:, 0, 0]      # z stations of g
            zf = self.blocks["f"][0].centroid[:, 0, 0]
            return (_remap(g_lo, yg, zf), _remap(g_hi, yg, zf),
                    self.U["f"][0].mean(axis=1))

        inlet = self._exchange("e-f", _ef)
        g_lo_f, g_hi_f, hole = self._exchange("g-f", _gf)

        self.sol["f"][0].bcs = {
            "imin": BC("prescribed", {"state": inlet[None]}),
            "imax": BC("outflow", {"p_inf": p_inf}),
            "jmin": BC("prescribed", {"state": g_lo_f[:, None]}),
            "jmax": BC("prescribed", {"state": g_hi_f[:, None]}),
        }
        self.sol["g"][0].bcs = {
            "imin": far,
            "imax": BC("outflow", {"p_inf": p_inf}),
            "jmin": far,
            "jmax": far,
        }
        # the plume hole is the declared g-f interface, not a wall
        self.sol["g"][0].hole_state = hole

    def _advance_gas(self, agents, dt):
        for a in agents:
            for k, s in enumerate(self.sol[a]):
                self.U[a][k], n = s.advance(self.U[a][k], dt)
                self.substeps += n

    def _wall_flux(self, agent: str, k: int, side: str):
        """(heat flux, pressure) along a wall face of a gas block."""
        s = self.sol[agent][k]
        blk = s.block
        U = self.U[agent][k]
        cfg = s.cfg
        W = thermo.cons_to_prim(U, cfg.gamma)
        T = W[..., 3] / (W[..., 0] * cfg.R)
        if side == "jmax":
            dn = 0.5 * blk.vol[:, -1] / np.maximum(blk.a_j[:, -1], 1e-30)
            T_in, p_in = T[:, -1], W[:, -1, 3]
        else:
            dn = 0.5 * blk.vol[:, 0] / np.maximum(blk.a_j[:, 0], 1e-30)
            T_in, p_in = T[:, 0], W[:, 0, 3]
        mu = thermo.sutherland(T_in, cfg.mu_ref, cfg.T_mu_ref, cfg.sutherland_S)
        k_gas = mu * cfg.cp / cfg.Pr
        h = k_gas / np.maximum(dn, 1e-9)                # conduction-limited h
        return h, T_in, p_in

    def _step_structure(self, dt):
        _, p_inf, _, _ = self._atm()
        for k, ts in enumerate(self.shell):
            ni = ts.mesh.shape[0]
            h_in, T_in, p_in = self._wall_flux("b", 0, "jmax" if k == 1 else "jmin")
            h_out, T_out, p_out = self._wall_flux("d", k, "jmin" if k == 1 else "jmax")
            f = lambda v: np.interp(np.linspace(0, 1, ni), np.linspace(0, 1, v.size), v)  # noqa: E731
            self.T_shell[k] = ts.step_thermal(self.T_shell[k], dt, f(h_in), f(T_in),
                                              f(h_out), f(T_out), T_inf=self._atm()[0])
            self.u_shell[k], self.sigma[k] = ts.solve_mechanical(
                self.T_shell[k], f(p_in), f(p_out))

    # ---------------------------------------------------------- interface data
    def _iface_records(self) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """One [C_edge, N] record per interface, declared or merely recorded.

        Decision D4: record every geometrically real interface and declare a
        subset. The model reads the declared seven; the corpus supports either
        answer, and Phase 4's A4 declared-vs-dense ablation becomes runnable."""
        out, stamps = {}, {}
        for src, dst in self.iface_keys:
            key = f"{src}-{dst}"
            out[key] = self._iface_for(src, dst)
            stamps[key] = float(self.t_exch.get(key, self.t))
        return out, stamps

    def _iface_for(self, src: str, dst: str) -> np.ndarray:
        """Fluxes and traces on the shared interface. Eight channels, see
        schema.IFACE_CHANNELS."""
        face = IFACE_FACES[(src, dst)]
        if face == "a_lateral_walls":
            return np.concatenate([self._gas_face_record("a", 0, "jmin"),
                                   self._gas_face_record("a", 0, "jmax")], axis=1)
        if face == "f_inlet_outside_exit":
            y = self.blocks["f"][0].centroid[0, :, 1]
            return self._gas_face_record("f", 0, "imin",
                                         mask=np.abs(y) > self.geo.exit_halfheight)
        if face == "shell_aft":
            return self._iface_solid_aft()
        agent, k, side = face
        return self._gas_face_record(agent, k, side)

    def _iface_dst_records(self) -> dict[str, np.ndarray]:
        """The destination side of every conservation-typed edge, for gate G5."""
        out = {}
        for (src, dst), (agent, k, side, sel) in IFACE_DST_FACES.items():
            if sel == "core":
                y = self.blocks[agent][k].centroid[0, :, 1]
                mask = np.abs(y) <= self.geo.exit_halfheight
            elif sel == "unblanked":
                mask = ~self.blocks[agent][k].blanked[0]
            else:
                mask = None
            out[f"{src}-{dst}"] = self._gas_face_record(agent, k, side, mask=mask)
        return out

    def _exchange_flux(self) -> dict[str, float]:
        """Mass flux of the payload the DESTINATION is currently running on.

        Against the live source-side integral this isolates the pure lag error --
        what the destination believes minus what the source is doing now -- with
        no discretization or grid-mismatch term in it. That is the quantity that
        has to sit below Phase 3's conservation tolerance, because Phase 3 imposes
        hard flux matching on data generated with this lag in it."""
        out = {}
        payload = self._exch.get("e-f")
        if payload is not None:
            blk = self.blocks["f"][0]
            y = blk.centroid[0, :, 1]
            core = np.abs(y) <= self.geo.exit_halfheight
            n = blk.n_i[0]
            L = blk.a_i[0]
            W = thermo.cons_to_prim(np.asarray(payload), atmosphere.GAMMA_AIR)
            un = W[..., 1] * n[..., 0] + W[..., 2] * n[..., 1]
            out["e-f"] = float(np.sum((W[..., 0] * un * L)[core]))
        return out

    def _gas_face_record(self, agent: str, k: int, side: str,
                         mask: np.ndarray | None = None) -> np.ndarray:
        """Fluxes and traces taken from the SOURCE side's face cells."""
        return gas_face_record(self.sol[agent][k], self.U[agent][k], side,
                               self._wall_T("x"), mask)


    def _iface_solid_aft(self) -> np.ndarray:
        """The shell's aft face against the plume, recorded from the SOLID side.

        A solid face carries no mass and no convective energy flux, so those two
        channels are structurally zero rather than unmeasured -- the interface is
        carried by traction and conduction, which land in the momentum-flux and
        heat-flux channels. Written this way so the record keeps the same eight
        channels as every other interface and needs no special-case reader."""
        recs = []
        for k, mesh in enumerate(self.shell_mesh):
            ni, nj = mesh.shape
            Tn = self.T_shell[k].reshape(ni + 1, nj + 1)
            Tc = 0.25 * (Tn[:-1, :-1] + Tn[1:, :-1] + Tn[1:, 1:] + Tn[:-1, 1:])
            sig = self.sigma[k].reshape(ni, nj, 3)
            z_c = 0.5 * (mesh.nodes[:-1, :-1, 0] + mesh.nodes[1:, :-1, 0])
            dz = np.maximum(z_c[-1] - z_c[-2], 1e-9)
            q = -self.mat.k * (Tc[-1] - Tc[-2]) / dz                  # W/m2, +z out
            seg = np.abs(mesh.nodes[-1, 1:, 1] - mesh.nodes[-1, :-1, 1])
            zero = np.zeros(nj)
            recs.append(np.stack([zero, sig[-1, :, 0], sig[-1, :, 2], zero,
                                  q, -sig[-1, :, 0], Tc[-1], seg]))
        return np.concatenate(recs, axis=1)

    def _compute_loads(self):
        """Thrust from the e-f interface integral; aero from the c-d traction."""
        geo = self.geo
        _, p_inf, _, _ = self._atm()
        Ue = self.U["e"][0]
        s = self.sol["e"][0]
        W = thermo.cons_to_prim(Ue, s.cfg.gamma)
        L = s.block.a_i[-1]
        rho, u, p = W[-1, :, 0], W[-1, :, 1], W[-1, :, 3]
        thrust = float(np.sum((rho * u * u + (p - p_inf)) * L))

        axial = 0.0
        normal = 0.0
        for k, b in enumerate(self.blocks["d"]):
            Ud = self.U["d"][k]
            j = 0 if k == 1 else -1
            Wd = thermo.cons_to_prim(Ud, self.air_cfg.gamma)
            pw = Wd[:, j, 3]
            nrm = b.n_j[:, j if j == 0 else -1]
            Lw = b.a_j[:, j if j == 0 else -1]
            axial += float(np.sum(-(pw - p_inf) * nrm[:, 0] * Lw))
            normal += float(np.sum(-(pw - p_inf) * nrm[:, 1] * Lw))

        th = float(self.rigid[2])
        F_thrust = (-thrust * np.cos(th), -thrust * np.sin(th))   # body -z is 'up'
        F_aero = (axial * np.cos(th) - normal * np.sin(th),
                  axial * np.sin(th) + normal * np.cos(th))
        self.loads = np.array([F_thrust[0], F_thrust[1], F_aero[0], F_aero[1]])
        return trajectory.Loads(F_thrust=F_thrust, F_aero=F_aero, torque=0.0,
                                mdot=self.scales.mdot)

    # ------------------------------------------------------------- the episode
    def step_gas(self, dt: float) -> None:
        """One gas sub-step. The structure and the rigid body do NOT advance here:
        they step once per macro step, and route B-prime snapshots inside that
        window."""
        self._wire_engine()
        self._advance_gas(ENGINE_AGENTS, dt)
        self._wire_external()
        self._advance_gas(EXTERNAL_AGENTS, dt)
        self.t += dt

    def step_macro_tail(self, dt: float) -> None:
        """The slow half of a macro step: shell conduction and stress, the loads
        integral, and the rigid-body integration -- all over the full dt_macro,
        after the gas has already covered the window."""
        self._step_structure(dt)
        loads = self._compute_loads()
        self.rigid = trajectory.rk4_step(self.rigid, dt, loads)
        self.t_macro = self.t

    def step_macro(self) -> None:
        """A whole macro step, for callers that do not snapshot inside it."""
        tm = self.timing
        for _ in range(tm.n_sub):
            self.step_gas(tm.dt_snap)
        self.step_macro_tail(tm.dt_macro)

    def snapshot(self):
        """Nondimensionalized fields + conditioning, on the model's own cells.

        Under the hold-last policy (D1) the shell and rigid entries of a sub-macro
        snapshot repeat their last macro value. That is not an approximation: it
        is exactly what the multi-rate stepper does at inference, where `c`
        genuinely holds between its 5e-2 s steps. Interpolating instead would
        manufacture states the solver never computed and teach the surrogate the
        interpolation artifact."""
        fields, cond = {}, {}
        _, p_inf, _, _ = self._atm()
        for a in GAS_AGENTS:
            s = self.sol[a][0]
            gam, R = s.cfg.gamma, s.cfg.R
            Ws = [thermo.cons_to_prim(U, gam) for U in self.U[a]]
            W = np.concatenate(Ws, axis=1)
            raw = {"rho": W[..., 0], "u": W[..., 1], "v": W[..., 2], "p": W[..., 3],
                   "T": W[..., 3] / (W[..., 0] * R)}
            if W.shape[-1] > 4:
                raw["Y"] = W[..., 4]
            r = self.scales.per_agent[a]
            nd = r.nondim(raw)
            order = self.cfg.agent(a).fields
            fields[a] = np.stack([nd[f] for f in order])
            mu = thermo.sutherland(raw["T"].mean(), s.cfg.mu_ref, s.cfg.T_mu_ref,
                                   s.cfg.sutherland_S)
            cond[a] = normalize.gas_conditioning(raw, r, gam, R, float(mu), p_inf)

        Tn = [T.reshape(m.nodes.shape[0], m.nodes.shape[1]) for T, m in
              zip(self.T_shell, self.shell_mesh)]
        Tc = [0.25 * (t[:-1, :-1] + t[1:, :-1] + t[1:, 1:] + t[:-1, 1:]) for t in Tn]
        un = [u.reshape(m.nodes.shape[0], m.nodes.shape[1], 2) for u, m in
              zip(self.u_shell, self.shell_mesh)]
        uc = [0.25 * (u[:-1, :-1] + u[1:, :-1] + u[1:, 1:] + u[:-1, 1:]) for u in un]
        sg = [s.reshape(m.shape[0], m.shape[1], 3) for s, m in
              zip(self.sigma, self.shell_mesh)]
        rc = self.scales.per_agent["c"]
        fields["c"] = np.stack([
            np.concatenate(Tc, axis=1) / rc.T,
            np.concatenate([u[..., 0] for u in uc], axis=1) / rc.L,
            np.concatenate([u[..., 1] for u in uc], axis=1) / rc.L,
            np.concatenate([s[..., 0] for s in sg], axis=1) / rc.p,
            np.concatenate([s[..., 1] for s in sg], axis=1) / rc.p,
            np.concatenate([s[..., 2] for s in sg], axis=1) / rc.p,
        ])
        h_in, _, _ = self._wall_flux("b", 0, "jmin")
        cond["c"] = normalize.solid_conditioning(
            self.T_shell[0], float(np.mean(h_in)), self.mat, self.geo.shell_thickness,
            float(np.mean(self.T_shell[0]) - 288.15))
        iface, stamps = self._iface_records()
        return (fields, cond, iface, stamps, self._iface_dst_records(),
                self._exchange_flux())

    def run(self) -> dict:
        tm = self.timing
        # Prime the exchanges before the first snapshot. Without this the t = 0
        # snapshot has no payload to report and /iface_exch is one entry short of
        # /times -- an off-by-one that would only surface when gate G5 is computed.
        self._wire_engine()
        self._wire_external()
        fields: dict[str, list] = {}
        cond: dict[str, list] = {}
        iface: dict[str, list] = {}
        iface_dst: dict[str, list] = {}
        iface_exch: dict[str, list] = {}
        t_exch: dict[str, list] = {}
        times, rigid, loads = [], [], []
        t0 = time.perf_counter()
        n = tm.n_snapshots
        for i in range(n):
            f, c, ifc, st, ifd, ifx = self.snapshot()
            for store, src in ((fields, f), (cond, c), (iface, ifc), (t_exch, st),
                               (iface_dst, ifd), (iface_exch, ifx)):
                for k, val in src.items():
                    store.setdefault(k, []).append(val)
            times.append(self.t)
            rigid.append(self.rigid.copy())
            loads.append(self.loads.copy())
            if i == n - 1:
                break
            self.step_gas(tm.dt_snap)
            if (i + 1) % tm.n_sub == 0:
                self.step_macro_tail(tm.dt_macro)
        return {
            "fields": {k: np.stack(v) for k, v in fields.items()},
            "cond": {k: np.stack(v) for k, v in cond.items()},
            "iface": {k: np.stack(v) for k, v in iface.items()},
            "iface_dst": {k: np.stack(v) for k, v in iface_dst.items()},
            "iface_exch": {k: np.asarray(v, dtype=float) for k, v in iface_exch.items()},
            "iface_t_exch": {k: np.asarray(v, dtype=float) for k, v in t_exch.items()},
            "times": np.asarray(times, dtype=float),
            "rigid": np.stack(rigid), "loads": np.stack(loads),
            "wall_seconds": time.perf_counter() - t0, "substeps": self.substeps,
        }


def _git_sha() -> str:
    """The commit the corpus was generated at. A corpus whose provenance stops at
    "solver_versions" cannot be traced back to the code that made it, and the
    corpus outlives several weeks of that code (T3.4)."""
    import subprocess
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10, cwd=str(Path(__file__).resolve().parents[3]))
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:                                  # noqa: BLE001 - provenance only
        return "unknown"


def episode_meta(spec: EpisodeSpec, ep: "CoupledEpisode", res: dict) -> dict:
    """Everything needed to reproduce the episode and to refuse to mistake it for
    a different one. The D1-D5 decisions are recorded IN THE FILE, not only in the
    plan, because the plan is not shipped with the corpus."""
    tm = ep.timing
    return {
        **{k: v for k, v in spec.point.as_dict().items() if not isinstance(v, str)},
        "label": spec.point.label, "split": spec.point.split,
        "dt_macro": tm.dt_macro, "dt_snap": tm.dt_snap, "n_macro": tm.n_macro,
        "n_sub": tm.n_sub, "n_snapshots": tm.n_snapshots, "T_ep": tm.T_ep,
        "coarsen": spec.coarsen,
        "full_fidelity": bool(tm.full_fidelity),
        "reduction": spec.reduction,
        "snapshot_policy": spec.snapshot_policy,
        "farfield": spec.farfield,
        "couple_every": spec.couple_every,
        "throat_round_radius": float(ep.geo.throat_round_radius),
        "riemann": spec.riemann, "inviscid": bool(spec.inviscid),
        "schema_version": schema.SCHEMA_VERSION,
        "wall_seconds": res["wall_seconds"], "gas_substeps": res["substeps"],
        "solver_versions": "compressible2d/2 thermostruct2d/1 atmosphere/1976 trajectory/rk4",
        "git_sha": _git_sha(),
        "config_fingerprint": ep.cfg.fingerprint(),
        "decisions": "D1:B-prime+hold_last D2:rounded_throat D3:characteristic_farfield "
                     "D4:record_all_declare_seven D5:t_exch_stamps",
    }


def generate_episode(spec: EpisodeSpec, out_dir: str | Path,
                     cfg: AtlasConfig | None = None) -> Path:
    cfg = cfg if cfg is not None else load_config()
    ep = CoupledEpisode(spec, cfg)
    res = ep.run()
    refs = {a: r.as_dict() for a, r in ep.scales.per_agent.items()}
    iface_meta = {k: {"declared": k in ep.declared_keys,
                      "dt_exch": ep.dt_exch[k],
                      "dt_exch_model": ep.dt_exch_model[k],
                      "coupled": k in ep.t_exch}
                  for k in res["iface"]}
    name = f"episode_{spec.point.split}_{spec.point.idx:04d}.h5"
    return schema.write_episode(
        Path(out_dir) / name, episode_meta(spec, ep, res), refs, res["cond"],
        res["fields"], res["iface"], res["rigid"], res["loads"],
        iface_t_exch=res["iface_t_exch"], iface_dst=res["iface_dst"],
        iface_exch=res["iface_exch"], iface_meta=iface_meta, times=res["times"])


__all__ = ["EpisodeSpec", "EpisodeTiming", "CoupledEpisode", "generate_episode",
           "episode_meta", "edge_cadences", "coupling_cadences", "gas_face_record",
           "IFACE_FACES", "GAS_AGENTS"]
