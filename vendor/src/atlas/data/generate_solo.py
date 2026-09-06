"""Single-agent (uncoupled) episode generation -- tier T2 of the corpus plan.

The coupled corpus is the expensive artifact and the only source of *interface*
supervision, but it is not the only data the experts need, and for two of them it
is not even the most useful data:

* **The choked-throat gate cannot be trained or graded on it.** That gate asks
  whether predicted mass flow at the throat stays flat as downstream pressure
  varies at fixed chamber pressure. A coupled episode has exactly one
  back-pressure -- whatever the altitude gave it -- so the sweep the gate needs
  does not exist there, and no public dataset has a choked throat either. It has
  to come from here.
* **Agent `c` has no CFL limit.** Its conduction is backward Euler and its
  elasticity a linear solve factorized once per mesh, so a full 10 s shell
  history is ~200 solves: seconds, not hours. The expert with the worst public
  data coverage has the cheapest data generation, and its solo corpus can be
  larger, longer and more diverse than the coupled one.

Everything written here uses the *same* HDF5 schema and the *same* interface
channel definitions as the coupled generator (`gas_face_record` is shared, not
reimplemented), so Phase 2 can mix the two without a second reader and without
two silently different definitions of "interface sample". Solo files are tagged
`kind: solo` in `/meta` and carry only the agents in their group; validation
knows the difference.

    from atlas.data.generate_solo import SOLO_SWEEPS, generate_solo_episode
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import AtlasConfig, load_config
from ..solvers import atmosphere, thermo
from ..solvers.compressible2d import BC, Compressible2D, GasConfig
from ..solvers.grid import build_blocks
from ..solvers.thermostruct2d import ShellMesh, SolidMaterial, ThermoStruct2D
from . import normalize, schema
from .generate import _coarsen_block, _face_state, _remap, gas_face_record
from .sweep import SweepPoint

GROUPS = {
    "nozzle": ("b", "e"),          # chamber + nozzle, coupled to each other only
    "external": ("d",),            # forebody/afterbody external flow
    "shell": ("c",),               # airframe conduction + thermoelasticity
}


@dataclass(frozen=True)
class SoloSpec:
    """One uncoupled run. `group` selects which agents exist at all."""
    point: SweepPoint
    group: str
    T_ep: float                    # simulated horizon, seconds
    dt_snap: float
    idx: int = 0
    split: str = "train"
    coarsen: int = 1
    riemann: str = "hllc"
    # nozzle: back pressure as a fraction of the ideal exit pressure. Values
    # below ~1 are under-expanded, above ~1 over-expanded (and eventually
    # separating), and the whole point of the sweep is to span both.
    p_back_ratio: float = 1.0
    T_wall: float = 800.0
    # shell: which synthesized boundary-condition history to drive it with
    bc_mode: str = "ramp"
    bc_amplitude: float = 1.0

    @property
    def n_snapshots(self) -> int:
        return max(int(round(self.T_ep / self.dt_snap)), 1)

    @property
    def label(self) -> str:
        return f"{self.group}_{self.split}_{self.idx:04d}"


# --------------------------------------------------------------------------
# gas groups
# --------------------------------------------------------------------------


class SoloGasEpisode:
    """Agents `b`+`e` (nozzle) or `d` (external), run without the rest of the graph.

    The absent neighbours are replaced by *prescribed* conditions rather than by
    a frozen copy of a coupled run: that is what makes the sweep independent, and
    it is the only way to vary back pressure at fixed chamber pressure."""

    def __init__(self, spec: SoloSpec, cfg: AtlasConfig | None = None):
        self.spec = spec
        self.cfg = cfg = cfg if cfg is not None else load_config()
        self.p = spec.point
        self.geo = cfg.geometry
        self.agents = GROUPS[spec.group]
        self.scales = normalize.refs_for_episode(cfg, self.p)
        gamma, R = self.p.gamma_gas, self.p.R_gas
        self.gas_cfg = GasConfig(gamma=gamma, R=R, riemann=spec.riemann)
        self.air_cfg = GasConfig(gamma=atmosphere.GAMMA_AIR, R=atmosphere.R_AIR,
                                 riemann=spec.riemann)
        self.blocks = {a: tuple(_coarsen_block(b, spec.coarsen)
                                for b in build_blocks(cfg.agent(a), cfg))
                       for a in self.agents}
        self.sol = {a: [Compressible2D(b, self._cfg_for(a), bcs={})
                        for b in self.blocks[a]] for a in self.agents}
        self.t = 0.0
        self.substeps = 0
        self._init_state()

    def _cfg_for(self, agent: str) -> GasConfig:
        return self.gas_cfg if agent in ("a", "b", "e", "f") else self.air_cfg

    # -- ambient / exit references ----------------------------------------
    def _atm(self):
        T, p_, rho, c = (float(x) for x in atmosphere.properties(self.p.h0))
        return T, p_, rho, c

    @property
    def p_back(self) -> float:
        """Back pressure for the nozzle sweep.

        Referenced to the *ideal* exit pressure rather than to ambient, so one
        ratio means the same flow regime at every chamber pressure and altitude --
        which is what makes the sweep a regime sweep instead of an altitude
        sweep in disguise."""
        return self.spec.p_back_ratio * float(self.scales.p_exit)

    def _init_state(self):
        p = self.p
        gamma, R = p.gamma_gas, p.R_gas
        self.U: dict[str, list[np.ndarray]] = {}
        if self.spec.group == "nozzle":
            rho_c = p.p_c / (R * p.T_c)
            for a in self.agents:
                self.U[a] = []
                for b in self.blocks[a]:
                    nz, ny = b.shape
                    W = np.zeros((nz, ny, 4))
                    W[..., 0] = rho_c
                    W[..., 1] = 50.0
                    W[..., 3] = p.p_c
                    self.U[a].append(thermo.prim_to_cons(W, gamma))
        else:
            T_inf, p_inf, rho_inf, c_inf = self._atm()
            self.v_flight = p.M_inf * c_inf
            for a in self.agents:
                self.U[a] = []
                for b in self.blocks[a]:
                    nz, ny = b.shape
                    W = np.zeros((nz, ny, 4))
                    W[..., 0] = rho_inf
                    W[..., 1] = self.v_flight
                    W[..., 3] = p_inf
                    self.U[a].append(thermo.prim_to_cons(W, atmosphere.GAMMA_AIR))

    # -- wiring -------------------------------------------------------------
    def _wire(self):
        Tw = self.spec.T_wall
        if self.spec.group == "nozzle":
            Ub, Ue = self.U["b"][0], self.U["e"][0]
            yb1 = self.blocks["b"][0].centroid[-1, :, 1]
            ye0 = self.blocks["e"][0].centroid[0, :, 1]
            mdot_flux = self.scales.mdot / (2.0 * self.geo.chamber_halfheight)
            self.sol["b"][0].bcs = {
                # `a` is absent: the reaction zone becomes a mass-flow inlet at
                # the chamber total temperature, which is what it delivers anyway
                "imin": BC("inlet_massflow", {"mdot": mdot_flux, "T": self.p.T_c}),
                "imax": BC("prescribed", {"state": _remap(_face_state(Ue, "imin")[0],
                                                          ye0, yb1)[None]}),
                "jmin": BC("wall_noslip", {"T_wall": Tw}),
                "jmax": BC("wall_noslip", {"T_wall": Tw}),
            }
            self.sol["e"][0].bcs = {
                "imin": BC("prescribed", {"state": _remap(_face_state(Ub, "imax")[0],
                                                          yb1, ye0)[None]}),
                "imax": BC("outflow", {"p_inf": self.p_back}),
                "jmin": BC("wall_noslip", {"T_wall": Tw}),
                "jmax": BC("wall_noslip", {"T_wall": Tw}),
            }
        else:
            T_inf, p_inf, rho_inf, _ = self._atm()
            free = np.array([rho_inf, self.v_flight, 0.0, p_inf])
            for k, _ in enumerate(self.blocks["d"]):
                wall_side = "jmin" if k == 1 else "jmax"
                far_side = "jmax" if k == 1 else "jmin"
                self.sol["d"][k].bcs = {
                    "imin": BC("freestream", {"prim": free}),
                    "imax": BC("outflow", {"p_inf": p_inf}),
                    wall_side: BC("wall_noslip", {"T_wall": Tw}),
                    far_side: BC("freestream", {"prim": free}),
                }

    def step(self, dt: float):
        self._wire()
        for a in self.agents:
            for k, s in enumerate(self.sol[a]):
                self.U[a][k], n = s.advance(self.U[a][k], dt)
                self.substeps += n
        self.t += dt

    # -- records ------------------------------------------------------------
    def _faces(self):
        """(edge key, agent, block, side) for every interface whose SOURCE face
        lives inside this group. The throat plane is `e-b`, and it is the one the
        choked-throat gate reads."""
        if self.spec.group == "nozzle":
            return [("e-b", "e", 0, "imin"), ("e-f", "e", 0, "imax"),
                    ("b-c", "b", 0, "jmax")]
        return [("c-d", "d", 0, "jmax"), ("d-g", "d", 0, "imax")]

    def snapshot(self):
        fields, cond = {}, {}
        _, p_inf, _, _ = self._atm()
        for a in self.agents:
            s = self.sol[a][0]
            gam, R = s.cfg.gamma, s.cfg.R
            W = np.concatenate([thermo.cons_to_prim(U, gam) for U in self.U[a]], axis=1)
            raw = {"rho": W[..., 0], "u": W[..., 1], "v": W[..., 2], "p": W[..., 3],
                   "T": W[..., 3] / (W[..., 0] * R)}
            r = self.scales.per_agent[a]
            nd = r.nondim(raw)
            fields[a] = np.stack([nd[f] for f in self.cfg.agent(a).fields])
            mu = thermo.sutherland(raw["T"].mean(), s.cfg.mu_ref, s.cfg.T_mu_ref,
                                   s.cfg.sutherland_S)
            cond[a] = normalize.gas_conditioning(raw, r, gam, R, float(mu), p_inf)
        iface = {key: gas_face_record(self.sol[a][k], self.U[a][k], side,
                                      self.spec.T_wall)
                 for key, a, k, side in self._faces()}
        return fields, cond, iface

    def run(self) -> dict:
        n = self.spec.n_snapshots
        dt = self.spec.dt_snap
        acc: dict[str, dict] = {"fields": {}, "cond": {}, "iface": {}}
        times = []
        t0 = time.perf_counter()
        for i in range(n):
            f, c, ifc = self.snapshot()
            for name, src in (("fields", f), ("cond", c), ("iface", ifc)):
                for k, v in src.items():
                    acc[name].setdefault(k, []).append(v)
            times.append(self.t)
            if i < n - 1:
                self.step(dt)
        return {
            **{k: {a: np.stack(v) for a, v in d.items()} for k, d in acc.items()},
            "times": np.asarray(times, dtype=float),
            "wall_seconds": time.perf_counter() - t0,
            "substeps": self.substeps,
        }


# --------------------------------------------------------------------------
# the shell, which is nearly free
# --------------------------------------------------------------------------


class SoloShellEpisode:
    """Agent `c` alone, driven by SYNTHESIZED boundary-condition histories.

    Extracting `c`'s boundary conditions from coupled runs would tie its coverage
    to the coupled corpus, which is the one thing that is expensive. Synthesizing
    them decouples the two: ramps, steps and oscillations in (h_in, T_gas_in,
    h_out, T_gas_out) cost nothing and span far more of the boundary-response
    space than any affordable number of coupled episodes.

    **[AI Inference]:** synthesized BCs risk teaching an off-distribution
    response. Two reasons to accept that here: M4 grades on held-out *corner*
    configs, i.e. deliberate extrapolation, where broader coverage is the point;
    and the coupled corpus is still used for fine-tuning, which anchors the
    in-distribution response."""

    def __init__(self, spec: SoloSpec, cfg: AtlasConfig | None = None):
        self.spec = spec
        self.cfg = cfg = cfg if cfg is not None else load_config()
        self.p = spec.point
        self.mat = SolidMaterial()
        self.scales = normalize.refs_for_episode(cfg, self.p)
        blocks = tuple(_coarsen_block(b, spec.coarsen)
                       for b in build_blocks(cfg.agent("c"), cfg))
        self.mesh = [ShellMesh(b.nodes) for b in blocks]
        self.solver = [ThermoStruct2D(m, self.mat) for m in self.mesh]
        self.T = [np.full(m.n_nodes, 288.15) for m in self.mesh]
        self.sigma = [np.zeros((m.shape[0] * m.shape[1], 3)) for m in self.mesh]
        self.u = [np.zeros((m.n_nodes, 2)) for m in self.mesh]
        self.t = 0.0

    def _bc(self, t: float):
        """(h_in, T_gas_in, h_out, T_gas_out, p_in, p_out) at time `t`.

        Amplitudes are anchored to the episode's own chamber and ambient
        conditions, so a sweep over `bc_amplitude` stays inside physically
        reachable loads instead of drifting into numbers no engine produces."""
        s = self.spec
        T_gas_in = s.bc_amplitude * self.p.T_c
        _, p_inf, _, _ = (float(x) for x in atmosphere.properties(self.p.h0))
        p_in = s.bc_amplitude * 0.6 * self.p.p_c
        frac = min(t / max(s.T_ep, 1e-12), 1.0)
        if s.bc_mode == "ramp":
            g = frac
        elif s.bc_mode == "step":
            g = 1.0 if frac > 0.05 else 0.0
        elif s.bc_mode == "oscillation":
            g = 0.5 * (1.0 - np.cos(6.0 * np.pi * frac))
        elif s.bc_mode == "shutdown":
            g = 1.0 if frac < 0.5 else max(0.0, 1.0 - 4.0 * (frac - 0.5))
        else:
            raise ValueError(f"unknown bc_mode {s.bc_mode!r}")
        h_in = 500.0 + 4500.0 * g          # W/m2K, conduction-limited hot-gas side
        h_out = 50.0 + 250.0 * g
        return (h_in, 288.15 + g * (T_gas_in - 288.15), h_out,
                288.15, g * p_in, p_inf)

    def step(self, dt: float):
        h_in, T_in, h_out, T_out, p_in, p_out = self._bc(self.t)
        for k, ts in enumerate(self.solver):
            ni = ts.mesh.shape[0]
            ones = np.ones(ni)
            self.T[k] = ts.step_thermal(self.T[k], dt, ones * h_in, ones * T_in,
                                        ones * h_out, ones * T_out, T_inf=T_out)
            self.u[k], self.sigma[k] = ts.solve_mechanical(
                self.T[k], ones * p_in, ones * p_out)
        self.t += dt

    def snapshot(self):
        Tn = [T.reshape(m.nodes.shape[0], m.nodes.shape[1]) for T, m in
              zip(self.T, self.mesh)]
        Tc = [0.25 * (t[:-1, :-1] + t[1:, :-1] + t[1:, 1:] + t[:-1, 1:]) for t in Tn]
        un = [u.reshape(m.nodes.shape[0], m.nodes.shape[1], 2) for u, m in
              zip(self.u, self.mesh)]
        uc = [0.25 * (u[:-1, :-1] + u[1:, :-1] + u[1:, 1:] + u[:-1, 1:]) for u in un]
        sg = [s.reshape(m.shape[0], m.shape[1], 3) for s, m in zip(self.sigma, self.mesh)]
        r = self.scales.per_agent["c"]
        fields = {"c": np.stack([
            np.concatenate(Tc, axis=1) / r.T,
            np.concatenate([u[..., 0] for u in uc], axis=1) / r.L,
            np.concatenate([u[..., 1] for u in uc], axis=1) / r.L,
            np.concatenate([s[..., 0] for s in sg], axis=1) / r.p,
            np.concatenate([s[..., 1] for s in sg], axis=1) / r.p,
            np.concatenate([s[..., 2] for s in sg], axis=1) / r.p,
        ])}
        h_in, T_in, _, _, _, _ = self._bc(self.t)
        cond = {"c": normalize.solid_conditioning(
            self.T[0], float(h_in), self.mat, self.cfg.geometry.shell_thickness,
            float(np.mean(self.T[0]) - 288.15))}
        # the two gas-facing surfaces, recorded from the SOLID side: no mass or
        # convective energy flux, the interface is carried by heat and traction
        iface = {}
        for key, j in (("b-c", 0), ("c-d", -1)):
            recs = []
            for k, m in enumerate(self.mesh):
                ni, nj = m.shape
                Tcell = Tc[k]
                sig = sg[k]
                dn = np.abs(m.nodes[:-1, 1, 1] - m.nodes[:-1, 0, 1])
                q = -self.mat.k * (Tcell[:, min(j + 1, nj - 1)] - Tcell[:, j]) / \
                    np.maximum(dn, 1e-9)
                seg = np.abs(m.nodes[1:, j, 0] - m.nodes[:-1, j, 0])
                zero = np.zeros(ni)
                recs.append(np.stack([zero, sig[:, j, 0], sig[:, j, 2], zero,
                                      q, -sig[:, j, 1], Tcell[:, j], seg]))
            iface[key] = np.concatenate(recs, axis=1)
        return fields, cond, iface

    def run(self) -> dict:
        n = self.spec.n_snapshots
        dt = self.spec.dt_snap
        acc: dict[str, dict] = {"fields": {}, "cond": {}, "iface": {}}
        times = []
        t0 = time.perf_counter()
        for i in range(n):
            f, c, ifc = self.snapshot()
            for name, src in (("fields", f), ("cond", c), ("iface", ifc)):
                for k, v in src.items():
                    acc[name].setdefault(k, []).append(v)
            times.append(self.t)
            if i < n - 1:
                self.step(dt)
        return {
            **{k: {a: np.stack(v) for a, v in d.items()} for k, d in acc.items()},
            "times": np.asarray(times, dtype=float),
            "wall_seconds": time.perf_counter() - t0,
            "substeps": 0,
        }


# --------------------------------------------------------------------------
# sweeps
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SoloSweep:
    """A named T2 sweep. `note` records what gate or coverage claim it serves,
    because a sweep whose purpose is not written down gets cut first."""
    name: str
    group: str
    T_ep: float
    dt_snap: float
    n_runs: int
    note: str = ""
    axes: dict = field(default_factory=dict)


SOLO_SWEEPS = {
    # ~10 flow-through times of the chamber+nozzle at 200 snapshots. Back
    # pressure is swept at FIXED chamber pressure, which the coupled corpus
    # never does -- there, back pressure is whatever the altitude gave.
    "nozzle": SoloSweep(
        name="nozzle", group="nozzle", T_ep=6.0e-3, dt_snap=3.0e-5, n_runs=48,
        axes={"p_back_ratio": (0.25, 3.0), "T_wall": (600.0, 1100.0)},
        note="the ONLY source for the choked-throat gate (M4) and for gate G6's "
             "over/under-expanded coverage; no public dataset has a choked throat"),
    # ~5 flow-throughs of the 5 m external domain
    "external": SoloSweep(
        name="external", group="external", T_ep=7.0e-2, dt_snap=3.5e-4, n_runs=32,
        axes={"M_inf": (0.1, 3.5), "h0": (0.0, 25000.0), "T_wall": (300.0, 1000.0)},
        note="external_flow coverage in Mach and altitude, at a prescribed wall "
             "temperature so it is independent of the shell"),
    # full flight-scale horizon, because `c` has no CFL limit
    "shell": SoloSweep(
        name="shell", group="shell", T_ep=10.0, dt_snap=5.0e-2, n_runs=64,
        axes={"bc_mode": ("ramp", "step", "oscillation", "shutdown"),
              "bc_amplitude": (0.4, 1.0)},
        note="full 10 s histories at ~free cost; the thermostruct expert has the "
             "worst public-data coverage and the cheapest generation"),
}


def build_solo_specs(sweep: SoloSweep, points: list[SweepPoint],
                     seed: int = 20260818) -> list[SoloSpec]:
    """Cross the sweep's own axes with the episode sweep points.

    Deterministic and stratified rather than random: the choked-throat gate needs
    the back-pressure axis actually spanned, and a Latin hypercube over four axes
    at 48 runs can leave a regime empty by luck."""
    rng = np.random.default_rng(seed)
    specs = []
    for i in range(sweep.n_runs):
        pt = points[i % len(points)]
        kw = {}
        if sweep.group == "nozzle":
            lo, hi = sweep.axes["p_back_ratio"]
            # log-spaced and STRATIFIED: one run per cell of the ratio axis, so
            # both the under-expanded and separating ends are always populated
            frac = (i + 0.5) / sweep.n_runs
            kw["p_back_ratio"] = float(lo * (hi / lo) ** frac)
            kw["T_wall"] = float(rng.uniform(*sweep.axes["T_wall"]))
        elif sweep.group == "external":
            kw["T_wall"] = float(rng.uniform(*sweep.axes["T_wall"]))
        else:
            modes = sweep.axes["bc_mode"]
            kw["bc_mode"] = modes[i % len(modes)]
            lo, hi = sweep.axes["bc_amplitude"]
            kw["bc_amplitude"] = float(lo + (hi - lo) * ((i // len(modes)) %
                                                         max(sweep.n_runs // len(modes), 1))
                                       / max(sweep.n_runs // len(modes) - 1, 1))
        specs.append(SoloSpec(point=pt, group=sweep.group, T_ep=sweep.T_ep,
                              dt_snap=sweep.dt_snap, idx=i,
                              split="train" if i % 8 else "test", **kw))
    return specs


def solo_meta(spec: SoloSpec, res: dict) -> dict:
    from .generate import _git_sha
    return {
        **{k: v for k, v in spec.point.as_dict().items() if not isinstance(v, str)},
        "kind": "solo", "group": spec.group, "label": spec.label,
        "split": spec.split, "idx": spec.idx,
        "T_ep": spec.T_ep, "dt_snap": spec.dt_snap, "dt_macro": spec.dt_snap,
        "n_snapshots": spec.n_snapshots, "coarsen": spec.coarsen,
        "full_fidelity": False, "reduction": "solo",
        "snapshot_policy": "hold_last", "couple_every": "n/a",
        "p_back_ratio": spec.p_back_ratio, "T_wall": spec.T_wall,
        "bc_mode": spec.bc_mode, "bc_amplitude": spec.bc_amplitude,
        "riemann": spec.riemann,
        "schema_version": schema.SCHEMA_VERSION,
        "wall_seconds": res["wall_seconds"], "gas_substeps": res["substeps"],
        "solver_versions": "compressible2d/2 thermostruct2d/1 atmosphere/1976",
        "git_sha": _git_sha(),
        "decisions": "T2 solo tier; D2 rounded throat; D4 channel definitions shared",
    }


def generate_solo_episode(spec: SoloSpec, out_dir: str | Path,
                          cfg: AtlasConfig | None = None) -> Path:
    cfg = cfg if cfg is not None else load_config()
    ep = (SoloShellEpisode(spec, cfg) if spec.group == "shell"
          else SoloGasEpisode(spec, cfg))
    res = ep.run()
    refs = {a: r.as_dict() for a, r in ep.scales.per_agent.items()
            if a in GROUPS[spec.group]}
    iface_meta = {k: {"declared": True, "dt_exch": spec.dt_snap,
                      "dt_exch_model": spec.dt_snap, "coupled": False}
                  for k in res["iface"]}
    n = spec.n_snapshots
    name = f"solo_{spec.group}_{spec.split}_{spec.idx:04d}.h5"
    return schema.write_episode(
        Path(out_dir) / name, solo_meta(spec, res), refs, res["cond"], res["fields"],
        res["iface"], np.zeros((n, 7)), np.zeros((n, 4)),
        iface_meta=iface_meta, times=res["times"])


__all__ = ["SoloSpec", "SoloSweep", "SOLO_SWEEPS", "GROUPS", "SoloGasEpisode",
           "SoloShellEpisode", "build_solo_specs", "generate_solo_episode"]
