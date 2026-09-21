"""The rocket's real experts: R0 rungs 1 and 2, the shell `c` and the chamber gas `b`.

`rocket.py` has been a **declaration exercise** since it was written: every
agent's ``boundary_response`` is `capability.linear_response` on a seeded random
matrix, so no rocket physics had ever been run through Atlas.  This module is the
first half of closing that -- the two sides of the **b-c seam**, wired to the
build repo's own solvers, on the build repo's own geometry, at the build repo's
own operating point.

**Nothing here reimplements a solver.**  `ThermoStruct2D` and `Compressible2D`
are imported verbatim under the private name `atlas_build_solvers` (both repos
have a top-level ``atlas``; see `thermal_seam.load_solvers`), the meshes come from
`grid.build_blocks` rather than being rebuilt here, and the operating point is the
midpoint of `data/sweep.py`'s own declared corpus range.  A number measured
through this module is a number about the build repo at the commit it is pinned
to, not about a second model of a rocket written inside the vault.

**Build repo pinned at 0a407b7** (branch ``atlas-0.1-windfarm``), 2026-09-17. It
is a second checkout and it can drift; a measurement quoted from here carries the
commit or it carries nothing.

---

## What this module does NOT do, and why

**MECH is not wired.**  The b-c edge carries MECH as well as THERM, and the MECH
response is left on its stub deliberately rather than by omission.  Two reasons,
both structural:

1. `solve_mechanical` is **quasi-static and free-body** -- it removes the three
   planar rigid-body modes by projection.  A net pressure load on the inner face
   has a nonzero resultant, which is exactly a rigid-body direction, so the
   response to the physically natural trace is the part of the load that is *not*
   the thrust.  The missing part is not small and it is not noise: it is the
   force the trajectory agent integrates.
2. MECH's declared bond is ``(traction, velocity)`` and a quasi-static solve
   returns a **displacement**.  Turning it into the declared flow needs a time
   derivative, and which time derivative depends on the multirate convention R2
   is about.

So the MECH port is the *field-to-lumped* coupling W97 closed at CS-10 wearing a
different hat, and it belongs with the trajectory agent rather than ahead of it.
Wiring it here would have produced a number, and the number would have been the
projection's null space rather than the shell's response.

## The seam, exactly

    gas   `b`  chamber, 112 x 80 body-fitted cells, z in [0.12, 0.40]
               the seam is its **jmax** wall -- `generate.py::_iface_for` maps
               ("b","c") to ("b", 0, "jmax"), and the upper shell panel sits at
               y = +h_in(z), so this is the same face the build repo couples
    shell `c`  upper panel, 232 x 8 cells, z in [-4.0, 0.70]
               the seam is the part of its **inner** face (j = 0) whose cell
               centres lie in [0.12, 0.40] -- **14 cells of the 232**

**112 cells against 14 is a genuinely non-conforming interface**, which is what
the declared prolongation pair is for: both sides map to one interface space M,
and `probe_block` says a non-conforming interface costs no probes beyond the ones
already budgeted.

**And it is where `rocket.py`'s uniform ``M_EFF = 12`` stops being true.**  The
airframe mesh is uniform over 4.7 m, so a cell is 20.3 mm and the chamber wall
gets 14 of them.  At `ground_effect`'s cutoff wavelength of 4 cells -- the value
every classical-solver case in this vault uses -- that side resolves
``2*14//4 + 1 = 8`` modes, and ``dim M = min_i m_i_eff`` is therefore **8, not
12**, however finely the gas declares its own side.  The fixture's 12 was a
convenience of a graph with no geometry in it.
"""
from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from ..capability import (
    BCChannel, ClaimType, Differentiable, Direction, EllipticSubsolve,
    ExpertCapabilities, MotionClass, TimeDiscretization, port_decl,
)
from ..ports import PortType, ResponseHalf
from ..transfer import Prolongation
from .thermal_seam import build_repo, load_solvers

# ---------------------------------------------------------------------------
# the build repo, under the private name
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_rocket_modules():
    """`config`, `geometry.contours` and `solvers.grid` from the build repo.

    `thermal_seam.load_solvers` binds the whole package under
    ``atlas_build_solvers`` and is `lru_cache`d, so calling it first is what makes
    these three importable at all.  Kept separate from that function because the
    thermal seam needs none of them and importing `config` there would have made
    a schematic duct depend on a rocket's YAML.
    """
    load_solvers()
    return (
        importlib.import_module("atlas_build_solvers.config"),
        importlib.import_module("atlas_build_solvers.geometry.contours"),
        importlib.import_module("atlas_build_solvers.solvers.grid"),
    )


@lru_cache(maxsize=1)
def rocket_config():
    """`config/atlas_0_1.yaml`, the declared single source of truth.

    **A discrepancy this module does not paper over.**  That file's ``dt_model``
    is ``a,b,e = 1e-3 | d,f,g = 5e-3 | c = 5e-2`` with ``dt_macro = 5e-2``, a
    clock spread of **50:1**, and both plan documents quote those numbers.
    `rocket.py` as written declares ``1e-5 / 1e-4 / 1e-3`` with ``macro_dt =
    1e-3``, a spread of **100:1**.  The fixture contradicts the source of truth,
    and `CLOCKS` below takes the YAML's side -- because the YAML says it is the
    source of truth and the fixture's docstring says it is a fixture.
    """
    cfgmod, _contours, _grid = load_rocket_modules()
    return cfgmod.load_config()


# ---------------------------------------------------------------------------
# the operating point -- every number from the build repo, none invented here
# ---------------------------------------------------------------------------

#: Midpoint of `data/sweep.py`'s own declared corpus ranges, ``P_C_RANGE =
#: (2.0e6, 8.0e6)`` and ``T_C_RANGE = (2200.0, 3400.0)``.  Taking the midpoint of
#: the range the corpus is swept over is the one choice here that is not a number
#: read off a file, and it is stated rather than buried: a probe base is a
#: declaration (W74) and a declaration with no provenance is the thing this vault
#: keeps finding.
P_CHAMBER = 5.0e6            # Pa
T_CHAMBER = 2800.0           # K
GAMMA_GAS = 1.22             # sweep.GAMMA_GAS -- hot combustion products
R_GAS = 361.0                # sweep.R_GAS, J/(kg K), ~23 g/mol products
U_CHAMBER = 50.0             # m/s, subsonic chamber flow at A_c/A_t = 2.5

#: `generate.py` line 166: the shell is initialised at 288.15 K, which is also
#: `solve_mechanical`'s ``T_ref``.  That is the COLD START, not the operating
#: point, and using it as a probe base would be W74 with a different number.
T_SHELL_COLD = 288.15
T_AMBIENT = 250.0            # K, `thermostruct2d.step_thermal`'s own T_inf default
H_OUTER = 20.0               # W/(m^2 K), free-convection outer face, thermal_seam's value

#: **The shell has no steady state during a burn, and calling one `settled` would
#: have been a number with a hidden horizon.**  Measured 2026-09-17: under the
#: gas's own conduction-limited h the seam wall passes 300 K at 0.2 s, 550 K at
#: 10 s and 1390 K at 100 s, still rising -- it is a march, not a settle.  So the
#: probe base carries a BURN TIME, the way CS-10's sensitivity carries its
#: rollout horizon after that horizon reversed the sign of the answer.
#:
#: 5.0 s is a round number inside the window the shell's own `validity` predicate
#: admits -- that predicate declines at 10.35 s, when the throat-end node reaches
#: the Al-Li solidus -- and `march` reports the whole curve so the choice is
#: visible rather than load-bearing.
T_PROBE_BURN = 5.0           # s of burn at which the probe base is taken

#: Inner face away from the b-c window: forward of z = 0.12 is the tank barrel,
#: whose interior the config declares ``unmodelled``, and aft of the throat is
#: agent `e`.  Neither is this port, so neither gets the chamber's h.  A small
#: number rather than zero: an adiabatic patch next to a 2800 K one is a stronger
#: claim than a weakly coupled one, and it is not the claim being made here.
H_INNER_BACKGROUND = 5.0     # W/(m^2 K)

#: The z window of the b-c seam, from `contours.wall_b_c`: the chamber inner wall
#: from the a-b plane to the throat.  0.12 is a literal in that function; the
#: throat is `geometry.z_throat`.
Z_SEAM_LO = 0.12

#: `ground_effect.LAMBDA_CUT_CELLS`, and `wing_fsi`'s -- the value every
#: CLASSICAL-solver case in this vault uses.  `wake_array` uses 8, for a frozen
#: neural checkpoint whose own receptive field is coarser.  These are classical
#: solvers, so 4.
LAMBDA_CUT_CELLS = 4


def modes_for(n_cells: int) -> int:
    """``m_eff`` from a cutoff WAVELENGTH, not from a mode count (W0 4.2).

    `ground_effect.modes_for` verbatim, and verbatim on purpose: the basis is a
    declaration, and comparing two experts through two different declarations
    confounds the expert with the presentation.
    """
    return int(2 * n_cells // LAMBDA_CUT_CELLS + 1)


#: The YAML's ``dt_model``, which is what the plan documents quote.
def clocks() -> dict[str, float]:
    return dict(rocket_config().dt_model)


# ---------------------------------------------------------------------------
# geometry -- taken from `grid.build_blocks`, never rebuilt
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def agent_blocks(agent_id: str):
    """`grid.build_blocks` for one agent. The build repo's own mesh, not a copy.

    `generate.py` builds its solvers from exactly this call, so the cells a
    number here is measured on are the cells that repo's own runs march.
    """
    _cfgmod, _contours, grid = load_rocket_modules()
    cfg = rocket_config()
    return grid.build_blocks(cfg.agent(agent_id), cfg)


def shell_panel():
    """The UPPER airframe panel. ``_panel_nodes(upper=True)`` runs j from the
    inner surface to the outer one, so j = 0 is the gas side -- which is
    `ShellMesh`'s own ``inner_j`` default, so nothing has to be told."""
    return agent_blocks("c")[1]


def chamber_block():
    """Agent `b`, one body-fitted block, y from ``-h_in(z)`` to ``+h_in(z)``.

    jmax is therefore the UPPER wall, at ``y = +h_in``, where the upper shell
    panel's inner face is.  `generate.py::_iface_for` maps ("b","c") to
    ``("b", 0, "jmax")``, so this is the same face.
    """
    return agent_blocks("b")[0]


def seam_cells_bc(block) -> np.ndarray:
    """Indices of the shell cells on the b-c seam: cell centres in [0.12, z_throat].

    Selected by centroid rather than by an index arithmetic that would silently
    stop agreeing with the config if ``shell_z`` or the grid ever moved.
    """
    geo = rocket_config().geometry
    zc = 0.5 * (block.nodes[:-1, 0, 0] + block.nodes[1:, 0, 0])
    return np.nonzero((zc >= Z_SEAM_LO) & (zc <= geo.z_throat))[0]


def face_lengths(block, j: int) -> np.ndarray:
    """Arc lengths of the constant-j face cells -- the V-space measure."""
    return np.asarray(block.a_j[:, j], dtype=float)


def fourier_basis(weights: np.ndarray, m: int) -> np.ndarray:
    """(n, m) real Fourier modes, orthonormal in the ``weights``-weighted pairing.

    `ground_effect.fourier_basis` generalised from a uniform ``dx`` to a
    per-cell length, because the shell's seam cells are uniform in z and the
    gas's are not once the nozzle starts contracting.  Orthonormalised against
    the actual Gram rather than assumed orthonormal: on a non-uniform grid the
    closed-form cosines are NOT orthogonal, and assuming it would put the error
    in the forced adjoint where nothing looks for it.
    """
    w = np.asarray(weights, dtype=float).ravel()
    n = w.size
    s = np.concatenate([[0.0], np.cumsum(w)])
    y = 0.5 * (s[:-1] + s[1:])                  # arc-length cell centres
    length = float(s[-1])
    cols = [np.ones(n)]
    k = 1
    while len(cols) < m:
        phi = 2.0 * np.pi * k * y / length
        cols.append(np.cos(phi))
        if len(cols) < m:
            cols.append(np.sin(phi))
        k += 1
    A = np.column_stack(cols[:m])
    # Gram-Schmidt in the w-weighted inner product. G = A^T diag(w) A = I after.
    Q = np.zeros_like(A)
    for j in range(A.shape[1]):
        v = A[:, j].copy()
        for i in range(j):
            v -= (Q[:, i] * w * v).sum() * Q[:, i]
        nrm = np.sqrt(float((v * w * v).sum()))
        if nrm <= 0.0:
            raise ValueError(f"interface basis column {j} is dependent on its predecessors")
        Q[:, j] = v / nrm
    return Q


def face_prolongation(agent_id: str, port_name: str, weights: np.ndarray,
                      m: int, label: str) -> Prolongation:
    return Prolongation(
        agent_id=agent_id,
        port_name=port_name,
        matrix=fourier_basis(weights, m),
        gram_V=np.diag(np.asarray(weights, dtype=float).ravel()),
        label=label,
    )


# ---------------------------------------------------------------------------
# the flux convention -- W66's subject, and the reason gate 4 can be run at all
# ---------------------------------------------------------------------------

#: `thermal_seam.FLUX_CONVENTIONS`, same meaning.  ``entropy`` returns THERM's
#: declared flow ``q_n/T``; ``heat`` returns ``q_n``, the pseudo-bond every
#: co-simulation code exchanges and the one `generate.py` itself passes around.
#:
#: **The difference is what makes the linearity control runnable.**  ``heat`` is
#: exactly AFFINE in the trace on the shell side, so the probed block must
#: reproduce an independently assembled operator to solver tolerance -- an
#: instrument check that can fail.  ``entropy`` divides by T and is not affine,
#: which is W74's cause stated as an algebraic fact.
FLUX_CONVENTIONS = ("entropy", "heat")


def _as_flow(q_n: np.ndarray, T_face: np.ndarray, convention: str) -> np.ndarray:
    if convention == "entropy":
        return q_n / np.maximum(T_face, 1.0)
    if convention == "heat":
        return q_n
    raise ValueError(f"flux_convention must be one of {FLUX_CONVENTIONS}, got {convention!r}")


# ---------------------------------------------------------------------------
# agent `c` -- the airframe shell
# ---------------------------------------------------------------------------


@dataclass
class ShellAgent:
    """`thermostruct2d.ThermoStruct2D` on the upper airframe panel.

    The boundary channel is genuinely **Robin** -- ``-k dT/dn = h (T - T_gas)``,
    the agent's own ``_robin`` surface term -- so the declaration
    ``bc_channel=ROBIN`` is a fact about the solver rather than a convenience.
    `thermal_seam`'s shell says the same thing about the same class, and it is
    still the only channel in this vault above ``dirichlet`` on the ladder.

    The trace enters on the b-c window of the inner face.  **The rest of that
    face is not part of the port** and is held at a declared background: forward
    of z = 0.12 is the tank barrel, whose interior the config declares
    ``unmodelled``, and aft of the throat is agent `e`.  Which background is a
    modelling choice, so it is a field and it is reported in ``probe_state``.
    """

    agent_id: str = "c"
    dt: float | None = None                 # None -> the YAML's dt_model['c']
    flux_convention: str = "entropy"
    expose_elliptic: bool = False
    #: Per-SEAM-CELL Robin coefficient, from the gas's own near-wall state.  A
    #: scalar is broadcast.  It is NOT uniform in reality -- measured over this
    #: wall it runs 149.4 to 356.5 W/(m^2 K), a factor of 2.4 -- and collapsing
    #: it to a mean would discard structure the seam response actually has.
    h_in: Any = 186.6                       # W/(m^2 K); set from gas.wall_h()
    T_background: float = T_SHELL_COLD      # the non-seam inner face
    _ts: Any = field(default=None, repr=False)
    _blk: Any = field(default=None, repr=False)
    _cells: Any = field(default=None, repr=False)
    _T0: Any = field(default=None, repr=False)
    _n_face: int = 0
    _calls: int = 0

    def __post_init__(self) -> None:
        _C2, TS, _TH, _GR = load_solvers()
        if self.dt is None:
            self.dt = float(clocks()["c"])
        self._blk = shell_panel()
        self._ts = (_explicit_shell_class() if self.expose_elliptic
                    else TS.ThermoStruct2D)(TS.ShellMesh(self._blk.nodes), TS.SolidMaterial())
        self._cells = seam_cells_bc(self._blk)
        self._n_face = int(self._ts.mesh.shape[0])
        self._T0 = np.full(self._ts.mesh.n_nodes, T_SHELL_COLD)

    # -- geometry ----------------------------------------------------------
    @property
    def n_seam(self) -> int:
        return int(self._cells.size)

    def seam_weights(self) -> np.ndarray:
        """Arc lengths of the seam's own face cells -- the V-space measure."""
        _a, _b, L = self._ts._face("inner")
        return np.asarray(L, dtype=float)[self._cells]

    def _face_T(self, T: np.ndarray) -> np.ndarray:
        a, b, _L = self._ts._face("inner")
        return 0.5 * (T[a] + T[b])

    def _full_trace(self, seam_trace: np.ndarray) -> np.ndarray:
        """Scatter the port's trace onto the whole inner face, background elsewhere."""
        full = np.full(self._n_face, float(self.T_background))
        full[self._cells] = np.asarray(seam_trace, dtype=float).ravel()
        return full

    def _h_field(self) -> np.ndarray:
        """The Robin coefficient over the WHOLE inner face: the gas's h on the
        seam window, the declared background elsewhere."""
        full = np.full(self._n_face, H_INNER_BACKGROUND)
        full[self._cells] = np.broadcast_to(
            np.asarray(self.h_in, dtype=float).ravel(), (self.n_seam,))
        return full

    def h_seam(self) -> np.ndarray:
        """The Robin coefficient on the seam cells alone."""
        return self._h_field()[self._cells]

    # -- the solve ---------------------------------------------------------
    def _step(self, T: np.ndarray, T_gas_full: np.ndarray) -> np.ndarray:
        self._calls += 1
        return self._ts.step_thermal(T, self.dt, self._h_field(), T_gas_full,
                                     H_OUTER, T_AMBIENT, T_inf=T_AMBIENT)

    def n_port(self, port_name: str) -> int:
        """Cells on the named port. `b:*` is the chamber window; `d:*` the OUTER
        face over the whole airframe."""
        return self._n_face if _neighbour_of(port_name) == "d" else self.n_seam

    def port_weights(self, port_name: str) -> np.ndarray:
        which = "outer" if _neighbour_of(port_name) == "d" else "inner"
        _a, _b, L = self._ts._face(which)
        L = np.asarray(L, dtype=float)
        return L if which == "outer" else L[self._cells]

    def _outer_face_T(self, T: np.ndarray) -> np.ndarray:
        a, b, _L = self._ts._face("outer")
        return 0.5 * (T[a] + T[b])

    def respond(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        """(gas temperature on the port) -> (the declared THERM flow there).

        Two ports, and they are different faces of the same expert:

        ``b:THERM``  the chamber window of the INNER face -- one backward-Euler
                     step, then the heat that crossed it, ``h (T_gas - T_wall)``.
        ``d:THERM``  the OUTER face over the whole airframe, driven through
                     `step_thermal`'s own ``h_out, T_gas_out`` arguments. This is
                     the face the atmosphere sees, and it is the one `c-d`
                     declares.
        """
        if _neighbour_of(port_name) == "d":
            T_out = np.asarray(trace, dtype=float).reshape(self._n_face)
            full_in = self._full_trace(self.base_trace())
            T = self._ts.step_thermal(self._T0, self.dt, self._h_field(), full_in,
                                      H_OUTER, T_out, T_inf=T_AMBIENT)
            T_face = self._outer_face_T(T)
            q = H_OUTER * (T_out - T_face)                  # W/m^2, into the shell
            return _as_flow(q, 0.5 * (T_out + T_face), self.flux_convention)
        full = self._full_trace(trace)
        T = self._step(self._T0, full)
        T_face = self._face_T(T)[self._cells]
        T_gas = full[self._cells]
        q = self.h_seam() * (T_gas - T_face)                # W/m^2, into the shell
        return _as_flow(q, 0.5 * (T_gas + T_face), self.flux_convention)

    def respond_integrated(self, port_name: str, trace: np.ndarray,
                           n_substeps: int) -> np.ndarray:
        """The same flow, averaged over ``n_substeps`` of the shell's own clock.

        See `thermal_seam.ShellAgent.respond_integrated`: `respond` restarts from
        ``_T0`` every call, so calling it n times recomputes the first step n
        times rather than marching n.  R9 requires BOTH sides to be able to state
        their integral -- a matching condition only the fast side has to satisfy
        is not a matching condition -- and the shell is the slow side here.
        """
        n = max(1, int(n_substeps))
        full = self._full_trace(trace)
        T_gas = full[self._cells]
        h_s = self.h_seam()
        Tn = self._T0
        acc = np.zeros(self.n_seam)
        for _ in range(n):
            Tn = self._step(Tn, full)
            T_face = self._face_T(Tn)[self._cells]
            q = h_s * (T_gas - T_face)
            acc = acc + _as_flow(q, 0.5 * (T_gas + T_face), self.flux_convention)
        return acc / n

    # -- the operating point -----------------------------------------------
    def march(self, T_gas, h_in, burn_time: float = T_PROBE_BURN,
              record: bool = False) -> dict:
        """March the shell under the chamber load for ``burn_time`` seconds.

        **W74's requirement met by measurement, and CS-10's convention obeyed.**
        The probe linearises about `probe_base`, and the shell's cold start at
        288.15 K is a state it occupies for one instant of a burn and never
        again.  But there is no state to settle ON either: measured here, the
        seam wall is still rising at 100 s.  So this is a MARCH with a stated
        horizon, not a settle, and the horizon is part of the probe state -- the
        same convention CS-10 bought after a sensitivity changed SIGN with its
        rollout horizon.

        Returns the curve as well as the endpoint, because a base quoted without
        the curve it sits on is a number whose reader cannot tell whether it was
        chosen or derived.  ``validity_lost_s`` is the first burn time at which
        the agent's OWN declared predicate declines -- read off the predicate,
        not off a threshold chosen here.
        """
        self.h_in = h_in
        full = np.full(self._n_face, float(self.T_background))
        full[self._cells] = np.broadcast_to(
            np.asarray(T_gas, dtype=float).ravel(), (self.n_seam,))
        T = np.full(self._ts.mesh.n_nodes, T_SHELL_COLD)
        n = max(1, int(round(float(burn_time) / self.dt)))
        curve, lost = [], None
        for k in range(1, n + 1):
            T = self._step(T, full)
            if lost is None and not self.validity(T):
                lost = k * self.dt
            if record:
                face = self._face_T(T)[self._cells]
                curve.append((k * self.dt, float(face.mean()), float(face.max())))
        self._T0 = T
        face = self._face_T(T)[self._cells]
        return {
            "burn_time_s": n * self.dt,
            "steps": n,
            "T_wall_mean_K": float(face.mean()),
            "T_wall_min_K": float(face.min()),
            "T_wall_max_K": float(face.max()),
            "valid_at_base": bool(self.validity(T)),
            "validity_lost_s": lost,
            "curve": curve,
        }

    def base_trace(self, port_name: str = "b:THERM") -> np.ndarray:
        """The probe base for the named port -- W74, per face.

        The inner window is linearised about the chamber; the outer face about
        the atmosphere it actually radiates to. Using one number for both would
        put the outer face at 2800 K, which is the error W74 is about with a
        different sign.
        """
        if _neighbour_of(port_name) == "d":
            return np.full(self._n_face, ambient_state()[2])
        return np.full(self.n_seam, T_CHAMBER)

    # -- the record's callables --------------------------------------------
    def storage(self, state=None) -> float:
        """Thermal energy ``1/2 T^T M T``: the conduction operator's own Lyapunov
        functional, SPD by construction from the Q1 mass matrix."""
        T = self._T0 if state is None else np.asarray(state)
        return float(0.5 * T @ self._ts.M_th.dot(T))

    def validity(self, state=None, cond=None) -> bool:
        """Where the declaration is believed, stated as a predicate that can say no.

        `SolidMaterial` is an aluminium-lithium-like alloy with temperature
        INDEPENDENT k, cp, E and alpha.  Above the solidus those constants are
        not a small error, they are the wrong model, so the predicate declines
        rather than extrapolating.  770 K is a conservative Al-Li solidus.
        """
        T = self._T0 if state is None else np.asarray(state)
        return bool(np.all(np.asarray(T, dtype=float) < 770.0))

    # -- MECH, and the half of it this solver cannot carry (W305) -----------
    def face_normals(self, which: str = "inner") -> np.ndarray:
        """Outward unit normals of a face's cells, as `solve_mechanical` builds
        them: from the node pair, rotated, normalised by the cell length."""
        a, b, L = self._ts._face(which)
        nodes = self._ts.mesh.nodes.reshape(-1, 2)
        tang = nodes[b] - nodes[a]
        return np.stack([-tang[:, 1], tang[:, 0]], axis=1) / np.maximum(L, 1e-30)[:, None]

    def _p_fields(self, p_seam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(inner, outer) pressure over the WHOLE faces, with the seam window
        carrying ``p_seam`` and the declared ambient elsewhere."""
        p_in = np.full(self._n_face, float(ambient_state()[1]))
        p_in[self._cells] = np.asarray(p_seam, dtype=float).ravel()
        p_out = np.full(self._n_face, float(ambient_state()[1]))
        return p_in, p_out

    def respond_mech(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        """(normal traction on the seam) -> (normal VELOCITY on the seam).

        **The time derivative W305 said R2 would have to decide.**  MECH's bond
        is ``(traction, velocity)`` and `solve_mechanical` is quasi-static: it
        returns a DISPLACEMENT.  R2 settled the convention by declaring
        `FluxMatching.TIME_INTEGRATED` over the macro-step, so the flow is the
        displacement increment over that interval, ``v = du / dt``, taken
        against the unloaded configuration.  It is a declaration, it is reported,
        and §14's own lesson applies: a quantity divided by dt is not cadence
        free, so a beta measured here carries its clock.
        """
        del port_name
        p_in, p_out = self._p_fields(trace)
        u, _sigma = self._ts.solve_mechanical(self._T0, p_in, p_out)
        self._calls += 1
        a, b, _L = self._ts._face("inner")
        u_face = 0.5 * (u[a] + u[b])                      # per inner-face cell
        nrm = self.face_normals("inner")
        return (u_face * nrm).sum(1)[self._cells] / self.dt

    def rigid_load(self, p_seam: np.ndarray) -> np.ndarray:
        """The net rigid-body load the free-body projection REMOVES, as ``V^T f``.

        **This quantity already exists inside the build repo and is thrown
        away.**  `_solve_free` solves the bordered system

            [ K   V ] [u]   [f]
            [ V^T 0 ] [l] = [0]

        and returns ``lu.solve(rhs)[: f.size]`` -- it computes the multiplier and
        slices it off on the next character.  Because ``K V = 0`` and ``V`` is
        orthonormal, left-multiplying the first block row by ``V^T`` gives
        ``l = V^T f`` exactly: the net force and torque on the body, which is
        precisely what `trajectory.rk4_step` integrates.  So the structure and
        the trajectory already share an interface variable; nothing declares it.

        It is recovered here by the surface integral rather than by changing the
        build repo, which is the same construction `generate.py::_compute_loads`
        uses for `F_aero`.  The thermal body force is self-equilibrated on a free
        body and contributes nothing to ``V^T f``, which is asserted by the
        tier's test rather than assumed.
        """
        p_in, p_out = self._p_fields(p_seam)
        V = self._ts._rigid_modes()
        f = np.zeros(self._ts.mesh.n_nodes * 2)
        for which, p, sign in (("inner", p_in, +1.0), ("outer", p_out, -1.0)):
            a, b, L = self._ts._face(which)
            nrm = self.face_normals(which)
            trac = sign * np.asarray(p, float)[:, None] * nrm * L[:, None] * 0.5
            for nid in (a, b):
                np.add.at(f, 2 * nid, trac[:, 0])
                np.add.at(f, 2 * nid + 1, trac[:, 1])
        return V.T @ f

    @property
    def solver_calls(self) -> int:
        return self._calls


#: **W305.**  `Compressible2D`'s wall conditions, verbatim from its own `BC`
#: docstring.  None of them takes a wall VELOCITY: `_reflect(no_slip=True)`
#: negates the ghost momentum, which is a stationary wall, and the only
#: parameter `wall_noslip` reads is `T_wall` (plus `no_slip_mask`).  So a gas
#: agent can SUPPLY a traction and cannot RESPOND to a velocity, and the MECH
#: bond at every gas-solid seam in this graph is one-sided by a missing
#: capability rather than by a modelling choice.
COMPRESSIBLE2D_BC_KINDS = ("wall_slip", "wall_noslip", "symmetry",
                           "inlet_massflow", "freestream", "outflow",
                           "prescribed", "extrapolate")
WALL_BC_PARAMS = ("T_wall", "no_slip_mask")


@dataclass
class TrajectoryAgent:
    """The 3-DOF planar rigid body -- `solvers/trajectory.py` used verbatim.

    That module's own docstring says it is *"written once and reused verbatim by
    Atlas's non-learned `rigid_body` expert ... not a data-generation helper that
    gets replaced later"*, so this wrapper adds a port and a record and changes
    no physics.  State is ``(x, y, theta, vx, vy, omega, m)`` in the world frame.

    **Its port is the rigid-mode space, and that space has dimension 3, not 1.**
    The brief called this agent *"lumped, dim M = 1 -- the cheapest of all"*; a
    planar rigid body has two translations and one rotation, and the structure's
    own `_rigid_modes` returns a ``[2n, 3]`` basis.  One is the count for a
    single scalar channel, and this bond is not one.
    """

    dt: float = 5.0e-2
    state: Any = None
    mdot: float = 0.0
    inertia: float = 1.0e4
    _traj: Any = field(default=None, repr=False)
    _calls: int = 0

    def __post_init__(self) -> None:
        self._traj = _trajectory_module()
        if self.state is None:
            h0, v0 = _launch_state()
            self.state = self._traj.initial_state(h0=h0, v0=v0)

    @property
    def n_seam(self) -> int:
        return 3

    def seam_weights(self) -> np.ndarray:
        """A lumped port has no measure; the three modes are already orthonormal
        in the structure's own basis, so the weight is one apiece."""
        return np.ones(3)

    def _loads_from(self, rigid: np.ndarray):
        """A rigid-mode load 3-vector -> the build repo's own `Loads` record.

        The structure's `V` is QR-orthonormalised from (translation-z,
        translation-y, rotation about the centroid), so component 2 is a torque
        and the first two are forces.  `Loads` keeps thrust and aero apart
        because the trajectory records them separately; a probe perturbs the
        total, so the whole vector is handed over as `F_aero` and the split is
        left to whoever has both halves.
        """
        r = np.asarray(rigid, dtype=float).ravel()
        return self._traj.Loads(F_thrust=(0.0, 0.0),
                                F_aero=(float(r[0]), float(r[1])),
                                torque=float(r[2]), mdot=self.mdot,
                                inertia=self.inertia)

    def respond(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        """(rigid-mode load) -> (the body's velocity over one step).

        The conjugate flow is a velocity, which this agent has natively -- it is
        the one port in this graph that needs no time derivative supplied, and
        the contrast with `ShellAgent.respond_mech` is the point of the pair.
        """
        del port_name
        s = self._traj.rk4_step(self.state, self.dt, self._loads_from(trace))
        self._calls += 1
        return np.array([s[3], s[4], s[5]])

    def base_trace(self) -> np.ndarray:
        """W74 on a lumped port: the load the body is actually flying under."""
        return np.zeros(3)

    def march(self, loads, steps: int = 1):
        for _ in range(int(steps)):
            self.state = self._traj.rk4_step(self.state, self.dt,
                                             self._loads_from(loads))
            self._calls += 1
        return self.state

    def storage(self, state=None) -> float:
        """Kinetic plus gravitational potential -- the natural Lyapunov
        functional, and NOT conserved here, which is why `trajectory.py`'s own
        docstring refuses a symplectic integrator: mass is expelled, drag
        dissipates and thrust does work."""
        s = self.state if state is None else np.asarray(state, dtype=float)
        _atm = _atmosphere_module()
        m = max(float(s[6]), 1e-6)
        ke = 0.5 * m * (s[3] ** 2 + s[4] ** 2) + 0.5 * self.inertia * s[5] ** 2
        return float(ke + m * _atm.gravity(float(s[1])) * float(s[1]))

    def validity(self, state=None, cond=None) -> bool:
        """Two ways this agent stops being believed, both read off its sources.

        `atmosphere.py` is the 1976 standard, tabulated to 86 km; above it the
        density it returns is an extrapolation of a model that has ended.  And
        `rhs` floors the mass at 1e-6 kg, so a burn past the propellant load
        returns a number rather than failing -- the floor is a guard, not a
        licence, and the predicate says so.
        """
        del cond
        s = self.state if state is None else np.asarray(state, dtype=float)
        return bool(float(s[6]) > 1.0 and 0.0 <= float(s[1]) <= 86.0e3)

    @property
    def solver_calls(self) -> int:
        return self._calls


@lru_cache(maxsize=1)
def _trajectory_module():
    load_solvers()
    return importlib.import_module("atlas_build_solvers.solvers.trajectory")


@lru_cache(maxsize=1)
def _atmosphere_module():
    load_solvers()
    return importlib.import_module("atlas_build_solvers.solvers.atmosphere")


def _launch_state() -> tuple[float, float]:
    """``(h0, v0)`` from the config, so the body starts where the corpus does."""
    cfg = rocket_config()
    return (float(getattr(cfg, "h0", 0.0)),
            float(getattr(cfg, "v_flight", 0.0)))


@lru_cache(maxsize=2)
def _explicit_shell_class():
    """`ThermoStruct2D` with the implicit conduction solve replaced by sub-steps.

    `thermal_seam._explicit_shell_class` verbatim in structure and for its
    reason: the agent is not ours to change, and what the composition layer is
    allowed to do is DECLINE TO USE a part of it and supply that part itself.
    The assembly and the Robin terms are untouched.

    The sub-step count is COMPUTED from the diffusion number of this mesh, never
    hard-coded -- hard-coding one is `CASE-STUDY-GUIDE` mistake 5 and cost two
    orders of magnitude in tau the one time it was done.
    """
    _C2, TS, _TH, _GR = load_solvers()

    class ExplicitThermoStruct(TS.ThermoStruct2D):
        def step_thermal(self, T, dt, h_in, T_gas_in, h_out, T_gas_out,
                         radiate=True, T_inf=250.0):
            Kb_i, fb_i = self._robin("inner", h_in, T_gas_in)
            h_o = np.asarray(h_out, dtype=float)
            if radiate:
                a, b, _ = self._face("outer")
                Tw = 0.5 * (T[a] + T[b])
                h_rad = (self.mat.emissivity * 5.670374419e-8
                         * (Tw ** 2 + T_inf ** 2) * (Tw + T_inf))
                h_o = np.broadcast_to(h_o, a.shape) + h_rad
            Kb_o, fb_o = self._robin("outer", h_o, T_gas_out)
            K = self.K_th + Kb_i + Kb_o
            f = fb_i + fb_o
            m = np.asarray(self.M_th.sum(axis=1)).ravel()          # lumped mass
            m = np.maximum(m, 1e-30)
            n = self._substeps_at(dt)
            h = dt / n
            for _ in range(n):
                T = T + h * (f - K.dot(T)) / m
            return T

        def _substeps_at(self, dt: float) -> int:
            alpha = self.mat.diffusivity
            p = self.mesh.nodes.reshape(-1, 2)
            a, b, _L = self._face("inner")
            # wall-normal spacing: one element through the thickness
            nj = self.mesh.shape[1]
            dy = float(np.abs(p[self.mesh.nid(0, 1), 1] - p[self.mesh.nid(0, 0), 1]))
            dy = dy if dy > 0 else float(self.mat.k)        # never silently zero
            dt_stable = 0.4 * dy * dy / (2.0 * alpha)
            del a, b, nj
            return max(1, int(np.ceil(dt / dt_stable)))

    return ExplicitThermoStruct


# ---------------------------------------------------------------------------
# agent `b` -- the combustion chamber
# ---------------------------------------------------------------------------


@dataclass
class ChamberGasAgent:
    """`compressible2d.Compressible2D` on the chamber, seam on the ``jmax`` wall.

    The boundary channel is a genuine isothermal wall: ``BC("wall_noslip",
    {"T_wall": ...})`` sets the ghost density so the FACE temperature is the
    imposed one, which is a Dirichlet channel in temperature and is what makes
    the trace side of the THERM bond real rather than emulated.
    `thermal_seam.GasAgent` verbatim in structure, on the rocket's own mesh.

    **The probe cadence is a declaration, and it is measured, not assumed.**  The
    chamber's CFL sub-step at the operating point is ``2.99e-7`` s, so the YAML's
    own ``dt_model['b'] = 1e-3`` is about 3350 explicit sub-steps -- roughly 2
    minutes a probe column on this machine, half an hour a block.  `dt` is
    therefore a field, its default is the YAML's, and `probe_cadence_sweep`
    exists so that a reduced cadence is reported as a number rather than taken as
    a convenience.
    """

    agent_id: str = "b"
    dt: float | None = None                 # None -> the YAML's dt_model['b']
    flux_convention: str = "entropy"
    T_wall_base: float = T_SHELL_COLD       # overwritten from the shell's settle()
    _sol: Any = field(default=None, repr=False)
    _blk: Any = field(default=None, repr=False)
    _cfg: Any = field(default=None, repr=False)
    _U0: Any = field(default=None, repr=False)
    _n_face: int = 0
    _calls: int = 0
    _substeps: int = 0

    def __post_init__(self) -> None:
        C2, _TS, TH, _GR = load_solvers()
        if self.dt is None:
            self.dt = float(clocks()["b"])
        self._blk = chamber_block()
        self._n_face = int(self._blk.shape[0])
        self._cfg = C2.GasConfig(gamma=GAMMA_GAS, R=R_GAS)
        nz, ny = self._blk.shape
        W = np.zeros((nz, ny, 4))
        W[..., 0] = P_CHAMBER / (R_GAS * T_CHAMBER)
        W[..., 1] = U_CHAMBER
        W[..., 3] = P_CHAMBER
        self._U0 = TH.prim_to_cons(W, GAMMA_GAS)
        self._C2, self._TH = C2, TH

    # -- geometry ----------------------------------------------------------
    @property
    def n_seam(self) -> int:
        return self._n_face

    def seam_weights(self) -> np.ndarray:
        """Arc lengths of the jmax face cells -- the V-space measure.

        NOT uniform: the chamber wall is flat to z = 0.30 and then converges to
        the throat, so the last third of these are longer than the first.  That
        is why `fourier_basis` orthonormalises against the real Gram instead of
        assuming the uniform closed form.
        """
        return face_lengths(self._blk, -1)

    # -- the solve ---------------------------------------------------------
    def _solver(self, T_wall: np.ndarray):
        C2 = self._C2
        rho_in = P_CHAMBER / (R_GAS * T_CHAMBER)
        return C2.Compressible2D(self._blk, self._cfg, bcs={
            "imin": C2.BC("freestream",
                          {"prim": np.array([rho_in, U_CHAMBER, 0.0, P_CHAMBER])}),
            "imax": C2.BC("outflow", {"p_inf": P_CHAMBER}),
            # the LOWER wall is the mirror panel of the same shell; holding it at
            # the same temperature is the symmetry the vehicle has, not a
            # convenience -- a different value there would break it.
            "jmin": C2.BC("wall_noslip", {"T_wall": np.full(self._n_face,
                                                            float(self.T_wall_base))}),
            "jmax": C2.BC("wall_noslip", {"T_wall": T_wall}),
        })

    def _wall_heat_flux(self, U, T_wall) -> tuple[np.ndarray, np.ndarray]:
        """(q_n into the wall [W/m^2], face temperature [K]) on jmax.

        `data/generate.py::_wall_flux` verbatim for ``side == "jmax"``, so the
        number probed here is the one the build repo's own coupler exchanges.
        """
        TH, cfg, blk = self._TH, self._cfg, self._blk
        W = TH.cons_to_prim(U, cfg.gamma)
        T = W[..., 3] / (W[..., 0] * cfg.R)
        dn = 0.5 * blk.vol[:, -1] / np.maximum(blk.a_j[:, -1], 1e-30)
        mu = TH.sutherland(T[:, -1], cfg.mu_ref, cfg.T_mu_ref, cfg.sutherland_S)
        k_gas = mu * cfg.cp / cfg.Pr
        Tw = np.broadcast_to(np.asarray(T_wall, float), T[:, -1].shape)
        q = k_gas * (T[:, -1] - Tw) / np.maximum(dn, 1e-9)
        return q, 0.5 * (T[:, -1] + Tw)

    def respond(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        """(wall temperature on the seam) -> (the declared THERM flow on the seam)."""
        T_wall = np.asarray(trace, dtype=float).reshape(self._n_face)
        sol = self._solver(T_wall)
        U, n = sol.advance(self._U0, self.dt)
        self._calls += 1
        self._substeps += int(n)
        q, T_face = self._wall_heat_flux(U, T_wall)
        return _as_flow(q, T_face, self.flux_convention)

    def respond_integrated(self, port_name: str, trace: np.ndarray,
                           n_substeps: int) -> np.ndarray:
        """The same flow, averaged over ``n_substeps`` of this agent's own clock."""
        n = max(1, int(n_substeps))
        T_wall = np.asarray(trace, dtype=float).reshape(self._n_face)
        sol = self._solver(T_wall)
        U = self._U0
        acc = np.zeros(self._n_face)
        for _ in range(n):
            U, k = sol.advance(U, self.dt)
            self._calls += 1
            self._substeps += int(k)
            q, T_face = self._wall_heat_flux(U, T_wall)
            acc = acc + _as_flow(q, T_face, self.flux_convention)
        return acc / n

    def base_trace(self) -> np.ndarray:
        """The probe base: the wall temperature this side is linearised about.

        **W74.**  THERM's effort is an absolute temperature, so the probe's
        default zero trace asks a 2800 K gas for its flux against a **0 K** wall.
        On `thermal_seam` declaring the operating point instead moved beta by a
        factor of 12.6; here the default would be 0 K against a settled wall, so
        the same class of error is available and is closed the same way.
        """
        return np.full(self._n_face, float(self.T_wall_base))

    def wall_h(self, T_wall: float | None = None) -> np.ndarray:
        """The conduction-limited ``h`` the shell's Robin channel needs.

        This is the one quantity that crosses in the opposite direction to the
        port: `generate.py::_step_structure` takes ``h`` from the gas and hands
        it to `step_thermal`.  It is NOT on a port and is not certified; it is a
        coefficient of the shell's own channel, and it is computed here so the
        two sides are set up at the same operating point rather than at two.
        """
        Tw = self.T_wall_base if T_wall is None else float(T_wall)
        sol = self._solver(np.full(self._n_face, Tw))
        U, n = sol.advance(self._U0, self.dt)
        self._calls += 1
        self._substeps += int(n)
        TH, cfg, blk = self._TH, self._cfg, self._blk
        W = TH.cons_to_prim(U, cfg.gamma)
        T = W[..., 3] / (W[..., 0] * cfg.R)
        dn = 0.5 * blk.vol[:, -1] / np.maximum(blk.a_j[:, -1], 1e-30)
        mu = TH.sutherland(T[:, -1], cfg.mu_ref, cfg.T_mu_ref, cfg.sutherland_S)
        return mu * cfg.cp / cfg.Pr / np.maximum(dn, 1e-9)

    # -- the record's callables --------------------------------------------
    def storage(self, state=None) -> float:
        """Total energy of the chamber gas, the natural Lyapunov functional here."""
        U = self._U0 if state is None else state
        return float(np.sum(U[..., 3] * self._blk.vol))

    def validity(self, state=None, cond=None) -> bool:
        """Where the declaration is believed.

        `GasConfig` is a calorically perfect gas at fixed gamma and R with
        Sutherland viscosity fitted to air.  It declines on a state with a
        negative density or pressure -- the failure a shock-capturing scheme
        actually has -- rather than on a threshold nobody measured.
        """
        U = self._U0 if state is None else np.asarray(state)
        W = self._TH.cons_to_prim(U, self._cfg.gamma)
        return bool(np.all(W[..., 0] > 0.0) and np.all(W[..., 3] > 0.0))

    @property
    def solver_calls(self) -> int:
        return self._calls

    @property
    def solver_substeps(self) -> int:
        return self._substeps


# ---------------------------------------------------------------------------
# the exact shell operator -- R0's gate 4, the control that must PASS
# ---------------------------------------------------------------------------


def shell_thermal_operator(shell: ShellAgent, convention: str = "heat") -> np.ndarray:
    """The shell's seam block, ASSEMBLED rather than probed. (n_seam, n_seam).

    **R0's must-pass control.**  `thermostruct2d`'s conduction step is linear, so
    the shell's response to the seam trace is an operator that exists in closed
    form, and a probe harness that cannot reproduce it is wrong about the
    harness rather than about the physics.  Assembled here from the same
    matrices `step_thermal` builds, and from nothing else:

        A T_new = M/dt T_old + f_in(T_gas) + f_out
        T_face  = E T_new,        q = h (T_gas - T_face)

    so with ``G = d f_in / d T_gas`` restricted to the seam window,

        dq/dT_gas = h (I - E A^-1 G).

    ``radiate`` is off here because the radiative ``h_rad`` is linearised about
    the CURRENT wall temperature inside `step_thermal`, which makes the map
    affine-with-a-frozen-coefficient rather than affine.  That is a real
    difference and it is why the control declares its own configuration: this
    operator is the derivative of the non-radiating step, and the probe it is
    compared against must be run the same way.

    ``convention="heat"`` returns the AFFINE operator.  ``"entropy"`` applies the
    chain rule through ``q/T_bar``, which is not affine -- W74's cause as an
    algebraic fact rather than as a caveat.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve

    ts = shell._ts
    cells = shell._cells
    n_seam = int(cells.size)
    n_nodes = int(ts.mesh.n_nodes)

    h_field = shell._h_field()
    T_gas_full = shell._full_trace(np.full(n_seam, 0.0))
    Kb_i, _f_i = ts._robin("inner", h_field, T_gas_full)
    Kb_o, f_o = ts._robin("outer", H_OUTER, T_AMBIENT)
    A = (ts.M_th / shell.dt + ts.K_th + Kb_i + Kb_o).tocsc()

    # G[:, k] = d f_in / d T_gas[cell k]. `_robin` puts h * T_gas * L / 2 on each
    # of the face's two nodes, so the column is exact and needs no difference.
    a, b, L = ts._face("inner")
    rows, cols, vals = [], [], []
    for k, cell in enumerate(cells):
        w = float(h_field[cell]) * float(L[cell]) / 2.0
        rows.extend([int(a[cell]), int(b[cell])])
        cols.extend([k, k])
        vals.extend([w, w])
    G = csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_seam))

    # E: node temperatures -> seam face temperatures, T_face = 0.5 (T[a] + T[b]).
    erows, ecols, evals = [], [], []
    for k, cell in enumerate(cells):
        erows.extend([k, k])
        ecols.extend([int(a[cell]), int(b[cell])])
        evals.extend([0.5, 0.5])
    E = csr_matrix((evals, (erows, ecols)), shape=(n_seam, n_nodes))

    X = spsolve(A, G.toarray())                      # A^-1 G, (n_nodes, n_seam)
    J_face = np.asarray(E @ X)                       # d T_face / d T_gas
    hh = np.asarray(h_field, dtype=float)[cells]
    dq = np.diag(hh) @ (np.eye(n_seam) - J_face)
    if convention == "heat":
        return dq

    # entropy: phi = q / T_bar, T_bar = 0.5 (T_gas + T_face). Chain rule at the
    # base, which is the point the probe linearises about and must be the same.
    base = shell.base_trace()
    full = shell._full_trace(base)
    T_new = ts.step_thermal(shell._T0, shell.dt, h_field, full,
                            H_OUTER, T_AMBIENT, radiate=False, T_inf=T_AMBIENT)
    T_face = shell._face_T(T_new)[cells]
    T_gas = full[cells]
    q = hh * (T_gas - T_face)
    T_bar = 0.5 * (T_gas + T_face)
    dT_bar = 0.5 * (np.eye(n_seam) + J_face)
    del f_o
    return np.diag(1.0 / T_bar) @ dq - np.diag(q / T_bar ** 2) @ dT_bar


# ---------------------------------------------------------------------------
# the records
# ---------------------------------------------------------------------------

#: (T, q_n/T) in K and W/(m^2 K): s_e s_f = s_P with s_P in W/m^2.  Sized on the
#: chamber rather than on `rocket.py`'s fixture values, which were declared before
#: any physics existed to size them against.
THERM_SCALES = {"temperature": T_CHAMBER,
                "entropy_flux": 1.0e6 / T_CHAMBER,
                "power_area": 1.0e6}


def shell_capabilities(expert: ShellAgent, m_eff: int | None = None,
                       response_half: ResponseHalf = ResponseHalf.FLOW,
                       elliptic: EllipticSubsolve | None = None) -> ExpertCapabilities:
    """The shell's L1 record.

    ``response_half`` is a PARAMETER so that R0's gate-3 control can flip it and
    watch L3/C9 decide.  It defaults to the truth.
    """
    exposed = expert.expose_elliptic
    n = expert.n_seam
    m = modes_for(n) if m_eff is None else int(m_eff)
    w = expert.seam_weights()
    if elliptic is None:
        elliptic = (EllipticSubsolve.EXPOSED if exposed else EllipticSubsolve.EMBEDDED)
    return ExpertCapabilities(
        expert_id=expert.agent_id,
        ports=[port_decl(
            name="b:THERM", port_type=PortType.THERM,
            geometry=f"inner face of the upper airframe panel, z in [{Z_SEAM_LO}, "
                     f"z_throat], {n} line elements of the panel's 232",
            direction=Direction.BIDIRECTIONAL,
            nondim=dict(THERM_SCALES),
            effective_resolution=m,
            motion_class=MotionClass.STATIC,
            response_half=response_half,
            prolongation=face_prolongation(
                expert.agent_id, "b:THERM", w, m,
                f"{m}-mode real Fourier basis, orthonormalised in the arc-length "
                f"Gram of {n} non-uniform seam cells"),
            note="the agent's own Robin surface term, -k dT/dn = h (T - T_gas); "
                 "the rest of the inner face is held at the declared background",
        )],
        # `_robin` IS a Robin condition, not an emulation of one.
        bc_channel=BCChannel.ROBIN,
        bc_time_varying=True,
        # Backward Euler solves (M/dt + K + K_robin) T over the WHOLE panel. That
        # is EMBEDDED in R10's exact sense, and `probe.support_reach` measures it
        # rather than taking the declaration's word.
        elliptic_subsolve=elliptic,
        time_discretization=(TimeDiscretization.EXPLICIT if exposed
                             else TimeDiscretization.IMPLICIT),
        stencil_radius=1,                       # Q1 elements, one ring
        substeps_per_macro_step=1,
        differentiable=Differentiable.NONE,
        dt_native=expert.dt,
        storage=expert.storage,
        equivariances=(),
        validity=expert.validity,
        governing_family="parabolic-conduction-quasistatic-elasticity",
        lambda_ref="thermostruct2d.ThermoStruct2D, same panel and discretization",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash=f"thermostruct2d/1-{'explicit' if exposed else 'backward-euler'}",
        boundary_response=expert.respond,
        boundary_response_integrated=expert.respond_integrated,
        # W74: THERM's effort is an absolute temperature, so the default zero
        # trace is 0 K -- a state this shell is never in.
        probe_base=lambda _p, _e=expert: _e.base_trace(),
        reproducibility_floor=float(np.finfo(float).eps),
        deterministic=True,
        note="solvers/thermostruct2d.py, build repo 0a407b7, unmodified; mesh from "
             "solvers/grid.build_blocks, conduction graded against the erf slab by "
             "that repo's M1 suite",
    )


def chamber_gas_capabilities(expert: ChamberGasAgent, m_eff: int | None = None,
                             response_half: ResponseHalf = ResponseHalf.FLOW,
                             ) -> ExpertCapabilities:
    """The chamber gas's L1 record."""
    n = expert.n_seam
    m = modes_for(n) if m_eff is None else int(m_eff)
    w = expert.seam_weights()
    return ExpertCapabilities(
        expert_id=expert.agent_id,
        ports=[port_decl(
            name="c:THERM", port_type=PortType.THERM,
            geometry=f"jmax wall of the chamber block, {n} cells, z in [0.12, 0.40]",
            direction=Direction.BIDIRECTIONAL,
            nondim=dict(THERM_SCALES),
            effective_resolution=m,
            motion_class=MotionClass.STATIC,
            response_half=response_half,
            prolongation=face_prolongation(
                expert.agent_id, "c:THERM", w, m,
                f"{m}-mode real Fourier basis, orthonormalised in the arc-length "
                f"Gram of {n} non-uniform wall cells"),
            note="isothermal no-slip wall; the trace is T_wall and the response is "
                 "the conduction-limited wall flux generate.py exchanges",
        )],
        # BC("wall_noslip", {"T_wall": ...}) sets the ghost so the FACE
        # temperature is the imposed one. A temperature Dirichlet channel.
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=True,
        # MUSCL + SSP-RK2 + HLLC. Explicit finite volume, no solve of any kind.
        elliptic_subsolve=EllipticSubsolve.NONE,
        time_discretization=TimeDiscretization.EXPLICIT,
        stencil_radius=2,                       # NG = 2; MUSCL needs two ghosts
        # NOT 1. The agent's macro step is many CFL sub-steps and R10's halo is
        # radius x substeps, so declaring 1 here would under-report the domain of
        # dependence by the sub-step count -- which is the W46 failure mode, with
        # the declaration too SMALL rather than too large.
        substeps_per_macro_step=max(1, int(np.ceil(expert.dt / 2.99e-7))),
        differentiable=Differentiable.NONE,
        dt_native=expert.dt,
        storage=expert.storage,
        equivariances=(),
        validity=expert.validity,
        governing_family="compressible-navier-stokes-2d",
        lambda_ref="compressible2d.Compressible2D, same chamber block and discretization",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash="compressible2d/2-hllc-minmod-ssprk2",
        boundary_response=expert.respond,
        boundary_response_integrated=expert.respond_integrated,
        probe_base=lambda _p, _e=expert: _e.base_trace(),
        reproducibility_floor=float(np.finfo(float).eps),
        deterministic=True,
        note="solvers/compressible2d.py, build repo 0a407b7, unmodified; mesh from "
             "solvers/grid.build_blocks, graded against the exact Riemann solution "
             "and the isentropic vortex by that repo's M1 suite",
    )


# ---------------------------------------------------------------------------
# setting the two sides up at ONE operating point
# ---------------------------------------------------------------------------


def gas_h_on_shell_seam(gas: "ChamberGasAgent", shell: "ShellAgent",
                        T_wall: float) -> np.ndarray:
    """The gas's ``h``, reduced from its 112 wall cells onto the shell's 14.

    **This crossing is not on a port and is not certified, and saying so is the
    point.**  `generate.py::_step_structure` takes ``h`` from the gas and hands
    it to `step_thermal` -- it is a coefficient of the SHELL's own channel, not a
    conjugate half, so no bond carries it and no rule bounds it.  It is reduced
    here by conservative area-weighted averaging rather than by
    `generate.py`'s normalized-index `np.interp`, because the two faces cover the
    same physical z window and an index map between different cell counts does
    not.
    """
    h_gas = np.asarray(gas.wall_h(T_wall), dtype=float)
    zg = gas._blk.nodes[:, 0, 0]                       # gas face node z, 113 of them
    zs = shell._blk.nodes[:, 0, 0]                     # shell face node z, 233 of them
    cells = shell._cells
    out = np.empty(cells.size)
    for k, c in enumerate(cells):
        lo, hi = zs[c], zs[c + 1]
        ov = np.clip(np.minimum(zg[1:], hi) - np.maximum(zg[:-1], lo), 0.0, None)
        out[k] = float((h_gas * ov).sum() / ov.sum()) if ov.sum() > 0 else float(h_gas.mean())
    return out


def make_bc_experts(flux_convention: str = "entropy",
                    dt_gas: float | None = None,
                    dt_shell: float | None = None,
                    expose_elliptic: bool = False,
                    burn_time: float = T_PROBE_BURN,
                    record_curve: bool = False,
                    ) -> tuple[ShellAgent, ChamberGasAgent, dict]:
    """Both sides of the b-c seam, marched to ONE operating point.

    The order matters and is not arbitrary.  The gas's conduction-limited ``h``
    is a property of its own near-wall state, and the shell's wall temperature is
    a property of that ``h`` -- so the gas is asked for ``h`` at the cold wall
    first, the shell is marched under it, and the gas's probe base is then set to
    the wall the shell actually reached.  Doing it in the other order gives two
    sides linearised about two different walls, which is exactly the 500 K
    disagreement `base_disagreement` was written to catch on `thermal_seam`.

    ``h`` is then RE-MEASURED at the marched wall and both values are reported.
    One update, not a fixed point: the fixed point is the composition's job, and
    solving it here would be hand-written coupling code in the setup of a case
    study about not having any.

    Returns ``(shell, gas, provenance)`` with the provenance dict carrying every
    number the probe state has to quote.
    """
    t0 = time.perf_counter()
    gas = ChamberGasAgent(dt=dt_gas, flux_convention=flux_convention)
    shell = ShellAgent(dt=dt_shell, flux_convention=flux_convention,
                       expose_elliptic=expose_elliptic)

    h_cold = gas_h_on_shell_seam(gas, shell, T_SHELL_COLD)
    marched = shell.march(T_CHAMBER, h_cold, burn_time=burn_time, record=record_curve)
    T_wall = marched["T_wall_mean_K"]
    h_hot = gas_h_on_shell_seam(gas, shell, T_wall)
    gas.T_wall_base = T_wall
    elapsed = time.perf_counter() - t0

    prov = {
        "build_repo_commit": "0a407b7",
        "p_chamber_Pa": P_CHAMBER,
        "T_chamber_K": T_CHAMBER,
        "gamma": GAMMA_GAS,
        "R_gas": R_GAS,
        "h_cold_mean_W_m2K": float(h_cold.mean()),
        "h_cold_min_W_m2K": float(h_cold.min()),
        "h_cold_max_W_m2K": float(h_cold.max()),
        "h_hot_mean_W_m2K": float(h_hot.mean()),
        "h_hot_over_cold": float(h_hot.mean() / h_cold.mean()),
        "T_shell_cold_K": T_SHELL_COLD,
        "dt_gas_s": gas.dt,
        "dt_shell_s": shell.dt,
        "gas_cells": int(np.prod(gas._blk.shape)),
        "shell_cells": int(np.prod(shell._blk.shape)),
        "n_seam_gas": gas.n_seam,
        "n_seam_shell": shell.n_seam,
        "m_eff_gas": modes_for(gas.n_seam),
        "m_eff_shell": modes_for(shell.n_seam),
        "dim_M": min(modes_for(gas.n_seam), modes_for(shell.n_seam)),
        "flux_convention": flux_convention,
        "setup_seconds": elapsed,
        "setup_gas_calls": gas.solver_calls,
        "setup_shell_calls": shell.solver_calls,
    }
    prov.update({k: v for k, v in marched.items() if k != "curve"})
    if record_curve:
        prov["curve"] = marched["curve"]
    return shell, gas, prov


def probe_state_string(prov: dict) -> str:
    """The one-line probe state every number measured here must carry."""
    return (
        "rocket b-c, build repo {commit}: chamber p_c={p:.3g} Pa, T_c={T:.1f} K, "
        "gamma={g}, R={R}; shell wall MARCHED {bt:.2f} s of burn to {Tw:.4f} K "
        "mean ({Tmx:.4f} K max) under h={h:.4f} W/(m^2 K); dt_gas={dtg:.3g} s, "
        "dt_shell={dts:.3g} s; dim M={M}; validity {v}"
    ).format(commit=prov["build_repo_commit"], p=prov["p_chamber_Pa"],
             T=prov["T_chamber_K"], g=prov["gamma"], R=prov["R_gas"],
             bt=prov["burn_time_s"], Tw=prov["T_wall_mean_K"],
             Tmx=prov["T_wall_max_K"], h=prov["h_cold_mean_W_m2K"],
             dtg=prov["dt_gas_s"], dts=prov["dt_shell_s"], M=prov["dim_M"],
             v="holds" if prov["valid_at_base"] else "DECLINES")


# ---------------------------------------------------------------------------
# the two-agent graph, so the seam is assembled by the COMPILER's own route
# ---------------------------------------------------------------------------


def build_bc_graph(shell: ShellAgent, gas: ChamberGasAgent,
                   shell_half: ResponseHalf = ResponseHalf.FLOW,
                   gas_half: ResponseHalf = ResponseHalf.FLOW,
                   shell_elliptic: EllipticSubsolve | None = None,
                   flux_matching=None, measured=None):
    """The b-c THERM seam alone, as a two-agent `CaseGraph`.

    Going through a graph rather than calling `probe_block` twice by hand is
    `w87_referent_cost_and_lag._seam_operator`'s pattern, and for its reason: the
    space and the prolongations are then the ones a COMPILE would use, rather
    than a second construction that could quietly disagree with it.

    ``shell_half`` and ``gas_half`` are parameters so R0's gate-3 control can
    flip one, then both, and watch L3/C9 decide.  They default to the truth.
    """
    from ..graph import Agent, CaseGraph, Connection, Decomposition, FluxMatching

    m = min(modes_for(shell.n_seam), modes_for(gas.n_seam))
    agents = [
        Agent("b", chamber_gas_capabilities(gas, m_eff=m, response_half=gas_half),
              domain="combustion chamber, z in [0.12, 0.40]"),
        Agent("c", shell_capabilities(shell, m_eff=m, response_half=shell_half,
                                      elliptic=shell_elliptic),
              domain="airframe upper panel, z in [-4.0, 0.70]", role="structure"),
    ]
    conn = Connection(
        seam_id="b-c:THERM",
        a=("b", "c:THERM"),
        b=("c", "b:THERM"),
        port_type=PortType.THERM,
        geometrically_coincident=True,
        derive_space=True,
        # n_0(Gamma) = 0, for `thermal_seam`'s reason: a conjugate-heat-transfer
        # seam constrains nothing the way incompressibility constrains a MECH
        # trace, so a uniform temperature shift produces a uniform flux change
        # and no direction of the trace space is invisible to the response.
        expected_null_dim=0,
        # **W303, measured 2026-09-17.** Both sides return a heat flux that is
        # POSITIVE FOR HEAT FLOWING GAS -> SHELL: the gas's `_wall_flux` gives
        # `k_gas (T_i - T_w)/dn` and the shell's Robin term gives
        # `h (T_gas - T_face)`. Read off the meshes, +y at this wall IS agent
        # `b`'s outward normal (its jmax face normal is [0, +1]) and is AGAINST
        # the shell's inner-face normal, which points back into the gas. So both
        # sides report against `b`'s outward normal, which is exactly what this
        # field names -- and `assemble_seam`'s own W138 note says a seam of that
        # shape needs the DIFFERENCE, not the Steklov-Poincare sum.
        #
        # It was undeclared through Tier 76, so both blocks entered with +1 and
        # the sum was assembled. Measured at the consistent base, declaring it
        # takes sigma_min from 0.001315 to 0.12319 -- a factor of 94 -- and
        # turns a MIXED spectrum, [-0.0055601, +0.18727], into a ONE-SIGNED one,
        # [-0.45753, -0.12318]. E7 still reports a defect, and that is W306: it
        # tests lambda_min > 0 and never looks at lambda_max, so it cannot tell a
        # global sign from an amplified mode.
        effort_normal="b",
        note="conjugate heat transfer, chamber gas against the airframe panel. "
             "NON-CONFORMING: 112 gas cells against 14 shell cells over the same "
             "0.28 m of z, mapped through the declared prolongation pair",
    )
    if flux_matching is None:
        # Across one clock the two conventions coincide exactly; across 50:1 R9
        # requires the integral. This graph is the 50:1 one whenever the clocks
        # are the YAML's, so the default follows the clocks rather than a taste.
        ratio = max(shell.dt, gas.dt) / max(1e-300, min(shell.dt, gas.dt))
        flux_matching = (FluxMatching.TIME_INTEGRATED if ratio > 1.5
                         else FluxMatching.POINTWISE)
    return CaseGraph(
        name="rocket-bc-therm",
        agents=agents,
        connections=[conn],
        # The gas and the shell are different MATERIALS sharing a surface, not a
        # region: no cell belongs to both, so there is no partition of unity to
        # declare. `thermal_seam` says the same about the same kind of seam.
        decomposition=Decomposition.NON_OVERLAPPING,
        macro_dt=max(shell.dt, gas.dt),
        flux_matching=flux_matching,
        measured=measured,
        note="R0 rungs 1 and 2: the first rocket seam with real physics on both "
             "sides. compressible2d against thermostruct2d, build repo 0a407b7, "
             "both unmodified, on grid.build_blocks' own meshes",
    )


def seam_operator(graph, budget=None, probe_state: str = "unspecified",
                  seam_base=None):
    """The seam's probed S and its transfer, by the compiler's own route."""
    from ..compiler import _Context, _derive_transfer          # noqa: PLC2701
    from ..envelope import EnvelopeStamp
    from ..holes import HoleLedger
    from ..probe import ProbeBudget, assemble_seam
    from ..verdict import DecisionRecord

    budget = budget or ProbeBudget()
    ctx = _Context(graph=graph, budget=None, probe_budget=budget, references={},
                   record=DecisionRecord(), stamp=EnvelopeStamp(), holes=HoleLedger(),
                   probe_state=probe_state, depth=0)
    conn = graph.connections[0]
    tr = _derive_transfer(ctx, conn)
    op = assemble_seam(graph, conn, tr, budget, {},
                       expected_null_dim=conn.expected_null_dim,
                       probe_state=probe_state, seam_base=seam_base)
    return op, tr


#: The cadence at which the shell's Robin coefficient is measured when a caller
#: does not say.  **Justified by measurement, not by cost alone**: h at the b-c
#: wall is 187.79 W/(m^2 K) at dt = 1e-6 and 183.11 at dt = 1e-4, a spread of
#: 2.5% over a 100x cadence range, while the YAML's own 1e-3 s clock costs ~3350
#: CFL sub-steps and about a minute a call on this machine.  The number is
#: recorded so a reader can price the choice instead of discovering it.
DT_H_MEASURE = 1.0e-5


def bc_dim_M() -> int:
    """``dim M = min_i m_i_eff`` for the b-c seam, from the two meshes.

    Both sides must be DECLARED at this one number.  Declaring each side at its
    own ``m_i_eff`` leaves the prolongation matrices different widths, and
    `L3/C2/C3/C6` then refuses the seam for want of a consistent pair -- which is
    the compiler catching a real inconsistency, and is how this function came to
    exist.
    """
    return min(modes_for(seam_cells_bc(shell_panel()).size),
               modes_for(chamber_block().shape[0]))


# ---------------------------------------------------------------------------
# R0 rungs 3-7: the remaining gas agents, and the two KINDS of port they have
# ---------------------------------------------------------------------------

#: **The undeclared interface this tier found, recorded where the code is.**
#:
#: The airframe `c` spans ``shell_z = [-4.0, 0.70]`` and agent `e`, the nozzle,
#: spans ``[0.40, 0.70]``.  At z = 0.55 `domains.contains` puts the nozzle
#: interior in `e` and the shell thickness in `c`: **they share a boundary and no
#: edge declares it.**  `wall_b_c` stops at the throat and `wall_c_d` is the
#: OUTER wall, so no interface curve runs along the diverging nozzle's inner
#: surface at all.
#:
#: `contours.plane_d_g`'s own docstring already records a SECOND one -- *"a real,
#: currently UNDECLARED d-f interface"* over the strip between the exit radius
#: and the plume half-width.  So this is not a one-off: **`domains.verify`
#: checks that the agents PARTITION the box and that every declared edge lies on
#: both its agents' boundaries, and nothing checks the converse** -- that every
#: shared boundary has an edge.  Two are missing.
#: **W311 and W317.**  Interfaces two agents physically share that no edge in
#: `config/atlas_0_1.yaml` declares.  Measured on the shell's own inner face
#: (232 cells, 4.7254 m of arc), taking each cell by its midpoint: the
#: ENGINE-SIDE wall -- the part with gas on the other side, z in [0, 0.70] -- is
#: 35 cells and 0.734463 m, and the one DECLARED window is 14 cells and
#: 0.299018 m of it.  So **the declared conjugate seam is 40.7% of the wall that
#: is actually conjugate**, and the other 59.3% is in here.
UNDECLARED_INTERFACES = {
    "a-c": "the injector/combustion zone's lateral walls below the chamber "
           "window, 6 cells and 0.121552 m. **Named by the build repo's own "
           "source**: generate.py::_wall_T's docstring says `a`'s lateral walls "
           "are an UNDECLARED a-c interface, and that is the stated reason it "
           "hands every gas agent one scalar wall temperature instead of the "
           "per-station map it already has",
    "e-c": "the diverging nozzle's inner wall, z in [z_throat, z_exit]. `e` and "
           "`c` share it and no edge declares it; wall_b_c stops at the throat "
           "and wall_c_d is the outer wall. 15 cells and 0.313893 m, and W311 "
           "measured it as the hottest part of the engine",
    "d-f": "the strip exit_halfheight < |y| <= plume_halfwidth at z = z_exit, "
           "recorded by contours.plane_d_g's own docstring",
}

#: Arc length in metres on the shell's inner face, cells taken by midpoint.
#: Measured by `scripts/w314_rocket_multirate_defect.py`.  Only `b-c` is
#: declared, and it is the smaller half.
ENGINE_WALL_M = {"a-c": 0.121552, "b-c": 0.299018, "e-c": 0.313893}
ENGINE_WALL_TOTAL_M = 0.734463
SHELL_INNER_FACE_M = 4.725411

#: Which block face each agent's declared neighbours sit on, and whether that
#: face is a WALL (the trace is a wall temperature, the response a wall flux) or
#: a PLANE the flow crosses (the trace is a state, the response a flux through
#: it).  Read off `config/atlas_0_1.yaml`'s geometry and `contours.py`'s
#: interface constructors, not chosen.
#:
#: The two kinds need different channels and it is not a detail: `wall_noslip`
#: clamps its ghost temperature and `prescribed` does not, which is exactly the
#: difference W301 is about.
AGENT_FACES: dict[str, dict[str, tuple[str, str]]] = {
    #        neighbour: (block side, kind)
    "a": {"b": ("imax", "plane")},
    "b": {"a": ("imin", "plane"), "e": ("imax", "plane"), "c": ("jmax", "wall")},
    "e": {"b": ("imin", "plane"), "f": ("imax", "plane"), "c": ("jmax", "wall")},
    "d": {"c": ("jmin", "wall"), "g": ("imax", "plane")},
    "f": {"e": ("imin", "plane"), "g": ("jmax", "plane")},
    "g": {"d": ("imin", "plane"), "f": ("hole", "plane")},
}

#: `data/generate.py` line 129: the ENGINE agents and the plume get `gas_cfg`
#: (combustion products); `d` and `g` get `air_cfg`.  Different parameters of the
#: same equations, which is why Tier 76 gave them one governing family.
ENGINE_AGENTS = ("a", "b", "e", "f")

#: US Standard 1976 at 5 km, the mid-point of the ascent this PoC marches.  Read
#: from the build repo's own `atmosphere` module rather than tabulated here.
ALTITUDE_M = 5000.0


@lru_cache(maxsize=1)
def ambient_state() -> tuple[float, float, float]:
    """(rho, p, T) at `ALTITUDE_M`, from the build repo's own atmosphere.

    `atmosphere.properties` returns ``(T, p, rho, a)`` -- temperature FIRST.
    Unpacking it as ``(rho, p, T)`` is the obvious mistake and this function
    made it on its first run, reporting a density of 255.7 and a temperature of
    0.74.  The order is taken from the source rather than assumed, and the
    assertion below is what would catch it next time.
    """
    load_solvers()
    atm = importlib.import_module("atlas_build_solvers.solvers.atmosphere")
    T, p, rho, _a = atm.properties(ALTITUDE_M)
    T, p, rho = float(T), float(p), float(rho)
    assert 200.0 < T < 320.0 and 1.0e4 < p < 1.1e5 and 0.1 < rho < 1.5, (T, p, rho)
    return rho, p, T


#: The freestream Mach the external agents see. 0.8 at 5 km is a representative
#: point of the ascent `trajectory.py` marches; it is a STATE, not a result, and
#: every number probed against it carries it in `probe_state`.
M_INF = 0.8


@dataclass
class GasAgent:
    """Any of the six `compressible2d` agents, built from the config.

    **One class rather than six, because the six differ only in declarations the
    config already carries** -- mesh, z range, gas constants, clock -- and a
    per-agent class would be six places for the same bug.  What genuinely differs
    is the KIND of each face, and that is `AGENT_FACES`.

    Two kinds of port, and the difference is the whole reason `b`'s wall and
    `e`'s throat plane could not share a code path:

    ``wall``   `BC("wall_noslip", {"T_wall": ...})`.  The trace is a wall
               temperature and the response the conduction-limited wall flux --
               `thermal_seam.GasAgent`'s channel, and the one W301 found is
               SATURATED below ``T_wall = (T_i + 20)/2``.
    ``plane``  `BC("prescribed", {"state": ...})`.  The flow CROSSES it, so the
               trace is a total enthalpy and the response the mass flux through
               the face -- ADVEC's declared bond ``(h0, rho u.n)``.  This channel
               writes the ghost state directly and has **no clamp**, which is the
               cross-check that W301 is about `wall_noslip` and not about
               `compressible2d`.
    """

    agent_id: str
    dt: float | None = None
    _blk: Any = field(default=None, repr=False)
    _sol_cfg: Any = field(default=None, repr=False)
    _U0: Any = field(default=None, repr=False)
    _calls: int = 0
    _substeps: int = 0

    def __post_init__(self) -> None:
        C2, _TS, TH, _GR = load_solvers()
        cfg = rocket_config()
        if self.dt is None:
            self.dt = float(cfg.dt_model[self.agent_id])
        # `d` is a two-panel map; the UPPER panel is taken, for the same symmetry
        # reason `c` is, and `generate.py` indexes its panels the same way.
        blocks = agent_blocks(self.agent_id)
        self._blk = blocks[-1] if len(blocks) > 1 else blocks[0]
        engine = self.agent_id in ENGINE_AGENTS
        self._sol_cfg = C2.GasConfig(gamma=GAMMA_GAS if engine else 1.4,
                                     R=R_GAS if engine else 287.0)
        nz, ny = self._blk.shape
        W = np.zeros((nz, ny, 4))
        if engine:
            W[..., 0] = P_CHAMBER / (R_GAS * T_CHAMBER)
            W[..., 1] = U_CHAMBER
            W[..., 3] = P_CHAMBER
        else:
            rho, p, T = ambient_state()
            W[..., 0] = rho
            W[..., 1] = M_INF * float(np.sqrt(1.4 * 287.0 * T))
            W[..., 3] = p
        self._U0 = TH.prim_to_cons(W, self._sol_cfg.gamma)
        self._C2, self._TH = C2, TH

    # -- geometry ----------------------------------------------------------
    def face(self, neighbour: str) -> tuple[str, str]:
        try:
            return AGENT_FACES[self.agent_id][neighbour]
        except KeyError:
            raise KeyError(
                f"agent {self.agent_id!r} declares no face toward {neighbour!r}; "
                f"it has {sorted(AGENT_FACES[self.agent_id])}. If this is the "
                f"nozzle wall, see UNDECLARED_INTERFACES"
            ) from None

    def hole_band(self):
        """(j_lo, j_hi) of the blanked band, or None. Agent `g` carries the plume
        as a hole rather than as a face, so its `f` port is a different channel
        again -- `Compressible2D.hole_state`, which the solver's own comment calls
        *"set by the coupler"* and which nothing in this vault had ever set."""
        b = self._blk.blanked
        if not b.any():
            return None
        js = np.nonzero(b.any(axis=0))[0]
        return int(js.min()), int(js.max())

    def n_face(self, neighbour: str) -> int:
        side, _kind = self.face(neighbour)
        nz, ny = self._blk.shape
        if side == "hole":
            return nz
        return ny if side in ("imin", "imax") else nz

    def face_weights(self, neighbour: str) -> np.ndarray:
        """Arc lengths of that face's cells -- the V-space measure."""
        side, _kind = self.face(neighbour)
        blk = self._blk
        if side == "hole":
            band = self.hole_band()
            return np.asarray(blk.a_j[:, band[0]], dtype=float)
        if side == "imin":
            return np.asarray(blk.a_i[0], dtype=float)
        if side == "imax":
            return np.asarray(blk.a_i[-1], dtype=float)
        if side == "jmin":
            return np.asarray(blk.a_j[:, 0], dtype=float)
        return np.asarray(blk.a_j[:, -1], dtype=float)

    # -- the solve ---------------------------------------------------------
    def _bcs(self, neighbour: str, trace: np.ndarray) -> dict:
        """Every side's BC, with the PORT's side carrying the imposed trace.

        The non-port sides are held at the agent's own operating condition. They
        are a modelling choice and they are reported in `probe_state`: a probe is
        a derivative with respect to ONE face, and what the other three do sets
        the point it is taken at.
        """
        C2 = self._C2
        cfg = self._sol_cfg
        side, kind = self.face(neighbour)
        nz, ny = self._blk.shape
        engine = self.agent_id in ENGINE_AGENTS
        if engine:
            rho0, u0, p0 = P_CHAMBER / (R_GAS * T_CHAMBER), U_CHAMBER, P_CHAMBER
            T_wall_bg = T_SHELL_COLD
        else:
            rho_a, p_a, T_a = ambient_state()
            rho0, u0, p0 = rho_a, M_INF * float(np.sqrt(1.4 * 287.0 * T_a)), p_a
            T_wall_bg = T_a
        free = C2.BC("freestream", {"prim": np.array([rho0, u0, 0.0, p0])})
        out = {
            "imin": free,
            "imax": C2.BC("outflow", {"p_inf": p0}),
            "jmin": C2.BC("wall_noslip", {"T_wall": np.full(nz, T_wall_bg)}),
            "jmax": C2.BC("wall_noslip", {"T_wall": np.full(nz, T_wall_bg)}),
        }
        # the external agents' outboard face is the farfield, not a wall
        if not engine:
            out["jmax"] = free
        if side == "hole":
            return out                      # the hole is not a face; see _advance
        if kind == "wall":
            out[side] = C2.BC("wall_noslip",
                              {"T_wall": np.asarray(trace, float).reshape(nz)})
        else:
            # ADVEC's effort is a total enthalpy. Hold (p, u, v) at the
            # operating point and let h0 set the density, so the trace moves the
            # ONE declared variable and nothing else: h0 = gamma/(gamma-1) p/rho
            # + |u|^2/2 solved for rho.
            h0 = np.asarray(trace, float).reshape(-1)
            n = ny if side in ("imin", "imax") else nz
            h0 = h0.reshape(n)
            ke = 0.5 * u0 * u0
            cp_over = cfg.gamma / (cfg.gamma - 1.0)
            rho = cp_over * p0 / np.maximum(h0 - ke, 1.0)
            W = np.zeros((n, 4))
            W[:, 0], W[:, 1], W[:, 3] = rho, u0, p0
            U = self._TH.prim_to_cons(W, cfg.gamma)
            state = U[None, :, :] if side in ("imin", "imax") else U[:, None, :]
            out[side] = C2.BC("prescribed", {"state": state})
        return out

    def _state_from_h0(self, h0) -> np.ndarray:
        """A conservative state carrying the declared effort and nothing else.

        (p, u, v) are held at the operating point and h0 sets the density, so the
        trace moves the ONE declared variable:
        ``h0 = gamma/(gamma-1) p/rho + |u|^2/2`` solved for rho.
        """
        cfg = self._sol_cfg
        engine = self.agent_id in ENGINE_AGENTS
        if engine:
            p0, u0 = P_CHAMBER, U_CHAMBER
        else:
            rho_a, p_a, T_a = ambient_state()
            p0, u0 = p_a, M_INF * float(np.sqrt(1.4 * 287.0 * T_a))
            del rho_a
        h0 = np.atleast_1d(np.asarray(h0, dtype=float))
        rho = cfg.gamma / (cfg.gamma - 1.0) * p0 / np.maximum(h0 - 0.5 * u0 * u0, 1.0)
        W = np.zeros((h0.size, 4))
        W[:, 0], W[:, 1], W[:, 3] = rho, u0, p0
        return self._TH.prim_to_cons(W, cfg.gamma)

    def _advance(self, neighbour: str, trace: np.ndarray):
        side, _kind = self.face(neighbour)
        sol = self._C2.Compressible2D(self._blk, self._sol_cfg,
                                      bcs=self._bcs(neighbour, trace))
        if side == "hole":
            # the band's ghost is ONE state, so the trace is reduced to the mean
            # -- and that is a real loss of resolution, declared rather than
            # hidden: `effective_resolution` on this port is 1.
            sol.hole_state = self._state_from_h0(
                float(np.mean(np.asarray(trace, dtype=float))))[0]
        U, n = sol.advance(self._U0, self.dt)
        self._calls += 1
        self._substeps += int(n)
        return U

    def wall_h(self, T_wall: float | None = None, neighbour: str = "c") -> np.ndarray:
        """The conduction-limited ``h`` the shell's Robin channel needs.

        `generate.py::_step_structure` takes this from the gas and hands it to
        `step_thermal`. It is NOT on a port and is not certified: it is a
        coefficient of the SHELL's own channel, computed here so the two sides
        are set up at one operating point rather than at two.
        """
        side, kind = self.face(neighbour)
        if kind != "wall":
            raise ValueError(
                f"agent {self.agent_id!r} face toward {neighbour!r} is a "
                f"{kind!r}, not a wall; there is no film coefficient on it"
            )
        nz, _ny = self._blk.shape
        Tw = (T_SHELL_COLD if self.agent_id in ENGINE_AGENTS
              else ambient_state()[2]) if T_wall is None else float(T_wall)
        U = self._advance(neighbour, np.full(nz, Tw))
        TH, cfg, blk = self._TH, self._sol_cfg, self._blk
        W = TH.cons_to_prim(U, cfg.gamma)
        T = W[..., 3] / (W[..., 0] * cfg.R)
        j = 0 if side == "jmin" else -1
        dn = 0.5 * blk.vol[:, j] / np.maximum(blk.a_j[:, j], 1e-30)
        mu = TH.sutherland(T[:, j], cfg.mu_ref, cfg.T_mu_ref, cfg.sutherland_S)
        return mu * cfg.cp / cfg.Pr / np.maximum(dn, 1e-9)

    def h0_operating(self) -> float:
        """The total enthalpy the plane ports are linearised about -- W74 again.

        ADVEC's effort has no absolute origin the way a temperature does, but it
        has no natural ZERO either: h0 = 0 is a gas with no thermal energy, which
        is not a state any of these agents is ever in.
        """
        cfg = self._sol_cfg
        engine = self.agent_id in ENGINE_AGENTS
        if engine:
            p, rho, u = P_CHAMBER, P_CHAMBER / (R_GAS * T_CHAMBER), U_CHAMBER
        else:
            rho, p, T = ambient_state()
            u = M_INF * float(np.sqrt(1.4 * 287.0 * T))
        return float(cfg.gamma / (cfg.gamma - 1.0) * p / rho + 0.5 * u * u)

    def base_trace(self, neighbour: str) -> np.ndarray:
        side, kind = self.face(neighbour)
        n = self.n_face(neighbour)
        if kind == "wall":
            return np.full(n, T_SHELL_COLD if self.agent_id in ENGINE_AGENTS
                           else ambient_state()[2])
        return np.full(n, self.h0_operating())

    def respond(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        """(the declared effort on one face) -> (the declared flow on that face).

        ``wall``   (T_wall) -> q_n/T, `generate.py::_wall_flux`'s own quantity.
        ``plane``  (h0)     -> rho u.n, ADVEC's declared conjugate flow.
        """
        neighbour = _neighbour_of(port_name)
        side, kind = self.face(neighbour)
        U = self._advance(neighbour, trace)
        TH, cfg, blk = self._TH, self._sol_cfg, self._blk
        W = TH.cons_to_prim(U, cfg.gamma)
        if kind == "wall":
            T = W[..., 3] / (W[..., 0] * cfg.R)
            j = 0 if side == "jmin" else -1
            dn = 0.5 * blk.vol[:, j] / np.maximum(blk.a_j[:, j], 1e-30)
            mu = TH.sutherland(T[:, j], cfg.mu_ref, cfg.T_mu_ref, cfg.sutherland_S)
            k_gas = mu * cfg.cp / cfg.Pr
            Tw = np.asarray(trace, float).reshape(T[:, j].shape)
            q = k_gas * (T[:, j] - Tw) / np.maximum(dn, 1e-9)
            return q / np.maximum(0.5 * (T[:, j] + Tw), 1.0)
        if side == "hole":
            band = self.hole_band()
            j = band[0]
            rho, u, v = W[:, j, 0], W[:, j, 1], W[:, j, 2]
            nrm = blk.n_j[:, j]
            return rho * (u * nrm[..., 0] + v * nrm[..., 1])
        # a plane: the mass flux through it, against the face's own normal
        if side in ("imin", "imax"):
            i = 0 if side == "imin" else -1
            rho, u, v = W[i, :, 0], W[i, :, 1], W[i, :, 2]
            nrm = blk.n_i[0 if side == "imin" else -1]
        else:
            j = 0 if side == "jmin" else -1
            rho, u, v = W[:, j, 0], W[:, j, 1], W[:, j, 2]
            nrm = blk.n_j[:, 0 if side == "jmin" else -1]
        return rho * (u * nrm[..., 0] + v * nrm[..., 1])

    def respond_integrated(self, port_name: str, trace: np.ndarray,
                           n_substeps: int) -> np.ndarray:
        """The same flow, averaged over ``n_substeps`` of this agent's own clock."""
        n = max(1, int(n_substeps))
        acc = None
        saved = self._U0
        try:
            for _ in range(n):
                out = np.asarray(self.respond(port_name, trace), float)
                acc = out if acc is None else acc + out
                self._U0 = self._advance(_neighbour_of(port_name), trace)
        finally:
            self._U0 = saved
        return acc / n

    # -- the record's callables --------------------------------------------
    def storage(self, state=None) -> float:
        U = self._U0 if state is None else state
        return float(np.sum(U[..., 3] * self._blk.vol))

    def validity(self, state=None, cond=None) -> bool:
        U = self._U0 if state is None else np.asarray(state)
        W = self._TH.cons_to_prim(U, self._sol_cfg.gamma)
        return bool(np.all(W[..., 0] > 0.0) and np.all(W[..., 3] > 0.0))

    @property
    def solver_calls(self) -> int:
        return self._calls

    @property
    def solver_substeps(self) -> int:
        return self._substeps


def _neighbour_of(port_name: str) -> str:
    """``"c:THERM"`` -> ``"c"``. The rocket names a port for the agent across it."""
    return port_name.split(":")[0]


#: **W309, measured 2026-09-17.  `grid.Block`'s face normals are oriented by
#: INDEX, not outward -- so every gas-gas seam in this graph needs
#: `effort_normal`, and Tier 77's b-c was not a special case.**
#:
#: Measured directly: for every gas block, ``n_i[0]`` and ``n_i[-1]`` both point
#: ``+z``, and ``n_j[:, 0]`` and ``n_j[:, -1]`` both point ``+y`` (with the
#: body-fitted tilt).  They are ``[e_y, -e_x]`` and ``[-e_y, e_x]`` of the edge
#: vectors, which is a consistent orientation and NOT an outward normal.  So a
#: seam between two blocks has both sides reporting their flux against the same
#: physical direction -- exactly the shape `assemble_seam`'s W138 note describes,
#: where the well-posed condition is the DIFFERENCE and the Steklov-Poincare sum
#: is a matrix no scheme differentiates.
#:
#: The rule that follows: **`effort_normal` names the agent for which the seam is
#: its `imax` or `jmax` face**, because that is the one side where the
#: index-oriented normal coincides with the agent's own outward normal.
#:
#: **And it reproduces the one seam that was measured independently.**  Tier 77
#: derived ``b-c -> "b"`` from the geometry of the shell's inner face and the
#: chamber's jmax wall, before any of this was looked at; the rule gives the same
#: answer.  That is the cross-check that makes it a rule rather than a pattern.
#:
#: `c-d` is the one edge the rule cannot reach, because `c` is a `ThermoStruct2D`
#: and has no Block: agent `d`'s face toward the shell is its **jmin**, whose
#: index normal ``+y`` points away from the vehicle and is therefore the SHELL's
#: outward normal, so the named agent is `c`.
EFFORT_NORMAL_ROCKET = {
    "a-b": "a",   # a's face to b is its imax
    "e-b": "b",   # b's face to e is its imax
    "b-c": "b",   # b's wall to c is its jmax -- and Tier 77 measured this one
    "c-d": "c",   # d's face to c is its jmin; +y is the shell's outward normal
    "e-f": "e",   # e's face to f is its imax
    "d-g": "d",   # d's face to g is its imax
    "g-f": "f",   # f's face to g is its jmax; g carries it as a blanked hole
}


#: **The motion class each port really carries, from `rocket.AGENTS`.**
#:
#: The first version of `build_rocket_real` declared every port STATIC, and the
#: measurement caught it immediately: the nine `L2/InterfaceMotion` refusals
#: vanished. **That is the "declare it away" failure exactly** -- the combustion
#: front and the plume boundary are the one genuine research hole in this graph,
#: and a builder that quietly makes them static has removed the product rather
#: than earned it. They are carried here so the new graph refuses for the same
#: reasons the old one did.
PORT_MOTION = {
    ("a", "b"): MotionClass.SOLUTION_DEPENDENT,   # the combustion front moves
    ("f", "e"): MotionClass.PRESCRIBED,           # the plume boundary moves
    ("f", "g"): MotionClass.PRESCRIBED,
}


def gas_capabilities(expert: GasAgent, neighbour: str, port_type: PortType,
                     m_eff: int | None = None,
                     response_half: ResponseHalf = ResponseHalf.FLOW,
                     ) -> ExpertCapabilities:
    """One agent's record for ONE declared face.

    The rocket's graph gives each agent several faces and several port types per
    face; this builds the record for one of them, so a caller can assemble
    exactly the ports a seam needs and no more.
    """
    side, kind = expert.face(neighbour)
    n = expert.n_face(neighbour)
    m = modes_for(n) if m_eff is None else int(m_eff)
    w = expert.face_weights(neighbour)
    scales = dict(THERM_SCALES) if port_type is PortType.THERM else dict(ADVEC_SCALES_R)
    name = f"{neighbour}:{port_type.value}"
    dt_cfl = 2.99e-7 if expert.agent_id in ENGINE_AGENTS else 1.6e-6
    return ExpertCapabilities(
        expert_id=expert.agent_id,
        ports=[port_decl(
            name=name, port_type=port_type,
            geometry=f"{expert.agent_id}.{side} ({kind}), {n} cells",
            direction=Direction.BIDIRECTIONAL,
            nondim=scales,
            passengers=("h0",) if port_type is PortType.ADVEC else (),
            effective_resolution=m,
            motion_class=PORT_MOTION.get((expert.agent_id, neighbour),
                                         MotionClass.STATIC),
            response_half=response_half,
            prolongation=face_prolongation(
                expert.agent_id, name, w, m,
                f"{m}-mode real Fourier basis on {n} non-uniform face cells"),
            note=("isothermal no-slip wall; trace T_wall, response the "
                  "conduction-limited flux generate.py exchanges"
                  if kind == "wall" else
                  "a plane the flow CROSSES: BC('prescribed') writes the ghost "
                  "state directly, with NO clamp -- the trace is h0 and the "
                  "response the mass flux through the face"),
        )],
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=True,
        elliptic_subsolve=EllipticSubsolve.NONE,
        time_discretization=TimeDiscretization.EXPLICIT,
        stencil_radius=2,
        substeps_per_macro_step=max(1, int(np.ceil(expert.dt / dt_cfl))),
        differentiable=Differentiable.NONE,
        dt_native=expert.dt,
        storage=expert.storage,
        equivariances=(),
        validity=expert.validity,
        governing_family=("reacting-compressible-flow" if expert.agent_id == "a"
                          else "compressible-navier-stokes-2d"),
        lambda_ref=f"compressible2d.Compressible2D, agent {expert.agent_id}'s own "
                   "block and discretization",
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash="compressible2d/2-hllc-minmod-ssprk2",
        boundary_response=expert.respond,
        boundary_response_integrated=expert.respond_integrated,
        probe_base=lambda p, _e=expert: _e.base_trace(_neighbour_of(p)),
        reproducibility_floor=float(np.finfo(float).eps),
        deterministic=True,
        note=f"solvers/compressible2d.py, build repo 0a407b7, unmodified; mesh "
             f"from grid.build_blocks; {kind} port on {side}",
    )


#: ADVEC pairs (h0, rho u.n) in J/kg and kg/(m^2 s): s_e s_f = s_P in W/m^2.
#: Sized on the chamber's own operating point rather than on the fixture's
#: numbers, which were declared before any physics existed to size them against.
ADVEC_SCALES_R = {
    "enthalpy": 5.6e6, "mass_flux": 1.0e2, "power_area": 5.6e8,
    "h0_effort": 5.6e6, "h0_flow": 1.0e2, "h0_power": 5.6e8,
}


#: **The port type each declared edge is probed on, and why it is that one.**
#:
#: `rocket.py` declares MECH, THERM and ADVEC on most edges.  This names the ONE
#: whose conjugate bond the wired channel actually carries, so a number is
#: measured where it means something rather than on every type indiscriminately.
#:
#: The `g-f` row is the finding.  `rocket.py` declares ADVEC on it; the config's
#: own `edge_list` says ``(heat, fluid)``; and **ADVEC's declared flow is
#: identically zero there** -- the shear layer's normal is exactly ``+y`` and the
#: flow is ``+z``, so ``rho u.n == 0`` for any trace.  Measured: perturbing h0 by
#: 1000 J/kg moves the response by 4.2e-10, the solver's own transverse velocity
#: noise.  That is `CASE-STUDY-GUIDE` mistake 6 -- an interface problem that is
#: EMPTY rather than hard -- and it is a declaration error, not physics.
SEAM_PORT_TYPE = {
    "a-b": PortType.ADVEC,   # the reaction zone's exit plane; mass crosses
    "e-b": PortType.ADVEC,   # the throat plane
    "b-c": PortType.THERM,   # the chamber wall
    "c-d": PortType.THERM,   # the airframe's outer face
    "e-f": PortType.ADVEC,   # the nozzle exit plane
    "d-g": PortType.ADVEC,   # the atmosphere front/wake plane
    "g-f": PortType.ADVEC,   # DECLARED, and measured EMPTY -- see above
}

#: Which agent sits on each end of each declared edge, from `rocket.py`'s EDGES.
SEAM_SIDES = {
    "a-b": ("a", "b"), "e-b": ("e", "b"), "b-c": ("b", "c"),
    "c-d": ("c", "d"), "e-f": ("e", "f"), "d-g": ("d", "g"),
    "g-f": ("g", "f"),
}


def make_rocket_experts(dt_scale: float = 1.0e-3,
                        burn_time: float = T_PROBE_BURN) -> dict[str, Any]:
    """All seven agents, at a stated probe cadence.

    ``dt_scale`` multiplies each agent's declared clock.  It is a DECLARATION and
    it is reported: the chamber alone is 3310 CFL sub-steps at its own 1 ms, and
    Tier 76 measured beta moving 0.3% over a 100x cadence range against an 84.75x
    cost, so the trade is priced rather than assumed.
    """
    cfg = rocket_config()
    experts: dict[str, Any] = {}
    for aid in ("a", "b", "e", "d", "f", "g"):
        experts[aid] = GasAgent(aid, dt=float(cfg.dt_model[aid]) * dt_scale)
    shell = ShellAgent(dt=float(cfg.dt_model["c"]))
    shell.march(T_CHAMBER, gas_h_on_shell_seam(experts["b"], shell, T_SHELL_COLD),
                burn_time=burn_time)
    experts["c"] = shell
    return experts


def rocket_port_name(agent_id: str, neighbour: str, seam_id: str) -> str:
    del agent_id
    return "%s:%s" % (neighbour, SEAM_PORT_TYPE[seam_id].value)


def _n_on(expert, agent_id: str, neighbour: str, seam_id: str) -> int:
    if agent_id == "c":
        return expert.n_port(rocket_port_name(agent_id, neighbour, seam_id))
    return expert.n_face(neighbour)


def _caps_for(expert, agent_id: str, ports):
    """One record carrying every port that agent owns, each at its seam's dim M."""
    if agent_id == "c":
        caps = shell_capabilities(expert, m_eff=ports[0][2])
        decls = []
        for neighbour, seam_id, m in ports:
            name = rocket_port_name(agent_id, neighbour, seam_id)
            n = expert.n_port(name)
            decls.append(port_decl(
                name=name, port_type=SEAM_PORT_TYPE[seam_id],
                geometry="shell %s, %d line elements"
                         % ("outer face" if neighbour == "d" else "inner window", n),
                direction=Direction.BIDIRECTIONAL, nondim=dict(THERM_SCALES),
                effective_resolution=m, motion_class=MotionClass.STATIC,
                response_half=ResponseHalf.FLOW,
                prolongation=face_prolongation(
                    agent_id, name, expert.port_weights(name), m,
                    "%d-mode real Fourier basis on %d face cells" % (m, n)),
                note="the agent's own Robin surface term",
            ))
        caps.ports = decls
        caps.probe_base = lambda p, _e=expert: _e.base_trace(p)
        return caps
    base, decls = None, []
    for neighbour, seam_id, m in ports:
        one = gas_capabilities(expert, neighbour, SEAM_PORT_TYPE[seam_id], m_eff=m)
        decls.append(one.ports[0])
        base = base or one
    base.ports = decls
    return base


def build_rocket_real(experts=None, dt_scale: float = 1.0e-3,
                      declare_effort_normal: bool = True,
                      m_cap: int | None = None,
                      measured=None):
    """The whole rocket graph with **all seven agents on real physics**.

    R0's completion.  Every agent's `boundary_response` steps a build-repo solver
    on the build repo's own mesh; nothing here is a seeded random matrix.

    One port type per edge (`SEAM_PORT_TYPE`), so a number is measured where its
    conjugate bond means something.  The MECH ports `rocket.py` also declares are
    NOT carried here -- see the module docstring.
    """
    from ..graph import Agent, CaseGraph, Connection, Decomposition, FluxMatching

    experts = experts or make_rocket_experts(dt_scale=dt_scale)
    # dim M per seam FIRST: both sides must be declared at the same number or
    # L3/C2/C3/C6 refuses the pair. Tier 77 learned that one the loud way.
    #: ``m_cap`` DECLARES a coarser interface space, which is not the same thing
    #: as capping the probe's budget.  `ProbeBudget.max_modes` shrinks the space
    #: and leaves the declared prolongation at its original width, so
    #: `Prolongation.prolong` raises ``dim mismatch``; a budget cannot narrow a
    #: space somebody else declared.  Declaring the narrower space builds the
    #: matching prolongation with it, and it is a DECLARATION -- reported, and
    #: the reason a number taken under it is not the number at full resolution.
    dims, owned = {}, {a: [] for a in experts}
    for seam_id, (x, y) in SEAM_SIDES.items():
        m = min(modes_for(_n_on(experts[x], x, y, seam_id)),
                modes_for(_n_on(experts[y], y, x, seam_id)))
        if m_cap is not None:
            m = min(m, int(m_cap))
        dims[seam_id] = m
        owned[x].append((y, seam_id, m))
        owned[y].append((x, seam_id, m))

    agents = [Agent(a, _caps_for(experts[a], a, owned[a]),
                    domain=a, role="structure" if a == "c" else "")
              for a in ("a", "b", "e", "c", "d", "f", "g")]
    connections = []
    for seam_id, (x, y) in SEAM_SIDES.items():
        connections.append(Connection(
            seam_id=seam_id,
            a=(x, rocket_port_name(x, y, seam_id)),
            b=(y, rocket_port_name(y, x, seam_id)),
            port_type=SEAM_PORT_TYPE[seam_id],
            geometrically_coincident=True,
            derive_space=True,
            # W309: Block face normals are index-oriented, so both sides report
            # against one shared direction on every gas-gas seam.
            effort_normal=(EFFORT_NORMAL_ROCKET[seam_id]
                           if declare_effort_normal else ""),
            # n_0(Gamma) = 0: a uniform shift of the effort produces a uniform
            # change in the flow, so no direction of the trace space is invisible.
            # `rocket.py`'s own connections declare NOTHING here, which is why
            # `excess_null_directions` is None right across that graph.
            expected_null_dim=0,
            note="%s, %s, real physics on both sides"
                 % (seam_id, SEAM_PORT_TYPE[seam_id].value),
        ))
    return CaseGraph(
        name="rocket-ascent-2d-real",
        agents=agents,
        connections=connections,
        decomposition=Decomposition.NON_OVERLAPPING,
        macro_dt=max(a.capabilities.dt_native for a in agents),
        flux_matching=FluxMatching.TIME_INTEGRATED,
        #: **Tier 80.**  `L7/R9/lag`'s own message ends *"declare sigma with the
        #: sigma_lag it was measured at"*, and until this tier nothing on this
        #: graph could: sigma is a property of a RUN and no run existed.  It is
        #: passed rather than hard-wired because a constant measured at one
        #: operating point is not a constant of the case -- W314's probe state is
        #: part of the number.  This graph is not in the W189 census (see
        #: `tests/test_tier45_region_assembly.py`'s exemption), so declaring it
        #: costs the byte-identity control nothing.
        measured=measured,
        note="R0 complete: all seven agents wired to build-repo solvers at "
             "dt_scale=%g%s" % (dt_scale,
                                "" if measured is None else ", sigma declared"),
    ), experts
