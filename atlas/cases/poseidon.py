"""The THIRD real case study, and the first that is not a discretization at all.

`window_ns.py` and `channel_ns.py` are two discretizations of the incompressible
Navier-Stokes equations.  This one is **Poseidon-T**, a frozen 20.8M-parameter
neural operator (`camlab-ethz/Poseidon-T`, wrapped by the build repo's
`adapters.FrozenFluidExpert`).  W55 asks for "a different solver class, or a
trained checkpoint with a boundary channel", and the two are not the same test:
a second discretization asks whether a rule survives different internals, and a
trained checkpoint asks whether it survives having no internals to speak of.

What this expert cannot do, and why that is the finding
------------------------------------------------------

**It is fixed at 128x128.**  `adapters.EXPERT_RES` is not a default; the wrapper
raises if the checkpoint's `image_size` disagrees with it.  So there is **no
same-class monolith**: you cannot ask this expert to solve the whole domain, at
any resolution, ever.  Everything Tier 0 measures against a monolithic reference
-- `tau`, `sigma`, `C_mu`, and L2/C2's reference form -- is **structurally
unavailable**, not merely unmeasured.

That is worth stating as a property of learned experts rather than a limitation
of this one.  A fixed-resolution operator can be composed and it can be probed,
and it cannot be *validated against itself at scale*.  L2/C2's **reference-free**
surrogate is the only form of the cut criterion available here, which is the case
it was derived for.

**Its boundary channel is an initial condition, not a boundary condition.**
`FrozenFluidExpert.step(u, v, dt)` takes no boundary argument: there is nowhere
to hold a trace *during* the step, because the step has no interior structure the
caller can reach.  What this file does instead is overwrite the ring of the input
field, which is what a Dirichlet channel degenerates to for a one-shot map.

`BCChannel` has no value for that distinction and it should: `DIRICHLET` on this
record means *"the ring of the initial condition is settable"*, which is strictly
weaker than what `WindowNS` provides and is not the same object §2.2's
Steklov-Poincare operator is defined from.  **Opened as W59.**

**Its elliptic content is not declarable.**  `EllipticSubsolve` is
`none | embedded | exposed` with no `unknown`, and for a black box the honest
answer is `unknown`: declaring `NONE` asserts something nobody measured and
switches R10 off, and declaring `EMBEDDED` asserts something nobody measured and
makes L2 refuse.  **Opened as W60** -- and `elliptic_signature()` below is the
measurement that would decide it, from the probe alone, with two calibration
points already in hand.

What it CAN do
--------------

Everything on the probe side: `Lambda`, `beta`, `kappa`, `mu`, the null count,
the asymmetry, `Xi` against another expert -- and, for the first time in this
vault, a **real reproducibility floor**.  §1 recorded that the probe floor is
machine epsilon "not OP-6's 1e-6", which was true of a float64 deterministic
solver and says nothing about a checkpoint.  Here it is measured.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from ..capability import (
    BCChannel,
    ClaimType,
    Differentiable,
    Direction,
    EllipticSubsolve,
    ExpertCapabilities,
    TimeDiscretization,
    MotionClass,
    port_decl,
)
from ..graph import (
    Agent,
    CaseGraph,
    Connection,
    Decomposition,
    MeasuredConstants,
)
from ..assembly import GridPartitionOfUnity
from ..ports import PortType, ResponseHalf
from ..probe import OPERATOR_CONTENT_FLOOR, operator_content
from ..transfer import Prolongation
from .window_ns import H, M_EFF, MACRO_DT, NU, U_INF, fourier_basis, load_reference

#: The checkpoint's image size. Not a default -- `FrozenFluidExpert.__init__`
#: raises if the checkpoint disagrees with it.
EXPERT_RES = 128

#: Two 128-cell windows with a 21-cell overlap span 235 cells. Chosen so the
#: halo rule can be SATISFIED: with the domain fixed at 255 the leftover overlap
#: is 1 cell, below the 2-cell domain of dependence, and L2 would refuse for a
#: reason that is about arithmetic rather than about this expert.
MONO_N = 2 * EXPERT_RES - 21

MECH_SCALES = {"stress": U_INF**2, "velocity": U_INF, "power_area": U_INF**3}

_FACE_GEOM = {"xlo": (-1.0, "x"), "xhi": (+1.0, "x"),
              "ylo": (-1.0, "y"), "yhi": (+1.0, "y")}


@lru_cache(maxsize=1)
def load_expert(device: str = "cpu", threads: int = 8):
    """`adapters.FrozenFluidExpert`, under the private-module name.

    Both repositories have a top-level `atlas` package, so the build repo cannot
    go on `sys.path` -- `window_ns.load_reference` sets up the private package
    and this rides on it.
    """
    import importlib

    load_reference()
    ad = importlib.import_module("atlas_windfarm_reference.adapters")
    return ad.FrozenFluidExpert(device=device, threads=threads)


# ---------------------------------------------------------------------------
# the tiling -- this case study's own geometry, since the window size is fixed
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseidonTiling:
    """2x2 tiling of `MONO_N` by windows of exactly `EXPERT_RES` cells.

    ``n`` is not a parameter: the expert is fixed at 128x128 and the geometry has
    to accommodate it rather than the other way round. That inversion is the
    practical face of the finding above -- with a solver you choose the window to
    suit the halo, and with a checkpoint you choose the domain to suit the window.
    """

    ramp: int = 8
    mono_n: int = MONO_N

    @property
    def n(self) -> int:
        return EXPERT_RES

    @property
    def halo(self) -> int:
        return 2 * EXPERT_RES - self.mono_n

    @property
    def offsets(self):
        o = self.mono_n - self.n
        return [(0, 0), (o, 0), (0, o), (o, o)]

    @property
    def names(self):
        return ["P00", "P10", "P01", "P11"]

    def artificial_faces(self, ox, oy):
        out = []
        if ox > 0:
            out.append("xlo")
        if ox + self.n < self.mono_n:
            out.append("xhi")
        if oy > 0:
            out.append("ylo")
        if oy + self.n < self.mono_n:
            out.append("yhi")
        return tuple(out)

    def cut(self, f):
        return np.stack([f[oy:oy + self.n, ox:ox + self.n]
                         for ox, oy in self.offsets])

    @lru_cache(maxsize=8)
    def weights(self):
        r = max(self.ramp, 1)
        idx = np.arange(self.n) + 0.5
        raw = []
        for ox, oy in self.offsets:
            wx, wy = np.ones(self.n), np.ones(self.n)
            if ox > 0:
                wx = np.minimum(wx, np.clip(idx / r, 0.0, 1.0))
            if ox + self.n < self.mono_n:
                wx = np.minimum(wx, np.clip((self.n - idx) / r, 0.0, 1.0))
            if oy > 0:
                wy = np.minimum(wy, np.clip(idx / r, 0.0, 1.0))
            if oy + self.n < self.mono_n:
                wy = np.minimum(wy, np.clip((self.n - idx) / r, 0.0, 1.0))
            w = np.zeros((self.mono_n, self.mono_n))
            w[oy:oy + self.n, ox:ox + self.n] = np.minimum(wy[:, None],
                                                           wx[None, :]) ** 2
            raw.append(w)
        tot = np.sum(raw, axis=0)
        tot = np.where(tot <= 0.0, 1.0, tot)
        return [w / tot for w in raw]

    def assemble(self, us, vs):
        au = np.zeros((self.mono_n, self.mono_n))
        av = np.zeros((self.mono_n, self.mono_n))
        ws = self.weights()
        for k, (ox, oy) in enumerate(self.offsets):
            chi = ws[k][oy:oy + self.n, ox:ox + self.n]
            au[oy:oy + self.n, ox:ox + self.n] += chi * us[k]
            av[oy:oy + self.n, ox:ox + self.n] += chi * vs[k]
        return au, av

    def contaminated(self, d_cells=2):
        out = []
        ii = np.arange(self.n) + 0.5
        for ox, oy in self.offsets:
            faces = self.artificial_faces(ox, oy)
            dx = np.full(self.n, np.inf)
            dy = np.full(self.n, np.inf)
            if "xlo" in faces:
                dx = np.minimum(dx, ii)
            if "xhi" in faces:
                dx = np.minimum(dx, self.n - ii)
            if "ylo" in faces:
                dy = np.minimum(dy, ii)
            if "yhi" in faces:
                dy = np.minimum(dy, self.n - ii)
            out.append(np.minimum(dy[:, None], dx[None, :]) <= d_cells)
        return out

    def partition_of_unity(self, d_cells=2):
        idx, wts, bad = {}, {}, {}
        ws, cont = self.weights(), self.contaminated(d_cells)
        for k, (ox, oy) in enumerate(self.offsets):
            rows, cols = np.meshgrid(np.arange(oy, oy + self.n),
                                     np.arange(ox, ox + self.n), indexing="ij")
            idx[self.names[k]] = (rows * self.mono_n + cols).reshape(-1)
            wts[self.names[k]] = ws[k][oy:oy + self.n,
                                       ox:ox + self.n].reshape(-1)
            bad[self.names[k]] = cont[k].reshape(-1)
        return GridPartitionOfUnity(self.mono_n * self.mono_n, idx, wts,
                                    contaminated=bad)


DEFAULT_TILING = PoseidonTiling()


# ---------------------------------------------------------------------------
# the agent
# ---------------------------------------------------------------------------


@dataclass
class PoseidonAgent:
    """One Poseidon-T window exposed as a `boundary_response`.

    The ring is written into the **initial condition** and the operator is run
    once. For a one-shot map that is what a Dirichlet channel degenerates to, and
    the record says `DIRICHLET` because that is the closest available value --
    see W59, and this class's module docstring, for why that is not good enough.
    """

    agent_id: str
    u0: np.ndarray
    v0: np.ndarray
    shared_faces: tuple[str, ...]
    dt: float = MACRO_DT
    nu: float = NU
    expert: Any = None
    n_calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.u0 = np.ascontiguousarray(np.asarray(self.u0, dtype=float))
        self.v0 = np.ascontiguousarray(np.asarray(self.v0, dtype=float))
        self.n = int(self.u0.shape[0])
        if self.n != EXPERT_RES:
            raise ValueError(
                f"Poseidon-T is fixed at {EXPERT_RES}x{EXPERT_RES}; got {self.n}. "
                "This is the constraint, not a configuration error."
            )
        self.h = H
        self.expert = self.expert or load_expert()

    @staticmethod
    def face_of(port_name):
        return port_name.split(":", 1)[0]

    def _ring_index(self, face):
        return {
            "xlo": ((slice(None), 0), (slice(None), 1)),
            "xhi": ((slice(None), -1), (slice(None), -2)),
            "ylo": ((0, slice(None)), (1, slice(None))),
            "yhi": ((-1, slice(None)), (-2, slice(None))),
        }[face]

    def _normal_field(self, face, u, v):
        return u if _FACE_GEOM[face][1] == "x" else v

    def step(self, u, v, dt=None):
        self.n_calls += 1
        return self.expert.step(np.ascontiguousarray(u), np.ascontiguousarray(v),
                                dt or self.dt)

    def respond(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        face = self.face_of(port_name)
        ring, interior = self._ring_index(face)
        trace = np.asarray(trace, dtype=float).reshape(-1)
        if trace.shape[0] != self.n:
            raise ValueError(
                f"trace on {port_name} has length {trace.shape[0]}, expected {self.n}")
        r_u, r_v = self.u0.copy(), self.v0.copy()
        if _FACE_GEOM[face][1] == "x":
            r_u[ring] = r_u[ring] + trace
        else:
            r_v[ring] = r_v[ring] + trace
        u1, v1 = self.step(r_u, r_v)
        w = self._normal_field(face, u1, v1)
        return self.nu * (w[ring] - w[interior]) / self.h

    def storage(self, u=None, v=None):
        if u is None:
            u, v = self.u0, self.v0
        u = np.asarray(u, dtype=float)
        v = self.v0 if v is None else np.asarray(v, dtype=float)
        return 0.5 * float(np.sum(u * u + v * v)) * self.h**2

    # NOTE: no `validity`. The checkpoint's trained envelope is a lead-time band
    # (`EXPERT_NATIVE_DT` to `EXPERT_MAX_DT`) that the wrapper already enforces by
    # raising, and nothing measures whether a given FIELD is in distribution.
    # Declaring a predicate that only checks finiteness would be declaring a
    # validity claim nobody earned -- spec §3.4's decertification is the honest
    # outcome, and W13's per-expert declination rate `p` is what would lift it.


# ---------------------------------------------------------------------------
# named measurements this expert makes possible
# ---------------------------------------------------------------------------


def batch_position_spread(agent, u0, v0, dt, batch=8):
    """OP-6's claim, measured: does the answer depend on where in a batch it sits?

    `window_ns`'s record says the probe floor is machine epsilon "and NOT the
    neural checkpoint's 1e-6 batch-position dependence" -- an assertion about an
    expert the vault had never probed. This measures it.
    """
    e = agent.expert
    if not hasattr(e, "step_many"):
        return float("nan")
    U = np.repeat(u0[None], batch, axis=0)
    V = np.repeat(v0[None], batch, axis=0)
    out_u, _out_v = e.step_many(U, V, dt)
    ref = out_u[0]
    return float(max(np.max(np.abs(out_u[k] - ref)) for k in range(1, batch)))


def probe_columns(agent, u0, v0, dt, face="xhi", eps=1e-4, m_eff=M_EFF):
    """One block of Lambda on the declared Fourier basis, by finite differences."""
    B = fourier_basis(EXPERT_RES, m_eff)
    ring, interior = agent._ring_index(face)
    axis = _FACE_GEOM[face][1]
    h, nu = agent.h, agent.nu

    def flux(trace):
        r_u, r_v = u0.copy(), v0.copy()
        if axis == "x":
            r_u[ring] = r_u[ring] + trace
        else:
            r_v[ring] = r_v[ring] + trace
        u1, v1 = agent.step(r_u, r_v, dt)
        w = u1 if axis == "x" else v1
        return nu * (w[ring] - w[interior]) / h

    z = flux(np.zeros(EXPERT_RES))
    cols = [(flux(eps * B[:, k]) - z) / eps for k in range(m_eff)]
    return H * (B.T @ np.column_stack(cols))


def eps_sweep(agent, u0, v0, dt, m_eff=M_EFF,
              steps=(1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)):
    """The probe floor, measured rather than asserted.

    A finite-difference probe of a float32 network divides an output difference
    of relative size ~1e-7 by ``eps``, so below some ``eps`` the columns are
    noise amplified by ``1/eps``.  Above some ``eps`` the chord stops being a
    derivative.  **The trusted window is where consecutive rows agree**, and for
    `WindowNS` it was every row from 1e-2 to 1e-8 (§1) -- that is what a float64
    deterministic solver buys and it is not what a checkpoint buys.
    """
    rows = []
    for e in steps:
        S = probe_columns(agent, u0, v0, dt, eps=e, m_eff=m_eff)
        sv = np.linalg.svd(S, compute_uv=False)
        rows.append({"eps": e, "beta": float(sv[-1]),
                     "kappa": float(sv[0] / max(sv[-1], 1e-300)),
                     "norm_S": float(np.linalg.norm(S))})
    # the trusted window: consecutive ||S|| agreeing to better than 5%
    ok = []
    for i in range(len(rows) - 1):
        a, b = rows[i]["norm_S"], rows[i + 1]["norm_S"]
        if abs(a - b) / max(abs(a), 1e-300) < 0.05:
            ok.extend([rows[i]["eps"], rows[i + 1]["eps"]])
    window = [max(ok), min(ok)] if ok else [None, None]
    best = float(np.sqrt(window[0] * window[1])) if ok else 1e-4
    return {"rows": rows, "trusted_window": window, "best_eps": best,
            "floor": min(ok) if ok else None,
            "windowns_comparison": "every eps from 1e-2 to 1e-8 agreed to 7 digits"}


def elliptic_signature(S: np.ndarray, *, floor: float | None = None) -> dict:
    """**W60's measurement.** Does this black box behave as if it embeds an elliptic solve?

    **Corrected 2026-08-30 (W68).**  Read the precondition below before the
    thresholds: this statistic has no input at all unless the block it is given
    carries resolvable operator content, and it must be taken on a **per-agent
    block**, never on the assembled seam.

    `EllipticSubsolve` has no ``unknown``, so for a checkpoint the field cannot be
    declared honestly.  It can be *measured*, because §8.3 left two calibration
    points on the same instrument, the same basis and the same state:

        WindowNS, elliptic EMBEDDED   kappa = 21.73   asymmetry = 0.153
        WindowNS, elliptic EXPOSED    kappa =  1.196  asymmetry = 0.002

    The mechanism is legible: the pressure coupling is what makes the interface
    operator non-normal, and removing it takes the asymmetry down by 76x.

    **This is a diagnosis and not a proof**, and the difference matters. A high
    asymmetry is *consistent with* an embedded elliptic part; a learned operator
    could be non-normal for reasons that have nothing to do with pressure. What
    the statistic supports is a decertification -- "this record's
    `elliptic_subsolve` is not corroborated by the probe" -- and never a silent
    promotion of one value over another.

    **The precondition, and why W60's promotion was refused (W68).**  The shell
    of `thermal_seam` is a *known* embedded backward-Euler conduction solve and
    this statistic read ``kappa = 1.0002, asymmetry 8.8e-5`` on it -- "consistent
    with EXPOSED or NONE" -- and stayed there across 10,000x of exchange
    interval.  The 2026-08-29 diagnosis was that the statistic measures
    non-normality and had been read as measuring globality.  That is true and it
    is not the whole story, because measuring the shell's own block per-agent
    gives ``kappa = 1.0002`` too.  The 2026-08-30 measurement finds the actual
    mechanism: the block was ``5.0001 I`` to five digits -- the shell's film
    coefficient -- with the conduction operator 300x underneath it.

    A delta at one seam cell moves that cell by 0.471 and its neighbour by
    1.6e-3, decaying exponentially over 5 cells: the globality is **present and
    resolved**, and it is simply 300x below the boundary condition sitting on
    top.  So kappa and the asymmetry were not blind, they were **swamped**, and
    no threshold on them could have helped.  `probe.operator_content` is the
    quantity that separates the two situations, and it is a precondition here
    rather than a caveat: below the floor this returns NO OPERATOR RESOLVED,
    which is neither of the two calibration points and is not a weaker form of
    either.

    Passing ``floor=0.0`` restores the pre-2026-08-30 behaviour, which is what
    the two synthetic calibration matrices want and nothing else should.
    """
    n = float(np.linalg.norm(S))
    if n == 0.0:
        return {"kappa": None, "asymmetry": None, "verdict": "empty operator"}
    sv = np.linalg.svd(S, compute_uv=False)
    kappa = float(sv[0] / max(sv[-1], 1e-300))
    asym = float(np.linalg.norm(S - S.T) / n)
    floor = OPERATOR_CONTENT_FLOOR if floor is None else float(floor)
    omega = operator_content(S)
    if S.shape[0] == S.shape[1] and omega == omega and omega < floor:
        return {
            "kappa": kappa, "asymmetry": asym, "operator_content": omega,
            "verdict": (f"NO OPERATOR RESOLVED -- identity defect {omega:.3g} is below "
                        f"the {floor:g} floor, so this block is a boundary coefficient "
                        "and kappa and the asymmetry are properties of THAT. Neither "
                        "calibration point applies; the record's elliptic_subsolve is "
                        "neither corroborated nor contradicted here"),
            "calibration": {"embedded": {"kappa": 21.73, "asymmetry": 0.153},
                            "exposed": {"kappa": 1.196, "asymmetry": 0.002}},
        }
    if asym < 0.02 and kappa < 3.0:
        v = "consistent with EXPOSED or NONE (near-normal, well conditioned)"
    elif asym > 0.10 or kappa > 10.0:
        v = "consistent with EMBEDDED (non-normal and/or ill conditioned)"
    else:
        v = "INDETERMINATE -- between the two calibration points"
    return {"kappa": kappa, "asymmetry": asym, "verdict": v,
            "calibration": {"embedded": {"kappa": 21.73, "asymmetry": 0.153},
                            "exposed": {"kappa": 1.196, "asymmetry": 0.002}}}


def probe_seam(agent, u0, v0, dt, eps=1e-4, m_eff=M_EFF):
    """S = Lambda_A + Lambda_B on one seam, both sides being this same expert."""
    a = probe_columns(agent, u0, v0, dt, face="xhi", eps=eps, m_eff=m_eff)
    b = probe_columns(agent, u0, v0, dt, face="xlo", eps=eps, m_eff=m_eff)
    S = a + b
    sv = np.linalg.svd(S, compute_uv=False)
    sym = 0.5 * (S + S.T)
    mu = float(np.linalg.eigvalsh(sym)[0])
    floor = 1e-12 * float(np.linalg.norm(S))
    return {
        "beta": float(sv[-1]),
        "kappa": float(sv[0] / max(sv[-1], 1e-300)),
        "null_dim": int(np.sum(sv <= floor)),
        "mu": mu, "pi": float(max(0.0, -mu)),
        "asymmetry": float(np.linalg.norm(S - S.T) / max(np.linalg.norm(S), 1e-300)),
        "norm_S": float(np.linalg.norm(S)),
        "elliptic_signature_W60": elliptic_signature(S),
    }


def xi_against_window(agent, u_full, v_full, dt, eps=1e-4, m_eff=M_EFF):
    """Xi = ||Lambda_this|| / ||Lambda_ref||, the composability index.

    The reference is `window_ns`'s exposed agent on the same 128-cell window and
    the same state, so the ratio is between two experts and not between two
    harnesses.  §8.3 measured 0.19-0.53 for the embedded-vs-exposed pair of ONE
    solver; this is the first cross-solver value.
    """
    from .window_ns import WindowAgent                              # noqa: PLC0415

    u0 = np.ascontiguousarray(u_full[:EXPERT_RES, :EXPERT_RES])
    v0 = np.ascontiguousarray(v_full[:EXPERT_RES, :EXPERT_RES])
    mine = probe_columns(agent, u0, v0, dt, face="xhi", eps=eps, m_eff=m_eff)
    ref_agent = WindowAgent(agent_id="ref", u0=u0, v0=v0, shared_faces=("xhi",),
                            dt=dt, nu=agent.nu, expose_elliptic=True)
    B = fourier_basis(EXPERT_RES, m_eff)
    z = ref_agent.respond("xhi:MECH", np.zeros(EXPERT_RES))
    cols = [(ref_agent.respond("xhi:MECH", eps * B[:, k]) - z) / eps
            for k in range(m_eff)]
    ref = H * (B.T @ np.column_stack(cols))
    nm, nr = float(np.linalg.norm(mine)), float(np.linalg.norm(ref))
    return {
        "norm_Lambda_poseidon": nm,
        "norm_Lambda_windowns_exposed": nr,
        "Xi": nm / max(nr, 1e-300),
        "operator_defect": float(np.linalg.norm(mine - ref) / max(nr, 1e-300)),
    }


# ---------------------------------------------------------------------------
# the capability record and the graph
# ---------------------------------------------------------------------------


def face_prolongation(agent_id, port_name, n_cells):
    return Prolongation(
        agent_id=agent_id, port_name=port_name,
        matrix=fourier_basis(n_cells, M_EFF),
        gram_V=H * np.eye(n_cells),
        label="16-mode real Fourier basis (probed-dtn-coupling 2.2)",
    )


def poseidon_capabilities(expert: PoseidonAgent,
                          reproducibility_floor: float = 1e-5,
                          elliptic: EllipticSubsolve = EllipticSubsolve.NONE,
                          ) -> ExpertCapabilities:
    """The L1 record, with two fields the schema cannot express honestly.

    ``elliptic_subsolve`` defaults to ``NONE`` and **that is a declaration
    nobody measured** -- W60. It is a parameter here rather than a constant so a
    caller can compile the same graph both ways and see that the verdict flips,
    which is the point: an undeclarable field is not a small thing when R10 turns
    on it.

    ``reproducibility_floor`` is 1e-5 by default and should be passed the value
    `eps_sweep` measures. It is not machine epsilon and this is the first expert
    in the vault for which that is true.
    """
    ports = [
        port_decl(
            name=f"{f}:MECH", port_type=PortType.MECH,
            geometry=f"{f} face of {expert.agent_id}",
            direction=Direction.BIDIRECTIONAL, nondim=dict(MECH_SCALES),
            effective_resolution=M_EFF, motion_class=MotionClass.STATIC,
            # W66 / L3-C9: nu du/dn, the MECH EFFORT. Declared even though this
            # expert is a black box -- the half a callable RETURNS is a fact
            # about the wrapper, not about the weights, so `unknown` would be
            # dishonest in the opposite direction from W60.
            response_half=ResponseHalf.EFFORT,
            prolongation=face_prolongation(expert.agent_id, f"{f}:MECH", expert.n),
        )
        for f in expert.shared_faces
    ]
    return ExpertCapabilities(
        expert_id=expert.agent_id,
        ports=ports,
        # The ring of the INITIAL CONDITION is settable; there is no boundary
        # condition held through the step, because there is no "through". W59.
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=False,
        elliptic_subsolve=elliptic,
        stencil_radius=2,
        substeps_per_macro_step=1,
        # A learned one-shot map is neither explicit nor implicit. R2b reads this
        # and must not lift the rung to probed-DtN on it.
        time_discretization=TimeDiscretization.UNKNOWN,
        differentiable=Differentiable.NONE,
        dt_native=expert.dt,
        L_native=expert.n * H,
        storage=expert.storage,
        equivariances=(),
        validity=None,
        governing_family="incompressible-navier-stokes-2d",
        lambda_ref=None,
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash="poseidon-t-camlab-ethz-frozen-res128",
        boundary_response=expert.respond,
        boundary_response_jvp=None,
        reproducibility_floor=reproducibility_floor,
        deterministic=True,
        note="Poseidon-T, frozen, 20.8M parameters; the ring is an initial "
             "condition (W59) and the elliptic content is undeclarable (W60)",
    )


SEAMS = {
    "sx0": (("P00", "xhi"), ("P10", "xlo")),
    "sx1": (("P01", "xhi"), ("P11", "xlo")),
    "sy0": (("P00", "yhi"), ("P01", "ylo")),
    "sy1": (("P10", "yhi"), ("P11", "ylo")),
}


def build(u_full, v_full, dt=MACRO_DT, tiling=DEFAULT_TILING, nu=NU,
          measured: MeasuredConstants | None = None,
          reproducibility_floor: float = 1e-5,
          elliptic: EllipticSubsolve = EllipticSubsolve.NONE,
          experts=None):
    """The four-window Poseidon-T graph.

    ``measured`` defaults to None, as `channel_ns` does and for the same reason:
    a constant measured on another expert is not this expert's, and the compile
    must say so.  Here it is stronger than a default -- `tau`, `sigma` and `C_mu`
    are **not measurable at all** for this expert, so the decertification is
    permanent until either the checkpoint becomes resolution-free or the bound
    acquires a branch that does not need a monolith.
    """
    sub = tiling.mono_n
    u_full = np.asarray(u_full)[:sub, :sub]
    v_full = np.asarray(v_full)[:sub, :sub]
    ex = load_expert()
    us, vs = tiling.cut(u_full), tiling.cut(v_full)
    experts = experts or {
        name: PoseidonAgent(agent_id=name, u0=us[k], v0=vs[k],
                            shared_faces=tiling.artificial_faces(*tiling.offsets[k]),
                            dt=dt, nu=nu, expert=ex)
        for k, name in enumerate(tiling.names)
    }
    agents = [Agent(n, poseidon_capabilities(experts[n], reproducibility_floor,
                                             elliptic), domain=f"tile {n}")
              for n in tiling.names]
    connections = [
        Connection(seam_id=sid, a=(a, f"{af}:MECH"), b=(b, f"{bf}:MECH"),
                   port_type=PortType.MECH, derive_space=True,
                   geometrically_coincident=True, expected_null_dim=0,
                   note="seam presented through a declared 16-mode Fourier basis")
        for sid, ((a, af), (b, bf)) in SEAMS.items()
    ]
    return (
        CaseGraph(
            name="poseidon-t-2x2",
            agents=agents,
            connections=connections,
            decomposition=Decomposition.OVERLAPPING,
            overlap=tiling.halo * H,
            overlap_cells=tiling.halo,
            partition_of_unity=tiling.partition_of_unity(),
            cross_points=("centre",),
            macro_dt=dt,
            measured=measured,
            note=(f"four Poseidon-T windows of {EXPERT_RES} cells tiling "
                  f"{tiling.mono_n}x{tiling.mono_n}, halo {tiling.halo}. "
                  f"There is NO same-class monolith and there cannot be: the "
                  f"checkpoint is fixed at {EXPERT_RES}. W55's third expert"),
        ),
        experts,
    )
