"""The SECOND real case study: `reference.ChannelNS`, a different discretization.

**W55, and why a second expert is worth a whole case study.** Every constant in
the vault -- `C_mu = 1.2`, `Pi`'s two limits, R10b's cadence rule, `L`, `tau`,
`sigma`, `beta`, `kappa`, `mu`, and now L2/C2's `cut_defect_bound` -- was measured
against `reference.WindowNS` and its no-projection subclass, plus one `SpectralNS`
control that measures zero by construction.  `tier0-measurements` §9.8 calls that
"the largest remaining caveat" and it has been open ever since.

The question this file exists to answer is **not** whether the numbers match.
They will not: a different discretization of the same equations has a different
truncation error and there is no reason its `tau` should equal anybody else's.
The question is whether the **rules** hold structurally -- R10, R10b, R11, R2b --
and whether the **factorizations** do: the bias-variance identity behind L6/C1,
`Pi`'s composition behind W49, and the restriction-defect identity behind L2/C2.
A rule that only holds for one solver's internals is not a rule.

What is actually different, and it is not cosmetic
--------------------------------------------------

`WindowNS`                          `ChannelNS`
-----------------------------       --------------------------------------
skew-symmetric advection            advective (non-conservative) form,
``1/2[u.grad f + div(uf)]``         ``u dx f + v dy f`` via ``np.gradient``
5-point centred Laplacian           5-point Laplacian by ``np.roll``
projection with homogeneous          ``pressure.project_outflow``: a MAC
Neumann pressure on all four        projection, Neumann at inlet and walls,
faces -- **singular in the          **Dirichlet at the outlet** -- and
constant mode**                     therefore **non-singular**
explicit fractional step            explicit Heun plus projection

**The Neumann/Dirichlet difference is the one to watch.**  §4.1 measured that
`WindowNS`'s all-Neumann pressure solve has a compatibility condition -- the
right-hand side must integrate to zero, i.e. the ring's net flux must balance --
and that a subdomain of a through-flow violates it by construction.  That was
99.8% of the first composed defect and it is the measurement R10 came out of.
`project_outflow` has an outlet Dirichlet, so **it has no compatibility condition
at all**: the flux imbalance simply leaves.

So R10's *measured mechanism* is absent here while R10's *stated reason* -- an
elliptic operator is global over whatever domain it runs on, so cutting the
domain cuts the operator -- is untouched.  Whether R10 still bites is a real
prediction with a real chance of coming out either way, and it is the sharpest
single thing this case study tests.

Two modes, the same two as `window_ns.py`
-----------------------------------------

``build(mode="as-built")``   the agent keeps `project_outflow`; L2 refuses (R10)
``build(mode="split-step")`` the elliptic part is the composition layer's

The transmission is this file's, as it must be: `ChannelNS.step` applies the wind
farm's own outer boundary condition and poses no boundary channel, so an agent
built on it needs one supplied.  That is not hand-written *coupling* code -- it is
the expert's Dirichlet ring, exactly what `WindowNS.transmission='dirichlet'` is
inside `WindowNS`, and the compiler never sees it.  What the compiler sees is a
`boundary_response` callable, the same as everywhere else.
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
    GlobalField,
    MeasuredConstants,
)
from ..ports import PortType, ResponseHalf
from ..transfer import Prolongation
from .window_ns import (
    DEFAULT_BUILD_REPO,
    H,
    M_EFF,
    MACRO_DT,
    MONO_N,
    NU,
    U_INF,
    Tiling,
    fourier_basis,
    load_reference,
)

#: `ChannelNS` is Heun on a 3-point stencil, the same as `WindowNS`, so one
#: sub-step moves information two cells.  Declared rather than inferred, because
#: the compiler cannot see inside the expert -- and identical to `WindowNS`'s
#: only because the two happen to share a stencil width, not because anything
#: makes them agree.
STENCIL_RADIUS = 2

#: Sub-steps per macro-step at `MACRO_DT`.  `ChannelNS.dt_max()` is a CFL bound
#: rather than a fixed count, so this is derived from it at the working state
#: rather than copied from `window_ns`.  R10b makes the exchange interval
#: `dt_native / substeps`, so getting it wrong is two orders in `tau`.
SUBSTEPS = 10

MECH_SCALES = {"stress": U_INF**2, "velocity": U_INF, "power_area": U_INF**3}

FACES = ("xlo", "xhi", "ylo", "yhi")

_FACE_GEOM = {
    "xlo": (-1.0, "x"),
    "xhi": (+1.0, "x"),
    "ylo": (-1.0, "y"),
    "yhi": (+1.0, "y"),
}


@lru_cache(maxsize=1)
def load_pressure():
    """`pressure.project_outflow` from the build repo, resolved at call time.

    Goes through `window_ns.load_reference`'s private-module trick rather than
    putting the build repo on ``sys.path``: **both repositories have a top-level
    package called `atlas` and they are different packages.**  Importing the
    build repo's by its own name shadows this one, and the failure is an
    unhelpful `ModuleNotFoundError` several frames away.
    """
    import importlib

    load_reference()          # ensures the private package is in sys.modules
    return importlib.import_module("atlas_windfarm_reference.pressure")


def substeps_at(dt: float) -> int:
    """The agent's own sub-step count at ``dt``, which R10b must match.

    Proportional to ``dt`` because the CFL bound is, and derived from ``SUBSTEPS``
    at `MACRO_DT` for the same reason `window_ns.substeps_at` exists: hard-coding
    it at one macro-step and reusing it at another is the bug R10b was found by.
    """
    return max(1, int(round(SUBSTEPS * dt / MACRO_DT)))


# ---------------------------------------------------------------------------
# the agent: ChannelNS's spatial operators, with a Dirichlet ring
# ---------------------------------------------------------------------------


@dataclass
class ChannelWindow:
    """One window whose interior operator is `ChannelNS`'s, with a pinned ring.

    The operator is the expert's -- ``_rhs`` is `ChannelNS`'s advective-form
    advection and rolled Laplacian, and the projection is `project_outflow`.
    What this class adds is the **transmission**: a Dirichlet ring re-imposed
    after every stage, which is the boundary channel `ChannelNS.step` does not
    have because it applies the wind farm's outer boundary condition instead.

    ``expose_elliptic`` drops the projection, which is what `split-step` means
    and what R10 requires before an agent with an elliptic part is decomposable.
    """

    nu: float = NU
    n: int = 138
    h: float = H
    expose_elliptic: bool = False
    _solver: Any = field(default=None, init=False, repr=False)
    _proj: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        ref = load_reference()
        # domain is square and of this window's own size; ChannelNS derives its
        # grid from (domain, dx), so one instance per window shape is enough.
        L = self.n * self.h
        self._solver = ref.ChannelNS(nu=self.nu, dx=self.h, domain=(0.0, L, 0.0, L))
        self._proj = load_pressure().project_outflow

    def dt_max(self, u: np.ndarray, v: np.ndarray) -> float:
        umax = float(np.max(np.hypot(u, v))) + 1e-12
        return min(0.4 * self.h / umax, 0.25 * self.h**2 / max(self.nu, 1e-12))

    def step(self, u: np.ndarray, v: np.ndarray, dt: float,
             ring: tuple[np.ndarray, np.ndarray] | None = None,
             fx: np.ndarray | float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """One Heun sub-step with the ring pinned, then optionally projected."""
        s = self._solver
        k1u, k1v = s._rhs(u, v, fx)
        pu, pv = u + dt * k1u, v + dt * k1v
        if ring is not None:
            pu, pv = _pin(pu, ring[0]), _pin(pv, ring[1])
        k2u, k2v = s._rhs(pu, pv, fx)
        u1 = u + 0.5 * dt * (k1u + k2u)
        v1 = v + 0.5 * dt * (k1v + k2v)
        if ring is not None:
            u1, v1 = _pin(u1, ring[0]), _pin(v1, ring[1])
        if not self.expose_elliptic:
            u1, v1, _rep = self._proj(u1, v1, self.h)
            if ring is not None:
                u1, v1 = _pin(u1, ring[0]), _pin(v1, ring[1])
        return u1, v1

    def step_macro(self, u, v, dt, ring=None, fx=0.0, substeps=None):
        substeps = substeps or substeps_at(dt)
        hs = dt / substeps
        for _ in range(substeps):
            u, v = self.step(u, v, hs, ring=ring, fx=fx)
        return u, v


def _pin(a: np.ndarray, r: np.ndarray) -> np.ndarray:
    out = a.copy()
    out[0, :] = r[0, :]
    out[-1, :] = r[-1, :]
    out[:, 0] = r[:, 0]
    out[:, -1] = r[:, -1]
    return out


@dataclass
class ChannelAgent:
    """`ChannelWindow` exposed as a `boundary_response`, mirroring `WindowAgent`.

    Deliberately line-for-line the same shape as `window_ns.WindowAgent`: same
    ring indexing, same one-sided outward normal derivative, same zero-probe
    subtraction, same base state.  If the two agents' probes differ, it is the
    expert that differs and not the instrument -- which is the only way a second
    expert can say anything about the first one's numbers.
    """

    agent_id: str
    u0: np.ndarray
    v0: np.ndarray
    shared_faces: tuple[str, ...]
    dt: float = MACRO_DT
    nu: float = NU
    expose_elliptic: bool = False
    n_calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.u0 = np.asarray(self.u0, dtype=float)
        self.v0 = np.asarray(self.v0, dtype=float)
        self.n = int(self.u0.shape[0])
        self.h = H
        self._win = ChannelWindow(nu=self.nu, n=self.n, h=self.h,
                                  expose_elliptic=self.expose_elliptic)
        self.transmission = "dirichlet"

    @staticmethod
    def face_of(port_name: str) -> str:
        return port_name.split(":", 1)[0]

    def _ring_index(self, face: str):
        return {
            "xlo": ((slice(None), 0), (slice(None), 1)),
            "xhi": ((slice(None), -1), (slice(None), -2)),
            "ylo": ((0, slice(None)), (1, slice(None))),
            "yhi": ((-1, slice(None)), (-2, slice(None))),
        }[face]

    def _normal_field(self, face, u, v):
        return u if _FACE_GEOM[face][1] == "x" else v

    def respond(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        """(port, trace) -> flux.  `probed-dtn-coupling` §2.2's form, normative."""
        face = self.face_of(port_name)
        ring, interior = self._ring_index(face)
        trace = np.asarray(trace, dtype=float).reshape(-1)
        if trace.shape[0] != self.n:
            raise ValueError(
                f"trace on {port_name} has length {trace.shape[0]}, expected {self.n}"
            )
        r_u, r_v = self.u0.copy(), self.v0.copy()
        if _FACE_GEOM[face][1] == "x":
            r_u[ring] = r_u[ring] + trace
        else:
            r_v[ring] = r_v[ring] + trace
        u1, v1 = self._win.step_macro(self.u0, self.v0, self.dt, ring=(r_u, r_v))
        self.n_calls += 1
        w = self._normal_field(face, u1, v1)
        return self.nu * (w[ring] - w[interior]) / self.h

    def storage(self, u: Any = None, v: Any = None) -> float:
        if u is None:
            u, v = self.u0, self.v0
        u = np.asarray(u, dtype=float)
        v = self.v0 if v is None else np.asarray(v, dtype=float)
        return 0.5 * float(np.sum(u * u + v * v)) * self.h**2

    def validity(self, state: Any = None, cond: Any = None) -> bool:
        """Not falsified on the suite this expert was validated against.

        `ChannelNS`'s own docstring names the envelope: centred advection needs a
        cell Reynolds number of order 1 for a wiggle-free steady state, and it
        says so as a condition the caller must respect rather than one it
        enforces.  **The band is tighter than `WindowNS`'s** -- the skew form
        buys `WindowNS` a cell Re of 4 without dissipation and the advective
        form does not -- so this predicate is not a copy of that one, and a
        declaration that differs between two experts of the same family is the
        thing W33 exists to catch.
        """
        if state is None:
            u, v = self.u0, self.v0
        else:
            u, v = state
        u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
        if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
            return False
        return bool(self.h * float(np.max(np.hypot(u, v))) / self.nu <= 4.0)


# ---------------------------------------------------------------------------
# the capability record
# ---------------------------------------------------------------------------


def face_prolongation(agent_id: str, port_name: str, n_cells: int) -> Prolongation:
    """The same declared 16-mode Fourier prolongation `window_ns` uses.

    Identical on purpose: the interface basis is a declaration, and comparing
    two experts through two different declarations would confound the expert
    with the presentation. It is also the thing §10.2 showed `cut_score` is a
    function of, which is a second reason to hold it fixed.
    """
    return Prolongation(
        agent_id=agent_id,
        port_name=port_name,
        matrix=fourier_basis(n_cells, M_EFF),
        gram_V=H * np.eye(n_cells),
        label="16-mode real Fourier basis (probed-dtn-coupling 2.2)",
    )


def channel_capabilities(expert: ChannelAgent) -> ExpertCapabilities:
    """The L1 record, field for field the same shape as `window_ns`'s.

    Two fields differ, and both differences are the expert's rather than the
    harness's: ``validity``'s cell-Reynolds band is 4 rather than 8, because the
    advective form does not buy what the skew form does, and ``weight_hash``
    names a different solver. Everything else is held identical so that any
    difference the probe reports is attributable.
    """
    ports = [
        port_decl(
            name=f"{f}:MECH",
            port_type=PortType.MECH,
            geometry=f"{f} face of {expert.agent_id}",
            direction=Direction.BIDIRECTIONAL,
            nondim=dict(MECH_SCALES),
            # W66 / L3-C9: nu du/dn, the MECH EFFORT, as `window_ns`.
            response_half=ResponseHalf.EFFORT,
            effective_resolution=M_EFF,
            motion_class=MotionClass.STATIC,
            prolongation=face_prolongation(expert.agent_id, f"{f}:MECH", expert.n),
        )
        for f in expert.shared_faces
    ]
    return ExpertCapabilities(
        expert_id=expert.agent_id,
        ports=ports,
        bc_channel=BCChannel.DIRICHLET,
        bc_time_varying=True,
        elliptic_subsolve=(EllipticSubsolve.EXPOSED if expert.expose_elliptic
                           else EllipticSubsolve.EMBEDDED),
        stencil_radius=STENCIL_RADIUS,
        substeps_per_macro_step=SUBSTEPS,
        time_discretization=TimeDiscretization.EXPLICIT,
        differentiable=Differentiable.NONE,
        dt_native=expert.dt,
        L_native=expert.n * H,
        regime_law=lambda T_s, nu_p, L: T_s / (nu_p * L**2),
        storage=expert.storage,
        equivariances=(),
        validity=expert.validity,
        # The SAME family as `window_ns`, deliberately: E3 compares this string,
        # and two discretizations of the incompressible Navier-Stokes equations
        # are the same governing family. Declaring otherwise would make tau
        # UNDEFINED at every seam and would be recording a discretization
        # difference as a physics difference.
        governing_family="incompressible-navier-stokes-2d",
        lambda_ref=None,
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash=(f"channelns-nu{expert.nu:.6g}-n{expert.n}-dirichlet"
                     f"-{'exposed' if expert.expose_elliptic else 'embedded'}"),
        boundary_response=expert.respond,
        boundary_response_jvp=None,
        reproducibility_floor=np.finfo(float).eps,
        deterministic=True,
        note="reference.ChannelNS operators, Dirichlet ring, real forward pass",
    )


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------

SEAMS = {
    "sx0": (("C00", "xhi"), ("C10", "xlo")),
    "sx1": (("C01", "xhi"), ("C11", "xlo")),
    "sy0": (("C00", "yhi"), ("C01", "ylo")),
    "sy1": (("C10", "yhi"), ("C11", "ylo")),
}

NAMES = ["C00", "C10", "C01", "C11"]


def make_experts(u_full, v_full, tiling: Tiling, dt=MACRO_DT, nu=NU,
                 expose_elliptic=False) -> dict[str, ChannelAgent]:
    us, vs = tiling.cut(u_full), tiling.cut(v_full)
    out = {}
    for k, (ox, oy) in enumerate(tiling.offsets):
        name = NAMES[k]
        out[name] = ChannelAgent(
            agent_id=name, u0=us[k], v0=vs[k],
            shared_faces=tiling.artificial_faces(ox, oy),
            dt=dt, nu=nu, expose_elliptic=expose_elliptic,
        )
    return out


def build(
    u_full: np.ndarray,
    v_full: np.ndarray,
    mode: str = "split-step",
    tiling: Tiling | None = None,
    dt: float = MACRO_DT,
    nu: float = NU,
    measured: MeasuredConstants | None = None,
    experts: dict[str, ChannelAgent] | None = None,
) -> tuple[CaseGraph, dict[str, ChannelAgent]]:
    """The four-window `ChannelNS` graph, plus the experts backing it.

    ``measured`` defaults to **None and not to a copy of `window_ns`'s**. That
    is the whole point of W55: a constant measured on one expert is not a
    constant for another, and the compile must decertify here until this expert's
    own numbers are measured and declared. Borrowing them would be the W56 bug
    committed on purpose.
    """
    from .window_ns import DEFAULT_TILING

    if mode not in ("as-built", "split-step"):
        raise ValueError(mode)
    tiling = tiling or DEFAULT_TILING
    expose = mode == "split-step"
    experts = experts or make_experts(u_full, v_full, tiling, dt, nu, expose)
    agents = [Agent(n, channel_capabilities(experts[n]), domain=f"tile {n}")
              for n in NAMES]
    connections = [
        Connection(
            seam_id=seam_id,
            a=(a_id, f"{a_face}:MECH"),
            b=(b_id, f"{b_face}:MECH"),
            port_type=PortType.MECH,
            derive_space=True,
            geometrically_coincident=True,
            # n_0(Gamma) = 0 under `split-step` for the reason section 8.4 gives:
            # the null space comes from incompressibility being INSIDE Lambda_i,
            # and here it is in the composition layer. Under `as-built` it is
            # inside -- but `project_outflow` is non-singular, so the constant
            # mode is not null there either. Declared 0 for both, and the probe
            # is what decides whether that is right.
            expected_null_dim=0,
            note="seam presented through a declared 16-mode Fourier prolongation",
        )
        for seam_id, ((a_id, a_face), (b_id, b_face)) in SEAMS.items()
    ]
    return (
        CaseGraph(
            name=f"channel-ns-2x2-{mode}",
            agents=agents,
            connections=connections,
            decomposition=Decomposition.OVERLAPPING,
            overlap=tiling.halo * H,
            overlap_cells=tiling.halo,
            partition_of_unity=tiling.partition_of_unity(),
            global_fields=[
                GlobalField(
                    "pressure",
                    note=("the elliptic part. Global under mode='split-step'; "
                          "per-window `project_outflow` under mode='as-built', "
                          "which R10 refuses -- and note that this projection is "
                          "NON-SINGULAR, unlike WindowNS's all-Neumann one, so "
                          "R10's measured mechanism is absent while its stated "
                          "reason is not"),
                ),
            ],
            cross_points=("centre",),
            macro_dt=dt,
            measured=measured,
            note=(f"four reference.ChannelNS windows of {tiling.n} cells tiling "
                  f"{MONO_N}x{MONO_N} at h = 1/64, halo {tiling.halo} cells; the "
                  f"monolithic reference is the same class at n = {MONO_N}. "
                  f"W55's second expert"),
        ),
        experts,
    )
