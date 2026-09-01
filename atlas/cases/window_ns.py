"""The first REAL case study: `reference.WindowNS`, probed and composed for real.

`wind_farm.py` and `rocket.py` are fixtures -- their ``boundary_response`` is a
matrix somebody wrote down.  This one calls a solver.  Every number the compiler
reports for this graph is a measurement of an object that integrates the
Navier-Stokes equations, which is the distinction `CASE-STUDY-GUIDE.md` draws
between a fixture and a real case study, and the reason this file is the only
one that can move a Tier 0 row.

The expert lives in the build repo (`src/atlas/cases/windfarm/reference.py` on
`github.com/nonidino/physics-foundation-model`), not in this tree, so the import
is resolved at call time and the failure is a clear message rather than an
ImportError at package import.  Set ``ATLAS_BUILD_REPO`` to override the path.

Two graphs, and the difference between them is the point
--------------------------------------------------------

``build(mode="as-built")`` declares what `WindowNS` actually is: an agent with an
**embedded** pressure solve.  L2 refuses it under R10, and the refusal is the
whole finding of 2026-08-27 -- a projection method's Poisson solve is global over
whatever domain it runs on, so decomposing the domain decomposes the operator,
and the resulting error is elliptic: flat in distance from the cut, flat in
`dt`, and untouched by any halo, partition of unity or interface condition.
Measured at **99.8% of the total defect**, wearing a `tau` label.

``build(mode="split-step")`` takes the elliptic part out of the agent and gives
it to the composition layer, which is exactly what `probed-dtn-coupling` §2.1
assumes when it says the elliptic part "stays where it is".  With that, a halo
wide enough for the agent's own domain of dependence, a partition of unity that
vanishes at each window's artificial edge, and an exchange every sub-step, the
composed step goes from `3.7e-4` to `1.4e-6` against an identical monolith.

**All three are necessary and none is sufficient.**  That is why each was
measured alone first and each looked like a failure: the halo alone buys 28%, the
partition of unity alone buys nothing, and the global projection alone makes it
worse.

What the probe measures, stated exactly
---------------------------------------

The trace is the **normal velocity component** on a seam face -- ``u`` on an
x-normal face, ``v`` on a y-normal face -- presented to the expert through a
declared 16-mode Fourier prolongation.  The response is the scalar
Steklov-Poincare flux

    Lambda_i : w|_Gamma  ->  nu * dw/dn |_Gamma ,     n outward from Omega_i

evaluated one-sided as ``nu (w_ring - w_interior) / h``.  This is
`probed-dtn-coupling` §2.2's definition literally, and for the Laplacian it is
the positive-semidefinite orientation, which is what makes §4.2's passivity
eigenvalue mean what it says.  A zero-state control confirms the sign.

``flux_mode='momentum'`` switches to §4.1's conservative co-normal
``nu dw/dn - (u.n) w``.  **W47 settled it 2026-08-27 and §2.2's form is
normative**: the two agree to `1.1e-9` on the *assembled* seam, because the
advective term cancels between the two sides, and differ by a factor `1e3` per
*block*, where §4.1's form reports a transport term as a passivity defect.  The
momentum form is kept as the seam-level equivalent it is, never as the block one.
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

# ---------------------------------------------------------------------------
# resolving the expert, which lives in the other repository
# ---------------------------------------------------------------------------

DEFAULT_BUILD_REPO = r"C:\Users\Nauni\physics-foundation-model"

#: Wind-farm case-study constants, carried over unchanged so the numbers here are
#: comparable with the ones already in the vault.  ``nu`` is `couple.py`'s
#: default 1/255; ``U_INF`` is 1 in the nondimensionalization the case uses.
NU = 1.0 / 255.0
U_INF = 1.0

#: The grid.  ``h`` is the expert's native spacing and never changes; the window
#: size does, because the halo is the thing under test.
H = 2.0 / 128.0
MONO_N = 255
MONO_L = MONO_N * H

#: `probed-dtn-coupling` §2.2: a seam truncated at the expert's own measured
#: spectral cutoff gives 16 modes.  The multiplier space and the probe basis are
#: the same space.
M_EFF = 16

#: The spec's macro-step (`spec-wind-farm-wake-atlas-0.1` §8.1).  The wind-farm
#: code runs 0.25, which OP-2 records as "five notches off spec".
MACRO_DT = 0.05

#: `WindowNS` is Heun on a 3-point stencil, so one sub-step moves information two
#: cells, and it takes ten sub-steps per macro-step at MACRO_DT on this state.
#: These two are what ``required_halo()`` multiplies, and they are declared rather
#: than inferred because the compiler cannot see inside the expert.
STENCIL_RADIUS = 2
SUBSTEPS = 10


def substeps_at(dt: float, solver=None) -> int:
    """The agent's OWN internal sub-step count at ``dt``, which R10b must match.

    **`SUBSTEPS = 10` above is the value at `MACRO_DT` and nowhere else**, and
    that was measured the hard way on 2026-08-28: at ``dt = 0.025`` the monolith
    takes 5 internal sub-steps while a composition hard-coded to 10 exchanges
    takes 10, and the composed macro-step's ``tau`` goes from 8.0e-7 to 1.6e-4 --
    two orders, charged to the agent, with sigma unchanged at 1e-7 and every
    other diagnostic healthy. Matching the cadence restores it, and the
    improvement over the as-built scheme is then 361x at dt = 0.025, 217x at
    0.05 and 138x at 0.10.

    ``SUBSTEPS`` is kept because it is what the capability record declares at the
    spec's macro-step, and because ``required_halo()`` is quoted against it.
    Anything that varies ``dt`` must call this instead.
    """
    if solver is None:
        solver = load_reference().WindowNS(nu=NU, length=MONO_L, n=MONO_N, cfl=0.4,
                                           transmission="dirichlet")
    if getattr(solver, "last_substeps", None) and dt == MACRO_DT:
        return int(solver.last_substeps)
    return max(1, int(round(SUBSTEPS * dt / MACRO_DT)))

MECH_SCALES = {"stress": U_INF**2, "velocity": U_INF, "power_area": U_INF**3}

FACES = ("xlo", "xhi", "ylo", "yhi")

#: Outward normal sign along the face's own axis, and the axis itself.
_FACE_GEOM = {
    "xlo": (-1.0, "x"),
    "xhi": (+1.0, "x"),
    "ylo": (-1.0, "y"),
    "yhi": (+1.0, "y"),
}


def build_repo() -> str:
    return os.environ.get("ATLAS_BUILD_REPO", DEFAULT_BUILD_REPO)


@lru_cache(maxsize=1)
def load_reference(module_name: str = "atlas_windfarm_reference"):
    """Import `reference.py` from the build repo under a private module name.

    Both repositories have a top-level package called ``atlas`` and they are
    different packages, so the build repo cannot simply go on ``sys.path``.  The
    windfarm subpackage is loaded under a name of its own instead, which keeps
    the relative imports inside it working and collides with nothing.
    """
    import importlib
    import importlib.util
    import sys

    pkg_dir = os.path.join(build_repo(), "src", "atlas", "cases", "windfarm")
    init = os.path.join(pkg_dir, "__init__.py")
    if not os.path.isfile(init):
        raise RuntimeError(
            f"the expert is not where this case study expects it: {init!r} does not "
            "exist. reference.WindowNS lives in the build repo "
            "(github.com/nonidino/physics-foundation-model, src/atlas/cases/windfarm/); "
            "set ATLAS_BUILD_REPO to the checkout."
        )
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            module_name, init, submodule_search_locations=[pkg_dir]
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
    return importlib.import_module(module_name + ".reference")


@lru_cache(maxsize=4)
def _no_projection_class():
    """`WindowNS` with the elliptic part removed, for ``elliptic_subsolve=exposed``.

    Subclassing rather than editing the expert is deliberate: the agent is not
    ours to change, and what the composition layer is allowed to do is *decline
    to use* a part of it and supply that part itself.  The velocity update is
    untouched.
    """
    ref = load_reference()

    class NoProjectionWindowNS(ref.WindowNS):
        def _project(self, u, v):
            return u, v

    return NoProjectionWindowNS


# ---------------------------------------------------------------------------
# the tiling, the halo, and the partition of unity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tiling:
    """A 2x2 tiling of the monolith by windows of ``n`` cells, overlapping.

    ``halo = 2n - MONO_N`` is the overlap in cells and it is the quantity R10's
    halo rule checks.  ``ramp`` is how far the partition of unity takes to fall
    from one to zero at a window's *artificial* edge -- a domain edge is a real
    boundary and keeps full weight there.
    """

    n: int = 138
    ramp: int = 8

    @property
    def halo(self) -> int:
        return 2 * self.n - MONO_N

    @property
    def offsets(self) -> list[tuple[int, int]]:
        o = MONO_N - self.n
        return [(0, 0), (o, 0), (0, o), (o, o)]

    @property
    def names(self) -> list[str]:
        return ["W00", "W10", "W01", "W11"]

    def artificial_faces(self, ox: int, oy: int) -> tuple[str, ...]:
        """The faces of this window that are NOT the domain boundary."""
        out = []
        if ox > 0:
            out.append("xlo")
        if ox + self.n < MONO_N:
            out.append("xhi")
        if oy > 0:
            out.append("ylo")
        if oy + self.n < MONO_N:
            out.append("yhi")
        return tuple(out)

    def cut(self, field_full: np.ndarray) -> np.ndarray:
        return np.stack([field_full[oy:oy + self.n, ox:ox + self.n]
                         for ox, oy in self.offsets])

    @lru_cache(maxsize=8)
    def weights(self) -> list[np.ndarray]:
        """chi_i on the global grid, normalized to a partition of unity.

        The shape is a squared linear ramp, zero AT each artificial edge.  Zero
        at the edge is the load-bearing part: `chi = 1/(owner count)` gives a
        window full weight exactly where its own solution is worst, which is what
        the first measurement was reading as agent infidelity.
        """
        raw = []
        for ox, oy in self.offsets:
            r = max(self.ramp, 1)
            idx = np.arange(self.n) + 0.5
            wx = np.ones(self.n)
            wy = np.ones(self.n)
            if ox > 0:
                wx = np.minimum(wx, np.clip(idx / r, 0.0, 1.0))
            if ox + self.n < MONO_N:
                wx = np.minimum(wx, np.clip((self.n - idx) / r, 0.0, 1.0))
            if oy > 0:
                wy = np.minimum(wy, np.clip(idx / r, 0.0, 1.0))
            if oy + self.n < MONO_N:
                wy = np.minimum(wy, np.clip((self.n - idx) / r, 0.0, 1.0))
            w = np.zeros((MONO_N, MONO_N))
            w[oy:oy + self.n, ox:ox + self.n] = np.minimum(wy[:, None], wx[None, :]) ** 2
            raw.append(w)
        total = np.sum(raw, axis=0)
        total = np.where(total <= 0.0, 1.0, total)
        return [w / total for w in raw]

    def assemble(self, us: np.ndarray, vs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        au = np.zeros((MONO_N, MONO_N))
        av = np.zeros((MONO_N, MONO_N))
        ws = self.weights()
        for k, (ox, oy) in enumerate(self.offsets):
            chi = ws[k][oy:oy + self.n, ox:ox + self.n]
            au[oy:oy + self.n, ox:ox + self.n] += chi * us[k]
            av[oy:oy + self.n, ox:ox + self.n] += chi * vs[k]
        return au, av

    def contaminated(self, d_cells: int) -> list[np.ndarray]:
        """W49: cells within ``d_cells`` of a window's own ARTIFICIAL face.

        A face that coincides with the domain boundary is a real boundary and
        carries no stale datum, so it does not contaminate anything.  ``d_cells``
        is the agent's domain of dependence over ONE EXCHANGE INTERVAL --
        ``stencil_radius * substeps_per_exchange`` -- which for the split-step
        scheme is `STENCIL_RADIUS` alone, because it exchanges every sub-step.
        This is geometry and the case study owns it; `assembly.py` consumes it.
        """
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

    def partition_of_unity(self, d_cells: int = STENCIL_RADIUS):
        """The L6 object, in the index form a 65025-cell grid can afford."""
        from ..assembly import GridPartitionOfUnity

        idx, wts, bad = {}, {}, {}
        ws = self.weights()
        cont = self.contaminated(d_cells)
        for k, (ox, oy) in enumerate(self.offsets):
            rows, cols = np.meshgrid(np.arange(oy, oy + self.n),
                                     np.arange(ox, ox + self.n), indexing="ij")
            flat = (rows * MONO_N + cols).reshape(-1)
            idx[self.names[k]] = flat
            wts[self.names[k]] = ws[k][oy:oy + self.n, ox:ox + self.n].reshape(-1)
            bad[self.names[k]] = cont[k].reshape(-1)
        return GridPartitionOfUnity(MONO_N * MONO_N, idx, wts, contaminated=bad)


#: The configuration the measurements are reported at: halo 21 covers the
#: 20-cell domain of dependence (STENCIL_RADIUS x SUBSTEPS) with one cell spare.
DEFAULT_TILING = Tiling(n=138, ramp=8)


# ---------------------------------------------------------------------------
# the interface basis, and the prolongation that presents it to the expert
# ---------------------------------------------------------------------------


def fourier_basis(n_cells: int, m: int = M_EFF) -> np.ndarray:
    """(n_cells, m) real Fourier modes, orthonormal in the h-weighted pairing.

    ``sum_j phi_k(y_j) phi_l(y_j) h = delta_kl`` exactly on the midpoint grid, so
    the V-space Gram is ``h I``, the M-space Gram is the identity, and the forced
    adjoint ``R = G_M^-1 P^T G_V = h P^T`` needs no solve.
    """
    length = n_cells * H
    y = (np.arange(n_cells) + 0.5) * H
    cols = [np.full(n_cells, 1.0 / np.sqrt(length))]
    k = 1
    while len(cols) < m:
        w = 2.0 * np.pi * k * y / length
        cols.append(np.sqrt(2.0 / length) * np.cos(w))
        if len(cols) < m:
            cols.append(np.sqrt(2.0 / length) * np.sin(w))
        k += 1
    return np.column_stack(cols[:m])


def face_prolongation(agent_id: str, port_name: str, n_cells: int) -> Prolongation:
    """P : M (16 Fourier modes) -> V (n_cells raw face cells).

    Declared rather than derived, which is the point: the reduction is then
    *forced* as ``P^*`` by `transfer.py` and the modal projection is nowhere
    written by hand.
    """
    return Prolongation(
        agent_id=agent_id,
        port_name=port_name,
        matrix=fourier_basis(n_cells),
        gram_V=H * np.eye(n_cells),
        label="16-mode real Fourier basis (probed-dtn-coupling 2.2)",
    )


# ---------------------------------------------------------------------------
# the expert: one window, its base state, and its boundary channel
# ---------------------------------------------------------------------------


@dataclass
class WindowAgent:
    """One `WindowNS` window with a base state, exposed as a boundary response.

    The base state is not a detail.  ``Lambda_i`` is the derivative of a
    *nonlinear* operator, so it exists only at a state -- `probed-dtn-coupling`
    §7 says this plainly, and every number a probe returns is local to the state
    it was taken at.  ``probe_state`` on the compile records which one.
    """

    agent_id: str
    u0: np.ndarray                    # [n, n], axis 0 is y, axis 1 is x
    v0: np.ndarray
    shared_faces: tuple[str, ...]
    dt: float = MACRO_DT
    nu: float = NU
    flux_mode: str = "diffusive"      # 'diffusive' (2.2, normative) or 'momentum' (4.1)
    transmission: str = "dirichlet"
    expose_elliptic: bool = False     # take the projection out of the agent
    n_calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.flux_mode not in ("diffusive", "momentum"):
            raise ValueError(self.flux_mode)
        self.u0 = np.asarray(self.u0, dtype=float)
        self.v0 = np.asarray(self.v0, dtype=float)
        self.n = int(self.u0.shape[0])
        cls = _no_projection_class() if self.expose_elliptic else load_reference().WindowNS
        self._solver = cls(nu=self.nu, length=self.n * H, n=self.n,
                           cfl=0.4, transmission=self.transmission)
        self.h = self._solver.h

    # -- face bookkeeping --------------------------------------------------

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

    def _normal_field(self, face: str, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return u if _FACE_GEOM[face][1] == "x" else v

    # -- the boundary channel ---------------------------------------------

    def respond(self, port_name: str, trace: np.ndarray) -> np.ndarray:
        """(port, trace) -> flux.  The only callable the probe needs.

        ``trace`` is a *perturbation* of the base ring's normal velocity.  The
        probe differences against its own zero-probe, so an incremental trace and
        an absolute one give the same operator -- and the incremental one is the
        object §4.2's incremental passivity is about.
        """
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

        u1, v1 = self._solver.step_batch(
            self.u0[None], self.v0[None], self.dt,
            bc0=(r_u[None], r_v[None]), bc1=None,
        )
        self.n_calls += 1
        w = self._normal_field(face, u1[0], v1[0])
        # Outward one-sided normal derivative: (boundary - one cell inward) / h
        # holds for all four faces, the inward direction flipping with the face.
        flux = self.nu * (w[ring] - w[interior]) / self.h
        if self.flux_mode == "momentum":
            s = _FACE_GEOM[face][0]
            flux = flux - (s * w[ring]) * w[ring]
        return flux

    # -- declared properties ----------------------------------------------

    def storage(self, u: Any = None, v: Any = None) -> float:
        """Kinetic energy. A real one: `WindowNS` conserves it under the skew form."""
        if u is None:
            u, v = self.u0, self.v0
        u = np.asarray(u, dtype=float)
        v = self.v0 if v is None else np.asarray(v, dtype=float)
        return 0.5 * float(np.sum(u * u + v * v)) * self.h**2

    def validity(self, state: Any = None, cond: Any = None) -> bool:
        """Not falsified on the suite this expert was validated against.

        The stated envelope is finite fields and a cell Reynolds number at or
        below the 4 the class documents, doubled for headroom.  Above it the
        centred advection is outside the regime the skew form was measured stable
        in, and the honest answer is a declination rather than a number.
        """
        if state is None:
            u, v = self.u0, self.v0
        else:
            u, v = state
        u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
        if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
            return False
        return bool(self.h * float(np.max(np.hypot(u, v))) / self.nu <= 8.0)


# ---------------------------------------------------------------------------
# the capability record
# ---------------------------------------------------------------------------


def window_capabilities(expert: WindowAgent) -> ExpertCapabilities:
    """The L1 record for one window, with every field either measured or absent.

    ``differentiable`` is `NONE`, and that is a **tested** finding rather than an
    omission.  The torch backend is an array backend, not an autodiff path:
    ``step_batch`` ends with ``self.b.to_numpy(u)`` -- a deliberate decision, so
    that "the assembly, the ledger, the metrics and every existing test are
    untouched" -- which severs the tape at the class boundary whatever happens
    inside.  Measured: with ``backend='torch'`` the call still returns
    ``numpy.ndarray``, and ``torch.func.jvp`` through it raises ``Cannot access
    data pointer of Tensor that doesn't have storage``.  So §5.2's "second unspent
    use of the differentiability claim" stays unspent, and the reason is one line
    of a port written for an unrelated and good reason.

    ``elliptic_subsolve`` is the field the 2026-08-27 measurement added, and it is
    the one that decides whether this graph compiles.
    """
    ports = [
        port_decl(
            name=f"{f}:MECH",
            port_type=PortType.MECH,
            geometry=f"{f} face of {expert.agent_id}",
            direction=Direction.BIDIRECTIONAL,
            nondim=dict(MECH_SCALES),
            # W66 / L3-C9: `respond` returns nu du/dn, the viscous traction --
            # the MECH EFFORT. The imposed trace is the velocity, the FLOW.
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
        # Heun with internal sub-stepping. This is what R2b gates the rung on,
        # and declaring it is what stops the compiler recommending probed-DtN.
        time_discretization=TimeDiscretization.EXPLICIT,
        differentiable=Differentiable.NONE,
        dt_native=expert.dt,
        L_native=expert.n * H,
        regime_law=lambda T_s, nu_p, L: T_s / (nu_p * L**2),
        storage=expert.storage,
        # Nothing has measured an equivariance here, so nothing is declared. The
        # conformance suite tests declarations, and an undeclared equivariance is
        # untested rather than assumed absent.
        equivariances=(),
        validity=expert.validity,
        governing_family="incompressible-navier-stokes-2d",
        lambda_ref=None,
        claim_types=frozenset({ClaimType.TRAJECTORY}),
        weight_hash=(f"windowns-nu{expert.nu:.6g}-n{expert.n}-{expert.transmission}"
                     f"-{expert.flux_mode}-{'exposed' if expert.expose_elliptic else 'embedded'}"),
        boundary_response=expert.respond,
        boundary_response_jvp=None,
        # Deterministic to the last bit, so the finite-difference floor is machine
        # epsilon and NOT the neural checkpoint's 1e-6 batch-position dependence.
        reproducibility_floor=np.finfo(float).eps,
        deterministic=True,
        note="reference.WindowNS, one window, Dirichlet ring, real forward pass",
    )


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------

#: seam_id -> ((agent, face), (agent, face)).  The two vertical seams and the two
#: horizontal ones; the four windows meet at a cross-point in the middle, which is
#: declared rather than hidden.
SEAMS = {
    "sx0": (("W00", "xhi"), ("W10", "xlo")),
    "sx1": (("W01", "xhi"), ("W11", "xlo")),
    "sy0": (("W00", "yhi"), ("W01", "ylo")),
    "sy1": (("W10", "yhi"), ("W11", "ylo")),
}

#: The constants W1/W2/W3 measured for the split-step scheme, 2026-08-27.
#: Re-measured after the R10 fix; see `tier0-measurements` and `out/tier0b/`.
MEASURED_SPLIT_STEP = MeasuredConstants(
    L=0.979644, L_stderr=0.00041,
    tau=1.3353e-06, sigma=3.7334e-08, gamma=0.0,
    norm_A=1.0,
    # W49, measured 2026-08-28. This is C_mu for the OVERLAPPING branch of the
    # master bound -- `sigma <= C_mu * Pi * ||d_lambda||`, master-error-bound 4.1
    # -- and it is not the same constant as section 4's, which is the
    # substructuring branch's and is what the 2.2e-8 diagnosis was about. Over 11
    # configurations spanning halo 1 to 61 and Pi 0.02 to 1.0, the implied value
    # stayed in [0.297, 1.178]; 1.2 is the sampled maximum rounded up.
    C_mu=1.2,
    # L2/C2, measured 2026-08-28. `|| sum_i chi_i |E_i R_i u - R_i E u| ||` over
    # ONE exchange interval, relative, on the joint (u, v) field -- the cut's own
    # contribution to the composed defect, and the quantity that replaced the cut
    # score. Measured against the actual one-interval defect of 2.6216e-07, so
    # the bound is tight to 0.2%; the max-over-agents form of the same bound is
    # 5.7525e-05, i.e. 220x loose, and the difference is entirely what the ramped
    # partition of unity buys. See `tier0-measurements` section 10.
    cut_defect_bound=2.6268e-07,
    # **W58, 2026-08-30.** WHICH of the two definitions the number above is. It
    # is the chi-weighted one, which is why it needed a monolith and why the
    # 0.2% tightness above could be quoted at all: the reference-free surrogate
    # has nothing to be tight against. This graph's contaminated sets share 1120
    # cells, so the two are NOT provably equal here -- they agreed to 1.00007
    # anyway, and L2/C2/W58 now says which fact is which.
    cut_defect_bound_form="chi-weighted",
    probe_state="developed wake, t = 5, dt = 0.05, nu = 1/255",
    scheme="split-step, halo 21, ramp 8, exchange every sub-step (overlapping branch)",
    depth=0,
    source="scripts/tier0_window_ns.py, scripts/w49_sigma_halo.py, "
           "scripts/w16_cut_policy.py, out/tier0b/, out/w16/",
)

#: The same graph with the agent as it actually is. Kept because the contrast is
#: the finding: every constant is worse, most of them by two orders.
MEASURED_AS_BUILT = MeasuredConstants(
    L=0.972776, L_stderr=0.00043,
    tau=2.9347e-04, sigma=2.2532e-05, gamma=0.0,
    norm_A=1.0, C_mu=None,
    probe_state="developed wake, t = 5, dt = 0.05, nu = 1/255",
    scheme="as-built, halo 21, one exchange per macro-step, chi = 1/(owner count)",
    depth=0,
    source="scripts/tier0_window_ns.py, out/tier0b/",
)


def make_experts(u_full, v_full, tiling: Tiling = DEFAULT_TILING,
                 dt: float = MACRO_DT, flux_mode: str = "diffusive",
                 nu: float = NU, expose_elliptic: bool = False):
    us, vs = tiling.cut(u_full), tiling.cut(v_full)
    out = {}
    for k, name in enumerate(tiling.names):
        ox, oy = tiling.offsets[k]
        out[name] = WindowAgent(
            agent_id=name, u0=us[k], v0=vs[k],
            shared_faces=tiling.artificial_faces(ox, oy),
            dt=dt, nu=nu, flux_mode=flux_mode, expose_elliptic=expose_elliptic,
        )
    return out


def build(
    u_full: np.ndarray,
    v_full: np.ndarray,
    mode: str = "split-step",
    tiling: Tiling = DEFAULT_TILING,
    dt: float = MACRO_DT,
    flux_mode: str = "diffusive",
    nu: float = NU,
    measured: MeasuredConstants | None = "default",
    experts: dict[str, WindowAgent] | None = None,
) -> tuple[CaseGraph, dict[str, WindowAgent]]:
    """The four-window graph at a given state, plus the experts backing it.

    ``mode='as-built'``  the agent keeps its own pressure solve; L2 refuses (R10)
    ``mode='split-step'`` the elliptic part is the composition layer's; it compiles

    Returns the experts too, because a real case study's agents own solver state
    and the caller needs the same objects the graph was built from.
    """
    if mode not in ("as-built", "split-step"):
        raise ValueError(mode)
    expose = mode == "split-step"
    if measured == "default":
        measured = MEASURED_SPLIT_STEP if expose else MEASURED_AS_BUILT
    experts = experts or make_experts(u_full, v_full, tiling, dt, flux_mode, nu, expose)
    agents = [Agent(n, window_capabilities(experts[n]), domain=f"tile {n}")
              for n in tiling.names]
    connections = [
        Connection(
            seam_id=seam_id,
            a=(a_id, f"{a_face}:MECH"),
            b=(b_id, f"{b_face}:MECH"),
            port_type=PortType.MECH,
            derive_space=True,
            geometrically_coincident=True,
            # n_0(Gamma) = 0, corrected 2026-08-27 from a measurement.
            #
            # interface-transfer-theory 7 gives 1 for a fluid-fluid MECH seam,
            # from incompressibility constraining the trace to zero net flux.
            # That is right only when the incompressibility constraint is INSIDE
            # Lambda_i -- and probed-dtn-coupling 2.1 puts the elliptic part
            # outside it. So n_0 is not a property of the seam alone: it depends
            # on where the elliptic solve lives, which is exactly what
            # ``elliptic_subsolve`` declares. With the projection in the
            # composition layer, Lambda_i is the pure advection-diffusion DtN and
            # is nonsingular: measured 0 on four seams, in every control, with no
            # spectral gap at the bottom (kappa = 1.10 to 1.36).
            expected_null_dim=0,
            note="seam presented through a declared 16-mode Fourier prolongation",
        )
        for seam_id, ((a_id, a_face), (b_id, b_face)) in SEAMS.items()
    ]
    return (
        CaseGraph(
            name=f"window-ns-2x2-{mode}",
            agents=agents,
            connections=connections,
            decomposition=Decomposition.OVERLAPPING,
            overlap=tiling.halo * H,
            overlap_cells=tiling.halo,
            partition_of_unity=tiling.partition_of_unity(),
            global_fields=[
                GlobalField(
                    "pressure",
                    note=("the elliptic part. Global under mode='split-step', which is "
                          "what probed-dtn-coupling 2.1 assumes; per-window under "
                          "mode='as-built', which is what R10 refuses"),
                ),
            ],
            cross_points=("centre",),
            macro_dt=dt,
            measured=measured,
            note=(f"four reference.WindowNS windows of {tiling.n} cells tiling "
                  f"{MONO_N}x{MONO_N} at h = 1/64, halo {tiling.halo} cells; the "
                  f"monolithic reference is the same class at n = {MONO_N}"),
        ),
        experts,
    )
