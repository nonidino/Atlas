"""W149 -- Tier 32.  Is a globally receptive operator uncontrollable, or unconfined?

Two gates and their controls.  No optimiser is built here, `compiler.py` is not
touched, and no `Accelerator` member is added: this script is ONE measurement and
the controls that make it readable.  If it comes back the wrong way the rest of
the idea is dead, and a session is the right price to find that out.

THE ARGUMENT THIS TESTS
-----------------------

`master-error-bound` 4.1 bounds the transmission error of an OVERLAPPING scheme:

    sigma <= C_mu * Pi * ||d_lambda||,
    Pi    =  max_j sum_i chi_ij * 1[ b_i(j) <= d_i ]

with ``d_i`` the agent's domain of dependence over one exchange interval.  A
globally receptive agent has ``d_i = infinity``, so the indicator is identically
1 and ``Pi = max_j sum_i chi_ij = 1`` EXACTLY -- pinned by the partition-of-unity
identity, for every halo width and every ramp.  The lever that took ``sigma``
from 1.20e-4 to 6.26e-9 (ramp 1 -> ramp 21 at fixed halo, Pi 0.75 -> 0.020) is
not weakened; it is unavailable.

So the bound is not violated, it is **vacuous**, and exactly one factor is left
free: ``||d_lambda||``, the staleness of the artificial-boundary datum.  Virtual
control -- solve for the boundary datum instead of lagging it -- is the mechanism
for that other factor.  Whether it is available is a question about
**observability**: does the jump between two agents on their overlap actually
move when the seam datum moves, and how well conditioned is that map?

    Confinement and observability are different properties, not ordered by
    strictness.  Global receptivity is fatal to the first and neutral-or-
    favourable to the second.

`probe.support_reach`'s 128-of-128 result is the NUMERATOR of that argument and
not the whole of it.  Nonzero everywhere is not well-conditioned everywhere; a
response can be dense and nearly rank-deficient.  ``beta_ctrl`` is what closes
that gap, and nothing in `support_reach` establishes it.

GATE 0 -- the objective, before anything else
---------------------------------------------

`probed-dtn-coupling` 2.3 killed flux balance for an explicit one-step map, and
the way it died is the standard this objective has to clear: the reference trace
did not reduce the residual (2.6421e-5 -> 2.6520e-5, WORSE), solving overshot
41x/75x/150x as ``dt`` refined, and the identity turned out to be
``F_A + F_B = -nu h d2w/dn2`` -- a discrete second derivative, not a jump.  The
proposed objective is a different statement and must be shown to be:

    J(g) = 1/2 sum_{i<j} int_{Omega_i cap Omega_j} omega_ij
                | E_i[u^n, g_i] - E_j[u^n, g_j] |^2   +  (eta/2) ||g - g_lag||^2

``eta = 0`` throughout this gate, and that is not a simplification -- a Tikhonov
term is a solver aid centred ON the lagged trace, so leaving it in would hand the
lagged column a bonus and the reference column a penalty in the one comparison
the gate exists to make.

The check is `probe.reference_trace_check`, run at ``dt`` = 0.05, 0.01 and 0.002
-- three points because three points are what exposed flux balance's dt-scaling.
``J(reference) < J(lagged)`` strictly at all three, or the tier stops here.

GATE 1 -- the decisive measurement
-----------------------------------

Assemble ``T = d(jump)/dg`` on ONE seam and report its spectrum: ``sigma_min``
(this is ``beta_ctrl``), ``sigma_max``, ``kappa``, the decay profile, and the
numerical rank against two floors.  ``m + 1`` forward calls per side.  No
optimiser, no rollout.

Four agents, and the first two are controls:

  1. ``reference.SpectralNS`` (periodic, ``bc_channel = none``).  MUST return T
     bitwise zero.  `Xi` already reproduces as exactly zero with 34 solver calls
     on this configuration, so this control has a known answer -- and if it does
     not come back the instrument is broken and nothing else here means anything.
  2. ``reference.WindowNS``, elliptic EXPOSED.  A known-good boundary channel;
     this is the scale ``beta_ctrl`` is read against.
  3. ``reference.WindowNS``, elliptic EMBEDDED.  `probed-dtn-coupling` 2.1
     measured ``n_0 = 1`` with incompressibility inside ``Lambda_i`` and 0 with
     it exposed, so a rank-one null space is predicted.  A measured null
     dimension other than 1 indicts the implementation, not the physics.
  4. Poseidon-T.  The number the tier exists for.

CADENCE, stated because it is a real choice and it moves numbers
----------------------------------------------------------------

**One exchange per macro-step, for all four cases.**  It is the only cadence all
four share -- Poseidon-T is a one-shot map at a fixed lead time and cannot
sub-step at all -- and it is the cadence `WindowAgent.respond` uses, so every
``Lambda`` in this vault was probed at it.

Two consequences, both stated rather than discovered later.  First, the EXPOSED
column here is NOT the split-step scheme of `tier0-measurements`, which exchanges
every sub-step (R10b); its numbers are not comparable to that scheme's
``sigma = 3.73e-8``.  Second, at this cadence ``d = stencil_radius * substeps =
20`` cells against a 21-cell overlap, so the classical agent's CONFINEMENT
advantage is nearly absent here too.  That is the right way round for this tier:
holding confinement roughly equal across the columns is what leaves observability
as the thing being compared.

Run:

    python scripts/w149_control_observability.py                 # gates 0 and 1
    python scripts/w149_control_observability.py --part gate0
    python scripts/w149_control_observability.py --no-poseidon
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

# torch and numpy each bring their own OpenMP on Windows; without this the
# process aborts at Poseidon-T's first call.  Set before ANY import that could
# reach torch, which is why it sits above the atlas imports rather than beside
# the checkpoint that needs it.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from atlas.cases import window_ns as W                             # noqa: E402
from atlas.probe import reference_trace_check                      # noqa: E402

#: Which velocity component a face's declared MECH port carries.  The port is the
#: NORMAL one -- that is the effort-flow pair `port-algebra-atlas-0.1` declares --
#: so the control space is narrower than the datum, which has two components on
#: every ring.  The gap is not hidden: `controllable_fraction` below measures
#: exactly how much of the jump a normal-only control can reach, and the
#: ``--both-components`` sensitivity widens it so the cost of the restriction is
#: a number rather than a caveat.
FACE_COMPONENT = {"xlo": "u", "xhi": "u", "ylo": "v", "yhi": "v"}

#: The one seam.  Everything else about both windows is held.
SEAM = "sx0"


# ---------------------------------------------------------------------------
# the two sides of one seam, with a march that returns a FIELD
# ---------------------------------------------------------------------------


class Side:
    """One window of a seam, marched one exchange interval under a ring datum.

    ``march`` is deliberately not `WindowAgent.respond` with a different return
    statement bolted on -- it builds the ring exactly as `respond` does and then
    reads the whole field instead of the flux.  `check_flux_agrees` asserts the
    flux read off this march equals `respond`'s BITWISE, which is what keeps the
    two from drifting apart while nobody is looking.
    """

    def __init__(self, agent, seam_face: str, ox: int, oy: int, n: int,
                 force=None):
        self.agent = agent
        self.seam_face = seam_face
        self.ox, self.oy, self.n = ox, oy, n
        self.force = force
        self.n_calls = 0

    # -- ring bookkeeping, borrowed from the agent so it cannot diverge ------

    def _ring(self, face):
        return self.agent._ring_index(face)

    def ring_state(self, perturb: dict[str, np.ndarray]):
        """(u, v) whose RINGS carry the boundary datum; the interior is ignored."""
        r_u, r_v = self.agent.u0.copy(), self.agent.v0.copy()
        for face, tr in perturb.items():
            ring, _ = self._ring(face)
            tr = np.asarray(tr, dtype=float)
            if tr.ndim == 1:                       # normal component only
                if FACE_COMPONENT[face] == "u":
                    r_u[ring] = r_u[ring] + tr
                else:
                    r_v[ring] = r_v[ring] + tr
            else:                                  # (2, n): both components
                r_u[ring] = r_u[ring] + tr[0]
                r_v[ring] = r_v[ring] + tr[1]
        return r_u, r_v

    # -- the march ----------------------------------------------------------

    def march(self, perturb: dict[str, np.ndarray]):
        raise NotImplementedError

    def flux(self, u1, v1, face=None):
        """The 2.2 diffusive flux, on the same convention `WindowAgent` uses."""
        face = face or self.seam_face
        ring, interior = self._ring(face)
        w = u1 if FACE_COMPONENT[face] == "u" else v1
        return self.agent.nu * (w[ring] - w[interior]) / self.agent.h

    # -- geometry -----------------------------------------------------------

    def overlap_cols(self, other: "Side"):
        """This side's LOCAL column range of the shared overlap."""
        lo = max(self.ox, other.ox)
        hi = min(self.ox + self.n, other.ox + other.n)
        return lo - self.ox, hi - self.ox


class WindowSide(Side):
    """`reference.WindowNS`, exposed or embedded, one macro-step, ring held."""

    def march(self, perturb):
        r_u, r_v = self.ring_state(perturb)
        a = self.agent
        force = None if self.force is None else (self.force[None],
                                                 np.zeros_like(self.force)[None])
        u1, v1 = a._solver.step_batch(a.u0[None], a.v0[None], a.dt,
                                      bc0=(r_u[None], r_v[None]), bc1=None,
                                      force=force)
        self.n_calls += 1
        return u1[0], v1[0]


class SpectralSide(Side):
    """`reference.SpectralNS`: the trace is built, and there is nowhere to put it.

    The positive control.  A periodic window has no boundary, so ``step`` takes
    no boundary argument -- the ring is constructed exactly as the Dirichlet side
    constructs it and then dropped, which is the honest depiction of an agent
    that is OFFERED a datum and cannot accept one.
    """

    def march(self, perturb):
        _unused_ring_datum = self.ring_state(perturb)
        a = self.agent
        force = None if self.force is None else (self.force[None],
                                                 np.zeros_like(self.force)[None])
        u1, v1 = a._solver.step(a.u0[None], a.v0[None], a.dt, force=force)
        self.n_calls += 1
        return u1[0], v1[0]


class PoseidonSide(Side):
    """Poseidon-T: the ring is written into the INITIAL CONDITION and run once.

    W59 -- for a one-shot map that is what a Dirichlet channel degenerates to,
    and the record says `DIRICHLET` because it is the closest available value.
    No forcing: the checkpoint has no force channel.
    """

    def march(self, perturb):
        r_u, r_v = self.ring_state(perturb)
        u1, v1 = self.agent.step(r_u, r_v)
        self.n_calls += 1
        return np.asarray(u1, dtype=float), np.asarray(v1, dtype=float)


# ---------------------------------------------------------------------------
# the jump, and the objective built on it
# ---------------------------------------------------------------------------


def jump_vector(A: Side, B: Side, fa, fb, weight: np.ndarray | None = None):
    """sqrt(omega) * (E_A - E_B) on the overlap, both components, area-normed.

    Scaling by ``h`` is what makes ``||j||_2`` the L2 norm over the overlap area
    rather than a grid-count-dependent number, so ``sigma_min(T)`` is an operator
    norm from a unit-normalised M control to a physical field norm and can be
    compared across the three cases without a hidden resolution factor.
    """
    a0, a1 = A.overlap_cols(B)
    b0, b1 = B.overlap_cols(A)
    du = fa[0][:, a0:a1] - fb[0][:, b0:b1]
    dv = fa[1][:, a0:a1] - fb[1][:, b0:b1]
    if weight is not None:
        du, dv = du * weight, dv * weight
    return A.agent.h * np.concatenate([du.reshape(-1), dv.reshape(-1)])


def objective(A: Side, B: Side, pa: dict, pb: dict, weight=None) -> float:
    """J at eta = 0.  One march per side."""
    j = jump_vector(A, B, A.march(pa), B.march(pb), weight)
    return 0.5 * float(j @ j)


# ---------------------------------------------------------------------------
# T -- the linearised control-to-jump map
# ---------------------------------------------------------------------------


def control_basis(n_cells: int, m: int, both_components: bool) -> np.ndarray:
    """The declared 16-mode Fourier prolongation, optionally on both components.

    `window_ns.fourier_basis` is orthonormal in the h-weighted face pairing, so a
    unit vector in M is a unit-norm trace in V and ``sigma_min`` needs no further
    normalisation.
    """
    P = W.fourier_basis(n_cells, m)
    if not both_components:
        return P                                   # (n, m)
    Z = np.zeros_like(P)
    return np.stack([np.concatenate([P, Z], axis=1),
                     np.concatenate([Z, P], axis=1)])   # (2, n, 2m)


def assemble_T(A: Side, B: Side, eps: float, m: int = W.M_EFF,
               both_components: bool = False, weight=None,
               hold: dict | None = None):
    """T = d(jump)/dg on one seam.  ``m + 1`` marches per side.

    ``hold`` supplies the lagged datum on every artificial face that is NOT on
    this seam; those are held fixed, which is what makes this ONE seam's map.
    """
    hold = hold or {}
    n = A.n
    basis = control_basis(n, m, both_components)
    ncol = (basis.shape[-1])

    # The zero-probe of each side is the OTHER side's fixed argument, so the
    # trace-independent part of both agents cancels in every column exactly as
    # `_columns_finite_difference`'s base probe makes it cancel in Lambda.
    base_A = A.march(dict(hold.get(A, {})))
    base_B = B.march(dict(hold.get(B, {})))
    j0 = jump_vector(A, B, base_A, base_B, weight)

    cols = []
    for side, other, base_other in ((A, B, base_B), (B, A, base_A)):
        pert = dict(hold.get(side, {}))
        for k in range(ncol):
            g = basis[:, :, k] if both_components else basis[:, k]
            p = dict(pert)
            p[side.seam_face] = eps * g
            plus = side.march(p)
            jp = (jump_vector(A, B, plus, base_B, weight) if side is A
                  else jump_vector(A, B, base_A, plus, weight))
            cols.append((jp - j0) / eps)
    return np.column_stack(cols), j0


def spectrum(T: np.ndarray, noise: float | None = None) -> dict:
    """beta_ctrl and everything read off the same SVD."""
    sv = np.linalg.svd(T, compute_uv=False)
    smax = float(sv[0]) if sv.size else 0.0
    scale = max(smax, 1e-300)
    tol_mach = max(T.shape) * np.finfo(float).eps * scale
    out = {
        "shape": list(T.shape),
        "sigma_max": smax,
        "beta_ctrl": float(sv[-1]) if sv.size else 0.0,
        "kappa": float(sv[0] / sv[-1]) if sv.size and sv[-1] > 0 else float("inf"),
        "profile": [float(s / scale) if scale else 0.0 for s in sv],
        "singular_values": [float(s) for s in sv],
        "tol_machine": float(tol_mach),
        "rank_machine": int(np.sum(sv > tol_mach)),
        "null_dim_machine": int(np.sum(sv <= tol_mach)),
        "exactly_zero": bool(np.all(T == 0.0)),
        "max_abs": float(np.abs(T).max()) if T.size else 0.0,
        "frobenius": float(np.linalg.norm(T, "fro")),
    }
    if noise is not None:
        # NOT a noise floor.  The solver is deterministic to machine epsilon,
        # so this is the spread of T between two finite-difference steps a
        # decade apart -- i.e. the LINEARISATION error of a nonlinear map,
        # which is the real floor a derivative-based rank has to clear.
        out["tol_eps_spread"] = float(noise)
        out["rank_above_eps_spread"] = int(np.sum(sv > noise))
        out["null_dim_above_eps_spread"] = int(np.sum(sv <= noise))
    # A near-null direction is invisible to a machine-epsilon count.  The
    # predicted rank-one incompressibility direction is not expected to be
    # exactly singular -- the projection is discrete and the trace is
    # prolonged from 16 modes -- so the honest instrument is the largest
    # RATIO drop at the bottom of the spectrum, reported with the index it
    # occurs at rather than thresholded.
    if sv.size >= 3:
        ratios = [float(sv[k + 1] / sv[k]) if sv[k] > 0 else 1.0
                  for k in range(sv.size - 1)]
        k = int(np.argmin(ratios))
        out["bottom_gap_ratio"] = ratios[k]
        out["bottom_gap_index"] = k + 1
        out["null_dim_gap"] = int(sv.size - (k + 1))
    return out


def controllable_fraction(T: np.ndarray, j0: np.ndarray, tol: float) -> float:
    """||P_range(T) j0|| / ||j0||: how much of the disagreement the control reaches.

    This is the quantity the whole tier turns on that a spectrum alone does not
    give.  ``beta_ctrl > 0`` says every control direction moves the jump; this
    says whether the jump that is actually there is a jump the control can move.
    """
    nj = float(np.linalg.norm(j0))
    if nj == 0.0:
        return float("nan")
    U, sv, _ = np.linalg.svd(T, full_matrices=False)
    keep = sv > tol
    if not np.any(keep):
        return 0.0
    return float(np.linalg.norm(U[:, keep].T @ j0) / nj)


def composed_evaluator(u, v, fx, dt, expose: bool, tiling=None):
    """One composed macro-step of the WHOLE graph, against the monolith.

    **This is the check that decides the tier.**  Flux balance reduced its
    own residual and made the composed macro-step 8.9x WORSE, so a control
    that lowers J has proved nothing until the field it assembles is
    compared with the truth.  All four windows march, the DECLARED partition
    of unity assembles them, the composition layer applies the global
    projection when the elliptic part is exposed -- which is what `exposed`
    MEANS, and leaving it out was measuring a scheme nobody proposes -- the
    domain ring is pinned, and the relative L2 against the monolith comes
    back.

    Returns a dict of closures and data.  ``err`` is the composed field
    against the monolith, which at this cadence is dominated by R10b's
    splitting term and is kept only as context.  ``sigma`` is the quantity a
    boundary datum can actually move: the same composed step run against
    ITSELF under the monolith's own datum, so tau cancels.  Both take
    ``{agent: {face: trace}}``, so the seam under test moves and the other
    three stay lagged.
    """
    tiling = tiling or W.DEFAULT_TILING
    ref = W.load_reference()
    ex = W.make_experts(u, v, tiling, dt=dt, expose_elliptic=expose)
    fxs = tiling.cut(fx)
    zero = np.zeros_like(fxs[0])
    mono = ref.WindowNS(nu=W.NU, length=W.MONO_L, n=W.MONO_N, cfl=0.4,
                        transmission="dirichlet")
    ur, vr = mono.step_batch(u[None], v[None], dt, bc0=None, bc1=None,
                             force=(fx[None], np.zeros_like(fx)[None]))
    ur, vr = ur[0], vr[0]
    ring = (u.copy(), v.copy())
    den = float(np.sum(ur ** 2) + np.sum(vr ** 2))

    def field(perturb_by_agent: dict):
        us, vs = [], []
        for k, nm in enumerate(tiling.names):
            a = ex[nm]
            r_u, r_v = a.u0.copy(), a.v0.copy()
            for face, tr in (perturb_by_agent.get(nm) or {}).items():
                idx, _ = a._ring_index(face)
                tr = np.asarray(tr, dtype=float)
                if tr.ndim == 1:
                    if FACE_COMPONENT[face] == "u":
                        r_u[idx] = r_u[idx] + tr
                    else:
                        r_v[idx] = r_v[idx] + tr
                else:
                    r_u[idx] = r_u[idx] + tr[0]
                    r_v[idx] = r_v[idx] + tr[1]
            u1, v1 = a._solver.step_batch(
                a.u0[None], a.v0[None], dt, bc0=(r_u[None], r_v[None]),
                bc1=None, force=(fxs[k][None], zero[None]))
            us.append(u1[0])
            vs.append(v1[0])
        U, V = tiling.assemble(np.asarray(us), np.asarray(vs))
        if expose:
            pu, pv = mono._project(U[None], V[None])
            U, V = pu[0], pv[0]
        for Arr, R in ((U, ring[0]), (V, ring[1])):
            Arr[0, :] = R[0, :]
            Arr[-1, :] = R[-1, :]
            Arr[:, 0] = R[:, 0]
            Arr[:, -1] = R[:, -1]
        return U, V

    def err(perturb_by_agent: dict) -> float:
        U, V = field(perturb_by_agent)
        return float(np.sqrt((np.sum((U - ur) ** 2)
                              + np.sum((V - vr) ** 2)) / den))

    # the monolith's own datum on EVERY artificial face of every window --
    # both components, because a ring carries both and only the declared PORT
    # is one-component
    exact = {}
    for k, nm in enumerate(tiling.names):
        ox, oy = tiling.offsets[k]
        n = tiling.n
        du = (ur - u)[oy:oy + n, ox:ox + n]
        dv = (vr - v)[oy:oy + n, ox:ox + n]
        faces = {}
        for f in tiling.artificial_faces(ox, oy):
            idx, _ = ex[nm]._ring_index(f)
            faces[f] = np.stack([du[idx], dv[idx]])
        exact[nm] = faces
    U_ex, V_ex = field(exact)

    def sigma(perturb_by_agent: dict) -> float:
        """The composed step against the SAME composed step under exact data.

        tau cancels: both runs are the same splitting of the same graph and
        differ only in the artificial-boundary datum, which is the whole of
        what sigma is defined to be.
        """
        U, V = field(perturb_by_agent)
        return float(np.sqrt((np.sum((U - U_ex) ** 2)
                              + np.sum((V - V_ex) ** 2)) / den))

    return {"err": err, "sigma": sigma, "exact": exact,
            "u_ref": ur, "v_ref": vr}


def solve_and_check(A: Side, B: Side, T: np.ndarray, j0: np.ndarray,
                    tol: float, m: int, g_ref: np.ndarray | None,
                    weight=None, hold=None, assembly=None,
                    both: bool = False):
    """Solve the linearised problem, then MARCH the solved control.

    This is the test flux balance failed, and the reason its failure was
    fatal: solving it moved the trace 41x / 75x / 150x past the truth, and
    the overshoot GREW as dt refined.  Two numbers here, and the second is
    the one that cannot be argued with -- ``overshoot`` compares the solved
    move with the monolith's own move in the same coordinates, and
    ``J_after`` is J re-evaluated by actually MARCHING both agents at the
    solved control, so a linear prediction the nonlinear system does not
    honour shows up as a rise rather than as a small residual.
    """
    hold = hold or {}
    U, sv, Vt = np.linalg.svd(T, full_matrices=False)
    Vt = np.asarray(Vt)
    keep = sv > tol
    if not np.any(keep):
        return {"solvable": False,
                "note": "T has no direction above the floor"}
    dg = -(Vt[keep].T @ ((U[:, keep].T @ j0) / sv[keep]))
    basis = control_basis(A.n, m, both)
    per = 2 * m if both else m

    def prolong(d):
        """One side's M coefficients back to a ring trace."""
        return (np.einsum("cnk,k->cn", basis, d) if both else basis @ d)

    def spread(d):
        """The full control, as the two perturbation dicts it names."""
        pa_, pb_ = dict(hold.get(A, {})), dict(hold.get(B, {}))
        pa_[A.seam_face] = prolong(d[:per])
        pb_[B.seam_face] = prolong(d[per:2 * per])
        return pa_, pb_

    pa, pb = spread(dg)
    J_after = objective(A, B, pa, pb, weight)
    J_before = 0.5 * float(j0 @ j0)
    out = {
        "solvable": True,
        "rank_used": int(np.sum(keep)),
        "solved_move": float(np.linalg.norm(dg)),
        "J_before": J_before,
        "J_after": J_after,
        "J_ratio": J_after / J_before if J_before else float("inf"),
        "linear_residual": float(np.linalg.norm(T @ dg + j0)
                                 / np.linalg.norm(j0)),
    }
    if g_ref is not None:
        out["reference_move"] = float(np.linalg.norm(g_ref))
        out["overshoot"] = (out["solved_move"] / out["reference_move"]
                            if out["reference_move"] else float("inf"))
        # T checked as a derivative, not assumed to be one: the change in J
        # it PREDICTS at the reference control against the change a march
        # actually produces there.
        pa_r, pb_r = spread(g_ref)
        J_ref_marched = objective(A, B, pa_r, pb_r, weight)
        r = j0 + T @ g_ref
        J_ref_linear = 0.5 * float(r @ r)
        dm, dl = J_ref_marched - J_before, J_ref_linear - J_before
        out["J_at_reference_marched"] = J_ref_marched
        out["J_at_reference_linear"] = J_ref_linear
        out["derivative_check"] = (dl / dm) if dm else float("inf")
    # The unregularised move chases the part of the jump no seam control can
    # reach, so it is large by construction.  Both standard knobs are swept
    # -- Tikhonov in eta and truncated SVD in the retained rank -- and BOTH J
    # and sigma are re-evaluated by MARCHING at every point.  Sweeping sigma
    # rather than J is the whole point: choosing eta by minimising J would
    # select on the quantity that is itself under suspicion.
    sig = None if assembly is None else assembly[0]["sigma"]
    na_, nb_ = (None, None) if assembly is None else (assembly[1], assembly[2])

    def ctl(d):
        return {na_: {A.seam_face: prolong(d[:per])},
                nb_: {B.seam_face: prolong(d[per:2 * per])}}

    def score(d, extra):
        pe, pf = spread(d)
        Je = objective(A, B, pe, pf, weight)
        row = dict(extra)
        row["move"] = float(np.linalg.norm(d))
        row["J"] = Je
        row["J_ratio"] = Je / J_before if J_before else float("inf")
        if sig is not None:
            row["sigma"] = sig(ctl(d))
        return row

    sweep = []
    smax = float(sv[0])
    for e in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6):
        eta = e * smax ** 2
        d = -(Vt.T @ ((sv / (sv ** 2 + eta)) * (U.T @ j0)))
        sweep.append(score(d, {"eta_rel": e}))
    out["tikhonov_sweep"] = sweep

    # A damped Newton step and a short step along the same direction are
    # different questions, and the second is the one that says whether
    # DESCENDING J at all moves sigma the right way.
    out["step_sweep"] = [score(alpha * dg, {"alpha": alpha})
                         for alpha in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1,
                                       3e-1, 1.0)]

    ranks = sorted({1, 2, 4, 8, 16, int(sv.size)})
    tsvd = []
    for r in ranks:
        r = min(r, int(sv.size))
        d = -(Vt[:r].T @ ((U[:, :r].T @ j0) / sv[:r]))
        tsvd.append(score(d, {"rank": r}))
    out["tsvd_sweep"] = tsvd
    best = min(sweep, key=lambda r: r["J_ratio"])
    out["best_by_J_eta_rel"] = best["eta_rel"]
    out["best_by_J_ratio"] = best["J_ratio"]
    out["best_by_J_move"] = best["move"]
    if g_ref is not None and out["reference_move"]:
        out["best_by_J_overshoot"] = best["move"] / out["reference_move"]
    if assembly is not None:
        ev, na, nb = assembly
        cerr, exact = ev["err"], ev["exact"]
        # the floor a PERFECT control on this one seam could reach: the two
        # seam faces carry the monolith's own datum, the other artificial
        # faces stay lagged, so this is what is on the table for one seam
        seam_exact = {na: {A.seam_face: exact[na][A.seam_face]},
                      nb: {B.seam_face: exact[nb][B.seam_face]}}
        all_rows = (out["tikhonov_sweep"] + out["tsvd_sweep"]
                    + out["step_sweep"])
        by_sigma = min(all_rows, key=lambda r: r["sigma"])
        asm = {
            "sigma_lagged": sig({}),
            "sigma_seam_exact": sig(seam_exact),
            "sigma_all_exact": sig(exact),
            "sigma_solved": sig(ctl(dg)),
            # the best over BOTH regularisation families, selected on sigma
            # itself -- the most favourable reading the data supports
            "sigma_best_regularised": by_sigma["sigma"],
            "best_regulariser": {k: v for k, v in by_sigma.items()
                                 if k in ("eta_rel", "rank", "alpha")},
            "best_regularised_J_ratio": by_sigma["J_ratio"],
            "total_lagged": cerr({}),
            "total_solved": cerr(ctl(dg)),
        }
        if g_ref is not None:
            asm["sigma_reference_control"] = sig(ctl(g_ref))
        lag = asm["sigma_lagged"]
        asm["solved_over_lagged"] = (asm["sigma_solved"] / lag if lag
                                     else float("inf"))
        asm["floor_over_lagged"] = (asm["sigma_seam_exact"] / lag if lag
                                    else float("inf"))
        asm["best_over_lagged"] = (asm["sigma_best_regularised"] / lag
                                   if lag else float("inf"))
        # the tier turns on these two lines: is there anything on the table
        # for one seam, and does solving the objective capture any of it
        asm["reduction_available"] = 1.0 - asm["floor_over_lagged"]
        asm["reduction_captured"] = 1.0 - min(asm["solved_over_lagged"],
                                              asm["best_over_lagged"])
        asm["moves_sigma_toward_zero"] = bool(
            min(asm["sigma_solved"], asm["sigma_best_regularised"]) < lag)
        out["assembly"] = asm
    return out


def reference_control(A: Side, B: Side, u, v, fx, dt, m: int,
                      both: bool = False):
    """The monolith's own seam datum, in the SAME M coordinates the solve uses.

    Returns ``None`` when there is no same-class monolith -- which for
    Poseidon-T is permanent rather than an omission: the checkpoint is fixed
    at 128 cells and `poseidon.build` records that there is no monolith and
    that there cannot be one.
    """
    u_ref, v_ref, _mono = mono_reference(u, v, fx, dt)
    P = W.fourier_basis(A.n, m)
    R = W.H * P.T
    parts = []
    for side in (A, B):
        ox, oy, n = side.ox, side.oy, side.n
        du = (u_ref - u)[oy:oy + n, ox:ox + n]
        dv = (v_ref - v)[oy:oy + n, ox:ox + n]
        ring, _ = side._ring(side.seam_face)
        if both:
            parts.append(np.concatenate([R @ du[ring], R @ dv[ring]]))
        else:
            # the DECLARED MECH port is the normal component alone
            keep = du if FACE_COMPONENT[side.seam_face] == "u" else dv
            parts.append(R @ keep[ring])
    return np.concatenate(parts)


def null_direction_report(T: np.ndarray, m: int, both: bool) -> dict:
    """WHICH control direction is unobservable, in terms this seam can read.

    The predicted null direction is the constant (net-flux) mode, because
    incompressibility inside ``Lambda_i`` constrains the trace to zero net flux.
    Mode 0 of `fourier_basis` IS that mode, so the overlap of the smallest right
    singular vector with it is the check -- and reporting the A/B split alongside
    separates 'incompressibility' from 'the two sides cancel each other', which
    is a different null direction with a different meaning.
    """
    if np.all(T == 0.0):
        return {"note": "T is identically zero; its right singular vectors "
                        "are arbitrary and nothing here would mean anything"}
    _, sv, Vt = np.linalg.svd(T, full_matrices=False)
    w = Vt[-1]
    per = 2 * m if both else m
    const_idx = [0, per] if not both else [0, per]
    e = np.zeros_like(w)
    for i in const_idx:
        if i < e.size:
            e[i] = 1.0
    e = e / np.linalg.norm(e)
    return {
        "smallest_right_singular_vector": [float(x) for x in w],
        "overlap_with_constant_modes": float(abs(w @ e)),
        "share_side_a": float(np.linalg.norm(w[:per])),
        "share_side_b": float(np.linalg.norm(w[per:])),
    }


# ---------------------------------------------------------------------------
# building the two sides, per case
# ---------------------------------------------------------------------------


def mono_reference(u, v, fx, dt, n=W.MONO_N, nu=W.NU):
    ref = W.load_reference()
    mono = ref.WindowNS(nu=nu, length=n * W.H, n=n, cfl=0.4,
                        transmission="dirichlet")
    u1, v1 = mono.step_batch(u[None], v[None], dt, bc0=None, bc1=None,
                             force=(fx[None], np.zeros_like(fx)[None]))
    return u1[0], v1[0], mono


def window_sides(u, v, fx, dt, expose: bool, tiling=None, nu=W.NU,
                 with_force=True):
    tiling = tiling or W.DEFAULT_TILING
    ex = W.make_experts(u, v, tiling, dt=dt, nu=nu, expose_elliptic=expose)
    fxs = tiling.cut(fx)
    off = dict(zip(tiling.names, tiling.offsets))
    A = WindowSide(ex["W00"], "xhi", *off["W00"], tiling.n,
                   force=fxs[0] if with_force else None)
    B = WindowSide(ex["W10"], "xlo", *off["W10"], tiling.n,
                   force=fxs[1] if with_force else None)
    return A, B, tiling


def spectral_sides(u, v, fx, dt, tiling=None, nu=W.NU):
    """The same two windows under the periodic solver.  Everything else identical."""
    tiling = tiling or W.DEFAULT_TILING
    ref = W.load_reference()
    us, vs = tiling.cut(u), tiling.cut(v)
    fxs = tiling.cut(fx)
    off = dict(zip(tiling.names, tiling.offsets))
    sides = []
    for k, (name, face) in enumerate((("W00", "xhi"), ("W10", "xlo"))):
        agent = W.WindowAgent(agent_id=name, u0=us[k], v0=vs[k],
                              shared_faces=tiling.artificial_faces(*off[name]),
                              dt=dt, nu=nu, expose_elliptic=True)
        # swap the solver for the periodic one; the ring bookkeeping, the base
        # state, the macro-step, the flux and the prolongation all stay put, so
        # the single difference between this column and column 2 is the boundary
        # channel -- which is what makes it a control.
        agent._solver = ref.SpectralNS(nu=nu, length=tiling.n * W.H,
                                       n=tiling.n, cfl=0.4)
        sides.append(SpectralSide(agent, face, *off[name], tiling.n, force=fxs[k]))
    return sides[0], sides[1], tiling


def poseidon_sides(u, v, dt, threads: int = 1):
    """Poseidon-T on the two windows of its own seam.

    One thread, measured: on tensors this size eight are about 2x slower here,
    and a probe that spends its time in thread hand-off is measuring the runtime
    rather than the operator.
    """
    import torch

    torch.set_num_threads(threads)
    from atlas.cases import poseidon as PZ
    tiling = PZ.DEFAULT_TILING
    sub = tiling.mono_n
    u, v = np.asarray(u)[:sub, :sub], np.asarray(v)[:sub, :sub]
    us, vs = tiling.cut(u), tiling.cut(v)
    ex = PZ.load_expert()
    off = dict(zip(tiling.names, tiling.offsets))
    sides = []
    for k, (name, face) in enumerate((("P00", "xhi"), ("P10", "xlo"))):
        agent = PZ.PoseidonAgent(agent_id=name, u0=us[k], v0=vs[k],
                                 shared_faces=tiling.artificial_faces(*off[name]),
                                 dt=dt, expert=ex)
        sides.append(PoseidonSide(agent, face, *off[name], tiling.n))
    return sides[0], sides[1], tiling


# ---------------------------------------------------------------------------
# GATE 0
# ---------------------------------------------------------------------------


def gate0_row(u, v, fx, dt, expose: bool, scope: str, s_grid=(0.0, 0.25, 0.5,
                                                              0.75, 1.0)):
    """J along the line from the lagged trace to the monolith's own trace.

    ``scope='seam'`` moves only the sx0 faces; ``scope='all'`` moves every
    artificial face of both windows.  The second is the statement the OBJECTIVE
    makes -- 'the true boundary data makes the agents agree' -- and the first is
    the statement the seam CONTROL can make, which is narrower and is what gate 1
    is about.  Reporting one without the other would be reporting half a check.
    """
    u_ref, v_ref, _mono = mono_reference(u, v, fx, dt)   # noqa: F841
    A, B, tiling = window_sides(u, v, fx, dt, expose)
    off = dict(zip(tiling.names, tiling.offsets))

    def faces(side, name):
        art = tiling.artificial_faces(*off[name])
        return art if scope == "all" else (side.seam_face,)

    def ring_delta(side, name, face):
        """(2, n) increment taking this ring from u^n to the monolith's own."""
        ox, oy, n = side.ox, side.oy, side.n
        du = (u_ref - u)[oy:oy + n, ox:ox + n]
        dv = (v_ref - v)[oy:oy + n, ox:ox + n]
        ring, _ = side._ring(face)
        d = np.stack([du[ring], dv[ring]])
        if scope == "seam_normal":
            # the DECLARED MECH port carries the normal component alone, so
            # this row is what a control living in the declared port space
            # can actually express of the monolith's own datum
            keep = 0 if FACE_COMPONENT[face] == "u" else 1
            z = np.zeros_like(d)
            z[keep] = d[keep]
            d = z
        return d

    deltas = {(side, f): ring_delta(side, nm, f)
              for side, nm in ((A, "W00"), (B, "W10"))
              for f in faces(side, nm)}

    def J(s):
        pa = {f: s * d for (sd, f), d in deltas.items() if sd is A}
        pb = {f: s * d for (sd, f), d in deltas.items() if sd is B}
        return objective(A, B, pa, pb)

    Js = {float(s): J(float(s)) for s in s_grid}
    chk = reference_trace_check(
        lambda s: J(float(s)), 0.0, 0.5, None,
        note=f"J at eta=0, dt={dt}, elliptic={'exposed' if expose else 'embedded'}, "
             f"scope={scope}; the reference trace is the monolith's own ring, in "
             f"the constant-over-the-interval representation the scheme can hold")
    best = min(Js.items(), key=lambda kv: kv[1])
    return {
        "dt": dt, "elliptic": "exposed" if expose else "embedded", "scope": scope,
        "J": Js,
        "J_lagged": Js[0.0], "J_reference": Js[0.5], "J_endpoint": Js[1.0],
        "ratio_midpoint": chk.ratio, "ratio_endpoint": Js[1.0] / Js[0.0]
        if Js[0.0] else float("inf"),
        "passes": bool(chk.passes and chk.ratio < 1.0),
        "argmin_s": best[0], "J_min": best[1],
        "reduction": 1.0 - best[1] / Js[0.0] if Js[0.0] else float("nan"),
        "check": chk.as_dict(),
    }


def gate0(u, v, fx, dts=(0.05, 0.01, 0.002), verbose=True):
    rows = []
    for expose in (True, False):
        for scope in ("all", "seam", "seam_normal"):
            for d in dts:
                t0 = time.perf_counter()
                r = gate0_row(u, v, fx, d, expose, scope)
                r["seconds"] = time.perf_counter() - t0
                rows.append(r)
                if verbose:
                    print(f"  {r['elliptic']:<8s} scope={scope:<11s} dt={d:<6g} "
                          f"J_lag={r['J_lagged']:.6e}  J_ref={r['J_reference']:.6e}  "
                          f"ratio={r['ratio_midpoint']:.4f}  "
                          f"{'PASS' if r['passes'] else 'FAIL'}  "
                          f"(min at s={r['argmin_s']}, "
                          f"reduction {100 * r['reduction']:.1f}%)")
    by_col = {}
    for col in ("exposed", "embedded"):
        sel = [r for r in rows if r["elliptic"] == col and r["scope"] == "all"]
        by_col[col] = {"passes": all(r["passes"] for r in sel),
                       "ratios": [r["ratio_midpoint"] for r in sel],
                       "ratios_endpoint": [r["ratio_endpoint"] for r in sel]}
    return {
        "rows": rows,
        "by_column": by_col,
        "passes": all(c["passes"] for c in by_col.values()),
        "passes_admissible": by_col["exposed"]["passes"],
        "primary_scope": "all",
        "note": "the gate is taken on scope='all' -- the objective's own "
                "statement is about the boundary data, and scope='seam' is "
                "reported beside it because it is what one seam's control can "
                "actually reach. The verdict is recorded PER COLUMN: 'passes' "
                "is the literal form of the gate (every row) and "
                "'passes_admissible' is the exposed column alone, which is "
                "the configuration L2/R10 admits.",
    }


# ---------------------------------------------------------------------------
# GATE 1
# ---------------------------------------------------------------------------


def gate1_case(make_sides, label: str, states, eps: float, m: int = W.M_EFF,
               both_components: bool = False, eps_control: float | None = None,
               has_monolith: bool = True, dt: float = W.MACRO_DT,
               expose: bool | None = None, verbose=True):
    """T's spectrum at three probe states along a marched trajectory.

    Three, not one.  `Xi` moves 55.9% with probe state and ``beta`` moved 22,500x
    between two blocks of one graph, so a ``beta_ctrl`` quoted at one state would
    be the same mistake with a new name.
    """
    out = []
    u0_ref = np.asarray(states[0][1])
    v0_ref = np.asarray(states[0][2])
    denom = float(np.sqrt(np.sum(u0_ref ** 2) + np.sum(v0_ref ** 2)))
    for si, (tag, u, v, fx) in enumerate(states):
        A, B, tiling = make_sides(u, v, fx)
        hold = _lagged_hold(A, B, tiling)
        t0 = time.perf_counter()
        T, j0 = assemble_T(A, B, eps, m, both_components, hold=hold)
        noise = None
        if eps_control is not None:
            T2, _ = assemble_T(A, B, eps_control, m, both_components, hold=hold)
            noise = float(np.linalg.norm(T - T2, 2))
        sp = spectrum(T, noise)
        sp.update({
            "probe_state": tag,
            # so that a flat beta_ctrl is read against a trajectory that
            # went somewhere rather than one that did not
            "state_moved_rel": float(np.sqrt(np.sum((u - u0_ref) ** 2)
                                            + np.sum((v - v0_ref) ** 2))
                                    / denom),
            "eps": eps,
            "eps_control": eps_control,
            "seconds": time.perf_counter() - t0,
            "n_marches": A.n_calls + B.n_calls,
            "jump_norm_lagged": float(np.linalg.norm(j0)),
            "controllable_fraction": controllable_fraction(
                T, j0, noise if noise is not None else sp["tol_machine"]),
            "null": null_direction_report(T, m, both_components),
        })
        # the DtN block on the same marches, for the scale beta_ctrl is read
        # against -- one instrument, one state, one forcing, so the ratio is a
        # ratio rather than two numbers from two runs
        sp["dtn"] = _dtn_from_same_calls(A, B, eps, m, hold)
        tol = noise if noise is not None else sp["tol_machine"]
        g_ref = (reference_control(A, B, u, v, fx, dt, m, both_components)
                 if has_monolith else None)
        asm = None
        if has_monolith and expose is not None:
            ev = composed_evaluator(u, v, fx, dt, expose, tiling)
            asm = (ev, A.agent.agent_id, B.agent.agent_id)
        sp["solve"] = solve_and_check(A, B, T, j0, tol, m, g_ref, hold=hold,
                                      assembly=asm, both=both_components)
        out.append(sp)
        if verbose:
            print(f"  {label:<28s} [{tag}]  beta_ctrl={sp['beta_ctrl']:.6e}  "
                  f"sigma_max={sp['sigma_max']:.6e}  kappa={sp['kappa']:.4g}  "
                  f"rank={sp['rank_machine']}/{sp['shape'][1]}  "
                  f"ctrl_frac={sp['controllable_fraction']:.4f}  "
                  f"({sp['seconds']:.1f} s, {sp['n_marches']} marches)")
    betas = [r["beta_ctrl"] for r in out]
    lo, hi = min(betas), max(betas)
    return {
        "label": label, "states": out,
        "beta_ctrl_min": lo, "beta_ctrl_max": hi,
        "beta_ctrl_spread": (hi / lo) if lo > 0 else float("inf"),
        "beta_ctrl_median": float(np.median(betas)),
    }


def _lagged_hold(A, B, tiling):
    """Every artificial face NOT on this seam, held at its lagged value.

    Held at ZERO increment, which is the lagged datum: the ring already carries
    ``u^n`` and the halo scheme's staleness is that it is not updated, not that
    it is wrong at ``t^n``.
    """
    off = dict(zip(tiling.names, tiling.offsets))
    hold = {}
    for side, name in ((A, tiling.names[0]), (B, tiling.names[1])):
        art = tiling.artificial_faces(*off[name])
        hold[side] = {f: np.zeros((2, side.n)) for f in art if f != side.seam_face}
    return hold


def _dtn_from_same_calls(A: Side, B: Side, eps: float, m: int, hold):
    """S_i = P^* Lambda_i P on each side, from marches identical to T's.

    Quoting the vault's published ``beta = 0.349`` as the scale would be quoting
    a number measured in a different configuration on a different day.  This is
    the same instrument at the same state under the same forcing, which is the
    only way ``beta_ctrl / beta_dtn`` is a ratio.
    """
    P = W.fourier_basis(A.n, m)
    R = W.H * P.T
    out = {}
    for side in (A, B):
        pert = dict(hold.get(side, {}))
        base = side.flux(*side.march(pert))
        cols = []
        for k in range(m):
            p = dict(pert)
            p[side.seam_face] = eps * P[:, k]
            cols.append((side.flux(*side.march(p)) - base) / eps)
        S = R @ np.column_stack(cols)
        sv = np.linalg.svd(S, compute_uv=False)
        tol = max(S.shape) * np.finfo(float).eps * max(float(sv[0]), 1e-300)
        nz = sv[sv > tol]
        out[side.agent.agent_id] = {
            "norm": float(sv[0]),
            "beta": float(nz[-1]) if nz.size else 0.0,
            "kappa": float(nz[0] / nz[-1]) if nz.size else float("inf"),
            "null_dim": int(np.sum(sv <= tol)),
        }
    return out


def eps_sweep(make_sides, u, v, fx, m: int = W.M_EFF,
              steps=(1e-1, 1e-2, 1e-3, 1e-4, 1e-5)) -> dict:
    """beta_ctrl across four decades of the finite-difference step.

    The vault's standard for a probed operator, and the reason it is the
    standard: a derivative that moves with the step is not a derivative.  On
    a deterministic solver the probe floor is machine epsilon rather than a
    checkpoint's 1e-6, so what this sweep can see at the small end is
    cancellation and at the large end is the map's own curvature.
    """
    A, B, tiling = make_sides(u, v, fx)
    hold = _lagged_hold(A, B, tiling)
    rows = []
    for e in steps:
        T, _ = assemble_T(A, B, e, m, hold=hold)
        sv = np.linalg.svd(T, compute_uv=False)
        rows.append({"eps": e, "beta_ctrl": float(sv[-1]),
                     "sigma_max": float(sv[0]),
                     "kappa": float(sv[0] / sv[-1]) if sv[-1] > 0
                     else float("inf")})
    b = [r["beta_ctrl"] for r in rows]
    # The smallest step is where cancellation against the expert's own
    # reproducibility floor shows up first, so the spread WITHOUT it says
    # whether the derivative had converged before the floor was reached.  Both
    # are reported; neither is the headline on its own.
    coarse = b[:-1] if len(b) > 2 else b
    return {"rows": rows,
            "beta_ctrl_spread": (max(b) / min(b)) if min(b) > 0 else float("inf"),
            "beta_ctrl_spread_dropping_smallest_step":
                (max(coarse) / min(coarse)) if min(coarse) > 0 else float("inf"),
            "smallest_step": float(min(r["eps"] for r in rows))}


def check_flux_agrees(u, v, fx, dt=W.MACRO_DT, expose=True) -> dict:
    """Does the flux read off `Side.march` equal `WindowAgent.respond` BITWISE?

    The instrument's own control.  `march` reimplements `respond`'s ring
    construction so it can return a field, and the one thing that would silently
    invalidate every number in gate 1 is the two drifting apart.  Run with the
    forcing OFF, because `respond` passes none.
    """
    A, _B, tiling = window_sides(u, v, fx, dt, expose, with_force=False)
    tr = 1e-3 * np.sin(np.linspace(0, 7.0, A.n))
    mine = A.flux(*A.march({A.seam_face: tr}))
    theirs = A.agent.respond(f"{A.seam_face}:MECH", tr)
    return {"bitwise_equal": bool(np.array_equal(mine, theirs)),
            "max_abs_difference": float(np.abs(mine - theirs).max()),
            "flux_scale": float(np.abs(theirs).max())}


# ---------------------------------------------------------------------------
# the trajectory the three probe states come from
# ---------------------------------------------------------------------------


def probe_states(u, v, fx, n_states=3, stride=10, dt=W.MACRO_DT,
                 n=W.MONO_N, trajectory="spinup",
                 marks=(2, 20, 100)):
    """Three probe states that are genuinely three states.

    ``trajectory="settled"`` marches the supplied state, which is what this
    started as and is kept for continuity -- but that state is a FIXED
    POINT: 200 macro-steps move it 6.03e-4 relative and it saturates by step
    40.  A spread quoted across those three is a spread across one state
    with three labels, which is the mistake `positive-controls-need-a-
    horizon` is about, in its purest form -- the horizon was not short, it
    was absent.

    ``trajectory="spinup"`` (the default) marches from uniform inflow under
    the same forcing and takes ``marks`` along the way.  Measured, those are
    10.5%, 7.9% and 0% from the settled state -- two orders more motion --
    at cell Reynolds numbers 4.24, 4.74 and 4.83 against
    `WindowAgent.validity`'s declared 8, so all three are INSIDE the
    envelope and the spread is not being bought with an out-of-envelope
    state.  The last mark reproduces the supplied artifact to 0.0e+00, which
    says that artifact was made by this march and is a free check that the
    trajectory is the right one -- reported as ``reproduces_artifact``.
    """
    ref = W.load_reference()
    mono = ref.WindowNS(nu=W.NU, length=n * W.H, n=n, cfl=0.4,
                        transmission="dirichlet")
    force = (fx[None], np.zeros_like(fx)[None])

    if trajectory == "spinup":
        U = np.full((n, n), W.U_INF)
        V = np.zeros((n, n))
        want = list(marks)[:n_states]
        label = lambda k: f"spin-up t={k * dt:g}"          # noqa: E731
    else:
        U, V = u.copy(), v.copy()
        want = [s * stride for s in range(n_states)]
        label = lambda k: f"settled t+{k * dt:g}"          # noqa: E731

    ring = (U.copy(), V.copy())
    out = []
    if 0 in want:
        out.append((label(0), U.copy(), V.copy(), fx))
    for k in range(1, max(want) + 1):
        U, V = mono.step_batch(U[None], V[None], dt, bc0=None, bc1=None,
                               force=force)
        U, V = U[0], V[0]
        for Arr, R in ((U, ring[0]), (V, ring[1])):
            Arr[0, :] = R[0, :]
            Arr[-1, :] = R[-1, :]
            Arr[:, 0] = R[:, 0]
            Arr[:, -1] = R[:, -1]
        if k in want:
            out.append((label(k), U.copy(), V.copy(), fx))
    return out


def trajectory_report(states, u, v) -> dict:
    """How far apart the probe states are, and whether the march is the right one.

    Printed and stored so that ``beta_ctrl_spread = 1.00x`` is read against a
    trajectory that went somewhere.
    """
    den = float(np.sqrt(np.sum(u ** 2) + np.sum(v ** 2)))
    rows = []
    for tag, U, V, _fx in states:
        cre = W.H * float(np.max(np.hypot(U, V))) / W.NU
        rows.append({
            "probe_state": tag,
            "distance_from_supplied_state": float(
                np.sqrt(np.sum((U - u) ** 2) + np.sum((V - v) ** 2)) / den),
            "cell_reynolds": cre,
            "inside_declared_envelope": bool(cre <= 8.0),
        })
    d0 = np.stack([states[0][1], states[0][2]])
    spread = max(
        float(np.linalg.norm(np.stack([U, V]) - d0) / np.linalg.norm(d0))
        for _t, U, V, _f in states)
    return {"states": rows, "max_separation": spread,
            "reproduces_artifact": rows[-1]["distance_from_supplied_state"],
            "envelope": "WindowAgent.validity: cell Reynolds <= 8"}


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", default=os.path.join(_ROOT, "out", "tier0_verify",
                                                    "s0_state.npz"))
    ap.add_argument("--out", default=os.path.join(_ROOT, "out", "w149"))
    ap.add_argument("--part", default="all", choices=("all", "gate0", "gate1"))
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--eps-control", type=float, default=1e-3)
    ap.add_argument("--no-poseidon", action="store_true")
    ap.add_argument("--states", type=int, default=3)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--trajectory", default="spinup",
                    choices=("spinup", "settled"),
                    help="spinup: three genuinely different states from "
                         "uniform inflow. settled: march the supplied state, "
                         "which is a fixed point and moves 6.03e-4")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    d = np.load(a.state)
    u, v, fx = d["u"], d["v"], d["fx"]
    art = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "state": os.path.relpath(a.state, _ROOT).replace("\\", "/"),
        "config": {"eps": a.eps, "eps_control": a.eps_control,
                   "m": W.M_EFF, "seam": SEAM, "nu": W.NU, "h": W.H,
                   "macro_dt": W.MACRO_DT, "mono_n": W.MONO_N,
                   "tiling": {"n": W.DEFAULT_TILING.n,
                              "halo": W.DEFAULT_TILING.halo,
                              "ramp": W.DEFAULT_TILING.ramp},
                   "cadence": "one exchange per macro-step, all cases",
                   "states": a.states, "stride": a.stride},
    }

    print("instrument control: the field march against WindowAgent.respond")
    art["controls"] = {"flux_bitwise": check_flux_agrees(u, v, fx)}
    c = art["controls"]["flux_bitwise"]
    print(f"  bitwise equal: {c['bitwise_equal']}  "
          f"(max |difference| {c['max_abs_difference']:.3e} on a flux of "
          f"{c['flux_scale']:.3e})")
    if not c["bitwise_equal"]:
        print("  [!] the field march and the published flux response disagree. "
              "Every number below would be about a different operator.")
        return 1

    if a.part in ("all", "gate0"):
        print("\nGATE 0 -- does the monolith's own trace reduce J?")
        t0 = time.perf_counter()
        art["gate0"] = gate0(u, v, fx)
        print(f"  ({time.perf_counter() - t0:.0f} s)  "
              f"VERDICT: {'PASS' if art['gate0']['passes'] else 'FAIL'}")
        _write(a.out, art)
        g0 = art["gate0"]
        if not g0["passes_admissible"]:
            print("\n  Gate 0 failed on the ADMISSIBLE column. The equation "
                  "is wrong, which is worse than the solver being wrong, and "
                  "it ends the tier. Gate 1 is not run.")
            return 0
        if not g0["passes"]:
            print("\n  Gate 0 failed on the EMBEDDED column and passed on "
                  "the exposed one. That configuration is the one L2/R10 "
                  "already refuses, and the failure carries R10's own "
                  "signature -- the ratio GROWS with dt instead of "
                  "shrinking, so it is the per-window elliptic solve and "
                  "not a dt-scaling defect of the objective. Gate 1 runs, "
                  "and case 3 carries this flag.")

    if a.part in ("all", "gate1"):
        print("\nbuilding the probe trajectory")
        t0 = time.perf_counter()
        states = probe_states(u, v, fx, a.states, a.stride,
                              trajectory=a.trajectory)
        tr = trajectory_report(states, u, v)
        art["controls"]["trajectory"] = dict(tr, kind=a.trajectory)
        print(f"  {len(states)} states on the {a.trajectory} trajectory "
              f"({time.perf_counter() - t0:.0f} s)")
        for r in tr["states"]:
            print(f"    {r['probe_state']:<18s} "
                  f"{r['distance_from_supplied_state']:.4e} from the "
                  f"supplied state, cell Re {r['cell_reynolds']:.2f} "
                  f"({'inside' if r['inside_declared_envelope'] else 'OUTSIDE'} "
                  f"the declared envelope)")
        print(f"    separation across the three: "
              f"{tr['max_separation']:.4e}")

        print("\nGATE 1 -- the control-to-jump spectrum")
        cases = {}
        cases["spectral_periodic"] = gate1_case(
            lambda U, V, F: spectral_sides(U, V, F, W.MACRO_DT),
            "1. SpectralNS (periodic)", states, a.eps, eps_control=a.eps_control)
        cases["windowns_exposed"] = gate1_case(
            lambda U, V, F: window_sides(U, V, F, W.MACRO_DT, True),
            "2. WindowNS elliptic exposed", states, a.eps,
            eps_control=a.eps_control, expose=True)
        cases["windowns_embedded"] = gate1_case(
            lambda U, V, F: window_sides(U, V, F, W.MACRO_DT, False),
            "3. WindowNS elliptic embedded", states, a.eps,
            eps_control=a.eps_control, expose=False)
        cases["windowns_exposed_2c"] = gate1_case(
            lambda U, V, F: window_sides(U, V, F, W.MACRO_DT, True),
            "2b. WindowNS exposed, both components", states, a.eps,
            both_components=True, eps_control=a.eps_control, expose=True)
        if not a.no_poseidon:
            try:
                cases["poseidon"] = gate1_case(
                    lambda U, V, F: poseidon_sides(U, V, W.MACRO_DT),
                    "4. Poseidon-T", states, a.eps,
                    eps_control=a.eps_control, has_monolith=False)
            except Exception as exc:                            # pragma: no cover
                print(f"  Poseidon-T unreachable: {type(exc).__name__}: {exc}")
                cases["poseidon"] = {"unreachable": f"{type(exc).__name__}: {exc}"}
        print("\ncontrol: beta_ctrl across four decades of the FD step")
        sweeps = {}
        u0, v0, fx0 = states[0][1], states[0][2], states[0][3]
        grids = {}
        pairs = [("windowns_exposed",
                  lambda U, V, F: window_sides(U, V, F, W.MACRO_DT, True)),
                 ("windowns_embedded",
                  lambda U, V, F: window_sides(U, V, F, W.MACRO_DT, False))]
        if not a.no_poseidon and "unreachable" not in cases.get(
                "poseidon", {"unreachable": 1}):
            # the checkpoint's declared reproducibility floor is 1e-6, six
            # orders above the solvers', so its grid stops where a
            # finite difference would start reading OP-6's batch-position
            # dependence instead of the operator
            pairs.append(("poseidon",
                          lambda U, V, F: poseidon_sides(U, V, W.MACRO_DT)))
            grids["poseidon"] = (1e-1, 1e-2, 1e-3, 1e-4)
        for key, mk in pairs:
            kw = ({"steps": grids[key]} if key in grids else {})
            sweeps[key] = eps_sweep(mk, u0, v0, fx0, **kw)
            r = sweeps[key]
            print(f"  {key:<20s} " + "  ".join(
                f"{x['eps']:g}:{x['beta_ctrl']:.6e}" for x in r["rows"]))
            print(f"  {'':<20s} spread {r['beta_ctrl_spread']:.4f}x")
        art["controls"]["eps_sweep"] = sweeps
        art["gate1"] = {"cases": cases}
        _write(a.out, art)
        _summary(cases)

    _write(a.out, art)
    return 0


def _write(out, art):
    p = os.path.join(out, "w149.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(art, fh, indent=2, default=float)
    print(f"  wrote {os.path.relpath(p, _ROOT)}")


def _summary(cases):
    print("\n" + "=" * 78)
    ref = cases.get("windowns_exposed", {})
    scale = ref.get("beta_ctrl_median")
    ref_norm = (ref.get("states") or [{}])[0].get("sigma_max")
    for c in cases.values():
        if "unreachable" in c:
            continue
        for st in c["states"]:
            # probed-dtn-coupling 4.5's index, asked of the CONTROL map:
            # how much of the reference agent's control authority does this
            # one reproduce?
            st["Xi_control"] = (st["sigma_max"] / ref_norm) if ref_norm else None
    print(f"{'case':<32s} {'beta_ctrl':>12s} {'/exposed':>9s} {'kappa':>9s} "
          f"{'gap':>8s} {'n_gap':>6s} {'ctrl_f':>7s} {'Xi_ctl':>7s} "
          f"{'sigma':>8s} {'spread':>7s}")
    for k, c in cases.items():
        if "unreachable" in c:
            print(f"{k:<32s} {c['unreachable']}")
            continue
        b = c["beta_ctrl_median"]
        st = c["states"][0]
        rel = (b / scale) if scale else float("nan")
        asm = (st.get("solve") or {}).get("assembly") or {}
        ratio = asm.get("best_over_lagged", float("nan"))
        print(f"{c['label']:<32s} {b:12.4e} {rel:9.3f} {st['kappa']:9.4g} "
              f"{st.get('bottom_gap_ratio', float('nan')):8.4f} "
              f"{st.get('null_dim_gap', -1):6d} "
              f"{st['controllable_fraction']:7.4f} "
              f"{st.get('Xi_control') or float('nan'):7.4f} "
              f"{ratio:8.4f} "
              f"{c['beta_ctrl_spread']:6.2f}x")
    print("=" * 78)


if __name__ == "__main__":
    raise SystemExit(main())
