"""Group averaging -- making a frozen expert equivariant it was never trained to be.

**The problem this exists for.** Gate W7 asks the wind-farm system for a mirror
residual below 1e-6 under symmetric inflow. Measured 2026-08-23, one call of the
frozen checkpoint on a *uniform inflow* -- the most trivial mirror-symmetric
field there is -- returns

    max|u - M u| = 2.42e-2,

which is 2.4e4 times the gate, with no graph, no ports, no agents and no tiling
anywhere in the loop. It is deterministic (two calls on identical input agree to
exactly 0), it accumulates (3.16e-2 -> 1.02e-1 over eight calls), and its
antisymmetric part is a comb at 32/16/8 cells -- the checkpoint's shifted-window
stride and its harmonics, on an input with no structure at all. The defect is
keyed to the attention grid, not to the flow. Recorded as OP-6.

So no arrangement of agents, tiles, ports, blending or Schwarz iteration can make
W7 pass: equivariance is a property of the operator before it is a property of
the composition. Either the expert has it or the composition layer supplies it.

**What this module does.** Supplies it, by the Reynolds operator over a finite
group G of symmetries the problem declares:

    E~(u) = (1/|G|) sum_{g in G} g^{-1} E(g u)

Nothing is trained and the expert is not touched; the cost is exactly |G| forward
passes where there was one.

**Why this is exactly equivariant, and why G must be a group.** For any h in G,

    E~(h u) = (1/|G|) sum_g g^{-1} E(g h u)

and substituting g' = g h -- which ranges over all of G precisely because G is
closed under composition --

            = (1/|G|) sum_{g'} h g'^{-1} E(g' u)
            = h E~(u).

The closure is not decoration. Average over a *set* that is not a group and the
substitution fails: the result is smoother, it is not equivariant, and the
failure is quiet. `SymmetryGroup` therefore verifies closure numerically at
construction rather than trusting the caller to have declared a group.

**What it costs and what it preserves.** |G| expert calls per step. Averaging is
linear, so every linear property of the expert's output survives it: a family of
divergence-free fields averages to a divergence-free field, so C1 is untouched;
a family of fields with the same mean averages to that mean, so W0's mean-flow
handling is untouched. What it cannot do is repair a *nonlinear* defect -- this
makes the operator equivariant, not correct.

**Scope.** `cases/windfarm` is the first caller, but nothing here knows about
wind farms, actuator disks or Poseidon. The wind farm declares the mirror; a
later case study declaring a rotation or a translation group gets the same
machinery, which is the point of it living in `atlas/invariants/` rather than
next to the adapter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SymmetryOp", "Identity", "MirrorY", "MirrorX",
    "SymmetryGroup", "MIRROR_Y", "MIRROR_X", "SymmetrizedExpert", "symmetrize",
]


# ---------------------------------------------------------------------------
# group elements
# ---------------------------------------------------------------------------


class SymmetryOp:
    """One element of a symmetry group, acting on a 2-D vector field.

    Fields are `[..., ny, nx]` with the case-study convention that the last two
    axes are `[y, x]`; a leading batch axis is optional and is left alone. An op
    acts on the *pair* `(u, v)` because a reflection is not a permutation of
    samples -- it flips the sign of the velocity component normal to the mirror,
    and an implementation that reindexes without re-signing produces a field that
    looks plausible and is not the reflection of anything.

    `frame` is the same transformation applied to a spatially constant vector,
    which is what the Galilean frame velocity and any uniform forcing are. It has
    to travel with the field: mirroring a window while leaving its frame velocity
    pointing the old way is not a symmetry operation, and because the frame is
    subtracted before the forward pass and added back after, the error would be
    invisible in the transformed field and present in the result.
    """

    name: str = "op"

    def field(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def frame(self, fu: float, fv: float) -> tuple[float, float]:
        raise NotImplementedError

    @property
    def inverse(self) -> "SymmetryOp":
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.name}>"


class Identity(SymmetryOp):
    name = "e"

    def field(self, u, v):
        return u, v

    def frame(self, fu, fv):
        return fu, fv

    @property
    def inverse(self):
        return self


class MirrorY(SymmetryOp):
    """Reflection in the horizontal centreline: (x, y) -> (x, -y).

    u(x, y) -> u(x, -y),   v(x, y) -> -v(x, -y)

    On a lattice whose cell centres are symmetric about the axis -- which
    `build_global` and every 128x128 expert window satisfy exactly, verified as
    max|y_c + reverse(y_c)| = 0 -- the reflection *is* the array reversal, with
    no interpolation and therefore no interpolation error. That exactness is why
    the averaged operator is equivariant to machine precision rather than to
    whatever a resampling would leave behind.

    An involution, so it is its own inverse and {e, M_y} is a group of order 2.
    """

    name = "M_y"

    def field(self, u, v):
        return u[..., ::-1, :].copy(), -v[..., ::-1, :].copy()

    def frame(self, fu, fv):
        return fu, -fv

    @property
    def inverse(self):
        return self


class MirrorX(SymmetryOp):
    """Reflection in the vertical centreline: (x, y) -> (-x, y).

    Provided for completeness and for the closure tests -- the wind farm has no
    streamwise symmetry to declare, since inflow and outflow are not alike.
    """

    name = "M_x"

    def field(self, u, v):
        return -u[..., :, ::-1].copy(), v[..., :, ::-1].copy()

    def frame(self, fu, fv):
        return -fu, fv

    @property
    def inverse(self):
        return self


# ---------------------------------------------------------------------------
# the group
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymmetryGroup:
    """A finite group of symmetry operations, closure-checked at construction.

    The check is numerical rather than symbolic: compose every pair on a random
    field and require the result to match some element of the group to machine
    precision. That catches the two mistakes a symbolic check would not -- an op
    whose `field` and `frame` disagree, and an op that reindexes correctly while
    getting a component sign wrong -- because both show up as a composition that
    matches nothing.
    """

    ops: tuple[SymmetryOp, ...]
    name: str = "G"

    def __post_init__(self):
        if not any(isinstance(o, Identity) for o in self.ops):
            raise ValueError(
                f"group {self.name} has no identity element. The Reynolds average "
                "over a set without the identity is not an average of the operator "
                "with anything -- it replaces E by a reflection of E.")
        self.check_closure()

    def __len__(self):
        return len(self.ops)

    def check_closure(self, n: int = 24, seed: int = 0, tol: float = 1e-14) -> None:
        rng = np.random.default_rng(seed)
        u = rng.standard_normal((n, n))
        v = rng.standard_normal((n, n))
        table = [(o.name, o.field(u, v)) for o in self.ops]

        def matches(a, b):
            return (float(np.max(np.abs(a[0] - b[0]))) <= tol
                    and float(np.max(np.abs(a[1] - b[1]))) <= tol)

        for g in self.ops:
            for h in self.ops:
                gh = g.field(*h.field(u, v))
                if not any(matches(gh, entry) for _, entry in table):
                    raise ValueError(
                        f"group {self.name} is not closed: {g.name} o {h.name} is not "
                        "in the group. The Reynolds average is equivariant only over a "
                        "group -- over a mere set the substitution g' = gh does not "
                        "range over the sum, and the result is a smoothing that is not "
                        "equivariant and does not announce it.")
            if not any(g.inverse.name == o.name for o in self.ops):
                raise ValueError(f"group {self.name} lacks the inverse of {g.name}")


#: The wind farm's declared symmetry: the domain, the tiling, the agent boxes and
#: both rotors are mirror-symmetric about y = 0, verified 2026-08-23 as zero
#: mirror-unpaired tiles out of 124 in both the eight-agent and monolithic builds.
MIRROR_Y = SymmetryGroup((Identity(), MirrorY()), name="{e, M_y}")
MIRROR_X = SymmetryGroup((Identity(), MirrorX()), name="{e, M_x}")


# ---------------------------------------------------------------------------
# the averaged operator
# ---------------------------------------------------------------------------


class SymmetrizedExpert:
    """A frozen expert wrapped in the Reynolds average over a declared group.

    Drop-in for `FrozenFluidExpert` / `SolverExpert`: same `step` and `step_many`
    signatures, `|G|` times the forward passes, **nothing trained and the wrapped
    expert not modified**. Attributes the wrapped object has and this one does not
    are forwarded, so `scaling`, `checkpoint` and the rest read through.

    `n_calls` therefore counts the true number of expert forwards, which is |G|
    times what an unwrapped run would report -- the honest number, and the one the
    cost of this construction should be judged on.
    """

    def __init__(self, expert, group: SymmetryGroup = MIRROR_Y):
        self.expert = expert
        self.group = group
        self.last_mean_drift = (0.0, 0.0)

    def __getattr__(self, item):
        # only reached for attributes this object does not define
        return getattr(self.__dict__["expert"], item)

    def __repr__(self):
        return f"SymmetrizedExpert({self.expert!r}, {self.group.name})"

    # -- the average -------------------------------------------------------

    def _average(self, call, u, v, force, frame, bc=None):
        acc_u = np.zeros_like(np.asarray(u, dtype=np.float64))
        acc_v = np.zeros_like(np.asarray(v, dtype=np.float64))
        drift = 0.0
        for g in self.group.ops:
            gu, gv = g.field(u, v)
            gframe = None if frame is None else g.frame(float(frame[0]), float(frame[1]))
            gforce = None if force is None else g.field(
                np.asarray(force[0], dtype=np.float64),
                np.asarray(force[1], dtype=np.float64))
            # A Schwarz transmission ring is a *vector field* on the same window,
            # so it travels with the window exactly as the force does. Mirroring
            # the interior while leaving the ring pointing the old way would hand
            # the operator a boundary condition belonging to the unmirrored
            # problem -- and the average would still come back looking symmetric,
            # which is the failure mode this class exists to prevent rather than
            # to create. Same argument as section 2.2's frame velocity, one
            # argument further out.
            gbc = None if bc is None else g.field(
                np.asarray(bc[0], dtype=np.float64),
                np.asarray(bc[1], dtype=np.float64))
            u1, v1 = call(gu, gv, gforce, gframe, gbc)
            iu, iv = g.inverse.field(np.asarray(u1, dtype=np.float64),
                                     np.asarray(v1, dtype=np.float64))
            acc_u += iu
            acc_v += iv
            d = getattr(self.expert, "last_mean_drift", (0.0, 0.0))
            drift = max(drift, abs(float(d[0])), abs(float(d[1])))
        n = float(len(self.group))
        self.last_mean_drift = (drift, drift)
        return acc_u / n, acc_v / n

    def step_many(self, u, v, dt, project=False, galilean=None, force=None,
                  chunk: int = 32, frame=None, bc=None):
        def call(gu, gv, gforce, gframe, gbc):
            kw = {} if gbc is None else {"bc": gbc}
            return self.expert.step_many(gu, gv, dt, project=project, galilean=galilean,
                                         force=gforce, chunk=chunk, frame=gframe, **kw)
        return self._average(call, u, v, force, frame, bc)

    def step(self, u, v, dt, project=False, galilean=None, force=None, bc=None):
        def call(gu, gv, gforce, gframe, gbc):
            kw = {} if gbc is None else {"bc": tuple(gbc)}
            return self.expert.step(gu, gv, dt, project=project, galilean=galilean,
                                    force=None if gforce is None else tuple(gforce), **kw)
        return self._average(call, u, v, force, None, bc)


def symmetrize(expert, group: SymmetryGroup = MIRROR_Y) -> SymmetrizedExpert:
    """`symmetrize(e)` reads better than the constructor at a call site."""
    return SymmetrizedExpert(expert, group)
