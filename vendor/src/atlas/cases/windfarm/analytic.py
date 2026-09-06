"""Fields with known answers.

Phase W4 exists because of one instruction in `impl-wind-farm-guide` section 5:
build every measurement *before* the coupled system can produce numbers nobody
can check. A metric that cannot recover a known answer cannot grade a model, so
each metric in `metrics.py` is validated here against a field whose parameters
were put in by hand.

Four families, in increasing order of how much they are entitled to claim:

`UniformFlow`
    An exact steady solution of the incompressible equations at any viscosity.
    Every residual, every symmetry measure and the global power residual must
    vanish on it to floating point. This is the null test, and it is the one
    that catches sign errors.

`JensenWake`
    The 1983 Park model: a top-hat deficit inside a linearly expanding wake.
    Not a solution of anything -- it is an engineering correlation, and it is
    used here only as the *corridor* gate W8 measures against, and as a field
    whose two parameters the wake diagnostics must recover exactly.

`BastankhahWake`
    The Gaussian self-similar far-wake model, derived from mass and momentum
    conservation applied to an assumed Gaussian profile. Better physics than
    Jensen in the far field and the second half of gate W8's corridor.

`ForcedUniformFlow`
    Uniform flow plus the actuator disk's own body force. Not a solution -- a
    uniform field under a body force would accelerate -- but the *instantaneous*
    energy bookkeeping is exact and analytic, which is what gate W4's third item
    tests: the extracted power is what closes the budget.

All fields are in case-study units: `D = U_inf = rho = 1`, `x` streamwise, `y`
lateral, turbine 1 at `x = 0`.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field

import numpy as np

from .disk import c_t as _c_t

U_INF = 1.0


# --------------------------------------------------------------------------
# the field protocol
# --------------------------------------------------------------------------


class Field:
    """A steady analytic field on the domain.

    Subclasses supply `velocity`; `pressure` defaults to zero, which is correct
    for every field here that has no pressure model -- and is *declared* rather
    than assumed, so a metric that needs pressure cannot silently read a
    fabricated one."""

    has_pressure: bool = False

    def velocity(self, x, y):
        raise NotImplementedError

    def pressure(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        return np.zeros_like(x)

    def body_force(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        return np.zeros_like(x), np.zeros_like(x)

    # -- convenience --------------------------------------------------------

    def on_grid(self, x_c: np.ndarray, y_c: np.ndarray):
        """(u, v) on the tensor grid `x_c` x `y_c`, indexed [y, x] to match
        `adapters` -- the convention W0/E2 measured rather than assumed."""
        X, Y = np.meshgrid(np.asarray(x_c, float), np.asarray(y_c, float), indexing="xy")
        return self.velocity(X, Y)


@dataclass(frozen=True)
class UniformFlow(Field):
    u_inf: float = U_INF

    def velocity(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        return np.full(np.shape(x), float(self.u_inf)), np.zeros(np.shape(x))


# --------------------------------------------------------------------------
# engineering wake models -- corridors, not predictions
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class JensenWake(Field):
    """Jensen / Park top-hat wake.

    Half-width grows linearly, `r_w(x) = r_0 + k x` with `r_0 = D/2 = 0.5`, and
    the deficit follows from momentum conservation applied to a top hat:

        u(x)/U_inf = 1 - 2a (r_0 / r_w)^2 = 1 - 2a / (1 + 2 k x)^2.

    `k = 0.075` is the standard onshore value and the one spec section 2.4 uses
    to justify the wake corridor's width. The profile is *discontinuous* at the
    wake edge, which is physically wrong and is the reason gate W8 is a corridor
    rather than a fit."""

    a: float = 1.0 / 3.0
    k: float = 0.075
    x_turbine: float = 0.0
    u_inf: float = U_INF
    r0: float = 0.5

    def half_width(self, x):
        """`r_w(x)`, clipped upstream of the rotor where the model says nothing."""
        s = np.maximum(np.asarray(x, dtype=np.float64) - self.x_turbine, 0.0)
        return self.r0 + self.k * s

    def centreline(self, x):
        s = np.asarray(x, dtype=np.float64) - self.x_turbine
        rw = self.half_width(x)
        u = self.u_inf * (1.0 - 2.0 * self.a * (self.r0 / rw) ** 2)
        return np.where(s >= 0.0, u, self.u_inf)

    def velocity(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        inside = (x >= self.x_turbine) & (np.abs(y) <= self.half_width(x))
        u = np.where(inside, self.centreline(x), self.u_inf)
        return np.broadcast_to(u, np.broadcast(x, y).shape).astype(np.float64), \
            np.zeros(np.broadcast(x, y).shape)


@dataclass(frozen=True)
class BastankhahWake(Field):
    """Bastankhah & Porte-Agel (2014) Gaussian far wake.

    Self-similar Gaussian deficit whose amplitude follows from mass and momentum
    conservation rather than being fitted:

        sigma(x)/D = k* x/D + eps,    eps = 0.2 sqrt(beta),
        beta = (1 + sqrt(1-C_T)) / (2 sqrt(1-C_T)),
        dU/U_inf = (1 - sqrt(1 - C_T / (8 (sigma/D)^2))) exp(-y^2 / (2 sigma^2)).

    The square root goes imaginary very near the rotor, where the model does not
    apply; `valid_from` is where it becomes real and the deficit is clipped
    upstream of it rather than being quietly made up."""

    c_t: float = 8.0 / 9.0                  # C_T at a = 1/3
    k_star: float = 0.04
    x_turbine: float = 0.0
    u_inf: float = U_INF

    @property
    def beta(self) -> float:
        r = np.sqrt(1.0 - self.c_t)
        return float((1.0 + r) / (2.0 * r))

    @property
    def eps(self) -> float:
        return float(0.2 * np.sqrt(self.beta))

    def sigma(self, x):
        s = np.maximum(np.asarray(x, dtype=np.float64) - self.x_turbine, 0.0)
        return self.k_star * s + self.eps

    @property
    def valid_from(self) -> float:
        """Smallest `x - x_turbine` at which the radicand is non-negative."""
        s_min = np.sqrt(self.c_t / 8.0)
        return float(max((s_min - self.eps) / self.k_star, 0.0))

    def amplitude(self, x):
        sig = self.sigma(x)
        rad = 1.0 - self.c_t / (8.0 * sig ** 2)
        return np.where(rad >= 0.0, 1.0 - np.sqrt(np.clip(rad, 0.0, None)), np.nan)

    def centreline(self, x):
        s = np.asarray(x, dtype=np.float64) - self.x_turbine
        return np.where(s >= self.valid_from,
                        self.u_inf * (1.0 - self.amplitude(x)), self.u_inf)

    def velocity(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        s = x - self.x_turbine
        sig = self.sigma(x)
        amp = np.where(s >= self.valid_from, self.amplitude(x), 0.0)
        amp = np.nan_to_num(amp, nan=0.0)
        u = self.u_inf * (1.0 - amp * np.exp(-y ** 2 / (2.0 * sig ** 2)))
        shape = np.broadcast(x, y).shape
        return np.broadcast_to(u, shape).astype(np.float64), np.zeros(shape)


@dataclass(frozen=True)
class SuperposedWakes(Field):
    """Two or more wakes combined.

    `mode='linear'` is Jensen's own sum of deficits; `mode='rss'` is the
    root-sum-square rule of Katic. Both are *conventions*, not derivations --
    superposition of wakes has no theoretical justification -- so the choice is
    named at the call site instead of being buried."""

    wakes: tuple = ()
    u_inf: float = U_INF
    mode: str = "linear"

    def velocity(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        shape = np.broadcast(x, y).shape
        deficits = []
        for w in self.wakes:
            uw, _ = w.velocity(x, y)
            deficits.append(1.0 - np.broadcast_to(uw, shape) / self.u_inf)
        if not deficits:
            return np.full(shape, self.u_inf), np.zeros(shape)
        d = np.stack(deficits, axis=0)
        total = d.sum(axis=0) if self.mode == "linear" else np.sqrt((d ** 2).sum(axis=0))
        return self.u_inf * (1.0 - total), np.zeros(shape)


# --------------------------------------------------------------------------
# the forced field that closes the energy budget analytically
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ForcedUniformFlow(Field):
    """Uniform flow carrying the actuator disk's body force.

    Deliberately *not* a solution: a uniform field under a body force would
    accelerate. What is exact is the instantaneous bookkeeping, and that is all
    gate W4's third item needs. With `u = (U_d, 0)` constant and the disk force
    integrating to `-T` over the strip,

        W = integral of u . f dV = -T U_d = -P,

    so the extracted power `P_ext = -W` is exactly `T U_d`, while a uniform field
    contributes zero to the energy flux and zero to the dissipation. The global
    power residual therefore has to come out at exactly `P` with the `P_ext`
    term omitted, and exactly zero with it included -- which is the sharpest
    available statement that `P_ext` is what closes the budget rather than a
    fudge that happens to be small."""

    u_disk: float = 2.0 / 3.0
    thrust: float = 4.0 / 9.0               # T = 0.5 C_T' <U_d>^2 at a = 1/3
    x_face: float = 0.0
    thickness: float = 0.1
    halfspan: float = 0.5

    def velocity(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        shape = np.broadcast(x, y).shape
        return np.full(shape, float(self.u_disk)), np.zeros(shape)

    def in_strip(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return ((x >= self.x_face) & (x <= self.x_face + self.thickness)
                & (np.abs(y) <= self.halfspan))

    def body_force(self, x, y):
        """`f = -T / (A Delta_d)` inside the strip, zero outside; `A = 2*halfspan`."""
        shape = np.broadcast(np.asarray(x), np.asarray(y)).shape
        area = 2.0 * self.halfspan * self.thickness
        fx = np.where(self.in_strip(x, y), -self.thrust / area, 0.0)
        return np.broadcast_to(fx, shape).astype(np.float64), np.zeros(shape)

    @property
    def extracted_power(self) -> float:
        return float(self.thrust * self.u_disk)


# --------------------------------------------------------------------------
# corridors
# --------------------------------------------------------------------------


def jensen_corridor(x, a: float = 1.0 / 3.0, k_lo: float = 0.04, k_hi: float = 0.10):
    """Centreline velocity between two Jensen expansion rates.

    Gate W8 asks whether the measured recovery lies *inside* a corridor, not
    whether it matches a curve. `k` is the only free parameter of the model and
    the literature spread for onshore terrain is roughly 0.04-0.10, so the
    corridor is that spread rather than an error bar on anything."""
    lo = JensenWake(a=a, k=k_hi).centreline(x)       # faster spread -> faster recovery
    hi = JensenWake(a=a, k=k_lo).centreline(x)
    return np.minimum(lo, hi), np.maximum(lo, hi)


def bpa_corridor(x, c_t: float = 8.0 / 9.0, k_lo: float = 0.025, k_hi: float = 0.06):
    """Centreline velocity between two Bastankhah expansion rates.

    **NaN upstream of where every member of the corridor is valid.** The model's
    amplitude carries a square root that goes imaginary near the rotor, and
    `valid_from` moves with `k*` -- so close in, the slowest-spreading member is
    still clipped to the free stream while the fastest already has a deficit,
    and the "corridor" inverts. It is a far-wake model; saying so with a NaN is
    better than quietly returning bounds that a correct wake can fall outside."""
    x = np.asarray(x, dtype=np.float64)
    wl = BastankhahWake(c_t=c_t, k_star=k_hi)
    wh = BastankhahWake(c_t=c_t, k_star=k_lo)
    lo = wl.centreline(x)
    hi = wh.centreline(x)
    valid = (x - wl.x_turbine) >= max(wl.valid_from, wh.valid_from)
    return (np.where(valid, np.minimum(lo, hi), np.nan),
            np.where(valid, np.maximum(lo, hi), np.nan))


def c_t_from_induction(a: float) -> float:
    return float(_c_t(a))


# --------------------------------------------------------------------------
# upstream induction -- how much a rotor slows the flow ahead of itself
# --------------------------------------------------------------------------
#
# W5's coupled rollout measured a 17.3% centreline deficit one diameter
# *upstream* of turbine 1 and called it excessive. Whether it is excessive
# depends on which theory it is compared against, and the two below differ by
# a factor of nearly three at that station. Both are exact potential-flow
# results, so neither is an opinion.


def induction_2d_strip(x, a: float = 1.0 / 3.0, diameter: float = 1.0) -> np.ndarray:
    """Axial induction on the centreline of a **2-D** actuator strip.

    The 2-D wake is bounded by two semi-infinite vortex sheets trailing from
    the strip's tips at `y = +-D/2`, equal and opposite. A sheet element
    `gamma d(xi)` at `(xi, h)` induces `u = gamma h / (2 pi r^2) d(xi)` at
    `(x, 0)`, and the mirror sheet contributes the same sign, so

        u_ind(x) = (gamma / pi) [ pi/2 + arctan(x / h) ],   h = D/2.

    This is `gamma/2` at the strip and `gamma` far downstream, which is the
    actuator-disk half-rule and fixes `gamma = -2 a U`. Hence

        1 - u/U = (2a / pi) [ pi/2 + arctan(x / h) ].

    At `a = 1/3`: **9.84% one diameter upstream, 3.51% at three.**

    Unbounded. Channel confinement raises it, and the wind farm's rotor blocks
    12.5% of an 8 D channel, which is not negligible."""
    x = np.asarray(x, dtype=np.float64)
    h = diameter / 2.0
    return (2.0 * a / np.pi) * (np.pi / 2.0 + np.arctan(x / h))


def induction_3d_disk(x, a: float = 1.0 / 3.0, diameter: float = 1.0) -> np.ndarray:
    """Axial induction on the axis of a **3-D** actuator disk.

    The classical vortex-cylinder result,

        1 - u/U = a [ 1 + (x/R) / sqrt(1 + (x/R)^2) ],   R = D/2,

    equal to `a` at the disk and `2a` far downstream. At `a = 1/3`: **3.52%
    one diameter upstream, 0.90% at three.**

    Kept beside the 2-D law because grading a 2-D case study against this one
    is how a correct number gets called a bug. A trailing vortex *cylinder*
    closes on itself and its influence decays far faster than that of two
    infinite sheets."""
    x = np.asarray(x, dtype=np.float64)
    r = diameter / 2.0
    s = x / r
    return a * (1.0 + s / np.sqrt(1.0 + s * s))


__all__ = [
    "Field", "UniformFlow", "JensenWake", "BastankhahWake", "SuperposedWakes",
    "ForcedUniformFlow", "jensen_corridor", "bpa_corridor", "c_t_from_induction",
    "induction_2d_strip", "induction_3d_disk", "U_INF",
]
