"""The acoustics family, run from a case file: style C with an explicit step.

Sound crossing from one medium into another (showcase case 4): linear
acoustics as a pressure-velocity system,

    rho dv/dt = -grad p,        dp/dt = -K div v,        K = rho c^2,

on a staggered grid -- pressure at cell centres, each velocity component on the
faces normal to it -- marched by leapfrog, the scheme of Yee and of every
finite-difference acoustics code.  A face between two media carries the
arithmetic mean of their densities (the mass of the half-cells either side of
it).  Every edge of the domain is a rigid wall: the normal velocity there is
zero, so the domain is closed and holds its energy.

**Style C, explicit: the pieces meet along a face and trade traces every step,
with nothing to iterate.**  The case cuts the domain into two pieces along a
vertical line; the interface faces are owned by the first piece.  Each step:

1. the second piece sends its pressures next to the interface (a Dirichlet
   datum for the first);
2. the first piece updates its face velocities, the interface's included;
3. it sends the interface velocities (a Neumann datum -- the flux -- for the
   second), and the second updates its own faces;
4. both update their pressures.

That is the Dirichlet-Neumann exchange of style C with the iteration gone: an
explicit step needs only the neighbour's OLD values, so one exchange closes it.
Every face and cell is updated from the same numbers by the same arithmetic as
on the undivided domain, so the two pieces ARE the full domain, bit for bit
(the check), and the lesson is the contrast with the iterated style C of
conduction.

**What is measured against the textbook.**  A plane pulse runs along x in the
first medium, hits the interface, and splits.  At normal incidence the
reflected pressure is ``R = (Z2 - Z1) / (Z2 + Z1)`` of the incident, with
``Z = rho c``.  The discrete interface reflects each wavenumber with
``(Z2 cos(k2 dx/2) - Z1 cos(k1 dx/2)) / (Z2 cos(k2 dx/2) + Z1 cos(k1 dx/2))``
(derived from the semi-discrete scheme; the leapfrog's discrete impedance is
``rho c`` at every wavenumber), which is ``R`` exactly at zero wavenumber.  So
the measure is the pulse's integral: once the reflected pulse has left the
interface and before anything comes back to it, ``sum of p over the first
medium / sum of p of the incident`` is the zero-wavenumber component and must be
``R``, whatever the dispersion does to the pulse's shape.  The window in which
that holds is computed from the case's geometry (the pulse's centre and width,
the two media's lengths and speeds), with seven widths of margin.

**The energy.**  Leapfrog conserves ``sum rho_f v_{n+1/2}^2 / 2 + sum p_n
p_{n+1} / (2 K)`` exactly in a closed domain, so its drift is round-off (the
second check).  The share of it in the second medium is what the page plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import geometry as geo
from ..checks import CheckSpec, exact, judge

FAMILY = "acoustics-2d"
STYLE = "C"
ARMS = ("serial", "full")
FIELD_LABEL = "pressure (Pa)"
LENGTH_UNIT = "m"
SERIES = {"energy_second": "Share of the energy in the second medium | fraction"}
#: the separation margin, in pulse widths (a Gaussian's tail there is e^-24.5)
MARGIN = 7.0

REGISTERED = "2026-09-29, before the acoustics family's first run"

CHECKS: tuple[CheckSpec, ...] = (
    CheckSpec(
        key="energy", title="Energy: the closed domain keeps it", kind="balance",
        tolerance=1e-10, registered=REGISTERED,
        measure=("max over steps of |E_n - E_0| / E_0, E the leapfrog's own energy "
                 "sum rho_f v^2/2 + sum p_n p_(n+1) / (2 K), worst arm"),
        why=("leapfrog conserves that quantity exactly on a closed domain, so what is "
             "left is the round-off of the updates and the sums, about 1e-16 per step "
             "over a few thousand steps")),
    CheckSpec(
        key="closed_form", title="Reflection = (Z2 - Z1) / (Z2 + Z1)", kind="reference",
        tolerance=1e-6, registered=REGISTERED,
        measure=("max over the steps when the reflected pulse is alone in the first "
                 "medium of |sum of p there / sum of p of the incident pulse - R|, "
                 "R the textbook normal-incidence reflection with Z = rho c"),
        why=("the discrete interface reflects the zero-wavenumber component with R "
             "exactly (the module's derivation), and the pulse's integral is that "
             "component; what is left is the Gaussian's tail beyond the seven-width "
             "margin (about 2e-11) and round-off. 1e-6 leaves room for what the "
             "derivation did not see")),
    CheckSpec(
        key="bitwise", title="The two pieces equal the full domain, bit for bit",
        kind="control", tolerance=None, registered=REGISTERED,
        measure="np.array_equal on the pressure and both velocities after every step",
        why=("every face and cell is updated from the same numbers by the same "
             "arithmetic in either arm; the exchange only moves the numbers")),
)


# ---------------------------------------------------------------------------
# the case as a staggered grid
# ---------------------------------------------------------------------------


def media(spec) -> tuple[np.ndarray, np.ndarray]:
    """(density ``rho``, sound speed ``c``) per cell, ``(ny, nx)``."""
    d = spec.domain
    owner = geo.region_owner(spec.regions, d.nx, d.ny)
    rho = np.zeros((d.ny, d.nx))
    c = np.zeros((d.ny, d.nx))
    for i, r in enumerate(spec.regions):
        m = spec.materials[r.material]
        rho[owner == i] = float(m["rho"])
        c[owner == i] = float(m["c"])
    return rho, c


def stable_dt(spec) -> float:
    """The largest leapfrog step this grid is stable at, ``2 / sqrt(lambda)``, with
    ``lambda`` bounded by Gershgorin on the symmetrized operator: per cell,
    ``sum over its interior faces of (K_i + sqrt(K_i K_j)) / (rho_f dx^2)``.

    On one medium that is ``dx / (c sqrt 2)`` exactly; at an interface the
    face's mean density can make it slightly tighter, and the bound says so.  On a
    drawn domain only the open faces -- both cells in the domain -- count.
    """
    d = spec.domain
    rho, c = media(spec)
    if d.outline is not None or d.holes:
        rho, c, K, open_x, open_y = masked_media(spec)
    else:
        K = rho * c * c
        open_x = open_y = 1.0
    rows = np.zeros((d.ny, d.nx))
    rf = 0.5 * (rho[:, :-1] + rho[:, 1:])
    t_l = (K[:, :-1] + np.sqrt(K[:, :-1] * K[:, 1:])) / rf * open_x
    t_r = (K[:, 1:] + np.sqrt(K[:, :-1] * K[:, 1:])) / rf * open_x
    rows[:, :-1] += t_l
    rows[:, 1:] += t_r
    rf = 0.5 * (rho[:-1, :] + rho[1:, :])
    t_b = (K[:-1, :] + np.sqrt(K[:-1, :] * K[1:, :])) / rf * open_y
    t_t = (K[1:, :] + np.sqrt(K[:-1, :] * K[1:, :])) / rf * open_y
    rows[:-1, :] += t_b
    rows[1:, :] += t_t
    return float(2.0 * d.dx / np.sqrt(np.max(rows)))


def masked_media(spec):
    """A drawn domain's media (case file 0.4): ``(rho, c, K, open_x, open_y)`` with
    the void outside the domain given ``K = 0`` (its pressure never moves) and
    ``rho = 1`` (never read), and the faces open where both cells are in the
    domain.  A face to the void is a rigid wall: its velocity stays zero."""
    act = geo.domain_mask(spec.domain)
    rho, c = media(spec)
    rho = np.where(act, rho, 1.0)
    c = np.where(act, c, 0.0)
    return (rho, c, rho * c * c, (act[:, :-1] & act[:, 1:]).astype(float),
            (act[:-1, :] & act[1:, :]).astype(float))


def cut_column(spec) -> int | None:
    """The column where the second piece starts, for two windows side by side
    over the full height; None for any other layout."""
    d = spec.domain
    if len(spec.windows) != 2:
        return None
    a, b = sorted(spec.windows, key=lambda w: w.x0)
    if (a.x0, a.y0, b.y0) != (0, 0, 0) or a.ny != d.ny or b.ny != d.ny:
        return None
    if a.x0 + a.nx != b.x0 or b.x0 + b.nx != d.nx:
        return None
    return int(b.x0)


@dataclass
class Pulse:
    amplitude: float
    x0: float
    width: float


def pulse(spec) -> Pulse:
    return Pulse(float(spec.physics.get("amplitude")), float(spec.physics.get("pulse_x")),
                 float(spec.physics.get("pulse_width")))


def closed_form_reflection(spec) -> tuple[float | None, str]:
    """``(Z2 - Z1) / (Z2 + Z1)`` and why, when the case is two media meeting at the
    cut with the pulse in the first; None and the reason otherwise."""
    m = cut_column(spec)
    if m is None:
        return None, "the pieces are not two windows side by side"
    d = spec.domain
    rho, c = media(spec)
    left, right = (rho[:, :m], c[:, :m]), (rho[:, m:], c[:, m:])
    for r_, c_ in (left, right):
        if np.ptp(r_) != 0.0 or np.ptp(c_) != 0.0:
            return None, "a piece holds more than one medium"
    p = pulse(spec)
    #: a relative slack of 1e-9: 7 x 0.1 m is one unit in the last place over 0.7 m,
    #: and a pulse placed exactly seven widths clear was refused for it (seen)
    eps = 1e-9 * m * d.dx
    if not (MARGIN * p.width - eps <= p.x0 <= m * d.dx - MARGIN * p.width + eps):
        return None, (f"the pulse is not {MARGIN:g} widths clear of the wall and the "
                      f"interface in the first medium")
    z1 = float(left[0][0, 0] * left[1][0, 0])
    z2 = float(right[0][0, 0] * right[1][0, 0])
    return (z2 - z1) / (z2 + z1), "two media meeting at the cut"


def plateau(spec, dt: float) -> tuple[int, int] | None:
    """The macro-steps (first, last) when the reflected pulse is alone in the first
    medium: after the incident's trailing edge has reached the interface, and
    before either the reflected pulse's leading edge (back off the left wall) or
    the transmitted one's (back off the right wall) returns to it."""
    m = cut_column(spec)
    if m is None:
        return None
    d = spec.domain
    _rho, c = media(spec)
    c1, c2 = float(c[0, 0]), float(c[0, -1])
    p = pulse(spec)
    x_i, l2 = m * d.dx, (d.nx - m) * d.dx
    lead = (x_i - p.x0 - MARGIN * p.width) / c1           # leading edge reaches it
    t_sep = (x_i - p.x0 + MARGIN * p.width) / c1          # trailing edge has
    t_end = min(lead + 2.0 * x_i / c1, lead + 2.0 * l2 / c2)
    first, last = int(np.ceil(t_sep / dt)), int(np.floor(t_end / dt))
    return (first, last) if last > first else None


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


@dataclass
class Wave:
    """Pressure at ``t_n`` and velocities at ``t_(n-1/2)``; for the decomposed arm
    each array is the pieces' own, joined only for reading."""

    p: np.ndarray                 # (ny, nx)
    vx: np.ndarray                # (ny, nx + 1)
    vy: np.ndarray                # (ny + 1, nx)
    pieces: tuple | None = None   # the decomposed arm's (left, right) arrays
    energy: float | None = None


class AcousticsRun:
    family = FAMILY

    def __init__(self, spec, arms=ARMS, threads: int = 1):
        self.spec = spec
        self.arms_asked = tuple(arms)
        d = spec.domain
        self.nx, self.ny, self.dx = d.nx, d.ny, float(d.dx)
        self.dt = float(spec.run.macro_dt)
        self.limit = stable_dt(spec)
        if self.dt > self.limit:                          # the case check refuses it first
            raise ValueError(f"the macro-step {self.dt:.4g} s is over the leapfrog's "
                             f"stability limit {self.limit:.4g} s")
        #: a drawn domain or drawn pieces (case file 0.4): the same leapfrog on the
        #: whole grid with the void's faces shut, pieces as masks (`_init_masked`)
        self.plain = geo.is_plain(spec)
        if not self.plain:
            self._init_masked(spec)
            return
        self.m = cut_column(spec)
        if self.m is None:
            raise ValueError("the acoustics family's pieces are two windows side by side, "
                             "each the full height")
        self.rho, self.c = media(spec)
        self.K = self.rho * self.c * self.c
        rho_fx = 0.5 * (self.rho[:, :-1] + self.rho[:, 1:])   # interior x-faces
        rho_fy = 0.5 * (self.rho[:-1, :] + self.rho[1:, :])   # interior y-faces
        self.ax = self.dt / (rho_fx * self.dx)
        self.ay = self.dt / (rho_fy * self.dx)
        self.bk = self.dt * self.K / self.dx
        self.rho_fx, self.rho_fy = rho_fx, rho_fy
        self.arms = tuple(a for a in ARMS if a in arms)
        self.R, self.R_why = closed_form_reflection(spec)
        self.window = plateau(spec, self.dt)
        p = pulse(spec)
        self.pulse = p
        x = (np.arange(self.nx) + 0.5) * self.dx
        xf = np.arange(self.nx + 1) * self.dx
        c1, rho1 = float(self.c[0, 0]), float(self.rho[0, 0])

        def f(s):
            return p.amplitude * np.exp(-0.5 * ((s - p.x0) / p.width) ** 2)
        self.p0 = np.tile(f(x), (self.ny, 1))
        vx0 = np.tile(f(xf + 0.5 * c1 * self.dt) / (rho1 * c1), (self.ny, 1))
        vx0[:, 0] = vx0[:, -1] = 0.0                      # rigid walls
        self.vx0 = vx0
        self.incident = float(np.sum(self.p0[:, :self.m]))

    def _init_masked(self, spec) -> None:
        """Any domain, any two pieces.  The void's pressure never moves (``K = 0``)
        and its faces never open (their update is multiplied by zero), so a drawn
        edge is a rigid wall.  The pieces are the two windows' cells: the first
        owns its own faces and the faces it shares with the second, the second its
        own; each holds whole-grid arrays and touches only what it owns, updated
        by the same expressions as the full domain -- so they are it, to the bit."""
        d = spec.domain
        act = geo.domain_mask(d)
        self.act = act
        self.m = None
        rho, c, K, open_x, open_y = masked_media(spec)
        self.rho, self.c, self.K = rho, c, K
        self.K_safe = np.where(act, K, 1.0)             # for the energy's p^2 / K
        rho_fx = 0.5 * (rho[:, :-1] + rho[:, 1:])
        rho_fy = 0.5 * (rho[:-1, :] + rho[1:, :])
        self.ax = self.dt / (rho_fx * self.dx) * open_x
        self.ay = self.dt / (rho_fy * self.dx) * open_y
        self.bk = self.dt * K / self.dx
        self.rho_fx, self.rho_fy = rho_fx, rho_fy
        self.arms = tuple(a for a in ARMS if a in self.arms_asked)
        self.R, self.R_why = None, ("a drawn domain or pieces: the textbook reflection needs "
                                    "two uniform media meeting at a straight cut")
        self.window = None
        masks = [m for _n, m in geo.window_masks(spec)]
        if len(masks) != 2:
            raise ValueError("style C couples exactly two pieces")
        mA, mB = masks
        self.mA, self.mB = mA, mB
        ox, oy = open_x > 0, open_y > 0
        ownA_x = ox & (mA[:, :-1] | mA[:, 1:])
        ownA_y = oy & (mA[:-1, :] | mA[1:, :])
        self.axA, self.ayA = self.ax * ownA_x, self.ay * ownA_y
        self.axB = self.ax * (ox & mB[:, :-1] & mB[:, 1:])
        self.ayB = self.ay * (oy & mB[:-1, :] & mB[1:, :])
        self.iface_x = ox & ((mA[:, :-1] & mB[:, 1:]) | (mB[:, :-1] & mA[:, 1:]))
        self.iface_y = oy & ((mA[:-1, :] & mB[1:, :]) | (mB[:-1, :] & mA[1:, :]))
        self.ownA_x, self.ownA_y = ownA_x, ownA_y
        self.bkA, self.bkB = self.bk * mA, self.bk * mB
        p = pulse(spec)
        self.pulse = p
        # the first medium: the piece's commonest (rho, c)
        pairs, counts = np.unique(np.column_stack([rho[mA], c[mA]]), axis=0,
                                  return_counts=True)
        rho1, c1 = (float(v) for v in pairs[int(np.argmax(counts))])
        x = (np.arange(self.nx) + 0.5) * self.dx
        xf = np.arange(self.nx + 1) * self.dx

        def f(s):
            return p.amplitude * np.exp(-0.5 * ((s - p.x0) / p.width) ** 2)
        self.p0 = np.tile(f(x), (self.ny, 1)) * act
        vx0 = np.tile(f(xf + 0.5 * c1 * self.dt) / (rho1 * c1), (self.ny, 1))
        vx0[:, 1:-1] *= open_x
        vx0[:, 0] = vx0[:, -1] = 0.0                      # rigid walls
        self.vx0 = vx0
        self.incident = float(np.sum(self.p0[mA]))
        self.cells = int(act.sum())

    # -- the arms -------------------------------------------------------------

    def initial(self, arm: str) -> Wave:
        p, vx, vy = self.p0.copy(), self.vx0.copy(), np.zeros((self.ny + 1, self.nx))
        if arm == "full":
            return Wave(p, vx, vy)
        if not self.plain:
            a = (p.copy(), vx.copy(), vy.copy())
            b = (p.copy(), vx.copy(), vy.copy())
            return Wave(p, vx, vy, pieces=(a, b))
        m = self.m
        left = (p[:, :m].copy(), vx[:, :m + 1].copy(), vy[:, :m].copy())
        right = (p[:, m:].copy(), vx[:, m:].copy(), vy[:, m:].copy())
        return Wave(p, vx, vy, pieces=(left, right))

    def _full(self, s: Wave) -> Wave:
        p, vx, vy = s.p.copy(), s.vx.copy(), s.vy.copy()
        vx[:, 1:-1] -= self.ax * (p[:, 1:] - p[:, :-1])
        vy[1:-1, :] -= self.ay * (p[1:, :] - p[:-1, :])
        p -= self.bk * ((vx[:, 1:] - vx[:, :-1]) + (vy[1:, :] - vy[:-1, :]))
        return Wave(p, vx, vy)

    def _pieces(self, s: Wave) -> Wave:
        m = self.m
        (pl, vxl, vyl), (pr, vxr, vyr) = s.pieces
        pl, vxl, vyl = pl.copy(), vxl.copy(), vyl.copy()
        pr, vxr, vyr = pr.copy(), vxr.copy(), vyr.copy()
        # 1. the second piece's pressures beside the interface, to the first
        trace_p = pr[:, 0].copy()
        # 2. the first piece's faces, the interface's included
        vxl[:, 1:m] -= self.ax[:, :m - 1] * (pl[:, 1:] - pl[:, :-1])
        vxl[:, m] -= self.ax[:, m - 1] * (trace_p - pl[:, m - 1])
        vyl[1:-1, :] -= self.ay[:, :m] * (pl[1:, :] - pl[:-1, :])
        # 3. the interface velocities, to the second piece; its own faces
        vxr[:, 0] = vxl[:, m]
        vxr[:, 1:-1] -= self.ax[:, m:] * (pr[:, 1:] - pr[:, :-1])
        vyr[1:-1, :] -= self.ay[:, m:] * (pr[1:, :] - pr[:-1, :])
        # 4. the pressures
        pl -= self.bk[:, :m] * ((vxl[:, 1:] - vxl[:, :-1]) + (vyl[1:, :] - vyl[:-1, :]))
        pr -= self.bk[:, m:] * ((vxr[:, 1:] - vxr[:, :-1]) + (vyr[1:, :] - vyr[:-1, :]))
        p = np.concatenate([pl, pr], axis=1)
        vx = np.concatenate([vxl, vxr[:, 1:]], axis=1)
        vy = np.concatenate([vyl, vyr], axis=1)
        return Wave(p, vx, vy, pieces=((pl, vxl, vyl), (pr, vxr, vyr)))

    def _pieces_masked(self, s: Wave) -> Wave:
        """The two pieces of any shape (_init_masked): the same four moves."""
        (pA, vxA, vyA), (pB, vxB, vyB) = s.pieces
        pA, vxA, vyA = pA.copy(), vxA.copy(), vyA.copy()
        pB, vxB, vyB = pB.copy(), vxB.copy(), vyB.copy()
        # 1. the second piece's pressures, to the first (it reads those beside the
        #    interface; its own faces never reach further into the second piece)
        seen = np.where(self.mB, pB, pA)
        # 2. the first piece's faces, the interface's included
        vxA[:, 1:-1] -= self.axA * (seen[:, 1:] - seen[:, :-1])
        vyA[1:-1, :] -= self.ayA * (seen[1:, :] - seen[:-1, :])
        # 3. the interface velocities, to the second piece; its own faces
        vxB[:, 1:-1] = np.where(self.iface_x, vxA[:, 1:-1], vxB[:, 1:-1])
        vyB[1:-1, :] = np.where(self.iface_y, vyA[1:-1, :], vyB[1:-1, :])
        vxB[:, 1:-1] -= self.axB * (pB[:, 1:] - pB[:, :-1])
        vyB[1:-1, :] -= self.ayB * (pB[1:, :] - pB[:-1, :])
        # 4. the pressures
        pA -= self.bkA * ((vxA[:, 1:] - vxA[:, :-1]) + (vyA[1:, :] - vyA[:-1, :]))
        pB -= self.bkB * ((vxB[:, 1:] - vxB[:, :-1]) + (vyB[1:, :] - vyB[:-1, :]))
        p = np.where(self.mA, pA, pB)
        vx, vy = vxB.copy(), vyB.copy()
        vx[:, 1:-1] = np.where(self.ownA_x, vxA[:, 1:-1], vxB[:, 1:-1])
        vy[1:-1, :] = np.where(self.ownA_y, vyA[1:-1, :], vyB[1:-1, :])
        return Wave(p, vx, vy, pieces=((pA, vxA, vyA), (pB, vxB, vyB)))

    def step(self, arm: str, s: Wave) -> Wave:
        if arm == "full":
            return self._full(s)
        return self._pieces(s) if self.plain else self._pieces_masked(s)

    # -- instruments ---------------------------------------------------------

    def _energies(self, p_old: np.ndarray, s: Wave) -> tuple[float, float]:
        """(the leapfrog's conserved energy at t_(n+1/2), the part of it in the
        second medium), J per metre of depth."""
        area = self.dx * self.dx
        kin_x = 0.5 * self.rho_fx * s.vx[:, 1:-1] ** 2
        kin_y = 0.5 * self.rho_fy * s.vy[1:-1, :] ** 2
        if not self.plain:
            pot = 0.5 * p_old * s.p / self.K_safe      # the void holds no pressure
            total = (float(np.sum(kin_x)) + float(np.sum(kin_y)) + float(np.sum(pot))) * area
            second = (float(np.sum(kin_x[self.axB > 0])) + float(np.sum(kin_y[self.ayB > 0]))
                      + float(np.sum(pot[self.mB]))) * area
            return total, second
        pot = 0.5 * p_old * s.p / self.K
        m = self.m
        total = (float(np.sum(kin_x)) + float(np.sum(kin_y)) + float(np.sum(pot))) * area
        second = (float(np.sum(kin_x[:, m:])) + float(np.sum(kin_y[:, m:]))
                  + float(np.sum(pot[:, m:]))) * area
        return total, second

    def observe(self, arm: str, s: Wave, prev: Wave | None = None) -> dict:
        p_old = prev.p if prev is not None else self.p0
        total, second = self._energies(p_old, s)
        first = (float(np.sum(s.p[:, :self.m])) if self.plain
                 else float(np.sum(s.p[self.mA])))
        return {"energy": total, "energy_second": second / total if total else 0.0,
                "reflected": first / self.incident}

    def bitwise_equal(self, a: Wave, b: Wave) -> bool:
        return bool(np.array_equal(a.p, b.p) and np.array_equal(a.vx, b.vx)
                    and np.array_equal(a.vy, b.vy))

    def field(self, s: Wave) -> np.ndarray:
        if self.plain or self.act.all():
            return s.p
        return np.where(self.act, s.p, np.nan)          # the void: not drawn

    def close(self) -> None:
        pass

    # -- the end of a run ---------------------------------------------------

    def compare(self, states, history, bitwise):
        metrics: dict[str, Any] = {}
        checks = []
        drift = {}
        for arm, rows in history.items():
            if not rows:
                continue
            e0 = rows[0]["energy"]
            drift[arm] = max(abs(r["energy"] - e0) for r in rows) / e0
            metrics[arm] = {"energy_drift": drift[arm],
                            "energy_second": rows[-1]["energy_second"],
                            "reflected_last": rows[-1]["reflected"]}
        checks.append(judge(CHECKS[0], max(drift.values()) if drift else None,
                            "; ".join(f"{a}: {v:.3g}" for a, v in drift.items())))
        rows = history.get("serial") or history.get("full") or []
        if self.R is None:
            checks.append(judge(CHECKS[1], None, f"no closed form: {self.R_why}"))
        elif self.window is None:
            checks.append(judge(CHECKS[1], None, "the domain is too short for the reflected "
                                                 "pulse to be alone in the first medium"))
        else:
            first, last = self.window
            vals = [r["reflected"] for k, r in enumerate(rows, start=1) if first <= k <= last]
            if not vals:
                checks.append(judge(CHECKS[1], None,
                                    f"the run ended at step {len(rows)}, before the "
                                    f"reflected pulse was alone in the first medium "
                                    f"(steps {first} to {last})"))
            else:
                err = max(abs(v - self.R) for v in vals)
                for a in metrics:
                    arm_rows = history[a]
                    arm_vals = [r["reflected"] for k, r in enumerate(arm_rows, start=1)
                                if first <= k <= last]
                    metrics[a]["reflection"] = float(np.mean(arm_vals))
                    metrics[a]["reflection_vs_textbook"] = float(np.mean(arm_vals)) - self.R
                checks.append(judge(CHECKS[1], err,
                                    f"R measured {float(np.mean(vals)):.12f} (mean over steps "
                                    f"{first}-{min(last, len(rows))}; largest departure "
                                    f"{err:.2g}), textbook {self.R:.12f}", textbook=self.R))
        if "serial" in states and "full" in states and history.get("serial"):
            ok = bitwise.get("first_difference") is None and self.bitwise_equal(
                states["serial"], states["full"])
            checks.append(exact(CHECKS[2], ok, f"equal after all {len(history['serial'])} "
                                               f"steps" if ok else "differs"))
        else:
            checks.append(exact(CHECKS[2], None, "needs the pieces and the full domain"))
        return metrics, checks

    def notes(self, done: int) -> list[str]:
        rho, c = self.rho, self.c
        m = self.m
        out = [f"Leapfrog step {self.dt:.4g} s against the stability limit {self.limit:.4g} "
               f"s (Gershgorin on the grid's own operator). The pieces trade the pressures "
               f"beside the interface and its velocities once a step; nothing iterates."]
        if not self.plain:
            out.append("A drawn domain or pieces: every face to the void is a rigid wall "
                       "(its velocity stays zero), so the domain is closed and keeps its "
                       "energy. The reflection is not read: the textbook reflection needs "
                       "two uniform media meeting at a straight cut.")
            return out
        z1, z2 = float(rho[0, 0] * c[0, 0]), float(rho[0, -1] * c[0, -1])
        if self.R is not None:
            t = 2.0 * z2 / (z1 + z2)
            out.append(f"Impedances Z = rho c: {z1:.4g} and {z2:.4g} Pa s/m. Textbook "
                       f"reflection R = {self.R:.6f}, transmission T = 1 + R = {t:.6f} (the "
                       f"transmitted pressure is {t:.3g} times the incident), energy "
                       f"transmitted 1 - R^2 = {1.0 - self.R ** 2:.4g} at zero wavenumber. "
                       f"The energy share the page plots is carried by the pulse's whole "
                       f"spectrum, whose discrete reflection departs from R at order "
                       f"(k dx)^2, so it is close to that and not equal; the reflection "
                       f"check reads the zero-wavenumber part, where the scheme is exact.")
        if self.window is not None:
            first, last = self.window
            out.append(f"The reflected pulse is alone in the first medium from step {first} "
                       f"to step {last}; the reflection is read there.")
        return out

    def describe(self) -> dict[str, Any]:
        return {"style": "C", "explicit": True,
                "cells": self.nx * self.ny if self.plain else self.cells,
                "cut_column": self.m, "dt_s": self.dt, "stable_dt_s": self.limit,
                "textbook_R": self.R, "plateau_steps": list(self.window) if self.window
                else None, "pulse": vars(self.pulse)}


def available_arms(spec) -> tuple[tuple[str, ...], dict[str, str]]:
    return ARMS, {"parallel": "the second piece needs the interface velocities the first "
                              "computes in the same step"}


def build(spec, arms=ARMS, threads: int = 1) -> AcousticsRun:
    return AcousticsRun(spec, arms=arms, threads=threads)


def case_graph(spec):
    """The case for the compiler (`compile.py`): the two pieces as two agents meeting
    at the interface's faces, one MECH seam.

    A piece's step is explicit -- no solve, a one-cell stencil, one leapfrog step
    per exchange -- so it declares no elliptic solve.  Its response to the
    pressure on the far side of each interface face (the effort) is the normal
    velocity into it there after one velocity update from rest (the flow), the
    face's own density: the half of the explicit exchange each piece does."""
    from atlas.capability import (BCChannel, ClaimType, Direction, EllipticSubsolve,
                                  ExpertCapabilities, MotionClass, TimeDiscretization,
                                  port_decl)
    from atlas.graph import Agent, CaseGraph, Connection, Decomposition
    from atlas.ports import PortType, ResponseHalf

    from ..compile import face_prolongation, modes_for
    d = spec.domain
    dt = float(spec.run.macro_dt)
    amp = float(spec.physics.get("amplitude"))
    if geo.is_plain(spec):
        m = cut_column(spec)
        if m is None:
            from ..compile import CompileRefused
            raise CompileRefused("the acoustics family's pieces are two windows side by "
                                 "side")
        rho, c = media(spec)
        rho_f = 0.5 * (rho[:, m - 1] + rho[:, m])              # the interface faces' density
        left, right = sorted(spec.windows, key=lambda w: w.x0)
        z1 = float(rho[0, 0] * c[0, 0])
    else:
        rho_f, z1 = _interface_faces(spec)
        left, right = spec.windows
    n_faces = int(rho_f.size)
    scales = {"stress": amp, "velocity": amp / z1, "power_area": amp * amp / z1}
    port = "interface:MECH"

    def piece(name, sign):
        def respond(_port, trace):
            # from rest, the inside pressure is zero: the velocity into the piece is
            # dt / (rho_f dx) times the pressure pushing in from the far side
            return dt / (rho_f * d.dx) * np.asarray(trace, dtype=float).ravel()
        return ExpertCapabilities(
            expert_id=name, ports=[port_decl(
                name=port, port_type=PortType.MECH,
                geometry=f"the {n_faces} interface faces, from the "
                         f"{'first' if sign > 0 else 'second'} piece's side",
                direction=Direction.BIDIRECTIONAL, nondim=dict(scales),
                effective_resolution=modes_for(n_faces), motion_class=MotionClass.STATIC,
                response_half=ResponseHalf.FLOW,
                prolongation=face_prolongation(name, port, n_faces, d.dx),
                note="the normal velocity into the piece after one velocity update")],
            bc_channel=BCChannel.DIRICHLET, bc_time_varying=True,
            elliptic_subsolve=EllipticSubsolve.NONE,
            time_discretization=TimeDiscretization.EXPLICIT, stencil_radius=1,
            substeps_per_macro_step=1, dt_native=dt,
            validity=lambda state=None, cond=None: dt <= stable_dt(spec),
            governing_family="linear-acoustics-2d",
            lambda_ref="the same leapfrog on the whole domain, the full-domain arm",
            claim_types=frozenset({ClaimType.TRAJECTORY}),
            weight_hash="workbench/acoustics-leapfrog", boundary_response=respond,
            probe_base=lambda _p: np.zeros(n_faces),
            reproducibility_floor=float(np.finfo(float).eps), deterministic=True,
            note="atlas/workbench/families/acoustics.py: staggered-grid leapfrog")
    return CaseGraph(
        name=f"workbench-{spec.name}",
        agents=[Agent(left.id, piece(left.id, 1), domain=f"piece {left.id}"),
                Agent(right.id, piece(right.id, -1), domain=f"piece {right.id}")],
        connections=[Connection(
            seam_id="interface", a=(left.id, port), b=(right.id, port),
            port_type=PortType.MECH, derive_space=True, geometrically_coincident=True,
            expected_null_dim=0, cut_axis=Decomposition.NON_OVERLAPPING,
            note="the explicit Dirichlet-Neumann exchange: pressures one way, the "
                 "interface velocities the other")],
        decomposition=Decomposition.NON_OVERLAPPING, cross_points=(), macro_dt=dt,
        note=f"the workbench case {spec.name!r}: style C, explicit")


def _interface_faces(spec) -> tuple[np.ndarray, float]:
    """Two pieces of any shape (case file 0.4): the density of each face between
    them, in order along the interface (`compile.curve_order`), and the first
    piece's impedance ``rho c`` (its commonest medium)."""
    from ..compile import CutFaces, curve_order
    d = spec.domain
    rho, c, _K, ox, oy = masked_media(spec)
    (_na, mA), (_nb, mB) = geo.window_masks(spec)
    nx = d.nx
    rows_in, outs, dens = [], [], []
    for axis in (1, 0):
        if axis == 1:
            open_ = ox > 0
            a, b = mA[:, :-1], mB[:, 1:]
            a2, b2 = mB[:, :-1], mA[:, 1:]
            jj, ii = np.nonzero(open_ & ((a & b) | (a2 & b2)))
            left_in_a = mA[jj, ii]
            cin = np.where(left_in_a, jj * nx + ii, jj * nx + ii + 1)
            cout = np.where(left_in_a, jj * nx + ii + 1, jj * nx + ii)
            rf = 0.5 * (rho[jj, ii] + rho[jj, ii + 1])
        else:
            open_ = oy > 0
            a, b = mA[:-1, :], mB[1:, :]
            a2, b2 = mB[:-1, :], mA[1:, :]
            jj, ii = np.nonzero(open_ & ((a & b) | (a2 & b2)))
            low_in_a = mA[jj, ii]
            cin = np.where(low_in_a, jj * nx + ii, (jj + 1) * nx + ii)
            cout = np.where(low_in_a, (jj + 1) * nx + ii, jj * nx + ii)
            rf = 0.5 * (rho[jj, ii] + rho[jj + 1, ii])
        rows_in.append(cin)
        outs.append(cout)
        dens.append(rf)
    cin, cout, rf = np.concatenate(rows_in), np.concatenate(outs), np.concatenate(dens)
    idx_a = np.flatnonzero(mA.ravel())
    pos = np.full(mA.size, -1, dtype=np.int64)
    pos[idx_a] = np.arange(idx_a.size)
    order = curve_order(CutFaces(idx_a, pos[cin], cout), np.arange(cin.size), nx)
    pairs, counts = np.unique(np.column_stack([rho[mA], c[mA]]), axis=0, return_counts=True)
    rho1, c1 = (float(v) for v in pairs[int(np.argmax(counts))])
    return rf[order], rho1 * c1


__all__ = ["FAMILY", "STYLE", "ARMS", "CHECKS", "AcousticsRun", "Wave", "build", "media",
           "masked_media", "stable_dt", "cut_column", "closed_form_reflection", "plateau",
           "available_arms"]
