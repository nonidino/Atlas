"""A linear advection-diffusion torus, decomposed into two overlapping strips.

**Not a case study and not a fixture.**  `CASE-STUDY-GUIDE.md` draws that line
between a synthetic ``boundary_response`` (a fixture) and a real solver (a case
study); this is a third thing, and the distinction matters for what it is allowed
to earn.  It is a **model problem for a rule** -- the object `tests/` needs to
test `G5/W16`'s decomposition criterion *directly*, the way
`test_tier9_assembly_condition.py` tests `L6/C1` against the bias-variance
identity rather than against `reference.WindowNS`.

It earns nothing on Tier 0.  It measures no constant of any real expert.  What it
does have that no fixture has and no real case study affords is **an exactly
computable reference and a freely movable cut**: the same physics, the same
agents, the same partition of unity, the same halo, and the only thing that
changes between two runs is *where the domain is cut*.  That is the one
experiment a cut-placement criterion can be falsified by, and the four-window
tiling of `window_ns.py` cannot run it -- its cut is at the centre by
construction and there is nowhere else to put it.

The model
---------

On a periodic unit torus, ``N_x x N_y`` cells,

    du/dt  =  -a . grad u  +  div(nu(x, y) grad u)

centred second order, ``nu`` at faces by arithmetic mean, advanced by Heun.  The
reference ``E`` is that scheme on the whole torus.  ``nu`` varies in **y** so
that moving the cut moves it into stiffer or softer material, and in **x** so the
seam operator is not diagonal in the Fourier basis the interface is declared on
-- a translation-invariant seam makes Fourier the eigenbasis and drives the
off-diagonal mass to zero identically, which would make the criterion under test
vacuous rather than wrong.

The decomposition is two strips in ``y``, each ``N_y/2 + halo`` rows tall, so
each has two artificial boundaries and the overlap is ``halo`` rows in each of
two bands.  ``cut`` rotates the whole decomposition around the torus: the strip
heights, the halo, the ramp, the agent and the number of agents are all held
fixed and **only the cut location moves**.

The scheme mirrors `window_ns.py`'s split-step composition exactly: lagged
Dirichlet data on each strip's own artificial rows, an exchange every sub-step
(R10b), a convex partition of unity that vanishes at each artificial edge
(R11 / L6/C1), and an overlap wider than the agent's domain of dependence over
one exchange interval (the halo rule).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# the physics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Physics:
    """Advection-diffusion on a torus, with a spatially varying diffusivity.

    ``nu_y_amp`` is the load-bearing parameter.  It is what makes two cut
    placements differ in the *operator* rather than only in the solution, which
    is what separates a criterion computed from ``S`` from one computed from the
    scheme's own defect.
    """

    nx: int = 48
    ny: int = 48
    a_x: float = 0.2
    a_y: float = 0.1
    nu0: float = 4.0e-3
    nu_y_amp: float = 0.8      # nu varies across the cut direction: cos(4 pi y)
    nu_x_amp: float = 0.3      # ... and along it, so S is not diagonal
    nu_y_waves: float = 2.0    # two periods in y, so BOTH cuts see the same band

    @property
    def hx(self) -> float:
        return 1.0 / self.nx

    @property
    def hy(self) -> float:
        return 1.0 / self.ny

    def nu_field(self) -> np.ndarray:
        """(ny, nx) cell-centred diffusivity, strictly positive."""
        y = (np.arange(self.ny) + 0.5) * self.hy
        x = (np.arange(self.nx) + 0.5) * self.hx
        fy = 1.0 + self.nu_y_amp * np.cos(2.0 * np.pi * self.nu_y_waves * y)
        fx = 1.0 + self.nu_x_amp * np.cos(2.0 * np.pi * x)
        return self.nu0 * np.outer(fy, fx)

    def stable_substep(self, safety: float = 0.35) -> float:
        nu_max = float(self.nu_field().max())
        d_diff = 1.0 / (2.0 * nu_max * (1.0 / self.hx**2 + 1.0 / self.hy**2))
        d_adv = 1.0 / (abs(self.a_x) / self.hx + abs(self.a_y) / self.hy + 1e-300)
        return safety * min(d_diff, d_adv)


def _rhs(u: np.ndarray, phys: Physics, nu: np.ndarray,
         y_periodic: bool) -> np.ndarray:
    """-a.grad u + div(nu grad u), centred, periodic in x.

    ``y_periodic`` is False on a strip, where rows 0 and -1 are the artificial
    Dirichlet boundary and the operator is not evaluated there.
    """
    hx, hy = phys.hx, phys.hy
    up_x = np.roll(u, -1, axis=1)
    dn_x = np.roll(u, 1, axis=1)
    nu_xp = 0.5 * (nu + np.roll(nu, -1, axis=1))
    nu_xm = 0.5 * (nu + np.roll(nu, 1, axis=1))
    adv = -phys.a_x * (up_x - dn_x) / (2.0 * hx)
    dif = (nu_xp * (up_x - u) - nu_xm * (u - dn_x)) / hx**2

    if y_periodic:
        up_y, dn_y = np.roll(u, -1, axis=0), np.roll(u, 1, axis=0)
        nu_yp = 0.5 * (nu + np.roll(nu, -1, axis=0))
        nu_ym = 0.5 * (nu + np.roll(nu, 1, axis=0))
    else:
        up_y = np.vstack([u[1:], u[-1:]])
        dn_y = np.vstack([u[:1], u[:-1]])
        nu_yp = 0.5 * (nu + np.vstack([nu[1:], nu[-1:]]))
        nu_ym = 0.5 * (nu + np.vstack([nu[:1], nu[:-1]]))
    adv = adv - phys.a_y * (up_y - dn_y) / (2.0 * hy)
    dif = dif + (nu_yp * (up_y - u) - nu_ym * (u - dn_y)) / hy**2
    out = adv + dif
    if not y_periodic:
        out[0] = 0.0
        out[-1] = 0.0
    return out


def heun(u: np.ndarray, dt: float, phys: Physics, nu: np.ndarray,
         y_periodic: bool) -> np.ndarray:
    k1 = _rhs(u, phys, nu, y_periodic)
    k2 = _rhs(u + dt * k1, phys, nu, y_periodic)
    return u + 0.5 * dt * (k1 + k2)


#: Heun evaluates the 3-point stencil twice, so one sub-step moves information
#: two cells.  With an exchange every sub-step (R10b) that is the agent's whole
#: domain of dependence per exchange interval, and it is what the halo must beat.
STENCIL_RADIUS = 2
SUBSTEPS_PER_EXCHANGE = 1


# ---------------------------------------------------------------------------
# the decomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StripDecomposition:
    """Two overlapping strips on the torus, rotated by ``cut``.

    ``cut`` is the only free parameter and it is a pure cut-placement knob: the
    strip height, the overlap, the ramp and the agents are identical for every
    value of it.  Strip 0 owns global rows ``[cut, cut + h)`` mod ny, strip 1
    owns ``[cut + ny/2, cut + ny/2 + h)`` mod ny, with ``h = ny/2 + halo``.
    """

    phys: Physics = field(default_factory=Physics)
    halo: int = 6
    ramp: int = 4
    cut: int = 0

    def __post_init__(self) -> None:
        if self.phys.ny % 2:
            raise ValueError("ny must be even")
        if self.strip_rows > self.phys.ny:
            raise ValueError("strips overlap themselves; reduce halo")

    @property
    def strip_rows(self) -> int:
        return self.phys.ny // 2 + self.halo

    @property
    def n_strips(self) -> int:
        return 2

    def rows(self, i: int) -> np.ndarray:
        """Global y-rows owned by strip ``i``, in the strip's own order."""
        start = self.cut + i * (self.phys.ny // 2)
        return (np.arange(self.strip_rows) + start) % self.phys.ny

    def restrict(self, u: np.ndarray, i: int) -> np.ndarray:
        return u[self.rows(i), :]

    def nu_of(self, i: int) -> np.ndarray:
        return self.phys.nu_field()[self.rows(i), :]

    def weights(self) -> list[np.ndarray]:
        """chi_i on the global grid: squared linear ramp, zero AT each edge.

        Both of a strip's y-faces are artificial (the torus has no real
        boundary), so both ramp.  Convex and summing to one by construction,
        which is L6/C1 and what R11 refuses without.
        """
        r = max(self.ramp, 1)
        idx = np.arange(self.strip_rows) + 0.5
        prof = np.minimum(np.clip(idx / r, 0.0, 1.0),
                          np.clip((self.strip_rows - idx) / r, 0.0, 1.0)) ** 2
        raw = []
        for i in range(self.n_strips):
            w = np.zeros((self.phys.ny, self.phys.nx))
            w[self.rows(i), :] = prof[:, None]
            raw.append(w)
        total = np.sum(raw, axis=0)
        total = np.where(total <= 0.0, 1.0, total)
        return [w / total for w in raw]

    def assemble(self, locals_: list[np.ndarray]) -> np.ndarray:
        out = np.zeros((self.phys.ny, self.phys.nx))
        for i, w in enumerate(self.weights()):
            out[self.rows(i), :] += w[self.rows(i), :] * locals_[i]
        return out

    # -- the two artificial faces of each strip, as global rows ------------

    def face_row(self, i: int, face: str) -> int:
        rows = self.rows(i)
        return int(rows[0] if face == "ylo" else rows[-1])


# ---------------------------------------------------------------------------
# the scheme: local solves with lagged Dirichlet halo data, then assembly
# ---------------------------------------------------------------------------


def local_step(dec: StripDecomposition, u_strip: np.ndarray, bc: np.ndarray,
               dt: float, i: int) -> np.ndarray:
    """One sub-step on strip ``i`` with its artificial rows pinned to ``bc``.

    ``bc`` is the strip-shaped array the boundary rows are read from -- the
    lagged global field in the composed scheme, the reference's own field when
    measuring ``tau``.
    """
    w = u_strip.copy()
    w[0] = bc[0]
    w[-1] = bc[-1]
    out = heun(w, dt, dec.phys, dec.nu_of(i), y_periodic=False)
    out[0] = bc[0]
    out[-1] = bc[-1]
    return out


def reference_step(dec: StripDecomposition, u: np.ndarray, dt: float) -> np.ndarray:
    """E: one sub-step of the SAME scheme on the whole torus."""
    return heun(u, dt, dec.phys, dec.phys.nu_field(), y_periodic=True)


def composed_step(dec: StripDecomposition, u: np.ndarray, dt: float,
                  exact_bc: np.ndarray | None = None) -> np.ndarray:
    """One composed exchange interval: cut, solve locally, assemble.

    ``exact_bc`` supplies the reference's own field as boundary data, which is
    what makes the resulting defect ``tau`` rather than the total.
    """
    src = u if exact_bc is None else exact_bc
    locals_ = [local_step(dec, dec.restrict(u, i), dec.restrict(src, i), dt, i)
               for i in range(dec.n_strips)]
    return dec.assemble(locals_)


def restriction_defect(dec: StripDecomposition, u: np.ndarray,
                       dt: float) -> list[np.ndarray]:
    """D_i = E_i R_i u - R_i E u, one per strip, on the strip's own cells.

    **This is the object G5/W16's phrase "the exact operator is closest to
    local" denotes.**  It is the failure of the exact one-interval operator to
    commute with restriction to the strip -- a commutator, with the scheme's own
    boundary handling inside ``E_i`` because the truncation of the domain and the
    staleness of the datum it forces are two consequences of one cut and the
    criterion must charge for both.
    """
    ref = reference_step(dec, u, dt)
    out = []
    for i in range(dec.n_strips):
        ri = dec.restrict(u, i)
        out.append(local_step(dec, ri, ri, dt, i) - dec.restrict(ref, i))
    return out


def neighbour_disagreement(dec: StripDecomposition, u: np.ndarray,
                           dt: float) -> np.ndarray:
    """|E_i R_i u - E_j R_j u| on the overlap: D_i - D_j, with no reference.

    ``restriction_defect`` needs ``E u`` -- the monolithic solve a decomposition
    exists precisely to avoid -- so a criterion built on it is a diagnostic and
    not a condition, which is the distinction `AssemblyCertificate`'s
    ``must_satisfy`` drew and L6/C1 had to satisfy.  This is the computable half.

    On the overlap ``R_i E u = R_j E u``, so the two agents' disagreement is
    exactly ``D_i - D_j`` there: the common mode cancels and what survives is the
    part of the restriction defect **the cut itself creates**.  It costs nothing
    -- both local solves are already computed by the composed step.
    """
    locals_ = [local_step(dec, dec.restrict(u, i), dec.restrict(u, i), dt, i)
               for i in range(dec.n_strips)]
    glob = [np.zeros((dec.phys.ny, dec.phys.nx)) for _ in range(dec.n_strips)]
    own = [np.zeros((dec.phys.ny, dec.phys.nx), dtype=bool)
           for _ in range(dec.n_strips)]
    for i in range(dec.n_strips):
        r = dec.rows(i)
        glob[i][r, :] = locals_[i]
        own[i][r, :] = True
    out = np.zeros((dec.phys.ny, dec.phys.nx))
    for i in range(dec.n_strips):
        for j in range(i + 1, dec.n_strips):
            both = own[i] & own[j]
            out = np.maximum(out, np.abs(glob[i] - glob[j]) * both)
    return out


def cellwise_defect_bound(dec: StripDecomposition, defects: list[np.ndarray],
                          weighted: bool = False) -> np.ndarray:
    """The right-hand side of the L2/C2 inequality, cell by cell.

    Two forms, both following from ``chi >= 0`` and ``sum_i chi_i = 1`` and
    nothing else:

    ``weighted=False``  ``max_i |D_i|``, the same shape as L6/C1's cellwise
                        statement and the one that reads as *"the cut is only as
                        good as its worst subdomain"*.
    ``weighted=True``   ``sum_i chi_i |D_i|``, the triangle inequality applied to
                        the identity directly.  Convexity makes it **tighter**
                        (``sum_i chi_i |D_i| <= max_i |D_i|``) and it is the form
                        that charges the cut for exactly the cells the assembly
                        weights, which is the same accounting W49's ``Pi`` does
                        on the transmission side.

    ``chi_ij > 0`` is the ownership test rather than mere membership of the
    strip: a cell a strip carries at zero weight contributes nothing to the blend
    and charging the cut for it would make the bound loose exactly where the
    partition of unity was designed to make it tight.
    """
    ws = dec.weights()
    out = np.zeros((dec.phys.ny, dec.phys.nx))
    for i in range(dec.n_strips):
        rows = dec.rows(i)
        chi = ws[i][rows, :]
        if weighted:
            out[rows, :] += chi * np.abs(defects[i])
        else:
            out[rows, :] = np.maximum(out[rows, :], np.abs(defects[i]) * (chi > 0.0))
    return out


# ---------------------------------------------------------------------------
# the interface operator, probed exactly as `atlas.probe` probes a real one
# ---------------------------------------------------------------------------


def fourier_basis(n: int, m: int, h: float) -> np.ndarray:
    """(n, m) real Fourier modes, orthonormal in the h-weighted pairing."""
    x = (np.arange(n) + 0.5) / n
    cols = [np.ones(n)]
    k = 1
    while len(cols) < m:
        cols.append(np.sqrt(2.0) * np.cos(2.0 * np.pi * k * x))
        if len(cols) < m:
            cols.append(np.sqrt(2.0) * np.sin(2.0 * np.pi * k * x))
        k += 1
    B = np.column_stack(cols[:m]) / np.sqrt(h * n)
    return B


def probe_block(dec: StripDecomposition, u: np.ndarray, i: int, face: str,
                dt: float, basis: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Lambda_i on one strip's own artificial face: trace -> nu du/dn, outward.

    `probed-dtn-coupling` §2.2's definition literally, and `window_ns.py`'s
    ``WindowAgent.respond`` line for line: perturb the ring, take one interval,
    read the one-sided outward normal derivative, difference against the zero
    probe.
    """
    ring, interior = (0, 1) if face == "ylo" else (-1, -2)
    hy = dec.phys.hy
    nu_face = dec.nu_of(i)[ring]
    base = dec.restrict(u, i)

    def flux_for(trace: np.ndarray) -> np.ndarray:
        bc = base.copy()
        bc[ring] = bc[ring] + trace
        w = local_step(dec, base, bc, dt, i)
        return nu_face * (w[ring] - w[interior]) / hy

    zero = flux_for(np.zeros(dec.phys.nx))
    cols = [(flux_for(eps * basis[:, k]) - zero) / eps
            for k in range(basis.shape[1])]
    # The reduction is forced as P* -- B^T G_V with G_V = h I -- exactly as
    # `window_ns.py` declares it and `interface-transfer-theory` §4.3 requires.
    return dec.phys.hx * (basis.T @ np.column_stack(cols))


def seam_operator(dec: StripDecomposition, u: np.ndarray, dt: float,
                  m: int = 12, eps: float = 1e-6) -> np.ndarray:
    """S = Lambda_0 + Lambda_1 on the overlap band bounded by strip 0's ylo.

    The two sides read faces ``halo - 1`` rows apart, which is what an
    *overlapping* decomposition means and is the configuration §8.5 measured the
    two flux conventions differing by 3.9e-2 on.  There is no shared layer here
    and the sum is the assembled operator the compiler would report.
    """
    B = fourier_basis(dec.phys.nx, m, dec.phys.hx)
    a = probe_block(dec, u, 0, "ylo", dt, B, eps)
    b = probe_block(dec, u, 1, "yhi", dt, B, eps)
    return a + b


# ---------------------------------------------------------------------------
# a smooth random state
# ---------------------------------------------------------------------------


def smooth_field(phys: Physics, seed: int = 0, k_max: int = 4) -> np.ndarray:
    """A band-limited random field, normalized to unit RMS.

    Band-limited so the defect is a property of the operator and the cut rather
    than of grid-scale noise, and normalized so defects are comparable across
    seeds.
    """
    rng = np.random.default_rng(seed)
    y = (np.arange(phys.ny) + 0.5) / phys.ny
    x = (np.arange(phys.nx) + 0.5) / phys.nx
    X, Y = np.meshgrid(x, y)
    out = np.zeros((phys.ny, phys.nx))
    for kx in range(-k_max, k_max + 1):
        for ky in range(-k_max, k_max + 1):
            if kx == 0 and ky == 0:
                continue
            amp = rng.normal() / (1.0 + kx * kx + ky * ky)
            ph = rng.uniform(0.0, 2.0 * np.pi)
            out += amp * np.cos(2.0 * np.pi * (kx * X + ky * Y) + ph)
    return out / np.sqrt(np.mean(out**2))


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))
