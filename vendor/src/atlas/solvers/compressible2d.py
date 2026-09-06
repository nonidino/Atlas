"""2D compressible Navier-Stokes finite-volume solver.

One class serves all six gas agents, in two modes: *internal* (walls + a heat
release source) for `a`, `b`, `e`, and *external* (freestream in/out) for `d`,
`f`, `g`. The difference is entirely in the boundary conditions and the presence
of a source term — the discretization is identical, which is what makes the two
modes testable against the same oracles.

Scheme (impl-atlas-0.1-phase0-scope-and-data §2.1):
  * MUSCL reconstruction on primitives with the minmod limiter — second order in
    smooth regions, TVD across shocks. The plume genuinely contains shocks, so a
    non-TVD scheme is not an option.
  * HLLC for the inviscid flux, Rusanov as the robust fallback. Rusanov is the
    bring-up default; HLLC is the production default.
  * Central differences for the viscous flux.
  * SSP-RK2 in time, CFL 0.4.
  * Strang splitting for the reaction source (half source / full convection /
    half source).

Gradients use the metric identity that every Atlas block satisfies by
construction — **z depends only on the i index** — so with $J = z_i y_j$,

    df/dz = (f_i y_j - f_j y_i)/J,      df/dy = f_j z_i / J

which is exact on the curvilinear nozzle blocks, not an orthogonality
assumption. `Block` asserts the premise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grid import Block
from .riemann import FLUXES
from . import thermo

NG = 2                    # ghost layers (MUSCL needs 2)


@dataclass
class GasConfig:
    gamma: float
    R: float
    mu_ref: float = 1.716e-5
    T_mu_ref: float = 273.15
    sutherland_S: float = 110.4
    Pr: float = 0.72
    riemann: str = "hllc"
    limiter: str = "minmod"
    cfl: float = 0.4
    inviscid: bool = False

    @property
    def cp(self) -> float:
        return self.gamma * self.R / (self.gamma - 1.0)


@dataclass
class ReactionConfig:
    """One-step progress variable, unburnt (Y=0) -> burnt (Y=1).

    Deliberately not chemistry: it produces a heat-release-driven temperature
    and pressure rise and its coupling to the downstream chamber, which is all
    Atlas needs to learn. Upgrading it is explicitly out of scope."""
    A: float                 # pre-exponential, 1/s
    T_act: float             # activation temperature, K
    q_rxn: float             # heat of reaction, J/kg


@dataclass
class BC:
    """kind ∈ {wall_slip, wall_noslip, symmetry, inlet_massflow, freestream,
    outflow, prescribed, extrapolate, farfield}."""
    kind: str
    params: dict = field(default_factory=dict)


def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(a * b <= 0.0, 0.0, np.where(np.abs(a) < np.abs(b), a, b))


LIMITERS = {"minmod": minmod, "none": lambda a, b: 0.5 * (a + b)}


class Compressible2D:
    def __init__(self, block: Block, cfg: GasConfig,
                 reaction: ReactionConfig | None = None,
                 bcs: dict[str, BC] | None = None):
        self.block, self.cfg, self.reaction = block, cfg, reaction
        self.bcs = bcs or {}
        self.nv = 5 if reaction is not None else 4
        self.hole_state: np.ndarray | None = None   # set by the coupler; see ghost()
        self._check_metric_premise()
        self._hole = self._hole_band()

    # -- geometry helpers ---------------------------------------------------
    def _check_metric_premise(self) -> None:
        z = self.block.nodes[..., 0]
        if not np.allclose(z, z[:, :1]):
            raise ValueError(
                f"block {self.block.name}: z varies with the j index; the gradient "
                "metric identity in this solver assumes z = z(i)")

    def _hole_band(self):
        """Blanked cells must form a contiguous j band, constant in i (agent `g`'s
        plume). Returns (j_lo, j_hi) or None."""
        b = self.block.blanked
        if not b.any():
            return None
        if not np.all(b == b[:1]):
            raise ValueError(f"block {self.block.name}: blanked region varies with i")
        js = np.flatnonzero(b[0])
        if js.size != js[-1] - js[0] + 1:
            raise ValueError(f"block {self.block.name}: blanked region is not contiguous in j")
        return int(js[0]), int(js[-1])

    # -- ghosts -------------------------------------------------------------
    def _reflect(self, U, n, no_slip: bool):
        """Mirror a state across a face with unit normal `n`."""
        out = U.copy()
        if no_slip:
            out[..., 1] = -U[..., 1]
            out[..., 2] = -U[..., 2]
            return out
        mom = U[..., 1:3]
        dot = (mom * n).sum(-1, keepdims=True)
        out[..., 1:3] = mom - 2.0 * dot * n
        return out

    def _fill_side(self, Ue, U, side: str) -> None:
        """Fill the two ghost layers of one side, IN GHOST ORDER.

        Mirror-type conditions (wall, symmetry) reverse the interior stack: the
        outermost ghost mirrors the second interior cell. Everything else is
        built from the single adjacent edge cell and written to both layers --
        writing a reversed stack there would put a spurious gradient in the
        ghosts and make an outflow boundary weakly reflective."""
        g, cfg, blk = NG, self.cfg, self.block
        bc = self.bcs.get(side, BC("extrapolate"))
        nz, ny = blk.shape
        i_dir = side in ("imin", "imax")

        if i_dir:
            interior = U[:g] if side == "imin" else U[-g:]
            edge = U[:1] if side == "imin" else U[-1:]
            nrm = (blk.n_i[0] if side == "imin" else blk.n_i[-1])[None, :, :]
        else:
            interior = U[:, :g] if side == "jmin" else U[:, -g:]
            edge = U[:, :1] if side == "jmin" else U[:, -1:]
            nrm = (blk.n_j[:, 0] if side == "jmin" else blk.n_j[:, -1])[:, None, :]

        n_out = nrm if side in ("imax", "jmax") else -nrm

        k = bc.kind
        if k in ("wall_slip", "symmetry", "wall_noslip"):
            ghost = self._reflect(interior, nrm, no_slip=(k == "wall_noslip"))
            if k == "wall_noslip" and "no_slip_mask" in bc.params:
                # Per-station no-slip. A flat plate whose leading edge sits ON the
                # inflow plane is not the Blasius problem: the singularity lands on
                # the boundary and the resulting favourable pressure gradient thins
                # the layer by ~30%. The standard setup runs a slip section first
                # and starts the plate downstream of it.
                m = np.asarray(bc.params["no_slip_mask"], dtype=bool)
                m = m[:, None, None] if not i_dir else m[None, :, None]
                ghost = np.where(m, ghost, self._reflect(interior, nrm, no_slip=False))
            if k == "wall_noslip" and "T_wall" in bc.params:
                # isothermal: choose rho so the FACE temperature is T_wall, at
                # the interior pressure (zero normal pressure gradient).
                p = thermo.pressure_from_cons(interior, cfg.gamma)
                T_i = thermo.temperature(interior, cfg.gamma, cfg.R)
                T_g = np.maximum(2.0 * _broadcast_wall(bc.params["T_wall"], T_i, side) - T_i, 20.0)
                rho_g = p / (cfg.R * T_g)
                scale = rho_g / np.maximum(interior[..., 0], thermo.RHO_FLOOR)
                ghost[..., 0] = rho_g
                ghost[..., 1] = ghost[..., 1] * scale
                ghost[..., 2] = ghost[..., 2] * scale
                ghost[..., 3] = p / (cfg.gamma - 1.0) + 0.5 * (
                    ghost[..., 1] ** 2 + ghost[..., 2] ** 2) / rho_g
            ghost = ghost[::-1] if i_dir else ghost[:, ::-1]
        else:
            if k == "extrapolate":
                one = edge.copy()
            elif k == "prescribed":
                one = np.broadcast_to(np.asarray(bc.params["state"], dtype=float),
                                      edge.shape).copy()
            elif k == "freestream":
                W = np.asarray(bc.params["prim"], dtype=float)      # (rho, u, v, p[, Y])
                one = np.broadcast_to(thermo.prim_to_cons(W, cfg.gamma), edge.shape).copy()
            elif k == "farfield":
                one = self._characteristic_farfield(edge, bc.params["prim"], n_out)
            elif k == "inlet_massflow":
                mdot, T_in = float(bc.params["mdot"]), float(bc.params["T"])
                p = thermo.pressure_from_cons(edge, cfg.gamma)      # extrapolated
                rho = p / (cfg.R * T_in)
                one = np.empty_like(edge)
                one[..., 0] = rho
                one[..., 1] = mdot if side == "imin" else -mdot
                one[..., 2] = 0.0
                one[..., 3] = p / (cfg.gamma - 1.0) + 0.5 * mdot ** 2 / rho
                if self.nv == 5:
                    one[..., 4] = 0.0                               # unburnt
            elif k == "outflow":
                one = edge.copy()
                p_inf = bc.params.get("p_inf")
                if p_inf is not None:
                    # subsonic outflow: prescribe back pressure; supersonic:
                    # extrapolate everything (no upstream influence)
                    p_new = np.where(thermo.mach(edge, cfg.gamma) < 1.0, float(p_inf),
                                     thermo.pressure_from_cons(edge, cfg.gamma))
                    ke = 0.5 * (one[..., 1] ** 2 + one[..., 2] ** 2) / one[..., 0]
                    one[..., 3] = p_new / (cfg.gamma - 1.0) + ke
            else:
                raise KeyError(f"unknown BC kind {k!r}")
            ghost = np.repeat(one, g, axis=0 if i_dir else 1)

        if side == "imin":
            Ue[:g, NG:ny + NG] = ghost
        elif side == "imax":
            Ue[nz + g:, NG:ny + NG] = ghost
        elif side == "jmin":
            Ue[g:nz + g, :g] = ghost
        else:
            Ue[g:nz + g, ny + g:] = ghost

    def _characteristic_farfield(self, edge, prim_inf, n_out):
        """Riemann-invariant far field (decision D3).

        A `freestream` ghost PRESCRIBES all four primitives, which over-specifies a
        subsonic boundary: the outgoing acoustic wave has nowhere to go and is
        reflected back into the domain. On a flat plate that shows up as a ~3.5%
        freestream acceleration, and an accelerating outer flow thins the boundary
        layer -- which is exactly why the Blasius oracle ran 15-19% low
        (implementation log, 2026-08-09, open action item 8).

        This BC instead carries the RIGHT number of pieces of information across
        the boundary, chosen by the sign of the normal characteristics. With `n`
        the OUTWARD normal and un = u.n:

            R+ = un_i + 2 c_i / (gamma - 1)      travels out of the domain
            R- = un_o - 2 c_o / (gamma - 1)      travels in from the far field

        so un_b = (R+ + R-)/2 and c_b = (gamma-1)(R+ - R-)/4, while entropy and the
        tangential velocity come from whichever side the flow is arriving from.
        Supersonic faces degenerate correctly: everything from outside on inflow,
        everything extrapolated on outflow.

        `prim_inf` is (rho, u, v, p[, Y]) in the body frame; `n_out` broadcasts
        against `edge`."""
        cfg = self.cfg
        g = cfg.gamma
        gm1 = g - 1.0
        W = thermo.cons_to_prim(edge, g)
        rho_i = np.maximum(W[..., 0], thermo.RHO_FLOOR)
        u_i, v_i = W[..., 1], W[..., 2]
        p_i = np.maximum(W[..., 3], thermo.P_FLOOR)
        c_i = np.sqrt(g * p_i / rho_i)
        nz_, ny_ = n_out[..., 0], n_out[..., 1]
        un_i = u_i * nz_ + v_i * ny_

        Wo = np.asarray(prim_inf, dtype=float)
        rho_o, u_o, v_o, p_o = float(Wo[0]), float(Wo[1]), float(Wo[2]), float(Wo[3])
        c_o = np.sqrt(g * p_o / rho_o)
        un_o = u_o * nz_ + v_o * ny_

        r_plus = un_i + 2.0 * c_i / gm1
        r_minus = un_o - 2.0 * c_o / gm1
        un_b = 0.5 * (r_plus + r_minus)
        c_b = np.maximum(0.25 * gm1 * (r_plus - r_minus), 1e-6)

        inflow = un_b <= 0.0
        s_i = p_i / rho_i ** g
        s_o = p_o / rho_o ** g
        s_b = np.where(inflow, s_o, s_i)
        ut_z = np.where(inflow, u_o - un_o * nz_, u_i - un_i * nz_)
        ut_y = np.where(inflow, v_o - un_o * ny_, v_i - un_i * ny_)

        rho_b = (c_b * c_b / (g * s_b)) ** (1.0 / gm1)
        p_b = rho_b * c_b * c_b / g
        u_b = ut_z + un_b * nz_
        v_b = ut_y + un_b * ny_

        # supersonic faces: no mixing, the whole state comes from one side
        mn = un_i / c_i
        sup_in = mn <= -1.0
        sup_out = mn >= 1.0
        rho_b = np.where(sup_in, rho_o, np.where(sup_out, rho_i, rho_b))
        u_b = np.where(sup_in, u_o, np.where(sup_out, u_i, u_b))
        v_b = np.where(sup_in, v_o, np.where(sup_out, v_i, v_b))
        p_b = np.where(sup_in, p_o, np.where(sup_out, p_i, p_b))

        out = np.empty(edge.shape)
        Wb = np.stack([rho_b, u_b, v_b, p_b], axis=-1)
        if self.nv == 5:
            y_o = float(Wo[4]) if Wo.size > 4 else 0.0
            Wb = np.concatenate([Wb, np.where(inflow, y_o, W[..., 4])[..., None]], axis=-1)
        out[...] = thermo.prim_to_cons(Wb, g)
        return out

    def ghost(self, U: np.ndarray) -> np.ndarray:
        nz, ny = self.block.shape
        Ue = np.zeros((nz + 2 * NG, ny + 2 * NG, self.nv))
        Ue[NG:nz + NG, NG:ny + NG] = U
        for side in ("imin", "imax", "jmin", "jmax"):
            self._fill_side(Ue, U, side)
        # corners: copy from the nearest filled i-ghost row (never used by the
        # dimension-by-dimension reconstruction, but must not be NaN/zero)
        Ue[:NG, :NG] = Ue[:NG, NG:NG + 1]; Ue[-NG:, :NG] = Ue[-NG:, NG:NG + 1]
        Ue[:NG, -NG:] = Ue[:NG, ny + NG - 1:ny + NG]
        Ue[-NG:, -NG:] = Ue[-NG:, ny + NG - 1:ny + NG]

        if self._hole is not None:
            # The blanked band is another agent's domain (agent `g`'s plume hole).
            # If the coupler supplies that agent's state, use it -- the g-f edge is
            # a declared interface, not a wall. Fall back to a slip wall only when
            # the block is run solo.
            j0, j1 = self._hole
            n = self.block.n_j[:, j0][:, None, :]
            hs = self.hole_state
            for d in range(NG):
                if hs is None:
                    lo = self._reflect(U[:, j0 - 1 - d][:, None], n, False)[:, 0]
                    hi = self._reflect(U[:, j1 + 1 + d][:, None], n, False)[:, 0]
                else:
                    lo = hi = hs
                Ue[NG:nz + NG, NG + j0 + d] = lo
                Ue[NG:nz + NG, NG + j1 - d] = hi
        return Ue

    # -- reconstruction and fluxes -----------------------------------------
    def _faces(self, Ue: np.ndarray, axis: int):
        """MUSCL-limited left/right states on every face along `axis` (0 = i)."""
        gamma = self.cfg.gamma
        We = thermo.cons_to_prim(Ue, gamma)
        lim = LIMITERS[self.cfg.limiter]
        d = np.diff(We, axis=axis)
        sl = [slice(None)] * We.ndim

        def take(a, b):
            s = list(sl); s[axis] = slice(a, b); return tuple(s)

        slope = lim(d[take(None, -1)], d[take(1, None)])            # cells 1..-1
        core = We[take(1, -1)]
        WL = (core + 0.5 * slope)[take(None, -1)]
        WR = (core - 0.5 * slope)[take(1, None)]
        return thermo.prim_to_cons(WL, gamma), thermo.prim_to_cons(WR, gamma)

    def _inviscid_residual(self, Ue: np.ndarray) -> np.ndarray:
        blk, gamma = self.block, self.cfg.gamma
        flux = FLUXES[self.cfg.riemann]
        nz, ny = blk.shape

        # _faces on a stack of n cells returns n-3 face states; with 2 ghost
        # layers per side that is exactly the n_faces of the block.
        L, R = self._faces(Ue[:, NG:ny + NG], axis=0)               # [nz+1, ny, nv]
        Fi = flux(L, R, blk.n_i, gamma) * blk.a_i[..., None]

        L, R = self._faces(Ue[NG:nz + NG, :], axis=1)               # [nz, ny+1, nv]
        Fj = flux(L, R, blk.n_j, gamma) * blk.a_j[..., None]

        return -((Fi[1:] - Fi[:-1]) + (Fj[:, 1:] - Fj[:, :-1])) / blk.vol[..., None]

    # -- gradients and viscous flux ----------------------------------------
    def _grad(self, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """d/dz, d/dy of a cell-centred field defined on the GHOSTED array,
        returned on the ghosted array (one-sided at its outer edge)."""
        zc, yc = self._ghost_coords()
        fi = np.gradient(f, axis=0)
        fj = np.gradient(f, axis=1)
        zi = np.gradient(zc, axis=0)
        yi, yj = np.gradient(yc, axis=0), np.gradient(yc, axis=1)
        J = np.maximum(np.abs(zi * yj), 1e-30) * np.sign(zi * yj + 1e-300)
        return (fi * yj - fj * yi) / J, fj * zi / J

    def _ghost_coords(self):
        cache = getattr(self, "_gc", None)
        if cache is not None:
            return cache
        c = self.block.centroid
        z = np.pad(c[..., 0], NG, mode="reflect", reflect_type="odd")
        y = np.pad(c[..., 1], NG, mode="reflect", reflect_type="odd")
        self._gc = (z, y)
        return self._gc

    def _viscous_residual(self, Ue: np.ndarray) -> np.ndarray:
        cfg, blk = self.cfg, self.block
        nz, ny = blk.shape
        W = thermo.cons_to_prim(Ue, cfg.gamma)
        u, v, p, rho = W[..., 1], W[..., 2], W[..., 3], W[..., 0]
        T = p / (rho * cfg.R)
        mu = thermo.sutherland(T, cfg.mu_ref, cfg.T_mu_ref, cfg.sutherland_S)
        k = mu * cfg.cp / cfg.Pr

        uz, uy = self._grad(u)
        vz, vy = self._grad(v)
        Tz, Ty = self._grad(T)
        div = uz + vy
        lam = -2.0 / 3.0 * mu
        t_zz = 2.0 * mu * uz + lam * div
        t_yy = 2.0 * mu * vy + lam * div
        t_zy = mu * (uy + vz)

        # viscous flux vectors in z and y, averaged onto faces
        Fz = np.stack([np.zeros_like(t_zz), t_zz, t_zy,
                       u * t_zz + v * t_zy + k * Tz], axis=-1)
        Fy = np.stack([np.zeros_like(t_yy), t_zy, t_yy,
                       u * t_zy + v * t_yy + k * Ty], axis=-1)
        if self.nv == 5:
            Fz = np.concatenate([Fz, np.zeros_like(Fz[..., :1])], axis=-1)
            Fy = np.concatenate([Fy, np.zeros_like(Fy[..., :1])], axis=-1)

        # a block face f sits between ghosted cells f+NG-1 and f+NG, so the
        # stack needed to average onto all n+1 faces is [NG-1 : n+NG+1].
        gz = Fz[NG - 1:nz + NG + 1, NG:ny + NG]                     # [nz+2, ny, nv]
        gy = Fy[NG - 1:nz + NG + 1, NG:ny + NG]
        Fi = (0.5 * (gz[:-1] + gz[1:]) * blk.n_i[..., 0, None]
              + 0.5 * (gy[:-1] + gy[1:]) * blk.n_i[..., 1, None]) * blk.a_i[..., None]

        hz = Fz[NG:nz + NG, NG - 1:ny + NG + 1]                     # [nz, ny+2, nv]
        hy = Fy[NG:nz + NG, NG - 1:ny + NG + 1]
        Fj = (0.5 * (hz[:, :-1] + hz[:, 1:]) * blk.n_j[..., 0, None]
              + 0.5 * (hy[:, :-1] + hy[:, 1:]) * blk.n_j[..., 1, None]) * blk.a_j[..., None]

        return ((Fi[1:] - Fi[:-1]) + (Fj[:, 1:] - Fj[:, :-1])) / blk.vol[..., None]

    # -- source -------------------------------------------------------------
    def react(self, U: np.ndarray, dt: float) -> np.ndarray:
        """Strang half-step of the progress-variable source, integrated
        sub-cyclically so a stiff Arrhenius term cannot overshoot Y > 1."""
        if self.reaction is None:
            return U
        r, cfg = self.reaction, self.cfg
        out = U.copy()
        rho = np.maximum(out[..., 0], thermo.RHO_FLOOR)
        Y = np.clip(out[..., 4] / rho, 0.0, 1.0)
        T = thermo.temperature(out, cfg.gamma, cfg.R)
        n_sub = 4
        h = dt / n_sub
        for _ in range(n_sub):
            w = r.A * (1.0 - Y) * np.exp(-r.T_act / np.maximum(T, 1.0))
            dY = np.clip(w * h, 0.0, np.maximum(1.0 - Y, 0.0))
            Y = Y + dY
            out[..., 3] = out[..., 3] + rho * dY * r.q_rxn          # heat release
            T = thermo.temperature(out, cfg.gamma, cfg.R)
        out[..., 4] = rho * Y
        return out

    # -- time stepping ------------------------------------------------------
    def max_stable_dt(self, U: np.ndarray) -> float:
        cfg = self.cfg
        rho = np.maximum(U[..., 0], thermo.RHO_FLOOR)
        speed = np.sqrt(U[..., 1] ** 2 + U[..., 2] ** 2) / rho
        a = thermo.sound_speed(U, cfg.gamma)
        h = self.block.min_spacing()
        live = ~self.block.blanked
        dt = cfg.cfl * np.min(h[live] / (speed + a)[live])
        if not cfg.inviscid:
            mu = thermo.sutherland(thermo.temperature(U, cfg.gamma, cfg.R),
                                   cfg.mu_ref, cfg.T_mu_ref, cfg.sutherland_S)
            dt_v = 0.25 * np.min(rho[live] * h[live] ** 2 / np.maximum(mu[live], 1e-30))
            dt = min(dt, dt_v)
        return float(dt)

    def residual(self, U: np.ndarray) -> np.ndarray:
        Ue = self.ghost(U)
        r = self._inviscid_residual(Ue)
        if not self.cfg.inviscid:
            r = r + self._viscous_residual(Ue)
        if self._hole is not None:
            j0, j1 = self._hole
            r[:, j0:j1 + 1] = 0.0
        return r

    def step(self, U: np.ndarray, dt: float) -> np.ndarray:
        """One SSP-RK2 step, Strang-split around the reaction source."""
        if self.reaction is not None:
            U = self.react(U, 0.5 * dt)
        U1 = U + dt * self.residual(U)
        U2 = 0.5 * (U + U1 + dt * self.residual(U1))
        if self.reaction is not None:
            U2 = self.react(U2, 0.5 * dt)
        return U2

    def advance(self, U: np.ndarray, t_end: float, dt_cap: float | None = None,
                max_steps: int = 10_000_000) -> tuple[np.ndarray, int]:
        """March to `t_end`, subcycling at the CFL limit. Returns (U, n_steps)."""
        t, n = 0.0, 0
        while t < t_end - 1e-15 and n < max_steps:
            dt = self.max_stable_dt(U)
            if dt_cap is not None:
                dt = min(dt, dt_cap)
            dt = min(dt, t_end - t)
            if not np.isfinite(dt) or dt <= 0:
                raise FloatingPointError(f"block {self.block.name}: non-positive dt at t={t}")
            U = self.step(U, dt)
            t += dt
            n += 1
        return U, n

    def run_to(self, U0: np.ndarray, t_end: float, save_every: float):
        """Trajectory of snapshots at fixed intervals."""
        U, t, out, times = U0.copy(), 0.0, [U0.copy()], [0.0]
        while t < t_end - 1e-15:
            dt_win = min(save_every, t_end - t)
            U, _ = self.advance(U, dt_win)
            t += dt_win
            out.append(U.copy())
            times.append(t)
        return np.asarray(times), np.stack(out)


def _broadcast_wall(T_w, like, side):
    T_w = np.asarray(T_w, dtype=float)
    if T_w.ndim == 0:
        return T_w
    return T_w[None, :] if side in ("imin", "imax") else T_w[:, None]


def uniform_state(block: Block, prim, gamma: float, nv: int = 4) -> np.ndarray:
    """Constant-primitive initial condition, [n_z, n_y, nv]."""
    nz, ny = block.shape
    W = np.zeros((nz, ny, nv))
    W[...] = np.asarray(prim, dtype=float)[None, None, :nv]
    return thermo.prim_to_cons(W, gamma)


__all__ = [
    "GasConfig", "ReactionConfig", "BC", "Compressible2D", "uniform_state",
    "minmod", "NG",
]
