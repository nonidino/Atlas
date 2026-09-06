"""Body-frame contours and the declared interface curves.

Everything geometric in Atlas 0.1 is generated from three profiles:

    h_in(z)   nozzle/chamber INNER wall  -- the gas cavity half-height
    h_out(z)  airframe OUTER wall        -- h_in(z) + shell thickness
    h_p(z)    plume half-width           -- the f/g partition

and the seven declared interface curves are slices of those three plus three
constant-z planes. Keeping them in one file is what makes "every EdgeSpec.curve
lies on both agents' boundaries" checkable rather than aspirational: the curve
and the agent mask are computed from the *same* function, so they cannot drift.

All lengths are metres in the body-fixed planar frame: z axial and increasing
downstream (injector face z=0), y transverse and symmetric about y=0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import GeometryCfg

# --------------------------------------------------------------------------
# profiles
# --------------------------------------------------------------------------


def _throat_tangent_slope(dz: float, dh: float, R: float, prefer_max: bool) -> float:
    """Slope of the line through a fixed endpoint that is tangent to the throat arc.

    The arc is the circle of radius `R` centred at (z_throat, h_throat + R), so its
    lowest point IS the declared throat: rounding the corner must not move the
    throat station or open the throat area, or the area-Mach oracle would be
    graded against a nozzle the config does not describe.

    `dz`, `dh` are the centre's offset from the endpoint. Requiring the distance
    from the centre to the line h = h0 + m (z - z0) to equal R gives a quadratic
    in m; the two roots are the two tangent lines from an external point and
    `prefer_max` picks the branch that actually wraps the arc."""
    A = dz * dz - R * R
    B = -2.0 * dz * dh
    C = dh * dh - R * R
    if abs(A) < 1e-14:
        return -C / B
    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        raise ValueError(f"no tangent line: endpoint is inside the throat arc (R={R})")
    r = np.sqrt(disc)
    m1, m2 = (-B + r) / (2.0 * A), (-B - r) / (2.0 * A)
    return max(m1, m2) if prefer_max else min(m1, m2)


def throat_arc(geo: GeometryCfg):
    """(R, z_lo, m_conv, z_hi, m_div) for the rounded throat, or None if R == 0.

    Decision D2 (corpus completion plan): the declared contour has a hard corner
    at the throat where the wall angle jumps from -31 deg to +15 deg, which curves
    the sonic line and produces a Prandtl-Meyer fan -- so quasi-1D theory does not
    describe the flow within ~15% of the throat, and the area-Mach oracle fails on
    geometry rather than on discretization (implementation log, 2026-08-09).
    Refining made it worse; smoothing halved it. The corpus bakes the geometry in
    permanently, so the throat is rounded BEFORE generation, not after."""
    R = float(getattr(geo, "throat_round_radius", 0.0) or 0.0)
    if R <= 0.0:
        return None
    ht, zt = geo.throat_halfheight, geo.z_throat
    z0c, z1c = geo.z_converge
    z0d, z1d = geo.z_diverge
    m_conv = _throat_tangent_slope(zt - z0c, (ht + R) - geo.chamber_halfheight,
                                   R, prefer_max=False)
    m_div = _throat_tangent_slope(zt - z1d, (ht + R) - geo.exit_halfheight,
                                  R, prefer_max=True)
    # tangency stations: h'(s) = s / sqrt(R^2 - s^2) = m  =>  s = m R / sqrt(1 + m^2)
    z_lo = zt + m_conv * R / np.sqrt(1.0 + m_conv * m_conv)
    z_hi = zt + m_div * R / np.sqrt(1.0 + m_div * m_div)
    if not (z0c < z_lo < zt < z_hi < z1d):
        raise ValueError(
            f"throat_round_radius={R} puts the arc outside the converging/diverging "
            f"sections (z_lo={z_lo:.4f}, z_hi={z_hi:.4f})")
    return R, z_lo, m_conv, z_hi, m_div


def nozzle_inner(z, geo: GeometryCfg):
    """Gas-cavity half-height h_in(z), vectorized over `z`.

    Flat chamber, converging section, rounded throat, diverging section. Defined
    for all z <= z_exit; forward of the injector it continues flat (that is the
    tank barrel, whose interior is declared unmodelled, but the WALL still exists
    and agent `c` and agent `d` both need it).

    With `geometry.throat_round_radius > 0` the corner at z_throat is replaced by
    a circular arc whose lowest point is exactly (z_throat, throat_halfheight),
    and the converging and diverging walls become the tangent lines from their
    fixed endpoints to that arc -- so h(z_throat), h(z_converge[0]) and
    h(z_diverge[1]) are all unchanged and the profile is C1 everywhere between
    them. Setting the radius to 0 reproduces the original piecewise-linear
    contour exactly, which is what the geometry regression test uses."""
    z = np.asarray(z, dtype=np.float64)
    hc, ht, he = geo.chamber_halfheight, geo.throat_halfheight, geo.exit_halfheight
    z0c, z1c = geo.z_converge
    z0d, z1d = geo.z_diverge

    arc = throat_arc(geo)
    if arc is None:
        h = np.full_like(z, hc)
        conv = (z > z0c) & (z <= z1c)
        h = np.where(conv, hc + (ht - hc) * (z - z0c) / (z1c - z0c), h)
        div = z > z0d
        h = np.where(div, ht + (he - ht) * np.clip((z - z0d) / (z1d - z0d), 0.0, 1.0), h)
        return h

    R, z_lo, m_conv, z_hi, m_div = arc
    zt = geo.z_throat
    h = np.full_like(z, hc)
    h = np.where(z > z0c, hc + m_conv * (z - z0c), h)                  # converging wall
    s = np.clip(z - zt, -R + 1e-15, R - 1e-15)
    h = np.where((z >= z_lo) & (z <= z_hi), ht + R - np.sqrt(R * R - s * s), h)
    h = np.where(z > z_hi, he + m_div * (z - z1d), h)                  # diverging wall
    return np.where(z > z1d, he, h)


def shell_outer(z, geo: GeometryCfg):
    """Airframe outer half-height h_out(z) = h_in(z) + t over the vehicle, held
    constant forward of the nose.

    The constant extension for z < shell_z[0] is a *simplification*: it keeps
    agent `d`'s body-fitted grid a single two-panel map over its whole z range
    instead of special-casing the region ahead of the vehicle. The cost is the
    `forebody_slot` region, declared unmodelled in the config."""
    z = np.asarray(z, dtype=np.float64)
    z_nose = geo.shell_z[0]
    zc = np.maximum(z, z_nose)
    return nozzle_inner(zc, geo) + geo.shell_thickness


def plume_half(z, geo: GeometryCfg):
    """Plume half-width h_p(z). Constant in 0.1 -- a spreading shear layer is a
    Phase-5 refinement (it changes agent `g`'s token count, so it is not a
    free change)."""
    z = np.asarray(z, dtype=np.float64)
    return np.full_like(z, geo.plume_halfwidth)


# --------------------------------------------------------------------------
# polyline machinery
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Branch:
    """One connected polyline of an interface, arc-length parameterized.

    Interfaces that appear twice by symmetry (an upper and a lower wall) are two
    branches, never one polyline: matching happens WITHIN a branch, so an
    upper-wall token can never bond to a lower-wall token even though the two
    sit at the same arc length."""
    points: np.ndarray      # [M, 2] (z, y)

    @property
    def seg_vec(self) -> np.ndarray:
        return self.points[1:] - self.points[:-1]          # [M-1, 2]

    @property
    def seg_len(self) -> np.ndarray:
        return np.linalg.norm(self.seg_vec, axis=1)        # [M-1]

    @property
    def s_nodes(self) -> np.ndarray:
        return np.concatenate([[0.0], np.cumsum(self.seg_len)])

    @property
    def length(self) -> float:
        return float(self.s_nodes[-1])

    def project(self, pts: np.ndarray):
        """Nearest point on this branch for each query.

        pts [N, 2] -> (s [N], dist [N], normal [N, 2]) where `normal` is the unit
        segment normal at the foot of the projection (arbitrary sign; callers
        orient it per token pair)."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        a = self.points[:-1][None]                          # [1, S, 2]
        v = self.seg_vec[None]                              # [1, S, 2]
        vv = np.maximum((v * v).sum(-1), 1e-30)             # [1, S]
        w = pts[:, None, :] - a                             # [N, S, 2]
        t = np.clip((w * v).sum(-1) / vv, 0.0, 1.0)         # [N, S]
        foot = a + t[..., None] * v                         # [N, S, 2]
        d = np.linalg.norm(pts[:, None, :] - foot, axis=-1)  # [N, S]
        j = np.argmin(d, axis=1)                            # [N]
        n = np.arange(pts.shape[0])
        s = self.s_nodes[j] + t[n, j] * self.seg_len[j]
        tang = self.seg_vec[j] / np.maximum(self.seg_len[j][:, None], 1e-30)
        normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
        return s, d[n, j], normal


@dataclass(frozen=True)
class InterfaceCurve:
    """The geometric realization of one declared typed edge."""
    name: str
    branches: tuple[Branch, ...]
    declared_span: str = ""     # non-empty when the master plan's written span and
                                # the implemented curve differ; see plane_d_g.

    @property
    def length(self) -> float:
        return float(sum(b.length for b in self.branches))

    def project(self, pts: np.ndarray):
        """pts [N,2] -> (branch [N] int, s [N], dist [N], normal [N,2]), taking
        the nearest branch per query."""
        res = [b.project(pts) for b in self.branches]
        dists = np.stack([r[1] for r in res], axis=0)       # [B, N]
        bi = np.argmin(dists, axis=0)                       # [N]
        n = np.arange(pts.shape[0])
        s = np.stack([r[0] for r in res], axis=0)[bi, n]
        d = dists[bi, n]
        nrm = np.stack([r[2] for r in res], axis=0)[bi, n]
        return bi, s, d, nrm


def _zs(z0: float, z1: float, kinks: tuple[float, ...], n: int = 241) -> np.ndarray:
    """Samples of [z0, z1] that always include the profile kinks, so a polyline
    never rounds a corner the mask treats as sharp."""
    pts = np.linspace(z0, z1, n)
    extra = [k for k in kinks if z0 < k < z1]
    return np.unique(np.concatenate([pts, np.asarray(extra, dtype=np.float64)]))


def _pair(z: np.ndarray, h: np.ndarray) -> tuple[Branch, Branch]:
    """Lower and upper branches of a symmetric wall profile."""
    return (
        Branch(np.stack([z, -h], axis=1)),
        Branch(np.stack([z, +h], axis=1)),
    )


# --------------------------------------------------------------------------
# the seven declared interfaces
# --------------------------------------------------------------------------


def plane_a_b(geo: GeometryCfg) -> InterfaceCurve:
    """Reaction zone -> chamber: the plane z = 0.12, |y| <= chamber half-height."""
    zc = 0.12
    h = geo.chamber_halfheight
    y = np.linspace(-h, h, 81)
    return InterfaceCurve("a-b", (Branch(np.stack([np.full_like(y, zc), y], axis=1)),))


def plane_e_b(geo: GeometryCfg) -> InterfaceCurve:
    """Chamber -> nozzle: the THROAT plane z = z_throat, |y| <= throat half-height."""
    ht = geo.throat_halfheight
    y = np.linspace(-ht, ht, 41)
    return InterfaceCurve("e-b", (Branch(np.stack([np.full_like(y, geo.z_throat), y], axis=1)),))


def wall_b_c(geo: GeometryCfg) -> InterfaceCurve:
    """Chamber inner wall, z in [0.12, z_throat] -- the converging contour."""
    z = _zs(0.12, geo.z_throat, (geo.z_converge[0],))
    return InterfaceCurve("b-c", _pair(z, nozzle_inner(z, geo)))


def wall_c_d(geo: GeometryCfg) -> InterfaceCurve:
    """Airframe outer wall over the whole vehicle, z in shell_z."""
    z0, z1 = geo.shell_z
    z = _zs(z0, z1, (geo.z_converge[0], geo.z_throat))
    return InterfaceCurve("c-d", _pair(z, shell_outer(z, geo)))


def plane_e_f(geo: GeometryCfg) -> InterfaceCurve:
    """Nozzle exit plane z = z_exit, |y| <= exit half-height. Conservation-typed."""
    he = geo.exit_halfheight
    y = np.linspace(-he, he, 61)
    return InterfaceCurve("e-f", (Branch(np.stack([np.full_like(y, geo.z_exit), y], axis=1)),))


def plane_d_g(geo: GeometryCfg) -> InterfaceCurve:
    """Atmosphere front -> wake across z = z_exit, OUTBOARD OF THE PLUME.

    The master plan writes this interface as `0.12 < |y| <= 1.5`. Implemented as
    `plume_halfwidth < |y| <= 1.5`, because in the partition the plan itself
    defines, the strip `exit_halfheight < |y| <= plume_halfwidth` at z = z_exit
    has agent `f` on its downstream side, not `g` -- so the wider curve does not
    lie on `g`'s boundary and fails this phase's own import-time assertion.

    The consequence is a real, currently UNDECLARED d-f interface over that
    strip. Recorded in the implementation log as an open item for Phase 3; not
    silently patched here, because adding an edge is a design decision."""
    hp, hf = geo.plume_halfwidth, geo.farfield_halfwidth
    y = np.linspace(hp, hf, 61)
    ze = np.full_like(y, geo.z_exit)
    return InterfaceCurve(
        "d-g",
        (Branch(np.stack([ze, -y], axis=1)), Branch(np.stack([ze, y], axis=1))),
        declared_span="master plan: 0.12 < |y| <= 1.5; implemented: plume_halfwidth < |y| <= 1.5",
    )


def shear_g_f(geo: GeometryCfg) -> InterfaceCurve:
    """Plume shear layer |y| = plume_halfwidth over the plume's z extent."""
    z0, z1 = geo.plume_z
    z = np.linspace(z0, z1, 241)
    return InterfaceCurve("g-f", _pair(z, plume_half(z, geo)))


INTERFACES = {
    "plane_a_b": plane_a_b,
    "plane_e_b": plane_e_b,
    "wall_b_c": wall_b_c,
    "wall_c_d": wall_c_d,
    "plane_e_f": plane_e_f,
    "plane_d_g": plane_d_g,
    "shear_g_f": shear_g_f,
}


def build_interface(name: str, geo: GeometryCfg) -> InterfaceCurve:
    try:
        return INTERFACES[name](geo)
    except KeyError:
        raise KeyError(f"unknown interface {name!r}; have {sorted(INTERFACES)}") from None


__all__ = [
    "nozzle_inner", "shell_outer", "plume_half", "throat_arc",
    "Branch", "InterfaceCurve", "build_interface", "INTERFACES",
]
