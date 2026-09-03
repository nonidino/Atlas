"""Every measurement gate W4 through W11 asks for, built before the system that produces them.

`impl-wind-farm-guide` section 5 is explicit about the ordering and about why:
building the metrics first is *the only way to avoid tuning the system until the
metric looks good*. So each function here is validated in
`tests/atlas/windfarm/test_w4_metrics.py` against a field from `analytic` whose
parameters were put in by hand, and gate W4 is passed by recovering them -- not
by producing a plausible number.

What is here
------------
* **Per-port residuals** for all fifteen declared interfaces. Each side samples
  the field on *its own* token lattice and one side is then mapped onto the
  other's discretization through the port's mapping mode, which is the actual
  partitioned-coupling operation rather than a comparison of two evaluations of
  the same formula at the same point.
* **Thrust residual** per rotor, pre- and post-projection, in a record that
  cannot carry one without the other (guide rule 2).
* **The global power residual** `R(t)`, and the same balance with `P_ext`
  removed -- because a budget that closes only because `P_ext` was chosen to
  close it has proved nothing.
* **Wake diagnostics**: centreline deficit, lateral profiles, Gaussian and
  Jensen fits, and the Jensen / Bastankhah-Porte-Agel corridors.
* **Symmetry** between the two bypass agents, which needs no reference data:
  under symmetric inflow `B+` and `B-` must remain mirror images, so a broken
  symmetry is detectable on its own.
* **Array efficiency** `P_2/P_1`.

What is *not* here
------------------
The thrust projection itself (spec section 7.3) is W5's, in `couple.py`. This
module measures; it does not enforce. Keeping that line sharp is the
enforce-or-measure rule of `conservation-as-constraint-atlas-0.1`, and eleven of
the fifteen ports are on the measure side of it permanently.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field

import numpy as np

from ...ports.residual import (
    PortResidual, PowerLedger, curve_norm, port_residual,
    residual_from_shared_normal, summarize,
)
from ...ports.types import MappingMode, port
from .analytic import Field, U_INF, bpa_corridor, jensen_corridor
from .geometry import (
    AGENT, AGENTS, DOMAIN, GRIDS, INTERFACES, Interface, ROTOR_HALFSPAN,
    WAKE_HALFWIDTH, X_T1, X_T2,
)

#: Gauss-Legendre order used for every interface and area integral. High enough
#: that a smooth field integrates to machine precision, so a residual that is
#: not zero is the field's fault and not the quadrature's.
QUAD_ORDER = 8


# --------------------------------------------------------------------------
# quadrature helpers
# --------------------------------------------------------------------------


def _gl(order: int = QUAD_ORDER):
    return np.polynomial.legendre.leggauss(order)


def _segment_average(f, lo: np.ndarray, hi: np.ndarray, order: int = QUAD_ORDER):
    """Average of `f(s)` over each [lo, hi] segment, by Gauss-Legendre.

    `f` maps a 1-D array of positions to values, either scalar [n] or vector
    [n, 2]. Returns [n_seg] or [n_seg, 2]."""
    xg, wg = _gl(order)
    mid = 0.5 * (lo + hi)[:, None]
    half = 0.5 * (hi - lo)[:, None]
    s = mid + half * xg[None, :]                       # [n_seg, order]
    vals = f(s.ravel())
    if isinstance(vals, tuple):
        vals = np.stack(vals, axis=-1)                 # [n_seg*order, 2]
    vals = np.asarray(vals, dtype=np.float64)
    if vals.ndim == 1:
        vals = vals.reshape(s.shape)
        return 0.5 * np.sum(vals * wg[None, :], axis=1)
    vals = vals.reshape(s.shape + (vals.shape[-1],))
    return 0.5 * np.sum(vals * wg[None, :, None], axis=1)


def _lattice_edges(lo: float, hi: float, spacing: float, origin: float) -> np.ndarray:
    """Boundaries of an agent's token lattice, clipped to [lo, hi].

    The lattice is anchored at `origin` (the agent's own box edge), so the
    segmentation is the agent's real discretization rather than a convenient
    subdivision of the span."""
    i0 = int(np.floor((lo - origin) / spacing + 1e-9))
    i1 = int(np.ceil((hi - origin) / spacing - 1e-9))
    e = origin + np.arange(i0, i1 + 1) * spacing
    e = np.clip(e, lo, hi)
    e = np.unique(np.concatenate([[lo], e, [hi]]))
    return e[(e >= lo - 1e-12) & (e <= hi + 1e-12)]


def _overlap_matrix(a_edges: np.ndarray, b_edges: np.ndarray) -> np.ndarray:
    """`W[i, j]` = length shared by source segment i and destination segment j."""
    a_lo, a_hi = a_edges[:-1, None], a_edges[1:, None]
    b_lo, b_hi = b_edges[None, :-1], b_edges[None, 1:]
    return np.clip(np.minimum(a_hi, b_hi) - np.maximum(a_lo, b_lo), 0.0, None)


# --------------------------------------------------------------------------
# interface sampling
# --------------------------------------------------------------------------


def _tangential_spacing(agent: str, axis: str) -> float:
    """Token size along an interface curve, on the given agent's lattice."""
    dx, dy = AGENT[agent].spacing
    return dy if axis == "x" else dx


def _tangential_origin(agent: str, axis: str) -> float:
    box = AGENT[agent].box
    return box.y0 if axis == "x" else box.x0


def _curve_points(e: Interface, s: np.ndarray):
    """Points on the curve at tangential coordinate `s`."""
    c = np.full_like(s, e.pos, dtype=np.float64)
    return (c, s) if e.axis == "x" else (s, c)


@dataclass(frozen=True)
class InterfaceSample:
    """One side's view of one interface, on that side's own lattice."""
    agent: str
    edges: np.ndarray                 # [n_seg+1] segment boundaries
    velocity: np.ndarray              # [n_seg, 2] segment-average velocity
    mass_flux: np.ndarray             # [n_seg]    segment-average u . n_outward
    traction: np.ndarray              # [n_seg, 2] segment-average sigma . n_outward
    outward: np.ndarray               # [2] this agent's outward normal


def sample_interface(field: Field, e: Interface, side: str, re_eff: float = 1000.0,
                     branch: int = 0) -> InterfaceSample:
    """Sample `field` on one branch of one interface, on `side`'s token lattice.

    Quantities are expressed with respect to **that agent's own outward
    normal**, which is what makes agreement mean `f_src + f_dst = 0` rather than
    a difference -- see `ports.residual`'s convention note."""
    agent = e.src if side == "src" else e.dst
    lo, hi = e.spans[branch]
    edges = _lattice_edges(lo, hi, _tangential_spacing(agent, e.axis),
                           _tangential_origin(agent, e.axis))
    n = e.normal
    sign = 1.0 if side == "src" else -1.0             # normal points src -> dst
    nx, ny = sign * n[0], sign * n[1]

    def vel(s):
        return field.velocity(*_curve_points(e, s))

    v = _segment_average(vel, edges[:-1], edges[1:])   # [n_seg, 2]
    flux = v[:, 0] * nx + v[:, 1] * ny

    def trac(s):
        x, y = _curve_points(e, s)
        p = field.pressure(x, y)
        # sigma . n = -p n + (1/Re)(grad u + grad u^T) . n, evaluated by central
        # differences on the analytic field -- exact enough at 1e-6 for a smooth
        # field and it keeps this module free of any derivative the field must
        # supply itself.
        h = 1e-6
        ux_p, uy_p = field.velocity(x + h * nx, y + h * ny)
        ux_m, uy_m = field.velocity(x - h * nx, y - h * ny)
        dundn_x = (ux_p - ux_m) / (2 * h)
        dundn_y = (uy_p - uy_m) / (2 * h)
        return (-p * nx + 2.0 * dundn_x / re_eff, -p * ny + 2.0 * dundn_y / re_eff)

    t = _segment_average(trac, edges[:-1], edges[1:])
    return InterfaceSample(agent=agent, edges=edges, velocity=v, mass_flux=flux,
                           traction=t, outward=np.array([nx, ny], dtype=np.float64))


def _map_onto(src: InterfaceSample, dst: InterfaceSample, values: np.ndarray,
              mode: MappingMode) -> np.ndarray:
    """Transfer `values` from `src`'s segments to `dst`'s, in the given mode."""
    w = _overlap_matrix(src.edges, dst.edges)
    w_src = np.diff(src.edges)
    w_dst = np.diff(dst.edges)
    if values.ndim == 1:
        return mode.apply(values, w_src, w_dst, w)
    cols = [mode.apply(values[:, c], w_src, w_dst, w) for c in range(values.shape[1])]
    return np.stack(cols, axis=1)


def interface_residuals(field: Field, re_eff: float = 1000.0,
                        interfaces=INTERFACES) -> list[PortResidual]:
    """Residual of every port on every declared interface.

    Both sides sample the field on their own lattices; the source is then mapped
    onto the destination's discretization through the port's own mapping mode,
    and the two are compared there. On a field both agents represent exactly,
    the residual is zero *whatever the two lattices are*, which is the property
    gate W4's first item tests."""
    out: list[PortResidual] = []
    for e in interfaces:
        for b in range(len(e.spans)):
            src = sample_interface(field, e, "src", re_eff, b)
            dst = sample_interface(field, e, "dst", re_eff, b)
            w_dst = np.diff(dst.edges)
            tag = f"{e.src}-{e.dst}" + (f"[{b}]" if len(e.spans) > 1 else "")
            key = (tag.split("-")[0], "-".join(tag.split("-")[1:]))
            for p_name in e.ports:
                p = port(p_name)
                if p_name == "ADVEC":
                    a = _map_onto(src, dst, src.mass_flux, p.flow_mapping)
                    out.append(port_residual(a, dst.mass_flux, w_dst, key, p_name))
                elif p_name == "MECH":
                    # The MECH *flow* is the velocity vector itself, which is a
                    # property of the point and does not flip with the normal --
                    # unlike the mass flux and unlike the traction. Agreement is
                    # therefore v_src - v_dst = 0, not a sum. Comparing it the
                    # other way makes every velocity port read exactly r = 2,
                    # which is what the first run of the W4 null test did.
                    a = _map_onto(src, dst, src.velocity, p.flow_mapping)
                    out.append(residual_from_shared_normal(a, dst.velocity, w_dst,
                                                           key, "MECH:flow"))
                    # The traction *does* flip: sigma.n with opposite normals is
                    # equal and opposite, which is Newton's third law.
                    te = _map_onto(src, dst, src.traction, p.effort_mapping)
                    out.append(port_residual(te, dst.traction, w_dst, key, "MECH:effort"))
    return out


# --------------------------------------------------------------------------
# thrust residual -- pre and post projection, never one without the other
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ThrustResidual:
    """`r_T` for one rotor, with the pre-projection value mandatory.

    Guide rule 2: the conservation projection makes the residual zero by
    construction, so a post-projection number proves the projection works and
    nothing else. **The scientific result is the residual before projection.**
    This record therefore cannot be built without `pre`, and `__str__` always
    shows it first."""

    rotor: str
    thrust_disk: float
    flux_pre: float
    pre: float
    flux_post: float | None = None
    post: float | None = None

    @property
    def projected(self) -> bool:
        return self.post is not None

    def as_dict(self) -> dict:
        d = {"rotor": self.rotor, "thrust_disk": self.thrust_disk,
             "flux_pre": self.flux_pre, "r_T_pre": self.pre}
        if self.projected:
            d.update({"flux_post": self.flux_post, "r_T_post": self.post})
        return d

    def __str__(self) -> str:
        s = f"{self.rotor}: r_T(pre) = {self.pre:.4e}"
        return s + (f", r_T(post) = {self.post:.4e}" if self.projected else " [unprojected]")


def thrust_residual(rotor: str, thrust_disk: float, flux_pre: float,
                    flux_post: float | None = None) -> ThrustResidual:
    """`r_T = |T_disk - |Phi_x|| / T_disk`, spec section 11 gate W4."""
    def r(flux):
        return abs(thrust_disk - abs(flux)) / max(abs(thrust_disk), 1e-30)
    return ThrustResidual(rotor=rotor, thrust_disk=thrust_disk, flux_pre=flux_pre,
                          pre=r(flux_pre), flux_post=flux_post,
                          post=None if flux_post is None else r(flux_post))


# --------------------------------------------------------------------------
# the global power residual
# --------------------------------------------------------------------------


def _box_faces(box):
    """(x0, x1, y0, y1, outward normal) for the four faces of a box."""
    return [
        ((box.x0, box.x0), (box.y0, box.y1), (-1.0, 0.0)),
        ((box.x1, box.x1), (box.y0, box.y1), (+1.0, 0.0)),
        ((box.x0, box.x1), (box.y0, box.y0), (0.0, -1.0)),
        ((box.x0, box.x1), (box.y1, box.y1), (0.0, +1.0)),
    ]


def region_faces(agent: str):
    """Every boundary face of one agent, with the agent's outward normal.

    A hole contributes its four faces with the sign flipped: leaving the agent
    into the hole is still leaving the agent."""
    a = AGENT[agent]
    faces = [(xs, ys, n, +1.0) for xs, ys, n in _box_faces(a.box)]
    for h in a.holes:
        faces += [(xs, ys, (-n[0], -n[1]), +1.0) for xs, ys, n in _box_faces(h)]
    return faces


def _face_integral(field: Field, xs, ys, n, re_eff: float, order: int = QUAD_ORDER):
    """Outward energy flux through one axis-aligned face.

        integral of [ (|u|^2/2 + p) u.n - (1/Re) u . (grad u . n) ] ds
    """
    horiz = xs[0] == xs[1]
    lo, hi = (ys if horiz else xs)
    if hi - lo <= 0:
        return 0.0
    xg, wg = _gl(order)
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    s = mid + half * xg
    x = np.full_like(s, xs[0]) if horiz else s
    y = s if horiz else np.full_like(s, ys[0])
    u, v = field.velocity(x, y)
    p = field.pressure(x, y)
    un = u * n[0] + v * n[1]
    h = 1e-6
    up, vp = field.velocity(x + h * n[0], y + h * n[1])
    um, vm = field.velocity(x - h * n[0], y - h * n[1])
    visc = (u * (up - um) + v * (vp - vm)) / (2 * h) / re_eff
    integrand = (0.5 * (u * u + v * v) + p) * un - visc
    return float(half * np.sum(integrand * wg))


def _rect_minus(rect, holes):
    """`rect` minus a list of axis-aligned `holes`, as disjoint rectangles.

    Needed because multiplying a whole-cell quadrature by the cell's area
    fraction is only correct for an integrand that is constant across the cell.
    The disk body force is not: it lives entirely inside agent `N`'s notch, so
    that shortcut credited `N` with a share of a force that belongs to `R1`, and
    the total work came out 48% too large."""
    out = [rect]
    for h in holes:
        nxt = []
        for (x0, x1, y0, y1) in out:
            hx0, hx1 = max(x0, h.x0), min(x1, h.x1)
            hy0, hy1 = max(y0, h.y0), min(y1, h.y1)
            if hx0 >= hx1 or hy0 >= hy1:              # no overlap: keep whole
                nxt.append((x0, x1, y0, y1))
                continue
            if y0 < hy0:
                nxt.append((x0, x1, y0, hy0))         # below the hole
            if hy1 < y1:
                nxt.append((x0, x1, hy1, y1))         # above it
            if x0 < hx0:
                nxt.append((x0, hx0, hy0, hy1))       # left of it
            if hx1 < x1:
                nxt.append((hx1, x1, hy0, hy1))       # right of it
        out = nxt
    return [r for r in out if r[1] > r[0] and r[3] > r[2]]


def _integrate_rect(field: Field, rect, what: str, re_eff: float,
                    order: int = QUAD_ORDER) -> float:
    x0, x1, y0, y1 = rect
    xg, wg = _gl(order)
    X = (0.5 * (x0 + x1) + 0.5 * (x1 - x0) * xg)[:, None]
    Y = (0.5 * (y0 + y1) + 0.5 * (y1 - y0) * xg)[None, :]
    X, Y = np.broadcast_arrays(X, Y)
    u, v = field.velocity(X, Y)
    if what == "work":
        fx, fy = field.body_force(X, Y)
        val = u * fx + v * fy
    elif what == "energy":
        val = 0.5 * (u * u + v * v)
    elif what == "dissipation":
        h = 1e-5
        ux = (field.velocity(X + h, Y)[0] - field.velocity(X - h, Y)[0]) / (2 * h)
        uy = (field.velocity(X, Y + h)[0] - field.velocity(X, Y - h)[0]) / (2 * h)
        vx = (field.velocity(X + h, Y)[1] - field.velocity(X - h, Y)[1]) / (2 * h)
        vy = (field.velocity(X, Y + h)[1] - field.velocity(X, Y - h)[1]) / (2 * h)
        val = (ux * ux + uy * uy + vx * vx + vy * vy) / re_eff
    else:
        raise KeyError(what)
    return float(0.25 * (x1 - x0) * (y1 - y0) * np.einsum("i,j,ij->", wg, wg, val))


def _area_integral(field: Field, agent: str, what: str, re_eff: float,
                   order: int = QUAD_ORDER) -> float:
    """Volume integral over one agent, cell by cell, with each cell clipped to
    the agent's actual region rather than weighted by its area fraction."""
    a = AGENT[agent]
    g = GRIDS[agent]
    dx, dy = a.spacing
    total = 0.0
    for i, xc in enumerate(g.x_c):
        for j, yc in enumerate(g.y_c):
            if g.frac[i, j] <= 0.0:
                continue
            cell = (xc - dx / 2, xc + dx / 2, yc - dy / 2, yc + dy / 2)
            for r in _rect_minus(cell, a.holes):
                total += _integrate_rect(field, r, what, re_eff, order)
    return float(total)


def power_ledger(field: Field, re_eff: float = 1000.0, p_ext: float = 0.0,
                 d_energy_dt: dict | None = None, agents=None) -> PowerLedger:
    """The global power residual `R(t)` of spec section 7.1 (C3b).

    Convention, stated because the guide's written form and this one differ in
    sign on two terms: every entry is a power, and for a true solution

        dE/dt + (net outward energy flux) + (dissipation) - (work by body force) = 0,

    with `P_ext = -work` for a device that extracts. `R` subtracts `P_ext`, so it
    vanishes; `residual_without_p_ext` does not, and equals the extracted power
    exactly -- which is the form gate W4's third item is stated in.

    `d_energy_dt` is per agent and defaults to zero, which is exact for the
    steady analytic fields W4 validates against and must be supplied by the
    caller for anything else."""
    names = [a.name for a in AGENTS] if agents is None else list(agents)
    d_energy_dt = d_energy_dt or {}
    led = PowerLedger(p_ext=float(p_ext))
    for name in names:
        flux = sum(_face_integral(field, xs, ys, n, re_eff)
                   for xs, ys, n, _ in region_faces(name))
        diss = _area_integral(field, name, "dissipation", re_eff)
        work = _area_integral(field, name, "work", re_eff)
        de = float(d_energy_dt.get(name, 0.0))
        led.outflux += flux
        led.dissipation += diss
        led.work += work
        led.d_energy_dt += de
        led.per_agent[name] = {"outflux": flux, "dissipation": diss,
                               "work": work, "d_energy_dt": de}
    return led


# --------------------------------------------------------------------------
# wake diagnostics
# --------------------------------------------------------------------------


#: Stations spec section 8 / guide section 5.1 asks for lateral profiles at.
PROFILE_STATIONS = (1.0, 3.0, 5.0, 7.0, 10.0, 14.0)


def centreline(field: Field, x: np.ndarray, u_inf: float = U_INF):
    x = np.asarray(x, dtype=np.float64)
    u, _ = field.velocity(x, np.zeros_like(x))
    return np.asarray(u, dtype=np.float64)


def centreline_deficit(field: Field, x: np.ndarray, u_inf: float = U_INF):
    return 1.0 - centreline(field, x, u_inf) / u_inf


def lateral_profile(field: Field, x: float, y: np.ndarray | None = None):
    y = np.linspace(-WAKE_HALFWIDTH, WAKE_HALFWIDTH, 401) if y is None else np.asarray(y, float)
    u, _ = field.velocity(np.full_like(y, float(x)), y)
    return y, np.asarray(u, dtype=np.float64)


@dataclass(frozen=True)
class GaussianFit:
    amplitude: float
    sigma: float
    r2: float

    @property
    def good(self) -> bool:
        return self.r2 > 0.99


def fit_gaussian(y: np.ndarray, u: np.ndarray, u_inf: float = U_INF) -> GaussianFit:
    """Fit `1 - u/u_inf = A exp(-y^2 / 2 sigma^2)` by linear regression on logs.

    Only the points with a genuine deficit are used: taking the log of a deficit
    that is zero in the free stream would either blow up or, if floored, drag the
    fit toward the floor."""
    y = np.asarray(y, dtype=np.float64)
    d = 1.0 - np.asarray(u, dtype=np.float64) / u_inf
    m = d > max(1e-9, 1e-3 * float(np.max(d)) if np.max(d) > 0 else 1e-9)
    if m.sum() < 3:
        return GaussianFit(0.0, float("nan"), 0.0)
    slope, intercept = np.polyfit(y[m] ** 2, np.log(d[m]), 1)
    if slope >= 0:
        return GaussianFit(float(np.max(d)), float("inf"), 0.0)
    sigma = float(np.sqrt(-1.0 / (2.0 * slope)))
    amp = float(np.exp(intercept))
    pred = amp * np.exp(-y[m] ** 2 / (2 * sigma ** 2))
    ss_res = float(np.sum((d[m] - pred) ** 2))
    ss_tot = float(np.sum((d[m] - d[m].mean()) ** 2))
    if ss_tot <= 1e-24 * max(float(np.mean(d[m] ** 2)), 1e-300):
        # A deficit that is *constant* across the fitted region carries no shape
        # to fit. `1 - ss_res/ss_tot` is 0/0 there, and returning 1.0 for it
        # declared a Jensen top hat a perfect Gaussian at x = 14 -- which would
        # have let gate W8's "self-similar Gaussian far wake" pass on a profile
        # that is the opposite of one. No variance, no fit.
        return GaussianFit(amp, float("inf"), 0.0)
    return GaussianFit(amp, sigma, 1.0 - ss_res / ss_tot)


@dataclass(frozen=True)
class JensenFit:
    a: float
    k: float
    r2: float


def fit_jensen(field: Field, x: np.ndarray | None = None, x_turbine: float = X_T1,
               u_inf: float = U_INF, r0: float = ROTOR_HALFSPAN) -> JensenFit:
    """Recover `(a, k)` from a top-hat wake, exactly, by linearizing.

    The centreline deficit of the Park model is `d(x) = 2a / (1 + 2 k x)^2`, so

        d(x)^(-1/2) = (1 + 2 k x) / sqrt(2a)

    is *linear in x*: the intercept gives `a` and the slope gives `k`. No
    iterative fit, no initial guess, and the residual of the straight line says
    whether the field really was a Jensen wake."""
    x = np.linspace(x_turbine + 0.5, x_turbine + 14.0, 60) if x is None else np.asarray(x, float)
    d = centreline_deficit(field, x, u_inf)
    m = d > 1e-12
    if m.sum() < 3:
        return JensenFit(0.0, float("nan"), 0.0)
    s = x[m] - x_turbine
    inv = d[m] ** -0.5
    slope, intercept = np.polyfit(s, inv, 1)
    # inv = (1 + 2 k s) / sqrt(2a):  intercept = 1/sqrt(2a), slope = 2k/sqrt(2a)
    a = 1.0 / (2.0 * intercept ** 2)
    k = 0.5 * slope / intercept
    pred = slope * s + intercept
    ss_res = float(np.sum((inv - pred) ** 2))
    ss_tot = float(np.sum((inv - inv.mean()) ** 2))
    return JensenFit(float(a), float(k), 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0)


def wake_half_width(field: Field, x: float, tol: float = 1e-9,
                    u_inf: float = U_INF, y_max: float = WAKE_HALFWIDTH) -> float:
    """Outermost |y| at which the deficit still exceeds `tol`."""
    y = np.linspace(0.0, y_max, 4001)
    _, u = lateral_profile(field, x, y)
    d = 1.0 - u / u_inf
    idx = np.nonzero(d > tol)[0]
    return float(y[idx[-1]]) if len(idx) else 0.0


def in_corridor(x, u, model: str = "jensen", **kw):
    """Elementwise: does the centreline velocity lie inside the model corridor?

    Where the corridor is NaN the model does not apply -- BPA near the rotor --
    and those stations count as *not violated* rather than as failures. The
    applicability mask is returned so a caller can never quietly report "inside
    the corridor" for a range where there was no corridor."""
    lo, hi = (jensen_corridor(x, **kw) if model == "jensen" else bpa_corridor(x, **kw))
    u = np.asarray(u, dtype=np.float64)
    applicable = np.isfinite(lo) & np.isfinite(hi)
    ok = np.where(applicable, (u >= lo - 1e-12) & (u <= hi + 1e-12), True)
    return ok, lo, hi, applicable


# --------------------------------------------------------------------------
# symmetry and array efficiency
# --------------------------------------------------------------------------


def symmetry_residual(field: Field, u_inf: float = U_INF) -> float:
    """`|| u(B+) - M u(B-) || / || u ||` with `M` the mirror `y -> -y`.

    Needs no reference data: under symmetric inflow the two bypass agents must
    remain mirror images, so this detects a broken symmetry on its own -- which
    is spec section 3.1's stated reason for making them two agents rather than
    one two-component agent."""
    g = GRIDS["B+"]
    X, Y = np.meshgrid(g.x_c, g.y_c, indexing="ij")
    up, vp = field.velocity(X, Y)
    um, vm = field.velocity(X, -Y)                  # the mirrored partner in B-
    du = np.asarray(up) - np.asarray(um)
    dv = np.asarray(vp) + np.asarray(vm)            # v is odd under the mirror
    w = g.frac
    num = float(np.sqrt(np.sum((du ** 2 + dv ** 2) * w)))
    den = float(np.sqrt(np.sum((np.asarray(up) ** 2 + np.asarray(vp) ** 2) * w)))
    return num / max(den, 1e-30)


def array_efficiency(p1: float, p2: float) -> float:
    """`P_2 / P_1`. Gate W10's band is deliberately wide -- see the 2D discount."""
    return float(p2) / max(float(p1), 1e-30)


# --------------------------------------------------------------------------
# the whole report
# --------------------------------------------------------------------------


def report(field: Field, re_eff: float = 1000.0, p_ext: float = 0.0,
           thrusts: dict | None = None, d_energy_dt: dict | None = None) -> dict:
    """Every metric at once, in the shape the GUI and the gate script both read.

    `d_energy_dt` is forwarded to `power_ledger` and defaults to `None`, which
    that function treats as zero -- exact for the steady analytic fields W4
    validates against and **wrong for a snapshot of a running system**. Whether
    it was supplied is recorded in the returned dict as `unsteady_term_supplied`
    rather than left for a reader to infer, because a power residual computed
    without it is not comparable with one computed with it: measured on the
    coupled system at t = 5, omitting it inflated `|R|` to 88% of the extracted
    power, essentially all of it the missing term."""
    res = interface_residuals(field, re_eff)
    led = power_ledger(field, re_eff, p_ext, d_energy_dt=d_energy_dt)
    xs = np.linspace(0.25, 17.0, 120)
    prof = {}
    for st in PROFILE_STATIONS:
        y, u = lateral_profile(field, st)
        f = fit_gaussian(y, u)
        prof[st] = {"y": y.tolist(), "u": u.tolist(),
                    "gaussian": {"amplitude": f.amplitude, "sigma": f.sigma, "r2": f.r2},
                    "half_width": wake_half_width(field, st)}
    uc = centreline(field, xs)
    ok_j, jlo, jhi, app_j = in_corridor(xs, uc, "jensen")
    ok_b, blo, bhi, app_b = in_corridor(xs, uc, "bpa")
    # Fit over the region where only turbine 1's wake exists. Run across the
    # whole domain on a two-turbine field, the single-wake law is fitted to a
    # superposition and returns nonsense that still looks like a fit
    # (a = 0.174, k = 0.002 for a field built with a = 1/3, k = 0.075).
    jf = fit_jensen(field, x=np.linspace(X_T1 + 0.5, X_T2 - 0.5, 60))
    return {
        "ports": {"summary": summarize(res),
                  "residuals": [{"edge": f"{r.edge[0]}-{r.edge[1]}", "port": r.port,
                                 "value": r.value, "scale": r.scale,
                                 "degenerate": r.degenerate} for r in res]},
        "power": led.as_dict(),
        "symmetry": symmetry_residual(field),
        "centreline": {"x": xs.tolist(), "u": np.asarray(uc).tolist(),
                       "jensen_lo": np.asarray(jlo).tolist(),
                       "jensen_hi": np.asarray(jhi).tolist(),
                       "bpa_lo": np.asarray(blo).tolist(),
                       "bpa_hi": np.asarray(bhi).tolist(),
                       "in_jensen": bool(np.all(ok_j)), "in_bpa": bool(np.all(ok_b)),
                       "jensen_graded_fraction": float(np.mean(app_j)),
                       "bpa_graded_fraction": float(np.mean(app_b))},
        "profiles": prof,
        "jensen_fit": {"a": jf.a, "k": jf.k, "r2": jf.r2},
        "thrusts": thrusts or {},
        "unsteady_term_supplied": d_energy_dt is not None,
    }


__all__ = [
    "QUAD_ORDER", "InterfaceSample", "sample_interface", "interface_residuals",
    "ThrustResidual", "thrust_residual", "region_faces", "power_ledger",
    "PROFILE_STATIONS", "centreline", "centreline_deficit", "lateral_profile",
    "GaussianFit", "fit_gaussian", "JensenFit", "fit_jensen", "wake_half_width",
    "in_corridor", "symmetry_residual", "array_efficiency", "report",
]
