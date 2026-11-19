"""CO-T layouts built in code, for Phase 1's seam checks and the tests.

The coolant plan's development and evaluation layouts are drawn in the studio
([[chart-operator-stage-0b-and-1-plan]] section 6); these are not those.  They are the
smallest layouts that exercise what Phase 1 must show: a seam that carries the flow,
a wall between the coolant and the plate at the real contrast, and a curved channel.
Materials come from the workbench's conjugate-heat library (water, aluminium), never
retyped.  Every piece is one material (model (a), side walls only).

- ``channel``: a straight channel ``K`` pieces long, each ``L x w``, plane Poiseuille
  flow, the inlet held at ``T_IN``, the bottom wall hot (`wall_profile`: ``T_IN`` at the
  inlet's corner, ``T_WALL`` a short way on: a Graetz problem, a thermal layer growing
  along the hot wall), the top wall insulated, the outlet free.
- ``plate``: a cold plate in plan view: an aluminium strip on each side of the channel,
  ``K`` columns of three pieces, after one coolant piece of unheated entry (the inlet's
  held temperature touches only coolant).  The device beneath heats the plate pieces of
  the middle columns (the source ``Q_DEVICE``); the plate's outer edges are insulated,
  so in the steady state every watt leaves with the coolant, or diffuses back out
  through the inlet.  Contrast k_Al / k_water 342.
- ``bend``: a channel round a quarter turn, ``n_r x n_th`` pieces on log-polar charts
  (conformal, right corners), flow from the end on the y axis to the end on the x axis,
  the inner wall hot (`wall_profile` along its arc), the outer insulated.

Each is ``steady`` or one implicit ``step`` from ``T_OLD`` everywhere.  ``pe`` scales the
flow so the largest piece Peclet number is exactly ``pe`` (Peclet numbers are linear in
the flow); 0 is pure conduction on the same pieces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import chart as ch
from . import transport as tr
from .transport_host import TransportLayout

NAMES = ("channel", "plate", "bend")
T_IN, T_WALL, T_OLD = 300.0, 320.0, 300.0
#: the device's heat under the plate, per unit volume of the plan-view slab (W/m^3)
Q_DEVICE = 2.0e7
#: geometry (m): channel width, piece length, plate strip height, bend radii
W_CH, L_PIECE, H_PLATE = 1.0e-3, 2.0e-3, 1.5e-3
R_IN, R_OUT = 1.0e-3, 2.0e-3


def wall_profile(s, ell: float) -> np.ndarray:
    """A hot wall's temperature at distance ``s`` from the inlet corner: ``T_IN`` at the
    corner, which the inlet also holds (so the shared corner is one value), rising to
    ``T_WALL`` over a few ``ell``."""
    s = np.asarray(s, float)
    return T_IN + (T_WALL - T_IN) * (1.0 - np.exp(-np.maximum(s, 0.0) / ell))


def material(name: str) -> dict:
    from atlas.workbench import registry
    m = dict(registry.family("conjugate-heat-2d").material_library()[name])
    m["alpha"] = m["k"] / (m["rho"] * m["cp"])
    return m


@dataclass
class TransportBench:
    name: str
    problem: str
    problems: list
    bc: object
    meta: dict = field(default_factory=dict)

    def layout(self, kind: str = "sine-hat", m: int = 16, split: str = "skew") -> TransportLayout:
        return TransportLayout(self.problems, self.bc, kind=kind, m=m, split=split)


def _piece(c: ch.Chart, mat: dict, psi_fn, problem: str, dt: float, q: float,
           label: str) -> tr.TransportPiece:
    el = c.element_fields()
    area = el["area"]
    n = c.n
    psi = (np.zeros((n + 1, n + 1)) if psi_fn is None
           else tr.psi_nodes(c, psi_fn, mat["alpha"]))
    if problem == "steady":
        fo = None
        g = np.full((n + 1, n + 1), q * area / mat["k"])
    elif problem == "step":
        fo = mat["alpha"] * dt / area
        g = np.full((n + 1, n + 1), q * area / mat["k"] + T_OLD / fo)
    else:
        raise ValueError(f"problem {problem!r}: 'steady' or 'step'")
    return tr.make_piece(c, psi, k=mat["k"], alpha=mat["alpha"], fo=fo, g=g, label=label,
                         meta={"area": area, "material": mat.get("name", ""), "q": q})


def _rect(x0, y0, x1, y1, n) -> ch.Chart:
    return ch.Chart(ch.from_map(lambda xi, eta: (x0 + (x1 - x0) * xi, y0 + (y1 - y0) * eta), n),
                    "affine")


def _finish_pe(probs, pe: float) -> float:
    if pe <= 0.0:
        for p in probs:
            p.psi = np.zeros_like(p.psi)
        return 0.0
    return tr.scale_to_peclet(probs, pe)


def channel(n: int = 64, K: int = 4, pe: float = 50.0, problem: str = "steady",
            dt: float = 0.05, prof: str = "parabolic") -> TransportBench:
    w, L = W_CH, L_PIECE
    water = dict(material("water"), name="water")

    def psi(x, y):
        return w * tr.profile(np.clip(y / w, 0.0, 1.0), prof)
    probs = [_piece(_rect(a * L, 0.0, (a + 1) * L, w, n), water, psi, problem, dt, 0.0,
                    f"channel {a}") for a in range(K)]
    f = _finish_pe(probs, pe)
    tol = 1e-9 * K * L

    def bc(i, s, P):
        if np.all(np.abs(P[:, 0]) < tol):
            return ("dirichlet", np.full(len(P), T_IN))
        if np.all(np.abs(P[:, 1]) < tol):
            return ("dirichlet", wall_profile(P[:, 0], L / 4.0))
        return ("insulated", None)
    return TransportBench("channel", problem, probs, bc,
                          {"K": K, "pe": pe, "dt": dt, "flow_factor": f, "n": n})


def plate(n: int = 64, K: int = 4, pe: float = 50.0, problem: str = "steady",
          dt: float = 0.05, heated=None) -> TransportBench:
    w, L, h = W_CH, L_PIECE, H_PLATE
    water = dict(material("water"), name="water")
    al = dict(material("aluminium"), name="aluminium")
    heated = set(range(1, K - 1) if K > 2 else range(K)) if heated is None else set(heated)

    def psi(x, y):
        return w * tr.profile(np.clip((y - h) / w, 0.0, 1.0))
    # the entry: one coolant piece upstream of the plate, its walls insulated, so the
    # inlet's held temperature touches only coolant (a held corner of the heated plate
    # would be a point sink, a log singularity that no port count resolves)
    probs = [_piece(_rect(-L, h, 0.0, h + w, n), water, psi, problem, dt, 0.0, "entry")]
    for a in range(K):
        x0, x1 = a * L, (a + 1) * L
        q = Q_DEVICE if a in heated else 0.0
        probs.append(_piece(_rect(x0, 0.0, x1, h, n), al, None, problem, dt, q, f"plate low {a}"))
        probs.append(_piece(_rect(x0, h, x1, h + w, n), water, psi, problem, dt, 0.0,
                            f"coolant {a}"))
        probs.append(_piece(_rect(x0, h + w, x1, 2 * h + w, n), al, None, problem, dt, q,
                            f"plate high {a}"))
    f = _finish_pe([p for p in probs if p.label.startswith(("coolant", "entry"))], pe)
    tol = 1e-9 * K * L

    def bc(i, s, P):
        on_inlet = (np.all(np.abs(P[:, 0] + L) < tol)
                    and np.all((P[:, 1] > h - tol) & (P[:, 1] < h + w + tol)))
        if on_inlet:
            return ("dirichlet", np.full(len(P), T_IN))
        return ("insulated", None)
    return TransportBench("plate", problem, probs, bc,
                          {"K": K, "pe": pe, "dt": dt, "flow_factor": f, "n": n,
                           "heated_columns": sorted(heated)})


def bend(n: int = 64, n_r: int = 2, n_th: int = 3, pe: float = 50.0,
         problem: str = "steady", dt: float = 0.05) -> TransportBench:
    r0, r1 = R_IN, R_OUT
    rho0, rho1 = math.log(r0), math.log(r1)
    dr, dth = (rho1 - rho0) / n_r, (math.pi / 2) / n_th
    water = dict(material("water"), name="water")

    def psi(x, y):
        s = (np.log(np.hypot(x, y)) - rho0) / (rho1 - rho0)
        return (r1 - r0) * tr.profile(np.clip(s, 0.0, 1.0))
    probs = []
    for b in range(n_th):
        for a in range(n_r):
            def F(xi, eta, a=a, b=b):
                rr = np.exp(rho0 + (a + xi) * dr)
                th = (b + eta) * dth
                return rr * np.cos(th), rr * np.sin(th)
            c = ch.Chart(ch.from_map(F, n), "conformal-log-polar", {"modulus": dr / dth})
            probs.append(_piece(c, water, psi, problem, dt, 0.0, f"bend {a},{b}"))
    f = _finish_pe(probs, pe)
    tol = 1e-9 * r1

    def bc(i, s, P):
        if np.all(np.abs(P[:, 0]) < tol):                       # the end on the y axis
            return ("dirichlet", np.full(len(P), T_IN))
        if np.all(np.abs(np.hypot(P[:, 0], P[:, 1]) - r0) < 1e-6 * r0):
            arc = r0 * (math.pi / 2 - np.arctan2(P[:, 1], P[:, 0]))
            return ("dirichlet", wall_profile(arc, r0 * dth / 4.0))
        return ("insulated", None)
    return TransportBench("bend", problem, probs, bc,
                          {"pieces": (n_r, n_th), "pe": pe, "dt": dt, "flow_factor": f,
                           "n": n})


def build(name: str, **kw) -> TransportBench:
    if name == "channel":
        return channel(**kw)
    if name == "plate":
        return plate(**kw)
    if name == "bend":
        return bend(**kw)
    raise ValueError(f"{name!r}: one of {NAMES}")
