"""Port residuals and the global power residual.

**Not case-study code.** This is the measurement half of the port algebra of
`port-algebra-atlas-0.1`, and every later case study reuses it, so nothing about
turbines or wakes belongs here.

Two quantities, and the difference between them matters
-------------------------------------------------------
`port_residual` is *local*: for one interface it asks whether the two agents
agree about the quantity crossing it. It is a consistency check between two
neural predictions and, at eleven of this case study's fifteen ports, there is
no exact side to project onto -- so it is reported, never enforced. Saying that
plainly is the enforce-or-measure rule of `conservation-as-constraint-atlas-0.1`.

`power_residual` is *global*: because every port is an effort/flow pair whose
product is a power, the whole graph admits one scalar diagnostic. That is the
property that makes the port algebra worth having -- fifteen incommensurable
interface errors collapse into one number in watts.

The sign convention, stated once
--------------------------------
An interface normal points from the first-named agent to the second. A flow is
positive along that normal. So the *same* physical quantity is an outflow for
the source agent and an inflow for the destination, and two agents that agree
report values satisfying

    f_src(with its own outward normal) + f_dst(with its own outward normal) = 0.

`port_residual` takes both sides already expressed with their own outward
normals, so agreement means they sum to zero. Passing both with respect to the
*same* normal instead makes every residual read 2 rather than 0, which is loud
enough to notice -- but `residual_from_shared_normal` exists so the caller never
has to do that flip by hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field

import numpy as np


# --------------------------------------------------------------------------
# curve norms
# --------------------------------------------------------------------------


def curve_norm(values: np.ndarray, weights: np.ndarray) -> float:
    """Length-weighted L2 norm over an interface curve.

    Weighted so a residual does not change when a curve is resampled, which it
    would under a plain vector norm -- and interfaces here are sampled at
    whatever spacing each agent's own token lattice gives."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.ndim == 2:                       # vector-valued: sum components first
        sq = np.sum(v * v, axis=-1)
    else:
        sq = v * v
    total = float(np.sum(w))
    if total <= 0.0:
        return 0.0
    return float(np.sqrt(np.sum(sq * w) / total))


# --------------------------------------------------------------------------
# per-port residual
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PortResidual:
    """One interface's disagreement, with the pieces kept separate.

    `value` is the guide's

        r = ||f_A + f_B|| / (0.5 (||f_A|| + ||f_B||)),

    normalized by the *average magnitude of the two sides* rather than by either
    one, so it is symmetric in the agents and does not blow up when one side is
    near zero while the other is not. `scale` is that denominator: a residual of
    1e-3 on a scale of 1e-12 is noise, not agreement, and the pair is reported
    together so that cannot be misread."""

    edge: tuple[str, str]
    port: str
    value: float
    scale: float
    mismatch: float
    signed_mean: float

    @property
    def degenerate(self) -> bool:
        """True when both sides are essentially zero, making `value` meaningless."""
        return self.scale < 1e-30

    def __str__(self) -> str:
        a, b = self.edge
        return (f"{a}-{b} [{self.port}] r={self.value:.3e} "
                f"(scale {self.scale:.3e}{', DEGENERATE' if self.degenerate else ''})")


def port_residual(f_src: np.ndarray, f_dst: np.ndarray, weights: np.ndarray,
                  edge: tuple[str, str] = ("?", "?"), port: str = "?") -> PortResidual:
    """Residual between two sides of one interface.

    `f_src` and `f_dst` are each expressed with respect to their *own* outward
    normal, so consistency is `f_src + f_dst = 0`."""
    a = np.asarray(f_src, dtype=np.float64)
    b = np.asarray(f_dst, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"edge {edge}: sides have shapes {a.shape} and {b.shape}; "
                         "map them onto a common discretization first")
    w = np.asarray(weights, dtype=np.float64)
    mismatch = curve_norm(a + b, w)
    scale = 0.5 * (curve_norm(a, w) + curve_norm(b, w))
    value = mismatch / scale if scale > 1e-30 else 0.0
    # which way the disagreement leans: a systematic sign localizes a leak,
    # a sign-free residual only says that one exists
    diff = a + b
    signed = float(np.sum((diff if diff.ndim == 1 else diff[..., 0]) * w) / max(np.sum(w), 1e-300))
    return PortResidual(edge=edge, port=port, value=float(value), scale=float(scale),
                        mismatch=float(mismatch), signed_mean=signed)


def residual_from_shared_normal(f_a: np.ndarray, f_b: np.ndarray, weights: np.ndarray,
                                edge: tuple[str, str] = ("?", "?"),
                                port: str = "?") -> PortResidual:
    """As `port_residual`, for sides given with respect to the *same* normal.

    Agreement then means `f_a - f_b = 0`, so the second side is flipped here
    rather than at the call site."""
    return port_residual(f_a, -np.asarray(f_b, dtype=np.float64), weights, edge, port)


# --------------------------------------------------------------------------
# the global power residual
# --------------------------------------------------------------------------


@dataclass
class PowerLedger:
    """The terms of the global power balance, kept apart until the last moment.

    Convention: every term is a power, and the balance that must hold for a true
    solution is

        dE/dt + (net outward energy flux) + (dissipation) - (work by body force) = 0.

    `P_ext` is the power leaving through *unconnected* ports -- the two turbine
    shafts here. It is `-work` for a device that extracts, so a budget that omits
    it fails by exactly the extracted power. `residual_without_p_ext` is
    therefore not a diagnostic curiosity: gate W4 asks for it explicitly,
    because a residual that is small only because `P_ext` was tuned to make it
    small would prove nothing."""

    d_energy_dt: float = 0.0
    outflux: float = 0.0
    dissipation: float = 0.0
    work: float = 0.0
    p_ext: float = 0.0
    per_agent: dict = _field(default_factory=dict)
    per_port: dict = _field(default_factory=dict)

    @property
    def residual_without_p_ext(self) -> float:
        return self.d_energy_dt + self.outflux + self.dissipation - self.work

    @property
    def residual(self) -> float:
        """The guide's `R(t)`. Zero for a consistent solution."""
        return self.residual_without_p_ext - self.p_ext

    @property
    def relative(self) -> float:
        """`|R|` as a fraction of the extracted power -- gate W11's form.

        Falls back to the largest single term when nothing is being extracted,
        so an unforced case does not divide by zero and silently report a
        perfect score."""
        denom = abs(self.p_ext)
        if denom < 1e-30:
            denom = max(abs(self.d_energy_dt), abs(self.outflux),
                        abs(self.dissipation), abs(self.work), 1e-30)
        return abs(self.residual) / denom

    def as_dict(self) -> dict:
        return {
            "d_energy_dt": self.d_energy_dt, "outflux": self.outflux,
            "dissipation": self.dissipation, "work": self.work, "p_ext": self.p_ext,
            "residual": self.residual,
            "residual_without_p_ext": self.residual_without_p_ext,
            "relative": self.relative,
            "per_agent": dict(self.per_agent), "per_port": dict(self.per_port),
        }

    def __str__(self) -> str:
        return (f"R = {self.residual:+.4e}  (dE/dt {self.d_energy_dt:+.3e}, "
                f"flux {self.outflux:+.3e}, diss {self.dissipation:+.3e}, "
                f"work {self.work:+.3e}, P_ext {self.p_ext:+.3e}) "
                f"-> {self.relative*100:.3f}% of extracted")


def summarize(residuals) -> dict:
    """Worst, median and total over a collection of `PortResidual`.

    Degenerate ports are counted and excluded rather than averaged in as zeros,
    which would make a graph of mostly-empty interfaces look excellent."""
    live = [r for r in residuals if not r.degenerate]
    vals = np.array([r.value for r in live], dtype=np.float64)
    worst = max(live, key=lambda r: r.value) if live else None
    return {
        "n_ports": len(list(residuals)),
        "n_live": len(live),
        "n_degenerate": len(list(residuals)) - len(live),
        "max": float(vals.max()) if len(vals) else 0.0,
        "median": float(np.median(vals)) if len(vals) else 0.0,
        "mean": float(vals.mean()) if len(vals) else 0.0,
        "worst_edge": f"{worst.edge[0]}-{worst.edge[1]}" if worst else None,
        "worst_port": worst.port if worst else None,
    }


__all__ = ["curve_norm", "PortResidual", "port_residual", "residual_from_shared_normal",
           "PowerLedger", "summarize"]
