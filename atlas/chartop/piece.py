"""One charted piece: its pulled-back problem, its network inputs, its exact solve.

**The pulled-back step.**  Backward Euler for ``rho c dT/dt = div(kappa grad T) + q`` on
a piece of area ``|Omega|``, pulled back through its chart and divided by a reference
conductivity ``kappa_ref`` (the geometric mean of ``kappa`` on the piece):

    int grad T . A_hat grad v  +  int (J~ / Fo) T v  =  int J~ g v,
    A_hat = kappa_hat J G^-1,   J~ = J / |Omega|,   Fo = kappa_ref dt / (rho c |Omega|),
    g = q |Omega| / kappa_ref + T_old / Fo,

all on the unit square.  Steady: ``1/Fo = 0``.  The physical energy is ``kappa_ref`` times
this one, which the host applies when it couples pieces of different materials; the
network never sees the scale.  The source ``g`` is nodal, its load is
``fe.mass(n, J~) g``, consistent with the reaction's element weights.

**The network's inputs** (decision 1, version 1's input list) at the ``(n+1)^2`` nodes:
``log J - mean(log J)``, ``Re mu``, ``Im mu``, ``log kappa_hat`` and the source ``f_hat = J~ g``
divided by its root mean square (the particular field is linear in it, so the host
multiplies the network's particular field by that scale); and the scalar ``Fo``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import chart as ch
from . import fe


def element_from_nodal(f: np.ndarray) -> np.ndarray:
    """Element values as the mean of each element's four nodal values."""
    return 0.25 * (f[:-1, :-1] + f[:-1, 1:] + f[1:, 1:] + f[1:, :-1])


@dataclass
class PieceProblem:
    """A charted piece with its material and its step.

    ``log_kappa_hat``: nodal (n+1, n+1), the log conductivity minus its geometric
    mean (zero for one uniform material); ``log_kappa_hat_e``: its element values,
    taken from the nodal ones unless given (a material jump is resolved at element
    centres by the generator).  ``fo``: the Fourier number, ``None`` when steady.
    ``g``: the nodal source of the step (zero by default).  ``kappa_ref``: W/(m K)."""
    chart: ch.Chart
    log_kappa_hat: np.ndarray | None = None
    log_kappa_hat_e: np.ndarray | None = None
    fo: float | None = None
    g: np.ndarray | None = None
    kappa_ref: float = 1.0
    label: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        n = self.chart.n
        if self.log_kappa_hat is None:
            self.log_kappa_hat = np.zeros((n + 1, n + 1))
        if self.log_kappa_hat_e is None:
            self.log_kappa_hat_e = element_from_nodal(self.log_kappa_hat)
        if self.g is None:
            self.g = np.zeros((n + 1, n + 1))
        self._el = None
        self._nd = None

    @property
    def n(self) -> int:
        return self.chart.n

    @property
    def steady(self) -> bool:
        return self.fo is None or math.isinf(self.fo)

    @property
    def s0(self) -> float:
        """The plain square's reaction 1/Fo (the principal part's)."""
        return 0.0 if self.steady else 1.0 / self.fo

    def element_fields(self) -> dict:
        if self._el is None:
            self._el = self.chart.element_fields()
        return self._el

    def nodal_fields(self) -> dict:
        if self._nd is None:
            self._nd = self.chart.nodal_fields()
        return self._nd

    def tensors(self) -> dict:
        """Element arrays a11, a12, a22 (kappa_hat J G^-1), sigma (J~/Fo) and the
        load weight J~."""
        el = self.element_fields()
        k = np.exp(self.log_kappa_hat_e)
        sig = np.zeros_like(el["Jt"]) if self.steady else el["Jt"] / self.fo
        return {"a11": k * el["a11"], "a12": k * el["a12"], "a22": k * el["a22"],
                "sigma": sig, "w": el["Jt"]}

    def matrix(self):
        t = self.tensors()
        return fe.assemble(self.n, t["a11"], t["a12"], t["a22"], t["sigma"])

    def load(self) -> np.ndarray:
        """The full nodal load vector b = mass(J~) g (dimensionless problem)."""
        return fe.mass(self.n, self.element_fields()["Jt"]) @ self.g.ravel()

    def inputs(self) -> tuple[np.ndarray, np.ndarray, float]:
        """(channels (5, n+1, n+1) float32, scalars (2,), source scale).

        Channels: log J - mean, Re mu, Im mu, log kappa_hat, f_hat / rms(f_hat)."""
        nd = self.nodal_fields()
        el = self.element_fields()
        logJ = np.log(nd["J"])
        f_hat = (nd["J"] / el["area"]) * self.g
        rms = float(np.sqrt(np.mean(f_hat ** 2)))
        scale = rms if rms > 0 else 1.0
        chans = np.stack([logJ - logJ.mean(), nd["mu"].real, nd["mu"].imag,
                          self.log_kappa_hat, f_hat / scale]).astype(np.float32)
        return chans, fo_scalars(self.fo), (rms if rms > 0 else 0.0)


def fo_scalars(fo: float | None) -> np.ndarray:
    """The FiLM scalars of a Fourier number: r = sqrt(s0) / (1 + sqrt(s0)) with
    s0 = 1/Fo (0 when steady, towards 1 for small steps), and log10(s0) / 3 when
    transient (0 when steady)."""
    if fo is None or math.isinf(fo):
        return np.array([0.0, 0.0], np.float32)
    s0 = 1.0 / fo
    r = math.sqrt(s0) / (1.0 + math.sqrt(s0))
    return np.array([r, math.log10(s0) / 3.0], np.float32)


@dataclass
class ExactResponse:
    """The exact constraint modes, particular field and port matrix of one piece."""
    H: np.ndarray
    up: np.ndarray
    Lam: np.ndarray
    solver: fe.Exact


def exact_response(p: PieceProblem, Q: np.ndarray) -> ExactResponse:
    M = p.matrix()
    ex = fe.Exact(M, p.n)
    H = ex.modes(Q)
    b = p.load()
    up = ex.particular(b[fe.grid(p.n)[1]])
    return ExactResponse(H, up, ex.port_matrix(Q, H), ex)
