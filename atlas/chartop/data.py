"""The Stage 0 data generator: random charted pieces with random materials and steps.

[[chart-operator-training-and-cost]] section 1.1, as built.  Each example is drawn
from its own registered seed, so a set is reproducible from its seed range alone and
nothing but code has to travel to a rented box.

**Pieces.**  Two families.
- ``spline-quad``: a random quadrilateral (aspect ratio log-uniform in [1/2, 2],
  corners perturbed, one corner made very obtuse or acute in a third of the draws so
  that the 150-160 degree corners of the s-channel's end pieces are covered), each
  side a line, a circular arc or a centripetal Catmull-Rom spline through random
  control points.  Charted by Winslow with arc-length data (two thirds) or by the
  Coons patch (one third, Winslow if the patch fails Proposition 3).
- ``conformal-map``: ``F = G(M xi + i eta)`` with ``G`` a random cubic polynomial or a
  random exponential (an annular sector, the bend's shape class), so every corner is a
  right angle and ``mu`` is the constant ``(M - 1) / (M + 1)``.  Comparison C1's
  near-conformal arm trains on this family alone with ``M`` in [0.8, 1.25].
Every chart is certified, and drawn again if it fails Proposition 3 or falls outside
the training envelope (`ENVELOPE`); the rejections are counted, and coverage is
measured by binning, not assumed.

**Materials.**  ``log kappa_hat`` is a Gaussian random field (random Fourier features
of a Gaussian kernel, correlation length ``ell``) of amplitude up to 0.7, plus, in 40%
of draws, a material jump of plus or minus 0.5-2.0 across a smooth random interface
(a level set of a second, smoother field), resolved sharply at element centres; then
centred and clipped to ``|log kappa_hat| <= 2.3``.  The distribution is symmetric
under ``log kappa_hat -> -log kappa_hat``, so the 2-D dual (CO-D on ``1 / kappa``) is
in distribution (v2 addition (b)).

**Sources and steps.**  ``g = c0 + sigma_g x field`` with the same correlation length
(the constant part is what a step from a uniform state gives).  ``Fo`` is steady in
30% of draws, otherwise log-uniform in [1e-2, 1e2].  The **input frequency** of gate
E1 is the correlation length's class: low (``ell = 0.5``), mid (0.25), high (0.125).

**Resolutions.**  ``n`` = 32, 64, 128 with probabilities 0.3, 0.5, 0.2 (decision 10:
trained on a mix around the reference 64).  Spline-quad charts are computed at
``n = 64`` and resampled (every other node for 32, bicubic for 128).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import certificate as cert
from . import chart as ch
from . import fe
from .piece import PieceProblem, fo_scalars
from .gates import SEED_BASE

#: the training envelope: what the generator admits, and what the card declares
ENVELOPE = cert.Envelope(k_mu=0.65, l_J=math.log(8.0), s_grad=40.0, l_kappa=2.3,
                         fo_min=1e-2, fo_max=1e2, steady=True)
RESOLUTIONS = ((32, 0.3), (64, 0.5), (128, 0.2))
FREQUENCIES = {"low": 0.5, "mid": 0.25, "high": 0.125}
P_STEADY = 0.3
FO_RANGE = (1e-2, 1e2)

#: registered seed bases (`gates.SEED_BASE`); example i of a split has seed base + i
SEEDS = SEED_BASE


# ---------------------------------------------------------------------------
# random fields
# ---------------------------------------------------------------------------


class Field:
    """A Gaussian random field by random Fourier features (unit variance, Gaussian
    kernel of length ``ell``), evaluable anywhere in the reference square."""

    def __init__(self, rng, ell: float, K: int = 128):
        self.w = rng.normal(size=(K, 2)) / ell
        self.p = rng.uniform(0.0, 2.0 * math.pi, K)
        self.K = K

    def __call__(self, xi, eta):
        arg = (self.w[:, 0, None, None] * xi[None] + self.w[:, 1, None, None] * eta[None]
               + self.p[:, None, None])
        return math.sqrt(2.0 / self.K) * np.cos(arg).sum(axis=0)


def _grids(n):
    t = np.arange(n + 1) / n
    XI, ET = np.meshgrid(t, t)
    c = (np.arange(n) + 0.5) / n
    XIc, ETc = np.meshgrid(c, c)
    return XI, ET, XIc, ETc


# ---------------------------------------------------------------------------
# random pieces
# ---------------------------------------------------------------------------


def _catmull_rom(points: np.ndarray, dense: int) -> np.ndarray:
    """A centripetal Catmull-Rom curve through points (ends extended by reflection)."""
    P = np.vstack([2 * points[0] - points[1], points, 2 * points[-1] - points[-2]])
    out = []
    segs = len(points) - 1
    per = max(8, dense // segs)
    for k in range(segs):
        p0, p1, p2, p3 = P[k], P[k + 1], P[k + 2], P[k + 3]

        def knot(ti, pa, pb):
            return ti + max(float(np.hypot(*(pb - pa))) ** 0.5, 1e-9)
        t0 = 0.0
        t1 = knot(t0, p0, p1)
        t2 = knot(t1, p1, p2)
        t3 = knot(t2, p2, p3)
        tt = (t1 + (t2 - t1) * np.arange(per) / per)[:, None]
        a1 = (t1 - tt) / (t1 - t0) * p0 + (tt - t0) / (t1 - t0) * p1
        a2 = (t2 - tt) / (t2 - t1) * p1 + (tt - t1) / (t2 - t1) * p2
        a3 = (t3 - tt) / (t3 - t2) * p2 + (tt - t2) / (t3 - t2) * p3
        b1 = (t2 - tt) / (t2 - t0) * a1 + (tt - t0) / (t2 - t0) * a2
        b2 = (t3 - tt) / (t3 - t1) * a2 + (tt - t1) / (t3 - t1) * a3
        out.append((t2 - tt) / (t2 - t1) * b1 + (tt - t1) / (t2 - t1) * b2)
    out.append(points[-1][None, :])
    return np.vstack(out)


def _side_curve(rng, a: np.ndarray, b: np.ndarray, dense: int = 2049) -> tuple[np.ndarray, str]:
    chord = b - a
    c = float(np.hypot(*chord))
    nrm = np.array([-chord[1], chord[0]]) / c
    u = rng.uniform()
    if u < 0.3:
        t = np.linspace(0, 1, dense)[:, None]
        return a + t * chord, "line"
    if u < 0.65:
        bulge = rng.uniform(-0.3, 0.3)
        if abs(bulge) < 1e-3:
            bulge = 1e-3
        th = 4.0 * math.atan(bulge)                       # included angle
        R = c / (2.0 * math.sin(th / 2.0))
        mid = 0.5 * (a + b)
        h = R * math.cos(th / 2.0)
        centre = mid + nrm * h                            # bows to the right of travel for bulge > 0
        a0 = math.atan2(*(a - centre)[::-1])
        a1 = math.atan2(*(b - centre)[::-1])
        d = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        if abs(d) < 1e-12:
            d = th
        ang = a0 + np.linspace(0.0, 1.0, dense) * d
        P = centre + abs(R) * np.stack([np.cos(ang), np.sin(ang)], -1)
        P[0], P[-1] = a, b
        return P, "arc"
    k = int(rng.integers(1, 4))
    fr = np.sort(rng.uniform(0.15, 0.85, k))
    cps = a + fr[:, None] * chord + (rng.normal(0.0, 0.07, k) * c)[:, None] * nrm
    P = _catmull_rom(np.vstack([a, cps, b]), dense)
    return P, "spline"


def _angle_at(P_in: np.ndarray, P_out: np.ndarray) -> float:
    """Interior angle (degrees) between the arriving curve's last segment and the
    leaving curve's first segment, counter-clockwise piece."""
    v_out = P_out[min(8, len(P_out) - 1)] - P_out[0]
    v_in = P_in[-1 - min(8, len(P_in) - 1)] - P_in[-1]
    ang = math.atan2(v_out[0] * v_in[1] - v_out[1] * v_in[0], v_out @ v_in)
    return math.degrees(ang) % 360.0


def random_quad(rng) -> tuple[list, dict]:
    """Four sides of a random spline quadrilateral, counter-clockwise, arc-length
    parametrised; rejection-sampled to corner angles in [25, 170] degrees and a
    simple boundary."""
    for _attempt in range(200):
        M = math.exp(rng.uniform(math.log(0.5), math.log(2.0)))
        base = np.array([[0, 0], [M, 0], [M, 1], [0, 1]], float) / math.sqrt(M)
        cr = base + rng.uniform(-0.22, 0.22, (4, 2))
        if rng.uniform() < 0.35:
            k = int(rng.integers(4))
            cen = cr.mean(axis=0)
            f = rng.uniform(-0.45, 0.30)
            cr[k] = cen + (1.0 + f) * (cr[k] - cen)
        curves = []
        kinds = []
        for k in range(4):
            P, kind = _side_curve(rng, cr[k], cr[(k + 1) % 4])
            curves.append(P)
            kinds.append(kind)
        ring = np.vstack([c[:-1] for c in curves])
        if cert.signed_area(ring) <= 0:
            continue
        if not cert.polygon_simple(ring[::16]):
            continue
        angles = [_angle_at(curves[(k - 1) % 4], curves[k]) for k in range(4)]
        if min(angles) < 25.0 or max(angles) > 170.0:
            continue
        sides = [ch.side_from_polyline(c) for c in curves]
        return sides, {"aspect": M, "kinds": kinds, "angles": angles}
    raise RuntimeError("no admissible random quadrilateral in 200 draws")


def random_conformal(rng, M_range=(0.5, 2.0)):
    """F(xi, eta) = G(M xi + i eta): G a random cubic or a random exponential map."""
    M = math.exp(rng.uniform(math.log(M_range[0]), math.log(M_range[1])))
    if rng.uniform() < 0.5:
        a2 = complex(*rng.normal(0.0, 0.10, 2))
        a3 = complex(*rng.normal(0.0, 0.03, 2))
        s = 1.0 / max(M, 1.0)

        def G(w):
            w = w * s
            return w + a2 * w * w + a3 * w * w * w
        kind = "cubic"
    else:
        c_ = complex(rng.uniform(-0.6, 0.6), rng.uniform(-1.4, 1.4))
        s = 1.0 / max(M, 1.0)

        def G(w):
            return np.expm1(c_ * w * s) / c_
        kind = "exponential"

    def F(xi, eta):
        z = G(M * xi + 1j * eta)
        return z.real, z.imag
    return F, {"modulus": M, "map": kind}


# ---------------------------------------------------------------------------
# one example
# ---------------------------------------------------------------------------


@dataclass
class Example:
    seed: int
    n: int
    family: str
    method: str
    freq: str
    fo: float | None
    problem: PieceProblem
    envelope: dict
    meta: dict = field(default_factory=dict)


def _chart(rng, family: str, n: int, M_range=(0.5, 2.0)) -> tuple[ch.Chart, dict]:
    if family == "conformal-map":
        F, meta = random_conformal(rng, M_range)
        return ch.Chart(ch.from_map(F, n), "conformal-map", meta), meta
    sides, meta = random_quad(rng)
    use_tfi = rng.uniform() < 1.0 / 3.0
    X64 = None
    method = "transfinite"
    if use_tfi:
        X64 = ch.transfinite(sides, 64)
        if not cert.certify(X64).ok:
            X64 = None
    if X64 is None:
        X64, info = ch.winslow(sides, 64, tol=1e-6, maxit=300)     # any certified chart will do
        method = "winslow-arc-length"
        meta = dict(meta, winslow_iterations=info["iterations"])
    return ch.Chart(ch.resample(X64, n), method, meta), meta


def draw(seed: int, family: str | None = None, n: int | None = None,
         M_range=(0.5, 2.0), envelope: cert.Envelope = ENVELOPE) -> Example:
    """The example of one seed (deterministic); ``family``/``n`` may be forced."""
    rng = np.random.default_rng(seed)
    if n is None:
        n = int(rng.choice([r for r, _ in RESOLUTIONS], p=[p for _, p in RESOLUTIONS]))
    if family is None:
        family = "conformal-map" if rng.uniform() < 0.2 else "spline-quad"
    rejected = 0
    for _attempt in range(100):
        try:
            c, meta = _chart(rng, family, n, M_range)
        except ch.WinslowDiverged:
            rejected += 1
            continue
        if not cert.certify(c.X).ok:
            rejected += 1
            continue
        m = cert.measure(c)
        if (m["sup_abs_mu"] > envelope.k_mu or m["logJ_range"] > envelope.l_J
                or m["s_grad"] > envelope.s_grad):
            rejected += 1
            continue
        break
    else:
        raise RuntimeError(f"seed {seed}: no admissible chart in 100 draws")
    freq = str(rng.choice(list(FREQUENCIES)))
    ell = FREQUENCIES[freq]
    XI, ET, XIc, ETc = _grids(n)
    lk_n = np.zeros((n + 1, n + 1))
    lk_e = np.zeros((n, n))
    if rng.uniform() < 0.7:
        amp = rng.uniform(0.0, 0.7)
        f1 = Field(rng, ell)
        lk_n += amp * f1(XI, ET)
        lk_e += amp * f1(XIc, ETc)
    if rng.uniform() < 0.4:
        jump = rng.uniform(0.5, 2.0) * (1 if rng.uniform() < 0.5 else -1)
        f2 = Field(rng, 0.4)
        vals = f2(XI, ET)
        thr = float(np.quantile(vals, rng.uniform(0.15, 0.85)))
        lk_n += jump * (vals > thr)
        lk_e += jump * (f2(XIc, ETc) > thr)
    mean = float(lk_n.mean())
    lk_n = np.clip(lk_n - mean, -envelope.l_kappa, envelope.l_kappa)
    lk_e = np.clip(lk_e - mean, -envelope.l_kappa, envelope.l_kappa)
    if rng.uniform() < P_STEADY:
        fo = None
    else:
        fo = float(10.0 ** rng.uniform(math.log10(FO_RANGE[0]), math.log10(FO_RANGE[1])))
    c0 = rng.uniform(-1.0, 1.0)
    sg = rng.uniform(0.0, 1.5)
    fg = Field(rng, ell)
    g = c0 + sg * fg(XI, ET)
    p = PieceProblem(c, log_kappa_hat=lk_n, log_kappa_hat_e=lk_e, fo=fo, g=g)
    nd = p.nodal_fields()
    el = p.element_fields()
    f_hat = (nd["J"] / el["area"]) * g
    rms = float(np.sqrt(np.mean(f_hat ** 2)))
    p.g = g / rms                                # the stored problem is normalised
    m = cert.measure(c, lk_n, fo)
    return Example(seed, n, family, c.method, freq, fo, p, m,
                   dict(meta, rejected=rejected, modulus=meta.get("modulus", meta.get("aspect"))))


def arrays(ex: Example) -> dict:
    """What a training example stores: the network's inputs and the energy's
    coefficients (float32), with its bins."""
    p = ex.problem
    chans, scal, _scale = p.inputs()
    t = p.tensors()
    return {"inputs": chans.astype(np.float32), "scalars": scal.astype(np.float32),
            "a11": t["a11"].astype(np.float32), "a12": t["a12"].astype(np.float32),
            "a22": t["a22"].astype(np.float32), "sigma": t["sigma"].astype(np.float32),
            "w": t["w"].astype(np.float32), "g": p.g.astype(np.float32),
            "s0": np.float32(p.s0)}


def labels(ex: Example, Q: np.ndarray) -> dict:
    """The exact constraint modes and particular field (validation and test only), of
    the problem **as stored**: the float32 coefficients of `arrays`, read back in
    float64, so that the labels are exact for the energy the loss evaluates (T26 then
    holds to float64 rounding, not to float32's)."""
    return labels_from_arrays(arrays(ex), Q)


def labels_from_arrays(a: dict, Q: np.ndarray) -> dict:
    n = a["a11"].shape[-1]
    d = {k: np.asarray(a[k], np.float64) for k in ("a11", "a12", "a22", "sigma", "w", "g")}
    M = fe.assemble(n, d["a11"], d["a12"], d["a22"], d["sigma"])
    exs = fe.Exact(M, n)
    H = exs.modes(Q)
    b = fe.mass(n, d["w"]) @ d["g"].ravel()
    up = exs.particular(b[fe.grid(n)[1]])
    Lam = exs.port_matrix(Q, H)
    return {"H": H, "up": up, "trace_Lambda": float(np.trace(Lam)),
            "up_energy": float(up @ (exs.A_I @ up))}


def mu_bin(sup_mu: float) -> int:
    """Gate E1's sup|mu| bins: [0, 0.15), [0.15, 0.3), [0.3, 0.45), [0.45, k_mu]."""
    return min(3, int(sup_mu / 0.15))


def timed_draws(seeds, **kw) -> tuple[list, float]:
    t0 = time.perf_counter()
    out = [draw(s, **kw) for s in seeds]
    return out, time.perf_counter() - t0
