"""Output forms after Stage 0: the kernel-exact form (W356) and the split form (W357).

Stage 0's bubble form ``H = H0 + beta N`` keeps every trace exact whatever the weights,
and nothing else.  Two defects followed ([[chart-operator-stage-0]] section 9.7):
arm B's correction broke the sum of the corner-hat modes, which the coupled answer
multiplies by the temperature level (24% fell to 4.5% on bend-3 when every temperature
was lowered by 300 K), and the modes depend on the source channel, though the exact
modes do not.  This module builds the two fixes; `model` keeps Stage 0's forms
unchanged, so its weights still load and evaluate as they did.

**The identity.**  For any piece -- any chart, material and step -- the energy is
``M = K + mass(sigma)`` with ``K 1 = 0`` and ``sigma = J~ s0`` (``s0 = 1/Fo``, zero when
steady).  Write ``c1`` for the port coefficients of the constant, ``Q c1 = 1`` (the four
hats of the sine-hat basis, the four ``j = 0`` cosines of the cosine basis), and
``u_p(1) = A_I^-1 (mass(J~) 1)_I`` for the particular field of the unit uniform source.
The interior rows of ``M 1 = s0 mass(J~) 1`` give

    H c1 + s0 u_p(1) = 1          on the interior nodes,

the discrete statement that a uniform temperature is a fixed point of the step: data
``T0`` on the boundary and ``T_old = T0`` (source ``g = s0 T0``) give ``T0`` back.  Steady,
it is ``H c1 = 1``, the hats summing to one, which B's correction broke.  Transient,
the hats alone do not sum to one (``1 - H c1`` reaches 0.98 at ``Fo = 0.01``), so the
backward-Euler analogue of the steady constraint is this fixed point, and it involves
the particular field.  The plain square's ``H0`` and ``u_1^0 = (A_I^0)^-1 (h^2 1)``
satisfy it too.

**The kernel-exact form, ``"kernel"`` (W356).**  Beside the ``4m`` mode corrections ``N_k``
and the particular correction ``N_p``, the network returns a **level-field** correction
``N_1``:

    H_raw = H0 + beta N,      u_1 = u_1^0 + a(s0) beta N_1,      a(s0) = max |u_1^0|,

so ``u_1`` is the form's answer for ``u_p(1)``, scaled by the plain square's own level
(0.074 steady, 1/s0 for short steps).  Then, in float64,

    r = H_raw c1 + s0 u_1 - 1,        H_k = H_raw,k - W_k r,

with ``W_k = c1_k H0_k^steady / sum_j c1_j^2 H0_j^steady``: the plain square's steady
modes of the kernel columns, a partition of unity on the interior nodes (for sine-hat,
the bilinear corner functions).  Since ``sum_k c1_k W_k = 1``, ``H c1 + s0 u_1 = 1`` to
rounding whatever the weights, as the bubble keeps the traces exact.  ``W`` is zero off
the kernel columns, so the side modes are exactly ``H_raw``'s, and the defect lands on
the hats in proportion to each one's share of the constant, near its own corner.  The
projection is affine and idempotent, and the exact modes are its fixed point, so the
objective's minimiser is unchanged (T26).

The source is split into its level and the rest, ``g = gbar + g'`` with ``gbar`` the
``J~``-weighted (load-weighted) mean, and

    u_p = gbar u_1 + rms(g') beta N_p,

the network seeing ``g'`` alone, as ``J g' / rms(J g')`` (version 1's ``f_hat`` with the
level removed).  So lowering every temperature by ``T0`` -- boundary data and ``T_old``
together, ``g -> g - s0 T0`` -- leaves the network's input unchanged, subtracts ``T0 c1``
from the coefficients and ``s0 T0 u_1`` from ``u_p``, and the coupled answer moves by
exactly ``T0``: the datum drops out, steady or not.

**Why a level field, rather than ``u_p(1) = (1 - H c1) / s0``.**  Deriving the uniform
source's response from the modes divides the modes' error by ``s0``: at ``Fo = 100`` a
uniform heating would inherit the hat error amplified a hundredfold.  Deriving the hat
sum from ``u_1`` multiplies ``u_1``'s error by ``s0``, and ``u_1`` is itself of size
``Fo`` when ``s0`` is large, so the product is a relative error in both limits.

**The split form, ``"split"`` (W357).**  A mode network sees ``(log J, Re mu, Im mu,
log kappa_hat)`` and the Fourier number, and returns the ``4m`` mode corrections and the
level correction: every field that does not depend on the load.  A particular-field
network sees the same four channels and the level-free source, and returns ``N_p``.
The modes and the level field are then the same for every load, bit for bit, which
the exact modes are by T25's structure; and a march that keeps the first call's modes
(the rollout, S11) keeps a level field consistent with them.

**The measure.**  ``e_H`` weights each port mode by its energy, and a uniform temperature
has none on a steady piece, so the level direction carries no weight at all.  The
coupled answer weights it by the level: at temperature ``theta`` its network term
(Theorem 2) carries ``theta r_K``, with ``r_K = u_p(s0 1) + H c1 - 1`` the **uniform state's
residual** (``H c1 - 1`` when steady).  So, per piece,

    e_K = ||r_K||_{A_I} / sqrt(tr Lambda),          e_K,inf = max |r_K|,

``e_K`` on ``e_H``'s scale and ``e_K,inf`` the pointwise error of a unit uniform state.  For
traces ``c = theta c1 + xi`` (``xi`` unit and uncorrelated) with source ``theta s0``, the
mean square network term is ``e_H^2 tr Lambda + theta^2 ||r_K||^2_{A_I}``; over the energy
the variation ``xi`` carries, ``tr Lambda``,

    e_H(theta) = (e_H^2 + theta^2 e_K^2)^(1/2),

the level adding ``theta e_K`` in quadrature.  (A step's uniform state has energy of its
own, ``theta^2 s0`` since ``J~`` has unit mean; it is left out of the denominator, so that a
large level does not make an error look small.)  The kernel-exact forms make ``e_K``
zero to rounding, so ``e_H(theta) = e_H`` at every level.
"""

from __future__ import annotations

import functools
import math
from dataclasses import replace

import numpy as np
import torch
import torch.nn as nn

from . import model as md
from . import ports as pt

FORMS = ("kernel", "split")
#: a level-free source whose RMS is below this fraction of the source's own size is
#: rounding (a uniform source minus its float64 mean), and is set to zero
LEVEL_FREE_FLOOR = 1e-12


# ---------------------------------------------------------------------------
# the constant in the port basis, and the partition of unity
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def kernel_coefficients(kind: str, n: int, m: int) -> np.ndarray:
    """``c1`` with ``Q c1 = 1`` on the loop: the hats (sine-hat) or the four ``j = 0``
    cosines (cosine), read from the basis's own column labels and checked."""
    pb = pt.basis(kind, n, m)
    if kind == "sine-hat":
        c = np.array([1.0 if col.what == "vertex" else 0.0 for col in pb.columns])
    else:
        c = np.array([1.0 if (col.what == "side" and col.j == 0) else 0.0 for col in pb.columns])
    err = float(np.abs(pb.Q @ c - 1.0).max())
    if err > 1e-12:
        raise ValueError(f"the constant is not c1 in the {kind} basis (residual {err:.2e})")
    c.setflags(write=False)
    return c


class KernelSquare:
    """Cached per (n, kind, m, device): ``c1``, the partition of unity ``W`` [K, n-1, n-1]
    (zero off the kernel columns) and the level field's right-hand side ``h^2``."""

    def __init__(self, n: int, kind: str, m: int, device):
        dd = dict(dtype=torch.float64, device=device)
        ps = md.plain_square(n, kind, m, str(device))
        self.c1 = torch.tensor(kernel_coefficients(kind, n, m), **dd)
        H0s = ps.principal(torch.zeros(1, **dd), torch.float64)[0]
        W = self.c1[:, None, None] * H0s
        D = (self.c1[:, None, None] * W).sum(0)
        if float(D.min()) <= 0.0:
            raise ValueError("the kernel columns' steady modes are not a partition of unity")
        self.W = W / D
        self.h2 = 1.0 / n ** 2


@functools.lru_cache(maxsize=None)
def kernel_square(n: int, kind: str, m: int, device: str) -> KernelSquare:
    return KernelSquare(n, kind, m, torch.device(device))


def level_principal(ps: md.PlainSquare, s0: torch.Tensor, dtype=torch.float64) -> torch.Tensor:
    """The plain square's level field ``u_1^0 = (A_I^0)^-1 (h^2 1)`` [B, n-1, n-1]: its
    particular field for the unit uniform source (``mass(1) 1 = h^2`` on interior nodes)."""
    n = ps.n
    rhs = torch.full((s0.shape[0], 1, n - 1, n - 1), 1.0 / n ** 2, dtype=torch.float64,
                     device=s0.device)
    return ps.solve(rhs, s0, dtype)[:, 0]


def project(H_raw: torch.Tensor, u1: torch.Tensor, s0: torch.Tensor, ks: KernelSquare):
    """``H = H_raw - W r`` with ``r = H_raw c1 + s0 u1 - 1``, so that ``H c1 + s0 u1 = 1``.
    Float64 in, float64 out; returns (H, r)."""
    r = torch.einsum("k,bkij->bij", ks.c1, H_raw) + s0[:, None, None] * u1 - 1.0
    return H_raw - ks.W[None] * r[:, None], r


def residual(H: torch.Tensor, u1: torch.Tensor, s0: torch.Tensor, ks: KernelSquare) -> torch.Tensor:
    """``H c1 + s0 u1 - 1`` [B, n-1, n-1]: zero to rounding for the kernel-exact forms."""
    return torch.einsum("k,bkij->bij", ks.c1, H) + s0[:, None, None] * u1 - 1.0


# ---------------------------------------------------------------------------
# the source: its level and the rest
# ---------------------------------------------------------------------------


def split_source(x: torch.Tensor, w: torch.Tensor | None, g: torch.Tensor | None):
    """(gbar [B], level-free channel [B, n+1, n+1], scale [B]), float64.

    ``gbar`` is the load-weighted mean of the nodal source ``g`` (element weights ``w``
    = J~, each element's value the mean of its four nodes), ``g' = g - gbar``, the channel
    is ``J g' / rms(J g')`` with ``J`` from the log-Jacobian channel (its unknown constant
    cancels in the ratio), and the scale is ``rms(g')``.  Both ``gbar`` and the scale are
    homogeneous of degree one in ``g``, the channel of degree zero, and none of the three
    changes when a constant is added to ``g`` except ``gbar``, by that constant."""
    B, n1 = x.shape[0], x.shape[-1]
    z = torch.zeros(B, dtype=torch.float64, device=x.device)
    if g is None:
        return z, torch.zeros(B, n1, n1, dtype=torch.float64, device=x.device), z
    if w is None:
        raise ValueError("a source needs the element weights w (J~) to find its level")
    g = g.to(torch.float64)
    w = w.to(torch.float64)
    ge = 0.25 * (g[..., :-1, :-1] + g[..., :-1, 1:] + g[..., 1:, 1:] + g[..., 1:, :-1])
    gbar = (w * ge).sum((-2, -1)) / w.sum((-2, -1))
    gp = g - gbar[:, None, None]
    rms_g = torch.sqrt((g * g).mean((-2, -1)))
    scale = torch.sqrt((gp * gp).mean((-2, -1)))
    floor = LEVEL_FREE_FLOOR * torch.maximum(rms_g, gbar.abs())
    keep = scale > floor
    gp = torch.where(keep[:, None, None], gp, torch.zeros_like(gp))
    scale = torch.where(keep, scale, z)
    f = torch.exp(x[:, 0].to(torch.float64)) * gp
    rf = torch.sqrt((f * f).mean((-2, -1)))
    chan = torch.where((rf > 0)[:, None, None], f / torch.clamp(rf, min=1e-300)[:, None, None],
                       torch.zeros_like(f))
    return gbar, chan, scale


# ---------------------------------------------------------------------------
# the expert
# ---------------------------------------------------------------------------


def _backbone(backbone: str, size: str, cout: int, cin: int) -> nn.Module:
    if backbone == "unet":
        return md.UNet(md.WIDTHS[size], cout, cin)
    if backbone == "spectral":
        return md.SpectralNet(cout=cout, cin=cin)
    raise ValueError(backbone)


class KernelExpert(nn.Module):
    """The kernel-exact expert.  ``forward(x, s, s0, w, g)`` returns the interior modes
    [B, 4m, n-1, n-1] and the interior particular field [B, n-1, n-1] in float64, with
    ``H c1 + s0 u_1 = 1`` to rounding whatever the weights.  ``x`` holds Stage 0's five
    channels; the fifth (the source) is not read, the level-free channel being built
    from ``w`` (element J~) and ``g`` (the nodal source the particular field answers for;
    ``None`` for none).  Kept after each call: ``last_principal`` (H0, the objective's
    normaliser), ``last_level`` (u_1), ``last_gbar`` and ``last_kernel_residual`` (r before
    the projection)."""

    needs_load_fields = True

    def __init__(self, form: str = "split", backbone: str = "unet", size: str = "1M",
                 kind: str = "sine-hat", m: int = 16, p_size: str | None = None):
        super().__init__()
        if form not in FORMS:
            raise ValueError(form)
        self.form, self.backbone_name, self.size, self.kind, self.m = form, backbone, size, kind, m
        self.p_size = None if form == "kernel" else (p_size or size)
        K = 4 * m
        if form == "kernel":
            # outputs: K mode corrections, the particular correction, the level correction
            self.net = _backbone(backbone, size, K + 2, 5)
            self.pnet = None
        else:
            # the mode network: no load channel; K mode corrections and the level correction
            self.net = _backbone(backbone, size, K + 1, 4)
            self.pnet = _backbone(backbone, self.p_size, 1, 5)

    def config(self) -> dict:
        return {"backbone": self.backbone_name, "size": self.size, "form": self.form,
                "kind": self.kind, "m": self.m, "p_size": self.p_size,
                "parameters": md.parameters(self)}

    def forward(self, x, s, s0, w=None, g=None):
        n = x.shape[-1] - 1
        dev = str(x.device)
        ps = md.plain_square(n, self.kind, self.m, dev)
        ks = kernel_square(n, self.kind, self.m, dev)
        s0 = s0.to(torch.float64)
        gbar, chan, scale = split_source(x, w, g)
        xp = torch.cat([x[:, :4], chan[:, None].to(x.dtype)], dim=1)
        if self.form == "kernel":
            out = self.net(xp, s).to(torch.float64)
            N, Np, N1 = out[:, :-2], out[:, -2], out[:, -1]
        else:
            out = self.net(x[:, :4], s).to(torch.float64)
            N, N1 = out[:, :-1], out[:, -1]
            Np = self.pnet(xp, s).to(torch.float64)[:, 0]
        beta = ps.beta[1:-1, 1:-1]
        H0 = ps.principal(s0, torch.float32).to(torch.float64)
        u10 = level_principal(ps, s0, torch.float32).to(torch.float64)
        a = u10.abs().amax(dim=(-2, -1))
        H_raw = H0 + beta * N[..., 1:-1, 1:-1]
        u1 = u10 + a[:, None, None] * beta * N1[..., 1:-1, 1:-1]
        H, r = project(H_raw, u1, s0, ks)
        up = gbar[:, None, None] * u1 + scale[:, None, None] * beta * Np[..., 1:-1, 1:-1]
        self.last_principal = H0.detach()
        self.last_level = u1.detach()
        self.last_gbar = gbar.detach()
        self.last_kernel_residual = r.detach()
        return H, up


def make(backbone: str, size: str, form: str, kind: str = "sine-hat", m: int = 16,
         p_size: str | None = None) -> nn.Module:
    """Stage 0's `model.Expert` for its forms, `KernelExpert` for these."""
    if form in FORMS:
        return KernelExpert(form, backbone, size, kind, m, p_size)
    return md.Expert(backbone, size, form, kind, m)


def call(expert: nn.Module, batch: dict):
    """One forward pass on a training batch, passing ``w`` and ``g`` where the form reads
    them (Stage 0's forms are called exactly as before)."""
    if getattr(expert, "needs_load_fields", False):
        return expert(batch["inputs"], batch["scalars"], batch["s0"], w=batch["w"], g=batch["g"])
    return expert(batch["inputs"], batch["scalars"], batch["s0"])


def kernel_error_torch(batch: dict, r: torch.Tensor, trace_Lambda: torch.Tensor) -> torch.Tensor:
    """``e_K = ||r||_{A_I} / sqrt(tr Lambda)`` per example, r [B, n-1, n-1] interior."""
    E = md.energy(batch["a11"], batch["a12"], batch["a22"], batch["sigma"], md.full_p(r))[:, 0]
    return torch.sqrt(torch.clamp(E, min=0.0) / trace_Lambda)


# ---------------------------------------------------------------------------
# the measure for any expert, through the host's interface (numpy, float64)
# ---------------------------------------------------------------------------


def uniform_state(problem):
    """The piece's uniform state at unit level: the same chart, material and step, with
    the source ``g = s0`` (``T_old = 1``; zero when steady).  With data 1 on every side its
    exact answer is 1."""
    return replace(problem, g=np.full_like(problem.g, problem.s0, dtype=float))


def kernel_residual(expert, problem, pb) -> np.ndarray:
    """``r_K = u_p(s0 1) + H c1 - 1`` on the interior nodes, from one call to ``expert``
    (any expert with ``respond``) on the piece's uniform state."""
    r = expert.respond([uniform_state(problem)], lambda n_: pb)[0]
    c1 = kernel_coefficients(pb.kind, pb.n, pb.m)
    return r.up + r.H @ c1 - 1.0


def kernel_measures(r: np.ndarray, A_I, trace_Lambda: float) -> dict:
    """``e_K``, ``e_K,inf`` and ``||r_K||^2_{A_I}`` for one piece."""
    rr = float(r @ (A_I @ r))
    return {"e_K": math.sqrt(max(rr, 0.0) / trace_Lambda), "e_K_inf": float(np.abs(r).max()),
            "r_energy": rr}


def level_weighted_e_H(e_H: float, e_K: float, theta: float) -> float:
    """``e_H(theta) = (e_H^2 + theta^2 e_K^2)^(1/2)``: the root-mean-square network term for
    traces ``theta c1 + xi`` (``xi`` unit and uncorrelated) with source ``theta s0``, over the
    energy of the variation ``xi`` (``tr Lambda``)."""
    return math.sqrt(e_H ** 2 + (theta * e_K) ** 2)
