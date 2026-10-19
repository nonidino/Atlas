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

**The corner bubble, ``bubble="corner"`` (W358).**  Stage 0's bubble
``16 xi (1 - xi) eta (1 - eta)`` vanishes like the distance to a side along the sides,
but like its square at a corner.  At a corner whose physical angle is ``alpha`` the
exact modes vanish like ``r^(pi / alpha)``: slower than ``r^2`` when the corner is
obtuse (the chart shears there, ``Im mu != 0``), and like ``r^2 log r`` at a right
angle with varying coefficients.  So the quotient the head must output grows without
bound at the node diagonally next to such a corner (16, 50, 89 at n = 32, 64, 128 on
the validation pieces of `scripts/chartop_diagnose.py`).  The corner bubble

    beta_c = 4 xi (1 - xi) eta (1 - eta) / (r_0 r_1 r_2 r_3),

``r_c`` the distance to corner ``c``, is one at the centre and vanishes like the distance
to the boundary everywhere, corners included (``xi eta / r`` near a corner).  Every
convex corner has ``pi / alpha > 1``, so the quotient stays bounded whatever the chart,
with no corner angle to estimate.  It is the same fixed field for every piece,
invariant under the square's eight symmetries bit for bit, and zero on the loop, so the
traces stay exact and the projection is untouched.  `scripts/chartop_corner_controls.py`
compares it with the other candidates on the same validation pieces.

**The strips, ``strips=True`` (W358's C3).**  68.5% of Stage 0's ``e_H^2`` sat in the
side sines ``j = 9-15``, whose exact modes carry layers about ``1/(j pi)`` wide along
their own side.  With the strips on, those 28 columns are not the network's: each is
computed classically by a **strip solve** along its side (`strip_modes`), the piece's
true interior operator on the interior nodes within ``n/8`` cells of that side, with
the network's own principal part ``H0`` as data beyond the strip.  Beyond the strip a
strip column is ``H0``, whatever the weights.  The network returns the other 36 columns
(the hats and ``j = 1-8``) and nothing for these.  The strips need each piece's assembled
operator, which the call carries as ``ops`` (`interior_blocks` per example), never as a
network input: `call` assembles it from a batch's element arrays, and
`expert.LearnedExpert` from each piece's own matrix.  The projection comes last and
moves only the kernel columns, so the strip columns pass through it bit for bit and
``H c1 + s0 u_1 = 1`` still holds to rounding.  Off by default; sine-hat with
``m = 16`` only, the basis and width the controls measured.
"""

from __future__ import annotations

import functools
import math
from dataclasses import replace

import time

import numpy as np
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn

from . import fe
from . import model as md
from . import ports as pt

FORMS = ("kernel", "split")
#: Stage 0's bubble, and W358's, which vanishes like the distance to the boundary
BUBBLES = ("square", "corner")
#: W358's C3: the side sines j = 9-15 by strip solves, each strip n/8 cells deep
STRIP_J = (9, 15)
STRIP_WIDTH = 1.0 / 8.0
#: a level-free source whose RMS is below this fraction of the source's own size is
#: rounding (a uniform source minus its float64 mean), and is set to zero
LEVEL_FREE_FLOOR = 1e-12


# ---------------------------------------------------------------------------
# the bubbles
# ---------------------------------------------------------------------------


def corner_bubble(n: int) -> np.ndarray:
    """W358's bubble on the (n+1)^2 nodes [j, i]: ``4 xi (1 - xi) eta (1 - eta) /
    (r_0 r_1 r_2 r_3)`` with ``r_c`` the distance to corner ``c``, zero on the loop.  The
    factors are paired so that the field is invariant under the square's symmetries
    bit for bit, not only to rounding."""
    i = np.arange(n + 1, dtype=np.float64)
    xi, xm = i[None, 1:-1] / n, (n - i[None, 1:-1]) / n
    et, em = i[1:-1, None] / n, (n - i[1:-1, None]) / n
    r00, r10, r11, r01 = np.hypot(xi, et), np.hypot(xm, et), np.hypot(xm, em), np.hypot(xi, em)
    b = np.zeros((n + 1, n + 1))
    b[1:-1, 1:-1] = 4.0 * ((xi * xm) * (et * em)) / ((r00 * r11) * (r10 * r01))
    b.setflags(write=False)
    return b


@functools.lru_cache(maxsize=None)
def _corner_bubble_torch(n: int, device: str) -> torch.Tensor:
    return torch.tensor(corner_bubble(n), dtype=torch.float64, device=torch.device(device))


def bubble_field(name: str, ps: md.PlainSquare) -> torch.Tensor:
    """The bubble [n+1, n+1] (float64) on ``ps``'s grid and device: Stage 0's ``ps.beta``
    for ``"square"``, `corner_bubble` for ``"corner"``."""
    if name == "square":
        return ps.beta
    if name == "corner":
        return _corner_bubble_torch(ps.n, str(ps.device))
    raise ValueError(f"bubble {name!r}: one of {BUBBLES}")


# ---------------------------------------------------------------------------
# the strips (W358's C3)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def strip_columns(kind: str, m: int) -> tuple:
    """Per side ``s``, the columns of side ``s``'s sines ``j = 9-15``, which the strips
    compute.  Read from the basis's own labels, which do not depend on ``n``."""
    if kind != "sine-hat":
        raise ValueError(f"the strips were measured in the sine-hat basis, not {kind!r}")
    if m - 1 < STRIP_J[1]:
        raise ValueError(f"the strips compute the sines j = {STRIP_J[0]}-{STRIP_J[1]}: "
                         f"need m >= {STRIP_J[1] + 1}, not {m}")
    cols = pt.basis(kind, m, m).columns
    return tuple(tuple(k for k, c in enumerate(cols)
                       if c.what == "side" and c.index == s and STRIP_J[0] <= c.j <= STRIP_J[1])
                 for s in range(4))


def learned_columns(kind: str, m: int) -> tuple:
    """The columns the network returns when the strips are on: all the others."""
    high = {k for cs in strip_columns(kind, m) for k in cs}
    return tuple(k for k in range(4 * m) if k not in high)


@functools.lru_cache(maxsize=None)
def strip_nodes(n: int) -> tuple:
    """Per side, a mask over the (n-1)^2 interior nodes (index order): those within
    ``STRIP_WIDTH n`` cells of that side."""
    JJ, II = np.meshgrid(np.arange(1, n), np.arange(1, n), indexing="ij")
    dist = [JJ.ravel(), (n - II).ravel(), (n - JJ).ravel(), II.ravel()]
    out = tuple(d <= STRIP_WIDTH * n for d in dist)
    for S in out:
        S.setflags(write=False)
    return out


def interior_blocks(M, n: int):
    """``(A_I, M_IG)`` of a piece's energy matrix ``M``, sliced as `fe.Exact` slices it,
    with no factorisation: what the strips need of the piece."""
    loop, inner = fe.grid(n)
    M = M.tocsr()
    return M[inner][:, inner].tocsc(), M[inner][:, loop].tocsr()


def batch_operators(batch: dict) -> list:
    """`interior_blocks` per example, from a batch's element arrays ``a11, a12, a22,
    sigma`` [B, n, n], assembled as `piece.PieceProblem.matrix` assembles them."""
    a = {k: batch[k].detach().to("cpu", torch.float64).numpy() for k in ("a11", "a12", "a22", "sigma")}
    n = a["a11"].shape[-1]
    return [interior_blocks(fe.assemble(n, a["a11"][b], a["a12"][b], a["a22"][b], a["sigma"][b]), n)
            for b in range(a["a11"].shape[0])]


def strip_modes(A_I, M_IG, Q, H0, n: int, kind: str = "sine-hat") -> np.ndarray:
    """``H0`` [(n-1)^2, 4m] with the strip columns replaced on their strips.  For side
    ``s`` with strip nodes ``S`` and sines ``j = 9-15``,

        A_I[S, S] H[S] = (B Q)[S] - A_I[S, ~S] H0[~S],      B = -M_IG,

    the piece's own rows on the strip, with the boundary data and ``H0`` beyond the
    strip on the right.  The operations are those of `scripts/chartop_corner_controls.py`
    (class ``Piece``, C3) in its order, so the answer is the controls' bit for bit.  Each
    side's columns are solved on its own strip only, so the sides' order does not
    matter."""
    m = Q.shape[1] // 4
    cols = strip_columns(kind, m)
    A = A_I.tocsr()
    BQ = -np.asarray(M_IG @ Q)
    Hs = np.array(H0, dtype=np.float64, order="C", copy=True)
    for s, S in enumerate(strip_nodes(n)):
        cs = list(cols[s])
        rhs = BQ[S][:, cs] - A[S][:, ~S] @ Hs[~S][:, cs]
        Hs[np.flatnonzero(S)[:, None], cs] = spla.splu(A[S][:, S].tocsc()).solve(rhs)
    return Hs


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
    """The kernel-exact expert.  ``forward(x, s, s0, w, g, ops)`` returns the interior modes
    [B, 4m, n-1, n-1] and the interior particular field [B, n-1, n-1] in float64, with
    ``H c1 + s0 u_1 = 1`` to rounding whatever the weights.  ``x`` holds Stage 0's five
    channels; the fifth (the source) is not read, the level-free channel being built
    from ``w`` (element J~) and ``g`` (the nodal source the particular field answers for;
    ``None`` for none).  Kept after each call: ``last_principal`` (H0, the objective's
    normaliser), ``last_level`` (u_1), ``last_gbar`` and ``last_kernel_residual`` (r before
    the projection).  ``bubble`` multiplies every correction: Stage 0's ``"square"`` (the
    default) or W358's ``"corner"``.

    ``strips=True`` (W358's C3, off by default): the side sines ``j = 9-15`` come from
    `strip_modes`, and the network returns corrections for the ``learned`` columns only.
    The call then needs ``ops``, each example's ``(A_I, M_IG)`` (`interior_blocks`), and
    keeps the strips' own time on ``last_strip_seconds``."""

    needs_load_fields = True

    def __init__(self, form: str = "split", backbone: str = "unet", size: str = "1M",
                 kind: str = "sine-hat", m: int = 16, p_size: str | None = None,
                 bubble: str = "square", strips: bool = False):
        super().__init__()
        if form not in FORMS:
            raise ValueError(form)
        if bubble not in BUBBLES:
            raise ValueError(bubble)
        self.bubble = bubble
        self.strips = bool(strips)
        self.form, self.backbone_name, self.size, self.kind, self.m = form, backbone, size, kind, m
        self.p_size = None if form == "kernel" else (p_size or size)
        K = 4 * m
        if self.strips:
            self.learned = learned_columns(kind, m)
            self.stripped = tuple(k for cs in strip_columns(kind, m) for k in cs)
            K = len(self.learned)
        if form == "kernel":
            # outputs: K mode corrections, the particular correction, the level correction
            self.net = _backbone(backbone, size, K + 2, 5)
            self.pnet = None
        else:
            # the mode network: no load channel; K mode corrections and the level correction
            self.net = _backbone(backbone, size, K + 1, 4)
            self.pnet = _backbone(backbone, self.p_size, 1, 5)

    @property
    def needs_operator(self) -> bool:
        return self.strips

    def config(self) -> dict:
        return {"backbone": self.backbone_name, "size": self.size, "form": self.form,
                "kind": self.kind, "m": self.m, "p_size": self.p_size, "bubble": self.bubble,
                "strips": self.strips, "parameters": md.parameters(self)}

    def _strip_columns(self, ops, H0: torch.Tensor) -> torch.Tensor:
        """The strip columns [B, 28, n-1, n-1] (float64), with this call's ``H0`` beyond
        the strips."""
        B, K, n1 = H0.shape[0], H0.shape[1], H0.shape[-1]
        n = n1 + 1
        Q = pt.basis(self.kind, n, self.m).Q
        H0n = H0.detach().cpu().numpy()
        cols = list(self.stripped)
        t0 = time.perf_counter()
        out = np.empty((B, len(cols), n1, n1))
        for b, (A_I, M_IG) in enumerate(ops):
            Hs = strip_modes(A_I, M_IG, Q, H0n[b].reshape(K, -1).T, n, self.kind)
            out[b] = Hs[:, cols].T.reshape(len(cols), n1, n1)
        self.last_strip_seconds = time.perf_counter() - t0
        return torch.tensor(out, dtype=torch.float64, device=H0.device)

    def forward(self, x, s, s0, w=None, g=None, ops=None):
        if self.strips and (ops is None or len(ops) != x.shape[0]):
            raise ValueError("the strips need each example's operator: ops=[(A_I, M_IG), ...]")
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
        beta = bubble_field(self.bubble, ps)[1:-1, 1:-1]
        H0 = ps.principal(s0, torch.float32).to(torch.float64)
        u10 = level_principal(ps, s0, torch.float32).to(torch.float64)
        a = u10.abs().amax(dim=(-2, -1))
        if self.strips:
            learned = torch.tensor(self.learned, device=H0.device)
            H_raw = H0.clone()
            H_raw[:, learned] = H0[:, learned] + beta * N[..., 1:-1, 1:-1]
            H_raw[:, torch.tensor(self.stripped, device=H0.device)] = self._strip_columns(ops, H0)
        else:
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
         p_size: str | None = None, bubble: str = "square", strips: bool = False) -> nn.Module:
    """Stage 0's `model.Expert` for its forms, `KernelExpert` for these.  Stage 0's forms
    have only the square bubble and no strips."""
    if form in FORMS:
        return KernelExpert(form, backbone, size, kind, m, p_size, bubble, strips)
    if bubble != "square":
        raise ValueError(f"Stage 0's form {form!r} has only the square bubble")
    if strips:
        raise ValueError(f"Stage 0's form {form!r} has no strips")
    return md.Expert(backbone, size, form, kind, m)


def call(expert: nn.Module, batch: dict):
    """One forward pass on a training batch, passing ``w`` and ``g`` where the form reads
    them, and the operators the strips need, assembled from the batch's element arrays
    (Stage 0's forms are called exactly as before)."""
    if getattr(expert, "needs_operator", False):
        return expert(batch["inputs"], batch["scalars"], batch["s0"], w=batch["w"], g=batch["g"],
                      ops=batch_operators(batch))
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
