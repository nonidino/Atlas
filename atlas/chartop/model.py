"""The chart operator's network, its output forms, and the energy objectives, in torch.

**The backbone** ([[chart-operator-architecture]] section 3.4, decision 10): a U-shaped
convolutional operator in the manner of CNO on the reference grid's ``(n+1)^2``
vertices.  Four levels; two 3 x 3 convolutions per level with GELU; node-aligned
downsampling (a 3 x 3 average with stride 2 keeps the corners, ``2^k + 1 -> 2^(k-1) + 1``)
and bilinear upsampling with aligned corners; skip connections; FiLM on the scalars.
Every convolution pads by **even reflection about the boundary node** (the inputs'
extension); the outputs are corrections multiplied by the bubble
``beta = 16 xi (1 - xi) eta (1 - eta)``, which vanishes on the boundary nodes, so the
traces are exact whatever the weights (the **odd** side of the even/odd rule).
Widths (24, 48, 96, 192) give about 1.1 M parameters, the default; (12, 24, 48, 96)
and (40, 80, 160, 320) are comparison C5's 0.3 M and 3 M.

**Comparison C4's arm**: a sine-cosine spectral operator -- Fourier layers on the
even extension of the node grid (period ``2n``), whose modes are the cosine basis of
the square, with the same bubble on its outputs.

**Output forms** (one forward pass returns ``4m + 1 = 65`` fields):
- ``bubble`` (the default): ``H_k = H0_k + beta N_k``, ``u_p = beta N_p``, with
  ``H0`` the plain square's exact modes at the piece's Fourier number (DST-I, exact);
- ``direct`` (comparison C3): ``H_k = N_k`` and ``u_p = N_p`` on the interior nodes, no
  principal part;
- ``fastsolve`` (comparison C7): ``H_k = H0_k + (A_I^0)^-1 (h^2 N_k)``,
  ``u_p = (A_I^0)^-1 (h^2 N_p)``, the network's field passed through the plain square's
  inverse (``V = V0 + L0^-1 N``).

**The energy** of a nodal field, exactly as `fe.energy` (and checked against it), in
float64: decision D4 puts the network in bfloat16 on the GPU and every energy, Gram and
loss in float64.  The label-free objective (T26) per example is

    L_en = [ sum_k E(V_k) ] / Z_H + [ E(u_p) - 2 <g, u_p>_w ] / Z_p,

with ``Z_H``, ``Z_p`` the energies of the principal part's own trial fields in the
same example (independent of the weights, so the minimiser is unchanged);
``L_sup`` replaces the brackets by ``sum_k ||H_k - H_k^exact||^2_{A_I}`` and
``||u_p - u_p^exact||^2_{A_I}``.  They differ by a constant (T26), so C2 measures only
floating point.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import fe
from . import ports as pt

N_IN, N_SCALARS = 5, 2
WIDTHS = {"0.3M": (12, 24, 48, 96), "1M": (24, 48, 96, 192), "3M": (40, 80, 160, 320)}
FORMS = ("bubble", "direct", "fastsolve")


# ---------------------------------------------------------------------------
# backbones
# ---------------------------------------------------------------------------


def _pad(x):
    return F.pad(x, (1, 1, 1, 1), mode="reflect")


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3)
        self.c2 = nn.Conv2d(cout, cout, 3)
        self.film = nn.Linear(N_SCALARS, 2 * cout)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, s):
        g, b = self.film(s).chunk(2, dim=1)
        x = F.gelu(self.c1(_pad(x)))
        x = self.c2(_pad(x))
        return F.gelu(x * (1 + g[:, :, None, None]) + b[:, :, None, None])


def _down(x):
    """Node-aligned coarsening: (2^k + 1) -> (2^(k-1) + 1), a 3 x 3 average, stride 2."""
    return F.avg_pool2d(_pad(x), 3, stride=2)


def _up(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=True)


class UNet(nn.Module):
    def __init__(self, widths=WIDTHS["1M"], cout: int = 65):
        super().__init__()
        w = list(widths)
        self.enc = nn.ModuleList([Block(N_IN, w[0])] + [Block(w[i - 1], w[i]) for i in range(1, len(w))])
        self.dec = nn.ModuleList([Block(w[i + 1] + w[i], w[i]) for i in reversed(range(len(w) - 1))])
        self.head = nn.Conv2d(w[0], cout, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, s):
        skips = []
        for k, blk in enumerate(self.enc):
            if k > 0:
                x = _down(x)
            x = blk(x, s)
            skips.append(x)
        for k, blk in enumerate(self.dec):
            sk = skips[-2 - k]
            x = blk(torch.cat([_up(x, sk.shape[-2:]), sk], dim=1), s)
        return self.head(x)


class SpectralLayer(nn.Module):
    """A Fourier layer on the even extension of a node grid: the cosine basis of the
    square, truncated to ``k`` modes per direction."""

    def __init__(self, cin, cout, k):
        super().__init__()
        self.k = k
        scale = 1.0 / (cin * cout)
        self.w1 = nn.Parameter(scale * torch.randn(cin, cout, k, k, 2))
        self.w2 = nn.Parameter(scale * torch.randn(cin, cout, k, k, 2))
        self.lin = nn.Conv2d(cin, cout, 1)

    def forward(self, x):
        N1 = x.shape[-1]
        ext = torch.cat([x, x.flip(-1)[..., 1:-1]], dim=-1)
        ext = torch.cat([ext, ext.flip(-2)[..., 1:-1, :]], dim=-2)
        X = torch.fft.rfft2(ext.float())
        k = min(self.k, X.shape[-1], X.shape[-2] // 2)
        W1 = torch.view_as_complex(self.w1[:, :, :k, :k].contiguous())
        W2 = torch.view_as_complex(self.w2[:, :, :k, :k].contiguous())
        out = torch.zeros(x.shape[0], W1.shape[1], X.shape[-2], X.shape[-1],
                          dtype=X.dtype, device=x.device)
        out[..., :k, :k] = torch.einsum("bixy,ioxy->boxy", X[..., :k, :k], W1)
        out[..., -k:, :k] = torch.einsum("bixy,ioxy->boxy", X[..., -k:, :k], W2)
        y = torch.fft.irfft2(out, s=ext.shape[-2:])[..., :N1, :N1]
        return y.to(x.dtype) + self.lin(x)


class SpectralNet(nn.Module):
    def __init__(self, width: int = 32, k: int = 8, layers: int = 4, cout: int = 65):
        super().__init__()
        self.lift = nn.Conv2d(N_IN, width, 1)
        self.layers = nn.ModuleList([SpectralLayer(width, width, k) for _ in range(layers)])
        self.film = nn.ModuleList([nn.Linear(N_SCALARS, 2 * width) for _ in range(layers)])
        for f_ in self.film:
            nn.init.zeros_(f_.weight)
            nn.init.zeros_(f_.bias)
        self.head = nn.Conv2d(width, cout, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, s):
        x = self.lift(x)
        for lay, f_ in zip(self.layers, self.film):
            g, b = f_(s).chunk(2, dim=1)
            x = F.gelu(lay(x) * (1 + g[:, :, None, None]) + b[:, :, None, None])
        return self.head(x)


def parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------------------
# the plain square in torch: DST-I, the symbol, the principal part
# ---------------------------------------------------------------------------


def dst1(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Orthonormal type-I DST along ``dim`` (self-inverse), via the FFT of the odd
    extension of length 2(N+1)."""
    x = x.transpose(dim, -1)
    N = x.shape[-1]
    z = torch.zeros(x.shape[:-1] + (2 * (N + 1),), dtype=x.dtype, device=x.device)
    z[..., 1:N + 1] = x
    z[..., N + 2:] = -x.flip(-1)
    Z = torch.fft.fft(z, dim=-1)
    y = -Z.imag[..., 1:N + 1] * math.sqrt(2.0 / (N + 1)) / 2.0
    return y.transpose(dim, -1)


class PlainSquare:
    """Cached per (n, device): the 1-D eigenvalues, and the port basis's coupling to
    the first interior ring, split into its stiffness and mass parts."""

    def __init__(self, n: int, kind: str, m: int, device):
        self.n, self.device = n, device
        c = np.cos(np.arange(1, n) * math.pi / n)
        k1 = 2.0 * n * (1.0 - c)
        m1 = (4.0 + 2.0 * c) / (6.0 * n)
        dd = dict(dtype=torch.float64, device=device)
        self.km = torch.tensor(np.outer(m1, k1) + np.outer(k1, m1), **dd)     # [q, p]
        self.mm = torch.tensor(np.outer(m1, m1), **dd)
        pb = pt.basis(kind, n, m)
        loop, inner = fe.grid(n)
        K = fe.assemble(n, 1.0, 0.0, 1.0)
        Mm = fe.mass(n, 1.0)
        KQ = np.asarray(K[inner][:, loop] @ pb.Q)
        MQ = np.asarray(Mm[inner][:, loop] @ pb.Q)
        self.KQ = torch.tensor(KQ.T.reshape(-1, n - 1, n - 1), **dd)        # [K, j, i]
        self.MQ = torch.tensor(MQ.T.reshape(-1, n - 1, n - 1), **dd)
        QB = np.zeros((pb.size, (n + 1) ** 2))
        QB[:, loop] = pb.Q.T
        self.QB = torch.tensor(QB.reshape(-1, n + 1, n + 1), **dd)           # boundary images
        t = torch.arange(n + 1, dtype=torch.float64, device=device) / n
        XI, ET = torch.meshgrid(t, t, indexing="xy")
        self.beta = 16.0 * XI * (1 - XI) * ET * (1 - ET)
        self.inner = torch.zeros(n + 1, n + 1, dtype=torch.bool, device=device)
        self.inner[1:-1, 1:-1] = True

    def solve(self, r: torch.Tensor, s0: torch.Tensor, dtype=torch.float64) -> torch.Tensor:
        """(A_I^0)^-1 r for r [B, K, n-1, n-1] with per-example reactions s0 [B], in
        ``dtype`` (float32 inside the network: its result is part of the output, so
        its rounding is an interior error like any other; float64 elsewhere)."""
        lam = (self.km[None, None] + s0.to(torch.float64)[:, None, None, None]
               * self.mm[None, None]).to(dtype)
        R = dst1(dst1(r.to(dtype), -1), -2)
        return dst1(dst1(R / lam, -1), -2)

    def principal(self, s0: torch.Tensor, dtype=torch.float64) -> torch.Tensor:
        """H0 [B, K, n-1, n-1] at reactions s0 [B]."""
        rhs = -(self.KQ[None] + s0.to(torch.float64)[:, None, None, None] * self.MQ[None])
        return self.solve(rhs, s0, dtype)


@functools.lru_cache(maxsize=None)
def plain_square(n: int, kind: str, m: int, device: str) -> PlainSquare:
    return PlainSquare(n, kind, m, torch.device(device))


# ---------------------------------------------------------------------------
# the expert
# ---------------------------------------------------------------------------


class Expert(nn.Module):
    """Backbone + output form.  ``forward`` returns the interior modes
    [B, 4m, n-1, n-1] and the interior particular field [B, n-1, n-1], in float64."""

    def __init__(self, backbone: str = "unet", size: str = "1M", form: str = "bubble",
                 kind: str = "sine-hat", m: int = 16):
        super().__init__()
        if form not in FORMS:
            raise ValueError(form)
        self.backbone_name, self.size, self.form, self.kind, self.m = backbone, size, form, kind, m
        cout = 4 * m + 1
        if backbone == "unet":
            self.net = UNet(WIDTHS[size], cout)
        elif backbone == "spectral":
            self.net = SpectralNet(cout=cout)
        else:
            raise ValueError(backbone)

    def config(self) -> dict:
        return {"backbone": self.backbone_name, "size": self.size, "form": self.form,
                "kind": self.kind, "m": self.m, "parameters": parameters(self)}

    def forward(self, x, s, s0):
        """The principal part is computed in float32 (its rounding is part of the output)
        and kept on ``self.last_principal`` for the objective's normaliser."""
        n = x.shape[-1] - 1
        ps = plain_square(n, self.kind, self.m, str(x.device))
        out = self.net(x, s).to(torch.float64)
        N, Np = out[:, :-1], out[:, -1]
        H0 = None
        if self.form == "bubble":
            H0 = ps.principal(s0, torch.float32).to(torch.float64)
            H = H0 + (ps.beta[None, None] * N)[..., 1:-1, 1:-1]
            up = (ps.beta[None] * Np)[..., 1:-1, 1:-1]
        elif self.form == "direct":
            H = N[..., 1:-1, 1:-1]
            up = Np[..., 1:-1, 1:-1]
        else:
            h2 = 1.0 / n ** 2
            H0 = ps.principal(s0, torch.float32).to(torch.float64)
            H = H0 + ps.solve(h2 * N[..., 1:-1, 1:-1], s0, torch.float32).to(torch.float64)
            up = ps.solve(h2 * Np[:, None, 1:-1, 1:-1], s0, torch.float32).to(torch.float64)[:, 0]
        self.last_principal = None if H0 is None else H0.detach()
        return H, up


# ---------------------------------------------------------------------------
# energies and objectives (float64)
# ---------------------------------------------------------------------------


def energy(a11, a12, a22, sigma, U):
    """u^T M u for nodal fields U [B, K, n+1, n+1]; element arrays [B, n, n]."""
    n = U.shape[-1] - 1
    a11, a12, a22 = a11[:, None], a12[:, None], a22[:, None]
    u00, u10 = U[..., :-1, :-1], U[..., :-1, 1:]
    u11, u01 = U[..., 1:, 1:], U[..., 1:, :-1]
    d0, d1 = u10 - u00, u11 - u01
    e0, e1 = u01 - u00, u11 - u10
    dens = (a11 * (d0 * d0 + d0 * d1 + d1 * d1) / 3.0 + a22 * (e0 * e0 + e0 * e1 + e1 * e1) / 3.0
            + a12 * (d0 + d1) * (e0 + e1) / 2.0)
    if sigma is not None:
        sq = u00 * u00 + u10 * u10 + u11 * u11 + u01 * u01
        ed = u00 * u10 + u10 * u11 + u11 * u01 + u01 * u00
        dg = u00 * u11 + u10 * u01
        dens = dens + (sigma[:, None] / n ** 2) * (sq / 9.0 + ed / 9.0 + dg / 18.0)
    return dens.sum(dim=(-2, -1))


def mass_form(w, G, U):
    """<g, u>_w = g^T mass(w) u per field: G [B, n+1, n+1], U [B, K, n+1, n+1]."""
    n = U.shape[-1] - 1
    G = G[:, None]
    g00, g10, g11, g01 = G[..., :-1, :-1], G[..., :-1, 1:], G[..., 1:, 1:], G[..., 1:, :-1]
    u00, u10, u11, u01 = U[..., :-1, :-1], U[..., :-1, 1:], U[..., 1:, 1:], U[..., 1:, :-1]
    sq = g00 * u00 + g10 * u10 + g11 * u11 + g01 * u01
    ed = (g00 * u10 + g10 * u00 + g10 * u11 + g11 * u10 + g11 * u01 + g01 * u11
          + g01 * u00 + g00 * u01)
    dg = g00 * u11 + g11 * u00 + g10 * u01 + g01 * u10
    dens = (w[:, None] / n ** 2) * (sq / 9.0 + ed / 18.0 + dg / 36.0)
    return dens.sum(dim=(-2, -1))


def full(ps: PlainSquare, H: torch.Tensor) -> torch.Tensor:
    """Full nodal mode fields [B, K, n+1, n+1]: boundary from Q, interior from H."""
    V = ps.QB[None].expand(H.shape[0], -1, -1, -1).clone()
    V[..., 1:-1, 1:-1] = H
    return V


def full_p(up: torch.Tensor) -> torch.Tensor:
    B, m1 = up.shape[0], up.shape[-1]
    U = torch.zeros(B, 1, m1 + 2, m1 + 2, dtype=up.dtype, device=up.device)
    U[:, 0, 1:-1, 1:-1] = up
    return U


def objectives(batch: dict, H, up, kind: str, m: int, labels: dict | None = None,
               objective: str = "label-free", H0: torch.Tensor | None = None) -> dict:
    """Per-example normalised objective and diagnostics.  ``batch`` holds float64
    a11, a12, a22, sigma, w, g [B, ...] and s0 [B]."""
    n = H.shape[-1] + 1
    ps = plain_square(n, kind, m, str(H.device))
    a11, a12, a22, sig = batch["a11"], batch["a12"], batch["a22"], batch["sigma"]
    w, g, s0 = batch["w"], batch["g"], batch["s0"]
    with torch.no_grad():
        V0 = full(ps, ps.principal(s0, torch.float32).to(torch.float64) if H0 is None else H0)
        ZH = energy(a11, a12, a22, sig, V0).sum(-1)
        # the plain square's particular field for this load, as Z_p's reference
        Gb = _load(w, g)
        up0 = ps.solve(Gb[:, None, 1:-1, 1:-1], s0, torch.float32).to(torch.float64)[:, 0]
        U0 = full_p(up0)
        Zp = (energy(a11, a12, a22, sig, U0)[:, 0] - 2 * mass_form(w, g, U0)[:, 0]).abs()
        Zp = torch.clamp(Zp, min=1e-30)
    V = full(ps, H)
    Up = full_p(up)
    EH = energy(a11, a12, a22, sig, V).sum(-1)
    Ep = energy(a11, a12, a22, sig, Up)[:, 0] - 2 * mass_form(w, g, Up)[:, 0]
    out = {"ZH": ZH, "Zp": Zp, "trace_gram": EH, "p_energy": Ep}
    if objective == "label-free":
        out["loss"] = EH / ZH + Ep / Zp
    elif objective == "supervised":
        if labels is None:
            raise ValueError("the supervised objective needs labels")
        dH = full(ps, H - labels["H"]) - ps.QB[None]
        dp = full_p(up - labels["up"])
        LH = energy(a11, a12, a22, sig, dH).sum(-1)
        Lp = energy(a11, a12, a22, sig, dp)[:, 0]
        out["loss"] = LH / ZH + Lp / Zp
    else:
        raise ValueError(objective)
    if labels is not None:
        with torch.no_grad():
            dH = full(ps, H - labels["H"]) - ps.QB[None]
            dp = full_p(up - labels["up"])
            out["e_H"] = torch.sqrt(energy(a11, a12, a22, sig, dH).sum(-1) / labels["trace_Lambda"])
            out["e_p"] = torch.sqrt(energy(a11, a12, a22, sig, dp)[:, 0]
                                    / torch.clamp(labels["up_energy"], min=1e-300))
    return out


def _load(w, g):
    """The nodal load b = mass(w) g as a field [B, n+1, n+1] (its interior is b_I)."""
    n = g.shape[-1] - 1
    out = torch.zeros_like(g)
    W = w / n ** 2
    g00, g10, g11, g01 = g[..., :-1, :-1], g[..., :-1, 1:], g[..., 1:, 1:], g[..., 1:, :-1]
    # row of the unit mass matrix for each local node: (1/9, 1/18, 1/36, 1/18) around it
    b00 = W * (g00 / 9 + (g10 + g01) / 18 + g11 / 36)
    b10 = W * (g10 / 9 + (g00 + g11) / 18 + g01 / 36)
    b11 = W * (g11 / 9 + (g10 + g01) / 18 + g00 / 36)
    b01 = W * (g01 / 9 + (g00 + g11) / 18 + g10 / 36)
    out[..., :-1, :-1] += b00
    out[..., :-1, 1:] += b10
    out[..., 1:, 1:] += b11
    out[..., 1:, :-1] += b01
    return out


# ---------------------------------------------------------------------------
# D4 augmentation
# ---------------------------------------------------------------------------


def d4(batch: dict, k: int) -> dict:
    """Apply symmetry k of the square (0..7) to a batch: nodal fields [B, ..., n+1, n+1]
    and element fields [B, n, n] move with the reference coordinates; the tensor and
    mu transform as R A R^T.  k = 4 t + 2 fy + fx: flip xi (fx), flip eta (fy), then
    transpose (t).  The port basis is fixed: the network answers for the transformed
    square in its own standard basis, and the energy is the transformed problem's."""
    fx, fy, t = k & 1, (k >> 1) & 1, (k >> 2) & 1
    out = dict(batch)
    sp = lambda a: _spatial(a, fx, fy, t)
    x = sp(batch["inputs"])
    re, im = x[:, 1], x[:, 2]
    a11, a12, a22 = sp(batch["a11"]), sp(batch["a12"]), sp(batch["a22"])
    if fx ^ fy:
        im = -im
        a12 = -a12
    if t:
        re = -re
        a11, a22 = a22, a11
    x = torch.stack([x[:, 0], re, im, x[:, 3], x[:, 4]], dim=1)
    out.update(inputs=x, a11=a11, a12=a12, a22=a22, sigma=sp(batch["sigma"]),
               w=sp(batch["w"]), g=sp(batch["g"]))
    return out


@functools.lru_cache(maxsize=None)
def d4_ports(kind: str, n: int, m: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """How symmetry k acts on the port basis: S_k(q_j) = sign[j] q_perm[j] for every
    column j (a signed permutation: hats to hats, a side's mode to the image side's,
    with the reversal sign where the symmetry reverses the loop).  So the transformed
    problem's modes are ``H'[perm] = sign * S_k(H)``."""
    ps = plain_square(n, kind, m, "cpu")
    QB = ps.QB
    T = _spatial(QB, k & 1, (k >> 1) & 1, (k >> 2) & 1).reshape(QB.shape[0], -1)
    flat = QB.reshape(QB.shape[0], -1)
    perm = np.empty(QB.shape[0], int)
    sign = np.empty(QB.shape[0])
    for j in range(QB.shape[0]):
        dp = (flat - T[j]).abs().amax(1)
        dm = (flat + T[j]).abs().amax(1)
        i = int(torch.argmin(torch.minimum(dp, dm)))
        if min(float(dp[i]), float(dm[i])) > 1e-12:
            raise ValueError(f"symmetry {k} does not map port {j} onto the basis")
        perm[j], sign[j] = i, (1.0 if dp[i] <= dm[i] else -1.0)
    return perm, sign


def d4_labels(lab: dict, kind: str, m: int, k: int) -> dict:
    """The transformed problem's exact fields from the original's (see `d4_ports`)."""
    n = lab["H"].shape[-1] + 1
    perm, sign = d4_ports(kind, n, m, k)
    fx, fy, t = k & 1, (k >> 1) & 1, (k >> 2) & 1
    SH = _spatial(lab["H"], fx, fy, t)
    H = torch.empty_like(SH)
    H[:, torch.as_tensor(perm, device=SH.device)] = SH * torch.as_tensor(
        sign, dtype=SH.dtype, device=SH.device)[None, :, None, None]
    out = dict(lab)
    out["H"] = H
    out["up"] = _spatial(lab["up"], fx, fy, t)
    return out


def _spatial(a, fx, fy, t):
    if fx:
        a = a.flip(-1)
    if fy:
        a = a.flip(-2)
    if t:
        a = a.transpose(-1, -2)
    return a
