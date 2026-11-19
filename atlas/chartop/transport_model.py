"""The CO-T network and its two-level output forms (the coolant plan's arms R and P;
[[chart-operator-stage-0b-and-1-plan]] sections 4-6).  Nothing here is trained.

**The backbone** is CO-D's U-Net (`model.UNet`, called, not edited), on the reference
grid's (n+1)^2 nodes.  Its inputs are the CO-D chart channels and the flow:

    log J - mean, Re mu, Im mu,  b_1 / s, b_2 / s,  f_hat / rms

with b = curl psi the reference-square advection (`transport.centre_velocity`, averaged
to the nodes) and s = max(1, Pe_piece) its scale; the FiLM scalars are the Fourier
number's two (`piece.fo_scalars`) and log10(1 + Pe) / log10(201).

**One network, trial and adjoint modes** (the family catalog's device for G0'): the
adjoint modes are the trial modes of the same piece with its flow reversed
(`transport`, "the adjoint"), so they are the same network's output on the inputs with
b -> -b.  Two forward passes per piece, one set of weights.

**Arm R** (the fine residual on TL0): the outputs are corrections in CO-D's bubble form,
then the coarse correction with the piece's true operator (`transport_coarse`):

    H = C(H0 + beta N),   G = C*(H0 + beta N(b -> -b)),   u_p = C_p(u_p0 + beta N_p),

H0 and u_p0 the plain square's (TL0's fine part).  The traces stay exact whatever the
weights (the bubble), and the coarse component is exact whatever the weights
(Proposition A item 1).  With N = 0 it is TL0, bit for bit.

**Arm P** (a learned coarse basis): the network also returns, at every fine node, four
corrections to the bilinear functions of the coarse cell holding it; each coarse node's
basis function is its hat plus the corrections of its four cells' functions for that
corner, times a cell bubble, so P_theta keeps the hat's support and stays full rank near
P.  The coarse correction is exact for any full-rank P (Proposition A), so the guarantee
does not depend on the weights; only how much of the error the coarse space removes does.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from . import fe
from . import model as md
from . import ports as pt
from . import principal as pr
from . import transport as tr
from . import transport_coarse as tco
from .piece import fo_scalars

CIN = 6
COUT_R = 4 * 16 + 1          # the modes' corrections and the particular field's (m = 16)
N_SCALARS = 3


def nodal_flow(psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """b at the nodes: the element-centre velocity averaged over each node's elements."""
    b1, b2 = tr.centre_velocity(psi)
    n = psi.shape[0] - 1
    out = []
    for e in (b1, b2):
        acc = np.zeros((n + 1, n + 1))
        cnt = np.zeros((n + 1, n + 1))
        for dj in (0, 1):
            for di in (0, 1):
                acc[dj:dj + n, di:di + n] += e
                cnt[dj:dj + n, di:di + n] += 1.0
        out.append(acc / cnt)
    return out[0], out[1]


def inputs(p: tr.TransportPiece, reverse: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
    """(channels (6, n+1, n+1) float32, scalars (3,), the source's scale)."""
    nd = p.nodal_fields()
    el = p.element_fields()
    logJ = np.log(nd["J"])
    f_hat = (nd["J"] / el["area"]) * p.g
    rms = float(np.sqrt(np.mean(f_hat ** 2)))
    scale = rms if rms > 0 else 1.0
    pe = p.peclet()
    b1, b2 = nodal_flow(-p.psi if reverse else p.psi)
    s = max(1.0, pe)
    ch_ = np.stack([logJ - logJ.mean(), nd["mu"].real, nd["mu"].imag, b1 / s, b2 / s,
                    f_hat / scale]).astype(np.float32)
    sc = np.concatenate([fo_scalars(p.fo), [math.log10(1.0 + pe) / math.log10(201.0)]]).astype(np.float32)
    return ch_, sc, (rms if rms > 0 else 0.0)


def network(size: str = "1M", arm: str = "R") -> torch.nn.Module:
    """The planned backbone, untrained: CO-D's U-Net with CO-T's inputs and outputs."""
    cout = COUT_R + (4 if arm == "P" else 0)
    net = md.UNet(md.WIDTHS[size], cout=cout, cin=CIN)
    return _with_scalars(net, N_SCALARS)


def _with_scalars(net, k: int):
    """CO-D's U-Net reads two FiLM scalars; CO-T passes three, the first two as CO-D's
    and the third folded into them (a fixed linear map, so the backbone is unchanged)."""
    class Wrapped(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.fold = torch.nn.Linear(k, 2, bias=False)
            with torch.no_grad():
                self.fold.weight.zero_()
                self.fold.weight[0, 0] = 1.0
                self.fold.weight[1, 1] = 1.0
                self.fold.weight[:, 2] = 0.5

        def forward(self, x, s):
            return self.inner(x, self.fold(s))
    return Wrapped(net)


def bubble(n: int) -> np.ndarray:
    t = np.arange(n + 1) / n
    XI, ETA = np.meshgrid(t, t)
    return 16.0 * XI * (1 - XI) * ETA * (1 - ETA)


class ArmR:
    """Arm R's expert for the transport host: TL0's form plus the network's bubble
    corrections, trial and adjoint from one network, then the coarse correction.  With no
    network (``net=None``) it is TL0 for CO-T exactly."""

    def __init__(self, net=None, n_c: int = 16, m: int = 16, device: str = "cpu"):
        self.net, self.n_c, self.m, self.device = net, n_c, m, device
        self.name = "arm-R" if net is not None else "TL0-cot (arm R form, no network)"
        self.envelope = None
        self.calls = 0

    def _forward(self, p: tr.TransportPiece, reverse: bool) -> np.ndarray:
        x, s, scale = inputs(p, reverse)
        with torch.no_grad():
            y = self.net(torch.from_numpy(x)[None].to(self.device),
                         torch.from_numpy(s)[None].to(self.device))
        y = y[0].double().cpu().numpy()                     # (cout, n+1, n+1)
        return y, scale

    def response(self, p: tr.TransportPiece, Q: np.ndarray) -> tr.TransportResponse:
        n = p.n
        loop, inner = fe.grid(n)
        t = tco.TransportCoarse(p.matrix(), n, min(self.n_c, n))
        b_I = p.load()[inner]
        X0 = pr.modes(n, Q, p.s0)
        u0 = pr.solve(n, b_I, p.s0)
        Xt, Xa, up0 = X0, X0, u0
        if self.net is not None:
            beta = bubble(n).ravel()[inner]
            k = Q.shape[1]
            yt, sc = self._forward(p, False)
            ya, _ = self._forward(p, True)
            Nt = yt[:k].reshape(k, -1)[:, inner].T
            Na = ya[:k].reshape(k, -1)[:, inner].T
            Np = yt[k].ravel()[inner] * sc
            Xt = X0 + beta[:, None] * Nt
            Xa = X0 + beta[:, None] * Na
            up0 = u0 + beta * Np
        H = t.correct(Xt, t.trial_data(Q))
        G = t.correct_adjoint(Xa, t.adjoint_data(Q))
        up = t.correct(up0, b_I)
        self.calls += 1
        return tr.TransportResponse(H, G, up)

    def respond(self, problems, pb_of) -> list:
        return [self.response(p, pb_of(p.n).Q) for p in problems]


def forward_seconds(n: int = 128, size: str = "1M", repeats: int = 5, threads: int | None = None) -> dict:
    """The planned network's forward pass on one piece at n, untrained, on this CPU: two
    passes (trial and adjoint), the median of ``repeats``; for the oracle's price."""
    import time
    if threads:
        torch.set_num_threads(int(threads))
    net = network(size).eval()
    x = torch.zeros(1, CIN, n + 1, n + 1)
    s = torch.zeros(1, N_SCALARS)
    with torch.no_grad():
        net(x, s)
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            net(x, s)
            net(x, s)
            ts.append(time.perf_counter() - t0)
    return {"n": n, "size": size, "parameters": md.parameters(net), "seconds_two_passes": float(np.median(ts)),
            "threads": torch.get_num_threads()}


__all__ = ["ArmR", "CIN", "COUT_R", "bubble", "forward_seconds", "inputs", "network", "nodal_flow"]
