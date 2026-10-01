"""The learned window stepper (demo item 1.5, design L-A of `demo-learned-case-plan`).

One call advances one window over one macro-step:

    E_theta(u - U, v, f) -> (du, dv),     the window's new velocity = old + (du, dv)

It stands exactly where the classical exposed window stands in the wind farm's
decomposed step (`families/windfarm.py`, `step_batch(u, v, dt, force=(f, 0))`):
the windows are cut from the global field, each one advanced, then blended by the
partition of unity, projected ONCE on the whole domain and the freestream band
held, all classically.  So incompressibility is enforced exactly whatever the
network returns (level 5 of the physics-encoding spectrum), and the network is
asked for nothing but the window's own advection, diffusion and forcing over a
macro-step: the 21-32 CFL-limited sub-steps the classical window takes.

**The network** is a small U-shaped convolutional operator on the window, the
rectangular-chart special case of `chart-operator-architecture` section 3: a
two-convolution block at each of ``depth + 1`` resolutions, strided convolutions
down, transposed convolutions up, skip connections across, and a 1x1 head.  Its
width sets its cost, and the width is chosen by step 0's speed budget, not by
accuracy ambition (`scripts/learned_step0.py`).

**The window's own edge is held.** The classical window keeps its edge cells at
the values it was cut with (its ``dirichlet`` transmission), so the output's
increment is zero on the edge ring: the network cannot move what the classical
window does not move.

torch is imported here at module level, and this module only where the learned
case is opened, so the workbench's other types never need torch.
"""

from __future__ import annotations

import torch
from torch import nn


class Block(nn.Module):
    """Two 3x3 convolutions, each followed by GELU."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.c2(self.act(self.c1(x))))


class WindowUNet(nn.Module):
    """``[B, 3, H, W] -> [B, 2, H, W]``: (u - U, v, f) in, the increment (du, dv) out.

    ``width`` channels at full resolution, doubling at each of ``depth`` strided
    downsamplings; ``H`` and ``W`` must be divisible by ``2**depth``.  ``edge`` cells
    of the window's border get a zero increment (the classical window holds them).
    """

    def __init__(self, width: int = 16, depth: int = 3, cin: int = 3, cout: int = 2,
                 edge: int = 1):
        super().__init__()
        self.width, self.depth, self.edge = int(width), int(depth), int(edge)
        #: the inputs are divided by ``in_scale`` and the outputs multiplied by
        #: ``out_scale``, per channel: set from the training data, saved with the weights
        self.register_buffer("in_scale", torch.ones(cin))
        self.register_buffer("out_scale", torch.ones(cout))
        w = [self.width * 2 ** k for k in range(self.depth + 1)]
        self.inc = Block(cin, w[0])
        self.down = nn.ModuleList(nn.Conv2d(w[k], w[k + 1], 2, stride=2)
                                  for k in range(self.depth))
        self.enc = nn.ModuleList(Block(w[k + 1], w[k + 1]) for k in range(self.depth))
        self.up = nn.ModuleList(nn.ConvTranspose2d(w[k + 1], w[k], 2, stride=2)
                                for k in reversed(range(self.depth)))
        self.dec = nn.ModuleList(Block(2 * w[k], w[k]) for k in reversed(range(self.depth)))
        self.head = nn.Conv2d(w[0], cout, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = x / self.in_scale.view(1, -1, 1, 1)
        skips = [self.inc(x)]
        h = skips[0]
        for down, enc in zip(self.down, self.enc):
            h = enc(self.act(down(h)))
            skips.append(h)
        h = skips.pop()
        for up, dec in zip(self.up, self.dec):
            h = dec(torch.cat([self.act(up(h)), skips.pop()], dim=1))
        out = self.head(h) * self.out_scale.view(1, -1, 1, 1)
        if self.edge:
            e = self.edge
            mask = torch.zeros_like(out[:1, :1])
            mask[..., e:-e, e:-e] = 1.0
            out = out * mask
        return out


def n_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def save_window_net(net: WindowUNet, path: str, **meta) -> None:
    """The weights, the scales and the architecture, in one file (no code in it)."""
    torch.save({"config": {"width": net.width, "depth": net.depth, "edge": net.edge},
                "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
                "meta": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                         for k, v in meta.items()}}, path)


def load_window_net(path: str) -> WindowUNet:
    """Load with ``weights_only=True``: tensors and plain values only, never code."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = WindowUNet(**ck["config"])
    net.load_state_dict(ck["state_dict"])
    return net.eval()


__all__ = ["Block", "WindowUNet", "n_parameters", "save_window_net", "load_window_net"]
