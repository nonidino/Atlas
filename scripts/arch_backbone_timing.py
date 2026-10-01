"""arch_backbone_timing -- forward cost of the candidate backbone sizes (decision 10).

The recommended trunk ([[chart-operator-design-decisions]] decisions 6 and 10): a
U-shaped convolutional operator on the n x n reference grid, 4 levels, two 3 x 3
convolutions per level, GELU, FiLM conditioning on the scalars, odd/even reflection
padding instead of periodic, average-pool down and bilinear up-sampling, skip
connections; inputs log J, Re mu, Im mu, log kappa, a source and a state (6 channels);
outputs 4m = 64 corrections (16 port modes per side) and one particular field, each
multiplied by the bubble 16 xi (1-xi) eta (1-eta) so the traces stay exact.  Untrained
weights: only the cost is measured, not accuracy.

Three widths -- the sweep of decision 10b -- at n = 32, 64, 128, batch 1 and 16, float32,
on this cloud container's CPU (torch threads as installed).  Parameters, FLOPs counted
analytically (multiply-adds x 2 of every convolution), and wall time per forward pass
(median of 10 after 3 warm-ups, under torch.inference_mode).  The container is not the
owner's laptop: the numbers are a proxy, ratios to the measured classical local solve of
`arch_coupling_cost.py` (0.85 ms at n = 64) are the point.

Writes ``out/arch/backbone_timing.txt`` and ``.json``.

    set PYTHONIOENCODING=utf-8
    python scripts/arch_backbone_timing.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "out", "arch")

IN_CH, OUT_CH, N_SCALARS = 6, 65, 2
WIDTHS = {"small": (16, 32, 64, 128), "medium": (32, 64, 128, 256), "large": (64, 128, 256, 384)}


class FiLMBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=0)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=0)
        self.film = nn.Linear(N_SCALARS, 2 * cout)

    def forward(self, x, s):
        g, b = self.film(s).chunk(2, dim=1)
        x = F.gelu(self.c1(F.pad(x, (1, 1, 1, 1), mode="reflect")))
        x = self.c2(F.pad(x, (1, 1, 1, 1), mode="reflect"))
        return F.gelu(x * (1 + g[:, :, None, None]) + b[:, :, None, None])


class UNet(nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.enc = nn.ModuleList()
        c = IN_CH
        for w in widths:
            self.enc.append(FiLMBlock(c, w))
            c = w
        self.dec = nn.ModuleList()
        for w_skip in reversed(widths[:-1]):
            self.dec.append(FiLMBlock(c + w_skip, w_skip))
            c = w_skip
        self.head = nn.Conv2d(c, OUT_CH, 1)

    def forward(self, x, s):
        n = x.shape[-1]
        xi = (torch.arange(n, dtype=x.dtype) + 0.5) / n
        bubble = 16 * (xi * (1 - xi))[:, None] * (xi * (1 - xi))[None, :]
        skips = []
        for k, blk in enumerate(self.enc):
            x = blk(x, s)
            if k < len(self.enc) - 1:
                skips.append(x)
                x = F.avg_pool2d(x, 2)
        for blk in self.dec:
            sk = skips.pop()
            x = F.interpolate(x, size=sk.shape[-2:], mode="bilinear", align_corners=False)
            x = blk(torch.cat([x, sk], dim=1), s)
        return self.head(x) * bubble


def flops(model: UNet, n: int) -> float:
    """Multiply-adds x 2 of every convolution, from the layer shapes."""
    total = 0.0
    side = n
    for k, blk in enumerate(model.enc):
        for conv in (blk.c1, blk.c2):
            total += 2 * conv.in_channels * conv.out_channels * 9 * side * side
        if k < len(model.enc) - 1:
            side //= 2
    for blk in model.dec:
        side *= 2
        for conv in (blk.c1, blk.c2):
            total += 2 * conv.in_channels * conv.out_channels * 9 * side * side
    total += 2 * model.head.in_channels * model.head.out_channels * n * n
    return total


def time_forward(model, n, batch):
    x = torch.randn(batch, IN_CH, n, n)
    s = torch.randn(batch, N_SCALARS)
    with torch.inference_mode():
        for _ in range(3):
            model(x, s)
        ts = []
        for _ in range(10):
            t0 = time.perf_counter()
            model(x, s)
            ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main() -> None:
    torch.manual_seed(0)
    rows = []
    for name, widths in WIDTHS.items():
        model = UNet(widths).eval()
        params = sum(p.numel() for p in model.parameters())
        for n in (32, 64, 128):
            for batch in (1, 16):
                t = time_forward(model, n, batch)
                rows.append({"size": name, "widths": widths, "params": params, "n": n,
                             "batch": batch, "gflop_per_piece": flops(model, n) / 1e9,
                             "ms_per_forward": 1e3 * t, "ms_per_piece": 1e3 * t / batch})
                sys.stdout.write(f"{name} n={n} batch={batch}: {1e3 * t / batch:.2f} ms/piece\n")
                sys.stdout.flush()
    L = ["arch_backbone_timing -- forward cost of the candidate U-shaped trunks (untrained)",
         "=" * 82,
         f"torch {torch.__version__}, {torch.get_num_threads()} threads, float32, inference mode;"
         " container CPU, a proxy for the owner's laptop",
         "outputs: 64 port-mode corrections + 1 particular field; inputs: 6 fields, 2 scalars",
         "", " size    params   n   GFLOP/piece | ms/piece batch 1 | ms/piece batch 16"]
    for name in WIDTHS:
        for n in (32, 64, 128):
            r1 = next(r for r in rows if r["size"] == name and r["n"] == n and r["batch"] == 1)
            r16 = next(r for r in rows if r["size"] == name and r["n"] == n and r["batch"] == 16)
            L.append(f" {name:<7} {r1['params'] / 1e6:5.2f}M {n:>4}   {r1['gflop_per_piece']:7.3f}   |"
                     f"   {r1['ms_per_piece']:9.2f}      |   {r16['ms_per_piece']:9.2f}")
    text = "\n".join(L) + "\n"
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "backbone_timing.json"), "w", encoding="utf-8") as fh:
        json.dump({"probe": "arch_backbone_timing", "torch": torch.__version__,
                   "threads": torch.get_num_threads(), "rows": rows}, fh, indent=1)
    with open(os.path.join(OUT, "backbone_timing.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
