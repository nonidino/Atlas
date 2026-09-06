"""Shared init helper.

Several heads in Atlas want to start as (near) no-ops: the tokenizer's FiLM
(gamma=1, beta=0 -> conditioning earns its effect), the hierarchy's unpool
(the bottleneck starts by not overwriting local detail), the decoder's output
(the increment head starts at "nothing changes"). The obvious way to do that is
to zero-initialize the last linear of each.

**Do not zero them exactly.** A zero output weight makes the gradient of every
parameter FEEDING that layer exactly zero on step 0, which is indistinguishable
from a module that is not wired to the loss at all -- the failure the M3
acceptance test exists to catch ("no parameter receives exactly zero gradient
across a batch"). Near-zero keeps the near-identity behaviour and keeps every
parameter connected, so the test can still do its job.
"""
from __future__ import annotations

from torch import nn


def near_zero_(layer: nn.Linear, scale: float = 1e-3) -> nn.Linear:
    """Small-but-nonzero init for a head that should start as a near no-op."""
    nn.init.normal_(layer.weight, mean=0.0, std=scale)
    if layer.bias is not None:
        nn.init.normal_(layer.bias, mean=0.0, std=scale)
    return layer


__all__ = ["near_zero_"]
