"""Expert interface.

Experts are cut by GOVERNING-EQUATION FAMILY, not by edge-type label
(expert-library-atlas-0.1, design invariant 2). "Heat", "pressure", "stress",
"fluid" describe what must be continuous across an interface; they are not each a
separate module. Atlas 0.1 has four experts:

    reacting_flow   agents a, b, e        learned, from scratch
    external_flow   agents d, f, g        learned, Poseidon bootstrap
    thermostruct    agent  c              learned, Poseidon bootstrap
    rigid_body      the bottleneck        NOT learned -- closed-form Newtonian

The rigid-body expert has a different signature on purpose: it acts on the
whole-vehicle bottleneck vector, not on a token set, and in Phase 3 it becomes a
closed-form integrator rather than a network.

In Phase 1 every entry is an identity. That is the point of the phase: if shapes,
graph, pooling, decoding and gradient flow are all correct BEFORE any expert
exists, then a Phase 2 failure is attributable to the expert rather than to a
silently mismatched interface three layers away.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class Expert(nn.Module, ABC):
    """Acts on one agent's tokens, in place of that agent's governing equations."""

    family: str = "abstract"

    @abstractmethod
    def forward(self, tokens: Tensor, cond: Tensor, agent: str) -> Tensor:
        """tokens [B, P_a, d] -> [B, P_a, d]."""


class RigidBodyExpert(nn.Module, ABC):
    """Acts on the whole-vehicle bottleneck vector."""

    family: str = "rigid_body"

    @abstractmethod
    def forward(self, zeta: Tensor, rigid: Tensor, dt: float) -> Tensor:
        """zeta [B, d], rigid [B, n_rigid] -> zeta [B, d]."""


__all__ = ["Expert", "RigidBodyExpert"]
