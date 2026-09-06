"""Phase-1 placeholders: every expert is the identity function.

Nothing these output is physically meaningful, and that is deliberate -- Phase 1
tests the plumbing, Phase 2 replaces these with the real four
(impl-atlas-0.1-phase2-experts).

They hold no parameters. The M3 gradient test therefore checks the parameters
that DO exist (tokenizers, edge layer, hierarchy, decoders); an identity expert
contributing no gradient is correct, not a wiring bug.
"""
from __future__ import annotations

from torch import Tensor

from .base import Expert, RigidBodyExpert


class IdentityExpert(Expert):
    family = "identity"

    def forward(self, tokens: Tensor, cond: Tensor, agent: str) -> Tensor:
        return tokens


class IdentityRigidBody(RigidBodyExpert):
    family = "identity"

    def forward(self, zeta: Tensor, rigid: Tensor, dt: float) -> Tensor:
        return zeta


__all__ = ["IdentityExpert", "IdentityRigidBody"]
