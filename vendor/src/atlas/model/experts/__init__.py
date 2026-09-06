"""Expert library. Phase 1 registers identities for all four families."""
from __future__ import annotations

from .base import Expert, RigidBodyExpert
from .identity import IdentityExpert, IdentityRigidBody

# family name -> constructor. Phase 2 swaps entries here; nothing else changes.
FIELD_EXPERTS = {
    "reacting_flow": IdentityExpert,
    "external_flow": IdentityExpert,
    "thermostruct": IdentityExpert,
}
RIGID_EXPERT = IdentityRigidBody

__all__ = [
    "Expert", "RigidBodyExpert", "IdentityExpert", "IdentityRigidBody",
    "FIELD_EXPERTS", "RIGID_EXPERT",
]
