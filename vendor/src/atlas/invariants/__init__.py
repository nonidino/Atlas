"""Invariants Atlas imposes on frozen experts -- shared by every case study.

The composition layer's job is not only to pass quantities between agents; it is
to make the *composed* system satisfy properties no individual frozen expert was
trained to satisfy. `conservation-as-constraint-atlas-0.1`'s enforce-or-measure
rule names the contract: a declared invariant is either enforced to a stated
tolerance or measured and reported as unenforced, never quietly assumed.

First member: `symmetry`, the Reynolds average that makes a non-equivariant
expert equivariant over a declared finite group. By Noether a symmetry and a
conserved quantity are two views of one object, so the momentum projection of
OP-1 belongs beside this rather than inside the wind farm -- once it has a
correction direction that is not degenerate.
"""
from .symmetry import (  # noqa: F401
    MIRROR_X, MIRROR_Y, Identity, MirrorX, MirrorY, SymmetrizedExpert,
    SymmetryGroup, SymmetryOp, symmetrize,
)
