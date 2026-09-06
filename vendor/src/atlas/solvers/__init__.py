"""Phase-0 classical solvers: the teacher AND the judge.

Nothing here is a throwaway data generator. `trajectory.py` is imported verbatim
by Atlas's non-learned `rigid_body` expert, and every solver is reused in Phase 4
as the baseline the surrogate is graded against -- which is why M1 grades them
against closed-form solutions before any corpus exists.
"""
from __future__ import annotations

from .grid import Block, build_blocks, block_map
from .compressible2d import BC, Compressible2D, GasConfig, ReactionConfig
from .thermostruct2d import ShellMesh, SolidMaterial, ThermoStruct2D

__all__ = [
    "Block", "build_blocks", "block_map", "Compressible2D", "GasConfig",
    "ReactionConfig", "BC", "ShellMesh", "SolidMaterial", "ThermoStruct2D",
]
