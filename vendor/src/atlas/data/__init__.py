"""Phase-0 corpus: sweep design, coupled generation, HDF5 schema, normalization."""
from __future__ import annotations

from .generate import CoupledEpisode, EpisodeSpec, generate_episode
from .schema import read_episode, validate_episode, write_episode
from .sweep import SweepPoint, build_sweep

__all__ = [
    "EpisodeSpec", "CoupledEpisode", "generate_episode", "build_sweep", "SweepPoint",
    "write_episode", "read_episode", "validate_episode",
]
