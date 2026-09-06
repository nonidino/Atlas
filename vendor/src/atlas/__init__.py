"""Atlas 0.1 -- agents as coordinate charts, typed edges as transition maps.

A parallel architecture track to `noether11`, NOT a modification of it: the two
share no mutable state, and nothing in this package may import from that one.
Design authority is the wiki (`wiki/concepts/Atlas 0.1/`); the executable plan is
`00-atlas-0.1-implementation-plan.md`.

Phase 1 (this scaffold) builds the plumbing with identity experts:

    from atlas.config import load_config
    from atlas.model.atlas import Atlas, AtlasState

    cfg   = load_config()
    model = Atlas(cfg)
    out   = model(state, dt=cfg.dt_macro)
"""
from __future__ import annotations

__version__ = "0.1.0.dev0"
