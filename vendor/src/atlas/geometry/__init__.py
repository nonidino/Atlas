"""Static geometry: contours, agent domains, and the patch layout.

Nothing here depends on field values or on model weights, so everything is
computed once and memoized (phase-1 spec 3.3 -- recomputing static geometry per
forward pass was a measured cost in the parallel Noether track).

Re-exports are LAZY (PEP 562). `solvers.grid` imports `geometry.contours`, and
`geometry.domains` derives the model's cells from `solvers.grid` so the two can
never drift; eager re-exports here would close that into an import cycle.
"""
from __future__ import annotations

_LAZY = {
    "ATLAS_01_AGENTS": "domains", "ATLAS_01_EDGES": "domains", "AgentSpec": "domains",
    "EdgeSpec": "domains", "GridSpec": "domains",
    "PatchLayout": "layout", "build_layout": "layout",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    return getattr(import_module(f".{_LAZY[name]}", __name__), name)


def __dir__():
    return sorted(__all__)
