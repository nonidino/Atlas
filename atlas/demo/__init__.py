"""Live, editable, optimisable demonstration of the PoC 1a composed graph.

`python -m atlas.demo` (or `python scripts/w112_farm_demo.py`) serves a browser
UI at http://127.0.0.1:8011/. See `atlas/demo/README.md`.

Nothing in here modifies `atlas.cases.wind_farm_design`; the one class that
extends it, `engine.DemoRollout`, is pinned bitwise against its parent at the
default inflow in `tests/test_tier22_demo.py`.
"""

from .engine import DOMAINS, LIMITS, DemoConfig, DemoRollout, Engine   # noqa: F401

__all__ = ["DOMAINS", "LIMITS", "DemoConfig", "DemoRollout", "Engine"]
