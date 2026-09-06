"""Live, editable, optimisable demonstration of the PoC 2 assembled graph.

`python -m atlas.demo_frontwing` serves a browser UI at http://127.0.0.1:8012/.
See `atlas/demo_frontwing/README.md`.

Nothing in here modifies `atlas.cases.front_wing`: the engine calls into it and
subclasses nothing, and the certification panel is `compile_scheme`'s own
output grouped per seam by the same function the driver uses -- asserted equal
in `tests/test_tier30_front_wing.py`.
"""

from .engine import DemoConfig, Engine, seam_verdicts       # noqa: F401

__all__ = ["DemoConfig", "Engine", "seam_verdicts"]
