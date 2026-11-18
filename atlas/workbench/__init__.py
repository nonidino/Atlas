"""Atlas Workbench: build a domain decomposition by hand, run it, and compare it
with the full-domain solve.  This package is the engine: the case file (`spec.py`),
the physics families, the runner and the compile.  Its page is the studio
(`atlas/studio`):

    python -m atlas.studio --open             # http://127.0.0.1:8021/

Plan: [[outcome-c4-path-to-declarative-cases]].  The Panel page this package once
served was retired for the studio and deleted on 2026-11-18.
"""

from .spec import CaseSpec, check, example_case  # noqa: F401
