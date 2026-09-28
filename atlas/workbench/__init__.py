"""Atlas Workbench: build a domain decomposition by hand, run it, and compare it
with the full-domain solve.

    python -m atlas.workbench --open          # http://127.0.0.1:8020/

Plan: [[outcome-c4-path-to-declarative-cases]].  This package is the shell --
the menus, the workflow and the case file (`spec.py`); the geometry tools are
being designed with the owner, and the runner is plan step 3.
"""

from .spec import CaseSpec, check, example_case  # noqa: F401
