"""Make `pytest` work from a bare checkout of this bundle.

The case studies resolve both external solvers through `ATLAS_BUILD_REPO` at
call time, and it defaults to a path on the machine this was developed on.
`run.py` sets it for the server; this sets it for the tests, so `pytest tests/`
works with no environment set up by hand.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("ATLAS_BUILD_REPO", os.path.join(HERE, "vendor"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, HERE)
