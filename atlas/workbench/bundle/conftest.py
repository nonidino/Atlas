"""Make `pytest` work from a bare checkout of this folder.

The wind farm's solver is resolved through `ATLAS_BUILD_REPO` at call time, and
by default that names a path on the machine the workbench was developed on.
`run.py` sets it for the server, and this sets the same thing for the tests:
SET, not defaulted, so a machine that already points it at another checkout
still tests this folder's own copy.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["ATLAS_BUILD_REPO"] = os.path.join(HERE, "vendor")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, HERE)
