"""Make `pytest` work from a bare checkout of this bundle.

The case studies resolve the build repository's solvers through
`ATLAS_BUILD_REPO` at call time, and the checkpoint through the Hugging Face
cache; both default to paths on the machine this was developed on.  `run.py`
sets them for the server, and this sets the same ones for the tests -- SET, not
defaulted, so a machine that already points them somewhere else still tests
this bundle's own copies, and the tests never reach the network for weights
that are already in this checkout.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["ATLAS_BUILD_REPO"] = os.path.join(HERE, "vendor")
os.environ["HF_HOME"] = os.path.join(HERE, "vendor", "hf-cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, HERE)
