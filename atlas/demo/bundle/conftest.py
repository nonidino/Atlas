"""Make `pytest` work from a bare checkout of this bundle.

The case studies resolve the fluid expert through `ATLAS_BUILD_REPO` at call
time, and the checkpoint through the Hugging Face cache; both default to paths
on the machine this was developed on. `run.py` sets them for the server, and
this sets the same three for the tests, so `pytest tests/` works with no
environment set up by hand -- and, in particular, without the tests reaching the
network for weights that are already in this checkout.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("ATLAS_BUILD_REPO", os.path.join(HERE, "vendor"))
os.environ.setdefault("HF_HOME", os.path.join(HERE, "vendor", "hf-cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, HERE)
