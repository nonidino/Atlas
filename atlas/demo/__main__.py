"""`python -m atlas.demo` -- see `scripts/w112_farm_demo.py` for the flags."""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from atlas.demo.cli import main       # noqa: E402

raise SystemExit(main())
