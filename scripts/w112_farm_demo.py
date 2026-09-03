"""W112 -- one command to start the PoC 1a live demo.

    python scripts/w112_farm_demo.py
    python scripts/w112_farm_demo.py --domain large --turbines 12

Then open http://127.0.0.1:8011/. See `atlas/demo/README.md` for the demo
script to follow in front of an audience.
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.demo.cli import main       # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
