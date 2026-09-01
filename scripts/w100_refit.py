"""Re-run CS-7's fit and gate over an existing `out/w100/w100.json`.

`stage_fit` and `verdict_block` are pure functions of the artifact, so the
exponents can be recomputed without re-marching anything.  That matters here for
one reason: **which rungs a series is allowed to contain is a judgement, and a
judgement in a measurement should be cheap to revise and expensive to hide.**

The revision this exists for: the rollout-defect series originally dropped a rung
only if its column had crossed the divergence band *before* the horizon the
defect is quoted at.  That is too weak. The growth is exponential well before the
band is crossed, so a column that diverges at macro-step 14 is already
contaminated at macro-step 10 -- measured, the classical column's rollout defect
at N=6 reads 57.3 while the same column with the composition layer's projection
restored reads 9.0. A rung whose column diverged **at all** inside the measured
window is dropped instead, and the series is empty rather than wrong.

    python scripts/w100_refit.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from w100_scaling_ladder import OUT, _f, stage_fit, verdict_block   # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", default=os.path.join(OUT, "w100.json"))
    args = ap.parse_args(argv)

    with open(args.artifact, encoding="utf-8") as fh:
        art = json.load(fh)
    art["fit"] = _f(stage_fit(art))
    art["gate"] = _f(verdict_block(art))
    with open(args.artifact, "w", encoding="utf-8") as fh:
        json.dump(art, fh, indent=1)
    print(f"\n  rewrote fit and gate in {args.artifact}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
