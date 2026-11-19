"""cot_dev_layouts -- write the coolant plan's development layouts as studio case files.

`atlas/studio/cot_layouts.py` draws them (a plate, Channel-tool channels, a heated device);
this writes each, at each speed tier, as a case file 0.7 (`document.dumps`) under
``out/chartop/cot/layouts/``, with a manifest of their sha256s, the cut each is judged on
and the largest piece Peclet number there.  The files open in the studio (Case, Open, or
import their text) and cut on Pieces with the channel cut.  Development layouts only:
the evaluation layouts are drawn and hashed at registration.

    python scripts/cot_dev_layouts.py [--out DIR]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from atlas.studio import cot  # noqa: E402
from atlas.studio import cot_layouts as CL  # noqa: E402
from atlas.studio import document as D  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(HERE, "out", "chartop", "cot", "layouts"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    manifest = {"what": "the coolant plan's development layouts (atlas/studio/cot_layouts.py)",
                "written": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "role": "development: Phase 2 and the dataset's checks; never evaluation",
                "files": []}
    for name in CL.NAMES:
        for tier in CL.TIERS:
            spec, geom = CL.build(name, tier)
            text = D.dumps(spec, geom)
            fn = f"{name}-{tier}.json"
            with open(os.path.join(a.out, fn), "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            cut = CL.cut(spec, name, tier)
            pe = max(p.peclet() for p in cot.problems(cut, spec))
            manifest["files"].append({
                "file": fn, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "layout": name, "tier": tier, "u_coolant": spec.physics.params["u_coolant"],
                "cut": {"cutter": "channels", "cols": CL.CUT[name]["cols"], "n": cut.n},
                "pieces": len(cut.pieces), "pe_max": round(pe, 3)})
            print(f"{fn}: {len(cut.pieces)} pieces at n {cut.n}, Pe max {pe:.1f}", flush=True)
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=1)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
