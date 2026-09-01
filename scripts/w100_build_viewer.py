"""Inline the CS-7 artifacts into the scaling-ladder viewer page.

The template is HTML with two placeholders, ``/*__FRAMES__*/null`` and
``/*__RESULT__*/null``; this substitutes the measured frames and a trimmed
`w100.json` for them and writes the page.  A build step rather than a hand-paste,
so the viewer cannot drift from the run that produced it -- and the trim is
listed explicitly below rather than being whatever happened to fit.

    python scripts/w100_frames.py --rungs 2,6,24
    python scripts/w100_build_viewer.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "out", "w100")

#: What the page reads. Everything else in the artifact -- the per-seam splits,
#: the substitution sweeps, the compile decision records -- is measurement the
#: wiki carries and the viewer does not, and inlining it would double the page
#: for nothing.
KEEP = ("geometry", "steps", "date", "march", "defect", "fixed_physics",
        "stability", "timing", "fit", "gate")


def trim(art: dict) -> dict:
    out = {k: art[k] for k in KEEP if k in art}
    # the marches carry a per-turbine table the viewer does not use
    for m in out.get("march", []):
        m.pop("turbines", None)
        m.pop("turbine_free_control", None)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template",
                    default=os.path.join(OUT, "scaling-ladder-template.html"))
    ap.add_argument("--frames", default=os.path.join(OUT, "frames.json"))
    ap.add_argument("--result", default=os.path.join(OUT, "w100.json"))
    ap.add_argument("--out", default=os.path.join(OUT, "scaling-ladder.html"))
    args = ap.parse_args(argv)

    with open(args.template, encoding="utf-8") as fh:
        html = fh.read()
    with open(args.frames, encoding="utf-8") as fh:
        frames = fh.read()
    with open(args.result, encoding="utf-8") as fh:
        art = json.load(fh)

    for token, payload in (("/*__FRAMES__*/null", frames),
                           ("/*__RESULT__*/null", json.dumps(trim(art)))):
        if token not in html:
            raise SystemExit(f"template has no {token!r} placeholder")
        html = html.replace(token, payload)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)
    size = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out} ({size:.2f} MB)")
    if size > 15.0:
        print("  !! over 15 MB -- an artifact must render under 16, so raise "
              "--snap-every or drop a rung from scripts/w100_frames.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
