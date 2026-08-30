"""Inline `out/w93/frames.json` into the wake-array viewer page.

The template is HTML with one placeholder, ``/*__FRAMES__*/``; this substitutes
the measured frames for it and writes the page.  Kept as a build step rather than
a hand-paste so the viewer cannot drift from the run that produced it.

    python scripts/w93_frames.py --steps 60
    python scripts/w93_build_viewer.py --template <path> --out out/w93/wake-array.html
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_FRAMES = os.path.join(ROOT, "out", "w93", "frames.json")
DEFAULT_OUT = os.path.join(ROOT, "out", "w93", "wake-array.html")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", required=True)
    ap.add_argument("--frames", default=DEFAULT_FRAMES)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    with open(args.template, encoding="utf-8") as fh:
        html = fh.read()
    with open(args.frames, encoding="utf-8") as fh:
        frames = fh.read()

    token = "/*__FRAMES__*/null"
    if token not in html:
        raise SystemExit(f"template has no {token!r} placeholder")
    html = html.replace(token, frames)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
