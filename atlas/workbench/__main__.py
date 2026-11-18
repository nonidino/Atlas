"""python -m atlas.workbench: retired.

This started the Atlas Workbench's Panel page, which the studio (the integrated UI,
`atlas/studio`) replaced; its modules were deleted on 2026-11-18, once their tests
had been rewritten against the studio's protocol.  The app is
``python -m atlas.studio --open``, and the one-command install serves it.  The
engine the studio runs on (the case file, the families, the runner, the compile) is
still this package."""

from __future__ import annotations

import sys


def main(argv=None) -> int:
    print("The Atlas Workbench's Panel page is retired; the Atlas app is the studio:\n"
          "    python -m atlas.studio --open", file=sys.stderr, flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
