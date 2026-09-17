"""``python -m atlas.demo_racelab [--column porous|body-fitted] [--open] [--port N]``.

**Two columns, switchable, which is requirements 13.3's decision.**  The porous
column is fourteen rectangular windows on one lattice and is the only place
section 12's per-window criteria can be told; the body-fitted column is twelve
overset grids with the car as real walls, and is where the solids, the duct, the
devices and the **certified mode** are.  Neither replaces the other.

They are separate servers on separate ports because they are separate engines:
each builds a composite and marches it in its own worker thread, and running one
process that built both would pay for the one nobody is looking at.
"""

from __future__ import annotations

import argparse

COLUMNS = ("porous", "body-fitted")

#: Each column's default port, so the two can run side by side without either
#: being told to move.
PORTS = {"porous": 8013, "body-fitted": 8014}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="atlas.demo_racelab",
                                description="PoC 3 -- RaceLab, live")
    p.add_argument("--column", choices=COLUMNS, default="porous",
                   help="which column to serve (default: porous). "
                        "'body-fitted' is the one with the certified mode.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=None,
                   help="default: 8013 for porous, 8014 for body-fitted")
    p.add_argument("--open", action="store_true",
                   help="open a browser at the page once the server is up")
    p.add_argument("--log-level", default="warning")
    a = p.parse_args(argv)
    port = a.port if a.port is not None else PORTS[a.column]
    print("RaceLab (%s) on http://%s:%d/  (Ctrl-C to stop)"
          % (a.column, a.host, port))
    if a.column == "porous":
        from .server import run
        run(host=a.host, port=port, open_browser=a.open, log_level=a.log_level)
    else:
        from .bodyfitted_server import run
        run(host=a.host, port=port, open_browser=a.open, log_level=a.log_level)


if __name__ == "__main__":
    main()
