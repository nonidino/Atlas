"""``python -m atlas.demo_racelab [--open] [--port N] [--host H]``."""

from __future__ import annotations

import argparse

from .server import run


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="atlas.demo_racelab",
                                description="PoC 3 -- RaceLab, live")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8013)
    p.add_argument("--open", action="store_true",
                   help="open a browser at the page once the server is up")
    p.add_argument("--log-level", default="warning")
    a = p.parse_args(argv)
    print("RaceLab on http://%s:%d/  (Ctrl-C to stop)" % (a.host, a.port))
    run(host=a.host, port=a.port, open_browser=a.open, log_level=a.log_level)


if __name__ == "__main__":
    main()
