"""python -m atlas.studio [--port 8021] [--open] [--cases-dir DIR] [--fetch-fonts]"""

import argparse
import os
import sys


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="atlas.studio",
                                description="Atlas studio: draw, choose the physics, cut "
                                            "into pieces, run, and read the guarantees")
    p.add_argument("--port", type=int, default=8021)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--open", action="store_true", help="open a browser at the page")
    p.add_argument("--cases-dir", default=None,
                   help="where cases are saved (default: out/workbench/cases)")
    p.add_argument("--fetch-fonts", action="store_true",
                   help="download Ranade from Fontshare (never committed) and exit")
    a = p.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except Exception:
            pass
    if a.fetch_fonts:
        from .fonts import fetch_ranade
        return 0 if fetch_ranade() else 3
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    from .server import serve
    serve(port=a.port, host=a.host, open_browser=a.open, cases_dir=a.cases_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
