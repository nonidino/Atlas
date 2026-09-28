"""python -m atlas.workbench [--port 8020] [--open] [--cases-dir DIR]"""

from __future__ import annotations

import argparse


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="atlas.workbench",
                                description="Atlas Workbench: build a domain decomposition "
                                            "by hand and compare it with the full domain")
    p.add_argument("--port", type=int, default=8020)
    p.add_argument("--open", action="store_true", help="open a browser at the page")
    p.add_argument("--cases-dir", default=None,
                   help="where cases are saved (default: out/workbench/cases)")
    a = p.parse_args(argv)
    from .app import serve
    serve(port=a.port, open_browser=a.open, cases_dir=a.cases_dir)


if __name__ == "__main__":
    main()
