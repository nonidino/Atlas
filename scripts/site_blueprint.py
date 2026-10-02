"""Build the Lean blueprint's web version into the website: ``site/proofs/blueprint/``.

The blueprint's sources (``lean/blueprint/src``) are not edited. They are copied to a
temporary folder and three things are changed in the copy only:

1. the three addresses in ``web.tex`` (``\\home``, ``\\github``, ``\\dochome``) point at
   the public site and its repository (``site/data/site.json``) instead of the private
   one, as the comment in ``web.tex`` asks for at launch;
2. ``\\input{chapters/x}`` becomes ``\\input{x.tex}`` with the chapter folder on
   ``TEXINPUTS``: plasTeX finds sub-folder inputs only through ``kpsewhich``, which a
   machine without a TeX distribution does not have;
3. the macro files are copied beside ``web.tex`` under distinct names, because
   ``macros/web.tex`` and the top-level ``web.tex`` would otherwise share a name.

Then plasTeX (``leanblueprint`` 0.0.20 on plasTeX 3.1, the versions the proofs chat
used) renders it, and the output replaces ``site/proofs/blueprint/``.

Run:  python scripts/site_blueprint.py [path/to/plastex]
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "lean", "blueprint", "src")
DEST = os.path.join(ROOT, "site", "proofs", "blueprint")


def main(plastex: str = "plastex") -> int:
    with open(os.path.join(ROOT, "site", "data", "site.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src")
        shutil.copytree(SRC, src)
        web = os.path.join(src, "web.tex")
        with open(web, encoding="utf-8") as fh:
            t = fh.read()
        t = re.sub(r"\\home\{[^}]*\}", lambda m: "\\home{%s}" % cfg["pages_url"], t)
        t = re.sub(r"\\github\{[^}]*\}", lambda m: "\\github{%s}" % cfg["public_repo"], t)
        t = re.sub(r"\\dochome\{[^}]*\}", lambda m: "\\dochome{%s/tree/%s/lean}" % (cfg["public_repo"], cfg["source_branch"]), t)
        for name in ("common", "web"):
            shutil.copy(os.path.join(src, "macros", name + ".tex"), os.path.join(src, f"macros-{name}.tex"))
            t = t.replace("\\input{macros/%s}" % name, "\\input{macros-%s.tex}" % name)
        with open(web, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(t)
        content = os.path.join(src, "content.tex")
        with open(content, encoding="utf-8") as fh:
            c = fh.read()
        c = re.sub(r"\\input\{chapters/([\w-]+)\}", r"\\input{\1.tex}", c)
        with open(content, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(c)
        env = dict(os.environ, TEXINPUTS=os.pathsep.join([src, os.path.join(src, "chapters")]))
        r = subprocess.run([plastex, "-c", "plastex.cfg", "web.tex"], cwd=src, env=env,
                           capture_output=True, text=True)
        log = r.stdout + r.stderr
        problems = [ln for ln in log.splitlines() if ln.startswith(("WARNING", "ERROR"))]
        if r.returncode != 0 or problems:
            print(log[-3000:])
            print("blueprint build failed:", r.returncode, problems[:10])
            return 1
        out = os.path.join(tmp, "web")
        if os.path.isdir(DEST):
            shutil.rmtree(DEST)
        shutil.copytree(out, DEST)
    pages = [f for f in os.listdir(DEST) if f.endswith(".html")]
    print(f"blueprint: {len(pages)} pages in site/proofs/blueprint/")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
