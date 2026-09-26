"""SHA-256 manifest of a code tree: `manifest.py make ROOT OUT.json` on the
laptop, `manifest.py check ROOT IN.json` on the box. Every .py/.yaml/.yml file,
by path relative to ROOT; __pycache__ skipped. A shipped copy that differs from
the tree the identity string was read from is refused."""
import hashlib
import json
import os
import sys


def walk(root):
    out = {}
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d != "__pycache__")
        for fn in sorted(fns):
            if fn.endswith((".py", ".yaml", ".yml")):
                p = os.path.join(dp, fn)
                rel = os.path.relpath(p, root).replace(os.sep, "/")
                with open(p, "rb") as fh:
                    out[rel] = hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()
    return out


if __name__ == "__main__":
    mode, root, path = sys.argv[1:4]
    if mode == "make":
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(walk(root), fh, indent=0, sort_keys=True)
        print("manifest: %d files" % len(walk(root)))
    else:
        want = json.load(open(path, encoding="utf-8"))
        have = walk(root)
        missing = sorted(set(want) - set(have))
        changed = sorted(k for k in want if k in have and have[k] != want[k])
        extra = sorted(set(have) - set(want))
        print("manifest check %s: %d files, %d missing, %d changed, %d extra"
              % (root, len(want), len(missing), len(changed), len(extra)))
        for k in (missing + changed)[:20]:
            print("   ", k)
        sys.exit(1 if (missing or changed) else 0)
