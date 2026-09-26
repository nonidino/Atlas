r"""Build a payload for a rented Linux box: the vault's `atlas/` and `scripts/`,
the build repo's `src/`, any `out/` records a job reads, the build repo's
identity string, SHA-256 manifests of the code trees, and the box scripts.

The box gets no git history, so the identity is read HERE, where the history is
(`atlas.cases.thermal_seam.build_repo_identity`: the commit, plus a hash of the
uncommitted diff when there is one), and exported on the box as
``ATLAS_BUILD_REPO_IDENTITY``. The manifests are checked on the box before
anything runs (`manifest.py check`), so a shipped copy that differs from the tree
the identity was read from is refused rather than run.

    python scripts/box/make_payload.py --out payload.tgz \
        [--build ~/physics-foundation-model] [--expect-identity 0a407b7+dirty:...] \
        [--record out/w321/episode.npz ...] [--root-name job]

On the box (after scp and `tar xzf payload.tgz` in /root):

    cd /root/<root-name> && for t in vault/atlas vault/scripts build/src; do
        python manifest.py check $t <(python -c "import json,sys; print(json.dumps(
            json.load(open('manifests.json'))['<root-name>/'+sys.argv[1]]))" $t); done
    BOX_ROOT=/root/<root-name> nohup bash run_<job>.sh > runner.log 2>&1 < /dev/null &

History: W334 (2026-09-23) and W336 (2026-09-24) ran through this; their
runners are `run_w334.sh` and `run_w336.sh` beside it.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
VAULT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import manifest  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="payload.tgz")
    ap.add_argument("--build", default=os.environ.get(
        "ATLAS_BUILD_REPO", os.path.join(os.path.expanduser("~"), "physics-foundation-model")))
    ap.add_argument("--expect-identity", default=None,
                    help="refuse to build unless the build repo reads as this identity")
    ap.add_argument("--record", action="append", default=[],
                    help="a vault-relative file under out/ the job reads; repeatable")
    ap.add_argument("--root-name", default="job", help="top directory inside the tarball")
    a = ap.parse_args(argv)

    env = {**os.environ, "ATLAS_BUILD_REPO": a.build}
    env.pop("ATLAS_BUILD_REPO_IDENTITY", None)
    identity = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r); from atlas.cases import thermal_seam as T; "
         "print(T.build_repo_identity())" % VAULT],
        capture_output=True, text=True, env=env).stdout.strip()
    if not identity:
        raise SystemExit("could not read the build repo's identity from %s" % a.build)
    if a.expect_identity and identity != a.expect_identity:
        raise SystemExit("build repo reads as %s, expected %s" % (identity, a.expect_identity))
    print("identity", identity)

    R = a.root_name
    trees = {"%s/vault/atlas" % R: os.path.join(VAULT, "atlas"),
             "%s/vault/scripts" % R: os.path.join(VAULT, "scripts"),
             "%s/build/src" % R: os.path.join(a.build, "src")}
    man = {arc: manifest.walk(src) for arc, src in trees.items()}

    def skip(name):
        parts = name.replace("\\", "/").split("/")
        return "__pycache__" in parts or name.endswith((".pyc", ".pyo"))

    with tarfile.open(a.out, "w:gz") as tar:
        for arc, src in trees.items():
            tar.add(src, arcname=arc, filter=lambda ti: None if skip(ti.name) else ti)
        for rec in a.record:
            src = os.path.join(VAULT, rec)
            if not os.path.exists(src):
                raise SystemExit("record %s does not exist" % rec)
            tar.add(src, arcname="%s/vault/%s" % (R, rec.replace("\\", "/")))

        def add_bytes(arcname, data):
            ti = tarfile.TarInfo(arcname)
            ti.size = len(data)
            ti.mode = 0o755 if arcname.endswith(".sh") else 0o644
            tar.addfile(ti, io.BytesIO(data))

        add_bytes("%s/IDENTITY" % R, (identity + "\n").encode())
        add_bytes("%s/manifests.json" % R, json.dumps(man, sort_keys=True).encode())
        for fn in sorted(os.listdir(HERE)):
            if fn.endswith((".sh", ".py")):
                with open(os.path.join(HERE, fn), "rb") as fh:
                    add_bytes("%s/%s" % (R, fn), fh.read().replace(b"\r\n", b"\n"))
    print("payload %s: %.1f MB; manifests %s" % (a.out, os.path.getsize(a.out) / 1e6,
          {k: len(v) for k, v in man.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
