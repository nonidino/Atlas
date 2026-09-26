r"""Run a vault script with Tier 88's fixes put back -- the attribution control.

A re-run carries every change made since the original, not only the one under
test. Running the same job twice on the same tree, once as it is and once with
the fixes reverted (`w336_episode_residuals.revert`), separates them: whatever
the reverted run reproduces of the old record was not the fixes'.

    python scripts/box/with_revert.py all scripts/w300_rocket_bc_seam.py --stage all --json out/x.json

The build repo is loaded under its private name first and the named fixes are
reverted on its classes; the script then runs as ``__main__`` in this process,
so its own `load_solvers()` finds the patched package already in
``sys.modules``. Forked workers inherit the patch; a *spawned* worker (Windows'
only start method) would not, so this refuses to run where fork is missing.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VAULT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, VAULT)
sys.path.insert(0, os.path.join(VAULT, "scripts"))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        raise SystemExit(__doc__)
    fixes, script, rest = argv[0], argv[1], argv[2:]
    if "fork" not in mp.get_all_start_methods():
        raise SystemExit("with_revert needs the fork start method: a spawned worker "
                         "would re-import the build repo unreverted")
    import w336_episode_residuals as W
    M = W._mods()
    done = W.revert(M, fixes)
    print("with_revert: %s reverted, running %s" % (",".join(done), script), flush=True)
    sys.argv = [script] + rest
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
