# vendor/: the one package of the build repository the workbench loads

The wind farm's fluid expert, `reference.WindowNS`, lives in the build
repository (`nonidino/physics-foundation-model`, `src/atlas/cases/windfarm/`).
The workbench finds it through `ATLAS_BUILD_REPO`; `run.py` at the root of
this branch SETS that variable to this folder, so a clone of `atlas-0.1` runs
the workbench with nothing else to fetch.

**Where this copy came from.** It is the copy already published on this
repository's `poc1-windfarm-demo` and `poc2-frontwing-demo` branches (identical
on both, file for file), placed here on 2026-10-03 so that the install works
from `atlas-0.1`. The workbench's own install branch, which was built but never
pushed, vendored the build repository at commit `98df350`; that commit was not
reachable from the machine that placed this copy, so whether the two differ is
not known. The self-test (`./run.sh --check`) runs every simulation type and
the learned case against this copy.

Nothing else of the build repository is here: no weights, no data.
