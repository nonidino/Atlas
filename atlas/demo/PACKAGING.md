# Running PoC 1a on a machine that has neither repository

The demo needs two things that live in two different private repositories: the
Atlas framework (this one) and the fluid expert `reference.WindowNS`
(`nonidino/physics-foundation-model`, `src/atlas/cases/windfarm/`). On a fresh
machine — a colleague's laptop, a rented GPU box, a Mac — neither is present.

**The `poc1-windfarm-demo` branch of this repository is a self-contained copy of
everything the PoC needs**, laid out so that one command runs it.

```bash
git clone --branch poc1-windfarm-demo --single-branch https://github.com/nonidino/Atlas.git poc1
cd poc1
./run.sh                # Windows: run.cmd
```

macOS, Linux and Windows. Python 3.10+. No GPU required. The launcher builds a
`.venv` beside itself, installs six packages into it, runs a self-test, and
starts the server; nothing is installed outside that folder. `./run.sh --check`
runs only the self-test — it loads the expert, marches four composed macro-steps,
and prints what they cost, which answers "does this work here" in about a minute.

## How the copy resolves the expert without patching anything

`atlas/cases/window_ns.py` finds the expert at

```
os.path.join(os.environ["ATLAS_BUILD_REPO"], "src", "atlas", "cases", "windfarm")
```

so the bundle vendors it at `vendor/src/atlas/cases/windfarm/` and points
`ATLAS_BUILD_REPO` at `./vendor` from `run.py` and `conftest.py`. **No source
file is modified for the bundle.** That is deliberate and it is what earns the
copy its credibility: `tests/test_tier21_wind_farm_design.py` and
`tests/test_tier22_demo.py` on that branch are the same files, character for
character, as the ones here, so passing them there means the same thing as
passing them here.

The two repositories both have a top-level package called `atlas`, and they do
not collide because the loader imports the windfarm subpackage under a private
module name through `importlib.util.spec_from_file_location` rather than putting
it on `sys.path`.

## What the branch contains

`atlas/` (the whole framework — the demo's import closure is 31 of its modules),
`vendor/` (the expert), `scripts/`, both test tiers, four wiki pages under
`docs/`, and the launchers. About 1.7 MB. `PROVENANCE.md` on that branch records
the exact commit of each source repository the copy was taken from.

## Rebuilding it

The branch is a **copy, not a fork**: develop here, then rebuild and force the
branch. It is an orphan branch — it shares no history with `atlas-0.1` — so it
can be replaced wholesale without touching anything else.

Both source repositories are private and so is that branch. The expert source is
first-party unpublished work; do not make the branch public without deciding,
deliberately, to publish that too.
