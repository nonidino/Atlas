# Running PoC 1a on a machine that has nothing

The demo needs three things that live in three different places: the Atlas
framework (this repository), the fluid expert `reference.WindowNS`
(`nonidino/physics-foundation-model`, `src/atlas/cases/windfarm/`), and 83 MB of
frozen **Poseidon-T** weights (the Hugging Face hub). On a fresh machine — a
colleague's laptop, a rented GPU box, a Mac — none of them is present, and the
last one used to be a download the first run performed silently.

**The `poc1-windfarm-demo` branch of this repository is a self-contained copy of
all three**, laid out so that one command runs it.

```bash
git clone --branch poc1-windfarm-demo --single-branch https://github.com/nonidino/Atlas.git poc1
cd poc1
./run.sh                # Windows: run.cmd
```

macOS, Linux and Windows. Python 3.10-3.12. No GPU required, CUDA used if it is
there. The launcher builds a `.venv` beside itself, installs every dependency at
a **pinned** version, installs `scOT` from its pinned upstream commit, runs a
self-test that names any missing package and marches both columns, and only then
starts the server. Nothing is installed outside that folder. `./run.sh --check`
runs only the self-test, which answers "does this work here" in about a minute.

## What is bundled, and how each thing resolves without patching anything

| what | where in the bundle | how it resolves |
|---|---|---|
| the framework | `atlas/` | on `sys.path` from `run.py` |
| the fluid expert | `vendor/src/atlas/cases/windfarm/` | `window_ns.load_reference` reads `$ATLAS_BUILD_REPO/src/atlas/cases/windfarm`; `run.py` points it at `./vendor` |
| **the checkpoint** | `vendor/hf-cache/hub/models--camlab-ethz--Poseidon-T/` | `adapters.FrozenFluidExpert` asks the hub for a repository id, so the bundle hands the hub **its own cache**: `run.py` sets `HF_HOME` there and `HF_HUB_OFFLINE=1` |

**No source file is modified for the bundle.** That is deliberate and it is what
earns the copy its credibility: `tests/test_tier21_wind_farm_design.py`,
`tests/test_tier22_demo.py` and `tests/test_tier26_poseidon_design.py` on that
branch are the same files, character for character, as the ones here, so passing
them there means the same thing as passing them here.

The two repositories both have a top-level package called `atlas`, and they do
not collide because the loader imports the windfarm subpackage under a private
module name through `importlib.util.spec_from_file_location` rather than putting
it on `sys.path`.

## What is NOT bundled, and why

**`scOT`** — the model class the checkpoint is loaded through. It lives at
`github.com/camlab-ethz/poseidon`, which publishes **no licence file**, so it is
not ours to redistribute. The launchers install it from the upstream source
archive at a pinned commit, with `--no-deps` (its own pins are `torch==2.0.1`,
`transformers==4.29.2`, `wandb`, and would tear down everything else). It is the
only thing the bundle fetches from the network at install time; with no network
the classical column still runs: `./run.sh --expert reference_exposed`.

## Licence

The bundled weights are **CC-BY-NC-4.0** — research use only, no commercial use
— and the bundle carries `vendor/POSEIDON-T-LICENCE.md` saying so, with the
attribution. The demo also names the expert and its licence on screen. The
project's own consequence is recorded as **W120**: a permissively licensed donor
(Walrus, DPOT, GPhyT) is what a production version would need.

## Pinning, and why every version is fixed

The previous bundle listed six packages with no pins and left `torch` to pip,
which on Linux means a 2.5 GB CUDA wheel whether or not there is a GPU. Now:

- `requirements.txt` pins **every** package to an exact version;
- `torch` is pinned too and installed *separately*, because which wheel is right
  depends on the machine: the launcher looks for an answering `nvidia-smi` and
  installs the CUDA build if there is one and the CPU-only build (a tenth of the
  size) otherwise; on macOS there is nothing to choose;
- the Python version is checked up front (3.10-3.12, the range the pins have
  wheels for) rather than discovered halfway through an install;
- `run.py` imports every dependency by name before the server starts and names
  the missing one, because a `pip install` that failed halfway leaves an
  environment that looks built and is not.

## Rebuilding it

```bash
python scripts/w112_build_bundle.py --out out/bundle --commit
git -C out/bundle push --force <remote> poc1-windfarm-demo
```

The branch is a **copy, not a fork**: develop here, then rebuild and force the
branch. It is an orphan branch — it shares no history with `atlas-0.1` — so it
can be replaced wholesale without touching anything else. The build script reads
the checkpoint out of the local Hugging Face cache (or `$HF_HOME`), copies it
rather than linking it, and records the snapshot hash and both source commits in
`SOURCE_COMMITS`. It never pushes: publishing is a deliberate separate act,
because the branch is force-replaced.

Both source repositories are private and so is that branch. The expert source is
first-party unpublished work and the bundled weights are non-commercial; do not
make the branch public without deciding, deliberately, about both.
