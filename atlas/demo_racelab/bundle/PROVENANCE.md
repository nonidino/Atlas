# Where every file in this bundle came from

This branch is a **copy**, assembled by `scripts/build_racelab_bundle.py` in the
`Atlas` repository so the demo runs on a machine that has neither of the two
repositories it normally needs, and no downloaded weights. It is not where any
of this code is developed, and a change made here will be lost. Fix things
upstream.

## Sources, pinned

| directory | source | path there |
|---|---|---|
| `atlas/` | `nonidino/Atlas`, branch `atlas-0.1` | `atlas/` |
| `vendor/src/atlas/` | `nonidino/physics-foundation-model` | `src/atlas/` |
| `vendor/hf-cache/` | Hugging Face hub | `camlab-ethz/Poseidon-T`, the snapshot named in `SOURCE_COMMITS` |
| `out/racelab*/` | `nonidino/Atlas` | the recorded runs, same paths |
| `scripts/`, `tests/` | `nonidino/Atlas` | same paths |
| `wiki/concepts/Atlas 0.1/common/` | `nonidino/Atlas` | same path — the pages the tests quote |
| `run.py`, `run.sh`, `run.cmd`, `requirements.txt`, `constraints.txt`, `conftest.py`, `README.md`, this file | `nonidino/Atlas` | `atlas/demo_racelab/bundle/` |

`SOURCE_COMMITS` in this directory records the exact commit of each source
repository the copy was taken from, the checkpoint's snapshot and its SHA-256,
the pinned `scOT` commit, the fingerprint of the car, and the date it was
assembled.

## Why the solvers are vendored at `vendor/src/atlas/`

That looks like a strange place to put them, and it is the whole trick. The
loaders in `atlas/cases/` resolve the build repository as

```
os.environ["ATLAS_BUILD_REPO"]/src/atlas/cases/windfarm     the fluid expert
os.environ["ATLAS_BUILD_REPO"]/src/atlas                    the solver package
```

so laying the vendored copy out in that shape and pointing `ATLAS_BUILD_REPO` at
`./vendor` makes both resolve **with no change to any source file**. The whole
solver package is copied rather than only the files the car calls, because the
loader binds the package and its modules import each other (`solvers/grid.py`
imports `..config`, which reads `config/atlas_0_1.yaml`).

The two repositories both have a top-level package called `atlas`, and they are
different packages. They do not collide, because each loader imports the build
repository's code under a private module name (`atlas_windfarm_reference`,
`atlas_build_solvers`) through `importlib.util.spec_from_file_location` rather
than putting it on `sys.path`. Nothing in `atlas/` is patched for this bundle,
which is what lets the tests here be the same tests, character for character,
as the ones upstream.

## Why the checkpoint is vendored at `vendor/hf-cache/`

Same trick, one layer out. The adapter calls
`ScOT.from_pretrained("camlab-ethz/Poseidon-T")` — a hub repository id, not a
path — so the way to make that resolve from this checkout without editing the
adapter is to hand the hub **its own cache**, laid out exactly as it lays one
out itself:

```
vendor/hf-cache/hub/version.txt
vendor/hf-cache/hub/models--camlab-ethz--Poseidon-T/
  refs/main                          -> the snapshot hash
  snapshots/<hash>/config.json
  snapshots/<hash>/model.safetensors      83 MB
```

`run.py` and `conftest.py` set `HF_HOME` to that directory and
`HF_HUB_OFFLINE=1`, so the load is offline, deterministic, and cannot silently
pull a different revision. **Those weights are CC-BY-NC-4.0 — research use
only.** See `vendor/POSEIDON-T-LICENCE.md`, which is also where the attribution
lives.

## Why the settled field is here, and why it carries a fingerprint

`out/racelab5/cache/settled.npz` is the flow field the demo releases the car
from. Without it the first frame would be a uniform freestream and the transient
after it is one no recorded number was measured at. **A settled field belongs to
the car it was settled around**, and nothing used to record which: this project
twice released one car from another car's flow before the field started carrying
`racelab.geometry_fingerprint()`. The build refuses a field whose fingerprint is
not the car's, and the page says which car the field was settled around.

## What is NOT here

- **`scOT`**, the model class the checkpoint is loaded through.
  `github.com/camlab-ethz/poseidon` publishes **no licence file**, so it is not
  ours to redistribute. `run.sh` / `run.cmd` install it from the upstream source
  archive at a pinned commit, with `--no-deps` because its own pins are years
  old. Without it the classical column still runs.
- **No learned structural expert of any kind.** The only one this project has
  measured is unlicensed and kept outside every repository; the build scans for
  any file of it, path into its cache or import of it and refuses to finish if
  it finds one.
- **No three-dimensional field.** The half-car's generator
  (`atlas/cases/racelab3d.py`) is here and builds its field when the 3-D toggle is
  pressed; no 3-D data is shipped.
- The rest of `physics-foundation-model`: training data, the GPU harness, every
  other checkpoint.
- The rest of the `Atlas` wiki. The pages the tests quote are copied to their
  own paths.

## Both source repositories are private

So is this branch. The solver source is first-party unpublished work, and the
bundled weights are non-commercial. Do not make this branch public without
deciding, deliberately, about both.
