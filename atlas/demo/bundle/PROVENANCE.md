# Where every file in this bundle came from

This branch is a **copy**, assembled by `scripts/w112_build_bundle.py` in the
`Atlas` repository so the demo runs on a machine that has neither of the two
repositories it normally needs, and no downloaded weights. It is not where any
of this code is developed, and a change made here will be lost. Fix things
upstream.

## Sources, pinned

| directory | source | path there |
|---|---|---|
| `atlas/` | `nonidino/Atlas`, branch `atlas-0.1` | `atlas/` |
| `vendor/src/atlas/cases/windfarm/` | `nonidino/physics-foundation-model` | `src/atlas/cases/windfarm/` |
| `vendor/hf-cache/` | Hugging Face hub | `camlab-ethz/Poseidon-T`, snapshot `ec976ed5` |
| `scripts/`, `tests/` | `nonidino/Atlas` | same paths |
| `docs/` | `nonidino/Atlas` | `wiki/concepts/Atlas 0.1/common/` |
| `run.py`, `run.sh`, `run.cmd`, `requirements.txt`, `conftest.py`, this file | `nonidino/Atlas` | `atlas/demo/bundle/` |

`SOURCE_COMMITS` in this directory records the exact commit of each source
repository the copy was taken from, and the date it was assembled.

## Why the expert is vendored at `vendor/src/atlas/cases/windfarm/`

That looks like a strange place to put it, and it is the whole trick. The
loader in `atlas/cases/window_ns.py` resolves the expert as

```
os.path.join(os.environ["ATLAS_BUILD_REPO"], "src", "atlas", "cases", "windfarm")
```

so laying the vendored copy out in that shape and pointing `ATLAS_BUILD_REPO` at
`./vendor` makes it resolve **with no change to any source file**. Nothing in
`atlas/` is patched for this bundle, which is what lets the tests here be the
same tests, character for character, as the ones upstream.

The two repositories both have a top-level package called `atlas` and they are
different packages. They do not collide, because the loader imports the windfarm
subpackage under a private module name (`atlas_windfarm_reference`) through
`importlib.util.spec_from_file_location` rather than putting it on `sys.path`.

## Why the checkpoint is vendored at `vendor/hf-cache/`

Same trick, one layer out. `adapters.FrozenFluidExpert` calls
`ScOT.from_pretrained("camlab-ethz/Poseidon-T")` — a hub repository id, not a
path — so the way to make that resolve from this checkout without editing the
adapter is to hand the hub **its own cache**, laid out exactly as it lays one
out itself:

```
vendor/hf-cache/hub/models--camlab-ethz--Poseidon-T/
  refs/main                          -> ec976ed5d25883ec9db4e486ebbeeefa9e08303b
  snapshots/ec976ed5.../config.json
  snapshots/ec976ed5.../model.safetensors      83 MB
```

`run.py` sets `HF_HOME` to that directory and `HF_HUB_OFFLINE=1`, so the load is
offline, deterministic, and cannot silently pull a different revision.

**Those weights are CC-BY-NC-4.0 — research use only.** See
`vendor/POSEIDON-T-LICENCE.md`, which is also where the attribution lives.

## What is NOT here

- **`scOT`**, the model class the checkpoint is loaded through.
  `github.com/camlab-ethz/poseidon` publishes **no licence file**, so it is not
  ours to redistribute. `run.sh` / `run.cmd` install it from the upstream source
  archive at a pinned commit, with `--no-deps` because its own pins are years
  old. It is the only thing this bundle fetches from the network at install
  time.
- The rest of `physics-foundation-model`: training data, the GPU harness, every
  other checkpoint.
- The rest of the `Atlas` wiki. The pages the results actually rest on are
  copied into `docs/`.

## Both source repositories are private

So is this branch. The expert source is first-party unpublished work, and the
bundled weights are non-commercial. Do not make this branch public without
deciding, deliberately, about both.
