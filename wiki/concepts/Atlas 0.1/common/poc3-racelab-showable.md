# PoC 3 — showable: the bundle carrying the column it claims to

**Tier 74, 2026-09-16.** Two of section 12's criteria, neither of them polish.
Six of six registered predictions held, and the work found **four** defects that
every test in this repository passed straight through — one of them a dependency
the demo has always needed and never declared, found by starting the page on a
machine that did not happen to have it.

Related: [[poc3-racelab-certified-screen]], [[poc3-racelab-certified-step]],
[[poc3-racelab-dashboard]], [[poc3-racelab-bundle]], [[gap-worklist]].

---

## 1. What was actually wrong

**The bundle's file lists stopped at Tier 57.** Its last verification was commit
`5a84ec8`, and `SCRIPTS`, `TESTS`, `WIKI_PAGES` and `ARTIFACTS` named nothing
after it. So the branch a person would clone shipped the **porous column and
nothing else**: no body-fitted car, no solids, no duct, no devices, no certified
mode, no second dashboard — thirteen tiers of work that existed only on this
machine. Criterion 1 was not "nearly done"; it was describing a different demo.

**And a geometry commit cost ninety seconds to be refused.** A viewer could move
one of the six geometry sliders, press Commit, watch a progress overlay re-cut
the car, and then be told the column would not march it (W293) — having lost the
car that *was* marching.

---

## 2. Criterion 2, for the six geometry knobs

`BodyFittedColumn.commit_check` answers *would committing this land on a car that
can march?* before anything is rebuilt. `geometry_fingerprint` is a function of
the **parameters**, so the car a commit would produce can be named without
building it, and whether it has a settled field is then one `isfile`.

| state | verdict | the field it names |
|---|---|---|
| nothing pending | `nothing_pending` | — |
| the nominal car | **committable** | `bodyfitted_t12_a1f67a8e2792.npz` |
| rake moved to $0.10$ | **refused**, `no_spun_up_field` | `bodyfitted_t12_44eb4bdc0245.npz` |
| rake moved back | **committable** | `bodyfitted_t12_a1f67a8e2792.npz` |

Four checks in $\mathbf{0.014}$ s with **no composite built** — that is R1. The
first and last rows are R2's control: a check that refused everything would pass
a test that only moved a knob, so the nominal car being committable is what
makes the refusal mean something. The refused row names `44eb4bdc0245`, which is
the fingerprint **Tier 70 recorded** for the rake $= 0.10$ car — the prediction
of the new car is right, not merely different.

On the page the Commit button reads **"Commit refused — this car has no spun-up
field"**, greyed, with the reason and what would lift it beneath; the header
still reads `MARCHING THE CAR BEFORE THESE CHANGES` and the status stays
`marching`. **The car that was running keeps running.** This does not close
W293 — the walls-ramped spin-up is still a script — it moves the refusal to
where it costs nothing.

---

## 3. Criterion 1, on two operating systems

The lists now carry Tiers 58–73: **16 scripts, 16 tests, 16 pages, 18 records**,
plus the two things the body-fitted demo *reads from disk* rather than tests
against — `tier62_car_solids` and `tier63_duct_openings`, which
`BodyFittedColumn.build` and `_devices` import while building, and the settled
field it releases from. Without that field the page settles for about four
minutes before its first frame, or refuses outright.

Built: **112.4 MB** — framework 3.0, solvers 0.5, checkpoint 83.2, recorded runs
24.3. Four allowed binaries, **zero unexpected**, no trace of the unlicensed
structural checkpoint, no vendored scOT.

### Verified by cloning, and then by running the clone

| | Windows 11 / Python 3.12.7 | Ubuntu (WSL) / Python 3.10.21 |
|---|---|---|
| `run.py --check` | **OK**, both columns | **OK**, both columns, exit $0$ |
| classical column | $376$ ms/macro-step, rms vs referent $0.0$ | $665$ ms, rms $0.0$ |
| learned column | $2931$ ms, rms $0.244$ | $5963$ ms, rms $0.244$ |
| per-window one-step error | median $0.230$, max $0.619$ | median $0.230$, max $0.619$ |
| porous page's socket | JSON + PNG over `/ws` | JSON + PNG over `/ws` |
| **body-fitted page's socket** | JSON + PNG over `/ws` | JSON + PNG over `/ws` |

The two machines disagree on **speed** and agree on **every number that is not a
time** — which is what a correct port looks like.

**The clone is the test, not the build directory.** `git clone --branch
poc3-racelab-demo` into a short path, then `run.py --check` in the clone: **OK,
both columns marched**, on both machines. The clone carries 23 scripts, 24
tests, 22 pages, 23 record directories, the $15$ MB settled field, and `run.sh`
at **100755** in the index.

### And the body-fitted page, started the way a person starts it

On the Linux clone, `python run.py --column body-fitted`:

| | |
|---|---|
| first frame | after about $120$ s (the grids, the donors, the composite, the probe operator) |
| marching | step $8$, car `a1f67a8e2792`, downforce $2.9246$, drag $2.9010$ |
| flipped to **certified** over the socket | outer iterations $2$, inner cheap calls $2$, residual $\mathbf{9.660\times10^{-9}}$, **converged** |

So section 12's criterion 4 is reachable **from a clone, on the other operating
system** — not only on the machine it was built on.

---

### The one the self-test could not see

**`run.py --check` passed on Linux while the body-fitted page could not start
at all.** Started the way a person starts it, the column reached *"cutting the
solids"* and stopped: `ModuleNotFoundError: No module named 'shapely'`.

`shapely` is imported **inside** the functions that use it — `car_solids` and
`car_union` cut the drawn panels into closed solids, the duct's openings and the
device rings with plane geometry — so nothing at module level ever went looking
for it, it was never in `requirements.txt`, and it was present on the
development machine because Anaconda ships it.

**And the check I had just written gave false assurance.** `body_fitted_check`
verified the settled field, the tier scripts and the socket, and every one of
those was fine; none of them builds a composite, which is where the import
lives. This is W251's lesson in a new place, one layer up: *a self-test that
does not do what the page does will pass while the page cannot start.*

Three layers repaired, because one would have been luck:

1. `shapely==2.1.0` pinned in `requirements.txt`, with why it is not obvious;
2. it is in the launcher's `REQUIRED` list, so a missing install is reported in
   the packages section where a missing package belongs, rather than as a
   progress overlay that stops;
3. `body_fitted_check` now **imports what the column imports while building** —
   both tier scripts, `car_solids`, `car_union` and `shapely`. Actually cutting
   the solids would have been the stronger check and costs $63$ s on every
   self-test, which is not what this check is for.

---

## 4. Three more defects, each invisible to every test here

**The orphan branch would have dropped the settled field.** `commit()`
force-adds every path in `ARTIFACTS` past the `out/*` ignore rule — and the
body-fitted field is *not* in `ARTIFACTS`, because its name carries the car's
fingerprint and is resolved at build time. So `git add` would have skipped it in
silence, and a fresh clone would have had a body-fitted page with nothing to
release from, while the bundle directory on this machine had it and every test
passed. `git add` on an ignored path says nothing at all.

The repair is two-part and the second half is the one that matters: the field is
force-added, **and** `commit()` now reads `git ls-files` back and refuses if any
expected path is not tracked. Measured: `25 recorded runs tracked, including
bodyfitted_t12_a1f67a8e2792.npz`.

**The ignore template and the artifact list were two lists that had to agree.**
The template's hand-written allowlist stopped at `racelab5`; `ARTIFACTS` reached
`racelab22`. The allowlist is now **generated by the build from `ARTIFACTS`
itself** — one list, two readers.

Closing that one **broke the test that pinned it**, which is what diagnosed
defects do here: `test_tier57` asserted the hand-written allowlist was in the
*template*, and the template is exactly the copy that had to go. Rewritten
rather than deleted, diagnosis kept — it asserts the generated allowlist in the
**bundle's** `.gitignore`, including the body-fitted field by pattern, and
asserts the template does **not** carry a hand-written copy, naming the drift as
the reason. Inside a built bundle: 34 passed, 21 skipped, and the `IN_BUNDLE`
branch confirmed to run rather than skip. `test_tier74_showable.py` is carried
into the bundle as well, its builder-and-template tests skipping there with a
reason — so the bundle still holds every PoC 3 test.

**And the bundle grew a binary by being used.** Running the demo writes the probe
operator's raster cache into `out/cache/`, $6.7$ MB the build never put there —
found because the tier's scan of a *used* bundle disagreed with the build's scan
of a fresh one. It is now ignored by name. **[AI Inference]:** the general shape
is that a bundle's audit is a property of a *fresh* build, and anything that
audits a directory someone has run is auditing the run too; the two scans
disagreeing is the signal, not the noise.

---

## 5. What a clone still needs from its own machine

**The path length.** The deepest path inside the branch is $120$ characters
(`vendor/hf-cache/hub/models--camlab-ethz--Poseidon-T/snapshots/<40 hex>/model.safetensors`),
and Windows' limit is $260$ — so the directory cloned into must be shorter than
about $139$. Found by cloning into this session's scratchpad, $148$ characters
deep: `git clone` reported **"Filename too long"** and stopped **part way
through the checkout**, leaving a directory that looked plausible and was missing
its pages and its `run.sh`. `C:\poc3` leaves $133$ to spare; a synced
`OneDrive\Documents\...` path can be most of the way there on its own. The
README now carries the number rather than the advice alone.

**About a minute before the first frame**, on the body-fitted column: twelve
curvilinear grids, their donor searches, one composite pressure system, and then
the probe operator the picture is drawn through — which is cached afterwards and
is *not* shipped, because its key is a function of the composite and a cache
that cannot be verified to match would be dead weight at best.

---

## 6. What this tier did NOT do

- **W293 is not closed.** The walls-ramped spin-up is still a script, so a new
  geometry still cannot be marched — it is refused earlier and more cheaply.
- **P1–P7 are not re-run** on the body-fitted column (§12.1's third blocker), so
  criterion 5 leans on the tier records rather than on the gate.
- **W275 is untouched** — the front wing still carries the filled wedge.
- **macOS is not tested.** "Windows and macOS or Linux" is satisfied by Linux;
  nothing here ran on a Mac.
- **The branch is built locally and not pushed.** `--commit` refreshes the
  orphan branch inside the build directory; publishing is a separate act.
- Nothing was downloaded into this repository, no machine was rented, the
  unlicensed structural checkpoint was not loaded. The Linux install went into
  `~/poc3-bundle/.venv` only, with the user's approval.

---

## 7. Reproducing

```
python scripts/tier74_showable.py --out out/racelab23 \
    --stages commit_check,bundle,summary --built-bundle <the build directory>
python scripts/build_racelab_bundle.py --out <dir> --commit
python -m pytest tests/test_tier74_showable.py -p no:cacheprovider
```

Then, in a clone of the branch: `python run.py --check`, and
`python run.py --column body-fitted` for the certified mode.
