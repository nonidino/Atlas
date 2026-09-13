# PoC 3 phase 5 — the bundle

*Case study, PoC 3 phase 5. New 2026-09-13. Tier 57.*

---

# 0. The result, in one paragraph

**The `poc3-racelab-demo` branch is built, and a fresh clone of it installs, self-tests and serves a working page on Windows 11 and on Linux, with the classical column still running when the network is cut.** It took four rounds of verification, and **each of the first three passed every check it had while hiding a defect a later check found**: the car's fingerprint read the platform's C library, so Linux refused the settled field Windows had built (W250); the page could never receive a frame, because plain `uvicorn` has no WebSocket implementation and nothing but opening the page would have shown it (W251); and the check written to catch that went through the environment's proxy, so a user behind one would have been told a working page was broken (W253). Round four passes all of it: $164$ of $164$ copied files identical by blob hash on both platforms, the launcher's self-test exits `0` on both — including a real WebSocket frame from the demo's own server — the bundle's own tests run $160$ passed and $6$ skipped on both, every control on the page does what it says when clicked, and with no network the self-test exits `3`, runs the classical column, and opens no connection off the machine. **The bundle ships the drawn car as it is**, which leaves its declared envelopes after twenty macro-steps ([[poc3-racelab-drawn-car]]); the page stamps it, and that is a decision about the car, not the bundle. **The branch was built and committed locally and not pushed.**

---

# 1. What the bundle is, and how each part resolves without patching anything

`scripts/build_racelab_bundle.py`, in the shape of `w146_build_frontwing_bundle.py`, builds a $90.3$ MB directory — framework $2.6$ MB, build-repo solvers $0.5$, checkpoint $83.2$, recorded runs $3.5$ — and with `--commit` an orphan branch inside it.

| what | where in the bundle | how it resolves |
|---|---|---|
| the framework | `atlas/` | on `sys.path` from `run.py` and `conftest.py` |
| the build repository's solvers | `vendor/src/atlas/` | `window_ns.load_reference` reads `$ATLAS_BUILD_REPO/src/atlas/cases/windfarm`; `cooling_loop.load_solvers` binds `$ATLAS_BUILD_REPO/src/atlas` whole; `run.py` sets the variable to `./vendor` |
| the checkpoint | `vendor/hf-cache/hub/models--camlab-ethz--Poseidon-T/` | the adapter asks the hub for a repository id; `run.py` points `HF_HOME` at the bundle's own cache with `HF_HUB_OFFLINE=1` |
| the release state | `out/racelab5/cache/settled.npz` | `demo_racelab.engine.find_release`, matched by the car's fingerprint |
| the recorded runs | `out/racelab*/racelab*.json` | where the tests already look |
| the pages the tests quote | `wiki/concepts/Atlas 0.1/common/` | where the tests already look |

**No source file is modified for the bundle.** The builder copies and never edits, and every fix this tier made went into `atlas-0.1` first and was rebuilt from a commit. The two top-level `atlas` packages do not collide, because each loader binds the build repository's code under a private module name through `importlib.util.spec_from_file_location` — PoC 1's resolution, unchanged.

Three differences from the precedent, each for a reason:

- **The whole build-repo solver package is vendored**, not only `cases/windfarm/` as the requirements' §7 lists. The car's block and wing load the structural and conduction solvers through a loader that binds the package, and `solvers/grid.py` imports `..config`, which imports `yaml`. PoC 2's bundle copied the whole package for the same reason.
- **`run.py` and `conftest.py` SET the paths instead of defaulting them**, and the self-test asserts each solver was loaded from `vendor/`. A machine that already had `ATLAS_BUILD_REPO` or `HF_HOME` set would otherwise have run the bundle against another copy of the solvers and still passed.
- **The pages go to their own paths, not flattened into `docs/`.** PoC 1's and PoC 2's page tests could only skip inside their bundles, and a skip is not a pass.

# 2. What must not be in it, and the scans that refuse it

The builder scans the finished directory and **refuses to finish** on any of:

- **the unlicensed structural checkpoint** — a file or directory named for it, an import of it, an attribute access on it, a path into its cache. **None of these is in the bundle.** Its name in prose — the on-screen reason a structural window has no learned option, a docstring saying it was not loaded, the tests' own guards — is **counted and reported, not refused**: $14$ mentions in $12$ files. That is the rule `test_tier51_racelab_graph.py` already states — a substring test on the name flags the honesty rather than the violation.
- **any binary or weight-shaped file** other than Poseidon-T's `model.safetensors` and the settled field, matched by exact path. So no three-dimensional field can ride along: the half-car's generator ships, and none of its data does.
- **a vendored `scOT`** — it has no licence to redistribute, and the launchers install it.
- **a settled field for a different car.**

Each scan has a planted control in `tests/test_tier57_racelab_bundle.py`: an import and a cache path are caught and prose is passed; a stray `.pt` and a second `.npz` are caught while the settled field is allowed; a `scOT/` directory is caught.

**The weights are the published object.** Their SHA-256, recorded in `SOURCE_COMMITS`, is `e97428c9…40fb2`, which is the value the hub's own page shows for `model.safetensors` at revision `ec976ed5`.

# 3. W243 — a switch that cannot switch is refused and greyed out

The requirements' third constraint — *with no network, the classical column must still run* — makes a state reachable that the demo had never been in: the learned expert **absent**. Reading the page for that state found this project's recurring defect a third time. `Engine._drain` changed `assignment` first and built the rollout after, catching the failure into `notes["error"]`, and `index.html` never read that note, nor `learned_available`, nor `stack_error`. On a machine without `scOT`, **"all learned" would have tinted every window learned and counted fourteen learned windows in the ledger while the march went on classical**, and nothing would have said so.

Now the flip is **refused before the assignment moves**; a rollout that still cannot be built **puts the old assignment back**; and the page **greys out every control that needs the expert** and draws the reason beside them, while the classical preset and mode stay live. Verified in the real no-network bundle (§5.5): a click on the greyed-out preset does nothing, and the same preset sent straight down the socket is refused within $1.6$ s, drawn in red, with the ledger still fourteen classical.

# 4. The launchers

| | PoC 1 | **PoC 3** | why |
|---|---|---|---|
| torch wheel | CUDA if an NVIDIA driver answers | **CPU-only, always** | every RaceLab column runs on the CPU (W248) |
| pinning | top-level packages | top level in `requirements.txt`, **everything they pull in** in `constraints.txt` | the requirements say *every* dependency; a constraint pins a package only if something needs it, so one file serves every platform |
| PyYAML | not listed | **pinned** | the classical column's import chain reaches it (W249) |
| a WebSocket library | none | **`websockets` pinned** | without one the page never receives a frame (W251) |
| `scOT` fetch fails | the launcher exits | **warns and continues** | with no network the classical column must still run |
| self-test exit codes | 0 or 1 | **0, 1, or 3** | `3` is "the classical column ran and the page can receive it; the learned expert is not here" |
| paths | defaulted | **set**, and asserted | a pre-set variable would silently test other copies |
| bash | arrays under `set -u` | **no array expansion** | macOS ships bash 3.2, where an empty array under `set -u` is fatal |

`constraints.txt` was generated from `pip freeze` of the first verified installs — Windows with Python $3.12$ and Linux with Python $3.10$ — which agreed on every shared package except two. Each of those takes the version the $3.10$ install resolved, because that is the one that installs across the whole supported range: `networkx` $3.4.2$ (Python $3.12$ had resolved $3.6.1$, which needs $3.11$) and `setuptools` $79.0.1$. On both later platforms every installed package matched its constraint.

The self-test runs **through the demo's own engine**, not a parallel code path. It releases the car, compiles the graph, marches the all-classical column beside its referent — which must agree **bitwise**, rms $0$ — flips every window to learned through the message the page sends, marches that beside the referent, measures the per-window one-step error, and then asks the demo's own server for a frame over the page's socket.

**macOS means Apple silicon.** PyPI's own file list for torch $2.7.1$ has `macosx_11_0_arm64` wheels and no `x86_64` one, so the pin cannot install on an Intel Mac; the README says so.

# 5. Verification, and the defects only verification found

Every check was made on a **fresh `git clone` of the branch** into a directory that did not exist, because the index and not the working tree decides `run.sh`'s mode, and `.gitattributes` and not the author's editor decides its line endings. `scripts/verify_racelab_bundle.py` records each clone and `scripts/verify_racelab_offline.py` the network cut; every round's record is kept in `out/racelab_bundle/`, the failed ones under their round's name.

## 5.1 Round 1 — Linux refuses its own settled field (W250)

Windows installed and self-tested cleanly. Linux, on Python $3.10$, **installed every pinned package and then refused the settled field the bundle had been built with**: the same `car_geometry.json` fingerprinted as `993c358a` on Windows and `f591a83c` on Linux.

**Four of $52$ hashed rows differed, and all four were wheel segments.** A wheel's twelve chords come out of `cos`, `sin` and `atan2`, and the Windows C runtime and glibc round those differently in the last bit — an `alpha_deg` of $-14.999999999999996$ against $-15.000000000000012$. Tier 56's fingerprint hashed full-precision floats and so hashed the platform's C library. Now a wheel is hashed by what defines it — centre, radius, segment count, coefficient — and a plate's numbers were already the file's own or IEEE sums of them: **`a1f67a8e` on both**. The control nudges every trigonometric result by one ulp; the wheel segments move and the new fingerprint does not, and the same nudge **changes the old formula on both platforms**, so the test can fail. The re-spun settled field is bitwise the old one, and only its fingerprint moved.

Round 1 also recorded the first contact on **stock Ubuntu 24.04**: its Python $3.12$ has no `venv` module, and the launcher stopped with Python's own message and its own — `sudo apt install python3.12-venv`, or `PYTHON=` another interpreter — and left no half-built `.venv` behind to confuse the next run.

## 5.2 Round 2 — both platforms pass; the page receives nothing (W251)

With W250 fixed and `constraints.txt` in place, both clones installed, self-tested and exited `0`. **Then the page was opened**, served from the verified Windows clone, and nothing drew: no field, an empty inspector, dead controls. The server's log said why on every connection — *"No supported WebSocket library detected."* The page's frames travel over `/ws`, plain `uvicorn` carries no WebSocket implementation, and every upgrade was refused while the engine behind it marched on.

**Nothing short of opening the page could have found it.** The self-test drove the engine, not the socket. **No PoC 3 test had ever opened the server** — they drive the engine, read the page's source and assert against the records. And the development machine had `websockets` installed by `uvicorn[standard]` long ago, so the page had always worked there. Fixed: `websockets` $15.0.1$ is pinned, and the self-test asks **the demo's own server, on a real port, for a JSON frame and a PNG frame over a real WebSocket** (`demo_racelab.server.socket_check`), with a control that makes uvicorn's automatic protocol resolve to nothing and requires the check to fail.

**PoC 1's and PoC 2's pages use the same transport, and their bundles pin the same plain `uvicorn`.** Not verified, because that means installing those branches, and recorded as **W252**.

## 5.3 Round 3 — the page works, and the network cut finds the check's own defect (W253)

With `websockets` pinned, both clones passed the socket check and the page worked — every control is in §5.4. **Then the network was cut.** `scripts/verify_racelab_offline.py` uninstalls `scOT` from the clone's venv and points every proxy variable at a closed local port, which is what no network looks like to pip. `scOT`'s fetch failed as it should and the classical column marched — **and the self-test then called the page's socket broken.** `websockets` $15$ routes even a `ws://127.0.0.1` connection through the environment's proxy settings, so a user behind a corporate proxy would have been told a working page could not receive frames. The check now connects to its own loopback server directly. A test sets every proxy variable to a dead port and requires the check to pass; it failed on the unfixed code with the same `ConnectionRefusedError`.

## 5.4 Round 4 — everything passes

The two platforms ran **one after the other, on mains**, so the costs below are the only ones in this tier taken on a quiet machine.

| | Windows 11 | Linux (Ubuntu 24.04 under WSL2) |
|---|---|---|
| Python | $3.12.7$ (Anaconda's, found by `where python`) | $3.10.21$ (micromamba's, through `PYTHON=`) |
| `run.sh` in the fresh clone's index | `100755` | `100755`, executable on disk |
| line endings as checked out | `run.sh` LF, `run.cmd` CRLF | `run.sh` LF, `run.cmd` CRLF |
| copied files identical by blob hash | **$164$ of $164$** | **$164$ of $164$** |
| `--check` | **exit `0`** | **exit `0`** |
| the page's socket, from the self-test | a JSON frame and a PNG frame over `/ws` | the same |
| the bundle's own tests, in its venv | **$160$ passed, $6$ skipped** | **$160$ passed, $6$ skipped** |
| the compile | `refuse` at L7/R9, $8.0$ s | `refuse` at L7/R9, $12.5$ s |
| classical column | $483$ ms a macro-step, rms against its referent $0$ | $369$ ms, rms $0$ |
| learned column | $3360$ ms a macro-step, rms $0.227$ | $3569$ ms, rms $0.227$ |
| per-window one-step error | median $0.241$, max $0.672$ | median $0.241$, max $0.672$ |

The six skips are the builder's scan tests, which the bundle cannot run because it does not carry its builder, and say so; they run upstream. **The learned column's numbers agree between the two platforms to the precision printed**, which is agreement of the kind two C runtimes leave room for and not bitwise identity — W250 is the reminder. **The identity check was broken on purpose**: one letter of one test file changed in a throwaway clone and committed there, and the check named exactly that file with the other $163$ identical.

## 5.5 The page, opened and clicked, with the expert present and absent

The server was started from verified clones and every control was exercised against the state it should change, reading the frame back rather than trusting the click:

- **modes**: *learned* tints the window and moves the ledger to $13/1$; *certified* is refused **and the refusal is drawn** (W233); *classical* restores $14/0$ and clears the note.
- **presets**: *wake-learned* sets the six downstream windows, *upper-learned* the seven of the top row, *learned* all fourteen, *classical* none.
- **fields**: *speed*, *u*, *v*, *vorticity* and *error* each redraw the image and light their button.
- **march**: *pause* freezes the step and relabels itself *resume*; *step* advances exactly one macro-step while paused; *reset* returns to macro-step $0$ and clears the stamp.
- **measure windows** fills all fourteen one-step errors into the inspector.
- **3-D** disables all eight learned controls with the reason drawn, hides the 2-D overlay and says **NO DECLARED ENVELOPE IN 3-D**; **2-D** restores the switch, the overlay and the layout note.
- **the stamp, live**: `OUTSIDE THE MODEL` on ROTOR from macro-step $20$, MGU from $27$, FLUID from $58$ — Tier 56's W244 and W245 reproduced on screen by the bundle.
- **with `scOT` removed**: every control needing the expert is greyed out with "No module named 'scOT'" drawn beside it, and W243's refusal works as §3 describes.

The mode, preset, march and 3-D controls were clicked with the mouse; the five field buttons were clicked through the page's own click handler from script. **One observation, not a defect**: a click is applied between macro-steps, so with learned windows marching a preset takes three to four seconds to show, and the page shows nothing in the meantime — the first reading of *upper-learned* was taken inside that gap and looked like the wrong preset.

## 5.6 With no network

On the round-4 Windows clone, with `scOT` uninstalled and every proxy variable pointed at a closed port, the launcher reported that the fetch failed, **marched the classical column, carried a frame over the page's socket, and exited `3`**. Run again in-process with every socket connection off the machine refused and recorded, the self-test exited `3` with **zero connection attempts** — only its own loopback connection to the demo's server, which is allowed and recorded. **With no network, the classical column still runs.**

# 6. What this tier did NOT do, named

- **The branch is not pushed.** It exists as a commit in the local build directory; `git -C <build dir> push --force <remote> poc3-racelab-demo` publishes it, and that is a deliberate separate act.
- **Neither verification machine is a machine that has nothing.** Both platforms ran on the development laptop: Windows with its Anaconda Python and a warm pip cache, Linux under WSL2 on the same hardware with a micromamba Python. The clones came from the local branch, not from GitHub. A clean VM or a borrowed laptop is the stronger test, and it was not run.
- **macOS was not verified, and neither was Python $3.11$.** The requirements ask for Windows and one of macOS or Linux, which is met; the range in between is claimed from the pins' published wheels, not from a run.
- **The CUDA path does not exist** in this demo (W248), so nothing about a GPU was verified.
- **PoC 1's and PoC 2's bundles were not checked** for W249, W252 or W254, which are suspicions about them, recorded.
- **The car was not changed.** The bundle ships the drawn car, which leaves its envelopes after twenty macro-steps and is stamped for it.
- **The verification clones and their virtual environments were left on disk**, about $13$ GB in all: `C:\Users\Nauni\poc3w` to `poc3z` at $1.8$ GB each, `poc3v` and the build directory `poc3_build` at $0.16$ GB each, and under WSL `~/poc3b` to `~/poc3e` at $1.3$ GB each, `~/poc3a`, and `~/mm-py310` at $0.22$ GB.
- **Four rounds' self-test costs are not comparable with each other.** The laptop ran on battery from 15:07 to 16:39 by the system's own power-source events — rounds 1 to 3, the re-spun settled field and the first browser session — and rounds 2 and 3 overlapped the two platforms. Only round 4's costs, above, were taken on mains with the platforms sequential.
- **Downloads were confined to what was agreed**: pip packages into the clones' virtual environments, a Python 3.10 from conda-forge into WSL, and `scOT`'s pinned archive from GitHub. The hub's model page and PyPI's release metadata were read, not downloaded. No machine was rented, and the unlicensed structural checkpoint was not loaded.

---

## See Also

- [[poc3-racelab-drawn-car]] — Tier 56, the car this bundle ships and why it is stamped.
- [[poc3-racelab-demo]] — phase 3, the dashboard.
- [[poc3-racelab-3d]] — phase 4, the 3-D column the toggle marches.
- [[poc1a-frozen-expert-results]] — PoC 1, whose bundle this one follows.
- [[gap-worklist]] — W243 and W248 to W254.
