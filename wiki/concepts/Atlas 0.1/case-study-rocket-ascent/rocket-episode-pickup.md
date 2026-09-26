# Rocket ascent episode — self-contained pickup (W337–W340, re-march, re-audit)

**Type:** Core Concept — Build Brief / pickup prompt (folder: `Atlas 0.1/case-study-rocket-ascent/`)
**Status:** opened 2026-09-26, for a session that has **none** of the context the episode was built in: another person's Claude, on a clone of the two repositories. Written to be the only page it needs before starting, in the sense of [[cs7-scaling-ladder-pickup]]. Everything below is quoted from a record or marked as a choice.
**Carried out 2026-09-26 as Tier 88** ([[case-study-rocket-bc-seam-atlas-0.1]] §20). W337–W340 are closed. W340's fix differs from §5 below: the overlap average alone left the floor, which was the ghost's depth, and two-way seams now take two ghost layers. The rest of this page is kept as the brief it was.
**Related:** [[case-study-rocket-bc-seam-atlas-0.1]] (§18 Tier 86, §19 Tier 87) · [[gap-worklist]] (Tiers 86, 86-continued, 87) · [[rocket-episode-seam-audit]] · [[case-study-brake-thermal-atlas-0.1]]

---

# 1. The mission, in one paragraph

A coupled rocket-ascent episode — six gas blocks, an airframe shell, a rigid-body trajectory, marched together by the build repo's own coupler — was rebuilt at Tier 86 and audited for conservation at Tier 87 (W336). The audit found the bookkeeping exact and four **code** defects with clear fixes: **W337** the injector delivers $17.7\%$ more mass than declared; **W338** the skin radiates to the adjacent air instead of the ambient; **W339** stationary no-slip walls do work, destroying $4.7\%$ of the wall heat; **W340** plane seams between non-matching grids are handed over by point interpolation, so $3.0\%$ of the mass flow vanishes at the throat every step. **Fix those four in the build repo, pin each with a test, re-march the episode, re-run the W336 audit to show each residual now closes to its coupling lag, re-derive the records the fixes touch, and update the wiki, tests and page.** W341, W342 and W335 are the owner's modelling decisions and are **not** this task.

# 2. The two repositories, and how they meet

| | vault (research & design) | build repo (the solvers) |
|---|---|---|
| remote | `github.com/nonidino/Atlas` | `github.com/nonidino/physics-foundation-model` (private) |
| branch | `atlas-0.1` | **`atlas-0.1-windfarm`** — the one the vault pins; also `atlas-0.1`, a second line carrying the same fixes plus the corpus generator: keep the two in step |
| what lives there | `wiki/` (this page, the case study, the worklist, `index.md`, `log.md`), `atlas/` (the vault's own framework: `cases/`, `probe.py`, `compiler.py`), `scripts/` (every driver, `scripts/box/` for rented machines), `tests/`, `out/` (records) | `src/atlas/solvers/compressible2d.py` (finite-volume gas: MUSCL + minmod, HLLC, SSP-RK2), `thermostruct2d.py` (FE conduction + plane stress), `trajectory.py` (RK4 rigid body), `src/atlas/data/generate.py` (the coupler, `CoupledEpisode`), `src/atlas/config/atlas_0_1.yaml`, `tests/test_atlas_seams.py`, `tests/test_atlas_solvers.py` |

**How the vault reaches the build repo.** Both repositories have a top-level package named `atlas`, so the vault imports the build repo's `src/atlas` under the private name `atlas_build_solvers` (`atlas.cases.rocket_experts.load_solvers`, `thermal_seam.load_solvers`), from **`ATLAS_BUILD_REPO`** (the build repo's root; default `~/physics-foundation-model`). Never `import atlas.solvers...` from the vault — it is the wrong package.

**How a number names its code.** `atlas.cases.thermal_seam.build_repo_identity()` returns the build repo's commit, plus a hash of `git diff HEAD` when the tree is dirty. Tiers 86–87's records carry **`0a407b7+dirty:b6bbcec4240a5943`**: that uncommitted tree was committed on 2026-09-26 as **`adc470b`** on `atlas-0.1-windfarm`, with its tests (`tests/test_atlas_seams.py`) in `76edd8b`, and carried onto `atlas-0.1` as `7213129`. `git diff 0a407b7 adc470b | sha256sum` gives `b6bbcec4240a5943…`, the same hash, so every such record is a record of commit `adc470b`. A record made after this names a clean commit. On a rented machine with no history, `ATLAS_BUILD_REPO_IDENTITY` carries the string instead.

# 3. Orientation — read in this order, then stop reading

1. The vault's `CLAUDE.md` — conventions (bare `[[links]]`, `**[AI Inference]:**`, LaTeX, append-only log).
2. [[case-study-rocket-bc-seam-atlas-0.1]] **§18** (Tier 86: the seams rebuilt; W332, the wall that passed mass) and **§19** (Tier 87: the audit — the tables you will re-run).
3. [[gap-worklist]] — the blocks *Tier 86*, *Tier 86, continued — W334* and *Tier 87*: every open row with its definition of done.
4. [[rocket-episode-seam-audit]] — how the episode is pieced together, block by block.
5. The scripts: `scripts/w321_rocket_episode.py` (the march), `scripts/w321_episode_viz_data.py` + `scripts/w321_episode_page_template.html` (the page), `scripts/w336_episode_residuals.py` (the audit) and `scripts/w336_thermal_seam_diag.py`, `scripts/w334_*` (re-deriving records), `scripts/box/` (rented machines).
6. The build repo: `generate.py` — `CoupledEpisode.step_macro`, `_wire_engine`, `_wire_external`, `_remap`, `_cell_average`, `_as_gas`, `_wall_flux`, `_shell_inner_loads`; `compressible2d.py` — `_fill_side`, `residual`, `_inviscid_residual`, `_viscous_residual`, `_isothermal_wall_conduction`; `thermostruct2d.py` — `step_thermal`, `_robin`.

# 4. The episode, as it stands

Corner case 0 of the build repo's sweep, `underexpanded_max`: $p_c = 8$ MPa, $T_c = 3400$ K, $\gamma = 1.22$, $R = 361$ J/(kg K), launched at $25$ km and Mach $3.5$, $\alpha = 0$. Coarsened $4\times$ per dimension (blocks rebuilt at that resolution), $\Delta t_{\text{macro}} = 5$ ms against the configuration's $50$ ms, 29 steps ($0.145$ s), $647\,637$ CFL sub-steps. Records in `out/w321/` (`episode.json`, `episode.npz` — every step's fields, loads and rigid state — and the launch diff); the page is `out/w321/rocket-episode.html`, published privately by the owner at `https://claude.ai/artifact/NhQYULFQtE5HrFCJWFQ74K` (another account cannot update that URL unless the owner grants edit access from the page's Share menu; otherwise publish a new artifact from the rebuilt HTML).

**What the audit measured (W336, `out/w336/`), per 5 ms step, late in the run:**

| | reading | cause |
|---|---|---|
| every block's content vs its boundary inflows | $1.2\times10^{-13}$ | exact |
| mass through walls | $5\times10^{-17}$ of $\dot m\,\Delta t$ | exact (with W332 reverted: $5.2\%$) |
| shell energy | $1.1\times10^{-11}$ of stored | exact |
| re-run vs `out/w321` | bitwise in $y$, $\theta$, $v_y$, $m$, all 29 steps | exact, across machines and BLAS |
| injector | $1.1772\,\dot m^{\ast}$ | **W337** |
| throat seam b\|e | $-3.0\%$ of $\dot m$, same at 5 and 2.5 ms, halves with the grid | **W340** |
| engine thermal seam | gas loses $626.2$, shell gains $597.3$ J/m per side: conduction $596.6$ matches to $0.13\%$, $29.6$ is wall work | **W339** |
| outer radiation | $0.95$ J/m per step per panel against the air, not the ambient | **W338** |
| d\|f, g\|f | $-37\%$, $+23\%$ | W341 (not this task) |
| every wall flux | doubles per grid halving; airframe heating has the wrong sign at coarsen 4 | W342 (not this task) |
| nozzle thrust given its inflow | $0.989$ of ideal planar theory | right to about 1% |

# 5. The four fixes

**W337 — the injector.** `_fill_side`'s `inlet_massflow` builds a ghost ($\rho u = \dot m''$, $T = T_{\text{inject}} = 900$ K, the interior's pressure) and the face flux is the Riemann solution between that ghost and the burning gas in `a`'s first cells — $1.177\,\dot m''$. *Choice to make:* (a) prescribe the face's inviscid flux directly in `_inviscid_residual` for an `inlet_massflow` side — mass $\dot m''$; normal momentum $\dot m'' u + p$ with $u = \dot m''/\rho$, $\rho = p/(R T_{\text{inject}})$ and $p$ the interior's; no transverse momentum; energy $\dot m''(c_p T_{\text{inject}} + u^2/2)$; progress variable $0$ — the same pattern W301 used for the wall's conduction; or (b) iterate the ghost's velocity until the face flux meets the target. (a) is exact by construction. **Done when** the audit's injector reads $1.000000 \pm 10^{-6}$, a build-repo test with the reaction on asserts it, and `test_nozzle_centreline_mach_matches_area_mach` (no reaction) still passes.

**W338 — radiation.** `ThermoStruct2D.step_thermal` adds $h_{\text{rad}}$ to the outer coefficient and hands the sum to `_robin("outer", h_o, T_gas_out)`, so radiation drives the skin toward the adjacent gas. Target $T_{\text{eff}} = (h_{\text{out}} T_{\text{gas}} + h_{\text{rad}} T_\infty)/(h_{\text{out}} + h_{\text{rad}})$, or two Robin terms. **Done when** a test with $h_{\text{out}} = 0$ shows the skin exchanging with $T_\infty$, and the audit's radiation slip reads zero.

**W339 — work at a stationary wall.** `_viscous_residual` averages the viscous flux, work term $\mathbf u\cdot\boldsymbol\tau$ included, onto the wall face from the cell and its mirrored ghost; `_isothermal_wall_conduction` then replaces only the conduction (`fix = wall - avg`). At every **no-slip** face the viscous energy flux must be conduction alone: the one-sided `wall` term at isothermal stations, zero at adiabatic ones. The shear (momentum) stays. **Done when** `scripts/w336_thermal_seam_diag.py` reads zero work and the shell/gas heat ratio closes to the lag (it reads $1.00125$ against conduction now), and a build-repo test pins zero work at a no-slip face.

**W340 — the plane-seam handover.** `generate._remap` is `np.interp` by position and is used for a\|b, b\|e, d→f, d→g and the f\|g hole; e→f's core already uses `_cell_average` (integral-preserving) and shows no floor ($9.9\times10^{-5}$ at 2.5 ms, order $1.97$). Hand every plane seam over with `_cell_average` on the faces' node edges. An integral-preserving average of *states* does not make the two sides' fluxes equal, but it removes the interpolation floor. **Done when** b\|e's mass creation falls to the lag (below a\|b's, and shrinking with the coupling step), and a test shows a remap preserving a conserved quantity's face integral. W341's gas-model conversion is a separate floor that this does not remove.

# 6. The run plan

1. **Build repo.** Implement on `atlas-0.1-windfarm`, port to `atlas-0.1`; run `tests/test_atlas_seams.py` and `tests/test_atlas_solvers.py` in both; keep every earlier diagnosis as a control (a monkeypatch that restores the old behaviour and shows the old reading).
2. **Vault tests** that load the build repo: tiers 15, 16, 76–81 and 87 (`pytest tests/test_tier1[56]*.py tests/test_tier7[6-9]*.py tests/test_tier8[01]*.py tests/test_tier87*.py`). Pins will move: rewrite each to pin the fix, keep the diagnosis as its control ([[gap-worklist]]'s practice; the tier80 and tier87 files show the pattern).
3. **Register predictions** for the fixed run in the audit script *before* running it, with evaluator code that matches the prose — `tests/test_tier87_episode_residuals.py` shows how each must be able to fail and to pass.
4. **Re-march** the episode: first move `out/w321` to `out/w321_pre_w337` (never overwrite a record), then `python scripts/w321_rocket_episode.py --coarsen 4 --n-macro 30 --out out/w321`. About 21 min on a 7960X box, about 70 min on a laptop on mains.
5. **Re-run the audit** against the new record: `scripts/w336_episode_residuals.py --stage audit`, `control_wall`, `dt --dt-macro 1e-2` and `2.5e-3`, `grid_engine`/`grid_external --coarsen 4, 2, 1`, `report`. The full-resolution engine takes about 13 min per 5 ms step (6 steps): run it **last**, and never leave it on a box nobody is watching (§8).
6. **Re-derive the records the fixes touch.** W339 changes every no-slip isothermal wall's energy flux, so the vault's b–c probes (`out/w300`–`out/w314`) and the thermal seam (Tiers 15–16) move: use W334's procedure (`scripts/w334_swap_records.py`, `scripts/w334_compare_records.py`, and `scripts/box/run_w334.sh` as the job list; about 40 min on a box). W337 touches agent `a` (the episode, and `out/w310`'s a–b probe); W338 and W340 touch the episode only.
7. **The page:** `python scripts/w321_episode_viz_data.py` rebuilds `out/w321/rocket-episode.html`; update the template's notes ("The macro step is a declaration", "What these numbers can be trusted for") to the new readings.
8. **The wiki:** a Tier 88 section in the case study, the worklist rows closed with numbers, an `index.md` entry, and a `log.md` entry (append-only; take the date from the environment, not from memory). After every wiki edit: `python scripts/vault_scan.py` and `pytest tests/test_tier14_locality_and_scope.py -k vault`.

# 7. Environment

- Python 3.11 or 3.12 with **numpy 1.26.4, scipy 1.13.1, pyyaml 6.0.1, h5py 3.11.0, torch 2.7.1 (CPU)** — the versions every record was made with.
- `KMP_DUPLICATE_LIB_OK=TRUE` before any import when torch and numpy's linear algebra share a process; `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1` for the solvers (their arrays are small; one thread is faster).
- On Windows: the console is cp1252, so `print` only ASCII; OneDrive can hold a just-written file, so every persist step retries `os.replace`; Git Bash rewrites `/mnt/c` paths. The vault's longest path is 109 characters, so a clone in a deep directory can pass the 260-character limit and fail its checkout silently: clone with `git -c core.longpaths=true clone ...` there.
- **Checked from clean clones on 2026-09-26**: a shallow clone of each repository, `ATLAS_BUILD_REPO` pointed at the build clone, and `pytest tests/test_tier87_episode_residuals.py tests/test_tier80_multirate_defect.py` gives 36 of 36, with nothing from the machine that made them.
- **LaTeX in wiki pages is written with an editor tool, never inside Python string literals or shell heredocs**: `\t` in `\times` becomes a TAB and `\a` in `\ast` a BEL, silently. The vault scan catches control characters.
- Long runs write to a file (`> log 2>&1`), never a pipe to `tail`, which hides progress until exit.

# 8. Rented compute (vast.ai)

The solvers are CPU numpy; a GPU host is rented for its CPU. Use **your own** vast.ai account, and ask your owner before renting.

- Windows CLI: always `--raw` (the table output crashes on cp1252); the ssh address comes from `vastai ssh-url <id>`, not `show instance`.
- Search: `vastai search offers 'num_gpus=1 rentable=true reliability>0.98 cpu_cores_effective>=32 dph<=1.0' --raw`. W336 ran on a Threadripper 7960X (the episode at about 44 s per 5 ms step with the audit on) and W334 on an EPYC 9655; the jobs differed, so no per-core comparison between them is on record.
- Create: `vastai create instance <offer> --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime --disk 24 --ssh --direct --raw`. An interrupted `create` can still reach the API: list instances and destroy strays.
- On the box: `pip install --no-cache-dir numpy==1.26.4 scipy==1.13.1 pyyaml==6.0.1 h5py==3.11.0` and `pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu`.
- Payload: `python scripts/box/make_payload.py --out payload.tgz --root-name job --record <each out/ file the jobs read>`; `scp` it, `tar xzf` in `/root`, run the three `manifest.py check`s its docstring lists, then `BOX_ROOT=/root/job nohup bash run_<job>.sh > runner.log 2>&1 < /dev/null &`. Poll with `bash status.sh`. Download, compare `sha256sum` on the box with the local files, and only then destroy.
- **Destroy it, verify `vastai show instances --raw` is `[]`, and report the credit delta.** **Never let a session end with a box up**: on 2026-09-24 a session ended while the full-resolution engine was still running, the waits that would have collected it died with the session, and the box idled until the account's whole credit was gone — and that run's record was lost. If a job must outlive the session, give the box its own deadline and say so.
- The box reproduces the laptop: the audit re-run on Linux/OpenBLAS matched the Windows/MKL record bitwise in the flight's own components.

# 9. Traps, all measured here

- **A relative error needs a quantity for its denominator.** Gate A4 first divided each rigid-state component by itself and read $O(1)$ on $10^{-20}$ lateral round-off while the flight matched bitwise.
- **Vary the coupling step three ways** before attributing a seam residual: a lag shrinks with it, a floor (interpolation, gas model, wall work) does not.
- **A re-run carries every change made since the original**, declarations included (Tier 77's `effort_normal` explained most of a $\beta$ change W334 first credited to the wall). Put the old behaviour back on today's code as the control.
- **A defect can sign a finding**: Tier 84's registered, mechanism-predicted "knee" was the leaking wall.
- **Closing a defect breaks its tests**: rewrite each pin to the fix and keep the diagnosis as a control.
- **A prediction evaluator can be more permissive than its prose** (Tier 81's R5): test both directions.
- **A prediction that holds on a dominated quantity says nothing** (P9: $v_y$ is mostly the $1044$ m/s it started at; the velocity *gained* moved $5\%$).
- **Assert every patch replacement**: an unasserted `str.replace` no-ops silently.

# 10. What is not this task

W341 (a two-gas plume, or a declared non-conservation), W342 (a wall model, or a finer episode) and W335 (the 5 s probe base is past solidus, and its time depends on W342's $h$) are the owner's decisions. So are declaring the a–c, e–c, d–f and c–f interfaces, and any commit or push.

## See Also

[[case-study-rocket-bc-seam-atlas-0.1]] · [[gap-worklist]] · [[rocket-episode-seam-audit]] · [[cs7-scaling-ladder-pickup]]
