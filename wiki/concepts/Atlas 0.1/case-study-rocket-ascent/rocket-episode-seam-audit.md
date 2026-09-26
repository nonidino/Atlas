# The marched rocket's seams — picture, mesh, or coupler?

**Type:** Concept page — **audit note** (folder: `Atlas 0.1/case-study-rocket-ascent/`)
**Status:** written 2026-09-23 in answer to a question about the W321 episode page: *"the geometry is horrendous and the seams are joined terribly — is it just execution, with the internal systems fine?"* **Not a tier.** No predictions were registered, and nothing was changed while it was written — §6 lists the candidates it left for the tier process. **Every one was then opened and fixed the same day, at Tier 86 (§7).** Verifying those fixes turned up two defects this note had missed. One is inside a solver: **W332**, the isothermal wall passed mass. The other is **W333**, the forebody slot counted as airframe. So §4's first bullet (*"nothing here found a defect inside one"*) was true of this note's reading and false of the solver.
**Code:** `scripts/w321_seam_audit.py` → `out/w321_prefix/seam_audit.json`; with `--figures` it also writes `seam_same_frame.png`, `seam_throat.png` and `seam_plume_edge.png` there. It reads `out/w321_prefix/episode.npz`, the run recorded in [[case-study-rocket-bc-seam-atlas-0.1]] §17.4. That run was moved from `out/w321` to `out/w321_prefix` when the fixed runs took its place; the script refuses to run against the fixed coupler. **Every number on this page is quoted from that JSON**, which a re-run reproduced value for value (94 numbers).
**Build repo:** `0a407b7`, branch `atlas-0.1-windfarm` — the pin the case study declares. §5 checks the `atlas-0.1` branch as well.
**Related:** [[case-study-rocket-bc-seam-atlas-0.1]] · [[gap-worklist]] · [[case-study-rocket-ascent-2d-atlas-0.1]] · [[edge-generation-atlas-0.1]] · [[conservation-as-constraint-atlas-0.1]] · [[port-algebra-atlas-0.1]]

---

## 0. The verdict

**Half right.** Nothing here was found inside a solver. The trajectory integrates the loads it is handed to $1.00001$ (the case study's acceleration audit), and every defect below sits between solvers. But *"the piecing together"* is three separate layers, and only the first one is presentation:

| layer | what is wrong | what it does | where the fix lives |
|---|---|---|---|
| **1. the picture** | per-agent $T/T_{\text{ref}}$ on one colour ramp; f and g labels swapped; captions wrong | every seam between agent families becomes a colour jump, and a 3400 K chamber is drawn the same colour as 222 K air | the page and `w321_episode_viz_data.py` |
| **2. the meshes** | coarsening subsamples nodes and the blank mask instead of re-deriving the geometry | **real** gaps and overlaps: at the nozzle throat and along the plume edge | build repo, `generate._coarsen_block` (and where the shell and `d` put their nodes) |
| **3. the coupler** | wall data mapped by array index, one airframe panel loaded inside out, loads integrated against index-oriented normals, an inverted body frame, conserved states passed between two gas models | **the physics at the seams is wrong**, and so are the loads on the vehicle | build repo, `generate.CoupledEpisode` |

**Why layer 3 is not a detail.** `generate.py` says so itself: *"interface behaviour only exists in a coupled run"*. The coupler is what writes the training corpus. The layer with the most defects is the one Atlas is supposed to learn from.

**[AI Inference]:** a surrogate trained on episodes from this coupler would learn the index maps of §3.1 and the gas-model factors of §3.5 as if they were physics. Nothing in the corpus would mark them as artefacts, because the solvers integrate them consistently.

---

## 1. The picture

### 1.1 One ramp, three units

`CoupledEpisode.snapshot` stores each gas agent as $\hat T = T/T_{\text{ref},\,a}$, with the reference set frozen per agent family by `normalize.refs_for_episode`:

| agents | $T_{\text{ref}}$ | at the last step, $\hat T$ | which is |
|---|---|---|---|
| `a`, `b`, `e` (engine) | $3400$ K | `b`: $[0.390,\ 0.997]$ | $1327.6$ – $3391.1$ K |
| `f` (plume) | $2067.8$ K | $[0.187,\ 1.009]$ | $387.1$ – $2086.2$ K |
| `d`, `g` (atmosphere) | $221.6$ K | `g`: $[1.000,\ 9.819]$ | $221.6$ – $2175.3$ K |

The page has one colour ramp for all of them, over $[0.0469,\ 13.94]$. So $\hat T = 1$ is **3400 K in the chamber and 221.6 K in the air, drawn as the same colour.** At a seam where the physical temperature is continuous, the drawn value still jumps by the ratio of the two references:

$$\frac{\hat T_{\text{left}}}{\hat T_{\text{right}}} = \frac{T_{\text{ref,right}}}{T_{\text{ref,left}}}: \qquad e|f:\ 1.644, \qquad f|g:\ 9.333, \qquad \{a,b,e\}|d:\ 15.35 .$$

The fix is to draw dimensional fields (K, Pa, m/s) on one physical scale. The reference sets are frozen before the run, so the conversion is exact. `seam_same_frame.png` shows the last frame both ways.

### 1.2 The page's other errors

- **f and g are swapped.** In the config, `f` is the plume ($|y| \le 0.6$, `free_jet`) and `g` is the atmosphere-wake with the plume carved out of it. The page and `LABEL` in `w321_episode_viz_data.py` call `f` "far field" and `g` "plume". The two labels are also drawn on top of each other: the page puts each label on its block's middle cell by index, and `g`'s middle cell is inside the hole, next to `f`'s.
- **The nose is on the left.** The frame puts the injector at $z = 0$ and the airframe at $z \in [-4.0,\ 0.70]$, so the nose is at $z = -4$. The airframe caption says *"Nose is at the right, where the engine is"*, and the strip plot labels $z = -4$ as "tail".
- **The skin is 8 mm thick**, not 86 mm: `geometry.shell_thickness = 0.008`.

### 1.3 What the page gets right

The page's geometry is faithful. Its node coordinates and field samples match the build repo's blocks cell for cell (checked on `g`, the block that looked wrong). **The dark triangles the page shows at the throat are real** — §2.

---

## 2. The meshes

### 2.1 The nozzle kinks, cut three different ways

The inner wall $h_{\text{in}}(z)$ has two corners: where the converging section starts ($z = 0.30$) and at the throat ($z = 0.40$). Three blocks meet along that V, and each approximates it with a polyline through its own nodes:

| block | node spacing at `coarsen = 4` | node at $z = 0.30$? | node at $z = 0.40$? |
|---|---|---|---|
| gas `b`, `e` | $10$ mm | yes | yes |
| airframe `c` | $81$ mm | no | no |
| atmosphere `d` | $124$ mm | no | no |

The shell and `d` have no node at either kink **at any resolution**: their nodes are uniform in $z$ over their own ranges, and the kinks fall at fractional node indices $212.26$ and $217.19$ of `c`'s 232, and $171.09$ and $174.32$ of `d`'s 184. Coarsening makes it much worse. `_coarsen_block` keeps every fourth node (`nodes[::k]`), so each chord that cuts a corner becomes four times longer. Rebuilding the block at the coarse resolution would not help: a uniform grid still has no node at a kink unless the kink is made a breakpoint. The result is three different chords across the same V:

| upper side, max over $z \in [0.12,\ 0.70]$ | coarse (the run) | full resolution | coarse, `atlas-0.1`'s rounded throat |
|---|---|---|---|
| gas wall ↔ shell inner face: gap | $\mathbf{14.7}$ mm | $2.7$ mm | $8.4$ mm |
| gas wall ↔ shell inner face: overlap | $2.9$ mm | $2.3$ mm | $3.4$ mm |
| shell outer face ↔ `d`: gap | $\mathbf{17.4}$ mm | $3.5$ mm | $23.1$ mm |
| shell outer face ↔ `d`: overlap | $\mathbf{12.4}$ mm | $0.7$ mm | $14.7$ mm |

The skin itself is $8$ mm thick. At the throat the configured wall is at $y = 0.0400$ and the gas block is exactly on it. The shell's inner face is at $0.0547$ and its outer face at $0.0627$ (configured: $0.0480$), and `d` starts at $0.0742$. There are two empty wedges and no solid between them. `seam_throat.png` draws it.

### 2.2 The plume hole moves

`g`'s blank mask is coarsened the same way (`blanked[::k]`), so each coarse cell takes the flag of its **first** fine cell. That breaks the mask's symmetry:

| | span |
|---|---|
| agent `f` | $[-0.600,\ 0.600]$ |
| `g`'s hole, full resolution | $[-0.59375,\ 0.59375]$ — symmetric, the docstring's $0.59375$ |
| `g`'s hole, coarsened | $[\mathbf{-0.500},\ \mathbf{0.625}]$ |

Below the plume, `g`'s live cells reach $0.10$ m into `f`'s domain along the whole $2.3$ m plume, so both blocks claim that strip. Above it, a $0.025$ m strip belongs to neither block. `f`'s lower boundary condition is read from `g`'s cell at $y = -0.5625$, which lies **inside `f`'s own domain**.

Deriving the mask from the coarse cell centres instead would restore the symmetry, giving $[-0.625,\ 0.625]$. It still would not match `f`, because $|y| = 0.6$ is not a node line of `g`'s uniform grid at either resolution: the nearest are $0.59375$ at 96 cells and $0.625$ at 24. The two blocks share a seam that only one of them has a node line on.

**Coverage, measured by raster** (every point should belong to exactly one block, apart from the declared unmodelled regions):

| window | uncovered | covered twice |
|---|---|---|
| engine, $z \in [-0.1,\ 1.0]$, $0.5$ mm raster | $33.5$ cm² | $16.6$ cm² |
| whole domain, $5$ mm raster | $0.0606$ m² | $0.2321$ m² |

The whole-domain figures are almost all the plume strip of §2.2: $0.10 \times 2.3 = 0.23$ m² and $0.025 \times 2.3 = 0.058$ m².

---

## 3. The coupler

### 3.1 The airframe reads its walls by array index

`_step_structure` gives each shell panel its inner-wall data from `b`'s wall and its outer-wall data from `d`'s, through

```python
f = lambda v: np.interp(np.linspace(0, 1, ni), np.linspace(0, 1, v.size), v)
```

This maps **fraction of array length to fraction of array length**. For uniformly spaced stations it amounts to

$$z_{\text{src}}(z) = z_{\text{src},0} + \frac{z - z_{s,0}}{z_{s,\text{end}} - z_{s,0}}\,\big(z_{\text{src,end}} - z_{\text{src},0}\big),$$

so the $0.28$ m chamber wall is stretched over the $4.7$ m airframe:

| shell station $z$ | inner data taken from `b` at $z =$ | outer data taken from `d` at $z =$ |
|---|---|---|
| $-3.959$ (nose) | $0.125$ | $-4.938$ |
| $-2.339$ | $0.220$ | $-2.982$ |
| $-0.718$ | $0.314$ | $-1.025$ |
| $0.659$ (tail) | $0.395$ | $0.638$ |

**95% of the shell's 58 stations receive chamber data from a wall they are not on**, up to $4.08$ m away. The outer side is off by up to $0.98$ m, because `d` starts a metre ahead of the nose. The 49 stations over the tank barrel, $z \in [-4, 0]$, whose interior the config declares unmodelled, carry chamber-wall pressure of $9.14$–$9.87$ MPa on their inner face.

The vault's own b–c seam does this correctly: it selects **the 14 of 232 cells whose centres lie in $[0.12,\ z_{\text{throat}}]$**, by position ([[case-study-rocket-bc-seam-atlas-0.1]] §2.1). The marched episode doesn't use that seam. It uses the build repo's coupler, unmodified.

### 3.2 The lower panel is loaded inside out

`ThermoStruct2D._face("inner")` is the $j = 0$ edge, always. `ShellMesh` has `inner_j` and `outer_j` fields meant to say which edge faces which gas, and `_face` never reads them. `grid._panel_nodes` builds the lower panel mirrored (`y = -r[:, ::-1]`), so **its $j = 0$ edge is the atmosphere side**. Chamber heat and chamber pressure go onto the lower panel's outer skin, and the atmosphere's heat, pressure and radiation onto its inner skin.

The stored run shows it three ways:

- **Temperature through the thickness.** Panel 0 is $289.70$ K on its outer skin and $288.31$ K on its inner; panel 1 is $289.71$ K on its inner and $288.31$ K on its outer. Each panel is hottest on its $j = 0$ face, which is the atmosphere side of one panel and the gas side of the other.
- **The two panels bend identically.** $\max|u_{y}^{(0)} - u_{y}^{(1)}| / \max|u_y| = 1.41\times10^{-4}$, while $\max|u_{y}^{(0)} + u_{y}^{(1)}| / \max|u_y| = 2.00$. A load that is symmetric about the axis makes the *second* ratio zero.
- **By metres.** $u_y = -4.173$ m at the nose, $+2.351$ m at mid-length and $-2.036$ m at the tail, on a body $0.216$ m wide. That is the stretched chamber pressure of §3.1, about $9$–$10$ MPa, bending a $4.7$ m, $8$ mm plate in small-strain elasticity. The shell's temperature and von Mises plots on the page show the result of §3.1 and §3.2, not an airframe.

The vault's probes never met this: `ShellAgent` is built on **the upper panel only** (`rocket_experts.py`), so ten tiers of seam probing never loaded the lower panel.

### 3.3 The aero integral inherits W309

The pressure force on the body is

$$\mathbf F = -\oint_{\partial\Omega_{\text{body}}} (p - p_\infty)\,\mathbf n_{\text{out}}\,\mathrm ds ,$$

with $\mathbf n_{\text{out}}$ the body's outward normal. `_compute_loads` uses `Block.n_j`, which [[gap-worklist]] W309 measured to be **index-oriented**: it points towards increasing $j$, not outward. On the upper `d` panel, $j$ increases away from the body, so $\mathbf n_j = \mathbf n_{\text{out}}$. The lower panel is mirrored, so there $\mathbf n_j = -\mathbf n_{\text{out}}$. The code therefore computes $\mathbf F^{(1)} - \mathbf F^{(0)}$ instead of $\mathbf F^{(1)} + \mathbf F^{(0)}$. The flow at $\alpha = 0$ is mirror-symmetric ($F^{(0)}_z = F^{(1)}_z$, $F^{(0)}_y = -F^{(1)}_y$), so

$$F_z^{\text{coded}} = 0, \qquad F_y^{\text{coded}} = 2F_y^{(1)} .$$

| | axial (drag, $+z_{\text{body}}$) | normal (side) |
|---|---|---|
| as coded | $\mathbf{0.000}$ N/m | $\mathbf{-1781.828}$ N/m |
| outward normals | $715.483$ N/m | $0.000$ N/m |

The recomputation reproduces the stored $F_{\text{aero}} = (1781.828,\ 1.88\times10^{-12})$ exactly. So the vehicle has had **no pressure drag**, and it is pushed sideways by a force that symmetry forbids. This side force accounts for the stored lateral drift: $a_x = 1781.8/5\times10^4 = 0.0356$ m/s² gives $v_x \approx 5.2$ mm/s and $x \approx 0.37$ mm by $t = 0.145$ s, against a stored $5.23$ mm/s and $0.38$ mm.

W309 already asked whether *"any graph built on `grid.Block` inherits this"*. `generate._compute_loads` is one that does.

### 3.4 W322 is the frame rotation, not the thrust sign

`_compute_loads` maps body axes to inertial ones with $\hat{\mathbf z}_{\text{body}} \mapsto (\cos\theta,\ \sin\theta)$ **for both the thrust and the aero force**. The geometry puts the nose at $-z_{\text{body}}$, and the freestream enters at $z = -5$ flowing in $+z$, so the correct map is

$$\hat{\mathbf z}_{\text{body}} \;\mapsto\; -(\cos\theta,\ \sin\theta) .$$

Under that map the thrust is $+\,\text{thrust}\,(\cos\theta, \sin\theta)$ (upward at $\theta = \pi/2$) and the drag is $-715.5\,(\cos\theta, \sin\theta)$ (downward). **Fixing only the thrust sign**, as W322 is currently framed, would leave the drag mapped upward once §3.3 is fixed, so the vehicle would be pushed forward by its own drag.

### 3.5 Conserved states cross two gas models unconverted

The `prescribed` boundary condition takes a conserved vector $(\rho,\ \rho u,\ \rho v,\ E)$ as it is. `f` runs the combustion products ($\gamma = 1.22$, $R = 361$); `g` and the freestream run air ($\gamma = 1.4$, $R = 287.05$). A conserved state built with $(\gamma, R)$ and read with $(\gamma', R')$ gives

$$p' = \frac{\gamma' - 1}{\gamma - 1}\,p, \qquad T' = \frac{\gamma' - 1}{\gamma - 1}\,\frac{R}{R'}\,T .$$

That conversion is skipped at three places:

| crossing | the state | read as |
|---|---|---|
| freestream → `f`'s inlet outside the nozzle exit, and `g` → `f`'s side walls | air | $\times 0.437$ in $T$ and $\times 0.550$ in $p$: $221.6$ K becomes $96.9$ K |
| `f` → `g`'s plume hole (`hole_state`) | plume gas | $\times 2.287$ in $T$ and $\times 1.818$ in $p$ |

The hole is also handed the plume's **transverse mean** rather than its edge state. The case study records that the hole's ghost is *"one state for the whole band"*; it does not record what that does here:

| | temperature along the plume |
|---|---|
| `f`'s edge cell, what the shear layer should see | $387$ – $964$ K |
| `f`'s transverse mean, what the coupler hands over | $869$ – $1367$ K |
| that mean, read by `g` as air | $\mathbf{1987}$ – $\mathbf{3126}$ K, at $1.82\times$ the pressure |

**The consequence is visible in `g`.** For adiabatic air entering at $M_\infty = 3.5$, the static temperature is bounded by the stagnation temperature

$$T \le T_0 = T_\infty\left(1 + \tfrac{\gamma - 1}{2}M_\infty^2\right) = 764 \text{ K}.$$

`g`'s far-field and inlet boundaries supply exactly that air, yet `g` reaches $\mathbf{2175}$ K, and is $1300$–$2200$ K across the whole wake. The excess energy can only have entered through the plume hole. `seam_plume_edge.png` shows the hole, the strips of §2.2 and the hot wake together.

**[AI Inference]:** this is probably also why Tier 84 found `g` carrying *"exactly 0.0 both ways"* on the g–f seam. A ghost set to a transverse mean cannot respond to a perturbation of `g`'s edge.

### 3.6 Declared, undeclared, one-way

- **d–g is declared, and no data crosses it into `g`.** `g`'s inlet is `freestream`; `d`'s outflow never reaches it. The recorded d–g channel is `d`'s face alone.
- **Undeclared, and known:** a–c and e–c (the nozzle wall never heats the shell), c–f, d–f. These are in the config and the `_wall_T` and `plane_d_g` docstrings, and `atlas-0.1` records three of them.
- **One wall temperature for every gas wall** — the mean over the whole shell. Declared in `_wall_T`'s docstring.
- **Recorded interfaces take one side:** c–d records only `d`'s lower panel, and b–c and g–f only the upper walls.

---

## 4. What is fine

- **The solvers' interiors, on the evidence already recorded** — the acceleration audit's $1.00001$ and the build repo's own tests. They were not re-verified here, and nothing here found a defect inside one.
- **The plane seams' geometry.** a|b nodes coincide ($20|20$); the b|e and e|f spans match.
- **The vault's per-seam machinery (Tiers 76–84).** It is built on the upper panel, with position-selected seam cells and W309's `effort_normal`, so none of §3's coupler defects apply to it.
- **The page's transcription of the geometry** (§1.3).

## 5. Would the `atlas-0.1` branch have helped?

The pinned branch `atlas-0.1-windfarm` forked from `atlas-0.1` at `50e57d8`, the Phase-0 commit, and **does not contain `a00d3af`**, where M2 settled D1–D6. So the episode marched the sharp throat, not D2's $R = 1.25\,h_t$ fillet. It also marched without the conservative e→f remap and without the recording of the undeclared d–f, a–c and c–f interfaces. (D3's characteristic far field exists on `atlas-0.1` only as an option; its default is still `prescribed`.)

It would not have fixed this audit's findings. The audit reads `atlas-0.1`'s `generate.py` and finds all six coupler patterns present: the index-stretched shell data, the default `ShellMesh` for both panels, the transverse-mean hole, subsampling coarsening, the inverted body map, and the air-$\gamma$ state handed to `f`. The rounded throat changes the shape of the kink gaps without closing them (§2.1, last column).

## 6. Candidate rows, not opened

Listed without W-numbers — opening them, and pinning each with a test, is the tier process's job:

1. **Shell wall data mapped by position, not index** (§3.1). This includes deciding what the shell's inner face sees over the tank barrel, which no agent covers, and wiring e–c.
2. **`_face` honours `inner_j`/`outer_j`**, or the lower panel is built with $j = 0$ on the gas side (§3.2).
3. **Outward normals in `_compute_loads`** — W309's rule applied to the load integral (§3.3).
4. **W322 widened to the body→inertial rotation**, thrust and aero together (§3.4).
5. **Every `prescribed` state passed as primitives and converted by the receiving block**, and the hole given the plume's edge state (§3.5).
6. **Blocks that share a seam share its node lines** (§2). The shell and `d` get breakpoints at $z = 0.30$ and $0.40$, and `g` gets a node line at $|y| = 0.6$. Coarsening then derives the blank mask from coarse geometry rather than subsampling it.
7. **The page:** dimensional fields, corrected labels and captions (§1).

**[AI Inference]:** items 1–5 change what the solvers integrate, so they change every number the W321 run produced. Items 6 and 7 change geometry and presentation. Doing 1–5 before re-marching avoids pricing a run whose inputs are known to be wrong.

---

## 7. What happened next — Tier 86

All seven candidates were opened and closed the same day. Two more rows were found while checking the fixes. The full record is [[case-study-rocket-bc-seam-atlas-0.1]] §18; the rows are on [[gap-worklist]].

| candidate | row | the fix, in one line |
|---|---|---|
| 1. shell data by position | **W323** (with W315, W311, W317) | every shell station averages the gas wall cells at its own $z$; over the tank barrel the config now *declares* the wall (adiabatic, ambient pressure) |
| 2. the inside-out panel | **W324** | `ShellMesh` takes its gas faces from the geometry (`grid.shell_faces`), and the traction sign follows the face |
| 3. outward normals in the loads | **W325** | both panels on the body's outward normal, plus wall shear $\mu u_t/\Delta n$ |
| 4. the body frame | **W322**, widened | the nose is body $-z$; thrust and drag are rotated by the same proper rotation |
| 5. states across gas models | **W326** | `_as_gas` converts at equal $p$, $T$ and $\mathbf u$; the plume hole gets `f`'s own edge rows, lower and upper separately |
| 6. shared node lines | **W327** | axial breakpoints at both kinks and at every agent's $z$ extent; blocks *rebuilt* at the coarse resolution; the hole on `f`'s node lines |
| 7. the page | **W329** | SI units on one scale per field, labels from the config |
| — | **W328** | the freestream is the relative wind from the rigid state, so $\alpha$ is applied |
| — | **W330** | records on both walls and both panels, outward, with a–c, e–c and d–f recorded |
| — | **W331** | plane remaps by position, and `d`'s outlet feeds `g`'s inlet and `f`'s outer inlet |
| — | **W301**, closed | the saturated isothermal channel: wall conduction taken one-sided from $T_w$ |
| found by the fix | **W332** | the isothermal wall's ghost made the inviscid wall flux permeable (§4 was wrong about the solver) |
| found by the fix | **W333** | `d`'s wall over the forebody slot was no-slip, and was counted as airframe |

**What §4 got wrong, stated plainly.** This note said no defect had been found inside a solver, and that was true of what it read. The fixed run then failed its two symmetry gates, and the cause was in `Compressible2D`. A wall's isothermal ghost is ~125× denser than the cell beside it. Handed to the Riemann solver, that pair is not a reflection, so the wall face passed mass at $123\times$ the cell's own normal mass flux whenever the gas moved toward it. With HLLC it also carried a mode that grew from round-off by $\sim10^5$ per 5 ms in the nozzle. The pre-fix run already had that asymmetry, at $4.6\times10^{-3}$ by its third step. The mis-mapped seams hid it. The shell never read the nozzle wall. The thrust was a scalar laid along the body axis, so a sideways component had nowhere to appear. (§3.3's side force was a different artefact, from the index normals.)
