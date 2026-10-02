# The one learned case — a purpose-trained expert on the wind farm, with the gate written first

**Type:** Concept page — **plan and pre-registration draft** (folder: `Atlas 0.1/atlas-0.1-proposal/demo/`)
**Status:** written 2026-09-30. **Nothing is trained, generated or timed.** The gate in §5 is a draft; it becomes binding when the next chat registers it in code, **before any data is generated**. Every cost here is an estimate labelled **[AI Inference]**, to be replaced by the Step-0 measurement of §4. **Built and evaluated once on 2026-10-01 by the demo chat (§9):** the gate was registered in code before any data existed. G1–G4 and G6 pass. **G5 fails**: a classical solver on a grid twice as coarse is closer to the truth than the learned arm, in both measures at both horizons. The learned arm matches the classical decomposition's accuracy, not the full domain's. The page shows the case with all of this on its card.
**Hub:** [[00-proposal-workstreams]] · **Siblings:** [[demo-finish-plan]] · [[demo-fast-examples-plan]] · **Architecture it previews:** [[chart-operator-architecture]]
**The record it must answer to:** [[outcome-c2-learned-experts-in-the-loop]] · [[outcome-c3-learned-speed-at-scale]] · [[outcome-c5-requirements-for-dd-native-experts]] · [[learned-contribution-kill-tests]] · [[matched-shrink-and-coarse-competitor]]

---

## 0. The request, and the record it lands on

> *"Have just one case (just one case in one type of simulation, that the user can't toy around with) where learnt neural experts are used and are as accurate as classical decomp AND classical full domain, but are faster."*

**The record says this has never happened here.** Across 89 tiers, the one frozen neural operator the programme could use, Poseidon-T, couples stably and is cheaper per agent from twelve windows up. But it is biased: it inverts the wind farm's array efficiency. A $2\times$-coarse classical solver beats it at the same job by $9$–$17\times$, and in the one slot where its contribution was certified, its learned content was worth 13 classical calls against a 69-call noise floor ([[00-atlas-0.1-outcome]] §2).

**What is different now, and why success is plausible rather than hoped for.** Every learned result in the vault used a **frozen** checkpoint pretrained on periodic boxes, driven through an overwritten halo, at one native step, under a non-commercial licence ([[outcome-c5-requirements-for-dd-native-experts]] S1–S12). This case uses a **small expert trained for the slot**: on this tiling's windows, with the interface data as a declared input, at the macro-step the coupling runs at. **[AI Inference]:** most of S1–S12 are properties of *which* network was used, not of learned experts as such, and a narrow, purpose-trained network removes S2, S3, S8, S10 and S12 by construction. What remains open is S5 (the right response to interface perturbations), S7 (cheaper than the cheapest classical alternative) and S9/S11 (accuracy and a stable rollout). The gate in §5 tests exactly those.

**What this case does not show, stated before it is built.** It is one family, on rectangular windows, trained on the family of layouts it is shown on. It is a preview of the proposal ([[chart-operator-architecture]] Stage 0), not evidence for C5. The card in the page says so.

---

## 1. The decision: the wind farm

| family | for | against | verdict |
|---|---|---|---|
| **wind farm** | the only family where the classical decomposition already beats the full domain ($4.8\times$ on 12 rotors; [[showcase-gallery]] §3), so a learned expert **compounds** a win rather than fighting a loss. Nonlinear advection over a macro-step is where one learned call can replace **21–32 CFL-limited sub-steps** ([[decomposition-speed-by-rotor-count]] §1). The vault has its instruments, records and failure modes | the hardest physics in the workbench; a rollout can drift (S11) | **chosen** |
| river plume | cheap to train; easy to be accurate | the physics is **linear** in the concentration. The $K$-step map of a linear scheme on a window is a matrix the classical side can precompute exactly, so a learned version has no honest role | rejected |
| conduction, elasticity, current | steady or implicit and linear: the classical local solve with a reused factorization is a triangular solve, far cheaper than any network forward pass | a learned expert would lose on speed by construction | rejected |
| sound, heated structure, cooled block | — | linear, or split by physics; the same objection | rejected |

**The case:** the 12-rotor farm, `farm-12`: $6\times4$ windows of $128^2$ cells, 319,232 cells, where the classical decomposition on threads is $4.83\times$ faster than the full domain and its farm power is $8.0\%$ from it ([[showcase-gallery]] §2–§3). If Step 0 (§4) says the 5-rotor rung is the better stage, the case moves there, before anything is trained.

---

## 2. The arms

| arm | what runs | why it is there |
|---|---|---|
| **F** | the full domain, fine, classical | the workbench's reference |
| **Ep** | the classical decomposition, fine, on threads | the arm to beat on speed, and the accuracy the learned arm must match |
| **L** | **the learned decomposition**: every window stepped by the learned expert, on threads, and the composition layer's global projection kept classical | the case |
| **Cc** | a classical decomposition on windows **coarsened $2\times$** | **the honest competitor** (S7): a cheaper classical solver doing the same job. If it matches L's accuracy at lower cost, the learned expert has not earned its place, and the page says so |
| **T** | the full domain at $2\times$ the resolution, **run once and stored**, not timed | the truth all four are measured against, so that "as accurate as the full domain" has a meaning |

**Why a truth arm is needed.** Measured against F, the full domain has error zero by definition, and "as accurate as the full domain" means nothing. Against a finer solution, every arm has an error, F's included, and the owner's sentence becomes a comparison. **[AI Inference]:** this is the protocol of the learned-CFD literature. Kochkov et al. report their learned solver *"as accurate as baseline solvers with 8 to 10× finer resolution"* (*PNAS* 118, 2021), which is only meaningful against a resolved reference.

---

## 3. The expert

**Design L-A, recommended: a learned window stepper.** One call advances one window over one macro-step:

$$E_\theta:\ \bigl(u_W^n,\ g_{\partial W}^n,\ f_W,\ \Delta t\bigr)\ \longmapsto\ \tilde u_W^{\,n+1},$$

with $u_W$ the window's velocity, $g_{\partial W}$ its ring from the neighbours (a **declared input**: S3), $f_W$ the actuator disks' body force in the window and $\Delta t$ the macro-step (an input: S2). The composition layer then assembles the windows and applies its **one global projection per macro-step**, exactly as the classical exposed column does ([[decomposition-speed-by-rotor-count]] §1). Incompressibility is therefore enforced classically and exactly, **level 5 on the physics-encoding spectrum**, whatever the network returns.

- **Backbone:** a small U-shaped convolutional operator on the $128^2$ window with its ring, conditioned on $\Delta t$ by adaptive normalisation. It is the rectangular-chart special case of [[chart-operator-architecture]] §3, where the metric inputs are constant.
- **Size:** set by Step 0's speed budget (§4), not by accuracy ambition. **[AI Inference]:** of the order of $10^5$–$10^6$ parameters. Poseidon-T, the vault's only checkpoint, cost $86.1$ ms per agent per step at 24 windows against a classical window's $340.8$ ([[outcome-c3-learned-speed-at-scale]] §2), and a network a tenth its size has a correspondingly smaller forward cost.
- **Loss:** state error over one macro-step; a **Jacobian term** matching the classical window's response to random low-frequency perturbations of its ring (S5, the property that decided convergence at rank correlation $+0.83$ against accuracy's $+0.69$; [[corrupted-checkpoint-and-jacobian-fidelity]]); and a two-step rollout term for stability (S11).

**Design L-B, the fallback: a learned correction on a coarse window.** March the window classically on a $2\times$-coarse grid, then correct it with a network trained against the fine solution, in the manner of Kochkov et al. **It is kept in reserve** because it is less on-message: its speed comes from the coarse grid, and it must beat Cc by exactly the margin the network adds.

---

## 4. Step 0: price the network before training it

**Speed does not depend on the weights.** So the first thing the next chat does, before a single sample is generated, is time **a randomly initialised network of the intended size** in the workbench's own process on the owner's laptop, on AC power:

1. the classical window step per macro-step, batched and threaded, as `Ep` runs it;
2. the network's forward pass over all 24 windows, batched, with torch's intra-op threads at the same count;
3. the composition layer's projection, which both arms share.

**Registered budget:** $t_L\le t_{Ep}/1.5$. If no network in the size range meets it, the case moves to design L-B, or to the 5-rotor rung, or stops. **All of this happens before data generation, which is the expensive part.** This is the rule F9 followed when it priced adaptation at $0.35$ h per 2000 iterations and did not run it ([[learned-contribution-kill-tests]] §4.7).

---

## 5. The gate — a draft, registered in code before any data exists

Measured over the demo's 12 macro-steps and over W346's 40, on the **held-out** layout the page shows:

| # | condition | registered bar (draft) |
|---|---|---|
| **G1 speed** | L's mean `step` time against the fastest classical arm | $t_{Ep}/t_L\ge1.5$ and $t_F/t_L\ge3$ |
| **G2 accuracy** | error against T, in farm power and in the rms velocity: L's against the worse of F's and Ep's | $e_L\le1.1\,\max(e_F,e_{Ep})$ in both measures |
| **G3 conservation** | the assembled field's divergence after the projection | $\le10^{-9}$, as every farm |
| **G4 stability** | fluctuation energy over 40 macro-steps against Ep's | within 10% at every step |
| **G5 the competitor** | Cc's error against T | **must exceed L's** in at least one of the two measures; otherwise a coarser classical solver does the job, and the case reports that |
| **G6 held out** | the page's rotor layout and inflow are absent from the training set | the split written to `out/learned-case/registered.txt` before generation |

**What the page shows in each outcome:**
- **All pass:** the case, with G5's comparison on its card.
- **G1 fails:** the timings, and why.
- **G2 or G4 fails:** design L-B, once. If that fails too, the case reports it, and the vision stays a vision.
- **G5 fails:** *"a classical solver on a grid twice as coarse matches it"*, which is the C3 finding again, and the website does not claim otherwise.

**[AI Inference]:** the owner asked for a case that works. The honest route to one is a gate that could fail. A case tuned until it passed would be the best of many draws again, and the website's sceptical reader is the one this case exists for.

---

## 6. Data, training, weights

- **Data: the vault's own classical solver.** Farm trajectories over **randomised layouts** on the same $6\times4$ tiling: rotor positions and thrust coefficients drawn per trajectory, inflow direction within a stated range. Each window's state, ring, forcing and next state are one sample. The held-out layout of G6 is excluded. **[AI Inference] on scale:** $10^2$ trajectories of $40$ macro-steps and 24 windows give about $10^5$ window samples, from about $10^2\times40$ fine full-domain macro-steps of CPU time. On the laptop that is **hours, probably more than two, so it needs the owner's go-ahead** under the long-run rule, and it must save its state as it goes.
- **Training: a rented GPU**, under the standing rule: vast.ai, at most \$0.70 an hour, destroyed after use, hashes checked ([[vast-ai-windfarm-runbook]]). **[AI Inference]:** a network of this size on $10^5$ samples at $128^2$ trains in single-digit GPU-hours, so a few dollars. To be priced by a timed epoch before the full run.
- **Weights: the owner's.** Trained from scratch on the vault's own data, with no Poseidon anywhere in the chain (S12), so the licence is whatever the owner chooses. At this size they are a few megabytes, committed beside the case, with their hash in the record.
- **In the workbench:** torch is imported only when the learned case is opened, so the other seven types keep their dependencies. `KMP_DUPLICATE_LIB_OK=TRUE` is set, since torch and numpy's linear algebra share the process.

---

## 7. In the page

- **The header's *Learned experts* button**, shown when the type is the wind farm, opens the case **read-only**. The Model tab shows the windows and rotors with a banner: *"This case is fixed: its experts were trained for this farm."*
- **Run** marches F, Ep, L and Cc in turns, as every workbench run does, and reads T from its stored record.
- **The cards:** speed (L against Ep and F), accuracy (all four against T, as a small bar chart), conservation, and **the competitor**, with G5's line.
- **The caveat line**, always shown: *"Trained on this farm's family of layouts. Experts that work on any shape are what the proposal builds."* It links to the architecture page.

---

## 8. Build order and what to ask the owner

**Approved in advance (O6, the owner, 2026-09-30): torch, the long data-generation run and the GPU rental, yes to each.** Each is still **announced** before it starts, with its expected time or cost, and the long-run and rental rules still hold: save state as it goes, stay under \$0.70 an hour, and destroy the instance.

1. Confirm that torch is installed in the owner's Python (the vault ran Poseidon-T, so it probably is).
2. Step 0 (§4): time the random network. Report. **Stop here if G1's budget cannot be met.**
3. Register §5 and the G6 split in code.
4. **Announce** the data generation (a long run) and the GPU rental, with their expected time and cost. Both are approved (O6).
5. Generate, train, and evaluate on the held-out layout, once.
6. Build the page's case and cards, and record the run like every gallery row.
7. [[showcase-gallery]], [[outcome-c2-learned-experts-in-the-loop]] and [[outcome-c3-learned-speed-at-scale]] get what it measured, **whatever it measured**.

---

## 9. What was built and measured (2026-10-01)

**Every step of §8 ran, in order, and the gate was evaluated once. G5 failed.** The code is `atlas/workbench/learned_*.py` and `scripts/learned_*.py`. The records are in `out/learned-case/`, in commits `a7a300a` (step 0 and the registration), `60dd865` (the data, the truth and the trainer) and `f8bbc9b` (the evaluation, the page and the installer).

### 9.1 Step 0: the network priced before any data

`scripts/learned_step0.py` timed farm-12 at 4 threads, 21 sub-steps per macro-step, with medians over five rounds (`step0-20261001-143036.json`):
- **Conditions:** on battery (88%), no other Python process, the other two chats writing documentation and theory. It supersedes `step0-20261001-142815.json`, whose note said "on AC" while its own power field said battery; both are kept.
- **The classical arms:** Ep's macro-step took $1.34$ s, of which the 24 windows took $1.04$ s; F's took $5.92$ s. The part every decomposed arm shares, the blend, the projection and the band, took $0.29$ s.
- **A random `WindowUNet` of depth 3**, over all 24 windows in one batch:

| width | parameters | forward (s) | $t_{Ep}/t_L$ | $t_F/t_L$ | budget |
|---|---|---|---|---|---|
| 8 | 155,890 | 0.221 | 2.60 | 11.5 | met |
| **12** | **350,186** | **0.399** | **1.93** | **8.55** | **met** |
| 16 | 622,050 | 0.553 | 1.58 | 6.99 | met |
| 20 | 971,482 | 0.871 | 1.15 | 5.08 | missed |
| 24 | 1,398,482 | 1.067 | 0.98 | 4.35 | missed |

**Width 12 was registered**, leaving margin for the evaluation's own noise. The network replaces only the windows' $1.04$ s; the shared $0.29$ s stays.

### 9.2 The registration, before any data

`atlas/workbench/learned_gate.py` and its readable copy `out/learned-case/registered.txt` were committed in `a7a300a`, before the first training sample was generated. They hold §5's six bars with two refinements, both written before any data:
- **G1 is judged only on AC power.** A run on battery does not count.
- **G6 holds out the layout, not the inflow.** This family's freestream band holds the inflow at $(U,0)$ in every case, so it cannot vary between trajectories.

**The split.** 100 training layouts (seeds 1000–1099) and 10 validation layouts (2000–2009). Each has 8 to 16 rotors at least $2.5D$ apart, with induction in $[0.20,\tfrac13]$. A layout is refused if more than 3 of farm-12's rotors have one of its rotors within $0.5D$.

**The measures**, at 12 and 40 macro-steps, against the truth $T$:
- $e_P$: the relative error in farm power, averaged over the last five macro-steps;
- $e_V$: the rms velocity difference over $U$, with every arm block-averaged to cells of $D/16$.

### 9.3 The truth, the data and the training

- **The truth** (`scripts/learned_truth.py`, `truth-farm-12.npz`): farm-12 on the full domain at cells of $D/64$, $1376\times928$ cells, run for 40 macro-steps on the laptop in $1404$ s (on battery at the start and AC at the end; not a registered timing). It is stored on the grid of $D/16$ at the two horizons, with the power at every macro-step.
- **The data** (`scripts/learned_data.py`): run on a rented vast.ai box with an EPYC 9684X (32 cores) and an RTX 4090.
  - 110 layouts of 40 macro-steps each. A layout gives 1,200 window samples: 960 steps, and 240 ring perturbations (2% of $U$, three sine modes, every fourth macro-step) for the Jacobian term.
  - 132,000 samples in $137$ s. Every file's sha256 is in `data-manifest.json`.
- **The training** (`scripts/learned_train.py`, `training-log.json`): 60 epochs in $2{,}002$ s on the GPU.
  - The loss is the step error weighted by the blend weight $\chi$, plus the ring-pair Jacobian term. It ran in fp32 with TF32 off.
  - The weights were chosen by 40-macro-step rollouts against the classical decomposition on validation layouts 2000–2002, every fifth epoch. The best score was the last epoch's: $1.27\times10^{-3}$.
- **The weights' sha256** (`9ce87a71…`) matched on the box and here (`box-logs/sha256-on-the-box.txt`).
- **The cost:** the account's credit fell by \$0.35 over the box's whole life. The box was destroyed, no instance remained, and no later charge appeared.

### 9.4 The one evaluation

`scripts/learned_evaluate.py` refuses to evaluate the same weights twice. It wrote `gate-20261001-154054.json` from 40 macro-steps, arms taking turns in a rotating order. The run was on AC at all 42 power samples (99%), with no other Python process; the other two chats were on documentation and theory. The mean macro-step: F $3.70$ s, Ep $0.757$ s, L $0.497$ s, Cc $0.169$ s.

| bar | registered | measured (12 / 40 macro-steps) | verdict |
|---|---|---|---|
| G1 speed | $t_{Ep}/t_L\ge1.5$, $t_F/t_L\ge3$ | $1.52$, $7.45$ | pass |
| G2 accuracy | $e_L\le1.1\max(e_F,e_{Ep})$ | power: L $7.50\%$ / $6.50\%$, Ep $7.53\%$ / $6.48\%$; velocity: L $3.82\%$ / $5.06\%$, Ep $3.81\%$ / $5.04\%$ | pass |
| G3 conservation | divergence $\le10^{-9}$ | $5.9\times10^{-18}$ | pass |
| G4 stability | energy within 10% of Ep's | $0.63\%$ at worst | pass |
| G5 the competitor | Cc's error above L's in one measure | Cc power $3.63\%$ / $3.02\%$, velocity $3.15\%$ / $4.96\%$: closer than L in both, at both | **fail** |
| G6 held out | no training layout near farm-12 | 110 layouts, every one a registered seed, none within the rule | pass |

The full domain F is within $0.51\%$ / $0.39\%$ of $T$ in power and $0.15\%$ / $0.19\%$ in velocity.

**Disclosed before the run, and kept in its record:** a smoke test of the evaluation script had shown the classical arms' errors at 12 macro-steps, with Cc closer to $T$ than Ep in both measures. That test used an untrained network and wrote its output to a scratch folder. The gate had been registered in `a7a300a` before it, and nothing was changed after it.

### 9.5 What it means for the request in §0

The owner asked for learned experts *"as accurate as classical decomp AND classical full domain, but are faster"*.
- **As accurate as the classical decomposition, and faster:** yes. L tracks Ep to within $0.03$ percentage points in every measure, at $1.52\times$ Ep's speed and $7.45\times$ F's.
- **As accurate as the full domain:** no. F is $15\times$ closer to $T$ in power at 12 macro-steps. The registered G2 compared L with the *worse* of F and Ep, as §5 drafted it, so it passed. This section says so, because the owner's sentence asked for more.
- **The competitor:** a classical solver on a grid twice as coarse is closer to $T$ than L in every measure, in about a third of L's time. That is §5's G5 outcome, *"a classical solver on a grid twice as coarse matches it"*: the C3 finding again. The website does not claim otherwise.
- **Design L-B was not run.** §5 triggers it only when G2 or G4 fails, and neither did.

**[AI Inference]:** L was trained to reproduce the classical window step, so its error against $T$ is the classical decomposition's. It matches Ep's to the second digit, and longer training could at best reproduce Ep more exactly. Ep's own error is the decomposition's, $7.5\%$ in power against F's $0.5\%$. A learned window can beat Cc only by being trained against a resolved reference rather than against the classical step. That is design L-B's direction, or an expert whose target is the fine solution. Either needs a new gate, registered before its data.

### 9.6 In the page and in the installer

- **Where it is:** the first line under the wind farm's *Fast example* arrow, *"Learned experts: a network trained for this farm steps every window (a fixed case)"*. §7 asked for a header button; at 1090 px the header has no width left.
- **Read-only:** a banner gives §7's sentence. An edit is refused with a note, and Run settings' arms, steps and threads are fixed. One Undo restores the case before it.
- **The run:** Ep, F, L and Cc march in turns for 12 macro-steps, as every workbench run does. Then each is measured against $T$ at 12.
- **The card:** this run's speed, and the four arms against $T$ as two bar charts. Beside them, the registered evaluation's six verdicts and *"Not every bar was met, and this card says so"*, then §7's caveat. The headline Speed card names the learned arm, not the coarse competitor.
- **Walked in the served page** at $1090\times620$. The run took about 45 s. Its errors at 12 macro-steps were the evaluation's to the digits shown, since the march is deterministic. Its speed was its own: Ep $1.92\times$ and F $10.09\times$ L's time, which is not a registered measure. No horizontal scroll.
  - The first walk found a fault no test had: the Run view's arm colours had no entry for `learned`, and the view did not draw. It was fixed and the page walked again.
- **The installer** carries the weights and the truth: the only binary files its builder allows, each by exact path. It also carries the registration, the training log and the gate record. Its self-test steps the case once and prints the registered verdict. Without torch the learned arm is not offered, and the page says why.

**Not done here:** §8 step 7 asks [[outcome-c2-learned-experts-in-the-loop]] and [[outcome-c3-learned-speed-at-scale]] to get what this case measured. Those pages belong to another chat. §9.4 and §9.5 are what they need.

---

## See Also

- [[chart-operator-architecture]] — the general expert this case is the smallest instance of
- [[chart-operator-training-and-cost]] — the training plan it is Stage 0 of
- [[outcome-c5-requirements-for-dd-native-experts]] — S1–S12, which §0 and §5 answer to
- [[defect-correction-learned-operator]] — the slot where a learned map cannot change the answer, the alternative if L-A and L-B both fail
- [[demo-fast-examples-plan]] — the classical fast examples
