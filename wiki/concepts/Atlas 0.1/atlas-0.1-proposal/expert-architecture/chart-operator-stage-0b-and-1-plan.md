# Chart operator after Stage 0: the coolant plan (CO-T on laminar cold plates), then the rest of Stage 1

**Type:** Concept page — **plan**, agreed with the owner, and its build record from Phase 1 on; nothing is trained or registered (folder: `Atlas 0.1/atlas-0.1-proposal/expert-architecture/`)
**Status:** written 2026-11-18 from a planning conversation with the owner, who made every decision in §2. **Phase 1 is built and measured (§12, 2026-11-18):** the owner chose model (a), side walls only; the classical CO-T expert, the `ADVEC` seams and the studio's CO-T path are built, and the seams are **stable at a piece Péclet number of 200** by six criteria registered in the script before it ran. **Phase 2 ran (§13, 2026-11-19), and its stop rule is not met:** no classical option meets E4 on every development problem under the 0.90 price bar (finer cutting meets E4 on all six at 1.02; TL0 at $n_c=32$ meets it cheaply on the fast tier, but no coarse size works at $n=32$). On the way, the channel cut (curved channels cut along their walls), the general flow and Danckwerts' inlet were built. **Phase 3's first check, the oracle ceiling, fails (§14, 2026-11-19):** a perfect network priced at TL0's cost, plus its forward passes and the switch, costs 3.35 times the classical solve (1.19 with no switch) against a bar of 0.90 and a competitor at 0.86. **The owner then amended the plan (§15, 2026-11-19): price is a reading, not a gate; the gates are accuracy (E4, and CL-A: meet E4 where TL0 at the same coarse size misses it) and capacity (a probe on the hardest pieces before the dataset is built).** Phase 3 resumes under the amendment. No data, no training yet. The one number measured in §9 is a stand-in for the label cost, and it is labelled as one. Every other cost is **[AI Inference]**, with its arithmetic. The gates in §7 are drafts: their bars are registered in Phase 3, before any result, and never moved afterwards. **Reviewed the same day** (marked "added in review"): the plate–coolant heat path is now an open modelling choice (§3); the safety switch's norm, guarantee and cost are spelt out (§4); development and evaluation layouts are separated, and the dataset's size and transfer are fixed (§6); the oracle is corrected (§6); CO-T's E4 measure is left to registration (§7); training runs in \$5 segments (§2, §5).
**Follows:** [[chart-operator-stage-0-rework]] §20 · [[chart-operator-stage-0]] §12 · **Family:** [[chart-operator-family-catalog]] (CO-T, class G0′) · **Case:** [[case-study-cooling-loop-atlas-0.1]] · **Worklist:** [[gap-worklist]] W364 (this plan), W362 (open), W363 (done) · **UI:** [[integrated-ui-plan]] §12

---

## 0. In one paragraph

Stage 0 and its rework showed that, for CO-D, the untrained coarse level does the work and a network adds nothing to the coupled answer (§1). This plan does not retry diffusion. It moves the learned expert to a case where the simple coarse level is expected to fail and a network could have real work: **heat carried by laminar coolant through a cold plate** (CO-T, advection–diffusion, class G0′). The order is set so that nothing is trained until three things are shown:
- the classical expert couples correctly on pieces;
- the simple coarse level and the classical alternatives fail there;
- a perfect network would beat them at equal price.

The training data is built, checked and frozen before any training. Every trained model, good or bad, gets a card in one expert store that the studio reads. After the coolant, the rest of Stage 1 is built classically. Stage 0b is deferred, and Helmholtz is out.

---

## 1. Why training did not help before (the diagnosis this plan answers)

From [[chart-operator-stage-0-rework]] §19–§20:

1. **The loss and the coupled answer measure different things (P6).** The loss weights each mode by its energy, and most of the energy is in the mid sines ($j=4$–8). The coupled answer reads mostly the corner hats and the low sines ($j=1$–3). Trained, the mid sines improved (median $\rho$ 0.706), while the hats and the low sines barely moved (0.044 and 0.049). **[AI Inference]**, as the rework labels it: the network improved what was cheap to measure, not what counts.
2. **There was little left to learn where it counts.** Untrained, TL0 removes 87–96% of the coupled error above the floor. What it leaves halves with each doubling of $n_c$, as a resolution limit does. The trained network is worth about TL0's own $n_c=16$ ($n_c^\ast$ 13–17.5). Raising $n_c$ is exact and cheap.
3. **The gains did not reach the held-out problems.** On generated test pieces the network removes a median 0.669 of TL0's error. With shuffled inputs it removes −0.569, so the gain is real and specific to each piece. On the held-out problems it removes between −0.50 and +0.015 of the error. $\rho$ also falls with resolution, from 0.736 at $n=32$ to 0.356 at 128.
4. **Before the rework, it was asked for what it could not produce:** global responses, and a corner quotient that grew with $n$ (16, 50, 89). The rework gave both to classical structure: the coarse level, and B's corner patches.

**Not established:** that the network cannot learn the low modes. On its own four pieces the overfit probe reached $\rho$ 0.95. Training used one configuration for 37,500 steps from S's weights. No loss has yet targeted the modes the coupled answer reads under $\mathcal C$.

**What the plan takes from this.** Train only where (a) the cheap classical options fail, (b) a perfect network would win at equal price, and (c) the loss is shown, before training, to rank errors the way the coupled answer does.

---

## 2. The owner's decisions (2026-11-18)

| question | decision |
|---|---|
| what comes next | the **coolant case** (CO-T) first, not Stage 0b |
| which regime | **laminar cold plates** (batteries, power electronics), not turbulent radiator coolant |
| Péclet cap | **200 per piece**; fast plates are handled by cutting channels into shorter pieces, and the price shows what the extra seams cost |
| order | classical seams first, then the decisive test without training, **then** freeze the dataset |
| training data | built, checked against the criteria and frozen **before** any training; never generated during it |
| where to train | **a rented box** (every box under \$0.70/h, with the vast.ai rules) |
| step budget | **37,500 steps per run** |
| how the box runs | **in segments of about \$5**: one box works through the run queue for a whole segment, survives the owner logging off, checkpoints so every run can resume, and stops itself at the segment's budget (§5, Phase 5) |
| expert store | one store, a card per trained model; **failed experts get cards too** |
| publishing weights | **yes**, in principle. Where and under what licence is decided when the first weights are ready, and nothing is pushed without asking |
| Stage 0b (high-contrast diffusion) | **deferred** until after the coolant |
| CO-E, the thermoelastic bond, the biharmonic network | **after the coolant, classical**: built on TL0, trained only if their own test without training shows a job for a network |
| Helmholtz | **out**, noted as a later candidate for a learned coarse basis |
| **price as a gate** (decided 2026-11-19, after the oracle check failed, §14–§15) | **not a gate for the coolant.** Price is measured and reported at every stage, and no "faster" claim is made without a measurement. The gates are **accuracy and capacity**: the learned expert must meet E4 where TL0 at the same coarse size misses it, and the network must first be shown able to fit the hardest pieces. The owner's reasoning: a learned expert is expected to be cheaper on the harder, larger pieces this case will grow into, and that is a priori sound (§15.1), so the coolant's question is whether a network can learn the accuracy at all |

---

## 3. The case: a laminar cold plate

**The device.** A liquid cold plate under a battery cell or a power module: a metal plate with channels about 0.1–2 mm wide, carrying water or water–glycol in laminar flow (Reynolds number below about 1,000).

**The model: a 2-D plan view.**
- The plate conducts heat. The device beneath it is the source $q$.
- The coolant in each channel carries its depth-averaged laminar profile.
- The pieces are cut at the channel wall, so every piece is all coolant or all solid, as Atlas requires ([[chart-operator-family-catalog]] §2, observation 2).

**The PDE**, steady or as one implicit step:
$$\rho c_p\left(\frac{T-T^n}{\Delta t}+\mathbf u\cdot\nabla T\right)-\nabla\cdot(k\nabla T)=q,$$
- $\mathbf u=0$ in the solid;
- at the wall, $T$ and the normal heat flux $-k\,\partial_nT$ are continuous;
- the coolant enters at a given temperature and leaves with what it carries.

**The flow is given, not solved.** On each piece it is divergence-free by construction: a stream function on the reference square, mapped by the Piola transform (catalog §6, CO-T).
- **[AI Inference]:** in a plan view, depth-averaged laminar flow obeys the Hele-Shaw equation $\nabla\cdot\big(\tfrac{h^2}{12\mu}\nabla p\big)=0$, $\bar{\mathbf u}=-\tfrac{h^2}{12\mu}\nabla p$. That is group A, which CO-D already solves. So the diffusion expert could supply the flow and the transport expert carry the heat on it: two Atlas families chained on one device. That needs checking before it is relied on. Hele-Shaw holds only where a channel is much wider than it is deep; in a narrow duct the profile across the width is prescribed instead.

**A modelling choice still open (added in review, 2026-11-18).** In a real cold plate most of the heat enters the coolant through the channel's floor, from the plate beneath, not through its side walls. A plan view whose only plate–coolant contact is the side walls would understate that path. Depth-averaging a non-uniform profile also adds Taylor dispersion along the flow. Two models, for the owner to choose at the start of Phase 1:
- **(a) side walls only:** the equation above, and nothing more. The simplest; the coolant and the plate meet only at seams.
- **(b) side walls plus a floor exchange:** the coolant pieces gain a term $h_{\text{eff}}(T-T_{\text{plate}})/d$, with $h_{\text{eff}}$ from a laminar Nusselt number. That is a co-located bond (catalog group J) between a plate layer and a coolant layer. It adds a positive reaction term, which keeps the class G0′, but it removes the uniform temperature from the coolant pieces' kernel, so gate K changes.

**The regime numbers.** With $\alpha=k/(\rho c_p)$ the coolant's diffusivity, the Péclet number of a piece is
$$\mathrm{Pe}_{\text{piece}}=\frac{\lvert\mathbf u\rvert_{\max}\,L_{\text{piece}}}{\alpha}\le 200 .$$
- For water, $\mathrm{Pe}\approx 7\,\mathrm{Re}$ on the same length, so real microchannels (about Pe 700–7,000 on the channel) exceed the cap.
- The decomposition meets the cap by cutting channels into shorter pieces. The extra seams are part of the price, so what a fast plate costs is reported, not hidden.
- 200 is the highest Péclet number at which the G0′ coupling is measured quadratic (slopes 1.97–2.00 at 0, 20 and 200; catalog §3).
- The exponential transform $c=e^{\mathbf b\cdot x/(2D)}w$ (catalog §4) is not relied on: it exhausts double precision near Pe 70.

**What the studio has today.** `conjugate-heat-2d` runs a plug flow beside a heated block, steady, on one finite-volume grid (`atlas/workbench/families/cooling.py`), with no pieces and no chart operator. [[case-study-cooling-loop-atlas-0.1]]'s passage is a lumped leg.

---

## 4. What makes it hard (and why a network might have work here)

1. **Thin layers.** At moderate Péclet the temperature changes sharply in boundary layers along the wall and at outlets, of thickness about $L/\mathrm{Pe}$. A bilinear coarse space cannot bend into them. That is the first failure case [[chart-operator-stage-0-rework]] §6 names ("advection-dominated transport"). A **learned coarse basis $P_\theta$** bent along the streamlines is the natural thing to learn. Proposition A holds for any full-rank $P$, so the coarse correction is still exact whatever the weights.
2. **It is one-directional (G0′).** The operator is not symmetric, so:
   - there is no energy minimum and no $\tilde\Lambda-\Lambda\succeq0$;
   - the error is $E_W^{\top}A_{II}E_V$, with no sign (catalog §1.3);
   - the expert needs adjoint modes, from the same network with $\hat{\mathbf b}\to-\hat{\mathbf b}$;
   - SE1's second clause has no analogue, and its gate must be rewritten ([[chart-operator-stage-0]] §12).

   **The safety switch** therefore cannot be the energy (an earlier draft in the conversation proposed that, wrongly for G0′). It is the **residual**: a piece's learned modes are kept only if their true residual on the piece's own operator is below TL0's. Otherwise TL0's are used. The residual is a sparse matrix–vector product, needs no reference, and can be checked whatever the weights.

   **Three limits of the switch (added in review):**
   - **The norm decides what it protects.** A plain $\ell^2$ residual is dominated by the fine modes, which would repeat P6 inside the switch. It must be a dual norm, $\lVert r\rVert_{A_s^{-1}}$ with $A_s$ the symmetric (diffusion) part, or a cheap approximation of one: the coarse part exactly through $\mathcal C$'s factorisation plus a local estimate of the rest.
   - **It bounds the residual, not the error.** The error is at most the residual divided by the operator's inf-sup constant, which shrinks as Pe grows. So "never worse than TL0" holds for the residual, and the error is still judged by E4.
   - **Its cost is part of the price.** An exact dual norm needs a solve with $A_s$ on every piece, which could cost as much as the classical solve. Phases 1–3 choose the norm and measure its cost before the registration.
3. **The seams must carry the flow's heat.** At Pe 20 and 200 the exact port matrix's symmetric part is indefinite (relative smallest eigenvalue $-4.4\times10^{-2}$ and $-1.5\times10^{-1}$, catalog §6). Value plus diffusive flux is not a passive pairing, so the seams need the `ADVEC` port with the advective flux ([[port-algebra-atlas-0.1]]). That is classical framework work, and it comes before anything learned.
4. **A high-contrast wall.** Water against aluminium is a conductivity ratio of about 300. So the coolant case also contains part of Stage 0b's question.
5. **The labels must be trusted.** Galerkin on a grid oscillates once the cell Péclet number $\lvert\mathbf u\rvert h/\alpha$ exceeds 2. At $\mathrm{Pe}_{\text{piece}}=200$ that needs $n\ge 100$. References are computed at $n=128$ and checked at $n=256$ (§6, check 2).

---

## 5. The phases

Each phase ends at a stop point. Each feature with a page goes into the studio, with a handler in `atlas/studio/protocol.py`, a view, a `tests/test_studio_*.py` test and a page walk ([[integrated-ui-plan]] §12). The pinned modules (the R2 and TL pins) are not edited: new code goes in new modules.

**Phase 1: the classical transport expert on pieces** (laptop; no learning)
- **First, the owner chooses the model**, (a) or (b) of §3.
- A CO-T piece solver (Petrov–Galerkin, trial and adjoint modes) on the reference square, in a new module beside `atlas/chartop/`.
- `ADVEC` seams in the host, carrying the advective flux.
- In the studio:
  - conjugate-heat cases cut into pieces (channel strips and the plate);
  - the CO-T classical expert on the Pieces screen;
  - Run's Compare against the full domain.
- **Positive control:** the coupled classical answer against one monolithic solve, to round-off with whole traces.
- Seam stability measured up to $\mathrm{Pe}_{\text{piece}}=200$.
- **Stop if** the seams are not stable at 200: report it, and lower the cap with the owner.

**Phase 2: the decisive test, no training** (laptop)
- TL0 for CO-T: the coarse level with a bilinear $P$, untrained.
- The classical alternatives, each priced:
  - a raised $n_c$;
  - a streamline-adapted classical coarse basis (MsFEM-type with oversampling, Hou & Wu 1997);
  - finer cutting (more, shorter pieces);
  - the classical solve.
- Run on the **development layouts** (§6), never on the evaluation layouts.
- **Stop if** any classical option meets E4 under the price bar. Then the coolant needs no network, and that is reported as CO-D's result was.

**Phase 3: the dataset frozen, the gates registered** (laptop, then the owner's approval)
- The dataset is built and must pass §6's five checks, including the oracle ceiling. Then it is hashed and frozen.
- The evaluation layouts are drawn and hashed, and only their classical floors are measured.
- The gates of §7 are written with their bars, accepted by the owner and hash-pinned. The price protocol gets a dry run on a rented whole machine before pinning.
- The studio's Records screen today lists a placeholder "Stage 0b, not registered" row with arms TL0, P_theta, classical and H2r (`tests/test_studio_records.py`, `scripts/studio_walks/p6.py`). It is replaced by the coolant registration's row, with its test and walk updated.

**Phase 4: laptop checks before any box**
- **The overfit probe** on a few pieces. It must move the modes the coupled answer reads (by group), not only $\rho$ overall.
- **A small learning curve** on the CPU, to catch a dead end before money is spent.

**Phase 5: training on a rented box**
- **The arms:**
  - **arm P:** the learned coarse basis $P_\theta$ plus the fine residual;
  - **arm R:** the fine residual on TL0 only, as the control for whether learning the basis is what pays.
- **The scale:** 3 seeds per arm at 37,500 steps; the learning curve on nested subsets (§6) at one seed.
- **The loss:** the error of the port responses, weighted by port traces sampled from coupled solves on the training layouts. **[AI Inference]:** this weights the modes as the coupled answer reads them, which is P6's fix by construction. Check 5 of §6 tests it before training.
- **The training step** is priced on the box before the long runs (F9's discipline).
- **Segments of about \$5** (the owner's choice). One box works through the queue of runs for a whole segment, so it is not created and destroyed for each run.
  - The runner on the box needs neither the laptop nor the chat. It keeps its queue and state on the box's disk, and checkpoints every run atomically (weights, optimiser, random states, step and data position) at least every 30 minutes.
  - At the segment's budget, or when the queue is empty, it writes a DONE marker with every file's sha256 and stops the instance itself. `scripts/chartop_box_run.sh` already stops a box at a deadline with the container's own key (`CONTAINER_API_KEY`). The TL pin hashes `chartop_box_*`, so the new runner copies that pattern into a new script.
  - A run cut by the deadline resumes in the next segment from its last checkpoint.
  - The laptop's watcher only collects. If the laptop is off, the results wait on the stopped box's disk, which bills for storage only. A stopped box's GPU can be re-rented by someone else, so the next segment may need a new box, with the checkpoints and the dataset uploaded and checked by sha256.
  - The owner approves each segment before it starts.

**Phase 6: one evaluation, and the write-up**
- Accuracy on the held-out layouts; the price on a rented whole machine with fixed BLAS threads and an A/A check.
- Every trained model gets a card (§8).
- The write-up goes on this page, with a worklist row, the current-state page and the log.

---

## 6. The training data: built, checked, frozen

**How it is made.**
- **Pieces:** the existing generator's curved four-sided pieces, at $n=32$, 64 and 128. Each is given:
  - a divergence-free flow (stream function, Piola-mapped; channel profiles among them);
  - $\mathrm{Pe}_{\text{piece}}$ drawn across $[0, 200]$;
  - a material, coolant or solid (contrast up to about 300);
  - a Fourier number for the step.
- **Labels:** each piece's trial and adjoint port responses, solved classically at $n=128$.
- **Three sets of layouts, kept apart** (added in review: one held-out set used both to decide whether to go on and to judge the result would be judged on the problems it was chosen from):
  - **training layouts**, generated (random channel networks on a plate). Their coupled solves supply the port traces that weight the loss;
  - **development layouts**, drawn in the studio with its Channel tool (a serpentine plate, parallel channels, a block beside a channel). Phase 2 and the checks below run on them;
  - **evaluation layouts**, drawn and hashed at registration, with only their classical floors measured. No learned or coarse arm runs on them before the one evaluation.
- **The learning curve:** nested subsets of the frozen set (25%, 50%, 100%), so the curve needs no new data.
- **Size and transfer** (added in review). Full port responses at $n=128$ are about 16 MB per piece, so 50,000 pieces would not fit or upload. The stored labels must be compact: the trial and adjoint port maps (64 × 64 each), the coarse projections, and full fields only for the validation pieces. The dataset's size is estimated before it is built. It is moved to a box as files and checked by sha256, never regenerated there: sin, cos and atan2 differ in the last bit between Windows and glibc, so regenerating on Linux would not reproduce the frozen bytes.

**The five checks the dataset must pass before it is frozen:**
1. **Coverage:** the registered ranges of Péclet number, shape, contrast and step are filled as declared.
2. **Label quality:** a sample's labels change by less than a stated tolerance from $n=128$ to $n=256$.
3. **The control:** TL0 misses E4 on the development layouts (Phase 2). If it does not, stop.
4. **The oracle ceiling** (*a reading since 2026-11-19, not a stop; §15*)**:** the exact port responses stand in for a perfect network. They are priced at TL0's cost, plus the planned network's forward pass measured untrained on the laptop, plus the safety switch. That price must be under the price bar and under every classical option that reaches the same accuracy. If even a perfect network cannot pay for itself, no trained one can. (Corrected in review: an earlier draft used the best classical coarse basis as the oracle, which is itself one of the competitors.)
5. **Loss–gate alignment:** across a fixed set of perturbed networks, the training loss and the coupled error on the development layouts must rank them alike, with a rank correlation above a threshold registered beforehand. This is the check that would have caught P6 before training.

---

## 7. The gates and the stopping rule (drafts; bars registered in Phase 3)

| gate | what it asks | kind |
|---|---|---|
| CX, K, D | the coarse correction exact and the kernel (a uniform temperature) reproduced, whatever the weights | structural, carried over |
| SAFE | on every piece, the kept modes' residual is at most TL0's (the switch of §4) | structural |
| E4 | the coupled error on the evaluation layouts under E4's bar. CO-D's E4 is an energy norm, which G0′ lacks, so CO-T's measure is fixed at registration: for example the relative $H^1$-seminorm error of the coupled field, with the outlet's bulk temperature and the heat balance beside it, each against its classical floor | accuracy |
| CL | *(replaced on 2026-11-19 by CL-A, §15.3; kept here as first drafted)* the trained expert beats the **best classical option at equal measured price** (raised $n_c$, the classical multiscale basis, finer cutting). Compared per layout, timing the whole coupled solve, since finer cutting changes the pieces themselves | the decisive comparison |
| SEEDS | CL holds for every seed of the winning arm, not just the mean | robustness |
| PR | price against the classical solve, measured on a rented whole machine | measured, not assumed |
| AMORT | how many piece solves repay the training cost | reported |

**The stopping rule (draft).** If the coolant's trained expert fails CL, the learned-expert track pauses. Atlas's claim stays **"certified composition of classical experts"**. Stage 0b is then decided with the owner in light of what the coolant showed.

---

## 8. The expert store

**Why.** Every stage trains experts: specific or broad, good or bad. Today they are scattered:
- the weights are under `out/chartop/runs/<kind>/<name>/best.pt` (git-ignored);
- their sha256 is in the evaluation records;
- the studio's list of experts is written by hand in `atlas/studio/experts.py`.

**The layout.**
```
out/experts/
  catalog.json                 one row per expert: id, family, verdict, card path
  <family>/<expert-id>/
    card.json                  what it is, how it was built, how it did
    weights.pt                 not committed until publishing is decided; sha256 in the card
```

**The card holds:**
- **what it is:** the PDE, its guarantee class, and its envelope (the range of Péclet, shape, contrast and step it is valid for);
- **how it was built:** the code commit and module, the architecture, the training settings, the frozen dataset's hash and the seed;
- **how it did:** its registration, its evaluation record, every verdict and its measured price.

**Rules.**
- Every trained model gets a card, **including the failures**.
- Stage 0's eleven arms, the forms runs and TL-s0/s1 are backfilled from their existing records. A card says whether its weights are present locally. Some earlier checkpoints may not be, and a card never claims a file it cannot hash.
- The studio reads `catalog.json` in place of the list written by hand. An expert is offered only when its weights' sha256 matches its card, as now. This change touches `experts.py`, which no registration record under `out/chartop/` names (checked 2026-11-18; check again before editing).
- **Publishing:** the owner said yes in principle. Where (a GitHub release or a model hub) and under what licence are decided when the first weights are ready. Nothing is pushed without asking. NeuberNet is never stored or published here, and no Poseidon weights are used.

---

## 9. Price

**Measured earlier:**
- a TL training step on a rented RTX 3090: 0.207 s over the training mix, at \$0.217 an hour with storage ([[chart-operator-stage-0-rework]] §19.1);
- the same step on this laptop (Core Ultra 7 155H, 16 GB, no CUDA GPU), on AC at 8 threads: 1.5 s ([[chart-operator-stage-0-rework]] §14);
- the price check on a rented 5900X: \$0.35; a price protocol's dry run: \$0.32.

**Measured here, a stand-in.** A non-symmetric 5-point advection–diffusion matrix (Pe 50), one sparse LU (SciPy `splu`) and 64 trial plus 64 adjoint right-hand sides, single-threaded. **On battery**, so it is an upper bound, and it is not Atlas's solver and not recorded in `out/`:

| grid | median time |
|---|---|
| $n=32$ | 11 ms |
| $n=64$ | 49 ms |
| $n=128$ | 265 ms |

**[AI Inference]:** with the chart's overhead, about 0.15–0.3 s per labelled piece. So 100,000 pieces take about 20–40 minutes on about 12 of the laptop's cores. Data stays cheap.

**[AI Inference]: training.** Assumptions:
- a CO-T step costs 1.5–2.5 times TL's, because it carries trial and adjoint modes;
- the plan is about 7.5 run-equivalents: 2 arms × 3 seeds, plus the learning curve at one seed.

| | per run (37,500 steps) | whole plan |
|---|---|---|
| rented RTX 3090 (about \$0.22/h) | 3–5 h | 23–39 h of training: **about \$7–12** with data, the dry run, pricing and margin |
| this laptop, on AC (for comparison only) | 24–39 h | 180–300 h: about 9–15 kWh at about 50 W, but one to two weeks of continuous running |

- Two boxes in parallel halve the wall time at the same cost.
- **Phases 1–4** run on the laptop and cost hours, not money.
- The real training step is priced on the box before the long runs, and replaces these figures.

---

## 10. After the coolant

- **CO-E, the thermoelastic bond on CS-9, the biharmonic network:** built **classically on TL0** as framework demonstrations, with $\mathcal C$ taking a vector bilinear $P$ for CO-E ([[chart-operator-stage-0-rework]] §20.4). Each trains a network only if its own test without training (Phase 2's kind) shows a remainder the coupled answer reads. The bond needs a CO-D that meets E4, and H2r is one.
- **Stage 0b** (thin high-contrast features in diffusion): deferred. When it returns, it needs a named Atlas use for features too small or too many to cut into pieces (laminations, composites, porous media), and "cut the features out" as a classical competitor.
- **Helmholtz:** out of this plan. It is recorded as a later candidate for $P_\theta$, since oscillatory physics is §6's third failure case of a bilinear coarse space.
- **W362** (pieces that are not squares) stays open. **W358** stays closed. Its open remainders are points where an interface lies inside a piece (P3, untested) and the corner singularities of elasticity and plates. The proposal is a new worklist row when CO-E starts, not a reopened W358.

---

## 11. What this page does not do

- It builds, trains and measures nothing beyond §9's stand-in.
- It registers no gate. §7's bars are set in Phase 3, accepted by the owner and hash-pinned before any result.
- It puts no number on the website or in `proposal/`.
- It does not cover turbulent coolant: fast radiator flow is G3 ([[chart-operator-family-catalog]] §1.3) and outside this design.

---

## 12. Phase 1: the classical transport expert on pieces (built 2026-11-18)

**The owner's choice of model:** (a), side walls only, in chat on 2026-11-18 ("(a) Side walls only (Recommended)"), before any code. The plate and the coolant meet only at seams, and every piece is one material and one field. The floor exchange, (b), is not built.

**Built.** Nothing is learned, and no pinned module was edited.

| file | what |
|---|---|
| `atlas/chartop/transport.py` | the CO-T piece: Galerkin advection from a nodal stream function, trial and adjoint modes from one LU, the Péclet numbers, the envelope |
| `atlas/chartop/transport_host.py` | `TransportLayout`, a subclass of the pinned `host.Layout`: Petrov–Galerkin coupling, the `ADVEC` seams and their refusals, the passivity and positive-real certificates, the outlet's mixing-cup temperature, the balance |
| `atlas/chartop/transport_cases.py` | three layouts built in code: a channel, a cold plate, a bend |
| `scripts/cot_phase1_seams.py` | the seam measurement, its criteria fixed in the file before its first run |
| `atlas/studio/cot.py`, the Pieces and Run views | a conjugate-heat case cut into pieces, the classical CO-T expert, Run's Single and Compare |
| tests and walk | `tests/test_chartop_transport.py` (28), `tests/test_studio_cot.py` (13), `tests/test_studio_physics.py` updated; the page walk `scripts/studio_walks/cot1.py` (25 checks) |

### 12.1 The piece

On a piece of one material, divided by its conductivity $k$ and pulled back to the reference square, one step or the steady state reads
$$\int\nabla T\cdot A\nabla v+\int(\mathbf b\cdot\nabla T)\,v+\int\frac{\tilde J}{\mathrm{Fo}}\,Tv=\int\tilde J\,g\,v,\qquad A=JG^{-1},\quad\mathbf b=\frac{J\,DF^{-1}\mathbf u}{\alpha}.$$
- **The flow is a stream function.** The physical stream function over $\alpha$ is sampled at the chart's nodes and interpolated bilinearly. It gives $\mathbf b=(\partial_\eta\psi,\,-\partial_\xi\psi)$, the contravariant Piola transform of $\mathbf u/\alpha$. So $\nabla\cdot\mathbf b=0$ at every point, and the flux through any element edge is the increment of $\psi$ along it.
- **Galerkin, integrated exactly.** The advection matrix $C_{kl}=\int(\mathbf b\cdot\nabla\phi_l)\,\phi_k$ is trilinear in $(\psi,\phi_l,\phi_k)$, of degree 2 in each direction, so $2\times2$ Gauss points are exact.
- **The loop identity.** $C+C^\top=B$ exactly, with $B$ the loop's flux mass. On a counter-clockwise loop $(\mathbf b\cdot\mathbf n)\,ds=d\psi$, so $\int(\mathbf b\cdot\nabla T)v+(\mathbf b\cdot\nabla v)T=\oint(\mathbf b\cdot\mathbf n)\,Tv$ is a sum over the boundary edges of $\Delta\psi$ times each edge's mass. Measured to $2\times10^{-16}$ on a curved chart with a random $\psi$.
- **The adjoint.** $K^\top=K(-\psi)+B$, and $B$ lives on the loop. So the adjoint modes $G=-K_{II}^{-\top}K_{GI}^\top Q$ are the trial modes of the same piece with its flow reversed, to $4\times10^{-16}$: the catalog's $\hat{\mathbf b}\to-\hat{\mathbf b}$, which lets one network give both.
- **Péclet numbers.** A cell's is $h\max(\lvert b_1\rvert/A_{11},\,\lvert b_2\rvert/A_{22})$ at its centre, and a piece's is $n$ times its largest. For a straight piece of length $L$ along the flow that is $\lvert\mathbf u\rvert_{\max}L/\alpha$, the plan's $\mathrm{Pe}_{\text{piece}}$. The envelope refuses, by name, a piece above the cap of 200 ("cut the channel into shorter pieces"), a cell above 2 ("the grid needs $n\ge$ …"), and a chart Proposition 3 does not certify.

### 12.2 The seams

- **Petrov–Galerkin.** The host forms $\Lambda_i=W_i^\top K_iV_i$ with $V=(H;Q)$ and $W=(G;Q)$, assembles them and solves by LU. For any modes with exact traces, $\tilde W^\top K\tilde V-\Lambda=E_W^\top K_{II}E_V$.
- **Why `ADVEC`.** A Galerkin piece's symmetric part is its diffusion and reaction plus $B/2$: positive where the flow leaves and negative where it enters. Coupled by value and diffusive flux alone (`THERM`), a piece is not passive.
- **The split.** On every seam the flow crosses, the host takes $\theta B$ out of each piece's matrix and puts it back as the seam's junction. The default is $\theta=\tfrac12$ ("skew"): the port's flow is then the diffusive flux minus half the advective flux, whose product with $T$ is the piece's dissipation. The alternative is $\theta=1$ where the flow enters and 0 where it leaves ("upwind", a Robin–Robin split, which keeps a margin). An inlet must hold a temperature, and an outlet keeps its $B/2\succeq0$. Then every piece's exact port matrix has a positive semidefinite symmetric part, since a Schur complement keeps one.
- **The split changes no coupled answer.** $B$ lives on the traces, which are exact, so $W^\top K^\theta V=W^\top KV-\theta\,Q^\top BQ$ for any modes, learned or classical, and the junction restores the sum. It changes what each piece certifies and where the advective heat crosses. The three splits' coupled fields agreed to $6\times10^{-14}$.
- **Refused, by name, before any solve:** flow across a seam into a solid piece (it declares no `ADVEC` port); heat-capacity fluxes $k\,\Delta\psi$ that disagree across a seam (mass not conserved); a seam that carries flow and does not conform; flow entering through a side not held at a temperature.

### 12.3 Seam stability up to Pe 200 (measured)

`scripts/cot_phase1_seams.py` wrote `out/chartop/cot/phase1-seams.json` and `.txt` (commit `70308fc` with this phase's files, 702 s on AC). The layouts were the channel (4 pieces), the plate (an aluminium strip on each side of the channel, 342 times the water's conductivity, after one piece of unheated entry) and the bend ($2\times3$, log-polar). It ran Pe 0, 20, 50, 100 and 200, steady and one implicit step (Fo about 0.07 in the water). Criteria S1–S6 were fixed in the script before its first run.

| | criterion | result |
|---|---|---|
| S1 | with whole traces, the coupled answer within $10^{-10}$ of $\max\lvert T\rvert$ of the monolithic solve | **met**: at most $5.0\times10^{-11}$ (the plate at Pe 200) |
| S2 | the coupled interface matrix positive real at 4, 8, 16 and 32 modes | **met**: smallest symmetric eigenvalue over the norm $8.7\times10^{-7}$ to $3.4\times10^{-2}$ |
| S3 | every piece passive under the `ADVEC` splits | **met**: at least $-4.1\times10^{-14}$ relative |
| S4 | at 16 modes, the coupled energy error within 10 times the oracle's (the reference's traces projected onto the same ports) | **met**: the ratio 0.947–1.000 |
| S5 | no amplification along a channel cut into 1, 2, 4 and 8 pieces at Pe 200 | **met**: the ratio 0.999–1.000, the energy error $1.3\times10^{-3}$ down to $5.7\times10^{-4}$ |
| S6 | Petrov–Galerkin: the interface matrix's error quadratic in the modes' error (slope at least 1.8) | **met**: 2.000 on every layout and Pe |

- **Reported beside them, not judged.** The `THERM`-only pairing leaves pieces indefinite: $-0.068$ at Pe 20, $-0.14$ at 50, $-0.19$ at 100 and $-0.18$ at 200 on the channel, slightly more on the bend. The catalog measured $-0.044$ and $-0.15$ on finite volumes. Tested with the trial modes instead of the adjoint ones, the channel's slope is 1.02 at Pe 20 and 200: first order, as G0′ says it must be.
- **The 16-mode floors.** On the channel and the bend, 0.07–0.31% in energy, steady, with the outlet's mixing-cup temperature within $10^{-6}$ of the range. On the plate, steady, 5.9–13%, though the outlet errs by at most $1.2\times10^{-4}$ of the range. **[AI Inference]:** the plate's re-entrant corners and its contrast put the energy into a few corner-singular traces. That floor, not the seams, will bound what E4 can ask on plates, so Phase 2 measures it on each development layout first.
- **Found while building:** with the inlet's held corner touching the heated plate, the floor was 19% even at Pe 0, a point sink that no port count resolves. The plate now starts with an unheated coolant entry, as a manifold-fed plate does.

**The stop point is not triggered.** The seams are stable at 200, and the cap stays.

### 12.4 In the studio

- **The pieces.** A `conjugate-heat-2d` case is cut on Pieces as a heat case is, and `atlas/studio/cot.py` makes the pieces CO-T's. Coolant materials (`flows`) carry plane Poiseuille flow at the case's mean speed `u_coolant`, from the side their `coolant-inlet` is on. Each material's `heat` is its source, and `coolant-outlet` is the do-nothing outlet. A coolant region that is not a straight rectangle is refused by name: a curved channel needs its flow solved on it, which Phase 2's development layouts need.
- **Pieces screen:** each piece's Pe and cell Pe, the CO-T envelope, its sides as `THERM` or `THERM+ADVEC`, the classical CO-T expert (the only one), and the check (whole traces and 16 modes, with the outlet and the balance).
- **Run:** Single and Compare, steady, since the family runs steady. The cards add the outlet's temperature, the `ADVEC` seams and class G0′. CO-T's E4 is not registered yet, so the agree chip shows no bar.
- **The workbench's `cooled-block`.** At its own 1 cm/s, cut in two columns, its coolant pieces are near Pe 8,400 and are refused with what to do. At 0.1 mm/s, in two columns at $n=64$, it has twelve pieces, three `ADVEC` seams and Pe at most 84. Whole traces agree with the full domain to $4.9\times10^{-11}$. At 16 modes the floor is 31% in energy and 9.4% of the range at the outlet: pieces of 16 to 1, copper against water (667 to 1) and the inlet's corner on the block.
- **The page walk** `scripts/studio_walks/cot1.py`: 25 checks with real input, no console error, both themes, nothing overflowing at 1090 × 620.

### 12.5 What Phase 2 starts from

- The development layouts are drawn in the studio. The serpentine needs a flow solved on a curved channel and pieces cut along it; neither is built.
- The decisive test (TL0 for CO-T and the priced classical options) runs on those layouts, each layout's 16-mode floor measured first.

---

## 13. Phase 2: the decisive test, with no training (2026-11-19)

**The owner's decisions for it** (in chat, 2026-11-19, before it ran):
- **E4 for the decisive test:** "Floor + CO-D margins". Against the full-domain solve on the same mesh, an option's energy error is at most the classical 16-mode expert's own plus 0.02, its largest nodal error at most the classical's plus 0.01 of the range, and its outlet mixing-cup temperature error at most the classical's plus 0.01 of the range. Phase 3 re-registers it.
- **The price bar:** "0.90 of classical". The price is the median time of an option's whole coupled solve over the classical expert's, on the same problem, timed in turns; over the development problems, the geometric mean.
- **The development layouts:** asked whether to build them from rectangles, the owner asked instead: "can't you use the same domain decomposition tools we've already developed (conformal, laplacian, tfi)?" They could. The chart methods chart any four-sided piece, curved or not; what was missing was only where to cut (§13.1).

### 13.1 What was built for it

| file | what |
|---|---|
| `atlas/studio/channel_cut.py` | **the channel cut**: a plate with channels drawn by the Channel tool, cut across at stations and along every channel's walls and every rectangular region's edges. Each piece is a curved quadrilateral charted by TFI (Laplacian if TFI is not certified), conforming by construction (neighbours share the same side objects). `rows` splits each layer, so twice the columns and rows at half the grid is the same mesh |
| `atlas/chartop/transport_flow.py` | **the flow on any channel the pieces tile**: $\psi=Q\,f(v)$ with $v$ the channel's harmonic across-coordinate (one sparse solve on the merged coolant mesh) and $f(s)=3s^2-2s^3$. Plane Poiseuille in a straight channel, mass conserved on every seam, the velocity zero at the walls |
| `atlas/chartop/transport.py` (changed) | **Danckwerts' inlet**: the inflow term $\lvert\mathbf b\cdot\mathbf n\rvert(T-T_{\text{in}})v$ on a declared inlet side, in place of a held temperature |
| `atlas/chartop/transport_host.py` (changed) | the interface assembled sparse and solved by sparse LU; the algebra is unchanged, and every coupled answer is the same to rounding |
| `atlas/chartop/transport_coarse.py` | **TL0 for CO-T**: the plain square's modes, then the coarse correction with the piece's true operator, trial and adjoint. Also the multiscale bases (MsFEM, conforming and oversampled), their local problems solved together from the matrix's nine-point stencil |
| `atlas/chartop/transport_switch.py` | **the safety switch's residual**: the dual norm $\lVert r\rVert_{D^{-1}}$ exactly, by a two-level additive Schwarz estimate, and by Jacobi; `SwitchedExpert` |
| `atlas/studio/cot_layouts.py`, `scripts/cot_dev_layouts.py` | **the development layouts**, drawn as the studio draws them, at two speeds each, and written as case files with a manifest of sha256s (`out/chartop/cot/layouts/`) |
| `scripts/cot_phase2_decisive.py` | the decisive test, its criteria fixed in the file before its first run |
| tests and walk | `test_chartop_transport_coarse.py` (9), `test_chartop_transport_phase2.py` (10), `test_studio_channel_cut.py` (16), `test_studio_cot.py` updated; the page walk `cot2` (14 checks) |

**A finding while building: the held inlet pinned the plate.** On the first winding layout, the classical expert's 16-mode floor was 31% in energy. The inlet was a Dirichlet side, and its two corners are nodes of the aluminium beside the channel, so the plate was held at 300 K at two points. Those are point sinks, log-singular, which no port count resolves. Danckwerts' inlet, $-k\partial_nT+\rho c\,(\mathbf u\cdot\mathbf n)T=\rho c\,(\mathbf u\cdot\mathbf n)T_{\text{in}}$, imposes nothing where the velocity vanishes, at the walls. With it, the same layout's floor fell to 0.31% and the outlet temperature's error to $10^{-12}$ of the range, and every watt the device makes leaves with the coolant, to rounding (tested). **[AI Inference]:** Phase 1's plate floors (5.9–13%) were measured with held inlets on a layout with re-entrant corners; the development layouts have neither.

**The development layouts** (aluminium plates 16 mm wide, 2 mm water channels drawn with the Channel tool, a heated device of $10^7$ W/m³; each at a slow speed, Pe 20 per piece at $n=32$, and a fast one):

| layout | what | pieces | fast tier |
|---|---|---|---|
| `dev-block` | a block beside a straight channel | 32 | Pe 180 at $n=128$ |
| `dev-parallel` | two straight channels, the device below | 48 | Pe 110 at $n=64$ |
| `dev-winding` | one S-shaped winding channel (the serpentine) | 32 | Pe 180 at $n=128$ |

### 13.2 The result: the stop rule is not met

`scripts/cot_phase2_decisive.py`, record `out/chartop/cot/phase2-decisive.json` and `.txt` (commit `536f686` with this phase's files; 85 minutes on AC; BLAS threads fixed at four and checked). Each option's energy error against the full-domain solve, and its price (the classical solve is 1):

| problem | floor | TL0-8 | TL0-16 | TL0-32 | TL0-64 | MsFEM-16 | MsFEM-OS-16 | finer | finer + TL0-16 |
|---|---|---|---|---|---|---|---|---|---|
| block, Pe 20 | 0.13% | 6.3%, 0.63 | 2.8%, 0.82 | — | — | 6.1%, 1.96 | 6.9%, 3.46 | **0.00%, 2.38** | 2.9%, 3.37 |
| block, Pe 180 | 2.05% | 12.5%, 0.33 | 6.6%, 0.33 | **3.7%, 0.34** | **2.5%, 0.44** | 41.5%, 0.70 | 42.0%, 2.41 | **0.26%, 0.69** | **3.0%, 0.36** |
| parallel, Pe 20 | 0.11% | 5.2%, 0.48 | 2.4%, 0.63 | — | — | 5.0%, 1.86 | 5.6%, 2.21 | **0.00%, 1.55** | 2.4%, 1.59 |
| parallel, Pe 110 | 1.09% | 10.1%, 0.49 | 5.1%, 0.49 | **2.5%, 0.60** | — | 28.7%, 0.94 | 28.8%, 1.34 | **0.12%, 0.91** | **2.2%, 0.61** |
| winding, Pe 20 | 0.12% | 6.6%, 0.54 | 3.0%, 0.66 | — | — | 5.9%, 2.28 | 6.6%, 2.84 | **0.00%, 1.52** | 3.1%, 1.65 |
| winding, Pe 180 | 1.63% | 11.8%, 0.31 | 6.2%, 0.33 | **3.4%, 0.34** | **2.2%, 0.46** | 38.8%, 0.69 | 40.0%, 1.08 | **0.21%, 0.75** | **2.9%, 0.35** |

Bold marks an option that meets E4 on that problem. A dash: the coarse size is not below the piece's grid there. The block's slow problem failed its A/A check (1.509, outside [0.90, 1.10]), so its prices are reported and not counted; the other five held (0.94–1.01).

| option | E4 met on | price over the five counted problems | stops Phase 2 |
|---|---|---|---|
| finer cutting | **6 of 6** | 1.023 | no: above 0.90 |
| finer cutting with TL0-16 | 3 of 6 (the fast tier) | 0.726 | no |
| TL0-8, TL0-16 | 0 of 6 | 0.42, 0.47 | no |
| MsFEM-16, MsFEM-OS-16 | 0 of 6 | 1.14, 1.85 | no |

**No classical option meets E4 on every development problem under the price bar, so Phase 2 does not stop.** What the table says beside the verdict:
- **The bilinear coarse level does not break down with the Péclet number, as §4 expected it might.** Its error roughly halves with each doubling of $n_c$ at every speed, as a resolution limit does (CO-D's pattern, [[chart-operator-stage-0-rework]] §13). On the fast tier, TL0 at $n_c=32$ meets E4 at 0.34–0.60 of the classical solve. TL0 fails on the slow tier only because those pieces are at $n=32$, where $n_c=16$ is the largest coarse size below the grid; it misses E4's energy bar by 0.2–0.9 points.
- **The multiscale bases fail badly where the flow matters**, 29–42% at Pe 110–180, more than the bilinear basis at the same size. **[AI Inference]:** MsFEM's linear boundary data on every coarse cell forces artificial layers where the flow crosses cell edges, a known weakness of MsFEM for advection; oversampling did not help.
- **Finer cutting is the most accurate option**, and dearer than the classical solve on the slow tier (1.52–2.38: more pieces, more seams), cheaper on the fast one (0.69–0.91).
- **The full-domain solve** cost 1.19–2.81 of the classical decomposed solve on the same mesh.
- **A diagnostic of the record misread the $n=128$ meshes** ("references agree to -"): it keyed nodes by coordinates rounded to 1 nm, and many sat on half a nanometre. Matched by distance, the two cuts' nodes coincide within $1.3\times10^{-17}$ m and their references agree to $7\times10^{-9}$ K. The diagnostic was fixed after the run; no criterion changed, and the finer arms were always measured against the reference on their own, identical mesh.

### 13.3 The safety switch's cost

On TL0-16's answers, piece by piece, against one classical piece solve (the record, every problem):

| the dual norm | cost | its ratio to the exact one |
|---|---|---|
| exact (a sparse LU of $D_{II}$) | 0.85–1.04 | 1 |
| two-level Schwarz (blocks inverted together) | 1.22–2.52 | 1.08–1.27 at $n=128$; 2.2–2.4 at $n=32$ |
| Jacobi | 0.18–0.34 | 0.30–1.06: not equivalent; it leans on the fine modes |

**The switch as built costs about one classical piece solve**, before any network runs. **[AI Inference]:** that alone would leave even a perfect network unable to pay for itself under the 0.90 bar, so Phase 3 must find a cheaper equivalent of the dual norm, and it runs the oracle ceiling (§6, check 4) first, before building the dataset.

### 13.4 What Phase 3 starts from

- **Check 3, the control, holds:** TL0 at $n_c=16$ misses E4 on all six development problems.
- **The competitors a network must beat at equal price:** on the fast tier TL0 at $n_c=32$ (0.34–0.60) and finer cutting with TL0 (0.35–0.61); on the slow tier the classical solve itself, since nothing cheaper meets E4 there.
- **The oracle ceiling decides whether anything is trained:** TL0's cost, plus the planned network's forward pass measured untrained, plus the switch, against the bar and those competitors.

---

## 14. Phase 3, check 4: the oracle ceiling fails (2026-11-19)

Run first in Phase 3, before any dataset (§13.4). The script is `scripts/cot_phase3_oracle.py`; the record is `out/chartop/cot/phase3-oracle.json` (also `.txt` and `.log`), at commit `9fd35b2`. **Check 4 fails, with or without the switch, so the plan stops here: no dataset, no training, no box.** What happens to the coolant case next is the owner's decision.

### 14.1 What was priced

The criteria were written into the script before its first run.
- **The oracle:** a perfect network behind the switch. Its price is the sum of:
  - the whole coupled solve of TL0 at the network's coarse size ($n_c=\min(16,n/2)$);
  - on every piece, the network's inputs and two forward passes (trial, and adjoint by the reversed flow). The network is CO-D's U-Net at 1.1 M parameters with CO-T's six input channels, untrained (`transport_model.py`);
  - the switch's two dual residuals, one for the learned answer and one for TL0's.

  Its accuracy is credited as the classical floor's, which is what a perfect network would have.
- **The switch priced:** the cheapest Schwarz estimate within a factor 3 of the exact dual norm on every development piece. Both $n_c=16$ and $n_c=32$ qualify (ratios 1.07–2.40). The oracle is also priced with no switch, as a lower bound.
- **The competitor:** the cheapest classical option reaching the floor's accuracy, per problem: $\min(1, \text{finer cutting})$.
- **The pass condition:** the oracle's geometric-mean price is under 0.90 and under the competitor's, over the problems whose A/A check holds.

### 14.2 The result

Prices are multiples of the classical solve: medians of five interleaved turns, with BLAS and torch threads fixed at 4.

| problem | $n$ | pieces | A/A | finer | TL0 | oracle, no switch | oracle, switch | competitor |
|---|---|---|---|---|---|---|---|---|
| dev-block slow | 32 | 32 | 1.042 | 2.03 | 0.79 | 2.79 | 7.39 | 1.00 |
| dev-block fast | 128 | 32 | 0.993 | 0.75 | 0.32 | 0.58 | 1.15 | 0.75 |
| dev-parallel slow | 32 | 48 | 1.012 | 1.78 | 0.67 | 2.38 | 6.22 | 1.00 |
| dev-parallel fast | 64 | 48 | 0.861 (not counted) | 0.92 | 0.47 | 0.97 | 4.37 | 0.92 |
| dev-winding slow | 32 | 32 | 1.120 (not counted) | 2.05 | 0.71 | 2.73 | 7.42 | 1.00 |
| dev-winding fast | 128 | 32 | 0.973 | 0.74 | 0.33 | 0.53 | 2.36 | 0.74 |

Four problems were counted. The geometric means are:

| | Geometric mean |
|---|---|
| Oracle with the switch | **3.35** |
| Oracle with no switch | **1.19** |
| Competitor | 0.86 |
| Price bar | 0.90 |

Check 4 fails both ways. The untrained network's two forward passes cost 48 ms per piece at $n=32$, 84 ms at $n=64$ and 166 ms at $n=128$.

**The run was disturbed.** The laptop was on battery, and it entered standby three times during the run (Kernel-Power 506/507: 13:38–13:53, 14:14–14:18, 14:33–14:35; about 22 minutes of the 68). Each price is a median over interleaved turns, which absorbs a stall in one turn of one arm. But the two A/A failures (0.861 and 1.120) may come from these stalls. **[AI Inference]:** the verdict should survive a clean re-run, for two reasons:
- On the slow tier the oracle with no switch is 2.4–2.8 times the classical solve, three times the bar. A standby stall would have to fall on the classical arm in at least three of five turns to make that.
- The forward passes' cost was measured separately, outside the turns.

### 14.3 What it says

- **On the slow tier ($n=32$) the network alone costs more than the problem.** The two forward passes on every piece cost more than the whole classical solve. The pieces are small, so the classical expert's eigen-solves are cheap, and a 1.1 M-parameter U-Net is not. No switch, no accuracy and no training changes this.
- **On the fast tier ($n=128$) a perfect network with no switch would pay** (0.53–0.58, under the bar and under finer cutting's 0.74–0.75). **The switch costs it the win:** two Schwarz residuals per piece roughly double the price, to 1.15–2.36.
- **[AI Inference]:** even there, TL0 at $n_c=32$ already meets E4 at 0.34–0.60 (§13.2). A network credited with the floor's accuracy beats it only where E4 is not enough. That is the question the owner's E4 already answered.

### 14.4 What the owner chooses between

The plan's stop rule applies (§5, Phase 3). These are the ways on that the measurements leave; none has been started.
1. **Close the coolant case as a negative result** and move to §10 (CO-E, the bond and the biharmonic network, classically on TL0). The classical CO-T expert, the channel cut, Danckwerts' inlet and the store stay built.
2. **Re-scope and register a new check before any data:** a smaller network, larger pieces only (the fast tier's $n\ge128$), or a cheaper switch (one residual, or Jacobi with a measured bound). Each changes a criterion fixed before this run, so it would be a new registration, not a re-reading of this one.
3. **Re-run this check unchanged on mains power with standby held,** (the owner chose a fourth way, §15) to remove the disturbance before the decision. **[AI Inference]:** it would take about 70 minutes, and the verdict is not expected to change.

---

## 15. The owner's amendment: accuracy and capacity, price as a reading (2026-11-19)

After §14 the owner decided to go on: *if the learned expert would be faster in the more complicated cases we are not at yet, and that is sound a priori, then focus on what matters here, accuracy and whether the network has the capacity to learn.* The stop rule of §5, Phase 3 (check 4) and the gate CL of §7 are amended as below. Everything else in §2–§8 stands. Nothing has yet been run under the amendment.

### 15.1 Why the price argument is sound a priori, and where it is not

- **Harder pieces are larger pieces.** $\mathrm{Pe}_{\text{piece}} = n \cdot \max \mathrm{Pe}_{\text{cell}}$, and Galerkin needs $\mathrm{Pe}_{\text{cell}} \le 2$. So $\mathrm{Pe}_{\text{piece}}=200$ needs $n \ge 100$; this is why the fast tier runs at $n=128$.
- **The classical piece solve grows faster than the node count $N=(n+1)^2$.** A sparse 2-D factorisation costs about $N^{3/2}$, and in 3-D nested dissection costs about $N^2$. **The network's forward pass grows at most like $N$** with a fixed architecture.
- **Measured on mains on 2026-11-19, one piece of `dev-block`** (a scratch measurement, not a registered record):

  | $n$ | classical piece (trial and adjoint) | one forward pass | multiply-adds per pass |
  |---|---|---|---|
  | 32 | 9.0 ms | 6.4 ms | 114 M |
  | 128 | 315 ms | 22 ms | 1,601 M |

  Two passes cost 1.42 of the classical piece at $n=32$ and 0.14 at $n=128$. **[AI Inference]:** from these two points the crossover is near $n\approx40$. The same forward passes timed on battery in §14 were about 3.7 times slower, so §14's network shares are overstated by an unknown amount.
- **Where it is not sound: the safety switch.** An estimator equivalent to the dual norm of the residual needs a solve-like application of the piece's own operator. Measured, the exact one costs 0.85–1.04 of a classical piece solve and the Schwarz estimate 1.2–2.5; Jacobi costs 0.2–0.3 but is not equivalent (§13.3). A switch that costs a solve erases the gain at every scale. **[AI Inference]:** the candidate is an estimator at $O(N)$ cost, for example a multigrid-preconditioned residual. Its equivalence constants likely worsen with Péclet number, which must be measured. This is an open design item (§15.4), not a gate.
- **[AI Inference]:** TL0 also scales like $N$, and at $n_c=32$ it met E4 at 0.34–0.60 (§13.2). A network adds value only as accuracy beyond TL0 at TL0's cost. So accuracy is the coolant's real question.

### 15.2 What changes in the phases

- **Phase 3, check 4 (the oracle ceiling):** a reading, recorded and reported, no longer a stop. §14's record stands as the first reading.
- **Phase 3 gains a capacity probe before the dataset is built** (moved forward from Phase 4, and sharpened):
  - the overfit probe runs first on the hardest corner: $\mathrm{Pe}_{\text{piece}}$ near 200 at $n=128$, thin boundary layers, high contrast;
  - a handful of pieces, labelled classically on the laptop;
  - judged by mode group, as Phase 4 already says.

  If the planned 1.1 M-parameter U-Net cannot fit them, the network is sized up (and the size recorded) before any data is built at scale. **[AI Inference]:** the first check on whether a network fits is cheaper than a dataset.
- **The dataset (§6)** spans $n=32$, 64 and 128 and $\mathrm{Pe}_{\text{piece}}$ up to 200, **weighted toward the large pieces**: at least as many $n=128$ pieces as $n=32$ ones, because that is the regime the price argument is about. Checks 1, 2, 3 and 5 still stop the plan if they fail.
- **The safety switch** stays in the design as the accuracy guarantee (gate SAFE). It is unpriced until the $O(N)$ estimator exists.
- **Phases 4–6 are unchanged**, except that the Phase 4 overfit probe now repeats the capacity probe on the frozen dataset's pieces.

### 15.3 What changes in the gates (§7)

| gate | as amended | kind |
|---|---|---|
| E4 | unchanged: the coupled error on the evaluation layouts under E4's bar, its measure fixed at registration | accuracy, **the primary gate** |
| **CL-A** (replaces CL) | where TL0 at the network's coarse size misses E4, the trained expert meets it. Per evaluation layout; arm P against arm R shows whether learning the basis is what pays | accuracy, the decisive comparison |
| CAP | the capacity probe and the Phase 4 overfit probe: on the hardest corner, every mode group's training error falls under a registered fraction of its starting value | capacity |
| SEEDS | CL-A holds for every seed of the winning arm | robustness |
| CX, K, D, SAFE | unchanged | structural |
| PR, AMORT, the oracle reading | measured and reported against the classical solve, TL0, finer cutting and $n_c=32$; **not gates**, and no "faster" claim without them | reported |

**The stopping rule, amended.** If the trained expert fails CL-A, the learned-expert track pauses as before, and Atlas's claim stays "certified composition of classical experts". A pass makes no claim on price: it says a network can learn the accuracy TL0 lacks, at a price reported beside it.

### 15.4 Open items the amendment leaves

- **The $O(N)$ switch:** a cheap estimator equivalent to the dual norm within a measured factor across Péclet numbers up to 200, with its own check before Phase 6.
- **The clean price reading:** §14 re-run on mains with standby held, folded into Phase 6's price measurement on a rented whole machine.

---

## See Also

- [[chart-operator-stage-0-rework]] — the two-level expert, its verdicts (§19.4) and what carries over (§20)
- [[chart-operator-stage-0]] — Stage 0, and §12, Stage 1 as first registered
- [[chart-operator-family-catalog]] — G0′ and G3 (§1.3), the PDE groups (§2), CO-T (§6)
- [[chart-operator-training-and-cost]] — the cost model this page's estimates follow
- [[chart-operator-design-decisions]] — the owner's earlier decisions
- [[case-study-cooling-loop-atlas-0.1]] — CS-13, the coolant circuit with lumped legs
- [[port-algebra-atlas-0.1]] — `THERM` and `ADVEC` ports
- [[integrated-ui-plan]] — §12, every feature with a page goes into the studio
- [[gap-worklist]] — W358, W362, W363
- [[00-atlas-current-state]] — what is true now
