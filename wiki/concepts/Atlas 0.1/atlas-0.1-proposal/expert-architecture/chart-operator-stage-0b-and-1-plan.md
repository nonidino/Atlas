# Chart operator after Stage 0: the coolant plan (CO-T on laminar cold plates), then the rest of Stage 1

**Type:** Concept page — **plan**, agreed with the owner; nothing on it is built, trained or registered (folder: `Atlas 0.1/atlas-0.1-proposal/expert-architecture/`)
**Status:** written 2026-11-18 from a planning conversation with the owner, who made every decision in §2. **No code, no data, no training.** The one number measured here is a stand-in for the label cost (§9), and it is labelled as one. Every other cost is **[AI Inference]**, with its arithmetic. The gates in §7 are drafts: their bars are registered in Phase 3, before any result, and never moved afterwards.
**Follows:** [[chart-operator-stage-0-rework]] §20 · [[chart-operator-stage-0]] §12 · **Family:** [[chart-operator-family-catalog]] (CO-T, class G0′) · **Case:** [[case-study-cooling-loop-atlas-0.1]] · **Worklist:** [[gap-worklist]] W362 (open), W363 (done) · **UI:** [[integrated-ui-plan]] §12

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
| expert store | one store, a card per trained model; **failed experts get cards too** |
| publishing weights | **yes**, in principle. Where and under what licence is decided when the first weights are ready, and nothing is pushed without asking |
| Stage 0b (high-contrast diffusion) | **deferred** until after the coolant |
| CO-E, the thermoelastic bond, the biharmonic network | **after the coolant, classical**: built on TL0, trained only if their own test without training shows a job for a network |
| Helmholtz | **out**, noted as a later candidate for a learned coarse basis |

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
- **[AI Inference]:** in a plan view, depth-averaged laminar flow obeys the Hele-Shaw equation $\nabla\cdot\big(\tfrac{h^2}{12\mu}\nabla p\big)=0$, $\bar{\mathbf u}=-\tfrac{h^2}{12\mu}\nabla p$. That is group A, which CO-D already solves. So the diffusion expert could supply the flow and the transport expert carry the heat on it: two Atlas families chained on one device. That needs checking before it is relied on.

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
3. **The seams must carry the flow's heat.** At Pe 20 and 200 the exact port matrix's symmetric part is indefinite (relative smallest eigenvalue $-4.4\times10^{-2}$ and $-1.5\times10^{-1}$, catalog §6). Value plus diffusive flux is not a passive pairing, so the seams need the `ADVEC` port with the advective flux ([[port-algebra-atlas-0.1]]). That is classical framework work, and it comes before anything learned.
4. **A high-contrast wall.** Water against aluminium is a conductivity ratio of about 300. So the coolant case also contains part of Stage 0b's question.
5. **The labels must be trusted.** Galerkin on a grid oscillates once the cell Péclet number $\lvert\mathbf u\rvert h/\alpha$ exceeds 2. At $\mathrm{Pe}_{\text{piece}}=200$ that needs $n\ge 100$. References are computed at $n=128$ and checked at $n=256$ (§6, check 2).

---

## 5. The phases

Each phase ends at a stop point. Each feature with a page goes into the studio, with a handler in `atlas/studio/protocol.py`, a view, a `tests/test_studio_*.py` test and a page walk ([[integrated-ui-plan]] §12). The pinned modules (the R2 and TL pins) are not edited: new code goes in new modules.

**Phase 1: the classical transport expert on pieces** (laptop; no learning)
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
- Run on held-out cold-plate layouts drawn in the studio (§6).
- **Stop if** any classical option meets E4 under the price bar. Then the coolant needs no network, and that is reported as CO-D's result was.

**Phase 3: the dataset frozen, the gates registered** (laptop, then the owner's approval)
- The dataset is built and must pass §6's five checks, including the oracle ceiling. Then it is hashed and frozen.
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
- **Held-out problems:** real cold-plate layouts drawn in the studio with its Channel tool (a serpentine plate, parallel channels, a block beside a channel). They are fixed before training starts, never generated and never trained on.
- **The learning curve:** nested subsets of the frozen set (25%, 50%, 100%), so the curve needs no new data.

**The five checks the dataset must pass before it is frozen:**
1. **Coverage:** the registered ranges of Péclet number, shape, contrast and step are filled as declared.
2. **Label quality:** a sample's labels change by less than a stated tolerance from $n=128$ to $n=256$.
3. **The control:** TL0 misses E4 on the held-out layouts (Phase 2). If it does not, stop.
4. **The oracle ceiling:** the best classical coarse basis on each piece stands in for the network. Priced at TL0's cost plus an estimated forward pass, it must beat every classical option at equal price. If even the oracle cannot, no network can.
5. **Loss–gate alignment:** random perturbations of an untrained network must be ranked alike by the training loss and by the coupled error on the held-out layouts. This is the check that would have caught P6 before training.

---

## 7. The gates and the stopping rule (drafts; bars registered in Phase 3)

| gate | what it asks | kind |
|---|---|---|
| CX, K, D | the coarse correction exact and the kernel (a uniform temperature) reproduced, whatever the weights | structural, carried over |
| SAFE | on every piece, the kept modes' residual is at most TL0's (the switch of §4) | structural |
| E4 | the coupled error on the held-out layouts under E4's bar | accuracy |
| CL | the trained expert beats the **best classical option at equal measured price** (raised $n_c$, the classical multiscale basis, finer cutting) | the decisive comparison |
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
- Stage 0's eleven arms, the forms runs and TL-s0/s1 are backfilled from their existing records.
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
