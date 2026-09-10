# The PoC 2 demo, and the novelty argument it is built to carry

**Type:** Concept page — demonstration record and research positioning (folder: `Atlas 0.1/common/`)
**Status:** built **2026-09-05**, **re-cut 2026-09-08** (§2.0). The demo is `atlas/demo_frontwing/`, packaged as the standalone `poc2-frontwing-demo` branch by `scripts/w146_build_frontwing_bundle.py`. Every recorded figure quoted here and on screen is read from `out/w141/w141.json` — and, since the re-cut, `out/w153/w153.json` — at load time; none is transcribed. Graded by `tests/test_tier31_frontwing_demo.py` (**42 tests**, was 27), which checks the prose against both artifacts as well as the code, and reads the beat order back out of the page rather than out of Python.
**Related:** [[poc2-frontwing-results]] · [[prior-art-and-novelty-atlas-0.1]] · [[case-study-ladder-to-f1]] · [[f1-pathmap-and-end-goal]] · [[gap-worklist]] · [[case-study-ground-effect-atlas-0.1]] · [[case-study-wing-fsi-atlas-0.1]] · [[poc1a-frozen-expert-results]] · [[port-algebra-atlas-0.1]] · [[end-to-end-architecture-spec]] · [[interaction-horizon]] · [[control-observability]]

---

## 0. What this page is for

[[poc2-frontwing-results]] records **what was measured**. This page records **what
the demonstration claims, to whom, and what it is forbidden from claiming** — and
it exists because the demo is the first artifact in this project aimed at someone
who has not read the vault.

That makes it the piece most likely to overclaim, so the discipline is written
down here and enforced in a test:

> **Say the smaller true thing.** There is a version of this project that sounds
> much more impressive and it is not true. Every panel is written to survive
> somebody checking it.

`tests/test_tier31_frontwing_demo.py::test_the_explainer_does_not_use_the_vocabulary_this_project_has_not_earned`
asserts the absence of `revolutionary`, `breakthrough`, `unprecedented`, `solves
the problem`, `certified accurate`, `replaces cfd` and `guaranteed` from every
user-facing string, and
`::test_every_number_the_explainer_quotes_is_in_the_artifact` re-derives each
figure in the prose from `out/w141/w141.json`, with
`::test_the_reach_prose_quotes_the_artifacts_own_numbers` doing the same against
`out/w153/w153.json` for §2④b. A stale sentence fails in CI rather than being
read off a screen by somebody who believes it.

---

## 1. The novelty argument, in the order it has to be made

[[prior-art-and-novelty-atlas-0.1]] is the rigorous version and it is unchanged.
This is the version that fits on a screen, and its structure is deliberate: the
first two points are what the audience already believes, the third and fourth are
borrowed, and **only the fifth is the claim**.

| # | claim | status |
|---|---|---|
| 1 | The problem is not that we cannot simulate a car — it is that one evaluation costs hours, so teams evaluate hundreds of designs rather than millions | **not novel.** Existing industrial practice; preCICE, ANSYS System Coupling, Modelica |
| 2 | Replacing the solver with a trained model has two holes: a car is not only fields (a torque map is not a field), and gluing trained models at their boundaries has no stability theory | **not novel**, and §2.1 and §2.3 of the prior-art page say so with citations |
| 3 | A shared connector — every joint an effort–flow pair whose product is power — so $O(K)$ port declarations replace $O(K^2)$ pairwise adapters | **not novel.** Bond graphs, 1959; port-Hamiltonian systems. Adopted for exactly that reason |
| 4 | One diagnostic at every joint: is energy appearing or disappearing at the seams | **not novel.** Power-bond energy-residual analysis is standard in co-simulation |
| 5 | **A compiler that reads what each component declares about itself and returns a per-seam verdict — certified, runs-but-uncertified with the specific unmeasured constant named, or refused with the rule named** | **this is the claim** |

**The one-sentence form of why 5 matters:** a fast simulator that is confidently
wrong is worse than a slow one that is right, because an optimiser will find and
exploit precisely the places where it is confidently wrong. Existing coupling
tools couple; they do not judge.

**[AI Inference]:** the closest published thing to point 5 is the *conservative*
vs *consistent* mapping distinction in preCICE and the energy-residual
step-control in the co-simulation literature — both of which are per-quantity
checks a user opts into, not a per-seam verdict the framework returns unasked.
The distinction being drawn is between a diagnostic and a *refusal*, and it has
not been checked against a systematic survey of the co-simulation tooling
landscape. It should be before this framing is used outside the project.

---

## 2. The five beats, and what each is evidence of

Each is on screen in two voices, and each is a **measurement this project already
made**, staged so a viewer can watch it happen rather than read that it did.

### 2.0 The order, and why it changed on 2026-09-08

**The order is the argument, and the first order was wrong for the audience.**

It opened on the substitution refusal — §2④ below. That is §1's point 5, the only
row in that table this project can claim, and it is the right first thing to show
a specialist. It is also a **negative result**, so the first thing a reader who
has not read this vault saw was the machine saying *no*. §0 says this demo is the
first artifact here aimed at someone who has not read the vault; on that audience,
opening on a refusal reads as a project without a result rather than a project
with a check.

**The positive result was already in the demo, already measured, sequenced third
and fourth**: a real coupled multiphysics design search running end to end with
gradients on two live seams, finding $+19.3\%$ downforce inside both ceilings and
declined eight times by an expert's own validity check on the way, and beating
CMA-ES by $4.31\times$ in wall-clock on the same objective and budget.

So the sequence is now:

| | beat | what it is doing in this position |
|---|---|---|
| ① | **the race** | the win, and the thing a non-specialist can watch happen: a bar climbing faster |
| ② | **the envelope** | the receipt for ①. The immediate sceptical question — *did it cheat to get that number?* — answered with $19.3\%$ enforced against $27.5\%$ unenforced-and-invalid |
| ③ | **the ablation** | where the gain came from, joint by joint |
| ④ | **the substitution** | the trust layer: how you know when *not* to believe ①, shown correctly declining a case it should decline |
| ⑤ | **the balance** | the ledger, for anyone checking the arithmetic |

**What the re-cut is not.** Nothing was re-measured, no figure moved, and no
caveat was removed or weakened. §3 still carries the tension whole and now says
so in the place a reader meets it: ①'s $4.31\times$ is one *search method* against
another **on the same classical solvers**, not a claim about the solvers, and the
orders of magnitude still arrive only with learned experts that ④ is the framework
refusing. The refusal is not softened. It stops being the headline and becomes the
reason to trust the headline.

**Enforced rather than described.** The order lives in `explain.BEATS` and nowhere
else: the tab nav, the panel sequence and every panel's heading and ordinal are
built from that tuple at load time. Before the re-cut the nav was a second
hand-written list with its own ordinals — and the nav is what a viewer clicks, so
the stale copy would have been the authoritative one.
`::test_the_pages_own_tab_nav_produces_the_new_beat_order` runs the page's own
`tabsOf` through node and asserts the keys it returns, and
`::test_the_refusal_is_resequenced_and_not_softened` asserts the hard sentences
survived the move.

### ① The race — evidence about the gradient, quoted the way W143 says

Adam on the adjoint against CMA-ES on the same objective, one evaluation each in
turn, **with the live march held for the duration** — this beat reports a
wall-clock ratio, and a 12 fps march in the background would be measured instead
of the search. Measured: with the march running, the race reached 4% of its budget
in 25 seconds; held, it finishes in 75.

**Both columns are warmed before either clock starts**, and this is not optional.
Unwarmed, a three-step race reads a $25\times$ adjoint premium where a warmed one
reads $5.1$ — the same error `stage_cost` made when it published $9.47$ against a
warmed re-measurement's $5.4$–$5.8$. The warm-up is timed separately and shown.

**The live budgets are scaled, not chosen**: the recorded run's $30 : 200$ ratio
is preserved at $14 : 93$ and the horizon is shortened instead. Picking them
independently would mean picking them until the answer came out the way the panel
prefers. Below about nine gradient iterates Adam has not turned and the
population column wins on wall-clock; the screen says so.

Recorded: $4.31\times$ in wall-clock, $22.2\times$ in rollouts, quality
$1.003\times$. The first is quoted first and the second is labelled flattering,
which is **W143** applied rather than described.

**Why this is the opening beat.** Differentiability, not raw speed, is the
property the whole approach rests on, and this is the measurement of what having
it buys — on a coupled multiphysics problem with both interfaces live, which is
the part that is not routine. It is also the only beat whose value a
non-specialist can read off the screen without being told what to look for.

### ② The envelope, on and off — evidence that abstention is load-bearing

**W145**, staged as one click. *Put it on the floor* sets the free ride height and
the spring rate to the bottom of the box; that design settles below
`Suspension`'s declared floor.

| | with the check | without it |
|---|---|---|
| live, at that design | stops at macro-step 0, names the expert | marches to $h = 0.0067$ against a floor of $0.0703$, reports the largest downforce on the screen |
| recorded search | $+19.3\%$, 8 of 31 gradient iterates declined | $+27.5\%$, 0 declined |

The live unenforced march lands at $J \approx 0.290$, within a percent of the
recorded pre-W145 optimum's $0.2886$ — so the fiction the fix removed is
reproducible on demand, which is a better demonstration than describing it.

Two things the panel is careful about:

- **`declined` and `infeasible` are separate rows.** 39 of the population's 200
  evaluations were declined by an expert; 66 were infeasible. The other 27 were
  inside the model and over a ceiling. Reporting 66 as declines would overstate
  W145 by a factor of $1.7$.
- **The live optimiser does not reproduce the recorded search's direction**, and
  the panel says so rather than promising a demonstration it does not deliver:
  it is a warm-started twelve-step estimator and it ascends differently from the
  120-step one. That is itself worth knowing, and it is why the recorded columns
  sit beside it.

**Why this generalises.** [[f1-pathmap-and-end-goal]] §6 calls abstention *"the
single largest unsolved problem on the path"*: with 15–20 components the
probability that at least one is outside its training regime on a novel design
approaches 1, and a novel design is the entire point. This is a small working
instance of it, on the path where it matters — the one a search walks.

### ③ The seam ablation — evidence that the couplings are the physics

Freeze each seam and read the downforce. $k \to \infty$ recovers CS-12;
$E^* \to \infty$ recovers CS-10; statically, those limits agree with their parents
to $3.37\times10^{-10}$ and $6.36\times10^{-8}$, which is what earns the word
*assembly*.

Recorded, the suspension seam is worth $0.49\%$ at the reference design and
$8.16\%$ at the optimum, and the elastic column at the optimum **declines** — a
rigid wing there makes more downforce and drives itself through the floor.

**One implementation change was made for the demo and it is a correction, not a
port.** `ablation.freeze_each_seam` **measures** the height its frozen column is
pinned at, from the live column it marches first, instead of taking it as an
argument. The caller's instantaneous ride height is not the height a fresh
rollout settles to: measured at the reference design, a twenty-step march ends at
$0.236268$ against the sixty-step settled $0.231339$, an error of $2.13\%$ — and
pinning at the wrong one measures a transit rather than the seam. The driver
already passed the settled value explicitly; the demo removes the opportunity to
pass the wrong one. A hint may still be supplied and its error is reported beside
the measurement.

### ④ The substitution refusal — evidence for point 5, and a negative result

**In fourth place its job is different from what it was in first, and the copy
says so.** A search that wins on the clock is only worth having if you know when
*not* to believe it; this is that check, running on the same graph, correctly
declining a case it should decline. Nothing about the mechanism on screen
changed — same three declarations, same verdicts, same W93 provenance, same
six-window requirement.

The same graph, compiled three times, with three *descriptions* of the fluid
expert. Nothing about any model changes.

| description | verdict | what appears |
|---|---|---|
| `WindowNS`, the classical incumbent | 2 seams red, 7 amber | — |
| Poseidon-T, as `poseidon_capabilities` declares it | 2 red, 7 amber | `L2/R10/W60`, `L2/R10/halo`, `L4/R2b/W46` at every fluid seam |
| Poseidon-T with `elliptic_subsolve = embedded` | **9 red** | `L2/R10` refuses |

**This beat needs no weights**, and that is the load-bearing observation rather
than a convenience: the verdict is a function of the capability record, so a
20.8 M-parameter checkpoint and a record with the same declared fields get the
same answer. It is why the standalone bundle is 4.9 MB and touches no network.

The measurement it rests on is **W93** and is quoted with its provenance rather
than re-run: `probe.support_reach` found Poseidon-T's response nonzero in every
one of the 128 seam cells, 64 cells from the poke, where its record declares a
stencil radius of 2 — a factor of 32, at every amplitude from 1 to $10^{-3}$.

The third row is **declared, not measured**, and the panel says so in the same
place it shows the refusal: **W60** records that `elliptic_signature` measures
*non-normality* and was read as measuring *globality*, and the two come apart for
a self-adjoint operator, so this field is genuinely undeclarable for a learned
operator.

**And it needs the six-window column.** On the single-window referent the fluid
family has one member, nothing has been decomposed, and `L2/R10`'s premise does
not hold — so the rule correctly stays quiet however the expert is declared. That
is **W114**'s premise check working. The payload carries a sentence saying which
column it is in, because a quiet rule read as a passing one would invert the
beat's meaning.

> **What this must not be read as saying.** Not "neural operators do not work."
> Poseidon-T is a good model. It is that *this framework cannot certify a domain
> decomposition whose agents are globally receptive*, which is nearly every
> pretrained operator, because a global receptive field is a fact about the
> architecture and not about the physics. The programme's response is scheduled
> rather than hidden ([[case-study-ladder-to-f1]] §2): climb classically, price
> each learned substitution one seam at a time.

#### ④b Underneath the three verdicts: the measured reach ([[interaction-horizon]])

**Added by the re-cut, folded into this beat rather than given a sixth.** The
point of the measurement is that it sits *under* these three verdicts and says
why they are what they are; a separate tab would split one trust-layer story in
two.

Until Tier 33 the rule behind the verdicts rested on a **0/1 indicator**: does a
component's influence reach past the cut, or not? At one exchange per macro-step
that reads $\Pi = 1.0000$ for **every** agent in this vault, the plainest
classical solver included — which is the certificate being *unavailable* rather
than a fact about any of them. $\Pi_w$ replaces the indicator with the
sensitivity it was standing in for, and the three separate:

| measured agent | $\Pi$ | $\Pi_w$ | influence past the cut | which declaration above |
|---|---|---|---|---|
| `WindowNS`, exposed — the local classical solver | $1.0000$ | $\mathbf{0.186}$ | **bitwise zero** past $b = 20$ cells, its declared $\rho s = 2\times10$, over all $13{,}924$ far cells | `windowns` |
| `WindowNS`, embedded — the same solver carrying its own elliptic part | $1.0000$ | $\mathbf{0.496}$ | still $\mathbf{26\%}$ of its peak at $b = 20$ and $12\%$ at $b = 60$ — **still falling, slowly** | *none — the control* |
| **Poseidon-T** | $1.0000$ | $\mathbf{0.790}$ | still $\mathbf{61\%}$ at $b = 20$ and $\mathbf{61\%}$ at $b = 60$ — **it has stopped falling off** | `poseidon_declared`, `poseidon_probed` |

Read from `out/w153/w153.json` by `engine.load_horizon`, on the same rule as
`load_recorded`: never re-run here, never transcribed, and **absent, the beat
shows its three verdicts without the bars** rather than a number with no run
behind it.

**Three things this panel is careful about.**

- **It changes no verdict and could not have.** Poseidon-T is refused before the
  measurement and after it. The refinement returns the indicator *exactly* where
  the indicator was right — at halo 1, where the overlap is narrower than the
  domain of dependence, $\Pi_w/\Pi = 1.000000$ — so no past verdict can have
  moved silently. What it upgrades is the *reason*: the compiler is not
  pattern-matching "neural network, therefore no", it measured a graded property
  and got the ordering right.
- **The bitwise-zero claim is taken over every probed state, not one.**
  $d_{\text{eff}}(0)$ moves between $19$ and $20$ across the three spin-up states
  while `exactly_zero` holds in all of them, so the panel quotes the bitwise
  property at the declared $b = 20$ rather than a per-state fit.
- **The figure beside each bar is a decay, not a magnitude — and that is the one
  way this panel could have misled.** The far-field **Frobenius** norm is in each
  agent's own units and it ranks the three the *other way round*: $76.07$ for the
  embedded classical solver against $9.78$ for Poseidon-T, while Poseidon-T is
  the one whose influence does not decay. A bigger bar beside a smaller number
  reads as a broken panel. What is shown is `profile_max` at $b = d$, the
  response normalised by its own peak, taken as the **worst** over the probed
  states — $0\%$, $26\%$, $61\%$, monotone with $\Pi_w$ — and
  `::test_the_figure_beside_each_bar_ranks_the_same_way_the_bar_does` asserts
  both that ordering *and* that the Frobenius really does invert, so the test
  keeps meaning something if anyone swaps it back.
- **It is sampled at two distances, because one would let the panel call both
  non-local rows flat and only one of them is.** At $b = 60$ the embedded
  classical solve has halved again to $12\%$ while Poseidon-T sits at $61\%$,
  unmoved. That difference is the whole reason the two bars are at different
  heights, so the screen reads its verdict off the pair — *still falling,
  slowly* against *it has stopped falling off* — rather than asserting either.
  This is Tier 33's "three regimes from one instrument" reduced to the one
  sentence a non-specialist needs.
- **The mapping from bars to columns is deliberately not one-to-one.** Both
  Poseidon rows light the same bar, because *the verdict is a property of the
  declaration and the reach is a property of the model* — which is the beat's own
  point seen from the other side. The embedded classical row lights none: it is
  the control that separates *global* from *global because it is learned*, and
  labelling it as one of the three columns would claim a declaration this screen
  does not offer.

**[AI Inference]:** the reason this belongs in the demo at all is that it is the
first thing on the screen that makes the refusal look like a *measurement of a
continuum* rather than a category judgement, and a reader deciding whether to
trust ① is being asked to trust exactly that distinction. It has not been tested
on a reader who has not read this vault, which is the only test that would settle
whether it lands.

### ⑤ The power residual — the one diagnostic that spans the roadmap

Three accountings of $\mathcal R(t)$, live. Recorded over 480 macro-steps:
$1.0000$ (static, the normalising level) $\to 5.93\times10^{-2}$ (with the motion
terms) $\to 3.30\times10^{-2}$ (with the half-step term), and $3.32\times10^{-8}$
once settled.

The live panel makes two admissions the recorded one does not need. The
normalising level is a **running maximum still growing** over the first frames,
so those are not a reading; and **after a march has settled all three accountings
agree**, because there is nothing left moving for the motion terms to account
for. The gap between them exists only during the transient, which is what
*reset* replays.

---

## 3. What the demo is forbidden from claiming

Reproduced from `explain.NOT_CLAIMED`, which is what the page renders and what
the tests read:

1. **Certified at a fixed shape — and not certified when it rides.** At a fixed
   shape all nine seams are green and the graph verdict is **`admit`**, the
   first assembly this project has certified on merit. Declare the interface to
   move and the same software **refuses the two seams on the wing's surface**
   at `L2/InterfaceMotion`, leaving the seven inside the air green. That
   refusal is a **named hole** and not a defect: nothing here certifies a joint
   whose geometry is a function of the answer. **A racing wing rides**, so the
   machinery works on the problem it can state and the problem it cannot state
   is the one the sport actually has.

   **How it got there, because the route qualifies the result.** Until
   2026-09-08 the reason nothing was green was *bookkeeping* — $L$, $\sigma$
   and $C_\mu$ had been measured in tier 0 on another graph and were not
   declared on this one, so W56's backstop forced `admit-uncertified` and the
   refusal said nothing about this wing. **W157 measured all three here**
   ([[poc2-novelty-audit]] §3.1) and left five named rules standing. **Tier 35
   took three**: `L2/C2` got the number nobody had measured (**W159**:
   $5.622\times10^{-5}$, chi-weighted against a single-window monolith, tight to
   $0.05\%$, identity closing to $10^{-10}$, zero-cut control at exactly $0$);
   `R10/halo` was scoped to the agents the decomposition actually cuts
   (**W136**); and the seam operator became **oriented**, taking both surface
   seams' passivity defect to exactly zero (**W138**) — checked against the
   case's own residual Jacobian rather than asserted. **Tier 39 took the last
   two**, and both were checker defects Tier 35 had already diagnosed and
   deliberately left open: `L2/R10` read a two-line algebraic spring as hiding a
   pressure solve and now checks `stencil_radius` (**W160**, censused at six
   agents across four graphs, every one algebraic); `R12` named the *elasticity*
   agent as enforcing incompressibility when it is not even a subdomain of the
   partition of unity, and `_enforcing_agents` now intersects with the family
   the constraint comes from (**W161**, which also takes `wing_fsi` to `admit`).

   **So of the five rules that stood after W157, four were defects in the
   checker and one was a missing number. Zero were facts about this assembly**
   — which is §4's point arriving with a count, and it cuts both ways: a
   compiler that reaches `admit` by having its own false alarms removed has
   shown its declarations are checkable, not that its refusals are earned.
2. **This is not faster than the solvers it is made of.** Every expert is
   classical, and beat ①'s $4.31\times$ is one *search method* against another on
   the same solvers — not a claim about the solvers. The four-to-six orders of
   magnitude in the unit economics of search arrive only with learned experts —
   and beat ④ is the framework refusing the most obvious candidate. **That
   tension is the honest state of the programme, not a footnote**, it is on the
   screen, and **moving that beat from first to fourth does not soften it**
   (§2.0).
3. **The physics is a demonstration.** $Re_c = 125$, one grid, one incidence, a
   porous plate with no Kutta condition, a rolling road with no viscous gap
   choke, Q1 elements that lock in bending, and $E^*$ a knob in flow units.
4. **The decisive question is unanswered.** Composition error against interface
   count has never been measured at any $N$ ([[f1-pathmap-and-end-goal]] Claim B,
   rung 9). Nor has rung 4, whether a learned expert transfers between scenarios.

---

## 4. The package

`scripts/w146_build_frontwing_bundle.py` assembles the standalone
`poc2-frontwing-demo` branch: **4.9 MB**, one command to run, macOS / Linux /
Windows, Python 3.10–3.12, and **no network access at run time or install time
past `pip`**.

| | PoC 1a bundle | this one |
|---|---|---|
| size | ~90 MB | **4.9 MB** (4.3 before `w153.json` joined it) |
| third-party weights | Poseidon-T, 83 MB, CC-BY-NC-4.0 | **none** |
| fetched at install time | `scOT` from a pinned upstream archive | **nothing** |
| what makes the difference | — | every expert is a classical solver, and beat ④ hands the compiler a *declaration* rather than weights |

The absence of a licence encumbrance is a real consequence: [[poc1a-frozen-expert-results]]
records **W120**, that a permissively licensed donor is what a production version
would need, and this bundle sidesteps the question entirely by not carrying
anyone's weights.

Both solvers resolve through `ATLAS_BUILD_REPO`, vendored at `vendor/src/atlas/`
in the exact layout `window_ns.load_reference` and `thermal_strain.load_solvers`
expect, so **no source file is modified for the bundle** — which is what lets the
tests in it be the same files, character for character, as the ones here.
Verified **from a cold clone of the built branch**, not from the working tree: `pytest tests/` inside it passes **92 tests**, and
`run.py --check` loads both solvers, compiles all three substitution columns and
marches.

`out/w141/settled.npz`, `out/w141/w141.json` and — since the re-cut —
`out/w153/w153.json` travel with it. **The demo never writes an artifact and
never falls back to a hard-coded copy of a number in one**: if `w141.json` is
absent the recorded columns say so, and if `w153.json` is absent §2④ shows its
three verdicts without the reach bars, which is the state that beat was in
before the measurement existed. That rule is
[[poc1-retrospective-and-hybrid-roadmap]]'s lesson — a milliseconds-per-step
table in this project's own documents was wrong by a factor of four the next day
on unchanged code — and it is asserted by
`::test_the_loader_says_so_rather_than_inventing_a_number_when_it_is_absent` and
`::test_the_reach_loader_says_so_rather_than_inventing_a_number`.

**`w153.json` is carried whole, ~0.46 MB, rather than reduced to the six figures
the screen quotes.** That is the whole of the size increase from 4.3 MB, and it
is not laziness: a six-number excerpt living in the demo *is* the hard-coded copy
the rule above bans. A number and the run that produced it travel together or the
number rots.

---

## 5. What building it turned up

Four things, none of which was the point of the exercise:

- **W147** — the message `check_envelopes` raised bundled two physically
  different violations under one sentence naming both bounds, so a wing at
  $h = 0.348$ above a free height of $0.20$ was reported *"against a floor of
  0.0703"*. The condition is unchanged and no number moves; the two now raise
  separately and say which bound was crossed. Found by pressing the demo's own
  button.
- **W148** — the W145 fix landed as **three** implementations of one rule: a
  shared `check_envelopes` in `front_wing`, and inline copies in `wing_fsi` and
  `ground_effect`. Three copies of one rule is the exact shape that produced
  W145. `tests/test_tier31` asserts the property rather than the name until they
  are unified.
- **The ablation's operating point could be passed wrong**, and now cannot (§2③).
- **The race was measuring contention**, and the fix — hold the march — is the
  same discipline the PoC 1a measurement panel already used and this one had not
  inherited.

The first two are on [[gap-worklist]]; the second two are fixed in place.

**And one more, from the 2026-09-08 re-cut** (§2.0): **the beat order was written
down twice.** `explain.BEATS` decided it, and `static/index.html` carried a
second hand-written `TABS` array with its own ordinals and its own words. Every
test read the tuple, so re-sequencing the beats in Python would have left the nav
— *the thing a viewer actually clicks* — showing the old order and the old
labels, with a green suite. This is [[gap-worklist]] W148's shape exactly: three
copies of one rule, one of which nothing consults. Fixed in place rather than
opened: the nav, the panel sequence and every heading are derived from the tuple
at load time, and `::test_the_pages_own_tab_nav_produces_the_new_beat_order`
executes the page's own `tabsOf` through node so the assertion is made against
the screen rather than against the source of truth it is supposed to be checking.

> **[AI Inference]:** the general form is that *a test which reads the same
> variable the code reads is not a test of the rendering*. This vault's demo
> tests have been written against Python objects throughout, and the two
> stale-ordinal sentences elsewhere in the page — caught here only by counting
> the numeral glyphs — suggest the same gap exists in whatever else the template
> asserts about itself. Not audited.

---

## See Also

- [[poc2-frontwing-results]] — what was measured, and the limitations section this page's §3 is drawn from
- [[prior-art-and-novelty-atlas-0.1]] — the rigorous version of §1, with sources. **Its §0 table predates the compiler** and should gain a row for the per-seam verdict
- [[case-study-ladder-to-f1]] — §1.1's W93 framing, and the classical-first revision beat ④ is evidence for
- [[interaction-horizon]] — Tier 33, the source of §2④b's three reach figures and the artifact the panel reads
- [[control-observability]] — Tier 32, whose venue confound Tier 33 settled; the shape §2④b's measurement was made in
- [[f1-pathmap-and-end-goal]] — Claim B and the falsification criteria §3 point 4 refers to
- [[gap-worklist]] — W93, W114, W120, W136, W138, W143, W145, W147, W148
- [[poc1a-frozen-expert-results]] — the demo this one is modelled on, and W120's licence problem this one does not have
