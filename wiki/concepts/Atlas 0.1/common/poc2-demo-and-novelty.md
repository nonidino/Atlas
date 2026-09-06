# The PoC 2 demo, and the novelty argument it is built to carry

**Type:** Concept page — demonstration record and research positioning (folder: `Atlas 0.1/common/`)
**Status:** built **2026-09-05**. The demo is `atlas/demo_frontwing/`, packaged as the standalone `poc2-frontwing-demo` branch by `scripts/w146_build_frontwing_bundle.py`. Every recorded figure quoted here and on screen is read from `out/w141/w141.json` at load time; none is transcribed. Graded by `tests/test_tier31_frontwing_demo.py` (27 tests), which checks the prose against the artifact as well as the code.
**Related:** [[poc2-frontwing-results]] · [[prior-art-and-novelty-atlas-0.1]] · [[case-study-ladder-to-f1]] · [[f1-pathmap-and-end-goal]] · [[gap-worklist]] · [[case-study-ground-effect-atlas-0.1]] · [[case-study-wing-fsi-atlas-0.1]] · [[poc1a-frozen-expert-results]] · [[port-algebra-atlas-0.1]] · [[end-to-end-architecture-spec]]

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
figure in the prose from `out/w141/w141.json`. A stale sentence fails in CI
rather than being read off a screen by somebody who believes it.

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

### ① The substitution refusal — evidence for point 5, and a negative result

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
same answer. It is why the standalone bundle is 4.3 MB and touches no network.

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

### ④ The race — evidence about the gradient, quoted the way W143 says

Adam on the adjoint against CMA-ES on the same objective, one evaluation each in
turn, **with the live march held for the duration** — beat ④ reports a wall-clock
ratio, and a 12 fps march in the background would be measured instead of the
search. Measured: with the march running, the race reached 4% of its budget in
25 seconds; held, it finishes in 75.

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

1. **Nothing here is certified.** No seam is green and none ever has been in this
   package: $L$, $\sigma$ and $C_\mu$ are unmeasured (W1, W3, W49) and a
   non-empty `unmeasured` list has forced `admit-uncertified` since **W56**.
2. **This is not faster than the solvers it is made of.** Every expert is
   classical. The four-to-six orders of magnitude in the unit economics of search
   arrive only with learned experts — and beat ① is the framework refusing the
   most obvious candidate. **That tension is the honest state of the programme,
   not a footnote**, and it is on the screen.
3. **The physics is a demonstration.** $Re_c = 125$, one grid, one incidence, a
   porous plate with no Kutta condition, a rolling road with no viscous gap
   choke, Q1 elements that lock in bending, and $E^*$ a knob in flow units.
4. **The decisive question is unanswered.** Composition error against interface
   count has never been measured at any $N$ ([[f1-pathmap-and-end-goal]] Claim B,
   rung 9). Nor has rung 4, whether a learned expert transfers between scenarios.

---

## 4. The package

`scripts/w146_build_frontwing_bundle.py` assembles the standalone
`poc2-frontwing-demo` branch: **4.3 MB**, one command to run, macOS / Linux /
Windows, Python 3.10–3.12, and **no network access at run time or install time
past `pip`**.

| | PoC 1a bundle | this one |
|---|---|---|
| size | ~90 MB | **4.3 MB** |
| third-party weights | Poseidon-T, 83 MB, CC-BY-NC-4.0 | **none** |
| fetched at install time | `scOT` from a pinned upstream archive | **nothing** |
| what makes the difference | — | every expert is a classical solver, and beat ① hands the compiler a *declaration* rather than weights |

The absence of a licence encumbrance is a real consequence: [[poc1a-frozen-expert-results]]
records **W120**, that a permissively licensed donor is what a production version
would need, and this bundle sidesteps the question entirely by not carrying
anyone's weights.

Both solvers resolve through `ATLAS_BUILD_REPO`, vendored at `vendor/src/atlas/`
in the exact layout `window_ns.load_reference` and `thermal_strain.load_solvers`
expect, so **no source file is modified for the bundle** — which is what lets the
tests in it be the same files, character for character, as the ones here.
Verified: `pytest tests/` inside a freshly built bundle passes 79 tests, and
`run.py --check` loads both solvers, compiles all three substitution columns and
marches.

`out/w141/settled.npz` and `out/w141/w141.json` travel with it. **The demo never
writes the artifact and never falls back to a hard-coded copy of a number in
it**: if it is absent, the recorded columns say so. That rule is
[[poc1-retrospective-and-hybrid-roadmap]]'s lesson — a milliseconds-per-step
table in this project's own documents was wrong by a factor of four the next day
on unchanged code — and it is asserted by
`::test_the_loader_says_so_rather_than_inventing_a_number_when_it_is_absent`.

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

---

## See Also

- [[poc2-frontwing-results]] — what was measured, and the limitations section this page's §3 is drawn from
- [[prior-art-and-novelty-atlas-0.1]] — the rigorous version of §1, with sources. **Its §0 table predates the compiler** and should gain a row for the per-seam verdict
- [[case-study-ladder-to-f1]] — §1.1's W93 framing, and the classical-first revision beat ① is evidence for
- [[f1-pathmap-and-end-goal]] — Claim B and the falsification criteria §3 point 4 refers to
- [[gap-worklist]] — W93, W114, W120, W136, W138, W143, W145, W147, W148
- [[poc1a-frozen-expert-results]] — the demo this one is modelled on, and W120's licence problem this one does not have
