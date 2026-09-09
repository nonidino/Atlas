# What PoC 2 actually shows, audited against the prior-art page

**Type:** Concept page — critical assessment, filed in answer to a query (folder: `Atlas 0.1/common/`)
**Status:** written **2026-09-08**, after the demo's beat re-cut. This is the adversarial reading of [[poc2-demo-and-novelty]] §1: it takes the claim the demo is built to carry and asks what survives contact with [[prior-art-and-novelty-atlas-0.1]] and with [[gap-worklist]]. **Two of its findings are measured this session** and are marked as such; the prior-art suggestions in §5 are recall and are marked `[AI Inference]`.
**Related:** [[poc2-demo-and-novelty]] · [[prior-art-and-novelty-atlas-0.1]] · [[poc2-frontwing-results]] · [[gap-worklist]] · [[interaction-horizon]] · [[master-error-bound]] · [[f1-pathmap-and-end-goal]] · [[case-study-ladder-to-f1]] · [[atlas-and-standard-dd-theory]] · [[theory-closure-audit]]

---

## 0. Why this page exists

[[poc2-demo-and-novelty]] records what the demonstration claims. This page asks
the next question — **what of it is actually new, and what is competent
engineering wearing a claim** — because the demo is the artifact most likely to
be read by somebody who will not check, and the project's own discipline is
*say the smaller true thing*.

The verdict, up front:

> **One thing in PoC 2 is novel, and it is narrower than the demo implies: the
> per-seam verdict is a static property of the declarations, measured to be so.
> Everything else on the screen is standard practice done carefully, or a
> negative result. And the novel thing is currently *unfalsified rather than
> validated*, because the compiler has never certified anything and no refusal
> has been checked against a coupling that actually fails.**

---

## 1. The one claim that survives

**The verdict is a function of the declaration and not of the design point, and
that was measured rather than asserted.**

Two measurements carry it, both already on [[poc2-frontwing-results]]:

- over the **sixteen corners of the design box, zero** produce a different
  per-seam verdict map;
- **three descriptions of one unchanged model** produce three different maps —
  2 red, 2 red, 9 red.

Together those say the thing that matters: admissibility here is decided
**before anything runs**, **per interface**, and **from declared fields alone**.
That is a different kind of object from a diagnostic you read afterwards. The
useful analogy is a **type or effect system for simulator composition** — and
§5 argues the more accurate one is *assume–guarantee contract checking*.

**This is the whole of it.** The claim is not "a compiler for physics"; it is
"the hypotheses partitioned-coupling stability theory rests on can be written as
declarable fields, and once they are, admissibility is decidable per-seam before
anything runs." PoC 2 is the existence proof that this is mechanizable at all on
a real assembly with two live interfaces. That is modest, true and checkable.

---

## 2. What is on the screen and is not novel

Stated plainly, because the re-cut moved these to the front and the front is
where an overclaim does the most damage.

| beat | what it is | novel? |
|---|---|---|
| ① the race | adjoint-based design optimisation of a coupled FSI problem | **No.** Adjoint FSI optimisation is decades old. Four design parameters is a space small enough that a gradient beating CMA-ES by $4.31\times$ is the expected result, not a finding — and the demo's own copy says the honest number is smaller than the project used to quote |
| ② the envelope | a search made to respect each component's declared validity range | **No.** Trust regions in surrogate-based optimisation, applicability domains in QSAR. **W145 was a bug in this repository**, honestly reported; a fixed bug is not a contribution |
| ③ the ablation | freeze a coupling, price it | **No.** Standard sensitivity practice. Its interest here is that it *falsified a written prediction* — which is good scientific hygiene, not novelty |
| ④ the substitution | the compiler declining a pretrained operator | **The verdict mechanism is §1's claim.** The *content* of the refusal — that a global receptive field defeats domain decomposition — is close to obvious once stated, and the demo says so in one line |
| ⑤ the balance | power residual at the seams | **No**, and [[prior-art-and-novelty-atlas-0.1]] §0 already says so |

**The re-cut sequenced these into an argument. Sequencing is presentation, not a
research contribution**, and this page exists partly so that the improved demo
does not get read as an improved result.

---

## 3. The compiler has never certified anything — measured 2026-09-08

Every case study in the package that builds with no arguments was compiled this
session and its graph-level verdict recorded:

| case | graph verdict | `unmeasured` |
|---|---|---|
| `wind_farm` | **refuse** | $L$ (W1), $C_\mu$ (W3), a norm |
| `rocket` | **refuse** | same |
| `thermal_seam` | **admit-uncertified** | same |
| `brake_thermal` | **admit-uncertified** | same |
| `thermal_strain` | **admit-uncertified** | same |
| `front_wing` | **admit-uncertified**, 9 seams, none green | same |

**No graph reaches `admit`.** And per [[gap-worklist]] **W56**, the only graph
that ever did reached it *because a module-default $C_\mu$ was being read as a
measurement* — the bug was found, in the worklist's own words, by "the first
graph that ever reached `admit`, which is the only way it could have been found."

> **So the positive branch of a three-verdict system has been exercised exactly
> once, and that once was a false positive.** A checker that has only ever said
> *no* is not yet demonstrated to be a checker.

**Two qualifications, both of which matter and neither of which rescues it.**

1. **Per-*decision* `admit` is routine and tested** — `tests/test_tier14`,
   `test_tier17`, `test_tier24` all assert individual rules clearing. It is the
   *graph-level aggregate* that never clears.
2. **It never clears for a bookkeeping reason, not a physical one.** $L$,
   $\sigma$ and $C_\mu$ **are measured** — W1 gives $L = 0.948441\pm0.0048$, W3
   gives $\tau,\sigma,\gamma$, W49 measures $C_\mu\in[0.228,1.178]$ — they are
   simply **not declared on these graphs' records**, and W56's backstop then
   forces `admit-uncertified`. That backstop is correct and was hard-won. But it
   means the distance between this framework and its own central claim is
   *provenance plumbing*, not theory.

**[AI Inference]:** this is the single highest-value move available. Measure
$L$, $\sigma$ and $C_\mu$ **on the front-wing graph** the way Tier 33 measured
$\Pi$, declare them on the records, and the front wing becomes the first graph
to reach `admit` **on merit**. That would convert §3 from a hole into the
result the whole package is missing. It is also the only way to find out whether
the positive branch has bugs of its own — and W56 is direct evidence that it
does, because the first graph to reach `admit` found one immediately.

---

## 4. No refusal has been validated, and two known ones are false alarms

This is the more serious of the two gaps.

The rules are the project's own. Nobody has yet exhibited **a seam the compiler
refused that would in fact have diverged**, or **a seam it admitted that held**.
Until one of those exists, "it judges correctly" is a statement about internal
consistency, not about the world.

**And at the seam PoC 2 exists to demonstrate, what the panel reports is two
recorded bugs in the checker.** The `wet` seam — fluid ↔ structure, 32 stations,
the interface that makes this an assembly rather than CS-10 again — carries
exactly two decertifications, and [[gap-worklist]] records both as **open false
alarms**:

- **W136**, `L2/R10/halo` — "there is no halo on this seam to be inadequate."
  $\Gamma$ is a *physical* boundary of the solid, the structure is not tiled,
  and no overlap exists on that side for anything to outrun. The rule reads
  `time_discretization` and `stencil_radius` and never asks whether the agent has
  an artificial boundary at all.
- **W138**, `L4/E7/passivity` — an **orientation** artefact. Flipping one sign
  takes the defect to **exactly $0$**. `Connection.orientation` exists for this
  and nothing reads it.

Both are deliberately left open, and the reasoning is good — a decertification
that over-fires is honest and noisy where a refusal that over-fires is terminal.
**But the consequence for the novelty claim is not softened by good reasoning:
at the headline seam, the compiler's output is its own two defects.**

> The general form is [[gap-worklist]]'s own Tier 26 lesson — *a declaration
> nobody consults is indistinguishable from a declaration nobody made* — and
> W136/W138 say it is still live in the two rules the demo puts on screen most.

---

## 5. Prior art the novelty claim has not been checked against

[[prior-art-and-novelty-atlas-0.1]] §0's verdict table **has no row for the
compiler** — [[poc2-demo-and-novelty]]'s own See Also flags this — and that
page's [AI Inference] admits the co-simulation tooling landscape has not been
systematically surveyed. Two specific bodies are close enough that the claim
should not travel outside the project until they are checked.

**[AI Inference] — the following is recall, not a survey, and is exactly what
§0's missing row should be built by checking.**

- **FMI capability flags.** The Functional Mock-up Interface has an FMU declare
  `providesDirectionalDerivative`, `canGetAndSetFMUstate`,
  `maxOutputDerivativeOrder`, `canInterpolateInputs` and others, and a
  co-simulation master reads them to select a master algorithm — declining, in
  effect, to run schemes the FMU cannot support. **That is already "components
  declare what they can do and the orchestrator decides what is valid."** The
  Atlas delta is real but narrower than *"existing tools couple; they do not
  judge"*: the decision is **per-seam**, it cites a **named stability rule**, and
  it names the **specific unmeasured constant**. That is a delta and should be
  stated as one, not as a first.
- **Assume–guarantee contracts for cyber-physical systems** — Benveniste et al.,
  *Contracts for System Design*; Nuzzo and Sangiovanni-Vincentelli. An entire
  formal tradition of "declare assumptions and guarantees, check that
  composition preserves them," **with composition and refinement theorems
  already proved**. Atlas's per-seam verdict looks like an instance of it
  specialised to numerical coupling hypotheses. Placing it there would
  **strengthen** the work — it would inherit theorems rather than re-derive them
  — while deflating the "nobody does this" framing. Both effects are worth
  having and the second is the price of the first.

**Recommended:** [[prior-art-and-novelty-atlas-0.1]] §0 gains a row reading
something like *"Return a per-seam admissibility verdict from declared component
capabilities, naming the rule and the unmeasured constant"* — **`Yes, narrowly`,
against FMI capability flags and assume-guarantee contract theory** — and §9's
sources-checked list gains both.

---

## 6. The economics are circular as of today, and this bounds what PoC 2 can claim

The motivation for composition is that learned experts are four to six orders of
magnitude cheaper than the solvers they replace. **The framework refuses learned
experts.** So the demonstrated artifact is a slower way to do what classical
solvers already do.

The demo says this, in `NOT_CLAIMED` and on screen, and the re-cut kept it. The
consequence for *novelty* is the part worth writing down: **PoC 2's contribution
cannot be "it makes search cheap." It can only be "it makes the trust question
decidable"** — which returns to §3 and §4, where the trust question is not yet
demonstrated to be decided *correctly*, only *consistently*.

---

## 7. What Tier 33 changed, which the demo undersells

[[interaction-horizon]] is the most important thing that has happened to the §1
claim and the demo carries it as a supporting panel.

Moving $\Pi$ from a **0/1 indicator** to a **measured graded sensitivity**
changes what kind of object the compiler is: from a *classifier over
declarations* into an *instrument that measures a physical property and ranks by
it*. The indicator reads $\Pi = 1.0000$ for every agent in the vault — the
certificate being unavailable rather than a fact about any of them — while
$\Pi_w$ separates the same three at $0.186 / 0.496 / 0.790$ and makes
Poseidon-T's bound **non-vacuous for the first time**.

Two consequences the demo does not draw:

1. **It is the template for §3's fix.** The reason nothing is green is three
   constants on `unmeasured`; Tier 33 just demonstrated the technique for
   measuring exactly that kind of constant, complete with a positive control
   (bitwise zero where the old rule was right) that licenses adopting it without
   re-auditing past verdicts.
2. **It moves the claim from "refuses learned operators" to "measures how global
   an operator is and prices it."** The second is a much better claim, because it
   admits a *quantitative* answer to "which learned expert could be admitted, and
   at what seam" — which is the question [[case-study-ladder-to-f1]] §2's
   climb-classically schedule actually has to answer.

---

## 8. Summary table

| # | claim | status |
|---|---|---|
| 1 | Per-seam verdict is a static property of declarations, measured over 16 box corners and 3 declarations | **Novel, narrowly. §1.** Should be positioned against FMI capability flags and assume-guarantee contracts (§5) before it leaves the project |
| 2 | Differentiable coupled design search, $4.31\times$ over CMA-ES | Not novel; expected on four parameters |
| 3 | A search made to respect declared validity envelopes | Not novel; W145 was a bug here |
| 4 | Seam ablation | Not novel; falsified a written prediction, which is hygiene |
| 5 | Power residual at seams | Not novel, and already recorded as such |
| 6 | The compiler certifying something | **Never happened.** Once, as a bug (W56). Measured 2026-09-08 |
| 7 | A refusal validated against a real divergence | **Never happened**, and the two decertifications at the headline seam are recorded false alarms (W136, W138) |
| 8 | $\Pi_w$ as a measured graded reach | **The strongest recent result**, and undersold on screen (§7) |

---

## See Also

- [[poc2-demo-and-novelty]] — what the demonstration claims; this page is its adversarial reading, and §2.0 there records the beat re-cut this audit followed
- [[prior-art-and-novelty-atlas-0.1]] — §0's table, which **still has no row for the compiler**; §5 above says what the row should say and what to check first
- [[gap-worklist]] — W1, W3, W49 (the constants that are measured but undeclared), W56 (the one `admit`, and why it was wrong), W136 and W138 (the two false alarms on the `wet` seam)
- [[interaction-horizon]] — Tier 33, §7's subject and §3's template
- [[poc2-frontwing-results]] — the sixteen-corner and three-declaration measurements §1 rests on
- [[theory-closure-audit]] — the standing audit of which hypotheses are checked, which this page's §3 and §4 are the demo-facing instance of
