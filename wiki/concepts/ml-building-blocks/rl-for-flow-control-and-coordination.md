# Reinforcement Learning for Flow Control, and RL's Three Roles in Atlas

**Type:** Core Concept — ML Building Block / Program Assessment
**Status:** Written 2026-08-21, prompted by the **HydroGym** publication in *Nature*. Assesses an external tool against [[f1-pathmap-and-end-goal]]'s ladder and formalizes the one role for RL that the wiki had not yet written down.
**Sourcing caveat:** the *Nature* article is paywalled and was **not read in full**. Every quantitative claim below about HydroGym comes from the preprint [arXiv:2512.17534](https://arxiv.org/abs/2512.17534), the [dynamicslab/hydrogym](https://github.com/dynamicslab/hydrogym) repository, and the PMLR version — not from the *Nature* text. Treat the numbers as reported secondhand until the paper is ingested properly into `summaries/`.
**Related Concepts:** [[edge-generation-atlas-0.1]], [[unet-hierarchy-atlas-0.1]], [[port-algebra-atlas-0.1]], [[f1-pathmap-and-end-goal]], [[agent-definition-atlas-0.1]], [[case-study-wind-farm-wake-2d-atlas-0.1]], [[results-w0-w3-wind-farm]], [[expert-library-atlas-0.1]], [[mixture-of-experts]], [[transfer-learning-fine-tuning]], [[neural-surrogates]], [[world-models-physics-ai]], [[physics-simulation-datasets]]

---

# 0. The three-part answer, up front

**Does HydroGym help Atlas as an agent?** **No** — and the reason is not a near miss, it is a category error that the shared vocabulary invites. An Atlas *agent* is a chart: a declared geometric domain with a physical state and a governing family, computed by a frozen expert ([[agent-definition-atlas-0.1]]). An RL *agent* is a policy $\pi(a\mid s)$ that emits actions. Nothing in HydroGym is a chart, and no Atlas agent is a policy. §2.

**Does it help as infrastructure?** **Yes, in two specific and currently scheduled places** — as a standardized out-of-distribution battery for **rung 4** of the ladder, the expert-reuse test that [[f1-pathmap-and-end-goal]] §3.1 flags as "most likely to be skipped by accident"; and later as a differentiable gradient-check reference at **rung 10**. It is worth nothing to W5/W6. §3.

**Does it clarify how RL and Atlas interact later?** **Yes, and this is the largest return.** HydroGym is the current best statement of the interface a flow-control RL problem expects, and mapping that interface onto [[port-algebra-atlas-0.1]] gives a clean formalization of the rung-11/12 agent layer that the wiki did not previously have: **an RL action is the assignment of one variable of an effort–flow pair at an unconnected port.** §5.

---

# 1. What HydroGym is

A solver-independent, Gymnasium-compatible reinforcement-learning platform for **active flow control**. Reported contents:

| | |
|---|---|
| **Backends** | six — Firedrake (FEM), NEK5000 (spectral element), MAIA LBM, MAIA structured FV, JAX (fully differentiable), JAX-Fluids (compressible) |
| **Environments** | 61+ validated (88 pre-configured): cylinder, rotating cylinder, pinball, cavity, backward-facing step, square cylinder, sphere, cube, NACA0012 steady/gust, DRA2303 airfoil, turbulent boundary layer, turbulent channel, Kolmogorov flow, shock-vector control |
| **Regimes** | $\mathrm{Re}\approx 30$ to $4\times10^{5}$; 2D and 3D; incompressible through compressible |
| **API** | `reset`/`step`; action = actuator setting (jet blowing/suction, cylinder rotation, surface waves), observation = sensor readings, reward = a drag / lift / skin-friction functional |
| **Headline result** | a policy trained **only on a simplified channel flow** transfers zero-shot to an unseen 3D wing at $\mathrm{Re}=2\times10^{5}$, cutting local skin friction by $38\%$, at roughly $10^{-4}$ the exploration cost of optimizing on the wing directly |

The platform is not new — it has been presented since APS DFD 2023 and the preprint dates to December 2025. **The practical consequence of the *Nature* publication is not new capability but maturity:** it is installable today, and its environment set is now a credible community benchmark rather than one group's harness.

---

# 2. The word "agent" is overloaded, and this will cause real confusion

| | Atlas "agent" | RL "agent" |
|---|---|---|
| **What it is** | a declared geometric subdomain + its field state + its conditioning | a policy $\pi_\theta(a\mid s)$ |
| **What it produces** | the next state of *its own* fields | an action applied to *someone else's* system |
| **Where it lives** | inside the model | outside the model, acting on it |
| **How many** | 7 in the wind farm; ~15–20 at rung 9 | 1, or one per turbine in a multi-agent formulation |
| **Trained?** | frozen — nothing is trained in the current build ([[results-w0-w3-wind-farm]]) | necessarily trained, by reward |

The collision is worse than cosmetic, because **multi-agent RL is a real field with a real meaning for "agent,"** and the wind-farm scenario — $N$ turbines, each with a local controller, coupled through wakes — is a textbook multi-agent RL problem *and* a textbook Atlas agent graph, simultaneously, over the same physical objects, meaning something different each time.

**Recommendation, carried to [[agent-definition-atlas-0.1]]:** keep **agent** for the chart, and use **policy** or **controller** — never "agent" — for anything reward-trained. Where a page must discuss both, say *chart-agent* and *control policy* explicitly.

---

# 3. What Atlas can actually use today

## 3.1 Not as an expert, and not as an agent

An Atlas expert replaces a solver; HydroGym's environments **are** solvers. Handing Atlas a Firedrake cylinder environment gives it the thing it exists to make unnecessary.

There is also no actuator-disk, rotor, or wind-farm environment anywhere in the set — the nearest objects are bluff-body wakes, which share the deficit-and-recovery structure of a turbine wake but not the momentum-sink mechanism that makes [[case-study-wind-farm-wake-2d-atlas-0.1]] a *field ↔ lumped* coupling test. The one physical feature the active case study is built around is the one feature HydroGym does not have.

## 3.2 As a reference solver: real, but probably not worth the switch

[[results-w0-w3-wind-farm]] records that a ~100-line pseudo-spectral reference solver "was not in the plan and should have been," and W6 still needs a monolithic baseline. HydroGym's JAX backend (turbulent channel, Kolmogorov flow) is a validated, citable, *differentiable* version of roughly that object.

But the adaptation cost is not zero and the incumbent is already built and already trusted: the existing `reference.py` closes the momentum balance to $10^{-16}$ unforced, drives the *identical* probe code, and has already caught three distinct bugs. Swapping it for a general-purpose environment means re-establishing all of that, plus adding the body-force disk term HydroGym does not ship. **Verdict: do not switch for W5/W6.** Revisit at rung 10, where differentiability stops being a convenience and becomes the deliverable (§7).

## 3.3 As the rung-4 battery — the real value

Rung 4 is *"does an expert trained for one scenario work in another?"* — the foundation-model claim itself. The ladder specifies the rung but **not a test set**, and a reuse test whose test set is chosen after the fact is not evidence. HydroGym supplies one off the shelf: 61 environments, systematic $\mathrm{Re}$ progressions, a fixed API, and a community that will run the same battery.

The protocol writes itself, and needs no RL at all:

1. Take the frozen `Poseidon-T` checkpoint adopted at W0.
2. Roll it out on the **uncontrolled** baseline trajectory of $k$ HydroGym environments (cylinder at several $\mathrm{Re}$, cavity, step, Kolmogorov).
3. Grade against each environment's own solver, using the metrics already built at W4.
4. Report error as a curve against $\mathrm{Re}$ and against geometry class, not as a single number.

Two constraints inherited from W0/W3 that the protocol must respect: the checkpoint **takes no forcing input** and **destroys a uniform mean flow**, so environments must either be zero-mean or be run through the Galilean change of frame already implemented. Environments whose defining feature is wall-bounded shear sit outside the plan's own stated comfort zone and should be reported separately rather than quietly dropped.

## 3.4 What not to do

**Do not train on it.** Fine-tuning the fluid expert on HydroGym trajectories forfeits the zero-training, zero-data constraint that is the entire reason [[case-study-wind-farm-wake-2d-atlas-0.1]] is buildable, and converts a clean composition test into a confounded one. The W0 findings also suggest data would not be the fix: the uniform-flow failure is a **frame** problem, not a coverage problem, and was solved exactly by a change of variables rather than approximately by more data.

---

# 4. What the zero-shot result does and does not support

The channel $\to$ 3D wing transfer is a striking result, and it is **not** evidence for Atlas's Claim A.

$$\underbrace{\pi:\ s\mapsto a}_{\text{what transferred there}}\qquad\text{vs.}\qquad\underbrace{\mathcal G_{\Delta t}:\ u(t)\mapsto u(t+\Delta t)}_{\text{what an Atlas expert must transfer}}$$

A control policy needs only the **local sensor-to-actuator map** to generalize — near-wall streak dynamics look similar in a channel and on a wing, and a policy that damps them needs no global model of either flow. A dynamics operator must reproduce the **full evolution**, including everything the policy is free to ignore. Policy transfer is therefore a strictly weaker claim than operator transfer, and the $10^{-4}$ cost reduction is a claim about *search*, not about *fidelity*.

What it does support, stated honestly: the hypothesis underneath both — **that the physics governing a local region is substantially configuration-independent, and that a model of it can be reused across geometries.** That is the same bet [[expert-library-atlas-0.1]] makes when it organizes experts by governing-equation family rather than by scenario, and the same bet [[transfer-learning-fine-tuning]]'s diversity principle qualifies. A second, independent field finding it holds for policies raises the prior that it holds for operators. It does not measure it.

It is also a direct corroboration of the **deployment story** in [[f1-pathmap-and-end-goal]] §1.2 — search wide and cheap, verify the shortlist classically — arriving from a group with no stake in Atlas's architecture.

---

# 5. RL's three roles in Atlas — two already deferred, one newly formalized

The wiki already contains two RL proposals. The third is what HydroGym prompts.

| Role | What the policy decides | Action space | Status |
|---|---|---|---|
| **(i) Topology** — [[edge-generation-atlas-0.1]] Mechanism B | which agent pairs are connected, by which ports | discrete, combinatorial | **deferred** — trigger: not before scenario 2 |
| **(ii) Routing** — [[unet-hierarchy-atlas-0.1]] coarse-level gating | which expert handles an agent at a coarse level | discrete regime switch | **deferred** — same trigger |
| **(iii) Coordination** — *this page* | what the physical system is commanded to do | continuous actuator settings at **unconnected ports** | rungs 11–12; not previously formalized |

## 5.1 HydroGym does not move the trigger on (i) and (ii)

Both were deferred for a specific, non-arbitrary reason: their justification is *learning from experience across regime pairings*, and one scenario provides none. HydroGym does not supply that experience either — its 61 environments are all **single-domain flow control with no graph structure at all.** There is no interface topology to learn from, because none of its environments is decomposed. The trigger stands unchanged: **not before scenario 2** ([[impl-atlas-0.1-phase5-expansion]] §3.3).

## 5.2 Role (iii): the coordinator, stated in port algebra

This is the useful synthesis, and it falls out of [[port-algebra-atlas-0.1]] almost mechanically.

Every port is an **effort–flow pair** $(e_p, f_p)$ whose product is power, $P_p = e_p f_p$. An **unconnected port** is one whose conjugate side has no agent attached — the wind-farm spec already declares two, the `ROT` shafts of $R_1$ and $R_2$, and [[case-study-wind-farm-wake-2d-atlas-0.1]] §3.2 already calls out that an unconnected port is *a measurable power flow* rather than an absence.

**The formalization:** a control action is the assignment of **exactly one** variable of an effort–flow pair at an unconnected port; the model then determines the conjugate variable.

$$
a_t \;\in\; \prod_{p\,\in\,\mathcal U}\mathcal A_p,
\qquad
\mathcal A_p \subseteq \{e_p\}\ \veebar\ \{f_p\},
\qquad
\mathcal U \;=\; \{\text{unconnected ports}\}\,\cup\,\{\text{declared design parameters}\}
$$

Setting one and letting the model supply the other is exactly the **causality assignment** convention of bond-graph modeling, which is where the port formalism comes from in the first place — so this is not a new mechanism bolted on, it is the existing algebra used in the direction it was already built to be used. Setting *both* over-determines the port, and is the formal statement of "commanding a physically impossible actuator."

For the wind farm concretely: the induction factor $a_i$ and the yaw angle $\gamma_i$ of each rotor, with extracted shaft power appearing at the `ROT` port as the conjugate response.

**The reward, and why it is better posed here than on a black-box surrogate.** The obvious objective is delivered power,

$$
r_t \;=\; \sum_{p\,\in\,\mathcal P_{\text{out}}} e_p f_p\,\Delta t \;-\; \lambda\,\lvert\mathcal R(t)\rvert ,
$$

where $\mathcal R(t)$ is the **global power residual** of [[port-algebra-atlas-0.1]] §6 — already the single cross-case-study diagnostic, and already required to satisfy $\lvert\mathcal R(t)\rvert < 1\%$ of extracted power at gate W11.

**[AI Inference]:** the second term is the strongest architectural argument for Atlas-as-RL-environment, and it is testable rather than rhetorical. The standard, well-documented failure of RL trained against a learned surrogate ([[neural-surrogates]], [[world-models-physics-ai]]) is that the policy **discovers the surrogate's error and exploits it**, returning a control law that is excellent against the model and worthless against reality. Detecting this normally requires an external check the surrogate cannot provide. Atlas measures a conservation residual at every port *by construction*, so surrogate exploitation has a **visible signature inside the environment itself**: a policy driving $\mathcal R(t)$ up is, definitionally, extracting power the composed model is not accounting for. A black-box neural surrogate has no comparable observable. This is speculative in the sense that it has never been run; it is not speculative about the mechanism, which exists and is already gated.

**The falsification is cheap and should be scheduled as such:** train a deliberately weak policy against the composed wind-farm model with $\lambda = 0$, and test whether the reward it achieves correlates with $\mathcal R(t)$. If a reward-maximizing policy leaves the residual flat, the concern was overstated and $\lambda$ can be dropped. If reward and residual rise together, the diagnostic works and the claim is earned.

## 5.3 The three roles must never share a policy

This is the sharpest thing on the page, and it is a design constraint rather than an observation.

Roles (i) and (ii) are policies **about the model** — their reward includes prediction accuracy against ground truth. Role (iii) is a policy **about the physical system** — its reward is a physical objective. Merge them, or let one policy carry both terms, and you create an agent that can **increase its reward by steering the physical system into a regime its own model predicts well.**

The wind farm makes the pathology concrete: a merged policy could learn to yaw the turbines so that no wake ever strikes a downstream rotor — thereby weakening the $F\!-\!R_2$ edge to nothing, removing the hardest coupling from the graph, lowering prediction error, and destroying array power. Both halves of the reward would be satisfied. The result is a controller that has optimized away the problem instead of solving it.

**The rule:** model-side RL (topology, routing) is trained offline, frozen, and only then exposed to a control policy. The control policy may never receive gradient or reward signal that depends on the model's own accuracy.

---

# 6. Where a coordinator sits on the ladder

[[f1-pathmap-and-end-goal]]'s rung 12 is *"agent layer — autonomous proposal and evaluation of design changes,"* annotated **"none — different research program."** That annotation stands, and this page does not soften it. What HydroGym adds is evidence about **which half of that program is expensive**: its own headline is a four-orders-of-magnitude reduction in *exploration* cost, obtained by making evaluation cheap. The controller was never the bottleneck; the evaluator was. **Atlas is a bet on exactly that bottleneck**, so the right reading of HydroGym is not "here is a component to adopt" but "here is independent confirmation that the thing being built is the thing that is scarce."

Note also that rungs 10–11 (design gradients, optimizer in the loop) sit *below* the agent layer deliberately. A differentiable composed model admits **analytic policy gradients** — differentiating the objective through the environment — rather than the score-function estimators model-free RL is restricted to on a non-differentiable CFD backend. That is a different and much better-conditioned optimization problem than the one HydroGym's environments pose, and it is reachable at rung 10 with no RL machinery at all.

---

# 7. What to actually do, in priority order

1. **Nothing changes for W5–W11.** The current gates are unaffected, the reference solver is not replaced, no dependency is added. This is the correct default and it is stated explicitly so it is not quietly eroded.
2. **Record HydroGym against rung 4 as the candidate reuse battery**, with the protocol of §3.3 and its two inherited constraints. This is the only change with a scheduled home.
3. **At Phase F (yaw / wake steering), write the Gymnasium adapter — not the policy.** Roughly a day: wrap the composed graph in `reset`/`step`, expose $(a_i,\gamma_i)$ as actions and shaft `ROT` power as reward. The deliverable is the *interface* — a demonstration that a composed Atlas rollout is drop-in where a CFD backend goes. Whether the policy learns anything is secondary, and any quantitative wake-steering claim from a 2D wake is barred by the G5 discount regardless.
4. **Run the §5.2 falsification** ($\lambda=0$, reward-vs-residual correlation) at the same time. It is nearly free once (3) exists, and it decides whether the residual-penalty argument is real.
5. **Fix the naming** in [[agent-definition-atlas-0.1]] before role (iii) gets its own page.
6. **Ingest the paper properly** into `summaries/solvers-and-simulation/` once the full text is available. This page is an assessment, not a summary, and must not be cited as one.

---

# 8. Open questions

- **Is a frozen-expert composed model stable under adversarial control input?** Every rollout measured so far is passive. A policy actively searching the action space is a far harsher stability test than a gust ramp, and W3's finding that the checkpoint has a *minimum viable coupling step* suggests the composed system's stable region may be smaller than the action space a policy would explore.
- **Does the residual signature of §5.2 survive at rung 9 scale**, where $\mathcal R(t)$ aggregates over ~20 agents and a single exploited interface may be buried in the sum? The per-port residual breakdown already required at W11 is the right instrument; its discriminating power at scale is unknown.
- **Multi-agent RL over an agent graph** — one policy per turbine, communicating along the same typed edges the physics uses — is the obvious next formulation, and the one where the two senses of "agent" finally coincide instead of colliding. Whether that coincidence is useful or merely tidy is untested.

---

## See Also

- [[edge-generation-atlas-0.1]] — Mechanism B, role (i); the deferral and its trigger
- [[unet-hierarchy-atlas-0.1]] — coarse-level gating, role (ii); MLP where smooth, RL where discrete
- [[port-algebra-atlas-0.1]] — the effort–flow algebra §5.2 builds the action space from; the residual $\mathcal R(t)$
- [[f1-pathmap-and-end-goal]] — rungs 4, 10, 11, 12; the deployment story §4 corroborates
- [[case-study-wind-farm-wake-2d-atlas-0.1]] — Phase F's yaw axis, natural home for the adapter; the two unconnected `ROT` ports
- [[results-w0-w3-wind-farm]] — the reference solver, and the no-forcing-input and uniform-flow findings that constrain §3.3
- [[agent-definition-atlas-0.1]] — the page that needs the naming fix
- [[neural-surrogates]], [[world-models-physics-ai]] — the surrogate-exploitation failure mode §5.2 is about
- [[transfer-learning-fine-tuning]] — the diversity principle qualifying §4's transfer reading
- [[expert-library-atlas-0.1]] — the by-governing-family bet §4 connects to
- [[physics-simulation-datasets]] — where a HydroGym-derived evaluation set would be catalogued
