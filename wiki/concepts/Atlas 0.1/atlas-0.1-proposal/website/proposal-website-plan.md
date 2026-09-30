# The proposal website — a product launch that a sceptic can audit

**Type:** Concept page — **plan: story, structure, design language, build, and honesty rules** (folder: `Atlas 0.1/atlas-0.1-proposal/website/`)
**Status:** written 2026-09-30. **The three reference sites could not be opened from the planning session** (its network blocked `simscale.com`, `antigravity.google` and `googlebook.google`). §2's design notes come from search results describing them and from general knowledge of the genre. **The website chat should open all three on the owner's machine before designing.**
**Revised 2026-09-30 with the owner's decisions** ([[00-proposal-workstreams]] §5):
- **O2:** the demo is a local install, so there are no replays;
- **O4:** GitHub Pages;
- **O7:** the headline, *"A full world. Decomposed and simulated by experts."*;
- **O8:** the architecture document is an arXiv-style HTML page;
- **O9:** everything is public at launch.

**The section-by-section outline, with the statistics, the style and the image prompts, is [[website-outline]].**
**Hub:** [[00-proposal-workstreams]] · **Content sources:** [[website-evidence-and-citations]] · [[vision-scenarios-and-image-prompts]] · [[chart-operator-architecture]] · [[formal-proofs-plan]] · [[demo-finish-plan]]

---

## 0. What it is for

The owner, 2026-09-30:

> *"Functions as an advertisement, a mini-'pitch'. It's not just scientific — it's flashy, unique, cool, and convincing to the non-technical person."*

**Two readers, one page.** The non-technical reader should leave believing that simulation can be assembled from cooperating pieces, faster, and that this project has shown it. The technical reader, a reviewer, a funder's scientific adviser, someone from the field, should find that **every number clicks through to a record, and every claim is sized to its evidence**. These do not conflict. **[AI Inference]:** in this genre, the pages that convince specialists are the ones whose big numbers survive a click, and a single inflated number costs the whole page.

**The story, in five beats:**
1. **The promise**: simulate the whole by solving the parts.
2. **It works today**: the demo's measured results.
3. **Science agrees**: published results, classical and neural.
4. **The missing piece**: learned experts built for this, the chart operator, with proofs.
5. **What it unlocks**: the vision's three scenarios.

---

## 1. The sections

| # | section | what it shows | source |
|---|---|---|---|
| 1 | **Hero** | **"A full world. Decomposed and simulated by experts."** (O7), one line under it, two pill buttons (*Try the demo*, *Read the architecture*), and a full-bleed image of **a whole world cut into charts**, captioned **Illustration** | [[website-outline]] section 1 |
| 2 | **How it works, in three moves**: *Cut. Solve. Stitch.* | a scroll-driven animation built from a **real workbench geometry**: the S-channel, cut into its four generated windows, then the seams exchanging, then the assembled field. No stock diagram | the workbench's `s-channel` example and its record |
| 3 | **Working today** | a band of four or five big numbers, each with a one-line caption and a link to its record; the passage O3 asks for (today is classical; the gain grows with complexity, three dimensions and learned experts); the **install card** (O2) | [[website-evidence-and-citations]] §1 · [[website-outline]] section 3 |
| 4 | **The science agrees** | cards of published results: classical decomposition at scale, and neural operators and neural DD, each clearly marked **published by others** | [[website-evidence-and-citations]] §2–§3 |
| 5 | **The missing piece** | plainly: learned experts today are not built for this; the measured reasons (S1–S12, drawn as a checklist); and the chart operator that answers them, in three pictures, *charts for geometry, waves for agreement, operators for physics* | [[outcome-c5-requirements-for-dd-native-experts]] · [[chart-operator-architecture]] |
| 6 | **Mathematically secure** | the headline theorem in one sentence, the blueprint's dependency graph with its nodes coloured by status, and a link to the Lean project | [[formal-proofs-plan]] §3–§4 |
| 7 | **The vision** | three full-bleed scenarios: a whole race car, a wildfire forecast and its control, a Moon base run by cooperating experts. Each has its image, an agent-graph overlay drawn in code over it, and one paragraph | [[vision-scenarios-and-image-prompts]] |
| 8 | **The path** | stages 0 to 3 from [[chart-operator-training-and-cost]], as a horizontal timeline | that page §2 |
| 9 | **Footer** | the demo, the architecture document, the proofs, the code, a contact | — |

**The owner's required links:** the demo (§1, §3, §9), the learned-expert architecture proposal (§1, §5, §9) and the Lean proofs (§6, §9).

---

## 2. The design language

**What the three references share**, from their descriptions ([[website-evidence-and-citations]] §5 has the sources):
- **antigravity.google**: a spare Google product register. A white surface, a thin top navigation line, **one large typographic anchor per screen**, a compact set of **pill actions** with a black pill as the primary, decorative **particle fields** that add energy without taking over, 4 px spacing steps. Google Sans Flex; primary colour $\#121317$.
- **googlebook.google**: a hardware product launch (Google's laptop category, announced 2026-09-21). Full-bleed product photography, feature sections that each make one point, and a specification band.
- **simscale.com**: *"AI-Native Engineering Simulation in the Cloud"*. Engineering simulation sold to engineers, with agentic AI as the message, and proof by product imagery and results.

**The recommendation for Atlas:**
- **Surface:** white, with **one dark chapter** (the vision) so that the scenario images carry their light.
- **Type:** one large sans-serif anchor per screen. **Use an openly licensed family and host it with the site.** Candidates are Inter or Geist; Google Sans Flex only if its licence permits, to verify.
- **Colour:** near-black text ($\#121317$, as Antigravity's) and **one accent for seams and waves**, a cyan. Fields use a perceptual map. Every chart follows one palette ([[website-evidence-and-citations]] §4).
- **Motion:** scroll-driven reveals and **one signature motion, the seam**: a thin line of light passing between two pieces. It appears in the hero, the explainer and the vision overlays. Every animation honours `prefers-reduced-motion`.
- **Particles, with a meaning:** where Antigravity's particle field is decoration, Atlas's can be **tracers carried by a real recorded flow field** (the wind farm's wakes). The decoration then *is* the evidence.
- **Imagery:** AI-generated for the vision only, each **labelled as an illustration**. Every image of today's results is a real record, rendered.

---

## 3. How it is built

| part | recommendation | why |
|---|---|---|
| framework | a static site: **Astro**, with plain components and no client framework except where a section needs one | fast, no server, easy to host; the numbers are data at build time |
| animation | CSS scroll-driven animations where they suffice; **GSAP ScrollTrigger** for the explainer (its licence terms to be checked) | the genre's scroll storytelling |
| charts | Observable Plot or D3, from JSON built out of the records | one palette; every mark from data |
| real screenshots and renders | captured from the served workbench and from records | O2 made the local install the demo's way in, so there are no replays |
| hosting | **GitHub Pages (O4)**. On the free plan Pages publishes from a **public** repository: the site lives in its own repository, private while built and previewed locally, made public at launch (O9) | free |

### 3.1 The claims pipeline: no number typed by hand

- `scripts/site_claims.py` reads the run records and writes `site/data/claims.json`. Each claim carries its value, units, the sentence it appears in, the record's path, the date, the machine's state, its caveat and its vault page.
- `site/data/literature.json` holds each published number with its exact quotation, page or figure, DOI and the date someone read it.
- **A test fails the build if a number shown on the site is not in one of the two files**, or if a claim's value differs from its record.

**[AI Inference]:** this is the vault's own discipline, *never report an unmeasured number*, made mechanical, and it is the site's best defence against its own enthusiasm.

---

## 4. The honesty rules

1. **Every number links to its record or its paper.**
2. **Vision is labelled as vision**, today as today, others' results as others'. Three visual treatments, used consistently.
3. **Speeds are ratios, each with its machine.** For example: *"4.8× faster than the undivided solver, 12 rotors on 4 threads of a 22-core laptop"*.
4. **Where the undivided solve wins, the site says so once, plainly**, in the *Working today* section: *"On small problems the undivided solve is faster; decomposition pays with size, or when it lets each piece take its own time step."* **[AI Inference]:** leaving this out is what a sceptic would look for first, and finding it stated is disarming.
5. **The learned case is shown with its competitor** (the coarse classical solver) and its caveat, as [[demo-learned-case-plan]] §7 specifies.
6. **No trademarks in the imagery**: no team liveries, no sponsor marks, no agency logos ([[vision-scenarios-and-image-prompts]] §1).

---

## 5. Headline candidates

| headline | line under it |
|---|---|
| **Physics, assembled.** | Atlas solves a simulation in pieces that talk to each other — so the pieces can be faster, smarter, and anywhere. |
| **Simulate the whole by solving the parts.** | Classical solvers and learned experts, coupled through one standard, with proofs that it converges. |
| **Every piece an expert. Every seam a guarantee.** | A framework for simulations built from cooperating solvers, and the learned experts designed for it. |
| **Solve anything, one chart at a time.** | The atlas of a manifold, applied to physics. |

**Decided (O7, the owner, 2026-09-30): "A full world. Decomposed and simulated by experts."**, none of the four above.

---

## 6. Build order for the website chat

1. Open the three reference sites and two or three others the owner likes. Capture screenshots into the site's design notes.
2. Build from [[website-outline]]. O4 and O7 are settled.
3. Build the claims pipeline and its test first, with today's records ([[website-evidence-and-citations]] §1).
4. Build the skeleton with every section, using placeholder images labelled as placeholders.
5. Write the image prompts' final versions and hand them to the owner ([[vision-scenarios-and-image-prompts]]). The owner generates the images, and the chat places them with their overlays.
6. Add the demo's fast examples and the learned case **only as their records arrive**, never as expectations.
7. Link the architecture document and the proofs' blueprint once each is public.
8. Check performance, accessibility, reduced motion, a phone's width and a laptop's, and every link.

---

## See Also

- [[website-evidence-and-citations]] — every number the site may show, with its source and status
- [[vision-scenarios-and-image-prompts]] — the three scenarios and their prompts
- [[00-proposal-workstreams]] — why the website comes last
- [[outcome-evidence-ledger]] — the vault's ledger of quotable numbers
- [[outcome-demos-and-artifacts]] — the interactive pages that exist already
