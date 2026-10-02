# The website, section by section — topics, the numbers to hit, the written style, and the visuals

**Type:** Concept page — **content outline and style guide for the proposal website** (folder: `Atlas 0.1/atlas-0.1-proposal/website/`)
**Status:** written 2026-09-30, after the owner's decisions O1–O9 ([[00-proposal-workstreams]] §5). **An outline, not copy**: the draft lines are there to fix the voice, and the website chat rewrites them. Every statistic is named by its id in [[website-evidence-and-citations]]; **a number marked S or U there is not shown until it is verified**, and a number from a run that does not exist yet is shown only when its record does.
**Revised 2026-10-02 with the owner's yes to the website chat's proposal (W354 step 1); the site in `site/` follows this outline with these changes:**
- **the learned case opens section 5**, not section 3, shown with all four arms and the sentence that the coarse solver won G5. Section 3 then stays all classical, so O3's passage stays true;
- **section 5 follows the architecture's version 1**: *charts for geometry, superelements for agreement, operators for physics* replaces "waves for agreement" ([[chart-operator-architecture]] §1);
- **section 3's stat band** shows the demo's Fast examples (A11) in place of A1, and drops A7 and A10, which have no run record; A2 stays;
- **section 6** shows the three headline theorems (tier 0 first), the counts read from the Lean records, and the T3 finding; "a first" is reworded ([[website-evidence-and-citations]] §3.2);
- **the install page has three tabs**: Windows and Linux verified, macOS marked not verified;
- **an evidence page** lists every number with its record or quotation, and **cleaned copies of the cited records** are published with the site; the literature cards carry a claim, a number and a citation, with the quotation on the evidence page;
- a top bar with section links and a black *Try the demo* pill; the typeface is Inter ([[website-design-notes]] §6).
**Hub:** [[00-proposal-workstreams]] · **Plan:** [[proposal-website-plan]] · **Evidence:** [[website-evidence-and-citations]] · **Scenario images:** [[vision-scenarios-and-image-prompts]] · **Design notes:** [[website-design-notes]]

---

## 0. The decisions this outline is built on

| # | the owner's decision (2026-09-30) | what it means for the site |
|---|---|---|
| O2 | the demo reaches people as **a one-command local install, on Windows and on a Mac** | every *Try the demo* button goes to an install page: one command per system, what it installs, how long it takes, and what the visitor will see. There is no live demo on the site |
| O3 | a fast example that cannot reach the bar loads its fastest honest setup and names its limit, **and the site says that today's results are classical, and that the gains grow with complexity, with three dimensions and with learned experts** | a short, explicit passage in section 3, and the bridge into section 5 |
| O4 | **GitHub Pages**, free | a static site. On the free plan, Pages publishes from a **public** repository, which fits O9: the site's repository becomes public when the site launches |
| O7 | the headline: **"A full world. Decomposed and simulated by experts."** | the hero, and the idea every section returns to: a world cut into pieces, each piece an expert |
| O8 | the architecture document is **an HTML page in arXiv's HTML style** | the *Read the architecture* button opens it; section 5 quotes it |
| O9 | the proofs and the site stay **private while built, public once the site links to them** | the Lean blueprint and the architecture page go public on launch day, with the site |

---

## 1. The style, described

**The voice.**
- **Plain words and short sentences.** A number comes with its unit and its conditions: *"4.8× faster on 12 rotors, on four cores of a laptop"*.
- **Confident about what is measured, exact about what is not.** No superlative without a number behind it. Every vision sentence is in the future or conditional tense.
- **One idea per screen**, written first as a headline of three to seven words, then one or two sentences under it.
- **Words to use:** *pieces*, *experts*, *seams*, *checked*, *measured*, *proved*. **Words to avoid:** *revolutionary*, *magic*, *AGI*, *infinite*, and any claim that a learned model is "as good as physics" without the number that says how close.

**The look.**
- **Surface:** white for today's results and the science; **one dark chapter**, deep charcoal $\#121317$, for the vision.
- **Type:** one openly licensed sans-serif family hosted with the site, such as Inter or Geist. Display sizes large enough that a headline fills most of a laptop's width. Body text 18–20 px, lines of 60–75 characters.
- **Colour:** near-black text, warm grey secondary text, and **one accent, electric cyan**, used only for the physics made visible: seams, waves between pieces, streamlines. Fields use one perceptual colour map site-wide.
- **Space:** generous. Sections are at least a screen tall on a laptop, and nothing competes with the headline.
- **Buttons:** pills. A black pill for the primary action; secondary actions are quiet text links with an arrow.

**The motion.**
- **The seam** is the site's one signature motion: a thin line of cyan light running along the boundary between two pieces, then a pulse crossing it. It appears in the hero, in *How it works*, and on every vision image's overlay.
- **Scroll-driven, never autoplaying sound.** Reveals fade and rise by about 24 px. Numbers count up once, the first time they enter the view.
- **Reduced motion honoured:** with `prefers-reduced-motion`, everything is static and every animation's final frame is shown.

**Mobile.** Every section collapses to one column. The hero uses its 4:5 crop. Stat bands become a two-by-two grid, and tables become stacked cards.

---

## 2. The sections

### Section 1 — Hero

- **Topics:** the headline; one sentence on what Atlas is; the two actions.
- **Copy:**
  - headline: **A full world. Decomposed and simulated by experts.**
  - under it, a draft: *Atlas splits a simulation into pieces, gives each piece an expert solver, and lets the experts agree at their seams: checked, conserved and fast.*
  - buttons: **Try the demo** (black pill, to the install page) and **Read the architecture** (text link).
- **Statistics:** none. The hero sells the idea, and the numbers start one scroll down.
- **Visual and motion:** a full-bleed image of a whole planet whose surface is divided into curved glowing charts, the atlas of the name. On scroll, the camera sinks toward the surface and the seams brighten, handing off to section 2. The image is labelled **Illustration** in small type in its corner.
- **AI visual prompts:**

> **H1, the world as an atlas (21:9, with a 4:5 crop).** A photoreal planet seen from low orbit at the day–night terminator, oceans, cloud bands, deserts and ice caps, lit by a low sun, with a thin blue atmosphere glowing along the limb. Its whole surface is subtly divided into large curved tiles that follow the terrain like the charts of an atlas: each tile edge a thin line of soft electric-cyan light, brightest on the night side where city lights also glow. The tiles are not a latitude-longitude grid; they curve with coastlines and mountain ranges. Cinematic, calm, awe-inspiring, ultra-detailed, deep black space with a few stars, generous empty space in the upper third for a headline. No text, no logos, no labels, no UI.

> **H2, the descent (16:9), for the scroll hand-off.** The same planet much closer: a curved horizon, a coastline with a river delta, mountains and a forest, seen from high altitude at golden hour. Thin electric-cyan seams trace curved tile boundaries across the landscape, following ridgelines and the coast, and one seam in the foreground has a small bright pulse travelling along it. Photoreal, aerial photography, haze and depth, no text, no logos.

> **H3, alternative hero: a world in miniature (21:9).** A tilt-shift photograph of a meticulously detailed miniature world on a dark table: a race track with a single-seater race car, a forested hillside with a small controlled wildfire and a firebreak, and a lunar landing pad with a lander, all joined on one sculpted landscape. The landscape is cut into precise curved pieces like a jigsaw, each piece's edge lit from within by a thin cyan line. Soft studio light, shallow depth of field, photoreal, whimsical but premium, no text, no logos, no people.

### Section 2 — How it works: *Cut. Solve. Stitch.*

- **Topics:**
  1. **Cut**: the domain is cut into pieces that follow its shape, not boxes.
  2. **Solve**: each piece is solved by an expert, a classical solver today and a learned one next.
  3. **Stitch**: the pieces exchange what crosses their seams (force, heat, current, flow) until they agree, and nothing is lost at a seam.
  4. **Checked before it runs**: a compiler reads every connection and admits it, admits it without a certificate, or refuses it with the reason.
- **Statistics to hit:** A5 (8 kinds of physics, one framework); A6 (zero hand-written coupling code per case).
- **Copy, drafts:** *"Cut it where it bends."* · *"Every piece gets its own expert."* · *"Seams carry power, and power adds up."*
- **Visual and motion:** **drawn in code from a real workbench case, not generated.** The S-channel: its outline draws itself; the four generated windows appear, each tinted; seams light up; the temperature field fills in window by window; the windows fade and the assembled field remains, identical to the undivided solve. Three sticky steps on the left, the animation on the right.
- **AI visual prompts:** none. Here the real geometry *is* the point.

### Section 3 — Working today

- **Topics:**
  - it runs today, on a laptop, on eight kinds of physics;
  - on large problems, cut into pieces, it is faster;
  - the pieces agree with the undivided answer to parts per billion;
  - nothing is lost at a seam;
  - **the honest part (O3)**: on small problems the undivided solve is often faster. Today's demo is classical, and it wins where parallel pieces or local time steps pay. **The gains grow as problems become larger, more complex and three-dimensional, and with learned experts built for the pieces**, which is the next section;
  - *Try the demo*: the one-command install;
  - the fast example per type and the learned case, **once their records exist**.
- **Statistics to hit, now:**
  - **A1**: $4.83\times$ (12 rotors, 4 threads) and $4.55\times$ (21 rotors, 8 threads), with the machine;
  - **A2**: $3.5$–$5.7\times$ from 5 to 21 rotors, farm power within $2.3$–$8.3\%$;
  - **A3**: agreement within $3.6\times10^{-9}$ of the scale in every iterated example;
  - **A4**: every balance check passes;
  - **A7**: the test count;
  - **A10**: 89 tiers of measurement.
- **Statistics to hit, when recorded:** each type's fast example (ratio, agreement, the named mechanism or limit); the learned case (ratio against the classical decomposition and the full domain, error against the truth run, and **the coarse classical competitor beside it**).
- **Copy, drafts:** *"4.8× faster. Same answer."* · *"Faster where it should be. Honest where it isn't."* · for O3: *"Everything on this page is classical physics, cut into pieces. That already pays on big problems. It pays far more in three dimensions, on complicated shapes, and with experts that learn, which is what we build next."*
- **Visual and motion:**
  - a stat band of four big numbers that count up, each with a one-line caption and a small *source* link;
  - under it, a speed chart: one bar per example against a line at $1\times$;
  - the install card: two tabs (Windows, macOS), one command each, and a short *what you will see* line with a real screenshot of the workbench;
  - **Optional, not required by O2:** a short muted video or GIF of the workbench running `farm-12`, screen-recorded from the real page.
- **AI visual prompts:** none. Every image here is a real screenshot, render or chart.

### Section 4 — The science agrees

- **Topics:**
  - **decomposition is how the largest simulations already run**: classical, proven, at supercomputer scale;
  - **learned solvers are already much faster on their own problems**;
  - **learned solvers inside decomposition are an active research front**, with Atlas's place in it stated in one line.
- **Statistics to hit, each verified before it is shown:**
  - classical: **L1** (nonlinear FETI-DP/BDDC at about 786,000 cores); **L3** (iteration counts that do not grow with the number of pieces); **L4** (convergence $1-O(h^{1/2})$ against $1-O(h)$);
  - neural: **N1** ($40$–$80\times$ at equal accuracy in 2-D turbulence); **N2** ($26{,}000\times$ on a car's drag at about 3% error); **N3** (DIMON: geometry-dependent operators); **N5** and **N6** (neural Schwarz on new geometries); **N7** (HINTS: uniform convergence).
- **Copy:** each card is one claim, one number, one citation. A line above the cards: *"Results by other researchers, on their own problems."*
- **Visual and motion:** a grid of cards, each with a small monochrome line icon (a grid, a car, a heart, a mesh). The cards rise in sequence.
- **AI visual prompts:** none, to keep others' results visually separate from Atlas's.

### Section 5 — The missing piece: experts built for the pieces

- **Topics:**
  - **the gap, plainly**: today's learned models were built for whole, square, periodic problems. Inside a decomposition they fail in measurable ways (A9: S1–S12, drawn as a checklist);
  - **the chart operator**, in three pictures:
    - **charts for geometry**: every piece is mapped onto a square by a map that never folds;
    - **waves for agreement**: pieces exchange impedance-weighted waves through typed ports;
    - **operators for physics**: a learned operator works on the square, given the chart's metric;
  - **the guarantee**: if each expert passes a checkable bound, the pieces converge from any start, and the answer's distance from the classical one is bounded;
  - *Read the architecture*: the arXiv-style HTML page (O8).
- **Statistics to hit:** A9's headline failures (a uniform $128^2$ grid only; $61\%$ of a car's unknowns unreachable; $0.067\times$ classical speed at the coupling's cadence); A8 only beside that caveat, or not at all.
- **Copy, drafts:** *"Today's AI solvers weren't built to be pieces."* · *"We measured exactly why, in 12 ways."* · *"Charts. Waves. Operators."*
- **Visual and motion:**
  - the S1–S12 checklist with each item's number, the failing ones in grey;
  - three panels, each drawn in code from real geometry: a curved piece mapped onto a square, with its grid lines bending; two pieces trading wave pulses across a seam; the operator's square filling with a field;
  - an optional AI background under the panels (below).
- **AI visual prompts:**

> **M1, a piece that fits (16:9).** A single sculpted piece of frosted translucent glass shaped like a curved section of a river channel, resting on a dark studio surface. Inside the glass, a fine glowing grid bends to follow the piece's curves, and on its flat top face the same grid becomes a perfect square lattice. Thin electric-cyan light runs along its four edges. Minimal, elegant, premium product photography, soft rim light, no text, no logos.

> **M2, pieces in conversation (16:9).** Four frosted-glass pieces of different curved shapes fitted together like a puzzle on a dark studio surface, with narrow gaps between them. Across each gap, tiny pulses of cyan light travel in both directions like messages. Soft top light, shallow depth of field, photoreal, minimal, no text.

### Section 6 — Mathematically secure

- **Topics:**
  - **the headline theorem in one sentence**: *if every learned expert passes a checkable bound, the coupled simulation converges from any start, and its answer is within a stated distance of the classical one*;
  - **the second**: *inside defect correction, a learned expert can change how fast the answer comes, never what it is*;
  - **machine-checked**: proved in Lean, a proof assistant whose small kernel checks every step, so the proof is trusted however it was written;
  - **a first**: no machine-checked domain-decomposition theory was found before this one (to be re-verified at launch);
  - link to the blueprint and the Lean project, public on launch day (O9).
- **Statistics to hit:** the number of theorems checked, the number of lines of Lean, and `sorry` count $0$, all read from the Lean project at build time.
- **Copy, drafts:** *"Not a promise. A proof."* · *"Checked by machine, line by line."*
- **Visual and motion:** the blueprint's dependency graph, redrawn in the site's palette: nodes turn from outline to filled as the reader scrolls, in dependency order, ending on the headline theorem. One theorem's statement is shown in clean typeset mathematics, beside its plain-English reading.
- **AI visual prompts:** none required. The graph is the picture.

### Section 7 — The vision (the dark chapter)

- **Topics:** three scenarios, each a full-bleed image with a one-paragraph story, the agent graph drawn over it, and a *today* line in small type:
  - **the whole race car**, air, heat, tyres, brakes, power;
  - **a wildfire, forecast and steered**;
  - **the agentic Moon base**.
- **Statistics to hit:** none. This is the only section without numbers, and it says so: *"Vision. Not built yet."*
- **Copy:** the stories in [[vision-scenarios-and-image-prompts]] §2–§4. Each ends on its *today* line, which says what exists toward it.
- **Visual and motion:** each image pins while its agent graph draws on (nodes at the parts, edges along the ports, cyan pulses along the edges), then releases to the next. The port type labels (`MECH`, `THERM`, `ELEC`, `ROT`, `ADVEC`) appear on hover or tap.
- **AI visual prompts:** the clean plates 1a, 2a and 3a, and their details, in [[vision-scenarios-and-image-prompts]]. The overlays are drawn in code, not generated.

### Section 8 — The path

- **Topics:** four stages as a horizontal timeline:
  - **Stage 0**: 2-D diffusion on random charts;
  - **Stage 1**: advection, elasticity and current, plus a multiphysics seam;
  - **Stage 2**: fluid flow, and the fine-tuning comparison;
  - **Stage 3**: three dimensions and the scenarios.
  
  Each stage has what it proves and its gate, the iteration count first. Cost appears as an order of magnitude only if the training chat has measured Stage 0's throughput by launch; otherwise the timeline shows no cost at all.
- **Statistics to hit:** none until measured ([[chart-operator-training-and-cost]] §5's figures are estimates and stay off the site).
- **Copy, drafts:** *"From a square of heat to a whole world."*
- **Visual and motion:** a horizontal line with four nodes that light in turn; each opens a small card on hover or tap.
- **AI visual prompts:** none.

### Section 9 — Footer

- **Links:** Try the demo (install page) · Read the architecture (the arXiv-style page) · The proofs (the blueprint) · The code (public repositories, at launch) · Contact.
- **Fine print:** how every number was measured, and a link to the evidence register's public version. *"Illustrations are marked. Results link to their records."*

---

## 3. The install page (its own page, linked from every *Try the demo*)

- **Topics:**
  - what the demo is, in one sentence;
  - requirements (Windows 10 or 11; macOS on Apple silicon if the torch pin demands it, as the RaceLab bundle found);
  - **one command per system**;
  - what it downloads and roughly how big (read from the installer's own record);
  - how long the first launch takes;
  - what the visitor will see, with a real screenshot;
  - how to uninstall;
  - the licence notes (Gmsh is installed from PyPI, not bundled).
- **Style:** a single calm column, two tabs (Windows and macOS), a copy button on each command.

---

## 4. Handing over the images

As in [[vision-scenarios-and-image-prompts]] §5: each generated image is saved with its prompt, the generator and its version, and the date; it is checked for text, marks and palette; it is labelled **Illustration**; and overlays are drawn in code.

---

## See Also

- [[proposal-website-plan]] — the build, the claims pipeline and the honesty rules
- [[website-evidence-and-citations]] — every id used above
- [[vision-scenarios-and-image-prompts]] — the scenario stories and their prompts
- [[demo-finish-plan]] — the install the *Try the demo* buttons lead to
- [[chart-operator-architecture]] · [[formal-proofs-plan]] — what sections 5 and 6 draw on
