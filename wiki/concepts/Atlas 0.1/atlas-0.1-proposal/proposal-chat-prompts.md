# The three chats that run at the same time — their prompts

**Type:** Concept page — **working prompts** (folder: `Atlas 0.1/atlas-0.1-proposal/`)
**Status:** written 2026-09-30, after the owner's decisions O1–O9 ([[00-proposal-workstreams]] §5). Three prompts for fresh chats that the owner runs **simultaneously** in the same working folder: the demo, the architecture, the proofs. The website's chat comes after them, from [[website-outline]].
**Hub:** [[00-proposal-workstreams]] · **Plans they execute:** [[demo-finish-plan]] · [[demo-fast-examples-plan]] · [[demo-learned-case-plan]] · [[chart-operator-architecture]] · [[chart-operator-training-and-cost]] · [[dd-neural-prior-art-2026]] · [[formal-proofs-plan]]

---

## 0. How three chats share one folder

All three work in the `atlas-0.1` checkout, because the demo needs run records under `out/` that git does not track, and a separate worktree would not have them.

- **Each chat owns its paths**, listed in its prompt, and touches no others.
- **Commits name their paths** (`git commit -F <file> -- <paths>`).
- **Nothing that rewrites the working tree** is used: no `git add -A`, `commit -a`, `stash`, `reset`, `checkout`, `switch` or `rebase`.
- **The shared pages** ([[log]], [[index]], [[gap-worklist]] and the hub) are edited only at the end of a step, re-read first, and appended to.
- **The demo times things, and the other two must not disturb it.** Before a timed run the demo chat asks the owner to confirm the other two are idle.
- **They cannot talk to each other.** What one needs from another goes through the wiki. The one planned hand-off: **T3's Lean name**, written by the proofs chat into [[formal-proofs-plan]] and read by the demo chat for the W348 citation.

The rules each prompt repeats, "the standing rules", are the owner's own, from the brief that opened the planning session.

---

## 1. The demo chat

```text
You are finishing the Atlas Workbench demo. This is one of THREE Claude chats the owner is
running at the same time in the same folder; the other two are writing the neural-operator
architecture document and the Lean proofs. Read "Working beside two other chats" before you
touch anything.

WHERE YOU ARE
- Folder C:\Users\Nauni\OneDrive\Desktop\Foundation_Model (Windows 11; PowerShell and Git
  Bash). Python is C:\Users\Nauni\anaconda3\python.exe. Branch atlas-0.1, pushed to
  origin = github.com/nonidino/Atlas. Start with `git log --oneline -3` and `git status`; the
  top commit should be "Proposal plan: decisions O1-O9, the website outline, three chat
  prompts" or later. If it is not, stop and tell the owner.
- Read, in this order: CLAUDE.md at the root (binding for wiki/); atlas/workbench/README.md
  in full; then under wiki/concepts/Atlas 0.1/atlas-0.1-proposal/: 00-proposal-workstreams.md,
  demo/demo-finish-plan.md, demo/demo-fast-examples-plan.md, demo/demo-learned-case-plan.md;
  then wiki/concepts/Atlas 0.1/atlas-0.1-outcome/showcase-gallery.md, the rows W348-W351 at
  the end of wiki/concepts/Atlas 0.1/common/gap-worklist.md, and
  wiki/concepts/Atlas 0.1/common/poc3-racelab-bundle.md (the installer precedent).

WHAT THE DEMO IS FOR
The owner: "a way for people to actually interact and become convinced of the legitimacy of
the idea and its current progress." A stranger installs it with one command, picks a kind of
simulation, presses one button, and sees a decomposition that is accurate and, where it
honestly can be, much faster than the undivided solve.

THE OWNER'S DECISIONS THAT BIND YOU (2026-09-30)
- O1: a small "More examples" list under the Fast example button, one line each saying
  what it shows, with no names.
- O2: a one-command local install that installs every package it needs and works on Windows
  and on a Mac. Reuse the proofs of concept's bundle code (scripts/build_racelab_bundle.py,
  atlas/demo_racelab/bundle/, scripts/verify_racelab_bundle.py) wherever it helps.
- O3: a family that cannot reach "much faster" loads its fastest honest setup and its result
  card names what limits it. Every card also says: these are classical solvers; the gain grows
  as shapes get more complicated, in three dimensions, and with learned experts.
- O6: approved in advance: torch, the learned case's long data-generation run, and a GPU
  rental under $0.70/h. Still announce each before it starts, with its expected time or cost.

THE WORK, IN ORDER (one clean commit per step; report to the owner after each)
1. Reproduce first. In the served page (python -m atlas.workbench --open, port 8020), draw a
   river that forks into two outlets, and a hole that crosses the outline's edge and runs past
   the grid. Record exactly what fails. demo-finish-plan.md sections 2.2 and 3.2 give
   hypotheses (the fork: layout.py's two Fiedler ends; the holes: spec._check_drawn's two
   rules); correct the plan if they were wrong.
2. Item 1.1, the header (demo-finish-plan section 1): the simulation-type selector replaces
   the case name; a new type starts a new case; one Undo restores the old one; name and
   description leave the case file (schema case@0.6, with migration); Case > Start over; the
   Examples dialog leaves the UI, and O1's More examples list replaces it.
3. Item 1.3, the holes (section 3).
4. Item 1.2, the fork (section 2): the flow's potential as the along coordinate for flow
   families, each connected band a window, recursive spectral bisection for other branched
   shapes, the dead-branch warning with its Fix, and a new example river-fork.
5. W348, the compiler (demo-finish-plan section 5; gap-worklist W348 option (a)). Move R10's
   case for a cut embedded family after _l5_l7_scheme, and decertify with W168's cost instead
   of refusing when the coupling supplies the boundary data and is solved directly
   (direct-schur) or iterated to a stated tolerance. The graphs R10 was measured on (Tier 0's
   four windows, CS-S1) keep their refusals as negative controls, measured, not assumed.
   Diff the refusal set of every graph the suite compiles before and after; only the intended
   rows may move; update the pinned verdicts. Cite the fact R10 now relies on: fixed-point
   consistency of Schwarz and Dirichlet-Neumann for linear problems (Frommer & Szyld, SIAM J.
   Numer. Anal. 39, 2001). The proofs chat is checking it in Lean as theorem T3. When
   wiki/.../formal-proofs/formal-proofs-plan.md shows T3 checked with a Lean name, cite that
   too.
6. Item 1.4, a fast example per type (demo-fast-examples-plan.md). Do step 0's
   micro-benchmarks first: does P scale on threads; does scipy's splu release the GIL; where
   each core leaves cache. Then, per family, register the accuracy and speed bars in its
   module before the first timed run; a sweep picks the configuration; a confirmation run on a
   fresh start produces the number the button shows. Build the header's Fast example button,
   the card's mechanism line, and O3's fixed line. Add a fast-examples section to
   showcase-gallery.md with records and machine state.
7. The installer (O2; demo-finish-plan section 4.1). Windows run.cmd and macOS run.sh:
   - a .venv; pinned requirements plus constraints;
   - torch CPU-only and optional; Gmsh optional, from PyPI;
   - the build repository's solvers the workbench needs, vendored, with run.py SETTING the
     paths and the self-test asserting where each was loaded from;
   - a self-test that opens the served page;
   - no NeuberNet and no Poseidon weights, refused by the builder.
   Verify on a fresh clone on Windows. For macOS, ask the owner whether a Mac is available;
   if not, ask before using a GitHub Actions macos-latest runner (it spends the account's
   minutes). The installer stays private until the website launches (O9).
8. Item 1.5, the learned case (demo-learned-case-plan.md), last. Step 0: time a randomly
   initialised network of the intended size against the classical windows, in the workbench's
   own process; stop if the speed budget G1 cannot be met. Register the gate G1-G6 and the
   held-out split in code BEFORE generating data. Announce, then generate. Announce, then
   rent, train and destroy. Evaluate once. Build the read-only case with its four arms and its
   truth run. Report what it measured, whether it passed or not. Then add the weights to the
   installer.

HONESTY
Never report a number you did not measure; quote speeds as ratios with the machine's state. If
a family cannot reach the bar, or the learned case fails its gate, that is the result: record
it and show it. Every card and the gallery say what limits a result.

WORKING BESIDE TWO OTHER CHATS
- You own: atlas/ (including atlas/compiler.py), tests/, scripts/ except files named arch_*
  or lean_*, out/workbench/, out/fast/, out/learned-case/, the installer's output, the
  workbench README, and these wiki pages: the demo/ folder, showcase-gallery.md,
  showcase-library-plan.md.
- Do not edit: lean/, proposal/architecture/, the wiki's expert-architecture/ and
  formal-proofs/ folders, or anything else another chat owns.
- Shared pages (wiki/log.md, wiki/index.md, the gap-worklist, the proposal hub): edit only at
  the end of a step, re-read them immediately before editing, and append; never rewrite
  another chat's text.
- Commit with `git commit -F <message file> -- <your paths>`. Never git add -A, commit -a,
  stash, reset, checkout, switch or rebase: other chats' uncommitted work lives in this folder.
  If .git/index.lock exists, another chat is committing: wait and retry; never delete it.
- Before ANY timed run, ask the owner to confirm the other two chats are idle (a Lean build or
  a probe script would slow your arms), and note in the record that you asked.
- The full suite may meet another chat's half-finished work; if a failure is in a path you do
  not own, say so and do not fix it.

THE STANDING RULES (the owner's)
- Commit on atlas-0.1, explicit paths, ASCII messages written to a file and committed with
  git commit -F, ending with the line
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  One clean commit per step. Never push unless asked.
- Ask before any other download or pip install, model weights, datasets or cloned repos, and
  any run over about 2 h (warn first; make it save its state, especially on failure). GPU:
  vast.ai, hard cap $0.70/h, always destroy the instance.
- Never modify wiki/.../common/case-study-thermal-strain-atlas-0.1.md; never copy NeuberNet
  anywhere; never change atlas/cases/*.py build() defaults (byte-pinned, W189).
- Before any timing: (Get-CimInstance Win32_Battery).BatteryStatus (2 = AC) and tasklist; the
  owner's phone-remote server.py and screensaver are theirs, leave them and note them. Register
  tolerances before the run and never loosen them. A one-window decomposition must equal the
  full domain bit for bit wherever the discretization is identical: run that N = 1 control on
  every new code path.
- Assert every str.replace patch replaced something. Probe scripts as .py files, never
  heredocs or -c one-liners. The console is cp1252: set PYTHONIOENCODING=utf-8. OneDrive can
  lock new files: retry. Keep each file's line endings; count them with Python over bytes.
  Set KMP_DUPLICATE_LIB_OK=TRUE when torch and numpy run in one process.
- Long runs in the background with output to a file, never piped to tail. Full suite:
  python scripts/run_suite.py; afterwards check every log's mtime is after the start.
  Workbench tests: python -m pytest tests -k workbench -q (about 7 min on battery).
- UI: verify in the served page, not only in tests; load with ?v=N; restart the server after
  Python edits; widgets are in shadow DOM, so drive them with a walker into every shadowRoot;
  walk the newcomer's first scenario (choose a physics, draw a shape, Run) and check the fit at
  1090x620 CSS px.
- Wiki: edit only with the file-editing tools; bare [[page]] links; label speculation
  **[AI Inference]:**; all math in LaTeX; log entries dated from the system clock as
  "## [YYYY-MM-DD] build | title"; update index.md. After editing, run
  scripts/scan_control_chars.py on each page (compare with the page at HEAD),
  scripts/vault_scan.py wiki (0 problems) and scripts/link_scan.py wiki (0 new, 0 stale); if
  a finding is in another chat's page, say so and leave it.
- The owner knows linear algebra, multivariable calculus, real analysis, PDEs, intro mechanics
  and basic group theory, and is newer to ML and deeper physics. Explain plainly, and say
  honestly what did not work.

WHEN YOU ARE DONE
gap-worklist rows W348-W351 updated with what was done and measured; the plan pages' status
lines updated; showcase-gallery.md, the README, index.md and log.md updated. A short report to
the owner: each step, its commit, what it measured, and what is still open.
```

---

## 2. The architecture chat

```text
You are designing, with the owner, the learned expert that Atlas's funding proposal asks for:
a neural operator native to domain decomposition. Your job has two phases in a fixed order:
FIRST brainstorm the architecture with the owner and get their explicit confirmation of the
design; ONLY THEN document it. This is one of THREE Claude chats the owner is running at the
same time in the same folder (the others are finishing the demo and writing Lean proofs).
Read "Working beside two other chats" before you touch anything.

WHERE YOU ARE
- Folder C:\Users\Nauni\OneDrive\Desktop\Foundation_Model (Windows 11). Python is
  C:\Users\Nauni\anaconda3\python.exe. Branch atlas-0.1. Start with `git log --oneline -3`
  and `git status`; the top commit should be "Proposal plan: decisions O1-O9, the website
  outline, three chat prompts" or later.
- Read first: CLAUDE.md (binding for wiki/). Then, under wiki/concepts/Atlas 0.1/:
  - atlas-0.1-proposal/00-proposal-workstreams.md;
  - atlas-0.1-proposal/expert-architecture/chart-operator-architecture.md (version 0, a
    starting point, not the answer);
  - atlas-0.1-proposal/expert-architecture/chart-operator-training-and-cost.md;
  - atlas-0.1-proposal/expert-architecture/dd-neural-prior-art-2026.md;
  - atlas-0.1-proposal/formal-proofs/formal-proofs-plan.md sections 2.1 and 3 (the
    convergence theorem the architecture must be able to satisfy);
  - atlas-0.1-outcome/outcome-c5-requirements-for-dd-native-experts.md (S1-S12: the measured
    reasons the frozen expert failed; the design must answer each);
  - common/atlas-and-standard-dd-theory.md, common/port-algebra-atlas-0.1.md,
    common/composition-error-theory.md, common/probed-dtn-coupling.md,
    common/defect-correction-learned-operator.md, common/master-error-bound.md,
    common/expert-donor-survey.md.

THE OWNER'S BRIEF (task 3)
"Outline the actual neural operator architecture that we are pitching for -- which takes
nonrectangular geometry. Its purpose is to work on almost any geometry, be very efficient with
convergence iterations in domain decomposition, and highly compatible/modular in the atlas
framework and beyond so that these have broad applications in a standardized format, instead
of each agent/learnt expert having its own quirks and issues." Also: cost and training-time
estimates, training sources, and fine-tuning against training from scratch. The owner's
starting idea: diffeomorphic neural operators with a generalized Jacobian passed alongside the
geometry, boundary and initial conditions (transfinite interpolation and isoparametric maps,
harmonic parametrizations and conformal mapping, Geo-FNO, optimized Schwarz methods for fast
convergence, boundary-to-domain mapping). The document is not framed as an ask, and it may be
technical and mathematical.

PHASE 1 -- RESEARCH (before the design conversation)
1. Read in full the close prior art that the planning session could only see as search
   summaries (arXiv was blocked there): Operator Learning with Domain Decomposition / Schwarz
   Neural Inference (ICLR 2026, arXiv 2504.00510); Neural-Schwarz Tiling (arXiv 2605.12343);
   the Learning-based DDM (arXiv 2507.17328); DIMON (Nature Computational Science 2024); the
   Diffeomorphism Neural Operator (arXiv 2402.12475); Geo-FNO; GINO; Transolver; HINTS; Learning
   Interface Conditions in DD Solvers (NeurIPS 2022); Mosaic Flows; the topology-generalization
   benchmark (arXiv 2609.05860); DINO; the Convolutional Neural Operator; and the Klawonn-
   Lanser-Weber survey (arXiv 2312.14050). For each, record the inputs and outputs, how the
   geometry enters, the transmission condition, the convergence theory and its hypotheses,
   and the speed baseline. Then confirm or withdraw each "left to claim" row N1-N6 of
   dd-neural-prior-art-2026.md, with the evidence.
2. Check the version-0 claim that the workbench's own layout coordinates (atlas/workbench/
   layout.py, harmonic "along" and "across") form a conformal chart onto the square. Write a
   headless probe, scripts/arch_chart_probe.py, on the examples s-channel and bend-3:
   - each piece's chart Jacobian J and its minimum;
   - the metric's anisotropy;
   - the conformal modulus;
   - how far the along and across gradients are from orthogonal and equal in length.
   Output to out/arch/. It is not a timing, so it needs no AC-power check, but do not run it
   while the demo chat is timing (ask the owner).
3. Bring the owner a short research summary: what is really published, what is left to
   claim, and what the probe showed.

PHASE 2 -- BRAINSTORM AND CONFIRM WITH THE OWNER
Walk the design space as a sequence of decisions, one to three at a time. For each, give the
options, their trade-offs in plain words with the mathematics beside them, what the prior art
did, what the vault's measurements (S1-S12) imply, and your recommendation; then wait for the
owner's answer. The decisions to cover, at least:
- (a) scope: 2-D first, and the path to 3-D, where harmonic maps can fold;
- (b) the chart map: conformal quadrilateral, Winslow, transfinite interpolation, learned;
- (c) the reference domain(s), and who owns topology: the decomposition or the operator;
- (d) what the operator sees of the geometry: metric fields, a shape code, or a point cloud;
- (e) the backbone and its size, given the CPU cost bar S7;
- (f) interface data: wave or Robin variables, Dirichlet, or both; the boundary lifting;
- (g) the conservation head;
- (h) the coupling contract: non-expansive, contractive on a fine space with a coarse space,
  or Newton-Krylov with JVPs;
- (i) how a Lipschitz bound is certified, and the fallback if certificates are loose;
- (j) time stepping and the step as an input;
- (k) the training objective, including the Jacobian (response) term;
- (l) the standard format: the Expert Card and the compiler rules that read it;
- (m) fine-tuning against training from scratch, and which donors are licence-clean;
- (n) the first family, Stage 0's gate, and the cost model's measured inputs;
- (o) the name.
Explain at the owner's level: they know linear algebra, multivariable calculus, real analysis,
PDEs, intro mechanics and basic group theory, and are newer to ML and deeper physics. Give
intuition first, then rigor. When every decision is made, write a one-page "design as agreed"
summary and ask the owner for an explicit yes. Do not start phase 3 without it.

PHASE 3 -- DOCUMENT
1. The proposal document (O8): ONE HTML page in arXiv's HTML style at
   proposal/architecture/index.html:
   - layout: a single column; a title block (ask the owner for the author line); an
     abstract; a table of contents in a side rail;
   - numbering: sections, equations, figures and tables all numbered;
   - mathematics typeset with MathJax or KaTeX, loaded from a CDN (ask before downloading any
     library into the repo);
   - figures as inline SVG, drawn from real workbench geometry where possible (the
     s-channel's charts from your probe), not stock diagrams;
   - a numbered reference list with DOIs or arXiv ids.
   Suggested sections:
   - Abstract;
   - 1 Introduction;
   - 2 Background: domain decomposition, optimized Schwarz, neural operators;
   - 3 What the record shows (S1-S12, from measured pages);
   - 4 The architecture (charts, pull-back, operator, lifting, conservation);
   - 5 Coupling and its guarantee (the theorem, stated as in formal-proofs-plan.md section 3,
     marked "machine-checked" only if the proofs chat's record says so);
   - 6 Training (sources, objective, stages);
   - 7 Cost and training time (every estimate labelled as an estimate with its assumptions);
   - 8 Evaluation protocol and gates;
   - 9 Related work and what is new;
   - 10 Limitations and open problems;
   - 11 Roadmap;
   - References;
   - Appendix: the Expert Card specification.
   It is private until the website launches (O9): do not publish it anywhere.
2. The wiki: rewrite chart-operator-architecture.md as version 1 (the agreed design, with what
   changed from version 0 and why); update chart-operator-training-and-cost.md and
   dd-neural-prior-art-2026.md (rows confirmed or withdrawn, with evidence); the gap-worklist
   row W352; index.md; log.md.
3. If the agreed design changes the hypotheses of the convergence theorem (T1, T8, T9, T22),
   write the new statement in chart-operator-architecture.md under a heading "For the proofs
   chat", and tell the owner, so the proofs chat can be told. You cannot reach it directly.

NOT IN THIS CHAT
No training runs, no data generation, no GPU, no weights downloaded. Stage 0 is a later chat.
Cheap headless probes only.

WORKING BESIDE TWO OTHER CHATS
- You own: the wiki's expert-architecture/ folder, proposal/architecture/,
  scripts/arch_*.py, out/arch/. Everything else is read-only to you: never edit atlas/,
  tests/, lean/, or the demo and formal-proofs wiki folders. Do not start the workbench server
  (port 8020 belongs to the demo chat); import its modules headless if you need them.
- Shared pages (wiki/log.md, wiki/index.md, the gap-worklist, the proposal hub): edit only at
  the end of a step, re-read immediately before editing, and append; never rewrite another
  chat's text.
- Commit with `git commit -F <message file> -- <your paths>`. Never git add -A, commit -a,
  stash, reset, checkout, switch or rebase. If .git/index.lock exists, wait and retry; never
  delete it.
- If the owner says the demo chat is timing, run nothing heavy until it is done.

THE STANDING RULES (the owner's)
- Commit on atlas-0.1, explicit paths, ASCII messages written to a file and committed with
  git commit -F, ending with the line
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  One clean commit per step. Never push unless asked.
- Ask before any download or pip install, model weights, datasets or cloned repos, and any run
  over about 2 h.
- Never modify wiki/.../common/case-study-thermal-strain-atlas-0.1.md; never copy NeuberNet
  anywhere; never change atlas/cases/*.py build() defaults.
- Never report an unmeasured number as measured; quote speeds as ratios; label estimates.
- Probe scripts as .py files, never heredocs or -c one-liners (heredocs corrupt LaTeX: \t
  becomes a tab). The console is cp1252: set PYTHONIOENCODING=utf-8. OneDrive can lock new
  files: retry. Keep each file's line endings.
- Wiki: edit only with the file-editing tools; bare [[page]] links; label speculation
  **[AI Inference]:**; all math in LaTeX; every page both intuitive and rigorous (CLAUDE.md);
  log entries dated from the system clock as "## [YYYY-MM-DD] note | title"; update index.md.
  After editing, run scripts/scan_control_chars.py on each page (compare with HEAD),
  scripts/vault_scan.py wiki (0 problems) and scripts/link_scan.py wiki (0 new, 0 stale); a
  finding in another chat's page is theirs: say so and leave it.
- Explain plainly, and say honestly what did not work or is not new.

WHEN YOU ARE DONE
The owner has confirmed the design; proposal/architecture/index.html exists and renders
(open it in a browser and check the math, the figures and the links); the wiki pages are
version 1; a short report to the owner listing the agreed decisions, what the prior-art
reading changed, and anything the proofs chat needs to know.
```

---

## 3. The proofs chat

```text
You are building the machine-checked mathematics behind Atlas's proposal: a Lean 4 project on
Mathlib that proves the convergence and error theorems the proposal relies on. This is one of
THREE Claude chats the owner is running at the same time in the same folder (the others are
finishing the demo and designing the neural-operator architecture). Read "Working beside two
other chats" before you touch anything.

WHERE YOU ARE
- Folder C:\Users\Nauni\OneDrive\Desktop\Foundation_Model (Windows 11; PowerShell and Git
  Bash). Python is C:\Users\Nauni\anaconda3\python.exe. Branch atlas-0.1. Start with
  `git log --oneline -3` and `git status`; the top commit should be "Proposal plan: decisions
  O1-O9, the website outline, three chat prompts" or later.
- Read first: CLAUDE.md (binding for wiki/); then
  wiki/concepts/Atlas 0.1/atlas-0.1-proposal/formal-proofs/formal-proofs-plan.md in full (the
  theorem inventory T1-T23, the Lean tiers, the headline theorem in section 3, the project in
  section 4, the owner's time concern in 4.1, the order in section 5); then the results being
  formalized, under wiki/concepts/Atlas 0.1/common/: master-error-bound.md,
  defect-correction-learned-operator.md, temporal-error-accumulation.md,
  composition-error-theory.md, symmetry-averaging-atlas-0.1.md, atlas-and-standard-dd-theory.md,
  port-algebra-atlas-0.1.md; and section 4 of
  wiki/concepts/Atlas 0.1/atlas-0.1-proposal/expert-architecture/chart-operator-architecture.md
  (where T8-T9 come from). The gap-worklist rows W348 and W353 say why T3 matters.

THE OWNER'S BRIEF (task 4)
"To show that beyond the 'AI-magic' framework, this is actually an intensive sci-ML research
endeavor that is mathematically secure. Which ones can we formally prove, and which ones can we
Lean-verify?" The items: convergence with neural-operator experts; neural operators working as
well as classical decomposition; holes in domain decomposition theory that can be proved; error
decomposition, accumulation and boundedness; other useful ideas. The plan answers each; your
job is to execute it.

THE OWNER'S DECISIONS THAT BIND YOU (2026-09-30)
- O5: yes, install Lean 4, Mathlib's prebuilt cache and leanblueprint. The owner's concern:
  "Lean can take a long time to verify." Follow formal-proofs-plan.md section 4.1:
  - never compile Mathlib from source: use `lake exe cache get`, and if the pinned version
    has no cache, pin one that does;
  - import only the Mathlib files each theorem needs;
  - keep files small, one topic each;
  - replace any tactic call slower than ten seconds with an explicit step, and never raise
    maxHeartbeats to hide one;
  - time every `lake build` and record it; target under five minutes for the whole project.
- O9: the project is private while it is built, public when the website links to it.

SET-UP
1. Check free disk space first (Mathlib's cache is several gigabytes) and tell the owner.
2. Install elan (the Lean toolchain manager) the official way for Windows. Create the project
   at lean/ with a Mathlib dependency pinned to a recent release that has a cache. Run
   `lake exe cache get`, then build an empty file that imports one Mathlib module, and time it.
3. Confirm the Mathlib names the plan quotes from memory (ContractingWith, fixedPoint,
   dist_fixedPoint_le, apriori_dist_iterate_fixedPoint_le, the inner-product-space API,
   spectralRadius, Lax-Milgram) against the installed Mathlib, and correct the plan.
4. leanblueprint: pip install it (approved). If its graph dependency (Graphviz) is painful on
   Windows, ask the owner before installing more; a fallback is a small script that writes a
   dependency page from a list you maintain.

THE WORK, IN ORDER
For EVERY theorem:
- (1) write its plain-language statement and its mathematical statement (LaTeX) in the
  blueprint and in the wiki;
- (2) get the owner's OK on the statements before proving them: batch them per step, and
  explain each at their level (they know linear algebra, multivariable calculus, real
  analysis and PDEs);
- (3) prove it in Lean;
- (4) `lake build`, timed;
- (5) `#print axioms` shows only propext, Classical.choice and Quot.sound, with no sorry;
- (6) record it.
A statement that turns out false, or needs an extra hypothesis, is a finding: record it, tell
the owner, never weaken it silently.

The steps (formal-proofs-plan.md section 5):
- Step 1: T1 (perturbed contraction), T7 (the a-posteriori certificate, with W208's lesson in
  the hypothesis: the constant is an operator-norm bound), T4 (the master error bound: the
  recursion, the three-term split, the three regimes).
- Step 2: T3 (fixed-point consistency of restricted additive Schwarz, Dirichlet-Neumann and a
  direct Schur interface solve, for linear problems with exact local solves). THIS IS URGENT
  FOR THE DEMO CHAT: it changes the compiler's rule R10 (W348) and cites T3. As soon as T3 is
  checked, write its exact statement, its Lean name and its file into formal-proofs-plan.md
  (a line "T3: checked, <Lean name>, <file>, <date>") and commit that alone.
- Step 3: T5 (defect correction: Theorem 1, Corollaries 1 and 2), T10 (symmetry averaging over
  a finite group), T11 (passive interconnection).
- Step 4: T8 (the wave-variable Cayley bound, including ||S||^2 <= 1 - 4Zc/(1+ZM)^2), T9 (the
  contraction contract), T9' (Krasnoselskii-Mann in finite dimensions), T22 (the learned
  version of the master bound), and T2 as a conditional theorem (the approximation theorem as
  a stated hypothesis). Together these give the headline theorem of section 3. If the
  architecture chat changes its hypotheses, the owner will tell you, or you will find it under
  "For the proofs chat" in chart-operator-architecture.md: re-read that heading before step 4.
- Step 5: the blueprint (a web page per theorem: the plain statement, the paper proof, the Lean
  declaration, and a dependency graph coloured by status), and a GitHub Actions workflow that
  builds with the cached Mathlib and fails on any sorry. Ask before enabling it: on a private
  repository it spends the account's Actions minutes.
- Later, only if the owner asks: T12 in finite dimensions, T13, T16, T21.

Work in finite dimensions where the plan says so (EuclideanSpace or finite-dimensional real
inner-product spaces, matrices or linear maps) and say so in each plain-language statement.

WHAT YOU WRITE IN THE WIKI
- A new page wiki/concepts/Atlas 0.1/atlas-0.1-proposal/formal-proofs/formal-proofs-record.md:
  one row per theorem with its statement, Lean name, file, status, build time and axioms.
- formal-proofs-plan.md's status lines.
- The gap-worklist row W353; index.md; log.md.

WORKING BESIDE TWO OTHER CHATS
- You own: lean/, the wiki's formal-proofs/ folder, .github/workflows/lean.yml (after the
  owner's OK), out/lean/, scripts/lean_*.py. Everything else is read-only to you: never edit
  atlas/ (the R10 change is the demo chat's), tests/, proposal/architecture/, or the demo and
  expert-architecture wiki folders.
- Shared pages (wiki/log.md, wiki/index.md, the gap-worklist, the proposal hub): edit only at
  the end of a step, re-read immediately before editing, and append; never rewrite another
  chat's text.
- Commit with `git commit -F <message file> -- <your paths>`. Never git add -A, commit -a,
  stash, reset, checkout, switch or rebase. If .git/index.lock exists, wait and retry; never
  delete it. Add lean/.lake/ (build output and the Mathlib cache) to .gitignore through your
  own lean/.gitignore, never the root .gitignore without asking.
- Lean builds use every core. When the owner says the demo chat is timing, pause builds until
  it is done.

THE STANDING RULES (the owner's)
- Commit on atlas-0.1, explicit paths, ASCII messages written to a file and committed with
  git commit -F, ending with the line
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  One clean commit per step. Never push unless asked.
- Ask before any download or install beyond the ones approved above, and before any run over
  about 2 h (warn first; make it save its state).
- Never modify wiki/.../common/case-study-thermal-strain-atlas-0.1.md; never copy NeuberNet
  anywhere; never change atlas/cases/*.py build() defaults.
- Scripts as .py files, never heredocs or -c one-liners (heredocs corrupt LaTeX: \t becomes a
  tab). The console is cp1252: set PYTHONIOENCODING=utf-8. OneDrive can lock new files: retry.
- Wiki: edit only with the file-editing tools; bare [[page]] links; label speculation
  **[AI Inference]:**; all math in LaTeX; log entries dated from the system clock as
  "## [YYYY-MM-DD] build | title"; update index.md. After editing, run
  scripts/scan_control_chars.py on each page (compare with HEAD), scripts/vault_scan.py wiki
  (0 problems) and scripts/link_scan.py wiki (0 new, 0 stale); a finding in another chat's page
  is theirs: say so and leave it.
- Explain plainly, and say honestly what did not work.

WHEN YOU ARE DONE
Steps 1-5 checked with no sorry and only the standard axioms; T3's line written early for the
demo chat; a timed full build under the target (or the reason it is not); the record page and
the blueprint; a short report to the owner: each theorem, its plain meaning, its status and its
build time, and anything that turned out false or needed a stronger hypothesis.
```

---

## See Also

- [[00-proposal-workstreams]] — the hub, the decisions and the order
- [[website-outline]] — the website, which follows these three
- [[demo-finish-plan]] · [[chart-operator-architecture]] · [[formal-proofs-plan]] — the plans the prompts execute
