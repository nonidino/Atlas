# Context router: what to read for each kind of task

**Type:** Navigation page, for LLM sessions (humans: [[00-start-here]]).
**Status:** created 2026-11-18 in the wiki lint and compaction pass. Keep it short: it is read at the start of most sessions.
**Purpose:** get the right context with the fewest tokens. Read this page, then only the pages its route names. [[index]] is the full catalog (about 12k tokens); open it only when a route does not name what you need.

---

## 1. Always first

1. **[[00-atlas-current-state]]** (about 4k tokens): what is built, measured, open and decided now, and which pages are history. For any Atlas task.
2. The route below for your task.

**Read cheaply.** Most pages open with a `**Status:**` line that says what is true now; many are 40–220 kB. Read the first 15 lines, list the headings (`grep -n '^#' page.md`), then read only the section you need. Never read a log, an archive or a record end to end unless the task names it.

## 2. Routes

| task | read, in order | then, only if needed |
|---|---|---|
| **Build** a feature with a page (the studio) | [[00-atlas-current-state]] §7 · [[integrated-ui-plan]] §12 · `atlas/studio/README.md` | the plan's phase section (P0–P6) nearest your feature |
| **Build** in the framework (a case, a port, an expert) | [[atlas-implementation]] · [[end-to-end-architecture-spec]] (the layer you touch) · [[port-algebra-atlas-0.1]] | [[showcase-gallery]] for a case like yours; [[gap-worklist]] for its W-item |
| **Build** the chart operator (Stage 0b on) | [[chart-operator-stage-0-rework]] §20 · [[chart-operator-design-decisions]] (latest entries) · [[integrated-ui-plan]] §7 | [[chart-operator-general-form]] · [[chart-operator-training-and-cost]] |
| **Build** a Lean proof | [[formal-proofs-implementation-plan]] · [[formal-proofs-record]] (status and the batch's section) | [[formal-proofs-plan]] (T1–T23) |
| **Build** on the rocket | [[rocket-episode-pickup]] | [[case-study-rocket-bc-seam-atlas-0.1]] (the § named there) |
| **Reference**: a formula, symbol or bound | [[mathematical-compendium]] (§2 notation table, then the item) | the derivation page it links |
| **Reference**: a measured number | [[outcome-evidence-ledger]] | the claim page; then the record in `out/` |
| **Reference**: a W-item | `grep -n 'W123' "wiki/concepts/Atlas 0.1/common/gap-worklist.md"` (board row, then its tier) | the archive it points to |
| **Reference**: a paper or code | [[code-and-papers]] | the paper's summary under `summaries/` |
| **Design** (architecture, a new expert or family) | [[00-atlas-current-state]] · [[chart-operator-architecture]] · [[chart-operator-family-catalog]] · [[outcome-c5-requirements-for-dd-native-experts]] | [[dd-neural-prior-art-2026]] · [[prior-art-and-novelty-atlas-0.1]] · [[expert-donor-survey]] |
| **Design** (coupling, theory) | [[master-error-bound]] · [[general-coupling-scheme]] · [[interface-transfer-theory]] | [[composition-error-theory]] · [[plug-in-composition-theorems]] · [[theory-closure-audit]] · [[atlas-and-standard-dd-theory]] |
| **Decision** (what next, go or no-go) | [[00-atlas-current-state]] §3–§6 · [[00-atlas-0.1-outcome]] · [[open-problems-atlas-0.1]] | [[f1-pathmap-and-end-goal]] (the end goal) · [[generalization-requirements]] |
| **Brainstorm** (new directions) | [[00-atlas-current-state]] · [[pfm-purpose-and-direction]] · [[pfm-architecture-approaches]] | [[learned-contribution-kill-tests]] (what has already been killed) · [[open-problems-atlas-0.1]] · the research layer below |
| **Claims and novelty** (proposal, website) | [[00-atlas-0.1-outcome]] · [[prior-art-and-novelty-atlas-0.1]] · [[website-evidence-and-citations]] | [[poc2-novelty-audit]] · [[dd-neural-prior-art-2026]] |
| **Ingest** a source | CLAUDE.md "Ingest" · the theme's section of [[index]] | the concept pages the source touches |
| **Lint** the wiki | CLAUDE.md "Lint" · `python scripts/link_scan.py` · `python scripts/vault_scan.py wiki` | the latest `lint` entry in [[log]] |

## 3. The research layer (before Atlas)

The preliminary research is stable and small: start at [[physics-foundation-models]] and [[pfm-architecture-approaches]], then the theme hubs [[00-token-representation-overview]] and [[00-attention-overview]], and the source summaries listed in [[index]]. The Noether 1.0 and 1.1 designs ([[00-initial-model-overview]], [[00-noether-1.1-overview]]) are dormant: read them for an idea's origin, not for the current design.

## 4. Do not read unless the task names them

| page or folder | size | why it exists |
|---|---|---|
| [[log]], `log/` | 0.2 MB, 1.2 MB | append-only history; read the last few entries (`grep -n '^## \[' wiki/log.md \| tail`) |
| [[index-digest]] | 0.35 MB | the index's long row texts, kept word for word |
| [[gap-worklist-tiers-0-37]], [[gap-worklist-tiers-38-75]] | 0.6 MB | archived worklist tiers; the board in [[gap-worklist]] has every status |
| [[tier0-measurements]] | 0.4 MB | the measurement record from Tier 0; tests read it |
| [[chart-operator-kernel-exact-forms]], [[chart-operator-stage-0]] | 0.28 MB | Stage 0's history; the rework's §20 summarises what carries over |
| the implementation logs ([[wind-farm-implementation-log]], [[atlas-0.1-implementation-log]], [[implementation-log]]) | 0.3 MB | append-only build records |
| `case-study-rocket-ascent/implementation/` | 0.25 MB | the rocket's first plan, replaced by CS-21 |
| `atlas-0.1-outcome/interactive/` | 6.4 MB of HTML | viewers for the website; never read as text |

## 5. Conventions a session needs

Bare `[[page-name]]` links only. LaTeX for all mathematics, written with the Write or Edit tool (never through a heredoc or a Python string literal). Label speculation `**[AI Inference]:**`. Page names are unique across folders. Many pages under `Atlas 0.1/` are read by tests by path: grep `tests/` and `scripts/` before moving or renaming one.
