# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## What This Repository Is

An LLM-maintained knowledge wiki for developing a physics foundation model ("AGI-like" physics FM). The goal is to accumulate and synthesize research across AI, mathematics, and physics into a structured, interlinked knowledge base that supports both conceptual understanding and rigorous theoretical reasoning.

**Code home:** this repository, [github.com/nonidino/Atlas](https://github.com/nonidino/Atlas) (**public**), default branch `atlas-0.1`. It holds the wiki *and* the build: the Atlas framework, the Lean proofs, the experiments and the website. The earlier build repository, [nonidino/physics-foundation-model](https://github.com/nonidino/physics-foundation-model) (private), still holds the wind farm's solver; `vendor/` is a copy of it. Both repos + per-paper reference implementations are catalogued at `wiki/resources/code-and-papers.md`.

> **Status:** the current track is **Atlas 0.1**. A classical framework, its case studies, a workbench demo and 105 machine-checked theorems are built; the proposed learned expert, the chart operator, is designed but not yet built or trained. Everything here is public, so never commit secrets or unlicensed weights.

Layers / folders:
- **`raw/`** — immutable source documents (papers, articles). Never modify these. **Local only**: git-ignored since 2026-10-17, because the repository is public and these are other people's texts. When an ingest finishes, move the source here from `new/`. Because `raw/` is not published, every summary should cite its source's URL or arXiv id. On 2026-10-17 four did not: `aion-1-astronomy`, `eqnn-hep-lhc`, `gns-graph-network-simulators`, `genesis-physics-engine`.
- **`new/`** — inbox for sources awaiting ingestion, also local only. **Do not reorganize or move this folder** — it is the drop point for new material to be processed.
- **`wiki/`** — LLM-maintained markdown pages. You own and maintain this layer entirely.
- **`system/CLAUDE.md`** — the schema document describing this workflow (Karpathy's wiki pattern).
- **`atlas/`** — the composition framework and the workbench (`atlas/workbench/`); **`vendor/`** — the wind farm's solver from the build repository; **`tests/`** — its tests (`python scripts/run_suite.py`).
- **`scripts/`** — one measurement script per W-number, plus the vault checks `vault_scan.py` and `link_scan.py`; **`out/`** — the records the wiki and website cite.
- **`lean/`** — the Lean 4 proofs (build with `scripts/lean_build.py`, never `lake` inside OneDrive).
- **`proposal/`** — the architecture document; **`site/`** — the website, published by `.github/workflows/pages.yml`.
- **`docs/plans/`** — the PoC plans and requirements (`ROCKET-POC-PLAN.md`, `ATLAS-CLASSICAL-PLAN.md`, `COMPUTE-PLAN.md`, `POC3-RACELAB-REQUIREMENTS.md`), cited by file name from the wiki and the code.
- **`run.py`, `run.cmd`, `run.sh`** — the workbench installer and launcher (see `README.md`).

---

## Wiki Structure

```
wiki/
  00-start-here.md      ← conceptual on-ramp + guided reading path (for humans)
  index.md              ← exhaustive catalog of all pages; read this first when answering queries
  index-digest.md       ← the index's long descriptions (rows marked ↗), kept verbatim
  log.md                ← append-only ingest/query/lint record (October 2026 onward)
  log/                  ← earlier months, archived verbatim: log-YYYY-MM.md
  resources/
    code-and-papers.md  ← GitHub repos + per-paper code/arXiv links
  misc/                 ← reading list, session handoffs
  summaries/            ← one page per source document, in themed subfolders:
    foundation-models/  transformers-theory/  neural-operators/  graph-networks/
    hamiltonian-lagrangian/  diffusion-generative/  solvers-and-simulation/  other-methods/
  concepts/             ← synthesized topic pages, in themed subfolders:
    00-pfm-core/  physics-foundations/  ml-building-blocks/
    benchmark-architectures/  extension-architectures/
    adjusted-attention-mechanism/  unified-token-representation/
    Noether 1.0/  Noether 1.1/      ← the first in-house designs
    Atlas 0.1/                      ← the current track (about half the vault):
      common/                       ← the framework's theory, case studies, gap-worklist.md (W-numbers: a status
                                       board + Tiers 76 on; Tiers 0–75 archived verbatim beside it)
      atlas-0.1-outcome/            ← what was measured, claim by claim
      atlas-0.1-proposal/           ← expert architecture, formal proofs, demo, website
      case-study-*/                 ← one folder per case study, with its implementation log
```

Before committing wiki edits, run `python scripts/vault_scan.py wiki` (bytes: LaTeX escape corruption) and `python scripts/link_scan.py` (dead links). Both must report no new problems.

The index and log are the two navigation anchors. Always read `wiki/index.md` first before answering a query or starting an ingest. `00-start-here.md` is the human-facing reading path.

---

## Operations

### Ingest a new source
1. Read the source from `new/`
2. Create `wiki/summaries/<theme>/<slug>.md` in the matching themed subfolder — include: overview, key equations in LaTeX, main results, relevance to the PFM goal, cross-links to related summaries and concepts
3. Create or update `wiki/concepts/<theme>/<topic>.md` for any new concepts introduced
4. Update `wiki/index.md` — add entries under the right theme section, update page count and source count. Keep each row's description to about 300 characters (what the page is and its main result); a longer note goes to `wiki/index-digest.md`, and the row is marked ↗
5. Add a row to `wiki/resources/code-and-papers.md` with the source's arXiv + code repo link
6. Append to `wiki/log.md` — format: `## [YYYY-MM-DD] ingest | Source Title`

### Answer a query
1. Read `wiki/index.md` to identify relevant pages
2. Read those pages
3. Synthesize and answer
4. **File the answer back as a wiki page** if it's a useful synthesis (comparisons, analyses, cross-domain connections). Add it to `wiki/concepts/` or a new category. Log it as `## [YYYY-MM-DD] note | description`.

### Lint the wiki
Check for: contradictions between pages, orphan pages (no inbound links), important concepts mentioned without a dedicated page, missing cross-references, stale claims superseded by newer sources.

---

## Conventions

**Cross-linking:** Every summary links to all relevant concept pages and related summaries. Every concept page links back to relevant summaries and to sibling concepts. Use **bare Obsidian links — `[[page-name]]`, never `[[folder/page-name]]`**. All page filenames are unique, so bare links resolve regardless of which subfolder a page lives in. This keeps the vault reorganization-proof: pages can be moved between theme folders without breaking any links.

**AI Inferences:** Any speculative connection, hypothesis, or implication not directly supported by the source must be labeled `**[AI Inference]:**`. This is mandatory. Mark clearly; never blend with factual content.

**LaTeX:** All mathematics must use LaTeX. Inline: `$...$`. Display: `$$...$$`.

**Dual purpose:** Every page must serve both as (1) an intuitive conceptual explanation and (2) a rigorous theoretical reference. Don't sacrifice one for the other.

**Physics encoding spectrum** (useful framing for evaluating approaches):
1. Data-driven only
2. Soft loss constraints
3. Test-time physics guidance (DPS)
4. Architectural soft biases (variational form)
5. Hard architectural constraints (exact divergence-free, etc.)

---

## Key Pages to Read for Context

- `wiki/00-start-here.md` — conceptual on-ramp + reading path
- `wiki/index.md` — full catalog
- `wiki/concepts/00-pfm-core/physics-foundation-models.md` — the central goal
- `wiki/concepts/00-pfm-core/pfm-architecture-approaches.md` — 7 paradigms compared with pros/cons
- `wiki/resources/code-and-papers.md` — code repo + per-paper implementations
- `wiki/log.md` — what has been ingested and when

## The Nature of this Wiki

In this case, the end goal of this wiki is to make an "AGI-like" foundation model for physics. This could be bridging knowledge across the web or creating an entirely new architecture is yet to be seen, but this wiki is meant to contain knowledge across AI research, mathematics, mathematical modeling, and physics. 

This wiki has two purposes: it needs to serve as an intuitive framework that makes very difficult concepts easy to understand conceptually and it needs to serve as a rigorous theoretical background for sound reasoning.