# Launching the website — the owner's checklist, in order

**Type:** Concept page — **launch checklist** (folder: `Atlas 0.1/atlas-0.1-proposal/website/`)
**Status:** written 2026-10-02 by the website chat (W354, step 9). **Nothing on this list has been done.** Every step is outward-facing (it creates a repository, publishes something, or makes something public), so each waits for the owner's own go-ahead, one by one. The site itself is complete in `site/` on the branch `claude/practical-cray-n0vz9e` (built on `atlas-0.1`), with placeholders where the owner's illustrations go ([[website-image-prompts]]).
**Hub:** [[00-proposal-workstreams]] · **Plan:** [[proposal-website-plan]] · **Outline:** [[website-outline]] · **Evidence:** [[website-evidence-and-citations]]

---

## 0. Before launch day

1. **The illustrations.** Generate the four required images of [[website-image-prompts]] §0 (H1, 1a, 2a, 3a, each in two crops) and hand them to a chat, which places them, moves the vision graphs onto their parts and re-walks the page. Check the image tool's terms for commercial use.
2. **Decide the names**, or keep the defaults written into `site/data/site.json`:
   - the site's public repository: `nonidino/atlas-proposal`, served at `https://nonidino.github.io/atlas-proposal/`;
   - the demo's public repository: `nonidino/atlas-workbench`, holding the branch `atlas-workbench`.
   If either changes, edit `site/data/site.json`, then run `python scripts/site_claims.py`, `python scripts/site_blueprint.py` and `python scripts/site_build.py`.
3. **One fix the website chat could not make, because the file is not its to edit:** the architecture page (`proposal/architecture/index.html`) still says, in its section on the guarantees, that its finite-dimensional theorems *"have not been machine-checked"*. They now are (`Atlas.tier0_headline` and the batch-4 theorems, [[formal-proofs-record]]). **Proposed replacement:** *"These statements are machine-checked in Lean 4 on Mathlib: the tier-0 guarantee as `Atlas.tier0_headline`, with no `sorry` and only Lean's three standard axioms; see the blueprint."* After the edit, `python scripts/site_build.py` copies the page again.
4. **macOS.** If anyone with a Mac can run `./run.sh --check` in a fresh clone of the demo, the install page's *not verified yet* can come off (edit the macOS tab in `site/install.html`). Until then it stays.

---

## 1. The launch, as done (2026-10-02)

**The owner made `nonidino/Atlas` itself public and enabled Pages**, instead of a separate site repository. So the site is published from this repository: the workflow `.github/workflows/pages.yml` uploads `site/` to Pages on every push to `atlas-0.1` that changes it, at `https://nonidino.github.io/Atlas/`. `site/data/site.json` points every link there. The owner's settings: **Settings → Pages → Build and deployment → Source: GitHub Actions**. Images are uploaded to `incoming/images/` (its README says how). The demo's branch `atlas-workbench` still has to be pushed to this repository from the laptop (step 5 below); then `"launched": true` in `site/data/site.json` removes the install page's banner.

**What being public now exposes** beyond the plan's site-only release (O9): the whole vault, the raw source documents in `raw/`, and the full run records, which carry the laptop's name and the user's folder paths. No Poseidon or NeuberNet weights are tracked. The owner may want to weigh `raw/` (other people's papers) in particular.

## 1a. The launch as first planned, for a separate site repository

| # | step | how | check |
|---|---|---|---|
| 1 | **Create the public repository** for the site | GitHub → New repository → `atlas-proposal`, public, no README | it exists and is empty |
| 2 | **Push `site/` to it** | from a clone of this repository: `git subtree split --prefix site -b site-publish`, then `git push https://github.com/nonidino/atlas-proposal.git site-publish:main` (keeps the site's history; a plain copy of the folder also works) | the repository's root holds `index.html`, `.nojekyll`, `records/`, `proofs/`, `architecture/` |
| 3 | **Enable GitHub Pages** | the new repository → Settings → Pages → Deploy from a branch → `main`, folder `/ (root)` | after a minute, `https://nonidino.github.io/atlas-proposal/` loads |
| 4 | **The blueprint and the architecture page go public with it** | nothing more to do: both are inside `site/` (`proofs/blueprint/`, `architecture/`), with the Lean sources in `proofs/lean/`, under O9 | open `proofs/blueprint/index.html` and `architecture/index.html` on the live site; the mathematics renders (MathJax loads from `cdn.jsdelivr.net`) |
| 5 | **Make the demo's install branch reachable** | create the public repository `atlas-workbench`, and push the branch built by `scripts/build_workbench_bundle.py` to it as `atlas-workbench` (or attach it to a release); then set `"launched": true` in `site/data/site.json`, run `python scripts/site_build.py` and push `site/` again, which removes the *before launch* banner | the three commands on the install page work in a fresh terminal |
| 6 | **Re-run the claims test** | `python -m pytest tests/test_site_claims.py -q` in this repository | 8 passed |
| 7 | **The final link check** | on the live site: every button and link, the three pages, the blueprint's dependency graph, two record links, two literature links; the browser console shows no errors | nothing returns 404 |

---

## 2. What is public after launch, and what is not

- **Public:** the site; the cleaned copies of the cited records (the computer's name, process ids, local paths and log excerpts removed; each copy names its original and its checksum); the blueprint; the architecture page; the Lean sources; the demo's install branch.
- **Not public:** this repository, the vault, the run records in full, the learned case's training data. NeuberNet appears nowhere on the site, and no weights of Poseidon (the test checks both).

---

## See Also

- [[website-image-prompts]] — the images to make before launch
- [[website-evidence-and-citations]] — every number on the site
- [[website-design-notes]] — the reference sites and the look
- [[proposal-website-plan]] — the plan this completes
