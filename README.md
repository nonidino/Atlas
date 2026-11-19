# Atlas

Atlas splits a simulation into pieces, gives each piece an expert solver
(learned, classical, or a group of smaller experts), and lets the experts agree
at their seams through one standard interface, checked before anything runs.

**The website:** <https://nonidino.github.io/Atlas/> covers what works today,
the evidence behind every number, the architecture, and the Lean proofs.

## Run the studio

The studio (`atlas/studio/`) is the demo, and Atlas's one UI: every new
feature with a page is built into it. You draw the domain, choose the
equations and the regime, cut it into pieces and give each piece an expert,
run the pieces against the undivided solve on your own machine, and read the
chart operator's records. Everything it needs is on this branch.

**Windows** (PowerShell):

```powershell
git clone --depth 1 --branch atlas-0.1 https://github.com/nonidino/Atlas.git
cd Atlas
.\run.cmd
```

**macOS or Linux:**

```bash
git clone --depth 1 --branch atlas-0.1 https://github.com/nonidino/Atlas.git
cd Atlas
./run.sh
```

The launcher finds Python 3.10, 3.11 or 3.12. It builds a private environment
in `.venv` and installs pinned packages. It then runs a self-test once and
opens the page in your browser. Its options (`--check`, `--no-open`,
`--reinstall`, `--no-torch`, `--no-gmsh`) and requirements are described in
[`atlas/workbench/bundle/README.md`](atlas/workbench/bundle/README.md). That
file was written for a separate install branch that was never pushed. Use the
clone commands above, and read "this folder" as the root of this branch. To
uninstall, delete the folder.

## What is where

| folder | what it holds |
|---|---|
| `atlas/` | the framework, the studio (`atlas/studio/`, the UI; [its README](atlas/studio/README.md)) and the workbench engine it calls (`atlas/workbench/`) |
| `vendor/` | the wind farm's solver, from the build repository ([`vendor/README.md`](vendor/README.md)) |
| `lean/` | the machine-checked proofs (Lean 4, Mathlib) |
| `scripts/`, `tests/` | the experiments and their tests |
| `out/` | the records the website cites |
| `wiki/` | the research notes, an Obsidian vault (start at `wiki/00-start-here.md`) |
| `site/` | the website, published by `.github/workflows/pages.yml` |
| `proposal/` | the architecture page |
| `docs/plans/` | the PoC plans and requirements the wiki cites by name: the rocket (PoC 4), the classical plan and its compute plan, and RaceLab (PoC 3) |
