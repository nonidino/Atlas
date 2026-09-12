# Tier 51 — W214 first (it is tiny, and it may retire the whole corruption story), then W215, then Task 3

## Where you are

Working directory: `C:\Users\Nauni\OneDrive\Desktop\Foundation_Model` — branch
`atlas-0.1`, HEAD `4ebf347` (Tier 50). **`2323d2b` (Tier 49) is pushed; Tier 50
is committed locally and NOT pushed.** One repo, two layers:

    wiki/      the research vault — 229 pages, LLM-maintained. Read CLAUDE.md.
    atlas/     the Atlas composition framework — nine layers, a seven-hypothesis
               envelope, a three-verdict compiler. 22 case modules.
    tests/     51 files, 1307 tests passing at HEAD.
    scripts/   per-W-number measurement scripts, plus vault_scan.py and
               run_suite.py.

Verify before you start — `git log --oneline -1`, and `python scripts/run_suite.py`
if you want the count confirmed (389 s).

**Next free worklist number: W216. This is Tier 51.** Next free case-study
numbers: **CS-19** on the ladder, **CS-S6** off it.

---

## Read first, in this order

1. `wiki/concepts/Atlas 0.1/common/corrupted-checkpoint-and-jacobian-fidelity.md`
   — **the whole page.** §4.4 is the mechanism and its boxed line is what W214
   comes out of.
2. `wiki/concepts/Atlas 0.1/common/defect-correction-learned-operator.md` — §2.2
   (the reading that survived), §4 (the entry condition, W204), §7 (the gate and
   where $\alpha^\star$ came from), and the two 2026-09-12 annotations.
3. `wiki/concepts/Atlas 0.1/common/gap-worklist.md` — rows **W214**, **W215**,
   **W204**, **W76**, and Tier 50's whole section.
4. `wiki/concepts/Atlas 0.1/common/case-study-ladder-to-f1.md` §24.
5. `wiki/index.md` and the tail of `wiki/log.md`.

---

## TASK 1 — W214: the shrink overshoots, and testing the fix costs three forward calls

**The cheapest experiment in the package. Do it first.**

Tier 50 measured, on the derivative rather than by inference:

| quantity, slow band | value |
|---|---|
| classical map $\phi_{\text{slow}}$ | $0.4780$ |
| clean column $\psi_{\text{slow}}$ | $0.5578$ — **preserves what the monolith damps** |
| after the shrink, $(1-\alpha^\star)\psi$ at $\alpha^\star = 0.5$ | $\mathbf{0.2789}$ |

The column sits $0.080$ **above** the classical map before the correction and
$0.199$ **below** it after: the shrink overshoots by two and a half times. And
$\alpha^\star = 0.5$ was chosen on $N=2$ as *the only value on the grid at which
the checkpoint converged without falling back* — a **stability** criterion
applied to a **fidelity** quantity.

**Do:** choose $\alpha$ per rung from the probed $\psi_{\text{slow}}$ — matching
$(1-\alpha)\,\psi_{\text{slow}} \approx \phi_{\text{slow}}$ gives
$\alpha \approx 0.143$ at six windows — and re-run the `P` arm there with
**$\alpha^\star = 0.5$ as the control**, on both out-of-sample rungs.
`scripts/w205_corruption_sweep.py` already has the probe (`--stages probe`), and
`Rung.arm(op, alpha=...)` takes the shrink.

**Pre-register what would count**, in the shape of
`defect-correction-learned-operator` §7, **before** running:

- classical calls at the matched $\alpha$ against $99$ (six windows) and $222$
  (twelve);
- whether it beats the **best corrupted direction**, $71$ and $138$ — the bar
  the corruption set by accident;
- whether **G5** moves at all. It almost certainly does not: the $2\times$-coarse
  classical solver still returns $38.1$ classical-equivalents at twelve windows,
  and that is the clause the slot actually fails on.

**Predict, in writing, that it improves the rate and not the verdict.** The
interesting number is the size: if matching $\alpha$ recovers most of the
$28$-call spread the corruption found, the composition layer's own knob was the
whole story and **the checkpoint was never the variable**.

**Budget:** one probe (~90 s) plus two to four arms (55 s at six windows,
$300$–$500$ s at twelve). Half a tier.

---

## TASK 2 — W215: a randomised arm scored as if it were deterministic

Small, structural, and it is what let Tier 48's finding happen.

`DEC_SETTINGS` fixes every setting except the corruption's seed; the artifact
stores one number per arm; and G6's second half is
`pw_["phi_calls"] >= p_["phi_calls"]` — a Bernoulli trial evaluated once, on a
quantity whose across-direction spread is **$69$ calls at six windows and $171$
at twelve**. The spread *grows* with the rung ($77\%$ and $74\%$ of the mean), so
a single draw is **less** informative at scale, not more.

**Do:** let an arm declare whether it is randomised and with what replicate
count, and make a gate clause over a randomised arm read a **distribution** — a
count, a median and a spread — rather than a value. Re-judge Tier 48's G6 under
it and report **not measurable as posed** rather than *failed*.

Relates to **W106**, the reproducibility-floor row: the same defect one level up,
on an arm rather than on a probe.

---

## TASK 3 — is there a cheap *correct* coarse competitor across a coupled seam?

**Unchanged from the last three briefs and still not started.** Only after Tasks
1 and 2. Needs **no download, no new expert and no training run.**

Tier 48's G5 charges a learned expert against the cheapest *classical*
alternative in the same slot. In 2-D single-family flow that alternative is
brutal: a $2\times$-coarse classical solver cost $0.197$, $0.132$, $0.0320$ of a
classical step at two, six and twelve windows — a $\mathbf{7.0\times}$ saving at
twelve — while the checkpoint's per-call price fell $3.43 \to 2.01 \to 0.296$ and
it still did not pay.

**The open question: a coupled seam may have no valid coarse surrogate at all.**
If restriction/prolongation stops commuting with the coupling, G5's competitor
collapses to the cold march itself and the bar drops enormously.

**Do:** build the `C` arm's analogue (`restrict2` → coarse monolith →
`prolong2`, as in `scripts/w202_kill_tests.py`) across the coupled seams of the
**existing classical 2-D graphs** — CS-12 `wing_fsi`, CS-13 `cooling_loop`,
CS-14 `powertrain` — and measure whether it stays **correct** (defect correction
with it still converges to the classical settled state) and **cheap** (its cost
ratio against a classical step there, against $0.197 / 0.132 / 0.0320$).

**Mark clearly that this measures the COMPETITOR, not a learned expert.** If the
coarse map stays valid and cheap across a seam, that strengthens the Tier 48
negative and makes a 3-D experiment *less* attractive. If it breaks, say how.

---

## Cheap high-leverage rows, if a task finishes early

**W203** (boundary data declared as problem data — small and structural),
**W208** ($\Theta$ from a power iteration on $I - D\Phi$ at $w^\star$ rather than
from a march), **W209** (the union that marches and the union that compiles are
not the same object), **W195**.

---

## Scope realism

Task 1 is **half a tier and goes first** — three forward calls of measurement and
a handful of arms, and it may retire the whole corruption episode as a story
about $\alpha$. Task 2 is half a tier. Task 3 is a tier of its own. **An honest
partial tier is worth more here than three thin ones.** Name what you did not do,
in a section headed **"What this tier did NOT do, named"**.

---

## Constraints — not negotiable

- **NeuberNet is UNLICENSED and local-only.** Only derived numbers are
  committed, no test loads it, and **ask before copying it anywhere**.
- **Do not download anything without asking first.** Reading papers is fine.
- **Ask before:** other weights; datasets, pip installs or cloned repos; GPU
  rental (a vast.ai CLI is set up and authed); copying NeuberNet anywhere; any
  training or fine-tuning run over ~2 hours.
- **Leave alone and never stage:**
  `wiki/concepts/Atlas 0.1/common/case-study-thermal-strain-atlas-0.1.md`. It
  shows as modified; it is `autocrlf` normalisation, not a change.
- **This file** is deliberately **untracked**. Update it; do not commit it.
- **Push only when asked.** Tier 49 was pushed on instruction; **Tier 50 is
  committed and not pushed.**

---

## Traps this project has already paid for

*(Everything in the previous brief still applies. The ones Tier 50 added are
first.)*

- **`np.savez` appends `.npz` to the filename**, so `savez(tmp);
  os.replace(tmp, final)` with `tmp = final + ".tmp"` writes `final.npz.tmp.npz`
  and dies with `FileNotFoundError` — which the `_retry` wrapper does **not**
  catch, since it only retries `PermissionError`. Name the temp file so it
  already ends in `.npz`.
- **`hash()` of a str is salted per process** unless `PYTHONHASHSEED` is set, so
  seeding an RNG from it makes a paired comparison unreproducible between runs.
  Use an explicit integer.
- **A duck-typed loop over a result dict will find your own `settings` key.**
  A loop looking for any entry with a `bands` field found the settings block,
  which holds the band *definitions* as a list. Skip by name.
- **`manual_seed(s)` fixes a DIRECTION, not a draw.** Scaling one fixed noise
  pattern by $\sigma$ gives a single ray through weight space at several radii,
  so seeds across magnitudes are not independent samples. Sample many seeds at
  the magnitude the question is about.
- **A verdict rule can mis-score its own control.** W205's first rule counted the
  $\sigma \to 0$ continuity control's failure to beat the clean checkpoint as
  evidence of noise, which is backwards. Fix it **before** reading any cell, and
  record that you did.
- **Verify today's date from the environment before stamping anything.**
  `wiki/log.md` is append-only and a wrong date cannot be repaired.
- **Author and patch wiki pages with the Write/Edit tools, never through a Bash
  heredoc**, and **never put LaTeX in a Python string literal** — a `\mathbf` in
  a `.py` patch script is an invalid escape that runs and is wrong. Check any
  scripted `.py` edit with
  `python -W error::SyntaxWarning -c "import ast; ast.parse(open(p, encoding='utf-8').read())"`.
- **Assert every `str.replace`'s match count.** An unasserted replace that
  matches nothing fails silently.
- **The console is cp1252.** `print()` of any non-ASCII raises, *after* the file
  is written; encode with `.encode('ascii','replace')` before printing vault text.
- **Set `KMP_DUPLICATE_LIB_OK=TRUE` before any import** mixing torch with numpy
  linalg; **disable TF32**; **assert the flags**. Set `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1` in every process that loads Poseidon.
- **Redirect long runs to a file and persist state as you go.** A pipe to `tail`
  hides progress until exit.
- **Check `tasklist` for stray processes before trusting any timing**, and quote
  ratios rather than wall-clock numbers.
- **A criterion in prose and its evaluation in code drift, permissively.** Assert
  them against each other; distinguish **not measured** from **failed**.
- **Closing a defect breaks its tests.** Budget a rewrite per W-number and keep
  the diagnosis — or, as Tier 50 did, keep the assertions and annotate the
  docstrings when only the interpretation moved.

---

## How to finish the tier

- Theory/record page(s) in `wiki/concepts/Atlas 0.1/common/`.
- Row(s) in `wiki/index.md`.
- Append to `wiki/log.md`: `## [YYYY-MM-DD] tier 51 | ...`, with a **"How it was
  run"** block carrying the exact commands, the suite total, and
  **Changed:/Added:** lines.
- A **Tier 51** section in `gap-worklist.md` starting at **W216**, closing or
  annotating every row you touched.
- **§25** in `case-study-ladder-to-f1.md`, in the shape of §14–§24, including
  **"What this tier did NOT do, named"**.
- Update `f1-pathmap-and-end-goal.md` §3.3 if any rung's status moves.
- Tests in `tests/test_tier51_<topic>.py`.
- Artifacts in `out/<row>/` with an explicit `.gitignore` allowlist pair and a
  comment saying **what rebuilding actually costs**.
- `python scripts/vault_scan.py wiki` → **0 problems**.
- Full suite via `python scripts/run_suite.py`. Record the total. **Re-run it if
  you change any test after it starts.**
- **Commit on `atlas-0.1`, staging by explicit path — never `git add -A`** —
  with an **ASCII-only** message ending:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

---

## Where the programme stands

**Two halves, still in very different places.**

**The coupling half** has climbed to rung 9 and marched it (Tier 49, CS-18): the
union of eighteen agents across five families marches, its three joins' receiver
balances are measured over a march with their null arms, and the joins'
composition errors **add** rather than compounding — but below the flow's own
noise, so G3 is reported **not resolved** rather than passed. The three clocks
are $400$ apart in seconds and no single march spans them.

**The foundation-model half** still has **no learned expert admitted with a
nonzero contribution** — and the sentence now has a number. In the one certified
slot the checkpoint's learned content is worth **13 classical calls** over the
identity, against a **69-call** spread from perturbing its own weights by $3\%$:
nonzero, and about a fifth of its own noise floor. Tier 50 also relocated the
defect — the composition layer's shrink $\alpha^\star = 0.5$ overshoots the
classical map's slow-mode response by two and a half times — so **W214 may show
the checkpoint was never the variable.**

### Rung status (pathmap §3.3)

| rung | status |
|---|---|
| **0**–**2** | built |
| **3** | built as a **compile**, out of order (CS-17) |
| **4** | measured, and qualified |
| **5**, **6**, **7** | built **classically** — CS-12, CS-13, CS-14 |
| **8** | first step done, rung open — a named hole |
| **9** | **built, compiled and MARCHED (CS-18)**: G1 met, G2 measured over a march with its null arms, G3 built and **NOT RESOLVED**. Not complete |
| **10** | small graphs only; the union march is **not** differentiable through the devices (W209) |
| **11** | gate as written has not been posed |
| **12** | a different research programme |
