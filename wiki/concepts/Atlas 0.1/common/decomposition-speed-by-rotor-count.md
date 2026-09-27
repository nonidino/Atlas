# Classical decomposition against the monolith, by rotor count (W346)

**Type:** Concept page — **measurement record** (folder: `Atlas 0.1/common/`)
**Status:** measured 2026-09-26, Tier 89, on the dev laptop (22 logical cores, on mains power, Balanced plan, the process holding a keep-awake request). `scripts/w346_rotor_count_speed.py` → `out/w346/w346.json`; `tests/test_tier89_competitor_and_speed.py`. Predictions B1–B6 were registered in the script before any arm ran (`out/w346/registered.txt`). Worklist row **W346**.
**Related:** [[outcome-c1-decomposition-vs-monolith]] · [[case-study-scaling-ladder-atlas-0.1]] · [[learned-contribution-kill-tests]] · [[atlas-proof-of-concept-1]] · [[poc1-retrospective-and-hybrid-roadmap]] · [[matched-shrink-and-coarse-competitor]] · [[gap-worklist]]

---

## 0. The result, in one paragraph

**The owner's target for C1, measured: with enough rotors, classical domain decomposition does run faster than the full-domain solve.** Across two independent timing draws it is $3.5$–$5.7\times$ faster from 5 to 21 rotors, and $2.1$–$3.7\times$ at 36. Its farm power stays within $2.3$–$8.3\%$ of the full-domain solve's. The speed comes from one mechanism only: parallelism over cache-sized windows.** The same composed column stepped serially is never faster (0.95–1.25× the monolith's cost), and local time stepping saves only 5–8% of sub-steps. The mechanism is a memory cliff. The monolith's cost per cell rises fourfold (from about 2 to 8.7 µs) between 85,000 and 163,000 cells, as its working set leaves the cache. A serial decomposed column falls off the same cliff, because it steps every window as one batched array. Spreading the windows across threads keeps each thread's working set small, and the threaded column is **bit for bit** the serial one (asserted at all 24 rung-and-thread-count pairs). Below 5 rotors the monolith is still in cache and threads only add overhead: 1 and 3 rotors run 1.6× *slower* threaded. The best thread count grows with the farm (4 at 5–12 rotors, 8 at 21, 14 at 36). At a fixed 8 threads the gain falls from 4.5× to 2.0× at 36 rotors, as each thread's share of windows outgrows its cache. **Four of the six registered predictions failed**, and each failure is part of the finding.

---

## 1. The question, and why the earlier record could not answer it

[[outcome-c1-decomposition-vs-monolith]] found the classical composed column costing $0.93$–$1.31\times$ the monolith (Tier 48), $1.54\times$ slower at six windows (PoC 1a), and $3.98\times$ faster at twelve on one box. None of those runs controlled for **where** a decomposed column can save time. There are three places:

1. **Algorithm and cache.** The exposed column (the arrangement `L2/R10` prescribes) does no pressure solve inside its windows. The composition layer projects the assembled field once per macro-step, and 128×128 windows are small.
2. **Parallelism.** Windows are independent within a macro-step, and a monolith has nothing to split without being cut.
3. **Local time stepping.** The window solver chooses its sub-step count from its own fastest velocity, advection dominating ($\lceil \Delta t / (0.4\,h/u_{\max}) \rceil \approx 21$–$24$ against the viscous limit's $5$). So a window in freestream needs fewer sub-steps than one in a wake.

---

## 2. The arms

| arm | what runs | its answer |
|---|---|---|
| `F` | `RectangularNS` on the undivided domain: the same discretization, one process, as built | the referent |
| `Fw8` | the same, with `scipy.fft` given 8 workers | the monolith's one threadable global operation |
| `E` | the composed column, elliptic part exposed, windows stepped as one serial batch, one global projection per macro-step | bit for bit Tier 48's `Maps.E` (control) |
| `Ep<T>` | `E` with its windows split into $T$ chunks on $T$ threads, every chunk forced to the batch's own sub-step count | **bit for bit `E`** |
| `Elts8` | each window its own sub-step count, on 8 threads | a different scheme; accuracy measured |

**How the bitwise arm is made bitwise.** `WindowNS.step_batch` reads its sub-step count from `b.amax(b.hypot(u, v))` of the batch it is handed. That is the class's only call to `amax`, so each chunk is given a backend proxy whose `amax` returns the whole batch's maximum. Nothing in the build repo is edited.

The rungs are CS-7's tilings, extended by its own rotor rule:

| windows | tiling | domain (cells) | cells | rotors |
|---|---|---|---|---|
| $2$ | $2\times1$ | $128\times240$ | $30\,720$ | $1$ |
| $6$ | $3\times2$ | $240\times352$ | $84\,480$ | $3$ |
| $12$ | $4\times3$ | $352\times464$ | $163\,328$ | $5$ |
| $24$ | $6\times4$ | $464\times688$ | $319\,232$ | $12$ |
| $48$ | $8\times6$ | $688\times912$ | $627\,456$ | $21$ |
| $80$ | $10\times8$ | $912\times1136$ | $1\,036\,032$ | $36$ |

---

## 3. Accuracy: the price of the cut

Each arm marched 40 macro-steps from the freestream, beside the monolith. Farm power is the mean of the last 5 steps. The 80-window rung was timed only.

| rotors | farm power, monolith | composed column vs monolith | rms velocity difference | local time stepping vs monolith |
|---|---|---|---|---|
| $1$ | $0.6223$ | $-4.3\%$ | $0.036$ | $-4.3\%$ |
| $3$ | $1.358$ | $-4.1\%$ | $0.047$ | $-4.1\%$ |
| $5$ | $2.061$ | $-2.3\%$ | $0.045$ | $-2.3\%$ |
| $12$ | $5.168$ | $-6.8\%$ | $0.051$ | $-6.8\%$ |
| $21$ | $9.016$ | $-8.3\%$ | $0.050$ | $-8.3\%$ |

The cut costs a few percent of farm power, growing slowly with the farm. That is the same sub-linear growth [[case-study-scaling-ladder-atlas-0.1]] measured. Local time stepping changes farm power in the fifth digit.

---

## 4. Speed: the first pass

Cost per macro-step over the monolith's, on the state the monolith reached at step 40: the minimum over 3 repeats of the mean of 2 calls, interleaved.

| rotors | monolith, s/step | µs per cell | `E` (serial) | `Ep2` | `Ep4` | `Ep8` | `Ep14` | `Elts8` | `Fw8` |
|---|---|---|---|---|---|---|---|---|---|
| $1$ | $0.041$ | $1.35$ | $0.949$ | $1.587$ | $1.577$ | $1.601$ | $1.569$ | $1.613$ | $1.039$ |
| $3$ | $0.181$ | $2.15$ | $1.079$ | $0.778$ | $1.309$ | $1.551$ | $1.498$ | $1.324$ | $0.977$ |
| $5$ | $1.436$ | $\mathbf{8.79}$ | $1.028$ | $0.198$ | $\mathbf{0.186}$ | $0.295$ | $0.398$ | $0.344$ | $0.966$ |
| $12$ | $2.740$ | $8.58$ | $1.071$ | $0.681$ | $\mathbf{0.176}$ | $0.221$ | $0.324$ | $0.394$ | $0.955$ |
| $21$ | $5.492$ | $8.75$ | $1.124$ | $0.691$ | $0.481$ | $\mathbf{0.230}$ | $0.281$ | $0.422$ | — |
| $36$ | $8.994$ | $8.68$ | $1.248$ | $0.769$ | $0.523$ | $0.508$ | $\mathbf{0.267}$ | $0.446$ | — |

**Read the µs-per-cell column first.** It is the whole mechanism. The monolith's cost per cell is flat at about $8.7$ µs from 5 rotors up. Below that its working set fits in cache and it is four to six times cheaper per cell. The serial column (`E`) tracks it within 3–25%: it is a single batched array of the same size.

**The replication** is §5.

---

## 5. The replication

The first pass's thread-count column was non-monotone: at 12 rotors two threads read $0.681$ and four $0.176$. That is the signature of a timing taken once. A fresh process re-timed the arms that carry the claim on the same saved states: 5 repeats of 3 calls, minimum and median.

| rotors | serial `E`: draw 1 / draw 2 | `Ep4`: draw 1 / draw 2 (median) | `Ep8`: draw 1 / draw 2 (median) | best arm, the slower draw | **speedup range** |
|---|---|---|---|---|---|
| $5$ | $1.028$ / $1.052$ | $0.186$ / $0.286$ ($0.303$) | $0.295$ / $0.478$ ($0.467$) | $0.286$ | $\mathbf{3.5}$–$\mathbf{5.4\times}$ |
| $12$ | $1.071$ / $1.175$ | $0.176$ / $0.247$ ($0.228$) | $0.221$ / $0.310$ ($0.283$) | $0.247$ | $\mathbf{4.0}$–$\mathbf{5.7\times}$ |
| $21$ | $1.124$ / $1.128$ | $0.481$ / $0.479$ ($0.461$) | $0.230$ / $0.213$ ($0.213$) | $0.230$ | $\mathbf{4.3}$–$\mathbf{4.7\times}$ |
| $36$ | $1.248$ / $1.103$ | $0.523$ / $0.480$ ($0.480$) | $0.508$ / $0.467$ ($0.458$) | $0.467$ (8 threads) | $\mathbf{2.1}$–$\mathbf{3.7\times}$ |

**The two draws disagree by up to 50% on one arm, and agree on everything the claim needs.**
- The serial column is never faster, in either draw.
- The threaded column is faster than the monolith from 5 rotors up, in both draws, by at least $3.4\times$ through 21 rotors ($3.499\times$ at 5 rotors in the slower draw, which the table rounds to $3.5$).
- At 36 rotors, 8 threads give $2.0$–$2.1\times$ in both draws. The $3.7\times$ at 14 threads is **one draw**; 14 threads were not re-timed.

**Quote the range, not the best cell.** The first pass's minima were the optimistic draw at 5 and 12 rotors.

---

## 6. The registered predictions

| | prediction | got | |
|---|---|---|---|
| B1 | the threaded column is bit for bit the serial one at every rung and thread count; the serial one is bit for bit `Maps.E` at six windows | 24 of 24; yes | held |
| B2 | the serial column is faster than the monolith at $N \ge 24$, and its ratio falls with $N$ | $1.071$, $1.124$, $1.248$, rising | **FAILED** |
| B3 | 8 threads at least $3\times$ faster at $N \ge 24$ | $4.53$, $4.35$, $1.97$ | **FAILED** at 36 rotors |
| B4 | local time stepping saves at least 10% of sub-steps at $N \ge 24$ | $5.4\%$, $6.8\%$ | **FAILED** |
| B5 | farm power within 25%; local time stepping moves it under 1% | $2.3$–$8.3\%$; fifth digit | held |
| B6 | the 8-thread speedup rises monotonically with rotor count | $0.62, 0.64, 3.39, 4.53, 4.35, 1.97$ | **FAILED** |

**Each failure is a piece of the mechanism.**
- **B2:** serial decomposition cannot escape a memory cliff it steps into as one array.
- **B4:** a wind farm's windows all see roughly the freestream's velocity, so their sub-step counts differ by a few percent.
- **B3 and B6:** a fixed thread count stops fitting the cache as the farm grows. The per-rung best (§4, bold) stays at $3.7$–$5.7\times$.

---

## 7. What this does and does not establish

**Established, on this machine:**
- From 5 to 21 rotors, a classical decomposed wind farm runs $3.5$–$5.7\times$ faster than the undivided solver of the same discretization, across two timing draws, with the thread count chosen per farm size. At 36 rotors it runs $2.1\times$ faster at 8 threads in both draws. It gets the same answer as the serial decomposition, bit for bit, and farm power within $2.3$–$8.3\%$ of the undivided solver's.
- The mechanism is measured, not inferred: the cache cliff in µs per cell, and the bitwise equality that rules out any difference in the arithmetic.

**Not established:**
- **Another machine.** The cliff's position is a property of this laptop's caches, and the vault's own record (0.43× to 3.98× on the PoC 1a demo across machines) says the ratio moves with hardware. A many-core box should keep scaling past 36 rotors, but that is an inference.
- **A parallel monolith.** A monolith can be parallelised too: a threaded stencil library, or MPI. **[AI Inference]:** the standard way to parallelise a stencil code *is* to decompose its domain, so this comparison is "decomposition used for parallelism" against "no decomposition", the textbook reason domain decomposition exists. It is not a new effect, and the proposal should say so.
- **Multiphysics.** One family, one geometry. The rocket, whose blocks differ in physics, has no monolith to compare with ([[outcome-c1-decomposition-vs-monolith]] §2.3).
- **Processes rather than threads.** Python threads contend for the interpreter lock, which is why threading costs 1.6× below 5 rotors. A process pool or a compiled kernel would move the crossover. Not tested.

---

## See Also

- [[outcome-c1-decomposition-vs-monolith]] — the claim this measures, rewritten on it
- [[case-study-scaling-ladder-atlas-0.1]] — the rungs, the rotor rule, and the classical column's accuracy against the monolith
- [[learned-contribution-kill-tests]] §2 — Tier 48's cost ledger, which this reproduces in kind at 2, 6 and 12 windows (E/F $0.932$, $1.31$, $1.02$ there; $0.949$, $1.079$, $1.028$ here)
- [[atlas-proof-of-concept-1]] §11 — the $3.98\times$ at twelve windows on this same laptop, where this page's serial column reads $1.03\times$. **[AI Inference]:** the demo's composed column runs in torch, whose element-wise kernels use several threads, so its figure is consistent with parallelism being the mechanism. It was not re-measured here, and it still has no worklist row
