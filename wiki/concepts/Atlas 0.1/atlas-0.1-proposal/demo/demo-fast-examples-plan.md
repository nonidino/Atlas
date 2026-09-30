# One fast example per simulation type — where decomposition can honestly win, and by how much

**Type:** Concept page — **plan and feasibility analysis, with a registration protocol** (folder: `Atlas 0.1/atlas-0.1-proposal/demo/`)
**Status:** written 2026-09-30. **No run was made for this page**, and it quotes no new timing. Every ceiling below is arithmetic from a stated model, labelled **[AI Inference]**, and is there to decide what to try first, not to be quoted. Every measured number is from a named page.
**Hub:** [[00-proposal-workstreams]] · **Siblings:** [[demo-finish-plan]] · [[demo-learned-case-plan]]
**Built on:** [[showcase-gallery]] §3 · [[decomposition-speed-by-rotor-count]] · [[outcome-c1-decomposition-vs-monolith]] · [[atlas-and-standard-dd-theory]]

---

## 0. The request, and the fact it runs into

> *"For each type of case, have one geometry+material+boundary+domain case where the decomposition is highly accurate, but much faster. The user should be able to press a button in the header that chooses that specific example for them, so that they can simulate it without choosing any settings themselves."*

**The fact:** at showcase sizes, the undivided solve beats every spatially decomposed arm in seven of the eight families, by $1.3\times$ to about $490\times$. The wind farm is the exception: $4.83\times$ and $4.55\times$ faster at 12 and 21 rotors on threads ([[showcase-gallery]] §3). The gallery's own reading is that below about 30,000 cells, the per-window bookkeeping and the iteration count cost more than the smaller solves save.

**So this item is not a matter of choosing good examples from what exists. For most families it needs a mechanism the workbench does not have yet.** Decomposition is not faster by being cut into pieces; it is faster only when the cut buys something the undivided solve cannot have. §2 names the four things it can buy, and §3 assigns one to each family, with the most it could give.

---

## 1. What "highly accurate" and "much faster" will mean

Both are **registered in each family's module before its first timed run** and never loosened, as every workbench check has been.

**Much faster.** The speed ratio the workbench already reports,

$$s=\frac{t_{\text{full}}}{t_{\text{arm}}},$$

each $t$ the mean time of the family's `step` alone. It is measured on the owner's laptop on AC power, with the machine's state in the record (`machine.py`). **The registered bar is $s\ge3$ on the fastest decomposed arm.** The full-domain arm stays *the same discretization on the undivided domain*, as everywhere in the workbench.

**One honesty rule added for the demo:** where a *different* classical solver would beat the full-domain arm on this machine (a better direct solver ordering, an implicit step, multigrid), the result card says so in one line. **[AI Inference]:** a sceptical engineer will ask exactly this, and the website ([[proposal-website-plan]]) inherits every card.

**Highly accurate.** Per family, against the full domain:
- where the decomposed arm is the same algebra solved another way (styles B, C, D; the substructuring of §2.3), round-off: **$10^{-9}$ of the field's scale**, tighter than the family's existing $10^{-6}$;
- where the scheme itself differs (multirate time steps, §2.2), a registered tolerance **stated before the run**, proposed at $10^{-3}$ of the scale, with the conservation check kept at round-off;
- the wind farm: farm power within **3%** of the full domain, where the plain farms measured 2.3–8.3% ([[decomposition-speed-by-rotor-count]] §3).

**Selection is not measurement.** A sweep finds each family's configuration. **A confirmation run on a fresh start then produces the number the button shows.** Tier 50 found that Tier 48's most-quoted number had been the best of twelve draws ([[corrupted-checkpoint-and-jacobian-fidelity]]).

**The two-minute rule stands**: each fast example opens, runs all its arms and compares in under two minutes, the slow full-domain arm included. This caps how large a case can be, and so how much the size-driven mechanisms can give.

---

## 2. The four ways a decomposition can be genuinely faster

### 2.1 P — parallel pieces that fit in cache

**The mechanism measured in this vault.** Windows are independent within a macro-step, so $T$ threads run them at once. And a window of $128^2$ cells stays in cache where the undivided farm does not: the monolith's cost per cell rose about fourfold between 85,000 and 163,000 cells ([[decomposition-speed-by-rotor-count]] §0).

$$s_P\;\le\;T\cdot\frac{c_{\text{full}}(N)}{c_{\text{win}}(N/p)}\cdot\frac{1}{1+\text{exchange share}},$$

with $c(\cdot)$ the cost per cell at that size. **It needs work large enough to split and a kernel that releases Python's GIL.** numpy's array kernels do; whether scipy's sparse LU does must be measured before anything is built on it (§4, step 0).

### 2.2 M — multirate: each piece at its own stable time step

An explicit scheme's step is set by the most restrictive cell in the domain. The full domain takes that step **everywhere**; a decomposition can take it only where it is needed and **sub-cycle** there. With piece $i$ holding a fraction $n_i$ of the cells and allowed step $\Delta t_i$:

$$s_M\;\le\;\frac{1}{\sum_i n_i\,\Delta t_{\min}/\Delta t_i}.$$

**[AI Inference]**, from the families' own material data (`registry.py`):
- **sound:** water's speed is $1480/343=4.31$ times air's, so an air piece may step $4.31\times$ longer. With 90% of the cells in air, $s_M\le1/(0.1+0.9/4.31)=3.2$;
- **heat:** copper's diffusivity $400/(8960\cdot385)=1.16\times10^{-4}$ is $9.9$ times steel's $45/(7850\cdot490)=1.17\times10^{-5}\ \mathrm{m^2/s}$, and the explicit step scales as $\Delta x^2/\alpha$. With a copper spreader on 10% of a steel plate, $s_M\le1/(0.1+0.9/9.9)=5.2$.

Threads multiply either. **The accuracy cost is real and must be measured:** the interface is interpolated in time, so the arm no longer equals the full domain to the bit. The literature has schemes that keep what matters exactly. Diaz & Grote's local time stepping for the wave equation conserves a discrete energy (*SIAM J. Sci. Comput.* 31, 2009). Berger–Colella refluxing keeps a conservation law exact across sub-cycled patches (*J. Comput. Phys.* 82, 1989). **The balance check stays at round-off; only the agreement tolerance loosens, to its registered value.**

### 2.3 S — substructuring: independent factorizations, one interface solve

For a steady linear problem, eliminate each piece's interior and solve the interface directly. This is the `direct-schur` accelerator the compiler already derives for the refused examples ([[gap-worklist]] W348):

$$\begin{pmatrix}A_{II}&A_{I\Gamma}\\A_{\Gamma I}&A_{\Gamma\Gamma}\end{pmatrix}\begin{pmatrix}u_I\\u_\Gamma\end{pmatrix}=\begin{pmatrix}f_I\\f_\Gamma\end{pmatrix},\qquad S\,u_\Gamma=f_\Gamma-A_{\Gamma I}A_{II}^{-1}f_I,\quad S=A_{\Gamma\Gamma}-A_{\Gamma I}A_{II}^{-1}A_{I\Gamma}.$$

$A_{II}$ is block-diagonal, one block per piece, so the pieces factor **in parallel**, and the answer is the undivided solution **to round-off**. There is no iteration and no 616 Schwarz sweeps.

**The honest ceiling.** A good sparse direct solver on the whole domain already does this recursively: that is nested dissection. On a 2-D grid it costs $O(N^{3/2})$ to factor, and $p$ pieces cost $p\,(N/p)^{3/2}=N^{3/2}/\sqrt p$ plus the interface. **[AI Inference]:** the gain over a nested-dissection solve of the whole is mostly the $T$ threads, not the algorithm. Against scipy's default (SuperLU with COLAMD, one thread), the margin will look larger than it would against a nested-dissection Cholesky. That is exactly the case for §1's one-line card note. The crossover size must be measured, and it may exceed what the two-minute rule allows.

### 2.4 X — split by physics

Two physics on two threads can at most halve the time: $s_X\le$ the number of physics, and less when their costs differ. The drawn arc's lagged split measured $1.41\times$ once, in one run of 60 steps, not investigated ([[showcase-gallery]] §3).

---

## 3. The eight families

| type | mechanism | proposed fast example | ceiling, **[AI Inference]** unless measured | new code | risk |
|---|---|---|---|---|---|
| **wind farm** | P | **the 5-rotor rung**: $4\times3$ windows, 163,328 cells, on threads as W346 ran it | **measured** by W346: $3.5$–$5.7\times$ from 5 to 21 rotors, farm power $-2.3\%$ at 5 rotors. So it probably meets both bars as it is | an example entry, and the 3% check | **low** |
| **river plume** | P first, then M | a long river at $\ge 300{,}000$ cells on 8–12 threads; if that fails, a fast shallow reach feeding a slow deep one, sub-cycled | P: unknown, since `plume-2` lost at 28,800 cells ($0.307$ threaded). M: set by the two reaches' explicit limits | P: none. M: sub-cycling with refluxing | medium |
| **sound** | M | a large air room ($\ge90\%$ of the cells) around a small water tank | $3.2\times$ from the speeds, times threads if the air is cut too | local time stepping at the interface (Diaz–Grote), with its conserved energy | medium |
| **heat conduction** | M, explicit transient | a steel plate with a copper heat spreader on 10% of its area, heated at one end | $5.2\times$ from the diffusivities. **The card note of §1 applies**: an implicit step with a reused factorization may beat both arms | an explicit conduction step (the family is implicit today), sub-cycling | medium |
| **current in a plate** | S | a large film cut into $2\times2$ pieces on the same circuit | at most the thread count; the circuit adds nothing to the cost | style D over a **decomposed** plate (today the plate is one piece), with a Schur interface | medium–high |
| **loaded structure** | S | the plate with a hole at $\ge10^5$ elements, $2\times2$ pieces | at most the thread count, if the pieces' factorizations run in parallel | substructuring on `fe.py`'s element masks | medium–high |
| **heated structure** | X | the heated strip at a size where the two physics cost about the same | **$\le2\times$ by construction** | none | low, and a low ceiling |
| **cooled block** | S, or X on the two sides | the cooled winding at a size where both sides cost about the same, the interface solved directly | **$\le2\times$ without a spatial cut too** | a direct Schur interface for the physics seam | medium, and a low ceiling |

**What this adds up to, stated plainly.** Four families have a mechanism that can plausibly clear $s\ge3$: the wind farm (already measured), the plume, the sound and transient conduction. Two can reach it only if threaded substructuring pays at a size the two-minute rule allows. **Two cannot reach it by construction**: the heated structure, and the cooled block without a spatial cut too.

**Decided (O3, the owner, 2026-09-30):** for a family whose ceiling is below the bar, the Fast example loads its **fastest honest setup**, and the result card **names what limits it**, for example *"a split by physics can at most halve the time; this one takes 0.61 of it"*. **And the demo says the larger truth plainly:** at the end of the day these are classical solvers, and decomposition becomes much more beneficial as domains get more complicated and three-dimensional, and with neural operators as the experts. §5 gives the line every card carries. The website carries the same passage ([[website-outline]] section 3).

---

## 4. Build order

**Step 0 — the micro-benchmarks, before any new scheme.** On the owner's laptop, on AC power, each registered first:
1. numpy stencil kernels on $T=1,2,4,8,12$ threads over independent arrays: does P scale?
2. scipy `splu` factor and solve on $T$ threads over independent matrices: **does SuperLU release the GIL?** If not, S needs a process pool with shared memory, and its overhead must be timed too.
3. The undivided solve's cost per cell against size, for each family's core (`fv.py`, `fe.py`, the leapfrog), to find where each leaves cache.

These take minutes and decide which rows of §3 are worth building.

**Step 1 — the wind farm** (no new scheme): the 5-rotor example, its 3% check, the confirmation run.

**Step 2 — P on the plume.** If it fails, M on the plume.

**Step 3 — M on sound and on conduction.** One sub-cycling driver in `styles.py`, shared: each piece's step, the interface's time interpolation, and a flux ledger for refluxing. **N = 1 control:** equal steps everywhere must give the full domain bit for bit.

**Step 4 — S on the structure and the plate**, and on the cooled block's seam. It needs W348 settled first ([[demo-finish-plan]] §5), or every one of these compiles to a red `refuse`.

**Step 5 — the heated structure:** its best honest split.

**Step 6 — the header button** (`Fast example`, per type; [[demo-finish-plan]] §1.3 D6), the result card's one line naming the mechanism, and the gallery rows. [[showcase-gallery]] gets a section for the fast examples, with the records, the machine state and the confirmation runs.

---

## 5. What the button shows

The Fast example loads a complete case: geometry, materials, boundaries, windows, time and grid. The visitor presses Run. The **speed card** comes first, with one line naming the mechanism, so the result teaches as well as impresses:

| mechanism | the card's line |
|---|---|
| P | *"Faster because the 12 pieces run at once on 12 cores, and each fits in the processor's cache."* |
| M | *"Faster because the air steps 4.3 times longer than the water: each piece takes the largest step that is stable for it."* |
| S | *"Faster because the 4 pieces are factored at once; the answer is the undivided one to round-off."* |
| X | *"A split by physics can at most halve the time; this one takes 0.61 of it."* |

The numbers in these lines are placeholders for the format, not predictions.

**Every speed card also carries one fixed line (O3)**, below the mechanism's:

> *These are classical solvers, cut into pieces. The gain grows as the shape gets more complicated, in three dimensions, and when the pieces are learned experts, which is what Atlas is built for.*

**[AI Inference], why the claim is fair and how it is bounded:**
- In three dimensions a direct solve's factorization grows as $O(N^2)$ against two dimensions' $O(N^{3/2})$, and an explicit step's working set leaves cache sooner, so substructuring and parallel pieces have more to win.
- A complicated shape has more contrast between its pieces, for local time steps to exploit.
- Learned experts are the proposal's own claim ([[chart-operator-architecture]]).

The line says *grows*, not a number, because no three-dimensional ratio has been measured in the workbench.

---

## See Also

- [[demo-finish-plan]] — items 1.1–1.3, hosting, and why W348 comes first
- [[demo-learned-case-plan]] — item 1.5
- [[decomposition-speed-by-rotor-count]] — the one mechanism measured so far
- [[showcase-gallery]] — every example's measured ratio
- [[atlas-and-standard-dd-theory]] — the classical home of every scheme named here
- [[website-evidence-and-citations]] — where these results will be shown
