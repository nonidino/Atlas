# PoC 3 — the certified mode on the screen, and what it costs over a march

**Tier 73, 2026-09-16.** Section 12's criterion 4 — *they can flip it again to
certified and watch the error go to the classical answer, and see what that
cost* — is **met** on the body-fitted column. Tier 72 built the mechanism and
priced one step; this puts it on the dashboard and measures the two things one
step cannot answer.

Related: [[poc3-racelab-certified-step]], [[poc3-racelab-dashboard]],
[[poc3-racelab-bodyfitted-demo]], [[poc3-racelab-knobs-wired]],
[[defect-correction-learned-operator]], [[gap-worklist]].

---

## 1. What is on the screen

`atlas/demo_racelab/static/bodyfitted.html`, port 8014, right-hand panel:

- **A three-way switch** — `classical`, `certified`, and `learned` drawn
  **refused with its reason**, never hidden (§4.3's rule, applied to a mode
  instead of a family).
- **§5.2's certified row**: outer iterations, inner cheap calls, the residual,
  the system residual, the error to the classical answer, which cheap map ran,
  and the status — beside the classical solve's own BiCGSTAB iteration count, so
  the two are read against each other rather than in isolation.
- **A `verify` toggle**, off by default, and **two separate prices**.
- **The reason the mode is not per window** (W295), in full, under the panel.

### The two prices, and why they are two

**The residual is free and the error is not.** The residual is computed every
outer iteration whatever else is on. *Measuring* the distance to the classical
answer needs the classical solve run beside the certified one — so `verify` is a
toggle, and a verified certified step is **its own timing arm**:

| arm | median | n | range |
|---|---|---|---|
| classical | $0.684$ s | $20$ | $0.561$–$0.786$ |
| certified | $0.826$ s | $20$ | $0.623$–$1.035$ |
| certified + verify | $1.253$ s | $20$ | $1.031$–$1.621$ |

**cost of the mode $\mathbf{1.209\times}$, cost with verify $\mathbf{1.833\times}$.**

Quoting only the compared arm while verify is on would price the *instrument*
and call it the mode — the same confound Tier 72 found inside its own
`certified_step_report`. Every number comes with its replicate count and its
range, and the ratio is **withheld entirely** below five steps an arm (§5.4's
rule 5: never a single-draw number as an effect size).

§5.4's rule 4 is what keeps the mode honest with `verify` off: the certified
limit is the classical answer by a **proof**, and the residual is what bounds
the distance to it.

---

## 2. What a single step could not answer

### The certified march tracks the classical one

Two columns built and released from the **same** settled state at $t = 12$, one
classical and one certified, marched $24$ steps side by side — each arm re-spun
rather than continued, because continuing one march into the other measures the
continuation.

$$\text{gap} = 2.467\times10^{-8},\qquad
  \text{relative gap} = \mathbf{2.849\times10^{-8}}$$

**With its positive control**, which is the half that makes it a result: over the
same $24$ steps the field itself **moved** by $8.522\times10^{-3}$, a relative
$9.841\times10^{-3}$. So $\text{gap}/\text{move} = 2.895\times10^{-6}$ — the two
marches agree five and a half orders of magnitude tighter than the physics
changed. A gap that is small only because nothing happened is not a result, and
this vault has published one of those before.

### Every step converges, and the certificate is never violated

Over $24$ certified steps on the live column: **all converged**, outer
iterations $2$–$3$, residual at most $1.038\times10^{-8}$. With `verify` on, over
another $24$: **all converged**, measured error $2.976\times10^{-9}$ to
$9.727\times10^{-9}$, and

$$\max_k \frac{\lVert w_k - w^{\star}_k\rVert}{r_k} = \mathbf{1.0000}$$

so $\Theta \approx 1$ on every step and the certificate is tight, not merely
satisfied. **Q1 and Q2 held.**

### The classical mode is still the classical mode

With the switch at `classical` the flow's `momentum_solver` is `None` — the seam
is empty, not merely inert — and with `verify` off **no step reported an error**
($0$ of $24$), because nothing computed one. **Q6 held.** If flipping back left
anything installed, every classical number this demo reports would be about a
different solver than the one it names.

---

## 3. Q5 failed, and the probe was three steps against a four-hundred-step clock

**Q5 as registered:** *criterion 2 survives criterion 4 — moving the ambient
temperature knob while the certified mode is ON still moves the marching coolant
return.* Judged **False**, and it stays judged False.

What actually happened: `vehicle_march.N_FLUID_PER_COOLANT` is $\mathbf{400}$.
The coolant circuit sub-cycles once every four hundred fluid steps, and the
probe took **three**. The loop never ran, the return temperature was `None`, and
nothing could have moved it — **in either mode**.

Three separate faults, and only one of them is about the coolant:

1. **The horizon.** A probe $1/133$ of the cadence it is probing cannot see the
   quantity it names. This is the vault's own *positive controls need a horizon*
   row: march a proposed measurement as far as the thing it measures.
2. **No control.** Q5's `why` argued about the certified mode, and the failure
   had nothing to do with the mode — the identical probe fails identically in
   the classical mode. **The tier registered a measurement with no control**,
   and the control is what would have shown that at once.
3. **`False` where `None` belonged.** The probe's own `moved` flag required both
   readings finite and different — which correctly refuses `nan`, the defect
   Tier 69 built it for — but it collapsed *absent* into *did not move*. Two
   `None` readings are **not measured**; they are not a knob that failed.

The repair is `_moved`, which returns three states and never two, and a
`knobhorizon` stage that runs **both** arms.

**The control, run at last:** the identical three-step probe in the **classical**
mode also reads `None`. The failure is the probe's horizon and not the mode's,
and one run of that control at registration time would have said so.

**The measurement Q5 wanted:** marched past a real sub-cycle boundary, with the
knob moved while the certified mode was on, the marching coolant return goes

$$311.727499\ \text{K} \;\longrightarrow\; \mathbf{331.837236}\ \text{K},
  \qquad \Delta = +20.11\ \text{K}$$

at steps $402$ and $804$. **So criterion 2 does survive criterion 4** — a knob
reaches its subsystem with the certified mode running. Tier 71 measured
$+19.31$ K for the same knob on the classical march, so the two agree to about
four percent, which is the cross-check that the certified mode did not quietly
change the circuit.

Q5's verdict is **left as judged**. What is recorded beside it is that the
prediction was unmeasurable as written, and why.

**[AI Inference]:** the general shape is that a prediction about *X surviving Y*
needs a control in which $Y$ is absent, or a failure of $X$ for reasons of its
own reads as a failure of $Y$. Q5 would have been unfalsifiable in the useful
direction either way: with a 3-step probe it was guaranteed to fail, and nothing
in it could have distinguished the guaranteed failure from a real one.

---

## 4. What the page found that no test would

Both were caught by opening the dashboard and clicking, not by any assertion.

**The panel showed one arm's numbers under another arm's name.** It rendered
`arms[cost.comparing]`, and while the mode was `classical` that *is* the
classical arm — so the row labelled **"certified step"** displayed
$0.516$ s $(n=20)$, the classical arm's own timing, and the cost row read
"20 this arm" counting classical steps. The numbers were real and the labels
were wrong. This is the third instance of that defect class in four tiers —
Tier 71's knob reporting a response the march did not have, Tier 72's
`detuned_psi` answering under its own name when it could not be built, and now
this. Each arm is now rendered under its own key, and a test asserts
`A[cost.comparing]` does not appear in the page at all.

**And the certified sweep does not converge through an impulsive start.** Found
while writing the fixture: `cylinder_flow` starts from rest, the sweep returns
`fallback_unconverged`, and the next step blows up. The real column never
marches from rest either — that is exactly **W293** — so the fixture settles
first, in the classical mode, and the switch is exercised on a flow that is
actually marching.

---

## 5. What this tier did NOT do

- **No learned expert, and none is possible here.** `learned` is refused with
  **both** its blocks named — the uniform $128\times128$ interface against
  $61.0\%$ of the unknowns (W288), and a lead of $0.0250$ against the
  checkpoint's native $0.1$ (W294). Naming only the first would leave a reader
  thinking a better tiling could fix it.
- **The certified mode is not per window** (W295) and the page says so rather
  than faking a split. Section 5.2's per-window certified telemetry is
  unaddressed on this column and told on the porous one.
- **The porous column's dashboard is untouched.**
- **W293 still stands**, so a geometry commit still cannot be spun up; it is
  now written into the requirements' §12.1 as a blocker rather than left in the
  worklist alone.
- **The cost was measured at one point in the incomplete factor's refresh
  cycle**, on a machine at $1.4$ GHz of $3.8$, with the user's two
  `phone-remote` processes running.
- Nothing downloaded or installed, no machine rented, the unlicensed structural
  checkpoint not loaded, nothing pushed.

---

## 6. Reproducing

```
python scripts/tier73_certified_screen.py --out out/racelab22 \
    --stages switch,trajectory,page,knobhorizon,summary
python -m pytest tests/test_tier73_certified_screen.py -p no:cacheprovider
python -m uvicorn atlas.demo_racelab.bodyfitted_server:create_app --factory --port 8014
```

The switch and trajectory stages build the composite three times (about $110$ s
each) and need the settled release state at $t = 12$. Quote the **ratios**, not
the seconds: both halves of each are medians over the same process.
