# Compute plan — what the classical stack costs and where it runs

**Status:** written 2026-09-17, companion to `ATLAS-CLASSICAL-PLAN.md`.
**Prices below are orders of magnitude and must be re-checked before anything is
rented.** Cloud pricing moves, and this document is not a quote.

---

# 0. The one-line answer

**Classical CFD is CPU-, RAM- and interconnect-bound. It is not GPU work.**
vast.ai is a GPU marketplace, so it is the **wrong tool for P0–P5** and the
**right tool for the neural phase that follows**. Keep the key; do not use it
for this.

---

# 1. The development machine

| | |
|---|---|
| CPU | Intel Core Ultra 7 155H — 16 cores, 22 threads, 3.8 GHz max |
| RAM | **15.4 GB** |
| GPU | Intel Arc integrated, 2 GB — **not a compute device** for this work |

**What it has already been measured doing**, on the body-fitted column:

| quantity | value |
|---|---|
| composite unknowns | $204{,}599$ |
| grids | 10, cut by each other and by the road |
| march | median $\mathbf{1.4}$ s a step |
| SuperLU factorisation | $3.8$ s, once per geometry |
| SuperLU solve | $0.103$ s, once per macro-step |
| full test suite | 1774 tests, $481$ s |

That is a capable laptop for two-dimensional work and it is **not** the
bottleneck for P0 through P3.

**Three standing hazards, from this project's own record.** Each has already
cost a session:

- **Battery throttling.** On battery this machine runs at $1.4$ GHz —
  about $4.4\times$ slower. Check `BatteryStatus` before quoting any timing.
- **Standby suspends a suite.** A four-hour mid-suite sleep has happened.
  Check power events (Id 105, 506/507) before trusting an elapsed time.
- **Wall-clock numbers rot.** A ms/step figure in the docs was wrong by $4\times$
  hours later. **Make code measure itself and quote ratios**, not absolutes.

---

# 2. What each phase actually needs

| Phase | CPU | RAM | GPU | Runs where |
|---|---|---|---|---|
| **P0** close PoC 3 | light | < 8 GB | none | **laptop** |
| **P1** CAD ingest | light | < 8 GB | none | **laptop** |
| **P2** external expert | moderate — probes are $\dim M$ solver calls | 8–16 GB on a small case | none | **laptop** |
| **P3** preCICE backend | light | < 8 GB | none | **laptop** |
| **P4** 3-D on real CAD | **heavy, many cores** | **64–256 GB** | none | **rented** |
| **P5** refusal demo | mixed | mixed | none | rented + laptop |
| *(later)* neural experts | — | — | **yes** | **vast.ai** |

**Four of the six phases run on the machine you already have.** Only P4 forces a
rental, and P5 partly inherits it.

---

# 3. The RAM arithmetic, which is the real wall

This is the number that decides everything, so it is written out rather than
asserted.

The current two-dimensional car is $204{,}599$ unknowns. A credible
three-dimensional RANS mesh for a car is **5–20 million cells** for a
demonstration; real F1 teams run well past 100 million.

A common rule of thumb for steady RANS is roughly **1–2 GB per million cells**,
with coupled or transient schemes higher. So:

| mesh | rough RAM | fits in 15.4 GB? |
|---|---|---|
| 2 M cells (crude) | 2–4 GB | yes, barely, alone |
| **10 M cells (credible demo)** | **10–20 GB** | **no** |
| 20 M cells | 20–40 GB | no |
| 100 M cells (team-scale) | 100–200 GB+ | no |

And that is the *fluid alone*, before the structural or thermal agent on the
other side of the seam, before preCICE's own mesh and mapping storage, and
before any probe that holds several states at once.

**Conclusion: P4 is not slow on this machine, it is impossible.** RAM is the
wall. Do not attempt it locally first and then "move it when it gets slow" — size
the box up front.

> **Verify rather than trust this table.** Run one small case on the adopted
> solver, measure peak RSS per cell, and extrapolate from *your* numbers. The
> rule of thumb above is a planning figure, not a measurement, and this project's
> own discipline is that an unmeasured constant refuses to behave like a number.

---

# 4. Why not vast.ai for the classical phases

Not a criticism of vast.ai — a mismatch of market.

1. **It is a GPU marketplace.** The pricing, the supply and the tooling are all
   built around GPU rental. CPU-only instances exist but are not the product.
2. **RANS solvers do not meaningfully use a GPU.** OpenFOAM's GPU support is
   partial and SU2's is limited. Renting an A100 to run RANS means paying for
   silicon that sits idle while 16 CPU cores do the work.
3. **Consumer hardware, variable reliability, no fast interconnect.** Fine for a
   training run that checkpoints. Poor for a multi-node MPI solve where one slow
   link sets the pace for every rank.

**Keep it for what it is good at.** GPU rental is already pre-authorised for this
project (A100 included), with the standing rules: use `--raw` on Windows, and
**always destroy the instance when done.** That is the right tool for training
neural experts after the raise.

---

# 5. What to use for P4 instead

| option | what it is | best for | rough cost | caveat |
|---|---|---|---|---|
| **Hetzner dedicated** (AX/EPYC line, 48+ cores, 256 GB) | a real bare-metal box, monthly | **single-node CFD — best value by a wide margin** | low hundreds €/month | monthly commitment; setup is yours; no fast interconnect between boxes |
| **AWS `c7a` / `c7i`, spot** | general compute, per-hour | bursty runs, first experiments | moderate/hour, much less on spot | spot can be reclaimed — checkpoint |
| **AWS `hpc7a` / Azure `HBv4` / GCP `H3`** | purpose-built HPC with EFA or InfiniBand | **multi-node MPI**, if one box cannot hold the mesh | expensive | only worth it when a single node genuinely cannot fit |
| **Oracle Cloud bare metal** | cheap bare metal, often overlooked | single-node, price-sensitive | often the cheapest bare metal | smaller ecosystem |

**Recommendation: one big single-node CPU box, not a cluster.**

A single machine with 48–64 cores and 256 GB holds a 10–20 M cell case
comfortably, and **a single node needs no interconnect at all** — which removes
the hardest and most expensive variable from the problem. Go multi-node only when
a measurement, not a guess, shows one node cannot hold the mesh.

Of the four, **Hetzner is the one I would start with** if the work is steady over
a month; **AWS spot** if it is bursty and you would rather pay by the hour than
commit. Azure HBv4 is the technically best option for multi-node CFD and is
overkill until P4's gate has already passed once on a single node.

---

# 6. The recommendation, in order

1. **P0–P3 on the laptop.** Four phases, several weeks of work, no rental. Do
   not spend money before P2's gate, because P2 can end the plan.
2. **P4: rent one big-RAM CPU box.** Size it from a measured RSS-per-cell on a
   small case, not from §3's rule of thumb. Single node first.
3. **Neural phase: vast.ai**, as already set up and authorised.

**So the honest answer to "is there a better site": yes, for this work — but it
is not another GPU marketplace.** It is a CPU box with a lot of RAM, and for that
the cheapest credible option is a dedicated server rather than a cloud GPU host.
Keep vast.ai for the phase it was approved for.

---

# 7. Operating rules for any rented machine

From this project's own record, each one written after it cost something:

- **Persist state on every step, especially on failure.** Serialise the whole
  trajectory, not the endpoints — writing only the endpoints after every step
  keeps no history at all.
- **Never pipe a long run to `tail`.** Redirect to a file; a pipe hides progress
  until exit.
- **Warn before any multi-hour run**, and make the cost visible before it starts.
- **Destroy the instance when done.** Every time.
- **Check for stray processes before trusting a timing** — `ps -W | grep` misses
  detached Windows processes; use `tasklist`. A `python server.py` from
  `C:\Users\Nauni\phone-remote\start.bat` is the user's own and should be left
  alone, but noted beside any timing taken here.
- **Quote ratios, not wall-clock.** See §1.
