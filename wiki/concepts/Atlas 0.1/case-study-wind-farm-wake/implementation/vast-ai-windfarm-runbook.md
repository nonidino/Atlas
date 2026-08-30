# Atlas 0.1 Wind Farm — vast.ai Rollout Runbook

**Type:** Operations runbook (folder: Atlas 0.1 / case-study-wind-farm-wake / implementation)
**Status:** Live as of 2026-08-24. The GPU path is verified equal to the numpy reference **on CPU**; `device='cuda'` is verified by §3 on the box itself and nowhere else.
**Code:** `github.com/nonidino/physics-foundation-model`, branch `atlas-0.1-windfarm`. `vastai_windfarm.sh`, `scripts/windfarm_backend_check.py`, `src/atlas/cases/windfarm/backend.py`.
**Related:** [[vast-ai-training-runbook]] (Noether 1.1 — a *different* machine profile, see §1), [[wind-farm-implementation-log]], [[open-problems-atlas-0.1]]

---

## 0. Do not reuse the Noether machine profile

[[vast-ai-training-runbook]] rents for **VRAM and bf16 throughput**, because it trains a 44 M-parameter network and its failure mode is `CUDA out of memory`. Renting the same box for the wind farm would be paying for the wrong resource entirely.

The wind farm runs a **classical solver**. Profiling one macro-step — 124 windows of $128\times128$, `scripts/windfarm_profile_step.py` — gives:

| owner | self time | share |
|---|---|---|
| window solve (`reference.py` stencils) | 173.1 s | **95.2%** |
| numpy ufuncs / BLAS | 7.6 s | 4.2% |
| assembly + thrust fixed point (`couple.py`) | 0.6 s | 0.3% |
| global pressure solve (`pressure.py`) | 0.2 s | **0.1%** |

122 756 Python calls in 182 s, so essentially none of it is interpreter overhead. It is batched elementwise work on a `[124, 128, 128]` **float64** array — 16 MB a pass, roughly 1700 passes per macro-step. Peak live memory is a few hundred MB.

So the profile is: **~2 GB VRAM, float64, memory-bandwidth-bound.** Bandwidth is the thing to buy.

---

## 1. Why a GPU at all — the cheap alternative was checked first

The obvious free fix is the 21 idle cores on the development machine. numpy releases the GIL inside large ufunc loops, so a thread pool over the batch dimension needs no copies and no pickling. Measured on the real kernel shape:

| threads | ms/call | speed-up |
|---|---|---|
| 1 | 81.0 | — |
| 4 | 35.4 | 2.29× |
| **8** | **28.8** | **2.81×** |
| 11 | 35.5 | 2.28× |
| 22 | 35.4 | 2.29× |

**It saturates at 8 threads and then gets worse.** That is the signature of a memory-bandwidth wall, not a core shortage — which settles two things at once. Renting a bigger *CPU* box would buy almost nothing, and a GPU's advantage here is real and quantifiable, because on a 5-point stencil a GPU's edge is its ~1 TB/s against a desktop's tens of GB/s, not its FLOPs.

**[AI Inference]:** the measured CPU rate is ~1.8 GB/s of effective traffic for `_ddx`, far below what the DRAM can do, which points at the per-call `np.empty_like` (a fresh 16 MB allocation, 1700 times a step) rather than at raw bandwidth. If so, part of the GPU's win will come from its caching allocator rather than its memory system, and a caching allocator on CPU would recover some of it for free. **Not measured — do not quote this as a result.**

---

## 2. Renting

Pick for **memory bandwidth per dollar**, with enough fp64 not to be crippled:

| card | bandwidth | fp64 rate | verdict |
|---|---|---|---|
| A100-40 / A100-80 | 1.55 / 2.0 TB/s | 1/2 | the safe pick |
| H100 | 3.35 TB/s | 1/2 | fastest, usually not worth the premium here |
| RTX 4090 | 1.0 TB/s | **1/64** | likely fine — see below — and much cheaper |
| L40S / A6000 | 0.86 / 0.77 TB/s | 1/64, 1/32 | no advantage over a 4090 |
| V100 | 0.9 TB/s | 1/2 | fine if cheap |

**On the consumer cards, and why the table is not the last word.** A crippled fp64 rate sounds disqualifying, but the stencils have an arithmetic intensity of roughly 0.1–0.2 FLOP/byte, so at 1 TB/s they need only ~150 GFLOP/s — well inside even a 4090's 1.3 TFLOP/s fp64. The compute-bound part is the DCT in the pressure solve, and that is ~356 GFLOP per macro-step: 0.27 s on a 4090 against 182 s of stencils. **[AI Inference]** on both counts, which is exactly why `vastai_windfarm.sh` *measures* the fp64 stencil rate on the actual card during setup instead of trusting this paragraph.

**Disk:** ~5 GB. There is no committed corpus — the wind farm generates its own initial condition. Field checkpoints are ~11 MB each.

---

## 3. Bring-up — verify before you spend hours

```bash
git clone -b atlas-0.1-windfarm git@github.com:nonidino/physics-foundation-model.git
cd physics-foundation-model
bash vastai_windfarm.sh
export GH_TOKEN=ghp_...        # 'repo' scope, so run state is mirrored off the box
```

Then, **before any rollout**:

```bash
python -m pytest tests/atlas/windfarm/test_backend.py -q
python scripts/windfarm_backend_check.py --device cuda
```

The pytest file is the same equivalence suite that runs on the development machine (torch-CPU against numpy). The second script is the part that **cannot** run anywhere else: it repeats the comparison on the actual GPU, across all three transmission conditions, and calibrates the tolerance against a physically negligible perturbation — a $10^{-6}$ relative change in $\nu$ — rather than a threshold picked to pass. If the backends ever agreed only as well as two genuinely different solves agree, the check would be meaningless, and this is the one place that is asserted.

It also prints the measured macro-step speed-up, so **"was this worth renting" is answered before the long run rather than after it.**

Exit code 1 means the GPU is not reproducing the reference. Do not run a rollout on that device.

---

## 4. Running

```bash
tmux new -s wf
python scripts/windfarm_schwarz.py --mode dirichlet --force-mode rhs \
    --backend torch --device cuda --t-end 20 \
    --checkpoint-every 5 --publish wf-dirichlet-rhs
```

`Ctrl-b` then `d` detaches; `tmux attach -t wf` returns.

- `--backend torch --device cuda` is the only change from a local run. It is refused on `--mode periodic`, where it would silently do nothing (`SpectralNS` is FFT-bound and is not ported).
- `--publish TAG` mirrors **both** the per-step JSONL and the last good `.npz` to a GitHub Release as the run proceeds. Both halves matter and for different reasons: the `.npz` is what lets a fresh box continue, and the JSONL **is the result** — a run killed after fifteen hours still said something, and that sentence should not live only on a disk that is about to be reclaimed.
- Uploads are best-effort by contract. A failed or stalled upload warns and the run continues; every `gh` call is time-bounded with stdin closed.

**Crash recovery.** Rent a new box, repeat §3, and add `--resume-release wf-dirichlet-rhs`. At most one checkpoint interval of progress is ever at risk.

---

## 5. Two safeguards that failed on 2026-08-24, and the rules they cost

Both are already fixed in the code. They are here because each was invisible while it was happening, which is the property that makes them worth writing down rather than just patching.

**The checkpoint directory was keyed on the config tag, not the output path.** Two runs differing only in `--out` therefore shared a state directory, and the verification run overwrote the `last_good.npz` of the rollout it had just resumed from. Every individual save *succeeded*; the loss came from two runs agreeing on a name. **A rule about keeping artifacts needs a companion rule about not colliding on them, and only the first had been written down.** `ckpt_dir` is now derived from `--out`.

**The upload slot is coalescing, so leaking it is silent.** An early return between acquiring the in-flight lock and handing it to the worker thread would leave it held forever; the symptom is not a crash but every subsequent upload reporting "previous upload still in flight" and skipping. The run keeps going and saves nothing — precisely the failure the sync exists to prevent. There is now a single `_abandon_staging` path that every early return must go through, and a regression test that is verified to fail when the release is removed.

---

## 6. What the port did and did not change

`WindowNS` is written **once**, against a small backend interface (`backend.py`); numpy is a transparent pass-through, so the default path executes exactly the calls it always did. `step_batch` takes numpy in and hands numpy back out, so the assembly, the ledger, the metrics and every existing test are untouched — a GPU run changes one flag rather than the type of every array in the case study. The transfer cost is three round trips per macro-step, ~20 ms against 182 s.

Two things the port surfaced that are worth keeping:

- **The DCT is a matmul against a matrix built by scipy itself**, not a hand-rolled FFT reordering. Deriving DCT-II from an FFT by hand is easy to get wrong by $\sqrt{2}$ in the $k=0$ row alone, and that slip does not raise — it produces a smooth, plausible pressure field with a wrong constant mode, inside a solve that is *already singular* in the constant mode. This case study has spent enough sessions on errors that looked like physics. The matmul form is also the better GPU shape, being the one compute-bound piece.
- **The body force was left as a numpy array while the fields became tensors.** That works on CPU through numpy's interop protocol (with a `DeprecationWarning`) and **raises on CUDA**. It would have passed every test on the development machine and failed on the box — the exact failure this port's CPU-verifiability was meant to prevent, caught only because the warning was treated as a signal instead of noise.

---

## See Also
- [[vast-ai-training-runbook]] — Noether 1.1's runbook; different bottleneck, different machine (§0)
- [[wind-farm-implementation-log]] — what each rollout measured
- [[open-problems-atlas-0.1]] — OP-2, OP-3, OP-5
- [[case-study-wind-farm-wake-2d-atlas-0.1]] — the case study this serves
