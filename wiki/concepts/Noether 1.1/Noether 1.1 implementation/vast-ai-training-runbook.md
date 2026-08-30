# Noether 1.1 — vast.ai Training Runbook

**Type:** Operations runbook (folder: Noether 1.1 / Noether 1.1 implementation)
**Status:** Live. Exact, copy-pasteable procedure for training Noether 1.1 on a rented vast.ai GPU box, phase by phase, with checkpoints synced to GitHub Releases (so an ephemeral box that dies is recoverable). Reflects the code as of [[implementation-log]] 2026-07-14 (P0→P4 wired end-to-end).
**Code:** `github.com/nonidino/physics-foundation-model`, branch `noether-1.1`. **Design:** [[training-scheme-1.1]], [[00-implementation-plan]].

---

## 0. The mental model (read once)

Training moves along **two independent axes**:

1. **Module phases P0→P4** (the training curriculum, per system — [[training-scheme-1.1]]). Each phase turns on the next set of modules on top of a validated foundation, and only advances when a **gate metric** passes:

   | Phase | Trains | Gate to advance |
   |---|---|---|
   | **P0** | tokenizer + descriptors + SIREN heads (as an autoencoder) | reconstruction error < discretization floor |
   | **P1** | backbone + decoder-as-increment; gates frozen at g=1; geometric edges only | single-step rel-L² < 0.05 |
   | **P2** | learned edge scorer + push-forward (rollout) loss | rollout rel-L² < 0.10 |
   | **P3** | conservation discovery (SFA + whitening); gates freed | conservation drift < `p3_drift_threshold` (0.01) |
   | **P4** | post-decoder diffusion (residual gap-fill), phase-separated | terminal — trains for its `--epochs` budget |

2. **Regime rungs** (which system — the "extend, don't restart" foundation-model axis): **NS (2D incompressible)** → **RBC (2D convection)**. RBC continues from the NS-trained model rather than starting fresh. (1D **Burgers** exists as a cheap warm-up but the main path is NS → RBC.)

**Checkpoints live on GitHub Releases**, one tag per (system, phase): `ckpt-ns-P0`, `ckpt-ns-P1`, …. Each phase's command **pulls** the previous phase's release if the local disk is empty, **trains** exactly that phase, and **pushes** its result back. So a box dying mid-run costs only that phase's since-last-save progress; re-running the same command on a fresh box resumes it.

**Data is already in the repo** — `data/{burgers,ns,rbc}/{train,val}` (train+val for all three rungs) is committed, so there is **no data-generation step**; the clone brings ~575 MB of corpus with it.

---

## 1. One-time prerequisites (on your laptop)

- A **GitHub Personal Access Token** with `repo` scope (Settings → Developer settings → Tokens). Used on the box so `gh` can push/pull checkpoint releases. Keep it handy as `ghp_…`.
- The repo is **private**; the box needs read access to clone. Easiest is a throwaway **SSH deploy key** generated on the box (below), added to your GitHub account.

---

## 2. Provision + connect to the box

1. On **vast.ai**, rent an instance with a **CUDA PyTorch image** (e.g. a `pytorch/pytorch` or vast "PyTorch" template) and a single **A100/H100/4090**. See §2b for the memory floor — pick the GPU from that table, not by price alone.
2. SSH in (vast shows the exact host/port):

```bash
ssh -p <PORT> root@<HOST>
```

---

## 2b. Minimum machine requirements

### GPU VRAM

| Config   | Params  | Optimizer+grads (exact) | Tokens/field | **Min VRAM**          | Comfortable                  | Notes                                  |
| -------- | ------- | ----------------------- | ------------ | --------------------- | ---------------------------- | -------------------------------------- |
| `smoke`  | 0.1 M   | <0.01 GB                | ~clamped     | **CPU ok**            | any                          | CI/local only                          |
| `medium` | 43.8 M  | 0.70 GB                 | 512          | **24 GB** (3090/4090) | **40–48 GB** (A100-40, L40S) | batch 16 @ `patch_domain_size=0.0625`  |
| `large`  | 327.4 M | 5.24 GB                 | 2048         | **48 GB**             | **80 GB** (A100-80/H100)     | batch 8; enable gradient checkpointing |

**How to read this.** The params/optimizer column is *exact* (fp32 master + grad + 2 AdamW moments = 16 B/param) — it is the floor before a single activation. The rest is activation memory, which scales roughly **linearly in `batch_size` × token count**, so the VRAM column assumes the shipped `batch_size` and `patch_domain_size` for that config. The activation figures are engineering estimates; **watch `nvidia-smi` on your first phase and adjust** rather than trusting them blindly.

> **Token density is a memory decision, not just a quality one.** `patch_domain_size` was lowered 0.25→0.0625 (medium) to fix the pixelated output ([[noether-1.1-fineness-and-token-density]]), which is a **16× token increase**. Memory was re-tuned to match — `batch_size` 48→16 (medium), 24→8 (large) — and the learned-edge candidate envelope is now bounded (below). A pre-2026-07-15 checkout at the new density **will OOM at P2**.

### Disk

- **≥ 60 GB** free. The clone brings ~575 MB of corpus; a `medium` full checkpoint is ~450 MB (`large` ~3 GB) and several are retained at once (`keep_top_k=3` periodic + phase finals).
- Checkpoint disk is now bounded: periodic saves rotate at `keep_top_k` (3), and **stale phase finals are swept** (`keep_phase_finals=1` keeps current + previous). Before this, phase finals accumulated forever because they are saved `prune=False`.

### System RAM

- **≥ 32 GB** recommended; the DataLoader holds decoded trajectory windows in host memory.

---

## 2c. If you hit `CUDA out of memory`

Turn these knobs **in order** — each is a one-line change to the same phase command, no code edit:

1. **`--batch-size N`** — halve it (16 → 8 → 4). Activation memory is ~linear in batch. First and best lever.
2. **`--patch-domain-size S`** — raise it to spend fewer tokens (0.0625 → 0.125 halves each axis, ~4× fewer tokens). Costs fineness; use only if batch is already small.
3. **`--no-amp` off** — make sure you are *not* passing `--no-amp`: bf16 autocast roughly halves activation memory vs fp32.
4. **`learned_edge_cutoff_patches`** (in `configs/__init__.py`, default 8.0) — lower to 4.0 to shrink the learned-edge candidate envelope further.

**Why P2 specifically OOMs.** The learned-edge scorer only comes online at **P2**, so P0/P1 can pass and the box dies later — exactly the "ran for a while, then OOM" signature. The scorer builds one row per *candidate pair*, and with no envelope that is all $\binom{P}{2}$ pairs: **496 at 32 tokens but 130 816 at 512 and 2.1 M at 2048** (~0.6 GB and ~19.5 GB for a single activation, before autograd). The `learned_edge_cutoff_patches=8.0` envelope ($R=8s\approx4r$, the $R{=}4r$ of [[edge-generation-1.1]]) bounds candidates to a fixed count *per node*, making the cost $O(P)$ instead of $O(P^2)$ — measured: 130 816→39 616 (medium), 2 096 128→179 584 (large). If you are on an older checkout, that envelope is `None` (all pairs) and no batch size will save `large`.

**A DIFFERENT signature — OOM specifically on RESUME, a run that trained fine before pausing.** Two mitigations landed 2026-07-19, both automatic on a fresh `git pull`, no flags needed:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is now set by both `train_phase.sh` and `train.py` itself (before `import torch`, so it's correct however you invoke it) unless you've already set it yourself. This is PyTorch's own documented mitigation for allocator fragmentation from varying tensor shapes — exactly what `patch_size_jitter` causes, since it changes token count (and therefore nearly every shape-dependent tensor in the model) from step to step.
- `patch_size_jitter` now **dwells** on one density for `patch_jitter_dwell_steps` (default 25) consecutive steps before redrawing, instead of every single step. This bounds the shape churn while still exercising every configured density many times per epoch.

If you still see it, the tell is the error's own "`X GiB is reserved by PyTorch but unallocated`" line — a large gap there (not just "allocated" near the ceiling) is fragmentation, not a real memory shortfall, and the fix is one of the two above (lower `patch_jitter_dwell_steps` isn't the direction — *raise* it, or set `patch_size_jitter=()` to disable the augmentation entirely for a memory-constrained box).

**A THIRD signature — genuine occupancy, not fragmentation.** `... GiB is allocated by PyTorch, and <a small amount, tens/low-hundreds of MiB> is reserved by PyTorch but unallocated` — nearly the whole card is live tensors, not slack. This is push-forward's **horizon** (`H`, from `pushforward_horizons=(1,2,4,8)`), not the tokenizer's density. The `H` unroll substeps feed forward autoregressively but *detached* (no backprop through the rollout chain) — yet each substep's own forward still contributes to **one accumulated loss that `.backward()`s once at the end**, so autograd must hold all `H` substeps' full activations simultaneously. At `H=8` (the terminal, priciest horizon — reached once training has run long enough, which is exactly the "worked for a while, then OOM" pattern) combined with a sustained fine-density dwell window, this can genuinely exceed a 40GB card with zero fragmentation involved.

Fixed 2026-07-19, also automatic: `pushforward_grad_checkpoint` (default on) gradient-checkpoints each push-forward substep when `H>1` and gradients are being tracked — trading one extra recomputed forward per substep (during backward) for peak activation memory that no longer scales with `H`. Verified bit-exact (loss + gradients, including P2's stochastic Gumbel edge sampling) against the uncheckpointed path. If you need to trade the extra compute back for speed on a card with headroom, set `pushforward_grad_checkpoint: False` in the config.

---

## 2d. If it stalls at `[publish] uploading to existing release …`

Checkpoint uploads are **best-effort and must never block training**. Since 2026-07-19:

- Every `gh` call is **time-bounded** (`UPLOAD_TIMEOUT`, default 1800 s; override with `NOETHER_UPLOAD_TIMEOUT`) and runs with **stdin closed**, so a stalled transfer or an interactive `gh` prompt can no longer hang the run forever. On expiry you get a warning and training continues.
- **Periodic** checkpoint uploads run on a background thread — training keeps stepping while ~500 MB goes out. If a previous upload is still in flight the next one is **skipped, not queued** (the newer checkpoint supersedes it anyway).
- The **phase-end** upload is deliberately synchronous (and drains any in-flight background upload first): it is the seed the next phase pulls, so it must land before the process exits.

If uploads are the bottleneck on a slow link, halve the bytes with **`--no-publish-full`** (skips the ~335 MB resumable checkpoint, keeps the ~155 MB inference one) or upload less often with **`--publish-every N`**. Note `--no-publish-full` trades away mid-phase crash *resumability*, so prefer `--publish-every`.

---

## 3. GitHub SSH key (on the box) → clone

```bash
ssh-keygen -t ed25519 -C "nonidino@gmail.com"   # press Enter through the prompts
cat ~/.ssh/id_ed25519.pub                        # copy this, add at github.com/settings/keys
# (verify)   ssh -T git@github.com

git clone -b noether-1.1 git@github.com:nonidino/physics-foundation-model.git
cd physics-foundation-model
```

---

## 4. Environment setup (uv — reuses the image's CUDA torch)

**Fastest path** (one line — installs the package + small deps into the image's Python, keeping its CUDA torch):

```bash
bash vastai_setup.sh        # installs uv + gh, `uv pip install --system -e .`, verifies CUDA
```

<details><summary>What that does / manual equivalents</summary>

```bash
# install uv, then install noether11 editable into the image Python (torch reused):
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv pip install --system -e .

# OR a fully isolated env (downloads a CUDA torch wheel, ~2 GB, but reproducible):
uv sync && source .venv/bin/activate
```
With `uv sync` you must keep the venv **activated** in every shell you train from (so `python` is the venv's). With `uv pip install --system` there is nothing to activate.
</details>

Sanity check the GPU:

```bash
python -c "import torch; print('CUDA', torch.cuda.is_available(), '| bf16', torch.cuda.is_bf16_supported(), '|', torch.cuda.get_device_name(0))"
```

Authenticate GitHub so checkpoints can sync (the token from §1):

```bash
#export GH_TOKEN=ghp_...      # a 'repo'-scoped token   (or run: gh auth login)

#unset GH_TOKEN GITHUB_TOKEN           if needed 

gh auth login -h github.com -s repo -w
```

> Export `GH_TOKEN` **before** you start `tmux` (below) — tmux copies the environment at creation time. The repo (`nonidino/physics-foundation-model`) is auto-detected from the `origin` remote; override with `NOETHER_GH_REPO=owner/repo` only if publishing elsewhere.

---

## 5. Train — the full P0 → P4 curriculum

Each phase is **its own command** via `scripts/train_phase.sh <PHASE> <SYSTEM>`. Run inside **tmux** so it survives an SSH disconnect.

### 5a. NS (rung 2) — the base model

```bash
tmux new -s train          # Ctrl-b then d to detach later; `tmux attach -t train` to return

for P in P0 P1 P2 P3 P4; do
  bash scripts/train_phase.sh $P ns --config medium --epochs 10000
done
```

- `train_phase.sh` wires each phase for you: it pulls `ckpt-ns-P{n-1}` as the seed, trains phase `P{n}`, and pushes the result to release **`ckpt-ns-P{n}`**. It tees a log to `train_ns_P{n}.log`.
- You can also run them **one at a time** (recommended for the first run, so you watch each gate pass before committing GPU-hours to the next):

```bash
bash scripts/train_phase.sh P0 ns --config medium --epochs 10000
# watch the "--- validation step … metrics=… patience=…/3" lines; when it prints
#   ">>> PHASE ADVANCE: P0 -> P1 <<<"  the phase is done and the release is pushed.
bash scripts/train_phase.sh P1 ns --config medium --epochs 10000
# … P2, P3, P4
```

### 5b. RBC (rung 3) — extend the NS model onto convection

```bash
bash scripts/train_phase.sh P0 rbc --config medium --epochs 10000   # seeds from ckpt-ns-P2
for P in P1 P2 P3 P4; do
  bash scripts/train_phase.sh $P rbc --config medium --epochs 10000
done
```

RBC `P0` seeds from **`ckpt-ns-P2`** (the deterministic core, before NS-specific conservation/diffusion) — "extend, don't restart". To seed from a different tag, pass `--seed-tag ckpt-ns-P4`.

### Optional: Burgers (rung 1) as a cheap 1D sanity first

```bash
for P in P0 P1 P2; do bash scripts/train_phase.sh $P burgers --config medium --epochs 3000; done
```

---

## 6. What "done" looks like per phase

During a phase you'll see periodic lines like:

```
  epoch 3/…  | step  1450 | batch  40/98 | P1 | eps=0.30 | H=1 | mse=0.0121 spectral=3.4 total=0.44 | 812.3s
  --- validation step 1600 (epoch 3): P1 metrics=rel_l2=0.061, … patience=2/3
  >>> PHASE ADVANCE: P1 -> P2 (epoch 3, step 1600) <<<
```

- The phase **advances** after the gate metric passes `patience` (3) consecutive validation checks, then the command exits and the release is pushed. The next command picks it up.
- P4 has **no gate** (it's terminal): it trains for the full `--epochs` and then pushes `ckpt-<sys>-P4`. Give it enough epochs; stop it (`Ctrl-c` inside tmux) when the diffusion loss has plateaued.

**If a phase exhausts `--epochs` without advancing**, its checkpoint stays at that phase and the *next* phase's command will refuse to start (a clear "run P{n} first" error). That's the same gated model as P0–P2 today — raise `--epochs`, or, if a gate is genuinely mis-calibrated, adjust its threshold in `src/noether11/configs/__init__.py` (`p1_rel_l2_threshold`, `p2_rollout_threshold`, `p3_drift_threshold`) and re-run.

---

## 7. Crash recovery (the whole point of the GitHub sync)

If the box dies, times out, or you want to move GPUs:

1. Rent a new box, redo §2–§4 (clone + `bash vastai_setup.sh` + `export GH_TOKEN=…`).
2. Re-run **the exact same phase command** that was interrupted, e.g. `bash scripts/train_phase.sh P2 ns`. Because the local disk is empty, it pulls that phase's latest release (`ckpt-ns-P2`, pushed at the last checkpoint interval) and continues from there.

Checkpoints are pushed **every checkpoint interval** (not just at phase end), so at most one interval of progress is ever at risk. Local copies live in `./checkpoints/<system>/`.

**"`--phase PX` says it's starting at P{X-1}" (missing phase-X seed).** Each phase advances by writing a *final* checkpoint at the **new** phase (`ckpt_final_P{X}.pt`) — that's the seed the next phase resumes from. If `--phase PX` can't find one (locally or in the `ckpt-<sys>-P{X-1}` release), it starts at the previous phase and the start-phase guard stops it. Recovery: **re-run `--phase P{X-1}`** — it resumes the (already-trained) previous phase from its newest checkpoint, re-passes the gate within a few validation checks, and regenerates + pushes the phase-X seed. Then `--phase PX` proceeds. (Checkpoints saved by builds **before 2026-07-14** could lose that final checkpoint to the keep-top-K pruning; `git pull` for the fix, then re-run the previous phase once.)

---

## 8. Getting weights back / running the GUI

- Every release holds two assets: `ckpt_latest.pt` (full, resumable) and `ckpt_latest.infer.pt` (model-only, ~⅓ size — for the GUI/eval on another machine).
- Pull one anywhere:

```bash
python -m noether11.train.publish pull --tag ckpt-ns-P4 --dir ./checkpoints --pattern ckpt_latest.infer.pt
```

- Load it for inference/eval: `from noether11 import load_checkpoint, rollout` (config is embedded in the checkpoint — no `--config` needed).
- 2D GUI (needs the extra deps): `uv pip install --system -e ".[gui]"` then `python gui/server.py` (or point it at a checkpoint).

> **Note (portion 10, not yet wired):** the eval-time **diffusion sampling** and **hard conservation projection** toggles in the rollout are still stubs, so the "blurred-vs-sharp" (P4) and "drift with/without projection" (P3) *contrast* views aren't runnable yet — the trained modules exist, the inference-time application is the next build item.

---

## 9. Knobs & reference

| Flag / setting | Default | Meaning |
|---|---|---|
| `--config` | `medium` | `medium` (~44M), `large` (~327M), or `smoke` (tiny, CPU) |
| `--epochs` | 10000 (via `train_phase.sh`) | max epochs for the phase (an epoch = one full pass over all (t,t+1) pairs) |
| `--data-root` | `./data` | committed corpus location |
| `--checkpoint-dir` | `./checkpoints/<sys>` | local checkpoints |
| `--no-amp` | off | disable bf16 autocast (pure fp32 — safe fallback if an AMP dtype issue appears). Roughly **doubles** activation memory; don't set it casually |
| `--batch-size` | config (`medium` 16, `large` 8) | **primary OOM knob** — halve on `CUDA out of memory` (§2c) |
| `--patch-domain-size` | config (`medium` 0.0625, `large` 0.03125) | token density; smaller = finer/sharper but more memory. Second OOM knob ([[noether-1.1-fineness-and-token-density]]) |
| `--config large` | — | for the 327M run; use ≥48–80 GB GPU, expect much longer per phase |
| `learned_edge_cutoff_patches` | 8.0 | learned-edge candidate envelope $R=8s$, scaled to the *live* token density; lower to 4.0 to cut P2 memory further |
| `patch_size_jitter` | (1.0, 1.5, 2.0) | multi-scale training rungs. **Keep every multiplier ≥ 1** — a sub-1 rung makes patches finer than the base and silently sets peak memory (0.5 → 4× the tokens) |
| `dataloader_workers` | 4 | DataLoader worker processes (CUDA only). Lower if the box is CPU-starved |
| `keep_top_k` / `keep_phase_finals` | 3 / 1 | checkpoint disk retention (§2b) |
| `p3_drift_threshold` etc. | in `configs/__init__.py` | gate thresholds, if a gate needs re-calibrating |

**Cumulative alternative** (all phases in one process, no per-phase releases): `python scripts/train.py --system ns --config medium --phase-until P4 --data-root ./data --checkpoint-dir ./checkpoints/ns --publish --epochs 10000`. The per-phase (`train_phase.sh`) flow is preferred on ephemeral boxes because each phase is independently resumable.

**Compute expectation** ([[implementation-log]] 2026-07-08 estimate, unvalidated): `medium` through the NS+RBC rungs is on the order of weeks of continuous A100 time; `large` across all rungs is not realistic on one GPU — stage it. Watch the throughput (`…s` at each log line) on the first phase to extrapolate.

---

## See Also
- [[noether-1.1-fineness-and-token-density]] — why token density changed, and why that made VRAM a first-class concern (§2b/§2c)
- [[edge-generation-1.1]] — the $R{=}4r$ candidate envelope the P2 memory fix implements
- [[training-scheme-1.1]] — the P0→P5 module order + regime curriculum this executes
- [[00-implementation-plan]] — the build plan / milestones M0–M7
- [[implementation-log]] — what's built (P0→P4 wired 2026-07-14) and what's next (portion 10 eval toggles)
- [[noether-1.1-medium]] / [[noether-1.1-large]] — the size configs `--config` selects
