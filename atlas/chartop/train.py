"""Training one registered arm: data in memory, resolution-bucketed batches, D4
augmentation, bfloat16 network with float64 objectives (decision D4), checkpoints.

A dataset directory holds shards ``<split>_n<res>_<k>.npz`` written by
`scripts/chartop_data.py` (the arrays of `data.arrays`, stacked) and, for labelled
sets, ``labels_<split>_n<res>_<k>.npz`` (`data.labels`).  An arm trains for a fixed
number of optimiser steps (equal compute across comparison arms) with AdamW, a short
linear warm-up and a cosine decay; every resolution bucket is visited in proportion
to its size, so every example is seen about equally often.  The validation set picks
the weights: the checkpoint with the lowest median ``e_H`` over the whole labelled
validation set is kept as ``best.pt`` beside ``last.pt``.  Everything is written
atomically, and the log is a JSON line per report, so a run killed at any point has
said what it measured.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from . import forms as fm
from . import model as md

KEYS = ("inputs", "scalars", "a11", "a12", "a22", "sigma", "w", "g", "s0")
META = ("seed", "sup_mu", "logJ_range", "s_grad", "fo", "freq", "family", "method")


@dataclass
class Arm:
    name: str
    backbone: str = "unet"
    size: str = "1M"
    form: str = "bubble"
    kind: str = "sine-hat"
    objective: str = "label-free"
    data: str = "train"                 # the training split's name
    limit: int | None = None            # use only the first `limit` examples
    steps: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch: dict = field(default_factory=lambda: {32: 32, 64: 32, 128: 8})
    seed: int = 0
    augment: bool = True
    val_every: int = 500
    p_size: str | None = None           # the split form's particular-field network (`forms`)
    bubble: str = "square"              # the kernel-exact forms' bubble (`forms.BUBBLES`, W358)
    strips: bool = False                # the side sines j = 9-15 by strip solves (W358's C3)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["batch"] = {str(k): v for k, v in self.batch.items()}
        return d


def _save_atomic(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    for k in range(20):
        try:
            os.replace(tmp, path)
            return
        except OSError:
            time.sleep(0.5 * (k + 1))
    raise OSError(f"could not replace {path}")


def _append(path: str, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()


def load_split(directory: str, split: str, limit: int | None = None,
               with_labels: bool = False, kind: str = "sine-hat") -> dict:
    """{n: {key: array}} for one split, every shard concatenated (and the labels when
    asked for; examples without labels are dropped then)."""
    out: dict[int, dict] = {}
    files = sorted(glob.glob(os.path.join(directory, f"{split}_n*_*.npz")))
    if not files:
        raise FileNotFoundError(f"no shards for split {split!r} in {directory}")
    for f in files:
        n = int(re.search(r"_n(\d+)_", os.path.basename(f)).group(1))
        z = np.load(f, allow_pickle=False)
        b = out.setdefault(n, {k: [] for k in KEYS + META})
        for k in KEYS + META:
            b[k].append(z[k])
        if with_labels:
            pre = "labels_" if kind == "sine-hat" else f"labels-{kind}_"
            lf = os.path.join(directory, pre + os.path.basename(f))
            zl = np.load(lf, allow_pickle=False)
            for k in ("H", "up", "trace_Lambda", "up_energy", "lseed"):
                b.setdefault(k, []).append(zl[k])
    for n, b in out.items():
        for k in list(b):
            b[k] = np.concatenate(b[k])
        if with_labels and not np.array_equal(b["seed"], b["lseed"]):
            raise ValueError(f"labels and examples disagree on seeds at n={n}")
    if limit is not None:
        # the first `limit` examples by seed, across resolutions
        allseeds = np.sort(np.concatenate([b["seed"] for b in out.values()]))
        cut = allseeds[min(limit, len(allseeds)) - 1]
        for n in list(out):
            keep = out[n]["seed"] <= cut
            out[n] = {k: v[keep] for k, v in out[n].items()}
            if not keep.any():
                del out[n]
    return out


def to_device(bucket: dict, device, labels: bool, labels_device=None) -> dict:
    """Examples to ``device`` in float32 (coefficients are cast to float64 per batch);
    labels, which are large, to ``labels_device`` (pinned host memory by default on a
    GPU) and moved per batch."""
    t = {k: torch.tensor(bucket[k], dtype=torch.float32, device=device) for k in KEYS}
    if labels:
        ld = labels_device or "cpu"
        for k in ("H", "up"):
            x = torch.tensor(bucket[k], dtype=torch.float32, device=ld)
            t[k] = x.pin_memory() if (ld == "cpu" and torch.device(device).type == "cuda") else x
        for k in ("trace_Lambda", "up_energy"):
            t[k] = torch.tensor(bucket[k], dtype=torch.float64, device=device)
    return t


def _batch(t: dict, idx, labels: bool):
    b = {k: t[k][idx] for k in KEYS}
    for k in ("a11", "a12", "a22", "sigma", "w", "g", "s0"):
        b[k] = b[k].to(torch.float64)
    lab = None
    if labels:
        n = t["inputs"].shape[-1] - 1
        dev = b["a11"].device
        ic = idx.to(t["H"].device)
        lab = {"H": t["H"][ic].to(dev, torch.float64, non_blocking=True).reshape(len(idx), -1, n - 1, n - 1),
               "up": t["up"][ic].to(dev, torch.float64, non_blocking=True).reshape(len(idx), n - 1, n - 1),
               "trace_Lambda": t["trace_Lambda"][idx], "up_energy": t["up_energy"][idx]}
    return b, lab


def validate(expert: md.Expert, val: dict, device, autocast: bool) -> dict:
    """Per-example e_H and e_p on a labelled set, with their bins; and, for the
    kernel-exact forms, e_K of the uniform state (`forms`), zero to rounding."""
    expert.eval()
    rows = {"e_H": [], "e_p": [], "sup_mu": [], "freq": [], "n": [], "seed": []}
    kernel = getattr(expert, "needs_load_fields", False)
    if kernel:
        rows["e_K"] = []
    with torch.no_grad():
        for n, t in val.items():
            N = t["inputs"].shape[0]
            bs = 16 if n <= 64 else 4
            for i0 in range(0, N, bs):
                idx = torch.arange(i0, min(N, i0 + bs), device=device)
                b, lab = _batch(t, idx, True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
                    H, up = fm.call(expert, b)
                o = md.objectives(b, H, up, expert.kind, expert.m, lab, "label-free")
                rows["e_H"] += o["e_H"].cpu().tolist()
                rows["e_p"] += o["e_p"].cpu().tolist()
                if kernel:
                    ks = fm.kernel_square(n, expert.kind, expert.m, str(H.device))
                    r = fm.residual(H, expert.last_level, b["s0"], ks)
                    rows["e_K"] += fm.kernel_error_torch(b, r, lab["trace_Lambda"]).cpu().tolist()
            rows["sup_mu"] += t["meta_sup_mu"].tolist()
            rows["freq"] += t["meta_freq"].tolist()
            rows["seed"] += t["meta_seed"].tolist()
            rows["n"] += [n] * N
    expert.train()
    return {k: np.asarray(v) for k, v in rows.items()}


def train(arm: Arm, data_dir: str, out_dir: str, device: str = "cuda",
          val_split: str = "val", minutes: float | None = None, log_every: int = 100) -> dict:
    """Train one arm; returns its summary.  Resumes from ``last.pt`` if present."""
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    autocast = dev.type == "cuda"
    torch.manual_seed(arm.seed)
    rng = np.random.default_rng(arm.seed)
    labelled = arm.objective == "supervised"
    tr = load_split(data_dir, arm.data, arm.limit, with_labels=labelled, kind=arm.kind)
    va = load_split(data_dir, val_split, None, with_labels=True, kind=arm.kind)
    T = {n: to_device(b, dev, labelled) for n, b in tr.items()}
    V = {}
    for n, b in va.items():
        V[n] = to_device(b, dev, True)
        for k in ("sup_mu", "freq", "seed"):
            V[n]["meta_" + k] = b[k]
    expert = fm.make(arm.backbone, arm.size, arm.form, arm.kind, p_size=arm.p_size,
                     bubble=arm.bubble, strips=arm.strips).to(dev)
    opt = torch.optim.AdamW(expert.parameters(), lr=arm.lr, weight_decay=arm.weight_decay)
    warm = max(1, arm.steps // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / arm.steps))))
    step, best = 0, float("inf")
    last = os.path.join(out_dir, "last.pt")
    if os.path.exists(last):
        st = torch.load(last, map_location=dev, weights_only=False)
        expert.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, best = st["step"], st["best"]
        rng = np.random.default_rng(st["rng_seed"] + step)
    log = os.path.join(out_dir, "log.jsonl")
    with open(os.path.join(out_dir, "arm.json"), "w", encoding="utf-8") as fh:
        json.dump({"arm": arm.as_dict(), "config": expert.config(),
                   "examples": {str(n): int(t["inputs"].shape[0]) for n, t in T.items()},
                   "val_examples": {str(n): int(t["inputs"].shape[0]) for n, t in V.items()}},
                  fh, indent=1)
    sizes = {n: t["inputs"].shape[0] for n, t in T.items()}
    total = sum(sizes.values())
    # each step draws a resolution in proportion to its examples per batch-step
    res = sorted(sizes)
    w_res = np.array([sizes[n] / arm.batch[n] for n in res], float)
    w_res /= w_res.sum()
    t0 = time.perf_counter()
    acc, acc_n = 0.0, 0
    while step < arm.steps:
        n = int(rng.choice(res, p=w_res))
        bs = min(arm.batch[n], sizes[n])
        idx = torch.as_tensor(rng.choice(sizes[n], bs, replace=False), device=dev)
        b, lab = _batch(T[n], idx, labelled)
        if arm.augment:
            k = int(rng.integers(8))
            b = md.d4(b, k)
            if lab is not None:
                lab = md.d4_labels(lab, arm.kind, expert.m, k)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=autocast):
            H, up = fm.call(expert, b)
        o = md.objectives(b, H, up, arm.kind, expert.m, lab, arm.objective,
                          H0=getattr(expert, "last_principal", None))
        loss = o["loss"].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        acc += float(loss.detach())
        acc_n += 1
        if step % log_every == 0 or step == arm.steps:
            _append(log, {"step": step, "loss": acc / acc_n, "lr": sched.get_last_lr()[0],
                          "elapsed_s": time.perf_counter() - t0, "n": n})
            acc, acc_n = 0.0, 0
        if step % arm.val_every == 0 or step == arm.steps:
            r = validate(expert, V, dev, autocast)
            med = float(np.median(r["e_H"]))
            rec = {"step": step, "val_e_H_median": med, "val_e_H_p90": float(np.quantile(r["e_H"], 0.9)),
                   "val_e_p_median": float(np.median(r["e_p"])),
                   "elapsed_s": time.perf_counter() - t0}
            if "e_K" in r:
                rec["val_e_K_max"] = float(np.max(r["e_K"]))
            _append(log, rec)
            if med < best:
                best = med
                _save_atomic({"model": expert.state_dict(), "config": expert.config(),
                              "step": step, "val_e_H_median": med}, os.path.join(out_dir, "best.pt"))
            _save_atomic({"model": expert.state_dict(), "opt": opt.state_dict(),
                          "sched": sched.state_dict(), "step": step, "best": best,
                          "rng_seed": arm.seed, "config": expert.config()}, last)
        if minutes is not None and (time.perf_counter() - t0) > 60.0 * minutes:
            _append(log, {"stopped": "wall-clock budget", "step": step})
            break
    _save_atomic({"model": expert.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                  "step": step, "best": best, "rng_seed": arm.seed, "config": expert.config()}, last)
    summary = {"arm": arm.name, "steps": step, "best_val_e_H_median": best,
               "elapsed_s": time.perf_counter() - t0, "examples": total}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def load_expert(path: str, device: str = "cpu") -> md.Expert:
    st = torch.load(path, map_location=device, weights_only=False)
    c = st["config"]
    e = fm.make(c["backbone"], c["size"], c["form"], c["kind"], c["m"], c.get("p_size"),
                c.get("bubble", "square"), c.get("strips", False))
    e.load_state_dict(st["model"])
    return e.to(device).eval()
