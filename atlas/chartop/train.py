"""Training one registered arm: data in memory, resolution-bucketed batches, D4
augmentation, bfloat16 network with float64 objectives (decision D4), checkpoints.

A dataset directory holds shards ``<split>_n<res>_<k>.npz`` written by
`scripts/chartop_data.py` (the arrays of `data.arrays`, stacked) and, for labelled
sets, ``labels_<split>_n<res>_<k>.npz`` (`data.labels`), or, where a shard has only its
light labels (``light_<split>_n<res>_<k>.npz``: each exact column's energy ``lam``, the trace
and the particular field's energy, `chartop_data.py --light-labels`), those: enough for the
two-level expert's relative objective, at 64 numbers an example.  An arm trains for a fixed
number of optimiser steps (equal compute across comparison arms) with AdamW, a short
linear warm-up and a cosine decay; every resolution bucket is visited in proportion
to its size, so every example is seen about equally often.  The validation set picks
the weights: the checkpoint with the lowest median ``e_H`` over the whole labelled
validation set is kept as ``best.pt`` beside ``last.pt``.  Everything is written
atomically, and the log is a JSON line per report, so a run killed at any point has
said what it measured.
"""

from __future__ import annotations

import functools
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
from . import twolevel as tl

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
    #: False: the training steps skip the strips (`forms.call(strip_solve=False)`), which carry
    #: no gradient, so the weights are the same; validation and every answer still solve them.
    #: The logged training loss then counts the plain square's strip columns, not the strips'.
    strips_in_training: bool = True
    #: the kernel columns' source before the projection (`forms.HATS`; section 11's B1)
    hats: str = "learned"
    #: None: every port mode weighted alike, as before.  A name in `forms.MODE_PRIORS`: the
    #: objective weights the modes' errors by that prior's covariance of a piece's traces,
    #: ``tr(Sigma E^T A E)`` (`forms.mode_mixing`; section 11's B2).  Validation is unchanged.
    mode_prior: str | None = None
    #: the Stage 0 rework (`twolevel`): None, as before; an integer n_c: the coarse correction at
    #: that coarse size after the form (`twolevel.TwoLevelExpert`), validation logging the
    #: network's share rho of TL0's error, with the dead-network stop
    coarse: int | None = None
    #: a checkpoint whose weights start the run (S's, for instance); None: as before
    init_from: str | None = None
    #: with ``init_from``: "zero" sets the loaded output heads to zero, so a two-level run starts
    #: exactly at TL0; "loaded" keeps them.  Without it the heads start at zero, as the U-Net's do
    head_init: str = "zero"
    #: the relative objective (``objective="relative"``, two-level arms only) leaves out a piece's
    #: term when TL0's error there is below this fraction of its own energy, and counts it
    drop_easy: float = 1e-8
    #: branch 2c's corner treatment (two-level arms only): None, as before; ``frac`` (8): the
    #: coarse space enlarged at each example's non-right corners with its blocks of n/frac
    #: (`coarse_patches`).  The flags come from each shard's ``corners_`` file
    #: (`scripts/chartop_corner_flags.py`), which `load_split` then requires
    corners: int | None = None

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
               with_labels: bool = False, kind: str = "sine-hat", with_corners: bool = False) -> dict:
    """{n: {key: array}} for one split, every shard concatenated (and the labels when
    asked for; examples without labels are dropped then).  ``with_corners``: each shard's
    ``corners_`` file too, its seeds checked against the shard's (``corners`` [N, 4] bool)."""
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
            light = os.path.join(directory, "light_" + os.path.basename(f))
            if os.path.exists(lf) or kind != "sine-hat" or not os.path.exists(light):
                zl = np.load(lf, allow_pickle=False)
                keys = ("H", "up", "trace_Lambda", "up_energy", "lseed")
            else:
                zl = np.load(light, allow_pickle=False)
                keys = ("lam", "trace_Lambda", "up_energy", "lseed")
            if b.get("_labels", keys) != keys:
                raise ValueError(f"split {split!r} mixes full and light labels at n={n}")
            b["_labels"] = keys
            for k in keys:
                b.setdefault(k, []).append(zl[k])
        if with_corners:
            zc = np.load(os.path.join(directory, "corners_" + os.path.basename(f)), allow_pickle=False)
            if not np.array_equal(zc["cseed"], z["seed"]):
                raise ValueError(f"the corner flags and the examples disagree on seeds in {f}")
            b.setdefault("corners", []).append(np.asarray(zc["corners"], bool))
    for n, b in out.items():
        b.pop("_labels", None)
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
    if "corners" in bucket:
        t["corners"] = torch.tensor(bucket["corners"], dtype=torch.bool, device=device)
    if labels:
        if "H" not in bucket:                                       # light labels
            for k in ("lam", "trace_Lambda", "up_energy"):
                t[k] = torch.tensor(bucket[k], dtype=torch.float64, device=device)
            return t
        ld = labels_device or "cpu"
        for k in ("H", "up"):
            x = torch.tensor(bucket[k], dtype=torch.float32, device=ld)
            t[k] = x.pin_memory() if (ld == "cpu" and torch.device(device).type == "cuda") else x
        for k in ("trace_Lambda", "up_energy"):
            t[k] = torch.tensor(bucket[k], dtype=torch.float64, device=device)
    return t


@functools.lru_cache(maxsize=None)
def corner_perm(k: int) -> tuple:
    """Where symmetry ``k`` of the square (`model.d4`'s numbering) takes each corner (0 at
    (0, 0), then counter-clockwise): read off `model._spatial` applied to a 2 x 2 node grid
    marked at its corners, so it cannot disagree with how the batch's arrays move."""
    pos = ((0, 0), (0, 1), (1, 1), (1, 0))                  # corner c's node, [j, i], n = 1
    a = torch.zeros(1, 2, 2)
    for c, (j, i) in enumerate(pos):
        a[0, j, i] = c + 1
    s = md._spatial(a, k & 1, (k >> 1) & 1, (k >> 2) & 1)
    return tuple(next(d for d, (j, i) in enumerate(pos) if int(s[0, j, i]) == c + 1) for c in range(4))


def d4_corners(flags, k: int):
    """The transformed batch's corner flags: corner ``corner_perm(k)[c]`` of the new square is
    corner ``c`` of the old."""
    perm = corner_perm(k)
    out = torch.empty_like(flags)
    for c in range(4):
        out[:, perm[c]] = flags[:, c]
    return out


def d4_light_labels(lab: dict, kind: str, m: int, n: int, k: int) -> dict:
    """Light labels under symmetry ``k``: the exact column energies follow the port basis's
    signed permutation (`model.d4_ports`; an energy has no sign); the totals do not move."""
    perm, _sign = md.d4_ports(kind, n, m, k)
    lam = torch.empty_like(lab["lam"])
    lam[:, torch.as_tensor(perm, device=lam.device)] = lab["lam"]
    return dict(lab, lam=lam)


def _batch(t: dict, idx, labels: bool):
    b = {k: t[k][idx] for k in KEYS}
    for k in ("a11", "a12", "a22", "sigma", "w", "g", "s0"):
        b[k] = b[k].to(torch.float64)
    if "corners" in t:
        b["corners"] = t["corners"][idx]
    lab = None
    if labels and "lam" in t:                                       # light labels
        lab = {"lam": t["lam"][idx], "trace_Lambda": t["trace_Lambda"][idx], "up_energy": t["up_energy"][idx]}
    elif labels:
        n = t["inputs"].shape[-1] - 1
        dev = b["a11"].device
        ic = idx.to(t["H"].device)
        lab = {"H": t["H"][ic].to(dev, torch.float64, non_blocking=True).reshape(len(idx), -1, n - 1, n - 1),
               "up": t["up"][ic].to(dev, torch.float64, non_blocking=True).reshape(len(idx), n - 1, n - 1),
               "trace_Lambda": t["trace_Lambda"][idx], "up_energy": t["up_energy"][idx]}
    return b, lab


def validate(expert: md.Expert, val: dict, device, autocast: bool, twin=None, easy: float = tl.EASY) -> dict:
    """Per-example e_H and e_p on a labelled set, with their bins; and, for the
    kernel-exact forms, e_K of the uniform state (`forms`), zero to rounding.  With ``twin``
    (`twolevel.zero_heads_twin` of a two-level expert), also the network's share ``rho`` of
    TL0's error over the learned columns, ``nan`` where TL0 left nothing to learn."""
    expert.eval()
    rows = {"e_H": [], "e_p": [], "sup_mu": [], "freq": [], "n": [], "seed": []}
    kernel = getattr(expert, "needs_load_fields", False)
    if kernel:
        rows["e_K"] = []
    if twin is not None:
        rows["rho"] = []
    with torch.no_grad():
        for n, t in val.items():
            N = t["inputs"].shape[0]
            bs = 16 if n <= 64 else 4
            for i0 in range(0, N, bs):
                idx = torch.arange(i0, min(N, i0 + bs), device=device)
                b, lab = _batch(t, idx, True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast):
                    H, up = tl.call(expert, b)
                o = md.objectives(b, H, up, expert.kind, expert.m, lab, "label-free")
                rows["e_H"] += o["e_H"].cpu().tolist()
                rows["e_p"] += o["e_p"].cpu().tolist()
                if kernel:
                    ks = fm.kernel_square(n, expert.kind, expert.m, str(H.device))
                    r = fm.residual(H, expert.last_level, b["s0"], ks)
                    rows["e_K"] += fm.kernel_error_torch(b, r, lab["trace_Lambda"]).cpu().tolist()
                if twin is not None:
                    ref = tl.reference(twin, b, lab, strip_solve=True, easy=easy)
                    L = sorted(sum(tl.learned_groups(expert.kind, expert.m).values(), []))
                    eps = (tl.column_energies(b, H, expert.kind, expert.m)[:, L].sum(-1) - ref["lam_L"])
                    rows["rho"] += tl.rho(eps.cpu().numpy(), ref["eps0"].cpu().numpy(),
                                          lab["trace_Lambda"].cpu().numpy(), easy).tolist()
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
    if arm.objective == "relative" and arm.coarse is None:
        raise ValueError("the relative objective is the two-level expert's: set Arm.coarse")
    if arm.head_init not in ("zero", "loaded"):
        raise ValueError(f"head_init {arm.head_init!r}: 'zero' or 'loaded'")
    if arm.corners is not None and arm.coarse is None:
        raise ValueError("the corner treatment enlarges the coarse level: set Arm.coarse")
    if arm.objective == "supervised" and any(
            os.path.exists(os.path.join(data_dir, "light_" + os.path.basename(f)))
            and not os.path.exists(os.path.join(data_dir, "labels_" + os.path.basename(f)))
            for f in glob.glob(os.path.join(data_dir, f"{arm.data}_n*_*.npz"))):
        raise ValueError("the supervised objective needs the exact modes: light labels carry only their energies")
    torch.manual_seed(arm.seed)
    rng = np.random.default_rng(arm.seed)
    labelled = arm.objective in ("supervised", "relative")
    flags = arm.corners is not None
    tr = load_split(data_dir, arm.data, arm.limit, with_labels=labelled, kind=arm.kind, with_corners=flags)
    va = load_split(data_dir, val_split, None, with_labels=True, kind=arm.kind, with_corners=flags)
    T = {n: to_device(b, dev, labelled) for n, b in tr.items()}
    V = {}
    for n, b in va.items():
        V[n] = to_device(b, dev, True)
        for k in ("sup_mu", "freq", "seed"):
            V[n]["meta_" + k] = b[k]
    if arm.coarse is None:
        expert = fm.make(arm.backbone, arm.size, arm.form, arm.kind, p_size=arm.p_size,
                         bubble=arm.bubble, strips=arm.strips, hats=arm.hats).to(dev)
    else:
        expert = tl.make(arm.backbone, arm.size, arm.form, arm.kind, p_size=arm.p_size,
                         bubble=arm.bubble, strips=arm.strips, hats=arm.hats, n_c=arm.coarse,
                         corners=arm.corners).to(dev)
    if arm.init_from is not None:
        st0 = torch.load(arm.init_from, map_location=dev, weights_only=False)
        expert.load_state_dict(st0["model"], strict=True)
        if arm.head_init == "zero":
            with torch.no_grad():
                for net in (expert.net, getattr(expert, "pnet", None)):
                    if net is not None:
                        net.head.weight.zero_()
                        net.head.bias.zero_()
    twin = tl.zero_heads_twin(expert) if arm.coarse is not None else None
    mixes = {}

    def mix_for(n_: int):
        if arm.mode_prior is None:
            return None
        if n_ not in mixes:
            F = fm.mode_mixing(arm.mode_prior, arm.kind, n_, expert.m, fm.trained_columns(expert))
            mixes[n_] = torch.tensor(F, dtype=torch.float64, device=dev)
        return mixes[n_]
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
            if "corners" in b:
                b["corners"] = d4_corners(b["corners"], k)
            if lab is not None:
                lab = (md.d4_labels(lab, arm.kind, expert.m, k) if "H" in lab
                       else d4_light_labels(lab, arm.kind, expert.m, n, k))
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=autocast):
            H, up = tl.call(expert, b, strip_solve=arm.strips_in_training)
        if arm.objective == "relative":
            ref = tl.reference(twin, b, lab, strip_solve=arm.strips_in_training, easy=arm.drop_easy)
            o = tl.relative_objective(b, H, up, ref, arm.kind, expert.m)
        else:
            o = md.objectives(b, H, up, arm.kind, expert.m, lab, arm.objective,
                              H0=getattr(expert, "last_principal", None), mix=mix_for(n))
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
            r = validate(expert, V, dev, autocast, twin=twin, easy=arm.drop_easy)
            med = float(np.median(r["e_H"]))
            rec = {"step": step, "val_e_H_median": med, "val_e_H_p90": float(np.quantile(r["e_H"], 0.9)),
                   "val_e_p_median": float(np.median(r["e_p"])),
                   "elapsed_s": time.perf_counter() - t0}
            if "e_K" in r:
                rec["val_e_K_max"] = float(np.max(r["e_K"]))
            rho_dead = False
            if "rho" in r:
                s = tl.summarise_rho(r["rho"])
                rec.update({f"val_rho_{k}": v for k, v in s.items()})
                rho_dead = tl.dead(step, arm.steps, s["median"])
            _append(log, rec)
            if med < best:
                best = med
                _save_atomic({"model": expert.state_dict(), "config": expert.config(),
                              "step": step, "val_e_H_median": med}, os.path.join(out_dir, "best.pt"))
            _save_atomic({"model": expert.state_dict(), "opt": opt.state_dict(),
                          "sched": sched.state_dict(), "step": step, "best": best,
                          "rng_seed": arm.seed, "config": expert.config()}, last)
            if rho_dead:
                _append(log, {"stopped": "dead network: validation median rho below "
                              f"{tl.DEAD_RHO} after {tl.DEAD_AFTER:.0%} of the steps", "step": step})
                break
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
    if c.get("n_c") is not None:                     # a two-level expert (`twolevel`)
        return tl.load(path, device)
    e = fm.make(c["backbone"], c["size"], c["form"], c["kind"], c["m"], c.get("p_size"),
                c.get("bubble", "square"), c.get("strips", False), c.get("hats", "learned"))
    e.load_state_dict(st["model"])
    return e.to(device).eval()
