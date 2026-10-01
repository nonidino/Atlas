"""Demo step 8: train the learned window stepper (on the rented GPU).

    python scripts/learned_train.py --data out/learned-case/data --minutes 75
    python scripts/learned_train.py --data <smoke dir> --smoke          # a CPU smoke test

**What is learned.**  `learned_net.WindowUNet` at the registered size
(`learned_gate.NET_WIDTH`, `NET_DEPTH`) maps a window's (u - U, v, f) to the
classical exposed window's increment over one macro-step (`learned_data.py`).

**The loss**, per sample, normalised per channel by the increment's rms:
  * the increment's squared error, weighted cell by cell by the window's own blend
    weight chi (`RectangleTiling.chi`): only chi times the window's output reaches
    the global field, so a cell the blend ignores is not asked for;
  * for the ring pairs (S5), the same weighted error on the RESPONSE: the
    difference between the network on the perturbed and on the unperturbed window,
    against the classical window's difference.

**Choosing the weights, on the validation layouts alone.**  Every few epochs the
learned decomposition (`learned_arms.LearnedFarmRun`: the network in the windows,
the blend, projection and band classical) is marched 40 macro-steps on three
validation layouts and compared with their classical marches: the rms velocity
difference at 40 macro-steps, on the gate's D/16 grid, averaged.  The weights with
the smallest are kept.  The held-out layout is never seen here.

fp32 throughout, TF32 off (a float32 expert's small increments are what TF32
rounds away).  Writes ``out/learned-case/window-net.pt`` (the chosen weights, with
their scales) and ``training-log.json``.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse                                                         # noqa: E402
import datetime as dt                                                   # noqa: E402
import hashlib                                                          # noqa: E402
import json                                                             # noqa: E402
import math                                                             # noqa: E402
import sys                                                              # noqa: E402
import time                                                             # noqa: E402

import numpy as np                                                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out", "learned-case")

PAIR_WEIGHT = 1.0
ROLLOUT_SEEDS = (2000, 2001, 2002)


def load(data_dir: str, seeds) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All rows of these layouts in one float16 array, and per row its window and
    (for a ring pair) the row of its unperturbed base; -1 for a base row."""
    metas = [json.load(open(os.path.join(data_dir, "traj-%d.json" % s))) for s in seeds]
    n = sum(len(m["rows"]) for m in metas)
    first = np.load(os.path.join(data_dir, "traj-%d.npy" % seeds[0]), mmap_mode="r")
    X = np.empty((n,) + first.shape[1:], dtype=np.float16)
    win = np.empty(n, dtype=np.int64)
    ref = np.empty(n, dtype=np.int64)
    at = 0
    for s, m in zip(seeds, metas):
        a = np.load(os.path.join(data_dir, "traj-%d.npy" % s), mmap_mode="r")
        k = len(m["rows"])
        X[at:at + k] = a
        rows = np.asarray(m["rows"], dtype=np.int64)
        win[at:at + k] = rows[:, 1]
        ref[at:at + k] = np.where(rows[:, 2] >= 0, rows[:, 2] + at, -1)
        at += k
    return X, win, ref


def chi_weights() -> np.ndarray:
    """Each window's blend weight on its box, farm-12's tiling: ``[K, h, w]``."""
    from atlas.workbench import learned_gate as G
    from atlas.workbench.spec import example_case
    from atlas.workbench.tiling import RectangleTiling
    s = example_case(G.CASE)
    t = RectangleTiling(s.domain.nx, s.domain.ny,
                        [(w.id, (w.x0, w.y0, w.nx, w.ny)) for w in s.windows],
                        s.coupling.ramp_cells)
    w = np.stack(t.chi).astype(np.float32)
    w[:, 0, :] = w[:, -1, :] = w[:, :, 0] = w[:, :, -1] = 0.0      # the held ring
    return w


def rollout_score(net, data_dir: str, seeds, steps: int) -> dict:
    """The learned decomposition against the classical one on validation layouts."""
    from atlas.workbench import learned_arms as LA
    from atlas.workbench import learned_gate as G
    out = {}
    for seed in seeds:
        lay = G.sample_layout(seed)
        spec, ind = LA.layout_spec(lay)
        run = LA.LearnedFarmRun(spec, net, arms=("learned",), threads=1, induction=ind)
        ref = np.load(os.path.join(data_dir, "traj-%d-global.npz" % seed))
        s = run.initial("learned")
        e_max, ok = 0.0, True
        for n in range(1, steps + 1):
            s = run.step("learned", s)
            if not (np.all(np.isfinite(s.u)) and np.all(np.isfinite(s.v))):
                ok = False
                break
            ue, ve = ref["u"][n - 1].astype(float), ref["v"][n - 1].astype(float)
            el = G.fluctuation_energy(s.u, s.v, run.dx)
            ee = G.fluctuation_energy(ue, ve, run.dx)
            e_max = max(e_max, abs(el - ee) / ee)
        if ok:
            ev = G.error_velocity((s.u, s.v), 32, (ue, ve), 32)
        else:
            ev = float("inf")
        out[str(seed)] = {"e_V_vs_classical": ev, "energy_max_rel": e_max, "finite": ok}
        run.close()
    out["score"] = float(np.mean([out[str(s)]["e_V_vs_classical"] for s in seeds]))
    return out


def main(argv=None) -> int:
    import torch
    from atlas.workbench import learned_gate as G
    from atlas.workbench.learned_net import WindowUNet, n_parameters, save_window_net

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=os.path.join(OUT, "data"))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--minutes", type=float, default=75.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--rollout-every", type=int, default=5)
    ap.add_argument("--smoke", action="store_true", help="train on 1000, validate on 2000")
    args = ap.parse_args(argv)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    torch.backends.cudnn.benchmark = True

    train_seeds = [1000] if args.smoke else list(G.TRAIN_SEEDS)
    val_seeds = [2000] if args.smoke else list(G.VALIDATION_SEEDS)
    roll_seeds = (2000,) if args.smoke else ROLLOUT_SEEDS
    roll_steps = 2 if args.smoke else G.STEPS
    t_load = time.perf_counter()
    Xt, wt, rt = load(args.data, train_seeds)
    Xv, wv, rv = load(args.data, val_seeds)
    t_load = time.perf_counter() - t_load
    base_t = np.flatnonzero(rt < 0)
    pair_t = np.flatnonzero(rt >= 0)
    base_v = np.flatnonzero(rv < 0)
    print("loaded %d training rows (%d pairs), %d validation rows in %.0f s on %s"
          % (len(Xt), len(pair_t), len(Xv), t_load, dev), flush=True)

    sub = rng.choice(base_t, size=min(4000, len(base_t)), replace=False)
    sample = Xt[np.sort(sub)].astype(np.float32)
    in_scale = np.sqrt(np.mean(sample[:, :3] ** 2, axis=(0, 2, 3))) + 1e-6
    out_scale = np.sqrt(np.mean(sample[:, 3:] ** 2, axis=(0, 2, 3))) + 1e-8
    del sample
    net = WindowUNet(width=G.NET_WIDTH, depth=G.NET_DEPTH).to(dev)
    net.in_scale.copy_(torch.tensor(in_scale, dtype=torch.float32))
    net.out_scale.copy_(torch.tensor(out_scale, dtype=torch.float32))
    print("network: width %d depth %d, %d parameters; in_scale %s, out_scale %s"
          % (G.NET_WIDTH, G.NET_DEPTH, n_parameters(net), np.round(in_scale, 5),
             np.round(out_scale, 6)), flush=True)

    W = torch.tensor(chi_weights(), device=dev)                        # [K, h, w]
    inv = torch.tensor(1.0 / out_scale ** 2, dtype=torch.float32, device=dev).view(1, -1, 1, 1)

    def to_dev(rows):
        return torch.from_numpy(np.ascontiguousarray(rows)).to(dev, non_blocking=True).float()

    def wmse(pred, target, k):
        w = W[k].unsqueeze(1)                                          # [B, 1, h, w]
        return ((pred - target) ** 2 * inv * w).sum() / (w.sum() * pred.shape[1])

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    steps_per_epoch = max(1, len(base_t) // args.batch)
    total = steps_per_epoch * args.epochs
    warm = steps_per_epoch * 2

    def lr_at(i):
        if i < warm:
            return args.lr * (i + 1) / warm
        p = (i - warm) / max(1, total - warm)
        return 2e-5 + 0.5 * (args.lr - 2e-5) * (1 + math.cos(math.pi * min(1.0, p)))

    log = {"started": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "device": str(dev), "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
           "train_seeds": train_seeds, "validation_seeds": val_seeds,
           "rollout_seeds": list(roll_seeds), "rows_train": int(len(Xt)),
           "pairs_train": int(len(pair_t)), "rows_validation": int(len(Xv)),
           "batch": args.batch, "lr": args.lr, "pair_weight": PAIR_WEIGHT,
           "in_scale": in_scale.tolist(), "out_scale": out_scale.tolist(),
           "parameters": n_parameters(net), "epochs": []}
    best = {"score": float("inf"), "epoch": None}
    t0 = time.perf_counter()
    it = 0
    best_path = os.path.join(args.out, "window-net.pt")
    for epoch in range(1, args.epochs + 1):
        net.train()
        order = rng.permutation(base_t)
        losses, plosses = [], []
        for b in range(steps_per_epoch):
            rows = np.sort(order[b * args.batch:(b + 1) * args.batch])
            for g in opt.param_groups:
                g["lr"] = lr_at(it)
            xb = to_dev(Xt[rows])
            k = torch.from_numpy(wt[rows]).to(dev)
            loss = wmse(net(xb[:, :3]), xb[:, 3:], k)
            if len(pair_t):
                pr = np.sort(rng.choice(pair_t, size=max(1, args.batch // 4), replace=False))
                xp, xb0 = to_dev(Xt[pr]), to_dev(Xt[rt[pr]])
                kp = torch.from_numpy(wt[pr]).to(dev)
                both = net(torch.cat([xp[:, :3], xb0[:, :3]]))
                resp = both[:len(pr)] - both[len(pr):]
                ploss = wmse(resp, xp[:, 3:] - xb0[:, 3:], kp)
                loss = loss + PAIR_WEIGHT * ploss
                plosses.append(float(ploss))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))
            it += 1
        net.eval()
        with torch.no_grad():
            vl = []
            for b in range(0, len(base_v), 128):
                rows = base_v[b:b + 128]
                xv = to_dev(Xv[rows])
                vl.append(float(wmse(net(xv[:, :3]), xv[:, 3:],
                                     torch.from_numpy(wv[rows]).to(dev))) * len(rows))
            val = sum(vl) / max(1, len(base_v))
        row = {"epoch": epoch, "train": float(np.mean(losses)),
               "pairs": float(np.mean(plosses)) if plosses else None, "validation": val,
               "lr": lr_at(it - 1), "elapsed_s": time.perf_counter() - t0}
        if epoch == 1:
            # fit the schedule to the time budget, so the cosine decays to its floor
            # inside it rather than being cut off at a high learning rate (each epoch
            # also pays a rollout now and then: budget 15% for it)
            per = time.perf_counter() - t0
            fit = max(2, int(0.85 * args.minutes * 60 / per))
            if fit < args.epochs:
                args.epochs = fit
                total = steps_per_epoch * args.epochs
            log["epochs_planned"] = args.epochs
            print("  first epoch %.0f s: %d epochs fit the %.0f-minute budget"
                  % (per, args.epochs, args.minutes), flush=True)
        last = epoch == args.epochs or (time.perf_counter() - t0) / 60 > args.minutes
        if epoch % args.rollout_every == 0 or last:
            rs = rollout_score(net, args.data, roll_seeds, roll_steps)
            row["rollout"] = rs
            if rs["score"] < best["score"]:
                best = {"score": rs["score"], "epoch": epoch}
                save_window_net(net, best_path, epoch=epoch, score=rs["score"])
        log["epochs"].append(row)
        print("epoch %3d  train %.4e  pairs %s  validation %.4e  %s  (%.0f s)"
              % (epoch, row["train"], "%.3e" % row["pairs"] if row["pairs"] else "-", val,
                 "rollout %.4e" % row["rollout"]["score"] if "rollout" in row else "",
                 row["elapsed_s"]), flush=True)
        with open(os.path.join(args.out, "training-log.json"), "w", encoding="utf-8") as fh:
            json.dump(dict(log, best=best), fh, indent=1)
        if last:
            break
    log["best"] = best
    log["finished"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    log["weights"] = {"file": "out/learned-case/window-net.pt",
                      "sha256": hashlib.sha256(open(best_path, "rb").read()).hexdigest()
                      if os.path.isfile(best_path) else None}
    with open(os.path.join(args.out, "training-log.json"), "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=1)
    print("best epoch %s, rollout score %.4e -> %s" % (best["epoch"], best["score"], best_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
