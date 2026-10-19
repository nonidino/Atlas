"""Stage 0 (W355): price the network before training it (F9's discipline).

    python scripts/chartop_price.py [--device cuda] [--out out/chartop]

Speed does not depend on the weights, so untrained networks are timed:

1. **forward**: every registered backbone and size (U-Net 0.3/1/3 M, the spectral
   operator) at n = 32, 64, 128, batch 1 and 16 -- bfloat16 autocast on a GPU (decision
   D4), float32 on a CPU; median of 10 after 3 warm-ups, synchronised;
2. **one training step** of every arm type at its planned batch size (forward, the
   float64 objective, backward, AdamW), on examples drawn from the generator's
   own seeds -- the comparison of [[chart-operator-training-and-cost]] section 5.1's
   estimate with a measurement;
3. **the generator's rate** on this machine's CPUs (examples per second with every core).

Writes ``price-<host>-<time>.json``.  The laptop's numbers count only on AC power;
the script records the power state where it can read it.

**E5 of the registration after Stage 0** (`atlas.chartop.gates_forms`):

    python scripts/chartop_price.py --forms [--device cuda] [--out out/chartop]

prices the registered arms' forms before any long run: S's (split, corner bubble,
strips) and W's (split, Stage 0's bubble, no strips), at the registered (n, batch)
pairs, `gates_forms.E5_PRICED`.  Stage 0's arm B is timed in the same run as the
unchanged reference, so that the result is a ratio to a step already measured, not
only a wall-clock figure.  For each, on the same examples as the step pricing above:
- **the forward pass** through `forms.call`, as training and the host call it, with
  the assembly of the operators the strips need and the strips themselves included;
- **the assembly** (`forms.batch_operators`) alone, and the strips' own seconds;
- **one training step** as `train.train` takes it: D4 augmentation, the forward pass,
  the float64 objective, backward, gradient clipping, AdamW.  S's step is also timed
  with the strips skipped (``train.Arm.strips_in_training=False``: they carry no
  gradient, so the weights are the same).
The record ends with a projection: the mean step over Stage 0's training-set mix
(each resolution drawn in proportion to its examples per batch, as `train.train`
draws it), and the hours of each arm.  Writes ``price-forms-<host>-<time>.json``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
import platform
import socket
import statistics
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import chartop_cpus  # noqa: E402
from atlas.chartop import data, model as md, ports  # noqa: E402
from atlas.chartop import train as tr  # noqa: E402

BATCH = {32: 32, 64: 32, 128: 8}
ARM_TYPES = {
    "B (U-Net 1M, bubble, label-free)": dict(backbone="unet", size="1M", form="bubble"),
    "C2-sup (supervised)": dict(backbone="unet", size="1M", form="bubble", objective="supervised"),
    "C3 (direct)": dict(backbone="unet", size="1M", form="direct"),
    "C4 (spectral)": dict(backbone="spectral", size="1M", form="bubble"),
    "C5 (0.3M)": dict(backbone="unet", size="0.3M", form="bubble"),
    "C5 (3M)": dict(backbone="unet", size="3M", form="bubble"),
    "C6 (cosine)": dict(backbone="unet", size="1M", form="bubble", kind="cosine"),
    "C7 (fast-solve)": dict(backbone="unet", size="1M", form="fastsolve"),
}


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _median_time(fn, dev, warm=3, reps=10):
    for _ in range(warm):
        fn()
    _sync(dev)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync(dev)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def power_state() -> dict:
    if platform.system() != "Windows":
        return {"platform": platform.system()}
    try:
        import subprocess
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Battery | Select-Object -First 1 BatteryStatus,"
                              "EstimatedChargeRemaining | ConvertTo-Json -Compress)"],
                             capture_output=True, text=True, timeout=30).stdout
        d = json.loads(out)
        d["on_ac"] = d.get("BatteryStatus") == 2
        return d
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def forward_times(dev, ns=(32, 64, 128), batches=(1, 16)) -> list:
    rows = []
    autocast = dev.type == "cuda"
    for backbone, size in (("unet", "0.3M"), ("unet", "1M"), ("unet", "3M"), ("spectral", "1M")):
        e = md.Expert(backbone, size).to(dev).eval()
        for p in e.parameters():          # random weights: the zero-initialised heads would
            torch.nn.init.normal_(p, 0.0, 0.02)   # let subnormals or zeros flatter the timing
        for n in ns:
            for bs in batches:
                x = torch.randn(bs, 5, n + 1, n + 1, device=dev)
                s = torch.randn(bs, 2, device=dev)
                s0 = torch.rand(bs, device=dev, dtype=torch.float64) * 10

                def f():
                    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                                         enabled=autocast):
                        e(x, s, s0)
                t = _median_time(f, dev)
                rows.append({"backbone": backbone, "size": size, "parameters": md.parameters(e),
                             "n": n, "batch": bs, "s_per_call": t, "ms_per_piece": 1e3 * t / bs})
                print(f"forward {backbone:8s} {size:4s} n={n:3d} batch={bs:2d}: "
                      f"{1e3 * t / bs:8.3f} ms per piece", flush=True)
    return rows


def _examples(n, k, seed0):
    return [data.draw(seed0 + i, n=n) for i in range(k)]


def step_times(dev, ns=(32, 64, 128)) -> list:
    autocast = dev.type == "cuda"
    rows = []
    pools = {n: _examples(n, BATCH[n], data.SEEDS["train"] + 50_000 + 1000 * n) for n in ns}
    for name, cfg in ARM_TYPES.items():
        kind = cfg.get("kind", "sine-hat")
        objective = cfg.get("objective", "label-free")
        for n in ns:
            exs = pools[n]
            arrs = [data.arrays(e) for e in exs]
            bucket = {k: np.stack([a[k] for a in arrs]) for k in tr.KEYS}
            labelled = objective == "supervised"
            if labelled:
                Q = ports.basis(kind, n, 16).Q
                L = [data.labels_from_arrays(a, Q) for a in arrs]
                bucket["H"] = np.stack([l["H"].T.astype(np.float32) for l in L])
                bucket["up"] = np.stack([l["up"].astype(np.float32) for l in L])
                bucket["trace_Lambda"] = np.array([l["trace_Lambda"] for l in L])
                bucket["up_energy"] = np.array([l["up_energy"] for l in L])
            T = tr.to_device(bucket, dev, labelled)
            e = md.Expert(cfg["backbone"], cfg["size"], cfg["form"], kind).to(dev)
            for p in e.parameters():
                torch.nn.init.normal_(p, 0.0, 0.02)
            opt = torch.optim.AdamW(e.parameters(), lr=1e-4)
            idx = torch.arange(len(exs), device=dev)
            rng = np.random.default_rng(0)

            def step():
                b, lab = tr._batch(T, idx, labelled)
                k = int(rng.integers(8))
                b = md.d4(b, k)
                if lab is not None:
                    lab = md.d4_labels(lab, kind, 16, k)
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=autocast):
                    H, up = e(b["inputs"], b["scalars"], b["s0"])
                o = md.objectives(b, H, up, kind, 16, lab, objective,
                                  H0=getattr(e, "last_principal", None))
                loss = o["loss"].mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t = _median_time(step, dev, warm=3, reps=8)
            mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else None
            rows.append({"arm": name, "n": n, "batch": len(exs), "s_per_step": t,
                         "ms_per_example": 1e3 * t / len(exs), "peak_gib": mem})
            print(f"step {name:34s} n={n:3d} batch={len(exs):2d}: {t:7.4f} s "
                  f"({1e3 * t / len(exs):7.2f} ms/example)" + (f", {mem:.2f} GiB" if mem else ""),
                  flush=True)
    return rows


#: Stage 0's training set per resolution (``out/chartop/runs/B/arm.json``): the step mix
#: `train.train` draws from, each resolution in proportion to its examples per batch
STAGE0_TRAIN_EXAMPLES = {32: 15110, 64: 24901, 128: 9989}


def forms_times(dev, ns=(32, 64, 128)) -> dict:
    """E5 of `gates_forms`: S's and W's forms, and Stage 0's B as the reference."""
    from atlas.chartop import forms as fm, gates_forms as GF
    autocast = dev.type == "cuda"
    batch = dict(GF.E5_PRICED)
    c = GF.COMMON
    off = "S, strips only at validation"     # train.Arm.strips_in_training=False
    arms = {"S": GF.ARMS["S"], off: GF.ARMS["S"], "W": GF.ARMS["W"], "B (Stage 0, reference)": None}
    rows = []
    for n in ns:
        exs = _examples(n, batch[n], data.SEEDS["train"] + 50_000 + 1000 * n)
        arrs = [data.arrays(e) for e in exs]
        T = tr.to_device({k: np.stack([a[k] for a in arrs]) for k in tr.KEYS}, dev, False)
        idx = torch.arange(len(exs), device=dev)
        for label, cfg in arms.items():
            if cfg is None:
                e = fm.make("unet", "1M", "bubble", "sine-hat")
            else:
                e = fm.make(c["backbone"], c["size"], c["form"], c["kind"], c["m"], p_size=c["p_size"],
                            bubble=cfg["bubble"], strips=cfg["strips"])
            e = e.to(dev)
            for p in e.parameters():      # random weights, as above: zero heads could flatter
                torch.nn.init.normal_(p, 0.0, 0.02)
            strips = getattr(e, "needs_operator", False)
            solve = label != off                # the training step solves the strips
            b0, _ = tr._batch(T, idx, False)
            row = {"arm": label, "n": n, "batch": len(exs), "parameters": md.parameters(e),
                   "config": e.config() if hasattr(e, "config") else {"form": "bubble"}}
            strip_s = []

            def fwd():
                e.eval()
                with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                                     enabled=autocast):
                    fm.call(e, b0)
                if strips:
                    strip_s.append(e.last_strip_seconds)
            row["forward_s"] = _median_time(fwd, dev)
            if strips:
                row["forward_strip_s"] = statistics.median(strip_s[-10:])
                row["assembly_s"] = _median_time(lambda: fm.batch_operators(b0), dev)
            e.train()
            opt = torch.optim.AdamW(e.parameters(), lr=1e-3, weight_decay=1e-5)
            rng = np.random.default_rng(0)
            strip_s.clear()

            def step():
                b, _lab = tr._batch(T, idx, False)
                b = md.d4(b, int(rng.integers(8)))
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=autocast):
                    H, up = fm.call(e, b, strip_solve=solve)
                if strips and solve:
                    strip_s.append(e.last_strip_seconds)
                o = md.objectives(b, H, up, "sine-hat", 16, None, "label-free",
                                  H0=getattr(e, "last_principal", None))
                loss = o["loss"].mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(e.parameters(), 1.0)
                opt.step()
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            row["step_s"] = _median_time(step, dev, warm=3, reps=8)
            row["step_solves_strips"] = bool(strips and solve)
            if strips and solve:
                row["step_strip_s"] = statistics.median(strip_s[-8:])
            row["peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else None
            rows.append(row)
            print(f"forms {label:24s} n={n:3d} batch={len(exs):2d}: forward {row['forward_s']:.4f} s, "
                  f"step {row['step_s']:.4f} s"
                  + (f" (strips {row['step_strip_s']:.4f} s, assembly {row['assembly_s']:.4f} s)"
                     if strips and solve else ""), flush=True)
    # the projection: the mean step over Stage 0's mix, and each arm's hours
    w = {n: STAGE0_TRAIN_EXAMPLES[n] / batch[n] for n in ns}
    tot = sum(w.values())
    w = {n: v / tot for n, v in w.items()}
    proj = {"step_mix": {str(n): w[n] for n in ns}, "arms": {}}
    for label in arms:
        mean = sum(w[r["n"]] * r["step_s"] for r in rows if r["arm"] == label)
        proj["arms"][label] = {"mean_step_s": mean}
    ref = proj["arms"]["B (Stage 0, reference)"]["mean_step_s"]
    for label, cfg in arms.items():
        p = proj["arms"][label]
        p["over_B"] = p["mean_step_s"] / ref
        if cfg is not None:
            runs = len(cfg["seeds"])
            p["registered_steps"] = cfg["steps"] * runs
            p["hours"] = p["mean_step_s"] * cfg["steps"] * runs / 3600.0
            if label in ("S", off):   # S' runs S's form at the comparison budget, two seeds
                sp = GF.ARMS["Sp"]
                p["Sp_hours"] = p["mean_step_s"] * sp["steps"] * len(sp["seeds"]) / 3600.0
    proj["training_hours"] = (proj["arms"]["S"]["hours"] + proj["arms"]["S"]["Sp_hours"]
                              + proj["arms"]["W"]["hours"])
    proj["training_hours_strips_only_at_validation"] = (
        proj["arms"][off]["hours"] + proj["arms"][off]["Sp_hours"] + proj["arms"]["W"]["hours"])
    print("projection", json.dumps(proj), flush=True)
    return {"rows": rows, "projection": proj}


def _draw(seed):
    t0 = time.perf_counter()
    data.draw(seed)
    return time.perf_counter() - t0


def generator_rate(workers: int, count: int) -> dict:
    seeds = [data.SEEDS["train"] + 60_000 + i for i in range(count)]
    t0 = time.perf_counter()
    # one BLAS/OpenMP thread per worker, set before the workers import numpy:
    # 63 workers x 64 OpenBLAS threads each failed to start on a 64-CPU box
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    with mp.get_context("spawn").Pool(workers) as pool:
        per = pool.map(_draw, seeds, chunksize=2)
    wall = time.perf_counter() - t0
    return {"workers": workers, "count": count, "wall_s": wall, "examples_per_s": count / wall,
            "per_example_s_mean": float(np.mean(per))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "chartop"))
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--skip", default="", help="comma list of forward,step,generator")
    ap.add_argument("--gen-count", type=int, default=96)
    ap.add_argument("--quick", action="store_true", help="a smoke run: n = 32 and 64 only")
    ap.add_argument("--forms", action="store_true",
                    help="E5 of gates_forms: the registered arms' forms (and the generator), not Stage 0's")
    args = ap.parse_args()
    dev = torch.device(args.device)
    if args.threads:
        torch.set_num_threads(args.threads)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    skip = set(filter(None, args.skip.split(",")))
    rec = {"written": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "host": socket.gethostname(), "platform": platform.platform(),
           "python": platform.python_version(), "torch": torch.__version__,
           "numpy": np.__version__, "device": str(dev),
           "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
           "cpus": os.cpu_count(), "cpu_sources": chartop_cpus.sources(),
           "torch_threads": torch.get_num_threads(),
           "power": power_state(), "weights": "untrained (random); cost only"}
    if args.forms:
        from atlas.chartop import gates_forms as GF
        rec["registration"] = GF.REGISTERED
        rec["forms"] = forms_times(dev, (32,) if args.quick else tuple(n for n, _b in GF.E5_PRICED))
    if "forward" not in skip and not args.forms:
        rec["forward"] = forward_times(dev, (64,) if args.quick else (32, 64, 128),
                                       (16,) if args.quick else (1, 16))
    if "step" not in skip and not args.forms:
        rec["step"] = step_times(dev, (32,) if args.quick else (32, 64, 128))
    if "generator" not in skip:
        rec["generator"] = generator_rate(chartop_cpus.default_workers(1), args.gen_count)
        print("generator", rec["generator"], flush=True)
    os.makedirs(args.out, exist_ok=True)
    name = "price-%s%s-%s.json" % ("forms-" if args.forms else "",
                                   socket.gethostname().split(".")[0][:24],
                                   dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    with open(os.path.join(args.out, name), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print("wrote", os.path.join(args.out, name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
