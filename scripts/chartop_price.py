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

import numpy as np  # noqa: E402
import torch  # noqa: E402

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
           "cpus": os.cpu_count(), "torch_threads": torch.get_num_threads(),
           "power": power_state(), "weights": "untrained (random); cost only"}
    if "forward" not in skip:
        rec["forward"] = forward_times(dev, (64,) if args.quick else (32, 64, 128),
                                       (16,) if args.quick else (1, 16))
    if "step" not in skip:
        rec["step"] = step_times(dev, (32,) if args.quick else (32, 64, 128))
    if "generator" not in skip:
        rec["generator"] = generator_rate(max(1, (os.cpu_count() or 2) - 1), args.gen_count)
        print("generator", rec["generator"], flush=True)
    os.makedirs(args.out, exist_ok=True)
    name = "price-%s-%s.json" % (socket.gethostname().split(".")[0][:24],
                                 dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    with open(os.path.join(args.out, name), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print("wrote", os.path.join(args.out, name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
