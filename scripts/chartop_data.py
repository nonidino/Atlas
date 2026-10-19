"""Stage 0 data: draw a split's examples from their registered seeds, in parallel, and
write shards (`atlas.chartop.data`; W355).

    python scripts/chartop_data.py --split train --count 50000 --workers 16 --out DIR
    python scripts/chartop_data.py --split val --count 600 --labels sine-hat,cosine --out DIR
    python scripts/chartop_data.py --split train --count 12000 --name train-c2 --labels sine-hat --out DIR

Splits and their seeds (`data.SEEDS`, registered in `atlas/chartop/gates.py`): ``train``
(general), ``train-conformal`` (comparison C1: the conformal-map family, modulus in
[0.8, 1.25]), ``val`` (labelled; picks the weights), ``test`` (labelled; held out, drawn
only at the one evaluation).  Each shard holds up to ``--shard`` examples of one
resolution: ``<split>_n<res>_<k>.npz`` with the arrays of `data.arrays` and the bins.
``--labels`` adds ``labels[-<basis>]_<split>_n<res>_<k>.npz`` with the exact modes of
the stored problem.  ``--name`` writes the seeds of ``--split`` under another name: comparison
C2's two arms train on ``train-c2``, the first examples of ``train`` drawn again with labels.  ``manifest-<split>.json`` records the seeds, the generator's
source hash, counts, timings, rejections and coverage -- the definition of the data,
since floating-point outputs differ in the last bit between platforms.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

import chartop_cpus  # noqa: E402
from atlas.chartop.gates import SEED_BASE as SPLIT_SEED  # noqa: E402  (registered)


def _draw_one(args):
    seed, split, label_kinds = args
    from atlas.chartop import data, ports
    t0 = time.perf_counter()
    if split == "train-conformal":
        ex = data.draw(seed, family="conformal-map", M_range=(0.8, 1.25))
    else:
        ex = data.draw(seed)
    a = data.arrays(ex)
    meta = {"seed": seed, "sup_mu": ex.envelope["sup_abs_mu"], "logJ_range": ex.envelope["logJ_range"],
            "s_grad": ex.envelope["s_grad"], "fo": -1.0 if ex.fo is None else ex.fo,
            "freq": ex.freq, "family": ex.family, "method": ex.method,
            "rejected": ex.meta.get("rejected", 0), "modulus": float(ex.meta.get("modulus") or 0.0)}
    labs = {}
    for kind in label_kinds:
        Q = ports.basis(kind, ex.n, 16).Q
        L = data.labels_from_arrays(a, Q)
        labs[kind] = {"H": L["H"].T.astype(np.float32), "up": L["up"].astype(np.float32),
                      "trace_Lambda": L["trace_Lambda"], "up_energy": L["up_energy"]}
    return ex.n, a, meta, labs, time.perf_counter() - t0


def source_hash() -> str:
    h = hashlib.sha256()
    for f in ("data.py", "chart.py", "certificate.py", "piece.py", "fe.py", "ports.py"):
        with open(os.path.join(ROOT, "atlas", "chartop", f), "rb") as fh:
            h.update(fh.read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def _write(out, name, arrays):
    tmp = os.path.join(out, name + ".tmp.npz")
    np.savez(tmp, **arrays)
    final = os.path.join(out, name)
    for k in range(20):
        try:
            os.replace(tmp, final)
            return
        except OSError:
            time.sleep(0.5 * (k + 1))
    raise OSError("could not write " + final)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, choices=sorted(SPLIT_SEED))
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=chartop_cpus.default_workers(2),
                    help="pool size; the default is the CPUs this process may use less 2 (the "
                         "container's CPU quota counts, not os.cpu_count(): `chartop_cpus.py`)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=1000)
    ap.add_argument("--labels", default="", help="comma-separated port bases to label")
    ap.add_argument("--name", default=None, help="the written split's name (default --split)")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    os.makedirs(args.out, exist_ok=True)
    kinds = [k for k in args.labels.split(",") if k]
    outname = args.name or args.split
    base = SPLIT_SEED[args.split]
    seeds = [base + i for i in range(args.start, args.start + args.count)]
    t0 = time.perf_counter()
    buckets = collections.defaultdict(list)
    shard_no = collections.Counter()
    times, rej, cover = [], 0, collections.Counter()

    def flush(n, final=False):
        rows = buckets[n]
        while rows and (len(rows) >= args.shard or final):
            chunk, rows[:] = rows[:args.shard], rows[args.shard:]
            k = shard_no[n]
            shard_no[n] += 1
            name = f"{outname}_n{n}_{args.start // args.shard + k:04d}.npz"
            if True:
                arrs = {key: np.stack([c[1][key] for c in chunk]) for key in chunk[0][1]}
                for key in chunk[0][2]:
                    arrs[key] = np.array([c[2][key] for c in chunk])
                _write(args.out, name, arrs)
            for kind in kinds:
                lab = {key: np.stack([c[3][kind][key] for c in chunk]) for key in ("H", "up")}
                lab["trace_Lambda"] = np.array([c[3][kind]["trace_Lambda"] for c in chunk])
                lab["up_energy"] = np.array([c[3][kind]["up_energy"] for c in chunk])
                lab["lseed"] = np.array([c[2]["seed"] for c in chunk])
                pre = "labels_" if kind == "sine-hat" else f"labels-{kind}_"
                _write(args.out, pre + name, lab)
            if not final:
                break
    # one BLAS/OpenMP thread per worker, set before the workers import numpy:
    # 63 workers x 64 OpenBLAS threads each failed to start on a 64-CPU box
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        done = 0
        for n, a, meta, labs, dt in pool.imap(_draw_one, [(s, args.split, kinds) for s in seeds],
                                              chunksize=8):
            buckets[n].append((n, a, meta, labs))
            times.append(dt)
            rej += meta["rejected"]
            cover[(n, min(3, int(meta["sup_mu"] / 0.15)), meta["freq"], meta["family"])] += 1
            done += 1
            if len(buckets[n]) >= args.shard:
                flush(n)
            if done % 500 == 0:
                print(f"{done}/{len(seeds)} examples, {time.perf_counter() - t0:.0f} s", flush=True)
    for n in list(buckets):
        flush(n, final=True)
    man = {"split": args.split, "name": outname, "seeds": [seeds[0], seeds[-1]], "count": len(seeds),
           "source_sha256": source_hash(), "labels": kinds,
           "elapsed_s": time.perf_counter() - t0, "workers": args.workers,
           "cpu_sources": chartop_cpus.sources(),
           "per_example_s_median": float(np.median(times)), "per_example_s_mean": float(np.mean(times)),
           "rejected_charts": rej,
           "coverage": {f"n={k[0]} mu_bin={k[1]} freq={k[2]} family={k[3]}": v
                        for k, v in sorted(cover.items())}}
    with open(os.path.join(args.out, f"manifest-{outname}-{args.start}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    print(json.dumps({k: man[k] for k in ("split", "count", "elapsed_s", "per_example_s_mean",
                                          "rejected_charts")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
