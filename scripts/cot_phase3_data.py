"""cot_phase3_data -- the coolant plan's dataset: drawn, labelled and written as shards
with a manifest (Phase 3; [[chart-operator-stage-0b-and-1-plan]] sections 6 and 15).

Example i of a split is `transport_data.draw` (base + i): the piece's chart, Fourier number
and source from the pinned CO-D generator, its flow and Peclet number from its own stream,
its resolution from `transport_data.resolution` (0.25 / 0.35 / 0.40 at n = 32 / 64 / 128,
the owner's amendment).  What each example stores is `transport_data.arrays` (float32 chart
nodes, stream function and source, with the scalars) and its labels, computed from those
float32 arrays read back in float64 (`transport_data.labels`): the Petrov-Galerkin port
matrix ``Lam`` (64 x 64) and the condensed load ``chi`` (64); with ``--full`` (the
validation split), the trial and adjoint modes and the particular field as well.

**Training does not read the labels**: its loss is label-free (`transport_torch.objective`),
as CO-D's was.  The labels are for the validation split's errors, check 2 and the
records.

Shards hold up to ``--shard`` examples of one resolution, in seed order:
``<split>_n<res>_<k>.npz``.  ``manifest-<split>.json`` records the seeds, the generator's
source hashes, the counts and coverage, rejections, timings and every shard's content
sha256 (`chartop_registry.npz_content_sha256`, blind to the zip's timestamps): the
definition of the data, since floating-point outputs differ in the last bit between
platforms (the data is moved to a box as these files, never drawn again there).  The
manifest is also copied beside the records, ``out/chartop/cot/data-manifest-<split>.json``.

    python scripts/cot_phase3_data.py --estimate 60
    python scripts/cot_phase3_data.py --split train --count 50000 --workers 12
    python scripts/cot_phase3_data.py --split val --count 600 --full --workers 12
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import io
import json
import multiprocessing as mp
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

import chartop_registry as registry  # noqa: E402

OUT = os.path.join(ROOT, "out", "chartop", "data", "cot")
RECORDS = os.path.join(ROOT, "out", "chartop", "cot")
KIND_CODE = {"solid": 0, "through": 1, "turn": 2, "wavy": 3}
PROFILE_CODE = {"laminar": 0, "plug": 1, "blend": 2}
FAMILY_CODE = {"spline-quad": 0, "conformal-map": 1}
#: the modules whose source defines the data
SOURCES = ("atlas/chartop/transport_data.py", "atlas/chartop/data.py", "atlas/chartop/chart.py",
           "atlas/chartop/certificate.py", "atlas/chartop/transport.py", "atlas/chartop/ports.py",
           "atlas/chartop/fe.py", "atlas/chartop/piece.py", "scripts/cot_phase3_data.py")


def source_hashes() -> dict:
    out = {}
    for rel in SOURCES:
        with open(os.path.join(ROOT, rel), "rb") as fh:
            out[rel] = hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()
    return out


def _worker_init():
    import winqos
    winqos.opt_out()


def one(args):
    """(seed, full) -> the example's stored record, as plain arrays."""
    seed, full = args
    from atlas.chartop import transport_data as td
    t0 = time.perf_counter()
    p, meta = td.draw(int(seed))
    t1 = time.perf_counter()
    a = td.arrays(p, meta)
    lab = td.labels(td.piece_from_arrays(a), full=full)
    t2 = time.perf_counter()
    rec = {"X": a["X"], "psi": a["psi"], "g": a["g"], "n": int(a["n"]), "fo": float(a["fo"]),
           "pe": float(a["pe"]), "seed": int(a["seed"]), "kind": KIND_CODE[meta["kind"]],
           "profile": PROFILE_CODE[meta["profile"]], "family": FAMILY_CODE.get(meta["family"], -1),
           "rotation": int(meta["rotation"]), "Lam": lab["Lam"], "chi": lab["chi"],
           "draw_s": t1 - t0, "label_s": t2 - t1}
    if full:
        rec.update(H=lab["H"].astype(np.float32), G=lab["G"].astype(np.float32), up=lab["up"].astype(np.float32))
    return rec


def _stack(recs: list, full: bool) -> dict:
    keys = ["X", "psi", "g", "n", "fo", "pe", "seed", "kind", "profile", "family", "rotation", "Lam", "chi"]
    if full:
        keys += ["H", "G", "up"]
    out = {}
    for k in keys:
        v = [r[k] for r in recs]
        if k in ("n", "kind", "profile", "family", "rotation"):
            out[k] = np.asarray(v, np.int32)
        elif k == "seed":
            out[k] = np.asarray(v, np.int64)
        elif k in ("fo", "pe"):
            out[k] = np.asarray(v, np.float64)
        else:
            out[k] = np.stack(v)
    return out


def _write_npz(path: str, arrays: dict) -> None:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    registry.write_atomic(path, buf.getvalue())


def coverage(pe, fo, n, kind, profile, family) -> dict:
    """What check 1 reads: counts by resolution, kind, profile and family; the Peclet
    number in tenths of each resolution's cap; the Fourier number's decades and the steady
    share."""
    from atlas.chartop import transport_data as td
    pe, fo, n = np.asarray(pe), np.asarray(fo), np.asarray(n)
    out = {"count": int(len(pe)),
           "by_n": {str(k): int((n == k).sum()) for k, _ in td.RESOLUTIONS},
           "by_kind": {k: int((np.asarray(kind) == c).sum()) for k, c in KIND_CODE.items()},
           "by_profile_flowing": {k: int(((np.asarray(profile) == c) & (np.asarray(kind) != 0)).sum())
                                  for k, c in PROFILE_CODE.items()},
           "by_family": {k: int((np.asarray(family) == c).sum()) for k, c in FAMILY_CODE.items()},
           "steady": int(np.isnan(fo).sum()),
           "fo_decades": {str(d): int(((np.log10(fo[~np.isnan(fo)]) >= d) & (np.log10(fo[~np.isnan(fo)]) < d + 1)).sum())
                          for d in (-2, -1, 0, 1)}}
    pe_tenths = {}
    for k, _ in td.RESOLUTIONS:
        sel = (n == k) & (np.asarray(kind) != 0)
        cap = td.pe_cap(k)
        h = np.histogram(pe[sel] / cap, bins=10, range=(0.0, 1.0 + 1e-12))[0]
        pe_tenths[str(k)] = [int(x) for x in h]
    out["pe_tenths_of_cap_flowing"] = pe_tenths
    return out


def build(split: str, count: int, full: bool, workers: int, shard: int, out_dir: str, records: str) -> dict:
    from atlas.chartop import transport_data as td
    base = td.SEED_BASE[split]
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()
    buf = collections.defaultdict(list)
    files, k_of = [], collections.Counter()
    meta = {"pe": [], "fo": [], "n": [], "kind": [], "profile": [], "family": []}
    secs = {"draw": 0.0, "label": 0.0}

    def flush(nres):
        recs = buf.pop(nres)
        fn = f"{split}_n{nres}_{k_of[nres]:04d}.npz"
        k_of[nres] += 1
        path = os.path.join(out_dir, fn)
        _write_npz(path, _stack(recs, full))
        files.append(fn)

    seeds = [(base + i, full) for i in range(count)]
    ctx = mp.get_context("spawn")
    import winqos
    winqos.opt_out()
    with ctx.Pool(workers, initializer=_worker_init) as pool:
        for k, rec in enumerate(pool.imap(one, seeds, chunksize=8)):
            buf[rec["n"]].append(rec)
            for key in meta:
                meta[key].append(rec[key])
            secs["draw"] += rec["draw_s"]
            secs["label"] += rec["label_s"]
            if len(buf[rec["n"]]) >= shard:
                flush(rec["n"])
            if (k + 1) % 2000 == 0:
                print(f"{split}: {k + 1} of {count} ({time.perf_counter() - t0:.0f} s)", flush=True)
    for nres in sorted(list(buf)):
        flush(nres)
    files.sort()
    paths = [os.path.join(out_dir, f) for f in files]
    shard_sha = {f: registry.npz_content_sha256([p]) for f, p in zip(files, paths)}
    man = {"what": "the coolant plan's dataset (CO-T pieces and compact labels)", "split": split,
           "seeds": [base, base + count - 1], "count": count, "full_fields": full,
           "labels": "Lam (W^T K V, 64 x 64) and chi (64) from the stored float32 arrays read back in float64"
                     + ("; H, G, up (float32)" if full else ""),
           "written": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "platform": sys.platform, "numpy": np.__version__, "workers": workers,
           "sources_sha256": source_hashes(), "wall_s": time.perf_counter() - t0,
           "cpu_s": secs, "coverage": coverage(**meta),
           "files": files, "files_content_sha256": shard_sha,
           "content_sha256": registry.npz_content_sha256(paths),
           "bytes": int(sum(os.path.getsize(p) for p in paths))}
    text = json.dumps(man, indent=1).encode("utf-8")
    registry.write_atomic(os.path.join(out_dir, f"manifest-{split}.json"), text)
    os.makedirs(records, exist_ok=True)
    registry.write_atomic(os.path.join(records, f"data-manifest-{split}.json"), text)
    return man


def estimate(sample: int) -> dict:
    """Bytes and label seconds per example on ``sample`` training seeds, and the
    extrapolation to the planned sizes (an estimate until the set is built)."""
    from atlas.chartop import transport_data as td
    base = td.SEED_BASE["train"]
    rows = [one((base + i, False)) for i in range(sample)]
    out = {"sample": sample, "by_n": {}}
    for nres, _p in td.RESOLUTIONS:
        sel = [r for r in rows if r["n"] == nres]
        if not sel:
            continue
        st = _stack(sel, False)
        b = io.BytesIO()
        np.savez(b, **st)
        out["by_n"][str(nres)] = {"count": len(sel), "bytes_per_example": b.tell() / len(sel),
                                  "draw_s": float(np.mean([r["draw_s"] for r in sel])),
                                  "label_s": float(np.mean([r["label_s"] for r in sel]))}
    mix = dict(td.RESOLUTIONS)
    per = sum(mix[int(k)] * v["bytes_per_example"] for k, v in out["by_n"].items())
    cpu = sum(mix[int(k)] * (v["draw_s"] + v["label_s"]) for k, v in out["by_n"].items())
    out["mix_bytes_per_example"] = per
    out["mix_cpu_s_per_example"] = cpu
    out["train_50000"] = {"GB": 50000 * per / 1e9, "cpu_h": 50000 * cpu / 3600}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", choices=("train", "val"))
    ap.add_argument("--count", type=int)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--shard", type=int, default=2000)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--records", default=RECORDS)
    ap.add_argument("--estimate", type=int, default=0)
    a = ap.parse_args(argv)
    if a.estimate:
        print(json.dumps(estimate(a.estimate), indent=1))
        return 0
    if not a.split or not a.count:
        ap.error("--split and --count, or --estimate")
    man = build(a.split, a.count, a.full, a.workers, a.shard, a.out, a.records)
    print(f"{a.split}: {man['count']} examples, {man['bytes'] / 1e9:.2f} GB in {len(man['files'])} shards, "
          f"{man['wall_s'] / 60:.0f} min; content sha256 {man['content_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
