"""Demo step 8: the learned case's training data, from the vault's own classical solver.

    python scripts/learned_data.py --set train --workers 30      # on the rented box
    python scripts/learned_data.py --seeds 1000 --steps 2         # a smoke test

For each registered layout (`learned_gate.TRAIN_SEEDS`, `VALIDATION_SEEDS`; never
the held-out one, which `sample_layout` refuses and this script checks again),
march the classical decomposition from the uniform freestream for 40 macro-steps,
exactly as the workbench's ``serial`` arm does (`learned_arms.FarmRun`, which is
`WindFarmRun` with each rotor's induction), and keep, every macro-step and for
every window:

  x = (u - U, v, f)      the window as the classical window step receives it
  y = (du, dv)           what the classical exposed window does to it in one
                         macro-step, before the blend and the projection

which is exactly the map the network replaces.  **Ring pairs** (the plan's Jacobian
term, S5): every fourth macro-step, each window is also stepped with its held
edge ring perturbed by a smooth random profile (2% of U, three sine modes per
edge), at the same sub-step count, so training can match the window's RESPONSE
to its neighbours' data and not only its value.

Writes per layout ``<out>/traj-<seed>.npy`` (float16, ``[N, 5, h, w]``: u-U, v, f,
du, dv), ``traj-<seed>.json`` (each row's macro-step, window and pair reference)
and, for validation layouts, ``traj-<seed>-global.npz`` (the classical march's
global velocity and farm power at every macro-step, for rollout validation).
``manifest.json`` lists every file with its sha256.  Each layout is one process;
a finished layout is skipped on a rerun, so an interrupted run resumes.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one core per layout; the layouts run in parallel
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse                                                         # noqa: E402
import hashlib                                                          # noqa: E402
import json                                                             # noqa: E402
import sys                                                              # noqa: E402
import time                                                             # noqa: E402

import numpy as np                                                      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PAIR_EVERY = 4
PAIR_AMP = 0.02
PAIR_MODES = 3


def ring_perturbation(rng, h: int, w: int) -> np.ndarray:
    """A smooth random profile on the window's held edge ring, zero elsewhere."""
    d = np.zeros((h, w))
    for edge in range(4):
        n = w if edge < 2 else h
        s = (np.arange(n) + 0.5) / n
        c = rng.normal(size=PAIR_MODES) / np.arange(1, PAIR_MODES + 1)
        prof = PAIR_AMP * sum(c[m] * np.sin((m + 1) * np.pi * s) for m in range(PAIR_MODES))
        if edge == 0:
            d[0, :] = prof
        elif edge == 1:
            d[-1, :] = prof
        elif edge == 2:
            d[:, 0] = prof
        else:
            d[:, -1] = prof
    return d


def generate(seed: int, out: str, steps: int, keep_global: bool) -> dict:
    from atlas.workbench import learned_arms as LA
    from atlas.workbench import learned_gate as G
    from atlas.workbench.families import windfarm as wf

    lay = G.sample_layout(seed)
    if not G.held_out_ok(lay.rotors):                 # sample_layout guarantees it
        raise SystemExit("layout %d is too close to the held-out one" % seed)
    spec, induction = LA.layout_spec(lay)
    run = LA.FarmRun(spec, arms=("serial",), threads=1, induction=induction)
    groups = list(run.groups.items())
    if len(groups) != 1:
        raise SystemExit("the learned case's tiling is one window shape")
    shape, idx = groups[0]
    sol = run.serial[shape]
    rng = np.random.default_rng(10_000 + seed)
    t = run.tiling
    rows_x, rows_y, meta = [], [], []
    g_u, g_v, g_p = [], [], []
    s = run.initial("serial")
    t0 = time.perf_counter()
    for n in range(1, steps + 1):
        fx, _rec = run.forcing(s.u)
        x = run.window_inputs(s, fx)                     # [K, 3, h, w]
        us, vs, fs = t.cut(s.u, idx), t.cut(s.v, idx), t.cut(fx, idx)
        umax = float(np.max(np.hypot(us, vs)))
        nxt = run.step("serial", s)                      # the workbench's own step
        out_u, out_v = nxt.locals_
        base0 = len(meta)
        for k in range(t.n_windows):
            rows_x.append(x[k])
            rows_y.append(np.stack([out_u[k] - us[k], out_v[k] - vs[k]]))
            meta.append((n, k, -1))
        if n % PAIR_EVERY == 1:
            du = np.stack([ring_perturbation(rng, *shape) for _ in idx])
            dv = np.stack([ring_perturbation(rng, *shape) for _ in idx])
            sol.b = wf._Umax(getattr(sol.b, "_inner", sol.b), umax)
            a, b = sol.step_batch((us + du).copy(), (vs + dv).copy(), run.dt, bc0=None,
                                  force=(fs, np.zeros_like(fs)))
            for k in range(t.n_windows):
                xp = x[k].copy()
                xp[0] += du[k]
                xp[1] += dv[k]
                rows_x.append(xp)
                rows_y.append(np.stack([a[k] - (us[k] + du[k]), b[k] - (vs[k] + dv[k])]))
                meta.append((n, k, base0 + k))
        if keep_global:
            g_u.append(nxt.u.astype(np.float32))
            g_v.append(nxt.v.astype(np.float32))
            g_p.append(run.power(nxt.rec))
        s = nxt
    arr = np.concatenate([np.stack(rows_x), np.stack(rows_y)], axis=1).astype(np.float16)
    if not np.all(np.isfinite(arr)):
        raise SystemExit("layout %d produced a non-finite sample" % seed)
    files = {}
    path = os.path.join(out, "traj-%d.npy" % seed)
    np.save(path + ".tmp.npy", arr)
    os.replace(path + ".tmp.npy", path)
    files[os.path.basename(path)] = path
    mpath = os.path.join(out, "traj-%d.json" % seed)
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump({"seed": seed, "rotors": lay.rotors, "steps": steps,
                   "rows": meta, "pair_every": PAIR_EVERY, "pair_amp": PAIR_AMP,
                   "seconds": time.perf_counter() - t0}, fh)
    files[os.path.basename(mpath)] = mpath
    if keep_global:
        gpath = os.path.join(out, "traj-%d-global.npz" % seed)
        np.savez(gpath[:-4] + ".tmp.npz", u=np.stack(g_u), v=np.stack(g_v), power=np.array(g_p))
        os.replace(gpath[:-4] + ".tmp.npz", gpath)
        files[os.path.basename(gpath)] = gpath
    return {"seed": seed, "rows": int(arr.shape[0]), "seconds": time.perf_counter() - t0,
            "sha256": {k: hashlib.sha256(open(p, "rb").read()).hexdigest()
                       for k, p in files.items()}}


def _job(a):
    seed, out, steps, keep_global = a
    done = os.path.join(out, "traj-%d.done" % seed)
    if os.path.isfile(done):
        return json.load(open(done))
    r = generate(seed, out, steps, keep_global)
    with open(done, "w") as fh:
        json.dump(r, fh)
    return r


def main(argv=None) -> int:
    from atlas.workbench import learned_gate as G
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", choices=("train", "validation", "all"), default=None)
    ap.add_argument("--seeds", default=None, help="comma-separated seeds instead of a set")
    ap.add_argument("--steps", type=int, default=G.STEPS)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "learned-case", "data"))
    args = ap.parse_args(argv)
    if args.seeds:
        seeds = [int(v) for v in args.seeds.split(",")]
    else:
        seeds = (list(G.TRAIN_SEEDS) if args.set in ("train", "all") else []) + \
                (list(G.VALIDATION_SEEDS) if args.set in ("validation", "all") else [])
    os.makedirs(args.out, exist_ok=True)
    jobs = [(s, args.out, args.steps, s in G.VALIDATION_SEEDS) for s in seeds]
    t0 = time.perf_counter()
    results = []
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for r in pool.imap_unordered(_job, jobs):
                results.append(r)
                print("  layout %d: %d rows in %.0f s  (%d of %d, %.0f s)"
                      % (r["seed"], r["rows"], r["seconds"], len(results), len(jobs),
                         time.perf_counter() - t0), flush=True)
    else:
        for j in jobs:
            r = _job(j)
            results.append(r)
            print("  layout %d: %d rows in %.0f s" % (r["seed"], r["rows"], r["seconds"]),
                  flush=True)
    man = os.path.join(args.out, "manifest.json")
    old = json.load(open(man)) if os.path.isfile(man) else {}
    for r in results:
        old[str(r["seed"])] = r
    with open(man, "w", encoding="utf-8") as fh:
        json.dump(old, fh, indent=1)
    print("  %d layouts, %d rows, %.0f s" % (len(results), sum(r["rows"] for r in results),
                                              time.perf_counter() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
