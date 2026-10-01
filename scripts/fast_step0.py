"""Demo item 1.4, step 0: the micro-benchmarks that decide which fast examples to build.

    python scripts/fast_step0.py [--wait-ac=MIN]   # -> out/workbench/records/fast/step0-<time>.json
    python scripts/fast_step0.py --quick           # smaller sizes, a smoke test (not a result)

Not on AC power, the full run waits up to ``--wait-ac`` minutes, then exits without timing.

`demo-fast-examples-plan` §4, step 0.  Three questions, each with its rule
REGISTERED HERE, before the first run (2026-09-30), and never loosened:

1. **Does P scale?**  Two explicit kernels over 12 independent windows, on
   T = 1, 2, 4, 8, 12, 16 threads, at four window sizes: a numpy five-point update
   (the leapfrog's shape) and a scipy CSR mat-vec step (the river's window step,
   `families/plume.py`).  Strong scaling: the same windows and steps at every T;
   speed-up = t(T=1) / t(T); the threaded answer must equal the serial one bit for
   bit.
   **Rule:** P is worth building for a family only where its kernel's speed-up
   reaches 3 at some T (the bar of `demo-fast-examples-plan` §1 cannot be met by
   threads that do not reach it themselves).
2. **Does SuperLU release the GIL?**  scipy `splu` of 12 independent five-point
   Laplacians, then 12 solves, on T threads.  **Rule:** it does if the factor's
   speed-up at T = 4 is at least 2; if it stays under 1.3, threads are serialised
   and style S needs processes (whose overhead is then timed too).
3. **Where does each core leave cache?**  The undivided solve's cost per cell
   against size: the five-point implicit factor + solve (conduction, the plate, the
   cooled block), the explicit update (the river, the sound), and the Q1
   plane-stress stiffness factor + solve on `fe.py`'s own assembly (the structure).
   **Rule:** the cliff is the first size whose cost per cell is at least 1.5 times
   the smallest size's (for the explicit kernels, whose cost should be flat), and
   the growth exponent of the direct solves is fitted on log-log for reference
   (nested dissection on a 2-D grid is $O(N^{3/2})$ to factor).

Each timing is the median of ``REPEATS`` after one warm-up.  The machine's state
(power source, other Python processes, the CPU) is recorded before and after; a run
on battery is recorded as such and is not a result.
"""

from __future__ import annotations

import concurrent.futures as cf
import datetime
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np                                              # noqa: E402
import scipy.sparse as sp                                       # noqa: E402
import scipy.sparse.linalg as spla                              # noqa: E402

from atlas.workbench import fe, machine                         # noqa: E402

REPEATS = 5
THREADS = (1, 2, 4, 8, 12, 16)
WINDOWS = 12
#: registered 2026-09-30, before the first run (see the docstring)
RULES = {
    "p_scales": "speed-up >= 3 at some T",
    "gil_released": "splu factor speed-up at T=4 >= 2; serialised if < 1.3",
    "cache_cliff": "first size with cost per cell >= 1.5 x the smallest size's",
}


def cpu_facts() -> dict:
    cmd = ("Get-CimInstance Win32_Processor | ForEach-Object { \"$($_.Name)`t"
           "$($_.NumberOfCores)`t$($_.NumberOfLogicalProcessors)`t$($_.MaxClockSpeed)`t"
           "$($_.L2CacheSize)`t$($_.L3CacheSize)\" }")
    try:
        txt = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        name, cores, logical, mhz, l2, l3 = txt.split("\t")
        return {"name": name.strip(), "cores": int(cores), "logical": int(logical),
                "max_mhz": int(mhz), "l2_kb": int(l2), "l3_kb": int(l3)}
    except Exception as exc:                                      # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def median_time(fn, repeats: int = REPEATS) -> float:
    fn()                                                          # warm-up
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


# -- the kernels -------------------------------------------------------------


def explicit_steps(u: np.ndarray, steps: int, c: float = 0.2) -> np.ndarray:
    """Five-point explicit diffusion, in place, as `fv.py`'s explicit step does it."""
    for _ in range(steps):
        u[1:-1, 1:-1] += c * (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2]
                              - 4.0 * u[1:-1, 1:-1])
    return u


def laplacian(n: int) -> sp.csc_matrix:
    """The five-point Laplacian on n x n cells with Dirichlet walls."""
    e = np.ones(n)
    t = sp.diags([-e[:-1], 2 * e, -e[:-1]], [-1, 0, 1])
    i = sp.identity(n)
    return (sp.kron(i, t) + sp.kron(t, i)).tocsc()


def elastic(n: int):
    """`fe.py`'s plane-stress stiffness on an n x n plate, clamped on its left edge."""
    g = fe.QuadGrid(n, n, 1.0 / n)
    K = fe.assemble_elastic(g, np.full(g.n_elem, 200e9), np.full(g.n_elem, 0.3))
    nodes = g.edge_nodes("left")
    fixed = np.concatenate([2 * nodes, 2 * nodes + 1])            # fe.elastic_dofs' order
    free = np.setdiff1d(np.arange(K.shape[0]), fixed)
    return K[free][:, free].tocsc()


# -- the three questions -----------------------------------------------------


def csr_steps(u: np.ndarray, A, steps: int, c: float = 0.2) -> np.ndarray:
    """The river's window step as `families/plume.py` takes it: ``u + c (b - A u)``
    with ``A`` a scipy CSR matrix (here b = 0), in place."""
    for _ in range(steps):
        u -= c * (A @ u)
    return u


def q1_threads(sizes, kernel: str = "numpy") -> list[dict]:
    rows = []
    for n in sizes:
        steps = max(4, int(4e7 / (n * n * WINDOWS)))
        if kernel == "numpy":
            base = [np.random.default_rng(i).random((n, n)) for i in range(WINDOWS)]
            one = lambda a: explicit_steps(a, steps)          # noqa: E731
        else:
            A = (laplacian(n) * 0.25).tocsr()
            base = [np.random.default_rng(i).random(n * n) for i in range(WINDOWS)]
            one = lambda a: csr_steps(a, A, steps)            # noqa: E731

        def run(T):
            arrays = [b.copy() for b in base]
            if T == 1:
                for a in arrays:
                    one(a)
            else:
                with cf.ThreadPoolExecutor(T) as pool:
                    list(pool.map(one, arrays))
            return arrays

        ref = run(1)
        times = {}
        for T in THREADS:
            out = run(T)
            assert all(np.array_equal(a, b) for a, b in zip(out, ref)), "threads changed bits"
            times[T] = median_time(lambda T=T: run(T))
        rows.append({"kernel": kernel, "cells_per_window": n * n, "windows": WINDOWS,
                     "steps": steps, "seconds": times,
                     "speedup": {T: times[1] / times[T] for T in THREADS}})
        print(f"  P {kernel} n={n}: "
              + ", ".join(f"T{T} {times[1] / times[T]:.2f}x" for T in THREADS), flush=True)
    return rows


def q2_splu(sizes, threads=THREADS) -> list[dict]:
    """The factors are timed without holding any: twelve 65,536-unknown factors in
    flight while twelve more were held ran this 15 GB laptop out of memory (seen,
    2026-09-30), so the largest size runs up to 8 threads."""
    rows = []
    for n in sizes:
        ts_ = tuple(T for T in threads if n < 256 or T <= 8)
        mats = [laplacian(n) for _ in range(WINDOWS)]
        rhs = [np.random.default_rng(i).random(n * n) for i in range(WINDOWS)]

        def factor(T):
            if T == 1:
                for A in mats:
                    spla.splu(A)
                return
            with cf.ThreadPoolExecutor(T) as pool:
                for _lu in pool.map(spla.splu, mats):
                    pass

        tf = {T: median_time(lambda T=T: factor(T), 3) for T in ts_}
        lus = [spla.splu(A) for A in mats]

        def solve(T):
            if T == 1:
                return [lu.solve(b) for lu, b in zip(lus, rhs)]
            with cf.ThreadPoolExecutor(T) as pool:
                return list(pool.map(lambda p: p[0].solve(p[1]), zip(lus, rhs)))

        ref = solve(1)
        assert all(np.array_equal(a, b) for a, b in zip(solve(4), ref)), "threads changed bits"
        ts = {T: median_time(lambda T=T: solve(T)) for T in ts_}
        del lus
        rows.append({"cells_per_window": n * n, "windows": WINDOWS,
                     "factor_seconds": tf, "solve_seconds": ts,
                     "factor_speedup": {T: tf[1] / tf[T] for T in ts_},
                     "solve_speedup": {T: ts[1] / ts[T] for T in ts_}})
        print(f"  splu n={n}: factor " + ", ".join(f"T{T} {tf[1] / tf[T]:.2f}x" for T in ts_)
              + "; solve " + ", ".join(f"T{T} {ts[1] / ts[T]:.2f}x" for T in ts_), flush=True)
    return rows


def q3_cache(sizes_direct, sizes_explicit, sizes_fe) -> dict:
    out = {"implicit_fv": [], "explicit": [], "fe": []}
    for n in sizes_direct:
        A = laplacian(n)
        b = np.ones(n * n)
        t = median_time(lambda: spla.splu(A).solve(b), 3)
        out["implicit_fv"].append({"cells": n * n, "seconds": t, "per_cell": t / (n * n)})
        print(f"  implicit n={n}: {t / (n * n) * 1e9:.1f} ns/cell", flush=True)
    for n in sizes_explicit:
        u = np.random.default_rng(0).random((n, n))
        steps = max(2, int(2e7 / (n * n)))
        t = median_time(lambda: explicit_steps(u, steps))
        out["explicit"].append({"cells": n * n, "steps": steps, "seconds": t,
                                "per_cell_step": t / (n * n * steps)})
        print(f"  explicit n={n}: {t / (n * n * steps) * 1e9:.2f} ns/cell/step", flush=True)
    for n in sizes_fe:
        K = elastic(n)
        b = np.ones(K.shape[0])
        t = median_time(lambda: spla.splu(K).solve(b), 3)
        out["fe"].append({"elements": n * n, "dofs": K.shape[0], "seconds": t,
                          "per_element": t / (n * n)})
        print(f"  fe n={n}: {t / (n * n) * 1e9:.1f} ns/element", flush=True)
    return out


def wait_for_ac(minutes: float) -> bool:
    """Poll the power source until it is AC, for at most ``minutes``.  A timing on
    battery is not a result here (4.4x slower at 1.4 GHz once), so the run waits."""
    t_end = time.time() + 60 * minutes
    while True:
        if machine.power_status().get("ac"):
            return True
        if time.time() >= t_end:
            return False
        time.sleep(10)


def main(argv) -> int:
    quick = "--quick" in argv
    if not quick:
        wait = next((float(a.split("=", 1)[1]) for a in argv if a.startswith("--wait-ac=")),
                    0.0)
        print(f"waiting up to {wait:g} min for AC power ...", flush=True)
        if not wait_for_ac(wait):
            print("still on battery: no timing taken (not a result on battery)", flush=True)
            return 3
        print("on AC", flush=True)
    rec = {"script": "scripts/fast_step0.py", "quick": quick, "rules": RULES,
           "repeats": REPEATS, "threads": THREADS, "cpu": cpu_facts(),
           "machine_before": machine.machine_record(),
           "note": ("the owner, 2026-09-30: the other two chats are not idle but are "
                    "writing documentation and theory, nothing on the demo")}
    awake = machine.keep_awake(True)
    rec["keep_awake"] = awake
    print(f"power: {rec['machine_before']['power']}; cpu: {rec['cpu']}", flush=True)
    folder = os.path.join(ROOT, "out", "workbench", "records", "fast") if not quick \
        else os.path.join(ROOT, "out", "fast", "step0")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"step0-{time.strftime('%Y%m%d-%H%M%S')}"
                        f"{'-quick' if quick else ''}.json")

    def save():
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1, default=str)

    try:
        t0 = time.perf_counter()
        only = next((a.split("=", 1)[1].split(",") for a in argv if a.startswith("--only=")),
                    ["q1", "q2", "q3"])
        rec["only"] = only
        if "q1" in only:
            print("1. does P scale?", flush=True)
            sizes = (32, 64) if quick else (64, 128, 256, 512)
            rec["q1_threads"] = q1_threads(sizes, "numpy") + q1_threads(sizes, "csr")
            save()
        if "q2" in only:
            print("2. does splu release the GIL?", flush=True)
            sz = next((tuple(int(x) for x in a.split("=", 1)[1].split(",")) for a in argv
                       if a.startswith("--q2-sizes=")), None)
            rec["q2_splu"] = q2_splu(sz or ((32,) if quick else (64, 128, 256)))
            save()
        if "q3" not in only:
            rec["seconds"] = time.perf_counter() - t0
            return 0
        print("3. where does each core leave cache?", flush=True)
        rec["q3_cache"] = q3_cache(
            (32, 64) if quick else (32, 64, 128, 256, 512),
            (64, 128) if quick else (64, 128, 256, 512, 1024, 2048),
            (16, 32) if quick else (16, 32, 64, 128, 256))
        rec["seconds"] = time.perf_counter() - t0
    except Exception as exc:                                      # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        rec["machine_after"] = machine.machine_record()
        machine.keep_awake(False)
        save()
        print(f"written {os.path.relpath(path, ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
