"""How much the BLAS thread count moves the squares price protocol's classical time (W358,
section 11.18): the classical expert's best of 5 per piece on two of section 11.17's
problems, with torch at `PRICE_THREADS`, under whatever BLAS thread variables the caller set.

    python scripts/chartop_blas_probe.py
    OPENBLAS_NUM_THREADS=1 python scripts/chartop_blas_probe.py      (pip numpy: OpenBLAS)
    MKL_NUM_THREADS=1 python scripts/chartop_blas_probe.py           (Anaconda numpy: MKL)

Prints one line: the CPU, numpy's BLAS, the variables set, and the two times.  It runs only
the classical expert, on problems that have been read.
"""
from __future__ import annotations

import io
import os
import sys
import time
from contextlib import redirect_stdout

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import chartop_price_squares as PS  # noqa: E402
from atlas.chartop import gates_squares as GS, host as hs  # noqa: E402

PROBLEMS = (("wall-2", "steady"), ("plate-hole", "steady"))
VARIABLES = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")


def blas() -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        np.show_config()
    cfg = buf.getvalue().lower()
    return "mkl" if "mkl" in cfg else ("openblas" if "openblas" in cfg else "unknown")


def ms_per_piece(name: str, problem: str, reps: int = 5) -> float:
    b = PS.build_problem(("workbench", name, problem), GS.BENCHMARK_N)
    L = b.layout(GS.KIND, GS.M_PORTS)
    C = hs.ClassicalExpert()
    C.respond(L.problems, L.pb)                       # warm-up
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        C.respond(L.problems, L.pb)
        best = min(best, time.perf_counter() - t0)
    return 1000.0 * best / L.P


def main() -> int:
    torch.set_num_threads(GS.PRICE_THREADS)
    env = {k: os.environ[k] for k in VARIABLES if os.environ.get(k)}
    times = ", ".join(f"{n} {p} {ms_per_piece(n, p):.1f}" for n, p in PROBLEMS)
    print(f"{PS.cpu_model()} | numpy {np.__version__} {blas()} | {env or 'BLAS threads default'} | "
          f"classical ms/piece: {times}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
