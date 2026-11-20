"""CO-T training pieces and their labels (the coolant plan's Phase 3,
[[chart-operator-stage-0b-and-1-plan]] section 6).

Each example comes from its own seed, so a set is reproducible from its seed range, and
it is built on the laptop and moved to a box as files, checked by sha256, never drawn again
there (sin and cos differ in the last bit between Windows and glibc).

**The piece** (`draw`):
- its chart, its Fourier number (steady in 30%, else log-uniform in [1e-2, 1e2]) and its
  source come from the CO-D generator (`data.draw`, pinned and only called), so the
  shapes are Stage 0's: curved four-sided pieces at n = 32, 64 and 128, certified and inside
  the CO-D card's envelope; its conductivity field is discarded, a CO-T piece being one
  material (the contrast lives in the host's k, never in the piece);
- its flow (a second random stream from the same seed), as a stream function on the
  reference square: ``solid`` (no flow, 15%), ``through`` (in through one side and out
  through the opposite one, walls on the other two: a channel strip, 55%), ``turn`` (in
  through one side and out through an adjacent one: a channel's corner, 20%) and
  ``wavy`` (a through-flow whose streamlines bend inside the piece, 10%); the profile
  across is laminar (3s^2 - 2s^3), plug (s), or a blend;
- its Peclet number, uniform in [0, cap(n)] with cap(n) = min(200, 1.8 n): the owner's cap,
  and a cell Peclet number of at most 1.8, so Galerkin does not oscillate;
- its resolution, from its own stream (`resolution`): 0.25 / 0.35 / 0.40 at n = 32 / 64 / 128,
  weighted toward the large pieces as the owner's amendment asks (plan section 15.2).

**What is stored** (`arrays`), compact: the chart's nodes, the stream function and the
source, as float32, with the scalars; and the **labels**, computed from those float32
arrays (cast back to float64), so that training sees exactly the operator its labels came
from (Stage 0's lesson: arrays rounded separately broke an identity at 1e-8):
- ``Lam``: the exact Petrov-Galerkin port matrix W^T K V (64 x 64), the trial and adjoint
  port maps at once;
- ``chi``: the particular field's port load (64);
- for validation pieces only, the full fields (trial modes, adjoint modes, particular).
At n = 128 a piece's full responses are about 16 MB; this is about 0.2 MB.

**Training does not read these labels.**  With exact adjoint modes, G^T K V~ = Lambda for any
trial modes V~ with exact traces, so Lambda cannot see a trial mode's error; the training
loss is label-free instead (`transport_torch.objective`).  The labels serve the validation
split's errors, the dataset's check 2 and the records.

**The size, measured** on 40 seeds (`scripts/cot_phase3_data.py --estimate`, 2026-11-19, the
CPU shared with other jobs): about 51 kB a piece at n = 32, 101 kB at 64 and 300 kB at 128,
so 168 kB at the mix and 8.4 GB for 50,000 pieces; a piece at n = 128 takes about 1.6 s to
draw (its chart) and 0.3-1.7 s to label on one core.
"""

from __future__ import annotations

import math

import numpy as np

from . import chart as ch
from . import data as d0
from . import fe
from . import ports as pt
from . import transport as tr

KINDS = (("solid", 0.15), ("through", 0.55), ("turn", 0.20), ("wavy", 0.10))
PROFILES = (("laminar", 0.6), ("plug", 0.2), ("blend", 0.2))
SEED_BASE = {"train": 10_000_000, "val": 11_000_000, "probe": 12_000_000}
CELL_PE_CAP = 1.8
#: the resolution mix, weighted toward the large pieces (the owner's amendment, plan
#: section 15.2: at least as many n = 128 pieces as n = 32 ones); CO-D's was 0.3/0.5/0.2
RESOLUTIONS = ((32, 0.25), (64, 0.35), (128, 0.40))


def pe_cap(n: int) -> float:
    return min(tr.PE_MAX, CELL_PE_CAP * n)


def resolution(seed: int) -> int:
    """The resolution of a seed's piece, from its own stream (deterministic)."""
    rng = np.random.default_rng([seed, 5])
    ns, ps = zip(*RESOLUTIONS)
    return int(rng.choice(ns, p=ps))


def _profile(s, kind: str, a: float):
    s = np.clip(np.asarray(s, float), 0.0, 1.0)
    lam = 3.0 * s ** 2 - 2.0 * s ** 3
    if kind == "laminar":
        return lam
    if kind == "plug":
        return s
    return a * lam + (1.0 - a) * s


def flow_on_square(rng, n: int) -> tuple[np.ndarray, dict]:
    """A unit-flux stream function on the reference grid, (n+1, n+1) [j, i], and what it is."""
    kinds, pk = zip(*KINDS)
    kind = str(rng.choice(kinds, p=pk))
    t = np.arange(n + 1) / n
    XI, ETA = np.meshgrid(t, t)
    profs, pp = zip(*PROFILES)
    prof = str(rng.choice(profs, p=pp))
    a = float(rng.uniform(0.2, 0.8))
    rot = int(rng.integers(4))
    sign = 1.0 if rng.uniform() < 0.5 else -1.0
    meta = {"kind": kind, "profile": prof, "rotation": rot, "sign": sign}
    if kind == "solid":
        return np.zeros((n + 1, n + 1)), meta
    # the across-coordinate s in the piece's own frame, rotated by quarter turns
    u, v = XI, ETA
    for _ in range(rot):
        u, v = v, 1.0 - u
    if kind == "through":
        s = v
    elif kind == "turn":
        s = 1.0 - (1.0 - u) * (1.0 - v)
    else:
        dlt = float(rng.uniform(-0.25, 0.25))
        s = v + dlt * np.sin(math.pi * u) * np.sin(math.pi * v)
        meta["bend"] = dlt
    return sign * _profile(s, prof, a), meta


def draw(seed: int, n: int | None = None) -> tuple[tr.TransportPiece, dict]:
    """The CO-T piece of one seed (deterministic), and what was drawn.  ``n`` defaults to
    `resolution` (seed), the CO-T mix, never CO-D's."""
    ex = d0.draw(seed, n=resolution(seed) if n is None else n)
    n = ex.n
    rng = np.random.default_rng([seed, 7])
    psi0, meta = flow_on_square(rng, n)
    pe = 0.0 if meta["kind"] == "solid" else float(rng.uniform(0.0, pe_cap(n)))
    p = tr.make_piece(ex.problem.chart, psi0, k=1.0, alpha=1.0, fo=ex.fo, g=ex.problem.g,
                      label=f"seed {seed}")
    if pe > 0.0:
        p.psi = p.psi * (pe / p.peclet())
    meta.update(seed=seed, n=n, fo=ex.fo, pe=pe, family=ex.family, method=ex.method)
    return p, meta


def arrays(p: tr.TransportPiece, meta: dict) -> dict:
    """What an example stores, float32, with its scalars."""
    return {"X": p.chart.X.astype(np.float32), "psi": p.psi.astype(np.float32),
            "g": np.asarray(p.g, float).astype(np.float32),
            "n": np.int32(p.n), "fo": np.float64(np.nan if p.fo is None else p.fo),
            "pe": np.float64(meta["pe"]), "kind": meta["kind"], "seed": np.int64(meta["seed"])}


def piece_from_arrays(a: dict) -> tr.TransportPiece:
    """The piece exactly as training sees it: the stored float32 arrays, in float64."""
    X = np.asarray(a["X"], np.float64)
    fo = float(a["fo"])
    return tr.make_piece(ch.Chart(X, "stored"), np.asarray(a["psi"], np.float64), k=1.0, alpha=1.0,
                         fo=None if math.isnan(fo) else fo, g=np.asarray(a["g"], np.float64))


def labels(p: tr.TransportPiece, m: int = 16, full: bool = False) -> dict:
    """The compact labels of a piece (from `piece_from_arrays`)."""
    Q = pt.basis("sine-hat", p.n, m).Q
    r, ex = tr.exact_response(p, Q)
    Lam = ex.port_matrix(Q, r.H, r.G)
    b = p.load()
    loop, inner = fe.grid(p.n)
    # the particular field's port load, as the host forms it: W^T (b - K U_p)
    Up = fe.full_field(p.n, r.up, np.zeros(4 * p.n))
    W = fe.full_field(p.n, r.G, Q)
    chi = W.T @ (b - ex.K @ Up)
    out = {"Lam": Lam, "chi": chi}
    if full:
        out.update(H=r.H, G=r.G, up=r.up)
    return out


def estimate(seeds, m: int = 16) -> dict:
    """Bytes and label seconds per example, measured on a sample of seeds."""
    import io
    import time
    sizes, secs, ns = [], [], []
    for s in seeds:
        p, meta = draw(int(s))
        a = arrays(p, meta)
        t0 = time.perf_counter()
        lab = labels(piece_from_arrays(a), m)
        secs.append(time.perf_counter() - t0)
        buf = io.BytesIO()
        np.savez(buf, **{k: v for k, v in a.items() if k != "kind"}, **lab)
        sizes.append(buf.tell())
        ns.append(meta["n"])
    return {"sample": len(sizes), "bytes_mean": float(np.mean(sizes)), "seconds_mean": float(np.mean(secs)),
            "by_n": {int(k): {"count": int(sum(1 for x in ns if x == k)),
                              "bytes": float(np.mean([z for z, x in zip(sizes, ns) if x == k])),
                              "seconds": float(np.mean([z for z, x in zip(secs, ns) if x == k]))}
                     for k in sorted(set(ns))}}


__all__ = ["KINDS", "RESOLUTIONS", "SEED_BASE", "arrays", "draw", "estimate", "flow_on_square", "labels",
           "pe_cap", "piece_from_arrays", "resolution"]
