"""Phase 2 of the Stage 0 rework: TL0, the two-level expert, measured untrained.

[[chart-operator-stage-0-rework]] section 8 (Phase 2) and section 9 (the predictions).

**TL0** is U's configuration -- S's form (split, corner bubble, strips) with zero heads,
`chartop_forms_diagnose.zero_heads` -- followed by the coarse correction ``C`` of
`atlas.chartop.coarse` (`correct_fields`): the modes against their traces, the particular
field against the piece's load, and on a step the level field against
``b_1 = (M 1)_I / s0``.  Nothing is trained, no weights are read, and the test split is
never drawn.

    python scripts/chartop_twolevel_controls.py --part modes --ns 32      # one resolution per process
    python scripts/chartop_twolevel_controls.py --part e4 --problem s-channel:steady
    python scripts/chartop_twolevel_controls.py --part cost               # on AC: a first reading
    python scripts/chartop_twolevel_controls.py --judge                   # the registered predictions

**Parts**, each writing ``out/chartop/twolevel/<part>-<unit>.json`` after every measurement
(a stopped run resumes where it stopped):
- ``modes``: per piece, for U and TL0 at each ``n_c <= n``: ``e_H`` by column group
  (`chartop_forms_diagnose.column_groups`: hats, ``j = 1-3``, ``4-8``, ``9-15``; and the
  low group, hats with ``j = 1-3``), ``e_H`` over all 64 columns, ``e_H,L`` over the 36
  learned, ``e_p``, the level field's error ``e_1`` (steps), the kernel identity's residual
  ``e_K``, ``e_K,inf`` (``H c1 + s0 u_1 - 1``, as the registered K measured test pieces), CX
  of the modes and of the particular field, and the column-by-column check against U.  On:
  the post-hoc validation pieces 2,000,100 onward (``--count``, 64), labelled from their
  stored arrays as `data.labels_from_arrays` labels them; B4's four pieces, the first four
  at n = 128; and W362's piece types (`chartop_w362_diagnose.groups`).
- ``e4``: the six Stage 0 benchmarks (n = 64) in both frames, for U, TL0 at each ``n_c``,
  TL0 at ``n_c = 64`` (the positive control: the classical superelement) and the adapter
  with no coarse size (the control: U bit for bit).  Per variant: E4 by
  `gates_forms.judge_e4_both_frames`, D by `judge_d`, K through the host's uniform state
  (`forms.uniform_state`), CX per piece, the network term split by mode group
  (`chartop_forms_diagnose.split_term`, A1's split) and by where it sits (`REGIONS`: the
  corner blocks by the corner's angle, interface-side points, interfaces inside, the side
  layers, the interior).
- ``cost``: per piece at n = 32, 64, 128 (the first ``--cost-count`` validation pieces),
  the coarse level (`correct_fields` from the assembled matrix: the space, its factor and
  the three corrections) against `fe.Exact` (the LU, 64 mode columns and the particular
  field), the matrix's assembly excluded from both, best of 5; U's forward and strips
  beside.  Torch and the BLAS at 4 threads; the power state is sampled before and after
  every resolution, and the run refuses to start on battery.

**The predictions** (`PREDICTIONS`) are the rework page's section 9, each with the reading
`judge` applies.  They were registered, with the owner's go, before any part ran on the
pieces above.  ``--smoke`` runs every code path on seeds and a layout no registration
draws (validation-style seeds from 9,000,100, `benchmarks_squares` seed 9,000,008) at
n = 16 and 32, into a scratch directory, and prints only the controls.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
import time

THREADS = 4
if "numpy" not in sys.modules:             # run as a script: fix the BLAS pools before numpy loads
    for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_k] = str(THREADS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import chartop_evaluate as ev0  # noqa: E402
import chartop_forms_diagnose as fd  # noqa: E402
import chartop_registry as registry  # noqa: E402
import chartop_w362_diagnose as w362  # noqa: E402
from atlas.chartop import benchmarks as bm, benchmarks_squares as BS, chart as ch  # noqa: E402
from atlas.chartop import coarse as co, data, fe, forms as fm, gates_forms as GF  # noqa: E402
from atlas.chartop import host as hs, ports as pt  # noqa: E402
from atlas.chartop.expert import LearnedExpert  # noqa: E402

OUT = os.path.join(ROOT, "out", "chartop", "twolevel")
KIND, M_PORTS = GF.KIND, GF.M_PORTS
NCS = (4, 8, 16, 32)
NS = (32, 64, 128)
VAL_COUNT = 64
B4_PIECES, B4_N = 4, 128
COST_COUNT, COST_REPS = 8, 5
BENCH_N = GF.BENCHMARK_N
GROUPS = fd.column_groups()
LOW = sorted(GROUPS["hats"] + GROUPS["j1-3"])
LEARNED = list(fm.learned_columns(KIND, M_PORTS))
#: the column-by-column check's rounding allowance: TL0's error over U's, relative, plus
#: this times the column's own energy norm
CHECK_REL, CHECK_ABS = 1e-9, 1e-12
#: where the remaining error sits, per element: precedence in this order
REGIONS = ("corner, right", "corner, not right", "interface at a side", "interface inside",
           "side layer", "interior")
RIGHT_DEG = 2.0                      # a corner within this of 90 degrees is right
#: the smoke run's pieces and layout: seeds no registration draws
SMOKE_SEED, SMOKE_LAYOUT_SEED = 9_000_100, 9_000_008

PREDICTIONS = (
    {"id": "1",
     "text": "Proposition A and Lemma B hold to rounding on every piece: CX at most 1e-12, no "
             "column's e_H above U's, and n_c = n exact.",
     "reading": "For TL0 at every n_c, on every piece of the modes part and every benchmark piece "
                "in both frames: no column whose energy-norm error exceeds U's by more than 1e-9 of "
                "U's plus 1e-12 of the exact column's own energy norm; and where n_c < n, CX of the "
                "modes <= 1e-12, and of the particular field <= 1e-12 wherever the piece has a "
                "load.  e_K and e_K,inf <= 1e-12 for TL0 at every n_c: the level-field identity on "
                "every piece of the modes part, the uniform state through the host on every "
                "benchmark piece as registered.  At n_c = n: e_H <= 1e-10 on every n = 32 "
                "validation piece, and on every benchmark problem the energy error within 1e-10 "
                "(absolute, relative units) of the 16-mode floor in both frames."},
    {"id": "2",
     "text": "bend-3, steady, meets E4 at every n_c, as U alone already does.",
     "reading": "judge_e4_both_frames is met for TL0 at n_c = 4, 8, 16 and 32 on bend-3 steady."},
    {"id": "3",
     "text": "[AI Inference] at n_c = 16, the low group's e_H (hats and j = 1-3) is at most half "
             "of U's on the validation pieces.",
     "reading": "At each n in {32, 64, 128}: the median over the validation pieces of the low "
                "group's e_H (its error energy over its own exact energy) for TL0 at n_c = 16 is "
                "at most 0.5 times U's median."},
    {"id": "4",
     "text": "[AI Inference] s-channel and insert-round, steady, meet E4's energy bar untrained "
             "at n_c = 16 or 32.",
     "reading": "For each of the two steady problems, TL0's energy error is <= 0.02 in both "
                "frames at n_c = 16 or at n_c = 32."},
    {"id": "5",
     "text": "[AI Inference] the steps' maximum-error bar is the most at risk, because the "
             "one-step boundary layer is thinner than a coarse cell.  Refuted if all three "
             "steps meet it at n_c = 16.",
     "reading": "At n_c = 16, at least one of the three step problems misses E4's maximum-error "
                "bar in at least one frame."},
    {"id": "6",
     "text": "[AI Inference] at n = 128, n_c = 16, the coarse level costs under 0.1 of an exact "
             "solve.",
     "reading": "The median over the cost part's n = 128 pieces of the coarse level's time at "
                "n_c = 16 over the exact solve's is < 0.1, every sample on AC power; judged only "
                "if the power held."},
)


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# provenance and records
# ---------------------------------------------------------------------------


def provenance() -> dict:
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True,
                                  timeout=60).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"commit": git("rev-parse", "HEAD"),
            "dirty_code": git("status", "--porcelain", "--", "atlas", "scripts"),
            "script_sha256": registry.sha256_file(os.path.abspath(__file__)),
            "coarse_sha256": registry.sha256_file(os.path.join(ROOT, "atlas", "chartop", "coarse.py")),
            "threads": {"torch": torch.get_num_threads(), "blas_env": THREADS},
            "started": dt.datetime.now().astimezone().isoformat(timespec="seconds")}


def record_path(out: str, part: str, unit: str) -> str:
    return os.path.join(out, f"{part}-{unit}.json")


def load_record(path: str, config: dict) -> dict:
    """The unit's record so far (a resumed run), or a new one; a record of another
    configuration is refused."""
    if os.path.exists(path):
        rec = json.loads(registry._retry(lambda: open(path, encoding="utf-8").read()))
        if rec.get("config") != config:
            raise SystemExit(f"{path} holds another configuration: move it away to start again")
        return rec
    return {"config": config, "rows": [], "provenance": provenance()}


def save(path: str, rec: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    registry.write_atomic(path, json.dumps(rec, indent=1, default=_json).encode("utf-8"))


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


# ---------------------------------------------------------------------------
# one piece: its arrays, U's answers, the exact fields, the measures
# ---------------------------------------------------------------------------


def arrays_of_example(ex) -> dict:
    """A drawn example as stored (`data.arrays`: float32), coefficients read in float64."""
    a = data.arrays(ex)
    out = {k: np.asarray(a[k], np.float64) for k in ("a11", "a12", "a22", "sigma", "w", "g")}
    out.update(s0=float(a["s0"]), inputs=np.asarray(a["inputs"], np.float32),
               scalars=np.asarray(a["scalars"], np.float32))
    return out


def arrays_of_problem(p) -> dict:
    """A `PieceProblem` in float64, its own source unnormalised."""
    t = p.tensors()
    chans, scal, _rms = p.inputs()
    out = {k: np.asarray(t[k], np.float64) for k in ("a11", "a12", "a22", "sigma", "w")}
    out.update(g=np.asarray(p.g, np.float64), s0=float(p.s0), inputs=chans, scalars=scal)
    return out


def u_answers(U, a: dict):
    """U's modes ((n-1)^2, 64), particular field and level field, and its strips' seconds."""
    b = {k: torch.tensor(a[k][None]) for k in ("a11", "a12", "a22", "sigma", "w", "g")}
    b["inputs"] = torch.tensor(a["inputs"][None])
    b["scalars"] = torch.tensor(a["scalars"][None])
    b["s0"] = torch.tensor([a["s0"]], dtype=torch.float64)
    with torch.no_grad():
        H, up = fm.call(U, b)
    K = H.shape[1]
    return (H[0].reshape(K, -1).T.numpy().copy(), up[0].reshape(-1).numpy().copy(),
            U.last_level[0].reshape(-1).numpy().copy(), float(U.last_strip_seconds))


class Exact:
    """The piece's operator and exact fields, from the same arrays U read."""

    def __init__(self, a: dict, Q):
        n = a["a11"].shape[-1]
        self.n, self.Q, self.s0 = n, Q, a["s0"]
        self.M = fe.assemble(n, a["a11"], a["a12"], a["a22"], a["sigma"])
        ex = fe.Exact(self.M, n)
        self.A_I = ex.A_I.tocsr()
        self.H = ex.modes(Q)
        _loop, inner = fe.grid(n)
        self.b = (fe.mass(n, a["w"]) @ a["g"].ravel())[inner]
        self.up = ex.particular(self.b)
        self.lam = np.diag(ex.port_matrix(Q, self.H)).copy()
        self.BQ = -np.asarray(ex.M_IG @ Q)
        self.u1 = ex.particular(co.level_rhs(self.M, n, self.s0)) if self.s0 > 0 else None

    def energy(self, v) -> float:
        return float(v @ (self.A_I @ v))


def _ge(err, lam, cols) -> float:
    return math.sqrt(float(err[cols].sum()) / float(lam[cols].sum()))


def measures(x: Exact, H, up, level, c1, cx_nc: int | None, err_U=None) -> tuple[dict, np.ndarray]:
    """One answer's row and its per-column error energies."""
    E = H - x.H
    err = np.einsum("ik,ik->k", E, np.asarray(x.A_I @ E))
    row = {g: _ge(err, x.lam, cols) for g, cols in GROUPS.items()}
    row["low"] = _ge(err, x.lam, LOW)
    row["e_H"] = math.sqrt(float(err.sum()) / float(x.lam.sum()))
    row["e_H_L"] = GF.e_h_learned(err, x.lam, LEARNED)
    nu = x.energy(x.up)
    row["e_p"] = math.sqrt(max(x.energy(up - x.up), 0.0) / nu) if nu > 0 else None
    r = H @ c1 - 1.0 + (x.s0 * level if x.s0 > 0 else 0.0)
    row["e_K"] = math.sqrt(max(x.energy(r), 0.0) / float(x.lam.sum()))
    row["e_K_inf"] = float(np.abs(r).max())
    if x.u1 is not None:
        row["e_1"] = math.sqrt(max(x.energy(level - x.u1), 0.0) / x.energy(x.u1))
    if cx_nc is not None and cx_nc < x.n:
        P = co.prolongation(x.n, cx_nc)
        row["cx_nc"] = cx_nc
        row["cx"] = co.coarse_exactness(P, co.residual(x.M, x.n, H, x.Q), x.BQ)["rel"]
        cp = co.coarse_exactness(P, co.residual(x.M, x.n, up, rhs=x.b), x.b)
        row["cx_p"], row["cx_p_abs"] = cp["rel"], cp["num"]
    if err_U is not None:
        a, u, s = np.sqrt(np.maximum(err, 0)), np.sqrt(np.maximum(err_U, 0)), np.sqrt(np.maximum(x.lam, 0))
        row["check_violations"] = int(np.sum(a > u * (1.0 + CHECK_REL) + CHECK_ABS * s))
        pos = u > 0
        row["check_max_ratio"] = float((a[pos] / u[pos]).max()) if pos.any() else None
    return row, err


def measure_piece(U, a: dict, meta: dict, ncs) -> dict:
    """U and TL0 at every ``n_c`` in ``ncs`` with ``n_c <= n``, on one piece."""
    n = a["a11"].shape[-1]
    Q = pt.basis(KIND, n, M_PORTS).Q
    c1 = fm.kernel_coefficients(KIND, n, M_PORTS)
    x = Exact(a, Q)
    H, up, lev, strip_s = u_answers(U, a)
    rowU, errU = measures(x, H, up, lev, c1, 16 if 16 < n else n // 2)
    out = dict(meta, n=n, s0=a["s0"], trace_Lambda=float(x.lam.sum()), U=rowU, u_strip_s=strip_s)
    for nc in ncs:
        if nc > n:
            continue
        t0 = time.perf_counter()
        Hc, upc, levc = co.correct_fields(x.M, n, nc, H, up, lev, a["s0"], x.b, Q)
        dt_c = time.perf_counter() - t0
        row, _ = measures(x, Hc, upc, levc, c1, nc, err_U=errU)
        row["coarse_s"] = dt_c
        out[f"TL0-{nc}"] = row
    return out


# ---------------------------------------------------------------------------
# the modes part
# ---------------------------------------------------------------------------


def modes_units(n: int, count: int, seed_base: int, w362_types: bool):
    """(unit, [(meta, arrays-maker)]) for one resolution."""
    val = [({"set": "validation", "seed": seed_base + k, "b4": bool(n == B4_N and k < B4_PIECES)},
            (lambda s=seed_base + k: _val(s, n))) for k in range(count)]
    yield f"val-n{n}", val
    if w362_types:
        items = []
        for typ, plist in w362.groups(n).items():
            for label, p in plist:
                items.append(({"set": "w362", "type": typ, "label": label},
                              (lambda p=p: (arrays_of_problem(p), {}))))
        yield f"w362-n{n}", items


def _val(seed: int, n: int):
    ex = data.draw(seed, n=n)
    return arrays_of_example(ex), {"fo": ex.fo, "freq": ex.freq, "family": ex.family,
                                   "sup_mu": float(ex.envelope["sup_abs_mu"]),
                                   "mu_bin": data.mu_bin(float(ex.envelope["sup_abs_mu"]))}


def part_modes(args, U) -> list:
    paths = []
    for n in args.ns:
        for unit, items in modes_units(n, args.count, args.seed_base, not args.smoke):
            config = {"part": "modes", "unit": unit, "ncs": list(args.ncs), "count": args.count,
                      "seed_base": args.seed_base}
            path = record_path(args.out, "modes", unit)
            rec = load_record(path, config)
            done = {json.dumps(r["key"], sort_keys=True) for r in rec["rows"]}
            for meta, make in items:
                key = {k: meta[k] for k in ("set", "seed", "label") if k in meta}
                if json.dumps(key, sort_keys=True) in done:
                    continue
                t0 = time.perf_counter()
                a, extra = make()
                row = measure_piece(U, a, dict(meta, **extra), args.ncs)
                row["key"] = key
                row["elapsed_s"] = time.perf_counter() - t0
                rec["rows"].append(row)
                save(path, rec)
            rec["finished"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
            save(path, rec)
            log(f"modes {unit}: {len(rec['rows'])} pieces -> {os.path.relpath(path, ROOT)}")
            paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# the e4 part
# ---------------------------------------------------------------------------


def element_energy(a11, a12, a22, sigma, U) -> np.ndarray:
    """`fe.energy`'s element densities [n, n] for one nodal field [n+1, n+1], not summed."""
    n = U.shape[-1] - 1
    u00, u10, u11, u01 = U[:-1, :-1], U[:-1, 1:], U[1:, 1:], U[1:, :-1]
    d0, d1 = u10 - u00, u11 - u01
    e0, e1 = u01 - u00, u11 - u10
    dens = (a11 * (d0 * d0 + d0 * d1 + d1 * d1) / 3.0 + a22 * (e0 * e0 + e0 * e1 + e1 * e1) / 3.0
            + a12 * (d0 + d1) * (e0 + e1) / 2.0)
    sq = u00 * u00 + u10 * u10 + u11 * u11 + u01 * u01
    ed = u00 * u10 + u10 * u11 + u11 * u01 + u01 * u00
    dg = u00 * u11 + u10 * u01
    return dens + (sigma / n ** 2) * (sq / 9.0 + ed / 9.0 + dg / 18.0)


def element_regions(p) -> np.ndarray:
    """Per element [n, n], the index into `REGIONS`: the n/8 x n/8 block at each corner, by
    that corner's angle (`chart.corner_angles`); then within n/8 of a side and
    `chartop_forms_diagnose.NEAR` cells of a material interface (its `_jump_nodes`); then
    at an interface inside; then the rest of the side layers; then the interior."""
    n = p.n
    w = max(1, n // 8)
    j, i = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    lo_i, hi_i, lo_j, hi_j = i < w, i >= n - w, j < w, j >= n - w
    side = lo_i | hi_i | lo_j | hi_j
    jn = fd._jump_nodes(np.asarray(p.log_kappa_hat_e, float))
    at_jump = jn[:-1, :-1] | jn[:-1, 1:] | jn[1:, 1:] | jn[1:, :-1]
    reg = np.full((n, n), REGIONS.index("interior"))
    reg[side] = REGIONS.index("side layer")
    reg[at_jump & ~side] = REGIONS.index("interface inside")
    reg[at_jump & side] = REGIONS.index("interface at a side")
    angles = ch.corner_angles(np.asarray(p.chart.X))
    for c, blk in enumerate((lo_i & lo_j, hi_i & lo_j, hi_i & hi_j, lo_i & hi_j)):
        right = abs(float(angles[c]) - 90.0) <= RIGHT_DEG
        reg[blk] = REGIONS.index("corner, right" if right else "corner, not right")
    return reg


#: below this fraction of ||u||_M the network term is at the floor: its splits are not formed
SPLIT_FLOOR = 1e-9


def where_split(L, cp, cl) -> dict:
    """Theorem 2's network term, ``sum_i kappa_i ||w_i||^2_A``, by `REGIONS`: each piece's
    ``w_i = e_p,i + E_i c*_i`` (zero on the loop) as element energies, summed by region.
    The shares sum to one exactly; the total is checked against `chartop_evaluate.network_term`."""
    by = np.zeros(len(REGIONS))
    for i, p in enumerate(L.problems):
        ci = L.R[i] @ cl.c
        rL, rC = cp.responses[i], cl.responses[i]
        w = (rL.up - rC.up) + (rL.H - rC.H) @ ci
        W = fe.full_field(p.n, w, np.zeros(4 * p.n)).reshape(p.n + 1, p.n + 1)
        t = p.tensors()
        dens = p.kappa_ref * element_energy(t["a11"], t["a12"], t["a22"], t["sigma"], W)
        by += np.bincount(element_regions(p).ravel(), weights=dens.ravel(), minlength=len(REGIONS))
    tot = float(by.sum())
    nt = ev0.network_term(L, cp, cl)
    return {"total": tot, "check_rel": abs(math.sqrt(max(tot, 0.0)) - nt) / max(nt, 1e-300),
            "share": {r: float(v / tot) for r, v in zip(REGIONS, by)}}


def corrected(problems, resp, pb_of, nc):
    """TL0's answers from U's, piece by piece (`coarse.correct_fields`, as the host's
    adapter `coarse.CoarseCorrected` computes them)."""
    out = []
    for p, r in zip(problems, resp):
        _loop, inner = fe.grid(p.n)
        H, up, lev = co.correct_fields(p.matrix(), p.n, nc, r.H, r.up, r.level, p.s0,
                                       p.load()[inner], pb_of(p.n).Q)
        out.append(hs.Response(H, up, lev))
    return out


def column_violations(A_I, H, H_U, H_exact, lam) -> int:
    """Columns whose energy-norm error exceeds U's by more than `CHECK_REL` of U's plus
    `CHECK_ABS` of the exact column's own energy norm, ``sqrt(Lambda_kk)`` (``lam``, the
    diagonal of the exact port matrix), as `measures` checks it."""
    def norms(E):
        return np.sqrt(np.maximum(np.einsum("ik,ik->k", E, np.asarray(A_I @ E)), 0.0))
    a, u = norms(H - H_exact), norms(H_U - H_exact)
    return int(np.sum(a > u * (1.0 + CHECK_REL) + CHECK_ABS * np.sqrt(np.maximum(lam, 0.0))))


def piece_checks(L, resp, ures, cl, nc, respU=None) -> dict:
    """Per piece: CX of the modes and of the particular field (when ``nc < n``), the
    column-by-column check against U's answers ``respU`` (when given), and, when ``ures``
    (the variant's answers for each piece's uniform state) is given, K through the uniform
    state as `chartop_evaluate_forms.piece_measures` measures it."""
    out = {"cx": [], "cx_p": [], "e_K": [], "e_K_inf": [], "check_violations": []}
    for i, p in enumerate(L.problems):
        n, Q = p.n, L.pb(p.n).Q
        M = p.matrix().tocsr()
        loop, inner = fe.grid(n)
        if nc is not None and nc < n:
            P = co.prolongation(n, nc)
            BQ = -np.asarray(M[inner][:, loop] @ Q)
            out["cx"].append(co.coarse_exactness(P, co.residual(M, n, resp[i].H, Q), BQ)["rel"])
            b = p.load()[inner]
            out["cx_p"].append(co.coarse_exactness(P, co.residual(M, n, resp[i].up, rhs=b), b)["rel"])
        if respU is not None:
            out["check_violations"].append(column_violations(M[inner][:, inner], resp[i].H, respU[i].H,
                                                             cl.responses[i].H, np.diag(cl.grams[i])))
        if ures is not None:
            r = ures[i].up + ures[i].H @ fm.kernel_coefficients(KIND, n, M_PORTS) - 1.0
            km = fm.kernel_measures(r, M[inner][:, inner], float(np.trace(cl.grams[i])))
            out["e_K"].append(km["e_K"])
            out["e_K_inf"].append(km["e_K_inf"])
    return out


def e4_unit(name: str, problem: str, U_le, ncs, n: int = BENCH_N, smoke: bool = False) -> dict:
    if smoke:
        b = BS.build(BS.draw(SMOKE_LAYOUT_SEED), n, problem)
    else:
        kw = {"envelope": data.ENVELOPE} if name == "s-channel" else {}
        b = bm.build(name, n, problem, **kw)
    frames = {"registered": b, "lowered": bm.datum_shift(b, GF.DATUM_SHIFT_K)}
    rec = {"name": name, "problem": problem, "n": n, "pieces": len(b.problems), "frames": {}}
    top = f"TL0-{n} (control)"
    for frame, bb in frames.items():
        L = bb.layout(KIND, M_PORTS)
        mono = hs.monolithic(L)
        cl = L.solve(hs.ClassicalExpert())
        un, ud = cl.energy_error(mono.fields)
        mx0, span = ev0.field_errors(mono.fields, cl.fields)
        fr = {"floor_energy_rel": un / ud, "floor_abs": un, "reference_energy": ud, "floor_max": mx0,
              "span": span, "refusals": L.check_envelope(data.ENVELOPE), "variants": {}}
        respU = U_le.respond(L.problems, L.pb)
        none = L.solve(co.CoarseCorrected(U_le, None), refuse_outside=False)
        fr["control_none_bit_for_bit"] = bool(all(
            np.array_equal(r.H, q.H) and np.array_equal(r.up, q.up) for r, q in zip(none.responses, respU)))
        ups = [fm.uniform_state(p) for p in L.problems] if frame == "registered" else None
        uresU = U_le.respond(ups, L.pb) if ups else None
        variants = {"U": (respU, uresU, None)}
        for nc in [c for c in ncs if c < n] + [n]:
            variants[top if nc == n else f"TL0-{nc}"] = (
                corrected(L.problems, respU, L.pb, nc),
                corrected(ups, uresU, L.pb, nc) if ups else None, nc)
        for v, (resp, ures, nc) in variants.items():
            cp = L.couple(resp, calls=len(L.problems))
            num, den = cp.energy_error(mono.fields)
            mx, _ = ev0.field_errors(mono.fields, cp.fields)
            row = {"energy_abs": num, "energy_rel": num / den, "max_err": mx, "max_over_range": mx / span,
                   "min_eig": cp.min_eig(), "network_term": ev0.network_term(L, cp, cl)}
            if v != top and row["network_term"] > SPLIT_FLOOR * ud:
                st = fd.split_term(L, cp, cl)
                row.update(split_check_rel=st["check_rel"], split=st["share"],
                           split_alone=st["alone_over_total"], where=where_split(L, cp, cl))
            else:
                row["split"] = None                      # at the floor: no shares are formed
            row["pieces"] = piece_checks(L, resp, ures, cl, nc, respU if v != "U" else None)
            fr["variants"][v] = row
        rec["frames"][frame] = fr
    reg, low = rec["frames"]["registered"], rec["frames"]["lowered"]
    rec["verdicts"] = {}
    for v in reg["variants"]:
        a, z = reg["variants"][v], low["variants"][v]
        rec["verdicts"][v] = {
            "e4": GF.judge_e4_both_frames(
                {"energy_rel": a["energy_rel"], "max_err": a["max_err"], "span": reg["span"],
                 "floor_max": reg["floor_max"]},
                {"energy_rel": z["energy_rel"], "max_err": z["max_err"], "span": low["span"],
                 "floor_max": low["floor_max"]}),
            "d": GF.judge_d(a["energy_abs"], z["energy_abs"], reg["reference_energy"],
                            a["max_err"], z["max_err"], reg["span"]),
            "k": GF.judge_k(a["pieces"]["e_K"], a["pieces"]["e_K_inf"])}
    return rec


def part_e4(args, U) -> list:
    U_le = LearnedExpert(U, envelope=data.ENVELOPE)
    if args.smoke:
        units = [("smoke-squares", p) for p in ([args.problem] if args.problem else ["steady", "step"])]
    elif args.problem:
        units = [tuple(args.problem.split(":"))]
    else:
        units = [(b, p) for b in GF.HELD_OUT_BENCHMARKS for p in GF.BENCHMARK_PROBLEMS]
    paths = []
    for name, problem in units:
        unit = f"{name}-{problem}"
        path = record_path(args.out, "e4", unit)
        config = {"part": "e4", "unit": unit, "ncs": list(args.ncs), "n": args.bench_n}
        rec = load_record(path, config)
        if rec.get("result") is None:
            t0 = time.perf_counter()
            rec["result"] = e4_unit(name, problem, U_le, args.ncs, args.bench_n, args.smoke)
            rec["elapsed_s"] = time.perf_counter() - t0
            rec["finished"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
            save(path, rec)
        log(f"e4 {unit} -> {os.path.relpath(path, ROOT)}")
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# the cost part
# ---------------------------------------------------------------------------


def power_state() -> dict:
    """``{"on_ac": bool | None, "raw": ...}`` from Win32_Battery (2 = on AC)."""
    if os.name != "nt":
        return {"on_ac": None, "raw": "not Windows"}
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Battery | Select-Object -First 1).BatteryStatus"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        return {"on_ac": None, "raw": repr(e)}
    return {"on_ac": out == "2", "raw": out}


def python_processes() -> str:
    if os.name != "nt":
        return ""
    try:
        return subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe"], capture_output=True,
                              text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError) as e:
        return repr(e)


def _best(f, reps: int) -> float:
    f()                                            # warm
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def _exact_solve(M, n, Q, b):
    e = fe.Exact(M, n)
    e.modes(Q)
    e.particular(b)


def part_cost(args, U) -> list:
    path = record_path(args.out, "cost", "laptop")
    config = {"part": "cost", "ns": list(args.ns), "ncs": list(args.ncs), "count": args.cost_count,
              "reps": COST_REPS, "seed_base": args.seed_base}
    rec = load_record(path, config)
    start = power_state()
    if start["on_ac"] is not True and not args.allow_battery:
        raise SystemExit(f"the cost part runs on AC power only (power state {start['raw']!r})")
    rec.setdefault("power", []).append({"when": "start", **start})
    rec.setdefault("python_processes", []).append(python_processes())
    done = {(r["seed"], r["n"]) for r in rec["rows"]}
    for n in args.ns:
        Q = pt.basis(KIND, n, M_PORTS).Q
        rec["power"].append({"when": f"n={n} start", **power_state()})
        for k in range(args.cost_count):
            seed = args.seed_base + k
            if (seed, n) in done:
                continue
            a, extra = _val(seed, n)
            M = fe.assemble(n, a["a11"], a["a12"], a["a22"], a["sigma"])
            _loop, inner = fe.grid(n)
            b = (fe.mass(n, a["w"]) @ a["g"].ravel())[inner]
            t_exact = _best(lambda: _exact_solve(M, n, Q, b), COST_REPS)
            H, up, lev, strip_s = u_answers(U, a)
            t_u = _best(lambda: u_answers(U, a), COST_REPS)
            row = {"seed": seed, "n": n, "fo": extra["fo"], "exact_s": t_exact, "u_s": t_u,
                   "u_strips_s": strip_s, "u_over_exact": t_u / t_exact}
            for nc in args.ncs:
                if nc >= n:
                    continue
                t_c = _best(lambda nc=nc: co.correct_fields(M, n, nc, H, up, lev, a["s0"], b, Q), COST_REPS)
                row[f"coarse_{nc}_s"] = t_c
                row[f"ratio_{nc}"] = t_c / t_exact
            rec["rows"].append(row)
            save(path, rec)
        rec["power"].append({"when": f"n={n} end", **power_state()})
        save(path, rec)
    rec["on_ac_throughout"] = all(p["on_ac"] is True for p in rec["power"])
    rec["finished"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    save(path, rec)
    log(f"cost -> {os.path.relpath(path, ROOT)} (on AC throughout: {rec['on_ac_throughout']})")
    return [path]


# ---------------------------------------------------------------------------
# the predictions, read as registered
# ---------------------------------------------------------------------------


def _load_all(out: str) -> dict:
    recs = {}
    if not os.path.isdir(out):
        return recs
    for f in sorted(os.listdir(out)):
        if f.endswith(".json") and not f.startswith("verdicts"):
            recs[f[:-5]] = json.loads(registry._retry(lambda f=f: open(os.path.join(out, f), encoding="utf-8").read()))
    return recs


def _med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def judge(out: str) -> dict:
    recs = _load_all(out)
    modes = {k: r for k, r in recs.items() if k.startswith("modes-")}
    e4 = {k[3:]: r["result"] for k, r in recs.items() if k.startswith("e4-") and r.get("result")}
    cost = recs.get("cost-laptop")
    rows = [row for r in modes.values() for row in r["rows"]]
    tl = [f"TL0-{c}" for c in NCS]
    v = {}
    # 1
    bad = []
    for row in rows:
        for key in tl:
            t = row.get(key)
            if t is None:
                continue
            if t.get("cx") is not None and t["cx"] > 1e-12:
                bad.append(("cx", row["key"], key))
            if t.get("cx_p") is not None and t["cx_p"] > 1e-12:
                bad.append(("cx_p", row["key"], key))
            if t.get("check_violations"):
                bad.append(("column", row["key"], key))
            if t["e_K"] > 1e-12 or t["e_K_inf"] > 1e-12:
                bad.append(("K", row["key"], key))
        if row["n"] == 32 and row.get("set") == "validation" and row.get("TL0-32", {}).get("e_H", 1.0) > 1e-10:
            bad.append(("top", row["key"], "TL0-32"))
    for unit, res in e4.items():
        for frame, fr in res["frames"].items():
            for key, vr in fr["variants"].items():
                if not key.startswith("TL0") or "control" in key:
                    continue
                pc = vr.get("pieces", {})
                if any(x is not None and x > 1e-12 for x in pc.get("cx", []) + pc.get("cx_p", [])):
                    bad.append(("cx bench", unit, frame, key))
                if any(x > 1e-12 for x in pc.get("e_K", []) + pc.get("e_K_inf", [])):
                    bad.append(("K bench", unit, key))
                if sum(pc.get("check_violations", [])):
                    bad.append(("column bench", unit, frame, key))
            topv = [vr for key, vr in fr["variants"].items() if "control" in key]
            if not topv or abs(topv[0]["energy_rel"] - fr["floor_energy_rel"]) > 1e-10:
                bad.append(("top bench", unit, frame))
    v["1"] = {"held": not bad and bool(rows) and len(e4) == 6, "failures": bad[:20],
              "failures_count": len(bad), "pieces": len(rows), "problems": len(e4)}
    # 2
    b3 = e4.get("bend-3-steady")
    if b3:
        met = {k: b3["verdicts"][k]["e4"]["met"] for k in tl}
        v["2"] = {"held": all(met.values()), "met": met}
    # 3
    by_n = {}
    for n in NS:
        val = [r for r in rows if r.get("set") == "validation" and r["n"] == n]
        if val:
            mu, mt = _med([r["U"]["low"] for r in val]), _med([r["TL0-16"]["low"] for r in val])
            by_n[n] = {"U": mu, "TL0-16": mt, "ratio": mt / mu if mu else None, "pieces": len(val)}
    if len(by_n) == len(NS):
        v["3"] = {"held": all(x["ratio"] is not None and x["ratio"] <= 0.5 for x in by_n.values()), "by_n": by_n}
    # 4
    if all(k in e4 for k in ("s-channel-steady", "insert-round-steady")):
        def energy_met(res, key):
            return all(res["frames"][f]["variants"][key]["energy_rel"] <= GF.E4_ENERGY for f in res["frames"])
        per = {k: {nc: energy_met(e4[k], f"TL0-{nc}") for nc in (16, 32)}
               for k in ("s-channel-steady", "insert-round-steady")}
        v["4"] = {"held": all(any(d.values()) for d in per.values()), "met": per}
    # 5
    steps = [f"{b}-step" for b in GF.HELD_OUT_BENCHMARKS]
    if all(s in e4 for s in steps):
        maxmet = {}
        for s in steps:
            ver = e4[s]["verdicts"]["TL0-16"]["e4"]
            maxmet[s] = all(ver[f]["max_over_range"] <= ver[f]["bar_max_over_range"]
                            for f in ("as registered", "lowered by 300 K"))
        v["5"] = {"held": not all(maxmet.values()), "max_bar_met": maxmet}
    # 6
    if cost:
        r128 = [r["ratio_16"] for r in cost["rows"] if r["n"] == 128 and "ratio_16" in r]
        judged = bool(cost.get("on_ac_throughout")) and bool(r128)
        med = _med(r128)
        v["6"] = {"held": (med < 0.1) if judged else None, "judged": judged, "median": med,
                  "pieces": len(r128)}
    # the outcome's inputs: E4 per n_c on all six
    six = {}
    if len(e4) == 6:
        for k in tl + ["U"]:
            six[k] = {u: res["verdicts"][k]["e4"]["met"] for u, res in e4.items()}
    return {"predictions": v, "e4_by_variant": six,
            "smallest_nc_meeting_e4_on_all_six": next(
                (k for k in tl if six and all(six[k].values())), None)}


def write_verdicts(out: str) -> int:
    res = judge(out)
    res["written"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    save(os.path.join(out, "verdicts.json"), res)
    lines = ["TL0 (Phase 2), the predictions as registered:"]
    for p in PREDICTIONS:
        r = res["predictions"].get(p["id"])
        state = "not measured" if r is None else ("not judged" if r.get("held") is None
                                                  else ("held" if r["held"] else "REFUTED"))
        lines.append(f"  {p['id']}. {state}")
    lines.append(f"  E4 on all six first met at: {res['smallest_nc_meeting_e4_on_all_six']}")
    txt = "\n".join(lines)
    registry.write_atomic(os.path.join(out, "verdicts.txt"), (txt + "\n").encode("utf-8"))
    log(txt)
    return 0


# ---------------------------------------------------------------------------
# the smoke run: every code path, controls only
# ---------------------------------------------------------------------------


def smoke_report(paths: list) -> None:
    for path in paths:
        rec = json.loads(open(path, encoding="utf-8").read())
        if rec["config"]["part"] == "modes":
            tops = [r["TL0-32"]["e_H"] for r in rec["rows"] if r["n"] == 32 and "TL0-32" in r]
            cx = max(t.get("cx") or 0.0 for r in rec["rows"] for k, t in r.items()
                     if k.startswith("TL0-") and isinstance(t, dict))
            viol = sum(t.get("check_violations", 0) for r in rec["rows"] for k, t in r.items()
                       if k.startswith("TL0-") and isinstance(t, dict))
            ek = max(t["e_K_inf"] for r in rec["rows"] for k, t in r.items()
                     if k.startswith("TL0-") and isinstance(t, dict))
            log(f"  smoke modes {rec['config']['unit']}: top e_H max {max(tops) if tops else None}, "
                f"CX max {cx:.1e}, column violations {viol}, e_K,inf max {ek:.1e}")
        elif rec["config"]["part"] == "e4":
            res = rec["result"]
            for f, fr in res["frames"].items():
                top = [vr for k, vr in fr["variants"].items() if "control" in k][0]
                split = [vr for vr in fr["variants"].values() if vr.get("split") is not None]
                cx = max([x for vr in fr["variants"].values() for x in vr["pieces"]["cx"] + vr["pieces"]["cx_p"]
                          if x is not None] or [0.0])
                ek = max([x for k, vr in fr["variants"].items() if k.startswith("TL0")
                          for x in vr["pieces"]["e_K_inf"]] or [0.0])
                log(f"  smoke e4 {res['problem']} {f}: none = U bit for bit {fr['control_none_bit_for_bit']}, "
                    f"top - floor {top['energy_rel'] - fr['floor_energy_rel']:.1e}, CX max {cx:.1e}, "
                    f"TL0 e_K,inf max {ek:.1e}, where-check "
                    f"{max([vr['where']['check_rel'] for vr in split] or [0.0]):.1e}, split-check "
                    f"{max([vr['split_check_rel'] for vr in split] or [0.0]):.1e}")
        else:
            log(f"  smoke cost: {len(rec['rows'])} rows")


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--part", choices=("modes", "e4", "cost"))
    ap.add_argument("--judge", action="store_true", help="read the registered predictions")
    ap.add_argument("--ns", default=",".join(map(str, NS)))
    ap.add_argument("--ncs", default=",".join(map(str, NCS)))
    ap.add_argument("--count", type=int, default=VAL_COUNT)
    ap.add_argument("--cost-count", type=int, default=COST_COUNT)
    ap.add_argument("--problem", default=None, help="one benchmark problem, e.g. s-channel:steady")
    ap.add_argument("--bench-n", type=int, default=BENCH_N)
    ap.add_argument("--seed-base", type=int, default=fd.VAL_POSTHOC)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-battery", action="store_true", help="the cost part off AC (smoke only)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.ns = tuple(int(x) for x in args.ns.split(","))
    args.ncs = tuple(int(x) for x in args.ncs.split(","))
    if args.judge:
        return write_verdicts(args.out)
    if args.part is None:
        raise SystemExit("--part or --judge")
    if args.smoke:
        if args.out == OUT:
            raise SystemExit("--smoke writes to a scratch --out, never the registered record")
        args.seed_base, args.count, args.cost_count, args.bench_n = SMOKE_SEED, 2, 1, 16
        args.ns = tuple(n for n in args.ns if n <= 32) or (16,)
    elif args.seed_base != fd.VAL_POSTHOC or args.out != OUT:
        raise SystemExit("the registered run uses the post-hoc validation seeds and the registered record")
    torch.set_num_threads(THREADS)
    U = fd.zero_heads()
    paths = {"modes": part_modes, "e4": part_e4, "cost": part_cost}[args.part](args, U)
    if args.smoke:
        smoke_report(paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
