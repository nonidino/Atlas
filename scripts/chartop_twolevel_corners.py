"""Branch 2c of the Stage 0 rework: the corner treatment, measured untrained.

[[chart-operator-stage-0-rework]] section 15 (the draft, accepted on 2026-11-09) and section
16 (this script, the predictions' readings).

**The arms.**  TL0 is U (`chartop_forms_diagnose.zero_heads`) followed by the coarse
correction of `atlas.chartop.coarse` (`correct_fields`).  The treatments enlarge its coarse
space at each piece's non-right corners (`coarse.CornerTreatment`, more than 2 degrees from
90): **A**, ``K`` singular functions of the corner's frozen tensor per corner (``K = 1, 2,
3``); **B**, the corner block's fine hats beside the coarse grid (blocks of ``n/8`` and
``n/16``).  At ``n_c = 16``: TL0 and all five.  For the curve, at ``n_c = 8`` and 32: TL0,
A with ``K = 3`` and both B.  At ``n_c = n`` every treatment adds nothing (the control).
Nothing is trained, no weights are read, and the test split is never drawn.

    python scripts/chartop_twolevel_corners.py --part e4 --problem s-channel:steady
    python scripts/chartop_twolevel_corners.py --part modes --ns 32      # one resolution per process
    python scripts/chartop_twolevel_corners.py --part cost               # on AC, alone
    python scripts/chartop_twolevel_corners.py --judge                   # the registered predictions

**Parts**, each writing ``out/chartop/twolevel/corners/<part>-<unit>.json`` after every
measurement (a stopped run resumes where it stopped):
- ``e4``: the six Stage 0 benchmarks (n = 64) in both frames.  Per arm: E4, D and K by
  `gates_forms`' own judges; the excess over the 16-mode floor, ``||u_m - u~||_M / ||u||_M``
  (with the check that ``e^2 = floor^2 + excess^2``); the network term split by mode group
  (A1's `split_term`) and by place (Phase 2's regions); per piece, CX on the arm's own space
  (``P_e``), the column-by-column check against TL0 at the same ``n_c`` (TL0's own against
  U), K through the uniform state, and, where a piece has no non-right corner, its answers
  against TL0's bit for bit.  At ``n_c = n``, each treatment's answers against TL0's.
- ``modes``: Phase 2's pieces (the post-hoc validation pieces 2,000,100 onward at n = 32, 64
  and 128; W362's piece types).  Per piece and arm: ``e_H`` by column group, ``e_H``,
  ``e_H,L``, ``e_p``, ``e_1``, ``e_K``, CX on the arm's own space and the column check, as
  above; the piece's corner angles, frozen wedges and non-right corners.
- ``cost``: at n = 128, ``n_c = 16``: the coarse level of TL0, A with ``K = 3`` and both B
  (building the columns included) against `fe.Exact` (the LU, 64 mode columns, the particular
  field), assembly excluded, best of 5.  On the first 16 validation pieces and the
  s-channel's two end pieces (the pieces with two non-right corners it was drafted for).
  Torch and the BLAS at 4 threads; the power sampled before and after; the part refuses to
  start on battery.

**The predictions** (`PREDICTIONS`) are section 15's, each with the reading `judge` applies
(section 16).  They were registered, with the owner's go (`REGISTERED`), before any part ran
on the pieces above; while `REGISTERED` is ``None`` only ``--smoke`` runs.  ``--smoke`` runs every
code path on seeds and a layout no registration draws (validation-style seeds from
9,000,100 at n = 16 and 32; `benchmarks_squares` seed 9,000,008, which has eight pieces with
a non-right corner, at n = 32), into a scratch directory, and prints only the controls.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
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
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

import chartop_evaluate as ev0  # noqa: E402
import chartop_forms_diagnose as fd  # noqa: E402
import chartop_registry as registry  # noqa: E402
import chartop_twolevel_controls as tc  # noqa: E402
import chartop_w362_diagnose as w362  # noqa: E402
from atlas.chartop import benchmarks as bm, benchmarks_squares as BS, chart as ch  # noqa: E402
from atlas.chartop import coarse as co, data, fe, forms as fm, gates_forms as GF  # noqa: E402
from atlas.chartop import host as hs, ports as pt  # noqa: E402
from atlas.chartop.expert import LearnedExpert  # noqa: E402

OUT = os.path.join(ROOT, "out", "chartop", "twolevel", "corners")
KIND, M_PORTS = GF.KIND, GF.M_PORTS
NS = (32, 64, 128)
VAL_COUNT = 64
BENCH_N = GF.BENCHMARK_N
MAIN_NC = 16
CURVE_NCS = (8, 32)
TREATMENTS = {t.name: t for t in (co.CornerTreatment("A", K=1), co.CornerTreatment("A", K=2),
                                  co.CornerTreatment("A", K=3), co.CornerTreatment("B", frac=8),
                                  co.CornerTreatment("B", frac=16))}
CURVE_TREATMENTS = ("A-K3", "B-n/8", "B-n/16")
#: the predictions' arms
A3, B8 = "A-K3", "B-n/8"
COST_N, COST_COUNT, COST_REPS = 128, 16, 5
COST_TREATMENTS = (None, "A-K3", "B-n/8", "B-n/16")
#: the excess at which the s-channel steady problem meets E4's energy bar over its floor
#: (section 13: sqrt(2^2 - 0.93^2) = 1.77 %)
EXCESS_BAR = 0.0177
CORNER_SHARE_BAR = 0.5
COST_BAR = 0.15
TOL = 1e-12                          # CX and K, as Phase 2
TOP_EH = 1e-10                       # n_c = n: exact
SMOKE_SEED, SMOKE_LAYOUT_SEED, SMOKE_BENCH_N = 9_000_100, 9_000_008, 32

REGISTERED = ("registered by the owner on 2026-11-10, in chat ('Register as drafted (Recommended)'), "
              "as drafted in the rework page's section 15 with the readings of its section 16, "
              "before any part ran on the pieces it measures")

PREDICTIONS = (
    {"id": "1",
     "text": "The controls and Proposition A hold for both enlarged spaces, to rounding: CX at most "
             "1e-12 on V_e, no column worse than TL0's, TL0 bit for bit where no corner is non-right, "
             "n_c = n exact.",
     "reading": "For every treatment (A with K = 1, 2, 3; B with n/8 and n/16 blocks) at every n_c "
                "measured, on every piece of the modes part and every benchmark piece in both frames: "
                "where n_c < n, CX on the arm's own space of the modes <= 1e-12, and of the particular "
                "field <= 1e-12 wherever the piece has a load; no column whose energy-norm error "
                "exceeds TL0's at the same n_c by more than 1e-9 of TL0's plus 1e-12 of the exact "
                "column's own energy norm; e_K and e_K,inf <= 1e-12 (the level-field identity on every "
                "piece of the modes part, the uniform state through the host on every benchmark piece "
                "in the registered frame).  On every piece with no non-right corner, each treatment's "
                "answers equal TL0's bit for bit, and on bend-3 every treatment's energy and maximum "
                "errors equal TL0's.  At n_c = n every treatment adds no column and its answers equal "
                "TL0's bit for bit, and e_H <= 1e-10 on every n = 32 validation piece."},
    {"id": "2",
     "text": "[AI Inference] B with n/8 blocks brings the s-channel steady problem's excess below "
             "1.77% at n_c = 16: E4 met on 5 of 6 at n_c = 16, all but the s-channel step lowered by "
             "300 K, which no 16-mode expert can meet.",
     "reading": "For B with n/8 blocks at n_c = 16: the s-channel steady problem's excess over the "
                "16-mode floor is below 0.0177 in both frames, and judge_e4_both_frames is met on each "
                "of the five problems other than the s-channel step."},
    {"id": "3",
     "text": "[AI Inference] A gains less than B on the s-channel: at K = 3 its excess there stays "
             "above B's.",
     "reading": "At n_c = 16, A with K = 3 has a larger excess over the floor than B with n/8 blocks, "
                "on the s-channel steady problem and on the s-channel step, in both frames."},
    {"id": "4",
     "text": "[AI Inference] With B, the non-right corner blocks hold less than half of the "
             "s-channel's remaining excess.",
     "reading": "For B with n/8 blocks at n_c = 16, the split by place (Phase 2's regions) puts less "
                "than 0.5 of the network term in the non-right corner blocks, on the s-channel steady "
                "problem and on the s-channel step, in both frames; a frame whose network term is "
                "below the split floor (1e-9 of the reference energy) has no share and counts as held."},
    {"id": "5",
     "text": "[AI Inference] B's coarse level at n = 128, n_c = 16, costs under 0.15 of an exact solve "
             "per piece with two non-right corners (TL0's was 0.050).",
     "reading": "The median, over the cost part's pieces with exactly two non-right corners, of the "
                "coarse level's time for B with n/8 blocks at n = 128, n_c = 16 (building its columns "
                "included) over the exact solve's time is < 0.15; judged only if every power sample "
                "read AC."},
)


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# provenance and records (Phase 2's)
# ---------------------------------------------------------------------------


def provenance() -> dict:
    out = tc.provenance()
    out.update(script_sha256=registry.sha256_file(os.path.abspath(__file__)), registered=REGISTERED)
    return out


def record_path(out: str, part: str, unit: str) -> str:
    return os.path.join(out, f"{part}-{unit}.json")


def load_record(path: str, config: dict) -> dict:
    if os.path.exists(path):
        rec = json.loads(registry._retry(lambda: open(path, encoding="utf-8").read()))
        if rec.get("config") != config:
            raise SystemExit(f"{path} holds another configuration: move it away to start again")
        return rec
    return {"config": config, "rows": [], "provenance": provenance()}


save = tc.save


# ---------------------------------------------------------------------------
# the arms
# ---------------------------------------------------------------------------


def arm_key(tname: str | None, nc: int) -> str:
    return f"{tname or 'TL0'}, n_c={nc}"


def arms(n: int) -> list:
    """``(key, treatment or None, n_c)`` for every arm with ``n_c < n``, TL0 before the
    treatments at each size: at 16 all five, at 8 and 32 `CURVE_TREATMENTS`."""
    out = []
    for nc in sorted({MAIN_NC, *CURVE_NCS}):
        if nc >= n:
            continue
        names = list(TREATMENTS) if nc == MAIN_NC else list(CURVE_TREATMENTS)
        out.append((arm_key(None, nc), None, nc))
        out += [(arm_key(t, nc), TREATMENTS[t], nc) for t in names]
    return out


def space_P(n: int, nc: int, S):
    """The arm's ``P`` (``P_e`` with columns)."""
    P = co.prolongation(n, nc)
    if S is not None and S.shape[1]:
        P = sp.hstack([P, S], format="csr")
    return P


def _same(a, b) -> bool:
    """Answers ``(H, up, level)`` equal bit for bit (a missing level field on both)."""
    return all((x is None and y is None) or (x is not None and y is not None and np.array_equal(x, y))
               for x, y in zip(a, b))


def corner_info(X, a11, a12, a22) -> dict:
    ang = ch.corner_angles(np.asarray(X))
    tens = co.corner_tensors(a11, a12, a22)
    wedges = []
    for c in range(4):
        try:
            wedges.append(math.degrees(co.corner_wedge(*tens[c], c)[0]))
        except ValueError:
            wedges.append(None)
    return {"corner_angles": ang.tolist(), "frozen_wedges": wedges,
            "non_right": list(co.non_right_corners(X))}


# ---------------------------------------------------------------------------
# the modes part
# ---------------------------------------------------------------------------


def cx_on_space(x, n: int, nc: int, S, H, up) -> dict:
    if nc >= n:
        return {}
    P = space_P(n, nc, S)
    out = {"cx_nc": nc, "cx": co.coarse_exactness(P, co.residual(x.M, n, H, x.Q), x.BQ)["rel"]}
    cp = co.coarse_exactness(P, co.residual(x.M, n, up, rhs=x.b), x.b)
    out["cx_p"], out["cx_p_abs"] = cp["rel"], cp["num"]
    return out


def measure_piece(U, a: dict, X, meta: dict) -> dict:
    """U, TL0 and every treatment (`arms`) on one piece; at n = 32 also the top."""
    n = a["a11"].shape[-1]
    Q = pt.basis(KIND, n, M_PORTS).Q
    c1 = fm.kernel_coefficients(KIND, n, M_PORTS)
    x = tc.Exact(a, Q)
    H, up, lev, strip_s = tc.u_answers(U, a)
    rowU, errU = tc.measures(x, H, up, lev, c1, None)
    out = dict(meta, n=n, s0=a["s0"], trace_Lambda=float(x.lam.sum()), U=rowU, u_strip_s=strip_s,
               **corner_info(X, a["a11"], a["a12"], a["a22"]))
    tl0 = {}
    for key, tr, nc in arms(n):
        S = None if tr is None else tr.columns(n, nc, X, a["a11"], a["a12"], a["a22"])
        t0 = time.perf_counter()
        ans = co.correct_fields(x.M, n, nc, H, up, lev, a["s0"], x.b, Q, extra=S)
        dt_c = time.perf_counter() - t0
        row, err = tc.measures(x, *ans, c1, None, err_U=errU if tr is None else tl0[nc][1])
        row.update(cx_on_space(x, n, nc, S, ans[0], ans[1]))
        row["check_against"] = "U" if tr is None else "TL0"
        row["extra"] = 0 if S is None else int(S.shape[1])
        row["coarse_s"] = dt_c
        if tr is None:
            tl0[nc] = (ans, err)
        elif row["extra"] == 0:
            row["bit_for_bit_tl0"] = _same(ans, tl0[nc][0])
        out[key] = row
    if n == 32:
        top = co.correct_fields(x.M, n, n, H, up, lev, a["s0"], x.b, Q)
        rowT, _ = tc.measures(x, *top, c1, None)
        out["top"] = {"e_H": rowT["e_H"], "treatments": {}}
        for name, tr in TREATMENTS.items():
            S = tr.columns(n, n, X, a["a11"], a["a12"], a["a22"])
            ans = co.correct_fields(x.M, n, n, H, up, lev, a["s0"], x.b, Q, extra=S)
            out["top"]["treatments"][name] = {"extra": int(S.shape[1]), "bit_for_bit": _same(ans, top)}
    return out


def _val(seed: int, n: int):
    ex = data.draw(seed, n=n)
    a, meta = tc.arrays_of_example(ex), {"fo": ex.fo, "freq": ex.freq, "family": ex.family}
    return a, meta, np.asarray(ex.problem.chart.X)


def modes_units(n: int, count: int, seed_base: int, w362_types: bool):
    yield f"val-n{n}", [({"set": "validation", "seed": seed_base + k}, (lambda s=seed_base + k: _val(s, n)))
                        for k in range(count)]
    if w362_types:
        items = []
        for typ, plist in w362.groups(n).items():
            for label, p in plist:
                items.append(({"set": "w362", "type": typ, "label": label},
                              (lambda p=p: (tc.arrays_of_problem(p), {}, np.asarray(p.chart.X)))))
        yield f"w362-n{n}", items


def part_modes(args, U) -> list:
    paths = []
    for n in args.ns:
        for unit, items in modes_units(n, args.count, args.seed_base, not args.smoke):
            config = {"part": "modes", "unit": unit, "count": args.count, "seed_base": args.seed_base}
            path = record_path(args.out, "modes", unit)
            rec = load_record(path, config)
            done = {json.dumps(r["key"], sort_keys=True) for r in rec["rows"]}
            for meta, make in items:
                key = {k: meta[k] for k in ("set", "seed", "label") if k in meta}
                if json.dumps(key, sort_keys=True) in done:
                    continue
                t0 = time.perf_counter()
                a, extra, X = make()
                row = measure_piece(U, a, X, dict(meta, **extra))
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


def corrected(problems, resp, pb_of, nc: int, tr):
    """The arm's answers from U's, piece by piece, and each piece's added columns."""
    out, extras = [], []
    for p, r in zip(problems, resp):
        _loop, inner = fe.grid(p.n)
        S = None if tr is None else tr.of_problem(p, nc)
        H, up, lev = co.correct_fields(p.matrix(), p.n, nc, r.H, r.up, r.level, p.s0, p.load()[inner],
                                       pb_of(p.n).Q, extra=S)
        out.append(hs.Response(H, up, lev))
        extras.append(S)
    return out, extras


def piece_checks(L, resp, extras, ures, cl, nc: int, ref) -> dict:
    """Per piece: CX on the arm's own space, the column check against ``ref``'s answers,
    K through the uniform state (``ures``), and the added columns."""
    out = {"cx": [], "cx_p": [], "e_K": [], "e_K_inf": [], "check_violations": [], "extra": []}
    for i, p in enumerate(L.problems):
        n, Q = p.n, L.pb(p.n).Q
        M = p.matrix().tocsr()
        loop, inner = fe.grid(n)
        S = extras[i]
        out["extra"].append(0 if S is None else int(S.shape[1]))
        if nc < n:
            P = space_P(n, nc, S)
            BQ = -np.asarray(M[inner][:, loop] @ Q)
            out["cx"].append(co.coarse_exactness(P, co.residual(M, n, resp[i].H, Q), BQ)["rel"])
            b = p.load()[inner]
            out["cx_p"].append(co.coarse_exactness(P, co.residual(M, n, resp[i].up, rhs=b), b)["rel"])
        out["check_violations"].append(tc.column_violations(M[inner][:, inner], resp[i].H, ref[i].H,
                                                            cl.responses[i].H, np.diag(cl.grams[i])))
        if ures is not None:
            r = ures[i].up + ures[i].H @ fm.kernel_coefficients(KIND, n, M_PORTS) - 1.0
            km = fm.kernel_measures(r, M[inner][:, inner], float(np.trace(cl.grams[i])))
            out["e_K"].append(km["e_K"])
            out["e_K_inf"].append(km["e_K_inf"])
    return out


def e4_unit(name: str, problem: str, U_le, n: int = BENCH_N, smoke: bool = False) -> dict:
    if smoke:
        b = BS.build(BS.draw(SMOKE_LAYOUT_SEED), n, problem)
    else:
        kw = {"envelope": data.ENVELOPE} if name == "s-channel" else {}
        b = bm.build(name, n, problem, **kw)
    frames = {"registered": b, "lowered": bm.datum_shift(b, GF.DATUM_SHIFT_K)}
    rec = {"name": name, "problem": problem, "n": n, "pieces": len(b.problems), "frames": {}}
    for frame, bb in frames.items():
        L = bb.layout(KIND, M_PORTS)
        mono = hs.monolithic(L)
        cl = L.solve(hs.ClassicalExpert())
        un, ud = cl.energy_error(mono.fields)
        mx0, span = ev0.field_errors(mono.fields, cl.fields)
        fr = {"floor_energy_rel": un / ud, "floor_abs": un, "reference_energy": ud, "floor_max": mx0,
              "span": span, "refusals": L.check_envelope(data.ENVELOPE),
              "non_right": [list(co.non_right_corners(p.chart.X)) for p in L.problems], "variants": {}}
        respU = U_le.respond(L.problems, L.pb)
        ups = [fm.uniform_state(p) for p in L.problems] if frame == "registered" else None
        uresU = U_le.respond(ups, L.pb) if ups else None
        tl0 = {}
        for key, tr, nc in arms(n):
            resp, extras = corrected(L.problems, respU, L.pb, nc, tr)
            ures = corrected(ups, uresU, L.pb, nc, tr)[0] if ups else None
            cp = L.couple(resp, calls=len(L.problems))
            num, den = cp.energy_error(mono.fields)
            exc, _ = cp.energy_error(cl.fields)
            mx, _ = ev0.field_errors(mono.fields, cp.fields)
            row = {"energy_abs": num, "energy_rel": num / den, "excess_rel": exc / ud,
                   "pythagoras_rel": abs(num * num - un * un - exc * exc) / max(num * num, 1e-300),
                   "max_err": mx, "max_over_range": mx / span, "min_eig": cp.min_eig(),
                   "network_term": ev0.network_term(L, cp, cl)}
            if row["network_term"] > tc.SPLIT_FLOOR * ud:
                st = fd.split_term(L, cp, cl)
                row.update(split_check_rel=st["check_rel"], split=st["share"],
                           split_alone=st["alone_over_total"], where=tc.where_split(L, cp, cl))
            else:
                row["split"] = None                      # at the floor: no shares are formed
            row["pieces"] = piece_checks(L, resp, extras, ures, cl, nc, respU if tr is None else tl0[nc])
            if tr is None:
                tl0[nc] = resp
            else:
                row["pieces"]["bit_for_bit_tl0"] = [_same((r.H, r.up, r.level), (q.H, q.up, q.level))
                                                    for r, q, S in zip(resp, tl0[nc], extras)
                                                    if S is None or S.shape[1] == 0]
                ref = fr["variants"][arm_key(None, nc)]
                row["same_errors_as_tl0"] = bool(row["energy_abs"] == ref["energy_abs"]
                                                 and row["max_err"] == ref["max_err"])
            fr["variants"][key] = row
        top, _ = corrected(L.problems, respU, L.pb, n, None)
        fr["top"] = {}
        for tname, tr in TREATMENTS.items():
            resp, extras = corrected(L.problems, respU, L.pb, n, tr)
            fr["top"][tname] = {"extra": int(sum(0 if S is None else S.shape[1] for S in extras)),
                                "bit_for_bit": all(_same((r.H, r.up, r.level), (q.H, q.up, q.level))
                                                   for r, q in zip(resp, top))}
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
        rec = load_record(path, {"part": "e4", "unit": unit, "n": args.bench_n})
        if rec.get("result") is None:
            t0 = time.perf_counter()
            rec["result"] = e4_unit(name, problem, U_le, args.bench_n, args.smoke)
            rec["elapsed_s"] = time.perf_counter() - t0
            rec["finished"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
            save(path, rec)
        log(f"e4 {unit} -> {os.path.relpath(path, ROOT)}")
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# the cost part
# ---------------------------------------------------------------------------


def cost_items(args):
    """``(meta, maker)``: the validation pieces, then (registered run) the s-channel's end
    pieces at n = 128, the pieces with non-right corners."""
    n = args.cost_n
    for k in range(args.cost_count):
        yield {"set": "validation", "seed": args.seed_base + k}, (lambda s=args.seed_base + k: _val(s, n)[::2])
    if not args.smoke:
        b = bm.build("s-channel", n, "steady", envelope=data.ENVELOPE)
        for i, p in enumerate(b.problems):
            if co.non_right_corners(p.chart.X):
                yield {"set": "s-channel", "piece": i}, (lambda p=p: (tc.arrays_of_problem(p), np.asarray(p.chart.X)))


def part_cost(args, U) -> list:
    path = record_path(args.out, "cost", "laptop")
    config = {"part": "cost", "n": args.cost_n, "n_c": MAIN_NC, "count": args.cost_count, "reps": COST_REPS,
              "seed_base": args.seed_base, "treatments": [t or "TL0" for t in COST_TREATMENTS]}
    rec = load_record(path, config)
    start = tc.power_state()
    if start["on_ac"] is not True and not args.allow_battery:
        raise SystemExit(f"the cost part runs on AC power only (power state {start['raw']!r})")
    rec.setdefault("power", []).append({"when": "start", **start})
    rec.setdefault("python_processes", []).append(tc.python_processes())
    done = {json.dumps(r["key"], sort_keys=True) for r in rec["rows"]}
    n, nc = args.cost_n, MAIN_NC
    Q = pt.basis(KIND, n, M_PORTS).Q
    for meta, make in cost_items(args):
        if json.dumps(meta, sort_keys=True) in done:
            continue
        a, X = make()
        M = fe.assemble(n, a["a11"], a["a12"], a["a22"], a["sigma"])
        _loop, inner = fe.grid(n)
        b = (fe.mass(n, a["w"]) @ a["g"].ravel())[inner]
        t_exact = tc._best(lambda: tc._exact_solve(M, n, Q, b), COST_REPS)
        H, up, lev, _s = tc.u_answers(U, a)
        row = {"key": meta, "n": n, "non_right": len(co.non_right_corners(X)), "exact_s": t_exact}
        for tname in COST_TREATMENTS:
            tr = None if tname is None else TREATMENTS[tname]

            def run(tr=tr):
                S = None if tr is None else tr.columns(n, nc, X, a["a11"], a["a12"], a["a22"])
                co.correct_fields(M, n, nc, H, up, lev, a["s0"], b, Q, extra=S)
            t_c = tc._best(run, COST_REPS)
            name = tname or "TL0"
            row[f"{name}_s"] = t_c
            row[f"{name}_ratio"] = t_c / t_exact
            row[f"{name}_extra"] = 0 if tr is None else int(tr.columns(n, nc, X, a["a11"], a["a12"],
                                                                       a["a22"]).shape[1])
        rec["rows"].append(row)
        save(path, rec)
    rec["power"].append({"when": "end", **tc.power_state()})
    rec["python_processes"].append(tc.python_processes())
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


def _treatment_keys(row: dict) -> list:
    return [k for k in row if ", n_c=" in k and not k.startswith("TL0")]


def judge(out: str) -> dict:
    recs = _load_all(out)
    modes = {k: r for k, r in recs.items() if k.startswith("modes-")}
    e4 = {k[3:]: r["result"] for k, r in recs.items() if k.startswith("e4-") and r.get("result")}
    cost = recs.get("cost-laptop")
    rows = [row for r in modes.values() for row in r["rows"]]
    six = [f"{b}-{p}" for b in GF.HELD_OUT_BENCHMARKS for p in GF.BENCHMARK_PROBLEMS]
    v = {}
    # 1
    bad = []
    for row in rows:
        for key in _treatment_keys(row):
            t = row[key]
            if t.get("cx") is not None and t["cx"] > TOL:
                bad.append(("cx", row["key"], key))
            if t.get("cx_p") is not None and t["cx_p"] > TOL:
                bad.append(("cx_p", row["key"], key))
            if t.get("check_violations"):
                bad.append(("column", row["key"], key))
            if t["e_K"] > TOL or t["e_K_inf"] > TOL:
                bad.append(("K", row["key"], key))
            if t["extra"] == 0 and t.get("bit_for_bit_tl0") is not True:
                bad.append(("bit for bit", row["key"], key))
        if row["n"] == 32:
            top = row.get("top")
            if top is None or any(x["extra"] or not x["bit_for_bit"] for x in top["treatments"].values()):
                bad.append(("top", row["key"]))
            elif row.get("set") == "validation" and top["e_H"] > TOP_EH:
                bad.append(("top e_H", row["key"]))
    for unit, res in e4.items():
        for frame, fr in res["frames"].items():
            for key, vr in fr["variants"].items():
                if key.startswith("TL0"):
                    continue
                pc = vr["pieces"]
                if any(x is not None and x > TOL for x in pc["cx"] + pc["cx_p"]):
                    bad.append(("cx bench", unit, frame, key))
                if any(x > TOL for x in pc["e_K"] + pc["e_K_inf"]):
                    bad.append(("K bench", unit, key))
                if sum(pc["check_violations"]):
                    bad.append(("column bench", unit, frame, key))
                if not all(pc.get("bit_for_bit_tl0", [])):
                    bad.append(("bit for bit bench", unit, frame, key))
                if unit.startswith("bend-3") and not vr.get("same_errors_as_tl0"):
                    bad.append(("bend-3", unit, frame, key))
            top = fr.get("top", {})
            if set(top) != set(TREATMENTS) or any(x["extra"] or not x["bit_for_bit"] for x in top.values()):
                bad.append(("top bench", unit, frame))
    v["1"] = {"held": not bad and bool(rows) and sorted(e4) == sorted(six), "failures": bad[:20],
              "failures_count": len(bad), "pieces": len(rows), "problems": len(e4)}
    k8, k3 = arm_key(B8, MAIN_NC), arm_key(A3, MAIN_NC)
    sch = ("s-channel-steady", "s-channel-step")
    # 2
    if sorted(e4) == sorted(six):
        exc = {f: e4["s-channel-steady"]["frames"][f]["variants"][k8]["excess_rel"]
               for f in e4["s-channel-steady"]["frames"]}
        met = {u: e4[u]["verdicts"][k8]["e4"]["met"] for u in six if u != "s-channel-step"}
        v["2"] = {"held": all(x < EXCESS_BAR for x in exc.values()) and all(met.values()),
                  "excess": exc, "met": met}
    # 3 and 4
    if all(u in e4 for u in sch):
        pairs, shares = {}, {}
        for u in sch:
            for f, fr in e4[u]["frames"].items():
                a3, b8 = fr["variants"][k3]["excess_rel"], fr["variants"][k8]["excess_rel"]
                pairs[f"{u}, {f}"] = {"A-K3": a3, "B-n/8": b8, "held": a3 > b8}
                w = fr["variants"][k8].get("where")
                share = None if fr["variants"][k8].get("split") is None else w["share"]["corner, not right"]
                shares[f"{u}, {f}"] = share
        v["3"] = {"held": all(p["held"] for p in pairs.values()), "excess": pairs}
        v["4"] = {"held": all(s is None or s < CORNER_SHARE_BAR for s in shares.values()),
                  "corner_not_right_share": shares}
    # 5
    if cost:
        two = [r[f"{B8}_ratio"] for r in cost["rows"] if r["non_right"] == 2]
        judged = bool(cost.get("on_ac_throughout")) and bool(two)
        med = _med(two)
        v["5"] = {"held": (med < COST_BAR) if judged else None, "judged": judged, "median": med,
                  "pieces": len(two),
                  "TL0_median_same_pieces": _med([r["TL0_ratio"] for r in cost["rows"] if r["non_right"] == 2])}
    e4_met = {}
    if sorted(e4) == sorted(six):
        for key in e4[six[0]]["verdicts"]:
            e4_met[key] = {u: e4[u]["verdicts"][key]["e4"]["met"] for u in six}
    return {"predictions": v, "e4_by_arm": e4_met}


def write_verdicts(out: str) -> int:
    res = judge(out)
    res["written"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    save(os.path.join(out, "verdicts.json"), res)
    lines = ["Branch 2c (the corner treatment), the predictions as registered:"]
    for p in PREDICTIONS:
        r = res["predictions"].get(p["id"])
        state = "not measured" if r is None else ("not judged" if r.get("held") is None
                                                  else ("held" if r["held"] else "REFUTED"))
        lines.append(f"  {p['id']}. {state}")
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
        part = rec["config"]["part"]
        if part == "modes":
            ts = [(r, k) for r in rec["rows"] for k in _treatment_keys(r)]
            cx = max([r[k].get("cx") or 0.0 for r, k in ts] or [0.0])
            viol = sum(r[k].get("check_violations", 0) for r, k in ts)
            ek = max([r[k]["e_K_inf"] for r, k in ts] or [0.0])
            bfb = all(r[k].get("bit_for_bit_tl0", True) for r, k in ts)
            extra = sum(r[k]["extra"] for r, k in ts)
            tops = [r["top"] for r in rec["rows"] if "top" in r]
            top_ok = all(not x["extra"] and x["bit_for_bit"] for t in tops for x in t["treatments"].values())
            log(f"  smoke modes {rec['config']['unit']}: treatments' CX max {cx:.1e}, column violations "
                f"{viol}, e_K,inf max {ek:.1e}, no-corner pieces bit for bit {bfb}, columns added {extra}, "
                f"top adds nothing and is TL0's {top_ok} (e_H max {max([t['e_H'] for t in tops] or [0.0]):.1e})")
        elif part == "e4":
            res = rec["result"]
            for f, fr in res["frames"].items():
                tv = {k: vr for k, vr in fr["variants"].items() if not k.startswith("TL0")}
                cx = max([x for vr in tv.values() for x in vr["pieces"]["cx"] + vr["pieces"]["cx_p"]
                          if x is not None] or [0.0])
                viol = sum(sum(vr["pieces"]["check_violations"]) for vr in tv.values())
                ek = max([x for vr in tv.values() for x in vr["pieces"]["e_K_inf"]] or [0.0])
                bfb = all(all(vr["pieces"]["bit_for_bit_tl0"]) for vr in tv.values())
                pyth = max(vr["pythagoras_rel"] for vr in fr["variants"].values())
                top_ok = all(not x["extra"] and x["bit_for_bit"] for x in fr["top"].values())
                nr = sum(1 for c in fr["non_right"] if c)
                log(f"  smoke e4 {res['problem']} {f}: {nr} of {res['pieces']} pieces with a non-right "
                    f"corner; treatments' CX max {cx:.1e}, column violations {viol}, e_K,inf max {ek:.1e}, "
                    f"no-corner pieces bit for bit {bfb}, top adds nothing {top_ok}, Pythagoras max {pyth:.1e}")
        else:
            log(f"  smoke cost: {len(rec['rows'])} rows")


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--part", choices=("modes", "e4", "cost"))
    ap.add_argument("--judge", action="store_true", help="read the registered predictions")
    ap.add_argument("--ns", default=",".join(map(str, NS)))
    ap.add_argument("--count", type=int, default=VAL_COUNT)
    ap.add_argument("--cost-count", type=int, default=COST_COUNT)
    ap.add_argument("--cost-n", type=int, default=COST_N)
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
    if args.judge:
        return write_verdicts(args.out)
    if args.part is None:
        raise SystemExit("--part or --judge")
    if args.smoke:
        if os.path.abspath(args.out) == os.path.abspath(OUT):
            raise SystemExit("--smoke writes to a scratch --out, never the registered record")
        args.seed_base, args.count, args.cost_count = SMOKE_SEED, 2, 2
        args.bench_n, args.cost_n = SMOKE_BENCH_N, 32
        args.ns = tuple(n for n in args.ns if n <= 32) or (16,)
    elif REGISTERED is None:
        raise SystemExit("the predictions are not registered: only --smoke runs until the owner's go")
    elif (args.seed_base != fd.VAL_POSTHOC or os.path.abspath(args.out) != os.path.abspath(OUT)
          or args.cost_n != COST_N or args.bench_n != BENCH_N):
        raise SystemExit("the registered run uses the post-hoc validation seeds, n = 64 benchmarks, "
                         "n = 128 costs and the registered record")
    elif args.allow_battery:
        raise SystemExit("--allow-battery is for the smoke run only")
    torch.set_num_threads(THREADS)
    U = fd.zero_heads()
    paths = {"modes": part_modes, "e4": part_e4, "cost": part_cost}[args.part](args, U)
    if args.smoke:
        smoke_report(paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
