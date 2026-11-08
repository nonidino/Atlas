"""W362's untrained diagnostic ([[chart-operator-kernel-exact-forms]] section 11.20): on the
pieces that are not squares, how far U's form is from the classical response; whether the
quotient a learned residual must output grows at corners, as W358's did; and how often such
pieces repeat.

    python scripts/chartop_w362_diagnose.py [--ns 32,64,128]

U's form with zero heads is the split form's principal part: the plain square's modes with
the high side sines by strip solves, under the corner bubble (`gates_hybrid.U_FORM`, which is
`chartop_corner_controls`' candidate "C4 + C3").  Per piece and resolution it records what
`chartop_corner_controls.Piece.candidate` computes for that candidate:
- ``e_H_classical``: e_H of U's form alone, the network's output zero, after the kernel
  projection (W356);
- ``e_H_smooth_K16``: e_H if a network added the energy-best smooth correction (W358's smooth
  class, K = 16), a proxy for the best a learned residual could do;
- ``peak`` and ``rms``: the quotient the network must output for the exact modes.

**The pieces.**
- Rectangles of one material, built here: aspect 1 (the control: U's form is exact) down to 0.5.
- The Winslow ring pieces of a hole and of a round inclusion, from `benchmarks_squares` seeds
  that no registration draws (`RING_SEEDS`).
- The pieces of the held-out benchmarks bend-3, s-channel and insert-round, which have been
  read.
- Stage 0's validation pieces are W358's record (``out/chartop/corner-controls.json``),
  quoted beside.

**The repetition.**  On each layout, the pieces `gates_squares.is_square` refuses, grouped by
translation: the chart's nodes less its first, to 1e-9 of the piece's size, with the same
conductivity and Fourier number.  The share reuse could answer is (pieces - classes) / pieces.

Classical solves and U's untrained form only: no weights, no training, never a test split.
Writes ``out/chartop/w362-diagnose.json`` and prints ten lines or fewer.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

import chartop_corner_controls as CC  # noqa: E402
import chartop_forms_diagnose as fd  # noqa: E402
import chartop_registry as registry  # noqa: E402
from atlas.chartop import benchmarks_squares as BS, benchmarks_workbench as BW  # noqa: E402
from atlas.chartop import chart as ch, forms, gates_squares as GS, ports, principal  # noqa: E402
from atlas.chartop.piece import PieceProblem  # noqa: E402

OUT = os.path.join(ROOT, "out", "chartop", "w362-diagnose.json")
ASPECTS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
#: seeds no registration draws (benchmarks_squares' features cycle by seed mod 5)
RING_SEEDS = {"hole": 9_000_003, "round-inclusion": 9_000_004}
HELD_OUT = ("bend-3", "s-channel", "insert-round")
#: U's form: the plain square's modes with the strips, under the corner bubble
CANDIDATE = ("strips", "corner", 0)
W358_REFERENCE = ("out/chartop/corner-controls.json", "C4 + C3")


def rectangle(aspect: float, n: int) -> PieceProblem:
    c = ch.Chart(ch.from_map(lambda xi, eta: (xi, aspect * eta), n), "affine")
    return PieceProblem(c, fo=None, kappa_ref=1.0, log_kappa_hat=None)


def arrays_of(p: PieceProblem) -> dict:
    t = p.tensors()
    return {**{k: np.asarray(t[k], np.float64) for k in ("a11", "a12", "a22", "sigma", "w")}, "s0": float(p.s0)}


class ArraysPiece(CC.Piece):
    """`chartop_corner_controls.Piece` on a given piece's arrays instead of a drawn seed."""

    def __init__(self, arrays: dict, label: str, n: int, Q, W, c1):
        draw, arr = CC.data.draw, CC.data.arrays
        CC.data.draw, CC.data.arrays = (lambda seed, n=None: None), (lambda e: arrays)
        try:
            super().__init__(label, n, Q, W, c1)
        finally:
            CC.data.draw, CC.data.arrays = draw, arr


def groups(n: int) -> dict:
    """{piece type: [(label, PieceProblem)]} at resolution ``n``."""
    out = {f"rectangle {a:g}": [(f"rect {a:g}", rectangle(a, n))] for a in ASPECTS}
    for feature, seed in RING_SEEDS.items():
        b = BS.build(BS.draw(seed), n, "steady")
        out[f"{feature} rings"] = [(f"{seed} piece {i}", p) for i, p in enumerate(b.problems)
                                   if p.chart.method.startswith("winslow")]
    out["held-out curved"] = [(f"{name} piece {i}", p) for name in HELD_OUT
                              for i, p in enumerate(fd.bench(name, "registered", "held-out", n, "steady").problems)
                              if not GS.is_square(p)]
    return out


def repetition() -> dict:
    """Per layout: the non-square pieces, their classes under translation, the share reuse
    could answer."""
    layouts = {f"squares {s} ({f})": BS.build(BS.draw(s), 16, "steady") for f, s in RING_SEEDS.items()}
    layouts.update({f"workbench {name}": BW.build(name, 16, "steady") for name in ("plate-insert", "fast-heat", "plate-hole")})
    layouts.update({f"held-out {name}": fd.bench(name, "registered", "held-out", 16, "steady") for name in HELD_OUT})
    out = {}
    for key, b in layouts.items():
        pieces = [p for p in b.problems if not GS.is_square(p)]
        classes = set()
        for p in pieces:
            X = np.asarray(p.chart.X, float)
            scale = float(np.ptp(X.reshape(-1, 2), axis=0).max()) or 1.0
            shape = tuple(np.round((X - X[0, 0]) / scale, 9).ravel())
            kappa = tuple(np.round(np.asarray(p.log_kappa_hat_e, float), 9).ravel())
            classes.add((shape, kappa, None if p.steady else round(float(p.fo), 12)))
        k = len(pieces)
        out[key] = {"pieces": len(b.problems), "non_square": k, "classes": len(classes),
                    "share_reuse_answers": (k - len(classes)) / k if k else None}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ns", default="32,64,128")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    ns = [int(x) for x in args.ns.split(",")]
    t0 = time.perf_counter()
    rec = {"script": "scripts/chartop_w362_diagnose.py", "ns": ns, "candidate": "C4 + C3 (U's form)",
           "aspects": list(ASPECTS), "ring_seeds": RING_SEEDS, "held_out": list(HELD_OUT), "types": {}}
    for n in ns:
        Q = ports.basis("sine-hat", n, CC.M_PORTS).Q
        c1 = forms.kernel_coefficients("sine-hat", n, CC.M_PORTS)
        H0s = principal.modes(n, Q, 0.0)
        W = (c1[None, :] * H0s) / (H0s @ (c1 * c1))[:, None]
        Phis = {16: CC.smooth_basis(n, 16)}
        beta = forms.corner_bubble(n)[1:-1, 1:-1].ravel()
        for kind, pieces in groups(n).items():
            rows = []
            for label, p in pieces:
                r = ArraysPiece(arrays_of(p), label, n, Q, W, c1).candidate(*CANDIDATE, Phis, beta)
                rows.append({"piece": label, **{k: r[k] for k in ("e_H_classical", "e_H_smooth_K16", "peak", "rms")}})
            med = {k: float(np.median([r[k] for r in rows])) for k in ("e_H_classical", "e_H_smooth_K16", "peak", "rms")}
            rec["types"].setdefault(kind, {})[str(n)] = {"count": len(rows), "median": med, "pieces": rows}
    ref = json.load(open(os.path.join(ROOT, *W358_REFERENCE[0].split("/")), encoding="utf-8"))["results"][W358_REFERENCE[1]]
    rec["stage0_validation_W358"] = {n: {k: ref[n]["median"][k] for k in ("e_H_classical", "e_H_smooth_K16", "peak", "rms")}
                                     for n in ref if int(n) in ns}
    rec["repetition"] = repetition()
    rec["elapsed_s"] = time.perf_counter() - t0
    rec["written"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    registry.write_atomic(args.out, json.dumps(rec, indent=1).encode("utf-8"))
    cell = lambda d, k, f: " ".join(f.format(d[str(n)]["median"][k]) if str(n) in d else "-" for n in ns)  # noqa: E731
    print(f"W362      piece type (count)          e_H, U's form, n={','.join(map(str, ns))}   best smooth K16       quotient peak")
    for kind in ("rectangle 1", "rectangle 0.9", "rectangle 0.7", "rectangle 0.5", "hole rings", "round-inclusion rings",
                 "held-out curved"):
        d = rec["types"][kind]
        print(f"          {kind + ' (' + str(d[str(ns[0])]['count']) + ')':27s} {cell(d, 'e_H_classical', '{:.4f}'):22s} "
              f"{cell(d, 'e_H_smooth_K16', '{:.4f}'):21s} {cell(d, 'peak', '{:5.1f}')}")
    s0 = rec["stage0_validation_W358"]
    ref = {k: " ".join(fmt.format(s0[str(n)][k]) for n in ns if str(n) in s0)
           for k, fmt in (("e_H_classical", "{:.4f}"), ("e_H_smooth_K16", "{:.4f}"), ("peak", "{:5.1f}"))}
    print(f"          {'Stage 0 validation (W358)':27s} {ref['e_H_classical']:22s} {ref['e_H_smooth_K16']:21s} {ref['peak']}")
    reps = rec["repetition"]
    print("          reuse could answer: " + "; ".join(f"{k.split(' ', 1)[1]} {v['share_reuse_answers']:.2f}"
                                                     for k, v in reps.items() if v["non_square"])[:200])
    print(f"          wrote {os.path.relpath(args.out, ROOT)} ({rec['elapsed_s']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
