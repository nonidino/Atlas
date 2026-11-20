"""cot_phase3_traces -- the port-trace weights of the coolant's training loss, from coupled
classical solves on the training layouts ([[chart-operator-stage-0b-and-1-plan]] sections 5,
6 and 16; the owner, 2026-11-19, after check 5 failed: "Realign loss, re-check (Recommended)").

On each training layout (`atlas/studio/cot_train_layouts.py`: random plates, never a
development or evaluation layout) the classical CO-T expert's coupled answer gives every
piece its port coefficients c (64, sine-hat, 16 per side).  The coupled field inside a piece is
H c + u_p, so its error is E c + E_p: **column k's error enters as c_k times itself.**  So:
- **the datum is removed** first (Stage 0's lesson: a 300 K offset in the hats hid the modes'
  errors): c' = c - mean(c_hats) * one, with ``one`` the uniform temperature's coefficients
  (the hats sum to 1 along the loop);
- each piece's weights are c'_k^2 / |c'|^2, averaged over the pieces of each kind (flowing,
  solid), and over the four sides (hats together, each sine j together), since the dataset's
  pieces come in every orientation;
- **the particular field's share**: over the pieces with a source, the mean of
  |U_p|_D^2 / (|V c'|_D^2 + |U_p|_D^2) in the piece's diffusion norm.

Writes ``out/chartop/cot/phase3-trace-weights.json`` (the weights, each layout's drawing and its
sha256).  `transport_torch.trace_weights` reads it.

    python scripts/cot_phase3_traces.py [--layouts 12]
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

import chartop_registry as registry  # noqa: E402
from atlas.chartop import fe  # noqa: E402
from atlas.chartop import ports as pt  # noqa: E402
from atlas.chartop import transport as tr  # noqa: E402
from atlas.studio import cot  # noqa: E402
from atlas.studio import cot_train_layouts as TL  # noqa: E402
from atlas.studio import document as D  # noqa: E402

OUT = os.path.join(ROOT, "out", "chartop", "cot", "phase3-trace-weights.json")


def side_classes(m: int = 16) -> list:
    """The column classes the dataset's orientations make equivalent: the four hats, then each
    sine j over the four sides."""
    cols = pt.basis("sine-hat", m, m).columns
    out = [[k for k, c in enumerate(cols) if c.what == "vertex"]]
    for j in range(1, m):
        out.append([k for k, c in enumerate(cols) if c.what == "side" and c.j == j])
    return out


def symmetrise(w: np.ndarray, classes: list) -> np.ndarray:
    out = np.zeros_like(w)
    for cl in classes:
        out[cl] = w[cl].mean()
    return out


def layout_traces(i: int) -> dict:
    name, spec, geom, cut, meta = TL.build(i)
    L = cot.layout(cut, spec)
    co = L.solve(tr.ClassicalTransportExpert())
    rows = []
    for k, (p, r) in enumerate(zip(L.problems, co.responses)):
        Q = L.pb(p.n).Q
        hats = [j for j, c in enumerate(L.pb(p.n).columns) if c.what == "vertex"]
        c = L.R[k] @ co.c
        one = np.zeros_like(c)
        one[hats] = 1.0
        cs = c - c[hats].mean() * one
        nrm = float(cs @ cs)
        Dm = p.diffusion_matrix()
        V = fe.full_field(p.n, r.H @ cs, Q @ cs)
        U = fe.full_field(p.n, r.up, np.zeros(4 * p.n))
        eV, eU = float(V @ (Dm @ V)), float(U @ (Dm @ U))
        rows.append({"flows": bool(p.flows), "w": (cs ** 2 / nrm).tolist() if nrm > 0 else None,
                     "source": bool(np.abs(r.up).max() > 0), "p_share": eU / (eV + eU) if eV + eU > 0 else None})
    text = D.dumps(spec, geom)
    return {"name": name, "meta": meta, "pieces": len(rows), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--layouts", type=int, default=12)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    import winqos
    winqos.opt_out()
    t0 = time.perf_counter()
    lays = []
    for i in range(a.layouts):
        t1 = time.perf_counter()
        lays.append(layout_traces(i))
        m = lays[-1]["meta"]
        print(f"{lays[-1]['name']}: {m['tier']} n {m['n']}, {lays[-1]['pieces']} pieces, channels {m['channels']}, "
              f"{len(m['devices'])} devices ({time.perf_counter() - t1:.0f} s)", flush=True)
    classes = side_classes()
    weights = {}
    for kind, flag in (("flowing", True), ("solid", False)):
        ws = [np.array(r["w"]) for L in lays for r in L["rows"] if r["flows"] == flag and r["w"] is not None]
        w = symmetrise(np.mean(ws, axis=0), classes)
        weights[kind] = {"w": (w / w.sum()).tolist(), "pieces": len(ws)}
    shares = [r["p_share"] for L in lays for r in L["rows"] if r["source"] and r["p_share"] is not None]
    groups = {"hats": [0], "j1-3": [1, 2, 3], "j4-8": [4, 5, 6, 7, 8], "j9-15": list(range(9, 16))}
    summary = {}
    for kind in weights:
        w = np.array(weights[kind]["w"])
        summary[kind] = {g: float(sum(w[classes[c]].sum() for c in cl)) for g, cl in groups.items()}
    rec = {"what": "the port-trace weights of the coolant's training loss (training layouts only)",
           "written": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "layouts": [{k: v for k, v in L.items() if k != "rows"} for L in lays],
           "weights": weights, "group_shares": summary,
           "particular_share": {"mean": float(np.mean(shares)) if shares else 0.0, "pieces_with_source": len(shares)},
           "seconds": time.perf_counter() - t0}
    registry.write_atomic(a.out, json.dumps(rec, indent=1).encode("utf-8"))
    print(json.dumps({"group_shares": summary, "particular_share": rec["particular_share"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
