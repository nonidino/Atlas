"""Stage 0 (W355), POST HOC, after the one evaluation: is the zero source what breaks the modes?

    python scripts/chartop_posthoc_zero_source.py --weights out/chartop/runs/B/best.pt

Not a gate, and run only after `scripts/chartop_evaluate.py` had judged the registered
arms.  On the three steady benchmarks (no source: the network's fifth input channel is
identically zero, a value the generator never drew), arm B's coupled answer three ways:

- **as registered**: the source channel zero, as the evaluation fed it;
- **unit source channel**: the same weights with the channel set to the constant 1 (a
  constant, unit-RMS source, inside the training distribution) for the call that
  returns the modes, and the particular field taken as zero, which it is when g = 0;
- **the principal part alone** (no weights), for reference.

The modes should not depend on the source; if the second row is much better than the
first, the failure on the steady problems is the input distribution, not the modes'
learnability.

**Then the modes themselves, piece by piece**, on every piece of the six benchmark
problems (steady and step, n = 64): the energy-norm error of B's modes against the
exact ones, ``e_H`` as E1 defines it, beside the principal part's, and the share of
each in the port groups (corner hats; side sines j = 1-3, 4-8, 9-15), with each
group's own relative error.  This says whether the trained correction improves or
damages the modes on pieces it was never trained on.

**And a datum shift**: the three steady benchmarks again with every Dirichlet value
lowered by ``T_COLD`` (the cold wall at 0 K, the hot at 100 K).  The problem is linear
and a constant carries no steady energy, so the exact field only shifts and its energy
norm is unchanged; a trial space that reproduces constants exactly gives the same
error either way.  If B's error falls with the shift, the coupled error is dominated
by how its hat modes reproduce the temperature level, which e_H weights at a fraction
of a per cent.  Writes ``out/chartop/posthoc-zero-source.json``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from atlas.chartop import benchmarks as bm, data, fe, host as hs, model as md  # noqa: E402
from atlas.chartop.expert import LearnedExpert  # noqa: E402
from atlas.chartop.host import Response  # noqa: E402
from atlas.chartop.train import load_expert  # noqa: E402

GROUPS = (("hats", None, None), ("sines 1-3", 1, 3), ("sines 4-8", 4, 8), ("sines 9-15", 9, 15))


class UnitSourceExpert(LearnedExpert):
    """The same network, with the source channel set to the constant 1 for every piece
    and the particular field returned as zero (valid only where g = 0)."""

    def respond(self, problems, pb_of, modes_from=None):
        out = []
        with torch.no_grad():
            for p in problems:
                assert float(np.abs(p.g).max()) == 0.0, "only for sourceless pieces"
                chans, scal, _r = p.inputs()
                chans = chans.copy()
                chans[4] = 1.0
                x = torch.tensor(chans[None])
                s = torch.tensor(scal[None])
                s0 = torch.tensor([p.s0], dtype=torch.float64)
                H, _up = self.net(x, s, s0)
                H = H.double().numpy()[0]
                out.append(Response(H.reshape(H.shape[0], -1).T.copy(), np.zeros((p.n - 1) ** 2)))
        self.calls += len(problems)
        return out


def group_of(col) -> str:
    if col.what == "vertex":
        return "hats"
    for g, lo, hi in GROUPS[1:]:
        if lo <= col.j <= hi:
            return g
    raise ValueError(col)


def mode_errors(p, pb, H_try: np.ndarray, ex: fe.Exact, H: np.ndarray, Lam_diag: np.ndarray) -> dict:
    E = H_try - H
    e_k = np.einsum("ik,ik->k", E, np.asarray(ex.A_I @ E))
    groups = [group_of(c) for c in pb.columns]
    out = {"e_H": float(np.sqrt(e_k.sum() / Lam_diag.sum()))}
    for g, _lo, _hi in GROUPS:
        sel = np.array([x == g for x in groups])
        out[g] = {"share": float(e_k[sel].sum() / e_k.sum()),
                  "relative": float(np.sqrt(e_k[sel].sum() / Lam_diag[sel].sum()))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=os.path.join(ROOT, "out", "chartop", "runs", "B", "best.pt"))
    args = ap.parse_args()
    torch.set_num_threads(8)
    net = load_expert(args.weights, "cpu")
    learned = LearnedExpert(net, envelope=data.ENVELOPE)
    principal = LearnedExpert(md.Expert("unet", "0.3M"), envelope=data.ENVELOPE)
    experts = {"as registered (source channel 0)": learned,
               "unit source channel": UnitSourceExpert(net, envelope=data.ENVELOPE),
               "principal part alone": principal}
    rows = []
    for name in bm.NAMES:
        kw = {"envelope": data.ENVELOPE} if name == "s-channel" else {}
        b = bm.build(name, 64, "steady", **kw)
        L = b.layout("sine-hat", 16)
        mono = hs.monolithic(L)
        span = max(float(np.max(u)) for u in mono.fields) - min(float(np.min(u)) for u in mono.fields)
        r = {"name": name}
        for label, ex in experts.items():
            cp = L.solve(ex)
            num, den = cp.energy_error(mono.fields)
            mx = max(float(np.max(np.abs(u - v))) for u, v in zip(mono.fields, cp.fields))
            r[label] = {"energy_rel": num / den, "max_over_range": mx / span}
        print(json.dumps(r), flush=True)
        rows.append(r)
    pieces = []
    for name in bm.NAMES:
        for problem in ("steady", "step"):
            kw = {"envelope": data.ENVELOPE} if name == "s-channel" else {}
            b = bm.build(name, 64, problem, **kw)
            L = b.layout("sine-hat", 16)
            fw_learned = learned.forward(L.problems)
            fw_principal = principal.forward(L.problems)
            agg = {"learned": [], "principal": []}
            for i, p in enumerate(L.problems):
                pb = L.pb(p.n)
                ex = fe.Exact(p.matrix(), p.n)
                H = ex.modes(pb.Q)
                Lam = ex.port_matrix(pb.Q, H)
                d = np.diag(Lam)
                agg["learned"].append(mode_errors(p, pb, fw_learned[i][0], ex, H, d))
                agg["principal"].append(mode_errors(p, pb, fw_principal[i][0], ex, H, d))
            rec = {"name": name, "problem": problem, "pieces": len(L.problems)}
            for k, v in agg.items():
                eh = np.array([x["e_H"] for x in v])
                rec[k] = {"e_H_median": float(np.median(eh)), "e_H_min": float(eh.min()),
                          "e_H_max": float(eh.max()),
                          "groups": {g: {"share_median": float(np.median([x[g]["share"] for x in v])),
                                         "relative_median": float(np.median([x[g]["relative"] for x in v]))}
                                     for g, _lo, _hi in GROUPS}}
            worse = int(sum(a["e_H"] > c["e_H"] for a, c in zip(agg["learned"], agg["principal"])))
            rec["pieces_where_learned_is_worse"] = worse
            print(json.dumps(rec), flush=True)
            pieces.append(rec)
    shifted = []
    for name in bm.NAMES:
        kw = {"envelope": data.ENVELOPE} if name == "s-channel" else {}
        b = bm.build(name, 64, "steady", **kw)

        def bc(i, s, P, _bc=b.bc):
            kind, val = _bc(i, s, P)
            return (kind, val - bm.T_COLD) if kind == "dirichlet" else (kind, val)

        r = {"name": name, "shift_K": -bm.T_COLD}
        for label, L in (("as registered", b.layout("sine-hat", 16)),
                         ("datum shifted", hs.Layout(b.problems, bc, "sine-hat", 16))):
            mono = hs.monolithic(L)
            span = (max(float(np.max(u)) for u in mono.fields)
                    - min(float(np.min(u)) for u in mono.fields))
            for who, ex in (("B", learned), ("principal part", principal)):
                cp = L.solve(ex)
                num, den = cp.energy_error(mono.fields)
                mx = max(float(np.max(np.abs(u - v))) for u, v in zip(mono.fields, cp.fields))
                r[f"{label}, {who}"] = {"energy_rel": num / den, "max_over_range": mx / span,
                                        "reference_energy": den}
        print(json.dumps(r), flush=True)
        shifted.append(r)
    path = os.path.join(ROOT, "out", "chartop", "posthoc-zero-source.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"written": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                   "post_hoc": "run after the one evaluation; not a gate",
                   "weights": os.path.relpath(args.weights, ROOT), "coupled": rows,
                   "modes_by_piece": pieces, "datum_shift": shifted}, fh, indent=1)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
