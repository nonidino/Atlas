"""B4, the overfit probe (the forms page, section 11): can the network reach ``e_H,L``
about 0.01 on the very pieces it trains on?

Four validation pieces (seeds 2,000,100 onward, drawn at n = 128; never the test split),
S's form (split, corner bubble, strips), the registered label-free objective, no
augmentation, batch = the four pieces.  Both arms start from S's registered weights
(checked against ``weights-sha256.txt``): ``unet`` is S itself; ``spectral`` is S with
B3's cosine path added at the coarsest level, whose output layer starts at zero, so at
step 0 it is S bit for bit.  The training steps skip the strips (they carry no
gradient); ``e_H,L`` is measured over the learned columns against their own exact
energies, exactly as gate E1 measures it, on the same four pieces.

If neither arm reaches about 0.01 here, the limit is the target or the network form,
not the data.  The record is rewritten after every measurement, so a stopped run keeps
its trajectory.  ``--time-only`` times a few steps and exits.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import chartop_registry as registry  # noqa: E402
from chartop_forms_diagnose import checked_weights, VAL_POSTHOC  # noqa: E402
from atlas.chartop import data, forms as fm, gates_forms as GF, model as md, ports as pt, train as tr  # noqa: E402

OUT = os.path.join(ROOT, "out", "chartop")


def pieces(n: int, count: int) -> tuple[dict, dict]:
    """The batch (float64 coefficients, float32 inputs) and exact labels of ``count``
    validation pieces at resolution ``n``."""
    exs = [data.draw(VAL_POSTHOC + k, n=n) for k in range(count)]
    arrs = [data.arrays(e) for e in exs]
    b = {k: torch.tensor(np.stack([a[k] for a in arrs]), dtype=torch.float64)
         for k in ("a11", "a12", "a22", "sigma", "w", "g")}
    b["inputs"] = torch.tensor(np.stack([a["inputs"] for a in arrs]), dtype=torch.float32)
    b["scalars"] = torch.tensor(np.stack([a["scalars"] for a in arrs]), dtype=torch.float32)
    b["s0"] = torch.tensor(np.array([a["s0"] for a in arrs]), dtype=torch.float64)
    Q = pt.basis(GF.KIND, n, GF.M_PORTS).Q
    lab = [data.labels_from_arrays(a, Q) for a in arrs]
    K = 4 * GF.M_PORTS
    labels = {"H": torch.tensor(np.stack([x["H"].T.reshape(K, n - 1, n - 1) for x in lab])),
              "up": torch.tensor(np.stack([x["up"].reshape(n - 1, n - 1) for x in lab])),
              "trace_Lambda": torch.tensor([x["trace_Lambda"] for x in lab], dtype=torch.float64),
              "up_energy": torch.tensor([x["up_energy"] for x in lab], dtype=torch.float64)}
    meta = [{"seed": e.seed, "fo": e.fo, "freq": e.freq, "family": e.family} for e in exs]
    return b, labels, meta


def expert(arm: str) -> fm.KernelExpert:
    S = tr.load_expert(checked_weights("S"), "cpu")
    if arm == "unet":
        return S
    c = S.config()
    e = fm.make("unet-spectral", c["size"], c["form"], c["kind"], c["m"], c["p_size"], c["bubble"], c["strips"])
    missing, unexpected = e.load_state_dict(S.state_dict(), strict=False)
    if unexpected or not missing or not all(".spec" in k for k in missing):
        raise SystemExit(f"S's weights do not load into the spectral arm: {missing} {unexpected}")
    return e


def measure(e, b, lab) -> dict:
    """``e_H,L`` (learned columns), ``e_H`` (all 64, the strips as the plain square's H0 in
    training, so ``e_H`` is reported but not judged) and ``e_p`` per piece."""
    e.eval()
    with torch.no_grad():
        H, up = fm.call(e, b, strip_solve=False)
        n = H.shape[-1] + 1
        ps = md.plain_square(n, GF.KIND, GF.M_PORTS, "cpu")
        coef = (b["a11"], b["a12"], b["a22"], b["sigma"])
        err = md.energy(*coef, md.full(ps, H - lab["H"]) - ps.QB[None]).numpy()
        lam = md.energy(*coef, md.full(ps, lab["H"])).numpy()
        o = md.objectives(b, H, up, GF.KIND, GF.M_PORTS, lab, "label-free", H0=e.last_principal)
    learned = list(fm.learned_columns(GF.KIND, GF.M_PORTS))
    ehl = [GF.e_h_learned(err[i], lam[i], learned) for i in range(err.shape[0])]
    hats = [int(k) for k in np.flatnonzero(fm.kernel_coefficients(GF.KIND, n, GF.M_PORTS))]
    hat_share = [float(err[i, hats].sum() / err[i, learned].sum()) for i in range(err.shape[0])]
    e.train()
    return {"e_H_L": ehl, "e_H_L_median": float(np.median(ehl)), "e_p": o["e_p"].tolist(),
            "hat_share_of_learned_err": hat_share, "loss": float(o["loss"].mean())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", required=True, choices=("unet", "spectral"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--pieces", type=int, default=4)
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--time-only", type=int, default=0, help="time this many steps and exit")
    ap.add_argument("--record", default=None,
                    help="the record's file name in out/chartop (default forms-overfit-<arm>.json)")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    b, lab, meta = pieces(args.n, args.pieces)
    e = expert(args.arm)
    e.train()
    opt = torch.optim.AdamW(e.parameters(), lr=args.lr, weight_decay=0.0)
    path = os.path.join(OUT, args.record or f"forms-overfit-{args.arm}.json")
    rec = {"arm": args.arm, "n": args.n, "pieces": meta, "lr": args.lr, "steps_planned": args.steps,
           "threads": args.threads, "start": "S's registered weights", "posthoc": True,
           "test_split_read": False, "trajectory": []}
    t0 = time.perf_counter()
    if not args.time_only:
        rec["trajectory"].append(dict(step=0, elapsed_s=0.0, **measure(e, b, lab)))
    steps = args.time_only or args.steps
    for step in range(1, steps + 1):
        H, up = fm.call(e, b, strip_solve=False)
        o = md.objectives(b, H, up, GF.KIND, GF.M_PORTS, lab, "label-free", H0=e.last_principal)
        loss = o["loss"].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(e.parameters(), 1.0)
        opt.step()
        if args.time_only:
            continue
        if step % args.every == 0 or step == steps:
            rec["trajectory"].append(dict(step=step, elapsed_s=time.perf_counter() - t0, **measure(e, b, lab)))
            rec["best_e_H_L_median"] = min(r["e_H_L_median"] for r in rec["trajectory"])
            registry.write_atomic(path, json.dumps(rec, indent=1).encode("utf-8"))
            print(f"step {step}: e_H,L median {rec['trajectory'][-1]['e_H_L_median']:.4g} "
                  f"({time.perf_counter() - t0:.0f} s)", flush=True)
    dt = (time.perf_counter() - t0) / steps
    if args.time_only:
        print(f"{args.arm}: {dt:.2f} s per step at n = {args.n}, batch {args.pieces}, "
              f"{args.threads} threads; {args.steps} steps would take {args.steps * dt / 60:.0f} min")
        return 0
    tr0, trN = rec["trajectory"][0], rec["trajectory"][-1]
    print(f"B4  {args.arm}: e_H,L median {tr0['e_H_L_median']:.4g} -> {trN['e_H_L_median']:.4g} "
          f"(best {rec['best_e_H_L_median']:.4g}) in {steps} steps, {dt:.2f} s/step; "
          f"hat share of the learned error {np.median(trN['hat_share_of_learned_err']):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
