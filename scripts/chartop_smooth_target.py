"""The smooth-target control for B4 (the forms page, section 11.6): is the network, or its
target, why the modes do not fit?

B4 trained S's form on four validation pieces and reached a median ``e_H,L`` of 0.16 in
3,000 steps, while the particular field fitted below E1's bar.  This control changes one
thing: the target of the learned columns.  In place of the exact modes it is the
energy-best member of S's own output class restricted to smooth corrections,

    H_s = project(H0 + beta_c * Phi_K a*,  u1_0 + beta_c * Phi_K a1*),

``Phi_K`` = ``cos(p pi xi) cos(q pi eta)``, ``p, q < K`` (W358's smooth class, as
`chartop_corner_controls.smooth_basis` draws it), ``a*`` the A_I-energy-best coefficients
per learned column and ``a1*`` the level field's, then the expert's own kernel projection.
So a network output exists, smooth, that gives ``H_s`` exactly.  Everything else is B4's:
the four pieces (seeds 2,000,100-2,000,103 at n = 128), S's weights as the start, AdamW at
3e-4, no augmentation, the strips skipped in training, 4 threads.  The objective is the
supervised energy loss (`model.objectives`, "supervised") against ``H_s``; the particular
field's target stays the exact one.

``--target exact`` is the positive control of this script: the supervised loss against the
exact modes has the label-free loss's gradient (T26), so it must start at B4's 0.3356 and
follow B4's trajectory.

Measured every ``--every`` steps, per piece, with E1's normaliser (the exact learned
columns' energies) throughout:
- ``e_fit``: the learned columns' distance to the target trained on;
- ``e_fit_low``: the same over the hats and the side sines j = 1-3 only, the columns the
  coupled answer reads (section 11.1);
- ``e_H_L``: the distance to the exact modes, as E1 measures it;
- ``e_p``.

Registered before the run (2026-11-09), read on the median ``e_fit`` at the last step:
- at most ``PASS_FIT`` (0.01, E1's bar): the network and optimiser fit a smooth target
  of this form to the bar, so B4's floor is the target's, not the training's;
- at least ``FAIL_FIT`` (0.05): the training cannot fit even a target it can represent,
  so the optimisation (or the loss's conditioning) is the limit;
- between: both contribute, and the trajectory's slope is reported.
The record is rewritten after every measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402
import torch  # noqa: E402

import chartop_registry as registry  # noqa: E402
from chartop_corner_controls import smooth_basis  # noqa: E402
from chartop_overfit_probe import expert, pieces  # noqa: E402
from atlas.chartop import fe, forms as fm, gates_forms as GF, model as md  # noqa: E402

OUT = os.path.join(ROOT, "out", "chartop")
PASS_FIT = 0.01
FAIL_FIT = 0.05
B4_UNET_START = 0.3356          # forms-overfit-unet-3000.json, step 0 (rounded as recorded)


def low_columns(learned) -> list:
    """The hats and the side sines j = 1-3 (column 4 + 15 s + j - 1), within ``learned``."""
    low = list(range(4)) + [4 + 15 * s + (j - 1) for s in range(4) for j in (1, 2, 3)]
    if not set(low) <= set(learned):
        raise ValueError("the low columns are not all learned")
    return low


def smooth_target(e, b, lab, K: int) -> tuple[torch.Tensor, dict]:
    """``H_s`` [B, 64, n-1, n-1] (float64) and its diagnostics."""
    n = lab["H"].shape[-1] + 1
    KK = lab["H"].shape[1]
    ps = md.plain_square(n, GF.KIND, GF.M_PORTS, "cpu")
    ks = fm.kernel_square(n, GF.KIND, GF.M_PORTS, "cpu")
    s0 = b["s0"]
    # the principal part and level field exactly as `KernelExpert.forward` forms them
    H0 = ps.principal(s0, torch.float32).to(torch.float64)
    u10 = fm.level_principal(ps, s0, torch.float32).to(torch.float64)
    beta = fm.bubble_field(e.bubble, ps)[1:-1, 1:-1].reshape(-1).numpy()
    Phi = smooth_basis(n, K)
    X = beta[:, None] * Phi
    learned = list(e.learned)
    _loop, inner = fe.grid(n)
    ones = np.ones((n + 1) ** 2)
    H_raw = H0.clone()
    u1s = u10.clone()
    cond = []
    for i, (A_I, _M_IG) in enumerate(fm.batch_operators(b)):
        A = A_I.tocsr()
        AX = A @ X
        G = X.T @ AX
        cond.append(float(np.linalg.cond(G)))
        Hex = lab["H"][i].reshape(KK, -1).T.numpy()
        P = H0[i].reshape(KK, -1).T.numpy()
        D = Hex[:, learned] - P[:, learned]
        a = np.linalg.lstsq(G, AX.T @ D, rcond=1e-13)[0]
        H_raw[i, learned] = H0[i, learned] + torch.tensor((X @ a).T.reshape(len(learned), n - 1, n - 1))
        u1 = spla.splu(A_I.tocsc()).solve((fe.mass(n, b["w"][i].numpy()) @ ones)[inner])
        a1 = np.linalg.lstsq(G, AX.T @ (u1 - u10[i].reshape(-1).numpy()), rcond=1e-13)[0]
        u1s[i] = u10[i] + torch.tensor((X @ a1).reshape(n - 1, n - 1))
    Hs, _r = fm.project(H_raw, u1s, s0, ks)
    kres = fm.residual(Hs, u1s, s0, ks).abs().amax().item()
    return Hs, {"gram_condition": cond, "target_kernel_residual": kres}


def energies(b, H, ref, ps) -> np.ndarray:
    coef = (b["a11"], b["a12"], b["a22"], b["sigma"])
    return md.energy(*coef, md.full(ps, H - ref) - ps.QB[None]).numpy()


def measure(e, b, lab, Hs, learned, low) -> dict:
    e.eval()
    with torch.no_grad():
        H, up = fm.call(e, b, strip_solve=False)
        n = H.shape[-1] + 1
        ps = md.plain_square(n, GF.KIND, GF.M_PORTS, "cpu")
        coef = (b["a11"], b["a12"], b["a22"], b["sigma"])
        lam = md.energy(*coef, md.full(ps, lab["H"])).numpy()
        err_x = energies(b, H, lab["H"], ps)
        err_s = energies(b, H, Hs, ps)
        o = md.objectives(b, H, up, GF.KIND, GF.M_PORTS, lab, "label-free", H0=e.last_principal)
    e.train()
    B = err_x.shape[0]
    return {"e_fit": [GF.e_h_learned(err_s[i], lam[i], learned) for i in range(B)],
            "e_fit_low": [GF.e_h_learned(err_s[i], lam[i], low) for i in range(B)],
            "e_H_L": [GF.e_h_learned(err_x[i], lam[i], learned) for i in range(B)],
            "e_p": o["e_p"].tolist()}


def _medians(m: dict) -> dict:
    return {f"{k}_median": float(np.median(v)) for k, v in m.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", required=True, choices=("smooth", "exact"))
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--pieces", type=int, default=4)
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--time-only", type=int, default=0, help="time this many steps and exit")
    ap.add_argument("--record", default=None)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    b, lab, meta = pieces(args.n, args.pieces)
    e = expert("unet")
    e.train()
    learned = list(e.learned)
    low = low_columns(learned)
    t0 = time.perf_counter()
    if args.target == "smooth":
        Hs, tdiag = smooth_target(e, b, lab, args.K)
    else:
        Hs, tdiag = lab["H"], {}
    tdiag["target_seconds"] = time.perf_counter() - t0
    ps = md.plain_square(args.n, GF.KIND, GF.M_PORTS, "cpu")
    coef = (b["a11"], b["a12"], b["a22"], b["sigma"])
    lam = md.energy(*coef, md.full(ps, lab["H"])).numpy()
    floor = energies(b, Hs, lab["H"], ps)
    tdiag["target_e_H_L"] = [GF.e_h_learned(floor[i], lam[i], learned) for i in range(floor.shape[0])]
    tdiag["target_e_H_L_low"] = [GF.e_h_learned(floor[i], lam[i], low) for i in range(floor.shape[0])]
    labs = dict(lab, H=Hs)
    opt = torch.optim.AdamW(e.parameters(), lr=args.lr, weight_decay=0.0)
    name = args.record or f"smooth-target-{args.target}" + (f"-K{args.K}" if args.target == "smooth" else "") + ".json"
    path = os.path.join(OUT, name)
    rec = {"target": args.target, "K": args.K if args.target == "smooth" else None, "n": args.n,
           "pieces": meta, "lr": args.lr, "steps_planned": args.steps, "threads": args.threads,
           "start": "S's registered weights", "objective": "supervised (energy norm) against the target",
           "posthoc": True, "test_split_read": False,
           "registered": {"pass_fit": PASS_FIT, "fail_fit": FAIL_FIT, "read_on": "median e_fit at the last step",
                          "written": "2026-11-09, before the run"},
           "target_diagnostics": tdiag, "trajectory": []}
    print(f"target {args.target}: built in {tdiag['target_seconds']:.1f} s; its e_H,L against the exact "
          f"modes, median {np.median(tdiag['target_e_H_L']):.4g} (low columns "
          f"{np.median(tdiag['target_e_H_L_low']):.4g})", flush=True)
    t0 = time.perf_counter()
    if not args.time_only:
        m = measure(e, b, lab, Hs, learned, low)
        rec["trajectory"].append(dict(step=0, elapsed_s=0.0, **m, **_medians(m)))
        if args.target == "exact" and abs(rec["trajectory"][0]["e_H_L_median"] - B4_UNET_START) > 5e-5:
            raise SystemExit(f"the control does not start at B4's {B4_UNET_START}: "
                             f"{rec['trajectory'][0]['e_H_L_median']:.6g}")
        registry.write_atomic(path, json.dumps(rec, indent=1).encode("utf-8"))
        print(f"step 0: e_fit {rec['trajectory'][0]['e_fit_median']:.4g}, "
              f"e_H,L {rec['trajectory'][0]['e_H_L_median']:.4g}", flush=True)
    steps = args.time_only or args.steps
    for step in range(1, steps + 1):
        H, up = fm.call(e, b, strip_solve=False)
        o = md.objectives(b, H, up, GF.KIND, GF.M_PORTS, labs, "supervised", H0=e.last_principal)
        loss = o["loss"].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(e.parameters(), 1.0)
        opt.step()
        if args.time_only:
            continue
        if step % args.every == 0 or step == steps:
            m = measure(e, b, lab, Hs, learned, low)
            rec["trajectory"].append(dict(step=step, elapsed_s=time.perf_counter() - t0, **m, **_medians(m)))
            last = rec["trajectory"][-1]
            last["loss"] = float(loss.detach())
            fit = last["e_fit_median"]
            rec["verdict"] = ("pass" if fit <= PASS_FIT else "fail" if fit >= FAIL_FIT else "between")
            rec["steps_done"] = step
            registry.write_atomic(path, json.dumps(rec, indent=1).encode("utf-8"))
            print(f"step {step}: e_fit {fit:.4g} (low {last['e_fit_low_median']:.4g}), "
                  f"e_H,L {last['e_H_L_median']:.4g}, e_p {last['e_p_median']:.3g} "
                  f"({time.perf_counter() - t0:.0f} s)", flush=True)
    dt = (time.perf_counter() - t0) / steps
    if args.time_only:
        print(f"{args.target}: {dt:.2f} s per step; {args.steps} steps would take {args.steps * dt / 60:.0f} min")
        return 0
    print(f"smooth-target control, {args.target}: median e_fit "
          f"{rec['trajectory'][0]['e_fit_median']:.4g} -> {rec['trajectory'][-1]['e_fit_median']:.4g} "
          f"in {steps} steps ({dt:.2f} s/step): {rec['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
