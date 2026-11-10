"""Phase 3's overfit probe: can a network add to TL0 on the very pieces it trains on?

[[chart-operator-stage-0-rework]], the handoff's section 6.  B4's protocol (the forms page's
section 11.6): the four validation pieces 2,000,100-2,000,103 at n = 128
(`chartop_overfit_probe.pieces`, labelled from their stored arrays), S's configuration (split,
corner bubble, strips), the batch the four pieces, no augmentation, AdamW at 3e-4 with no
weight decay, gradients clipped at 1, the strips skipped in every step (they carry no
gradient, and the learned columns do not read them).  The expert is
`twolevel.TwoLevelExpert` at ``--n-c`` (16), trained on the relative objective (method 1),
pieces with nothing to learn left out and counted (method 3), from one of two starts:
- ``scratch``: zero heads (method 2), so step 0 is TL0 and ``rho`` is 0 there exactly;
- ``S``: S's registered weights (checked against ``weights-sha256.txt``) with their heads, so
  step 0 is ``C`` applied to S's answer, and its ``rho`` may be negative.

Every ``--every`` steps (250) the record is rewritten with the ledger on the four pieces:
``rho`` per piece over the learned columns and per group (hats, ``j = 1-3``, ``j = 4-8``),
``rho_p`` of the particular field, ``e_H,L`` and ``e_p``, ``kappa_kill`` of the network's raw
contribution (the form's answer less TL0's, before ``C``), and the usage control: the same
weights with each piece's input channels and scalars taken from the next piece (a cyclic
shift), every operator, load and reaction its own, whose ``rho`` should be 0 or below.  The
dead-network stop: a median ``rho`` under 0.05 at or after 20% of the steps ends the run, and
the record says so.  At the end, the cost of one call per piece: the networks' forward, the
coarse level, the strips (one call with them solved).

Determinism: on the CPU every step is bit for bit reproducible.  On CUDA the backward of the
U-Net's bilinear upsampling uses atomics, so two runs differ by rounding: the record says
which device ran, and ``--repeat-check K`` reruns the first K steps to measure the spread.
TF32 is disabled and asserted on CUDA.  Records: ``out/chartop/twolevel/probe-<start>.json``.
"""
from __future__ import annotations

import argparse
import datetime as dt
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
from chartop_forms_diagnose import checked_weights  # noqa: E402
from chartop_overfit_probe import pieces  # noqa: E402
from atlas.chartop import gates_forms as GF, train as trn, twolevel as tl  # noqa: E402

OUT = os.path.join(ROOT, "out", "chartop", "twolevel")
STARTS = ("scratch", "S")
S_ARGS = ("split", "unet", "1M", GF.KIND, GF.M_PORTS, None, "corner", True, "learned")


def build(start: str, n_c: int, device) -> tl.TwoLevelExpert:
    if start == "scratch":
        torch.manual_seed(0)
        return tl.TwoLevelExpert(*S_ARGS, n_c=n_c).to(device)
    return tl.from_kernel(trn.load_expert(checked_weights("S"), str(device)), n_c).to(device)


def to_device(b: dict, lab: dict, device) -> tuple[dict, dict]:
    return ({k: v.to(device) for k, v in b.items()}, {k: v.to(device) for k, v in lab.items()})


def measure(e: tl.TwoLevelExpert, twin: tl.TwoLevelExpert, b: dict, lab: dict, ref: dict) -> dict:
    """The ledger on the batch (see the module's docstring)."""
    G = tl.learned_groups(e.kind, e.m)
    L = sorted(sum(G.values(), []))
    e.eval()
    with torch.no_grad():
        H, up = tl.call(e, b, strip_solve=False)
        raw = e.last_uncorrected[0]
        lam_t = tl.column_energies(b, H, e.kind, e.m)
        eps_g = {g: (lam_t - ref["lam"])[:, c].sum(-1) for g, c in G.items()}
        eps_L = sum(eps_g.values())
        eps_p = tl.particular_energy(b, up) + lab["up_energy"]
        delta = (raw - ref["raw0"])[:, L]
        kap = tl.kappa_kill(e, delta)                     # e's own coarse space, this call's
        perm = list(range(1, b["inputs"].shape[0])) + [0]
        bs = dict(b, inputs=b["inputs"][perm], scalars=b["scalars"][perm])
        Hs, ups = tl.call(e, bs, strip_solve=False)
        eps_s = (tl.column_energies(b, Hs, e.kind, e.m) - ref["lam"])[:, L].sum(-1)
        eps_ps = tl.particular_energy(b, ups) + lab["up_energy"]
    e.train()
    cpu = lambda t: t.detach().cpu().numpy()                 # noqa: E731
    tr = cpu(lab["trace_Lambda"])
    lamL = cpu(ref["lam_L"])
    upE = cpu(lab["up_energy"])
    rho_L = tl.rho(cpu(eps_L), cpu(ref["eps0"]), tr)
    rho_g = {g: tl.rho(cpu(eps_g[g]), cpu(ref["eps0_groups"][g]), cpu(ref["lam"][:, G[g]].sum(-1))) for g in G}
    rho_p = tl.rho(cpu(eps_p), cpu(ref["eps_p0"]), upE)
    rho_s = tl.rho(cpu(eps_s), cpu(ref["eps0"]), tr)
    rho_ps = tl.rho(cpu(eps_ps), cpu(ref["eps_p0"]), upE)
    nan = lambda a: [None if not np.isfinite(x) else float(x) for x in a]    # noqa: E731
    return {"rho": nan(rho_L), "rho_summary": tl.summarise_rho(rho_L),
            "rho_groups": {g: nan(r) for g, r in rho_g.items()},
            "rho_groups_median": {g: tl.summarise_rho(r)["median"] for g, r in rho_g.items()},
            "rho_p": nan(rho_p), "rho_p_summary": tl.summarise_rho(rho_p),
            "e_H_L": [math.sqrt(max(x, 0.0) / y) for x, y in zip(cpu(eps_L), lamL)],
            "e_p": [math.sqrt(max(x, 0.0) / y) if y > 0 else None for x, y in zip(cpu(eps_p), upE)],
            "kappa_kill": nan(cpu(kap)),
            "control_shuffled_inputs": {"rho": nan(rho_s), "rho_median": tl.summarise_rho(rho_s)["median"],
                                        "rho_p_median": tl.summarise_rho(rho_ps)["median"]}}


def cost(e: tl.TwoLevelExpert, b: dict, reps: int = 3) -> dict:
    """Seconds per piece of one call: the whole two-level call with the strips solved, the
    same expert without ``C``, and the strips alone (the form's own clock)."""
    nc = e.n_c

    def best(f):
        f()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            f()
            if b["inputs"].is_cuda:
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return min(ts)
    B = b["inputs"].shape[0]
    with torch.no_grad():
        t_tl = best(lambda: tl.call(e, b, strip_solve=True))
        strips = e.last_strip_seconds
        e.n_c = None
        t_s = best(lambda: tl.call(e, b, strip_solve=True))
        e.n_c = nc
    return {"two_level_s_per_piece": t_tl / B, "without_C_s_per_piece": t_s / B,
            "coarse_level_s_per_piece": (t_tl - t_s) / B, "strips_s_per_piece": strips / B}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True, choices=STARTS)
    ap.add_argument("--n-c", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--pieces", type=int, default=4)
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--time-only", type=int, default=0, help="time this many steps and exit")
    ap.add_argument("--repeat-check", type=int, default=0,
                    help="rerun the first K steps from the start and record how far the weights differ")
    ap.add_argument("--no-dead-stop", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.benchmark = False
    b, lab, meta = pieces(args.n, args.pieces)
    b, lab = to_device(b, lab, device)
    e = build(args.start, args.n_c, device)
    twin = tl.zero_heads_twin(e)
    ref = tl.reference(twin, b, lab)
    with torch.no_grad():
        tl.call(twin, b, strip_solve=False)
    ref["raw0"] = twin.last_uncorrected[0]
    e.train()
    opt = torch.optim.AdamW(e.parameters(), lr=args.lr, weight_decay=0.0)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"probe-{args.start}.json")
    rec = {"start": args.start, "n_c": args.n_c, "n": args.n, "pieces": meta, "lr": args.lr,
           "steps_planned": args.steps, "every": args.every, "threads": args.threads,
           "device": str(device), "cuda_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
           "torch": torch.__version__, "objective": "relative (twolevel.relative_objective)",
           "easy": {"modes": ref["easy_H"].tolist(), "particular": ref["easy_p"].tolist()},
           "posthoc": True, "test_split_read": False,
           "started": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "trajectory": []}
    t0 = time.perf_counter()

    def step_once():
        H, up = tl.call(e, b, strip_solve=False)
        loss = tl.relative_objective(b, H, up, ref, e.kind, e.m)["loss"].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(e.parameters(), 1.0)
        opt.step()
        return float(loss.detach())

    if args.time_only:
        for _ in range(args.time_only):
            step_once()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dts = (time.perf_counter() - t0) / args.time_only
        print(f"{args.start}: {dts:.3f} s per step at n = {args.n}, n_c = {args.n_c}, {args.pieces} pieces, "
              f"{device}, {args.threads} threads; {args.steps} steps would take {args.steps * dts / 60:.0f} min")
        return 0
    rec["trajectory"].append(dict(step=0, elapsed_s=0.0, loss=None, **measure(e, twin, b, lab, ref)))
    registry.write_atomic(path, json.dumps(rec, indent=1).encode("utf-8"))
    print(f"step 0: median rho {rec['trajectory'][0]['rho_summary']['median']}", flush=True)
    stopped = None
    for step in range(1, args.steps + 1):
        loss = step_once()
        if step % args.every == 0 or step == args.steps:
            row = dict(step=step, elapsed_s=time.perf_counter() - t0, loss=loss, **measure(e, twin, b, lab, ref))
            rec["trajectory"].append(row)
            med = row["rho_summary"]["median"]
            if not args.no_dead_stop and tl.dead(step, args.steps, med):
                stopped = {"step": step, "why": f"dead network: median rho {med} below {tl.DEAD_RHO} "
                                                f"at or after {tl.DEAD_AFTER:.0%} of the steps"}
                rec["stopped"] = stopped
            registry.write_atomic(path, json.dumps(rec, indent=1).encode("utf-8"))
            print(f"step {step}: median rho {med}, median e_H,L {np.median(row['e_H_L']):.4g}, "
                  f"shuffled {row['control_shuffled_inputs']['rho_median']} ({time.perf_counter() - t0:.0f} s)",
                  flush=True)
            if stopped:
                break
    rec["seconds_per_step"] = (time.perf_counter() - t0) / max(1, rec["trajectory"][-1]["step"])
    rec["cost"] = cost(e, b)
    if args.repeat_check:
        e2 = build(args.start, args.n_c, device)
        opt2 = torch.optim.AdamW(e2.parameters(), lr=args.lr, weight_decay=0.0)
        e3 = build(args.start, args.n_c, device)
        opt3 = torch.optim.AdamW(e3.parameters(), lr=args.lr, weight_decay=0.0)
        for ex, op in ((e2, opt2), (e3, opt3)):
            ex.train()
            for _ in range(args.repeat_check):
                H, up = tl.call(ex, b, strip_solve=False)
                loss = tl.relative_objective(b, H, up, ref, ex.kind, ex.m)["loss"].mean()
                op.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ex.parameters(), 1.0)
                op.step()
        diff = max(float((p - q).abs().max()) for p, q in zip(e2.parameters(), e3.parameters()))
        rec["repeat_check"] = {"steps": args.repeat_check, "max_weight_difference": diff,
                               "bit_for_bit": diff == 0.0}
    rec["finished"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    registry.write_atomic(path, json.dumps(rec, indent=1).encode("utf-8"))
    last = rec["trajectory"][-1]
    print(f"probe {args.start}: median rho {rec['trajectory'][0]['rho_summary']['median']} -> "
          f"{last['rho_summary']['median']} at step {last['step']}; shuffled control "
          f"{last['control_shuffled_inputs']['rho_median']}; {rec['seconds_per_step']:.3f} s/step"
          + (f"; STOPPED: {stopped['why']}" if stopped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
