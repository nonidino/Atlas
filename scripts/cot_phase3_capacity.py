"""The coolant plan's capacity probe (gate CAP): can the planned network fit the hardest
pieces at all?  [[chart-operator-stage-0b-and-1-plan]] section 15.2, run before any
dataset is built.

**The pieces** (the rule is fixed here, before the first run): drawn by
`transport_data.draw` at n = 128 from the probe's seed base (12,000,000) upward, the first
``PER_KIND`` seeds of each flowing kind (through, turn, wavy) whose piece Peclet number is
at least ``PE_MIN`` = 0.9 of the cap: the plan's "hardest corner", Pe near 200 at n = 128,
where the boundary layers are one or two cells thick.  Labelled classically on the laptop
from their stored float32 arrays (`transport_data.labels`, full fields).  **Contrast does
not enter a piece:** a CO-T piece is one material and its operator is divided by its own
conductivity, so the plan's "high contrast" lives in the host's seams, not in any piece's
modes; it is met at the layout level (E4 on the plates), not here.

**The expert** is arm R (`transport_torch.ArmR`): TL0 at n_c = 16 plus the network's
corrections, trial and adjoint from one network (b -> -b), each followed by the coarse
correction, so step 0 is TL0 exactly (zero heads).  ``--form``: ``free`` (the correction
at the interior nodes) or ``bubble`` (CO-D's form, times 16 xi(1-xi) eta(1-eta)).
``--size``: the U-Net's width (`model.WIDTHS`): 0.3M, 1M (the plan's) or 3M.

**Training** is the plan's label-free loss (`transport_torch.objective`): each mode group's
residual in the plain square's dual norm over TL0's, trial, adjoint and particular, with
equal weight.  The batch is the pieces, no augmentation, AdamW at 3e-4, no weight decay,
gradients clipped at 1, float32 network and float64 operators, on the CPU (bit for bit
reproducible; ``--repeat-check`` measures it).

**The ledger** (every ``--every`` steps): per piece and per group (trial hats, j = 1-3,
4-8, 9-15; the same for the adjoint; the particular field), the error in the piece's
diffusion norm over TL0's (``err``, the quantity CAP reads) and the dual-norm residual
over TL0's (``res``, the loss's); and the shuffled control: the same weights with each
piece fed the next piece's inputs, whose errors should not fall.

**The criteria** are in ``CRITERIA``, fixed before the first run.  State is checkpointed
atomically every ledger (``out/chartop/runs/cot-capacity/``, git-ignored), so a run cut
by standby or a restart resumes where it stopped.  Record:
``out/chartop/cot/phase3-capacity-<form>-<size>.json``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import chartop_registry as registry  # noqa: E402
from atlas.chartop import model as md  # noqa: E402
from atlas.chartop import transport_data as td  # noqa: E402
from atlas.chartop import transport_torch as tt  # noqa: E402

OUT = os.path.join(ROOT, "out", "chartop", "cot")
STATE = os.path.join(ROOT, "out", "chartop", "runs", "cot-capacity")
N = 128
N_C = 16
KINDS = ("through", "turn", "wavy")
PER_KIND = 2
PE_MIN = 0.9 * td.pe_cap(N)
LR = 3e-4

#: fixed before the first run; ``owner`` says who set the bar and when
CRITERIA = {
    "cap_fraction": 0.30,          # every group's median relative error at most this at the end
    "steps": 3000,                 # the step budget per run
    "groups": "trial hats, j1-3, j4-8, j9-15; adjoint the same; the particular field (9 in all)",
    "statistic": "per group, the median over the pieces of the error in the piece's diffusion norm "
                 "over TL0's; the worst piece reported beside it",
    "dead_stop": "the loss above 0.95 at or after 20% of the steps ends the run (recorded as dead)",
    "owner": "the owner, in chat on 2026-11-19, before the first run: the bar 0.30 (\"0.30 (Recommended)\": "
             "CL-A needs about 0.44 on held-out layouts, with margin for the gap to training pieces); both "
             "forms at 1M, 3,000 steps each, 3M only if both fail (\"Both forms, 3,000 steps (Recommended)\"); "
             "on mains",
    "runs": "free-1M, then bubble-1M; 3M (the form that did better) only if both fail",
    "groups_judged": "from 2026-11-19, after check 5 showed the coupled energy error reads only the trial modes (the owner: 'Trial and particular (Recommended)'): the four trial groups and the particular field must reach the bar; the adjoint groups are reported. Runs that ended before this are read both ways",
}
#: the groups CAP judges (see CRITERIA["groups_judged"])
JUDGED = ("trial hats", "trial j1-3", "trial j4-8", "trial j9-15", "particular")


def select(n: int = N, per_kind: int = PER_KIND, pe_min: float = PE_MIN, start: int | None = None):
    """The probe's pieces by the rule above: (seeds, metas)."""
    s = td.SEED_BASE["probe"] if start is None else start
    want = {k: per_kind for k in KINDS}
    seeds, metas = [], []
    while any(want.values()):
        p, meta = td.draw(s, n)
        if want.get(meta["kind"], 0) > 0 and meta["pe"] >= pe_min:
            want[meta["kind"]] -= 1
            seeds.append(s)
            metas.append({k: (v if not isinstance(v, np.generic) else v.item()) for k, v in meta.items()})
        s += 1
        if s - td.SEED_BASE["probe"] > 200_000:
            raise RuntimeError("no pieces in 200,000 seeds")
    order = sorted(range(len(seeds)), key=lambda i: (KINDS.index(metas[i]["kind"]), seeds[i]))
    return [seeds[i] for i in order], [metas[i] for i in order]


def build_batch(seeds, n: int = N, n_c: int = N_C, device: str = "cpu"):
    pieces, labels = [], []
    for s in seeds:
        p, meta = td.draw(s, n)
        q = td.piece_from_arrays(td.arrays(p, meta))
        pieces.append(q)
        labels.append(td.labels(q, full=True))
    return tt.PieceBatch(pieces, n_c=n_c, labels=labels, device=device)


def summarise(r: np.ndarray, names: list) -> dict:
    """r [B, groups]: per group the median and the worst piece."""
    return {nm: {"median": float(np.median(r[:, j])), "max": float(np.max(r[:, j]))}
            for j, nm in enumerate(names)}


def measure(e: tt.ArmR, b: tt.PieceBatch) -> dict:
    names = b.column_names()
    e.eval()
    with torch.no_grad():
        H, G, up = e(b)
        loss = float(tt.objective(b, H, G, up))
        err = tt.relative(b.group_errors(H, G, up), b.err0).cpu().numpy()
        res = tt.relative(b.group_residuals(H, G, up), b.res0).cpu().numpy()
        perm = list(range(1, b.B)) + [0]
        Hs, Gs, ups = e(b, x=b.x[perm], x_rev=b.x_rev[perm], s=b.s[perm])
        err_s = tt.relative(b.group_errors(Hs, Gs, ups), b.err0).cpu().numpy()
    e.train()
    return {"loss": loss, "err": summarise(err, names), "res": summarise(res, names),
            "err_per_piece": err.tolist(), "shuffled_err": summarise(err_s, names)}


def verdict(row: dict, fraction: float, judged=None) -> dict:
    """CAP on a ledger row: every judged group's median at most ``fraction`` (``judged`` None: all)."""
    err = {g: v for g, v in row["err"].items() if judged is None or g in judged}
    worst = max(v["median"] for v in err.values())
    failing = [g for g, v in err.items() if v["median"] > fraction]
    return {"met": not failing, "worst_group_median": worst, "groups_over_the_bar": failing,
            "fraction": fraction, "groups_judged": list(err)}


def power_state() -> dict:
    """BatteryStatus (2 = on mains) and charge, read through PowerShell; None off Windows."""
    if os.name != "nt":
        return {}
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Battery | Select-Object BatteryStatus,"
                              "EstimatedChargeRemaining | ConvertTo-Json -Compress)"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return json.loads(out) if out else {}
    except Exception as exc:                                        # noqa: BLE001
        return {"error": repr(exc)}


def save_state(path: str, e, opt, step: int, rec: dict) -> None:
    buf = io.BytesIO()
    torch.save({"model": e.state_dict(), "opt": opt.state_dict(), "step": step, "record": rec}, buf)
    registry.write_atomic(path, buf.getvalue())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--form", default="free", choices=tt.FORMS)
    ap.add_argument("--size", default="1M", choices=sorted(md.WIDTHS))
    ap.add_argument("--steps", type=int, default=None, help="default: CRITERIA['steps']")
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--time-only", type=int, default=0, help="time this many steps and exit")
    ap.add_argument("--repeat-check", type=int, default=0)
    ap.add_argument("--no-dead-stop", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore a saved state")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    import winqos
    winqos.opt_out()                        # EcoQoS parked these jobs on the low-power cores
    steps = args.steps or CRITERIA["steps"]
    if not args.time_only and (steps is None or CRITERIA["cap_fraction"] is None):
        raise SystemExit("the criteria are not fixed yet (CRITERIA): the probe does not run without its bar")
    torch.set_num_threads(args.threads)
    t_build = time.perf_counter()
    seeds, metas = select()
    b = build_batch(seeds)
    t_build = time.perf_counter() - t_build
    e = tt.ArmR(args.size, args.form, seed=0)
    opt = torch.optim.AdamW(e.parameters(), lr=LR, weight_decay=0.0)
    name = f"{args.form}-{args.size}"
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(STATE, exist_ok=True)
    path = os.path.join(args.out, f"phase3-capacity-{name}.json")
    spath = os.path.join(STATE, f"{name}.pt")

    def step_once():
        loss = tt.objective(b, *e(b))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(e.parameters(), 1.0)
        opt.step()
        return float(loss.detach())

    if args.time_only:
        t0 = time.perf_counter()
        for _ in range(args.time_only):
            step_once()
        dts = (time.perf_counter() - t0) / args.time_only
        print(f"{name}: {dts:.2f} s per step, {b.B} pieces at n = {N}, {args.threads} threads "
              f"(batch built in {t_build:.0f} s); 3,000 steps would take {3000 * dts / 3600:.1f} h")
        return 0
    start_step = 0
    rec = None
    if os.path.exists(spath) and not args.fresh:
        st = torch.load(io.BytesIO(registry.read_bytes(spath)), weights_only=False)
        e.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        start_step, rec = st["step"], st["record"]
        rec.setdefault("resumed_at", []).append({"step": start_step, "at": dt.datetime.now().astimezone()
                                                 .isoformat(timespec="seconds"), "power": power_state()})
        print(f"resumed {name} at step {start_step}", flush=True)
    if rec is None:
        rec = {"what": "the coolant plan's capacity probe (gate CAP)", "form": args.form, "size": args.size,
               "parameters": md.parameters(e.net), "n": N, "n_c": N_C, "seeds": seeds, "pieces": metas,
               "rule": f"first {PER_KIND} seeds of each of {KINDS} from {td.SEED_BASE['probe']} with Pe >= "
                       f"{PE_MIN:.0f} at n = {N}",
               "criteria": CRITERIA, "steps_planned": steps, "lr": LR, "every": args.every,
               "threads": args.threads, "torch": torch.__version__, "device": "cpu",
               "objective": "transport_torch.objective (relative, per group, label-free)",
               "groups": b.column_names(), "tl0_errors_per_piece": np.sqrt(b.err0.cpu().numpy()).tolist(),
               "started": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
               "power_at_start": power_state(), "trajectory": []}
        rec["trajectory"].append(dict(step=0, elapsed_s=0.0, **measure(e, b)))
        registry.write_atomic(path, json.dumps(rec, indent=1).encode("utf-8"))
        save_state(spath, e, opt, 0, rec)
    t0 = time.perf_counter() - (rec["trajectory"][-1]["elapsed_s"] if rec["trajectory"] else 0.0)
    stopped = rec.get("stopped")
    for step in range(start_step + 1, steps + 1):
        if stopped:
            break
        loss = step_once()
        if step % args.every == 0 or step == steps:
            row = dict(step=step, elapsed_s=time.perf_counter() - t0, train_loss=loss, **measure(e, b))
            rec["trajectory"].append(row)
            if not args.no_dead_stop and step >= 0.2 * steps and row["loss"] > 0.95:
                stopped = {"step": step, "why": f"dead network: the loss {row['loss']:.3f} above 0.95 "
                                                f"at or after 20% of the steps"}
                rec["stopped"] = stopped
            registry.write_atomic(path, json.dumps(rec, indent=1).encode("utf-8"))
            save_state(spath, e, opt, step, rec)
            worst = max(row["err"].items(), key=lambda kv: kv[1]["median"])
            print(f"step {step}: loss {row['loss']:.4f}, worst group {worst[0]} {worst[1]['median']:.3f}, "
                  f"shuffled worst {max(v['median'] for v in row['shuffled_err'].values()):.3f} "
                  f"({row['elapsed_s'] / 60:.0f} min)", flush=True)
    last = rec["trajectory"][-1]
    rec["seconds_per_step"] = last["elapsed_s"] / max(1, last["step"])
    rec["verdict"] = verdict(last, CRITERIA["cap_fraction"], JUDGED) if not stopped else {"met": False, "stopped": stopped}
    rec["verdict_all_nine"] = verdict(last, CRITERIA["cap_fraction"]) if not stopped else None
    if args.repeat_check:
        ws = []
        for _ in range(2):
            ex = tt.ArmR(args.size, args.form, seed=0)
            op = torch.optim.AdamW(ex.parameters(), lr=LR, weight_decay=0.0)
            for _k in range(args.repeat_check):
                loss = tt.objective(b, *ex(b))
                op.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ex.parameters(), 1.0)
                op.step()
            ws.append([p.detach().clone() for p in ex.parameters()])
        diff = max(float((p - q).abs().max()) for p, q in zip(*ws))
        rec["repeat_check"] = {"steps": args.repeat_check, "max_weight_difference": diff, "bit_for_bit": diff == 0.0}
    rec["finished"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    rec["power_at_end"] = power_state()
    registry.write_atomic(path, json.dumps(rec, indent=1).encode("utf-8"))
    print(f"CAP {name}: {'met' if rec['verdict']['met'] else 'NOT met'}; {json.dumps(rec['verdict'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
