"""cot_phase3_oracle -- the coolant plan's Phase 3, check 4: the oracle ceiling.

[[chart-operator-stage-0b-and-1-plan]] section 6, check 4: "the exact port responses stand
in for a perfect network.  They are priced at TL0's cost, plus the planned network's
forward pass measured untrained on the laptop, plus the safety switch.  That price must
be under the price bar and under every classical option that reaches the same accuracy.
If even a perfect network cannot pay for itself, no trained one can."  Run first in
Phase 3, before any dataset is built (section 13.4): if it fails, nothing is trained.

**The oracle's accuracy** is the classical expert's (the exact responses): the floor.  The
classical options that reach it on the development problems are the classical solve
itself and finer cutting (more accurate, `cot_phase2_decisive`).

**The oracle's price**, timed in turns against the classical solve on each development
problem, exactly as Phase 2 timed (BLAS and torch threads fixed at 4): the whole coupled
solve of TL0 at the network's coarse size, plus, on every piece, the planned network's
inputs and its two forward passes (trial, and adjoint by the reversed flow; CO-D's U-Net
at 1.1 M parameters with CO-T's inputs, `transport_model`, untrained), plus the safety
switch's two dual residuals (the learned answer's and TL0's).  The timed path computes
TL0's answers and the network's outputs; its accuracy is credited as the floor's, which
is what a perfect network would have.

**Fixed before the first run** (`CRITERIA`, written into the record):
- the switch priced is the cheapest one equivalent to the dual norm within a factor of
  three on the development pieces (the Schwarz estimates; Jacobi is not equivalent), and
  the oracle is also priced with no switch at all, as a bound;
- check 4 passes when, over the development problems whose A/A check holds, the
  oracle's geometric-mean price is under 0.90 **and** under the geometric mean, problem by
  problem, of the cheapest classical option reaching the floor's accuracy (the classical
  solve's 1 or finer cutting's price, whichever is less, from this run's own timings);
- if the oracle fails even with no switch, the failure does not depend on the switch.

Writes ``out/chartop/cot/phase3-oracle.json`` and ``.txt``.  ``--quick``: one problem, two
turns, for the tests; nothing is judged.

    python scripts/cot_phase3_oracle.py [--quick] [--out DIR]
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_v] = "4"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p_ in (HERE, os.path.join(HERE, "scripts")):
    if p_ not in sys.path:
        sys.path.insert(0, p_)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from atlas.chartop import transport as tr  # noqa: E402
from atlas.chartop import transport_coarse as tco  # noqa: E402
from atlas.chartop import transport_model as tm  # noqa: E402
from atlas.chartop import transport_switch as sw  # noqa: E402
from atlas.studio import cot  # noqa: E402
from atlas.studio import cot_layouts as CL  # noqa: E402
from atlas.studio.pieces import cut_channels  # noqa: E402

import cot_phase2_decisive as P2  # noqa: E402

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

PRICE_BAR = P2.PRICE_BAR
AA_BAND = P2.AA_BAND
TURNS = 5
EQUIV = 3.0
CRITERIA = {
    "oracle": "the whole coupled solve of TL0 at the network's coarse size, plus every piece's "
              "network inputs and two untrained forward passes (CO-D's U-Net, 1.1 M parameters, CO-T's "
              "inputs), plus the switch's two dual residuals; its accuracy credited as the floor's",
    "switch": "the cheapest Schwarz estimate whose ratio to the exact dual norm stays within a factor "
              "of 3 on the development pieces; also priced with no switch, as a bound",
    "check 4": "over the development problems whose A/A check holds, the oracle's geometric-mean "
               "price is under 0.90 and under the geometric mean of min(1, finer cutting's price) "
               "per problem",
    "A/A": "the classical expert timed twice per turn; outside [0.90, 1.10] a problem's prices are "
           "reported, not judged",
}
SWITCHES = (("schwarz-16", dict(mode="schwarz", n_c=16, overlap=1)),
            ("schwarz-32", dict(mode="schwarz", n_c=32, overlap=1)))


class Oracle:
    """What a perfect network behind the switch costs: TL0's answers, the network's
    forward passes, and the switch's residuals (switch=None: none)."""

    def __init__(self, n_c: int, net, switch: dict | None):
        self.tl0 = tco.TL0TransportExpert(n_c=n_c)
        self.net = net
        self.switch = switch
        self.calls = 0
        self.envelope = None
        self.name = "oracle"

    def respond(self, problems, pb_of):
        out = []
        for p in problems:
            Q = pb_of(p.n).Q
            r = self.tl0.response(p, Q)
            for rev in (False, True):
                x, s, _ = tm.inputs(p, rev)
                with torch.no_grad():
                    self.net(torch.from_numpy(x)[None], torch.from_numpy(s)[None])
            if self.switch is not None:
                kw = dict(self.switch)
                mode = kw.pop("mode")
                n_c = min(kw.pop("n_c"), p.n // 2)
                cache: dict = {}
                sw.piece_eta(p, r, Q, mode, n_c=n_c, cache=cache, **kw)      # the learned answer's
                sw.piece_eta(p, r, Q, mode, n_c=n_c, cache=cache, **kw)      # TL0's
            out.append(r)
            self.calls += 1
        return out


def equivalence(L, n_c_tl0: int) -> dict:
    """Each Schwarz estimate's ratio to the exact dual norm on TL0's answers, every piece."""
    tl0 = tco.TL0TransportExpert(n_c=n_c_tl0)
    out = {k: [] for k, _ in SWITCHES}
    for p in L.problems:
        Q = L.pb(p.n).Q
        r = tl0.response(p, Q)
        ex = sw.piece_eta(p, r, Q, "exact")["total"]
        for k, kw in SWITCHES:
            kw = dict(kw)
            mode = kw.pop("mode")
            n_c = min(kw.pop("n_c"), p.n // 2)
            out[k].append(sw.piece_eta(p, r, Q, mode, n_c=n_c, **kw)["total"] / ex)
    return {k: [float(min(v)), float(max(v))] for k, v in out.items()}


def problem(name: str, tier: str, quick: bool, turns: int, net) -> dict:
    spec, _g = CL.build(name, tier)
    n = CL.CUT[name]["n"][tier] if not quick else 32
    cols = CL.CUT[name]["cols"]
    base = cut_channels(spec, n, cols=cols)
    fine = cut_channels(spec, n // 2, cols=2 * cols, rows=2)
    Lb, Lf = cot.layout(base, spec), cot.layout(fine, spec)
    n_c = min(16, n // 2)
    eq = equivalence(Lb, n_c)
    faithful = [k for k, _ in SWITCHES if eq[k][0] >= 1.0 / EQUIV and eq[k][1] <= EQUIV]
    plan = [("classical", Lb, lambda: tr.ClassicalTransportExpert()),
            ("classical (A/A)", Lb, lambda: tr.ClassicalTransportExpert()),
            ("finer", Lf, lambda: tr.ClassicalTransportExpert()),
            (f"TL0-{n_c}", Lb, lambda: tco.TL0TransportExpert(n_c=n_c)),
            ("oracle, no switch", Lb, lambda: Oracle(n_c, net, None))]
    for k, kw in SWITCHES:
        plan.append((f"oracle, {k}", Lb, (lambda kw=kw: Oracle(n_c, net, kw))))
    times = {k: [] for k, _L, _m in plan}
    for t in range(turns):
        order = plan if t % 2 == 0 else plan[::-1]
        for k, L, make in order:
            ex = make()
            t0 = time.perf_counter()
            L.couple(ex.respond(L.problems, L.pb))
            times[k].append(time.perf_counter() - t0)
    med = {k: float(np.median(v)) for k, v in times.items()}
    price = {k: med[k] / med["classical"] for k in med}
    aa = price["classical (A/A)"]
    best = [k for k in faithful]
    cheapest = min(best, key=lambda k: price[f"oracle, {k}"]) if best else None
    return {"layout": name, "tier": tier, "n": n, "n_c": n_c, "pieces": len(base.pieces),
            "pe_max": max(p.peclet() for p in Lb.problems), "price": price, "aa": aa,
            "aa_ok": AA_BAND[0] <= aa <= AA_BAND[1], "equivalence": eq, "faithful": faithful,
            "switch": cheapest,
            "oracle": price[f"oracle, {cheapest}"] if cheapest else None,
            "oracle_no_switch": price["oracle, no switch"],
            "competitor": min(1.0, price["finer"]), "tl0_price": price[f"TL0-{n_c}"], "turns": turns}


def judge(problems: list) -> dict:
    ok = [p for p in problems if p["aa_ok"] and p["oracle"] is not None]

    def gm(xs):
        return float(math.exp(np.mean(np.log(xs)))) if xs else None
    o, o0, c = gm([p["oracle"] for p in ok]), gm([p["oracle_no_switch"] for p in ok]), gm([p["competitor"] for p in ok])
    passes = bool(o is not None and o < PRICE_BAR and o < c)
    passes0 = bool(o0 is not None and o0 < PRICE_BAR and o0 < c)
    return {"counted": len(ok), "oracle_geomean": o, "oracle_no_switch_geomean": o0,
            "competitor_geomean": c, "check4": passes, "check4_with_no_switch": passes0,
            "independent_of_the_switch": (not passes0) if not passes else None}


def run(quick: bool, out_dir: str) -> dict:
    t0 = time.time()
    net = tm.network("1M").eval()
    fwd = {n: tm.forward_seconds(n, repeats=3, threads=4) for n in ((32,) if quick else (32, 64, 128))}
    rec = {"what": "CO-T Phase 3, check 4: the oracle ceiling (the coolant plan)",
           "written": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "commit": P2._commit(), "quick": quick, "criteria": CRITERIA, "blas": P2._blas(),
           "torch_threads": torch.get_num_threads(), "forward": fwd,
           "network_parameters": int(sum(p.numel() for p in net.parameters())), "problems": []}
    try:
        from atlas.workbench import machine
        rec["machine"] = machine.machine_record()
    except Exception as exc:
        rec["machine"] = {"error": type(exc).__name__}
    todo = [("dev-block", "slow")] if quick else [(n_, t_) for n_ in CL.NAMES for t_ in CL.TIERS]
    for name, tier in todo:
        p = problem(name, tier, quick, 2 if quick else TURNS, net)
        rec["problems"].append(p)
        print(f"{name} {tier} (n {p['n']}): oracle {p['oracle']} (switch {p['switch']}), no switch "
              f"{p['oracle_no_switch']:.2f}, competitor {p['competitor']:.2f}, finer {p['price']['finer']:.2f}, "
              f"TL0 {p['tl0_price']:.2f}, A/A {p['aa']:.3f}; Schwarz/exact {p['equivalence']}",
              flush=True)
    rec["verdict"] = None if quick else judge(rec["problems"])
    rec["seconds"] = round(time.time() - t0, 1)
    os.makedirs(out_dir, exist_ok=True)
    stem = "phase3-oracle-quick" if quick else "phase3-oracle"
    P2._write(os.path.join(out_dir, stem + ".json"), json.dumps(rec, indent=1, default=float) + "\n")
    P2._write(os.path.join(out_dir, stem + ".txt"), report(rec))
    return rec


def report(rec: dict) -> str:
    L = [rec["what"], "", f"written  {rec['written']}", f"commit   {rec['commit']}",
         f"threads  BLAS fixed at 4: {rec['blas'].get('fixed_at_4')}; torch {rec['torch_threads']}",
         f"network  {rec['network_parameters']} parameters, untrained; two forward passes per piece: "
         + ", ".join(f"n {k} {v['seconds_two_passes'] * 1e3:.1f} ms" for k, v in rec["forward"].items()),
         "", "CRITERIA (fixed before the first run)"]
    for k, v in rec["criteria"].items():
        L.append(f"  {k}: {v}")
    L += ["", "problem              n    A/A    classical  finer  TL0   oracle(no switch)  oracle(switch)       competitor"]
    for p in rec["problems"]:
        L.append(f"  {p['layout']:12s} {p['tier']:4s} {p['n']:4d} {p['aa']:.3f}  1.000      {p['price']['finer']:.2f}   "
                 f"{p['tl0_price']:.2f}  {p['oracle_no_switch']:.2f}               "
                 f"{'-' if p['oracle'] is None else format(p['oracle'], '.2f')} ({p['switch']})   {p['competitor']:.2f}"
                 f"{'' if p['aa_ok'] else '   (A/A failed: not counted)'}")
        L.append(f"      Schwarz over exact: {p['equivalence']}; faithful within 3: {p['faithful']}")
    if rec.get("verdict"):
        v = rec["verdict"]
        L += ["", "VERDICT (check 4)",
              f"  counted problems: {v['counted']}",
              f"  oracle {v['oracle_geomean']}, with no switch {v['oracle_no_switch_geomean']}, "
              f"competitor {v['competitor_geomean']}, bar {PRICE_BAR}",
              f"  check 4 passes: {v['check4']}; with no switch: {v['check4_with_no_switch']}"]
    L += [f"  {rec['seconds']} s", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "chartop", "cot"))
    a = ap.parse_args(argv)
    rec = run(a.quick, a.out)
    print("verdict:", rec["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
