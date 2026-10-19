"""After Stage 0 (W356, W357): the output forms of `atlas.chartop.forms`, checked with
untrained weights only -- zero heads, or seeded random weights -- and never trained.

    python scripts/chartop_forms_controls.py [--count 24]

Not a gate, and no registration.  The controls of `scripts/chartop_posthoc_zero_source.py`
and the diagnostics of `scripts/chartop_diagnose.py`, re-run on the new forms, with the
same definitions (imported from those scripts):

1. **The coupled answer** on all six benchmark problems (n = 64), each expert: as
   registered; with **every temperature lowered by 300 K** (`benchmarks.datum_shift`:
   the Dirichlet data and, for the steps, T_old, so their source vanishes); and, on
   the steady three, with the modes taken from a call whose source is a unit-RMS
   random field (W357's control; the particular field then taken as zero, which it is
   where g = 0).  A step's reference energy contains the level (sigma int T^2) and moves
   with the shift, so the shift is judged on the absolute error ||u - u~||_M; steady,
   the reference energy does not move.
2. **The modes, piece by piece**, against the exact ones: e_H and its port groups as the
   post-hoc script defines them, the uniform state's residual (e_K, e_K,inf), and
   e_H(theta) at each piece's own level, theta being the mean of its four vertex values
   in the classical coupled solution over the RMS of the rest of its coefficients.
3. **What the form asks** (the diagnose script's second part) on validation seeds,
   n = 32, 64, 128: the side quotient (Stage 0's, untouched by the projection), the
   hats' smallest quotient once the projection carries their sum, and the level field's
   quotient N_1 = (u_p(1) - u_1^0) / (a beta).  And e_K of the as-initialised forms
   against Stage 0's principal part on the same pieces.

The experts: the principal part (Stage 0's zero head); the kernel and split forms with
zero heads, as initialised; Stage 0's bubble with seeded random weights (normal, every
parameter) at two scales that bracket arm B -- on the steady benchmarks this bubble's
coupled errors are 3.8-14% at 0.002 and 47-154% at 0.02, B's 24-67%, and its largest
hat-sum defect 9.4e-4 and 1.2e-2, B's 5.3e-3 to 6.1e-3; the kernel form with that
bubble's weights copied, its level row zero, so that on a sourceless piece the
projection is the only difference; the split form with random weights at the same
scales; and the classical 16-mode superelement.  Arm B's trained weights enter part 2
only, as a check of the measure on the form B was trained in.

Writes ``out/chartop/forms-controls.json``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import sys
import time
from dataclasses import replace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from atlas.chartop import benchmarks as bm, data, fe, forms, gates as G, host as hs  # noqa: E402
from atlas.chartop import model as md, ports  # noqa: E402
from atlas.chartop.expert import LearnedExpert  # noqa: E402
from atlas.chartop.host import Response  # noqa: E402
from atlas.chartop.train import load_expert  # noqa: E402

STDS = (0.002, 0.02)
SEED = 0
SHIFT = -bm.T_COLD


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


POSTHOC = _script("chartop_posthoc_zero_source")      # GROUPS, group_of, mode_errors
DIAGNOSE = _script("chartop_diagnose")                 # group_values


def randomised(net, std: float, seed: int = SEED):
    torch.manual_seed(seed)
    for prm in net.parameters():
        torch.nn.init.normal_(prm, 0.0, std)
    return net.eval()


def kernel_from_bubble(bub: md.Expert) -> forms.KernelExpert:
    """The kernel form carrying a Stage 0 bubble's weights, its level row zero."""
    k = forms.KernelExpert("kernel", bub.backbone_name, bub.size, bub.kind, bub.m)
    sd = {kk: v.clone() for kk, v in bub.net.state_dict().items()}
    sd["head.weight"] = torch.cat([sd["head.weight"], torch.zeros_like(sd["head.weight"][:1])])
    sd["head.bias"] = torch.cat([sd["head.bias"], torch.zeros_like(sd["head.bias"][:1])])
    k.net.load_state_dict(sd)
    return k.eval()


class ModesUnderLoad:
    """The base expert's modes from a call in which every piece's source is a unit-RMS
    random field (the generator's, correlation length 0.25); the particular field taken
    as zero, which is exact only for sourceless pieces."""

    def __init__(self, base, seed: int = 11):
        self.base, self.seed = base, seed
        self.envelope = getattr(base, "envelope", None)
        self.calls = 0

    def respond(self, problems, pb_of, modes_from=None):
        loaded = []
        for k, p in enumerate(problems):
            assert float(np.abs(p.g).max()) == 0.0, "only for sourceless pieces"
            t = np.arange(p.n + 1) / p.n
            XI, ET = np.meshgrid(t, t)
            f = data.Field(np.random.default_rng(self.seed + k), 0.25)(XI, ET)
            loaded.append(replace(p, g=f / float(np.sqrt(np.mean(f ** 2)))))
        resp = self.base.respond(loaded, pb_of)
        self.calls += len(problems)
        return [Response(r.H, np.zeros_like(r.up)) for r in resp]


def coupled(L, mono, expert) -> dict:
    cp = L.solve(expert)
    num, den = cp.energy_error(mono.fields)
    span = max(float(np.max(u)) for u in mono.fields) - min(float(np.min(u)) for u in mono.fields)
    mx = max(float(np.max(np.abs(u - v))) for u, v in zip(mono.fields, cp.fields))
    # an error at rounding is a floor, and a ratio of two floors says nothing
    return {"energy_abs": num, "reference_energy": den, "energy_rel": num / den,
            "max_over_range": mx / span, "at_rounding": bool(num / den < 1e-9)}, cp


def uniform_residuals(expert, problems, pb_of) -> list:
    ups = [forms.uniform_state(p) for p in problems]
    resp = expert.respond(ups, pb_of)
    out = []
    for p, r in zip(problems, resp):
        pb = pb_of(p.n)
        out.append(r.up + r.H @ forms.kernel_coefficients(pb.kind, pb.n, pb.m) - 1.0)
    return out


def summarise_modes(rows: list) -> dict:
    eh = np.array([r["e_H"] for r in rows])
    out = {"e_H_median": float(np.median(eh)), "e_H_min": float(eh.min()), "e_H_max": float(eh.max()),
           "e_K_median": float(np.median([r["e_K"] for r in rows])),
           "e_K_max": float(np.max([r["e_K"] for r in rows])),
           "e_K_inf_max": float(np.max([r["e_K_inf"] for r in rows])),
           "e_H_theta_median": float(np.median([r["e_H_theta"] for r in rows])),
           "groups": {g: {"share_median": float(np.median([r[g]["share"] for r in rows])),
                          "relative_median": float(np.median([r[g]["relative"] for r in rows]))}
                      for g, _lo, _hi in POSTHOC.GROUPS}}
    return out


def benchmarks_part(experts: dict, b_expert) -> list:
    out = []
    for name in bm.NAMES:
        for problem in G.BENCHMARK_PROBLEMS:
            t0 = time.perf_counter()
            kw = {"envelope": data.ENVELOPE} if name == "s-channel" else {}
            b = bm.build(name, G.BENCHMARK_N, problem, **kw)
            bs = bm.datum_shift(b, SHIFT)
            L, Ls = b.layout("sine-hat", 16), bs.layout("sine-hat", 16)
            mono, monos = hs.monolithic(L), hs.monolithic(Ls)
            rec = {"name": name, "problem": problem, "pieces": len(L.problems), "shift_K": SHIFT,
                   "coupled": {}, "modes": {}}
            # the exact pieces first: modes, port matrices, each piece's level theta
            r0, cl = coupled(L, mono, experts["classical"])
            r1, _ = coupled(Ls, monos, experts["classical"])
            rec["coupled"]["classical"] = {"as registered": r0, "datum shifted": r1,
                                           "shifted_over_registered_abs": r1["energy_abs"] / r0["energy_abs"]}
            exact = []
            for i, p in enumerate(L.problems):
                pb = L.pb(p.n)
                ex = fe.Exact(p.matrix(), p.n)
                H = cl.responses[i].H
                d = np.diag(ex.port_matrix(pb.Q, H))
                c1 = forms.kernel_coefficients(pb.kind, pb.n, pb.m)
                ci = L.R[i] @ cl.c
                lev = float(c1 @ ci / (c1 @ c1))
                xi = ci - lev * c1
                theta = abs(lev) / max(float(np.sqrt(np.mean(xi ** 2))), 1e-12 * max(abs(lev), 1.0))
                exact.append((p, pb, ex, H, d, float(d.sum()), theta))
            del cl
            rec["classical_e_K_inf_max"] = max(
                float(np.abs(r).max()) for r in uniform_residuals(experts["classical"], L.problems, L.pb))
            per = {}
            for label, ex_ in list(experts.items()) + ([("B (trained, bubble)", b_expert)] if b_expert else []):
                if label == "classical":
                    continue
                if not label.startswith("B ("):
                    r0, cp = coupled(L, mono, ex_)
                    r1, _ = coupled(Ls, monos, ex_)
                    row = {"as registered": r0, "datum shifted": r1,
                           "shifted_over_registered_abs": r1["energy_abs"] / r0["energy_abs"]}
                    if problem == "steady":
                        row["modes under a random source"] = coupled(L, mono, ModesUnderLoad(ex_))[0]
                    rec["coupled"][label] = row
                    resp = cp.responses
                    del cp
                else:
                    resp = ex_.respond(L.problems, L.pb)
                res_u = uniform_residuals(ex_, L.problems, L.pb)
                rows = []
                for i, (p, pb, ex, H, d, trL, theta) in enumerate(exact):
                    me = POSTHOC.mode_errors(p, pb, resp[i].H, ex, H, d)
                    km = forms.kernel_measures(res_u[i], ex.A_I, trL)
                    me.update(e_K=km["e_K"], e_K_inf=km["e_K_inf"], theta=theta,
                              e_H_theta=forms.level_weighted_e_H(me["e_H"], km["e_K"], theta))
                    rows.append(me)
                per[label] = rows
                del resp, res_u
            base = per["principal part"]
            for label, rows in per.items():
                s = summarise_modes(rows)
                s["pieces_worse_than_principal_part"] = int(sum(a["e_H"] > c["e_H"]
                                                                for a, c in zip(rows, base)))
                rec["modes"][label] = s
            rec["theta_median"] = float(np.median([r["theta"] for r in base]))
            rec["elapsed_s"] = time.perf_counter() - t0
            print(json.dumps({"name": name, "problem": problem,
                              "energy_rel": {k: round(v["as registered"]["energy_rel"], 6)
                                             for k, v in rec["coupled"].items()},
                              "shift_ratio": {k: v["shifted_over_registered_abs"]
                                              for k, v in rec["coupled"].items()},
                              "elapsed_s": round(rec["elapsed_s"], 1)}), flush=True)
            out.append(rec)
    return out


def diagnose_part(count: int) -> dict:
    """What the kernel form asks, on validation seeds (classical solves only)."""
    rec = {"seeds": [data.SEEDS["val"] + 100, data.SEEDS["val"] + 100 + count - 1],
           "as_initialised": [], "quotients": {}}
    pp = LearnedExpert(md.Expert("unet", "0.3M"), envelope=data.ENVELOPE)
    zk = LearnedExpert(forms.KernelExpert("kernel", "unet", "1M"), envelope=data.ENVELOPE)
    shares = []
    for i in range(count):
        e = data.draw(data.SEEDS["val"] + 100 + i, n=64)
        p = e.problem
        pb = ports.basis("sine-hat", 64, 16)
        ex = fe.Exact(p.matrix(), 64)
        H = ex.modes(pb.Q)
        trL = float(np.trace(ex.port_matrix(pb.Q, H)))
        row = {"seed": e.seed, "fo": e.fo, "sup_mu": e.envelope["sup_abs_mu"]}
        for label, x in (("principal part", pp), ("kernel, zero heads", zk)):
            km = forms.kernel_measures(uniform_residuals(x, [p], lambda n_: pb)[0], ex.A_I, trL)
            Ht = x.respond([p], lambda n_: pb)[0].H
            E = Ht - H
            ek = np.einsum("ik,ik->k", E, np.asarray(ex.A_I @ E))
            row[label] = {"e_K": km["e_K"], "e_K_inf": km["e_K_inf"],
                          "e_H": float(np.sqrt(ek.sum() / trL)),
                          "share_of_eH2": DIAGNOSE.group_values(ek / ek.sum())}
        shares.append(row["kernel, zero heads"]["share_of_eH2"])
        rec["as_initialised"].append(row)
    rec["as_initialised_median"] = {
        label: {"e_K_median": float(np.median([r[label]["e_K"] for r in rec["as_initialised"]])),
                "e_K_inf_max": float(np.max([r[label]["e_K_inf"] for r in rec["as_initialised"]])),
                "e_H_median": float(np.median([r[label]["e_H"] for r in rec["as_initialised"]]))}
        for label in ("principal part", "kernel, zero heads")}
    rec["share_of_eH2_median_kernel_zero_heads"] = {k: float(np.median([s[k] for s in shares]))
                                                     for k in shares[0]}
    for n in (32, 64, 128):
        ps = md.plain_square(n, "sine-hat", 16, "cpu")
        ks = forms.kernel_square(n, "sine-hat", 16, "cpu")
        beta = ps.beta[1:-1, 1:-1].numpy()
        W = ks.W.numpy()[:4]
        t = np.arange(1, n) / n
        XI, ET = np.meshgrid(t, t)
        dcorner = np.minimum.reduce([np.hypot(XI - x, ET - y) for x in (0, 1) for y in (0, 1)])
        Q = ports.basis("sine-hat", n, 16).Q
        rows = []
        for i in range(min(8, count)):
            e = data.draw(data.SEEDS["val"] + 100 + i, n=n)
            a = data.arrays(e)
            lab = data.labels_from_arrays(a, Q)
            s0 = float(a["s0"])
            s0t = torch.tensor([s0], dtype=torch.float64)
            H = lab["H"].T.reshape(64, n - 1, n - 1)
            H0 = ps.principal(s0t)[0].numpy()
            D = (H - H0) / beta[None]
            # the exact level field: the particular field of the unit source, stored coefficients
            dd = {k: np.asarray(a[k], np.float64) for k in ("a11", "a12", "a22", "sigma", "w")}
            M = fe.assemble(n, dd["a11"], dd["a12"], dd["a22"], dd["sigma"])
            u1 = fe.Exact(M, n).particular((fe.mass(n, dd["w"]) @ np.ones((n + 1) ** 2))[fe.grid(n)[1]])
            u1 = u1.reshape(n - 1, n - 1)
            u10 = forms.level_principal(ps, s0t)[0].numpy()
            N1 = (u1 - u10) / (np.abs(u10).max() * beta)
            tau = (W * D[:4]).sum(0) / (W * W).sum(0)
            Nh = D[:4] - W * tau[None]
            k1 = np.unravel_index(int(np.argmax(np.abs(N1))), N1.shape)
            rows.append({"seed": e.seed, "fo": e.fo,
                         "sides_max_abs": float(np.abs(D[4:]).max()),
                         "hats_rms_stage0": float(np.sqrt(np.mean(D[:4] ** 2))),
                         "hats_max_stage0": float(np.abs(D[:4]).max()),
                         "hats_rms_projected": float(np.sqrt(np.mean(Nh ** 2))),
                         "hats_max_projected": float(np.abs(Nh).max()),
                         "level_rms": float(np.sqrt(np.mean(N1 ** 2))),
                         "level_max_abs": float(np.abs(N1).max()),
                         "level_max_at_corner_distance": float(dcorner[k1])})
        med = {k: float(np.median([r[k] for r in rows])) for k in rows[0] if k not in ("seed", "fo")}
        rec["quotients"][str(n)] = {"examples": rows, "median": med}
        print(f"n={n}: " + ", ".join(f"{k} {v:.3g}" for k, v in med.items()), flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--b-weights", default=os.path.join(ROOT, "out", "chartop", "runs", "B", "best.pt"))
    args = ap.parse_args()
    torch.set_num_threads(8)
    t0 = time.perf_counter()
    nets = {"principal part": md.Expert("unet", "0.3M"),
            "kernel, zero heads": forms.KernelExpert("kernel", "unet", "1M"),
            "split, zero heads": forms.KernelExpert("split", "unet", "1M")}
    for std in STDS:
        bub = randomised(md.Expert("unet", "1M"), std)
        nets[f"bubble, random {std:g}"] = bub
        nets[f"kernel, the bubble's weights {std:g}"] = kernel_from_bubble(bub)
        nets[f"split, random {std:g}"] = randomised(forms.KernelExpert("split", "unet", "1M"), std)
    experts = {k: LearnedExpert(v.eval(), envelope=data.ENVELOPE) for k, v in nets.items()}
    experts["classical"] = hs.ClassicalExpert()
    b_expert = None
    if os.path.exists(args.b_weights):
        b_expert = LearnedExpert(load_expert(args.b_weights, "cpu"), envelope=data.ENVELOPE)
    rec = {"written": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "what": "untrained weights only (zero heads, or seeded random weights); not a gate, "
                   "nothing registered",
           "random_std": list(STDS), "seed": SEED, "shift_K": SHIFT,
           "experts": {k: (v.config() if hasattr(v, "config") else "classical 16-mode superelement")
                       for k, v in nets.items()},
           "b_weights": os.path.relpath(args.b_weights, ROOT) if b_expert else None}
    rec["benchmarks"] = benchmarks_part(experts, b_expert)
    rec["diagnose"] = diagnose_part(args.count)
    rec["elapsed_s"] = time.perf_counter() - t0
    path = os.path.join(ROOT, "out", "chartop", "forms-controls.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    for k in range(20):
        try:
            os.replace(tmp, path)
            break
        except OSError:
            time.sleep(0.5 * (k + 1))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
