"""cot_phase3_checks -- the coolant dataset's checks 1, 2, 3 and 5, before it is frozen
([[chart-operator-stage-0b-and-1-plan]] section 6; check 4, the oracle ceiling, is a reading
since the owner's amendment, section 15, and its first reading is `cot_phase3_oracle.py`'s).

Every bar is fixed here before the checks' first run (`CRITERIA`); any failure stops the plan.

- **check 1, coverage** (from the training split's manifest, `cot_phase3_data.py`): the
  declared mix is filled.  Shares of resolution, kind, profile (flowing pieces), family and
  steady pieces within ``SHARE_TOL`` of `transport_data`'s declared ones, with at least as
  many n = 128 pieces as n = 32 (the owner's amendment); in every resolution the flowing
  pieces' Peclet numbers fill every tenth of [0, cap(n)] to at least half the uniform
  count; the transient pieces' Fourier numbers fill each of the four decades of
  [1e-2, 1e2] to at least half the log-uniform count.
- **check 2, label quality**: 40 n = 128 training pieces (the 20 flowing ones of highest
  Peclet number and the first 20 others, in seed order), each refined to n = 256 by
  bilinear interpolation of its stored chart, stream function and source (the same
  piecewise-bilinear map, flow and source, exactly), its labels recomputed there.  The
  relative change of Lam and of chi (Frobenius) at most 5% on at least 95% of the
  pieces (the owner, 2026-11-19: "5% on 95% of the sample (Recommended)"); the largest
  reported.
- **check 3, the control**: TL0 at the network's coarse size, n_c = min(16, n/2), misses E4
  on every development problem (read from Phase 2's record, `phase2-decisive.json`).
- **check 5, loss-gate alignment**: on the six development problems, a fixed set of
  experts is ranked by the training loss (`transport_torch.objective`, label-free, on the
  problems' own pieces) and by the coupled error (the excess of the coupled field's
  relative energy error over the classical floor, over TL0's excess, averaged over the
  problems).  The set: TL0; the classical expert; 9 synthetic experts that each remove
  TL0's error in one of the nine groups (trial hats, j1-3, j4-8, j9-15; the same for
  the adjoint; the particular field) and keep the rest; 9 that keep one group's error
  and remove the rest; 20 with each group's error scaled by a factor drawn uniformly in
  [0, 1] (seed 0); and the capacity probe's trained networks, where their state exists.
  A synthetic expert's output is the exact answer plus its factors times TL0's error,
  which is coarse-exact (TL0's error is Galerkin-orthogonal to the coarse space).  The
  Spearman rank correlation at least 0.80 (the owner, 2026-11-19: "0.80 (Recommended)").

Writes ``out/chartop/cot/phase3-checks.json`` (and ``.txt``), one section per check run.

    python scripts/cot_phase3_checks.py --check 1 2 3 5 [--data DIR]
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

import chartop_registry as registry  # noqa: E402
from atlas.chartop import chart as ch  # noqa: E402
from atlas.chartop import transport as tr  # noqa: E402
from atlas.chartop import transport_data as td  # noqa: E402

RECORDS = os.path.join(ROOT, "out", "chartop", "cot")
DATA = os.path.join(ROOT, "out", "chartop", "data", "cot")
STATE = os.path.join(ROOT, "out", "chartop", "runs", "cot-capacity")

SHARE_TOL = 0.02
CHECK2_TOL, CHECK2_SHARE, CHECK2_HIGH, CHECK2_OTHER = 0.05, 0.95, 20, 20
CHECK5_RHO = 0.80
CRITERIA = {
    "check 1": f"declared shares within {SHARE_TOL}; n=128 count >= n=32 count; every tenth of [0, cap(n)] "
               f"of flowing Pe and every decade of transient Fo at least half its uniform count",
    "check 2": f"relative change of Lam and chi from n=128 to n=256 at most {CHECK2_TOL} on at least "
               f"{CHECK2_SHARE:.0%} of {CHECK2_HIGH + CHECK2_OTHER} pieces (the owner, 2026-11-19: "
               f"'5% on 95% of the sample (Recommended)')",
    "check 3": "TL0 at n_c = min(16, n/2) misses E4 on every development problem (Phase 2's record)",
    "check 5": f"Spearman correlation of the training loss and the coupled error over the fixed set of experts "
               f"at least {CHECK5_RHO} (the owner, 2026-11-19: '0.80 (Recommended)')",
}


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# check 1
# ---------------------------------------------------------------------------


def check1(manifest: dict) -> dict:
    cov = manifest["coverage"]
    N = cov["count"]
    fails, rows = [], []

    def share(name, got, want):
        ok = abs(got - want) <= SHARE_TOL
        rows.append({"what": name, "share": got, "declared": want, "ok": ok})
        if not ok:
            fails.append(name)

    for n, p in td.RESOLUTIONS:
        share(f"n = {n}", cov["by_n"][str(n)] / N, p)
    if cov["by_n"]["128"] < cov["by_n"]["32"]:
        fails.append("fewer n = 128 pieces than n = 32")
    for k, p in td.KINDS:
        share(f"kind {k}", cov["by_kind"][k] / N, p)
    flowing = N - cov["by_kind"]["solid"]
    for k, p in td.PROFILES:
        share(f"profile {k} (flowing)", cov["by_profile_flowing"][k] / flowing, p)
    share("family conformal-map", cov["by_family"]["conformal-map"] / N, 0.2)
    share("steady", cov["steady"] / N, td.d0.P_STEADY)
    for n, _p in td.RESOLUTIONS:
        h = cov["pe_tenths_of_cap_flowing"][str(n)]
        floor = 0.5 * sum(h) / 10
        bad = [i for i, c in enumerate(h) if c < floor]
        rows.append({"what": f"Pe tenths at n = {n}", "counts": h, "floor": floor, "ok": not bad})
        if bad:
            fails.append(f"Pe at n = {n}: tenths {bad} under half the uniform count")
    dec = cov["fo_decades"]
    floor = 0.5 * sum(dec.values()) / 4
    bad = [d for d, c in dec.items() if c < floor]
    rows.append({"what": "Fo decades", "counts": dec, "floor": floor, "ok": not bad})
    if bad:
        fails.append(f"Fo decades {bad} under half the log-uniform count")
    return {"passed": not fails, "fails": fails, "rows": rows, "count": N,
            "manifest_content_sha256": manifest.get("content_sha256")}


# ---------------------------------------------------------------------------
# check 2
# ---------------------------------------------------------------------------


def refine(a: np.ndarray) -> np.ndarray:
    """Bilinear refinement of a nodal array (n+1, n+1, ...) to (2n+1, 2n+1, ...): exact for a
    piecewise-bilinear function."""
    a = np.asarray(a, np.float64)
    n = a.shape[0] - 1
    out = np.zeros((2 * n + 1, 2 * n + 1) + a.shape[2:])
    out[::2, ::2] = a
    out[::2, 1::2] = 0.5 * (a[:, :-1] + a[:, 1:])
    out[1::2, ::2] = 0.5 * (a[:-1] + a[1:])
    out[1::2, 1::2] = 0.25 * (a[:-1, :-1] + a[:-1, 1:] + a[1:, :-1] + a[1:, 1:])
    return out


def _examples(data_dir: str, split: str, n: int):
    for f in sorted(glob.glob(os.path.join(data_dir, f"{split}_n{n}_*.npz"))):
        z = np.load(f)
        for i in range(len(z["seed"])):
            yield {k: z[k][i] for k in ("X", "psi", "g", "n", "fo", "pe", "seed", "kind", "Lam", "chi")}


def check2(data_dir: str) -> dict:
    ex = list(_examples(data_dir, "train", 128))
    flowing = [e for e in ex if int(e["kind"]) != 0]
    high = sorted(flowing, key=lambda e: -float(e["pe"]))[:CHECK2_HIGH]
    hs = {int(e["seed"]) for e in high}
    other = [e for e in ex if int(e["seed"]) not in hs][:CHECK2_OTHER]
    rows = []
    for e in high + other:
        t0 = time.perf_counter()
        fo = float(e["fo"])
        p = tr.make_piece(ch.Chart(refine(e["X"]), "refined"), refine(e["psi"]), k=1.0, alpha=1.0,
                          fo=None if math.isnan(fo) else fo, g=refine(e["g"]))
        lab = td.labels(p)
        dL = float(np.linalg.norm(lab["Lam"] - e["Lam"]) / np.linalg.norm(lab["Lam"]))
        nc = float(np.linalg.norm(lab["chi"]))
        dc = float(np.linalg.norm(lab["chi"] - e["chi"]) / nc) if nc > 0 else 0.0
        rows.append({"seed": int(e["seed"]), "pe": float(e["pe"]), "pe_256": p.peclet(), "kind": int(e["kind"]),
                     "steady": math.isnan(fo), "lam_change": dL, "chi_change": dc,
                     "ok": dL <= CHECK2_TOL and dc <= CHECK2_TOL, "seconds": time.perf_counter() - t0})
        print(f"check 2: seed {rows[-1]['seed']} Pe {rows[-1]['pe']:.0f}: Lam {dL:.3g}, chi {dc:.3g}", flush=True)
    share = sum(r["ok"] for r in rows) / len(rows)
    return {"passed": share >= CHECK2_SHARE, "share_within": share, "pieces": len(rows),
            "largest_lam_change": max(r["lam_change"] for r in rows),
            "largest_chi_change": max(r["chi_change"] for r in rows), "rows": rows}


# ---------------------------------------------------------------------------
# check 3
# ---------------------------------------------------------------------------


def check3() -> dict:
    rec = json.load(open(os.path.join(RECORDS, "phase2-decisive.json"), encoding="utf-8"))
    rows = []
    for p in rec["problems"]:
        nc = min(16, p["n"] // 2)
        key = f"TL0-{nc}"
        rows.append({"problem": f"{p['layout']} {p['tier']}", "n": p["n"], "arm": key, "meets_e4": p["e4"].get(key),
                     "energy_rel": p["accuracy"][key]["energy_rel"], "floor": p["floor"]["energy_rel"]})
    return {"passed": all(r["meets_e4"] is False for r in rows), "rows": rows,
            "record": "out/chartop/cot/phase2-decisive.json", "record_commit": rec.get("commit")}


# ---------------------------------------------------------------------------
# check 5
# ---------------------------------------------------------------------------


def _groups():
    from atlas.chartop import transport_torch as tt
    return list(tt.groups(16).values())


def synthetic_set(seed: int = 0) -> list:
    """(name, factors over the nine groups) of the fixed synthetic experts."""
    names = ["trial hats", "trial j1-3", "trial j4-8", "trial j9-15", "adjoint hats", "adjoint j1-3",
             "adjoint j4-8", "adjoint j9-15", "particular"]
    out = [("TL0", np.ones(9)), ("classical", np.zeros(9))]
    for i, nm in enumerate(names):
        a = np.ones(9)
        a[i] = 0.0
        out.append((f"fixes {nm}", a))
    for i, nm in enumerate(names):
        a = np.zeros(9)
        a[i] = 1.0
        out.append((f"keeps {nm}", a))
    rng = np.random.default_rng(seed)
    for k in range(20):
        out.append((f"random {k}", rng.uniform(0.0, 1.0, 9)))
    return out


class _Fixed:
    def __init__(self, resp):
        self.resp, self.calls, self.envelope = resp, 0, None

    def respond(self, problems, pb_of):
        return self.resp


def spearman(x, y) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def check5(networks: dict | None = None) -> dict:
    import torch
    torch.set_num_threads(4)
    from atlas.chartop import transport_coarse as tco
    from atlas.chartop import transport_torch as tt
    from atlas.chartop.host import monolithic
    from atlas.studio import cot
    from atlas.studio import cot_layouts as CL
    G = _groups()
    experts = synthetic_set()
    net_names = list(networks or {})
    coupled = {nm: [] for nm, _ in experts}
    coupled.update({nm: [] for nm in net_names})
    loss_parts = {nm: [] for nm in net_names}
    probs = []
    for name in CL.NAMES:
        for tier in CL.TIERS:
            t0 = time.perf_counter()
            spec, _ = CL.build(name, tier)
            L = cot.layout(CL.cut(spec, name, tier), spec)
            mono = monolithic(L)
            ex = tr.ClassicalTransportExpert().respond(L.problems, L.pb)
            t0r = [tco.TL0TransportExpert(n_c=min(16, p.n // 2)).response(p, L.pb(p.n).Q) for p in L.problems]
            floor = L.couple(ex).errors(mono.fields)["energy_rel"]
            e_tl0 = L.couple(t0r).errors(mono.fields)["energy_rel"]
            row = {"problem": f"{name} {tier}", "floor": floor, "tl0": e_tl0, "experts": {}}
            for nm, a in experts:
                resp = []
                for r, r0 in zip(ex, t0r):
                    H, Gm = r.H.copy(), r.G.copy()
                    for gi, cols in enumerate(G):
                        H[:, cols] += a[gi] * (r0.H[:, cols] - r.H[:, cols])
                        Gm[:, cols] += a[4 + gi] * (r0.G[:, cols] - r.G[:, cols])
                    up = r.up + a[8] * (r0.up - r.up)
                    resp.append(tr.TransportResponse(H, Gm, up))
                e = L.couple(resp).errors(mono.fields)["energy_rel"]
                excess = (e - floor) / (e_tl0 - floor)
                coupled[nm].append(excess)
                row["experts"][nm] = e
            for nm in net_names:
                he = tt.HostExpert(networks[nm], n_c=16, name=nm)
                resp = he.respond(L.problems, L.pb)
                e = L.couple(resp).errors(mono.fields)["energy_rel"]
                coupled[nm].append((e - floor) / (e_tl0 - floor))
                row["experts"][nm] = e
                # the training loss on this problem's pieces
                ls = []
                for p in L.problems:
                    b = tt.PieceBatch([p], n_c=min(16, p.n // 2))
                    with torch.no_grad():
                        ls.append(float(tt.objective(b, *networks[nm](b))))
                loss_parts[nm].append(float(np.mean(ls)))
            row["seconds"] = time.perf_counter() - t0
            probs.append(row)
            print(f"check 5: {row['problem']}: floor {floor:.4g}, TL0 {e_tl0:.4g} ({row['seconds']:.0f} s)", flush=True)
    loss = {nm: float(np.mean(a ** 2)) for nm, a in experts}          # each group's dual residual is a^2 TL0's
    loss.update({nm: float(np.mean(v)) for nm, v in loss_parts.items()})
    names = list(loss)
    err = {nm: float(np.mean(coupled[nm])) for nm in names}
    rho = spearman([loss[k] for k in names], [err[k] for k in names])
    return {"passed": rho >= CHECK5_RHO, "spearman": rho, "experts": len(names),
            "loss": loss, "coupled_excess": err, "problems": probs,
            "synthetic_loss_identity": "a synthetic expert's relative residual in group g is a_g^2 exactly, so its "
                                       "loss is mean(a^2)"}


def check5_realigned(networks: dict | None = None, tw: dict | None = None) -> dict:
    """Check 5 again with the realigned loss (the owner, 2026-11-19: "Realign loss, re-check
    (Recommended)"): the same experts, the same problems and the same bar; the loss is
    `transport_torch.objective` with the training layouts' trace weights
    (`transport_torch.trace_weights`), terms TL0 already solves left out.  A synthetic expert's
    residual in column k is its group's factor times TL0's, so its loss is computed exactly from
    TL0's column residuals."""
    import torch
    torch.set_num_threads(4)
    from atlas.chartop import transport_coarse as tco
    from atlas.chartop import transport_torch as tt
    from atlas.chartop.host import monolithic
    from atlas.studio import cot
    from atlas.studio import cot_layouts as CL
    tw = tw or tt.trace_weights()
    W, ap = tw["weights"], tw["particular"]
    G = _groups()
    gof = np.zeros(64, int)
    for gi, cols in enumerate(G):
        gof[cols] = gi
    experts = synthetic_set()
    net_names = list(networks or {})
    coupled = {nm: [] for nm, _ in experts}
    coupled.update({nm: [] for nm in net_names})
    losses = {nm: [] for nm in list(coupled)}
    probs = []
    for name in CL.NAMES:
        for tier in CL.TIERS:
            t0 = time.perf_counter()
            spec, _ = CL.build(name, tier)
            L = cot.layout(CL.cut(spec, name, tier), spec)
            mono = monolithic(L)
            ex = tr.ClassicalTransportExpert().respond(L.problems, L.pb)
            t0r = [tco.TL0TransportExpert(n_c=min(16, p.n // 2)).response(p, L.pb(p.n).Q) for p in L.problems]
            floor = L.couple(ex).errors(mono.fields)["energy_rel"]
            e_tl0 = L.couple(t0r).errors(mono.fields)["energy_rel"]
            batches = [tt.PieceBatch([p], n_c=min(16, p.n // 2)) for p in L.problems]
            pieces = []
            for b in batches:
                c0 = [x[0].numpy() for x in b.cols0]
                e = [bool(x[0]) for x in tt.easy_terms(b)]
                w = np.asarray(W["flowing" if bool(b.flowing[0]) else "solid"])
                pieces.append((c0, e, w))
            row = {"problem": f"{name} {tier}", "floor": floor, "tl0": e_tl0, "experts": {},
                   "pieces_left_out": sum(1 for _c, e, _w in pieces if all(e))}
            for nm, a in experts:
                resp = []
                for r, r0 in zip(ex, t0r):
                    H, Gm = r.H.copy(), r.G.copy()
                    for gi, cols in enumerate(G):
                        H[:, cols] += a[gi] * (r0.H[:, cols] - r.H[:, cols])
                        Gm[:, cols] += a[4 + gi] * (r0.G[:, cols] - r.G[:, cols])
                    resp.append(tr.TransportResponse(H, Gm, r.up + a[8] * (r0.up - r.up)))
                e_ = L.couple(resp).errors(mono.fields)["energy_rel"]
                coupled[nm].append((e_ - floor) / (e_tl0 - floor))
                row["experts"][nm] = e_
                vals = []
                for (t0c, a0c, p0c), (et, ea, ep), w in pieces:
                    terms = [(float((w * a[gof] ** 2 * t0c).sum() / (w * t0c).sum()) if not et else None, 1 - 0.1 - ap),
                             (float((w * a[4 + gof] ** 2 * a0c).sum() / (w * a0c).sum()) if not ea else None, 0.1),
                             (float(a[8] ** 2) if not ep else None, ap)]
                    den = sum(s for v, s in terms if v is not None)
                    if den > 0:
                        vals.append(sum(s * v for v, s in terms if v is not None) / den)
                losses[nm].append(float(np.mean(vals)))
            for nm in net_names:
                resp = tt.HostExpert(networks[nm], n_c=16, name=nm).respond(L.problems, L.pb)
                e_ = L.couple(resp).errors(mono.fields)["energy_rel"]
                coupled[nm].append((e_ - floor) / (e_tl0 - floor))
                row["experts"][nm] = e_
                with torch.no_grad():
                    vals = [float(tt.objective(b, *networks[nm](b), weights=W, particular=ap)) for b in batches]
                losses[nm].append(float(np.mean(vals)))
            row["seconds"] = time.perf_counter() - t0
            probs.append(row)
            print(f"check 5 (realigned): {row['problem']}: floor {floor:.4g}, TL0 {e_tl0:.4g}, "
                  f"{row['pieces_left_out']} pieces left out ({row['seconds']:.0f} s)", flush=True)
    names = list(losses)
    loss = {nm: float(np.mean(losses[nm])) for nm in names}
    err = {nm: float(np.mean(coupled[nm])) for nm in names}
    rho = spearman([loss[k] for k in names], [err[k] for k in names])
    syn = [nm for nm, _ in experts]
    rho_syn = spearman([loss[k] for k in syn], [err[k] for k in syn])
    return {"passed": rho >= CHECK5_RHO, "spearman": rho, "spearman_synthetic_only": rho_syn,
            "experts": len(names), "loss": loss, "coupled_excess": err, "problems": probs,
            "loss_definition": "transport_torch.objective with the trace weights of phase3-trace-weights.json, "
                               f"adjoint share 0.1, particular share {ap:.4f}, terms TL0 solves left out"}


# ---------------------------------------------------------------------------


def _load_probe_networks() -> dict:
    """The capacity probe's trained arms at 1M (their final state, step 3,000), where they exist.
    Larger probes are left out: one may still be running when this check runs."""
    import io
    import torch
    from atlas.chartop import transport_torch as tt
    out = {}
    for path in sorted(glob.glob(os.path.join(STATE, "*-1M.pt"))):
        name = os.path.basename(path)[:-3]
        form, size = name.split("-", 1)
        st = torch.load(io.BytesIO(registry.read_bytes(path)), weights_only=False)
        e = tt.ArmR(size, form, kernel=False)          # as the probe trained it
        e.load_state_dict(st["model"])
        e.eval()
        out[f"probe {name} (step {st['step']})"] = e
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", nargs="+", type=int, required=True, choices=(1, 2, 3, 5))
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--no-networks", action="store_true")
    ap.add_argument("--realigned", action="store_true", help="check 5 with the realigned loss (after its first run failed)")
    a = ap.parse_args(argv)
    import winqos
    winqos.opt_out()
    path = os.path.join(RECORDS, "phase3-checks.json")
    rec = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {
        "what": "the coolant dataset's checks before it is frozen", "criteria": CRITERIA}
    for c in a.check:
        t0 = time.perf_counter()
        if c == 1:
            man = json.load(open(os.path.join(a.data, "manifest-train.json"), encoding="utf-8"))
            res = check1(man)
        elif c == 2:
            res = check2(a.data)
        elif c == 3:
            res = check3()
        elif a.realigned:
            res = check5_realigned(None if a.no_networks else _load_probe_networks())
        else:
            res = check5(None if a.no_networks else _load_probe_networks())
        res.update(seconds=time.perf_counter() - t0, code=_commit(),
                   written=dt.datetime.now().astimezone().isoformat(timespec="seconds"))
        rec[f"check {c}" + (" (realigned loss)" if c == 5 and a.realigned else "")] = res
        registry.write_atomic(path, json.dumps(rec, indent=1).encode("utf-8"))
        print(f"check {c}: {'PASSED' if res['passed'] else 'FAILED'}", flush=True)
    lines = ["The coolant dataset's checks", ""]
    for c in (1, 2, 3, 5):
        r = rec.get(f"check {c}")
        if r:
            extra = {1: lambda r: f"fails {r['fails']}", 2: lambda r: f"within {r['share_within']:.2f}, largest Lam "
                     f"{r['largest_lam_change']:.3g}, chi {r['largest_chi_change']:.3g}",
                     3: lambda r: ", ".join(f"{x['problem']} {x['energy_rel']:.3g}" for x in r["rows"]),
                     5: lambda r: f"Spearman {r['spearman']:.3f} over {r['experts']} experts"}[c](r)
            lines.append(f"check {c}: {'passed' if r['passed'] else 'FAILED'}; {extra}")
        r = rec.get("check 5 (realigned loss)") if c == 5 else None
        if r:
            lines.append(f"check 5 (realigned loss): {'passed' if r['passed'] else 'FAILED'}; Spearman "
                         f"{r['spearman']:.3f} over {r['experts']} experts ({r['spearman_synthetic_only']:.3f} synthetic only)")
    registry.write_atomic(os.path.join(RECORDS, "phase3-checks.txt"), ("\n".join(lines) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
