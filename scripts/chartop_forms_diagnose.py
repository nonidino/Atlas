"""Post-hoc diagnostics of the W358 registration's one evaluation (the forms page, section 11).

Nothing here is a gate.  The test split is never drawn or read: the six benchmarks and
validation seeds from 2,000,100 onward are read post hoc, as Stage 0's section 9.7 did.
The weights are checked against ``weights-sha256.txt`` before they are read.

Parts (``--part``), each printing a verdict table of ten lines or fewer and writing
``out/chartop/forms-diagnose-<part>.json``:

- ``a1``: Theorem 2's network term on the steady benchmarks, split by mode group for the
  zero-head form U and the main run S.  Per piece ``w_i = e_p,i + E_i c*_i`` with ``c*``
  the classical coupled coefficients; the groups are the hats' corner-difference part
  ``E_hats (c*_hats - cbar)``, the hats' level part ``cbar E_hats 1``, the side sines
  ``j = 1-3``, ``4-8`` and ``9-15`` (the strips, on and off their strip nodes), and
  ``e_p``.  Their vector sum is ``w_i``; each group's share is ``<w_g, w>_A / ||w||_A^2``,
  which sums to one exactly, beside its own norm.  The strips are also re-solved at
  widths 1/4 and 1/2, the plain square's ``H0`` beyond as before.
- ``a2``: the U-Net's receptive field, exactly (the support of one output node's
  gradient, random weights, a grid wide enough not to fold) and effectively (S's
  trained mode network on validation pieces: the radius holding 90% and 99% of the
  gradient's mass), as a fraction of the piece at n = 32, 64, 128.
- ``a3``: S re-coupled with its hats replaced by the exact classical hats or by U's
  projected hats (and, beside them, with every learned column exact), steady
  benchmarks, both frames, judged by E4's own function.
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
import scipy.sparse.linalg as spla  # noqa: E402
import torch  # noqa: E402

import chartop_evaluate as ev0  # noqa: E402
import chartop_registry as registry  # noqa: E402
from atlas.chartop import benchmarks as bm, data, fe, forms as fm, gates_forms as GF  # noqa: E402
from atlas.chartop import host as hs, model as md, ports as pt, train as tr  # noqa: E402
from atlas.chartop.expert import LearnedExpert  # noqa: E402

OUT = os.path.join(ROOT, "out", "chartop")
RUNS = os.path.join(OUT, "runs", "forms")
VAL_POSTHOC = GF.SEED_BASE["val"] + 100
J_GROUPS = {"j1-3": (1, 3), "j4-8": (4, 8), "j9-15": (9, 15)}
RECORDED = {"bend-3": {"S": 3.9379, "Sp-s0": 1.2848}, "s-channel": {"S": 6.559, "Sp-s0": 5.4904},
            "insert-round": {"S": 7.8385, "Sp-s0": 4.3693}}


def checked_weights(run: str) -> str:
    """The run's ``best.pt``, refused unless its sha256 is the one recorded at download."""
    manifest = {}
    with open(os.path.join(RUNS, "weights-sha256.txt"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                h, rel = line.split()
                manifest[rel.lstrip("*").split("/")[0]] = h
    path = os.path.join(RUNS, run, "best.pt")
    h = registry.sha256_file(path)
    if manifest.get(run) != h:
        raise SystemExit(f"{path}: sha256 {h[:16]}... is not the recorded one")
    return path


def zero_heads() -> fm.KernelExpert:
    """U: S's form (split, corner bubble, strips) as initialised -- zero heads."""
    torch.manual_seed(0)
    return fm.KernelExpert("split", "unet", "1M", GF.KIND, GF.M_PORTS, None, "corner", True).eval()


def experts(names=("U", "S")) -> dict:
    out = {}
    for r in names:
        net = zero_heads() if r == "U" else tr.load_expert(checked_weights(r), "cpu")
        out[r] = LearnedExpert(net, envelope=data.ENVELOPE)
    return out


def column_groups(kind: str = GF.KIND, m: int = GF.M_PORTS) -> dict:
    cols = pt.basis(kind, m, m).columns
    g = {"hats": [k for k, c in enumerate(cols) if c.what == "vertex"]}
    for name, (a, b) in J_GROUPS.items():
        g[name] = [k for k, c in enumerate(cols) if c.what == "side" and a <= c.j <= b]
    if sorted(sum(g.values(), [])) != list(range(4 * m)):
        raise ValueError("the groups do not partition the columns")
    return g


def side_strip_columns(kind: str = GF.KIND, m: int = GF.M_PORTS) -> list:
    return [list(cs) for cs in fm.strip_columns(kind, m)]


def interior(p):
    _loop, inner = fe.grid(p.n)
    M = p.matrix().tocsr()
    return M[inner][:, inner]


def bench(name: str, frame: str = "registered") -> bm.Bench:
    kw = {"envelope": data.ENVELOPE} if name == "s-channel" else {}
    b = bm.build(name, GF.BENCHMARK_N, "steady", **kw)
    return b if frame == "registered" else bm.datum_shift(b, GF.DATUM_SHIFT_K)


# ---------------------------------------------------------------------------
# A1: the network term by mode group
# ---------------------------------------------------------------------------


def split_term(L: hs.Layout, cp: hs.Coupled, cl: hs.Coupled) -> dict:
    """Theorem 2's bracket split by group: the shares ``<w_g, w>_A / ||w||^2_A`` (summing to
    one) and each group's own ``||w_g||_A`` over ``||w||_A``, kappa-weighted over pieces."""
    G = column_groups()
    strips = side_strip_columns()
    T = 0.0
    share, alone = {}, {}
    for i, p in enumerate(L.problems):
        A = interior(p)
        rL, rC = cp.responses[i], cl.responses[i]
        ci = L.R[i] @ cl.c
        E = rL.H - rC.H
        h = G["hats"]
        cbar = float(ci[h].mean())
        comp = {"hats-diff": E[:, h] @ (ci[h] - cbar), "hats-level": cbar * E[:, h].sum(1)}
        for g in ("j1-3", "j4-8"):
            comp[g] = E[:, G[g]] @ ci[G[g]]
        on = np.zeros(E.shape[0])
        off = np.zeros(E.shape[0])
        for s, S in enumerate(fm.strip_nodes(p.n)):
            v = E[:, strips[s]] @ ci[strips[s]]
            on += np.where(S, v, 0.0)
            off += np.where(S, 0.0, v)
        comp["j9-15 on strip"], comp["j9-15 off strip"] = on, off
        comp["p"] = rL.up - rC.up
        w = sum(comp.values())
        Aw = A @ w
        T += p.kappa_ref * float(w @ Aw)
        for g, v in comp.items():
            share[g] = share.get(g, 0.0) + p.kappa_ref * float(v @ Aw)
            alone[g] = alone.get(g, 0.0) + p.kappa_ref * float(v @ (A @ v))
    nt = ev0.network_term(L, cp, cl)
    return {"network_term": math.sqrt(T), "network_term_ev0": nt,
            "check_rel": abs(math.sqrt(T) - nt) / max(nt, 1e-300),
            "share": {g: v / T for g, v in share.items()},
            "alone_over_total": {g: math.sqrt(max(v, 0.0) / T) for g, v in alone.items()}}


def strips_at_width(p, H0, width: float) -> np.ndarray:
    """The strip columns of `forms.strip_modes` with strips ``width n`` cells deep (the
    plain square's ``H0`` beyond), every other column ``H0``."""
    n = p.n
    JJ, II = np.meshgrid(np.arange(1, n), np.arange(1, n), indexing="ij")
    dist = [JJ.ravel(), (n - II).ravel(), (n - JJ).ravel(), II.ravel()]
    A_I, M_IG = fm.interior_blocks(p.matrix(), n)
    A = A_I.tocsr()
    BQ = -np.asarray(M_IG @ pt.basis(GF.KIND, n, GF.M_PORTS).Q)
    Hs = np.array(H0, dtype=np.float64, order="C", copy=True)
    for s, cs in enumerate(side_strip_columns()):
        S = dist[s] <= width * n
        rhs = BQ[S][:, cs] - A[S][:, ~S] @ Hs[~S][:, cs]
        Hs[np.flatnonzero(S)[:, None], cs] = spla.splu(A[S][:, S].tocsc()).solve(rhs)
    return Hs


def plain_H0(n: int) -> np.ndarray:
    """The plain square's steady modes [(n-1)^2, 4m] as the expert computes them (float32
    principal part, then float64)."""
    ps = md.plain_square(n, GF.KIND, GF.M_PORTS, "cpu")
    H0 = ps.principal(torch.zeros(1, dtype=torch.float64), torch.float32).to(torch.float64)[0]
    return H0.numpy().reshape(4 * GF.M_PORTS, -1).T


def with_columns(resp: list, src: list, cols: list) -> list:
    out = []
    for r, s in zip(resp, src):
        H = r.H.copy()
        H[:, cols] = s.H[:, cols]
        out.append(hs.Response(H, r.up, r.level))
    return out


def part_a1(args) -> dict:
    ex = experts(("U", "S"))
    rec = {"source": {}, "runs": {}, "widths": {}}
    rows = []
    for name in GF.HELD_OUT_BENCHMARKS:
        b = bench(name)
        gmax = max(float(np.abs(p.g).max()) if p.g is not None else 0.0 for p in b.problems)
        rec["source"][name] = {"max_abs_g": gmax, "pieces": len(b.problems),
                               "s0_max": max(float(p.s0) for p in b.problems)}
        L = b.layout(GF.KIND, GF.M_PORTS)
        mono = hs.monolithic(L)
        cl = L.solve(hs.ClassicalExpert())
        u, d = cl.energy_error(mono.fields)
        rec["runs"].setdefault(name, {"floor": u / d})
        for run, e in ex.items():
            cp = L.solve(e)
            num, den = cp.energy_error(mono.fields)
            st = split_term(L, cp, cl)
            st["energy_rel"] = num / den
            rec["runs"][name][run] = st
            rows.append((name, run, num / den, st))
            if run == "U":
                # the strips at other widths, U's columns otherwise; 1/8 must reproduce U
                wrec = {}
                for width in (1 / 8, 1 / 4, 1 / 2):
                    resp = []
                    for i, p in enumerate(L.problems):
                        Hs = strips_at_width(p, plain_H0(p.n), width)
                        H = cp.responses[i].H.copy()
                        cs = sum(side_strip_columns(), [])
                        H[:, cs] = Hs[:, cs]
                        resp.append(hs.Response(H, cp.responses[i].up, cp.responses[i].level))
                    if width == 1 / 8:
                        same = all(np.array_equal(r.H, q.H) for r, q in zip(resp, cp.responses))
                        wrec["eighth_reproduces_U_bit_for_bit"] = bool(same)
                    cw = L.couple(resp)
                    nw, dw = cw.energy_error(mono.fields)
                    sw = split_term(L, cw, cl)
                    wrec[f"{width:.3f}"] = {"energy_rel": nw / dw,
                                            "strip_share": sw["share"]["j9-15 on strip"]
                                            + sw["share"]["j9-15 off strip"],
                                            "network_term": sw["network_term"]}
                rec["widths"][name] = wrec
    # the verdict table
    print("A1  problem       run  E(%)    NT-chk   hat-diff hat-lvl  j1-3   j4-8  j9-15on j9-15off  p")
    for name, run, er, st in rows:
        sh = st["share"]
        print(f"    {name:13s} {run:3s} {100 * er:6.3f} {st['check_rel']:8.1e} "
              f"{sh['hats-diff']:7.3f} {sh['hats-level']:7.1e} {sh['j1-3']:6.3f} {sh['j4-8']:6.3f} "
              f"{sh['j9-15 on strip']:6.3f} {sh['j9-15 off strip']:7.3f} {sh['p']:6.3f}")
    for name, wr in rec["widths"].items():
        print(f"    {name:13s} U strips at width 1/8, 1/4, 1/2: E(%) "
              + ", ".join(f"{100 * wr[k]['energy_rel']:.3f}" for k in ("0.125", "0.250", "0.500"))
              + f"  (1/8 = U bit for bit: {wr['eighth_reproduces_U_bit_for_bit']})")
    srcs = ", ".join(f"{k} {v['max_abs_g']:.2g}" for k, v in rec["source"].items())
    print(f"    steady sources, max |g|: {srcs}")
    return rec


# ---------------------------------------------------------------------------
# A2: the receptive field
# ---------------------------------------------------------------------------


def _randomise(net: torch.nn.Module, std: float, seed: int = 0) -> torch.nn.Module:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for prm in net.parameters():
            prm.copy_(std * torch.randn(prm.shape, generator=g, dtype=prm.dtype))
    return net.eval()


def exact_support(n: int = 256) -> dict:
    """The support of one output node's gradient over the input, random weights (FiLM and
    head included), for output nodes at every residue mod 8 near the centre of a grid
    too wide to fold: the receptive field's radius in nodes (Chebyshev), min and max."""
    net = _randomise(md.UNet(md.WIDTHS["1M"], 1, 4), 0.2)
    radii = []
    c0 = n // 2
    for d in range(8):
        x = torch.randn(1, 4, n + 1, n + 1, dtype=torch.float32, requires_grad=True)
        s = torch.randn(1, md.N_SCALARS)
        y = net(x, s)
        jj, ii = c0 + d, c0 + (3 * d) % 8
        y[0, 0, jj, ii].backward()
        nz = (x.grad[0].abs().sum(0) > 0).nonzero()
        radii.append(int(max((nz[:, 0] - jj).abs().max(), (nz[:, 1] - ii).abs().max())))
    return {"grid_n": n, "radius_min": min(radii), "radius_max": max(radii), "radii": radii}


def effective_radius(expert: fm.KernelExpert, n: int, seeds: list) -> dict:
    """S's mode network: for the centre node and the node next to the middle of side 0, the
    radius (Chebyshev, nodes) holding 90% and 99% of ``sum_c sum_ch |d y_c / d x_ch|`` over
    the learned mode channels, on validation pieces at resolution ``n``."""
    net = expert.net
    out = {"centre": {"r90": [], "r99": []}, "side": {"r90": [], "r99": []}}
    for sd in seeds:
        p = data.draw(sd, n=n).problem
        ch, sc, _rms = p.inputs()
        x0 = torch.tensor(ch[None, :4], dtype=torch.float32)
        s = torch.tensor(sc[None], dtype=torch.float32)
        for where, (jj, ii) in {"centre": (n // 2, n // 2), "side": (1, n // 2)}.items():
            x = x0.clone().requires_grad_(True)
            y = net(x, s)
            G = torch.zeros(n + 1, n + 1, dtype=torch.float64)
            for c in range(y.shape[1] - 1):          # the learned mode channels, not the level
                gx, = torch.autograd.grad(y[0, c, jj, ii], x, retain_graph=True)
                G += gx[0].abs().sum(0).double()
            J, I = torch.meshgrid(torch.arange(n + 1), torch.arange(n + 1), indexing="ij")
            dist = torch.maximum((J - jj).abs(), (I - ii).abs())
            tot = float(G.sum())
            for q, key in ((0.90, "r90"), (0.99, "r99")):
                r = next(r for r in range(n + 1) if float(G[dist <= r].sum()) >= q * tot)
                out[where][key].append(r)
    return out


def part_a2(args) -> dict:
    t0 = time.perf_counter()
    rec = {"exact": exact_support(256)}
    S = tr.load_expert(checked_weights("S"), "cpu")
    seeds = [VAL_POSTHOC + k for k in range(args.pieces)]
    rec["effective"] = {}
    for n in (32, 64, 128):
        rec["effective"][str(n)] = effective_radius(S, n, seeds)
    rec["elapsed_s"] = time.perf_counter() - t0
    R = rec["exact"]["radius_max"]
    print(f"A2  exact receptive field: radius {rec['exact']['radius_min']}-{R} nodes "
          f"(output node's residue mod 8), grid n = 256, random weights")
    print("    n    R/n   side node reaches opposite side   effective r90/n, r99/n (centre | side node)")
    for n in (32, 64, 128):
        e = rec["effective"][str(n)]
        med = {w: {k: float(np.median(v)) / n for k, v in e[w].items()} for w in e}
        print(f"    {n:3d}  {R / n:4.2f}  {str(R >= n - 1):5s}                             "
              f"{med['centre']['r90']:.2f}, {med['centre']['r99']:.2f} | "
              f"{med['side']['r90']:.2f}, {med['side']['r99']:.2f}")
    return rec


# ---------------------------------------------------------------------------
# A3: S with its hats replaced
# ---------------------------------------------------------------------------


def part_a3(args) -> dict:
    ex = experts(("U", "S"))
    G = column_groups()
    learned = G["hats"] + G["j1-3"] + G["j4-8"]
    variants = ("S", "S + exact hats", "S + U's hats", "S + exact hats, j1-3", "S + exact learned", "U",
                "U + exact hats")
    rec = {}
    for name in GF.HELD_OUT_BENCHMARKS:
        fr = {}
        for frame in ("registered", "lowered"):
            b = bench(name, frame)
            L = b.layout(GF.KIND, GF.M_PORTS)
            mono = hs.monolithic(L)
            cl = L.solve(hs.ClassicalExpert())
            u, d = cl.energy_error(mono.fields)
            mx0, span = ev0.field_errors(mono.fields, cl.fields)
            rS = ex["S"].respond(L.problems, L.pb)
            rU = ex["U"].respond(L.problems, L.pb)
            rC = cl.responses
            resp = {"S": rS, "S + exact hats": with_columns(rS, rC, G["hats"]),
                    "S + U's hats": with_columns(rS, rU, G["hats"]),
                    "S + exact hats, j1-3": with_columns(rS, rC, G["hats"] + G["j1-3"]),
                    "S + exact learned": with_columns(rS, rC, learned),
                    "U": rU, "U + exact hats": with_columns(rU, rC, G["hats"])}
            out = {"floor": u / d}
            for v in variants:
                cp = L.couple(resp[v])
                num, den = cp.energy_error(mono.fields)
                mx, _ = ev0.field_errors(mono.fields, cp.fields)
                out[v] = {"energy_rel": num / den, "max_err": mx, "span": span, "floor_max": mx0}
            fr[frame] = out
        rec[name] = {"floor": fr["registered"]["floor"], "variants": {}}
        for v in variants:
            a, z = fr["registered"][v], fr["lowered"][v]
            j = GF.judge_e4_both_frames(a, z)
            rec[name]["variants"][v] = {"energy_rel": a["energy_rel"], "energy_rel_lowered": z["energy_rel"],
                                        "max_over_range": a["max_err"] / a["span"], "e4_met": j["met"]}
    print("A3  E(%) steady, registered frame [E4 both frames met?]; recovered = (S - x)/(S - floor)")
    print("    variant              " + "".join(f"{nm:>24s}" for nm in GF.HELD_OUT_BENCHMARKS))
    for v in variants:
        cells = []
        for nm in GF.HELD_OUT_BENCHMARKS:
            r = rec[nm]["variants"][v]
            s_, f_ = rec[nm]["variants"]["S"]["energy_rel"], rec[nm]["floor"]
            frac = (s_ - r["energy_rel"]) / (s_ - f_)
            cells.append(f"{100 * r['energy_rel']:8.4f} [{'Y' if r['e4_met'] else 'n'}] {100 * frac:6.1f}%")
        print(f"    {v:20s} " + "".join(f"{c:>24s}" for c in cells))
    print("    floor                " + "".join(f"{100 * rec[nm]['floor']:24.2e}" for nm in GF.HELD_OUT_BENCHMARKS))
    return rec


# ---------------------------------------------------------------------------
# B1's price: the classical hats against the exact solve
# ---------------------------------------------------------------------------


def _best(f, reps: int) -> float:
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def part_b1(args) -> dict:
    """Per validation piece, the best of ``--reps`` wall-clock times of: the exact classical
    response (`fe.Exact`: one sparse LU, 64 mode columns and the particular field), the
    classical hats (`forms.KernelExpert._classical_hats`' operations: one LU, 4 columns),
    the hats with j = 1-3 (one LU, 16 columns), the strips, and S's forward pass (batch
    one, ``--threads`` torch threads), each as a fraction of the exact solve."""
    import scipy.sparse.linalg as _spla
    G = column_groups()
    S = tr.load_expert(checked_weights("S"), "cpu")
    rec = {"threads": args.threads, "reps": args.reps, "by_n": {}}
    for n in (32, 64, 128):
        Q = pt.basis(GF.KIND, n, GF.M_PORTS).Q
        H0 = plain_H0(n)
        rows = []
        for k in range(args.pieces):
            p = data.draw(VAL_POSTHOC + k, n=n).problem
            M_ = p.matrix()
            A_I, M_IG = fm.interior_blocks(M_, n)
            loop_inner = fe.grid(n)[1]
            b_I = p.load()[loop_inner]

            def exact():
                ex = fe.Exact(M_, n)
                ex.modes(Q)
                ex.particular(b_I)

            def hats(cols):
                return lambda: -_spla.splu(A_I.tocsc()).solve(np.asarray(M_IG @ Q[:, cols]))

            def strips():
                fm.strip_modes(A_I, M_IG, Q, H0, n)

            ch, sc, _rms = p.inputs()
            x = torch.tensor(ch[None])
            s = torch.tensor(sc[None])
            s0 = torch.tensor([p.s0], dtype=torch.float64)
            w = torch.tensor(p.tensors()["w"][None], dtype=torch.float64)
            g = torch.tensor(p.g[None] / (_rms or 1.0), dtype=torch.float64)

            def forward():
                with torch.no_grad():
                    S(x, s, s0, w=w, g=g, strip_solve=False)

            t = {"exact": _best(exact, args.reps), "hats": _best(hats(G["hats"]), args.reps),
                 "hats_j1_3": _best(hats(G["hats"] + G["j1-3"]), args.reps),
                 "strips": _best(strips, args.reps), "forward": _best(forward, args.reps)}
            rows.append(t)
        med = {key: float(np.median([r[key] / r["exact"] for r in rows])) for key in rows[0]}
        rec["by_n"][str(n)] = {"fraction_of_exact_median": med,
                               "exact_s_median": float(np.median([r["exact"] for r in rows])), "rows": rows}
    print(f"B1  price per piece, fraction of the exact classical solve (median of {args.pieces} "
          f"validation pieces, best of {args.reps}; torch threads {args.threads})")
    print("    n    hats (4 cols)  hats+j1-3 (16)  strips (28)  S forward (no strips)")
    for n, r in rec["by_n"].items():
        f = r["fraction_of_exact_median"]
        print(f"    {int(n):3d}  {f['hats']:13.2f}  {f['hats_j1_3']:14.2f}  {f['strips']:11.2f}  {f['forward']:8.2f}")
    return rec


# ---------------------------------------------------------------------------
# issues 1 and 3: where the target's quotient peaks, and whether it grows there
# ---------------------------------------------------------------------------

JUMP = 0.25          # |delta log kappa| between edge-adjacent elements: a material interface
NEAR = 2             # cells: a node this close to an interface element is "at" the interface


def _jump_nodes(lk_e: np.ndarray) -> np.ndarray:
    """Nodes [n+1, n+1] within ``NEAR`` cells of an element edge across which log kappa
    jumps by ``JUMP`` or more."""
    n = lk_e.shape[0]
    J = np.zeros((n, n), bool)
    dx = np.abs(np.diff(lk_e, axis=1)) >= JUMP
    dy = np.abs(np.diff(lk_e, axis=0)) >= JUMP
    J[:, :-1] |= dx
    J[:, 1:] |= dx
    J[:-1, :] |= dy
    J[1:, :] |= dy
    node = np.zeros((n + 1, n + 1), bool)
    for a, b in np.argwhere(J):
        node[max(0, a - NEAR + 1):a + NEAR + 1, max(0, b - NEAR + 1):b + NEAR + 1] = True
    return node


def part_i13(args) -> dict:
    """On validation pieces at n = 32-256 (the same seeds, so the same pieces refined): the
    quotient the head must output, ``N* = (H - H0) / beta_c`` over S's learned columns
    (``H0`` the plain square's), its peak, and where the peak sits: within ``n/8`` of a
    corner (with that corner's angle), within ``n/8`` of a side and ``NEAR`` cells of a
    material interface (issue 1), at an interface inside, or elsewhere.  Beside it, the
    peak over the nodes that are none of these (the regular part)."""
    from atlas.chartop import chart as ch
    learned = list(fm.learned_columns(GF.KIND, GF.M_PORTS))
    rec = {"by_n": {}, "jump": JUMP, "near_cells": NEAR, "pieces": args.pieces}
    for n in (32, 64, 128, 256):
        Q = pt.basis(GF.KIND, n, GF.M_PORTS).Q
        H0 = plain_H0(n)
        beta = fm.corner_bubble(n)[1:-1, 1:-1].ravel()
        JJ, II = np.meshgrid(np.arange(1, n), np.arange(1, n), indexing="ij")
        jj, ii = JJ.ravel(), II.ravel()
        dcorner = np.minimum.reduce([np.maximum(abs(jj - a), abs(ii - b)) for a in (0, n) for b in (0, n)])
        dside = np.minimum.reduce([jj, ii, n - jj, n - ii])
        rows = []
        for k in range(args.pieces):
            ex = data.draw(VAL_POSTHOC + k, n=n)
            p = ex.problem
            H = fe.Exact(p.matrix(), n).modes(Q)
            Ns = np.abs((H - H0)[:, learned] / beta[:, None]).max(axis=1)
            at_jump = _jump_nodes(np.asarray(p.log_kappa_hat_e))[1:-1, 1:-1].ravel()
            corner = dcorner <= n // 8
            iface_side = (~corner) & (dside <= n // 8) & at_jump
            iface = (~corner) & (~iface_side) & at_jump
            regular = ~(corner | at_jump)
            q = int(np.argmax(Ns))
            where = ("corner" if corner[q] else "interface at a side" if iface_side[q]
                     else "interface inside" if iface[q] else "elsewhere")
            ang = ch.corner_angles(p.chart.X)
            rows.append({"seed": ex.seed, "peak": float(Ns[q]), "where": where,
                         "peak_regular": float(Ns[regular].max()) if regular.any() else None,
                         "peak_corner": float(Ns[corner].max()),
                         "peak_interface_side": float(Ns[iface_side].max()) if iface_side.any() else None,
                         "has_interface": bool(at_jump.any()),
                         "corner_angles": ang.tolist(), "max_corner_angle": float(ang.max())})
        rec["by_n"][str(n)] = rows
    print("I13 N* = (H - H0)/beta_c peak, median over pieces [where the peaks sit: c corner, "
          "s interface at a side, i inside, e elsewhere]")
    print("    n    peak   regular  corner  iface@side(pieces with one)")
    for n, rows in rec["by_n"].items():
        med = lambda key: float(np.median([r[key] for r in rows if r[key] is not None])) if any(
            r[key] is not None for r in rows) else float("nan")
        tags = "".join({"corner": "c", "interface at a side": "s", "interface inside": "i",
                        "elsewhere": "e"}[r["where"]] for r in rows)
        nif = sum(r["peak_interface_side"] is not None for r in rows)
        print(f"    {int(n):3d}  {med('peak'):6.2f}  {med('peak_regular'):6.2f}  {med('peak_corner'):6.2f}  "
              f"{med('peak_interface_side'):6.2f} ({nif})   [{tags}]")
    angs = [round(r["max_corner_angle"]) for r in rec["by_n"]["64"]]
    print(f"    largest corner angle per piece (deg): {angs}")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--part", required=True, choices=("a1", "a2", "a3", "b1", "i13"))
    ap.add_argument("--reps", type=int, default=3, help="b1: timings per piece (the best is kept)")
    ap.add_argument("--pieces", type=int, default=4, help="a2: validation pieces per resolution")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    t0 = time.perf_counter()
    rec = {"part": args.part, "written": time.strftime("%Y-%m-%d %H:%M:%S"), "posthoc": True,
           "test_split_read": False, "torch_threads": args.threads}
    rec.update({"a1": part_a1, "a2": part_a2, "a3": part_a3, "b1": part_b1, "i13": part_i13}[args.part](args))
    rec["elapsed_s"] = time.perf_counter() - t0
    path = os.path.join(OUT, f"forms-diagnose-{args.part}.json")
    registry.write_atomic(path, json.dumps(ev0.strip(rec), indent=1).encode("utf-8"))
    print(f"    wrote {os.path.relpath(path, ROOT)} ({rec['elapsed_s']:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
