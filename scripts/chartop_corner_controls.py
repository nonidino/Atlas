"""W358: the bubble form's corner growth, and the candidate fixes, on validation pieces.

    python scripts/chartop_corner_controls.py [--ns 32,64,128,256] [--count 8] [--progress PATH]

Classical solves only, on the validation pieces of `scripts/chartop_diagnose.py`'s second
part (``SEED_BASE['val'] + 100 + i``, i < 8), never the test split.  No network, no
training, nothing registered.  Prints only a short table; writes
``out/chartop/corner-controls.json``.

**What is measured, per candidate.**  A candidate is a classical principal part ``P``
for the modes (and ``U`` for the level field), a bubble ``beta`` and, optionally,
damped-Jacobi sweeps on the piece's true operator after the network's output.
- **The target**: the quotient the head must output for the form to return the exact
  modes, ``N* = (H - P) / beta``, over the columns the network learns.  Its peak and
  root mean square, median over the pieces, at each resolution.  Sweeps do not change
  the target (any output whose error they remove would do as well).
- **e_H if the network learned only what is smooth**: each output is the energy-best
  function in ``beta x span{cos(p pi xi) cos(q pi eta), p, q < K}`` (the minimiser
  the label-free objective would reach in that class, T26), passed through the sweeps,
  and then the kernel-exact form's projection (W356), so the uniform state stays a
  fixed point.  ``e_H`` of the result, and of the classical part alone (``N = 0``).
- **Cost per piece**: the classical work the candidate adds to the network's call
  (the piece's matrix assembly excluded throughout), against the exact classical
  solve of the same piece (one sparse LU, 64 mode solves and the level field).

**The candidates** (W358's list, [[chart-operator-stage-0]] section 12 items 3-4):
- C1, the frozen-coefficient principal part: the piece's own tensor and reaction at
  each corner element, constant, solved exactly on the whole square, the four blended
  by a smooth partition of unity that is one near its corner (quintic steps on
  [1/4, 3/4]).  Same traces, so still exact.
- C2, damped Jacobi sweeps (weight 2/3) on the true operator, after the output and
  before the projection.  They preserve the fixed point when applied to the level
  field too, and the projection comes last anyway.
- C3, the high side sines (j = 9-15) by a strip solve along their side: the true
  operator on the interior nodes within 1/8 of the side, with the plain square's modes
  as data beyond the strip.  The network learns the other 36 columns only.
- C4, a bubble that follows the corners: the angle bubble (``r^min(pi/omega, 2)`` at
  each corner, ``omega`` from the corner element's tensor, blended over r ~ 1/4), and
  the corner bubble of `forms.corner_bubble`, which vanishes like the distance to the
  boundary at every corner whatever the angle.

The positive control: Stage 0's bubble must reproduce diagnose's medians (16.4, 50.2,
89.2 at n = 32, 64, 128) from its own record.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402

from atlas.chartop import data, fe, forms, ports, principal  # noqa: E402

M_PORTS = 16
HIGH = [4 + 15 * s + (j - 1) for s in range(4) for j in range(9, 16)]
LOW = [k for k in range(4 * M_PORTS) if k not in HIGH]
CORNERS = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
#: (element row, element column, sign of a12 in the corner's own frame) per corner
CORNER_ELEMENTS = lambda n: [(0, 0, 1.0), (0, n - 1, -1.0), (n - 1, n - 1, 1.0), (n - 1, 0, -1.0)]  # noqa: E731
STRIP = 1.0 / 8.0
JACOBI_WEIGHT = 2.0 / 3.0
KS = (8, 16)

#: name -> (modes principal, bubble, sweeps)
CANDIDATES = {
    "stage-0 bubble": ("plain", "square", 0),
    "C1 frozen principal": ("frozen", "square", 0),
    "C2 sweeps x4": ("plain", "square", 4),
    "C2 sweeps x16": ("plain", "square", 16),
    "C3 strips j9-15": ("strips", "square", 0),
    "C4 angle bubble": ("plain", "angle", 0),
    "C4 corner bubble": ("plain", "corner", 0),
    "C4 + C1": ("frozen", "corner", 0),
    "C4 + C2 x16": ("plain", "corner", 16),
    "C4 + C3": ("strips", "corner", 0),
    "C4 + C3 + C2 x4": ("strips", "corner", 4),
}


def _smoothstep(x):
    y = np.clip((x - 0.25) / 0.5, 0.0, 1.0)
    return y ** 3 * (10.0 - 15.0 * y + 6.0 * y * y)


def _grid(n):
    t = np.arange(1, n) / n
    return np.meshgrid(t, t)                     # [j, i] on the interior nodes


def angle_bubble(n, a, r0=0.25):
    """``16 xi (1-xi) eta (1-eta) prod_c (r_c^2 / (r_c^2 + r0^2))^((mu_c - 2) / 2)`` with
    ``mu_c = min(pi / omega_c, 2)`` and ``cos omega_c = -a12 / sqrt(a11 a22)`` in corner c's
    own frame (the corner element's tensor): ``r^mu_c`` at each corner."""
    XI, ET = _grid(n)
    b = 16.0 * XI * (1 - XI) * ET * (1 - ET)
    mus = []
    for (x, y), (je, ie, sgn) in zip(CORNERS, CORNER_ELEMENTS(n)):
        a11, a12, a22 = (float(a[k][je, ie]) for k in ("a11", "a12", "a22"))
        omega = math.acos(max(-1.0, min(1.0, -sgn * a12 / math.sqrt(a11 * a22))))
        mu = min(math.pi / omega, 2.0)
        mus.append(mu)
        r2 = (XI - x) ** 2 + (ET - y) ** 2
        b = b * (r2 / (r2 + r0 * r0)) ** ((mu - 2.0) / 2.0)
    return b.ravel(), mus


def smooth_basis(n, K):
    XI, ET = _grid(n)
    return np.stack([(np.cos(p * math.pi * XI) * np.cos(q * math.pi * ET)).ravel()
                     for q in range(K) for p in range(K)], axis=1)


def best_fit(A, X, T):
    """The error ``T - X a`` of the A-energy-best ``a`` per column of ``T``."""
    AX = A @ X
    coef = np.linalg.lstsq(X.T @ AX, AX.T @ T, rcond=1e-13)[0]
    return T - X @ coef


def jacobi(A, diag, nu):
    def post(E):
        E = np.array(E, dtype=float, copy=True)
        for _ in range(nu):
            E -= JACOBI_WEIGHT * (A @ E) / diag[:, None]
        return E
    return post


class Piece:
    """One validation piece at one resolution: the exact objects, the principal parts of
    every candidate (each timed), and the measure."""

    def __init__(self, seed, n, Q, W, c1):
        self.seed, self.n = seed, n
        e = data.draw(seed, n=n)
        a = data.arrays(e)
        self.arrays = a
        d = {k: np.asarray(a[k], np.float64) for k in ("a11", "a12", "a22", "sigma", "w")}
        self.s0 = float(a["s0"])
        loop, inner = fe.grid(n)
        M = fe.assemble(n, d["a11"], d["a12"], d["a22"], d["sigma"]).tocsr()
        ones = np.ones((n + 1) ** 2)
        t = {}
        t0 = time.perf_counter()
        ex = fe.Exact(M, n)
        self.H = ex.modes(Q)
        self.u1 = ex.particular((fe.mass(n, d["w"]) @ ones)[inner])
        t["exact"] = time.perf_counter() - t0
        self.A = ex.A_I.tocsr()
        self.trL = float(np.trace(ex.port_matrix(Q, self.H)))
        # the stored arrays round sigma and J~ to float32 separately, so sigma = s0 J~ and
        # with it the identity hold to float32's relative rounding, not to float64's (the
        # arrays are kept as stored so that the control reproduces diagnose's labels);
        # measured: 2e-10 at n = 32, 3e-8 at n = 256, against e_H of 1e-3 and more
        ident = float(np.abs(self.H @ c1 + self.s0 * self.u1 - 1.0).max())
        self.identity_residual = ident
        if ident > 1e-6:
            raise RuntimeError(f"seed {seed} n={n}: the exact identity misses by {ident:.1e}")
        self.W, self.c1 = W, c1
        # the plain square (Stage 0's principal part and level field), by sine transforms
        t0 = time.perf_counter()
        self.P = {"plain": principal.modes(n, Q, self.s0)}
        u10 = principal.solve(n, np.full((n - 1) ** 2, 1.0 / n ** 2), self.s0)
        t["plain"] = time.perf_counter() - t0
        self.U = {"plain": u10, "strips": u10}
        self.a_level = float(np.abs(u10).max())
        # C1: the frozen-coefficient principal part, blended
        t0 = time.perf_counter()
        XI, ET = _grid(n)
        sx, sy = _smoothstep(XI).ravel(), _smoothstep(ET).ravel()
        chis = [(1 - sx) * (1 - sy), sx * (1 - sy), sx * sy, (1 - sx) * sy]
        Hf = np.zeros_like(self.H)
        uf = np.zeros_like(self.u1)
        for chi, (je, ie, _sgn) in zip(chis, CORNER_ELEMENTS(n)):
            Mc = fe.assemble(n, d["a11"][je, ie], d["a12"][je, ie], d["a22"][je, ie], d["sigma"][je, ie])
            exc = fe.Exact(Mc, n)
            Hf += chi[:, None] * exc.modes(Q)
            uf += chi * exc.particular((fe.mass(n, d["w"][je, ie]) @ ones)[inner])
        t["frozen"] = time.perf_counter() - t0
        self.P["frozen"], self.U["frozen"] = Hf, uf
        # C3: the high sines by a strip solve along their side
        t0 = time.perf_counter()
        BQ = -np.asarray(ex.M_IG @ Q)
        Hs = self.P["plain"].copy()
        JJ, II = np.meshgrid(np.arange(1, n), np.arange(1, n), indexing="ij")
        dist = [JJ.ravel(), (n - II).ravel(), (n - JJ).ravel(), II.ravel()]
        for s in range(4):
            S = dist[s] <= STRIP * n
            cs = [4 + 15 * s + (j - 1) for j in range(9, 16)]
            rhs = BQ[S][:, cs] - self.A[S][:, ~S] @ Hs[~S][:, cs]
            Hs[np.flatnonzero(S)[:, None], cs] = spla.splu(self.A[S][:, S].tocsc()).solve(rhs)
        t["strips"] = time.perf_counter() - t0
        self.P["strips"] = Hs
        # C2: the sweeps' cost on 65 columns
        self.diag = self.A.diagonal()
        for nu in (4, 16):
            Z = np.random.default_rng(0).standard_normal((self.A.shape[0], 4 * M_PORTS + 1))
            t0 = time.perf_counter()
            jacobi(self.A, self.diag, nu)(Z)
            t[f"sweeps{nu}"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.angle_beta, self.angle_mus = angle_bubble(n, a)
        t["angle"] = time.perf_counter() - t0
        self.times = t

    def energy(self, E):
        return np.einsum("ik,ik->k", E, self.A @ E)

    def e_H(self, E_modes, E_level):
        """e_H after the kernel projection of the fields with these errors (``H - E``)."""
        Hh = self.H - E_modes
        uh = self.u1 - E_level
        r = Hh @ self.c1 + self.s0 * uh - 1.0
        err = (Hh - self.W * r[:, None]) - self.H
        return math.sqrt(float(self.energy(err).sum()) / self.trL)

    def candidate(self, principal_name, bubble_name, sweeps, Phis, corner_beta):
        P = self.P[principal_name]
        U = self.U.get(principal_name, self.U["plain"])
        beta = {"square": None, "corner": corner_beta, "angle": self.angle_beta}[bubble_name]
        if beta is None:
            XI, ET = _grid(self.n)
            beta = (16.0 * XI * (1 - XI) * ET * (1 - ET)).ravel()
        learn = LOW if principal_name == "strips" else list(range(4 * M_PORTS))
        D = self.H - P
        D1 = self.u1 - U
        Nst = D[:, learn] / beta[:, None]
        N1 = D1 / (self.a_level * beta)
        post = jacobi(self.A, self.diag, sweeps) if sweeps else (lambda E: E)
        T, T1 = post(D), post(D1[:, None])[:, 0]
        out = {"peak": float(np.abs(Nst).max()), "rms": float(np.sqrt(np.mean(Nst ** 2))),
               "level_peak": float(np.abs(N1).max()), "e_H_classical": self.e_H(T, T1)}
        for K, Phi in Phis.items():
            X = post(beta[:, None] * Phi)
            E = T.copy()
            E[:, learn] = best_fit(self.A, X, T[:, learn])
            E1 = best_fit(self.A, X, T1[:, None])[:, 0]
            out[f"e_H_smooth_K{K}"] = self.e_H(E, E1)
        cost = self.times["frozen" if principal_name == "frozen" else "plain"]
        cost += self.times["strips"] if principal_name == "strips" else 0.0
        cost += self.times[f"sweeps{sweeps}"] if sweeps else 0.0
        cost += self.times["angle"] if bubble_name == "angle" else 0.0
        out["cost_s"] = cost
        out["cost_over_exact"] = cost / self.times["exact"]
        return out


def _diagnose_medians():
    paths = sorted(glob.glob(os.path.join(ROOT, "out", "chartop", "diagnose-*.json")))
    with open(paths[-1], encoding="utf-8") as fh:
        rec = json.load(fh)
    return os.path.basename(paths[-1]), {int(n): b["max_abs_median"] for n, b in rec["bubble"].items()}


def _write(path, rec):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    for k in range(20):                          # OneDrive can hold a fresh file briefly
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.5 * (k + 1))
    raise PermissionError(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ns", default="32,64,128,256")
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "chartop", "corner-controls.json"))
    ap.add_argument("--progress", default=None, help="a file for progress lines (nothing is printed)")
    args = ap.parse_args()
    ns = [int(x) for x in args.ns.split(",")]
    seeds = [data.SEEDS["val"] + 100 + i for i in range(args.count)]
    started = time.perf_counter()
    prog = open(args.progress, "a", encoding="utf-8") if args.progress else None

    def say(msg):
        if prog:
            prog.write(f"[{time.perf_counter() - started:7.0f}s] {msg}\n")
            prog.flush()

    rec = {"written": None, "script": "scripts/chartop_corner_controls.py", "seeds": [seeds[0], seeds[-1]],
           "ns": ns, "smooth_K": list(KS), "strip_width": STRIP, "jacobi_weight": JACOBI_WEIGHT,
           "candidates": {k: {"principal": v[0], "bubble": v[1], "sweeps": v[2]} for k, v in CANDIDATES.items()},
           "results": {k: {} for k in CANDIDATES}, "pieces": {}, "costs_s": {}}
    for n in ns:
        Q = ports.basis("sine-hat", n, M_PORTS).Q
        c1 = forms.kernel_coefficients("sine-hat", n, M_PORTS)
        H0s = principal.modes(n, Q, 0.0)
        W = (c1[None, :] * H0s) / (H0s @ (c1 * c1))[:, None]
        Phis = {K: smooth_basis(n, K) for K in KS}
        corner_beta = forms.corner_bubble(n)[1:-1, 1:-1].ravel()
        rows = {k: [] for k in CANDIDATES}
        times = []
        for sd in seeds:
            p = Piece(sd, n, Q, W, c1)
            times.append(p.times)
            rec["pieces"].setdefault(str(sd), {})[str(n)] = {"s0": p.s0, "angle_bubble_exponents": p.angle_mus,
                                                             "exact_identity_residual": p.identity_residual}
            for name, (pn, bn, nu) in CANDIDATES.items():
                rows[name].append(p.candidate(pn, bn, nu, Phis, corner_beta))
            say(f"n={n} seed {sd} done")
        for name, rs in rows.items():
            med = {k: float(np.median([r[k] for r in rs])) for k in rs[0]}
            rec["results"][name][str(n)] = {"median": med, "per_seed": rs}
        rec["costs_s"][str(n)] = {k: float(np.median([t[k] for t in times])) for k in times[0]}
        rec["written"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        _write(args.out, rec)                    # kept after every resolution
    # the positive control: Stage 0's bubble against diagnose's own record
    dname, dmed = _diagnose_medians()
    ctrl = {n: (rec["results"]["stage-0 bubble"][str(n)]["median"]["peak"], dmed[n]) for n in ns if n in dmed}
    rec["control"] = {"diagnose_record": dname,
                      "stage0_peak_vs_diagnose": {str(n): list(v) for n, v in ctrl.items()},
                      "reproduced": all(abs(a - b) <= 1e-6 * b for a, b in ctrl.values())}
    rec["elapsed_s"] = time.perf_counter() - started
    rec["written"] = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    _write(args.out, rec)
    # the table
    big = max(ns)
    ref = 128 if 128 in ns else big
    head = ("candidate".ljust(20) + "peak " + " ".join(f"{n:>6d}" for n in ns) + " | rms" + f"{ref:<4d}"
            + "| eH,N=0 | eH smooth K=8 " + " ".join(f"{n:>5d}" for n in ns) + f" | K=16@{ref} | cost/exact@{ref}")
    print(head)
    for name in CANDIDATES:
        R = rec["results"][name]
        peaks = " ".join(f"{R[str(n)]['median']['peak']:6.1f}" for n in ns)
        eh8 = " ".join(f"{R[str(n)]['median']['e_H_smooth_K8']:.3f}" for n in ns)
        m = R[str(ref)]["median"]
        print(f"{name[:19]:20s}     {peaks} | {m['rms']:6.3f} | {m['e_H_classical']:.3f}  |               {eh8}"
              f" | {m['e_H_smooth_K16']:.4f}   | {m['cost_over_exact']:.2f}")
    print(f"control: stage-0 peaks {'reproduce' if rec['control']['reproduced'] else 'DO NOT reproduce'} "
          f"{dname}; wrote {os.path.relpath(args.out, ROOT)}")
    return 0 if rec["control"]["reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
