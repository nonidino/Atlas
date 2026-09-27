"""W345, CS-12 arm -- the coarse competitor across the wing's two-way MECH seam.

`scripts/w345_coarse_competitor.py` first assumed CS-12 has no settled state,
reading the case study's "settled unsteadiness 1.503% peak to peak".  Its own
floor stage refuted that: over 40 macro-steps the one-step residual FELL, and a
2000-step march (``out/w345/fsi_long.json``) shows it falling geometrically from
2.9e-3 to 1.3e-9 of the field, with the load's peak-to-peak dropping from 40% in
the first quarter to 2.2e-6 in the last.  The 1.5% was the transient 120 steps
in.  So the defect-correction slot exists on CS-12 too, and this builds its
competitor.

**The coarse wing is the case study at half resolution, not a new model.**  The
case's resolution is module constants in `ground_effect` (``DX``, ``NW``,
``HALO``, ``RAMP``, ``BAND``, ``PAD_CELLS``, ``X_LE``, ``D_OFFSET``, ``SIGMA_N``) and
one in `wing_fsi` (``Y_MOUNT_CELLS``), so both modules are loaded as PRIVATE COPIES
with those constants halved in cells and held in physical units -- every
replacement asserted to match exactly once -- and nothing in `atlas/` is edited.
The control that makes the copy trustworthy: the same loader with NO replacement
must reproduce the real module's macro-step bit for bit (``loader_control``).

Arms, on one settled state (fluid u, v, deflection, plate velocity):

  ``Phi``   `FSIRollout.macro_step`, lagged coupling at the case's own lag of 1
  ``C``     restrict (2x2 average), the coarse copy's macro-step with the SAME 33
            structural stations, prolong (bilinear): the competitor
  ``Phi0``, ``C0``  the seam frozen: `motion=False`, the deflection held at the
            settled one.  Same fixed point, no two-way coupling.

**This measures the COMPETITOR, not a learned expert.**
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse                                                         # noqa: E402
import importlib.util                                                   # noqa: E402
import io                                                               # noqa: E402
import json                                                             # noqa: E402
import sys                                                              # noqa: E402
import time                                                             # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

import numpy as np                                                      # noqa: E402

import w345_coarse_competitor as W                                      # noqa: E402
from w202_kill_tests import prolong2, restrict2                         # noqa: E402
from atlas.defect_correction import defect_correct, theta_from_march, vector_rms  # noqa: E402

OUT = os.path.join(HERE, "out", "w345")
CASES = os.path.join(HERE, "atlas", "cases")

#: **Registered before any arm ran** (2026-09-26, after the long march above and
#: before the coarse copy was first built).  Evaluated by `evaluate`.
PREDICTIONS = {
    "F0": "loader control: an UNPATCHED private copy reproduces the real "
          "`wing_fsi` macro-step bit for bit",
    "F1": "the coarse wing (cell Reynolds 7.8 where the fine one is 3.9) settles, "
          "and its settled load is within 10% of the fine wing's",
    "F2": "the coarse competitor stays CORRECT across the two-way seam: defect "
          "correction with it returns a state inside its certificate at every "
          "inner-march setting",
    "F3": "the seam costs the competitor nothing it would not pay frozen: the "
          "coupled arm needs no more than 1.5x the classical calls of the frozen "
          "control at the same settings",
    "F4": "a coarse call costs at most 0.35 of a fine call (a quarter of the cells, "
          "plus torch overhead)",
}

G_PATCH = [("DX = 1.0 / 64.0", "DX = 1.0 / 32.0"),
           ("NW = 80\n", "NW = 40\n"),
           ("HALO = 16\n", "HALO = 8\n"),
           ("RAMP = 8\n", "RAMP = 4\n"),
           ("BAND = 8\n", "BAND = 4\n"),
           ("PAD_CELLS = 32\n", "PAD_CELLS = 16\n"),
           ("X_LE = 88 * DX", "X_LE = 44 * DX"),
           ("D_OFFSET = 2.5\n", "D_OFFSET = 1.25\n"),
           ("SIGMA_N = 1.5\n", "SIGMA_N = 0.75\n")]
F_PATCH = [("Y_MOUNT_CELLS = 40\n", "Y_MOUNT_CELLS = 20\n")]


def _load_copy(stem: str, suffix: str, g_patch, f_patch):
    """Private copies of ground_effect and wing_fsi, under atlas.cases.*_<suffix>."""
    mods = {}
    for name, patch in (("ground_effect", g_patch), ("wing_fsi", f_patch)):
        with open(os.path.join(CASES, name + ".py"), encoding="utf-8") as fh:
            src = fh.read()
        for old, new in patch:
            n = src.count(old)
            if n != 1:
                raise RuntimeError(f"{name}: {old!r} matched {n} times, expected 1")
            src = src.replace(old, new)
        if name == "wing_fsi":
            old = "from .ground_effect import ("
            assert src.count(old) == 1, "wing_fsi's ground_effect import moved"
            src = src.replace(old, f"from .ground_effect_{suffix} import (")
        full = f"atlas.cases.{name}_{suffix}"
        spec = importlib.util.spec_from_loader(full, loader=None)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "atlas.cases"
        mod.__file__ = os.path.join(CASES, f"{name}_{suffix}.py")
        sys.modules[full] = mod
        exec(compile(src, mod.__file__, "exec"), mod.__dict__)
        mods[name] = mod
    return mods["wing_fsi"]


class FSIMaps:
    def __init__(self):
        import torch                                                      # noqa: PLC0415
        from atlas.cases import wing_fsi as wf                            # noqa: PLC0415
        self.torch = torch
        self.wf = wf
        self.wc = _load_copy("wing_fsi", "c2", G_PATCH, F_PATCH)
        self.fine = wf.FSIRollout()
        self.coarse = self.wc.FSIRollout()
        self.fine0 = wf.FSIRollout(motion=False)
        self.coarse0 = self.wc.FSIRollout(motion=False)
        self.e = torch.tensor(float(wf.E_STAR), dtype=wf.TORCH_DTYPE)
        self.ny, self.nx = self.fine.ny, self.fine.nx
        self.nyc, self.nxc = self.coarse.ny, self.coarse.nx
        assert (self.nyc * 2, self.nxc * 2) == (self.ny, self.nx), (
            "the coarse copy is not half the fine grid")
        self.ns = wf.N_STATION
        self.delta_star = None

    # -- state packing -----------------------------------------------------
    def pack(self, u, v, d, w):
        return np.concatenate([np.asarray(u).ravel(), np.asarray(v).ravel(),
                               np.asarray(d).ravel(), np.asarray(w).ravel()])

    def unpack(self, x):
        n = self.ny * self.nx
        return (x[:n].reshape(self.ny, self.nx), x[n:2 * n].reshape(self.ny, self.nx),
                x[2 * n:2 * n + self.ns], x[2 * n + self.ns:])

    def _t(self, a):
        return self.torch.as_tensor(np.asarray(a), dtype=self.wf.TORCH_DTYPE)

    def _step(self, ro, u, v, d, w):
        with self.torch.no_grad():
            out = ro.macro_step(self._t(u), self._t(v), self._t(d), self._t(w), self.e, 0)
        return tuple(o.detach().cpu().numpy() for o in out[:4])

    def w0(self):
        return self.pack(np.full((self.ny, self.nx), self.wf.U_INF),
                         np.zeros((self.ny, self.nx)), np.zeros(self.ns), np.zeros(self.ns))

    def Phi(self, x):
        return self.pack(*self._step(self.fine, *self.unpack(x)))

    def C(self, x):
        u, v, d, w = self.unpack(x)
        uc, vc, d1, w1 = self._step(self.coarse, restrict2(u), restrict2(v), d, w)
        return self.pack(prolong2(uc), prolong2(vc), d1, w1)

    def fluid(self, x):
        n = self.ny * self.nx
        return x[:2 * n]

    def Phi0(self, xf):
        n = self.ny * self.nx
        u, v = xf[:n].reshape(self.ny, self.nx), xf[n:].reshape(self.ny, self.nx)
        u1, v1, _, _ = self._step(self.fine0, u, v, self.delta_star, np.zeros(self.ns))
        return np.concatenate([u1.ravel(), v1.ravel()])

    def C0(self, xf):
        n = self.ny * self.nx
        u, v = xf[:n].reshape(self.ny, self.nx), xf[n:].reshape(self.ny, self.nx)
        u1, v1, _, _ = self._step(self.coarse0, restrict2(u), restrict2(v),
                                  self.delta_star, np.zeros(self.ns))
        return np.concatenate([prolong2(u1).ravel(), prolong2(v1).ravel()])

    def load(self, x, coarse=False):
        u, v, d, w = self.unpack(x)
        ro = self.fine
        with self.torch.no_grad():
            out = ro.macro_step(self._t(u), self._t(v), self._t(d), self._t(w), self.e, 0)
        return float(out[4])


def evaluate(art):
    out = {}

    def v(ok):
        return "not measured" if ok is None else ("held" if ok else "FAILED")

    f = art.get("fsi", {})
    lc = f.get("loader_control")
    out["F0"] = {"got": lc, "verdict": v(lc)}
    b = f.get("bias", {}).get("load_rel_err")
    out["F1"] = {"got": b, "verdict": v(None if b is None else b <= 0.10)}
    dc = f.get("dc", {})
    ins = {k: d.get("inside_certificate") for k, d in dc.items() if k.startswith("C_")}
    out["F2"] = {"got": ins, "verdict": v(None if not ins else all(ins.values()))}
    pairs = {k: (d["phi_calls"], dc[f"C0_{k[2:]}"]["phi_calls"]) for k, d in dc.items()
             if k.startswith("C_") and f"C0_{k[2:]}" in dc}
    out["F3"] = {"got": pairs, "verdict": v(None if not pairs else
                                             all(a <= 1.5 * b_ for a, b_ in pairs.values()))}
    c = f.get("cost", {}).get("ratio_C")
    out["F4"] = {"got": c, "verdict": v(None if c is None else c <= 0.35)}
    return out


def loader_control(art):
    """F0: the loader with NO patch reproduces the real module bit for bit."""
    import torch                                                          # noqa: PLC0415
    from atlas.cases import wing_fsi as wf                                # noqa: PLC0415
    wid = _load_copy("wing_fsi", "c1", [], [])
    a, b = wf.FSIRollout(), wid.FSIRollout()
    e = torch.tensor(float(wf.E_STAR), dtype=wf.TORCH_DTYPE)
    ny, nx, ns = a.ny, a.nx, wf.N_STATION
    rng = np.random.default_rng(20260926)
    u = 1.0 + 0.05 * rng.standard_normal((ny, nx))
    v = 0.05 * rng.standard_normal((ny, nx))
    d = 1e-3 * rng.standard_normal(ns)
    w = 1e-3 * rng.standard_normal(ns)
    t = lambda z: torch.as_tensor(z, dtype=wf.TORCH_DTYPE)              # noqa: E731
    with torch.no_grad():
        oa = a.macro_step(t(u), t(v), t(d), t(w), e, 0)
        ob = b.macro_step(t(u), t(v), t(d), t(w), e, 0)
    ok = all(bool(torch.equal(x, y)) for x, y in zip(oa[:6], ob[:6]))
    art.setdefault("fsi", {})["loader_control"] = ok
    W.persist(art)
    print(f"  loader control (unpatched copy == module): {ok}", flush=True)
    return ok


def stage_reference(M, art, max_steps=6000, floor=1e-12):
    t0 = time.time()
    x = M.w0()
    res, states = [], []
    for j in range(max_steps):
        fx = M.Phi(x)
        r = vector_rms(fx - x)
        res.append(r)
        if j % 10 == 0:
            states.append((j, x.copy()))
        x = fx
        if r <= floor * vector_rms(x):
            break
    idx = [j for j, _ in states]
    err = [vector_rms(s - x) for _, s in states]
    target = 1e-3 * err[0]
    m_hit = next(m for m, e_ in zip(idx, err) if e_ <= target)
    r_stop = res[m_hit]
    cold = next(j for j, r in enumerate(res) if r <= r_stop) + 1
    th = theta_from_march(err, [res[j] for j in idx])
    np.save(os.path.join(OUT, "fsi_x_star.npy"), x)
    f = art.setdefault("fsi", {})
    f["reference"] = {"steps": len(res), "final_residual": res[-1], "r_stop": r_stop,
                      "cold_calls": cold, "cold_distance": err[0], "theta": th,
                      "load": M.load(x), "tip": float(M.unpack(x)[2][-1]),
                      "wall_s": time.time() - t0}
    W.persist(art)
    print(f"  fsi reference: {len(res)} steps, cold calls {cold}, theta {th['theta']:.3g}, "
          f"load {f['reference']['load']:.6f}", flush=True)
    return x


def stage_bias(M, art, x_star, max_steps=6000):
    """F1: the coarse wing marched alone to its own settled state."""
    u, v, d, w = M.unpack(M.w0())
    uc, vc = restrict2(u), restrict2(v)
    last = None
    for j in range(max_steps):
        uc1, vc1, d1, w1 = M._step(M.coarse, uc, vc, d, w)
        r = float(np.sqrt(np.mean((uc1 - uc) ** 2 + (vc1 - vc) ** 2)))
        uc, vc, d, w = uc1, vc1, d1, w1
        if not np.isfinite(r):
            break
        last = r
        if r <= 1e-12:
            break
    with M.torch.no_grad():
        out = M.coarse.macro_step(M._t(uc), M._t(vc), M._t(d), M._t(w), M.e, 0)
    load_c = float(out[4])
    load_f = art["fsi"]["reference"]["load"]
    uf, vf, df, _ = M.unpack(x_star)
    art["fsi"]["bias"] = {
        "coarse_steps": j + 1, "coarse_final_residual": last,
        "load_coarse": load_c, "load_fine": load_f,
        "load_rel_err": abs(load_c - load_f) / abs(load_f),
        "tip_coarse": float(d[-1]), "tip_fine": float(df[-1]),
        "field_rms_err": float(np.sqrt(np.mean((prolong2(uc) - uf) ** 2
                                               + (prolong2(vc) - vf) ** 2)))}
    W.persist(art)
    print(f"  fsi bias: load coarse {load_c:.6f} fine {load_f:.6f} "
          f"({art['fsi']['bias']['load_rel_err']:.3%}); tip {d[-1]:.6f} vs {df[-1]:.6f}",
          flush=True)


def stage_dc(M, art, x_star, m_values=(40, 400)):
    f = art["fsi"]
    ref = f["reference"]
    r_stop, theta = ref["r_stop"], ref["theta"]["theta"]
    uf, vf, df, wfv = M.unpack(x_star)
    M.delta_star = df
    xf_star = M.fluid(x_star)
    # the frozen control's own cold march and stopping rule
    x = M.fluid(M.w0())
    res0, states0 = [], []
    for j in range(6000):
        fx = M.Phi0(x)
        r = vector_rms(fx - x)
        res0.append(r)
        if j % 10 == 0:
            states0.append((j, x.copy()))
        x = fx
        if r <= 1e-12 * vector_rms(x):
            break
    err0 = [vector_rms(s - x) for _, s in states0]
    idx0 = [j for j, _ in states0]
    m_hit = next(m for m, e_ in zip(idx0, err0) if e_ <= 1e-3 * err0[0])
    r_stop0 = res0[m_hit]
    cold0 = next(j for j, r in enumerate(res0) if r <= r_stop0) + 1
    theta0 = theta_from_march(err0, [res0[j] for j in idx0])["theta"]
    x0_star = x
    f["frozen_reference"] = {"cold_calls": cold0, "r_stop": r_stop0, "theta": theta0,
                             "distance_to_coupled_fluid": vector_rms(x0_star - xf_star)}
    W.persist(art)
    print(f"  frozen control: cold calls {cold0}, its fixed point is "
          f"{f['frozen_reference']['distance_to_coupled_fluid']:.2e} from the coupled one",
          flush=True)
    dc = f.setdefault("dc", {})
    for m in m_values:
        settings = {"k_max": 60, "m_max": int(m), "inner_frac": 0.1, "stall_ratio": 1.0,
                    "stall_patience": 2, "fallback_steps": 20000}
        for key, phi, psi, w0, rs, th, rf in (
                (f"C_m{m}", M.Phi, M.C, M.w0(), r_stop, theta, x_star),
                (f"C0_m{m}", M.Phi0, M.C0, M.fluid(M.w0()), r_stop0, theta0, x0_star)):
            if key in dc:
                continue
            t0 = time.time()
            res = defect_correct(phi, psi, w0, r_stop=rs, reference=rf, fallback=True,
                                 **settings)
            err = vector_rms(res.state - rf)
            d = res.as_dict()
            d.pop("rows", None)
            d.pop("fallback_rows", None)
            d.update({"settings": settings, "final_error": err,
                      "certificate_bound": th * res.residual,
                      "inside_certificate": bool(res.converged and err <= th * res.residual),
                      "wall_s": time.time() - t0})
            dc[key] = d
            W.persist(art)
            print(f"  [fsi {key}] phi {d['phi_calls']:5d}  psi {d['psi_calls']:6d}  "
                  f"{d['status']:18s} err {err:.3e} inside {d['inside_certificate']}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    f["cold_calls"] = ref["cold_calls"]
    W.persist(art)


def stage_cost(M, art, x_star, repeats=5, calls=10):
    M.delta_star = M.unpack(x_star)[2]
    xf = M.fluid(x_star)
    fns = {"Phi": (M.Phi, x_star), "C": (M.C, x_star), "Phi0": (M.Phi0, xf),
           "C0": (M.C0, xf)}
    best = {k: float("inf") for k in fns}
    for _ in range(repeats):
        for k, (fn, x) in fns.items():
            t0 = time.perf_counter()
            for _i in range(calls):
                fn(x)
            best[k] = min(best[k], (time.perf_counter() - t0) / calls)
    art["fsi"]["cost"] = {"seconds_per_call": best, "ratio_C": best["C"] / best["Phi"],
                          "ratio_C0": best["C0"] / best["Phi0"],
                          "torch_threads": M.torch.get_num_threads()}
    W.persist(art)
    print(f"  fsi cost: C/Phi {best['C'] / best['Phi']:.3f}  Phi {best['Phi'] * 1e3:.1f} ms",
          flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="control,reference,bias,dc")
    ap.add_argument("--m", default="40,400")
    a = ap.parse_args(argv)
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    art = W.load_art()
    art["fsi_predictions"] = PREDICTIONS
    art.setdefault("fsi_registered", "2026-09-26, before the coarse copy was first built")
    if "control" in stages and not loader_control(art):
        print("  the loader does not reproduce the module; stopping", flush=True)
        return
    M = FSIMaps()
    xs = os.path.join(OUT, "fsi_x_star.npy")
    x_star = stage_reference(M, art) if ("reference" in stages or not os.path.isfile(xs)) \
        else np.load(xs)
    if "bias" in stages:
        stage_bias(M, art, x_star)
    if "dc" in stages:
        stage_dc(M, art, x_star, tuple(int(z) for z in a.m.split(",")))
    if "cost" in stages:
        stage_cost(M, art, x_star)
    art["fsi_evaluation"] = evaluate(art)
    print("wrote", W.persist(art), flush=True)
    for k, v in art["fsi_evaluation"].items():
        print(f"  {k}: {v['verdict']}  {v['got']}", flush=True)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    main()
