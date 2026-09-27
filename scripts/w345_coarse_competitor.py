"""W345 -- is there a cheap CORRECT coarse competitor across a coupled seam?

**This measures the COMPETITOR, not a learned expert.**  Tier 48's G5 charges a
learned expert against the cheapest classical alternative in the same slot, and on
2-D single-family flow that alternative is brutal: a 2x-coarse classical monolith,
used inside defect correction, reached the certified state at 52 classical-
equivalents where the checkpoint cost 314 (six windows).  The open question the
Tier 51 brief carried for thirty-seven tiers: **a coupled seam may have no valid
coarse surrogate at all.**  If restriction and prolongation stop commuting with the
coupling, the competitor collapses to the classical solve itself and the bar a
learned expert must clear drops enormously.

The three coupled 2-D classical graphs the brief names, and what each can answer:

  CS-13 `cooling_loop`  a conduction block (the build repo's `step_thermal`,
                        backward Euler, unmodified) coupled through a THERM seam to
                        a closed four-leg coolant circuit.  It has a settled state,
                        so the defect-correction slot exists.  **Measured here.**
  CS-12 `wing_fsi`      the plate sheds: the settled load swings 1.503% peak to
                        peak (case study section 4), so the classical composed map
                        has NO isolated fixed point and the slot does not exist.
                        ``fsi_floor`` measures the one-step residual's floor to say
                        so with a number rather than a citation.
  CS-14 `powertrain`    every agent is lumped algebra (rotor disk, machine, bus,
                        battery, inverter): there is no field to restrict, so the
                        cheapest classical alternative IS the classical solve.
                        ``powertrain_cost`` prices it.

The CS-13 arms, all on one settled state:

  ``Phi``      the classical composed macro-step: block step against the passage's
               bulk temperature, the heat it delivers, the loop's closed form.
               Reproduces `LoopSolve.solve` bit for bit (``repro``).
  ``C``        the coarse competitor: restrict the block's nodal field 2x in both
               directions (injection), one coarse coupled macro-step on a 24 x 3
               block with the SAME circuit, prolong back (bilinear).
  ``Phi0``, ``C0``  the uncoupled CONTROL: the same block with the coolant held at
               the settled return temperature.  Its fixed point is the coupled one
               (at the fixed point the return temperature IS that value), so the
               two differ by the seam alone.

Writes ``out/w345/w345.json`` after every stage.  Nothing is downloaded.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse                                                         # noqa: E402
import io                                                               # noqa: E402
import json                                                             # noqa: E402
import sys                                                              # noqa: E402
import time                                                             # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import numpy as np                                                      # noqa: E402

from atlas.cases import cooling_loop as cl                              # noqa: E402
from atlas.defect_correction import (classical_march, defect_correct,   # noqa: E402
                                     theta_from_march, vector_rms)

OUT = os.path.join(HERE, "out", "w345")

#: **Registered before any arm ran** (2026-09-26).  Evaluated by `evaluate`.
PREDICTIONS = {
    "Q1": "repro: the script's classical map reproduces LoopSolve.solve's block "
          "field and return temperature bit for bit over its first 50 steps",
    "Q2": "the coarse competitor stays CORRECT across the coupled seam: defect "
          "correction with it returns a state inside its certificate, at every "
          "inner-march setting run",
    "Q3": "the coarse competitor's OWN settled state (marched alone, prolonged) is "
          "within 1% of the fine one in the heat the block delivers -- a conduction "
          "block is the easy case for a coarse model",
    "Q4": "the seam costs the competitor nothing it would not pay uncoupled: with "
          "the same settings, the coupled DC arm needs no more than 1.5x the "
          "classical calls of the uncoupled control",
    "Q5": "the coarse competitor is NOT cheap on this block: a coarse call costs "
          "more than 0.25 of a classical call, because at 343 nodes the solve is "
          "overhead-bound rather than cell-bound",
    "Q6": "the fsi_floor: CS-12's one-step residual does not fall below 1e-3 of the "
          "field's rms over 40 settled macro-steps (no isolated fixed point)",
}


def evaluate(art):
    out = {}

    def verdict(ok):
        return "not measured" if ok is None else ("held" if ok else "FAILED")

    r = art.get("repro", {})
    out["Q1"] = {"got": r.get("max_abs_diff"),
                 "verdict": verdict(None if not r else r.get("max_abs_diff") == 0.0)}
    dc = art.get("dc", {})
    ins = {k: v.get("inside_certificate") for k, v in dc.items()
           if k.startswith("C_") and isinstance(v, dict)}
    out["Q2"] = {"got": ins, "verdict": verdict(None if not ins else all(ins.values()))}
    b = art.get("bias", {})
    rel = b.get("coupled", {}).get("q_block_rel_err")
    out["Q3"] = {"got": rel, "verdict": verdict(None if rel is None else rel <= 0.01)}
    pairs = {}
    for k, v in dc.items():
        if k.startswith("C_") and f"C0_{k[2:]}" in dc:
            pairs[k] = (v["phi_calls"], dc[f"C0_{k[2:]}"]["phi_calls"])
    out["Q4"] = {"got": pairs,
                 "verdict": verdict(None if not pairs else
                                    all(a <= 1.5 * b_ for a, b_ in pairs.values()))}
    c = art.get("cost", {}).get("ratio_C")
    out["Q5"] = {"got": c, "verdict": verdict(None if c is None else c > 0.25)}
    f = art.get("fsi_floor", {}).get("min_relative_residual")
    out["Q6"] = {"got": f, "verdict": verdict(None if f is None else f >= 1e-3)}
    return out


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def _retry(fn, attempts=40, pause=0.25):
    for k in range(attempts):
        try:
            return fn()
        except PermissionError:
            if k == attempts - 1:
                raise
            time.sleep(pause)


def _f(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return [_f(v) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): _f(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_f(v) for v in x]
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    return x


def persist(art):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "w345.json")
    tmp = path + ".tmp"

    def write():
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_f(art), fh, indent=1)

    _retry(write)
    _retry(lambda: os.replace(tmp, path))
    return path


def load_art():
    path = os.path.join(OUT, "w345.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


# ---------------------------------------------------------------------------
# CS-13: the block at any resolution, and the coupled map
# ---------------------------------------------------------------------------


class Block:
    """`cooling_loop.BlockAgent` at a declared resolution, as a pure function.

    Everything is `BlockAgent`'s -- the mesh construction, the material, the
    Robin data on both faces, the heat-in integral -- with ``N_SEAM`` and
    ``NJ_BLOCK`` made parameters.  At (48, 6) it must reproduce the agent bit
    for bit (``repro``).
    """

    def __init__(self, n_seam: int = cl.N_SEAM, nj: int = cl.NJ_BLOCK):
        TS = cl.load_solvers()
        self.n_seam, self.nj = n_seam, nj
        nodes = np.stack(np.meshgrid(np.linspace(0.0, cl.L_Z, n_seam + 1),
                                     np.linspace(0.0, cl.T_BLOCK, nj + 1),
                                     indexing="ij"), -1)
        self.mesh = TS.ShellMesh(nodes)
        self.ts = TS.ThermoStruct2D(self.mesh, TS.SolidMaterial())
        self.h_seam = cl.L_Z / n_seam
        area = cl.L_Z * cl.WIDTH
        self.t_source = cl.T_COOLANT_0 + 40.0 + cl.Q_SOURCE / (cl.H_OUTER * area)
        self.n_nodes = self.mesh.n_nodes
        self.T0 = np.full(self.n_nodes, cl.T_COOLANT_0 + 40.0)

    def step(self, T, t_cool):
        Tc = np.full(self.n_seam, float(t_cool))
        return self.ts.step_thermal(T, cl.MACRO_DT, cl.H_WALL, Tc, cl.H_OUTER,
                                    self.t_source, radiate=False, T_inf=cl.T_AMB)

    def heat_in(self, T, t_cool):
        a, b, _ = self.ts._face("inner")
        Tw = 0.5 * (T[a] + T[b])
        return float(np.sum(cl.H_WALL * (Tw - float(t_cool))) * self.h_seam * cl.WIDTH)


def restrict_nodes(T, fine: Block, coarse: Block):
    """Nodal injection: coarse node (i, j) takes fine node (2i, 2j)."""
    return T.reshape(fine.n_seam + 1, fine.nj + 1)[::2, ::2].ravel().copy()


def prolong_nodes(Tc, fine: Block, coarse: Block):
    """Bilinear nodal interpolation on the uniform mesh, separable."""
    c = Tc.reshape(coarse.n_seam + 1, coarse.nj + 1)
    xi_c = np.arange(coarse.n_seam + 1) * 2.0
    xj_c = np.arange(coarse.nj + 1) * 2.0
    xi_f = np.arange(fine.n_seam + 1, dtype=float)
    xj_f = np.arange(fine.nj + 1, dtype=float)
    a = np.stack([np.interp(xi_f, xi_c, c[:, j]) for j in range(c.shape[1])], axis=1)
    f = np.stack([np.interp(xj_f, xj_c, a[i, :]) for i in range(a.shape[0])], axis=0)
    return f.ravel()


class LoopMaps:
    """The coupled map, the coarse competitor, and the uncoupled control."""

    def __init__(self):
        self.fine = Block(cl.N_SEAM, cl.NJ_BLOCK)
        self.coarse = Block(cl.N_SEAM // 2, cl.NJ_BLOCK // 2)
        self.loop = cl.LoopSolve()             # the legs and their closed form
        self.t_frozen = None                   # the uncoupled control's coolant
        self.calls = {"Phi": 0, "C": 0, "Phi0": 0, "C0": 0}

    def closed(self, q):
        return self.loop._closed_form(q)

    def w0(self):
        return np.concatenate([self.fine.T0, [cl.T_COOLANT_0]])

    def Phi(self, w):
        self.calls["Phi"] += 1
        T, t_in = w[:-1], w[-1]
        T1 = self.fine.step(T, t_in)
        return np.concatenate([T1, [self.closed(self.fine.heat_in(T1, t_in))]])

    def C(self, w):
        self.calls["C"] += 1
        T, t_in = w[:-1], w[-1]
        Tc = restrict_nodes(T, self.fine, self.coarse)
        Tc1 = self.coarse.step(Tc, t_in)
        q = self.coarse.heat_in(Tc1, t_in)
        return np.concatenate([prolong_nodes(Tc1, self.fine, self.coarse),
                               [self.closed(q)]])

    def Phi0(self, T):
        self.calls["Phi0"] += 1
        return self.fine.step(T, self.t_frozen)

    def C0(self, T):
        self.calls["C0"] += 1
        Tc = restrict_nodes(T, self.fine, self.coarse)
        return prolong_nodes(self.coarse.step(Tc, self.t_frozen), self.fine, self.coarse)

    def coarse_alone(self, w):
        """The competitor's own coupled macro-step on the COARSE state."""
        Tc, t_in = w[:-1], w[-1]
        Tc1 = self.coarse.step(Tc, t_in)
        return np.concatenate([Tc1, [self.closed(self.coarse.heat_in(Tc1, t_in))]])


def stage_repro(M: LoopMaps, art):
    """Q1: the map against `LoopSolve.solve`'s own loop body, 50 steps."""
    ls = cl.LoopSolve()
    blk = ls.block
    w = M.w0()
    t_in = cl.T_COOLANT_0
    worst = 0.0
    for _ in range(50):
        ls.legs["PASS"].t_in = t_in
        wall = ls.legs["PASS"].wall_temperature()
        blk.step(wall)
        t_in = ls._closed_form(blk.heat_in(wall))
        w = M.Phi(w)
        worst = max(worst, float(np.max(np.abs(w[:-1] - blk._T))),
                    abs(float(w[-1]) - t_in))
    art["repro"] = {"steps": 50, "max_abs_diff": worst}
    persist(art)
    print(f"  repro: max |diff| over 50 steps = {worst:.3e}", flush=True)


def stage_reference(M: LoopMaps, art, max_steps=80000, floor=1e-12):
    """The classical settled state, by the classical march, to the CG floor."""
    t0 = time.time()
    w = M.w0()
    res_hist, states = [], []
    for j in range(max_steps):
        fw = M.Phi(w)
        r = vector_rms(fw - w)
        res_hist.append(r)
        if j % 10 == 0:
            states.append((j, w.copy()))
        w = fw
        if r <= floor * max(1.0, vector_rms(w)):
            break
    w_star = w
    idx = [j for j, _ in states]
    err = [vector_rms(s - w_star) for _, s in states]
    cold = err[0]
    target = 1e-3 * cold
    m_hit = next(m for m, e in zip(idx, err) if e <= target)
    r_stop = res_hist[m_hit]
    cold_calls = next(j for j, r in enumerate(res_hist) if r <= r_stop) + 1
    th = theta_from_march(err, [res_hist[j] for j in idx])
    np.save(os.path.join(OUT, "w_star.npy"), w_star)
    art["reference"] = {
        "steps": len(res_hist), "final_residual": res_hist[-1],
        "cold_distance": cold, "r_stop": r_stop, "cold_calls": cold_calls,
        "theta": th, "t_return": float(w_star[-1]),
        "q_block": M.fine.heat_in(w_star[:-1], w_star[-1]),
        "wall_s": time.time() - t0}
    persist(art)
    print(f"  reference: {len(res_hist)} steps, t_return {w_star[-1]:.6f} K, "
          f"cold calls to 1e-3: {cold_calls}, theta {th['theta']:.3g}", flush=True)
    return w_star


def stage_bias(M: LoopMaps, art, w_star, max_steps=80000):
    """Q3: the competitor marched ALONE to its own settled state, prolonged."""
    out = {}
    fine_q = M.fine.heat_in(w_star[:-1], w_star[-1])
    # coupled coarse loop
    wc = np.concatenate([restrict_nodes(M.w0()[:-1], M.fine, M.coarse), [cl.T_COOLANT_0]])
    for _ in range(max_steps):
        nx = M.coarse_alone(wc)
        if vector_rms(nx - wc) <= 1e-12 * vector_rms(nx):
            wc = nx
            break
        wc = nx
    qc = M.coarse.heat_in(wc[:-1], wc[-1])
    Tp = prolong_nodes(wc[:-1], M.fine, M.coarse)
    out["coupled"] = {
        "t_return": float(wc[-1]), "t_return_fine": float(w_star[-1]),
        "t_return_abs_err": abs(float(wc[-1]) - float(w_star[-1])),
        "q_block": qc, "q_block_fine": fine_q,
        "q_block_rel_err": abs(qc - fine_q) / abs(fine_q),
        "field_rms_err": vector_rms(Tp - w_star[:-1])}
    # uncoupled control: coolant frozen at the fine settled return
    Tc = restrict_nodes(M.w0()[:-1], M.fine, M.coarse)
    for _ in range(max_steps):
        nx = M.coarse.step(Tc, w_star[-1])
        if vector_rms(nx - Tc) <= 1e-12 * vector_rms(nx):
            Tc = nx
            break
        Tc = nx
    qc0 = M.coarse.heat_in(Tc, w_star[-1])
    out["uncoupled"] = {
        "q_block": qc0, "q_block_rel_err": abs(qc0 - fine_q) / abs(fine_q),
        "field_rms_err": vector_rms(prolong_nodes(Tc, M.fine, M.coarse) - w_star[:-1])}
    art["bias"] = out
    persist(art)
    print(f"  bias: coupled q err {out['coupled']['q_block_rel_err']:.3e}, "
          f"t_return err {out['coupled']['t_return_abs_err']:.3e} K; uncoupled q err "
          f"{out['uncoupled']['q_block_rel_err']:.3e}", flush=True)


def stage_dc(M: LoopMaps, art, w_star, m_values=(40, 400, 4000)):
    """The arms: C and the uncoupled C0, at several inner-march lengths."""
    ref = art["reference"]
    r_stop, theta = ref["r_stop"], ref["theta"]["theta"]
    M.t_frozen = float(w_star[-1])
    dc = art.setdefault("dc", {})
    # the uncoupled control's own cold march and stopping rule, same recipe
    T_star = w_star[:-1]
    cm0 = classical_march(M.Phi0, M.w0()[:-1], r_stop=r_stop * 1e-3, max_steps=80000,
                          reference=T_star)
    err0 = [r["error"] for r in cm0.rows]
    res0 = [r["residual"] for r in cm0.rows]
    m_hit = next(m for m, e in enumerate(err0) if e <= 1e-3 * err0[0])
    r_stop0 = res0[m_hit]
    cold0 = next(j for j, r in enumerate(res0) if r <= r_stop0) + 1
    theta0 = theta_from_march(err0, res0)["theta"]
    dc["uncoupled_reference"] = {"r_stop": r_stop0, "cold_calls": cold0,
                                 "theta": theta0}
    print(f"  uncoupled control: cold calls {cold0}", flush=True)
    for m in m_values:
        settings = {"k_max": 60, "m_max": int(m), "inner_frac": 0.1,
                    "stall_ratio": 1.0, "stall_patience": 2,
                    "fallback_steps": 100000}
        for key, phi, psi, w0, rs, th, ref_state in (
                (f"C_m{m}", M.Phi, M.C, M.w0(), r_stop, theta, w_star),
                (f"C0_m{m}", M.Phi0, M.C0, M.w0()[:-1], r_stop0, theta0, T_star)):
            if key in dc:
                continue
            t0 = time.time()
            res = defect_correct(phi, psi, w0, r_stop=rs, reference=ref_state,
                                 fallback=True, **settings)
            err = vector_rms(res.state - ref_state)
            d = res.as_dict()
            d.pop("rows", None)
            d.pop("fallback_rows", None)
            d.update({"settings": settings, "final_error": err,
                      "certificate_bound": th * res.residual,
                      "inside_certificate": bool(res.converged
                                                 and err <= th * res.residual),
                      "wall_s": time.time() - t0})
            dc[key] = d
            persist(art)
            print(f"  [{key}] phi {d['phi_calls']:6d}  psi {d['psi_calls']:7d}  "
                  f"{d['status']:18s} err {err:.3e}  inside {d['inside_certificate']}",
                  flush=True)
    dc["cold_calls_coupled"] = ref["cold_calls"]


def stage_cost(M: LoopMaps, art, w_star, repeats=5, calls=40):
    """Per-call cost of each map, interleaved, min over repeats of the mean."""
    M.t_frozen = float(w_star[-1])
    fns = {"Phi": (M.Phi, w_star), "C": (M.C, w_star),
           "Phi0": (M.Phi0, w_star[:-1]), "C0": (M.C0, w_star[:-1])}
    best = {k: float("inf") for k in fns}
    for _ in range(repeats):
        for k, (fn, w) in fns.items():
            t0 = time.perf_counter()
            for _i in range(calls):
                fn(w)
            best[k] = min(best[k], (time.perf_counter() - t0) / calls)
    art["cost"] = {"seconds_per_call": best,
                   "ratio_C": best["C"] / best["Phi"],
                   "ratio_C0": best["C0"] / best["Phi0"],
                   "repeats": repeats, "calls": calls,
                   "note": "min over repeats of the mean over calls; one process"}
    persist(art)
    print(f"  cost: C/Phi {art['cost']['ratio_C']:.3f}   C0/Phi0 "
          f"{art['cost']['ratio_C0']:.3f}   Phi {best['Phi'] * 1e3:.2f} ms", flush=True)


# ---------------------------------------------------------------------------
# CS-12 and CS-14
# ---------------------------------------------------------------------------


def stage_fsi_floor(art, spin=160, window=40):
    """Q6: does CS-12's composed map have a fixed point?  Its one-step residual."""
    import torch                                                          # noqa: PLC0415
    from atlas.cases import wing_fsi as wf                                # noqa: PLC0415
    t0 = time.time()
    ro = wf.FSIRollout()
    r = ro.run(steps=spin + window, keep_fields=tuple(range(spin, spin + window + 1)))
    rel = []
    for s in range(spin, spin + window):
        u0, v0, d0 = r.fields[s]
        u1, v1, d1 = r.fields[s + 1]
        num = np.sqrt(np.mean((u1 - u0) ** 2 + (v1 - v0) ** 2))
        den = np.sqrt(np.mean(u1 ** 2 + v1 ** 2))
        rel.append(float(num / den))
    art["fsi_floor"] = {
        "spin_steps": spin, "window": window, "relative_residuals": rel,
        "min_relative_residual": min(rel), "max_relative_residual": max(rel),
        "load_last_window_peak_to_peak": float(
            (r.load[-window:].max() - r.load[-window:].min())
            / abs(r.load[-window:].mean())),
        "wall_s": time.time() - t0, "torch_threads": torch.get_num_threads()}
    persist(art)
    print(f"  fsi floor: one-step residual {min(rel):.3e}..{max(rel):.3e} of the "
          f"field rms; load p-p {art['fsi_floor']['load_last_window_peak_to_peak']:.3%}",
          flush=True)


def stage_powertrain(art, repeats=5, calls=200):
    from atlas.cases import powertrain as pt                              # noqa: PLC0415
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _i in range(calls):
            pt.CircuitSolve().solve()
        best = min(best, (time.perf_counter() - t0) / calls)
    art["powertrain"] = {"seconds_per_solve": best,
                         "note": "every agent is lumped: there is no field to "
                                 "restrict, so no coarse competitor exists and the "
                                 "cheapest classical alternative is this solve"}
    persist(art)
    print(f"  powertrain: one full circuit solve {best * 1e3:.3f} ms", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="repro,reference,bias,dc,cost,powertrain,fsi")
    ap.add_argument("--m", default="40,400,4000")
    a = ap.parse_args(argv)
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    os.makedirs(OUT, exist_ok=True)
    art = load_art()
    art["what"] = "W345: a cheap correct coarse competitor across a coupled seam?"
    art["predictions"] = PREDICTIONS
    art.setdefault("registered", "2026-09-26, before any arm ran")
    M = LoopMaps()
    w_star = None
    if "repro" in stages:
        stage_repro(M, art)
    if "reference" in stages or not os.path.isfile(os.path.join(OUT, "w_star.npy")):
        w_star = stage_reference(M, art)
    if w_star is None:
        w_star = np.load(os.path.join(OUT, "w_star.npy"))
    if "bias" in stages:
        stage_bias(M, art, w_star)
    if "dc" in stages:
        stage_dc(M, art, w_star, tuple(int(x) for x in a.m.split(",")))
    if "cost" in stages:
        stage_cost(M, art, w_star)
    if "powertrain" in stages:
        stage_powertrain(art)
    if "fsi" in stages:
        stage_fsi_floor(art)
    art["evaluation"] = evaluate(art)
    print("wrote", persist(art), flush=True)
    for k, v in art["evaluation"].items():
        print(f"  {k}: {v['verdict']}  {v['got']}", flush=True)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    main()
