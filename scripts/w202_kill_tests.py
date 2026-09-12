"""Tier 48, phase 2 -- the kill tests, minutes each, on the wake array's own graphs.

Driver for [[learned-contribution-kill-tests]] (the phase-1/phase-2 record) and the
input to [[defect-correction-learned-operator]] (the converged formulation).
Artifact: ``out/w202/w202.json``.  Caches: ``out/w202/*.npz`` (gitignored).

    python scripts/w202_kill_tests.py                      # every stage, N=2
    python scripts/w202_kill_tests.py --stages cost --rungs 2,6
    python scripts/w202_kill_tests.py --stages reference,dec --rung 6

What each stage decides
-----------------------

``repro``      The control the rest leans on: the Poseidon map here IS CS-7's
               composed step, bitwise, and the classical map IS the long march's
               monolith advance, bitwise.
``cost``       Per-call wall cost, in-process, as RATIOS to one classical monolith
               macro-step: the Poseidon composed step (P), the composition layer
               with the checkpoint replaced by the identity (Z), a 2x-coarse
               classical monolith used as a fine-grid operator (C), and the
               classical composed column (E, exposed + projected).
``reference``  The classical settled state: the monolith marched from the
               freestream until its one-step change stops falling, with the error
               and residual histories and the stability constant Theta read off it.
``warm``       Does a learned START shorten the classical march to the settled
               state?  Cold start, Poseidon's own settled state, a start at the
               same distance built from smooth noise, the null expert's settled
               state, and the ring control that exhibits a non-isolated fixed point.
``dec``        Defect correction (`atlas/defect_correction.py`): the classical
               settled state found by classical defects and a cheap operator's
               inner march.  Cheap operators P, Z, C and a detuned checkpoint (Pc),
               each also shrunk toward the null element.  Converges to the
               CLASSICAL answer whatever the cheap operator is; what the operator
               changes is how many classical calls it takes.
``energy``     The relative-energy (weak-strong) a-posteriori certificate: its
               floor on exact data over one macro-step, and its growth factor
               over one transit of the domain.
``selfcons``   Lead-time self-consistency, one step at 2 dt against two at dt: a
               referent-free LOWER bound on the checkpoint's error, and the
               identity map it cannot see.
``finetune``   The price of one fine-tuning iteration of Poseidon-T on this CPU.
               Nothing is trained.

Every number is a ratio or a norm measured in this process.  Nothing is
downloaded; the checkpoint loads from the local Hugging Face cache offline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np                                                    # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from atlas.cases import scaling_ladder as sl                         # noqa: E402
from atlas.cases import wake_array as wa                             # noqa: E402
from atlas.defect_correction import (                                # noqa: E402
    classical_march,
    defect_correct,
    shrink,
    theta_from_march,
    vector_rms,
)

OUT = os.path.join(_ROOT, "out", "w202")
ART = os.path.join(OUT, "w202.json")


# ---------------------------------------------------------------------------
# pure helpers -- importable by the tests without either expert
# ---------------------------------------------------------------------------


def rms_pair(du: np.ndarray, dv: np.ndarray) -> float:
    """rms of a velocity-vector difference over all cells."""
    return float(np.sqrt(np.mean(du * du + dv * dv)))


def restrict2(a: np.ndarray) -> np.ndarray:
    """2x2 cell average, cell-centred.  Shape (ny, nx) -> (ny/2, nx/2)."""
    ny, nx = a.shape
    if ny % 2 or nx % 2:
        raise ValueError(f"restrict2 needs even dimensions, got {a.shape}")
    return a.reshape(ny // 2, 2, nx // 2, 2).mean(axis=(1, 3))


def prolong2(c: np.ndarray) -> np.ndarray:
    """Cell-centred bilinear prolongation, (nyc, nxc) -> (2 nyc, 2 nxc).

    Each fine cell takes 9/16 of its parent and 3/16, 3/16, 1/16 of the parent's
    three nearest neighbours, with edge replication.  Reproduces constants and
    linear fields in the interior, which is the property a two-grid correction
    needs of it.
    """
    p = np.pad(c, 1, mode="edge")
    nyc, nxc = c.shape
    out = np.empty((2 * nyc, 2 * nxc))
    ctr = p[1:-1, 1:-1]
    for dy, sy in ((0, -1), (1, +1)):
        for dx, sx in ((0, -1), (1, +1)):
            ny_ = p[1 + sy:nyc + 1 + sy, 1:-1]
            nx_ = p[1:-1, 1 + sx:nxc + 1 + sx]
            dg = p[1 + sy:nyc + 1 + sy, 1 + sx:nxc + 1 + sx]
            out[dy::2, dx::2] = (9.0 * ctr + 3.0 * ny_ + 3.0 * nx_ + dg) / 16.0
    return out


def hermite(w0, f0, w1, f1, delta: float, s: float):
    """Cubic Hermite reconstruction on [0, delta] and its time derivative at s."""
    t = s / delta
    h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
    h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
    d00, d10 = 6 * t**2 - 6 * t, 3 * t**2 - 4 * t + 1
    d01, d11 = -6 * t**2 + 6 * t, 3 * t**2 - 2 * t
    val = h00 * w0 + delta * h10 * f0 + h01 * w1 + delta * h11 * f1
    der = (d00 * w0 + delta * d10 * f0 + d01 * w1 + delta * d11 * f1) / delta
    return val, der


GAUSS3 = ((0.5 - np.sqrt(15.0) / 10.0, 5.0 / 18.0), (0.5, 8.0 / 18.0),
          (0.5 + np.sqrt(15.0) / 10.0, 5.0 / 18.0))


def strain_growth_bound(u: np.ndarray, v: np.ndarray, dx: float, interior: int) -> float:
    """max over cells of -lambda_min(sym grad u): the pointwise energy growth rate.

    For e solving the error equation of 2-D incompressible Navier-Stokes the
    nonlinear term (e.grad e, e) vanishes, and -(e.grad u, e) = -(e, S e) with S
    the symmetric velocity gradient, so d/dt ||e|| <= s ||e|| + ||R|| with
    s <= max_x(-lambda_min(S(x))).  Centred differences, band cells excluded.
    """
    ux = (u[:, 2:] - u[:, :-2]) / (2 * dx)
    vx = (v[:, 2:] - v[:, :-2]) / (2 * dx)
    uy = (u[2:, :] - u[:-2, :]) / (2 * dx)
    vy = (v[2:, :] - v[:-2, :]) / (2 * dx)
    ux, vx = ux[1:-1, :], vx[1:-1, :]
    uy, vy = uy[:, 1:-1], vy[:, 1:-1]
    s11, s22, s12 = ux, vy, 0.5 * (uy + vx)
    lam_min = 0.5 * (s11 + s22) - np.sqrt((0.5 * (s11 - s22)) ** 2 + s12 ** 2)
    k = max(interior - 1, 0)
    core = (-lam_min)[k:-k or None, k:-k or None]
    return float(np.max(core))


def error_structure(e: np.ndarray, weights, offsets, n_win: int) -> dict:
    """Where an error field lives: its x-uniform share, its lowest-mode share and
    the share carried by a blend of per-window constants.

    ``e`` is ``(2, ny, nx)``.  The window-mean field is ``sum_k chi_k m_k`` with
    ``m_k`` the unweighted mean of ``e`` over window k's box -- the quantity the
    Poseidon column's composition wrapper holds fixed per window.
    """
    tot = float(np.sum(e * e))
    if tot <= 0.0:
        return {"x_uniform_share": None, "lowest_modes_share": None,
                "window_mean_share": None}
    xm = np.broadcast_to(e.mean(axis=-1, keepdims=True), e.shape)
    x_share = float(np.sum(xm * xm)) / tot
    ny, nx = e.shape[1], e.shape[2]
    ky = np.abs(np.fft.fftfreq(ny) * ny)[:, None]
    kx = np.abs(np.fft.fftfreq(nx) * nx)[None, :]
    mask = (ky <= 2) & (kx <= 2)
    low = sum(float(np.sum(np.abs(np.fft.fft2(e[c])[mask]) ** 2)) / (ny * nx)
              for c in range(e.shape[0]))
    wm = np.zeros_like(e)
    for k, (ox, oy) in enumerate(offsets):
        m = e[:, oy:oy + n_win, ox:ox + n_win].mean(axis=(1, 2))
        wm += np.asarray(weights[k])[None, :, :] * m[:, None, None]
    return {"x_uniform_share": x_share, "lowest_modes_share": low / tot,
            "window_mean_share": float(np.sum(wm * wm)) / tot}


def _f(x):
    """JSON-safe."""
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


# ---------------------------------------------------------------------------
# the maps -- each one is (u, v) -> (u, v) including the disks and the band
# ---------------------------------------------------------------------------


class Maps:
    """The one-macro-step maps on one rung, sharing forcing and band.

    ``F`` is the classical monolith, the referent.  ``P`` is Poseidon-T's composed
    column exactly as `w100_scaling_ladder.composed_step` runs it, with the time
    step as a parameter (stage ``repro`` asserts it bit-identical at the native
    step).  ``Z`` is the same composition layer with the checkpoint replaced by the
    identity -- the W76 control: whatever ``P`` does that ``Z`` also does is not the
    checkpoint's.  ``C`` is a 2x-coarse classical monolith used as a fine-grid
    operator: restrict, one coarse macro-step, prolong.  ``Pc`` is the checkpoint
    with its output fluctuation scaled by 0.8, a detuned expert.
    """

    def __init__(self, n_windows: int, load_poseidon: bool = True):
        import w100_scaling_ladder as L                                  # noqa: PLC0415
        self.L = L
        cr = {1: (1, 1), 2: (2, 1), 6: (3, 2), 12: (4, 3), 24: (6, 4)}[n_windows]
        self.r = sl.rung(*cr)
        self.active = {x.rotor_id for x in self.r.tiling.rotors}
        self.ny, self.nx = self.r.shape
        self.mono = sl.reference_monolith(self.nx, self.ny, wa.NU_REF)
        self.coarse = sl.RectangularNS(nu=wa.NU_REF, length=(self.nx // 2) * 2 * wa.DX,
                                       n=self.nx // 2, ny=self.ny // 2, cfl=0.4,
                                       transmission="dirichlet")
        self.exposed = wa.exposed_reference_solver(wa.NU_REF)
        self.assembly = L.assembly_for(self.r)
        self.ex = None
        if load_poseidon:
            import torch                                                 # noqa: PLC0415
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert torch.backends.cudnn.allow_tf32 is False
            self.ex = wa.scaled_expert(threads=8)
        self.calls = {"F": 0, "P": 0, "Z": 0, "C": 0, "E": 0, "Pc": 0, "Pw": 0}
        #: `Pw`'s corruption: Gaussian noise on every weight tensor at this fraction
        #: of the tensor's own rms, drawn once from a fixed seed.
        self.corrupt_sigma = 0.03
        self._corrupted = None
        #: **The outflow ring is problem data, not state.**  `WindowNS.step_batch`
        #: with ``bc0=None`` holds every ring at its INPUT value and `_band` resets
        #: only the inlet and the two laterals, so the last column is whatever the
        #: starting state put there, forever.  The classical map's fixed points are
        #: then a family parameterised by that column, and a start carrying a wake
        #: at the outlet converges to a different member (stage ``warm``'s ring
        #: control).  From the freestream the column is U_INF at every step, so
        #: pinning it changes the reference trajectory by nothing.
        self.pin_outflow = True

    # -- shared pieces -----------------------------------------------------

    def forcing(self, u, active=None):
        fx, rec = self.L._forcing(self.r, u, self.active if active is None else active)
        return fx, rec

    def band(self, u, v):
        u, v = self.L._band(self.r, u, v)
        if self.pin_outflow:
            u[:, -1] = wa.U_INF
            v[:, -1] = 0.0
        return u, v

    # -- the maps ----------------------------------------------------------

    def F(self, u, v, dt=wa.MACRO_DT, active=None):
        self.calls["F"] += 1
        fx, _ = self.forcing(u, active)
        uu, vv = self.mono.step_batch(u[None], v[None], dt, bc0=None,
                                      force=(fx[None], np.zeros_like(fx)[None]))
        return self.band(uu[0], vv[0])

    def _composed(self, u, v, fx, dt, expert_step):
        t = self.r.tiling
        ufs, vfs = t.cut(u - wa.U_INF), t.cut(v)
        u1, v1 = expert_step(ufs, vfs, dt)
        u1 = u1 + (ufs.mean(axis=(1, 2)) - u1.mean(axis=(1, 2)))[:, None, None]
        v1 = v1 + (vfs.mean(axis=(1, 2)) - v1.mean(axis=(1, 2)))[:, None, None]
        uf, vf = t.assemble(u1, v1)
        uf = uf + fx * dt
        uf, vf = wa.transport_and_project(uf, vf, dt, wa.U_INF)
        return self.band(wa.U_INF + uf, vf)

    def _poseidon_step(self, ufs, vfs, dt):
        return self.ex.step_many(ufs, vfs, dt, galilean=False, project=False)

    def P(self, u, v, dt=wa.MACRO_DT, active=None):
        self.calls["P"] += 1
        fx, _ = self.forcing(u, active)
        return self._composed(u, v, fx, dt, self._poseidon_step)

    def Pc(self, u, v, dt=wa.MACRO_DT, active=None):
        self.calls["Pc"] += 1
        fx, _ = self.forcing(u, active)

        def detuned(ufs, vfs, dt_):
            a, b = self._poseidon_step(ufs, vfs, dt_)
            return 0.8 * a, 0.8 * b

        return self._composed(u, v, fx, dt, detuned)

    def Pw(self, u, v, dt=wa.MACRO_DT, active=None):
        """The checkpoint with Gaussian noise on its WEIGHTS: deterministic and wrong.

        ``corrupt_sigma`` of each tensor's own rms, drawn once with a fixed seed.  A
        corrupted checkpoint rather than a noisy one: the same input gives the same
        output, so what it gets wrong is its Jacobian as well as its value -- the
        failure `Pc`'s uniform detuning turned out too gentle to be.
        """
        self.calls["Pw"] += 1
        if self._corrupted is None:
            import copy                                                   # noqa: PLC0415
            import torch                                                  # noqa: PLC0415
            ex = copy.copy(self.ex)
            model = copy.deepcopy(self.ex.model)
            gen = torch.Generator().manual_seed(20260911)
            with torch.no_grad():
                for p in model.parameters():
                    scale = p.detach().float().pow(2).mean().sqrt()
                    p.add_(torch.randn(p.shape, generator=gen) * self.corrupt_sigma * scale)
            ex.model = model
            self._corrupted = ex

        def corrupted(ufs, vfs, dt_):
            return self._corrupted.step_many(ufs, vfs, dt_, galilean=False, project=False)

        fx, _ = self.forcing(u, active)
        return self._composed(u, v, fx, dt, corrupted)

    def Z(self, u, v, dt=wa.MACRO_DT, active=None):
        self.calls["Z"] += 1
        fx, _ = self.forcing(u, active)
        return self._composed(u, v, fx, dt, lambda a, b, _dt: (a.copy(), b.copy()))

    def C(self, u, v, dt=wa.MACRO_DT, active=None):
        self.calls["C"] += 1
        fx, _ = self.forcing(u, active)
        uc, vc, fc = restrict2(u), restrict2(v), restrict2(fx)
        uu, vv = self.coarse.step_batch(uc[None], vc[None], dt, bc0=None,
                                        force=(fc[None], np.zeros_like(fc)[None]))
        return self.band(prolong2(uu[0]), prolong2(vv[0]))

    def E(self, u, v, dt=wa.MACRO_DT, active=None):
        self.calls["E"] += 1
        fx, _ = self.forcing(u, active)
        u1, v1, _loc = self.L.composed_step(self.r, u, v, fx, "reference_exposed",
                                            self.exposed, assembly=self.assembly)
        return self.band(u1, v1)

    def freestream(self):
        return np.ones(self.r.shape), np.zeros(self.r.shape)


def _stack(fn):
    """A ``(u, v) -> (u, v)`` map as a ``(2, ny, nx) -> (2, ny, nx)`` map."""
    def g(w):
        u, v = fn(np.array(w[0], copy=True), np.array(w[1], copy=True))
        return np.stack([u, v])
    return g


# ---------------------------------------------------------------------------
# the stages
# ---------------------------------------------------------------------------


def _load_art():
    if os.path.isfile(ART):
        with open(ART, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _persist(art):
    os.makedirs(OUT, exist_ok=True)
    tmp = ART + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_f(art), fh, indent=1)
    for attempt in range(20):
        try:
            os.replace(tmp, ART)
            break
        except PermissionError:
            time.sleep(0.5 * (attempt + 1))
    print(f"  [artifact -> {ART}]", flush=True)


def _cache(name):
    return os.path.join(OUT, name)


def stage_repro(n, art):
    """The control every other stage leans on: ``Maps.P`` IS CS-7's Poseidon step.

    `Maps` re-writes `w100_scaling_ladder.composed_step`'s Poseidon branch with
    the time step as a parameter.  A re-implementation that drifted from the
    original would make every number below about a different column, so the two
    are compared at the native step on a developed state, bitwise, and the stage
    refuses to pass otherwise.  The classical map gets the same check against the
    long march's monolith advance.
    """
    M = Maps(n)
    z = np.load(os.path.join(_ROOT, "out", "w100", f"state_N{n}.npz"))
    u, v = z["poseidon_all_u"].copy(), z["poseidon_all_v"].copy()
    fx, _ = M.forcing(u)
    ou, ov, _loc = M.L.composed_step(M.r, u.copy(), v.copy(), fx, "poseidon", M.ex)
    ou, ov = M.band(ou, ov)
    pu, pv = M.P(u.copy(), v.copy())
    diff_p = float(max(np.max(np.abs(pu - ou)), np.max(np.abs(pv - ov))))
    mu_, mv_ = M.mono.step_batch(u[None], v[None], wa.MACRO_DT, bc0=None,
                                 force=(fx[None], np.zeros_like(fx)[None]))
    mu_, mv_ = M.band(mu_[0], mv_[0])
    fu, fv = M.F(u.copy(), v.copy())
    diff_f = float(max(np.max(np.abs(fu - mu_)), np.max(np.abs(fv - mv_))))
    zu, zv = M.Z(u.copy(), v.copy())
    p_minus_z = rms_pair(pu - zu, pv - zv)
    rec = {"rung": f"N{n}", "poseidon_vs_composed_step_maxabs": diff_p,
           "monolith_vs_long_march_advance_maxabs": diff_f,
           "poseidon_minus_null_expert_rms": p_minus_z,
           "passed": bool(diff_p == 0.0 and diff_f == 0.0 and p_minus_z > 0.0)}
    art.setdefault("repro", {})[f"N{n}"] = rec
    _persist(art)
    print(f"    repro N{n}: P vs composed_step {diff_p:.3e}  F vs advance {diff_f:.3e}  "
          f"P-Z {p_minus_z:.3e}  passed={rec['passed']}", flush=True)
    if not rec["passed"]:
        raise SystemExit("the reproduction control failed; nothing below it means anything")


def stage_cost(rungs, art):
    """Per-call cost ratios to the classical monolith, interleaved, min over repeats."""
    out = {"note": "min over 3 interleaved repeats of the mean over 3 calls; "
                   "ratios to the monolith macro-step F in the same process",
           "rows": []}
    for n in rungs:
        M = Maps(n)
        if n == 1:
            u, v = M.freestream()
        else:
            z = np.load(os.path.join(_ROOT, "out", "w100", f"state_N{n}.npz"))
            u, v = z["poseidon_all_u"].copy(), z["poseidon_all_v"].copy()
        maps = {"F": M.F, "P": M.P, "Z": M.Z, "C": M.C, "E": M.E}
        for fn in maps.values():                  # first-call costs out of the way
            fn(u.copy(), v.copy())
        best = {k: float("inf") for k in maps}
        for _rep in range(3):
            for k, fn in maps.items():
                t0 = time.perf_counter()
                for _ in range(3):
                    fn(u.copy(), v.copy())
                best[k] = min(best[k], (time.perf_counter() - t0) / 3.0)
        row = {"rung": f"N{n}", "seconds_per_call": best,
               "ratio_to_F": {k: best[k] / best["F"] for k in best},
               "substeps_monolith": int(M.mono.last_substeps),
               "substeps_coarse": int(M.coarse.last_substeps)}
        out["rows"].append(row)
        print(f"  N{n}: " + "  ".join(f"{k}={best[k]*1e3:.1f}ms ({best[k]/best['F']:.3f}F)"
                                     for k in best), flush=True)
        art["cost"] = out
        _persist(art)


def stage_reference(n, art, max_steps=900, sub_every=5):
    """The classical settled state, and the cold march's residual and error history.

    Up to N=6 the error history is exact at every step, from a second pass over
    the same deterministic march that also checks the march IS deterministic.
    Above that a monolith step costs seconds, so the states are kept every
    ``sub_every`` steps instead and the error history is exact at those steps only;
    `stage_dec` reads the cold march's classical-call count off the residual
    history, which is exact at every step either way.
    """
    M = Maps(n, load_poseidon=False)
    u, v = M.freestream()
    hist_res, snaps, sub = [], {}, {}
    replay_pass = n <= 6
    t0 = time.time()
    for m in range(max_steps):
        if not replay_pass and m % sub_every == 0:
            sub[m] = (u.copy(), v.copy())
        un, vn = M.F(u, v)
        res = rms_pair(un - u, vn - v)
        hist_res.append(res)
        if m in (5, 6, 20, 21, 60, 61):
            snaps[m] = (u.copy(), v.copy())
        u, v = un, vn
        if (m + 1) % 50 == 0:
            print(f"    reference N{n} step {m+1} residual {res:.3e}  {time.time()-t0:.0f}s",
                  flush=True)
        if not np.all(np.isfinite(u)):
            raise RuntimeError("monolith reference diverged")
        if m > 100 and res < 1e-11:
            break
    np.savez_compressed(_cache(f"ref_N{n}.npz"), u=u, v=v,
                        **{f"s{k}_u": a for k, (a, _b) in snaps.items()},
                        **{f"s{k}_v": b for k, (_a, b) in snaps.items()})
    if replay_pass:
        uu, vv = M.freestream()
        hist_err = []
        for m in range(len(hist_res)):
            hist_err.append(rms_pair(uu - u, vv - v))
            uu, vv = M.F(uu, vv)
        replay = rms_pair(uu - u, vv - v)
        err_index = list(range(len(hist_res)))
    else:
        err_index = sorted(sub)
        hist_err = [rms_pair(sub[m][0] - u, sub[m][1] - v) for m in err_index]
        replay = None
    rec = {"rung": f"N{n}", "steps": len(hist_res), "final_residual": hist_res[-1],
           "residual": hist_res, "error_to_final": hist_err, "error_index": err_index,
           "replay_difference": replay, "wall_s": time.time() - t0,
           "cold_distance": hist_err[0],
           "theta": theta_from_march(hist_err, [hist_res[m] for m in err_index])}
    art.setdefault("reference", {})[f"N{n}"] = rec
    _persist(art)
    rep = "not run (subsampled)" if replay is None else f"{replay:.3e}"
    print(f"  reference N{n}: {len(hist_res)} steps, final residual {hist_res[-1]:.3e}, "
          f"replay {rep}, theta {rec['theta']}", flush=True)


def _load_ref(n):
    return np.load(_cache(f"ref_N{n}.npz"))


def _steps_to(err_hist, levels):
    out = {}
    for lv in levels:
        hit = next((i for i, e in enumerate(err_hist) if e <= lv), None)
        out[f"{lv:.0e}"] = hit
    return out


def _march_err(fn, u, v, ref, cap, stop):
    hist = []
    for _m in range(cap):
        hist.append(rms_pair(u - ref[0], v - ref[1]))
        if hist[-1] <= stop:
            break
        u, v = fn(u, v)
        if not np.all(np.isfinite(u)):
            hist.append(float("inf"))
            break
    return u, v, hist


def stage_warm(n, art, cap=320):
    """Cold, learned, matched-noise and null-expert starts for the classical march."""
    M = Maps(n)
    z = _load_ref(n)
    ref = (z["u"], z["v"])
    d_cold = rms_pair(np.ones_like(ref[0]) - ref[0], -ref[1])
    levels = [d_cold * f for f in (1e-1, 1e-2, 1e-3, 1e-4)]
    out = {"rung": f"N{n}", "cold_distance": d_cold,
           "levels_relative": [1e-1, 1e-2, 1e-3, 1e-4], "arms": {}}

    starts = {}
    # the learned expert's own settled state, and the null expert's -- with the
    # outflow ring pinned, which is the problem the reference solves
    for tag, fn in (("poseidon_settled", M.P), ("null_expert_settled", M.Z)):
        u, v = M.freestream()
        t0 = time.time()
        for _m in range(100):
            u, v = fn(u, v)
        starts[tag] = (u, v)
        out[f"{tag}_distance"] = rms_pair(u - ref[0], v - ref[1])
        print(f"    {tag}: distance to classical settled {out[f'{tag}_distance']:.4e} "
              f"({out[f'{tag}_distance']/d_cold:.3f} of cold)  {time.time()-t0:.0f}s",
              flush=True)
    # the ring control: Poseidon's settled state with the outflow ring left as
    # state, marched classically with it left as state -- the arm that exhibits
    # a non-isolated classical fixed point
    M.pin_outflow = False
    u, v = M.freestream()
    for _m in range(100):
        u, v = M.P(u, v)
    ring_start = (u, v)
    out["ring_control_outflow_ring_departure"] = rms_pair(u[:, -1] - wa.U_INF, v[:, -1])
    M.pin_outflow = True
    # smooth noise at the checkpoint's distance: the start that has only its size
    rng = np.random.default_rng(20260911)
    nu_, nv_ = rng.standard_normal(ref[0].shape), rng.standard_normal(ref[0].shape)
    for _ in range(12):
        nu_ = 0.2 * (np.roll(nu_, 1, 0) + np.roll(nu_, -1, 0) + np.roll(nu_, 1, 1)
                     + np.roll(nu_, -1, 1) + nu_)
        nv_ = 0.2 * (np.roll(nv_, 1, 0) + np.roll(nv_, -1, 0) + np.roll(nv_, 1, 1)
                     + np.roll(nv_, -1, 1) + nv_)
    s = out["poseidon_settled_distance"] / max(rms_pair(nu_, nv_), 1e-300)
    su, sv = M.band(ref[0] + s * nu_, ref[1] + s * nv_)
    starts["matched_noise"] = (su, sv)
    out["matched_noise_distance"] = rms_pair(su - ref[0], sv - ref[1])
    starts["cold"] = M.freestream()

    arms = [(tag, st, True) for tag, st in starts.items()]
    arms.append(("ring_control_unpinned", ring_start, False))
    for tag, (u0, v0), pinned in arms:
        t0 = time.time()
        f0 = M.calls["F"]
        M.pin_outflow = pinned
        u0, v0 = u0.copy(), v0.copy()
        if pinned:
            u0, v0 = M.band(u0, v0)
        _u, _v, hist = _march_err(M.F, u0, v0, ref, cap, levels[-1])
        # the returned state's own one-step residual, under the same pinning:
        # what says a march that ends away from the reference has CONVERGED there
        fu, fv = M.F(_u.copy(), _v.copy())
        final_residual = rms_pair(fu - _u, fv - _v)
        M.pin_outflow = True
        out["arms"][tag] = {"start_distance": hist[0], "steps_to": _steps_to(hist, levels),
                            "F_calls": M.calls["F"] - f0, "history": hist,
                            "outflow_ring_pinned": pinned, "final_error": hist[-1],
                            "final_residual": final_residual,
                            "wall_s": time.time() - t0}
        print(f"    warm arm {tag:24s} start {hist[0]:.3e}  final {hist[-1]:.3e}  steps_to "
              f"{out['arms'][tag]['steps_to']}  {time.time()-t0:.0f}s", flush=True)
        art.setdefault("warm", {})[f"N{n}"] = out
        _persist(art)


#: The corrected iteration's settings, fixed here on N=2 and carried unchanged to
#: every other rung -- the out-of-sample clause of the gate is about exactly this.
#: ``stall_ratio`` is one: the rule catches a residual that does not fall, and a
#: ratio below one would misfire on the classical march itself, which contracts
#: at 0.894 per step on this rung's tail.
DEC_SETTINGS = {"k_max": 25, "m_max": 40, "inner_frac": 0.1, "stall_ratio": 1.0,
                "stall_patience": 2, "fallback_steps": 1000, "divergence_bound": 30.0}


def stage_dec(n, art, arms=("C", "Z", "P", "Pc"), alphas=(0.0, 0.5, 0.8),
              keys=None, marched=(), marched_steps=50):
    """Defect correction to the classical settled state, per cheap operator and shrink.

    ``keys``, when given, is an explicit list like ``["P_a0.5", "C_a0"]`` and
    replaces the ``arms`` x ``alphas`` grid.  ``marched`` lists arms whose answer is
    marched ``marched_steps`` further classically, to check that the certified
    state is settled rather than merely low in residual.
    """
    M = Maps(n)
    z = _load_ref(n)
    ref = np.stack([z["u"], z["v"]])
    rec = art["reference"][f"N{n}"]
    target = 1e-3 * rec["cold_distance"]
    idx = rec.get("error_index") or list(range(len(rec["error_to_final"])))
    m_hit = next(m for m, e in zip(idx, rec["error_to_final"]) if e <= target)
    r_stop = rec["residual"][m_hit]
    # the cold count is the first step whose RESIDUAL is at or below r_stop, which
    # the residual history gives exactly at every step on every rung
    cold_calls = next(j for j, r in enumerate(rec["residual"]) if r <= r_stop) + 1
    phi = _stack(M.F)
    w0 = np.stack(M.freestream())
    out = art.setdefault("dec", {}).setdefault(f"N{n}", {})
    out.update({"r_stop": r_stop, "target_error": target, "settings": DEC_SETTINGS,
                "theta": rec["theta"], "cold_calls_from_reference": cold_calls,
                "corrupt_sigma": M.corrupt_sigma})
    # a full-grid run OWNS the arms block: an arm from an earlier naming scheme
    # must not survive into the committed artifact just because nothing replaced it
    out["arms"] = {} if keys is None else out.get("arms", {})
    if n <= 6:
        t0 = time.time()
        cold = classical_march(phi, w0, r_stop=r_stop, reference=ref, max_steps=2000)
        out["cold"] = {"status": cold.status, "phi_calls": cold.phi_calls,
                       "final_error": vector_rms(cold.state - ref), "residual": cold.residual,
                       "reference_index_check": cold_calls, "wall_s": time.time() - t0}
        print(f"    cold march: {cold.phi_calls} classical calls to r_stop {r_stop:.3e} "
              f"(reference count {cold_calls}), error {out['cold']['final_error']:.3e}",
              flush=True)
    else:
        out["cold"] = {"status": "from_reference", "phi_calls": cold_calls}
        print(f"    cold march (from the reference history): {cold_calls} classical calls "
              f"to r_stop {r_stop:.3e}", flush=True)
    _persist(art)
    ops = {"C": M.C, "Z": M.Z, "P": M.P, "Pc": M.Pc, "Pw": M.Pw}
    t = M.r.tiling
    grid = keys if keys else [f"{a}_a{alpha:g}" for a in arms for alpha in alphas]
    theta = rec["theta"]["theta"]
    for key in grid:
        a, alpha = key.split("_a")[0], float(key.split("_a")[1])
        psi = _stack(ops[a])
        if alpha > 0.0:
            psi = shrink(psi, alpha)
        calls0 = dict(M.calls)
        t0 = time.time()

        def printer(row, key=key, t0=t0):
            where = (f"k={row['k']:2d}" if "k" in row
                     else f"fallback step {row['j']:3d}")
            print(f"      dec[{key}] {where} residual {row['residual']:.3e} "
                  f"error {row.get('error', float('nan')):.3e} "
                  f"phi={row['phi_calls']} psi={row['psi_calls']}  "
                  f"{time.time()-t0:.0f}s", flush=True)

        res = defect_correct(phi, psi, w0, r_stop=r_stop, reference=ref,
                             on_row=printer, fallback=True, **DEC_SETTINGS)
        d = res.as_dict()
        final_error = vector_rms(res.state - ref)
        d.update({"operator": a, "alpha": alpha, "wall_s": time.time() - t0,
                  "final_error": final_error,
                  "model_calls": {k: M.calls[k] - calls0[k] for k in M.calls},
                  "certificate_bound": theta * res.residual if res.converged else None,
                  "inside_certificate": bool(res.converged
                                             and final_error <= theta * res.residual)})
        if res.pre_fallback_state is not None:
            e = res.pre_fallback_state - ref
            d["pre_fallback_error"] = vector_rms(e)
            d["pre_fallback_error_structure"] = error_structure(
                e, t.weights(), t.offsets, wa.N)
        if key in marched and res.converged:
            # G4: the certified state is settled, not merely low in residual --
            # the classical march continued from it stays within 2 Theta r
            w = res.state.copy()
            dev = []
            for _j in range(marched_steps):
                w = phi(w)
                dev.append(vector_rms(w - res.state))
            d["marched"] = {"steps": marched_steps, "max_deviation": max(dev),
                            "final_deviation": dev[-1],
                            "two_theta_r": 2.0 * theta * res.residual,
                            "within": bool(max(dev) <= 2.0 * theta * res.residual)}
            print(f"    marched {key}: {marched_steps} classical steps, max deviation "
                  f"{max(dev):.3e} against 2 Theta r {2.0 * theta * res.residual:.3e}",
                  flush=True)
        print(f"    dec {key}: status {res.status} (stopped: {res.stop_reason})  "
              f"phi {res.phi_calls}  psi {res.psi_calls}  final error {final_error:.3e}  "
              f"fallback_from {res.fallback_from}", flush=True)
        out["arms"][key] = d
        _persist(art)


def stage_account(art):
    """The gate's arithmetic, in the artifact rather than in prose.

    Reads the cost ledger and every defect-correction arm already recorded, and
    writes per-arm cost in classical calls, the saving against the cold march, the
    certificate check, and the six clauses of the gate evaluated per rung.  It
    recomputes nothing physical; it is arithmetic over the record, so the pages and
    the tests can quote one place.  The detuned and corrupted checkpoints are the
    same network and are charged at the checkpoint's own per-call ratio.
    """
    ratios = {row["rung"]: row["ratio_to_F"] for row in art.get("cost", {}).get("rows", [])}
    out = {}
    for rung_key, d in sorted(art.get("dec", {}).items()):
        if rung_key not in ratios or "cold" not in d:
            continue
        cold = d["cold"]["phi_calls"]
        theta = d["theta"]["theta"]
        cert = theta * d["r_stop"]
        rows = {}
        for key, arm in d["arms"].items():
            if "operator" not in arm:
                continue
            op = arm["operator"]
            c = ratios[rung_key]["P" if op in ("Pc", "Pw") else op]
            total = arm["phi_calls"] + c * arm["psi_calls"]
            rows[key] = {
                "operator": op, "alpha": arm["alpha"], "c_psi": c,
                "phi_calls": arm["phi_calls"], "psi_calls": arm["psi_calls"],
                "cost_in_classical_calls": total,
                "saving_ratio": cold / total,
                "classical_calls_over_cold": arm["phi_calls"] / cold,
                "final_error": arm["final_error"],
                # Section 3's certificate is the RETURNED state's own residual times
                # Theta, not the stopping threshold's.  A fallback march can overshoot
                # far below r_stop, and then the bound it carries is the tighter one it
                # actually earned; evaluating the gate against theta*r_stop would be a
                # looser test than the one that was pre-registered.
                "certificate": theta * arm["residual"],
                "certificate_at_r_stop": cert,
                "inside_certificate": bool(arm["final_error"] <= theta * arm["residual"]),
                "constant_this_arm_needs": arm["final_error"] / arm["residual"],
                "marched_within": (arm.get("marched") or {}).get("within"),
            }
        p_, z_, pw_ = rows.get("P_a0.5"), rows.get("Z_a0.5"), rows.get("Pw_a0.5")
        c_costs = [v["cost_in_classical_calls"] for v in rows.values()
                   if v["operator"] == "C"]
        gate = {"alpha_star": 0.5, "cold_calls": cold, "certificate": cert}
        if p_:
            nulls = min([cold] + ([z_["phi_calls"]] if z_ else []))
            gate["G1_non_null"] = bool(p_["phi_calls"] <= 0.8 * nulls)
            gate["G1_margin"] = p_["phi_calls"] / (0.8 * nulls)
            gate["G2_non_vacuous"] = bool(all(v["inside_certificate"] for v in rows.values())
                                          and p_["certificate"] / p_["final_error"] <= 10.0)
            gate["G2_bound_over_error"] = p_["certificate"] / p_["final_error"]
            outside = sorted(k for k, v in rows.items() if not v["inside_certificate"])
            gate["G2_arms_outside"] = outside
            gate["G2_worst_constant_needed"] = max(
                v["constant_this_arm_needs"] for v in rows.values())
            gate["G2_theta_from_cold_march"] = theta
            marched = [v["marched_within"] for v in rows.values()
                       if v["marched_within"] is not None]
            # No marched arm on a rung means G4 was not MEASURED there, which is not
            # the same as failing it: section 7 asks for it at six and twelve windows.
            gate["G4_marched"] = all(marched) if marched else None
            if c_costs:
                gate["G5_accounted"] = bool(p_["cost_in_classical_calls"] <= cold / 1.5
                                            and p_["cost_in_classical_calls"] <= min(c_costs))
                gate["G5_cost"] = p_["cost_in_classical_calls"]
                gate["G5_best_classical_cost"] = min(c_costs)
        if pw_ and p_:
            gate["G6_loud"] = bool(pw_["inside_certificate"]
                                   and pw_["phi_calls"] >= p_["phi_calls"])
            gate["G6_corrupted_classical_calls"] = pw_["phi_calls"]
        out[rung_key] = {"arms": rows, "gate": gate}
        passed = {k: v for k, v in gate.items() if k.startswith("G") and isinstance(v, bool)}
        print(f"    {rung_key}: cold {cold} calls, certificate {cert:.3e}; gate {passed}",
              flush=True)
    art["accounting"] = out
    _persist(art)


def _f_of(M, u, v):
    """The scheme's semi-discrete time derivative at (u, v).

    The divergence-free part of the momentum right-hand side -- advection,
    diffusion and the disks' force -- with the ring and the band held.  The
    PROJECTION acts on the right-hand side and never on the state, so a
    reconstructed intermediate state that is not exactly solenoidal does not read
    as a time derivative of size divergence over step.
    """
    fx, _ = M.forcing(u)
    ru, rv = M.mono._rhs(u[None], v[None], fx[None], np.zeros_like(fx)[None])
    pu, pv = M.mono._project(ru, rv)
    du, dv = np.array(pu[0]), np.array(pv[0])
    b = M.L.BAND
    for a in (du, dv):
        a[:, :b] = 0.0
        a[:b, :] = 0.0
        a[-b:, :] = 0.0
        a[:, -1] = 0.0
    return du, dv


def _eta_certificate(M, w0, w1, delta, s_star):
    """Hermite-reconstruction residual certificate over one macro-step.

    ``integral over [0, delta] of exp(s_star (delta - s)) ||R(s)|| ds`` by
    three-point Gauss-Legendre, with ``R = d/ds u_rec - f(u_rec)`` and ``f`` the
    scheme's semi-discrete time derivative.  The quadrature is not itself a bound;
    the three residual norms are returned so their spread can be read.
    """
    f0 = _f_of(M, *w0)
    f1 = _f_of(M, *w1)
    eta = 0.0
    norms = []
    for (tq, wq) in GAUSS3:
        s = tq * delta
        uval, uder = hermite(w0[0], f0[0], w1[0], f1[0], delta, s)
        vval, vder = hermite(w0[1], f0[1], w1[1], f1[1], delta, s)
        fu, fv = _f_of(M, uval, vval)
        r = rms_pair(uder - fu, vder - fv)
        norms.append(r)
        eta += delta * wq * r * float(np.exp(s_star * (delta - s)))
    return eta, norms


def stage_energy(n, art):
    M = Maps(n)
    z = _load_ref(n)
    ref = (z["u"], z["v"])
    dt = wa.MACRO_DT
    s_settled = strain_growth_bound(ref[0], ref[1], wa.DX, M.L.BAND)
    transit = M.nx * wa.DX / wa.U_INF
    out = {"rung": f"N{n}", "s_star_settled": s_settled, "transit_time": transit,
           "growth_one_step": float(np.exp(s_settled * dt)),
           "growth_one_transit_log10": float(s_settled * transit / np.log(10.0)),
           "f_operator": "projected momentum right-hand side, ring and band held",
           "rows": []}
    for m in (5, 20):
        w0 = (z[f"s{m}_u"], z[f"s{m}_v"])
        w1 = (z[f"s{m+1}_u"], z[f"s{m+1}_v"])
        s_m = strain_growth_bound(w0[0], w0[1], wa.DX, M.L.BAND)
        change = rms_pair(w1[0] - w0[0], w1[1] - w0[1])
        row = {"step": m, "s_star": s_m, "flow_one_step_change": change}
        eta_c, n_c = _eta_certificate(M, w0, w1, dt, s_m)
        row["classical"] = {"eta": eta_c, "true_error": 0.0, "residual_norms": n_c,
                            "eta_over_change": eta_c / max(change, 1e-300)}
        pu, pv = M.P(*w0)
        err_p = rms_pair(pu - w1[0], pv - w1[1])
        eta_p, n_p = _eta_certificate(M, w0, (pu, pv), dt, s_m)
        row["poseidon"] = {"eta": eta_p, "true_error": err_p, "residual_norms": n_p,
                           "eta_over_true": eta_p / max(err_p, 1e-300)}
        eta_i, n_i = _eta_certificate(M, w0, w0, dt, s_m)
        row["persistence"] = {"eta": eta_i, "true_error": change, "residual_norms": n_i,
                              "eta_over_true": eta_i / max(change, 1e-300)}
        out["rows"].append(row)
        print(f"    energy step {m}: s* {s_m:.3g}  change {change:.3e}  "
              f"eta classical {eta_c:.3e}  poseidon eta/true {eta_p:.3e}/{err_p:.3e}  "
              f"persistence eta/true {eta_i:.3e}/{change:.3e}", flush=True)
    art["energy"] = out
    _persist(art)


def stage_selfcons(n, art):
    """One step at 2 dt against two at dt, disks off, from the settled state."""
    M = Maps(n)
    z = _load_ref(n)
    w = (z["u"].copy(), z["v"].copy())
    dt = wa.MACRO_DT
    off = set()

    def pair(fn):
        a1 = fn(w[0].copy(), w[1].copy(), dt, off)
        a11 = fn(a1[0].copy(), a1[1].copy(), dt, off)
        a2 = fn(w[0].copy(), w[1].copy(), 2 * dt, off)
        return a1, a11, a2

    m1, m11, m2 = pair(M.F)
    p1, p11, p2 = pair(M.P)
    scheme_gap = rms_pair(m2[0] - m11[0], m2[1] - m11[1])
    d_sc = rms_pair(p2[0] - p11[0], p2[1] - p11[1])
    e2 = rms_pair(p2[0] - m2[0], p2[1] - m2[1])
    e11 = rms_pair(p11[0] - m11[0], p11[1] - m11[1])
    e1 = rms_pair(p1[0] - m1[0], p1[1] - m1[1])
    change2 = rms_pair(m2[0] - w[0], m2[1] - w[1])
    out = {"rung": f"N{n}", "delta_sc_poseidon": d_sc, "scheme_gap_2dt_vs_2xdt": scheme_gap,
           "error_one_step": e1, "error_two_steps": e11, "error_double_step": e2,
           "lower_bound_holds": bool(d_sc - scheme_gap <= e2 + e11 + 1e-15),
           "lower_bound_tightness": (d_sc - scheme_gap) / max(e2 + e11, 1e-300),
           "identity": {"delta_sc": 0.0, "true_error_double_step": change2,
                        "note": "the identity map is exactly self-consistent and "
                                "wrong by the flow's own change: this certificate "
                                "cannot see a null replacement"}}
    art["selfcons"] = out
    _persist(art)
    print(f"    selfcons: delta_sc {d_sc:.3e}  e1 {e1:.3e}  e11 {e11:.3e}  e2 {e2:.3e}  "
          f"scheme gap {scheme_gap:.3e}  identity true error {change2:.3e}", flush=True)


def stage_finetune(art, batch=2, iters=3):
    import torch                                                         # noqa: PLC0415
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ex = wa.scaled_expert(threads=8)
    model = ex.model
    x = torch.randn(batch, 4, 128, 128)
    t = torch.full((batch,), 0.1)
    with torch.no_grad():
        model(pixel_values=x, time=t)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(pixel_values=x, time=t)
        fwd = (time.perf_counter() - t0) / iters
    for p in model.parameters():
        p.requires_grad_(True)
    try:
        out = model(pixel_values=x, time=t).output
        out.pow(2).mean().backward()
        model.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        for _ in range(iters):
            out = model(pixel_values=x, time=t).output
            out.pow(2).mean().backward()
            model.zero_grad(set_to_none=True)
        fb = (time.perf_counter() - t0) / iters
    finally:
        for p in model.parameters():
            p.requires_grad_(False)
    rec = {"batch": batch, "forward_s": fwd, "forward_backward_s": fb,
           "fb_over_forward": fb / fwd,
           "hours_for_2000_iterations": 2000 * fb / 3600.0,
           "n_params": int(ex.n_params),
           "note": "no optimiser step and no data pipeline: a lower bound on an iteration"}
    art["finetune"] = rec
    _persist(art)
    print(f"    finetune: forward {fwd:.3f}s  forward+backward {fb:.3f}s  "
          f"2000 iterations = {rec['hours_for_2000_iterations']:.2f} h", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages",
                    default="repro,reference,warm,dec,energy,selfcons,finetune")
    ap.add_argument("--rung", type=int, default=2)
    ap.add_argument("--rungs", default="2,6", help="for the cost stage")
    ap.add_argument("--arms", default="C,Z,P,Pc")
    ap.add_argument("--alphas", default="0,0.5,0.8")
    ap.add_argument("--keys", default="",
                    help="explicit arms, e.g. P_a0.5,C_a0 -- replaces arms x alphas")
    ap.add_argument("--marched", default="",
                    help="arms whose certified answer is marched further classically")
    ap.add_argument("--marched-steps", type=int, default=50)
    args = ap.parse_args(argv)
    assert os.environ.get("KMP_DUPLICATE_LIB_OK") == "TRUE"
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    os.makedirs(OUT, exist_ok=True)
    art = _load_art()
    art.setdefault("date", time.strftime("%Y-%m-%d"))
    art["last_run"] = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv[1:]}
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        print(f"== stage {s}", flush=True)
        t0 = time.time()
        if s == "repro":
            stage_repro(args.rung, art)
        elif s == "cost":
            stage_cost([int(x) for x in args.rungs.split(",")], art)
        elif s == "reference":
            stage_reference(args.rung, art)
        elif s == "warm":
            stage_warm(args.rung, art)
        elif s == "dec":
            stage_dec(args.rung, art, arms=tuple(a.strip() for a in args.arms.split(",")),
                      alphas=tuple(float(x) for x in args.alphas.split(",")),
                      keys=[k.strip() for k in args.keys.split(",") if k.strip()] or None,
                      marched=tuple(k.strip() for k in args.marched.split(",") if k.strip()),
                      marched_steps=args.marched_steps)
        elif s == "account":
            stage_account(art)
        elif s == "energy":
            stage_energy(args.rung, art)
        elif s == "selfcons":
            stage_selfcons(args.rung, art)
        elif s == "finetune":
            stage_finetune(art)
        else:
            raise SystemExit(f"unknown stage {s}")
        print(f"== stage {s} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
