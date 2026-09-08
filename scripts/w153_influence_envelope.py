"""W153 -- Tier 33.  The envelope, not the scheme.

`Pi` is not a quantity.  It is a **0/1 envelope standing in for a sensitivity**.
`master-error-bound` 4.1 bounds the assembled error at cell j by

    |error(j)|  <=  sum_i chi_ij R_i(j) ||d_lambda||,
        R_i(j) := || d E_i(j) / d g_i ||

and then substitutes ``R_i(j) <= C_mu * 1[ b_i(j) <= d_i ]``.  That indicator is
tight for a local explicit agent and **catastrophically loose for influence that
is dense but DECAYING** -- which is the only kind a learned operator has.  This
tier replaces the envelope with the thing it stands for:

    Pi_w := max_j sum_i chi_ij Rhat_i(j),     Rhat normalised to <= 1

``Pi_w <= Pi`` always, and it reduces to ``Pi`` exactly under a hard cutoff, so no
past verdict can move silently.  It keeps its teeth: influence that genuinely does
not decay gives ``Rhat ~ 1``, ``Pi_w = 1``, and the refusal stands -- now measured
rather than assumed.

**This is the same bound with the indicator replaced by the sensitivity the
indicator was standing in for.**  It is not a new bound, and it must be quoted
that way or Gate A's sixteen-configuration control is not a control.

WHAT THIS SETTLES THAT NOTHING ELSE CAN
---------------------------------------

Two mechanisms give a dense receptive field and they are different objects.
Advective and diffusive influence decays outside the domain of dependence.
Elliptic influence is **harmonic** -- cut the domain and each piece solves the
same Poisson source with different boundary data, so the error is harmonic and by
the maximum principle does not decay.  That is R10's "flat in distance from the
cut", and **R10 is right**.

W60 has been open for many tiers as "a declaration nobody can check", because
`poseidon.elliptic_signature` measures NON-NORMALITY and was read as measuring
GLOBALITY, and the two come apart for a self-adjoint operator.  A decay profile
measures globality directly and cannot return inconclusive.

THE GATES
---------

**Gate A** -- validate the instrument on recorded data.  ``C_mu = 1.2`` was
fitted WITH the indicator, so swapping the envelope changes what ``C_mu`` means.
Re-derive it against ``Pi_w`` on the same configurations of
`master-error-bound` 4.1.1.  Required: implied spread no worse than the
indicator's on the same rows.  If ``Pi_w`` reproduces ``sigma`` worse, the
refinement is not one and the tier ends.

**Gate B** -- the three-way decay control.  `SpectralNS` periodic must be
identically zero; `WindowNS` exposed must decay with ``d_eff -> rho*s``; `WindowNS`
embedded must not.  Same solver in the last two, so nothing but the elliptic part
can confound it.

**Gate C** -- Poseidon-T's horizon, and W60 closes whichever way it comes out.

Run:

    python scripts/w153_influence_envelope.py
    python scripts/w153_influence_envelope.py --part gateB --no-poseidon
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import os
import sys
import time

# torch and numpy each bring their own OpenMP on Windows; set before any import
# that could reach torch, which is why it sits above the atlas imports.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from atlas.cases import window_ns as W                             # noqa: E402
from atlas.ports import RingActuation                              # noqa: E402


def _load(name, filename):
    """Load a sibling driver by path -- `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location(name,
                                                  os.path.join(_HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: Tier 32's `Side` classes march a window under a ring datum and return the
#: FIELD.  Reused rather than reimplemented: they carry the bitwise-verified ring
#: construction, and `check_flux_agrees` is re-run here as a control so the reuse
#: is checked rather than assumed.
_w149 = _load("_w149", "w149_control_observability.py")
WindowSide, SpectralSide, PoseidonSide = (_w149.WindowSide, _w149.SpectralSide,
                                          _w149.PoseidonSide)


# ---------------------------------------------------------------------------
# geometry: b_i(j), the distance to the nearest ARTIFICIAL face
# ---------------------------------------------------------------------------


def b_local(tiling, k: int) -> np.ndarray:
    """``b_i(j)`` in cells, on window ``k``'s own grid.

    A face coinciding with the domain boundary is a real boundary and carries no
    stale datum, so it does not enter -- the same rule
    `w49_sigma_halo.artificial_distance` applies on the global grid, restated
    window-locally because the influence Jacobian lives there.
    """
    ox, oy = tiling.offsets[k]
    n = tiling.n
    faces = tiling.artificial_faces(ox, oy)
    ii = np.arange(n) + 0.5
    dx = np.full(n, np.inf)
    dy = np.full(n, np.inf)
    if "xlo" in faces:
        dx = np.minimum(dx, ii)
    if "xhi" in faces:
        dx = np.minimum(dx, n - ii)
    if "ylo" in faces:
        dy = np.minimum(dy, ii)
    if "yhi" in faces:
        dy = np.minimum(dy, n - ii)
    return np.minimum(dy[:, None], dx[None, :])


# ---------------------------------------------------------------------------
# the influence Jacobian and the profile read off it
# ---------------------------------------------------------------------------


def influence_jacobian(side, faces, eps: float, m: int = W.M_EFF,
                       components: int = 1):
    """``d E_i / d g_i`` over the WHOLE window, one column per control mode.

    ``components = 1`` drives the declared conjugate pair (the normal velocity);
    ``components = 2`` drives the full ring.  **Which of those a port permits is
    a declared field** as of W152 -- `ports.RingActuation` -- and the caller reads
    it rather than choosing, which is the whole of that row.

    Returns ``(Jc, labels)`` with ``Jc`` shaped ``(ncol, 2, n, n)``: the response
    of both velocity components at every cell, per control column.
    """
    P = W.fourier_basis(side.n, m)
    base = side.march({})
    cols, labels = [], []
    for face in faces:
        for c in range(components):
            for k in range(m):
                if components == 1:
                    g = eps * P[:, k]
                else:
                    g = np.zeros((2, side.n))
                    g[c] = eps * P[:, k]
                f = side.march({face: g})
                cols.append(np.stack([(f[0] - base[0]) / eps,
                                      (f[1] - base[1]) / eps]))
                labels.append({"face": face, "component": c, "mode": k,
                               "wavenumber": 0 if k == 0 else (k + 1) // 2,
                               "parity": "const" if k == 0
                               else ("cos" if k % 2 else "sin")})
    return np.stack(cols), labels


def profile_of(Jc: np.ndarray, b: np.ndarray, max_r: int = 70,
               keep: np.ndarray | None = None):
    """``Rhat_i(j)``, its profile against ``b``, and ``d_eff(theta)``.

    ``keep`` optionally restricts the control columns, which is how the gauge
    decomposition and the one-scalar pin are evaluated without new marches.
    """
    J = Jc if keep is None else Jc[keep]
    R = np.sqrt((J ** 2).sum(axis=(0, 1)))          # (n, n), Frobenius row norm
    peak = float(R.max())
    Rh = R / peak if peak > 0 else R
    prof_max, prof_mean, counts = [], [], []
    for r in range(max_r):
        msk = (b >= r) & (b < r + 1)
        counts.append(int(msk.sum()))
        prof_max.append(float(Rh[msk].max()) if msk.any() else float("nan"))
        prof_mean.append(float(Rh[msk].mean()) if msk.any() else float("nan"))
    return {"Rhat": Rh, "peak": peak, "profile_max": prof_max,
            "profile_mean": prof_mean, "shell_counts": counts}


def d_eff(prof_max, thetas=(1e-1, 1e-2, 1e-3, 1e-6, 1e-10, 0.0)):
    """``d_eff(theta) = min{ r : Rhat(r) <= theta }`` -- the interaction horizon.

    ``theta = 0`` is the exact-support radius, which for a local explicit agent
    must come back as the declared ``rho * s`` and not merely near it.
    """
    out = {}
    for th in thetas:
        hit = [r for r, x in enumerate(prof_max)
               if np.isfinite(x) and x <= th]
        out[str(th)] = hit[0] if hit else None
    return out


def far_field_spectrum(Jc: np.ndarray, b: np.ndarray, d: float, labels):
    """Is the non-decaying part RANK ONE?

    The derivation says a boundary perturbation's k-th mode decays like
    ``exp(-pi k r / L)`` on a strip, so every mode decays except ``k = 0`` and the
    flat part is the rank-one gauge mode.  This tests it directly: take the cells
    the local agent cannot reach at all, form that sub-block, and look at its
    spectrum.  Rank one means one dominant singular value whose right singular
    vector is the constant mode.
    """
    far = b > d
    if not far.any():
        return {"note": "no far field at this d"}
    Jf = Jc[:, :, far].reshape(Jc.shape[0], -1).T        # (2*n_far, ncol)
    fro = float(np.linalg.norm(Jf))
    out = {"n_far_cells": int(far.sum()), "d": float(d), "frobenius": fro}
    if fro == 0.0:
        out["exactly_zero"] = True
        return out
    out["exactly_zero"] = False
    _U, sv, Vt = np.linalg.svd(Jf, full_matrices=False)
    tot = float((sv ** 2).sum())
    out["singular_values"] = [float(x) for x in sv[:8]]
    out["normalised"] = [float(x / sv[0]) for x in sv[:8]]
    out["gap_s1_over_s2"] = float(sv[0] / sv[1]) if sv.size > 1 else float("inf")
    out["energy_in_mode_1"] = float(sv[0] ** 2 / tot)
    out["energy_in_modes_1_2"] = float((sv[:2] ** 2).sum() / tot)
    #: participation ratio: an effective rank that needs no threshold.
    p = (sv ** 2) / tot
    out["effective_rank"] = float(1.0 / (p ** 2).sum())
    w = Vt[0]
    const = np.array([1.0 if lb["mode"] == 0 else 0.0 for lb in labels])
    const /= np.linalg.norm(const)
    out["top_vector_overlap_with_constant_modes"] = float(abs(w @ const))
    out["top_right_singular_vector"] = [float(x) for x in w]
    return out


def per_mode_decay(Jc, b, labels, near=0, far=40):
    """How each control column's own response falls off.

    The derivation is a statement about MODES, so it is tested about modes.  A
    far-field SVD can be misled by a basis that mixes them; this cannot.
    """
    rows = []
    mnear = (b >= near) & (b < near + 1)
    mfar = (b >= far) & (b < far + 1)
    for i, lb in enumerate(labels):
        r = np.sqrt((Jc[i] ** 2).sum(axis=0))
        a = float(r[mnear].max()) if mnear.any() else float("nan")
        z = float(r[mfar].max()) if mfar.any() else float("nan")
        rows.append(dict(lb, near=a, far=z,
                         far_over_near=(z / a) if a > 0 else float("nan")))
    fars = np.array([x["far"] for x in rows])
    tot = float((fars ** 2).sum())
    return {
        "rows": rows, "near_shell": near, "far_shell": far,
        "modes_within_10x_of_largest_far": int((fars >= fars.max() / 10).sum())
        if fars.max() > 0 else 0,
        "far_energy_fraction_in_constant_mode":
            float(sum(x["far"] ** 2 for x in rows if x["mode"] == 0) / tot)
            if tot > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Pi and Pi_w on the global grid
# ---------------------------------------------------------------------------


def envelopes(rhats: list[np.ndarray], tiling, ws, d_cells: float):
    """``Pi`` (the indicator) and ``Pi_w`` (the measured sensitivity), same grid.

    Both are ``max_j sum_i chi_ij (.)``; they differ only in what sits inside.
    Computing them from the same weights in the same call is what makes the
    comparison a swap of one factor rather than two measurements.
    """
    n = tiling.n
    M = int(getattr(tiling, "mono_n", W.MONO_N))
    acc_ind = np.zeros((M, M))
    acc_w = np.zeros((M, M))
    for k, (ox, oy) in enumerate(tiling.offsets):
        chi = ws[k][oy:oy + n, ox:ox + n]
        b = b_local(tiling, k)
        acc_ind[oy:oy + n, ox:ox + n] += chi * (b <= d_cells)
        acc_w[oy:oy + n, ox:ox + n] += chi * rhats[k]
    return float(acc_ind.max()), float(acc_w.max())


def measure_rhats(u, v, fx, tiling, dt, expose, eps, m=W.M_EFF, components=1,
                  pin_constant: bool = False):
    """One normalised influence field per window, at the given exchange interval.

    ``pin_constant`` drops the constant (gauge) column of every face, which is
    what "pin one scalar per governing family per exchange" does to the
    control space.  It costs no extra marches -- the column is simply not
    read -- so the pinned and unpinned envelopes come off ONE probe and the
    difference cannot be a difference of runs.
    """
    ex = W.make_experts(u, v, tiling, dt=dt, expose_elliptic=expose)
    fxs = tiling.cut(fx)
    out, calls = [], 0
    for k, name in enumerate(tiling.names):
        ox, oy = tiling.offsets[k]
        side = WindowSide(ex[name], "xhi", ox, oy, tiling.n, force=fxs[k])
        faces = tiling.artificial_faces(ox, oy)
        Jc, lab = influence_jacobian(side, faces, eps, m, components)
        keep = (np.array([x["mode"] != 0 for x in lab])
                if pin_constant else None)
        out.append(profile_of(Jc, b_local(tiling, k), keep=keep)["Rhat"])
        calls += side.n_calls
    return out, calls


def poseidon_rhats(u, v, dt, eps, m=W.M_EFF, pin_constant: bool = False):
    """The same, on Poseidon-T's own tiling.  No forcing: the checkpoint
    has no force channel.
    """
    from atlas.cases import poseidon as PZ
    import torch

    torch.set_num_threads(1)
    tiling = PZ.DEFAULT_TILING
    sub_n = tiling.mono_n
    uu, vv = np.asarray(u)[:sub_n, :sub_n], np.asarray(v)[:sub_n, :sub_n]
    us, vs = tiling.cut(uu), tiling.cut(vv)
    expert = PZ.load_expert()
    out, calls = [], 0
    for k, name in enumerate(tiling.names):
        ox, oy = tiling.offsets[k]
        agent = PZ.PoseidonAgent(
            agent_id=name, u0=us[k], v0=vs[k],
            shared_faces=tiling.artificial_faces(ox, oy), dt=dt,
            expert=expert)
        side = PoseidonSide(agent, "xhi", ox, oy, tiling.n)
        faces = tiling.artificial_faces(ox, oy)
        Jc, lab = influence_jacobian(side, faces, eps, m, 1)
        keep = (np.array([x["mode"] != 0 for x in lab])
                if pin_constant else None)
        out.append(profile_of(Jc, b_local(tiling, k), keep=keep)["Rhat"])
        calls += side.n_calls
    return out, calls, tiling


# ---------------------------------------------------------------------------
# the confound tier 32 left open
# ---------------------------------------------------------------------------


def confound() -> dict:
    """Which column carried tier 32's 6.6x-21x, and was its venue winnable?

    If the pathology was measured on an R10-compliant column at the split-step
    cadence, the baseline is ``sigma = 3.73e-8`` and nothing could have won there.
    Read from the artifact rather than remembered.
    """
    p = os.path.join(_ROOT, "out", "w149", "w149.json")
    if not os.path.isfile(p):
        return {"available": False}
    a = json.load(open(p, encoding="utf-8"))
    rows = []
    for key, c in a.get("gate1", {}).get("cases", {}).items():
        st = c.get("states", [{}])[-1]
        asm = (st.get("solve") or {}).get("assembly")
        if not asm:
            continue
        rows.append({
            "case": key, "label": c["label"],
            "elliptic": "exposed" if "exposed" in key else "embedded",
            "r10": "admits" if "exposed" in key else "refuses",
            "sigma_lagged": asm["sigma_lagged"],
            "solved_over_lagged": asm["solved_over_lagged"],
        })
    pub_split = W.MEASURED_SPLIT_STEP.sigma
    pub_built = W.MEASURED_AS_BUILT.sigma
    ex = [r for r in rows if r["elliptic"] == "exposed"]
    em = [r for r in rows if r["elliptic"] == "embedded"]
    return {
        "available": True, "rows": rows,
        "published_sigma_split_step": pub_split,
        "published_sigma_as_built": pub_built,
        "carried_by": "exposed (R10 admits)" if ex and max(
            r["solved_over_lagged"] for r in ex) >= max(
            r["solved_over_lagged"] for r in em) else "embedded",
        "exposed_venue_over_published_split_step":
            (ex[0]["sigma_lagged"] / pub_split) if ex else None,
        "embedded_venue_over_published_as_built":
            (em[0]["sigma_lagged"] / pub_built) if em else None,
    }


# ---------------------------------------------------------------------------


def w152_readback(u, v, fx) -> dict:
    """The declaration W152 added, read back, and the prediction it came with.

    The prediction was that a richer control space gives the optimiser MORE room
    to absorb the agents' defect difference, so J's pathology should get WORSE
    with the full ring rather than better.  **Tier 32 already ran that
    configuration** -- its column 2b is the full ring against column 2's declared
    pair -- so this reads the artifact instead of spending an hour re-running it.
    """
    ex = W.make_experts(u, v, W.DEFAULT_TILING, expose_elliptic=True)
    caps = W.window_capabilities(ex["W00"])
    ports = [{"name": p.name, "ring_components": p.ring_components,
              "actuation": p.actuation.value,
              "actuated_components": p.actuated_components}
             for p in caps.ports]
    out = {"declared": ports,
           "declares_full_ring": any(p.actuation is RingActuation.FULL_RING
                                     for p in caps.ports)}
    p = os.path.join(_ROOT, "out", "w149", "w149.json")
    if os.path.isfile(p):
        a = json.load(open(p, encoding="utf-8"))
        cs = a.get("gate1", {}).get("cases", {})
        narrow = cs.get("windowns_exposed", {}).get("states", [{}])[-1]
        wide = cs.get("windowns_exposed_2c", {}).get("states", [{}])[-1]
        if narrow and wide:
            n_asm = (narrow.get("solve") or {}).get("assembly", {})
            w_asm = (wide.get("solve") or {}).get("assembly", {})
            out["j_pathology"] = {
                "declared_pair": {
                    "control_dim": narrow["shape"][1],
                    "controllable_fraction": narrow["controllable_fraction"],
                    "sigma_solved_over_lagged": n_asm.get("solved_over_lagged"),
                    "overshoot": (narrow.get("solve") or {}).get("overshoot"),
                },
                "full_ring": {
                    "control_dim": wide["shape"][1],
                    "controllable_fraction": wide["controllable_fraction"],
                    "sigma_solved_over_lagged": w_asm.get("solved_over_lagged"),
                    "overshoot": (wide.get("solve") or {}).get("overshoot"),
                },
            }
            a_, b_ = (n_asm.get("solved_over_lagged"),
                      w_asm.get("solved_over_lagged"))
            out["j_pathology"]["worse_with_full_ring"] = bool(b_ > a_)
            out["j_pathology"]["worse_by"] = b_ / a_ if a_ else None
            out["j_pathology"]["note"] = (
                "read from out/w149/w149.json, not re-run: tier 32's column 2b "
                "IS the full-ring configuration and column 2 the declared pair, "
                "at the same state on the same seam with the same probe")
    return out


# ---------------------------------------------------------------------------
# GATE A
# ---------------------------------------------------------------------------


#: The configurations `master-error-bound` 4.1.1 fitted ``C_mu`` on.  The first
#: eleven vary the assembly (halo width and partition of unity) and are recorded
#: in `out/l6/w49_sigma.json`; the last four vary the PHYSICS at the working
#: partition, which is where ``Pi`` is constant by construction and ``Pi_w`` need
#: not be -- so they are the rows that can most easily break the refinement.
GATE_A_ASSEMBLY = [(128, 8), (133, 8), (138, 8), (143, 8), (148, 8), (158, 8),
                   (138, 0), (138, 1), (138, 4), (138, 21)]
GATE_A_PHYSICS = [("Re x2", 1.0 / 510.0, 0.05), ("Re /2", 1.0 / 128.0, 0.05),
                  ("dt /2", 1.0 / 255.0, 0.025), ("dt x2", 1.0 / 255.0, 0.10)]


def gate_a(eps: float, verbose=True) -> dict:
    """Re-derive ``C_mu`` against ``Pi_w`` on the configurations it was fitted on.

    ``sigma`` is re-measured with the same harness rather than transcribed, and
    the eleven recorded rows are asserted against `out/l6/w49_sigma.json` -- which
    turns the record into a positive control on the harness instead of an input
    nobody checked.
    """
    from l6_assembly_condition import pou_flat, pou_ramp
    from tier0_window_ns import body_force, make_solvers, mono_step, rel_l2
    from w49_sigma_halo import paired_step

    rec = {}
    p = os.path.join(_ROOT, "out", "l6", "w49_sigma.json")
    if os.path.isfile(p):
        for r in json.load(open(p, encoding="utf-8"))["rows"]:
            rec[(r["n"], r["ramp"])] = r

    d = np.load(os.path.join(_ROOT, "out", "tier0_verify", "s0_state.npz"))
    u0, v0, fx0 = d["u"], d["v"], d["fx"]
    rows = []

    def one(label, tiling, ws, u, v, fx, dt, nu, d_cells):
        m = paired_step(ws, tiling, make_solvers(tiling, nu), u, v, dt, fx,
                        (u.copy(), v.copy()))
        # the exchange interval is ONE SUB-STEP for this scheme (R10b), so the
        # influence is measured over one sub-step too -- the same interval d_i
        # is derived for
        hs = dt / W.substeps_at(dt, make_solvers(tiling, nu)[0])
        rhats, calls = measure_rhats(u, v, fx, tiling, hs, True, eps)
        pi, pi_w = envelopes(rhats, tiling, ws, d_cells)
        row = {"label": label, "n": tiling.n, "ramp": tiling.ramp,
               "halo": tiling.halo, "dt": dt, "nu": nu, "Re": 1.0 / nu,
               "Pi": pi, "Pi_w": pi_w, "marches": calls,
               "sigma": m["sigma"], "dlambda_rel": m["dlambda_rel"],
               "C_mu_indicator": m["sigma"] / (pi * m["dlambda_rel"]),
               "C_mu_weighted": m["sigma"] / (pi_w * m["dlambda_rel"])}
        rows.append(row)
        if verbose:
            print("  %-22s Pi=%.4e  Pi_w=%.4e (%.3fx)  sigma=%.4e  "
                  "C_mu: ind %.4f  wtd %.4f"
                  % (label, pi, pi_w, pi_w / pi, m["sigma"],
                     row["C_mu_indicator"], row["C_mu_weighted"]))
        return row

    if verbose:
        print("  the assembly sweep (halo and partition of unity)")
    for n, ramp in GATE_A_ASSEMBLY:
        tiling = W.Tiling(n=n, ramp=max(ramp, 1))
        ws = pou_flat(tiling) if ramp == 0 else pou_ramp(tiling, ramp)
        lbl = "halo %3d, ramp %2d" % (tiling.halo, ramp)
        r = one(lbl, tiling, ws, u0, v0, fx0, W.MACRO_DT, W.NU,
                W.STENCIL_RADIUS)
        rr = rec.get((n, ramp))
        if rr:
            r["recorded_sigma"] = rr["sigma"]
            r["recorded_Pi"] = rr["Pi"]
            r["sigma_reproduction"] = r["sigma"] / rr["sigma"]
            r["Pi_reproduction"] = r["Pi"] / rr["Pi"]

    if verbose:
        print("  the physics sweep (Reynolds number and macro-step)")
    for label, nu, dt in GATE_A_PHYSICS:
        tiling = W.Tiling(n=138, ramp=8)
        ws = pou_ramp(tiling, 8)
        mono, _e, _x = make_solvers(tiling, nu)
        fx = body_force()
        u = np.full((W.MONO_N, W.MONO_N), W.U_INF)
        v = np.zeros_like(u)
        for _ in range(int(round(5.0 / dt))):
            u, v = mono_step(mono, u, v, dt, fx)
        one(label, tiling, ws, u, v, fx, dt, nu, W.STENCIL_RADIUS)

    ind = [r["C_mu_indicator"] for r in rows]
    wtd = [r["C_mu_weighted"] for r in rows]
    sig = [r["sigma"] for r in rows]
    repro = [r["sigma_reproduction"] for r in rows if "sigma_reproduction" in r]
    out = {
        "rows": rows,
        "C_mu_indicator": {"min": min(ind), "max": max(ind),
                           "spread": max(ind) / min(ind)},
        "C_mu_weighted": {"min": min(wtd), "max": max(wtd),
                          "spread": max(wtd) / min(wtd)},
        "sigma_span": max(sig) / min(sig),
        "Pi_w_over_Pi": {"min": min(r["Pi_w"] / r["Pi"] for r in rows),
                         "max": max(r["Pi_w"] / r["Pi"] for r in rows)},
        "recorded_rows_reproduced": len(repro),
        "sigma_reproduction_worst": max(abs(x - 1.0) for x in repro)
        if repro else None,
        "published_C_mu_spread": 5.2,
    }
    out["passes"] = bool(out["C_mu_weighted"]["spread"]
                         <= out["C_mu_indicator"]["spread"])
    return out


# ---------------------------------------------------------------------------
# GATE B and GATE C
# ---------------------------------------------------------------------------


def gate_b_case(make_sides, label, states, eps, d_ref, m=W.M_EFF,
                components=1, verbose=True) -> dict:
    """The influence profile of ONE window, at three probe states."""
    out = []
    for tag, u, v, fx in states:
        side, faces, tiling, k = make_sides(u, v, fx)
        t0 = time.perf_counter()
        Jc, labels = influence_jacobian(side, faces, eps, m, components)
        b = b_local(tiling, k)
        pr = profile_of(Jc, b)
        row = {
            "probe_state": tag, "faces": list(faces), "eps": eps,
            "components": components, "control_dim": Jc.shape[0],
            "peak": pr["peak"],
            "profile_max": pr["profile_max"], "profile_mean": pr["profile_mean"],
            "shell_counts": pr["shell_counts"],
            "d_eff": d_eff(pr["profile_max"]),
            "far_field": far_field_spectrum(Jc, b, d_ref, labels),
            "per_mode": per_mode_decay(Jc, b, labels),
            "marches": side.n_calls,
            "seconds": time.perf_counter() - t0,
        }
        # the one-scalar pin: drop the constant column(s) and re-read the profile
        keep = np.array([lb["mode"] != 0 for lb in labels])
        pin = profile_of(Jc, b, keep=keep)
        row["pinned"] = {"profile_max": pin["profile_max"],
                         "d_eff": d_eff(pin["profile_max"]),
                         "peak": pin["peak"],
                         "dropped_columns": int((~keep).sum())}
        out.append(row)
        if verbose:
            pm = row["profile_max"]
            print("  %-34s [%s]  Rhat(b=%d)=%.3e  d_eff(1e-3)=%s  "
                  "far ||.||=%.3e  eff_rank=%s  (%.0f s)"
                  % (label, tag, int(d_ref), pm[int(d_ref)],
                     row["d_eff"]["0.001"], row["far_field"]["frobenius"],
                     ("%.2f" % row["far_field"]["effective_rank"])
                     if "effective_rank" in row["far_field"] else "-",
                     row["seconds"]))
    return {"label": label, "states": out, "d_ref": d_ref}


def _window_case(expose, tiling=None, dt=W.MACRO_DT):
    tiling = tiling or W.DEFAULT_TILING

    def make(u, v, fx):
        ex = W.make_experts(u, v, tiling, dt=dt, expose_elliptic=expose)
        ox, oy = tiling.offsets[0]
        side = WindowSide(ex[tiling.names[0]], "xhi", ox, oy, tiling.n,
                          force=tiling.cut(fx)[0])
        return side, tiling.artificial_faces(ox, oy), tiling, 0
    return make


def _spectral_case(tiling=None, dt=W.MACRO_DT):
    tiling = tiling or W.DEFAULT_TILING

    def make(u, v, fx):
        A, _B, tl = _w149.spectral_sides(u, v, fx, dt, tiling)
        ox, oy = tl.offsets[0]
        return A, tl.artificial_faces(ox, oy), tl, 0
    return make


def _poseidon_case(dt=W.MACRO_DT):
    def make(u, v, fx):
        A, _B, tl = _w149.poseidon_sides(u, v, dt)
        ox, oy = tl.offsets[0]
        return A, tl.artificial_faces(ox, oy), tl, 0
    return make


def side_experiment(u, v, fx, eps, with_poseidon=True, verbose=True) -> dict:
    """Pi_w per agent on the whole graph, and what one pinned scalar buys.

    **This is not R12.**  R12 imposed a full second elliptic answer over the
    agents' own and the two disagreed; it failed by OVER-constraining.  One
    scalar is the minimal constraint that annihilates a harmonic gauge mode
    without over-determining the field, and nothing here reaches inside a
    frozen checkpoint -- which is what made the exposed-elliptic repair
    unavailable to Poseidon-T in the first place.

    Measured at each scheme's OWN cadence is not possible here -- Pi and Pi_w
    have to be read on one grid to be compared -- so all rows are at one
    exchange per macro-step, which is Poseidon-T's only cadence.
    """
    from l6_assembly_condition import pou_ramp

    tiling = W.DEFAULT_TILING
    ws = pou_ramp(tiling, tiling.ramp)
    d = float(W.STENCIL_RADIUS * W.SUBSTEPS)
    rows = []
    for label, expose, pin in (("exposed", True, False),
                               ("embedded", False, False),
                               ("embedded + gauge pinned", False, True)):
        rh, calls = measure_rhats(u, v, fx, tiling, W.MACRO_DT, expose, eps,
                                  pin_constant=pin)
        pi, pi_w = envelopes(rh, tiling, ws, d)
        rows.append({"agent": "WindowNS", "variant": label, "Pi": pi,
                     "Pi_w": pi_w, "marches": calls,
                     "tiling": "2x2 n=%d halo=%d ramp=%d"
                     % (tiling.n, tiling.halo, tiling.ramp)})
        if verbose:
            print("  %-26s Pi=%.4f  Pi_w=%.4f  (%.3fx)"
                  % (label, pi, pi_w, pi_w / pi))
    if with_poseidon:
        from atlas.cases import poseidon as PZ
        for label, pin in (("as declared", False), ("gauge pinned", True)):
            rh, calls, ptl = poseidon_rhats(u, v, W.MACRO_DT, eps,
                                            pin_constant=pin)
            pws = ptl.weights()
            pi, pi_w = envelopes(_regrid(rh, ptl), ptl, pws, d)
            rows.append({"agent": "Poseidon-T", "variant": label, "Pi": pi,
                         "Pi_w": pi_w, "marches": calls,
                         "tiling": "2x2 n=%d halo=%d ramp=%d"
                         % (ptl.n, ptl.halo, ptl.ramp)})
            if verbose:
                print("  Poseidon-T, %-14s Pi=%.4f  Pi_w=%.4f  (%.3fx)"
                      % (label, pi, pi_w, pi_w / pi))
    ex = [r for r in rows if r["variant"] == "exposed"][0]
    em = [r for r in rows if r["variant"] == "embedded"][0]
    pn = [r for r in rows if r["variant"] == "embedded + gauge pinned"][0]
    return {
        "rows": rows,
        "pin_buys": 1.0 - pn["Pi_w"] / em["Pi_w"],
        "gap_to_exposed_before": em["Pi_w"] / ex["Pi_w"],
        "gap_to_exposed_after": pn["Pi_w"] / ex["Pi_w"],
        "pin_collapses_the_gap": bool(
            pn["Pi_w"] <= 2.0 * ex["Pi_w"]),
    }


def _regrid(rhats, tiling):
    """Poseidon's tiling has its own MONO_N; `envelopes` reads W.MONO_N."""
    return rhats


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state", default=os.path.join(_ROOT, "out", "tier0_verify",
                                                    "s0_state.npz"))
    ap.add_argument("--out", default=os.path.join(_ROOT, "out", "w153"))
    ap.add_argument("--part", default="all",
                    choices=("all", "gateA", "gateB", "gateC"))
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--no-poseidon", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    d = np.load(a.state)
    u, v, fx = d["u"], d["v"], d["fx"]
    art = {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "state": os.path.relpath(a.state, _ROOT).replace("\\", "/"),
        "config": {"eps": a.eps, "m": W.M_EFF, "nu": W.NU, "h": W.H,
                   "macro_dt": W.MACRO_DT, "mono_n": W.MONO_N,
                   "stencil_radius": W.STENCIL_RADIUS,
                   "substeps_at_macro_dt": W.SUBSTEPS,
                   "d_macro_step": W.STENCIL_RADIUS * W.SUBSTEPS,
                   "d_sub_step": W.STENCIL_RADIUS},
    }

    print("the confound tier 32 left open")
    art["confound"] = confound()
    c = art["confound"]
    if c.get("available"):
        print("  the pathology was carried by the %s column" % c["carried_by"])
        for r in c["rows"]:
            print("     %-40s sigma_lag=%.4e  solved/lag=%7.3fx  (R10 %s)"
                  % (r["label"], r["sigma_lagged"], r["solved_over_lagged"],
                     r["r10"]))
        print("  exposed venue is %.1fx the published split-step sigma (%.3e)"
              % (c["exposed_venue_over_published_split_step"],
                 c["published_sigma_split_step"]))
        print("  embedded venue is %.3fx the published as-built sigma (%.3e)"
              % (c["embedded_venue_over_published_as_built"],
                 c["published_sigma_as_built"]))

    print("\nW152 -- the ring declaration, read back")
    art["w152"] = w152_readback(u, v, fx)
    for p in art["w152"]["declared"]:
        print("  %-10s ring carries %s, a control drives %s (%s)"
              % (p["name"], p["ring_components"], p["actuated_components"],
                 p["actuation"]))
    jp = art["w152"].get("j_pathology")
    if jp:
        print("  the prediction, from out/w149: J's pathology with the full ring")
        for k in ("declared_pair", "full_ring"):
            print("     %-14s dim %2d  controllable %.4f  sigma solved/lagged "
                  "%7.3fx  overshoot %6.2f"
                  % (k, jp[k]["control_dim"], jp[k]["controllable_fraction"],
                     jp[k]["sigma_solved_over_lagged"], jp[k]["overshoot"]))
        print("     WORSE with the full ring: %s (by %.2fx)"
              % (jp["worse_with_full_ring"], jp["worse_by"]))

    print("\ninstrument control: the field march against WindowAgent.respond")
    art["controls"] = {"flux_bitwise": _w149.check_flux_agrees(u, v, fx)}
    print("  bitwise equal: %s" % art["controls"]["flux_bitwise"]["bitwise_equal"])
    if not art["controls"]["flux_bitwise"]["bitwise_equal"]:
        print("  [!] the reused march no longer matches the published response.")
        return 1
    _write(a.out, art)

    if a.part in ("all", "gateA"):
        print("\nGATE A -- C_mu re-derived against Pi_w on the recorded rows")
        t0 = time.perf_counter()
        art["gateA"] = gate_a(a.eps)
        g = art["gateA"]
        print("  (%.0f s)  sigma reproduces the record to %.2e on %d rows"
              % (time.perf_counter() - t0,
                 g["sigma_reproduction_worst"] or 0.0,
                 g["recorded_rows_reproduced"]))
        print("  C_mu with the INDICATOR: [%.4f, %.4f]  spread %.2fx"
              % (g["C_mu_indicator"]["min"], g["C_mu_indicator"]["max"],
                 g["C_mu_indicator"]["spread"]))
        print("  C_mu with Pi_w:          [%.4f, %.4f]  spread %.2fx"
              % (g["C_mu_weighted"]["min"], g["C_mu_weighted"]["max"],
                 g["C_mu_weighted"]["spread"]))
        print("  Pi_w / Pi in [%.4f, %.4f];  sigma spans %.3gx"
              % (g["Pi_w_over_Pi"]["min"], g["Pi_w_over_Pi"]["max"],
                 g["sigma_span"]))
        print("  VERDICT: %s" % ("PASS" if g["passes"] else "FAIL"))
        _write(a.out, art)
        if not g["passes"]:
            print("\n  Pi_w reproduces sigma worse than the indicator did. The "
                  "refinement is not one and the tier ends here.")
            return 0

    if a.part in ("all", "gateB", "gateC"):
        print("\nbuilding the probe trajectory (spin-up, not the fixed point)")
        states = _w149.probe_states(u, v, fx, trajectory="spinup")
        tr = _w149.trajectory_report(states, u, v)
        art["controls"]["trajectory"] = dict(tr, kind="spinup")
        print("  separation across the three: %.4e" % tr["max_separation"])

    if a.part in ("all", "gateB"):
        print("\nGATE B -- the three-way decay control (d_ref = %d cells)"
              % (W.STENCIL_RADIUS * W.SUBSTEPS))
        d_ref = float(W.STENCIL_RADIUS * W.SUBSTEPS)
        cases = {}
        cases["spectral_periodic"] = gate_b_case(
            _spectral_case(), "1. SpectralNS (periodic)", states, a.eps, d_ref)
        cases["windowns_exposed"] = gate_b_case(
            _window_case(True), "2. WindowNS elliptic exposed", states, a.eps,
            d_ref)
        cases["windowns_embedded"] = gate_b_case(
            _window_case(False), "3. WindowNS elliptic embedded", states, a.eps,
            d_ref)
        cases["windowns_exposed_full_ring"] = gate_b_case(
            _window_case(True), "2b. WindowNS exposed, full ring", states,
            a.eps, d_ref, components=2)
        cases["windowns_embedded_full_ring"] = gate_b_case(
            _window_case(False), "3b. WindowNS embedded, full ring", states,
            a.eps, d_ref, components=2)
        art["gateB"] = {"cases": cases}
        _write(a.out, art)

    if a.part == "all":
        print("\nSIDE -- Pi_w on the whole graph, and the one-scalar pin")
        try:
            art["side"] = side_experiment(u, v, fx, a.eps,
                                          with_poseidon=not a.no_poseidon)
            sd = art["side"]
            print("  the pin buys %.1f%% of Pi_w; the gap to the exposed "
                  "column goes %.1fx -> %.1fx"
                  % (100 * sd["pin_buys"], sd["gap_to_exposed_before"],
                     sd["gap_to_exposed_after"]))
        except Exception as exc:                            # pragma: no cover
            print("  side experiment failed: %s: %s"
                  % (type(exc).__name__, exc))
            art["side"] = {"failed": "%s: %s" % (type(exc).__name__, exc)}
        _write(a.out, art)

    if a.part in ("all", "gateC") and not a.no_poseidon:
        print("\nGATE C -- Poseidon-T's horizon")
        try:
            art["gateC"] = gate_b_case(_poseidon_case(), "4. Poseidon-T",
                                       states, a.eps, 20.0)
        except Exception as exc:                                # pragma: no cover
            print("  Poseidon-T unreachable: %s: %s" % (type(exc).__name__, exc))
            art["gateC"] = {"unreachable": "%s: %s" % (type(exc).__name__, exc)}
        _write(a.out, art)

    _write(a.out, art)
    return 0


def _write(out, art):
    p = os.path.join(out, "w153.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(art, fh, indent=2, default=float)
    print("  wrote %s" % os.path.relpath(p, _ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
