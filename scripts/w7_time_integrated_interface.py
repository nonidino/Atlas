"""W7 / W17: the interface condition an EXPLICIT one-step map actually poses.

`tier0-measurements` section 5 measured that flux balance -- ``sum_i Lambda_i
lambda = chi`` -- is a steady condition: substituting the reference's own trace
does not reduce its residual, and solving it exactly makes one composed macro-step
8.9x worse, with the overshoot growing to 150x as dt is refined.  R2b turned that
into a refusal.  **This script builds the positive construction R2b does not
supply**, and puts it through the same two tests the pointwise one failed.

The construction
----------------

An explicit macro-step poses no boundary-value problem, but it does pose a
CONSERVATION statement.  Integrating over a subdomain and over the macro-step,

    int_Omega_i (u^{n+1} - u^n)  =  - int_{t^n}^{t^{n+1}} oint_{dOmega_i} F.n

so what the two sides of a seam must agree on is the TIME-INTEGRATED flux over
the step, not the pointwise flux at its end:

    Gbar(lambda) := int_{t^n}^{t^{n+1}} [ F_A(lambda(t)) + F_B(lambda(t)) ] dt = 0

and the interface datum is a FUNCTION on the step, carried as the W=2 waveform
``lambda(t) = (1-s) lambda^n + s lambda^{n+1}`` that `WindowNS.step_batch` already
accepts through ``bc0``/``bc1``.  The unknown is ``lambda^{n+1}``; the operator is
the time-integrated DtN ``Lambda_i_bar``, probed the same way as ``Lambda_i`` but
with a ramped trace and an integrated response.

The two tests, from section 5
-----------------------------

  A  substitute the monolith's OWN trace.  A condition the true solution
     satisfies must have its residual FALL.  Pointwise: 2.6421e-5 -> 2.6520e-5,
     i.e. it rose.
  B  solve it exactly and measure the composed macro-step.  Pointwise: 8.9x
     WORSE than not solving it at all.

Configuration: the shared-layer tiling (n=128, halo=1) of section 1, because that
is the only one with a genuine single seam trace -- a wide halo has a ring per
window and no shared unknown, which is exactly why R2b holds the rung there.

Run:  python scripts/w7_time_integrated_interface.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from atlas.cases import window_ns as W                             # noqa: E402
from tier0_window_ns import (                                      # noqa: E402
    _pin_domain, body_force, flat_assemble, make_solvers, mono_step, rel_l2,
)

#: The shared-layer tiling: 2*128 - 255 = 1, so the two windows share exactly one
#: cell layer and there is one trace, not two rings.
SHARED = W.Tiling(n=128, ramp=1)

#: seam -> ((agent, face), (agent, face)), same as the case study's.
SEAMS = {
    "sx0": (("W00", "xhi"), ("W10", "xlo")),
    "sx1": (("W01", "xhi"), ("W11", "xlo")),
    "sy0": (("W00", "yhi"), ("W01", "ylo")),
    "sy1": (("W10", "yhi"), ("W11", "ylo")),
}

RING = {"xlo": ((slice(None), 0), (slice(None), 1)),
        "xhi": ((slice(None), -1), (slice(None), -2)),
        "ylo": ((0, slice(None)), (1, slice(None))),
        "yhi": ((-1, slice(None)), (-2, slice(None)))}

#: which velocity component is normal to each face, and the outward sign
NORMAL = {"xlo": ("u", -1.0), "xhi": ("u", +1.0),
          "ylo": ("v", -1.0), "yhi": ("v", +1.0)}


# ---------------------------------------------------------------------------
# one window's macro-step, sub-stepped, with the seam flux recorded throughout
# ---------------------------------------------------------------------------


def window_step(solver, u0, v0, r0, r1, dt, n_sub, faces, fx=None):
    """Advance one window with a ring ramping r0 -> r1, recording seam fluxes.

    Returns ``(u1, v1, fluxes)`` where ``fluxes[face]`` is an ``(n_sub, n)`` array
    of the one-sided outward normal flux ``nu (w_ring - w_interior)/h`` at the end
    of each sub-step.  The POINTWISE operator reads the last row; the
    TIME-INTEGRATED one reads the mean.  Both come from the same run, so the
    comparison costs nothing extra and no convention drifts between them.
    """
    u, v = u0.copy(), v0.copy()
    hs = dt / n_sub
    rec = {f: [] for f in faces}
    force = None if fx is None else (fx[None], np.zeros_like(fx)[None])
    for m in range(n_sub):
        s0, s1 = m / n_sub, (m + 1) / n_sub
        a = ((1 - s0) * r0[0] + s0 * r1[0], (1 - s0) * r0[1] + s0 * r1[1])
        b = ((1 - s1) * r0[0] + s1 * r1[0], (1 - s1) * r0[1] + s1 * r1[1])
        u1, v1 = solver.step_batch(u[None], v[None], hs,
                                   bc0=(a[0][None], a[1][None]),
                                   bc1=(b[0][None], b[1][None]), force=force)
        u, v = u1[0], v1[0]
        for f in faces:
            comp, _sgn = NORMAL[f]
            w = u if comp == "u" else v
            ring, inner = RING[f]
            rec[f].append(W.NU * (w[ring] - w[inner]) / solver.h)
    return u, v, {f: np.array(rec[f]) for f in faces}


def seam_faces(tiling):
    """agent -> the faces of it that are artificial, i.e. carry a seam."""
    out = {}
    for k, (ox, oy) in enumerate(tiling.offsets):
        out[tiling.names[k]] = tiling.artificial_faces(ox, oy)
    return out


# ---------------------------------------------------------------------------
# the residual, in both conventions, from one composed macro-step
# ---------------------------------------------------------------------------


class Interface:
    """The seam basis, the composed step, and the residual in both conventions."""

    def __init__(self, u, v, fx, tiling=SHARED, dt=W.MACRO_DT, n_sub=W.SUBSTEPS,
                 agent="exposed"):
        self.t, self.dt, self.n_sub = tiling, dt, n_sub
        self.u, self.v, self.fx = u, v, fx
        self.agent = agent
        self.solvers = make_solvers(tiling)
        self.mono, emb, exp = self.solvers
        # 'embedded' is the agent section 5 measured with -- R10 refuses it, and
        # the point of running it here is to attribute section 5's 8.9x
        self.win = exp if agent == "exposed" else emb
        self.faces = seam_faces(tiling)
        self.basis = W.fourier_basis(tiling.n, W.M_EFF)          # (n, 16)
        self.u_ref, self.v_ref = mono_step(self.mono, u, v, dt, fx)
        self.calls = 0

    # -- the trace, as a vector of modal coefficients per seam -------------

    def zero(self):
        return {s: np.zeros(W.M_EFF) for s in SEAMS}

    def true_trace(self):
        """The monolith's own normal velocity on each seam, in the modal basis.

        Expressed as a DELTA from the lagged trace, which is what the solve moves.
        """
        out = {}
        for s, ((a_id, a_face), (_b_id, _b_face)) in SEAMS.items():
            k = self.t.names.index(a_id)
            ox, oy = self.t.offsets[k]
            comp, _ = NORMAL[a_face]
            ref = self.u_ref if comp == "u" else self.v_ref
            lag = self.u if comp == "u" else self.v
            if a_face == "xhi":
                col = ox + self.t.n - 1
                d = (ref[oy:oy + self.t.n, col] - lag[oy:oy + self.t.n, col])
            else:                                                 # yhi
                row = oy + self.t.n - 1
                d = (ref[row, ox:ox + self.t.n] - lag[row, ox:ox + self.t.n])
            out[s] = W.H * (self.basis.T @ d)                     # P* = h P^T
        return out

    # -- one composed macro-step under a given trace delta -----------------

    def compose(self, delta):
        """Run all four windows with the ring ramping toward ``delta``.

        Returns the assembled field and, per seam, the two sides' pointwise and
        time-integrated fluxes.
        """
        t = self.t
        us, vs = t.cut(self.u), t.cut(self.v)
        fxs = t.cut(self.fx)
        # the ring perturbation each window sees on each of its seam faces
        pert = {n: {} for n in t.names}
        for s, ((a_id, a_face), (b_id, b_face)) in SEAMS.items():
            d = self.basis @ np.asarray(delta[s], float)          # (n,)
            pert[a_id][a_face] = d
            pert[b_id][b_face] = d

        outs, flux = {}, {}
        for k, name in enumerate(t.names):
            r0u, r0v = us[k].copy(), vs[k].copy()
            r1u, r1v = us[k].copy(), vs[k].copy()
            for f, d in pert[name].items():
                comp, _ = NORMAL[f]
                ring, _ = RING[f]
                tgt = r1u if comp == "u" else r1v
                tgt[ring] = tgt[ring] + d
            u1, v1, fl = window_step(self.win, us[k], vs[k], (r0u, r0v),
                                     (r1u, r1v), self.dt, self.n_sub,
                                     self.faces[name], fxs[k])
            self.calls += 1
            outs[name] = (u1, v1)
            flux[name] = fl

        for s, ((a_id, a_face), (b_id, b_face)) in SEAMS.items():
            fa, fb = flux[a_id][a_face], flux[b_id][b_face]
            # outward normals on the two sides are opposite, so a matched seam
            # has fa + fb = 0 under either convention
            flux[s] = {"pointwise": fa[-1] + fb[-1],
                       "integrated": fa.mean(axis=0) + fb.mean(axis=0)}
        return outs, flux

    def assemble(self, outs):
        t = self.t
        u1 = np.stack([outs[n][0] for n in t.names])
        v1 = np.stack([outs[n][1] for n in t.names])
        U, V = flat_assemble(t, u1, v1)
        if self.agent == "exposed":
            # the composition layer owns the elliptic part (R10)
            pu, pv = self.mono._project(U[None], V[None])
            U, V = pu[0], pv[0]
        return _pin_domain(U, V, (self.u, self.v))

    def residual(self, delta, mode):
        outs, flux = self.compose(delta)
        return {s: W.H * (self.basis.T @ flux[s][mode]) for s in SEAMS}, outs

    # -- the operator, probed in whichever convention ----------------------

    def probe(self, mode, eps=1e-6):
        """S[seam] = d residual[seam] / d delta[seam], on the 16-mode basis."""
        base, _ = self.residual(self.zero(), mode)
        S = {s: np.zeros((W.M_EFF, W.M_EFF)) for s in SEAMS}
        for j in range(W.M_EFF):
            d = self.zero()
            for s in SEAMS:
                d[s][j] = eps
            pert, _ = self.residual(d, mode)
            for s in SEAMS:
                S[s][:, j] = (pert[s] - base[s]) / eps
        return base, S


def norm(d):
    return float(np.sqrt(sum(float(np.dot(x, x)) for x in d.values())))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", default=os.path.join("out", "tier0_verify", "s0_state.npz"))
    ap.add_argument("--out", default=os.path.join("out", "w7"))
    ap.add_argument("--dt", type=float, default=W.MACRO_DT)
    ap.add_argument("--agent", default="exposed", choices=("exposed", "embedded"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    d = np.load(a.state)
    u, v, fx = d["u"], d["v"], d["fx"]
    I = Interface(u, v, fx, dt=a.dt, agent=a.agent)
    print(f"shared-layer tiling n={SHARED.n} halo={SHARED.halo}  dt={a.dt} "
          f"n_sub={I.n_sub}\n")

    lag = I.zero()
    true = I.true_trace()
    print(f"true trace move ||.||_inf = "
          f"{max(float(np.abs(I.basis @ true[s]).max()) for s in SEAMS):.4e}\n")

    result = {"dt": a.dt, "agent": a.agent,
              "tiling": {"n": SHARED.n, "halo": SHARED.halo}}

    for mode in ("pointwise", "integrated"):
        print(f"--- {mode} flux convention " + "-" * 40)
        base, S = I.probe(mode)
        r_true, _ = I.residual(true, mode)

        Ssum = {s: S[s] for s in SEAMS}
        cond = {s: float(np.linalg.cond(Ssum[s])) for s in SEAMS}
        beta = {s: float(np.linalg.svd(Ssum[s], compute_uv=False)[-1]) for s in SEAMS}

        solved = {s: np.linalg.solve(Ssum[s], -base[s]) for s in SEAMS}
        r_solved, _ = I.residual(solved, mode)

        # TEST A: does the reference's own trace reduce the residual?
        # TEST B: what does solving it do to the composed macro-step?
        rows = {}
        for label, delta in (("lagged", lag), ("true", true), ("solved", solved)):
            r, outs = I.residual(delta, mode)
            U, V = I.assemble(outs)
            rows[label] = {
                "residual": norm(r),
                "defect": rel_l2(U, I.u_ref, V, I.v_ref),
                "trace_move_inf": max(float(np.abs(I.basis @ delta[s]).max())
                                      for s in SEAMS),
            }
        base_defect = rows["lagged"]["defect"]
        print(f"{'trace':10s} {'||residual||':>13s} {'vs lagged':>10s} "
              f"{'trace move':>12s} {'step defect':>13s} {'vs lagged':>10s}")
        for label, r in rows.items():
            print(f"{label:10s} {r['residual']:13.4e} "
                  f"{r['residual'] / rows['lagged']['residual']:10.3f} "
                  f"{r['trace_move_inf']:12.4e} {r['defect']:13.4e} "
                  f"{r['defect'] / base_defect:10.3f}")
        print(f"kappa = {[round(cond[s], 2) for s in SEAMS]}")
        print(f"beta  = {['%.3e' % beta[s] for s in SEAMS]}\n")
        result[mode] = {"rows": rows, "kappa": cond, "beta": beta,
                        "residual_at_true_over_lagged":
                            rows["true"]["residual"] / rows["lagged"]["residual"]}

    print(f"solver calls: {I.calls}")
    name = f"w7_{a.agent}_dt{a.dt:g}"
    with open(os.path.join(a.out, name + ".json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=float)
    print(f"wrote {os.path.join(a.out, name + '.json')}")


if __name__ == "__main__":
    main()
