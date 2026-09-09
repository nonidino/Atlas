"""W159: measure `cut_defect_bound` on the FRONT-WING graph, for `L2/C2`.

Tier 34 measured `L`, `sigma` and `C_mu` here and left five named rules standing
on the fixed-shape column.  Tier 35 closed two of them as checker defects (W136,
W138).  This is the third, and it is the only one of the five that was a missing
**measurement** rather than a defect or a named hole.  `L2/C2` states the
quantity in its own decertification::

    the graph declares no measured cut_defect_bound, so the criterion is a
    theorem with no value in it.  Measure ||sum_i chi_i |E_i R_i u - R_i E u|||
    over one exchange interval and declare it on MeasuredConstants.

What is measured
----------------

``D_i = E_i R_i u - R_i E u`` is the failure of the exact one-exchange-interval
operator to commute with restriction to window ``i``.  ``E`` exists only where a
monolith does, and this graph has one: `solver_for` builds the SAME class at
``80x80`` and at ``208x144``, so ``E`` is the identical code path over one
window covering the whole domain -- which is what makes the zero-cut control
below bitwise rather than approximate.

Both operators are the RAW solver step over ``dt_ex``.  Neither carries the
blend or the projection, because the identity the bound rests on is

    A({E_i R_i u}) - E u  =  sum_i R_i^T chi_i D_i

with ``A`` the assembly, so the projection sits OUTSIDE ``A`` and belongs to
neither side of it.  Putting it inside ``E`` and not inside ``E_i`` would
measure the projection.

**W58 is respected and it is the reason this driver runs a monolith at all.**
``cut_defect_bound`` has four named readings in `graph.CUT_DEFECT_FORMS` and two
of them are inequivalent quantities rather than two aggregations of one.  What
is declared is the **chi-weighted** form -- L2/C2's own quantity, the one the
derivation bounds the composed defect by -- and `MeasuredConstants` carries
`cut_defect_bound_form` saying so.  The two reference-free surrogates are
measured beside it and reported, because their ratio to the bound is the whole
content of W58, but they are NOT what is declared: this tiling has six windows
and seven overlapping pairs, so the pairwise form is an under-estimate for a
reason unrelated to disjointness, and the compile would decertify it.

The three controls, and each one can fail
-----------------------------------------

**1. The identity, evaluated rather than cited.**  ``sum_i R_i^T chi_i D_i``
with SIGNED ``D_i`` must equal ``A({E_i R_i u}) - E u`` to round-off.  This is
the control on the whole construction -- the lift, the weights, the cut and the
forcing all have to be right for it to hold, and any one of them being wrong
moves it far above round-off.  A bound whose identity does not close is a norm
of the wrong array.

**2. The zero-cut control, marched as far as the thing it controls.**  On
`SINGLE_TILING` there is one window covering the domain, ``chi == 1``, and
``E_1 R_1 u`` is `E u` through the same solver at the same size, so every
``D_i`` must be **exactly** zero -- not small.  It is run through the identical
harness over the identical number of exchanges, because a control that stops
short of the thing it controls has not controlled it.

**3. The floor.**  The same six-window column is run twice and differenced.  A
flat ratio is a floor and not a null, so the bound is compared against the level
at which two identical runs already disagree before it is believed.

Run::

    python scripts/w159_frontwing_cut_defect.py
    python scripts/w159_frontwing_cut_defect.py --steps 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                                       # noqa: E402

from atlas.cases import front_wing as F                            # noqa: E402
from atlas.cases.ground_effect import solver_for                   # noqa: E402
from atlas.cases.wing_fsi import projected_assembly                # noqa: E402
from atlas.probe import (aggregated_neighbour_disagreement,        # noqa: E402
                         restriction_defect_bound)


# ---------------------------------------------------------------------------
# the instrumented rollout: the monolith taken beside every exchange
# ---------------------------------------------------------------------------


class _CutProbe(F.FrontWingRollout):
    """`FrontWingRollout` with ``E`` evaluated alongside every ``E_i``.

    `exchange` is the parent's, copied rather than called, with one block added
    between the local solve and the blend.  The monolithic step is taken from
    the SAME ``(u, v)`` and the SAME global forcing the local solves were handed,
    so the only difference between the two sides is the cut.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.mono = solver_for(self.tiling.nx, self.tiling.ny, self.nu,
                               self.device)
        self.part = projected_assembly(self.tiling).partition
        self.rows: list[dict] = []
        self._ex_i = 0
        ng = self.nx * self.ny
        self._ng = ng
        self._gidx = {}
        self._wts = {}
        for name in self.tiling.names:
            idx = np.asarray(self.part.indices[name])
            self._gidx[name] = np.concatenate([idx, ng + idx])
            w = np.asarray(self.part.weights[name], dtype=float)
            g = np.zeros(2 * ng)
            g[self._gidx[name]] = np.concatenate([w, w])
            self._wts[name] = g
        #: the overlap masks, on the global index space, for the two
        #: reference-free surrogates
        support = {}
        for name in self.tiling.names:
            m = np.zeros(ng, dtype=bool)
            w = np.asarray(self.part.weights[name], dtype=float)
            m[np.asarray(self.part.indices[name])[w > 0.0]] = True
            support[name] = m
        self._overlaps = {}
        names = self.tiling.names
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                m = support[a] & support[b]
                if m.any():
                    self._overlaps[(a, b)] = np.concatenate([m, m])

    # -- the parent's exchange, with the measurement inside it -------------

    def exchange(self, u, v, delta, h, w_plate):
        i = self._ex_i
        self._ex_i += 1
        fx, fy, w, load = self.wing.forcing(u, v, delta, w_plate, self.ny,
                                            self.nx, h=h)
        us, vs = self.cut(u), self.cut(v)
        fxs, fys = self.cut(fx), self.cut(fy)
        u1, v1 = self.solver.step_batch(us, vs, self.dt_ex, bc0=None, bc1=None,
                                        force=(fxs, fys))
        self.substep_log.append(int(self.solver.last_substeps))
        au, av = self.blend(u1, v1)

        #: E: the same class, the same interval, the same forcing, one window
        mu, mv = self.mono.step_batch(u[None], v[None], self.dt_ex, bc0=None,
                                      bc1=None, force=(fx[None], fy[None]))
        self.rows.append(self._measure(i, u1, v1, mu[0], mv[0], au, av))

        au, av = self.project(au, av)
        bu, bv = self.band(au, av)
        f = self.wing.normal_traction(w)
        drag = (f * self.n_hat_x * self.ds).sum()
        return bu, bv, load, f, drag

    # -- the measurement ---------------------------------------------------

    def _measure(self, index, u1, v1, mono_u, mono_v, blend_u, blend_v) -> dict:
        ng = self._ng
        ref_u = self.tiling.cut(mono_u)
        ref_v = self.tiling.cut(mono_v)
        steps, refs = {}, {}
        signed = np.zeros(2 * ng)
        for k, name in enumerate(self.tiling.names):
            gidx = self._gidx[name]
            s_ = np.zeros(2 * ng)
            r_ = np.zeros(2 * ng)
            s_[gidx] = np.concatenate([
                _np(u1[k]).reshape(-1), _np(v1[k]).reshape(-1)])
            r_[gidx] = np.concatenate([
                _np(ref_u[k]).reshape(-1), _np(ref_v[k]).reshape(-1)])
            steps[name], refs[name] = s_, r_
            signed += self._wts[name] * (s_ - r_)

        cut = restriction_defect_bound(steps, refs, self._wts,
                                       n_global=2 * ng)
        nd = aggregated_neighbour_disagreement(steps, self._overlaps,
                                               n_global=2 * ng)

        #: the identity, evaluated: A({E_i R_i u}) - E u against sum R_i^T chi_i D_i
        composed = np.concatenate([
            (_np(blend_u) - _np(mono_u)).reshape(-1),
            (_np(blend_v) - _np(mono_v)).reshape(-1)])
        comp_norm = float(np.linalg.norm(composed))
        resid = float(np.linalg.norm(signed - composed))
        mono_norm = float(np.linalg.norm(np.concatenate([
            _np(mono_u).reshape(-1), _np(mono_v).reshape(-1)])))

        return {
            "exchange": index,
            "chi_weighted": cut["cut_defect_bound_chi_weighted"],
            "max": cut["cut_defect_bound_max"],
            "worst_agent": cut["worst_agent"],
            "neighbour_disagreement_aggregated": nd["aggregated"],
            "neighbour_disagreement_pairwise": nd["pairwise_max"],
            "n_pairs": nd["n_pairs"],
            "composed_defect": comp_norm,
            "monolith_norm": mono_norm,
            "identity_residual": resid,
            "identity_residual_rel": (resid / comp_norm) if comp_norm else 0.0,
            "tightness_chi_over_composed": (
                cut["cut_defect_bound_chi_weighted"] / comp_norm
                if comp_norm > 0 else None),
            "tightness_max_over_composed": (
                cut["cut_defect_bound_max"] / comp_norm
                if comp_norm > 0 else None),
            "aggregated_over_max": (
                nd["aggregated"] / cut["cut_defect_bound_max"]
                if cut["cut_defect_bound_max"] > 0 else None),
            "pairwise_over_aggregated": nd["pairwise_over_aggregated"],
        }


def _np(x):
    return np.asarray(x.detach().cpu() if torch.is_tensor(x) else x,
                      dtype=float)


# ---------------------------------------------------------------------------
# state and marching -- w157's harness, so the two drivers share a start
# ---------------------------------------------------------------------------


def settled_state(root: str):
    p = os.path.join(root, "out", "w141", "settled.npz")
    if os.path.isfile(p):
        d = np.load(p)
        return d["u"], d["v"], "out/w141/settled.npz"
    return (np.full((F.NY, F.NX), F.U_INF), np.zeros((F.NY, F.NX)),
            "freestream (out/w141/settled.npz absent)")


def march(tiling, motion: bool, u, v, steps: int) -> _CutProbe:
    ro = _CutProbe(tiling=tiling, coupling="tight", motion=motion,
                   design=dict(F.DESIGN_REF))
    opt = ro._opt
    kn = ro.knobs(dict(F.DESIGN_REF))
    e_star, tc, k, h0 = kn["e_star"], kn["tc"], kn["k"], kn["h0"]
    st = dict(u=torch.as_tensor(np.asarray(u), **opt),
              v=torch.as_tensor(np.asarray(v), **opt),
              delta=torch.zeros(F.N_STATION, **opt),
              h=ro.release_height(h0, k),
              w_delta=torch.zeros(F.N_STATION, **opt),
              v_mount=torch.zeros((), **opt))
    for n in range(steps):
        ro._ex_i = 0
        out = ro.macro_step(st["u"], st["v"], st["delta"], st["h"],
                            st["w_delta"], st["v_mount"],
                            e_star, tc, k, h0, n)
        (uu, vv, delta, h, w_delta, v_mount, *_rest) = out
        st = dict(u=uu, v=vv, delta=delta, h=h,
                  w_delta=w_delta, v_mount=v_mount)
    ro.final = st
    return ro


def summarise(rows: list[dict]) -> dict:
    def m(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return {"max": max(vals), "min": min(vals),
                "mean": float(np.mean(vals)), "spread": (
                    max(vals) / min(vals) if min(vals) > 0 else None)}
    return {
        "n_exchanges": len(rows),
        "chi_weighted": m("chi_weighted"),
        "max_form": m("max"),
        "composed_defect": m("composed_defect"),
        "identity_residual_rel": m("identity_residual_rel"),
        "tightness_chi_over_composed": m("tightness_chi_over_composed"),
        "aggregated_over_max": m("aggregated_over_max"),
        "pairwise_over_aggregated": m("pairwise_over_aggregated"),
    }


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join("out", "w159"))
    ap.add_argument("--steps", type=int, default=3,
                    help="macro-steps to march; the bound is the max over "
                         "every exchange of every one of them")
    ap.add_argument("--columns", nargs="*", default=["static", "moving"],
                    choices=["static", "moving"])
    a = ap.parse_args(argv)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(1)

    u, v, state = settled_state(root)
    six, one = F.DEFAULT_TILING, F.SINGLE_TILING

    print(f"state      {state}")
    print(f"tiling     {six.n_col}x{six.n_row} windows {six.wx}x{six.wy} on "
          f"{six.nx}x{six.ny}, halo {six.halo}, ramp {six.ramp}")
    print(f"exchanges  {F.EXCHANGES} per macro-step, {a.steps} macro-steps\n")

    #: **W58's hypothesis, decided from geometry alone.**  The aggregated
    #: reference-free surrogate equals the max form EXACTLY when the agents'
    #: contaminated sets are pairwise disjoint; where they are not, an equality
    #: is a measurement rather than a theorem and does not transfer.  The
    #: compile discloses this on every graph; the driver prints it so the ratio
    #: below is read with the right status.
    pou = projected_assembly(six)
    try:
        mult = pou.contaminated_multiplicity()
    except Exception:                                              # noqa: BLE001
        mult = None
    if mult is not None:
        print(f"W58        contaminated sets pairwise disjoint: "
              f"{mult['disjoint']}   "
              f"{mult['shared_cells']} of {mult['contaminated_cells']} cells in "
              f"more than one subdomain (up to {mult['max_multiplicity']})\n")

    out: dict = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "w58_disjointness": mult,
        "state": state,
        "steps": a.steps,
        "exchanges_per_macro_step": F.EXCHANGES,
        "tiling": {"n_col": six.n_col, "n_row": six.n_row, "wx": six.wx,
                   "wy": six.wy, "nx": six.nx, "ny": six.ny,
                   "halo": six.halo, "ramp": six.ramp},
        "form": "chi-weighted",
        "columns": {}, "control": {},
    }

    # -- control 2: the zero-cut column, through the identical harness ------
    print("--- positive control: SINGLE window, marched the same distance ---")
    t0 = time.perf_counter()
    ctrl = march(one, False, u, v, a.steps)
    c_rows = ctrl.rows
    c_max = max(r["chi_weighted"] for r in c_rows)
    c_ident = max(r["identity_residual"] for r in c_rows)
    print(f"  exchanges measured   {len(c_rows)}")
    print(f"  max chi-weighted     {c_max:.6e}   (must be EXACTLY 0)")
    print(f"  max identity resid   {c_ident:.6e}")
    passes = bool(c_max == 0.0)
    print(f"  control {'PASSES' if passes else 'FAILS'}: one window is E "
          f"through the same solver at the same size, so every D_i is bitwise "
          f"zero   ({time.perf_counter() - t0:.1f} s)")
    out["control"] = {"n_exchanges": len(c_rows),
                      "chi_weighted_max": c_max,
                      "identity_residual_max": c_ident,
                      "passes": passes}
    if not passes:
        print("  [!] a nonzero bound on a zero-cut tiling means the harness is "
              "measuring its own plumbing, not the decomposition.")

    for col in a.columns:
        motion = col == "moving"
        print(f"\n--- {col} column ---")
        t0 = time.perf_counter()
        ro = march(six, motion, u, v, a.steps)
        rows = ro.rows
        s = summarise(rows)
        print(f"  exchanges measured    {s['n_exchanges']}  "
              f"({time.perf_counter() - t0:.1f} s)")
        print(f"  chi-weighted   max    {s['chi_weighted']['max']:.6e}   "
              f"min {s['chi_weighted']['min']:.6e}   "
              f"spread {s['chi_weighted']['spread']:.3f}x")
        print(f"  max form       max    {s['max_form']['max']:.6e}")
        print(f"  composed defect max   {s['composed_defect']['max']:.6e}")
        print(f"  identity residual     "
              f"{s['identity_residual_rel']['max']:.3e} relative   "
              f"<- control 1: the identity closes or the bound is a norm of "
              f"the wrong array")
        print(f"  tightness chi/actual  "
              f"{s['tightness_chi_over_composed']['min']:.4f} to "
              f"{s['tightness_chi_over_composed']['max']:.4f}")
        print(f"  surrogate/max form    "
              f"{s['aggregated_over_max']['min']:.8f} to "
              f"{s['aggregated_over_max']['max']:.8f}  (aggregated); "
              f"pairwise/aggregated "
              f"{s['pairwise_over_aggregated']['min']:.4f} to "
              f"{s['pairwise_over_aggregated']['max']:.4f}")

        # -- control 3: the floor, differenced against a repeat run ---------
        ro2 = march(six, motion, u, v, a.steps)
        floor = max(abs(x["chi_weighted"] - y["chi_weighted"])
                    for x, y in zip(rows, ro2.rows))
        declared = s["chi_weighted"]["max"]
        print(f"  repeat floor          {floor:.6e}   "
              f"bound/floor "
              f"{'inf' if floor == 0 else f'{declared / floor:.3e}'}")

        out["columns"][col] = {
            "summary": s,
            "declared": declared,
            "form": "chi-weighted",
            "repeat_floor": floor,
            "above_floor": bool(floor == 0.0 or declared > 10.0 * floor),
            "rows": rows,
        }
        print(f"  DECLARE cut_defect_bound = {declared:.6e} "
              f"(chi-weighted, max over {s['n_exchanges']} exchanges)")

    path = os.path.join(a.out, "w159.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
