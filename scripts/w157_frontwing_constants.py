"""W157: measure L, sigma and C_mu on the FRONT-WING graph.

[[poc2-novelty-audit]] section 3 found that the compiler has never certified
anything, and that the reason is **provenance rather than physics**: `L`,
`sigma` and `C_mu` are measured in tier 0 -- on `window_ns`, at one state, one
scheme, one depth -- and are simply not declared on this graph's records, so
`W56`'s backstop forces `admit-uncertified` on every compile of it.

A constant measured at one state, one scheme and one depth is not the same
constant somewhere else.  `MeasuredConstants` says so in its own docstring and
carries `probe_state` / `scheme` / `depth` for exactly that reason.  So this
driver does not copy tier 0's numbers across.  It measures the three **here**.

What each one is on this graph
------------------------------

**L** -- the composed rollout's amplification.  Two marches from the same
settled state, one perturbed by a divergence-free blob, least squares on
``log||e^n||`` against ``n``.  The perturbation is a streamfunction bump so the
assembly's own Leray projection does not simply remove it: perturbing with
something the projection deletes measures the projection, not the operator.
Reported per column, because fixed-shape and aeroelastic are different operators
and `wing_fsi` declares a separate record for each.

**sigma** -- the transmission error: what the composed column loses *because its
artificial-boundary datum is stale*.  `reference.WindowNS.step_batch` supplies
the mechanism directly, and its own docstring is the definition --

    ``bc0`` / ``bc1`` are ``(u, v)`` pairs whose **rings** supply the
    transmission data at the start and end of the step; the ring is ramped
    linearly between them across the sub-steps.  ``bc0=None`` takes the ring
    from the input itself, which is the one-pass case and makes the boundary
    data ``t^n`` data held constant.

-- so the two columns are one flag apart:

    lagged   ``bc0=None, bc1=None``   the ring held at ``t^n``: what runs today
    exact    ``bc0=None, bc1=(ring from the SINGLE-WINDOW referent at the end
                              of this exchange)``

and their difference is the stale ring and nothing else.  This is `w49_sigma_halo`'s
construction with the linear ramp done by the solver instead of by hand.

The referent is the same code path over one window, so the two columns differ
**by the cut and by nothing else** -- which is also why ``tau`` is declared 0 for
this graph and is not re-measured here: there is no second operator for a
discretisation defect to live between.  The cut's own contribution is
``cut_defect_bound``, which `L2/C2` asks for separately and this driver does not
supply.

**C_mu** -- implied, exactly as W49 defines it on the overlapping branch:

    sigma <= C_mu * Pi * ||d_lambda||     so    C_mu = sigma / (Pi ||d_lambda||)

``Pi`` is read from **the compiler's own partition of unity**
(``GridPartitionOfUnity.contaminated_weight``) rather than recomputed here, so
the constant is measured against the same quantity the rule consumes.

The positive control, and why it is the shape it is
---------------------------------------------------

The single-window referent is run through the **identical harness** as a column
in its own right.  With one window there are no artificial faces, so:

  * ``Pi`` must be exactly ``0``;
  * ``bc1`` reaches no ring, so lagged and exact must be **bitwise identical**
    and ``sigma`` must be exactly ``0``.

A harness that reports a nonzero sigma there is measuring its own plumbing.
This control is marched **the full macro-step**, not one exchange -- a control
that stops short of the thing it controls has not controlled it.

And ``sigma`` is compared against the state's own round-off level before being
believed: a ratio that comes back flat is a **floor**, not a null.

Run::

    python scripts/w157_frontwing_constants.py
    python scripts/w157_frontwing_constants.py --steps 32 --columns static
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
from atlas.cases.wing_fsi import projected_assembly                # noqa: E402


# ---------------------------------------------------------------------------
# the instrumented rollout: record the ring, and optionally supply it
# ---------------------------------------------------------------------------


class _Instrumented(F.FrontWingRollout):
    """`FrontWingRollout` with `exchange` opened up at the transmission ring.

    Two hooks and nothing else, so the column being measured is the column that
    runs:

      * ``ring_sink``   -- called with (index, u, v) BEFORE the sub-steps, so a
        referent can record the field each exchange starts from;
      * ``ring_source`` -- called with (index) and, if it returns a field, that
        field's cut is passed as ``bc1``, making the ring ramp to the exact
        datum instead of being held.

    Everything else in `exchange` is the parent's, copied rather than called,
    because the parent hard-codes ``bc0=None`` in the middle of it.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.ring_sink = None
        self.ring_source = None
        self._ex_i = 0
        self.ring_pairs: list[tuple[np.ndarray, np.ndarray]] = []

    def exchange(self, u, v, delta, h, w_plate):
        i = self._ex_i
        self._ex_i += 1
        if self.ring_sink is not None:
            self.ring_sink(i, u, v)
        fx, fy, w, load = self.wing.forcing(u, v, delta, w_plate, self.ny,
                                            self.nx, h=h)
        us, vs = self.cut(u), self.cut(v)
        fxs, fys = self.cut(fx), self.cut(fy)

        bc1 = None
        if self.ring_source is not None:
            ref = self.ring_source(i)
            if ref is not None:
                ru, rv = ref
                rus = self.cut(torch.as_tensor(np.asarray(ru), dtype=u.dtype))
                rvs = self.cut(torch.as_tensor(np.asarray(rv), dtype=v.dtype))
                #: **Only the ARTIFICIAL faces may be replaced**, and this is the
                #: whole reason the positive control exists.  `step_batch` reads
                #: the ring and knows nothing about which of its faces are cuts:
                #: handing it the referent's whole ring also overwrites the
                #: DOMAIN boundary, which is a real boundary carrying no stale
                #: datum.  Doing that reported sigma = 6.86e-5 on a SINGLE window
                #: -- where there is no artificial face at all and sigma must be
                #: exactly 0 -- and to four significant figures the same 6.86e-5
                #: on six windows, which is the signature of a harness measuring
                #: its own plumbing rather than the scheme.
                bc1 = _replace_artificial_ring(us, vs, rus, rvs, self.tiling)
                #: ||d_lambda|| is the difference between the ring HELD (r0,
                #: taken from the input) and the ring the referent supplies at
                #: the end of the exchange -- the two the solver ramps between.
                self.ring_pairs.append(
                    (_faces(us, vs, self.tiling),
                     _faces(bc1[0], bc1[1], self.tiling)))

        u1, v1 = self.solver.step_batch(us, vs, self.dt_ex, bc0=None, bc1=bc1,
                                        force=(fxs, fys))
        self.substep_log.append(int(self.solver.last_substeps))
        au, av = self.blend(u1, v1)
        au, av = self.project(au, av)
        bu, bv = self.band(au, av)
        f = self.wing.normal_traction(w)
        drag = (f * self.n_hat_x * self.ds).sum()
        return bu, bv, load, f, drag


def _replace_artificial_ring(us, vs, ref_us, ref_vs, tiling):
    """``(us, vs)`` with ONLY the artificial face lines taken from the referent.

    `step_batch` reads the outermost row/column of each window as its
    transmission ring and cannot tell a cut from the domain edge.  Everything
    that is not an artificial face is left at the held value, so ``r0 == r1``
    there and the solver's linear ramp is a no-op on it.

    With a single window `artificial_faces` is empty, so this returns the input
    unchanged and the exact column becomes bitwise identical to the lagged one.
    That is the positive control.
    """
    bu, bv = us.clone(), vs.clone()
    for k, (ox, oy) in enumerate(tiling.offsets):
        faces = tiling.artificial_faces(ox, oy)
        if "xlo" in faces:
            bu[k][:, 0] = ref_us[k][:, 0]
            bv[k][:, 0] = ref_vs[k][:, 0]
        if "xhi" in faces:
            bu[k][:, -1] = ref_us[k][:, -1]
            bv[k][:, -1] = ref_vs[k][:, -1]
        if "ylo" in faces:
            bu[k][0, :] = ref_us[k][0, :]
            bv[k][0, :] = ref_vs[k][0, :]
        if "yhi" in faces:
            bu[k][-1, :] = ref_us[k][-1, :]
            bv[k][-1, :] = ref_vs[k][-1, :]
    return bu, bv


def _faces(cut_u, cut_v, tiling) -> np.ndarray:
    """Every window's ARTIFICIAL rings, flattened -- the datum ``lambda``.

    A face that coincides with the domain boundary is a real boundary and
    carries no stale datum, so it does not enter.  With one window there are no
    artificial faces at all and this is empty, which is the positive control.
    """
    out = []
    cu = np.asarray(cut_u.detach() if torch.is_tensor(cut_u) else cut_u)
    cv = np.asarray(cut_v.detach() if torch.is_tensor(cut_v) else cut_v)
    for k, (ox, oy) in enumerate(tiling.offsets):
        faces = tiling.artificial_faces(ox, oy)
        for A in (cu[k], cv[k]):
            if "xlo" in faces:
                out.append(A[:, 0])
            if "xhi" in faces:
                out.append(A[:, -1])
            if "ylo" in faces:
                out.append(A[0, :])
            if "yhi" in faces:
                out.append(A[-1, :])
    return np.concatenate(out) if out else np.zeros(0)


# ---------------------------------------------------------------------------
# state and marching
# ---------------------------------------------------------------------------


def settled_state(root: str):
    p = os.path.join(root, "out", "w141", "settled.npz")
    if os.path.isfile(p):
        d = np.load(p)
        return d["u"], d["v"], "out/w141/settled.npz"
    return (np.full((F.NY, F.NX), F.U_INF), np.zeros((F.NY, F.NX)),
            "freestream (out/w141/settled.npz absent)")


def make(tiling, motion: bool) -> _Instrumented:
    return _Instrumented(tiling=tiling, coupling="tight", motion=motion,
                         design=dict(F.DESIGN_REF))


def start(ro, u, v):
    """The macro_step argument tuple `run` would build at the reference design."""
    opt = ro._opt
    kn = ro.knobs(dict(F.DESIGN_REF))
    e_star, tc, k, h0 = kn["e_star"], kn["tc"], kn["k"], kn["h0"]
    U = torch.as_tensor(np.asarray(u), **opt)
    V = torch.as_tensor(np.asarray(v), **opt)
    delta = torch.zeros(F.N_STATION, **opt)
    h = ro.release_height(h0, k)
    return dict(u=U, v=V, delta=delta, h=h,
                w_delta=torch.zeros(F.N_STATION, **opt),
                v_mount=torch.zeros((), **opt),
                knobs=(e_star, tc, k, h0))


def step_once(ro, st, index: int = 0):
    e_star, tc, k, h0 = st["knobs"]
    ro._ex_i = 0
    out = ro.macro_step(st["u"], st["v"], st["delta"], st["h"],
                        st["w_delta"], st["v_mount"],
                        e_star, tc, k, h0, index)
    (u, v, delta, h, w_delta, v_mount, *_rest) = out
    return dict(st, u=u, v=v, delta=delta, h=h,
                w_delta=w_delta, v_mount=v_mount)


def rel_l2(a, b) -> float:
    num = float(torch.sqrt(torch.sum((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)))
    den = float(torch.sqrt(torch.sum(b[0] ** 2 + b[1] ** 2))) or 1.0
    return num / den


# ---------------------------------------------------------------------------
# L
# ---------------------------------------------------------------------------


def blob(ny: int, nx: int, amp: float, dx: float):
    """A streamfunction bump: divergence-free, so the projection keeps it."""
    gy = (np.arange(ny) + 0.5) * dx
    gx = (np.arange(nx) + 0.5) * dx
    X, Y = np.meshgrid(gx, gy)
    r2 = (((X - 0.35 * gx[-1]) ** 2 + (Y - 0.5 * gy[-1]) ** 2)
          / (0.12 * gx[-1]) ** 2)
    psi = amp * np.exp(-r2)
    return np.gradient(psi, dx, axis=0), -np.gradient(psi, dx, axis=1)


def fit_L(tiling, motion: bool, u, v, n_steps: int, amp: float,
          say=print) -> dict:
    du, dv = blob(u.shape[0], u.shape[1], amp * float(np.abs(u).max()), F.DX)
    a = make(tiling, motion)
    b = make(tiling, motion)
    sa, sb = start(a, u, v), start(b, u + du, v + dv)
    errs = []
    for i in range(n_steps):
        sa = step_once(a, sa, i)
        sb = step_once(b, sb, i)
        errs.append(rel_l2((sb["u"], sb["v"]), (sa["u"], sa["v"])))
        if not np.isfinite(errs[-1]):
            say(f"    [!] the paired march went non-finite at step {i}; "
                "stopping and reporting the horizon reached")
            break
    e = np.log(np.maximum(np.asarray(errs), 1e-300))
    k = np.arange(1, len(errs) + 1)
    sl, ic = np.polyfit(k, e, 1)
    resid = e - (sl * k + ic)
    se = float(np.sqrt(np.sum(resid ** 2) / max(len(errs) - 2, 1)
                       / np.sum((k - k.mean()) ** 2)))
    L = float(np.exp(sl))
    return dict(L=L, L_stderr=float(L * se), n_steps=len(errs), amp=amp,
                errors=[float(x) for x in errs],
                rms_log_residual=float(np.sqrt(np.mean(resid ** 2))),
                e_first=float(errs[0]), e_last=float(errs[-1]),
                branch=("contractive" if L < 1.0 else "expanding"))


# ---------------------------------------------------------------------------
# sigma
# ---------------------------------------------------------------------------


def measure_sigma(tiling, motion: bool, u, v) -> dict:
    """One macro-step, twice: ring held, and ring ramped to the referent's."""
    #: the referent, recording the field each exchange starts from
    ref = make(F.SINGLE_TILING, motion)
    frames: list[tuple[np.ndarray, np.ndarray]] = []

    def sink(i, uu, vv):
        frames.append((uu.detach().cpu().numpy().copy(),
                       vv.detach().cpu().numpy().copy()))

    ref.ring_sink = sink
    sr = start(ref, u, v)
    sr = step_once(ref, sr)
    #: the referent's END-of-exchange field: exchange i ends where i+1 begins,
    #: and the last one ends at the macro-step's own end
    ends = frames[1:] + [(sr["u"].detach().cpu().numpy(),
                          sr["v"].detach().cpu().numpy())]

    lag = make(tiling, motion)
    s_lag = step_once(lag, start(lag, u, v))

    ex = make(tiling, motion)
    ex.ring_source = lambda i: ends[i] if i < len(ends) else None
    s_ex = step_once(ex, start(ex, u, v))

    #: **The measurement floor, differenced against a run rather than assumed.**
    #: `flat ratios are floors, not nulls`: a sigma near the level at which two
    #: identical runs already disagree is not a measurement of transmission, it
    #: is the harness's own noise.  So the same column is run a second time and
    #: the two are differenced.  On a deterministic solver this is bitwise 0 and
    #: any nonzero sigma is signal; if it is not, it is the number sigma has to
    #: clear before it means anything.
    lag2 = make(tiling, motion)
    s_lag2 = step_once(lag2, start(lag2, u, v))
    floor = rel_l2((s_lag["u"], s_lag["v"]), (s_lag2["u"], s_lag2["v"]))

    dl = [float(np.linalg.norm(a - b)) for a, b in ex.ring_pairs]
    ring_level = [float(np.linalg.norm(b)) for _a, b in ex.ring_pairs]
    norm = float(torch.sqrt(torch.sum(s_ex["u"] ** 2 + s_ex["v"] ** 2)))
    sigma = rel_l2((s_lag["u"], s_lag["v"]), (s_ex["u"], s_ex["v"]))
    return dict(sigma=sigma,
                dlambda_abs=max(dl) if dl else 0.0,
                dlambda_rel=(max(dl) / norm) if dl and norm else 0.0,
                ring_norm=max(ring_level) if ring_level else 0.0,
                n_exchanges_with_a_ring=len(dl),
                field_norm=norm,
                repeat_floor=floor,
                sigma_over_floor=(sigma / floor) if floor else float("inf"),
                above_floor=bool(floor == 0.0 or sigma > 10.0 * floor))


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join("out", "w157"))
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--amp", type=float, default=1e-3)
    ap.add_argument("--columns", nargs="*", default=["static", "moving"],
                    choices=["static", "moving"])
    ap.add_argument("--ramps", nargs="*", type=int,
                    default=[1, 2, 4, 8, 12, 16],
                    help="partition-of-unity ramp widths for the C_mu sweep")
    a = ap.parse_args(argv)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(1)

    u, v, state = settled_state(root)
    six, one = F.DEFAULT_TILING, F.SINGLE_TILING

    pi = projected_assembly(six).partition.contaminated_weight()
    pi_one = projected_assembly(one).partition.contaminated_weight()

    print(f"state      {state}")
    print(f"tiling     {six.n_col}x{six.n_row} windows {six.wx}x{six.wy} on "
          f"{six.nx}x{six.ny}, halo {six.halo}, ramp {six.ramp}")
    print(f"exchanges  {F.EXCHANGES} per macro-step of {F.MACRO_DT}\n")
    print(f"Pi  six windows   {pi}")
    print(f"Pi  one window    {pi_one}   <- control: no artificial faces")
    if pi_one not in (0.0, None):
        print("  [!] nonzero Pi on a single window means the harness is "
              "measuring its own plumbing.")

    out: dict = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "state": state,
        "Pi": pi, "Pi_single_window": pi_one,
        "tiling": {"n_col": six.n_col, "n_row": six.n_row, "wx": six.wx,
                   "wy": six.wy, "nx": six.nx, "ny": six.ny,
                   "halo": six.halo, "ramp": six.ramp},
        "exchanges": F.EXCHANGES, "macro_dt": F.MACRO_DT,
        "columns": {}, "control": {},
    }

    print("\n--- positive control: the referent through the same harness ---")
    t0 = time.perf_counter()
    ctrl = measure_sigma(one, False, u, v)
    print(f"  sigma = {ctrl['sigma']:.6e}   rings seen = "
          f"{ctrl['n_exchanges_with_a_ring']}   ({time.perf_counter()-t0:.1f} s)")
    ctrl["passes"] = bool(ctrl["sigma"] == 0.0 and pi_one in (0.0, None))
    print(f"  control {'PASSES' if ctrl['passes'] else 'FAILS'}: with one "
          "window there is no artificial ring, so both columns must be "
          "identical and sigma exactly 0")
    out["control"] = ctrl

    for col in a.columns:
        motion = col == "moving"
        print(f"\n--- {col} column (motion={motion}) ---")
        t0 = time.perf_counter()
        Lm = fit_L(six, motion, u, v, a.steps, a.amp)
        print(f"  L      = {Lm['L']:.6f} +/- {Lm['L_stderr']:.6f}  "
              f"({Lm['branch']}, rms log resid {Lm['rms_log_residual']:.3f}, "
              f"e {Lm['e_first']:.3e} -> {Lm['e_last']:.3e} over "
              f"{Lm['n_steps']} steps)")
        sg = measure_sigma(six, motion, u, v)
        c_mu = (sg["sigma"] / (pi * sg["dlambda_rel"])
                if pi and sg["dlambda_rel"] else float("nan"))
        print(f"  sigma  = {sg['sigma']:.6e}   |dlambda|_rel = "
              f"{sg['dlambda_rel']:.6e}")
        print(f"  floor  = {sg['repeat_floor']:.6e}  (two identical runs "
              f"differenced)  sigma/floor = {sg['sigma_over_floor']:.3g}")
        if not sg["above_floor"]:
            print("  [!] sigma is at or near the repeat floor, so it is UNDER "
                  "a measurement floor rather than small. Reported as a floor.")
        print(f"  C_mu   = {c_mu:.6f}   (implied: sigma / (Pi |dlambda|))")
        out["columns"][col] = dict(motion=motion, C_mu_implied=float(c_mu),
                                   seconds=time.perf_counter() - t0, **Lm, **sg)
        print(f"  ({time.perf_counter()-t0:.1f} s)")

    # -- the C_mu sweep ----------------------------------------------------
    #: **Two configurations is not a sample, and a constant fitted from two
    #: points is a fit dressed as a bound.**  W49 measured C_mu over sixteen
    #: configurations -- a halo sweep and five partitions of unity -- and took
    #: the sampled maximum.  Here the geometry is *determined*: with n_col=3,
    #: n_row=2 on 208x144, ``2*stride + nw = 208`` and ``stride + nw = 144``
    #: force stride=64 and nw=80, so the halo cannot be varied without changing
    #: the domain.  The partition of unity can, and that is the axis W49 found
    #: C_mu actually moves along, so the sweep runs it.
    print("\n--- C_mu over the partition-of-unity sweep ---")
    print(f"{'ramp':>5} {'column':>8} {'Pi':>10} {'|dlambda|':>12} "
          f"{'sigma':>12} {'C_mu implied':>13} {'floor':>10}")
    sweep = []
    for ramp in a.ramps:
        try:
            t = type(six)(n_col=six.n_col, n_row=six.n_row, nw=six.nw,
                          stride=six.stride, ramp=ramp)
            pi_r = projected_assembly(t).partition.contaminated_weight()
        except Exception as exc:                                   # noqa: BLE001
            print(f"{ramp:>5}  tiling refused: {type(exc).__name__}: "
                  f"{str(exc)[:48]}")
            continue
        for col in a.columns:
            motion = col == "moving"
            try:
                m = measure_sigma(t, motion, u, v)
            except Exception as exc:                               # noqa: BLE001
                print(f"{ramp:>5} {col:>8}  failed: {type(exc).__name__}: "
                      f"{str(exc)[:44]}")
                continue
            imp = (m["sigma"] / (pi_r * m["dlambda_rel"])
                   if pi_r and m["dlambda_rel"] else float("nan"))
            row = dict(ramp=ramp, column=col, Pi=pi_r, C_mu_implied=float(imp),
                       **m)
            sweep.append(row)
            print(f"{ramp:>5} {col:>8} {pi_r:10.6f} {m['dlambda_rel']:12.4e} "
                  f"{m['sigma']:12.4e} {imp:13.6f} "
                  f"{m['repeat_floor']:10.2e}")
    out["sweep"] = sweep

    good = [r for r in sweep if np.isfinite(r["C_mu_implied"])
            and r["above_floor"]]
    if good:
        imp = [r["C_mu_implied"] for r in good]
        sig = [r["sigma"] for r in good]
        lo, hi = min(imp), max(imp)
        #: the same convention W49 used: the SAMPLED MAXIMUM, rounded up, so the
        #: reported constant is a bound over what was measured rather than a fit
        #: through it.  Two decimals, because the sample is 6 partitions and not
        #: 16 configurations and a third digit would claim resolution the sample
        #: does not have.
        c_declare = float(np.ceil(hi * 100.0) / 100.0)
        out["C_mu"] = dict(
            min=lo, max=hi, spread=hi / lo if lo else None,
            declared=c_declare, n_configurations=len(good),
            sigma_span=max(sig) / min(sig) if min(sig) else None,
            convention="sampled maximum, rounded up (W49's)",
            holds_on=sum(1 for r in good
                         if c_declare * r["Pi"] * r["dlambda_rel"] >= r["sigma"]),
        )
        print(f"\nC_mu implied over {len(good)} configurations: "
              f"[{lo:.6f}, {hi:.6f}], spread {hi / lo:.2f}x, "
              f"while sigma moves {max(sig) / min(sig):.3g}x")
        print(f"C_mu declared (sampled max, rounded up): {c_declare}")
        print(f"  the bound holds on {out['C_mu']['holds_on']}/{len(good)}")
        #: tier 0's constant, for comparison -- NOT adopted, just placed beside
        from atlas.assembly import C_MU_HALO                        # noqa: PLC0415
        out["C_mu"]["tier0_C_MU_HALO"] = float(C_MU_HALO)
        out["C_mu"]["tier0_margin_over_this_graph"] = float(C_MU_HALO / hi)
        print(f"  tier 0's C_mu = {C_MU_HALO} also holds here, with "
              f"{C_MU_HALO / hi:.1f}x of margin over this graph's worst case")

    p = os.path.join(a.out, "w157.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
