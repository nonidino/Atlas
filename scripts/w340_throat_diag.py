r"""W340 -- what the throat's 3% floor was, measured with no lag in it.

Tier 87 (W336) found 3% of the mass flow vanishing at the throat seam b|e every
macro step, the same at 5 and 2.5 ms, halving with the grid. Its diagnosis was
the handover: `generate._remap`, point interpolation between b's 20 transverse
cells and e's 24. The fix it asked for was an integral-preserving overlap
average. This measures the floor directly, before and after, without marching:

* take the pre-fix run's engine state at one instant (`out/w321_pre_w337`,
  frames 10, 20 and 29), rebuilt in SI from the record;
* wire the throat three ways -- ``remap`` (build repo adc470b, verbatim),
  ``overlap`` (one overlap-averaged state repeated in both ghost layers, the
  briefed fix) and ``layers`` (the neighbour's first two cell layers, each
  overlap-averaged: what the build repo now does);
* evaluate each block's own face fluxes at that same instant, with the
  ledger, and read the mass the seam creates: b's outflow plus e's inflow.

Both sides see the same instant, so there is no lag in any number here: what
is left is the handover's own floor.

    python scripts/w340_throat_diag.py [--record out/w321_pre_w337/episode.npz]
"""
from __future__ import annotations

import argparse
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

import w336_episode_residuals as W  # noqa: E402
import w321_episode_viz_data as V  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", default=os.path.join(ROOT, "out", "w321_pre_w337", "episode.npz"))
    ap.add_argument("--frames", default="10,20,29")
    ap.add_argument("--json", default=os.path.join(ROOT, "out", "w340_throat_diag.json"))
    a = ap.parse_args(argv)
    M = W._mods()
    gen, thermo, NG = M["gen"], M["thermo"], M["C2"].NG
    cfg = M["config"].load_config()
    p = M["sweep"].corner_cases()[0]
    ep = gen.CoupledEpisode(gen.EpisodeSpec(point=p, n_macro=2, dt_macro=5e-3, coarsen=4), cfg)
    z = np.load(a.record)
    refs = ep.scales.per_agent
    mdot = float(ep.scales.mdot)
    led = W.Ledger(M)
    led.install()
    old_wire, _ = W._old_wiring(gen)

    def load_frame(k):
        for ag in ("a", "b", "e"):
            names = list(cfg.agent(ag).fields)
            cube = V.dimensional(z["field_%s" % ag][k:k + 1], names, refs[ag])[0]
            nv = 5 if ag == "a" else 4
            Wp = np.zeros(cube.shape[1:] + (nv,))
            for i, nm in enumerate(("rho", "u", "v", "p")):
                Wp[..., i] = cube[names.index(nm)]
            if nv == 5:
                Wp[..., 4] = cube[names.index("Y")]
            ep.U[ag][0] = thermo.prim_to_cons(Wp, ep.sol[ag][0].cfg.gamma)

    def one_layer():
        """The briefed fix alone: each seam's layer 0, overlap-averaged, in both
        ghost layers."""
        ep._wire_engine()
        for ag, side in (("a", "imax"), ("b", "imin"), ("b", "imax"), ("e", "imin")):
            st = ep.sol[ag][0].bcs[side].params["state"]
            ep.sol[ag][0].bcs[side] = gen.BC("prescribed", {"state": np.asarray(st)[:1]})

    def seams():
        fl = {ag: led.instant(ep.sol[ag][0], ep.U[ag][0]) for ag in ("a", "b", "e")}
        ab = W._sum_side(fl["a"], "imax")[0] + W._sum_side(fl["b"], "imin")[0]
        be = W._sum_side(fl["b"], "imax")[0] + W._sum_side(fl["e"], "imin")[0]
        return dict(ab=float(ab / mdot), be=float(be / mdot),
                    b_throat_outflow=float(-W._sum_side(fl["b"], "imax")[0] / mdot))

    rows = []
    for k in [int(x) for x in a.frames.split(",")]:
        load_frame(k)
        row = dict(frame=k, t=float(z["t"][k]))
        old_wire(ep)
        row["remap"] = seams()
        one_layer()
        row["overlap"] = seams()
        ep._wire_engine()
        assert np.asarray(ep.sol["e"][0].bcs["imin"].params["state"]).shape[0] == NG
        row["layers"] = seams()
        rows.append(row)
        print("frame %2d t=%.3f  b|e created, of mdot:  remap %+.5f  overlap %+.5f  layers %+.6f"
              "   (a|b: %+.1e %+.1e %+.1e)"
              % (k, row["t"], row["remap"]["be"], row["overlap"]["be"], row["layers"]["be"],
                 row["remap"]["ab"], row["overlap"]["ab"], row["layers"]["ab"]))
    # d|g, found while reading the fixed run (W343): the same frozen instants,
    # the whole vehicle loaded, d's outflow on g's side against g's own inflow
    # faces cell by cell, beside g's first-column Mach number
    sm = W.Seams(ep, M)
    dg = []
    for k in [int(x) for x in a.frames.split(",")]:
        for ag in ("d", "f", "g"):
            names = list(cfg.agent(ag).fields)
            cube = V.dimensional(z["field_%s" % ag][k:k + 1], names, refs[ag])[0]
            Wp = np.stack([cube[names.index(nm)] for nm in ("rho", "u", "v", "p")], axis=-1)
            U = thermo.prim_to_cons(Wp, ep.sol[ag][0].cfg.gamma)
            off = 0
            for kk, b in enumerate(ep.blocks[ag]):
                ny = b.shape[1]
                ep.U[ag][kk] = U[:, off:off + ny].copy()
                off += ny
        ep._wire_external()
        d_to_g = 0.0
        for kk in sm.d_frac:
            acc = led.instant(ep.sol["d"][kk], ep.U["d"][kk])
            d_to_g += float(np.sum(-acc["inv"]["imax"][:, 0] * sm.d_frac[kk]["g"]))
        g = led.instant(ep.sol["g"][0], ep.U["g"][0])
        live = sm.g_live_in
        st = np.asarray(ep.sol["g"][0].bcs["imin"].params["state"])[0]
        L = ep.blocks["g"][0].a_i[0]
        Wg = thermo.cons_to_prim(ep.U["g"][0][0], ep.air_cfg.gamma)
        mach = np.hypot(Wg[:, 1], Wg[:, 2]) / np.sqrt(ep.air_cfg.gamma * Wg[:, 3] / Wg[:, 0])
        row = dict(frame=k, d_to_g=d_to_g, ghost_mass=float(np.sum(st[live, 1] * L[live])),
                   g_face=float(g["inv"]["imin"][live, 0].sum()),
                   g_face_by_cell=[float(x) for x in g["inv"]["imin"][live, 0]],
                   ghost_by_cell=[float(x) for x in st[live, 1] * L[live]],
                   g_first_column_mach=[float(x) for x in mach[live]])
        row["created"] = (row["g_face"] - row["d_to_g"]) / row["d_to_g"]
        dg.append(row)
        print("frame %2d  d|g: d's outflow %.5g, handed over %.5g, g's own inflow %.5g (%+.2f%%);"
              " g's first column Mach %.2f..%.2f"
              % (k, d_to_g, row["ghost_mass"], row["g_face"], 100 * row["created"],
                 min(row["g_first_column_mach"]), max(row["g_first_column_mach"])))
    W._persist(a.json, dict(record=os.path.relpath(a.record, ROOT).replace("\\", "/"),
                            build_repo=W.build_repo_identity(), rows=rows, dg=dg))
    print("wrote", a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
