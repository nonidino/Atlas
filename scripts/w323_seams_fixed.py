r"""W323-W331 -- the rocket's seams after the fix: gates written before the run.

`scripts/w321_seam_audit.py` measured what was wrong with the first marched
episode (out/w321_prefix, build repo 0a407b7 as committed). The build repo was
then fixed (research vault page `rocket-episode-seam-audit`, Tier 86). This
script is the other half: every defect that audit found becomes a GATE, and the
gates were written down BEFORE the fixed episode was marched, so the ones that
read the episode are predictions, not descriptions.

Three groups, the same three layers the audit separated:

* **geometry** -- read off the build repo's blocks, no episode needed;
* **wiring** -- the coupler's exchanges probed with synthetic states whose
  answer is known (a pressure equal to z reads back as the source's z);
* **the marched episode** -- symmetry, load directions, the energy bound on
  the wake, read from ``out/w321/episode.npz``.

Each gate prints PASS or FAIL with the number it judged; the JSON records them.
``--figures`` redraws the audit's three pictures from the fixed run.

The first fixed run (now out/w321_seams_only) passed 11 of 13: E1 and E2, the
mirror-symmetry gates, failed at 2.7e-4 and 1.2e-5. The diagnosis was not the
seams: the isothermal wall's ghost, handed to the Riemann solver, let the
nozzle's wall face pass mass and carried a mode that grew from round-off by
~1e5 per 5 ms (W332). The first march's own data also showed the forebody
slot's wall counted as airframe (W333). Four gates for those were added
2026-09-23, before the re-march, and E1-E5 keep their thresholds.

    python scripts/w323_seams_fixed.py [--figures]
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from atlas.cases import rocket_experts as RE

#: The gates, registered before the fixed episode was marched (2026-09-23).
#: name -> (what it judges, the threshold it must meet)
GATES = {
    "G1_polylines":   ("max |gap or overlap| between gas wall, shell faces and d, coarsen 1 and 4", 1e-12),
    "G2_coverage":    ("raster points claimed by no block or by two, engine window and whole domain", 0),
    "G3_hole":        ("|g's hole edge - f's edge|, coarsen 1 and 4", 1e-12),
    "C1_inner":       ("max |source z - station z| on the shell's inner face (with a p = z probe)", None),
    "C2_outer":       ("max |source z - station z| on the shell's outer face", None),
    "C3_barrel":      ("barrel stations: h and p - p_inf", 0.0),
    "C4_gas_model":   ("max relative change of p, T, u, v across _as_gas", 1e-12),
    "C5_hole_state":  ("hole_state is (lo, hi) of f's own edge rows", None),
    "E1_mirror":      ("max |uy_lower + uy_upper(mirrored)| / max |uy|, every step", 1e-6),
    "E2_directions":  ("thrust up, drag to the tail and down, |side| / drag, every marched step", 1e-6),
    "E3_climbs":      ("vy(end) - vy(0) > 0", 0.0),
    "E4_audit":       ("dvy/dt against the loads it was handed", 0.02),
    "E5_wake_energy": ("max T in g over max inflow stagnation T (air, d's outlet, f's edge rows)", 1.02),
    # W332 and W333, registered before the re-march (2026-09-23)
    "C6_wall_mass":   ("closed engine blocks with the coupler's isothermal walls, gas pushed at "
                       "both walls: |net mass rate| / (rho |v| wall length)", 1e-12),
    "C7_slot_wall":   ("d's wall is slip and adiabatic at exactly the stations ahead of the nose", None),
    "C8_slot_record": ("c-d length against the airframe's outer face (rel.), and the aero load's "
                       "change when the slot's cells change", 1e-12),
    "E6_engine_mirror": ("max over steps of the p mirror asymmetry of a, b and e", 1e-8),
}


def _mods():
    RE.load_solvers()
    names = ("data.generate", "data.sweep", "config", "geometry.contours",
             "solvers.thermo", "solvers.atmosphere", "solvers.grid")
    return [importlib.import_module("atlas_build_solvers." + n) for n in names]


def _raster(blocks, unmodelled, z0, z1, y0, y1, dx):
    zs = np.arange(z0 + dx / 2, z1, dx)
    ys = np.arange(y0 + dx / 2, y1, dx)
    ZZ, YY = np.meshgrid(zs, ys, indexing="ij")
    cnt = np.zeros(ZZ.shape, np.int16)
    for blk in blocks:
        n = blk.nodes
        q = np.stack([n[:-1, :-1], n[1:, :-1], n[1:, 1:], n[:-1, 1:]], axis=2)[~blk.blanked]
        for quad in q:
            i0, i1 = np.searchsorted(zs, quad[:, 0].min()), np.searchsorted(zs, quad[:, 0].max())
            j0, j1 = np.searchsorted(ys, quad[:, 1].min()), np.searchsorted(ys, quad[:, 1].max())
            if i1 <= i0 or j1 <= j0:
                continue
            PZ, PY = ZZ[i0:i1, j0:j1], YY[i0:i1, j0:j1]
            inside = np.ones(PZ.shape, bool)
            for m in range(4):
                (az, ay), (bz, by), (cz, cy) = quad[m], quad[(m + 1) % 4], quad[(m + 2) % 4]
                side = np.sign((bz - az) * (cy - ay) - (by - ay) * (cz - az))
                inside &= ((bz - az) * (PY - ay) - (by - ay) * (PZ - az)) * side >= 0
            cnt[i0:i1, j0:j1] += inside
    unm = np.zeros(ZZ.shape, bool)
    for r in unmodelled:
        unm |= (ZZ >= r.z[0]) & (ZZ <= r.z[1]) & (np.abs(YY) <= r.y_halfwidth)
    return dict(zs=zs, ys=ys, cnt=cnt, unm=unm)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="out/w321")
    ap.add_argument("--out", default="out/w323_seams_fixed.json")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args(argv)

    gen, sw, cfgm, con, thermo, atm, grid = _mods()
    cfg = cfgm.load_config()
    geo = cfg.geometry
    R = {"gates": {}, "registered": {k: v[0] for k, v in GATES.items()}}

    def gate(name, value, ok, detail=""):
        R["gates"][name] = dict(value=value, passed=bool(ok), detail=detail)
        print("  %-15s %s  %s  %s" % (name, "PASS" if ok else "FAIL", value, detail), flush=True)

    print("=" * 84)
    print("W323-W331 -- the rocket's seams after the fix")
    print("=" * 84)

    # ------------------------------------------------------------ geometry
    print("\n--- geometry")
    worst = 0.0
    for k in (1, 4):
        blk = {x.id: grid.build_blocks(x, cfg, k) for x in cfg.agents}
        zz = np.linspace(0.0, geo.z_exit, 7001)
        gz = np.concatenate([blk["a"][0].nodes[:, -1, 0], blk["b"][0].nodes[1:, -1, 0],
                             blk["e"][0].nodes[1:, -1, 0]])
        gy = np.concatenate([blk["a"][0].nodes[:, -1, 1], blk["b"][0].nodes[1:, -1, 1],
                             blk["e"][0].nodes[1:, -1, 1]])
        c1, d1 = blk["c"][1], blk["d"][1]
        gas = np.interp(zz, gz, gy)
        si = np.interp(zz, c1.nodes[:, 0, 0], c1.nodes[:, 0, 1])
        so = np.interp(zz, c1.nodes[:, -1, 0], c1.nodes[:, -1, 1])
        di = np.interp(zz, d1.nodes[:, 0, 0], d1.nodes[:, 0, 1])
        worst = max(worst, float(np.abs(si - gas).max()), float(np.abs(di - so).max()),
                    float(np.abs(gas - con.nozzle_inner(zz, geo)).max()))
    gate("G1_polylines", "%.2e m" % worst, worst < GATES["G1_polylines"][1])

    blocks4 = [b for x in cfg.agents for b in grid.build_blocks(x, cfg, 4)]
    cov_e = _raster(blocks4, cfg.unmodelled, -0.1, 1.0, -0.2, 0.2, 5e-4)
    cov_f = _raster(blocks4, cfg.unmodelled, geo.atmos_front_z[0], geo.plume_z[1],
                    -geo.farfield_halfwidth, geo.farfield_halfwidth, 5e-3)
    bad = 0
    for cv in (cov_e, cov_f):
        bad += int(((cv["cnt"] == 0) & ~cv["unm"]).sum()) + int((cv["cnt"] >= 2).sum())
    gate("G2_coverage", "%d points" % bad, bad == 0)

    herr = 0.0
    for k in (1, 4):
        g = grid.build_blocks(cfg.agent("g"), cfg, k)[0]
        f = grid.build_blocks(cfg.agent("f"), cfg, k)[0]
        js = np.flatnonzero(g.blanked[0])
        herr = max(herr, abs(g.nodes[0, js[0], 1] - f.nodes[0, 0, 1]),
                   abs(g.nodes[0, js[-1] + 1, 1] - f.nodes[0, -1, 1]))
    gate("G3_hole", "%.2e m" % herr, herr < GATES["G3_hole"][1])

    # -------------------------------------------------------------- wiring
    print("\n--- wiring (synthetic probes; nothing marched)")
    with open(os.path.join(a.run, "episode.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    pt = sw.corner_cases()[meta["case"]]
    ep = gen.CoupledEpisode(gen.EpisodeSpec(point=pt, n_macro=2, dt_macro=meta["dt_macro"],
                                            coarsen=meta["coarsen"]), cfg)
    _, p_inf, _, _ = ep._atm()
    offset = 1.0e3                                  # keep the probe pressure positive
    for agent in gen.ENGINE_AGENTS:
        blk = ep.blocks[agent][0]
        W = thermo.cons_to_prim(ep.U[agent][0], ep.gas_cfg.gamma)
        W[..., 3] = offset + 1.0e3 * blk.centroid[..., 0]
        ep.U[agent][0] = thermo.prim_to_cons(W, ep.gas_cfg.gamma)
    inner_mis, barrel_bad = 0.0, 0.0
    for k in (0, 1):
        h, T, p = ep._shell_inner_loads(k, p_inf)
        zs, _ = ep._shell_face(k, "inner")
        zm = 0.5 * (zs[:-1] + zs[1:])
        eng = zm > 0.0
        src = (p[eng] - offset) / 1.0e3
        inner_mis = max(inner_mis, float(np.abs(src - zm[eng]).max()))
        barrel_bad = max(barrel_bad, float(np.abs(h[~eng]).max()),
                         float(np.abs(p[~eng] - p_inf).max()))
    gate("C1_inner", "%.4f m" % inner_mis, inner_mis <= 0.5 * 0.01 * meta["coarsen"] / 4 + 1e-12,
         "(half a gas wall cell is %.4f m)" % (0.5 * 0.01 * meta["coarsen"] / 4))
    gate("C3_barrel", "%.3g" % barrel_bad, barrel_bad == 0.0)

    offset_d = 5.0e3
    for k, blk in enumerate(ep.blocks["d"]):
        W = thermo.cons_to_prim(ep.U["d"][k], ep.air_cfg.gamma)
        W[..., 3] = offset_d + 1.0e3 * blk.centroid[..., 0]
        ep.U["d"][k] = thermo.prim_to_cons(W, ep.air_cfg.gamma)
    outer_mis = 0.0
    for k in (0, 1):
        _h, _T, p = ep._shell_outer_loads(k)
        zs, _ = ep._shell_face(k, "outer")
        zm = 0.5 * (zs[:-1] + zs[1:])
        outer_mis = max(outer_mis, float(np.abs((p - offset_d) / 1.0e3 - zm).max()))
    dz_d = float(np.diff(ep.blocks["d"][0].nodes[:, 0, 0]).max())
    gate("C2_outer", "%.4f m" % outer_mis, outer_mis <= 0.5 * dz_d + 1e-12,
         "(half of d's largest wall cell is %.4f m)" % (0.5 * dz_d))

    W = np.array([[0.04, 1044.0, 3.0, 2549.0]])
    U = thermo.prim_to_cons(W, ep.air_cfg.gamma)
    V = thermo.cons_to_prim(gen._as_gas(U, ep.air_cfg, ep.gas_cfg), ep.gas_cfg.gamma)
    Ta, Tg = W[0, 3] / (W[0, 0] * ep.air_cfg.R), V[0, 3] / (V[0, 0] * ep.gas_cfg.R)
    rel = max(abs(V[0, 3] / W[0, 3] - 1), abs(Tg / Ta - 1), abs(V[0, 1] / W[0, 1] - 1),
              abs(V[0, 2] / W[0, 2] - 1))
    gate("C4_gas_model", "%.2e" % rel, rel < GATES["C4_gas_model"][1])

    ep2 = gen.CoupledEpisode(gen.EpisodeSpec(point=pt, n_macro=2, dt_macro=meta["dt_macro"],
                                             coarsen=meta["coarsen"]), cfg)
    ep2._wire_engine()
    ep2._wire_external()
    hs = ep2.sol["g"][0].hole_state
    Uf = ep2.U["f"][0]
    ok = isinstance(hs, tuple) and len(hs) == 2
    if ok:
        lo = gen._as_gas(np.asarray(hs[0])[:, 0], ep2.air_cfg, ep2.gas_cfg)
        hi = gen._as_gas(np.asarray(hs[1])[:, 0], ep2.air_cfg, ep2.gas_cfg)
        ok = bool(np.allclose(lo, Uf[:, 0], rtol=1e-10) and np.allclose(hi, Uf[:, -1], rtol=1e-10))
    gate("C5_hole_state", "pair of f's edge rows" if ok else "not f's edge rows", ok)

    # W332: every engine block, closed -- its own isothermal walls as the coupler
    # wires them on j, adiabatic walls on i -- with the gas pushed at both walls.
    # An impermeable box conserves mass; the thermal ghost in the inviscid flux
    # did not.
    BCk = type(ep2.sol["a"][0].bcs["jmin"])
    worst_mass = 0.0
    for agent in gen.ENGINE_AGENTS:
        s = ep2.sol[agent][0]
        blk = s.block
        saved_bcs = dict(s.bcs)
        try:
            s.bcs = {"jmin": saved_bcs["jmin"], "jmax": saved_bcs["jmax"],
                     "imin": BCk("wall_noslip"), "imax": BCk("wall_noslip")}
            W = thermo.cons_to_prim(ep2.U[agent][0], s.cfg.gamma).copy()
            W[..., 2] = 50.0 * np.sign(blk.centroid[..., 1])      # at both walls
            U = thermo.prim_to_cons(W, s.cfg.gamma)
            r = s.residual(U)
            net = abs(float((r[..., 0] * blk.vol).sum()))
            scale = float(W[..., 0].max()) * 50.0 * float(blk.a_j[:, 0].sum() + blk.a_j[:, -1].sum())
            worst_mass = max(worst_mass, net / scale)
        finally:
            s.bcs = saved_bcs
    gate("C6_wall_mass", "%.2e" % worst_mass, worst_mass < GATES["C6_wall_mass"][1])

    # W333: the forebody slot
    z_nose = float(geo.shell_z[0])
    ok, why = True, "slip and adiabatic ahead of the nose, no-slip isothermal behind it"
    for k in range(len(ep2.blocks["d"])):
        side = ep2._d_wall_side(k)
        bc = ep2.sol["d"][k].bcs[side]
        z = ep2._wall_edges("d", k, side)
        behind = 0.5 * (z[:-1] + z[1:]) >= z_nose
        if not (np.array_equal(np.asarray(bc.params.get("no_slip_mask")), behind)
                and np.array_equal(np.asarray(bc.params.get("isothermal_mask")), behind)):
            ok, why = False, "panel %d's masks do not match the airframe's extent" % k
        if np.abs(z - z_nose).min() > 1e-12:
            ok, why = False, "no node line at the nose on panel %d" % k
    gate("C7_slot_wall", "ok" if ok else "wrong", ok, why)

    rec = ep2._iface_records()["c-d"]
    outer = sum(float(np.linalg.norm(np.diff(m.nodes[:, m.outer_j], axis=0), axis=-1).sum())
                for m in ep2.shell_mesh)
    len_err = abs(float(rec[7].sum()) / outer - 1.0)
    ep2._compute_loads()
    base = ep2.loads_body.copy()
    for k, blk in enumerate(ep2.blocks["d"]):
        ahead = blk.centroid[:, 0, 0] < z_nose
        U = ep2.U["d"][k].copy()
        Wa = thermo.cons_to_prim(U[ahead], ep2.air_cfg.gamma)
        Wa[..., 1] *= 3.0
        Wa[..., 3] *= 2.0
        U[ahead] = thermo.prim_to_cons(Wa, ep2.air_cfg.gamma)
        ep2.U["d"][k] = U
    ep2._compute_loads()
    dload = float(np.abs(ep2.loads_body[2:] - base[2:]).max())
    gate("C8_slot_record", "length %.2e, load change %.2e" % (len_err, dload),
         len_err < GATES["C8_slot_record"][1] and dload == 0.0,
         "(c-d %.4f m, the airframe's outer face %.4f m)" % (float(rec[7].sum()), outer))

    # ------------------------------------------------------------- episode
    print("\n--- the marched episode (%s)" % a.run)
    Z = np.load(os.path.join(a.run, "episode.npz"))
    rigid, loads = Z["rigid"], Z["loads"]
    body = Z["loads_body"] if "loads_body" in Z.files else None
    SC = ep.scales.per_agent
    rc = SC["c"]
    fc = Z["field_c"]
    worst_m = 0.0
    for s in range(fc.shape[0]):
        uy = fc[s, 2] * rc.L                        # (58, 4): panel 0 cols 0-1, panel 1 cols 2-3
        lo, hi = uy[:, 0:2], uy[:, 2:4]
        sc = max(np.abs(hi).max(), 1e-300)
        if np.abs(hi).max() > 0:
            worst_m = max(worst_m, float(np.abs(lo[:, ::-1] + hi).max() / sc))
    gate("E1_mirror", "%.2e" % worst_m, worst_m < GATES["E1_mirror"][1])

    marched = slice(1, None)
    up = bool((loads[marched, 1] > 0).all())
    down = bool((loads[marched, 3] < 0).all())
    if body is not None:
        tail = bool((body[marched, 2] > 0).all())
        side = float(np.max(np.abs(body[marched, 3]) / np.maximum(np.abs(body[marched, 2]), 1e-300)))
    else:
        tail, side = False, float("inf")
    ok = up and down and tail and side < GATES["E2_directions"][1]
    gate("E2_directions", "thrust up %s, drag to tail %s, drag down %s, |side|/drag %.2e"
         % (up, tail, down, side), ok)

    dv = float(rigid[-1, 4] - rigid[0, 4])
    gate("E3_climbs", "%+.3f m/s" % dv, dv > 0.0)

    dt = meta["dt_macro"]
    worst_a = 0.0
    for i in range(rigid.shape[0] - 1):
        m = float(rigid[i, 6])
        g = float(atm.gravity(float(rigid[i, 1])))
        want = (loads[i + 1, 1] + loads[i + 1, 3]) / m - g
        got = (rigid[i + 1, 4] - rigid[i, 4]) / dt
        worst_a = max(worst_a, abs(got / want - 1.0))
    gate("E4_audit", "%.2e" % worst_a, worst_a < GATES["E4_audit"][1])

    # the wake's energy bound: adiabatic air cannot get hotter than the hottest
    # stagnation temperature flowing into it -- the freestream, d's outlet, and
    # f's edge rows as g's hole sees them (plume gas at its own T and speed)
    cp_air = atm.GAMMA_AIR * atm.R_AIR / (atm.GAMMA_AIR - 1.0)
    rd, rf, rg = SC["d"], SC["f"], SC["g"]
    names = list(cfg.agent("g").fields)
    iT, iu, iv = names.index("T"), names.index("u"), names.index("v")
    worst_e, which = 0.0, ""
    g_blk = ep.blocks["g"][0]
    live = ~g_blk.blanked[0]
    for s in range(1, rigid.shape[0]):
        _, _, _, c_s = (float(x) for x in atm.properties(float(rigid[s, 1])))
        V = float(np.hypot(*rigid[s, 3:5]))
        T_inf = float(atm.properties(float(rigid[s, 1]))[0])
        T0 = [T_inf + V ** 2 / (2 * cp_air)]
        Fd = Z["field_d"][s]
        Td = Fd[iT, -1] * rd.T
        spd = np.hypot(Fd[iu, -1], Fd[iv, -1]) * rd.u
        T0.append(float((Td + spd ** 2 / (2 * cp_air)).max()))
        Ff = Z["field_f"][s]
        for j in (0, -1):
            Tf = Ff[iT, :, j] * rf.T
            sf = np.hypot(Ff[iu, :, j], Ff[iv, :, j]) * rf.u
            T0.append(float((Tf + sf ** 2 / (2 * cp_air)).max()))
        Tg = float((Z["field_g"][s][iT][:, live] * rg.T).max())
        r = Tg / max(T0)
        if r > worst_e:
            worst_e, which = r, "step %d: g %.0f K against %.0f K" % (s, Tg, max(T0))
    gate("E5_wake_energy", "%.3f" % worst_e, worst_e <= GATES["E5_wake_energy"][1], which)

    # W332: the engine's mirror symmetry through the whole march
    worst_eng, where = 0.0, ""
    for aid in gen.ENGINE_AGENTS:
        ip = list(cfg.agent(aid).fields).index("p")
        F = Z["field_%s" % aid]
        for s in range(F.shape[0]):
            p = F[s, ip]
            v = float(np.abs(p - p[:, ::-1]).max() / max(np.abs(p).max(), 1e-300))
            if v > worst_eng:
                worst_eng, where = v, "%s at step %d" % (aid, s)
    gate("E6_engine_mirror", "%.2e" % worst_eng, worst_eng < GATES["E6_engine_mirror"][1], where)

    R["episode"] = dict(run=a.run, build_repo=meta.get("build_repo"),
                        vy=[float(rigid[0, 4]), float(rigid[-1, 4])],
                        loads_last=[float(x) for x in loads[-1]],
                        loads_body_last=None if body is None else [float(x) for x in body[-1]])
    R["all_passed"] = all(v["passed"] for v in R["gates"].values())
    print("\n  %d of %d gates pass" % (sum(v["passed"] for v in R["gates"].values()), len(R["gates"])))

    for k in range(40):
        try:
            with open(a.out, "w", encoding="utf-8") as fh:
                json.dump(R, fh, indent=1)
            break
        except PermissionError:
            time.sleep(0.25)
    print("wrote %s" % a.out)

    if a.figures:
        _figures(os.path.dirname(os.path.abspath(os.path.join(a.run, "x"))), cfg, geo, con,
                 gen, Z, ep.blocks, SC, cov_e, cov_f)
    return 0 if R["all_passed"] else 1


def _figures(outdir, cfg, geo, con, gen, Z, B, SC, cov_e, cov_f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LogNorm

    PATCH = []
    for aid in gen.GAS_AGENTS:
        Tn = Z["field_%s" % aid][:, list(cfg.agent(aid).fields).index("T")]
        off = 0
        for k, b in enumerate(B[aid]):
            PATCH.append((aid, b, Tn[:, :, off:off + b.shape[1]]))
            off += b.shape[1]
    for k, b in enumerate(B["c"]):
        PATCH.append(("c", b, Z["field_c"][:, 0, :, 2 * k:2 * k + 2]))

    def draw(ax, win, edges=False, grey_shell=False):
        ax.set_facecolor("#0b1017")
        for aid, b, cube in PATCH:
            n = b.nodes
            q = np.stack([n[:-1, :-1], n[1:, :-1], n[1:, 1:], n[:-1, 1:]], axis=2)[~b.blanked]
            if aid == "c" and grey_shell:
                ax.add_collection(PolyCollection(q, facecolors=(0.78, 0.80, 0.84, 1.0),
                                                 edgecolors="#2b2f36", linewidths=0.4, zorder=3))
                continue
            ax.add_collection(PolyCollection(q, array=cube[-1][~b.blanked] * SC[aid].T,
                                             cmap="inferno", norm=LogNorm(150.0, 3500.0),
                                             edgecolors=("#9aa3ad" if edges else "face"),
                                             linewidths=(0.25 if edges else 0.3), zorder=2))
        ax.set_xlim(win[0], win[1]); ax.set_ylim(win[2], win[3]); ax.set_aspect("equal")

    def cover(ax, cv, alpha):
        rgba = np.zeros(cv["cnt"].shape + (4,))
        rgba[(cv["cnt"] == 0) & ~cv["unm"]] = [1.0, 0.1, 0.85, alpha]
        rgba[cv["cnt"] >= 2] = [0.1, 0.95, 1.0, alpha]
        zs, ys = cv["zs"], cv["ys"]
        dz, dy = zs[1] - zs[0], ys[1] - ys[0]
        ax.imshow(np.transpose(rgba, (1, 0, 2)), origin="lower", interpolation="nearest",
                  extent=[zs[0] - dz / 2, zs[-1] + dz / 2, ys[0] - dy / 2, ys[-1] + dy / 2],
                  zorder=6)

    fig, ax = plt.subplots(figsize=(13, 6.6))
    thr = (0.24, 0.56, 0.015, 0.145)
    draw(ax, thr, edges=True, grey_shell=True); cover(ax, cov_e, 0.95)
    zz = np.linspace(thr[0], thr[1], 800)
    ax.plot(zz, con.nozzle_inner(zz, geo), color="#39ff88", lw=1.4, ls="--", zorder=8,
            label="nozzle wall h_in(z), as configured")
    ax.plot(zz, con.shell_outer(zz, geo), color="#ffd54a", lw=1.4, ls="--", zorder=8,
            label="airframe outer surface h_in + 8 mm")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.85)
    ax.set_title("AFTER. Nozzle throat, upper half, coarsen = 4: the airframe and d now have nodes at "
                 "both kinks, so every block traces the\nsame contour. Magenta (no block) and cyan "
                 "(two blocks) would show here; there are none.", fontsize=10, loc="left")
    ax.set_xlabel("z [m]"); ax.set_ylabel("y [m]")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "seam_throat.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 9.2))
    pl = (0.45, 3.02, -1.05, 1.05)
    draw(ax, pl, edges=True); cover(ax, cov_f, 0.75)
    for yb in (geo.plume_halfwidth, -geo.plume_halfwidth):
        ax.axhline(yb, color="#39ff88", lw=1.2, ls="--", zorder=8)
    ax.set_title("AFTER. Plume f and wake g, coarsen = 4: g's hole is exactly f's |y| <= 0.6 on node "
                 "lines both blocks share,\nand g sees f's own edge rows in its own gas model.",
                 fontsize=10, loc="left")
    ax.set_xlabel("z [m]"); ax.set_ylabel("y [m]")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "seam_plume_edge.png"), dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 7.4))
    eng = (-0.15, 1.35, -0.40, 0.40)
    draw(ax, eng); cover(ax, cov_e, 0.9)
    ax.set_title("AFTER. The last frame in kelvin, one scale for every block (t = %.3f s)"
                 % Z["t"][-1], fontsize=10.5, loc="left")
    fig.colorbar(plt.cm.ScalarMappable(norm=LogNorm(150.0, 3500.0), cmap="inferno"), ax=ax,
                 fraction=0.025, pad=0.01, label="T [K] (log)")
    ax.set_xlabel("z [m] (downstream; the nose is at z = -4)"); ax.set_ylabel("y [m]")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "seam_same_frame.png"), dpi=110)
    plt.close(fig)
    print("wrote seam_throat.png, seam_plume_edge.png, seam_same_frame.png in %s" % outdir)


if __name__ == "__main__":
    raise SystemExit(main())
