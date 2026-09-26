r"""W321 seam audit -- is the marched rocket wrong in the PICTURE, the MESH, or the COUPLER?

The episode page (`w321_episode_viz_data.py` -> the "Rocket Ascent Episode"
artifact) looks broken at every seam. That can mean three different things, and
they need three different repairs, so this script separates them and measures
each against the build repo's own blocks and the stored episode:

1.  **The picture.** `snapshot` divides every gas agent by its OWN reference
    set (`normalize.refs_for_episode`): T_ref is 3400 K for a/b/e, the ideal
    exit temperature for f, the ambient temperature for d/g. A page that puts
    those ratios on one colour ramp draws a seam between families as a colour
    jump whatever the physics does.
2.  **The meshes.** The shell and `d` have no node at either nozzle kink at any
    resolution, so each cuts its own chord across the throat; subsampling
    (`generate._coarsen_block`, `nodes[::k]`) makes those chords four times
    longer. Subsampling the blank mask (`blanked[::k]`) moves `g`'s plume hole.
3.  **The coupler.** `generate.CoupledEpisode` pieces the solvers together.
    Where it maps one block's wall onto another by array index, loads a panel
    through the wrong face, integrates a load against an index-oriented normal
    (W309's pattern), or hands a conserved state to a block with a different
    gas model, the solvers integrate the wrong inputs correctly.

Nothing here edits the build repo. Every number is read from it or from
the episode it audits.

**This is the PRE-FIX audit, and it only means something against the pre-fix
code.** Tier 86 fixed every defect it found; `scripts/w323_seams_fixed.py` is
the post-fix verification, with gates. Section 3 re-derives the OLD coupler's
index map and normal convention from their formulas, so pointed at the fixed
build repo it would print the old defects beside the new geometry -- it refuses
instead. To reproduce the audit: export the build repo at 0a407b7
(``git archive 0a407b7 src``), point ``ATLAS_BUILD_REPO`` at it, and run on the
preserved pre-fix episode.

    ATLAS_BUILD_REPO=<0a407b7 export> python scripts/w321_seam_audit.py [--figures]
        # -> out/w321_prefix/seam_audit.json (+ three PNGs there)
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
import types

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from atlas.cases import rocket_experts as RE
from atlas.cases.thermal_seam import build_repo


def _retry(fn, attempts=40, pause=0.25):
    """OneDrive holds a just-written file open for a moment; retry, do not lose it."""
    for k in range(attempts):
        try:
            return fn()
        except PermissionError:
            if k == attempts - 1:
                raise
            time.sleep(pause)


def _mods():
    RE.load_solvers()
    names = ("data.generate", "data.sweep", "config", "geometry.contours",
             "solvers.thermo", "solvers.atmosphere", "solvers.grid")
    return [importlib.import_module("atlas_build_solvers." + n) for n in names]


def _git(*args):
    try:
        return subprocess.run(["git", "-C", build_repo(), *args], capture_output=True,
                              text=True, encoding="utf-8", timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _poly(z, y, zz):
    return np.interp(zz, np.asarray(z, float), np.asarray(y, float), left=np.nan, right=np.nan)


def _quads(blk):
    n = blk.nodes
    q = np.stack([n[:-1, :-1], n[1:, :-1], n[1:, 1:], n[:-1, 1:]], axis=2)
    return q[~blk.blanked]


def _raster(patches, unmodelled, z0, z1, y0, y1, dx):
    """How many blocks claim each point: 0 is a gap, 2+ an overlap."""
    zs = np.arange(z0 + dx / 2, z1, dx)
    ys = np.arange(y0 + dx / 2, y1, dx)
    cnt = np.zeros((zs.size, ys.size), np.int16)
    for _, blk in patches:
        for q in _quads(blk):
            i0, i1 = np.searchsorted(zs, q[:, 0].min()), np.searchsorted(zs, q[:, 0].max())
            j0, j1 = np.searchsorted(ys, q[:, 1].min()), np.searchsorted(ys, q[:, 1].max())
            if i1 <= i0 or j1 <= j0:
                continue
            PZ, PY = np.meshgrid(zs[i0:i1], ys[j0:j1], indexing="ij")
            inside = np.ones(PZ.shape, bool)
            for k in range(4):
                (az, ay), (bz, by), (cz, cy) = q[k], q[(k + 1) % 4], q[(k + 2) % 4]
                side = np.sign((bz - az) * (cy - ay) - (by - ay) * (cz - az))
                inside &= ((bz - az) * (PY - ay) - (by - ay) * (PZ - az)) * side >= 0
            cnt[i0:i1, j0:j1] += inside
    ZZ, YY = np.meshgrid(zs, ys, indexing="ij")
    unm = np.zeros(ZZ.shape, bool)
    for r in unmodelled:
        unm |= (ZZ >= r.z[0]) & (ZZ <= r.z[1]) & (np.abs(YY) <= r.y_halfwidth)
    return dict(zs=zs, ys=ys, cnt=cnt, unm=unm)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="out/w321_prefix")
    ap.add_argument("--out", default="out/w321_prefix/seam_audit.json")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args(argv)

    gen, sw, cfgm, con, thermo, atm, grid = _mods()
    if hasattr(gen.CoupledEpisode, "_shell_inner_loads"):
        print("this build repo has the Tier 86 seam fixes; this script audits the "
              "PRE-fix coupler (see its docstring). The post-fix check is "
              "scripts/w323_seams_fixed.py.")
        return 2
    cfg = cfgm.load_config()
    geo = cfg.geometry
    with open(os.path.join(a.run, "episode.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    Z = np.load(os.path.join(a.run, "episode.npz"))
    pt = sw.corner_cases()[meta["case"]]
    ep = gen.CoupledEpisode(gen.EpisodeSpec(point=pt, n_macro=1, dt_macro=meta["dt_macro"],
                                            coarsen=meta["coarsen"]), cfg)
    B, SC = ep.blocks, ep.scales.per_agent
    R = dict(build_repo=dict(head=_git("rev-parse", "--short", "HEAD").strip(),
                             branch=_git("rev-parse", "--abbrev-ref", "HEAD").strip()),
             coarsen=meta["coarsen"])

    print("=" * 84)
    print("W321 seam audit -- picture, mesh, or coupler?")
    print("=" * 84)
    print("  build repo %s on %s; episode coarsen = %d"
          % (R["build_repo"]["head"], R["build_repo"]["branch"], meta["coarsen"]))

    # ------------------------------------------------------------ 1. picture
    print("\n--- 1. the picture: one colour ramp over per-agent references")
    R["T_ref"] = {k: float(SC[k].T) for k in "abedfgc"}
    for k in "abedfg":
        names = list(cfg.agent(k).fields)
        Tn = Z["field_%s" % k][-1, names.index("T")]
        print("  %s  T_ref %7.1f K   T/T_ref [%6.3f, %6.3f]  ->  T [%7.1f, %7.1f] K"
              % (k, SC[k].T, Tn.min(), Tn.max(), Tn.min() * SC[k].T, Tn.max() * SC[k].T))
    print("  so 1.0 on the ramp is 3400 K in the chamber and %.1f K in the air"
          % SC["d"].T)

    # ------------------------------------------------------------ 2. meshes
    print("\n--- 2a. the meshes: agent g's plume hole against agent f")
    g, f = B["g"][0], B["f"][0]
    bl = g.blanked[0]
    js = np.flatnonzero(bl)
    yg = g.nodes[0, :, 1]
    ycc = 0.5 * (yg[:-1] + yg[1:])
    hole = [float(yg[js[0]]), float(yg[js[-1] + 1])]
    fsp = [float(f.nodes[0, 0, 1]), float(f.nodes[0, -1, 1])]
    fine = grid.build_blocks(cfg.agent("g"), cfg)[0]
    fj, fy = np.flatnonzero(fine.blanked[0]), fine.nodes[0, :, 1]
    R["g_hole"] = dict(hole=hole, f=fsp, full_res=[float(fy[fj[0]]), float(fy[fj[-1] + 1])],
                       overlap_below=hole[0] - fsp[0], gap_above=hole[1] - fsp[1],
                       f_jmin_reads_g_cell_y=float(ycc[js[0] - 1]))
    print("  hole [%.4f, %.4f] vs f [%.4f, %.4f]: %.3f m overlap below, %.3f m gap above"
          % (hole[0], hole[1], fsp[0], fsp[1], hole[0] - fsp[0], hole[1] - fsp[1]))
    print("  full resolution: [%.5f, %.5f] (symmetric)" % tuple(R["g_hole"]["full_res"]))
    print("  f's jmin BC reads g's cell at y = %.4f -- inside f's own domain"
          % R["g_hole"]["f_jmin_reads_g_cell_y"])

    print("\n--- 2b. the meshes: gas wall / shell / d at the nozzle kinks (upper side)")
    zz = np.linspace(0.12, geo.z_exit, 5801)
    walls = {}
    for label, blocks in (("coarse", B), ("full", {x.id: grid.build_blocks(x, cfg)
                                                    for x in cfg.agents})):
        gz = np.concatenate([blocks["b"][0].nodes[:, -1, 0], blocks["e"][0].nodes[1:, -1, 0]])
        gy = np.concatenate([blocks["b"][0].nodes[:, -1, 1], blocks["e"][0].nodes[1:, -1, 1]])
        c1, d1 = blocks["c"][1], blocks["d"][1]
        gi = _poly(gz, gy, zz)
        si = _poly(c1.nodes[:, 0, 0], c1.nodes[:, 0, 1], zz)
        so = _poly(c1.nodes[:, -1, 0], c1.nodes[:, -1, 1], zz)
        di = _poly(d1.nodes[:, 0, 0], d1.nodes[:, 0, 1], zz)
        w = dict(gas_shell_gap=float(np.nanmax(si - gi)), gas_shell_overlap=float(-np.nanmin(si - gi)),
                 shell_d_gap=float(np.nanmax(di - so)), shell_d_overlap=float(-np.nanmin(di - so)),
                 kink_nodes={"gas": [bool(np.any(np.isclose(gz, k))) for k in (0.30, 0.40)],
                             "shell": [bool(np.any(np.isclose(c1.nodes[:, 0, 0], k))) for k in (0.30, 0.40)],
                             "d": [bool(np.any(np.isclose(d1.nodes[:, 0, 0], k))) for k in (0.30, 0.40)]})
        walls[label] = w
        print("  %-6s gas|shell gap %.4f overlap %.4f   shell|d gap %.4f overlap %.4f   "
              "(kink nodes 0.30/0.40: gas %s shell %s d %s)"
              % (label, w["gas_shell_gap"], w["gas_shell_overlap"], w["shell_d_gap"],
                 w["shell_d_overlap"], w["kink_nodes"]["gas"], w["kink_nodes"]["shell"],
                 w["kink_nodes"]["d"]))
    print("  the shell is %.3f m thick" % geo.shell_thickness)
    R["walls"] = walls

    # atlas-0.1's D2 rounded throat, if that branch is reachable: does it close the gaps?
    src = _git("show", "atlas-0.1:src/atlas/geometry/contours.py")
    if "throat_arc" in src:
        mod = types.ModuleType("_atlas01_contours")
        sys.modules[mod.__name__] = mod
        exec(compile(src.replace("from ..config import GeometryCfg", "GeometryCfg = object"),
                     "atlas-0.1:contours.py", "exec"), mod.__dict__)
        g01 = types.SimpleNamespace(**{k: getattr(geo, k) for k in (
            "chamber_halfheight", "throat_halfheight", "exit_halfheight", "z_converge",
            "z_diverge", "z_throat", "z_exit", "shell_thickness", "shell_z")},
            throat_round_radius=0.05)
        rounded = {}
        for label, k in (("coarse", meta["coarsen"]), ("full", 1)):
            nb, ne = cfg.agent("b").grid[0], cfg.agent("e").grid[0]
            nc, nd = cfg.agent("c").grid[0], cfg.agent("d").grid[0]
            zb = np.concatenate([np.linspace(0.12, geo.z_throat, nb + 1)[::k],
                                 np.linspace(geo.z_throat, geo.z_exit, ne + 1)[::k][1:]])
            zc = np.linspace(geo.shell_z[0], geo.shell_z[1], nc + 1)[::k]
            zd = np.linspace(geo.atmos_front_z[0], geo.atmos_front_z[1], nd + 1)[::k]
            h = lambda z: mod.nozzle_inner(np.asarray(z, float), g01)  # noqa: E731
            gi = _poly(zb, h(zb), zz)
            si = _poly(zc, h(zc), zz)
            di = _poly(zd, h(np.maximum(zd, geo.shell_z[0])) + geo.shell_thickness, zz)
            so = si + geo.shell_thickness
            rounded[label] = dict(gas_shell_gap=float(np.nanmax(si - gi)),
                                  gas_shell_overlap=float(-np.nanmin(si - gi)),
                                  shell_d_gap=float(np.nanmax(di - so)),
                                  shell_d_overlap=float(-np.nanmin(di - so)))
            print("  %-6s with atlas-0.1's rounded throat (R = 0.05): gas|shell gap %.4f "
                  "overlap %.4f   shell|d gap %.4f overlap %.4f"
                  % (label, *rounded[label].values()))
        R["walls_rounded_throat_atlas01"] = rounded

    print("\n--- 2c. the meshes: coverage raster (0 = no block, 2+ = two blocks)")
    PATCHES = [("a", B["a"][0]), ("b", B["b"][0]), ("e", B["e"][0]), ("d0", B["d"][0]),
               ("d1", B["d"][1]), ("f", B["f"][0]), ("g", B["g"][0]),
               ("c0", B["c"][0]), ("c1", B["c"][1])]
    cov_e = _raster(PATCHES, cfg.unmodelled, -0.1, 1.0, -0.2, 0.2, 5e-4)
    cov_f = _raster(PATCHES, cfg.unmodelled, geo.atmos_front_z[0], geo.plume_z[1],
                    -geo.farfield_halfwidth, geo.farfield_halfwidth, 5e-3)
    R["coverage"] = {}
    for label, cv, dx in (("engine_window_0.5mm", cov_e, 5e-4), ("full_domain_5mm", cov_f, 5e-3)):
        gap = float(((cv["cnt"] == 0) & ~cv["unm"]).sum() * dx * dx)
        ovl = float((cv["cnt"] >= 2).sum() * dx * dx)
        R["coverage"][label] = dict(gap_m2=gap, overlap_m2=ovl)
        print("  %-20s gap %.5f m^2   overlap %.5f m^2" % (label, gap, ovl))

    # ------------------------------------------------------------ 3. coupler
    print("\n--- 3a. the coupler: where the shell's wall data comes from (_step_structure)")
    c1 = B["c"][1]
    cz = 0.5 * (c1.nodes[:-1, 0, 0] + c1.nodes[1:, 0, 0])
    ni = cz.size
    bz = B["b"][0].centroid[:, -1, 0]
    dz = B["d"][1].centroid[:, 0, 0]
    src_b = np.interp(np.linspace(0, 1, ni), np.linspace(0, 1, bz.size), bz)
    src_d = np.interp(np.linspace(0, 1, ni), np.linspace(0, 1, dz.size), dz)
    outside = float(np.mean((cz < 0.12) | (cz > geo.z_throat)))
    R["shell_map"] = dict(stations=int(ni), inner_max_misplacement=float(np.abs(cz - src_b).max()),
                          outer_max_misplacement=float(np.abs(cz - src_d).max()),
                          fraction_outside_b=outside,
                          rows=[[float(cz[m]), float(src_b[m]), float(src_d[m])]
                                for m in (0, 14, 28, 42, ni - 1)])
    print("  %d shell stations over z [%.2f, %.2f] take b's wall (z [%.2f, %.2f]) and d's (z [%.2f, %.2f])"
          % (ni, cz[0], cz[-1], bz[0], bz[-1], dz[0], dz[-1]))
    print("  by array index: %.0f%% of stations get chamber data from a wall they are not on;"
          " worst %.2f m off inside, %.2f m outside"
          % (100 * outside, R["shell_map"]["inner_max_misplacement"],
             R["shell_map"]["outer_max_misplacement"]))

    print("\n--- 3b. the coupler: which face ThermoStruct2D loads as 'inner'")
    rc = SC["c"]
    fc = Z["field_c"]                                   # (n, 6, 58, 4): panel 0 cols 0-1
    faces = []
    for k, blk in enumerate(B["c"]):
        y0, y1 = np.abs(blk.nodes[:, 0, 1]).mean(), np.abs(blk.nodes[:, -1, 1]).mean()
        faces.append("gas side" if y0 < y1 else "ATMOSPHERE side")
        print("  panel %d: j = 0 (loaded as 'inner': chamber heat and pressure) is the %s"
              % (k, faces[-1]))
    T = fc[-1, 0] * rc.T
    uy = fc[-1, 2] * rc.L
    u0, u1 = uy[:, 0:2].mean(axis=1), uy[:, 2:4].mean(axis=1)
    ident = float(np.abs(u0 - u1).max() / np.abs(u1).max())
    mirror = float(np.abs(u0 + u1).max() / np.abs(u1).max())
    R["shell_orientation"] = dict(inner_face=faces, T_cols_K=[float(T[:, c].mean()) for c in range(4)],
                                  uy_nose_mid_tail=[float(u1[0]), float(u1[ni // 2]), float(u1[-1])],
                                  uy_max_abs=float(np.abs(u1).max()),
                                  panels_identical_rel=ident, panels_mirror_rel=mirror)
    print("  T [K] through thickness: panel 0 %.2f (outer) %.2f (inner) | panel 1 %.2f (inner) %.2f (outer)"
          % tuple(R["shell_orientation"]["T_cols_K"]))
    print("  uy [m] nose/mid/tail: %+.3f %+.3f %+.3f; max |uy| %.3f m on a body %.3f m wide"
          % (*R["shell_orientation"]["uy_nose_mid_tail"], R["shell_orientation"]["uy_max_abs"],
             2 * (geo.chamber_halfheight + geo.shell_thickness)))
    print("  panel 0 vs panel 1: |u0 - u1|/max = %.2e (identical)   |u0 + u1|/max = %.2f "
          "(a symmetric load would make THIS zero)" % (ident, mirror))

    print("\n--- 3c. the coupler: the aero load against index-oriented normals (W309's pattern)")
    _, p_inf, _, _ = (float(x) for x in atm.properties(float(Z["rigid"][-2, 1])))
    pd = Z["field_d"][-1, 3] * SC["d"].p
    ny = B["d"][0].shape[1]
    aero = {}
    for conv in ("as_coded", "outward"):
        ax_ = nm_ = 0.0
        for k, blk in enumerate(B["d"]):
            j = 0 if k == 1 else -1
            pw = pd[:, ny * k:ny * (k + 1)][:, j]
            nrm, L = blk.n_j[:, j], blk.a_j[:, j]
            s = -1.0 if (conv == "outward" and k == 0) else 1.0
            ax_ += float(np.sum(-(pw - p_inf) * s * nrm[:, 0] * L))
            nm_ += float(np.sum(-(pw - p_inf) * s * nrm[:, 1] * L))
        aero[conv] = dict(axial=ax_, normal=nm_)
    th = float(Z["rigid"][-1, 2])
    recomputed = aero["as_coded"]["axial"] * np.cos(th) - aero["as_coded"]["normal"] * np.sin(th)
    aero["stored_F_aero_last"] = [float(Z["loads"][-1, 2]), float(Z["loads"][-1, 3])]
    aero["recomputed_F_aero_x"] = float(recomputed)
    R["aero"] = aero
    print("  as coded: axial %.3f N/m, normal %.3f N/m  (stored F_aero = (%.3f, %.2e), recomputed x %.3f)"
          % (aero["as_coded"]["axial"], aero["as_coded"]["normal"], *aero["stored_F_aero_last"], recomputed))
    print("  outward : axial %.3f N/m, normal %.3f N/m" % (aero["outward"]["axial"],
                                                           aero["outward"]["normal"]))
    print("  and _compute_loads maps body +z to (cos th, sin th) for thrust AND aero, while")
    print("  the nose is at body -z: flipping only the thrust sign (W322) would leave the")
    print("  corrected drag pushing the vehicle forward")

    print("\n--- 3d. the coupler: conserved states across two gas models at f's boundaries")
    T_inf, p0, rho_inf, _ = (float(x) for x in atm.properties(pt.h0))
    Wa = thermo.cons_to_prim(thermo.prim_to_cons(np.array([[rho_inf, ep.v_flight, 0.0, p0]]),
                                                 atm.GAMMA_AIR), pt.gamma_gas)[0]
    rf = SC["f"]
    Ff = Z["field_f"][-1]
    Wf = np.stack([Ff[0] * rf.rho, Ff[1] * rf.u, Ff[2] * rf.u, Ff[3] * rf.p], axis=-1)
    Um = thermo.prim_to_cons(Wf, pt.gamma_gas).mean(axis=1)          # g's hole_state
    Wg = thermo.cons_to_prim(Um, atm.GAMMA_AIR)
    Wt = thermo.cons_to_prim(Um, pt.gamma_gas)
    T_hole_air = Wg[:, 3] / (Wg[:, 0] * atm.R_AIR)
    T_mean = Wt[:, 3] / (Wt[:, 0] * pt.R_gas)
    T_edge = Wf[:, -1, 3] / (Wf[:, -1, 0] * pt.R_gas)
    T0_air = T_inf * (1.0 + 0.5 * (atm.GAMMA_AIR - 1.0) * pt.M_inf ** 2)
    Tg = Z["field_g"][-1, 4] * SC["g"].T
    R["gas_model"] = dict(freestream_T=T_inf, freestream_T_read_by_f=float(Wa[3] / (Wa[0] * pt.R_gas)),
                          freestream_p=p0, freestream_p_read_by_f=float(Wa[3]),
                          plume_edge_T=[float(T_edge.min()), float(T_edge.max())],
                          plume_mean_T=[float(T_mean.min()), float(T_mean.max())],
                          hole_T_read_by_g=[float(T_hole_air.min()), float(T_hole_air.max())],
                          hole_p_factor=float(np.median(Wg[:, 3] / Wt[:, 3])),
                          air_T0=float(T0_air), g_T_max=float(Tg[:, ~bl].max()))
    gm = R["gas_model"]
    print("  freestream air %.1f K / %.0f Pa, read by f's gas model: %.1f K / %.0f Pa"
          % (T_inf, p0, gm["freestream_T_read_by_f"], gm["freestream_p_read_by_f"]))
    print("  plume edge %.0f-%.0f K; the coupler hands g the plume MEAN %.0f-%.0f K; read as air %.0f-%.0f K at %.2fx p"
          % (*gm["plume_edge_T"], *gm["plume_mean_T"], *gm["hole_T_read_by_g"], gm["hole_p_factor"]))
    print("  g's air reaches %.0f K; its stagnation temperature is %.0f K, which nothing but"
          " that seam can exceed" % (gm["g_T_max"], gm["air_T0"]))

    print("\n--- 4. is the atlas-0.1 branch any different on these lines?")
    gsrc = _git("show", "atlas-0.1:src/atlas/data/generate.py")
    pats = {"index-stretched shell data": "np.interp(np.linspace(0, 1, ni)",
            "default ShellMesh for both panels": "ShellMesh(b.nodes) for b in shell",
            "plume hole = transverse mean": "self.U[\"f\"][0].mean(axis=1)",
            "subsampling coarsen": "b.nodes[::k, ::k]",
            "inverted body->inertial map": "-thrust * np.cos(th)",
            "air-gamma inlet handed to f": "prim_to_cons(free, atmosphere.GAMMA_AIR)"}
    R["atlas01_has"] = {k: (v in gsrc) if gsrc else None for k, v in pats.items()}
    for k, v in R["atlas01_has"].items():
        print("  %-36s %s" % (k, {True: "present", False: "absent", None: "branch unreachable"}[v]))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)

    def _write():
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(R, fh, indent=1)
    _retry(_write)
    print("\nwrote %s" % a.out)

    if a.figures:
        _figures(os.path.dirname(os.path.abspath(a.out)), cfg, geo, con, gen, Z, B, SC,
                 cov_e, cov_f)
    return 0


def _figures(outdir, cfg, geo, con, gen, Z, B, SC, cov_e, cov_f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

    PATCH = []
    for aid in gen.GAS_AGENTS:
        Tn = Z["field_%s" % aid][:, list(cfg.agent(aid).fields).index("T")]
        off = 0
        for k, b in enumerate(B[aid]):
            PATCH.append(("%s%d" % (aid, k) if len(B[aid]) > 1 else aid, aid, b,
                          Tn[:, :, off:off + b.shape[1]]))
            off += b.shape[1]
    for k, b in enumerate(B["c"]):
        PATCH.append(("c%d" % k, "c", b, Z["field_c"][:, 0, :, 2 * k:2 * k + 2]))
    # the page's own sequential ramp and its shared-scale range
    page = LinearSegmentedColormap.from_list("page", np.array(
        [[10, 19, 48], [24, 66, 106], [22, 120, 124], [96, 164, 92], [208, 183, 70],
         [248, 236, 186]]) / 255.0)
    glo = min(float(p[3][:, ~p[2].blanked].min()) for p in PATCH if p[1] != "c")
    ghi = max(float(p[3][:, ~p[2].blanked].max()) for p in PATCH if p[1] != "c")
    LAB = {"a": "a injector", "b": "b chamber", "e": "e nozzle", "f": "f PLUME", "g": "g WAKE"}

    def draw(ax, mode, win, edges=False, grey_shell=False):
        ax.set_facecolor("#0b1017")
        for pid, aid, b, cube in PATCH:
            n = b.nodes
            q = np.stack([n[:-1, :-1], n[1:, :-1], n[1:, 1:], n[:-1, 1:]], axis=2)[~b.blanked]
            v = cube[-1][~b.blanked]
            if aid == "c" and mode == "page":
                continue                  # the page never colours the shell by a gas field
            if aid == "c" and grey_shell:
                ax.add_collection(PolyCollection(q, facecolors=(0.78, 0.80, 0.84, 1.0),
                                                 edgecolors="#2b2f36", linewidths=0.4, zorder=3))
                continue
            if mode == "page":
                pc = PolyCollection(q, array=v, cmap=page, norm=Normalize(glo, ghi),
                                    edgecolors="face", linewidths=0.3, zorder=2)
            else:
                pc = PolyCollection(q, array=v * SC[aid].T, cmap="inferno",
                                    norm=LogNorm(150.0, 3500.0),
                                    edgecolors=("#9aa3ad" if edges else "face"),
                                    linewidths=(0.25 if edges else 0.3), zorder=2)
            ax.add_collection(pc)
        ax.set_xlim(win[0], win[1]); ax.set_ylim(win[2], win[3]); ax.set_aspect("equal")

    def cover(ax, cv, alpha):
        gap = (cv["cnt"] == 0) & ~cv["unm"]
        rgba = np.zeros(cv["cnt"].shape + (4,))
        rgba[gap] = [1.0, 0.1, 0.85, alpha]
        rgba[cv["cnt"] >= 2] = [0.1, 0.95, 1.0, alpha]
        zs, ys = cv["zs"], cv["ys"]
        dz, dy = zs[1] - zs[0], ys[1] - ys[0]
        ax.imshow(np.transpose(rgba, (1, 0, 2)), origin="lower", interpolation="nearest",
                  extent=[zs[0] - dz / 2, zs[-1] + dz / 2, ys[0] - dy / 2, ys[-1] + dy / 2],
                  zorder=6)

    def label(ax, ids, win):
        for pid, aid, b, _ in PATCH:
            if pid not in ids:
                continue
            c = b.centroid[~b.blanked]
            zc = float(np.median(c[:, 0]))
            yc = {"f": 0.0, "g": 0.9 * win[3]}.get(pid, float(np.median(c[:, 1])))
            if win[0] < zc < win[1] and win[2] < yc < win[3]:
                ax.text(zc, yc, LAB.get(pid, pid), color="#eef2f7", fontsize=8.5, ha="center",
                        va="center", family="monospace", zorder=9,
                        bbox=dict(boxstyle="round,pad=0.25", fc=(0.03, 0.05, 0.08, 0.72), ec="none"))

    eng = (-0.15, 1.35, -0.40, 0.40)
    fig, axs = plt.subplots(2, 1, figsize=(13, 14.2))
    draw(axs[0], "page", eng); label(axs[0], ("a", "b", "e", "f"), eng)
    axs[0].set_title("As the page colours it: T / T_ref with T_ref = 3400 K (a, b, e), %.0f K (f), "
                     "%.0f K (d, g) -- one ramp, three units" % (SC["f"].T, SC["d"].T),
                     fontsize=10.5, loc="left")
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(glo, ghi), cmap=page), ax=axs[0],
                 fraction=0.025, pad=0.01, label="T / T_ref (per agent)")
    draw(axs[1], "kelvin", eng); label(axs[1], ("a", "b", "e", "f"), eng); cover(axs[1], cov_e, 0.9)
    axs[1].set_title("The same frame in kelvin, one scale. Magenta: no block covers it; cyan: two do "
                     "(t = %.3f s)" % Z["t"][-1], fontsize=10.5, loc="left")
    fig.colorbar(plt.cm.ScalarMappable(norm=LogNorm(150.0, 3500.0), cmap="inferno"), ax=axs[1],
                 fraction=0.025, pad=0.01, label="T [K] (log)")
    for ax in axs:
        ax.set_xlabel("z [m] (downstream; the nose is at z = -4)"); ax.set_ylabel("y [m]")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "seam_same_frame.png"), dpi=110)
    plt.close(fig)

    thr = (0.24, 0.56, 0.015, 0.145)
    fig, ax = plt.subplots(figsize=(13, 6.6))
    draw(ax, "kelvin", thr, edges=True, grey_shell=True); cover(ax, cov_e, 0.95)
    zz = np.linspace(thr[0], thr[1], 800)
    ax.plot(zz, con.nozzle_inner(zz, geo), color="#39ff88", lw=1.4, ls="--", zorder=8,
            label="nozzle wall h_in(z), as configured")
    ax.plot(zz, con.shell_outer(zz, geo), color="#ffd54a", lw=1.4, ls="--", zorder=8,
            label="airframe outer surface h_in + 8 mm, as configured")
    for zk in (geo.z_converge[0], geo.z_throat):
        ax.axvline(zk, color="w", lw=0.6, ls=":", zorder=7)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.85)
    ax.set_title("Nozzle throat, upper half, coarsen = 4. Gas cells have nodes every 10 mm and hit both "
                 "kinks; the shell (81 mm) and d (124 mm)\nhave a node at neither, so each cuts its own "
                 "chord. Grey: airframe. Magenta: no block. Cyan: two blocks.", fontsize=10, loc="left")
    ax.set_xlabel("z [m]"); ax.set_ylabel("y [m]")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "seam_throat.png"), dpi=120)
    plt.close(fig)

    pl = (0.45, 3.02, -1.05, 1.05)
    fig, ax = plt.subplots(figsize=(13, 9.2))
    draw(ax, "kelvin", pl, edges=True); cover(ax, cov_f, 0.75)
    for yb in (geo.plume_halfwidth, -geo.plume_halfwidth):
        ax.axhline(yb, color="#39ff88", lw=1.2, ls="--", zorder=8)
    g = B["g"][0]
    js = np.flatnonzero(g.blanked[0])
    for yb in (g.nodes[0, js[0], 1], g.nodes[0, js[-1] + 1, 1]):
        ax.axhline(yb, color="#ffd54a", lw=1.2, ls=":", zorder=8)
    label(ax, ("f", "g"), pl)
    ax.set_title("Plume f (green dashes, |y| <= 0.6) against g's hole (yellow dots) after coarsen = 4: "
                 "0.10 m overlap below (cyan), 0.025 m\nuncovered above (magenta). g's air here is "
                 "1300-2200 K, above the 764 K it could reach without the g-f seam.", fontsize=10, loc="left")
    ax.set_xlabel("z [m]"); ax.set_ylabel("y [m]")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "seam_plume_edge.png"), dpi=110)
    plt.close(fig)
    print("wrote seam_same_frame.png, seam_throat.png, seam_plume_edge.png in %s" % outdir)


if __name__ == "__main__":
    raise SystemExit(main())
