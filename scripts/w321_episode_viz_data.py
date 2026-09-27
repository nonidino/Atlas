r"""Turn an episode into the record a page can draw -- every block, every field.

**Three things the first version of this got wrong, all of them findable only by
checking the stored shapes against the geometry.**

1.  Multi-block agents are stored CONCATENATED along the transverse axis.
    `field_d` is ``(n, 5, 46, 24)`` for two blocks of ``(46, 12)``, and
    `field_c` is ``(n, 6, 58, 4)`` for two shell panels of ``(58, 2)``. Taking
    ``blocks[-1]``'s geometry and the field's first columns draws the LOWER
    panel's data at the UPPER panel's coordinates. Each block gets its own
    patch here, sliced by its own ``ny``.
2.  The shell was never emitted at all -- so the airframe, which is the one
    solid body in the picture, was missing from a drawing of a rocket.
3.  Cell CENTRES cannot draw a curved mesh. The nozzle contour and the plume
    boundary are curves; drawing axis-aligned rectangles at cell centres
    reports them as steps. Node coordinates are emitted so the page can fill
    true quads.

4.  One colour ramp over per-agent NONDIMENSIONAL fields. `snapshot` divides
    each agent by its own reference set -- T_ref is 3400 K for a/b/e, the
    ideal exit temperature for f, the ambient for d/g -- so 1.0 was 3400 K in
    the chamber and 222 K in the air, drawn as the same colour, and every seam
    between agent families was a colour jump whatever the physics did. Every
    field is emitted DIMENSIONAL here, through the same frozen references the
    run used (`normalize.refs_for_episode` depends only on episode inputs, so
    the conversion is exact).
5.  `f` and `g` were labelled the wrong way round: `f` is the plume
    (``free_jet``, |y| <= 0.6), `g` the atmosphere-wake with the plume carved
    out of it.

**Quantisation.** Each (patch, field) is scaled to uint16 against its own min
and max and base64'd -- about 2 MB instead of 8 MB of JSON text, recoverable to
roughly 1e-4 relative, which is finer than a colour ramp can show and fine for
a hover readout. The page says so.

**Units.** SI throughout: K, Pa, kg/m^3, m/s; the airframe's displacement in m
and stress in Pa. The interface records are summarised per step as a
length-weighted mean and, for the flux channels, the integral over the seam.
"""
from __future__ import annotations

import argparse
import base64
import importlib
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from atlas.cases import rocket_experts as RE

GAS_BASE = ("rho", "u", "v", "p", "T")
SHELL_BASE = ("T", "ux", "uy", "s_zz", "s_yy", "s_zy")
#: Signed fields want a zero-centred diverging ramp; magnitudes want a
#: sequential one. Getting this wrong hides a sign change, which on `u` and `v`
#: is the difference between a recirculation and a jet.
SIGNED = {"u", "v", "ux", "uy", "s_zz", "s_yy", "s_zy"}
IFACE = ("mass_flux", "mom_z_flux", "mom_y_flux", "energy_flux",
         "heat_flux", "p_trace", "T_trace", "seg_length")
RIGID = ("x", "y", "theta", "vx", "vy", "omega", "m")
#: WORLD frame (x horizontal, y up) -- the order `episode.npz` stores them in
LOADS = ("Fx_thrust", "Fy_thrust", "Fx_aero", "Fy_aero")
#: BODY frame: axial is +z_body (towards the tail), normal is +y_body
LOADS_BODY = ("thrust_axial", "thrust_normal", "aero_axial", "aero_normal")

LABEL = {"a": "injector", "b": "chamber", "e": "nozzle",
         "d": "atmosphere", "f": "plume", "g": "wake", "c": "airframe"}
UNITS = {"rho": "kg/m^3", "u": "m/s", "v": "m/s", "p": "Pa", "T": "K", "speed": "m/s",
         "Y": "-", "ux": "m", "uy": "m", "s_zz": "Pa", "s_yy": "Pa", "s_zy": "Pa", "vm": "Pa"}
FLUX = ("mass_flux", "mom_z_flux", "mom_y_flux", "energy_flux", "heat_flux")


def dimensional(cube, names, ref, solid=False):
    """Undo `snapshot`'s nondimensionalisation with the run's own references."""
    out = np.array(cube, dtype=np.float64, copy=True)
    for i, nm in enumerate(names):
        if solid:
            s = {"T": ref.T, "ux": ref.L, "uy": ref.L, "s_zz": ref.p, "s_yy": ref.p,
                 "s_zy": ref.p}[nm]
        else:
            s = {"rho": ref.rho, "u": ref.u, "v": ref.u, "p": ref.p, "T": ref.T, "Y": 1.0}[nm]
        out[:, i] *= s
    return out


def q16(arr):
    """(base64 uint16, lo, hi) -- the ramp cannot show more and a readout wants
    more than uint8 gives."""
    a = np.asarray(arr, dtype=np.float64)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    q = np.clip(np.round((a - lo) / (hi - lo) * 65535.0), 0, 65535).astype("<u2")
    return base64.b64encode(q.tobytes()).decode("ascii"), lo, hi


def _read_json(path):
    """An optional companion record; the page shows what it has."""
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def mirror_series(run, cfg):
    """Per step, one run: the engine's worst pressure mirror asymmetry (a, b, e),
    the airframe's (the E1 gate's measure), and |side| / drag. None when the run
    is absent."""
    path = os.path.join(run, "episode.npz") if run else ""
    if not path or not os.path.exists(path):
        return None
    z = np.load(path)
    eng = []
    for s in range(z["t"].shape[0]):
        worst = 0.0
        for aid in ("a", "b", "e"):
            p = z["field_%s" % aid][s, list(cfg.agent(aid).fields).index("p")]
            worst = max(worst, float(np.abs(p - p[:, ::-1]).max() / max(np.abs(p).max(), 1e-300)))
        eng.append(worst)
    shell = []
    for s in range(z["t"].shape[0]):
        uy = z["field_c"][s, 2]
        lo, hi = uy[:, 0:2], uy[:, 2:4]
        sc = np.abs(hi).max()
        shell.append(float(np.abs(lo[:, ::-1] + hi).max() / sc) if sc > 0 else 0.0)
    side = None
    if "loads_body" in z.files:
        lb = z["loads_body"]
        side = [float(abs(r[3]) / max(abs(r[2]), 1e-300)) for r in lb]
    return dict(t=[round(float(v), 6) for v in z["t"]],
                engine=[float("%.3g" % v) for v in eng],
                shell=[float("%.3g" % v) for v in shell],
                side_over_drag=None if side is None else [float("%.3g" % v) for v in side])


def _cd_length(run):
    """The c-d record's total length at a run's last step."""
    path = os.path.join(run, "episode.npz") if run else ""
    if not path or not os.path.exists(path):
        return None
    return float(np.load(path)["iface_c-d"][-1, 7].sum())


def w332_experiments(folder):
    """scripts/w332_nozzle_symmetry.py's records: the nozzle's asymmetry at the
    end of each 5 ms macro step, per variant."""
    if not folder or not os.path.isdir(folder):
        return None
    out = []
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json"):
            continue
        r = _read_json(os.path.join(folder, name))
        e = [row["asym"][-1] for row in r["rows"] if row["agent"] == "e"]
        out.append(dict(variant=r["variant"], flux=r["flux"], wall=r["wall"],
                        viscous=r["viscous"], old_ghost=r["old_ghost"],
                        nozzle=[float("%.2g" % v) for v in e]))
    return out or None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="out/w321")
    ap.add_argument("--out", default="out/w321/viz.json")
    ap.add_argument("--before", default="out/w321_prefix/seam_audit.json",
                    help="the pre-fix audit, for the page's before/after table")
    ap.add_argument("--gates", default="out/w323_seams_fixed_w343.json",
                    help="the post-fix verification's gates, read against this march "
                         "(Tier 86's own reading of its march is out/w323_seams_fixed.json)")
    ap.add_argument("--before-episode", default="out/w321_prefix/episode.json",
                    help="the pre-fix run's summary, for the before/after table")
    ap.add_argument("--prefix-run", default="out/w321_prefix",
                    help="the first march, before any fix")
    ap.add_argument("--seams-only", default="out/w321_seams_only",
                    help="the first fixed march: seams rebuilt, W332/W333 not yet found")
    ap.add_argument("--w332", default="out/w332",
                    help="scripts/w332_nozzle_symmetry.py's records")
    ap.add_argument("--wall-mass", default="out/w332_wall_mass.json",
                    help="scripts/w332_wall_mass.py's record")
    ap.add_argument("--template", default="scripts/w321_episode_page_template.html")
    ap.add_argument("--page", default="out/w321/rocket-episode.html",
                    help="write the page with this data inlined ('' to skip)")
    a = ap.parse_args(argv)

    with open(os.path.join(a.run, "episode.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    z = np.load(os.path.join(a.run, "episode.npz"))

    RE.load_solvers()
    gen = importlib.import_module("atlas_build_solvers.data.generate")
    cfg = importlib.import_module("atlas_build_solvers.config").load_config()
    sw = importlib.import_module("atlas_build_solvers.data.sweep")

    spec = gen.EpisodeSpec(point=sw.corner_cases()[meta["case"]], n_macro=1,
                           dt_macro=meta["dt_macro"], coarsen=meta["coarsen"])
    ep = gen.CoupledEpisode(spec, cfg)

    patches = []

    def add(pid, agent, nodes, blanked, cube, names):
        """cube: (n_macro, n_fields, nz, ny) for THIS block only."""
        n = np.asarray(nodes, dtype=float)
        nz, ny = cube.shape[2], cube.shape[3]
        assert n.shape[0] == nz + 1 and n.shape[1] == ny + 1, (pid, n.shape, nz, ny)
        fields = {}
        for i, nm in enumerate(names):
            b64, lo, hi = q16(cube[:, i])
            fields[nm] = dict(d=b64, lo=lo, hi=hi)
        if set(("u", "v")) <= set(names):
            u = cube[:, names.index("u")]
            v = cube[:, names.index("v")]
            b64, lo, hi = q16(np.hypot(u, v))
            fields["speed"] = dict(d=b64, lo=lo, hi=hi)
        if set(("s_zz", "s_yy", "s_zy")) <= set(names):
            szz = cube[:, names.index("s_zz")]
            syy = cube[:, names.index("s_yy")]
            szy = cube[:, names.index("s_zy")]
            vm = np.sqrt(np.maximum(szz ** 2 - szz * syy + syy ** 2 + 3 * szy ** 2, 0.0))
            b64, lo, hi = q16(vm)
            fields["vm"] = dict(d=b64, lo=lo, hi=hi)
        patches.append(dict(
            id=pid, agent=agent, label=LABEL.get(agent, agent),
            nz=int(nz), ny=int(ny),
            zn=[round(float(x), 5) for x in n[..., 0].ravel()],
            yn=[round(float(x), 5) for x in n[..., 1].ravel()],
            blank=base64.b64encode(
                np.asarray(blanked, dtype=bool).ravel().astype("u1").tobytes()
            ).decode("ascii"),
            fields=fields))

    refs = ep.scales.per_agent
    for aid in gen.GAS_AGENTS:
        names = list(cfg.agent(aid).fields)
        cube = dimensional(z["field_%s" % aid], names, refs[aid])   # (n, nf, nz, NY)
        off = 0
        blocks = ep.blocks[aid]
        for k, b in enumerate(blocks):
            ny = int(b.shape[1])
            sub = cube[:, :, :, off:off + ny]
            off += ny
            add("%s%d" % (aid, k) if len(blocks) > 1 else aid, aid,
                b.nodes, b.blanked, sub, names)
        assert off == cube.shape[3], (aid, off, cube.shape)

    names = list(cfg.agent("c").fields)
    cube = dimensional(z["field_c"], names, refs["c"], solid=True)
    off = 0
    for k, m in enumerate(ep.shell_mesh):
        ny = int(m.shape[1])
        sub = cube[:, :, :, off:off + ny]
        off += ny
        add("c%d" % k, "c", m.nodes, np.zeros(m.shape, bool), sub, names)
    assert off == cube.shape[3], ("c", off, cube.shape)

    rigid, loads, t = z["rigid"], z["loads"], z["t"]
    iface = {}
    for k in z.files:
        if not k.startswith("iface_"):
            continue
        arr = z[k]                                    # (n, 8, n_cells)
        L = arr[:, IFACE.index("seg_length"), :]
        rec = {}
        for i, ch in enumerate(IFACE):
            if ch == "seg_length":
                continue
            tot = (arr[:, i, :] * L).sum(axis=1)
            rec[ch] = [float("%.6g" % x) for x in tot / np.maximum(L.sum(axis=1), 1e-300)]
            if ch in FLUX:
                rec[ch + "_total"] = [float("%.6g" % x) for x in tot]
        rec["length"] = float(L[-1].sum())
        iface[k[6:]] = rec

    zn = np.concatenate([np.asarray(p["zn"]) for p in patches])
    yn = np.concatenate([np.asarray(p["yn"]) for p in patches])
    out = dict(
        meta=meta,
        gas_fields=list(GAS_BASE) + ["speed"],
        shell_fields=list(SHELL_BASE) + ["vm"],
        signed=sorted(SIGNED),
        extent=dict(z0=float(zn.min()), z1=float(zn.max()),
                    y0=float(yn.min()), y1=float(yn.max())),
        units=UNITS,
        note=("Every field is in SI units, converted back from the episode's "
              "per-agent nondimensional storage through the run's own frozen "
              "reference scales. Field samples are uint16-quantised per patch "
              "against that patch's own range (about 1e-4 relative)."),
        t=[round(float(v), 6) for v in t],
        patches=patches,
        rigid={k: [round(float(v), 6) for v in rigid[:, i]]
               for i, k in enumerate(RIGID)},
        loads={k: [float("%.6g" % v) for v in loads[:, i]]
               for i, k in enumerate(LOADS)},
        loads_body=({k: [float("%.6g" % v) for v in z["loads_body"][:, i]]
                     for i, k in enumerate(LOADS_BODY)}
                    if "loads_body" in z.files else None),
        declared=sorted("%s-%s" % (e.src, e.dst) for e in cfg.edge_list),
        recorded=sorted("%s-%s" % (r.src, r.dst)
                        for r in getattr(cfg, "recorded_interfaces", ())),
        before=_read_json(a.before), gates=_read_json(a.gates),
        before_episode=_read_json(a.before_episode),
        seams_only=dict(meta=_read_json(os.path.join(a.seams_only, "episode.json")),
                        gates=_read_json(os.path.join(a.seams_only, "gates.json")),
                        cd_length=_cd_length(a.seams_only)),
        airframe_outer_face=float(sum(
            np.linalg.norm(np.diff(m.nodes[:, m.outer_j], axis=0), axis=-1).sum()
            for m in ep.shell_mesh)),
        wall_mass=_read_json(a.wall_mass),
        symmetry=dict(prefix=mirror_series(a.prefix_run, cfg),
                      seams_only=mirror_series(a.seams_only, cfg),
                      this=mirror_series(a.run, cfg)),
        w332=w332_experiments(a.w332),
        iface=iface)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))
    if a.page:
        with open(a.template, encoding="utf-8") as fh:
            tpl = fh.read()
        marker = "/*__EPISODE_DATA__*/"
        assert tpl.count(marker) == 1, "the template must carry the data marker once"
        # a JSON body inside <script> must not close the tag early
        blob = json.dumps(out, separators=(",", ":")).replace("</", "<\\/")
        with open(a.page, "w", encoding="utf-8") as fh:
            fh.write(tpl.replace(marker, blob))
        print("wrote %s (%.1f MB)" % (a.page, os.path.getsize(a.page) / 1e6))
    print("wrote %s (%.1f MB): %d frames, %d patches, %d seams"
          % (a.out, os.path.getsize(a.out) / 1e6, len(t), len(patches), len(iface)))
    print("  domain z [%.3f, %.3f]  y [%.3f, %.3f]"
          % (out["extent"]["z0"], out["extent"]["z1"],
             out["extent"]["y0"], out["extent"]["y1"]))
    for p in patches:
        print("    %-4s %-14s %2dx%-2d  fields %s"
              % (p["id"], p["label"], p["nz"], p["ny"],
                 ",".join(sorted(p["fields"]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
