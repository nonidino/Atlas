"""CS-S2 -- a bounded-receptive-field donor through Tier 40's instrument, unchanged.

[[substitution-campaign-checkpoint]] section 6 names branch (a) -- a foundation
model on a constrained expert class -- the one live route, and says the Tier 40
scripts "run against any object with a `boundary_response`".  This is that run,
on the one continuum candidate [[expert-donor-survey]] names: NeuberNet, whose
input IS a boundary displacement.

What is measured, and against what:

  1. **Along the seam.**  `w173_epsilon_halo.measure`, unchanged, on NeuberNet's
     own ring -- the material sensors of the circle |p| = 5 R_n -- per component
     pair, at three base states (tension on the elastic branch, tension on the
     plastic branch, tension plus torsion on the plastic branch).  The full
     coupled 3M x 3M operator is also truncated by NODE distance, its spectrum
     taken, and its Betti reciprocity measured.
  2. **Across the seam.**  Poke one sensor and profile the change in the stress
     field against depth into the disc, read with
     `w173_transverse_reach.depth_star`, unchanged.
  3. **The instrument's floor.**  `w173_epsilon_halo.amplitude_ladder`, unchanged.
  4. **The physics control.**  The classical elastic patch on the SAME disc, the
     same sensors, the same gap-filling (`cs_s2_elastic_patch`), through the
     same functions, at two mesh resolutions.
  5. **Does the adapter wear NeuberNet correctly?**  On the elastic branch the
     two solve the same boundary-value problem; the same comparison with the
     query points mirrored is the control that says the check has teeth.
  6. **In distribution or not.**  NeuberNet was trained on far-field loads of
     tension and torsion; its response along those two directions is compared
     with the classical operator's, beside a smooth mode and a single sensor.
  7. **The positive control, re-run in this session.**  Split-step `WindowNS`.
  8. **W95's question on this donor.**

**The port, and the first run's lesson.**  The trace is the displacement at the
material sensors read as a hat interpolant; the flux is the traction weighted by
the same hats (Galerkin).  The first run sampled the traction POINTWISE at the
sensors, and the classical control's operator then moved 36-42% between two
meshes: a Dirichlet-to-Neumann response to a hat trace is logarithmically
singular at the hat's knots, so a pointwise flux there does not converge.  The
pointwise operators are still measured, and their mesh floor is reported beside
the Galerkin one.  The first run then died at the transverse stage looking up
the sensor at 180 deg, which the pipeline writes as -180, having written
nothing -- so every stage now writes the artifact as it finishes.

**The third run, and the defect the second could not see.**  Stage 5 ran at a
tension-only base, where both subjects' hoop shears are identically zero, so it
passed an adapter whose hoop input was the wrong way round, and the second run's
stage 6 then read the torsion far field at cosine -0.969 and -0.997.
`cs_s2_torsion_control.py` separated the two readings that fit that -- SignNet
deciding at its own boundary, or a sign convention -- at nonzero-torsion bases on
the elastic branch, and it was the convention
(`cs_s2_neubernet.NeuberNetPatch.HOOP_SIGN`).  So stage 5 now also runs at two
nonzero-torsion bases on all six stress and all three flux components, and the
artifact carries the check that would have caught it; stage 6 adds those bases
and two hoop directions, differences the torsion far field both ways and records
SignNet's call; and the classical transverse profile is repeated on the coarse
mesh, because its face value sits on a hat knot, where the response is
log-singular and so mesh-limited.

Outputs `out/cs_s2/cs_s2.json`; dense operators to `out/cs_s2/operators.npz`.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

torch.set_num_threads(1)

# Tier 40's instruments, imported and NOT modified.  `w173_transverse_reach`
# re-wraps stdout when it is imported, so this module must not wrap it again: a
# second wrapper orphans the first, and its collection closes the shared buffer.
from scripts import w173_epsilon_halo as H  # noqa: E402
from scripts import w173_transverse_reach as T  # noqa: E402
from scripts import cs_s2_elastic_patch as EP  # noqa: E402
from scripts import cs_s2_neubernet as NB  # noqa: E402
from atlas.cases import neural_interface as NI  # noqa: E402

OUT = os.path.join(ROOT, "out", "cs_s2")
PATH = os.path.join(OUT, "cs_s2.json")

ALPHA, R, NU = 30.0, 50.0, 0.3
SY_E, ET_E = 3e-3, 1e-2
PORTS = ("ring:ux", "ring:uy", "ring:ut")
AMP = 1e-2
REL = [1e-1, 1e-2, 1e-3, 1e-4, 1e-6]
TARGETS = {"relative": REL,
           # dimensionless here; NOT defect scales, which this patch has none of
           "absolute": [1e-2, 1e-4]}
STRESS6 = ("s_xx", "s_yy", "s_zz", "s_xy", "s_yz", "s_xz")
FLUX3 = ("t_x", "t_y", "t_theta")


def flush(res):
    with open(PATH, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


def rel(a, b):
    nb = float(np.linalg.norm(b))
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / nb) if nb > 0 else float("nan")


def rel_diff(A, B):
    return float(np.linalg.norm(A - B, 2) / np.linalg.norm(B, 2))


def compare(a, b):
    """NeuberNet ``a`` against classical ``b``: error, cosine, norm ratio, pointwise sign.

    Where the classical side is identically zero -- a hoop shear under a
    tension-only load, an in-plane stress under pure torsion -- only the two norms
    are reported, so the donor's spurious output is still visible.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    out = {"norm_neubernet": na, "norm_classical": nb,
           "rel_error": None, "cosine": None, "norm_ratio": None,
           "sign_agreement_where_classical_above_10pct": None}
    if na > 0.0 and nb > 0.0:
        big = np.abs(b) > 0.1 * float(np.max(np.abs(b)))
        out.update({
            "rel_error": float(np.linalg.norm(a - b) / nb),
            "cosine": float(a @ b / (na * nb)),
            "norm_ratio": na / nb,
            "sign_agreement_where_classical_above_10pct": float(
                np.mean(np.sign(a[big]) == np.sign(b[big])))})
    return out


def by_component(A, B, names):
    return {n: compare(A[:, k], B[:, k]) for k, n in enumerate(names)}


def _f(x, fmt="%.3f"):
    return "None" if x is None else fmt % x


def node_band_truncation(S, M, radii):
    """||S - S_r||_2 / ||S||_2 with the band taken in NODE distance on the arc."""
    node = np.arange(S.shape[0]) % M
    d = np.abs(node[:, None] - node[None, :])
    n2 = float(np.linalg.norm(S, 2))
    return {"norm2": n2,
            "rows": [{"r": int(r),
                      "E2_rel": float(np.linalg.norm(S * (d > r), 2) / n2)}
                     for r in radii]}


def spectrum(S, k=12):
    sv = np.linalg.svd(S, compute_uv=False)
    e = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    return {"singular_values": [float(x) for x in sv[:k]],
            "sigma_min": float(sv[-1]),
            "effective_rank_99": int(np.searchsorted(e, 0.99) + 1),
            "effective_rank_999": int(np.searchsorted(e, 0.999) + 1),
            "size": int(sv.size)}


def reciprocity(S, mass):
    """Betti: the Galerkin operator times the hat masses is symmetric."""
    D = np.tile(mass, S.shape[0] // mass.size)
    W = D[:, None] * S
    return float(np.linalg.norm(W - W.T, 2) / np.linalg.norm(W, 2))


def ring_points(depths, dtheta, alpha):
    per = []
    for d in depths:
        r = EP.RL - d
        if d == 0.0:
            r = EP.RL - 1e-2          # inside the FE mesh's polygonal boundary
        if r <= 1e-9:
            per.append(np.array([[0.0, 0.0]]))
            continue
        th = np.radians(np.arange(-180.0, 180.0, dtheta))
        pts = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
        per.append(pts[~EP.in_void(pts[:, 0], pts[:, 1], alpha)])
    return per


def transverse(stress_fn, s_base, sensor_index, comp, amp, per_depth, depths):
    """The w173 transverse profile's format, on a disc; NaN points are counted."""
    pts = np.vstack(per_depth)
    counts = [p.shape[0] for p in per_depth]
    s1 = s_base.copy()
    s1[sensor_index, comp] += amp
    dsig = (stress_fn(s1, pts) - stress_fn(s_base, pts)) / amp
    ok = np.isfinite(dsig).all(axis=1)
    g = np.full(pts.shape[0], np.nan)
    g[ok] = np.sqrt((dsig[ok] ** 2).sum(axis=1))
    mx, rms, nz, nan = [], [], [], []
    for p in np.split(g, np.cumsum(counts)[:-1]):
        f = p[np.isfinite(p)]
        mx.append(float(f.max()) if f.size else float("nan"))
        rms.append(float(np.sqrt(np.mean(f ** 2))) if f.size else float("nan"))
        nz.append(int(np.count_nonzero(f)))
        nan.append(int(p.size - f.size))
    prof = {"max_by_depth": mx, "rms_by_depth": rms, "nonzero_by_depth": nz,
            "nan_by_depth": nan,
            "nonzero_reach": int(max((k for k, c in enumerate(nz) if c), default=0)),
            "peak": mx[0], "depths_Rn": [float(x) for x in depths],
            "points_per_depth": counts}
    prof["depth_star_max"] = T.depth_star(prof, "max_by_depth", REL)
    prof["depth_star_rms"] = T.depth_star(prof, "rms_by_depth", REL)
    return prof


def main():
    os.makedirs(OUT, exist_ok=True)
    t_all = time.perf_counter()
    res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "complete": False,
           "study": "CS-S2", "donor": "NeuberNet",
           "source": NB.SOURCE, "files": NB.verify(),
           "torch": {"version": torch.__version__, "threads": torch.get_num_threads(),
                     "dtype": "float32", "flush_denormal": NB.FLUSH_DENORMAL},
           "geometry": {"alpha_deg": ALPHA, "beta_deg": 0.0, "R_over_Rn": R,
                        "ring_radius_Rn": EP.RL, "nu": NU, "sy_over_E": SY_E,
                        "Et_over_E": ET_E},
           "port": {"trace": "displacement at the material sensors, u E/(sigma_y R_n), "
                             "hat-interpolated along the arc and continued across the "
                             "notch opening along the line the pipeline fills it with",
                    "flux": "traction sigma.n/sigma_y weighted by the same hats: "
                            "int t phi_j rho ds / int phi_j rho ds",
                    "components": ["u_x -> t_x", "u_y -> t_y", "u_theta -> t_theta"],
                    "hoop_sign": NB.NeuberNetPatch.HOOP_SIGN,
                    "hoop": ("ROTY = hoop_sign * u_theta / rho, u_theta on the classical "
                             "patch's hoop axis"),
                    "defects": [
                        {"run": 1, "what": "flux sampled pointwise at the hat knots",
                         "found_by": "the classical operator moved 36-42% between two meshes",
                         "fix": "the Galerkin dual; the pointwise operators are still measured"},
                        {"run": 1, "what": "trace clamped at the notch flanks",
                         "found_by": "the two subjects disagreed at the flank-end sensors only",
                         "fix": "the pipeline's own gap filling, continued across the opening"},
                        {"run": 2, "what": "hoop input written as ROTY = +u_theta / rho",
                         "found_by": ("cs_s2_torsion_control.py: at nonzero-torsion elastic "
                                      "bases s_yz and s_xz came back negated, cosine -1.000 "
                                      "and -0.999, in-plane stresses agreeing"),
                         "fix": "HOOP_SIGN = -1; stage 5 now checks torsion as well"}]},
           "instrument": {"amplitude": AMP, "targets": TARGETS,
                          "unchanged": ["w173_epsilon_halo.measure",
                                        "w173_epsilon_halo.dense_seam_operator",
                                        "w173_epsilon_halo.amplitude_ladder",
                                        "w173_epsilon_halo.far_field_spectrum",
                                        "w173_epsilon_halo.window_subjects",
                                        "w173_transverse_reach.depth_star",
                                        "w173_transverse_reach.run_subject"]},
           "stage_seconds": {}}
    store = {}
    flush(res)

    def stage(name, t0):
        res["stage_seconds"][name] = time.perf_counter() - t0
        flush(res)
        print("[stage] %s done" % name)

    t0 = time.perf_counter()
    model = NB.load_model()
    probe = NB.NeuberNetPatch(model, ALPHA, R, SY_E, ET_E, NU)
    M = probe.M
    radii = list(range(0, M))
    res["geometry"].update({
        "material_sensors": int(M),
        "material_theta_deg": [float(t) for t in probe.theta_s],
        "void_theta_deg": [float(EP.SENSOR_ANGLES[i]) for i in probe.void],
        "void_half_angle_deg": EP.void_half_angle(ALPHA),
        "sensor_spacing_Rn": float(2 * np.pi * EP.RL / 36.0)})
    stage("setup", t0)

    # --- the classical patches ---------------------------------------------
    t0 = time.perf_counter()
    big = EP.ElasticPatch(ALPHA, R, NU, r_out=25.0, h_tip=0.04, grad=0.03, h_max=0.8)
    ep = EP.ElasticPatch(ALPHA, R, NU, r_out=EP.RL, h_tip=0.04, grad=0.03, h_max=0.2)
    epf = EP.ElasticPatch(ALPHA, R, NU, r_out=EP.RL, h_tip=0.02, grad=0.015, h_max=0.1)
    res["classical"] = {
        "meshes": {"far_field_disc": {"r_out": 25.0, "nodes": int(big.nodes.shape[0])},
                   "patch_coarse": {"nodes": int(ep.nodes.shape[0]), **ep.mesh},
                   "patch_fine": {"nodes": int(epf.nodes.shape[0]), **epf.mesh}},
        "self_test": EP.self_test(R, NU),
        "hat_mass_fe_vs_quadrature": float(np.max(np.abs(epf.hat_mass - probe.hat_mass)
                                                  / probe.hat_mass))}
    stage("classical_build", t0)

    # --- base states from a classical far field ------------------------------
    t0 = time.perf_counter()
    th36 = np.radians(EP.SENSOR_ANGLES)
    pts36 = np.stack([EP.RL * np.cos(th36), EP.RL * np.sin(th36)], axis=1)

    def far_field(axial, shear):
        u_ex, ut_ex = EP.homogeneous_field(big.nodes, R, NU, axial=axial, shear=shear)
        u, ut, _t, _f = big.solve_nodal(u_ex[big.B, 0], u_ex[big.B, 1], ut_ex[big.B])
        vals = big.interpolate(np.column_stack([u[:, 0], u[:, 1], ut]), pts36)
        return np.nan_to_num(vals)

    unit_t = far_field(1.0, 0.0)
    unit_s = far_field(0.0, 1.0)
    bases = {}
    res["bases"] = {}
    for label, direction, level in [
            ("tension-elastic", unit_t, 0.7),
            ("tension-plastic", unit_t, 1.6),
            ("tension-torsion-plastic", unit_t + 0.8 * unit_s, 1.6)]:
        y0 = probe.regime(direction)["elastic_von_mises_over_sy"]
        s36 = (level / y0) * direction
        bases[label] = s36
        res["bases"][label] = {"target_elastic_von_mises_over_sy": level,
                               "scale": float(level / y0),
                               "regime": probe.regime(s36)}
        print("base %-24s %s" % (label, res["bases"][label]["regime"]["branch"]))
    stage("bases", t0)

    # --- 5. does the adapter wear NeuberNet correctly -----------------------
    t0 = time.perf_counter()
    grid = []
    for rr in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5):
        th = np.radians(np.arange(-180.0, 180.0, 10.0))
        p = np.stack([rr * np.cos(th), rr * np.sin(th)], axis=1)
        grid.append(p[~EP.in_void(p[:, 0], p[:, 1], ALPHA)])
    grid = np.vstack(grid)
    s_el = bases["tension-elastic"]
    nn_el = NB.NeuberNetPatch(model, ALPHA, R, SY_E, ET_E, NU, base36=s_el)
    sig_nn = nn_el.stress(s_el, grid)
    sig_fe = epf.stress_at_sensors_bc(s_el[nn_el.mat], grid)
    fin = np.isfinite(sig_fe).all(axis=1)
    mirror = grid * np.array([-1.0, 1.0])
    okm = fin & ~EP.in_void(mirror[:, 0], mirror[:, 1], ALPHA)
    sig_mirror = nn_el.stress(s_el, mirror[okm])
    f_nn = nn_el.ring_flux(s_el)
    f_fe = epf.sensor_flux(s_el[nn_el.mat])
    t_nn = nn_el.traction(nn_el.stress(s_el, nn_el.points), nn_el.normals)
    t_fe = epf.sensor_traction(s_el[nn_el.mat])
    kpk = int(np.argmax(np.where(fin, sig_nn[:, 1], -np.inf)))
    res["convention_check"] = {
        "what": "elastic branch: NeuberNet and the classical patch solve the same "
                "boundary-value problem from the same sensor data",
        "points": int(fin.sum()),
        "dropped_points_outside_fe_mesh": grid[~fin].tolist(),
        "stress_rel_error": rel(sig_nn[fin], sig_fe[fin]),
        "stress_rel_error_by_component": [rel(sig_nn[fin, k], sig_fe[fin, k]) for k in range(4)],
        "control_mirrored_points_rel_error": rel(sig_mirror, sig_fe[okm]),
        "ring_flux_rel_error": rel(f_nn[:, :2], f_fe[:, :2]),
        "ring_flux_rel_error_by_component": [rel(f_nn[:, k], f_fe[:, k]) for k in range(2)],
        "ring_pointwise_traction_rel_error": rel(t_nn[:, :2], t_fe[:, :2]),
        "peak_s_yy": {"neubernet": float(sig_nn[kpk, 1]), "classical": float(sig_fe[kpk, 1]),
                      "location": grid[kpk].tolist()},
        #: every component, named -- the four-entry lists above stop at s_xy
        "stress_by_component": by_component(sig_nn[fin], sig_fe[fin], STRESS6),
        "control_mirrored_points_by_component": by_component(sig_mirror, sig_fe[okm], STRESS6),
        "ring_flux_by_component": by_component(f_nn, f_fe, FLUX3),
        "ring_pointwise_traction_by_component": by_component(t_nn, t_fe, FLUX3)}
    cc = res["convention_check"]
    print("convention check: stress %.3g (mirrored %.3g), ring flux %.3g, pointwise %.3g, dropped %d"
          % (cc["stress_rel_error"], cc["control_mirrored_points_rel_error"],
             cc["ring_flux_rel_error"], cc["ring_pointwise_traction_rel_error"], int((~fin).sum())))

    # --- 5b. the hoop component, which a tension-only base cannot see -------
    res["convention_check_hoop"] = {
        "what": ("elastic branch at bases with NONZERO torsion: NeuberNet against the "
                 "classical patch on all six stress and all three flux components -- the "
                 "check the second run did not have"),
        "hoop_sign": NB.NeuberNetPatch.HOOP_SIGN, "rows": []}
    elastic_torsion = {}
    for label, direction in (("torsion-elastic", unit_s),
                             ("tension-torsion-elastic", unit_t + 0.8 * unit_s)):
        y0 = probe.regime(direction)["elastic_von_mises_over_sy"]
        s_h = (0.7 / y0) * direction
        elastic_torsion[label] = s_h
        nn_h = NB.NeuberNetPatch(model, ALPHA, R, SY_E, ET_E, NU, base36=s_h)
        sfe = epf.stress_at_sensors_bc(s_h[nn_h.mat], grid)
        ok = np.isfinite(sfe).all(axis=1)
        okh = ok & ~EP.in_void(mirror[:, 0], mirror[:, 1], ALPHA)
        snn = nn_h.stress(s_h, grid)
        smir = nn_h.stress(s_h, mirror[okh])
        row = {"base": label, "scale": float(0.7 / y0), "regime": nn_h.regime(s_h),
               "points": int(ok.sum()),
               "stress_by_component": by_component(snn[ok], sfe[ok], STRESS6),
               "stress_in_plane": compare(snn[ok, :4], sfe[ok, :4]),
               "stress_hoop_shears": compare(snn[ok, 4:], sfe[ok, 4:]),
               "neubernet_in_plane_over_hoop_shears": float(
                   np.linalg.norm(snn[ok, :4]) / np.linalg.norm(snn[ok, 4:])),
               "control_mirrored_points_by_component": by_component(smir, sfe[okh], STRESS6),
               "control_mirrored_points_hoop_shears": compare(smir[:, 4:], sfe[okh, 4:]),
               "ring_flux_by_component": by_component(nn_h.ring_flux(s_h),
                                                      epf.sensor_flux(s_h[nn_h.mat]), FLUX3)}
        res["convention_check_hoop"]["rows"].append(row)
        sc, fc = row["stress_by_component"], row["ring_flux_by_component"]
        print("hoop check %-24s s_yz cos %s rel %s  s_xz cos %s  t_theta cos %s rel %s  mirrored %s"
              % (label, _f(sc["s_yz"]["cosine"]), _f(sc["s_yz"]["rel_error"]),
                 _f(sc["s_xz"]["cosine"]), _f(fc["t_theta"]["cosine"]),
                 _f(fc["t_theta"]["rel_error"]),
                 _f(row["control_mirrored_points_hoop_shears"]["rel_error"])))
    stage("convention_check", t0)

    # --- 4. the classical operator, through the instrument -------------------
    t0 = time.perf_counter()
    res["classical"]["subjects"] = []
    S_fe = {}
    for patch, tag in ((ep, "coarse"), (epf, "fine")):
        for port in PORTS:
            c = port.split(":")[1]
            key = f"elastic-{tag}_{c}"
            r = H.measure(f"elastic-patch {tag} {port}", patch, port, 1.0, radii,
                          TARGETS, "classical control, same disc, Galerkin flux", store, key)
            r["subject"] = f"elastic-patch-{tag}"
            r["global_by"] = "physics (elliptic patch)"
            for rad in (2, 4):
                r["far_field_spectrum"].append(H.far_field_spectrum(store[key], rad))
            r["reciprocity"] = reciprocity(store[key], patch.hat_mass)
            res["classical"]["subjects"].append(r)
            S_fe[(tag, port)] = store[key]
            S_pw = H.dense_seam_operator(patch.respond_pointwise, port, np.zeros(M), 1.0)[0]
            store[f"elastic-{tag}-pointwise_{c}"] = S_pw
            S_fe[(tag, port, "pointwise")] = S_pw
    Sall_fe = H.dense_seam_operator(epf.respond, "ring:all", np.zeros(3 * M), 1.0)[0]
    store["elastic-fine_all"] = Sall_fe
    res["classical"]["mesh_floor_galerkin"] = {
        port: rel_diff(S_fe[("coarse", port)], S_fe[("fine", port)]) for port in PORTS}
    res["classical"]["mesh_floor_pointwise"] = {
        port: rel_diff(S_fe[("coarse", port, "pointwise")], S_fe[("fine", port, "pointwise")])
        for port in PORTS}
    res["classical"]["full_operator"] = {
        "node_truncation": node_band_truncation(Sall_fe, M, radii),
        "spectrum": spectrum(Sall_fe),
        "reciprocity": reciprocity(Sall_fe, epf.hat_mass)}
    print("classical: galerkin floor", {k: round(v, 5) for k, v in res["classical"]["mesh_floor_galerkin"].items()},
          "pointwise floor", {k: round(v, 4) for k, v in res["classical"]["mesh_floor_pointwise"].items()})
    for s in res["classical"]["subjects"]:
        if s["subject"] == "elastic-patch-fine":
            rows = s["linear"]["rows"]
            print("  %-34s E2rel r=1,2,4,8,16: %s" % (s["label"], " ".join(
                "%.3g" % next(x["E2_rel"] for x in rows if x["r"] == q) for q in (1, 2, 4, 8, 16))))
    stage("classical_operator", t0)

    # --- 1. NeuberNet, along the seam ---------------------------------------
    t0 = time.perf_counter()
    res["neubernet"] = {"subjects": [], "full_operator": {}, "against_classical": {}}
    for label, s36 in bases.items():
        nn = NB.NeuberNetPatch(model, ALPHA, R, SY_E, ET_E, NU, base36=s36,
                               agent_id=f"NEUBERNET[{label}]")
        for port in PORTS:
            c = port.split(":")[1]
            key = f"neubernet-{label}_{c}"
            r = H.measure(f"neubernet {label} {port}", nn, port, AMP, radii,
                          TARGETS, "the donor, on its own ring, Galerkin flux", store, key)
            r["subject"] = "neubernet"
            r["base"] = label
            r["global_by"] = "under test"
            for rad in (2, 4):
                r["far_field_spectrum"].append(H.far_field_spectrum(store[key], rad))
            r["reciprocity"] = reciprocity(store[key], nn.hat_mass)
            res["neubernet"]["subjects"].append(r)
            Sn, Sf = store[key], S_fe[("fine", port)]
            res["neubernet"]["against_classical"].setdefault(label, {})[port] = {
                "rel_diff_vs_fine_patch": rel_diff(Sn, Sf),
                "norm2_neubernet": float(np.linalg.norm(Sn, 2)),
                "norm2_classical": float(np.linalg.norm(Sf, 2)),
                "diag_ratio_median": float(np.median(np.diag(Sn) / np.diag(Sf)))}
            rows = r["linear"]["rows"]
            print("  %-44s E2rel r=1,2,4,8,16: %s  |S| %.3g vs %.3g  diff %.3g"
                  % (r["label"], " ".join("%.3g" % next(x["E2_rel"] for x in rows if x["r"] == q)
                                          for q in (1, 2, 4, 8, 16)),
                     np.linalg.norm(Sn, 2), np.linalg.norm(Sf, 2), rel_diff(Sn, Sf)))
        Sall = H.dense_seam_operator(nn.respond, "ring:all", np.zeros(3 * M), AMP)[0]
        store[f"neubernet-{label}_all"] = Sall
        res["neubernet"]["full_operator"][label] = {
            "node_truncation": node_band_truncation(Sall, M, radii),
            "spectrum": spectrum(Sall),
            "reciprocity": reciprocity(Sall, nn.hat_mass),
            "rel_diff_vs_fine_patch": rel_diff(Sall, Sall_fe),
            "norm_ratio_vs_fine_patch": float(np.linalg.norm(Sall, 2) / np.linalg.norm(Sall_fe, 2))}
        flush(res)
    nn = NB.NeuberNetPatch(model, ALPHA, R, SY_E, ET_E, NU, base36=bases["tension-plastic"])
    S_nn_pw = H.dense_seam_operator(nn.respond_pointwise, "ring:uy", np.zeros(M), AMP)[0]
    res["neubernet"]["pointwise_vs_galerkin_uy_tension_plastic"] = rel_diff(
        S_nn_pw, store["neubernet-tension-plastic_uy"])
    stage("neubernet_operator", t0)

    # --- 6. in distribution or not -------------------------------------------
    t0 = time.perf_counter()
    ts = EP.void_half_angle(ALPHA)
    span = 360.0 - 2.0 * ts
    smooth = np.zeros((M, 3))
    smooth[:, 1] = np.cos(3.0 * np.pi * (probe.phi_s - ts) / span)
    one = np.zeros((M, 3))
    one[M // 2, 1] = 1.0
    smooth_t = np.zeros((M, 3))
    smooth_t[:, 2] = smooth[:, 1]
    one_t = np.zeros((M, 3))
    one_t[M // 2, 2] = 1.0
    dirs = [("tension far field", unit_t[probe.mat]),
            ("torsion far field", unit_s[probe.mat]),
            ("smooth mode k=3 on u_y", smooth),
            ("one sensor on u_y", one),
            ("smooth mode k=3 on u_theta", smooth_t),
            ("one sensor on u_theta", one_t)]
    res["directional"] = {"what": "d/da flux(base + a v) for NeuberNet against S_classical v",
                          "rows": []}
    dir_bases = [("tension-elastic", bases["tension-elastic"]),
                 ("tension-plastic", bases["tension-plastic"]),
                 ("torsion-elastic", elastic_torsion["torsion-elastic"]),
                 ("tension-torsion-elastic", elastic_torsion["tension-torsion-elastic"])]
    for label, s_base in dir_bases:
        nn = NB.NeuberNetPatch(model, ALPHA, R, SY_E, ET_E, NU, base36=s_base)
        f0 = nn.respond("ring:all", np.zeros(3 * M))
        for name, d3 in dirs:
            v = d3.T.ravel()
            v = v / np.max(np.abs(v))
            d_nn = (nn.respond("ring:all", AMP * v) - f0) / AMP
            d_fe = Sall_fe @ v
            row = {"base": label, "direction": name, "rel_error": rel(d_nn, d_fe),
                   "norm_ratio": float(np.linalg.norm(d_nn) / np.linalg.norm(d_fe)),
                   "cosine": float(d_nn @ d_fe / (np.linalg.norm(d_nn) * np.linalg.norm(d_fe))),
                   "by_block": {b: compare(d_nn[k * M:(k + 1) * M], d_fe[k * M:(k + 1) * M])
                                for k, b in enumerate(FLUX3)}}
            if name == "torsion far field":
                d_m = (nn.respond("ring:all", -AMP * v) - f0) / (-AMP)
                s_p, s_m = s_base.copy(), s_base.copy()
                s_p[probe.mat] += AMP * v.reshape(3, M).T
                s_m[probe.mat] -= AMP * v.reshape(3, M).T
                #: the base's own flux, split: at a tension-only base the classical hoop
                #: block is exactly zero, so a nonzero one here is the donor's offset
                #: at T = 0 -- the thing production mode's sign flip multiplies
                row.update({"f0_hoop_block_norm": float(np.linalg.norm(f0[2 * M:])),
                            "f0_in_plane_block_norm": float(np.linalg.norm(f0[:2 * M])),
                            "minus": compare(d_m, d_fe),
                            "central": compare(0.5 * (d_nn + d_m), d_fe),
                            "plus_vs_minus_cosine": compare(d_nn, d_m)["cosine"],
                            "sign_torsion": {"base": nn.regime(s_base)["sign_torsion"],
                                             "base+av": nn.regime(s_p)["sign_torsion"],
                                             "base-av": nn.regime(s_m)["sign_torsion"]}})
            res["directional"]["rows"].append(row)
            print("  directional %-24s %-27s rel err %.3g  norm ratio %.3g  cosine %.3f"
                  % (label, name, row["rel_error"], row["norm_ratio"], row["cosine"]))
        flush(res)
    stage("directional", t0)

    # --- 3. the floor -------------------------------------------------------
    t0 = time.perf_counter()
    amps = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    cols = list(range(0, M, 4))
    res["amplitude_ladder"] = {}
    for label in ("tension-elastic", "tension-plastic"):
        nn = NB.NeuberNetPatch(model, ALPHA, R, SY_E, ET_E, NU, base36=bases[label])
        res["amplitude_ladder"][f"neubernet {label} ring:uy"] = H.amplitude_ladder(
            nn.respond, "ring:uy", M, amps, cols, 1e-2)
    res["amplitude_ladder"]["elastic-patch fine ring:uy"] = H.amplitude_ladder(
        epf.respond, "ring:uy", M, amps, cols, 1e-2)
    for lab, lad in res["amplitude_ladder"].items():
        print("ladder %-34s " % lab + "  ".join(
            "a=%.0e:%.3g" % (x["amplitude"], x["rel_disagreement_vs_ref"]) for x in lad["rows"]))
    stage("amplitude_ladder", t0)

    # --- 2. across the seam -------------------------------------------------
    t0 = time.perf_counter()
    depths = np.round(np.arange(0.0, EP.RL + 1e-9, 0.1), 10)
    per_depth = ring_points(depths, 5.0, ALPHA)
    k180 = int(np.flatnonzero(np.isclose(np.mod(probe.theta_s, 360.0), 180.0))[0])
    k90 = int(np.flatnonzero(np.isclose(probe.theta_s, 90.0))[0])
    pokes = [("180deg u_x", k180, 0), ("180deg u_y", k180, 1), ("90deg u_y", k90, 1)]
    res["transverse"] = {"depth_unit_Rn": 0.1, "theta_step_deg": 5.0, "rows": []}

    def fe_fn(s, p):
        return epf.stress_at_sensors_bc(s[probe.mat], p)

    def fe_fn_coarse(s, p):
        return ep.stress_at_sensors_bc(s[probe.mat], p)

    for label in ("tension-elastic", "tension-plastic"):
        nn = NB.NeuberNetPatch(model, ALPHA, R, SY_E, ET_E, NU, base36=bases[label])
        for name, k, comp in pokes:
            prof = transverse(nn.stress, bases[label], int(probe.mat[k]), comp, AMP,
                              per_depth, depths)
            prof.update({"subject": "neubernet", "base": label, "poke": name})
            res["transverse"]["rows"].append(prof)
            print("  transverse neubernet %-16s %-11s d*(1e-1)=%s d*(1e-2)=%s d*(1e-4)=%s"
                  % (label, name, prof["depth_star_max"]["rows"][0]["depth_star"],
                     prof["depth_star_max"]["rows"][1]["depth_star"],
                     prof["depth_star_max"]["rows"][3]["depth_star"]))
        flush(res)
    for fn, subject in ((fe_fn, "elastic-patch-fine"), (fe_fn_coarse, "elastic-patch-coarse")):
        for name, k, comp in pokes:
            prof = transverse(fn, np.zeros((36, 3)), int(probe.mat[k]), comp, 1.0,
                              per_depth, depths)
            prof.update({"subject": subject, "base": "linear", "poke": name})
            if subject == "elastic-patch-coarse":
                prof["why"] = ("mesh control: the face value sits on a hat knot, where a "
                               "Dirichlet-to-Neumann response is log-singular, so d* is read "
                               "against a mesh-limited peak")
            res["transverse"]["rows"].append(prof)
            print("  transverse %-20s %-11s peak %.3g d*(1e-1)=%s d*(1e-2)=%s d*(1e-4)=%s"
                  % (subject, name, prof["peak"], prof["depth_star_max"]["rows"][0]["depth_star"],
                     prof["depth_star_max"]["rows"][1]["depth_star"],
                     prof["depth_star_max"]["rows"][3]["depth_star"]))
    stage("transverse", t0)

    # --- 7. the positive control, in this session ---------------------------
    t0 = time.perf_counter()
    u_full, v_full = NI.load_state()
    wag = H.window_subjects(u_full, v_full, NI.DEFAULT_TILING, True)
    pc = H.measure("windowns-split-step C00 xhi:MECH a=0.01", wag["C00"], "xhi:MECH",
                   1e-2, list(range(0, 65)),
                   {"relative": REL, "absolute": [3.7014e-4, 2.6995e-5, 1e-6, 1e-9]},
                   "positive control, re-run in this session", store, "windowns-split-step_C00")
    pc["subject"] = "windowns-split-step"
    pct = T.run_subject("windowns-split-step C00 xhi", wag["C00"], "xhi",
                        "windowns-split-step", "nothing -- the elliptic part is exposed",
                        NI.DOMAIN_OF_DEPENDENCE, amps=(1e-2,), js=(64,))
    first_zero = next((x["r"] for x in pc["linear"]["rows"]
                       if all(y["E2_rel"] == 0.0 for y in pc["linear"]["rows"]
                              if y["r"] >= x["r"])), None)
    res["positive_control"] = {"along": pc, "transverse": pct,
                               "exactly_zero_from_r": first_zero}
    print("positive control: E2 exactly zero from r = %s" % first_zero)
    stage("positive_control", t0)

    # --- 8. W95 on this donor -----------------------------------------------
    res["w95"] = {
        "question": "is there a same-class reference pair, so that tau and sigma can exist",
        "same_class_monolith": False,
        "why_not": ("NeuberNet is defined on one sub-domain -- the disc |p| <= 5 R_n "
                    "around one notch tip, at parameters inside its training ranges. "
                    "There is no larger or undecomposed domain it can be evaluated on, "
                    "so it cannot be its own referent."),
        "classical_referent_elastic_branch": (
            "exists in this repository: the classical elastic patch on the same disc, "
            "compared above in tension and in torsion -- a referent of a different class, "
            "so what it measures is the donor's model error, not tau or sigma"),
        "classical_referent_plastic_branch": (
            "exists only as the authors' ANSYS elastic-plastic analyses (inputs and "
            "fields in the CC BY 4.0 Zenodo dataset); no solver here computes it"),
        "graph": "no compiling Atlas graph holds an axisymmetric elastic-plastic agent"}

    res["elapsed_seconds"] = time.perf_counter() - t_all
    np.savez_compressed(os.path.join(OUT, "operators.npz"), **store)
    res["operators"] = {"path": "out/cs_s2/operators.npz", "keys": sorted(store)}
    res["complete"] = True
    flush(res)
    print("wrote", PATH, "in %.1f s" % res["elapsed_seconds"])


if __name__ == "__main__":
    main()
