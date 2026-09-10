"""CS-S2 -- the torsion control: is the adapter wearing NeuberNet's hoop component the right way round?

**Why this exists.**  `cs_s2_bounded_donor.py`'s convention check compared
NeuberNet with the classical patch at ONE base -- tension on the elastic branch --
where both subjects' torsion shears are identically zero.  So nothing had checked
the third input (ROTY, which the adapter writes as ``+u_theta / rho``) or the two
outputs it drives (``s_yz``, ``s_xz``, and through them the hoop flux).  The
directional stage then read the torsion far field at cosine -0.969 on the elastic
base and -0.997 on the plastic one.  Two readings fit that, and they predict
different things:

  (1) **SignNet deciding at its own boundary.**  Both bases carry zero torsion, so
      a small torsion perturbation sits where SignNet's torsion output crosses
      0.5.  If the canonicalised network is not odd in torsion there, the sign of
      a one-sided difference follows SignNet's call rather than the physics.
  (2) **A sign-convention mismatch** between the adapter's ROTY and the one
      NeuberNet was trained in.

Production mode multiplies the input's ROTY block by SignNet's torsion sign and
the output shears (columns 6:8) by the same sign (`definitions.py`,
``NeuberNet.forward``), so at a base whose torsion is firmly signed the donor is
odd in torsion by construction.  There:

  * (1) predicts the shears AGREE with linear elasticity, in-plane and hoop alike;
  * (2) predicts the in-plane stresses agree and the hoop shears are NEGATED.

This script measures both, under both hoop conventions stated explicitly -- the
adapter's ``ROTY = +u_theta / rho`` and the other one, ``-u_theta / rho`` -- so the
labels stay true whichever one `cs_s2_neubernet` carries when it is run:

  A. the convention check, all six stress components and all three ring-flux
     components, at two NONZERO-torsion bases on the elastic branch (pure
     torsion, and the driver's tension + 0.8 torsion direction scaled elastic),
     beside the driver's tension-only base and the mirrored-point control;
  B. directional rows d/da flux(base + a v) against the classical operator, with
     the torsion far field differenced both ways (+a and -a) so an even part
     shows up as a disagreement between them;
  C. SignNet's torsion sign at base, base + a v and base - a v on both
     zero-torsion bases, a = 1e-1 .. 1e-4: (1) needs a call that does not follow
     the sign of a.

Nothing in the adapter is changed here.  Writes `out/cs_s2/torsion_control.json`,
flushed after every stage.  ASCII output only; stdout is not re-wrapped.
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

# The instrument's operator builder, imported and not modified.  The driver is
# NOT imported: it imports `w173_transverse_reach`, which re-wraps stdout.
from scripts import w173_epsilon_halo as H  # noqa: E402
from scripts import cs_s2_elastic_patch as EP  # noqa: E402
from scripts import cs_s2_neubernet as NB  # noqa: E402

OUT = os.path.join(ROOT, "out", "cs_s2")
PATH = os.path.join(OUT, "torsion_control.json")
DRIVER = os.path.join(OUT, "cs_s2.json")

#: the driver's patch, restated and checked against its artifact below
ALPHA, R, NU = 30.0, 50.0, 0.3
SY_E, ET_E = 3e-3, 1e-2
AMP = 1e-2
STRESS6 = ("s_xx", "s_yy", "s_zz", "s_xy", "s_yz", "s_xz")
FLUX3 = ("t_x", "t_y", "t_theta")
HOOP = {"+1": 1.0, "-1": -1.0}


def flush(res):
    with open(PATH, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


class Hoop(NB.NeuberNetPatch):
    """The adapter with the sign of its hoop input stated rather than inherited.

    ``ROTY = hoop_sign * u_theta / rho``; everything else -- the gap filling, the
    forward, the Galerkin ring flux, the regime replica -- is the adapter's own.
    """

    def __init__(self, *args, hoop_sign=1.0, **kw):
        super().__init__(*args, **kw)
        self.hoop_sign = float(hoop_sign)

    def branch(self, s36):
        v = self.fill_opening(s36)
        return np.concatenate([v[:, 0], v[:, 1], self.hoop_sign * v[:, 2] / self.rho36])


def compare(a, b):
    """NeuberNet ``a`` against classical ``b``: error, cosine, norm ratio, pointwise sign."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    out = {"norm_neubernet": na, "norm_classical": nb}
    if na > 0.0 and nb > 0.0:
        big = np.abs(b) > 0.1 * float(np.max(np.abs(b)))
        out.update({
            "rel_error": float(np.linalg.norm(a - b) / nb),
            "cosine": float(a @ b / (na * nb)),
            "norm_ratio": na / nb,
            "sign_agreement_where_classical_above_10pct": float(
                np.mean(np.sign(a[big]) == np.sign(b[big]))),
        })
    else:
        out.update({"rel_error": None, "cosine": None, "norm_ratio": None,
                    "sign_agreement_where_classical_above_10pct": None})
    return out


def by_component(A, B, names):
    return {n: compare(A[:, k], B[:, k]) for k, n in enumerate(names)}


def main():
    os.makedirs(OUT, exist_ok=True)
    t_all = time.perf_counter()
    res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "complete": False,
           "study": "CS-S2 torsion control", "source": NB.SOURCE, "files": NB.verify(),
           "torch": {"version": torch.__version__, "threads": torch.get_num_threads(),
                     "device": "cpu", "dtype": "float32"},
           "hoop_conventions": {"+1": "ROTY = +u_theta / rho",
                                "-1": "ROTY = -u_theta / rho"},
           "amplitude": AMP, "stage_seconds": {}}
    flush(res)

    def stage(name, t0):
        res["stage_seconds"][name] = time.perf_counter() - t0
        flush(res)
        print("[stage] %s done" % name)

    # --- setup: the same patch as the driver, checked against its artifact ----
    t0 = time.perf_counter()
    model = NB.load_model()
    probe = NB.NeuberNetPatch(model, ALPHA, R, SY_E, ET_E, NU)
    M = probe.M
    with open(DRIVER, encoding="utf-8") as fh:
        drv = json.load(fh)
    g = drv["geometry"]
    same = {"alpha_deg": g["alpha_deg"] == ALPHA, "R_over_Rn": g["R_over_Rn"] == R,
            "nu": g["nu"] == NU, "sy_over_E": g["sy_over_E"] == SY_E,
            "Et_over_E": g["Et_over_E"] == ET_E, "material_sensors": g["material_sensors"] == M,
            "amplitude": drv["instrument"]["amplitude"] == AMP}
    if not all(same.values()):
        raise RuntimeError("the torsion control is not on the driver's patch: %s" % same)
    rng = np.random.default_rng(41)
    s_test = rng.normal(size=(36, 3))
    shipped = {k: bool(np.array_equal(probe.branch(s_test),
                                      Hoop(model, ALPHA, R, SY_E, ET_E, NU, hoop_sign=v).branch(s_test)))
               for k, v in HOOP.items()}
    res["setup"] = {"same_patch_as_driver": same, "material_sensors": int(M),
                    "adapter_in_cs_s2_neubernet_equals": shipped}
    print("adapter in cs_s2_neubernet equals hoop convention:", shipped)
    stage("setup", t0)

    # --- the classical patches and the classical operator ----------------------
    t0 = time.perf_counter()
    big = EP.ElasticPatch(ALPHA, R, NU, r_out=25.0, h_tip=0.04, grad=0.03, h_max=0.8)
    epf = EP.ElasticPatch(ALPHA, R, NU, r_out=EP.RL, h_tip=0.02, grad=0.015, h_max=0.1)
    Sall_fe = H.dense_seam_operator(epf.respond, "ring:all", np.zeros(3 * M), 1.0)[0]
    drv_nodes = drv["classical"]["meshes"]["patch_fine"]["nodes"]
    res["classical"] = {"far_field_disc_nodes": int(big.nodes.shape[0]),
                        "patch_fine_nodes": int(epf.nodes.shape[0]),
                        "patch_fine_nodes_in_driver": int(drv_nodes),
                        "operator_norm2": float(np.linalg.norm(Sall_fe, 2))}
    stage("classical", t0)

    # --- bases -------------------------------------------------------------------
    t0 = time.perf_counter()
    th36 = np.radians(EP.SENSOR_ANGLES)
    pts36 = np.stack([EP.RL * np.cos(th36), EP.RL * np.sin(th36)], axis=1)

    def far_field(axial, shear):
        u_ex, ut_ex = EP.homogeneous_field(big.nodes, R, NU, axial=axial, shear=shear)
        u, ut, _t, _f = big.solve_nodal(u_ex[big.B, 0], u_ex[big.B, 1], ut_ex[big.B])
        return np.nan_to_num(big.interpolate(np.column_stack([u[:, 0], u[:, 1], ut]), pts36))

    unit_t = far_field(1.0, 0.0)
    unit_s = far_field(0.0, 1.0)
    bases = {}
    res["bases"] = {}
    for label, direction, level in [
            ("tension-elastic", unit_t, 0.7),
            ("tension-plastic", unit_t, 1.6),
            ("torsion-elastic", unit_s, 0.7),
            ("tension-torsion-elastic", unit_t + 0.8 * unit_s, 0.7),
            ("tension-torsion-plastic", unit_t + 0.8 * unit_s, 1.6)]:
        y0 = probe.regime(direction)["elastic_von_mises_over_sy"]
        s36 = (level / y0) * direction
        bases[label] = s36
        row = {"target_elastic_von_mises_over_sy": level, "scale": float(level / y0),
               "regime": {k: Hoop(model, ALPHA, R, SY_E, ET_E, NU, hoop_sign=v).regime(s36)
                          for k, v in HOOP.items()}}
        if label in drv["bases"]:
            row["scale_in_driver"] = drv["bases"][label]["scale"]
        res["bases"][label] = row
        print("base %-24s scale %.6g  +1: %s sign_T %+.0f  -1: %s sign_T %+.0f"
              % (label, row["scale"], row["regime"]["+1"]["branch"],
                 row["regime"]["+1"]["sign_torsion"], row["regime"]["-1"]["branch"],
                 row["regime"]["-1"]["sign_torsion"]))
    stage("bases", t0)

    # --- A. the convention check, all components, both conventions -----------
    t0 = time.perf_counter()
    grid = []
    for rr in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5):
        th = np.radians(np.arange(-180.0, 180.0, 10.0))
        p = np.stack([rr * np.cos(th), rr * np.sin(th)], axis=1)
        grid.append(p[~EP.in_void(p[:, 0], p[:, 1], ALPHA)])
    grid = np.vstack(grid)
    mirror = grid * np.array([-1.0, 1.0])
    res["convention_check"] = {
        "what": ("elastic branch: NeuberNet under each hoop convention against the classical "
                 "patch, from the same sensor data, per component"),
        "grid_points": int(grid.shape[0]), "rows": []}
    for label in ("tension-elastic", "torsion-elastic", "tension-torsion-elastic"):
        s = bases[label]
        sig_fe = epf.stress_at_sensors_bc(s[probe.mat], grid)
        fin = np.isfinite(sig_fe).all(axis=1)
        okm = fin & ~EP.in_void(mirror[:, 0], mirror[:, 1], ALPHA)
        f_fe = epf.sensor_flux(s[probe.mat])
        for key, sgn in HOOP.items():
            nn = Hoop(model, ALPHA, R, SY_E, ET_E, NU, base36=s, hoop_sign=sgn)
            sig_nn = nn.stress(s, grid)
            sig_mir = nn.stress(s, mirror[okm])
            f_nn = nn.ring_flux(s)
            row = {"base": label, "hoop": key, "points": int(fin.sum()),
                   "regime": nn.regime(s),
                   "stress": by_component(sig_nn[fin], sig_fe[fin], STRESS6),
                   "stress_in_plane": compare(sig_nn[fin, :4], sig_fe[fin, :4]),
                   "stress_hoop_shears": compare(sig_nn[fin, 4:], sig_fe[fin, 4:]),
                   "control_mirrored_points": by_component(sig_mir, sig_fe[okm], STRESS6),
                   "control_mirrored_points_all": compare(sig_mir, sig_fe[okm]),
                   "ring_flux": by_component(f_nn, f_fe, FLUX3)}
            res["convention_check"]["rows"].append(row)
            st = row["stress"]
            print("  %-24s hoop %s  cos s_yy %s s_yz %s s_xz %s  t_theta %s  rel s_yz %s"
                  % (label, key, _f(st["s_yy"]["cosine"]), _f(st["s_yz"]["cosine"]),
                     _f(st["s_xz"]["cosine"]), _f(row["ring_flux"]["t_theta"]["cosine"]),
                     _f(st["s_yz"]["rel_error"])))
        flush(res)
    stage("convention_check", t0)

    # --- B. directional rows ------------------------------------------------------
    t0 = time.perf_counter()
    ts = EP.void_half_angle(ALPHA)
    span = 360.0 - 2.0 * ts
    smooth_y = np.zeros((M, 3))
    smooth_y[:, 1] = np.cos(3.0 * np.pi * (probe.phi_s - ts) / span)
    smooth_t = np.zeros((M, 3))
    smooth_t[:, 2] = smooth_y[:, 1]
    one_y = np.zeros((M, 3))
    one_y[M // 2, 1] = 1.0
    one_t = np.zeros((M, 3))
    one_t[M // 2, 2] = 1.0
    dirs = [("tension far field", unit_t[probe.mat]),
            ("torsion far field", unit_s[probe.mat]),
            ("smooth mode k=3 on u_y", smooth_y),
            ("smooth mode k=3 on u_theta", smooth_t),
            ("one sensor on u_y", one_y),
            ("one sensor on u_theta", one_t)]
    res["directional"] = {"what": "d/da flux(base + a v) for NeuberNet against S_classical v",
                          "rows": []}
    for label in ("tension-elastic", "tension-plastic", "torsion-elastic",
                  "tension-torsion-elastic"):
        for key, sgn in HOOP.items():
            nn = Hoop(model, ALPHA, R, SY_E, ET_E, NU, base36=bases[label], hoop_sign=sgn)
            f0 = nn.respond("ring:all", np.zeros(3 * M))
            for name, d3 in dirs:
                v = d3.T.ravel()
                v = v / np.max(np.abs(v))
                d_fe = Sall_fe @ v
                d_p = (nn.respond("ring:all", AMP * v) - f0) / AMP
                row = {"base": label, "hoop": key, "direction": name, "plus": compare(d_p, d_fe),
                       "plus_by_block": {b: compare(d_p[k * M:(k + 1) * M], d_fe[k * M:(k + 1) * M])
                                         for k, b in enumerate(FLUX3)}}
                if name == "torsion far field":
                    d_m = (nn.respond("ring:all", -AMP * v) - f0) / (-AMP)
                    cen = 0.5 * (d_p + d_m)
                    s_p = bases[label].copy()
                    s_m = bases[label].copy()
                    s_p[probe.mat] += AMP * v.reshape(3, M).T
                    s_m[probe.mat] -= AMP * v.reshape(3, M).T
                    row.update({
                        "minus": compare(d_m, d_fe),
                        "central": compare(cen, d_fe),
                        "plus_vs_minus_cosine": compare(d_p, d_m)["cosine"],
                        "sign_torsion_at": {"base": nn.regime(bases[label])["sign_torsion"],
                                            "base+av": nn.regime(s_p)["sign_torsion"],
                                            "base-av": nn.regime(s_m)["sign_torsion"]}})
                res["directional"]["rows"].append(row)
                extra = ""
                if name == "torsion far field":
                    extra = "  minus cos %s  central cos %s  signs %s" % (
                        _f(row["minus"]["cosine"]), _f(row["central"]["cosine"]),
                        row["sign_torsion_at"])
                print("  dir %-24s hoop %s %-27s cos %s ratio %s%s"
                      % (label, key, name, _f(row["plus"]["cosine"]),
                         _f(row["plus"]["norm_ratio"]), extra))
            flush(res)
    stage("directional", t0)

    # --- C. SignNet's torsion call on the zero-torsion bases ---------------------
    t0 = time.perf_counter()
    v_t = unit_s[probe.mat] / np.max(np.abs(unit_s[probe.mat]))
    res["signnet"] = {"what": ("regime() at base + a v and base - a v, v the torsion far field "
                               "scaled to max |v| = 1"), "rows": []}
    for label in ("tension-elastic", "tension-plastic"):
        for key, sgn in HOOP.items():
            nn = Hoop(model, ALPHA, R, SY_E, ET_E, NU, hoop_sign=sgn)
            for a in (1e-1, 1e-2, 1e-3, 1e-4):
                for side in (+1.0, -1.0):
                    s = bases[label].copy()
                    s[probe.mat] += side * a * v_t
                    reg = nn.regime(s)
                    res["signnet"]["rows"].append({"base": label, "hoop": key, "a": side * a,
                                                   **reg})
            print("  signnet %-16s hoop %s  %s" % (label, key, " ".join(
                "a=%+.0e:%+.0f" % (r["a"], r["sign_torsion"])
                for r in res["signnet"]["rows"] if r["base"] == label and r["hoop"] == key)))
    stage("signnet", t0)

    res["elapsed_seconds"] = time.perf_counter() - t_all
    res["complete"] = True
    flush(res)
    print("wrote", PATH, "in %.1f s" % res["elapsed_seconds"])


def _f(x):
    return "None" if x is None else "%+.3f" % x


if __name__ == "__main__":
    main()
