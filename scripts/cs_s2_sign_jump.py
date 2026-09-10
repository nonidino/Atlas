"""CS-S2 -- does NeuberNet's published forward JUMP where SignNet changes its mind?

`cs_s2_torsion_control.py` settled the hoop convention and left one reading
standing that the convention does not explain.  At a base with ZERO torsion on
the elastic branch, the torsion far field differenced one way and the other way
disagree in magnitude by a factor near 50 (norm ratio 0.18 against 8.7), and
SignNet's torsion call differs between the two steps; at the plastic base the
call does not change at that step and the two agree.  At the pure-torsion base
the TENSION far field reads the same way (norm ratio 6.1, cosine 0.18).  And
SignNet's call does not change AT zero: on the elastic base it holds +1 for
steps of either sign below 1e-3 and splits at 1e-2.

Production mode multiplies the torsion input by SignNet's call and the output
shears by the same call, and likewise tension with the in-plane stresses
(`definitions.py`, ``NeuberNet.forward``).  Where the call changes, the forward
switches between the canonical network and its sign-flipped image, and unless
the canonical network's shear output there is zero the two branches do not meet:
a discontinuity of the operator the authors publish, located at SignNet's own
decision point rather than at zero load.  That is an inference from the code
until it is measured, and this script measures it three ways, on the output
block the direction v drives:

  1. **The ladder.**  J(a) = || F(base + a v) - F(base - a v) ||, a = 1e-1 .. 1e-6,
     with SignNet's call at both ends.  Smooth through the interval, J / 2a is flat
     in a and near the classical ||S v||; straddling a jump, it is not.
  2. **The decision point.**  SignNet's call bisected along v between the largest
     step that keeps the base's call and the smallest that changes it -- regime()
     only, no forward pass.
  3. **The jump itself.**  F at the decision point plus and minus a step 1e-3 of
     its own size, against the classical response across the same interval.

The float32 floor is reported beside the ladder.  Series: tension-elastic and
tension-plastic (zero torsion, v = the torsion far field), torsion-elastic (zero
tension, v = the tension far field), and tension-torsion-elastic in both
directions -- the control, with neither load zero.  The adapter's hoop convention
is the corrected one and is recorded.  Writes `out/cs_s2/sign_jump.json`, flushed
after every series.  ASCII output; stdout is not re-wrapped.
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

from scripts import w173_epsilon_halo as H  # noqa: E402
from scripts import cs_s2_elastic_patch as EP  # noqa: E402
from scripts import cs_s2_neubernet as NB  # noqa: E402

OUT = os.path.join(ROOT, "out", "cs_s2")
PATH = os.path.join(OUT, "sign_jump.json")
ALPHA, R, NU = 30.0, 50.0, 0.3
SY_E, ET_E = 3e-3, 1e-2
AMPS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
EPS32 = float(np.finfo(np.float32).eps)
BISECT = 40
DELTA_REL = 1e-3


def flush(res):
    with open(PATH, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


def main():
    os.makedirs(OUT, exist_ok=True)
    t_all = time.perf_counter()
    res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "complete": False,
           "study": "CS-S2 sign jump", "source": NB.SOURCE, "files": NB.verify(),
           "torch": {"version": torch.__version__, "threads": torch.get_num_threads(),
                     "device": "cpu", "dtype": "float32", "flush_denormal": NB.FLUSH_DENORMAL},
           "hoop_sign": NB.NeuberNetPatch.HOOP_SIGN, "amplitudes": list(AMPS),
           "eps_float32": EPS32, "bisection_steps": BISECT, "delta_relative": DELTA_REL,
           "J": "|| F(base + a v) - F(base - a v) || on the output block v drives",
           "series": []}
    flush(res)

    model = NB.load_model()
    probe = NB.NeuberNetPatch(model, ALPHA, R, SY_E, ET_E, NU)
    M = probe.M
    big = EP.ElasticPatch(ALPHA, R, NU, r_out=25.0, h_tip=0.04, grad=0.03, h_max=0.8)
    epf = EP.ElasticPatch(ALPHA, R, NU, r_out=EP.RL, h_tip=0.02, grad=0.015, h_max=0.1)
    Sall_fe = H.dense_seam_operator(epf.respond, "ring:all", np.zeros(3 * M), 1.0)[0]
    th36 = np.radians(EP.SENSOR_ANGLES)
    pts36 = np.stack([EP.RL * np.cos(th36), EP.RL * np.sin(th36)], axis=1)

    def far_field(axial, shear):
        u_ex, ut_ex = EP.homogeneous_field(big.nodes, R, NU, axial=axial, shear=shear)
        u, ut, _t, _f = big.solve_nodal(u_ex[big.B, 0], u_ex[big.B, 1], ut_ex[big.B])
        return np.nan_to_num(big.interpolate(np.column_stack([u[:, 0], u[:, 1], ut]), pts36))

    unit_t = far_field(1.0, 0.0)
    unit_s = far_field(0.0, 1.0)

    def scaled(direction, level):
        return (level / probe.regime(direction)["elastic_von_mises_over_sy"]) * direction

    blocks = {"in_plane": slice(0, 2 * M), "hoop": slice(2 * M, 3 * M)}
    other = {"in_plane": "hoop", "hoop": "in_plane"}
    series = [
        ("tension-elastic", scaled(unit_t, 0.7), "torsion far field", unit_s, "hoop", "sign_torsion"),
        ("tension-plastic", scaled(unit_t, 1.6), "torsion far field", unit_s, "hoop", "sign_torsion"),
        ("torsion-elastic", scaled(unit_s, 0.7), "tension far field", unit_t, "in_plane", "sign_tension"),
        ("tension-torsion-elastic", scaled(unit_t + 0.8 * unit_s, 0.7), "torsion far field",
         unit_s, "hoop", "sign_torsion"),
        ("tension-torsion-elastic", scaled(unit_t + 0.8 * unit_s, 0.7), "tension far field",
         unit_t, "in_plane", "sign_tension")]
    for label, s_base, dname, field, blk, key in series:
        nn = NB.NeuberNetPatch(model, ALPHA, R, SY_E, ET_E, NU, base36=s_base)
        v = field[probe.mat].T.ravel()
        v = v / np.max(np.abs(v))
        v3 = v.reshape(3, M).T
        sl = blocks[blk]
        f0 = nn.respond("ring:all", np.zeros(3 * M))
        s_v = float(np.linalg.norm((Sall_fe @ v)[sl]))
        floor = EPS32 * float(np.max(np.abs(f0))) * np.sqrt(M) * 2.0

        def call(a):
            s = s_base.copy()
            s[probe.mat] += a * v3
            return nn.regime(s)[key]

        c0 = call(0.0)
        rows = []
        for a in AMPS:
            fp = nn.respond("ring:all", a * v)
            fm = nn.respond("ring:all", -a * v)
            J = float(np.linalg.norm((fp - fm)[sl]))
            row = {"a": a, "J": J, "J_over_2a": J / (2.0 * a),
                   "J_over_2a_over_classical": J / (2.0 * a) / s_v,
                   "even_part": float(np.linalg.norm((fp + fm - 2.0 * f0)[sl])),
                   "float32_floor_approx": floor,
                   key + "_plus": call(a), key + "_minus": call(-a)}
            rows.append(row)
            print("  %-24s %-18s a=%.0e  J/2a %.3e  (classical %.3e)  J %.3e  floor %.1e  %s %+.0f/%+.0f"
                  % (label, dname, a, row["J_over_2a"], s_v, J, floor, key,
                     row[key + "_plus"], row[key + "_minus"]))
        entry = {"base": label, "direction": dname, "block": blk,
                 "regime": nn.regime(s_base), "call_at_base": c0,
                 "f0_block_norm": float(np.linalg.norm(f0[sl])),
                 "f0_other_block_norm": float(np.linalg.norm(f0[blocks[other[blk]]])),
                 "classical_S_v_block_norm": s_v, "rows": rows, "decision_point": None}

        # --- the decision point, bisected, and the forward across it ---------
        side, lo, hi = None, None, None
        for sgn in (+1.0, -1.0):
            calls = {a: call(sgn * a) for a in AMPS}
            flips = sorted(a for a in AMPS if calls[a] != c0)
            if not flips:
                continue
            below = [a for a in AMPS if a < flips[0] and calls[a] == c0]
            if not below:
                continue
            side, hi, lo = sgn, flips[0], max(below)
            break
        if side is not None:
            for _ in range(BISECT):
                mid = 0.5 * (lo + hi)
                if call(side * mid) == c0:
                    lo = mid
                else:
                    hi = mid
            a_star = side * 0.5 * (lo + hi)
            d = DELTA_REL * abs(a_star)
            f_in = nn.respond("ring:all", (a_star - side * d) * v)
            f_out = nn.respond("ring:all", (a_star + side * d) * v)
            jump = float(np.linalg.norm((f_out - f_in)[sl]))
            entry["decision_point"] = {
                "side": side, "a_star": a_star, "bracket": [side * lo, side * hi],
                "delta": d,
                "call_inside": call(a_star - side * d), "call_outside": call(a_star + side * d),
                "jump_block_norm": jump,
                "classical_across_same_interval": 2.0 * d * s_v,
                "jump_over_classical_S_v": jump / s_v,
                "jump_over_f0_block": (jump / entry["f0_block_norm"]
                                       if entry["f0_block_norm"] > 0 else None)}
            dp = entry["decision_point"]
            print("  %-24s %-18s decision point a* = %.6e  jump %.3e  classical across it %.3e  jump/||Sv|| %.3g  calls %+.0f -> %+.0f"
                  % (label, dname, a_star, jump, dp["classical_across_same_interval"],
                     dp["jump_over_classical_S_v"], dp["call_inside"], dp["call_outside"]))
        else:
            print("  %-24s %-18s SignNet's call never changes on the ladder" % (label, dname))
        res["series"].append(entry)
        flush(res)

    res["elapsed_seconds"] = time.perf_counter() - t_all
    res["complete"] = True
    flush(res)
    print("wrote", PATH, "in %.1f s" % res["elapsed_seconds"])


if __name__ == "__main__":
    main()
