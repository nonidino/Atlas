r"""W334 -- Tier 77's four-configuration table, re-measured with the fixed wall.

Tier 77 (case study §10) assembled the rocket's b-c THERM seam four ways -- two
bases (each side's own, or the common interface state lambda*) against two
assemblies (the SUM that Tier 76 used, or the DIFFERENCE that `effort_normal`
declares) -- and found "two errors that nearly cancelled": the configuration
wrong in both respects landed nearest the one right in both. Those numbers were
taken with W301's saturated wall and W332's leak. This recomputes the table with
the test's own construction (`test_two_errors_that_nearly_cancelled`) at the old
root and at the root `w304_seam_base.py` re-measured.

At the cadence Tier 77's table was measured at, ``dt_gas = 1e-5`` (``w304``'s),
unless ``--dt-gas`` says otherwise. ``--old-wall`` puts W301 and W332 back the
way the test's control does: if that reproduces the published table, every
difference the fixed wall shows is the wall's and nothing else Tier 86 changed.

    python scripts/w334_tier77_table.py [--json out/w334/w334_tier77_table.json]
    python scripts/w334_tier77_table.py --old-wall --json out/w334/w334_tier77_table_oldwall.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

OLD_ROOT = 1035.487


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=os.path.join("out", "w334", "w334_tier77_table.json"))
    ap.add_argument("--new-root", type=float, default=None,
                    help="default: out/w334/w304.json's difference_root")
    ap.add_argument("--dt-gas", type=float, default=1.0e-5)
    ap.add_argument("--old-wall", action="store_true",
                    help="revert W301 and W332, as test_tier77's _old_wall does")
    a = ap.parse_args(argv)
    new_root = a.new_root
    if new_root is None:
        with open(os.path.join(ROOT, "out", "w334", "w304.json"), encoding="utf-8") as fh:
            new_root = float(json.load(fh)["lamstar"]["difference_root"])

    spec = importlib.util.spec_from_file_location(
        "t77", os.path.join(ROOT, "tests", "test_tier77_seam_base_and_convention.py"))
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    if a.old_wall:
        class _Patch:
            def setattr(self, obj, name, value):
                setattr(obj, name, value)
        T._old_wall(_Patch())
    shell, gas, g, tr = T._bc(dt_gas=a.dt_gas)
    caps = {x.agent_id: x.capabilities for x in g.agents}
    ports = {"b": "c:THERM", "c": "b:THERM"}
    cache = {}

    def blocks(base):
        if base not in cache:
            blk = {}
            for aid in ("b", "c"):
                P = tr.prolongations[aid]
                bv = None if base is None else P.prolong(T._uniform(P, base), tr.space)
                blk[aid] = T.probe_block(caps[aid], caps[aid].port(ports[aid]), tr.space,
                                         P, "w334", base_V=bv).S
            cache[base] = blk
        return cache[base]

    def row(base, sign_c):
        blk = blocks(base)
        S = blk["b"] + sign_c * blk["c"]
        sv = np.linalg.svd(S, compute_uv=False)
        ev = np.linalg.eigvalsh(0.5 * (S + S.T))
        return dict(sigma_min=float(sv.min()), kappa=float(sv.max() / sv.min()),
                    sym_min=float(ev.min()), sym_max=float(ev.max()),
                    one_signed=bool(ev.min() * ev.max() > 0))

    out = {"old_root_K": OLD_ROOT, "new_root_K": new_root, "dt_gas": a.dt_gas,
           "wall": "old (W301 and W332 reverted)" if a.old_wall else "fixed", "table": {}}
    print("%-34s %-11s %-9s %-24s %s" % ("configuration", "sigma_min", "kappa", "symmetric spectrum", "one-signed"))
    for label, base, sign in (("SUM, own bases (Tier 76)", None, +1.0),
                              ("SUM, old root", OLD_ROOT, +1.0),
                              ("SUM, new root", new_root, +1.0),
                              ("DIFFERENCE, own bases", None, -1.0),
                              ("DIFFERENCE, old root", OLD_ROOT, -1.0),
                              ("DIFFERENCE, new root (corrected)", new_root, -1.0)):
        r = row(base, sign)
        out["table"][label] = r
        print("%-34s %-11.6g %-9.5g [%.5g, %.5g]%s %s" % (label, r["sigma_min"], r["kappa"], r["sym_min"],
              r["sym_max"], " " * 4, r["one_signed"]))
    t = out["table"]
    ww, rw = t["SUM, own bases (Tier 76)"]["sigma_min"], t["SUM, new root"]["sigma_min"]
    wr, rr = t["DIFFERENCE, own bases"]["sigma_min"], t["DIFFERENCE, new root (corrected)"]["sigma_min"]
    out["ratios"] = dict(right_wrong_over_wrong_wrong=rw / ww, wrong_right_over_wrong_wrong=wr / ww,
                         wrong_wrong_over_right_right=ww / rr)
    print("right base, wrong assembly / Tier 76: %.4g   wrong base, right assembly / Tier 76: %.4g   "
          "Tier 76 / corrected: %.4g" % (rw / ww, wr / ww, ww / rr))
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.json)), exist_ok=True)
    with open(os.path.join(ROOT, a.json), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
