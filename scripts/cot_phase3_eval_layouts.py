"""cot_phase3_eval_layouts -- the coolant plan's evaluation layouts, drawn, hashed, and
their classical floors measured (Phase 3; [[chart-operator-stage-0b-and-1-plan]] sections 5-6).

`atlas/studio/cot_eval_layouts.py` draws three plates at two speeds each.  This writes each
as a studio case file 0.7 under ``out/chartop/cot/eval-layouts/`` with a manifest of their
sha256s, and measures **only the classical floor** of each: the classical 16-mode CO-T
expert on the registered cut against the full-domain solve on the same mesh (relative
energy error in the diffusion norm, largest nodal error and the outlet's mixing-cup error,
each over the temperature range), the quantities E4's bars are set from.  No TL0, no
coarse arm and no learned arm runs here: the module refuses to import one.

BLAS threads are fixed at 4 before numpy loads.  Writes the case files, ``manifest.json``
and ``out/chartop/cot/phase3-eval-floors.json`` (and ``.txt``).

    python scripts/cot_phase3_eval_layouts.py [--out DIR]
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_v] = "4"
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from atlas.chartop import transport as tr  # noqa: E402
from atlas.chartop.host import monolithic  # noqa: E402
from atlas.studio import cot  # noqa: E402
from atlas.studio import cot_eval_layouts as EL  # noqa: E402
from atlas.studio import document as D  # noqa: E402

#: arms that must never run on an evaluation layout before the one evaluation
FORBIDDEN = ("atlas.chartop.transport_coarse", "atlas.chartop.transport_model", "atlas.chartop.transport_torch")


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:
        return "unknown"


def _write(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    for k in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.5 * (k + 1))
    os.replace(tmp, path)


def floors(out_dir: str, rec_dir: str, names=EL.NAMES, tiers=EL.TIERS) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    loaded = [m for m in FORBIDDEN if m in sys.modules]
    if loaded:
        raise RuntimeError(f"a coarse or learned arm's module is loaded ({loaded}): only the classical "
                           f"floor may run on the evaluation layouts")
    manifest = {"what": "the coolant plan's evaluation layouts (atlas/studio/cot_eval_layouts.py)",
                "written": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                "role": "evaluation: hashed in Phase 3; only the classical floor measured before the one evaluation",
                "files": []}
    rows = []
    for name in names:
        for tier in tiers:
            t0 = time.perf_counter()
            spec, geom = EL.build(name, tier)
            text = D.dumps(spec, geom)
            fn = f"{name}-{tier}.json"
            _write(os.path.join(out_dir, fn), text)
            cut = EL.cut(spec, name, tier)
            L = cot.layout(cut, spec)
            pe = max(p.peclet() for p in L.problems)
            mono = monolithic(L)
            fl = L.solve(tr.ClassicalTransportExpert()).errors(mono.fields)
            sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            manifest["files"].append({"file": fn, "sha256": sha, "layout": name, "tier": tier,
                                      "u_coolant": spec.physics.params["u_coolant"],
                                      "cut": {"cutter": "channels", "cols": EL.CUT[name]["cols"], "n": cut.n},
                                      "pieces": len(cut.pieces), "pe_max": round(pe, 3)})
            rows.append({"layout": name, "tier": tier, "file": fn, "sha256": sha, "n": cut.n,
                         "pieces": len(cut.pieces), "pe_max": pe, "nodes": mono.nodes, "floor": fl,
                         "seconds": time.perf_counter() - t0})
            print(f"{fn}: {len(cut.pieces)} pieces at n {cut.n}, Pe max {pe:.1f}; floor energy "
                  f"{fl['energy_rel']:.4g}, max {fl['max_rel']:.3g}, outlet {fl.get('outlet_bulk_err_rel')}",
                  flush=True)
    _write(os.path.join(out_dir, "manifest.json"), json.dumps(manifest, indent=1) + "\n")
    rec = {"what": "the coolant plan's evaluation layouts: their classical floors only (Phase 3)",
           "written": manifest["written"], "code": _commit(), "blas_threads": 4,
           "rule": "only the classical 16-mode expert and the full-domain solve ran; no TL0, coarse or learned arm",
           "manifest_sha256": hashlib.sha256((json.dumps(manifest, indent=1) + "\n").encode("utf-8")).hexdigest(),
           "problems": rows}
    os.makedirs(rec_dir, exist_ok=True)
    _write(os.path.join(rec_dir, "phase3-eval-floors.json"), json.dumps(rec, indent=1) + "\n")
    lines = ["The coolant plan's evaluation layouts: classical floors (16 modes, against the full domain)", ""]
    lines.append(f"{'problem':32s} {'n':>4s} {'pieces':>6s} {'Pe':>6s} {'energy':>9s} {'max':>9s} {'outlet':>9s}")
    for r in rows:
        f = r["floor"]
        lines.append(f"{r['layout'] + ' ' + r['tier']:32s} {r['n']:4d} {r['pieces']:6d} {r['pe_max']:6.1f} "
                     f"{f['energy_rel']:9.3g} {f['max_rel']:9.3g} {(f.get('outlet_bulk_err_rel') or 0.0):9.3g}")
    _write(os.path.join(rec_dir, "phase3-eval-floors.txt"), "\n".join(lines) + "\n")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(HERE, "out", "chartop", "cot", "eval-layouts"))
    ap.add_argument("--records", default=os.path.join(HERE, "out", "chartop", "cot"))
    a = ap.parse_args(argv)
    import winqos
    winqos.opt_out()
    floors(a.out, a.records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
