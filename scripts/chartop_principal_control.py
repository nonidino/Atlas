"""Stage 0 (W355): the control with no trained weights -- the plain square's principal part.

    python scripts/chartop_principal_control.py

On the six benchmark problems (n = 64), the coupled tier-0 answer with **the principal
part alone** (an expert whose correction head is zero: ``H = H0`` at the piece's Fourier
number, ``u_p = 0``) against the full-domain solve: the energy error and the maximum
error over the range, beside the 16-mode classical superelement's floor.  This is the
baseline every learned arm starts from; it involves no network weights, and it is not a
gate.  Writes ``out/chartop/principal-control.json``.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from atlas.chartop import benchmarks as bm, data, host as hs, model as md  # noqa: E402
from atlas.chartop.expert import LearnedExpert  # noqa: E402


def main() -> int:
    ex = LearnedExpert(md.Expert("unet", "0.3M"), envelope=data.ENVELOPE)   # zero head: H = H0
    rows = []
    for name in bm.NAMES:
        for problem in ("steady", "step"):
            kw = {"envelope": data.ENVELOPE} if name == "s-channel" else {}
            b = bm.build(name, 64, problem, **kw)
            L = b.layout("sine-hat", 16)
            mono = hs.monolithic(L)
            span = max(float(np.max(u)) for u in mono.fields) - min(float(np.min(u)) for u in mono.fields)
            out = {"name": name, "problem": problem}
            for label, expert in (("principal_part", ex), ("classical_16", hs.ClassicalExpert())):
                cp = L.solve(expert)
                num, den = cp.energy_error(mono.fields)
                mx = max(float(np.max(np.abs(u - v))) for u, v in zip(mono.fields, cp.fields))
                out[label] = {"energy_rel": num / den, "max_over_range": mx / span}
            print(name, problem, {k: {kk: round(vv, 5) for kk, vv in v.items()} for k, v in out.items()
                                  if isinstance(v, dict)}, flush=True)
            rows.append(out)
    path = os.path.join(ROOT, "out", "chartop", "principal-control.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"written": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                   "rows": rows}, fh, indent=1)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
