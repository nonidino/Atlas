"""Export the workbench's S-channel case for the website's "How it works" section.

The site draws the S-channel in code: its outline, the four windows the layout
generates from its shape, the seams where they overlap, and the temperature field
filling in window by window. Nothing is drawn by hand: this script builds the case
exactly as the workbench's example does (`atlas.workbench.spec.example_case`), solves
it decomposed (style B, the windows iterated to agreement) and on the full domain,
and writes what the page needs:

- ``out/site/s-channel.json``: the full export with its provenance (the record);
- ``site/data/s-channel.json``: the compact copy the page loads.

The field is quantised to 0..1000 over the 300-400 K span (0.1 K steps) to keep the
file small; the agreement between the arms is computed on the unquantised fields and
written beside it.

Run:  python scripts/site_export_schannel.py
"""
from __future__ import annotations

import datetime
import json
import os
import platform
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from atlas.workbench import spec as wspec  # noqa: E402
from atlas.workbench.families import conduction  # noqa: E402


def main() -> int:
    case = wspec.example_case("s-channel")
    d = case.domain
    nx, ny = d.nx, d.ny
    run = conduction.build(case, arms=("serial", "full"), threads=1)
    try:
        states = {a: run.step(a, run.initial(a)) for a in ("serial", "full")}
        fields = {a: run.field(states[a]) for a in states}
    finally:
        run.close()
    full, dec = fields["full"], fields["serial"]
    inside = np.isfinite(full)
    t_lo, t_hi = 300.0, 400.0
    diff = float(np.nanmax(np.abs(dec - full)))
    rel = diff / (t_hi - t_lo)
    q = np.full((ny, nx), -1, dtype=int)
    q[inside] = np.clip(np.rint((full[inside] - t_lo) / (t_hi - t_lo) * 1000.0), 0, 1000).astype(int)

    windows = []
    for w in case.windows:
        cells = conduction.cells_of(case, w)
        mask = np.zeros(nx * ny, dtype=bool)
        mask[cells] = True
        windows.append({"id": w.id, "cells": int(mask.sum()), "mask": mask})
    # a cell's window membership as a bit set, row by row
    bits = np.zeros(nx * ny, dtype=int)
    for k, w in enumerate(windows):
        bits[w["mask"]] |= 1 << k
    seams = []
    for a in range(len(windows)):
        for b in range(a + 1, len(windows)):
            n = int((windows[a]["mask"] & windows[b]["mask"]).sum())
            if n:
                seams.append({"between": [windows[a]["id"], windows[b]["id"]], "cells": n})

    out = {
        "what": "the workbench's s-channel example, solved decomposed (style B) and whole",
        "made_by": "scripts/site_export_schannel.py",
        "when": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "grid": {"nx": nx, "ny": ny, "dx_m": d.dx},
        "outline": {"points": [list(p) for p in d.outline.points], "edges": list(d.outline.edges)},
        "span_K": [t_lo, t_hi],
        "field_q": q.ravel().tolist(),
        "field_note": "full-domain temperature, (T - 300 K) / 100 K * 1000, rounded; -1 outside the channel",
        "window_bits": bits.ravel().tolist(),
        "windows": [{"id": w["id"], "cells": w["cells"]} for w in windows],
        "seams": seams,
        "channel_cells": int(inside.sum()),
        "agreement": {"max_abs_K": diff, "relative_to_span": rel,
                      "note": "decomposed (windows iterated to 1e-10) against the full domain, same grid"},
    }
    os.makedirs(os.path.join(ROOT, "out", "site"), exist_ok=True)
    with open(os.path.join(ROOT, "out", "site", "s-channel.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    compact = {k: out[k] for k in ("grid", "outline", "span_K", "field_q", "window_bits",
                                   "windows", "seams", "channel_cells", "agreement")}
    os.makedirs(os.path.join(ROOT, "site", "data"), exist_ok=True)
    with open(os.path.join(ROOT, "site", "data", "s-channel.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(compact, fh, separators=(",", ":"))
        fh.write("\n")
    print(f"s-channel: {nx}x{ny} grid, {inside.sum()} channel cells, {len(windows)} windows, "
          f"{len(seams)} seams; decomposed vs whole: max {diff:.3e} K = {rel:.3e} of the span")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
