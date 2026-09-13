"""Check the car's geometry, and draw it.

    python scripts/car_check.py            # check and draw what is saved
    python scripts/car_check.py --plot-only

Runs the SAME checks the editor's Check button runs -- they are one function,
`car_editor.check`, so the two cannot drift apart -- and writes a picture to
``out/car_geometry.png``.

Exit code is 0 when nothing failed and 1 when something did, so it can be put
in front of a long run.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import car_editor as CE                                             # noqa: E402


def plot(doc, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    from atlas.cases import racelab as RL

    objs, flat = RL.car_bodies(geometry=doc)
    C = {"front-wing": "#3b82f6", "rear-wing": "#3b82f6", "body": "#e5e7eb",
         "floor": "#f59e0b", "duct": "#10b981", "wheel": "#ef4444"}

    fig, ax = plt.subplots(figsize=(14, 5.2), dpi=140)
    fig.patch.set_facecolor("#0f1420")
    ax.set_facecolor("#0f1420")
    try:
        tiling, info = RL.windows_from_geometry(flat, objects=objs)
        for (ox, oy) in tiling.offsets:
            ax.add_patch(plt.Rectangle((ox, oy), tiling.wx, tiling.wy,
                                       fill=False, ec="#2b3550", lw=0.8))
        for lo, hi in tiling.x_bands():
            ax.axvspan(lo, hi, color="#1d4ed8", alpha=0.10, lw=0)
        sub = "%d windows, %.1f%% of the car inside a cut" % (
            tiling.n_windows, 100 * info["banded_force_fraction"])
    except Exception as exc:
        sub = "NO WINDOW LAYOUT: %s" % str(exc)[:70]

    ax.axhline(112, color="#475569", ls=":", lw=1.0)
    ax.text(556, 114, "tiling seam y=112", color="#475569", fontsize=7)
    ax.axvspan(384, 400, color="#3b62d9", alpha=0.16, lw=0)
    for b in flat:
        ax.plot([b.x_le, b.x_te], [b.y_le, b.y_te], "-",
                color=C.get(str(b.group), "#94a3b8"), lw=3.0,
                solid_capstyle="round")
    ax.axhline(0, color="#64748b", lw=1.4)
    ax.set_xlim(20, 660)
    ax.set_ylim(-4, 136)
    ax.set_aspect("equal")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#2b3550")
    ax.set_title("car_geometry.json  —  %d plates + %d wheels, %d flat bodies  —  %s"
                 % (len(doc["plates"]), len(doc["wheels"]), len(flat), sub),
                 color="#e5e7eb", fontsize=10)
    ax.legend(handles=[Line2D([], [], color=v, lw=3, label=k)
                       for k, v in C.items()],
              loc="upper right", fontsize=7, facecolor="#0f1420",
              edgecolor="#2b3550", labelcolor="#cbd5e1", ncol=3)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", facecolor=fig.get_facecolor())
    return out_png


def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default=CE.GEOM)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)

    with open(a.geometry, encoding="utf-8") as fh:
        doc = json.load(fh)

    bad = 0
    if not a.plot_only:
        rep = CE.check(doc)
        for m in rep.get("errors", []):
            print("  FAIL  %s" % m)
        for m in rep.get("warnings", []):
            print("  warn  %s" % m)
        for m in rep.get("ok", []):
            print("  ok    %s" % m)
        bad = len(rep.get("errors", []))
        print()
        print("%d failed, %d warnings" % (bad, len(rep.get("warnings", []))))

    if not a.no_plot:
        p = plot(doc, os.path.join(ROOT, "out", "car_geometry.png"))
        print("picture: %s" % p)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
