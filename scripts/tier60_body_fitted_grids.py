"""Tier 60 -- body-fitted overset grids: the grids, the overlap, and the pressure solve.

    python scripts/tier60_body_fitted_grids.py --out out/racelab9
    python scripts/tier60_body_fitted_grids.py --out out/racelab9 --stages price,summary

On 2026-09-14 the user decided that RaceLab's fluid windows must follow the car
and chose BODY-FITTED GRIDS: curved grids wrapped round each part, overlapping a
Cartesian background, with the bodies as real walls.  That needs a new solver, a
grid generator and a pressure solve that couples every grid.  This tier builds
and verifies the foundation -- `atlas.cases.overset` -- before any car part is
put on it, and prices the pressure solve at the car's size, which is the number
the flow solver's design has to be built round.

Stages, each persisted to ``<out>/<basename>.json``:

  ``poisson``     a manufactured solution at four resolutions: the Cartesian
                  box and the body grid alone are the two single-grid controls;
                  the composite joins them by interpolation, orthogonal and
                  twisted, at interpolation widths 3 and 2.
  ``properties``  what must hold to round-off -- linear fields interpolate
                  exactly, Laplacian rows sum to zero, a linear field's gradient
                  is exact on a twisted grid -- and three refusals: a body grid
                  too thin for its overlap, one that leaves the box, and two
                  whose bodies intersect.
  ``generator``   grids generated round a circle (against the analytic one),
                  an ellipse, a finite-thickness plate and a sharp-edged
                  aerofoil; a dented circle that must be refused; and the
                  composite manufactured solution on a GENERATED grid.
  ``price``       the composite pressure system at the car's size -- the
                  672x240 background and eight body grids -- factored and
                  solved three ways, with the machine's state beside it.
  ``summary``     every registered prediction judged in code.

Nothing here touches RaceLab's column, its records, its gate or the car.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse                                                         # noqa: E402
import datetime as dt                                                   # noqa: E402
import json                                                             # noqa: E402
import math                                                             # noqa: E402
import sys                                                              # noqa: E402
import time                                                             # noqa: E402
import traceback                                                        # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np                                                      # noqa: E402

from atlas.cases import overset as OV                                   # noqa: E402

OUT = NAME = None

LEVELS = (1, 2, 3, 4)

#: The poisson stage's arms: (label, manufactured_poisson keyword arguments).
POISSON_ARMS = (
    ("box", {"case": "box"}),
    ("annulus", {"case": "annulus"}),
    ("annulus_twist", {"case": "annulus", "twist": 0.6}),
    ("composite", {"case": "composite"}),
    ("composite_twist", {"case": "composite", "twist": 0.6}),
    ("composite_width2", {"case": "composite", "width": 2}),
)

#: Generated grids round four shapes, and the refusal.  Thickness 0.3 is
#: about nineteen background cells at the car's h = 1/64.
GEN_THICKNESS = 0.3
GEN_NI, GEN_NJ = 512, 25
#: **Pinned to the smoothing the registered stages ran at.**  G4 failed at 1.0,
#: stage ``diagnose`` found the smoothing responsible, and `overset.KAPPA`
#: became 0.25 -- so every stage measured before that names 1.0 here, and a
#: re-run reproduces the record rather than a different generator.
GEN_KAPPA = 1.0
GEN_SHAPES = {
    "circle": lambda: OV.circle_outline(2.0, 1.5, 0.3),
    "ellipse": lambda: OV.ellipse_outline(2.0, 1.5, 0.6, 0.15, 10.0),
    "plate": lambda: OV.rounded_plate_outline(1.75, 1.45, 0.5, 0.05, 14.0),
    "naca0012": lambda: OV.naca4_outline(1.75, 1.5, 0.5, 0.12, 14.0),
}

#: **Read before this file was written**, and carried into the record.  A smoke
#: run of the module preceded the predictions below; every number here was seen,
#: and the predictions are about what it did NOT measure: the fourth
#: resolution, the generated shapes, the refusals and the price.
READ_BEFORE_THIS_RUN = (
    "Smoke run, 2026-09-14, levels 1-3 (h = 1/16, 1/32, 1/64), max-norm orders "
    "L1->L2, L2->L3: box p 2.00, 2.00; annulus p 1.97, 1.98, grad 1.96, 1.92; "
    "twisted annulus p 2.03, 2.02, grad 1.96, 1.92; composite (width 3) bg p "
    "1.94, 1.96, grad 1.78, 1.87, body p 1.97, 1.98, grad 1.94, 1.94; twisted "
    "composite bg p 1.99, 1.98, grad 1.78, 1.97, body p 2.01, 2.01, grad 1.96, "
    "1.92; composite width 2 bg p 1.91, 1.96, grad 1.57, 1.59, body p 1.93, "
    "1.97, grad 1.93, 1.94.",
    "L3 max errors in that run: box 4.04e-5, annulus 3.00e-4, composite bg "
    "4.52e-4 and body 5.48e-4, width-2 composite bg 4.22e-4 and body 5.08e-4. "
    "Every linear solve's relative residual was below 3e-12.",
    "The same smoke run first showed the BOX gradient at order 1.01 and 1.00: "
    "the background's Dirichlet faces used the linear ghost 2g - p0. It was "
    "replaced by the quadratic ghost (8g - 6p0 + p1)/3 before this file was "
    "written, and the box gradient orders became 1.98, 1.99.",
    "Code properties, 2026-09-14: linear-field interpolation exact to "
    "7.8e-16, 1.1e-15 and 9.4e-16 at widths 2, 3 and 4; a body grid 0.12 "
    "thick against a 0.15 hole margin refused with 180 orphans; body and "
    "background Laplacian row sums at most 3.7e-16 of the diagonal; a linear "
    "field's gradient on the twisted grid exact to 3.2e-14.",
    "Generator: a first, layer-marching version shrank a circle's grid by "
    "1.1e-2 at 96x13 and smoothed more at finer resolution, so it was replaced "
    "by the offset mapping, which reproduced the analytic annulus to 9.0e-8 at "
    "96x13 and 9.3e-8 at 192x25 -- the sag of the 4000-point input polygon.",
    "A stand-in pressure system at the car's size with RANDOM long-range "
    "donor couplings did not finish factoring in several minutes and was "
    "stopped: random couplings defeat any fill-reducing ordering, and real "
    "overset couplings are local. No price is taken from it.",
)

#: **The prediction, registered 2026-09-14 before any stage of this file ran**,
#: judged in code by `judge` under the same ids.  Orders are max-norm, between
#: levels 3 and 4 unless stated.
PREDICTION = (
    {"id": "P1", "claim": "box: p order in [1.95, 2.05] and gradient order in [1.9, 2.1]",
     "why": "the five-point operator and the quadratic ghost are both second order; "
            "L1-L3 read 2.00 and 1.98-1.99"},
    {"id": "P2", "claim": "annulus: p order in [1.9, 2.1], gradient order in [1.8, 2.1]",
     "why": "the nine-point conservative operator and the one-sided Neumann row are "
            "second order on a smooth grid"},
    {"id": "P3", "claim": "twisted annulus: p order in [1.9, 2.1], gradient order in [1.8, 2.1]",
     "why": "the cross terms are discretised at the same order as the rest"},
    {"id": "P4", "claim": "composite, width 3: p order in [1.9, 2.1] on both grids, background "
                          "gradient order >= 1.85, body gradient order in [1.8, 2.1], and the "
                          "body's L4 max error <= 3x the annulus control's",
     "why": "width-3 interpolation is third order once the donor is located on the "
            "same map it interpolates with, so the overlap should cost a constant, not "
            "an order; the background gradient read 1.78 then 1.87 and should keep rising"},
    {"id": "P5", "claim": "twisted composite, width 3: the same bounds as P4",
     "why": "non-orthogonality enters the operator and the donor search, not the order"},
    {"id": "P6", "claim": "composite, width 2: p order >= 1.85 on both grids, and background "
                          "gradient order <= 1.7",
     "why": "bilinear interpolation is second order in value, so its error is O(h^2) "
            "but not smooth from one fringe point to the next, and its gradient there "
            "is O(h); L1-L3 read 1.57, 1.59"},
    {"id": "X1", "claim": "linear fields interpolate exactly (<= 1e-12) at widths 2, 3 and 4; "
                          "every Laplacian row sums to zero (<= 1e-12 of its diagonal); a "
                          "linear field's gradient on the twisted grid is exact (<= 1e-12)",
     "why": "the weights are built on the map Newton inverts, and constants are in the "
            "null space of every stencil"},
    {"id": "X2", "claim": "three refusals raise OversetError: a body grid too thin for its "
                          "hole margin, a body grid that leaves the box, and two bodies whose "
                          "grids enter each other",
     "why": "an overlap that cannot be closed must stop the build, not leak"},
    {"id": "G1", "claim": "the generator reproduces the analytic circle's grid to <= 1e-6 "
                          "with max skew <= 0.01 degrees",
     "why": "smoothing a radial normal field along a circle leaves it radial"},
    {"id": "G2", "claim": "the ellipse, the finite plate at 14 degrees and the sharp-edged "
                          "NACA 0012 at 14 degrees all generate with zero folded cells at "
                          "thickness 0.3, 512 x 25",
     "why": "convex shapes, including a sharp convex corner, fan out rather than fold"},
    {"id": "G3", "claim": "the dented circle is refused (GridQualityError) with the normal "
                          "unsmoothed (kappa 0) AND smoothed (kappa 1)",
     "why": "the dent's concave radius is about 0.003, far inside the first rows"},
    {"id": "G4", "claim": "the composite manufactured solution on the GENERATED ellipse grid "
                          "converges with p order in [1.85, 2.15] on both grids L2->L3 and "
                          "L3->L4, and body gradient order >= 1.75 L3->L4",
     "why": "the offset mapping does not depend on the resolution, so refinement "
            "refines one smooth map"},
    {"id": "T1", "claim": "at the car's size (~200k unknowns) SuperLU factors in <= 30 s "
                          "and one solve takes <= 0.3 s",
     "why": "a 51k-unknown composite solved in 0.4 s including the factorisation; 2-D "
            "local couplings fill as N log N"},
    {"id": "T2", "claim": "AMG-preconditioned BiCGSTAB reaches relative residual 1e-10 in "
                          "<= 100 iterations at the car's size",
     "why": "smoothed aggregation handles the Laplacian blocks; the interpolation rows "
            "are few. Least sure of the fourteen."},
)

#: **What stage ``diagnose`` was written after**: G4 had failed, and an
#: exploration outside this file had already run most of its arms.  Carried
#: into the record, so the diagnosis is read as a diagnosis and not as a test.
READ_BEFORE_DIAGNOSIS = (
    "G4 failed: on the generated ellipse the composite's orders were bg p 1.92, "
    "1.88, 1.79 and body p 1.94, 1.95, 1.74, body gradient 1.13, 1.73, 1.06.",
    "Exploration, 2026-09-14: the ellipse grid ALONE (no background) read 2.02, "
    "1.88, 1.67, and 1.75 at L4->L5, so the overlap is not the cause; a "
    "64000-point input polygon gave errors identical to four figures; the "
    "largest error moved onto the tip (radius of curvature 0.0375) as the grid "
    "refined; a blunter ellipse (0.45 by 0.3, tip radius 0.2) read 1.95, 1.97, "
    "1.99 alone and 1.92-1.99 composite.",
    "At the thin ellipse, alone: thickness 0.15 read 1.84, 1.93, 1.98; kappa "
    "0.25 read 1.89, 1.97, 1.99; kappa 4 read 1.14, 1.56, 1.01.",
    "The overset construction was sped up after the price stage (edges "
    "vectorised, bounding-box screens): on the car-size stand-in, 12.6 s "
    "against 44.6 s in one process, with every point class, donor index and "
    "weight bitwise identical.",
)

#: **The prediction for the adopted default, registered 2026-09-14 before stage
#: ``diagnose`` ran** -- neither of these was measured in the exploration.
PREDICTION_ADOPTED = (
    {"id": "D1", "claim": "at kappa 0.25 the composite on the thin ellipse meets G4's own "
                          "bounds: p order in [1.85, 2.15] on both grids L2->L3 and L3->L4, "
                          "and body gradient order >= 1.75 L3->L4",
     "why": "the grid alone read 1.97 and 1.99 at kappa 0.25, and the overlap has cost "
            "a constant rather than an order on every other grid"},
    {"id": "D2", "claim": "at kappa 0.25 the ellipse, the plate and the NACA 0012 still "
                          "generate with zero folded cells at 512 x 25 and thickness 0.3, "
                          "and the dented circle is still refused",
     "why": "a convex shape needs no smoothing to stay unfolded, and the dent folded "
            "with none"},
)

#: The decision the price stage records.  Not a prediction: a rule written
#: before the number exists, so the number cannot choose the rule.
PRICE_RULE = (
    "Per RaceLab macro-step there are four exchanges and a 0.5 s ceiling "
    "(requirements section 3.1). If one factored solve costs <= 0.05 s, the flow "
    "solver may solve the pressure every exchange by back-substitution; if <= "
    "0.2 s, once per macro-step; above that, the pressure needs an iterative "
    "solve warm-started from the last one, or a smaller system, and that is "
    "Tier 61's first problem."
)


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


def configure(out: str) -> None:
    global OUT, NAME
    OUT = os.path.abspath(out)
    NAME = os.path.basename(os.path.normpath(OUT))


def _retry(fn, tries=6, delay=0.35):
    """OneDrive holds fresh files; os.replace can raise WinError 5."""
    for k in range(tries):
        try:
            return fn()
        except PermissionError:
            if k == tries - 1:
                raise
            time.sleep(delay * (k + 1))


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, float):
        return None if not math.isfinite(x) else x
    if x is None or isinstance(x, (str, int)):
        return x
    return str(x)


def json_path() -> str:
    return os.path.join(OUT, NAME + ".json")


def persist(res) -> str:
    os.makedirs(OUT, exist_ok=True)
    path = json_path()
    tmp = path + ".tmp"

    def write():
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(clean(res), fh, indent=1)

    _retry(write)
    _retry(lambda: os.replace(tmp, path))
    return path


def load_res() -> dict:
    if os.path.isfile(json_path()):
        with open(json_path(), encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def machine_state() -> dict:
    """What else was running, on what power, and how fast a fixed kernel ran.

    `tier54_traced_car.machine_state` plus a reference timing, because this
    tier's prices were taken on battery and a reader needs the scale.
    """
    import subprocess
    out: dict = {}
    try:
        r = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
                           capture_output=True, text=True, timeout=30)
        rows = [ln for ln in r.stdout.splitlines()[1:] if ln.strip() and "No tasks" not in ln]
        out["python_processes_running"] = len(rows)
        out["python_processes"] = rows[:8]
    except Exception as exc:                                     # pragma: no cover
        out["python_processes_running"] = None
        out["tasklist_failed"] = str(exc)
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "(Get-CimInstance Win32_Battery).BatteryStatus"],
                           capture_output=True, text=True, timeout=30)
        st = r.stdout.strip()
        out["battery_status_raw"] = st
        out["on_mains"] = (st == "2") if st else None
    except Exception as exc:                                     # pragma: no cover
        out["on_mains"] = None
        out["battery_query_failed"] = str(exc)
    rng = np.random.default_rng(0)
    a = rng.standard_normal((1024, 1024))
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        a @ a
        ts.append(time.perf_counter() - t0)
    out["reference_matmul_1024_s"] = sorted(ts)[2]
    out["taken_at"] = dt.datetime.now().isoformat(timespec="seconds")
    return out


def say(*a) -> None:
    print(*a, flush=True)


def _orders(errs):
    return [None if (errs[k] is None or errs[k + 1] is None or errs[k + 1] <= 0)
            else math.log2(errs[k] / errs[k + 1]) for k in range(len(errs) - 1)]


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------


def stage_poisson(res) -> dict:
    out = res.setdefault("poisson", {})
    for label, kw in POISSON_ARMS:
        rows = []
        for level in LEVELS:
            t0 = time.perf_counter()
            r = OV.manufactured_poisson(level=level, **kw)
            r["wall_s"] = time.perf_counter() - t0
            say("poisson %-17s L%d n=%7d  %.1f s  res %.1e" % (
                label, level, r["n_unknowns"], r["wall_s"], r["solve"]["relative_residual"]))
            rows.append(r)
        arm = {"kwargs": kw, "levels": rows, "orders": {}}
        for g in rows[0]["grids"]:
            for key in ("max_err", "rms_err", "grad_max_err", "grad_rms_err"):
                arm["orders"][f"{g}.{key}"] = _orders([row["grids"][g][key] for row in rows])
        out[label] = arm
        persist(res)
    return out


def stage_properties(res) -> dict:
    out: dict = {}
    h = 1.0 / 32
    bg = OV.CartesianGrid("bg", 128, 96, h)
    body = OV.ogrid_annulus("body", 2.0, 1.5, 0.3, 0.75, 96, 13, beta=1.0, twist=0.6)
    lin = {}
    for width in (2, 3, 4):
        ov = OV.Overset(bg, [body], hole_margin=0.15, width=width)
        worst = 0.0
        for name, spec in ov.donors.items():
            for e in spec["entries"]:
                donor = bg if e["donor"] == "bg" else body
                DX = donor.X if donor is bg else donor.x
                DY = donor.Y if donor is bg else donor.y
                tx = (bg.X if name == "bg" else body.x)[e["j"], e["i"]]
                ty = (bg.Y if name == "bg" else body.y)[e["j"], e["i"]]
                f = lambda xx, yy: 0.7 * xx - 1.3 * yy + 0.25    # noqa: E731
                val = np.sum(e["weights"] * f(DX.ravel()[e["flat"]], DY.ravel()[e["flat"]]), axis=1)
                worst = max(worst, float(np.abs(val - f(tx, ty)).max()))
        lin[width] = {"worst": worst, "report": ov.report()}
    out["linear_interpolation_worst"] = lin
    ov = OV.Overset(bg, [body], hole_margin=0.15, width=3)
    A, _b = ov.poisson(lambda x, y: np.zeros_like(x), lambda x, y: np.zeros_like(x),
                       lambda g: np.zeros(g.ni))
    rs = np.asarray(A.sum(axis=1)).ravel()
    diag = np.abs(A.diagonal())
    Sb, Sc = ov.status["bg"], ov.status["body"]
    interior = np.zeros_like(Sb, dtype=bool)
    interior[1:-1, 1:-1] = True
    rows = np.concatenate([ov.index["body"][Sc == OV.DISC], ov.index["body"][Sc == OV.WALL],
                           ov.index["bg"][(Sb == OV.DISC) & interior]])
    out["laplacian_row_sum_worst_relative"] = float((np.abs(rs[rows]) / diag[rows]).max())
    interp = np.concatenate([ov.index["bg"][Sb == OV.INTERP], ov.index["body"][Sc == OV.INTERP]])
    out["interpolation_row_sum_worst"] = float(np.abs(rs[interp]).max())
    gx, gy = body.gradient(0.7 * body.x - 1.3 * body.y)
    out["twisted_gradient_of_linear_worst"] = float(max(np.abs(gx - 0.7).max(),
                                                        np.abs(gy + 1.3).max()))
    refusals = {}
    tries = {
        "too_thin": lambda: OV.Overset(bg, [OV.ogrid_annulus("thin", 2.0, 1.5, 0.3, 0.42, 96, 5)],
                                       hole_margin=0.15),
        "leaves_the_box": lambda: OV.Overset(bg, [OV.ogrid_annulus("edge", 0.5, 1.5, 0.3, 0.75, 96, 13)],
                                             hole_margin=0.15),
        "bodies_intersect": lambda: OV.Overset(bg, [
            OV.ogrid_annulus("a", 1.6, 1.5, 0.3, 0.75, 96, 13),
            OV.ogrid_annulus("b", 2.4, 1.5, 0.3, 0.75, 96, 13)], hole_margin=0.15),
    }
    for label, fn in tries.items():
        try:
            fn()
            refusals[label] = {"refused": False}
        except OV.OversetError as exc:
            refusals[label] = {"refused": True, "message": str(exc)[:400]}
    out["refusals"] = refusals
    res["properties"] = out
    persist(res)
    say("properties", json.dumps(clean({k: v for k, v in out.items() if k != "linear_interpolation_worst"}))[:600])
    return out


def stage_generator(res) -> dict:
    out: dict = {"thickness": GEN_THICKNESS, "ni": GEN_NI, "nj": GEN_NJ, "shapes": {}}
    for label, fn in GEN_SHAPES.items():
        try:
            g = OV.ogrid_from_outline(label, fn(), GEN_NI, GEN_NJ, GEN_THICKNESS, beta=2.0,
                                      kappa=GEN_KAPPA)
            rec = {"generated": True, "quality": OV.grid_quality(g)}
            if label == "circle":
                ref = OV.ogrid_annulus("ref", 2.0, 1.5, 0.3, 0.3 + GEN_THICKNESS, GEN_NI, GEN_NJ,
                                       beta=2.0)
                rec["deviation_from_analytic"] = float(max(np.abs(g.x - ref.x).max(),
                                                           np.abs(g.y - ref.y).max()))
        except OV.GridQualityError as exc:
            rec = {"generated": False, "message": str(exc)[:400]}
        out["shapes"][label] = rec
        say("generator", label, json.dumps(clean(rec))[:300])
    dent = {}
    for kappa in (0.0, 1.0):
        try:
            OV.ogrid_from_outline("dent", OV.dented_circle_outline(2.0, 1.5, 0.3), GEN_NI, GEN_NJ,
                                  GEN_THICKNESS, beta=2.0, kappa=kappa)
            dent[str(kappa)] = {"refused": False}
        except OV.GridQualityError as exc:
            dent[str(kappa)] = {"refused": True, "message": str(exc)[:300]}
    out["dented_circle"] = dent
    say("generator dent", json.dumps(dent)[:300])
    rows = []
    for level in LEVELS:
        t0 = time.perf_counter()
        r = OV.manufactured_poisson("composite", level, body="ellipse")
        r["wall_s"] = time.perf_counter() - t0
        rows.append(r)
        say("generator ellipse composite L%d n=%d %.1f s" % (level, r["n_unknowns"], r["wall_s"]))
    arm = {"levels": rows, "orders": {}}
    for gname in rows[0]["grids"]:
        for key in ("max_err", "grad_max_err"):
            arm["orders"][f"{gname}.{key}"] = _orders([row["grids"][gname][key] for row in rows])
    out["ellipse_composite"] = arm
    res["generator"] = out
    persist(res)
    return out


def car_size_standin(kappa: float = GEN_KAPPA) -> tuple[OV.CartesianGrid, list[OV.CurvilinearGrid]]:
    """The 672 x 240 background at h = 1/64 and eight body grids of car-part size.

    A STAND-IN for pricing, not the car: two aerofoils, two wheels, a long body,
    a plate, a pod and a nose, kept apart so this tier's refusals (a grid leaving
    the box or entering another body) do not fire.  What it prices is size.
    """
    h = 1.0 / 64
    bg = OV.CartesianGrid("bg", 672, 240, h)
    parts = {
        "front_wing": OV.naca4_outline(0.9, 0.9, 0.6, 0.12, 14.0),
        "front_wheel": OV.circle_outline(2.5, 1.0, 0.38),
        "body": OV.ellipse_outline(4.9, 1.2, 1.5, 0.3, 0.0),
        "rear_wheel": OV.circle_outline(7.3, 1.0, 0.38),
        "rear_wing": OV.naca4_outline(8.6, 1.5, 0.6, 0.12, 14.0),
        "plate": OV.rounded_plate_outline(8.3, 0.5, 0.8, 0.05, 10.0),
        "pod": OV.ellipse_outline(5.0, 2.6, 0.6, 0.12, 0.0),
        "nose": OV.ellipse_outline(1.6, 2.2, 0.5, 0.1, -8.0),
    }
    grids = []
    for name, outline in parts.items():
        body = OV.orient_clockwise(outline)
        seg = np.hypot(np.roll(body[:, 0], -1) - body[:, 0], np.roll(body[:, 1], -1) - body[:, 1])
        perim_outer = float(seg.sum()) + 2.0 * math.pi * 0.3
        ni = int(64 * math.ceil(perim_outer / h / 64))
        grids.append(OV.ogrid_from_outline(name, outline, ni, 25, 0.3, beta=2.0, kappa=kappa))
    return bg, grids


def stage_price(res) -> dict:
    out: dict = {"rule": PRICE_RULE, "machine_before": machine_state()}
    t0 = time.perf_counter()
    bg, grids = car_size_standin()
    out["grids_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    ov = OV.Overset(bg, grids, hole_margin=0.1, width=3)
    out["overset_s"] = time.perf_counter() - t0
    out["report"] = ov.report()
    out["body_grids"] = {g.name: list(g.shape) for g in grids}

    def rhs(x, y):
        return np.sin(1.1 * x) * np.cos(0.7 * y)

    t0 = time.perf_counter()
    A, b = ov.poisson(rhs, lambda x, y: np.zeros_like(x), lambda g: np.zeros(g.ni))
    out["assemble_s"] = time.perf_counter() - t0
    say("price: n=%d nnz=%d overset %.1f s assemble %.1f s" % (A.shape[0], A.nnz,
                                                                out["overset_s"], out["assemble_s"]))
    persist(res | {"price": out})
    x, info = OV.solve_sparse(A, b, "splu")
    lu = info.pop("_lu")
    solves = []
    for _ in range(5):
        t0 = time.perf_counter()
        lu.solve(b)
        solves.append(time.perf_counter() - t0)
    info["repeat_solve_s"] = solves
    info["median_solve_s"] = sorted(solves)[2]
    out["splu"] = info
    del lu
    say("price splu", json.dumps(clean(info))[:300])
    persist(res | {"price": out})
    for method in ("ilu-gmres", "amg"):
        try:
            _x, inf = OV.solve_sparse(A, b, method, rtol=1e-10)
            out[method] = inf
        except Exception as exc:
            out[method] = {"failed": f"{type(exc).__name__}: {str(exc)[:300]}"}
        say("price", method, json.dumps(clean(out[method]))[:300])
        persist(res | {"price": out})
    out["machine_after"] = machine_state()
    s = out["splu"]["median_solve_s"]
    out["decision"] = ("every exchange" if s <= 0.05 else
                       "once per macro-step" if s <= 0.2 else
                       "iterative, warm-started, or a smaller system")
    res["price"] = out
    persist(res)
    return out


def _arm(levels, **kw) -> dict:
    rows = []
    for level in levels:
        t0 = time.perf_counter()
        r = OV.manufactured_poisson(level=level, **kw)
        r["wall_s"] = time.perf_counter() - t0
        rows.append(r)
    arm = {"kwargs": kw, "levels": rows, "orders": {}}
    for g in rows[0]["grids"]:
        for key in ("max_err", "grad_max_err"):
            arm["orders"][f"{g}.{key}"] = _orders([row["grids"][g][key] for row in rows])
    return arm


#: Stage ``diagnose``: each arm changes ONE thing, either from G4's configuration
#: (the thin ellipse, 0.45 deep, kappa 1, a 4000-point outline, composite) or
#: from G4's grid alone (``grid_alone_kappa1``), which is the control the
#: thickness and smoothing arms are read against.
DIAGNOSE_ARMS = (
    ("polygon_64000_composite", (1, 2, 3, 4),
     {"case": "composite", "body": "ellipse", "n_outline": 64000}),
    ("grid_alone_kappa1", (1, 2, 3, 4, 5), {"case": "annulus", "body": "ellipse"}),
    ("blunt_tip_alone", (1, 2, 3, 4),
     {"case": "annulus", "body": "ellipse", "ellipse": (0.45, 0.3, 10.0)}),
    ("blunt_tip_composite", (1, 2, 3, 4),
     {"case": "composite", "body": "ellipse", "ellipse": (0.45, 0.3, 10.0)}),
    ("thinner_grid_alone", (1, 2, 3, 4), {"case": "annulus", "body": "ellipse", "thickness": 0.15}),
    ("kappa_0.25_alone", (1, 2, 3, 4), {"case": "annulus", "body": "ellipse", "kappa": 0.25}),
    ("kappa_4_alone", (1, 2, 3, 4), {"case": "annulus", "body": "ellipse", "kappa": 4.0}),
    ("adopted_composite", (1, 2, 3, 4), {"case": "composite", "body": "ellipse", "kappa": OV.KAPPA}),
)


def stage_diagnose(res) -> dict:
    out = res.setdefault("diagnose", {})
    for label, levels, kw in DIAGNOSE_ARMS:
        out[label] = _arm(levels, **kw)
        say("diagnose %-24s orders %s" % (label, json.dumps(
            {k: [None if x is None else round(x, 3) for x in v]
             for k, v in out[label]["orders"].items()})))
        persist(res)
    shapes = {}
    for label in ("ellipse", "plate", "naca0012"):
        try:
            g = OV.ogrid_from_outline(label, GEN_SHAPES[label](), GEN_NI, GEN_NJ, GEN_THICKNESS,
                                      beta=2.0, kappa=OV.KAPPA)
            shapes[label] = {"generated": True, "quality": OV.grid_quality(g)}
        except OV.GridQualityError as exc:
            shapes[label] = {"generated": False, "message": str(exc)[:300]}
    try:
        OV.ogrid_from_outline("dent", OV.dented_circle_outline(2.0, 1.5, 0.3), GEN_NI, GEN_NJ,
                              GEN_THICKNESS, beta=2.0, kappa=OV.KAPPA)
        shapes["dented_circle"] = {"refused": False}
    except OV.GridQualityError as exc:
        shapes["dented_circle"] = {"refused": True, "message": str(exc)[:300]}
    out["adopted_shapes"] = {"kappa": OV.KAPPA, "shapes": shapes}
    say("diagnose adopted shapes", json.dumps(clean(shapes))[:400])
    persist(res)
    return out


def stage_build_cost(res) -> dict:
    """The overset construction at the car-size stand-in, with this file's code.

    Stage ``price`` timed it BEFORE the construction was sped up; this reads it
    after, and checks the point counts against the ones that stage recorded.
    """
    out: dict = {"machine": machine_state()}
    bg, grids = car_size_standin()
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        ov = OV.Overset(bg, grids, hole_margin=0.1, width=3)
        times.append(time.perf_counter() - t0)
    out["construction_s"] = times
    out["median_s"] = sorted(times)[1]
    out["report"] = ov.report()
    rec = res.get("price", {}).get("report")
    out["counts_match_the_price_stage"] = (None if rec is None else
                                           rec["grids"] == clean(ov.report())["grids"]
                                           and rec["n_unknowns"] == ov.n_unknowns)
    out["price_stage_construction_s"] = res.get("price", {}).get("overset_s")
    res["build_cost"] = out
    persist(res)
    say("build_cost", json.dumps(clean({k: v for k, v in out.items() if k != "report"}))[:400])
    return out


# ---------------------------------------------------------------------------
# the verdicts
# ---------------------------------------------------------------------------


def _last(arm, key):
    o = arm["orders"][key]
    return o[-1] if o else None


def _in(v, lo, hi):
    return v is not None and lo <= v <= hi


def judge(res) -> dict:
    v: dict = {}
    P = res.get("poisson", {})
    if "box" in P:
        v["P1"] = (_in(_last(P["box"], "bg.max_err"), 1.95, 2.05)
                   and _in(_last(P["box"], "bg.grad_max_err"), 1.9, 2.1))
    if "annulus" in P:
        v["P2"] = (_in(_last(P["annulus"], "body.max_err"), 1.9, 2.1)
                   and _in(_last(P["annulus"], "body.grad_max_err"), 1.8, 2.1))
    if "annulus_twist" in P:
        v["P3"] = (_in(_last(P["annulus_twist"], "body.max_err"), 1.9, 2.1)
                   and _in(_last(P["annulus_twist"], "body.grad_max_err"), 1.8, 2.1))

    def p4like(arm):
        ann = P["annulus"]["levels"][-1]["grids"]["body"]["max_err"]
        comp = arm["levels"][-1]["grids"]["body"]["max_err"]
        return (_in(_last(arm, "bg.max_err"), 1.9, 2.1) and _in(_last(arm, "body.max_err"), 1.9, 2.1)
                and (_last(arm, "bg.grad_max_err") or 0) >= 1.85
                and _in(_last(arm, "body.grad_max_err"), 1.8, 2.1) and comp <= 3.0 * ann)

    if "composite" in P and "annulus" in P:
        v["P4"] = p4like(P["composite"])
    if "composite_twist" in P and "annulus" in P:
        v["P5"] = p4like(P["composite_twist"])
    if "composite_width2" in P:
        arm = P["composite_width2"]
        v["P6"] = ((_last(arm, "bg.max_err") or 0) >= 1.85 and (_last(arm, "body.max_err") or 0) >= 1.85
                   and (_last(arm, "bg.grad_max_err") or 9) <= 1.7)
    X = res.get("properties")
    if X:
        v["X1"] = (max(r["worst"] for r in X["linear_interpolation_worst"].values()) <= 1e-12
                   and X["laplacian_row_sum_worst_relative"] <= 1e-12
                   and X["twisted_gradient_of_linear_worst"] <= 1e-12)
        v["X2"] = all(r["refused"] for r in X["refusals"].values())
    G = res.get("generator")
    if G:
        c = G["shapes"]["circle"]
        v["G1"] = (c.get("generated") and c["deviation_from_analytic"] <= 1e-6
                   and c["quality"]["max_skew_deg"] <= 0.01)
        v["G2"] = all(G["shapes"][s].get("generated") and G["shapes"][s]["quality"]["folded_cells"] == 0
                      for s in ("ellipse", "plate", "naca0012"))
        v["G3"] = all(d["refused"] for d in G["dented_circle"].values())
        arm = G["ellipse_composite"]
        ok = True
        for gname in ("bg", "body"):
            o = arm["orders"][f"{gname}.max_err"]
            ok = ok and _in(o[1], 1.85, 2.15) and _in(o[2], 1.85, 2.15)
        v["G4"] = ok and (arm["orders"]["body.grad_max_err"][2] or 0) >= 1.75
    T = res.get("price")
    if T and "splu" in T:
        v["T1"] = T["splu"]["factor_s"] <= 30.0 and T["splu"]["median_solve_s"] <= 0.3
        amg = T.get("amg", {})
        v["T2"] = (("failed" not in amg) and amg.get("iterations", 999) <= 100
                   and amg.get("relative_residual", 1.0) <= 1e-9)
    D = res.get("diagnose")
    if D and "adopted_composite" in D:
        arm = D["adopted_composite"]
        ok = True
        for gname in ("bg", "body"):
            o = arm["orders"][f"{gname}.max_err"]
            ok = ok and _in(o[1], 1.85, 2.15) and _in(o[2], 1.85, 2.15)
        v["D1"] = ok and (arm["orders"]["body.grad_max_err"][2] or 0) >= 1.75
    if D and "adopted_shapes" in D:
        sh = D["adopted_shapes"]["shapes"]
        v["D2"] = (all(sh[s].get("generated") and sh[s]["quality"]["folded_cells"] == 0
                       for s in ("ellipse", "plate", "naca0012"))
                   and sh["dented_circle"]["refused"])
    return v


def stage_summary(res) -> dict:
    verdicts = judge(res)
    res["verdicts"] = verdicts
    registered = {p["id"] for p in PREDICTION}
    if "prediction_adopted" in res:
        registered |= {p["id"] for p in PREDICTION_ADOPTED}
    res["verdicts_missing"] = sorted(registered - set(verdicts))
    persist(res)
    for p in PREDICTION + (PREDICTION_ADOPTED if "prediction_adopted" in res else ()):
        say("%-3s %-6s %s" % (p["id"], verdicts.get(p["id"]), p["claim"][:110]))
    return verdicts


STAGES = ("poisson", "properties", "generator", "price", "diagnose", "build_cost", "summary")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default=",".join(STAGES))
    args = ap.parse_args(argv)
    configure(args.out)
    res = load_res()
    if "prediction" not in res:
        res.update(tier=60, prediction=list(PREDICTION),
                   prediction_recorded_at=dt.datetime.now().isoformat(timespec="seconds"),
                   read_before_this_run=list(READ_BEFORE_THIS_RUN), price_rule=PRICE_RULE)
        persist(res)
    elif [p["id"] for p in res["prediction"]] != [p["id"] for p in PREDICTION]:
        raise SystemExit("the record's prediction differs from this file's -- a new record, "
                         "not an edit of this one")
    fns = {"poisson": stage_poisson, "properties": stage_properties,
           "generator": stage_generator, "price": stage_price, "diagnose": stage_diagnose,
           "build_cost": stage_build_cost, "summary": stage_summary}
    for st in [s.strip() for s in args.stages.split(",") if s.strip()]:
        if st not in fns:
            raise SystemExit(f"unknown stage {st!r}; stages are {STAGES}")
        if st == "diagnose" and "prediction_adopted" not in res:
            res.update(prediction_adopted=list(PREDICTION_ADOPTED),
                       prediction_adopted_recorded_at=dt.datetime.now().isoformat(timespec="seconds"),
                       read_before_diagnosis=list(READ_BEFORE_DIAGNOSIS))
            persist(res)
        t0 = time.perf_counter()
        try:
            fns[st](res)
        except Exception as exc:
            res.setdefault("stage_errors", {})[st] = traceback.format_exc()[-3000:]
            persist(res)
            say(f"stage {st} FAILED: {type(exc).__name__}: {exc}")
            raise
        res.setdefault("stage_wall_s", {})[st] = time.perf_counter() - t0
        persist(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
