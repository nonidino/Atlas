"""arch_paper_figures -- the proposal page's figures, as inline SVG, from real geometry and data.

Every figure is drawn from a measured record, never from invented numbers:

- ``overview``     the pipeline, on the workbench's s-channel (its drawn outline and the
                   layout's own along/across coordinates);
- ``decomposition`` the s-channel cut at equal along values into 9 pieces of conformal
                   modulus about one, with the conformal chart net (``arch_chart_maps``);
- ``charts``       the s-channel's end piece under the conformal, Winslow and transfinite
                   charts (``out/arch/chart_maps_fields.npz``, ``chart_maps.json``);
- ``modes``        the analytic principal parts of three constraint modes on the unit
                   square, steady and for one implicit step, computed here by the same
                   finite-volume discretisation as the probes (64 x 64 cells, unit
                   coefficient);
- ``rounds``       rounds of local solves against the number of pieces
                   (``out/arch/coupling_cost.json``);
- ``gram``         the coupled solution's error, energy Gram against a direct matrix head
                   (``out/arch/superelement_gram.json``);
- ``forward``      the forward cost of the candidate backbones against two classical
                   costs (``out/arch/backbone_timing.json``, ``coupling_cost.json``).

The SVGs use CSS classes that the page defines, so they follow its light and dark themes.
Writes ``out/arch/figures/<name>.svg``; ``scripts/arch_paper_build.py`` inlines them.

    set PYTHONIOENCODING=utf-8
    python scripts/arch_paper_figures.py
"""

from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))

import numpy as np                                                     # noqa: E402
import scipy.sparse as sp                                              # noqa: E402
import scipy.sparse.linalg as spla                                     # noqa: E402

ARCH = os.path.join(HERE, "out", "arch")
OUT = os.path.join(ARCH, "figures")


# ---------------------------------------------------------------------------
# SVG primitives
# ---------------------------------------------------------------------------


def f1(v: float) -> str:
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


def path_d(pts) -> str:
    pts = np.asarray(pts, dtype=float)
    return "M" + " L".join(f"{f1(x)},{f1(y)}" for x, y in pts)


class Svg:
    def __init__(self, w: float, h: float, label: str):
        self.w, self.h, self.label = w, h, label
        self.items: list[str] = []

    def add(self, s: str) -> None:
        self.items.append(s)

    def line(self, x0, y0, x1, y1, cls="ln", extra=""):
        self.add(f'<line x1="{f1(x0)}" y1="{f1(y0)}" x2="{f1(x1)}" y2="{f1(y1)}" class="{cls}"{extra}/>')

    def poly(self, pts, cls="ln", closed=False, extra=""):
        if len(pts) < 2:
            return
        self.add(f'<path d="{path_d(pts)}{" Z" if closed else ""}" class="{cls}"{extra}/>')

    def text(self, x, y, s, cls="", anchor="start", size=None, extra=""):
        a = f' text-anchor="{anchor}"' if anchor != "start" else ""
        z = f' font-size="{size}"' if size else ""
        c = f' class="{cls}"' if cls else ""
        self.add(f'<text x="{f1(x)}" y="{f1(y)}"{a}{z}{c}{extra}>{s}</text>')

    def arrow(self, x0, y0, x1, y1, head=6.0):
        ang = math.atan2(y1 - y0, x1 - x0)
        self.line(x0, y0, x1 - head * 0.8 * math.cos(ang), y1 - head * 0.8 * math.sin(ang), "arrow")
        pts = [(x1, y1),
               (x1 - head * math.cos(ang) + 0.5 * head * math.sin(ang), y1 - head * math.sin(ang) - 0.5 * head * math.cos(ang)),
               (x1 - head * math.cos(ang) - 0.5 * head * math.sin(ang), y1 - head * math.sin(ang) + 0.5 * head * math.cos(ang))]
        self.poly(pts, "arrowhead", closed=True)

    def write(self, name: str) -> str:
        body = "\n".join(self.items)
        svg = (f'<svg class="figsvg" viewBox="0 0 {f1(self.w)} {f1(self.h)}" role="img" '
               f'aria-label="{self.label}" xmlns="http://www.w3.org/2000/svg">\n{body}\n</svg>\n')
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, name + ".svg"), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(svg)
        return svg


# ---------------------------------------------------------------------------
# contouring (marching squares) and polyline simplification
# ---------------------------------------------------------------------------


def contour(F: np.ndarray, level: float) -> list[np.ndarray]:
    """Level set of F (values at points (j, i), NaN outside) as polylines in index units."""
    a, b = F[:-1, :-1], F[:-1, 1:]
    c, d = F[1:, 1:], F[1:, :-1]
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & np.isfinite(d)
    code = ((a > level) * 1 + (b > level) * 2 + (c > level) * 4 + (d > level) * 8)
    cells = np.argwhere(ok & (code > 0) & (code < 15))
    segs = []

    def interp(p, q, fp, fq):
        t = (level - fp) / (fq - fp)
        return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))

    for i, j in cells:
        va, vb, vc, vd = F[i, j], F[i, j + 1], F[i + 1, j + 1], F[i + 1, j]
        pa, pb, pc, pd = (j, i), (j + 1, i), (j + 1, i + 1), (j, i + 1)
        e = {
            "ab": lambda: interp(pa, pb, va, vb), "bc": lambda: interp(pb, pc, vb, vc),
            "cd": lambda: interp(pc, pd, vc, vd), "da": lambda: interp(pd, pa, vd, va),
        }
        k = int(code[i, j])
        table = {1: [("da", "ab")], 2: [("ab", "bc")], 3: [("da", "bc")], 4: [("bc", "cd")],
                 6: [("ab", "cd")], 7: [("da", "cd")], 8: [("cd", "da")], 9: [("ab", "cd")],
                 11: [("bc", "cd")], 12: [("bc", "da")], 13: [("ab", "bc")], 14: [("da", "ab")]}
        if k in (5, 10):
            centre = 0.25 * (va + vb + vc + vd) > level
            if k == 5:
                pairs = [("da", "cd"), ("ab", "bc")] if centre else [("da", "ab"), ("bc", "cd")]
            else:
                pairs = [("ab", "da"), ("bc", "cd")] if centre else [("ab", "bc"), ("cd", "da")]
        else:
            pairs = table[k]
        for p, q in pairs:
            segs.append((e[p](), e[q]()))
    return chain(segs)


def chain(segs) -> list[np.ndarray]:
    key = lambda p: (round(p[0], 6), round(p[1], 6))                 # noqa: E731
    ends: dict = {}
    for n, (p, q) in enumerate(segs):
        ends.setdefault(key(p), []).append((n, 0))
        ends.setdefault(key(q), []).append((n, 1))
    used = [False] * len(segs)
    lines = []
    for n in range(len(segs)):
        if used[n]:
            continue
        used[n] = True
        pts = [segs[n][0], segs[n][1]]
        for direction in (1, 0):
            while True:
                tip = pts[-1] if direction == 1 else pts[0]
                nxt = None
                for m, side in ends.get(key(tip), []):
                    if not used[m]:
                        nxt = (m, side)
                        break
                if nxt is None:
                    break
                m, side = nxt
                used[m] = True
                other = segs[m][1 - side]
                if direction == 1:
                    pts.append(other)
                else:
                    pts.insert(0, other)
        lines.append(np.array(pts))
    return lines


def rdp(pts: np.ndarray, tol: float) -> np.ndarray:
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    L = math.hypot(*ab)
    if L == 0:
        dists = np.hypot(*(pts - a).T)
    else:
        dists = np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / L
    k = int(np.argmax(dists))
    if dists[k] > tol:
        left = rdp(pts[: k + 1], tol)
        right = rdp(pts[k:], tol)
        return np.vstack([left[:-1], right])
    return np.vstack([a, b])


class Frame:
    """An affine map from data units into a panel of the SVG, preserving aspect ratio."""

    def __init__(self, xmin, xmax, ymin, ymax, x0, y0, w, h, flip_y=False):
        s = min(w / (xmax - xmin), h / (ymax - ymin))
        self.s = s
        self.ox = x0 + 0.5 * (w - s * (xmax - xmin)) - s * xmin
        self.oy = y0 + 0.5 * (h - s * (ymax - ymin)) - s * ymin
        self.flip, self.y0, self.h = flip_y, y0, h
        self.ymin, self.ymax = ymin, ymax

    def __call__(self, pts):
        pts = np.asarray(pts, dtype=float)
        x = self.ox + self.s * pts[..., 0]
        if self.flip:
            y = self.oy + self.s * (self.ymax + self.ymin - pts[..., 1])
        else:
            y = self.oy + self.s * pts[..., 1]
        return np.stack([x, y], axis=-1)


def draw_contours(svg, frame, F, levels, cls, offset=(0.5, 0.5), tol=0.25, extra=""):
    for lv in levels:
        for ln in contour(F, lv):
            pts = frame(ln + np.array(offset))
            svg.poly(rdp(pts, tol), cls, extra=extra)


# ---------------------------------------------------------------------------
# the workbench geometry
# ---------------------------------------------------------------------------


def s_channel():
    import arch_chart_maps as CM
    from atlas.workbench import geometry as geo
    D = CM.Domain("s-channel")
    act = D.act
    U = np.where(act, D.along, np.nan)
    V = np.where(act, D.across, np.nan)
    edges = [np.asarray(p, dtype=float) for _, p in geo.drawn_edges(D.d)]
    return D, U, V, edges


def corner_angle(p, q1, q2) -> float:
    v1, v2 = np.asarray(q1) - p, np.asarray(q2) - p
    c = float(np.dot(v1, v2) / (np.hypot(*v1) * np.hypot(*v2)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def piece_polygon(f, tag):
    xa, xb, xc, xd = (f[f"{tag}__side_{s}"] for s in "ABCD")
    return np.vstack([xb, xc, xd[::-1], xa[::-1]]), (xa, xb, xc, xd)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def fig_decomposition(D, U, V, edges):
    ny, nx = U.shape
    svg = Svg(680, 400, "The s-channel decomposed into nine pieces at equal values of the along coordinate")
    fr = Frame(0, nx, 0, ny, 10, 10, 660, 360)
    pid = np.load(os.path.join(ARCH, "chart_maps_fields.npz"))["equal_value_s-channel_9x1__pid"]
    # shade the end piece, which Figure (charts) examines
    f = np.load(os.path.join(ARCH, "chart_maps_fields.npz"))
    poly, _ = piece_polygon(f, "s-channel_9x1_F00")
    svg.poly(fr(poly), "piece-fill", closed=True)
    draw_contours(svg, fr, U, [k / 36 for k in range(1, 36) if k % 4], "net")
    draw_contours(svg, fr, V, [k / 6 for k in range(1, 6)], "net")
    draw_contours(svg, fr, U, [k / 9 for k in range(1, 9)], "cut")
    for e in edges:
        svg.poly(fr(e), "wall")
    for k in range(9):
        ii, jj = np.nonzero(pid == k)
        if ii.size:
            cx, cy = fr(np.array([jj.mean() + 0.5, ii.mean() + 0.5]))
            svg.text(cx, cy + 5, f"P<tspan baseline-shift=\"sub\" font-size=\"10\">{k + 1}</tspan>",
                     anchor="middle", size=14, cls="lbl")
    svg.text(340, 392, "thin lines: level sets of the along and across coordinates; "
             "heavy lines: the eight equal-value cuts", anchor="middle", size=12, cls="muted")
    return svg.write("decomposition")


def fig_charts():
    f = np.load(os.path.join(ARCH, "chart_maps_fields.npz"))
    cmp_ = json.load(open(os.path.join(ARCH, "chart_maps.json"), encoding="utf-8"))["compare"][0]
    tag = "s-channel_9x1_F00"
    poly, (xa, xb, xc, xd) = piece_polygon(f, tag)
    xmin, ymin = poly.min(axis=0) - 2
    xmax, ymax = poly.max(axis=0) + 2
    svg = Svg(690, 330, "The s-channel end piece under three charts")
    W = 220
    names = [("conformal", "conformal (the layout's own pair)"), ("winslow", "Winslow (harmonic, arc length)"),
             ("transfinite", "transfinite interpolation")]
    # corner angles, read over the first probe's chord of CHORD_CELLS cells
    import arch_chart_probe as CP
    corners = [(xa[0], CP._chord(xa, True), CP._chord(xb, True)),
               (xa[-1], CP._chord(xa, False), CP._chord(xd, True)),
               (xc[0], CP._chord(xc, True), CP._chord(xb, False)),
               (xc[-1], CP._chord(xc, False), CP._chord(xd, False))]
    angles = [corner_angle(np.zeros(2), t1, t2) for _, t1, t2 in corners]
    for n, (key, title) in enumerate(names):
        x0 = 10 + n * (W + 10)
        fr = Frame(xmin, xmax, ymin, ymax, x0, 30, W, 230)
        svg.poly(fr(poly), "piece-fill", closed=True)
        if key == "transfinite":
            X = f[f"{tag}__tfi_X"]
            m = X.shape[0] - 1
            idx = sorted(set([round(k * m / 8) for k in range(9)]))
            for i in idx:
                svg.poly(rdp(fr(X[i, :, :]), 0.2), "net2")
                svg.poly(rdp(fr(X[:, i, :]), 0.2), "net2")
        else:
            act = f[f"{tag}__act"]
            XI = np.where(act, f[f"{tag}__xi_{key}"], np.nan)
            ETA = np.where(act, f[f"{tag}__eta_{key}"], np.nan)
            draw_contours(svg, fr, XI, [k / 8 for k in range(1, 8)], "net2", tol=0.2)
            draw_contours(svg, fr, ETA, [k / 8 for k in range(1, 8)], "net2", tol=0.2)
        svg.poly(fr(poly), "wall", closed=True)
        if n == 0:
            for (p, _, _), ang in zip(corners, angles):
                if abs(ang - 90) > 10:
                    px, py = fr(np.asarray(p))
                    dx = -6 if px < x0 + W / 2 else 6
                    svg.text(px + dx, py + (4 if py > 150 else 10), f"{ang:.0f}&#176;",
                             anchor=("end" if dx < 0 else "start"), size=12, cls="lbl")
        r = cmp_[key]
        svg.text(x0 + W / 2, 18, title, anchor="middle", size=13)
        svg.text(x0 + W / 2, 282, f"J<tspan baseline-shift=\"sub\" font-size=\"9\">max</tspan>/"
                 f"J<tspan baseline-shift=\"sub\" font-size=\"9\">min</tspan> = {r['J_ratio']:.3g}",
                 anchor="middle", size=13)
        svg.text(x0 + W / 2, 302, f"|A&#770;<tspan baseline-shift=\"sub\" font-size=\"9\">12</tspan>| "
                 f"95th pct = {r['K12_abs_p95']:.2f}", anchor="middle", size=13)
    return svg.write("charts"), angles


def fv_square(n: int, fo, side_data: np.ndarray) -> np.ndarray:
    """Cell-centred finite volumes on the unit square (n x n, unit coefficient), Dirichlet
    face data on the bottom side (side_data at its n faces), zero on the other three."""
    N = n * n
    idx = np.arange(N).reshape(n, n)
    cap = 0.0 if fo is None else 1.0 / (fo * n * n)
    rows, cols, vals = [], [], []
    diag = np.full(N, cap)
    for a, b in ((idx[:, :-1], idx[:, 1:]), (idx[:-1, :], idx[1:, :])):
        a, b = a.ravel(), b.ravel()
        rows += [a, b]; cols += [b, a]; vals += [-np.ones(a.size), -np.ones(a.size)]
        np.add.at(diag, a, 1.0); np.add.at(diag, b, 1.0)
    rhs = np.zeros(N)
    for sl in (idx[:, 0], idx[:, -1], idx[0, :], idx[-1, :]):
        np.add.at(diag, sl.ravel(), 2.0)
    rhs[idx[0, :]] += 2.0 * side_data
    rows.append(np.arange(N)); cols.append(np.arange(N)); vals.append(diag)
    A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    return spla.spsolve(A, rhs).reshape(n, n)


def fig_modes():
    n = 64
    s = (np.arange(n) + 0.5) / n
    ks = (0, 1, 4)
    rows = ((None, "steady"), (0.01, "one implicit step, Fo = 0.01"))
    svg = Svg(680, 490, "Principal parts of three constraint modes on the unit square")
    P = 150
    levels = [x / 10 for x in range(-9, 10) if x != 0]
    for r, (fo, rtitle) in enumerate(rows):
        y0 = 60 + r * 215
        svg.text(30, y0 - 8, rtitle, size=12.5, cls="muted")
        for c, k in enumerate(ks):
            x0 = 30 + c * 215
            F = fv_square(n, fo, np.cos(k * math.pi * s))
            fr = Frame(0, n, 0, n, x0, y0, P, P, flip_y=True)
            svg.add(f'<rect x="{f1(x0)}" y="{f1(y0)}" width="{P}" height="{P}" class="sq"/>')
            pos = [lv for lv in levels if lv > 0]
            neg = [lv for lv in levels if lv < 0]
            draw_contours(svg, fr, F, pos, "pos", offset=(0.5, 0.5), tol=0.2)
            draw_contours(svg, fr, F, neg, "neg", offset=(0.5, 0.5), tol=0.2)
            # the trace on the bottom side
            tr = np.column_stack([s * P + x0, y0 + P + 22 - 14 * np.cos(k * math.pi * s)])
            svg.line(x0, y0 + P + 22, x0 + P, y0 + P + 22, "axis-thin")
            svg.poly(tr, "trace")
            if r == 0:
                svg.text(x0 + P / 2, 22, f"k = {k}", anchor="middle", size=13)
    svg.text(340, 482, "contours at multiples of 0.1 (solid positive, dashed negative); "
             "below each square, the prescribed trace cos(k&#960;&#958;) on the bottom side",
             anchor="middle", size=12, cls="muted")
    return svg.write("modes")


def log_axis(svg, fr_x, fr_y, xt, yt, xlab, ylab, x0, y0, w, h, xfmt=None, yfmt=None):
    svg.add(f'<rect x="{f1(x0)}" y="{f1(y0)}" width="{f1(w)}" height="{f1(h)}" class="plotbox"/>')
    for t in xt:
        X = fr_x(t)
        svg.line(X, y0 + h, X, y0 + h + 5, "axis")
        svg.line(X, y0, X, y0 + h, "gridln")
        svg.text(X, y0 + h + 19, xfmt(t) if xfmt else f"{t:g}", anchor="middle", size=12)
    for t in yt:
        Y = fr_y(t)
        svg.line(x0 - 5, Y, x0, Y, "axis")
        svg.line(x0, Y, x0 + w, Y, "gridln")
        svg.text(x0 - 8, Y + 4, yfmt(t) if yfmt else f"{t:g}", anchor="end", size=12)
    svg.text(x0 + w / 2, y0 + h + 40, xlab, anchor="middle", size=13)
    svg.add(f'<text x="{f1(x0 - 52)}" y="{f1(y0 + h / 2)}" text-anchor="middle" font-size="13" '
            f'transform="rotate(-90 {f1(x0 - 52)} {f1(y0 + h / 2)})">{ylab}</text>')


def logmap(lo, hi, a, b):
    la, lb = math.log10(lo), math.log10(hi)
    return lambda v: a + (math.log10(v) - la) / (lb - la) * (b - a)


def marker(svg, x, y, cls, shape="o", open_=False):
    fillcls = f"mk-open {cls}" if open_ else f"mk {cls}"
    if shape == "o":
        svg.add(f'<circle cx="{f1(x)}" cy="{f1(y)}" r="4" class="{fillcls}"/>')
    elif shape == "s":
        svg.add(f'<rect x="{f1(x - 4)}" y="{f1(y - 4)}" width="8" height="8" class="{fillcls}"/>')
    else:
        svg.add(f'<path d="M{f1(x)},{f1(y - 5)} L{f1(x + 5)},{f1(y + 4)} L{f1(x - 5)},{f1(y + 4)} Z" class="{fillcls}"/>')


def fig_rounds():
    data = json.load(open(os.path.join(ARCH, "coupling_cost.json"), encoding="utf-8"))["results"]
    sel = [r for r in data if r["n"] == 32 and r["Fo_piece"] is None and r["pieces"][0] == r["pieces"][1]]
    sel.sort(key=lambda r: r["pieces"][0])
    P = [r["pieces"][0] * r["pieces"][1] for r in sel]
    series = [("RAS-Rich", "classical Schwarz (RAS)", "c-b", "o"),
              ("OSM-Rich", "optimised Schwarz (Robin)", "c-d", "s"),
              ("OSM-AA", "optimised Schwarz + Anderson", "c-c", "t"),
              ("RAS-GMRES", "RAS inside GMRES", "c-g", "o"),
              ("OSM-GMRES", "optimised Schwarz inside GMRES", "c-a", "s"),
              ("2L-GMRES", "two-level (coarse space) + GMRES", "c-l", "t")]
    svg = Svg(680, 360, "Rounds of local solves against the number of pieces")
    x0, y0, w, h = 80, 20, 330, 270
    fx = logmap(3, 80, x0, x0 + w)
    fy = logmap(10, 3000, y0 + h, y0)
    log_axis(svg, fx, fy, P, [10, 30, 100, 300, 1000, 3000], "number of pieces (square layouts, 32 &#215; 32 cells each)",
             "rounds to relative residual 10&#8315;&#8312;", x0, y0, w, h)
    for n, (key, lab, cls, shp) in enumerate(series):
        ys = [r[key]["iterations"] for r in sel]
        pts = [(fx(p), fy(y)) for p, y in zip(P, ys)]
        svg.poly(pts, f"series {cls}")
        for x, y in pts:
            marker(svg, x, y, cls, shp)
        ly = 40 + n * 24
        svg.line(430, ly, 458, ly, f"series {cls}")
        marker(svg, 444, ly, cls, shp)
        svg.text(466, ly + 4, lab, size=12.5)
        if key in ("RAS-Rich", "OSM-Rich", "OSM-AA"):
            svg.text(fx(P[-1]) + 8, fy(ys[-1]) + 4, f"{ys[-1]}", size=11, cls="muted")
    svg.text(430, 40 + 6 * 24 + 8, "superelement (port matrices assembled", size=12.5)
    svg.text(430, 40 + 6 * 24 + 26, "and one interface solve): no rounds", size=12.5)
    return svg.write("rounds")


def fig_gram():
    g = json.load(open(os.path.join(ARCH, "superelement_gram.json"), encoding="utf-8"))
    svg = Svg(680, 360, "Coupled solution error: energy Gram against a direct matrix head")
    x0, y0, w, h = 80, 20, 330, 270
    fx = logmap(0.007, 0.4, x0, x0 + w)
    fy = logmap(1e-4, 30, y0 + h, y0)
    log_axis(svg, fx, fy, [0.01, 0.03, 0.1, 0.3], [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
             "relative perturbation &#949; of the expert's output",
             "relative L&#178; error of the coupled solution", x0, y0, w, h,
             yfmt=lambda t: {1e-4: "0.0001", 1e-3: "0.001", 1e-2: "0.01", 1e-1: "0.1", 1: "1", 10: "10"}[t])
    leg = 0
    for run in g["results"]:
        fo = run["Fo_piece"]
        tagname = "steady" if fo is None else f"Fo = {fo:g}"
        rows = run["rows"]
        for key, cls, shp, lab in (("gram", "c-a", "o", "energy Gram"), ("direct", "c-b", "s", "direct head")):
            xs = [r["eps"] for r in rows]
            ys = [r[f"{key}_solution_rel_L2"] for r in rows]
            indef = [r[f"{key}_assembled_min_eig"] < 0 for r in rows]
            dash = "" if fo is None else " dashed"
            pts = [(fx(x), fy(y)) for x, y in zip(xs, ys)]
            svg.poly(pts, f"series {cls}{dash}")
            for (X, Y), ind in zip(pts, indef):
                marker(svg, X, Y, cls, shp, open_=ind)
            ly = 40 + leg * 24
            svg.line(430, ly, 458, ly, f"series {cls}{dash}")
            marker(svg, 444, ly, cls, shp)
            svg.text(466, ly + 4, f"{lab}, {tagname}", size=12.5)
            leg += 1
        floor = run.get("truncation_floor_rel_L2")
        if floor and fo is None:
            Y = fy(floor)
            svg.line(x0, Y, x0 + w, Y, "floor")
            svg.text(x0 + 6, Y - 5, "16-mode truncation floor", size=11, cls="muted")
    ly = 40 + leg * 24 + 6
    marker(svg, 444, ly, "c-b", "s", open_=True)
    svg.text(466, ly + 4, "open marker: assembled interface", size=12.5)
    svg.text(466, ly + 22, "matrix indefinite", size=12.5)
    return svg.write("gram")


def fig_forward():
    bt = json.load(open(os.path.join(ARCH, "backbone_timing.json"), encoding="utf-8"))["rows"]
    cc = json.load(open(os.path.join(ARCH, "coupling_cost.json"), encoding="utf-8"))["results"]
    case = next(r for r in cc if r["n"] == 64)
    npieces = case["pieces"][0] * case["pieces"][1]
    recompute_ms = 1e3 * (case["setup"]["substructure_factor_s"] + case["Port-16"]["form_port_matrices_s"]) / npieces
    solve_ms = 1e3 * case["RAS-Rich"]["time_per_iteration_s"] / npieces
    bars = []
    for size, lab in (("small", "small, 0.49 M"), ("medium", "medium, 1.95 M"), ("large", "large, 6.2 M")):
        r1 = next(r for r in bt if r["size"] == size and r["n"] == 64 and r["batch"] == 1)
        r16 = next(r for r in bt if r["size"] == size and r["n"] == 64 and r["batch"] == 16)
        bars.append((f"network {lab}", r1["ms_per_piece"], r16["ms_per_piece"], "c-a"))
    svg = Svg(680, 300, "Forward cost per piece of the candidate backbones against classical costs, 64 x 64 cells")
    x0, y0, w = 250, 20, 400
    fx = logmap(0.3, 300, x0, x0 + w)
    rows = bars + [("classical: one local solve", solve_ms, None, "c-c"),
                   ("classical: recompute the port matrix", recompute_ms, None, "c-b")]
    bh, gap = 26, 14
    h = len(rows) * (bh + gap)
    svg.add(f'<rect x="{f1(x0)}" y="{f1(y0)}" width="{f1(w)}" height="{f1(h)}" class="plotbox"/>')
    for t in (0.3, 1, 3, 10, 30, 100, 300):
        X = fx(t)
        svg.line(X, y0, X, y0 + h, "gridln")
        svg.line(X, y0 + h, X, y0 + h + 5, "axis")
        svg.text(X, y0 + h + 19, f"{t:g}", anchor="middle", size=12)
    svg.text(x0 + w / 2, y0 + h + 40, "milliseconds per piece (log scale; this container's CPU, 4 threads)",
             anchor="middle", size=13)
    for n, (lab, v1, v16, cls) in enumerate(rows):
        y = y0 + gap / 2 + n * (bh + gap)
        svg.text(x0 - 8, y + bh / 2 + 4, lab, anchor="end", size=12.5)
        if v16 is None:
            svg.add(f'<rect x="{f1(x0)}" y="{f1(y)}" width="{f1(fx(v1) - x0)}" height="{bh}" class="bar {cls}"/>')
            svg.text(fx(v1) + 6, y + bh / 2 + 4, f"{v1:.3g} ms", size=11.5)
        else:
            svg.add(f'<rect x="{f1(x0)}" y="{f1(y)}" width="{f1(fx(v1) - x0)}" height="{bh / 2 - 1}" class="bar {cls}"/>')
            svg.add(f'<rect x="{f1(x0)}" y="{f1(y + bh / 2 + 1)}" width="{f1(fx(v16) - x0)}" height="{bh / 2 - 1}" '
                    f'class="bar-light {cls}"/>')
            svg.text(fx(v1) + 6, y + bh / 2 - 2, f"{v1:.3g} ms alone", size=11)
            svg.text(fx(v16) + 6, y + bh - 1, f"{v16:.3g} ms in a batch of 16", size=11)
    return svg.write("forward"), recompute_ms, solve_ms


def fig_overview(D, U, V, edges):
    f = np.load(os.path.join(ARCH, "chart_maps_fields.npz"))
    ny, nx = U.shape
    svg = Svg(690, 300, "Overview of the chart operator pipeline")
    # (a) the domain and its decomposition
    fa = Frame(0, nx, 0, ny, 5, 40, 200, 150)
    poly, _ = piece_polygon(f, "s-channel_9x1_F00")
    svg.poly(fa(poly), "piece-fill", closed=True)
    draw_contours(svg, fa, U, [k / 9 for k in range(1, 9)], "cut-thin", tol=0.3)
    for e in edges:
        svg.poly(fa(e), "wall-thin")
    svg.text(105, 22, "(a) decomposition", anchor="middle", size=13)
    svg.text(105, 215, "topology: four-sided pieces", anchor="middle", size=11.5, cls="muted")
    # (b) one piece and its chart onto the square
    xmin, ymin = poly.min(axis=0) - 1
    xmax, ymax = poly.max(axis=0) + 1
    fb = Frame(xmin, xmax, ymin, ymax, 215, 55, 80, 120)
    svg.poly(fb(poly), "piece-fill", closed=True)
    svg.poly(fb(poly), "wall-thin", closed=True)
    sx, sy, S = 335, 75, 80
    svg.add(f'<rect x="{sx}" y="{sy}" width="{S}" height="{S}" class="sq-in"/>')
    for k in range(1, 4):
        svg.line(sx + k * S / 4, sy, sx + k * S / 4, sy + S, "net")
        svg.line(sx, sy + k * S / 4, sx + S, sy + k * S / 4, "net")
    svg.arrow(300, 115, 330, 115)
    svg.text(314, 105, "chart", anchor="middle", size=11.5)
    svg.text(315, 22, "(b) certified chart", anchor="middle", size=13)
    svg.text(375, 175, "log J, &#956;, log &#954;, Fo", anchor="middle", size=11.5)
    svg.text(315, 215, "geometry: an anisotropic", anchor="middle", size=11.5, cls="muted")
    svg.text(315, 230, "coefficient on the square", anchor="middle", size=11.5, cls="muted")
    # (c) the learned operator and its constraint modes
    svg.arrow(420, 115, 442, 115)
    svg.add('<rect x="444" y="95" width="56" height="40" rx="4" class="netbox"/>')
    svg.text(472, 120, "N<tspan baseline-shift=\"sub\" font-size=\"10\">&#952;</tspan>", anchor="middle", size=15)
    svg.arrow(503, 115, 522, 115)
    for k in range(4):
        x, y = 526 + 7 * k, 80 + 7 * k
        svg.add(f'<rect x="{x}" y="{y}" width="44" height="44" class="sq-out"/>')
    svg.text(560, 22, "(c) constraint modes", anchor="middle", size=13)
    svg.text(560, 160, "H&#770;<tspan baseline-shift=\"sub\" font-size=\"9\">k</tspan> = H&#770;<tspan "
             "baseline-shift=\"sub\" font-size=\"9\">k</tspan><tspan baseline-shift=\"super\" font-size=\"9\">(0)</tspan>"
             " + &#946; N<tspan baseline-shift=\"sub\" font-size=\"9\">&#952;,k</tspan>", anchor="middle", size=12)
    svg.text(560, 215, "physics: exact traces,", anchor="middle", size=11.5, cls="muted")
    svg.text(560, 230, "learned interior", anchor="middle", size=11.5, cls="muted")
    # (d) host assembly
    svg.arrow(560, 238, 560, 256)
    svg.add('<rect x="120" y="258" width="560" height="34" rx="4" class="hostbox"/>')
    svg.text(400, 280, "(d) host: &#923;<tspan baseline-shift=\"sub\" font-size=\"9\">i</tspan> = "
             "V<tspan baseline-shift=\"sub\" font-size=\"9\">i</tspan><tspan baseline-shift=\"super\" font-size=\"9\">T</tspan>"
             " A<tspan baseline-shift=\"sub\" font-size=\"9\">i</tspan> V<tspan baseline-shift=\"sub\" font-size=\"9\">i</tspan>"
             " per piece, assembled; one interface solve", anchor="middle", size=13)
    return svg.write("overview")


def main() -> None:
    D, U, V, edges = s_channel()
    fig_overview(D, U, V, edges)
    fig_decomposition(D, U, V, edges)
    _, angles = fig_charts()
    fig_modes()
    fig_rounds()
    fig_gram()
    _, recompute_ms, solve_ms = fig_forward()
    sizes = {n: os.path.getsize(os.path.join(OUT, n + ".svg")) for n in
             ("overview", "decomposition", "charts", "modes", "rounds", "gram", "forward")}
    sys.stdout.write(f"end-piece corner angles (deg): {[round(a, 1) for a in angles]}\n"
                     f"classical port-matrix recompute at n=64: {recompute_ms:.1f} ms per piece; "
                     f"one OSM round per piece: {solve_ms:.2f} ms\n"
                     f"svg sizes (bytes): {sizes}\n")


if __name__ == "__main__":
    main()
