"""Geometry from Gmsh, by naming physical groups.

The hybrid's second half (the owner's choice, 2026-09-28): anything that is not a
rectangle on the grid is drawn in Gmsh's own GUI and imported here.  Gmsh is
already installed (4.15, GPL-2+ with a linking exception that does not cover this
code, so the workbench only *calls* it and nothing of Gmsh is copied or shipped).

**The naming convention**, on physical groups:

========================  ====  ===============================================
name                      dim   becomes
========================  ====  ===============================================
``domain``                2     the domain's extent (else: the bounding box of
                                every meshed surface)
``region:<material>``     2     one region per connected piece, with its holes
``window:<id>``           2     a window; must be an axis-aligned rectangle
                                on cell boundaries
``bc:<kind>``             1     boundary segments on the domain's edges
``bc:<kind>=<value>``     1     the same, with a value (a temperature, ...)
========================  ====  ===============================================

Anything else is listed as ignored.  Coordinates are in the physics' length unit
and are divided by the cell size; the domain's lower-left corner becomes (0, 0).

**Only meshed ``.msh`` files are read**, not ``.geo`` scripts: a ``.geo`` file is a
program (its language has ``SystemCall``), and a case file should not be able to
run one.  In Gmsh: build the geometry, name the physical groups, *Mesh > 2D*,
*File > Save Mesh*.  Shapes are read from the mesh's own edges, so a curve is as
smooth as its mesh; a mesh size at or below the cell size is enough.
"""

from __future__ import annotations

import os
import tempfile
import threading
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from . import geometry as geo

#: Gmsh keeps one global model per process; one import at a time.
_LOCK = threading.Lock()


class GmshImportError(ValueError):
    """The file cannot become a case; the message says why, for a person."""


@dataclass
class GmshGeometry:
    nx: int
    ny: int
    regions: list[dict] = field(default_factory=list)
    windows: list[dict] = field(default_factory=list)
    boundaries: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# mesh -> loops
# ---------------------------------------------------------------------------


def _loops_from_faces(faces: list[list[int]]) -> list[list[int]]:
    """The boundary loops of a set of polygonal faces, as node-tag cycles.

    An edge used by exactly one face is on the boundary.  Directions come from
    the faces, so consistently oriented faces give consistently oriented loops.
    """
    count: dict[tuple[int, int], int] = defaultdict(int)
    directed: list[tuple[int, int]] = []
    for f in faces:
        for a, b in zip(f, f[1:] + f[:1]):
            count[(min(a, b), max(a, b))] += 1
            directed.append((a, b))
    nxt: dict[int, list[int]] = defaultdict(list)
    for a, b in directed:
        if count[(min(a, b), max(a, b))] == 1:
            nxt[a].append(b)
    if any(len(v) > 1 for v in nxt.values()):
        raise GmshImportError("a surface touches itself at a single point; split it into "
                              "pieces that meet along edges")
    loops, seen = [], set()
    for start in list(nxt):
        if start in seen:
            continue
        loop, cur = [], start
        while cur not in seen:
            seen.add(cur)
            loop.append(cur)
            if not nxt.get(cur):
                raise GmshImportError("a surface's boundary does not close")
            cur = nxt[cur][0]
        if cur != start:
            raise GmshImportError("a surface's boundary does not close")
        loops.append(loop)
    return loops


def _simplify(pts: np.ndarray, tol: float) -> np.ndarray:
    """Drop points that lie on the straight line through their neighbours."""
    keep = []
    n = len(pts)
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(cross) > tol * max(np.hypot(*(c - a)), tol):
            keep.append(i)
    return pts[keep] if len(keep) >= 3 else pts


def _inside(pt, ring: np.ndarray) -> bool:
    x, y = pt
    xa, ya = ring[:, 0], ring[:, 1]
    xb, yb = np.roll(xa, -1), np.roll(ya, -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        crosses = (ya > y) != (yb > y)
        xc = xa + (y - ya) * (xb - xa) / (yb - ya)
    return bool(np.sum(crosses & (x < xc)) % 2)


def _pieces(loops: list[np.ndarray]) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """Group loops into (outer ring, holes): a loop inside exactly an odd number
    of others is a hole of the smallest loop that contains it."""
    order = sorted(range(len(loops)), key=lambda i: -geo.polygon_area(loops[i]))
    depth = {}
    parent = {}
    for i in order:
        containers = [j for j in order if j != i
                      and geo.polygon_area(loops[j]) > geo.polygon_area(loops[i])
                      and _inside(loops[i][0], loops[j])]
        depth[i] = len(containers)
        parent[i] = min(containers, key=lambda j: geo.polygon_area(loops[j])) if containers else None
    out = []
    for i in order:
        if depth[i] % 2 == 0:
            holes = [loops[j] for j in order if parent[j] == i and depth[j] % 2 == 1]
            out.append((loops[i], holes))
    return out


# ---------------------------------------------------------------------------
# the import
# ---------------------------------------------------------------------------


def read_msh(path: str, dx: float) -> GmshGeometry:
    """Read a meshed Gmsh file into case geometry, in cells."""
    if not path.lower().endswith(".msh"):
        raise GmshImportError("only meshed .msh files are read (a .geo file is a script); "
                              "in Gmsh use Mesh > 2D, then File > Save Mesh")
    if not dx > 0:
        raise GmshImportError("the cell size must be positive")
    try:
        import gmsh                                       # GPL-2+; called, not copied
    except ImportError as exc:                            # pragma: no cover
        raise GmshImportError("Gmsh's Python module is not installed") from exc
    except OSError as exc:
        # Installed but its library will not load: on a Linux machine without the
        # system's OpenGL, the wheel's import fails with "libGLU.so.1: cannot open
        # shared object file" (found by the one-command install on Ubuntu, demo
        # step 7). The page says so, instead of a click that does nothing.
        raise GmshImportError(f"Gmsh is installed but its library will not load here "
                              f"({exc}); on Linux it needs the system's OpenGL "
                              f"libraries") from exc
    with _LOCK:
        gmsh.initialize(readConfigFiles=False, interruptible=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            try:
                gmsh.open(path)
            except Exception as exc:
                raise GmshImportError(f"Gmsh could not read the file: {exc}") from exc
            return _read_model(gmsh, dx)
        finally:
            gmsh.finalize()


def read_msh_bytes(data: bytes, dx: float, name: str = "upload.msh") -> GmshGeometry:
    """The same, for an uploaded file's bytes."""
    if not name.lower().endswith(".msh"):
        raise GmshImportError("only meshed .msh files are read (a .geo file is a script); "
                              "in Gmsh use Mesh > 2D, then File > Save Mesh")
    fd, tmp = tempfile.mkstemp(suffix=".msh")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return read_msh(tmp, dx)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _read_model(gmsh, dx: float) -> GmshGeometry:
    tags, coords, _ = gmsh.model.mesh.getNodes()
    if len(tags) == 0:
        raise GmshImportError("the file has no mesh; in Gmsh use Mesh > 2D before saving")
    xyz = dict(zip(np.asarray(tags, dtype=np.int64).tolist(),
                   np.asarray(coords, dtype=float).reshape(-1, 3)[:, :2]))

    def faces_of(dim: int, entity: int) -> list[list[int]]:
        out = []
        types, _etags, ntags = gmsh.model.mesh.getElements(dim, entity)
        for t, nt in zip(types, ntags):
            _name, edim, _order, nnodes, _loc, nprim = gmsh.model.mesh.getElementProperties(t)
            if edim != dim:
                continue
            arr = np.asarray(nt, dtype=np.int64).reshape(-1, nnodes)[:, :nprim]
            out += arr.tolist()
        return out

    groups = []
    for dim, ptag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, ptag).strip()
        groups.append((dim, ptag, name, list(gmsh.model.getEntitiesForPhysicalGroup(dim, ptag))))
    groups.sort(key=lambda g: (g[0], g[1]))

    # the extent: the "domain" group, else every meshed surface
    dom = [g for g in groups if g[0] == 2 and g[2] == "domain"]
    ents = dom[0][3] if dom else [t for _d, t in gmsh.model.getEntities(2)]
    pts = [xyz[n] for e in ents for f in faces_of(2, e) for n in f]
    if not pts:
        raise GmshImportError("no meshed surface found; in Gmsh use Mesh > 2D before saving")
    pts = np.asarray(pts)
    (xmin, ymin), (xmax, ymax) = pts.min(axis=0), pts.max(axis=0)
    origin = np.array([xmin, ymin])
    fx, fy = (xmax - xmin) / dx, (ymax - ymin) / dx
    nx, ny = int(round(fx)), int(round(fy))
    if nx < 1 or ny < 1:
        raise GmshImportError(f"the domain is {fx:.3g} x {fy:.3g} cells at this cell size")
    res = GmshGeometry(nx, ny)
    if max(abs(fx - nx), abs(fy - ny)) > 1e-6 * max(nx, ny, 1):
        res.notes.append(f"the domain is {fx:.4f} x {fy:.4f} cells, rounded to {nx} x {ny}")
    if abs(xmin) > 1e-12 or abs(ymin) > 1e-12:
        res.notes.append(f"moved so the domain's lower-left corner ({xmin:g}, {ymin:g}) is "
                         f"the origin")
    tol = 1e-6 * max(nx, ny)

    def cells(p):
        return (np.asarray(p, dtype=float) - origin) / dx

    n_region = defaultdict(int)
    for dim, _ptag, name, ents in groups:
        kind, _, rest = name.partition(":")
        if name == "domain":
            continue
        if dim == 2 and kind in ("region", "window") and rest:
            faces = [f for e in ents for f in faces_of(2, e)]
            if not faces:
                res.ignored.append(f"{name}: no mesh on its surfaces")
                continue
            loops = [np.array([cells(xyz[n]) for n in lp]) for lp in _loops_from_faces(faces)]
            loops = [_simplify(lp, 1e-9) for lp in loops]
            for outer, holes in _pieces(loops):
                if kind == "window":
                    res.windows.append(_window(rest, outer, holes, tol, res, nx, ny))
                    continue
                n_region[rest] += 1
                rid = rest if n_region[rest] == 1 else f"{rest}-{n_region[rest]}"
                res.regions.append(_region(rid, rest, outer, holes, tol))
        elif dim == 1 and kind == "bc" and rest:
            bkind, _, val = rest.partition("=")
            try:
                value = float(val) if val else None
            except ValueError:
                raise GmshImportError(f"{name}: the value after '=' is not a number")
            for e in ents:
                seg = [cells(xyz[n]) for f in faces_of(1, e) for n in f]
                if not seg:
                    res.ignored.append(f"{name}: curve {e} has no mesh")
                    continue
                res.boundaries.append(_boundary(bkind.strip(), value, np.asarray(seg),
                                                nx, ny, tol, res))
        else:
            res.ignored.append(f"{name or '(unnamed)'} (dimension {dim})")
    res.windows = [w for w in res.windows if w is not None]
    _name_boundaries(res.boundaries)
    return res


def _window(wid: str, outer: np.ndarray, holes, tol: float, res: GmshGeometry,
            nx: int, ny: int):
    xs, ys = outer[:, 0], outer[:, 1]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    on_box = np.all((np.isclose(xs, x0, atol=tol) | np.isclose(xs, x1, atol=tol))
                    & (np.isclose(ys, y0, atol=tol) | np.isclose(ys, y1, atol=tol)))
    if holes or len(outer) != 4 or not on_box:
        raise GmshImportError(f"window:{wid} is not an axis-aligned rectangle; windows "
                              f"are rectangles on the grid")
    r = [round(v) for v in (x0, y0, x1, y1)]
    if max(abs(a - b) for a, b in zip(r, (x0, y0, x1, y1))) > 1e-3:
        raise GmshImportError(f"window:{wid} does not lie on cell boundaries "
                              f"(x {x0:.3f}..{x1:.3f}, y {y0:.3f}..{y1:.3f} cells)")
    return dict(id=wid, x0=int(r[0]), y0=int(r[1]), nx=int(r[2] - r[0]), ny=int(r[3] - r[1]))


def _region(rid: str, material: str, outer: np.ndarray, holes, tol: float) -> dict:
    xs, ys = outer[:, 0], outer[:, 1]
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    r = [round(v) for v in (x0, y0, x1, y1)]
    is_rect = (not holes and len(outer) == 4
               and np.all((np.isclose(xs, x0, atol=tol) | np.isclose(xs, x1, atol=tol))
                          & (np.isclose(ys, y0, atol=tol) | np.isclose(ys, y1, atol=tol)))
               and max(abs(a - b) for a, b in zip(r, (x0, y0, x1, y1))) <= 1e-3)
    if is_rect:
        return dict(id=rid, material=material, shape="rect", x0=int(r[0]), y0=int(r[1]),
                    nx=int(r[2] - r[0]), ny=int(r[3] - r[1]))
    rnd = lambda a: [(round(float(p[0]), 6), round(float(p[1]), 6)) for p in a]   # noqa: E731
    return dict(id=rid, material=material, shape="polygon",
                points=[(max(x, 0.0), max(y, 0.0)) for x, y in rnd(outer)],
                holes=[rnd(h) for h in holes] or None)


def _boundary(kind: str, value, seg: np.ndarray, nx: int, ny: int, tol: float,
              res: GmshGeometry) -> dict:
    xs, ys = seg[:, 0], seg[:, 1]
    for edge, coord, fixed, along in (("left", xs, 0.0, ys), ("right", xs, float(nx), ys),
                                      ("bottom", ys, 0.0, xs), ("top", ys, float(ny), xs)):
        if np.all(np.abs(coord - fixed) <= max(tol, 1e-3)):
            a, b = float(along.min()), float(along.max())
            s, e = int(round(a)), int(round(b))
            if max(abs(s - a), abs(e - b)) > 1e-3:
                res.notes.append(f"bc:{kind} on the {edge} edge ({a:.3f}..{b:.3f} cells) "
                                 f"rounded to cells {s}..{e}")
            if e <= s:
                raise GmshImportError(f"bc:{kind} on the {edge} edge is shorter than a cell")
            return dict(id="", edge=edge, kind=kind, start=s, stop=e, value=value)
    raise GmshImportError(f"bc:{kind} has a curve that is not on the domain's edge; "
                          f"boundary conditions go on the outer edges")


def _name_boundaries(bs: list[dict]) -> None:
    """Merge touching segments of one kind on one edge, then name them."""
    bs.sort(key=lambda b: (b["edge"], b["start"]))
    merged: list[dict] = []
    for b in bs:
        m = merged[-1] if merged else None
        if (m and m["edge"] == b["edge"] and m["kind"] == b["kind"]
                and m["value"] == b["value"] and b["start"] <= m["stop"]):
            m["stop"] = max(m["stop"], b["stop"])
        else:
            merged.append(dict(b))
    count = defaultdict(int)
    for b in merged:
        count[(b["kind"], b["edge"])] += 1
        k = count[(b["kind"], b["edge"])]
        b["id"] = f"{b['kind']}-{b['edge']}" + (f"-{k}" if k > 1 else "")
    bs[:] = merged


__all__ = ["GmshImportError", "GmshGeometry", "read_msh", "read_msh_bytes"]
