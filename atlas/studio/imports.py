"""CAD files as features (plan §3.6).

* **.msh**: the workbench's own Gmsh reader (`gmsh_import.read_msh_bytes`, unchanged:
  physical groups ``domain``, ``region:<material>``, ``window:<id>``, ``bc:<kind>``).
  The mesh defines the whole case's geometry, so it REPLACES the drawing: a plate the
  size of its domain, a region feature per region, and its windows and boundaries
  into the case.
* **STEP**: Gmsh's OpenCASCADE importer (`occ.import_step`), in metres; a solid is cut
  at half its height.  Each face becomes a shape that adds, and its holes shapes that
  cut, beside what is already drawn (moved to the origin when nothing is).
* **DXF and SVG** are read in the page, into pen paths, and arrive as an ordinary edit.
"""

from __future__ import annotations

from atlas.workbench.spec import Boundary, CaseSpec, Region, Window

from . import geometry_eval as GE
from .document import Feature, Geometry, next_id


def _ring_feature(fid: str, ring: dict, role: str, name: str, material=None) -> Feature:
    return Feature(id=fid, kind="path", role=role, name=name, material=material,
                   points=[tuple(p) for p in ring["points"]], edges=ring["edges"],
                   bulge=ring["bulge"])


def load(fmt: str, data: bytes, name: str, spec: CaseSpec, geom: Geometry):
    """(features to add, engine fields to set).  A ``.msh`` sets ``replace`` in the
    returned fields: its features are the whole drawing."""
    fmt = fmt.lower().lstrip(".")
    dx = float(spec.domain.dx)
    if fmt == "msh":
        from atlas.workbench.gmsh_import import GmshImportError, read_msh_bytes
        try:
            g = read_msh_bytes(data, dx, name)
        except GmshImportError as exc:
            raise ValueError(str(exc)) from None
        base = GE.default_material(spec)
        feats = [Feature(id="f1", kind="rect", role="add", name="Domain", material=base,
                         x=0.0, y=0.0, w=g.nx * dx, h=g.ny * dx)]
        for r in g.regions:
            reg = Region(**r)
            fid = f"f{len(feats) + 1}"
            if reg.shape == "rect":
                feats.append(Feature(id=fid, kind="rect", role="region", name=reg.id,
                                     material=reg.material, x=reg.x0 * dx, y=reg.y0 * dx,
                                     w=reg.nx * dx, h=reg.ny * dx))
            else:
                pts = [(x * dx, y * dx) for x, y in reg.points]
                feats.append(Feature(id=fid, kind="path", role="region", name=reg.id,
                                     material=reg.material, points=pts))
                if reg.holes:
                    cuts = []
                    for j, hole in enumerate(reg.holes):
                        hid = f"f{len(feats) + 1}"
                        feats.append(Feature(id=hid, kind="path", role="cut",
                                             name=f"{reg.id} hole {j + 1}",
                                             points=[(x * dx, y * dx) for x, y in hole]))
                        cuts.append(hid)
                    feats.append(Feature(id=f"f{len(feats) + 1}", kind="combine",
                                         role="region", name=reg.id, material=reg.material,
                                         op="subtract", inputs=[fid, *cuts]))
        extra: dict = {"replace": True}
        if g.windows:
            extra["windows"] = [Window(**w) for w in g.windows]
            extra["layout"] = None
        if g.boundaries:
            extra["boundaries"] = [Boundary(**b) for b in g.boundaries]
        extra["notes"] = list(g.notes) + [f"ignored: {i}" for i in g.ignored]
        return feats, extra
    if fmt in ("step", "stp"):
        from . import occ
        if not occ.available():
            raise ValueError("Gmsh is not installed here; it reads STEP files")
        faces = occ.import_step(data, name)
        feats: list[Feature] = []
        probe = Geometry(features=list(geom.features))
        for k, face in enumerate(faces):
            fid = next_id(probe)
            f = _ring_feature(fid, face["outer"], "add", f"{name} face {k + 1}",
                              GE.default_material(spec))
            probe.features.append(f)
            feats.append(f)
            for j, hole in enumerate(face["holes"]):
                hid = next_id(probe)
                h = _ring_feature(hid, hole, "cut", f"{name} hole {j + 1}")
                probe.features.append(h)
                feats.append(h)
        if not any(f.role == "add" for f in geom.features if not f.hidden) and feats:
            from .session import translate
            xs = [p[0] for f in feats for p in f.points]
            ys = [p[1] for f in feats for p in f.points]
            for f in feats:
                translate(f, -min(xs), -min(ys))
        return feats, {}
    raise ValueError(f"{name}: .msh and STEP are read on the server; DXF and SVG in the page")


__all__ = ["load"]
