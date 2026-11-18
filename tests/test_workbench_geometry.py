"""The Atlas Workbench geometry section: its rules, its edits and its Gmsh import.

The rules are tested as plain functions (`geometry.py`), against the real tilings
the measurements were taken on.  The edits are tested the way the studio's page sends
them (its protocol, `tests/studio_driver.py`): a gesture ends in one edit of the case
as data, validated as a whole case and refused whole.  The gestures themselves
(select, drag, draw, snap) are the page's own and are driven in the real page
(`scripts/studio_drive.py`), per the project's rule that a socket cannot see an
undrawn page.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from atlas.workbench import geometry as geo                               # noqa: E402
from atlas.workbench import registry                                      # noqa: E402
from atlas.workbench.spec import (EXAMPLES, Boundary, CaseSpec, Region,     # noqa: E402
                                  Window, blank_case, check, example_case)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import studio_driver as D                                                 # noqa: E402


def _msgs(spec, severity=None):
    return [i.message for i in check(spec) if severity is None or i.severity == severity]


# ---------------------------------------------------------------------------
# tilings and the full-weight rule
# ---------------------------------------------------------------------------


#: the measured tilings: the scaling ladder's rungs (a drawn farm, such as the hill, is not one)
FARM_EXAMPLES = [k for k, e in EXAMPLES.items()
                 if e.family == "incompressible-2d" and "cols" in dict(e.params)]


@pytest.mark.parametrize("key", FARM_EXAMPLES)
def test_the_generator_reproduces_the_measured_tilings(key):
    """Same offsets, sizes and names as `scaling_ladder.rung(...).tiling`."""
    from atlas.cases import scaling_ladder as sl
    cols, rows = EXAMPLES[key].param("cols"), EXAMPLES[key].param("rows")
    t = sl.rung(cols, rows).tiling
    got = geo.tile(t.nx, t.ny, cols, rows, 16)
    assert [n for n, _ in got] == list(t.names)
    assert [(b[0], b[1]) for _, b in got] == list(t.offsets)
    assert {(b[2], b[3]) for _, b in got} == {(128, 128)}


@pytest.mark.parametrize("key", FARM_EXAMPLES)
def test_measured_tilings_have_a_full_weight_window_everywhere(key):
    s = example_case(key)
    cols, rows = EXAMPLES[key].param("cols"), EXAMPLES[key].param("rows")
    an = geo.analyse_windows(s.domain.nx, s.domain.ny,
                             [(w.id, (w.x0, w.y0, w.nx, w.ny)) for w in s.windows], 8)
    assert int(an.uncovered.sum()) == 0 and int(an.ramp_only.sum()) == 0
    assert an.thin_pairs == [] and an.nested == []
    assert len(an.cross_points) == (cols - 1) * (rows - 1)
    assert all(len(names) == 4 for names, _b in an.cross_points)


def test_full_weight_cells_are_the_weights_of_the_measured_assembly():
    """`_axis_full` is `ArrayTiling.weights` before normalisation: the cells at
    weight 1 are exactly the ones this rule calls full."""
    from atlas.cases import scaling_ladder as sl
    t = sl.rung(3, 2).tiling
    raw_full = np.zeros((t.ny, t.nx), dtype=bool)
    idx = np.arange(128) + 0.5
    for ox, oy in t.offsets:
        wx, wy = np.ones(128), np.ones(128)
        if ox > 0:
            wx = np.minimum(wx, np.clip(idx / 8, 0, 1))
        if ox + 128 < t.nx:
            wx = np.minimum(wx, np.clip((128 - idx) / 8, 0, 1))
        if oy > 0:
            wy = np.minimum(wy, np.clip(idx / 8, 0, 1))
        if oy + 128 < t.ny:
            wy = np.minimum(wy, np.clip((128 - idx) / 8, 0, 1))
        raw_full[oy:oy + 128, ox:ox + 128] |= (np.minimum(wy[:, None], wx[None, :]) ** 2) == 1
    an = geo.analyse_windows(t.nx, t.ny, list(zip(t.names, [(ox, oy, 128, 128)
                                                          for ox, oy in t.offsets])), 8)
    assert np.array_equal(an.full, raw_full)


@pytest.mark.parametrize("overlap, ramp_only", [(0, 2 * 8 * 240), (8, 8 * 240), (16, 0),
                                                (24, 0)])
def test_two_windows_need_twice_the_ramp(overlap, ramp_only):
    s = example_case("wake-array-3")
    half = (352 + overlap) // 2
    s.windows = [Window(id="A", x0=0, y0=0, nx=half, ny=240),
                 Window(id="B", x0=352 - half, y0=0, nx=half, ny=240)]
    an = geo.analyse_windows(352, 240, [(w.id, (w.x0, w.y0, w.nx, w.ny)) for w in s.windows], 8)
    assert int(an.ramp_only.sum()) == ramp_only
    errors = _msgs(s, "error")
    assert any("no window at full weight" in m for m in errors) == (ramp_only > 0)
    warns = _msgs(s, "warning")
    if overlap == 0:
        assert any("touch without overlapping" in m for m in warns)
    elif overlap < 16:
        assert any(f"overlap by only {overlap} cells" in m for m in warns)
    else:
        assert not any("A and B" in m for m in warns)


def test_a_window_inside_another_is_named():
    s = example_case("wake-array-3")
    s.windows.append(Window(id="X", x0=20, y0=20, nx=40, ny=40))
    assert any("X lies entirely inside F00" in m for m in _msgs(s, "warning"))


def test_tiling_refuses_what_does_not_fit():
    with pytest.raises(ValueError, match="on top of each other"):
        geo.tile(64, 64, 2, 1, 64)
    with pytest.raises(ValueError, match="at least one"):
        geo.tile(64, 64, 0, 1, 16)
    assert geo.tile(64, 64, 1, 1, 16) == [("F00", (0, 0, 64, 64))]
    # ten 21-cell windows fit.  Each keeps 5 full-weight cells in its middle and
    # the stride is about 5, so those middles tile the domain: the rule is about
    # cells, not window sizes, and this layout passes it.
    t = geo.tile(64, 64, 10, 1, 16)
    assert len(t) == 10 and all(b[2] == 21 for _n, b in t)
    an = geo.analyse_windows(64, 64, t, 8)
    assert int(an.ramp_only.sum()) == 0
    assert an.thin_pairs == []          # F70 and F90 overlap by 11, but F80 covers it
    # four windows overlapping by 12 (under twice the ramp): the ramps meet
    an = geo.analyse_windows(64, 64, geo.tile(64, 64, 4, 1, 12), 8)
    assert int(an.ramp_only.sum()) > 0


# ---------------------------------------------------------------------------
# snapping, corners, masks
# ---------------------------------------------------------------------------


def test_snapping_keeps_size_and_stops_at_the_edge():
    assert geo.snap_box(125, 5, 253, 133, 8, 352, 240) == (128, 8, 128, 128)
    assert geo.snap_box(300, 0, 428, 128, 8, 352, 240) == (224, 0, 128, 128)   # pushed back
    assert geo.snap_box(0, 0, 900, 900, 8, 352, 240) == (0, 0, 352, 240)       # clamped
    assert geo.snap_value(349, 16, 350) == 350          # the edge is a target too


def test_corner_drag_keeps_the_opposite_corner():
    box = (0, 0, 128, 128)
    assert geo.resize_from_corner(box, 2, 148, 98, 8, 352, 240) == (0, 0, 144, 96)
    assert geo.resize_from_corner(box, 0, 10, 20, 8, 352, 240) == (8, 16, 120, 112)
    assert geo.resize_from_corner(box, 2, 0.5, 0.5, 8, 352, 240) is None


def test_mask_to_boxes_is_exact():
    rng = np.random.default_rng(0)
    m = rng.random((40, 60)) > 0.6
    m[5:20, 10:30] = True
    back = np.zeros_like(m)
    for x0, y0, w, h in geo.mask_to_boxes(m):
        assert not back[y0:y0 + h, x0:x0 + w].any()                # no overlap
        back[y0:y0 + h, x0:x0 + w] = True
    assert np.array_equal(back, m)


def test_polygons_holes_and_stacking():
    circle = [(50 + 20 * math.cos(t), 40 + 20 * math.sin(t))
              for t in np.linspace(0, 2 * math.pi, 200, endpoint=False)]
    disk = Region(id="cu", material="copper", shape="polygon", points=circle)
    assert (disk.x0, disk.y0, disk.nx, disk.ny) == (30, 20, 40, 40)
    assert abs(int(geo.region_mask(disk, 100, 80).sum()) - math.pi * 400) < 0.02 * math.pi * 400
    plate = Region(id="st", material="steel", shape="polygon",
                   points=[(0, 0), (100, 0), (100, 80), (0, 80)], holes=[circle])
    both = geo.region_mask(plate, 100, 80) & geo.region_mask(disk, 100, 80)
    assert not both.any()                                           # the hole is a hole
    # stacking: a rectangle drawn on top of a plate takes its cells
    s = blank_case(100, 80)
    s.regions = [Region(id="P", material="steel", x0=0, y0=0, nx=100, ny=80),
                 Region(id="I", material="copper", x0=40, y0=30, nx=20, ny=20)]
    owner = geo.region_owner(s.regions, 100, 80)
    assert int((owner == 1).sum()) == 400 and int((owner == 0).sum()) == 8000 - 400
    assert any("take 400 of its 8000 cells" in m for m in _msgs(s, "info"))


def test_region_shapes_are_validated():
    with pytest.raises(ValueError):
        Region(id="p", shape="polygon", points=[(0, 0), (1, 1)])
    with pytest.raises(ValueError):
        Region(id="r", shape="rect", points=[(0, 0), (1, 0), (1, 1)])


# ---------------------------------------------------------------------------
# boundaries and the schema
# ---------------------------------------------------------------------------


def test_the_wind_farm_family_boundary_is_fixed():
    s = example_case("wake-array-3")
    assert [(b.edge, b.kind) for b in s.boundaries] == [
        ("left", "inlet"), ("right", "outlet"), ("bottom", "freestream"), ("top", "freestream")]
    assert _msgs(s, "error") == []
    # a wall is a kind the family imposes -- on a drawn edge, by penalization (case
    # file 0.4) -- but not on the grid's own edges, which its solver fixes
    s.boundaries[0].kind = "wall"
    errors = _msgs(s, "error")
    assert not any("cannot impose 'wall'" in m for m in errors)
    assert any("fixes its outer boundary" in m for m in errors)
    s.boundaries[0].kind = "clamped"
    assert any("cannot impose 'clamped'" in m for m in _msgs(s, "error"))


def test_boundary_rules_on_a_family_that_takes_them():
    s = blank_case(100, 80)
    s.physics.family = "conduction-2d"
    s.boundaries = [Boundary(id="hot", edge="left", kind="fixed-temperature", value=350),
                    Boundary(id="cold", edge="right", kind="fixed-temperature"),
                    Boundary(id="a", edge="bottom", kind="insulated", start=0, stop=60),
                    Boundary(id="b", edge="bottom", kind="insulated", start=50),
                    Boundary(id="c", edge="top", kind="insulated", start=0, stop=90)]
    errors = _msgs(s, "error")
    assert any("cold: fixed-temperature needs a value" in m for m in errors)
    assert any("a and b overlap on the bottom edge" in m for m in errors)
    assert any("10 of the 100 cells on the top edge" in m for m in errors)
    assert any("8,000 cells of the domain have no material yet" in m for m in errors)
    assert not any("cannot impose" in m for m in errors)


def test_version_0_1_files_load_and_unknown_ones_do_not():
    """A 0.1 file as the shell wrote it (physics with nu and u_inf, no regions,
    no boundaries) loads as the same case, through both migrations."""
    ref = example_case("farm-12")
    d = json.loads(ref.to_json())
    d["schema_id"] = "atlas-workbench/case@0.1"
    for key in ("regions", "boundaries", "materials", "attachments"):
        d.pop(key)
    d["physics"] = {"family": "incompressible-2d", **d["physics"]["params"]}
    d["coupling"].pop("style")
    s = CaseSpec.model_validate(d)
    assert s.schema_id == "atlas-workbench/case@0.6"
    assert s == ref
    d["schema_id"] = "atlas-workbench/case@9.9"
    with pytest.raises(ValueError):
        CaseSpec.model_validate(d)


def test_registry_families_name_valid_boundary_kinds():
    known = {k.id for k in registry.BOUNDARY_KINDS}
    for f in registry.FAMILIES:
        assert set(f.boundary_kinds) <= known, f.id
        assert {b.kind for b in f.fixed_boundaries} <= set(f.boundary_kinds), f.id
        assert set(f.layers) <= {"windows", "regions", "devices", "boundaries",
                                 "attachments"}, f.id


# ---------------------------------------------------------------------------
# Gmsh
# ---------------------------------------------------------------------------


def _gmsh():
    """Gmsh, or a skip that says why.  `importorskip` skips only an ImportError, and
    on a Linux machine without the system's OpenGL the wheel is installed and its
    import raises OSError (libGLU.so.1): four tests failed there instead of skipping
    (the one-command install on Ubuntu, demo step 7)."""
    try:
        import gmsh
    except (ImportError, OSError) as exc:
        pytest.skip(f"Gmsh cannot be loaded here: {exc}")
    return gmsh


def test_an_installed_gmsh_that_will_not_load_is_said_so(monkeypatch):
    """The import dialog's error, not a raw OSError, when Gmsh's library will not load."""
    import importlib.abc

    from atlas.workbench.gmsh_import import GmshImportError, read_msh

    class Unloadable(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "gmsh":
                raise OSError("libGLU.so.1: cannot open shared object file")
            return None

    monkeypatch.delitem(sys.modules, "gmsh", raising=False)
    monkeypatch.setattr(sys, "meta_path", [Unloadable(), *sys.meta_path])
    with pytest.raises(GmshImportError, match="will not load here.*libGLU"):
        read_msh("plate.msh", 1 / 32)


def _plate_msh(path, *, window_ok=True, bc_on_edge=True, version=4.1):
    gmsh = _gmsh()
    gmsh.initialize(readConfigFiles=False, interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        plate = occ.addRectangle(0, 0, 0, 4, 3)
        disk = occ.addDisk(2, 1.5, 0, 0.5, 0.5)
        frag, _ = occ.fragment([(2, plate)], [(2, disk)])
        w1 = occ.addRectangle(0, 0, 0, 2.25, 3)
        w2 = (occ.addRectangle(1.75, 0, 0, 2.25, 3) if window_ok
              else occ.addDisk(3, 1.5, 0, 0.5, 0.5))
        inner = occ.addLine(occ.addPoint(1, 1, 0), occ.addPoint(1, 2, 0))
        occ.synchronize()
        surfs = [t for _d, t in frag]
        cu = [t for t in surfs if abs(occ.getMass(2, t) - math.pi / 4) < 1e-6]
        gmsh.model.addPhysicalGroup(2, [t for t in surfs if t not in cu], name="region:steel")
        gmsh.model.addPhysicalGroup(2, cu, name="region:copper")
        gmsh.model.addPhysicalGroup(2, [w1], name="window:A")
        gmsh.model.addPhysicalGroup(2, [w2], name="window:B")
        e = 1e-6
        left = [t for _d, t in gmsh.model.getEntitiesInBoundingBox(-e, -e, -e, e, 3 + e, e, 1)]
        bottom = [t for _d, t in gmsh.model.getEntitiesInBoundingBox(-e, -e, -e, 4 + e, e, e, 1)]
        gmsh.model.addPhysicalGroup(1, left[:1], name="bc:fixed-temperature=300")
        gmsh.model.addPhysicalGroup(1, bottom[:1] if bc_on_edge else [inner],
                                    name="bc:insulated")
        gmsh.model.addPhysicalGroup(0, [1], name="a-point")
        gmsh.option.setNumber("Mesh.MeshSizeMax", 1 / 16)
        gmsh.option.setNumber("Mesh.MshFileVersion", version)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return str(path)


@pytest.mark.parametrize("version", [4.1, 2.2])
def test_gmsh_import_reads_regions_holes_windows_and_boundaries(tmp_path, version):
    from atlas.workbench.gmsh_import import read_msh
    g = read_msh(_plate_msh(tmp_path / "plate.msh", version=version), 1 / 32)
    assert (g.nx, g.ny) == (128, 96)
    assert g.windows == [dict(id="A", x0=0, y0=0, nx=72, ny=96),
                         dict(id="B", x0=56, y0=0, nx=72, ny=96)]
    regions = [Region(**r) for r in g.regions]
    assert [(r.id, r.shape, len(r.holes or [])) for r in regions] == [
        ("steel", "polygon", 1), ("copper", "polygon", 0)]
    owner = geo.region_owner(regions, 128, 96)
    assert int((owner < 0).sum()) == 0                                # a partition
    assert abs(int((owner == 1).sum()) - math.pi * 256) < 0.03 * math.pi * 256
    assert sorted(g.boundaries, key=lambda b: b["edge"]) == [
        dict(id="insulated-bottom", edge="bottom", kind="insulated", start=0, stop=128,
             value=None),
        dict(id="fixed-temperature-left", edge="left", kind="fixed-temperature", start=0,
             stop=96, value=300.0)]
    assert any("a-point" in i for i in g.ignored)


def test_gmsh_import_refuses_what_it_cannot_represent(tmp_path):
    from atlas.workbench.gmsh_import import GmshImportError, read_msh, read_msh_bytes
    with pytest.raises(GmshImportError, match="not an axis-aligned rectangle"):
        read_msh(_plate_msh(tmp_path / "w.msh", window_ok=False), 1 / 32)
    with pytest.raises(GmshImportError, match="not on the domain's edge"):
        read_msh(_plate_msh(tmp_path / "b.msh", bc_on_edge=False), 1 / 32)
    with pytest.raises(GmshImportError, match="a .geo file is a script"):
        read_msh_bytes(b"SystemCall 'calc';", 1 / 32, "evil.geo")


# ---------------------------------------------------------------------------
# the editor: what a gesture sends, through the studio's protocol
# ---------------------------------------------------------------------------


@pytest.fixture
def s(tmp_path):
    """A page's session on the wind farm: a gesture in the page arrives as one edit
    (the gestures themselves, snapping among them, are the page's own and are checked
    in the real page; `geo.snap_box` above is the rule the retired editor snapped by)."""
    return D.session(tmp_path)


def test_windows_are_moved_drawn_refused_and_deleted_as_edits(s):
    r, _ = D.edit(s, D.set_op(["windows", 1, "x0"], 128), D.set_op(["windows", 1, "y0"], 8),
                  label="F10: moved to (128, 8)")
    assert r["ok"] and (s.spec.windows[1].x0, s.spec.windows[1].y0) == (128, 8)
    assert s.log_lines[0].endswith("F10: moved to (128, 8)")
    assert any("no window at full weight" in m for m in _msgs(s.spec, "error"))
    new = {"id": "F7", "x0": 72, "y0": 40, "nx": 48, "ny": 32}
    r, _ = D.edit(s, {"op": "insert", "path": ["windows"], "value": new}, label="drew F7")
    assert r["ok"] and s.spec.windows[-1] == Window(id="F7", x0=72, y0=40, nx=48, ny=32)
    n = len(s.spec.windows)
    flat = {"id": "F8", "x0": 10, "y0": 10, "nx": 0, "ny": 0}          # a click, not a box
    r, pushes = D.edit(s, {"op": "insert", "path": ["windows"], "value": flat}, label="F8")
    assert r["ok"] is False and len(s.spec.windows) == n
    assert any("not applied" in t for t in D.notices(pushes, "error"))
    r, _ = D.edit(s, {"op": "remove", "path": ["windows", 6]}, label="deleted F7")
    assert r["ok"] and [w.id for w in s.spec.windows][-1] == "F21"
    for _ in range(3):
        D.ok(s, "edit.undo")
    assert s.spec == example_case("wake-array-3")


def test_a_window_is_resized_and_placed_by_its_numbers(s):
    r, _ = D.edit(s, D.set_op(["windows", 0, "nx"], 144), D.set_op(["windows", 0, "ny"], 96),
                  label="resized F00")
    assert r["ok"] and s.spec.windows[0] == Window(id="F00", x0=0, y0=0, nx=144, ny=96)
    assert s.log_lines[0].endswith("resized F00")
    r, _ = D.edit(s, D.set_op(["windows", 0, "x0"], 16), label="F00: x = 16")
    assert r["ok"] and s.spec.windows[0].x0 == 16


def test_rotors_are_placed_moved_and_deleted(s):
    rotor = s.spec.devices[0].model_dump(mode="json") | {"id": "R4", "x": 6.25, "y": 3.25}
    r, _ = D.edit(s, {"op": "insert", "path": ["devices"], "value": rotor}, label="R4")
    assert r["ok"] and s.spec.devices[-1].id == "R4"
    assert (s.spec.devices[-1].x, s.spec.devices[-1].y) == (6.25, 3.25)      # in D
    r, _ = D.edit(s, D.set_op(["devices", 3, "x"], 6.75), label="R4 moved")
    assert r["ok"] and s.spec.devices[3].x == 6.75
    r, _ = D.edit(s, {"op": "remove", "path": ["devices", 3]}, label="deleted R4")
    assert r["ok"] and [v.id for v in s.spec.devices] == ["R1", "R2", "R3"]


def test_regions_are_drawn_and_edited_in_the_inspector(s):
    D.ok(s, "case.example", key="wall-2")
    dx = s.spec.domain.dx
    r, _ = D.add_feature(s, D.rect("f9", 64, 8, 32, 16, dx, material="copper", name="Patch"))
    assert r["ok"], r
    k = len(s.spec.regions) - 1
    reg = s.spec.regions[k]
    assert (reg.material, reg.shape, reg.x0, reg.y0, reg.nx, reg.ny) == ("copper", "rect", 64,
                                                                         8, 32, 16)
    i = len(D.features(s)) - 1
    r, _ = D.edit(s, D.set_op(["geometry", "features", i, "material"], "steel"), label="steel")
    assert r["ok"] and s.spec.regions[k].material == "steel"
    r, _ = D.edit(s, D.set_op(["geometry", "features", i, "x"], 16 * dx), label="x = 16")
    assert r["ok"] and s.spec.regions[k].x0 == 16
    r, _ = D.edit(s, {"op": "remove", "path": ["geometry", "features", i]}, label="deleted")
    assert r["ok"] and len(s.spec.regions) == k


def test_a_refused_size_leaves_the_region_as_it_was(s):
    """A value the schema refuses changes nothing, says so, and the page goes back to
    the case as it is (seen in the retired page: a refused width stayed, reading as
    applied)."""
    D.ok(s, "case.example", key="wall-2")
    before = s.spec.to_json()
    i = next(k for k, f in enumerate(D.features(s)) if f["role"] == "region")
    w = -3 * s.spec.domain.dx
    reply, pushes = D.send(s, "edit.apply", label="width = -3",
                           ops=[D.set_op(["geometry", "features", i, "w"], w)])
    assert reply["ok"] is False and s.spec.to_json() == before
    assert any("not applied" in t and "width and a height above 0" in t
               for t in D.notices(pushes, "error"))
    assert pushes[0]["type"] == "state"                     # the page is told the case again


def test_regions_a_family_ignores_are_shown_and_warned_of(s):
    """The wind farm reads no regions, so the catalog gives it no Materials layer; a
    region that arrives anyway (from a file) is kept and warned of."""
    assert "regions" not in D.family(s, "incompressible-2d")["layers"]
    region = {"id": "R1", "material": "copper", "x0": 32, "y0": 32, "nx": 64, "ny": 64}
    r, _ = D.edit(s, {"op": "insert", "path": ["regions"], "value": region}, label="a region")
    assert r["ok"] and s.spec.regions
    assert any("does not read material regions" in m for m in _msgs(s.spec, "warning"))


def test_an_imported_polygon_region_is_a_shape_the_studio_moves(s):
    """The retired editor could not move a polygon region from a file, and said so.  In
    the studio it is a path in the drawing, moved like any other shape: its cells move
    with it."""
    by_file = json.loads(example_case("wall-2").to_json())
    by_file["regions"].append({"id": "P", "material": "copper", "shape": "polygon",
                               "points": [[10, 10], [60, 10], [35, 30]]})
    D.ok(s, "case.import", text=json.dumps(by_file), name="poly.json")
    feats = D.features(s)
    i = next(k for k, f in enumerate(feats) if f["name"] == "P")
    assert feats[i]["kind"] == "path" and feats[i]["role"] == "region"
    dx = s.spec.domain.dx
    owner = geo.region_owner(s.spec.regions, s.spec.domain.nx, s.spec.domain.ny)
    before = int((owner == len(s.spec.regions) - 1).sum())
    moved = [[x + 8 * dx, y] for x, y in feats[i]["points"]]
    r, _ = D.edit(s, D.set_op(["geometry", "features", i, "points"], moved), label="moved P")
    assert r["ok"], r
    owner = geo.region_owner(s.spec.regions, s.spec.domain.nx, s.spec.domain.ny)
    cells = np.argwhere(owner == len(s.spec.regions) - 1)
    assert s.spec.regions[-1].material == "copper" and cells[:, 1].min() >= 17
    assert abs(len(cells) - before) <= 0.05 * before


def test_windows_laid_as_a_grid_of_rectangles_are_one_edit(s):
    """A grid of rectangles (`geo.tile`, which the retired tiling dialog called) is one
    edit, and Undo brings the windows before it back."""
    nx, ny = s.spec.domain.nx, s.spec.domain.ny
    tiles = [{"id": n, "x0": b[0], "y0": b[1], "nx": b[2], "ny": b[3]}
             for n, b in geo.tile(nx, ny, 4, 3, 16)]
    r, _ = D.edit(s, D.set_op(["windows"], tiles), label="a grid of 4 x 3 windows")
    assert r["ok"] and len(s.spec.windows) == 12 and _msgs(s.spec, "error") == []
    D.ok(s, "edit.undo")
    assert s.spec == example_case("wake-array-3")


def test_a_gmsh_mesh_replaces_the_drawing_with_its_windows_and_regions(s, tmp_path):
    """The Gmsh path (``geometry.import``): the mesh's grid, its windows, its regions and
    its boundary conditions replace the drawing, in one edit that Undo takes back.  In
    style B, whose windows overlap (style C would cut its two pieces again, one per
    material, as a newly drawn shape's starting values do)."""
    import base64
    path = _plate_msh(tmp_path / "p.msh")
    D.ok(s, "case.new", family="conduction-2d")
    r, _ = D.edit(s, D.set_op(["domain", "dx"], 1 / 32), D.set_op(["coupling", "style"], "B"),
                  label="cells of 1/32, style B")
    assert r["ok"], r
    before = s.spec.model_dump()
    r, pushes = D.send(s, "geometry.import", format="msh", name="p.msh",
                       data=base64.b64encode(open(path, "rb").read()).decode())
    assert r["ok"], (r, D.notices(pushes))
    sp = s.spec
    assert (sp.domain.nx, sp.domain.ny) == (128, 96)
    assert [w.id for w in sp.windows] == ["A", "B"] and sp.layout is None
    assert {x.material for x in sp.regions} >= {"steel", "copper"}
    kinds = {(b.edge, b.kind) for b in sp.boundaries}
    assert {("left", "fixed-temperature"), ("bottom", "insulated")} <= kinds
    D.ok(s, "edit.undo")
    assert s.spec.model_dump() == before
