"""The studio's geometry (integrated UI plan, phase P1): case file 0.7's feature tree,
its exact evaluation into the engine's domain, the Check view, the imports, and the
first scenario through the protocol.

The invariant that keeps every example honest: a 0.6 case migrated into a tree and
evaluated back gives the engine's own geometry again, edge for edge, cell for cell,
material for material; so an unedited example runs exactly as it always did.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

pytest.importorskip("shapely")

from atlas.studio import document as D  # noqa: E402
from atlas.studio import geocheck  # noqa: E402
from atlas.studio import geometry_eval as GE  # noqa: E402
from atlas.studio import occ, protocol  # noqa: E402
from atlas.studio.session import Session  # noqa: E402
from atlas.workbench import geometry as wgeo  # noqa: E402
from atlas.workbench.spec import EXAMPLES, CaseSpec, check, example_case  # noqa: E402

GMSH = occ.available()


def F(**kw) -> D.Feature:
    kw.setdefault("material", None)
    return D.Feature(**kw)


def _same_outline(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return (len(a.points) == len(b.points) and a.kinds() == b.kinds()
            and np.abs(np.array(a.points) - np.array(b.points)).max() < 1e-9
            and np.abs(np.array(a.bulges()) - np.array(b.bulges())).max() < 1e-9)


@pytest.mark.parametrize("key", sorted(EXAMPLES))
def test_every_example_migrates_and_evaluates_back_to_its_own_geometry(key):
    s = example_case(key)
    r = GE.evaluate(D.migrate(s), s, keep=(s.domain.nx, s.domain.ny))
    d = r.domain
    assert (d.nx, d.ny, d.dx) == (s.domain.nx, s.domain.ny, s.domain.dx)
    assert _same_outline(d.outline, s.domain.outline)
    assert len(d.holes) == len(s.domain.holes)
    assert all(_same_outline(a, b) for a, b in zip(d.holes, s.domain.holes))
    m = wgeo.domain_mask(s.domain)
    assert np.array_equal(wgeo.domain_mask(d), m)
    on = wgeo.region_owner(r.regions, d.nx, d.ny)
    oo = wgeo.region_owner(s.regions, d.nx, d.ny)
    mat = lambda regs, own: np.array([regs[i].material if i >= 0 else "-" for i in own[m]])  # noqa: E731
    if s.regions:
        assert (mat(r.regions, on) == mat(s.regions, oo)).all()
    assert not [n for n in r.notes if n["severity"] == "error"]


def test_a_plate_minus_a_circle_is_the_closed_form_with_four_arcs():
    s = example_case("wall-2")
    dx = s.domain.dx
    W, H = s.domain.nx * dx, s.domain.ny * dx
    R = H / 5
    g = D.Geometry(features=[F(id="f1", kind="rect", role="add", x=0, y=0, w=W, h=H, material="steel"),
                             F(id="f2", kind="ellipse", role="cut", cx=W / 2, cy=H / 2, rx=R, ry=R)])
    r = GE.evaluate(g, s)
    assert r.domain.outline is None                      # the whole grid, holed
    (hole,) = r.domain.holes
    assert hole.kinds() == ["arc"] * 4
    assert np.allclose(hole.bulges(), math.tan(math.pi / 8))
    exact = W * H - math.pi * R * R
    # Shapely holds the samples, within 1/64 of a cell of the curve: the area is off
    # by at most the perimeter times that
    assert abs(r.shape.area - exact) <= 2 * math.pi * R * dx * GE.TOL_CELLS
    cells = wgeo.domain_mask(r.domain).sum() * dx * dx
    assert abs(cells - exact) / exact < 5e-3              # what the solver sees
    if GMSH:
        assert abs(occ.areas(g)["domain"] - exact) / exact < 1e-9


def test_booleans_keep_the_sides_as_drawn():
    s = example_case("wall-2")
    # an L from two rectangles: six straight edges, not the samples
    g = D.Geometry(features=[F(id="f1", kind="rect", role="add", x=0, y=0, w=0.4, h=0.1),
                             F(id="f2", kind="rect", role="add", x=0, y=0, w=0.1, h=0.3)])
    o = GE.evaluate(g, s).domain.outline
    assert o.kinds() == ["line"] * 6
    # a notch cut through the edge: eight straight edges
    g = D.Geometry(features=[F(id="f1", kind="rect", role="add", x=0, y=0, w=0.4, h=0.1),
                             F(id="f2", kind="rect", role="cut", x=0.15, y=0.05, w=0.1, h=0.2)])
    assert GE.evaluate(g, s).domain.outline.kinds() == ["line"] * 8
    # a rounded rectangle: four lines and four quarter arcs
    g = D.Geometry(features=[F(id="f1", kind="rect", role="add", x=0, y=0, w=0.4, h=0.1, r=0.02)])
    o = GE.evaluate(g, s).domain.outline
    assert sorted(o.kinds()) == ["arc"] * 4 + ["line"] * 4
    # a plate with a bite out of its corner by a circle: the bite is one arc, cut short
    g = D.Geometry(features=[F(id="f1", kind="rect", role="add", x=0, y=0, w=0.4, h=0.1),
                             F(id="f2", kind="ellipse", role="cut", cx=0.4, cy=0.1, rx=0.05, ry=0.05)])
    o = GE.evaluate(g, s).domain.outline
    assert o.kinds().count("arc") == 1 and o.kinds().count("line") == 4
    k = o.kinds().index("arc")
    assert abs(abs(o.bulges()[k]) - math.tan(math.pi / 8)) < 1e-6       # a quarter circle


def test_regions_on_cell_boundaries_are_rectangles_and_curves_stay_curves():
    s = example_case("wall-2")
    dx = s.domain.dx
    g = D.Geometry(features=[F(id="f1", kind="rect", role="add", x=0, y=0, w=40 * dx, h=20 * dx, material="steel"),
                             F(id="f2", kind="rect", role="region", x=10 * dx, y=5 * dx, w=8 * dx, h=4 * dx, material="copper"),
                             F(id="f3", kind="ellipse", role="region", cx=30 * dx, cy=10 * dx, rx=4 * dx, ry=4 * dx, material="copper")])
    r = GE.evaluate(g, s)
    shapes = [(x.shape, x.material) for x in r.regions]
    assert shapes == [("rect", "steel"), ("rect", "copper"), ("curve", "copper")]
    assert (r.regions[1].x0, r.regions[1].y0, r.regions[1].nx, r.regions[1].ny) == (10, 5, 8, 4)
    assert r.regions[2].outline.kinds() == ["arc"] * 4


def test_boundaries_follow_the_sides_they_were_set_on():
    s = Session(spec=example_case("wall-2"), key="x")
    protocol.handle(s, {"id": 1, "type": "geometry.start", "shape": "l-bracket", "params": {"W": 0.4, "H": 0.3}})
    o = s.spec.domain.outline
    # set a distinctive condition on the bracket's long bottom side (side 0)
    i = next(k for k, b in enumerate(s.spec.boundaries) if b.edge == "outline:0")
    protocol.handle(s, {"id": 2, "type": "edit.apply", "ops": [
        {"op": "set", "path": ["boundaries", i, "kind"], "value": "fixed-temperature"},
        {"op": "set", "path": ["boundaries", i, "value"], "value": 412.0}]})
    # a hole drawn in the corner renumbers nothing on the outline, but a notch does
    protocol.handle(s, {"id": 3, "type": "edit.apply", "ops": [{"op": "insert", "path": ["geometry", "features"], "value": {
        "id": "f9", "kind": "rect", "role": "cut", "name": "Notch", "x": 0.25, "y": -0.05, "w": 0.05, "h": 0.08}}]})
    held = [b for b in s.spec.boundaries if b.value == 412.0]
    assert len(held) == 2, [(b.edge, b.kind, b.value) for b in s.spec.boundaries]   # the side cut in two
    sides = GE.edge_sides(s._provenance())
    assert {sides[b.edge] for b in held} == {("f1", 0)}
    assert len(s.spec.domain.outline.points) == len(o.points) + 4


def test_a_07_file_round_trips_and_an_unedited_06_stays_06(tmp_path):
    s = Session(cases_dir=str(tmp_path), spec=example_case("bend-3"), key="x")
    s.save("as06")
    assert json.loads((tmp_path / "as06.json").read_text())["schema_id"] == D.SCHEMA_06
    protocol.handle(s, {"id": 1, "type": "edit.apply", "ops": [{"op": "insert", "path": ["geometry", "features"], "value": {
        "id": "f9", "kind": "ellipse", "role": "cut", "name": "Bore", "cx": 0.3, "cy": 0.3, "rx": 0.03, "ry": 0.03}}]})
    s.save("as07")
    text = (tmp_path / "as07.json").read_text()
    assert json.loads(text)["schema_id"] == D.SCHEMA_07
    s2 = Session(cases_dir=str(tmp_path), spec=example_case("wall-2"), key="y")
    s2.open("as07")
    assert s2.spec.to_json() == s.spec.to_json() and s2.geom == s.geom
    s2.save("as07-again")
    assert (tmp_path / "as07-again.json").read_text() == text
    # the old reader refuses a 0.7 file with its reason, not a crash
    with pytest.raises(Exception):
        CaseSpec.from_json(text)


def test_the_check_finds_a_strip_thinner_than_a_cell_and_the_grid_fix_resolves_it():
    s = Session(spec=example_case("wall-2"), key="x")
    dx = s.spec.domain.dx
    protocol.handle(s, {"id": 1, "type": "geometry.start", "shape": "plate", "params": {"W": 0.4, "H": 0.1}})
    protocol.handle(s, {"id": 2, "type": "edit.apply", "ops": [{"op": "insert", "path": ["geometry", "features"], "value": {
        "id": "f2", "kind": "rect", "role": "region", "name": "Strip", "material": "copper",
        "x": 0.0, "y": 0.05, "w": 0.4, "h": 0.6 * dx}}]})
    c = s.geometry_check()
    thin = [n for n in c["notes"] if n["kind"] == "thin"]
    assert thin and thin[0]["feature"] == "f2" and abs(thin[0]["width_cells"] - 0.6) < 0.05
    keys = [f["key"] for f in thin[0]["fixes"]]
    assert "widen:f2" in keys and "keep" in keys and any(k.startswith("grid:") for k in keys)
    strip = next(a for a in c["areas"] if a["feature"] == "f2")
    assert strip["exact"] == pytest.approx(0.4 * 0.6 * dx, rel=1e-6)
    s.geometry_fix("widen:f2")
    assert s.geom.by_id()["f2"].h == pytest.approx(dx)
    assert not [n for n in s.geometry_check()["notes"] if n["kind"] == "thin"]


@pytest.mark.skipif(not GMSH, reason="Gmsh is not installed")
def test_opencascade_areas_match_closed_forms():
    g = D.Geometry(features=[
        F(id="f1", kind="rect", role="add", x=0, y=0, w=1.0, h=0.5, r=0.1),
        F(id="f2", kind="ellipse", role="cut", cx=0.5, cy=0.25, rx=0.2, ry=0.1),
        F(id="f3", kind="path", role="region", points=[(0.05, 0.05), (0.25, 0.05), (0.25, 0.2)],
          edges=["line", "arc", "line"], bulge=[0, -0.3, 0])])
    a = occ.areas(g)
    rounded = 1.0 * 0.5 - (4 - math.pi) * 0.1 ** 2
    assert a["domain"] == pytest.approx(rounded - math.pi * 0.2 * 0.1, rel=1e-9)
    ev = GE.Evaluator(g, 1e-6)
    # sampled within 1e-6 of the arc: the area is off by at most that times its length
    assert ev.shapes["f3"].area == pytest.approx(a["features"]["f3"], abs=1e-6 * 0.3)


@pytest.mark.skipif(not GMSH, reason="Gmsh is not installed")
def test_a_step_file_imports_as_a_plate_with_its_hole(tmp_path):
    import gmsh
    from atlas.studio.occ import _Model
    path = str(tmp_path / "plate.step")
    with _Model() as gm:
        o = gm.model.occ
        plate = o.addRectangle(0, 0, 0, 100, 50)              # millimetres, as CAD writes
        hole = o.addDisk(60, 25, 0, 10, 10)
        o.cut([(2, plate)], [(2, hole)])
        o.synchronize()
        gmsh.write(path)
    faces = occ.import_step(open(path, "rb").read(), "plate.step")
    assert len(faces) == 1
    outer, holes = faces[0]["outer"], faces[0]["holes"]
    assert outer["edges"] == ["line"] * 4
    assert len(holes) == 1 and set(holes[0]["edges"]) == {"arc"}
    xs = [p[0] for p in outer["points"]]
    assert max(xs) == pytest.approx(0.1, rel=1e-9)            # metres
    s = Session(spec=example_case("wall-2"), key="x")
    protocol.handle(s, {"id": 1, "type": "geometry.start", "shape": "blank"})
    import base64
    r, _ = protocol.handle(s, {"id": 2, "type": "geometry.import", "format": "step",
                               "data": base64.b64encode(open(path, "rb").read()).decode(),
                               "name": "plate.step"})
    assert r["ok"], r
    assert [f.role for f in s.geom.features] == ["add", "cut"]
    assert len(s.spec.domain.holes) == 1


@pytest.mark.skipif(not GMSH, reason="Gmsh is not installed")
def test_a_gmsh_mesh_replaces_the_drawing_with_its_regions(tmp_path):
    import gmsh
    from atlas.studio.occ import _Model
    path = str(tmp_path / "two.msh")
    with _Model() as gm:
        o = gm.model.occ
        a = o.addRectangle(0, 0, 0, 0.2, 0.1)
        b = o.addRectangle(0.2, 0, 0, 0.2, 0.1)
        o.fragment([(2, a)], [(2, b)])
        o.synchronize()
        gm.model.addPhysicalGroup(2, [a], name="region:steel")
        gm.model.addPhysicalGroup(2, [b], name="region:copper")
        gm.option.setNumber("Mesh.MeshSizeMax", 0.01)
        gm.model.mesh.generate(2)
        gmsh.write(path)
    s = Session(spec=example_case("wall-2"), key="x")
    import base64
    r, _ = protocol.handle(s, {"id": 1, "type": "geometry.import", "format": "msh",
                               "data": base64.b64encode(open(path, "rb").read()).decode(),
                               "name": "two.msh"})
    assert r["ok"], r
    roles = [(f.role, f.material) for f in s.geom.features]
    assert roles[0][0] == "add" and ("region", "steel") in roles and ("region", "copper") in roles
    assert {x.material for x in s.spec.regions} >= {"steel", "copper"}


def test_every_start_shape_is_ready_to_run_for_heat():
    s = Session(spec=example_case("wall-2"), key="x")
    s.start_new("conduction-2d")
    for shape in ("plate", "plate-hole", "l-bracket", "bend", "s-channel", "annulus", "high-contrast"):
        r, _ = protocol.handle(s, {"id": 1, "type": "geometry.start", "shape": shape,
                                   "params": {"W": 0.4, "H": 0.2}})
        assert r["ok"], (shape, r)
        errors = [i.message for i in check(s.spec) if i.severity == "error"]
        assert errors == [] and not s.geometry_errors(), (shape, errors)
    # the high-contrast layout is the same for the same seed, byte for byte
    a = s.geom.model_dump_json()
    protocol.handle(s, {"id": 2, "type": "geometry.start", "shape": "high-contrast", "params": {"W": 0.4, "H": 0.2}})
    assert s.geom.model_dump_json() == a


def test_a_start_shape_smaller_than_four_cells_is_refused_with_its_reason():
    s = Session(spec=example_case("wall-2"), key="x")
    r, pushes = protocol.handle(s, {"id": 1, "type": "geometry.start", "shape": "plate",
                                    "params": {"W": 0.004, "H": 0.004}})
    assert r["ok"] is False and "4 cells" in r["error"]


def test_the_first_scenario_through_the_protocol(tmp_path):
    """Choose a kind, draw a plate and a hole (as the page sends them), run: the run
    finishes with every measured check passing (P1's done-when, without a browser;
    the page check drives the same messages with real drags)."""
    s = Session(cases_dir=str(tmp_path), spec=example_case("wall-2"), key="x")
    protocol.handle(s, {"id": 1, "type": "case.new", "family": "conduction-2d"})
    protocol.handle(s, {"id": 2, "type": "geometry.start", "shape": "blank"})
    assert s.status()["text"] == "No shape yet"
    r, _ = protocol.handle(s, {"id": 3, "type": "edit.apply", "label": "drew Rectangle 1", "ops": [
        {"op": "insert", "path": ["geometry", "features"], "value": {
            "id": "f1", "kind": "rect", "role": "add", "name": "Rectangle 1", "material": "steel",
            "x": 0.0, "y": 0.0, "w": 0.3, "h": 0.15}}]})
    assert r["ok"]
    r, _ = protocol.handle(s, {"id": 4, "type": "edit.apply", "label": "drew Circle 1", "ops": [
        {"op": "insert", "path": ["geometry", "features"], "value": {
            "id": "f2", "kind": "ellipse", "role": "cut", "name": "Circle 1",
            "cx": 0.15, "cy": 0.075, "rx": 0.03, "ry": 0.03}}]})
    assert r["ok"] and len(s.spec.domain.holes) == 1
    assert s.status()["kind"] in ("ok", "warn"), s.status()
    s.spec.run.steps = 1
    run = s.start_run(["serial", "full"], blocking=True)
    assert run.status == "done", run.progress().error
    assert not [c for c in run.results["checks"] if c["passed"] is False]


SHOWCASE = [("conduction-2d", "wall-2"), ("electric-2d", "plate-circuit"),
            ("transport-2d", "plume-2"), ("acoustics-2d", "sound-air-water"),
            ("elasticity-2d", "plate-hole"), ("thermoelastic-2d", "heated-strip"),
            ("conjugate-heat-2d", "cooled-block"), ("incompressible-2d", "wake-array-3")]


@pytest.mark.parametrize("fid,key", SHOWCASE)
def test_every_kind_of_showcase_example_opens_unchanged_and_runs_with_its_checks(fid, key):
    """P1's done-when: every family's showcase example still opens and runs in the
    studio, its checks unchanged: the session holds the engine's case byte for byte, and
    a short run's checks are the engine's own run's, check for check."""
    from atlas.workbench.runner import CaseRun
    s = Session(spec=example_case("wall-2"), key="x")
    s.example(key)
    assert s.spec.to_json() == example_case(key).to_json()
    assert D.is_migration(s.spec, s.geom)
    s.spec.run.steps = 1
    a = s.start_run(["serial", "full"], blocking=True)
    direct = example_case(key)
    direct.run.steps = 1
    b = CaseRun(direct, arms=("serial", "full"), steps=1, threads=direct.run.threads)
    b.run_blocking()
    assert a.status == b.status == "done"
    pick = lambda r: [(c["key"], c["passed"]) for c in r.results["checks"]]  # noqa: E731
    assert pick(a) == pick(b)
