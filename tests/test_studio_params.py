"""The studio's geometry II (integrated UI plan, phase P2): parameters and formulas,
seeded scatter, pattern, mirror, channel, fillet, coefficient fields and sweeps.

The plan's done-when, where a test can hold it: changing W moves every dependent
shape; the same seed gives the same inclusions, byte for byte in the case file; a
formula field's range is the server's own evaluation at the cells (and the page's
formula language agrees with the server's, checked through Node when it is installed).
A dimensioned plate reaching "fully defined" is planegcs's count, checked in the page.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess

import numpy as np
import pytest

pytest.importorskip("shapely")

from atlas.studio import document as D  # noqa: E402
from atlas.studio import formula  # noqa: E402
from atlas.studio import geometry_eval as GE  # noqa: E402
from atlas.studio import protocol  # noqa: E402
from atlas.studio.resolve import resolve  # noqa: E402
from atlas.studio.session import Session  # noqa: E402
from atlas.workbench import geometry as wgeo  # noqa: E402
from atlas.workbench.spec import example_case  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _session(shape="plate", W=0.4, H=0.2):
    s = Session(spec=example_case("wall-2"), key="x")
    protocol.handle(s, {"id": 1, "type": "geometry.start", "shape": shape, "params": {"W": W, "H": H}})
    return s


def _edit(s, ops, label="edit"):
    r, pushes = protocol.handle(s, {"id": 9, "type": "edit.apply", "ops": ops, "label": label})
    return r, pushes


# -- formulas -----------------------------------------------------------------------------

def test_the_formula_language_and_its_refusals():
    env = {"W": 0.6, "x": np.array([0.0, 1.0])}
    assert formula.value("W − 0.15", env) == pytest.approx(0.45)
    assert formula.value("2^3^2", {}) == 512                       # right-associative
    assert formula.value("-2^2", {}) == -4
    assert formula.value("max(1, 2) * min(3, 4) + sqrt(16)", {}) == 10
    assert formula.value("3 × 4 ÷ 2", {}) == 6
    assert np.allclose(formula.evaluate(formula.parse("exp(-x)"), env), np.exp(-env["x"]))
    for bad, why in (("__import__('os')", "cannot read"), ("open(1)", "no function"),
                     ("W +", "unexpected"), ("1 2", "unexpected"), ("q", "not defined"),
                     ("min(1)", "takes 2")):
        with pytest.raises(formula.FormulaError) as e:
            formula.value(bad, env)
        assert why in str(e.value), (bad, str(e.value))


def test_parameters_resolve_in_order_and_a_circle_is_refused():
    P = D.Parameter
    vals = formula.resolve_parameters([P(name="x_h", expr="W - 0.15"), P(name="W", value=0.6),
                                       P(name="d", expr="x_h / 3")])
    assert vals == pytest.approx({"x_h": 0.45, "W": 0.6, "d": 0.15})
    with pytest.raises(formula.FormulaError) as e:
        formula.resolve_parameters([P(name="a", expr="b"), P(name="b", expr="a + 1")])
    assert "circle" in str(e.value)


def test_changing_w_moves_every_dependent_shape():
    """P2's done-when: the plate's width, the hole's position and a slot's width all
    follow W."""
    s = _session("plate-hole", 0.6, 0.3)
    _edit(s, [{"op": "set", "path": ["geometry", "parameters"], "value": [
        {"name": "W", "value": 0.6}, {"name": "x_h", "expr": "W - 0.15"}]},
        {"op": "set", "path": ["geometry", "features", 0, "bind"], "value": {"w": "W"}},
        {"op": "set", "path": ["geometry", "features", 1, "bind"], "value": {"cx": "x_h"}}],
        "bound W")
    g = s.geom.by_id()
    assert g["f1"].w == pytest.approx(0.6) and g["f2"].cx == pytest.approx(0.45)
    r, _ = _edit(s, [{"op": "set", "path": ["geometry", "parameters", 0, "value"], "value": 0.8}],
                 "W = 0.8")
    assert r["ok"]
    g = s.geom.by_id()
    assert g["f1"].w == pytest.approx(0.8) and g["f2"].cx == pytest.approx(0.65)
    assert s.geom.parameters[1].value == pytest.approx(0.65)
    assert s.spec.domain.nx * s.spec.domain.dx == pytest.approx(0.8, abs=s.spec.domain.dx)
    # a binding the kind cannot hold is refused, and changes nothing
    r, pushes = _edit(s, [{"op": "set", "path": ["geometry", "features", 1, "bind"],
                           "value": {"w": "W"}}])
    assert r["ok"] is False and s.geom.by_id()["f2"].bind == {"cx": "x_h"}


def test_the_same_seed_gives_the_same_inclusions_byte_for_byte(tmp_path):
    s = _session("plate", 0.6, 0.3)
    scat = {"id": "f2", "kind": "scatter", "role": "region", "name": "Inclusions",
            "material": "copper", "x": 0.05, "y": 0.05, "w": 0.5, "h": 0.2,
            "params": {"count": 10, "rmin": 0.008, "rmax": 0.02, "gap": 0.005, "seed": 7}}
    _edit(s, [{"op": "insert", "path": ["geometry", "features"], "value": scat}], "scatter")
    items = s.geom.by_id()["f2"].items
    assert len(items) == 10
    s.save("a")
    s2 = _session("plate", 0.6, 0.3)
    _edit(s2, [{"op": "insert", "path": ["geometry", "features"], "value": scat}], "scatter")
    s2.cases_dir = s.cases_dir = str(tmp_path)
    s.save("a")
    s2.save("b")
    a = json.loads((tmp_path / "a.json").read_text())["geometry"]
    b = json.loads((tmp_path / "b.json").read_text())["geometry"]
    assert json.dumps(a["features"][1]["items"]) == json.dumps(b["features"][1]["items"])
    # another seed, another layout; the inclusions are the engine's regions
    _edit(s2, [{"op": "set", "path": ["geometry", "features", 1, "params", "seed"], "value": 8}])
    assert s2.geom.by_id()["f2"].items != items
    assert sum(1 for r in s2.spec.regions if r.material == "copper") == 1
    assert len(s2.spec.domain.holes) == 0


def test_pattern_mirror_and_channel_keep_their_sides():
    s = example_case("wall-2")
    F = lambda **k: D.Feature(**k)  # noqa: E731
    g = D.Geometry(features=[
        F(id="f1", kind="rect", role="add", x=0, y=0, w=0.4, h=0.1),
        F(id="f2", kind="ellipse", role="cut", cx=0.05, cy=0.05, rx=0.01, ry=0.01),
        F(id="f3", kind="pattern", role="cut", inputs=["f2"], params={"mode": "line", "count": 4, "dx": 0.1})])
    r = GE.evaluate(g, s)
    assert len(r.domain.holes) == 4 and all(h.kinds() == ["arc"] * 4 for h in r.domain.holes)
    g = D.Geometry(features=[
        F(id="f1", kind="rect", role="add", x=0, y=0, w=0.4, h=0.1),
        F(id="f2", kind="ellipse", role="cut", cx=0.1, cy=0.05, rx=0.01, ry=0.01),
        F(id="f3", kind="mirror", role="cut", inputs=["f2"], params={"axis": "x", "at": 0.2})])
    r = GE.evaluate(g, s)
    xs = sorted(round(float(np.mean(np.array(h.points)[:, 0]) * s.domain.dx), 4) for h in r.domain.holes)
    assert len(r.domain.holes) == 2 and xs[1] == pytest.approx(0.3, abs=0.02)
    g = D.Geometry(features=[
        F(id="f1", kind="rect", role="add", x=0, y=0, w=0.4, h=0.1, material="steel"),
        F(id="f2", kind="channel", role="region", material="copper", w=0.01,
          points=[(0.0, 0.05), (0.2, 0.06), (0.4, 0.05)], edges=["line", "line"])])
    r = GE.evaluate(g, s)
    shp = r.evaluator.shapes["f2"]
    assert shp.area == pytest.approx(0.01 * (math.hypot(0.2, 0.01) * 2), rel=0.02)
    assert [reg.material for reg in r.regions] == ["steel", "copper"]


def test_a_fillet_rounds_a_corner_with_a_tangent_arc():
    s = _session("l-bracket", 0.4, 0.3)
    f = s.geom.by_id()["f1"]
    n0 = len(f.points)
    r, _ = protocol.handle(s, {"id": 1, "type": "geometry.fillet", "feature": "f1", "vertex": 1,
                               "r": 0.02})
    assert r["ok"], r
    f = s.geom.by_id()["f1"]
    assert len(f.points) == n0 + 1 and f.edges[1] == "arc"
    assert abs(f.bulge[1]) == pytest.approx(math.tan(math.pi / 8))        # a quarter turn
    # the arc is tangent: its ends are r from the corner (0.4, 0) along each edge
    assert f.points[1] == pytest.approx((0.38, 0.0)) and f.points[2] == pytest.approx((0.4, 0.02))
    r, _ = protocol.handle(s, {"id": 2, "type": "geometry.fillet", "feature": "f1", "vertex": 1,
                               "r": 1.0})
    assert r["ok"] is False


def test_a_formula_field_is_binned_into_materials_and_reports_its_range():
    """P2's done-when: the range the page shows is the server's evaluation at the
    cells; the engine runs the field as at most 16 materials."""
    s = _session("plate", 0.4, 0.2)
    expr = "45 + 955 * exp(-((x - 0.2)^2 + (y - 0.1)^2) / 0.002)"
    _edit(s, [{"op": "insert", "path": ["geometry", "features"], "value": {
        "id": "f2", "kind": "rect", "role": "region", "name": "Patch", "material": "steel",
        "x": 0.1, "y": 0.05, "w": 0.2, "h": 0.1, "field": {"kind": "formula", "prop": "k", "expr": expr}}}],
        "field")
    info = s.fields["f2"]
    d = s.spec.domain
    # the cells the engine gives the field's materials, read back from its own regions
    own = wgeo.region_owner(s.spec.regions, d.nx, d.ny)
    cells = np.zeros(own.shape, dtype=bool)
    for k, reg in enumerate(s.spec.regions):
        if reg.material.startswith("f2-"):
            cells |= own == k
    cells &= wgeo.domain_mask(d)
    rows, cols = np.nonzero(cells)
    x, y = (cols + 0.5) * d.dx, (rows + 0.5) * d.dx
    v = 45 + 955 * np.exp(-((x - 0.2) ** 2 + (y - 0.1) ** 2) / 0.002)
    # the patch is 0.2 x 0.1 on whole cells: every one of its cells, and no other
    assert info["cells"] == len(v) == round(0.2 / d.dx) * round(0.1 / d.dx)
    assert info["min"] == pytest.approx(v.min()) and info["max"] == pytest.approx(v.max())
    assert 2 <= info["levels"] <= 16
    field_mats = {m for m in s.spec.materials if m.startswith("f2-")}
    assert len(field_mats) == info["levels"]
    ks = sorted(s.spec.materials[m]["k"] for m in field_mats)
    assert ks[0] >= v.min() - 1e-9 and ks[-1] <= v.max() + 1e-9
    # a field reading T, or an unknown name, is refused with its reason
    r, pushes = _edit(s, [{"op": "set", "path": ["geometry", "features", 1, "field", "expr"],
                           "value": "45 + T"}])
    assert any("T" in n["message"] for n in s.geo_notes if n.get("kind") == "field")


def test_channel_inclusions_are_strips_and_a_seed_is_a_layout():
    """Scatter's channels: strips of length 2r and width 2r/aspect at a drawn angle;
    the same seed gives the same strips, and the engine's region area is their sum."""
    s = _session("plate", 0.6, 0.3)
    scat = {"id": "f2", "kind": "scatter", "role": "region", "name": "Strips",
            "material": "copper", "x": 0.05, "y": 0.05, "w": 0.5, "h": 0.2,
            "params": {"count": 6, "rmin": 0.02, "rmax": 0.04, "gap": 0.005, "seed": 3,
                       "shape": "channels", "aspect": 8}}
    _edit(s, [{"op": "insert", "path": ["geometry", "features"], "value": scat}], "strips")
    items = s.geom.by_id()["f2"].items
    assert len(items) == 6 and all(len(it) == 4 for it in items)
    shape = GE.Evaluator(s.geom, 1e-4).shapes["f2"]
    want = sum((2 * it[2]) * (2 * it[2] / 8) for it in items)
    assert shape.area == pytest.approx(want, rel=1e-9)
    s2 = _session("plate", 0.6, 0.3)
    _edit(s2, [{"op": "insert", "path": ["geometry", "features"], "value": scat}], "strips")
    assert json.dumps(s2.geom.by_id()["f2"].items) == json.dumps(items)
    # both: each inclusion one or the other
    _edit(s2, [{"op": "set", "path": ["geometry", "features", 1, "params", "shape"], "value": "both"},
               {"op": "set", "path": ["geometry", "features", 1, "params", "count"], "value": 14}])
    kinds = {len(it) for it in s2.geom.by_id()["f2"].items}
    assert kinds == {3, 4}
    r, _ = _edit(s2, [{"op": "set", "path": ["geometry", "features", 1, "params", "shape"], "value": "stars"}])
    assert r["ok"] is False or s2.geom.by_id()["f2"].params["shape"] != "stars"


def test_an_image_field_maps_its_grey_levels_onto_the_range():
    """An image field: black is the minimum, white the maximum, on a log scale."""
    import base64
    s = _session("plate", 0.4, 0.2)
    grid = {"nx": 2, "ny": 1, "data": base64.b64encode(bytes([0, 255])).decode("ascii")}
    field = {"kind": "image", "prop": "k", "min": 1.0, "max": 100.0, "scale": "log",
             "grid": grid, "box": [0.1, 0.05, 0.3, 0.15]}
    _edit(s, [{"op": "insert", "path": ["geometry", "features"], "value": {
        "id": "f2", "kind": "rect", "role": "region", "name": "Patch", "material": "steel",
        "x": 0.1, "y": 0.05, "w": 0.2, "h": 0.1, "field": field}}], "image field")
    info = s.fields["f2"]
    assert info["min"] == pytest.approx(1.0) and info["max"] == pytest.approx(100.0)
    assert info["levels"] == 2
    ks = sorted(s.spec.materials[m]["k"] for m in s.spec.materials if m.startswith("f2-"))
    assert ks == pytest.approx([1.0, 100.0])
    # the middle of the grey scale is the geometric mean on a log scale
    v = GE.image_values({**field, "grid": {"nx": 1, "ny": 1, "data": base64.b64encode(bytes([128])).decode()}},
                        np.array([0.2]), np.array([0.1]))
    assert v[0] == pytest.approx(100 ** (128 / 255))


NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="Node is not installed")
def test_the_page_formula_language_agrees_with_the_server():
    src = os.path.join(ROOT, "atlas", "studio", "static", "js", "geo", "formula.js")
    cases = ["W - 0.15", "45 + 955 * exp(-((x - 0.2)^2 + (y - 0.1)^2) / 0.002)",
             "-2^2", "2^3^2", "max(x, y) * min(3, 4) + sqrt(16)", "3 × 4 ÷ 2 − 1",
             "tanh(x) + abs(-y) + log(e) + cos(pi)"]
    env = {"W": 0.6, "x": 0.13, "y": 0.07}
    script = ("import('file:///" + src.replace("\\", "/") + "').then(m => {"
              f"const env = {json.dumps(env)}; const out = {json.dumps(cases)}.map(c => m.value(c, env));"
              "console.log(JSON.stringify(out)); });")
    out = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True, text=True,
                         timeout=60)
    assert out.returncode == 0, out.stderr
    js = json.loads(out.stdout)
    py = [formula.value(c, env) for c in cases]
    assert js == pytest.approx(py, rel=1e-12)


@pytest.mark.skipif(NODE is None, reason="Node is not installed")
def test_a_dimensioned_plate_reaches_fully_defined_in_the_page_solver():
    """P2's done-when, in the page's own solver (planegcs through geo/gcs.js): a plate
    has 4 degrees of freedom; its width and height take 2; a fix takes the last 2. A
    hole adds 4, and its diameter and a position bound to W and H take them all. A
    constrained path keeps its constraints while one of its points is dragged."""
    js = os.path.join(ROOT, "atlas", "studio", "static", "js", "geo", "gcs.js").replace("\\", "/")
    dist = os.path.join(ROOT, "atlas", "studio", "static", "vendor", "planegcs", "dist").replace("\\", "/")
    rect = {"id": "f1", "kind": "rect", "x": 0, "y": 0, "w": 0.6, "h": 0.3, "bind": {}}
    hole = {"id": "f2", "kind": "ellipse", "cx": 0.45, "cy": 0.15, "rx": 0.04, "ry": 0.04, "bind": {}}
    stages = [
        {"features": [rect], "constraints": []},
        {"features": [{**rect, "bind": {"w": "W", "h": "H"}}], "constraints": []},
        {"features": [{**rect, "bind": {"w": "W", "h": "H"}}], "constraints": [{"id": "k1", "type": "fix", "refs": ["f1"]}]},
        {"features": [{**rect, "bind": {"w": "W", "h": "H"}}, hole], "constraints": [{"id": "k1", "type": "fix", "refs": ["f1"]}]},
        {"features": [{**rect, "bind": {"w": "W", "h": "H"}},
                      {**hole, "bind": {"rx": "d/2", "ry": "d/2", "cx": "W - 0.15", "cy": "H/2"}}],
         "constraints": [{"id": "k1", "type": "fix", "refs": ["f1"]},
                         {"id": "k2", "type": "distance", "refs": ["f1:c1", "f2:c"], "param": "x_h", "value": None}]},
    ]
    square = {"id": "f3", "kind": "path", "closed": True, "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
              "edges": ["line"] * 4, "bulge": [0] * 4, "handles": [None] * 4}
    cons = [{"id": f"k{i}", "type": t, "refs": [f"f3:e{i}"]} for i, t in enumerate(["horizontal", "vertical", "horizontal", "vertical"])]
    script = (
        f"const g = await import('file:///{js}'); const pg = await import('file:///{dist}/index.js');"
        f"const mod = {{ make: () => pg.make_gcs_wrapper('{dist}/planegcs_dist/planegcs.wasm') }};"
        f"const out = []; for (const st of {json.dumps(stages)}) out.push((await g.freedom(st, mod)).dof);"
        f"const sq = {json.dumps(square)}; const geo = {{ features: [sq], constraints: {json.dumps(cons)} }};"
        "out.push((await g.freedom(geo, mod)).dof);"
        "out.push(await g.dragPoint(geo, sq, 2, 1.5, 1.25, mod));"
        "console.log(JSON.stringify(out));")
    out = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    res = json.loads(out.stdout)
    assert res[:5] == [4, 2, 0, 4, 0]
    assert res[5] == 4                      # a square held to H and V edges: x, y, w, h
    pts = res[6]
    assert pts[2] == pytest.approx([1.5, 1.25], abs=1e-9)
    assert pts[1][0] == pytest.approx(1.5, abs=1e-9) and pts[3][1] == pytest.approx(1.25, abs=1e-9)
    assert pts[0][1] == pytest.approx(pts[1][1], abs=1e-9) and pts[0][0] == pytest.approx(pts[3][0], abs=1e-9)


def test_a_sweep_runs_the_case_once_per_value_on_copies():
    s = _session("plate", 0.3, 0.15)
    _edit(s, [{"op": "set", "path": ["geometry", "parameters"], "value": [
        {"name": "W", "value": 0.3, "sweep": [0.2, 0.3]}]},
        {"op": "set", "path": ["geometry", "features", 0, "bind"], "value": {"w": "W"}},
        {"op": "set", "path": ["run", "steps"], "value": 1}], "sweep W")
    before = s.spec.to_json()
    sw = s.start_sweep("W", ["serial", "full"])
    sw._thread.join(600)
    p = sw.progress()
    assert p["status"] == "done" and len(p["results"]) == 2, p
    assert all(r.get("status") == "done" and r["cards"]["failed"] == 0 for r in p["results"]), p
    assert s.spec.to_json() == before                       # the case itself is untouched
