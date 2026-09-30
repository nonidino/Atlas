"""The owner's first scenario, and what a new shape or a new physics is given to run.

The owner, 2026-09-30, after choosing the river and drawing a smooth blob: "these
four problems don't make sense. I just drew the geometry... This is only the first
scenario I've tried, and its unintuitive."  The four were a material no one had
given, an outfall at the river's default place in a grid still cut at the
wind farm's cell size, and no inlet or outlet.  These tests pin:

* the scenario itself, for every family: the case is ready to run the moment the
  shape is drawn, or says plainly the one thing only the person can decide;
* a new kind chosen in the header starts a new case with its example's scale and
  settings, and one Undo brings the old case back (2026-09-30: "When the simulation
  type changes, the full geometry should reset");
* the starting values fill only what is missing (`starter.fill_defaults`);
* the problems list's Fix buttons, one by one and all at once;
* the check never raises (it did, on a newly drawn cooled block);
* the canvas cannot move under the pointer while a shape is drawn.
"""

from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

pn = pytest.importorskip("panel")

from atlas.workbench import registry                                      # noqa: E402
from atlas.workbench.spec import Outline, check, example_case             # noqa: E402
from atlas.workbench.starter import apply_fix, fill_defaults              # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from workbench_ui import click, texts, widget, widgets                     # noqa: E402

#: the owner's blob, read off the screenshot: eight points, smooth edges
BLOB = [(55, 118), (106, 151), (163, 168), (225, 152), (262, 123), (237, 84), (157, 81),
        (97, 90)]


@pytest.fixture()
def wb(tmp_path, monkeypatch):
    from atlas.workbench import runner
    from atlas.workbench.app import Workbench
    monkeypatch.setattr(runner.CaseRun, "start", lambda self: None)
    return Workbench(cases_dir=str(tmp_path))


def _scenario(wb, fid):
    """The page's default case, the kind chosen in the header (a new case of it, since
    2026-09-30), the blob drawn as the domain's outline."""
    wb.type_sel.value = fid
    wb.geo_editor._switch(layer="domain", tool="draw")
    # the blob was drawn on the 352 x 240 grid every new case has; a kind that starts
    # from its example (the cooled block: 200 x 60) gets it scaled to its own grid
    d = wb.spec.domain
    pts = [(x * d.nx / 352.0, y * d.ny / 240.0) for x, y in BLOB]
    wb.geo_editor.add_drawn(Outline(points=pts, edges=["spline"] * 8, bulge=[0.0] * 8))
    return [i.message for i in wb.issues() if i.severity == "error"]


@pytest.mark.parametrize("fid", ["conduction-2d", "electric-2d", "transport-2d",
                                 "acoustics-2d", "elasticity-2d", "thermoelastic-2d"])
def test_the_first_scenario_is_ready_to_run(wb, fid):
    """Draw a shape, and the case runs: a material under it, and its family's ends
    (a river's inlet and outlet, a hot and a cold end, a clamp and a load, a plate's
    electrodes and circuit) on its leftmost and rightmost edges."""
    assert _scenario(wb, fid) == []
    assert wb.status_btn.name.startswith("✓ Ready")
    s = wb.spec
    assert s.regions and s.regions[0].id == "base"
    kinds = {b.kind for b in s.boundaries}
    want = {"transport-2d": {"river-inlet", "river-outlet"}, "electric-2d": {"electrode"},
            "conduction-2d": {"fixed-temperature"}, "elasticity-2d": {"clamped", "load-y"},
            "thermoelastic-2d": {"fixed-temperature"}}.get(fid, set())
    assert want <= kinds
    if fid == "electric-2d":
        assert {a.kind for a in s.attachments} == {"battery", "resistor"}
    assert any("to start" in line for line in wb.log_lines)


def test_the_owners_river(wb):
    """The four problems of the owner's screenshot, each given what it lacked: the
    river's own scale (cells of 5 m, not the farm's 0.03125), a material, the outfall
    in the water, an inlet and an outlet, and a step under the stability limit."""
    assert _scenario(wb, "transport-2d") == []
    s = wb.spec
    assert s.domain.dx == 5.0 and not s.devices
    from atlas.workbench import geometry as geo
    d = s.domain
    i, j = int(s.physics.get("source_x") // d.dx), int(s.physics.get("source_y") // d.dx)
    assert geo.domain_mask(d)[j, i]                          # the outfall is in the water
    from atlas.workbench.families.plume import explicit_limit
    assert s.run.macro_dt <= explicit_limit(s)


@pytest.mark.parametrize("fid, only", [
    ("incompressible-2d", ("rotor", "the flow enters on the grid's left edge",
                           "the flow leaves on the grid's right edge")),
    ("conjugate-heat-2d", ("nothing cools the block",))])
def test_what_only_the_person_can_decide_is_said_plainly(wb, fid, only):
    """A wind farm's air must reach the grid's left and right edges, and a cooled
    block needs its channel drawn: no default can guess these, so the problems say
    what to do, and the check does not raise (it did, on the block)."""
    errors = _scenario(wb, fid)
    assert errors and all(any(o in e for o in only) for e in errors), errors
    assert not any("have no material yet" in e for e in errors)


def test_a_new_type_starts_from_its_examples_setup_and_undo_restores_the_old(wb):
    """Choosing a kind in the header starts a new case of it with its first example's
    scale and settings (`spec.adapt_to_family` on a blank case), and nothing of the
    old case: the owner's "When the simulation type changes, the full geometry should
    reset".  One Undo brings the old case back, bit for bit."""
    before = wb.spec.model_dump()
    wb.type_sel.value = "elasticity-2d"
    s = wb.spec
    ex = example_case("bracket-2")
    assert (s.domain.dx, s.run.steps, s.coupling.tolerance, s.coupling.max_iterations) == (
        ex.domain.dx, ex.run.steps, ex.coupling.tolerance, ex.coupling.max_iterations)
    assert s.layout is not None and s.layout.along == len(ex.windows)
    assert not s.devices and s.domain.outline is None
    assert any("Undo" in line and "wind farm" in line for line in wb.log_lines[:3])
    wb.undo()
    assert wb.spec.model_dump() == before and len(wb.spec.devices) == 3


def test_starting_values_fill_only_what_is_missing():
    """A river that has its inlet, its material and its outfall keeps them."""
    s = example_case("river-bend")
    before = s.model_dump()
    assert fill_defaults(s) == [] and s.model_dump() == before
    s.boundaries = [b.model_copy(update={"kind": "bank"}) if b.kind == "river-outlet" else b
                    for b in s.boundaries]
    inlet = [b.id for b in s.boundaries if b.kind == "river-inlet"]
    done = fill_defaults(s)
    assert len(done) == 1 and "outlet" in done[0]
    assert [b.id for b in s.boundaries if b.kind == "river-inlet"] == inlet
    assert not any(i.severity == "error" for i in check(s))
    assert apply_fix(s, "river-outlet") is None               # nothing more to do


def test_the_windows_fix_cuts_dirichlet_neumann_at_the_interface():
    """A style-C case with two media gets its two pieces at the interface it already
    has, and a newly drawn domain is never refused for its windows' sake: the cooled
    block, which now starts from its example, had its drawing refused when drawn
    without its water (its windows are one per physics; 2026-09-30)."""
    s = example_case("wall-2")
    s.windows = []
    assert "one per material" in apply_fix(s, "windows")
    assert s.layout.cut == "materials" and sorted(w.id for w in s.windows) == ["copper",
                                                                               "steel"]
    assert not [i for i in check(s) if i.severity == "error"]


def test_the_problems_list_fixes_one_or_all(wb):
    wb.type_sel.value = "transport-2d"
    wb.geo_editor._switch(layer="domain", tool="draw")
    wb.geo_editor.add_drawn(Outline(points=BLOB, edges=["line"] * 8, bulge=[0.0] * 8))
    # take the inlet and the outlet away again, and put the outfall on dry ground
    wb.edit(lambda c: [setattr(b, "kind", "bank") for b in c.boundaries],
            "every edge a bank")
    wb.edit(lambda c: c.physics.params.update(source_x=10.0, source_y=10.0), "outfall")
    click(wb.status_btn)
    shows = widgets(wb.modal_body, pn.widgets.Button, name="Show me")
    fixes = widgets(wb.modal_body, pn.widgets.Button, prefix="Make the leftmost")
    assert len(shows) == len(wb.issues()) and len(fixes) == 1
    click(fixes[0])
    assert any(b.kind == "river-inlet" for b in wb.spec.boundaries)
    assert "Done:" in wb.log_lines[1] or "Done:" in wb.log_lines[0]
    # the list came back with what is left, and fixes all of it at once
    click(widget(wb.modal_body, "Fix the 2 that can be fixed", pn.widgets.Button))
    assert [i for i in wb.issues() if i.severity == "error"] == []
    wb.undo()
    wb.undo()
    assert not any(b.kind in ("river-inlet", "river-outlet") for b in wb.spec.boundaries)


def test_every_type_from_the_header_is_ready_to_run(wb):
    """Done-when of item 1.1: each of the eight kinds picked from the header is ready to
    run on the whole grid (a blank case was a wind farm with no windows, an error from
    the start), the cooled block, which needs its channel, from its own example; and
    Case > Start over is a new case of the same kind."""
    assert list(wb.type_sel.options.values()) == registry.available_ids()
    for fid in registry.available_ids():
        wb.type_sel.value = fid
        assert wb.spec.physics.family == fid and wb.save_state.object == "not saved yet"
        assert [i.message for i in wb.issues() if i.severity == "error"] == [], fid
    wb.type_sel.value = "incompressible-2d"
    wb.edit(lambda c: setattr(c.run, "steps", 3), "steps")
    wb.dispatch("file:start-over")
    assert wb.spec.run.steps != 3 and wb.spec.physics.family == "incompressible-2d"
    assert [(w.x0, w.y0, w.nx) for w in wb.spec.windows] == [
        (w.x0, w.y0, w.nx) for w in example_case("wake-array-3").windows]   # the rung


def test_the_check_never_raises():
    """A family's check that raised left a newly drawn cooled block unusable: every
    part of the check now reports instead."""
    s = example_case("cooled-block")
    s.domain.outline = Outline(points=BLOB, edges=["line"] * 8, bulge=[0.0] * 8)
    s.boundaries = []
    issues = check(s)
    assert any(i.severity == "error" for i in issues)


def test_the_canvas_holds_still_while_a_shape_is_drawn(wb):
    """Eight clicks made a three-vertex outline: the hint grew a line as points were
    placed, the canvas shrank under the pointer, and the next click landed elsewhere.
    The hint and the summary have fixed heights, and the canvas keeps its cells
    square in the browser (SQUARE_JS) whatever size it is given."""
    ed = wb.geo_editor
    assert ed.hint.height and ed.summary.height
    cbs = ed.fig.js_property_callbacks
    assert "change:inner_width" in cbs and "change:inner_height" in cbs
    assert ed.fig.sizing_mode == "stretch_both"


def test_the_header_holds_what_it_simulates_and_fits_a_laptop(wb):
    """The header's first control is what the case simulates, the one place it is
    chosen (the Physics layer only says so); Undo and Redo sit in the canvas's toolbar;
    the header holds no name (it ran past a laptop's 1,090 px, seen; the page's fit is
    measured in the served page)."""
    header = wb.tpl.header[0]
    selects = list(pn.panel(header).select(pn.widgets.Select))
    assert selects == [wb.type_sel] and wb.type_sel.value == "incompressible-2d"
    assert wb.undo_btn not in list(pn.panel(header).select(pn.widgets.Button))
    assert wb.examples_menu in list(pn.panel(header).select(pn.widgets.MenuButton))
    wb.geo_editor._switch(layer="physics")
    assert not widgets(wb.geo_editor.inspector, pn.widgets.Select, name="What it simulates")
    assert "chosen in the header" in texts(wb.geo_editor.inspector)
    assert wb.undo_btn in widgets(wb.workspace, pn.widgets.Button)
