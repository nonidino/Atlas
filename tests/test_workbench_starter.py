"""The owner's first scenario, and what a new shape or a new physics is given to run.

The owner, 2026-09-30, after choosing the river and drawing a smooth blob: "these
four problems don't make sense. I just drew the geometry... This is only the first
scenario I've tried, and its unintuitive."  The four were a material no one had
given, an outfall at the river's default place in a grid still cut at the
wind farm's cell size, and no inlet or outlet.  These tests pin, driven as the
studio's page drives them (its protocol, `tests/studio_driver.py`):

* the scenario itself, for every family: the case is ready to run the moment the
  shape is drawn, or says plainly the one thing only the person can decide;
* a new kind chosen starts a new case with its example's scale and settings, and
  one Undo brings the old case back (2026-09-30: "When the simulation type changes,
  the full geometry should reset");
* the starting values fill only what is missing (`starter.fill_defaults`);
* the problems list's fixes, one by one and all at once;
* the check never raises (it did, on a newly drawn cooled block).

That the canvas holds still while a shape is drawn is the page's own, checked in the
real page (`scripts/studio_drive.py`).
"""

from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from atlas.workbench import registry                                      # noqa: E402
from atlas.workbench.spec import Outline, check, example_case             # noqa: E402
from atlas.workbench.starter import apply_fix, fill_defaults              # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import studio_driver as D                                                 # noqa: E402

#: the owner's blob, read off the screenshot: eight points, smooth edges
BLOB = [(55, 118), (106, 151), (163, 168), (225, 152), (262, 123), (237, 84), (157, 81),
        (97, 90)]


@pytest.fixture()
def s(tmp_path, monkeypatch):
    """A page's session on the wind farm (where the retired page opened), whose runs
    are created and never marched."""
    D.idle(monkeypatch)
    return D.session(tmp_path)


def _scenario(s, fid, edges=None):
    """A new case of the kind ``fid``, and the blob drawn as the domain's outline.  The
    blob was drawn on the 352 x 240 grid every new case has; a kind that starts from
    its example (the cooled block: 200 x 60) gets it scaled to its own grid."""
    D.ok(s, "case.new", family=fid)
    d = s.spec.domain
    pts = [(x * d.nx / 352.0, y * d.ny / 240.0) for x, y in BLOB]
    r, pushes = D.draw_domain(s, pts, edges=edges or ["spline"] * 8)
    assert r["ok"], (fid, r, D.notices(pushes))
    return D.errors(s), pushes


@pytest.mark.parametrize("fid", ["conduction-2d", "electric-2d", "transport-2d",
                                 "acoustics-2d", "elasticity-2d", "thermoelastic-2d"])
def test_the_first_scenario_is_ready_to_run(s, fid):
    """Draw a shape, and the case runs: a material under it, and its family's ends
    (a river's inlet and outlet, a hot and a cold end, a clamp and a load, a plate's
    electrodes and circuit) on its leftmost and rightmost edges, each said."""
    errors, pushes = _scenario(s, fid)
    assert errors == []
    assert s.status()["kind"] == "ok"
    sp = s.spec
    lib = registry.family(fid).material_library()
    assert sp.regions and all(r.material in sp.materials or r.material in lib
                              for r in sp.regions)
    kinds = {b.kind for b in sp.boundaries}
    want = {"transport-2d": {"river-inlet", "river-outlet"}, "electric-2d": {"electrode"},
            "conduction-2d": {"fixed-temperature"}, "elasticity-2d": {"clamped", "load-y"},
            "thermoelastic-2d": {"fixed-temperature"}}.get(fid, set())
    assert want <= kinds
    if fid == "electric-2d":
        assert {a.kind for a in sp.attachments} == {"battery", "resistor"}
    if want:
        said = " ".join(D.notices(pushes, "info"))
        assert "To start" in said and "Undo takes it all back" in said, said


def test_the_owners_river(s):
    """The four problems of the owner's screenshot, each given what it lacked: the
    river's own scale (cells of 5 m, not the farm's 0.03125), a material, the outfall
    in the water, an inlet and an outlet, and a step under the stability limit."""
    assert _scenario(s, "transport-2d")[0] == []
    sp = s.spec
    assert sp.domain.dx == 5.0 and not sp.devices
    from atlas.workbench import geometry as geo
    d = sp.domain
    i, j = int(sp.physics.get("source_x") // d.dx), int(sp.physics.get("source_y") // d.dx)
    assert geo.domain_mask(d)[j, i]                          # the outfall is in the water
    from atlas.workbench.families.plume import explicit_limit
    assert sp.run.macro_dt <= explicit_limit(sp)


@pytest.mark.parametrize("fid, only", [
    ("incompressible-2d", ("rotor", "the flow enters on the grid's left edge",
                           "the flow leaves on the grid's right edge")),
    ("conjugate-heat-2d", ("nothing cools the block",))])
def test_what_only_the_person_can_decide_is_said_plainly(s, fid, only):
    """A wind farm's air must reach the grid's left and right edges, and a cooled
    block needs its channel drawn: no default can guess these, so the problems say
    what to do, and the check does not raise (it did, on the block)."""
    errors, _ = _scenario(s, fid)
    assert errors and all(any(o in e for o in only) for e in errors), errors
    assert not any("have no material yet" in e for e in errors)


def test_a_new_type_starts_from_its_examples_setup_and_undo_restores_the_old(s):
    """Choosing a kind starts a new case of it with its first example's scale and
    settings (`spec.adapt_to_family` on a blank case), and nothing of the old case: the
    owner's "When the simulation type changes, the full geometry should reset".  One
    Undo brings the old case back, bit for bit."""
    before = s.spec.model_dump()
    said = D.notices(D.ok(s, "case.new", family="elasticity-2d"))
    sp = s.spec
    ex = example_case("bracket-2")
    assert (sp.domain.dx, sp.run.steps, sp.coupling.tolerance, sp.coupling.max_iterations) == (
        ex.domain.dx, ex.run.steps, ex.coupling.tolerance, ex.coupling.max_iterations)
    assert sp.layout is not None and sp.layout.along == len(ex.windows)
    assert not sp.devices and sp.domain.outline is None
    assert any("Undo" in n and "wind farm" in n for n in said), said
    D.ok(s, "edit.undo")
    assert s.spec.model_dump() == before and len(s.spec.devices) == 3


def test_starting_values_fill_only_what_is_missing():
    """A river that has its inlet, its material and its outfall keeps them."""
    sp = example_case("river-bend")
    before = sp.model_dump()
    assert fill_defaults(sp) == [] and sp.model_dump() == before
    sp.boundaries = [b.model_copy(update={"kind": "bank"}) if b.kind == "river-outlet" else b
                     for b in sp.boundaries]
    inlet = [b.id for b in sp.boundaries if b.kind == "river-inlet"]
    done = fill_defaults(sp)
    assert len(done) == 1 and "outlet" in done[0]
    assert [b.id for b in sp.boundaries if b.kind == "river-inlet"] == inlet
    assert not any(i.severity == "error" for i in check(sp))
    assert apply_fix(sp, "river-outlet") is None               # nothing more to do


def test_the_windows_fix_cuts_dirichlet_neumann_at_the_interface():
    """A style-C case with two media gets its two pieces at the interface it already
    has, and a newly drawn domain is never refused for its windows' sake: the cooled
    block, which now starts from its example, had its drawing refused when drawn
    without its water (its windows are one per physics; 2026-09-30)."""
    sp = example_case("wall-2")
    sp.windows = []
    assert "one per material" in apply_fix(sp, "windows")
    assert sp.layout.cut == "materials" and sorted(w.id for w in sp.windows) == ["copper",
                                                                                 "steel"]
    assert not [i for i in check(sp) if i.severity == "error"]


def test_the_problems_list_fixes_one_or_all(s):
    """The problems list offers each one-click fix; one fixes one problem and says
    what it did, and *Fix all* sends every key at once (``case.fix``); each is one
    edit, and Undo takes it back."""
    assert _scenario(s, "transport-2d", edges=["line"] * 8)[0] == []
    # take the inlet and the outlet away again, and put the outfall on dry ground
    n = len(s.spec.boundaries)
    r, _ = D.edit(s, *[D.set_op(["boundaries", k, "kind"], "bank") for k in range(n)],
                  label="every edge a bank")
    assert r["ok"]
    r, _ = D.edit(s, D.set_op(["physics", "params", "source_x"], 10.0),
                  D.set_op(["physics", "params", "source_y"], 10.0), label="outfall")
    assert r["ok"]
    rows = s.state()["issues"]
    assert len(rows) == len(check(s.spec))
    inlet = [r for r in rows if (r["fix_label"] or "").startswith("Make the leftmost")]
    assert len(inlet) == 1
    said = D.notices(D.ok(s, "case.fix", keys=[inlet[0]["fix"]]))
    assert any(b.kind == "river-inlet" for b in s.spec.boundaries)
    assert any(n.startswith("Done:") for n in said), said
    # the list comes back with what is left, and fixes all of it at once
    left = sorted({r["fix"] for r in s.state()["issues"] if r["fix"]})
    assert len(left) == 2, left
    D.ok(s, "case.fix", keys=left)
    assert D.errors(s) == []
    D.ok(s, "edit.undo")
    D.ok(s, "edit.undo")
    assert not any(b.kind in ("river-inlet", "river-outlet") for b in s.spec.boundaries)


def test_every_type_is_ready_to_run(s):
    """Done-when of item 1.1: each of the eight kinds is ready to run on the whole grid
    (a blank case was a wind farm with no windows, an error from the start), the cooled
    block, which needs its channel, from its own example; and Start over is a new case
    of the same kind."""
    ids = [f["id"] for f in s.catalog()["families"] if f["runs"]]
    assert ids == registry.available_ids()
    for fid in ids:
        D.ok(s, "case.new", family=fid)
        meta = s.state()["meta"]
        assert s.spec.physics.family == fid and meta["file"] is None and not meta["dirty"]
        assert D.errors(s) == [], fid
    D.ok(s, "case.new", family="incompressible-2d")
    D.edit(s, D.set_op(["run", "steps"], 3), label="steps")
    D.ok(s, "case.start_over")
    assert s.spec.run.steps != 3 and s.spec.physics.family == "incompressible-2d"
    assert [(w.x0, w.y0, w.nx) for w in s.spec.windows] == [
        (w.x0, w.y0, w.nx) for w in example_case("wake-array-3").windows]   # the rung


def test_the_check_never_raises():
    """A family's check that raised left a newly drawn cooled block unusable: every
    part of the check now reports instead."""
    sp = example_case("cooled-block")
    sp.domain.outline = Outline(points=BLOB, edges=["line"] * 8, bulge=[0.0] * 8)
    sp.boundaries = []
    issues = check(sp)
    assert any(i.severity == "error" for i in issues)
