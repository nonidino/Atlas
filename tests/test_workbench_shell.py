"""The Atlas Workbench shell: the case file, the registry, and every menu action.

The GUI is driven through `Workbench.dispatch`, the one entry point every menu
item and button uses, so these tests exercise the same code the browser does
without a browser.  What they cannot see is whether the page draws it: that was
checked by clicking every control in a served page (the project's own rule, from
the RaceLab control that did nothing).
"""

from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

pn = pytest.importorskip("panel")

from atlas.workbench import registry                                    # noqa: E402
from atlas.workbench.spec import (EXAMPLES, CaseSpec, Window, blank_case,  # noqa: E402
                                  check, example_case)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from workbench_ui import click, texts, widget, widgets                  # noqa: E402


# ---------------------------------------------------------------------------
# the case file
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", list(EXAMPLES))
def test_every_example_is_a_clean_case(key):
    spec = example_case(key)
    assert [i for i in check(spec) if i.severity == "error"] == []
    assert CaseSpec.from_json(spec.to_json()) == spec


def test_examples_are_the_real_tilings():
    """Read from the case modules, so they cannot drift from the record."""
    from atlas.cases import scaling_ladder as sl
    s = example_case("wake-array-3")
    t = sl.rung(3, 2).tiling
    assert (s.domain.nx, s.domain.ny) == (t.nx, t.ny) == (352, 240)
    assert [(w.x0, w.y0) for w in s.windows] == list(t.offsets)
    assert [w.nx for w in s.windows] == [128] * 6
    assert [v.id for v in s.devices] == ["R1", "R2", "R3"]
    assert s.devices[0].x == pytest.approx(3.75)            # mid-overlap, in D
    assert len(example_case("farm-12").devices) == 12
    assert len(example_case("farm-21").devices) == 21


def test_check_finds_what_is_wrong():
    s = blank_case()
    assert any("not cut into any windows" in i.message for i in check(s))
    s = example_case("wake-array-3")
    s.windows.append(Window(id="F00", x0=300, y0=200, nx=128, ny=128))
    msgs = [i.message for i in check(s)]
    assert any("both called 'F00'" in m for m in msgs)
    assert any("reaches past the domain" in m for m in msgs)
    s = example_case("wake-array-3")
    s.windows = s.windows[:-1]                               # uncover a corner
    assert any("lie in no window" in i.message for i in check(s))


def test_save_and_load_round_trip(tmp_path):
    s = example_case("farm-12")
    path = s.save(str(tmp_path / "farm.json"))
    assert CaseSpec.load(path) == s
    saved = json.loads(open(path, encoding="utf-8").read())
    assert saved["schema_id"] == "atlas-workbench/case@0.6"   # any family, drawn or not
    assert "name" not in saved and "description" not in saved


def test_a_case_has_no_name_and_an_older_file_loses_its_own():
    """The owner, 2026-09-30: "There's no need to have names and descriptions for the
    individual cases, remove that."  case@0.6 has neither, and a 0.5 file that has
    them loads without them (a stray name in a 0.6 file is ignored, not kept)."""
    assert "name" not in CaseSpec.model_fields and "description" not in CaseSpec.model_fields
    old = json.loads(example_case("s-channel").to_json())
    old.update(schema_id="atlas-workbench/case@0.5", name="my channel",
               description="the one I drew")
    s = CaseSpec.from_json(json.dumps(old))
    assert s.schema_id == "atlas-workbench/case@0.6"
    assert s == example_case("s-channel")
    assert "my channel" not in s.to_json() and "the one I drew" not in s.to_json()


def test_registry_says_which_families_run_and_why_the_rest_do_not():
    assert registry.available_ids() == ["incompressible-2d", "conduction-2d", "electric-2d",
                                        "transport-2d", "acoustics-2d", "elasticity-2d",
                                        "thermoelastic-2d", "conjugate-heat-2d"]
    for f in registry.FAMILIES:
        assert f.note, f.id
        for src in f.sources:
            assert os.path.isfile(os.path.join(HERE, src)), src


# ---------------------------------------------------------------------------
# the shell
# ---------------------------------------------------------------------------


@pytest.fixture()
def wb(tmp_path, monkeypatch):
    """A workbench whose Run creates the run but does not march it: the menu tests
    are about the dispatcher, and a real march here would be a 30-second background
    thread timing itself against the rest of the suite (runs are tested in
    `test_workbench_runner.py`)."""
    from atlas.workbench import compile as compile_
    from atlas.workbench import runner
    from atlas.workbench.app import Workbench
    monkeypatch.setattr(runner.CaseRun, "start", lambda self: None)
    monkeypatch.setattr(compile_.CompileJob, "start", lambda self: None)
    return Workbench(cases_dir=str(tmp_path))


ALL_ACTIONS = ["file:start-over", "case:type:conduction-2d",
               *[f"file:example:{k}" for k in EXAMPLES], "file:open",
               "file:save-as", "file:import", "file:export", "edit:undo", "edit:redo",
               "edit:revert", "view:grid", "view:overlaps", "view:labels", "view:log",
               "run:check", "run:compile", "run:decomposed", "run:full", "run:both",
               "run:stop", "run:results", "help:guide", "help:about", "help:docs"]


def test_every_menu_action_runs(wb):
    for action in ALL_ACTIONS:
        wb.dispatch(action)
    for key in ("case", "geometry", "physics", "check", "run", "results"):
        wb.show(key)
        assert wb.workspace.objects


def test_every_header_control_reaches_the_dispatcher(wb):
    """A control nobody handles would be a control that does nothing: the Case and ?
    menus' items and the header's buttons all go through `dispatch`."""
    items = [it[1] for m in (wb.case_menu, wb.help_menu, wb.examples_menu)
             for it in m.items if it is not None]
    assert {"file:start-over", "file:open", "file:save", "file:save-as",
            "file:import", "file:export", "file:import-gmsh", "edit:revert",
            "view:history", "help:guide", "help:about", "help:docs"} <= set(items)
    assert not {"file:new", "file:gallery", "file:rename"} & set(items)
    for action in items:
        wb.dispatch(action)
    for button in (wb.status_btn, wb.undo_btn, wb.redo_btn, wb.run_btn):
        click(button)
    wb.type_sel.value = "electric-2d"
    assert wb.spec.physics.family == "electric-2d"
    assert not any("unknown action" in line for line in wb.log_lines)


def test_help_names_the_wiki_only_where_it_is(wb, monkeypatch, tmp_path):
    """The one-command install carries the code and its README, not the wiki (demo
    step 7): *Where things are documented* names the wiki's pages only where they
    are, and says plainly where they are not; *About* claims nothing about what was
    installed beforehand."""
    from atlas.workbench import app
    here = wb._docs().object
    if os.path.isdir(os.path.join(app.ROOT, "wiki")):
        assert "showcase-gallery.md" in here
    else:                                            # this test, inside the install
        assert "does not carry" in here
    monkeypatch.setattr(app, "ROOT", str(tmp_path))  # a copy with no wiki beside it
    there = wb._docs().object
    assert "does not carry" in there and "showcase-gallery.md" not in there
    assert "`atlas/workbench/README.md`" in there
    assert "already installed" not in wb._about().object


def test_more_examples_lists_the_types_own_with_no_names(wb):
    """The owner's O1 (2026-09-30): under the header's example button, the chosen
    kind's examples, one line each saying what it shows, with no name; the gallery of
    all 21, with their names and descriptions, left the page."""
    for fid in registry.available_ids():
        wb.dispatch(f"case:type:{fid}")
        items = wb.examples_menu.items
        # the kind's Fast example is the button's own part, not a line under it
        from atlas.workbench.spec import FAST_EXAMPLES
        mine = [k for k, ex in EXAMPLES.items() if ex.family == fid
                and k != FAST_EXAMPLES.get(fid)]
        assert [tuple(i) for i in items] == [(EXAMPLES[k].shows, f"file:example:{k}")
                                             for k in mine], fid
        for label, _action in items:
            assert not any(k in label or EXAMPLES[k].label in label for k in EXAMPLES)
    wb.dispatch("case:type:conduction-2d")
    before = wb.spec.model_dump()
    wb.dispatch("file:example:bend-3")
    assert wb.spec == example_case("bend-3") and wb.case_key == "bend-3"
    assert wb.path is None and wb.save_state.object == "not saved yet"
    wb.undo()
    assert wb.spec.model_dump() == before


def test_the_type_selector_starts_a_new_case_and_one_undo_restores_the_old(wb, tmp_path):
    """D1-D3 of the demo plan: the header's first control is what the case simulates;
    another kind starts a new case of it (the geometry resets), with no confirmation,
    and one Undo brings the old case back, bit for bit, with its file."""
    from atlas.workbench.spec import Outline
    from atlas.workbench.starter import new_case
    wb.dispatch("case:type:conduction-2d")
    wb.geo_editor._switch(layer="domain", tool="draw")
    wb.geo_editor.add_drawn(Outline(points=[(40, 40), (200, 60), (180, 200)],
                                    edges=["line"] * 3, bulge=[0.0] * 3))
    path = str(tmp_path / "mine.json")
    wb.save_to(path)
    before, key = wb.spec.model_dump(), wb.case_key
    assert wb.spec.domain.outline is not None and key == "mine"
    wb.type_sel.value = "transport-2d"                    # the header's selector
    s = wb.spec
    assert s.physics.family == "transport-2d" and s.domain.outline is None
    assert s.model_dump() == new_case("transport-2d").model_dump()
    assert wb.path is None and wb.case_key.startswith("river-plume-")
    assert "Undo" in wb.log_lines[0] and "heat conduction" in wb.log_lines[0]
    wb.undo()                                             # one Undo
    assert wb.spec.model_dump() == before
    assert wb.path == path and wb.case_key == "mine" and not wb.dirty
    assert wb.type_sel.value == "conduction-2d"           # the selector follows, no new case
    assert wb.spec.model_dump() == before
    wb.redo()
    assert wb.spec.physics.family == "transport-2d" and wb.path is None
    wb.dispatch("file:start-over")                        # a new case of the same kind
    assert wb.spec.physics.family == "transport-2d" and wb.path is None
    assert "Started over" in wb.log_lines[0]


def test_no_control_shows_a_case_name(wb):
    """Done-when of item 1.1: no control anywhere shows a case name or a description.
    Every example is opened; the header and the Model tab hold none of the example
    keys, labels or descriptions."""
    for key, ex in EXAMPLES.items():
        wb.dispatch(f"file:example:{key}")
        shown = texts(wb.tpl.header[0]) + "\n" + texts(wb.workspace)
        assert key not in shown.split(), key
        assert ex.label not in shown and ex.description not in shown, key


def test_nothing_is_left_unbuilt(wb):
    """Every NOT_BUILT went once its action worked; the compile was the last, and
    is a button on the Run & results tab."""
    import atlas.workbench.app as app
    assert not hasattr(app, "NOT_BUILT")
    wb.show("run")
    click(widget(wb.workspace, "Compile", pn.widgets.Button))
    assert wb.compile_job is not None              # started (the fixture does not march it)
    assert any("compile started" in line for line in wb.log_lines[:3])


def test_a_case_with_errors_does_not_run(wb):
    """Run on a case that cannot run lists its problems, where they are."""
    wb.set_spec(blank_case(), "blank")             # a blank case: no windows
    click(wb.run_btn)
    assert wb.run is None and wb.active == "model"
    assert any("cannot run" in line for line in wb.log_lines[:3])
    assert "stop the run" in texts(wb.modal_body) or "stops the run" in texts(wb.modal_body)
    assert widgets(wb.modal_body, pn.widgets.Button, name="Show me")
    wb.show("run")                                 # and the tab's own Run is off
    assert widget(wb.workspace, "▶ Run and compare", pn.widgets.Button).disabled


def test_edits_are_validated_undone_and_redone(wb):
    assert wb.edit(lambda s: setattr(s.domain, "nx", -5), "bad") is False
    assert wb.spec.domain.nx == 352
    assert wb.edit(lambda s: setattr(s.run, "steps", 7), "steps") is True
    assert wb.dirty and wb.spec.run.steps == 7
    wb.dispatch("edit:undo")
    assert wb.spec.run.steps == 40
    wb.dispatch("edit:redo")
    assert wb.spec.run.steps == 7


def test_save_open_and_the_case_bar(wb, tmp_path):
    path = str(tmp_path / "mine.json")
    assert wb.save_state.object == "not saved yet"
    wb.edit(lambda s: setattr(s.run, "steps", 7), "steps")
    assert wb.save_state.object == "unsaved changes"
    wb.save_to(path)
    assert not wb.dirty and wb.save_state.object == "saved" and wb.case_key == "mine"
    wb.dispatch("file:example:farm-12")
    assert wb.spec == example_case("farm-12") and wb.path is None
    assert wb.case_key == "farm-12"                  # its records' folder, not a name shown
    # a fresh example has nothing of the person's to lose (seen: it read "unsaved
    # changes" the moment it opened)
    assert wb.save_state.object == "not saved yet"
    wb.open_path(path)
    assert wb.spec.run.steps == 7 and wb.path == path and not wb.dirty
    assert wb.case_key == "mine"


def test_the_status_chip_lists_the_problems_and_where_each_is_fixed(wb):
    """The six steps' error counts became one chip in the header: it says whether
    the case can run, and its list sends each problem to the layer that fixes it."""
    assert wb.status_btn.name == "✓ Ready to run"
    wb.set_spec(blank_case(), "blank")
    assert wb.status_btn.name.startswith("⚠") and "problem" in wb.status_btn.name
    click(wb.status_btn)
    shows = widgets(wb.modal_body, pn.widgets.Button, name="Show me")
    issues = wb.issues()
    assert len(shows) == len(issues)
    k = next(i for i, x in enumerate(issues) if "not cut into any windows" in x.message)
    click(shows[k])
    assert wb.active == "model" and wb.geo_state["layer"] == "windows"
    assert wb.geo_state["tool"] == "select" and wb.geo_state["selected"] is None


def test_import_refuses_what_is_not_a_case(wb):
    before = wb.spec
    wb.import_text("{not json", "junk.json")
    assert wb.spec == before
    assert "is not a workbench case" in wb.log_lines[0]


def test_a_url_cannot_open_a_file_outside_the_cases_folder(tmp_path):
    from atlas.workbench.app import case_from_url
    (tmp_path / "ok.json").write_text(example_case().to_json(), encoding="utf-8")
    assert case_from_url("ok.json", str(tmp_path)) == str(tmp_path / "ok.json")
    assert case_from_url("../ok.json", str(tmp_path)) == str(tmp_path / "ok.json")
    assert case_from_url("..\\..\\secret.json", str(tmp_path)) is None
    assert case_from_url("C:/Windows/win.ini", str(tmp_path)) is None
    assert case_from_url("missing.json", str(tmp_path)) is None


def test_one_light_theme_and_no_outside_request(wb):
    """The owner's 2026-09-28 screenshot: on a dark-mode PC the Fast template's
    components went dark while the page stayed light.  The shell now uses the
    Bootstrap template, pinned to its light theme, with native controls pinned
    light too; and nothing it loads comes from outside the local server."""
    from panel.theme.base import DefaultTheme
    from panel.theme.bootstrap import Bootstrap
    from atlas.workbench.app import LIGHT_CSS
    assert isinstance(wb.tpl, pn.template.BootstrapTemplate)
    assert wb.tpl.design is Bootstrap and wb.tpl.theme is DefaultTheme
    assert "color-scheme: light" in LIGHT_CSS and LIGHT_CSS in wb.tpl.config.raw_css
    res = str(wb.tpl.resolve_resources())
    assert "googleapis" not in res and "cdn." not in res
