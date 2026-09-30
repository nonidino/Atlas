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
    assert json.loads(open(path, encoding="utf-8").read())["schema_id"] == \
        "atlas-workbench/case@0.5"          # any family's version, drawn shapes or not


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


ALL_ACTIONS = ["file:new", "file:gallery", *[f"file:example:{k}" for k in EXAMPLES], "file:open",
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
    items = [it[1] for m in (wb.case_menu, wb.help_menu) for it in m.items if it is not None]
    assert {"file:new", "file:gallery", "file:open", "file:save", "file:save-as",
            "file:import", "file:export", "file:import-gmsh", "edit:revert",
            "view:history", "help:guide", "help:about", "help:docs"} <= set(items)
    for action in items:
        wb.dispatch(action)
    for button in (wb.name_btn, wb.status_btn, wb.undo_btn, wb.redo_btn, wb.run_btn):
        click(button)
    assert not any("unknown action" in line for line in wb.log_lines)


def test_the_examples_are_in_the_gallery_not_the_menu(wb):
    """Listed in the Case menu as well, the examples ran it off the page (seen)."""
    items = [it[1] for it in wb.case_menu.items if it is not None]
    assert not any(a.startswith("file:example:") for a in items)
    wb.dispatch("file:gallery")
    assert len(widgets(wb.modal_body, pn.widgets.Button, name="Open")) == len(EXAMPLES)


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
    wb.dispatch("file:new")                        # a blank case: no windows
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
    assert wb.edit(lambda s: setattr(s, "name", "renamed"), "rename") is True
    assert wb.dirty and wb.spec.name == "renamed"
    wb.dispatch("edit:undo")
    assert wb.spec.name == "wake-array-3"
    wb.dispatch("edit:redo")
    assert wb.spec.name == "renamed"


def test_save_open_and_the_case_bar(wb, tmp_path):
    path = str(tmp_path / "mine.json")
    assert wb.save_state.object == "not saved yet"
    wb.edit(lambda s: setattr(s, "name", "mine"), "rename")
    assert wb.save_state.object == "unsaved changes" and wb.name_btn.name.startswith("mine")
    wb.save_to(path)
    assert not wb.dirty and wb.save_state.object == "saved"
    wb.dispatch("file:example:farm-12")
    assert wb.spec.name == "farm-12" and wb.path is None
    # a fresh example has nothing of the person's to lose (seen: it read "unsaved
    # changes" the moment it opened)
    assert wb.save_state.object == "not saved yet"
    wb.open_path(path)
    assert wb.spec.name == "mine" and wb.path == path and not wb.dirty


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
