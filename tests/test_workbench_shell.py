"""The Atlas Workbench shell: the case file, the registry, and every action on a case.

The actions are driven as the studio's page sends them (`atlas/studio/protocol.py`),
one message at a time through `protocol.handle` on a `Session`, so these tests
exercise the code a browser does without one; the retired Panel page's menus,
header and dialogs went with it (2026-11-18).  What they cannot see is whether the
page draws it: that is checked by driving the real page (`scripts/studio_drive.py`),
the project's own rule from the RaceLab control that did nothing.  The page's own
files (served from this machine, nothing fetched from outside) are checked in
``test_studio_shell.py``.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from atlas.workbench import registry                                    # noqa: E402
from atlas.workbench.spec import (EXAMPLES, FAST_EXAMPLES, CaseSpec,     # noqa: E402
                                  Window, blank_case, check, example_case)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import studio_driver as D                                               # noqa: E402


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
# the shell, as the studio's page drives it
# ---------------------------------------------------------------------------


@pytest.fixture()
def s(tmp_path, monkeypatch):
    """A page's session whose Run creates the run but does not march it: the tests
    here are about the actions, and a real march would be a 30-second background
    thread timing itself against the rest of the suite (runs are tested in
    `test_workbench_runner.py`)."""
    D.idle(monkeypatch)
    return D.session(tmp_path)


def test_every_action_on_a_case_runs(s, tmp_path):
    """Every action the retired page's menus offered, as the message the studio sends
    for it (new, start over, every example, the Fast example, open, save, import,
    export, undo, redo, revert, history, check, compile, the three runs, Stop): each
    is handled and applied.  A control nobody handles would be a control that does
    nothing; that every message the page sends has a handler is checked in
    ``test_studio_shell.py``."""
    (tmp_path / "saved.json").write_text(example_case("wall-2").to_json(), encoding="utf-8")
    messages = [("case.start_over", {}), *[("case.example", {"key": k}) for k in EXAMPLES],
                ("case.new", {"family": "conduction-2d"}), ("case.fast", {}),
                ("case.files", {}), ("case.open", {"name": "saved"}),
                ("case.save", {"name": "mine"}), ("case.save", {}),
                ("case.import", {"text": example_case("bend-3").to_json(), "name": "b.json"}),
                ("case.export", {}), ("edit.undo", {}), ("edit.redo", {}),
                ("case.revert", {}), ("case.history", {}), ("geometry.check", {}),
                ("compile.start", {}), ("run.start", {"arms": ["serial", "parallel"]}),
                ("run.start", {"arms": ["full"]}), ("run.start", {}), ("run.stop", {}),
                ("state", {}), ("case.new", {"family": "electric-2d"})]
    for kind, args in messages:
        D.ok(s, kind, **args)
    assert s.spec.physics.family == "electric-2d"
    # a case has no name to rename (case@0.6), and the gallery of every example left
    # the page (the owner's O1, 2026-09-30)
    from atlas.studio import protocol
    assert not {"case.rename", "case.gallery"} & set(protocol.HANDLERS)


def test_every_example_is_reached_from_its_kind_by_what_it_shows(s):
    """The examples the page lists come from the catalog it is sent once: each under
    its kind with the one line saying what it shows, its kind's Fast example marked,
    the learned case marked read-only, and no label or description (case@0.6 has
    none).  Every example is within reach: the Fast example by ``case.fast``, every
    one by its key; opening one opens the case it names with nothing of the person's
    to lose, and one Undo brings back the case before it."""
    cat = s.catalog()
    assert sorted(e["key"] for e in cat["examples"]) == sorted(EXAMPLES)
    for e in cat["examples"]:
        assert set(e) == {"key", "family", "shows", "style", "fast", "fixed"}, e
        assert e["shows"] == EXAMPLES[e["key"]].shows
        assert e["fast"] == (FAST_EXAMPLES.get(e["family"]) == e["key"])
    from atlas.workbench import learned_case
    assert [e["key"] for e in cat["examples"] if e["fixed"]] == [learned_case.KEY]
    reached = set()
    for fid in registry.available_ids():
        D.ok(s, "case.new", family=fid)
        mine = [e["key"] for e in cat["examples"] if e["family"] == fid]
        assert mine, fid
        reached |= set(mine)
        if fid in FAST_EXAMPLES:
            D.ok(s, "case.fast")
            assert s.key == FAST_EXAMPLES[fid] and s.spec.physics.family == fid
        else:
            assert D.refused(s, "case.fast")[0].startswith("there is no fast example")
    assert reached == set(EXAMPLES)
    D.ok(s, "case.new", family="conduction-2d")
    before = s.spec.model_dump()
    D.ok(s, "case.example", key="bend-3")
    assert s.spec == example_case("bend-3") and s.key == "bend-3"
    meta = s.state()["meta"]
    assert meta["file"] is None and meta["dirty"] is False
    D.ok(s, "edit.undo")
    assert s.spec.model_dump() == before
    D.ok(s, "case.new", family="conjugate-heat-2d")
    D.ok(s, "case.example", key="cooled-winding")
    assert s.spec == example_case("cooled-winding")


def test_a_new_kind_starts_a_new_case_and_one_undo_restores_the_old(s, tmp_path):
    """D1-D3 of the demo plan: another kind starts a new case of it (the geometry
    resets), with no confirmation, and one Undo brings the old case back, bit for bit,
    with its file; Start over is a new case of the same kind."""
    from atlas.workbench.starter import new_case
    D.ok(s, "case.new", family="conduction-2d")
    r, _ = D.draw_domain(s, [(40, 40), (200, 60), (180, 200)])
    assert r["ok"], r
    D.ok(s, "case.save", name="mine")
    before, path = s.spec.model_dump(), s.path
    assert s.spec.domain.outline is not None and s.key == "mine"
    said = D.notices(D.ok(s, "case.new", family="transport-2d"))
    assert s.spec.physics.family == "transport-2d" and s.spec.domain.outline is None
    assert s.spec.model_dump() == new_case("transport-2d").model_dump()
    assert s.path is None and s.key.startswith("river-plume-")
    assert any("Undo" in n and "heat conduction" in n for n in said), said
    D.ok(s, "edit.undo")                                  # one Undo
    assert s.spec.model_dump() == before
    assert s.path == path and s.key == "mine" and not s.dirty
    assert s.state()["meta"]["family"] == "conduction-2d"  # the page's kind follows
    D.ok(s, "edit.redo")
    assert s.spec.physics.family == "transport-2d" and s.path is None
    said = D.notices(D.ok(s, "case.start_over"))          # a new case of the same kind
    assert s.spec.physics.family == "transport-2d" and s.path is None
    assert any(n.startswith("Started over") for n in said), said


def test_no_example_label_or_description_reaches_the_page(s):
    """Done-when of item 1.1, as far as the server holds it: what the page is told of
    an open example (its state) holds none of the example labels or descriptions.  The
    studio's header names the case by its file, or by its key until it is saved."""
    for key in EXAMPLES:
        D.ok(s, "case.example", key=key)
        shown = json.dumps(s.state())
        for ex in EXAMPLES.values():
            assert ex.label not in shown and ex.description not in shown, (key, ex.key)
        assert s.state()["meta"]["key"] == key


def test_a_compile_starts_from_its_message(s):
    """Every action that was once *not built* is built; the compile was the last."""
    D.ok(s, "compile.start")
    assert s.compile_job is not None               # started (the fixture does not march it)
    assert any("compile started" in line for line in s.log_lines[:3])


def test_a_case_with_errors_does_not_run(s):
    """Run on a case that cannot run says why, and the problems list says where each
    is fixed."""
    D.ok(s, "case.import", text=blank_case().to_json(), name="blank.json")   # no windows
    why, pushes = D.refused(s, "run.start")
    assert s.run is None and "cannot run" in why
    assert any("cannot run" in n for n in D.notices(pushes, "error"))
    rows = s.state()["issues"]
    assert rows and all(r["where"] in ("geometry", "physics", "pieces") for r in rows)
    assert s.state()["status"]["kind"] == "err"


def test_edits_are_validated_undone_and_redone(s):
    r, pushes = D.edit(s, D.set_op(["domain", "nx"], -5), label="bad")
    assert r["ok"] is False and s.spec.domain.nx == 352
    assert any("not applied" in n for n in D.notices(pushes, "error"))
    r, _ = D.edit(s, D.set_op(["run", "steps"], 7), label="steps")
    assert r["ok"] and s.dirty and s.spec.run.steps == 7
    D.ok(s, "edit.undo")
    assert s.spec.run.steps == 40
    D.ok(s, "edit.redo")
    assert s.spec.run.steps == 7


def test_save_open_and_the_case_bar(s, tmp_path):
    """The header's file state: not saved yet, unsaved changes, saved."""
    def meta():
        return s.state()["meta"]
    assert meta()["file"] is None and meta()["dirty"] is False         # not saved yet
    D.edit(s, D.set_op(["run", "steps"], 7), label="steps")
    assert meta()["dirty"] is True                                      # unsaved changes
    D.ok(s, "case.save", name="mine")
    assert meta()["file"] == "mine.json" and meta()["dirty"] is False and s.key == "mine"
    D.ok(s, "case.example", key="farm-12")
    assert s.spec == example_case("farm-12") and s.path is None
    assert s.key == "farm-12"                        # its records' folder
    # a fresh example has nothing of the person's to lose (seen: it read "unsaved
    # changes" the moment it opened)
    assert meta()["dirty"] is False and meta()["file"] is None
    D.ok(s, "case.open", name="mine")
    assert s.spec.run.steps == 7 and s.path == str(tmp_path / "mine.json") and not s.dirty
    assert s.key == "mine"


def test_the_status_chip_lists_the_problems_and_where_each_is_fixed(s):
    """The six steps' error counts became one chip in the header: it says whether the
    case can run, and its list sends each problem to the tab that fixes it."""
    assert s.state()["status"] == {"kind": "ok", "text": "Ready",
                                   "tip": "Everything needed to run is set"}
    D.ok(s, "case.import", text=blank_case().to_json(), name="blank.json")
    st = s.state()
    assert st["status"]["kind"] == "err" and "problem" in st["status"]["text"]
    issues = check(s.spec)
    assert len(st["issues"]) == len(issues)
    k = next(i for i, x in enumerate(issues) if "not cut into any windows" in x.message)
    assert st["issues"][k]["where"] == "pieces" and st["issues"][k]["fix"] == "windows"
    assert st["issues"][k]["fix_label"]


def test_import_refuses_what_is_not_a_case(s):
    before = s.spec
    why, _ = D.refused(s, "case.import", text="{not json", name="junk.json")
    assert s.spec == before
    assert "junk.json is not an Atlas case" in why and "is not an Atlas case" in s.log_lines[0]


def test_a_name_cannot_open_a_file_outside_the_cases_folder(s, tmp_path):
    (tmp_path / "ok.json").write_text(example_case().to_json(), encoding="utf-8")
    D.ok(s, "case.open", name="ok.json")
    assert s.path == str(tmp_path / "ok.json")
    for name in ("../ok.json", "..\\..\\secret.json", "C:/Windows/win.ini", "missing.json"):
        D.refused(s, "case.open", name=name)
        assert s.path == str(tmp_path / "ok.json"), name
