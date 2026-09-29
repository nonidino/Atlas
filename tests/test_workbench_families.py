"""The conduction and electric families, and the case file's schema 0.3.

Pinned here:

* **one window is the full domain, to the bit**, for conduction in style B,
  steady and transient (the family-level form of the N = 1 control);
* style C's two pieces converge to the full domain, and its arms say why there
  is no threaded one;
* a steady wall of layers is recognised and meets its closed form; a transient
  or a non-layered case is told it has none;
* style D converges to the plate and the circuit solved as one system.  Its
  partition is physical (a field against a lumped network), so its full-domain
  reference is different linear algebra on the same discretization, and the
  control is round-off, not bits: stated here rather than skipped;
* the circuit's nodal analysis carries an ideal battery as well as a Norton one;
* every family's registered tolerances, asserted against the code;
* schema 0.3: 0.1 and 0.2 files migrate (genuine fixtures, not relabelled 0.3
  files), and the style rules, the attachment rules and the family switch.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

pytest.importorskip("panel")

from atlas.workbench import registry, runner                             # noqa: E402
from atlas.workbench.families import conduction as cd                    # noqa: E402
from atlas.workbench.families import electric as el                      # noqa: E402
from atlas.workbench.spec import (Attachment, CaseSpec, Window,           # noqa: E402
                                  adapt_to_family, blank_case, check, example_case)


def _errors(spec):
    return [i.message for i in check(spec) if i.severity == "error"]


# ---------------------------------------------------------------------------
# conduction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["steady", "transient"])
def test_one_window_is_the_full_domain_to_the_bit(mode):
    s = example_case("plate-insert")
    s.windows = [Window(id="all", x0=0, y0=0, nx=s.domain.nx, ny=s.domain.ny)]
    s.run.mode = mode
    assert _errors(s) == []
    run = cd.build(s, arms=("serial", "full"))
    a, f = run.initial("serial"), run.initial("full")
    for _ in range(3):
        a, f = run.step("serial", a), run.step("full", f)
        assert np.array_equal(a.u, f.u)
    run.close()


def test_style_c_runs_without_a_threaded_arm_and_says_why():
    r = runner.CaseRun(example_case("wall-2"), arms=("serial", "parallel", "full"),
                       steps=2).run_blocking()
    assert r.status == "done", r.error_trace
    assert r.arms == ("serial", "full")
    assert "sequential by construction" in r.dropped["parallel"]
    checks = {c["key"]: c for c in r.results["checks"]}
    assert checks["balance"]["passed"] and checks["reference"]["passed"]
    assert checks["closed_form"]["passed"] and checks["bitwise"]["passed"] is None
    assert r.results["problem"]["dirichlet"] == "steel"          # the lower conductivity
    assert r.results["problem"]["rho_1d"] == pytest.approx(45 * 64 / (400 * 96))


def test_the_closed_form_is_the_layered_walls_only():
    wall = example_case("wall-2")
    q = cd.closed_form_heat(wall)
    assert q == pytest.approx(100.0 * 0.1 / (0.24 / 45.0 + 0.16 / 400.0), rel=1e-12)
    assert cd.closed_form_heat(example_case("plate-insert")) is None     # an insert
    wall.run.mode = "transient"
    assert cd.closed_form_heat(wall) is None


def test_style_b_transient_run_passes_its_checks_threaded_or_not():
    s = example_case("plate-insert")
    s.run.steps = 4
    r = runner.CaseRun(s, arms=("serial", "parallel", "full"), steps=4,
                       threads=2).run_blocking()
    assert r.status == "done", r.error_trace
    res = r.results
    got = {c["key"]: c["passed"] for c in res["checks"]}
    assert got == {"balance": True, "reference": True, "closed_form": None, "bitwise": True}
    assert res["problem"]["partition_of_unity"]["holds"]


# ---------------------------------------------------------------------------
# electric, style D
# ---------------------------------------------------------------------------


def test_style_d_meets_the_joint_system():
    s = example_case("plate-circuit")
    s.coupling.tolerance = 1e-13
    run = el.build(s)
    d, f = run.step("serial", run.initial("serial")), run.step("full", run.initial("full"))
    assert d.converged
    for k in f.currents:
        assert abs(d.currents[k] - f.currents[k]) <= 1e-12 * abs(f.currents["B1"])
    assert np.max(np.abs(d.phi - f.phi)) <= 1e-11 * 12.0
    # the loop's closed form: EMF over the series resistance, with the plate's own
    r_plate = abs(f.potentials["E1"] - f.potentials["E2"]) / abs(f.currents["B1"])
    assert abs(f.currents["B1"]) == pytest.approx(12.0 / (0.5 + 1.0 + r_plate), rel=1e-12)


def test_style_d_example_passes_its_checks():
    s = example_case("plate-circuit")
    r = runner.CaseRun(s, arms=("serial", "full"), steps=2).run_blocking()
    assert r.status == "done", r.error_trace
    assert all(c["passed"] for c in r.results["checks"]), r.results["checks"]
    assert any("over 1: unrelaxed it would diverge" in n for n in r.results["notes"])


def test_an_ideal_battery_is_a_voltage_source():
    s = example_case("plate-circuit")
    s.attachments[0].internal = 0.0
    run = el.build(s)
    f = run.step("full", run.initial("full"))
    d = run.step("serial", run.initial("serial"))
    # an ideal 12 V source: the loop's drop is all outside it
    drop = f.potentials["n1"] - f.potentials["E2"]
    assert drop == pytest.approx(12.0, rel=1e-12)
    assert abs(d.currents["B1"] - f.currents["B1"]) <= 1e-9 * abs(f.currents["B1"])
    assert run.kirchhoff(f) < 1e-10 and run.powers(f)["balance"] < 1e-10


# ---------------------------------------------------------------------------
# tolerances, registry, schema, rules
# ---------------------------------------------------------------------------


def test_the_registered_tolerances():
    assert {c.key: c.tolerance for c in cd.CHECKS} == {
        "balance": 1e-6, "reference": 1e-6, "closed_form": 1e-6, "bitwise": None}
    assert {c.key: c.tolerance for c in el.CHECKS} == {
        "kirchhoff": 1e-6, "energy": 1e-6, "reference": 1e-6}
    for fam in ("conduction-2d", "electric-2d"):
        assert registry.family(fam).adapter
        assert registry.family(fam).status == "ready-to-wire"


def test_a_version_0_2_file_migrates():
    """A 0.2 file as the 0.2 workbench wrote it: nu and u_inf as fields."""
    d = {"schema_id": "atlas-workbench/case@0.2", "name": "old",
         "domain": {"nx": 352, "ny": 240, "dx": 0.03125},
         "physics": {"family": "incompressible-2d", "nu": 0.004, "u_inf": 1.0},
         "windows": [{"id": "F00", "x0": 0, "y0": 0, "nx": 352, "ny": 240}],
         "boundaries": [{"id": "B-left", "edge": "left", "kind": "inlet"},
                        {"id": "B-right", "edge": "right", "kind": "outlet"},
                        {"id": "B-bottom", "edge": "bottom", "kind": "freestream"},
                        {"id": "B-top", "edge": "top", "kind": "freestream"}]}
    s = CaseSpec.model_validate(d)
    assert s.schema_id == "atlas-workbench/case@0.4"
    assert s.physics.params == {"nu": 0.004, "u_inf": 1.0} and s.physics.nu == 0.004
    assert s.coupling.style == "A" and s.run.mode == "transient"
    assert s.materials == {} and s.attachments == []
    d.pop("boundaries")
    d["schema_id"] = "atlas-workbench/case@0.1"
    s1 = CaseSpec.from_json(__import__("json").dumps(d))
    assert s1.schema_id == "atlas-workbench/case@0.4" and len(s1.boundaries) == 4


def test_style_c_and_d_geometry_rules():
    s = example_case("wall-2")
    s.windows[1].x0 -= 8                          # overlap the two pieces
    s.windows[1].nx += 8
    assert any("more than one window" in m for m in _errors(s))
    s = example_case("wall-2")
    s.windows.append(Window(id="third", x0=0, y0=0, nx=8, ny=8))
    assert any("exactly two pieces" in m for m in _errors(s))
    s = example_case("plate-circuit")
    s.windows = [Window(id="a", x0=0, y0=0, nx=80, ny=80),
                 Window(id="b", x0=80, y0=0, nx=80, ny=80)]
    assert any("one window covering the domain" in m for m in _errors(s))


def test_attachment_rules():
    s = example_case("plate-circuit")
    s.attachments = [a for a in s.attachments if a.kind != "battery"]
    assert any("no battery" in m for m in _errors(s))
    s = example_case("plate-circuit")
    s.attachments.append(Attachment(id="R9", kind="resistor", value=2.0, a="x", b="y"))
    assert any("not connected to the plate" in m for m in _errors(s))
    s = example_case("plate-circuit")
    s.boundaries = [b for b in s.boundaries if b.kind != "electrode"]
    assert any("no electrode" in m for m in _errors(s))


@pytest.fixture()
def wb(tmp_path):
    import panel as pn
    pn.extension("tabulator")
    from atlas.workbench.app import Workbench
    return Workbench(cases_dir=str(tmp_path))


def _texts(layout) -> str:
    """Every label, text, title and option a layout holds, walked through its
    children (a repr of a Panel layout shows neither widget names nor text)."""
    out: list[str] = []

    def walk(o):
        for attr in ("name", "object", "title"):
            v = getattr(o, attr, None)
            if isinstance(v, str):
                out.append(v)
        opts = getattr(o, "options", None)
        if isinstance(opts, dict):
            out.extend(str(k) for k in opts)
        for c in getattr(o, "objects", None) or []:
            walk(c)
    walk(layout)
    return "\n".join(out)


@pytest.mark.parametrize("key", ["wall-2", "plate-insert", "plate-circuit"])
def test_every_step_draws_for_the_new_families(wb, key):
    wb.dispatch(f"file:example:{key}")
    for step in ("case", "geometry", "physics", "check", "run", "results"):
        wb.show(step)
        assert wb.workspace.objects
    wb.show("geometry")
    lead = wb.geo_editor.summary.object
    if key in ("wall-2", "plate-circuit"):
        assert "tile the domain without overlapping" in lead
    else:
        assert "every cell has a window at full weight" in lead


def test_the_physics_step_is_the_familys(wb):
    wb.dispatch("file:example:wall-2")
    wb.show("physics")
    text = _texts(wb.workspace.objects[0])
    assert "Initial temperature (K)" in text and "Dirichlet side" in text
    assert "First relaxation factor" in text and "Materials" in text
    wb.dispatch("file:example:wake-array-3")
    wb.show("physics")
    text = _texts(wb.workspace.objects[0])
    assert "Viscosity nu (D U)" in text and "Pressure solve" in text
    assert "Materials" not in text


def test_planned_families_are_disabled_on_the_case_step(wb):
    """The disabled list must name the Select's own option values: in the page a
    dict-valued Select renders labels as values, and a list of family ids there
    disabled nothing (seen in the served page)."""
    import panel as pn
    wb.show("case")
    found = []

    def walk(o):
        if isinstance(o, pn.widgets.Select) and o.name == "Physics family":
            found.append(o)
        for c in getattr(o, "objects", None) or []:
            walk(c)
    walk(wb.workspace.objects[0])
    sel = found[0]
    planned = [f.label for f in registry.FAMILIES if f.status != "ready-to-wire"]
    assert sel.disabled_options == planned and set(planned) <= set(sel.options)
    assert sel.value == registry.family(wb.spec.physics.family).label


def test_the_materials_table_edits_the_case(wb):
    """A cell edited in the Materials table is an edit to the case (undoable), a
    value that is not a number is refused, and the library adds a material."""
    import panel as pn
    from types import SimpleNamespace
    wb.dispatch("file:example:wall-2")
    wb.show("physics")
    found = []

    def walk(o):
        if isinstance(o, pn.widgets.Tabulator):
            found.append(o)
        if isinstance(o, pn.widgets.Select) and o.name == "":
            found.append(o)
        for c in getattr(o, "objects", None) or []:
            walk(c)
    walk(wb.workspace.objects[0])
    table = next(o for o in found if isinstance(o, pn.widgets.Tabulator))
    before = wb.spec.materials["copper"]["k"]
    for cb in table._on_edit_callbacks:
        cb(SimpleNamespace(row=1, column="k (W/(m K))", value="390"))
    assert wb.spec.materials["copper"]["k"] == 390.0 and before == 400.0
    wb.undo()
    assert wb.spec.materials["copper"]["k"] == 400.0
    wb.show("physics")
    found.clear()
    walk(wb.workspace.objects[0])
    table = next(o for o in found if isinstance(o, pn.widgets.Tabulator))
    for cb in table._on_edit_callbacks:
        cb(SimpleNamespace(row=0, column="k (W/(m K))", value="lots"))
    assert wb.spec.materials["steel"]["k"] == 45.0 and "is not a number" in wb.log_lines[0]
    pick = next(o for o in found if isinstance(o, pn.widgets.Select))
    pick.value = "aluminium"
    assert wb.spec.materials["aluminium"]["k"] == 205.0


def test_a_style_c_case_offers_no_threaded_arm(wb):
    wb.dispatch("file:example:wall-2")
    wb.show("run")
    text = _texts(wb.workspace.objects[0])
    assert "sequential by construction" in text
    assert "Timed repeats" in text
    assert "Decomposed, parallel" not in text


@pytest.mark.parametrize("key, column", [("wall-2", "heat in (W per m of depth)"),
                                         ("plate-circuit", "circuit current (A)")])
def test_a_new_family_runs_through_the_page(wb, key, column):
    from atlas.workbench.runview import results_frames
    wb.dispatch(f"file:example:{key}")
    wb.spec.run.steps = 2
    run = wb.start_run(["serial", "parallel", "full"], blocking=True)
    assert run is not None and run.status == "done", run and run.error_trace
    table, checks = results_frames(run.results)
    assert column in table.columns
    assert "FAIL" not in set(checks["verdict"])
    assert wb.run_panel.conv_fig is not None           # the convergence curve is drawn
    assert wb.run_panel.conv_src["serial"].data["value"]


def test_switching_family_adapts_the_case():
    s = blank_case(100, 80)
    adapt_to_family(s, "conduction-2d")
    assert s.physics.family == "conduction-2d" and set(s.physics.params) == {"T0"}
    assert s.coupling.style == "B" and s.run.mode == "transient"     # both modes run
    assert [(b.edge, b.kind) for b in s.boundaries] == [
        ("left", "fixed-temperature"), ("right", "fixed-temperature"),
        ("bottom", "insulated"), ("top", "insulated")]
    adapt_to_family(s, "incompressible-2d")
    assert s.coupling.style == "A" and set(s.physics.params) == {"nu", "u_inf"}
