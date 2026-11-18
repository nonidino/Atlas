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

from atlas.workbench import registry, runner                             # noqa: E402
from atlas.workbench.families import conduction as cd                    # noqa: E402
from atlas.workbench.families import electric as el                      # noqa: E402
from atlas.workbench.spec import (Attachment, CaseSpec, Window,           # noqa: E402
                                  adapt_to_family, blank_case, check, example_case)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import studio_driver as D                                               # noqa: E402


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
        "balance": 1e-6, "reference": 1e-6, "closed_form": 1e-6, "bitwise": None,
        # style M's own agreement, registered before its first run (demo item 1.4)
        "reference_multirate": 1e-3}
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
    assert s.schema_id == "atlas-workbench/case@0.6"
    assert "old" not in s.to_json()                       # its name is dropped (0.6)
    assert s.physics.params == {"nu": 0.004, "u_inf": 1.0} and s.physics.nu == 0.004
    assert s.coupling.style == "A" and s.run.mode == "transient"
    assert s.materials == {} and s.attachments == []
    d.pop("boundaries")
    d["schema_id"] = "atlas-workbench/case@0.1"
    s1 = CaseSpec.from_json(__import__("json").dumps(d))
    assert s1.schema_id == "atlas-workbench/case@0.6" and len(s1.boundaries) == 4


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
def s(tmp_path, monkeypatch):
    """A page's session (the studio's protocol); its runs march inside the message
    that starts them."""
    D.synchronous(monkeypatch)
    return D.session(tmp_path)


@pytest.mark.parametrize("key", ["wall-2", "plate-insert", "plate-circuit"])
def test_the_page_is_sent_what_it_draws_for_the_new_families(s, key):
    """Every layer the kind reads reaches the page: its regions with their materials,
    its windows, its boundaries, edge for edge, in the physics' unit."""
    D.ok(s, "case.example", key=key)
    st = s.state()
    d = st["drawing"]
    assert [w["id"] for w in d["windows"]] == [w.id for w in s.spec.windows]
    assert all(w["rings"] for w in d["windows"])
    assert [r["material"] for r in d["regions"]] == [r.material for r in s.spec.regions]
    assert [b["edge"] for b in d["boundaries"]] == [b.edge for b in s.spec.boundaries]
    assert d["width"] == pytest.approx(s.spec.domain.nx * s.spec.domain.dx)
    fam = D.family(s, s.spec.physics.family)
    assert {"regions", "windows", "boundaries"} <= set(fam["layers"])


def test_the_physics_is_the_familys(s):
    """The page's physics screen is built from the catalog: each kind's own parameters
    and material properties, with their units; the wind farm has no materials."""
    heat = D.family(s, "conduction-2d")
    assert [(p["label"], p["unit"]) for p in heat["params"]] == [("Initial temperature", "K")]
    assert {p["label"] for p in heat["material_props"]} == {"Conductivity", "Density",
                                                           "Specific heat"}
    assert set(heat["styles"]) == {"B", "C", "M"}
    farm = D.family(s, "incompressible-2d")
    assert ("Viscosity nu", "D U") in [(p["label"], p["unit"]) for p in farm["params"]]
    assert farm["material_props"] == [] and farm["materials"] == {}


def test_only_the_families_that_run_are_offered(s):
    """The catalog marks the kinds that run; choosing one starts a new case of it
    (`starter.new_case`), and a planned kind is refused.  One Undo brings the old case
    back, and the page's kind follows it."""
    cat = s.catalog()
    ready = [f.id for f in registry.FAMILIES if f.status == "ready-to-wire"]
    planned = [f.id for f in registry.FAMILIES if f.status != "ready-to-wire"]
    assert [f["id"] for f in cat["families"] if f["runs"]] == ready
    assert not {f["id"] for f in cat["families"] if f["runs"]} & set(planned)
    for fid in planned:
        assert "runs here" in D.refused(s, "case.new", family=fid)[0]
    D.ok(s, "case.new", family="conduction-2d")
    assert s.spec.physics.family == "conduction-2d" and s.spec.coupling.style in ("B", "C")
    D.ok(s, "edit.undo")
    assert s.spec.physics.family == "incompressible-2d"
    assert s.state()["meta"]["family"] == "incompressible-2d"


def test_materials_are_edited_as_data(s):
    """A material's property is an edit to the case (undoable); a value the family
    refuses is the check's error; a material from the library, or a new one with the
    family's defaults, is added by name (as the page adds one)."""
    D.ok(s, "case.example", key="wall-2")
    assert s.spec.materials["copper"]["k"] == 400.0
    r, _ = D.edit(s, D.set_op(["materials", "copper", "k"], 390.0), label="copper: k = 390")
    assert r["ok"] and s.spec.materials["copper"]["k"] == 390.0
    D.ok(s, "edit.undo")
    assert s.spec.materials["copper"]["k"] == 400.0
    r, _ = D.edit(s, D.set_op(["materials", "steel", "k"], -5.0), label="steel: k = -5")
    assert any("steel" in m for m in D.errors(s))
    D.ok(s, "edit.undo")
    assert D.errors(s) == []
    lib = D.family(s, "conduction-2d")["materials"]
    r, _ = D.edit(s, D.set_op(["materials", "aluminium"], lib["aluminium"]), label="aluminium")
    assert r["ok"] and s.spec.materials["aluminium"]["k"] == 205.0
    fam = registry.family("conduction-2d")
    new = {p.name: p.default for p in fam.material_props}
    r, _ = D.edit(s, D.set_op(["materials", "my-alloy"], new), label="my-alloy")
    assert r["ok"] and s.spec.materials["my-alloy"] == new


def test_a_style_c_case_offers_no_threaded_arm(s):
    D.ok(s, "case.example", key="wall-2")
    r, _ = D.edit(s, D.set_op(["run", "steps"], 2), label="steps = 2")
    reply, _ = D.send(s, "run.start", arms=["serial", "parallel", "full"])
    assert reply["ok"] and reply["arms"] == ["serial", "full"]
    assert "sequential by construction" in reply["dropped"]["parallel"]


@pytest.mark.parametrize("key, metric", [("wall-2", "heat_in"),
                                         ("plate-circuit", "current_A")])
def test_a_new_family_runs_through_the_page(s, key, metric):
    """A run through the page passes its checks, measures the family's own quantity in
    every arm, and sends the convergence series of the iterated coupling."""
    D.ok(s, "case.example", key=key)
    r, _ = D.edit(s, D.set_op(["run", "steps"], 2), label="steps = 2")
    D.ok(s, "run.start", arms=["serial", "parallel", "full"])
    run = s.run
    assert run.status == "done", run.error_trace
    frame = D.frames(s)["run"]
    assert frame["cards"]["failed"] == 0 and frame["cards"]["checks"]
    assert all(metric in run.results["metrics"][a] for a in run.arms)
    assert frame["convergence"] and frame["convergence"].get("serial"), frame["convergence"]


def test_switching_family_adapts_the_case():
    s = blank_case(100, 80)
    adapt_to_family(s, "conduction-2d")
    assert s.physics.family == "conduction-2d" and set(s.physics.params) == {"T0"}
    # a family changed starts from its first example's setup (the wall: style C,
    # steady, cells of 2.5 mm, two windows cut automatically)
    assert s.coupling.style == "C" and s.run.mode == "steady"
    assert s.domain.dx == 0.0025 and s.layout is not None and s.layout.along == 2
    assert [(b.edge, b.kind) for b in s.boundaries] == [
        ("left", "fixed-temperature"), ("right", "fixed-temperature"),
        ("bottom", "insulated"), ("top", "insulated")]
    adapt_to_family(s, "incompressible-2d")
    assert s.coupling.style == "A" and set(s.physics.params) == {"nu", "u_inf"}
