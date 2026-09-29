"""The showcase cases added in step C of the plan: one family each.

Pinned here, per family:

* **the plume (case 3, style A with an explicit step)**: one window is the full
  domain to the bit; the explicit limit the check computes is the assembled
  operator's own; a decomposed march that carries the plume across the seam
  agrees with the full domain to round-off and closes its mass balance, threaded
  equal to serial; the check refuses an unstable step and an outfall outside the
  river; the family's boundaries are fixed.
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

from atlas.workbench import fv, registry, runner                         # noqa: E402
from atlas.workbench.families import acoustics as ac                     # noqa: E402
from atlas.workbench.families import plume as pl                         # noqa: E402
from atlas.workbench.spec import Boundary, Window, check, example_case   # noqa: E402


def _errors(spec):
    return [i.message for i in check(spec) if i.severity == "error"]


# ---------------------------------------------------------------------------
# case 3: the plume
# ---------------------------------------------------------------------------


def test_plume_one_window_is_the_full_domain_to_the_bit():
    s = example_case("plume-2")
    s.windows = [Window(id="all", x0=0, y0=0, nx=s.domain.nx, ny=s.domain.ny)]
    assert _errors(s) == []
    run = pl.build(s, arms=("serial", "full"))
    a, f = run.initial("serial"), run.initial("full")
    for _ in range(50):
        a, f = run.step("serial", a), run.step("full", f)
    assert np.array_equal(a.u, f.u) and float(np.max(f.u)) > 0.0
    run.close()


def test_plume_explicit_limit_is_the_assembled_operators():
    s = example_case("plume-2")
    f, h = pl.field_from_case(s)
    diag = fv.assemble(f).A.diagonal()
    assert pl.explicit_limit(s) == float(np.min(h.ravel() * f.dx * f.dx / diag))
    # the deep reach's interior sets it: dx^2 / (4 D + u dx) = 25 / (8 + 2)
    assert pl.explicit_limit(s) == pytest.approx(2.5, rel=1e-12)


def test_plume_crosses_the_seam_and_the_arms_agree():
    """Marched until the plume is well inside the lower window's own cells, so the
    agreement is not the trivial one of an overlap the plume has not reached."""
    s = example_case("plume-2")
    steps = 1000                                    # the front passes x = 1320 m at ~775
    r = runner.CaseRun(s, arms=("serial", "parallel", "full"), steps=steps,
                       threads=2).run_blocking()
    assert r.status == "done", r.error_trace
    checks = {c["key"]: c for c in r.results["checks"]}
    assert checks["mass"]["passed"] and checks["reference"]["passed"]
    assert checks["bitwise"]["passed"] is True
    full = r.results["metrics"]["full"]
    assert full["mass_g"] == pytest.approx(10.0 * 2.0 * steps, rel=1e-9)   # nothing left yet
    # the run's states stay inside it: march the full domain again to see where
    # the plume is
    run = pl.build(s, arms=("serial", "full"))
    u = run.initial("full")
    for _ in range(steps):
        u = run.step("full", u)
    field = run.field(u)
    lower_only = field[:, 264:]                     # past the upper window's last cell
    assert float(lower_only.max()) > 0.01 * float(field.max())
    run.close()


def test_plume_check_refuses_an_unstable_step_and_a_lost_outfall():
    s = example_case("plume-2")
    s.run.macro_dt = 3.0
    assert any("stability limit here, 2.5 s" in m for m in _errors(s))
    with pytest.raises(ValueError, match="over the explicit step's stability limit"):
        pl.build(s)
    s = example_case("plume-2")
    s.physics.params["source_x"] = 2500.0
    assert any("is not in the river" in m for m in _errors(s))


# ---------------------------------------------------------------------------
# case 4: sound through two media
# ---------------------------------------------------------------------------


def test_sound_pieces_are_the_full_domain_and_reflect_the_textbook_R():
    r = runner.CaseRun(example_case("sound-air-water"), arms=("serial", "parallel", "full"),
                       steps=1100).run_blocking()           # the plateau starts at 1021
    assert r.status == "done", r.error_trace
    checks = {c["key"]: c for c in r.results["checks"]}
    assert all(c["passed"] for c in checks.values()), checks
    z1, z2 = 1.2 * 343.0, 1000.0 * 1480.0
    assert checks["closed_form"]["extra"]["textbook"] == pytest.approx((z2 - z1) / (z2 + z1), rel=1e-14)
    assert r.results["metrics"]["serial"]["reflection_vs_textbook"] == pytest.approx(0.0,
                                                                                     abs=1e-9)
    assert "parallel" in r.dropped


def test_sound_with_no_interface_reflects_nothing():
    """The positive control: air on both sides of the cut, so R = 0 and the pulse's
    integral must leave the first piece entirely."""
    s = example_case("sound-air-water")
    s.regions[1].material = "air"
    s.materials.pop("water")
    first, _last = ac.plateau(s, s.run.macro_dt)
    r = runner.CaseRun(s, arms=("serial", "full"), steps=first + 20).run_blocking()
    assert r.status == "done", r.error_trace
    checks = {c["key"]: c for c in r.results["checks"]}
    assert checks["closed_form"]["extra"]["textbook"] == 0.0 and checks["closed_form"]["passed"]
    assert abs(r.results["metrics"]["serial"]["reflection"]) < 1e-9
    assert checks["bitwise"]["passed"] and checks["energy"]["passed"]


def test_sound_stability_limit_and_rules():
    s = example_case("sound-air-water")
    s.regions[0].material = "water"
    s.materials = {"water": {"rho": 1000.0, "c": 1480.0}}
    assert ac.stable_dt(s) == pytest.approx(0.01 / (1480.0 * np.sqrt(2.0)), rel=1e-12)
    s = example_case("sound-air-water")
    lim = ac.stable_dt(s)
    assert lim < 0.01 / (1480.0 * np.sqrt(2.0))          # the interface face tightens it
    s.run.macro_dt = 1.01 * lim
    assert any("over the leapfrog's stability limit" in m for m in _errors(s))
    s = example_case("sound-air-water")
    s.windows = [Window(id="low", x0=0, y0=0, nx=740, ny=20),
                 Window(id="high", x0=0, y0=20, nx=740, ny=20)]
    assert any("side by side, each the full height" in m for m in _errors(s))
    s = example_case("sound-air-water")
    s.physics.params["pulse_x"] = 3.0
    assert any("outside the first piece" in m for m in _errors(s))
    s = example_case("sound-air-water")
    s.physics.params["pulse_x"] = 0.5                    # 5 widths from the wall
    assert any("will not be read against the textbook" in i.message for i in check(s))


# ---------------------------------------------------------------------------
# case 6: the bracket, and fe.py against the build repo's ThermoStruct2D
# ---------------------------------------------------------------------------


def _build_repo_solver():
    try:
        from atlas.cases import thermal_strain as ts_case
        return ts_case.load_solvers()
    except Exception as exc:                               # pragma: no cover
        pytest.skip(f"the build repo's ThermoStruct2D is not available: {exc}")


def test_fe_is_thermostruct2d_on_one_material():
    """On one material the stiffness, conduction and mass matrices are the build
    repo's own, and so is the thermal load (compared through a clamped solve)."""
    from atlas.workbench import fe
    TS = _build_repo_solver()
    nx, ny, h = 12, 5, 0.01
    nodes = np.stack(np.meshgrid(np.arange(nx + 1) * h, np.arange(ny + 1) * h,
                                 indexing="ij"), -1)
    mat = TS.SolidMaterial()
    ts = TS.ThermoStruct2D(TS.ShellMesh(nodes), mat)
    g = fe.QuadGrid(nx, ny, h)
    ii, jj = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing="ij")
    ours = g.node(ii.ravel(), jj.ravel())                  # our id of each TS node
    ne = nx * ny
    E, nu = np.full(ne, mat.E), np.full(ne, mat.nu)
    K = fe.assemble_elastic(g, E, nu).toarray()
    dofs = np.stack([2 * ours, 2 * ours + 1], axis=1).ravel()
    Kts = ts.K_me.toarray()
    assert np.max(np.abs(K[np.ix_(dofs, dofs)] - Kts)) <= 1e-12 * np.max(np.abs(Kts))
    Kth, Mth = fe.assemble_thermal(g, np.full(ne, mat.k), np.full(ne, mat.rho * mat.cp))
    for mine, theirs in ((Kth, ts.K_th), (Mth, ts.M_th)):
        a, b = mine.toarray()[np.ix_(ours, ours)], theirs.toarray()
        assert np.max(np.abs(a - b)) <= 1e-12 * np.max(np.abs(b))
    # the thermal load: a clamped solve under a temperature field, both ways
    T = 300.0 + 50.0 * np.sin(np.arange(g.n_nodes) * 0.37)
    T_ts = T[ours]
    clamp = np.arange(ny + 1)                              # TS nodes (0, j): the i = 0 edge
    u_ts, _s = ts.solve_mechanical(T_ts, 0.0, 0.0, T_ref=288.15, clamp_nodes=clamp)
    G = fe.thermal_load_matrix(g, E, nu, np.full(ne, mat.alpha))
    f = G @ (T - 288.15)
    fixed = np.concatenate([2 * ours[clamp], 2 * ours[clamp] + 1])
    free = np.setdiff1d(np.arange(2 * g.n_nodes), fixed)
    import scipy.sparse.linalg as spla
    Ks = fe.assemble_elastic(g, E, nu)
    u = np.zeros(2 * g.n_nodes)
    u[free] = spla.spsolve(Ks[free][:, free].tocsc(), f[free])
    mine = u.reshape(-1, 2)[ours]
    assert np.max(np.abs(mine - u_ts)) <= 1e-10 * np.max(np.abs(u_ts))


def test_bracket_one_window_is_the_full_domain_to_the_bit():
    from atlas.workbench.families import elasticity as el_
    s = example_case("bracket-2")
    s.windows = [Window(id="all", x0=0, y0=0, nx=s.domain.nx, ny=s.domain.ny)]
    assert _errors(s) == []
    run = el_.build(s, arms=("serial", "full"))
    a = run.step("serial", run.initial("serial"))
    f = run.step("full", run.initial("full"))
    assert a.converged and a.iterations == 2 and np.array_equal(a.u, f.u)
    run.close()


def test_bracket_passes_its_checks_and_balances_its_load():
    s = example_case("bracket-2")
    r = runner.CaseRun(s, arms=("serial", "parallel", "full"), steps=1,
                       threads=2).run_blocking()
    assert r.status == "done", r.error_trace
    checks = {c["key"]: c for c in r.results["checks"]}
    assert all(c["passed"] for c in checks.values()), checks
    full = r.results["metrics"]["full"]
    assert full["load_N"] == pytest.approx(5.0e6 * 0.1 * 0.01, rel=1e-12)     # 5 kN
    assert r.results["metrics"]["parallel"]["all_converged"]


def test_bracket_rules():
    s = example_case("bracket-2")
    s.boundaries = [b for b in s.boundaries if b.kind != "clamped"]
    from atlas.workbench.spec import Boundary as B_
    s.boundaries.append(B_(id="B-left", edge="left", kind="free"))
    assert any("nothing holds the plate" in m for m in _errors(s))
    from atlas.workbench.families import elasticity as el_
    assert {c.key: c.tolerance for c in el_.CHECKS} == {
        "force": 1e-6, "reference": 1e-6, "bitwise": None}
    assert example_case("bracket-2").coupling.tolerance == 1e-12   # what "force" assumes


# ---------------------------------------------------------------------------
# case 7: the heated strip, split by physics
# ---------------------------------------------------------------------------


def test_strip_splits_are_exact_and_the_lag_is_one_step():
    s = example_case("heated-strip")
    r = runner.CaseRun(s, arms=("serial", "parallel", "full"), steps=8,
                       threads=2).run_blocking()
    assert r.status == "done", r.error_trace
    checks = {c["key"]: c for c in r.results["checks"]}
    assert all(c["passed"] for c in checks.values()), checks
    assert r.results["arm_labels"]["full"] == "Unsplit solver"
    assert r.results["metrics"]["parallel"]["lag_stress_error"] > 0.0   # it IS late


def test_strip_uniform_heating_bends_a_bimetal_and_not_a_plain_plate():
    """The positive control for the bond: heated uniformly, one material expands
    freely with no stress, and two materials cannot, so they are stressed."""
    from atlas.workbench.families import thermoelastic as te
    for mats, stressed in ((("steel", "steel"), False), (("steel", "copper"), True)):
        s = example_case("heated-strip")
        s.regions[0].material, s.regions[1].material = mats
        run = te.build(s, arms=("full",))
        st = run.initial("full")
        T = np.full(st.T.size, 400.0)
        st = te.ThermoState(T, run._mech(T), T)
        peak = float(np.max(run.field(st)))
        assert (peak > 1.0) if stressed else (peak < 1e-6), (mats, peak)
        run.close()


def test_strip_panels_name_the_split_shown(wb):
    """Seen in the page: the split's panels said "Decomposed". They name the arm
    they show: the lagged split when it runs, else the synchronous one."""
    wb.dispatch("file:example:heated-strip")
    wb.spec.run.steps = 2
    run = wb.start_run(["serial", "parallel", "full"], blocking=True)
    assert run is not None and run.status == "done", run and run.error_trace
    figs = wb.run_panel.figs
    assert figs["decomposed"].title.text == "Lagged split: von Mises stress (MPa)"
    assert figs["full"].title.text == "Unsplit: von Mises stress (MPa)"
    assert figs["difference"].title.text.startswith("Lagged minus unsplit")
    run = wb.start_run(["serial", "full"], blocking=True)
    assert wb.run_panel.figs["decomposed"].title.text.startswith("Synchronous split")
    assert wb.run_panel.figs["difference"].title.text.startswith(
        "Synchronous minus unsplit (max |d| 0)")


def test_strip_rules():
    s = example_case("heated-strip")
    s.windows = [Window(id="a", x0=0, y0=0, nx=100, ny=20),
                 Window(id="b", x0=100, y0=0, nx=100, ny=20)]
    assert any("one window covering the domain" in m for m in _errors(s))
    from atlas.workbench.families import thermoelastic as te
    assert {c.key: c.tolerance for c in te.CHECKS} == {
        "energy": 1e-9, "reference": None, "lag": None}


# ---------------------------------------------------------------------------
# case 8: the cooled block, style C at a seam between two physics
# ---------------------------------------------------------------------------


def test_block_every_watt_leaves_in_the_coolant():
    s = example_case("cooled-block")
    r = runner.CaseRun(s, arms=("serial", "parallel", "full"), steps=1).run_blocking()
    assert r.status == "done", r.error_trace
    checks = {c["key"]: c for c in r.results["checks"]}
    assert all(c["passed"] for c in checks.values()), checks
    m = r.results["metrics"]
    assert m["full"]["power_W"] == pytest.approx(1.0e7 * 0.040 * 0.005, rel=1e-12)
    # the closed form of the balance: the bulk rise is P / (mdot cp)
    mdot_cp = 1000.0 * 4180.0 * 0.01 * 0.010
    assert m["full"]["bulk_rise_K"] == pytest.approx(2000.0 / mdot_cp, rel=1e-9)
    assert m["serial"]["all_converged"] and r.results["problem"]["dirichlet"] == "block"
    assert "sequential by construction" in r.dropped["parallel"]


def test_a_floating_piece_takes_the_dirichlet_side():
    """The block is insulated all round: as the Neumann side it would float. The
    first run of this example made it so, on the conductivity rule alone, and ran
    out of iterations; auto now picks it for the Dirichlet side, and an explicit
    choice that floats it is refused."""
    from atlas.workbench.families import conduction as cd
    from atlas.workbench.families import cooling as co
    s = example_case("cooled-block")
    f = co.field_from_case(s)
    assert cd.floating(f, cd.window_cells(200, (0, 0, 200, 50)))          # the block
    assert not cd.floating(f, cd.window_cells(200, (0, 50, 200, 10)))     # the channel
    assert cd.dirichlet_side(s, f) == ("block", "channel")
    s.coupling.dirichlet_side = "channel"
    assert any("the Neumann side, block, floats" in m for m in _errors(s))
    s = example_case("wall-2")                     # held at both ends: neither floats
    assert _errors(s) == []


def test_block_geometry_rules():
    s = example_case("cooled-block")
    s.regions[0].ny = 60                           # copper under the whole domain ...
    s.regions[2].nx = 150                          # ... so the channel ends in copper
    assert any("must fill whole rows" in m for m in _errors(s))
    s = example_case("cooled-block")
    s.boundaries[1].start = 40                     # the inlet over block rows too
    s.boundaries[0].stop = 40
    assert any("covers rows the coolant does not run in" in m for m in _errors(s))
    s = example_case("cooled-block")
    s.windows = [Window(id="upstream", x0=0, y0=0, nx=100, ny=60),
                 Window(id="downstream", x0=100, y0=0, nx=100, ny=60)]
    assert any("cuts the coolant's flow" in m for m in _errors(s))
    from atlas.workbench.families import cooling as co
    assert {c.key: c.tolerance for c in co.CHECKS} == {"energy": 1e-6, "reference": 1e-6}


@pytest.fixture()
def wb(tmp_path):
    import panel as pn
    pn.extension("tabulator")
    from atlas.workbench.app import Workbench
    return Workbench(cases_dir=str(tmp_path))


def _texts(layout) -> str:
    """Every label, text, title and option a layout holds (see
    test_workbench_families._texts)."""
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


def test_plume_in_the_page(wb):
    from atlas.workbench.runview import results_frames
    wb.dispatch("file:example:plume-2")
    for step in ("case", "geometry", "physics", "check", "run", "results"):
        wb.show(step)
        assert wb.workspace.objects
    wb.show("physics")
    text = _texts(wb.workspace.objects[0])
    assert "Discharge per metre of width (m^2/s)" in text
    assert "Reaches (the regions' materials)" in text and "Macro-step (s)" in text
    wb.spec.run.steps = 20
    run = wb.start_run(["serial", "parallel", "full"], blocking=True)
    assert run is not None and run.status == "done", run and run.error_trace
    table, checks = results_frames(run.results)
    assert "pollutant in the river (g)" in table.columns
    assert "FAIL" not in set(checks["verdict"])
    assert wb.run_panel.conv_fig is None                # nothing iterates in style A
    # drawn on a log scale three decades deep (display only), with no empty
    # iterations column for a step that does not iterate
    from bokeh.models import LogColorMapper
    from atlas.workbench.runview import timing_frame
    panel = wb.run_panel
    assert isinstance(panel.field_mapper, LogColorMapper)
    assert panel.field_mapper.low == pytest.approx(panel.field_mapper.high * 1e-3)
    assert panel.figs["full"].title.text == "Full domain: concentration (mg/L, log)"
    cols = timing_frame(run, run.progress()).columns
    assert not any("iterations" in c or "sub-steps" in c for c in cols)


def test_sound_in_the_page(wb):
    """An explicit style C offers no tolerance, relaxation or convergence curve."""
    from atlas.workbench.runview import results_frames
    wb.dispatch("file:example:sound-air-water")
    for step in ("case", "geometry", "physics", "check", "run", "results"):
        wb.show(step)
        assert wb.workspace.objects
    wb.show("physics")
    text = _texts(wb.workspace.objects[0])
    assert "Pulse amplitude (Pa)" in text and "Media (the regions' materials)" in text
    assert "nothing iterates" in text
    assert "Tolerance" not in text and "First relaxation factor" not in text
    assert "Dirichlet side" not in text
    wb.spec.run.steps = 30
    run = wb.start_run(["serial", "parallel", "full"], blocking=True)
    assert run is not None and run.status == "done", run and run.error_trace
    assert run.arms == ("serial", "full")
    table, checks = results_frames(run.results)
    assert "reflection R (measured)" in table.columns
    assert "FAIL" not in set(checks["verdict"])
    assert wb.run_panel.conv_fig is None


def test_the_circuit_layer_draws_and_edits_the_attachments(wb):
    """Case 5's circuit, drawn and edited in the page: electrodes beside their edge
    segments, free nodes below the domain, a part between its two nodes; the table
    edits the case (undoably) and refuses a part the schema refuses."""
    from types import SimpleNamespace
    wb.dispatch("file:example:plate-circuit")
    wb.show("geometry")
    wb.geo_editor._switch(layer="attachments")
    ed = wb.geo_editor
    nodes = dict(zip(ed.src_att_nodes.data["name"], zip(ed.src_att_nodes.data["x"],
                                                          ed.src_att_nodes.data["y"])))
    assert set(nodes) == {"E1", "E2", "n1"}
    assert nodes["E1"][0] < 0 < nodes["E2"][0] - 160          # outside the left and right
    assert nodes["n1"][1] < 0                                   # below the domain
    assert ed.src_att.data["id"] == ["B1", "R1"]
    assert ed.src_att.data["text"][0] == "B1: 12 V, 0.5 ohm inside (+ at n1)"
    table = ed.tables["attachments"].value
    assert list(table["id"]) == ["B1", "R1"] and list(table["unit"]) == ["V", "ohm"]
    ed._on_table_edit("attachments", SimpleNamespace(row=1, column="value", value="2.5"))
    assert wb.spec.attachments[1].value == 2.5
    wb.undo()
    assert wb.spec.attachments[1].value == 1.0
    wb.geo_editor._on_table_edit("attachments",
                                 SimpleNamespace(row=1, column="value", value="-1"))
    assert wb.spec.attachments[1].value == 1.0 and "not applied" in wb.log_lines[0]
    wb.geo_editor.add_part("resistor")
    assert [a.id for a in wb.spec.attachments] == ["B1", "R1", "R3"]
    assert wb.spec.attachments[-1].a == "E1"
    wb.geo_editor.tables["attachments"].selection = [2]
    wb.geo_editor.delete_selected()
    assert [a.id for a in wb.spec.attachments] == ["B1", "R1"]


def test_the_circuit_layer_says_when_a_family_ignores_it(wb):
    """And offers no part to add (seen in the page: the buttons were live)."""
    import panel as pn
    wb.dispatch("file:example:wall-2")
    wb.show("geometry")
    wb.geo_editor._switch(layer="attachments")
    text = _texts(wb.workspace.objects[0])
    assert "does not read lumped parts" in text
    found = []

    def walk(o):
        if isinstance(o, pn.widgets.Button) and o.name.startswith("Add "):
            found.append(o)
        for c in getattr(o, "objects", None) or []:
            walk(c)
    walk(wb.workspace.objects[0])
    assert found and all(b.disabled for b in found)


def test_electrode_nodes_are_inside_the_canvas(wb):
    """Seen in the page: the electrodes' nodes sat on the canvas's edges."""
    wb.dispatch("file:example:plate-circuit")
    wb.show("geometry")
    ed = wb.geo_editor
    x0, x1, y0, y1 = ed.ranges()
    for x, y in ed.circuit_nodes().values():
        assert x0 + 2 <= x <= x1 - 2 and y0 + 2 <= y <= y1 - 2


def test_a_region_drawn_in_a_library_material_brings_its_properties(wb):
    """Regions feed materials: a region in a material the case lacks takes the family
    library's properties in the same edit (one undo), and the box offers the case's
    and the library's names instead of the shell's generic ``material-1``."""
    from types import SimpleNamespace
    from atlas.workbench.editor import material_names
    wb.dispatch("file:example:wall-2")
    wb.show("geometry")
    wb.geo_editor._switch(layer="regions")
    assert material_names(wb.spec) == ["steel", "copper", "aluminium"]
    assert wb.geo_state["material"] == "steel"
    wb.geo_state["material"] = "aluminium"
    ed = wb.geo_editor
    r = {k: list(v) for k, v in ed.src_reg_edit.data.items()}
    for k, v in dict(x=40, y=20, w=16, h=16, id="").items():
        r[k].append(v)
    ed.src_reg_edit.data = r
    assert wb.spec.regions[-1].material == "aluminium"
    assert wb.spec.materials["aluminium"] == {"k": 205.0, "rho": 2700.0, "cp": 900.0}
    assert _errors(wb.spec) == []
    wb.undo()
    assert "aluminium" not in wb.spec.materials and len(wb.spec.regions) == 2
    wb.geo_editor._on_table_edit("regions", SimpleNamespace(row=0, column="material",
                                                             value="aluminium"))
    assert wb.spec.regions[0].material == "aluminium" and "aluminium" in wb.spec.materials
    wb.geo_editor._on_table_edit("regions", SimpleNamespace(row=0, column="material",
                                                             value="unobtainium"))
    assert any("'unobtainium' is used by a region and has no properties" in m
               for m in _errors(wb.spec))


def test_plume_boundaries_are_the_familys():
    s = example_case("plume-2")
    s.boundaries[0] = Boundary(id="B-left", edge="left", kind="bank")
    assert any("fixes its outer boundary" in m for m in _errors(s))
    fam = registry.family("transport-2d")
    assert fam.adapter == "atlas.workbench.families.plume" and fam.styles == ("A",)
    assert {c.key: c.tolerance for c in pl.CHECKS} == {
        "mass": 1e-9, "reference": 1e-10, "bitwise": None}
