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

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from atlas.workbench import fv, registry, runner                         # noqa: E402
from atlas.workbench.families import acoustics as ac                     # noqa: E402
from atlas.workbench.families import plume as pl                         # noqa: E402
from atlas.workbench.spec import Boundary, Window, check, example_case   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import studio_driver as D                                               # noqa: E402


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
    assert any("not in the river" in m for m in _errors(s))


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


def test_strip_panels_name_the_split_shown(s):
    """Seen in the retired page: the split's panels said "Decomposed".  Each arm's
    field is sent under the arm's own label, and the difference shown is the lagged
    split's when it runs, else the synchronous one's, which is the unsplit solve to
    the bit."""
    D.ok(s, "case.example", key="heated-strip")
    _run(s, ["serial", "parallel", "full"], steps=2)
    f = D.frames(s)
    assert f["run"]["labels"] == {"serial": "Split, synchronous",
                                  "parallel": "Split, lagged (two threads)",
                                  "full": "Unsplit solver"}
    assert f["run"]["cards"]["field_label"].startswith("von Mises stress")
    assert set(f["fields"]["arms"]) == {"serial", "parallel", "full", "difference"}
    assert f["fields"]["arms"]["difference"]["max"] > 0.0          # the lag, shown
    _run(s, ["serial", "full"], steps=2)
    f = D.frames(s)
    diff = f["fields"]["arms"]["difference"]
    assert diff["min"] == diff["max"] == 0.0                          # synchronous: exact
    assert f["run"]["cards"]["field_difference"]["serial"]["max"] == 0.0


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
    # a coolant that is not whole rows has its flow solved (case file 0.4, flow.py);
    # this one's outlet is on the copper, so it has nowhere to go
    assert any("a coolant inlet or outlet lies on the block's cells" in m
               for m in _errors(s))
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


# ---------------------------------------------------------------------------
# step D: the compile
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key, verdict", [
    ("wall-2", "admit-uncertified"), ("plate-insert", "admit-uncertified"),
    ("bracket-2", "admit-uncertified"),
    ("cooled-block", "admit-uncertified"), ("plume-2", "admit-uncertified"),
    ("plate-circuit", "admit-uncertified"), ("sound-air-water", "admit-uncertified")])
def test_every_family_compiles_to_the_compilers_verdict(key, verdict):
    """The verdicts are the compiler's, pinned so a change in them is seen: every
    family is admitted uncertified, every seam probed through the families' own
    solves.  The iterated styles of one physics were refused at L2/R10 until W348
    (2026-09-30): their pieces take their boundary data from the coupling and the
    scheme solves the interface directly, so R10 decertifies them after the
    scheme, at L5."""
    from atlas.workbench.compile import compile_case
    s = compile_case(example_case(key))
    assert s.refused_before is None and s.verdict == verdict, (s.verdict, s.other[:3])
    assert s.seams and all(sv.verdict in ("admit", "admit-uncertified") for sv in s.seams)
    assert s.cross_points == ()                          # W162: declared, not detected
    r10 = [(r["layer"], r["verdict"]) for r in s.other if r["rule"] == "R10"]
    if key in ("wall-2", "plate-insert", "bracket-2"):
        assert r10 == [("L5", "admit-uncertified")]      # W348
    assert not any(r["verdict"] == "refuse" for r in s.other)


def test_the_split_is_refused_by_the_port_vocabulary():
    from atlas.workbench.compile import compile_case
    s = compile_case(example_case("heated-strip"))
    assert s.verdict == "refuse" and s.refused_before.startswith("NamedHoleError")
    assert "PortAmendment" in s.refused_before and "VOLUMETRIC" in s.refused_before


def test_the_wind_farm_compiles_through_its_own_graph():
    """wake_array's graph, at the example's rung: the cross-points it declares, and
    a verdict per seam the compiler decided."""
    from atlas.workbench.compile import compile_case
    s = compile_case(example_case("wake-array-3"))
    assert s.refused_before is None and s.verdict == "admit-uncertified"
    assert s.cross_points == ("x0y0", "x1y0") and len(s.seams) == 13 and s.agents == 9
    moved = example_case("wake-array-3")
    moved.windows[0].x0 += 8                               # no longer a ladder rung
    s = compile_case(moved)
    assert s.refused_before and "not one" in s.refused_before


def test_a_seam_is_a_real_response(tmp_path):
    """The probed blocks are the families' own solves: an FV window's response to
    its face temperatures rises with them (the Steklov-Poincare sign the first
    draft had backwards), and the compile writes its record beside the case."""
    from atlas.workbench import compile as C
    from atlas.workbench.families import conduction as cd
    s = example_case("wall-2")
    f = cd.field_from_case(s)
    agent = C.FVAgent("steel", f, cd.window_cells(160, (0, 0, 96, 40)), implicit=True,
                      base=350.0, per_effort=True)
    faces = np.arange(agent.cut.face_rows.size)
    agent.ports["p"] = faces
    lo = agent.respond("p", np.full(faces.size, 340.0))
    hi = agent.respond("p", np.full(faces.size, 360.0))
    assert np.all(hi > lo)
    job = C.CompileJob(s, results_dir=str(tmp_path)).run_blocking()
    assert job.status == "done" and job.record_path.endswith(".json")
    assert os.path.basename(job.record_path).startswith("compile-")


@pytest.fixture()
def s(tmp_path, monkeypatch):
    """A page's session (the studio's protocol); its runs and compiles march inside
    the message that starts them."""
    D.synchronous(monkeypatch)
    return D.session(tmp_path)


def _run(s, arms, steps):
    """The case run as the page runs it: its steps set by an edit, then Run."""
    r, _ = D.edit(s, D.set_op(["run", "steps"], steps), label=f"steps = {steps}")
    assert r["ok"], r
    D.ok(s, "run.start", arms=list(arms))
    run = s.run
    assert run is not None and run.status == "done", run and run.error_trace
    return run


def test_plume_in_the_studio(s):
    """The river's own words reach the page (the catalog: its parameters and its
    reaches' properties, each with its unit), and a run of it passes every check,
    with the pollutant in the river measured per arm.  Nothing iterates in style A,
    so no convergence series is sent."""
    fam = D.family(s, "transport-2d")
    params = {p["label"]: p["unit"] for p in fam["params"]}
    assert params["Discharge per metre of width"] == "m^2/s"
    assert {p["label"]: p["unit"] for p in fam["material_props"]} == {
        "Depth": "m", "Mixing coefficient": "m^2/s"}
    D.ok(s, "case.example", key="plume-2")
    run = _run(s, ["serial", "parallel", "full"], steps=20)
    frame = D.frames(s)["run"]
    cards = frame["cards"]
    assert cards["checks"] and cards["failed"] == 0
    assert all(run.results["metrics"][a]["mass_g"] > 0.0 for a in run.arms)
    assert not frame["convergence"]
    assert not any(v for series in frame["series"].values() for k, v in series.items()
                   if "iteration" in k or "sub" in k)


def test_sound_in_the_studio(s):
    """An explicit style C: its parameters reach the page, its run drops the threaded
    arm and says why, every check passes, what has come back from the interface is
    measured (R itself only from the plateau, step 1021), and nothing iterates."""
    fam = D.family(s, "acoustics-2d")
    assert {p["label"]: p["unit"] for p in fam["params"]}["Pulse amplitude"] == "Pa"
    D.ok(s, "case.example", key="sound-air-water")
    run = _run(s, ["serial", "parallel", "full"], steps=30)
    assert run.arms == ("serial", "full")
    frame = D.frames(s)["run"]
    assert frame["cards"]["failed"] == 0 and frame["cards"]["dropped"].get("parallel")
    assert "reflected_last" in run.results["metrics"]["serial"]
    assert not frame["convergence"]


def test_the_circuit_is_edited_as_data_and_undone(s):
    """Case 5's circuit: a part's value is an edit to the case (undoable); a value the
    schema refuses is refused, changes nothing and says so; a part is added and
    removed."""
    D.ok(s, "case.example", key="plate-circuit")
    assert [a.id for a in s.spec.attachments] == ["B1", "R1"]
    assert s.spec.attachments[1].value == 1.0
    r, _ = D.edit(s, D.set_op(["attachments", 1, "value"], 2.5), label="R1 = 2.5 ohm")
    assert r["ok"] and s.spec.attachments[1].value == 2.5
    D.ok(s, "edit.undo")
    assert s.spec.attachments[1].value == 1.0
    r, pushes = D.edit(s, D.set_op(["attachments", 1, "value"], -1.0), label="R1 = -1 ohm")
    assert r["ok"] is False and s.spec.attachments[1].value == 1.0
    assert any("not applied" in n for n in D.notices(pushes, "error"))
    part = s.spec.attachments[1].model_dump(mode="json") | {"id": "R3", "a": "E1"}
    r, _ = D.edit(s, {"op": "insert", "path": ["attachments"], "value": part}, label="a part")
    assert r["ok"] and [a.id for a in s.spec.attachments] == ["B1", "R1", "R3"]
    assert s.spec.attachments[-1].a == "E1"
    r, _ = D.edit(s, {"op": "remove", "path": ["attachments", 2]}, label="removed R3")
    assert r["ok"] and [a.id for a in s.spec.attachments] == ["B1", "R1"]


def test_a_circuit_a_kind_does_not_read_is_said_to_be_ignored(s):
    """A change of physics removes the circuit (and says so); a circuit that arrives
    anyway, from a file written by hand, is kept and said to be ignored.  A kind that
    reads no circuit has no circuit layer in the catalog."""
    from atlas.workbench.spec import Attachment, adapt_to_family
    assert "attachments" not in D.family(s, "conduction-2d")["layers"]
    assert "attachments" in D.family(s, "electric-2d")["layers"]
    c = example_case("plate-circuit")
    parts = list(c.attachments)
    notes = adapt_to_family(c, "conduction-2d")
    assert c.attachments == [] and any("2 circuit parts removed" in n for n in notes)
    c.attachments = [Attachment.model_validate(p.model_dump()) for p in parts]
    D.ok(s, "case.import", text=c.to_json(), name="by-hand.json")
    assert s.spec.attachments
    warns = [r["message"] for r in s.state()["issues"] if r["severity"] == "warning"]
    assert any("does not read lumped" in m for m in warns), warns


def test_a_compile_is_shown_per_seam_and_only_for_its_own_case(s):
    """The compiler's verdict reaches the page with every seam it decided; the last
    compile is not lent to another case opened after it (the gallery pass found
    farm-21's shown beside farm-12); and R10 decertifies instead of refusing (W348)."""
    D.ok(s, "case.example", key="plate-circuit")
    D.ok(s, "compile.start")
    f = D.frames(s)["compile"]
    assert f["status"] == "done" and f["this_case"] is True, f
    summ = f["summary"]
    assert summ["verdict"] == "admit-uncertified" and summ["seams"]
    seams = json.dumps(summ["seams"])
    assert "E1" in seams and "E2" in seams
    D.ok(s, "case.example", key="wall-2")
    assert D.frames(s)["compile"]["this_case"] is False
    D.ok(s, "compile.start")
    f = D.frames(s)["compile"]
    assert f["this_case"] is True and f["summary"]["verdict"] == "admit-uncertified"
    r10 = [(r["layer"], r["verdict"]) for r in f["summary"]["other"] if r["rule"] == "R10"]
    assert r10 == [("L5", "admit-uncertified")]


def test_a_region_drawn_in_a_library_material_brings_its_properties(s):
    """Regions feed materials: a region drawn in a material the case lacks takes the
    family library's properties in the same edit (one undo), and so does an existing
    region given one; a material no one defined (a file written by hand) is the
    check's error."""
    D.ok(s, "case.example", key="wall-2")
    assert "aluminium" in D.family(s, "conduction-2d")["materials"]
    assert "aluminium" not in s.spec.materials and len(s.spec.regions) == 2
    dx = s.spec.domain.dx
    r, _ = D.add_feature(s, D.rect("f9", 32, 16, 16, 16, dx, material="aluminium",
                                   name="Insert"))
    assert r["ok"], r
    assert s.spec.regions[-1].material == "aluminium"
    assert s.spec.materials["aluminium"] == {"k": 205.0, "rho": 2700.0, "cp": 900.0}
    assert D.errors(s) == []
    D.ok(s, "edit.undo")
    assert "aluminium" not in s.spec.materials and len(s.spec.regions) == 2
    i = next(k for k, f in enumerate(D.features(s)) if f["role"] == "region")
    r, _ = D.edit(s, D.set_op(["geometry", "features", i, "material"], "aluminium"),
                  label="made of aluminium")
    assert r["ok"] and "aluminium" in s.spec.materials
    assert "aluminium" in {x.material for x in s.spec.regions}
    by_hand = json.loads(example_case("wall-2").to_json())
    by_hand["regions"][0]["material"] = "unobtainium"
    D.ok(s, "case.import", text=json.dumps(by_hand), name="by-hand.json")
    assert any("'unobtainium' is used by a region and has no properties" in m
               for m in D.errors(s))


def test_plume_boundaries_are_the_familys():
    s = example_case("plume-2")
    s.boundaries[0] = Boundary(id="B-left", edge="left", kind="bank")
    assert any("fixes its outer boundary" in m for m in _errors(s))
    fam = registry.family("transport-2d")
    assert fam.adapter == "atlas.workbench.families.plume" and fam.styles == ("A",)
    assert {c.key: c.tolerance for c in pl.CHECKS} == {
        "mass": 1e-9, "reference": 1e-10, "bitwise": None, "continuity": 1e-12}
