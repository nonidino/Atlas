"""The workbench runner: style A for the wind-farm family, and the page that drives it
(the studio's, through its protocol).

What is pinned here, and why each one:

* the partition of unity over arbitrary rectangles is `ArrayTiling.weights`
  **to the bit** on every measured tiling, and certified on an irregular one;
* the three arms are **W346's own columns to the bit** -- serial is ``E``,
  threaded is ``Ep4``, full is ``F`` -- on the six-window example, so a run is
  W346's arithmetic rather than a re-derivation of it;
* **a single window equals the full domain to the bit** where the discretization
  is identical (the elliptic part embedded, the blend alone), and the measured
  arrangement (exposed, projected) is NOT the full domain's discretization even
  with one window, because it moves the projection out of the window;
* serial equals threaded to the bit on an irregular tiling with two window shapes;
* a run records its machine, its times and its checks; Stop keeps whole
  macro-steps; one run at a time; and the checks' tolerances are the registered
  ones (asserted, so the prose and the code cannot drift).

Kept small: two or three macro-steps at six windows, about a second each.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "scripts"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from atlas.workbench import runner                                      # noqa: E402
from atlas.workbench.families import windfarm as wf                     # noqa: E402
from atlas.workbench.spec import (EXAMPLES, CaseSpec, Window, check,     # noqa: E402
                                  example_case)
from atlas.workbench.tiling import RectangleTiling                      # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import studio_driver as D                                               # noqa: E402


def _small(key="wake-array-3", steps=2, **run):
    s = example_case(key)
    s.run.steps = steps
    for k, v in run.items():
        setattr(s.run, k, v)
    return s


# ---------------------------------------------------------------------------
# the partition of unity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", [k for k, e in EXAMPLES.items()      # the ladder's rungs
                                 if e.family == "incompressible-2d" and "cols" in dict(e.params)])
def test_rectangle_tiling_is_array_tiling_to_the_bit(key):
    from atlas.cases import scaling_ladder as sl
    ex = EXAMPLES[key]
    t = sl.rung(ex.param("cols"), ex.param("rows")).tiling
    s = example_case(key)
    rt = RectangleTiling(s.domain.nx, s.domain.ny,
                         [(w.id, (w.x0, w.y0, w.nx, w.ny)) for w in s.windows], 8)
    ref = t.weights()
    for k, (ox, oy) in enumerate(t.offsets):
        assert np.array_equal(rt.chi[k], ref[k][oy:oy + 128, ox:ox + 128])
    cert = rt.certify()
    assert cert.holds and cert.identity_residual <= 1e-15


def test_an_irregular_tiling_is_certified():
    boxes = [("A", (0, 0, 120, 100)), ("B", (104, 0, 96, 60)), ("C", (104, 44, 96, 56))]
    rt = RectangleTiling(200, 100, boxes, 8)
    cert = rt.certify()
    assert cert.holds and cert.chi_min >= 0.0
    assert set(rt.shapes()) == {(100, 120), (60, 96), (56, 96)}
    # a blend of three equal fields is that field, to round-off
    ones = [np.full((h, w), 2.5) for _n, (_x, _y, w, h) in boxes]
    assert np.max(np.abs(rt.assemble(ones) - 2.5)) <= 1e-14


# ---------------------------------------------------------------------------
# the arms against W346's own columns
# ---------------------------------------------------------------------------


def test_the_arms_are_w346s_columns_to_the_bit():
    import w346_rotor_count_speed as W
    run = wf.build(example_case("wake-array-3"), threads=4)
    R = W.Rung(6, threads=(4,))
    try:
        ref = {"F": R.freestream(), "E": R.freestream(), "Ep4": R.freestream()}
        mine = {a: run.initial(a) for a in wf.ARMS}
        for _ in range(2):
            ref["F"] = R.F(*ref["F"])[:2]
            ref["E"] = R.E(*ref["E"])[:2]
            ref["Ep4"] = R.Ep(*ref["Ep4"], 4)[:2]
            for a in wf.ARMS:
                mine[a] = run.step(a, mine[a])
            for a, k in (("full", "F"), ("serial", "E"), ("parallel", "Ep4")):
                assert np.array_equal(mine[a].u, ref[k][0]), (a, k)
                assert np.array_equal(mine[a].v, ref[k][1]), (a, k)
    finally:
        run.close()
        for p in R.pools.values():
            p.shutdown()


def _one_window(elliptic, assembly, n=128):
    s = example_case("wake-array-3")
    s.domain.nx = s.domain.ny = n
    s.windows = [Window(id="W", x0=0, y0=0, nx=n, ny=n)]
    s.devices = s.devices[:1]
    s.devices[0].x, s.devices[0].y = 1.5, 2.0
    s.coupling.elliptic, s.coupling.assembly = elliptic, assembly
    return s


def test_one_window_is_the_full_domain_to_the_bit_where_the_scheme_is_the_same():
    """The N = 1 control, through the runner: one window, its own projection,
    no global one, is the full-domain solver's arithmetic exactly."""
    run = wf.build(_one_window("embedded", "blend"), arms=("serial", "full"))
    try:
        s, f = run.initial("serial"), run.initial("full")
        for _ in range(3):
            s, f = run.step("serial", s), run.step("full", f)
            assert np.array_equal(s.u, f.u) and np.array_equal(s.v, f.v)
        assert run.observe("serial", s)["mass"] <= 1e-9
    finally:
        run.close()


def test_the_measured_arrangement_is_not_the_full_domain_scheme_even_at_one_window():
    """Exposed windows with one global projection move the elliptic part out of
    the window, so even one window is a different discretization: a
    difference here is the arrangement, not the decomposition."""
    run = wf.build(_one_window("exposed", "projected"), arms=("serial", "full"))
    try:
        s, f = run.initial("serial"), run.initial("full")
        for _ in range(2):
            s, f = run.step("serial", s), run.step("full", f)
        assert not np.array_equal(s.u, f.u)
    finally:
        run.close()


def test_serial_equals_threaded_on_two_window_shapes():
    s = example_case("wake-array-3")
    s.windows = [Window(id="L", x0=0, y0=0, nx=192, ny=240),
                 Window(id="R", x0=176, y0=0, nx=176, ny=128),
                 Window(id="Q", x0=176, y0=112, nx=176, ny=128)]
    assert [i for i in check(s) if i.severity == "error"] == []
    run = wf.build(s, threads=3)
    try:
        assert len(run.groups) == 2 and len(run.chunks) >= 2
        a, b = run.initial("serial"), run.initial("parallel")
        for _ in range(2):
            a, b = run.step("serial", a), run.step("parallel", b)
            assert run.bitwise_equal(a, b)
        assert run.certificate.holds
    finally:
        run.close()


# ---------------------------------------------------------------------------
# a run: the record, the checks, Stop, one at a time
# ---------------------------------------------------------------------------


def test_a_run_records_its_machine_times_and_checks(tmp_path):
    r = runner.CaseRun(_small(steps=2), arms=("serial", "parallel", "full"), steps=2,
                       threads=4, results_dir=str(tmp_path)).run_blocking()
    assert r.status == "done", r.error_trace
    rec = json.loads(open(r.record_path, encoding="utf-8").read())
    assert rec["schema"] == runner.RESULTS_SCHEMA and rec["steps_done"] == 2
    assert set(rec["timing"]) == {"serial", "parallel", "full"}
    assert all(len(t["seconds_per_step"]) == 2 for t in rec["timing"].values())
    assert "speedup_vs_full" in rec["timing"]["parallel"]
    assert rec["machine"]["cpu_count"] == os.cpu_count()
    assert "power" in rec["machine"] and "other_python_processes" in rec["machine"]
    assert CaseSpec.model_validate(rec["case"]) == _small(steps=2)
    assert rec["case_name"] == "wind farm"   # no key given: its kind (a case has no name)
    checks = {c["key"]: c for c in rec["checks"]}
    assert set(checks) == {"mass", "power", "bitwise"}
    assert all(c["passed"] for c in checks.values()), checks
    assert rec["bitwise"] == {"steps": 2, "first_difference": None}
    # every decomposed arm is measured against the full domain, not only the one
    # the power check reads (the gallery pass found the serial row blank); the two
    # are one state to the bit, so one number
    m = rec["metrics"]
    for key in ("farm_power_vs_full", "rms_velocity_difference"):
        assert m["parallel"][key] is not None and m["serial"][key] == m["parallel"][key]
    # the start-up note: two steps is well short of a wake reaching the next rotor
    assert any("start-up only" in n for n in rec["notes"])


def test_stop_keeps_whole_macro_steps():
    r = runner.CaseRun(_small(steps=60), arms=("serial", "full"), steps=60, threads=2)
    r.start()
    t0 = time.time()
    while r.progress().step < 1 and time.time() - t0 < 60:
        time.sleep(0.05)
    r.stop()
    r.join(120)
    assert r.status == "stopped"
    res = r.results
    assert res["stopped"] and 1 <= res["steps_done"] < 60
    n = {len(t["seconds_per_step"]) for t in res["timing"].values()}
    assert n == {res["steps_done"]}


def test_one_run_at_a_time():
    runner._ACTIVE.acquire()
    try:
        r = runner.CaseRun(_small(steps=1), arms=("full",), steps=1).run_blocking()
    finally:
        runner._ACTIVE.release()
    assert r.status == "failed" and "another case is running" in r.progress().error


def test_the_registered_tolerances():
    """The numbers the page and the wiki quote, asserted against the code."""
    got = {c.key: c.tolerance for c in wf.CHECKS}
    assert got == {"mass": 1e-9, "power": 0.25, "bitwise": None}
    assert all(c.registered for c in wf.CHECKS)


def test_a_yawed_disk_is_refused():
    s = example_case("wake-array-3")
    s.devices[0].yaw_deg = 10.0
    assert any("no yaw model" in i.message for i in check(s) if i.severity == "error")


# ---------------------------------------------------------------------------
# the page: the studio's protocol
# ---------------------------------------------------------------------------


@pytest.fixture()
def s(tmp_path, monkeypatch):
    """A page's session on the wind farm, two macro-steps a run; a run marches inside
    the message that starts it."""
    D.synchronous(monkeypatch)
    s = D.session(tmp_path)
    r, _ = D.edit(s, D.set_op(["run", "steps"], 2), label="steps = 2")
    assert r["ok"]
    return s


def test_the_page_runs_a_case_and_shows_its_results(s, tmp_path):
    reply, _ = D.send(s, "run.start", arms=["serial", "parallel", "full"])
    assert reply["ok"] and reply["arms"] == ["serial", "parallel", "full"]
    run = s.run
    assert run is not None and run.status == "done"
    assert run.record_path.startswith(str(tmp_path))
    assert s.status()["kind"] == "ok" and s.status()["text"] == "Ready"
    f = D.frames(s)
    frame, cards = f["run"], f["run"]["cards"]
    assert frame["status"] == "done" and frame["this_case"] is True
    assert [cards["labels"][a] for a in cards["arms"]] == [
        "Decomposed, serial", "Decomposed, parallel", "Full domain"]
    assert cards["checks"] and cards["passed"] == len(cards["checks"]) and cards["failed"] == 0
    assert cards["speed"] and cards["speed"]["ratio"] > 0.0
    # every family's results carry the displayed field's rms difference from the full
    # domain, and the two decomposed arms are bitwise one
    fd = cards["field_difference"]
    assert set(fd) == {"serial", "parallel"} and fd["serial"] == fd["parallel"]
    assert fd["serial"]["rms"] > 0.0
    # the fields reached the page: the arms and their difference, downsampled alike
    p = run.progress()
    assert set(p.fields) == set(f["fields"]["arms"]) == {"serial", "parallel", "full",
                                                         "difference"}
    q = f["fields"]["arms"]["difference"]
    assert (q["ny"], q["nx"]) == p.fields["difference"].shape
    # another case opened: the last run's results are not its results, and the frame
    # says so
    D.ok(s, "case.example", key="plate-circuit")
    assert D.frames(s)["run"]["this_case"] is False


def test_a_finished_run_leaves_the_page_ready_to_run_again(s):
    """Once a run ends, the header's chip is no longer busy and Run is live again (the
    retired page once kept its notice that a run was marching)."""
    D.ok(s, "run.start", arms=["full"])
    assert s.run.status == "done" and not s.run.active
    assert s.status()["kind"] != "busy"
    assert D.frames(s)["run"]["status"] == "done"
    D.ok(s, "run.start", arms=["full"])


def test_the_page_says_which_configuration_is_marching(s):
    """A run is a commit: the case is copied when Run is pressed, the frame says when
    and in what steps, and an edit afterwards does not reach the run."""
    D.ok(s, "run.start", arms=["full"])
    run = s.run
    frame = D.frames(s)["run"]
    assert frame["committed_at"] and frame["steps"] == 2 and frame["step_label"] == "macro-step"
    r, _ = D.edit(s, D.set_op(["run", "threads"], 3), label="threads")
    assert r["ok"] and s.spec.run.threads == 3
    assert run.spec.run.threads != 3                # the edit did not reach the run
    assert run.results["case_name"] == "wake-array-3"  # the record's key, not a name shown
