"""Tier 22 -- the live demo over the PoC 1a composed graph.

`atlas/demo/` and `scripts/w112_farm_demo.py`.

A demo is a thing shown to people who cannot check it, so the assertions here
are about the two ways it could quietly lie:

  * **it could be showing a different model.** `DemoRollout` overrides the
    freestream in two places the parent hard-codes, so at the default inflow it
    is pinned **bitwise** against `wind_farm_design.Rollout` on a real
    `macro_step`. If that ever drifts, everything the PoC measured stops
    applying to what is on the screen.
  * **it could be reassuring.** The validity panel's rows are declared
    predicates with declared limits, and the one that a developed wake always
    violates is marked as a diagnostic rather than silently dropped -- so the
    panel is asserted to be green at the default AND red when the inflow is
    turned somewhere the projection's hypothesis does not hold.

The last group boots the real server and drives it the way the browser does.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.cases import wake_array as wa                              # noqa: E402
from atlas.cases import wind_farm_design as wd                        # noqa: E402

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")

from atlas.demo import engine as de                                   # noqa: E402


def _needs_expert():
    try:
        wa.exposed_reference_solver(wa.NU_REF)
    except Exception as exc:                                # pragma: no cover
        pytest.skip(f"reference.WindowNS unavailable: {exc}")


@pytest.fixture(scope="module")
def small():
    _needs_expert()
    torch.set_num_threads(4)
    cfg = de.DemoConfig(domain="small", k=4, horizon=3,
                        expert="reference_exposed").clamped()
    return cfg, cfg.case()


# ---------------------------------------------------------------------------
# 1. the demo is showing the model the PoC measured
# ---------------------------------------------------------------------------


def test_demo_rollout_is_the_poc_rollout_bitwise_at_the_default_inflow(small):
    """The licence for `DemoRollout` to exist at all.

    It overrides `band` and `project` so the freestream can be pointed and
    scaled. At u_inf = 1 and zero yaw those overrides must reduce to the
    parent's, to the bit, on a real composed macro-step -- otherwise the demo is
    animating a different column from the one every number in
    [[poc1-results-differentiable-design]] was measured on.
    """
    cfg, case = small
    base = wd.Rollout(case)
    demo = de.DemoRollout(case, u_inf=1.0, inflow_deg=0.0)
    th = torch.as_tensor(de.layout_grid(case), dtype=wd.TORCH_DTYPE)
    u = torch.full(case.shape, wa.U_INF, dtype=wd.TORCH_DTYPE)
    v = torch.zeros_like(u)
    with torch.no_grad():
        a = base.macro_step(u, v, th)
        b = demo.macro_step(u, v, th)
    assert torch.equal(a[0], b[0]), "streamwise field differs from the PoC column"
    assert torch.equal(a[1], b[1])
    assert torch.equal(a[2], b[2]), "disk power differs from the PoC column"


def test_changing_the_inflow_actually_changes_the_march(small):
    cfg, case = small
    demo = de.DemoRollout(case, u_inf=1.0, inflow_deg=0.0)
    th = torch.as_tensor(de.layout_grid(case), dtype=wd.TORCH_DTYPE)
    u, v = demo.freestream()
    with torch.no_grad():
        a = demo.macro_step(u, v, th)
    demo.set_inflow(0.8, 6.0)
    u2, v2 = demo.freestream()
    assert float(u2[0, 0]) == pytest.approx(0.8 * math.cos(math.radians(6.0)))
    assert float(v2[0, 0]) == pytest.approx(0.8 * math.sin(math.radians(6.0)))
    with torch.no_grad():
        b = demo.macro_step(u2, v2, th)
    assert not torch.equal(a[0], b[0])
    # a slower wind must extract less power: P scales as U^3
    assert float(b[2].sum()) < float(a[2].sum())


def test_freestream_is_a_fixed_point_of_the_unforced_step(small):
    """No turbines, no change: the composed step must not manufacture a flow."""
    cfg, case = small
    demo = de.DemoRollout(case, u_inf=0.9, inflow_deg=0.0)
    u, v = demo.freestream()
    zero = torch.zeros(case.k * 3, dtype=wd.TORCH_DTYPE)
    with torch.no_grad():
        # a disk at zero thrust: put every rotor outside the box is not possible,
        # so instead check the FIELD half -- blend, project, band on a uniform
        # state must return the uniform state
        bu, bv = demo.blend(demo.cut(u), demo.cut(v))
        pu, pv = demo.project(bu, bv)
        fu, fv = demo.band(pu, pv)
    assert float((fu - 0.9).abs().max()) < 1e-9
    assert float(fv.abs().max()) < 1e-9
    assert zero.numel() == 3 * case.k


# ---------------------------------------------------------------------------
# 2. the knobs are clamped and the layouts are legal
# ---------------------------------------------------------------------------


def test_config_clamps_every_knob():
    c = de.DemoConfig(domain="nonsense", k=999, u_inf=99.0, inflow_deg=-999.0,
                      horizon=999, lr_pos=99.0, verify_steps=-5).clamped()
    assert c.domain == "medium"
    assert c.k == de.LIMITS["k"][1]
    assert c.u_inf == de.LIMITS["u_inf"][1]
    assert c.inflow_deg == de.LIMITS["inflow_deg"][0]
    assert c.horizon == de.LIMITS["horizon"][1]
    assert c.verify_steps == de.LIMITS["verify_steps"][0]


@pytest.mark.parametrize("domain", sorted(de.DOMAINS))
@pytest.mark.parametrize("k", [4, 9, 16, 25])
def test_preset_layouts_respect_the_spacing_limit(domain, k):
    """Both presets, every size, must start the user inside the constraints.

    They did not, before `capacity` existed: the default layout on the smallest
    domain opened with a spacing violation, which is the first thing a
    first-time user would have seen.
    """
    cfg = de.DemoConfig(domain=domain, k=k).clamped()
    case = cfg.case()
    k_fit = min(k, de.capacity(case))
    case = de.DemoConfig(domain=domain, k=k_fit).clamped().case()
    for name, fn in de.LAYOUTS.items():
        th = fn(case)
        assert th.size == 3 * k_fit, name
        assert wd.min_spacing(th) >= case.s_min - 1e-6, (name, domain, k)
        lo, hi = case.box
        assert np.all(th >= lo - 1e-6) and np.all(th <= hi + 1e-6), name


def test_capacity_is_what_the_box_actually_holds():
    for domain in de.DOMAINS:
        case = de.DemoConfig(domain=domain, k=4).clamped().case()
        cap = de.capacity(case)
        big = de.DemoConfig(domain=domain, k=min(25, cap)).clamped().case()
        assert wd.min_spacing(de.layout_grid(big)) >= big.s_min - 1e-6
        assert cap >= 4, domain


# ---------------------------------------------------------------------------
# 3. the validity panel is a citation, not a mood
# ---------------------------------------------------------------------------


def test_validity_is_green_at_the_default_and_names_its_limits(small):
    cfg, case = small
    v = de.validity(cfg, case, de.layout_grid(case), u_max=1.4)
    assert v["ok"] is True
    keys = {r["key"] for r in v["rows"]}
    assert {"cell_re", "band", "inflow_deg", "spacing", "box"} <= keys
    for r in v["rows"]:
        assert r["why"], r["key"]
        assert "limit" in r and "value" in r


def test_live_cell_reynolds_is_a_diagnostic_not_a_gate(small):
    """It is above 8 in ANY developed wake, which the case study records.

    Gating on it would paint the panel red permanently for a condition the user
    cannot act on, and a warning that is always on is not a warning.
    """
    cfg, case = small
    v = de.validity(cfg, case, de.layout_grid(case), u_max=1.6)
    live = next(r for r in v["rows"] if r["key"] == "cell_re_live")
    assert live["gate"] is False
    assert live["value"] > 8.0
    assert v["ok"] is True


def test_turning_the_wind_leaves_the_validated_range(small):
    """The projection declares its hypothesis; the demo must surface breaking it."""
    cfg, case = small
    off = de.DemoConfig(**{**cfg.__dict__, "inflow_deg": 9.0}).clamped()
    v = de.validity(off, case, de.layout_grid(case), u_max=1.4)
    assert v["ok"] is False
    assert "OUTSIDE" in v["headline"]
    bad = next(r for r in v["rows"] if r["key"] == "inflow_deg")
    assert bad["gate"] and not bad["ok"]


def test_over_speeding_the_wind_leaves_the_validated_range(small):
    cfg, case = small
    fast = de.DemoConfig(**{**cfg.__dict__, "u_inf": 1.3}).clamped()
    v = de.validity(fast, case, de.layout_grid(case), u_max=1.4)
    assert v["ok"] is False
    assert next(r for r in v["rows"] if r["key"] == "cell_re")["ok"] is False


# ---------------------------------------------------------------------------
# 4. the colour ramp is the wake-array viewer's
# ---------------------------------------------------------------------------


def test_ramp_is_neutral_at_the_freestream():
    """u = 1 must land on the grey stop, or the picture means something else."""
    t = (1.0 - de.U_LO) / (de.U_HI - de.U_LO)
    assert t == pytest.approx(0.615, abs=0.002)
    lut = de.colour_lut()
    c = lut[int(round(t * 255))]
    assert abs(int(c[0]) - int(c[2])) < 14, f"freestream colour is not neutral: {c}"
    assert 190 < int(c.mean()) < 225
    assert lut[0].mean() < 40, "the wake end of the ramp should be dark"
    assert lut[255][0] > lut[255][2], "the fast end of the ramp should be warm"


def test_field_png_round_trips(small):
    from PIL import Image
    import io as _io
    cfg, case = small
    u = np.full(case.shape, 1.0)
    u[10:20, 10:20] = 0.4
    png = de.field_png(u, stride=1)
    im = Image.open(_io.BytesIO(png))
    assert im.size == (case.shape[1], case.shape[0])
    a = np.asarray(im)
    # row 0 of the image is the TOP, i.e. the largest y
    assert a[-11, 15][2] > a[-11, 15][0] or a[-11, 15].mean() < 200


# ---------------------------------------------------------------------------
# 5. the smoke test: boot it, optimise, verify
# ---------------------------------------------------------------------------


def test_server_boots_optimises_and_verifies():
    """The whole thing, driven the way the browser drives it.

    Boots the real app, opens the real websocket, takes optimiser steps on the
    default config and waits for one classical verification to land. Slow by
    design -- it is the only test that proves the pieces are wired together.
    """
    _needs_expert()
    from fastapi.testclient import TestClient
    from atlas.demo.server import create_app

    torch.set_num_threads(4)
    eng = de.Engine(de.DemoConfig(domain="small", k=4, horizon=3,
                                  verify_steps=10,
                                  expert="reference_exposed"))
    app = create_app(eng)
    with TestClient(app) as client:
        meta = client.get("/api/meta").json()
        assert set(meta["domains"]) == set(de.DOMAINS)
        assert meta["limits"]["k"] == list(de.LIMITS["k"])

        with client.websocket_connect("/ws") as ws:
            state = json.loads(ws.receive_text())
            assert state["type"] == "state"
            png = ws.receive_bytes()
            assert png[:8] == b"\x89PNG\r\n\x1a\n", "the field frame is not a PNG"
            assert len(state["turbines"]) == 4

            # a drag
            th = [[t["x"], t["y"], 0.0] for t in state["turbines"]]
            th[0][1] += 0.4
            ws.send_text(json.dumps({"type": "theta",
                                     "theta": [c for t in th for c in t]}))

            # two optimiser steps
            for _ in range(2):
                ws.send_text(json.dumps({"type": "mode", "mode": "step"}))
                deadline = time.time() + 240
                start = eng.opt_iter
                while eng.opt_iter == start and time.time() < deadline:
                    time.sleep(0.4)
                assert eng.opt_iter > start, "no optimiser step completed"

            assert len(eng.history) >= 2
            assert all(np.isfinite(h["J"]) for h in eng.history)
            assert eng.history[-1]["grad_norm"] > 0.0

            # the classical verification
            ws.send_text(json.dumps({"type": "verify"}))
            deadline = time.time() + 420
            h = eng.layout_hash()
            while h not in eng._verify_cache and time.time() < deadline:
                time.sleep(0.5)
            assert h in eng._verify_cache, "verification never completed"
            r = eng._verify_cache[h]
            assert "error" not in r, r.get("error")
            assert r["composed_power"] > 0 and r["classical_power"] > 0
            assert r["steps"] == 10
            # the cache is what makes a repeat instant
            t0 = time.perf_counter()
            eng.request_verify()
            assert time.perf_counter() - t0 < 0.5

    eng.stop()


def test_scrub_restores_a_recorded_layout():
    """The replay slider must put back a layout the optimiser actually visited."""
    _needs_expert()
    torch.set_num_threads(4)
    eng = de.Engine(de.DemoConfig(domain="small", k=4, horizon=3,
                                  expert="reference_exposed"))
    eng.history = [
        {"iter": 1, "J": 1.0, "power": 1.0, "grad_norm": 0.1, "wall_s": 0.1,
         "theta": [float(x) for x in de.layout_staggered(eng.case)]},
    ]
    eng._apply("scrub", 0)
    assert eng.mode == "replay"
    assert eng.replay_play is False, "scrubbing must land paused, not playing"
    assert np.allclose(eng.theta,
                       wd.project_design(de.layout_staggered(eng.case), eng.case))
    eng.stop()


# ---------------------------------------------------------------------------
# 6. the replay plays and pauses
# ---------------------------------------------------------------------------


def _fake_history(eng, n=4):
    """`n` recorded optimiser steps, each a genuinely different layout."""
    base = de.layout_grid(eng.case).copy()
    hist = []
    for i in range(n):
        th = base.copy()
        th[0::3] += 0.05 * i                       # nudge every x downwind
        hist.append({"iter": i + 1, "J": float(i), "power": float(i),
                     "grad_norm": 0.1, "wall_s": 0.1,
                     "theta": [float(x) for x in wd.project_design(th, eng.case)]})
    eng.history = hist
    return hist


def test_replay_play_pause_advances_and_holds():
    """Play walks the cursor forward; pause stops it where it is.

    The cursor advances once every REPLAY_HOLD macro-steps rather than once a
    step, so the flow has a chance to follow the layout it is supposed to be
    explaining -- checked here by ticking the replay by hand rather than by
    waiting on the worker thread.
    """
    _needs_expert()
    torch.set_num_threads(4)
    eng = de.Engine(de.DemoConfig(domain="small", k=4, horizon=3,
                                  expert="reference_exposed"))
    hist = _fake_history(eng, 4)

    eng._apply("replay", {"index": 0, "play": True})
    assert eng.mode == "replay" and eng.replay_play is True
    assert eng.replay_i == 0

    for _ in range(de.REPLAY_HOLD):
        eng._replay_step()
    assert eng.replay_i == 1, "the cursor did not advance after REPLAY_HOLD steps"
    assert np.allclose(eng.theta, np.array(hist[1]["theta"]))

    eng._apply("replay", {"play": False})
    for _ in range(2 * de.REPLAY_HOLD):
        eng._replay_step()
    assert eng.replay_i == 1, "a paused replay must not advance"
    assert eng.mode == "replay", "pausing the replay must not leave replay"

    # played to the end, it stops there rather than wrapping
    eng._apply("replay", {"play": True})
    for _ in range(de.REPLAY_HOLD * 10):
        eng._replay_step()
    assert eng.replay_i == len(hist) - 1
    assert eng.replay_play is False
    eng.stop()


def test_replay_needs_a_trajectory_and_says_so():
    _needs_expert()
    eng = de.Engine(de.DemoConfig(domain="small", k=4, horizon=3,
                                  expert="reference_exposed"))
    eng.history = []
    eng._apply("replay", {"play": True})
    assert eng.replay_play is False
    assert "Optimise" in eng.notice
    assert eng.mode != "replay"
    eng.stop()


# ---------------------------------------------------------------------------
# 7. the head-to-head: same layout, both solvers, timed
# ---------------------------------------------------------------------------


def test_ms_stats_holds_out_the_warm_up_step():
    """Step 1 pays for FFT plans and allocation; averaging it in reports a cost
    nobody pays after the first second."""
    s = de._ms_stats([900.0, 100.0, 102.0, 98.0, 100.0])
    assert s["first_ms"] == 900.0
    assert s["median_ms"] == pytest.approx(100.0)
    assert s["mean_ms"] < 105.0, "the warm-up leaked into the headline"
    assert s["total_ms"] == pytest.approx(1300.0)
    assert de._ms_stats([])["n"] == 0
    one = de._ms_stats([7.0])
    assert one["median_ms"] == 7.0, "a single sample must still report something"


def test_head_to_head_times_both_solvers_and_serves_a_second_window():
    """The second point of the demo, end to end.

    One layout, marched by the coupled graph and by the undivided classical
    solver, alternating. The assertions are about the things that would make the
    timing a lie: that both sides were actually timed per step, that the warm-up
    is held out of the headline, and that the live march was parked while it ran
    rather than competing with it.
    """
    _needs_expert()
    from fastapi.testclient import TestClient
    from atlas.demo.server import create_app

    torch.set_num_threads(4)
    eng = de.Engine(de.DemoConfig(domain="small", k=4, horizon=3, verify_steps=10,
                                  expert="reference_exposed"))
    app = create_app(eng)
    with TestClient(app) as client:
        assert b"<html" in client.get("/compare").content.lower()
        assert client.get("/api/compare").json()["state"] == "idle"
        # the second window seeds its step count from the RUNNING config, not
        # the dataclass defaults -- seeding from the defaults meant a server
        # launched with --steps 14 quietly ran 30 and labelled the result 30
        meta = client.get("/api/meta").json()
        assert meta["config"]["verify_steps"] == 10
        assert meta["defaults"]["verify_steps"] != 10, "the two must be distinguishable"

        r = client.post("/api/compare", json={"steps": 10, "force": True,
                                              "pause_live": True})
        assert r.status_code == 200
        h = r.json()["hash"]

        deadline = time.time() + 600
        while time.time() < deadline:
            v = client.get("/api/compare").json()
            if v["state"] in ("done", "error"):
                break
            if v["state"] == "running" and v["i"] >= 1:
                # mid-run the second window must be able to draw both fields
                for side in ("composed", "classical"):
                    px = client.get(f"/api/compare/field?side={side}&seq={v['seq']}")
                    assert px.status_code == 200
                    assert px.content[:8] == b"\x89PNG\r\n\x1a\n", side
                assert eng._parked.is_set(), "the live march kept running while timing"
            time.sleep(0.5)

        v = client.get("/api/compare").json()
        assert v["state"] == "done", v.get("error")
        assert v["i"] == 10 and v["steps"] == 10

        # both sides timed, every step, and the two are directly comparable
        for side in ("composed", "classical"):
            assert v[side]["ms"]["n"] == 10, side
            assert v[side]["ms"]["median_ms"] > 0.0, side
            assert v[side]["power"] > 0.0, side
            assert len(v[side]["trace"]) == 10, side
        assert v["composed"]["where"] == eng.cfg.device
        assert v["classical"]["where"] == "cpu", "the monolith is numpy: CPU only"
        # with the classical solver in the windows there is no third column to
        # run: it would be the first one again
        assert v["has_third"] is False and v["third"] is None
        assert v["expert"] == "reference_exposed"
        assert v["ms_ratio"] == pytest.approx(
            v["composed"]["ms"]["median_ms"] / v["classical"]["ms"]["median_ms"])
        assert v["paused_live"] is True
        assert len(v["linf"]) == 10 and v["linf_final"] >= 0.0
        assert v["theta"] and len(v["theta"]) == 3 * eng.cfg.k

        # the same run, filed for the main window's panel
        rec = eng._verify_cache[h]
        assert rec["composed_ms_step"] > 0 and rec["classical_ms_step"] > 0
        assert rec["steps"] == 10
        assert rec["timed_alone"] is True

        # and the live march comes back on its own once the timing is over
        assert not eng._timing_now()

    eng.stop()


# ---------------------------------------------------------------------------
# 8. the measurement region, and the three panels read off it
# ---------------------------------------------------------------------------


def _needs_poseidon():
    if not de._expert_ok("poseidon"):                       # pragma: no cover
        pytest.skip("the Poseidon-T checkpoint is not available here")


def test_nothing_is_quoted_before_it_is_measured():
    """The demo may not put a wall-clock number on screen that it did not produce.

    Section 10 of `atlas-proof-of-concept-1` exists because a per-step cost was
    written into this repository as a constant and was wrong by 4x the next time
    anyone measured it -- same box, same code, a different clock state. So the
    rule is structural rather than a promise: a fresh engine reports `None` for
    every timing, and there is no table left in the module for anything to fall
    back to.
    """
    _needs_expert()
    eng = de.Engine(de.DemoConfig(domain="small", k=4, horizon=3,
                                  expert="reference_exposed"))
    sp = eng.speed_view()
    assert sp["measured"] is False and sp["ratio"] is None
    assert all(c["ms"] is None for c in sp["columns"].values())
    assert sp["regions"] == 0 and sp["steps"] == 0
    assert eng.eta_s() is None, "an unmeasured machine must not get a number"
    assert eng.accuracy_view()["measured"] is False
    assert eng.scoring_view()["baseline"] is None
    assert not hasattr(de.DemoConfig, "horizon_eta_s"), \
        "the cold-start table is back; it is exactly the thing that was wrong"
    # the published numbers travel, but only in their own field, with a source
    pub = sp["published"]
    assert "poc1a-frozen-expert-results" in pub["source"]
    assert pub["speedup_k12"] == 3.15 and pub["capture_k12"] == 71.2
    eng.stop()


def test_a_measurement_times_every_column_alone_and_fills_all_three_panels():
    """The one march the whole screen is read off.

    Three columns -- the frozen-expert composed graph, the same cut with the
    classical solver in the windows, and the undivided classical monolith --
    marched from still air, one macro-step each in turn. The assertions are
    about the things that would make the screen a lie: that each column was
    timed for every step with the others idle, that the accuracy figures are
    against the undivided solver, and that the scoring row is the same march
    rather than a second one taken under other conditions.
    """
    _needs_expert()
    _needs_poseidon()
    torch.set_num_threads(4)
    eng = de.Engine(de.DemoConfig(domain="small", k=4, horizon=3,
                                  verify_steps=10, expert="poseidon",
                                  three_way=True))
    eng.start()
    try:
        assert eng.cfg.columns == ("composed", "composed_classical", "monolith")
        eng.request_verify(force=True, pause_live=True)
        deadline = time.time() + 900
        while time.time() < deadline:
            if eng.compare.state in ("done", "error"):
                break
            time.sleep(0.5)
        c = eng.compare
        assert c.state == "done", c.error
        assert c.i == 10

        # every column, every step, timed
        for ms in (c.composed_ms, c.classical_ms, c.third_ms):
            assert len(ms) == 10
            assert all(x > 0 for x in ms)
        assert len(c.linf) == 10 and len(c.third_linf) == 10
        assert set(c.png) >= {"composed", "monolith", "composed_classical"}

        v = c.view()
        assert v["has_third"] is True and v["third"] is not None
        assert v["ms_ratio"] == pytest.approx(
            v["composed"]["ms"]["median_ms"] / v["classical"]["ms"]["median_ms"])
        assert v["speedup"] == pytest.approx(1.0 / v["ms_ratio"])

        # panel 1: speed, measured here, over one region
        sp = eng.speed_view()
        assert sp["measured"] is True
        assert sp["regions"] == 1 and sp["steps"] == 10
        for key in ("composed", "composed_classical", "monolith"):
            assert sp["columns"][key]["ms"] > 0, key
        assert sp["ratio"] == pytest.approx(
            sp["columns"]["monolith"]["ms"] / sp["columns"]["composed"]["ms"])
        assert sp["ratio_cut"] == pytest.approx(
            sp["columns"]["monolith"]["ms"]
            / sp["columns"]["composed_classical"]["ms"])

        # panel 2: accuracy, against the undivided solver and nothing else
        acc = eng.accuracy_view()
        assert acc["measured"] is True and acc["stale"] is False
        assert acc["linf"] > 0 and acc["third_linf"] > 0
        assert acc["composed_power"] > 0 and acc["monolith_power"] > 0
        assert acc["delta_pct"] == pytest.approx(
            100.0 * (acc["composed_power"] - acc["monolith_power"])
            / acc["monolith_power"])

        # panel 3: the score book, from the SAME march
        sc = eng.scoring_view()
        base = sc["baseline"]
        assert base is not None and base["at_iter"] == 0
        assert base["from_expert"] is None, "nobody optimised the starting layout"
        assert base["powers"]["composed"] == pytest.approx(acc["composed_power"])
        assert base["powers"]["monolith"] == pytest.approx(acc["monolith_power"])
        assert base["powers"]["composed_classical"] != base["powers"]["composed"]
        # one layout only: there is no gain to report yet and none is invented
        assert sc["gain"] is None and sc["capture"] is None
    finally:
        eng.stop()


def test_a_two_column_measurement_skips_the_row_that_would_repeat_itself():
    """With the classical solver in the windows the third column IS the first."""
    _needs_expert()
    cfg = de.DemoConfig(domain="small", k=4, expert="reference_exposed",
                        three_way=True).clamped()
    assert cfg.columns == ("composed", "monolith")
    cmp_ = de.Comparison(columns=cfg.columns)
    assert cmp_.has_third is False
    assert cmp_.view()["third"] is None


def test_the_score_book_survives_a_reset_and_not_a_change_of_conditions():
    """A gain is a difference between two numbers measured the same way.

    Reset puts the same layout back under the same wind, so the reference score
    is still a statement about this farm and is kept. Moving the wind, the box,
    the turbine count or the expert makes every stored number a statement about
    something else, so the book goes.
    """
    _needs_expert()
    eng = de.Engine(de.DemoConfig(domain="small", k=4, horizon=3,
                                  expert="reference_exposed"))
    fake = {"hash": "h0", "at_iter": 0, "theta": [0.0] * 12, "steps": 10, "k": 4,
            "expert": "reference_exposed", "composed_power": 2.0,
            "classical_power": 2.5, "third_power": 2.1,
            "composed_ms_stats": {"median_ms": 100.0},
            "classical_ms_stats": {"median_ms": 300.0}}
    eng._file_measurement(fake)
    assert eng.baseline == "h0" and eng.speed_regions == 1

    eng._apply("reset", None)
    assert eng.baseline == "h0", "a plain reset threw away a valid reference"

    eng._apply("config", {"u_inf": 1.2})
    assert eng.baseline is None and eng.scores == {}
    assert "conditions changed" in eng.notice
    eng.stop()


def test_the_gain_is_the_verifier_s_and_the_capture_fraction_is_withheld():
    """`poc1a-frozen-expert-results` section 7.2, in the form the demo can reach.

    The headline gain is the MONOLITH's, because it is the only column with no
    composition error in it. The 71 % capture fraction compares an optimum found
    on the frozen column against one found on the classical column, under that
    same verifier -- so it is reported as measured HERE only when this session
    holds both, and is a citation with a source attached otherwise.
    """
    _needs_expert()
    eng = de.Engine(de.DemoConfig(domain="small", k=4, horizon=3,
                                  expert="reference_exposed"))

    def file(h, at_iter, expert, composed, monolith):
        eng._file_measurement({
            "hash": h, "at_iter": at_iter, "theta": [0.0] * 12, "steps": 10,
            "k": 4, "expert": expert, "composed_power": composed,
            "classical_power": monolith, "third_power": composed,
            "composed_ms_stats": {"median_ms": 100.0},
            "classical_ms_stats": {"median_ms": 300.0}})

    file("start", 0, "poseidon", 2.0, 2.5)
    file("opt_p", 18, "poseidon", 6.0, 7.5)
    eng.theta = eng.theta                      # the layout on screen is neither
    sc = eng.scoring_view()
    assert sc["gain"]["monolith"] == pytest.approx(200.0)
    assert sc["capture"]["measured_here"] is False
    assert sc["capture"]["have"] == ["poseidon"]
    assert sc["published"]["capture_k12"] == 71.2

    # ... and once a layout optimised on the classical column has been scored
    # too, the fraction is this session's own number
    file("opt_c", 18, "reference_exposed", 6.0, 10.0)
    cap = eng.scoring_view()["capture"]
    assert cap["measured_here"] is True
    assert cap["poseidon_gain"] == pytest.approx(200.0)
    assert cap["classical_gain"] == pytest.approx(300.0)
    assert cap["pct"] == pytest.approx(200.0 / 300.0 * 100.0)
    eng.stop()


def test_swapping_the_expert_rebuilds_every_solver_and_reseeds():
    """A demo that changed the label and not the solver would be the worst bug here."""
    _needs_expert()
    _needs_poseidon()
    eng = de.Engine(de.DemoConfig(domain="small", k=4, horizon=3,
                                  expert="reference_exposed"))
    assert isinstance(eng.ro, de.DemoRollout)
    eng._apply("config", {"expert": "poseidon"})
    assert eng.cfg.expert == "poseidon"
    assert isinstance(eng.ro, de.DemoPoseidonRollout)
    assert "fluid expert changed" in eng.notice
    assert eng.opt_iter == 0 and eng.history == []
    eng.stop()


def test_the_expert_is_named_on_screen_with_its_licence():
    """Poseidon-T is CC-BY-NC-4.0 and a demo that does not say so is a trap.

    The label and the licence travel in the state payload, so the page cannot
    render a power number without the name of the thing that produced it being
    available beside it -- and the page is asserted to have somewhere to put it.
    """
    _needs_expert()
    eng = de.Engine(de.DemoConfig(domain="small", k=4, horizon=3,
                                  expert="reference_exposed"))
    eng._publish()
    p = eng.frame.payload
    assert p["expert_label"] == de.EXPERT_LABEL["reference_exposed"]
    assert p["expert_licence"] == ""
    assert "CC-BY-NC-4.0" in de.EXPERT_NOTE["poseidon"]
    assert "CC-BY-NC-4.0" in de.EXPERT_LICENCE["poseidon"]
    page = os.path.join(os.path.dirname(de.__file__), "static", "index.html")
    html = open(page, encoding="utf-8").read()
    assert 'id="explic"' in html and 'id="expname"' in html
    readme = os.path.join(os.path.dirname(de.__file__), "README.md")
    assert "CC-BY-NC-4.0" in open(readme, encoding="utf-8").read()
    eng.stop()


def test_measure_on_start_is_off_unless_asked_for():
    """Constructing an Engine in a test must not start a two-minute timing run."""
    _needs_expert()
    assert de.DemoConfig().measure_on_start is False
    eng = de.Engine(de.DemoConfig(domain="small", k=4, measure_on_start=True,
                                  expert="reference_exposed"))
    assert eng._measure_pending is True
    quiet = de.Engine(de.DemoConfig(domain="small", k=4,
                                    expert="reference_exposed"))
    assert quiet._measure_pending is False
    eng.stop()
    quiet.stop()


def test_the_bundle_pins_every_dependency_and_carries_the_checkpoint():
    """The packaging promise, checked as text rather than trusted.

    A bundle that resolves `torch` to whatever is newest is a bundle whose
    numbers cannot be compared between two machines, and one that downloads the
    checkpoint on first use does not run on a machine with no network. Both were
    true of the previous version.
    """
    # Upstream the launchers live in `atlas/demo/bundle/`; in the assembled
    # bundle they ARE the checkout root, which is the copy a user actually runs.
    # Checking whichever is present means this file asserts the same property in
    # both places -- which is the point of it being the same file in both.
    b = os.path.join(os.path.dirname(de.__file__), "bundle")
    if not os.path.isdir(b):                                # pragma: no cover
        b = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(de.__file__))))
    req = open(os.path.join(b, "requirements.txt"), encoding="utf-8").read()
    pins = [ln.strip() for ln in req.splitlines()
            if ln.strip() and not ln.startswith("#")]
    assert pins, "no requirements at all"
    for ln in pins:
        assert "==" in ln, f"{ln!r} is not pinned"
    for pkg in ("numpy", "scipy", "pillow", "fastapi", "uvicorn",
                "transformers", "safetensors", "huggingface-hub"):
        assert any(ln.startswith(pkg) for ln in pins), pkg

    for name in ("run.sh", "run.cmd"):
        sh = open(os.path.join(b, name), encoding="utf-8").read()
        assert "torch==" in sh, f"{name} does not pin torch"
        assert "nvidia-smi" in sh, f"{name} does not look for a GPU"
        assert "whl/cpu" in sh, f"{name} has no CPU-only path"
        assert "poseidon/archive/" in sh, f"{name} does not install scOT"
        assert "--check" in sh, f"{name} does not self-test before serving"

    run = open(os.path.join(b, "run.py"), encoding="utf-8").read()
    assert "HF_HOME" in run and "HF_HUB_OFFLINE" in run, \
        "the bundled checkpoint is not wired up, so it would be downloaded"
    assert "ATLAS_BUILD_REPO" in run
    for pkg in ("torch", "scOT", "transformers", "fastapi"):
        assert f'("{pkg}"' in run, f"run.py --check does not verify {pkg}"


def test_the_windows_command_is_written_the_way_powershell_needs_it():
    """The first command, in the shell Windows 11 actually opens.

    PowerShell does not run a program from the current directory, so a bare
    `run.cmd` fails with "The term 'run.cmd' is not recognized" while the file
    sits in plain sight. The launcher was verified from a real clone and this
    was still missed, because the verification never typed the documented
    command into the documented shell -- so the promise of one command from a
    bare checkout was false in the default case. Every place a reader is told
    what to type must carry the `.\\`.
    """
    b = os.path.join(os.path.dirname(de.__file__), "bundle")
    if not os.path.isdir(b):                                # pragma: no cover
        b = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(de.__file__))))

    for name in ("README.md", "run.cmd"):
        text = open(os.path.join(b, name), encoding="utf-8").read()
        for i, line in enumerate(text.splitlines(), 1):
            stripped = line.lstrip(" \t#>REM").lstrip()
            if stripped.startswith("run.cmd"):
                raise AssertionError(
                    f"{name}:{i} tells the reader to type a bare `run.cmd`, "
                    f"which PowerShell refuses: {line.strip()!r}")
        assert ".\\run.cmd" in text, f"{name} never shows the working form"
