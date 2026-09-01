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
    cfg = de.DemoConfig(domain="small", k=4, horizon=3).clamped()
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
                                  verify_steps=10))
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
    eng = de.Engine(de.DemoConfig(domain="small", k=4, horizon=3))
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
    eng = de.Engine(de.DemoConfig(domain="small", k=4, horizon=3))
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
    eng = de.Engine(de.DemoConfig(domain="small", k=4, horizon=3))
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
    eng = de.Engine(de.DemoConfig(domain="small", k=4, horizon=3, verify_steps=10))
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
