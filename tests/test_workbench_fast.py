"""Demo item 1.4: one Fast example per simulation type.

The bars registered in every family module before the first timed run; the
multirate style M on conduction (the N = 1 control to the bit, the balance at
round-off, the agreement against its registered 1e-3); the river's and the sound's
several steps per exchange on a halo (the full domain to round-off and to the bit,
threaded equal to serial, the balance closed); the river's leaner window step and
blend, the same values as the code they replaced; the header's *Fast example*
button with *More examples* under it; and the speed card.  No timing is asserted
here: the speeds are measured by `scripts/fast_examples.py` and recorded in
``out/workbench/records/fast/``.
"""

from __future__ import annotations

import numpy as np
import pytest

from atlas.workbench import fast, fv, layout
from atlas.workbench.spec import (EXAMPLES, FAST_EXAMPLES, Layout, check, example_case)

FAMILIES = ("incompressible-2d", "conduction-2d", "transport-2d", "acoustics-2d",
            "elasticity-2d", "electric-2d", "thermoelastic-2d", "conjugate-heat-2d")


# ---------------------------------------------------------------------------
# the bars and the examples
# ---------------------------------------------------------------------------


def test_every_family_registers_its_bars_before_its_first_timed_run():
    from atlas.workbench.runner import adapter_for
    for fid in FAMILIES:
        bars = adapter_for(fid).FAST
        assert bars and all(isinstance(b, fast.FastBars) for b in bars), fid
        for b in bars:
            assert b.speed == 3.0, (fid, b)                    # the plan's bar, unloosened
            assert b.registered.startswith("2026-09-30, before the Fast example's first")
            if b.mechanism in ("P", "S"):
                assert b.agreement in (1e-9, 0.03), (fid, b)   # round-off; the farm's 3 %
            if b.mechanism == "M":
                assert b.agreement == 1e-3, (fid, b)
            if b.mechanism == "X":
                assert b.ceiling and "halve" in b.ceiling


def test_every_kind_has_a_fast_example_that_is_ready_to_run():
    assert sorted(FAST_EXAMPLES) == sorted(FAMILIES)
    for fid, key in FAST_EXAMPLES.items():
        spec = example_case(key)
        assert EXAMPLES[key].family == fid == spec.physics.family
        assert not [i for i in check(spec) if i.severity == "error"], key


def test_judge_reads_both_bars():
    b = fast.FastBars("P", 3.0, 1e-9, "x")
    assert fast.judge(b, 4.0, 1e-12)["met"] is True
    assert fast.judge(b, 2.9, 1e-12)["met"] is False
    assert fast.judge(b, 4.0, 1e-6)["met"] is False
    x = fast.FastBars("X", 3.0, None, "own controls", ceiling="at most half")
    j = fast.judge(x, 1.4, None)
    assert j["met"] is False and j["close"] is None and j["ceiling"] == "at most half"
    with pytest.raises(ValueError):
        fast.FastBars("P", 1.0, 1e-9, "x")


# ---------------------------------------------------------------------------
# style M: conduction, each piece at its own explicit step
# ---------------------------------------------------------------------------


def _insert_m(dt_frac: float, steps: int):
    from atlas.workbench.families import conduction as C
    s = example_case("plate-insert")
    s.coupling.style = "M"
    s.layout = Layout(cut="materials")
    layout.refresh(s)
    lim = C.piece_limits(s)
    s.run.macro_dt = dt_frac
    s.run.steps = steps
    return s, lim


def _march(run, steps):
    st = {a: run.initial(a) for a in run.arms}
    hist = {a: [] for a in run.arms}
    for _ in range(steps):
        for a in run.arms:
            new = run.step(a, st[a])
            hist[a].append(run.observe(a, new, st[a]))
            st[a] = new
    return st, hist


def test_multirate_with_equal_steps_is_the_full_domain_to_the_bit():
    """The N = 1 control: a macro-step every piece can take whole gives rate 1, and
    the two-rate step is the full domain's explicit step bit for bit."""
    from atlas.workbench.families import conduction as C
    s, lim = _insert_m(1.0, 30)
    s.run.macro_dt = 0.999 * min(lim.values())
    assert not [i for i in check(s) if i.severity == "error"]
    run = C.build(s, arms=("serial", "full"))
    assert run.rate == 1
    st, _ = _march(run, 30)
    assert np.array_equal(st["serial"].u, st["full"].u)


def test_multirate_closes_its_balance_and_meets_its_registered_agreement():
    from atlas.workbench.families import conduction as C
    s, lim = _insert_m(1.0, 300)
    s.run.macro_dt = 0.999 * lim["steel"]
    run = C.build(s, arms=("serial", "full"))
    assert run.rate == 8 and run.fast_ids == ["copper"] and run.slow_ids == ["steel"]
    st, hist = _march(run, 300)
    metrics, checks = run.compare(st, hist, {})
    by = {c.spec.key: c for c in checks}
    assert by["balance"].passed and by["balance"].value < 1e-12   # refluxed: round-off
    ref = by["reference_multirate"]
    assert ref.spec.tolerance == 1e-3 and ref.passed and 1e-5 < ref.value < 1e-3
    assert "reference" not in by                                  # judged on its own check


def test_style_m_is_refused_on_a_steady_case():
    s, _lim = _insert_m(1.0, 3)
    s.run.mode = "steady"
    assert any("style M steps each piece" in i.message for i in check(s)
               if i.severity == "error")


def test_the_fast_heat_example_takes_the_steel_step_whole():
    from atlas.workbench.families import conduction as C
    s = example_case("fast-heat")
    lim = C.piece_limits(s)
    assert s.run.macro_dt <= lim["steel"] and s.run.macro_dt > lim["copper"]
    assert {w.id for w in s.windows} == {"steel", "copper"}


# ---------------------------------------------------------------------------
# the river: several explicit steps per exchange, on a halo
# ---------------------------------------------------------------------------


def _river(key, k=None):
    from atlas.workbench.families import plume as Pl
    s = example_case(key)
    if k is not None:
        s.run.exchange_every = k
        s.run.macro_dt = s.run.macro_dt * k * 0.999
    return s, Pl


@pytest.mark.parametrize("key, k, steps", [("plume-2", 3, 120), ("river-fork", 4, 60)])
def test_the_river_on_a_halo_is_the_full_domain_and_closes_its_mass(key, k, steps):
    s, Pl = _river(key, k)
    assert not [i for i in check(s) if i.severity == "error"]
    run = Pl.build(s, arms=("serial", "parallel", "full"), threads=3)
    st = {a: run.initial(a) for a in run.arms}
    worst = 0.0
    for _ in range(steps):
        for a in run.arms:
            new = run.step(a, st[a])
            worst = max(worst, run.observe(a, new, st[a])["balance"])
            st[a] = new
        assert np.array_equal(st["serial"].u, st["parallel"].u)
    run.close()
    peak = float(np.max(np.abs(st["full"].u)))
    assert peak > 0 and float(np.max(np.abs(st["serial"].u - st["full"].u))) <= 1e-12 * peak
    assert worst < 1e-12


def test_the_rivers_leaner_window_and_blend_are_the_code_they_replaced():
    """The window step's ring rows, slices and in-place arithmetic, and the blend from
    each window's own cells: the same values, every step (exchange_every = 1)."""
    s, Pl = _river("plume-2")
    run = Pl.build(s, arms=("serial", "full"), threads=1)
    u = run.initial("serial")
    for _ in range(40):
        locals_ = []
        for k in range(len(run.systems)):
            sy = run.systems[k]
            old = (u.u[sy.idx] + run.coefs[k] * ((sy.b - sy.C @ u.u) - sy.A @ u.u[sy.idx]))
            _x0, _y0, w, h = run.windows[k][1]
            new = run._window(k, u.u)
            assert np.array_equal(new, old.reshape(h, w))
            locals_.append(new)
        nxt = run.step("serial", u)
        assert np.array_equal(nxt.u, run.tiling.assemble(locals_).ravel())
        u = nxt


def test_the_affine_ledger_is_the_face_inflow():
    s, Pl = _river("plume-2")
    run = Pl.build(s, arms=("full",), threads=1)
    u = run.initial("full")
    for _ in range(50):
        u = run.step("full", u)
    bc = fv.boundary_faces(run.f)
    c, a, b = fv.face_inflow_affine(run.f, *bc)
    direct = float(np.sum(fv._face_inflow(run.f, u.u, *bc)))
    assert abs((float(a @ u.u[c]) + b) - direct) <= 1e-12 * max(abs(direct), 1.0)


def test_the_fast_river_overlaps_by_its_steps_per_exchange():
    """24 explicit steps per exchange on six windows, which overlap by 24 cells, so the
    graph declares the halo the scheme needs (the compiler's halo rule refused the
    first layout's 16)."""
    s = example_case("fast-river")
    from atlas.workbench.families import plume as Pl
    assert s.run.exchange_every == 24 and len(s.windows) == 6
    assert s.run.macro_dt / s.run.exchange_every <= Pl.explicit_limit(s)
    ws = sorted(s.windows, key=lambda w: w.x0)
    assert min(a.x0 + a.nx - b.x0 for a, b in zip(ws, ws[1:])) >= s.run.exchange_every


# ---------------------------------------------------------------------------
# sound: style A, windows side by side on a halo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k, steps", [(1, 2000), (8, 260)])
def test_sound_on_windows_is_the_full_domain_to_the_bit(k, steps):
    """Air into water on four windows (fast-sound's layout at its original height):
    the windows equal the full domain bit for bit at every step, serial and threaded,
    the energy holds and the reflection reads the textbook's, with one leapfrog step
    per exchange and with eight."""
    from atlas.workbench.families import acoustics as Ac
    s = example_case("fast-sound")
    s.domain.ny = 40
    for r in s.regions:
        r.ny = 40
    for w in s.windows:
        w.ny = 40
    s.run.exchange_every = k
    s.run.macro_dt = 4.0e-6 * k
    s.run.steps = steps
    assert not [i for i in check(s) if i.severity == "error"]
    run = Ac.build(s, arms=("serial", "parallel", "full"), threads=3)
    st = {a: run.initial(a) for a in run.arms}
    hist = {a: [] for a in run.arms}
    for _ in range(steps):
        for a in run.arms:
            new = run.step(a, st[a])
            hist[a].append(run.observe(a, new, st[a]))
            st[a] = new
        assert run.bitwise_equal(st["serial"], st["full"])
        assert run.bitwise_equal(st["parallel"], st["full"])
    metrics, checks = run.compare(st, hist, {"steps": steps, "first_difference": None})
    run.close()
    by = {c.spec.key: c for c in checks}
    assert by["energy"].passed and by["closed_form"].passed and by["bitwise"].passed


# ---------------------------------------------------------------------------
# the header and the card
# ---------------------------------------------------------------------------


@pytest.fixture
def wb(tmp_path):
    import panel as pn
    pn.extension("tabulator")
    from atlas.workbench.app import Workbench
    return Workbench(cases_dir=str(tmp_path))


def test_the_header_has_fast_example_with_more_examples_under_it(wb):
    from atlas.workbench.app import FAST_LABEL
    m = wb.examples_menu
    assert m.name == FAST_LABEL and m.split
    for fid in ("conduction-2d", "elasticity-2d"):
        wb.type_sel.value = fid
        items = [it[1] for it in (m.items or [])]
        assert f"file:example:{FAST_EXAMPLES[fid]}" not in items     # the button's own
        m.clicked = FAST_LABEL                                       # the button's part
        assert wb.spec == example_case(FAST_EXAMPLES[fid])
        assert wb.case_key == FAST_EXAMPLES[fid]


def _record(key, speed, agreement, extra=None, check="reference"):
    spec = example_case(key)
    return {"family": spec.physics.family, "case_name": key, "threads": spec.run.threads,
            "case": {"coupling": {"style": spec.coupling.style},
                     "windows": [{} for _ in spec.windows]},
            "timing": {"parallel": {"speedup_vs_full": speed},
                       "full": {"speedup_vs_full": 1.0}},
            "checks": [{"key": check, "kind": "reference", "value": agreement}],
            "problem": extra or {"windows": len(spec.windows)}}


def test_the_agreement_is_read_from_the_familys_own_check():
    assert fast.agreement([{"key": "power", "value": 0.02}]) == 0.02
    assert fast.agreement([{"key": "reference", "value": 1e-12},
                           {"key": "reference_multirate", "value": 2e-4}]) == 2e-4
    sound = [{"key": "closed_form", "value": 3e-10, "passed": True},
             {"key": "bitwise", "value": None, "passed": True,
              "title": "The two pieces equal the full domain, bit for bit"}]
    assert fast.agreement(sound) == 0.0
    river = [{"key": "bitwise", "value": None, "passed": True,
              "title": "Threaded equals serial, bit for bit"}]
    assert fast.agreement(river) is None


def test_the_card_names_the_mechanism_the_bars_and_the_fixed_line():
    from atlas.workbench.runview import fast_card
    html_ = fast_card(_record("fast-farm", 4.57, 0.0256, check="power")).object
    assert "Faster because the 12 pieces run at once on 4 threads" in html_
    assert "meets both bars" in html_ and fast.FIXED_LINE in html_
    low = fast_card(_record("fast-farm", 2.4, 0.0256, check="power")).object
    assert "below the 3× bar" in low and fast.FIXED_LINE in low
    m = fast_card(_record("fast-heat", 5.5, 1.2e-4,
                          {"rate": 9, "slow_pieces": ["steel"], "fast_pieces": ["copper"]},
                          check="reference_multirate"))
    assert "the steel steps 9 times longer than the copper" in m.object
    assert "meets both bars" in m.object
    o3 = fast_card(_record("plate-hole", 0.01, 3e-12)).object
    from atlas.workbench.families import elasticity
    assert elasticity.FAST_LIMIT in o3.replace("&#x27;", "'")
    assert fast_card(_record("wall-2", 4.0, 1e-12)) is None          # not a fast example
