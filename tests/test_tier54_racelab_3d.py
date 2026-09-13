"""PoC 3 phase 4 -- the half-car in three dimensions, classical only.

These pin the three things phase 4 can get silently wrong: a projection that is
not a projection, a solver whose order is claimed rather than measured, and a
learned switch that is greyed out without saying why.
"""

from __future__ import annotations

import math
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np                                                  # noqa: E402
import pytest                                                       # noqa: E402
from scipy.fft import dctn, idctn                                   # noqa: E402

from atlas.cases import racelab3d as R3                             # noqa: E402

DEMO = os.path.join(HERE, "atlas", "demo_racelab")


# ---------------------------------------------------------------------------
# the solver
# ---------------------------------------------------------------------------


def test_the_divergence_and_the_gradient_compose_to_the_operator_inverted():
    """``div grad`` must BE the Laplacian whose eigenvalues the DCT inverts.

    If it is not, the projection is not a projection: it inverts one operator
    and applies another, and leaves a residual divergence behind that no amount
    of grid refinement removes.  The first version of this solver failed this
    and converged at first order because of it.
    """
    n, L = 8, 1.0
    s = R3.WindowNS3D(n=n, length=L)
    h = s.h
    rng = np.random.default_rng(1)
    p = rng.standard_normal((n, n, n))

    # div(grad p) with the solver's own staggered operators
    u = np.zeros((n + 1, n, n))
    v = np.zeros((n, n + 1, n))
    w = np.zeros((n, n, n + 1))
    u[1:-1] = (p[1:] - p[:-1]) / h
    v[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / h
    w[:, :, 1:-1] = (p[:, :, 1:] - p[:, :, :-1]) / h
    fd = s.divergence(u, v, w)

    spec = idctn(s._eig() * dctn(p, type=2, norm="ortho"),
                 type=2, norm="ortho")
    assert np.abs(fd - spec).max() < 1e-9, "the projection inverts the wrong operator"


def test_the_projection_removes_everything_a_neumann_solve_can():
    """A homogeneous-Neumann projection cannot remove NET FLUX, and asserting
    that it does is asserting the impossible.

    With zero normal velocity on every face the divergence goes to round-off.
    With a field that carries flux through the boundary, what is left is a
    CONSTANT -- the mean the compatibility condition forces out of the
    right-hand side -- and the fluctuation about it is at round-off. Both are
    checked, because the first alone would pass on a solver that quietly left
    an O(h) field behind and the second alone would hide a real leak.
    """
    s = R3.WindowNS3D(n=12, length=1.0)
    rng = np.random.default_rng(0)
    u, v, w = (rng.standard_normal(sh) for sh in s.shapes())
    assert s.divergence_norm(u, v, w) > 1.0
    u2, v2, w2, _p = s.project(u, v, w)
    d = s.divergence(u2, v2, w2)
    assert np.abs(d - d.mean()).max() < 1e-10, "more than a constant is left"
    assert abs(d.mean() - s.divergence(u, v, w).mean()) < 1e-10

    # flux-balanced: now there IS nothing a Neumann solve cannot remove
    u[0] = u[-1] = 0.0
    v[:, 0] = v[:, -1] = 0.0
    w[:, :, 0] = w[:, :, -1] = 0.0
    u3, v3, w3, _p = s.project(u, v, w)
    assert np.abs(s.divergence(u3, v3, w3)).max() < 1e-10


def test_the_projection_leaves_a_divergence_free_field_alone_to_second_order():
    """The measurement that condemned the collocated arrangement.

    A projection applied to an analytically divergence-free field should be
    nearly the identity, and the amount by which it is not must fall as
    ``h**2``.  On the collocated grid it fell as ``h``, which is what made the
    whole scheme first order while its divergence looked healthy.
    """
    err = []
    for n in (16, 32):
        s = R3.WindowNS3D(nu=0.05, length=1.0, n=n)
        f = R3.beltrami(s, 0.0)
        g = s.project(*f)[:3]
        num = math.sqrt(sum(float(np.mean((a - b) ** 2))
                            for a, b in zip(g, f)))
        den = math.sqrt(sum(float(np.mean(b ** 2)) for b in f))
        err.append(num / den)
    order = math.log(err[0] / err[1]) / math.log(2.0)
    assert order > 1.7, "the projection perturbs a solenoidal field at O(h)"


def test_the_beltrami_error_falls_with_resolution():
    """The control is an EXACT solution, so the whole step is under test."""
    a = R3.beltrami_error(n=16)
    b = R3.beltrami_error(n=32)
    assert b["rel_l2"] < a["rel_l2"]
    assert b["divergence"] < a["divergence"]
    order = math.log(a["rel_l2"] / b["rel_l2"]) / math.log(2.0)
    assert order > 1.0, "no better than first order"


def test_the_measured_order_is_recorded_and_not_rounded_up():
    """The module must carry what was MEASURED, not the interior operators'
    order.  Phase 4's solver converges between first and second order over the
    range measured and the table says so."""
    t = R3.BELTRAMI_ORDER
    assert t["projection_alone_order"] > 1.9
    assert max(t["order_between_successive"]) < 2.0
    assert len(t["n"]) == len(t["rel_l2"]) == len(t["divergence"])
    assert all(a > b for a, b in zip(t["rel_l2"][:-1], t["rel_l2"][1:]))


def test_a_window_need_not_be_a_cube():
    """A box that holds a half-car is long in x and short in z."""
    s = R3.WindowNS3D(n=(20, 12, 8), length=20.0)
    assert s.shapes() == ((21, 12, 8), (20, 13, 8), (20, 12, 9))
    u, v, w = s.zeros()
    u[:] = 1.0
    u2, v2, w2, _p = s.project(u, v, w)
    assert s.divergence_norm(u2, v2, w2) < 1e-10


# ---------------------------------------------------------------------------
# the half-car
# ---------------------------------------------------------------------------


def test_the_car_is_a_HALF_car_with_a_symmetry_plane():
    car = R3.half_car(ns=4, nz=4)
    z = np.array([p.centre[2] for p in car.panels])
    assert z.min() > 0.0, "a panel IN the symmetry plane has no wetted area"
    assert z.max() > 20.0, "the half-car has no width"


def test_the_half_car_has_the_parts_section_3_2_asks_for():
    """body, front wing with endplate, floor, diffuser, sidepod duct, wheels."""
    car = R3.half_car(ns=4, nz=4)
    groups = car.by_group()
    for want in ("body", "front-wing", "floor", "duct", "rear-wing",
                 "wheel", "endplate", "side"):
        assert groups.get(want, 0) > 0, "the half-car has no %s" % want
    ids = {p.body_id for p in car.panels}
    assert "FW_ENDPLATE" in ids
    assert {"WHEEL_F", "WHEEL_R"} <= ids


def test_the_wetted_area_converges_under_refinement():
    """A panelisation whose area moves with the knob is measuring the knob."""
    a = R3.half_car(ns=4, nz=4).wetted_area()
    b = R3.half_car(ns=16, nz=16).wetted_area()
    assert abs(b - a) / a < 0.05


def test_the_three_d_surface_is_quadratic_where_the_curve_was_linear():
    """**CS-19 section 7.4's [AI Inference], measured at last.**

    That page says the 3-D phase's cost is dominated by the stamping because a
    surface carries O(n^2) stations where a curve carries O(n), labels it an
    inference, and says phase 4 is the place to measure it.  This is that
    measurement, and it is a real one: the exponent is FITTED and the test
    would fail if the panel count were linear.
    """
    s = R3.station_scaling()
    e = s["fitted_exponent_of_the_3d_panel_count"]
    assert 1.8 <= e <= 2.2, "the panel count is not quadratic: %.3f" % e
    assert s["verdict"] == "CONFIRMED"
    counts = [r["stations_2d"] for r in s["rows"]]
    assert len(set(counts)) == 1, "the 2-D station count must not move"


# ---------------------------------------------------------------------------
# the march, and the learned switch
# ---------------------------------------------------------------------------


def test_the_march_runs_and_offers_no_learned_option():
    t = R3.Tiling3D()
    r = R3.march3d(steps=2, tiling=t)
    assert r["learned_option"] is None
    assert "Poseidon" in r["why_no_learned_option"]
    assert r["windows"] >= 2
    assert np.isfinite(r["u"]).all()
    assert r["history"][-1]["u_max"] < 10.0, "the 3-D march blew up"


def test_the_partition_of_unity_sums_to_one():
    t = R3.Tiling3D()
    w = t.weights()
    assert np.abs(w.sum(axis=0) - 1.0).max() < 1e-12


def test_the_resumable_march_advances_one_macro_step_at_a_time():
    m = R3.March3D()
    m.step()
    assert m.step_i == 1
    m.step()
    assert m.step_i == 2
    d = m.as_dict()
    assert d["learned_available"] is False
    assert d["panels"] > 100
    assert m.speed_slice().shape == (m.tiling.nx, m.tiling.ny)


def test_the_reason_names_the_checkpoint_and_the_open_decision():
    """A greyed-out control with no reason is the thing this project does not
    ship.  The reason must name WHY -- a 2-D operator at a fixed resolution --
    and must say the training route is an open decision rather than pretend it
    was impossible."""
    r = R3.NO_LEARNED_IN_3D
    assert "Poseidon-T" in r
    assert "128x128" in r
    assert "9.1" in r or "open decision" in r
    assert len(r) > 200


# ---------------------------------------------------------------------------
# the dashboard
# ---------------------------------------------------------------------------


def test_the_engine_marches_in_three_dimensions_and_drops_the_learned_option():
    from atlas.demo_racelab import engine as E
    e = E.Engine()
    e.post({"kind": "dims", "value": "3d"})
    e._drain()
    assert e.cfg.dims == "3d"
    e._march_once()
    e._publish()
    p = e.frame.payload
    assert p["dims"] == "3d"
    assert p["learned_available"] is False
    assert p["three_d"]["panels"] > 100
    assert "Poseidon-T" in p["three_d"]["why_no_learned_option"]
    assert len(e.frame.png) > 0


def test_the_page_carries_the_toggle_and_greys_the_switch_out_with_the_reason():
    html = open(os.path.join(DEMO, "static", "index.html"),
                encoding="utf-8").read()
    assert 'data-dims="3d"' in html, "no 3-D toggle"
    assert "why_no_learned_option" in html, "the reason is never drawn"
    assert "dimsnote" in html
    # the switch must be DISABLED in 3-D, not hidden
    assert "disabled = d3" in html or "disabled=d3" in html


def test_the_page_does_not_draw_the_2d_overlay_over_a_3d_slice():
    """Found by looking at it, and it is worse than W239 describes.

    The vector overlay is the 2-D car: fourteen 128x128 windows, the centreline
    bodies, the device planes.  Drawn over a z-slice of the half-car it
    superimposes two different objects that happen to render alike.  So the
    overlay is hidden in 3-D and the layout note is replaced -- and the note
    has to be RESTORED on the way back, which is the bug this pins: the note is
    set once from the meta message, so overwriting it in 3-D made it permanent
    and 2-D then described a decomposition it was not showing.
    """
    html = open(os.path.join(DEMO, "static", "index.html"),
                encoding="utf-8").read()
    assert 'S.layoutnote2d' in html, "the 2-D layout note is never remembered"
    assert html.count("S.layoutnote2d") >= 3, "stash, restore and reset"
    i3 = html.index('f.dims === "3d"')
    assert '$("svg").style.display = "none"' in html[i3:i3 + 1200]
    assert 'S.layoutnote2d) { $("layoutnote")' in html, "never restored"


def test_the_page_says_there_is_no_envelope_in_3d_rather_than_nothing():
    """A stamp carried over from the 2-D march would report on a column that is
    not running, and silence would read as a clean bill of health.  W238 is
    neither."""
    html = open(os.path.join(DEMO, "static", "index.html"),
                encoding="utf-8").read()
    assert "NO DECLARED ENVELOPE IN 3-D" in html
    assert "W238" in html


def test_the_demo_releases_from_the_traced_car_s_own_settled_field():
    """A settled field belongs to a geometry.

    The demo released the traced car from the HAND-DRAWN car's cache and
    stamped OUTSIDE THE MODEL from macro-step 6 for it -- correctly, which is
    how it was noticed.

    **This test then pinned the repair as a directory ORDER**, and the next car
    walked straight through it: when the user drew the car in Tier 55 the
    demo went on preferring ``out/racelab4`` -- the traced car's field -- for a
    car that was no longer traced, and this test went on passing.  An order
    says which cache is newest, not which car it belongs to.  So it pins the
    MATCH now: the engine asks for the car's fingerprint and keeps the answer
    of whether the field is this car's, and the fallback still says, in
    capitals, when it is not.  `test_tier57_racelab_bundle.py` exercises the
    mechanism on real files.
    """
    src = open(os.path.join(DEMO, "engine.py"), encoding="utf-8").read()
    i = src.index("def _release")
    body = src[i:i + 2000]
    assert "find_release(root, RL.geometry_fingerprint())" in body, \
        "the release is not matched to the car"
    assert 'self.notes["release_is_this_car_s"]' in body
    j = src.index("def find_release")
    fn = src[j:j + 3000]
    assert "DIFFERENT CAR" in fn, "the fallback does not say it is the wrong car"
    assert "fp == fingerprint" in fn


def test_a_two_d_engine_is_unchanged_by_any_of_this():
    """Phase 4 must not move phase 3.  The default is still 2-D."""
    from atlas.demo_racelab import engine as E
    assert E.RaceConfig().dims == "2d"
    e = E.Engine()
    assert e.cfg.dims == "2d"
    assert e._m3 is None
