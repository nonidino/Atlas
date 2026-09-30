"""Drawn shapes in every family (2026-09-29, the owner: "make the smooth domain/spline/
other stuff available for all cases").

Until this step only conduction ran on a drawn domain; the other seven families said
why not.  Each now runs on the cells a drawn shape contains, and these tests pin, per
family, what its own arithmetic promises on such a domain:

* electric: electrodes on drawn edges; the ring film's resistance against
  ``theta / (sigma t ln(ro / ri))``;
* transport and the cooled block: a flow SOLVED through the drawn shape
  (`flow.py`), divergence-free to round-off, crossing no bank and no wall;
* acoustics: rigid drawn walls, and two pieces of any shape equal to the full domain
  to the bit;
* elasticity and the heated arc: an element mask, clamps and loads on drawn edges;
* the wind farm: a solid held at rest by penalization, and the grid's edges keeping
  the solver's conditions;
* the one-window control, bit for bit, wherever the discretization is identical.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.workbench import fv  # noqa: E402
from atlas.workbench import geometry as geo  # noqa: E402
from atlas.workbench import shapes as S  # noqa: E402
from atlas.workbench.runner import adapter_for, arms_for  # noqa: E402
from atlas.workbench.spec import (EXAMPLES, Boundary, CaseSpec, Outline, Window,  # noqa: E402
                                  check, derive_boundaries, example_case)

DRAWN = ["ring-film", "river-bend", "sound-lens", "plate-hole", "bimetal-arc",
         "cooled-winding", "farm-hill"]


def _errors(spec):
    return [i.message for i in check(spec) if i.severity == "error"]


def _march(spec, steps, arms=None, threads=2):
    mod = adapter_for(spec.physics.family)
    arms = arms or arms_for(mod, spec)[0]
    r = mod.build(spec, arms=arms, threads=threads)
    st = {a: r.initial(a) for a in r.arms}
    hist = {a: [] for a in r.arms}
    same = True
    for _ in range(steps):
        for a in r.arms:
            prev = st[a]
            st[a] = r.step(a, prev)
            hist[a].append(r.observe(a, st[a], prev))
        if "parallel" in st:
            same &= r.bitwise_equal(st["serial"], st["parallel"])
    metrics, checks = r.compare(st, hist, {"steps": steps,
                                           "first_difference": None if same else 1})
    r.close()
    return r, st, hist, {c.spec.key: c for c in checks}


# ---------------------------------------------------------------------------
# every family, every drawn example
# ---------------------------------------------------------------------------


def test_every_family_that_runs_has_a_drawn_example():
    from atlas.workbench import registry
    families = {EXAMPLES[k].family for k in DRAWN} | {"conduction-2d"}
    assert families == set(registry.available_ids())
    for key in DRAWN:
        d = example_case(key).domain
        assert d.outline is not None or d.holes, key


@pytest.mark.parametrize("key", DRAWN)
def test_each_drawn_example_is_clean_and_passes_its_checks(key):
    s = example_case(key)
    assert _errors(s) == []
    assert CaseSpec.from_json(s.to_json()) == s
    steps = {"river-bend": 20, "sound-lens": 30, "bimetal-arc": 3, "farm-hill": 1}.get(key, 1)
    _r, _st, _h, checks = _march(s, steps)
    for c in checks.values():
        assert c.passed is not False, (key, c.as_dict())


# ---------------------------------------------------------------------------
# the one-window control, where the discretization is identical
# ---------------------------------------------------------------------------


def _one_window(key):
    s = example_case(key)
    s.layout = None
    s.windows = [Window(id="ALL", shape="cells",
                        runs=geo.mask_to_runs(geo.domain_mask(s.domain)))]
    return s


def test_one_window_over_a_drawn_river_is_the_full_river_to_the_bit():
    s = _one_window("river-bend")
    assert _errors(s) == []
    _r, st, _h, _c = _march(s, 15, arms=("serial", "full"))
    assert np.array_equal(st["serial"].u, st["full"].u)


def test_one_window_over_a_drawn_plate_is_the_full_plate_to_the_bit():
    s = _one_window("plate-hole")
    assert _errors(s) == []
    _r, st, _h, _c = _march(s, 1, arms=("serial", "full"))
    assert np.array_equal(st["serial"].u, st["full"].u)


# ---------------------------------------------------------------------------
# electric
# ---------------------------------------------------------------------------


def test_the_ring_films_resistance_is_close_to_its_closed_form():
    """Information, pinned loosely: the staircase's own error against the ring's
    continuum resistance theta / (sigma t ln(ro / ri)) is about 1% (the bend's heat
    flow: -0.97%)."""
    from atlas.workbench.families import electric as el
    s = example_case("ring-film")
    r = el.build(s, arms=("serial",))
    cur, _phi = r.plate_solve(np.array([1.0, 0.0]))          # E1 at 1 V, E2 at 0 V
    resistance = 1.0 / abs(cur[r.e_ids.index("E1")])
    closed = (math.pi / 2) / (1.0e5 * 3.0e-5 * math.log(100 / 40))
    assert abs(resistance - closed) / closed < 0.03
    assert set(r.faces) == {"E1", "E2"} and all(c.size > 50 for c, _g in r.faces.values())


# ---------------------------------------------------------------------------
# a solved flow: the river and the coolant
# ---------------------------------------------------------------------------


def _face_flow_through(f: fv.Field, kinds_closed=(fv.NO_FLUX, fv.FLUX, fv.FIXED)):
    """The advective flow through every boundary face of a closed kind."""
    _c, kind, _v, _g, fin = fv.boundary_faces(f)
    closed = np.isin(kind, kinds_closed)
    return fin[closed]


def test_the_rivers_flow_follows_its_banks():
    from atlas.workbench.families import plume as pl
    s = example_case("river-bend")
    flow = pl.river_flow(s)
    assert flow.residual < 1e-12                               # divergence-free, face by face
    f, _h = pl.field_from_case(s)
    assert np.all(_face_flow_through(f) == 0.0)                # nothing crosses a bank
    # the discharge per metre of the inlet's width, the inlet's width as drawn (200 m)
    assert flow.inflow == pytest.approx(1.0 * 200.0, rel=1e-9)
    act = geo.domain_mask(s.domain)
    assert np.all(f.fx[:, 1:-1][~(act[:, :-1] & act[:, 1:])] == 0.0)   # none on dry land


def test_the_coolants_flow_stays_in_its_channel():
    from atlas.workbench.families import cooling as co
    s = example_case("cooled-winding")
    assert not co.is_plug(s)
    wet = co.coolant_mask(s)
    f = co.field_from_case(s)
    # no flow through any face between the channel and the block
    both_x = wet[:, :-1] & wet[:, 1:]
    both_y = wet[:-1, :] & wet[1:, :]
    assert np.all(f.fx[:, 1:-1][~both_x] == 0.0) and np.all(f.fy[1:-1, :][~both_y] == 0.0)
    assert co.coolant_flow(s).residual < 1e-12
    # every watt the chip makes leaves in the water
    _r, st, hist, checks = _march(s, 1)
    assert checks["energy"].passed and hist["full"][-1]["carried_out_W"] == pytest.approx(
        2000.0, rel=1e-9)


def test_a_river_without_an_outlet_says_so():
    s = example_case("river-bend")
    s.boundaries = [b.model_copy(update={"kind": "bank"}) if b.kind == "river-outlet" else b
                    for b in s.boundaries]
    assert any("the river has no outlet" in m for m in _errors(s))


# ---------------------------------------------------------------------------
# acoustics
# ---------------------------------------------------------------------------


def test_drawn_walls_are_rigid_and_the_pieces_are_the_full_domain():
    from atlas.workbench.families import acoustics as ac
    s = example_case("sound-lens")
    r, st, hist, checks = _march(s, 120)
    assert checks["bitwise"].passed is True
    assert checks["energy"].passed and checks["energy"].value < 1e-12
    assert checks["closed_form"].passed is None                 # no textbook on a lens
    act = geo.domain_mask(s.domain)
    shut_x = ~(act[:, :-1] & act[:, 1:])
    assert np.all(st["full"].vx[:, 1:-1][shut_x] == 0.0)       # nothing moves into a wall
    assert np.all(st["full"].p[~act] == 0.0)
    assert ac.stable_dt(s) > s.run.macro_dt


# ---------------------------------------------------------------------------
# structures
# ---------------------------------------------------------------------------


def test_the_plate_with_a_hole_balances_and_its_hole_is_empty():
    from atlas.workbench.families import elasticity as el
    s = example_case("plate-hole")
    r, st, _h, checks = _march(s, 1, arms=("serial", "full"))
    assert checks["force"].passed and checks["reference"].passed
    hole = ~geo.domain_mask(s.domain)
    assert hole.sum() > 400                                    # a circle of radius 12 cells
    assert np.all(np.isnan(r.field(st["full"])[hole]))
    # the stress rises round the hole: the largest von Mises over the pull
    vm = r.field(st["full"])
    assert np.nanmax(vm) / 10.0 > 2.0


def test_the_heated_arc_splits_to_the_bit():
    s = example_case("bimetal-arc")
    _r, st, hist, checks = _march(s, 4)
    assert checks["reference"].passed is True and checks["lag"].passed is True
    assert checks["energy"].passed


# ---------------------------------------------------------------------------
# the wind farm
# ---------------------------------------------------------------------------


def test_the_farm_over_a_hill_keeps_its_ground_at_rest():
    s = example_case("farm-hill")
    kinds = {b.edge: b.kind for b in s.boundaries}
    assert kinds["outline:10"] == "inlet" and kinds["outline:8"] == "outlet"
    assert kinds["outline:9"] == "freestream"
    assert all(kinds[f"outline:{k}"] == "wall" for k in range(8))
    r, st, hist, checks = _march(s, 1, threads=2)
    assert checks["mass"].passed and checks["bitwise"].passed is True
    solid = ~geo.domain_mask(s.domain)
    for a in r.arms:
        assert np.all(st[a].u[solid] == 0.0) and np.all(st[a].v[solid] == 0.0)


def test_one_window_over_a_drawn_farm_is_its_full_domain_to_the_bit():
    """The N = 1 control on a drawn domain, in the arrangement where one window is the
    full-domain solver's arithmetic (embedded, blended).  A window's box is the window
    as drawn, solid included; boxed on its fluid cells alone, a window over the ground
    never held it, and this control failed by 0.61 U after one macro-step."""
    from atlas.workbench.families import windfarm as wf
    s = example_case("farm-hill")
    d = s.domain
    s.windows = [Window(id="W", x0=0, y0=0, nx=d.nx, ny=d.ny)]
    s.layout = None
    s.coupling.elliptic, s.coupling.assembly = "embedded", "blend"
    run = wf.build(s, arms=("serial", "full"), threads=1)
    try:
        a, b = run.initial("serial"), run.initial("full")
        for _ in range(2):
            a, b = run.step("serial", a), run.step("full", b)
            assert np.array_equal(a.u, b.u) and np.array_equal(a.v, b.v)
    finally:
        run.close()


@pytest.mark.parametrize("elliptic, assembly", [("embedded", "blend"),
                                                 ("exposed", "projected")])
def test_a_drawn_farm_that_is_the_whole_rectangle_is_the_plain_farm_to_the_bit(elliptic,
                                                                                 assembly):
    """The drawn machinery -- boxes blended by the mask partition of unity, penalized
    solvers -- adds nothing where there is no solid: both arms are the plain farm's,
    bit for bit.  So what six windows depart by over the hill is the hill's."""
    from atlas.workbench.families import windfarm as wf
    p = example_case("wake-array-3")
    p.coupling.elliptic, p.coupling.assembly = elliptic, assembly
    r = p.copy_deep()
    r.domain.outline = Outline.of(S.rectangle(0.0, 0.0, float(r.domain.nx),
                                              float(r.domain.ny)))
    derive_boundaries(r)
    runs = [wf.build(x, arms=("serial", "full"), threads=1) for x in (p, r)]
    try:
        assert runs[0].plain and not runs[1].plain
        arms = ("serial", "full")
        st = [[run.initial(a) for a in arms] for run in runs]
        for _ in range(2):
            st = [[run.step(a, x) for a, x in zip(arms, xs)] for run, xs in zip(runs, st)]
            for k in range(2):
                assert np.array_equal(st[0][k].u, st[1][k].u)
                assert np.array_equal(st[0][k].v, st[1][k].v)
    finally:
        for run in runs:
            run.close()


def test_a_farm_domain_that_misses_the_inlet_is_refused():
    s = example_case("farm-hill")
    d = s.domain
    pts = list(d.outline.points)
    pts[-1] = (24.0, float(d.ny))                                 # pull the left edge in
    pts[0] = (24.0, pts[0][1])
    d.outline = Outline(points=pts, edges=d.outline.edges, bulge=d.outline.bulge)
    derive_boundaries(s)
    assert any("no edge of the drawn domain runs along it" in m for m in _errors(s))


def test_a_farm_domain_that_misses_the_outlet_is_refused():
    """The page's first drawn farm touched the grid's right edge at one vertex: the
    flow had nowhere to leave, and the check said nothing (2026-09-29)."""
    s = example_case("farm-hill")
    d = s.domain
    pts = list(d.outline.points)
    right = [k for k, p in enumerate(pts) if p[0] == d.nx]
    assert len(right) == 2
    pts[right[0]] = (d.nx - 24.0, pts[right[0]][1])               # one vertex left on it
    d.outline = Outline(points=pts, edges=d.outline.edges, bulge=d.outline.bulge)
    derive_boundaries(s)
    msgs = _errors(s)
    assert any("the flow leaves on the grid's right edge" in m for m in msgs)
    assert not any("the flow enters on the grid's left edge" in m for m in msgs)
    assert not any("the flow leaves" in m for m in _errors(example_case("farm-hill")))


@pytest.mark.parametrize("key, verdict", [
    ("ring-film", "admit-uncertified"), ("river-bend", "admit-uncertified"),
    ("sound-lens", "admit-uncertified"), ("cooled-winding", "admit-uncertified"),
    ("plate-hole", "refuse"), ("s-channel", "refuse")])
def test_every_drawn_example_compiles_to_the_compilers_verdict(key, verdict):
    """Every seam probed through the family's own arithmetic on the drawn cells (the
    river's compile divided by its zero depth past the banks until this test); the
    iterated one-physics pieces refuse at R10 alone, as their rectangles do (W348)."""
    from atlas.workbench.compile import compile_case
    s = compile_case(example_case(key))
    assert s.refused_before is None and s.verdict == verdict, (s.verdict, s.other[:3])
    assert s.seams and all(sv.verdict in ("admit", "admit-uncertified") for sv in s.seams)
    if verdict == "refuse":
        assert [r["rule"] for r in s.other if r["verdict"] == "refuse"] == ["R10"]


def test_the_split_and_the_drawn_farm_are_refused_before_the_compiler():
    """The heated arc's bond is volumetric, as the strip's is; and the wind farm's
    graph is the rung's full rectangle of fluid, so a drawn farm is refused with that
    reason rather than judged as the plain farm (it was, until this test)."""
    from atlas.workbench.compile import compile_case
    s = compile_case(example_case("bimetal-arc"))
    assert s.verdict == "refuse" and s.refused_before.startswith("NamedHoleError")
    assert "VOLUMETRIC" in s.refused_before
    s = compile_case(example_case("farm-hill"))
    assert s.verdict == "refuse" and s.refused_before.startswith("CompileRefused")
    assert "a drawn farm has no declared graph yet" in s.refused_before


def test_derived_conditions_follow_the_geometry():
    """A family whose solver fixes its edges (the wind farm): a drawn edge along the
    grid's edge takes that edge's condition, and moving it off turns it into a wall."""
    s = example_case("farm-hill")
    assert derive_boundaries(s) is False                         # already derived
    s.boundaries = [b.model_copy(update={"kind": "wall"}) for b in s.boundaries]
    assert any("fixes it as inlet" in m for m in _errors(s))
    assert derive_boundaries(s) is True and _errors(s) == []


# ---------------------------------------------------------------------------
# a flux on a drawn edge is per unit length of the edge as drawn
# ---------------------------------------------------------------------------


def test_a_flux_on_a_slanted_edge_carries_its_true_length():
    """A diamond (a square turned 45 degrees): its staircase edges are sqrt 2 longer
    than the edges, so each face carries the flux times 1 / sqrt 2."""
    from atlas.workbench.families import conduction as cd
    s = example_case("bend-3")
    c, r_ = 54.0, 40.0
    s.domain.outline = Outline(points=[(c + r_, c), (c, c + r_), (c - r_, c), (c, c - r_)])
    s.windows = [Window(id="ALL", shape="curve", outline=s.domain.outline)]
    s.boundaries = [Boundary(id="in", edge="outline:0", kind="heat-flux", value=1000.0),
                    Boundary(id="a", edge="outline:1", kind="insulated"),
                    Boundary(id="out", edge="outline:2", kind="fixed-temperature",
                             value=300.0),
                    Boundary(id="b", edge="outline:3", kind="insulated")]
    f = cd.field_from_case(s)
    q = fv.boundary_inflow(f, np.full(f.n, 300.0))
    true_length = r_ * math.sqrt(2) * s.domain.dx                  # metres
    flux_in = sum(v for k, v in q.items() if v > 0)
    assert flux_in == pytest.approx(1000.0 * true_length, rel=0.02)
    assert geo.edge_lengths(s.domain)["outline:0"] == pytest.approx(r_ * math.sqrt(2),
                                                                    rel=1e-6)
    assert S.problems(s.domain.outline.points) == []
