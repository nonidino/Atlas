"""Tier 79 -- R0 complete: all seven rocket agents on real physics.

Tier 76 wired two.  This wires `a`, `e`, `d`, `f` and `g`, so no agent in the
rocket graph is a seeded random matrix any more.

The tests pin six things, and four of them are findings rather than features:

* **the motion classes**, because the first version of the builder declared every
  port STATIC and the nine `InterfaceMotion` refusals silently vanished -- the
  "declare it away" failure in its purest form;
* **W309**, that `grid.Block` face normals are index-oriented, which is why every
  gas seam needs `effort_normal` and why Tier 77's b-c was not special;
* **W311**, the second undeclared interface -- the nozzle wall;
* **the g-f ADVEC operator is identically zero**, which is a declaration error
  with an exact cause rather than a hard interface problem;
* **W312**, that a plane port's response needs a cadence a wall's does not, which
  is the correction to Tier 76's cadence justification;
* **W313**, that no case in this vault sets `Prolongation.nondim_diag`.

The declaration tests run at ``dt_scale = 1e-3``. They decide DECLARATIONS, not
physical numbers, so the cadence is chosen for cost and said out loud -- and
§13.2 is exactly about not confusing the two.
"""
from __future__ import annotations

import numpy as np
import pytest

from atlas.cases import rocket, rocket_experts as RE
from atlas.capability import MotionClass
from atlas.compiler import compile_scheme
from atlas.probe import support_reach

DT_SCALE = 1.0e-3


def _have_expert() -> bool:
    try:
        RE.load_rocket_modules()
        return True
    except Exception:
        return False


needs_expert = pytest.mark.skipif(
    not _have_expert(), reason="the build repo is not importable; set ATLAS_BUILD_REPO")


# ===========================================================================
# 1. R0's completion
# ===========================================================================


@needs_expert
def test_every_agent_is_backed_by_a_real_solver():
    """R0's bar: seven agents, no seeded random matrices."""
    graph, experts = RE.build_rocket_real(dt_scale=DT_SCALE)
    assert sorted(a.agent_id for a in graph.agents) == ["a", "b", "c", "d", "e", "f", "g"]
    for a in graph.agents:
        fn = a.capabilities.boundary_response
        assert fn is not None, a.agent_id
        # `linear_response` closures are identifiable by qualname; none here.
        assert "linear_response.<locals>" not in (getattr(fn, "__qualname__", "") or ""), \
            f"{a.agent_id} is still a fixture"
    assert len(graph.connections) == 7


@needs_expert
def test_every_seam_probes_and_the_null_check_actually_runs():
    """`rocket.py`'s own connections declare no `expected_null_dim`, so
    `excess_null_directions` is None right across that graph and the check that
    caught a real declaration error at W2 is not running. This graph declares it."""
    graph, _ex = RE.build_rocket_real(dt_scale=DT_SCALE)
    for c in graph.connections:
        assert c.expected_null_dim == 0, c.seam_id
    fixture = rocket.build(declarations="fixture")
    assert all(c.expected_null_dim is None for c in fixture.connections)


# ===========================================================================
# 2. the mistake that mattered most
# ===========================================================================


@needs_expert
def test_the_moving_ports_still_refuse():
    """**The first builder declared every port STATIC and the InterfaceMotion
    refusals vanished.** The combustion front and the plume boundary are the one
    genuine research hole in this graph; a default that removes them has removed
    the product. Carried from `rocket.AGENTS` and pinned here."""
    graph, _ex = RE.build_rocket_real(dt_scale=DT_SCALE)
    moving = {p.name: p.motion_class for a in graph.agents
              for p in a.capabilities.ports
              if p.motion_class is not MotionClass.STATIC}
    assert len(moving) == 3, moving
    assert MotionClass.SOLUTION_DEPENDENT in moving.values()   # the combustion front
    assert MotionClass.PRESCRIBED in moving.values()           # the plume boundary
    r = compile_scheme(graph)
    im = [d for d in r.decisions.refusals if d.rule == "InterfaceMotion"]
    assert len(im) == 3, [d.subject for d in im]


# ===========================================================================
# 3. W309 -- the normals, and the rule that follows
# ===========================================================================


@needs_expert
def test_W309_block_face_normals_are_index_oriented_not_outward():
    """Both i-faces point +z and both j-faces share a direction, so on every
    gas-gas seam the two sides report against ONE shared direction -- which is
    why they all need `effort_normal`."""
    for aid in ("a", "b", "e", "d", "f", "g"):
        blk = RE.agent_blocks(aid)[-1]
        ni0, ni1 = blk.n_i[0].mean(axis=0), blk.n_i[-1].mean(axis=0)
        nj0, nj1 = blk.n_j[:, 0].mean(axis=0), blk.n_j[:, -1].mean(axis=0)
        assert float(ni0 @ ni1) > 0.0, aid      # NOT opposite: not outward normals
        assert float(nj0 @ nj1) > 0.0, aid


@needs_expert
def test_W309s_rule_reproduces_the_seam_tier_77_measured_independently():
    """The rule is *name the agent for which the seam is its imax/jmax face*. It
    was derived from the normals; b-c was derived a tier earlier from the
    geometry of a wall. They agree, which is what makes it a rule."""
    assert RE.EFFORT_NORMAL_ROCKET["b-c"] == "b" == rocket.EFFORT_NORMAL["b-c:THERM"]
    graph, _ex = RE.build_rocket_real(dt_scale=DT_SCALE)
    assert all(c.effort_normal for c in graph.connections)


# ===========================================================================
# 4. W311 -- the second undeclared interface
# ===========================================================================


@needs_expert
def test_W311_the_nozzle_wall_is_a_shared_boundary_no_edge_declares():
    """`e` and `c` share the diverging nozzle's inner wall and no edge carries
    it. `contours.plane_d_g`'s own docstring records a second, d-f. So
    `domains.verify` checks the agents PARTITION the box and that every declared
    edge lies on both its agents' boundaries -- and never the converse."""
    import importlib

    RE.load_rocket_modules()
    dom = importlib.import_module("atlas_build_solvers.geometry.domains")
    contours = importlib.import_module("atlas_build_solvers.geometry.contours")
    geo = RE.rocket_config().geometry
    z = np.array([0.55])                       # between the throat and the exit
    h = float(contours.nozzle_inner(z, geo)[0])
    assert bool(dom.contains("e", z, np.array([h * 0.99]), geo)[0])
    assert bool(dom.contains("c", z, np.array([h * 1.0005]), geo)[0])
    declared = {(e.src, e.dst) for e in RE.rocket_config().edge_list}
    assert ("e", "c") not in declared and ("c", "e") not in declared
    assert "e-c" in RE.UNDECLARED_INTERFACES and "d-f" in RE.UNDECLARED_INTERFACES


@needs_expert
def test_W311_the_undeclared_nozzle_wall_carries_MORE_heat_than_the_declared_one():
    """The consequence, and the prediction that it would be >2x was REFUTED --
    measured 1.47x. Material, and smaller than claimed."""
    experts = RE.make_rocket_experts(dt_scale=DT_SCALE)
    h_chamber = float(np.mean(experts["b"].wall_h(neighbour="c")))
    h_nozzle = float(np.mean(experts["e"].wall_h(neighbour="c")))
    assert h_nozzle > h_chamber
    assert 1.2 < h_nozzle / h_chamber < 2.0, (h_nozzle, h_chamber)


# ===========================================================================
# 5. the empty operator, and the cadence
# ===========================================================================


@needs_expert
def test_the_shear_layer_ADVEC_operator_is_identically_zero():
    """`rocket.py` declares ADVEC on g-f; the config says `(heat, fluid)`; and
    the shear layer's normal is exactly +y while the flow is +z, so the declared
    flow is zero for ANY trace. An interface problem that is EMPTY rather than
    hard -- CASE-STUDY-GUIDE mistake 6, with an exact cause."""
    f = RE.GasAgent("f", dt=float(RE.rocket_config().dt_model["f"]) * DT_SCALE)
    blk = RE.agent_blocks("f")[0]
    assert np.allclose(blk.n_j[:, -1].mean(axis=0), [0.0, 1.0], atol=1e-12)
    base = f.base_trace("g")
    f0 = np.asarray(f.respond("g:ADVEC", base), float)
    d = np.zeros(base.size)
    d[base.size // 2] = 1.0
    f1 = np.asarray(f.respond("g:ADVEC", base + 1.0e3 * d), float)
    # Measured RELATIVE to the flow's own mass flux, not against an absolute
    # bound: a first version of this test used 1e-8 and read 1.05e-8, which
    # says more about the number 1e-8 than about the operator. The scale that
    # matters is rho|u| through the plume, which is what a NON-empty ADVEC
    # response would be a fraction of.
    scale = float(np.abs(f.respond("e:ADVEC", f.base_trace("e"))).mean())
    assert scale > 1.0, scale                       # the reference is real
    assert np.abs(f1 - f0).max() / scale < 1e-9, (np.abs(f1 - f0).max(), scale)


@needs_expert
def test_W312_a_wall_converges_at_any_cadence_and_a_plane_does_not():
    """**The correction to Tier 76's cadence justification.** It was measured on
    a WALL, whose response is algebraic in the trace, and over-generalised to
    planes, whose response needs the wave to cross cells."""
    cfg = RE.rocket_config()

    def peak(aid, nb, scale):
        g = RE.GasAgent(aid, dt=float(cfg.dt_model[aid]) * scale)
        base = g.base_trace(nb)
        step = 1.0e-3 * float(np.abs(base).mean())
        d = np.zeros(base.size)
        d[base.size // 2] = 1.0
        f0 = np.asarray(g.respond(nb + ":X", base), float)
        f1 = np.asarray(g.respond(nb + ":X", base + step * d), float)
        return float(np.abs((f1 - f0) / step).max())

    wall_lo, wall_hi = peak("b", "c", 1e-3), peak("b", "c", 1e-2)
    plane_lo, plane_hi = peak("e", "f", 1e-3), peak("e", "f", 1e-2)
    # the wall is flat over the same range the plane moves by an order
    assert abs(wall_hi - wall_lo) / wall_lo < 1e-3, (wall_lo, wall_hi)
    assert plane_hi / plane_lo > 3.0, (plane_lo, plane_hi)


@needs_expert
def test_W301_closed_neither_wall_reads_the_saturation_signature():
    """Until Tier 86 this pinned W301's signature on ONE of two walls: the
    chamber's (exact_zeros == n - 1, the clamp active) and not the atmosphere's
    -- same BC, same solver, but the atmosphere sits at 255.7 K against a
    255.7 K wall, far above the clamp threshold (T_i + 20)/2 = 138 K. That was
    what showed W301 was about `wall_noslip` and not about the solver. The wall
    face's conduction is now one-sided from T_wall, so the chamber's wall
    transmits too, and neither reads the signature."""
    experts = RE.make_rocket_experts(dt_scale=DT_SCALE)
    b, d = experts["b"], experts["d"]
    sr_b = support_reach(b.respond, "c:THERM", b.base_trace("c"))
    sr_d = support_reach(d.respond, "c:THERM", d.base_trace("c"))
    assert sr_b.exact_zeros < sr_b.n - 1, "the chamber wall no longer saturates"
    assert sr_d.exact_zeros < sr_d.n - 1, "and the atmosphere's never did"


# ===========================================================================
# 6. W313 -- the units nobody converts
# ===========================================================================


def test_W313_no_case_in_this_vault_sets_the_declared_nondim():
    """`Prolongation.nondim_diag` is specified by interface-transfer-theory §6 as
    the diagonal unit conversion, and nothing sets it -- so every beta here is in
    raw physical units and is not comparable across port types."""
    import glob
    import os
    import re

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hits = []
    for path in glob.glob(os.path.join(root, "atlas", "cases", "*.py")):
        with open(path, encoding="utf-8") as fh:
            if re.search(r"nondim_diag\s*=", fh.read()):
                hits.append(os.path.basename(path))
    assert hits == [], hits


@needs_expert
def test_the_two_agent_b_classes_are_one_agent():
    """**Tier 76 built `ChamberGasAgent`; Tier 79 built a general `GasAgent`.
    Both are now in the module and both claim to be agent `b`.**

    If they disagreed, Tier 79's b-c number would not be comparable with Tier
    76-78's and the module would hold two different agents wearing one name.
    They agree to the BIT, which is what lets the earlier tiers' numbers carry
    over -- and is why `ChamberGasAgent` is kept rather than deleted: every
    number from Tiers 76, 77 and 78 was measured through it.
    """
    dt = 1.0e-6
    old = RE.ChamberGasAgent(dt=dt)
    new = RE.GasAgent("b", dt=dt)
    assert old._blk.shape == new._blk.shape
    assert old.n_seam == new.n_face("c")
    b_old, b_new = old.base_trace(), new.base_trace("c")
    assert np.allclose(b_old, b_new)
    r_old = np.asarray(old.respond("c:THERM", b_old), float)
    r_new = np.asarray(new.respond("c:THERM", b_new), float)
    assert np.array_equal(r_old, r_new), np.abs(r_new - r_old).max()
