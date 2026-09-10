"""Tier 39, W162 + W163 -- the topology the compiler could not see.

`atlas/graph.py`'s `directed_cycles`, `DeclaredLoopGain`, `cross_points`;
`atlas/compiler.py`'s `_r13_directed_cycle` and `_cross_point_provenance`.

**W163: the compiler could not tell a circuit from a chain.**  Compile
`cooling_loop` with the return seam and without it and the entire difference in
the decision record was the extra seam's own per-seam rows -- same verdict, same
rule set, same per-agent decisions -- while the measured consequence of the
cycle is 1.5 K of order dependence at one sweep per macro-step.  `powertrain`
reproduced it on unrelated physics.  The row was deliberately left open when
only one cyclic graph existed, because a rule invented the day its first graph
appears is a rule validated on one graph.

R13 is written here against both, and it says something DIFFERENT and true on
each, which is the test that it was not fitted to one:

  * CS-13's ADVEC loop carries a temperature and composes to a gain of 0.9238 --
    a contraction, with the radiator's 0.925 doing all the work.  Swept, it
    would converge from any start and the sweep order would stop mattering.
  * CS-14's ELEC loop carries a potential and composes to a gain of EXACTLY 1,
    structurally: each element maps ``V -> V - emf + I R``, a translation.
    Swept, it is refused -- and that refusal is right, and was already known:
    `CircuitSolve` solves the loop rather than sweeping it precisely because
    "a directed loop has no first element".  Kirchhoff's voltage law round a
    loop is a constraint, the potential has an arbitrary datum, and there is no
    isolated fixed point in it to reach.

**And R13 is scoped to schemes that iterate**, which is the defect the rule's
own first version had: it fired at L2 and refused `powertrain` for a scheme the
compiler had not chosen.  Both circuits compile to `DIRECT_SCHUR`, which is
order-free -- R5's own sentence -- so the gain does not bind, and the cycle is
still reported rather than passed over in silence.

**W162: a triangle in the adjacency is not a cross-point on a circuit.**  Three
legs joined by three DISTINCT planes are pairwise adjacent and share no point,
and `cooling_loop.build(n_legs=3)` was refused at `L2/I2/G1` for a multi-valued
shared cell that does not exist.  The graph had no way to say so: `cross_points`
defaulted to `()`, which is falsy, so a declaration of NONE was indistinguishable
from silence.
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import atlas.compiler as K                                        # noqa: E402
from atlas import compile_scheme                                  # noqa: E402
from atlas.cases import cooling_loop as C                         # noqa: E402
from atlas.cases import powertrain as P                           # noqa: E402
from atlas.graph import CaseGraph, DeclaredLoopGain                          # noqa: E402
from atlas.scheme import Accelerator                              # noqa: E402
from atlas.verdict import DecisionRecord                          # noqa: E402


def _r13(g, accelerator=None):
    """R13's rows for this graph, optionally against a FORCED accelerator."""
    r = compile_scheme(g)
    if accelerator is None:
        return [d for d in r.decisions.decisions if d.rule == "R13"]
    ctx = K._Context(graph=g, budget=K.Budget(), probe_budget=K.ProbeBudget(),
                     references={}, record=DecisionRecord(),
                     stamp=K.EnvelopeStamp(), holes=K.HoleLedger(),
                     probe_state="forced", depth=0)
    K._r13_directed_cycle(ctx, replace(r.scheme, accelerator=accelerator))
    return ctx.record.decisions


# ---------------------------------------------------------------------------
# 1. the detector, censused rather than assumed
# ---------------------------------------------------------------------------


def test_the_cycle_detector_fires_on_the_two_circuits_and_neither_control():
    assert C.build()[0].directed_cycles() == [("COLD", "PASS", "HOT", "RAD")]
    assert P.build()[0].directed_cycles() == [("BATT", "INV", "MGU", "BUS")]
    assert C.build(close_loop=False)[0].directed_cycles() == []
    assert P.build(close_loop=False)[0].directed_cycles() == []


def test_a_tiling_has_no_directed_cycle_and_that_is_why_the_proxy_works():
    """The discriminator, measured on every tiling in the package.

    A tiling orients its seams monotonically along the grid axes -- `xhi -> xlo`,
    `yhi -> ylo` -- so its digraph is a grid poset and is acyclic.  If any tiling
    here had a directed cycle the proxy would be firing on graphs the rule is
    not about, and the rule would need a different detector.
    """
    import numpy as np

    from atlas.cases import front_wing as FW
    from atlas.cases import ground_effect as GE
    from atlas.cases import thermal_seam as TS
    from atlas.cases import wing_fsi as WF

    p = None
    for name in ("w141", "w136"):
        q = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "out", name, "settled.npz")
        if os.path.isfile(q):
            p = q
            break
    if p is None:
        pytest.skip("no settled field on disk")
    d = np.load(p)
    u, v = d["u"], d["v"]

    for label, g in (("front_wing", FW.build(u, v, motion=False)[0]),
                     ("ground_effect", GE.build(u, v)[0]),
                     ("wing_fsi", WF.build(u, v)[0]),
                     ("thermal_seam", TS.build()[0])):
        assert g.directed_cycles() == [], (label, g.directed_cycles())


def test_a_cycle_is_matched_by_SET_because_it_has_no_first_agent():
    """The declaration and the detection may disagree about the rotation, and
    that disagreement is the property the whole rule is about."""
    g, _e = C.build()
    cycle = g.directed_cycles()[0]
    assert g.declared_loop_gain(cycle) is not None
    rotated = cycle[2:] + cycle[:2]
    assert rotated != cycle
    assert g.declared_loop_gain(rotated) is g.declared_loop_gain(cycle)
    assert g.declared_loop_gain(("PASS", "HOT")) is None


# ---------------------------------------------------------------------------
# 2. the two graphs, and the rule says something different on each
# ---------------------------------------------------------------------------


def test_CS13s_loop_contracts_and_the_declaration_is_composed_not_written_out():
    g, _e = C.build()
    dec = g.declared_loop_gain(g.directed_cycles()[0])
    assert dec is not None and dec.contracts
    #: the declaration IS the solver's own product, so the two cannot drift
    assert dec.gain == pytest.approx(C.LoopSolve().loop_gain, rel=1e-12)
    assert dec.gain == pytest.approx(0.9238, abs=1e-4)

    rows = _r13(g, Accelerator.RICHARDSON)
    assert len(rows) == 1 and rows[0].verdict.value == "admit"
    assert "is a contraction" in rows[0].message
    #: and it does not overclaim -- rho < 1 is about the LIMIT, not about one sweep
    assert "does NOT certify is a finite-sweep answer" in rows[0].message
    assert "1.5 K" in rows[0].message


def test_CS14s_loop_gain_is_exactly_one_and_swept_it_is_REFUSED():
    """The second graph, and it does not agree with the first.

    A rule that said the same thing on both would not have been tested by the
    second one.  This one refuses, and the refusal reproduces a conclusion
    `powertrain` had already reached by hand.
    """
    g, _e = P.build()
    dec = g.declared_loop_gain(g.directed_cycles()[0])
    assert dec is not None and not dec.contracts
    assert dec.gain == pytest.approx(1.0, abs=1e-12)

    rows = _r13(g, Accelerator.RICHARDSON)
    assert len(rows) == 1 and rows[0].verdict.value == "refuse"
    assert "not a contraction" in rows[0].message
    assert "no isolated fixed point" in rows[0].message


def test_CS14s_gain_is_structural_and_survives_changing_every_element():
    """Exactly 1 whatever the emfs and resistances are, because each element's
    declared map is a TRANSLATION in the potential the seam carries.  If this
    ever stops being one, an element has stopped being affine in the potential
    and the declaration has to be re-derived rather than re-typed."""
    els = dict(P.make_elements())
    base = P.composed_loop_gain(els, P.CIRCUIT_ORDER)
    assert base == pytest.approx(1.0, abs=1e-12)
    for name in P.CIRCUIT_ORDER:
        els[name].resistance *= 7.0
    assert P.composed_loop_gain(els, P.CIRCUIT_ORDER) == pytest.approx(
        1.0, abs=1e-12)


def test_CS13s_gain_does_not_depend_on_the_operating_point():
    """`a` is the coefficient of the inlet temperature and the block heat enters
    through `b`, so one number is the gain at every operating point.  That is
    what makes it declarable on the GRAPH rather than only measurable on a run.
    """
    s = C.LoopSolve()
    for q in (0.0, 100.0, 5.0e3, 5.0e4):
        g = 1.0
        for n in s.order:
            g *= s.legs[n].coeffs(q if n == "PASS" else 0.0)[0]
        assert g == pytest.approx(s.loop_gain, rel=1e-12)


# ---------------------------------------------------------------------------
# 3. the scoping, which is the rule's own first defect
# ---------------------------------------------------------------------------


def test_a_direct_solve_does_not_sweep_so_the_gain_does_not_bind():
    """**The rule's first version refused `powertrain` for a scheme nobody chose.**

    Both circuits compile to `DIRECT_SCHUR`, which solves the assembled
    interface system in one shot: no first agent, no sweep order, no fixed-point
    sequence.  R5 already says a direct Schur solve is order-free, and this is
    that sentence reaching the case R5 was written before anyone had.
    """
    for build in (C.build, P.build):
        g, _e = build()
        r = compile_scheme(g)
        assert r.scheme.accelerator is Accelerator.DIRECT_SCHUR
        rows = [d for d in r.decisions.decisions if d.rule == "R13"]
        assert len(rows) == 1
        assert rows[0].verdict.value == "admit"
        assert rows[0].evidence.get("binding") is False
        assert "does not sweep it" in rows[0].message
        #: and it does not pretend the cycle is gone
        assert "does NOT clear is the cycle" in rows[0].message


def test_powertrain_is_not_refused_as_declared():
    """The corollary, asserted separately because it is a whole-graph verdict:
    the refusal above is conditional on a scheme this graph does not use, so
    `powertrain` as declared must not be refused by R13."""
    r = compile_scheme(P.build()[0])
    assert "R13" not in {d.rule for d in r.decisions.refusals}


# ---------------------------------------------------------------------------
# 4. the defect the row actually opened
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("build", [C.build, P.build])
def test_the_record_now_differs_between_the_circuit_and_the_chain(build):
    """W163's complaint, restated as its test.

    Before R13 the only difference between the closed and open compiles was the
    extra seam's own per-seam rows.  Now there is a GRAPH-level rule that fires
    on one and not the other, which is what "the compiler can tell a circuit
    from a chain" means operationally.
    """
    graph_subjects = ("<graph>", "<assembly>", "<run>")

    def graph_rules(g):
        return {f"{d.layer}/{d.rule}" for d in compile_scheme(g).decisions
                if d.subject in graph_subjects}

    closed = graph_rules(build()[0])
    opened = graph_rules(build(close_loop=False)[0])
    assert "L5/R13" in closed
    assert "L5/R13" not in opened
    assert closed - opened == {"L5/R13"}


def test_an_undeclared_cycle_decertifies_rather_than_being_guessed():
    """The gain cannot be probed -- `boundary_response` is a same-port DtN map
    and a cycle's gain is a cross-port transfer -- so a graph that declares none
    gets a decertification naming the quantity, not an assumption."""
    g, _e = C.build()
    g = replace(g, loop_gains=())
    rows = _r13(g, Accelerator.RICHARDSON)
    assert len(rows) == 1
    assert rows[0].verdict.value == "admit-uncertified"
    assert "declares no loop gain" in rows[0].message
    assert "cannot measure it" in rows[0].message


def test_a_declared_gain_above_one_is_refused_too():
    """The rule has three outcomes and the third needs a subject.  A gain above
    one is a diverging iteration, which is a different failure from unit gain's
    missing fixed point and gets the same verdict."""
    g, _e = C.build()
    cycle = g.directed_cycles()[0]
    g = replace(g, loop_gains=(DeclaredLoopGain(
        agents=cycle, gain=1.4, source="synthetic: a loop with a gain stage"),))
    rows = _r13(g, Accelerator.RICHARDSON)
    assert len(rows) == 1 and rows[0].verdict.value == "refuse"
    assert "1.4" in rows[0].message


# ---------------------------------------------------------------------------
# 5. W162 -- the cross-point that is not there
# ---------------------------------------------------------------------------


def test_a_graph_can_now_declare_that_it_has_NO_cross_points():
    """Three states, where there used to be two.  `()` was falsy and fell
    through to detection exactly as an absent declaration did, so a graph could
    say WHICH cross-points it had and could not say it had none."""
    g, _e = C.build()
    assert g.cross_points == ()
    assert g.cross_points_declared
    assert g.detected_cross_points() == []

    silent = replace(g, cross_points=None)
    assert not silent.cross_points_declared


def test_the_three_leg_circuit_is_no_longer_refused_for_a_point_it_does_not_have():
    """W162's reproduction, and its repair.

    Three legs pairwise adjacent is an adjacency TRIANGLE with no shared vertex:
    the three seams are three distinct planes at three distinct places along the
    circuit.  `L2/I2/G1` refused it for a multi-valued shared cell that does not
    exist.
    """
    g, _e = C.build(n_legs=3)
    #: the triangle is still there in the adjacency -- nothing was hidden
    adjacency = {a.agent_id: set(g.neighbours(a.agent_id)) for a in g.agents}
    legs = [n for n in ("PASS", "HOT", "RAD", "COLD") if n in adjacency]
    assert len(legs) == 3
    assert all(b in adjacency[a] for a in legs for b in legs if a != b)
    #: and the graph's declaration is what stops it being read as geometry
    assert g.detected_cross_points() == []
    r = compile_scheme(g)
    assert "I2/G1" not in {d.rule for d in r.decisions.refusals}
    assert r.verdict.value != "refuse", [
        f"{d.layer}/{d.rule}" for d in r.decisions.refusals]


def test_the_cross_point_refusal_names_its_proxy_as_a_proxy():
    """A `refuse` that cannot be told it is wrong about its own subject is the
    failure W162 is about.  Undeclared, the message has to say the subject was
    INFERRED and how, and name the escape."""
    g, _e = C.build(n_legs=3)
    g = replace(g, cross_points=None)
    assert g.detected_cross_points() == [("COLD", "PASS", "RAD")]
    r = compile_scheme(g)
    ref = [d for d in r.decisions.refusals if d.rule == "I2/G1"]
    assert len(ref) == 1
    msg = ref[0].message
    assert "inferred from the ADJACENCY and not declared" in msg
    assert "This is a PROXY" in msg
    assert "cross_points=()" in msg


def test_a_DECLARED_cross_point_still_refuses_and_says_it_was_declared():
    """The control.  A rule with nothing to fire on is not a rule, and the
    escape must not have switched the refusal off -- only told it where its
    subject came from."""
    g, _e = C.build(n_legs=3)
    g = replace(g, cross_points=("junction",))
    r = compile_scheme(g)
    ref = [d for d in r.decisions.refusals if d.rule == "I2/G1"]
    assert len(ref) == 1
    assert "from the graph's own `cross_points` declaration" in ref[0].message
    assert "This is not a proxy" in ref[0].message


# ---------------------------------------------------------------------------
# 6. the detector's own cost, which is factorial
# ---------------------------------------------------------------------------


class _FakeConn:
    def __init__(self, a, b):
        self.a = (a, "p")
        self.b = (b, "q")


class _FakeGraph:
    name = "synthetic-complete-digraph"

    def __init__(self, conns):
        self.connections = conns


def _complete_digraph(n):
    names = [f"A{i:02d}" for i in range(n)]
    return _FakeGraph([_FakeConn(a, b) for a in names for b in names if a != b])


def test_the_cycle_detector_is_BOUNDED_because_the_count_is_factorial():
    """**Found by stress-testing the detector, not by reading it.**

    The number of elementary cycles in a digraph is factorial in the worst
    case, and on a complete digraph `directed_cycles` did not return at ten
    nodes within two minutes. Every graph in this package is sparse and the real
    cost is microseconds -- but a compiler that hangs on a declaration is the
    worst failure mode available, and it is the shape where a blow-up reads as a
    hang rather than as a blow-up.

    The bound is a **visit budget, not a cap on the output**, and the
    distinction is the whole of it: a cap would return a silent subset, and a
    rule reasoning over some of a graph's cycles is silently reasoning about a
    different graph.
    """
    from atlas.graph import CycleEnumerationBudget

    #: small complete digraphs still enumerate exactly
    assert len(CaseGraph.directed_cycles(_complete_digraph(4))) == 11
    assert len(CaseGraph.directed_cycles(_complete_digraph(5))) == 26

    #: and a big one raises rather than running away
    with pytest.raises(CycleEnumerationBudget) as e:
        CaseGraph.directed_cycles(_complete_digraph(12))
    msg = str(e.value)
    assert "search steps" in msg
    assert "PARTIAL list is worse than none" in msg

    #: the budget is a parameter, so raising it deliberately is available and
    #: the default is not a hidden ceiling
    small = _complete_digraph(6)
    assert CaseGraph.directed_cycles(small, max_steps=10_000)
    with pytest.raises(CycleEnumerationBudget):
        CaseGraph.directed_cycles(small, max_steps=5)


def test_every_real_graph_is_far_inside_the_budget():
    """The bound must not be load-bearing on anything that actually exists,
    or it is a ceiling rather than a backstop."""
    for build in (C.build, P.build):
        g, _e = build()
        #: a budget of 500 steps is 400x smaller than the default and still
        #: enumerates these graphs completely
        assert g.directed_cycles(max_steps=500) == g.directed_cycles()


def test_R13_decertifies_rather_than_propagating_the_budget_error():
    """A rule that lets an internal budget escape as an exception has turned a
    declaration into a crash. R13 catches it and says what it could not check."""
    g, _e = C.build()

    class _Blown:
        """A graph whose cycle enumeration always exceeds its budget."""
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def directed_cycles(self, max_steps=200_000):
            from atlas.graph import CycleEnumerationBudget
            raise CycleEnumerationBudget("synthetic: 999 search steps")

    ctx = K._Context(graph=_Blown(g), budget=K.Budget(),
                     probe_budget=K.ProbeBudget(), references={},
                     record=DecisionRecord(), stamp=K.EnvelopeStamp(),
                     holes=K.HoleLedger(), probe_state="forced", depth=0)
    r = compile_scheme(g)
    K._r13_directed_cycle(ctx, replace(r.scheme,
                                       accelerator=Accelerator.RICHARDSON))
    rows = ctx.record.decisions
    assert len(rows) == 1
    assert rows[0].verdict.value == "admit-uncertified"
    assert "could not be enumerated" in rows[0].message
    #: and it does not overclaim -- a budget is not evidence of many cycles
    assert "not evidence that the graph HAS many cycles" in rows[0].message
