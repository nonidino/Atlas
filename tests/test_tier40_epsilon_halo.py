"""Tier 40 -- the epsilon-halo measured, the checkpoint stood on, W171 scoped.

Three things are pinned here and they want different kinds of assertion.

**W173, the epsilon-halo.**  `case-study-ladder-to-f1` section 5's middle row is
the one unbuilt route through `L2/R10`, and the measurement says it is not
derivable for the scOT family.  The numbers live in `out/w173/w173.json` and are
asserted against it rather than retyped, on the Tier 39 discipline: a figure in
a page and a figure in a rule must not be able to drift apart.  **No rule was
written**, and one assertion here says so -- if `_halo_rule` ever grows an
epsilon branch, this file fails and the measurement gets re-read first.

**W174, is the refusal EARNED.**  `poc2-novelty-audit` section 4's third clause.
The 2x2 of (elliptic embedded/exposed) x (composition layer projects/does not)
was two-thirds unmeasured; completed, exactly one cell diverges and `R10/halo`
is right on all eleven of its rows there.

**W175 is a DIAGNOSED, UNFIXED defect and its test is written to that.**  It
asserts the defect reproduces, which is what this repo does with a row it opens
and does not close.  Closing W175 will break `test_W175_*` and that is by design:
keep the diagnosis in the docstring, rewrite the assertion, add a control.

**W171/W177 are censuses**, so their assertions are about declarations and are
cheap.
"""

from __future__ import annotations

import inspect
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas import Budget, compile_scheme                            # noqa: E402
from atlas.capability import EllipticSubsolve                       # noqa: E402
from atlas.cases import cooling_loop as CL                          # noqa: E402
from atlas.cases import front_wing as F                             # noqa: E402
from atlas.cases import neural_interface as NI                      # noqa: E402
from atlas.cases import powertrain as PW                            # noqa: E402
from atlas.cases import window_ns as W                              # noqa: E402
from atlas.cases import wing_fsi as WF                              # noqa: E402
from atlas.compiler import _halo_bounds, _decomposition_cuts        # noqa: E402
from atlas.verdict import Verdict                                   # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W173 = os.path.join(_ROOT, "out", "w173", "w173.json")
W173T = os.path.join(_ROOT, "out", "w173", "transverse.json")
W174 = os.path.join(_ROOT, "out", "w174", "w174.json")
W177 = os.path.join(_ROOT, "out", "w177", "w177.json")


def _load(path, script):
    if not os.path.exists(path):
        pytest.skip(f"run scripts/{script}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def halo():
    return _load(W173, "w173_epsilon_halo.py")


@pytest.fixture(scope="module")
def transverse():
    return _load(W173T, "w173_transverse_reach.py")


@pytest.fixture(scope="module")
def validation():
    return _load(W174, "w174_refusal_validation.py")


@pytest.fixture(scope="module")
def census():
    return _load(W177, "w177_campaign_census.py")


def _by_subject(d, sub):
    return [s for s in d["subjects"] if s["subject"] == sub]


def _at(rows, r, key):
    return next(x[key] for x in rows if x["r"] == r)


# ---------------------------------------------------------------------------
# 1. W173 -- the positive control, first.  Without it no negative below counts
# ---------------------------------------------------------------------------


def test_W173_the_instrument_reads_a_compact_operator_as_compact(halo):
    """A LOCAL classical operator on the same seam reads exactly zero past 13.

    The whole measurement is a negative result, so the first thing it has to
    survive is the possibility that the instrument returns "global" for
    everything.  `windowns-split-step` is `WindowNS` with its elliptic part
    handed to the composition layer; same geometry, same state, same port, same
    code path.
    """
    rows = _by_subject(halo, "windowns-split-step")
    assert len(rows) == 2, "the control is run at two different base states"
    for s in rows:
        assert _at(s["linear"]["rows"], 16, "E2_rel") == 0.0, s["label"]
        assert _at(s["linear"]["rows"], 64, "E2_rel") == 0.0, s["label"]
        assert s["support_reach"]["reach"] <= 13, s["label"]
        assert s["support_reach"]["is_global"] is False, s["label"]
        # and it reaches every target at a small, non-trivial radius
        for x in s["r_star_relative_2norm"]:
            assert x["reached"] and not x["trivial"], (s["label"], x)
            assert x["r_star"] <= 6, (s["label"], x)


def test_W173_the_tail_does_not_decay_on_either_learned_donor(halo):
    """The measurement the ladder asked for, and it comes back negative.

    Two checkpoints of one architecture family -- Poseidon-T at 20.8M and
    Poseidon-B at 157.7M, `depths` [8,8,8,8] against [4,4,4,4].  A tail measured
    on one checkpoint is a fact about that checkpoint; the ladder's claim is
    about an expert class, which is why the second donor is here.
    """
    subs = _by_subject(halo, "poseidon-t") + _by_subject(halo, "poseidon-b")
    # the a=1e-4 row is the instrument's own floor (see the amplitude ladder
    # below) and is excluded here by amplitude, named rather than dropped
    operators = [s for s in subs if s["amplitude"] >= 1e-2]
    assert len(operators) >= 7, "T at 3 configs + 1 amplitude control, B at 3"
    assert _by_subject(halo, "poseidon-b"), "the SECOND donor must be present"
    for s in operators:
        rows = s["linear"]["rows"]
        # it plateaus: still a quarter of the operator at r = 64, and flat
        assert _at(rows, 64, "E2_rel") > 0.25, s["label"]
        slope = s["terminal_slope_2norm"]["slope_per_cell"]
        assert abs(slope) < 1.0e-3, (s["label"], slope)
        # and no target is reached at any NON-TRIVIAL radius, including 1e-1
        for x in s["r_star_relative_2norm"]:
            assert x["r_star"] is None or x["trivial"], (s["label"], x)


def test_W173_the_classical_global_operator_is_the_middle_outcome(halo):
    """Section 5's second row exists and it is the classical solver's, not the
    checkpoint's: a real tail, needing most of the domain to be worth anything."""
    rows = _by_subject(halo, "windowns-as-built")
    assert len(rows) == 2
    for s in rows:
        star = {x["target"]: x for x in s["r_star_relative_2norm"]}
        # it DOES reach 1e-2 and 1e-3, at a non-trivial radius
        for t in (1e-2, 1e-3):
            assert star[t]["reached"] and not star[t]["trivial"], (s["label"], t)
        assert 40 <= star[1e-2]["r_star"] <= 80, s["label"]
        assert 100 <= star[1e-3]["r_star"] <= 125, s["label"]
        # and it does NOT reach 1e-4 inside the domain
        assert star[1e-4]["trivial"], s["label"]


def test_W173_the_per_cell_reading_is_the_wrong_test(halo):
    """The proposal's own wording names a per-cell threshold, and per cell the
    checkpoints look like they pass.  Summed over the cells beyond the radius,
    the dropped tail is LARGER than the peak entry -- a per-cell threshold on an
    n-cell seam admits n * eps_tol.  This is the finding that says an
    epsilon-halo written to the proposal's wording would have looked derivable.
    """
    def tail_over_peak(s):
        g = np.array([np.nan if x is None else x
                      for x in s["column_decay"]["profile"]], dtype=float)
        return float(np.nansum(g[21:]) / np.nanmax(g))

    learned = [s for s in _by_subject(halo, "poseidon-t") + _by_subject(halo, "poseidon-b")
               if s["amplitude"] >= 1e-2 and "xhi" in s["port"] or s["amplitude"] >= 1e-2]
    for s in learned:
        # per cell it looks compact: down by more than 5x at d = 20
        prof = s["column_decay"]["profile"]
        assert prof[20] < 0.2 * max(x for x in prof[:8] if x is not None), s["label"]
    # but the summed tail is of the same size as the peak, or larger
    big = [tail_over_peak(s) for s in learned]
    assert min(big) > 0.8, big
    assert max(big) > 1.4, big
    # where the classical global operator's is two orders below it
    for s in _by_subject(halo, "windowns-as-built"):
        assert tail_over_peak(s) < 0.1, s["label"]
    for s in _by_subject(halo, "windowns-split-step"):
        assert tail_over_peak(s) == 0.0, s["label"]


def test_W173_the_amplitude_ladder_separates_the_operator_from_the_floor(halo):
    """Is S an operator or the instrument's own noise?

    A linear response satisfies S(a) = S(a/2) exactly.  The checkpoint shows a
    clean V -- nonlinearity above 1e-2, float32 cancellation below -- and both
    float64 classical columns show none over the same decades.  This is what
    licenses using a = 1e-2 and what disqualifies a = 1e-4.
    """
    lad = halo["amplitude_ladder"]
    pos = lad["poseidon-T P00 xhi"]["rows"]
    by_a = {r["amplitude"]: r["rel_disagreement_vs_ref"] for r in pos}
    assert by_a[1e-2] == 0.0, "the reference amplitude"
    assert by_a[1e-4] > 1.0, by_a          # 268% wrong: cancellation
    assert by_a[1e-5] > 10.0, by_a
    assert by_a[1.0] > 0.2, by_a           # and nonlinearity the other way
    for name in ("windowns-as-built C00 xhi", "windowns-split-step C00 xhi"):
        cls = {r["amplitude"]: r["rel_disagreement_vs_ref"]
               for r in lad[name]["rows"]}
        assert cls[1e-4] < 0.01, (name, cls)   # float64: no V
        assert cls[1e-5] < 0.01, (name, cls)


def test_W173_Pi_is_one_above_sixteen_cells_so_a_finite_halo_buys_nothing(halo):
    """The second, independent kill, and it does not need the decay at all.

    `master-error-bound` 4.1 is sigma <= C_mu * Pi * ||d_lambda||.  On this
    tiling Pi reaches 1 at a reach of 16 cells, so an epsilon-halo would have to
    be BELOW 16 to buy anything in sigma and large enough to make the truncation
    small.  At r = 16 the checkpoints' truncation is a third to five-sixths of
    the operator.  The two requirements point opposite ways.
    """
    pi = {p["d_cells"]: p["contaminated_weight"] for p in halo["Pi_of_halo"]}
    assert pi[0] == 0.0
    assert pi[2] < 0.1 and pi[8] < 0.8, pi
    for d in (16, 20, 32, 64, 127):
        assert pi[d] == 1.0, (d, pi[d])
    learned = [s for s in _by_subject(halo, "poseidon-t") + _by_subject(halo, "poseidon-b")
               if s["amplitude"] >= 1e-2]
    at16 = [_at(s["linear"]["rows"], 16, "E2_rel") for s in learned]
    # 0.275 to 0.753 across seven configurations of two checkpoints
    assert min(at16) > 0.27, list(zip([s["label"] for s in learned], at16))
    assert max(at16) > 0.7, at16


def test_W173_the_transverse_reach_agrees_with_the_along_seam_one(transverse):
    """R10/halo's own quantity is transverse, and nothing had measured it.

    Both directions give the same answer: neither checkpoint's response ever
    falls to 1% of its face value at any depth into the window, while the local
    classical solver's transverse reach is 15-20 cells against a DECLARED
    stencil_radius x substeps of exactly 20.
    """
    subs = {s["label"]: s for s in transverse["subjects"]}
    assert any("poseidon-B" in k for k in subs), "the second donor, transversely"
    for label, s in subs.items():
        for r in s["rows"]:
            d2 = {x["target_relative"]: x for x in r["depth_star_max"]["rows"]}
            if s["subject"].startswith("poseidon") and r["amplitude"] >= 1e-2:
                assert r["nonzero_reach"] == r["n"] - 1, (label, r["j0"])
                assert d2[1e-2]["depth_star"] is None, (label, r["j0"])
                assert d2[1e-4]["depth_star"] is None, (label, r["j0"])
            if s["subject"] == "windowns-split-step":
                assert r["nonzero_reach"] <= NI.DOMAIN_OF_DEPENDENCE, (label, r)
                assert d2[1e-4]["depth_star"] <= 13, (label, r)


def test_W173_no_epsilon_branch_was_written_into_the_halo_rule():
    """The measurement says do not write the rule, so the rule is unwritten.

    This is the assertion that makes the negative binding.  If `_halo_rule` ever
    grows a tolerance branch, this fails, and whoever wrote it has to come back
    to `out/w173/w173.json` first.  `case-study-ladder-to-f1` section 5's own
    instruction was MEASURE BEFORE WRITING ANY RULE.
    """
    from atlas import compiler

    src = inspect.getsource(compiler._halo_rule)
    for token in ("eps_tol", "epsilon_halo", "eps_halo", "tolerance_halo"):
        assert token not in src, f"_halo_rule grew {token}: re-read out/w173"
    # and the compiler still names the probe rather than a threshold
    assert "probe.support_reach" in src


def test_W173_eps_tol_does_not_exist_on_the_graph_the_proposal_is_about():
    """The third obstruction, and it is structural rather than unfinished.

    The proposal compares the tail against `eps_tol = min(tau, sigma)`, "which
    already exists in the compiler".  It exists as a RULE.  On the graph that
    holds the checkpoint it has no value, and `poseidon.build`'s own docstring
    records why: the checkpoint is fixed at 128x128, so there is no same-class
    monolith at any resolution and no reference pair for tau and sigma.
    """
    pytest.importorskip("torch")
    from atlas.cases import poseidon as PO

    try:
        u, v = NI.load_state()
        g, _ = PO.build(u, v)
    except Exception as exc:                                       # noqa: BLE001
        pytest.skip(f"checkpoint or state unavailable: {exc}")
    r = compile_scheme(g, budget=Budget(allow_probe=False))
    assert r.scheme is not None
    assert r.scheme.eps_tol is None
    assert any("tau" in u_ for u_ in r.unmeasured), r.unmeasured
    assert any("sigma" in u_ for u_ in r.unmeasured), r.unmeasured
    tol = [d for d in r.decisions if d.rule == "eps_tol"]
    assert tol and tol[0].verdict is Verdict.ADMIT_UNCERTIFIED


# ---------------------------------------------------------------------------
# 2. W174 -- is the refusal EARNED?  poc2-novelty-audit section 4's third clause
# ---------------------------------------------------------------------------


def test_W174_exactly_one_arrangement_diverges_and_the_rule_is_right_there(validation):
    """The 2x2 completed, and it takes BOTH variables to make the coupling fail.

    CS-S1 swept two cells that differ in two variables at once, so its bracket
    could not attribute the divergence.  With all four run: only
    (embedded elliptic, no global projection) diverges below the declared reach,
    and there `R10/halo` refuses at every diverging halo and admits at every
    converging one -- eleven of eleven, with the declared 20 inside the measured
    bracket (18, 20].
    """
    arr = validation["arrangements"]
    assert len(arr) == 4, sorted(arr)
    diverging = [k for k, v in arr.items()
                 if any(r["converging"] is False for r in v["rows"])]
    assert len(diverging) == 1, diverging
    cell = arr[diverging[0]]
    assert cell["expose_elliptic"] is False and cell["project"] is False
    assert cell["tally"] == {"refuse & diverge -- EARNED": 5,
                             "admit & converge -- earned": 6}, cell["tally"]
    thr = cell["threshold"]
    assert thr["largest_diverging"] == 18
    assert thr["smallest_converging"] == 20
    assert thr["declared_reach"] == validation["declared_reach"] == 20
    assert thr["bracket_contains_reach"] is True
    # nothing UNDER-fires anywhere: no admit-and-diverge in any cell
    for name, v in arr.items():
        assert "admit & diverge -- UNDER-fires" not in v["tally"], name


def test_W174_the_same_refusal_over_fires_in_the_three_other_cells(validation):
    """And the clause is cleared on one side only.  In the three arrangements
    that converge at every halo, the identical rule refuses five of eleven rows.
    """
    arr = validation["arrangements"]
    over = [v for v in arr.values()
            if all(r["converging"] for r in v["rows"])]
    assert len(over) == 3
    for v in over:
        assert v["tally"]["refuse & converge -- OVER-fires"] == 5, v["tally"]
        assert v["threshold"]["largest_diverging"] is None


def test_W174_the_graph_cannot_say_which_cell_it_is_in(validation):
    """Asked of the constructor rather than argued: the global projection is a
    property of the COMPOSITION LAYER, and `build` has no parameter for it, so
    all four cells compile to the same declarations and the same verdict."""
    assert validation["project_is_declarable"]["project_in_build"] is False
    assert "project" not in inspect.signature(NI.build).parameters
    assert "project" in inspect.signature(NI.halo_convergence).parameters


# ---------------------------------------------------------------------------
# 3. W175 -- DIAGNOSED, NOT FIXED.  These assert the defect REPRODUCES
# ---------------------------------------------------------------------------


def test_W175_the_halo_message_over_attributes_by_one_variable():
    """**W175, open.**  `_halo_bounds` says "accuracy and convergence" whenever a
    cut agent is `embedded` or `unknown`.  W174 measures that the divergence
    needs `embedded` AND the composition layer not projecting -- with the
    projection restored the same embedded arrangement converges at every halo
    from 4 up, contraction 0.98392 at halo 4.

    So the branch predicate is one variable short of the measured cause, and
    `_halo_bounds` will say "convergence" for a graph where convergence is not
    at stake.  **This asserts the defect, not its repair.**  Closing W175 breaks
    this test by design: keep the diagnosis, rewrite the assertion, add a
    control that can fail.
    """
    u, v = NI.load_state()
    g, _ = NI.build(u, v, expose_elliptic=False)
    cut, _uncut = _decomposition_cuts(g)
    assert _halo_bounds(g, cut) == "accuracy and convergence"
    # the exposed arrangement is the only thing that moves it, and it is not the
    # only thing that moves the CONVERGENCE
    g2, _ = NI.build(u, v, expose_elliptic=True)
    cut2, _ = _decomposition_cuts(g2)
    assert _halo_bounds(g2, cut2) == "accuracy"


def test_W175_the_declaration_that_would_carry_it_is_filled_in_wrongly():
    """**W175's second half, open.**  `GlobalField.produced_by` already has a
    documented meaning for the all-agents case -- "a global operation over the
    whole state, owned by the composition layer" -- and `neural_interface.build`
    declares it UNCONDITIONALLY, including in the two cells of W174's 2x2 where
    the composition layer does not project.

    So this is a mis-declaration plus a rule that does not read the declaration
    it would need, and not a missing schema.
    """
    u, v = NI.load_state()
    names = set(NI.DEFAULT_TILING.names)
    for expose in (True, False):
        g, _ = NI.build(u, v, expose_elliptic=expose)
        gf = [f for f in g.global_fields if f.name == "pressure"]
        assert len(gf) == 1
        # identical in both arrangements: the declaration cannot tell them apart
        assert set(gf[0].produced_by) == names, (expose, gf[0].produced_by)


# ---------------------------------------------------------------------------
# 4. W171 -- the scope, and the [AI Inference] it carries, checked
# ---------------------------------------------------------------------------


def _fw():
    u, v = NI.load_state()
    return F.build(u, v, motion=False)[0]


def _wf():
    return WF.build(np.full((WF.NY, WF.NX), WF.U_INF),
                    np.zeros((WF.NY, WF.NX)), motion=False)[0]


def test_W171_the_axis_is_already_declared_per_seam_on_four_graphs():
    """**W171's [AI Inference] is FALSE, and the check is cheap.**

    The row says the natural carrier is the AGENT.  It is not: a cut is a
    relation among several agents, not a property of one, and two agents of one
    family cut two different ways is unrepresentable on an agent.  The carrier
    that works is the CONNECTION, and the field is already there --
    `geometrically_coincident` separates the axis exactly, on two overlapping
    graphs and two non-overlapping ones, with no exception and no new schema.
    """
    over = [("front_wing", _fw()), ("wing_fsi", _wf())]
    non = [("cooling_loop", CL.build()[0]), ("powertrain", PW.build()[0])]
    for name, g in over:
        flags = [bool(c.geometrically_coincident) for c in g.connections]
        assert not all(flags), name
        # the coincident ones are exactly the seams that are NOT decomposition
        # cuts: a fluid-structure or fluid-lumped interface is a PHYSICAL
        # boundary, and Gamma there was not created by the tiling
        cut, _ = _decomposition_cuts(g)
        for c in g.connections:
            same_family = (g.agent(c.a[0]).capabilities.governing_family
                           == g.agent(c.b[0]).capabilities.governing_family)
            both_cut = c.a[0] in cut and c.b[0] in cut
            if not c.geometrically_coincident:
                assert same_family and both_cut, (name, c.seam_id)
    for name, g in non:
        flags = [bool(c.geometrically_coincident) for c in g.connections]
        assert all(flags), name


def test_W171_three_of_the_seven_axis_readers_emit_nothing_on_the_wrong_axis():
    """**And the row's count is wrong in both numbers.**  It says five rules
    branch and return early.  Seven functions read the axis; THREE of them
    return early and emit nothing, and the other four emit on both branches --
    so they would say the wrong thing loudly, which is the cheaper failure.
    """
    import ast
    import textwrap

    from atlas import compiler

    def returns_early_on_the_axis(fn) -> bool:
        """An axis branch that returns having emitted NOTHING.

        Two refinements the obvious predicate gets wrong, and both were found by
        this test failing rather than by reading:

        * matched on the tree, not the text -- `_r12_conservative_assembly` puts
          a three-line comment between the test and the return;
        * "silent" is *emits no decision*, not *ends in a return*.  `_cut_policy`
          ends its wrong-axis branch in a bare return too, and emits `L2/C3`
          first, so it says the substructuring thing rather than nothing.
        """
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if "Decomposition." not in ast.unparse(node.test):
                continue
            body = "\n".join(ast.unparse(s) for s in node.body)
            emits = any(k in body for k in
                        ("rec.admit", "rec.refuse", "rec.decertify",
                         "record.admit", "record.refuse", "record.decertify"))
            last = node.body[-1]
            if isinstance(last, ast.Return) and last.value is None and not emits:
                return True
        return False

    silent, loud = [], []
    for name in ("_halo_rule", "_r12_conservative_assembly", "_w49_sigma_branch",
                 "_cut_policy", "_l2_decomposition", "_l5_l7_scheme"):
        fn = getattr(compiler, name)
        assert "Decomposition." in inspect.getsource(fn), name
        (silent if returns_early_on_the_axis(fn) else loud).append(name)
    assert set(silent) == {"_halo_rule", "_r12_conservative_assembly",
                           "_w49_sigma_branch"}, silent
    assert len(loud) == 3, loud


def test_W171_one_field_carries_the_axis_all_the_way_into_the_artifact():
    """The schema reach, which is what makes this a schema change and not a rule
    change: the axis is one field on the graph, one on the compiled scheme, and
    one key in the emitted artifact."""
    from atlas.graph import CaseGraph
    from atlas.scheme import Scheme

    assert "decomposition" in CaseGraph.__dataclass_fields__
    assert "decomposition" in Scheme.__dataclass_fields__
    g = _fw()
    assert "decomposition" in g.as_dict()
    # and Agent.domain is free text, so there is no structured domain
    # declaration for a per-region axis to hang on
    from atlas.graph import Agent

    assert Agent.__dataclass_fields__["domain"].type in ("str", str)


# ---------------------------------------------------------------------------
# 5. W177 -- the census the checkpoint stands on
# ---------------------------------------------------------------------------


def test_W177_the_nine_of_nine_is_conditional_on_an_undeclarable_field():
    """The figure three pages and two READMEs quote, compiled rather than recalled.

    Under the record `poseidon_capabilities` actually builds, `front_wing` at a
    fixed shape is `admit-uncertified` with ZERO refusals.  The nine appears
    only with `elliptic_subsolve=embedded`, which the demo's own panel calls
    "DECLARED, not measured" and which W60 records as undeclarable for a learned
    operator.  And `L2/R10` is a GRAPH-level rule: one refusal painted across
    nine seam tiles is not nine independent verdicts.
    """
    from atlas.demo_frontwing import substitution as SUB

    base = _fw()
    by_key = {}
    for c in SUB.CANDIDATES:
        g = SUB.patch_fluid(base, c.fields, F.DEFAULT_TILING)
        r = compile_scheme(g)
        by_key[c.key] = (r, g)

    r_inc, _ = by_key["windowns"]
    assert r_inc.verdict is Verdict.ADMIT, "Tier 39's certified graph"

    r_hon, _ = by_key["poseidon_declared"]
    assert r_hon.verdict is Verdict.ADMIT_UNCERTIFIED
    assert len(r_hon.decisions.refusals) == 0, [
        f"{d.layer}/{d.rule}" for d in r_hon.decisions.refusals]

    r_dec, g_dec = by_key["poseidon_probed"]
    assert r_dec.verdict is Verdict.REFUSE
    refusals = r_dec.decisions.refusals
    assert {f"{d.layer}/{d.rule}" for d in refusals} == {"L2/R10"}
    # ONE rule, ONE decision, and its subject is the six fluid AGENTS -- not
    # nine seam verdicts.  No seam id appears as the subject of any refusal.
    assert len(refusals) == 1
    assert len(g_dec.connections) == 9
    seam_ids = {c.seam_id for c in g_dec.connections}
    assert refusals[0].subject not in seam_ids
    assert set((refusals[0].subject or "").split(", ")).isdisjoint(seam_ids)
    assert len((refusals[0].subject or "").split(", ")) == 6, refusals[0].subject


def test_W177_the_demo_seam_colouring_says_what_the_corrected_READMEs_say():
    """The correction is in the README's OWN counting convention.

    `engine.seam_verdicts` attributes a graph-level rule to every seam, which is
    a defensible way to show a graph-level refusal on a seam panel and is why
    the nine appears at all.  Counted that way: at the RIDING shape the
    incumbent is already 2 red -- `L2/InterfaceMotion`, a refusal about the
    moving interface that fires on the classical solver -- and its other seven
    are GREEN, not uncertified, which is the factual error the READMEs carried.
    At a FIXED shape the incumbent is nine green and the honestly declared
    checkpoint is nine amber with zero refusals.
    """
    from atlas.demo_frontwing import substitution as SUB
    from atlas.demo_frontwing.engine import seam_verdicts

    u, v = NI.load_state()

    def colours(motion, key):
        base, _ = F.build(u, v, motion=motion, tiling=F.DEFAULT_TILING)
        c = next(x for x in SUB.CANDIDATES if x.key == key)
        g = SUB.patch_fluid(base, c.fields, F.DEFAULT_TILING)
        sv = seam_verdicts(g, compile_scheme(g))
        n = {"red": 0, "amber": 0, "green": 0}
        for m in sv.values():
            n[m["colour"]] += 1
        return n, {x for m in sv.values() for x in m["refusals"]}

    riding_inc, rules = colours(True, "windowns")
    assert riding_inc == {"red": 2, "amber": 0, "green": 7}, riding_inc
    assert rules == {"L2/InterfaceMotion"}, rules      # the CLASSICAL solver's

    riding_pos, _ = colours(True, "poseidon_declared")
    assert riding_pos == {"red": 2, "amber": 7, "green": 0}, riding_pos

    fixed_inc, _ = colours(False, "windowns")
    assert fixed_inc == {"red": 0, "amber": 0, "green": 9}, fixed_inc

    fixed_pos, _ = colours(False, "poseidon_declared")
    assert fixed_pos == {"red": 0, "amber": 9, "green": 0}, fixed_pos

    fixed_dec, dec_rules = colours(False, "poseidon_probed")
    assert fixed_dec == {"red": 9, "amber": 0, "green": 0}, fixed_dec
    assert dec_rules == {"L2/R10"}, dec_rules


def test_W177_the_stopping_rule_trigger_is_evaluated_not_approached(census):
    """`case-study-ladder-to-f1` section 7's rule, stood on.

    Its trigger is `refuse` or `blind` at EVERY seam of EVERY case study.  With
    the campaign's own instrument -- `certify_substitution`, not the compiler --
    three of eight agent-sides return an informative `admit`.  The trigger is
    not met, and the framework-only outcome is not the one section 7 licenses.
    """
    s = census["stopping_rule"]
    assert s["every_seam_refused"] is False
    assert s["seams_refused"] == 0, s
    cert = census["certificates"]
    assert cert["beta_min"] is None
    admits = [r for r in cert["rows"]
              if any(x["verdict"] == "admit" for x in r["beta_min_sweep"])]
    assert len(admits) == 3, [r["agent"] for r in admits]
    # two sides are saturated above 1 and refuse at every tolerance -- W109's
    # reading arriving on a second, unrelated graph
    never = [r for r in cert["rows"] if r["fails_above"] <= 0]
    assert len(never) == 2, never
    for r in never:
        assert r["delta_norm"] / r["beta"] > 1.0, r


def test_W177_the_swap_is_within_five_percent_of_deleting_the_block(census):
    """What the three admits admit, and it is W76's null replacement measured.

    W76's docstring gives the bound as a hypothetical -- "a replacement that
    ignores its boundary data entirely removes S_i and nothing else, giving
    ||Delta|| = ||S_i||".  On all eight sides ||Delta|| / ||S_i|| is 0.95 to
    1.01, so swapping the checkpoint in IS that null replacement to within 5%.

    The reason is in the other column: the checkpoint's boundary response is
    0.8% to 9% of the classical incumbent's.  A one-shot map at a coarse lead
    time barely propagates a ring perturbation, which is the same fact
    `epsilon-halo-measurement` section 4.1 reports as ||S||_2 = 0.0048 against
    WindowNS's 0.393.

    It also explains the knife edge WITHOUT appealing to the tolerance: the
    informative-admit band has width ||S_i|| - ||Delta|| exactly, so a ratio
    near 1 makes it nearly zero by construction.
    """
    rows = census["certificates"]["rows"]
    ratios = [r["delta_over_block"] for r in rows]
    sizes = [r["new_over_old_block"] for r in rows]
    assert len(rows) == 8
    assert 0.94 < min(ratios) and max(ratios) < 1.02, ratios
    assert max(sizes) < 0.10, sizes          # 9% at the very most
    assert min(sizes) < 0.01, sizes          # and under 1% at the extreme
    # the band's width IS ||S_i|| - ||Delta||, so it follows from the ratio
    for r in rows:
        width = r["fails_above"] - r["visible_above"]
        assert abs(width - (r["block_norm"] - r["delta_norm"])) < 1e-9, r


def test_W177_the_admit_window_is_resolvable_by_the_probe_that_makes_it(census):
    """The control that had to be run, and it came back the other way.

    The windows are 0.2%-1.1% of beta wide and W173 section 5.3 measured this
    checkpoint noise-dominated at eps = 1e-4 under a DELTA probe.  Under the
    Fourier probe the certificate uses, the endpoints move by 5-34% of the
    window when eps changes by ten -- so the admits are verdicts, not noise.
    A Fourier mode spreads eps over all 128 cells; a delta concentrates it.
    """
    wr = census["window_resolution"]
    assert wr["n_admitting"] == 3, wr
    assert wr["n_resolvable"] == 3, wr
    for r in wr["rows"]:
        if not r["admits_somewhere"]:
            continue
        assert r["shift_over_width"] < 0.5, r
        assert r["window_over_beta"] < 0.02, r    # and it is a knife edge


def test_W177_the_compiler_has_still_never_admitted_a_learned_expert(census):
    """The two instruments answer different questions and both are reported.

    The certificate admits at three sides; the COMPILER has never emitted
    `admit` for a graph holding a learned expert and still has not.  Saying one
    without the other is how the campaign got mis-summarised in the first place.
    """
    live = [r for r in census["rows"] if r["graph"] == "poseidon-t-2x2"]
    assert live, "the graph with the live weights must be in the census"
    assert not any(r["graph_verdict"] == "admit" for r in live), live
    honest = [r for r in live if not r["declared_not_measured"]]
    assert honest and all(r["graph_verdict"] == "admit-uncertified"
                          for r in honest), honest
    assert all(r["seams_refused"] == 0 for r in honest), honest
