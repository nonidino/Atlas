"""Tier 42 -- rung 8's admissibility question, pinned.

W165 asks whether a seam whose interface condition is an INEQUALITY can be
compiled at all.  `inequality-seam-admissibility` answers it, and this file
pins the three kinds of claim that page makes.

**Most of what is asserted here is a DEFECT, on purpose.**  This repo's practice
with a row it opens and does not close is a test that asserts the defect
reproduces, so that closing it is a visible act rather than a silent one
(`test_tier40_epsilon_halo`'s W175 block says the same).  Every assertion marked
`_the_defect_` below will fail when W182 is closed, and that is the design: keep
the diagnosis in the docstring, rewrite the assertion, add a control.

**The controls are the other half and they must NOT move.**  A smooth fixture
compiled and probed through the identical instrument is what makes the kinked
numbers mean anything, and one of them -- `L2/InterfaceMotion` firing on the
SMOOTH control as hard as on the kinked graph -- is the whole evidence for the
claim that the refusal a contact graph gets today is about the wrong property.

**And the guard.**  `test_W182_no_inequality_branch_was_written` fails if a
complementarity branch appears in the interface solve without the page being
re-read first.  That is `test_W173_no_epsilon_branch_was_written_into_the_halo_rule`'s
shape, one rule along.

Numbers are asserted against `out/w181/w181.json` rather than retyped, on the
Tier 39 discipline: a figure in a page and a figure in a rule must not be able
to drift apart.  Run `python scripts/w181_inequality_seam_probe.py` first.
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

from atlas import Budget, compile_scheme                              # noqa: E402
from atlas import capability, graph as graph_mod, solve as solve_mod  # noqa: E402
from atlas import scheme as scheme_mod                                # noqa: E402
from atlas.capability import MotionClass                              # noqa: E402
from atlas.probe import ProbeBudget                                   # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import w181_inequality_seam_probe as W181                             # noqa: E402

ARTIFACT = os.path.join(_ROOT, "out", "w181", "w181.json")


@pytest.fixture(scope="module")
def measured():
    if not os.path.exists(ARTIFACT):
        pytest.skip("run scripts/w181_inequality_seam_probe.py")
    with open(ARTIFACT, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def compiles():
    """Three graphs differing only in the shape of one boundary response."""
    out = {}
    for shape in ("smooth", "kink-at-base", "kink-at-gap"):
        g = W181.build(shape)
        out[shape] = (g, compile_scheme(g, Budget(), ProbeBudget()))
    return out


def _rules(result):
    return [(d.layer, d.rule, d.verdict.value, d.subject)
            for d in result.decisions.decisions]


# --------------------------------------------------------------------------
# the defect: nine layers cannot see a kink
# --------------------------------------------------------------------------


def test_W182_the_defect_a_kinked_seam_compiles_identically_to_a_linear_one(compiles):
    """The admissibility question, answered by experiment.

    If an inequality-shaped seam reaches the same verdict under the same rules
    as an equality one, then nothing in L1..L9 is asking the question -- which
    is what makes this a NAMED HOLE rather than a rule that needs widening.
    """
    base = _rules(compiles["smooth"][1])
    for shape in ("kink-at-base", "kink-at-gap"):
        result = compiles[shape][1]
        assert _rules(result) == base, f"{shape} differs from the linear control"
        assert result.verdict is compiles["smooth"][1].verdict
        assert result.runnable is True
    # And the summary statistic a reader would reach for does not separate them.
    betas = [compiles[s][1].seam_operators["s"].beta
             for s in ("smooth", "kink-at-base", "kink-at-gap")]
    assert max(betas) / min(betas) < 1.01, "beta separates them after all -- re-read"


def test_W182_the_defect_no_declaration_can_say_the_condition_is_an_inequality():
    """There is no field to declare it in, on the connection or on the port.

    C9 is the precedent for the repair: `ResponseHalf` was added because no
    value-based test could discriminate two conventions, and it refuses only
    the case it can be sure about.  This asserts that no such field exists yet.
    """
    conn_fields = set(graph_mod.Connection.__dataclass_fields__)
    port_fields = set(capability.PortDecl.__dataclass_fields__)
    caps_fields = set(capability.ExpertCapabilities.__dataclass_fields__)
    for name in ("condition", "seam_condition", "complementarity", "inequality"):
        assert name not in conn_fields
        assert name not in port_fields
        assert name not in caps_fields
    # And nothing anywhere carries the vocabulary.
    assert not hasattr(graph_mod, "SeamCondition")


def test_W181_the_defect_R7_selects_a_branch_whose_premise_it_does_not_check(compiles):
    """R7's own sentence names SMOOTH and `probe_route` has no third clause."""
    text = scheme_mod.RULES["R7"]
    assert "smooth" in text, "R7's sentence changed; re-read the page before editing"
    src = inspect.getsource(capability.ExpertCapabilities.probe_route)
    assert "has_jvp" in src and "deterministic" in src
    assert "smooth" not in src.lower(), "a smoothness clause appeared -- re-read W181"
    # The triple selects finite differences on every agent of every fixture.
    for shape, (g, _r) in compiles.items():
        for a in g.agents:
            assert a.capabilities.differentiable is capability.Differentiable.NONE
            assert a.capabilities.deterministic is True
            assert a.capabilities.probe_route() == "finite-difference"


def test_W181_the_defect_no_decision_anywhere_cites_R7(compiles):
    """A rule nothing cites cannot be appealed to and cannot be audited."""
    for shape, (_g, result) in compiles.items():
        cited = [d.rule for d in result.decisions.decisions if "R7" in str(d.rule)]
        assert cited == [], f"{shape} cites R7 now: {cited}"


def test_W182_no_inequality_branch_was_written(compiles):
    """The guard. Nothing was written into the solve, and this says so.

    `solve_interface` drives ``S lam - chi`` to a root by four routes and none
    of them projects onto a cone.  If a complementarity branch appears here,
    this test fails and `inequality-seam-admissibility` gets re-read before the
    branch is accepted -- which is the point of the guard, not an objection to
    the branch.
    """
    src = inspect.getsource(solve_mod)
    for token in ("complementar", "lcp", "projected_gradient", "active_set",
                  "np.maximum(lam", "clip(lam"):
        assert token not in src.lower(), f"{token!r} appeared in solve.py"
    # And the residual is affine, which is the structural form of the same claim.
    resid = inspect.getsource(solve_mod.InterfaceProblem.residual)
    assert "self.S @" in resid and "- self.chi" in resid


# --------------------------------------------------------------------------
# the controls. These must not move.
# --------------------------------------------------------------------------


def test_control_the_smooth_fixture_is_stable_over_nine_decades(measured):
    """Every epsilon sweep in this vault has come back stable. This is the one
    that says the instrument used on the kink is the same instrument."""
    smooth = measured["sweep"]["smooth"]
    assert smooth["digits_stable"] >= 7
    assert smooth["max_drift"] < 1e-8
    assert len(smooth["rows"]) == 9


def test_control_the_reported_residual_is_the_true_one_on_a_linear_seam(measured):
    """On an affine seam ``S lam - chi`` IS the port-matching condition."""
    row = measured["residual"]["smooth"]
    assert row["true_relative"] < 1e-12
    assert row["ratio_true_over_reported"] < 10.0


def test_control_the_motion_refusal_fires_on_the_LINEAR_fixture_too(measured):
    """The refusal a truthfully declared contact graph gets is about the
    PATCH MOVING, and this is the control that says so: declare the same motion
    on a seam with no inequality anywhere and the identical refusal fires.

    That is why `L2/InterfaceMotion` does not answer W165: a rule that cannot
    tell the two apart is not adjudicating the difference between them.
    """
    for shape in ("smooth", "kink-at-gap"):
        row = measured["declared_truthfully"][shape]
        assert row["verdict"] == "refuse"
        assert row["runnable"] is False
        rules = {r[0] + "/" + str(r[1]) for r in row["refusing_rules"]}
        assert rules == {"L2/InterfaceMotion"}, rules


def test_control_declaring_motion_really_does_refuse_here_and_now():
    """The measurement above, re-run live rather than read off the artifact."""
    g = W181.build("smooth")
    for a in g.agents:
        for port in a.capabilities.ports:
            port.motion_class = MotionClass.SOLUTION_DEPENDENT
    result = compile_scheme(g, Budget(), ProbeBudget())
    assert result.runnable is False
    rules = {(d.layer, d.rule) for d in result.decisions.decisions
             if d.verdict.value == "refuse"}
    assert rules == {("L2", "InterfaceMotion")}


# --------------------------------------------------------------------------
# the measured numbers the page quotes
# --------------------------------------------------------------------------


def test_W181_the_epsilon_sweep_settles_at_a_homogeneous_kink(measured):
    """W165 predicted the sweep would not settle. At a kink AT the base it does.

    ``relu(eps x) == eps relu(x)``, so the one-sided quotient is independent of
    the step and nine decades of sweep return the same matrix -- the same
    stability the smooth control returns, on an operator that is 26% away from
    the other branch.  The row's own proposed discriminator passes here, which
    is why the page corrects it rather than quoting it.
    """
    smooth = measured["sweep"]["smooth"]
    base = measured["sweep"]["kink-at-base"]
    gap = measured["sweep"]["kink-at-gap"]
    assert base["digits_stable"] >= 7
    assert base["max_drift"] < 10.0 * smooth["max_drift"]
    # and it settles on ONE of the two one-sided Jacobians, to the last digit
    rows = [r for r in measured["truth"]["rows"] if r["shape"] == "kink-at-base"]
    assert all(r["to_J_plus"] < 1e-9 for r in rows)
    assert all(r["to_J_minus"] > 0.2 for r in rows)
    # the offset kink is the case the sweep DOES see, and only that one
    assert gap["digits_stable"] == 0
    assert gap["max_drift"] > 0.1


def test_W181_below_the_gap_the_kinked_seam_is_bit_identical_to_the_linear_one(measured):
    """The contact is simply not in the operator at a small enough step.

    And the compiler's own default step is 1e-2, on the other side of a gap of
    1e-3 -- so the default probe reports a partly engaged contact and nothing in
    the run says which side of the switch it was taken on.
    """
    gap_g = measured["truth"]["gap"]
    smooth_rows = {r["eps"]: r for r in measured["sweep"]["smooth"]["rows"]}
    kink_rows = {r["eps"]: r for r in measured["sweep"]["kink-at-gap"]["rows"]}
    for eps, row in kink_rows.items():
        if eps <= gap_g:
            assert row["norm_S"] == pytest.approx(smooth_rows[eps]["norm_S"], rel=1e-15)
            assert row["beta"] == pytest.approx(smooth_rows[eps]["beta"], rel=1e-15)
        else:
            assert row["norm_S"] > smooth_rows[eps]["norm_S"]
    default_step = max(1e-2, 100.0 * float(np.finfo(float).eps))
    assert default_step > gap_g, "the default probe no longer straddles the gap"


def test_W182_the_framework_reports_machine_precision_on_a_violated_condition(measured):
    """The silent-wrongness measurement, with its control.

    ``||S lam - chi||`` at the solved trace against ``||sum_i P_i^* f_i(P_i lam)||``
    -- the condition the first is standing in for.  On the linear control they
    are the same number.  On both kinked seams the first is at 5e-16 and the
    second is at 4% of the trace scale, and the compile that produced it is the
    linear control's compile rule for rule.
    """
    control = measured["residual"]["smooth"]
    assert control["ratio_true_over_reported"] < 10.0
    for shape in ("kink-at-base", "kink-at-gap"):
        row = measured["residual"][shape]
        assert row["reported_relative"] < 1e-12
        assert row["true_relative"] > 1e-2
        assert row["ratio_true_over_reported"] > 1e10


def test_W181_the_reproducibility_floor_sets_the_probe_step_for_nobody(measured):
    """R7 says the step is set by the reproducibility floor. Counted, it is not.

    ``step = budget.fd_step or max(1e-2, 100 * floor)``, so a floor below 1e-4
    loses to the constant.  The floor is a sensible guard against differencing
    into noise; what it is not is the mechanism R7's sentence describes.
    """
    built = measured["step_census"]
    assert built["n_step_set_by_floor"] == 0
    assert built["n_agents"] >= 20
    src = measured["step_census"]["source"]
    assert src["n_step_set_by_floor"] == 0
    assert src["n_resolved"] >= 15


def test_W182_the_two_one_sided_blocks_differ_by_more_than_any_probe_floor(measured):
    """The quantity the missing rule would constrain, and it is not small."""
    assert measured["truth"]["jump_relative"] > 0.2
    # and a probe returns exactly one of them, with nothing recording a choice
    op_keys = set(measured["compiles"]["kink-at-base"].keys())
    assert "beta" in op_keys and "passivity_defect" in op_keys
    assert measured["compiles"]["kink-at-base"]["passivity_defect"] == 0.0
