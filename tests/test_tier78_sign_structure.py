"""Tier 78 -- W306: the sign structure `L4/E7/passivity` never read.

`_fill_diagnostics` set ``passivity_defect = abs(lam_min)`` and never touched
``lam_max``, so two different failures reported the same number:

    ONE-SIGNED NEGATIVE   every mode damped, and the whole operator a GLOBAL
                          SIGN away from passive. ``S lam = chi`` is indifferent
                          to that sign because chi flips with S, so it is a
                          DECLARATION inconsistency -- an undeclared or inverted
                          `effort_normal` -- and the remedy is one character.
    MIXED                 ``lam_min < 0 < lam_max``: a genuinely amplified
                          interface mode, which is what E7 is about.

**The tier's own gate was whether this is a rocket detail or a rule.** Swept over
60 seams in 8 cases: 4 report a defect and **3 are one-signed** -- the rocket's
``b-c:THERM`` and `car_graph`'s ``J1_core_strip`` and ``J3_rotor_strip``, the
latter two on a graph that DOES declare `effort_normal` and has it inverted.
Only `thermal_strain`'s ``thermal-pressure`` is a real amplified mode.

The verdict is deliberately UNCHANGED: a negative-definite operator still does
not give the ``L <= 1`` branch and still needs fixing. What changes is that the
certificate now says which of the two it is, and what to do about it.
"""
from __future__ import annotations

import numpy as np
import pytest

from atlas.compiler import compile_scheme
from atlas.probe import ProbedBlock, _fill_diagnostics


def _diag(S):
    blk = ProbedBlock(agent_id="t", seam_id="s", S=np.asarray(S, float),
                      route="assembled", n_solves=0, dim_M=np.shape(S)[0])
    _fill_diagnostics(blk, np.asarray(S, float))
    return blk


# ===========================================================================
# 1. the algebra -- the claim is about the statistic, not about a rocket
# ===========================================================================


def test_the_two_failures_that_used_to_report_the_same_number():
    """Same `lam_min`, different `lam_max`. Before W306 these were indistinguishable
    in everything the probe emitted."""
    one_signed = _diag(np.diag([-3.0, -2.0, -1.0]))
    mixed = _diag(np.diag([-3.0, 2.0, 3.0]))
    # the old quantity still cannot tell them apart, and that is the point
    assert one_signed.passivity_defect == pytest.approx(mixed.passivity_defect)
    assert one_signed.passivity_lambda_min == pytest.approx(mixed.passivity_lambda_min)
    # the new one can
    assert one_signed.sign_structure == "negative"
    assert mixed.sign_structure == "mixed"
    assert one_signed.passivity_lambda_max < 0.0 < mixed.passivity_lambda_max


@pytest.mark.parametrize("S,expected", [
    (np.diag([1.0, 2.0, 3.0]), "positive"),
    (np.diag([-1.0, -2.0, -3.0]), "negative"),
    (np.diag([-1.0, 2.0, 3.0]), "mixed"),
    (np.zeros((3, 3)), "zero"),
])
def test_the_four_classes(S, expected):
    assert _diag(S).sign_structure == expected


def test_the_classification_uses_the_same_tolerance_as_the_defect():
    """A negative eigenvalue at arithmetic-noise level is not a defect, and it
    must not be a sign structure either -- otherwise every symmetric operator the
    probe assembles would read `mixed` and the field would be noise."""
    eps = np.finfo(float).eps
    blk = _diag(np.diag([1.0, 1.0, -eps]))
    assert blk.passivity_defect == 0.0
    assert blk.sign_structure == "positive"


def test_a_global_sign_leaves_the_singular_values_alone():
    """Why a one-signed negative operator is a declaration issue and not a
    physical one: negating S changes no singular value, so beta and kappa -- the
    quantities the bound actually uses -- are untouched. Only E7's test moves."""
    S = np.array([[3.0, 0.4], [0.4, 2.0]])
    a, b = _diag(S), _diag(-S)
    assert a.beta == pytest.approx(b.beta)
    assert a.kappa == pytest.approx(b.kappa)
    assert a.sign_structure == "positive" and b.sign_structure == "negative"


# ===========================================================================
# 2. the rule says which one it is, and what to do
# ===========================================================================


def test_the_certificate_names_the_structure_and_the_remedy():
    """`thermal_strain` is the vault's only MIXED spectrum and the rocket's b-c
    the one-signed case, so the two messages are checked against the two cases
    that actually produce them rather than against a fixture."""
    from atlas.cases import thermal_strain

    built = thermal_strain.build()
    g = built[0] if isinstance(built, tuple) else built
    r = compile_scheme(g)
    pas = [d for d in r.decisions.decertifications if d.rule == "E7/passivity"]
    assert pas, "thermal_strain no longer reports a passivity defect"
    assert "W306: the spectrum is MIXED" in pas[0].message
    assert "no choice of effort_normal removes it" in pas[0].message


def test_the_verdict_is_deliberately_unchanged():
    """W306 changes what the certificate SAYS, not what it decides.

    A negative-definite operator still does not give the L <= 1 branch, so it is
    still a decertification. Changing that would move every artifact with one,
    and the evidence for it is not in hand.
    """
    from atlas.cases import thermal_strain

    built = thermal_strain.build()
    g = built[0] if isinstance(built, tuple) else built
    r = compile_scheme(g)
    assert [d for d in r.decisions.decertifications if d.rule == "E7/passivity"]


# ===========================================================================
# 3. W308 -- car_graph declares effort_normal and has it inverted on two seams
# ===========================================================================


def test_W308_car_graph_has_two_inverted_declarations_and_one_correct_one():
    """**Found by W306, and deliberately NOT fixed here.**

    Two of `car_graph`'s three declared seams name the agent that makes the
    operator negative definite; naming the other side makes it positive definite
    with the singular values untouched. The third, ``J2_heat``, already names the
    right one -- which is the control that makes this a finding rather than a
    blanket flip.

    It is not fixed in this tier because
    [[poc3-racelab-car-graph]] records those two seams as earning
    `L4/E7/passivity` in "the package's standing decertification set", so the fix
    changes a PoC 3 case study's recorded decision counts and belongs with an
    audit of that page rather than as a side effect of a diagnostic tier.
    """
    from atlas.cases import car_graph

    built = car_graph.build()
    g = built[0] if isinstance(built, tuple) else built
    r = compile_scheme(g)
    got = {s: r.seam_operators[s].sign_structure
           for s in ("J1_core_strip", "J3_rotor_strip", "J2_heat")}
    assert got == {"J1_core_strip": "negative",
                   "J3_rotor_strip": "negative",
                   "J2_heat": "positive"}, got
    for seam in ("J1_core_strip", "J3_rotor_strip"):
        op = r.seam_operators[seam]
        assert op.effort_normal, f"{seam} declares no effort_normal at all"
        # naming the other side negates S, which makes it positive definite ...
        ev = np.linalg.eigvalsh(0.5 * (op.S + op.S.T))
        assert (-ev).min() > 0.0
        # ... and leaves every singular value exactly where it was
        sv_a = np.linalg.svd(op.S, compute_uv=False)
        sv_b = np.linalg.svd(-op.S, compute_uv=False)
        assert np.allclose(sv_a, sv_b)
