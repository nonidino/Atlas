"""Tier 39, W167 -- `alpha_star` wired as a transmission condition, with its guard.

`atlas/probe.py`'s `robin_condition` and `RobinCondition`; `atlas/scheme.py`'s
`Accelerator.ROBIN_RICHARDSON`; `atlas/solve.py`'s `_robin_richardson`.

**The field is older than every rule that could have used it.**  `alpha_star` is
``diag(S)``, which in a Fourier interface basis IS the measured symbol mode by
mode -- the optimal Robin coefficient, read off the probe rather than derived
from an operator nobody has, which for a frozen expert is the only route to one.
The probe has emitted it since Tier 0.  Until this tier the only thing in the
vault that read it was `SeamOperator.mode_shares`, which reads it as a ratio.

**And it cannot be used bare.**  CS-S1 measured both halves on the as-built
neural-interface seam: preconditioning Richardson by ``D = diag(alpha_star)``
cut the seam's condition number from 22.686 to 7.345 **and diverged**, because
Richardson contracts iff the preconditioned spectrum sits inside ``(0, 2)`` and
``rho(D^-1 S) = 2.976`` does not.  Damped at ``omega = 1/rho = 0.336`` the same
coefficient beat plain Richardson 594 -> 172 expert calls to ``tol = 1e-6``,
counting the 34-call probe both arms need.  A better-conditioned operator that
diverges is exactly the shape of number this framework exists to stop being
quoted, so the coefficient and the spectral radius that makes it safe are
produced together, by one function, off one S.

Four groups, and each can fail:

  1. **the coefficient is the measured one** -- the damping this code chooses
     matches the ladder's ``1/rho`` on both recorded arrangements, read out of
     `out/w166/w166.json` rather than retyped;
  2. **it pays** -- on a passive (SPD) seam, which is the class
     `L4/E7/passivity` certifies and the class CS-S1's arrangement is, the wired
     accelerator reaches the same answer at the same tolerance in strictly fewer
     sweeps than plain Richardson;
  3. **the guard fires on the arrangements where the coefficient is not
     available**, and there are two of them -- a singular diagonal, and a
     preconditioned spectrum that is not positive, where NO damping is a
     contraction.  The second is a defect this tier's own first version had:
     it damped by ``1/rho`` and called that safe, which is right for a positive
     real spectrum and false for an indefinite one, and
     `_robin_richardson`'s divergence check caught it at iteration 707;
  4. **the decline is not a silent fallback** -- a seam that cannot take this
     accelerator raises rather than quietly running plain Richardson under this
     accelerator's name, which is R3's complaint about a scheme wearing a longer
     name, one object along.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas.probe import RobinCondition, robin_condition        # noqa: E402
from atlas.scheme import Accelerator                           # noqa: E402
from atlas.solve import InterfaceProblem, solve_interface      # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_W166 = os.path.join(_ROOT, "out", "w166", "w166.json")


def _recorded():
    if not os.path.isfile(_W166):
        pytest.skip(f"{_W166} is absent; run scripts/w166_neural_interface.py")
    with open(_W166) as f:
        return json.load(f)["D_accelerators"]


def _spd_seam(n: int = 16, seed: int = 3, spread: float = 60.0):
    """A PASSIVE seam: symmetric positive definite, poorly diagonally scaled.

    For SPD ``S`` the preconditioned ``D^-1 S`` is similar to the SPD matrix
    ``D^-1/2 S D^-1/2``, so its spectrum is positive real -- which is the
    hypothesis ``omega = 1/rho`` needs, and the reason group 3's indefinite seam
    is a different case rather than a harder one.
    """
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.normal(size=(n, n)))[0]
    S = Q @ np.diag(np.logspace(0, np.log10(spread), n)) @ Q.T
    return 0.5 * (S + S.T), rng


# ---------------------------------------------------------------------------
# 1. the coefficient is the one that was measured
# ---------------------------------------------------------------------------


def test_the_damping_is_the_one_the_measured_ladder_used():
    """Read off the artifact, not retyped: a number and its run travel together."""
    rec = _recorded()
    for arm in ("as-built", "split-step"):
        rho = float(rec[arm]["probe"]["rho_D_inv_S"])
        cond = RobinCondition(alpha=np.ones(1), rho=rho,
                              omega=min(1.0, 1.0 / rho), usable=True)
        assert cond.omega == pytest.approx(1.0 / rho, rel=1e-15)


def test_the_measurement_this_is_wired_from_is_still_on_disk_and_says_what_it_said():
    """The claim the accelerator rests on, asserted against the artifact.

    If this fails the accelerator has not changed -- the evidence for it has --
    and every sentence written about `alpha_star` in this tier needs re-reading.
    """
    rec = _recorded()
    ab = {r["accelerator"]: r for r in rec["as-built"]["ladder"]
          if r["tol_frac"] == 1e-6}
    #: undamped DIVERGES while cutting kappa 3x -- the whole point
    assert rec["as-built"]["probe"]["kappa"] == pytest.approx(22.686, abs=1e-2)
    assert rec["as-built"]["probe"]["kappa_preconditioned"] == pytest.approx(
        7.3448, abs=1e-3)
    assert rec["as-built"]["probe"]["rho_D_inv_S"] == pytest.approx(
        2.9764, abs=1e-3)
    assert not ab["richardson, alpha_star undamped"]["converged"]
    #: and damped it pays, counting the probe both arms need
    assert ab["richardson, theta = 1/||S||"]["total_calls"] == 594
    assert ab["richardson, alpha_star damped"]["total_calls"] == 172


# ---------------------------------------------------------------------------
# 2. it pays on the class it was measured on
# ---------------------------------------------------------------------------


def test_on_a_passive_seam_the_robin_condition_beats_plain_richardson():
    S, rng = _spd_seam()
    n = S.shape[0]
    cond = robin_condition(S)
    assert cond is not None and cond.usable, cond.why
    #: the arrangement is the measured one: rho outside the disc, so the
    #: undamped coefficient would diverge and the damping is what buys the win
    assert cond.rho > 2.0
    assert cond.omega == pytest.approx(1.0 / cond.rho)
    assert cond.contraction < 1.0

    chi = rng.normal(size=n)
    prob = InterfaceProblem("passive", S, chi, n)
    ref = np.linalg.solve(S, chi)
    tol = 1e-6 * float(np.linalg.norm(prob.residual(np.zeros(n))))

    iters = {}
    for acc in (Accelerator.RICHARDSON, Accelerator.ROBIN_RICHARDSON):
        lam, k, hist = solve_interface(prob, acc, tol=tol, max_iter=20000)
        iters[acc] = k
        assert hist[-1] <= tol, (acc, hist[-1])
        #: the SAME answer -- an accelerator that converges somewhere else is
        #: not an accelerator, and `solve_interface`'s own docstring says it
        #: cannot fix a problem posed with the wrong operator
        assert np.linalg.norm(lam - ref) / np.linalg.norm(ref) < 1e-4

    assert iters[Accelerator.ROBIN_RICHARDSON] < iters[Accelerator.RICHARDSON]


def test_the_undamped_coefficient_is_what_would_have_diverged():
    """The guard is load-bearing, not decorative: run the same seam undamped.

    This is CS-S1's finding reproduced inside the suite rather than cited from
    an artifact -- omega = 1 on a seam with rho > 2, which is what using
    `alpha_star` bare means.
    """
    S, rng = _spd_seam()
    n = S.shape[0]
    cond = robin_condition(S)
    assert cond.rho > 2.0

    chi = rng.normal(size=n)
    prob = InterfaceProblem("passive", S, chi, n)
    lam = np.zeros(n)
    M_inv_bare = np.diag(1.0 / np.diag(S))          # omega = 1: no damping
    for _ in range(400):
        lam = lam - M_inv_bare @ prob.residual(lam)
    nr = float(np.linalg.norm(prob.residual(lam)))
    assert not np.isfinite(nr) or nr > 1.0, nr

    #: and the damped one on the identical seam converges
    lam, k, hist = solve_interface(prob, Accelerator.ROBIN_RICHARDSON,
                                   tol=1e-8, max_iter=20000)
    assert np.isfinite(hist[-1]) and hist[-1] <= 1e-8


# ---------------------------------------------------------------------------
# 3. the guard, on both ways the coefficient is unavailable
# ---------------------------------------------------------------------------


def test_an_indefinite_seam_is_declined_because_no_damping_contracts_it():
    """**The defect this tier's own first version had.**

    Damping by ``1/rho`` puts the largest-MAGNITUDE eigenvalue at one; it says
    nothing about a NEGATIVE one, and Richardson's iteration matrix
    ``I - omega D^-1 S`` sends a negative ``mu`` above one for every positive
    omega.  The first version checked ``rho`` and called that safe.  It was
    caught by `_robin_richardson`'s divergence check at iteration 707, and the
    predicate is now the contraction factor itself rather than a proxy for it.
    """
    rng = np.random.default_rng(0)
    n = 16
    A = rng.normal(size=(n, n))
    S = 0.5 * (A + A.T) + np.diag(np.full(n, 1.5))
    ev = np.linalg.eigvals(S / np.diag(S)[:, None])
    assert np.any(np.real(ev) <= 0.0), "the seam is not indefinite"

    cond = robin_condition(S)
    assert cond is not None and not cond.usable
    assert cond.contraction >= 1.0
    assert "not a contraction" in cond.why
    assert cond.M_inv is None


def test_a_zero_diagonal_entry_declines_the_PRECONDITIONER_and_not_the_seam():
    """A zero DIAGONAL entry is not a zero row, and the message has to say so.

    The distinction is the whole content: the seam can be perfectly well posed
    while having a mode its two agents do not respond to on the diagonal, and a
    message that refused the seam would be refusing something that is fine.
    """
    S, _rng = _spd_seam()
    S = S.copy()
    S[3, 3] = 0.0
    cond = robin_condition(S)
    assert cond is not None and not cond.usable
    assert "singular" in cond.why
    assert "not the seam" in cond.why
    assert cond.M_inv is None


def test_a_non_square_operator_has_no_diagonal_symbol_to_read():
    assert robin_condition(np.ones((3, 5))) is None
    assert robin_condition(np.zeros((0, 0))) is None


# ---------------------------------------------------------------------------
# 4. the decline is not a silent fallback
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["indefinite", "singular-diagonal"])
def test_the_solver_raises_rather_than_quietly_running_plain_richardson(kind):
    rng = np.random.default_rng(0)
    n = 16
    if kind == "indefinite":
        A = rng.normal(size=(n, n))
        S = 0.5 * (A + A.T) + np.diag(np.full(n, 1.5))
        needle = "not a contraction"
    else:
        S, _ = _spd_seam(n=n)
        S = S.copy()
        S[3, 3] = 0.0
        needle = "zero diagonal response"
    chi = rng.normal(size=n)
    prob = InterfaceProblem(kind, S, chi, n)

    with pytest.raises(ValueError) as e:
        solve_interface(prob, Accelerator.ROBIN_RICHARDSON, tol=1e-10)
    msg = str(e.value)
    assert needle in msg
    #: and it names the alternative rather than leaving the caller stuck
    assert "another accelerator" in msg


def test_the_accelerator_is_declared_in_the_enum_and_is_distinct():
    """A branch nothing can select is not wired.  `Accelerator` is what the
    scheme tuple carries, so the value has to exist there and not only in
    `solve_interface`'s dispatch."""
    assert Accelerator.ROBIN_RICHARDSON.value == "robin-richardson"
    assert Accelerator.ROBIN_RICHARDSON is not Accelerator.RICHARDSON
    assert len({a.value for a in Accelerator}) == len(list(Accelerator))


def test_the_empty_seam_is_still_the_empty_seam():
    """R6's arrangement: ``Lambda == 0``.  Every trace is equally consistent and
    the honest behaviour is to return the initial guess, which is checked here
    because the new branch sits after that early return and could have moved it.
    """
    n = 4
    prob = InterfaceProblem("empty", np.zeros((n, n)), np.zeros(n), n)
    lam, k, hist = solve_interface(prob, Accelerator.ROBIN_RICHARDSON)
    assert k == 1 and np.allclose(lam, 0.0) and len(hist) == 1


# ===========================================================================
# W168 -- two rules whose messages did not say what they were worth
# ===========================================================================
#
# Neither of these is a new rule and neither changes a verdict. They are the
# TEXT of decisions the compiler already made, and the row is about text
# because a refusal that reads as bookkeeping when it is the largest measured
# effect in the package is a refusal people route around.


def _field():
    for name in ("w141", "w136"):
        p = os.path.join(_ROOT, "out", name, "settled.npz")
        if os.path.isfile(p):
            d = np.load(p)
            return d["u"], d["v"]
    pytest.skip("no settled field on disk")


def test_R10s_refusal_says_what_taking_the_repair_is_worth():
    """149x in coupling sweeps, and the message used to be silent about it.

    The number is asserted against `out/w166/w166.json` as well as against the
    message, so the two cannot drift: a figure quoted in a rule and a figure in
    the artifact that measured it have to travel together, which is the lesson
    the PoC-1a millisecond table taught this project.
    """
    from atlas import compile_scheme
    from atlas.cases import front_wing as FW

    rec = _recorded() if os.path.isfile(_W166) else None
    with open(_W166) as f:
        art = json.load(f)["A_exchange"]
    spd = art["sweeps_per_decade"]
    embedded = spd["as-built (per-window pressure solve)"]
    exposed = spd["split-step + global projection"]
    assert embedded == pytest.approx(74.81, abs=0.01)
    assert exposed == pytest.approx(0.502, abs=0.001)
    assert art["embedded_over_exposed"] == pytest.approx(149.0, abs=0.5)
    #: the control that makes it the EXPOSURE and not the projection
    assert spd["split-step, no projection [control]"] == pytest.approx(
        0.486, abs=0.002)
    assert rec is not None

    u, v = _field()
    g, _e = FW.build(u, v, motion=False,
                     struct_family="incompressible-navier-stokes-2d")
    r = compile_scheme(g)
    r10 = [d for d in r.decisions.refusals if d.rule == "R10"]
    assert len(r10) == 1, [f"{d.layer}/{d.rule}" for d in r.decisions.refusals]
    msg = r10[0].message
    assert "149x in coupling sweeps" in msg
    assert "74.81" in msg and "0.50" in msg
    assert "the gain is the EXPOSURE and not the projection" in msg


def test_the_halo_rule_says_WHICH_quantity_it_is_bounding():
    """Accuracy always; convergence too when a cut agent is embedded.

    Measured, the same number is a sharp convergence threshold in one
    arrangement and an accuracy margin in the other -- the as-built contraction
    crosses 1 between halo 18 and halo 20 against a declared reach of exactly
    20, while the exposed arrangement converges at every halo from 4 up. A
    reader told only to "widen the overlap" cannot tell whether a narrower one
    costs them accuracy or costs them the fixed point.
    """
    from dataclasses import replace

    from atlas import compile_scheme
    from atlas.capability import EllipticSubsolve
    from atlas.cases import front_wing as FW

    #: the artifact the sentence quotes, so the two cannot drift
    with open(_W166) as f:
        halo = json.load(f)["B_halo"]
    ab = halo["as-built"]["threshold"]
    assert ab["largest_diverging"] == 18 and ab["smallest_converging"] == 20
    assert ab["declared_reach"] == 20 and ab["bracket_contains_reach"]
    assert halo["split-step"]["threshold"]["smallest_converging"] == 4
    rows = {r["halo"]: r for r in halo["as-built"]["rows"]}
    assert rows[18]["contraction"] == pytest.approx(1.00642, abs=1e-4)
    assert rows[20]["contraction"] == pytest.approx(0.980531, abs=1e-4)

    u, v = _field()
    g, _e = FW.build(u, v, motion=False)

    #: every cut agent EXPOSED -> accuracy alone
    r = compile_scheme(g)
    hal = [d for d in r.decisions if d.rule == "R10/halo"]
    assert len(hal) == 1
    assert hal[0].evidence.get("bounds") == "accuracy"
    assert "is ACCURACY" in hal[0].message
    assert "at EVERY halo from 4 up" in hal[0].message

    #: one cut agent EMBEDDED -> accuracy AND convergence, on the same graph
    agents = [replace(a, capabilities=replace(
        a.capabilities, elliptic_subsolve=EllipticSubsolve.EMBEDDED))
        if a.agent_id == "F00" else a for a in g.agents]
    r2 = compile_scheme(replace(g, agents=agents))
    hal2 = [d for d in r2.decisions if d.rule == "R10/halo"]
    assert len(hal2) == 1
    assert hal2[0].evidence.get("bounds") == "accuracy and convergence"
    assert "ACCURACY *and* CONVERGENCE" in hal2[0].message
    assert "18 (1.00642)" in hal2[0].message and "20 (0.980531)" in hal2[0].message
