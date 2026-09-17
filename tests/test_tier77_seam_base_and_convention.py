"""Tier 77 -- the b-c seam's base and its assembly convention, and W306.

Tier 76 published ``beta = 0.149619`` for the rocket's b-c seam **with a
decertification attached** saying the two sides were linearised 1374 apart on M,
876% of the base norm, so it is "not merely the wrong point but no point at
all"; and it flipped `E7` to `fails` with no verdict on whether that was physics
or a convention.  This tier settles both, and both answers went against the
predictions registered for them.

The tests pin four things:

* **the sum residual has no root and the difference does** -- the evidence that
  the assembly convention was wrong, and it is a root rather than an argument;
* **the corrected numbers**, so a future change that moves them is loud;
* **`thermal_seam`'s E7 holds by an accident of its film coefficient**, not by a
  structural property of conjugate heat transfer;
* **W306**: `passivity_defect` cannot distinguish a one-signed spectrum from a
  mixed one, and those are different physical statements.

The declaration tests run at ``dt_gas = 1e-6``: they decide a DECLARATION, not a
physical number, so the cadence is chosen for cost and said out loud.
"""
from __future__ import annotations

import numpy as np
import pytest

from atlas.cases import rocket, rocket_experts as RE
from atlas.probe import ProbedBlock, _fill_diagnostics, probe_block

DT_GAS_TEST = 1.0e-6


def _have_expert() -> bool:
    try:
        RE.load_rocket_modules()
        return True
    except Exception:
        return False


needs_expert = pytest.mark.skipif(
    not _have_expert(), reason="the build repo is not importable; set ATLAS_BUILD_REPO")


def _bc(dt_gas=DT_GAS_TEST):
    shell, gas, _prov = RE.make_bc_experts(dt_gas=dt_gas)
    graph = RE.build_bc_graph(shell, gas)
    from atlas.compiler import _Context, _derive_transfer          # noqa: PLC2701
    from atlas.envelope import EnvelopeStamp
    from atlas.holes import HoleLedger
    from atlas.probe import ProbeBudget
    from atlas.verdict import DecisionRecord
    ctx = _Context(graph=graph, budget=None, probe_budget=ProbeBudget(), references={},
                   record=DecisionRecord(), stamp=EnvelopeStamp(), holes=HoleLedger(),
                   probe_state="t77", depth=0)
    return shell, gas, graph, _derive_transfer(ctx, graph.connections[0])


def _uniform(P, value):
    M = P.effective_matrix()
    mu, _r, _rk, _sv = np.linalg.lstsq(M, np.full(M.shape[0], float(value)), rcond=None)
    return mu


# ===========================================================================
# 1. the declaration, and where it is and is not made
# ===========================================================================


def test_effort_normal_is_declared_only_where_it_was_measured():
    """One seam has evidence; the other six do not.

    A declaration copied across a graph on the strength of one measurement is
    the defaulting this vault keeps finding, so the map has exactly one entry.
    """
    assert rocket.EFFORT_NORMAL == {"b-c:THERM": "b"}
    g = rocket.build(declarations="known")
    named = {c.seam_id: c.effort_normal for c in g.connections if c.effort_normal}
    assert named == {"b-c:THERM": "b"}
    conn = [c for c in g.connections if c.seam_id == "b-c:THERM"][0]
    assert conn.effort_signs == {"b": 1.0, "c": -1.0}


def test_the_fixture_level_is_untouched_by_the_declaration():
    """Tier 76's byte-identity control is pinned to the fixture level, and a
    convention declared at Tier 77 must not reach back into it."""
    g = rocket.build(declarations="fixture")
    assert all(not c.effort_normal for c in g.connections)


# ===========================================================================
# 2. the evidence: a root, not an argument
# ===========================================================================


@needs_expert
def test_the_sum_residual_has_no_root_and_the_difference_does():
    """**The measurement that decided the convention.**

    `generate.py` hands the GAS's own conduction-limited h to the SHELL's Robin
    channel, so both sides carry the same film coefficient and the interface
    state cancels out of their SUM to first order:

        q_gas = h (T_i - lam),  q_shell = h (lam - T_face(lam))
        sum   = h (T_i - T_face(lam))          -- lam cancels

    so the sum's root is thermal equilibrium, which one macro step cannot reach,
    and it has no sign change anywhere in the admissible interval. The first
    attempt at this tier ran Newton on that residual and was killed after 204 s
    of CPU without converging; the scan is both cheaper and the actual evidence.
    """
    shell, gas, _g, tr = _bc()
    Pb, Pc = tr.prolongations["b"], tr.prolongations["c"]
    Rb, Rc = Pb.adjoint(tr.space), Pc.adjoint(tr.space)

    def halves(v):
        fb = np.asarray(gas.respond("c:THERM", Pb.prolong(_uniform(Pb, v), tr.space)), float)
        fc = np.asarray(shell.respond("b:THERM", Pc.prolong(_uniform(Pc, v), tr.space)), float)
        return Rb @ fb, Rc @ fc

    sums, diffs = [], []
    for v in np.linspace(RE.T_AMBIENT, RE.T_CHAMBER, 7):
        gb, gc = halves(v)
        one = np.ones_like(gb)
        sums.append(float(np.dot(gb + gc, one)))
        diffs.append(float(np.dot(gb - gc, one)))
    assert all(x > 0 for x in sums), sums          # no sign change at all
    assert diffs[0] > 0 > diffs[-1]                 # exactly one crossing


# ===========================================================================
# 3. the corrected numbers
# ===========================================================================


@needs_expert
def test_the_difference_is_one_signed_and_the_sum_is_not_at_the_root():
    """The heart of the correction, and of W306.

    At the CONSISTENT base the sum is genuinely MIXED -- it has an amplified
    mode -- while the difference is one-signed. `L4/E7/passivity` rejects both,
    because it tests ``lambda_min > 0`` and never looks at ``lambda_max``.
    """
    shell, gas, g, tr = _bc()
    caps = {a.agent_id: a.capabilities for a in g.agents}
    ports = {"b": "c:THERM", "c": "b:THERM"}
    blocks = {}
    for aid in ("b", "c"):
        P = tr.prolongations[aid]
        blocks[aid] = probe_block(
            caps[aid], caps[aid].port(ports[aid]), tr.space, P, "t77",
            base_V=P.prolong(_uniform(P, 1035.487), tr.space)).S

    def spectrum(S):
        return np.linalg.eigvalsh(0.5 * (S + S.T))

    ev_sum = spectrum(blocks["b"] + blocks["c"])
    ev_dif = spectrum(blocks["b"] - blocks["c"])
    assert ev_sum.min() < 0.0 < ev_sum.max(), "the sum should be MIXED at lambda*"
    assert ev_dif.max() < 0.0, "the difference should be one-signed"
    # and the difference's smallest singular value is two orders larger
    s_sum = np.linalg.svd(blocks["b"] + blocks["c"], compute_uv=False).min()
    s_dif = np.linalg.svd(blocks["b"] - blocks["c"], compute_uv=False).min()
    assert s_dif / s_sum > 50.0, (s_dif, s_sum)


@needs_expert
def test_two_errors_that_nearly_cancelled():
    """**Why Tier 76's number looked unremarkable.**

    It was wrong twice -- the wrong base AND the wrong assembly -- and the two
    errors nearly cancelled. Fixing either one alone moves it far further than
    fixing both.
    """
    shell, gas, g, tr = _bc()
    caps = {a.agent_id: a.capabilities for a in g.agents}
    ports = {"b": "c:THERM", "c": "b:THERM"}

    def sigma_min(base, sign_c):
        blk = {}
        for aid in ("b", "c"):
            P = tr.prolongations[aid]
            bv = None if base is None else P.prolong(_uniform(P, base), tr.space)
            blk[aid] = probe_block(caps[aid], caps[aid].port(ports[aid]), tr.space,
                                   P, "t77", base_V=bv).S
        S = blk["b"] + sign_c * blk["c"]
        return float(np.linalg.svd(S, compute_uv=False).min())

    wrong_wrong = sigma_min(None, +1.0)          # Tier 76 as published
    right_wrong = sigma_min(1035.487, +1.0)      # right base, wrong assembly
    wrong_right = sigma_min(None, -1.0)          # wrong base, right assembly
    right_right = sigma_min(1035.487, -1.0)      # the corrected number
    # fixing ONE of the two moves it much further than fixing BOTH
    assert right_wrong < 0.02 * wrong_wrong, (right_wrong, wrong_wrong)
    assert 1.0 < wrong_right / wrong_wrong < 2.0
    assert 0.5 < wrong_wrong / right_right < 2.0, (wrong_wrong, right_right)


# ===========================================================================
# 4. thermal_seam's E7 holds by an accident, not a property
# ===========================================================================


@needs_expert
def test_thermal_seams_passivity_is_a_film_coefficient_accident():
    """Both CHT seams have the same SIGN structure -- fluid block negative, solid
    block positive -- so the sum cancels and the difference adds. Whether the sum
    is definite is then a question of which block DOMINATES, and that is a
    modelling choice: `thermal_seam` declares ``H_IN_NOMINAL = 500``
    independently of its gas, while the rocket follows `generate.py` and hands
    the shell the gas's own h, so the rocket's two blocks nearly cancel.

    This matters because `thermal_seam`'s E7 `holds` is quoted as though it were
    a property of conjugate heat transfer. It is a property of a 500.
    """
    from atlas.cases import thermal_seam as T
    graph, experts = T.build(mode="split-step", clocks="matched")
    from atlas.compiler import _Context, _derive_transfer          # noqa: PLC2701
    from atlas.envelope import EnvelopeStamp
    from atlas.holes import HoleLedger
    from atlas.probe import ProbeBudget
    from atlas.verdict import DecisionRecord
    ctx = _Context(graph=graph, budget=None, probe_budget=ProbeBudget(), references={},
                   record=DecisionRecord(), stamp=EnvelopeStamp(), holes=HoleLedger(),
                   probe_state="t77", depth=0)
    tr = _derive_transfer(ctx, graph.connections[0])
    caps = {a.agent_id: a.capabilities for a in graph.agents}
    ports = {"gas": "wall:THERM", "shell": "inner:THERM"}
    blk = {}
    for aid in ("gas", "shell"):
        P = tr.prolongations[aid]
        blk[aid] = probe_block(caps[aid], caps[aid].port(ports[aid]), tr.space, P,
                               "t77", base_V=P.prolong(_uniform(P, 453.125), tr.space)).S
    ev_gas = np.linalg.eigvalsh(0.5 * (blk["gas"] + blk["gas"].T))
    ev_shell = np.linalg.eigvalsh(0.5 * (blk["shell"] + blk["shell"].T))
    assert ev_gas.max() < 0.0, "the fluid block should be negative"
    assert ev_shell.min() > 0.0, "the solid block should be positive"
    # and the solid dominates by an order of magnitude, which is what saves the sum
    assert ev_shell.max() / abs(ev_gas.min()) > 8.0
    ev_sum = np.linalg.eigvalsh(0.5 * ((blk["gas"] + blk["shell"])
                                       + (blk["gas"] + blk["shell"]).T))
    assert ev_sum.min() > 0.0, "thermal_seam's SUM is definite -- by the dominance"


# ===========================================================================
# 5. W306 -- the diagnostic cannot tell the two apart
# ===========================================================================


def test_W306_passivity_defect_cannot_tell_a_global_sign_from_an_amplified_mode():
    """**The rule gap this tier names, tested on the diagnostic itself.**

    `_fill_diagnostics` sets ``passivity_defect = abs(lambda_min)`` whenever
    ``lambda_min`` is below the tolerance, and never reads ``lambda_max``. So a
    matrix that is ONE-SIGNED NEGATIVE -- passive up to a global sign, which the
    interface solve is indifferent to because chi flips with S -- reports the
    same kind of defect as one with a genuinely amplified mode.

    Built as pure algebra rather than through a case study, because the claim is
    about the statistic and not about a rocket.
    """
    # The two must share a lambda_min, or the test is about the eigenvalue and
    # not about the statistic. Both have lambda_min = -3; they differ only in
    # lambda_max, which is the quantity nothing reads.
    one_signed = np.diag([-3.0, -2.0, -1.0])          # every mode damped, sign flipped
    mixed = np.diag([-3.0, 2.0, 3.0])                 # one mode genuinely amplified
    out = {}
    for tag, S in (("one_signed", one_signed), ("mixed", mixed)):
        blk = ProbedBlock(agent_id="t", seam_id="s", S=S, route="assembled",
                          n_solves=0, dim_M=3)
        _fill_diagnostics(blk, S)
        out[tag] = (blk.passivity_defect, blk.passivity_lambda_min)
    # both report a defect ...
    assert out["one_signed"][0] > 0 and out["mixed"][0] > 0
    # ... and it is the SAME defect, from the same lambda_min
    assert out["one_signed"][0] == pytest.approx(out["mixed"][0])
    assert out["one_signed"][1] == pytest.approx(out["mixed"][1])
    # while the two matrices are different in the way that matters:
    assert np.linalg.eigvalsh(one_signed).max() < 0.0
    assert np.linalg.eigvalsh(mixed).max() > 0.0
    # **CLOSED at Tier 78 (W306), and the diagnosis above is kept rather than
    # rewritten.** This test originally ended by asserting that
    # `passivity_lambda_max` did NOT exist -- that was the gap it named. It does
    # now, and everything above still holds: the OLD quantity still cannot tell
    # the two apart, which is exactly why a new one was needed.
    blk_one = ProbedBlock(agent_id="t", seam_id="s", S=one_signed,
                          route="assembled", n_solves=0, dim_M=3)
    blk_mix = ProbedBlock(agent_id="t", seam_id="s", S=mixed,
                          route="assembled", n_solves=0, dim_M=3)
    _fill_diagnostics(blk_one, one_signed)
    _fill_diagnostics(blk_mix, mixed)
    assert blk_one.sign_structure == "negative"
    assert blk_mix.sign_structure == "mixed"
    assert blk_one.passivity_lambda_max < 0.0 < blk_mix.passivity_lambda_max
