"""W348: R10's case for a cut piece whose coupling supplies its elliptic solve's
boundary data is decided after the scheme (`compiler._r10_scheme`).

The three outcomes on stub schemes; the workbench's `wall-2` graph decertified at
L5, refused at L2 exactly as before when the declaration is withdrawn (the
reduce-to-the-old-path control), and refused at L5 by a sweeping scheme with no
tolerance; and the graph R10 came out of, Tier 0's four windows as built, still
refused at L2.  CS-S1's refusal is pinned by `test_tier38_neural_interface`, and
`scripts/w348_controls.py` measures both negative controls and every workbench
example in one record.
"""

from __future__ import annotations

import dataclasses
import os
import types

import numpy as np
import pytest

from atlas import Budget, compile_scheme
from atlas.compiler import SWEEPING_ACCELERATORS, _r10_scheme
from atlas.scheme import Accelerator
from atlas.verdict import DecisionRecord

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _decide(acc, eps):
    ctx = types.SimpleNamespace(r10_deferred=["a", "b"], record=DecisionRecord())
    _r10_scheme(ctx, types.SimpleNamespace(accelerator=acc, eps_tol=eps))
    return [(d.layer, d.rule, d.verdict.value, d.subject) for d in ctx.record], ctx.record


def test_a_direct_interface_solve_decertifies_with_W168s_cost():
    rows, rec = _decide(Accelerator.DIRECT_SCHUR, None)
    assert rows == [("L5", "R10", "admit-uncertified", "a, b")]
    msg = list(rec)[0].message
    assert "direct-schur" in msg and "149x" in msg and "W168" in msg


@pytest.mark.parametrize("acc", sorted(SWEEPING_ACCELERATORS, key=lambda a: a.value))
def test_a_sweep_to_a_stated_tolerance_decertifies_citing_frommer_and_szyld(acc):
    rows, rec = _decide(acc, 1e-8)
    assert rows == [("L5", "R10", "admit-uncertified", "a, b")]
    msg = list(rec)[0].message
    assert "Frommer & Szyld" in msg and "eps_tol = 1e-08" in msg and "149x" in msg


@pytest.mark.parametrize("acc, eps", [(Accelerator.KRYLOV, None),
                                      (Accelerator.KRYLOV, 0.0),
                                      (Accelerator.RICHARDSON, None), (None, None)])
def test_no_tolerance_to_iterate_to_refuses(acc, eps):
    rows, rec = _decide(acc, eps)
    assert rows == [("L5", "R10", "refuse", "a, b")]
    assert "neither solves the interface directly" in list(rec)[0].message


def test_nothing_deferred_records_nothing():
    ctx = types.SimpleNamespace(r10_deferred=[], record=DecisionRecord())
    _r10_scheme(ctx, types.SimpleNamespace(accelerator=None, eps_tol=None))
    assert len(ctx.record) == 0


# ---------------------------------------------------------------------------
# a real graph: the workbench's wall-2
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wall2():
    from atlas.workbench.compile import case_graph
    from atlas.workbench.spec import example_case
    built = case_graph(example_case("wall-2"))
    return built[0] if isinstance(built, tuple) else built


def _r10(result):
    return [(d.layer, d.rule, d.verdict.value) for d in result.decisions if d.rule == "R10"]


def test_the_pieces_declare_their_data_comes_from_the_ports(wall2):
    caps = [a.capabilities for a in wall2.agents]
    assert caps and all(c.elliptic_data_from_ports for c in caps)
    assert all(c.as_dict()["elliptic_data_from_ports"] is True for c in caps)
    # written only when declared, so records from before W348 keep their bytes
    plain = dataclasses.replace(caps[0], elliptic_data_from_ports=False).as_dict()
    assert "elliptic_data_from_ports" not in plain


def test_wall2_is_decertified_after_the_scheme(wall2):
    r = compile_scheme(wall2)
    assert r.scheme.accelerator is Accelerator.DIRECT_SCHUR
    assert _r10(r) == [("L5", "R10", "admit-uncertified")]
    assert r.verdict.value == "admit-uncertified"


def test_withdrawn_the_declaration_is_refused_at_L2_as_before(wall2):
    """The reduce-to-the-old-path control: the same graph, the declaration
    withdrawn, refused at L2 by R10 as every iterated workbench example was."""
    saved = [a.capabilities.elliptic_data_from_ports for a in wall2.agents]
    try:
        for a in wall2.agents:
            a.capabilities.elliptic_data_from_ports = False
        r = compile_scheme(wall2)
    finally:
        for a, s in zip(wall2.agents, saved):
            a.capabilities.elliptic_data_from_ports = s
    assert _r10(r) == [("L2", "R10", "refuse")]
    assert r.verdict.value == "refuse"


def test_a_sweeping_scheme_follows_its_tolerance(wall2):
    r = compile_scheme(wall2, budget=Budget(allow_direct_schur=False))
    assert r.scheme.accelerator in SWEEPING_ACCELERATORS
    expected = "admit-uncertified" if r.scheme.eps_tol else "refuse"
    assert _r10(r) == [("L5", "R10", expected)]


# ---------------------------------------------------------------------------
# the negative control: the graph R10 came out of
# ---------------------------------------------------------------------------


def _have_tier0() -> bool:
    try:
        from atlas.cases import window_ns as W
        W.load_reference()
    except Exception:                                              # noqa: BLE001
        return False
    return os.path.exists(os.path.join(ROOT, "out", "tier0b", "s0_state.npz"))


@pytest.mark.skipif(not _have_tier0(),
                    reason="the windfarm reference package or out/tier0b/s0_state.npz is missing")
def test_tier0s_four_windows_as_built_still_refuse_at_L2():
    from atlas.cases import window_ns as W
    st = np.load(os.path.join(ROOT, "out", "tier0b", "s0_state.npz"))
    g, _ = W.build(st["u"], st["v"], mode="as-built")
    assert len(g.agents) == 4
    assert not any(a.capabilities.elliptic_data_from_ports for a in g.agents)
    r = compile_scheme(g)
    assert ("L2", "R10", "refuse") in _r10(r)
    assert ("L5", "R10", "admit-uncertified") not in _r10(r)
    assert r.verdict.value == "refuse"
