"""TL0 for CO-T and the classical coarse bases (`atlas/chartop/transport_coarse.py`):
what carries over from the rework's Proposition A to a non-symmetric piece, and what
the expert gives the transport host."""

from __future__ import annotations

import numpy as np
import pytest

from atlas.chartop import fe
from atlas.chartop import ports as pt
from atlas.chartop import transport as tr
from atlas.chartop import transport_cases as tc
from atlas.chartop import transport_coarse as tco
from atlas.chartop.host import monolithic


@pytest.fixture(scope="module")
def piece():
    return tc.channel(n=32, K=2, pe=50.0).problems[0]


@pytest.fixture(scope="module")
def Q():
    return pt.basis("sine-hat", 32, 16).Q


@pytest.mark.parametrize("basis", tco.BASES)
def test_galerkin_orthogonality_and_idempotence(piece, Q, basis):
    t = tco.TransportCoarse(piece.matrix(), 32, 8, basis)
    assert t.P.shape == (31 * 31, 7 * 7)
    assert np.linalg.matrix_rank(t.P.toarray()) == 49
    X = np.random.default_rng(0).standard_normal((31 * 31, Q.shape[1]))
    F, Fs = t.trial_data(Q), t.adjoint_data(Q)
    H, G = t.correct(X, F), t.correct_adjoint(X, Fs)
    assert t.cx(X, F) > 1e-3
    assert t.cx(H, F) <= 1e-12 and t.cx(G, Fs, adjoint=True) <= 1e-12
    assert np.max(np.abs(t.correct(H, F) - H)) <= 1e-12 * np.max(np.abs(H))
    v = np.random.default_rng(1).standard_normal((49, Q.shape[1]))
    assert np.max(np.abs(t.correct(X + t.P @ v, F) - H)) <= 1e-10 * np.max(np.abs(H))


def test_exact_at_the_top(piece, Q):
    exact, _ = tr.exact_response(piece, Q)
    t = tco.TransportCoarse(piece.matrix(), 32, 32)
    zero = np.zeros_like(exact.H)
    assert np.max(np.abs(t.correct(zero, t.trial_data(Q)) - exact.H)) <= 1e-12
    assert np.max(np.abs(t.correct_adjoint(zero, t.adjoint_data(Q)) - exact.G)) <= 1e-12


def test_the_adjoint_correction_is_the_reversed_pieces(piece, Q):
    a = tco.TransportCoarse(piece.matrix(), 32, 8)
    b = tco.TransportCoarse(piece.reversed().matrix(), 32, 8)
    Y = np.random.default_rng(2).standard_normal((31 * 31, Q.shape[1]))
    assert np.max(np.abs(a.correct_adjoint(Y, a.adjoint_data(Q)) - b.correct(Y, b.trial_data(Q)))) <= 1e-12


def test_lemma_b_on_a_steady_piece(piece, Q):
    c1 = np.zeros(Q.shape[1])
    c1[:4] = 1.0                                      # the hats' sum is the constant
    for basis in tco.BASES:
        r = tco.TL0TransportExpert(n_c=8, basis=basis).response(piece, Q)
        assert np.max(np.abs(r.H @ c1 - 1.0)) <= 1e-12
        assert np.max(np.abs(r.G @ c1 - 1.0)) <= 1e-12


def test_the_conforming_msfem_basis_is_a_partition_of_unity_with_bilinear_traces():
    """Conforming MsFEM: every cell's basis is K-harmonic inside it with the bilinear
    hats' traces, so with the loop's hats added the columns sum to one."""
    p = tc.channel(n=32, K=2, pe=50.0).problems[0]
    P = tco.msfem_prolongation(p.matrix(), 32, 8, 0).toarray()
    Pb = tco.prolongation(p.matrix(), 32, 8, "bilinear").toarray()
    # on the coarse grid's lines the multiscale functions are the hats
    t = np.arange(1, 32)
    on_lines = ((t[None, :] % 4 == 0) | (t[:, None] % 4 == 0)).ravel()
    assert np.max(np.abs(P[on_lines] - Pb[on_lines])) <= 1e-12


def test_tl0_couples_and_halves_with_the_coarse_size():
    b = tc.channel(n=32, K=2, pe=20.0)
    L = b.layout(m=16)
    mono = monolithic(L)
    errs = [L.solve(tco.TL0TransportExpert(n_c=nc)).errors(mono.fields)["energy_rel"] for nc in (4, 8, 16)]
    assert errs[0] > errs[1] > errs[2]
    assert 0.3 < errs[2] / errs[1] < 0.7                # a resolution limit, not a stall
    full = L.solve(tco.TL0TransportExpert(n_c=32)).errors(mono.fields)["energy_rel"]
    cl = L.solve(tr.ClassicalTransportExpert()).errors(mono.fields)["energy_rel"]
    assert full == pytest.approx(cl, rel=1e-9)          # n_c = n: the classical expert


def test_unknown_options():
    with pytest.raises(ValueError):
        tco.TL0TransportExpert(principal="exact")
    with pytest.raises(ValueError):
        tco.TL0TransportExpert(basis="spectral")
    with pytest.raises(ValueError):
        tco.prolongation(None, 32, 8, "spectral")
