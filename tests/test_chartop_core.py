"""The chart operator's classical core (Stage 0, W355): bilinear elements, port bases,
the plain square's fast solve, charts, Proposition 3's certificate, and the tier-0
host with exact classical modes.  Every check here is algebra or a classical solve;
none involves a network.

The Lean statements these mirror: T25 (`Atlas.Piece.gram_posSemidef`,
`Atlas.Piece.gram_sub_gram`, `Atlas.assemble_gram_sub_posSemidef`) and T26
(`Atlas.Piece.labelFree_eq_supervised_add_const`) -- a floating-point guard against
an implementation that does not compute what the theorems are about, not a proof.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from atlas.chartop import certificate as cert
from atlas.chartop import chart as ch
from atlas.chartop import fe, host, ports, principal
from atlas.chartop.piece import PieceProblem, exact_response


def _random_tensor(n, rng, k_mu=0.5, contrast=4.0):
    re, im = rng.normal(size=(n, n)), rng.normal(size=(n, n))
    mu = (re + 1j * im)
    mu *= k_mu / np.abs(mu).max()
    kap = np.exp(rng.uniform(0, math.log(contrast), (n, n)))
    a11, a12, a22 = ch.tensor_from_mu(mu)
    return kap * a11, kap * a12, kap * a22


def _curved_piece(n=24):
    sides = [ch.side_from_function(lambda t: np.stack([t, 0.08 * np.sin(math.pi * t)], -1)),
             ch.side_from_function(lambda t: np.stack([1 + 0.1 * np.sin(math.pi * t), t], -1)),
             ch.side_from_function(lambda t: np.stack([1 - t, 1 + 0.05 * np.sin(2 * math.pi * t)], -1)),
             ch.line((0.0, 1.0), (0.0, 0.0))]
    return sides


def test_energy_formula_is_the_matrix():
    rng = np.random.default_rng(1)
    n = 12
    a11, a12, a22 = _random_tensor(n, rng)
    sig = rng.uniform(0, 3, (n, n))
    M = fe.assemble(n, a11, a12, a22, sig)
    U = rng.normal(size=((n + 1) ** 2, 4))
    e1 = np.einsum("ik,ik->k", U, M @ U)
    e2 = fe.energy(a11, a12, a22, sig, U.T.reshape(4, n + 1, n + 1))
    assert np.max(np.abs(e1 - e2)) <= 1e-12 * np.max(np.abs(e1))
    assert abs(M - M.T).max() == 0.0


@pytest.mark.parametrize("kind", ports.KINDS)
@pytest.mark.parametrize("s0", [0.0, 12.5])
def test_principal_part_is_the_plain_square_exactly(kind, s0):
    n = 16
    pb = ports.basis(kind, n, 8)
    H0 = principal.modes(n, pb.Q, s0)
    H = fe.Exact(fe.assemble(n, 1.0, 0.0, 1.0, s0), n).modes(pb.Q)
    assert np.max(np.abs(H0 - H)) <= 1e-12 * np.max(np.abs(H))


def test_port_bases():
    n, m = 16, 6
    sh = ports.basis("sine-hat", n, m)
    assert sh.size == 4 * m
    assert np.allclose(sh.Q[:, :4].sum(axis=1), 1.0)        # the constant is the hats' sum
    co = ports.basis("cosine", n, m)
    assert np.allclose(co.Q[:, [co.side_columns(s)[0] for s in range(4)]].sum(axis=1), 1.0)
    # reversal: a column's values read backwards along its side equal its sign times itself
    for pb in (sh, co):
        for k, col in enumerate(pb.columns):
            if col.what != "side":
                continue
            pos = ports.side_positions(n, col.index)
            v = pb.Q[pos, k]
            inner = slice(1, n)                                 # corners belong to others
            assert np.allclose(v[::-1][inner], pb.reversal_sign(k) * v[inner])


def test_gram_identity_psd_and_label_free_objective():
    """T25 (i), (ii) and T26 on a random piece with random 'network' fields."""
    rng = np.random.default_rng(2)
    n = 16
    a11, a12, a22 = _random_tensor(n, rng)
    sig = rng.uniform(0, 5, (n, n))
    M = fe.assemble(n, a11, a12, a22, sig)
    pb = ports.basis("sine-hat", n, 8)
    ex = fe.Exact(M, n)
    H = ex.modes(pb.Q)
    Lam = ex.port_matrix(pb.Q, H)
    f = rng.normal(size=(n - 1) ** 2)
    up = ex.particular(f)
    for scale in (1e-3, 1.0, 30.0):
        E = scale * rng.normal(size=H.shape)
        Lt = fe.gram(M, fe.full_field(n, H + E, pb.Q))
        ident = Lt - Lam - E.T @ (ex.A_I @ E)
        assert np.max(np.abs(ident)) <= 1e-10 * max(1.0, np.max(np.abs(Lt)))
        assert np.linalg.eigvalsh(Lt)[0] >= -1e-10 * np.max(np.abs(Lt))
    # T26: L_en - L_sup is the same constant for any two outputs
    def l_en(Hh, uh):
        return (np.trace(fe.gram(M, fe.full_field(n, Hh, pb.Q)))
                + uh @ (ex.A_I @ uh) - 2 * f @ uh)

    def l_sup(Hh, uh):
        D = Hh - H
        return np.einsum("ik,ik->", D, ex.A_I @ D) + (uh - up) @ (ex.A_I @ (uh - up))
    outs = [(H + rng.normal(size=H.shape), up + rng.normal(size=up.shape)) for _ in range(3)]
    consts = [l_en(*o) - l_sup(*o) for o in outs]
    assert max(consts) - min(consts) <= 1e-8 * max(abs(c) for c in consts)


def test_chart_pullback_invariance():
    """The inputs (log J - mean, mu) and the tensor are invariant to translating,
    rotating and uniformly scaling the piece."""
    n = 16
    X = ch.transfinite(_curved_piece(), n)
    c0 = ch.Chart(X, "transfinite")
    th, lam, shift = 0.7, 3.5, np.array([2.0, -1.0])
    Rm = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    c1 = ch.Chart(lam * X @ Rm.T + shift, "transfinite")
    n0, n1 = c0.nodal_fields(), c1.nodal_fields()
    l0, l1 = np.log(n0["J"]), np.log(n1["J"])
    assert np.allclose(l0 - l0.mean(), l1 - l1.mean(), atol=1e-12)
    assert np.allclose(n0["mu"], n1["mu"], atol=1e-12)
    e0, e1 = c0.element_fields(), c1.element_fields()
    for k in ("a11", "a12", "a22", "Jt"):
        assert np.allclose(e0[k], e1[k], atol=1e-12)


def test_tensor_from_mu_is_the_metric():
    n = 16
    X = ch.transfinite(_curved_piece(), n)
    De = ch.element_jacobian(X)
    a11, a12, a22, J = ch.metric_tensor(De)
    _J, mu = ch.beltrami(De)
    b11, b12, b22 = ch.tensor_from_mu(mu)
    assert np.allclose([a11, a12, a22], [b11, b12, b22], atol=1e-12)
    assert np.allclose(a11 * a22 - a12 ** 2, 1.0, atol=1e-12)        # det J G^-1 = 1


def test_certificate_accepts_and_refuses():
    n = 16
    X = ch.winslow(_curved_piece(), n)[0]
    assert cert.certify(X).ok
    # a straight-sided square (collinear boundary segments) is certified
    sq = ch.from_map(lambda x, y: (x, y), n)
    assert cert.certify(sq).ok
    # a fold: one interior node pushed across its neighbour
    Y = sq.copy()
    Y[5, 5] = Y[5, 7] + (0.02, 0.0)
    c = cert.certify(Y)
    assert not c.ok and c.nonpositive_corners > 0
    # reflected: negative orientation
    Z = sq.copy()
    Z[..., 0] = -Z[..., 0]
    assert not cert.certify(Z).ok
    # a self-intersecting boundary with positive cell Jacobians cannot exist, but the
    # polygon test itself must see a figure eight
    eight = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], float)
    assert not cert.polygon_simple(eight)
    assert cert.polygon_simple(np.array([[0, 0], [1, 0], [2, 0], [2, 1], [0, 1]], float))


def test_winslow_converges_on_a_curved_piece():
    X, info = ch.winslow(_curved_piece(), 24)
    assert info["iterations"] < 200
    assert cert.certify(X).ok


def test_envelope_refusal_names_the_quantity():
    n = 16
    c = ch.Chart(ch.from_map(lambda x, y: (4.0 * x, y), n), "affine")     # modulus 4
    m = cert.measure(c, np.zeros((n + 1, n + 1)), 1e-4)
    env = cert.Envelope(k_mu=0.5, l_J=2.0, s_grad=40.0, l_kappa=2.3, fo_min=1e-2, fo_max=1e2)
    r = cert.refusals(m, env)
    assert any("sup|mu|" in x for x in r) and any("Fo = " in x for x in r)


def _two_pieces(n_left, n_right, fo=None):
    """Two unit squares side by side, the left hot (400) on x = 0, the right cold (300)
    on x = 2, a smooth source, so the solution is not in any port span."""
    def piece(x0, n):
        c = ch.Chart(ch.from_map(lambda x, y: (x0 + x, y + 0.15 * np.sin(math.pi * x) * y * (1 - y)), n), "map")
        t = np.arange(n + 1) / n
        Xg = c.X
        g = 50.0 * np.sin(math.pi * Xg[..., 1]) * np.cos(1.3 * Xg[..., 0])
        return PieceProblem(c, fo=fo, g=g, kappa_ref=2.0)
    probs = [piece(0.0, n_left), piece(1.0, n_right)]

    def bc(i, s, P):
        if np.all(np.abs(P[:, 0]) < 1e-9):
            return ("dirichlet", np.full(len(P), 400.0))
        if np.all(np.abs(P[:, 0] - 2.0) < 1e-9):
            return ("dirichlet", np.full(len(P), 300.0))
        return ("insulated", None)
    return probs, bc


@pytest.mark.parametrize("fo", [None, 0.05])
def test_host_with_all_exact_modes_is_the_monolithic_solve(fo):
    n = 16
    probs, bc = _two_pieces(n, n, fo)
    L = host.Layout(probs, bc, "sine-hat", m=n)
    assert all(e.conforming for e in L.edges)
    mono = host.monolithic(L)
    cp = L.solve(host.ClassicalExpert())
    num, den = cp.energy_error(mono.fields)
    assert num <= 1e-10 * den
    assert cp.calls == 2                                   # SE2: one response per piece


def test_se1_lambda_min_rises_and_gram_identity_on_assembly():
    """SE1 on an assembled layout: perturbed modes give S~ - S >= 0 to rounding, so
    lambda_min(S~) >= lambda_min(S)."""
    rng = np.random.default_rng(3)
    n = 16
    probs, bc = _two_pieces(n, n, None)
    L = host.Layout(probs, bc, "sine-hat", m=8)
    exact = host.ClassicalExpert().respond(probs, L.pb)
    cp0 = L.couple(exact)
    pert = [host.Response(r.H + 0.3 * rng.normal(size=r.H.shape), r.up) for r in exact]
    cp1 = L.couple(pert)
    D = cp1.S - cp0.S
    assert np.linalg.eigvalsh(0.5 * (D + D.T))[0] >= -1e-10 * np.abs(cp0.S).max()
    assert cp1.min_eig() >= cp0.min_eig() * (1 - 1e-10)


def test_mortar_couples_non_matching_resolutions():
    """A 16-element piece against a 24-element one: the seam is a mortar (L2 in arc
    length), the solve is well posed and close to the conforming 16-16 answer."""
    probs, bc = _two_pieces(16, 24, None)
    L = host.Layout(probs, bc, "sine-hat", m=8)
    assert not L.edges[L.edge_of[(0, 1)]].conforming
    cp = L.solve(host.ClassicalExpert())
    probs2, _ = _two_pieces(16, 16, None)
    ref = host.Layout(probs2, bc, "sine-hat", m=16)
    mono = host.monolithic(ref)
    # compare along the left piece, which both runs share
    e = float(np.max(np.abs(cp.fields[0] - mono.fields[0])))
    assert e < 0.05                                        # kelvin, on a 100 K span (measured 0.008)
    assert cp.min_eig() > 0


def test_a_vertex_fixed_twice_is_refused():
    n = 8
    probs, _bc = _two_pieces(n, n, None)

    def bad(i, s, P):
        return ("dirichlet", np.full(len(P), 400.0 if s == 0 else 300.0))
    with pytest.raises(host.Refused):
        host.Layout(probs, bad, "sine-hat", m=4)
