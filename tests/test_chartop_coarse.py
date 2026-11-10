"""The coarse level of the reworked Stage 0 expert (`atlas/chartop/coarse.py`): Proposition A
and Lemma B of [[chart-operator-stage-0-rework]] section 3.2, numerically.  Classical solves
and U's untrained form only: no trained weights, no training.

The pieces are generator pieces from seeds no registration draws (9,000,100 onward), at
n = 16 and 32, read from their stored float32 arrays as `data.labels` reads them; Phase 2
of the rework measures the post-hoc validation seeds, so the tests stay off them.

- the prolongation is the bilinear coarse hats, in `fe.grid`'s interior order, zero on the
  loop;
- the residual from the full field is ``B Q - A_I X``; `apply_M` is ``fe.assemble(...) @ U``;
  the restricted residual (`CoarseSpace.apply`) is the fine one restricted;
- Proposition A: Galerkin orthogonality (CX), never worse column by column and equal to a
  brute-force least-squares fit over the coarse space, Pythagoras, idempotence, blindness
  to coarse shifts, exactness at ``n_c = n``, traces untouched, the gradient ``Q_c^T``;
- Lemma B: the kernel identity survives the correction to rounding with ``b_1`` from the
  operator's row sums, and misses by about 1e-8 with ``mass(J~) 1`` on a short step,
  because the stored ``sigma`` and ``J~`` are rounded to float32 separately;
- numpy against torch to 1e-12, and the correction commuting with the square's eight
  symmetries;
- `CoarseCorrected`, the host's adapter: ``n_c=None`` is the inner expert bit for bit,
  ``n_c = n`` the classical solve, its outputs coarse-exact and kernel-exact."""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from atlas.chartop import coarse as co
from atlas.chartop import data, fe, forms, host, model, ports, principal
from atlas.chartop.expert import LearnedExpert

torch.set_num_threads(1)

KIND, M_PORTS = "sine-hat", 16
#: seeds no registration draws; the same piece type at n = 16 and 32
STEP, SHORT_STEP, STEADY = 9_000_100, 9_000_112, 9_000_102      # Fo 0.63, Fo 0.0117, steady


class Piece:
    """A drawn piece's stored arrays (float32, read back in float64, as `data.labels`
    reads them), its operator, exact modes and an input far from exact."""

    def __init__(self, seed: int, n: int):
        self.seed, self.n = seed, n
        self.example = data.draw(seed, n=n)
        self.a = {k: np.asarray(v, np.float64) for k, v in data.arrays(self.example).items()}
        self.s0 = float(self.a["s0"])
        self.M = fe.assemble(n, self.a["a11"], self.a["a12"], self.a["a22"], self.a["sigma"])
        self.exact = fe.Exact(self.M, n)
        self.A_I = self.exact.A_I
        self.Q = ports.basis(KIND, n, M_PORTS).Q
        self.H = self.exact.modes(self.Q)
        self.BQ = -np.asarray(self.exact.M_IG @ self.Q)
        self.c1 = forms.kernel_coefficients(KIND, n, M_PORTS)
        rng = np.random.default_rng(seed)
        H0 = principal.modes(n, self.Q, self.s0)
        # the plain square's modes, perturbed: an input far from exact
        self.X = H0 + 0.05 * np.abs(H0).max() * rng.standard_normal(H0.shape)
        _loop, inner = fe.grid(n)
        self.b = (fe.mass(n, self.a["w"]) @ self.a["g"].ravel())[inner]

    def space(self, n_c: int) -> co.CoarseSpace:
        return co.CoarseSpace.of_matrix(self.M, self.n, n_c)

    def torch_arrays(self):
        return [torch.tensor(self.a[k][None]) for k in ("a11", "a12", "a22", "sigma")]

    def norms_A(self, E):
        return np.sqrt(np.maximum(np.einsum("ik,ik->k", E, np.asarray(self.A_I @ E)), 0.0))


_PIECES: dict = {}


def piece(seed: int, n: int = 16) -> Piece:
    if (seed, n) not in _PIECES:
        _PIECES[(seed, n)] = Piece(seed, n)
    return _PIECES[(seed, n)]


def _rel(a, b) -> float:
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max())


def test_the_pieces_are_the_types_named():
    assert 0.5 < piece(STEP).example.fo < 1.0 and piece(STEP, 32).example.fo == piece(STEP).example.fo
    assert piece(SHORT_STEP).s0 > 50.0 and piece(SHORT_STEP, 32).s0 > 50.0
    assert piece(STEADY).s0 == 0.0 and piece(STEADY, 32).s0 == 0.0


# ---------------------------------------------------------------------------
# the prolongation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n, n_c", [(16, 2), (16, 4), (16, 8), (32, 8), (32, 32)])
def test_the_prolongation_is_the_coarse_hats_in_fe_grids_order(n, n_c):
    """Column ``(J-1)(n_c-1) + (I-1)`` of ``P`` is the bilinear hat of coarse node (I, J),
    sampled at `fe.grid`'s interior nodes in their own order; sampled on the loop the same
    hats are exactly zero, so ``P`` maps into the interior (traces untouched)."""
    P = co.prolongation(n, n_c).toarray()
    loop, inner = fe.grid(n)
    assert P.shape == ((n - 1) ** 2, (n_c - 1) ** 2)
    xi_all = (np.arange((n + 1) ** 2) % (n + 1)) / n
    eta_all = (np.arange((n + 1) ** 2) // (n + 1)) / n
    for J in range(1, n_c):
        for I in range(1, n_c):
            hat = (np.maximum(0.0, 1.0 - np.abs(xi_all * n_c - I))
                   * np.maximum(0.0, 1.0 - np.abs(eta_all * n_c - J)))
            col = P[:, (J - 1) * (n_c - 1) + (I - 1)]
            assert np.array_equal(col, hat[inner]), (I, J)
            assert not np.any(hat[loop])
    r = n // n_c
    assert np.array_equal(P * r * r, np.round(P * r * r))      # every entry exact
    assert np.array_equal(co.prolongation_1d(n, n_c), co.prolongation_torch(n, n_c).numpy())
    if n_c == n:
        assert np.array_equal(P, np.eye((n - 1) ** 2))


@pytest.mark.parametrize("n, n_c", [(16, 3), (16, 32), (24, 8), (16, 1)])
def test_the_coarse_sizes_are_checked(n, n_c):
    with pytest.raises(ValueError):
        co.prolongation_1d(n, n_c)


# ---------------------------------------------------------------------------
# the residual, the element-stencil product, the restricted residual
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed, n", [(STEP, 16), (STEADY, 16), (STEP, 32)])
def test_the_residual_from_the_full_field_is_BQ_minus_A_I_X(seed, n):
    p = piece(seed, n)
    assert _rel(co.residual(p.M, n, p.X, p.Q), p.BQ - p.A_I @ p.X) < 1e-13
    rp = co.residual(p.M, n, p.X[:, 0], rhs=p.b)          # a field: nothing on the loop
    assert _rel(rp, p.b - p.A_I @ p.X[:, 0]) < 1e-13


@pytest.mark.parametrize("seed, n", [(STEP, 16), (STEADY, 16), (STEP, 32)])
def test_apply_M_is_the_assembled_matrix(seed, n):
    p = piece(seed, n)
    U = torch.randn(1, 3, n + 1, n + 1, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))
    MU = co.apply_M(*p.torch_arrays(), U)[0].reshape(3, -1).numpy()
    ref = np.stack([p.M @ U[0, k].numpy().ravel() for k in range(3)])
    assert _rel(MU, ref) < 1e-12
    # half the gradient of model.energy: U . M U is the energy
    E = model.energy(*p.torch_arrays(), U)[0].numpy()
    assert _rel((U[0].reshape(3, -1).numpy() * MU).sum(1), E) < 1e-12
    noreact = co.apply_M(*p.torch_arrays()[:3], None, U)[0].reshape(3, -1).numpy()
    K = fe.assemble(n, p.a["a11"], p.a["a12"], p.a["a22"])
    assert _rel(noreact, np.stack([K @ U[0, k].numpy().ravel() for k in range(3)])) < 1e-12


@pytest.mark.parametrize("seed", [STEP, STEADY])
@pytest.mark.parametrize("n_c", [4, 8, 16])
def test_the_restricted_residual_is_the_fine_one_restricted(seed, n_c):
    """`CoarseSpace.apply` (``P^T rhs - G u_full``) gives `correct` with the fine residual,
    for the modes and for a field, including the sparse top level ``n_c = n``."""
    p = piece(seed)
    ref = co.CoarseSpace(p.A_I, p.n, n_c)
    cs = p.space(n_c)
    assert cs.top == (n_c == p.n) and ref.top == cs.top
    fine = ref.correct(p.X, co.residual(p.M, p.n, p.X, p.Q))
    assert _rel(cs.apply(p.X, p.Q), fine) < 1e-12
    finep = ref.correct(p.X[:, 1], co.residual(p.M, p.n, p.X[:, 1], rhs=p.b))
    assert _rel(cs.apply(p.X[:, 1], rhs=p.b), finep) < 1e-12
    with pytest.raises(ValueError):
        ref.apply(p.X, p.Q)                                 # no matrix kept: no restricted residual


# ---------------------------------------------------------------------------
# Proposition A
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [STEP, SHORT_STEP, STEADY])
@pytest.mark.parametrize("n_c", [2, 4, 8])
def test_galerkin_orthogonality_cx(seed, n_c):
    """Item 1: ``P^T (B Q - A_I C(X)) = 0`` to 1e-12 of ``P^T B Q``; likewise a field."""
    p = piece(seed)
    cs = p.space(n_c)
    CX = cs.apply(p.X, p.Q)
    cx = co.coarse_exactness(cs.P, co.residual(p.M, p.n, CX, p.Q), p.BQ)
    assert cx["rel"] is not None and cx["rel"] < 1e-12
    before = co.coarse_exactness(cs.P, co.residual(p.M, p.n, p.X, p.Q), p.BQ)
    assert before["rel"] > 1e-3                         # the input was not coarse-exact
    up = cs.apply(p.X[:, 0], rhs=p.b)
    assert co.coarse_exactness(cs.P, co.residual(p.M, p.n, up, rhs=p.b), p.b)["rel"] < 1e-12


def test_coarse_exactness_has_no_ratio_without_data():
    p = piece(STEADY)
    cs = p.space(4)
    zero = np.zeros((p.n - 1) ** 2)
    out = cs.apply(p.X[:, 0], rhs=zero)
    cx = co.coarse_exactness(cs.P, co.residual(p.M, p.n, out, rhs=zero), zero)
    assert cx["rel"] is None and cx["den"] == 0.0
    assert cx["num"] < 1e-12 * float(np.linalg.norm(cs.P.T @ (p.A_I @ p.X[:, 0])))


@pytest.mark.parametrize("seed", [STEP, SHORT_STEP, STEADY])
@pytest.mark.parametrize("n_c", [2, 4, 8])
def test_never_worse_and_the_least_squares_optimum(seed, n_c):
    """Item 2, column by column: ``||H - C(X)||_A <= ||H - X||_A``, equal to the minimum over
    the coarse space found by brute force (a dense Cholesky of ``A_I`` and least squares),
    and Pythagoras ``||C(X) - H||^2 + ||C(X) - X||^2 = ||X - H||^2``."""
    p = piece(seed)
    cs = p.space(n_c)
    CX = cs.apply(p.X, p.Q)
    E, EC = p.X - p.H, CX - p.H
    nE, nEC = p.norms_A(E), p.norms_A(EC)
    assert np.all(nEC <= nE * (1.0 + 1e-12))
    L = np.linalg.cholesky(p.A_I.toarray())
    LP = L.T @ cs.P.toarray()
    best = np.array([np.linalg.norm(L.T @ E[:, k] + LP @ np.linalg.lstsq(LP, -(L.T @ E[:, k]),
                                                                          rcond=None)[0])
                     for k in range(E.shape[1])])
    assert np.max(np.abs(nEC - best) / nE) < 1e-10
    nD = p.norms_A(CX - p.X)
    assert np.max(np.abs(nEC ** 2 + nD ** 2 - nE ** 2) / nE ** 2) < 1e-10


@pytest.mark.parametrize("seed", [STEP, STEADY])
def test_idempotent_and_blind_to_coarse_shifts(seed):
    """Item 3: ``C(C(X)) = C(X)`` and ``C(X + P v) = C(X)`` for random coarse ``v``."""
    p = piece(seed)
    cs = p.space(4)
    CX = cs.apply(p.X, p.Q)
    assert _rel(cs.apply(CX, p.Q), CX) < 1e-13
    v = np.random.default_rng(1).standard_normal((cs.size, p.X.shape[1]))
    assert _rel(cs.apply(p.X + cs.P @ v, p.Q), CX) < 1e-12


@pytest.mark.parametrize("seed", [STEP, SHORT_STEP, STEADY])
def test_exact_at_the_top_and_traces_untouched(seed):
    """Item 4, the positive control: at ``n_c = n`` the output is the exact solve whatever
    the input, for the modes, the particular field and (steps) the level field.  Item 5:
    the corrected fields' traces are ``Q`` bit for bit."""
    p = piece(seed)
    cs = p.space(p.n)
    assert _rel(cs.apply(p.X, p.Q), p.H) < 1e-10
    assert _rel(cs.apply(np.zeros((p.n - 1) ** 2), rhs=p.b), p.exact.particular(p.b)) < 1e-10
    if p.s0 > 0:
        b1 = co.level_rhs(p.M, p.n, p.s0)
        assert _rel(cs.apply(p.X[:, 0], rhs=b1), p.exact.particular(b1)) < 1e-10
    loop, _inner = fe.grid(p.n)
    V = fe.full_field(p.n, p.space(4).apply(p.X, p.Q), p.Q)
    assert np.array_equal(V[loop], p.Q)


def test_the_gradient_through_C_is_Q_c_transposed():
    """Item 6: for ``L = <G, C(X)>``, ``dL/dX = Q_c^T G = G - A_I P A_c^-1 P^T G``, through the
    torch correction with the residual formed from the full field in the graph."""
    p = piece(STEP)
    n, n_c = p.n, 4
    cs = p.space(n_c)
    ta = p.torch_arrays()
    tcs = co.TorchCoarseSpace(co.coarse_matrix_torch(*ta, n_c), n, n_c)
    ps = model.plain_square(n, KIND, M_PORTS, "cpu")
    K = p.X.shape[1]
    X = torch.tensor(p.X.T.reshape(1, K, n - 1, n - 1), requires_grad=True)
    G = torch.randn(1, K, n - 1, n - 1, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    (tcs.correct(X, co.residual_torch(*ta, X, ps.QB)) * G).sum().backward()
    Gn = G[0].reshape(K, -1).T.numpy()
    want = Gn - p.A_I @ (cs.P @ cs.solve(cs.P.T @ Gn))
    assert _rel(X.grad[0].reshape(K, -1).T.numpy(), want) < 1e-12


# ---------------------------------------------------------------------------
# Lemma B
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [STEP, SHORT_STEP])
@pytest.mark.parametrize("n_c", [4, 8])
def test_lemma_b_the_kernel_identity_survives_with_the_row_sums(seed, n_c):
    """Inputs with ``X c1 + s0 X_1 = 1`` give outputs with ``C(X) c1 + s0 C_1(X_1) = 1`` to
    rounding when ``b_1 = (M 1)_I / s0``."""
    p = piece(seed)
    X1 = (1.0 - p.X @ p.c1) / p.s0
    assert np.abs(p.X @ p.c1 + p.s0 * X1 - 1.0).max() < 1e-14
    cs = p.space(n_c)
    CX = cs.apply(p.X, p.Q)
    C1 = cs.apply(X1, rhs=co.level_rhs(p.M, p.n, p.s0))
    assert np.abs(CX @ p.c1 + p.s0 * C1 - 1.0).max() < 1e-13
    assert _rel(C1, X1) > 1e-6                         # the inputs moved


def test_lemma_b_misses_with_mass_of_J_because_the_storage_is_float32():
    """The documented reason for ``b_1``'s definition.  On a short step from the stored
    arrays, ``b_1 = mass(J~) 1`` misses the identity by about 1e-8, since ``sigma`` and
    ``J~`` are rounded to float32 separately; on the same piece's float64 problem (where
    ``sigma = J~ / Fo`` in one rounding) both definitions hold to rounding."""
    p = piece(SHORT_STEP)
    _loop, inner = fe.grid(p.n)
    X1 = (1.0 - p.X @ p.c1) / p.s0
    cs = p.space(4)
    CX = cs.apply(p.X, p.Q)
    rows = cs.apply(X1, rhs=co.level_rhs(p.M, p.n, p.s0))
    massJ = cs.apply(X1, rhs=(fe.mass(p.n, p.a["w"]) @ np.ones((p.n + 1) ** 2))[inner])
    miss_rows = np.abs(CX @ p.c1 + p.s0 * rows - 1.0).max()
    miss_mass = np.abs(CX @ p.c1 + p.s0 * massJ - 1.0).max()
    assert miss_rows < 1e-13
    assert 1e-9 < miss_mass < 1e-6 and miss_mass > 1e4 * miss_rows
    prob = p.example.problem                           # float64: sigma = J~ / Fo in one rounding
    t = prob.tensors()
    M64 = prob.matrix()
    s0 = prob.s0
    X1b = (1.0 - p.X @ p.c1) / s0
    cs64 = co.CoarseSpace.of_matrix(M64, p.n, 4)
    CX64 = cs64.apply(p.X, p.Q)
    for b1 in (co.level_rhs(M64, p.n, s0), (fe.mass(p.n, t["w"]) @ np.ones((p.n + 1) ** 2))[inner]):
        assert np.abs(CX64 @ p.c1 + s0 * cs64.apply(X1b, rhs=b1) - 1.0).max() < 1e-13


def test_lemma_b_steady_the_hats_sum_survives():
    """Steady (``s0 = 0``): ``X c1 = 1`` gives ``C(X) c1 = 1`` to rounding, and there is no
    level field (`level_rhs` refuses)."""
    p = piece(STEADY)
    X = p.X.copy()
    h = np.flatnonzero(p.c1)
    X[:, h[0]] += 1.0 - X @ p.c1                       # the input's hats now sum to one
    assert np.abs(X @ p.c1 - 1.0).max() < 1e-14
    assert np.abs(p.space(4).apply(X, p.Q) @ p.c1 - 1.0).max() < 1e-13
    with pytest.raises(ValueError):
        co.level_rhs(p.M, p.n, 0.0)


# ---------------------------------------------------------------------------
# numpy against torch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n, n_c", [(16, 4), (16, 8), (32, 8), (32, 16)])
def test_numpy_equals_torch(n, n_c):
    """A batch of two pieces (a step and a steady one): the coarse operator, the residual,
    the correction of the modes and of a field, and the level field, numpy against torch
    to 1e-12."""
    ps_ = [piece(STEP, n), piece(STEADY, n)]
    ta = [torch.tensor(np.stack([p.a[k] for p in ps_])) for k in ("a11", "a12", "a22", "sigma")]
    Ac = co.coarse_matrix_torch(*ta, n_c)
    tcs = co.TorchCoarseSpace(Ac, n, n_c)
    sq = model.plain_square(n, KIND, M_PORTS, "cpu")
    K = 4 * M_PORTS
    X = torch.tensor(np.stack([p.X.T.reshape(K, n - 1, n - 1) for p in ps_]))
    r = co.residual_torch(*ta, X, sq.QB)
    CX = tcs.correct(X, r)
    for b, p in enumerate(ps_):
        cs = p.space(n_c)
        assert _rel(Ac[b].numpy(), cs.A_c) < 1e-12
        rn = co.residual(p.M, n, p.X, p.Q)
        assert _rel(r[b].reshape(K, -1).T.numpy(), rn) < 1e-12
        assert _rel(CX[b].reshape(K, -1).T.numpy(), cs.apply(p.X, p.Q)) < 1e-12
    xf = torch.tensor(np.stack([p.X[:, 5].reshape(n - 1, n - 1) for p in ps_]))
    bf = torch.tensor(np.stack([p.b.reshape(n - 1, n - 1) for p in ps_]))
    upt = tcs.correct(xf, co.residual_torch(*ta, xf[:, None], rhs=bf[:, None])[:, 0])
    for b, p in enumerate(ps_):
        assert _rel(upt[b].numpy().ravel(), p.space(n_c).apply(p.X[:, 5], rhs=p.b)) < 1e-12
    # the level field: torch's M 1 has an exactly zero stiffness part, numpy's carries the
    # stiffness rows' rounding, so b_1 agrees to that rounding over s0, and the corrected
    # level fields agree to 1e-12
    s0 = torch.tensor([p.s0 for p in ps_])
    b1t, transient = co.level_rhs_torch(*ta, s0)
    assert transient.tolist() == [True, False] and not torch.any(b1t[1])
    p = ps_[0]
    b1n = co.level_rhs(p.M, n, p.s0)
    stiff = float(max(np.abs(p.a["a11"]).max(), np.abs(p.a["a22"]).max()))
    assert np.abs(p.s0 * (b1t[0].numpy().ravel() - b1n)).max() < 1e-14 * stiff
    X1 = (1.0 - p.X @ p.c1) / p.s0
    x1t = torch.tensor(X1.reshape(1, n - 1, n - 1))
    one = co.TorchCoarseSpace(Ac[:1], n, n_c)
    u1t = one.correct(x1t, co.residual_torch(*[a[:1] for a in ta], x1t[:, None], rhs=b1t[:1, None])[:, 0])
    assert _rel(u1t[0].numpy().ravel(), p.space(n_c).apply(X1, rhs=b1n)) < 1e-12


# ---------------------------------------------------------------------------
# the square's eight symmetries
# ---------------------------------------------------------------------------


def _spatial(a, k):
    """Symmetry k = 4 t + 2 fy + fx on images [..., j, i], as `model._spatial`."""
    fx, fy, t = k & 1, (k >> 1) & 1, (k >> 2) & 1
    if fx:
        a = a[..., :, ::-1]
    if fy:
        a = a[..., ::-1, :]
    if t:
        a = np.swapaxes(a, -1, -2)
    return np.ascontiguousarray(a)


def _arrays(a, k):
    """The transformed problem's element arrays, by `model.d4`'s rules."""
    fx, fy, t = k & 1, (k >> 1) & 1, (k >> 2) & 1
    a11, a12, a22, sig = (_spatial(a[key], k) for key in ("a11", "a12", "a22", "sigma"))
    if fx ^ fy:
        a12 = -a12
    if t:
        a11, a22 = a22, a11
    return a11, a12, a22, sig


@pytest.mark.parametrize("k", range(8))
def test_the_correction_commutes_with_the_squares_symmetries(k):
    """``C'(S X, S F) = S C(X, F)`` for the transformed piece, generic fields ``X`` and
    right-hand sides ``F``: nested uniform grids are symmetric, so ``P`` commutes with every
    symmetry of the square.  numpy, and torch through `model.d4`'s own batch transform."""
    p = piece(STEP)
    n, n_c = p.n, 4
    rng = np.random.default_rng(10 + k)
    X = rng.standard_normal(((n - 1) ** 2, 3))
    F = rng.standard_normal(((n - 1) ** 2, 3))

    def img(V):                                     # interior columns -> [K, j, i]
        return V.T.reshape(-1, n - 1, n - 1)

    def col(I):
        return I.reshape(I.shape[0], -1).T

    out = co.CoarseSpace(p.A_I, n, n_c).correct(X, F - p.A_I @ X)
    want = col(_spatial(img(out), k))
    _loop, inner = fe.grid(n)
    Ak = fe.assemble(n, *_arrays(p.a, k)).tocsr()[inner][:, inner]
    Xk, Fk = col(_spatial(img(X), k)), col(_spatial(img(F), k))
    assert _rel(co.CoarseSpace(Ak, n, n_c).correct(Xk, Fk - Ak @ Xk), want) < 1e-12
    batch = {key: torch.tensor(p.a[key][None]) for key in ("a11", "a12", "a22", "sigma", "w", "g")}
    batch["inputs"] = torch.tensor(np.asarray(p.a["inputs"], np.float32)[None])
    bk = model.d4(batch, k)
    arr = (bk["a11"], bk["a12"], bk["a22"], bk["sigma"])
    tk = co.TorchCoarseSpace(co.coarse_matrix_torch(*arr, n_c), n, n_c)
    Xt = torch.tensor(_spatial(img(X), k)[None])
    got = tk.correct(Xt, co.residual_torch(*arr, Xt, rhs=torch.tensor(_spatial(img(F), k)[None])))
    assert _rel(col(got[0].numpy()), want) < 1e-12


# ---------------------------------------------------------------------------
# the host's adapter
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def u_expert():
    """U: S's form (split, corner bubble, strips) with zero heads, as the diagnostics
    build it, behind the host's adapter."""
    torch.manual_seed(0)
    net = forms.KernelExpert("split", "unet", "1M", KIND, M_PORTS, None, "corner", True).eval()
    return LearnedExpert(net, envelope=data.ENVELOPE)


def _problems():
    return [piece(s).example.problem for s in (STEP, SHORT_STEP, STEADY)]


def test_the_adapter_without_a_coarse_size_is_the_inner_expert_bit_for_bit(u_expert):
    probs = _problems()
    pb = lambda n: ports.basis(KIND, n, M_PORTS)            # noqa: E731
    inner = u_expert.respond(probs, pb)
    ad = co.CoarseCorrected(u_expert, None)
    got = ad.respond(probs, pb)
    assert ad.calls == len(probs)
    for r, q in zip(got, inner):
        assert np.array_equal(r.H, q.H) and np.array_equal(r.up, q.up) and np.array_equal(r.level, q.level)
    with pytest.raises(NotImplementedError):
        ad.respond(probs, pb, modes_from=got)


def test_the_adapter_at_the_top_is_the_classical_solve(u_expert):
    probs = _problems()
    pb = lambda n: ports.basis(KIND, n, M_PORTS)            # noqa: E731
    got = co.CoarseCorrected(u_expert, 16).respond(probs, pb)
    ref = host.ClassicalExpert().respond(probs, pb)
    for p, r, q in zip(probs, got, ref):
        assert _rel(r.H, q.H) < 1e-10
        if np.any(q.up):
            assert _rel(r.up, q.up) < 1e-10


@pytest.mark.parametrize("n_c", [4, 8])
def test_the_adapters_answers_are_coarse_exact_and_kernel_exact(u_expert, n_c):
    """TL0 on float64 pieces: CX of the modes and the particular field at most 1e-12; the
    level-field identity ``H c1 + s0 u_1 = 1`` and the uniform state through the host's
    measure (`forms.kernel_residual`) to 1e-12; a steady piece's level field passed on."""
    probs = _problems()
    pb = lambda n: ports.basis(KIND, n, M_PORTS)            # noqa: E731
    ad = co.CoarseCorrected(u_expert, n_c)
    got = ad.respond(probs, pb)
    inner = u_expert.respond(probs, pb)
    assert ad.coarse_seconds > 0.0
    for p, r, q in zip(probs, got, inner):
        n = p.n
        M = p.matrix()
        Q = pb(n).Q
        P = co.prolongation(n, n_c)
        _loop, I = fe.grid(n)
        BQ = -np.asarray(M.tocsr()[I][:, _loop] @ Q)
        assert co.coarse_exactness(P, co.residual(M, n, r.H, Q), BQ)["rel"] < 1e-12
        b = p.load()[I]
        cxp = co.coarse_exactness(P, co.residual(M, n, r.up, rhs=b), b)
        assert cxp["rel"] < 1e-12
        c1 = forms.kernel_coefficients(KIND, n, M_PORTS)
        if p.s0 > 0:
            assert np.abs(r.H @ c1 + p.s0 * r.level - 1.0).max() < 1e-12
            assert not np.array_equal(r.level, q.level)
        else:
            assert np.abs(r.H @ c1 - 1.0).max() < 1e-12
            assert np.array_equal(r.level, q.level)
        rk = forms.kernel_residual(ad, p, pb(n))
        assert np.abs(rk).max() < 1e-12
