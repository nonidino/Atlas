"""Branch 2c's enlarged coarse level in torch (`atlas/chartop/coarse_patches.py`): candidate
B's corner patches for training, against the numpy reference (`coarse.CoarseSpace` with
``extra=``).  Classical solves only.

- the extra nodes are `coarse.corner_patch_columns`' hats, in its order, for any flags;
- ``A_e`` assembled by probing equals ``P_e^T A_I P_e`` to 1e-12, padding decoupled;
- the correction equals numpy's for the modes and a field, to 1e-12, in a batch whose
  examples have different corners; a batch with no flagged corner is
  `coarse.TorchCoarseSpace` bit for bit;
- Galerkin orthogonality on each example's own ``V_e``, Lemma B, the gradient ``Q_e^T``,
  and a repeated call equal bit for bit.

The pieces are generator pieces from seeds no registration draws (9,000,100 onward)."""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from atlas.chartop import coarse as co
from atlas.chartop import coarse_patches as cp
from atlas.chartop import data, fe, forms, model, ports, principal

torch.set_num_threads(1)

KIND, M_PORTS = "sine-hat", 16
SEEDS = (9_000_100, 9_000_112, 9_000_102)          # a step, a short step, a steady piece
FLAGS = np.array([[1, 1, 1, 1], [0, 0, 0, 0], [1, 0, 1, 0]], dtype=bool)


class Batch:
    """Three drawn pieces' stored arrays (float32 read in float64), operators, exact modes
    and inputs far from exact, with `FLAGS` as their corners (any flags give a valid space)."""

    def __init__(self, n: int):
        self.n = n
        self.a = [{k: np.asarray(v, np.float64) for k, v in data.arrays(data.draw(s, n=n)).items()} for s in SEEDS]
        self.M = [fe.assemble(n, a["a11"], a["a12"], a["a22"], a["sigma"]) for a in self.a]
        self.Q = ports.basis(KIND, n, M_PORTS).Q
        self.c1 = forms.kernel_coefficients(KIND, n, M_PORTS)
        rng = np.random.default_rng(n)
        self.X = []
        for a in self.a:
            H0 = principal.modes(n, self.Q, float(a["s0"]))
            self.X.append(H0 + 0.05 * np.abs(H0).max() * rng.standard_normal(H0.shape))
        _loop, inner = fe.grid(n)
        self.b = [(fe.mass(n, a["w"]) @ a["g"].ravel())[inner] for a in self.a]

    def torch_arrays(self, which=None):
        which = range(len(SEEDS)) if which is None else which
        return [torch.tensor(np.stack([self.a[i][k] for i in which])) for k in ("a11", "a12", "a22", "sigma")]

    def S(self, i: int, n_c: int):
        corners = tuple(c for c in range(4) if FLAGS[i][c])
        return co.corner_patch_columns(self.n, n_c, corners, cp.FRAC)

    def numpy_space(self, i: int, n_c: int):
        return co.CoarseSpace.of_matrix(self.M[i], self.n, n_c, extra=self.S(i, n_c))


_B: dict = {}


def batch(n: int = 32) -> Batch:
    if n not in _B:
        _B[n] = Batch(n)
    return _B[n]


def _rel(a, b) -> float:
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max())


def _img(V, n):                  # interior columns ((n-1)^2, K) -> [K, n-1, n-1]
    return V.T.reshape(-1, n - 1, n - 1)


def test_the_extra_nodes_are_the_patch_columns_in_order():
    n, n_c = 32, 8
    nodes = cp.patch_nodes(n, n_c, FLAGS)
    for i, x in enumerate(nodes):
        S = batch(n).S(i, n_c).tocsc()
        assert np.array_equal(x, S.indices) and len(x) == S.shape[1]
    assert [len(x) for x in nodes] == [4 * 15, 0, 2 * 15]                  # w = 4, one coarse node each
    assert cp.patch_nodes(n, n, FLAGS)[0].size == 0                         # nothing at the top
    ex = data.draw(SEEDS[0], n=n)
    assert np.array_equal(cp.corner_flags(ex.problem.chart.X), [True, True, True, True])
    assert cp.FRAC == 8


@pytest.mark.parametrize("n, n_c", [(16, 4), (32, 8), (32, 16)])
def test_the_probed_matrix_is_the_galerkin_operator(n, n_c):
    """``A_e`` for the batch against ``P_e^T A_I P_e`` per example: each example's own block
    to 1e-12, its padding decoupled (a one on the diagonal, no coupling)."""
    bt = batch(n)
    nodes = cp.patch_nodes(n, n_c, FLAGS)
    Ae, idx = cp.enlarged_matrix_torch(*bt.torch_arrays(), n_c, nodes)
    m2, s = (n_c - 1) ** 2, max(len(x) for x in nodes)
    assert Ae.shape == (3, m2 + s, m2 + s) and idx.shape == (3, s)
    for i in range(3):
        k = m2 + len(nodes[i])
        ref = bt.numpy_space(i, n_c).A_c
        assert _rel(Ae[i, :k, :k].numpy(), ref) < 1e-12
        tail = Ae[i, k:, :].numpy()
        assert np.array_equal(tail[:, k:], np.eye(m2 + s - k)) and not np.any(tail[:, :k])
        assert np.array_equal(Ae[i].numpy(), Ae[i].numpy().T)


@pytest.mark.parametrize("n, n_c", [(16, 4), (32, 8), (32, 16)])
def test_the_correction_is_numpys_and_galerkin_orthogonal(n, n_c):
    """The modes (residual from the full field, ``Q`` on the loop) and a field (against the
    load), batched across examples with different corners, against `CoarseSpace.apply` per
    example to 1e-12; CX on each example's own ``V_e`` at most 1e-12."""
    bt = batch(n)
    ta = bt.torch_arrays()
    ts = cp.TorchEnlargedSpace.of_arrays(*ta, n_c, cp.patch_nodes(n, n_c, FLAGS))
    sq = model.plain_square(n, KIND, M_PORTS, "cpu")
    X = torch.tensor(np.stack([_img(x, n) for x in bt.X]))
    CX = ts.correct(X, co.residual_torch(*ta, X, sq.QB))
    xf = torch.tensor(np.stack([x[:, 5].reshape(n - 1, n - 1) for x in bt.X]))
    bf = torch.tensor(np.stack([b.reshape(n - 1, n - 1) for b in bt.b]))
    upt = ts.correct(xf, co.residual_torch(*ta, xf[:, None], rhs=bf[:, None])[:, 0])
    for i in range(3):
        cs = bt.numpy_space(i, n_c)
        got = CX[i].reshape(4 * M_PORTS, -1).T.numpy()
        assert _rel(got, cs.apply(bt.X[i], bt.Q)) < 1e-12
        BQ = -np.asarray(fe.Exact(bt.M[i], n).M_IG @ bt.Q)
        assert co.coarse_exactness(cs.P, co.residual(bt.M[i], n, got, bt.Q), BQ)["rel"] < 1e-12
        assert _rel(upt[i].numpy().ravel(), cs.apply(bt.X[i][:, 5], rhs=bt.b[i])) < 1e-12


def test_no_flagged_corner_is_the_coarse_grid_bit_for_bit():
    n, n_c = 32, 8
    bt = batch(n)
    ta = bt.torch_arrays()
    none = np.zeros((3, 4), bool)
    ts = cp.TorchEnlargedSpace.of_arrays(*ta, n_c, cp.patch_nodes(n, n_c, none))
    base = co.TorchCoarseSpace(co.coarse_matrix_torch(*ta, n_c), n, n_c)
    X = torch.tensor(np.stack([_img(x, n) for x in bt.X]))
    r = co.residual_torch(*ta, X, model.plain_square(n, KIND, M_PORTS, "cpu").QB)
    assert ts.base is not None and torch.equal(ts.correct(X, r), base.correct(X, r))
    again = cp.TorchEnlargedSpace.of_arrays(*ta, n_c, cp.patch_nodes(n, n_c, FLAGS))
    assert torch.equal(again.correct(X, r), cp.TorchEnlargedSpace.of_arrays(
        *ta, n_c, cp.patch_nodes(n, n_c, FLAGS)).correct(X, r))           # a repeat, bit for bit


def test_lemma_b_survives_the_torch_correction():
    """Steps only: inputs with ``X c1 + s0 X_1 = 1`` give outputs with the identity to
    rounding, with ``b_1`` from `coarse.level_rhs_torch`."""
    n, n_c = 32, 8
    bt = batch(n)
    which = [0, 1]                                         # the two steps
    ta = bt.torch_arrays(which)
    ts = cp.TorchEnlargedSpace.of_arrays(*ta, n_c, cp.patch_nodes(n, n_c, FLAGS[which]))
    s0 = torch.tensor([float(bt.a[i]["s0"]) for i in which])
    X = torch.tensor(np.stack([_img(bt.X[i], n) for i in which]))
    X1 = torch.tensor(np.stack([((1.0 - bt.X[i] @ bt.c1) / float(bt.a[i]["s0"])).reshape(n - 1, n - 1)
                                for i in which]))
    CX = ts.correct(X, co.residual_torch(*ta, X, model.plain_square(n, KIND, M_PORTS, "cpu").QB))
    b1, transient = co.level_rhs_torch(*ta, s0)
    assert bool(transient.all())
    C1 = ts.correct(X1, co.residual_torch(*ta, X1[:, None], rhs=b1[:, None])[:, 0])
    c1 = torch.tensor(bt.c1)
    ident = torch.einsum("bkij,k->bij", CX, c1) + s0[:, None, None] * C1
    assert float((ident - 1.0).abs().max()) < 1e-12


def test_the_gradient_is_Q_e_transposed():
    """For ``L = <G, C_e(X)>`` with the residual in the graph, ``dL/dX = G - A_I P_e A_e^-1
    P_e^T G`` per example."""
    n, n_c = 16, 4
    bt = batch(n)
    ta = bt.torch_arrays()
    ts = cp.TorchEnlargedSpace.of_arrays(*ta, n_c, cp.patch_nodes(n, n_c, FLAGS))
    sq = model.plain_square(n, KIND, M_PORTS, "cpu")
    K = 4 * M_PORTS
    X = torch.tensor(np.stack([_img(x, n) for x in bt.X]), requires_grad=True)
    G = torch.randn(3, K, n - 1, n - 1, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    (ts.correct(X, co.residual_torch(*ta, X, sq.QB)) * G).sum().backward()
    for i in range(3):
        cs = bt.numpy_space(i, n_c)
        A_I = fe.Exact(bt.M[i], n).A_I
        Gn = G[i].reshape(K, -1).T.numpy()
        want = Gn - A_I @ (cs.P @ cs.solve(cs.P.T @ Gn))
        assert _rel(X.grad[i].reshape(K, -1).T.numpy(), want) < 1e-12


def test_the_nodes_are_checked():
    bt = batch(16)
    with pytest.raises(ValueError):
        cp.enlarged_matrix_torch(*bt.torch_arrays(), 4, [np.array([3, 3]), np.array([]), np.array([])])
    with pytest.raises(ValueError):
        cp.enlarged_matrix_torch(*bt.torch_arrays(), 4, [np.array([15 ** 2]), np.array([]), np.array([])])
    with pytest.raises(ValueError):
        cp.enlarged_matrix_torch(*bt.torch_arrays(), 4, [np.array([1])])
