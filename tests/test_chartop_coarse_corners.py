"""Branch 2c's enlarged coarse space (`atlas/chartop/coarse.py`, `CornerTreatment`): the
rework page's section 15, numerically.  Classical solves and U's untrained form only.

``P_e = [P | S]`` with ``S`` built at a piece's non-right corners, by singular functions
(candidate A) or corner patches (candidate B).  Proposition A needs only full column rank,
so every item of it is checked again on ``V_e``:
- the corners: more than 2 degrees from 90 is non-right; on an affine chart the frozen
  tensor's wedge is the corner's own angle; the singular functions live in the wedge
  (the first is positive at every node inside the cut-off, zero beyond it); the patches
  are the corner block's fine hats less the coarse nodes, as many as the page's table says;
- Proposition A on ``V_e``: CX at most 1e-12, never worse than TL0 column by column and
  equal to a brute-force least-squares fit over ``V_e``, idempotence, blindness to shifts
  in ``V_e``, full column rank, traces untouched;
- Lemma B with the enlarged space;
- the controls: no non-right corner, or no columns, gives TL0 bit for bit; at ``n_c = n``
  nothing is added and columns are refused; the host's adapter with a treatment.

The pieces are generator pieces from seeds no registration draws (9,000,100 onward), as
in ``tests/test_chartop_coarse.py``."""
from __future__ import annotations

import math
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from atlas.chartop import chart as ch
from atlas.chartop import coarse as co
from atlas.chartop import data, fe, forms, ports, principal
from atlas.chartop.expert import LearnedExpert

torch.set_num_threads(1)

KIND, M_PORTS = "sine-hat", 16
#: seeds no registration draws: two steps and a steady piece, every corner non-right
STEP, SHORT_STEP, STEADY = 9_000_100, 9_000_112, 9_000_102
#: a conformal-map piece: every corner right
CONFORMAL = 9_000_104
TREATMENTS = (co.CornerTreatment("A", K=1), co.CornerTreatment("A", K=3),
              co.CornerTreatment("B", frac=8), co.CornerTreatment("B", frac=16))


class Piece:
    """A drawn piece's stored arrays (float32 read in float64), operator, exact modes, an
    input far from exact and its chart."""

    def __init__(self, seed: int, n: int, family: str | None = None):
        self.seed, self.n = seed, n
        self.example = data.draw(seed, family=family, n=n)
        self.a = {k: np.asarray(v, np.float64) for k, v in data.arrays(self.example).items()}
        self.X_chart = np.asarray(self.example.problem.chart.X)
        self.s0 = float(self.a["s0"])
        self.M = fe.assemble(n, self.a["a11"], self.a["a12"], self.a["a22"], self.a["sigma"])
        self.exact = fe.Exact(self.M, n)
        self.A_I = self.exact.A_I.tocsr()
        self.Q = ports.basis(KIND, n, M_PORTS).Q
        self.H = self.exact.modes(self.Q)
        self.BQ = -np.asarray(self.exact.M_IG @ self.Q)
        self.c1 = forms.kernel_coefficients(KIND, n, M_PORTS)
        rng = np.random.default_rng(seed)
        H0 = principal.modes(n, self.Q, self.s0)
        self.X = H0 + 0.05 * np.abs(H0).max() * rng.standard_normal(H0.shape)
        _loop, inner = fe.grid(n)
        self.b = (fe.mass(n, self.a["w"]) @ self.a["g"].ravel())[inner]

    def S(self, tr: co.CornerTreatment, n_c: int):
        return tr.columns(self.n, n_c, self.X_chart, self.a["a11"], self.a["a12"], self.a["a22"])

    def space(self, n_c: int, tr: co.CornerTreatment | None = None) -> co.CoarseSpace:
        return co.CoarseSpace.of_matrix(self.M, self.n, n_c, extra=None if tr is None else self.S(tr, n_c))

    def norms_A(self, E):
        return np.sqrt(np.maximum(np.einsum("ik,ik->k", E, np.asarray(self.A_I @ E)), 0.0))


_PIECES: dict = {}


def piece(seed: int, n: int = 16, family: str | None = None) -> Piece:
    if (seed, n, family) not in _PIECES:
        _PIECES[(seed, n, family)] = Piece(seed, n, family)
    return _PIECES[(seed, n, family)]


def _rel(a, b) -> float:
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max())


def parallelogram(n: int, theta_deg: float) -> np.ndarray:
    """The affine chart ``x = xi + eta cos theta``, ``y = eta sin theta`` [n+1, n+1, 2]."""
    t = math.radians(theta_deg)
    jj, ii = np.meshgrid(np.arange(n + 1) / n, np.arange(n + 1) / n, indexing="ij")
    return np.stack([ii + jj * math.cos(t), jj * math.sin(t)], axis=-1)


# ---------------------------------------------------------------------------
# the corners
# ---------------------------------------------------------------------------


def test_the_pieces_are_the_types_named():
    for seed in (STEP, SHORT_STEP, STEADY):
        assert co.non_right_corners(piece(seed).X_chart) == (0, 1, 2, 3)
    assert piece(STEP).s0 > 0 and piece(SHORT_STEP).s0 > 50.0 and piece(STEADY).s0 == 0.0
    conf = piece(CONFORMAL, family="conformal-map")
    assert conf.example.family == "conformal-map" and co.non_right_corners(conf.X_chart) == ()


@pytest.mark.parametrize("theta, right", [(88.1, True), (91.9, True), (87.9, False), (92.1, False),
                                          (120.0, False)])
def test_non_right_means_more_than_two_degrees_from_ninety(theta, right):
    """A parallelogram's corners are ``theta``, ``180 - theta``, ``theta``, ``180 - theta``."""
    X = parallelogram(16, theta)
    assert np.allclose(ch.corner_angles(X), [theta, 180 - theta, theta, 180 - theta], atol=1e-9)
    assert co.non_right_corners(X) == (() if right else (0, 1, 2, 3))
    assert co.RIGHT_DEG == 2.0


@pytest.mark.parametrize("theta", [60.0, 75.0, 105.2, 154.0])
def test_the_frozen_wedge_is_the_corners_angle_on_an_affine_chart(theta):
    """On an affine chart the pulled-back tensor is constant, and ``y = T x`` maps each
    corner's quadrant onto a wedge of exactly the corner's angle."""
    X = parallelogram(16, theta)
    a11, a12, a22, _J = ch.metric_tensor(ch.element_jacobian(X))
    angles = ch.corner_angles(X)
    for c, (t11, t12, t22) in enumerate(co.corner_tensors(a11, a12, a22)):
        omega, T = co.corner_wedge(t11, t12, t22, c)
        assert abs(math.degrees(omega) - angles[c]) < 1e-9
        v1, v2 = T[:, 0], T[:, 1]
        between = math.atan2(v1[0] * v2[1] - v1[1] * v2[0], v1 @ v2)       # counter-clockwise
        assert abs(between - omega) < 1e-12
        assert abs(np.linalg.det(T) - 1.0) < 1e-12


def test_the_corner_offsets_and_tensors_are_each_corners_own():
    n = 8
    a11 = np.arange(n * n, dtype=float).reshape(n, n)
    got = co.corner_tensors(a11, -a11, 2 * a11)
    assert [t[0] for t in got] == [a11[0, 0], a11[0, n - 1], a11[n - 1, n - 1], a11[n - 1, 0]]
    for c in range(4):
        a, b = co.corner_offsets(n, c)
        assert a.min() == b.min() == 1 and a.max() == b.max() == n - 1
        k = int(np.flatnonzero((a == 1) & (b == 1))[0])       # the node next to the corner
        j, i = divmod(k, n - 1)
        (xc, ec), _s = co.CORNERS[c]
        assert (i + 1, j + 1) == (1 if xc == 0 else n - 1, 1 if ec == 0 else n - 1)


@pytest.mark.parametrize("seed", [STEP, STEADY])
@pytest.mark.parametrize("K", [1, 3])
def test_the_singular_functions_live_in_the_wedge(seed, K):
    """Each column's largest absolute value is one; it vanishes at every node beyond the
    cut-off (``|x| >= R``).  The first function of each corner, ``sin(pi phi / omega)``
    with ``phi`` in the wedge, is positive at every node inside the cut-off: ``phi`` is
    measured within ``(0, omega)`` at all four corners, whichever way the frame turns."""
    p = piece(seed)
    S = p.S(co.CornerTreatment("A", K=K), 8).toarray()
    assert S.shape == ((p.n - 1) ** 2, 4 * K)
    assert np.allclose(np.abs(S).max(axis=0), 1.0, rtol=0, atol=1e-15)
    for c in range(4):
        a, b = co.corner_offsets(p.n, c)
        r = np.hypot(a / p.n, b / p.n)
        col = S[:, c * K]
        assert not np.any(S[r >= co.SINGULAR_R, c * K:(c + 1) * K])
        assert np.all(col[r < co.SINGULAR_R] > 0.0)


def test_the_patches_are_the_corner_blocks_less_the_coarse_nodes():
    """The page's table: ``(n/8)^2`` nodes less four coarse ones per corner at ``n_c = 16``,
    60 at ``n = 64`` and 252 at 128; a quarter with ``n/16`` blocks.  Each column is one
    fine hat, at a node of its corner's block that is not a coarse node, and no node twice."""
    assert co.corner_patch_columns(64, 16, (0,), 8).shape[1] == 60
    assert co.corner_patch_columns(128, 16, (0,), 8).shape[1] == 252
    assert co.corner_patch_columns(64, 16, (0,), 16).shape[1] == 15
    assert co.corner_patch_columns(128, 16, (0,), 16).shape[1] == 63
    n, n_c, w = 32, 8, 4
    S = co.corner_patch_columns(n, n_c, (0, 1, 2, 3), 8)
    assert S.shape[1] == 4 * (w * w - 1)
    assert np.array_equal(S.sum(axis=0).A.ravel(), np.ones(S.shape[1]))
    rows = S.tocsc().indices
    assert len(set(rows.tolist())) == len(rows)
    r = n // n_c
    for c in range(4):
        a, b = co.corner_offsets(n, c)
        mine = np.flatnonzero((a <= w) & (b <= w))
        jj, ii = np.divmod(mine, n - 1)
        coarse = ((ii + 1) % r == 0) & ((jj + 1) % r == 0)
        assert set(mine[~coarse].tolist()) <= set(rows.tolist())
        assert not set(mine[coarse].tolist()) & set(rows.tolist())
    for frac in (2, 32, 0):
        with pytest.raises(ValueError):
            co.corner_patch_columns(16, 4, (0,), frac)
    with pytest.raises(ValueError):
        co.CornerTreatment("C")


# ---------------------------------------------------------------------------
# Proposition A on the enlarged space
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tr", TREATMENTS, ids=lambda t: t.name)
@pytest.mark.parametrize("seed", [STEP, SHORT_STEP, STEADY])
@pytest.mark.parametrize("n_c", [4, 8])
def test_galerkin_orthogonality_on_the_enlarged_space(tr, seed, n_c):
    """``P_e^T (B Q - A_I C_e(X)) = 0`` to 1e-12 of ``P_e^T B Q``; likewise a field.  The
    space really is larger, and ``P_e`` has full column rank."""
    p = piece(seed)
    cs = p.space(n_c, tr)
    assert cs.n_extra > 0 and cs.size == (n_c - 1) ** 2 + cs.n_extra
    sv = np.linalg.svd(cs.P.toarray(), compute_uv=False)
    assert sv.min() > 1e-3 * sv.max()
    CX = cs.apply(p.X, p.Q)
    assert co.coarse_exactness(cs.P, co.residual(p.M, p.n, CX, p.Q), p.BQ)["rel"] < 1e-12
    up = cs.apply(p.X[:, 0], rhs=p.b)
    assert co.coarse_exactness(cs.P, co.residual(p.M, p.n, up, rhs=p.b), p.b)["rel"] < 1e-12
    base = p.space(n_c)                       # the coarse grid's own CX of C_e(X) is zero too
    assert co.coarse_exactness(base.P, co.residual(p.M, p.n, CX, p.Q), p.BQ)["rel"] < 1e-12


@pytest.mark.parametrize("tr", TREATMENTS, ids=lambda t: t.name)
@pytest.mark.parametrize("seed", [STEP, STEADY])
def test_never_worse_than_tl0_and_the_least_squares_optimum(tr, seed):
    """Column by column ``||H - C_e(X)||_A <= ||H - C(X)||_A``, and equal to the minimum
    over ``V_e`` found by brute force (a dense Cholesky of ``A_I`` and least squares)."""
    p = piece(seed)
    n_c = 4
    cs, base = p.space(n_c, tr), p.space(n_c)
    E = p.X - p.H
    nE = p.norms_A(E)
    ne = p.norms_A(cs.apply(p.X, p.Q) - p.H)
    nb = p.norms_A(base.apply(p.X, p.Q) - p.H)
    assert np.all(ne <= nb * (1.0 + 1e-12))
    assert np.sum(ne < nb * (1.0 - 1e-6)) > 0                 # and better somewhere
    L = np.linalg.cholesky(p.A_I.toarray())
    LP = L.T @ cs.P.toarray()
    best = np.array([np.linalg.norm(L.T @ E[:, k] + LP @ np.linalg.lstsq(LP, -(L.T @ E[:, k]),
                                                                          rcond=None)[0])
                     for k in range(E.shape[1])])
    assert np.max(np.abs(ne - best) / nE) < 1e-10


@pytest.mark.parametrize("tr", TREATMENTS, ids=lambda t: t.name)
def test_idempotent_blind_to_enlarged_shifts_and_traces_untouched(tr):
    p = piece(STEP)
    cs = p.space(4, tr)
    CX = cs.apply(p.X, p.Q)
    assert _rel(cs.apply(CX, p.Q), CX) < 1e-13
    v = np.random.default_rng(2).standard_normal((cs.size, p.X.shape[1]))
    assert _rel(cs.apply(p.X + cs.P @ v, p.Q), CX) < 1e-12
    loop, _inner = fe.grid(p.n)
    assert np.array_equal(fe.full_field(p.n, CX, p.Q)[loop], p.Q)


@pytest.mark.parametrize("tr", TREATMENTS, ids=lambda t: t.name)
@pytest.mark.parametrize("seed", [STEP, SHORT_STEP])
def test_lemma_b_survives_the_enlarged_correction(tr, seed):
    """Inputs with ``X c1 + s0 X_1 = 1`` give ``C_e(X) c1 + s0 C_e,1(X_1) = 1`` to rounding
    with ``b_1 = (M 1)_I / s0``: every column sees the same operator."""
    p = piece(seed)
    X1 = (1.0 - p.X @ p.c1) / p.s0
    S = p.S(tr, 4)
    Hc, upc, levc = co.correct_fields(p.M, p.n, 4, p.X, p.X[:, 0], X1, p.s0, p.b, p.Q, extra=S)
    assert np.abs(Hc @ p.c1 + p.s0 * levc - 1.0).max() < 1e-13
    assert _rel(levc, X1) > 1e-6


# ---------------------------------------------------------------------------
# the controls
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tr", TREATMENTS, ids=lambda t: t.name)
def test_no_non_right_corner_is_tl0_bit_for_bit(tr):
    """A conformal piece's corners are all right: ``S`` is empty and every output is the
    coarse grid's own, bit for bit; so are empty columns passed by hand."""
    p = piece(CONFORMAL, family="conformal-map")
    S = p.S(tr, 4)
    assert S.shape == ((p.n - 1) ** 2, 0)
    for n_c in (4, 8):
        a = co.correct_fields(p.M, p.n, n_c, p.X, p.X[:, 0], p.X[:, 1], p.s0, p.b, p.Q, extra=S)
        b = co.correct_fields(p.M, p.n, n_c, p.X, p.X[:, 0], p.X[:, 1], p.s0, p.b, p.Q)
        assert all(np.array_equal(x, y) for x, y in zip(a, b))
        cs = co.CoarseSpace.of_matrix(p.M, p.n, n_c, extra=S)
        assert cs.n_extra == 0 and np.array_equal(cs.A_c, co.CoarseSpace.of_matrix(p.M, p.n, n_c).A_c)


def test_at_the_top_nothing_is_added():
    """At ``n_c = n`` the coarse space is the whole interior: both treatments add nothing,
    the correction is the exact solve, and columns passed by hand are refused."""
    p = piece(STEP)
    for tr in TREATMENTS:
        S = p.S(tr, p.n)
        assert S.shape[1] == 0
        cs = co.CoarseSpace.of_matrix(p.M, p.n, p.n, extra=S)
        assert cs.top and _rel(cs.apply(p.X, p.Q), p.H) < 1e-10
    with pytest.raises(ValueError):
        co.CoarseSpace.of_matrix(p.M, p.n, p.n, extra=p.S(TREATMENTS[2], 8))
    with pytest.raises(ValueError):
        co.CoarseSpace.of_matrix(p.M, p.n, 8, extra=np.ones((10, 2)))


@pytest.fixture(scope="module")
def u_expert():
    """U: S's form with zero heads, behind the host's adapter (as ``test_chartop_coarse``)."""
    torch.manual_seed(0)
    net = forms.KernelExpert("split", "unet", "1M", KIND, M_PORTS, None, "corner", True).eval()
    return LearnedExpert(net, envelope=data.ENVELOPE)


@pytest.mark.parametrize("tr", [TREATMENTS[1], TREATMENTS[2]], ids=lambda t: t.name)
def test_the_adapter_with_a_corner_treatment(u_expert, tr):
    """Through the host's adapter: on pieces with non-right corners the answers are
    coarse-exact on ``V_e`` (the columns built from each problem's own chart and tensors),
    kernel-exact, and no column is worse than TL0's; on a conformal piece they are TL0's,
    bit for bit."""
    probs = [piece(s).example.problem for s in (STEP, SHORT_STEP, STEADY)]
    pb = lambda n: ports.basis(KIND, n, M_PORTS)            # noqa: E731
    n_c = 4
    ad = co.CoarseCorrected(u_expert, n_c, corners=tr)
    got = ad.respond(probs, pb)
    tl0 = co.CoarseCorrected(u_expert, n_c).respond(probs, pb)
    assert ad.extra_columns > 0 and ad.coarse_seconds > 0.0
    for p, r, q in zip(probs, got, tl0):
        n, Q, M = p.n, pb(p.n).Q, p.matrix()
        loop, inner = fe.grid(n)
        cs = co.CoarseSpace.of_matrix(M, n, n_c, extra=tr.of_problem(p, n_c))
        BQ = -np.asarray(M.tocsr()[inner][:, loop] @ Q)
        assert co.coarse_exactness(cs.P, co.residual(M, n, r.H, Q), BQ)["rel"] < 1e-12
        b = p.load()[inner]
        assert co.coarse_exactness(cs.P, co.residual(M, n, r.up, rhs=b), b)["rel"] < 1e-12
        c1 = forms.kernel_coefficients(KIND, n, M_PORTS)
        lev = p.s0 * r.level if p.s0 > 0 else 0.0
        assert np.abs(r.H @ c1 + lev - 1.0).max() < 1e-12
        ex = fe.Exact(M, n)
        H = ex.modes(Q)
        A_I = ex.A_I.tocsr()

        def norms(E):
            return np.sqrt(np.maximum(np.einsum("ik,ik->k", E, np.asarray(A_I @ E)), 0.0))
        assert np.all(norms(r.H - H) <= norms(q.H - H) * (1.0 + 1e-9) + 1e-12)
    conf = [piece(CONFORMAL, family="conformal-map").example.problem]
    a = co.CoarseCorrected(u_expert, n_c, corners=tr).respond(conf, pb)[0]
    b = co.CoarseCorrected(u_expert, n_c).respond(conf, pb)[0]
    assert np.array_equal(a.H, b.H) and np.array_equal(a.up, b.up) and np.array_equal(a.level, b.level)
