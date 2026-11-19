"""The coolant plan's Phase 2 building blocks: the channel flow on pieces
(`transport_flow`) and the safety switch's residual (`transport_switch`)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from atlas.chartop import chart as ch
from atlas.chartop import fe
from atlas.chartop import ports as pt
from atlas.chartop import transport as tr
from atlas.chartop import transport_cases as tc
from atlas.chartop import transport_coarse as tco
from atlas.chartop import transport_flow as tf
from atlas.chartop import transport_host as th
from atlas.chartop import transport_switch as sw


# ---------------------------------------------------------------------------
# the flow
# ---------------------------------------------------------------------------


def test_a_straight_channel_gets_the_plane_poiseuille_profile():
    b = tc.channel(n=16, K=3, pe=0.0)
    charts = [p.chart for p in b.problems]

    def face(i, s, P):
        if np.all(np.abs(P[:, 0]) < 1e-12):
            return "inlet"
        if np.all(np.abs(P[:, 0] - 3 * tc.L_PIECE) < 1e-12):
            return "outlet"
        return "wall"
    psi, info = tf.stream_functions(charts, face, speed=0.01)
    want = [0.01 * tc.W_CH * tr.profile(np.clip(c.X[..., 1] / tc.W_CH, 0, 1)) for c in charts]
    assert max(float(np.abs(a - w).max()) for a, w in zip(psi, want)) <= 1e-17
    assert info[0]["flux"] == pytest.approx(0.01 * tc.W_CH, rel=1e-12)


def test_the_bend_gets_the_log_radial_profile():
    bb = tc.bend(n=16, n_r=2, n_th=3, pe=0.0)

    def face(i, s, P):
        if np.all(np.abs(P[:, 0]) < 1e-12):
            return "inlet"
        if np.all(np.abs(P[:, 1]) < 1e-12):
            return "outlet"
        return "wall"
    psi, _ = tf.stream_functions([p.chart for p in bb.problems], face, speed=0.01)
    r0, r1 = tc.R_IN, tc.R_OUT
    for p, a in zip(bb.problems, psi):
        X = p.chart.X
        s = (np.log(np.hypot(X[..., 0], X[..., 1])) - math.log(r0)) / math.log(r1 / r0)
        assert np.abs(a - 0.01 * (r1 - r0) * tr.profile(np.clip(s, 0, 1))).max() <= 1e-17


def _u_turn(n=16):
    def sq(x0, y0):
        return ch.Chart(ch.from_map(lambda xi, eta: (x0 + xi, y0 + eta), n), "affine")
    cells = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (2, 2), (1, 2), (0, 2)]

    def face(i, s, P):
        if np.all(np.abs(P[:, 0]) < 1e-12) and np.all(P[:, 1] <= 1 + 1e-12):
            return "inlet"
        if np.all(np.abs(P[:, 0]) < 1e-12) and np.all(P[:, 1] >= 2 - 1e-12):
            return "outlet"
        return "wall"
    return [sq(*c) for c in cells], face


def test_a_u_turn_conserves_mass_on_every_seam():
    charts, face = _u_turn()
    psi, info = tf.stream_functions(charts, face, speed=1.0)
    assert info[0]["flux"] == pytest.approx(1.0) and info[0]["inlet_sides"] == [[0, 3]]
    pieces = [tr.make_piece(c, p, k=1.0, alpha=1.0) for c, p in zip(charts, psi)]

    def bc(i, s, P):
        return ("dirichlet", np.zeros(len(P))) if face(i, s, P) == "inlet" else ("insulated", None)
    L = th.TransportLayout(pieces, bc, m=8)               # refuses a seam that loses mass
    assert sum(1 for sm in L.seams if sm.kind == "advective") == 8
    # the walls carry no flow: psi is 0 on the right-hand wall and the flux on the left
    for p in pieces:
        for s in range(4):
            if not p.side_flow(s):
                assert abs(p.side_flux(s)) <= 1e-12


def test_a_channel_without_an_outlet_is_refused():
    charts, face = _u_turn()
    with pytest.raises(tf.FlowError, match="no outlet"):
        tf.stream_functions(charts, lambda i, s, P: "inlet" if face(i, s, P) == "inlet" else "wall", 1.0)


def test_a_channel_with_two_inlet_faces_is_refused():
    charts, face = _u_turn()

    def two(i, s, P):
        k = face(i, s, P)
        if k == "wall" and np.all(np.abs(P[:, 1] - 3.0) < 1e-12) and np.all(P[:, 0] <= 1 + 1e-12):
            return "inlet"
        return k
    with pytest.raises(tf.FlowError, match="one inlet face"):
        tf.stream_functions(charts, two, 1.0)


# ---------------------------------------------------------------------------
# the switch
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def piece():
    return tc.channel(n=32, K=2, pe=50.0).problems[0]


def test_the_exact_dual_norm_is_the_d_norm_of_the_error_for_a_symmetric_piece():
    p = tc.channel(n=16, K=2, pe=0.0).problems[0]
    Q = pt.basis("sine-hat", 16, 8).Q
    ex, _ = tr.exact_response(p, Q)
    rng = np.random.default_rng(3)
    E = 1e-2 * rng.standard_normal(ex.H.shape)
    D = sw.interior_diffusion(p)
    loop, inner = fe.grid(16)
    K = p.matrix().tocsr()
    R = -np.asarray(K[inner][:, loop] @ Q) - K[inner][:, inner] @ (ex.H + E)
    eta = sw.dual_residual(R, D, "exact")
    assert eta == pytest.approx(math.sqrt(float(np.einsum("ij,ij->", E, D @ E))), rel=1e-10)


def test_the_schwarz_estimate_is_equivalent_to_the_exact_one(piece):
    from atlas.chartop import coarse as co
    D = sw.interior_diffusion(piece)
    rng = np.random.default_rng(4)
    ratios = []
    for k in range(6):
        R = rng.standard_normal((31 * 31, 4))
        ratios.append(sw.dual_residual(R, D, "schwarz", n=32, n_c=8) / sw.dual_residual(R, D, "exact"))
    assert 0.3 < min(ratios) and max(ratios) < 3.0          # spectrally equivalent
    t = np.arange(1, 32) / 32
    X, Y = np.meshgrid(t, t)
    smooth = (np.sin(np.pi * X) * np.sin(np.pi * Y)).ravel()[:, None]
    full = sw.dual_residual(smooth, D, "schwarz", n=32, n_c=8, P=co.prolongation(32, 8))
    assert 0.5 < full / sw.dual_residual(smooth, D, "exact") < 2.0     # the coarse term carries it
    # Jacobi leans on the fine modes: it under-weights a smooth residual many times more
    rough = rng.standard_normal((31 * 31, 1))
    js = sw.dual_residual(smooth, D, "jacobi") / sw.dual_residual(smooth, D, "exact")
    jr = sw.dual_residual(rough, D, "jacobi") / sw.dual_residual(rough, D, "exact")
    assert js < 0.3 * jr


def test_tl0s_residual_falls_with_the_coarse_size(piece):
    Q = pt.basis("sine-hat", 32, 16).Q
    etas = [sw.piece_eta(piece, tco.TL0TransportExpert(n_c=nc).response(piece, Q), Q, "exact")["total"]
            for nc in (4, 8, 16, 32)]
    assert etas[0] > etas[1] > etas[2] > etas[3] and etas[3] <= 1e-9 * etas[0]


def test_the_switch_keeps_the_better_answer(piece):
    """An exact 'learned' expert is kept on every piece; one worse than TL0 never is,
    and the switched answers are then TL0's."""
    b = tc.channel(n=32, K=2, pe=50.0)
    L = b.layout(m=16)
    good = sw.SwitchedExpert(tr.ClassicalTransportExpert(), tco.TL0TransportExpert(n_c=8), mode="exact")
    L.solve(good)
    assert all(k["kept"] for k in good.kept)

    class Worse:
        name = "worse"
        envelope = None

        def respond(self, problems, pb_of):
            out = []
            for p in problems:
                r = tco.TL0TransportExpert(n_c=8).response(p, pb_of(p.n).Q)
                out.append(tr.TransportResponse(r.H * 1.1, r.G * 1.1, r.up * 1.1))
            return out
    bad = sw.SwitchedExpert(Worse(), tco.TL0TransportExpert(n_c=8), mode="schwarz", n_c=8)
    a = L.solve(bad)
    assert not any(k["kept"] for k in bad.kept)
    ref = L.solve(tco.TL0TransportExpert(n_c=8))
    for u, v in zip(a.fields, ref.fields):
        assert np.max(np.abs(u - v)) <= 1e-12 * np.max(np.abs(v))


def test_unknown_mode():
    with pytest.raises(ValueError):
        sw.dual_residual(np.zeros((4, 1)), None, "l2")
    with pytest.raises(ValueError):
        sw.dual_residual(np.zeros((4, 1)), None, "schwarz")
