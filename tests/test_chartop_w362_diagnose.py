"""W362's untrained diagnostic (`scripts/chartop_w362_diagnose.py`): its wrapper is W358's own
measure, and its control holds.  Classical solves and U's untrained form only."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import chartop_corner_controls as CC  # noqa: E402
import chartop_w362_diagnose as D  # noqa: E402
from atlas.chartop import data, forms, ports, principal  # noqa: E402

N = 16


def _setup(n=N):
    Q = ports.basis("sine-hat", n, CC.M_PORTS).Q
    c1 = forms.kernel_coefficients("sine-hat", n, CC.M_PORTS)
    H0s = principal.modes(n, Q, 0.0)
    W = (c1[None, :] * H0s) / (H0s @ (c1 * c1))[:, None]
    return Q, W, c1, {16: CC.smooth_basis(n, 16)}, forms.corner_bubble(n)[1:-1, 1:-1].ravel()


def test_the_wrapper_is_w358s_measure_on_a_drawn_piece():
    """On a validation piece W358 measured, the wrapper given that piece's arrays answers as
    `chartop_corner_controls.Piece` does from its seed."""
    Q, W, c1, Phis, beta = _setup()
    seed = data.SEEDS["val"] + 100
    a = data.arrays(data.draw(seed, n=N))
    want = CC.Piece(seed, N, Q, W, c1).candidate(*D.CANDIDATE, Phis, beta)
    got = D.ArraysPiece(dict(a), "drawn", N, Q, W, c1).candidate(*D.CANDIDATE, Phis, beta)
    for k in ("e_H_classical", "e_H_smooth_K16", "peak", "rms"):
        assert got[k] == want[k], k
    assert CC.data.draw is data.draw and CC.data.arrays is data.arrays     # restored after the wrapper


def test_the_control_u_is_exact_on_a_square_and_not_on_a_rectangle():
    Q, W, c1, Phis, beta = _setup()
    sq = D.ArraysPiece(D.arrays_of(D.rectangle(1.0, N)), "square", N, Q, W, c1).candidate(*D.CANDIDATE, Phis, beta)
    assert sq["e_H_classical"] < 1e-12 and sq["peak"] < 1e-9
    rect = D.ArraysPiece(D.arrays_of(D.rectangle(0.7, N)), "0.7", N, Q, W, c1).candidate(*D.CANDIDATE, Phis, beta)
    assert rect["e_H_classical"] > 0.1 and rect["e_H_smooth_K16"] < rect["e_H_classical"]


def test_the_ring_pieces_do_not_repeat_by_translation():
    rep = D.repetition()
    assert rep["squares 9000003 (hole)"]["non_square"] == 8 and rep["squares 9000003 (hole)"]["share_reuse_answers"] == 0.0
    assert rep["workbench plate-insert"]["share_reuse_answers"] > 0.5
