"""The evaluation's own machinery (Stage 0, W355): the dual bracket's Loewner chain
for any outputs, the frozen-coefficient route against the Kirchhoff transform, and the
learned expert through the host (one call per piece, the envelope refusal).  Untrained
networks only: what is checked here holds for any weights."""
from __future__ import annotations

import math
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from atlas.chartop import benchmarks as bm
from atlas.chartop import chart as ch
from atlas.chartop import data, dual, fe, host, model, nonlinear as nl, ports
from atlas.chartop.expert import LearnedExpert
from atlas.chartop.piece import PieceProblem

torch.set_num_threads(1)


def _steady_piece(n=16, seed=4):
    e = data.draw(data.SEEDS["train"] + 7000 + seed, n=n)
    p = e.problem
    return PieceProblem(p.chart, log_kappa_hat=p.log_kappa_hat, log_kappa_hat_e=p.log_kappa_hat_e,
                        fo=None, g=np.zeros_like(p.g))


@pytest.mark.parametrize("scale", [0.0, 0.05, 1.0])
def test_dual_chain_for_any_outputs(scale):
    rng = np.random.default_rng(1)
    p = _steady_piece()
    n = p.n
    pb = ports.basis("sine-hat", n, 8)
    H = fe.Exact(p.matrix(), n).modes(pb.Q)
    Hd = fe.Exact(dual.dual_problem(p).matrix(), n).modes(pb.Q)
    r = dual.bracket(p, pb, H + scale * rng.normal(size=H.shape),
                     Hd + scale * rng.normal(size=Hd.shape),
                     traces=[rng.normal(size=pb.size) for _ in range(3)])
    assert min(r["min_eig_rel"].values()) >= -1e-10
    for d in r["data"]:
        assert d["discrete_bracket"] >= -1e-10 * d["energy"]
        assert d["computable_gap"] >= d["network_error"] - 1e-10 * d["energy"]


def test_frozen_coefficients_converge_to_kirchhoff_on_the_bend():
    b = bm.bend3(16, "steady")
    beta = -3e-3
    it = nl.iterate(b, [beta] * len(b.problems), None, tol=1e-9, maxit=60)
    assert it["converged"] and it["iterations"] < 30
    lin = host.monolithic(b.layout("sine-hat", 16)).fields
    kh = nl.kirchhoff(b, beta, lin, bm.T_HOT, bm.T_COLD)
    err = max(float(np.max(np.abs(u - v))) for u, v in zip(it["fields"], kh))
    assert err < 1e-6 * 100.0
    # and the nonlinearity is not negligible (the demonstration measures something)
    assert max(float(np.max(np.abs(u - v))) for u, v in zip(it["fields"], lin)) > 1.0


def test_untrained_expert_through_the_host():
    """SE1 and SE2 hold for any network: one call per piece, the Gram identity, and
    lambda_min no lower than the classical superelement's with the same modes."""
    b = bm.bend3(32, "step")
    L = b.layout("sine-hat", 16)
    net = model.Expert("unet", "0.3M")
    for prm in net.parameters():
        torch.nn.init.normal_(prm, 0.0, 0.02)
    ex = LearnedExpert(net, envelope=data.ENVELOPE)
    cp = L.solve(ex)
    assert cp.calls == len(b.problems) == ex.calls
    cl = L.solve(host.ClassicalExpert())
    for i, p in enumerate(L.problems):
        E = cp.responses[i].H - cl.responses[i].H
        A_I = p.matrix().tocsr()[fe.grid(p.n)[1]][:, fe.grid(p.n)[1]]
        res = cp.grams[i] - cl.grams[i] - E.T @ (A_I @ E)
        assert np.max(np.abs(res)) <= 1e-10 * np.max(np.abs(cl.grams[i]))
    assert cp.min_eig() >= cl.min_eig() * (1 - 1e-10)


def test_the_host_refuses_a_piece_outside_the_envelope():
    n = 16
    c = ch.Chart(ch.from_map(lambda x, y: (5.0 * x, y), n), "affine")   # modulus 5: |mu| = 2/3
    p = PieceProblem(c, fo=None)

    def bc(i, s, P):
        return ("dirichlet", np.full(len(P), 1.0)) if s == 3 else ("insulated", None)
    L = host.Layout([p], bc, "sine-hat", 8)
    ex = LearnedExpert(model.Expert("unet", "0.3M", m=8), envelope=data.ENVELOPE)
    with pytest.raises(host.Refused, match="sup\\|mu\\|"):
        L.solve(ex)
    assert ex.calls == 0


def test_the_card_carries_its_envelope_and_certificates(tmp_path):
    from atlas.chartop import card
    w = tmp_path / "w.pt"
    w.write_bytes(b"weights")
    ev = {"written": "2026-10-18", "gates_sha256": "abc",
          "E1": {"met": False, "cells": [{"mu_bin": [0.0, 0.15], "freq": "low", "count": 40}]},
          "benchmarks": [{"arms": {"B": {"se1": {"identity_rel": 2e-15}}}}],
          "DUAL": {"benchmark_records": [{"data": [{"computable_gap": 2.0, "network_error": 1.0}]}]}}
    plan = {"arms": [{"name": "B", "steps": 10}], "data": {"train": 5}}
    c = card.card(ev, str(w), plan, {"kind": "sine-hat", "m": 16, "form": "bubble",
                                     "backbone": "unet", "size": "1M", "parameters": 1})
    assert c["chart"]["envelope"]["sup_abs_mu"] == data.ENVELOPE.k_mu
    assert c["time"]["fourier_range"] == [data.ENVELOPE.fo_min, data.ENVELOPE.fo_max]
    assert c["certificates"]["gram_identity_residual_SE1"]["value"] == 2e-15
    assert c["certificates"]["dual_gap_over_network_error_smooth_data"]["median"] == 2.0
    assert c["certificates"]["contraction_factor_tier2"] is None      # reported missing, not assumed
    assert len(c["weights"]["sha256"]) == 64 and "Poseidon" in c["licence"]
