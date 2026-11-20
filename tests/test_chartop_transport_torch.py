"""The CO-T expert as a differentiable map (`atlas/chartop/transport_torch.py`): the
stencils are K_II, the batch's TL0 is `transport_coarse`'s, zero heads are TL0, the
coarse correction stays exact for any weights, the label-free loss is the plain square's
dual norm, and gradients reach the network."""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from atlas.chartop import coarse as co  # noqa: E402
from atlas.chartop import fe  # noqa: E402
from atlas.chartop import ports as pt  # noqa: E402
from atlas.chartop import principal as pr  # noqa: E402
from atlas.chartop import transport_coarse as tco  # noqa: E402
from atlas.chartop import transport_data as td  # noqa: E402
from atlas.chartop import transport_torch as tt  # noqa: E402

N = 32
SEEDS = (12_000_003, 12_000_005)           # flowing pieces at n = 32 (asserted below)


@pytest.fixture(scope="module")
def pieces():
    out = []
    for s in SEEDS:
        p, meta = td.draw(s, N)
        out.append(td.piece_from_arrays(td.arrays(p, meta)))
    assert all(p.flows for p in out), "the fixture needs flowing pieces"
    return out


@pytest.fixture(scope="module")
def batch(pieces):
    labels = [td.labels(p, full=True) for p in pieces]
    return tt.PieceBatch(pieces, n_c=8, labels=labels)


def test_groups_partition_the_columns():
    g = tt.groups(16)
    assert [len(c) for c in g.values()] == [4, 12, 20, 28]
    assert sorted(sum(g.values(), [])) == list(range(64))


def test_the_dst_diagonalises_the_plain_square():
    S = tt.dst_matrix(N)
    assert np.allclose(S @ S, np.eye(N - 1), atol=1e-13)
    r = np.random.default_rng(0).standard_normal((N - 1) ** 2)
    for s0 in (0.0, 3.7):
        want = r @ pr.solve(N, r, s0)
        R = r.reshape(N - 1, N - 1)
        Rh = S @ R @ S
        got = float(np.sum(Rh * Rh / pr.symbol(N, s0)))
        assert abs(got - want) <= 1e-12 * abs(want)


def test_the_stencils_are_the_interior_block(pieces):
    p = pieces[0]
    loop, inner = fe.grid(N)
    K_II = p.matrix().tocsr()[inner][:, inner]
    c = torch.tensor(tt.interior_stencil(p.matrix(), N))[None]
    X = np.random.default_rng(1).standard_normal(((N - 1) ** 2, 3))
    Xi = torch.tensor(X.T.reshape(3, N - 1, N - 1))[None]
    got = tt.to_interior(tt.apply_stencil(c, Xi)[0])
    got_t = tt.to_interior(tt.apply_stencil_t(c, Xi)[0])
    assert np.max(np.abs(got - K_II @ X)) <= 1e-12 * np.max(np.abs(K_II @ X))
    assert np.max(np.abs(got_t - K_II.T @ X)) <= 1e-12 * np.max(np.abs(K_II.T @ X))


def test_the_batch_tl0_is_transport_coarse_tl0(pieces, batch):
    H, G, up = batch.tl0()
    Q = pt.basis("sine-hat", N, 16).Q
    for b, p in enumerate(pieces):
        ref = tco.TL0TransportExpert(8).response(p, Q)
        for got, want in ((tt.to_interior(H[b]), ref.H), (tt.to_interior(G[b]), ref.G),
                          (tt.to_interior(up[b]), ref.up)):
            assert np.max(np.abs(got - want)) <= 1e-11 * max(1.0, np.max(np.abs(want)))


def test_zero_heads_are_tl0_bit_for_bit(batch):
    for form in tt.FORMS:
        e = tt.ArmR("0.3M", form)
        with torch.no_grad():
            H, G, up = e(batch)
        H0, G0, u0 = batch.tl0()
        assert torch.equal(H, H0) and torch.equal(G, G0) and torch.equal(up, u0)
        assert abs(float(tt.objective(batch, H, G, up)) - 1.0) <= 1e-12


def test_the_coarse_correction_is_exact_for_any_weights(batch):
    e = tt.ArmR("0.3M", "free", seed=3)
    with torch.no_grad():
        for prm in e.net.inner.head.parameters():
            prm.normal_(0.0, 0.05)
        H, G, up = e(batch)
    from atlas.chartop.transport_torch import apply_stencil, apply_stencil_t
    rc = co.restrict(batch.F - apply_stencil(batch.C, H), batch.p)
    rca = co.restrict(batch.Fa - apply_stencil_t(batch.C, G), batch.p)
    rcp = co.restrict(batch.b - apply_stencil(batch.C, up[:, None])[:, 0], batch.p)
    ref = co.restrict(batch.F, batch.p).abs().max()
    assert float(rc.abs().max() / ref) <= 1e-10
    assert float(rca.abs().max() / co.restrict(batch.Fa, batch.p).abs().max()) <= 1e-10
    assert float(rcp.abs().max() / co.restrict(batch.b, batch.p).abs().max()) <= 1e-10
    H0, _, _ = batch.tl0()
    assert float((H - H0).abs().max()) > 1e-4          # the weights did change the answer


def test_the_loss_vanishes_at_the_exact_modes_and_errors_at_tl0_are_one(batch):
    L = batch.labels
    t, a, p = batch.residuals(L["H"], L["G"], L["up"])
    t0, a0, p0 = batch.residuals(*batch.tl0())
    assert float(t.max() / t0.max()) <= 1e-20 and float(a.max() / a0.max()) <= 1e-20
    assert float(p.max() / p0.max()) <= 1e-20
    r = tt.relative(batch.group_errors(*batch.tl0()), batch.err0)
    assert torch.allclose(r, torch.ones_like(r))
    assert float(batch.err0.min()) > 0.0


def test_gradients_reach_the_network_and_one_step_lowers_the_loss(batch):
    e = tt.ArmR("0.3M", "free", seed=0)
    opt = torch.optim.Adam(e.parameters(), lr=1e-3)
    losses = []
    for _ in range(3):
        loss = tt.objective(batch, *e(batch))
        opt.zero_grad()
        loss.backward()
        g = e.net.inner.head.weight.grad
        assert g is not None and float(g.abs().max()) > 0.0
        opt.step()
        losses.append(float(loss))
    assert losses[0] == pytest.approx(1.0, abs=1e-12)
    assert losses[-1] < losses[0]


def test_gate_k_holds_whatever_the_weights_on_steady_pieces():
    steady = []
    s = 12_000_000
    while len(steady) < 2:
        p, meta = td.draw(s, N)
        if meta["fo"] is None and meta["kind"] != "solid":
            steady.append(p)
        s += 1
    b = tt.PieceBatch(steady, n_c=8)
    hats = tt.groups(16)["hats"]
    for kernel, holds in ((True, True), (False, False)):
        e = tt.ArmR("0.3M", "free", seed=5, kernel=kernel)
        with torch.no_grad():
            for prm in e.net.inner.head.parameters():
                prm.normal_(0.0, 0.05)
            H, _, _ = e(b)
        dev = float((H[:, hats].sum(1) - 1.0).abs().max())
        assert (dev <= 1e-10) == holds, (kernel, dev)
    H0, _, _ = b.tl0()
    assert float((H0[:, hats].sum(1) - 1.0).abs().max()) <= 1e-10      # TL0 holds it (Lemma B)


def test_the_host_expert_is_the_batch_arm(pieces):
    from atlas.studio import cot_layouts as CL  # noqa: F401  (the host's port basis is sine-hat, 16)
    e = tt.ArmR("0.3M", "free", seed=2)
    with torch.no_grad():
        for prm in e.net.inner.head.parameters():
            prm.normal_(0.0, 0.02)
    he = tt.HostExpert(e, n_c=8)
    resp = he.respond(pieces, lambda n: pt.basis("sine-hat", n, 16))
    b = tt.PieceBatch([pieces[0]], n_c=8)
    with torch.no_grad():
        H, G, up = e(b)
    assert np.array_equal(resp[0].H, tt.to_interior(H[0])) and np.array_equal(resp[0].up, tt.to_interior(up[0]))


def test_the_weighted_loss_is_one_at_tl0_and_zero_at_the_exact_modes(batch):
    w = {"flowing": np.full(64, 1 / 64), "solid": np.full(64, 1 / 64)}
    assert float(tt.objective(batch, *batch.tl0(), weights=w, particular=0.2)) == pytest.approx(1.0, abs=1e-12)
    L = batch.labels
    assert float(tt.objective(batch, L["H"], L["G"], L["up"], weights=w, particular=0.2)) <= 1e-20


def test_terms_tl0_solves_are_left_out():
    from atlas.studio import cot, cot_layouts as CL
    spec, _ = CL.build("dev-block", "slow")
    L = cot.layout(CL.cut(spec, "dev-block", "slow"), spec)
    solid = [p for p in L.problems if not p.flows]
    b = tt.PieceBatch(solid[:2], n_c=16)
    et, ea, ep = tt.easy_terms(b)
    assert bool(et.all()) and bool(ea.all())                     # square pieces of one material
    e = tt.ArmR("0.3M", "free", seed=1)
    with torch.no_grad():
        for prm in e.net.inner.head.parameters():
            prm.normal_(0.0, 0.05)
        w = {"flowing": np.full(64, 1 / 64), "solid": np.full(64, 1 / 64)}
        assert float(tt.objective(b, *e(b), weights=w, particular=0.2)) < 1e6     # no division by TL0's zero
        assert float(tt.objective(b, *e(b))) < 1e6


def test_a_batch_holds_one_resolution(pieces):
    p64, _ = td.draw(12_000_001, 64)
    with pytest.raises(ValueError, match="one resolution"):
        tt.PieceBatch([pieces[0], p64])
