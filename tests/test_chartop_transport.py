"""CO-T, the coolant plan's Phase 1: the transport piece and its host with ADVEC seams.

`atlas/chartop/transport.py`, `transport_host.py` and `transport_cases.py`.  What the
tests pin:

- the piece: the advection matrix's loop identity C + C^T = B, the constant in the
  kernel of a steady piece's interior rows, the adjoint modes as the reversed piece's
  trial modes, the port matrix as the Schur complement, the Peclet numbers, the
  envelope's refusals;
- the host: the positive control (whole traces against `host.monolithic`, to rounding,
  steady and step, on the channel, the plate and the bend), the splits' identical
  answers, each piece's passivity under the ADVEC splits (and its absence under the
  THERM-only pairing), the G0' identity through the host, a uniform temperature
  reproduced at any port count, the balance, and the seam refusals.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from atlas.chartop import chart as ch
from atlas.chartop import fe
from atlas.chartop import ports as pt
from atlas.chartop import transport as tr
from atlas.chartop import transport_cases as tc
from atlas.chartop import transport_host as th
from atlas.chartop.host import Refused, monolithic


def _curved_chart(n):
    def F(xi, eta):
        return (xi + 0.15 * np.sin(math.pi * eta) * xi * (1 - xi) + 0.1 * eta,
                0.7 * eta + 0.1 * np.sin(math.pi * xi))
    return ch.Chart(ch.from_map(F, n), "test")


def _random_piece(n=16, seed=0, **kw):
    rng = np.random.default_rng(seed)
    return tr.make_piece(_curved_chart(n), rng.standard_normal((n + 1, n + 1)),
                         k=1.0, alpha=1.0, **kw)


# ---------------------------------------------------------------------------
# the piece
# ---------------------------------------------------------------------------


def test_loop_identity_is_exact():
    p = _random_piece(16)
    C = p.advection_matrix().toarray()
    B = p.flux_mass().toarray()
    assert np.max(np.abs(C + C.T - B)) <= 1e-13 * np.max(np.abs(C))


def test_advection_of_a_constant_stream_function_vanishes():
    n = 8
    p = tr.make_piece(_curved_chart(n), np.full((n + 1, n + 1), 3.7), k=1.0, alpha=1.0)
    assert np.max(np.abs(p.advection_matrix().toarray())) <= 1e-14
    assert not p.flows


def test_steady_interior_rows_annihilate_a_constant():
    p = _random_piece(16, seed=1)
    loop, inner = fe.grid(16)
    r = p.matrix() @ np.ones(17 * 17)
    assert np.max(np.abs(r[inner])) <= 1e-13
    # the loop rows carry the advective outflow: their sum is the loop integral of b.n
    assert abs(float(r.sum())) <= 1e-12


def test_adjoint_modes_are_the_reversed_pieces_trial_modes():
    p = _random_piece(16, seed=2)
    Q = pt.basis("sine-hat", 16, 8).Q
    r, _ = tr.exact_response(p, Q)
    rr, _ = tr.exact_response(p.reversed(), Q)
    assert np.max(np.abs(r.G - rr.H)) <= 1e-12 * np.max(np.abs(r.G))
    assert np.max(np.abs(r.G - r.H)) > 1e-3          # not symmetric


def test_port_matrix_is_the_schur_complement():
    p = _random_piece(12, seed=3)
    Q = pt.basis("sine-hat", 12, 6).Q
    r, ex = tr.exact_response(p, Q)
    K = p.matrix().toarray()
    loop, inner = fe.grid(12)
    Sch = (K[np.ix_(loop, loop)]
           - K[np.ix_(loop, inner)] @ np.linalg.solve(K[np.ix_(inner, inner)],
                                                      K[np.ix_(inner, loop)]))
    L = ex.port_matrix(Q, r.H, r.G)
    assert np.max(np.abs(L - Q.T @ Sch @ Q)) <= 1e-12 * np.max(np.abs(L))


def test_g0_prime_identity_on_a_piece():
    """W~^T K V~ - Lambda = E_W^T K_II E_V for any perturbation of both mode sets."""
    p = _random_piece(12, seed=4)
    Q = pt.basis("sine-hat", 12, 6).Q
    r, ex = tr.exact_response(p, Q)
    rng = np.random.default_rng(9)
    EV = 0.05 * rng.standard_normal(r.H.shape)
    EW = 0.05 * rng.standard_normal(r.G.shape)
    L0 = ex.port_matrix(Q, r.H, r.G)
    L1 = ex.port_matrix(Q, r.H + EV, r.G + EW)
    assert np.max(np.abs(L1 - L0 - EW.T @ (ex.K_II @ EV))) <= 1e-11 * np.max(np.abs(L0))


def test_peclet_of_a_straight_piece():
    L, w, U, alpha, n = 2.0, 0.5, 3.0, 0.01, 16
    c = ch.Chart(ch.from_map(lambda xi, eta: (L * xi, w * eta), n), "affine")
    p = tr.make_piece(c, tr.psi_nodes(c, lambda x, y: U * y, alpha), k=1.0, alpha=alpha)
    assert p.peclet() == pytest.approx(U * L / alpha, rel=1e-12)
    assert p.cell_peclet_max() == pytest.approx(U * L / alpha / n, rel=1e-12)
    assert [p.side_flow(s) for s in range(4)] == [False, True, False, True]
    assert p.side_flux(1) == pytest.approx(U * w / alpha, rel=1e-12)
    assert p.side_flux(3) == pytest.approx(-U * w / alpha, rel=1e-12)
    assert p.ports() == ("THERM", "ADVEC")


def test_envelope_refusals():
    n = 32
    c = ch.Chart(ch.from_map(lambda xi, eta: (2e-3 * xi, 1e-3 * eta), n), "affine")
    flow = tr.psi_nodes(c, lambda x, y: y, 1.0)
    ok = tr.make_piece(c, flow * (40.0 / tr.make_piece(c, flow, k=1, alpha=1).peclet()),
                       k=1.0, alpha=1.0)
    assert tr.envelope_refusals(ok) == []
    fast = tr.make_piece(c, ok.psi * 2.0, k=1.0, alpha=1.0)        # Pe 80 on n = 32
    msgs = tr.envelope_refusals(fast)
    assert len(msgs) == 1 and "cell Peclet" in msgs[0] and "n >= 40" in msgs[0]
    over = tr.make_piece(c, ok.psi * 6.0, k=1.0, alpha=1.0)        # Pe 240
    msgs = tr.envelope_refusals(over)
    assert any("above the cap of 200" in m for m in msgs)


def test_a_piece_is_one_material():
    c = _curved_chart(8)
    with pytest.raises(ValueError):
        tr.TransportPiece(c, log_kappa_hat=np.ones((9, 9)))


# ---------------------------------------------------------------------------
# the host
# ---------------------------------------------------------------------------


CASES = [("channel", dict(n=32, K=3, pe=20.0)),
         ("plate", dict(n=32, K=3, pe=20.0)),
         ("bend", dict(n=32, n_r=2, n_th=2, pe=20.0)),
         ("channel", dict(n=32, K=3, pe=50.0, problem="step")),
         ("plate", dict(n=32, K=3, pe=50.0, problem="step"))]


@pytest.mark.parametrize("name,kw", CASES)
def test_whole_traces_equal_the_monolithic_solve(name, kw):
    b = tc.build(name, **kw)
    L = b.layout(m=kw["n"])
    mono = monolithic(L)
    cp = L.solve(tr.ClassicalTransportExpert())
    top = max(float(np.max(np.abs(u))) for u in mono.fields)
    err = max(float(np.max(np.abs(u - v))) for u, v in zip(mono.fields, cp.fields))
    assert err <= 1e-10 * top
    assert th.balance(L, cp.fields, mono)["relative"] <= 1e-11
    assert th.balance(L, mono.fields, mono)["relative"] <= 1e-11


@pytest.mark.parametrize("name,kw", CASES[:3])
def test_the_split_changes_no_coupled_answer(name, kw):
    b = tc.build(name, **kw)
    out = {}
    for split in th.SPLITS:
        L = b.layout(m=8, split=split)
        out[split] = L.solve(tr.ClassicalTransportExpert())
    ref = out["galerkin"]
    Sr = ref.dense_S()
    for split in ("skew", "upwind"):
        Sd = out[split].dense_S()[np.ix_(ref.free, ref.free)] - Sr[np.ix_(ref.free, ref.free)]
        assert np.max(np.abs(Sd)) <= 1e-12 * np.max(np.abs(Sr))
        for u, v in zip(out[split].fields, ref.fields):
            assert np.max(np.abs(u - v)) <= 1e-10 * np.max(np.abs(v))


def test_passivity_needs_the_advec_split():
    b = tc.channel(n=32, K=3, pe=20.0)
    worst = {s: min(b.layout(m=8, split=s).piece_passivity(i) for i in range(3))
             for s in th.SPLITS}
    assert worst["skew"] >= -1e-12
    assert worst["upwind"] > 1e-4                   # a margin: |B|/2 kept
    assert worst["galerkin"] < -1e-2                # THERM alone: not passive


def test_the_coupled_interface_matrix_is_positive_real():
    for name, kw in CASES[:3]:
        cp = tc.build(name, **kw).layout(m=8).solve(tr.ClassicalTransportExpert())
        assert cp.sym_min_eig() > 0.0


def test_a_uniform_temperature_is_reproduced_at_any_port_count():
    """Gate K's analogue: inlet at T_IN, every wall insulated, no source, steady: the
    answer is T_IN everywhere, which the hats' sum and the exact modes carry."""
    b = tc.channel(n=16, K=3, pe=20.0)

    def bc(i, s, P):
        if np.all(np.abs(P[:, 0]) < 1e-12):
            return ("dirichlet", np.full(len(P), tc.T_IN))
        return ("insulated", None)
    for m in (2, 4, 16):
        L = th.TransportLayout(b.problems, bc, m=m)
        cp = L.solve(tr.ClassicalTransportExpert())
        for u in cp.fields:
            assert np.max(np.abs(u - tc.T_IN)) <= 1e-10 * tc.T_IN


def test_g0_prime_through_the_host_is_quadratic():
    """Perturbing every piece's trial and adjoint modes by eps moves the interface
    matrix by O(eps^2) with adjoint modes, O(eps) with the trial modes as test."""
    b = tc.channel(n=16, K=3, pe=20.0)
    L = b.layout(m=6)
    exact = tr.ClassicalTransportExpert().respond(b.problems, L.pb)
    S0 = L.couple(exact).dense_S()
    rng = np.random.default_rng(5)
    E = [(rng.standard_normal(r.H.shape), rng.standard_normal(r.G.shape)) for r in exact]
    errs = {"pg": [], "g": []}
    for eps in (1e-2, 1e-3):
        pert = [tr.TransportResponse(r.H + eps * np.linalg.norm(r.H) / np.linalg.norm(ev) * ev,
                                     r.G + eps * np.linalg.norm(r.G) / np.linalg.norm(ew) * ew,
                                     r.up) for r, (ev, ew) in zip(exact, E)]
        errs["pg"].append(np.linalg.norm(L.couple(pert).dense_S() - S0))
        gal = [tr.TransportResponse(r.H, None, r.up) for r in pert]
        errs["g"].append(np.linalg.norm(L.couple(gal).dense_S() - S0))
    slope_pg = math.log10(errs["pg"][0] / errs["pg"][1])
    slope_g = math.log10(errs["g"][0] / errs["g"][1])
    assert slope_pg > 1.9
    assert 0.8 < slope_g < 1.2


def test_seams_carry_typed_ports():
    b = tc.plate(n=16, K=3, pe=20.0)
    L = b.layout(m=4)
    kinds = sorted({s.kind for s in L.seams})
    assert kinds == ["advective", "boundary", "inlet", "outlet", "wall"]
    for s in L.seams:
        if s.kind == "advective":
            i, j = s.owner[0], s.other[0]
            assert "ADVEC" in L.problems[i].ports() and "ADVEC" in L.problems[j].ports()


def test_refused_flow_into_a_solid_piece():
    b = tc.plate(n=16, K=2, pe=20.0)
    probs = list(b.problems)
    # make the second column's coolant solid: flow would cross into a piece without ADVEC
    k = next(i for i, p in enumerate(probs) if p.label == "coolant 1")
    solid = probs[k]
    probs[k] = tr.make_piece(solid.chart, None, k=solid.kappa_ref, alpha=solid.alpha,
                             label="dried")
    with pytest.raises(Refused, match="solid and declares no ADVEC"):
        th.TransportLayout(probs, b.bc, m=4)


def test_refused_mass_not_conserved():
    b = tc.channel(n=16, K=2, pe=20.0)
    probs = list(b.problems)
    p = probs[1]
    probs[1] = tr.make_piece(p.chart, p.psi * 1.5, k=p.kappa_ref, alpha=p.alpha, label="faster")
    with pytest.raises(Refused, match="mass is not conserved"):
        th.TransportLayout(probs, b.bc, m=4)


def test_refused_inflow_without_a_temperature():
    b = tc.channel(n=16, K=2, pe=20.0)

    def bc(i, s, P):
        return ("insulated", None)
    with pytest.raises(Refused, match="an inlet needs one"):
        th.TransportLayout(b.problems, bc, m=4)


def test_refused_a_nonconforming_advective_seam():
    a = tc.channel(n=16, K=2, pe=20.0)
    b = tc.channel(n=32, K=2, pe=20.0)
    probs = [a.problems[0], b.problems[1]]
    with pytest.raises(Refused, match="must be conforming"):
        th.TransportLayout(probs, a.bc, m=4)


def test_unknown_split_and_pieces():
    b = tc.channel(n=16, K=2, pe=20.0)
    with pytest.raises(ValueError):
        b.layout(split="central")
    with pytest.raises(TypeError):
        th.TransportLayout([object()], b.bc)


def test_outlet_bulk_of_a_heated_plate_balances():
    """Steady plate: the plate's edges insulated, so the device's heat leaves with the
    coolant or diffuses back out through the inlet (the reaction there)."""
    b = tc.plate(n=32, K=3, pe=50.0)
    L = b.layout(m=32)
    mono = monolithic(L)
    bal = th.balance(L, mono.fields, mono)
    water = tc.material("water")
    q_total = sum(p.meta["q"] * p.meta["area"] for p in b.problems)
    # the heat-capacity flow through the channel, per unit depth, from psi
    coolant = [p for p in b.problems if p.label.startswith("coolant")]
    mdot_cp = coolant[0].kappa_ref * abs(coolant[0].side_flux(3))
    cp_ = L.solve(tr.ClassicalTransportExpert())
    rise = cp_.outlet_bulk() - tc.T_IN
    # carried out + diffused back through the inlet = generated
    assert mdot_cp * rise - bal["reactions"] == pytest.approx(q_total, rel=1e-6)
    assert water["k"] == 0.6
