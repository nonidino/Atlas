"""CO-T training pieces and their compact labels (`atlas/chartop/transport_data.py`): a
seed's piece is deterministic and inside the envelope, its stored float32 arrays give the
operator the labels were computed from, and the labels are the exact Schur complement
and the condensed load."""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from atlas.chartop import fe  # noqa: E402
from atlas.chartop import ports as pt  # noqa: E402
from atlas.chartop import transport as tr  # noqa: E402
from atlas.chartop import transport_data as td  # noqa: E402

SEEDS = tuple(range(td.SEED_BASE["train"], td.SEED_BASE["train"] + 12))


@pytest.fixture(scope="module")
def drawn():
    return [td.draw(s, 32) for s in SEEDS]


def test_a_seed_gives_the_same_piece(drawn):
    p, meta = td.draw(SEEDS[3], 32)
    q, m2 = drawn[3]
    assert np.array_equal(p.chart.X, q.chart.X) and np.array_equal(p.psi, q.psi)
    assert np.array_equal(p.g, q.g) and meta == m2


def test_the_resolution_mix_favours_the_large_pieces():
    ns = [td.resolution(s) for s in range(td.SEED_BASE["train"], td.SEED_BASE["train"] + 4000)]
    c = {n: ns.count(n) for n in (32, 64, 128)}
    assert sum(c.values()) == 4000
    assert c[128] >= c[32]
    p, meta = td.draw(td.SEED_BASE["train"] + 7)
    assert meta["n"] == td.resolution(td.SEED_BASE["train"] + 7)


@pytest.mark.parametrize("n", [32, 64, 128])
def test_the_peclet_cap_keeps_galerkin_from_oscillating(n):
    assert td.pe_cap(n) == min(200.0, 1.8 * n)


def test_pieces_are_inside_the_envelope(drawn):
    kinds = set()
    for p, meta in drawn:
        kinds.add(meta["kind"])
        assert tr.envelope_refusals(p) == []
        assert 0.0 <= meta["pe"] <= td.pe_cap(32) + 1e-9
        assert abs(p.peclet() - meta["pe"]) <= 1e-9 * max(1.0, meta["pe"])
        assert p.cell_peclet_max() <= td.CELL_PE_CAP + 1e-9
        if meta["kind"] == "solid":
            assert not p.flows and meta["pe"] == 0.0
    assert {"solid", "through"} <= kinds


def test_flows_have_their_declared_shape(drawn):
    rng = np.random.default_rng(0)
    chart = drawn[0][0].chart
    seen = set()
    while len(seen) < 4:
        psi, meta = td.flow_on_square(rng, 32)
        seen.add(meta["kind"])
        if meta["kind"] == "solid":
            assert not psi.any()
            continue
        p = tr.make_piece(chart, psi, k=1.0, alpha=1.0)
        flux = [p.side_flux(s) for s in range(4)]
        assert abs(sum(flux)) <= 1e-12                          # mass is conserved
        carrying = [s for s in range(4) if p.side_flow(s)]
        if meta["kind"] in ("through", "wavy"):
            assert len(carrying) == 2 and abs(carrying[0] - carrying[1]) == 2
        else:
            assert len(carrying) == 2 and abs(carrying[0] - carrying[1]) in (1, 3)


def test_stored_arrays_are_float32_and_rebuild_the_piece(drawn):
    p, meta = drawn[1]
    a = td.arrays(p, meta)
    assert a["X"].dtype == np.float32 and a["psi"].dtype == np.float32 and a["g"].dtype == np.float32
    q = td.piece_from_arrays(a)
    K, Kq = p.matrix().toarray(), q.matrix().toarray()
    assert np.max(np.abs(K - Kq)) <= 1e-5 * np.max(np.abs(K))
    assert np.array_equal(q.psi, np.asarray(a["psi"], np.float64))


def test_the_labels_are_the_schur_complement_and_the_condensed_load(drawn):
    for p, meta in drawn[:4]:
        q = td.piece_from_arrays(td.arrays(p, meta))
        lab = td.labels(q, full=True)
        n = q.n
        Q = pt.basis("sine-hat", n, 16).Q
        loop, inner = fe.grid(n)
        K = q.matrix().toarray()
        KII, KIG, KGI, KGG = K[np.ix_(inner, inner)], K[np.ix_(inner, loop)], K[np.ix_(loop, inner)], K[np.ix_(loop, loop)]
        S = KGG - KGI @ np.linalg.solve(KII, KIG)
        assert np.max(np.abs(lab["Lam"] - Q.T @ S @ Q)) <= 1e-9 * np.max(np.abs(lab["Lam"]))
        b = q.load()
        chi = Q.T @ (b[loop] - KGI @ np.linalg.solve(KII, b[inner]))
        assert np.max(np.abs(lab["chi"] - chi)) <= 1e-9 * max(1.0, np.max(np.abs(chi)))
        ex, _ = tr.exact_response(q, Q)
        assert np.max(np.abs(lab["H"] - ex.H)) <= 1e-12 and np.max(np.abs(lab["G"] - ex.G)) <= 1e-12


def test_the_estimate_reports_bytes_and_seconds():
    e = td.estimate([td.SEED_BASE["train"], td.SEED_BASE["train"] + 1])
    assert e["sample"] == 2 and e["bytes_mean"] > 0 and e["seconds_mean"] > 0
    assert sum(v["count"] for v in e["by_n"].values()) == 2
