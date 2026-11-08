"""The held-out part of the forms' post-hoc diagnosis (`scripts/chartop_forms_diagnose.py
--part h1`), on small cases: bend-3's held-out and seen steady problems at n = 32, with the
zero-head form U and a throwaway-weight expert, so no trained weight is read.

What is checked: the split's new breakdown leaves the old split bit for bit and sums as it
says; a one-material problem's modes are the same on the held-out and the seen problem;
the as-trained variant is the expert's own coupled answer; with every learned column exact
the answer no longer depends on the learned weights; the variants take the columns they
name; the benchmark helper's defaults build what they built before; and the comparison
with the one evaluation's record reads the right rows."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pytest

torch = pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from atlas.chartop import data, gates_forms as GF, gates_prior as GP, host as hs  # noqa: E402
from atlas.chartop.expert import LearnedExpert  # noqa: E402

N = 32


@pytest.fixture(scope="module")
def fd():
    import chartop_forms_diagnose
    return chartop_forms_diagnose


def _noisy(fd, seed=3, std=0.02):
    """U's form with every weight nudged: heads that are not zero."""
    net = fd.zero_heads()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(std * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return net.eval()


@pytest.fixture(scope="module")
def experts(fd):
    return {"U": LearnedExpert(fd.zero_heads(), envelope=data.ENVELOPE),
            "R": LearnedExpert(_noisy(fd), envelope=data.ENVELOPE)}


@pytest.fixture(scope="module")
def h1(fd, experts):
    return fd.heldout_split(experts, N, name="bend-3", exact_pieces=[["bend 0"]])


def test_the_breakdown_leaves_the_split_and_sums(fd, experts):
    b = fd.bench("bend-3", "registered", GP.HELD_OUT_VARIANT, N)
    L = b.layout(GF.KIND, GF.M_PORTS)
    cl = L.solve(hs.ClassicalExpert())
    cp = L.solve(experts["R"])
    old, new = fd.split_term(L, cp, cl), fd.split_term(L, cp, cl, by_piece=True)
    assert set(new) - set(old) == {"piece_share", "side_share"}
    assert all(new[k] == old[k] for k in old)
    assert abs(sum(new["piece_share"]) - 1.0) < 1e-12
    assert abs(sum(new["share"].values()) - 1.0) < 1e-12
    for g in ("j1-3", "j4-8"):
        assert abs(sum(sum(d[g]) for d in new["side_share"]) - new["share"][g]) < 1e-12
        assert all(len(d[g]) == 4 for d in new["side_share"])


def test_held_out_and_seen_share_their_modes(h1, experts):
    assert h1["held_out"]["pieces"] == h1["seen"]["pieces"] == 15
    assert h1["end_piece_charts"] == []           # bend-3 records no end-piece charts
    assert h1["held_out"]["labels"] == h1["seen"]["labels"] and len(set(h1["held_out"]["labels"])) == 15
    assert h1["held_out"]["dirichlet_sides"] == h1["seen"]["dirichlet_sides"] != []
    for run in experts:
        assert h1["modes_same"][run]["bitwise"], (run, h1["modes_same"][run])
    assert h1["modes_same"]["classical"]["H_max_abs_diff"] < 1e-9
    assert h1["held_out"]["floor"] > h1["seen"]["floor"]
    for which in ("held_out", "seen"):
        for st in h1[which]["runs"].values():
            assert st["check_rel"] < 1e-9 and st["side_sum_check"] < 1e-12
            assert 0.0 <= st["ends_share"] <= 1.0 + 1e-12


def test_the_variants_couple_what_they_name(h1, experts):
    assert h1["as_trained_matches_split"] == 0.0
    V = h1["variants"]
    assert set(V) == set(experts)
    assert all(set(v) == {"as trained", "+ exact hats", "+ exact j1-3", "+ exact hats, j1-3",
                          "+ exact end sides' j1-8", "+ exact learned, end pieces", "+ exact learned",
                          "+ exact learned, on bend 0"}
               for v in V.values())
    assert h1["named_pieces"] == {"+ exact learned, on bend 0": [f"bend 0,{c}" for c in range(5)]}
    # every learned column exact on five of the fifteen pieces lies between none and all of them
    for v in V.values():
        assert v["+ exact learned"]["energy_rel"] <= v["+ exact learned, on bend 0"]["energy_rel"] \
            <= v["as trained"]["energy_rel"]
    a, b = V["U"]["+ exact learned"], V["R"]["+ exact learned"]
    assert abs(a["energy_rel"] - b["energy_rel"]) <= 1e-9 * a["energy_rel"]
    assert V["R"]["as trained"]["energy_rel"] != V["U"]["as trained"]["energy_rel"]
    for v in V.values():
        for r in v.values():
            assert set(r) == {"energy_rel", "energy_rel_lowered", "max_over_range", "e4_met"}


def test_the_variants_take_the_named_columns(fd):
    G, S = fd.column_groups(), fd.side_sines()
    v = fd.h1_variants(3, [(0, 3), (2, 1)])
    assert v["as trained"] == {}
    assert v["+ exact end sides' j1-8"] == {0: S["j1-3"][3] + S["j4-8"][3], 2: S["j1-3"][1] + S["j4-8"][1]}
    assert all(len(c) == 8 for c in v["+ exact end sides' j1-8"].values())
    assert v["+ exact learned"] == {i: G["hats"] + G["j1-3"] + G["j4-8"] for i in range(3)}
    assert all(len(c) == 36 for c in v["+ exact learned"].values())
    assert v["+ exact learned, end pieces"] == {i: G["hats"] + G["j1-3"] + G["j4-8"] for i in (0, 2)}
    assert sorted(sum(sum(S.values(), []), [])) == sorted(G["j1-3"] + G["j4-8"])
    w = fd.h1_variants(3, [(0, 3)], {"+ exact learned, on x": [1, 2]})
    assert w["+ exact learned, on x"] == {i: G["hats"] + G["j1-3"] + G["j4-8"] for i in (1, 2)}
    with pytest.raises(ValueError):
        fd.h1_variants(3, [], {"+ exact learned": [0]})


def test_a_prefix_must_name_a_piece(fd):
    labels = ["plate ring 0,0", "plate ring 3,2", "disc ring 1,0", "left column 0"]
    assert fd.pieces_named(labels, ["plate ring"]) == [0, 1]
    assert fd.pieces_named(labels, ["plate ring", "left column"]) == [0, 1, 3]
    with pytest.raises(ValueError):
        fd.pieces_named(labels, ["plate rings"])


def test_the_bench_helper_keeps_its_defaults(fd, monkeypatch):
    calls = []
    monkeypatch.setattr(fd.bm, "build", lambda name, n, problem, **kw: calls.append((name, n, problem, kw)))
    fd.bench("s-channel")
    fd.bench("bend-3")
    fd.bench("bend-3", "registered", "held-out", 32)
    assert calls == [("s-channel", GF.BENCHMARK_N, "steady", {"envelope": data.ENVELOPE}),
                     ("bend-3", GF.BENCHMARK_N, "steady", {}),
                     ("bend-3", 32, "steady", {"variant": "held-out"})]


def test_the_records_keep_their_names(fd):
    """The s-channel's h1 record keeps the path the wiki cites; another problem adds its name."""
    assert fd.record_path("h1") == fd.record_path("h1", "s-channel") == os.path.join(fd.OUT, "forms-diagnose-h1.json")
    assert fd.record_path("h1", "insert-round") == os.path.join(fd.OUT, "forms-diagnose-h1-insert-round.json")
    assert fd.record_path("a1", "insert-round") == os.path.join(fd.OUT, "forms-diagnose-a1.json")


def test_the_runs_are_read_from_their_own_folders(fd):
    assert fd.run_dir("P-s0") == fd.run_dir("PG-s1") == fd.PRIOR_RUNS
    assert fd.run_dir("Sp-s0") == fd.run_dir("S") == fd.RUNS


def test_the_record_comparison_reads_the_evaluations_rows(fd):
    import glob
    import json
    (path,) = glob.glob(os.path.join(fd.OUT, fd.EVAL_PRIOR))
    ev = json.load(open(path, encoding="utf-8"))
    held = next(r for r in ev["benchmarks"] if r["name"] == "s-channel" and r["problem"] == "steady")
    seen = next(r for r in ev["benchmarks_seen"] if r["name"] == "s-channel" and r["problem"] == "steady")
    reg, low = held["frames"]["registered"]["runs"], held["frames"]["lowered"]["runs"]
    runs = ("P-s0", "PG-s0", "Sp-s0")
    rec = {"name": "s-channel",
           "held_out": {"floor": held["frames"]["registered"]["truncation_energy_rel"]},
           "variants": {r: {"as trained": {"energy_rel": reg[r]["energy_rel"],
                                           "energy_rel_lowered": low[r]["energy_rel"]}} for r in runs},
           "seen": {"runs": {r: {"energy_rel": seen["frames"]["registered"]["runs"][r]["energy_rel"]} for r in runs}}}
    out = fd.reproduces(rec)
    assert out["max_rel"] == {"held-out, registered": 0.0, "held-out, lowered": 0.0, "seen, registered": 0.0}
    assert out["floor_rel"] == 0.0 and out["runs"] == sorted(runs)
    rec["variants"]["P-s0"]["as trained"]["energy_rel_lowered"] *= 1.01
    assert abs(fd.reproduces(rec)["max_rel"]["held-out, lowered"] - 0.01) < 1e-12
