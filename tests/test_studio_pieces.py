"""P4: pieces and experts (integrated-ui-plan §5, §8).

Done-when:
* the three held-out benchmarks, cut in the studio, reproduce `benchmarks.py`'s pieces
  (the same charts to the bit, the same conductivities);
* a coupled classical solve matches `host.monolithic` to round-off;
* a piece made to fold is refused with its det J shown.
And the experts: each trained network is offered only when its weights' sha256 matches
its record; through the host, TL-s0 on the studio's own bend-3 pieces gives the numbers
the two-level record holds for that problem.
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from atlas.chartop import benchmarks as bm  # noqa: E402
from atlas.chartop import chart as ch  # noqa: E402
from atlas.chartop import host as hs  # noqa: E402
from atlas.chartop.data import ENVELOPE  # noqa: E402
from atlas.studio import experts as X  # noqa: E402
from atlas.studio import pieces as P  # noqa: E402
from atlas.studio.session import Session  # noqa: E402
from atlas.workbench.spec import example_case  # noqa: E402

#: round-off for a coupled solve against the monolithic one, relative to the range: the
#: 16-mode truncation floor is 1e-5 to 1e-3 on these layouts
ROUND_OFF = 1e-10


def _same_pieces(cut, spec, bench):
    mine = P.problems(cut, spec)
    assert len(mine) == len(bench.problems)
    for a, b in zip(mine, bench.problems):
        assert a.chart.method == b.chart.method
        assert np.array_equal(a.chart.X, b.chart.X), a.label
        assert a.kappa_ref == b.kappa_ref


def test_bend3_cut_here_is_the_benchmarks_bend3():
    spec = example_case("bend-3")
    cut = P.auto_cut(spec, 64, n_r=3, n_th=5)
    assert cut.cutter == "polar"
    _same_pieces(cut, spec, bm.build("bend-3", 64, "steady"))
    # the coupled classical solve is the monolithic one, to round-off
    r = P.solve_classical(cut, spec)
    assert r["max_diff_rel"] < ROUND_OFF
    assert {s["verdict"] for s in r["seams"] if len(s["pieces"]) == 2} == {"admit"}


def test_insert_round_cut_here_is_the_benchmarks_insert_round():
    spec = example_case("insert-round")
    cut = P.auto_cut(spec, 32)
    assert cut.cutter == "insert"
    _same_pieces(cut, spec, bm.build("insert-round", 32, "steady"))
    full = P.solve_classical(cut, spec, m=32)              # each side's whole trace
    assert full["max_diff_rel"] < ROUND_OFF
    floor = P.solve_classical(cut, spec)                    # the experts' 16 modes: a floor
    assert floor["max_diff_rel"] > 1e3 * full["max_diff_rel"]


def test_s_channel_cut_here_is_the_benchmarks_s_channel():
    spec = example_case("s-channel")
    cut = P.auto_cut(spec, 32)
    assert cut.cutter == "strip"
    _same_pieces(cut, spec, bm.build("s-channel", 32, "steady", envelope=ENVELOPE))
    assert any("Winslow" in n for n in cut.notes)           # the end pieces' chart rule
    full = P.solve_classical(cut, spec, m=32)
    assert full["max_diff_rel"] < ROUND_OFF


def test_a_grid_cut_follows_the_materials_and_hand_cuts_stay_conforming():
    spec = example_case("wall-2")
    cut = P.auto_cut(spec, 32, cols=2, rows=1)
    assert cut.cutter == "grid"
    assert [p.material for p in cut.pieces] == ["steel", "steel", "copper"]   # cut at the interface
    c2 = P.split(cut, 0, 1, 0.4)
    assert len(c2.pieces) == 6 and P.chords(c2) == ["c1"]   # a chord runs through all three
    r = P.solve_classical(c2, spec)
    assert r["max_diff_rel"] < ROUND_OFF
    assert {s["verdict"] for s in r["seams"] if len(s["pieces"]) == 2} == {"admit"}
    c3 = P.merge(c2, "c1")
    assert len(c3.pieces) == 3 and not P.chords(c3)
    assert P.solve_classical(c3, spec)["max_diff_rel"] < ROUND_OFF


def _sector(theta: float, r0: float = 0.5, r1: float = 1.0):
    """An annular sector of angle theta as four sides: radial, outer arc, radial, inner arc."""
    def arc(r, a0, a1):
        return lambda t: np.stack([r * np.cos(a0 + (a1 - a0) * t), r * np.sin(a0 + (a1 - a0) * t)], -1)
    return [ch.line((r0, 0.0), (r1, 0.0)), ch.side_from_function(arc(r1, 0.0, theta)),
            ch.line((r1 * np.cos(theta), r1 * np.sin(theta)), (r0 * np.cos(theta), r0 * np.sin(theta))),
            ch.side_from_function(arc(r0, theta, 0.0))]


def test_a_folded_piece_is_refused_with_its_det_j():
    # a dart: its corner at (0.35, 0.35) is reflex, which no square's corner can map onto,
    # so every chart folds there; the certificate refuses it and says by how much
    pts = [(0, 0), (1, 0), (0.35, 0.35), (0, 1)]
    p = P.Piece([ch.line(pts[k], pts[(k + 1) % 4]) for k in range(4)], "steel", "dart", "tfi", None, "tfi")
    c = P.certificate(P.chart_of(p, 32))
    assert c["ok"] is False and c["min_detJ_rel"] < 0
    assert "non-positive bilinear Jacobian" in c["reason"]
    p.method = "laplacian"
    assert P.certificate(P.chart_of(p, 32))["ok"] is False


def test_dragging_an_inside_corner_past_a_neighbour_folds_it_and_is_refused():
    spec = example_case("wall-2")
    c2 = P.split(P.cut_grid(spec, 32, cols=2, rows=1), 0, 1, 0.5)
    small = P.move_corner(c2, (0.2, 0.05), (0.21, 0.06))
    assert all(P.certificate(P.chart_of(p, 32))["ok"] for p in small.pieces)
    assert P.solve_classical(small, spec, m=32)["max_diff_rel"] < ROUND_OFF   # still conforming
    far = P.move_corner(c2, (0.2, 0.05), (0.27, 0.05))
    folded = [(p.label, P.certificate(P.chart_of(p, 32))) for p in far.pieces]
    bad = [(lab, c["min_detJ_rel"]) for lab, c in folded if not c["ok"]]
    assert [lab for lab, _ in bad] == ["block 1,0a", "block 1,0b"] and all(d < 0 for _, d in bad)
    with pytest.raises(P.CutError, match="boundary"):
        P.move_corner(c2, (0.0, 0.05), (0.01, 0.05))


def test_a_domain_that_cannot_be_cut_says_why():
    spec = example_case("plate-hole")
    with pytest.raises(P.CutError, match="cannot be cut automatically"):
        P.auto_cut(spec, 32)


def test_the_conformal_chart_of_a_sector_is_the_log_polar_map():
    from atlas.studio import conformal as cf
    spec = example_case("bend-3")
    cut = P.cut_polar(spec, 32, n_r=3, n_th=5)
    p = cut.pieces[4]
    p.method = "conformal"
    c = P.chart_of(p, 32)
    assert c.method == "conformal-log-polar"                 # recognised: the closed form
    assert np.abs(c.X - p.own_chart.X).max() < 1e-12 * np.ptp(p.own_chart.X)
    # the numerical map on a sector: its module from the two harmonic measures, and the
    # duality E(v) E(w) = 1 that says how well they were solved
    sides = _sector(0.5 * np.pi)
    M = cf.measures(sides, fine=96)
    exact = np.log(2.0) / (0.5 * np.pi)                      # log(r1/r0) over the angle
    assert M["module"] == pytest.approx(exact, rel=2e-3) and M["duality_defect"] < 5e-3
    num = cf.conformal_chart([s.reparametrised(lambda t: t) for s in sides], 32)
    assert P.certificate(num)["ok"]


def test_experts_are_offered_only_when_their_weights_match_their_records(tmp_path):
    rows = X.registry()
    for r in rows:
        if r["kind"] in ("twolevel", "form") and r["offered"]:
            rec = json.load(open(os.path.join(X.CHARTOP, r["record"]), encoding="utf-8"))
            assert X.sha256_file(r["path"]) == rec["weights"][r["key"]]
    # a checkpoint whose bytes are not the recorded ones is withheld, and says why
    (tmp_path / "runs" / "twolevel" / "TL-s0").mkdir(parents=True)
    (tmp_path / "runs" / "twolevel" / "TL-s0" / "best.pt").write_bytes(b"not the weights")
    (tmp_path / "evaluation-twolevel-20261111-004223.json").write_text(json.dumps({"weights": {"TL-s0": "0" * 64}}))
    fake = {r["key"]: r for r in X.registry(str(tmp_path))}
    assert fake["TL-s0"]["offered"] is False and "sha256" in fake["TL-s0"]["why"]
    assert fake["TL-s1"]["offered"] is False and "not on this machine" in fake["TL-s1"]["why"]
    assert not any("neuber" in r["key"].lower() for r in rows)


def test_tl_s0_on_the_studios_bend3_gives_the_two_level_records_numbers():
    rows = {r["key"]: r for r in X.registry()}
    if not rows["TL-s0"]["offered"]:
        pytest.skip(f"TL-s0 is not offered here: {rows['TL-s0'].get('why')}")
    import torch
    torch.set_num_threads(8)                      # as the record's run (torch_threads 8)
    rec = json.load(open(os.path.join(X.CHARTOP, "evaluation-twolevel-20261111-004223-accuracy.json"),
                         encoding="utf-8"))
    seen = next(b for b in rec["benchmarks_seen"] if b["name"] == "bend-3" and b["problem"] == "steady")
    want = seen["frames"]["registered"]["runs"]["TL-s0"]
    spec = example_case("bend-3")
    cut = P.auto_cut(spec, 64, n_r=3, n_th=5)
    L = hs.Layout(P.problems(cut, spec), P.make_bc(spec), kind=P.KIND, m=P.M)
    mono = hs.monolithic(L)
    cp = L.solve(X.build("TL-s0"))
    num, den = cp.energy_error(mono.fields)
    mx = max(float(np.abs(a - b).max()) for a, b in zip(cp.fields, mono.fields))
    assert num / den == pytest.approx(want["energy_rel"], rel=1e-9)
    assert mx == pytest.approx(want["max_err"], rel=1e-9)
    assert cp.min_eig() == pytest.approx(want["min_eig"], rel=1e-9)


def test_the_session_cuts_charts_assigns_and_checks():
    s = Session(spec=example_case("bend-3"), key="x")
    s.pieces_cut(None, {"n_r": 3, "n_th": 5}, 32, blocking=True)
    assert s.pieces_job.status == "done", s.pieces_job.error
    summ = s.pieces_summary
    assert len(summ["pieces"]) == 15 and summ["certified"] == [15, 15]
    assert all(p["certificate"]["min_detJ_rel"] > 0 for p in summ["pieces"])
    assert len([x for x in summ["seams"] if len(x["pieces"]) == 2]) == 22
    s.pieces_method("tfi", None, blocking=True)
    assert all(p["chart"] == "transfinite" for p in s.pieces_summary["pieces"])
    s.pieces_expert("best", None, blocking=True)
    assert s.pieces_job.status == "done", s.pieces_job.error
    s.pieces_check(blocking=True)
    assert s.pieces_summary["check"]["full"]["max_diff_rel"] < ROUND_OFF
    frame = s.pieces_state()
    assert frame["type"] == "pieces" and frame["stale"] is False
    # a change to the materials makes the layout stale
    s.edit_ops([{"op": "set", "path": ["materials", "steel", "k"], "value": 40.0}], "k")
    assert s.pieces_state()["stale"] is True
